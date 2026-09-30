# train a contrastive vision-language model (CVLM) for spatial reasoning on VSI-Bench.
#
# Adapted from the Crafter CLAM trainer (train_contrastive_crafter.py):
#   state  -> (video, question)          encoded by a frozen VLM, then state_head
#   action -> canonical spatial state    encoded by the SAME frozen VLM, then target_head
#
#                SAME frozen VLM
#                    |
#          +---------+----------+
#          |                    |
#  Video + Question       Spatial-state target
#          |                    |
#     state_head            target_head
#          |                    |
#         z_s ---------------- z_t
#                 similarity
#
# NOTE: VSI-Bench is an evaluation benchmark. Anything trained here on a scene
# split of it must not be reported as an untouched VSI-Bench result.

# train_CVLM.py
import argparse
import hashlib
import os
import random
import re
from collections import Counter, defaultdict

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, AutoModel
from torch.optim import AdamW

# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen3.5-0.8B"  # natively multimodal; was Qwen/Qwen2.5-VL-3B-Instruct
VSI_REPO = "nyu-visionx/VSI-Bench"
VSI_FILES = {
    "full": ["test_debiased.parquet", "test_pruned.parquet"],
    "debiased": ["test_debiased.parquet"],
}
VIDEO_ROOT = os.path.expanduser("~/Desktop/SPReasoning/VSI-Bench/videos")
TEMPERATURE = 0.07
PROJ_HIDDEN = 512
PROJ_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUMERIC_TYPES = [
    "object_counting",
    "object_size_estimation",
    "object_abs_distance",
    "room_size_estimation",
]
UNITS = {
    "object_size_estimation": " cm",
    "object_abs_distance": " m",
    "room_size_estimation": " m^2",
}
# VSI-Bench Mean Relative Accuracy thresholds
MRA_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


# -------------------------
# Canonical spatial states
# -------------------------
def make_spatial_target(question_type, answer):
    """Canonical spatial-state description, e.g.
    'spatial_state: object_rel_direction_hard = back-left'
    'spatial_state: object_size_estimation = 71 cm'"""
    return f"spatial_state: {question_type} = {answer}{UNITS.get(question_type, '')}"


def option_text(option):
    # "A. back-left" -> "back-left"
    return re.sub(r"^[A-Z]\.\s*", "", option).strip()


def format_question(question, options):
    if not options:
        return question
    lettered = [f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options)]
    return question + "\nOptions:\n" + "\n".join(lettered)


# -------------------------
# VSI-Bench loading + scene split
# -------------------------
def load_vsibench(config, video_root):
    rows = []
    for fname in VSI_FILES[config]:
        path = hf_hub_download(VSI_REPO, fname, repo_type="dataset")
        rows += pq.read_table(path).to_pylist()
    rows.sort(key=lambda r: r["id"])

    samples = []
    type_counts = Counter()
    n_skipped = 0
    for row in rows:
        video_path = os.path.join(video_root, row["dataset"], row["scene_name"] + ".mp4")
        if not os.path.exists(video_path):
            n_skipped += 1
            continue
        qtype = row["question_type"]
        options = [option_text(o) for o in row["options"]] if row["options"] else None
        sample = {
            "id": row["id"],
            "aug": 0,
            "scene": f"{row['dataset']}/{row['scene_name']}",
            "video_path": video_path,
            "question_type": qtype,
            "raw_question": row["question"],
            "options": options,
            "question": format_question(row["question"], options),
        }
        if options:
            # MCQ: ground_truth is a letter; the spatial state is the option text
            answer_idx = ord(row["ground_truth"].strip().upper()) - ord("A")
            sample["candidates"] = [make_spatial_target(qtype, o.lower()) for o in options]
            sample["target"] = sample["candidates"][answer_idx]
            sample["value"] = None
        else:
            sample["candidates"] = None  # filled from the train bank for this type
            sample["target"] = make_spatial_target(qtype, row["ground_truth"])
            sample["value"] = float(row["ground_truth"])
        samples.append(sample)
        type_counts[qtype] += 1
    print(f"Valid samples:   {len(samples)}  (skipped {n_skipped} with missing video)")
    print("Question-type distribution:")
    for qtype, count in type_counts.most_common():
        print(f"  {qtype:<30} {count}")
    return samples


def split_by_scene(samples, val_ratio, seed, max_scenes=None):
    """Split by scene, not by question, so no video is shared between splits."""
    scenes = sorted({s["scene"] for s in samples})
    random.Random(seed).shuffle(scenes)
    if max_scenes:
        scenes = scenes[:max_scenes]
    n_val = max(1, int(len(scenes) * val_ratio))
    val_scenes = set(scenes[:n_val])
    train_scenes = set(scenes[n_val:])
    train = [s for s in samples if s["scene"] in train_scenes]
    val = [s for s in samples if s["scene"] in val_scenes]
    print(f"Train samples:   {len(train)}  ({len(train_scenes)} scenes)")
    print(f"Val samples:     {len(val)}  ({len(val_scenes)} scenes)")
    return train, val


def augment_samples(samples, k):
    """k extra views of each training question: a jittered frame sampling of the
    same video (see load_frames) and, for MCQ, a shuffled option order.
    The canonical spatial state (target) is unchanged."""
    augmented = []
    for s in samples:
        for a in range(1, k + 1):
            view = dict(s, id=f"{s['id']}#{a}", aug=a)
            if s["options"]:
                options = list(s["options"])
                random.Random(view["id"]).shuffle(options)
                view["options"] = options
                view["candidates"] = [
                    make_spatial_target(s["question_type"], o.lower()) for o in options
                ]
                view["question"] = format_question(s["raw_question"], options)
            augmented.append(view)
    print(f"Augmented views: {len(augmented)}  ({k} per train sample)")
    return samples + augmented


def make_balanced_sampler(question_types):
    """Inverse-frequency sampling over question types so small tasks
    (e.g. route_planning) aren't drowned out by size/distance questions."""
    counts = Counter(question_types)
    weights = [1.0 / counts[q] for q in question_types]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class CachedDataset(Dataset):
    def __init__(self, embeddings, target_ids):
        self.embeddings = embeddings
        self.target_ids = torch.tensor(target_ids, dtype=torch.long)

    def __len__(self):
        return len(self.target_ids)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.target_ids[idx]


# -------------------------
# Video loading
# -------------------------
def frame_indices(n, num_frames, aug=0):
    """aug=0: uniform sampling. aug>0: one random frame per uniform segment
    (seeded by aug, so the cached embedding is reproducible)."""
    if aug == 0:
        return np.linspace(0, n - 1, num_frames).round().astype(int)
    edges = np.linspace(0, n, num_frames + 1).astype(int)
    rng = np.random.default_rng(aug)
    return rng.integers(edges[:-1], np.maximum(edges[1:], edges[:-1] + 1))


def load_frames(path, num_frames, frame_size, aug=0):
    """Sample num_frames RGB frames, resized so the long side is frame_size.
    Returns (frames [T,H,W,3], metadata) -- the metadata gives the VLM real timestamps."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, indices = [], []
    for i in frame_indices(n, num_frames, aug):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = frame_size / max(h, w)
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        indices.append(int(i))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode any frames from {path}")
    if len(frames) % 2:  # Qwen-VL groups frames in temporal pairs
        frames.append(frames[-1])
        indices.append(indices[-1])
    metadata = {
        "total_num_frames": n,
        "fps": fps,
        "duration": n / fps,
        "frames_indices": indices,
    }
    return np.stack(frames), metadata


# -------------------------
# Projection head
# -------------------------
class ProjectionHead(nn.Module):
    def __init__(self, d_model, hidden=512, out_dim=256, dropout=0.0):
        super().__init__()
        # dropout on the frozen backbone features (outside `net`, so checkpoints
        # saved without dropout still load)
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h):
        z = self.net(self.drop(h))
        return F.normalize(z, dim=-1)


# -------------------------
# Frozen VLM encoder
# -------------------------
class FrozenEncoder:
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"  # last-token pooling below
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(DEVICE)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        config = self.model.config
        self.d_model = getattr(config, "text_config", config).hidden_size

    def _last_token(self, batch):
        out = self.model(**batch)
        # last non-padding token
        last_idx = batch["attention_mask"].sum(dim=1) - 1
        h = out.last_hidden_state[
            torch.arange(len(last_idx), device=DEVICE), last_idx
        ]
        return F.normalize(h.float(), dim=-1)

    @torch.no_grad()
    def encode_video_question(self, videos, questions):
        """videos: list of (frames, metadata) from load_frames, or None for a blind
        (text-only) run."""
        content = [] if videos is None else [{"type": "video"}]
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": content + [{"type": "text", "text": q}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]
        if videos is None:
            batch = self.tokenizer(texts, padding=True, return_tensors="pt")
        else:
            batch = self.processor(
                text=texts,
                videos=[frames for frames, _ in videos],
                video_metadata=[metadata for _, metadata in videos],
                padding=True,
                return_tensors="pt",
                do_sample_frames=False,
            )
        return self._last_token(batch.to(DEVICE))

    @torch.no_grad()
    def encode_text(self, texts):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(DEVICE)
        return self._last_token(batch)

    def encode_text_batched(self, texts, batch_size=64):
        chunks = []
        for i in range(0, len(texts), batch_size):
            chunks.append(self.encode_text(texts[i : i + batch_size]).cpu())
        return torch.cat(chunks, dim=0)


def encode_states_cached(encoder, samples, args):
    """Frozen backbone => encode every (video, question) once and cache to disk,
    keyed by sample id so other splits/seeds reuse it. Samples are grouped by
    scene so each video is decoded only once."""
    key = hashlib.md5(
        f"{MODEL_NAME}|{args.num_frames}|{args.frame_size}|{args.blind}".encode()
    ).hexdigest()[:10]
    cache_path = os.path.join(args.results_dir, "cache", f"states_{key}.pt")
    cache = torch.load(cache_path) if os.path.exists(cache_path) else {}
    missing = [i for i, s in enumerate(samples) if s["id"] not in cache]
    print(f"State embeddings: {len(samples) - len(missing)} cached, {len(missing)} to encode")

    order = sorted(missing, key=lambda i: (samples[i]["scene"], samples[i]["aug"]))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    frames_cache = {}
    for step, start in enumerate(range(0, len(order), args.encode_batch_size)):
        idx = order[start : start + args.encode_batch_size]
        questions = [samples[i]["question"] for i in idx]
        videos = None
        if not args.blind:
            videos = []
            for i in idx:
                view = (samples[i]["video_path"], samples[i]["aug"])
                if view not in frames_cache:
                    frames_cache.clear()  # views are contiguous in `order`
                    frames_cache[view] = load_frames(
                        view[0], args.num_frames, args.frame_size, aug=view[1]
                    )
                videos.append(frames_cache[view])
        h = encoder.encode_video_question(videos, questions).cpu()
        for i, h_i in zip(idx, h):
            cache[samples[i]["id"]] = h_i
        if step % 50 == 0:
            print(f"  encoded {start + len(idx)}/{len(order)}")
        if step % 500 == 499:  # long augmented runs: don't lose progress
            torch.save(cache, cache_path)
    if missing:
        torch.save(cache, cache_path)
        print(f"Saved state embeddings: {cache_path}")
    return torch.stack([cache[s["id"]] for s in samples])


# -------------------------
# Model
# -------------------------
class ContrastiveSpatialModel(nn.Module):
    def __init__(self, encoder, proj_hidden=PROJ_HIDDEN, proj_dim=PROJ_DIM, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        # (video + question) -> latent spatial state
        self.state_head = ProjectionHead(
            encoder.d_model,
            proj_hidden,
            proj_dim,
            dropout,
        )
        # canonical spatial target -> same latent space
        self.target_head = ProjectionHead(
            encoder.d_model,
            proj_hidden,
            proj_dim,
            dropout,
        )

    def encode_state(self, videos, questions):
        h = self.encoder.encode_video_question(videos, questions)
        return self.state_head(h)

    def encode_target(self, targets):
        # SAME frozen VLM backbone
        h = self.encoder.encode_text(targets)
        return self.target_head(h)

    def forward(self, videos, questions, targets):
        z_s = self.encode_state(videos, questions)  # [B, D]
        z_t = self.encode_target(targets)  # [N, D]
        logits = z_s @ z_t.T / TEMPERATURE  # [B, N]
        return logits

    def forward_cached(self, h_states, h_targets):
        z_s = self.state_head(h_states)
        z_t = self.target_head(h_targets)
        logits = z_s @ z_t.T / TEMPERATURE
        return logits


# -------------------------
# Losses
# -------------------------
def multipositive_loss(logits, target_ids):
    """In-batch contrastive loss where every sample sharing the same canonical
    spatial state is a positive (a batch often holds several 'object_counting = 4')."""
    positives = (target_ids[:, None] == target_ids[None, :]).float()
    positives = positives / positives.sum(dim=1, keepdim=True)
    log_p = F.log_softmax(logits, dim=1)
    return -(positives * log_p).sum(dim=1).mean()


def inbatch_loss(model, h_states, h_targets, target_ids):
    logits = model.forward_cached(h_states, h_targets)  # [B, B]
    # positives mask is symmetric, so the same loss applies to logits.T
    s2t = multipositive_loss(logits, target_ids)
    t2s = multipositive_loss(logits.T, target_ids)
    return 0.5 * (s2t + t2s)


def bank_loss(model, h_states, h_vocab, target_ids, vocab_mask):
    """CLAM-style: score every state against the whole (deduplicated) target
    vocabulary, like the fixed 17-action set in Crafter."""
    logits = model.forward_cached(h_states, h_vocab)  # [B, V]
    logits = logits.masked_fill(~vocab_mask, float("-inf"))
    return F.cross_entropy(logits, target_ids)


# -------------------------
# Candidate sets + metrics
# -------------------------
def build_vocab(train, val):
    """Every target string that can appear as a label or candidate."""
    vocab = sorted(
        {s["target"] for s in train + val}
        | {c for s in train + val if s["candidates"] for c in s["candidates"]}
    )
    vocab_index = {t: i for i, t in enumerate(vocab)}
    # train-time bank: only states observable in the train split
    train_strings = {s["target"] for s in train} | {
        c for s in train if s["candidates"] for c in s["candidates"]
    }
    train_mask = torch.tensor([t in train_strings for t in vocab])
    return vocab, vocab_index, train_mask


def numeric_banks(train, vocab_index):
    """Numeric candidates per question type = values seen in the train split."""
    banks = defaultdict(dict)
    for s in train:
        if s["value"] is not None:
            banks[s["question_type"]][vocab_index[s["target"]]] = s["value"]
    return {
        q: (torch.tensor(list(b.keys())), torch.tensor(list(b.values())))
        for q, b in banks.items()
    }


def mean_relative_accuracy(pred, gt):
    rel_err = abs(pred - gt) / abs(gt)
    return float(np.mean(rel_err < 1 - MRA_THRESHOLDS))


@torch.no_grad()
def evaluate(model, samples, h_states, h_vocab, vocab_index, banks):
    """MCQ -> accuracy over the question's options.
    Numeric -> MRA of the best-scoring train value of the same type.
    Overall score = mean over question types (VSI-Bench convention)."""
    model.eval()
    z_s = model.state_head(h_states.to(DEVICE))
    z_vocab = model.target_head(h_vocab)
    per_type = defaultdict(list)
    predictions = []
    for i, s in enumerate(samples):
        if s["candidates"]:
            cand_ids = torch.tensor([vocab_index[c] for c in s["candidates"]])
            scores = z_s[i] @ z_vocab[cand_ids.to(DEVICE)].T
            pred = s["candidates"][scores.argmax().item()]
            score = float(pred == s["target"])
        else:
            cand_ids, cand_values = banks[s["question_type"]]
            scores = z_s[i] @ z_vocab[cand_ids.to(DEVICE)].T
            best = scores.argmax().item()
            pred = cand_values[best].item()
            score = mean_relative_accuracy(pred, s["value"])
        per_type[s["question_type"]].append(score)
        predictions.append(pred)
    per_type = {q: float(np.mean(v)) for q, v in sorted(per_type.items())}
    overall = float(np.mean(list(per_type.values())))
    return overall, per_type, predictions


# -------------------------
# Checkpoint + plotting
# -------------------------
def save_checkpoint(model, path, val_score, args):
    torch.save(
        {
            "base_model_name": MODEL_NAME,
            "temperature": TEMPERATURE,
            "proj_hidden": args.proj_hidden,
            "proj_dim": args.proj_dim,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "num_frames": args.num_frames,
            "frame_size": args.frame_size,
            "blind": args.blind,
            # needed to rebuild the same scene split at comparison time
            "config": args.config,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "max_scenes": args.max_scenes,
            "state_head": model.state_head.state_dict(),
            "target_head": model.target_head.state_dict(),
            "val_score": val_score,
        },
        path,
    )


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)
    encoder = FrozenEncoder(ckpt["base_model_name"])
    model = ContrastiveSpatialModel(
        encoder, ckpt.get("proj_hidden", PROJ_HIDDEN), ckpt.get("proj_dim", PROJ_DIM)
    ).to(DEVICE)
    model.state_head.load_state_dict(ckpt["state_head"])
    model.target_head.load_state_dict(ckpt["target_head"])
    model.eval()
    return model, ckpt


def plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_score"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score (mean over question types)")
    axes[1].set_title("VSI score (acc / MRA)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -------------------------
# Train / eval loops
# -------------------------
def run_epoch(model, loader, h_vocab, vocab_mask, loss_type, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_n = 0.0, 0
    for h_states, target_ids in loader:
        h_states = h_states.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            if loss_type == "inbatch":
                loss = inbatch_loss(model, h_states, h_vocab[target_ids], target_ids)
            else:
                loss = bank_loss(model, h_states, h_vocab, target_ids, vocab_mask)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(target_ids)
        total_n += len(target_ids)
    return total_loss / total_n


def train(args):
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    plot_path = os.path.join(args.results_dir, "loss_curve.png")

    samples = load_vsibench(args.config, args.video_root)
    train_samples, val_samples = split_by_scene(
        samples, args.val_ratio, args.seed, args.max_scenes
    )
    if args.augment:
        train_samples = augment_samples(train_samples, args.augment)
    encoder = FrozenEncoder(MODEL_NAME)
    model = ContrastiveSpatialModel(
        encoder, args.proj_hidden, args.proj_dim, args.dropout
    ).to(DEVICE)
    optimizer = AdamW(
        list(model.state_head.parameters()) + list(model.target_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        "Trainable params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # frozen backbone => cache every embedding once
    used = train_samples + val_samples
    h_all = encode_states_cached(encoder, used, args)
    train_h, val_h = h_all[: len(train_samples)], h_all[len(train_samples) :]
    vocab, vocab_index, train_mask = build_vocab(train_samples, val_samples)
    h_vocab = encoder.encode_text_batched(vocab).to(DEVICE)
    banks = numeric_banks(train_samples, vocab_index)
    print(f"Target vocab:    {len(vocab)}  ({int(train_mask.sum())} in train bank)")

    train_ids = [vocab_index[s["target"]] for s in train_samples]
    val_ids = [vocab_index[s["target"]] for s in val_samples]
    train_types = [s["question_type"] for s in train_samples]
    sampler = make_balanced_sampler(train_types) if args.weighted_sampling else None
    train_loader = DataLoader(
        CachedDataset(train_h, train_ids),
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )
    val_loader = DataLoader(
        CachedDataset(val_h, val_ids),
        batch_size=args.batch_size,
        shuffle=False,
    )
    train_mask = train_mask.to(DEVICE)
    all_mask = torch.ones_like(train_mask)

    best_val_score = -1.0
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_score": []}
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, train_loader, h_vocab, train_mask, args.loss, optimizer
        )
        val_loss = run_epoch(model, val_loader, h_vocab, all_mask, args.loss)
        val_score, per_type, _ = evaluate(
            model, val_samples, val_h, h_vocab, vocab_index, banks
        )
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_score={val_score:.4f}"
        )
        for qtype, score in per_type.items():
            print(f"    {qtype:<30} {score:.4f}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_score"].append(val_score)
        plot_curves(history, plot_path)
        if val_score > best_val_score:
            best_val_score = val_score
            bad_epochs = 0
            save_checkpoint(model, checkpoint_path, best_val_score, args)
            print(
                f"  -> saved best checkpoint ({checkpoint_path}, val_score={best_val_score:.4f})"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping: val_score hasn't improved for "
                    f"{args.patience} epochs (best={best_val_score:.4f})"
                )
                break
    return model, val_samples


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def predict(model, video, question, candidates):
    """Pick the canonical spatial state that best matches (video, question).
    video: (frames, metadata) from load_frames, or None; candidates: canonical target strings."""
    model.eval()
    z_s = model.encode_state(None if video is None else [video], [question])  # [1, D]
    z_c = model.encode_target(candidates)  # [C, D]
    scores = z_s @ z_c.T  # [1, C]
    best = scores.argmax(dim=-1).item()
    probs = F.softmax(scores / TEMPERATURE, dim=-1)[0]
    return {
        "target": candidates[best],
        "confidence": probs[best].item(),
    }


# -------------------------
# Main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(VSI_FILES), default="full")
    parser.add_argument("--video-root", type=str, default=VIDEO_ROOT)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--frame-size", type=int, default=448, help="long side of each frame (px)"
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="encode the question without the video (non-visual shortcut baseline)",
    )
    parser.add_argument(
        "--loss",
        choices=["inbatch", "bank"],
        default="inbatch",
        help="inbatch: symmetric multi-positive InfoNCE; bank: CE over all train targets",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        help="extra views per train question (jittered frames + shuffled options)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="early-stop after this many epochs without val_score improvement",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    # regularization of the two heads
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout on the frozen VLM features entering each head")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--proj-hidden", type=int, default=PROJ_HIDDEN,
                        help="hidden width of each head (smaller = fewer params)")
    parser.add_argument("--proj-dim", type=int, default=PROJ_DIM,
                        help="shared embedding size")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--max-scenes", type=int, default=None, help="use only N scenes (debugging)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--weighted-sampling",
        action="store_true",
        help="inverse-frequency sampling over question types",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    model, val_samples = train(args)
    # example MCQ prediction on a held-out scene
    sample = next(s for s in val_samples if s["candidates"])
    video = None if args.blind else load_frames(
        sample["video_path"], args.num_frames, args.frame_size
    )
    result = predict(model, video, sample["question"], sample["candidates"])
    print(sample["question"])
    print("gt:", sample["target"])
    print(result)
