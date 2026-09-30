# compare_base_vs_cvlm.py
#
# VSI-Bench comparison of answerers built on the same Qwen VLM, under
# identical settings (same held-out scenes, same frames, batch size 1):
#   1. Base      - the VLM generates the answer autoregressively (letter / number)
#   2. CVLM      - frozen VLM + state/target contrastive heads (train_CVLM.py):
#                  one prefill pass, then argmax over candidate spatial states
#   3. LoRA-SFT  - (optional, --sft-checkpoint) the VLM fine-tuned with LoRA to
#                  generate the answer (train_sft_lora.py); adapter merged
#
# Evaluation only. The scene split is rebuilt from the CVLM checkpoint, so the
# questions are exactly the CVLM validation set. Latency covers preprocessing +
# model forward (+ decoding for Base); video decoding is shared and reported
# separately.
import argparse
import json
import os
import random
import re
import time
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_CVLM import (
    DEVICE,
    VIDEO_ROOT,
    ContrastiveSpatialModel,
    FrozenEncoder,
    build_vocab,
    load_frames,
    load_vsibench,
    mean_relative_accuracy,
    numeric_banks,
    split_by_scene,
)

# VSI-Bench's standard post-prompts for the generative baseline
MCQ_PROMPT = "Answer with the option's letter from the given choices directly."
NUMERIC_PROMPT = "Please answer the question using a single word or phrase."

# categorical slots 1-3 of the validated reference palette (dataviz skill)
COLORS = {"Base": "#2a78d6", "CVLM": "#eb6834", "LoRA-SFT": "#1baf7a"}
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# Generative prompt + answer parsing (Base, LoRA-SFT)
# ============================================================
def chat_prompt(processor, sample):
    prompt = sample["question"] + "\n" + (
        MCQ_PROMPT if sample["options"] else NUMERIC_PROMPT
    )
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def answer_text(sample):
    """The answer a generative model should produce: option letter or number."""
    if sample["options"]:
        return chr(ord("A") + sample["candidates"].index(sample["target"]))
    return f"{sample['value']:g}"


def parse_answer(sample, answer):
    """Canonical target (MCQ) or float (numeric) from generated text; None if unparseable."""
    if sample["options"]:
        idx = parse_letter(answer, sample["options"])
        return None if idx is None else sample["candidates"][idx]
    return parse_number(answer)


def parse_letter(text, options):
    """Option index from a generated answer, or None."""
    valid = "".join(chr(ord("A") + i) for i in range(len(options)))
    m = re.search(rf"\b([{valid}])\b", text.strip().upper())
    if m:
        return ord(m.group(1)) - ord("A")
    # fall back to an unambiguous option-text match
    hits = [i for i, o in enumerate(options) if o.lower() in text.lower()]
    return hits[0] if len(hits) == 1 else None


def parse_number(text):
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


# ============================================================
# Policies: act(sample, video) -> (prediction, latency_s, info)
# prediction = canonical target string (MCQ) or float (numeric); None = unparseable
# ============================================================
class BasePolicy:
    """Generative answerer. adapter: LoRA directory from train_sft_lora.py
    (merged into the weights, so latency matches the base model's architecture).
    model/processor: reuse already-loaded objects (in-training evaluation)."""

    def __init__(self, model_name, max_new_tokens, adapter=None, model=None, processor=None):
        self.processor = processor or AutoProcessor.from_pretrained(model_name)
        if model is None:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to(DEVICE)
            if adapter:
                model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # Qwen3.5-0.8B ships no generation_config.json, so generate() stops only at
        # <|endoftext|>. Also stop at <|im_end|> (end of the assistant turn): the
        # LoRA-SFT model is trained to end with it and never emits <|endoftext|>.
        tok = self.processor.tokenizer
        self.stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]

    @torch.no_grad()
    def act(self, sample, video):
        frames, metadata = video
        text = chat_prompt(self.processor, sample)
        sync()
        t0 = time.perf_counter()
        batch = self.processor(
            text=[text],
            videos=[frames],
            video_metadata=[metadata],
            return_tensors="pt",
            do_sample_frames=False,
        ).to(DEVICE)
        out = self.model.generate(
            **batch,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.stop_ids,
            pad_token_id=self.stop_ids[1],
        )
        sync()
        latency = time.perf_counter() - t0
        new_tokens = out[0, batch["input_ids"].shape[1] :]
        answer = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = parse_answer(sample, answer)
        return pred, latency, {"raw": answer, "new_tokens": len(new_tokens)}


class CVLMPolicy:
    def __init__(self, checkpoint, train, val):
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        self.blind = ckpt["blind"]
        self.encoder = FrozenEncoder(ckpt["base_model_name"])
        self.model = ContrastiveSpatialModel(
            self.encoder, ckpt.get("proj_hidden", 512), ckpt.get("proj_dim", 256)
        ).to(DEVICE)
        self.model.state_head.load_state_dict(ckpt["state_head"])
        self.model.target_head.load_state_dict(ckpt["target_head"])
        self.model.eval()
        # candidate spatial states are fixed -> embed once (like cached_action_z)
        vocab, self.vocab_index, _ = build_vocab(train, val)
        with torch.no_grad():
            self.z_vocab = self.model.target_head(
                self.encoder.encode_text_batched(vocab).to(DEVICE)
            )
        self.banks = {
            q: (ids.to(DEVICE), values.tolist())
            for q, (ids, values) in numeric_banks(train, self.vocab_index).items()
        }

    @torch.no_grad()
    def act(self, sample, video):
        sync()
        t0 = time.perf_counter()
        z_s = self.model.encode_state(
            None if self.blind else [video], [sample["question"]]
        )[0]
        if sample["candidates"]:
            cand_ids = [self.vocab_index[c] for c in sample["candidates"]]
            best = (self.z_vocab[cand_ids] @ z_s).argmax().item()
            pred = sample["candidates"][best]
        else:
            cand_ids, values = self.banks[sample["question_type"]]
            pred = values[(self.z_vocab[cand_ids] @ z_s).argmax().item()]
        sync()
        latency = time.perf_counter() - t0
        return pred, latency, {}


# ============================================================
# Evaluation
# ============================================================
def score_prediction(sample, pred):
    if pred is None:
        return 0.0
    if sample["options"]:
        return float(pred == sample["target"])
    return mean_relative_accuracy(pred, sample["value"])


def evaluate_policy(name, policy, samples, videos, warmup):
    print(f"\nEvaluating {name} on {len(samples)} questions...")
    for s in samples[:warmup]:
        policy.act(s, videos[s["video_path"]])
    records = []
    for i, s in enumerate(samples):
        pred, latency, info = policy.act(s, videos[s["video_path"]])
        records.append(
            {
                "id": s["id"],
                "question_type": s["question_type"],
                "target": s["target"] if s["options"] else s["value"],
                "prediction": pred,
                "score": score_prediction(s, pred),
                "latency_ms": latency * 1000,
                **info,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)}")
    return summarize(name, records)


def summarize(name, records):
    per_type = defaultdict(list)
    for r in records:
        per_type[r["question_type"]].append(r["score"])
    per_type = {q: float(np.mean(v)) for q, v in sorted(per_type.items())}
    lat = np.array([r["latency_ms"] for r in records])
    summary = {
        "name": name,
        "n": len(records),
        "overall_score": float(np.mean(list(per_type.values()))),
        "per_type_score": per_type,
        "unparsed": sum(r["prediction"] is None for r in records),
        "mean_latency_ms": float(lat.mean()),
        "median_latency_ms": float(np.median(lat)),
        "p90_latency_ms": float(np.percentile(lat, 90)),
        "throughput_qps": float(1000 / lat.mean()),
        "records": records,
    }
    if "new_tokens" in records[0]:
        summary["mean_new_tokens"] = float(np.mean([r["new_tokens"] for r in records]))
    return summary


# ============================================================
# Comparison table
# ============================================================
def print_comparison(results, decode_ms):
    names = list(results)
    cols = [results[n] for n in names]
    w = 15

    def row(label, values, fmt):
        print(f"{label:32s}" + "".join(
            f"{v:>{w}s}" if isinstance(v, str) else f"{v:{w}{fmt}}" for v in values
        ))

    print("\n" + "=" * (32 + w * len(cols)))
    print("COMPARISON")
    print("=" * (32 + w * len(cols)))
    print(f"{'Metric':32s}" + "".join(f"{n:>{w}s}" for n in names))
    row("Overall score (mean of types)", [c["overall_score"] for c in cols], ".4f")
    row("Unparsed answers", [c["unparsed"] for c in cols], "d")
    row("Latency mean (ms)", [c["mean_latency_ms"] for c in cols], ".1f")
    row("Latency median (ms)", [c["median_latency_ms"] for c in cols], ".1f")
    row("Latency p90 (ms)", [c["p90_latency_ms"] for c in cols], ".1f")
    row("Throughput (q/s)", [c["throughput_qps"] for c in cols], ".2f")
    base_lat = cols[0]["mean_latency_ms"]
    row("Speedup vs Base", ["-"] + [f"{base_lat / c['mean_latency_ms']:.2f}x" for c in cols[1:]], "")
    print(f"{'Video decode, shared (ms)':32s}{decode_ms:{w}.1f}")
    print("-" * (32 + w * len(cols)))
    print("Per-question-type score (MCQ acc / numeric MRA)")
    for qtype in cols[0]["per_type_score"]:
        row(f"  {qtype}", [c["per_type_score"][qtype] for c in cols], ".4f")


# ============================================================
# Plots
# ============================================================
def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_axisbelow(True)


def plot_comparison(results, path):
    names = list(results)
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.6, 1, 1]}
    )
    fig.patch.set_facecolor("white")

    # (a) score per question type, grouped horizontal bars
    ax = axes[0]
    types = list(results[names[0]]["per_type_score"])
    labels = ["OVERALL"] + types
    y = np.arange(len(labels))[::-1]
    h = 0.8 / len(names) - 0.02
    for k, n in enumerate(names):
        vals = [results[n]["overall_score"]] + [results[n]["per_type_score"][t] for t in types]
        offs = ((len(names) - 1) / 2 - k) * (h + 0.02)  # first model on top
        ax.barh(y + offs, vals, height=h, color=COLORS[n], label=n,
                edgecolor="white", linewidth=1)
        for yy, v in zip(y + offs, vals):
            ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=7.5, color=INK_MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("score (MCQ accuracy / numeric MRA)", color=INK_MUTED, fontsize=9)
    ax.set_title("(a) Score by question type", loc="left", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.08), ncol=len(names))
    style_axis(ax)

    # (b) per-question latency distribution (log scale)
    ax = axes[1]
    data = [[r["latency_ms"] for r in results[n]["records"]] for n in names]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "linewidth": 1.5},
                    whiskerprops={"color": INK_MUTED}, capprops={"color": INK_MUTED},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4,
                                "markeredgecolor": INK_MUTED})
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[n])
        patch.set_edgecolor("white")
    for k, n in enumerate(names):
        med = results[n]["median_latency_ms"]
        ax.text(med, k + 1.33, f"median {med:.0f} ms", ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_xlim(0, max(max(d) for d in data) * 1.1)
    ax.set_xlabel("latency per question (ms)", color=INK_MUTED, fontsize=9)
    ax.set_title("(b) Latency distribution", loc="left", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    style_axis(ax)

    # (c) accuracy-latency trade-off, one point per model
    ax = axes[2]
    for n in names:
        x, s = results[n]["mean_latency_ms"], results[n]["overall_score"]
        ax.scatter(x, s, s=120, color=COLORS[n], edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"{n}\n{s:.3f} @ {x:.0f} ms", (x, s), textcoords="offset points",
                    xytext=(10, -4), fontsize=9, color=INK)
    base = results[names[0]]
    for k, n in enumerate(names[1:]):
        r = results[n]
        speedup = base["mean_latency_ms"] / r["mean_latency_ms"]
        ax.annotate("", xy=(r["mean_latency_ms"], r["overall_score"]),
                    xytext=(base["mean_latency_ms"], base["overall_score"]),
                    arrowprops={"arrowstyle": "->", "color": INK_MUTED, "linewidth": 1})
        ax.text(0.03, 0.04 + 0.06 * k, f"{n}: {speedup:.2f}x speed, "
                f"{r['overall_score'] - base['overall_score']:+.3f} score vs Base",
                transform=ax.transAxes, fontsize=9, color=INK)
    lats = [results[n]["mean_latency_ms"] for n in names]
    scores = [results[n]["overall_score"] for n in names]
    ax.set_xlim(0, max(lats) * 1.5)
    ax.set_ylim(max(0, min(scores) - 0.1), min(1, max(scores) + 0.1))
    ax.set_xlabel("mean latency per question (ms)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("overall score", color=INK_MUTED, fontsize=9)
    ax.set_title("(c) Score vs latency", loc="left", color=INK, fontsize=11)
    ax.grid(color=GRID, linewidth=0.6)
    style_axis(ax)

    n_q = results[names[0]]["n"]
    fig.suptitle(f"{' vs '.join(names)} on VSI-Bench held-out scenes ({n_q} questions)",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/best.pt",
                        help="CVLM checkpoint from train_CVLM.py")
    parser.add_argument("--sft-checkpoint", default=None,
                        help="LoRA adapter dir from train_sft_lora.py (adds a LoRA-SFT column)")
    parser.add_argument("--base-model", default=None,
                        help="generative baseline (default: the CVLM's backbone)")
    parser.add_argument("--video-root", default=VIDEO_ROOT)
    parser.add_argument("--num-frames", type=int, default=None,
                        help="default: the checkpoint's value")
    parser.add_argument("--frame-size", type=int, default=None,
                        help="default: the checkpoint's value")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate a random subset of N held-out questions")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out-dir", default="results/compare")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Rebuild the CVLM's held-out scenes
    # --------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_frames = args.num_frames or ckpt["num_frames"]
    frame_size = args.frame_size or ckpt["frame_size"]
    base_model = args.base_model or ckpt["base_model_name"]
    samples = load_vsibench(ckpt.get("config", "full"), args.video_root)
    train, val = split_by_scene(
        samples, ckpt.get("val_ratio", 0.2), ckpt.get("seed", 42), ckpt.get("max_scenes")
    )
    if args.sft_checkpoint:
        with open(os.path.join(args.sft_checkpoint, "sft_config.json")) as f:
            sft_cfg = json.load(f)
        split = ("config", "seed", "val_ratio", "max_scenes", "num_frames", "frame_size")
        cvlm_cfg = {"config": ckpt.get("config", "full"), "seed": ckpt.get("seed", 42),
                    "val_ratio": ckpt.get("val_ratio", 0.2), "max_scenes": ckpt.get("max_scenes"),
                    "num_frames": num_frames, "frame_size": frame_size}
        mismatch = {k: (cvlm_cfg[k], sft_cfg[k]) for k in split if cvlm_cfg[k] != sft_cfg[k]}
        if mismatch:
            raise ValueError(f"CVLM vs SFT split/frame settings differ (cvlm, sft): {mismatch}")
    if args.limit:
        val = random.Random(0).sample(val, min(args.limit, len(val)))
    val.sort(key=lambda s: s["scene"])

    # --------------------------------------------------------
    # Decode each held-out video once, shared by all models
    # --------------------------------------------------------
    print(f"Decoding {len({s['video_path'] for s in val})} videos "
          f"({num_frames} frames @ {frame_size}px)...")
    videos, decode_ms = {}, []
    for s in val:
        if s["video_path"] not in videos:
            t0 = time.perf_counter()
            videos[s["video_path"]] = load_frames(s["video_path"], num_frames, frame_size)
            decode_ms.append((time.perf_counter() - t0) * 1000)

    # --------------------------------------------------------
    # Load models + evaluate
    # --------------------------------------------------------
    print("Loading CVLM...")
    cvlm = CVLMPolicy(args.checkpoint, train, val)
    print(f"Loading base model {base_model}...")
    base = BasePolicy(base_model, args.max_new_tokens)
    results = {
        "Base": evaluate_policy("Base", base, val, videos, args.warmup),
        "CVLM": evaluate_policy("CVLM", cvlm, val, videos, args.warmup),
    }
    if args.sft_checkpoint:
        del base
        torch.cuda.empty_cache()
        print(f"Loading LoRA-SFT adapter {args.sft_checkpoint}...")
        sft = BasePolicy(base_model, args.max_new_tokens, adapter=args.sft_checkpoint)
        results["LoRA-SFT"] = evaluate_policy("LoRA-SFT", sft, val, videos, args.warmup)
    print_comparison(results, float(np.mean(decode_ms)))

    # --------------------------------------------------------
    # Save JSON + plot
    # --------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "compare.json")
    out_png = os.path.join(args.out_dir, "comparison.png")
    with open(out_json, "w") as f:
        json.dump(
            {
                "config": {
                    "checkpoint": args.checkpoint,
                    "sft_checkpoint": args.sft_checkpoint,
                    "base_model": base_model,
                    "num_frames": num_frames,
                    "frame_size": frame_size,
                    "max_new_tokens": args.max_new_tokens,
                    "n_questions": len(val),
                    "mean_video_decode_ms": float(np.mean(decode_ms)),
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                },
                **results,
            },
            f,
            indent=2,
        )
    plot_comparison(results, out_png)
    print(f"\nSaved results to: {out_json}\nSaved plot to:    {out_png}")


if __name__ == "__main__":
    main()
