# train_sft_lora.py
#
# LoRA SFT baseline for VSI-Bench: fine-tune the same Qwen VLM that the CVLM
# uses (train_CVLM.py) to GENERATE the answer (option letter / number), with
# the same scene split, frames, augmentation and prompts as the Base model in
# compare_base_vs_cvlm.py.
#
#   (video, question + answer instruction) -> VLM + LoRA -> "B" / "71"
#
# Only LoRA adapters on the language model are trained; the vision encoder is
# frozen. Loss = next-token cross-entropy on the answer tokens only.
#
# NOTE: VSI-Bench is an evaluation benchmark. Anything trained here on a scene
# split of it must not be reported as an untouched VSI-Bench result.
import argparse
import hashlib
import json
import os
import random

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from compare_base_vs_cvlm import BasePolicy, answer_text, chat_prompt, score_prediction
from train_CVLM import (
    DEVICE,
    MODEL_NAME,
    VIDEO_ROOT,
    VSI_FILES,
    augment_samples,
    load_frames,
    load_vsibench,
    split_by_scene,
)

# every linear layer of the language model: full attention, gated-DeltaNet
# linear attention, and the MLPs (the vision encoder is left untouched)
LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z"
    r"|in_proj_a|in_proj_b|out_proj|gate_proj|up_proj|down_proj)"
)


# -------------------------
# Frame cache (decode each (video, augmentation) once, reuse across epochs)
# -------------------------
def cached_frames(sample, args):
    key = hashlib.md5(
        f"{sample['video_path']}|{sample['aug']}|{args.num_frames}|{args.frame_size}".encode()
    ).hexdigest()
    path = os.path.join(args.results_dir, "frames", key + ".npz")
    if os.path.exists(path):
        data = np.load(path)
        return data["frames"], json.loads(str(data["metadata"]))
    frames, metadata = load_frames(
        sample["video_path"], args.num_frames, args.frame_size, aug=sample["aug"]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, frames=frames, metadata=json.dumps(metadata))
    return frames, metadata


# -------------------------
# Training example: loss only on the answer tokens
# -------------------------
def build_example(processor, sample, video):
    frames, metadata = video
    answer = answer_text(sample) + "<|im_end|>"
    batch = processor(
        text=[chat_prompt(processor, sample) + answer],
        videos=[frames],
        video_metadata=[metadata],
        return_tensors="pt",
        do_sample_frames=False,
    )
    answer_ids = processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
    n = len(answer_ids)
    if batch["input_ids"][0, -n:].tolist() != answer_ids:
        raise ValueError(f"answer tokens not at the end of the sequence for id={sample['id']}")
    labels = torch.full_like(batch["input_ids"], -100)
    labels[0, -n:] = batch["input_ids"][0, -n:]
    batch["labels"] = labels
    return batch.to(DEVICE)


# -------------------------
# Evaluation (same generation + parsing as the Base model in the compare script)
# -------------------------
@torch.no_grad()
def evaluate(model, processor, samples, args):
    policy = BasePolicy(None, args.max_new_tokens, model=model, processor=processor)
    per_type = {}
    for s in samples:
        pred, _, _ = policy.act(s, cached_frames(s, args))
        per_type.setdefault(s["question_type"], []).append(score_prediction(s, pred))
    per_type = {q: float(np.mean(v)) for q, v in sorted(per_type.items())}
    return float(np.mean(list(per_type.values()))), per_type


def plot_curves(history, path):
    """Rewritten every --log-every steps, so the step-loss panel updates mid-epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["step"], history["step_loss"], label="train (mean per log window)")
    axes[0].set_xlabel("training question (step)")
    axes[0].set_ylabel("answer-token loss")
    axes[0].set_title("Loss by step")
    axes[0].legend()
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[1].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("answer-token loss")
    axes[1].set_title("Loss by epoch")
    axes[1].legend()
    axes[2].plot(epochs, history["val_score"], marker="o", label="val")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("score (mean over question types)")
    axes[2].set_title("VSI score (acc / MRA)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_adapter(model, path, args, val_score):
    model.save_pretrained(path)
    with open(os.path.join(path, "sft_config.json"), "w") as f:
        json.dump(
            {
                "base_model_name": MODEL_NAME,
                "config": args.config,
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "max_scenes": args.max_scenes,
                "num_frames": args.num_frames,
                "frame_size": args.frame_size,
                "augment": args.augment,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "val_score": val_score,
                "eval_limit": args.eval_limit,
            },
            f,
            indent=2,
        )


# -------------------------
# Train
# -------------------------
def train(args):
    os.makedirs(args.results_dir, exist_ok=True)
    best_path = os.path.join(args.results_dir, "best")
    plot_path = os.path.join(args.results_dir, "loss_curve.png")

    samples = load_vsibench(args.config, args.video_root)
    train_samples, val_samples = split_by_scene(
        samples, args.val_ratio, args.seed, args.max_scenes
    )
    if args.augment:
        train_samples = augment_samples(train_samples, args.augment)
    # fixed subset of held-out questions for per-epoch model selection
    eval_samples = val_samples
    if args.eval_limit and args.eval_limit < len(val_samples):
        eval_samples = random.Random(0).sample(val_samples, args.eval_limit)
    print(f"Per-epoch eval:  {len(eval_samples)} held-out questions")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(DEVICE)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    steps_per_epoch = len(train_samples) // args.grad_accum
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * steps_per_epoch * args.epochs),
        num_training_steps=steps_per_epoch * args.epochs,
    )

    best_val_score = -1.0
    bad_epochs = 0
    history = {"train_loss": [], "val_score": [], "step": [], "step_loss": []}
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_samples)))
        rng.shuffle(order)
        total_loss, running = 0.0, 0.0
        optimizer.zero_grad()
        for step, i in enumerate(order, 1):
            s = train_samples[i]
            loss = model(**build_example(processor, s, cached_frames(s, args))).loss
            (loss / args.grad_accum).backward()
            total_loss += loss.item()
            running += loss.item()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % args.log_every == 0:
                print(f"  epoch {epoch+1} step {step}/{len(order)} loss={running / args.log_every:.4f}")
                history["step"].append(epoch * len(order) + step)
                history["step_loss"].append(running / args.log_every)
                plot_curves(history, plot_path)
                running = 0.0
        train_loss = total_loss / len(order)

        model.eval()
        val_score, per_type = evaluate(model, processor, eval_samples, args)
        print(
            f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_score={val_score:.4f}"
        )
        for qtype, score in per_type.items():
            print(f"    {qtype:<30} {score:.4f}")
        history["train_loss"].append(train_loss)
        history["val_score"].append(val_score)
        plot_curves(history, plot_path)
        if val_score > best_val_score:
            best_val_score = val_score
            bad_epochs = 0
            save_adapter(model, best_path, args, best_val_score)
            print(f"  -> saved best adapter ({best_path}, val_score={best_val_score:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping: val_score hasn't improved for "
                    f"{args.patience} epochs (best={best_val_score:.4f})"
                )
                break


# -------------------------
# Main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # data: keep identical to train_CVLM.py so the held-out scenes match
    parser.add_argument("--config", choices=list(VSI_FILES), default="full")
    parser.add_argument("--video-root", type=str, default=VIDEO_ROOT)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=448)
    parser.add_argument("--augment", type=int, default=0,
                        help="extra views per train question (jittered frames + shuffled options)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # LoRA / optimization
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="questions per optimizer step (each forward is one video)")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=200,
                        help="print + update loss_curve.png every N training questions")
    # evaluation
    parser.add_argument("--eval-limit", type=int, default=240,
                        help="held-out questions generated per epoch for model selection")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--results-dir", type=str, default="results/sft_lora")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)
