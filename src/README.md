# Contrastive VLM for spatial reasoning (`train_CVLM.py`)

This is an adaptation of the Crafter CLAM trainer to VSI-Bench. In CLAM, a
game **state** is matched to an **action**. Here, a **(video, question)** pair
is matched to a **canonical spatial state**, which is a short text such as
`spatial_state: object_rel_direction_hard = back-left`.

## Model

A frozen VLM, `Qwen/Qwen3.5-0.8B`, encodes both sides. Each input is summarized
by the last-token hidden state $E(\cdot)$ of the model's final layer. Only two
small MLP projection heads are trained:

$$
z_s = \mathrm{norm}\big(H_s(E(V, Q))\big), \qquad
z_y = \mathrm{norm}\big(H_y(E(S_y))\big), \qquad
\ell_{ij} = \frac{z_{s,i}^\top z_{y,j}}{\tau}, \quad \tau = 0.07
$$

Here $H_s$ is the state head, $H_y$ is the target head, and $\mathrm{norm}$
scales a vector to unit length. The video goes into $E$ together with the
question. The target text $S_y$ goes through the same frozen model as text only.

## Canonical targets

| Question kind | Target text $S_y$ |
|---|---|
| Multiple choice (the ground truth is a letter) | The text of the correct option, e.g. `... object_rel_direction_hard = back-left` |
| Numeric | The value with its unit: `object_size_estimation = 71 cm`, `object_abs_distance = 0.9 m`, `room_size_estimation = 26.4 m^2`, `object_counting = 4` |

## Losses

`--loss inbatch` (default) is a symmetric multi-positive InfoNCE. Samples in a
batch that share the same target string all count as positives. Let
$P_{ij} = \mathbb{1}[y_i = y_j] / \sum_k \mathbb{1}[y_i = y_k]$ and
$p = \mathrm{softmax}_{\text{row}}$:

$$
\mathcal{L} = \tfrac12\Big[-\tfrac1B\sum_{i,j} P_{ij}\log p(\ell)_{ij}
\;-\; \tfrac1B\sum_{i,j} P_{ij}\log p(\ell^\top)_{ij}\Big]
$$

`--loss bank` works like CLAM's fixed action set. It is a cross-entropy over
every distinct target text $\mathcal{V}$ in the training split:

$$
\mathcal{L} = -\frac1B\sum_i \log \frac{\exp \ell(i, y_i)}{\sum_{v\in\mathcal{V}} \exp \ell(i, v)}
$$

## Evaluation

The data is split **by scene**, so no video appears in both train and
validation.

- **Multiple choice:** the model scores the question's own options and picks
  the best one. The metric is accuracy.
- **Numeric:** the model scores every value of the same question type seen in
  the training split and picks the best one. The metric is VSI-Bench's Mean
  Relative Accuracy, with $\theta \in \{0.50, 0.55, \dots, 0.95\}$:

$$
\mathrm{MRA} = \frac{1}{10}\sum_{\theta} \mathbb{1}\left[\frac{|\hat y - y|}{y} < 1 - \theta\right]
$$

- **Validation score:** the mean of the per-question-type scores. This score
  drives early stopping and checkpoint selection.

## Caching and augmentation

- **Caching:** because the backbone is frozen, each (video, question) is
  encoded once. The embedding is stored per sample in
  `<results-dir>/cache/states_*.pt`, so later runs that use the same model and
  frame settings skip encoding.
- **Augmentation:** `--augment K` adds K extra copies of every training
  question. Each copy uses a new random frame sampling (one frame per uniform
  segment of the video). Multiple-choice copies also get shuffled options. The
  target stays the same, and validation is never augmented.

## Commands

Run these from the repo root inside the uv environment:

```bash
uv venv spcvlm_venv --python 3.11
VIRTUAL_ENV=spcvlm_venv uv pip install -r requirements.txt
```

```bash
# default run
spcvlm_venv/bin/python src/train_CVLM.py

# full run with 4 augmented copies per training question
spcvlm_venv/bin/python src/train_CVLM.py \
  --config full --num-frames 16 --frame-size 448 \
  --augment 4 --loss inbatch --weighted-sampling \
  --epochs 50 --patience 8 --batch-size 64 --encode-batch-size 4 \
  --lr 1e-4 --val-ratio 0.2 --seed 42 --results-dir results/aug4_inbatch

# question-only baseline (no video): always compare against this
spcvlm_venv/bin/python src/train_CVLM.py --blind --augment 4 --weighted-sampling \
  --results-dir results/aug4_blind

# quick debug run on 12 scenes
spcvlm_venv/bin/python src/train_CVLM.py --max-scenes 12 --epochs 2
```

| Arg | Default | Meaning |
|---|---|---|
| `--config` | `full` | `full` has all 5,130 questions; `debiased` is the 2,362-question subset with fewer answer-without-looking shortcuts |
| `--video-root` | `~/Desktop/SPReasoning/VSI-Bench/videos` | Folder containing `{dataset}/{scene}.mp4` |
| `--num-frames` / `--frame-size` | 16 / 448 | Frames sampled per video / long side of each frame in pixels |
| `--augment` | 0 | Extra augmented copies per training question |
| `--blind` | off | Encode the question without the video |
| `--loss` | `inbatch` | `inbatch` or `bank` (see Losses) |
| `--weighted-sampling` | off | Balance training batches across question types |
| `--epochs` / `--patience` | 50 / 8 | Maximum epochs / early-stopping patience |
| `--batch-size` / `--encode-batch-size` | 64 / 4 | Training batch size / videos per frozen-VLM forward pass |
| `--lr` / `--val-ratio` / `--seed` | 1e-4 / 0.2 / 42 | Learning rate / fraction of scenes held out / random seed |
| `--dropout` | 0.0 | Dropout on the frozen VLM features entering each head |
| `--weight-decay` | 0.01 | AdamW weight decay on the heads |
| `--proj-hidden` / `--proj-dim` | 512 / 256 | Hidden width of each head / size of the shared embedding |
| `--max-scenes` | all | Use only N scenes, for debugging |
| `--results-dir` | `results` | Where the checkpoint (`checkpoints/best.pt`), `loss_curve.png` and the embedding cache go |

Each run reads the embedding cache in its own `--results-dir`, so a new
directory re-encodes everything. To reuse encodings, copy `cache/` into the new
directory first.

## LoRA SFT baseline (`train_sft_lora.py`)

This baseline fine-tunes the same VLM to **generate** the answer: an option
letter for multiple choice, or a number for numeric questions. It uses LoRA
adapters on every linear layer of the language model, with $r=16$ and
$\alpha=32$. The vision encoder stays frozen. The loss is cross-entropy on the
answer tokens only, with $a$ the answer and $x$ the video plus the prompt:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{answer}} \log p_\theta(a_t \mid V, Q, a_{<t})
$$

The scene split, frames, augmentation and prompts are the same as for the CVLM
and the base model. After each epoch, the model generates answers for
`--eval-limit` held-out questions and the best epoch is kept. The adapter is
saved to `results/sft_lora/best/` together with `sft_config.json`. Decoded frames
are cached in `results/sft_lora/frames/`.

```bash
spcvlm_venv/bin/python src/train_sft_lora.py --augment 4 --epochs 3 --results-dir results/sft_lora
```

Use the same `--seed`, `--val-ratio`, `--num-frames`, `--frame-size` and
`--config` as the CVLM run; the compare script refuses mismatched splits.

## Base VLM vs CVLM vs LoRA-SFT (`compare_base_vs_cvlm.py`)

This script compares the base VLM with the CVLM on the CVLM's held-out
questions. It rebuilds the validation split from the checkpoint, and both models
see the same frames with batch size 1.

- **Base:** the VLM generates the answer (a letter or a number) using
  VSI-Bench's standard answer instructions.
- **LoRA-SFT** (optional): the same generation with the LoRA adapter merged
  into the weights.
- **CVLM:** one forward pass over the input, then the best-matching candidate
  target.

Latency covers preprocessing and the model forward pass, plus decoding for
Base. Video decoding is shared by both models and reported separately. The
script prints a comparison table and saves `compare.json` (with per-question
predictions) and `comparison.png` (score per question type, latency
distribution, and score vs latency).

```bash
spcvlm_venv/bin/python src/compare_base_vs_cvlm.py \
  --checkpoint results/checkpoints/best.pt \
  --sft-checkpoint results/sft_lora/best \
  --out-dir results/compare
# omit --sft-checkpoint to compare only Base and CVLM
# quick check: --limit 60
```

## Reference result

This is one run with seed 42, no augmentation, and 960 validation questions
from 57 scenes.

| Input | Val score |
|---|---|
| Video + question | 0.403 |
| Question only | 0.322 |

## Note

VSI-Bench is an evaluation benchmark. Models trained on a scene split of it
must not be reported as untouched VSI-Bench results.
