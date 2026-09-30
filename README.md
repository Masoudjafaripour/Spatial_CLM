# Spatial_CLM
A repo for testing effectiveness of CLM/CVLM in spatial reasoning

**CVLM** (contrastive VLM) answers VSI-Bench video questions without generating
text. A frozen VLM (`Qwen/Qwen3.5-0.8B`) encodes both the video with its
question and a short text form of each possible answer, such as
`object_rel_direction_hard = back-left`. Two small trained heads (1.3M
parameters) map both into a shared space, and the closest answer wins.
Baselines are the same VLM generating the answer itself, zero-shot (**Base**)
or fine-tuned with LoRA (**LoRA-SFT**).

![Base vs CVLM](assets/comparison.png)

This plot comes from one run with seed 42, on the 960 questions from the 57
held-out scenes. Both models see the same 16 frames at 448px, on an RTX 3090.
The CVLM scores **0.409 vs 0.367** for Base, the mean of the per-question-type
scores (accuracy for multiple choice, MRA for numeric). Its mean latency is
about the same (**197 vs 204 ms**) because reading the ~1.5k video tokens
dominates both. Base generates only about 5 answer tokens, so skipping
generation saves little.

Details, equations and commands are in [src/README.md](src/README.md).

> VSI-Bench is an evaluation benchmark. These models are trained on a scene
> split of it, so the numbers are not untouched VSI-Bench results.
