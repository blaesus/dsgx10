# glm53-4xgx10-switchless

A from-scratch engine for **GLM-5.3** (zai-org's 753B-total / 40B-active `glm_moe_dsa` model: 78 layers,
256 routed experts top-8 plus one shared, DeepSeek-style sparse attention, one MTP layer) on the same four
GB10 boxes as [ds41f-4xgx10-switchless](../ds41f-4xgx10-switchless/) (two ASUS Ascent GX10 and two NVIDIA
DGX Spark, ranks 0–3, a switchless ConnectX-7 ring, TP=4).

Started 2026-10-13. The model does not fit the fleet at its released precision: 121 GiB of unified
memory per box, four boxes, and even NVIDIA's NVFP4 checkpoint is 432 GiB. So this configuration begins
with the quantisation question, and with what "exact output" means once the weights are quantised.

## Where it stands

Scouting. Nothing runs yet. The documents so far:

| | |
|---|---|
| [docs/scouting.md](docs/scouting.md) | what is published for GLM-5.3 on GB10-class and other memory-limited hardware: engines, recipes, checkpoints, speeds, quality |
| [docs/quantisation.md](docs/quantisation.md) | the per-rank memory budget, what each tensor class costs at each precision, the candidate formats and how to produce and validate them, and the definition of exactness for this configuration |
| [docs/model.md](docs/model.md) | the model's exact math, as the reference code defines it, and the tokenizer and chat template |
| [docs/reuse.md](docs/reuse.md) | what of the DeepSeek engine carries over, what must be re-specialised, and what is new |

## Memory per box

121 GiB unified, shared by CPU and GPU. The DeepSeek engine serves 72.4 GiB of weights per rank and
leaves 21–23 GiB available on each box.

## The checkpoint on disk

`nvidia/GLM-5.3-NVFP4` (ModelOpt, revision `e3b8f9f2`), 432.2 GiB, SHA-256 verified, on rank 0 only:
experts NVFP4 (355.5 GiB of E2M1 values, 42.2 GiB of E4M3 block scales at group 16), the three dense
MLPs FP8, everything else BF16 (attention 24.3 GiB, shared experts 5.3, head 1.8, embeddings 1.8).
