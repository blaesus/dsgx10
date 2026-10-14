# glm53-4xgx10-switchless

A from-scratch engine for **GLM-5.3** (zai-org's 753B-total / 40B-active `glm_moe_dsa` model: 78 layers,
256 routed experts top-8 plus one shared, DeepSeek-style sparse attention, one MTP layer) on the same four
GB10 boxes as [ds41f-4xgx10-switchless](../ds41f-4xgx10-switchless/) (two ASUS Ascent GX10 and two NVIDIA
DGX Spark, ranks 0–3, a switchless ConnectX-7 ring, TP=4).

Started 2026-10-13. The model does not fit the fleet at its released precision: 121 GiB of unified
memory per box, four boxes, and even NVIDIA's NVFP4 checkpoint is 432 GiB. So this configuration begins
with the quantisation question, and with what "exact output" means once the weights are quantised.

## Where it stands

Scouting done on 2026-10-13; the plan is in [DESIGN.md](DESIGN.md). Nothing runs yet. The findings:

- **Nothing published fits.** NVIDIA's NVFP4 checkpoint is 109.9 GiB a rank as shipped against a defensible
  weight budget of 87 GiB a rank (the KV cache is 52.7 KiB a token in FP8 and every rank holds it). The
  routed experts go to ~3.75 bits a weight in a code of our own, made from the BF16 master, and the
  non-expert weights to zai's own FP8: ~86 GiB a rank, a decode floor of ~32 ms a token before drafting.
- **The code is a 3-bit E2M1-grid code mixed with NVFP4** ([experiment 1](docs/findings.md), 2026-10-13 21:38):
  on the GB10's 48 SMs a faithful EXL3 trellis decode GEMV is bound by instruction issue at 55–75 % of the
  bandwidth floor, while NVFP4 reaches 95–96 % and the E2M1-subset code 91–93 %; only the E2M1-grid codes have the
  FP4 tensor-core path (429 TFLOPS against 107 for bf16). Trellis, whose quality per bit is better on the
  published evidence, stays open behind a tuned kernel and the KL bake-off.
- **Built so far (2026-10-14):** the front-end `glmfront` (tokenizer and template byte-exact,
  [docs/protocol.md](docs/protocol.md)); the attention side (FP8 projections at 95–103 % of bandwidth, the
  indexer, sparse MLA attention at 16 µs a layer at 256k, prefill GEMMs at 52–64 TFLOPS,
  [docs/attention.md](docs/attention.md)); the expert kernels (decode at 95–100 % of bandwidth in both formats
  at 6–16 rows, prefill at 35–37 % of the bf16 peak, [docs/experts.md](docs/experts.md)); the ring collectives
  (a step's 256 collectives in 4.40 ms, [docs/ring.md](docs/ring.md)); the per-rank weight files with every
  non-expert tensor converted and verified byte for byte (zai's FP8 rule reproduced bit for bit,
  [docs/layout.md](docs/layout.md)); the layer runner and the TP=4 decode step on all four ranks (43.6 ms at
  one row on synthetic experts, 1.27× the floor after the tuning pass: inside the graph the MoE, o_proj and
  the head sit at their bandwidth floors and the rest is the collectives' latency and attention; layers 0–3
  match the golden hidden states, [docs/engine.md](docs/engine.md)); the rank-0 server with exact prefix
  reuse, a paged store with a disk tier, batched and kept sessions ([docs/serving.md](docs/serving.md));
  the MTP layer's experts in the rank files at NVFP4; prefill at 854 / 802 / 704 tok/s at 2,600 / 8,192 / 32,768
  tokens after the GEMM wiring (2.0–2.2×, bit-identical; the ring collectives are now 38 % of a chunk);
  the service started on the ring, lending and reclaiming against the DeepSeek service; the CPU reference with golden data over 38k tokens
  ([docs/reference.md](docs/reference.md)). Every experiment is indexed in
  [docs/experiments.md](docs/experiments.md).
- **Not yet:** the routed experts in the rank files (disk: the fill is ~96 GiB a rank at NVFP4, ~80 at the
  mixture), the quantiser's 3-bit fill and the KL bake-off against the master, MTP speculation (G7, in progress),
  the end-to-end check of logits against the golden data.
| | |
|---|---|
| [DESIGN.md](DESIGN.md) | target, budgets, decisions, risks, the experiments in order, milestones |
| [docs/scouting.md](docs/scouting.md) | what is published for GLM-5.3 on GB10-class and other memory-limited hardware: the release, every quantised checkpoint, engines and their speeds, speculative decoding, quality against bits |
| [docs/quantisation.md](docs/quantisation.md) | the per-rank memory budget, what each tensor class costs at each precision, the candidate formats, how to produce and validate a requantisation, the exactness standard |
| [docs/model.md](docs/model.md) | the model's exact math as the reference code defines it, the NVFP4 and FP8 formats, the comparison with DeepSeek-V4.1-Flash, the tokenizer and chat template, the reference implementations |
| [docs/reuse.md](docs/reuse.md) | what of the DeepSeek engine carries over (~55 % by lines), what must be re-specialised, the build plan |
| `tools/` | `tally.py` (bytes by tensor class from safetensors headers), `nvfp4_entropy.py`, `provenance.py` (NVFP4 and FP8 against the BF16 master) |

## Memory per box

121 GiB unified, shared by CPU and GPU. The DeepSeek engine serves 72.4 GiB of weights per rank and
leaves 21–23 GiB available on each box.

## The checkpoint on disk

`nvidia/GLM-5.3-NVFP4` (ModelOpt, revision `e3b8f9f2`), 432.2 GiB, SHA-256 verified, on rank 0 only:
the 75 MoE layers' experts NVFP4 (337.5 GiB of E2M1 values, 42.2 GiB of E4M3 block scales at group 16: plain
round-to-nearest of the BF16 master, no weight calibration), the MTP layer's 256 experts plain BF16 (18.0 GiB),
the three dense MLPs FP8, everything else BF16 (attention 24.3 GiB, shared experts 5.3, head 1.8, embeddings
1.8). Its BF16 tensors are the master's own values ([quantisation § 3.0](docs/quantisation.md#30-what-the-checkpoints-numbers-are-measured)).
