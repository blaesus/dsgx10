# altar1-4xgx10-switchless

A from-scratch engine for AikidoSec/altar-1 (GLM-5.3 pruned to 168 of 256 routed experts by REAP, routed experts and most attention projections at compressed-tensors INT4 group 32, the rest BF16) on the four GB10 boxes (two ASUS Ascent GX10 and two NVIDIA DGX Spark, ranks 0-3, switchless ConnectX-7 ring, TP=4). Started 2026-10-14. Builds on [glm53-4xgx10-switchless](../glm53-4xgx10-switchless/).

## Where it stands (2026-10-14, evening)

Every milestone A0–A8 has run on the four installed rank files; the numbers below are from
[engine.md](docs/engine.md) § 10, [serving.md](docs/serving.md) and [reference.md](docs/reference.md), with
their jobs in [experiments.md](docs/experiments.md).

- **Weights.** Nothing requantised: the four rank files (82,411,745,280 B each, SHA-256s in
  [layout.md](docs/layout.md)) carry the checkpoint's INT4 nibbles, scales and zero points re-tiled, and its BF16
  tensors; the verifier un-tiled every value back to the checkpoint (0 bad over 473 G INT4 values, 15 G scales and
  zero points, 27 G BF16 elements). Every CRC matches on the GPU at load.
- **Exactness.** The engine's default follows the contract (DESIGN decision 2, glm53's decision 8: the dequantisation
  compressed-tensors' at its own rounding, f32 accumulation, routed and shared experts summed in f32, one rounding at
  the residual). Against the reference's contract-order golden set it is within the reference's own two-order
  variance on all four sequences (argmax agreement 0.869 / 0.913 / 0.863 / 0.738 on short / python / prose /
  agentic-8k, 0.99–1.05× the variance's disagreement; KL 1.00–1.42×). Outputs are bit-stable: refhash short
  `975edafcbf8d617a`, python `1dd62741b1158d49`, the same in graphs twice, eager, a second process, on four ranks;
  prefill chunking, decode = prefill, batch invariance and graph = eager all pass on the real files. Open: the
  hidden state after layer 0 is not yet bit-equal to the reference (0.18 % median relative error); the rounding point
  is being hunted (A9).
- **Quality of the model itself** (level 3, altar-1 against its parent GLM-5.3 on the same text): argmax agreement
  0.627 by tokens (code 0.787, prose 0.587), KL median 0.22–0.54 nats; MTP acceptance 0.492 against the main model.
- **Speed.** Decode 45.70 ms a step at one row on real weights (1.23× the 37.05 ms byte floor; 20.9 tok/s through
  the engine, 21.1 through altarfront); prefill 973 / 1,115 tok/s at 2,600 / 8,192 tokens. Drafts (MTP, k = 1–3 and a
  cost-model gate) change no token and give 27.9 / 29.9 / 29.5 / 28.9 tok/s on one greedy session, 39–41 tok/s on a
  pair; one-draft acceptance on the model's own text 0.846.
- **Serving.** The altar-1 service runs beside the production DeepSeek service on the same ring (its own port,
  socket, lock and spill paths): lent after 120 s idle, altar-1 serving 58 s later, DeepSeek reclaiming in 52 s; 268
  real-file checks pass. The pool holds 236,544 positions while rank 2's CPU reference runs; the 256k session of
  decision 4 needs ≥ 32.6 GiB free on every rank.
- **Not yet.** The two numbers under their targets: prefill 973 tok/s at 2,600 (target ≥ 1,106) and the multi-row
  step above one row on the bench's random routing; `glw_kcheck --gpu` for the INT4 and tiled layouts; the disk tier
  and kept sessions on rank 0 (its data disk is nearly full); the NVFP4 MTP drafter; nothing is pushed.
