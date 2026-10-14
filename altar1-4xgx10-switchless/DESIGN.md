# altar1-4xgx10-switchless design

The plan for an engine for `AikidoSec/altar-1` on four GB10 boxes in a ring. The evidence behind every number is in
[docs/scouting.md](docs/scouting.md) (what is published), [docs/quantisation.md](docs/quantisation.md) (the per-rank
budget from the shard headers), [docs/model.md](docs/model.md) (the model's exact math as a delta from GLM-5.3's, with
the INT4 format's semantics verified from the compressed-tensors source) and [docs/reuse.md](docs/reuse.md) (every
change point in the GLM-5.3 engine this one starts from). Written 2026-10-14; every decision below is revised as the
experiments land, and the revision is recorded here.

## Target

- **Model:** `AikidoSec/altar-1`, revision `5d591cd98958e4e7e429517887e80ca3bb7ce2c4` (2026-09-20): GLM-5.3 with
  168 of its 256 routed experts kept in every MoE layer (REAP, arXiv:2510.13999, experts renumbered 0..167 by
  `reap_prune_plan.json`), ~501B parameters, the same 78 layers + MTP layer, MLA attention, DeepSeek sparse attention
  and tokenizer as GLM-5.3 ([glm53 DESIGN](../glm53-4xgx10-switchless/DESIGN.md#target)). Precision as shipped: the
  routed experts and the `q_b`, `kv_b`, `o_proj` projections of layers 4–76 in compressed-tensors INT4 (`pack-quantized`,
  group 32 along K, asymmetric, BF16 scale and 4-bit zero point a group), everything else BF16, including the 168
  experts of layers 3, 77 and the MTP layer 78. 305.42 GiB in 39 shards.
- **Fleet:** the four boxes of [ds41f-4xgx10-switchless](../ds41f-4xgx10-switchless/DESIGN.md#target): two ASUS
  Ascent GX10 (ranks 0 and 1) and two NVIDIA DGX Spark (ranks 2 and 3), 121 GiB of unified memory each, ~235 GB/s,
  a switchless ConnectX-7 ring, TP=4. The DeepSeek engine keeps serving on it; GPU work goes through its `ring.sh`,
  short and preemptible.
- **Workload and standard:** as glm53's: agentic sessions, single-stream decode latency, prefill throughput, exact
  prefix reuse; the DeepSeek configuration's profiling discipline (every experiment indexed with its failures, A/B in
  both orders, end-to-end bars) and exact output in the sense of [decision 2](#decisions).

## The problem

This model fits: **76.6 GiB a rank** at TP=4 ([quantisation](docs/quantisation.md)), under glm53's 87 GiB budget,
with room for one 256k session in a BF16 cache (23.6 GiB) or a 12 GiB pool with the DeepSeek service's headroom. So
there is no format to choose and no quantiser to write: the served weights are the published bytes. What the
configuration has to get right instead:

1. **A decode GEMV for the INT4 format at the bandwidth floor** on 48 SMs: 0.578 B a weight (values, BF16 scale,
   4-bit zero point a group of 32), asymmetric, which glm53's NVFP4 kernels (95–96 % of 235 GB/s) do not read.
2. **The non-expert weights are BF16 and stay BF16** (they are the model), so a decode token reads 8.55 GB a rank,
   **36.4 ms at the floor** against glm53's 34.2 (its non-experts were FP8). Every exact byte-saving lever (lossless
   BF16 coding, the fused shared expert) is measured, none assumed.
3. **The BF16 tensors are not GLM-5.3's** ([model.md](docs/model.md), 2026-10-14): they are dequantised from zai's FP8
   release with AWQ's per-channel smoothing folded into the norms, the router and the expert input columns, so nothing
   of glm53's rank files carries over, and level 3 compares a different model with its parent.
4. **Exactness for a model that is already quantised**: the engine must compute the dequantised weight exactly as the
   model's own library does (the weight is one round-to-nearest-even of `(q − z) · s`, the nibbles stored offset by 8;
   43 % of the products are inexact in BF16, so the rounding point is the contract, verified exhaustively,
   [model.md](docs/model.md)), and its quality is measured against the parent, GLM-5.3, whose golden data
   this fleet already has.

## Budgets that drive the design

| | |
|---|---|
| Weights a rank | **76.6 GiB** (INT4 experts 62.3, BF16 experts of layers 3/77/78 8.9, attention 2.4, shared experts 1.3, the rest 1.7); budget 87, ceiling 94 |
| KV a position | 96,640 B a rank in BF16 (glm53's store: 95,232 B of latent and indexer keys + page overhead), every rank holds all of it: a 256k session 23.6 GiB; FP8 cache 52.7 KiB a position is a later, output-changing option |
| Bytes a decode token a rank | INT4 experts 3.19 GB, BF16 experts of layers 3 and 77 0.30, attention projections 2.57 (o_proj 1.06 INT4 + 0.37 BF16, q_b 0.35, kv_b 0.16, q_a 0.49, kv_a 0.14), shared experts 1.42, dense MLPs 0.34, head 0.48, routers 0.16, indexer 0.10: **8.55 GB → 36.4 ms at 235 GB/s** (the runner counts 8.708 GB → 37.05 ms at 1 row, with W_UK^T BF16 on the INT4 layers: [engine.md § 3](docs/engine.md#3-the-decode-step)); a draft pass 0.77 GB |
| Expected step | glm53's measured overhead on top (collectives ~5.5 ms, attention ~4 ms): **~45.8 ms, ~22 tok/s** before drafting (*estimate*) |
| Prefill | as glm53: all 78 layers over every token; BF16 tensor cores 107 TFLOPS; INT4 prefill through dequantised BF16 tiles |
| Collectives | glm53's 256 a step, 4.40 ms at 1 row, unchanged |
| Published bar | none for altar-1 on GB10-class hardware ([scouting](docs/scouting.md)); the card's own target is 4× H200 in vLLM; glm53's bar for GLM-5.3 on four GB10s: 49.3 / 30.0 tok/s code / prose, prefill 1,374 |
| Disk | checkpoint 327.9 GB on rank 3 (converted there, peak ~492 GB of 694); rank files ~82.3 GB each; rank 0's data disk holds its file with ~35 GB left |

## Decisions

1. **No requantisation: the served weights are the checkpoint's bytes.** Every INT4 value, scale and zero point and
   every BF16 tensor goes into the rank files unchanged, re-tiled for the kernels, and the verifier un-tiles every
   byte back to the checkpoint's. Level 1 of the exactness standard is therefore a manifest of the Hub's own LFS
   SHA-256s plus the rank files' hashes. The one exception is open: the MTP layer's experts (2.95 GiB a rank in BF16)
   may be requantised to NVFP4 by glm53's reproduced recipe, because the drafter's precision changes speed and not
   output ([decision 5](#decisions)); that is a measured lever after A4, not the default.
2. **Exactness, in three levels**, restated from glm53's decision 3 for a model with no master of its own:
   (1) the artefact as in decision 1;
   (2) the numerics contract: the weight of an INT4 tensor is compressed-tensors' dequantisation at its own rounding
   point, as [model.md](docs/model.md) quotes it from the source (verified on real tensors against the library in A0),
   BF16 activations, f32 accumulation, the KV format, the indexer key precision, top-k tie-breaking at 168, and
   glm53's reduction order (its decisions 8 and 11: f32 sums over a row's experts and ranks rounded to bf16 once;
   decode's K order differs from prefill's by default); the engine keeps pinned refhash outputs across every change
   and agrees with the CPU reference within 2× the reference's own order variance on all four golden sets;
   (3) quality against the parent: argmax agreement and KL against GLM-5.3's golden logits (`all` and `all-f32`,
   made by the glm53 reference on this fleet) on the same four sequences, the reference's two orders giving the
   noise floor; the published fidelity study's 0.506 nats against BF16 GLM-5.3 on its own 25-prompt panel is the
   sanity anchor, not a bar, since the panels differ. The MTP layer sits outside level 3.
3. **BF16 non-expert weights as shipped.** glm53 served FP8 because zai released FP8; altar-1 releases BF16, and an
   FP8 conversion would change outputs. The byte cost (~5.9 GB a token a rank) is paid; lossless BF16 coding
   (`kBF16X` of the DeepSeek engine, bit-identical) is the one output-preserving lever and is measured in A8.
4. **KV cache BF16, replicated, one 256k session in the pool.** The budget allows it; the FP8 cache stays an opt-in
   track with its KL measured, as in glm53.
5. **The MTP layer is the drafter**, as shipped (BF16 experts) first, since that is exact by construction; chained
   drafts, the gate and the spec graphs carry from glm53's G7. Its experts at NVFP4 are the first lever (−2.1 GiB a
   rank, a faster draft pass) once acceptance is measured against the BF16 drafter.
6. **Formats in the kernels: INT4-g32 asymmetric and a BF16 expert family, beside glm53's NVFP4 for the drafter
   option.** The INT4 decode kernel reproduces the contract's weight per element (a subtract and one BF16 multiply)
   inside glm53's expert and GEMV skeletons; prefill runs the INT4 tensors through dequantised BF16 tiles and
   glm53's `gemm_bf16` (bit-equal to the GEMV by construction), a fused INT4 tensor-core path only if it is bit-equal
   and faster. The BF16 experts of layers 3, 77 and 78 get their own decode kernel (per-layer templates) and the
   plain BF16 prefill GEMM.
7. **A copy of the GLM-5.3 engine, specialised**: 168 experts and every per-rank shape a compile-time constant, the
   format a per-layer property of the binding, the format dispatch failing on an unknown format rather than guessing
   ([reuse § 10](docs/reuse.md)). The ring, attention, indexer, step machinery, server, store, service scripts and
   the front-end port carry over; the quantiser, the S3 code paths, provenance tools and the master fetch are dropped.
8. **Batch invariance kept** (glm53 decision 9).
9. **Production keeps serving DeepSeek**; every GPU job is a short preemptible ring job; the reference and the
   converter run on rank 3's CPU, where the checkpoint is, under the fleet's memory rule (a CPU pass holds ≤ 3 GB
   RSS while real-weight ring jobs are possible, else earlyoom kills it; glm53 serving § 5.1).
10. **Routing at 168 is the config's routing**: sigmoid scores plus the 168-entry `e_score_correction_bias`, top-8,
    normalised, scaled by 2.5, one group; the card's "routing is untouched" means the kept experts' router rows and
    biases are the parent's, which the reference checks against the prune plan and the parent's tensors on two layers.

## Risks and open questions

- The asymmetric INT4 GEMV needs one more operand stream (zero points) and a subtract per weight on 48 SMs; NVFP4's
  kernel reached 95–96 % with a lighter dequantisation. Target ≥ 93 % at 1–16 rows; below 90 % the format's own
  bytes are the floor and the step estimate moves by ~1 ms a point.
- The model's MoE layers 3 and 77 and its MTP layer are BF16: a decode step reads 0.30 GB a rank of them, and the
  draft pass 0.77 GB; the BF16 expert kernel has no precedent in either engine.
- The dequantisation's rounding point must be the library's, verified on real tensors, before any kernel is trusted:
  a reference holding f32 weights could never be matched by BF16 MMAs.
- Quality: a 34 % expert prune is a different model from GLM-5.3; level 3 reports the distance, it does not bound
  it. KL on this family has a high floor (glm53's two orders: 0.05–0.41 by text type), so only paired comparisons
  on the same harness count.
- Rank 0's data disk: 35 GB left once its rank file lands, so no disk tier for sessions there until the user frees
  space (the 257 GiB partial checkpoint copy or glm53's 101 GiB file are theirs to reclaim).
- The step estimate (~45.8 ms) assumes glm53's overheads; the BF16 shared expert, now outside the FP8-fused routed
  kernel, may add to it until it is fused again.

## Experiments, in order

| # | What | Cost | Decides |
|---|---|---|---|
| A0 | The INT4 unpacking and dequantisation checked against compressed-tensors on ≥ 3 real tensors; INT4-g32 and BF16-expert decode GEMV prototypes in `micro/` at 1, 6, 16 rows on expert and o_proj shapes | CPU; one ≤ 10-min ring window | the contract's weight expression (decision 2); whether the floor is reachable (risk 1) |
| A1 | The CPU reference on altar-1 (168 experts, dequantised INT4, BF16 the rest) on rank 3: two short layer checks against transformers, then the four golden sequences in two orders, the MTP teacher-forced | hours of CPU at nice 19 | level-2 golden data, the order variance, level-3 numbers against GLM-5.3's golden |
| A2 | The conversion on rank 3 with a byte-for-byte verification against the shards and a second conversion of two layers | ~2 h of CPU, 492 GB peak | the artefact |
| A3 | Kernel bake-offs: INT4 decode and prefill variants, the BF16 expert kernel, the BF16 shared expert through the GEMV and fused | ring windows | the kernels |
| A4 | Exact byte-saving levers after the step runs: `kBF16X` on the BF16 reads, the NVFP4 drafter, the fused shared expert; A/B in both orders | ring windows | decisions 3 and 5's levers |

## Milestones

From [reuse § 9](docs/reuse.md#9-build-plan); effort relative to glm53's own milestones.

| # | Milestone | Done when |
|---|---|---|
| A0 | Facts and microbenchmarks | the dequantisation equals compressed-tensors' on 100 % of values; INT4 and BF16-expert GEMVs ≥ 93 % of 235 GB/s |
| A1a | Reference and first golden (short and ≥ 4k sequences, per-layer dumps) | layers 0, 3, 4, 77 agree with transformers + compressed-tensors to BF16 noise; greedy text reads right |
| A1b | Full golden: the four sequences, two orders, MTP acceptance teacher-forced | argmax agreement with the independent stack within noise; level-3 numbers against GLM-5.3 |
| A2 | Converter, verifier, loader: the four ~76.6 GiB rank files cut on rank 3 and installed | every byte un-tiled equals the checkpoint's; cross-rank CRCs; `glw_kcheck` |
| A3e / A3a | Expert kernels / attention-side kernels on one rank | INT4 kernels bit-equal to the BF16 kernels on the dequantised matrices; routing at 168 equals torch's; every layer type matches golden at BF16 level; GEMVs ≥ 93–95 % |
| A4 | TP=4 decode on the real files | layers match golden within the reference's envelope; bit-stable across graphs, eager and processes; refhash pinned; ≤ 46 ms at 1 row (1.27× the floor), ≤ 1.05× at 6 |
| A5 | Prefill | chunking changes no bit; ≥ glm53's 1,106 / 1,145 tok/s at 2,600 / 8,192 |
| A6 | Serving and the front-end (altarfront: done 2026-10-14, byte-exact on 3,107 cases) | serve tests pass; one 256k session in the pool |
| A7 | MTP drafts | tokens identical with and without drafts; acceptance and tok/s on real text |
| A8 | End to end and ship: service, refhashes, benches, the exact levers | within 2× the reference's order variance on all four sets; plain decode through the front-end ≥ ~21 tok/s |

Critical path: the download → A2 → A4; A0 → A3e → A4. A1 beside A2–A3 on rank 3's CPU; A5 and A7 after A4; A8 last.

Status on 2026-10-14: scouting and the budget done (S1, S2 in [experiments.md](docs/experiments.md)); the front-end
done; the checkpoint arriving on rank 3; A0–A3 starting.
