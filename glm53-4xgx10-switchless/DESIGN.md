# glm53-4xgx10-switchless design

The plan for a GLM-5.3 engine on four GB10 boxes in a ring. The evidence behind every number is in
[docs/scouting.md](docs/scouting.md) (what others published), [docs/quantisation.md](docs/quantisation.md)
(the budget, the formats, the exactness standard), [docs/model.md](docs/model.md) (the model's exact math)
and [docs/reuse.md](docs/reuse.md) (what the DeepSeek engine gives us). Written 2026-10-13 from the scouting;
every decision below is revised as the experiments land, and the revision is recorded here.

## Target

- **Model:** `zai-org/GLM-5.3` (`glm_moe_dsa`): 753B parameters, ~40B active a token; 78 layers (3 dense,
  75 MoE of 256 routed experts top-8 plus one shared), MLA attention (q latent 2048, kv latent 512, 64 heads
  of 192 + 64 rope by 256), DeepSeek sparse attention with a top-2048 indexer on 21 layers shared by the next
  three, one MTP layer, vocab 154,880, 1M positions. Released in block FP8 (703.8 GiB) with a BF16 master
  (`zai-org/GLM-5.3-BF16`, revision `9d2398f4`, 1,403 GiB). GLM-5.3 is GLM-5.2's base with new post-training.
- **Fleet:** the same four boxes as [ds41f-4xgx10-switchless](../ds41f-4xgx10-switchless/DESIGN.md#target):
  two ASUS Ascent GX10 (ranks 0 and 1) and two NVIDIA DGX Spark (ranks 2 and 3), 121 GiB of unified memory
  each, sm_121a, ~235 GB/s measured streaming read, a switchless ConnectX-7 ring, TP=4. The DeepSeek engine
  keeps serving on it while this one is built; GPU work goes through its `ring.sh`, short and preemptible.
- **Workload:** agentic, as for DeepSeek: one to a few sessions, long and growing contexts, short new
  suffixes per turn, tool calls, thinking on. Single-stream decode latency, prefill throughput, exact
  prefix reuse.
- **Standard:** the DeepSeek configuration's profiling discipline (every experiment indexed, failures
  included; A/B in both orders; end-to-end bars) and an exactness standard restated for a quantised model
  ([decision 3](#decisions)).

## The problem

The model does not fit at any released precision. NVIDIA's NVFP4 checkpoint (432 GiB; experts at 4.5 bits a
weight with their E4M3 scales, attention and head BF16, the MTP layer's experts plain BF16) is 109.9 GiB a
rank as shipped and 100.0 GiB in its leanest arrangement, against a defensible **weight budget of 87 GiB a
rank** (hard ceiling 94): the KV cache is 52.7 KiB a token in FP8, 93 KiB in BF16, every rank holds all of
it, and a 256k session needs 13.4 GiB ([quantisation § 1](docs/quantisation.md#1-the-per-rank-budget)).
It is reported to run out of memory on four Sparks ([scouting § 3](docs/scouting.md)).

So the routed experts must be requantised to about 3.75 bits a weight, and the non-expert weights, which
cost as many decode bytes as the experts (13.1B attention parameters are read every token), must be FP8.

## Budgets that drive the design

| | |
|---|---|
| Weights a rank | **87 GiB** budget, 94 ceiling; each GiB over 87 costs ~19.6k tokens of replicated context |
| KV a token | 52.7 KiB (FP8: 656 B latent + 132 B indexer key on 21 layers), 93 KiB (BF16); 13.4 GiB for 256k in FP8 |
| Bytes a decode token a rank | routed experts 2.65 GB at 3.75 bpw (3.19 at NVFP4); attention 3.22 GB at FP8 (6.44 at BF16); shared experts, head, dense, routers ~1.65 GB; **7.53 GB → 32.0 ms a token** at 235 GB/s, ~31 tok/s before drafting (51 ms with BF16 attention and NVFP4 experts) |
| Prefill | all 78 layers run over every token (no compressed-encoder trick as DeepSeek's CED): ~5× the FLOPs a token of the DeepSeek engine; tensor cores: `mxf4nvf4` block-scaled MMA ~214 TFLOPS measured on DeepSeek's formats, bf16 ~107 |
| Collectives | 3 a layer × 78 at ~12 µs on the DeepSeek ring code: ~2.9 ms a step |
| Published bar | the fastest four-GB10 engine found: code 49.3 / prose 30.0 tok/s, cold prefill 1,374 tok/s at 8k (kindlingai/gr8, 3.25 bpw trellis, 2026-10-13); the best vLLM fork on sparkDash: prose 40.2 / code 46.6 |

## Decisions

1. **Routed experts: a mixed 3/4-bit code of our own at ~3.75 bits a weight, made from the BF16 master.**
   The leading candidate is a trellis code (EXL3/QTIP family: K3 and K4 widths per expert, K4 on down
   projections and the most sensitive experts), on the scouting's evidence that a data-free 3.42 bpw trellis
   scores inside NVFP4's KL range at three quarters of the bytes, while GGUF at 3.6–3.9 bpw is twice as far.
   The fallback is a 3-bit code restricted to the E2M1 grid mixed with NVFP4, which keeps the FP4 tensor-core
   prefill path (1.6× NVFP4's weight error on one sampled expert). **Experiment 1 decides between them**: a
   trellis decode GEMV must reach ≥ 95 % of the bandwidth floor on 48 SMs and its prefill must have a
   tensor-core path. NVIDIA's NVFP4 experts are plain round-to-nearest of the master (reproduced bit for bit),
   so "NVFP4 as published" is a function we can compute ourselves and serves as the quality anchor.
2. **Non-expert weights: zai's own FP8 values** (attention, shared experts, indexer, dense MLPs; 128×128 block
   scales from the FP8 release, which is round-to-nearest of the master), the head and routers as shipped,
   the embedding split by vocabulary. ~5 GiB a rank. o_proj at 4 bits (−3.7 ms a token) is a later lever,
   gated on its own KL.
3. **Exactness, in three levels** ([quantisation § 5](docs/quantisation.md#5-exactness-for-this-configuration)):
   (1) the served weights are a hashed artefact made deterministically from the hashed BF16 master by a
   recorded tool, with a manifest (source hashes, tool commit, calibration set, per-tensor error);
   (2) a written numerics contract (dequantisation of each format, **BF16 activations with f32 accumulation
   and no FP4 activation quantisation**, the KV format, the indexer key precision, top-k tie-breaking) with a
   layer-streaming torch reference; the engine keeps its pinned refhash outputs across every change and
   agrees with the reference within 2× the reference's own order-variance; (3) quality against the master on
   one fixed harness: KL and top-1 against BF16 logits, GSM8K-style accuracy, a needle at 128k/256k; the bar
   is NVFP4-as-published scored on the same harness (KL no worse, top-1 within 0.5 points, tasks within
   noise). The MTP layer sits outside level 3: its drafts are verified by the target model.
4. **Activations stay BF16.** NVIDIA's recipe quantises activations to NVFP4 with static scales; it is a
   prefill speed option that changes outputs and is unmeasured. It enters the contract only with its KL
   measured (experiment 4).
5. **KV cache: format and placement open.** BF16 is the contract's default; FP8 (656 B a token, vLLM's
   layout) enters with its KL measured. Replicated on every rank (one 256k session in the budget) or split
   by position (four 256k sessions or one 1M, two extra collectives a layer): decided after experiment 2's
   usage data and G4's step measurements.
6. **The MTP layer is the drafter**, at our own NVFP4 (or K3) since its precision changes speed, not output;
   chained draft passes, shared top-k (`index_share_for_mtp_iteration`), the DeepSeek verify loop. Expected
   1–3 drafts a step pay; acceptance against MTP precision is measured (experiment 5).
7. **Written from scratch, copying where it helps**: every dimension a compile-time constant, every kernel for
   the per-rank shapes, the chosen formats and 1–16 rows from the start (slot tables and batching designed in,
   not retrofitted). The DeepSeek engine's ring, weight container, graph and PDL step machinery, verify loop,
   batching, session store, server, front-end core, scripts and bench tools carry over; its kernels keep only
   their structure ([reuse § 1](docs/reuse.md#1-component-table)).
8. **Production on the ring keeps serving DeepSeek.** Every GPU job here is a ring job: build and check on the
   CPU first, then short (< 15 min) preemptible windows; the reference and the quantiser run on the CPU, on
   rank 2 (the box with the disk).

## Risks and open questions

- A trellis GEMV may not stay at the bandwidth floor on 48 SMs (decode is a 16-bit-state bit-shift and a
  1-MAD decode per weight), and its prefill needs a dequantise-to-tile path on the bf16 tensor cores (~107
  TFLOPS) unless the code is grid-restricted. Experiment 1.
- The budget leaves rank 0 ~4 GiB available instead of today's 21; earlyoom's line there is 2 GiB.
- Rank 1 has 74 GB of disk free and needs ~20 GB freed for its rank file; the BF16 master's experts are
  1.45 TB to fetch once at ~24 MB/s (~17 h), streamed a layer at a time through rank 2's disk, never whole.
- KL on this model has a high floor (two exact BF16 runs differing in batching: 0.11 mean on long text):
  only paired, same-harness comparisons count, and the reference's own variance sets the engine's bar.
- MTP acceptance at quantised precision is unstudied anywhere.
- The decode floor is ~32 ms a token before drafting: a third of the DeepSeek engine's per-token speed is
  the physics of 7.5 GB a token; the published bar is low (30–49 tok/s) because every published engine reads
  more bytes or runs slower kernels, so the target is the floor plus drafts, not a multiple of the bar.

## Experiments, in order

| # | What | Cost | Decides |
|---|---|---|---|
| 1 | Decode GEMV microbenchmarks on one GB10: trellis K3, E2M1-subset 3-bit, NVFP4, BF16-widened, at 1 and 6 rows on one layer's expert shapes; `mxf4nvf4` with E4M3 scales at group 16 (bit-exact, rate, nibble order); the bf16 `mma.sync` rate for trellis prefill tiles | a day of code, one ≤ 15-min preemptible ring window | the expert format (decision 1) |
| 2 | A CPU layer-streaming reference of [model.md](docs/model.md)'s math on the NVFP4 checkpoint, on rank 2: 32k tokens of agentic, code and prose text; routing counts, per-expert input statistics, BF16-contract logits (the first golden data) | the checkpoint copied to rank 2; hours of CPU at nice 19 | expert-usage skew, level-2 golden data, calibration statistics |
| 3 | Format bake-off by KL in the reference: layers 3–77's experts requantised a layer at a time from the master into NVFP4-as-published, E2M1-subset, int3 and trellis, as parallel tracks; KL against the master track | the master's experts fetched once, ~20 GB on disk at a time | the format and bit allocation (level 3); NVFP4's KL on our harness |
| 4 | FP8 KV and FP4 activations as tracks of 3 | CPU | decisions 4 and 5 |
| 5 | MTP precision against draft acceptance, replayed in the reference | CPU | the MTP layer's bits |
| 6 | The full conversion at the chosen layout, with the manifest and a byte-identical second conversion of three layers; the rank files cut | ~17 h of download once; rank 1's disk | the artefact |

The front-end port (tokenizer, template, tool-call parser; byte-exact against `tokenizers` and the jinja
template) starts at once on the CPU, since nothing above changes it.

## Milestones

From [reuse § 6](docs/reuse.md#6-build-plan); effort relative to the DeepSeek build's own milestones.

| # | Milestone | Done when |
|---|---|---|
| G0 | Formats and microbenchmarks (experiments 1–2) | decisions 1, 4, 5 settled here; findings recorded |
| G1a | First exact-output checkpoint: golden data on a few layers from the reference, one short sequence and one ≥ 4k tokens so the top-2048 selects | the reference's logits agree with an independent stack, or its greedy text reads right |
| G1b | Full golden data: all layers and the MTP layer on prose, code and a ~30k agentic transcript; MTP acceptance teacher-forced | argmax agreement with an independent stack within noise |
| G2 | Converter, verifier, loader: per-rank files in the kernels' tile layouts, the manifest | plan coverage, every byte verified, cross-rank CRCs |
| G3 | Kernels at TP=4 shapes on one rank, fed G1a's inputs: GEMVs in the chosen formats, absorbed sparse MLA attention, the indexer, norms, routing; slot tables and 16 rows from the start | every layer type matches golden at bf16 level; each GEMV ≥ 95 % of bandwidth |
| G4 | TP=4 decode: layer runner, fused collectives, one graph per row count, PDL, paced L2 prefetch | greedy decode within the reference's envelope; step time within 1.25× of the floor at k = 1 |
| G5 | Prefill: chunked, tensor-core GEMMs, sparse attention and indexer in prefill, the MTP cache fill | prefill state equals the step path's up to summation order; chunking changes no bit |
| G6 | Serving: server, sessions (exact restart at any common prefix), store, disk tier, kept sessions; dsfront with the GLM template | pi works unchanged against a test port; encoding byte-exact |
| G7 | MTP speculation: chained drafts, the gate, the spec graph, batch drafts | tokens identical with and without drafts; faster on every prompt class |
| G8 | Hardening and ship: service.sh, soak, refhash hashes, benches against the published stacks | as ds41f's M9 list |

Critical path: G0 → G1a → G2 → G3 → G4; G1b beside G2–G3; G6's front end from G0; G5 and G7 after G4.
