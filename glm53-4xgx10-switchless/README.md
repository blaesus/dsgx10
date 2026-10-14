# glm53-4xgx10-switchless

A from-scratch engine for **GLM-5.3** (zai-org's 753B-total / 40B-active `glm_moe_dsa` model: 78 layers,
256 routed experts top-8 plus one shared, DeepSeek-style sparse attention, one MTP layer) on the same four
GB10 boxes as [ds41f-4xgx10-switchless](../ds41f-4xgx10-switchless/) (two ASUS Ascent GX10 and two NVIDIA
DGX Spark, ranks 0–3, a switchless ConnectX-7 ring, TP=4).

Started 2026-10-13. The model does not fit the fleet at its released precision: 121 GiB of unified
memory per box, four boxes, and even NVIDIA's NVFP4 checkpoint is 432 GiB. So this configuration begins
with the quantisation question, and with what "exact output" means once the weights are quantised.

## Where it stands

Started 2026-10-13, **paused 2026-10-14** after one day of building (the user moved to other work). The plan is in
[DESIGN.md](DESIGN.md); its milestones G0–G7 are built and G8 (end to end on real weights) is partly done. The engine
decodes, prefills, serves and drafts on all four ranks with every routed expert at NVFP4 in the rank files, bit-exact
to its contract; what it lacks for production is the 3.8-bit serving files (the NVFP4 fill is 101.4 GiB a rank, over
the 87 GiB budget) and the KL bake-off that confirms their code. [Resuming](#resuming) says how to pick it up. The findings:

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
  the MTP layer's experts in the rank files at NVFP4; prefill at 1,106 / 1,145 / 1,027 tok/s at 2,600 / 8,192 / 32,768
  tokens after the GEMM wiring, the collective overlap and the pipelined prefill attention (about 3× the first
  runner, bit-identical; the published bar is 1,374);
  the service started on the ring, lending and reclaiming against the DeepSeek service; the CPU reference with golden data over 38k tokens
  ([docs/reference.md](docs/reference.md)). Every experiment is indexed in
  [docs/experiments.md](docs/experiments.md).
- **End to end on real weights (G8, 2026-10-14, partial):** with every routed expert in the rank files (NVIDIA's
  NVFP4), the engine agrees with the CPU reference within the reference's own order variance on all four golden
  sets (argmax 0.93 / 0.89 / 0.83 / 0.70 on short / python / prose / agentic's first 8k, against 0.93 / 0.90 /
  0.84 / 0.72 between the reference's two orders; KL ≤ 1.8× that variance): DESIGN decision 3's 2× criterion
  is met. The refhashes are pinned and bit-stable (graphs, eager, two processes, four ranks). Through glmfront,
  with drafts off, it decodes 22–23 tok/s at one session and 33 at two, and the text reads as GLM-5.3. The draft
  configurations through the front-end were not run: the work was paused
  ([docs/engine.md](docs/engine.md#end-to-end-on-real-weights-g8-2026-10-14)).
- **The expert code is in the kernels (E1, 2026-10-14):** the two-subset 3-bit code for gate/up with NVFP4 down
  (3.83 bits a weight) decodes at 89–95 % of bandwidth, bit-exact to its definition, with the packer and the
  loader's checks ([docs/experts.md](docs/experts.md) § 13, [docs/quantiser.md](docs/quantiser.md)).
- **Not yet:** the serving files at the mixture (the NVFP4 fill of 2026-10-14 is 101.4 GiB a rank, over the
  87 GiB budget; the mixture is 87.12), the quantiser's 3-bit fill and the KL bake-off against the master, the
  drafts' acceptance on real text (G7's speculative decoding is in and exact,
  [docs/engine.md](docs/engine.md#drafts-g7-2026-10-14)), and G8's draft configurations through the front-end.
| | |
|---|---|
| [DESIGN.md](DESIGN.md) | target, budgets, decisions, risks, the experiments in order, milestones |
| [docs/scouting.md](docs/scouting.md) | what is published for GLM-5.3 on GB10-class and other memory-limited hardware: the release, every quantised checkpoint, engines and their speeds, speculative decoding, quality against bits |
| [docs/quantisation.md](docs/quantisation.md) | the per-rank memory budget, what each tensor class costs at each precision, the candidate formats, how to produce and validate a requantisation, the exactness standard |
| [docs/model.md](docs/model.md) | the model's exact math as the reference code defines it, the NVFP4 and FP8 formats, the comparison with DeepSeek-V4.1-Flash, the tokenizer and chat template, the reference implementations |
| [docs/reuse.md](docs/reuse.md) | what of the DeepSeek engine carries over (~55 % by lines), what must be re-specialised, the build plan |
| `tools/` | `tally.py` (bytes by tensor class from safetensors headers), `nvfp4_entropy.py`, `provenance.py` (NVFP4 and FP8 against the BF16 master) |

## Resuming

State on 2026-10-14 (branch `glm53`, not pushed):

- **Rank files:** `/data/dsgb10/glm53/weights/rank0.glw` on rank 0 and `~/glm53/weights/rank{r}.glw` on ranks 1–3,
  108,844,417,024 B each, every routed expert at NVFP4 ([layout § 8](docs/layout.md) has the hashes); the 6.9 GB
  non-expert files beside them as `rank{r}.glw.prefill-only`. They leave 7–11 GiB a box during a job: contexts ≤ 8k,
  one or two sessions, no disk tier; and a CPU pass on a box may hold at most 3 GB while a job can load, or
  `earlyoom` (`-m 3 --prefer python`) kills it ([serving § 5.1](docs/serving.md)).
- **To run it:** `OWNER=me engine/server/service.sh start` (borrows the ring from the DeepSeek service, port 18894,
  [serving.md](docs/serving.md)); checks with `engine/model/run.sh check` and the server's `tests/ring_job.sh`.
- **Two decisions were pending:** (1) the 9–10 h download of the BF16 master's routed experts (1.45 TB streamed
  through rank 2, 43 MB/s with four connections) for the full KL bake-off ([reference § 10](docs/reference.md))
  and the final files: `engine/weights/experts_fill.py --recipe mix --master DIR` ([layout § 10](docs/layout.md),
  ~8 h of CPU, 87.12 GiB a rank); (2) whether the non-expert weights may go to NVFP4 (~7 ms of the 43.6 ms step;
  the user chose to keep FP8 for now, [DESIGN decision 2](DESIGN.md)).
- **Partial work with resume points:** the reference's covariance pass (layer 3) and resumable KL tracks on rank 2
  ([reference.md](docs/reference.md) "How to resume"); G8's end-to-end runs ([engine.md](docs/engine.md), G8): the
  drafts through the front-end on real text are left, `engine/server/tests/g8_job.sh k1` (k2, k3, gate) with
  `FRONT=<a glmfront build>`, one ring job of ~6 min each, compared with `g8_front.py compare`.
- **Scratch that may be deleted:** rank 0 `/data/scratch/glm53-master` (37 GB of BF16 expert samples for layers 3
  and 40, refetched in an hour), `~/dsgb10-data/glm53-*` on every rank (synthetic weights, dumps, traces; 29 GB on
  rank 0), rank 2 `~/glm53-ref/` (the venv, 38k golden set, qcache: keep if resuming within weeks).
- **Open misses, in order of value:** the serving files and bake-off; decode at 43.6 ms = 1.27× the floor
  (collectives' latency, attention); drafts worth only 2–26 % against fresh expert reads per verify row; prefill
  at 83 % of the published bar; a 256k session needs the FP8 cache or a position split.

## Memory per box

121 GiB unified, shared by CPU and GPU. The DeepSeek engine serves 72.4 GiB of weights per rank and
leaves 21–23 GiB available on each box.

## The checkpoint on disk

`nvidia/GLM-5.3-NVFP4` (ModelOpt, revision `e3b8f9f2`), 432.2 GiB, SHA-256 verified, on rank 0 only:
the 75 MoE layers' experts NVFP4 (337.5 GiB of E2M1 values, 42.2 GiB of E4M3 block scales at group 16: plain
round-to-nearest of the BF16 master, no weight calibration), the MTP layer's 256 experts plain BF16 (18.0 GiB),
the three dense MLPs FP8, everything else BF16 (attention 24.3 GiB, shared experts 5.3, head 1.8, embeddings
1.8). Its BF16 tensors are the master's own values ([quantisation § 3.0](docs/quantisation.md#30-what-the-checkpoints-numbers-are-measured)).
