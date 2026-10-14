# The layer runner and the TP=4 step for altar-1 (milestones A4, A5)

What `engine/model/` does for altar-1, as a delta from GLM-5.3's
([glm53 engine.md](../../glm53-4xgx10-switchless/docs/engine.md), "glm53" below): which weights the runner binds
in which format, how the decode step and prefill changed, the checks, and what was measured. Written 2026-10-14.
Sections 1–9 ran on **synthetic weights in the real formats**: no rank file had been installed when these jobs
ran. The golden comparisons, `tf`, `refhash` and the profile on the installed rank files are § 10 (A8). The kernels are
[attention.md](attention.md) (A3a) and [experts.md](experts.md) (A3e); the plan is [reuse.md § 5](reuse.md) and
[DESIGN.md](../DESIGN.md).

## Contents

1. [What is identical](#1-what-is-identical)
2. [What the runner binds](#2-what-the-runner-binds)
3. [The decode step](#3-the-decode-step)
4. [Prefill](#4-prefill)
5. [Checks](#5-checks)
6. [Measurements](#6-measurements)
7. [Memory](#7-memory)
8. [Runs, failures and what remains](#8-runs-failures-and-what-remains)
9. [Tuning: the fused shared expert and the routed gate/up's prefetch](#9-tuning-the-fused-shared-expert-and-the-routed-gateups-prefetch)
10. [End to end on real weights (A8)](#10-end-to-end-on-real-weights-a8)

## 1. What is identical

The step's structure is glm53's: the embedding, 78 layers of x-projection → all-gather → attn_prep → q_b (+ the
indexer's wq_b) → index_select on the 21 full layers → absorbed W_UK → sparse_attn → W_UV → o_proj → all-reduce 1
with the FFN norm → MLP/MoE → all-reduce 2 with the next norm, then the head and the sampler; CUDA graphs per row
count with PDL, the L2 prefetch, the posted partial sums and the overlapped prefill collectives (T5) unchanged. The
caches, the MTP layer's flow (78), the sampling and `check` mode's comparisons are glm53's. What changed is the
weights' formats and therefore every projection call, the shared expert, and the byte count.

## 2. What the runner binds

`Model::bind` (`bind.cu`) takes a projection's format from the rank file's layout and **dies** when it does not
match the plan below (DESIGN decision 5): `Layout(4)` (INT4_TILE) is INT4-g32, `Layout(5)` (BF16_TILE) is kBF16T,
`Natural` is row-major BF16. `detail::Proj {fmt, w, n, k}` replaces glm53's `Fp8W`; every projection call is
format-tagged (`GemvSeg`/the GEMM's format) so one call site serves both families.

| Weight (a rank's share) | Layers 0–2 (dense) | Layer 3, 77 | Layers 4–76 | MTP (78) |
|---|---|---|---|---|
| x-projection (q_a, kv_a, the indexer's wk and weights fused as glm53) | BF16 | BF16 | BF16 | BF16 |
| q_b (+ indexer wq_b, BF16, one call, one launch a format family) | BF16 | BF16 | **INT4** | BF16 |
| W_UK^T (absorbed, 64 × [512][192]) | BF16 | BF16 | BF16 (the exact dequantisation of the INT4 kv_b) | BF16 |
| W_UV | BF16 | BF16 | **INT4** | BF16 |
| o_proj (plain GEMV + all-reduce; `oproj.cu` removed) | BF16 | BF16 | **INT4** | BF16 |
| indexer weights_proj (full layers) | row-major BF16 (a kBF16T file tensor is untiled once at bind, `k_untile_bf16t`, because `index_weights` reads rows) |||||
| dense SwiGLU gate_up, down | BF16 | — | — | — |
| router [168][6144] BF16 + bias f32 | — | yes | yes | yes |
| shared expert gate_up [2×512][6144], down [6144][512] | — | BF16 | BF16 | BF16 |
| routed experts, 168 a layer, one pool | — | **BF16** (tag 0x333) | **INT4** (0x222) | BF16 (or NVFP4) |
| eh_proj | — | — | — | BF16 |
| head [vocab/4][6144] | BF16 (kBF16T in the file) |||||

The routed pool must be 168 entries back to back from `experts.0`, each with the layer's one format tag, and every
descriptor in `experts.desc` is checked on the host (`gx::desc_ok`, offset inside the pool) before the first step;
a reserved, unfilled pool runs synthetic experts of the layer's format with `--fill-experts`. Both BF16 forms
(row-major and kBF16T) are accepted for every BF16 projection: they are bit-equal (attention.md § 4).

Synthetic weights (`bind_synthetic`, `SynthOptions {seed, expert_pools = 4, bf16_pools = 1, bf16_rowmajor}`) are
built in exactly these formats: INT4 tiles filled by `k_fill_i4`, BF16 in kBF16T (`--rowmajor`: row-major), INT4
and BF16 expert pools with valid descriptors; 12.68 GB a rank at 4 INT4 pools.

## 3. The decode step

- **Projections**: `gemv` with a format per segment (INT4 `k_gemv4`, BF16 `k_gemv`); q_b and wq_b stay one call.
  o_proj is a plain GEMV whose partial goes into all-reduce 1 (glm53's FP8 `oproj_post` fusion and `oproj.cu` are
  gone; `kTuneOprojRemoved` is accepted and ignored).
- **Shared expert** (A3e's measured best): gate_up GEMV with the SwiGLU epilogue, split-K 4, into `hm`; down GEMV
  to f32, split 1, into `ys` (`yr + R` rows, contiguous with the routed output), **before** `moe_decode`, which adds
  `ys` as glm53's fused FP8 path did. Since § 9 (`kTuneShFused`, in the default since 2026-10-14) the same work runs
  inside `k_up`/`k_down` with the GEMVs' bits; the GEMVs remain for row-major shared weights and with the bit off.
- **Routed experts**: `moe_decode` on the layer's format; `Options::moe_up` picks the up variant (0 default; 6 =
  `gx::kUpPrefillOrder`, the K order of prefill, which makes decode = prefill bit for bit in every layer: `check6`).
- **L2 prefetch windows** sized from each `Proj`'s bytes (attention window; FFN window: router, shared gate_up and
  down).
- **Bytes a step** (`step_bytes`) per format: 8.708 GB at 1 row → **floor 37.05 ms** at 235 GB/s. DESIGN's 8.55 GB /
  36.4 ms predates the BF16 W_UK^T on INT4 layers (+0.16 GB, attention.md § 2).

## 4. Prefill

`pf_proj` checks each projection's format and runs INT4 through `dequant_int4` into a kBF16T scratch then the BF16
GEMM (attention.md § 5): `Pre.wscratch` = 6144 × 4096 BF16 = 50.3 MB a rank. The shared expert runs as two GEMMs
into `hm` / `ys`; the router GEMM is BF16. Everything else (chunking, the indexer's scratch, T5's overlapped
collectives, T6's `k_attn_pf` behind `kTunePfAttn`) is glm53's.

## 5. Checks

All on synthetic weights in the real formats, four ranks, `run.sh` through `ring.sh` (OWNER altar1-model).

| Check | Job (2026-10-14) | Result |
|---|---|---|
| `check` (`--pools 2`): chunkings 79/79 caches bit-identical, decode = prefill layers 0–3, 16 rows = 16 × 1 row, graph = eager, ranks agree, sampling | 11:31, commit da79a5d | **PASS** all |
| `check6` (decode in prefill's K order): decode = prefill in **all 79** caches, plus everything of `check` | 11:34, ffab66e | **PASS** all |
| `ab` 0 vs 1f7 at 1, 2, 4, 6, 16 rows: the levers do not change a bit | 11:49, ffab66e | same bits, PASS |
| `pfab` f7 vs 1f7 at 2,600 / 8,192 / 8,192 @ 24,576, chunks of 8,192 and of 2,048: same bits; **the hashes equal across the two chunkings** (ac6d6970a45927db, b21d8c59c6827347, 71006d0c74943db1) | 11:45, 11:55, ffab66e | PASS |
| server sources (`engine/server/*.cpp`) against the changed `model.h` | `g++ -fsyntax-only`, ffab66e + this commit | compile |

Done on the rank files in § 10 (A8), except `glw_kcheck --gpu`: `weights` (CRC + `glw_kcheck --gpu`), `tf-prefill`/`tf-decode` against the golden
data with `tools/compare_golden.py` (layers 0, 2, 3, 4, 6, 76, 77, 78, short sequence), `refhash` pins, `mtpgold`.

## 6. Measurements

Binaries: sources sha256 912d506eebf5e34c, built 2026-10-14 11:32 from commit ffab66e. Synthetic weights, 4 INT4
pools (layer l on pool l mod 4; 3.4 GB of INT4 experts, far beyond L2), context 4,096, graphs + PDL + L2 prefetch,
the median of the last 4 of 32 steps. Rank 2 also ran the CPU reference's pass during these jobs.

**Decode** (`run.sh bench`, job 11:38):

| Rows | GPU ms | bytes GB | floor ms | × floor | tok/s | glm53 ms (× floor) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **46.27** | 8.708 | 37.05 | 1.249 | 21.1 | 46.4 (1.35) |
| 2 | 60.92 | 12.03 | 51.19 | 1.190 | 32.3 | 60.8 (1.28) |
| 4 | 80.30 | 18.21 | 77.48 | 1.036 | 48.7 | 81.4 (1.13) |
| 6 | 100.39 | 23.81 | 101.32 | 0.991 | 61.3 | 97.2 (1.01) |
| 16 | 145.68 | 44.91 | 191.12 | 0.762 | 111.5 | 146.8 (0.76) |

Above 6 rows the synthetic router overlaps more than independent routing, so the floor overstates the bytes (as
glm53). Host launch 6.6–7.9 µs. The `ab` job (11:49, A/B/B/A, chunk 2,048 and scratch 128 MiB) measured
46.78 ms at 1 row with all levers (1.73 % over none): the target ≤ 46 ms is met to within 0.3–0.8 ms on this
synthetic run.

**Where the time goes** (eager, no PDL, events between classes, ms; job 11:38):

| Class | 1 row | 2 | 4 | 6 | 16 |
|---|---:|---:|---:|---:|---:|
| routed MoE (router + experts) | 18.38 | 30.73 | 46.92 | 59.31 | 86.51 |
| shared expert (BF16 GEMVs) | 8.55 | 8.67 | 8.73 | 8.79 | 9.42 |
| o_proj | 6.37 | 6.45 | 6.46 | 6.44 | 7.03 |
| x-projection | 3.30 | 3.35 | 3.37 | 3.39 | 3.68 |
| q_b + wq_b | 3.09 | 3.16 | 3.16 | 3.17 | 3.44 |
| W_UK + W_UV | 2.43 | 2.47 | 2.50 | 2.52 | 2.81 |
| attn_prep + index_select + sparse_attn | 4.06 | 4.66 | 4.83 | 5.45 | 9.31 |
| head + sample | 2.15 | 2.20 | | | |
| collectives (x all-gather, idxq all-gather, all-reduce 1, 2) | 10.81 | 14.19 | 13.42 | 19.79 | 32.16 |
| total (eager) | 59.19 | 75.93 | 91.58 | 111.05 | 156.59 |

Per layer at 1 row: dense 1.108 ms, MoE + full indexer 0.767, MoE + shared indexer 0.699. Against glm53 at 1 row
(eager): o_proj is 2.6 ms faster (INT4, 9.0 → 6.37), but routed + shared expert together are 5.8 ms slower (glm53's
21.1 with the FP8 shared expert fused into the routed kernels, here 18.38 + 8.55); the separate BF16 shared expert
costs 8.55 ms against A3e's 6.4 ms for the same two GEMVs standalone (experts.md § 8.3); why the step's are slower
was not traced further.

**Critical path in the graph** (`run.sh trace`, nsys on every rank, job 11:41; trace steps 47.0 / 67.8 / 85.4 /
~100.8 / ~141.9 ms at 1 / 2 / 4 / 6 / 16 rows; reports in `~/dsgb10-data/altar1-model/run/nsys/syn1/`, summary
`profile.txt`). 1 row, ms over the step, ranks 0 / 1 / 2 / 3:

| Kernel | rank 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| `k_up` (routed gate/up) | 11.06 | 10.56 | 11.91 | 11.02 |
| `k_ar_norm_row` (all-reduce + norm, incl. waiting) | 7.48 | 8.86 | 5.52 | 7.78 |
| `k_down` (routed) | 4.97 | 4.75 | 5.22 | 4.85 |
| `k_gemv4` o_proj (INT4) | ~3.6 | ~3.6 | ~3.7 | ~3.5 |
| `k_gemv` shared down | ~3.2 | ~3.1 | ~3.3 | ~3.2 |
| `k_attn` | ~2.3 | ~2.3 | ~2.3 | ~2.3 |
| head GEMV | ~2.0 | ~2.0 | ~2.0 | ~2.0 |
| `k_gemv4` q_b + W_UV (INT4) | ~2.0 | ~2.0 | ~2.0 | ~2.0 |
| `allgather_rows` | 1.2–2.0 | | | |
| x-projection GEMVs | ~1.2 | | | |
| shared gate_up | ~1.1 | | | |
| router | ~0.9 | | | |

Rank 2 (the reference's box) is the slowest in its own kernels and waits least in the all-reduce: it sets the step.

**L2 prefetch, A/B and B/A** (`run.sh l2pf`, job 11:51): 1 row −3.36 % (off 47.97 / 50.09, on 52.84 / 48.52 ms:
one 52.8 ms outlier), 6 rows +2.15 %, 16 rows +2.40 %. Kept on (glm53: +0.1 / +4.1 / +2.8 %); the 1-row figure
needs a repeat.

**The levers together** (`run.sh ab`, base 0 against 1f7, A/B/B/A, job 11:49): 1 row 48.15 → 46.78 ms (+1.73 %,
spread 0.945–1.045), 2 rows +2.03 %, 4 rows +1.07 %, 6 rows −0.05 %, 16 rows −0.99 %; same bits.

**Prefill** (`run.sh pfab`, one MTP draft, A/B/B/A, tok/s; A = f7, B = 1f7 = the default, which adds `k_attn_pf`):

| Prompt | at | chunks of 8,192 (job 11:45) A / B | chunks of 2,048, scratch 128 MiB (job 11:55) A / B | glm53 (T6, chunks of 8,192) |
|---|---:|---|---|---:|
| 2,600 | 0 | 1,120 / **1,166** | 963 / **992** | 1,106 |
| 8,192 | 0 | 1,123 / **1,176** | 1,003 / **1,046** | 1,145 |
| 8,192 | 24,576 | 885 / **925** | 820 / **860** | 908 |

B/A 1.031–1.049, the same bits in both, and the same hashes in both chunkings. The 2,048-row chunks (the memory
rule's setting while the reference and converter share the boxes) cost 11–15 %.

## 7. Memory

`glm_run` records free device memory and MemAvailable on every rank before the load and after the weights and
the prefill buffers, and **stops if any rank has less than 4.5 GiB MemAvailable** after the load (`min_free_mib
4608`, every run, synthetic too). The launchers set `oom_score_adj` 1000 (`run4.sh`). With the DeepSeek service
lending the ring, MemAvailable before the load was 108.7–116.0 GiB a rank.

| Job (2026-10-14) | setting | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---|---|---|---|---|
| 11:31 check, before | chunk 2,048, scratch 128 MiB | 84.18 / 112.15 | 113.60 / 116.01 | 85.79 / 108.74 | 103.56 / 115.59 |
| 11:31 check, after (10.85 GB synthetic) | | 67.97 / 96.03 | 97.53 / 99.95 | 69.49 / 92.44 | 87.30 / 99.52 |
| 11:38 bench, after (12.68 GB) | chunk 8,192, scratch 512 MiB | 71.08 / 96.03 | 97.43 / 99.85 | 74.43 / 95.98 | 79.49 / 99.15 |
| 11:55 pfab, after (max_ctx 33,792) | chunk 2,048, scratch 128 MiB | 74.76 / 95.75 | 97.14 / 99.64 | 75.23 / 92.43 | 71.42 / 98.65 |

(free device / MemAvailable, GiB.) The real files add ~69.6 GB (64.8 GiB) over the synthetic weights (76.6 GiB a rank
against 12.68 GB): an estimated 27–35 GiB MemAvailable would remain at these settings, above the 10 GiB the memory rule keeps; still to be
measured.

## 8. Runs, failures and what remains

Results: `~/dsgb10-data/altar1-model/results.tsv` and `.log` (not in the repo); binaries built by `install.sh` from
the commit named in `bin/SOURCES`. Every job: `OWNER=altar1-model …/ring.sh …/engine/model/run.sh <steps>`.

| Job (2026-10-14) | Sources, commit | Steps |
|---|---|---|
| 11:31 | a9e2df74c3671dbb, da79a5d | check (`--pools 2`) |
| 11:34 | 912d506eebf5e34c, ffab66e | check6 |
| 11:38 | same | bench |
| 11:41 | same | trace (rows 1, 2, 4, 6, 16) |
| 11:45 | same | pfab (chunks of 8,192) |
| 11:49 | same | ab, l2pf (chunk 2,048, scratch 128 MiB) |
| 11:55 | same | pfab (chunk 2,048, scratch 128 MiB) |

**Failures.** The first 2,048-row `pfab` was skipped silently: `run.sh` marks a step DONE per sources hash, so a
step repeated with other `EXTRA` options needs its DONE line removed (done by hand before the 11:55 job).

**What remains.**
- **Real files** (done in § 10, A8; kept as written): the converter's `engine/weights/` (magic ALTR1W01, layouts 4 and 5) must be merged into altar1
  before `bind_file` can load a rank file (this branch's loader is glm53's and rejects layouts above 3). Then
  `weights`, `tf` against `~/altar1-ref/golden/` (compare_golden.py, layers 0, 2, 3, 4, 6, 76, 77, 78), the
  `refhash` pins, `check` on the real files, the profile at 1, 2, 4, 6, 16 rows, MemAvailable with 76.6 GiB loaded.
  `run.sh`'s `GOLD` points at `~/dsgb10-data/altar1-model/gold/all`, not made yet.
- ~~The shared expert~~ fused, ~~the routed INT4 gate/up below 93 %~~ 96–100 % now, ~~the 1-row `l2pf` repeat~~ done:
  § 9. Still open: a step-level kBF16T vs `--rowmajor` A/B (attention.md measured it per kernel: 98–100 % vs
  92–95 %), and the items at the end of § 9.

## 9. Tuning: the fused shared expert and the routed gate/up's prefetch

Two exact levers, written 2026-10-14 (commits 2c4f6f7 to 5b85463, merged with altar1 at 4d888c8 before the last jobs):
**`kTuneShFused`** (bit 9) and **`kTuneUpXpf`** (bit 10), both in `kTuneDefault`, now **0x7f7** (was 0x1f7). Neither
changes a bit: every A/B below printed the same output hash for A and B, and those hashes equal the earlier 0x1f7
default's at every row count (1 row aa5124b549d4091c, 2 rows 74f6421cf905b042, 4 rows 8a0e4df83ce778d4, 6 rows
4752875ee45d0d7b, 16 rows 909806d4f89f5930: last hidden rows, sampled tokens and every layer's cache, all four
ranks). Synthetic weights in the real formats throughout (4 INT4 pools, context 4,096; the real rank files were not
bound). Those hashes are of the bf16 combine order; § 9.1's later change of contract (decision 8, commit 6477b29)
changes the outputs, and the tuning's numbers in §§ 9.2–9.5 were measured before it.

### 9.1 The MoE combine: decision 8 (the shared expert inside the routed experts' f32 sum)

**Decided 2026-10-14 (coordinator): the contract is glm53 DESIGN decision 8 as written**, which altar1's decision 2
adopts and in which A1's all-f32 golden set is made: the weighted outputs of a row's routed experts *and the shared
expert* are summed in f32 across experts and ranks and rounded to bf16 once. Until then both engines computed
y = bf16(bf16(Σ_ranks yr) + bf16(Σ_ranks ys)) (the routed and the shared sums all-reduced as two partials, each
rounded, then added: the reference's `--combine bf16` order for this step). Since commit 6477b29:

- **Decode**: `k_down` adds the shared expert's f32 output into each value's routed sum after the routed experts,
  one `__fadd_rn`: yr = fl(Σ_experts in id order (w·s2)·down(h) + ys) (`gx::DecodeCfg::ysum`). With the fused shared
  expert ys is k_down's own write; on the GEMV path it is the GEMV's output. yr is the only partial: it goes to
  `comm::glm::allreduce_add_norm` (posted: `k_down` writes R rows into the send buffer and posts them, the consumer
  is `allreduce_add_norm_posted`), not `allreduce2_add_norm`. Every layer with a shared expert takes this path:
  3–76 (INT4 experts), 3 and 77 (BF16), the MTP layer 78 (its draft pass runs the same `layer()`).
- **Prefill**: `sublayer_end` reduce-scatters yr + ys as one contribution (`comm::reduce_scatter_rows_f32_sum2`: the
  send copy adds the two f32 inputs, one `__fadd_rn` an element), then the residual add and the norm on one sum
  (`ops::add_rows` with one input). The f32 add is the same as decode's, so decode = prefill still holds.
- **The residual** (second decision, commit 59e9515): decision 8's sum is "rounded to bf16 once, when added to the
  residual", and the reference's contract mode computes h = bf16(f32(res) + Σ) with Σ still f32 for every
  sublayer. The engine rounded twice, h = bf16(res + bf16(Σ)). Now every add-norm collective (`k_allreduce_hidden`,
  `k_ar_norm_row`; plain, posted, two-partial) and prefill's `ops::add_rows` compute **h = bf16(f32(res) + Σ)**, for
  o_proj, the dense MLPs (layers 0–2), the MoE layers and the MTP layer, up to the final norm before the head; the
  RMSNorm after it is unchanged. The double rounding stays selectable as **`kTuneResRound2`** (bit 12,
  output-changing, in neither `kTuneDefault` nor `kTuneAll`; `glm_run --res-round2`). With bits 11 and 12 both on
  the engine gives the outputs it gave before 5087b2f, bit for bit (below).
- **Saved**: the second partial's traffic. Decode: R × 24,576 B less in each MoE layer's all-reduce (its send is
  R rows, not 2R): 75 layers × 24,576 B = 1.8 MB a step at 1 row, 29.5 MB at 16 rows (each rank's contribution).
  Prefill: one reduce-scatter fewer a MoE layer, m × 24,576 B: **201 MB a layer for an 8,192-row chunk**.
- **The old order** stays selectable: `kTuneMoeBf16Combine` (bit 11), the one tune bit that changes outputs and is
  never in `kTuneDefault` or `kTuneAll` (`glm_run --combine bf16`). It gives the earlier default's outputs bit for
  bit (the A/B below: its hashes equal 0x7f7's before the change at every row count).

**Bits** (synthetic weights in the real formats; commit 6477b29, binaries bin6):

- *Kernel level* (`bench_experts --shfused i4|b16`, run.sh `shf-*`, rank 0, 2026-10-14 14:52): for every up and
  down variant and two shared copies at 1, 2, 4, 6, 9 and 16 rows, the fused path's yr with `ysum` and the unfused
  path's (the GEMV's ys added in k_down) equal the plain host f32 add of the GEMV's ys into the unfused yr, bit for
  bit; 26 of 26 cases at every row count, INT4 and BF16 (and the fused h, ys and separate yr as before).
- *Model* (`run.sh check`, `check6`, 15:02 and 15:03, default 0x7f7): chunkings 79/79 caches identical; decode =
  prefill in layers 0–3, and in **all 79 caches with `--moe-up 6`** (decode's ysum against prefill's summed
  reduce-scatter); 16 rows = 16 × 1 row; graph = eager; the four ranks identical; sampling. All PASS.

**The step** (`run.sh ab`, A = 0xff7, the old order, B = 0x7f7, 10 rounds A B / B A, job 15:09):

| Rows | old (0xff7) | decision 8 (0x7f7) | B/A median (range) | gain | hashes old → new |
|---:|---:|---:|---|---:|---|
| 1 | 45.691 ms | 46.058 | 1.0026 (0.962..1.082) | −0.26 % | aa5124b549d4091c → ff56682e56a05a80 |
| 2 | 61.808 | 60.720 | 0.9784 (0.940..1.045) | +2.16 % | 74f6421cf905b042 → e702c76a862bebbc |
| 4 | 80.673 | 80.178 | 1.0011 (0.944..1.056) | −0.11 % | 8a0e4df83ce778d4 → 5c80d7d05b5398dc |
| 6 | 97.533 | 96.097 | 0.9853 (0.952..1.018) | +1.47 % | 4752875ee45d0d7b → 33bcb5cc90b266cd |
| 16 | 143.177 | 138.621 | 0.9662 (0.958..0.984) | **+3.38 %** | 909806d4f89f5930 → eddf9159f2920072 |

Neutral at 1 and 4 rows (inside the spread), faster at 16 rows, where the halved all-reduce is largest. The job's
check printed FAIL at every row count: a bug in the check, not the engine. After an even number of rounds the last
block runs A, so its "B again after the rounds" hash was A's. Every rank agreed, and A's hashes are the old
default's. Fixed in glm_run (set B before that hash), not rerun (ring time is A8's).

**The residual's one rounding** (commit 59e9515, binaries bin7 and comm bin7, synthetic weights):

- *Kernel level* (`comm/run.sh test`, `coll_tests`, four ranks, 15:20, sources `24532428137205f8`): `allreduce_add_norm`'s h and xn equal the
  host's f32 add rounded once (and with round2, rounded twice) and attn's rmsnorm on it, at 1–16 rows, slices 1, 8,
  16, 32, with and without PDL, separate and in place; `allreduce2_add_norm` the same with round2 on and off (its y
  is bf16 already). 510 checks ok, 0 failures, identical on all ranks.
- *Model* (`check`, `check6`, 15:26 and 15:27, default 0x7f7): every check PASS, as in the list above (incl.
  decode = prefill in all 79 caches with `--moe-up 6`).
- *The old outputs* (`run.sh ab`, A = 0x1ff7, bits 11 and 12, B = 0x7f7, 4 rounds, job 15:34): A's hashes equal the
  pre-5087b2f default's at every row count (1 row aa5124b549d4091c, 2 rows 74f6421cf905b042, 4 rows
  8a0e4df83ce778d4, 6 rows 4752875ee45d0d7b, 16 rows 909806d4f89f5930); the contract's (B) are e9f715e6a3eed0b9,
  ce60356e87c69fa3, bfe839ddafe0d05a, 12e7b27f6728c7b6, 93df0cf3e434e600; the check passes (B the same after the
  rounds, every rank the same). The epilogue loses one rounding and does nothing else, so its cost does not change.
  The job's times (46.40 → 46.22, 62.39 → 61.01, 81.52 → 78.55, 96.94 → 94.03, 140.12 → 134.41 ms) mix both
  changes over 4 rounds, so they are not a measurement of this one.

**Left**: the comparison against A1's golden data on the real files (A8, with bits 11 and 12 for the old order); a
prefill A/B (`pfab`) of the saved reduce-scatter; no check here ran drafts, so the MTP layer's path is in the code but
not exercised.

### 9.2 The fused BF16 shared expert (`gx::SharedW::bgu, bd`; docs/experts.md § 11)

The shared gate/up [1024][6144] (interleaved) and down [6144][512] are kBF16T, which is byte for byte the BF16 expert
block's lane order, so the expert kernels read them with their own fragment code. To keep the GEMVs' bits:

- **k_up**: shared tasks first (they need no routing), each of BT 8-row tiles (BT = 3 at WK 6, 1 at WK 4, 2, 1);
  every tile's K in the GEMV's 4 parts of 1,536 k, each part one MMA chain over its 24 chunks in k order (the
  GEMV's `mma_chunk` = the experts' `mma4`), the parts added in part order in shared memory, then the GEMV's SwiGLU
  epilogue (`__fdiv_rn`, `__fadd_rn`, `__fmul_rn`, the same rounding points) into h's shared slot. Its weights
  are prefetched into L2 before `griddepcontrol.wait` (only an L2 prefetch: holding them in registers across the
  routed tasks took the default kernel from 132 to 223 registers).
- **k_down**: each warp's NT shared down strips (8 KB each) prefetched into L2 before the wait, then one MMA chain
  over the 512 k from zero (the GEMV at split-K 1) into ys, posted with yr (since § 9.1: added into yr's f32 sum, yr alone posted).
- A **template parameter** (`SHB`): the unfused instantiations compile to the earlier code (the default INT4
  `k_up<1,6,2>` 132 registers, `k_down<2,1>` 155, as before); the fused ones take 168 (8 B of spill) and 218.
  moe_decode picks SHB from `LayerW::sh.bf16()`. The runner sets it with `kTuneShFused` when both
  shared tensors are kBF16T (row-major ones keep the GEMVs).

**Kernel bench** (`bench_experts --shfused`, rank 0, job 12:24, experts commit 2c4f6f7; A = route + the two GEMVs +
the unfused experts, B = route + the fused experts, A B B A, µs a layer; bits: fused h, ys and yr against the GEMVs
and the unfused step for every up and down variant and two shared copies, **26 of 26 identical** at every row count):

| Rows | INT4 A | INT4 B | B/A | BF16 A | BF16 B | B/A |
|---:|---|---|---:|---|---|---:|
| 1 | 283.4 / 278.7 | 277.6 / 276.7 | 0.986 | 742.3 / 741.7 | 734.5 / 731.7 | 0.988 |
| 2 | 457.6 / 458.0 | 460.6 / 458.0 | 1.003 | 1360.1 / 1354.5 | 1346.1 / 1345.2 | 0.991 |
| 4 | 795.7 / 798.2 | 798.5 / 799.5 | 1.003 | 2518.0 / 2518.6 | 2515.0 / 2513.2 | 0.998 |
| 6 | 1111.6 / 1091.8 | 1114.2 / 1100.4 | 1.005 | 3548.7 / 3529.4 | 3531.9 / 3528.6 | 0.998 |
| 16 | 2279.4 / 2283.8 | 2246.4 / 2246.9 | 0.985 | 7413.4 / 7419.3 | 7416.0 / 7411.4 | 1.000 |

The shared expert's 18.87 MB cost 80.3 µs at 235 GB/s; in a graph with PDL the two GEMVs already add only 75–85 µs
to the experts at 1 row, so the fusion cannot save bytes, only launch boundaries and tails.

**In the step** (`run.sh ab`, 10 rounds A B / B A in one process, chunk 8,192; gain = 1 − median of the rounds'
B/A, its range in brackets):

| Rows | 0x1f7 → 0x3f7 (job 12:30, commit 528a058) | 0x3f7 → 0x1f7 (job 12:40) |
|---:|---|---|
| 1 | 48.63 → 47.52 ms, **+1.82 %** (0.906..1.060) | 46.78 → 47.27 ms, **+1.08 %** for 0x3f7 (0.935..1.087) |
| 2 | 63.22 → 63.08, +0.13 % | 62.94 → 63.67, +1.05 % |
| 4 | 83.19 → 83.11, +0.26 % | 81.97 → 82.66, +0.11 % |
| 6 | 100.65 → 99.04, +0.73 % | 98.31 → 100.55, +1.44 % |
| 16 | 142.50 → 141.82, +0.73 % | 144.04 → 145.49, +1.46 % |

Small, positive in every cell of both orders, the same bits: kept (default since commit fd5528c).

**Failure: four chunks in flight at 9–16 rows** (commit c3cb8c6, reverted in c9bc785). The R16 instantiation of the
fused k_up (before § 9.3's prefetch) uses 207 registers, so one CTA of 6 warps fits an SM (grid 48). Holding 4 chunks instead of 8 brought it
to 138 registers (grid 96) but was slower, A B B A over two binaries (job ~13:00): 9 rows 1472.7 / 1471.5 µs
(8 chunks) and 1503.8 / 1501.0, 1508.9 / 1515.3 (4 chunks) and 1472.1 / 1475.9 (8 chunks again); 16 rows 2227.6 /
2226.7 against 2279.9 / 2279.4, 2280.3 / 2282.1, then 2230.8 / 2230.7.

### 9.3 The routed gate/up's cross-task prefetch (`gx::DecodeCfg::xpf`, `kTuneUpXpf`)

The second lever of the list (experts.md § 10 (b)). With the default WK 6, a warp owns 2 of an INT4 gate/up's 12
blocks and loads both at the start of its task, so nothing of the next task is in flight while one finishes. Now,
after the task's partials' barrier, each warp loads the next task's first P blocks (the dynamic queue's next task is
known then) and the epilogue runs over them; the next task skips its own first loads. The blocks, the MMA order and
the reduction are unchanged (the same bits; WK 1 variants have no such barrier and stay as they were). Cost:
the default INT4 k_up (fused shared expert and prefetch) takes 231 registers, 233 unfused, 248 at 9–16 rows: one
CTA an SM (`-Xptxas -v` of decode.cu at commit 5b85463's sources, no spills).

**Kernel bench** (`bench_experts --decode --ups 0,0,0,0 --xpfs 0,1,1,0`, one process, job ~13:20, commit 5b85463;
the decode step = route + k_up + k_down, µs and % of 235 GB/s):

| Rows | INT4 off | INT4 on | BF16 off | BF16 on |
|---:|---|---|---|---|
| 1 | 199.7 / 199.3 (93.0, 93.2 %) | 192.9 / 191.2 (**96.3, 97.1 %**) | 653.8 / 654.4 (98.3, 98.2) | 655.5 / 653.2 (98.0, 98.4) |
| 2 | 382.6 / 385.3 (94.9, 94.2) | 367.4 / 367.7 (**98.8, 98.8**) | 1269.7 / 1268.5 | 1269.4 / 1269.8 |
| 4 | 733.1 / 731.3 (95.2, 95.4) | 697.4 / 697.6 (**100.1, 100.0**) | 2444.1 / 2430.1 | 2435.7 / 2432.6 |
| 6 | 1031.0 / 1028.9 (95.8, 96.0) | 987.0 / 985.6 (**100.1, 100.3**) | 3437.7 / 3439.2 | 3443.7 / 3446.6 |
| 16 | 2325.6 / 2317.1 (90.4, 90.8) | 2102.8 / 2112.0 (**100.0, 99.6**) | 7506.7 / 7496.8 (96.9, 97.0) | 7285.1 / 7280.2 (**99.9, 99.9**) |

The experts' GPU suite passes with it on (`test_experts --fmt i4|b16|q4 decode prefill`, `equiv`: INT4 = BF16 on the
dequantised matrices bit for bit, every variant, decode wk1 = prefill, batch invariance, determinism; 0 failures).
A first build with it unconditional (d701398, bin3) against 2c4f6f7, two binaries A B B A, gave the same picture.

**In the step**:

| Rows | 0x3f7 → 0x7f7 (job 13:23, commit 5b85463) | 0x7f7 → 0x3f7 (job 13:28, commit 0a594e2) |
|---:|---|---|
| 1 | 47.80 → 47.06 ms, **+1.25 %** (0.926..1.062) | 46.41 / 46.28 ms, B/A 1.0070: **+0.70 %** for 0x7f7 (0.913..1.090) |
| 2 | 62.62 → 61.28, +1.53 % | 64.84 / 65.68, +1.65 % |
| 4 | 83.02 → 81.25, +2.11 % | 81.74 / 82.97, +1.86 % |
| 6 | 98.52 → 96.48, +2.62 % | 97.72 / 99.51, +1.77 % |
| 16 | 142.43 → 139.86, +1.72 % | 143.13 / 145.40, +1.49 % |

(At 1 row in the reverse job the medians of the blocks and the median of the rounds' ratios disagree in sign;
the ratio, taken within each round, is the reading.)

### 9.4 Before and after, and the checks

**The step, 0x1f7 (A4's default) against 0x7f7** (`run.sh ab`, commit 0a594e2, ms medians of 10 rounds):

| Rows | 0x1f7 → 0x7f7 (job 13:30) | 0x7f7 → 0x1f7 (job 14:16) | gain, both orders |
|---:|---|---|---|
| 1 | 46.98 → 46.91, +0.99 % (0.940..1.059) | 46.90 / 47.45, +0.84 % (0.998..1.142) | **+0.8–1.0 %** |
| 2 | 62.29 → 60.87, +2.62 % | 62.15 / 63.39, +1.91 % | **+1.9–2.6 %** |
| 4 | 82.05 → 79.70, +3.44 % | 82.34 / 84.58, +2.86 % | **+2.9–3.4 %** |
| 6 | 97.84 → 95.26, +2.26 % | 97.25 / 99.76, +1.40 % | **+1.4–2.3 %** |
| 16 | 145.96 → 142.71, +2.01 % | 139.70 / 141.65, +1.53 % | **+1.5–2.0 %** |

At 1 row the step is ~46.9 ms in these jobs (floor 37.05 ms, × 1.27); A4's 46.27 ms was one `bench` reading, and
single readings move by ±2 ms between jobs on this fleet (the service's and the serving agent's ring jobs, the CPU
reference on rank 2), so only the interleaved rounds carry a gain.

**Where the time goes**, eager without PDL (`run.sh bench`, ms; before: `--tune 1f7` job 12:53, commit fd5528c;
after: job 14:22, commit 0a594e2; the other classes within ±0.2 ms except the collectives, which wander by job):

| Class | 1 row before → after | 2 | 4 | 6 | 16 |
|---|---|---|---|---|---|
| routed MoE + shared expert | 18.32 + 8.52 = 26.84 → **25.17** | 39.10 → 37.56 | 55.48 → 53.01 | 68.07 → 64.64 | 94.72 → 90.16 |
| o_proj | 6.37 → 6.36 | 6.41 → 6.47 | 6.46 → 6.42 | 6.50 → 6.42 | 6.89 → 6.89 |
| x-projection | 3.32 → 3.32 | 3.34 → 3.37 | 3.38 → 3.35 | 3.50 → 3.39 | 3.60 → 3.47 |
| q_b + wq_b | 3.09 → 3.07 | 3.16 → 3.15 | 3.17 → 3.18 | 3.25 → 3.18 | 3.51 → 3.43 |
| W_UK + W_UV | 2.41 → 2.43 | 2.47 → 2.48 | 2.51 → 2.50 | 2.54 → 2.51 | 2.77 → 2.77 |
| attn_prep + index_select + sparse_attn | 4.04 → 4.02 | 4.63 → 4.61 | 4.80 → 4.82 | 5.42 → 5.37 | 9.15 → 9.04 |
| head + sample | 2.47 → 2.27 | | | | |
| collectives | 9.16 → 7.22 | 10.93 → 10.15 | 19.25 → 12.77 | 22.32 → 18.85 | 39.04 → 32.97 |
| **total (eager)** | 57.76 → 53.91 | 72.30 → 69.98 | 97.33 → 88.59 | 114.41 → 106.56 | 161.85 → 151.09 |

The eager figure overstates the old shared expert (two launches with events between classes); in the graph the
routed + shared critical path on rank 0 at 1 row (`run.sh trace`, `tools/step_profile.py`, increments) went from
k_up 11.06 + k_down 4.97 + the shared GEMVs ~4.3 = ~20.3 ms (A4, job 11:41) to k_up 14.16 + k_down 5.67 = 19.83 ms
(fused only, job 12:50) and k_up 13.56 + k_down 6.08 = 19.64 ms (both levers, job 14:22; rank 2, the reference's
box, 15.04 + 7.17). The trace steps of that last job (49.57 / 65.59 / 85.83 / 102.98 / 147.05 ms) are slower than
the 12:50 job's (46.30 / 64.33 / 80.27 / 99.26 / 143.77) in every kernel class, the ring's conditions, not the
levers (the interleaved A/Bs above).

**Checks on 0x7f7** (`run.sh check`, `check6`, job 13:25–13:27, commit 5b85463; the same on 0x3f7 at 12:45, fd5528c):
chunkings 79/79 caches identical; decode = prefill in layers 0–3 (default K order) and **in all 79 caches with
`--moe-up 6`** (the fused shared expert's WK 1 path against prefill's GEMMs); 16 rows = 16 × 1 row (batch
invariance: tokens, logits and 78 caches); graph = eager; the four ranks identical; sampling. All PASS.

### 9.5 The L2 prefetch at 1 row, repeated

`glm_run l2pf` now runs `--ab-rounds` (10) interleaved rounds (off on / on off, 12 steps a block, the median of the
last 8; graphs recaptured at each switch) and prints every round's ratio. Job 12:35, commit 528a058, default 0x1f7:

| Rows | off | on | on/off median (range) | gain | ratios |
|---:|---:|---:|---|---:|---|
| 1 | 48.46 | 47.95 | 0.9911 (0.940..1.037) | **+0.89 %** | 1.022 0.971 1.021 0.996 0.966 1.028 1.037 0.986 0.958 0.940 |
| 6 | 102.15 | 100.86 | 0.9783 (0.951..1.014) | +2.17 % | |
| 16 | 148.02 | 144.20 | 0.9802 (0.904..1.003) | +1.98 % | |

A4.1's −3.36 % at 1 row was the outlier: over ten rounds the 1-row effect is +0.9 % with 6 of 10 rounds below 1,
inside the spread (a wash at 1 row); +2 % at 6 and 16 rows as before. Kept on.

### 9.6 Levers not taken, failures, what remains

- **(b) the shared expert on a second stream** beside the router / routed kernels: not built. The fused path
  removes both launches; the GEMV pair already added only 75–85 µs a layer in a graph against its 80 µs of bytes,
  and the persistent k_up holds every SM, so a concurrent stream has no idle bandwidth to use.
- **(c) a faster standalone GEMV**: only bit-neutral launch shapes are allowed (the split-K is part of the bits);
  the GEMV is at 93.7–94.7 % standalone (experts.md § 8.3) and superseded by the fusion.
- **Failure**: `run.sh` skipped a repeated `ab` with other `AB`/`ABBASE` (DONE per sources hash, as A5.1); the
  scratch job script now deletes the `ab` DONE line before each pair. A session interruption at 13:38 killed one
  reverse-order A/B while it waited for the ring; it was rerun at 14:16.
- **Failure**: the 4-chunk fused shared expert at 9–16 rows (§ 9.2), reverted.
- **Left**: the INT4 k_up now runs one CTA an SM (231 registers by default, 248 at 9–16 rows); a `__launch_bounds__(192, 2)` variant (≤ 170 registers, maybe spills) was not tried. The
  attention-side INT4 GEMVs (q_b, W_UV, o_proj: 95–100 % at 1 row) were not touched. The real rank files were not
  bound for these A/Bs: the profile on them is the next measurement.

## 10. End to end on real weights (A8)

The engine's first runs on the four installed rank files ([layout.md § 7](layout.md#7-files-written), SHA-256
`1e4f8f6e…` / `1f4f6610…` / `1c3d81e8…` / `5ce3b0fd…` for ranks 0–3), every routed expert from the file, written
2026-10-14 (+0100). glm53's G8 is the template ([glm53 engine.md, G8](../../glm53-4xgx10-switchless/docs/engine.md#end-to-end-on-real-weights-g8-2026-10-14)).
17 ring jobs, 14:44–16:37, all through `ring.sh` as OWNER altar1-g8, all preemptible, none preempted; every job
exited 0. Data and logs in `~/dsgb10-data/altar1-g8/` on rank 0 (`results-<job>.<time>.tsv`, `chain/job-<job>.<time>.log`,
`g8/<tag>/<seq>/` dumps, `g8-<name>.json` comparisons); the comparisons' copies on rank 2 in the same tree.

**Two engines were measured**, because the contract changed during the milestone:

| Binaries | Commit | `glm_run` sources | What |
|---|---|---|---|
| **pre-fix** | `59139aa` (altar1 `01dec5c` + this milestone's scripts) | `618114de8b64441f` | the tuning's default 0x7f7; MoE combine kTwo, `bf16(bf16(Σ routed) + bf16(Σ shared))`, and every residual add rounded twice |
| **fixed** (the default from now on) | `4cbe861` (altar1 `fb447a5` + the scripts) | `307d477e1feaf5fd` | decision 8 exactly (§ 9.1): shared and routed experts in one f32 sum, every sublayer's residual add rounded once, `h = bf16(f32(res) + sum)`; the old order is tune bits 11 + 12 (`--combine bf16 --res-round2`) |

Both built with `install.sh` (`DATA_NAME=altar1-g8`, nice 19, -j4) into `~/dsgb10-data/altar1-g8/build`, installed
into `~/dsgb10-data/altar1-g8/bin` on every rank (`bin/COMMIT`, `bin/SOURCES`; the sources hash also covers `tools/*.py`, so the docstring edit to `compare_golden.py` made after the builds changes it without touching a binary). The jobs are
`engine/model/g8_job.sh` (`w a b c d o m k p f`), run back to back by `g8_chain.sh`; `run.sh` takes `GW="--weights
fleet"` to run its synthetic steps (check, bench, pfab, …) on the rank files.

**Golden data.** `run.sh`'s `GOLD` is `~/dsgb10-data/altar1-model/gold/all` on every box: these are **the `all-f32`
set's raw copies** (`gold_prep.py` output, `~/altar1-ref/gold/all-f32.*` on rank 2, 0.93 GB) copied under the name
the scripts expect, SHA-256 equal on all four boxes (`gold_copy.log`). The comparisons ran on rank 2 against
`~/altar1-ref/golden/` with `compare_golden.py g8` (nice 19, 4 threads), the engine's dumps copied there.

### 10.1 The weights

Job `w` (14:44, pre-fix binaries): every rank loaded its file to the GPU with every tensor's CRC32C checked on the
device copy (`glw_load --verify-crc`): **13,961 / 13,961 tensors, 76.75 GiB, 0 reserved, every CRC matches**, on all
four ranks (load 8.4–15.9 s from the page cache on ranks 1–3, 40.6 s from rank 0's USB disk; the device CRCs 5.2–5.6 s).
`glw_kcheck --rank 0` (host binding): 154,542 checks ok, 1,175 bad; **the expert side has no bad check** (36,792 INT4
and 1,512 BF16 matrices, sizes by tag, contiguity, every descriptor), and all 1,175 bad checks are its non-expert
half still expecting glm53's FP8 T64 entries with scale tables (e.g. `layers.78.shared.gate_up.weight: BF16
1024x6144 bf16-tile, the kernels read F8_E4M3 1024x6144 T64`), as layout.md § 6 found. **`--gpu` was not run**: its
device half binds every projection as FP8 (`host_fp8` dereferences the missing `.scale` entries) and runs the head
row-major, so on these files it would crash before testing anything; porting it to the INT4 tile and kBF16T is a
rewrite of its device half, not done. What covers the projections on the files' bytes instead is § 10.3: every
layer's output against the reference at BF16 noise in layers 0–6.

### 10.2 Memory after the load

`glm_run` records free device memory and MemAvailable on every rank before the load and after the weights and the
prefill buffers, and stops below 4.5 GiB MemAvailable (`--min-free-mib 4608`); no job stopped. The weights bind
82.41 GB (76 MoE layers' experts from the file) in 43–49 s. The reference's `all` pass (≤ 6.8 GiB) ran on rank 2
throughout, and the DeepSeek service had lent the ring. GiB, free device / MemAvailable:

| Job | setting | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---|---|---|---|---|
| before the load (range over the jobs) | | 74–88 / 113.3–113.7 | 113.0–113.1 / 115.9–116.0 | 97.6–102.1 / 109.2–112.2 | 114.2–114.7 / 115.9–115.9 |
| `k` check (14:54) | chunk 2,048, scratch 512 MiB, 6 slots × 8,192 | 6.65 / 29.23 | 29.55 / 32.48 | 16.55 / 26.35 | 30.92 / 32.38 |
| `p` bench (15:04) | chunk 8,192, scratch 512 MiB | 5.24 / 31.86 | 31.20 / 34.14 | 20.28 / 29.81 | 32.46 / 34.02 |
| `f` pfab (15:05) | chunk 8,192, max_ctx 33,792, one draft | 2.20 / 29.05 | 29.04 / 31.99 | 16.58 / 26.37 | 30.37 / 31.86 |
| `a` tf + refhash (15:45, fixed) | chunk 2,048, scratch 128 MiB, 6,400 | 5.90 / 33.89 | 33.45 / 36.40 | 20.66 / 29.43 | 34.63 / 36.28 |
| `c` tf (15:26, pre-fix) | chunk 2,048, scratch 128 MiB, 8,192 | 0.97 / 33.23 | 33.28 / 36.22 | 18.79 / 30.42 | 34.79 / 36.10 |

**At least 26.35 GiB MemAvailable on every rank after the load in every job** (rank 2, beside the reference), 29.05
on rank 0, against glm53 G8's 6.7–11.9 GiB with 101 GiB of weights; § 7's estimate was 27–35. Free *device* memory
(cudaMemGetInfo) on rank 0 falls to ~1 GiB in some jobs while MemAvailable stays at 29–34 GiB: on unified memory
the driver's figure tracks what else holds pages on that box (the agents' processes, the page cache of its 82 GB
file on the USB disk), and MemAvailable is what earlyoom watches.

### 10.3 Against the golden data

`tf` teacher-forced over the golden sequences (short, python and prose whole; agentic's first 8,000 of 26,834
tokens) via prefill (2,048-row chunks), and via decode (16-row steps, one draft pass) on short and python; the dumps
against `all-f32` (the contract's order, the level-2 set) and `all` (transformers' bf16 order, finished on rank 2 at
17:01:41) with `compare_golden.py g8`, which also compares the two runs with each other over the same positions:
the reference's own order variance. KL(golden ‖ engine) and max |Δ logit| over the full logits at the golden sample
rows (every row of short; 89 / 84 / 65 rows of python, prose, agentic's prefix); argmax and top-20 over every row.

**The bar.** DESIGN decision 2: the engine within 2× the reference's own order variance, here `all` against
`all-f32` (KL(all-f32 ‖ all), the same direction as the engine's). Before `all` existed, the interim bar was
reference.md § 6.2's `a1a` pair (short 0.869 / KL 0.098 / 7.6; prose2400 0.874 / 0.160 / 14.7); short's `all`
equals `a1a-bf16` (the engine's comparison with both gives the same digits), and its 0.098 is the other direction
of the same pair (KL(bf16 ‖ f32); 0.092 this way round).

Via prefill, fixed engine (jobs `a` 15:45, `c` 15:57), and the pre-fix engine (jobs 15:14, 15:26) on the same text
(comparisons 17:03–17:20):

| Sequence | against | engine | argmax equal | top-20 overlap | KL mean (p99, max) | max \|Δ logit\| |
|---|---|---|---:|---:|---|---:|
| short (291) | `all-f32` | **fixed** | **0.869** | 0.865 | **0.095** (0.92, 3.82) | 6.3 |
| | | pre-fix | 0.904 | 0.864 | 0.081 (0.69, 1.46) | 6.7 |
| | `all` | fixed | 0.863 | 0.867 | 0.115 (1.29, 3.30) | 7.2 |
| | | pre-fix | 0.873 | 0.865 | 0.087 (0.81, 1.06) | 7.1 |
| | *orders* | | *0.869* | *0.860* | *0.092 (0.86, 2.10)* | *7.6* |
| python (6,106) | `all-f32` | **fixed** | **0.913** | 0.862 | **0.291** (3.66, 3.93) | 12.4 |
| | | pre-fix | 0.916 | 0.859 | 0.240 (3.68, 4.81) | 11.4 |
| | `all` | fixed | 0.909 | 0.859 | 0.224 (2.86, 3.12) | 10.9 |
| | | pre-fix | 0.920 | 0.861 | 0.190 (2.32, 6.28) | 10.3 |
| | *orders* | | *0.915* | *0.860* | *0.205 (3.50, 4.40)* | *10.5* |
| prose (4,762) | `all-f32` | **fixed** | **0.863** | 0.859 | **0.049** (0.47, 0.53) | 6.6 |
| | | pre-fix | 0.864 | 0.863 | 0.049 (0.47, 0.53) | 6.0 |
| | `all` | fixed | 0.860 | 0.857 | 0.052 (0.65, 0.76) | 5.8 |
| | | pre-fix | 0.862 | 0.859 | 0.063 (0.97, 0.98) | 7.7 |
| | *orders* | | *0.870* | *0.863* | *0.048 (0.42, 0.72)* | *4.8* |
| agentic (first 8,000) | `all-f32` | **fixed** | **0.738** | 0.759 | **0.364** (6.20, 6.78) | 13.5 |
| | | pre-fix | 0.729 | 0.753 | 0.453 (9.04, 11.07) | 14.6 |
| | `all` | fixed | 0.731 | 0.755 | 0.406 (6.78, 9.30) | 13.3 |
| | | pre-fix | 0.734 | 0.754 | 0.379 (6.15, 11.61) | 15.6 |
| | *orders* | | *0.736* | *0.754* | *0.323 (6.17, 9.15)* | *13.0* |

Top-1 against the text, engine (fixed) / `all-f32` / `all`: short 0.545 / 0.528 / 0.541, python 0.684 / 0.686 /
0.688, prose 0.424 / 0.422 / 0.427, agentic 0.472 / 0.475 / 0.470.

**DESIGN decision 2's 2× criterion: met on all four sequences.** The fixed engine against the contract's order
(`all-f32`), its disagreement over the reference's own:

| | short | python | prose | agentic (8k) |
|---|---:|---:|---:|---:|
| argmax (1 − equal) | 1.00× | 1.02× | 1.05× | 0.99× |
| KL mean | 1.04× | 1.42× | 1.00× | 1.13× |
| max \|Δ logit\| | 0.83× | 1.18× | 1.38× | 1.03× |

The pre-fix engine was inside the bound too (argmax 0.74 / 0.99 / 1.05 / 1.03×, KL 0.88 / 1.17 / 1.02 / 1.40×).
Python's KL (1.42×) is the closest to the bound: 89 sample rows, its p99 is the engine's and the orders' alike.

**Via decode, one draft pass** (jobs `b`; fixed 15:47, pre-fix 15:19): fixed short argmax **0.866** against
`all-f32` (KL 0.095, max |Δ logit| 8.6) and 0.859 against `all` (KL 0.112); python **0.913** (KL 0.157, 9.0) and
0.911 against `all` (KL 0.175). Pre-fix: short 0.894 / KL 0.096 / 6.3, python 0.917 / 0.222 / 11.6 against
`all-f32`. Within the same envelope as prefill (argmax 1.03× and 1.02× the orders' disagreement, KL 1.03× and
0.76×); decode's routed sums are in their own K order (decode = prefill bit for bit only with `--moe-up 6`, § 10.5).

**Hidden states, short, via prefill** (median relative error of the rows against `all-f32`; the orders' own in the
last line; cosine mean ≥ 0.99999 through layer 3 in all three):

| After layer | 0 | 1 | 2 | 3 | 4 | 6 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 77 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | **0.18 %** | 0.29 | 0.32 | 0.39 | 0.47 | 0.57 | 0.81 | 5.45 | 7.22 | 11.2 | 11.1 | 14.4 | 18.0 | 19.9 |
| pre-fix | 0.32 % | 0.40 | 0.43 | 0.49 | 0.57 | 0.68 | 1.10 | 5.92 | 8.12 | 11.0 | 10.9 | 14.4 | 18.3 | 20.2 |
| orders (a1a bf16 vs f32) | 0.32 % | 0.40 | 0.43 | 0.50 | 0.58 | 0.70 | 1.12 | 5.62 | 7.60 | 10.7 | 11.2 | 14.6 | 18.8 | 20.4 |

Python's rows give the same through layer 10 (fixed 0.18 / 0.28 / 0.40 / 0.83 % at layers 0, 1, 3, 10). Against
`a1a-bf16` the pre-fix engine was 0.20 % at layer 0 and the fixed one 0.32 %: **the fix moved the engine from the
bf16 order's side to the contract's**, and it now sits closer to `all-f32` than the reference's other order does,
by 25–45 % in layers 0–10. **It is not bit-equal to the contract even at layer 0**: 0.18 % median relative error after
a dense layer with BF16 weights on both sides is more than f32 accumulation order alone would flip; some rounding
point inside the attention or the dense MLP still differs (§ 10.7). From layer ~15 routing flips (a quarter of rows
have their 8th and 9th scores within 1e-3, reference.md § 6.3) amplify every difference, and engine and orders
reach the same ~20 % by layer 77.

**What the two fixes bought against golden:** the hidden states in layers 0–10 (−25 to −45 %, above). The final
logits do not resolve the fix: against `all-f32` short loses 3.4 points of argmax and python's KL rises 0.05, agentic gains 0.9 points
and 0.09 nats of KL, prose is unchanged. Both engines are inside the same envelope (§ 10.3's ratios), and from layer 15 the comparison measures the routing's chaos, not the arithmetic.

The old order through the fixed binaries (job `o`, 16:04, `--combine bf16 --res-round2`): the refhash pins below
equal the pre-fix engine's (`3d71bf700e718117`, `cefa4693767795d5`) and tf's top-1 and mean log p equal the pre-fix
run's on short and python: tune bits 11 + 12 reproduce the pre-fix engine on the real files, as § 9.1 found on
synthetic ones.

### 10.4 The MTP layer

**Teacher-forced on the golden final hidden** (`mtpgold`, job `m`; fixed 16:19, pre-fix 15:36; blocks of 16 rows,
five chained passes): the MTP's first draft equals the reference's MTP argmax on **0.9925** of 37,985 rows (fixed;
short 1.000, prose 0.994, python 0.996, agentic 0.991; pre-fix 0.9928), and the one-draft greedy acceptance
(draft = the reference's main argmax at t+1) is **0.4921** against the reference's **0.492** (short 0.661, prose
0.523, python 0.698, agentic 0.438, as reference.md § 6.1's 0.661 / 0.522 / 0.698 / 0.438). Layer 78's output
against the golden rows: cosine mean 0.9987–0.9998, mean relative error 0.9–1.7 % (short 1.07 %, prose 0.92,
python 1.74, agentic 1.55; pre-fix 1.28 / 1.10 / 1.79 / 1.65). Chained drafts from each block's last row, mean
accepted length against the main argmax at k = 5: 0.88 / 0.55 / 1.07 / 0.53 (short / prose / python / agentic).

**On the engine's own hidden states** (tf via decode with one draft pass, job `b`): the draft equals the reference's
MTP argmax on 0.869 (short) and 0.896 (python), fixed (pre-fix 0.841, 0.900); the reference's two orders agree on
0.872 and 0.909 of the drafts (`all` against `all-f32`; the engine against `all`: 0.862, 0.893). Acceptance on the engine's own text (draft = its own main argmax at t+1): **0.645**
and **0.687** (pre-fix 0.641, 0.693) against the reference's 0.662 and 0.698. These two jobs are the first runs of
layer 78 under the fixed combine; both pass.

### 10.5 The refhash pins and the checks

Greedy, 64 tokens after the prompt, each step queued one ahead (the server's way), `--max-ctx 6400 --chunk 2048
--index-scratch-mib 128`; the hash is over every token's id and logit bits (no drafts):

| Prompt | **fixed** refhash | runs (all four ranks agree in each) | pre-fix (= `--combine bf16 --res-round2`) |
|---|---|---|---|
| short (291 tokens) | **`975edafcbf8d617a`** | graphs, graphs, eager (job `a`, 15:45); graphs in a second process (job `d`, 16:13) | `3d71bf700e718117` (jobs 15:14 ×3, 15:34; job `o` 16:04) |
| python (6,106 tokens) | **`1dd62741b1158d49`** | the same four | `cefa4693767795d5` (the same five) |

The fixed engine's greedy continuation of short begins `785 1196 374 10156 … 25704 264` and leaves the pre-fix one
at the 24th token; python's at the 18th. tf hashes (rank 0; they cover the dumped layers, so they are per job
setting): fixed prefill short `89f0df63a231601f`, python `e6716385592bae07` (layers 0–77), prose `7d071927b1130a3c`,
agentic 8k `d009ace907c5bb42` (layer 77); fixed decode short `3f8b0e4a08d6b3d4`, python `04ba7ac6e3ae3b23`. Each
ran once: recorded, not pinned.

**`check` and `check6` on the real files** (job `k`; pre-fix 14:54–14:56, fixed 16:30–16:37): chunkings 79 / 79
caches identical; decode = prefill in layers 0–3 (default K order) and in **all 79 caches with `--moe-up 6`**; 16
rows in one step = 16 steps of one row (tokens, logit bits, 78 caches); graph = eager; the four ranks identical;
sampling. **All PASS on both engines.**

### 10.6 The step and prefill on real weights

**Decode** (`run.sh bench` with `GW="--weights fleet"`, context 4,096, graphs + PDL + L2 prefetch, tune 0x7f7, the
median of the last 4 of 32 steps; one reading a job):

| Rows | fixed (job `p`, 16:25) GPU ms | tok/s | pre-fix (15:04) GPU ms | synthetic, same default (§ 9.4, 0x7f7) | bytes GB | floor ms | fixed × floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **45.70** | 20.9 | 46.52 | 46.91 | 8.708 | 37.05 | 1.233 |
| 2 | 63.21 | 31.5 | 66.73 | 60.87 | 12.03 | 51.19 | 1.235 |
| 4 | 92.35 | 44.0 | 91.58 | 79.70 | 18.21 | 77.48 | 1.192 |
| 6 | 116.08 | 52.9 | 114.67 | 95.26 | 23.81 | 101.32 | 1.146 |
| 16 | 193.84 | 81.7 | 200.59 | 142.71 | 44.91 | 191.12 | 1.014 |

At 1 row the real files run as fast as the synthetic weights (45.7–46.5 against 46.3–46.9 ms; single readings
move ±2 ms between jobs), 1.23× the floor and inside A4's ≤ 46 ms. Above one row they are slower: the bench feeds
synthetic token ids, so with the real router the rows route close to independently and read the distinct experts
the floor column assumes (it is computed for independent uniform routing), while the synthetic router overlapped
(§ 6); at 16 rows the step is 1.01× that floor. Real text routes with more overlap than random ids: the server's
multi-session rates on real prompts are the bar for 2–16 rows, not this table. The fix's removed reduce-scatter
(one partial instead of two in all-reduce 2) does not show beyond the ±2 ms between single readings (45.70 against
46.52 ms at 1 row); an interleaved A/B of `--combine bf16 --res-round2` against the default would resolve it.

**Where the time goes**, eager without PDL (ms; events between classes; `bench`'s profiled step):

| Class | 1 row pre-fix | 1 row fixed | 16 rows pre-fix | 16 rows fixed |
|---|---:|---:|---:|---:|
| MLP / MoE (router, routed + shared experts) | 26.35 | 25.35 | 144.87 | 143.69 |
| o_proj | 7.03 | 6.43 | 7.51 | 7.04 |
| x-projection | 3.71 | 3.38 | 4.03 | 3.79 |
| q_b + wq_b | 3.32 | 3.16 | 3.59 | 3.53 |
| W_UK + W_UV | 2.50 | 2.46 | 2.72 | 2.76 |
| attn_prep + index_select + sparse_attn | 4.15 | 4.06 | 9.69 | 9.35 |
| head + sample | 2.27 | 2.34 | 2.56 | 2.23 |
| collectives (all-gathers, all-reduce 1 and 2) | 5.59 | 10.65 | 34.28 | 34.62 |
| total (eager) | 55.00 | 57.89 | 209.32 | 207.08 |

Per layer at 1 row (fixed): dense 1.096 ms, MoE + full indexer 0.775, MoE + shared indexer 0.671. The eager
collectives wander between jobs (5.6 against 10.7 ms here, 7.2–9.2 in § 9.4) with the ring's and the boxes'
conditions; the kernels' classes agree with § 9.4's synthetic ones to within ~1 ms (o_proj 6.4–7.0 against 6.4).

**Prefill** (`run.sh pfab`, job `f`, 15:05–15:14, pre-fix binaries; chunks of 8,192 rows, one MTP draft, A = f7,
B = 1f7, A B B A): **973 tok/s at 2,600** and **1,115 at 8,192** (913 at 8,192 @ 24,576), B/A 1.017 / 1.055 /
1.045, A and B the same bits (hashes `db18fb118b03300c`, `c31f9b58bc0059e8`, `55fd47b7d006ac96`). Against the
synthetic 1,166 / 1,176 / 925 (§ 6): −17 % at 2,600, −5 % at 8,192, −1 % at 24,576: real routing reads more
distinct experts a chunk than the synthetic router, which matters most for the short prompt. Against glm53's T6
1,106 / 1,145 / 908 and A5's target: below at 2,600 and 8,192 on real weights. Not repeated on the fixed engine:
the fix changes one rounding in `sublayer_end`, not a GEMM (python's 6,106-token prefill in the refhash runs, 2,048-row chunks: 6.83–6.91 s before, 6.58–6.67 s after).

### 10.7 What remains

- The `prefill-kTwo`, `prefill-fix` and `decode-fix` comparisons against `all` are in `g8-*-all.json`; reference.md
  § 10's own steps for `all` (summary, `cmp_runs.py`, `level3.py`, SHA256SUMS) are the reference agent's.
- **Layer 0 is not bit-equal to the contract** (0.18 % median relative error after a dense layer, § 10.3): a
  per-operation comparison of layer 0 (attention's intermediate roundings, the dense MLP's SwiGLU, the norms)
  against `a1a-f32`'s every-position dumps would find the rounding point that differs.
- `glw_kcheck`'s device half for INT4 tiles and kBF16T (§ 10.1).
- Prefill on real weights is 17 % below the synthetic figure at 2,600 tokens; drafts on real text through the server
  (the serving agent's); an interleaved A/B for the fix's step gain.
