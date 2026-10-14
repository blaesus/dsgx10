# The layer runner and the TP=4 step for altar-1 (milestones A4, A5)

What `engine/model/` does for altar-1, as a delta from GLM-5.3's
([glm53 engine.md](../../glm53-4xgx10-switchless/docs/engine.md), "glm53" below): which weights the runner binds
in which format, how the decode step and prefill changed, the checks, and what was measured. Written 2026-10-14.
Everything here ran on **synthetic weights in the real formats**: no rank file had been installed when these jobs
ran, so the golden comparisons, `tf`, `refhash` and the real-file profile are still to do (§ 8). The kernels are
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
  `ys` as glm53's fused FP8 path did. Not fused into `k_up`/`k_down` (§ 8).
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

Not done (need the rank files): `weights` (CRC + `glw_kcheck --gpu`), `tf-prefill`/`tf-decode` against the golden
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
- **Real files**: the converter's `engine/weights/` (magic ALTR1W01, layouts 4 and 5) must be merged into altar1
  before `bind_file` can load a rank file (this branch's loader is glm53's and rejects layouts above 3). Then
  `weights`, `tf` against `~/altar1-ref/golden/` (compare_golden.py, layers 0, 2, 3, 4, 6, 76, 77, 78), the
  `refhash` pins, `check` on the real files, the profile at 1, 2, 4, 6, 16 rows, MemAvailable with 76.6 GiB loaded.
  `run.sh`'s `GOLD` points at `~/dsgb10-data/altar1-model/gold/all`, not made yet.
- **The shared expert** (8.55 ms at 1 row, 6.4 ms at its floor): fuse it into `k_up`/`k_down` as glm53 did with FP8,
  or at least overlap the down GEMV; A3e measured the standalone GEMVs only.
- The INT4 GEMVs below 93 % at 1 and 16 rows (experts.md), the 1-row `l2pf` repeat, a step-level kBF16T vs
  `--rowmajor` A/B (attention.md measured it per kernel: 98–100 % vs 92–95 %).
