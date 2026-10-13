# The routed experts (milestone G3, expert half)

What `engine/experts/` computes for one rank at TP=4, the layout the converter must emit, the numerics
contract, how it was tested and what it measured. Written 2026-10-13. The math is
[model.md § 2](model.md#2-the-forward-pass); the API and every constant are in
[engine/experts/experts.h](../engine/experts/experts.h); the layout's executable specification is
[engine/experts/layout.h](../engine/experts/layout.h).

## Contents

1. [What is built](#1-what-is-built)
2. [Formats and the layout the converter emits](#2-formats-and-the-layout-the-converter-emits)
3. [Routing](#3-routing)
4. [Decode: the expert step for 1–16 rows](#4-decode-the-expert-step-for-116-rows)
5. [Prefill: the grouped expert GEMMs](#5-prefill-the-grouped-expert-gemms)
6. [Numerics contract](#6-numerics-contract)
7. [Tests](#7-tests)
8. [Measurements](#8-measurements)
9. [Experiment log](#9-experiment-log)
10. [What remains](#10-what-remains)

## 1. What is built

| File | What |
|---|---|
| `experts.h` | shapes (`kH` 6144, `kI` 512, `kE` 256, top-8, ×2.5), `MatDesc`, `SharedW`, `LayerW`, the API |
| `formats.cuh` | the decoders, host and device (`__host__ __device__`, so the CPU checks run the kernels' code): NVFP4 widening, the S3 prmt decode, FP8 e4m3, the routing sigmoid, the NVFP4 activation quantiser (host) |
| `layout.h` | the canonical tensors (checkpoint / quantiser layout), the packers into the engine layout, independent readers, `make_desc` |
| `route.cu` | `route()`: sigmoid + bias, top-8, normalise, ×2.5 |
| `decode.cu` | `moe_decode()`: `k_up` (union, gate+up fused, SwiGLU) and `k_down` (down + shared, f32 partials), PDL; `finalize()` |
| `prefill.cu` | `moe_prefill()`: bucketing, grouped gate/up and down GEMMs on bf16 tensor cores, combine; the FP4-activation (`mxf4nvf4`) variant behind `PrefillCfg::fp4`; the shared expert as a dense FP8 GEMM |
| `tests/test_experts.cu`, `tests/test_prefill.inc` | CPU checks, GPU checks against CPU references |
| `tests/bench_experts.cu`, `tests/bench_prefill.inc` | the rate benches |
| `run.sh` | the GPU steps (tests, benches), resumable, through ds41f's `ring.sh` |
| `install.sh` | builds this directory alone (-j4) and installs the binaries with the sources' hash |

Per rank: 256 experts a layer, gate and up [512, 6144] each, down [6144, 512]; the shared expert's slice
(gate, up [512, 6144], down [6144, 512], FP8 with 128×128 block scales). Every dimension is a compile-time
constant.

**Build.** `engine/CMakeLists.txt` picks up `experts/` like every subdirectory. On 2026-10-13 the engine-wide
configure failed in other directories (missing sources in `attn/`, then `comm/`), so `install.sh` builds through
a wrapper project outside the repo that holds only the engine's flags and this directory. Flags: nvcc 13.0
(V13.0.88), `-O3 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -lineinfo`, C++20; the test
links OpenMP.

## 2. Formats and the layout the converter emits

Each routed matrix (gate, up or down of one expert) has its own format, recorded in its `MatDesc`; both formats
run in one launch (the kernels branch on the descriptor, warp-uniformly).

**Canonical tensors** (what the checkpoint and the quantiser write, `layout.h`):

- **NVFP4** (`kNVFP4`): `w [N][K/2]` e2m1, low nibble = even k; `s [N][K/16]` e4m3; `s2` f32.
  W(n, k) = e2m1(nibble) · e4m3(s[n][k/16]) · s2.
- **S3, the E2M1-subset 3-bit code** (`kS3`, `quant/formats.py` levelset3): one or two subsets of four E2M1
  magnitudes per tensor (ascending, the first may be 0); codes `[N][K/16][6]`, each group of 16 a
  little-endian 48-bit word with value j at bits 3j…3j+2 (code c = sign << 2 | index); `s [N][K/16]` with
  bit 7 the group's subset and bits 6:0 its e4m3 scale; `s2` f32.
  W(n, k) = (−1)^(c>>2) · mags[s>>7][c & 3] · e4m3(s & 0x7f) · s2.

**The engine layout** (a lossless re-tiling; `pack_nvfp4`, `pack_s3` are the specification, and the
`--cpu` checks prove packer + kernel decoder = canonical reader for every value):

- A matrix [N][K] is a sequence of blocks of 8 rows × 512 k, block (n8, b) at byte
  `((n8 · K/512) + b) · block_size`, block_size 2304 (NVFP4) or 1792 (S3). A matrix starts at a multiple of
  256 bytes in the layer's pool (`MatDesc::off`).
- In a block, lane l = 4g + t (g = l >> 2, t = l & 3) holds row n = 8·n8 + g. Unit jj (0…3) is
  k = 512b + 128jj … +127; the lane's 32 values of a unit are v[16q + i] = (n, k = 512b + 128jj + 64q + 16t + i),
  q ∈ {0, 1}, i ∈ 0…15: two whole scale groups.
- **NVFP4:** unit jj's values at byte `512·jj + 16·l`: byte `8q + i/2`, low nibble = even i (the canonical
  nibble unchanged). Scales at byte `2048 + 8·l + 2·jj + q` = the canonical scale byte
  s[n][(512b + 128jj + 64q + 16t)/16].
- **S3:** three planes P0 (byte 0), P1 (512), Q (1024); the lane's word jj of each plane at
  `plane + 16·l + 4·jj`, where (`s3_pack_lane`) P0 nibble m holds bits 1:0 of v[m] and of v[m+8], P1 the same
  for v[16+m], v[24+m], and Q nibble m the sign bits of v[m], v[m+8], v[m+16], v[m+24]. Scales at
  `1536 + 8·l + 2·jj + q`, the canonical byte (subset bit included).
- **MatDesc** (64 B): `off`, `s2`, `fmt`, `tab[8]` (S3: the bf16 values of both subsets' eight codes as prmt
  tables), `nib[2]` (S3: code → e2m1 nibble per subset, for the FP4 path), padding. `make_desc(off, s2, fmt,
  nsub, mags)` builds it; the loader writes `desc[256][3]` (gate, up, down) a layer.
- **The shared expert** (`SharedW`): FP8 e4m3 in the "tile_fp8" layout `[N/8][K/128][32 lanes][32 B]`, lane
  (g, t) row 8·n8 + g, k = 128j + 64q + 16t + i at byte 16q + i (`pack_fp8`); scales f32 row-major
  `[N/128][K/128]` (`weight_scale_inv`).

Sizes a layer a rank: NVFP4 5.31 MB an expert (1.36 GB for 256), S3 4.13 MB (1.06 GB), shared 9.4 MB.

## 3. Routing

`route(logits [M][256] f32, bias [256], M, idx [M][8], w [M][8])`, one warp a row, eight experts a lane:

- s = sigmoid(logit) in f32 as `1 / (1 + exp(−x))` with a deterministic f32 exp (degree-7 Horner with fmaf,
  `ldexpf`), correctly-rounded division and add (`__fdiv_rn`, `__fadd_rn`); the host reference runs the same
  function.
- key = s + bias; the top 8 of key by eight rounds of a warp argmax, ties to the lower expert id; the picks
  come out in descending key order (the reference's `sorted=True`).
- sum of the 8 picked s in the fixed tree ((p0+p4)+(p2+p6)) + ((p1+p5)+(p3+p7)); w = s / (sum + 1e-20) · 2.5.

The test matches the CPU reference bit for bit on random rows and on rows with many tied logits.

## 4. Decode: the expert step for 1–16 rows

`moe_decode(L, x [M][6144] bf16, idx, w, M, ws, yr, ys, stream, cfg)`: two kernels with programmatic dependent
launch; `yr` is the routed partial, `ys` the shared expert's, both f32 [M][6144]; `finalize()` makes
bf16(bf16(yr) + bf16(ys)) after the all-reduce.

**`k_up`** — a persistent grid (one wave) over tasks in a fixed order. The shared expert's tasks come first:
they need nothing from the router, so the first wave loads their FP8 weights before `griddepcontrol.wait`.
Then every expert of the union (the rows' picks, each read once, in expert-id order; built in shared memory
from a bitmap and a prefix sum), its gate and up strips. A task is one CTA. WK warps split K (12 blocks of 512).
Each warp streams its blocks two ahead with streaming loads and decodes them in registers into bf16 B
fragments of m16n8k16 MMAs; the A rows are the step's rows (rows 8–15 in the upper half from 9 rows on).

The partials meet in shared memory. One warp a strip pair applies s2 (and NVFP4's 2^30), rounds gate and
up to bf16, applies SwiGLU, rounds h to bf16 and writes it to the slot's workspace. Tasks past the first wave
are claimed from a counter (`dyn`), which the last CTA resets.

**`k_down`** — a warp owns NT output strips (8 channels each). First the shared expert's down for them: its
weights are loaded before the PDL wait, and the result goes into `ys`. Then every slot of the union in expert-id
order, P slots in flight: acc = fmaf(s2 · fmt_k · w_row, c, acc) into `yr`, with routing weight 0 for rows that
did not pick the expert. A row's bits therefore do not depend on the other rows (tested: 16 rows together
equal each row alone).

**Variants** (`DecodeCfg`): up {nt1 wk6 p2 (default), nt1 wk4 p2, nt1 wk4 p1, nt2 wk4 p1, nt1 wk2 p2,
nt2 wk2 p1}; down {nt2 p1 (default), nt1 p2, nt1 p3, nt2 p2, nt1 p4}.

- All down variants, and the up variants with the same K split, give identical bits (tested).
- A different WK changes the order of the K sums. The default was chosen by the sweep in § 8; it is best or
  within noise of the best at every row count.

## 5. Prefill: the grouped expert GEMMs

`moe_prefill(L, x [T][6144], idx, w, T ≤ 8192, ws, yr, ys, stream, cfg)`, outputs as decode.

1. **Bucketing** (`k_count`, `k_scan`, `k_scatter`): counts per expert; offsets and a tile list in
   expert order, 128 picks a tile; each pick gets a position through an atomic per expert (`perm[pos] = row`).
   Each row's eight picks are kept sorted by expert id (`rpos`, `re`, `rw`) for the combine. The atomics change
   only where a pick sits in its bucket, which no row's result depends on: tested bit-identical across runs and
   across chunk composition.
2. **Gate/up** (`k_pup`): a CTA is a tile (128 picks of one expert) × 64 channels of gate and of up. Eight
   warps are arranged 2 (rows) × 4 (channels); a warp tile is 64 rows × 2 gate + 2 up strips, so SwiGLU runs in
   registers. Each k-step of 64:
   - A (the gathered x rows) by 16-byte `cp.async.ca` into a 3-stage ring, chunks XOR-swizzled by row, read with LDS.128 (conflict-free).
   - B: each warp decodes two of the 16 strips with the decode kernels' functions (so the same exact bf16
     values) from raw blocks loaded a block (512 k) ahead into registers, into a 2-stage shared buffer.
   - bf16 `mma.sync` m16n8k16, f32 accumulation.
   - A warp whose 64 rows all lie past the tile's end skips its MMAs.

   The epilogue is decode's: ×s2·fmt_k, bf16 gate and up, SwiGLU, bf16 h [position][512].
3. **Down** (`k_pdown_r`): a CTA is a tile × 4 column blocks of 128 outputs, as one pipeline of 32 k-steps
   (K = 512). The next blocks' weights load during the current one. Each pick's raw sums c go to `yp`
   [position][columns of the pass] in f32. At T = 2048+ the columns run in 2 or 4 passes, so that `yp` stays
   ≤ 420 MB.
4. **Combine** (`k_combine`): per row and column, its picks in expert-id order, acc = fmaf(s2 · fmt_k · w, c,
   acc). This is the decode step's arithmetic, so a row's routed output equals decode's up to the order of the K
   sums inside each pick (tested).
5. **Shared expert** (`k_pup<true>`, `k_pdown<true>`): dense over the chunk's rows, FP8 → bf16 tiles, each 128-k
   unit summed apart and added × its block scale (as decode).

**The FP4-activation path** (`PrefillCfg::fp4`, with `in_scale_gu` and `in_scale_dn` the layer's static
input scales). This path changes outputs, which is experiment 4's question.

- `k_quant` makes x and then h NVFP4 per 16, in IEEE f32 steps that the host function
  `nvfp4_quant16_host` repeats bit for bit:
  - sb = e4m3_rn_sat(amax / (6·S));
  - d = e4m3(sb)·S;
  - q = e2m1_rn_sat(x / d).
- `k_pup_f4` and `k_pdown_f4` run `mma.sync … kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (e2m1 × e2m1,
  ue4m3 scales), with k-steps of 128.
- The weights go to shared memory as natural-order nibbles plus scale bytes:
  - NVFP4: the bytes as stored.
  - S3: the code → nibble table, by prmt; the subset bit is dropped from the scale.
- The epilogues use alpha = S·s2.
- The shared expert stays on the bf16 path (its FP8 block scales cannot ride this MMA).

## 6. Numerics contract

The reference is `ref/glmref.py` (moe): g = bf16(s2·Σ x·W′), u likewise; h = bf16(bf16(silu(g))·u);
y_e = bf16(down(h)); out = bf16(Σ_e w_e · y_e in f32, in expert-id order) + shared, a bf16 add.

| Point | Engine |
|---|---|
| weights | decoded exactly: e2m1 × e4m3 (and the subset's magnitude) is exact in bf16. NVFP4 values carry 2^−30 (the ×2^96 scale trick of the widening), which the epilogues multiply back exactly (`fmt_k` = 2^30) |
| gate, up | f32 sums of exact bf16 products, ×s2·fmt_k, rounded to bf16, as the reference |
| h | bf16(bf16(silu(g)) · u), `silu = g / (1 + expf(−g))` |
| down | f32 sums; each expert's contribution added as fmaf(s2 · fmt_k · w, c, acc) in expert-id order |
| shared | per 128-k unit an f32 sum, then acc = fmaf(scale, unit, acc) |
| output | yr and ys are f32 partials of this rank; after the all-reduce, `finalize` makes bf16(bf16(yr) + bf16(ys)) |

**Deviation from the reference.** At TP=4 a rank holds the partial of each expert's down over its 512 of the
2048 intermediate channels, so it cannot round each expert's full y to bf16 before the weighted sum, as the
reference does. The engine sums the experts' partials in f32 and rounds once after the all-reduce. Rounding the
partials to bf16 instead would add an error the reference does not have, without reproducing its rounding.

- **Summation order.** A K split (WK) fixes the order of the K sums. Decode is batch-invariant (a row's bits do
  not depend on the other rows) and deterministic. Prefill equals decode up to the order of the K sums inside
  each pick. It is deterministic and independent of the chunk's other rows.
- **FP4 path.** It is not output-preserving. The activations are quantised. The MMA sums each k64 exactly and
  rounds toward zero to f32; the k-steps are then summed in f32 in order. Measured in § 7: 17 % relative RMS
  difference from the bf16 path on random data.

## 7. Tests

`test_experts` (GPU, through `run.sh test`); `test_experts --cpu` runs only the host checks. All pass as of the
last run (r9, § 9).

| Check | Result |
|---|---|
| CPU: NVFP4 widen, S3 pack + prmt decode (both subsets), FP8 widen (254 values), S3 code → nibble table, packers vs canonical readers (NVFP4, S3, FP8), routing reference (order, ties, sum) | 0 mismatches |
| GPU dequantisation, NVFP4 / S3 / S3 two subsets, both shapes | 0 of 3,145,728 values differ from bf16(value × scale) |
| route, random and many-tied rows | ids and weight bits identical to the reference |
| decode h, M = 1, 3, 5, 8, 9, 16 (the shared expert included) | ≥ 99.9 % bit-exact against bf16 of the double sums; every value inside the range that f32 sums of the same terms can give (bound 2e−5 · Σ\|terms\| on gate and up, each end rounded as the kernel rounds) |
| decode routed / shared out vs double from the GPU's h | ≤ 3.3e−7 / 8.4e−8 of Σ\|terms\| (tolerance 1e−4) |
| decode batch invariance (16 rows vs each alone), determinism | 0 differ |
| decode variants | same K split: bit-identical; other K splits: h inside the f32-sum range |
| prefill T = 1, 37, 300, 2500 (2 down passes): h of sampled rows (8 picks + shared) | ≥ 99.7 % bit-exact, all inside the range |
| prefill routed / shared out vs double from the GPU's h | ≤ 3.0e−7 / 8.1e−8 of Σ\|terms\| |
| prefill vs decode, every row (T ≤ 300) | about half the outputs bit-identical; the rest differ by h's rounding flips (≤ 1.7e−3 of the row's largest output) |
| prefill deterministic (T = 2500 twice); chunk invariance (300 rows alone vs inside 2500) | 0 differ |
| FP4: x and h quantised (115,200 and 76,800 groups) vs the host quantiser | 0 differ |
| FP4: h from the dequantised x | all inside the f32-sum range (24,576 of 24,576 bit-exact) |
| FP4: routed out vs double from the GPU's quantised h | 4.5e−8 of Σ\|terms\| |
| FP4: shared unchanged, deterministic | 0 differ |

Test data: 10 stored experts in five format patterns (all NVFP4; S3 with one subset; S3 with two; S3 gate/up +
NVFP4 down; NVFP4 gate/up + S3 down) mapped over the 256 ids. The subsets are {0.5, 2, 4, 6} and {0, 1, 3, 6}.
Random scales; x random bf16; picks with six hot experts (prefill).

## 8. Measurements

**Conditions.** One GB10 (rank 0's), through `ring.sh` with production paused (the job's MEM line: 8–9 GiB
used). Clocks unpinned: 2.18 GHz under load, 3003 MHz memory. The streaming-read probe measured
234–237 GB/s in every run. Builds and source hashes are listed in § 9.

**Decode** (the default variant nt1 wk6 p2 dyn / nt2 p1; the same decode code in three runs: r3 on bin-3,
sources `91b677a77cde4477`; r9 and r10 on bin-8, `20f0734bb91e2377`).

- The step is route + `k_up` + `k_down`, the shared expert included.
- Each figure is the median of 7 replays of a graph of 64 steps.
- A pool of ≥ 2.2 GB is cycled, so L2 holds no step's weights.
- Bytes = the union's weights and scales + the shared expert.
- "6c" = 6 rows over 28 experts (experiment 1's case).

Cells give the range over the three runs as % of 235 GB/s (run-to-run spread is 2–6 points; the streaming
probe gave 230–237 GB/s in the same windows).

| Rows | union | NVFP4 | S3 | mix (S3 gate/up + NVFP4 down) | ms a token, 75 layers (r3): NVFP4 / S3 / mix |
|---:|---:|---|---|---|---|
| 1 | 8 | 93.3–99.3 % | 92.2–95.9 % | 91.3–96.1 % | 16.70 / 14.13 / 15.15 |
| 2 | 15.8 | 97.1–99.8 % | 96.0–98.7 % | 90.7–99.5 % | 29.84 / 24.15 / 25.94 |
| 4 | 30.5 | 95.2–100.3 % | 96.7–98.9 % | 97.0–99.8 % | 54.60 / 43.73 / 47.17 |
| 6 | 44.4 | 97.7–100.3 % | 96.9–99.3 % | 96.5–99.8 % | 78.04 / 61.98 / 67.25 |
| 6c | 28 | 97.1–99.4 % | 96.4–98.3 % | 95.3–99.1 % | 50.74 / 40.61 / 43.82 |
| 8 | 57.9 | 96.9–100.4 % | 89.8–99.4 % | 97.1–99.9 % | 100.8 / 79.81 / 86.69 |
| 16 | 102.4 | 95.1–100.8 % | 95.2–98.5 % | 96.7–98.7 % | 175.1 / 140.0 / 152.7 |

Against the targets:

- **≥ 90 % at 16 rows:** met in every run.
- **≥ 95 % (NVFP4) and ≥ 94 % (S3) at 6 rows:** met in every run.
- **1 row:** NVFP4 met it in 2 of 3 runs (93.3 % in r9); S3 in 1 of 3 (95.9, 93.9, 92.2 %).
- **Per token** (one row), the 75 MoE layers' experts take 16.7–17.8 ms in NVFP4, 14.1–14.7 ms in S3 and
  15.2–16.0 ms mixed.
- **The spread** is larger than the gap to the target, so the 1-row figures need pinned clocks and repeats.
  At 1 row the step is ~200 µs, and fixed costs weigh most: the route kernel (~6 µs in the step), the PDL
  handover, and the tail of the up wave.

| Parts at 1 row (µs) | NVFP4 | S3 | mix |
|---|---:|---:|---:|
| step (route + up + down) | 222.6 | 188.4 | 201.9 |
| without the router | 216.9 | 182.2 | 193.6 |
| up alone / down alone | 149.2 / 74.7 | 124.1 / 59.6 | 123.7 / 72.9 |

The router costs 4.7 µs alone and ~6 µs in the step. Under PDL, up and down overlap: their sum exceeds the
step by ~7 µs.

**Prefill** (r9: bin-8, sources `20f0734bb91e2377`). One layer of 256 distinct experts in the format.

- The chunk's rows pick 8 distinct experts uniformly at random (balanced buckets, ~T/32 picks an expert).
- Each figure is the median of 5 replays of a 3-call graph.
- TFLOPS count the useful MACs × 2: routed T × 8 × 3 × 512 × 6144, shared T × 3 × 512 × 6144.
- "% of 107" is against the measured bf16 `mma.sync` peak.

| Path, format | 1k rows: ms, TFLOPS (% of 107) | 4k | 8k | ms for 1k rows through 75 layers (8k chunks) |
|---|---|---|---|---|
| bf16, NVFP4 | 10.2, 17.0 (15.9 %) | 23.0, 30.2 (28.2 %) | 44.0, 31.6 (29.6 %) | 412 |
| bf16, S3 | 9.6, 18.2 (17.0 %) | 22.5, 30.9 (28.9 %) | 42.8, 32.6 (30.4 %) | 401 |
| bf16, mix | 10.2, 17.0 (15.9 %) | 23.4, 29.7 (27.7 %) | 43.4, 32.1 (30.0 %) | 407 |
| FP4 activations, NVFP4 | 10.0, 17.4 | 20.8, 33.5 | 37.7, 36.9 | 353 |
| FP4 activations, S3 | 8.8, 19.7 | 19.8, 35.2 | 36.8, 37.8 | 345 |

Phases at 8k rows, NVFP4 (ms; TFLOPS of the phase's own MACs):

| Phase | bf16 path | FP4 path |
|---|---|---|
| bucketing | 0.03 | 0.03 |
| gate/up GEMM (FP4: with both quantisations) | 16.8 (49.0) | 10.9 (75.4) |
| down GEMM (writes the f32 per-pick sums, 1.6 GB) | 12.6 (32.8) | 12.7 (32.5) |
| combine (reads them back, writes yr) | 8.1 | 8.2 |
| shared expert (FP8, both paths bf16) | 5.0 (30.9) | 5.1 |

The gate/up kernel's time with parts switched off (bench diagnostics, r8, 8k rows, NVFP4, ms):

| Configuration | ms |
|---|---:|
| all | 16.6 |
| no weights | 13.2 |
| no MMAs | 15.2 |
| no A loads | 14.3 |
| only A loads | 9.9 |
| only MMAs | 10.6 |
| only weights | 6.0 |

The kernel is not bound by any one part. Its MMA loop alone reaches 77 TFLOPS. The gathered A (x rows, each
tile's 1.5 MB read by all 8 channel-block CTAs, ~7.4 GB at 8k from L2) alone takes 10 ms.

**The ≥ 60 % target (64 TFLOPS) is not met: 29–30 % at 8k rows.** Two separate reasons:

1. **The per-pick f32 sums are a memory floor.** Keeping decode's numerics (each pick's down sum in f32, added
   in expert-id order; no bf16 rounding of the rank's partial) means 8k rows write and read back
   65,536 × 6144 × 4 B = 1.6 GB twice, plus 0.2 GB of yr: 3.4 GB, 14.5 ms at 235 GB/s.
   - The GEMMs alone at the bf16 peak take 13.0 ms (1.39 PFLOP).
   - Even if the down GEMM fully hid its writes, the combine's 7.7 ms cannot hide behind anything in this
     layer. The ceiling is therefore ~67 TFLOPS (63 %), and only with GEMMs at 100 % of peak.
   - Storing the per-pick sums in bf16 halves the floor but changes outputs (not done: exact-output rule).
2. **The GEMM kernels themselves reach 30–47 % of peak.**
   - Gate/up runs at 49–52 TFLOPS and down at 33.
   - Shared expert: 31 TFLOPS (64-row tiles, per-unit scale accumulators).
   - At 1k rows, gate/up is bound by the weights (each tile decodes the whole expert slice; 0.9 GB of gate/up
     weights a layer is 3.9 ms at 235 GB/s), and partial tiles waste MMAs.

FP4 activations (`mxf4nvf4`, peak 429 TFLOPS): the gate/up GEMM reaches 76–83 TFLOPS of useful work, 1.6–1.7×
the bf16 kernel. The down GEMM (bound by its f32 writes) and the combine do not change, so the whole layer
speeds up 1.2× (36.9–37.8 TFLOPS of useful work at 8k).


## 9. Experiment log

All runs: one GPU (rank 0), `run.sh` steps through `ring.sh`, results in `~/dsgb10-data/glm53-experts/rN.tsv`
(not in the repo), binaries in `bin-N` with their `SOURCES` hash.

| Run | Sources | What | Outcome |
|---|---|---|---|
| r1 | bin-1 | test, decode sweeps (first kernels: a separate down wave, a double-precision exp in the router) | the test step ran nothing: `test_experts` rejected `--out` (fixed). Decode 86.7–97.8 %, NVFP4 94.6 % at 1 row, S3 86.7 % at 1 row; large spread between variants |
| r2 | `ceb7a54fbf426074` | test, decode: the f32 sigmoid, shared down fused into `k_down` with weights before the PDL wait, dynamic task claiming, per-part timings | decode 96–101 % with wk6; the tests flagged two things in the checks, not the kernels: an h check by "2 bf16 ulp" fails on cancelling sums (10 of ~400k values), and other-K-split variants were compared at a relative tolerance. Both replaced by the f32-sum range check |
| r3 | `91b677a77cde4477` | test (all pass), first prefill (2-stage, MMAs branching per m16 tile), decode with the new defaults | the decode table above; prefill 27–29 TFLOPS at 8k (up 45, down 25, combine 8.1 ms) |
| r4 | `e1c9c7da146078b2` | 3-stage A ring; tighter prefill checks from the workspace (`prefill_view`); the FP4 path and its tests | all pass; the 3rd stage changed nothing measurable (up 45 → 45 TFLOPS); FP4 up 76–84 TFLOPS |
| r5 | `9a3ff38b77831097` | branch-free MMA steps (warp-level skip, fragments loaded ahead), the down kernel over 4 column blocks with weights prefetched | all pass; down 25.8 → 32.3 TFLOPS, up 45 → 47, 8k layer 49.3 → 45.4 ms |
| r6 | bin-6 | the gate/up diagnostics (parts switched off) | only-A-loads 10.4 ms, only-MMAs 10.3 ms, only-weights 5.9 ms of 17.4 |
| r7 | `aeece0beb106d950` | A tile as 16-byte `cp.async.cg` chunks, read with LDS.128 (half the instructions, conflict-free) | **slower**: up 17.4 → 21.8 ms; A loads alone 10.4 → 20.5 ms. The cause was `.cg` (L2 only) instead of `.ca` |
| r8 | `20f0734bb91e2377` | the same with `cp.async.ca` | up 16.6 ms (49.7 TFLOPS); A loads alone 9.9 ms |
| r9 | `20f0734bb91e2377` | final: tests (all pass), prefill and FP4 tables, decode sweeps | the prefill table; decode 2–6 points under r3 with unchanged decode code |
| r10 | `20f0734bb91e2377` | decode sweeps again (spread check) | within the r3–r9 range; S3 at 1 row 92.2 % |

## 10. What remains

- **Prefill rate.** At 8k rows the bf16 prefill is at 30 % of the bf16 peak, against a 60 % target. The
  bounds are measured above.
  - Next for the GEMMs: cut the A re-reads (wider CTAs in N, or warp-specialised producer warps); 128-row
    tiles for the shared expert; overlap the combine with the shared expert on a second stream.
  - Beyond ~63 % a different numerics choice would be needed: bf16 per-pick sums, which change outputs.
- **Decode at 1 row** varies by run (S3 92–96 %). It needs pinned-clock repeats and a look at the fixed costs
  (route, the PDL handover, the up wave's tail).
- **FP4 path.** It is not output-preserving: on random data its outputs differ from the bf16 path by 17 %
  relative RMS, which is for experiment 4 to judge on real activations with calibrated input scales. Its down
  GEMM is unchanged in speed (f32-write bound) and uses 2 A stages.
- **Integration.** The loader that fills the pool and `desc` from the converter's output (`layout.h`'s packers
  are its specification). Wiring into the layer loop (the router's logits from the gate GEMV, yr/ys into the
  all-reduce, `finalize`). The engine-wide CMake configure, once `attn/` and `comm/` configure again.
- **Real weights.** All tests use synthetic tensors; a check against `ref/glmref.py` on a real layer's experts
  is still to do.
