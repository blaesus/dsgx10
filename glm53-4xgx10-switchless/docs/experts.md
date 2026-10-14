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
11. [2026-10-14: prefill tuning](#11-2026-10-14-prefill-tuning)

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

**Fused routing and posted outputs** (2026-10-14, the decode-gap pass, [engine.md](engine.md#closing-the-decode-gap-2026-10-14)):
- `DecodeCfg::logits`, `bias`: both kernels route the rows themselves from the router GEMV's logits (a warp a
  row, `route_warp`, the body `k_route` now calls too: the same bits), so no `route()` launch sits between the
  router and the experts. `k_down` reads the logits before its PDL wait, as it read `idx` before: they are
  complete once `k_up`'s CTAs have triggered (each past its own wait on the router GEMV).
- `DecodeCfg::post` (a ring channel): `k_down` writes `yr` and `ys` into the send buffer of the channel's next
  collective (M rows of `yr`, then M of `ys`: `allreduce2_add_norm`'s layout) and posts it from its epilogue
  (`comm::post_grid`, every block); the all-reduce kernel is then only the consumer half
  (`comm::glm::allreduce2_add_norm_posted`). The same values travel, so the same bits.

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
2. **Gate/up** (`k_pup`; since 2026-10-14 the default is its warp-specialised form `k_pup_ws`, § 11): a CTA is a tile (128 picks of one expert) × 64 channels of gate and of up. Eight
   warps are arranged 2 (rows) × 4 (channels); a warp tile is 64 rows × 2 gate + 2 up strips, so SwiGLU runs in
   registers. Each k-step of 64:
   - A (the gathered x rows) by 16-byte `cp.async.ca` into a 3-stage ring, chunks XOR-swizzled by row, read with LDS.128 (conflict-free).
   - B: each warp decodes two of the 16 strips with the decode kernels' functions (so the same exact bf16
     values) from raw blocks loaded a block (512 k) ahead into registers, into a 2-stage shared buffer.
   - bf16 `mma.sync` m16n8k16, f32 accumulation.
   - A warp whose 64 rows all lie past the tile's end skips its MMAs.

   The epilogue is decode's: ×s2·fmt_k, bf16 gate and up, SwiGLU, bf16 h [position][512].
3. **Down** (`k_pdown_r`; since 2026-10-14 the default is the chained down `k_pdown_c`, which replaces this
   step and the combine, § 11): a CTA is a tile × 4 column blocks of 128 outputs, as one pipeline of 32 k-steps
   (K = 512). The next blocks' weights load during the current one. Each pick's raw sums c go to `yp`
   [position][columns of the pass] in f32. At T = 2048+ the columns run in 2 or 4 passes, so that `yp` stays
   ≤ 420 MB.
4. **Combine** (`k_combine`): per row and column, its picks in expert-id order, acc = fmaf(s2 · fmt_k · w, c,
   acc). This is the decode step's arithmetic, so a row's routed output equals decode's up to the order of the K
   sums inside each pick (tested).
5. **Shared expert** (`k_pup<true>`, since 2026-10-14 `k_pup_ws<true>`; `k_pdown<true>`): dense over the chunk's rows, FP8 → bf16 tiles, each 128-k
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
last run (2026-10-14 r8: 90 checks, the 78 above plus 12 new; § 11). Since the defaults changed, the older
prefill checks (ranges, determinism, chunk invariance) now run on the new kernels.

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
| (2026-10-14) prefill T = 1, 37, 300, 2500: h of every pick, warp-specialised gate/up with 64 × 64 warps (`up = 2`) vs `k_pup` | 0 differ |
| (2026-10-14) prefill T = 1, 37, 300, 2500: every yr and ys value, the defaults (`k_pup_ws`, chained down) vs the kernels of 2026-10-13 (`k_pup`, `k_pdown_r` + `k_combine`) | 0 differ |
| (2026-10-14) the same with `up = 2`, `down = 2` (warp-specialised chained down), 256-column passes claimed 3 round-robin, the shared expert on a side stream | 0 differ |

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

- **Prefill rate.** At 8k rows the bf16 prefill is at 36–37 % of the bf16 peak after the 2026-10-14 tuning
  (§ 11), against a 60 % target. The bounds are measured in § 11.
  - The chained down (16 ms at 8k, 25 TFLOPS) is now the largest phase: its ordered read-add-write of yr
    and the waits between a row's picks. Next: fewer, larger column passes with yr held in shared memory
    per row block (needs tiles grouped by row, not by expert), or the down's MMAs at 128-row × 256-column
    CTAs to halve the yr traffic per MAC.
  - The gate/up kernel is 3.3 ms over its MMA-only time; the shared expert (4.6 ms, 33 TFLOPS) still uses
    64-row tiles.
  - A split K or bf16 per-pick sums would go further but change outputs (not done: exact-output rule).
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

## 11. 2026-10-14: prefill tuning

**Goal:** the bf16 routed-expert prefill from 30 % towards 60 % of the 107 TFLOPS bf16 peak at chunks of
≥ 4096 rows. The rule was exact output: every output bit-identical to the 2026-10-13 kernels. bf16
intermediates, a split K and FP4 activations were not allowed. Decode is untouched.

**Result:** 1.23× at 8k rows, 35.5–37.4 % of peak. Every yr and ys value is bit-identical to the old path
(tested at T = 1, 37, 300, 2500). The 60 % target is not met; the reasons are below.

### What changed

`PrefillCfg` gains `up`, `down`, `yr_kb`, `chain_q` and `side` (experts.h). The kernels of 2026-10-13 stay
selectable as `up = 0`, `down = 0`.

1. **The chained down** (`k_pdown_c`, `down = 1`, the default). It replaces the down GEMM plus combine.
   - The old down wrote each pick's f32 down sums to a 1.6 GB buffer; the combine read them back.
   - Now each tile's sums are added into yr as soon as a 128-column block is done:
     `yr = fmaf(s2 · fmt_k · w, c, yr)`, with the row's rank-0 pick starting from 0.
   - Each pick knows its rank among its row's eight picks in expert-id order (`prank`, `pw`, filled by
     `k_scatter`). Its add waits until the rank before it has written. The waits use tickets per
     (row, 32 columns) in the workspace:
     - a warp polls them with relaxed loads and nanosleep;
     - it releases with a fence, then stores rank + 1 (0 after rank 7, so the tickets reset themselves);
     - a spin bound traps rather than hangs.
   - The adds and their order are the combine's, so the output bits are the same. No summation order
     changes.
   - The CTAs are persistent and claim (pass, tile) tasks from a counter.
     - Passes are column slices whose yr fits an L2 budget: `yr_kb`, default 16 MB, divided by `chain_q`.
       The pass width is the widest of 256 × {24, 12, 8, 6, 4, 3, 2} columns that fits, at least 256.
     - `chain_q` passes are claimed round-robin, tiles in expert order within each pass.
     - A task is claimed only after every task it may wait on, so the waits cannot deadlock.
   - yr is read with `ld.global.cg` and an L2 evict_last hint, a whole m16 tile's loads in flight together.
2. **Warp-specialised gate/up** (`k_pup_ws`, `up = 1`, the default; also the shared expert's gate/up).
   - 512 threads: 8 consumer warps run the same MMAs in the same k order as `k_pup`. 8 producer warps load
     A and decode B into a 3-stage ring, handed over with named barriers (full / empty per stage).
   - The producer order: decode B(i), wait for A(i), arrive full, then wait empty and load A(i + 2).
3. **Coalesced A loads** (`load_a_co`). Eight threads cover one row's 128 bytes, so a warp instruction
   covers 4 whole lines instead of 16 part-lines. Used by both new kernels. Alone, the A loads go from
   9.9 to 5.7 ms at 8k.

**Variants kept, not defaults** (tested bit-identical):
- `up = 2`: 4 consumer warps of 64 × 64.
- `down = 2`: the chained down, warp-specialised (`k_pdown_cw`).
- `side`: the shared expert forked to a second stream.

### Before / after

r8 (bin-10, sources `84aeb0c6fe455ba2`; the commit adds comments only). One GB10 (rank 0's), conditions as
in § 8. The old and new paths were measured in the same job, A then B and then again in reverse order
(B, A); the cells give both, in run order.

| Format, rows | old (up 0, down 0): ms, TFLOPS | new (up 1, down 1): ms, TFLOPS | % of 107: old → new |
|---|---|---|---|
| NVFP4, 1k | 10.07 / 10.13, 17.3 / 17.2 | 8.56 / 8.44, 20.3 / 20.6 | 16.1 → 19.0–19.3 |
| NVFP4, 4k | 23.27 / 23.58, 29.9 / 29.5 | 19.01 / 19.09, 36.6 / 36.5 | 27.6–27.9 → 34.1–34.2 |
| NVFP4, 8k | 44.90 / 44.10, 31.0 / 31.6 | 36.62 / 36.48, 38.0 / 38.1 | 29.0–29.5 → 35.5–35.6 |
| S3, 1k | 9.61 / 9.68, 18.1 / 18.0 | 7.80 / 7.91, 22.3 / 22.0 | 16.8–16.9 → 20.5–20.8 |
| S3, 4k | 22.51 / 22.58, 30.9 / 30.8 | 18.37 / 18.11, 37.9 / 38.4 | 28.8–28.9 → 35.4–35.9 |
| S3, 8k | 43.10 / 42.96, 32.3 / 32.4 | 35.10 / 34.76, 39.6 / 40.0 | 30.2–30.3 → 37.1–37.4 |
| mix, 1k | 10.12 / 9.97, 17.2 / 17.4 | 7.91 / 7.76, 22.0 / 22.4 | 16.1–16.3 → 20.6–21.0 |
| mix, 4k | 23.09 / 23.06, 30.1 / 30.2 | 18.53 / 18.25, 37.5 / 38.1 | 28.2 → 35.1–35.6 |
| mix, 8k | 43.25 / 43.10, 32.2 / 32.3 | 35.35 / 35.25, 39.4 / 39.5 | 30.1–30.2 → 36.8–36.9 |

Through 75 layers at 8k-row chunks, NVFP4, the routed and shared experts take 417 → 343 ms per 1k rows (means of
the two orders), before the attention side.

The phases at 8k, NVFP4 (r8, first order), in ms with the phase's TFLOPS:

| Phase | old | new |
|---|---|---|
| bucketing | 0.05 | 0.05 |
| gate/up | 17.0 (48.6) | 14.2 (58.0) |
| down | 12.9 (31.9) | 16.2 (25.4), adds into yr included |
| combine | 8.3 | — |
| shared expert | 5.0 (30.7) | 4.7 (32.6) |

The two changes add up: the A/B grid also measured the mixed pairs. At 8k in NVFP4, the chained down alone
gives 44.9 → 39.8 ms and the WS gate/up alone 44.9 → 41.5 ms.

### Why not 60 %

60 % is 64 TFLOPS, 21.7 ms a layer at 8k. The new path takes 36.5 ms:

- **Gate/up**, 14.2 ms. With parts switched off (r6): MMAs only 10.9 ms, A loads only 5.7, weights
  only 5.9. It already runs within 3.3 ms of its own MMA loop, which reaches 75 TFLOPS on `mma.sync`.
- **Chained down**, 16.2 ms, against 3.9 ms of MACs at peak. It is bound by the ordered add into yr: each
  pick's 6144 columns are read, added and written in the row's expert order, 3.2 GB of L2 traffic at 8k.
  Its tiles also wait on other tiles. Narrower passes (more L2 hits) mean more waits and more re-reads of
  h; wider ones spill yr out of L2 (see `yr_kb` and `chain_q` below).
- **Shared expert**, 4.6 ms at 33 TFLOPS.

Even with gate/up at its MMA floor and the down at the old down's 12.9 ms without the combine, a layer
would take ~29 ms (48 TFLOPS, 45 %). Going further needs a down that avoids the ordered f32 read-add-write
per pick, or numerics that change outputs (§ 10).

### Runs

Results are in `~/dsgb10-data/glm53-experts-pf/rN.tsv` and binaries in `bin-N` (not in the repo). Each run
is one `ring.sh` job on rank 0's GPU. All runs are NVFP4 unless noted.

| Run | Sources | What | Outcome |
|---|---|---|---|
| r1 | `b9aaf884` | chained down, first form: one ticket per (row, pass) | 1k: down 84 ms (a pick waits for its row's previous pick over the whole pass). Chunk invariance failed: the workspace offsets moved with T (fixed: tickets at fixed offsets) |
| r2 | `1fb6a81d` | a ticket per (row, 128-column block); the side stream | all checks pass; 8k down 66 ms. Side stream: no gain |
| r3 | `1336999b` | WS gate/up (first form); `chain_q` | WS up 17.5 vs 17.0 ms: no gain. `chain_q` 4: 8k down 76 → 29 ms |
| r4 | `8b32c32f` | a ticket per (row, 32 columns), so each warp waits only for its own rows | 8k down 50 ms at cq 1, 25 at cq 4; the yr loads were found serialized behind their adds |
| r5 | `3f6137db` | producers reordered (decode B before waiting on A); an L2 prefetch of the next A rows | up 16.5 ms. Chained down 18.3 ms vs 21.5 for the old down plus combine. The prefetch hurt: A loads alone 9.9 → 14.6 ms (removed) |
| r6 | `eda26751` | coalesced A loads; yr loads batched per m16 tile | all checks pass; the best: 8k 36.4 ms, 38.3 TFLOPS. yrkb 65536: worse. up 2 (64 × 64 warps): 15.2 ms, worse. Diagnostics above |
| r7 | `a53d3d84` | sweep: down 1 / 2, cq 2 / 4 / 8, yrkb 16 / 32 MB, 1k–8k | all checks pass, including down 2. Best down 1, yrkb 16 MB, cq 2–4 (8k 36.3–36.6 ms). cq 8: 39.6 ms. yrkb 32 MB: 0.3–1 ms worse. Down 2: 0.4–1.5 ms worse |
| r8 | `84aeb0c6` (bin-10); `440bb14d` (bin-11) | final A/B in both orders, three formats (bin-10). bin-11: setmaxnreg 120 / 136 for producer / consumer warps | the table above; 90 checks pass on both. setmaxnreg: up 14.5 ms (no gain); down 2 16.1 vs down 1 16.3 ms (within noise). Kept off (`kWsRegP = kWsRegC = 0`): the consumers already fit in 120 registers, so they gain nothing; without the split, `k_pdown_cw` spills 64 bytes |

**Not done.** A split K for the down's or the gate/up's K sums: it changes the summation order, so it is
outside the rule.
