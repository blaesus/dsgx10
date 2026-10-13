# Findings

## Experiment 1: decode GEMVs and tensor cores for the expert format (2026-10-13)

Code, build and run: [micro/](../micro/README.md). Raw output: [micro/results/2026-10-13.tsv](../micro/results/2026-10-13.tsv).

**Conditions (measured).** One GPU window on rank 0 (one of the two ASUS Ascent GX10s), 2026-10-13
21:38:23–21:38:52 +0100, borrowed through ds41f's `ring.sh` while the service was idle (production engine
stopped for the lend; MemAvailable 113 GiB). GB10, 48 SMs, L2 24 MB, CUDA 13.0.88, binaries built with
`nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -lineinfo` from sources with sha256 prefix
`adb83c670b416e0f` (common.cuh + gemv.cu + mma.cu as they ran; afterwards only mma.cu's reporting was changed).
GPU at 611 MHz idle before, 2184 MHz / 52 °C / 25 W at the end; clocks not pinned. Streaming read in the same
window: **234.0–236.7 GB/s** (best grid each benchmark), so "% of 235" below is also % of the window's own.
Everything in this section is **measured** unless marked *estimate*.

### Routed-expert decode GEMV (per rank at TP=4, one MoE layer)

gate+up [2 × 512 × 6144] fused in one launch, down [6144 × 512], the union read once (8 experts at 1 row,
28 at 6 rows), from a 2.2 GB pool cycled so L2 holds nothing. Bytes = weights + their scales (and suh/svh
for trellis), the union's; activations and partials not counted. "step" = up + mid + down (+ the trellis
input transform), one CUDA graph, so it includes the small kernels and the gaps between launches. Best
configuration of the sweep (P = loads in flight a warp, S = K split of gate/up at 1 row); all
configurations are in the raw file.

| Format | rows | experts | P/S | up MB | up µs | GB/s | % of 235 | down MB | down µs | GB/s | % | pre / mid µs | step MB | step µs | GB/s | **% of 235** | ms a token, 75 layers |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| NVFP4 (q4), 4.5 bpw | 1 | 8 | 2/3 | 28.31 | 128.9 | 220 | 93.4 | 14.16 | 64.2 | 220 | 93.8 | – / 1.27 | 42.47 | 190.3 | 223 | **95.0** | 14.3 |
| NVFP4 | 6 | 28 | 2/1 | 99.09 | 434.4 | 228 | 97.1 | 49.55 | 218.3 | 227 | 96.6 | – / 1.73 | 148.64 | 657.5 | 226 | **96.2** | 49.3 |
| E2M1 subset (s3), 3.5 bpw | 1 | 8 | 2/2 | 22.02 | 99.9 | 220 | 93.8 | 11.01 | 52.2 | 211 | 89.7 | – / 1.07 | 33.03 | 153.8 | 215 | **91.4** | 11.5 |
| E2M1 subset | 6 | 28 | 2/1 | 77.07 | 360.2 | 214 | 91.0 | 38.54 | 177.7 | 217 | 92.3 | – / 1.75 | 115.61 | 530.0 | 218 | **92.8** | 39.8 |
| Trellis K3, 3INST (tk3), ~3.0 bpw | 1 | 8 | 8/4 | 19.09 | 127.0 | 150 | 64.0 | 9.54 | 87.7 | 109 | 46.3 | 1.76 / 2.83 | 28.63 | 221.4 | 129 | **55.0** | 16.6 |
| Trellis K3 | 6 | 28 | 8/1 | 66.81 | 429.7 | 156 | 66.2 | 33.40 | 322.9 | 103 | 44.0 | 13.69 / 2.99 | 100.21 | 776.1 | 129 | **54.9** | 58.2 |
| Trellis K4, 3INST (tk4), ~4.0 bpw | 1 | 8 | 8/4 | 25.38 | 124.4 | 204 | 86.8 | 12.69 | 85.4 | 148 | 63.2 | 1.73 / 2.86 | 38.07 | 215.3 | 177 | **75.2** | 16.1 |
| Trellis K4 | 6 | 28 | 8/1 | 88.83 | 429.6 | 207 | 88.0 | 44.41 | 315.1 | 141 | 60.0 | 13.84 / 3.15 | 133.24 | 762.0 | 175 | **74.4** | 57.1 |
| Trellis K3, MUL1 (tk3m) | 1 | 8 | 8/4 | 19.09 | 125.2 | 152 | 64.9 | 9.54 | 87.4 | 109 | 46.4 | 1.73 / 2.82 | 28.63 | 218.9 | 131 | **55.7** | 16.4 |
| Trellis K3, MUL1 | 6 | 28 | 16/1 | 66.81 | 434.7 | 154 | 65.4 | 33.40 | 314.9 | 106 | 45.1 | 13.70 / 2.99 | 100.21 | 769.2 | 130 | **55.4** | 57.7 |

The last column multiplies the step by 75 MoE layers: the routed experts' share of a decode token at that
kernel speed (the bandwidth floor would be 13.6 / 10.5 / 9.1 / 12.1 ms at 1 row for q4 / s3 / K3 / K4).

**Correctness (all passed, on the timed data, step 0 of every configuration).**
- Decode bit-exact through the kernels' own fragment mapping, expert 0's gate and down (3.1M + 3.1M weights):
  q4 = bf16(e2m1 × e4m3 × 2^-30), s3 = bf16(subset value × e4m3), trellis = exllamav3's `dq<K, cb>` ported
  literally plus `tensor_core_perm`: 0 mismatches for q4, s3, K3, K4, K3-MUL1.
- Trellis transforms: had(x ⊙ suh)/√128 → f16 bit-identical to the CPU (same butterfly order), 2.06M values at
  6 rows; the mid kernel (output had ⊙ svh, SwiGLU, bf16, ⊙ suh_down, had → f16) bit-identical, 86k values.
- GEMVs against double precision: worst |err| / Σ|terms| 2.6e-9 – 7.9e-7 over all formats, rows and K splits
  (up to 172k outputs a check); dense bf16 ≤ 5.0e-7, fp8 ≤ 7.2e-8.
- The host-only `gemv --cpu-check` (decoders, s3 packing, fp8 table, every lane's trellis fragment for K2/3/4 and
  all three codebooks against the literal port) passed before the window. The ported 3INST codebook's 65,536
  values have standard deviation 1.2437, exllamav3's `codebook_scale` (1.24371088); MUL1's 1.0003.

**Why the trellis kernels are slow (from the SASS, *analysis*).** Trellis K3 and K4 gate/up take the same time
at 6 rows (429.7 and 429.6 µs for 176M weights, ~410 G weights/s) although K4 reads 33 % more bytes, and the
1-row case runs at the same weight rate: the kernel is bound by issue and latency, not by memory. It runs ~11.4
instructions a weight per lane (`k_tr_up` K3: 1,464 SASS instructions for 16 MMAs × 8 weights a lane) against
6.0 for q4 and 4.9 for s3, plus three `SHFL` a tile, whose warp-collective wrappers (`WARPSYNC`,
`ENDCOLLECTIVE`) the compiler adds because the pipeline's guard is not provably uniform; the eight windows are
cut with 64-bit variable shifts. K3 at the floor needs 627 G weights/s (13 G a second an SM, ~5.4 a clock at
2.4 GHz). The down kernel (one CTA per 128-channel Hadamard block, 48 CTAs, a barrier an expert) is worse
(44–64 %): too few warps in flight for 3-bit tiles of 96 B. These are properties of this implementation;
exllamav3's own GEMV (funnel shifts with per-lane constants, warp-split K, register prefetch ring) was not run
here, and kindlingai's gr8 reaches 49.3 tok/s on a 3.25 bpw trellis on four GB10s with drafts (scouting § 3.1),
which says a faster trellis GEMV exists but not how close to the floor it gets.

### Attention weights (dense GEMV, a rank)

| Weights | Shape | rows | MB | µs | GB/s | % of 235 | Check (vs double) |
|---|---|---|---:|---:|---:|---:|---|
| BF16 | q_b [4096 × 2048] | 1 | 16.78 | 78.1 | 215 | 91.4 | 2.2e-7 |
| BF16 | q_b | 6 | 16.78 | 78.2 | 215 | 91.3 | 2.7e-7 |
| BF16 | o_proj [6144 × 4096] | 1 | 50.33 | 228.7 | 220 | 93.6 | 4.5e-7 |
| BF16 | o_proj | 6 | 50.33 | 229.0 | 220 | 93.5 | 5.0e-7 |
| FP8 + f32 per 128×128 | q_b | 1 | 8.39 | 37.3 | 225 | 95.8 | 7.2e-8 |
| FP8 | q_b | 6 | 8.39 | 37.3 | 225 | 95.8 | 7.2e-8 |
| FP8 | o_proj | 1 | 25.17 | 109.4 | 230 | 97.9 | 5.3e-8 |
| FP8 | o_proj | 6 | 25.17 | 109.5 | 230 | 97.8 | 5.3e-8 |

FP8 halves the attention bytes at no loss of bandwidth (e4m3 widened to bf16 exactly: bits into bf16's fields,
one multiply by 2^120; the block scale on the f32 partial of each 128-k block).

### Tensor cores

| Instruction (sm_121a) | Exactness on random operands | Rate, 8 chains (4 chains) |
|---|---|---:|
| `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` (NVFP4 × NVFP4, group 16) | layout confirmed; 8,188 of 8,192 values = round-to-nearest of the exact sum, 4 = **round toward zero** (all 6 values whose exact sum is not an f32 came out truncated) | **429.1** (427.1) TFLOPS |
| `kind::mxf4nvf4 scale_vec::2X … ue8m0` (MXFP4, group 32) | 31 of 4,096 differ, ≤ 1 ulp (rel 1.2e-7) | 429.2 (427.9) |
| `kind::mxf8f6f4 scale_vec::1X m16n8k32 e4m3.e4m3 … ue8m0` (FP8 × FP8) | 1,774 of 4,096 differ, rel ≤ 1.6e-5: **not exactly rounded** | 214.5 (213.6) |
| `kind::mxf8f6f4 … e4m3.e2m1 … ue8m0` | 28 of 4,096 differ, ≤ 1 ulp (rel 1.2e-7) | 214.5 (213.6) |
| `m16n8k16 bf16`, f32 accumulate | – | 107.0 (106.4) |
| `m16n8k16 f16`, f32 accumulate | – | 107.0 (106.3) |
| `m16n8k16 f16`, f16 accumulate | – | 107.1 (106.4) |

(32–64 trials each, a quarter with C ≠ 0, scale exponents spread ±6. The other kinds' differences were not
classified, only counted; ds41f's "bit-identical" for e4m3 × e2m1 was with C = 0 and narrower scales.)

**NVFP4 operand and scale conventions (measured, `mma --analyse` re-derives them from the dumped trials):**
- A (16 × 64, row): lane 4g + t holds rows g (regs 0, 2) and g + 8 (regs 1, 3), k = 32·(reg / 2) + 8t + i in
  nibble i, **low nibble first**; B (64 × 8, stored n-major): column g, k = 32·reg + 8t + i. Packed e2m1 as in
  the checkpoint (two a byte, low nibble = even k): the checkpoint's bytes feed the MMA without expansion (unlike
  `mxf8f6f4`, whose e2m1 sits in bits 5:2 of a byte). This is ds41f's 2X layout.
- Scales, selectors {0, 0}: a lane's A scale register holds row g + 8·(t & 1)'s four ue4m3 scales, **byte j =
  K block j** (k 16j … 16j + 15); B's, column g's. Lanes t = 2, 3 (A) and t ≠ 0 (B) are not read (changing them
  changed nothing). Accumulation is exact before one final rounding to f32 that is toward zero.
- `kind::mxf8f6f4` with `ue4m3` scales is **rejected by ptxas** (CUDA 13.0, `compile_probe.cu`: "Incorrect
  instruction type specified for mma with shape '.m16n8k32'"): NVFP4's E4M3 group scales cannot ride FP8-activation
  MMAs; FP8 activations × NVFP4 weights would need the weights re-scaled (lossy).
- The 429 TFLOPS is below nvfp4bench's 486–512 (ds41f findings); clocks here were unpinned (2.18 GHz at the end).

**Prefill ceilings per format** (*estimate*: GEMM MMA rate ÷ ~20 GFLOP of GEMMs a token a rank,
quantisation.md § 3.1, at 100 % of the measured peak; ds41f's engine reaches ~16 % of its MMA peak end to end):

| Format | Prefill path under the contract (BF16 activations) | Ceiling | With FP4 activations (experiment 4) |
|---|---|---|---|
| NVFP4 | widen to bf16 (exact) → bf16 MMA | 107 TFLOPS, ~5.3k tok/s | `mxf4nvf4` 4X, 429 TFLOPS, ~21k |
| E2M1 subset | the same (codes expand to e2m1 nibbles) | 107, ~5.3k | 429, ~21k (the codes *are* NVFP4 values) |
| Trellis (EXL3) | decode tiles to f16 → f16 MMA, activations rotated and rounded to f16 as EXL3 defines | 107, ~5.3k | none (values off the FP4 grid); FP8 activations × an E4M3-rounded codebook (exlr8's) → `mxf8f6f4`, 214, but changes outputs |

Under the BF16-activation contract all three formats have the same tensor-core ceiling (107 TFLOPS, bf16 and
f16 run at the same rate); the FP4-grid formats keep a 4× faster path if experiment 4 admits FP4 activations.

### Verdict

- **Does a trellis decode GEMV reach ≥ 95 % of the floor at 1 and 6 rows on 48 SMs? Not as measured here:**
  55 % (K3) and 74–75 % (K4) for the whole step, 64–66 % / 87–88 % for gate/up alone; the K3 kernel is
  issue-bound at ~410 G weights/s and is slower per token than NVFP4 (16.6 against 14.3 ms of experts at 1
  row) despite reading 33 % fewer bytes. Whether a better kernel closes the gap is **unknown**: it needs ~2× the
  achieved decode rate (fewer instructions a weight, no shuffles, a down kernel with more warps in flight).
- **NVFP4** reaches 95.0 % (1 row) and 96.2 % (6 rows); **the E2M1 subset** 91.4 % and 92.8 % with an untuned
  down kernel at 1 row (89.7 %); both have exact decodes and the FP4 tensor-core path.
- **Recommendation for decision 1 from experiment 1:** build on the **E2M1-subset 3-bit code mixed with NVFP4**
  (the DESIGN's fallback) as the format that meets the bandwidth bar now (s3 needs its last ~3 points of tuning;
  q4 already passes) and keeps the prefill path; keep **trellis open** as the quality candidate, gated on
  (1) a tuned trellis GEMV measured ≥ 95 % at 1 and 6 rows, which is a kernel project of its own, and
  (2) experiment 3's KL, where trellis's per-bit advantage (scouting § 5) must outweigh a decode that is today
  1.4× slower than NVFP4 per token. At the measured speeds, the subset code is the fastest format per token
  (11.5 ms of experts at 1 row, 39.8 at 6 rows) as well as the leanest that runs at the floor.
- **Unknowns.** A tuned trellis kernel's rate (and exllamav3's own GEMV on GB10, not run); s3 at 95 %;
  the subset's and trellis's KL (experiment 3); whether FP4 activations are admitted (experiment 4); the
  `mxf8f6f4` e4m3 × e4m3 accumulation rule (not exactly rounded, cause not isolated); 429 vs 486–512 TFLOPS
  for `mxf4nvf4` (clocks); all numbers are one rank, one window, unpinned clocks, not yet repeated in the
  other order or on the other three boxes.

**Deviations from EXL3 in the trellis kernels** (none changes the stored tensor; an exllamav3-quantised
trellis/suh/svh set runs after the tile reorder): tiles re-ordered from `[k/16][n/16]` to `[n/16][k/16]`;
MMAs accumulate in f32 (exllamav3's GEMV accumulates in f16 and folds to f32 every few tiles); x is BF16 and
x ⊙ suh is formed in f32 (exllamav3: half × half in half) before the f32 Hadamard and the f16 rounding; the GEMV
output stays f32 into the output Hadamard; h between up and down is rounded to bf16 (our contract); only the
integer K = 3, 4 (2 checked on the CPU) are built, not the fractional `HALF` rates; one codebook per tensor
(3INST or MUL1; MCG is implemented and CPU-checked, not timed). exllamav3 read at its master as fetched on
2026-10-13 (`quant/codebook.cuh`, `exl3_dq.cuh`, `pack.cu`, `hadamard.cu`, `exl3_gemv_kernel.cuh`,
`modules/quant/exl3.py`, `exl3_lib/quantize.py`; MIT); the commit hash could not be read (GitHub API rate limit).
