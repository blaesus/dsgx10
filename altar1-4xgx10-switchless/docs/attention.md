# The attention side for altar-1 (milestone A3a)

What `engine/attn/` does for altar-1, as a delta from GLM-5.3's
([glm53 attention.md](../../glm53-4xgx10-switchless/docs/attention.md), "glm53" below): the INT4 and BF16
projections that replace glm53's FP8 ones, their layouts, how they were checked, and what they measured. Written
2026-10-14. The API and every constant are in [engine/attn/attn.h](../engine/attn/attn.h); the plan is
[reuse.md § 3](reuse.md#3-attention-side) and [DESIGN.md](../DESIGN.md).

## Contents

1. [What is identical](#1-what-is-identical)
2. [The per-rank weights and their bytes](#2-the-per-rank-weights-and-their-bytes)
3. [The INT4 contract and its rounding point](#3-the-int4-contract-and-its-rounding-point)
4. [Layouts](#4-layouts)
5. [Kernels](#5-kernels)
6. [Tests](#6-tests)
7. [Measurements](#7-measurements)
8. [What the runner must change](#8-what-the-runner-must-change)
9. [Failures and what remains](#9-failures-and-what-remains)

## 1. What is identical

altar-1 has GLM-5.3's attention geometry (64 heads, q_lora 2048, kv_lora 512, nope 192, rope 64, v 256, the DSA
indexer, 78 layers, 21 with their own indexer) and glm53's per-rank split (16 heads a rank, the x-projection's
rows split and all-gathered, o_proj's columns split and all-reduced). Unchanged, bit for bit, and still tested by
the same checks: the cache entries (latent + k_rope, 1152 B; index key 256 B or the FP8 option), `prep.cu` (the
norms, RoPE, index keys and index weights), `indexer.cu` (scores, exact top-2048, position split), `sparse_attn.cu`,
the RoPE table, the head's BF16 GEMV, and the per-rank row and column cuts of every tensor (glm53 § 1 and § 2).
Only the projections' weight formats change: glm53's FP8 128 × 128 blocks become INT4 or BF16 (§ 2).

The FP8 path is kept (`WFmt::kFP8`, glm53's tiles) and still passes glm53's tests on FP8 data (`gen_ref.py --fmt
fp8`).

## 2. The per-rank weights and their bytes

| Projection (rank r) | Cut | Layers 4–76 | Layers 0–3, 77 (and the MTP layer 78) |
|---|---|---|---|
| x-projection: q_a [512r, +512), kv_a [128r, +128) and [512+16r, +16), wk [32r, +32) (full layers) | rows | BF16 688 × 6144 (656 shared), 8.45 MB (8.06) | the same |
| indexer `weights_proj` [8r, +8) | rows | BF16 8 × 6144, 0.10 MB | the same |
| q_b [4096r, +4096) | rows | **INT4** 4096 × 2048, 4.85 MB | BF16, 16.78 MB |
| indexer `wq_b` [1024r, +1024) | rows | BF16 1024 × 2048, 4.19 MB | the same |
| W_UK^T (kv_b's nope rows of the rank's 16 heads, transposed) | heads | **BF16, the exact dequantisation** 16 × 512 × 192, 3.15 MB | BF16, 3.15 MB |
| W_UV (kv_b's value rows) | heads | **INT4** 16 × 256 × 512, 1.21 MB | BF16, 4.19 MB |
| o_proj columns [4096r, +4096) | columns | **INT4** 6144 × 4096, 14.55 MB | BF16, 50.33 MB |
| dense gate+up / down (layers 0–2) | intermediates [3072r, +3072) | — | BF16 6144 × 6144 / 6144 × 3072, 75.5 / 37.7 MB |
| shared expert gate+up / down | intermediates [512r, +512) | BF16 1024 × 6144 / 6144 × 512, 12.6 / 6.3 MB | the same |
| head [38720r, +38720) | rows | BF16, 475.8 MB | |

INT4 bytes: `weight_bytes(kINT4, N, K)` = N/8 × K/512 × 2368 (0.578 B a weight: 4-bit values, a bf16 scale and a
4-bit zero point a group of 32). **Attention bytes a token a rank** (x-projection to o_proj, 78 layers): INT4
full-indexer layer 36.50 MB (155.3 µs at 235 GB/s) × 18, INT4 shared layer 31.82 MB (135.4 µs) × 55, BF16 full
87.20 MB × 3, BF16 shared 82.51 MB × 2: **2.83 GB, 12.06 ms** (glm53's FP8: 3.27 GB, 13.9 ms). With the dense MLPs
(0.34 GB), the shared experts (1.42 GB) and the head (0.48 GB): 5.06 GB, 21.6 ms. Of these, **3.56 GB are BF16**
(1.33 GB of them in attention), which is why the BF16 GEMV's efficiency matters (§ 7, kBF16T).

## 3. The INT4 contract and its rounding point

The weight the engine must reproduce is compressed-tensors' dequantisation, which rounds once:
`W(n, k) = bf16_rne((nib(n, k) − znib(n, k/32)) · s(n, k/32))`, with `nib = (weight_packed[n][k/8] >> 4(k%8)) & 15`,
`znib = (weight_zero_point[n/8][k/32] >> 4(n%8)) & 15` (both stored with +8, so q − z = nib − znib), `s` the bf16
`weight_scale[n][k/32]`. Source lines, compressed-tensors 0.18.0:

- `compressors/pack_quantized/base.py` 117–160, `decompress`: the zero points unpacked along dim 0 (line 150),
  the weights unpacked (155), then `dequantize` (156);
- `compressors/pack_quantized/helpers.py` 104–170, `unpack_from_int32`: the nibbles, then `(output − offset)`
  with offset 8, as int8;
- `quantization/lifecycle/forward_helpers.py` 550–567, `_dequantize`: `x_q.to(scale.dtype)` (bf16, exact),
  `− zero_point.to(scale.dtype)` (exact: an integer in [−15, 15]), `* scale` (bf16 × bf16 in torch: the f32
  product, exact, rounded once to bf16).

`tools/int4_check.py` runs the library's own `decompress` on real tensors of rank 0's checkpoint copy (shards
sha256-checked against the manifest; log `~/dsgb10-data/altar1-attn/int4_check.log`, 2026-10-14): q_b, kv_b and
o_proj of layers 4, 10 and 40 (33.5M, 14.7M and 100.7M values each): **0 values differ** from
`bf16_rne((nib − znib) · s)`; the product is inexact (so the rounding matters) in 15.9–25.9 % of them;
rounding `q · s` and `z · s` separately would differ in ~50 %, an f32 `q · s − bf16(z · s)` in ~42 %; torch's
bf16 `(q − z) · s` = the f32 product rounded to nearest even for all 2,023,680 (scale, q, z) cases tried. Zero
points span 0–15 (layer 4's q_b) and 2–13 elsewhere; scales 1.4e-5–0.42.

The kernels compute it the same way: the nibble becomes the bf16 `128 + nib` by bit placement (`lop3` with
0x4300), the subtraction `(128 + nib) − (128 + znib)` is exact in bf16 (`hsub2`), and one `hmul2` by the scale
rounds once (attn_common.cuh `int4x16_to_bf16`). Test D checks this for every finite bf16 scale pattern and all
256 (q, z) pairs: 16.7M products, 0 differ (19,160 of them subnormal).

## 4. Layouts

Reference implementation: [engine/attn/tools/pack.py](../engine/attn/tools/pack.py) (the converter's model).

**INT4 tile** (`int4_tiles`, `int4_q_b`, `int4_w_uv`, `int4_o_proj`): blocks of 8 rows × 512 k, 2,368 bytes,
block (n8, b) at byte ((n8 · K/512) + b) · 2368. Lane l = 4g + t holds row 8 n8 + g; unit jj (0–3) is
k = 512b + 128jj … + 127, in which the lane's 32 values are k = 512b + 128jj + 64q + 16t + i (q = 0, 1; i = 0–15):

| Part | Bytes | Where |
|---|---:|---|
| values: the checkpoint's nibbles (+8 kept), low nibble = even i | 2048 | 512 jj + 16 l + 8 q + i/2 |
| scales, bf16 s(n, 16b + 4jj + 2q + (t ≫ 1)) | 256 | 2048 + 32 g + 16 (t ≫ 1) + 2 (2jj + q) |
| zero points, the nibble of the same group | 64 | nibble 2jj + q (low first) of the u32 at 2304 + 8 g + 4 (t ≫ 1) |

Lanes t and t ^ 1 share their group (a lane's 16 values of a 64-k chunk lie in one group of 32). **This is the
expert agent's INT4 block byte for byte** (`engine/experts/layout.h` on its branch: the same values, scale and
zero-point offsets), so one packer serves both. q_b: rows [4096r, +4096); W_UV: kv_b rows 448H + 192 … + 447 of
heads H = 16r … 16r + 15 stacked (K = 512, groups along K); o_proj: columns [4096r, +4096) (K = 4096, groups
along K; every rank's column range starts on a group boundary). Any K that is a multiple of 512 tiles; all three
are.

**W_UK^T from an INT4 kv_b is BF16**: kv_b's nope rows transposed (`[512 (c)][192 (d)]` a head) put the groups of
32 along the output, so a lane's 16 consecutive k would need 16 scales. `int4_w_uk` stores the library's
dequantisation transposed (+0.15 GB a token against INT4; a transposed INT4 kernel is left, § 9). Checked:
the tile = compressed-tensors' `decompress` transposed, layer 10 rank 1.

**BF16**: row-major `[N][K]` (`bf16_xproj`, `bf16_rows`, `bf16_cols`, `bf16_w_uk`, `bf16_w_uv`, `bf16_gate_up`),
the cuts of glm53 § 2 with BF16 in place of FP8 codes and no scales. **kBF16T** (`bf16_tiles`, new, optional): the
same values in the GEMV's lane order, `[N/8][K/64][2 halves][32 lanes][8 values]`: in chunk j of tile i, lane
4g + t's half h holds row 8i + g, k = 64j + 16t + 8h … + 7. A warp's 64-k chunk is one contiguous 1 KB and each
16-byte load instruction reads 512 contiguous bytes. For K % 128 = 0 it is the expert agent's BF16 block
(2,048 B = two chunks; checked in Python against `layout.h`'s description). Outputs are identical bit for bit
to the row-major form's (test T); it is faster (§ 7), so **the converter should write every BF16 projection of
this table as kBF16T** (row-major stays accepted; one call may mix both).

## 5. Kernels

All in `gemv.cu` / `gemm.cu`. Every `GemvSeg` now carries its format (`WFmt`: kBF16, kBF16T, kINT4, kFP8; kNone
is refused, a format is never guessed). `gemv()` / `gemm()` take segments of mixed formats and launch once a
format family (BF16 and kBF16T share a launch), in the order BF16, INT4, FP8. glm53's `gemv(bool fp8, …)` stays
as a shim (kNone → the flag's format) for the runner's unchanged call sites.

**INT4 GEMV** (`k_gemv4`). A warp owns an 8-row tile and walks its 2,368-byte blocks: per block, the lane's 64
value bytes, its four scales and zero points; per 64-k chunk the 16 values are widened to bf16 (§ 3) and feed the
BF16 GEMV's four m16n8k16 MMAs accumulating straight into the f32 sum, in k order. Same lane map, same MMAs,
same order as the BF16 GEMV, so **an INT4 GEMV's output equals the BF16 GEMV's on the dequantised matrix bit for
bit** (test D), for any row count 1–16, split-K (in whole 512-k blocks), grouped mode (W_UV) and outputs bf16 /
f32 / SwiGLU. Launch shapes (blocks in flight a lane × warps a CTA): 1×4, 2×4 (default), 1×8, 2×8, 3×4, 4×4,
bit-neutral; 90–168 registers.

**BF16 GEMV** (`k_gemv`): glm53's BF16 path, plus kBF16T (only the address of a chunk changes), the index weights
fused into a BF16 x-projection's launch (`GemvIw`; glm53 had it FP8-only), and launch shapes by shape (§ 7):
K ≤ 512 → 4 chunks in flight × 8 warps; N·K ≤ 2^21 (wq_b alone) → 16 × 4; N ≥ 16,384 (the head) → 4 × 4;
otherwise 8 × 4 (glm53: 4 × 4 always).

**GEMM** (prefill). BF16 segments: `gemm_bf16_run`, glm53's tensor-core GEMM with BF16 B (128 × 64 tiles, 3
stages), now with SwiGLU, kBF16T (staged half-major, as it lies in memory; row-major staged lane-major, as
before: § 9 has the failed attempts), and a **grouped raster**: once a launch's weights reach 32 MiB the CTAs walk
8 row tiles fastest, so the CTAs in flight share their weight tiles instead of re-reading all the weights once a
row tile (the BF16 o_proj at 2,600 rows read 1 GB of weights for 50 MB). The raster changes which CTA computes a
tile, not its arithmetic (test P over rasters 0/1/3/16). INT4 segments: `dequant_int4` (the GEMV's element
arithmetic) into a caller's BF16 scratch in kBF16T order, then the BF16 GEMM; equal to the INT4 GEMV bit for bit
by construction and by test (P, R). A fused INT4 B loader was not built: the dequantisation costs 94 µs (q_b) and
301 µs (o_proj) a chunk against 0.81 / 2.44 ms for the whole INT4 GEMM at 2,600 rows (11–12 %, § 7).

## 6. Tests

`tests/attn_tests.cu` (test letters A D T C B P E F G S H; R for real tensors), data from `tests/gen_ref.py --fmt
int4|bf16|fp8` (random weights at the real shapes in each format: INT4 as compressed-tensors' canonical tensors
through pack.py, the torch reference with f32 accumulation over the dequantised weights) and `tools/real_ref.py`
(real tensors). Final run 2026-10-14 11:06 (+0100), commit `9b7081f` (later commits: comments and docs only) built fresh into
`~/dsgb10-data/altar1-attn/build`, rank 0's GPU through `ring.sh`:

| Data | Checks | Result |
|---|---|---|
| S = 300, INT4 layer (`s300`) | A D T C B P F G S H | **95 pass, 0 fail** |
| S = 300, BF16 layer (`s300b`) | the same (D skipped) | 72 pass |
| S = 300, FP8 (glm53's format, `s300f`) | the same | 72 pass |
| S = 4600, INT4 (`s4600`) | incl. E (indexer) | 119 pass |
| real tensors (`real`) | R | 12 pass |
| compute-sanitizer memcheck | A D T C B F P (s300), A T C B P (s300b), E (s4600) | **0 errors** (85 + 52 + 33 checks) |

What the new checks establish:

| Check | Result |
|---|---|
| A: INT4 q_b / o_proj tiles through the INT4 GEMV (one-hot rows) = `bf16_rne((nib − znib) · s)` from the canonical tensors | 0 of 262,144 / 393,216 differ |
| A: `dequant_int4` (row-major and kBF16T) = pack.py's `int4_dequant`, ranks 0 and 3, q_b / W_UV / o_proj | bit for bit |
| D: the element arithmetic, every finite bf16 scale × 256 (q, z) | 0 of 16,711,680 differ |
| D: INT4 GEMV = BF16 GEMV on the dequantised matrix, q_b / W_UV / o_proj, ranks 0 and 3 | bit for bit in 189 / 63 / 189 cases each (bf16 / f32 / SwiGLU out, split-K 1/2/4, 1/6/16 rows, launch shapes default and 0–5) |
| D: q_b (INT4) + wq_b (BF16) in one call = each alone | bit for bit |
| T: kBF16T = row-major BF16 (x-projection split-K 4, q_b, wq_b, W_UK grouped, W_UV grouped, o_proj, shared down) | bit for bit: GEMV at 1/6/16 rows in all launch shapes, GEMM at 300 / 600 rows in three rasters, three outputs |
| B: x-projection with the index weights in its launch (BF16) = two launches | bit for bit |
| B: q_b, wq_b, kv_a, q_a, wk vs the torch reference (INT4 and BF16 layers) | 4.9e-4 – 1.6e-3 of values differ, all by 1 ulp |
| P: every projection's GEMM (INT4 through the scratch, BF16, FP8) = the 16-row GEMVs | bit for bit in 6 tile configurations and rasters 0/1/3/16 |
| H: dense MLP (BF16 SwiGLU), shared expert, head vs reference | ≤ 3.6e-3 of values differ, all by 1 ulp |
| R (layer 10 INT4, layer 3 BF16, rank 1, real checkpoint tensors): `dequant_int4` of pack.py's tile = compressed-tensors' `decompress` | q_b, W_UV, o_proj: every value bit for bit |
| R: INT4 GEMV = BF16 GEMV on the library's weights = INT4 GEMM | bit for bit (1/6/16 rows, bf16 and f32 out) |
| R: GEMVs vs torch's f32-accumulated product on the library's weights (16 rows) | INT4 q_b 6.4e-4, W_UV 1.2e-4, o_proj 2.0e-3 of values differ; BF16 W_UK^T 7.6e-5, q_b (L3) 1.4e-3, o_proj (L3) 2.3e-3: all by 1 ulp (summation order) |

## 7. Measurements

Rank 0's GB10 (48 SMs) through ds41f's `ring.sh` in short windows between production turns, 2026-10-14. Each
figure is the median of 7 replays of a CUDA graph of L steps over ≥ 768 MB of distinct weights (nothing
L2-resident); "PDL chain" as the decode step runs. Final runs: commit `9b7081f`, sources sha256
`8bde61f69d5af3ff`, the case list forward (11:09:36 +0100) and reversed (11:09:45, `BENCH_REVERSE=1`): A/B in
both orders; streaming read 234.1 / 232.7 GB/s; SM clock 2184–2190 MHz, 47–52 °C. Full tables:
`~/dsgb10-data/altar1-attn/results.tsv` and `results-rev.tsv`.

### Decode GEMVs (% of 235 GB/s, PDL chain, rows 1 / 6 / 16; forward | reversed)

| Projection | MB | Row-major BF16 or INT4 | kBF16T |
|---|---:|---|---|
| x-projection BF16 688 × 6144, split-K 4 | 8.45 | 91.0 / 90.7 / 89.6 \| 92.2 / 91.7 / 92.3 | **96.7 / 95.9 / 81.8 \| 96.4 / 97.3 / 97.5** |
| x-projection, shared layer 656 × 6144 | 8.06 | 92.0 / 91.8 / 91.4 \| 91.7 / 91.2 / 91.0 | (not run; same kernel) |
| x-projection + index weights, one launch (BF16) | 8.55 | 90.9 / 90.6 / 85.9 \| 90.8 / 90.2 / 86.4 | |
| … the same as two launches | 8.55 | 91.6 / 87.7 / 80.3 \| 90.8 / 87.2 / 79.9 | |
| **q_b INT4** 4096 × 2048 | 4.85 | 99.9 / 100.9 / 85.9 \| 96.4 / 87.3 / 81.0 (12 runs: median 100.7 / 99.6 / 89.2) | |
| indexer wq_b BF16 1024 × 2048 alone | 4.19 | 72.9 / 72.1 / 65.2 \| 73.1 / 72.2 / 65.1 | 75.4 / 74.4 / 69.3 \| 75.6 / 74.8 / 69.4 |
| q_b INT4 + wq_b BF16, one call (two launches) | 9.04 | 85.0 / 84.0 / 75.0 \| 84.9 / 84.0 / 74.9 | 87.4 / 85.8 / 75.8 \| 86.7 / 85.8 / 75.7 |
| W_UK^T BF16 grouped 16 × 512 × 192 | 3.15 | 101.1 / 100.5 / 98.0 \| 100.6 / 100.0 / 98.6 | 102.7 / 103.3 / 104.0 \| 102.9 / 102.7 / 103.3 |
| **W_UV INT4** grouped 16 × 256 × 512 | 1.21 | 101.6 / 103.8 / 99.4 \| 104.6 / 100.9 / 99.5 | |
| **o_proj INT4** 6144 × 4096 (f32) | 14.55 | 95.5 / 94.9 / 90.5 \| 98.5 / 98.5 / 89.7 (12 runs: median 97.4 / 95.2 / 90.3) | |
| q_b BF16 (BF16 layers) + wq_b | 20.97 | 94.9 / 94.0 / 95.7 \| 93.9 / 93.5 / 95.3 | 100.0 / 99.8 / 100.8 \| 99.2 / 98.9 / 100.4 |
| W_UV BF16 grouped (BF16 layers) | 4.19 | 101.1 / 101.1 / 100.2 \| 100.8 / 99.9 / 98.3 | 104.1 / 104.0 / 103.9 \| 103.9 / 103.2 / 103.9 |
| o_proj BF16 (BF16 layers, f32) | 50.33 | 93.6 / 93.2 / 92.3 \| 93.5 / 92.9 / 92.5 | **99.6 / 98.7 / 99.1 \| 99.0 / 98.5 / 98.5** |
| dense gate+up BF16 6144 × 6144 (SwiGLU) | 75.50 | 94.4 / 94.4 / 94.6 \| 94.4 / 94.6 / 94.2 | **99.7 / 99.5 / 99.5 \| 99.6 / 100.0 / 100.3** |
| dense down BF16 6144 × 3072 (f32) | 37.75 | 95.1 / 93.9 / 94.4 \| 94.5 / 93.6 / 93.7 | 99.2 / 98.9 / 98.9 \| 99.3 / 98.7 / 98.9 |
| shared gate+up BF16 1024 × 6144 (SwiGLU, split-K 4) | 12.58 | 96.0 / 95.9 / 96.4 \| 95.4 / 95.1 / 95.5 | 100.3 / 100.0 / 100.8 \| 100.0 / 100.1 / 100.6 |
| shared down BF16 6144 × 512 (f32) | 6.29 | 100.8 / 101.6 / 101.8 \| 101.1 / 100.7 / 101.1 | 102.7 / 102.8 / 102.4 \| 102.7 / 102.9 / 102.8 |
| head BF16 38720 × 6144 | 475.8 | 94.9 / 93.9 / 94.0 \| 94.7 / 94.5 / 94.0 | **99.6 / 99.7 / 99.2 \| 99.7 / 99.4 / 98.9** |

Readings:

- **The INT4 GEMVs reach the floor at 1 and 6 rows** (q_b, W_UV, o_proj 95–105 %; above 100 % the PDL chain
  overlaps the next launch's loads). **At 16 rows they lose 10–20 %**: a warp reads its tile's activations for
  every 8 weight rows, and at 16 rows of 2-byte activations against 0.58-byte weights that is 7× the weight bytes
  from L2 (FP8 at glm53: 4×). Launch shapes do not fix it (sweep 10:34, best 91–94 % at 16 rows).
- **Row-major BF16 plateaus at 93–95 % on large shapes** whatever the launch shape (sweep 10:34: no shape lifts
  o_proj, the dense MLP or the head above 95.2 %; q_b BF16 reaches 97 % at 1 row with 16 chunks in flight): a
  warp's chunk is 8 rows × 128 bytes, 8 DRAM streams a warp. **kBF16T removes
  it**: 98.5–100.3 % on the same shapes, in both orders. Bytes a token at stake (3.56 GB of BF16): ~0.8 ms a
  token at 235 GB/s.
- **The small shapes**: the x-projection (8.45 MB, 86 tiles of 8 rows at split-K 4) is 90–92 % row-major and
  96–97 % as kBF16T (one forward 16-row value 81.8, an outlier the reversed run does not repeat). wq_b alone
  (4.19 MB, 17.8 µs at the floor) loses a fixed 4–6 µs of ramp and tail: 72–75 %; in the step it runs in the q_b
  call, two launches, 85–87 % at 1–6 rows and 75 % at 16 (§ 9).

Launch-shape sweep (2026-10-14 10:34, commit `0251096`, `agemvcfg`): the BF16 defaults above were chosen from it
(gains of 1–10 points on wq_b, the x-projection, W_UV and shared shapes; nothing changes a bit); the INT4 default
(2 blocks in flight × 4 warps) was best or within 2 points on all three INT4 shapes at 1 and 6 rows.

### Prefill GEMMs (TFLOPS at 2,600 / 8,192 rows; % of 107; forward | reversed)

| Projection | Row-major BF16 or INT4 | kBF16T |
|---|---|---|
| x-projection 688 × 6144 (split-K 4) | 50.8 / 53.9 \| 50.9 / 54.0 | 52.0 / 55.2 \| 52.0 / 55.2 |
| **q_b INT4** (dequant into scratch + GEMM) | 54.1 / 60.2 \| 48.2 / 60.3 | |
| q_b INT4 + wq_b | 53.7 / 59.1 \| 54.2 / 59.7 | (+ wq_b kBF16T) 54.6 / 59.9 \| 43.9 / 59.7 |
| W_UK^T BF16 grouped (K 192) | 23.3 / 24.0 \| 22.8 / 24.0 | 23.4 / 24.0 \| 23.5 / 24.1 |
| **W_UV INT4** grouped (K 512) | 33.3 / 34.8 \| 32.7 / 34.5 | |
| **o_proj INT4** (f32) | 53.7 / 60.1 \| 54.0 / 60.6 | |
| q_b + wq_b BF16 (BF16 layers) | 57.3 / 59.0 \| 58.4 / 60.5 | 55.6 / 55.9 \| 59.4 / 61.6 |
| o_proj BF16 (BF16 layers) | 59.5 / 61.5 \| 59.4 / 61.3 | 61.1 / 63.0 \| 60.9 / 63.0 |
| dense gate+up (SwiGLU) | 60.3 / 62.5 \| 60.3 / 62.6 | 61.8 / 64.1 \| 61.9 / 64.1 |
| dense down | 58.5 / 60.3 \| 57.9 / 60.0 | 60.0 / 61.9 \| 59.5 / 61.7 |
| shared gate+up (SwiGLU, split-K 4) | 53.6 / 54.5 \| 53.7 / 54.4 | 54.7 / 55.7 \| 54.9 / 55.7 |
| shared down (K 512) | 34.0 / 35.0 \| 31.7 / 35.5 | 35.6 / 36.4 \| 35.0 / 35.7 |
| head (2,600 rows) | 61.4 \| 59.0 | 63.2 \| 62.3 |

`dequant_int4` alone (read + write): q_b 93.6 µs (231 GB/s), o_proj 301 µs (215 GB/s), W_UV 8.5 µs (kBF16T
order; row-major 13.1 µs). glm53's FP8 GEMMs reached 57–60 % on the same large shapes; the small-K grouped
shapes (W_UK K = 192, W_UV and shared down K = 512) are bound by output and activation bytes there too (glm53:
28–40 TFLOPS). Single-sample outliers (q_b INT4 48.2 reversed at 2,600 rows; q_b + wq_b kBF16T 43.9; kBF16T
q_b + wq_b 55.6 forward) do not repeat in the other order.

**The raster** (`apgemmr`, 2026-10-14 10:34, commit `0251096`, rasters 0 → 16 → 0 in one job): before it the BF16
GEMM was weight-bandwidth-bound on wide large-K shapes: o_proj BF16 at 2,600 rows 4,250 µs (31 TFLOPS) in
blockIdx order, 2,186 µs (60) grouped by 8; dense gate+up 6,773 → 3,308 µs; the head 47.9 → 20.5 ms (26 → 60
TFLOPS); o_proj INT4 5,694–6,203 → 2,489 µs. Grouping costs 5–30 % on shapes whose weights fit in L2 (x-projection 428
→ 466 µs, W_UV 346 → 385 µs at 8), hence the 32 MiB threshold.

## 8. What the runner must change

The kernels take formats; the runner (another agent) binds them. Line numbers are `engine/model/` at this
branch's base (glm53 at 9e4a40b):

1. **Weight structs** (`impl.h` `Fp8W`, `bind.cu` 218–292): a projection is `{WFmt fmt; const void* w; const
   float* s}` (s only for FP8), and each `GemvSeg` built from it sets `fmt`. Per layer: 4–76 q_b, W_UV, o_proj
   INT4 tiles; W_UK BF16 (dequantised); x-projection, wq_b, weights_proj BF16; 0–3, 77, 78 everything BF16. The
   converter should write the BF16 ones as kBF16T (`pack.bf16_tiles`).
2. **The step** (`model.cu` `layer()`): 246–249 the x-projection BF16 with the index weights in its launch
   (`GemvIw` is now allowed with BF16; one launch measured 90–91 % against 80–91 % for two, so keep `kTuneIw`
   on); 266–270 q_b + wq_b in one `gemv()` call with each segment's format (INT4 + BF16: two launches inside);
   284 W_UK^T, 295 W_UV, 304 o_proj (f32), 316–319 the dense MLP (BF16, SwiGLU then f32); 326 and 386 (router,
   head) unchanged. Call the format-tagged `gemv()` (the `gemv(bool fp8, …)` shim only exists for glm53's sites).
   `oproj.cu` (FP8-only, off by default) must stay off or go.
3. **l2pf windows** (`model.cu` 184–210): one buffer per projection of `weight_bytes(fmt, N, K)` bytes and no
   scale buffer for INT4 / BF16 (the INT4 tile carries its scales and zero points); o_proj INT4 is 14.55 MB,
   q_b 4.85 MB, W_UV 1.21 MB, W_UK 3.15 MB, x-projection 8.45 / 8.06 MB.
4. **step_bytes** (`model.cu` 673–688): per format with `weight_bytes`; per layer as § 2 (36.50 / 31.82 MB INT4
   full / shared attention, 87.20 / 82.51 MB BF16), plus the BF16 dense MLP and shared expert, and the experts.
5. **Prefill** (`prefill.cu` `pf_proj` 100–112): pass the segments' formats (this branch maps kNone → FP8 there
   so glm53's binding still runs) and a BF16 scratch for INT4 segments of at least N × K elements of the largest
   INT4 projection called: o_proj 6144 × 4096 = **50.3 MB** (q_b 16.8 MB, W_UV 4.2 MB); `gemm()` refuses an INT4
   segment without one. Below 17 rows `pf_proj` uses the GEMVs, which need nothing.
6. **MTP** (`mtp.cu` 24–25): `eh_proj` BF16.

## 9. Failures and what remains

Failures and dead ends (each in [experiments.md](experiments.md)):

- **kBF16T's first in-chunk order** ([32 lanes][32 B], commit `82305cd`) gave the GEMV its 99 % but its B copies
  into the GEMM's shared memory scattered when the order changed to the expert block's ([2 halves][32 lanes]
  [16 B], `97d0dc9`): the kBF16T GEMM fell to 42–46 TFLOPS from 58–61. Remapping the copy threads (`eb4fda1`) did
  not help; staging every BF16 form half-major (`21d552d`) fixed kBF16T (58–64) but dropped row-major to 44–50.
  Kept (`9b7081f`): each form staged in its own source order (linear copies both sides).
- **The raster by default for every shape** costs 5–30 % on small shapes; hence by weight bytes.
- **o_proj INT4 at 16 rows** read 77.2 % once (10:37 run) against 90–91 % in every other run: an outlier, not
  reproduced in either order since.

What remains:

- **INT4 at 16 rows** (q_b 81–91 %, o_proj ~90 %): the activations' L2 traffic (§ 7). A CTA staging the rows' x
  in shared memory once for all its warps would cut it 4–8×; not done.
- **wq_b**: 72–75 % alone, 85–87 % with q_b as two launches. One launch for mixed INT4 + BF16 segments (k_gemv4
  running a BF16 warp loop for BF16 tiles) would save ~5 µs on each of the 18 INT4 full layers (~0.1 ms a token).
- **W_UK^T as INT4** (a transposed kernel, 32-row tiles so a scale serves a group): −2.2 MB a layer on layers
  4–76, 0.16 GB and ~0.7 ms a token at the floor.
- **Prefill INT4 without the scratch**: the dequantisation is 11–12 % of an INT4 GEMM at 2,600 rows and ~4 % at
  8,192; a fused loader only if a profile shows it.
- The x-projection + index weights fused launch uses row-major-tuned 4 × 4 warps; not measured with kBF16T.
