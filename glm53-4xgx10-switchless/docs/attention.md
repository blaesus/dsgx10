# The attention side (milestone G3, attention half)

What `engine/attn/` computes for one rank at TP=4, the layouts it reads and writes, the per-rank split, how it
was tested and what it measured. Written 2026-10-13; performance work 2026-10-14 (§ 5, last part). The math is [model.md § 2](model.md#2-the-forward-pass);
the API and every constant are in [engine/attn/attn.h](../engine/attn/attn.h).

## Contents

1. [The per-rank split](#1-the-per-rank-split)
2. [Layouts](#2-layouts)
3. [Kernels](#3-kernels)
4. [Numerics and tests](#4-numerics-and-tests)
5. [Measurements](#5-measurements)
6. [What remains](#6-what-remains)

## 1. The per-rank split

16 of the 64 heads a rank (heads 16r … 16r+15). Everything that reads the layer input `x` is one FP8 GEMV
whose rows are split by rank and all-gathered; everything per head stays on its rank.

| Weights | Split | Rows × K a rank | FP8 bytes a rank a layer |
|---|---|---|---:|
| x-projection: `q_a_proj` rows [512r, +512), `kv_a_proj_with_mqa` latent rows [128r, +128) and rope rows [512+16r, +16), indexer `wk` rows [32r, +32) (full layers only) | rows, one launch, all-gathered `[rows][4][688]` | 688 × 6144 (656 on shared layers) | 4.23 MB (4.03) |
| indexer `weights_proj` (BF16, f32 product) | rows [8r, +8), rides the same all-gather | 8 × 6144 | 0.10 MB (BF16) |
| `q_b_proj` | heads: rows [4096r, +4096) | 4096 × 2048 | 8.39 MB |
| indexer `wq_b` | heads: rows [1024r, +1024), the same launch as q_b; output all-gathered (2 KB a row) | 1024 × 2048 | 2.10 MB |
| W_UK^T (from `kv_b_proj`, absorbed into q) | heads, grouped GEMV | 16 × (512 × 192) | 1.57 MB |
| W_UV (from `kv_b_proj`, absorbed into the output) | heads, grouped GEMV | 16 × (256 × 512) | 2.10 MB |
| `o_proj` | columns [4096r, +4096); f32 partials all-reduced | 6144 × 4096 | 25.17 MB |
| dense MLP (layers 0–2) | intermediates [3072r, +3072); gate/up one launch with SwiGLU; down f32 partials | 6144 × 6144, 6144 × 3072 | 56.6 MB |
| shared expert (layers 3–77) | intermediates [512r, +512) | 1024 × 6144, 6144 × 512 | 9.4 MB |
| head (BF16) | vocabulary rows [38720r, +38720) | 38720 × 6144 | 476 MB |

**Why the x-projection is split, not replicated.** Replicating `q_a`+`kv_a`+`wk` reads 16.9 MB a layer a rank
instead of 4.2 (+12.7 MB × 78 layers ≈ 1 GB a token, ~4.2 ms); replicating only `kv_a` (576 rows, which do not
split on 128-row blocks) costs +2.7 MB a layer (0.9 ms a token). The split costs nothing extra: the
all-gather it needs (1.4 KB a row a rank) is the one the reference split needs anyway, and `kv_a`'s rows cut
cleanly once the scale table is kept per 16 rows (each rank: one 128-row latent block plus 16 of the 64 rope
rows, all inside block 4). Fusing the three tensors (and `wk`) makes one launch of 688 rows instead of three
small ones.

**Attention bytes a token a rank:** full-indexer layer 43.7 MB, shared layer 41.3 MB, 3.27 GB over 78
layers (13.9 ms at 235 GB/s); the dense MLPs 0.17 GB, the shared experts 0.70 GB, the head 0.48 GB.

**The cache is replicated** on every rank (all 16 local heads read the same latent): 1152 B a token a layer
(BF16 latent + roped k_rope) and 256 B on the 21 full-indexer layers (BF16 index key): 95,232 B a token. The
indexer's scan can still be split by position (each rank scans a quarter, then a merge of 4 × 2048
candidates, [§ 5](#5-measurements)); the cache itself stays whole on every rank in this milestone (DESIGN
decision 5 leaves the position-split cache to G4).

## 2. Layouts

### Cache entries (per layer, per session slot)

| Cache | Entry | Bytes | Layers |
|---|---|---:|---|
| KV | `[slot][pos][576]` bf16: the latent after `kv_a_layernorm` (512), then k_rope after RoPE (64), de-interleaved as the reference writes it (`out[i]`, `out[32+i]` from the pair `2i, 2i+1`) | 1152 | all 78 |
| index key, BF16 (default) | `[slot][pos][128]` bf16: dims 0–63 RoPE'd (de-interleaved), 64–127 as normed | 256 | 21 full |
| index key, FP8 (option) | codes `[slot][pos][128]` e4m3 and scales `[slot][pos]` f32 (amax/448 a key; vLLM's 132 B form) | 132 | 21 full |

A slot is `slot_stride` entries from the previous; rows carry their slot in a device table.

### Weight tiles the converter must emit (`engine/attn/tools/pack.py` is the reference implementation)

Every FP8 tensor is zai's codes (e4m3) and `weight_scale_inv` (f32 per 128 × 128), re-cut losslessly:

*Quantiser (2026-10-14):* `pack.py`'s `fp8_from_master` now uses zai's rule, `scale = f32(amax) × f32(1/448)`
(a multiply by the rounded reciprocal), not `amax / 448`, which differed by one ulp in ~half the scales and
matched 0 of 20 release tensors ([layout.md § 4](layout.md#4-fp8-conversion-zais-rule-reproduced)).
`tools/fp8_vs_converter.py --ckpt NV --cache CACHE` checks real tensors (L42 indexer wk, L5 kv_a with its
partial 128-row block, L0 q_a): pack.py's codes and scales are byte-identical to the converter's
(`engine/weights/quant.py`) and to zai's release, 3 of 3. The test data were regenerated with it (`gen_ref.py`,
S = 300 and 4600: scales changed in 44–67 % of blocks) and the attention tests rerun on the new data: 44 + 93
pass, memcheck clean. (The FP8 *index-key cache*, amax/448 a key, is the engine's own runtime format and is
unchanged.)

- **Tile layout** `[N/8][K/64][32 lanes][16 B]`: lane `4g + t` holds row `8i + g`, k = `64j + 16t + b` at
  byte `b` (512 contiguous bytes a warp-step). A warp widens its 16 bytes to bf16 exactly (bits into bf16's
  fields, one bf16 multiply by 2^120) and feeds them as the B operand of four m16n8k16 MMAs.
- **Scales** f32 `[N/rs][K/ks]`, every 8-row × 64-k chunk inside one of zai's blocks:

| Tensor (rank r) | Rows, in order | rs × ks | Scale entry |
|---|---|---|---|
| x-projection | q_a [512r, +512), kv_a [128r, +128), kv_a [512+16r, +16), wk [32r, +32) | 16 × 128 | the global block row of each 16-row group |
| q_b, indexer wq_b | rows [4096r, +4096), [1024r, +1024) | 128 × 128 | zai's, rows sliced |
| W_UK^T | head h (H = 16r+h): `kv_b` rows 448H … 448H+191 transposed to [512 (c)][192 (d)]; heads stacked | 128 × 64 | `[4h + c/128][j] = S[(448H + 64j)/128][c/128]` |
| W_UV | head h: `kv_b` rows 448H+192 … 448H+447, [256][512]; heads stacked | 64 × 128 | `[4h + v/64][cb] = S[(448H + 192 + 64(v/64))/128][cb]` |
| o_proj, down projections | columns [4096r, +4096) / the rank's intermediates | 128 × 128 | zai's, columns sliced |
| gate/up | the rank's intermediates, rows interleaved (2i gate, 2i+1 up) | 128 × 128 (intermediates) | `[inter/128][2][K/128]`: gate's block row, then up's |

(Head rows start at multiples of 448 = 3.5 blocks, so every head segment starts on a 64-row boundary and every
64-chunk lies in one block; hence ks = 64 for W_UK^T and rs = 64 for W_UV.)

- **BF16** (head, `weights_proj`): row-major `[N][K]`, unchanged.
- **RoPE table** `[pos][32]` u32 (bf16 cos low, bf16 sin high), written by `tools/rope_table.py` with torch
  itself (a double-precision build differs from torch's in 29 of 2^20 × 64 values); 128 MiB for 1M positions.

## 3. Kernels

A decode step of one layer on one rank, in launch order (all launches PDL-capable: each kernel issues its
weight loads, then waits for its predecessor):

| # | Kernel (file) | What it does |
|---|---|---|
| 1 | `rmsnorm` (prep.cu) | input_layernorm, one CTA a row, f32 statistics in a fixed order |
| 2 | `gemv` x-projection (gemv.cu) | 688 rows (656 shared layers) of FP8, split-K 4, bf16 out; then the all-gather (G4); can carry # 3 in its launch (`GemvIw`) |
| 3 | `index_weights` (prep.cu) | BF16 `weights_proj`, the rank's 8 heads, f32 out × 32^-0.5; all-gathered with the x-projection |
| 4 | `attn_prep` (prep.cu) | q_a_layernorm → qr; kv_a_layernorm + RoPE of k_rope → the cache entry; indexer key LayerNorm + RoPE → BF16 (or FP8 + scale) key |
| 5 | `gemv` q_b + indexer wq_b (one launch, two segments) | 4096 + 1024 rows, K 2048 |
| 6 | `gemv` W_UK^T grouped | per head: q_nope (192) → q_lat (512); x rows grouped by head |
| 7 | `index_select` (indexer.cu) | `k_score` then 5 selection passes → top-2048 positions (or a quarter's candidates + `index_merge`); prefill: `k_q_stage`, `k_score_pf`, `k_sel_thresh`, `k_sel_compact` |
| 8 | `sparse_attn` (sparse_attn.cu) | split-K over the 2048 keys, `k_attn` + `k_attn_combine` |
| 9 | `gemv` W_UV grouped | per head: latent (512) → v (256) |
| 10 | `gemv` o_proj | 6144 × 4096, f32 partials for the all-reduce (G4) |

**GEMV** (gemv.cu). A warp owns an 8-row tile; each step it loads 512 contiguous bytes (FP8) and runs four
m16n8k16 MMAs with the activations as A (rows g and g + 8 of the MMA: up to 16 rows at no extra weight cost)
and the widened weights as B. Up to P = 8 (FP8) or 4 (BF16) loads in flight a warp; the per-block f32 scale
multiplies each 64-k MMA partial (`acc += s · c`, the same order whatever the row count). Split-K (1, 2 or 4
warps a tile) sums the warps' partials in shared memory in warp order. Outputs: bf16, f32 (partials for
all-reduces) or SwiGLU (gate/up rows interleaved, the epilogue computes `silu(g) · u` in f32 and rounds once).
Grouped mode gives each `group_rows` block of output rows its own x row (absorption: head h's rows read head
h's q_nope or latent). Results are independent of the number of rows (tested 1/6/16 bit-equal). Loads in flight
and warps a CTA are chosen by shape (16 loads for the grouped W_UV, 8 warps a CTA otherwise); they do not change
any bit.

**GEMM** (gemm.cu). Prefill's projections: the GEMV's segments and outputs over any number of rows on tensor
cores (cp.async pipeline, 128-row tiles), each output equal to the GEMV's bit for bit (the same 64-k chunk MMAs
from zero, `acc + s · c` in k order, split-K parts in order).

**Indexer** (indexer.cu). `k_score`: a CTA takes up to RG rows of one session (decode: grouped by slot; prefill:
groups of 8) and 1024 keys. The rows' queries are RoPE'd while staged into padded shared memory; each warp
streams 16-key tiles straight from global memory into MMA A fragments (keys as M) and runs 4 n-tiles (8 heads)
× 8 k-steps against each staged row, then the head-weighted ReLU sum in a fixed order (n-tile, head 2t before
2t+1, quad xor 1 then 2). FP8 keys widen exactly to bf16 and their per-key scale multiplies the f32 dot before
the 128^-0.5 scale. 128 · RG threads (512 for RG ≥ 4, so a CTA with 74 KB of staged queries still has 16
warps an SM). Selection is exact: three histogram passes over a monotone 32-bit order key of the f32 score
(−0 mapped to +0) find the threshold and how many equal keys to take, a count pass and an ordered compaction
pass write the positions ascending; **ties are taken lowest position first** (torch.topk's CPU tie order is
implementation-defined, so the reference is compared up to ties at the threshold). Rows with p < 2048 take all
of 0..p; unused slots are −1. Position split: `index_select(lo, hi)` scans [lo, hi) and writes candidates with
their order keys; `index_merge` takes the top-2048 of the 4 ranks' 8192 (one 1024-thread CTA a row), ties by
list order = position order, so the merge equals the replicated result bit for bit.

**Sparse attention** (sparse_attn.cu). MQA over 576-wide entries: the 16 local heads are the MMA's M, keys in
blocks of 64 gathered by cp.async (one entry = 1152 B), S over 36 k-steps (latent 512 then rope 64), online
softmax in f32 with P rounded to bf16, O += P·V over the 512 latent dims (a warp per 128). Softmax scale
256^-0.5 (qk_head_dim), keys = the 2048 selected or 0..p. Splits: the 32 key blocks divide over `splits` CTAs a
row; `k_attn_combine` merges (m, l, O) in split order. Shared memory 96.5 KB, one CTA an SM. At one split (prefill,
and decode pinned to one split) `k_attn_pf` runs instead (since 2026-10-14, § 5 "T6"): `k_attn`'s arithmetic bit
for bit with the next key block's gathers in flight under the current block's compute.

**Prefill** runs the projections as GEMMs (`gemm()`), the indexer as `k_q_stage` (each row's RoPE'd query
staged once) and `k_score_pf` (256 keys in registers a CTA, the rows' queries streamed; scores = `k_score`'s bit
for bit, 52–66 TFLOPS), the selection as a sampled threshold (`k_sel_thresh`) and one compaction launch
(`k_sel_compact`, the exact top-2048 from the candidates), and `k_attn_pf` (`k_attn` before T6) with one row a CTA set. Chunking a prompt does not change any bit
(tested: two chunks = one, for cache, keys, top-k and attention output).

## 4. Numerics and tests

Reference: `tests/gen_ref.py` builds a layer with random weights at the real dimensions (FP8 through the
converter's own `pack.py`), runs model.md's math in torch, and stores inputs, intermediates and outputs
(`--mlp` adds the dense MLP, shared expert and head; `--hf` cross-checks against transformers'
`GlmMoeDsaAttention`). **This CPU's torch bf16 GEMM does not accumulate in f32** (about 1.6× the error of a
correctly rounded result, 41 % of outputs differ from it); with `linear`/`matmul` patched to f32 accumulation
the reference equals transformers' module bit for bit (S = 64, 300, 2600, including all 552 sparse rows'
top-k). Attention is also computed in f64 from the same bf16 inputs ("exact") to separate our error from the
reference's.

`tests/attn_tests.cu` on S = 300 and S = 4600 (16 decode rows, 2552 sparse rows), 2026-10-13, rank 0,
sources sha256 `d9fc5ba302e33670`: **30 + 51 checks pass, 0 fail**; compute-sanitizer memcheck 0 errors.

| Check | Result |
|---|---|
| tile dequant (x-projection, W_UK^T) through the kernel's fragment map | 0 of 32,768 / 262,144 differ |
| norms (q_a, kv_a, input RMSNorm; index-key LayerNorm) | 2e-6 – 2e-5 of values differ, all by 1 ulp except a handful at cancellation |
| k_rope RoPE into the cache | bit-exact (294,400 values); the RoPE table built by torch itself |
| GEMVs (q_a, kv_a, wk, q_b, wq_b ×4 ranks; dense/shared gate/up and down; head) | 3e-4 – 4e-3 of values differ, by 1 ulp (summation order); 1/6/16 rows bit-equal |
| index weights | max error 4.4e-6 of the row's largest weight |
| indexer scores | max 5.2e-7 of the row max |
| top-2048 | **identical to torch.topk for all 2552 rows**; decode 1/6/16 rows = prefill bit for bit; two prefill chunks = one; position split + merge = replicated, bit for bit |
| FP8 index keys (e4m3 + f32 amax/448 a key) | top-2048 overlap with BF16 keys mean 0.9930, min 0.9844 |
| attention vs f64 exact | rel L2 per head mean 5.5e-3, max 8.3e-3; **the reference's own 2.0e-2 / 3.1e-2**; 32 splits the same; decode = prefill bit for bit |
| layer chain (norm → o_proj, own projections) vs the reference | rel L2 2.5e-2, of which 2.1e-2 is the attention outputs' difference (the reference's bf16 attention) |
| slot tables (2 sessions in one call) | indexer, attention, cache writes = each row alone, bit for bit |
| determinism | a second run is bit-identical |
| (2026-10-14, T6) `k_attn_pf` | = `k_attn` at one split bit for bit, 300 rows dense and sparse; decode calls of 1, 6, 16 rows at one split = the prefill rows |
| dense / shared MLP chained | rel L2 4.6e-4 / 2.5e-4 |

## 5. Measurements

Rank 0's GB10 (48 SMs, 101,376 B opt-in shared memory), borrowed through ds41f's `ring.sh` while the service
was idle, 2026-10-13 22:20 (+0100), sources sha256 `01c2f224be9d560d` (benches; the later test-only change does
not touch kernels), SM clock 2184–2190 MHz at the end of each job, 47–60 °C. Each figure is the median of 7
replays of a CUDA graph of L steps (L layers' worth of distinct weights, ≥ 768 MB, so nothing is L2-resident).
"Plain" launches are graph nodes without PDL; "PDL chain" is the same with programmatic dependent launch, as the
decode path runs. Full tables: `~/dsgb10-data/glm53-attn/results.tsv`.

### GEMVs (% of 235 GB/s; rows 1 and 6, PDL chain; 16 rows within 1–5 % of these)

| Projection | MB | µs (1 / 6 rows) | % of 235 (1 / 6) | plain, 1 row |
|---|---:|---:|---:|---:|
| x-projection 688 × 6144 | 4.24 | 21.1 / 21.2 | **85.3 / 85.1** | 76.9 |
| x-projection, shared layer 656 × 6144 | 4.04 | 20.5 / 20.6 | **83.9 / 83.6** | 75.8 |
| q_b + wq_b (4096 + 1024) × 2048 | 10.49 | 44.9 / 44.4 | 99.5 / 100.6 | 93.1 |
| W_UK^T grouped 16 × 512 × 192 | 1.57 | 6.4 / 6.5 | 104.6 / 103.8 | 76.3 |
| W_UV grouped 16 × 256 × 512 | 2.10 | 11.1 / 10.8 | **80.7 / 82.6** | 72.1 |
| o_proj 6144 × 4096 | 25.17 | 106.6 / 107.1 | 100.4 / 100.0 | 96.1 |
| dense gate+up 6144 × 6144 (SwiGLU) | 37.76 | 163.4 / 162.9 | 98.4 / 98.6 | 95.6 |
| dense down 6144 × 3072 | 18.88 | 82.1 / 82.6 | 97.8 / 97.3 | 94.5 |
| shared gate+up 1024 × 6144 | 6.29 | 28.5 / 28.7 | **93.9 / 93.3** | 85.3 |
| shared down 6144 × 512 | 3.15 | 13.9 / 13.5 | 96.6 / 99.4 | 84.2 |
| head BF16 38720 × 6144 | 475.8 | 2116 / 2125 | 95.7 / 95.3 | 93.9 |
| index weights_proj BF16 8 × 6144 | 0.10 | 1.9 / 3.7 (plain) | — | — |

Above 100 %: with PDL the next kernel's weight loads overlap the previous kernel's tail, so a step can exceed
the single-kernel window. Below 95 %: the three small launches (x-projection, W_UV, shared gate/up) lose a
fixed ~2–3 µs of ramp and tail each; the x-projection has only 86 tiles (86 CTAs at split-K 4 for 48 SMs).
Attention-side decode weights a layer (x-proj → o_proj, full layer): ~192 µs at these rates (1 row) against 186 µs at
235 GB/s.

### Indexer, decode (one session; FP8 = e4m3 keys + f32 scale, 132 B a key; BF16 = 256 B)

| Context | Keys | Rows | Replicated µs (GB/s of keys) | Quarter scan + merge µs |
|---|---|---:|---:|---:|
| 32k | FP8 | 1 | 40.1 (105) | 34.6 + 10.5 |
| 32k | FP8 | 6 | 62.3 | 57.2 + 11.4 |
| 32k | BF16 | 1 | 88.7 (95) | 48.0 + 10.4 |
| 32k | BF16 | 6 | 108.8 | 70.8 + 11.4 |
| 128k | FP8 | 1 | 91.2 (184) | 44.6 + 10.4 |
| 128k | FP8 | 6 | 157.8 | 63.1 + 11.5 |
| 128k | BF16 | 1 | 308.8 (109) | 89.5 + 10.4 |
| 128k | BF16 | 6 | 340.4 | 112.0 + 11.4 |
| 256k | FP8 | 1 | **165.4 (203)** | **56.6 + 10.5** |
| 256k | FP8 | 6 | 291.3 | 104.9 + 11.4 |
| 256k | BF16 | 1 | 605.8 (111) | 157.0 + 10.4 |
| 256k | BF16 | 6 | 655.4 | 198.1 + 11.4 |

Includes scoring and all selection passes. The merge excludes the all-gather of 4 × 2048 candidates (32 KB a
row a rank; G4). Per token at 256k, 21 full layers, 1 row: FP8 replicated 3.5 ms, FP8 quarter + merge 1.4 ms;
BF16 replicated 12.7 ms, BF16 quarter + merge 3.5 ms. BF16 keys scan at only ~110 GB/s (the BF16 tile is twice
the registers a lane; not yet tuned); FP8 reaches 203 GB/s at 256k.

*Experiment (2026-10-13):* `k_score` first ran at 128 threads for every RG; at RG = 8 its 74 KB of staged
queries allowed one CTA an SM, 4 warps, and the MMA chain was latency-bound. 128 · RG threads (cap 512) took
256k FP8 6 rows from 545.6 to 291.3 µs replicated (186.7 → 104.9 quarter), 32k from 101.5 to 62.3; outputs
unchanged (scores are per key; tests rerun). Prefill indexer 18.7 → 20.6 TFLOPS.

### Sparse attention, decode (2048 selected keys, 16 heads)

| Context | Rows | splits 8 | splits 16 | splits 32 |
|---|---:|---:|---:|---:|
| 32k | 1 | 29.0 | 25.2 | 26.2 |
| 32k | 6 | **74.5** | 79.6 | 93.9 |
| 128k | 1 | 21.6 | **16.0** | 16.3 |
| 128k | 6 | **83.4** | 88.8 | 101.6 |
| 256k | 1 | 21.8 | **16.2** | 16.4 |
| 256k | 6 | **67.1** | 75.4 | 96.5 |

(µs a layer; sources sha `3680f438cfe04141`, the attention kernel unchanged since.) The target was ≤ 0.5 ms a
layer at 256k, 1 row: **16 µs**. Use 16 splits for 1–2 rows, 8 for 6. The gathered entries are 2.36 MB a row
(2048 × 1152 B); 6 rows reach 170–210 GB/s of gathered bytes. Per token, 78 layers: 1.3 ms (1 row), 5.2 ms
(6 rows).

### Prefill (a chunk of 2048 rows, one session)

| Chunk ends at | Indexer (scores + selection) | Attention |
|---|---:|---:|
| 32k | 25.9 ms (20.6 TFLOPS of score MMAs) | 8.4 ms (17.4 TFLOPS) |
| 128k | 105.8 ms (20.6 TFLOPS) | 19.4 ms (7.5 TFLOPS) |

256k was not run (the bench step's 5-minute budget; the indexer scales linearly: ~210 ms a chunk). The
attention slows at 128k because the 2048 selected entries are scattered over a 151 MB cache (L2 misses).

### 2026-10-14: prefill GEMM, prefill indexer, decode tails

Rank 0's GB10 through `ring.sh` between production turns, 2026-10-13 23:20 – 2026-10-14 00:09 (+0100); final
sources sha256 `5464b861c28b01ca` (benches) and `02e401c0ea49f9b1` (the committed sources, comments only since; tests rerun). Outputs are unchanged bit for bit:
every new path is tested equal to the old one (`attn_tests`: 44 checks at S = 300, 93 at S = 4600, all pass;
compute-sanitizer memcheck clean on A, B, C, F, P at S = 300 and E at S = 4600). "Before" is the base build
(`43bfd9a`) run in the same job where an A/B is given, else the tables above.

**Prefill projections: a tensor-core GEMM** (`gemm()`, gemm.cu). Any number of rows, the GEMV's segments, output
kinds, grouping and split-K; each output equals `gemv()`'s at the same split-K bit for bit: per 64-k chunk the
four m16n8k16 MMAs start from zero in the GEMV's fragment map and `acc = acc + s · c` in k order, split-K parts
added in part order. cp.async multi-stage pipeline (A 128 × 64 chunk, swizzled), B tiles in the engine's
warp-step layout, staged coalesced epilogue. Six tile configurations (bench knob `gemm_config`); the default is
by shape: 128 × 64 tiles, warps 64 × 16, 3 stages for split-K, 128 × 128, warps 64 × 32, 4 stages otherwise.
Test P: every projection (ranks 0 and 3, dense and shared MLP) in all six configurations = the 16-row GEMVs.

| Projection (TFLOPS at 2048 / 8192 rows; % of 107) | 16-row GEMVs (before) | GEMM, best configuration over the runs |
|---|---:|---:|
| x-projection 688 × 6144, split-K 4 | 4.4 / 3.9 | 51.8 / 56.3 (48 / 53 %) |
| q_b + wq_b 5120 × 2048 | 4.2 / 4.4 | 61.4 / 64.2 (57 / 60 %) |
| o_proj 6144 × 4096 | 5.3 / 5.0 | 64.4 / 62.7 (60 / 59 %) |
| dense gate+up 6144 × 6144 | 5.2 / 4.7 | 61.3 / 58.9 (57 / 55 %) |
| dense down 6144 × 3072 | 5.1 / 4.6 | 62.9 / 63.8 (59 / 60 %) |
| shared gate+up 1024 × 6144, split-K 4 | 5.3 / 4.6 | 49.9 / 54.9 (47 / 51 %) |
| W_UK grouped / W_UV grouped / shared down (best of either size) | 8.9 / 6.1 / 6.8 | 28.4 / 34.2 / 39.8 |

10–15× the GEMVs on the large shapes; the 60 % target is met by q_b, o_proj and dense down at one size each
and missed by 1–8 points elsewhere. The small-K shapes (W_UK K = 192, W_UV and shared down K = 512) are bound
by output and activation bytes, not MMAs. Tried and not kept: 64 × 128 tiles (config 1: 29–37 TFLOPS on wide
N); 128 × 128 with 16 warps (config 5: the split-K variant spills); 3 stages on unsplit shapes (equal or worse).

*BF16 weights (2026-10-14, the runner's prefill pass, [engine.md](engine.md#prefill-gemms-2026-10-14)):*
`gemm_bf16()` is the same kernel for row-major BF16 weights (the MoE router, the head): the BF16 GEMV's arithmetic
(the four MMAs of a 64-k chunk accumulate straight into the sum, no scale; split-K parts added in part order), a
stage's weights staged as each lane's 32 bytes, one tile shape (128 × 64, 3 stages, 72 KiB; warps 64 × 16 for a
split-K, 32 × 32 otherwise). Three new P checks at S = 4600 (rank 0's head rows: the router's shape 256 × 6144 at
split-K 4, the whole head 38,720 × 6144, 1000 rows at split-K 2 with bf16 out) = the 16-row BF16 GEMV bit for bit;
`attn_tests` 44 pass at S = 300 and 96 at S = 4600, memcheck (A C B F P, S = 300) clean. In the runner the router
at 8,192 rows takes 0.64 ms a layer against 15.1 ms as 16-row GEMVs.

**Prefill indexer: keys in registers, queries streamed** (`k_q_stage` + `k_score_pf`). A CTA holds 256 keys as
MMA A fragments (8 warps × 2 tiles, `k_score`'s lane map) and streams up to 256 rows' RoPE'd queries (staged once
a row by `k_q_stage`) through two cp.async buffers; each row runs `k_score`'s MMA chain and head sum, so the
scores equal decode's bit for bit (test E: decode calls of 1/6/16 rows = the prefill rows, scores and top-2048).
Selection at prefill: a 1-in-16 sample of first-level bins (bits 31..19) counted while scoring; `k_sel_thresh`
picks the bin where the sample reaches 256 (~4096 candidates); `k_sel_compact` (one launch) compacts candidates
at or above it per 4096-key chunk and its last CTA selects the exact top-2048 from them (three histogram levels
in registers), or from the scores when the sample missed (fewer than 2048 candidates) or the candidates
overflow the cap (8192; tested with caps 0 and 2100 = the default bit for bit). Launches a call: 4 (was 6).

| Chunk of 2048 rows ending at | Before (ms, TFLOPS of score MMAs) | After, calls of 2048 rows | calls of 512 |
|---|---:|---:|---:|
| 32k | 25.9 (20.6) | **10.1 (52.7)** | 11.2 |
| 128k | 105.8 (20.6) | **33.8 (64.6)** | 35.3 |
| 256k | ~210 (not run) | **66.1 (66.3)** | 66.2 |

TFLOPS count the whole call (selection included) against the score MMAs: 62 % of the 107 TFLOPS bf16 peak at
256k, 49 % at 32k (target 50 %). nsys of a near-final build (calls of 512 rows at 32k and 128k): `k_score_pf`
6.8 ms a launch on average, `k_sel_compact` 1.7, `k_sel_thresh` 0.33, `k_q_stage` 0.07.

**Decode GEMVs.** The launch shape is now chosen by shape (`gemv_config` -1, bit-neutral: test B runs every
configuration against the default): 16 loads in flight a lane for the grouped W_UV, 8 warps a CTA elsewhere.
The x-projection can carry the index weights in its launch (`GemvIw`: extra CTAs run `k_index_weights`'s
arithmetic, 128 threads doing 256 threads' sums in the same order; test B: = the two launches bit for bit).

| Projection (% of 235 GB/s, 1 / 6 rows, PDL chain) | Before | After |
|---|---:|---:|
| x-projection 688 × 6144 | 85.3 / 85.2 | 86.4 / 86.1 |
| x-projection + index weights (two launches → one) | 22.5 / 24.3 µs | **21.0 / 21.2 µs** |
| W_UK grouped | 98.7 / 98.5 | 102.0 / 103.1 |
| W_UV grouped | 81.6 / 82.8 | **103.3 / 101.5** |
| shared gate+up 1024 × 6144 | 91.6 / 93.0 | 93.2 / 92.6 |
| shared down 6144 × 512 | 95.6 / 98.9 | 98.8 / 97.9 |

The x-projection stays at ~86 %: every configuration (P 8/16/24, 2/4/8 warps) lands at 82–86 %, the limit
being its 86 tiles of 8 rows for 48 SMs; fusing the index weights hides their launch instead. Shared gate+up
stays at ~93 % (best configuration 93.7 %).

**Decode index-key scan.** `k_score`'s key loads have variants (`index_score_config`; tested equal for every
variant, BF16 and FP8, 1/6/16 rows): 0, each lane its own 16-byte pieces (half of each 32-byte sector an
instruction, L1 bypassed); 1, the same through L1; 2, lanes t and t ^ 1 each load whole sectors (their own piece
and the partner's) and swap halves by shuffle when the tile is used; 3, 2 with two tiles in flight; 4, 0 with two
tiles in flight. Default: 2 for BF16 keys, 0 for FP8 (2 costs FP8 40–90 %: 320 vs 174 µs at 256k).

| Context, keys, rows (replicated µs, GB/s of keys) | Before (base, same job) | After |
|---|---:|---:|
| 32k BF16 1 / 6 | 89.2 / 109.9 | **69.3 / 91.2** |
| 128k BF16 1 / 6 | 300.2 (112) / 345.0 | **242.2 (139) / 276.0** |
| 256k BF16 1 / 6 | 585.7 (115) / 664.1 | **464.7 (144) / 530.4** |
| 32k FP8 1 / 6 | 40.0 / 62.3 | 40.3 / 62.5 |
| 128k FP8 1 / 6 | 92.3 / 157.9 | 94.5 / 158.0 |
| 256k FP8 1 / 6 | 171.1 (196) / 293.5 | 173.0 (194) / 292.0 |

(Second run of the new build after the base, same job: within 3 % except one 256k BF16 6-row sample at 675 µs.)
BF16 reaches 144 GB/s, short of the 200 GB/s target.

**Decode selection: not merged.** The five parallel passes stay at decode. Tried: the first-level histogram (bits
31..19, 8192 bins) counted in `k_score` with warp-aggregated atomics, then one `k_sel_compact` launch (the exact
bin from the full histogram, candidates compacted, the last CTA selecting): one launch instead of five, but
slower in every case, e.g. 32k FP8 1 row 45.4 vs 40.0 µs, 256k FP8 6 rows 335 vs 291 µs (A/B in one job).
nsys: the merged pass took 21–23 µs at 32k (the five passes ~17 µs together under PDL) and 74 µs at 256k ×
6 rows (60). Skipping the last CTA's selection (a timing-only build) took 13–17 µs off, so the cost is that
serial tail; also tried in it: warp-aggregated shared-memory atomics (+2 µs: `__match_any_sync` costs more
than the contention it saves), and ranking the threshold bin's candidates pairwise (O(M²), +6 µs at 32k, more
at 6 rows: the bin holds up to ~2000 candidates). The histogram in `k_score` cost 1–13 µs on its own. Prefill,
with thousands of rows a call, keeps the merged compaction (the tail is per row and runs in parallel).

ncu is not available (GPU performance counters are not permitted for this user), so kernel-level attributions
above come from nsys timelines and A/B builds only.

### 2026-10-14 (T6): prefill sparse attention, `k_attn_pf`

Rank 0's GB10 through `ring.sh`, 2026-10-14 03:18–05:29 (+0100); final bench and tests on sources sha256
`1178708d35bca344` (`tools/run.sh`'s hash). `bench_attn --only pattn`: 8,192 rows at positions 24,576–32,767 of a
32,768-entry cache, 2,048 keys a row in three patterns (`rand`: sorted random 2,048 of 0..p; `win`: random 2,048
of the last 4,096; `same`: every row the same list), and the dense rows 0..2,047. A graph of one launch, median of 7.

**Where `k_attn`'s time goes.** Its FLOPs (584 GFLOP a layer at 8,192 rows) take 5.5 ms at 107 TFLOPS; its time
is 28–33 ms. A kernel that only gathers the same entries in `k_attn`'s shape (48 CTAs' worth of 64-key steps, each
waited for) takes 32–34 ms on `rand`: the gathers alone are `k_attn`'s time, latency-bound, not bandwidth-bound.
With more gathers in flight the same bytes move faster: 32 keys a step on 96 CTAs 17.4–18.9 ms, 16 keys on 192
CTAs 11.2–11.8 ms (1.6–1.7 TB/s, far above DRAM bandwidth: 19.3 GB a layer gathered from a 37.7 MB cache, mostly L2 hits). Shared memory
(101,376 B opt-in) admits one 74 KB block an SM, so the gathers have to be overlapped inside the CTA.

**The kernel.** `k_attn_pf` (sparse_attn.cu): 8 warps; a block's 64 keys lie in 4 slots of 16 keys. Warp s (0–3)
computes S on slot s exactly as `k_attn`'s warp s (the same keys, MMAs, masking, reductions); the softmax state is
`k_attn`'s; P·V is split over the 8 warps by 64 latent dims, each output the same chain of MMAs over the block's 4
key slices. Each warp copies its P·V operands of a block (its 64 dims of the 64 keys, `k_attn`'s `ldmatrix.trans`
fragments: 64 registers) out of shared memory as soon as each slot lands and then releases the slot (mbarrier
`empty[s]`, 8 arrivals); warp s then gathers the next block's slice into it, 16 bulk copies (`cp.async.bulk`, one
1,152-byte entry each; masked keys from a zero entry) completing on mbarrier `full[s]`. The next block's gathers fly
under the other slots' S, the softmax and P·V, with no CTA-wide wait between a slot's landing and its regathering.
Every output is `k_attn`'s bit for bit. `sparse_attn_config(0)` selects `k_attn` again; the model's `kTunePfAttn`
(0x100, in `kTuneDefault`) does that for prefill.

| 8,192 rows at 24,576 (ms, A B A B) | `k_attn` | `k_attn_pf` | speed |
|---|---:|---:|---:|
| `rand` | 32.34 / 33.39 | 29.74 / 30.71 | 1.09× |
| `win` | 28.15 / 28.15 | 21.88 / 21.95 | 1.28× |
| `same` | 29.47 / 28.82 | 22.13 / 21.98 | 1.32× |
| dense rows 0..2,047 | 3.71 / 3.72 | 2.72 / 2.69 | 1.37× |

All four with the same bits as `k_attn`. In the model's prefill (`pftrace`, 8,192 rows at 24,576, two pieces
beside the ring's collectives) the attention kernel went 3,355 → 2,990 ms a chunk on rank 0 (3,096–3,355 →
2,788–2,990 on the four ranks): 11 %, not the bench's 9–37 %, because beside the ring's copies every gather is
slower (21.5 ms a 4,096-row piece for `k_attn` there against ~15 alone).

**A race found on the way.** The first `k_attn_pf` gave the same bits as `k_attn` in every bench but not always in
the model's overlapped prefill: one row in a few thousand, a band of 32–128 output dims of all 16 heads (one warp's
P·V fragments), always a dense row. Inputs hashed after the fact were the same, and rerunning the kernel on them
gave the right bits. The cause: a warp's `mbarrier.arrive` on `empty[s]` issued before its `ldmatrix` loads of the
slot had returned (the registers are first used at P·V, much later), so the regather's bulk copy could land first.
The arrive now takes the XOR of one register of each of its four loads as an operand (it waits for them), and the
regathering warp issues `fence.proxy.async.shared::cta` after its wait. After the fix: the model's `pfab` the same
bits as `k_attn` in every run (below and docs/engine.md). A bench that runs the attention beside copy kernels
(`pstress`, low and high priority streams, copies arriving mid-launch) never reproduced the race, before or after.

**Variants tried** (all bit-identical to `k_attn` unless said; `rand` / `win` / dense ms):

| Variant | ms | |
|---|---|---|
| cp.async, 2 warps a slot (the first version) | 44 / – / – | slower; its bits differed (a key-position staging bug, fixed in the next) |
| bulk copies, 8 warps, named barriers per slot | 29.6 / 22.5 / 3.05 | |
| cp.async by all warps | 30.5 / 26.2 / 3.43 | |
| cp.async by the S warps, mbarrier no-increment arrivals | 29.5 / 26.1 / 3.42 | |
| + L2 prefetch of the next block | 31.3 / 29.0 / – | slower |
| all operands in registers, S split over 8 warps, cp.async | 41.7–56.8 / 54.5–55.1 / 6.44 | slower (the first build 54–57 ms on every pattern, a later one 41.7 on `rand`) |
| the same with bulk copies | 32.9 / 30.4 / 2.92 | |
| **kept: bulk copies, V operands in registers, full / empty mbarriers per slot** | 28.0–30.7 / 21.6–22.0 / 2.66–2.81 | |

A gather-only ring of 4 slots of 16 keys a CTA (no compute) moves the `rand` entries in 19.6–26.8 ms by either
copy kind and `win`/`same` in 11.7–12.2 (cp.async) or 22.6–22.9 ms (bulk copies, which top out near 850 GB/s);
the compute alone of the kept structure takes 12.3–12.5 ms. So `rand` stays bound by its gathers.

## 6. What remains

- **Prefill projections** (2026-10-14: GEMM at 48–60 % of 107 TFLOPS on the large shapes): the last points to
  60 % on the x-projection and shared gate/up (split-K 4 shapes) and dense gate/up; the small-K projections
  (W_UK, W_UV, shared down) need a fused attention-side kernel rather than a faster GEMM.
- **Prefill sparse attention** (T6: `k_attn_pf`, 1.09–1.37× `k_attn`): random selections over a long context are
  still bound by gathers in flight (one 74 KB CTA an SM). More in flight needs less shared memory a CTA (e.g. the
  V operands streamed from L2 rather than staged) or two rows a CTA sharing their keys where selections overlap.
- **Prefill indexer** at 49–62 % of peak: the remaining time is the score MMAs; a further step is keeping two
  key tiles a warp to raise MMA issue rate.
- **Small-GEMV tails:** x-projection at ~86 % in every launch shape (86 tiles for 48 SMs); its index weights now
  ride in its launch. Shared gate/up at ~93 %.
- **BF16 index-key scan** at 144 GB/s (was 115; FP8 at ~195): short of 200; the next candidate is a
  shared-memory staged key ring (cp.async, coalesced) feeding the fragments.
- **Decode selection:** still 5 passes; the merged single launch was slower (§ 5, 2026-10-14). A merged pass
  needs a parallel final step (not one CTA) to win.
- **G4:** the all-gathers (x-projection outputs, wq_b output, merge candidates) and the o_proj / down
  all-reduces; CUDA graphs over a whole layer; the position-split cache (DCP) if memory requires it; an FP8
  latent-cache track was not built.
- **Converter:** `tools/pack.py` is the reference for every tile; the real converter (zai's checkpoint →
  these tiles) is the G2 tool's job and must reproduce `pack.py` byte for byte (the FP8 quantiser now agrees,
  `tools/fp8_vs_converter.py`).
