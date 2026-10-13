# The attention side (milestone G3, attention half)

What `engine/attn/` computes for one rank at TP=4, the layouts it reads and writes, the per-rank split, how it
was tested and what it measured. Written 2026-10-13. The math is [model.md § 2](model.md#2-the-forward-pass);
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
| 2 | `gemv` x-projection (gemv.cu) | 688 rows (656 shared layers) of FP8, split-K 4, bf16 out; then the all-gather (G4) |
| 3 | `index_weights` (prep.cu) | BF16 `weights_proj`, the rank's 8 heads, f32 out × 32^-0.5; all-gathered with the x-projection |
| 4 | `attn_prep` (prep.cu) | q_a_layernorm → qr; kv_a_layernorm + RoPE of k_rope → the cache entry; indexer key LayerNorm + RoPE → BF16 (or FP8 + scale) key |
| 5 | `gemv` q_b + indexer wq_b (one launch, two segments) | 4096 + 1024 rows, K 2048 |
| 6 | `gemv` W_UK^T grouped | per head: q_nope (192) → q_lat (512); x rows grouped by head |
| 7 | `index_select` (indexer.cu) | `k_score` then 5 selection passes → top-2048 positions (or a quarter's candidates + `index_merge`) |
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
h's q_nope or latent). Results are independent of the number of rows (tested 1/6/16 bit-equal).

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
row; `k_attn_combine` merges (m, l, O) in split order. Shared memory 96.5 KB, one CTA an SM.

**Prefill** runs the same kernels: the projections as 16-row GEMVs (a placeholder: a tensor-core GEMM path for
prefill projections is G4 work), `k_score` with RG = 8 (tensor cores, 20.6 TFLOPS), the selection passes with
one grid row a query row, and `k_attn` with one row a CTA set. Chunking a prompt does not change any bit
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

## 6. What remains

- **Prefill projections:** the projections run as 16-row GEMV launches; a tensor-core GEMM over the chunk
  (FP8 widened to bf16, the DeepSeek engine's `gemm_woa` shape) is needed before prefill rates mean anything
  end to end.
- **Prefill indexer at 20 % of bf16 peak** (107 TFLOPS): stage keys in shared memory and keep query fragments
  in registers (one ldmatrix a key tile shared by all rows) instead of a shared-memory query load per MMA.
- **Small-GEMV tails:** x-projection (85 %), W_UV (81 %), shared gate/up (93 %) under 95 %; candidates: a
  persistent kernel running several projections back to back, or fusing W_UV into the attention combine.
- **BF16 index-key scan** at ~110 GB/s (FP8 at 203): register pressure of the 16-word tile; FP8 keys are the
  measured option (overlap 0.993) if model quality allows (G4 decision with the end-to-end check).
- **Selection launches:** 5 passes a call (~10 µs of the 1-row total); fusing into a single persistent pass is
  bit-neutral.
- **G4:** the all-gathers (x-projection outputs, wq_b output, merge candidates) and the o_proj / down
  all-reduces; CUDA graphs over a whole layer; the position-split cache (DCP) if memory requires it; an FP8
  latent-cache track was not built.
- **Converter:** `tools/pack.py` is the reference for every tile; the real converter (zai's checkpoint →
  these tiles) is the G2 tool's job and must reproduce `pack.py` byte for byte.
