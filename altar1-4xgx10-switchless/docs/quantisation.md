# The per-rank budget (2026-10-14)

What AikidoSec/altar-1 costs a rank at TP=4 in the glm53 engine's layout, what is left of a box for the KV cache and
buffers, the decode bandwidth floor, and where the checkpoint and the rank files can live. Unlike glm53 there is no
format to choose: the checkpoint is already quantised (compressed-tensors INT4 for the routed experts and three
attention projections of layers 4–76, BF16 everything else), so this page is only the budget. CPU work only: the 39
shard headers, fetched by HTTP range request, and two tensors read from rank 0's partial copy (§ 6).

Source: `AikidoSec/altar-1` at revision `5d591cd98958e4e7e429517887e80ca3bb7ce2c4`, 39 shards, 150,554 tensors,
**305.42 GiB** (327.9 GB) as stored. Paths are relative to the repository root; `glm53/` stands for
`glm53-4xgx10-switchless/`.

```
python3 -I tools/tally.py DIR [--json OUT]    # DIR: the shards, or their header prefixes saved as *.safetensors.hdr
```

`tools/tally.py` is glm53's (`glm53/tools/tally.py`) with the INT4 dtypes (I32, I64), eight parameters a packed word,
the INT4 parts (values, scales, zero points) kept apart, header-prefix input, and the decode read a token per class.
Numbers marked *estimate* are not measured.

## Summary

- **76.47 GiB a rank** of tensors at TP=4 in glm53's split, **~76.6 GiB** as a rank file (with the 0.125 GiB RoPE
  table and the TOC), against glm53's 101.37 GiB NVFP4 fill and its 87 GiB budget. The INT4 experts are 62.32 GiB of
  it, the BF16 experts of layers 3, 77 and 78 another 8.86.
- **Room.** Against glm53's measured state at 101.4 GiB (MemAvailable 6.7–11.9 GiB after the weights and the prefill
  buffers), altar-1 leaves ~31–37 GiB before the KV pool: one 256k session in BF16 (23.6 GiB) with ~8–13 GiB to
  spare, or a 12 GiB pool with ~20–25 GiB available, the DeepSeek service's state.
- **Decode floor: 8.55 GB a token a rank → 36.4 ms at 235 GB/s**, against glm53's 8.04 GB and 34.2 ms
  (measured 43.6 ms, 1.27×). The INT4 experts read what glm53's NVFP4 ones did; the extra half gigabyte is the BF16
  shared experts, dense MLPs and q_a, partly offset by the INT4 o_proj. At glm53's measured overhead the step would
  be **~45.8 ms** (*estimate*, ~22 tok/s before drafting).
- **Disk.** The converter needs the checkpoint local (327.9 GB) and writes ~82.3 GB a rank. Rank 3 can do it all
  (peak ~492 GB of its 694). Rank 0's data disk fits its file (117 GB free) only with 35 GB left; it already holds
  32 of the 39 shards of this revision (257 GiB), which is the space to reclaim.

## 1. Tensor classes (from the shard headers)

`tools/tally.py` over the 39 headers (2026-10-14). The INT4 tensors are compressed-tensors `pack-quantized`
(`quantization_config.config_groups.group_0`: int, 4 bits, group 32, `strategy: group`, `symmetric: false`,
`zp_dtype: torch.int8`, `scale_dtype: null`):

| Part | Name | Dtype | Shape | Bytes a weight |
|---|---|---|---|---:|
| values | `weight_packed` | I32 | [N, K/8]: eight 4-bit values a word, k = 8j + i at bits 4i | 0.5 |
| scales | `weight_scale` | **BF16** (checked in every header) | [N, K/32] | 0.0625 |
| zero points | `weight_zero_point` | I32 | [N/8, K/32]: eight 4-bit zero points a word along N | 0.015625 |
| shape | `weight_shape` | I64 | [2] | metadata, not in the rank files |

That is **0.578125 B a weight** (4.625 bits), against NVFP4's 0.5625. Which tensors are INT4: exactly the routed
experts' `gate_proj`, `up_proj`, `down_proj` and the attention's `q_b_proj`, `kv_b_proj`, `o_proj` of **layers
4–76** (73 layers). Layers 0–3, 77 and the MTP layer 78 are BF16 throughout, as is every `q_a_proj`,
`kv_a_proj_with_mqa`, indexer tensor, shared expert, dense MLP, router, the head and the embedding. No
`fused_qkv_a_proj` tensor exists, despite the name in the config's targets and ignore list.

Per rank: the split of glm53's rank files ([glm53 layout § 3](../../glm53-4xgx10-switchless/docs/layout.md#3-the-plan)):
attention by heads (q_b rows, kv_b per head, o_proj columns), q_a and kv_a and the indexer's `wq_b`, `wk`,
`weights_proj` by rows, experts (routed and shared) and dense MLPs by intermediate channel (gate/up rows
[512r, +512), down columns [512r, +512)), the head and the embedding by vocabulary (38,720 rows), and whole on every
rank only the norms, the routers with their bias, and the indexer's `k_norm`. Every INT4 cut is group-aligned: a
rank's down slice is groups [16r, +16) and packed columns [64r, +64); gate/up rows [512r, +512) are zero-point rows
[64r, +64); o_proj's K slice [4096r, +4096) is 128 groups.

| Class | Tensors | Params (M) | Stored GiB (values / scales / zero points) | Split | **GiB a rank** | Read a token a rank, GB |
|---|---:|---:|---|---|---:|---:|
| routed experts INT4, layers 4–76 (73 × 168) | 147,168 | 462,950.5 | 249.26 (215.58 / 26.95 / 6.74) | ÷4 | **62.316** | 3.186 (8 of 168) |
| routed experts BF16, layers 3 and 77 (2 × 168) | 1,008 | 12,683.6 | 23.63 | ÷4 | **5.906** | 0.302 (8 of 168) |
| MTP layer 78's routed experts BF16 (168) | 504 | 6,341.8 | 11.81 | ÷4 | **2.953** | draft pass only |
| attn o_proj INT4, layers 4–76 | 292 | 7,348.4 | 3.96 (3.42 / 0.43 / 0.11) | ÷4 | 0.989 | 1.062 |
| attn q_b_proj INT4 | 292 | 2,449.5 | 1.32 (1.14 / 0.14 / 0.04) | ÷4 | 0.330 | 0.354 |
| attn kv_b_proj INT4 | 292 | 1,071.6 | 0.58 (0.50 / 0.06 / 0.02) | ÷4 | 0.144 | 0.155 |
| attn q_b, kv_b, o_proj BF16, layers 0–3, 77 | 15 | 744.5 | 1.39 | ÷4 | 0.347 | 0.372 |
| attn q_a_proj BF16, layers 0–77 | 78 | 981.5 | 1.83 | ÷4 | 0.457 | 0.491 |
| attn kv_a_proj_with_mqa BF16 | 78 | 276.0 | 0.51 | ÷4 | 0.129 | 0.138 |
| indexer BF16, 21 full layers (`wq_b`, `wk`, `weights_proj`, `k_norm`) | 105 | 196.8 | 0.37 | ÷4 (k_norm whole) | 0.092 | 0.098 |
| shared expert BF16, layers 3–77 | 225 | 2,831.2 | 5.27 | ÷4 | 1.318 | 1.416 |
| dense MLP BF16, layers 0–2 | 9 | 679.5 | 1.27 | ÷4 | 0.316 | 0.340 |
| router [168, 6144] BF16 + bias [168] F32, layers 3–77 | 150 | 77.4 | 0.14 | whole | 0.144 | 0.155 |
| head [154880, 6144] BF16 | 1 | 951.6 | 1.77 | ÷4 (vocabulary) | 0.443 | 0.476 |
| embedding BF16 | 1 | 951.6 | 1.77 | ÷4 (vocabulary) | 0.443 | one row |
| norms | 313 | 1.2 | 0.002 | whole | 0.002 | 0.002 |
| MTP: attention, indexer, shared expert, router, `eh_proj`, norms (BF16) | 23 | 288.6 | 0.54 | ÷4 (norms, router whole) | 0.136 | draft pass only |
| **all** | **150,554** | | **305.42** | | **76.465** | **8.547** |

`weight_shape` (37,011 tensors × 16 B) is left out of the rank column. Totals by precision a rank: INT4 63.78 GiB,
BF16 12.68 GiB (of which experts 8.86).

**How the rank file adds up** (*estimate* for the TOC and padding):

| | GiB a rank |
|---|---:|
| tensors (above) | 76.465 |
| RoPE table (`rope.table`, generated, as glm53) | 0.125 |
| TOC (~15,000 entries: 76 expert layers × 169 entries, ~2,100 others; 512 B each) and 4 KiB padding | ~0.035 |
| **rank file** | **~76.6** (~82.3 GB) |
| if kv_b's nope half (W_UK) is kept BF16 ([reuse § 3](reuse.md#3-attention-side)) | +0.15 |
| if the MTP experts go to NVFP4 by glm53's `nvfp4_rtn` (drafts only, [reuse § 5](reuse.md#5-runner-step-prefill-server-mtp)) | −2.12 (0.83 instead of 2.95) |

The INT4 entries cost the same in the kernels' tiles as in the checkpoint: the proposed expert block (8 rows ×
512 k: 2,048 B of values, 256 B of scales, 64 B of zero points, 2,368 B, [reuse § 2](reuse.md#2-the-expert-formats))
is 0.578125 B a weight, and a matrix is a whole number of 4 KiB pages (1,818,624 B = 444 × 4,096).

Against glm53's rank file:

| GiB a rank | glm53 (C1, all NVFP4) | altar-1 |
|---|---:|---:|
| routed experts, main layers | 94.92 (75 × 256 NVFP4) | 68.22 (73 × 168 INT4 + 2 × 168 BF16) |
| MTP layer's experts | 1.27 (NVFP4, our RTN) | 2.95 (BF16 as shipped) |
| everything else (with the RoPE table and TOC) | 5.18 (FP8 attention, shared, dense) | 5.45 (BF16 except 3 INT4 projections) |
| **file** | **101.37** | **~76.6** |

## 2. KV cache a position

The same as glm53's, since the attention is unchanged ([glm53 serving § 4.1](../../glm53-4xgx10-switchless/docs/serving.md#41-the-device-pool),
`Model::cache_pos_bytes`, `glm53/engine/model/model.cu:583-588`): replicated on every rank, the BF16 latent and rope
key (1,152 B) on 79 layers (78 and the MTP layer) and the BF16 indexer key (256 B) on 22 layers: **96,640 B a
position**, 94.4 MiB a page of 1,024, **23.59 GiB a 256k session**. With the FP8 cache (not built): 54,728 B,
13.4 GiB a 256k session.

## 3. What is left of 121 GiB

**From glm53's budget arithmetic** ([glm53 quantisation § 1.2–1.3](../../glm53-4xgx10-switchless/docs/quantisation.md#12-glm-53s-non-weight-memory)):
121.62 GiB less 10.0 of system on rank 0 (7.3 on the workers), a margin of 4 (2), ~7 of engine buffers at 8,192-row
prefill chunks (*estimate*; ~5 at 4,096):

| Weights a rank | Pool a rank (rank 0 binds) | Positions (BF16) | 256k sessions (BF16) |
|---|---:|---:|---:|
| glm53's 101.4 GiB | none (−0.7) | 0 | 0 |
| glm53's 87 GiB budget | ~13.6 | ~151k | 0 |
| **altar-1, ~76.6 GiB** | **~24.0** | **~267k** | **1** (23.59 of 24.0) |
| altar-1 with NVFP4 MTP experts, ~74.5 GiB | ~26.1 | ~290k | 1 |

**From glm53's measurement** (the stronger anchor): with the 101.36 GiB files, G8 measured free device memory
2.05–9.56 GiB and MemAvailable 6.73–11.92 GiB on the four ranks after the weights and the prefill buffers (2,048-row
chunks, `glm53/docs/engine.md` G8; `glm53/docs/serving.md` § 5.1). The same engine with 24.8 GiB less weight
leaves **~31.5–36.7 GiB MemAvailable before the pool**, so the pool's offer (the smallest rank's, less its 4 or 2 GiB
margin) is ~27 GiB: about 300k BF16 positions (*estimate*: buffers grow with 8,192-row chunks by ~2 GiB).

| State | MemAvailable left a box | Like |
|---|---:|---|
| DeepSeek service: 72.4 GiB of weights, a 12 GiB pool | 21.4–23.2 (measured) | |
| altar-1 with a 12 GiB pool (~133k positions) | ~19.5–24.7 | the DeepSeek service: no oom guard needed beyond its margins |
| altar-1 with one 256k session's pool (23.6 GiB) | ~8–13 | glm53's G8: keep earlyoom's victim order (`oom_score_adj` 1000 on the ring job), `--min-free-mib`, and CPU passes on a ring box small |
| glm53 G8, no pool | 6.7–11.9 (measured) | |

So altar-1 runs where glm53 could not start a pool: one 256k session, or several agentic sessions of tens of
thousands of positions, in BF16 cache; the FP8 cache or a position split are not needed for a single 256k session.
A 256k session plus a 12 GiB margin for the agents on rank 0 does not both fit; the pool cap (`--kv-gib`) is the
user's trade.

## 4. Decode bytes a token and the floor

One row, every weight of layers 0–77 read once, the routed experts at 8 of 168 a layer, the head whole (its rank's
vocabulary), the routers whole (they are whole on every rank), the embedding one row, the MTP layer not at all; the
same accounting as glm53's floor (`Model::step_bytes`, `glm53/engine/model/model.cu:673-688`, which counts weights
only, no KV). Per expert a rank: 3 × 2048 × 6144 / 4 = 9.437M weights.

| GB a token a rank | glm53 (NVFP4 experts, FP8 rest) | altar-1 | altar-1 arithmetic |
|---|---:|---:|---|
| routed experts | 3.185 | **3.488** | 8 × 73 × 9.437M × 0.578125 = 3.186; 8 × 2 × 9.437M × 2 = 0.302 |
| attention projections | 3.219 (FP8) | **2.572** | o_proj INT4 1.062, q_b INT4 0.354, kv_b INT4 0.155, the three BF16 on layers 0–3 and 77 0.372, q_a 0.491, kv_a 0.138 |
| indexer | ~0.05 (FP8 wq_b) | 0.098 | 21 × (4096 × 2048 + 160 × 6144) / 4 × 2 |
| shared experts | 0.708 (FP8) | **1.416** | 75 × 9.437M × 2 |
| dense MLPs | 0.170 (FP8) | **0.340** | 3 × 3 × 12288 × 6144 / 4 × 2 |
| routers | 0.236 (256 rows) | 0.155 | 75 × 168 × 6144 × 2 |
| head | 0.476 | 0.476 | 38,720 × 6,144 × 2 |
| **total** | **8.04** | **8.547** | |
| **floor at 235 GB/s** | **34.2 ms** | **36.4 ms** | |
| measured | 43.6 ms (1.27×) | ~45.8 ms (*estimate*: glm53's 9.4 ms of collectives, attention and launch tails on top) | |

- **What moved.** The INT4 experts cost what glm53's NVFP4 ones did (eight experts a layer read either way; the
  pruning to 168 changes only the union at many rows). BF16 shared experts (+0.71 GB), dense MLPs (+0.17) and
  q_a/kv_a/indexer (+0.33 net) cost more than glm53's FP8; INT4 o_proj and q_b (−1.2 against FP8) cost less.
- **Several rows.** Each extra verified row adds its new distinct experts at 5.46 MB a rank each (INT4) and 18.9 MB
  (BF16 layers): up to 3.5 GB a row at no overlap, ~3.0 GB at 7 new a layer, against glm53's ~2.6. With 168 experts
  the rows' picks overlap more than with 256 (*to measure*), which pulls the other way.
- **A draft pass** (layer 78 at 8 of 168 BF16 experts, its attention, shared expert, `eh_proj`, and the head):
  0.773 GB, **3.3 ms**; 0.664 GB (2.8 ms) with NVFP4 MTP experts. glm53 measured 3.6–4.0 ms a pass on top of the step.
- **Exact levers** (outputs unchanged; the user's rule, [reuse § 9](reuse.md#9-build-plan)): 3.79 GB of the
  8.55 are BF16. The DeepSeek engine's lossless BF16 coding (`ds41f/engine/kernels/gemv16.cu` `kBF16X`, ~¾ of the
  bytes on its head, bit-identical) would save ~0.12 GB (0.5 ms) on the head alone, and up to ~0.9 GB (~4 ms) if
  the shared experts, q_a and dense MLPs code as well (*estimate*: their exponent entropy is not measured).
  Fusing the BF16 shared expert into the expert kernels (glm53 fuses its FP8 one) hides it behind the routed
  experts' PDL wait but reads the same bytes.
- **KV reads** come on top as in glm53: at 128k context the sparse attention reads 2,048 × 1,152 B × 79 ≈ 0.19 GB
  and the indexer scan 128k × 256 B × 21 ≈ 0.70 GB a token if every rank scans every position (BF16 keys, not
  position-split; [glm53 attention.md](../../glm53-4xgx10-switchless/docs/attention.md)).

**Prefill** is unchanged in FLOPs (40B active a token, BF16 MMA at 107 TFLOPS peak): the INT4 weights dequantise
in the producers as NVFP4's did, and the BF16 shared and dense weights stream at 2 B a weight, still far under the
compute time at ≥ 2,048 rows. glm53's 1,106 / 1,145 tok/s at 2,600 / 8,192 tokens is the expectation.

## 5. Disk: the conversion and where the files live

| | Size |
|---|---|
| checkpoint (39 shards + small files) | 327.9 GB (305.42 GiB) |
| one rank file | ~82.3 GB (76.6 GiB) |
| four rank files | ~329 GB |
| the converter's working memory | one unit's sources plus one rank's outputs: an INT4 expert layer is 3.66 GB as stored, a BF16 one 12.7 GB, the embedding and head 3.8 GB (glm53's peak was 1.7–3.8 GB, [glm53 layout § 6](../../glm53-4xgx10-switchless/docs/layout.md#6-the-routed-experts-reserved-entries-and-the-fill)) |

The glm53 converter reads a **local** checkpoint (`glm53/engine/weights/sources.py` `Checkpoint`, `preadv` with
`POSIX_FADV_DONTNEED`) and writes every requested rank's file in one pass on one box; nothing reads across boxes
(`fetch.py`'s HTTP range reads were only for zai's BF16 dense MLPs, which altar-1 does not need). So the conversion
runs where the checkpoint is.

| Rank | Free | Holds | Plan |
|---|---:|---|---|
| 3 | 694 GB (before the download) | the full checkpoint downloading (327.9 GB) | **convert here**, nice 19: write rank 3's own file (82 GB), then each other rank's file in turn, copy it out and delete it: peak 328 + 82 + 82 = **~492 GB**. Also the natural box for the CPU reference ([reuse § 7](reuse.md#7-the-cpu-reference-ref)), with glm53's caveat that a CPU pass beside a ring job slowed rank 2's kernels 8–16 % |
| 2 | 478 GB | glm53's reference venv and golden set | its rank file (82 GB) |
| 1 | 429 GB | glm53's rank file | its rank file (82 GB) |
| 0 | 117 GB on the data disk, 92–94 GB on the root disk | glm53's `rank0.glw` (108.8 GB) and `.prefill-only` (6.9 GB); **a partial copy of this very revision: 32 of 39 shards (all but 28 and 34–39), 257 GiB, sizes equal to the revision's manifest and three headers byte-equal (SHA-256 not checked)**; glm53's master scratch (37 GB, deletable per glm53's README) | its rank file on the data disk: 82 GB of 117 leaves 35 GB, below the disk tier's 32 GiB margin, so no disk tier. Deleting the partial altar-1 copy once rank 3's is verified frees ~276 GB; deleting glm53's scratch 37 GB. The root disk would hold the file with ~10 GB left: not advised |

The partial copy on rank 0 is also a second source: with the seven missing shards (~53 GB) fetched, rank 0 could
convert, but it is the box with the agents and the least disk, so rank 3 is the better converter. Whether to delete
glm53's rank-0 file (glm53 is paused, not abandoned) is the user's call.

## 6. Checks made on the data

- **The +8 offset.** compressed-tensors stores signed 4-bit values and zero points with 8 added before packing, so
  `(nibble_w − nibble_z) · s` = `(q − z) · s` with the offsets cancelled. Read from rank 0's partial copy (one
  expert's `gate_proj` of layer 4 and layer 10's `o_proj`, `python3 -I`): the zero-point nibbles lie in 2–13 centred
  on 7.5, the value nibbles span 0–15 centred on 7.5 with the clipping ends (0, 15) raised. Two's-complement
  storage without the offset would put the mass at 0 and 15. So both carry the offset.
- **Scales** are BF16, positive, ~0.002–0.008 on those tensors.
- **The nibble order** (k = 8j + i at bits 4i, low nibble first, the same byte order as NVFP4's `[N][K/2]`) is
  compressed-tensors' `pack_to_int32` as published; it is checked against the library's `unpack_from_int32` on real
  tensors at milestone A0 ([reuse § 9](reuse.md#9-build-plan)), not here.
- **The numerics contract's weight** is `bf16((q − z) · s)`: compressed-tensors' `dequantize` computes
  `(x_q − zp) * scale` in the scale's dtype, BF16. `q − z` (|q − z| ≤ 15) is exact in BF16; the product is not
  always (up to 12 significant bits against BF16's 8), so the dequantised weight is the product rounded once to
  nearest-even. A kernel's `__hmul2` of the exact BF16 operands rounds the same way.
