# Per-rank weight files

Each box holds one `.dsw` file: its rank's slice of every tensor the text model uses, cut from that
box's own copy of the checkpoint (DESIGN.md decisions 1 and 10). Values keep the checkpoint's formats
(e4m3, packed e2m1, ue8m0, bf16, fp32) and nothing is re-quantised. Since format version 2 the
weights the decode GEMVs stream are stored in those kernels' tile layouts (L1–L3 below), produced by
the kernels' own tiling functions. Routed experts are stacked per layer. Code: `engine/weights/`.

| Rank | Box | File |
|---|---|---|
| 0 | rank 0 | `/data/dsgb10/weights/rank0.dsw` (an external USB4 SSD) |
| 1 | rank 1 | `~/dsgb10/weights/rank1.dsw` |
| 2 | rank 2 | `~/dsgb10/weights/rank2.dsw` |
| 3 | rank 3 | `~/dsgb10/weights/rank3.dsw` |

## What a rank holds

Every rank file is **72.38 GiB with 1,609 tensors**. Left out on purpose: the 32-layer ViT, the
aligner and the image tokens (vision is not planned), and the two Engram tables (188.8 GiB). The
tables stay in the kit's per-rank NVMe shards (docs/findings.md#fleet-state-that-the-engine-reuses).

| Component | GiB per rank | Note |
|---|---:|---|
| Routed experts, 40 layers | 67.236 | tensor-split, 576 of 2304 per rank |
| DSpark routed experts, 3 blocks | 1.681 | same split, 128 experts each |
| Embedding | 1.233 | replicated |
| Attention projections | 1.180 | wq_a, wkv, wq_b, wo_a, wo_b, sinks |
| Shared experts | 0.330 | tensor-split |
| Head | 0.308 | vocab-split |
| DSpark draft heads | 0.095 | main_proj split; Markov embedding replicated, Markov head vocab-split; confidence head |
| DSpark attention | 0.089 | split like the main layers |
| Engram `wkv` and gates | 0.073 | `wkv` split per segment |
| Indexer (8 layers) | 0.042 | replicated |
| mHC `hc_*_fn` | 0.037 | split by columns |
| Router | 0.037 | split by output |
| DSpark shared experts, router, mHC, norms | 0.029 | |
| Compressor (4 layers) | 0.009 | split by output |
| Norms | 0.001 | replicated |

By layout: L3 68.917 GiB (258 tensors), L1 1.679 GiB (353), L2 0.361 GiB (45), natural 1.422 GiB (953).

## Partition

`r` is the rank (0–3). "Rows" cuts dim 0 into four contiguous quarters and "cols" cuts dim 1 the
same way. Each rank's slice of a split tensor is a quarter of it. Every name below exists for
`layers.N.` (N = 0–39) and, unless marked, for the DSpark blocks `mtp.N.` (N = 0–2). The last
column is the tensor's layout in the file (next section); scales not listed are natural.

| Tensor | Checkpoint | Rank r holds | Why | Layout |
|---|---|---|---|---|
| `embed.weight` | bf16 [129280, 5120] | all of it | a lookup reads one 10 KiB row, so replication saves a collective per token | natural |
| `head.weight` | bf16 [129280, 5120] | rows [32320r, +32320) | vocab-split head with distributed sampling | L2 |
| `norm.weight` | bf16 [5120] | all of it | | natural |
| `attn_norm`, `ffn_norm`, `attn.q_norm`, `attn.kv_norm` | bf16 | all of it | applied after the all-gathers | natural |
| `attn.wq_a` | e4m3 [1280, 5120] | rows [320r, +320) | split by output; all-gathered with `wkv` | L1 |
| `attn.wkv` | e4m3 [512, 5120] | rows [128r, +128) | split by output | L1 |
| `attn.wq_b` | e4m3 [32768, 1280] | rows [8192r, +8192) = heads 16r … 16r+15 | column-parallel, as in the reference | L1 |
| `attn.attn_sink` | fp32 [64] | [16r, +16), the same heads | | natural |
| `attn.wo_a` | e4m3 [8192, 4096] | rows [2048r, +2048) = groups 2r, 2r+1 | the groups of those heads | L1 |
| `attn.wo_b` | e4m3 [5120, 8192] | cols [2048r, +2048) | row-parallel: partial sums, all-reduced | L1 |
| `hc_attn_fn`, `hc_ffn_fn` | fp32 [24, 20480] | cols [5120r, +5120) → [24, 5120] | copy r of the 4 × 5120 stream: 24 partial dot products plus a partial sum of squares per row, all-reduced as 25 floats | natural |
| `hc_*_base`, `hc_*_scale` | fp32 [24], [3] | all of it | | natural |
| `ffn.gate.weight` | bf16 [384, 5120] (DSpark: [128, 5120]) | rows [96r, +96) (DSpark: [32r, +32)) | split by output; scores all-gathered | L2 |
| `ffn.gate.bias`, `bias_vl` | fp32 [384] | all of it | top-k runs on the gathered scores | natural |
| `ffn.shared_experts.w1`, `w3` | e4m3 [2304, 5120] | rows [576r, +576) | tensor-split | L1 |
| `ffn.shared_experts.w2` | e4m3 [5120, 2304] | cols [576r, +576) | partial sums, added into the MoE all-reduce | L1 |
| `ffn.experts.E.w1`, `w3` | e2m1×2 [2304, 2560 B] | rows [576r, +576), stacked → [384, 576, 2560 B] | tensor-split, every rank reads a quarter of every selected expert | L3, values and scales |
| `ffn.experts.E.w2` | e2m1×2 [5120, 1152 B] | bytes [288r, +288) = columns [576r, +576), stacked → [384, 5120, 288 B] | | L3, values and scales |
| `attn.compressor.wkv`, `wgate` (layers 2, 8, 14, 20; `wgate` not in 20) | bf16 [512, 5120] | rows [128r, +128) | a quarter of the projections (the attention line's split) | natural |
| `attn.compressor.norm` (same layers) | bf16 [512] | all of it | | natural |
| `attn.indexer.wq_b` (layers 2, 8, 14, 20, 24, 28, 32, 36) | e4m3 [4096, 1280] | all of it | decode splits the scanned positions across ranks, not the heads, so every rank needs every head's query | L1 |
| `attn.indexer.weights_proj` (same layers) | bf16 [32, 5120] | all of it | the same | natural |
| `attn.indexer.wk`, `k_norm` (layers 2, 8, 14, 20) | bf16 [128, 512], [128] | all of it | | natural |
| `engram.wkv` (layers 1, 14) | e4m3 [25600, 6144] | rows [s·5120 + 1280r, +1280) for each segment s = 0…4 → [6400, 6144] | the output is 4 keys + 1 value of 5120 each; every rank takes the same quarter of each, so the gate's dot products reduce as partial sums and only the value needs gathering | natural |
| `engram.q_weight`, `k_weight` (layers 1, 14) | bf16 [4, 5120] | all of it | | natural |
| `mtp.0.main_proj` | e4m3 [5120, 15360] | rows [1280r, +1280) | 79 MB, too big to replicate; split by output | L1 |
| `mtp.0.main_norm`, `mtp.2.norm` | bf16 [5120] | all of it | | natural |
| `mtp.2.markov_head.embed` | bf16 [129280, 256] | all of it | lookups, like `embed` | natural |
| `mtp.2.markov_head.head` | bf16 [129280, 256] | rows [32320r, +32320) | the same vocab slice as `head`, so its bias adds locally | L2 |
| `mtp.2.confidence_head.proj` | bf16 [1, 5376] | all of it | | natural (one row does not fit L2's 8-row tiles) |

The DSpark blocks reuse the main `embed` and `head` (they have none of their own) and have no compressor
or indexer: they attend over the sliding window only.

### Scales follow their weights' blocks

FP8 weights have one ue8m0 per 32×32 block (`scale` is [⌈R/32⌉, ⌈C/32⌉]). MXFP4 experts have one per
row and 32 elements along K (`scale` is [R, C/32], with C counted in values, not bytes). A scale is cut
with the same rule as its weight, and the converter checks that every piece of every weight starts
and ends on a block boundary and that the scale's piece is exactly that piece divided by the block.
All cuts here are aligned:

| Weight | Cut (values) | Blocks per rank |
|---|---|---|
| `wq_a` rows 320, `wkv` rows 128, `wq_b` rows 8192, `wo_a` rows 2048 | multiples of 32 | 10, 4, 256, 64 scale rows |
| `wo_b` cols 2048 | multiple of 32 | 64 scale columns |
| shared `w1`/`w3` rows 576, `w2` cols 576 | 576 = 18 × 32 | 18 |
| expert `w1`/`w3` rows 576 | any row works (per-row scales) | 576 scale rows |
| expert `w2` cols 576 values = 288 bytes | 576 = 18 × 32 | 18 scale columns |
| `engram.wkv` 5 × 1280 rows | 1280 = 40 × 32, segments of 5120 = 160 × 32 | 5 × 40 scale rows |
| `mtp.0.main_proj` rows 1280 | 40 × 32 | 40 scale rows |

`gemv_woa` widens wo_a's FP8 to bf16 exactly only while every block scale is ≤ 2^7, i.e. every scale
byte ≤ 134. The converter stops if one is larger and `dsw_verify` checks again: the largest byte in
the 43 `wo_a` scale tensors is 119 on rank 0 and 118 on ranks 1–3.

## Tile layouts (version 2)

The layouts are fixed by engine/kernels (its README, "Weight layouts the per-rank files should
hold"). `shape` in the TOC stays the natural, logical shape; the TOC's `layout` field says how the
bytes are arranged:

- **L1, FP8 dense:** e4m3 [N][K] with N % 8 = 0 and K % 64 = 0 becomes [N/8][K/64][8 rows][64 B]:
  element (n, k) moves to byte `((n/8)·(K/64) + k/64)·512 + (n%8)·64 + k%64`. Scales stay natural.
- **L2, bf16:** bf16 [N][K] viewed as bytes [N][2K], then L1 with K' = 2K.
- **L3, MXFP4 routed experts:** each expert's matrix is tiled on its own, values and scales alike, so
  expert e still starts at e times one expert's size (w1/w3 at `e·576·2560` bytes, scales at
  `e·576·160`; w2 at `e·5120·288`, scales at `e·5120·18`). With P = K/64, n = 16·n16 + 8·h + g,
  k = 64·p + 32·l + 8·t + i and j = 2h + l, the 4 value bytes `[n][32p + 16l + 4t, +4)` go to
  `((n16·P + p)·32 + 4g + t)·16 + 4j` and scale `[n][2p + l]` goes to `((n16·P + p)·8 + g)·4 + j`.

| Layout | Tensors (main layers and all 3 DSpark blocks) |
|---|---|
| L1 | `wq_a`, `wkv`, `wq_b`, `wo_a`, `wo_b`, shared `w1`/`w3`/`w2`, indexer `wq_b`, `mtp.0.main_proj` |
| L2 | `head`, router `gate`, `mtp.2.markov_head.head` |
| L3 | routed experts `w1`, `w3`, `w2`, values and scales |

Every DSpark counterpart of these fits its tiles. Two DSpark-only tensors are tiled because they feed
the same GEMVs: `main_proj` (L1, an FP8 GEMV) and the Markov head (L2, since `gemv_bf16` reads only
that layout). `confidence_head.proj` is the one that does not fit: it has one row, and L2 needs
multiples of 8. Everything the attention and Engram kernels read (compressor, `weights_proj`,
indexer `wk`, Engram `wkv`, mHC) stays natural, as do lookups (`embed`, Markov `embed`).

**The converter does not re-implement the tiling.** It runs engine/kernels' own `dsk::tile_fp8` and
`dsk::tile_fp4` on the GPU, one tensor at a time through fixed device buffers. Every rank tiles
70.96 GiB, and on ranks 2 and 3 conversion takes no longer than without tiling. The verifier un-tiles
with `engine/weights/layout.h`, a host implementation written from the README's formulas, so the
two directions are independent code. `dsw_kcheck` connects them to the kernels; see below.

### Settled since version 1

- **Compressor:** split by output, rows [128r, +128) of each [512, 5120] (the attention line).
- **Indexer:** stays replicated, now L1-tiled.
- **mHC `hc_*_fn`:** split by columns instead of rows. Each rank still holds 0.47 MB per tensor, and
  25 floats per row are all-reduced (the layer runner, consistent with the attention line's mHC kernels).

The embedding (and the Markov embedding) stays replicated: 10 KiB + 0.5 KiB of lookups per token,
against a collective per token to save 0.98 GiB of memory. Changing any split or layout is one line
in the table in `engine/weights/plan.cpp`, then a re-conversion (20 s to 2.5 min per box).

## File format (version 2)

```
[0, 4096)                  Header: magic "DSGB10W1", version, rank, world, tensor count,
                           TOC/data offsets and sizes, TOC CRC32C, model, source dir, creation time
[4096, data_offset)        TOC: one 512-byte record per tensor, zero-padded to 4 KiB
[data_offset, file_bytes)  tensors in TOC order, each 4 KiB-aligned and zero-padded to 4 KiB
```

A TOC record (`engine/weights/dsw.h`) has:

- **Identity:** name, the source tensor name (`{e}` stands for the expert index when stacked), and
  a description of the cut.
- **Shape and type:** dtype, ndim and the natural shape (≤ 4 dims; for e2m1×2 the last dim counts
  bytes).
- **Layout:** natural, L1, L2 or L3 (version 2; the field was reserved, and zero, in version 1).
- **Placement:** absolute offset and unpadded size.
- **How it was cut:** slice kind (replicated, rows, cols, rows-seg), number of stacked experts,
  number of segments, the source shape and the CRC32C of the tensor's bytes as stored.

Stacked expert tensors are `[n_experts, rows, row_bytes]`; expert e starts at `e × bytes / n_experts`
in every layout.

Tensor order is globals (`embed`, `head`, `norm`), then layers 0–39, then `mtp.0`–`mtp.2`; within a
group it is sorted by name. The loader maps the whole data region into one device allocation, so a
tensor's device address is `arena + (offset − data_offset)`. Every tensor is 4 KiB-aligned there.
The loader reads versions 1 and 2 and reports each tensor's layout. It can also load a subset of
the tensors into a smaller arena.

The magic is written last, after the data and TOC, and the file is renamed from `.tmp` into place.
A file with a valid magic is therefore complete.

The loader refuses a file whose header names another rank or world than the process's
(`dsw::rank_mismatch`): `FILE: the file was cut for rank 1 of 4, but this process is rank 0 of 4 (a
weight file serves only the rank it was cut for)`. `dsw_load FILE --rank R` checks the same, and
`dsw_rank_test` tests it on small files of every rank (`--file F R`: on a real one).

## Measured (2026-10-11, version 2)

All four files were converted from their box's own checkpoint copy and verified byte for byte, and
only then replaced the version-1 files. Each file is 77,719,715,840 bytes: 72.38 GiB of data plus the
header and TOC.

| Box | Rank | Convert | Verify (every byte, un-tiled) | Load to GPU |
|---|---|---|---|---|
| rank 0 | 0 | 134 s, 0.58 GB/s | 202 s | 2.0 GB/s on a 12 GiB prefix (full load ≈ 39 s; a full load would break rank 0's 16 GiB per-process limit) |
| rank 1 | 1 | 94 s, 0.83 GB/s | 77 s | 15.2 s, 5.11 GB/s |
| rank 2 | 2 | 33 s, 2.38 GB/s, with the golden-data job on the box | 45 s | 8.1 GB/s on a 24 GiB prefix: the golden-data job held ~60 GB, so a full load did not fit (version 1: 8.6 s, 9.07 GB/s, idle box) |
| rank 3 | 3 | 22 s, 3.53 GB/s | 35 s | 8.3 s, 9.36 GB/s |

On rank 0, conversion reads the checkpoint from the USB4 SSD and writes the file back to it. The
other boxes read and write their internal NVMe. Checking every tensor's CRC on the device copy takes
4.4 s for the full 72 GiB. Version 1 took 135, 113, 22 and 21 s to convert, so tiling on the GPU
costs no measurable time; rank 2's 33 s ran next to the golden-data job.

**The kernels read the file correctly** (`dsw_kcheck` on rank 0, against the kernel tests'
reference data from real layer-3 weights):

- **Bytes.** `dsk::tile_fp8` / `dsk::tile_fp4` applied to the reference's natural slices equal the
  file's tensors byte for byte, and so does `layout.h`'s host tiling:
  - `wq_a`, `wkv`, `wq_b`, `wo_a`, `wo_b`, shared `w1`/`w3`/`w2`, and layer 2's indexer `wq_b`, with
    their scales;
  - `head` and the router gate (L2);
  - the 12 reference experts' `w1`/`w3`/`w2` values and scales, each at its expert id in the stacks.
- **GEMVs run directly on the file's tensors** match the reference outputs as closely as the
  kernel tests do:
  - `gemv_fp8` (M = 8 and 1, mt = 1, 2, 4): 99.98–100 % bit-exact, none beyond 1 ulp;
  - `gemv_woa`: 99.95–99.98 %;
  - `gemv_bf16`: head max error 1.2e-6 of the row maximum, gate 4.4e-7;
  - `moe` (12 experts + shared expert, M = 8, 6, 1): 99.99–100 % exact;
  - `rmsnorm` with the file's `attn_norm`: 100 %.

The other checks also passed:

- The 476 replicated tensors have identical CRCs in all four files. This includes the 8 L1-tiled
  indexer `wq_b`s, tiled on four different GPUs.
- The torch check passed on all 36 tensors of rank 0's layer 3 (15 of them un-tiled with torch) and on
  60 random tensors of rank 0's file (25 un-tiled).
- The new loader read rank 0's version-1 file (layer 3, CRCs checked on the device) before it was
  replaced.

## Tools

Build with `cmake -S engine -B engine/build && cmake --build engine/build -j`. The workers run copies
in `~/dsgb10/bin/`, rank 0 in `/data/dsgb10/bin/`. `dsw_convert` needs the GPU, for the tiling.

```bash
# the partition, sizes and layouts, from the checkpoint headers only (also runs the cross-rank coverage check)
dsw_convert --ckpt DIR --rank R --plan
# convert (O_DIRECT reads and writes, 8 threads, ≤ 6 GiB of host buffers and ~1.2 GB on the GPU)
dsw_convert --ckpt DIR --rank R --out FILE [--threads 8] [--mem-gib 6]
# un-tile and compare every byte against the checkpoint, check CRCs, padding and the wo_a scales,
# and check every text-model tensor is referenced exactly once
dsw_verify --ckpt DIR --file FILE
# time loading onto the GPU, optionally a subset, optionally checking every tensor's CRC on the device copy
dsw_load FILE [--threads 4] [--chunk-mib 64] [--limit-gib G] [--only SUBSTR] [--verify-crc] [--rank R]
# rank 0 only: the kernels' tiling and GEMVs against the kernel tests' reference data (~/dsgb10-data/kerneltests)
dsw_kcheck FILE [REFDIR]
# print the TOC (standard library only); --replicated for cross-rank CRC comparison
python3 engine/weights/toc.py FILE [--replicated]
# a third check with torch slicing and torch un-tiling, in the canary-roce image
docker run --rm --entrypoint /opt/sglang/bin/python3 -v CKPT:/models:ro -v DIR:/dsw:ro -v REPO:/work \
    dsv41-4x-spark:canary-roce /work/engine/weights/check_dsw.py /models /dsw/FILE [--sample N]
```

`dsw_convert --only SUBSTR` writes a test file with just the matching tensors (`dsw_verify --subset`
checks it).

### How correctness is established

1. **Plan checks**, on every run:
   - every checkpoint tensor matches exactly one rule, or the run stops;
   - across the four ranks, a replicated tensor is whole on each rank, and a split one is cut into
     disjoint pieces that cover it exactly;
   - every scale is cut along its weight's blocks;
   - every tiled tensor's shape fits its tiles.
2. **`dsw_verify`** re-derives each slice from the TOC's declarative fields, not from the converter's
   piece list, un-tiles the file's bytes with `layout.h`, and compares every byte with the checkpoint.
3. **`check_dsw.py`** does the same with torch and safetensors, un-tiling with torch reshapes and
   permutes. It also confirms that `wq_b`, `wo_a`, `wo_b`, `attn_sink` and `head` are cut along the
   dims the reference's `convert.py` uses.
4. **`dsw_kcheck`** shows that the kernels' own tiling functions produce the file's bytes and that the
   kernels compute the reference outputs from the file.
5. **CRCs of replicated tensors are identical across the four files.** This shows the four boxes
   cut from the same checkpoint.
6. **`dsw_load --verify-crc`** checks the device copy of every tensor.

## The official checkpoint, converted from one box's copy (2026-10-13)

The files above were cut from dealignai's variant, `DeepSeek-V4.1-Flash-UNCENSORED-FP8` (`d61c59ea`),
which every box holds and production serves. Published measurements use the official checkpoint,
`deepseek-ai/DeepSeek-V4.1-Flash` at revision `fb2764a5`. All four of its rank files were cut on rank 0
from rank 0's copy, and the three worker files were copied to their boxes. Production keeps the
variant's files. The official ones sit beside them, and nothing under `/data/dsgb10/weights/` or
`~/dsgb10/weights/` changed:

| box | rank | file, 77,719,715,840 bytes each |
|---|---|---|
| rank 0 | 0 | `/data/dsgb10/weights-official/rank0.dsw` |
| rank 1 | 1 | `~/dsgb10/weights-official/rank1.dsw` |
| rank 2 | 2 | `~/dsgb10/weights-official/rank2.dsw` |
| rank 3 | 3 | `~/dsgb10/weights-official/rank3.dsw` |

### What differs between the two checkpoints

Read without tensor data (the headers, the index, the download records), the two have the same form:

- They have the same 48 shards and 96,085 tensors. Every name, dtype, shape, data offset and header
  metadata entry is equal.
- `model.safetensors.index.json`, `config.json` and the tokenizer files are the same files, hard links
  where both sets share a disk. The index's `total_size` is 510,286,023,000 in both.
- `dsw_convert --plan` makes the same plan for all four ranks: 1,609 tensors and 72.38 GiB per rank,
  with 270 checkpoint tensors left out (vision, Engram tables), and the same components and layouts.
- `tools/st_inventory.py` gives the same inventory for both, equal to `ref/inventory.txt`.

The data differs in 27 of the 48 shards:

- **The same files in both directories:** shards 00001–00012 and 00040–00048, hard links on every box
  that holds both sets. They hold the vision tower, the embedding, layers 0–9 and 37–39, the head, the
  three DSpark (`mtp`) blocks and the two Engram tables.
- **Different:** shards 00013–00039, which hold one layer each, layers 10–36. Each one's sha256 (the
  official download's LFS sha256) differs from the variant's `.expected.sha256`.

The converted files narrow this down to tensors. Their TOCs, compared CRC32C by CRC32C on all four
ranks, differ only in `layers.N.attn.wo_b.weight` and `layers.N.ffn.shared_experts.w2.weight` for
N = 10–36:

- That is 54 checkpoint tensors, and only their FP8 values differ. Their scales are equal, and so is
  every other tensor, the routed experts included.
- These are the two matrices through which a layer's attention and its shared expert write into the
  residual stream.

The engine reads Engram from the kit's shards ([engram.md](engram.md)), which were packed from the
variant. The tables live in shards 47 and 48, which both sets share, so the same Engram shards serve
the official weights. Checked: 3,003 sampled rows of each of the eight kit shards equal the official
tables.

### The three copies of the official checkpoint

- **rank 0** holds it at `/data/models/DeepSeek-V4.1-Flash`, with the Hugging Face download records:
  revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.
- **ranks 2 and 3** hold it at `~/DeepSeek-V4.1-Flash`, without download records, so the
  revision is not written down there. Their content shows it is the same:
  - the same 55 files at the same sizes;
  - on all three boxes, the 27 shards that differ from the variant hash to that revision's LFS sha256
    (O_DIRECT reads: 97 s per spark box at 2.05 GB/s, 253 s on rank 0's USB4 drive);
  - each box's copy shares the other 21 shards with its own variant copy.
- **rank 1** holds only the variant.

Ranks 2 and 3 had already been cut on rank 0 and copied when this came to light. Run on rank 2 and
rank 3 against those boxes' own copies, `dsw_verify` passes both files: 0 bad, every byte compared,
38 s each. A conversion on those boxes would therefore have produced the same tensors; only the
header's source path would differ.

### Converted, verified, copied, loaded

Every GPU step ran under `engine/server/ring.sh`; `dsw_verify` ran outside it, at 4 threads and nice
19. The binaries were built from f7adc24, whose engine code is release candidate 5's (c247a78).

| rank | convert, on rank 0 | `dsw_verify` on rank 0, every byte | copy to the worker | on the worker: sha256 as sent, `--crc-only`, TOC equal | load to the GPU; CRCs on the device copy |
|---|---|---|---|---|---|
| 0 | 131 s, 0.60 GB/s | 200 s | none (stays on rank 0) | | 38.2 s, 2.04 GB/s; 4.5 s |
| 1 → rank 1 | 146 s, 0.53 GB/s | 286 s | 164 s, 0.48 GB/s | yes (57 s), 14 s, yes | 14.1 s, 5.52 GB/s; 4.4 s |
| 2 → rank 2 | 128 s, 0.61 GB/s | 399 s, next to an engine run | 117 s, 0.66 GB/s | yes (49 s), 8 s, yes | 7.9 s, 9.81 GB/s; 4.4 s |
| 3 → rank 3 | 150 s, 0.52 GB/s | 210 s | 140 s, 0.55 GB/s | yes (50 s), 8 s, yes | 7.9 s, 9.82 GB/s; 4.4 s |

- **rank 0's USB4 drive sets the pace.** Conversion reads the checkpoint from it and writes the file
  back to it; the variant's rank 0 took 134 s. Verification reads both from it; the variant's took
  202 s.
- **The copy** is one O_DIRECT `dd` stream into `ssh -c aes128-gcm@openssh.com`. One stream reads the
  drive at about 0.8 GB/s (the shard hashing above). On its own, ssh carries 1.00 GB/s from rank 0 to
  rank 1 with that cipher and 0.58 GB/s with its default one.
- **Loads.** The workers load as fast as they did the variant's (5.11 and 9.36 GB/s then). Rank 0
  now loads whole: the 16 GiB per-process limit that held the variant's rank 0 to a prefix is gone.
- **The other checks passed:**
  - `dsw_kcheck` on rank 0 (2.5 s; layer 3, the head and layer 2's indexer are the same in both
    checkpoints);
  - the torch check, `check_dsw.py --sample 60`, on rank 0: 0 bad, 25 of the 60 tensors un-tiled with
    torch;
  - the 476 replicated tensors have identical CRCs in all four files;
  - against the variant's files, every rank's TOC has the same names, dtypes, shapes, kinds, layouts
    and cuts, and the CRCs differ in exactly the 54 tensors above.
- **Disk.** `/data` on rank 0 had 189 GB free, room for one more file above the 60 GB it must keep. So
  the ranks went one at a time, 1, 2, 3, then 0. Each was converted, verified, copied and checked on
  its worker, and then rank 0's copy was deleted; free space never fell below 111 GB. The whole run took
  55 minutes (08:43–09:38) including waits for the ring.

### One path on every box

`run4.sh` gives every rank the same arguments, and `$HOME` is the same on all four boxes. So each box
has a link `~/dsgb10/weights-official/rank.dsw` to its own rank's file. On rank 0 it points at
`/data/dsgb10/weights-official/rank0.dsw`; on a worker it points at `rankR.dsw` beside it. Every
engine run on the official weights passes

```bash
--weights ~/dsgb10/weights-official/rank.dsw
```

unquoted (or as `$HOME/...`), and each box resolves it to its own link. The engine needed no change.

The engine did not then compare a file's header with its rank, and a link to another rank's file
would have loaded the wrong slices. Every engine job on these files therefore first checked each box's
link: it must resolve to a file whose header says `rank r/4` and `from /data/models/DeepSeek-V4.1-Flash`.
The loader now refuses another rank's file, naming the file and both ranks, so a wrong link stops the
start. Where the four files differ only by their number, a path flag can also name each rank's own
with `{rank}` ([README § How to use](../README.md#how-to-use)).

### Repeating it with one copy of the checkpoint

On the box with the checkpoint (here rank 0, with the workers at 10.0.0.4, .3 and .2), from the
repository's root. Each worker rank took 10–12 minutes from rank 0's USB4 drive.

```bash
cmake -S engine -B engine/build && cmake --build engine/build -j 4
B=engine/build/weights
CKPT=/data/models/DeepSeek-V4.1-Flash
OUT=/data/dsgb10/weights-official                  # room for one 77.7 GB file at a time
mkdir -p $OUT
for r in 0 1 2 3; do $B/dsw_convert --ckpt $CKPT --rank $r --plan; done    # headers only

# each worker rank in turn: r=1 W=10.0.0.4, then r=2 W=10.0.0.3, then r=3 W=10.0.0.2
$B/dsw_convert --ckpt $CKPT --rank $r --out $OUT/rank$r.dsw        # uses ~1.2 GB of the GPU
$B/dsw_verify --ckpt $CKPT --file $OUT/rank$r.dsw --threads 4      # every byte, un-tiled
python3 engine/weights/toc.py $OUT/rank$r.dsw > rank$r.toc
ssh $W mkdir -p dsgb10/weights-official dsgb10/official-tools
scp $B/dsw_verify $B/dsw_load engine/weights/toc.py $W:dsgb10/official-tools/
dd if=$OUT/rank$r.dsw iflag=direct bs=64M status=none | tee >(sha256sum > rank$r.sent) |
  ssh -c aes128-gcm@openssh.com $W "dd of=dsgb10/weights-official/rank$r.dsw.part oflag=direct iflag=fullblock bs=64M status=none &&
    mv dsgb10/weights-official/rank$r.dsw.part dsgb10/weights-official/rank$r.dsw"
ssh $W "dd if=dsgb10/weights-official/rank$r.dsw iflag=direct bs=64M status=none | sha256sum"   # = rank$r.sent
ssh $W dsgb10/official-tools/dsw_verify --file dsgb10/weights-official/rank$r.dsw --crc-only
ssh $W dsgb10/official-tools/dsw_load dsgb10/weights-official/rank$r.dsw --verify-crc       # uses the GPU
ssh $W python3 dsgb10/official-tools/toc.py dsgb10/weights-official/rank$r.dsw | tail -n +2 | diff - <(tail -n +2 rank$r.toc)
rm $OUT/rank$r.dsw

# rank 0 stays where it was cut
$B/dsw_convert --ckpt $CKPT --rank 0 --out $OUT/rank0.dsw
$B/dsw_verify --ckpt $CKPT --file $OUT/rank0.dsw --threads 4
$B/dsw_kcheck $OUT/rank0.dsw                       # with the kernel tests' reference data
$B/dsw_load $OUT/rank0.dsw --verify-crc

# one path on every box
mkdir -p ~/dsgb10/weights-official && ln -sfn $OUT/rank0.dsw ~/dsgb10/weights-official/rank.dsw
ssh 10.0.0.4 ln -sfn rank1.dsw dsgb10/weights-official/rank.dsw    # likewise rank2 on .3, rank3 on .2
```

Where a worker holds a copy of the checkpoint, as ranks 2 and 3 do, it can convert and verify
its own rank instead, which skips the copy. Their NVMe makes this fast: the variant's files took
22–33 s to convert and 35–45 s to verify there, and the official files verified in 38 s.

The engine on these files: [engine.md § The official checkpoint's weights](engine.md#the-official-checkpoints-weights-2026-10-13).
