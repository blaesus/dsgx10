# Per-rank weight files (milestone A2)

Each rank holds one `.glw` file: its slice of every tensor the engine reads, in the formats and tile layouts the
kernels read, cut on the CPU from AikidoSec/altar-1 at revision `5d591cd98958e4e7e429517887e80ca3bb7ce2c4`.
Written 2026-10-14. Code: [engine/weights/](../engine/weights/). **This is a delta from glm53's
[layout.md](../../glm53-4xgx10-switchless/docs/layout.md)**: what is not said here (the container, the TOC, the
rank check, the loader, the fill's commit protocol) is unchanged. The formats are [model.md](model.md) and
[quantisation.md](quantisation.md), the layouts [experts.md § 3](experts.md) (`engine/experts/layout.h`,
`engine/weights/int4_pack.py`) and [attention.md § 4](attention.md) (`engine/attn/tools/pack.py`).

## Contents

1. [Status](#1-status)
2. [What changed in the container](#2-what-changed-in-the-container)
3. [The plan](#3-the-plan)
4. [The routed experts and the fill](#4-the-routed-experts-and-the-fill)
5. [The manifest and the checkpoint gate](#5-the-manifest-and-the-checkpoint-gate)
6. [Verification](#6-verification)
7. [Files written](#7-files-written)
8. [How to redo it](#8-how-to-redo-it)
9. [What remains](#9-what-remains)

## 1. Status

- **All four rank files are written, verified and installed** (2026-10-14): 82,411,745,280 B (76.752 GiB) each,
  13,961 TOC entries, no reserved entry left. Rank 0's is at `/data/dsgb10/altar1/weights/rank0.glw`; ranks 1–3's
  are at `~/altar1/weights/rankR.glw` on their own boxes, each with its manifest beside it (§ 7).
- **Every value is the checkpoint's** (§ 6). Over the four files, `verify.py --indep` compared:

  | What | Count |
  |---|---:|
  | INT4 values | 473,360,760,832 |
  | scales | 14,792,523,776 |
  | zero points | 14,792,523,776 |
  | BF16 non-expert elements (7,407,927,296 of them un-tiled from kBF16T) | 8,218,832,896 |
  | BF16 expert elements | 19,025,362,944 |
  | F32 elements | 51,072 |
  | descriptors | 153,216 |
  | RoPE words | 134,217,728 |
  | tensor checks | 55,844 |

  Every one is equal, 0 bad. The 459,276,288 dequantised BF16 elements of W_UK^T equal both `int4.dequant_bf16` and
  compressed-tensors 0.19.0's own `decompress` (`ctcheck.py --file`).
- **Layouts:**
  - INT4 routed experts (layers 4–76): the expert kernels' 2,368-byte blocks;
  - BF16 experts (3, 77, 78): their 2,048-byte blocks;
  - q_b, W_UV and o_proj of layers 4–76: the same INT4 block (int4-tile);
  - every BF16 projection: the attention side's kBF16T (bf16-tile, § 2);
  - embedding, norms, router and RoPE table: natural.

**Nothing is requantised** (DESIGN decision 1). BF16 and F32 tensors are copied: cut, stacked, transposed or
interleaved. The INT4 tensors' packed values, scales and zero points are cut on group boundaries and re-tiled
into the kernels' blocks, with every nibble, scale and zero point kept. One tensor class is derived: W_UK^T of the
INT4 layers 4–76. The attention side serves it in BF16, so the converter writes the contract's exact dequantisation
`bf16_rne((nib − znib) · s)` ([model.md](model.md) § 4.2) and checks it against compressed-tensors' own `decompress`.

## 2. What changed in the container

| | glm53 | altar-1 |
|---|---|---|
| magic | `GLM53W01` | `ALTR1W01` (each engine refuses the other's files) |
| layouts | natural 0, T64 1, T128 2, expert-pool 3 | the same numbers, plus **int4-tile 4** and **bf16-tile 5** (T64/T128 unused) |
| expert format tags | NVFP4 0, S3 1 | NVFP4 0, **INT4 2, BF16 3**; tag 1 refused. One format a layer: gate = up = down |
| flags | lossy, reserved, scale, interleaved | the same, plus **dequant 16**: BF16 dequantised from INT4 by the contract's rule (exact, not lossy) |
| experts a layer | 256 | 168 |
| FP8 | everything but the experts | none: no entry has the lossy or scale flag |

**int4-tile** (`Layout::Int4Tile`, q_b, W_UV and o_proj of layers 4–76): one U8 entry of shape
`[N/8][K/512][2368]` for the natural `[N][K]`, `blk_rows × blk_k = 1 × 32`. It uses the same 2,368-byte block as an
INT4 expert matrix (attention.md § 4, experts.md § 3). The block holds 2,048 B of the checkpoint's own nibbles (+8
kept), 256 B of bf16 group-of-32 scales and 64 B of zero-point nibbles.

**bf16-tile** (`Layout::Bf16Tile`, attention.md § 4 "kBF16T", `pack.bf16_tiles`): a BF16 `[N][K]` with its natural
shape in the TOC, stored in the GEMV's lane order `[N/8][K/64][2 halves][32 lanes][8 values]`. The 1 KB chunk
(i, j) holds rows 8i…8i+7 and k = 64j…64j+63, at byte ((i·K/64) + j)·1024. Lane 4g + t's half h holds row 8i + g,
k = 64j + 16t + 8h + 0…7. The values and byte count are the same as row-major. The kernels give the same outputs
bit for bit (attention.md test T), and the GEMV is 0.8 ms a token faster. Following the attention side's decision
(2026-10-14), every BF16 projection that a GEMV or GEMM reads is written in this layout:

- the head;
- the x-projection;
- q_b, W_UK^T, W_UV and o_proj in BF16 (W_UK^T on every layer);
- the indexer's wq_b and weights_proj;
- the dense MLPs and the shared experts;
- eh_proj.

Natural (row-major) stays for the embedding (a row lookup), the norms, the router and its bias, and the RoPE table.
`convert.py --bf16-layout natural` writes the row-major form instead; the kernels accept both.

`glw_check` checks both new layouts' shapes:

- int4-tile: U8, 3-D, last dimension 2368, blk 1 × 32;
- bf16-tile: BF16, 2-D, N % 8 = 0 and K % 64 = 0.

The loader accepts layouts up to 5.

## 3. The plan

Built from the checkpoint's headers alone (`convert.py --plan`, `--list` for every tensor). Rank r of 4, the same
cuts as glm53 § 3, with BF16 where glm53 had FP8:

| Tensors (engine names) | Source | Format, layout | Rank r holds | Bytes a rank |
|---|---|---|---|---:|
| `embed.weight` | `model.embed_tokens.weight` [154880, 6144] | BF16 natural | vocab rows [38720r, +38720) | 475.8 MB |
| `head.weight` | `lm_head.weight` | BF16 bf16-tile | vocab rows [38720r, +38720) | 475.8 MB |
| `norm.weight`, `rope.table` | `model.norm.weight`; `attn/tools/rope_table.py` | BF16; U32 [2^20][32] | all | 12 KB; 134.2 MB |
| `layers.L.{attn_norm,ffn_norm,attn.q_norm,attn.kv_norm}.weight`, `indexer.k_norm.{weight,bias}` | the norms | BF16 natural | all | 20 KB a layer |
| `layers.L.attn.xproj.weight` | `q_a_proj`, `kv_a_proj_with_mqa`, indexer `wk` | BF16 bf16-tile | q_a [512r, +512), kv_a [128r, +128) and [512+16r, +16), wk [32r, +32) on full layers: 688 (656) × 6144 | 8.45 (8.06) MB |
| `layers.L.attn.q_b.weight` | `q_b_proj` | L ∈ 4–76: **INT4 int4-tile** U8 [512][4][2368]; else BF16 bf16-tile [4096][2048] | rows [4096r, +4096): heads 16r…16r+15 | 4.85 / 16.78 MB |
| `layers.L.attn.w_uk.weight` | `kv_b_proj` | BF16 bf16-tile [8192][192]; on 4–76 **dequantised** (flag 16) | head H's rows 448H…+191 transposed to [512][192], 16 heads stacked | 3.15 MB |
| `layers.L.attn.w_uv.weight` | `kv_b_proj` | 4–76: INT4 int4-tile U8 [512][1][2368]; else BF16 bf16-tile [4096][512] | head H's rows 448H+192…+447, stacked | 1.21 / 4.19 MB |
| `layers.L.attn.o_proj.weight` | `o_proj` | 4–76: INT4 int4-tile U8 [768][8][2368]; else BF16 bf16-tile [6144][4096] | cols [4096r, +4096) | 14.55 / 50.33 MB |
| `layers.L.indexer.wq_b.weight` (full layers, incl. 78) | `indexer.wq_b` | BF16 bf16-tile [1024][2048] | rows [1024r, +1024) | 4.19 MB |
| `layers.L.indexer.weights_proj.weight` | `indexer.weights_proj` [32, 6144] | BF16 bf16-tile [8][6144] | rows [8r, +8) | 98 KB |
| `layers.{0,1,2}.mlp.gate_up.weight` | `mlp.gate_proj`, `up_proj` | BF16 bf16-tile, interleaved (rows 2i gate, 2i+1 up) | intermediates [3072r, +3072) | 75.5 MB |
| `layers.{0,1,2}.mlp.down.weight` | `mlp.down_proj` | BF16 bf16-tile | cols [3072r, +3072) | 37.7 MB |
| `layers.L.shared.gate_up.weight`, `.down.weight` (L = 3…78) | `shared_experts.*` | BF16 bf16-tile, gate/up interleaved | rows / cols [512r, +512) | 12.6 / 6.3 MB |
| `layers.L.router.weight`, `.bias` | `mlp.gate.weight` BF16 [168, 6144], `e_score_correction_bias` F32 [168] | as shipped, natural | all | 2.06 MB |
| `layers.78.mtp.{enorm,hnorm,head_norm}.weight` | `enorm`, `hnorm`, `shared_head.norm` | BF16 | all | 36 KB |
| `layers.78.mtp.eh_proj.weight` | `eh_proj` [6144, 12288] | BF16 bf16-tile | rows [1536r, +1536) | 37.7 MB |
| `layers.L.experts.desc`, `layers.L.experts.E` (L = 3…78, E = 0…167) | `experts.E.{gate,up,down}_proj.*` | reserved, then filled (§ 4) | gate/up rows, down cols [512r, +512) | 5.46 MB an INT4 expert, 18.87 MB a BF16 one |

**Totals a rank** (`convert.py --plan`; every rank is the same size):

| Class | GiB | Entries |
|---|---:|---:|
| embed, head, final norm, RoPE table | 1.011 | 4 |
| attention, BF16 layers 0–3, 77 | 0.397 | 57 |
| attention, INT4 layers 4–76 (int4-tile q_b, W_UV, o_proj; BF16 W_UK^T, x-projection, indexer) | 2.244 | 729 |
| dense MLPs | 0.316 | 6 |
| shared experts and routers, layers 3–77 | 1.463 | 300 |
| MTP layer: attention, shared expert, router, eh_proj, norms | 0.136 | 21 |
| **file before the fill** (5,985,759,232 B, incl. the 7,151,616-byte TOC) | **5.575** | 1,117 |
| routed experts INT4, layers 4–76 (73 × 168 + 73 desc) | 62.318 | 12,337 |
| routed experts BF16, layers 3, 77, 78 (3 × 168 + 3 desc) | 8.859 | 507 |
| **filled file** (82,411,745,280 B) | **76.752** | **13,961** |

By layout, before the fill: bf16-tile 3.450 GiB, int4-tile 1.401 GiB, natural 0.717 GiB.
[quantisation.md](quantisation.md) (S2) estimated 76.6 GiB a rank from the shard headers. The difference is the
RoPE table (0.125 GiB), which S2 did not count.

**Coverage** (`glw.coverage`, run by every `convert.py`): all 150,554 checkpoint tensors are accounted for. 1,874
are sources of the 239 non-expert units, and 148,680 are routed-expert tensors behind the reserved entries:

- 73 INT4 layers × 168 experts × 3 matrices × 4 parts (packed, scale, zero point, shape) = 147,168;
- 3 BF16 layers × 168 × 3 = 1,512.

No tensor is unused and none is used twice.

## 4. The routed experts and the fill

**The TOC** of each MoE layer (3–78) holds:

- `layers.L.experts.desc`: `gx::MatDesc [168][3]` (gate, up, down), 32,256 B, padded to 32,768;
- one entry an expert, `layers.L.experts.E`: its gate, up and down matrices back to back, each in the expert
  kernels' blocks for the layer's format.

The formats and sizes:

| Layers | Format | Block | One matrix | One expert |
|---|---|---|---:|---:|
| 4–76 | **INT4** (tag 0x222) | 8 rows × 512 k, 2,368 B | 1,818,624 B | 5,455,872 B |
| 3, 77, 78 | **BF16** (tag 0x333) | 8 rows × 128 k, 2,048 B | 6,291,456 B | 18,874,368 B |

Each matrix is a multiple of 4 KiB, so the 168 entries lie back to back and the layer's pool starts at expert 0's
entry. The descriptors are `{off = (3e + m) · matrix bytes, s2 = 1.0, fmt}`, from `int4_pack.matdesc`. The reserved
entries are sized for the layer's format when the plan is made.

**The fill** ([experts_fill.py](../engine/weights/experts_fill.py)) packs the rank's slice with `int4_pack.py`, the
expert kernels' Python port of `layout.h`, the same file at the expert agent's branch and on `altar1`
(SHA-256 `e7428921…`):

- gate/up rows [512r, +512);
- down's input columns [512r, +512). For INT4 these are packed words [64r, +64), scale and zero-point columns
  [16r, +16); all cuts fall on group boundaries.

For INT4 it calls `pack(canon_int4(packed, scale, zp))`; for BF16, `pack(canon_bf16(w))`. **Every matrix written is
read back through the packer's `untile` and compared with its source cut before the layer is committed.** Layers
are filled in TOC order. Each layer is committed (data synced, then TOC, header, manifest), so a rerun resumes
after the last whole layer. One pass fills several `--file`s.

## 5. The manifest and the checkpoint gate

The manifest is glm53 § 5 without the FP8 fields, plus:

- `checkpoint`: the revision, the index's SHA-256, and every shard's LFS SHA-256 and state;
- `packers`: the packer file names;
- `bf16_layout`;
- `rule`: "no requantisation".

The tool hash covers `glw.py`, `convert.py`, `sources.py`, `int4.py`, `crc32c.c`,
`rope_table.py` and the two packer files, `int4_pack.py` and `attn/tools/pack.py`.

**The gate** (`sources.Checkpoint`): the shards and index are untrusted data, read with `python -I` and only parsed.
A shard is read only if:

- its size equals `manifest.tsv`'s (the Hub's LFS record);
- its SHA-256, measured on the box, equals the manifest's. The hashes come from `--hashes`: `sha256sum` output, or a
  verify log's `OK <file>` lines mapped to the manifest's hash.

Otherwise the converter stops; `--unverified-ok` is for test files only.

On rank 3, the measured hashes were the copy's `verify-all.log` (38 shards OK at 10:20). Two files had failed there,
model-00011 and the index. They were re-copied from rank 2's verified copy through rank 0 at 10:31, checked against
the manifest on arrival (`OK` lines in `~/altar1/a2/shards-ok.log`), and renamed into place. At 10:57 the
coordinator re-copied the same two files, plus `tokenizer.json`, by atomic rename to new inodes with the same
manifest hash. Every conversion described here read model-00011 either before 10:57 from the 10:31 copy, or after it.

## 6. Verification

| Check | Tool | What |
|---|---|---|
| Structure and CRCs | `glw_check` (C++), `verify.py --crc` | header geometry, TOC CRC, offsets in order, shape × dtype = bytes, the int4-tile and bf16-tile shapes, no reserved entry left, every CRC32C, zero padding; the manifest's per-tensor and whole-file SHA-256s |
| **Independent** | `verify.py --indep` | written without glw.py's plan, the packers or their inverses. Each entry is described again from this document's § 3 (source pieces, cuts, transposes, interleaving). Every element is un-tiled by `verify_layouts.py`'s own byte formulas (INT4 block, BF16 block, kBF16T), each tested against a per-element gather and against the packers themselves (`--selftest`). Every INT4 value nibble, scale and zero-point nibble, and every BF16/F32 element, must equal the checkpoint's. W_UK^T of INT4 layers must equal `int4.dequant_bf16`. Every descriptor must equal `make_desc`'s. The RoPE table must equal `rope_table.py`'s |
| Library | `ctcheck.py --file` | every INT4 layer's kv_b through compressed-tensors 0.19.0's own `PackedQuantizationCompressor.decompress` (scheme from the checkpoint's `quantization_config`), against `int4.dequant_bf16` and against the W_UK^T entries of the rank files, transposed and cut as § 3 |
| Cross-rank | `verify.py --cross` (on the four saved TOCs) | the same entry names, shapes, layouts and formats on every rank; replicated entries (norms, router, RoPE) and the descriptor tables with equal CRCs on all four, every split entry with four distinct CRCs |
| Loader | `glw_load --host --rank R` on the rank's own box | the loader opens the file as its rank and reads every tensor back with its CRC32C |
| Rank refusal | `glw_rank_test --file F R` | refused as each of the three other ranks |
| Determinism | a second conversion of two layers | the entries' SHA-256s equal the full files' |

**Results, 2026-10-14** (all four installed files):

| Check | Result |
|---|---|
| `glw_check` | OK on every file, on rank 3 before the copy and on each rank's own box after it (6–41 s a file) |
| `verify.py --crc` | every entry's SHA-256 and the whole file's equal the manifest's, 4 of 4 |
| `verify.py --indep` | 55,844 tensor checks (13,961 a file), **0 bad**. The counts are in § 1. Run times: ranks 3 and 0 together 12.7 min of a 21.4-min `--crc --indep` run; rank 1 15.2 min, rank 2 13.5 min (each including `--crc`) |
| `ctcheck.py --file` | 73 kv_b tensors (1,071,644,672 weights) equal to compressed-tensors' `decompress`; the values and zero points equal its `unpack_from_int32`; W_UK^T in all four files equal to the library's weight, transposed and cut (459,276,288 elements) |
| `verify.py --cross` | 13,961 entries over ranks 0–3: 593 replicated entries and descriptor tables with equal CRCs, 13,368 split entries with four distinct CRCs, 0 bad |
| `glw_load --host --rank R` | each file opened as its own rank on its own box; 13,961 tensors (76.745 GiB) read with every CRC32C OK (25–103 s) |
| `glw_rank_test` | all ok on every file: refused as the three other ranks |
| Second conversion | layers 4 (INT4) and 77 (BF16) for rank 3 converted and filled again into a separate file: 364 entries, 4,243,810,624 B, every entry's SHA-256 equal to the installed rank3.glw's |
| Tamper (test files) | six flipped bytes (embedding, a kBF16T dense down, xproj, INT4 q_b, dequantised W_UK^T, a descriptor) all caught; an earlier six (two expert entries among them) also |
| `glw_kcheck --rank 0` (host binding) | **expert side 0 bad**: 38,304 matrices (36,792 INT4, 1,512 BF16), sizes by tag, contiguity, every descriptor. 1,175 bad on the non-expert entries: kcheck still checks glm53's attention (FP8 T64 with scale entries) and has not been ported to the INT4 tile and kBF16T (§ 9) |

`glw_kcheck --gpu` was not run. Its attention half expects glm53's FP8 entries, so a ring job now would test the
port, not the files. The expert agent's own device tests cover the expert blocks (experts.md § 7).

**Peak memory on rank 3** (`/usr/bin/time`; the budget there is 2 GB):

| Step | Peak |
|---|---:|
| `convert.py` (two ranks, or one) | 1.54–1.56 GB |
| `experts_fill.py` | 0.43–0.48 GB |
| `ctcheck.py` | 1.08–1.11 GB |
| `verify.py --indep` | 1.70 GB on one file after the fix below |

`verify.py --indep` peaked at **4.96 GB** in the first run: it read the whole embedding and head (1.9 GB each) to
cut a rank's rows. With row reads it peaked at 2.17 GB in the second run, which held the head's rows, its bytes and
two copies. Comparing in 4,096-row slices without copies brought it to 1.70 GB. At 10:34, during the first run, rank 3
showed 115 GiB available, so earlyoom was never near. The two runs still broke the budget.

## 7. Files written

Installed 2026-10-14, each hashed on arrival against its manifest before it was renamed into place:

| Rank | Location | Bytes | SHA-256 | TOC CRC | Tools |
|---|---|---:|---|---|---|
| 0 | `/data/dsgb10/altar1/weights/rank0.glw` | 82,411,745,280 | `1e4f8f6e00694628e1276689e6eaaa122848db2a838544b61d6853316d949587` | `3ad99213` | `9ac400f` |
| 1 | `~/altar1/weights/rank1.glw` | 82,411,745,280 | `1f4f66105c443364af15fad62ff80b5b8540f52de162202493ed3888b66216c1` | `13976662` | `2f39c4e` |
| 2 | `~/altar1/weights/rank2.glw` | 82,411,745,280 | `1c3d81e87da32ad2a2c9937dbabb48aef3d448991525ce6936ab5aeb7646654f` | `7846c7c9` | `2f39c4e` |
| 3 | `~/altar1/weights/rank3.glw` | 82,411,745,280 | `5ce3b0fd9c0d481cc6bdc06a70a5fd33ac1e1366ab2e61e35b4557990a55fa17` | `ce4623bb` | `9ac400f` |

`9ac400f` and `2f39c4e` differ only in `verify.py`. The converter's tool hash is the same,
`57001ada364cb11e…`, in all four headers. The packers were `int4_pack.py` `e7428921…` and `attn/tools/pack.py`
`6def2e9e…`. Rank 0's data disk had 32 GB free after its file landed.

**Timings** (rank 3's CPU; nice 19; cores 10–19; 8 threads):

| Step | Ranks 3 and 0 | Rank 1 | Rank 2 |
|---|---:|---:|---:|
| `convert.py`, non-experts | 70.6 s | 73.1 s | 60.0 s |
| `experts_fill.py`, layers 3–78 | 18 min 33 s | 14 min 25 s | 12 min 58 s |
| `verify.py --crc --indep` | 21 min 21 s | 15 min 11 s | 13 min 29 s |
| `ctcheck.py --file` | 18.5 s | 32.2 s | 17.4 s |

For two ranks, a fill layer took 11 s for INT4 (5 s of it reading) and 27 s for BF16. Copy and hash on arrival:

| File | Copy | Hash on arrival | Rate |
|---|---:|---:|---:|
| rank0.glw | 95 s | 56 s | 867 MB/s |
| rank1.glw | 161 s | 50 s | 512 MB/s |
| rank2.glw | 147 s | 49 s | 561 MB/s |

Rank 3's file was moved on its own disk after verification.

**Superseded files.** Ranks 3 and 0 were first cut at commit `1295186`, 10:32–10:52, with every BF16 projection
row-major:

- SHA-256 `b20af17d…` and `8d6a31de…`;
- the same checks, all 0 bad;
- `verify` peaked at 4.96 GB.

After the attention side's kBF16T decision, the files were deleted and cut again. Comparing rank 3's two
manifests: 13,581 of 13,581 entries outside bf16-tile kept their SHA-256, at the same offsets; all 380 bf16-tile
entries changed, being the same values in another order.

## 8. How to redo it

Run everything on the CPU of the box that holds the checkpoint. On rank 3: nice 19, `ionice -c3`, cores 10–19,
8 threads, `oom_score_adj` 1000, `CUDA_VISIBLE_DEVICES=`. Use a venv with numpy, torch (CPU) and
compressed-tensors (0.19.0 here). The C++ tools build with the engine (`cmake --build … -j4 --target glw_check
glw_load glw_rank_test glw_crc`); `GLW_CRC_LIB` points glw.py at `libglw_crc.so`. `CK` is the snapshot directory,
`MAN` the Hub's `manifest.tsv`, `H` the box's hash log. Take the tools from one commit; record it with `--commit`.

```
cd engine/weights
python -I convert.py --ckpt CK --manifest MAN --hashes H --plan [--list]                 # the plan, bytes, coverage
python -I convert.py --ckpt CK --manifest MAN --hashes H --out-dir OUT --ranks 3 0 --commit C   # non-experts, 71 s
python -I experts_fill.py --ckpt CK --manifest MAN --hashes H --file OUT/rank3.glw OUT/rank0.glw --layers 3 ... 78
glw_check OUT/rank*.glw;  glw_load OUT/rankR.glw --rank R --host;  glw_rank_test --file OUT/rankR.glw R
python -I verify.py OUT/rank3.glw OUT/rank0.glw --crc --indep --ckpt CK --manifest MAN --hashes H
python -I ctcheck.py --ckpt CK --manifest MAN --hashes H --config CK/config.json --file OUT/rank*.glw
python -I verify.py OUT/rankR.glw --save-toc TOC/rankR.toc;  python -I verify.py --cross TOC/rank{0,1,2,3}.toc
```

The packers default to the in-tree `engine/weights/int4_pack.py` and `engine/attn/tools/pack.py`
(`--expert-packer`, `--attn-packer` override them). `--test-packers` swaps in test stand-ins for exercising the
tools, never for rank files. The pass script used on rank 3 converts and fills two ranks, then runs every check
above. Two filled files (165 GB) is the most rank 3's disk takes next to the checkpoint. Each file is streamed to
its box through rank 0 (`ssh rank3 cat | ssh rankR 'cat > …part'`), hashed on arrival against its manifest,
renamed into place, and only then is the staged copy deleted.

## 9. What remains

- **Binding on the device.** The runner (A4, merged into `altar1` as `7a8aab4`, after these files were cut)
  binds by layout value: 4 is an INT4 projection, 5 is kBF16T (un-tiled into a copy for the index weights, which it
  reads row-major), and natural is row-major BF16. These are this file format's numbers. It has not yet opened
  these files on the GPU. What is left:
  - a ring job that binds the four installed files and runs a step;
  - `glw_kcheck`'s non-expert half, which still checks glm53's FP8 entries (its expert half passes, § 6);
  - then `glw_kcheck --gpu`.
  - `bind.cu` names the two layouts by value; it could use `glw::Layout::Int4Tile` and `Bf16Tile` from this
    branch's `glw.h`.
- `tests/expert_fill_test.cu` needs canonical dumps that `experts_fill.py` no longer writes; `verify.py --indep`
  covers the same comparison for every expert of every layer.
- **A merge conflict to expect.** The expert agent's branch also edited `glw.py` (168 experts,
  `FMT_S3_RETIRED`, `BLK_OF`). This branch's `glw.py` replaces those hunks: same values, a pluggable packer
  registry. It was resolved here in favour of this branch, and nothing else imports the removed names.
- The staged copies are deleted. Rank 3 keeps the run's logs in `~/altar1/a2/` (`passA-9ac400f.log`, `passB.log`,
  `passC.log`, `det.log`), the saved TOCs in `~/altar1/a2/toc/`, and the checks' JSON in `~/altar1/stage/`.
  Rank 0 keeps the relay logs in `~/dsgb10-data/altar1-a2/`.
