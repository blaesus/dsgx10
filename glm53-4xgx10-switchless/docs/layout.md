# Per-rank weight files (milestone G2)

Each rank holds one `.glw` file: its slice of every tensor the engine reads, in the formats and tile layouts the
kernels read, cut on the CPU from NVIDIA's checkpoint (the BF16 master) and zai's BF16 mirror. Written
2026-10-13. Code: [engine/weights/](../engine/weights/). Decisions this implements: DESIGN decisions 2, 3
(level 1), 6, 7; the formats are [quantisation.md § 2–4](quantisation.md), the layouts
[attention.md § 1–2](attention.md) (`engine/attn/tools/pack.py`) and [experts.md § 2](experts.md)
(`engine/experts/layout.h`).

## Contents

1. [Status](#1-status)
2. [The file format](#2-the-file-format)
3. [The plan](#3-the-plan)
4. [FP8 conversion: zai's rule, reproduced](#4-fp8-conversion-zais-rule-reproduced)
5. [The manifest](#5-the-manifest)
6. [The routed experts: reserved entries and the fill](#6-the-routed-experts-reserved-entries-and-the-fill)
7. [Verification](#7-verification)
8. [Files written](#8-files-written)
9. [Tools](#9-tools)
10. [What remains](#10-what-remains)

## 1. Status

- **The first fill (C1, 2026-10-14): all four rank files hold every routed expert at NVFP4**, 101.369 GiB
  each (108,844,417,024 B), no reserved entry left: layers 3–77 NVIDIA's own NVFP4 bytes re-tiled, the MTP layer
  nvfp4_rtn of its BF16; installed on every rank (§ 8), the non-expert-only files kept beside them as
  `rankR.glw.prefill-only`. Verified with every check in § 7 (§ 6, "The first fill, run").
- **Files for all four ranks written and verified**: 6.444 GiB each (6,919,143,424 B): every non-expert tensor
  (5.178 GiB) plus the MTP layer's 256 routed experts at NVFP4 (1.266 GiB); 21,377 TOC entries of which 19,275
  are reserved routed-expert entries (layers 3–77). Every byte checked against the master by an
  independent description, against a fresh conversion, and by CRC; the host binding against `attn.h` and
  `experts.h` passes on every rank.
- **FP8 reproduces zai's release bit for bit**: 20 of 20 tensors, 924,975,104 codes and 56,480 scales equal,
  once the scale is computed as zai does (`amax × f32(1/448)`, not `amax / 448`; § 4).
- **Routed experts**: the TOC reserves one entry an expert with a per-matrix format tag; the NVFP4 fill from
  NVIDIA's tensors is implemented and was run for one layer of one rank into a separate file (layer 3, rank 0):
  28.3M values read back through `layout.h`'s own readers, 0 differ. The full fill of layers 3–77 (~95 GiB a
  rank) is not run.
- **The MTP layer's experts are filled** (2026-10-14) by our NVFP4 round-to-nearest-even of their BF16 with
  NVIDIA's recipe (`quant/formats.py` `nvfp4_rtn`), which reproduces NVIDIA's own NVFP4 bytes on 18 of 18
  layer-3 tensors from the BF16 master (§ 6). Relative RMS error against the BF16 0.0945–0.0953; outside the
  exactness bar (DESIGN decision 6: the MTP layer only drafts).

## 2. The file format

The DeepSeek engine's `.dsw` container ([ds41f layout.md](../../ds41f-4xgx10-switchless/docs/layout.md),
`dsw.h`) kept in design, this engine's own copy: magic `GLM53W01`, version 1, extension `.glw`
([glw.h](../engine/weights/glw.h); [glw.py](../engine/weights/glw.py) mirrors the structs field by field).

```
[0, 4096)                  Header (rank, world, TOC offset and CRC32C, sizes, model, source, created, tool)
[4096, data_offset)        n_tensors TocEntry records of 512 B, zero-padded to 4 KiB
[data_offset, file_bytes)  tensors in TOC order, each on a 4 KiB boundary, zero-padded to 4 KiB
```

A TOC entry: name, source tensor(s), a readable cut, dtype, natural shape, offset, bytes, kind, layout, CRC32C,
layer (or global), the FP8 scale block `blk_rows × blk_k`, a format tag, flags, the relative error of the
source(s) against the master, and `reserve_bytes`. The device address of a tensor is arena + (offset −
data_offset), as for DeepSeek; the loader packs a selection in file order.

| Field | Values |
|---|---|
| dtype | F32, BF16, F8_E4M3, U8 (expert blocks, descriptor tables), U32 (the RoPE table) |
| kind | replicated, rows, cols, recut (several sources, transposed or interleaved: the attention re-cuts), generated (the RoPE table), reserved (no data yet) |
| layout | natural (row-major); **T64**: the attention GEMV tile `[N/8][K/64][32 lanes][16 B]` (pack.py `tiles`); **T128**: the expert kernels' shared-expert tile `[N/8][K/128][32 lanes][32 B]` (layout.h `pack_fp8`); **expert-pool**: 8-row × 512-k blocks (layout.h `pack_nvfp4`/`pack_s3`) |
| flags | lossy (FP8 from the master), reserved, scale (an FP8 weight's f32 scale table), interleaved (gate/up rows 2i, 2i+1) |
| fmt | expert entries: one nibble a matrix, gate \| up << 4 \| down << 8, each `gx::Fmt` (0 NVFP4, 1 S3); else none |

Byte (n, k) of a T64 tensor sits at `((n/8)(K/64) + k/64)·512 + (4(n%8) + (k%64)/16)·16 + k%16`; of a T128 one at
`(((n/8)(K/128) + k/128)·32 + 4(n%8) + (k%64)/16)·32 + 16((k%128)/64) + k%16` (`glw::t64_offset`,
`t128_offset`). Scale tables are f32 `[N/rs][K/ks]`, or `[N/2/rs][2][K/ks]` for an interleaved gate/up pair
(attn.h `GemvOut::kSwiGLU`); `rs × ks` is in the TOC of both the weight and its scales.

**Reserved entries** have offset 0, bytes 0, the reserved flag and `reserve_bytes`. They are the last entries of
the TOC, so filling them appends to the file and rewrites the TOC and header: no other byte moves and every
non-expert tensor keeps its offset, CRC and SHA-256.

**Rank check.** The header names the rank and world it was cut for; `WeightFile(path, rank, world)` dies on any
other rank's file with the path and both ranks (`glw_rank_test`: refused as each of the three other ranks, opened
as its own).

## 3. The plan

Built from the checkpoint's headers alone (`convert.py --plan`, `--list` for every tensor); rank r of 4. The FP8
re-cuts are pack.py's own functions, called by the converter.

| Tensors (engine names) | Source in NVIDIA's checkpoint | Format, layout | Rank r holds | Bytes a rank |
|---|---|---|---|---:|
| `embed.weight` | `model.embed_tokens.weight` [154880, 6144] | BF16 natural | vocab rows [38720r, +38720) | 475.8 MB |
| `head.weight` | `lm_head.weight` | BF16 natural (the BF16 GEMV reads rows) | vocab rows [38720r, +38720) | 475.8 MB |
| `norm.weight` | `model.norm.weight` | BF16 | all | 12 KB |
| `rope.table` | generated by `attn/tools/rope_table.py` (torch, as the reference) | U32 [2^20][32] | all | 134.2 MB |
| `layers.L.{attn_norm,ffn_norm,attn.q_norm,attn.kv_norm}.weight` | `input_layernorm`, `post_attention_layernorm`, `q_a_layernorm`, `kv_a_layernorm` | BF16 | all | 20 KB a layer |
| `layers.L.attn.xproj` | `q_a_proj`, `kv_a_proj_with_mqa`, indexer `wk` | FP8 T64, scales 16 × 128 | q_a [512r, +512), kv_a [128r, +128) and [512+16r, +16), wk [32r, +32) on full layers: 688 (656) × 6144 | 4.23 (4.03) MB |
| `layers.L.attn.q_b` | `q_b_proj` [16384, 2048] | FP8 T64, 128 × 128 | rows [4096r, +4096): heads 16r…16r+15 | 8.39 MB |
| `layers.L.attn.w_uk` | `kv_b_proj` [28672, 512] | FP8 T64, 128 × 64 | head H's rows 448H…+191 transposed to [512][192], 16 heads stacked | 1.57 MB |
| `layers.L.attn.w_uv` | `kv_b_proj` | FP8 T64, 64 × 128 | head H's rows 448H+192…+255, [256][512], stacked | 2.10 MB |
| `layers.L.attn.o_proj` | `o_proj` [6144, 16384] | FP8 T64, 128 × 128 | cols [4096r, +4096) | 25.17 MB |
| `layers.L.indexer.wq_b` (22 full layers incl. 78) | `indexer.wq_b` [4096, 2048] | FP8 T64, 128 × 128 | rows [1024r, +1024): index heads 8r…8r+7 | 2.10 MB |
| `layers.L.indexer.weights_proj.weight` | `indexer.weights_proj` [32, 6144] | BF16 natural | rows [8r, +8) | 98 KB |
| `layers.L.indexer.k_norm.{weight,bias}` | `indexer.k_norm` | BF16 | all | 0.5 KB |
| `layers.{0,1,2}.mlp.gate_up` | `mlp.gate_proj`, `up_proj` **from zai's BF16 mirror** | FP8 T64, interleaved, scales [24][2][48] | intermediates [3072r, +3072), rows 2i gate, 2i+1 up | 37.75 MB |
| `layers.{0,1,2}.mlp.down` | `mlp.down_proj` (mirror) | FP8 T64, 128 × 128 | cols [3072r, +3072) | 18.87 MB |
| `layers.L.shared.{gate,up}` (L = 3…78) | `shared_experts.gate_proj`, `up_proj` | FP8 **T128** (the expert kernels' `SharedW`), scales [4][48] | rows [512r, +512) | 3.15 MB each |
| `layers.L.shared.down` | `shared_experts.down_proj` | FP8 T128, scales [48][4] | cols [512r, +512) | 3.15 MB |
| `layers.L.router.weight`, `.bias` | `mlp.gate.weight` BF16 [256, 6144], `e_score_correction_bias` F32 | as shipped | all | 3.15 MB |
| `layers.78.mtp.{enorm,hnorm,head_norm}.weight` | `enorm`, `hnorm`, `shared_head.norm` | BF16 | all | 36 KB |
| `layers.78.mtp.eh_proj` | `eh_proj` [6144, 12288] | FP8 T64, 128 × 128 (our RTN by zai's rule; BF16 in zai's release) | rows [1536r, +1536) | 18.87 MB |
| `layers.L.experts.desc`, `layers.L.experts.E` (L = 3…78) | `experts.E.{gate,up,down}_proj.*` | reserved (§ 6) | gate/up rows, down cols [512r, +512) | 0 (1.27 GiB a layer when filled at NVFP4) |

Layer 78 (the MTP layer) is a full layer in this table: its attention, full indexer, router, shared expert, plus
the three norms and `eh_proj`. Shared layers have no indexer tensors in the checkpoint (21 full layers + 78).

**Totals a rank** (every rank is the same size):

| Class | GiB | quantisation.md § 2 (FP8 column, our split) |
|---|---:|---|
| embedding (vocab-split), head, final norm, RoPE table | 1.011 | embed 0.44 + head 0.44; the RoPE table (0.125) is new here |
| attention, layers 0–77 (incl. the indexer, split) | 3.047 | attention 3.00 + indexer: § 2 has it replicated (0.18 FP8); attention.md splits it (0.05) |
| dense MLPs | 0.158 | 0.16 |
| shared experts + routers, layers 3–77 | 0.879 | 0.66 + 0.22 |
| MTP layer: attention, shared expert, router, eh_proj, norms | 0.071 | 0.07 |
| **file** (incl. 10.9 MB of TOC) | **5.178** | A2's non-experts 5.11 (with a replicated indexer, no RoPE table) |
| routed experts when filled at NVFP4 (75 layers + MTP) | 96.19 | 94.92 + the MTP layer's 1.27 |

Coverage (`glw.coverage`, run by every `convert.py`): all 232,403 checkpoint tensors are accounted for: 1,208
are sources of exactly one unit, 231,168 are routed-expert tensors behind the reserved entries, and 27 (the dense
MLPs' per-tensor FP8 weight, scale, input scale) are replaced by the master from zai's BF16 mirror, because
NVIDIA requantised them per tensor (0.46 % of codes equal to the block FP8, [quantisation § 3.0](quantisation.md)).

## 4. FP8 conversion: zai's rule, reproduced

**The rule** ([quant.py](../engine/weights/quant.py)): per 128 × 128 block of the BF16 master (the last block row
may be partial, e.g. `kv_a`'s 576 rows), `scale = f32(amax) × f32(1/448)` (a multiply by the rounded
reciprocal), `codes = e4m3_rne(f32(w) / scale)`; the scale is stored as zai's `weight_scale_inv`
(W = e4m3 × scale). No block of the model is all zero (the converter dies on one).

**Reproduction** ([fp8check.py](../engine/weights/fp8check.py); zai's tensors fetched by HTTP range request from
`zai-org/GLM-5.3` at `c5087a76d148760811ffbdedfe6ff9f194e46e38`, as `tools/provenance.py` does; the dense MLPs'
master from `zai-org/GLM-5.3-BF16` at `9d2398f478cab2de883137db3a36ad2c96205e24`), 2026-10-13:

| Tensors | Master | Codes equal | Scales equal (bitwise) |
|---|---|---|---|
| L0 q_a, L5 kv_a (partial block), L20 q_b, L33 kv_b, L77 o_proj, L42 indexer wq_b and wk, L3 shared up, L60 shared down, L78 q_b, L78 shared gate | NVIDIA's BF16 | all | all |
| L0–2 dense gate, up, down (9) | zai's BF16 mirror | all | all |
| **20 of 20 identical** | | **924,975,104 / 924,975,104** | **56,480 / 56,480** |

**The first rule tried was not zai's.** `pack.py`'s `fp8_from_master` computed `scale = amax / 448` (a
division) until 2026-10-14, when it was changed to zai's rule (byte-identical to the converter on 3 of 3 tensors). On the same 20 tensors: 0 of 20 identical, codes 924,205,615 / 924,975,104 equal (99.917 %),
scales 26,249 / 56,480 (46.5 %), every differing scale one ulp away; with zai's own scales the codes were 100 %
equal, which isolated the scale. Candidates tried on three tensors: `amax/448` in f32 or f64, `1/(448/amax)`
in f32 or f64; only `amax × f32(1/448)` matched (48/48, 768/768, 240/240). `pack.py` is the attention
agent's test-data generator (its random masters are not zai's tensors), so its re-cut functions stay the
reference and only the quantiser here differs; `fp8_from_master` must adopt the multiply (the attention side's
change; § 10).

**Error against the master** (relative RMS, per source tensor, in every manifest): 0.0258–0.0265 for every
class (q_a, kv_a, q_b, kv_b, o_proj, indexer wq_b and wk, shared and dense gate/up/down, eh_proj), matching the
0.026 of quantisation § 3.0.

## 5. The manifest

Beside each file, `rankR.glw.manifest.json` (written by `convert.py`, extended by `experts_fill.py`):

- format, version, rank, world, file bytes, **the whole file's SHA-256**, the TOC's CRC32C, created, seconds;
- the tool: SHA-256 over its sources (`glw.py`, `convert.py`, `quant.py`, `sources.py`, `fetch.py`,
  `crc32c.c`, `attn/tools/pack.py`, `attn/tools/rope_table.py`; the first 16 hex digits are in the header),
  each file's SHA-256, the commit when given (`--commit`), argv, torch and numpy versions;
- the master: NVIDIA's snapshot (`models--nvidia--GLM-5.3-NVFP4@e3b8f9f2…`), the BF16 mirror's revision, the
  FP8 rule;
- every source tensor read: origin, **SHA-256 of its bytes as read**, dtype, shape, whether quantised, and the
  **relative RMS error** of its FP8 against the master;
- every TOC entry: offset, bytes, CRC32C, **SHA-256**, relative error, kind, layout, sources;
- after a fill: the expert entries' SHA-256s, the expert source tensors' SHA-256s, the fill's record.

## 6. The routed experts: reserved entries and the fill

**The TOC** of every MoE layer (3–77 and the MTP layer 78) holds `layers.L.experts.desc` (`gx::MatDesc
[256][3]`: gate, up, down) and one entry an expert, `layers.L.experts.E`: its gate, up and down matrices back
to back, each in the expert kernels' 8-row × 512-k blocks of its own format, with the format tag `fmt` = gate |
up << 4 | down << 8. Every matrix is a multiple of 4 KiB in either format (NVFP4 1,769,472 B, S3 1,376,256 B),
so the 256 expert entries lie back to back with no padding: **the layer's pool starts at expert 0's entry**,
`MatDesc::off` is relative to it, and the loader's arena keeps it contiguous. Reserved entries are sized for
NVFP4 (5,308,416 B an expert, 1.27 GiB a layer); an S3 expert is 4,128,768 B.

**The first fill** ([experts_fill.py](../engine/weights/experts_fill.py)): NVIDIA's NVFP4 codes and scales, the
rank's slice (gate/up rows [512r, +512); down input columns [512r, +512) = value bytes [256r, +256), scale
columns [32r, +32)), re-tiled by a numpy port of `layout.h`'s `pack_nvfp4` (the nibbles unchanged);
`MatDesc{off = (3e + m) · 1,769,472, s2 = weight_scale_2, fmt = kNVFP4}` (`make_desc` with zero tables).
Layers must be filled in TOC order (`glw_check` requires offsets in TOC order), so filling layers 3–77 now means
truncating the files at layer 78's descriptor table and filling 3–78 again (layer 78 takes 83 s for four ranks). Several `--file`s are
filled in one pass over the sources; each expert entry is written at its precomputed offset (the file's end +
the 49,152 B descriptor table + e × 5,308,416), then the descriptor table, the TOC and the header.

**The MTP layer (78)**: its experts are BF16 in NVIDIA's file (`model.layers.78.mlp.experts.E.{gate,up,down}_proj.weight`,
[2048][6144] and [6144][2048]). `experts_fill.py` quantises each full tensor once with `quant/formats.py`
`nvfp4_rtn` (NVIDIA's recipe: s2 = amax × f32(1/2688), gate and up sharing amax = max(amax_gate, amax_up); e4m3
scale per 16 along K; E2M1 round to nearest even, low nibble first) and then cuts the rank's slice as above.
The entries carry `kLossy` and `rel_err` (the expert's largest relative RMS error against the BF16 over its three
matrices); the manifest's `expert_rel_err` has every matrix's, `expert_sources` the BF16 sources' SHA-256s, and
`expert_fills` the format note (outside the exactness bar, DESIGN decision 6) and the SHA-256s of
`experts_fill.py` and `formats.py`. **The recipe checked first** (`experts_fill.py --rtn-check`): `nvfp4_rtn` on
the BF16 master of layer 3's experts 0, 1, 2, 3, 100, 255 (`/data/scratch/glm53-master`, its fetch SHA-256s
checked) gives NVIDIA's `weight`, `weight_scale` and `weight_scale_2` bytes exactly: 18 of 18 tensors.

**The MTP fill, run** (2026-10-14): all four rank files in one pass, 2 min 36 s wall, 1.7 GB peak, nice 19, CPU
only; relative RMS error 0.0945–0.0953 over the 768 matrices. Checks: `glw_check`; `verify.py --crc`, `--indep`
(now also every layer's descriptor table: offsets, format, scale_2 from the BF16 maxima; and experts 0, 1, 17,
128, 255 of a rank un-tiled by the byte formula, written out per index, against `nvfp4_rtn` of the BF16 cut to
the rank), `--fresh`; `glw_kcheck --rank R` (102,077 checks a rank, 1,024 more than before: the layer's sizes by
tag, contiguity, descriptors); `glw_expert_fill_test` on experts 0, 17, 255 of every rank (28,311,552 values a
rank read through `gx::read_q4`, 0 differ). Canonical dumps in `test-l3/canon78/`.

**One layer, one rank, run** (2026-10-13): `convert.py --ranks 0 --only layers.3.` then `experts_fill.py
--layers 3`: `/data/dsgb10/glm53/weights/test-l3/rank0.l3.glw`, 1,413,062,656 B (layer 3's attention,
shared expert, router and all 256 experts), 277 entries, SHA-256
`2d60e39ff59294f51e05f6154f58fc0d0ac11b5c5a9d333a37d6d2af18ee068e`, TOC CRC `a7e87fee`; the fill took 11 s
and 2.0 GB of memory. Checks: `glw_check` (CRCs, padding, geometry), `verify.py --crc` (manifest SHA-256s),
`glw_kcheck --partial --layers 3` (2,355 checks: sizes by format tag, contiguity, every descriptor's offset and
format), and `glw_expert_fill_test` on experts 0, 17, 255: every value of every matrix read through its
descriptor with `gx::read_q4` equals `gx::canon_q4` of the canonical slice (28,311,552 values, 0 differ).

**The first fill, run (C1, 2026-10-14)**: on rank 2 (its local copy of NVIDIA's checkpoint, same revision; the
copied 6.9 GB files as the start), `experts_fill.py --reset-from 78 --layers 3 … 78` over all four files in one
pass: the MTP layer (filled before 3–77) returned to reserved and the files cut at its descriptor table, then
layers 3–78 in TOC order, each layer committed (data synced, then TOC, header, manifest) so a rerun resumes after
the last whole layer. 48 min for the 76 layers (35–37 s a layer for four ranks, ~15 s of it after the last expert: the descriptor
tables and the commit's sync of 5.1 GB; layer 78 161 s with its quantisation), then 12 min for the four files' SHA-256; 1.4 GB peak, nice 19,
8 threads on cores 10–19, CPU only. Every NVFP4 matrix written is read back (`unpack_nvfp4`) against its source
slice before its entry is committed: 0 failures, no checkpoint tensor out of shape or with a non-positive
scale_2. Layer 78 came out byte-identical to the earlier MTP fill (257 of 257 entry SHA-256s) and every
non-expert entry kept its offset and SHA-256 (1,845 of 1,845). Checks: `glw_check` (CRCs of all 21,377 entries,
no reserved entry), `glw_kcheck --rank R` (178,877 checks a rank, 0 bad), `glw_expert_fill_test` on experts 0
and 255 of layers 3, 20, 40, 60, 77, 78 of every rank (18,874,368 values each, 0 differ), `verify.py --crc`
(every tensor's and the files' SHA-256, 24 min), `--indep` (6,596 checks, 0 bad: every layer's descriptor table
and experts 0, 1, 17, 128, 255 of every layer and rank against NVIDIA's bytes; 17 min), `--fresh` (7,380 of
7,380). Copies: rank 0's pulled by rsync (355 s, 306 MB/s), ranks 1 and 3 streamed through rank 0 (179–197 s,
~550 MB/s), each hashed on arrival, then renamed into place (the old file hard-linked as `.prefill-only` first,
so the name never goes missing). Logs: `~/glm53-fill/logs/` on rank 2.

**The second fill** (S3 from the quantiser) writes the same entries with their own tags; it rewrites the expert
tail (truncate at the first expert entry, append again), leaving the non-expert bytes in place. Its packing is
[s3x2_pack.py](../engine/weights/s3x2_pack.py) (E1, [experts.md § 13](experts.md#13-2026-10-14-e1-s32-the-quantisers-format-in-the-engine)):
`quantise_expert` → `rank_slice` → `expert_entry` / `expert_descs`; each entry's `fmt`, `shape` and `reserve_bytes`
must be re-declared for its format (mix 4,521,984 B, S3×2 4,128,768 B). Tried on a copy of the layer-3 test file
(`tests/s3x2_layer_test.py`, 2026-10-14): `glw_check`, `glw_kcheck --partial --layers 3 --blocks` (3,635 ok) and
`glw_expert_fill_test` (0 of 28,311,552 values differ) pass. `glw_kcheck` now checks every S3 descriptor's
tables against `make_desc` and, with `--blocks`, every expert scale byte.

**The converter's S3 fill (C2-prep, 2026-10-14)**: `experts_fill.py --recipe {nvfp4,mix,s3x2} --master BF16DIR`
quantises each expert once from its BF16 with `s3x2_pack.quantise_expert` (the master's tensors checked against
their fetch SHA-256s as they are read; the MTP layer's BF16 from NVIDIA's file) and cuts all four ranks from that
one result (`rank_slice` → `expert_entry` / `expert_descs`), so the quantisation costs once, not four times.
Each layer's 257 entries are re-declared for the recipe's tag (fmt, shape, `reserve_bytes`: mix 0x011, 4,521,984
B an expert; S3×2 0x111, 4,128,768 B) before they are written; layers stream and commit one at a time and a
rerun resumes as in the NVFP4 fill; entries carry `kLossy` and `rel_err`. The manifest gains `expert_layers`
(per layer: recipe, tag, entry bytes, source, the subset pair, the master's shard names and the SHA-256 of its
fetch manifest), `expert_sources` (each BF16 tensor's SHA-256), `expert_rel_err`, and a fill record with the
recipe, subsets, the SHA-256s of `experts_fill.py`, `s3x2_pack.py` and `formats.py`, and the provenance counts.
(The master is fetched by range, so no whole-shard SHA-256 exists to record; the per-tensor ones stand in.)
**Provenance**: with `--provenance all.jsonl` every S3×2 tensor that quant/run.py has a record of must hash
(run.py's `phash` of the packed dict) to the record; every NVFP4 tensor quantised from the master of layers 3–77
is compared with NVIDIA's own bytes instead (all.jsonl's `nvfp4` records come partly from an earlier
`nvfp4_rtn`: layer 3 expert 18's gate and up differ from NVIDIA's there and agree in `results-nvfp4v2.jsonl`).
Without `--master`, `--recipe nvfp4` copies NVIDIA's bytes as in C1.

Test on copies of the four 6.9 GB files under `/data/scratch/glm53-c2prep/` (rank 0, nice 19, 8 threads):
- `--recipe nvfp4 --master` on layers 3 and 40: 1,536 of 1,536 NVFP4 tensors equal to NVIDIA's bytes, and the 514
  layer-3 and layer-40 entries byte-identical to C1's (the master path and the copy path agree); 99–101 s a layer.
- `--reset-from 3 --recipe mix` on layers 3 and 40: 884 S3×2 tensors equal to the quantiser's records, 140
  without a record (experts it did not run), 512 NVFP4 downs equal to NVIDIA's; relative RMS error against the
  BF16 0.085–0.144 (S3×2 gate/up ~0.14, NVFP4 down ~0.095). `glw_check` OK; `glw_kcheck --partial --layers 3,40
  --blocks` 7,258 checks a rank, 0 bad, 1,024 S3 descriptors with the pair, every scale byte scanned (~39 % of
  the S3 groups in subset B); `glw_expert_fill_test` on experts 0, 17, 255 of both layers and every rank, 0 of 28,311,552
  values differ; `verify.py --crc` OK (`--indep` leaves non-NVFP4 expert entries to those two and says how many).
  Files 7,875,493,888 B; SHA-256 rank 0 `572945d9…1dad`, 1 `59f14145…0b72`, 2 `d3574a58…70a5af`, 3 `cb8b4c8c…aafa42`.
- **Time**: 377–381 s a layer for four ranks, 338–341 s of it reading and quantising (levelset3's search,
  ~1.3 s an expert), 2.0 GB peak. **Projected full run** (layers 3–78): 76 × ~380 s ≈ **8 h** at 8 threads,
  quantisation-bound (more cores or processes per layer would scale it), output 93,543,596,032 B a rank (87.12
  GiB) with the MTP layer also in mix, +201,326,592 B if it stays NVFP4.

## 7. Verification

| Check | Tool | What | Result on the four files |
|---|---|---|---|
| Structure and CRCs | `glw_check` (C++), `verify.py --crc` | header geometry, TOC CRC, offsets in order, shape × dtype = bytes, reserved entries empty, every CRC32C, zero padding; the manifest's per-tensor and whole-file SHA-256s | OK, 3 s a file |
| Fresh conversion | `verify.py --fresh` | every tensor rebuilt by the converter's own code and compared byte for byte (determinism) | 7,380 of 7,380 equal |
| **Independent** | `verify.py --indep` | without pack.py, quant.py or the plan: the natural matrix each FP8 tensor should hold, written out again from attention.md as source row/column pieces (transposes, interleaving); scales by the rule in numpy; every element's scale read through the TOC's rs × ks must be its source block's; every code un-tiled with glw.h's formula must be `e4m3_rne(master / scale)`; BF16/F32 tensors byte-equal to their source rows; the RoPE table equal to rope_table.py's; replicated tensors equal across ranks | 4,796 checks (24 of them the MTP experts, § 6), 0 bad; a flipped code bit and a flipped scale bit are both caught |
| Kernel binding | `glw_kcheck --rank R` | every layer against attn.h (`kXRows`, `kHeads`, …, the GemvSeg rs/ks, `full_indexer`), experts.h (`mat_bytes`, `MatDesc`, `SharedW` when T128), the head/embed/RoPE shapes; binds each FP8 tensor into a `glm::GemvSeg` | 102,077 checks a rank, 0 bad |
| Rank refusal | `glw_rank_test --file F R` | the loader refuses the three other ranks | all ok |
| Device | `glw_kcheck --gpu` | the chosen layers' GEMVs (`glm::gemv`, SwiGLU pairs, grouped W_UK/W_UV, the BF16 head) against a double reference from the file's own bytes; kF32 within 1e-4 of Σ\|x·w\|, SwiGLU within 2 bf16 ulps plus the f32 sums' rounding carried through (below) | rank 0, layers 0, 3, 10, 78 and the head: 105,136 ok, 1 bad under the old SwiGLU bound (below); layer 0 again with the new bound: 102,142 ok, 0 bad |

`verify.py --crc --fresh --indep` over all four files: ~14 min (the MTP descriptors' scale_2 re-reads the BF16), 5.6 GB peak, nice 19, no GPU.

**The SwiGLU bound (2026-10-14).** `glw_kcheck --gpu`'s first run found one of 12,288 SwiGLU outputs of
`layers.0.mlp.gate_up` outside the plain 2-ulp bound: row 1, intermediate 659 (flat 3,731 of the [4][3072]
output). The kernel gave 1.48266554e-06, kcheck's reference 1.50501728e-06, the exact value in f64
1.50243735e-06; the bound was 1.49e-08 (2 ulps), the difference 2.24e-08 (3.0 ulps). The cause is the bound,
not the kernel or the reference: the gate's sum cancels, g = -3.33342485e-06 against Σ\|x·w\| = 52.36 (6.4e-8 of
it, under f32's epsilon), so the kernel's f32 sum (-3.27324096e-06, off by 6.0e-08 = 1.1e-9 of Σ\|x·w\|, an
ordinary rounding error) is 1.8 % from the exact gate, and SiLU near 0 (≈ g/2) passes that straight to the output.
The up value (-0.901439162) is unremarkable. Reproduced on the CPU first: the same inputs (mt19937(1234) with the
earlier draws replayed), kcheck's double reference, and an emulation of `k_gemv`'s sums (each 64-k chunk exact,
rounded to f32, times its scale, added in k order) flag that one output, no other, and give the kernel's
value to the bit; the ring job then printed the same value from the GPU. A wrong byte would not cancel only
this one output, and the reference agrees with the exact value to 0.35 ulp. **The rule now:** the kernel's g is
within E = 2⁻²² Σ\|x·w·s\| (MMA-internal) + 2⁻²⁴ Σⱼ (\|Sⱼ\| + \|sⱼcⱼ\|) (the scale multiply and the running sum,
Sⱼ the exact sum after chunk j) of the exact one, and likewise u; the reference is recomputed at g ± E_g,
u ± E_u, and the output may be 2 ulps beyond the widest corner (`swiglu_tol` in kcheck.cu). Here E_g = 1.46e-05,
the bound 6.62e-06; the output is inside. On these inputs the rule widens 569 of 12,288 bounds (cancelling gates
and values on a bf16 rounding boundary) and still catches 3,997 of the 4,035 outputs that a wrong scale block on
one chunk moves out of the old bound (a worst-case 3·(K/64)·2⁻²³ Σ\|x·w\| was tried first and dropped: it caught
2,359). kcheck prints every output outside the plain 2 ulps with its values and the verdict.

## 8. Files written

**The first fill (C1, 2026-10-14)**, every routed expert at NVFP4 (layers 3–77 NVIDIA's bytes, layer 78
nvfp4_rtn); the non-expert bytes are those of the table below (same converter run, T128 shared expert). Installed
as `rankR.glw` with its manifest: rank 0 in `/data/dsgb10/glm53/weights/`, ranks 1–3 in `~/glm53/weights/` on
their boxes; the files below kept beside them as `rankR.glw.prefill-only`.

| File | Bytes | SHA-256 | TOC CRC |
|---|---:|---|---|
| `rank0.glw` | 108,844,417,024 | `f1010873955c07b6a9358c990cc4d48322da356957f66dab486f1ac284bb21e9` | `e9576659` |
| `rank1.glw` | 108,844,417,024 | `e4c1b6897aebce72e60a5c39834c3c8326037e597eeeb82596d35bf7e035d55f` | `2b56d7c8` |
| `rank2.glw` | 108,844,417,024 | `f8b0c628a5139c94aa2ca2f4951c2a39dec702d711bd0a0d58190814ac630da8` | `7ea92e9f` |
| `rank3.glw` | 108,844,417,024 | `a0c8e8cd80171a99e0f98114128d2691e9cd008c9e6ea5e33022ab85a3482f4e` | `6d1ccdb3` |

The MTP-only files (now `rankR.glw.prefill-only`):

`/data/dsgb10/glm53/weights/` on rank 0 (converted 2026-10-13 23:19, tool `63b15f221d924873f8a6358e1e2e8cd46bf41cb290005df4715788c41a75e4a5`, `--commit e33e73f`,
shared expert in T128; the MTP layer's experts filled 2026-10-14, § 6). The workers' copies (`~/glm53/weights/rank{r}.glw`
on ranks 1–3) held the 5,560,139,776-byte files without layer 78's experts until 2026-10-14 (G7 found the mixed state: rank
0 filled, ranks 1–3 not; `glm53_rank` now warns at start when the ranks' files differ); they were replaced that day by
atomic rename with the filled files, SHA-256 verified against this table:

| File | Bytes | SHA-256 | TOC CRC |
|---|---:|---|---|
| `rank0.glw` | 6,919,143,424 | `3d1a4f122a14e105f434ca9d78f495262e6d2ce838ccf6a2b4f4ce47ad738923` | `1f668ded` |
| `rank1.glw` | 6,919,143,424 | `4813956771c220263b9cbc0b5198ca796e30e7bba33778c9a3eaa81b4e4f2d17` | `76cf6d29` |
| `rank2.glw` | 6,919,143,424 | `0f63b55a2fafbe71ba016f6a62e798e8fc3b68577ee8342918540bf6df54c853` | `e0869b45` |
| `rank3.glw` | 6,919,143,424 | `5cdf0bade9d44713fd0a9106b53b011580bdb2fa72a0926fe9ec9be13117fd39` | `13dec971` |
| `test-l3/rank0.l3.glw` (layer 3 with its NVFP4 experts; cut before the T128 decision, tool `5dafdb3e…`, so its shared expert is T64) | 1,413,062,656 | `2d60e39ff59294f51e05f6154f58fc0d0ac11b5c5a9d333a37d6d2af18ee068e` | `a7e87fee` |

All four ranks in one pass: **2 min 47 s** wall (2 min 7 s in an earlier run), 5.0 GB peak resident, nice 19, 4 torch threads, CPU only
(reads ~60 GB of sources; page cache dropped after every read and write). The fetched sources (zai's FP8
tensors for the check, the dense MLPs' master, 2.2 GB) are in `/data/dsgb10/glm53/fetch/` with their SHA-256s.

## 9. Tools

Python is the CPU-only torch environment, run with `-I` and `CUDA_VISIBLE_DEVICES=` at nice 19; `NV` is
NVIDIA's snapshot directory, `CACHE` the fetch cache. C++ builds with the engine (`cmake --build … -j4`):
`glw_check`, `glw_load`, `glw_rank_test`, `glw_kcheck`, `glw_expert_fill_test`, and `libglw_crc.so` (glw.py's
CRC32C; without it glw.py compiles `crc32c.c` into `~/.cache/glw/`).

```
python -I fetch.py CACHE --repo fp8|bf16 NAME ...                 # single tensors by range request
python -I fp8check.py --ckpt NV --cache CACHE [--rule pack]       # FP8 reproduction against zai's release
python -I convert.py --ckpt NV --cache CACHE --plan [--list]      # the plan, bytes a rank, coverage
python -I convert.py --ckpt NV --cache CACHE --out-dir DIR [--ranks 0 1 2 3] [--only SUBSTR] [--shared-layout t64|t128]
python -I verify.py DIR/rank{0,1,2,3}.glw --crc --fresh --indep --ckpt NV --cache CACHE
python -I experts_fill.py --ckpt NV --file DIR/rank{0,1,2,3}.glw --layers 3 ... 78 [--reset-from L] [--recipe nvfp4|mix|s3x2]
    [--master BF16DIR] [--provenance all.jsonl] [--threads 8] [--rtn-check MASTER] [--dump-canon D]   # rerun to resume
glw_check FILE ...;  glw_kcheck FILE --rank R [--gpu | --partial --layers L];  glw_rank_test --file FILE R
glw_load FILE --rank R [--verify-crc];  glw_expert_fill_test FILE R D L E ...
```

## 10. What remains

- **The shared expert is in the expert kernels' layout** (decided 2026-10-13): the MoE launch fuses it
  (`moe_decode`/`moe_prefill` read `SharedW`: gate, up, down in T128 with `[N/128][K/128]` scales), so that is the
  path the engine runs and the attention side's shared-expert GEMV (pack.py's interleaved T64 form) is unused.
  T128 is the default; `--shared-layout t64` still writes the other form (kcheck and verify check either).
- The NVFP4 fill of every layer is done and installed (C1, § 6, § 8). The S3 fill from the quantiser (C2) is ready
  (C2-prep, § 6) and waits for the BF16 master of layers 3–77; ~8 h at 8 threads. `verify.py --indep` does not yet
  re-derive S3 entries independently (kcheck --blocks and the fill test cover them).
- `glw_load` on the device, in a ring window (`glw_kcheck --gpu` ran 2026-10-14, § 7).
- `engine/attn/tools/pack.py`'s `fp8_from_master` used `amax / 448` until 2026-10-14; it now uses zai's rule
  (`amax × f32(1/448)`, § 4), checked byte for byte against the converter by `engine/attn/tools/fp8_vs_converter.py`.
- The router is replicated (0.22 GiB, as quantisation § 2); a split by output (64 rows a rank, logits
  all-gathered) is a one-line plan change if G4 prefers it.
- Ranks 1–3's first-fill files are installed on their boxes (§ 8); the fill's working copies of ranks 0, 1 and 3
  (~300 GB) are still in `~/glm53-fill/out/` on rank 2, to delete when no longer wanted.
