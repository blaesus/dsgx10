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

- **Non-expert files for all four ranks written and verified**: 5.178 GiB each (5,560,139,776 B), 21,377 TOC
  entries of which 19,532 are reserved routed-expert entries. Every byte checked against the master by an
  independent description, against a fresh conversion, and by CRC; the host binding against `attn.h` and
  `experts.h` passes on every rank.
- **FP8 reproduces zai's release bit for bit**: 20 of 20 tensors, 924,975,104 codes and 56,480 scales equal,
  once the scale is computed as zai does (`amax × f32(1/448)`, not `amax / 448`; § 4).
- **Routed experts**: the TOC reserves one entry an expert with a per-matrix format tag; the NVFP4 fill from
  NVIDIA's tensors is implemented and was run for one layer of one rank into a separate file (layer 3, rank 0):
  28.3M values read back through `layout.h`'s own readers, 0 differ. The full fill (~96 GiB a rank) is not run.

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
Layers must be filled in TOC order. The MTP layer's experts are BF16 in NVIDIA's file and are refused for now.

**One layer, one rank, run** (2026-10-13): `convert.py --ranks 0 --only layers.3.` then `experts_fill.py
--layers 3`: `/data/dsgb10/glm53/weights/test-l3/rank0.l3.glw`, 1,413,062,656 B (layer 3's attention,
shared expert, router and all 256 experts), 277 entries, SHA-256
`2d60e39ff59294f51e05f6154f58fc0d0ac11b5c5a9d333a37d6d2af18ee068e`, TOC CRC `a7e87fee`; the fill took 11 s
and 2.0 GB of memory. Checks: `glw_check` (CRCs, padding, geometry), `verify.py --crc` (manifest SHA-256s),
`glw_kcheck --partial --layers 3` (2,355 checks: sizes by format tag, contiguity, every descriptor's offset and
format), and `glw_expert_fill_test` on experts 0, 17, 255: every value of every matrix read through its
descriptor with `gx::read_q4` equals `gx::canon_q4` of the canonical slice (28,311,552 values, 0 differ).

**The second fill** (S3 from the quantiser) writes the same entries with their own tags; it rewrites the expert
tail (truncate at the first expert entry, append again), leaving the non-expert bytes in place.

## 7. Verification

| Check | Tool | What | Result on the four files |
|---|---|---|---|
| Structure and CRCs | `glw_check` (C++), `verify.py --crc` | header geometry, TOC CRC, offsets in order, shape × dtype = bytes, reserved entries empty, every CRC32C, zero padding; the manifest's per-tensor and whole-file SHA-256s | OK, 3 s a file |
| Fresh conversion | `verify.py --fresh` | every tensor rebuilt by the converter's own code and compared byte for byte (determinism) | 7,380 of 7,380 equal |
| **Independent** | `verify.py --indep` | without pack.py, quant.py or the plan: the natural matrix each FP8 tensor should hold, written out again from attention.md as source row/column pieces (transposes, interleaving); scales by the rule in numpy; every element's scale read through the TOC's rs × ks must be its source block's; every code un-tiled with glw.h's formula must be `e4m3_rne(master / scale)`; BF16/F32 tensors byte-equal to their source rows; the RoPE table equal to rope_table.py's; replicated tensors equal across ranks | 4,772 checks, 0 bad; a flipped code bit and a flipped scale bit are both caught |
| Kernel binding | `glw_kcheck --rank R` | every layer against attn.h (`kXRows`, `kHeads`, …, the GemvSeg rs/ks, `full_indexer`), experts.h (`mat_bytes`, `MatDesc`, `SharedW` when T128), the head/embed/RoPE shapes; binds each FP8 tensor into a `glm::GemvSeg` | 101,053 checks a rank, 0 bad |
| Rank refusal | `glw_rank_test --file F R` | the loader refuses the three other ranks | all ok |
| Device | `glw_kcheck --gpu` | the chosen layers' GEMVs (`glm::gemv`, SwiGLU pairs, grouped W_UK/W_UV, the BF16 head) against a double reference from the file's own bytes; kF32 within 1e-4 of Σ\|x·w\|, SwiGLU within 2 bf16 ulps plus the f32 sums' rounding carried through (below) | rank 0, layers 0, 3, 10, 78 and the head: 105,136 ok, 1 bad under the old SwiGLU bound (below); layer 0 again with the new bound: 102,142 ok, 0 bad |

`verify.py --crc --fresh --indep` over all four files: ~7 min, 5.6 GB peak, nice 19, no GPU.

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

`/data/dsgb10/glm53/weights/` on rank 0 (2026-10-13 23:19; tool `63b15f221d924873f8a6358e1e2e8cd46bf41cb290005df4715788c41a75e4a5`, `--commit e33e73f`,
shared expert in T128):

| File | Bytes | SHA-256 | TOC CRC |
|---|---:|---|---|
| `rank0.glw` | 5,560,139,776 | `c7d3341fb94849104bebbeca3ba79d1b4325827c8ada646ab154bf77e3b1094e` | `836dc158` |
| `rank1.glw` | 5,560,139,776 | `6e740f1533b4b0d0a3ab32f4e151c2edd1bf367ae5da21280f19ac4a24d53734` | `9cf9364f` |
| `rank2.glw` | 5,560,139,776 | `f5d179aa996f8ca19ab1ca2a679a2c17bed95c7668e57b3d02580c1870e9132c` | `751ad407` |
| `rank3.glw` | 5,560,139,776 | `3feef748c4fb71a7c8f4a34bf637738edca720ce0d27fa1e525ff4f9de0cc1c7` | `781d1799` |
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
python -I experts_fill.py --ckpt NV --file DIR/rankR.glw --layers 3 ... [--dump-canon D]
glw_check FILE ...;  glw_kcheck FILE --rank R [--gpu | --partial --layers L];  glw_rank_test --file FILE R
glw_load FILE --rank R [--verify-crc];  glw_expert_fill_test FILE R D L E ...
```

## 10. What remains

- **The shared expert is in the expert kernels' layout** (decided 2026-10-13): the MoE launch fuses it
  (`moe_decode`/`moe_prefill` read `SharedW`: gate, up, down in T128 with `[N/128][K/128]` scales), so that is the
  path the engine runs and the attention side's shared-expert GEMV (pack.py's interleaved T64 form) is unused.
  T128 is the default; `--shared-layout t64` still writes the other form (kcheck and verify check either).
- **The full expert fill** (NVFP4 first: ~96 GiB a rank, ~10 s a layer a rank of CPU, so the time is disk) is for
  the orchestrator to schedule; rank 1 needs ~20 GB freed (quantisation § 4). The MTP layer's experts (NVFP4 by
  our RTN of its BF16) and the S3 fill from the quantiser are later items.
- `glw_load` on the device, in a ring window (`glw_kcheck --gpu` ran 2026-10-14, § 7).
- `engine/attn/tools/pack.py`'s `fp8_from_master` used `amax / 448` until 2026-10-14; it now uses zai's rule
  (`amax × f32(1/448)`, § 4), checked byte for byte against the converter by `engine/attn/tools/fp8_vs_converter.py`.
- The router is replicated (0.22 GiB, as quantisation § 2); a split by output (64 rows a rank, logits
  all-gathered) is a one-line plan change if G4 prefers it.
- Ranks 1–3's files are on rank 0's disk; copying them to their boxes (and recording the copies' hashes) is G4's.
