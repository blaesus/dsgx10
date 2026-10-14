# What carries over from the GLM-5.3 engine

How much of [glm53-4xgx10-switchless](../../glm53-4xgx10-switchless/) (the GLM-5.3 engine on the same four GB10
boxes, paused 2026-10-14 with G0–G7 built and G8 partly done) carries over to AikidoSec/altar-1, what must change and
how much, what is new, and the build plan that follows. Written 2026-10-14 from the glm53 code and documents at
`962fdcf` (branch `glm53`), the DeepSeek engine at the same commit, altar-1's `config.json`, `chat_template.jinja`,
`tokenizer.json` and the headers of its 39 shards at revision `5d591cd98958e4e7e429517887e80ca3bb7ce2c4`. Nothing
was built or run for it. The budget is [quantisation.md](quantisation.md)'s.

Paths are relative to the repository root. `glm53/` stands for `glm53-4xgx10-switchless/`, `ds41f/` for
`ds41f-4xgx10-switchless/`. Sizes are lines of code, *estimates* unless counted.

## In short

- **altar-1 is GLM-5.3 with three differences**, all visible in the headers and the config (the only config change
  is `n_routed_experts` 256 → 168):
  - **168 routed experts a layer** (REAP-pruned, renumbered 0..167; the router `[168, 6144]` and its bias `[168]`);
  - **compressed-tensors INT4 at group 32, asymmetric** for the routed experts and `q_b_proj`, `kv_b_proj`,
    `o_proj` of layers 4–76 (values I32 `[N, K/8]`, BF16 scales `[N, K/32]`, zero points I32 `[N/8, K/32]`;
    the model card's "attention stays BF16" is wrong for those three);
  - **BF16 everything else**, including the routed experts of layers 3, 77 and the MTP layer 78, where glm53 has
    zai's FP8 (attention, indexer, shared experts, dense MLPs) and NVFP4 experts.
- **About 92 % of glm53's code carries over unchanged or with constants**: the ring and every collective, the
  container, loader and verifier skeleton, the attention kernels (prep, indexer, sparse MLA attention, RoPE), the
  BF16 GEMV and GEMM, the step machinery (graphs, PDL, L2 prefetch, the posted all-reduce), the sampler, the
  speculative loop, the paged store, disk tier and kept sessions, the server, the protocol, the scripts, the
  front-end (the template differs by five lines) and the reference's structure. glm53 is ~37k lines with tests,
  front-end and reference; the change is **~2.5–3.2k lines** new or changed (§ 9's table).
- **The model math does not change**, so neither does any arithmetic outside the weight formats. What changes is
  formats: one new expert format (INT4-g32 asymmetric), one new expert format family (BF16), a BF16 shared expert
  inside the expert kernels, an INT4 GEMV and a prefill path for three attention projections, and BF16 in place of
  FP8 on every other projection (already supported by glm53's BF16 GEMV at decode; three gaps at prefill).
- **The weights are the checkpoint, not a requantisation.** Every INT4 and BF16 byte is copied, re-tiled, from the
  published tensors. glm53's whole quantisation track drops out: no BF16 master download, no quantiser, no KL
  bake-off, no FP8 rule. The numerics contract defines the weight as `bf16((q − z) · s)`, compressed-tensors' own
  decompression ([quantisation § 6](quantisation.md#6-checks-made-on-the-data)).
- **Seven things block a naive port** (§ 10): two out-of-bounds reads that changing `kE` alone would create, the
  transposed `W_UK` against INT4's K-groups, FP8 hard-coded in prefill's projection helper and in two fused
  paths, the mandatory FP8 shared expert, silent mis-sizing of an unknown expert format, and FP8-only byte
  accounting.
- **Budget**: ~76.6 GiB a rank, a decode floor of **36.4 ms** (glm53: 34.2), one 256k BF16 session fits.

## 1. Component table

Verdicts: **unchanged** (copy, rename at most), **constants** (copy, change dimensions or names in one place),
**re-specialise** (same structure and file, new formats inside), **new**, **dropped**.

### Communication (`glm53/engine/comm`)

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Ring, proxy, device channel | `ring.cpp`, `ring.h`, `ring_dev.cuh`, `ring_config.h`, `fleet.cpp` | **unchanged** | nothing: no expert count or format in it |
| Collectives, the fused all-reduce + residual + norm, posted all-reduces, the x-projection all-gather | `collectives.cu/.h`, `glmcoll.cu/.h` | **unchanged** | the all-gather's row geometry (`kXRows` 688, `kXRowsShared` 656, `glmcoll.h:23-37`) is kept by keeping glm53's x-projection recut in BF16 (§ 3); the all-reduces see f32 partials only |
| Tests, `run4.sh`, `install.sh` | `tests/*`, `run4.sh` | **unchanged** | their 256s are thread counts |

### Weights (`glm53/engine/weights`)

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Container (header, 512-B TOC, 4 KiB alignment, CRC32C, rank check), C++ reader, loader | `glw.h`, `io.h`, `loader.cpp/.h`, `load_main.cpp`, `check.cpp`, `crc32c.c` | **constants** | magic `ALTAR1W1`; `ExpertFmt` gains `INT4 = 2`, `BF16 = 3` (`glw.h:99-102`); a `Layout` for the INT4 attention tile (`glw.h:73-88`, accepted by `loader.cpp:39-41`, `check.cpp:60-63`). ~15 C++ + ~20 Python |
| The plan | `glw.py` (`attn_unit` 195-253, `mlp_unit` 255-309, `mtp_unit` 311-332, `reserved_unit` 349-368, `coverage` 402-431) | **re-specialise** | § 4: INT4 and BF16 sources instead of FP8 re-quantisation, `N_EXPERTS` 168 (`glw.py:26`), the compressed-tensors names in `coverage`. ~100–150 |
| Checkpoint reader | `sources.py` | **constants** | `DT` (`sources.py:12`) lacks I32 and I64: a KeyError today. ~3 |
| FP8 rule, zai fetch, FP8 check | `quant.py`, `fetch.py`, `fp8check.py` | **dropped** | nothing is converted to FP8; the dense MLPs come from the checkpoint (BF16), not zai's mirror |
| Expert packers and fill | `experts_fill.py`, `s3x2_pack.py` | **re-specialise** / **dropped** | an INT4 and a BF16 recipe from the checkpoint (`experts_fill.py:39` `E = 256`, defaults at `:378`, `:382` name expert 255); an INT4 packer on `s3x2_pack.py`'s structure, which itself is dropped with S3. ~270 |
| Verifier | `verify.py` | **re-specialise** | `EXPERT_SAMPLE` (`:249`) and `range(256)` (`:277`), the NVFP4-only offsets (`:283`); an independent INT4/BF16 un-tiling. Stronger than glm53's: every byte is a lossless copy, so the check is equality with the checkpoint. ~100 |
| Kernel-side check | `kcheck.cu` | **re-specialise** | the format tag check (`kcheck.cu:135`), scale checks, `--gpu` cases for INT4 and BF16; `rel_err` in (0, 0.05) (`:100`) becomes 0 for lossless copies. ~40 |
| Tests | `tests/expert_fill_test.cu`, `rank_test.cpp`, `s3x2_layer_test.py` | fill test **re-specialise** (~40), rank test **unchanged**, S3 test **dropped** | |

### Routed experts (`glm53/engine/experts`)

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Shapes, `kE`, top-8, scale 2.5 | `experts.h:16-23` | **constants** | `kE` 168 (§ 2.1) |
| Routing (`route_warp`, `route`, the fused routing in `k_up`) | `formats.cuh:244-282`, `route.cu` | **re-specialise** (small) | eight logits a lane assumes 256: § 2.1 |
| Decode `k_up`/`k_down`: union, task queue, PDL, posted partials, 16 rows | `decode.cu` | **re-specialise** | union bitmap 8 → 6 words; INT4 `dec_group` branch; a BF16 format family; a BF16 shared expert (§ 2) |
| Prefill grouped GEMMs (`k_pup`, `k_pup_ws`, `k_pdown*`, bucketing, combine) | `prefill.cu` | **re-specialise** | `k_scan`'s launch (`:1709`); INT4 `dec_group`; BF16 blocks copied straight to the B stage; the FP4-activation path (`F4Dec`, `:1371-1398`) refuses INT4 and BF16 |
| Formats: NVFP4 (`q4_widen16`), S3 tables, `load_blk`, `dec_group`, `fmt_k` | `formats.cuh:44-73`, `285-313`, `layout.h` | NVFP4 **unchanged** (kept for the MTP drafter, § 5), S3 **dropped**, INT4 and BF16 **new** | § 2.2–2.3 |
| Tests and benches | `tests/test_experts.cu`, `testdata.h`, `bench_experts.cu`, `test_prefill.inc`, `bench_prefill.inc` | **re-specialise** | most follow `kE`; new INT4/BF16 pattern makers and `k_deq` against the host's `bf16((q − z) · s)`. ~300–450 |

### Attention side (`glm53/engine/attn`)

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Model constants | `attn.h` | **unchanged** | the same dimensions |
| q/kv prep, RoPE, cache write, FP8 index keys option | `prep.cu`, `attn_common.cuh` | **unchanged** | |
| Indexer (scores, top-2048 select, merge) | `indexer.cu` | **unchanged** | its weights are inputs computed by the GEMVs |
| Sparse MLA attention, split-K, combine, prefill attention | `sparse_attn.cu` | **unchanged** | |
| GEMV, FP8 and BF16, 1–16 rows, split-K, grouped, SwiGLU | `gemv.cu` | BF16 path **unchanged**; **new** INT4 path; GemvIw **re-specialise** | § 3 |
| GEMM, FP8 tiles and `gemm_bf16` | `gemm.cu` | `gemm_bf16` **re-specialise** (SwiGLU, `gemm.cu:349`); FP8 `gemm` unused | § 3 |
| Packer, reference generator, tests, bench | `tools/pack.py`, `tests/gen_ref.py`, `tests/attn_tests.cu`, `tests/bench_attn.cu` | **re-specialise** | BF16 recuts and the INT4 tile in `pack.py`; INT4/BF16 cases. ~250–400 |

### Model runner (`glm53/engine/model`)

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Layer roles, stop ids, vocabulary | `config.h` | **unchanged** | the same layers, MTP index, vocabulary and stop ids |
| Weight structs, binding | `impl.h:40-61`, `bind.cu:110-178`, `218-292` | **re-specialise** | `Fp8W` → a tagged projection (format, values, scales, zero points); a per-layer format table; § 5. ~150–230 |
| The step: `layer()`, graphs, PDL, l2pf windows, `step_bytes` | `model.cu` | **re-specialise** | the projection calls by format, q_b and `wq_b` apart on INT4 layers, the FP8 byte accounting; § 5. ~80–120 |
| Prefill runner | `prefill.cu` | **re-specialise** | `pf_proj` hard-codes FP8 (`prefill.cu:100-112`); § 5. ~40–70 |
| MTP drafter | `mtp.cu` | **constants** | `eh_proj` BF16 (`mtp.cu:24-25`). ~10 |
| o_proj posting variant | `oproj.cu` | **dropped** | FP8-only (`oproj.cu:14-60`), off by default (`kTuneOproj`, `model.h:108`, `117`) |
| Sampler, ops, L2 prefetch kernel, spec state | `sample.cu`, `ops.cu/.h`, `l2pf.cu`, `spec.h` | **unchanged** | `spec.h:14-17`'s cost model (`c1` 44.5, `drow` 15.5 ms) is re-measured |
| Bench and golden tools | `tools/glm_run.cu`, `tools/compare_golden.py`, `tools/step_profile.py`, `run.sh`, `run4.sh`, `g8_*.sh` | **constants** | `glm_run.cu:165` (`union_of` 256.0, 8/256) and `:889-890` (router dump 256); names and paths. ~20 |

### Server, scripts, front-end, reference, tools

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Rank-0 server, commands, sessions, paged pool, disk tier, kept sessions | `glm53/engine/server/{main,serve,cmd,disk}.cpp`, `keepfile.h`, `protocol.h` | **unchanged** | the pool sizes itself from free memory (`main.cpp:264-282`); the kept sessions' run id hashes the weights' path, size and mtime (`main.cpp:100-115`), so glm53's kept sessions are refused, as they must be |
| Service and launch scripts | `service.sh`, `ranks.sh`, `ring.sh`, `ringlock.sh`, `run4.sh`, `install.sh` | **constants** | names (`GLM53_*` → `ALTAR1_*`, `service.sh:26-35`), the binary, port and paths; `MIN_AVAIL_GIB` (20, `service.sh:34`) can stay. ~40–60 |
| Fleet file | `glm53/fleet/homelab.conf` | **constants** | the weight paths (`:26-78`); the ring lock stays shared |
| glmfront | `glm53/frontend` | **constants** + one decision | § 6 (a port is under way in `altar1-4xgx10-switchless/frontend`) |
| CPU reference | `glm53/ref/glmref.py` and helpers | **re-specialise** | § 7. ~80–120 |
| Quantiser study, format microbenchmarks | `glm53/quant/*`, `glm53/micro/*` | quant **dropped**; micro **re-specialise** | an INT4-g32 mode in `micro/gemv.cu`. ~150–250 |
| Tools | `glm53/tools/tally.py`, `provenance.py`, `nvfp4_entropy.py` | tally **re-specialised** (done: `tools/tally.py`), the others **dropped** | |

## 2. The expert formats

### 2.1 Every place that assumes 256 routed experts

glm53 is tensor-parallel, not expert-parallel: every rank holds every expert, cut to its 512 of the 2048
intermediate channels (`experts.h:16-19`, glm53 [experts.md](../../glm53-4xgx10-switchless/docs/experts.md)). So 168
experts redistribute nothing; the count is a compile-time constant and most sites follow it.

| Where | What | At 168 | Size |
|---|---|---|---:|
| `experts/experts.h:19` | `constexpr int kE = 256` | the constant; `kMaxUnion` (128) and the workspaces do not depend on it | 1 |
| `experts/formats.cuh:244-282` `route_warp` | warp top-8: lane owns experts `lane + 32j`, j < 8; loads `logits_row + lane + 32*j` and `bias + lane + 32*j` unbounded | **out-of-bounds read** at 168 unless changed: j < 6 (192 slots) with `e >= kE` → key −∞, s 0; the shuffle tree and the lowest-index tie order are unchanged. Used by `gx::route` (`route.cu:17`) and by `k_up`/`k_down`'s fused routing (`kTuneRoute`, on by default) | ~10 |
| `experts/decode.cu:53-54`, `64`, `74`, `78`, `85` | `Union::mask[8]`, `pre[8]`: a 256-bit bitmap; clear, `atomicOr(mask[e >> 5])`, popcount scan over 8 words | `(kE + 31) / 32` = 6 words | ~6 |
| `experts/prefill.cu:107-125` `k_scan`, launched `k_scan<<<1, 256>>>` at `:1709` | a thread an expert, Hillis–Steele over `sc[kE]`, `st[kE]` | **out-of-bounds** at 168: threads 168–255 read `cnt[e]`, `sc[e]`. Launch `kE` threads (the scan loop `o < kE` works for any count) | ~3 |
| `experts/prefill.cu:24`, `59-61`, `98-105`, `1685` | tile bound, workspace, `k_count`'s `hist[kE]` | follow `kE` | 0 |
| `experts/route.cu:2-3` | comment "n_group = topk_group = 1 keeps all 256" | text; altar-1's `n_group` and `topk_group` are 1 | 1 |
| `model/model.cu:77`, `209`, `325-327`, `683`; `model/prefill.cu:77`, `294-305` | router logits buffers, the router GEMV/GEMM `n = kE`, the l2pf window, `step_bytes` | follow `kE`; 168 % 8 = 0 passes `gemv` (`attn/gemv.cu:287`) and `gemm` (`attn/gemm.cu:316`); `gemm_bf16`'s 64-row tiles leave a partial tile at 168, and its B loads are bounded (`bok`, `gemm.cu:91-92`) | 0 |
| `model/bind.cu:117-135`, `177-178`, `251-252`, `273-283` | synthetic pools of `kE`, router `kE × kDim`, bias `kE × 4`, the pool loop and the descriptor table `kE × 3` | follow `kE` | 0 |
| `model/impl.h:50` | comment `[256][6144]` | text | 1 |
| `model/tools/glm_run.cu:165`, `889-890` | `union_of()` with 256.0 and 8/256; the mtpgold router dump of 256 | constants | ~4 |
| `weights/glw.py:26`, `285-286`, `349-367`, `413` | `N_EXPERTS`; router and bias shapes; reserved units and their slice text; the coverage regex | the constant; the regex is § 4's | 1 |
| `weights/experts_fill.py:39`, `378`, `382`; `verify.py:249`, `277`; `tests/s3x2_layer_test.py:25` | `E = 256`, defaults naming expert 255 | constants (255 does not exist) | ~6 |
| `experts/tests/*` | `rnd_logits [M][kE]`, `route_ref float s[kE]`, `% kE` | follow `kE` | 0 |
| `ref/glmref.py:60`, `614-615`, `431`, `~1180`; `ref/qprep.py:80`; `ref/summarize.py:22`, `105-109` | `NE = 256`, the config assert, expert `.255.`, `range(256)`, `skew(ne=256)`, "top 64" | § 7 | ~15 |

There is no 8-bit expert id with a 0xFF sentinel, no 256-wide histogram besides `k_count`/`k_scan`, and no graph
keyed on the expert count (graphs are keyed by tune bits, passes, sessions and rows, `impl.h`). **Total ~45 lines**,
two of which are out-of-bounds fixes that changing the constant alone would miss. A binary serving both 256 and 168
would be structural (shared-memory arrays sized by `kE`): not proposed, since altar1 is its own kit.

### 2.2 Every place the expert format is chosen

| Where | What |
|---|---|
| `experts/experts.h:31-42` | `enum Fmt { kNVFP4, kS3 }`, `kBlkK` 512, `kQ4Blk` 2304, `kS3Blk` 1792; `mat_bytes()` sizes **any non-NVFP4 format as S3**, silently |
| `experts/experts.h:47-55` | `MatDesc` (offset, `s2`, `fmt`, S3 tables): enough for INT4 and BF16 as is (`s2` = 1) |
| `experts/experts.h:54-63` | `SharedW`: the shared expert is FP8 128×128 in the T128 tile, fused into `k_up`/`k_down` |
| `experts/formats.cuh:285-313` | `Raw` (`uint4 v[4]; uint2 s`, 72 B a lane), `load_blk`, `dec_group`, `fmt_k` (2^30 for NVFP4) |
| `experts/decode.cu:205`, `213`, `233`, `253-254`, `318`, `397`, `402` | block strides by ternary, `s2 · fmt_k`, `dec_group` calls |
| `experts/prefill.cu:265-268`, `294`, `448`, `490`, `517`, `604`, `627`, `646`, `726`, `739`, `859-883`, `1018-1055`, `1087`, `1212-1231`, `1371-1398`, `1455`, `1536`, `1593` | the same ternaries and `dec_group` calls in `k_pup`, `k_pup_ws`, `k_pup_w2`, `k_pdown_*`; `F4Dec` (the FP4-activation path); `k_combine`'s `fmt_k` |
| `experts/layout.h:24-185` | canonical readers, `pack_nvfp4`, `pack_s3`, `read_q4`, `read_s3`, `make_desc*`, `s3_desc_check`, `scale_bytes_check` |
| `weights/glw.h:99-102`, `glw.py:36`, `49`, `334-346` | `ExpertFmt` (a nibble a matrix, room for 16), `mat_bytes` (unknown → S3 again), `expert_entry_bytes` |
| `weights/kcheck.cu:131-176` | `expect(f == kNVFP4 || f == kS3)`, sizes, S3 tables |
| `weights/experts_fill.py:39-151`, `320`, `368-370`, `401-402`, `443-450`; `s3x2_pack.py` | the NVFP4 tensor names, packers, recipes, read-back |
| `weights/verify.py:252-304`, `357` | NVFP4 un-tiling with hard-coded offsets; other formats skipped |
| `model/bind.cu:274-283` | a layer's expert entries must be equal-sized and back to back (so a layer is single-format, which suits altar-1: INT4 on 4–76, BF16 on 3, 77, 78) |
| `model/bind.cu:281` | `exp_bytes` is one number for the whole model, used by `step_bytes`: wrong once layers differ |

**First, mechanically** (~40 lines): replace the ~25 `fmt == kNVFP4 ? kQ4Blk : kS3Blk` and `0x1p30f` ternaries
by `blk_bytes(fmt)` and `fmt_k(fmt)`, and make every unknown tag die. S3 is then deleted from the copy.

### 2.3 INT4-g32 asymmetric

**The block** (8 rows × 512 k, the existing pool unit):
- values: bytes [0, 2048) exactly NVFP4's (lane l = 4g + t, byte 512jj + 16l + 8q + i/2). compressed-tensors packs
  k = 8j + i at bits 4i of word j, low nibble first, so the checkpoint's `weight_packed` read as bytes **is**
  NVFP4's `[N][K/2]` nibble order: `pack_nvfp4`'s value path re-tiles it unchanged;
- scales: BF16 at 2048 + 32g + 16(t >> 1) + 2(2jj + q), 256 B (a lane's 16 consecutive k lie in one group of 32;
  lanes t = 0, 1 share it, a broadcast);
- zero points: nibbles at 2304 + 8g + 4(t >> 1), 64 B;
- **2,368 B a block**, 0.578125 B a weight (2.8 % over NVFP4), a matrix 444 × 4 KiB.

**The dequantisation** (per pair of values): `0x4300 | nibble` is the BF16 128 + u exactly (ulp 1 in [128, 256));
`__hsub2` against 128 + u_z gives q − z ∈ [−15, 15] exactly (the +8 storage offsets cancel,
[quantisation § 6](quantisation.md#6-checks-made-on-the-data)); `__hmul2` by the BF16 scale rounds the exact
product once to nearest-even. That is **bit-equal to compressed-tensors' `dequantize`** (`(x_q − zp) * scale` in
BF16), and the B fragments feed the existing `mma.sync m16n8k16 bf16 → f32`. About 8 `prmt`/`lop3` + 8 `hsub2` +
8 `hmul2` per 16 values, NVFP4's cost. The product is *not* always exact in BF16 (up to 12 significant bits), so
`formats.cuh:44`'s "every use … exact" comment becomes "bit-equal to the dequantised BF16 weight", and a GPU test
compares `__hmul2` with torch's BF16 multiply, subnormals included.

**What it reuses**: every kernel, tile, pipeline, union and PDL chain of decode and prefill; it adds a `load_blk`
and `dec_group` branch and a `Raw` 12 B wider (measure `k_up`'s registers: it holds 4 `Raw`s). Expected at NVFP4's
93–100 % of bandwidth. **~350–400 lines** with `layout.h`'s `pack_int4`/`read_int4`, tests and bench.

### 2.4 BF16 experts (layers 3, 77; the MTP layer 78)

The DeepSeek engine has **no BF16 expert path** (its routed experts are MXFP4 only, `ds41f/engine/kernels/moe.cu`,
`prefill.cu`), and neither engine has any integer-weight or zero-point kernel. ds41f's BF16 kernels (`gemv16.cu`,
`gemm_bf16_p`) read its tiled L2 layout and serve dense projections, not a routed union; glm53's own BF16 GEMV is
the better donor (§ 3). So BF16 experts are new, inside glm53's expert kernels:

- **Decode is structural.** A BF16 8 × 512 block is 8,192 B, 256 B a lane against `Raw`'s 64: a runtime branch in
  the shared kernels would cost every format its registers. Since a layer is single-format, make the format family
  a template parameter chosen by the host per layer at graph capture (two instantiations of `k_up`/`k_down`), the
  BF16 one loading 16 B × 4 a lane per jj with a shallower prefetch; or define the BF16 block as 8 × 128 k (2,048 B,
  `Raw`'s size) with the family's own strides.
- **Prefill is easy**: store BF16 already in the B-fragment order (`store_b`'s) and `cp.async` it into the B stage;
  the producer warps of `k_pup_ws` copy instead of decode.
- **Cost**: 18.9 MB an expert a rank against 5.5 for INT4: 0.30 GB a token for the two layers at one row.
- **~300–450 lines**, plus tuning.

### 2.5 The shared expert in BF16

glm53 fuses an FP8 T128 shared expert into `k_up`/`k_down` (`experts.h:54-63`; the decode kernels preload it before
their PDL wait) and runs it in prefill as a dense FP8 GEMM; `bind.cu:253-255` dies without the T128 tiles. altar-1's
shared experts are BF16 (1.42 GB a token a rank, the largest BF16 read):
- **First** (~40 lines in `model.cu`): run it through attn's BF16 GEMV, gate/up interleaved with `kSwiGLU`, down in
  `kF32` into the shared partial `ys` that the posted all-reduce already takes apart. Exact; loses the preload
  overlap (~10–20 µs a layer, *estimate*).
- **Then** (~150–250 lines): a BF16 `SharedW` in the expert kernels, the same lane map as the BF16 expert family.

## 3. Attention side

glm53's projections, per layer (`model/bind.cu:235-248`; [glm53 attention.md](../../glm53-4xgx10-switchless/docs/attention.md)):

| Projection | glm53 | altar-1, layers 4–76 | altar-1, layers 0–3, 77, 78 | Kernel for altar-1 |
|---|---|---|---|---|
| x-projection = q_a [512r, +512) + kv_a latent [128r, +128) and rope [512 + 16r, +16) + indexer `wk` [32r, +32), one recut tensor | FP8 T64, 16 × 128 scales | **BF16** | **BF16** | `gemv(false)`, natural rows; the same recut and all-gather geometry (688 / 656 rows) |
| q_b [4096r, +4096) | FP8 128 × 128 | **INT4** | BF16 | new INT4 GEMV; BF16 GEMV |
| W_UK^T (kv_b's nope rows per head, transposed to [512][192]) | FP8 128 × 64 | **INT4, groups transposed** | BF16 | BF16 (dequantised exactly by the converter) first; see below |
| W_UV (kv_b's value rows per head, [256][512]) | FP8 64 × 128 | **INT4** (groups along K) | BF16 | new INT4 GEMV, grouped |
| o_proj columns [4096r, +4096) | FP8 128 × 128 | **INT4** | BF16 | new INT4 GEMV, f32 out |
| indexer `wq_b` [1024r, +1024) | FP8 128 × 128 | BF16 | BF16 | `gemv(false)` |
| indexer `weights_proj` [8r, +8) | BF16 | BF16 | BF16 | unchanged |
| dense gate/up interleaved, down (layers 0–2) | FP8 | — | **BF16** | `gemv(false)` with `kSwiGLU` (decode); `gemm_bf16` lacks SwiGLU (prefill) |
| router | BF16 | BF16 [168] | BF16 [168] | unchanged |
| head | BF16 | BF16 | | unchanged |
| MTP `eh_proj` [1536r, +1536) | FP8 (our RTN) | | **BF16** | `gemv(false)` |

**What exists in BF16** (`attn/gemv.cu:82-223`, `278-327`; `attn/gemm.cu:343-361`):
- `gemv(false, …)`: row-major BF16 `[N][K]`, N a multiple of 8, K of 64, 1–16 rows, split-K 1/2/4, up to 4 segments,
  grouped mode (`group_rows`/`x_group`), outputs BF16, F32 and **SwiGLU** (`gemv.cu:321-325`). Measured on the head
  (38,720 × 6,144) at 95.7 % of 235 GB/s ([glm53 attention.md](../../glm53-4xgx10-switchless/docs/attention.md));
  smaller BF16 shapes (688 × 6144, 1024 × 2048) were never benched. Fixed at 4 loads in flight and 4 warps a CTA
  (`gemv.cu:271`): glm53's FP8 tuning configs do not apply.
- `gemm_bf16`: any rows, one tile (128 × 64, 3 stages, 72 KiB), BF16 or F32 out, bit-equal to the BF16 GEMV
  (tested on 256 × 6144, 38,720 × 6144, 1,000 rows); **refuses SwiGLU** (`gemm.cu:349`). The router at 8,192 rows
  0.64 ms a layer.
- The DeepSeek engine's `gemv16.cu` (BF16 in its tiled L2 layout, 94–102 %) and `gemm_bf16_p` add nothing glm53
  lacks, except **`kBF16X`**: BF16 coded losslessly at ~¾ of the bytes, bit-identical (`ds41f/engine/kernels/gemv16.cu:12-15`,
  `51-91`, `566-660`), the one exact byte lever for altar-1's 3.8 GB of BF16 reads ([quantisation § 4](quantisation.md#4-decode-bytes-a-token-and-the-floor)).

**What is missing:**
1. **The INT4-g32 GEMV** (~250–350 lines). glm53's FP8 path cannot be bent to it: each 64-k chunk's four MMAs mix
   both 32-k groups and the FP8 scale is applied after the MMA (`gemv.cu:106`, `146-172`). Dequantise per element
   instead, as § 2.3 does: lane (g, t) holds row g, k = 16t … 16t + 15 of a chunk, which lie in one group;
   B = `bf16((q − z) · s)`; the MMAs accumulate straight into the f32 sum, as the BF16 path does. **With glm53's
   BF16 lane map and order, its output equals the BF16 GEMV's on the dequantised matrix bit for bit**, the
   milestone check (§ 9). A tile `[N/8][K/128][32 lanes][32 B]` keeps 16-byte loads (two chunks a load) with scales
   and zero points beside it; grouped mode for W_UV, split-K, F32 out for o_proj.
2. **W_UK^T against INT4.** The absorbed query `q_lat[c] = Σ_d W[d, c] q[d]` reads kv_b's nope rows transposed
   (`glw.py:225-226`): the reduction runs over d while the groups of 32 run along c, the output. A per-element
   dequantisation does not care about the orientation, but the tile does: a lane's 16 consecutive k need 16
   different scales. So: **BF16 first**, dequantised by the converter (bit-equal to the kernel's dequantisation),
   +0.15 GiB a rank, +0.16 GB a token (+0.7 ms); a transposed INT4 GEMV (32-row tiles so a scale serves a group)
   later, ~150 lines, if the 0.7 ms is worth it.
3. **q_b and `wq_b` share one launch with one `fp8` flag** (`model/model.cu:266-270`, `model/prefill.cu:237-241`):
   on INT4 layers they are two launches (18 full layers: one extra launch each) or a per-segment format.
4. **GemvIw** (the index weights in the x-projection's launch) is FP8-only (`gemv.cu:249-250`, `307`) and on by
   default (`kTuneIw`, `model.h:117`; measured +0.0 %): turn it off for BF16, or instantiate it (~30 lines).
5. **Prefill**: `gemm_bf16` SwiGLU (~10–20 lines; the epilogue is generic); INT4 projections as **dequantise to a
   BF16 scratch, then `gemm_bf16`** (~60 lines; o_proj + q_b + W_UV are ~70 MB a layer a rank, ~0.4 ms a layer a
   chunk against ~100 ms of a chunk's compute), bit-equal to the GEMV path by construction; a native INT4 B loader
   (~200 lines) only if the scratch shows up in a profile.
6. **Packer and tests** (`attn/tools/pack.py`, `tests/gen_ref.py:1-112`, `attn_tests.cu:32-194`,
   `bench_attn.cu`): BF16 recuts, the INT4 tile, INT4/BF16 references. ~250–400.

The KV path has no FP8 weight scale anywhere (the only FP8 is the optional index-key cache with its own runtime
scale, `prep.cu:163-173`), so the cache, the indexer and the attention are untouched.

## 4. Weights: container, converter, loader, verifier

**The container** (`glw.h`; `glw.py` mirrors it): `DType` F32, BF16, F8_E4M3, U8, U32 (`glw.h:29-35`); `SliceKind`
replicated, rows, cols, recut, generated, reserved (`:52-59`); `Layout` natural, T64, T128 (both FP8-only),
expert-pool (`:73-88`); `ExpertFmt` NVFP4, S3 (`:99-102`); flags lossy, reserved, scale, interleaved. A BF16 entry
kind exists and serves every BF16 tensor as natural rows. What altar-1 adds:
- **INT4 experts**: no new dtype; U8 expert-pool entries with `ExpertFmt::INT4` and § 2.3's blocks, made from
  `weight_packed`, `weight_scale`, `weight_zero_point` (no raw I32 stored).
- **BF16 experts**: `ExpertFmt::BF16`, an entry of N · K · 2 bytes (512 × 6144 × 2 = 1,536 × 4 KiB).
- **INT4 attention**: a new `Layout` (the GEMV tile of § 3) with the scales and zero points beside the values in one
  entry or as `scale`-flagged siblings (as FP8's scale tables are today).

**The converter** (`convert.py`, `glw.py`'s units, `experts_fill.py`): reads a **local** safetensors checkpoint
(`sources.Checkpoint`: `preadv`, `POSIX_FADV_DONTNEED`, a SHA-256 of every source tensor), writes all requested rank
files in one pass on one box, appends the routed experts into reserved entries (`experts_fill.py`); nothing reads
across boxes, and `fetch.py`'s HTTP range reads were only for zai's dense MLPs. So it runs on rank 3, beside the
download ([quantisation § 5](quantisation.md#5-disk-the-conversion-and-where-the-files-live)). It dies on altar-1 as
written: `sources.DT` has no I32/I64; `attn_unit`'s sources are `q_b_proj.weight` (absent on INT4 layers) and
`Src(fp8=True)` re-quantises; `mlp_unit` fetches the dense MLPs from zai's mirror; `coverage` (`glw.py:413`) does not
know `weight_packed`, `weight_zero_point`, `weight_shape`. Changes:

| Change | Size |
|---|---:|
| `sources.py:12`: I32, I64 | 3 |
| `glw.py` plan: per-layer source format (INT4 on 4–76 for q_b, kv_b, o_proj; BF16 elsewhere), BF16 recuts (x-projection, W_UK, W_UV of BF16 layers, dense gate/up interleaved), W_UK of INT4 layers dequantised to BF16, the INT4 tiles, BF16 shared experts in the natural layout (and later the expert kernels' tile), `N_EXPERTS`, the coverage regex | 100–150 |
| INT4 packer: zero-point unpacking, rank slice (gate/up rows [512r, +512) = zero-point rows [64r, +64); down groups [16r, +16)), blocks, descriptors, a self-test against compressed-tensors' unpacking | ~150 |
| `experts_fill.py`: recipes `int4` (copy from the checkpoint) and `bf16`; for the MTP layer also glm53's `nvfp4` (§ 5) | ~120 |
| `verify.py`: independent INT4/BF16 un-tiling, compared for **equality** with the checkpoint's bytes (packed values, scales, zero points, BF16) | ~100 |
| `kcheck.cu`, `check.cpp`, `loader.cpp`, `glw.h` | ~55 |
| `tests/expert_fill_test.cu` | ~40 |

The manifest (source hashes, tool hashes, commit) carries; its `rel_err` per tensor is 0 except W_UK (exact
dequantisation, a format change, not a loss) and the MTP experts if NVFP4. Level 1 of glm53's exactness standard
becomes simpler and stronger: the rank files are a lossless re-tiling of the hashed checkpoint.

## 5. Runner, step, prefill, server, MTP

- **Weight structs and binding** (`model/impl.h:40-61`, `model/bind.cu:218-292`): `Fp8W` → a tagged projection
  `{fmt, w, s, z}` and a per-layer format table (INT4 attention and experts on 4–76; BF16 attention and experts on
  0–3, 77, 78; BF16 shared experts everywhere). The binding loop runs `l < kCacheLayers` (`bind.cu:227`), so the MTP
  layer binds like a main MoE layer, which stays true. `exp_bytes` per layer instead of one (`bind.cu:281`).
  ~150–230 lines.
- **The step** (`model/model.cu` `layer()`): the projection calls at `246-248` (x-projection), `266-270` (q_b + wq_b,
  § 3), `284` (W_UK), `295` (W_UV), `304` (o_proj), `316-319` (dense, BF16 with SwiGLU), `325-327` (router, already
  BF16); the l2pf windows' sizes assume FP8 at 1 B a weight (`model.cu:184-210`) and `step_bytes` too (`:673-688`):
  both per format. Graphs are re-captured per (tune, passes, sessions, rows) as now. ~80–120 lines.
- **Prefill** (`model/prefill.cu`): `pf_proj` (`:100-112`) calls the FP8 `gemm` or `gemv(true, …)` for every
  projection (`:226`, `241`, `260`, `270`, `273`, `287`, `290`, and the MTP's `:366`, `mtp.cu:25`): a format switch to
  `gemm_bf16`, the INT4 scratch path, or the BF16/INT4 GEMVs below 17 rows. The router (`:294-305`) is already
  BF16. ~40–70 lines.
- **Memory budgets**: nothing is hard-coded to glm53's file size. The pool is min(free device memory, MemAvailable)
  less 4 / 2 GiB (`server/main.cpp:89`, `264-282`), capped by `--kv-gib`; `--min-free-mib` 4,608 in `glm_run`
  (`tools/glm_run.cu:118`, `1428-1456`); prefill index scratch 512 MiB (`model.h:142`); chunks 8,192 by default
  (`main.cpp:88`; G8 had to run at 2,048 beside 101.4 GiB). At ~76.6 GiB a rank the defaults fit again
  ([quantisation § 3](quantisation.md#3-what-is-left-of-121-gib)). The paged store (`Model::cache_pos_bytes`,
  96,640 B a position, `model.cu:583-588`) is unchanged.
- **The MTP drafter** (`model/mtp.cu`): `eh_proj` BF16 (`mtp.cu:24-25`), its attention and shared expert BF16 like
  layers 0–3 and 77. Its 168 experts are BF16 in the checkpoint (2.95 GiB a rank, 0.15 GB a draft pass). glm53
  quantised its MTP experts to NVFP4 with `nvfp4_rtn` (DESIGN decision 6: the drafter's precision changes speed, not
  output, since every draft is verified). For altar-1: **NVFP4 first** (glm53's recipe and kernels unchanged:
  −2.12 GiB a rank, −0.45 ms a pass), BF16 once the BF16 family exists if acceptance measured in the reference says
  it pays (glm53's experiment 5 harness).
- **The server, service and sessions** carry unchanged apart from names (§ 1).

## 6. glmfront

glmfront ports the GLM-5.3 checkpoint's `chat_template.jinja` (`frontend/src/encoding.rs:1-15`, `render()` at
`453-553`). altar-1's template differs from that one (NVIDIA's snapshot `e3b8f9f2`, the copy glmfront was checked
against) in five lines:

| Line | GLM-5.3 | altar-1 | Effect |
|---|---|---|---|
| 65 (`visible_text`) | `{%- elif content is not none -%}` | `{%- else -%}` | a JSON `null` content now renders as the text `None` (undefined still renders nothing). glmfront mirrors GLM-5.3 at `encoding.rs:262` (`None \| Some(Value::Null) => ""`), and `server.rs:275` turns a missing content into `Null`, so under altar-1's template every assistant message with tool calls and no content would render `…</think>None` |
| 180, 184, 206, 214 | `{%- break -%}` in the tool-result `can_sort` check | removed | `can_sort` only goes from true to false, so the text is the same for any input that renders; only which error a malformed block raises first can change. glmfront's breaks are at `encoding.rs:359-379` |

`tokenizer.json` and `tokenizer_config.json` are byte-identical to GLM-5.3's; the stop ids (154820, 154827, 154829)
and the sampleable vocabulary (154,856, `encoding.rs:37-39`) hold. Hard-coded: the tokenizer path (`main.rs:5`,
`session.rs:392`), the model name `glm-5.3` (`main.rs:15`, `99`; `tools/smoke_test.py:117`;
`engine/server/tests/serve_test.py:38`), the socket (`protocol.rs:12`), `TABLE_MAGIC` (`session.rs:92`). **~20 lines
plus renames**; `frontend/tools/gen_encoding_cases.py`'s cases at `:266` and `:281` (null content) change and the
whole set is regenerated against altar-1's template.

**One decision.** Rendering the template on the request's JSON as it stands gives `None` for an explicit `null`
content; vLLM normalises a null content to an empty string before it renders (the note at the port's
`server.rs:263-264`), so what altar-1 sees from its reference server is `""`. The front-end port already under way
in this kit (`altar1-4xgx10-switchless/frontend`, its own [protocol.md](protocol.md)) renders `None` in `encoding.rs`,
as the template does, and maps `null` or missing content to `""` in `server.rs` before rendering, as vLLM does.
That is the right pair; the enccheck cases should cover both layers (the renderer against the jinja with `null`
inputs, the server's mapping against vLLM's), and the choice is written down beside the results.

## 7. The CPU reference (`ref/`)

`glm53/ref/glmref.py` streams the model a layer at a time on the CPU from NVIDIA's NVFP4 checkpoint (`Ckpt`,
`:80-145`, `pread`, no mmap), BF16 activations with documented f32 points, DESIGN decision 8's reduction order. For
altar-1 it reads altar-1's own checkpoint:

| Change | Where | Size |
|---|---|---:|
| `NE = 168`, the config assert, `.255.`, `--sub-experts …,255` | `glmref.py:60`, `614-615`, `431`, `~1180` | ~5 |
| I32, I64 in `ST_DTYPES`; `int4_ct_dequant(packed, zp, scale)` = `bf16((u_w − u_z) · s)` by nibble, checked against compressed-tensors' `unpack_from_int32` and `dequantize` on real tensors | `:66`, new | ~40 |
| `CkptExperts` by the tensors' names (`.weight_packed` → INT4, `.weight` → BF16) instead of `l != MTP` | `:408` | ~10 |
| `LayerW` dequantises q_b, kv_b, o_proj where `.weight_packed` exists | `:237-250` | ~10 |
| dense MLPs BF16 instead of per-tensor FP8 | `dense_mlp` `:381-391`, `:709` | ~5 |
| summaries: 168 experts, "top quarter" 42 | `summarize.py:22`, `105-109`; `qprep.py:80` | ~10 |
| the quantisation tracks (`QFORMATS`, `fp8_block`, `NE_FP8`/`NE_NVFP4`, `:253-297`, `488-540`) | | **dropped** |

Per token the work is unchanged (8 experts a layer); a layer's experts dequantised are 12.7 GB in BF16 (19 GB at
256 in glm53), so the peak is about glm53's: 10.9 GiB in the head stage, 3.7–4.4 in the layers
([glm53 reference § 5](../../glm53-4xgx10-switchless/docs/reference.md)); the full 38k-token pass took 1 h 52 min.
The token sets in `glm53/ref/tokens/` stay valid under altar-1's template (none has a null content) and the
tokenizer is the same, so the golden sets are the same texts with altar-1's outputs. It runs on rank 3 (the full
checkpoint) with glm53's venv copied; glm53's measured caveat holds: a CPU pass beside a ring job slowed rank 2's
kernels 8–16 % and was killed by earlyoom at 3.3 GiB under 101 GiB of weights.

## 8. Tools and tests

- `tools/tally.py`: done (this kit), [quantisation.md](quantisation.md).
- `micro/gemv.cu`: an INT4-g32 mode (values + BF16 scales + nibble zero points, the § 2.3 block) and a BF16-expert
  mode beside q4/s3, at 1, 6, 16 rows on the expert and o_proj shapes. ~150–250.
- Tests that hard-code the formats: `experts/tests/*` (NVFP4/S3 makers, `k_deq`, the real-dump reader: INT4 and BF16
  versions), `attn/tests/gen_ref.py`, `attn_tests.cu` (FP8 q_b, W_UK, W_UV, o_proj: INT4 and BF16 versions),
  `weights/tests/expert_fill_test.cu`. ~600–1,000 together.
- Tests that hold: `comm/tests/*`, `weights/tests/rank_test.cpp`, `server/tests/engine.py` (`STOP_IDS`, vocabulary
  154,856), `spec_test.py` (records its own reference), the front-end's encoding tests except the regenerated cases.
- No GLM-5.3 output hash is hard-coded in a test; the refhashes in glm53's docs are re-pinned on altar-1.
- `glm53/tools/provenance.py`, `nvfp4_entropy.py`, `glm53/quant/*`: dropped.

## 9. Build plan

The user's rules carry from glm53: everything a compile-time constant for this model's shapes, outputs exact (no
output-changing speedups), every experiment indexed with its failures, A/B in both orders, end-to-end bars, GPU work
in short preemptible ring windows borrowed from the DeepSeek service, CPU work at nice 19. Effort is relative to
glm53's own milestones (G0–G7 took about one day of several agent lines).

**Change sizes** (new or changed lines, *estimates*):

| Area | Lines |
|---|---:|
| experts: 168 (§ 2.1), format helpers, INT4, BF16 family, BF16 shared | 45 + 40 + 375 + 300–450 + 40 (then 150–250) |
| attention: INT4 GEMV, W_UK BF16 recut, GemvIw, `gemm_bf16` SwiGLU, INT4 prefill scratch, packer and tests | 600–900 |
| model: structs, binding, step, prefill, MTP, tools | 300–450 |
| weights: container, plan, packer, fill, verify, checks | 550–600 |
| reference | ~80–120 |
| front-end, server, scripts, fleet | ~80–120 |
| micro | 150–250 |
| **total** | **~2.5–3.2k** |

**Milestones.**

| # | Milestone | Done when (exactness) | Profiling target | Effort vs glm53 |
|---|---|---|---|---|
| A0 | **Facts and microbenchmarks.** INT4-g32 and BF16-expert decode GEMV prototypes in `micro/` (1, 6, 16 rows, expert and o_proj shapes); `__hmul2` against torch's BF16 multiply (subnormals); the nibble order and offsets against compressed-tensors' `unpack_from_int32`/`dequantize` on ≥ 3 real tensors (CPU, rank 3 or rank 0's partial copy) | the dequantisation equals compressed-tensors' on 100 % of values | INT4 ≥ 93 % of 235 GB/s at 1–16 rows (NVFP4 reached 95–96 % with 2.8 % fewer bytes); BF16 experts ≥ 93 % | 0.3× G0 (no format question) |
| A1a | **Reference and first golden**: `glmref.py` on altar-1 (§ 7); short and ≥ 4k sequences, per-layer dumps at 0, 2, 3, 4, 6, 76, 77, 78 | layers 0, 3, 4 and 77 agree with an independent stack (transformers' `GlmMoeDsa` with compressed-tensors' decompressed weights, a few layers) to BF16 noise; greedy text reads right | — | 0.5× G1a |
| A1b | **Full golden**: glm53's four token sets, the reference's own order variance (two orders), MTP acceptance teacher-forced with BF16 and NVFP4 MTP experts | argmax agreement with the independent stack within noise | ~2 h a pass on rank 3 | 0.5× G1b |
| A2 | **Converter, verifier, loader**: the altar-1 plan, INT4 and BF16 entries, the four rank files (~76.6 GiB) cut on rank 3 and copied out | every byte un-tiled back equals the checkpoint's (values, scales, zero points, BF16); W_UK equals the reference's dequantisation; cross-rank CRCs of whole tensors; `glw_kcheck --gpu` | conversion time; rank 3's disk peak ≤ ~492 GB | 0.6× G2 |
| A3e | **Expert kernels, one rank**: 168, INT4, BF16 family, BF16 shared (first through the GEMV) | INT4 decode and prefill **bit-equal to the BF16 kernels on the same matrices dequantised** (same lane map and order), both bit-equal to the host emulation's sums as glm53's tests are; routing at 168 equals torch.topk with its tie order | decode ≥ 93 % at 1–16 rows; prefill at glm53's 35–37 % of the BF16 peak | 0.6× G3's expert half |
| A3a | **Attention-side kernels, one rank**: INT4 GEMV (q_b, W_UV, o_proj), BF16 x-projection, W_UK, `wq_b`, dense with SwiGLU, `gemm_bf16` SwiGLU, INT4 prefill scratch | INT4 GEMV bit-equal to the BF16 GEMV on the dequantised matrix; `gemm_bf16` bit-equal to the GEMV as now; every layer type matches golden at BF16 level | each GEMV ≥ 95 % (glm53's FP8 ones: 95–103 %); the 688 × 6144 and 1024 × 2048 BF16 shapes measured | 0.5× G3's attention half |
| A4 | **TP=4 decode**: binding, the step's format dispatch, `step_bytes` and l2pf per format; synthetic pools first (as glm53), then the real files | layers 0–4 match golden within the reference's envelope; outputs bit-stable (graphs, eager, two processes); refhash pinned | **≤ 46 ms at 1 row** (1.27× the 36.4 ms floor, glm53's ratio), ≤ 1.05× at 6 rows | 0.3× G4 (the step machinery and its tuning carry) |
| A5 | **Prefill** with the new formats | chunking changes no bit; prefill state equals the step path's up to DESIGN decision 11's order | ≥ glm53's 1,106 / 1,145 tok/s at 2,600 / 8,192 tokens | 0.3× G5 |
| A6 | **Serving and front-end** (altarfront, under way) | `enccheck` byte-exact against altar-1's template (the null-content decision written); tokenizer byte-exact; pi works against a test port | one 256k BF16 session in the pool; `glm53`'s serve tests pass | 0.3× G6 |
| A7 | **MTP drafts**: NVFP4 MTP experts by glm53's recipe, BF16 shared and attention | tokens identical with and without drafts (greedy and sampled) | acceptance on real text; tok/s better than plain on every prompt class; `spec.h`'s cost model re-fitted | 0.3× G7 |
| A8 | **End to end and ship**: the service, soak, refhashes, benches | engine against the reference within 2× the reference's order variance on all four golden sets (glm53's G8 criterion); refhashes pinned and bit-stable | plain decode through the front-end ≥ ~21 tok/s at one session (glm53: 22–23 at 43.6 ms); exact levers (§ 3's `kBF16X`, the fused BF16 shared expert) measured A/B in both orders | 0.5× G8 |

**Order and parallel lines.**
- **At once, all CPU:** A0's CPU checks and A1a (rank 3, as soon as its download verifies; rank 0's partial copy
  already holds layers in its 32 shards for A0), A2's plan and packer (headers suffice; `convert.py --plan` needs no
  data), A6 (under way).
- **Then two GPU lines**, after A0's ring window: A3e (experts) and A3a (attention), each in short ring jobs on
  synthetic and real-layer inputs. A4 can start on synthetic pools before A2's files exist (glm53 did).
- **Critical path:** the download and A2's cut → A4 on real files; A0 → A3e → A4. A5 and A7 follow A4; A1b runs
  beside A3–A4; A8 last.
- **Not on the path:** the transposed INT4 W_UK, the fused BF16 shared expert, `kBF16X` coding, the native INT4
  prefill loader, BF16 MTP experts: each is an exact lever with a measured gain, after A4.

## 10. Surprises and blockers

1. **Two out-of-bounds reads appear if only `kE` changes**: `route_warp` loads eight logits a lane unbounded
   (`formats.cuh:249-253`), and `k_scan` is launched with 256 threads (`experts/prefill.cu:1709`) over arrays of
   `kE`. Both would read past 168 without a compile error.
2. **An unknown expert format is sized as S3, silently** (`experts.h:40-42`, `glw.py:339-341`): a missed branch
   mis-sizes the pool instead of failing.
3. **W_UK^T's transposition puts INT4's groups along the output** (`glw.py:225-226`): BF16 for that half, or a
   transposed kernel.
4. **FP8 is hard-coded where it is not obvious**: `pf_proj` (`model/prefill.cu:100-112`) for every prefill
   projection; one `fp8` flag for q_b + `wq_b` (`model.cu:266-270`); GemvIw (`gemv.cu:249-250`), on by default;
   `oproj.cu` (off by default); the l2pf window sizes and `step_bytes` (`model.cu:184-210`, `673-688`), which would
   report wrong floors and prefetch wrong lengths without failing.
5. **The shared expert must be FP8 T128** (`bind.cu:253-255` dies otherwise) and is fused into the routed kernels:
   experts are not the only quantised weights in the expert path.
6. **`exp_bytes` is one number for all layers** (`bind.cu:281`): wrong once layers 3 and 77 differ.
7. **The template's null content** renders `None` in altar-1's template (line 65), a behaviour change from GLM-5.3;
   the renderer must reproduce it and the server must map `null` to `""` as vLLM does (§ 6).
8. **The model card understates the INT4 set**: it says attention stays BF16, but q_b, kv_b and o_proj of layers
   4–76 are INT4 in the headers.
9. **Rank 0 already holds 32 of the 39 shards of this revision** (257 GiB on its data disk, all but 28 and 34–39,
   sizes and headers equal to the revision's; [quantisation § 5](quantisation.md#5-disk-the-conversion-and-where-the-files-live)):
   useful for A0, and the space to reclaim once rank 3's copy is verified.
10. **"Exact" needs its definition written down**: `(q − z) · s` is not always representable in BF16; the contract's
    weight is compressed-tensors' BF16-rounded dequantisation, which the kernels reproduce bit for bit. A reference
    keeping the weight in f32 could not be matched by BF16 MMAs.

## Sources

- glm53-4xgx10-switchless at `962fdcf`: `README.md`, `DESIGN.md`, `docs/{reuse,layout,experts,attention,engine,serving,quantisation,reference}.md`,
  and the code cited above.
- ds41f-4xgx10-switchless at the same commit: `engine/kernels/{common.cuh,gemv16.cu,gemv_fp8.cu,gemv_fp4.cu,moe.cu,prefill.cu}`,
  `docs/layout.md`.
- AikidoSec/altar-1 at `5d591cd98958e4e7e429517887e80ca3bb7ce2c4`: `config.json` (`quantization_config`),
  `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`,
  `reap_prune_plan.json` (`keep` 168, criterion `massmax168`, the kept original ids a layer, the MTP layer ranked by
  weight magnitude), the 39 shard headers by HTTP range request, and two tensors of rank 0's partial copy.
