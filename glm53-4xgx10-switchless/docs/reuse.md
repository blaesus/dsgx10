# What carries over from the DeepSeek engine

How much of [ds41f-4xgx10-switchless](../../ds41f-4xgx10-switchless/) (the DeepSeek-V4.1-Flash engine on
the same four GB10 boxes) carries over to GLM-5.3, what has to be re-specialised and how, what is new,
and the build plan that follows. Written 2026-10-13 from the DeepSeek engine's code and documents at
master `fd663e6`, the reference code (transformers 5.15.0, `models/glm_moe_dsa/modeling_glm_moe_dsa.py`)
and the headers of `nvidia/GLM-5.3-NVFP4` at revision `e3b8f9f2`. Nothing was built or run for it.
The model's exact math is [model.md](model.md)'s job, and the memory budget and formats are
[quantisation.md](quantisation.md)'s. This page covers only what decides reuse.

Paths below are relative to the repository root. `ds41f/` stands for `ds41f-4xgx10-switchless/`.

## In short

- **About half of the code carries over unchanged or with constants changed:** the ring and its
  collectives, the weight-file container, loader and verifier, the CUDA-graph and PDL step machinery,
  sampling, the speculative verify loop and its gate, batching (B1–B4), the session store and its disk
  tier, the server, the protocol, the scripts, the fleet file, the tokenizer's BPE core and the bench
  tools. By lines of code that is roughly 55 % of `ds41f/engine` and `ds41f/frontend`.
- **The kernels are the part that does not carry over.** Little of their arithmetic is reusable
  as written. Only their structure is: tiles, split-K, union-of-experts MoE, PDL chains, fixed-order
  reductions. Three things change underneath them:
  - **The formats.** NVFP4 experts with E4M3 scales per 16 do not fit the
    `kind::mxf8f6f4 … ue8m0` MMA every DeepSeek GEMV uses. The attention, shared experts and head
    are BF16, or an FP8 of our own.
  - **The attention.** It is MLA with absorbed latents, not MQA with sliding windows and compressors.
  - **The indexer.** Top-2048 over every position, with 21 full layers, not top-512 over compressed
    entries.
- **Dropped:** mHC, the compressor, the sliding window, CED and its bounded replay, Engram (with its
  NVMe store), DSpark's parallel blocks, Markov head and confidence head, window snapshots and their
  thinning, the compressor carries.
- **New:**
  - MLA absorption (the per-head `kv_b_proj` halves as grouped GEMVs);
  - a top-2048 indexer over every position, with a LayerNorm key and BF16/FP32 scores;
  - the chained MTP drafter;
  - an NVFP4 path, MMA or widened to BF16;
  - possibly a sub-4-bit expert format of our own;
  - the GLM chat template and tool-call parser.
- **The decode floor is ~2.2–3.6× DeepSeek's per token.** Per rank a token reads 7.2–11.8 GB, against
  3.26 GB for DeepSeek ([Memory layout](#5-memory-layout-per-rank)). Two things make speculation pay
  less than DSpark did:
  - each extra verified row brings ~0.4 GB per rank of new experts;
  - the MTP drafter is chained, one layer and the head per draft.

## 1. Component table

Verdicts: **unchanged** (copy it, rename at most), **constants** (copy it, and change the dimensions in
one place), **re-specialise** (the same structure and file, with new arithmetic or formats inside),
**new**, **dropped**.

### Communication

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Ring: ibverbs, proxy thread, host-NUMA buffers, device channel | `ds41f/engine/comm/ring.cpp`, `ring.h`, `ring_dev.cuh`, `ring_config.h`, `fleet.cpp` | **unchanged** | nothing: it is model-agnostic. Payloads grow from 10 KB to 12 KB (6144 bf16), still inside the measured 12.3 µs regime ([ring.md](../../ds41f-4xgx10-switchless/docs/ring.md)) |
| Collectives: all-reduce, all-gather, reduce-scatter (by rows, bulk), all-to-all rows, argmax | `ds41f/engine/comm/collectives.cu/.h` | **unchanged** | sizes are arguments. Per decode step about 3 collectives per layer × 78 + the MTP layer + sampling (~240, against DeepSeek's 125 + 40 mix), so ~2.9 ms at 12 µs each: merging them is a milestone of its own (G4) |
| The runner's fused collectives (`allreduce_hc` with hc_post, `allgather_attn_mix`) | `ds41f/engine/model/fused.cu` | **re-specialise** | no mHC: the epilogue becomes residual add + RMSNorm (+ the next GEMV's input format) instead of hc_post/hc_pre. The all-gather carries `q_a` (512 rows a rank), `kv_a` (144), the indexer's `wk` (32) and `weights_proj` (8) slices |
| `comm/run4.sh`, the ring tests | `ds41f/engine/comm/tests/ring_test.cu`, `race_model.py`, `seg_model.py` | **unchanged** | |

### Weights

| Component | Files | Verdict | What changes |
|---|---|---|---|
| File format (header, 512-byte TOC records, 4 KiB alignment, CRC32C, rank check) | `ds41f/engine/weights/dsw.h`, `io.h` | **unchanged** container | new `DType`s: e4m3 scale per 16 (`F8_E4M3` as a scale), an F32 per-tensor scale, and possibly our sub-4-bit code. New `Layout`s: L4 for NVFP4 experts, L5 for the own format (below). Magic and version bump |
| Partition and layout table | `ds41f/engine/weights/plan.cpp` (`kGlobal`, `kBlock`, `Rule`) | **re-specialise** (rewrite the table) | the GLM names (`model.layers.N.self_attn.q_a_proj` …), the splits in [§5](#5-memory-layout-per-rank), experts stacked from per-expert tensors as before (`experts.{e}.gate_proj` / `up_proj` / `down_proj`), the MTP layer as `layers.78`. The rule kinds (Rep, Rows, Cols) suffice; `RowsSeg5` (Engram) goes |
| Safetensors reader, O_DIRECT copy, thread pool | `ds41f/engine/weights/safetensors.cpp/.h`, `io.h` | **unchanged** | the checkpoint is 50 shards of the same container |
| Converter | `ds41f/engine/weights/convert.cpp` | **constants + new tiling calls** | it runs the kernels' own tiling functions on the GPU, so it gains `tile_nvfp4` (and the own format's quantiser, if that is done at conversion rather than offline: the quantisation study's choice). The BF16 → FP8 step for attention and shared experts, if chosen, also runs here, deterministically, with its scales checked as `wo_a`'s are |
| Verifier, host un-tiling | `ds41f/engine/weights/verify.cpp`, `layout.h`, `check_dsw.py`, `toc.py` | **re-specialise** | `layout.h` gains L4/L5 un-tiling written independently from the kernel README's formulas, as now. A re-quantised tensor cannot be compared byte for byte with the checkpoint, so the verifier compares it with a reference quantiser run on the CPU, and records the quantisation error |
| Loader, `dsw_load`, `dsw_kcheck` | `ds41f/engine/weights/loader.cpp`, `load_main.cpp`, `kcheck.cu` | loader **unchanged**; kcheck **re-specialise** | kcheck's reference data comes from GLM layers (one dense, one MoE, one full-indexer layer, the MTP layer) |

### Decode kernels (`ds41f/engine/kernels`)

| Component | Files | Verdict | What changes |
|---|---|---|---|
| FP8 block-scaled GEMV (L1 tiles, `mxf8f6f4` e4m3×e4m3, ue8m0 per 32×32), RMSNorm prologue, multi-segment launches | `gemv_fp8.cu`, `common.cuh`, `act_quant.cu` | **constants**, if non-expert weights become FP8 with ue8m0 per 32×32 blocks (DeepSeek's own FP8 recipe) | K 5120 → 6144 and 2048, the GLM shapes ([§5](#5-memory-layout-per-rank)). The three dense MLPs (FP8 per tensor, static input scale) also fit: every block scale is 127, the two F32 scales go in the epilogue, and activations take one per-tensor scale (the checkpoint's recipe), not `act_quant`'s pow-2 per 32 |
| BF16 GEMV (L2 tiles, `m16n8k16`), with optional norm prologue | `gemv16.cu` | **constants** | it already serves the head and router gate. It becomes the main attention and shared-expert path if those stay BF16, at 2 B per weight. The head: 38,720 rows a rank (154,880 / 4, a multiple of 8). The router: 256 × 6144 |
| Grouped (block-diagonal) GEMV, `kWOA` | `gemv16.cu` (wo_a) | **re-specialise** for MLA | MLA's two per-head products are block-diagonal over 16 heads a rank: absorbing `q_nope` (192) into the latent (512) with W_UK, and expanding the latent output (512) to `v` (256) with W_UV. That is `wo_a`'s pattern (2 groups of [1024][4096] there; 16 groups of [512][192] and [256][512] here). The weights are `kv_b_proj`'s rows, split per head at conversion; W_UK is transposed to [h][512][192] there so both are row-major GEMVs. BF16 directly, not FP8 widened |
| MXFP4 GEMV widened to BF16 (the drafter's head) | `gemv_fp4.cu` | **re-specialise** into the NVFP4 expert GEMV | the best starting point for the expert decode path. Its tile gives each lane 16 consecutive K values per step, which is exactly one NVFP4 group, so one E4M3 scale per lane per step. Swap `widen16_fp4`'s ue8m0 multiply for an e4m3 → bf16 multiply (e2m1 × e4m3 has at most 6 significant bits: exact in bf16), and apply `weight_scale_2` per expert in fp32 |
| Routed MoE: union of the rows' experts, `moe_up` (w1/w3, SwiGLU, routing weight, re-quantised h), `moe_down` (w2, bf16 per expert, fp32 sum in expert-id order, shared expert added), task queue, `_r16` builds, the 1-bit scale code | `moe.cu`, `dsk.h` (`MoeWeights`, `MoeCfg`) | **re-specialise** (the biggest single kernel job) | the structure carries: the union, the PDL overlap of up and down, the bf16-per-expert fp32 sum, 16 rows. Underneath, everything changes: K 5120 → 6144, intermediate a rank 576 → 512, top-6 → top-8 (the union grows to 8 × 16 = 128 at 16 rows; `kMaxUnion`), no SwiGLU clamp (GLM has no `swiglu_limit`), the weights' format (NVFP4 or own, [§2](#2-the-format-questions)), the activation side (bf16 or NVFP4, not e4m3 per 32), the shared expert in BF16 or FP8 (it was FP8 L1). The reference's expert sum differs too: HF adds each expert's weighted output into a **bf16** accumulator in expert order (`index_add_`), where DeepSeek's was fp32 ([model.md](model.md) has the exact order). The scale code (`dsk::scale_code`) does not apply: E4M3 scales are not two neighbouring exponents |
| Norm and router | `norm_router.cu` | **constants** | `sqrt∘softplus` → `sigmoid`; bias for selection only, kept; top-6 → top-8; `n_group` 1 (no group limit); ÷(sum + 1e-20) × 2.5; eps 1e-20 → 1e-5. `tests/probe_router_sum.py` must find torch's summation order of 8 weights again |
| L2 prefetch in the collectives' windows (B′) | `l2pf.cu` | **unchanged** mechanism | the windows and the next layer's addresses are the runner's |
| Decode chain bench, step bench | `tests/bench_chain.cu`, `bench_kernels.cu` | **re-specialise** | GLM shapes; the k=1 / k=4 cases recomputed |

### Prefill kernels

| Component | Files | Verdict | What changes |
|---|---|---|---|
| FP8 GEMM, 128×128 tiles, 3-stage cp.async, ldmatrix, `mxf8f6f4` m16n8k32; SwiGLU + act_quant epilogue | `prefill.cu` (`gemm_fp8_p`, `gemm_fp8_swiglu_p`) | **constants** if non-expert weights are FP8 per 32×32 | as for the GEMVs |
| BF16 GEMM (`gemm_bf16_p`, `gemm_woa_p`) | `prefill.cu` | **constants** | it becomes the main dense path if attention stays BF16, capped at the measured 107 TFLOPS of bf16 `mma.sync`, half of e4m3's 214 |
| Routed experts as grouped GEMMs: bucketing, 64-row tiles, up/down, combine inside the reduce-scatter (`moe_combine_rs`), L2 policies | `prefill.cu` (`moe_prefill`) | **re-specialise** | the same shape changes as decode. The MMA becomes `kind::mxf4nvf4 … scale_vec::4X … ue4m3` (NVFP4 activations) or BF16 with weights widened ([§2](#2-the-format-questions)). Each layer's experts are 1.33 GB a rank at NVFP4 (5.7 ms at 235 GB/s whatever the chunk), so chunks should be ≥ 8192 rows, as DeepSeek's are |
| Sequence-parallel prefill around the residual stream: all-gather in, reduce-scatter by rows out, `alltoall_rows` before wo_b | `ds41f/engine/model/prefill.cu`, `prefill.h` | **re-specialise** | the structure carries. mHC's `hc_boundary_p` becomes residual + RMSNorm + the next input's quantisation; o_proj takes the all-to-all wo_b took (16 heads × 512 latent → its 4096 columns) |
| Prefill attention, indexer, compressor, mHC | `prefill_attn.cu`, `prefill_index.cu`, `prefill_comp.cu`, `prefill_mhc.cu`, `pa_common.cuh` | attention and indexer **re-specialise**, compressor and mHC **dropped** | see [§3](#3-attention-and-indexer). `pa_common.cuh`'s helpers (bf16 MMA, `mxf4nvf4` wrapper, e2m1 → f16 converts) carry |

### Attention side (`ds41f/engine/attn`)

| Component | Files | Verdict | What changes |
|---|---|---|---|
| RoPE tables, `rope_rows` | `rope.cu`, `prep.cu` | **constants** | one table: theta 8e6, default (no YaRN), interleaved pairs like DeepSeek's. The rope dims are the **last** 64 of q's 256 and of the 576-wide cache entry, but the **first** 64 of the indexer's 128. HF casts cos/sin to bf16 before the multiply ([model.md](model.md)) |
| q_norm, kv_write (norm, RoPE, FP8 into a ring) | `prep.cu`, `decode_prep` | **re-specialise** | `q_a_layernorm` (2048), `kv_a_layernorm` (512) on the latent only, RoPE on the 64 `k_rot`, then a write into a per-layer **linear** cache at the position, not a 256-slot ring, in the cache format chosen ([§3](#3-attention-and-indexer)) |
| Sparse attention: online softmax over 64-key blocks, P in bf16, split-K + combine, fused q RoPE and inverse RoPE on the output, sink, window bounds | `sparse_attn.cu`, `attn_common.cuh` | **re-specialise** | MQA over the latent stays (64 heads, one shared KV row). Per key the dot product is 576 wide (512 latent + 64 rope) and V is the first 512. No window, no sink, no inverse RoPE on the output. Scale 1/√256. The selected set is up to 2048 positions (kmax is already a runtime argument, a multiple of 64), so 32 blocks a row in place of 2 + 8 |
| Indexer: q prep, FP4 scores (`mxf4nvf4 … 2X ue8m0`), torch.topk-exact select (16-bit keys, histogram + ordered gather), candidate pool, Reindex, rank split + merge | `indexer.cu` | **re-specialise**; the pool and Reindex are **dropped** | scores over every position, not compressed entries; key from `wk(x)` with a LayerNorm (weight and bias, eps 1e-6), not from a compressor latent; BF16/FP32 numerics, not FP4 (unless the study accepts an FP8/FP4 indexer); 32-bit sort keys (the reference's scores are fp32); top-2048. `index_merge` (4 × candidates in rank order) carries for the position-split scan |
| Compressor, carries | `compressor.cu/.cuh` | **dropped** | |
| mHC | `mhc.cu`, `mhc_dev.cuh` | **dropped** | GLM has a plain pre-norm residual (verified: `GlmMoeDsaDecoderLayer.forward`) |
| Session slot tables in every state struct | `attn.h` (`slot`, strides) | **unchanged** pattern | |
| Tests and generators | `tests/attn_tests.cu`, `fused_tests.cu`, `slot_tests.cu`, `gen_attn.py` | **re-specialise** | generators run the HF reference instead of `model.py` |

### Model runner, speculation, sessions

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Shapes and layer roles | `ds41f/engine/model/config.h` | **rewrite** (70 lines) | 78 layers + MTP, dense 0–2, `indexer_types` (full at 0, 1, 2, 6, 10, …, 74), no Engram/compressor/window roles |
| The step: `k_step_in`, embed, per-layer chain, head, sample; streams `sa`/`sb`; PDL; one graph per row count; StepIn/SampleOut rings; steps queued one ahead | `model.cu`, `ops.cu/.h`, `impl.h`, `state.cu` | **re-specialise** | the machinery carries. The layer body is new: no hc_pre/hc_post, no Engram waits, an absorbed-MLA attention, the full-or-shared indexer. The step's graph capture, PDL and the device-side row count carry as they are |
| Sampling: vocab-split head, per-rank summaries, Gumbel-max, top-k/top-p union, log-probs | `sample.cu` | **constants** | 38,720 rows a rank; three EOS ids |
| Spec loop: SpecState, the switch node over the row count, verdicts in the sampler, `k_spec_next`, the cost-weighted gate, `spec_loop` in `cmd.cpp`, "written past history" bounds | `spec.cu`, `spec.h`, `spec_batch.h`, `ds41f/engine/server/cmd.cpp` | **re-specialise** | the verify half carries, and so does the exactness argument (every emitted token is the one plain decode samples). The draft half is new ([§4](#4-speculative-decoding-with-the-mtp-layer)): `main_hidden` at layers 37–39 → the last layer's output; `accept` → the MTP layer's KV write; `draft` → k chained MTP passes; no confidence head, so the gate needs another survival estimate |
| DSpark module | `ds41f/engine/dspark/*` | **dropped** as a module; its pattern **reused** | `Draft::accept`/`draft` as one PDL graph with device-side counts is the template for `Mtp::accept`/`draft` |
| Batching B1–B4 (time slices, slot tables, batch spec step, 16 rows) | `spec_batch.h`, `serve.cpp`, the kernels' `_r16` | **unchanged** design, **re-specialise** per kernel | every new kernel needs the slot table and the 9–16-row form from the start |
| Session store: pool, 64 KiB blocks, LRU, eviction inside requests, disk tier, kept sessions across stops | `state.cu/.h`, `spill.cpp/.h`, `ds41f/engine/common/keepfile.h` | **constants** (page sizes) | a page holds one position range of every layer's cache and index keys: 51–91 KB a position ([§5](#5-memory-layout-per-rank)), against 890 B. Pages of 1024 positions are 53–94 MB |
| Window snapshots, thinning (`keep_snap`), end snapshots, carries, the restart rule, CED's `dec_lo` rules | `serve.cpp`, `state.cu` | **dropped** | GLM's state at position q is exactly the cache entries below q, so a session restarts exactly at any q ≤ the common prefix, with no snapshot. The "decode-step vs prefill roundings" caveat stays |
| Rank-0 server, commands (`Cmd`, `kCmdPrefill`, `kCmdDecode` …), watchdog, ring borrowing, preemption | `ds41f/engine/server/main.cpp`, `cmd.cpp/.h`, `serve.cpp/.h` | **constants** (minus Engram, CED and DSpark options) | |
| Golden check mode | `ds41f/engine/server/golden.h`, `npy.h` | **unchanged** format reader, **re-specialise** the check points | per-layer residuals at the layer boundaries (no 4 copies) |

### Front end, scripts, data, benches

| Component | Files | Verdict | What changes |
|---|---|---|---|
| Protocol (HELLO, SUBMIT, CANCEL, RELEASE, TOKENS, DONE) | `ds41f/docs/protocol.md`, `ds41f/engine/common/frontend_protocol.h`, `frontend/src/protocol.rs` | **constants** | HELLO's single `eos_id` → GLM has three (154820, 154827, 154829). Send them in `stop_ids`; bump the version |
| dsfront: OpenAI API, SSE with keep-alive, sessions and kept sessions, gate, guard, mock engine | `frontend/src/server.rs`, `session.rs`, `gate.rs`, `guard.rs`, `engine.rs`, `mock.rs`, `main.rs`, `pyjson.rs` | **unchanged** (model name and defaults aside) | |
| BPE tokenizer | `frontend/src/tokenizer/mod.rs` | **unchanged** core | GLM's `tokenizer.json` is byte-level BPE too (154,820 vocab + 36 added tokens, no normaliser) |
| Pre-tokeniser | `frontend/src/tokenizer/pretok.rs`, `unicode_tables.rs` | **rewrite** | GLM uses one cl100k-style split (`(?i:'s|'t|…)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|…`), not DeepSeek's three. The class table needs N apart from L |
| Chat encoding and streaming parser | `frontend/src/encoding.rs` (a port of DeepSeek's `encoding.py`) | **rewrite** | port GLM's `chat_template.jinja`: `[gMASK]<sop>`, `<|system|>`, reasoning effort, tools in `<tools>`, `<tool_call>` with `<arg_key>/<arg_value>`, `clear_thinking`. The `enccheck` method (byte-for-byte against the reference renderer over generated cases) carries |
| `run4.sh`, `ring.sh`, `ringlock.sh`, `fleet.sh`, `service.sh`, the fleet file | `ds41f/engine/server/*.sh`, `ds41f/fleet/homelab.conf` | **unchanged** | drop the Engram keys and options; new binary, weight and service names |
| Golden data tooling | `ds41f/golden/*`, `ds41f/ref/torchref/ref_run.py` | **re-specialise** | the layer-at-a-time runner pattern carries; the model is the HF `GlmMoeDsaForCausalLM` with experts dequantised one layer at a time (19 GB of bf16 a layer fits), plus an MTP reference (HF ignores `layers.78`). Collapse and Engram studies are dropped |
| `refhash.sh` | `ds41f/golden/refhash.sh` | **unchanged** method | new runs and hashes |
| Benches: `bench.py`, `editbench.py`, `abba.sh`, sparkDash, `piturn` | `ds41f/bench/*` | **unchanged** | new baseline files (`baseline-*.json`) from whatever GLM stack serves on these boxes ([scouting.md](scouting.md)) |
| Microbenchmarks | `ds41f/micro/*` | **unchanged**, plus new cases | `mma.cu` gains the NVFP4 forms ([§2](#2-the-format-questions)) |

### How the share was estimated

Line counts (`wc -l`) at `fd663e6`, excluding generated tables:
- unchanged, or constants only: comm (≈2.9k lines with tests), weights (≈2.4k), the server and scripts
  (≈5.5k), sampling and the store (≈1.6k), the front end except the encoding and pre-tokeniser (≈3.2k),
  golden and bench tools;
- re-specialised: kernels (≈9.6k with tests), attention (≈4.5k), the runner (≈4.3k);
- dropped: engram, dspark, mhc, the compressor, CED (≈4.5k).

Counted by lines, about 55 % carries with at most constants changed. Counted by effort, the part
that carries is the part that took the least of the DeepSeek build's time. The kernels and their
tuning took the most.

## 2. The format questions

### What the DeepSeek engine instantiates

- **Every weight GEMV and GEMM** uses `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`
  with **ue8m0 scales per 32**:
  - experts: `.e4m3.e2m1`, activations as A, weights as B (`ds41f/engine/kernels/common.cuh`,
    `mma_e4m3_e2m1`);
  - dense FP8: `.e4m3.e4m3`.

  Activations are quantised to **e4m3 with a power-of-two scale per 32** (`act_quant.cu`,
  `act_scale_byte`), as DeepSeek's reference does. The MXFP4 values are expanded from packed nibbles
  into bits 5:2 of a byte (`e2m1x4_expand`), because `mxf8f6f4` takes e2m1 one per byte.
- **`kind::mxf4nvf4` appears only as `scale_vec::2X … e2m1.e2m1 … ue8m0`**, that is MXFP4 with
  group 32, in the FP4 indexer (`ds41f/engine/attn/indexer.cu`) and the prefill attention helpers
  (`ds41f/engine/kernels/pa_common.cuh`, `mma_fp4`). The NVFP4 form (`scale_vec::4X`, `ue4m3`,
  group 16) is not instantiated anywhere.
- **`micro/mma.cu` checks only the `mxf8f6f4` forms**, bit-exact against a CPU reference, at ~214
  TFLOPS. The `mxf4nvf4` figures in `ds41f/docs/findings.md` (486–512 TFLOPS) are nvfp4bench's, not
  measured here.
- **The decode GEMV for experts** (`moe.cu`) reads the L3 tile: per expert matrix, 16 weight rows ×
  64 K values a step.
  - The 4 value words a lane loads are the B fragments of two n8 halves × two k32 halves.
  - The scales sit beside them in the same tile order: four ue8m0 per lane per step.
  - The token rows are the MMA's A (m16): rows 0–7 in the lower half and, from 9 rows, 8–15 in the
    upper half (`_r16`).
  - Each step of a warp therefore streams 512 B of values + 32 B of scales and runs 4 MMAs.
  - The union of the step's experts is read once whatever the row count.

### NVFP4 on sm_121a: what changes

NVFP4 is E2M1 values with an **E4M3** scale per **16** values, plus an F32 `weight_scale_2` per
tensor. `kind::mxf8f6f4` accepts only ue8m0 per 32, so NVFP4 weights cannot run through DeepSeek's
MMAs without re-quantisation. The choices, with what each costs in kernels:

1. **BF16 activations, weights widened to BF16 in registers** (`m16n8k16`).
   - **Exactness.** e2m1 × e4m3 is exact in bf16, so this computes the dequantised-weight reference
     up to fp32 summation order. `weight_scale_2` multiplies the fp32 accumulator, per expert.
   - **Decode.** It is bandwidth-bound either way. `gemv_fp4.cu` already does this for MXFP4, and its
     lane-per-16-values tile matches NVFP4's groups. Widening costs about 3 integer/bf16 operations
     per 2 values, well inside the ALU budget at 418 G values/s a box.
   - **Prefill.** It runs at bf16's 107 TFLOPS peak (measured), half of what FP8 got and a quarter
     of what `mxf4nvf4` should get. GLM's prefill is ~80 GFLOP of GEMMs per token, ~5× DeepSeek's
     CED prefill. At 60 % of peak over four boxes that is ~3.2k tok/s from the GEMMs alone, before
     attention.
2. **NVFP4 activations (the published ModelOpt recipe), `kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 … e2m1.e2m1.f32.ue4m3`.**
   - **The activation side.** Activations are quantised to e2m1 with dynamic E4M3 scales per 16,
     under the checkpoint's static per-expert `input_scale`. The epilogue multiplies by
     `weight_scale_2 × input_scale`.
   - **Operands.** Both are packed nibbles, so the checkpoint's bytes feed the MMA without expansion.
     That needs checking: the nibble order and the operand fragments for k64.
   - **Prefill.** ~2× the FP8 MMA rate.
   - **Decode.** Same bytes as option 1, plus an activation-quantisation pass per GEMV input. It can
     be fused into the producer's epilogue, as `act_quant` is in DeepSeek's.
   - **Exactness.** It changes outputs relative to bf16 activations. Whether that is "the model" is
     the quantisation study's decision. If it is, decode and prefill must both quantise, so the
     prefill path and the step path keep agreeing (DeepSeek's prefill had "the decode step's
     arithmetic").
   - **Before anything depends on it**, `micro/mma.cu` needs this form bit-exact against a CPU
     reference, and its throughput, as was done for `mxf8f6f4` on 2026-10-11.
3. **FP8 activations × FP4 weights (DeepSeek's `mxf8f6f4 e4m3.e2m1`)** works only if the weights are
   re-quantised to ue8m0 per 32 (MXFP4). That loses NVFP4's finer scales and is lossy against the
   checkpoint. It is listed only because it would reuse `moe.cu` nearly verbatim. Not recommended.
4. **F16 instead of BF16 in option 1** (`cvt.rn.f16x2.e2m1x2` is a hardware convert,
   `pa_common.cuh` `e2m1x2_h2`): it saves a few operations per value but makes activations f16,
   which is not exact for bf16 inputs. Drafter only.

**The dense MLPs (layers 0–2)** are FP8 with one F32 weight scale and a static per-tensor input
scale. Either they run as published, with activations quantised to e4m3 under `input_scale` and
DeepSeek's FP8 GEMV with all block scales at 2^0 and both F32 scales in the epilogue, or the e4m3
weights are widened to bf16 (exact) under bf16 activations. These are 0.17 GB a rank per token.

**The BF16 non-expert weights** (attention, shared experts, head, MTP, 26 GB in all) dominate the
decode bytes if kept: 6.5 + 1.4 + 0.5 GB a rank a token against 3.2 for the experts. Kept BF16, they
run on `gemv16.cu`'s path. If the study allows FP8 for them:
- **FP8 with ue8m0 per 32×32 blocks, DeepSeek's own recipe.** It reuses `gemv_fp8.cu`, the L1
  layout, `act_quant` and `gemm_fp8_p` with only shape changes, and halves those bytes.
- **Per-channel FP8 scales.** These need an epilogue scale per output row. That is new code, but
  small.

### A ~3.5-bit expert format of our own

GB10's MMAs take e2m1 (4-bit), e3m2/e2m3 (6-bit) and e4m3/e5m2 (8-bit). Nothing takes 3 bits, so a
sub-4-bit format is decoded in registers. What it needs:

- **Storage.** It must stay GPU-friendly at 16-value groups. Two candidates:
  - **Bit planes:** a 2-bit plane (4 B per 16 values) and a 1-bit plane (2 B per 16 values), with
    one 8-bit scale per 16 → 3.5 bits a value;
  - **3-bit codes packed** 32 per 12 B, with a scale per 32 → 3.25 bits.

  Bit planes keep every load 16-byte aligned and a lane's group in one place, so they are the
  better fit for `gemv_fp4.cu`'s tile.
- **Decode in registers.** An 8-entry table of bf16 high bytes fits in two registers and
  `__byte_perm`/`prmt` selects from 8 bytes: one `prmt` per 4 values gives the high bytes, then a
  shift/or makes bf16. Then the scale multiply. If the 8 code points are a subset of e2m1's grid, the
  same codes can instead expand to packed e2m1 for `mxf4nvf4` in prefill (3 → 4 bits is a bit
  shuffle).
- **Exactness.** It is a lossy format, so the quantisation study owns quality. The kernel side
  guarantees only that its decode is exact (code × scale exact in bf16) and deterministic.
- **Mixed formats.** If some layers or experts stay NVFP4 and others go 3.5-bit, the expert kernels
  take the format as a template parameter per layer, and the stacked tensors of one layer share one
  format, so a union never mixes them.
- **Bytes.** 3.5 bits against NVFP4's 4.5 (values + scales) is 22 % fewer expert bytes: 2.48 instead
  of 3.18 GB a rank a token at one row, and per extra distinct expert 0.31 instead of 0.40 GB a rank.

## 3. Attention and indexer

| | DeepSeek-V4.1-Flash (ds41f) | GLM-5.3 |
|---|---|---|
| Layers | 40 (20 encoder + 20 decoder under CED) + 3 DSpark blocks | 78 + 1 MTP layer (index 78) |
| Hidden | 5120 | 6144 |
| Query heads | 64 (16 a rank) | 64 (16 a rank) |
| Attention form | MQA: one 512-wide latent per position is K and V; RoPE on its last 64 dims | MLA: q per head 192 nope + 64 rope; K/V from a 512 latent via `kv_b_proj` [64 × (192 + 256), 512]; decoded **absorbed** as MQA over [512 latent ; 64 rope] = 576 for qk, the first 512 as V |
| q LoRA | 1280 | 2048 |
| KV projection | `wkv` 5120 → 512 | `kv_a_proj_with_mqa` 6144 → 576 (512 latent + 64 rope) |
| Output | `wo_a` 8 groups × [1024][4096] + `wo_b` 8192 → 5120 | W_UV per head [256][512], then `o_proj` 16384 → 6144 |
| Sink, window | learned sink per head; SWA window 128 on every layer | neither |
| Compressed KV | ratio 2 (layers 2–19) or 1 (20–39), 4 caches shared by all layers | none: every layer keeps its own latent at every position |
| Cache per position | 890 B for all 40 layers | 576 values × 79 layers: 91 KB bf16, ~52 KB as FP8 latent with bf16 rope (DeepSeek-V3.2's FlashMLA layout: 512 e4m3 + 4 fp32 scales + 64 bf16 = 656 B a layer) |
| Softmax scale | per model.py | 1/√256 (`qk_head_dim`), no YaRN mscale |
| RoPE | theta 1e4 (window layers, DSpark), 1.6e5 + YaRN (compressed); last 64 dims; inverse RoPE on the output | theta 8e6, default, interleaved pairs, on dims 192–255 of q and 512–575 of the cache entry; no inverse RoPE |
| Indexer heads × dim | 32 × 128 | 32 × 128 |
| Indexer query | `wq_b` [4096, 1280] on the q latent | `wq_b` [4096, 2048] on `q_resid` (after `q_a_layernorm`) |
| Indexer key | `wk` [128, 512] on the compressor latent + RMSNorm | `wk` [128, 6144] on the layer input + **LayerNorm** (weight and bias, eps 1e-6) |
| Indexer RoPE | dims 64–127 | dims **0–63** (rope first), interleaved |
| Indexer numerics | FP4 q and k, ue8m0 per 32 | q·k in fp32 from bf16 q, k; head weights `weights_proj(x)` in fp32 × 32^-0.5, scores × 128^-0.5 |
| Scored positions | compressed entries; layers 24–36 only layer 20's 16,384-position candidate pool | every position ≤ the row's |
| Top-k | 512 (+ the 128-position window) | **2048**; below 2048 positions it selects everything, which is plain causal attention |
| Which layers index | full 2, 8, 14, 20; Reindex 24, 28, 32, 36; others reuse their source | `indexer_types`: full at 0, 1, 2, 6, 10, …, 74 (21 layers); each of the other 57 reuses the latest full layer's top-k; the MTP layer has its own indexer (the README's "22" counts it) |

**`index_topk_freq` and `index_skip_topk_offset`.** `GlmMoeDsaConfig` derives the pattern as
`full if max(i − offset + 1, 0) % freq == 0`. With freq 4 and offset 3: layers 0–2 are full, then
every fourth layer from 6. The checkpoint's `config.json` spells the list out, and it matches. A
shared layer takes `prev_topk_indices`, the top-k of the latest full layer at the same query
positions. It attends to **its own** cache at those positions; only the choice of positions is shared.
`index_share_for_mtp_iteration: true` means that chained MTP passes reuse the first pass's indices,
as SGLang does. HF does not implement the MTP layer, so this comes from the serving stacks, not the
reference.

**What in `ds41f/engine/attn` is parameterised and what is hard-coded:**
- **Hard-coded.** Every dimension is an `inline constexpr` in `attn.h`: `kDim`, `kHead` 512,
  `kRope`, `kHeads` 16, `kQLora`, `kWindow`, `kRing`, `kIdxHeads`, `kIdxDim`, `kHc`. The layer roles
  are `constexpr` functions in `ds41f/engine/model/config.h` (`kv_source`, `index_source`,
  `idx_slot`). The cache formats are structs with fixed element formats: `Window` FP8 + ue8m0/32,
  `Compressed` e2m1 + e4m3/16, `IndexKeys` e2m1 in transposed words. The RoPE table builder knows
  exactly two variants. Sort keys are 16-bit (bf16 scores).
- **Parameterised.** `kmax`, the top-k (a runtime argument of `index_topk`, `index_merge`,
  `sparse_attn`, any multiple of 64); `IndexScan` (dense part/nparts or candidate blocks); rows per
  call up to 16; the session slot tables and strides; `splits` for split-K; `max_scan`, the scan
  length, read from device memory so one graph serves every context.

**Consequences for GLM's kernels:**
- **`sparse_attn`.** The online-softmax core, split-K and combine carry. The dot product is 576 wide,
  V is 512, and the entry format is the chosen cache format. There are 32 blocks of 64 keys a row,
  with no window blocks. Prefill and decode share it (absorbed MQA in both) so their numerics agree;
  absorbed MQA costs 64 × 2048 × 1088 × 2 = 285 MFLOP a token a layer at full top-k, ~22 GFLOP a
  token over 78 layers, which is a real share of prefill time.
- **Absorption.** `q_nope · W_UK` and `o_lat · W_UV` are two grouped GEMVs per layer per row: 7.3 MB
  of `kv_b_proj` a rank at bf16. HF does not absorb: it expands K/V per position. Absorbing is exact
  in real arithmetic but sums in another order, so it agrees with the reference to bf16 numerics,
  not bits ([model.md](model.md)).
- **Indexer.**
  - Rope on the first 64 dims.
  - A LayerNorm key on the layer input.
  - fp32 scores, so a 32-bit order key: the histogram select needs a second radix pass or a 32-bit
    first pass.
  - Ties go to the lowest position, as `index_topk` already does. torch.topk's tie order on CUDA
    should be checked against that.
  - Rows with position < 2048 skip scoring entirely. Their set is every position, exactly, which
    makes contexts under 2k free of indexer cost.
  - The query's `wq_b` (16.8 MB bf16 a layer) should be **split by heads** across ranks and its
    output all-gathered (8 KB a row), not replicated as DeepSeek's was. Replicated, it reads 353 MB a
    rank per token over 21 layers (~1.5 ms); split, it costs one extra ~10 µs all-gather per full
    layer.
- **Position-split scans.** At decode, every rank either scans all positions or a quarter, then runs
  `index_merge` over 4 × 2048 candidates. Keys cost 128–256 B a position a layer. At 128k context,
  scanning all is 355 MB a rank a token (~1.5 ms); a quarter is 0.4 ms plus 21 merges' all-gathers
  of 64 KB a row. The crossover is around 32k; DeepSeek's engine has both paths (`IndexScan`), and
  this one should too.
- **The cache.**
  - Linear per layer, append-only, indexed by position: no ring, no snapshots.
  - **Replicated** on every rank, as DeepSeek's compressed caches were (all 16 local heads read the
    same latent): 52 KB (FP8) to 91 KB (bf16) a position a rank, so a 128k session is 6.7–11.6 GB
    on **each** rank.
  - **Position-split** (each rank holds a quarter of the positions of every layer): 4× the capacity.
    Each rank then computes partial attention for **all 64 heads** over its positions of the
    selected set. That needs the 64 heads' absorbed queries all-gathered (73 KB a row) and the
    partial outputs with their log-sum-exps reduce-scattered to the heads' owners (128 KB a row): two
    extra collectives per layer, ~2–4 ms a step.
  - The memory study should decide between the two, since it decides how many long sessions fit.
    Either way the indexer's keys follow the same split.

## 4. Speculative decoding with the MTP layer

**What the MTP layer is** (checkpoint `layers.78.*`, all BF16):
- `enorm`, `hnorm`, `eh_proj` [6144, 12288];
- a full decoder layer: MLA attention with its own indexer, its own router and 256 BF16 routed
  experts (19.3 GB), and a shared expert;
- `shared_head.norm`, with the main `embed_tokens` and `lm_head`.

As in DeepSeek-V3's MTP, the input at position i is `eh_proj([enorm(embed(t_{i+1})); hnorm(h_i)])`,
where h_i is the main model's last hidden state (before `model.norm`). The output, through
`shared_head.norm` and the head, predicts t_{i+2}. Chaining feeds the MTP layer's own output hidden
state back in with the drafted token.

**What carries from DSpark's loop** (`ds41f/engine/model/spec.cu`, `cmd.cpp`'s `spec_loop`):
- The spec step graph: a switch over the row count, read from device memory; verdicts inside the
  sampler; `k_spec_next` choosing the next step's rows; steps queued one ahead.
- The exactness argument: greedy accepts the longest matching prefix, sampled is speculative
  sampling with one-hot proposals, and every emitted token is plain decode's.
- `DecodeEnd::written`, for positions written past the history.
- The batch spec step (B3).

**What is new:**
1. **`main_hidden`.** The last layer's output, one 6144-wide row, replaces the mean of 4 hc copies
   at layers 37–39.
2. **`accept`.** For each newly accepted position i, the MTP layer's **KV cache entry** at i is
   written: `eh_proj`, `input_layernorm`, `kv_a` + latent norm + RoPE, and the indexer key.
   Unlike DSpark's 128-slot window, the MTP layer attends over the **whole context** with its own
   top-2048. Its cache covers every position, including the prompt.
   **The prefill must fill it, but needs no MoE for it.** The KV at position j depends only on the
   layer's input at j, so prefill runs `eh_proj` + `kv_a` + `wk` (~0.2 GFLOP a token) over the
   prompt, and the attention, MoE and head only for the last row.
3. **`draft`: k chained passes.** Each pass is:
   - the MTP layer for one row (per session): attention over the context with the first pass's
     top-k (shared, per `index_share_for_mtp_iteration`), the MoE (8 routed + shared), the head;
   - its sample.

   Each pass also writes a provisional cache entry at its position. It is overwritten after the
   verify, and counted as written past history.
4. **The gate.** There is no confidence head. Candidates:
   - the draft's own probability under the MTP head (survival ≈ the product of p_j, recalibrated
     from measured acceptance);
   - a fixed k per prompt class;
   - the cost-weighted rule DESIGN decision 7 uses, fed by an EMA of the measured per-position
     acceptance.

   The gate's arithmetic (`k_spec_next`, λ·ΔC) carries.

**Cost of a draft pass, per rank:**

| Part | Bytes | Notes |
|---|---|---|
| Attention | 82 MB bf16, 41 MB FP8 | |
| 8 routed experts' quarters | 151 MB bf16, 42 MB NVFP4, 33 MB 3.5-bit | |
| Shared expert | 19 MB | |
| `eh_proj` | 38 MB | |
| **Head** | **476 MB bf16**; 238 MB as an FP8 copy; ~134 MB as MXFP4/NVFP4 | ds41f's FP8 and MXFP4 head kernels exist (`gemv_fp8`, `gemv_fp4.cu`) |
| **Total** | **0.35–0.75 GB, 1.5–3.2 ms a pass** at 235 GB/s | plus 4 collectives and a sample |

The drafter may be approximated (ds41f DESIGN decision 11: every draft is verified), so the MTP
experts should not stay BF16. 4.8 GB a rank at bf16 does not fit the budget. They should take the
main experts' format, and the head an FP8 or FP4 copy. Chaining k drafts costs k passes, against
DSpark's one pass for 5.

**Acceptance.**
- The GLM-5 paper ([arXiv 2602.15763](https://arxiv.org/pdf/2602.15763)) reports an **accept
  length of 2.76 at 4 speculative steps**, against 2.55 for DeepSeek-V3.2, on the authors' private
  prompts, with "parameter-shared MTP layers" (the one layer chained).
- SGLang's GLM-5 and GLM-5.1 recipes ([cookbook](https://cookbook.sglang.io/autoregressive/GLM/GLM-5))
  run EAGLE with 3 steps, top-k 1, 4 draft tokens. I found no published accept length at 3 steps
  for GLM-5.x, nor any for GLM-5.3 itself or for an NVFP4-quantised drafter.
- On this hardware it is **unknown** until measured. The golden run should measure it
  teacher-forced, as DSpark's was (0.80 accepted drafts per step on prose there).

**What it buys** (estimates):
- One row reads ~7.9 GB a rank (FP8 non-experts, NVFP4 experts): 33.5 ms.
- Each extra verified row adds ~6–7 new distinct experts per layer, ~2.6 GB a rank (~11 ms). It is
  smaller as rows' picks overlap; this is to be measured, as DeepSeek's 6 rows touched 25–30 experts.
- With drafts surviving at p ≈ 0.8, 0.6, 0.45:
  - k = 1: ~1.8 tokens in ~47 ms (+28 %);
  - k = 2: ~2.4 tokens in ~61 ms (+32 %);
  - k = 4 (2.76 tokens) breaks even.
- The gate should therefore keep k at 1–3. With BF16 non-experts the base step is larger and
  speculation pays relatively more.

## 5. Memory layout per rank

**How DeepSeek's engine splits at TP=4** (ds41f DESIGN decision 1, [layout.md](../../ds41f-4xgx10-switchless/docs/layout.md)):
- query heads, 16 a rank, with `wq_b` split by rows and `wo_b` by columns into partial sums;
- `wq_a`, `wkv`, the router gate and mHC split by output and all-gathered;
- routed and shared experts **tensor-split**: every rank reads a quarter of every selected expert's
  intermediate columns, so there is no all-to-all and the load is balanced;
- the head split by vocabulary, with distributed sampling;
- the embedding, norms and indexer replicated;
- the attention caches replicated.

**GLM's shapes under the same rules:**

| Tensor class | Split | Per rank | Bytes a rank (all layers) | Read per token a rank |
|---|---|---|---|---|
| `q_a_proj` [2048, 6144] | rows, all-gathered | 512 rows | 0.47 GiB bf16 | 6.3 MB a layer bf16 |
| `kv_a_proj_with_mqa` [576, 6144] | rows, all-gathered | 144 rows (576 / 4: aligned to 16, not to 32-row FP8 blocks; 128 + 16, or a 160/144/144/128 cut, if FP8 blocks are 32 × 32) | 0.13 GiB | 1.8 MB |
| `q_b_proj` [16384, 2048] | rows = heads 16r … 16r+15 | 4096 rows | 1.23 GiB | 16.8 MB |
| `kv_b_proj` [28672, 512] | rows = the same heads, then split into W_UK (transposed) and W_UV | 7168 rows | 0.54 GiB | 7.3 MB |
| `o_proj` [6144, 16384] | columns = the same heads, partial sums | 4096 columns | 3.70 GiB | 50.3 MB |
| Indexer `wq_b` [4096, 2048] | rows = 8 of 32 heads, all-gathered (not replicated, [§3](#3-attention-and-indexer)) | 1024 rows | 0.09 GiB (22 layers) | 4.2 MB a full layer |
| Indexer `wk` [128, 6144], `weights_proj` [32, 6144] | rows, riding the attention all-gather | 32, 8 rows | small | |
| Router `gate` [256, 6144] | rows, logits gathered; or whole on every rank, as DeepSeek's bind does | 64 rows | 0.06 / 0.22 GiB | 0.8 MB |
| Routed experts (75 layers × 256) | **tensor-split**: `gate_proj`/`up_proj` rows [512r, +512), `down_proj` columns [512r, +512) (32 NVFP4 groups, aligned) | a quarter of each | **94.9 GiB NVFP4; ~73.8 GiB at 3.5 bits** | 3.18 GB / 2.48 GB at 8 experts a layer |
| Shared experts (75) | tensor-split like the experts | 512 of 2048 | 1.31 GiB bf16, 0.66 FP8 | 18.9 MB a layer bf16 |
| Dense MLPs (layers 0–2), FP8 | tensor-split: 3072 of 12288 | | 0.16 GiB | 56.6 MB a layer |
| Head [154880, 6144] | vocabulary: 38,720 rows | | 0.44 GiB bf16 | 476 MB |
| Embedding [154880, 6144] | replicated (one 12 KB row a token), or vocabulary-split with a 12 KB all-reduce a step | | 1.77 GiB replicated, 0.44 split | 12 KB |
| MTP layer | like a main layer; experts in the main format | | ~1.6 GiB (NVFP4 experts, FP8 attention) | 0.35–0.75 GB a draft pass |
| KV cache | replicated or position-split ([§3](#3-attention-and-indexer)) | | 52–91 KB a position replicated; 13–23 KB split | 1.2 MB a layer a row at top-2048 (FP8) |

- **Splitting experts.** Tensor-splitting stays right for top-8 of 256. Expert parallelism (64
  experts a rank) would put on average 2 of a row's 8 experts on each rank, but the busiest rank
  would usually have 3–4, so a step would wait on 1.5–2× the reads, plus an all-to-all.
- **Per-token decode reads a rank, one row:**
  - BF16 non-experts and NVFP4 experts: attention 6.53 + shared 1.42 + experts 3.18 + dense 0.17 +
    head 0.48 + router 0.06 ≈ **11.8 GB → ~50 ms floor** (20 tok/s);
  - FP8 attention and shared: **7.9 GB → 33.5 ms** (30 tok/s);
  - plus 3.5-bit experts: **7.2 GB → 30.5 ms** (33 tok/s).

  DeepSeek's floor was 3.26 GB, 13.9 ms. Before speculation GLM decodes at roughly a third to a half
  of DeepSeek's speed on this fleet. The attention weights, not the experts, are the largest share.
- **Resident weights.** At NVFP4 experts and BF16 non-experts, about 107 GiB a rank before KV cache,
  graphs and the OS, against 72 GiB for DeepSeek in 121 GiB. The NVFP4 checkpoint does not fit as
  shipped, which is why [quantisation.md](quantisation.md) exists.
- **A README discrepancy.** Its "355.5 GiB of E2M1 values" includes the MTP layer's 18.0 GiB of
  BF16 experts; the main experts' values are 337.5 GiB.

## 6. Build plan

The user's instruction (2026-10-13): write it from scratch again, copying code where it helps, and
extract every bit of performance by specialising. So:
- every dimension is a compile-time constant;
- every kernel is written for GLM's per-rank shapes, the formats chosen and 1–16 rows;
- nothing is generalised for a model the engine will not run.

As in ds41f, correctness is checked layer by layer against golden activations, and end to end only
statistically.

Effort is relative to the DeepSeek build's own milestones. That whole build took about three days of
several agent lines (2026-10-11 to 2026-10-13, [experiments.md](../../ds41f-4xgx10-switchless/docs/experiments.md)).
"1×" means as much work as that milestone took there.

| # | Milestone | Done when | Effort vs ds41f |
|---|---|---|---|
| G0 | **Formats and microbenchmarks.** The quantisation study's decisions: expert format, non-expert format, activation side, KV format, KV split (replicated or position-split). `micro/mma.cu` gains `mxf4nvf4 … scale_vec::4X … ue4m3` (bit-exact against a CPU reference, throughput, and the nibble order of packed operands) and BF16-widened NVFP4 and 3.5-bit GEMV prototypes at % of 235 GB/s | decisions written in DESIGN.md; the microbenchmarks recorded in findings.md | 0.5× M0 (the hardware facts carry; only the new formats are measured) |
| G1a | **First exact-output checkpoint: golden data on a few layers.** A layer-at-a-time reference from HF `GlmMoeDsaForCausalLM` (experts dequantised per layer, or quantised as G0 decides), CPU or one GB10, on a short sequence (~300 tokens) and one of ≥ 4k tokens (so top-2048 selects), saving per-layer inputs and outputs, router choices and top-k indices for layers 0, 2 (dense, full indexer), 3 (first MoE, shared indexer), 6 (full), 7, 77 and the MTP layer | the reference's last-layer logits on the short sequence agree with an independent run (vLLM or SGLang GLM-5.3, if one runs on these boxes; else greedy text that reads right) | 0.4× M1 |
| G1b | Full golden data: all 78 layers + MTP on prose, code and an agentic transcript of ~30k tokens, MTP acceptance teacher-forced | accept length measured; argmax agreement with the independent stack within numeric noise | 0.6× M1 |
| G2 | **Converter, verifier, loader.** `plan.cpp`'s GLM table, L4/L5 tiling from the kernels' own functions, FP8 re-quantisation if chosen, per-rank files on all four boxes | the ds41f checks: plan coverage, `dsw_verify` on every byte (or against the reference quantiser), `check_dsw.py`, cross-rank CRCs of replicated tensors, `dsw_kcheck` | 0.6× M2 (the container, loader and verifier skeleton carry) |
| G3 | **Kernels at TP=4 shapes, one rank**, each fed G1a's inputs: GEMVs in the chosen formats (BF16/FP8 attention, grouped W_UK/W_UV, NVFP4/own experts with the 8-expert union, router), sparse MLA attention (576/512, top-2048, split-K), the indexer (LayerNorm key, fp32 scores, 32-bit select, skip below 2048, position split + merge), norms. Every kernel with slot tables and 16 rows from the start (B2–B4 retrofits cost a day in ds41f) | every layer type matches golden (block rel-L2 at bf16 level, routing ≥ 99.5 %, top-k sets 100 % away from ties); each GEMV ≥ 95 % of 235 GB/s | **1.3× M3**: more distinct kernels (absorption, a new indexer, a new expert format) and no CED/mHC/compressor to drop work onto |
| G4 | **TP=4 decode**: the layer runner, fused collectives (residual + norm epilogues), one CUDA graph per row count, PDL, the paced L2 prefetch, then merging collectives (~240 a step) | greedy decode matches G1b's logits within the reference's envelope; step time within 1.25× of the floor at k=1 on the chosen formats | 0.7× M4 (the step machinery carries; the measured gaps are already known) |
| G5 | **Prefill**: chunked, sequence-parallel, tensor-core GEMMs on the chosen MMA (`mxf4nvf4` or bf16), absorbed sparse attention, indexer scores in prefill, the MTP cache fill (no MoE), chunks ≥ 8192 | the prefill path's state equals the step path's up to summation order, and splitting a prompt into chunks gives identical bits; tok/s against the baseline stack's on this fleet | 1.2× M5: no CED, so all 78 layers run over every token (~5× the FLOPs per token), so performance work matters more |
| G6 | **Serving**: the server, sessions (no snapshots; exact restart at any common prefix), the store with large pages, the disk tier, kept sessions; dsfront with the GLM template, pre-tokeniser and `<tool_call>` parser (can start right after G0, on the CPU only) | pi works unchanged against a test port: tools, thinking, streaming, multi-turn reuse; `enccheck` byte-exact against the jinja template; tokenizer byte-exact against `tokenizers` | 0.6× M6 + the serving part of M4/M9 (protocol, server and store carry; the encoding is new) |
| G7 | **MTP speculation**: `Mtp::accept`/`draft` (chained, shared top-k), the drafter's approximated experts and head, the gate, the spec step graph, B3's batch drafts | tokens identical with and without drafts (greedy and sampled, as `refhash` checks); tok/s better than no drafts on every prompt class | 0.8× M7 (the verify loop carries; the drafter is simpler than DSpark but chained) |
| G8 | **Hardening and ship**: device memory at startup, service.sh, soak, `refhash.sh` hashes, benches against the baseline stack | as ds41f's M9 list | 0.5× M9 |

**Order.** The critical path to a correct engine is G0 → G1a → G2 → G3 → G4. G1b runs beside G2–G3,
since it needs only the reference. G6's front end can start right after G0. G5 and G7 are where the
speed comes from, after G4. G7 can start once G4's step exists, because the verify loop is the
same. B1–B4 batching is designed into G3 and G4, not added after.

**Where to specialise first, in order of decode bytes.**
1. **The attention weights** (6.5 GB a rank at bf16): their format, and grouped absorption fused
   into the `q_b` and `o_proj` GEMVs where the shapes allow.
2. **The routed experts' GEMV** (3.2 GB): a union read once, in the chosen format, at ≥ 95 % of
   bandwidth from 1 to 16 rows.
3. **The shared experts and head** (1.9 GB): fuse the shared expert into the MoE launch, as ds41f
   does.
4. **The collectives**: 3 a layer × 78, at 12 µs each, are ~2.9 ms of a ~30–50 ms step.

The indexer and attention compute are small at decode, except the indexer's scan beyond ~64k
context.

## Sources

- Reference model code: transformers 5.15.0, `models/glm_moe_dsa/modeling_glm_moe_dsa.py` and
  `configuration_glm_moe_dsa.py` (read 2026-10-13).
- Checkpoint headers: `nvidia/GLM-5.3-NVFP4` revision `e3b8f9f2`, `config.json`,
  `hf_quant_config.json`, `model.safetensors.index.json`, `tokenizer.json`, `chat_template.jinja`.
- GLM-5 MTP accept length: [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/pdf/2602.15763);
  SGLang's settings: [SGLang cookbook, GLM-5](https://cookbook.sglang.io/autoregressive/GLM/GLM-5),
  [GLM-5.1](https://cookbook.sglang.io/autoregressive/GLM/GLM-5.1).
- The DeepSeek engine: `ds41f-4xgx10-switchless/` at `fd663e6`.
