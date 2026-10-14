# The CPU reference (experiment A1)

altar-1's golden data: glm53's layer-streaming torch reference ([glm53 reference](../../glm53-4xgx10-switchless/docs/reference.md),
cited below as *glm53 § n*) adapted to altar-1's checkpoint, run teacher-forced on rank 2's CPU. This page is a
**delta** from glm53's: what is identical, what changed, the checks (A1a), the golden files and the level-3
comparison against GLM-5.3 (A1b), timings, and how to resume. Written 2026-10-14; everything was measured on that
date unless marked *estimate*.

## Contents

1. [What is identical to glm53's reference](#1-what-is-identical-to-glm53s-reference)
2. [What changed](#2-what-changed)
3. [Files and how to run it](#3-files-and-how-to-run-it)
4. [Checks (A1a)](#4-checks-a1a)
5. [Timing and memory on rank 2](#5-timing-and-memory-on-rank-2)
6. [Results](#6-results)
7. [Level 3: altar-1 against GLM-5.3](#7-level-3-altar-1-against-glm-53)
8. [The golden files](#8-the-golden-files)
9. [Open questions](#9-open-questions)
10. [How to resume](#10-how-to-resume)

## 1. What is identical to glm53's reference

The math and the numerics contract, line for line (`glmref.py`'s docstring): BF16 activations, f32 accumulation
with one output rounding, RMSNorm, interleaved RoPE, the expanded MLA in query blocks of 256 with eager numerics,
the DSA indexer and its top-2048 (sets saved ascending, the boundary margin saved), the f32 router (bias for
selection only, top-8 sorted by biased score, normalised, × 2.5), f32 logits, the MTP layer at position t with the
embedding zeroed at position 0 (glm53 § 2). The two rounding orders are kept:

- `--combine f32` (**the contract**, glm53 DESIGN decision 8): o_proj, the dense MLP's down projection, each routed
  expert's weighted down projection and the shared expert's down projection stay f32, are summed in f32 (routed
  experts in expert-id order, then the shared expert) and rounded once with the residual. Runs `all-f32`, `a1a-f32`.
- `--combine bf16` (transformers' order): each of those GEMMs rounded to bf16, the routed experts weighted and summed
  in f32 and rounded, the shared expert and the residual added in bf16. Runs `all`, `a1a-bf16`, `doc`.

**Both are the prefill order**: one forward from position 0 a sequence. The engine's decode sums a strip's K in
another order by default (glm53 DESIGN decision 11), so a decode-step comparison against these files is an
envelope comparison, as in glm53. The token sequences, the sample rows, the resumable layer-by-layer state, the
deterministic flag and every file format are glm53's, so the engine's `compare_golden.py` and `gold_prep.py` read
these runs unchanged with a path change.

## 2. What changed

| | glm53 (NVIDIA's NVFP4 checkpoint) | altar-1 |
|---|---|---|
| routed experts a layer | 256 | **168**; the config check asserts 168, top-8, 2.5, one group, three dense layers and the `pack-quantized` INT4 group-32 asymmetric scheme |
| routed experts, layers 4–76 | NVFP4: E2M1 · E4M3 block scale exact in f32, `weight_scale_2` on the f32 GEMM output | **INT4, dequantised to BF16 by compressed-tensors' own rule** (§ 2.1), then a plain BF16 GEMM |
| `q_b_proj`, `kv_b_proj`, `o_proj` of layers 4–76 | BF16 | **INT4**, the same rule, dequantised once a layer and cached |
| routed experts of layers 3, 77 and the MTP layer 78 | NVFP4 (3, 77), BF16 (78) | BF16 (all three) |
| dense MLPs, layers 0–2 | FP8 per-tensor, f32 GEMM with the scale after | **BF16**, row chunks of 2,048 |
| which tensors are INT4 | by layer (`l != 78`) | **by name**: a module with `weight_packed` is INT4, one with `weight` is BF16 (`Ckpt.is_int4`) |
| shards | all trusted after a copy check | `--verified FILE`: only shards listed `OK` in the SHA-256 log against the Hub manifest are opened; `--wait` sleeps until a shard is listed |
| head | `lm_head` widened to f32 whole (10.9 GiB peak) | kept BF16 and widened a 16,384-row vocabulary slab at a time (same f32 GEMM per logit) |
| memory in a layer | full f32 temporaries | q, K, V built in 4,096-row chunks (rows of a GEMM are independent), the attention output added to the residual a sequence at a time, the shared expert added into the routed experts' f32 accumulator in place, `malloc_trim` after a layer: **bit-identical** to the unchunked code on 37 of 37 files of 8 layers (`a1a-f32` layers 0–7, 2026-10-14 11:19) |
| tracks | `--track`, `ne:` presets, `--master`, `--qcache`, `--cov-out`, `qprep.py`, `check_nvfp4_rtn.py` | **dropped** (nothing to choose: the served weights are the checkpoint's bytes, DESIGN decision 1) |
| MTP experts at NVFP4 | `--mtp-from RUN --mtp-experts nvfp4_rtn` | kept (DESIGN decision 5's lever): glm53's `quant/formats.py` `nvfp4_rtn`, imported from `glm53-4xgx10-switchless/quant/` |
| summaries | 256 experts, top-64 | 168 experts, top quarter = 42 |

### 2.1 The INT4 weight

compressed-tensors 0.18.0 (`compressors/pack_quantized/base.py` `decompress`, lines 148–161: the zero point unpacked
with `unpack_from_int32(..., packed_dim=0)`, the values with `unpack_from_int32(packed, 4, shape)`, then
`dequantize`; `compressors/pack_quantized/helpers.py` `unpack_from_int32`: element i of each word at bits 4i,
`return (output - offset).to(torch.int8)` with offset 8; `quantization/lifecycle/forward.py` `dequantize` lines
132–133: `dtype = scale.dtype`, bf16 here; `quantization/lifecycle/forward_helpers.py` `_dequantize` lines 562–567):

```
dequant_value = x_q.to(scale.dtype)                                  # int8 → bf16, exact
dequant_value = dequant_value - zero_point.to(scale.dtype)           # bf16 − bf16, exact (|q − z| ≤ 15)
dequant_value = dequant_value * scale                                # bf16 × bf16: the one rounding
```

`glmref.int4_dequant` runs the same three torch ops on the same operands: q = nibble − 8 from `weight_packed`
(I32 [N, K/8], k = 8j + i at bits 4i), z = nibble − 8 from `weight_zero_point` (I32 [N/8, K/32], row 8w + i at
bits 4i), s = `weight_scale` (BF16 [N, K/32]), viewed [N, K/32, 32]; and asserts the result's shape equals
`weight_shape`. So W = bf16_rne((q − z) · s), as [model.md § 4.2](model.md#42-dequantisation) derives and A0a
([attention.md § 3](attention.md#3-the-int4-contract-and-its-rounding-point)) found on the attention tensors.

## 3. Files and how to run it

| File | What |
|---|---|
| [`ref/glmref.py`](../ref/glmref.py) | the reference (§ 1–2) |
| [`ref/passes.sh`](../ref/passes.sh) | the passes in order (`a1a-bf16`, `a1a-f32`, `all-f32`, `all`, `mtp-nvfp4`), each resumable and retried after a kill; a finished pass is skipped |
| [`ref/check_int4.py`](../ref/check_int4.py) | `int4_dequant` against compressed-tensors' `PackedQuantizationCompressor.decompress` with the checkpoint's scheme, bit for bit, on named tensors or all experts of a layer |
| [`ref/check_layer.py`](../ref/check_layer.py) | one real layer against transformers' GlmMoeDsa modules + compressed-tensors (§ 4.3) |
| [`ref/check_router.py`](../ref/check_router.py) | DESIGN decision 10: the kept router rows and biases against the parent's (§ 4.4) |
| [`ref/check_tokens.py`](../ref/check_tokens.py) | the token files re-rendered with altar-1's template (§ 4.1) |
| [`ref/level3.py`](../ref/level3.py) | level 3: argmax agreement, top-5 containment and KL against GLM-5.3's golden runs, with both models' order floors (§ 7) |
| [`ref/summarize.py`](../ref/summarize.py), [`ref/cmp_runs.py`](../ref/cmp_runs.py), [`ref/make_tokens.py`](../ref/make_tokens.py), [`ref/tokens/`](../ref/tokens) | glm53's, with 168 experts in `summarize.py`; the tokens unchanged |

Rank 2 holds the checkpoint at `~/altar-1` (39 shards, every one `OK` in `~/altar-1/verify.log` against the
revision's manifest; the seven small non-LFS files have no Hub SHA-256 and show as `BAD` there), glm53's golden runs
at `~/glm53-ref/golden`, and this reference's venv and outputs under `~/altar1-ref`:

```
python3 -m venv ~/altar1-ref/venv
~/altar1-ref/venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
~/altar1-ref/venv/bin/pip install numpy safetensors tokenizers jinja2 transformers==5.16.1 compressed-tensors==0.18.0
# the code: this repository's altar1-4xgx10-switchless/ref and glm53-4xgx10-switchless/quant/formats.py, in the
# same relative layout, under ~/altar1-ref/code

# every process: oom_score_adj 1000, cores 10-19, nice 19, idle I/O
cd ~/altar1-ref/code/altar1-4xgx10-switchless/ref
OMP_NUM_THREADS=8 setsid nohup taskset -c 10-19 nice -n 19 ionice -c3 ./passes.sh >> ~/altar1-ref/logs/passes.log 2>&1 < /dev/null &

~/altar1-ref/venv/bin/python -I check_int4.py --ckpt ~/altar-1 --verified ~/altar-1/verify.log [--tensors] [--all-experts 4 40 76]
~/altar1-ref/venv/bin/python -I check_layer.py --ckpt ~/altar-1 --run ~/altar1-ref/golden/a1a-bf16 --layer L --group 12
~/altar1-ref/venv/bin/python -I check_router.py extract --parent <GLM-5.3-NVFP4 snapshot> --out parent_routers.npz   # rank 0
~/altar1-ref/venv/bin/python -I check_router.py compare --ckpt ~/altar-1 --parent-npz parent_routers.npz
~/altar1-ref/venv/bin/python -I level3.py --altar ~/altar1-ref/golden --glm ~/glm53-ref/golden --json level3.json
```

Everything that reads the shards runs under `python -I`.

## 4. Checks (A1a)

### 4.1 Tokens

The four token files are glm53's (`ref/tokens/SHA256SUMS`, equal to the copies under rank 2's glm53 reference).
`check_tokens.py`: re-rendered with **altar-1's** `chat_template.jinja` and encoded with its `tokenizer.json`, every
sequence's text and ids equal the files (short 291, prose 4,762, python 6,106, agentic 26,834 tokens), and
transformers 5.16.1's `apply_chat_template` gives the same text and ids. None of the conversations has a null
content, so altar-1's older template ([model.md § 8.1](model.md#81-the-chat-template)) renders them as GLM-5.3's does.

### 4.2 The INT4 dequantisation against the library

`check_int4.py` against `PackedQuantizationCompressor.decompress` (compressed-tensors 0.18.0, torch 2.14.1):

| Tensors | Values | Bit-equal | Products not exact in BF16 (the rounding matters) |
|---|---:|---:|---:|
| 9 named: layer 4 expert 0 gate/up/down, layer 4 q_b/kv_b/o_proj, layer 40 expert 100 down, layer 76 expert 167 up, layer 76 o_proj | 312,475,648 | **312,475,648** | 24.5 % (q_b 15.9 %, o_proj 25.9–26.2 %, experts 22.6–25.2 %) |
| all 168 experts × 3 of layers 4, 40 and 76 (1,512 tensors) | 19,025,362,944 | **19,025,362,944** | 24.7 % |

A quarter of the weights are a rounded product, so a dequantiser that rounds elsewhere (`q·s − z·s`, an FMA, f16)
would differ on a large share; this one equals the library everywhere.

### 4.3 Real layers against transformers + compressed-tensors

`check_layer.py`, transformers 5.16.1: `GlmMoeDsaAttention`, `GlmMoeDsaRMSNorm`, `GlmMoeDsaTopkRouter`,
`GlmMoeDsaMLP` and `GlmMoeDsaExperts` (eager attention, `grouped_mm` experts) built from the checkpoint's config and
loaded with `load_state_dict(strict=True)` from the checkpoint; every INT4 tensor dequantised by compressed-tensors'
`decompress`, not by `glmref.py`. No released transformers loads altar-1 with `from_pretrained` correctly (it drops
the experts' zero points and leaves the BF16 experts of layers 3, 77 and 78 at random initialisation,
[model.md § 4.4](model.md#44-what-transformers-does-when-it-loads-this-checkpoint)), so the modules are loaded
here. The routed experts run through transformers' `grouped_mm` forward in groups of 12 experts (a whole layer is
12.7 GB in BF16), the other pairs sent to the group's sentinel id, the group sums kept f32 and rounded once:
transformers' arithmetic except the association of the f32 sum over a row's 8 experts. Input: `a1a-bf16`'s hidden
state before the layer at every position (the embedding for layer 0), so both sides see the same bf16 input;
`glmref.py`'s answer is `a1a-bf16`'s output of the layer (transformers' order). Two sequences: `prose2400` (the
first 2,400 tokens of `prose`: 352 rows past position 2048, so the sparse path) and `short` (291).

| Layer | What it tests | Attention output | Index sets (rows ≥ 2048) | Routes | Layer output: bit-equal, rel-l2, max \|diff\| (prose2400 / short) |
|---|---|---|---|---|---|
| 0 | dense BF16 MLP, full indexer, BF16 attention | bit-equal 100 % | 352/352, no boundary ties | — | **100 % / 100 %**, 0 |
| 3 | BF16 experts, shared indexer | 100 % | (layer 2's, from the run) | sets equal on 100 % of rows, weights within 1.2e-7 | 95.65 % / 94.87 %, 5.0e-4 / 6.3e-4, 9.8e-4 |
| 4 | INT4 experts + INT4 q_b, kv_b, o_proj | 100 % | — | 100 %, 1.8e-7 | 88.82 % / 87.90 %, 8.4e-4 / 7.9e-4, 9.8e-4 |
| 77 | BF16 experts, last layer | 100 % | — | 100 %, 2.4e-7 | 86.69 % / 83.08 %, 3.1e-3 / 4.2e-3, 16 |

- **The attention sublayer is bit-equal on every layer**, INT4 projections included, and layer 0 is bit-equal end
  to end. The routers choose the same eight experts on every row of every layer.
- The layer outputs differ only through the routed experts' f32 sum order (transformers' slot order and our
  grouping against `glmref.py`'s expert-id order), which moves the bf16 rounding of the experts' sum by one unit of
  that sum's magnitude: 9.8e-4 = 2⁻¹⁰ is one bf16 ulp at 0.125–0.25, 16 is one ulp at 2,048–4,096 (layer 77's
  outlier channels). glm53's random-weight check of the same kind (16 experts) was bit-equal on 99.67 %, rel-l2
  3.1e-4 (glm53 § 3); real weights with 168 experts and larger hidden magnitudes deeper in the stack give the larger
  share. This is BF16 noise in the A1a sense.
- Peak RSS of a check: 8.0 GiB on layer 0 (eager attention over 2,400 rows: transformers materialises the
  [64, 2400, 2400] scores and the indexer's [2400, 32, 2400] f32 scores), 5.5 GiB on the MoE layers.

### 4.4 The routers against the parent (DESIGN decision 10)

`check_router.py`: the parent GLM-5.3's `mlp.gate.weight` [256, 6144] BF16 and `e_score_correction_bias` [256] F32
read from NVIDIA's GLM-5.3-NVFP4 snapshot on rank 0 (BF16 copies of the parent's values), at the ids of
`reap_prune_plan.json` (`layers["L"][j]` = the parent id of altar-1's expert j), against altar-1's:

| Layer | Bias (168) | Router rows (1,032,192 values) |
|---|---|---|
| 3 | equal | **bit-equal** |
| 4 | equal | 314 bit-equal: columns scaled by AWQ's factors (0.68–2.54); \|ratio / column median − 1\| median 1.3e-3, p99 3.5e-3; bf16(parent × column factor) reproduces 97.1 % |
| 40 | equal | 70,981 bit-equal; factors 0.88–1.34; deviation median 1.2e-3, p99 4.1e-3; reproduces 95.1 % |
| 77 | equal | **bit-equal** |
| 78 (MTP) | equal | **bit-equal** |

So the kept experts' routers are the parent's: exactly on the BF16 layers 3, 77, 78, and up to one per-column factor
within bf16 rounding (the deviations stay under one bf16 ulp, 3.9e-3) where AWQ's smoothing is folded into the router's
input columns ([model.md § 5](model.md#5-the-bf16-tensors-are-cyankiwis-not-glm-53s)). A row matched to the wrong
parent expert would show a ratio spread of order one. The biases are equal everywhere.

### 4.5 Greedy text reads right: the document diagnostic

glm53's diagnostic (glm53 § 6.3): `doc`, the first 8,000 characters of `json/__init__.py` as a plain document
(2,114 tokens), and `docchat`, the same text quoted in a user turn (2,134), `--combine bf16`, no MTP:

| Sequence | altar-1 top-1 / mean log p | GLM-5.3 (glm53 § 6.3) |
|---|---|---|
| `doc` | **0.971** / −0.130 | 0.984 / −0.071 |
| `docchat` | 0.736 / −1.390 | 0.736 / −1.529 |

The memorised file comes back at 97 %, dense and sparse rows alike: the weights, the dequantisation, attention, the
indexer and the head are right. This is A1a's "greedy text reads right".

## 5. Timing and memory on rank 2

Rank 2 (DGX Spark), the DeepSeek service beside it (~98 GiB, MemAvailable ~21–35 GiB during the runs); 8 threads
pinned to cores 10–19, nice 19, `ionice -c3`, `oom_score_adj` 1000.

| Pass | Rows | Per layer | Whole pass | Peak RSS (`VmHWM`) |
|---|---:|---|---|---|
| `a1a-bf16` (short + prose2400, every position saved) | 2,691 | 4.8–5.2 s dense, 9.2–9.5 s MoE | 761 s (head 11 s, MTP 23 s) | 5.3 GiB (the head stage: full logits of all 2,691 rows) |
| `a1a-f32` | 2,691 | 11.6–13 s MoE | 936 s | 5.4 GiB |
| `doc` (`--no-mtp`) | 4,248 | | 906 s | 4.0 GiB |
| `all-f32` | 37,993 | ~120 s (attention 82–97 s) | FILL | FILL |
| `all` | 37,993 | FILL | FILL | FILL |

The first `all-f32` attempt (11:06–11:17) peaked at **8.4 GiB** in its fifth layer (the 26,834-row `agentic`
attention with full-size q_b/kv_b outputs and f32 temporaries) and RSS stayed 4–5 GiB between layers. It was
stopped, the chunking of § 2 added and checked bit-identical, and the pass resumed at layer 5 (its state after layer
4 is the same bits under either code). After that: FILL.

## 6. Results

FILL

## 7. Level 3: altar-1 against GLM-5.3

FILL

## 8. The golden files

FILL

## 9. Open questions

FILL

## 10. How to resume

FILL
