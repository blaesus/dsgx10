# The CPU reference (experiment A1)

altar-1's golden data: glm53's layer-streaming torch reference ([glm53 reference](../../glm53-4xgx10-switchless/docs/reference.md),
cited below as *glm53 § n*) adapted to altar-1's checkpoint, run teacher-forced on rank 2's CPU. This page is a
**delta** from glm53's: what is identical, what changed, the checks (A1a), the golden files and the level-3
comparison against GLM-5.3 (A1b, A1c), the second order's numbers (A1d), timings, and how to resume. Written
2026-10-14; everything was measured on that date (times +0100) unless marked *estimate*.

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
| [`ref/mtp_accept.py`](../ref/mtp_accept.py) | the MTP acceptance by tokens of `all-f32`, `all`, `mtp-nvfp4` and their drafts against `all-f32`'s (§ 6.1, § 8) |
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
| `all-f32` | 37,993 | median 124.6 s (119–141 s; attention 82–101 s of it) | **~10,870 s (3 h 1 min)**: layers 0–4 ~600 s in the first attempt, then 10,265 s for layers 5–77 (11:19:55–14:06), head 125 s, MTP 263 s (its layer 132 s) | **7.9 GiB** (7.3–7.6 in the layers, 7.6 in the head stage) |
| `all` | 37,993 | median 125.1 s (99–170 s; attention median 92.9 s, 73–131 s); dense layers 0–2 103 s | **10,241 s (2 h 51 min)**, one invocation 14:11:00–17:01:41: layers 9,821 s, head 125.6 s, MTP 261.9 s (its layer 112.0 s) | **7.4 GiB** (6.5–6.8 in the layers and the head stage, 7.4 at the pass's end: the MTP stage) |
| `mtp-nvfp4` (`--mtp-from all-f32 --mtp-experts nvfp4_rtn`, layer 78 only) | 37,993 | layer 78: 208.0 s (attention 133.9 s) | **350 s**, 17:01:41–17:07:31 (MTP stage 346.8 s) | not logged (the MTP replay prints no `hwm`) |

The first `all-f32` attempt (11:06–11:17) peaked at **8.4 GiB** in its fifth layer (the 26,834-row `agentic`
attention with full-size q_b/kv_b outputs and f32 temporaries) and RSS stayed 4–5 GiB between layers. It was
stopped, the chunking of § 2 added and checked bit-identical, and the pass resumed at layer 5 (its state after layer
4 is the same bits under either code). After that it peaked at 7.9 GiB, under the 8 GiB cap; RSS between layers
stayed ~4 GiB (the residual stream, the index sets, glibc's arenas). Against glm53's 6,739 s (bf16) and 12,733 s
(f32, re-pinned mid-run) on the same box class, the attention of the 26.8k-row sequence dominates here (~70 % of a
layer). By 14:16 the DeepSeek service no longer held its memory on rank 2 (MemAvailable ~95–110 GiB), so the `all`
pass's per-layer times are not comparable with `all-f32`'s; the engine's A8 ring jobs (14:44–16:37) ran beside it.
Source of every time and peak in this table: `~/altar1-ref/logs/passes.log` on rank 2 (the per-layer lines,
`pass done (… s this invocation) hwm …` and the `pass … exit 0` stamps).

## 6. Results

### 6.1 The full pass in the contract's order (`all-f32`)

Teacher-forced, the main model's argmax against the next token; the MTP layer's draft (row t drafts token t+2)
against the token and against the main model's argmax at t+1 (the one-draft greedy acceptance, glm53 § 6.7):

| Sequence | Tokens | Top-1 | Mean log p | Perplexity | MTP == token t+2 | **MTP == main argmax at t+1** | MTP == token where main right |
|---|---:|---:|---:|---:|---:|---:|---:|
| `short` | 291 | 0.528 | −2.593 | 13.4 | 0.481 | **0.661** | 0.797 |
| `prose` | 4,762 | 0.422 | −3.133 | 22.9 | 0.392 | **0.522** | 0.739 |
| `python` | 6,106 | 0.686 | −1.576 | 4.84 | 0.647 | **0.698** | 0.836 |
| `agentic` | 26,834 | 0.474 | −3.532 | 34.2 | 0.456 | **0.438** | 0.701 |
| all, by tokens | | | | | | **0.492** | |

GLM-5.3 in the same order (glm53 § 10.2): top-1 0.807 / 0.511 / 0.695 / 0.488, MTP acceptance 0.796 / 0.587 /
0.688 / 0.454 (0.511 by tokens). On code and the agentic transcript altar-1 is within 1–1.5 points of its parent; on
the two prose sequences, where GLM-5.3 recites memorised text, it is 9–28 points lower (§ 7). The MTP drafter, with
experts chosen by weight norm ([model.md § 7](model.md#7-layers-3-and-77-and-the-mtp-layer)), keeps its parent's
acceptance on python and agentic (0.698 vs 0.688, 0.438 vs 0.454) and loses 6–14 points on prose.

**`all`** (transformers' order; `summarize.py`, 17:07:16, `~/altar1-ref/logs/a1d-summarize-all.txt`): top-1 0.541 /
0.427 / 0.688 / 0.474, mean log p −2.605 / −3.109 / −1.561 / −3.540; MTP == main argmax at t+1 0.661 / 0.515 /
0.701 / 0.441, **0.494 by tokens** (18,764 / 37,985; `all-f32` 18,690 / 37,985 = 0.492; `~/altar1-ref/logs/a1d-mtp.txt`).
The order moves top-1 by at most 1.3 points (`short`) and the acceptance by at most 0.7. `short` inside `all` is
bit-identical to `short` in `a1a-bf16` (argmax, log p, the full logits at all 291 rows, every layer's routed sets
and hidden state; `cmp_runs.py`, 17:10, `~/altar1-ref/logs/a1d-cmp-all-a1abf16-short.txt`), so batch invariance
holds in both orders.

**Batch invariance.** `short` inside the 38k-row `all-f32` and `short` in `a1a-f32` (with `prose2400`) are
bit-identical: argmax, log p and the full logits at all 291 rows (`cmp_runs.py`, max |Δ logit| 0). `gold_prep.py`
reads the run unchanged: `hfinal` (37,993 rows) bit-equal to `final_hidden_sample` at the 634 sample rows.

### 6.2 The reference's own order variance

**The four sequences (A1d).** `cmp_runs.py all all-f32 --seq S` and the reverse (rank 2, 17:07–17:08,
`~/altar1-ref/logs/a1d-cmp-S.txt` and `a1d-cmp-rev-S.txt`), KL over the full logits at the sample rows, its median
from `level3.py` (§ 7, the `altar all | altar all-f32` row):

| Sequence | Rows (KL rows) | Argmax equal | KL(all ‖ all-f32) mean / median / max | KL(all-f32 ‖ all) mean / max | max \|Δ logit\| | \|Δ log p(next)\| mean |
|---|---:|---:|---|---|---:|---:|
| `short` | 291 (291) | 0.8694 | 0.098 / 0.029 / 1.86 | 0.092 / 2.10 | 7.61 | 0.320 |
| `prose` | 4,762 (84) | 0.8696 | 0.048 / 0.013 / 0.43 | 0.048 / 0.72 | 4.80 | 0.323 |
| `python` | 6,106 (89) | 0.9147 | 0.150 / 0.011 / 2.26 | 0.205 / 4.40 | 10.49 | 0.241 |
| `agentic` | 26,834 (170) | 0.7703 | 0.297 / 0.019 / 10.1 | 0.365 / 9.15 | 14.61 | 0.558 |
| by tokens | 37,993 | **0.8067** | mean of the four 0.148 | 0.178 | | |

GLM-5.3's same pair (glm53 § 10.2, recomputed by `level3.py`): argmax 0.935 / 0.844 / 0.898 / 0.755 (0.790 by
tokens), KL mean 0.052 / 0.070 / 0.410 / 0.318. altar-1 is more order-sensitive on `short` (13 % of argmaxes
against 6.5 %) and less on prose, python and agentic; over all rows the two models' floors are alike (0.807
against 0.790). Routed sets on `short` differ on 6 % of rows at layer 3 and on 68–76 % by layers 70–77, the hidden
state's median relative difference grows from 0.32 % at layer 0 to 20 % at layer 77 (`a1d-cmp-short.txt`).

**Agreement with A8.** The end-to-end milestone's "orders" rows ([engine.md § 10.3](engine.md#10-end-to-end-on-real-weights-a8),
KL(all-f32 ‖ all), compared by `compare_golden.py g8`) are the same numbers: short 0.869 / 0.092 / 7.6, python
0.915 / 0.205 / 10.5, prose 0.870 / 0.048 / 4.8 (argmax / KL mean / max |Δ logit|), equal to the reverse column
above to the digits shown. Its agentic row (0.736 / 0.323 / 13.0) covers the first 8,000 tokens only (65 sample
rows), so it is not the whole-sequence 0.770 / 0.365 / 14.6 here; and KL's direction matters on python (0.150 one
way, 0.205 the other).

**A1a's pair** (`short` and `prose2400`), `cmp_runs.py a1a-bf16 a1a-f32` (the same input, the two rounding orders):

| Sequence | Argmax equal | KL(bf16 ‖ f32) mean / max | max \|Δ logit\| | \|Δ log p(next)\| mean | GLM-5.3, same comparison (glm53 § 10.2) |
|---|---:|---|---:|---:|---|
| `short` | 0.869 | 0.098 / 1.86 | 7.6 | 0.32 | 0.935, KL 0.052 / 2.0 |
| `prose2400` | 0.874 | 0.160 / 12.9 | 14.7 | 0.34 | (`prose` 0.844, KL 0.070 / 0.74) |

The hidden state already differs by 0.3 % (relative, median) after layer 0, since the dense layers round o_proj and
the MLP down projection differently in the two orders, and 0.75 % by layer 7; routed sets differ on 6–13 % of rows
in layers 3–7. altar-1 is more order-sensitive than GLM-5.3 on `short` (13 % against 6.5 % of argmaxes).

### 6.3 Expert usage at 168

Routing mass (the normalised weights) per expert over the 37,993 rows of `all-f32`, layers 3–77:

| | Range over layers | Typical (layers 23–77) |
|---|---|---|
| experts carrying 50 % of the mass | 35–67 | 40–63 |
| experts carrying 90 % | 115–143 | 121–139 |
| experts carrying 99 % | 156–165 | 156–164 |
| share on the top quarter (42; uniform 0.25) | 0.337–0.556, mean 0.438 | |
| experts never routed to | 0 in every layer | |
| rows whose 8th and 9th biased scores are within 1e-3 | mean 0.256 over layers; 0.96, 0.88, 0.77 at layers 3, 4, 5; 0.11 at 40, 0.13 at 77 | |

- The most concentrated layers are 19–22 and 32 (top-quarter share 0.51–0.56), as in GLM-5.3 (layers 17–22).
- The hot set transfers no better than at 256: the 42 experts with the most mass on the other three sequences carry
  0.330 of `agentic`'s mass (minimum 0.278), its own top 42 carry 0.456.
- The MTP layer (78): 43 / 125 / 162 experts for 50 / 90 / 99 %, top-quarter share 0.498.
- `all` (transformers' order) routes alike: 35–67 / 116–143 / 156–165 experts for 50 / 90 / 99 % over layers 3–78,
  top-quarter share mean 0.439 (0.337–0.555), none unused; `agentic` on the others' hot 42: 0.330 (minimum 0.280).
- The router margins are as thin as GLM-5.3's (a quarter of rows under 1e-3, glm53 § 6.5): the engine's routing will
  flip against golden where its f32 sum order differs, so tolerances come from the order variance, not a bit match.

## 7. Level 3: altar-1 against GLM-5.3

`level3.py --altar ~/altar1-ref/golden --glm ~/glm53-ref/golden` (rank 2, 2026-10-14 14:20; rerun 17:07:27 with
`all` finished, `~/altar1-ref/logs/a1d-level3.txt` and `~/altar1-ref/golden/level3.json`, the `all-f32` rows
unchanged to the digit; the 14:20 JSON kept as `~/altar1-ref/logs/level3.before-a1d.json`). Argmax over every
row; KL(GLM-5.3 ‖ altar-1) over the 154,856 used vocabulary ids at the sample rows (all 291 of `short`; 84, 89 and
170 of the others); "in top-5" = altar-1's argmax differs from GLM-5.3's but is among GLM-5.3's top 5. The noise
floor is GLM-5.3's own two orders (`all` against `all-f32`).

| Sequence | altar-1 `all-f32` vs GLM-5.3 `all-f32`: argmax equal | differs, in GLM top-5 | differs, outside | KL mean / median / p99 / max | **floor** (GLM `all` vs `all-f32`): argmax equal | floor KL mean / median |
|---|---:|---:|---:|---|---:|---|
| `short` | 0.605 | 0.210 | 0.186 | 1.604 / 0.543 / 11.0 / 14.1 | 0.935 | 0.052 / 0.001 |
| `prose` | 0.587 | 0.254 | 0.159 | 0.707 / 0.349 / 4.2 / 4.5 | 0.844 | 0.070 / 0.011 |
| `python` | 0.787 | 0.148 | 0.064 | 1.116 / 0.251 / 17.0 / 18.3 | 0.898 | 0.410 / 0.015 |
| `agentic` | 0.598 | 0.179 | 0.223 | 0.850 / 0.225 / 7.6 / 9.3 | 0.755 | 0.318 / 0.017 |
| all rows, by tokens | **0.627** | **0.184** | 0.189 | mean of the four: **1.07** (floor 0.21) | 0.790 | |

- **Top-1 agreement between the two models: 62.7 % of positions** (by tokens), against 79.0 % between GLM-5.3's
  two rounding orders. Where they differ, altar-1's choice is inside GLM-5.3's top 5 half the time (18.4 % of all
  positions), and outside it on 18.9 %.
- **KL is 3–31× the order floor in mean and 13–420× in median**: the floor's median is ~0.01 nats (most rows agree
  to the last digits and a few diverge far), altar-1's median is 0.22–0.54 nats, i.e. the distance is spread over
  every row, not a few flips. The cross-order pairs (altar-1 `all-f32` against GLM-5.3 `all`) give the same numbers
  to within 0.15 nats (python 0.97 against 1.12), so the parent's order does not matter at this distance.
- **Against the published fidelity study's 0.506 nats** (KL against BF16 GLM-5.3 on its own 25-prompt panel,
  [scouting.md](scouting.md)): our medians (0.22–0.54) bracket it; our means (0.71–1.60) are higher, carried by
  rows of recited text (`short`'s p99 11 nats: GLM-5.3 is certain of the next word of *Three Men in a Boat*,
  altar-1 is not). Different panels, prompts as opposed to generations, and a mean over teacher-forced prompt
  positions: a sanity anchor, not a bar (DESIGN decision 2).
- On its own targets altar-1 loses little on code and the agentic transcript (top-1 −0.9 and −1.4 points, mean
  log p +0.01 and −0.10) and most on memorised prose (top-1 −27.9 points on `short`, −8.9 on `prose`; § 6.1); the
  document diagnostic (§ 4.5) shows memorisation of a code file is mostly kept (0.971 against 0.984).
- **In transformers' order (A1d)**, both models in `all`, with altar-1's own two-order floor beside GLM-5.3's:

| Sequence | altar-1 `all` vs GLM-5.3 `all`: argmax equal | differs, in GLM top-5 | differs, outside | KL mean / median / p99 / max | altar-1 floor (`all` vs `all-f32`): argmax / KL mean / median | GLM-5.3 floor: argmax / KL mean / median |
|---|---:|---:|---:|---|---|---|
| `short` | 0.605 | 0.203 | 0.192 | 1.615 / 0.522 / 10.8 / 12.2 | 0.869 / 0.098 / 0.029 | 0.935 / 0.052 / 0.001 |
| `prose` | 0.591 | 0.257 | 0.153 | 0.807 / 0.415 / 3.8 / 3.9 | 0.870 / 0.048 / 0.013 | 0.844 / 0.070 / 0.011 |
| `python` | 0.785 | 0.149 | 0.066 | 0.918 / 0.133 / 8.3 / 16.0 | 0.915 / 0.150 / 0.011 | 0.898 / 0.410 / 0.015 |
| `agentic` | 0.597 | 0.178 | 0.225 | 0.785 / 0.228 / 7.7 / 12.4 | 0.770 / 0.297 / 0.019 | 0.755 / 0.318 / 0.017 |
| all rows, by tokens | **0.627** | **0.184** | 0.190 | mean of the four **1.03** | **0.807** / 0.148 | 0.790 / 0.213 |

  The same picture as in the contract's order: 62.7 % top-1 agreement in both orders (0.6267 / 0.6272), KL means
  1.03 against 1.07, medians 0.13–0.52 nats against the floors' 0.001–0.029. altar-1's own floor is of the parent's
  size (0.807 against 0.790 argmax by tokens; KL mean 0.148 against 0.213), so the distance between the models is
  2–31× either floor in KL mean and 9–400× in median (12–33× against altar-1's own floor). The cross-order pairs
  (altar-1 `all` against GLM-5.3 `all-f32`, and the reverse: KL means 1.60–1.62 / 0.75–0.77 / 0.97–1.08 /
  0.77–0.85) lie within 0.02 nats of the range of the same-order pairs.

## 8. The golden files

On rank 2 under `~/altar1-ref/golden/<run>/`, each with `manifest.json` (sequences, offsets, the order, the file
list); the formats are glm53's (glm53 § 7, `glmref.py` `FILES`), with 168 experts in `route_idx` (0..167) and
`calib.npz`. The per-layer dumps are every layer's hidden state at the sample rows, so layers 0, 2, 3, 4, 6, 76, 77
and the MTP layer 78 on `short` are every position (`short` is under 1,024 tokens).

| Run | Order | Sequences | Size | Hashes (sha256) |
|---|---|---|---|---|
| **`all-f32`** (the level-2 golden set) | `--combine f32` (contract, prefill) | short, prose, python, agentic (37,993 rows; 634 sample rows) | 6.0 GB: `layers/` 4.4 GB, `final/` 381 MB, `mtp/` 379 MB, `state/` 891 MB (resume; `hfinal.pt` for `gold_prep.py`) | `SHA256SUMS` (615 files, outside `state/`) `d5911e345f7f7889…`; `final/argmax.npy` `1cf67680d1f7dfe5…`; `mtp/argmax.npy` `32e48b0c5faaff89…` |
| `all` | `--combine bf16` (transformers) | the same | 6.0 GB (6,384,595,339 bytes incl. `SHA256SUMS`): `layers/` 4.4 GB, `final/` 381 MB, `mtp/` 379 MB, `state/` 891 MB | `SHA256SUMS` (615 files, outside `state/`, written 17:07:55) `06eeff766fb42bbf…`; `final/argmax.npy` `5ff1d5e42a59b7a5…`; `mtp/argmax.npy` `61af0ab519c26694…`; `summary.json` `045568089b2565b4…` |
| `mtp-nvfp4` | the MTP layer only, `all-f32`'s inputs, routed experts NVFP4 round-to-nearest (glm53 `nvfp4_rtn`) | the same | 524 MB own (`layers/L78` 146 MB, `mtp/` 379 MB); `final/` is a symlink to `all-f32/final` | `mtp/argmax.npy` `3b9514563d823d55…` |
| `a1a-bf16`, `a1a-f32` | bf16 / f32 | short + prose2400 (2,691 rows, **every position saved**: the layer checks' inputs) | 8.6 GB each | |
| `doc` | bf16, no MTP | doc, docchat | 0.9 GB | |
| `check_layer_L{00,03,04,77}.json`, `router_check.json`, `level3.json` | | | | |

`gold_prep.py`'s raw copies of `all-f32` for the engine's `mtpgold` and `spec` modes are in `~/altar1-ref/gold/`
(`all-f32.{hfinal,l78}.bf16`, `.tokens.i32`, `.argmax.i32`, `.mtp_argmax.i32`, `.seqs.txt`, 0.93 GB). For the engine's
G8-style comparison (`compare_golden.py g8 ROOT ...`) ROOT is `~/altar1-ref/golden` (or a copy), with `--runs
all-f32,all`.

**The NVFP4 drafter (`mtp-nvfp4`, DESIGN decision 5's lever).** Layer 78's routed experts rounded to NVFP4 (glm53's
`nvfp4_rtn`), everything else and the inputs `all-f32`'s (`summarize.py` 17:07:48,
`~/altar1-ref/logs/a1d-summarize-mtp-nvfp4.txt`; `~/altar1-ref/logs/a1d-mtp.txt` 17:08): MTP == main argmax at t+1
0.664 / 0.522 / 0.698 / 0.437, **0.4915 by tokens** (18,670 / 37,985) against BF16 experts' 0.4920 (18,690): 20
fewer accepted drafts in 37,985. Its drafts equal the BF16 drafter's on 97.2 / 97.7 / 98.7 / 97.3 % of rows. The
NVFP4 drafter keeps the acceptance to 0.05 points; it changes the drafts, never the main model's tokens.

## 9. Open questions

1. ~~`all` and `mtp-nvfp4`~~: done 17:01:41 and 17:07:31 (A1d): the four-sequence order variance (§ 6.2) and the
   NVFP4 drafter's acceptance (§ 8, 0.4915 against 0.4920).
2. **The independent stack is transformers' modules on single layers**, not a whole model: no released transformers
   loads altar-1 (model.md § 4.4) and vLLM was not run. The document diagnostic is the end-to-end evidence.
3. **Router margins** at layers 3–5 (77–96 % of rows under 1e-3): as glm53 § 8.4, the engine's tolerance must come
   from order-variance runs.
4. **KL against the parent is far above the parent's order floor** on every sequence (§ 7); whether that matters
   for the agent workload is a quality question for the user, outside the exactness standard.
5. MTP acceptance on generated text is unmeasured, as in glm53.

## 10. How to resume

All on rank 2, from `~/altar1-ref/code/altar1-4xgx10-switchless/ref`, every process at `oom_score_adj` 1000, cores
10–19, nice 19, idle I/O, 8 threads; keep MemAvailable ≥ 8 GiB (the engine's ring jobs load ~77 GiB there):

1. **The pass chain is finished** (`passes.sh`, log `~/altar1-ref/logs/passes.log`): `a1a-bf16`, `a1a-f32`,
   `all-f32`, `all` (exit 0 at 17:01:41) and `mtp-nvfp4` (exit 0 at 17:07:31), `passes done` at 17:07:31. Rerunning
   `passes.sh` skips every finished pass; to redo one, move its directory aside first:
   `OMP_NUM_THREADS=8 setsid nohup taskset -c 10-19 nice -n 19 ionice -c3 ./passes.sh >> ~/altar1-ref/logs/passes.log 2>&1 < /dev/null &`
2. **The analyses of A1d are done** (17:07–17:08, all numpy, ≤ 1.6 GiB RSS, under `ulimit -v` 8 GiB, nice 19, cores
   10–19; driver `~/altar1-ref/logs/a1d.sh`, outputs `~/altar1-ref/logs/a1d-*.txt`). To redo them:
   - `python -I summarize.py ~/altar1-ref/golden/all --tokenizer ~/altar-1/tokenizer.json --json ~/altar1-ref/golden/all/summary.json`
     (`summary.json` is in `all`'s `SHA256SUMS`, as in `all-f32`'s);
   - `python -I cmp_runs.py ~/altar1-ref/golden/all ~/altar1-ref/golden/all-f32 --seq S` and the reverse, for S in
     short, prose, python, agentic (§ 6.2);
   - `python -I level3.py --altar ~/altar1-ref/golden --glm ~/glm53-ref/golden --json ~/altar1-ref/golden/level3.json` (§ 7);
   - `python -I summarize.py ~/altar1-ref/golden/mtp-nvfp4` (its `final/` links to `all-f32`'s; the JSON is kept
     outside the golden directory, `~/altar1-ref/logs/a1d-summary-mtp-nvfp4.json`) and `python -I mtp_accept.py
     ~/altar1-ref/golden` (acceptance by tokens and the drafts against `all-f32`'s);
   - verify a copy: `cd golden/all && sha256sum -c SHA256SUMS` (and the same in `all-f32`).
3. The layer checks and the router check are done (`check_layer_L*.json`, `router_check.json`); rerun them with the
   commands of § 3 if the code changes.
