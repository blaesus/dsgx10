# The CPU reference (experiment 2)

A layer-streaming torch reference of GLM-5.3's math ([model.md](model.md)) under the numerics contract
([quantisation.md § 5](quantisation.md#5-exactness-for-this-configuration), [DESIGN decision 3](../DESIGN.md#decisions)),
run teacher-forced on NVIDIA's NVFP4 checkpoint on rank 2's CPU. It produces the first golden data (milestones
G1a and G1b), the routing statistics behind the hybrid-bits question
([quantisation.md § 3.4](quantisation.md#34-c-hybrids-by-expert-usage)), the per-expert calibration sums, and
the MTP layer's teacher-forced acceptance. Written 2026-10-13. Everything here was measured on that date
unless marked *estimate*.

## Contents

1. [Files and how to run it](#1-files-and-how-to-run-it)
2. [The contract as implemented](#2-the-contract-as-implemented)
3. [Checks](#3-checks)
4. [The text](#4-the-text)
5. [Timing on rank 2](#5-timing-on-rank-2)
6. [Results](#6-results)
7. [The golden files](#7-the-golden-files)
8. [Open questions](#8-open-questions)
9. [Record](#9-record)
10. [Follow-ups (2026-10-14)](#10-follow-ups-2026-10-14)

## 1. Files and how to run it

| File | What |
|---|---|
| [`ref/glmref.py`](../ref/glmref.py) | the reference: reads one layer's tensors at a time with `pread` (no mmap; page cache dropped after each read), runs all sequences through it, saves the golden tensors, checkpoints the residual stream after every layer |
| [`ref/make_tokens.py`](../ref/make_tokens.py) | renders the conversations with the checkpoint's `chat_template.jinja` (jinja2, transformers' settings) and encodes them with `tokenizer.json` (`tokenizers`); `--check-transformers` compares with `AutoTokenizer.apply_chat_template` |
| [`ref/check_layer.py`](../ref/check_layer.py) | one decoder layer of `glmref.py` against transformers 5.15's `GlmMoeDsaDecoderLayer` on random weights |
| [`ref/summarize.py`](../ref/summarize.py) | top-1 and log-probability per sequence, expert-usage skew, MTP acceptance, index-set ties, timing |
| [`ref/cmp_runs.py`](../ref/cmp_runs.py) | two runs on a shared sequence: argmax agreement, \|Δ log p\|, KL on the sampled full logits, per-layer routing and hidden-state agreement |
| [`ref/check_nvfp4_rtn.py`](../ref/check_nvfp4_rtn.py) | `quant/formats.py`'s `nvfp4_rtn` on the BF16 master's experts against NVIDIA's NVFP4 bytes (codes, block scales, `weight_scale_2`) |
| [`ref/qprep.py`](../ref/qprep.py) | repacks the master's experts into the quantiser's formats once (S3×2, trellis K3/K4) into a cache the tracks read, and checks every packed tensor's SHA-256 against the quantiser's recorded hashes (§ 10.3) |
| `glmref.py` modes (2026-10-14) | `--mtp-from RUN --mtp-experts {bf16,nvfp4_rtn}`: rerun only the MTP layer on a finished pass's final hidden states (experiment 5); `--track FORMAT@LAYERS`, `--track ne:{fp8,o4,oqk4,oqks4}`, `--track-ref TRACK`, `--master DIR`, `--qcache DIR`, `--sub-experts LIST`: carry several precision tracks through one resumable, disk-backed pass and compare their logits (experiment 3, § 10.3–10.4); `--cov-out DIR --cov-layers L…`: per-expert input covariances (experiment 3b, § 10.6) |
| [`ref/text/`](../ref/text) | the two public-domain prose excerpts |
| [`ref/tokens/`](../ref/tokens) | the four sequences, and the two diagnostic ones (`--diagnostic`, § 6.3): conversation, rendered text, token ids, `SHA256SUMS` |

Rank 2 (the box with the disk), as done on 2026-10-13:

```
# from rank 0: the checkpoint over the ring, rate-limited so production's collectives are not disturbed
rsync -rtL --partial --bwlimit=600000 <hf cache>/models--nvidia--GLM-5.3-NVFP4/snapshots/e3b8f9f2…/ rank2:~/hf/glm53-nvfp4/

# on rank 2: a CPU-only venv
python3 -m venv ~/glm53-ref/venv
~/glm53-ref/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # 2.14.1+cpu, aarch64
~/glm53-ref/venv/bin/pip install safetensors tokenizers numpy jinja2 transformers==5.15.0  # transformers for check_layer.py only

# tokens (light; done on rank 0 with transformers' check)
# (--rename maps the four boxes' host names, not kept in the repo, to "rank 0" ... "rank 3")
python3 ref/make_tokens.py --ckpt CKPT --ds-transcript ds41f-4xgx10-switchless/golden/long.messages.json \
    --out-dir ref/tokens --diagnostic --check-transformers --rename NAME=REPLACEMENT ...

# a pass, detached, pinned to cores 10-19, at most 6 threads, nice 19 (resumable: rerun the same command
# after a preemption). See "Pinning" below.
cd ref && OMP_NUM_THREADS=6 setsid nohup taskset -c 10-19 nice -n 19 ionice -c3 ~/glm53-ref/venv/bin/python glmref.py \
    --threads 6 --ckpt ~/hf/glm53-nvfp4 --out-dir ~/glm53-ref/golden/all-f32 \
    --tokens tokens/short.tokens.json tokens/prose.tokens.json tokens/python.tokens.json tokens/agentic.tokens.json \
    > ~/glm53-ref/logs/all-f32.log 2>&1 < /dev/null &
~/glm53-ref/venv/bin/python summarize.py ~/glm53-ref/golden/all-f32 --tokenizer ~/hf/glm53-nvfp4/tokenizer.json
```

**Pinning (from 2026-10-14 01:10).** Unpinned at 8 threads, glmref.py ran at ~600 % CPU on cores 5–9. The
decode tuner measured that this slows rank 2's GPU kernels by 8–16 % and costs the ring 2–4 ms a decode step:
cores 5–8 are the ring proxies' cores, and the memory is shared with the GPU. Every run on rank 2 now goes under
`taskset -c 10-19`, with at most 6 threads, at nice 19. The `all-f32` pass was started unpinned on 8 threads
and re-pinned while running (`taskset -acp 10-19` on all 26 threads of the process, at layer 15, 01:07); it kept
its 8 threads. The `all`, `short*` and `doc` runs before it were unpinned on 8 threads. Pinning and thread count
change no output bit (§ 10.1).

`--layers N` stops after N layers (a wiring test: 5 layers of the short sequence take 17 s). The copy took
~22 min (433 GB, 59 files; sizes verified file by file against the source).

## 2. The contract as implemented

`glmref.py`'s docstring is the contract's executable form; each line cites model.md. The choices model.md
left open, and what this reference does:

| Question | Choice | Why |
|---|---|---|
| NVFP4 dequantisation | `W' = E2M1(code) · E4M3(block scale)`, exact in bf16 and f32 (≤ 6 significant bits); the per-tensor `weight_scale_2` multiplies the **f32 GEMM output** before its one rounding: `y = bf16(ws2 · Σ x·W')` | dequantising `W' · ws2` to bf16 would round every weight; this is the formula without a rounding, and what the engine's block-scaled MMA computes (f32 accumulate, alpha after) |
| FP8 dense MLPs (layers 0–2) | the same with their per-tensor `weight_scale` | as above |
| Activation quantisation | none (`input_scale` is ignored) | the contract (quantisation.md § 5, level 2); NVIDIA's FP4 activations are a later track |
| GEMMs on bf16 weights | torch's bf16 GEMM (f32 accumulation, one output rounding; checked: rounding equals exact rounding as often as an f32 GEMM's) | |
| GEMMs on scaled weights | an f32 GEMM of bf16-exact operands (exact products, f32 accumulation) | torch 2.14 has no CPU bf16 GEMM with f32 output (`mm.dtype` is CUDA-only); costs ~0.7 against ~1.8 TFLOPS on 8 cores |
| Router top-8 order | `topk(sorted=True)`: by descending biased score | the reference's `sorted=False` leaves it unspecified |
| **Sublayer outputs (`--combine f32`: the contract, the default since 2026-10-13 23:20)** | o_proj, the dense MLP's down projection, every routed expert's down projection (× its f32 weight) and the shared expert's down projection stay **f32**; they are summed in f32 (routed experts in expert-id order, then the shared expert) and rounded **once**, as `bf16(f32(h) + Σ)`, when added to the residual | the TP=4 engine holds only f32 partial sums of each of these and all-reduces them before any rounding, so it cannot round per GEMM as transformers does (a contract item from the expert-kernel builder, 2026-10-13). Effect against transformers' order: § 6.4 |
| `--combine bf16` (transformers' order; used by the `all` pass and the first `short` pass) | each of those GEMMs rounded to bf16; the routed experts weighted and summed in f32 in expert-id order, one rounding; the shared expert and the residual added in bf16 | transformers' default `grouped_mm` sums the 8 slots in f32 in unspecified slot order: the difference from it is f32 summation order only (§ 3) |
| Attention | transformers' expanded form with eager numerics (bf16 QKᵀ, × 1/16, mask, f32 softmax rounded to bf16, bf16 PV), in query blocks of 256 over the causal keys only | identical to eager: masked keys get probability exactly 0 either way |
| Indexer top-k ties | `torch.topk` (unspecified among ties); sets saved ascending; the gap between the 2048th and 2049th score saved per row | so a consumer can tell a tie from a genuine choice |
| Logits | f32 (bf16 hidden × bf16 head, f32 accumulation, no output rounding) | transformers emits bf16 logits; `bf16(ours)` equals them up to summation order. f32 keeps log-probabilities usable (bf16 logits are spaced 0.125 at 16–32) |
| MTP position and input | row t: `eh_proj(cat(enorm(embed(token t+1)), hnorm(h_t)))` at **position t**, `h_t` after `model.norm`, the embedding zeroed at position 0; its own indexer; BF16 experts | vLLM's `DeepSeekMTP` and EAGLE proposer (model.md § 3, § 8) |
| Calibration sums | per expert: count, routing mass, Σ x² over the rows routed to it for the gate/up input (6144) and the down input (2048), f32 | the diagonal of XᵀX, an imatrix in llama.cpp's sense |

## 3. Checks

- **Tokens.** For all four sequences, `make_tokens.py`'s rendering equals transformers 5.15's
  `apply_chat_template` text exactly and its ids equal `AutoTokenizer`'s, and every sequence decodes back to its
  text (rank 0, 2026-10-13).
- **One layer against transformers** (`check_layer.py`, rank 2): layer 6's configuration (a full-indexer MoE
  layer) at the checkpoint's dimensions with 16 routed experts, random N(0, 0.02) weights, router bias ~34.4,
  2,400 rows (352 of them sparse), transformers in eager attention with `grouped_mm` experts:

  | | result |
  |---|---|
  | index sets, rows 2048–2399 | equal on 352/352; no ties at the boundary |
  | attention sublayer output | bit-equal on 100.0 % of values (rel-l2 0 in one run, 5.3e-5 in another: see § 8) |
  | routing | the same 8 experts on every row; weights within 1.2e-7 |
  | whole layer output | bit-equal on 99.67 %, rel-l2 3.1e-4, max \|diff\| 0.0625 (the f32 summation order of the 8 experts and GEMM blocking) |

  With transformers' fallback experts loop instead (it rounds each weighted expert output to bf16 and sums in
  bf16, model.md § 2.7) only 64.6 % is bit-equal (rel-l2 3.6e-3): that path is not the reference.
- **End to end on real weights**: the short sequence's teacher-forced argmax reads right (§ 6.1).

## 4. The text

37,993 tokens in four sequences, each a GLM-5.3 conversation rendered by the checkpoint's template (default
`Reasoning Effort: Max`, past reasoning kept) and ending with the generation prompt `<|assistant|><think>`.

| Name | Tokens | What | sha256 of the ids (`json.dumps(ids)`) |
|---|---:|---|---|
| `short` | 291 | a user turn quoting the opening of Jerome K. Jerome, *Three Men in a Boat* (1889; Project Gutenberg #308), three paragraphs | `9d0e1ebae45f2177…` |
| `prose` | 4,762 | the opening of chapter I of Charles Darwin, *The Voyage of the Beagle* (Gutenberg #944), ~21 kB, with a summarising request | `67cf8d7c1c3e48d0…` |
| `python` | 6,106 | a system turn and a code-review request over this configuration's `tools/*.py` (written 2026-10-13, so unseen in training) | `751c873356cd7884…` |
| `agentic` | 26,834 | the DeepSeek configuration's golden agentic transcript ([ds41f golden](../../ds41f-4xgx10-switchless/golden/README.md): 4 tools, 3 tasks, 11 tool calls with real results, reasoning in every assistant turn) re-encoded as a GLM conversation: tool-call arguments parsed to mappings, the worktree path replaced by `/workspace/dsgb10`, box host names replaced by ranks | `4094c2fd4a9b3a13…` |

Full hashes of every file in [`ref/tokens/SHA256SUMS`](../ref/tokens/SHA256SUMS). The prose excerpts are
paragraph-joined (Gutenberg's hard line breaks removed); the files are in [`ref/text/`](../ref/text).

## 5. Timing on rank 2

Rank 2 (DGX Spark, Cortex-X925/A725, 121 GiB), production's rank running beside it; 8 threads at nice 19,
`ionice -c3`. Measured 2026-10-13.

| Pass | Rows | Per layer | Whole pass | Peak RSS |
|---|---:|---|---|---|
| `short` | 291 | 0.8–1.0 s dense, 5–6 s MoE (reading and dequantising 4.8 GB of NVFP4 experts) | 447 s: 78 layers 408 s, head 10 s, MTP 28 s | under 1 GiB in the layers |
| `short`, `--combine f32` | 291 | 6–10 s MoE (the f32 down projections) | 517 s | |
| `doc` + `docchat` (§ 6.3) | 4,248 | ~10 s MoE | ~13 min (no MTP) | 1.0 GiB in the layers |
| `all` (4 sequences, `--combine bf16`) | 37,993 | dense layers 71–78 s (attention 50–55 s); MoE layers 75–94 s: attention 42–46 s on shared layers, 52–56 s on full-indexer layers, experts ~35 s | **6,739 s (1 h 52 min)**: 78 layers 6,433 s (median 80.9 s), head 106 s, MTP 175 s (its layer 79 s) | 10.9 GiB (`VmHWM`, the head stage's f32 head); 3.7–4.4 GiB in the layers |

CPU rates on this box (8 threads, 2026-10-13): bf16 `linear` 1.5–2.2 TFLOPS, f32 0.65–0.82, a bf16 batched
attention matmul 1.5, the indexer's f32 matmul 1.2; one NVFP4 matrix dequantised in 5 ms.

## 6. Results

### 6.1 The first check: `short`

Teacher-forced over 291 tokens: **top-1 agreement 0.797** of the argmax with the next token, mean log-probability
−1.259 (perplexity 3.52). From position 145 the actual text and the argmax:

```
actual : ' _he_ had fits of giddiness too, and hardly knew what _he_ was doing. With me, it was my liver that was out of order. I knew it was my liver'
argmax : ' HEhe_ felt fits of giddiness come, nearly that knew what _he_ was doing there Harris me it it is my liver. was the of order. I knew it was my liver'
```

(each argmax token is a prediction from the true prefix, so the line reads as aligned next-word guesses: right
on the repeated phrases, plausibly wrong on the new content). MTP, teacher-forced: its argmax equals the actual
token t+2 at 0.737 and the main model's own argmax at t+1 at 0.785.

### 6.2 The full pass (`all`, `--combine bf16`)

Teacher-forced: the main model's argmax against the next token, and the mean log-probability of the next token.

| Sequence | Tokens | Top-1 | Mean log p | Perplexity | Top-1 by 1,024-token window |
|---|---:|---:|---:|---:|---|
| `short` | 291 | 0.793 | −1.273 | 3.57 | 0.79 |
| `prose` | 4,762 | 0.501 | −2.690 | 14.7 | 0.56 0.46 0.50 0.47 0.53 |
| `python` | 6,106 | 0.689 | −1.606 | 4.98 | 0.70 0.66 0.70 0.72 0.66 0.71 |
| `agentic` | 26,834 | 0.482 | −3.495 | 33.0 | 0.42 0.42 0.36 0.40 0.59 0.62 0.38 0.65 0.41 0.58 0.54 0.55 0.55 0.57 0.64 0.46 0.40 0.31 0.52 0.46 0.40 0.46 0.52 0.46 0.46 0.41 0.51 |

There is no step at position 2048, where queries go sparse. In 256-token windows from 1,024 to 4,608, python
stays at 0.64–0.79 except one window, and prose at 0.42–0.56 on both sides. The agentic transcript scores low
*inside its tool results*. From position ~1,100, in `json/__init__.py`'s docstring, the argmax stops continuing
the file and emits the conversation's topic words (`Infinity`, `insertion order`, `Decimal`, `float`), down to
0.16 top-1 and −7.2 mean log p in the window at 2,560. The DeepSeek golden data showed the same symptom on the
same transcript ([ds41f golden](../../ds41f-4xgx10-switchless/golden/README.md)), and there it was pinned on
Engram, which GLM-5.3 does not have. § 6.3 shows it is the model, not the reference.

### 6.3 The diagnostic: the same file as a document

`make_tokens.py --diagnostic` adds two sequences built from the transcript's first file result
(`json/__init__.py`, its first 8,000 characters): `doc`, the text as a plain document after `[gMASK]<sop>`
(2,114 tokens), and `docchat`, the same text quoted in one user turn (2,134 tokens). Run with
`--combine bf16 --no-mtp`.

| Sequence | Top-1 | Mean log p | By 1,024-token window |
|---|---:|---:|---|
| `doc` | **0.984** | −0.071 | 0.980 0.989 0.969 (the last window is past position 2048) |
| `docchat` | 0.736 | −1.529 | 0.758 0.728 0.565 |
| the same text inside `agentic` (positions ~700–4,500) | ~0.4 | −2.3 to −7.2 per window | |

As a document the reference reproduces a memorised file almost perfectly, dense and sparse rows alike. So the
weights, the dequantisation, attention, the indexer and the head are right: this is G1a's "greedy text reads
right". The same text costs 1.5 nats a token in a user turn, and more inside a tool result deep in an agentic
conversation: the post-trained model's predictions over quoted content are pulled toward what the assistant
will say about it. That is the model's behaviour, and the golden data reproduce it. Low top-1 over prompt
content is not an engine fault.

### 6.4 The combine order: contract (f32) against transformers (bf16), on `short`

| | `--combine bf16` | `--combine f32` |
|---|---:|---:|
| top-1 against the next token | 0.797 (0.793 in the `all` pass) | 0.807 |
| mean log p | −1.259 (−1.273) | −1.287 |
| MTP draft == token t+2 | 0.737 (0.747) | 0.737 |

Between the two orders (`cmp_runs.py`, first `short` against `short-f32`):

- argmax equal on 274/291 rows (94.2 %); |Δ log p(next)| mean 0.14, max 3.75;
- KL(bf16 ‖ f32) on the full logits: mean 0.047, max 2.06; **max |Δ logit| 7.97**;
- the routed expert sets already differ on 9 % of rows at layer 3, and on 68 % at layer 77;
- the hidden state differs by 0.5 % (relative, median) after layer 3, and by 17 % after layer 77.

Neither order is better on this sample (±1 token of 291). The golden data for the engine's level-2 check must
be made with `--combine f32`, and the `all` pass was not (§ 8).

### 6.5 The reference's own variance

- **Batch invariance and determinism.** `short` inside the 38k batch and `short` run alone afterwards
  (`short-bf16-rerun`) are **bit-identical** in every saved tensor of the sequence. The CPU reference's results
  do not depend on the other rows in the batch (GEMM blocking over M changes nothing here), and they repeat
  exactly.
- **One unexplained outlier.** The very first `short` run (21:28, minutes after the copy finished) agrees with
  both bit for bit through layer 41 and diverges from layer 42 on: route sets equal on 99 % of rows at layer 42
  and 57 % at layer 77; argmax equal 98.6 %, KL mean 0.011, max |Δ logit| 2.85. The copy verifies against the
  revision's LFS SHA-256s for all 52 files (checked afterwards), and layer 42's shard hashes the same on both
  ranks. Unresolved (§ 8). The run is kept as `golden/short`, but the reference sets are `all`,
  `short-bf16-rerun` and `short-f32`.
- **Amplification.** A difference entering at layer 42 grows to 9 % relative in the hidden state by layer 77
  and flips 1.4 % of argmaxes. The router margins show how readily routing flips. The gap between the 8th and
  9th biased scores is under 1e-3 for **24 %** of rows on average over the MoE layers, and for 97 %, 93 % and
  83 % of rows at layers 3, 4 and 5. At layer 3 the median gap is 1.6e-4 and the 1st percentile 3.8e-6, one
  f32 ulp at the bias's magnitude of ~34. At layers 38–40 it is 9 % of rows.
- **Index sets.** The 2048th and 2049th index scores tie exactly on at most 0.35 % of the sparse rows of any
  full layer (layer 1 on `agentic`), and on 0–0.07 % in most layers.

### 6.6 Expert usage

Routing mass (the normalised weights) per expert over the 37,993 rows of `all`, layers 3–77:

| | Range over layers | Typical (layers 23–77) |
|---|---|---|
| experts carrying 50 % of the mass | 41–97 | 65–80 |
| experts carrying 90 % | 169–215 | 181–195 |
| experts carrying 99 % | 235–251 | 236–244 |
| share of the mass on the top 64 (uniform: 0.25) | 0.36–0.61, mean 0.46 | 0.44–0.55 |
| experts never routed to | 0 in every layer | |

- **Where the skew is.** The most concentrated layers are 17–22 (top-64 share 0.49–0.61; 41–57 experts carry
  half the mass). Layers 3–14 and 23–26 are near-flat (0.36–0.43).
- **Sample size.** Per sequence, concentration rises as the sample shrinks (`short`'s 291 rows put half the
  mass on 6–71 experts), so per-domain figures on the small sequences overstate skew.
- **The MTP layer (78)** is flat: 84, 202 and 248 experts for 50/90/99 %, top-64 share 0.41.

**Does the hot set transfer?** On `agentic`, its own top-64 experts carry 0.480 of its mass (mean over layers).
The top-64 chosen from the other three sequences carry **0.345** (minimum 0.280), against 0.25 for any 64.

**Hybrid bits by usage: not worth it as a static allocation.** The overall top-quarter share (0.46) clears
quantisation.md § 3.4's 0.40 threshold. But that threshold assumed a stable hot set, and the hot set is not
stable: a 64-expert set chosen on other text covers 35 % of an agentic workload's routing mass, 10 points above
chance, while 99 % of the mass needs 235 or more experts in every layer. Allocate bits by measured sensitivity
instead (KL per expert group, experiment 3), with the calibration sums (`calib.npz`) as the imatrix.

### 6.7 MTP, teacher-forced

Row t of the MTP layer sees the main model's post-norm hidden state of row t and token t+1, and drafts token
t+2 (`all` pass):

| Sequence | Draft == token t+2 | Draft == main argmax at t+1 | Draft == token, where the main model is right |
|---|---:|---:|---:|
| `short` | 0.747 | 0.785 | 0.887 |
| `prose` | 0.483 | 0.581 | 0.791 |
| `python` | 0.650 | 0.679 | 0.824 |
| `agentic` | 0.479 | 0.449 | 0.715 |
| all, weighted by tokens | | 0.505 | |

The middle column is the one-draft greedy acceptance a verifier would see if the text were the model's own
output. Over prompt content (these sequences are prompts, not generations) it is 0.45–0.79. On text the model
generates itself it should be higher (*not measured*: that needs generated continuations).

## 7. The golden files

On rank 2 under `~/glm53-ref/golden/<run>/` (`all-f32`, `all`, `short*`), each with `manifest.json` (sequences and
offsets, the file list with shapes and meanings) and `SHA256SUMS` over every file outside `state/`. Rows are
all sequences concatenated; `P` is the sample rows (`sample_rows.npy`: every position of a sequence under 1,024
tokens; otherwise every 256th, the first and last 32, and 2047–2049).

| File | Shape | Content |
|---|---|---|
| `tokens.npy`, `sample_rows.npy` | [N], [P] | ids; sampled rows |
| `embed_sample.npy` | [P, 6144] | the embedding rows (bf16 values as f32) |
| `layers/LNN/hidden.npy` | [P, 6144] | the residual stream after layer NN (bf16 values as f32) |
| `layers/LNN/topk_SEQ.npy`, `topk_margin_SEQ.npy` | [n−2048, 2048] uint16, [n−2048] f32 | full-indexer layers: the key positions chosen for positions 2048…n−1, ascending; the 2048th score minus the 2049th |
| `layers/LNN/route_idx.npy`, `route_w.npy`, `route_margin.npy` | [N, 8] uint8, [N, 8] f32, [N] f32 | MoE layers: experts by descending biased score, their weights (× 2.5), the 8th minus the 9th biased score |
| `layers/LNN/calib.npz` | count [256], mass [256], `gate_up_in_sqsum` [256, 6144], `down_in_sqsum` [256, 2048] | the calibration sums |
| `final_hidden_sample.npy` | [P, 6144] | after `model.norm` |
| `final/top20_ids.npy`, `top20_logprobs.npy` | [N, 20] | from the f32 logits, every row |
| `final/lse.npy`, `next_logprob.npy`, `argmax.npy` | [N] | logsumexp, log p(actual next token) (NaN on a sequence's last row), argmax |
| `final/logits.npy`, `logits_rows.npy` | [P, 154880] f32 | full logits at the sample rows |
| `layers/L78/*`, `mtp/*` | over the MTP rows | the MTP layer and its draft logits (row t: hidden t and token t+1 in, token t+2 out; `rows_main.npy` maps them to main rows) |
| `state/` | | the resume checkpoint (residual stream, final hidden); not golden |

Hashes (sha256):

| Run | `SHA256SUMS` | `final/argmax.npy` | `mtp/argmax.npy` | size |
|---|---|---|---|---|
| `all-f32` (`--combine f32`, the level-2 golden set, § 10.2) | `6c49dd8d37340e46…` (615 files, `summary.json` included) | `2f7e85cac6b2e082…` | `35f0e80e1a7ed06c…` | 5.3 GB (+ 0.9 GB `state/`) |
| `all` (`--combine bf16`) | `e00d675bb0977b0b…` (615 files, `summary.json` included) | `2d1b224461894d78…` | `a9c633da1d6ddbaf…` | 5.3 GB (+ 0.9 GB `state/`) |
| `short-f32` (`--combine f32`, the contract) | `bbe5659360b937d9…` (482 files) | `f601866e90b71abb…` | `204e542354ad21e2…` | 1.5 GB |
| `short-bf16-rerun` | `8a3a458415675ea4…` (482 files) | `412635696e9385ec…` | `c1fecf73db931f3e…` | 1.5 GB |
| `short` (the outlier of § 6.5) | `51e4308e6d7a6138…` (482 files) | `b192f3ba1141d96c…` | `dae230453565701c…` | 1.5 GB |
| `doc` (`doc`, `docchat`; no MTP; `--combine bf16`) | not hashed | | | |

## 8. Open questions

1. ~~**The 38k golden set is in transformers' rounding order.**~~ Resolved 2026-10-14: `all-f32` (§ 10.2) is
   the contract-order golden set (12,733 s, not the ~2 h 10 min estimated: the pass was re-pinned to fewer, slower
   cores mid-run).
2. **The first `short` run's divergence at layer 42** (§ 6.5): seven later passes over those rows agree bit for
   bit, and no op in the reference varies with threads, cores or the deterministic flag (§ 10.1). Most likely a
   transient bad read just after the copy; it cannot be proven now.
3. **An independent stack** (the bar for G1a and G1b): the reference's logits have not been compared with vLLM
   or SGLang serving this checkpoint (prompt logprobs on `ref/tokens`), nor with logits from the BF16 master. The
   document test (§ 6.3) is the evidence so far.
4. **Router margins** (§ 6.5): at layers 3–5 most rows choose their 8th expert by less than 1e-3, and 1 % by
   one f32 ulp. So the engine's router (f32 GEMM order, sigmoid, bias add) will flip routes against golden, and
   its tolerance has to be set from runs like § 6.4, not from a per-layer bit match.
5. **MTP on generated text** is unmeasured. So are its input conventions (position t, the embedding zeroed at
   position 0), which are taken from vLLM: the alternative (position t+1) is untried.
6. **KV format and indexer key precision** are not tracks yet (experiment 4). This reference keeps K/V in bf16
   and the indexer in bf16/f32.
7. **Prefill only**: one forward from position 0 per sequence. Nothing here checks chunked prefill or the
   decode order.

`glmref.py --combine` (default `f32`) implements the TP=4 order. `check_layer.py` pins `bf16` to compare with
transformers. A resumed pass refuses to switch orders: `progress.json` records the order, and progress files
from before the option count as `bf16`.

## 9. Record

- 2026-10-13: checkpoint copied to rank 2 (433 GB, sizes verified), venv made, tokens built and checked against
  transformers, `check_layer.py` against transformers 5.15, the `short` pass (447 s), the `all` pass 21:36–23:28
  (6,739 s).
- 2026-10-13, 23:20: the f32 combine order became a contract item; `--combine` added; `short-f32` (517 s) and
  `short-bf16-rerun` run.
- 2026-10-13/14: the document diagnostic (`doc`, `docchat`); the copy verified against the revision's LFS
  SHA-256s (52 of 52 files). No GPU was used anywhere; rank 0 ran only tokenising, header reads and the API
  lookup.
- 2026-10-14: determinism checks (§ 10.1); the `all-f32` pass 00:36–04:09 (12,733 s), re-pinned to cores
  10–19 at layer 15; from then on every run pinned to 10–19, 6 threads, nice 19. The MTP NVFP4 replay (§ 10.5),
  the full `nvfp4_rtn` byte check (all 256 experts of layers 3 and 40 equal), and the S3×2 and trellis repack
  (every hash equal to the quantiser's). The 8-track pass (04:18) was killed by earlyoom at layer 55 during a
  GPU job's weight load (08:28); tracks mode made resumable and disk-backed. The covariance pass was started
  (09:00) and stopped after layer 2 to make room for a GPU job; the repack was stopped at 09:30 (§ 10.8). Work
  paused at the user's request; nothing of this experiment runs on rank 2.

## 10. Follow-ups (2026-10-14)

### 10.1 Determinism

- **Three more `short` passes in the contract order** (`short-f32-2`, `short-f32-3`, and `short-f32-det` with
  `torch.use_deterministic_algorithms(True)`) are byte-identical to `short-f32` in every `.npy`/`.npz` file.
  The deterministic flag is now always on in `glmref.py` and changed no byte.
- **The ops, one by one** (a scratch test on rank 2: bf16 and f32 `F.linear` at 7, 300 and 4,096 rows, bf16
  `matmul`, f32 softmax, `index_add_`, f32 sums and means, `topk`): bit-identical across 1, 2, 4 and 8 threads,
  and across the Cortex-X925 cores (5–9, 15–19) and the Cortex-A725 cores (0–4, 10–14).
- **The `all-f32` pass** changed thread placement mid-run (unpinned, then pinned to 10–19 at layer 15, § 1), and
  its `short` rows are still bit-identical to `short-f32`'s in every saved tensor (`cmp_runs.py`: argmax, log p,
  full logits, routes and hidden states all equal).

So nothing in the reference is nondeterministic on this CPU: not thread count, core type, batch, or the
deterministic flag. The first `short` run's divergence at layer 42 (§ 6.5) remains the only one seen in eight
passes over those rows. A transient bad read of layer 42's weights, minutes after the copy finished, is the
likely cause. Not proven: nothing from that read was kept.

### 10.2 The contract-order golden set (`all-f32`)

`--combine f32`, the four sequences (37,993 rows), 00:36–04:09: **12,733 s**. The layers took a median of 162 s
(~96 s a layer before the re-pinning at layer 15, 140–215 s after). The head took 129 s, the MTP layer 139 s,
and the MTP stage 284 s. Peak logged RSS 5.8 GiB. This is now **the level-2 golden set**; `all` stays as
the transformers-order record.

| Run | `SHA256SUMS` | `final/argmax.npy` | `mtp/argmax.npy` | size |
|---|---|---|---|---|
| `all-f32` | `6c49dd8d37340e46…` (615 files, `summary.json` included) | `2f7e85cac6b2e082…` | `35f0e80e1a7ed06c…` | 5.3 GB (+ 0.9 GB `state/`) |

| Sequence | top-1 (`all-f32`) | top-1 (`all`, bf16) | mean log p (f32) | MTP draft == token t+2 | MTP draft == main argmax at t+1 | where the main model is right |
|---|---:|---:|---:|---:|---:|---:|
| `short` | 0.807 | 0.793 | −1.287 | 0.737 | 0.796 | 0.872 |
| `prose` | 0.511 | 0.501 | −2.616 | 0.489 | 0.587 | 0.790 |
| `python` | 0.695 | 0.689 | −1.587 | 0.654 | 0.688 | 0.829 |
| `agentic` | 0.488 | 0.482 | −3.436 | 0.484 | 0.454 | 0.719 |
| all, weighted by tokens | | | | | 0.511 | |

**The two orders over the whole set** (`cmp_runs.py all all-f32`):

| Sequence | argmax equal | KL(bf16 ‖ f32) at the sample rows, mean / max | max \|Δ logit\| | \|Δ log p(next)\| mean |
|---|---:|---|---:|---:|
| `short` | 0.935 | 0.052 / 2.0 | 8.0 | 0.17 |
| `prose` | 0.844 | 0.070 / 0.74 | 5.7 | 0.40 |
| `python` | 0.898 | 0.41 / 8.3 | 13.1 | 0.26 |
| `agentic` | 0.755 | 0.32 / 12.3 | 14.8 | 0.61 |

On long sequences the rounding order alone moves a quarter of `agentic`'s argmaxes. The f32 order scores a
little higher on all four sequences (0.6–1.4 points of top-1), which is suggestive but within what one order
change can move. The expert-usage figures of § 6.6 hold in the f32 order (top-64 share mean 0.461, range
0.364–0.608; agentic's own top-64 0.480, the others' 0.346).

**The engine's tolerance.** These are two correct implementations of the same model. An engine in the contract
order should match `all-f32` far more tightly than this, but the table is the scale of what any order
difference produces.

### 10.3 Precision tracks (experiment 3): the method, the anchors, and what exists

**Status (2026-10-14, paused).** No full-depth track result exists yet. The first 8-track pass (04:18–08:28, short
and python) was killed at layer 55 of 78 by earlyoom (`--prefer python`), when a GPU job loaded the 101 GiB rank
files on rank 2 and available memory dipped. Tracks mode then could not resume, so the run has per-layer
evidence through layer 54 and no logits. Tracks mode has since been made resumable and disk-backed (below). The
merged 18-track pass is designed (§ 10.7) and not run.

**How tracks mode works.** `glmref.py --track …` carries one copy of the residual stream per precision variant
through the pass, layer by layer, in the contract's f32 combine order.
- **Sharing.** Tracks that enter a layer with the same state and use the same weights in it share the output:
  it is computed once. So a track that differs only at layer 40 costs nothing before layer 40.
- **Per layer.** It logs every track's hidden-state difference from the master track (relative L2, median and
  max over rows), and the share of rows whose routed expert set equals the master's.
- **At the end.** `model.norm` and the head run on every distinct final state. Per sequence and per pair, it
  reports KL(reference ‖ track) over the full vocabulary, top-1 agreement, and the largest |Δ logit|.
- **Which pairs.** The master is compared with every track. A `--track-ref` is compared with the tracks of its
  own kind (expert formats at the same layers, or non-expert presets).
- **States on disk.** Each distinct state (the bf16 residual stream and the index sets) is a file under
  `OUT/state/tracks/L<next>/`, and `current.json` maps every track to its state. RSS stays near one track's
  working set: 1.2 GiB on `short`. A killed run resumes at the last whole layer.
- **The head is streamed** over 1,024-row blocks and 8,192-row vocabulary slabs of `lm_head`. Logits are f32;
  logsumexp, argmax and the KL sum are accumulated in f64 with online rescaling:
  KL = Σ p_r (z_r − z_n) − lse_r + lse_n.
- **Checked against the old in-memory head.** On a 5-layer `short` smoke test, `nvfp4_rtn@3` vs master gives KL
  4.52e-5 against 4.5e-5 before, and `ne:fp8` vs master 2.85e-4, top-1 0.969, against 2.8e-4 and 0.969.
- **Not yet tested:** a kill followed by a resume. The test (`--layers 3`, then `--layers 5` in the same out
  dir, compared byte for byte with a straight `--layers 5`) was cut by the pause.

**The tracks.**

*The master track:*
- Experts: the BF16 master's where `--master` holds the layer, NVIDIA's NVFP4 everywhere else. Only layers 3 and
  40 are fetched (`~/glm53-master` on rank 2, 1,536 tensors, 38.66 GB).
- Non-expert weights as NVIDIA ships them: BF16 attention, indexer and shared experts; per-tensor FP8 dense
  MLPs in layers 0–2.

*A format track,* `FORMAT@L`:
- The master's experts in FORMAT at layers L. `nvfp4_rtn` is NVIDIA's recipe, recomputed.
- `s3x2gu_nvfp4d`: gate and up in S3×2, the E2M1-subset 3-bit code with subsets {0.5, 2, 4, 6} and
  {0.5, 1.5, 3, 6}, group 16, e4m3 scales; down in `nvfp4_rtn`. 3.83 bpw.
- `s3x2`: all three projections in S3×2. 3.5 bpw.
- `k3`, `k4`: trellis K3 or K4 on all three, exllamav3's code, data-free.
- The suffix `/sub` puts the format only on `--sub-experts`, which default to the quantiser's trellis sample
  0, 51, 102, 153, 204, 255. The other 250 experts get `nvfp4_rtn`, the same mixing for every format.
- These six carry 3.0 % (layer 3) and 3.7 % (layer 40) of the routing mass on `short`+`python`, and 2.2 % and
  3.0 % over the whole golden set (uniform: 2.3 %). 16 % and 27 % of rows route to at least one of them.

*A non-expert track,* `ne:NAME`: the master's experts, with the non-expert linears quantised by a preset.
- `ne:fp8` is DESIGN decision 2: zai's FP8 for `q_a`, `q_b`, `kv_a`, `kv_b`, `o_proj`, the indexer's `wq_b` and
  `wk`, and the shared expert's gate, up and down.
- `ne:o4` puts `o_proj` at NVFP4; `ne:oqk4` adds `q_b` and `kv_b`; `ne:oqks4` adds the shared expert.
- In every preset, layers 0–2's dense MLPs stay NVIDIA's per-tensor FP8. The routers, norms, the indexer's
  `weights_proj`, the embedding and the head stay as shipped.

**How the weights are computed.**
- **zai's FP8, recomputed.** Per 128×128 block, scale = amax · f32(1/448), codes = e4m3(w / scale). The scale
  rule was checked against zai's release (zai-org/GLM-5.3 at c5087a76, HTTP range reads): `weight_scale_inv` is
  equal on all six tensors read (layers 10 and 40: `o_proj`, the shared expert's `down_proj`, `q_b_proj`).
  NVFP4 for a non-expert tensor is `nvfp4_rtn` with a per-tensor amax: the shared expert's gate and up each get
  their own, unlike the routed experts' shared amax.
- **Dequantised non-expert weights are f32** (code × scale, rounded once), and that linear runs as an f32 GEMM
  with one rounding to bf16.
- **Routed experts.**
  - NVFP4 and S3×2 are held as the exact f32 product of a 4- or 3-bit E2M1 value and an e4m3 block scale, with
    the per-tensor scale applied after the GEMM. This is the kernel's order, and it is how the reference runs
    the checkpoint's NVFP4.
  - The quantiser's own S3×2 dequantiser rounds e4m3 · s2 first. The two agree within one f32 rounding
    (asserted on every tensor in `qprep.py`).
  - Trellis weights are the f32 output of `trellis.dequantise`, with a scale of 1.

**`nvfp4_rtn` against NVIDIA's bytes.** `check_nvfp4_rtn.py` checked all 256 experts of layers 3 and 40 (gate,
up and down). Every byte is equal:

| Layer | packed codes | e4m3 block scales | `weight_scale_2` |
|---|---|---|---|
| 3 | 4,831,838,208 of 4,831,838,208 | 603,979,776 of 603,979,776 | 768 of 768 |
| 40 | 4,831,838,208 of 4,831,838,208 | 603,979,776 of 603,979,776 | 768 of 768 |

So the `nvfp4_rtn` tracks are exactly NVIDIA's shipped quantisation at those layers. The master track against
them measures what NVIDIA's NVFP4 costs.

**Repack provenance** (`qprep.py` into `~/glm53-ref/qcache` on rank 2, 21 GB; `provenance.jsonl` there). The
packing calls `quant/run.py`'s own `make_formats` and `phash`, and every packed tensor's SHA-256 is compared with
the hashes recorded in the quantiser's results (`/data/scratch/glm53-quant/results-*.jsonl` on rank 0, copied to
`~/glm53-ref/quant-results`):

| Layer | Format | Tensors packed | Equal to the quantiser's SHA-256 | No recorded hash |
|---|---|--:|--:|--:|
| 3 | S3×2 (`e2m1sub3x2_g16`) | 768 (all 256 experts) | **768** | 0 |
| 40 | S3×2 | 768 | **558** | 210 (experts the quantiser's layer-40 run did not reach) |
| 3 | trellis K4 | 18 (the six experts) | **18** | 0 |
| 3 | trellis K3 | 15 of 18 (expert 255 not yet) | **15** | 0 |
| 40 | trellis K4, K3 | 0 of 36 | | |

No tensor disagreed. The bytes do not depend on the torch version (2.14.1 here, 2.11 in the quantiser's run).
A trellis tensor took ~133 s at 2 threads.

**Partial evidence from the killed run** (`golden/tracks/tracks.json`, layers 0–54, `short`+`python`, hidden
state relative to the master track, median over rows, / share of rows with the master's routed set):

| Track | after L3 | after L4 | after L39 | after L40 | after L41 | after L54 |
|---|---|---|---|---|---|---|
| `nvfp4_rtn@3` | 0.0023 / 1.00 | 0.0035 / 0.93 | 0.141 / 0.39 | 0.143 / 0.37 | 0.144 / 0.35 | 0.168 / 0.19 |
| `nvfp4_rtn@40` | 0 | 0 | 0 | 0.034 / 1.00 | 0.033 / 0.79 | 0.082 / 0.45 |
| `nvfp4_rtn@3,40` | 0.0023 / 1.00 | 0.0035 / 0.93 | 0.141 / 0.39 | 0.147 / 0.37 | 0.147 / 0.35 | 0.171 / 0.19 |
| `ne:fp8` | 0.0077 / 0.84 | 0.011 / 0.80 | 0.179 / 0.29 | 0.180 / 0.27 | 0.179 / 0.27 | 0.202 / 0.14 |
| `ne:o4` | 0.0097 / 0.81 | 0.015 / 0.74 | 0.232 / 0.20 | 0.234 / 0.18 | 0.232 / 0.18 | 0.252 / 0.09 |
| `ne:oqk4` | 0.011 / 0.77 | 0.017 / 0.71 | 0.297 / 0.13 | 0.297 / 0.12 | 0.294 / 0.12 | 0.307 / 0.06 |
| `ne:oqks4` | 0.015 / 0.77 | 0.021 / 0.67 | 0.312 / 0.12 | 0.312 / 0.11 | 0.308 / 0.11 | 0.322 / 0.05 |

- **Divergence grows through the stack.** A difference at one layer (0.2 % after layer 3) grows to ~15 % by
  layer 40, as § 6.4's rounding-order difference did. By the middle of the stack, most rows route differently.
  So per-layer hidden-state numbers rank variants but cannot set tolerances; the final KL and top-1 must.
- **The non-expert presets order as expected:** `fp8` < `o4` < `oqk4` < `oqks4`. Even `ne:fp8` (zai's FP8)
  moves the state more than `nvfp4_rtn` at layer 3, because it changes every layer.

**5-layer smoke test** (`short`, the stack cut after layer 4, so these are not model-quality numbers; they check
the wiring and give the order):

| Track | vs | KL mean | top-1 | max \|Δ logit\| |
|---|---|--:|--:|--:|
| `nvfp4_rtn@3` | master | 4.5e-5 | 0.986 | 0.19 |
| `s3x2gu_nvfp4d@3` | master | 7.4e-5 | 0.986 | 0.46 |
| `s3x2@3` | master | 3.1e-4 | 0.983 | 0.67 |
| `s3x2/sub@3` | master | 4.5e-5 | 0.986 | 0.19 |
| `s3x2gu_nvfp4d@3` | `nvfp4_rtn@3` | 6.3e-5 | 0.979 | 0.34 |
| `s3x2@3` | `nvfp4_rtn@3` | 3.1e-4 | 0.979 | 0.67 |
| `s3x2/sub@3` | `nvfp4_rtn@3` | 2.8e-6 | 0.993 | 0.11 |
| `ne:fp8` | master | 2.8e-4 | 0.969 | 0.59 |
| `ne:o4` | master | 4.0e-4 | 0.966 | 0.50 |
| `ne:o4` | `ne:fp8` | 3.2e-4 | 0.973 | 0.57 |

### 10.4 Non-expert precision tracks

Designed and wired (`ne:fp8`, `ne:o4`, `ne:oqk4`, `ne:oqks4`; § 10.3), with partial evidence only: the
per-layer table above through layer 54 and the 5-layer smoke test. The full-depth KL, top-1 and max |Δ logit|
on `short` and `python` come from the merged pass (§ 10.7). No decision is drawn: DESIGN decision 2 (FP8
non-experts) stands.

### 10.5 The MTP layer's experts at NVFP4 (experiment 5)

The checkpoint ships the MTP layer (78) in BF16. `--mtp-from all-f32 --mtp-experts nvfp4_rtn` reruns only that
layer on `all-f32`'s final hidden states, with its 256 routed experts requantised by NVIDIA's recipe
(`nvfp4_rtn`, gate and up sharing one amax) and everything else as shipped; 372 s, pinned, 6 threads
(`golden/mtp-nvfp4`, `final/` a symlink to `all-f32`'s). Acceptance is the draft against the main model's
argmax at t+1, as in § 6.7; KL and |Δ logit| are over the MTP layer's sample rows.

| Sequence | rows | drafts equal (BF16 vs NVFP4) | acceptance, BF16 experts | NVFP4 experts | Δ | KL(BF16 ‖ NVFP4) mean / max | max \|Δ logit\| |
|---|---:|---:|---:|---:|---:|---|---:|
| `short` | 290 | 0.993 | 0.7966 | 0.7966 | +0.0000 | 2.0e-3 / 0.049 | 1.31 |
| `prose` | 4,761 | 0.978 | 0.5869 | 0.5848 | −0.0021 | 4.1e-3 / 0.035 | 1.31 |
| `python` | 6,105 | 0.985 | 0.6883 | 0.6875 | −0.0008 | 4.1e-3 / 0.035 | 1.47 |
| `agentic` | 26,833 | 0.977 | 0.4544 | 0.4546 | +0.0001 | 3.4e-3 / 0.035 | 1.31 |

NVFP4 experts change about 2 % of the drafts, and the acceptance moves by at most 0.2 points (`prose`), with
no consistent sign. On this evidence the MTP layer's experts can be stored at NVFP4 like the main layers'. The
attention and shared expert of layer 78 stayed BF16 here.

### 10.6 Per-expert input covariances (experiment 3b): format, and where the pass stopped

`glmref.py --cov-out DIR --cov-layers 3 40 --master ~/glm53-master` runs the normal pass (the `all-f32` order,
the four golden sequences) and stops after the last covariance layer.
- **What is collected.** At each of those layers, for every routed expert, over the rows routed to it:
  - the gate/up input x (the post-attention RMSNorm output);
  - the down input h = silu(x W_gᵀ) ⊙ (x W_uᵀ), with the BF16 master's weights in bf16 GEMMs, as the reference
    computes it.
- **The file:** `DIR/layer{L}_expert{E}.pt`, as `quant/run.py --cov` reads it:
  `{'gate_up': f16 [6144, 6144], 'down': f16 [2048, 2048], 'count': rows, 'scale': {'gate_up': s, 'down': s}}`.
  Each matrix is the f32 mean of x xᵀ (an f32 GEMM), divided by its largest diagonal entry s, so the true
  second moment is the matrix × `scale`. run.py's ratio √(tr(EΣEᵀ)/tr(WΣWᵀ)) does not depend on the scale.
  An expert no row routes to gets no file (none is expected: every expert is used in every layer of the golden
  set, § 6.6).
- **Size** (*estimate*): 84 MB an expert, **43 GB** for 512 experts. They are to be copied to
  `/data/scratch/glm53-quant/cov/` on rank 0 (agreed cap 60 GB).
- **Where it stopped.** The first run (09:00–09:08, 6 threads) was stopped to make room for a GPU job, after
  layer 2: `golden/cov-pass/state/progress.json` is `{"next": 3}`. No covariance has been written yet.
- **Memory.** On the full golden set this pass holds ~6 GiB (the 26.8k-row `agentic` attention). That is above
  the 3 GB cap that holds while GPU jobs on real weights are possible, so it needs an exclusive window. It
  could instead be split per sequence, since the reference is batch-invariant (§ 6.5): `short`, `prose` and
  `python` each fit, and their accumulators can be summed by count. `agentic` alone does not fit without a
  head-group-chunked attention.

### 10.7 The planned merged tracks pass (18 tracks, `short` + `python`)

```
--master ~/glm53-master --qcache ~/glm53-ref/qcache
--track nvfp4_rtn@3 --track nvfp4_rtn@40 --track nvfp4_rtn@3,40
--track s3x2gu_nvfp4d@3 --track s3x2gu_nvfp4d@40 --track s3x2gu_nvfp4d@3,40
--track s3x2@3 --track s3x2@40 --track s3x2@3,40
--track s3x2gu_nvfp4d/sub@3,40 --track s3x2/sub@3,40 --track k4/sub@3,40 --track k3/sub@3,40
--track ne:fp8 --track ne:o4 --track ne:oqk4 --track ne:oqks4
--track-ref nvfp4_rtn@3 --track-ref nvfp4_rtn@40 --track-ref nvfp4_rtn@3,40 --track-ref ne:fp8
```

- **Comparisons.** Every track against the master. Every expert track also against `nvfp4_rtn` at the same
  layers: the `/sub` tracks against `nvfp4_rtn@3,40`, which they equal outside the six experts. The `ne:` tracks
  also against `ne:fp8`.
- **The six-expert tracks run at layers 3 and 40 together only** (agreed). They need the trellis repack of layer
  40 to finish first.
- **Cost.** About 1,140 track-layers after sharing (5 distinct states in layers 0–2, 13 in layers 3–39, 18
  from layer 40). That is ~11 h at 6 threads beside nothing else (~35 s a track-layer, measured in the killed
  run).
- **Fallback.** If the resume ever fails, three ~4 h passes give the same table: the expert tracks at 3 and
  3,40; at 40; and the `ne:` tracks.

### 10.8 How to resume (after the 2026-10-14 pause)

All on rank 2. Every command is pinned to cores 10–19, nice 19, from `~/glm53-ref/code`, with
`OMP_NUM_THREADS` equal to `--threads`. **A CPU pass may hold at most 3 GB RSS while GPU jobs on real weights are
possible.** When a fill holds cores 10–19, use ≤ 4 threads. Anything bigger needs an exclusive window.

1. **Finish the repack.** Rerun `~/glm53-ref/qprep.sh` (2 threads). It skips every tensor already in
   `~/glm53-ref/qcache` and appends to `provenance.jsonl`. Left to do: layer 3 trellis K3 expert 255 (3
   tensors), and layer 40 trellis K4 and K3 (36 tensors), ~1.5 h. ~0.5 GB RSS.
2. **The resume test.** Run a 7-track smoke on `short` with `--layers 3` and then `--layers 5` in one out dir,
   and compare `tracks.json` and `kl_*.npy` byte for byte with a straight `--layers 5`.
3. **The covariance pass** (exclusive window, ~6 GiB, ~2.5 h). Rerun `~/glm53-ref/covpass.sh`. It resumes at
   layer 3 from `golden/cov-pass/state/`, and after a kill it reruns itself up to 8 times. It writes to
   `~/glm53-ref/cov/`. Then:
   `rsync -a --bwlimit=600000 ~/glm53-ref/cov/ rank0:/data/scratch/glm53-quant/cov/`, check the sizes, and
   delete the rank-2 copy.
4. **The merged tracks pass** (§ 10.7, ≤ 3 GB, resumable):
   `glmref.py --threads 6 --ckpt ~/hf/glm53-nvfp4 --out-dir ~/glm53-ref/golden/tracks18 <the § 10.7 flags>
   --tokens tokens/short.tokens.json tokens/python.tokens.json`, in a retry loop like `covpass.sh`'s. It
   resumes from `golden/tracks18/state/tracks/current.json`.
