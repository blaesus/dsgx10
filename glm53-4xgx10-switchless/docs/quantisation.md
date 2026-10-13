# Quantisation and the memory budget (2026-10-13)

What has to fit on each box besides the weights, what each tensor class costs at each precision,
the candidate layouts with their decode floors, how a requantisation would be produced, and what
"exact output" means for a quantised GLM-5.3. CPU work only: shard headers, a few sampled tensors from
the local NVFP4 checkpoint, and single tensors fetched from Hugging Face by HTTP range requests.
Numbers marked *estimate* are not measured.

Paths: the NVFP4 checkpoint is rank 0's
`/data/hf/hub/models--nvidia--GLM-5.3-NVFP4/snapshots/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf` (`NV` below).
The tools are in [../tools](../tools):

```
python3 -I tools/tally.py NV --json tally.json             # the tensor-class table (headers only)
python -I tools/nvfp4_entropy.py NV                          # entropy of NVFP4 nibbles and scales (60 sampled tensors)
python -I tools/provenance.py NV OUT --formats               # where the checkpoint's numbers come from; 3-bit formats
```

(`python` is the CPU-only torch 2.11 environment, run with `-I`; `nvfp4_entropy.py` needs numpy only.)

## Summary

- **Budget.** A rank has about **100 GiB for weights and KV together** (121.6 GiB less rank 0's measured
  ~10 GiB of system and agents, a 4 GiB margin and ~7 GiB of engine buffers, the last an *estimate*
  scaled from the DeepSeek engine's measured 6.8 GiB). KV is GLM-5.3's big cost: 52.7 KiB a token in
  FP8, 93 KiB in BF16, against the DeepSeek engine's ~0.9 KiB. So the defensible **weight budget is
  87 GiB a rank**. That leaves 13.4 GiB of FP8 KV: one 256k session with the latent replicated on
  every rank, or one 1M session (or four 256k ones) with the KV split across ranks by position. The
  hard ceiling is 94 GiB (one 128k session). Every GiB over 87 costs 19.6k tokens of replicated
  context.
- **The published NVFP4 does not fit.** Experts as published are 94.9 GiB a rank, and the leanest
  layout around them is 100.0 GiB (no MTP layer, FP8 non-experts, the embedding split). That leaves
  ~0.6 GiB of KV.
- **Provenance, measured.** NVIDIA's BF16 tensors are the BF16 master. zai's FP8 release is
  round-to-nearest of them (100.0000 % of codes on 4 tensors, 127M values). NVIDIA's NVFP4 experts are
  plain round-to-nearest-even of the BF16 master, reproduced bit for bit on 2 tensors. They carry no
  calibration; only the activation `input_scale`s were calibrated. So "NVFP4 as published" is a
  deterministic function of `zai-org/GLM-5.3-BF16`, and any requantisation should start from that
  master too.
- **Decode bytes.** At one token a step, a rank reads 3.19 GB of NVFP4 experts and 3.22 GB of FP8
  attention (6.44 GB in BF16). Attention costs as much as the experts, so non-expert precision moves
  the floor as much as expert bits do. Floors at 235 GB/s: 51.0 ms a token (NVFP4 experts, BF16 rest),
  34.3 ms (FP8 non-experts), 32.3 ms (3.85 bpw experts, FP8 non-experts).
- **Recommendation.** Requantise the routed experts from the BF16 master to a mixed 3/4-bit code
  averaging ~3.75 bpw (79 GiB a rank). Trellis is the leading candidate on the scout's quality
  evidence; an E2M1-grid 3-bit code is the fallback that keeps the FP4 tensor-core path. Non-experts
  stay at zai's own FP8 values, the head BF16, and the MTP layer at ≤ 4.5 bpw (it only drafts).
  About 86 GiB a rank, with a decode floor of **~32.0 ms a token** (31 tok/s before drafting).
- **Exactness.** The weights are a hashed artefact made reproducibly from the hashed BF16 master. The
  engine is bit-stable against pinned output hashes, and agrees within stated tolerances with a
  layer-streaming torch reference of the same dequantised weights and documented numerics. Quality is
  a separate, reported number: KL and top-1 against BF16 logits on one fixed harness, with NVFP4 as
  published scored on the same harness as the anchor.

## 1. The per-rank budget

### 1.1 What the DeepSeek engine measured (the anchor)

| Fact | Value | Source |
|---|---|---|
| Memory a box | MemTotal 127,533,356 kB = **121.62 GiB** | `/proc/meminfo` on rank 0, 2026-10-13 |
| DeepSeek rank file | 72.38 GiB | [layout.md:18](../../ds41f-4xgx10-switchless/docs/layout.md): "Every rank file is **72.38 GiB with 1,609 tensors**" |
| Engine GPU memory while serving 262,144-token contexts | **93,352 MiB = 91.16 GiB** on every rank | [engine.md:3243](../../ds41f-4xgx10-switchless/docs/engine.md): "GPU memory was 93,352 MiB on every rank in every sample (everything is allocated at startup)" |
| MemAvailable at that point | **21.4 / 23.2 / 23.1 / 23.2 GiB**, ranks 0–3 | engine.md:3248, the soak's `start` row |
| of which the session store (KV) | 12 GiB | engine.md:108: "session store \| 12 GiB \| serve: one pool reserved at startup" |
| prefill buffers and gathered weights | 4.69 GiB at 266,240 with 8,192-row chunks (3.76 at 4,096 rows) | engine.md:104 |
| spec step graph | 0.24 GiB | engine.md:107 |
| Engine host resident set | 0.76–0.77 GiB at start (plus the Engram row cache, which GLM-5.3 does not have) | engine.md:3248 |
| Rank 0 today, serving | engine 92,864 MiB (90.69 GiB), MemAvailable 20.93 GiB; also running: 3 Claude Code and 2 Codex sessions, Postgres, the front-end (`ps`, `nvidia-smi`, `free`, 2026-10-13 20:49) | |

From these:

- **System, not the engine's GPU memory and not available**: 121.62 − 91.16 − 23.2 = **7.3 GiB on the
  workers**. On rank 0 it is 121.62 − 91.16 − 21.4 = 9.1 GiB at the soak's start, and 121.62 − 90.69 −
  20.93 = **10.0 GiB today**. This covers the OS, the desktop, daemons, the engine's host resident set
  and pinned buffers, the front-end, and on rank 0 the agents and Postgres.
- **The engine's own non-weight, non-KV memory**: 91.16 − 72.38 − 12 = **6.8 GiB**. That is the
  prefill buffers (4.69), graphs (0.24), all-gathered router and mHC copies (0.31), the FP8 drafter head
  (0.165), the RoPE tables, ring and step buffers, and the CUDA context (the remaining ~1.3 GiB,
  *inferred*).
- **Page cache for the loader**: none needed. The loader reads with O_DIRECT, and `run4.sh` drops the
  model files' cache first, because CUDA on GB10 allocates only free pages
  ([findings.md](../../ds41f-4xgx10-switchless/docs/findings.md#environment-gotchas)). Conversion jobs need their own
  page-cache discipline, but not at serve time.

### 1.2 GLM-5.3's non-weight memory

- **System**: 10.0 GiB on rank 0 and 7.3 on the workers, as measured. TP keeps every rank's weights
  equal, so rank 0 binds.
- **Margin**: 4 GiB of MemAvailable kept on rank 0, for the agents' growth and builds; 2 GiB on the
  workers, the DeepSeek store's `--session-margin-gib` default. Today's 21 GiB of headroom on rank 0
  would shrink to ~4. **That is the user's call**, and it is the first number to revisit.
- **Engine buffers**: **~7 GiB at 8,192-row prefill chunks, ~5 GiB at 4,096** (*estimate*: DeepSeek's
  6.8 GiB scaled by hidden size, 6,144 against 5,120, with GLM's wider o_proj input (16,384) in the
  prefill scratch). It is measurable only once the engine exists.
- **KV**, per token ([model.md § 4](model.md#4-what-must-be-cached-per-token)): the latent (512) and
  `k_rope` (64) on all 78 layers, the indexer key (128) on the 21 full-indexer layers (0, 1, 2, 6, 10,
  …, 74; `config.json` `indexer_types`, counted with `python3 -I`: 21, not 22), and the MTP layer's
  own latent and indexer key when it drafts.

| Format | Main model, B/token | + MTP layer | Tokens per GiB (with MTP) |
|---|---:|---:|---:|
| BF16 (1,152 B a layer + 256 B indexer) | 95,232 | 96,640 | 11,111 |
| FP8 as vLLM/SGLang store it (656 B a layer: 512 e4m3 + 4 f32 scales + 64 bf16 rope; indexer 132 B) | 53,940 | 54,728 | 19,620 |

The latent is shared by all heads, so splitting heads across ranks does not split it. Every rank
holds all of it, unless the sequence is split across ranks by position (decode context parallelism,
"DCP" below: each rank attends over its quarter of the positions for all 64 heads, then the partial
outputs are combined).

KV in GiB per rank, with the MTP layer:

| Context | Sessions | BF16, replicated | FP8, replicated | BF16, DCP 4 | FP8, DCP 4 |
|---|---:|---:|---:|---:|---:|
| 128k | 1 | 11.80 | 6.68 | 2.95 | 1.67 |
| 128k | 2 | 23.59 | 13.36 | 5.90 | 3.34 |
| 128k | 4 | 47.19 | 26.72 | 11.80 | 6.68 |
| 256k | 1 | 23.59 | 13.36 | 5.90 | 3.34 |
| 256k | 2 | 47.19 | 26.72 | 11.80 | 6.68 |
| 256k | 4 | 94.38 | 53.45 | 23.59 | 13.36 |
| 1M | 1 | 94.38 | 53.45 | 23.59 | 13.36 |
| 1M | 2 | 188.75 | 106.89 | 47.19 | 26.72 |
| 1M | 4 | 377.50 | 213.78 | 94.38 | 53.45 |

(Computed as context × bytes per token × sessions [÷ 4] / 2³⁰; 128k = 131,072, 256k = 262,144, 1M =
1,048,576.)

Two notes:
- The checkpoint declares `kv_cache_quant_algo: FP8` but ships no KV scales ([model.md § 4](model.md#4-what-must-be-cached-per-token)).
  An FP8 cache is therefore our own choice with our own scaling, and it changes outputs: it belongs in
  the numerics contract (§ 5), measured against a BF16 cache.
- DCP costs collectives per layer and token: an all-gather of the absorbed queries (64 heads × 576
  values) and a combine of partial outputs and log-sum-exps. At the DeepSeek ring's 12–25 µs per
  collective, two per layer over 78 layers is ~2–4 ms a token, 5–10 % of a ~32 ms step (*estimate*).
  That is a DESIGN decision; here it only sets how far a KV budget reaches.

### 1.3 The weight budget

Weights + KV ≤ 121.6 − 10.0 (system, rank 0) − 4 (margin) − 7 (engine buffers) = **~100.6 GiB a rank**
(105.3 on the workers; rank 0 binds).

| KV target | KV GiB | Weights at most |
|---|---:|---:|
| 128k × 1, FP8, replicated | 6.7 | 93.9 |
| 256k × 1, FP8, replicated (= 1M × 1 or 256k × 4 with FP8 DCP) | 13.4 | **87.2** |
| 128k × 1, BF16, replicated | 11.8 | 88.8 |
| 256k × 1, BF16, replicated | 23.6 | 77.0 |
| 128k × 4 or 256k × 2, FP8, replicated | 26.7 | 73.9 |

**Budget: 87 GiB of weights a rank.** It keeps the DeepSeek engine's 256k single-session context with
replicated FP8 KV, or a BF16 cache of ~144k. With DCP it reaches 1M for one session or 256k for four.
Above 87 GiB, each GiB costs 19.6k tokens of replicated FP8 context (11.1k BF16; ×4 with DCP).
Beyond **94 GiB** a 128k agentic session no longer fits. Shrinking the prefill chunks to 4,096 rows
buys ~2 GiB (*estimate*, from DeepSeek's 3.76 against 4.89 GiB) at some prefill speed, to be measured.

## 2. Tensor classes (from the shard headers)

`tools/tally.py NV` (2026-10-13). It reads only the 8-byte length and JSON header of each of the 50
shards. 232,403 tensors, 432.24 GiB. Per rank at TP=4: everything is split four ways except the
embedding, norms, routers and indexers, which are replicated (DeepSeek's partition rules,
[layout.md § Partition](../../ds41f-4xgx10-switchless/docs/layout.md#partition)). Precision columns are
bytes per parameter with scales: FP8 128×128 blocks with f32 scales (the official release's format)
1.00024 B; NVFP4 0.5625; MXFP4 0.53125; int3 with an f16 scale per 64 values 0.40625; trellis at 3.0 bpw
0.375.

| Class | Tensors | Params (M) | Stored | Stored GiB | Per rank as stored | BF16 | FP8 b128 | NVFP4 | MXFP4 | int3 g64 | trellis 3.0 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| routed experts, layers 3–77 | 230,400 | 724,775.7 | U8 337.50 + E4M3 42.19 + f32 | 379.69 | **94.92** | 337.50 | 168.79 | 94.92 | 89.65 | 68.56 | 63.28 |
| attn o_proj [6144, 16384] | 78 | 7,851.7 | BF16 | 14.63 | 3.66 | 3.66 | 1.83 | 1.03 | 0.97 | 0.74 | 0.69 |
| attn q_b_proj | 78 | 2,617.2 | BF16 | 4.88 | 1.22 | 1.22 | 0.61 | 0.34 | 0.32 | 0.25 | 0.23 |
| attn kv_b_proj | 78 | 1,145.0 | BF16 | 2.13 | 0.53 | 0.53 | 0.27 | 0.15 | 0.14 | 0.11 | 0.10 |
| attn q_a_proj | 78 | 981.5 | BF16 | 1.83 | 0.46 | 0.46 | 0.23 | 0.13 | 0.12 | 0.09 | 0.09 |
| attn kv_a_proj_with_mqa | 78 | 276.0 | BF16 | 0.51 | 0.13 | 0.13 | 0.06 | 0.04 | 0.03 | 0.03 | 0.02 |
| shared expert, layers 3–77 | 225 | 2,831.2 | BF16 | 5.27 | 1.32 | 1.32 | 0.66 | 0.37 | 0.35 | 0.27 | 0.25 |
| dense MLP, layers 0–2 | 27 | 679.5 | E4M3 per-tensor | 0.63 | 0.16 | 0.32 | 0.16 | 0.09 | 0.08 | 0.06 | 0.06 |
| lm_head (vocab-split) | 1 | 951.6 | BF16 | 1.77 | 0.44 | 0.44 | 0.22 | 0.12 | 0.12 | 0.09 | 0.08 |
| embed (replicated) | 1 | 951.6 | BF16 | 1.77 | 1.77 | 1.77 | 0.89 | 0.50 | 0.47 | 0.36 | 0.33 |
| indexer, 21 layers (replicated) | 105 | 196.8 | BF16 | 0.37 | 0.37 | 0.37 | 0.18 | 0.10 | 0.10 | 0.07 | 0.07 |
| router + bias (replicated) | 150 | 118.0 | BF16 + f32 | 0.22 | 0.22 | 0.22 | 0.11 | 0.06 | 0.06 | 0.04 | 0.04 |
| norms (replicated) | 320 | 1.2 | BF16 | 0.00 | 0.00 | | | | | | |
| **MTP layer 78: experts** | 768 | 9,663.7 | **BF16** | 18.00 | **4.50** | 4.50 | 2.25 | 1.27 | 1.20 | 0.91 | 0.84 |
| MTP attention, shared expert, eh_proj, indexer, router | 16 | 289.2 | BF16 | 0.54 | 0.15 | 0.15 | 0.07 | 0.04 | 0.04 | 0.03 | 0.03 |
| **all** | 232,403 | | | **432.24** | **109.85** | 352.58 | 176.33 | 99.16 | 93.66 | 71.62 | 66.11 |

Corrections to the task's tally:
- The 19,456 "U8" expert tensors per class are 19,200 U8 (layers 3–77) plus **256 BF16** (layer 78's,
  `[2048, 6144]`, excluded by `ignore: ["model.layers.78*"]`). That BF16 is 18.0 GiB, which is why each
  class appeared as 118.5 GiB: 112.5 of E2M1 and 6.0 of BF16.
- NVFP4's true expert bytes are 337.50 GiB of values and 42.19 of scales.

One bit per weight of routed experts costs **21.09 GiB a rank** (724.78e9 / 4 / 8 / 2³⁰).

## 3. Candidate layouts

### 3.0 What the checkpoint's numbers are (measured)

`tools/provenance.py NV OUT` fetched, by HTTP range requests, the named tensors from the official
FP8 release (`zai-org/GLM-5.3` revision `c5087a76`, 141 shards, 703.77 GiB, FP8 e4m3 with 128×128 f32
`weight_scale_inv` blocks for every linear layer including attention, shared and MTP experts and the
indexer; BF16 for embed, head, norms and routers; HF API `?blobs=true`) and from the BF16 mirror
(`zai-org/GLM-5.3-BF16` revision `9d2398f4`, 282 shards, **1,403.22 GiB**, all BF16). It found:

| Check | Tensors | Result |
|---|---|---|
| e4m3(NVFP4 checkpoint's BF16 ÷ official `scale_inv`) = official FP8 codes | layer 10 o_proj, shared gate, indexer wk; layer 78 expert 0 down | **100.0000 %** of 127,401,984 values. NVIDIA's BF16 is the master, and the FP8 release is its round-to-nearest |
| the same, with BF16 rounding of the FP8 dequantisation compared instead | same | 6.5 % equal: the FP8 release is lossy against the master (relative RMS error 0.026) |
| NVFP4 from the BF16 master: `scale_2 = amax/(6·448)`, block scale `= e4m3(amax₁₆/6/scale_2)`, value `= E2M1_rne(w/(scale·scale_2))` | layer 10 expert 0 gate and down | `scale_2` equal; block scales **100.0000 %**; values **100.0000 %** (rounding ties to the lower code instead gives 99.854 % on gate) |
| NVFP4 against the master | same | relative RMS error 0.0950 / 0.0951 |
| NVIDIA's per-tensor FP8 dense MLP against the official block FP8 | layer 0 gate_proj | 0.46 % of values equal, relative RMS difference 0.037: NVIDIA requantised it per tensor |

So the NVFP4 weights carry no calibration. They are round-to-nearest-even, the same rule as the
checkpoint's activation formula ([model.md § 5.2](model.md#52-nvfp4-routed-experts)), and anyone can
regenerate them from the BF16 master. Only `input_scale` (activations) was calibrated. Choosing
better scales alone ("best of 7 scales per block", minimising each block's squared error) takes the
same NVFP4 bytes from 0.0950 to 0.0884 relative RMS error (`--formats`). Weight error is only a proxy
for KL, though.

### 3.1 Decode bytes per token

At one token a step, rank-local, 128k context. Formula per class: active parameters × bytes ÷ 4 ranks:
- routed experts: 8 a layer × 75 layers × 37.75M;
- shared: 75 × 37.75M;
- dense MLP: 679.5M FP8;
- attention: 12,871M;
- indexer weights: 197M FP8, split by heads;
- router: 236M BF16, split by output;
- head: 951.6M;
- sparse attention: 2,048 selected entries × 656 B × 78 layers, read whole on every rank;
- indexer scan: 128k × 132 B × 21 layers, split by position.

| GB per rank per token | A: NVFP4, BF16 rest | A′: NVFP4, FP8 rest | B: 3.85 bpw, FP8 rest | B + o_proj NVFP4 | B + all attention NVFP4 | 3.25 bpw, FP8 head too |
|---|---:|---:|---:|---:|---:|---:|
| routed experts | 3.185 | 3.185 | 2.725 | 2.725 | 2.725 | 2.300 |
| shared expert | 1.416 | 0.708 | 0.708 | 0.708 | 0.708 | 0.708 |
| dense MLP | 0.170 | 0.170 | 0.170 | 0.170 | 0.170 | 0.170 |
| attention | 6.436 | 3.219 | 3.218 | 2.359 | 1.810 | 3.218 |
| indexer + router | 0.109 | 0.109 | 0.109 | 0.109 | 0.109 | 0.109 |
| head | 0.476 | 0.476 | 0.476 | 0.476 | 0.476 | 0.238 |
| KV read (sparse attention + 128k indexer scan) | 0.196 | 0.196 | 0.196 | 0.196 | 0.196 | 0.196 |
| **total** | **11.99** | **8.06** | **7.60** | **6.74** | **6.19** | **6.94** |
| **floor at 235 GB/s** | **51.0 ms** | **34.3 ms** | **32.3 ms** | **28.7 ms** | **26.4 ms** | **29.5 ms** |
| whole model, GB per token | 47.9 | 32.2 | 30.4 | 27.0 | 24.8 | 27.8 |

The whole-model column agrees with the scout's ~44.3 GB (BF16 non-experts) and ~26.7 GB (3.25 bpw,
FP8) per token ([scouting.md, Verdict](scouting.md#verdict)); the small differences are the head's
precision and the KV reads it left out. For scale: the DeepSeek engine reads 3.26 GB a rank per token
(13.9 ms; [DESIGN.md](../../ds41f-4xgx10-switchless/DESIGN.md#budgets-that-drive-the-design)), and
reached 214 GB/s overall (17.9 ms measured at k = 1). At its efficiency these floors become ~1.2×
longer. The indexer scan grows with context: +0.09 GB (+0.4 ms) a token per 128k.

**Prefill.** Active parameters are 40.2B a token (routed 22.65B, attention 12.87B, shared 2.83B, head
0.95B, dense 0.68B, indexer 0.20B). That is ~80 GFLOP a token in GEMMs plus ~22 GFLOP of sparse
attention at ≥ 2,048 context, ~25 GFLOP a rank. The DeepSeek engine prefills 8.0–8.3k tok/s
(engine.md soak table) on ~17 GFLOP a token: ~34 TFLOPS a box effective, against 214–256 for
block-scaled FP8×FP4 MMA in microbenchmarks
([findings.md § Tensor cores](../../ds41f-4xgx10-switchless/docs/findings.md#tensor-cores)). At the
same efficiency GLM-5.3 would prefill ~1.4k tok/s (*estimate*), the same as gr8's published 1,374
tok/s at 8K. Prefill is compute-bound. A layer's experts at NVFP4 are 1.27 GiB a rank, 5.8 ms to
stream once per chunk: ~0.7 µs a token at 8,192 rows, against ~100+ µs of compute. Two consequences:
- **Weight bits do not limit prefill.** Any format that dequantises its tiles into the block-scaled MMA
  in the producer warps costs the same as NVFP4.
- **Activation precision does.** `kind::mxf4nvf4` (FP4 × FP4, 486–512 TFLOPS microbenchmarked) is
  ~2× `kind::mxf8f6f4` (FP8 activations). Only NVIDIA's FP4-activation recipe (§ 5) can use it.

### 3.2 (a) NVFP4 experts as published

| Variant | GiB a rank | Against the 87 GiB budget | Floor |
|---|---:|---:|---:|
| A1: as shipped (BF16 non-experts, embed replicated, MTP BF16) | 109.85 | −22.6 (does not fit at all: 121.6 − 10 − 109.9 = 1.7 GiB for engine and KV) | 51.0 ms |
| A2: + attention, shared and indexer at zai's FP8 values, embed vocab-split, no MTP layer | 100.03 | −12.8: ~0.6 GiB of KV (~12k tokens replicated, ~47k with DCP) | 34.3 ms |
| A2 + MTP experts at NVFP4 (our RTN), MTP rest FP8 | 101.37 | −14.2: no KV | |
| A2 + MTP as shipped (BF16) | 104.68 | −17.5 | |

Kernels on sm_121a: the decode GEMV streams nibbles and e4m3 scales and multiplies in f32 (one more
scale than DeepSeek's MXFP4). Prefill uses `mma.sync … kind::mxf8f6f4 … e4m3.e2m1` with the scale
converted at tile load: e4m3 block scales are not ue8m0, so either `kind::mxf4nvf4`'s
`scale_vec::4X` with e4m3 scales (FP4 activations), or FP8 activations with the weight's e4m3 scale
folded into a per-block f32 multiply after the MMA (*to verify*: whether `mxf8f6f4` accepts e4m3
scales on sm_121a; findings.md lists only ue8m0 for it). **Verdict: does not fit**, even before any
context. It stays useful as the quality anchor and for short-context development runs (A2 at ~4k
context fits with the margins squeezed).

### 3.3 (b) Own requantisation of the experts at ~3–3.9 bpw

To reach 87 GiB with A2's non-experts (5.11 GiB) and an MTP layer at ~1.3 GiB, the experts get ≤ 80.6
GiB, **≤ 3.82 bpw on average**. Weight error on layer 10 expert 0, 512 random rows, from the BF16
master (`provenance.py --formats`):

| Format | bpw with scales | Relative RMS error, gate / down | Decode GEMV | Prefill GEMM |
|---|---:|---|---|---|
| NVFP4, amax scales (as published) | 4.5 | 0.0950 / 0.0951 | nibble → f32 | native (e2m1) |
| NVFP4, best of 7 scales per block | 4.5 | 0.0884 / 0.0883 | same | same |
| int3 {−4…3}, e4m3 scale per 16 | 3.5 | 0.1793 / 0.1790 | 3-bit unpack | **native**: −4…3 are E2M1 values, so the codes map to nibbles exactly |
| int3 {−3…3}, e4m3 per 32 | 3.25 | 0.1966 / 0.1962 | same | native |
| int3 {−4…3}, f16 per 32 | 3.5 | 0.1849 / 0.1846 | same | f16 scale outside the MMA |
| **E2M1 subset {±0.5, ±2, ±4, ±6}**, e4m3 per 16 (best of the 70 subsets of the E2M1 grid tried) | 3.5 | **0.1522 / 0.1520** | 3-bit → 4-bit table (one `prmt` per 8 values) | **native**: the values *are* NVFP4 values |
| trellis (EXL3/QTIP), 3.0–3.4 | 3.0–3.42 + ~0.02 | not measured here; scouting: KL 0.063 at 3.42 bpw data-free, inside NVFP4's 0.051–0.075 on the same harness | see below | decode tiles, then MMA (see below) |
| NVFP4 + 2-bit mixtures | per expert/projection | 2-bit (±0.5, ±1.5) at 2.5 bpw: 0.317 | | native |

Readings:
- At 3.5 bpw the E2M1-grid codes have ~1.6× NVFP4's error, ~2.6× its squared error. They keep every
  NVFP4 kernel path, but the scout's same-harness data favours trellis by a wide margin per bit.
- Weight RMS is not KL. Ranking formats needs the KL harness (§ 6, experiment 3).

**Trellis decode cost in a bandwidth-bound GEMV** (*estimate*). At 235 GB/s and 3.25 bpw a box must
decode 578 G weights/s, ~12 G a second per SM (48 SMs), ~4.7 a clock at 2.55 GHz. A bitshift trellis
decodes each weight independently:
- a funnel shift for the 16-bit state;
- the hash ("1MAD"/"3INST": a multiply-add, mask and add, or 3 logic ops);
- a conversion to f16;
- the FMA;
- about 6 instructions a weight.

That is ~28 of 128 lanes a clock, ~25–45 % of issue, with the inputs rotated once per token (a
Hadamard transform of 6,144 values: negligible). The DeepSeek Huffman decoder failed for the opposite
reason, a serial dependency: "v2 decodes 4.0–4.6 G bytes/s an SM against ~5.8 needed", and "At one row
it is still 10–58 % slower than streaming the raw bytes, on every site"
(engine.md:4983–4985, the Part 2 table in [experiments.md](../../ds41f-4xgx10-switchless/docs/experiments.md)).
Trellis has no such dependency, but the ALU share must be measured (§ 6, experiment 1).

**Prefill for trellis.** Decoded values are f16 numbers off the FP4 and FP8 grids. Three options:
- f16 MMA (`mma.sync … f16`, f32 accumulate) with exact products; its rate on sm_121a is unmeasured;
- rounding each decoded tile to e4m3 with a block scale for `kind::mxf8f6f4`, which adds an error of
  ~0.026 relative RMS (the FP8 release's) and would have to be the reference's definition in decode
  too;
- a codebook restricted to FP8 values.

The tile decode is amortised over the chunk's rows, so prefill speed is the MMA rate's question.

**Integer 3-bit with group scales** sits on the E2M1 grid. Its decode is cheap, but it is worse than
the E2M1 subset at the same bytes, and the GGUF-class formats score 0.10–0.14 KL at 3.6–3.9 bpw on
llama.cpp's harness (scouting § 5.1).

### 3.4 (c) Hybrids by expert usage

Spending 4.5 (or trellis K4) bits on the experts a workload routes to most, and 3 on the rest, puts
more of the decode bytes and more of the routing mass on the better format. The pay-off depends on
how skewed routing is on agentic text. GLM-5.3 balances experts without an auxiliary loss (the
`e_score_correction_bias`), so usage is near-uniform on the training mix and skewed per domain by an
unknown amount. Ways to measure it without the GPU:
- **Router replay on a reference's hidden states**: needs the hidden states, which a CPU reference
  produces. A layer-streaming torch forward (one layer's weights at a time, DeepSeek's `ref_run.py`
  pattern) reads the NVFP4 checkpoint once per pass (432 GiB: ~4 min at rank 0's USB4 SSD's ~2 GB/s,
  less from rank 2's NVMe) and does ~80 GFLOP a token. 1.12 TFLOPS was measured for a BF16 matmul
  on 8 of rank 0's 20 cores (`torch` 2048×6144×2048, `taskset -c 0-7`, 2 s, 2026-10-13), so ~0.08 s a
  token: **32k tokens of agentic transcripts in ~45 min of compute** (*estimate*). Keeping one layer's
  experts and the activations under ~10 GiB means dequantising one expert at a time.
- **The same pass yields** the per-expert routing counts, per-expert input second moments (an
  imatrix), and the golden logits (§ 5). It costs no GPU window.
- **Deferring to the engine**: a short-context A2 build (§ 3.2) could log routing while serving real
  sessions. That needs the GPU and the ring.

Proposed: measure on the CPU pass (§ 6, experiment 2). Adopt a hybrid only if the top quarter of
experts carries ≥ 40 % of the routing mass (*a threshold to argue over*). Otherwise allocate bits by
measured sensitivity (KL per expert group), not usage.

### 3.5 (d) Lossless packing of NVFP4

`tools/nvfp4_entropy.py NV`: layers 3, 10, 30, 50 and 77, experts 0, 85, 170 and 255, all three
projections (60 tensors, 425 MB read, 4 s):

| | gate | up | down | all |
|---|---:|---:|---:|---:|
| H(nibble), bits | 3.968 | 3.968 | 3.992 | 3.980 |
| H(nibble ∣ its group's scale) | 3.958 | 3.959 | 3.976 | 3.971 |
| H(scale byte), bits | 4.68 | 4.70 | 4.70 | 4.74 |
| H(scale ∣ previous scale in the row) | 3.79 | 3.80 | 3.90 | 3.84 |
| fixed-length offset from the row's smallest scale, mean bits | 4.69 | 4.69 | 4.37 | |
| rows whose scales fit 5 / 6 bits from the row minimum | 100 / 100 % | 100 / 100 % | 98.67 / 100 % | |
| distinct scale bytes | 102 | 102 | 70 | |

The most common scale bytes are 113, 114 and 112 (11.1, 9.3 and 9.1 %).

- **Values: nothing to gain.** 3.98 bits of 4 (0.5 %), unlike DeepSeek's MXFP4 (3.894, 4.6 %), because
  NVFP4's e4m3 scale per 16 fits each block more tightly.
- **Scales: about half.** An exact entropy coder at ~3.84 bits would save 4.16 of every 8. The scales
  are 10.55 GiB a rank, so that is **5.5 GiB a rank**, and 0.18 GB a token (0.8 ms). But the DeepSeek
  engine measured that "Two exact fused Huffman decoders are 10–58 % slower than today's kernels at
  one row" ("Entropy-coded FP8: no", engine.md § Lossless weight compression).
- **A fixed-length code is what it adopted** (its experts' ue8m0 scales at 1 bit, merged as master
  2812072, −1.97 % at k = 1). Here: a base byte per row plus **6-bit offsets** (100 % of the sample)
  saves 2/8 of the scales, **2.6 GiB a rank** and 0.09 GB (0.4 ms) a token. A per-tensor choice of 5
  bits for gate/up raises that to ~3.6 GiB.
- It is exact by construction, and the converter must check every row. It applies equally to any
  format with e4m3 group scales (the int3 and E2M1-subset codes above).

### 3.6 (e) The non-expert tensors

| Option | Saves a rank | Quality risk |
|---|---:|---|
| attention, shared, indexer, MTP linears at **zai's FP8 release values** (128×128 f32 scales) | attention 3.00 GiB, shared 0.66, indexer 0.18; decode −3.93 GB/token in all (−16.7 ms) | the publisher's own release. The whole FP8 release scores KL 0.022 against BF16 on the common harness (scouting § 2.3), and these classes are a part of that. Codes are verified identical to e4m3(master) (§ 3.0) |
| o_proj at NVFP4 or the E2M1 subset | 0.80 GiB, −0.86 GB/token (−3.7 ms) over FP8 | no published measurement; 4-bit attention output projections are commonly sensitive (*unmeasured here*) |
| all attention at NVFP4 | 1.31 GiB, −1.41 GB (−6.0 ms) over FP8 | larger; q_b and kv_b feed the attention logits |
| head at FP8 | 0.22 GiB, −0.24 GB (−1.0 ms) | every logit; changes the argmax at close margins. DeepSeek kept its head bf16 (DESIGN decision 11) |
| embed vocab-split (one row a token, then a 12 KB all-gather) | 1.33 GiB | none (exact); ~12 µs a token |
| embed left on NVMe, rows read with O_DIRECT | 1.77 GiB | none (exact); a host read on the critical path at each step's start |
| dense MLP from the official block FP8 instead of NVIDIA's per-tensor FP8 | 0 | official block FP8 is closer to the master (*inferred*: finer scales); NVIDIA's differs from it by 0.037 relative RMS |
| routers and norms | ~0 | keep BF16/f32: routing flips dominate this model's KL floor (scouting: 0.11 between two exact BF16 runs) |

## 4. Producing a requantisation

- **Source.** The BF16 mirror `zai-org/GLM-5.3-BF16` at revision `9d2398f478cab2de883137db3a36ad2c96205e24`
  (1,403.22 GiB, 282 shards, 1,506,693,048,350 bytes) is the master (§ 3.0).
  - Requantising from the FP8 release (703.77 GiB) starts 0.022 KL from BF16, a floor no later step
    can remove.
  - Requantising from NVFP4 compounds a 0.095 relative error.
  - Only the routed experts need fetching: every non-expert tensor is already in the NVFP4 checkpoint
    as the master (verified on 4 tensors). 724.8B × 2 B = **1.45 TB, ~16.8 h at 24 MB/s**. The FP8
    release would be 8.7 h.
- **Where.** Nothing needs the whole master at once. One MoE layer's experts are 19.3 GB (BF16), and
  the converter fetches a layer's tensors by range request and deletes them once quantised.
  - **Rank 2** (1.2 TB free) is the place, also for the 50 NVFP4 shards' copy (432 GiB) used by the
    CPU reference.
  - The output is ~320 GiB in all at the recommended layout, **~86 GiB a rank**. Rank 0's /data (103
    GB free) holds its rank file.
  - **Rank 1 (74 GB free) cannot hold a ~92 GB rank file**: next to the DeepSeek rank file it needs
    ~20 GB freed, or the DeepSeek file moved. Rank 3 (727 GB) is fine.
  - The 544 GB of GLM-5.3-Flash checkpoints on rank 0 are the user's to delete.
- **Calibration.**
  - Data-free trellis (H = I) needs none. The scout found it already inside NVFP4's range at 3.42 bpw.
  - Hessian (LDLQ/GPTQ) calibration buys ~10 % (scouting). It needs each layer's inputs: from the CPU
    reference pass (§ 3.4), or from exllamav3's own layer-by-layer forward on the GPU.
  - Data: agentic transcripts like the DeepSeek golden `long` (system prompt, tools, tool results,
    thinking), plus code and prose; ~32k–64k tokens.
- **Tools.**

| Tool | Fit here |
|---|---|
| exllamav3 `convert.py` (EXL3 trellis) | done on one Spark for this model (drowzeys, scouting § 5.4); layer by layer, resumable; GPU. Mixed K per expert needs a patched loader |
| kindlingai's exlr8 pack (every expert at K2/K3/K4) | no requantisation: download a profile (`bit338` 322 GB, ~3.7 h) |
| llama.cpp `quantize` + published imatrix | CPU, hours; the worst quality per bit on every comparison found |
| llm-compressor (`model_free_ptq` RTN; sequential GPTQ) | RTN in ~30 min and 10 GiB (Tech2wild); GPTQ needs one layer on the GPU |
| ModelOpt | built for B300 nodes; reproduces the published NVFP4, which § 3.0 shows is plain round-to-nearest anyway |
| own code | E2M1-subset and int3 codes are ~200 lines of torch (round-to-nearest or GPTQ per expert). The CPU reference pass doubles as the calibration source |

- **Time on a GB10** (*estimates*).
  - Data-free trellis: each tensor is independent, seconds each on the GPU, so it fits short
    preemptible windows of ~1 layer (768 tensors).
  - Hessian trellis: the forward per layer plus the LDLQ; exllamav3 on a Spark took "days" (scouting,
    unverified).
  - Own E2M1/int3 round-to-nearest on the CPU: ~1 h for all experts, bound by the download.
- **Reusing other people's packs**, against § 5's standard.
  - **kindlingai's exlr8, or anyone's EXL3**: an artefact we did not make. Level 1 holds if we pin its
    revision and hash every file, and its byte-level spec makes a reference decoder possible. But we
    cannot reproduce it from the master, and its quality numbers are its harness's.
  - **exllamav3's converter run by us**: reproducible if the converter's commit, the master's
    revision, the calibration set and its order, the seeds, and the converter's determinism (to be
    checked by converting one tensor twice) are recorded. Our standard prefers this, or own code.
- **The artefact to record.**
  - the master's revision, and the SHA-256 of every source tensor read (computed while streaming);
  - the converter's commit and arguments, the calibration set's token file and SHA-256;
  - per output tensor, its SHA-256 and its relative RMS error against the master;
  - per rank file, the DeepSeek-style TOC with CRCs and the whole file's SHA-256;
  - all of this in a manifest committed with the configuration;
  - check after conversion that a second conversion of 3 sampled layers is byte-identical.

## 5. Exactness for this configuration

**What the DeepSeek configuration did.**
- **Golden data**: the publisher's `model.py` run teacher-forced in torch, one layer at a time, on two
  sequences (prose 135 tokens, an agentic transcript of 28,489), with logits, top-k and logprobs saved
  ([golden/README.md](../../ds41f-4xgx10-switchless/golden/README.md)). Engine against reference was
  judged by agreement statistics, not bits (e.g. "Top-1 agreement with golden's argmax is 0.902 on the
  first 512 positions").
- **refhash**: FNV-1a over the engine's own chosen tokens and logits for 8 pinned runs. Every change
  must reproduce them ("exact-output work only").
- **CED**: a sanctioned approximation, measured by KL against the exact forward (0.0004–0.012 nats at
  the first sampled token; [ced.md](../../ds41f-4xgx10-switchless/docs/ced.md)).
- **Weights**: never requantised, only losslessly recoded (DESIGN decision 11).

**What carries over:** refhash and its rule; the layer-streaming torch reference; KL measurement of
every sanctioned approximation. **What cannot:** "weights only shrink losslessly". The model must be
quantised to fit (§ 1.3), so the quantised weights become the model, and the standard has to say which
weights and which arithmetic.

**Proposed standard: three levels.**

1. **Artefact.** The served weights are a fixed set with a manifest (§ 4), made deterministically from
   the hashed BF16 master by a recorded tool, or adopted by hash with the reason recorded. Any lossless
   recoding (the scale codes of § 3.5) is checked at load against the manifest.
2. **Numerics contract and engine exactness.** A written contract fixes:
   - the dequantisation of each format;
   - **activations at BF16 between operations with f32 accumulation, and no FP4 activation
     quantisation.** The NVFP4 recipe's FP4 activations (model.md § 5.2) are a speed option for
     prefill. They change outputs and probably worsen them (*unmeasured*), and NVIDIA's reported
     quality includes them. If adopted, they enter the contract with KL measured;
   - the KV format (BF16, or FP8 656 B with KL measured);
   - the indexer key precision;
   - top-k tie-breaking.

   A layer-streaming torch reference implements the contract (CPU, or GPU windows). The engine must:
   - (a) **be bit-stable against its pinned refhash runs** (any speed change keeps every hash); and
   - (b) **agree with the reference** on the golden sequences within tolerances set from the
     reference's own order-variance: run it twice with different reduction orders (f32 vs f64
     accumulation, split-K vs not), take the spread of top-1 agreement, |Δ logprob| and KL, and set
     the engine's bar at 2× that spread.

   Prefill and decode may differ in reduction order but not in the contract.
3. **Quality against the master.** One fixed harness:
   - KL(BF16 ‖ ours) and top-1 agreement on held-out agentic and code text, scored against BF16
     logits: brandonmusic's public BF16 captures, or malaiwah's sealed panel (scouting);
   - GSM8K-style accuracy at T = 0;
   - a needle at 128k/256k (the engine).

   **The anchor**: NVFP4 as published, dequantised and run through the same reference and harness. It
   is the configuration NVIDIA reports at parity with FP8 (MMLU-Pro 87.88 vs 87.81 and so on, model
   card) and scored 0.051–0.075 on the common panel. The bar: our layout's KL ≤ NVFP4-as-published's
   on that harness, top-1 no more than 0.5 points below it, and no task metric outside its run-to-run
   noise. Because two exact BF16 runs differ by 0.11 mean KL on long text (scouting), only paired,
   same-harness comparisons count.

The MTP layer sits outside level 3's bar: drafts are verified, so its precision changes speed, not
output. That only holds while the verifier is the target model. Acceptance against MTP precision is
unstudied (scouting § 4), so it is measured (§ 6).

## 6. Recommendation and next experiments

**Layout (target 86–87 GiB a rank):**

| Part | Choice | GiB a rank |
|---|---|---:|
| routed experts | mixed trellis K3/K4 at ~3.75 bpw average (K4 to down projections and the most sensitive experts), from the BF16 master. Fallback: the E2M1-subset 3-bit code mixed with NVFP4 (f = 0.25 at 3.75 bpw), which keeps the FP4 tensor-core path | 79.1 |
| attention, shared expert, indexer | zai's FP8 values, 128×128 blocks | 3.84 |
| dense MLPs | official block FP8 | 0.16 |
| head | BF16 | 0.44 |
| embed | vocab-split | 0.44 |
| routers, norms | as shipped | 0.22 |
| MTP layer | experts at NVFP4 by our round-to-nearest (or trellis K3), rest FP8 | 1.34 |
| NVFP4 scale codes (if the fallback) | −0.6 to −1.0 | |
| **total** | | **~85.5** |

The floor at one token a step: routed 2.65 + the rest of A′ (4.87) = 7.53 GB a rank → **32.0 ms a
token** (31 tok/s) at 235 GB/s, ~37 ms at the DeepSeek engine's measured 214 GB/s overall plus its
collective overhead (*estimate*). Drafting multiplies tokens per step; o_proj at 4 bits (−3.7 ms) is
the next lever and needs its own KL measurement.

**Experiments, in order:**

| # | What | Cost | Decides |
|---|---|---|---|
| 1 | **Trellis vs E2M1-subset decode GEMV microbenchmark** on one GB10: a K3 trellis GEMV (bitshift 16-bit state, 1MAD/3INST decode), the E2M1-subset GEMV and the NVFP4 GEMV on one layer's expert shapes, 1 and 6 rows; plus the f16 `mma.sync` rate for trellis prefill tiles | ~1 day of code, one ≤ 15-min preemptible GPU window on one box (`ring.sh`), no disk | whether trellis decodes at ≥ 95 % of the bandwidth floor and how fast its prefill can run on sm_121a, i.e. the expert format |
| 2 | **CPU layer-streaming reference** (own torch code, `model.md`'s math) on the NVFP4 checkpoint: 32k tokens of agentic transcripts, saving routing counts, per-expert imatrix, BF16-contract logits | copy NV to rank 2 (432 GiB, ~1 h over the LAN); ~1–2 h CPU a pass at nice 19 on 8 cores; ≤ 10 GiB RAM | expert-usage skew (hybrid or not), the golden data for level 2, the calibration statistics |
| 3 | **Format bake-off by KL**: in the same reference, layers 3–77's experts requantised on the fly from the master (range-fetched per layer, deleted after) into NVFP4-as-published, E2M1-subset, int3 and (via exllamav3's quantiser on a GPU window) trellis K3/K4, as parallel "tracks" of hidden state; KL against the BF16-master track | 16.8 h of download (once; ~20 GB on disk at a time), ~4× experiment 2's CPU time | the format and the bit allocation at level 3; NVFP4's KL on our harness as the anchor |
| 4 | **FP8 KV and FP4 activations** as tracks of experiment 3 | CPU only | whether either enters the contract |
| 5 | **MTP precision vs acceptance** (BF16 / NVFP4 / K3), drafts replayed in the reference | CPU, from experiment 2's hidden states | the MTP layer's bits (0.84–4.5 GiB a rank) |
| 6 | **Full conversion** at the chosen layout, with the manifest, a second conversion of 3 layers checked byte-identical, and the rank files cut | ~17 h download (or reuse experiment 3's per-layer pass), rank 2's disk; rank 1 needs ~20 GB freed | the artefact |

**First to run: experiment 1.** It is the cheapest test that can rule a format out: if a trellis
GEMV cannot stay near the bandwidth floor on GB10's 48 SMs, the format falls back to the E2M1-grid
codes, whatever their KL. Experiment 2 runs alongside it on the CPU.

## Record

- 2026-10-13: tally, entropy, provenance and 3-bit formats measured as above; all CPU, nothing on the
  GPUs. The sampled tensors fetched from Hugging Face (~222 MB) stayed in a scratch directory.
