# Scouting: GLM-5.3 on memory-limited hardware (2026-10-13)

What is published, as of 2026-10-13, for running the full **GLM-5.3** (`zai-org/GLM-5.3`, `glm_moe_dsa`, 753B
total / ~40B active; not GLM-5.3-Flash) on hardware that cannot hold its 704 GiB FP8 release: the release itself,
every quantised checkpoint found, who has run it on GB10-class and other clusters and how fast, speculative
decoding, and the quality-vs-bits evidence for re-quantising huge MoEs to 3–4.5 bits a weight.

**Method.** Model cards, configs, file sizes and commit histories were read from the Hugging Face API
(`/api/models/<repo>?blobs=true`, `/resolve/main/README.md`) and GitHub (`gh api`) on 2026-10-13; sizes below are
the sum of a repository's files at the revision named. Web search and two research agents covered forums,
issue trackers and the literature. No GPU job was run. Unless a row says otherwise, every number is the author's
own (self-reported); third-party measurements say so. **KL divergences from different harnesses are not
comparable** (the same checkpoint scores 0.024 on one and 0.063 on another, [§2.3](#23-one-harness-many-checkpoints));
compare only within a table that names one harness. "Inferred" marks our own arithmetic or reading.

## Verdict

**Bits against quality.** On GLM-5.3 itself the evidence is now dense, and it says three things.
1. **Four-bit-class experts are close to lossless on tasks.** The NVFP4 checkpoints' own evals sit within run noise
   of FP8 (NVIDIA: MMLU-Pro 87.81 → 87.88, Terminal-Bench 2.1 85.67 → 86.80), RedHat's IFEval (−4.1 points) aside; calibration-free MXFP4 is the
   exception (AIME25 98.3 → 91.3). On one common 25-context harness the FP8 release is itself **0.022** KL from BF16,
   unsloth's ~4.9 bpw GGUF 0.028, and the three NVFP4 checkpoints 0.051–0.075 ([§2.3](#23-one-harness-many-checkpoints)).
2. **Between 3 and 4 bits, the format matters more than the last half-bit.** On llama.cpp's harness, 3.6–3.9 bpw
   GGUFs score KLD 0.10–0.14, top-1 89–91 %, PPL +6–9 % ([§5.1](#51-glm-53-itself)). Trellis (EXL3/QTIP) does much
   better per bit: a *data-free* 3.42 bpw trellis lands inside the NVFP4 range on the common harness (0.063) at three
   quarters of the bytes; Hessian calibration buys a further ~10 %; spending extra bits on down projections and on the
   worst experts buys 5–10 %. Below ~3 bpw quality falls fast (KL roughly doubles from 3.25 to 2.25 bpw), and at 2 bits
   long greedy reasoning stops converging.
3. **KL on this model has a high floor**: two exact BF16 runs differing only in batching score 0.11 mean KL on long
   text (median 0.004), because top-8 routing flips. Mean KL from different harnesses differs 2–3× for the same
   checkpoint; only same-harness comparisons mean anything.

**Fastest published GB10-class setup** (single stream, four boxes, all with quantised experts):
- [kindlingai/gr8](https://github.com/kindlingai/gr8) (2026-10-13), a from-scratch CUDA engine on a 3.25 bpw trellis
  pack with FP8-class dense weights, MTP or DFlash2 drafts, RoCE through a switch: **RigMark code 49.3 / prose 30.0
  tok/s, cold prefill 1,374 tok/s at 8K**, 310K-token KV pool. Its ring mode costs ~10 % of decode.
- [knapcio](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4) (vLLM fork, INT4 RTN experts, NVFP4 non-experts,
  switched): **sparkDash prose 40.2 / code 46.6 / structured 49.8 / JSON 46.4**, RigMark 43.7 / 30.9, prefill 913–980.
- drowzeys' TensorFold at 2.75 bpw: code 56.3 / prose 38.3 on its own prompts (thinking on, greedy, 32K), and a
  1M-token context.
- For scale, this repository's DeepSeek-V4.1-Flash engine does 102.2 / 148.1 on sparkDash. GLM-5.3 moves ~40B active
  parameters a token, and at decode its **non-expert weights (13.1B attention + 2.8B shared + 1B head, all read every
  token) cost as many bytes as the eight routed experts** unless they too are compressed: ~26.7 GB a token at 3.25 bpw
  experts with FP8 non-experts against ~44.3 GB with BF16 non-experts, a k = 1 bound of ~35 vs ~21 tok/s at 4 × 235
  GB/s (inferred, [§2.2](#22-what-each-format-costs-on-this-fleet-inferred)). Every fast GB10 engine compresses them.

**What to start from.**
- Nothing published fits this budget at NVFP4: the 432 GiB NVFP4 checkpoints leave ~12 GB a box and are reported to
  OOM on four Sparks. NVIDIA's also carries the MTP layer's experts in BF16 (18.0 GiB), which the README's "355.5 GiB
  of E2M1" silently includes.
- The best-evidenced path inside ~3.5–3.9 bpw is **trellis-coded routed experts with per-expert K3/K4 widths, Hessian
  (LDLQ) calibration and FP8-class non-experts**, produced layer by layer with exllamav3's `convert.py` (done on a
  single Spark for this model) from the BF16 mirror (1,403 GiB) or the FP8 release (704 GiB, which starts 0.022 from
  BF16 on the common harness). kindlingai's exlr8 release already ships every expert at K2, K3 and K4 with
  per-half error tables and a byte-level spec, so a 3.38–4.0 bpw profile could be loaded without re-quantising
  (`bit338` is 322 GB and leaves ~3.5 GiB a box in gr8; `up1` at 4.0 bpw does not fit TP=4).
- Cheaper alternatives, with worse evidence: GGUF IQ4_XS / Q3_K_XL from the published imatrices (KLD 0.10–0.14 on
  llama.cpp's harness); RTN INT4 g128 experts (Tech2wild, 28 minutes and 10 GiB of RAM, no quality measured); or
  re-coding the NVFP4 experts already on disk (no published evidence for double quantisation).
- For validation without a BF16 run: brandonmusic's public BF16 full-logit captures and malaiwah's sealed panel.

## 1. The official release

| | |
|---|---|
| Repository | [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3), first commit 2026-08-27 (`7cda8193`), current `c5087a76` (2026-10-12, a model-card eval entry). Weights public 2026-08-29 per [Gigazine](https://gigazine.net/gsc_news/en/20260829-glm-5-3-open/); announced 2026-08-14 ([DataNorth](https://datanorth.ai/news/z-ai-releases-glm-5-3)) |
| Storage format | **FP8 E4M3, 128×128 block scales** (`quant_method: fp8`, `weight_block_size: [128, 128]`, dynamic activation scheme), with 541 modules left unconverted (every `mlp.gate` router and its `e_score_correction_bias`, every `self_attn.indexers_proj`, norms, `lm_head`): 751.2B FP8 parameters, 2.1B BF16 (config.json and the HF safetensors index at `c5087a76`) |
| Size | **755,663,689,422 bytes (703.8 GiB)**, 156 files |
| BF16 mirror | Official: [`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16), `9d2398f4` (2026-09-04), 1,506,693,048,350 bytes (1,403.2 GiB). Most community quantisations start from it ([§2](#2-quantised-checkpoints)) |
| Licence | "GLM-5.3 License": MIT-style permission to use, modify and redistribute, plus one condition: a Model-as-a-Service operator whose group revenue exceeds US$10 billion over 12 months must pass Z.ai's security review before commercial use ([LICENSE](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE)). Derivatives inherit it; NVIDIA's NVFP4 adds the NVIDIA Open Model Agreement on top ([card](https://huggingface.co/nvidia/GLM-5.3-NVFP4)) |
| Architecture (config.json) | 78 layers (3 dense, 75 MoE), hidden 6144, 64 heads, MLA (q_lora 2048, kv_lora 512, qk_nope 192 + rope 64, v 256), 256 routed experts (intermediate 2048) top-8 + 1 shared, sigmoid router with `noaux_tc` bias, `routed_scaling_factor` 2.5, vocab 154,880, rope θ 8,000,000, `max_position_embeddings` 1,048,576, `num_nextn_predict_layers` 1 (layer 78) |
| Tech report | [arXiv 2602.15763](https://arxiv.org/abs/2602.15763), "GLM-5: from Vibe Coding to Agentic Engineering" (2026-02); GLM-5.3 "uses the same base model as GLM-5.2 — every gain comes from post-training" (model card). No GLM-5.3-specific report was found |

**Base model and lineage.** GLM-5.3 = GLM-5.2's base + new post-training (card). This matters for reuse: GLM-5.2
checkpoints, quantisation recipes and fidelity tables (EXL3 TR3 lineage, unsloth/AesSedai/ubergarm GGUF for
GLM-5/5.1/5.2) are the nearest prior evidence.

**The DSA indexer.** DeepSeek Sparse Attention: a lightweight indexer (32 heads × 128 dims, `index_topk` 2048)
chooses the 2,048 past tokens each query attends to. GLM-5.2 added **IndexShare**
([HF paper 2603.12201](https://huggingface.co/papers/2603.12201)): one indexer is reused across every four sparse
attention layers, which the transformers documentation credits with 2.9× less per-token compute at 1M context
([glm_moe_dsa.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm_moe_dsa.md)).
In config.json this is `indexer_types` (21 `full`, 57 `shared`: layers 0, 1, 2 and every fourth layer from 6 run
the indexer, the rest reuse the top-k of the last full layer), `index_topk_freq` 4, `index_skip_topk_offset` 3 and
`index_share_for_mtp_iteration: true` (the MTP layer reuses the main model's top-k when it drafts). The
`indexers_proj` weights are kept BF16 in the FP8 release and in every quantisation read here; the
Int4-Int8Mix authors call the indexer and router "sacred" ([§2.1](#21-the-checkpoints)). Inferred from the config
names; the exact semantics of `index_skip_topk_offset` are not documented in anything read.

**MTP.** One next-token-prediction layer (layer 78, a full MoE layer with its own 256 experts, plus `eh_proj`,
`enorm`, `hnorm`). transformers does not implement it; vLLM and SGLang use it as an EAGLE-style drafter
([§4](#4-speculative-decoding)). GLM-5.2 "improve[d] the MTP layer … increasing the acceptance length by up to 20 %"
(transformers doc).

**Recommended stacks and settings** (model card): SGLang ([cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3)),
vLLM ([recipe](https://recipes.vllm.ai/zai-org/GLM-5.3), updated 2026-10-03), TokenSpeed, transformers, KTransformers
(its tutorial is GLM-5.2's), Unsloth, and vLLM-Ascend/xLLM/SGLang on Ascend. The vLLM recipe: vLLM ≥ 0.28 (badge says
0.29), TP=8 on H200/B200 with `--kv-cache-dtype fp8`, MTP with 5 speculative tokens, `--reasoning-parser glm45`
(the recipe also shows `glm47`), `--tool-call-parser glm47`; NVFP4 via `Inferact/GLM-5.3-NVFP4` at TP=8 + EP on
B200/B300; full 1M context on 8× B200 with FP8 KV. Nothing in either recipe covers GB10 / DGX Spark (the SGLang
cookbook has a `dgx-spark` hardware id only in its multi-node Docker flags).

**Sampling and thinking.** `generation_config.json` holds only the EOS ids 154820, 154827, 154829 (no sampling fields, checked 2026-10-13); the model card gives temperature 1.0, top-p 0.95.
Thinking is always on: `reasoning_effort` takes `low`, `high`, `max` and **defaults to `max`**; the chat template has
`clear_thinking` defaulting to false (pass true for chat). `enable_thinking=false` is not supported
([discussion #21](https://huggingface.co/zai-org/GLM-5.3/discussions/21), closed 2026-09-08). The chat template was
fixed on 2026-09-04 (`aca966e4`, "early break in tool result reordering check"); a pending PR speeds tool-result
validation ([#31](https://huggingface.co/zai-org/GLM-5.3/discussions/31), 2026-10-13). Card footnotes evaluate at
1M context with up to 128K output tokens.

**Long context.** 1,048,576 positions natively; the card's agentic evals run at 400K–1M context. Per-token KV is the
MLA latent (512 + 64 rope) plus the indexer key (128) per layer (inferred from config); the published GB10 engines
fit 0.3–1M tokens over four boxes with FP8 or int4 KV ([§3](#3-who-has-run-it-and-how-fast)).

## 2. Quantised checkpoints

Over 700 Hugging Face repositories match "GLM-5.3"; about 150 are the full model rather than Flash (search API,
2026-10-13). The table keeps every one with a quality figure, a GB10 recipe, or a distinct format; abliterated,
"uncensored" and fine-tuned derivatives are left out. Sizes are file totals at the revision given. "Experts bpw" is
the routed-expert width the author states, scales included where the author includes them.

### 2.1 The checkpoints

**NVFP4 (4.5 bpw experts: E2M1 + an E4M3 scale per 16, plus a per-tensor FP32 scale).** All quantise only the
routed experts of layers 3–77 (57,600 matrices, 724.8B of the 753.3B parameters) and keep attention, the indexer,
the shared experts and the head in BF16; all are ~432–433 GiB, which at TP=4 is ~108 GiB a box before KV
(inferred).

| Checkpoint (revision, date) | Size | Producer, calibration | Reported quality | Loader |
|---|---|---|---|---|
| [nvidia/GLM-5.3-NVFP4](https://huggingface.co/nvidia/GLM-5.3-NVFP4) (`e3b8f9f2`, 2026-09-14, card dated 2026-09-10) | 432.3 GiB | ModelOpt 0.47.0, W4A4; calibration cnn_dailymail + Nemotron-Post-Training-v2. Dense MLPs (layers 0–2) FP8; **the MTP layer (`model.layers.78*`) excluded, so its 256 experts are BF16 (18.0 GiB)**; KV FP8 | FP8 → NVFP4, SGLang TP=8 on B300, T 1.0, top-p 0.95, effort max: MMLU-Pro 87.81 → 87.88, GPQA-D 91.79 → 92.68, LCB v6 52.95 → 53.55, SciCode 58.65 → 58.03, AIME25 99.38 → 99.53, AA-LCR 69.75 → 69.63, IFBench 65.33 → 66.00, Terminal-Bench 2.1 85.67 → 86.80. All within noise | SGLang `--quantization modelopt_fp4` |
| [RadixArk/GLM-5.3-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-NVFP4) (`6e389189`, 2026-08-28) | 432.9 GiB | ModelOpt 0.47.0.dev91 from BF16; 1,024 × 512-token samples (cnn_dailymail + Nemotron v2); static activation scales; MTP BF16 | GSM8K 97.42 %, AIME26 pass@1 94.17 %, "exact match" / "within noise" of BF16 under the same protocol. Third party: KL 0.051, top-1 93.6 % ([§2.3](#23-one-harness-many-checkpoints)) | SGLang (cookbook) |
| [Inferact/GLM-5.3-NVFP4](https://huggingface.co/Inferact/GLM-5.3-NVFP4) (`ce67b36f`, 2026-08-28) | 432.9 GiB | From BF16; no method on the card; the vLLM recipe's NVFP4 | None on the card. Third party: KL 0.075, top-1 92.4 % | vLLM |
| [incoai/GLM-5.3-NVFP4](https://huggingface.co/incoai/GLM-5.3-NVFP4) (`54e52520`, 2026-08-27) | 432.9 GiB | In-house PTQ, ModelOpt format, static NVFP4 activations, FP8 KV with unit scales | FP8 → NVFP4: GPQA-D 91.1 → 91.2, AIME25 94.3 → 95.1, MATH-500 95.6 → 95.2, HLE 35.9 → 35.2, AA-LCR 73.6 → 73.0. Third party: KL 0.059, top-1 93.3 % | SGLang, vLLM ≥ 0.28 |
| [RedHatAI/GLM-5.3-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-NVFP4) (`c8917e42`, 2026-09-21) | 432.9 GiB | llm-compressor, "experimental layerwise decompression/compression", `perfectblend` calibration; compressed-tensors | Against FP8, same vLLM harness on 4× B200, 3–8 seeds: IFEval 89.77 → 85.70 (95.5 %), GSM8K-Platinum 99.1 %, MATH-500 101.3 %, AIME25 92.50 → 89.58 (96.8 %), GPQA-D 97.8 % | vLLM |
| [modal-labs/GLM-5.3-NVFP4](https://huggingface.co/modal-labs/GLM-5.3-NVFP4) (2026-08-28) | 432.9 GiB | ModelOpt max calibration, 1,024 × 1,024 tokens over text, chat, code, SWE, tool use, math | None found | SGLang |
| [local-inference-lab/GLM-5.3-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4) (`b472e4ee`, 2026-09-16) | 433.0 GiB | ModelOpt from BF16, calibration with natural top-k routing (three passes: agentic code, WildChat/LMSYS, >8K long context) | None found. Its [NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF) twin stores the scale planes losslessly compressed (45.3 → 23.3 GB; 412.5 GiB total) under a restrictive licence (no re-uploads, attribution) | vLLM (b12x MoE runner) |
| [nota-ai/…-NVFP4-Global-Pruned-17.75](https://huggingface.co/nota-ai/GLM-5.3-Nota-NVFP4-Global-Pruned-17.75) (2026-08-31) | 365.5 GiB | NVFP4 + 17.75 % of routed experts pruned (208–256 per layer); patched vLLM | Terminal-Bench 2.1 and DeepSWE "within sampling noise" (94.3 / 95.2 % relative) | patched vLLM |

**Known problems.** Tech2wild reports that the ~465 GB (432 GiB) NVFP4 checkpoints "OOM in practice" on four
DGX Sparks: ~116 GB a rank leaves ~12 GB for KV and runtime ([card](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix),
2026-08-29). RadixArk [discussion #1](https://huggingface.co/RadixArk/GLM-5.3-NVFP4/discussions/1) (2026-09-01)
reported looping output in over 1 % of replies; the comment is now hidden. Note for our README (inferred): the
"355.5 GiB of E2M1 values" for the NVIDIA checkpoint equals 337.5 GiB of NVFP4 values for layers 3–77
(724.8B × 0.5 bytes) **plus the MTP layer's 18.0 GiB of BF16 experts**; the layer-78 experts are not NVFP4.

**MXFP4 (4.25 bpw: E2M1 + an E8M0 scale per 32).**

| Checkpoint | Size | Method | Reported quality | Loader |
|---|---|---|---|---|
| [RedHatAI/GLM-5.3-MXFP4](https://huggingface.co/RedHatAI/GLM-5.3-MXFP4) (`88118422`, 2026-09-16) | 375.7 GiB | llm-compressor `model_free_ptq` from the FP8 release (FP8 → BF16 → MXFP4), **no calibration**, all linear layers except router, head, embeddings, `eh_proj`, indexer `weights_proj`/`wk`; W4A4 | Against FP8 (3 seeds): IFEval 99.26 %, GSM8K-P 98.73 %, MATH-500 98.26 %, **AIME25 98.33 → 91.25 (92.8 %)**, GPQA-D 92.42 → 89.06 (96.4 %) | vLLM |
| [amd/GLM-5.3-Quark-MXFP4-AttnFP8](https://huggingface.co/amd/GLM-5.3-Quark-MXFP4-AttnFP8) (`4992911b`, 2026-09-02) | 381.4 GiB | Quark 0.12; experts and shared experts MXFP4, attention FP8 block | GSM8K 91.89 → 91.28 (99.34 %) | vLLM ROCm (MI350/355) |
| [OneNexus/GLM-5.3-MXFP4](https://huggingface.co/OneNexus/GLM-5.3-MXFP4) (`eab18c67`, 2026-09-09) | 408.0 GiB | Quark from BF16, routed experts only, MTP and dense/shared BF16; "Strong7": folded SmoothQuant on 7 experts of layer 6 | DeepSWE v1.1 70.6 ± 4.9 (FP8) → 68.9 ± 4.0; sgl-eval GSM8K/MMLU/GPQA at T 0: 99.62 % of BF16's correct count, 98.9–99.1 % answer agreement | SGLang ROCm |

**INT4 / mixed integer (compressed-tensors, GPTQ, AWQ, RTN).**

| Checkpoint | Size | Method | Reported quality | Loader, notes |
|---|---|---|---|---|
| [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix) (`206507bb`, 2026-08-29) | 377.4 GiB | QuantTrio's GLM-5.2 layer map, **data-free RTN**, symmetric, g128: experts W4A16, attention/dense/shared W8A16, MTP W8A16 channelwise; layer 0, routers, indexer, head BF16. **Shard-streaming quantiser: one BF16 shard at a time, peak RAM ~10 GiB, 28.2 min** | "Coherence proven; quality-vs-FP8 parity NOT measured" (card) | vLLM 0.23.1rc1 + sm12x kernel overlays from [tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s); serves on four Sparks ([§3](#3-who-has-run-it-and-how-fast)) |
| [HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128](https://huggingface.co/HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128) (2026-10-09) | 381.3 GiB | GPTQ on BF16 activations (7 documents, 40,960 tokens); per-projection 3/4/6-bit mix (10.2 / 87.9 / 1.3 %), 4.05 bpw with scales | Own harness: KL to BF16 0.0409 vs 0.0505 for INT4/INT8 RTN g64 (−19 %), top-1 95.79 vs 95.22 %; a uniform 4-bit GPTQ control ties the mix on a second set (0.0328 vs 0.0326) | vLLM-style compressed-tensors |
| [gpustack/GLM-5.3-W4A8](https://huggingface.co/gpustack/GLM-5.3-W4A8) (2026-09-02) | 372.3 GiB | Experts INT4 g128, RTN + per-group MSE clip search; activations and non-expert FP8 | GPQA-D pass@1 (198 × 4) 89.65 vs FP8 90.28; perplexity 2.442 vs 2.445 | vLLM/SGLang, Hopper CUTLASS W4A8 |
| [PhalaCloud/GLM-5.3-W4AFP8](https://huggingface.co/PhalaCloud/GLM-5.3-W4AFP8) (2026-08-28) | 372.3 GiB | AWQ-calibrated INT4 g128 experts, FP8 elsewhere | GPQA-D 91.92 % (gpustack's card; different protocol) | SGLang Hopper |
| [JANGQ-AI/GLM-5.3-W4A16](https://huggingface.co/JANGQ-AI/GLM-5.3-W4A16) | 420.0 GiB | Data-free RTN INT4 g32 | None | vLLM |
| [cyankiwi/GLM-5.3-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-AWQ-INT4) | 454.7 GiB | AWQ | None found | vLLM |
| [Justvugg/GLM-5.3-colibri-int4-g64](https://huggingface.co/Justvugg/GLM-5.3-colibri-int4-g64) | 390.5 GiB | INT4 g64 for the "colibri" streaming runtime (~14 GB resident, experts paged from disk) | None | colibri |

**GGUF (llama.cpp, ik_llama.cpp, antirez's ds4).**

| Checkpoint | Levels and sizes | Quality | Notes |
|---|---|---|---|
| [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF) (`346b3591`, 2026-08-29), imatrix on `unsloth_calibration_GLM-5.3.txt` | UD-IQ1_S 201.8, UD-IQ1_M 212.8, UD-IQ2_M 222.2, UD-Q2_K_XL 236.4, UD-IQ3_XXS 262.3, **UD-Q3_K_XL 319.4**, **UD-IQ4_XS 340.2**, UD-Q4_K_XL 435.2, UD-Q5_K_XL 523.8, UD-Q6_K_XL 637.4, Q8_0 746.3, BF16 1,404.4 GiB | Third party: UD-Q4_K_XL KL 0.0275, top-1 95.4 % ([§2.3](#23-one-harness-many-checkpoints)). Unsloth's own per-level table: [§5](#5-quality-versus-bits-for-huge-moes) | Tool calls printed as raw XML in Q4_K_XL / Q3_K_XL until the first shard was re-uploaded ([discussion #2](https://huggingface.co/unsloth/GLM-5.3-GGUF/discussions/2), fixed 2026-08-29) |
| [AesSedai/GLM-5.3-GGUF](https://huggingface.co/AesSedai/GLM-5.3-GGUF) (2026-09-03), from BF16, imatrix | Q8_0 745.8 (8.50 BPW), Q5_K_M 523.1 (5.96), Q4_K_M 436.9 (4.98), **IQ4_XS 342.0 (3.90)**, IQ3_S 265.9 (3.03), IQ2_S 241.3 (2.75) GiB; non-FFN kept Q8_0 (Q6_K below 3.9 BPW) | llama.cpp `llama-perplexity --kl-divergence` against BF16 (corpus not stated): KLD 0.0131 / 0.0203 / 0.0371 / **0.1016** / 0.2752 / 0.3742; PPL +0.02 / +0.24 / +0.94 / **+6.23** / +21.8 / +31.3 % | Gate and up at one type below down. The only single-harness GGUF ladder for GLM-5.3 found |
| [DevQuasar/zai-org.GLM-5.3-GGUF](https://huggingface.co/DevQuasar/zai-org.GLM-5.3-GGUF) | IQ1_S 152.6 … Q3_K_M 334.1, Q4_K_M 423.3 … Q6_K 576.0 GiB | None | Needed llama.cpp PR #27754 at the time |
| [antirez/glm-5.3-gguf](https://huggingface.co/antirez/glm-5.3-gguf) (2026-08-31) | UD-IQ2_XXS (routed) + Q2_K block 78: 196.6 GiB | None on the card | For antirez's ds4 engine; [bibproj Q4K-ds4](https://huggingface.co/bibproj/GLM-5.3-Q4K-ds4-gguf) 404.4 GiB, no imatrix |
| [sokann/GLM-5.3-GGUF](https://huggingface.co/sokann/GLM-5.3-GGUF) | IQ2_KS 2.300 bpw, 201.7 GiB (ik_llama.cpp types) | None | ubergarm has no GLM-5.3 repository; his GLM-5 / GLM-5.1 tables are in [§5](#5-quality-versus-bits-for-huge-moes) |
| [autotrust/GLM-5.3-GGUF-DGX-Spark](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark) (`dc02b401`, 2026-10-11) | 149.7 GiB: **64 of 256 experts pruned per layer**, IQ2_XXS (2.06 bpw) experts, Q8_0 rest, no MTP | Pruning (FP8): GPQA-D 86.9 → 83.3, C-Eval 92.0 → 84.7; at 2 bits 27 of 92 reasoning items run out of a 16K budget | Two Sparks over llama.cpp RPC "not run on Spark hardware" by the author |
| [emwesoft/GLM-5.3-NVFP4-MTP-GGUF](https://huggingface.co/emwesoft/GLM-5.3-NVFP4-MTP-GGUF) (2026-08-29) | 433.1 GiB GGUF of incoai's NVFP4, MTP block 78 BF16 (18.54 GiB) | Speculation figures in [§4](#4-speculative-decoding) | llama.cpp NVFP4 GGUF |

**EXL3 / QTIP trellis.** Trellis codes (each 16×16 tile a Viterbi path through a 16-bit-state trellis, K bits a
weight) with Hadamard rotations, from exllamav3. Stock exllamav3 does not load GLM-5.3 (vcruz305's card); every
row needs a fork or its own engine.

| Checkpoint (date) | Experts bpw, size | Method | Quality (own harness unless noted) | Loader |
|---|---|---|---|---|
| [kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) (`f459f5d0`, v0.5, 2026-10-11) | Every expert stored at K2, K3 **and** K4 (780 GiB repo); a load-time manifest picks widths per expert half: 2.25 (219 GB) … `home`/`bit325` 3.25 (310 GB) … `up1` 4.0 (378 GB). Codebook rounded to E4M3 inside the search, so decoded weights are exact FP8 | Hessian-aware LDLQ, ~2M calibration tokens (977 × 2048), 64 held-out rows; dense trellis K4 / int4 g32, q_a/kv_a/indexer MXFP8, kv_b FP8 block, head/embedding FP8 per row | Full-vocab KL vs BF16, held-out text mean / median: 2.25 0.229 / 0.031; 3.25 0.147 / 0.012; 3.38 0.142 / 0.011; **4.0 0.127 / 0.008**; all-K3 0.156; all-K2 0.266. **Two BF16 runs differing only in batching score 0.11 / 0.004 against each other** on the same text, so the mean is dominated by routing flips. davidsyoung's 3.25 scores 0.161 here; `bit325` scores 0.025 on "the four-window panel" | [kindlingai/gr8](https://github.com/kindlingai/gr8) |
| [davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw) / [3.25](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw) / [3.0](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw) (2026-08-28/29) | 3.42 (330.9 GiB), 3.25 (316.2), 3.0 (294.8); per layer the experts with the worst K3 round-trip error at K4 | **Data-free** (identity Hessian), deterministic seeds; MTP experts quantised too; dense, attention, head BF16 | brandonmusic's BF16 teacher logits (4 windows × 2,047 positions): **3.42 0.0241, 3.25 0.0261** with FP8 KV, reproduced within ±5 % on other hardware; NVFP4 KV adds ~0.014 (more than the 3.25 → 3.42 step). Third party ([§2.3](#23-one-harness-many-checkpoints)): 0.063 / 0.073 / 0.084 | exllamav3-b12x / "sparkinfer" lineage vLLM image; "a stock loader … will produce fluent garbage" |
| [drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW) (`ae7b6967`, 2026-10-12) | 2.75 mean (5,436 experts at 2 bits, 13,128 at 3, 636 at 4); attention, shared, dense at EXL3 5 bpw; 257.9 GiB | Per-expert widths by measured layer-output error on calibration activations | exllamav3 `model_diff`, 65,536 tokens: KL 0.124 (median 0.017, p90 0.30), top-1 89.6 %, PPL 2.938 vs 2.750 | TensorFold fork; vLLM image (not public) |
| [drowzeys/keys-GLM-5.3-EXL3](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3) (2026-09-03) | 3.00 uniform, 307.5 GiB | exllamav3 `convert.py` from the FP8 release, **on one Spark, two layers at a time with a rolling 5-shard window** | Third party: KL 0.102 | Mia-AiLab vLLM container + patches |
| [Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold) (`2d747d0e`, 2026-10-03) | 2.75 mean (2/3/4 per projection), MTP experts 4 bpw, rest BF16; 272.7 GiB | Calibrated "for how TensorFold serves it" | 64 × 2,048 tokens vs BF16: KL 0.086 experts-only, 0.124 as served (TensorFold's q4 dense on top); top-1 92.8 / 90.8 %; by set: math 0.012, code 0.035, chat 0.043, wiki 0.126, **tool calls 0.201** | TensorFold |
| [vcruz305/GLM-5.3-EXL3-3.38bpw](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) (`ca06664b`, 2026-10-06) | 3.38 ("SAGE MixedK"), 8-bit head; 297.2 GiB; no MTP | | 10 × 1,024 held-out tokens: KL 0.0948 (median 0.0018, p99 1.61), top-1 92.98 %, PPL +2.1 % | TensorFold fork |
| [0xSero/GLM-5.3-EXL3-3.0bpw](https://huggingface.co/0xSero/GLM-5.3-EXL3-3.0bpw) (2026-09-07) and REAP-pruned cuts | 3.0, 272.6 GiB; REAP cuts keep 224 … 168 experts | | 25-prompt panel: 0.089 unpruned; 0.195 at 224 experts, 0.361 at 192, 0.511 at 168. **Its 4.0 bpw sibling scores KL 1.02 (broken encode, "do not use")** | exllamav3 / TabbyAPI |
| [wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1) (2026-08-30), [MikeRoz 4.04bpw-h8](https://huggingface.co/MikeRoz/GLM-5.3-4.04bpw-h8-exl3) | 4.0, 367.0 GiB; 4.04, 359.1 GiB | | Third party: K4 0.045 | |

**MLX (Apple).** [mlx-community/GLM-5.3-4bit](https://huggingface.co/mlx-community/GLM-5.3-4bit) 389.6 GiB,
[mixed-4_5bit](https://huggingface.co/mlx-community/GLM-5.3-mixed-4_5bit) 419.8 GiB (both 2026-09-02);
[pipenetwork](https://huggingface.co/pipenetwork/GLM-5.3-MLX-mixed-3_6bit)'s ladder on wikitext-2 (288,627 tokens,
one harness): 4-bit 2.8636, mixed 4/8 2.7420 (8-bit non-experts, 2.5 % of parameters, are "worth their 8 bits":
−4.3 % PPL for +9 GB), mixed 3/6 (332.6 GB) 3.0338, REAP25-4bit 3.2872, REAP50-4bit 5.0295.
[inferencerlabs Q4i](https://huggingface.co/inferencerlabs/GLM-5.3-MLX-Q4i) reports token accuracy 89.75 % for
plain Q4 against 97.70 % for its "Q4-INF" and 97.65 % for Q8 (M3 Ultra, its own harness).

### 2.2 What each format costs on this fleet (inferred)

Routed experts of layers 3–77 are 724.8B parameters. At *b* bits a weight they take 724.8e9 × *b* / 8 bytes:
2.75 → 232.0 GiB (58.0 a rank at TP=4), 3.25 → 274.2 (68.6), 3.5 → 295.3 (73.8), 3.9 → 329.1 (82.3), 4.0 → 337.5
(84.4), MXFP4 4.25 → 358.6 (89.6), NVFP4 4.5 → 379.7 (94.9). Everything else in BF16 is ~33 GiB plus the MTP
layer (18 GiB of BF16 experts). gr8 reports ~83 GB a rank at TP=4 for its 3.25 bpw `home` with FP8 dense
weights; drowzeys' vLLM path reports 67.5 GiB a rank at 2.75 bpw.

### 2.3 One harness, many checkpoints

The only common-harness comparison found: malaiwah's [quant-fidelity-registry](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry)
(`394b6475`, 2026-09-10; [suite](https://github.com/malaiwah/quant-fidelity-suite)). Panel `corpus5x5-v1`: 25 contexts,
51,175 scored positions, full-vocabulary KL(BF16 ‖ candidate) in float64, against a sealed BF16 GLM-5.3 capture;
each candidate dequantised to BF16 and run on the same engine, so **weights only** (NVFP4 activation quantisation
and KV quantisation are not in these numbers). 95 % intervals are window-block bootstrap; the author warns that
per-window scatter exceeds adjacent gaps and orders only by paired tests.

| Artifact (GLM-5.3 unless noted) | Experts bpw | KL mean [95 % CI] | Top-1 |
|---|---|---|---|
| zai-org FP8 release, dequantised | 8 (FP8 block) | **0.0223** [0.0189, 0.0289] | 95.6 % |
| unsloth UD-Q4_K_XL (GGUF) | ~4.9 (Q4_K/Q5_K, Q8_0 rest) | 0.0275 [0.0229, 0.0353] | 95.4 % |
| wrldsuksgo2mars EXL3 K4 | 4.0 | 0.0448 [0.0371, 0.0597] | 94.0 % |
| RadixArk NVFP4 | 4.5 | 0.0511 [0.0421, 0.0654] | 93.6 % |
| (GLM-5.2, nvidia NVFP4) | 4.5 | 0.0548 [0.0451, 0.0697] | 93.4 % |
| incoai NVFP4 | 4.5 | 0.0594 [0.0489, 0.0770] | 93.3 % |
| davidsyoung EXL3 TR3 3.42 (data-free) | 3.42 | 0.0628 [0.0510, 0.0821] | 93.1 % |
| davidsyoung EXL3 TR3 3.25 (data-free) | 3.25 | 0.0731 [0.0598, 0.0939] | 92.6 % |
| Inferact NVFP4 | 4.5 | 0.0754 [0.0606, 0.1037] | 92.4 % |
| davidsyoung EXL3 TR3 3.0 (data-free) | 3.0 | 0.0838 [0.0685, 0.1093] | 92.1 % |
| drowzeys keys EXL3 (calibrated, from FP8) | 3.0 | 0.1023 [0.0839, 0.1335] | 91.1 % |

Readings (ours): (1) the FP8 release is itself 0.022 from BF16, so nothing that starts from the FP8 weights can
score below that floor on this panel, and the best 4-bit quantisations sit 1.3–2.5× above it; (2) the three NVFP4
checkpoints of the same format span 0.051–0.075, so calibration matters as much as the 3.25 → 4.5 bit step;
(3) a data-free trellis at 3.42 bpw lands inside the NVFP4 range at three quarters of the bytes; (4) the nvidia
GLM-5.3 NVFP4 (released 2026-09-14, after the registry's last update) is not measured; its GLM-5.2 predecessor
scored 0.055. Separately, kindlingai's batching experiment (0.11 KL between two BF16 runs on its own harness) shows
that teacher-forced KL on this model mixes quantisation error with routing flips that any numerically different
but exact implementation also produces.

## 3. Who has run it, and how fast

All figures are single-stream decode (c1) unless marked, at temperature 0 / greedy unless marked. The harnesses
differ: [RigMark](https://github.com/alexellis/rigmark) 1.0.0 runs with thinking on at effort `low`; sparkDash (as in
the DeepSeek configuration's comparisons) runs thinking off, 256 tokens; the rest are each project's own prompts.
Compare within a column only. None of these engines is exact to the official weights: all run quantised experts.

### 3.1 Four GB10 boxes (the fleet's class)

| Project (revision, date) | Weights, engine, fabric | RigMark c1, code / prose / structured | sparkDash c1, prose / code / structured / JSON | Other decode | Cold prefill | Context |
|---|---|---|---|---|---|---|
| [kindlingai/gr8](https://github.com/kindlingai/gr8) (`246b04b1`, 2026-10-13) | exlr8 trellis, experts 3.25 bpw (`home`), dense K4 trellis / MXFP8 / FP8; from-scratch CUDA, own RoCE verbs fabric; a 200G switch (`--shape ring` without one costs ~10 % of c1 decode); clocks capped 2200 MHz; MTP or DFlash2 per request | **49.3** / **30.0** / – (41.5–49.3 code across boots) | – | aggregate 50.8 / 64.0 at 2 / 4 streams | 1,374 tok/s at 8K; cached 54,867 | KV pool 310,528 tokens (415,232 with int4 KV) |
| [knapcio/GLM-5.3-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4) (2026-10-13 boot) | Tech2wild's Int4-Int8Mix experts (INT4 g128 RTN, Marlin W4A16) with NVFP4 "sidecars" for attention, shared, dense and MTP; vLLM fork; native MTP K3; FP4x KV; switched RoCE (b12x/RoCEnante) | 43.66 / 30.90 / 48.47 | **40.19** / **46.56** / **49.77** / **46.38** | c4 aggregate prose 66.4, code 72.0 | 980 / 929 / 875 / 881 at 4K / 8K / 16K / 32K | 262,144 |
| [drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark), [checkpoint card](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW) (image 2026-10-05) | EXL3 2.75 bpw mixed-K; [TensorFold](https://github.com/ashhart/TensorFold) PR #159 fork; MTP 2 or DFlash2 | – | – | Thinking on, greedy, 300 tokens: prose 40.2, code 39.0; DFlash2 short context 36.1 / **47.7**; sampled T 1.0 at 32K: MTP 31.7 / 34.9, DFlash2 33.4 / 39.4; 4 streams 78.7 / 100.2 aggregate; 1M context 37.0 → 33.7 at 115K → 905K | 1,038 at 128K | 1M (decode context parallel) |
| drowzeys, [MLX-serve variant](https://github.com/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks) (2026-10-10) | Same 2.75 bpw, **abliterated** weights; 6-bit head | – | – | Thinking on (effort high), greedy, 512 tokens at 32K: prose 38.3, **code 56.3** | ~370–375 | 1M (DCP) |
| [bertholomus](https://forums.developer.nvidia.com/t/full-glm-5-3-753b-on-4x-dgx-spark-3-bit-exl3-quant-tensorfold-tp4-recipe/385042) (2026-10-04, repo `9a5c1100`) | EXL3 3.04 bpw, TensorFold, MTP 3 | ~43 / ~28 (agent's reading; thinking state not stated) | – | 4 code streams 82 aggregate; KL 0.109, top-1 90.2 % on wikitext-2 | – | – |
| [vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe), [card](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) (2026-10-06) | EXL3 3.38 bpw, TensorFold fork, DFlash2 | – | – | Greedy, 512 tokens: mean of 6 real prompts 41.1 (163,840 context), peak 69, math 56.5; **no drafter 26.5**; stock exllamav3 12.2–15.7 | – | 262,144 with int4 KV |
| [tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark) (`f5440eff`, 2026-10-06); [forum](https://forums.developer.nvidia.com/t/glm-5-3-743b-int4-int8mix-on-4x-dgx-spark-gb10-tp4-up-to-46-tok-s-200k-ctx/381755) | Int4-Int8Mix, vLLM + sm12x sparse-MLA overlays, switched | – | – | 2026-08-29, MTP 4, thinking off: 12.1 c1, 46.0 aggregate at c6. 2026-10-06, DCP4 + DFlash2 k 7, effort low: code 39.9, JSON 36.0, prose 17.0 | ~395–416 (DCP4) | 512K (needle at 490K) |
| [ajclark](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark) (`1e342e27`, 2026-10-06) | Int4-Int8Mix shape on an **abliterated graft**, vLLM 0.29.0, DFlash2, **switchless RoCE ring** with custom RDMA collectives, NVMe KV tier, clocks ~2000 MHz | – | – | Greedy: prose 26.7, code 42.8, replayed agent turns 33.0; 4 requests 53.7 / 73.3 | 759 / 747 / 740 at 4K / 32K / 60K | – |
| [Leoid/glm53-spark-tp4-recipe](https://github.com/Leoid/glm53-spark-tp4-recipe) (`4c03eb97`, 2026-10-03) | drowzeys' 3.0 bpw abliterated, exllamav3 1.4.5, MTP 3 | – | thinking off, 256 tokens: prose 15.0, code 25.5, JSON 20.3 | c8 code 70.7 aggregate | 549 / 466 / 552 | 200K |
| [kindlingai/glm-5.3-full-exl3-tp6](https://github.com/kindlingai/glm-5.3-full-exl3-tp6) (`3b9c548c`, 2026-10-03) | davidsyoung's TR3 3.25 on the MiaAI vLLM image (gr8's predecessor) | – | – | prose TP4 19.8 | 783 / 769 at 8K / 32K | – |

Two- and three-box and eight-box runs, for scale: MiaAI-Lab's [three-Spark TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold)
with its 2.75 bpw pack, sampled, 163,840 window: code 39.5–40.4, prose 29.1–30.2, prefill 638–656 at 8K–32K, "drafted
and concurrent replies equal serial ones, 12/12" ([card](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold), 2026-10-03);
gr8 at TP=3 with 2.25 bpw: RigMark 38.6 / 29.0, prefill 628; gr8 at TP=6 (MTP only, clocks uncapped): code 38.2,
prefill 1,953; drowzeys on **two** Sparks at 2.0 bpw: ~12 tok/s, and MTP produced garbage at that width
([repo](https://github.com/drowzeys/keys-Dual-DGX-Sparks-GLM5.3-Full.EXL3-Derisked.2.0BPW-128k.CTX), 2026-08-31);
SBE247Emperor's **native FP8 on eight Sparks** (vLLM fork, FLASHINFER_MLA_SPARSE_SM120, MTP 2, switched): best c1
27.6 tok/s, 3,364 of 3,364 probes over a 7-day soak ([repo](https://github.com/SBE247Emperor/glm53-fp8-8x-dgx-spark), `b07e5f6f`, 2026-10-01).
No NVFP4 checkpoint has been run on four Sparks (it does not fit with KV; [§2.1](#21-the-checkpoints)); the only
Spark run found is one box swapping, 0.81 tok/s ([oakmindai](https://huggingface.co/oakmindai/GLM-5.3-NVFP4-FTW)).

**Not found:** no full-GLM-5.3 recipe from Alex Ellis (his switchless recipes and RigMark are used by others; his
own GLM work is Flash-only), from MiaAI-Lab beyond the three-Spark one, from christopherowen, eugr's
spark-vllm-docker, sparkinfer or NVIDIA's DGX Spark playbooks. ajclark's is the only switchless-ring full GLM-5.3
recipe found; gr8 also supports a ring.

### 3.2 Other hardware

| Setup | Source | Result |
|---|---|---|
| 4× RTX PRO 6000 (96 GB) | [malaiwah/glm52-exl3-vast](https://github.com/malaiwah/glm52-exl3-vast) (2026-08-29 qualification of davidsyoung's 3.42 bpw) | vLLM TP4 + DCP4, MTP 3, NVFP4 KV, 393K envelope: c1 60.4 tok/s sampled (T 1, 512 tokens), 227.6 aggregate at c8; prefill 2,465 at 8K, 2,282 at 128K. The author says these do not carry over to the current image |
| 2× RTX PRO 6000 + 4× RTX 3090 + 251 GB RAM, llama.cpp | [emwesoft card](https://huggingface.co/emwesoft/GLM-5.3-NVFP4-MTP-GGUF) (2026-08-29) | NVFP4 GGUF with experts partly on CPU, 400K context: ~10 tok/s, 9.3–12.2 with MTP |
| One B200, llama.cpp | [autotrust card](https://huggingface.co/autotrust/GLM-5.3-GGUF-DGX-Spark) | Pruned IQ2_XXS 149.7 GiB: 42 tok/s c1, 177 aggregate at 32; pp512 730–760 |
| Mac M3 Ultra 512 GB, MLX | [PipeNetwork/glm53-mlx](https://github.com/PipeNetwork/glm53-mlx) (`79f89ca6`, 2026-08-29) | Perplexity only, no tok/s. **Stock mlx-lm builds an indexer on all 78 layers; only 21 ship weights, so with `strict=False` 57 load random, silently wrong past 2,048 tokens** |
| Mac M4 Max 128 GB, llama.cpp CPU | [llama.cpp #29010](https://github.com/ggml-org/llama.cpp/issues/29010) (2026-09-17, open) | UD-Q4_K_XL with `-ngl 0` emits only "@" |
| 6 Sparks, llama.cpp RPC | [llama.cpp #28047](https://github.com/ggml-org/llama.cpp/issues/28047) (closed) | UD-Q5_K_XL crashed with ≥ 2 RPC workers on the DSA top-k view |
| exo, mlx-distributed, ktransformers, ik_llama.cpp, prima.cpp | – | No GLM-5.3 (full) speed found. ik_llama.cpp fixed garbage aarch64 CPU decode for GLM-5.2/5.3 (fp16 block-scale products) in [PR #2503](https://github.com/ikawrakow/ik_llama.cpp/issues/2498) (2026-09-22) |

**llama.cpp support.** `glm-dsa` got the full/shared indexer schedule in [PR #25407](https://github.com/ggml-org/llama.cpp/pull/25407)
(merged 2026-07-24; at first it masked attention and was slower than dense: prefill −14 to −24.5 %, decode −6.5 to
−10 %), then true index-gathered sparse flash attention on CUDA in [PR #27970](https://github.com/ggml-org/llama.cpp/pull/27970)
(~2026-09-02), active only above a KV-length threshold and without a quantised-KV path
([#28454](https://github.com/ggml-org/llama.cpp/issues/28454)). Generic MTP landed in [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
(2026-05-16); that it works for glm-dsa is unverified.

### 3.3 Known problems on GB10 (from the trackers and recipes)

- Stock vLLM has no sparse-MLA backend for sm_121 ([vllm #45317](https://github.com/vllm-project/vllm/issues/45317), open; fix [#46055](https://github.com/vllm-project/vllm/pull/46055) open). Every vLLM recipe above carries overlays (CosmicRaisins' sm12x Triton kernels, b12x) or FLASHINFER_MLA_SPARSE_SM120, which hangs on padded CUDA-graph decode batches with speculation ([flashinfer #5015](https://github.com/flashinfer-ai/flashinfer/issues/5015); vLLM fix [#51538](https://github.com/vllm-project/vllm/pull/51538)).
- The prefill `persistent_topk` for top-k 2048 oversubscribes GB10's 48 SMs (drowzeys' [keys-GLM-5.3-EXL3 card](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3)).
- Loaders that ignore `indexer_types` give 57 layers random indexers: right below 2,048 tokens, wrong beyond (PipeNetwork for mlx-lm; [CosmicRaisins](https://github.com/CosmicRaisins/glm-5.2-gb10) for GLM-5.2 configs).
- Run-to-run nondeterminism from `VLLM_MARLIN_USE_ATOMIC_ADD=1` (3 of 8 temperature-0 runs matched for tonyd2wild); knapcio's deterministic MoE alignment gives 30 of 30.
- GB10 operations, as the recipes report them: set `vm.compaction_proactiveness=0` (gr8, drowzeys: background compaction near the memory limit stalls collectives); page cache eats CUDA-visible memory (tonyd2wild flushes every 60 s; drowzeys drops shards from the cache as they load); a GPU can stay at ~720 MHz after a crash until reboot (tonyd2wild); gr8 reboots once if the first 8K prefill is under ~580 tok/s.

## 4. Speculative decoding

**The built-in MTP layer.** Layer 78 is a complete MoE layer (256 experts + shared, its own attention) that takes
the last hidden state and the embedding of the next token (`eh_proj` over `enorm`/`hnorm`) and predicts one token
further; run recursively it drafts *k* tokens. With `index_share_for_mtp_iteration` it reuses the main model's
indexer top-k (config). vLLM runs it as `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`
([recipe](https://recipes.vllm.ai/zai-org/GLM-5.3)); SGLang as EAGLE (`--speculative-algorithm EAGLE`, e.g. 3 steps,
topk 1, 4 draft tokens on [OneNexus's MI350 recipe](https://huggingface.co/OneNexus/GLM-5.3-MXFP4); 5 steps / 6
draft tokens on [RadixArk's card](https://huggingface.co/RadixArk/GLM-5.3-NVFP4)). It costs a full MoE layer of
weights: 18.0 GiB of BF16 experts in the NVIDIA checkpoint (inferred), which is why GB10 packs quantise it (K3 in
exlr8, 4 bpw in Mia-AiLab's pack, 8 bpw in drowzeys', W8A16 in Int4-Int8Mix) or leave it out (vcruz305, autotrust).

**Reported acceptance** (mean accepted length a verification step, including the bonus token unless noted):

| Drafter | Setting | Acceptance | Source |
|---|---|---|---|
| MTP, 5 drafts | 8× MI325X / MI350X, ~500-token greedy generation | 2.77 / 3.19 | [vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3) (2026-10-03) |
| MTP, 7 drafts | FP8 target (hardware not stated), GSM8K / MATH-500 / HumanEval / MBPP / MT-Bench | 5.12 / 5.05 / 4.85 / 4.34 / 3.81 | [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) |
| **DFlash2**, 7 drafts (block diffusion, 4.6 GiB BF16) | same | **5.94 / 6.02 / 5.48 / 4.95 / 4.19**; c1 throughput 113 → 293 (MTP) → 367 tok/s (DFlash2) on GSM8K | same; **licence CC BY-NC-ND 4.0** (no redistribution of converted copies; gr8 and drowzeys convert at load) |
| DSpark (RedHat speculators, 3 layers, taps at layers 2/20/39/58/75, block 8) | 8× B300, TP=4 | HumanEval 4.76, math 6.13, tool calls 4.01, QA 3.18, writing 3.19; 524K–1M MRCR 4.29 (41 % per token) | [RedHatAI/GLM-5.3-speculator.dspark](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) (2026-09-23), GLM-5.3 licence |
| DSpark (AlayaNeW, 5 layers, block 8) | SGLang/vLLM, FP8 target | not stated on the card | [AlayaNeW/GLM-5.3-DSpark](https://huggingface.co/AlayaNeW/GLM-5.3-DSpark) (2026-09-24), MIT |
| DFlash (modal-labs), for their NVFP4 target | SGLang ≥ 0.5.19, block 8 | not stated; "keep the draft model unquantized. Quantizing it lowers the accept length" | [modal-labs/GLM-5.3-NVFP4-DFlash](https://huggingface.co/modal-labs/GLM-5.3-NVFP4-DFlash) (2026-09-20) |
| MTP `n-max 3` / DFlash2 Q8_0 `n-max 4`, llama.cpp | NVFP4 GGUF, partial CPU offload | 74.9 % (mean 3.24) / 67.7 % (mean 3.69) | [emwesoft card](https://huggingface.co/emwesoft/GLM-5.3-NVFP4-MTP-GGUF) (2026-08-29) |

**On GB10 with low-bit targets.** Drafting pays less here than on B200 because a GB10 step is long and the
verification of extra rows is cheap only up to a point; the published GB10 engines choose per request:
- gr8 uses MTP (k = 2) or, at TP=4 only, DFlash2 for a lone request when its model of step cost and acceptance
  predicts more tokens per ms; "DFlash2 wins on code, MTP on prose" ([README](https://github.com/kindlingai/gr8)).
  DFlash2's 8 KV heads do not split three or six ways, so TP=3/6 are MTP only.
- drowzeys' TensorFold at 2.75 bpw: MTP 2 against DFlash2, thinking on, greedy short context: prose 37.7 vs 36.1,
  code 41.4 vs 47.7 tok/s; sampled at 32K: 31.7 vs 33.4, 34.9 vs 39.4 ([card](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW)).
- vcruz305 at 3.38 bpw: 26.5 tok/s without a drafter against 41.1 with DFlash2 (mean of 6 prompts), so drafting
  is worth ~1.55× there ([card](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw)).
- **Drafting against a quantised target.** drowzeys fine-tuned RedHat's DSpark on 1.33M tokens of the 2.75 bpw
  target's own outputs ("ft2"): against built-in MTP on the same weights, prose +4 / +5 / +9 % and code +15 / +12 /
  +19 % at short / 32K / ~120K context, with a confidence policy (a cost policy calibrated at context 0 drafts too
  deep at long context) ([drowzeys/keys-GLM-5.3-speculator.dspark-ft2](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2),
  2026-10-07). An on-policy fine-tune of DFlash2 to the quantised target gave +10.5 % prose / +7.8 % code at 32K,
  but cannot be published under DFlash2's licence (drowzeys' card). At 2.0 bpw on two Sparks the MTP layer produced
  garbage ([§3.1](#31-four-gb10-boxes-the-fleets-class)).
- No published study of the MTP layer's own quantisation (BF16 vs FP8 vs 4-bit) on acceptance was found. knapcio
  runs the MTP layer as NVFP4 with a draft-only NVFP4 head and an FP8 `eh_proj`, checked for exact target output
  ([README](https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4)); acceptance is not reported separately.

## 5. Quality versus bits for huge MoEs

Every table names its harness; numbers from different tables must not be compared with each other.

### 5.1 GLM-5.3 itself

**llama.cpp `--kl-divergence`, wiki.test.raw, n_ctx 512, 565 chunks, against BF16.** AesSedai's card
([AesSedai/GLM-5.3-GGUF](https://huggingface.co/AesSedai/GLM-5.3-GGUF), 2026-09-03) and unsloth's guide
([unsloth.ai/docs/models/glm-5.3](https://unsloth.ai/docs/models/glm-5.3), "as of Aug 2026"; method not stated, but its
UD-IQ4_XS row matches AesSedai's IQ4_XS to the third digit, so probably the same harness: inferred). bpw over all
753.3B parameters, from the HF file sizes.

| Quant | GiB | bpw | Mean KLD | 99.9 % KLD | Top-1 | PPL ratio |
|---|---|---|---|---|---|---|
| UD-IQ2_M | 222 | 2.53 | 0.454 | 7.72 | 78.5 % | 1.40 |
| UD-Q2_K_XL | 236 | 2.70 | 0.374 | 7.08 | 80.9 % | 1.31 |
| UD-IQ3_XXS | 262 | 2.99 | 0.273 | 6.25 | 84.2 % | 1.21 |
| AesSedai IQ3_S | 266 | 3.03 | 0.275 | – | – | 1.218 |
| **UD-Q3_K_XL** | 319 | 3.64 | **0.141** | 4.13 | 88.9 % | 1.09 |
| **AesSedai IQ4_XS** (experts IQ3_S / IQ3_S / IQ4_XS) | 342 | 3.90 | **0.1016** | 3.11 | 90.5 % | 1.062 |
| UD-IQ4_XS | 340 | 3.88 | 0.1015 | 3.18 | 90.6 % | 1.064 |
| UD-Q4_K_XL | 435 | 4.96 | 0.0369 | 1.31 | 94.3 % | 1.010 |
| AesSedai Q8_0 | 746 | 8.50 | 0.0131 | – | – | 1.0002 |

**Trellis, long held-out documents, full vocabulary** (kindlingai exlr8, [card](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4)
v0.5, 2026-10-11; 131,072 held-out positions + 65,536 wikitext; calibrated LDLQ): 2.25 bpw 0.229 / 0.284 (held-out /
wikitext mean), 2.60 0.192 / 0.231, all-K3 3.0 0.156 / 0.120, 3.25 0.147 / 0.097, 3.38 0.142 / 0.092, all-K4 4.0
0.127 / 0.051; medians 0.031 → 0.008. Against this harness's own floor (two BF16 runs differing in batching: 0.11
mean, 0.004 median), the held-out means above 3 bpw are mostly routing noise and the medians and wikitext column
carry the signal. davidsyoung's data-free 3.25 scores 0.161 here against 0.147 for the calibrated `bit325`:
**calibration buys ~10 % KL at equal bits**; stepping down projections before gate/up and K2→K3 before K3→K4 buys
another 5–10 % at equal size.

**Short full-logit panel** (brandonmusic's public BF16 teacher logits,
[GLM-5.3-BF16-full-logits](https://huggingface.co/datasets/brandonmusic/GLM-5.3-BF16-full-logits), 4 × 2,047 positions):
data-free trellis 3.42 bpw 0.0241, 3.25 0.0261 with FP8 KV; NVFP4 KV adds ~0.012–0.014; `bit325` 0.025
([davidsyoung](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw), 2026-08-29; kindlingai). The common
25-context panel is [§2.3](#23-one-harness-many-checkpoints).

**Task benchmarks at ~4–4.5 bits.** Every NVFP4 card (NVIDIA, RadixArk, incoai, RedHat) reports parity with FP8 within
run noise on reasoning/coding sets, except RedHat's IFEval (−4.1 points); calibration-free MXFP4 (RedHat) lost 7.1
points on AIME25 and 3.4 on GPQA; Quark MXFP4 with SmoothQuant on 7 experts (OneNexus) held DeepSWE and 99.6 % of
sgl-eval ([§2.1](#21-the-checkpoints)). No task benchmark was found for any GLM-5.3 quantisation between 2.75 and 4
bpw: autotrust's pruned 2-bit runs out of a 16K reasoning budget on 27 of 92 items, and its card and drowzeys'
recommend sampling or a budget because "at 2 bits the model occasionally loops in very long greedy chains of thought".

**A possible reason 4-bit is near-lossless** (unverified): the GLM-5 report ([arXiv 2602.15763](https://arxiv.org/abs/2602.15763)
§2.4.3) says INT4 QAT was applied during SFT; Z.ai said on 2026-04-05 it has no plans to release INT4 weights
([GLM-5 #21](https://github.com/zai-org/GLM-5/issues/21)). Whether GLM-5.3's post-training kept it is ubergarm's
speculation on his GLM-5.1 card, not confirmed.

### 5.2 Other huge MoEs

| Model, harness | bpw → quality | Source, date |
|---|---|---|
| GLM-5, ik_llama.cpp PPL ratio (wiki, ctx 512) | IQ3_KS 3.65 → 1.059; IQ2_KL 2.99 → 1.149; smol-IQ2_KS 2.34 → 1.437; smol-IQ1_KT 1.93 → 1.750 | [ubergarm/GLM-5-GGUF](https://huggingface.co/ubergarm/GLM-5-GGUF), 2026-02-14 |
| GLM-5.1, same | smol-IQ4_K 4.62 → 1.011; IQ3_KS 3.65 → 1.056; IQ2_KL 2.99 → 1.147 | [ubergarm/GLM-5.1-GGUF](https://huggingface.co/ubergarm/GLM-5.1-GGUF), 2026-04-07 |
| GLM-4.7, same | IQ4_K 5.02 → 1.016; IQ3_KS 3.72 → 1.053; IQ2_KL 3.10 → 1.162 | ubergarm/GLM-4.7-GGUF, 2025-12 |
| DeepSeek-V3.1-Terminus, same | IQ4_K 4.90 → 1.008; smol-IQ4_KSS 4.08 → 1.015; IQ3_K 3.72 → 1.025; IQ3_KS 3.55 → 1.037; IQ2_KL 2.96 → 1.094 | ubergarm/DeepSeek-V3.1-Terminus-GGUF, 2025-09 |
| Kimi-K2.5, same (vs Q4_X) | IQ3_K 3.85 → 1.030; smol-IQ3_KS 3.25 → 1.073; smol-IQ2_KL 2.76 → 1.196 | ubergarm/Kimi-K2.5-GGUF, 2026-02 |
| GLM-4.6, exllamav3 `model_diff` KL / top-1 | 3.0 0.326 / 84.1 %; 3.84 tuned 0.159 / 88.8 %; 4.0 0.156 / 89.7 %; 4.16 tuned 0.133 / 90.6 %; 5.0 0.113 / 91.7 %. Promoting shared experts and attention to 6–8 bits gave "a very large improvement" | mratsim/GLM-4.6-EXL3, 2025-11 |
| GLM-5.2 EXL3, KV format | willfalco 3.0 / 3.25 / 3.36 bpw: NVFP4 KV 0.1195 / 0.0960 / 0.0778 vs FP8 KV 0.1025 / 0.0877 / 0.0685; brandonmusic 3.0 bpw, 5 runs: 0.1124 ± 0.0026 vs 0.1037 ± 0.0018 | willfalco / brandonmusic GLM-5.2 cards, 2026-07 |
| DeepSeek-V3.1, Aider Polyglot (unsloth dynamic GGUF) | thinking: full 76.1, 3-bit 75.6, 2-bit 66.7, 1-bit 57.8; non-thinking: full 71.6, 5-bit 70.7, 4-bit 69.7, 3-bit 68.4, 2-bit 65.8 | [unsloth](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot), 2025-09-10 |
| DeepSeek-R1, dynamic 3-bit | DQ3_K_M ≈ Q4_K_M (AIME24 57.1 / 53.3, FP8 57.9; table garbled in extraction, check the PDF) | [arXiv 2505.02390](https://arxiv.org/abs/2505.02390), 2025-05 |

GLM-5.x loses more per bit than DeepSeek on the same harness (PPL ratio 1.056 vs 1.037 at ~3.6 bpw). ubergarm's
GLM IQ3_KS recipe: attention q8_0/iq6_k, shared and dense iq5_ks, routed down iq4_ks, routed gate/up iq3_ks
(routed ≈ 3.4 bpw).

### 5.3 Methods

- **Trellis (QTIP / EXL3).** QTIP ([arXiv 2406.11235](https://arxiv.org/abs/2406.11235)), Llama-2-70B wikitext ctx 4096,
  FP16 3.12: 4-bit QTIP 3.16 / QuIP# 3.18 / AQLM 3.19; 3-bit 3.27 / 3.35 / 3.36. Llama-3-70B 3-bit: QTIP 3.18,
  QuIP# 3.59 (BF16 2.59). On GLM-5.3 it is the format every GB10 engine but one chose ([§3.1](#31-four-gb10-boxes-the-fleets-class)).
- **NVFP4 vs MXFP4** (MR-GPTQ, [arXiv 2509.23202](https://arxiv.org/abs/2509.23202), ICLR 2026; Llama-3.1-8B recovery):
  W4A16 NVFP4 RTN 98.0 / GPTQ 98.0 / AWQ 98.1, MXFP4 RTN 95.6 / GPTQ 96.8; W4A4 NVFP4 94.7 → 96.1 with MR-GPTQ, MXFP4
  87.8 → 93.3. A Hadamard rotation hurts NVFP4 RTN (98.0 → 96.8) because its group of 16 already absorbs outliers.
  Consistent with GLM-5.3: RTN MXFP4 lost points that calibrated NVFP4 did not.
- **Four Over Six** ([arXiv 2512.02010](https://arxiv.org/abs/2512.02010)): per-block choice of a ±4 or ±6 NVFP4
  scale; only training-loss gains are claimed, no PTQ numbers found.
- **MoE-specific mixed precision** (all on small MoEs; none at GLM scale): MxMoE ([2505.05799](https://arxiv.org/abs/2505.05799),
  2.4 lower PPL than GPTQ at 2.25 bit), MoEQuant ([2505.03804](https://arxiv.org/abs/2505.03804)), EAQuant
  ([2506.13329](https://arxiv.org/abs/2506.13329)), QuantMoE-Bench ([2406.08155](https://arxiv.org/abs/2406.08155),
  frequency-based 4/2-bit mixing). On GLM-5.3 itself the per-expert allocations that exist (davidsyoung: worst K3
  round-trip error at K4; drowzeys: layer-output error on calibration activations; kindlingai: per-half error tables
  and a knapsack) all allocate by measured error, not by routing frequency; kindlingai rules out shipping
  frequency-based manifests because "hot sets of different workloads barely overlap". Expert pruning costs far more
  than bits: REAP to 224 of 256 experts doubles KL at 3 bpw (0.089 → 0.195, [0xSero](https://huggingface.co/0xSero/GLM-5.3-EXL3-3.0bpw)).
- **"NVFP4 + 2-bit" mixtures.** Not found published for GLM-5.3. The closest are exlr8's K2/K3/K4 per-expert
  mixtures and HawkBearPig's 3/4/6-bit GPTQ mix (KL −19 % against uniform RTN INT4, but tied with a calibrated uniform
  4-bit control).
- **Non-expert precision.** pipenetwork (MLX): 8-bit instead of 4-bit non-experts, 2.5 % of the parameters, cut
  PPL 4.3 %; mratsim (GLM-4.6): 6–8-bit attention and shared experts gave "a very large improvement"; Mia-AiLab:
  TensorFold's q4 dense weights raise KL from 0.086 to 0.124; drowzeys: 5-bit EXL3 non-experts tie FP8 (PPL 3.560 vs
  3.567). Every 4-GB10 engine keeps the router, norms and indexer weights in BF16/FP32.
- **KV cache.** FP8 KV is what every GLM-5.3 deployment uses. NVFP4/4-bit KV costs measurable KL on GLM-5.2/5.3
  trellis packs (above: +0.009–0.017), which on the 4-window panel is more than the 3.25 → 3.42 weight step. gr8's int4
  KV is "about 3 % slower" with needles still found; no KL figure.
- **AQLM, PV-tuning, VPTQ, AutoRound with tuning**: no GLM-5.x results; OPEA's AutoRound INT2 of DeepSeek-R1 needed
  5 × 80 GB GPUs and 1.4–1.6 TB of RAM.

### 5.4 What can be produced layer by layer in ~100 GiB of RAM with one GB10

| Method | Streams the checkpoint? | Calibration | Feasibility (inferred unless sourced) |
|---|---|---|---|
| EXL3 trellis via exllamav3 `convert.py` | Yes, layer by layer, resumable (`-r`) | Forward of each layer on calibration rows, or data-free H = I | **Done on one Spark** for GLM-5.3 by drowzeys: two layers at a time, a rolling 5-shard window of the FP8 release, peak disk ~660 GB ([card](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3)). One BF16 MoE layer is ~19 GiB. Mixed K per expert needs a patched loader (davidsyoung, exlr8). Time on a GB10: days (unverified) |
| GGUF `llama-quantize` with an imatrix | Per tensor, mmap | Needs an imatrix | Published GLM-5.3 imatrices (unsloth, AesSedai, ~1.1 GiB) make it CPU-only, hours. Gives the 5.1 GGUF quality curve, which is worse per bit than trellis on every comparison found |
| RTN INT4 / NVFP4A16 / MXFP4 (compressed-tensors, llm-compressor `model_free_ptq`) | Per shard | None | Tech2wild: 28.2 min, ~10 GiB peak RAM for the whole model ([card](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix)). RTN is ~98 % of GPTQ for NVFP4 W4A16 on 8B; on GLM-5.3, RTN MXFP4 lost AIME points |
| GPTQ / AWQ (llm-compressor sequential, GPTQModel) | One layer on the GPU, the rest offloaded | Yes | Feasible but slow; HawkBearPig's GPTQ used 40,960 tokens; PhalaCloud's AWQ calibrated on agentic traces |
| ModelOpt NVFP4 W4A4 | Normally the whole model | Max calibration | Built on B300 nodes by NVIDIA and RadixArk; not practical here |
| AQLM, PV-tuning, VPTQ, tuned AutoRound | No | Codebook learning / fine-tuning | Not practical here |

Two practical facts for any calibrated method: (1) the starting weights — the 703.8 GiB FP8 release is itself
0.022 KL from BF16 on the common panel, and that error is not undone by re-quantising ([§2.3](#23-one-harness-many-checkpoints)); the 1,403 GiB BF16 mirror avoids it but is twice the download
and disk; (2) the reference for validation — brandonmusic's public BF16 full logits and malaiwah's sealed panel let
a candidate be scored without running the BF16 model.

## What could not be found

- A GLM-5.3-specific technical report; the semantics of `index_skip_topk_offset`.
- Any third-party KL for **nvidia/GLM-5.3-NVFP4** (the common registry stops at 2026-09-10; NVIDIA's release is
  2026-09-14). Task-level benchmarks for any GLM-5.3 quant between 2.75 and 4 bpw.
- A full-GLM-5.3 recipe from Alex Ellis, christopherowen, MiaAI-Lab (beyond three Sparks), eugr or NVIDIA's playbooks.
- Measured decode speed for GLM-5.3 (full) on Mac Studio clusters (exo, mlx-distributed), ktransformers,
  ik_llama.cpp or prima.cpp. Estimates exist but are labelled as such and are not cited.
- A published study of quantising the MTP layer on acceptance, or of "NVFP4 + 2-bit" expert mixtures on GLM-5.x.
- Unsloth's measurement method for its GLM-5.3 table (inferred to be llama.cpp's KLD on wikitext).
- The DFlash2 throughput table's hardware (incoai's card does not say).
