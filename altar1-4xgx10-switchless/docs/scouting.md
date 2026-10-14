# Scouting: altar-1 on memory-limited hardware (2026-10-14)

What is published, as of 2026-10-14, about **altar-1** (`AikidoSec/altar-1`, revision
`5d591cd98958e4e7e429517887e80ca3bb7ce2c4`): GLM-5.3 (`zai-org/GLM-5.3`) with 168 of its 256 routed experts kept
per MoE layer, routed experts and three attention projections at compressed-tensors INT4. This covers the release,
its quality evidence, the INT4 base it was cut from, the pruning method and what it does to the router, every engine
known to run it or its base, with speeds and hardware, and the licence. The GLM-5.3 background (the model, every other
quantised checkpoint, the GB10 engines) is in [glm53's scouting](../../glm53-4xgx10-switchless/docs/scouting.md)
and is not repeated here.

**Method.** Model cards, configs, commit histories and file listings were read from the Hugging Face API
(`/api/models/<repo>?blobs=true`, `/api/models/<repo>/commits/main`, `/resolve/<rev>/...`) on 2026-10-14. Source
code was read from PyPI wheels and GitHub at the commits named. Web search covered vendor posts, press, forums and
issue trackers. **Our own checks** (marked *checked*) read the 39 shard headers of altar-1 by HTTP range request and a
few hundred kilobytes of tensor data from altar-1, `cyankiwi/GLM-5.3-AWQ-INT4` (`20f265b6`), `zai-org/GLM-5.3`
(`c5087a76`, the FP8 release) and `zai-org/GLM-5.3-BF16` (`9d2398f4`). The scripts are kept out of the repo. No GPU
job was run. Unless a row says otherwise, every number is its author's own. "Vendor" means Aikido or 0xSero, who made
the cut; "independent" means anyone else. Inferred numbers are marked *inferred*. The configuration's facts already
established (layer and tensor layout, which classes are INT4, file sizes, tokenizer identity) come from the
configuration's own reading of the index and config on 2026-10-14, cited as "our reading".

## Verdict

- **What it is.** altar-1 is `cyankiwi/GLM-5.3-AWQ-INT4` with 88 experts a layer sliced out, and nothing else
  changed (*checked*). The kept experts' INT4 codes, scales and zero points are byte-identical to cyankiwi's. The
  router rows and `e_score_correction_bias` entries of the kept experts are byte-identical to cyankiwi's, and the
  removed experts' rows and entries are deleted. The bias is not re-centred and the router is not retrained.
  "Routing is untouched" (card) is true in that sense only. The router still picks 8 experts, now from 168: a token
  that would have gone to a removed expert goes to the next one, and the 8 weights are renormalised over the 8 that
  remain.
- **Its BF16 tensors are not GLM-5.3's** (*checked*). cyankiwi quantised from the **FP8 release**, not the BF16
  master: altar-1's BF16 `q_a_proj`, `kv_a_proj_with_mqa`, shared-expert `down_proj` and the BF16 experts of layers
  3, 77 and 78 equal `bf16(FP8 × block scale)` of `zai-org/GLM-5.3` bit for bit. AWQ's smoothing is also folded into
  the BF16 tensors: `post_attention_layernorm`, `q_a_layernorm` and `kv_a_layernorm` of the INT4 layers differ from
  GLM-5.3's by per-channel factors (0.27–6.1 in layer 4), and the input columns of the router and the shared
  expert's `gate_proj` carry the compensating factors. So **altar-1 must be served from its own tensors**; glm53's non-expert rank files do not apply.
- **Quality, vendor-run only.** KL to GLM-5.3 BF16 **0.506 nats** on 0xSero's 25-window panel (fp16-stored
  log-probabilities; 95 % interval 0.403–0.607). For scale, the unpruned GLM-5.3 at EXL3 3 bpw scores 0.089 and
  `Inferact/GLM-5.3-NVFP4` 0.117 on the same panel: altar-1's KL is 5.7 times the unpruned 3-bit build's, so most of
  it is the pruning (*inferred*). Aikido's CVE benchmark gives altar-1 a per-run recall of 60.4 % against 61.5 % for the unpruned AWQ base and
  65.6 % for GLM-5.3 BF16, on 32 CVEs × 3 runs. Nobody independent has evaluated it.
- **Engines.** vLLM with Marlin (vendor; no tok/s published). An independent MLX conversion decodes at **13.7
  tok/s** on one Mac Studio M3 Ultra (512 GB). Nothing has been published for altar-1 on DGX Spark or GB10, and no
  GGUF, EXL3 or NVFP4 build of altar-1 itself exists.
- **On this fleet** (*inferred* from the headers): 305.4 GiB of tensors is **76.4 GiB a rank** at TP=4, inside
  glm53's 87 GiB budget with no requantisation. A one-row decode token reads **33.7 GB** over the four ranks (13.95
  GB of routed experts, 19.76 GB of everything else), a floor of **~36 ms a token** at 235 GB/s a box before
  drafting, against glm53's ~32 ms (§6).
- **transformers cannot load it** ([model.md § 4.4](model.md#44-what-transformers-does-when-it-loads-this-checkpoint)).
  transformers 5.16.1 to 5.19.0 (and main of 2026-10-14) drop the experts' `weight_zero_point` when they fuse the
  experts at load, and compressed-tensors then refuses the asymmetric decompression. The same loader leaves the BF16
  experts of layers 3 and 77 at random initialisation. Both were reproduced on a tiny model on 2026-10-14. The exact
  math is still fully defined by the compressed-tensors source, and vLLM's Marlin dequantises to the same bits.

## 1. The release

| | |
|---|---|
| Repository | [`AikidoSec/altar-1`](https://huggingface.co/AikidoSec/altar-1), created 2026-09-07 18:50 UTC, current `5d591cd9` (2026-09-20 12:21 UTC). 48 files, 327,990,897,413 B (305.47 GiB; the card says "328 GB"). 2,425 downloads and 215 likes on 2026-10-14. No discussions |
| History | ([commits](https://huggingface.co/api/models/AikidoSec/altar-1/commits/main)) 2026-09-07: `a9d80df` initial, `e2a088c` "Build from cyankiwi/GLM-5.3-AWQ-INT4 base: REAP prune (W4A16)", `873ce02` "Add measured KL divergence: 0.506 nats", `accabac` card; then card-only commits on 2026-09-15 (`b8f9c42`, `c3f6c63`), 2026-09-16 (`174cd6f`) and 2026-09-20 (`5d591cd`). All weight files date from 2026-09-07 |
| Lineage | The repository began as 0xSero's "GLM-5.3-504B — REAP keep-168" (`0xSero/GLM-5.3-504B-W4A16`, which now redirects to `AikidoSec/altar-1`). It was renamed "Aura-1" and then "Altar-1" on 2026-09-15. The card credits 0xSero with the prune, "built on 8× NVIDIA RTX PRO 6000 Blackwell" |
| Base | [`cyankiwi/GLM-5.3-AWQ-INT4`](https://huggingface.co/cyankiwi/GLM-5.3-AWQ-INT4) (§3); `config.json` differs from cyankiwi's only in `n_routed_experts` (168 against 256), and `quantization_config` is identical, including cyankiwi's 2,862-entry `ignore` list, which still names experts 0–255 of layers 3, 77 and 78 (*checked*) |
| Format | compressed-tensors `pack-quantized`, INT4, group 32, asymmetric, zero points packed as int4; routed experts of layers 4–76 and `q_b_proj`, `kv_b_proj`, `o_proj` of layers 4–76 INT4, everything else BF16 (our reading; the exact semantics are in [model.md § 4](model.md)) |
| Prune plan | `reap_prune_plan.json`: `keep 168`, `criterion "massmax168"`, the kept original expert ids per layer for layers 3–78, and `mtp_ranking: "weight magnitude from …/mtp_expert_norms.json"`. **The MTP layer's experts were chosen by weight norm, not by routing statistics** (transformers does not run the MTP layer, so no observations exist for it) |
| Serving (card) | `vllm serve aikido/altar-1 --tensor-parallel-size 4 --trust-remote-code --max-model-len 131072`, "Requires Hopper (H100/H200)", "vLLM auto-selects the Marlin MoE kernel". The repository id in the command is wrong (`aikido/altar-1` returns 401; the repository is `AikidoSec/altar-1`). "or SGLang" was removed from the card on 2026-09-16 and "Requires Hopper" added, with no reason given |
| Card claims checked | "Only the routed experts are 4-bit; attention, the shared expert, the dense layers, and the head stay BF16": **not so**, since `q_b_proj`, `kv_b_proj` and `o_proj` of layers 4–76 are INT4 (5.85 GiB of the file; our reading). "~40B active parameters, same as the unpruned model": true, since top-8 is unchanged. "504B parameter": *inferred* 753.3B less 88 experts × 3 × 2048 × 6144 in each of 75 (504.1B) or 76 (500.8B, counting the MTP layer) MoE layers, consistent |
| Licence | "Inherits the [GLM-5.3 license](https://huggingface.co/zai-org/GLM-5.3)" (card `license: other`); **the repository ships no LICENSE file**. GLM-5.3's licence is MIT-style (keep the notice in all copies or substantial portions) plus a security review by Z.ai for a Model-as-a-Service operator above US$10B of group revenue in 12 months ([glm53 scouting § 1](../../glm53-4xgx10-switchless/docs/scouting.md#1-the-official-release)). cyankiwi ships the licence unchanged. Aikido adds no terms |

**Aikido's announcement.** Blog post "[Aikido Altar](https://www.aikido.dev/blog/aikido-altar-open-weight-ai-sovereign-security)"
by Selim Decamps and Tiburce Gridello, 2026-09-21 (updated 2026-09-22):
- **Sizes:** 1,506.7 GB BF16 → 488.2 GB AWQ → 328.0 GB pruned.
- **Calibration:** "traces from our pentesting harness running internal benchmarks" plus multilingual text, "no
  customer data". The card says "cybersecurity traces, coding, tool calling, reasoning, and English" plus
  multilingual Wikipedia.
- **Method:** "REAP's contribution score with a domain-preserving aggregation strategy" (but see §4: the published
  plan used routed mass, not REAP's score).
- **Toolkit:** links [0xSero/model-toolkit@3ee1d140](https://github.com/0xSero/model-toolkit/tree/3ee1d140efff3a3d5129fd207b9ce6eb32df88b0).
- **Coverage:** [Security Boulevard](https://securityboulevard.com/2026/09/aikido-altar-brings-glm-5-3-security-workloads-on-prem/)
  (2026-09-22), [Techzine](https://www.techzine.eu/news/security/144446/belgian-aikido-introduces-altar-a-local-open-weight-security-model/),
  [CybersecurityNews](https://cybersecuritynews.com/aikido-security-unveils-altar-1/) and MarkTechPost (2026-09-25)
  relay Aikido's numbers. One [Hacker News thread](https://news.ycombinator.com/item?id=49788228) (2026-09-21, 3
  points) has a single comment, unfavourable to Aikido's review bot, not to the model.

**The 32-CVE numbers** (vendor-run). Aikido's "internal CVE benchmark … AI Code Analysis harness: 32 known
vulnerabilities across 30 repositories, with three runs per case":

| Model | Recall, mean per run | CVEs found at least once (of 32) |
|---|---:|---:|
| altar-1 | 60.4 % | 23 |
| GLM-5.3 AWQ-INT4 (unpruned, cyankiwi) | 61.5 % | 23 (the same 23) |
| GLM-5.3 BF16 | 65.6 % | 25 |

Hence the post's "92 % of parent coverage" (23 of 25). The caveats:
- **Scope.** By the post's own account the benchmark measures "targeted CVE rediscovery within a pipeline that uses
  other models for surrounding stages", not blind discovery, exploit validation or fixing. Incomplete runs are
  "reported separately", but their counts are not given.
- **Sample size.** 32 cases × 3 runs, with no confidence intervals. One CVE is 3.1 points of coverage.
- **Possible overlap.** The calibration traces come from "internal benchmarks", and the post does not say whether
  they overlap this CVE set.
- **Not independent.** Aikido's own [2026-08-21 benchmark post](https://www.aikido.dev/blog/ai-model-benchmarks-aug-21-2026)
  (10 models, 3 runs each, 30-turn limit, no internet, CVEs chosen as recently disclosed to limit contamination)
  scored GLM-5.3 25/32 pooled and 18/32 in all three runs, and describes Aikido as Z.ai's "early evaluation partner
  for GLM-5.3". Security Boulevard: "The results come from Aikido's own benchmark, not an independent evaluation."

## 2. The fidelity study

[`0xSero/glm-5.3-reap-fidelity-study`](https://huggingface.co/datasets/0xSero/glm-5.3-reap-fidelity-study)
(dataset, `ed76f874`, created 2026-09-05, last modified 2026-09-08, Apache-2.0; vendor-run). Read from its
`kld_summary.json`, `panel.json` and README.

- **Reference:** `zai-org/GLM-5.3-BF16` run in vLLM "with 944GB CPU offload".
- **Panel:** 25 windows × 2,048 tokens: windows 0–9 general wikitext, 10–17 held-out code, 18–24 held-out
  multilingual; seed 20260905; panel sha256 `83bd87ea…`. The tokens are not published. The card's "sealed 25-prompt
  panel" is this set; it is 0xSero's own, not the third-party panel glm53's scouting cites.
- **Metric:** scored every 4th position (12,800 positions). KL(reference ‖ candidate) over the full vocabulary
  (154,880 in the README, 154,856 in `panel.json`), accumulated in float64 over **fp16-stored log-probabilities**, with
  a 95 % window-block bootstrap (B = 5000). Top-1 is next-token accuracy against the text: reference 0.7409, NVFP4
  0.7275, altar-1 0.6170.

| Run (dataset label) | Experts | KL (nats) | 95 % interval | Raw file |
|---|---:|---:|---|---|
| unpruned, EXL3 3.0 bpw | 256 | 0.0892 | 0.074–0.104 | yes |
| `Inferact/GLM-5.3-NVFP4` | 256 | 0.1167 | | yes |
| massmax keep-224 (EXL3 3 bpw) | 224 | 0.195 | | yes |
| massmax keep-208 (EXL3 3 bpw) | 208 | 0.283 | | yes |
| `0xSero/GLM-5.3-569B-W4A16` (massmax, cyankiwi base) | 192 | 0.3573 | 0.289–0.425 | yes |
| massmax keep-192 (EXL3 3 bpw) | 192 | 0.361 | | README only; the file labelled "569B-EXL3 keep-192" reads 0.5487, matching the "reap (mean)" criterion row |
| massmax keep-180 (EXL3) | 180 | 0.428 | | README only |
| **altar-1** (the 504B W4A16) | 168 | **0.5060** | 0.403–0.607; median per token 0.0909, p95 2.547 | yes |
| massmax keep-168 (EXL3 3 bpw, `0xSero/GLM-5.3-500B-EXL3-3.0bpw`) | 168 | 0.511 | | README only |
| NVFP4 keep-168 variants v2b / v2g / v1 | 168 | 0.745 / 0.787 / 0.889 | | yes |
| "4bpw base" (unexplained, latest commit) | ? | 1.022 | | yes |

- **altar-1 by domain** (*inferred*, our averages of its per-window values): prose 0.79, code 0.27, multilingual
  0.37. The pruning hurts general prose most.
- **Criterion sweep at keep-192** (KL / top-1 agreement with the reference): massmax 0.361 / 0.822, massbal 0.399,
  reapmax 0.406, **REAP's own mean saliency 0.549** / 0.775, routed mass 0.568, frequency 0.635 / 0.750, random 0.685.
  On 0xSero's own data the stock REAP criterion is barely better than frequency.
- **"At this bit-width the quantization format barely moves the result"** (card: 0.506 W4A16 against 0.511 EXL3):
  the 0.511 has no raw file, and the dataset's keep-192 EXL3 file contradicts the README's matching 0.361 claim. The
  pairwise "format doesn't matter" is unsupported by the data as published. 0xSero's toolkit README says these
  artefacts used fp16 log-probabilities and different panels, and should not be compared with its newer FP32
  protocol.
- **Comparability.** None of these KLs is comparable with glm53's harness numbers (different panel, fp16 storage;
  [glm53 scouting § 2.3](../../glm53-4xgx10-switchless/docs/scouting.md#23-one-harness-many-checkpoints)). Within this
  panel, the ordering unpruned 3 bpw (0.089) < NVFP4 (0.117) ≪ keep-192 (0.36) < keep-168 (0.51) is the usable
  information.
- **Other files:** `gpqa/gpqa_massmax192.json`: keep-192 EXL3 3 bpw scores 85.35 % on GPQA-Diamond (169 of 198; 39
  truncated, 13 without an answer), with no baseline in the same file.
- **Observations dataset:** [`0xSero/glm-5.3-reap-observations-v1`](https://huggingface.co/datasets/0xSero/glm-5.3-reap-observations-v1),
  linked from the card, returns 401 to anonymous requests.

## 3. The INT4 base: cyankiwi/GLM-5.3-AWQ-INT4

| | |
|---|---|
| Repository | [`cyankiwi/GLM-5.3-AWQ-INT4`](https://huggingface.co/cyankiwi/GLM-5.3-AWQ-INT4), `20f265b66eed1be007ac24d1814c1b47f8e21167`, two commits by cpatonn on 2026-09-01 (07:53 and 08:02 UTC); 488.22 GB; card "Version 26.05.01" |
| Method | AWQ (card tags) into compressed-tensors `pack-quantized`: INT4, group 32, `symmetric: false`, `zp_dtype torch.int8`, `observer "mse"`, `actorder null`, targets `Linear` and `re:.*fused_qkv_a_proj$`; written by compressed-tensors `0.1.dev537+g1cee215` (a development build) with transformers `5.16.0.dev0` (`config.json`). The card names no tool, recipe or sample count |
| Calibration | "STEM and Agentic", linking [`cyankiwi/calibration`](https://huggingface.co/datasets/cyankiwi/calibration) (`1dc050f7`, no README): 384 conversations, from Nemotron-SFT-Multilingual-v1 (136), MMLU-ProX (120), Nemotron-Post-Training-v1 (96) and Nemotron-SWE-v1 (32); English 128, Chinese 64, other languages 16–32 each |
| Source weights | **The FP8 release** (*checked*, 2026-10-14): in layers 3, 4, 77 and 78, every sampled BF16 weight of altar-1 (128 rows each of `q_a_proj`, `kv_a_proj_with_mqa`, the shared `down_proj`, and 10 BF16 expert matrices of layers 3, 77, 78) equals `bf16_rne(E4M3 × weight_scale_inv)` of `zai-org/GLM-5.3` at `c5087a76`, 100 % of values; none equals the BF16 master |
| AWQ smoothing | Folded into the BF16 tensors (*checked*, layer 4): `input_layernorm` is unchanged, but `post_attention_layernorm` differs from GLM-5.3's by per-channel factors 0.39–1.46, `q_a_layernorm` 0.37–6.05, `kv_a_layernorm` 0.27–3.45. The router `gate.weight` columns equal GLM-5.3's times the same per-channel factor as the norm's change (median deviation 0.13 %, i.e. bf16 rounding), so router logits are the same function up to rounding. The shared-expert `gate_proj` (corr 0.9955) also carries the factor. In layers 3 and 77, where everything is BF16, the norms equal the release's |
| What stays BF16 | Per the `ignore` list: `lm_head`; in every layer `q_a_proj`, `kv_a_proj_with_mqa`, the router and the shared expert; the indexer's `wq_b`, `wk` and `weights_proj` in the 22 layers that have one (0, 1, 2, 6, 10, …, 74 and the MTP layer 78); all attention projections of layers 0–3, 77 and 78; the dense MLPs of layers 0–2; and **all routed experts of layers 3, 77 and 78**. *Inferred:* the `q_a_proj` and `kv_a_proj` exclusion follows common practice for the small MLA down-projections; the indexer and router are excluded by every GLM-5.3 quantisation read ([glm53 scouting § 1](../../glm53-4xgx10-switchless/docs/scouting.md#1-the-official-release)) |
| Why layers 3 and 77 stay BF16 | **Not explained anywhere.** [Discussion #1](https://huggingface.co/cyankiwi/GLM-5.3-AWQ-INT4/discussions/1) asks why 5.3 is "~48 GB larger than 5.2" and whether "the BF16 retention [is] intentional"; no answer as of 2026-10-14. *Inferred:* the first and last MoE layers are the usual sensitive ones, and 78 is the MTP layer that transformers does not run (so an activation-calibrated method like AWQ cannot calibrate it). Layers 3 and 77 cost 23.6 GiB in altar-1 against ~6.8 at INT4 |
| Quality | None published for the quant itself (the card repeats Z.ai's GLM-5.3 benchmarks). In the fidelity study its pruned cuts are the W4A16 rows; the unpruned base was not scored there. Aikido's CVE table gives it 61.5 % recall against 65.6 % for BF16 |
| Engines | Z.ai's list (SGLang, vLLM and others). A user in the same discussion (2026-09-30): "running it on 4x H200 with vLLM. Has anyone managed to serve it with SGLang on Hopper?" |

## 4. REAP, the criterion actually used, and the router

**The paper.** REAP ("Router-weighted Expert Activation Pruning", Lasby et al., Cerebras,
[arXiv:2510.13999](https://arxiv.org/abs/2510.13999), v1 2025-10-15, v3 2026-05-13):
- **Saliency** (eq. 9): S_j = (1/|X_j|) Σ_{x∈X_j} g_j(x) · ‖f_j(x)‖₂, the mean over the tokens routed to expert j of
  its gate weight times the norm of its output. The lowest-scoring experts are removed.
- **Calibration:** 1,024 samples packed to 2,048 tokens for models up to 110B; 12,228 samples of up to 16,384 tokens
  above that. Domain-matched calibration matters: with C4 calibration, code generation "collapse[s]".
- **The router:** pruning "removes expert j and re-normalizes the router outputs over the remaining" experts. No
  fine-tuning; the paper analyses the error as substitution plus renormalisation.

**The code** ([CerebrasResearch/reap@1970473c](https://github.com/CerebrasResearch/reap), 2026-04-17, the commit
pinned by 0xSero's toolkit). `src/reap/prune.py` lines 124–133 slice the router:
`router.weight.data[retained_expert_indicies, :]`, the optional `router.bias`, and
`if hasattr(router, "e_score_correction_bias"): router.e_score_correction_bias.data = router.e_score_correction_bias.data[retained_expert_indicies]`.
Nothing else is changed. Its `MODEL_ATTRS` has `Glm4MoeForCausalLM` but no `GlmMoeDsa` entry. Its observer selects
experts by `torch.topk(router_logits, top_k)` on raw logits (`observer.py` 355–377), ignoring the correction bias, and
its metric applies a softmax even to sigmoid routers (`pruning_metrics.py` 172).

**What altar-1 used.** 0xSero's [model-toolkit@3ee1d140](https://github.com/0xSero/model-toolkit/tree/3ee1d140efff3a3d5129fd207b9ce6eb32df88b0)
`reap-pruning/plan.py` line 21 defines `massmax: score = normalized.max(0)`: each domain's routed mass (sum of
router weights, `weighted_expert_frequency_sum`, `export.py` 14; no activation norms) normalised per domain, then the
maximum over domains. So altar-1's experts were **not** chosen by REAP's saliency, despite the name. The observations
were captured on an NVFP4 GLM-5.3 in a vLLM TP8/EP8 pipeline (toolkit README), not on BF16. The weight surgery that
applied the plan is not published ("not claimed as a generic operation here", toolkit README).

**What the checkpoint shows** (*checked*, layers 4 and 40 against cyankiwi, layer 4 against the BF16 master):
`mlp.gate.weight` [168, 6144] is exactly cyankiwi's rows at the plan's kept ids; `e_score_correction_bias` [168]
is exactly cyankiwi's (and the master's) entries at those ids. In layer 4 the kept biases span 28.10–28.65 and the
removed ones 28.29–28.67. The removed experts were not the low-bias ones. The kept experts' INT4 tensors and
`o_proj` are byte-identical to cyankiwi's. So the routing function for a kept expert is GLM-5.3's (up to AWQ's
rounding of the router columns); the top-8 is taken over 168 candidates instead of 256, and the eight selected
sigmoid scores are renormalised as before ([model.md § 6](model.md)).

## 5. Engines that run altar-1 or its base

| Engine | Hardware | Speed | Who | Notes |
|---|---|---|---|---|
| vLLM, Marlin MoE (card's recipe) | 4× H200 | none published | vendor | On vLLM main (`bb1438a9`, 2026-10-14) Hopper now prefers the "Humming" kernels over Marlin for this format; v0.31.0 (`db9527a4`) uses Marlin. sm_120/sm_121 (GB10) uses Marlin for both MoE and the INT4 linears; Machete is ruled out by group 32. vLLM's dequantised weight equals compressed-tensors' bit for bit; its GEMM order does not ([model.md § 4.5](model.md)) |
| SGLang | — | — | — | Removed from the card on 2026-09-16; no working report found. Its compressed-tensors WNA16 MoE path is Marlin, but it converts compressed-tensors zero points with `moe_awq_to_marlin_zero_points`, which undoes AWQ's nibble interleave that compressed-tensors does not apply (`schemes/compressed_tensors_wNa16_moe.py` 385–401 at SGLang `72730d63`): read as a probable zero-point permutation bug for asymmetric checkpoints, not tested |
| MLX: [`openvuln/altar-1-4bit-mlx`](https://huggingface.co/openvuln/altar-1-4bit-mlx) (`557a04fe`, 2026-10-07, 338 GB) | Mac Studio M3 Ultra, 512 GB | **decode 13.7 tok/s** at 128 tokens of context, 12.6 at 2k, 11.5 at 128k; prefill 205 tok/s at 2k, 23.7 at 128k; 2 streams 18.6, 4 streams 26.4 aggregate | independent | INT4 codes bit-identical to altar-1's; `kv_b_proj` requantised; the MTP layer dropped; a custom `glm_moe_dsa_shared.py` restores indexer sharing |
| `Model-SafeTensors/altar-1-FP8` (503 GB, block-128 FP8) and `…/altar-1-INT4` ("int4_a128", FP8 non-experts, 262 GB) | — | — | independent | no card |
| Mirrors | — | — | — | byte-identical copies by three accounts |
| EXL3 of the same cut: [`0xSero/GLM-5.3-500B-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-500B-EXL3-3.0bpw) (`bada39f0`) | 2 DGX Spark (handbook) | "fits; speed not measured yet" | vendor | 197.1 GB; exllamav3 1.4.6, 3.0 bpw, head 6 bits, MTP 4 bits, 512 × 4096 calibration rows; KL 0.511 (card, no raw file). Listed in 0xSero's [DGX Spark Handbook](https://huggingface.co/blog/exolabs/the-dgx-spark-handbook) (2026-09-25) |
| Nearest GB10 datum (not altar-1): [`Tech2wild/GLM-5.3-Int4-Int8Mix`](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix) | 4× DGX Spark, TP=4 | 12.12 tok/s single stream, 46.03 at 6 concurrent | independent | full GLM-5.3, compressed-tensors W4A16 experts and W8A16 attention at group 128, vLLM 0.23.1rc1 with sm12x overlays; glm53's scouting has faster forks of the same family |

Not found: any GGUF, EXL3 or NVFP4 build of altar-1 itself; any altar-1 run on DGX Spark, GB10 or Strix Halo; any
published altar-1 speed on H200; any third-party quality evaluation of altar-1.

0xSero's other cuts of the same lineage, for reference (sizes and KL from their cards):

| Repository (revision) | Size | Experts | KL (fidelity panel) |
|---|---:|---:|---:|
| [`0xSero/GLM-5.3-569B-W4A16`](https://huggingface.co/0xSero/GLM-5.3-569B-W4A16) (`40c27559`) | 371.7 GB | 192 | 0.357 |
| [`0xSero/GLM-5.3-661B-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-661B-EXL3-3.0bpw) (`1ef885c6`) | 257.9 GB | 224 | 0.195 |
| [`0xSero/GLM-5.3-615B-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-615B-EXL3-3.0bpw) (`caba9474`) | 240.5 GB | 208 | 0.283 |
| [`0xSero/GLM-5.3-569B-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-569B-EXL3-3.0bpw) (`7d5ed867`) | 223.2 GB | 192 | 0.361 (dataset file: 0.549) |
| [`0xSero/GLM-5.3-533B-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-533B-EXL3-3.0bpw) (`a2ada5b6`) | 210.1 GB | 180 | 0.428 |
| [`0xSero/GLM-5.3-500B-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-500B-EXL3-3.0bpw) (`bada39f0`) | 197.1 GB | 168 | 0.511 |
| [`0xSero/GLM-5.3-EXL3-3.0bpw`](https://huggingface.co/0xSero/GLM-5.3-EXL3-3.0bpw) (`9b1d1eff`) | 292.7 GB | 256 | 0.089 |

## 6. What it costs on this fleet (inferred)

From the 39 shard headers (*checked* sizes; the rest *inferred*):

| Class | GiB | Read per decode token (one row) |
|---|---:|---:|
| Routed experts, layers 4–76, INT4 (4.625 bits a weight with scale and zero point) | 249.26 | 8 of 168 a layer |
| Routed experts, layers 3 and 77, BF16 | 23.63 | 8 of 168 a layer |
| MTP layer's 168 experts, BF16 | 11.81 | only when drafting |
| Attention INT4 (`q_b`, `kv_b`, `o_proj`, layers 4–76) | 5.85 | all |
| Attention BF16 (`q_a`, `kv_a`, indexer; all of layers 0–3 and 77) | 4.10 | all |
| Shared experts, BF16 | 5.27 | all |
| Head / embeddings, BF16 | 1.77 / 1.77 | head all, embedding one row |
| Dense MLPs (layers 0–2), BF16 | 1.27 | all |
| MTP layer's other tensors; routers; norms | 0.54; 0.14; 0.00 | |
| **Total** | **305.42** | **33.7 GB** (13.95 routed experts, 19.76 the rest) |

- **Memory.** 76.4 GiB a rank at TP=4 if every tensor splits evenly, against glm53's 87 GiB budget
  ([glm53 quantisation § 1](../../glm53-4xgx10-switchless/docs/quantisation.md#1-the-per-rank-budget)). It fits as
  shipped, with ~10 GiB a rank to spare for KV (52.7 KiB a token in FP8, as glm53).
- **Decode floor.** 33.7 GB a token over four ranks is 8.43 GB a rank, **~36 ms a token** (~28 tok/s) at
  235 GB/s before drafting, against glm53's 7.53 GB a rank and ~32 ms. The non-expert weights cost about the same
  in both (4.94 GB a rank here, INT4 attention offsetting the BF16 shared experts and latent projections; 4.87 in
  glm53 at FP8); the difference is the routed experts at 4.625 bits a weight (3.49 GB a rank) against glm53's 3.75
  (2.65), and layers 3 and 77 in BF16. Keeping altar-1 exact means keeping every tensor in its shipped format.
- **The MTP layer** is BF16 and 12.3 GiB in all (experts 11.8); a draft step reads ~0.6 GB of its experts plus its
  attention.

## 7. Could not verify

- The observations dataset (401), the panel's tokens, and the raw files behind the 0.511 (EXL3 keep-168) and 0.428
  KLs.
- The weight-surgery code that applied the plan; cyankiwi's AWQ recipe and tool; why layers 3 and 77 stay BF16.
- Whether Aikido's calibration traces overlap its CVE set; the incomplete-run counts.
- Any measured altar-1 speed on H200 or on any GB10-class box; whether SGLang's zero-point handling is in fact wrong
  (read, not run).
- Reddit and LinkedIn threads (blocked or not found); the X announcement was not read.
