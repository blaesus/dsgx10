# GLM-5.3: the model, exactly

The math of `GlmMoeDsaForCausalLM` (`model_type: glm_moe_dsa`) as the reference code defines it, the
NVFP4 format of the checkpoint on disk, how it differs from DeepSeek-V4.1-Flash, and the tokenizer and
chat template. Written 2026-10-13.

Sources, cited below by short name:

| Short name | What |
|---|---|
| `modeling` | transformers 5.15, `models/glm_moe_dsa/modeling_glm_moe_dsa.py` (generated from `modular_glm_moe_dsa.py`; line numbers are the generated file's) |
| `configuration` | transformers 5.15, `models/glm_moe_dsa/configuration_glm_moe_dsa.py` |
| `moe.py` | transformers 5.15, `integrations/moe.py` (the default experts implementation) |
| checkpoint | `nvidia/GLM-5.3-NVFP4` at revision `e3b8f9f2fad29727dfa95e766c5cac18ea1414bf`: `config.json`, `hf_quant_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, the shard headers, and a few tensors read with `safetensors` (scripts kept out of the repo) |
| ds41f | [../../ds41f-4xgx10-switchless/DESIGN.md](../../ds41f-4xgx10-switchless/DESIGN.md), [docs/findings.md](../../ds41f-4xgx10-switchless/docs/findings.md#the-model), `frontend/src/tokenizer/`, `frontend/src/encoding.rs` |

Anything not read directly from one of these is marked *inferred*.

## Contents

1. [Dimensions](#1-dimensions)
2. [The forward pass](#2-the-forward-pass)
3. [The MTP layer (78)](#3-the-mtp-layer-78)
4. [What must be cached per token](#4-what-must-be-cached-per-token)
5. [The checkpoint's formats (NVFP4, FP8, BF16)](#5-the-checkpoints-formats)
6. [Against DeepSeek-V4.1-Flash](#6-against-deepseek-v41-flash)
7. [Tokenizer and chat template](#7-tokenizer-and-chat-template)
8. [Reference implementations](#8-reference-implementations)
9. [Surprises, in one list](#9-surprises-in-one-list)

## 1. Dimensions

From `config.json` (the values the checkpoint ships) and `configuration.__post_init__` (lines 136–168).

| Symbol | Value | Config field |
|---|---:|---|
| L (main layers) | 78 | `num_hidden_layers`; plus one MTP layer stored as `model.layers.78.*` (`num_nextn_predict_layers 1`) |
| d (hidden) | 6144 | `hidden_size` |
| V (embedding / head rows) | 154,880 | `vocab_size`; the tokenizer uses ids 0–154,855 only (§7) |
| H (attention heads) | 64 | `num_attention_heads` = `num_key_value_heads` (no GQA in the expanded form) |
| r_q (query latent) | 2048 | `q_lora_rank` |
| r_kv (KV latent) | 512 | `kv_lora_rank` |
| d_nope | 192 | `qk_nope_head_dim` |
| d_rope | 64 | `qk_rope_head_dim` |
| d_qk | 256 | `qk_head_dim` = 192 + 64 |
| d_v | 256 | `v_head_dim` |
| `head_dim` | 64 | **the rotary dimension, nothing else**: `__post_init__` overwrites it with `qk_rope_head_dim` (line 153) so that the Llama-style rotary class (`modeling` line 99) builds 32 frequencies |
| RoPE | θ = 8,000,000, `rope_type default`, interleaved | `rope_parameters`, `rope_interleave`, `indexer_rope_interleave`; no YaRN, so no mscale (`yarn_apply_mscale`, line 301) |
| Max positions | 1,048,576 | `max_position_embeddings` (no effect on the math with default RoPE) |
| Indexer | 32 heads × 128, top-2048 | `index_n_heads`, `index_head_dim`, `index_topk` |
| Indexer pattern | 21 `full`, 57 `shared` layers | `indexer_types` (§2.5) |
| Dense MLP | layers 0–2, intermediate 12,288 | `first_k_dense_replace 3`, `intermediate_size`, `mlp_layer_types` |
| MoE | layers 3–77: 256 routed experts, top-8, 1 shared, intermediate 2048 | `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts`, `moe_intermediate_size` |
| Router | sigmoid, `noaux_tc` bias, `n_group 1`, `topk_group 1`, `norm_topk_prob`, × 2.5, fp32 | `scoring_func`, `topk_method`, `routed_scaling_factor`, `moe_router_dtype` |
| Activation | SiLU (SwiGLU), no clamp | `hidden_act` |
| RMSNorm eps | 1e-5 for the layer and final norms; **1e-6** for `q_a_layernorm` and `kv_a_layernorm` | `rms_norm_eps`; the two latent norms are built without an eps argument (`modeling` lines 343, 351) and take the class default 1e-6 (line 49) |
| Indexer `k_norm` | **LayerNorm with bias**, eps 1e-6 | `modeling` line 191 |
| Biases | none, except `k_norm.bias` | `attention_bias false`, `mlp_bias false` |
| Embeddings | untied (`lm_head.weight` is its own tensor) | `tie_word_embeddings false` |
| Special ids | bos 0 (unused by the template), pad 154820, eos [154820, 154827, 154829] | `config.json`, `generation_config.json` |

Attention softmax scale: `d_qk^-0.5 = 256^-0.5 = 0.0625` (`modeling` line 364; `yarn_apply_mscale`
returns it unchanged for `rope_type default`).

## 2. The forward pass

Shapes are for one sequence of S rows at positions `p_0 … p_0+S−1` against T = p_0+S keys in total
(a single decode token is S = 1). bf16 unless stated. "Linear" means `y = x Wᵀ` with `W` stored
`[out, in]` as in the checkpoint; the reference runs every linear in bf16 with fp32 accumulation
(PyTorch's bf16 GEMM) and rounds the output to bf16.

### 2.1 Embedding and the residual stream

```
h  = embed_tokens[ids]                       # [S, 6144] bf16           (modeling 708)
cos, sin = rotary(positions)                 # computed in fp32, then cast to bf16 (modeling 117–122)
for l in 0..77:  h, topk = layer_l(h, topk)  # (modeling 733–743)
h  = RMSNorm_final(h)                         # eps 1e-5               (modeling 745)
logits = lm_head(h)                          # [S, 154880], bf16 out; not upcast (modeling 812)
```

The residual stream is **bf16** throughout: each sublayer's output (bf16) is added to the bf16
residual with one bf16 rounding (`modeling` 632, 637). There are no other casts on the stream.

Positions are `arange(S) + past_len` (`modeling` 714–716); the template's `[gMASK]<sop>` are
positions 0 and 1.

### 2.2 RMSNorm

`modeling` 57–62:

```
x32 = float32(x)
y   = x32 * rsqrt(mean(x32²) + eps)           # fp32
out = weight_bf16 * bf16(y)                   # cast to bf16 first, then a bf16 multiply (a 2nd rounding)
```

The weight multiply is in **bf16**, after the cast: two roundings, not one. eps is 1e-5 for
`input_layernorm`, `post_attention_layernorm`, `model.norm`, the MTP's `enorm`/`hnorm`/`shared_head.norm`
(*inferred* for the MTP norms: DeepSeek-MTP norms are built with `rms_norm_eps`),
and **1e-6** for `q_a_layernorm` and `kv_a_layernorm`.

### 2.3 One decoder layer

`modeling` 619–638:

```
r = h;  x = input_layernorm(h)
a, topk = attention(x, prev_topk)             # §2.4–2.6
h = r + a                                     # bf16
r = h;  x = post_attention_layernorm(h)
h = r + mlp(x)                                # dense (§2.8) for l ≤ 2, MoE (§2.7) for l ≥ 3
```

### 2.4 Attention (MLA)

`GlmMoeDsaAttention.forward`, `modeling` 398–470. `x` is the normed input `[S, 6144]`.

| Step | Formula | Shape | Line |
|---|---|---|---|
| query latent | `cq = q_a_layernorm(x · q_a_projᵀ)` (eps 1e-6) | `[S, 2048]` | 401 |
| query | `q = cq · q_b_projᵀ` viewed `[S, 64, 256]`; per head `q_nope = q[..., 0:192]`, `q_rope = q[..., 192:256]` | `[S, 64, 256]` | 402–403 |
| KV latent | `ckv_full = x · kv_a_proj_with_mqaᵀ`; `c = ckv_full[:, 0:512]`, `k_rope = ckv_full[:, 512:576]` | `[S, 576]` | 405–406 |
| latent norm | `c = kv_a_layernorm(c)` (eps 1e-6). `k_rope` is **not** normed | `[S, 512]` | 408 |
| RoPE | `q_rope, k_rope = rope_interleaved(q_rope, k_rope)` (§2.4.1); `k_rope` is one 64-vector shared by all heads | `[S,64,64]`, `[S,64]` | 412 |
| query | `Q_h = [q_nope_h ‖ q_rope_h]` | `[S, 64, 256]` | 414 |
| key/value expansion | `kv = c · kv_b_projᵀ` viewed `[S, 64, 448]`; `k_nope_h = kv[h, 0:192]`, `v_h = kv[h, 192:448]`; `K_h = [k_nope_h ‖ k_rope]` | `[S,64,256]` each | 378–385 |
| scores | `s_h(i, j) = (Q_h(i) · K_h(j)) · 0.0625` | `[64, S, T]` | 283 |
| mask | causal, and on top the sparse mask: `−inf`/`finfo(bf16).min` at every key `j` not in `topk(i)` (§2.5) | | 440–449 |
| softmax | in **fp32**, then cast to bf16 (eager path); PV product in bf16 | | 287–289 |
| output | `o = concat_h(Σ_j P_h(i,j) v_h(j))` → `[S, 16384]`; `out = o · o_projᵀ` | `[S, 6144]` | 468–469 |

`kv_b_proj` rows are ordered head-major: rows `448h … 448h+191` are head h's `W_UK` (k_nope), rows
`448h+192 … 448h+447` its `W_UV` (value). There is no attention sink and no sliding window.

**The reference caches the expanded K and V**, not the latents (comment at `modeling` 418). It is
the same function as DeepSeek-V3's absorbed form, which is what an engine (and vLLM/SGLang) runs:

```
q_lat_h = q_nope_h · W_UK_h            # [512]   (W_UK_h = kv_b_proj rows 448h..448h+191, [192, 512])
s_h(i,j) = (q_lat_h · c_j + q_rope_h · k_rope_j) · 0.0625
u_h(i)   = Σ_j P_h(i,j) c_j            # [512]
o_h(i)   = W_UV_h · u_h(i)             # [256]   (W_UV_h = rows 448h+192..448h+447, [256, 512])
```

The two forms differ only in rounding (the reference rounds `k_nope`, `v` to bf16 after expansion;
the absorbed form rounds `q_lat` and `u`). Exact agreement with the reference at bf16 is therefore
not available to any engine that absorbs; the tolerance has to be set against the reference's own
spread, as for ds41f.

#### 2.4.1 Interleaved RoPE, exactly

`rotary.forward` (`modeling` 108–122) and `apply_rotary_pos_emb_interleave` (125–161).

```
inv_freq[i] = 1 / 8e6^(2i/64),  i = 0..31              # fp32 (line 103); inv_freq[0] = 1, inv_freq[31] ≈ 2.054e-7
angle[p, i] = p · inv_freq[i]                          # fp32
cos[p, i], sin[p, i] = bf16(cos(angle)), bf16(sin(angle))   # cast to the activation dtype (line 122)

input x[0..63]: the pairs are (x[2i], x[2i+1])
out[i]      = x[2i]   · cos_i − x[2i+1] · sin_i        # i = 0..31
out[32 + i] = x[2i+1] · cos_i + x[2i]   · sin_i
```

So the rotation is the interleaved (GPT-J-style) one, **and the output is written de-interleaved**:
evens' results first, odds' second. The permutation is `out_pos(2i) = i`, `out_pos(2i+1) = 32 + i`.
The reference applies it to both q and k, so dot products are unchanged by it; but an engine that
caches `k_rope` must use one layout for q and k (either keep the reference's de-interleaved output for
both, or apply a non-permuting interleaved rotation to both). Note the arithmetic is bf16 with
bf16-rounded cos/sin (each product rounded, then the sum), not fp32.

The same function with the same `cos`/`sin` is used for the main attention (rope on the **last** 64
of each 256-wide query head, and on the 64-dim `k_rope`) and for the indexer (rope on the **first**
64 of each 128-wide indexer head, §2.5).

### 2.5 The sparse attention (DSA indexer)

`GlmMoeDsaIndexer.forward`, `modeling` 196–255. Only `full` layers have an indexer; `x` is the same
normed layer input the attention sees, `cq` the same normed query latent.

```
q   = (cq · wq_bᵀ).view(S, 32, 128)                  # wq_b [4096, 2048]           (224–225)
k   = LayerNorm_{eps 1e-6}(x · wkᵀ) * k_norm.weight + k_norm.bias    # wk [128, 6144] → [S, 128]  (228)
q[..., 0:64], k[..., 0:64] = rope_interleaved(q[..., 0:64], k[..., 0:64])   # rope dims FIRST     (226–234)
K   = cache_append(k)                                # [T, 128]: the indexer key cache        (236–237)
s_hj = relu( fp32(q_h) · fp32(K_j) · 128^-0.5 )      # fp32, [S, 32, T]                         (239–240)
w_h  = fp32(x) · fp32(weights_projᵀ) · 32^-0.5       # weights_proj [32, 6144], fp32 GEMM       (243)
I(i, j) = Σ_h w_h(i) · s_hj(i)                         # fp32, [S, T]                             (244)
I(i, j) = −inf for j > position(i)                    # causal                                  (247–252)
topk(i) = indices of the min(2048, T) largest I(i, ·)  # int32 [S, min(2048,T)]                  (254–255)
```

Details that matter:

- `k_norm` is a full **LayerNorm** (mean subtraction, weight and bias; both tensors are BF16 in the
  checkpoint), unlike every other norm in the model.
- `weights_proj` is BF16 in the checkpoint but the reference promotes the module to fp32
  (`_keep_in_fp32_modules = ["indexer.weights_proj"]`, `modeling` 660) and casts `x` to fp32 before
  the GEMM: the head weights are an fp32 product of bf16 values.
- The weights `w_h` can be negative, so `I` can be negative, and keys whose 32 ReLU terms are all
  zero tie at exactly 0. `torch.topk` does not define the order among ties; an engine must pick one
  rule and keep it (*inferred* risk; it only matters when fewer than 2048 keys score above a tie).
- No quantisation, no Hadamard rotation of q/k in the reference. (The serving engines do quantise:
  §8.)
- The query's own key is not forced into the selection; it competes like any other (*inferred* from
  the absence of any such step).

**Which keys a query sees.** Query row i at position `p_i` attends to the keys in `topk(i)` that are
also causal (`modeling` 440–449). Because `topk = min(2048, T)` and non-causal keys score −inf:

- if `p_i < 2048` the query attends to **all** its `p_i + 1` causal keys: dense attention;
- if `p_i ≥ 2048` it attends to exactly the 2048 causal keys with the largest `I`.

There is no separate dense path and no threshold on the prompt length; the rule is per query
position. In a prefill chunk that crosses position 2048 the early rows are dense and the later rows
sparse.

**`full` and `shared` layers.** `config.json` ships `indexer_types` explicitly. It equals what
`__post_init__` derives from `index_topk_freq 4` and `index_skip_topk_offset 3` (`configuration`
lines 145–149):

```
full(l)  ⇔  max(l − offset + 1, 0) mod freq == 0      →  max(l − 2, 0) mod 4 == 0
```

| Layers | Type |
|---|---|
| 0, 1, 2 | full |
| 3, 4, 5 | shared (use layer 2's top-k) |
| 6 | full |
| 7, 8, 9 | shared (use layer 6's) |
| … every 4th: 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74 | full |
| the three after each | shared |
| 75, 76, 77 | shared (use layer 74's) |

21 full layers (0, 1, 2, 6, 10, …, 74), 57 shared. So **`index_topk_freq` is a frequency in layers,
not in tokens**: every token's top-k is recomputed at every full layer, and a shared layer reuses the
top-k of the nearest full layer below it, for the same query rows (`modeling` 433–436; the decoder layer
returns the indices it used, so a chain of shared layers passes them on, 616/734–743). Shared layers
have no indexer weights in the checkpoint and need no indexer key cache. The derivation is the
"IndexCache" cross-layer sharing the code cites (arXiv 2603.12201, `modeling` 365).

Each layer still has its **own** latent KV (`c`, `k_rope`); only the index set is shared.

### 2.6 Gathering the selected keys

With the absorbed form (§2.4) the selected keys for query i are the cached rows
`{(c_j, k_rope_j) : j ∈ topk(i)}`, 576 values each, shared by all 64 heads (MQA over the latent).
The output for head h is `W_UV_h · Σ_j P_h(i,j) c_j`. The reference instead scatters the indices
into a boolean mask over all T keys and runs dense SDPA/eager attention with it (`modeling` 440–449),
which is the same function.

### 2.7 MoE (layers 3–77)

`GlmMoeDsaTopkRouter` (`modeling` 502–527), `GlmMoeDsaMoE` (584–591), experts per `moe.py`
`grouped_mm_experts_forward` (the default implementation in transformers 5.15:
`get_correct_experts_implementation` picks `grouped_mm` when none is requested).

```
logits = fp32(x) · fp32(gate.weight)ᵀ                 # gate.weight BF16 [256, 6144]; fp32 GEMM     (504)
s      = sigmoid(logits)                               # fp32 [256]                                  (505)
sel    = s + e_score_correction_bias                   # bias F32 [256]                              (506)
         # n_group = topk_group = 1: the group step keeps all 256 (507–520)
idx    = top-8 of sel                                   # sorted=False                                (521)
w      = s[idx] / (Σ s[idx] + 1e-20) · 2.5              # fp32; the bias selects but does not weight  (522–526)
y_e    = down_e( silu(gate_e(x)) ⊙ up_e(x) )            # per expert, bf16 out
routed = bf16( Σ_{k=1..8} fp32(y_{idx_k}) · w_k )       # fp32 sum over the 8 slots, one rounding      (moe.py 461–477)
shared = down_s( silu(gate_s(x)) ⊙ up_s(x) )            # shared expert, intermediate 2048, BF16 weights
out    = routed + shared                                # bf16 add                                     (590)
```

- The bias `e_score_correction_bias` is large: about **34.2–34.6** in layer 3 (sigmoid scores are in
  (0, 1)). At that magnitude bf16's spacing is 0.25, so the selection sum must be fp32 as the
  reference does; in bf16 it would lose the scores entirely.
- `sorted=False` leaves slot order unspecified; with the fp32 sum the result can depend on it at the
  last bit. The older eager path (`modeling` 549–567) instead rounds each weighted expert output to
  bf16 and accumulates them in bf16 in expert-id order. Neither is "the" reference for the last bit.
- The shared expert reads the same normed `x` (`residuals` at 585 is the MoE's input, not the layer's
  residual).
- In this checkpoint the routed experts are NVFP4 with NVFP4 activations (§5), so the bf16 formula
  above is the reference only for a dequantised model; see §5.4 for the quantised one.

### 2.8 Dense MLP (layers 0–2) and the shared expert

`GlmMoeDsaMLP.forward`, `modeling` 484–486: `down(silu(gate(x)) * up(x))`, intermediate 12,288 for
layers 0–2 (FP8 in the checkpoint, §5.3) and 2048 for each layer's shared expert (BF16). No clamp.

### 2.9 Final norm and head

`model.norm` (RMSNorm, eps 1e-5) on every row, then `lm_head` [154880, 6144] BF16. The reference
emits bf16 logits (`modeling` 810–812: "do not upcast them to float"). Rows 154,856–154,879 of the
head are padding (no token has those ids, §7).

## 3. The MTP layer (78)

transformers does **not** implement it: `_keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]`
(`modeling` 659). Its tensors in the checkpoint (all BF16, none quantised; `hf_quant_config.json`
`exclude_modules: ["model.layers.78*"]`):

| Tensor | Shape |
|---|---|
| `enorm.weight`, `hnorm.weight` | [6144] |
| `eh_proj.weight` | [6144, 12288] |
| a full decoder layer: `input_layernorm`, `self_attn.*` **including a full indexer** (`wq_b`, `wk`, `k_norm` w+b, `weights_proj`), `post_attention_layernorm`, `mlp.gate` (+ bias), `mlp.shared_experts.*` | as for layers 3–77 |
| `mlp.experts.{0..255}.{gate,up}_proj.weight` | **BF16 [2048, 6144]** |
| `mlp.experts.{0..255}.down_proj.weight` | **BF16 [6144, 2048]** |
| `shared_head.norm.weight` | [6144] |

There is no `embed_tokens` and no `shared_head.head` under layer 78: the MTP layer shares the main
model's embedding and `lm_head`. Its 768 expert matrices are unquantised BF16 (no `weight_scale`,
`weight_scale_2` or `input_scale`), **~18.0 GiB** for the experts alone (256 × 3 × 2048 × 6144 × 2 B),
about 4 % of the checkpoint, kept at full precision because ModelOpt excluded the layer.

The forward, as DeepSeek-V3's MTP and the serving engines define it (*inferred* from the tensor names
and the DeepSeek-MTP convention; transformers has no code for it; see §8 for the engines' details):

```
# to draft token t+2 given the main model's hidden state for row t and the token t+1 just sampled
e   = enorm(embed_tokens[token_{t+1}])                 # RMSNorm eps 1e-5; vLLM zeroes e at position 0
g   = hnorm(h_t)                                        # h_t: the main model's hidden state for row t AFTER model.norm (vLLM)
x   = cat(e, g) · eh_projᵀ                              # [12288] → [6144]; embedding first, hidden second
x   = decoder_layer_78(x)                               # attention with its own KV cache and indexer, MoE
logits = lm_head(shared_head.norm(x))
```

With `num_nextn_predict_layers 1` the layer is iterated for k > 1 drafts, feeding its own output back
as `h`. **`index_share_for_mtp_iteration: true`**: per vLLM's `set_skip_topk`, "step 0 sets skip=False
(compute own indices), steps 1+ set skip=True": the MTP layer's indexer runs on the first draft step
and later draft steps reuse that top-k. It does not reuse the main model's top-k. Serving recipes run
3–5 draft tokens (§8).

## 4. What must be cached per token

Per token, per layer, for the absorbed form:

| Item | Values | bf16 bytes | Which layers |
|---|---:|---:|---|
| KV latent `c` (after `kv_a_layernorm`) | 512 | 1024 | all 78 (+ MTP) |
| `k_rope` (after RoPE) | 64 | 128 | all 78 (+ MTP) |
| indexer key (after `k_norm` and RoPE) | 128 | 256 | the 21 full layers (+ MTP) |

So **1152 B per token per layer** of attention state at bf16, plus **256 B** on full-indexer layers.
Whole model: 78 × 1152 + 21 × 256 = **95,232 B ≈ 93 KiB per token** (main layers only; the MTP layer
adds 1408 B). 128k tokens of context are 11.4 GiB at bf16. The latent is shared by all heads, so
tensor parallelism by heads does not split it: every rank holds it whole unless the sequence is
split across ranks.

The checkpoint declares `kv_cache_quant_algo: FP8` (`hf_quant_config.json`; `kv_cache_scheme`
static 8-bit float in `config.json`) but ships **no `k_scale`/`v_scale` tensors**, so a reader that
honours it uses a scale of 1.0 (*inferred*). For "exact" work the cache stays bf16.

What the serving engines store instead (§8): vLLM's `fp8_ds_mla` is **656 B** per token per layer
(512 FP8 latent values + 4 fp32 scales, one per 128, + the 64 rope values in bf16), and both vLLM and
SGLang keep the indexer key as **132 B** (128 FP8 + one 4-byte scale per 128). In that form the
model is 78 × 656 + 21 × 132 = 53,940 B ≈ 52.7 KiB per token.

## 5. The checkpoint's formats

### 5.1 Which tensors are in which format

| Class | Format | Tensors |
|---|---|---|
| Routed experts, layers 3–77 | **NVFP4** (weights and activations, group 16) | `experts.N.{gate,up,down}_proj.{weight U8, weight_scale F8_E4M3, weight_scale_2 F32, input_scale F32}` |
| Dense MLPs, layers 0–2 | **FP8 E4M3, per-tensor** (weights and activations, static) | `mlp.{gate,up,down}_proj.{weight F8_E4M3, weight_scale F32 [], input_scale F32 []}` |
| Everything else | **BF16** | embeddings, head, all norms, all attention and indexer weights, routers (`gate.weight` BF16, `e_score_correction_bias` **F32**), shared experts, the whole MTP layer including its 256 experts |

`config.json`'s `quantization_config` (`quant_method: modelopt`, `quant_algo: MIXED_PRECISION`)
lists the same split: `group_0` (FP8 w/a, 12 targets for layers 0–2) and `group_1` (NVFP4 w/a, group
16, 75 targets `model.layers.{3..77}.mlp.experts`), `ignore: ["model.layers.78*"]`,
`packed_modules_mapping: gate_up_proj → [gate_proj, up_proj]`.

### 5.2 NVFP4 (routed experts)

Shapes for one expert (K is the input dimension):

| Tensor | gate / up | down |
|---|---|---|
| `weight` U8 | [2048, 3072] (K = 6144, two values per byte) | [6144, 1024] (K = 2048) |
| `weight_scale` F8_E4M3 | [2048, 384]: one per 16 along **K** | [6144, 128] |
| `weight_scale_2` F32 | [] per tensor | [] |
| `input_scale` F32 | [] | [] |

**Weight dequantisation:**

```
code(n, k) = (weight[n, k // 2] >> (4 · (k mod 2))) & 0xF       # low nibble = even k, high nibble = odd k
E2M1(code) = (−1)^(code>>3) · {0, 0.5, 1, 1.5, 2, 3, 4, 6}[code & 7]
W(n, k)    = E2M1(code(n, k)) · float(E4M3(weight_scale[n, k // 16])) · weight_scale_2
```

Checks on the tensors (layers 3, 40, 77; experts 0, 17, 100, 255): with this table, every 16-block's
largest |E2M1| is exactly 6 (100 % of blocks), as ModelOpt's `scale = amax_block / 6` rounding
implies; block scales run up to 448 (the E4M3 maximum), consistent with
`weight_scale_2 = amax_tensor / (6 · 448)`. The nibble order cannot be told from such statistics (a
block holds both nibbles of 8 bytes); low-nibble-first is the ModelOpt/vLLM convention (§8) and the
same as ds41f's MXFP4 experts.

Two regularities an engine can use (checked over all 256 experts of layers 3, 40 and 77):

- **gate and up share `weight_scale_2`** in every expert, so `[gate; up]` can be one GEMM with one
  global scale (the `gate_up_proj` packing in `packed_modules_mapping`).
- **`input_scale` is the same for all 256 experts of a layer** (one value for gate/up, another for
  down), i.e. ModelOpt calibrated one activation scale per layer and projection. Examples:
  layer 3 gate/up 1.301e-4, down 3.749e-4; layer 77 gate/up 2.593e-3, down 0.497.

**Activation quantisation** (static global scale, dynamic block scales; the formula is vLLM's
`scaled_fp4_quant` as used by its ModelOpt NVFP4 linear and MoE methods; §8 has the exact sources):

```
g        = 1 / input_scale                                   # fp32 global scale (vLLM: input_scale_inv; max over experts, here all equal)
for each block b of 16 consecutive activations along K:
  sf_b   = E4M3( g · amax_b / 6 )                            # e4m3 round-to-nearest, saturating
  xq_k   = E2M1_rn( x_k · (g / float(sf_b)) )                 # round to nearest even, saturating at ±6
x ≈ xq · float(sf_b) / g  =  xq · float(sf_b) · input_scale
y(n) = alpha · Σ_k xq_k · float(sf_b(k)) · E2M1(w_nk) · float(E4M3(weight_scale[n, k//16])),
alpha = input_scale · weight_scale_2                          # fp32 accumulate, output bf16
```

So in the checkpoint's own numerics the expert input (the post-attention-normed `x`) and the
SwiGLU output entering `down_proj` are each re-quantised to E2M1 with E4M3 per-16 scales.
`input_scale = amax_calib / (6 · 448)` (*inferred* from ModelOpt's convention; `in·448·6` gives
plausible calibration maxima, e.g. ~0.35 for layer 3's normed input and ~1336 for layer 77's
down-projection input).

### 5.3 FP8 (dense MLPs of layers 0–2)

Per-tensor scales, static activations (ModelOpt FP8, as vLLM's `ModelOptFp8LinearMethod` applies it):

```
W   = float(E4M3(weight)) · weight_scale                     # weights saturate at |448| (observed)
xq  = E4M3_rn_sat( x / input_scale )
y   = (xq · E4M3(weight)ᵀ) · (input_scale · weight_scale)    # fp32 accumulate, bf16 out
```

Values: layer 0 gate 1.029e-3, up 5.493e-4, down 9.940e-4 (`weight_scale`); `input_scale` 3.05e-3
(gate/up, shared) and 1.98e-4 (down). gate and up share `input_scale` but not `weight_scale`.

### 5.4 What "the reference" is for a quantised model

transformers' modeling file is the reference for the architecture and the bf16 math; it is not a
reference for this checkpoint's quantised GEMMs. With the NVFP4 weights the publisher's intended
numerics are ModelOpt's: weights dequantised as in 5.2/5.3 **and activations fake-quantised** as
above. A CPU reference can be built from the formulas here (dequantise the weights; quantise
activations per block exactly as written), and compared with vLLM's `nvfp4_emulation_utils` path
(§8). Whether the engine quantises activations at all is a decision for
[quantisation.md](quantisation.md); this document only fixes what the checkpoint means.

## 6. Against DeepSeek-V4.1-Flash

The DeepSeek side from ds41f's `DESIGN.md` and `docs/findings.md` (§ The model).

### Same in kind

| | DeepSeek-V4.1-Flash | GLM-5.3 |
|---|---|---|
| Low-rank query | q LoRA 1280, norm, then up-projection | q LoRA **2048**, RMSNorm (eps 1e-6), up-projection |
| Shared KV latent, MQA over it | one 512-wide latent | one 512-wide latent (+ separate 64-dim rope key) |
| Lightning indexer | 32 heads × 128, `relu(q·k)` weighted by a per-head projection of x | the same formula, 32 × 128 |
| Router | fp32, bias for selection only, normalised top-k, scaled | fp32, bias for selection only, normalised, scaled |
| Shared expert | 1 | 1 |
| Activation | SwiGLU | SwiGLU |

### Different in dimension or detail

| | DeepSeek-V4.1-Flash | GLM-5.3 |
|---|---|---|
| Layers | 40 (20 encoder + 20 decoder, CED) | **78**, uniform decoder stack |
| Hidden | 5120 | **6144** |
| Head dims | 64 heads × 512; K = V = the latent; RoPE on the last 64 of 512 | 64 heads; **QK 256 = 192 nope + 64 rope, V 256**, both expanded from the latent by `kv_b_proj` [28672, 512] (absorbable) |
| Output projection | grouped: `wo_a` 8 groups × 1024, then `wo_b` 8192 → 5120 | plain `o_proj` [6144, 16384] |
| Attention sink | learned per head | **none** |
| Sliding window | every layer, 128, FP8 KV | **none** |
| Global KV | compressed (ratio 2 or 1), FP4 + E4M3/16, 890 B/token in all | **uncompressed**, every token, 576 values per layer (1152 B bf16) × 78 layers |
| RoPE | adjacent pairs as complex numbers in **fp32**, rotated in place (layout kept), and the inverse rotation applied to the attention output (`ref/inference/model.py` `apply_rotary_emb`) | adjacent pairs too, but in **bf16** with bf16 cos/sin, the output written **de-interleaved** (§2.4.1), no inverse rotation; θ = 8e6, no YaRN |
| Indexer inputs | FP4 q and k (ue8m0/32), over compressed positions | **bf16/fp32** in the reference, over every position; `k_norm` LayerNorm with bias; rope on the first 64 of 128 |
| Top-k | 512 (+ a 16,384-position candidate pool at layer 20 for reindex layers) | **2048**, no pool |
| Index reuse | CSA2 Full / Reindex / Reuse modes per layer | **full / shared**: 21 layers compute, 57 reuse the last full layer's top-k |
| Dense below sparse | n/a | each query is dense while its position < 2048 |
| MoE | every layer; 384 experts top-6, intermediate 2304 | **3 dense layers** (12,288) then 75 MoE layers; **256 experts top-8**, intermediate 2048 |
| Router function | `sqrt(softplus)`, ×1.5, a second bias for image tokens | **sigmoid**, ×**2.5**, one bias (values ~34) |
| Expert clamps | up ±10, gate ≤ 10 | **none** |
| Expert format | MXFP4 (E2M1 + ue8m0 per 32 along K), activations FP8 e4m3/32 pow-2 | **NVFP4**: E2M1 + **E4M3 per 16** along K + **F32 per-tensor** global; activations **NVFP4** with a static global scale |
| Dense/shared weights | FP8 with 32×32 ue8m0 blocks | shared experts and attention **BF16**; dense MLPs FP8 per-tensor |
| Norm eps | 1e-20 | 1e-5 (layer norms), 1e-6 (latent norms, indexer LayerNorm) |
| Residual | mHC, 4 copies, fp32 coefficients | a plain **bf16** residual |
| Vocabulary | 129,280 rows (ds41f `docs/layout.md`) | 154,880 rows, 154,856 used |

### Absent in GLM-5.3

Engram (n-gram memory tables, NVMe), mHC (hyper-connections, Sinkhorn), the compressor (and with it
CED, the encoder/decoder split and bounded replay), sliding-window attention, attention sinks, the
DSpark drafter (Markov head, confidence head) and the vision tower.

### New in GLM-5.3

One **MTP layer** (a full decoder layer with its own indexer and 256 **BF16** experts, ~18 GiB) for
speculative decoding; the three leading dense layers; cross-layer top-k sharing; the expanded
(192+64)/256 head dimensions; interleaved RoPE; the LayerNorm in the indexer.

## 7. Tokenizer and chat template

### 7.1 tokenizer.json

| Field | Value |
|---|---|
| model | **BPE**, byte-level (`ByteLevel` pre-tokenizer stage and decoder), `byte_fallback false`, no `unk_token` |
| **`ignore_merges`** | **true**: a pre-token that is itself a vocabulary entry is emitted as that one id without running merges. It matters: **893** vocabulary entries would not be reached by merges alone (mostly space-prefixed CJK, e.g. `ĠèĢĮ`, which merges would give as three tokens) |
| vocab | 154,820 entries, ids 0–154,819 |
| merges | 321,649, stored as **two-element JSON arrays** `["Ġ", "Ġ"]`, not as `"a b"` strings |
| normalizer | none |
| pre_tokenizer | `Sequence[ Split(Regex, behavior Isolated, invert false), ByteLevel(add_prefix_space false, use_regex false) ]` |
| post_processor | `ByteLevel` (offsets only; it adds no tokens) |
| decoder | `ByteLevel` |
| added tokens | 36, ids 154,820–154,855, all `normalized: false`, `lstrip/rstrip/single_word: false` |

The split regex (one pattern, the Qwen2/GLM-4 one):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Added tokens (`S` = `special: true`; all of them are cut out of the text before pre-tokenisation,
special or not):

| id | token | | id | token | |
|---:|---|---|---:|---|---|
| 154820 | `<\|endoftext\|>` | S, eos, pad | 154838 | `<\|code_prefix\|>` | |
| 154821 | `[MASK]` | S | 154839 | `<\|code_middle\|>` | |
| 154822 | `[gMASK]` | S | 154840 | `<\|code_suffix\|>` | |
| 154823 | `[sMASK]` | S | 154841 | `<think>` | |
| 154824 | `<sop>` | S | 154842 | `</think>` | |
| 154825 | `<eop>` | S | 154843 | `<tool_call>` | |
| 154826 | `<\|system\|>` | S | 154844 | `</tool_call>` | |
| 154827 | `<\|user\|>` | S, eos | 154845 | `<tool_response>` | |
| 154828 | `<\|assistant\|>` | S | 154846 | `</tool_response>` | |
| 154829 | `<\|observation\|>` | S, eos | 154847 | `<arg_key>` | |
| 154830–154837 | image/video/audio/transcription begin/end | S | 154848 | `</arg_key>` | |
| | | | 154849 | `<arg_value>` | |
| | | | 154850 | `</arg_value>` | |
| | | | 154851 | `/nothink` | |
| | | | 154852–154855 | `<\|begin_of_box\|>`, `<\|end_of_box\|>`, `<\|image\|>`, `<\|video\|>` | |

`tokenizer_config.json`: `tokenizer_class TokenizersBackend`, `eos_token` and `pad_token`
`<|endoftext|>`, no bos token, `padding_side left`, `clean_up_tokenization_spaces false`,
`model_max_length 1048576`. Encoding plain text adds **no** special tokens (`tok("hi") → [6023]`).
Generation stops on any of 154820, 154827 (`<|user|>`) or 154829 (`<|observation|>`, after a tool
call).

Tokenisations checked with transformers/`tokenizers` on CPU:

| Text | Ids | Pieces |
|---|---|---|
| `Hello world` | 9703, 1879 | `Hello` `Ġworld` |
| ` Hello  world\n\n` | 21873, 220, 1879, 271 | `ĠHello` `Ġ` `Ġworld` `ĊĊ` |
| `你好，世界` | 109377, 3837, 99011 | three tokens |
| `12345678` | 108714, 100461, 21, 100928 | `123` `45` `6` `78` (digits split 1–3 by the regex, then BPE) |
| `don't I'LL` | 14995, 944, 358, 6, 4086 | `don` `'t` `ĠI` `'` `LL` |
| `\t\tcode()\r\n` | 197, 43094, 3568 | `ĉ` `ĉcode` `()čĊ` |
| `naïve café 🙂` | 3376, 37377, 586, 51609, 27392 | `na` `Ã¯` `ve` `ĠcafÃ©` `ĠðŁĻĤ` |
| `<think>x</think>` | 154841, 87, 154842 | |
| `<tool_call>f<arg_key>a</arg_key>` | 154843, 69, 154847, 64, 154848 | |
| `[gMASK]<sop><\|user\|>` | 154822, 154824, 154827 | |

### 7.2 chat_template.jinja

Rendered by transformers' sandboxed Jinja with `trim_blocks` and `lstrip_blocks`; the renders below
are transformers' own output.

**Prefix.** Always `[gMASK]<sop>`, then always a reasoning-effort system turn:

```
{% set effective_reasoning_effort = reasoning_effort if reasoning_effort in ['low','high'] else 'max' %}
<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}
```

So the default is `Reasoning Effort: Max`; only `low` and `high` change it (anything else, `medium`
included, falls back to `Max`). The `is not none` test on line 3 is always true.

**Tools** (when `tools` is given), a second system turn right after it:

```
<|system|>
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "get_weather", "description": "Get weather", "parameters": {...}}
</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>
```

One line per tool; each is the tool's `function` object (or the tool itself when it has no
`function` key) serialised key by key in the given order, skipping `defer_loading` and `strict`,
values via `tojson(ensure_ascii=False)` (Python `json.dumps` separators `", "` and `": "`). Tools with
`defer_loading: true` are left out of the list.

**Messages**, in order, with **no newline after a role token**:

| Role | Rendering |
|---|---|
| system | `<\|system\|>` + text (wherever it appears in the list; after the effort/tools turns) |
| user | `<\|user\|>` + text |
| assistant | `<\|assistant\|>` + `<think>R</think>` or `<think></think>` + `content.strip()` + tool calls |
| tool | a run of consecutive tool messages renders once as `<\|observation\|>` + one `<tool_response>…</tool_response>` per result |

Text is `visible_text(content)`: strings as-is; for a list of parts, `text` parts concatenated and
media parts replaced by `<reminder>You are unable to process this {image|video|audio} because you
don't have multi-modal input ability. Try different methods.</reminder>`.

**Thinking.** The reasoning `R` is `reasoning_content` when it is a string, else the text between
`<think>` and `</think>` if the content contains `</think>` (the content then becomes what follows
`</think>`). It is kept in **every** past assistant turn by default (`clear_thinking` defaults to
**false**); with `clear_thinking=true` only assistant turns after the last user message keep it and
earlier ones render `<think></think>`. An assistant turn with no reasoning renders `<think></think>`.
There is **no `enable_thinking` flag**: the generation prompt is always `<|assistant|><think>`, so
the model always starts in thinking mode (a non-thinking reply would have to be forced by the
front-end, e.g. by appending `</think>`; *inferred*, the template has no such path; the vocabulary
also has a `/nothink` token that the template never emits).

**Tool calls** in an assistant turn, right after the content, no separators:

```
<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>
```

`V` is the raw string for string arguments and `tojson` otherwise (`2` → `2`, objects as JSON).
`arguments` must be a mapping (the template calls `.items()`); an OpenAI client's JSON-string
arguments must be parsed first.

**Tool results.** Inside one `<|observation|>`, results are emitted as
`<tool_response>TEXT</tool_response>` each. When every result in the run has an id
(`tool_call_id` or `id`), every id is unique and matches one of the preceding assistant's tool calls,
the results are **reordered to the order of the calls**; otherwise they keep message order. Results
can also be lists of `{output, id}` entries or `tool_reference` items (the latter expand to the
referenced tools' JSON inside `<tool_response><tools>…</tools></tool_response>`).

**Example** (system, user, assistant with reasoning and a tool call, tool result, assistant, user;
`add_generation_prompt=True`):

```
[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>
# Tools
…</tool_call><|system|>You are helpful.<|user|>Weather in Paris?<|assistant|><think>Need tool.</think>Let me check.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call><|observation|><tool_response>{"t":20}</tool_response><|assistant|><think>Got it.</think>It is 20C.<|user|>Thanks<|assistant|><think>
```

Minimal: `[{"role":"user","content":"hi"}]` →
`[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>` →
`[154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487, 154827, 6023, 154828, 154841]`.

With `clear_thinking` false (the default) a conversation's rendering is append-only across turns
(the previous turns render identically, reasoning included), which is what exact prefix reuse needs.

### 7.3 What ds41f's front-end would need changed

ds41f's tokenizer (`frontend/src/tokenizer/mod.rs`) is a byte-level BPE with a leftmost-longest
added-token trie, and `pretok.rs` hand-codes DeepSeek-V4.1's three chained `Split` patterns. For
GLM-5.3:

1. **Merges format.** `JsonModel.merges` is `Vec<String>` parsed with `split_once(' ')`; GLM's
   merges are `[a, b]` arrays, so deserialisation fails as written. Accept both forms.
2. **`ignore_merges`.** Before BPE on a piece, look the whole piece up in the vocabulary and emit it
   if present. Without this, 893 vocabulary tokens are tokenised differently (§7.1).
3. **Pre-tokeniser.** Replace `pretok.rs` with the single GLM pattern above (one pass; no separate
   digit or CJK stages; `\p{N}{1,3}` chunks digits inside the same alternation; letters are `\p{L}`
   only, no `\p{M}`; the contraction alternative is case-insensitive). The hand-written matcher's
   structure (leftmost-first alternatives, the `\s+(?!\S)` rule) carries over; the class tables
   need L, N and whitespace.
4. **Added tokens.** All 36 are `normalized: false`, so only the raw trie is used; ids are above the
   vocabulary (154,820+) and not in `model.vocab`, which the loader already handles. Non-special
   added tokens (`<think>`, `<tool_call>`, …) must also be cut out (they are, by the trie).
5. **Encoding.** `encoding.rs` ports DeepSeek's `encoding.py` (DSML tool calls, `<｜User｜>` etc.);
   it must be replaced by a port of `chat_template.jinja` as specified in §7.2: role tokens without
   newlines, `<think>…</think>` handling with `clear_thinking`, the reasoning-effort system turn,
   the XML-ish `<tool_call>`/`<arg_key>`/`<arg_value>` calls and `<|observation|>` results with
   their id-based reordering. `pyjson.rs` (Python `json.dumps` behaviour) is reusable for `tojson`
   with `ensure_ascii=False`.
6. **Parsing output.** Reasoning ends at `</think>` (id 154842); tool calls are
   `<tool_call>NAME(<arg_key>K</arg_key><arg_value>V</arg_value>)*</tool_call>`, with `V` a raw
   string unless the tool's schema says otherwise (vLLM's/SGLang's GLM tool parsers decide JSON vs.
   string per parameter type; *inferred*, §8). Stop on 154820, 154827, 154829.
7. **Vocabulary size** for sampling: mask ids ≥ 154,856 (padding rows of the head).

## 8. Reference implementations

Surveyed on 2026-10-13 from the projects' main branches on GitHub and their docs (web reading, not
run here).

### vLLM

- **Model.** The registry maps `GlmMoeDsaForCausalLM` to `vllm.models.deepseek_v32`, whose
  `__init__.py` aliases it to `DeepseekV32ForCausalLM` in `vllm/models/deepseek_v32/nvidia/model.py`
  on CUDA (`enable_glm52_low_latency_gemm` when `model_type == "glm_moe_dsa"`). On other platforms it
  falls back to a subclass of `DeepseekV2ForCausalLM` in `model_executor/models/deepseek_v2.py`.
- **Shared layers.** `DeepseekV32Attention.__init__` (`vllm/models/deepseek_v32/attention.py`)
  computes `skip_topk = max(layer_id − index_skip_topk_offset + 1, 0) % index_topk_freq != 0`, the
  same rule as transformers (`indexer_types` itself is not read; `index_topk_pattern` overrides).
  Skipped layers have no indexer and read a model-wide `topk_indices_buffer` that the last full layer
  wrote. The MTP layer always has an indexer.
- **Indexer numerics differ from transformers.** `layers/sparse_attn_indexer.py`: q goes through
  `fused_indexer_q_rope_quant` (RoPE, then FP8 with an e8m0 scale folded into the head weights); k is
  quantised to FP8 with a scale per 128 into a 132 B/token cache (`indexer_k_quant_and_cache`);
  logits come from DeepGEMM `fp8_fp4_mqa_logits` (prefill) and `fp8_fp4_paged_mqa_logits` (decode).
  There is no Hadamard rotation. RoPE style is `is_neox_style = not indexer_rope_interleave`, i.e.
  interleaved here. Unused top-k slots are −1; there is a short-prefill dense path (threshold not
  read).
- **Attention.** `FLASHMLA_SPARSE` backend over the 576-wide latent (`auto`/`bfloat16`, or
  `fp8_ds_mla` at 656 B/token; `nvfp4_ds_mla` on SM100 only).
- **MTP.** `vllm/models/deepseek_v32/nvidia/mtp.py` (`DeepseekV32MTP`) and the generic
  `model_executor/models/deepseek_mtp.py`: zero the embedding at position 0, `enorm(embeds)`,
  `hnorm(prev)` with the **post-final-norm** hidden state ("Recycle the POST-final-norm hidden"),
  `cat`, `eh_proj`; `embed_tokens` shared with the main model; `index_share_for_mtp_iteration` as in
  §3. The vLLM recipe for GLM-5.3 runs `--speculative-config.method mtp
  --speculative-config.num_speculative_tokens 5 --tool-call-parser glm47 --reasoning-parser glm47
  --kv-cache-dtype fp8`.
- **ModelOpt NVFP4.** `break_fp4_bytes` (`quantization/utils/nvfp4_emulation_utils.py`) stacks
  `(low, high)`: **low nibble = even element**, LUT `[0, .5, 1, 1.5, 2, 3, 4, 6]`, bit 3 the sign;
  dequant `fp4 · block_scale · global_scale`. `ref_nvfp4_quant`: block scale
  `e4m3(clamp(global · amax / 6, ±448))` over 16. Linear: `input_global_scale = input_scale.max()`,
  quantise with `1 / input_global_scale`, `alpha = input_global_scale · weight_scale_2`. MoE
  (`fused_moe/oracle/nvfp4.py`): per-expert `1 / a13_scale` (all equal in this checkpoint, §5.2),
  `w13_weight_scale_2[:, 0]` (gate's; warns if gate ≠ up, which never happens here). The
  `model.layers.78*` exclude is matched by `fnmatch`, so the MTP layer loads as BF16.

### SGLang

- **Model.** `python/sglang/srt/models/deepseek_v2.py` with `is_glm_moe_dsa(config)` branches;
  shared layers via `dsa_layer_skips_topk(config, layer_id)` and an `IndexTopKShareState` /
  `prev_topk_indices` threaded through the layer loop.
- **Indexer.** `layers/attention/dsa/dsa_indexer.py` (`nsa/nsa_indexer.py` is a shim): q to FP8 per
  128 (`act_quant`), a 132 B index-key cache (page 64), DeepGEMM `fp8_mqa_logits` /
  `fp8_paged_mqa_logits`. A Hadamard `rotate_activation` exists but is skipped on the fused CUDA
  paths. When `seq_lens.max() ≤ index_topk`, prefill skips the logits and emits identity indices,
  i.e. dense, as in transformers.
- **MTP.** `deepseek_nextn.py` with EAGLE (`IndexTopKShareState.from_mtp_carry`). The GLM-5 cookbook
  uses EAGLE steps 3 / topk 1 / draft 4 (5/1/6 or 1/1/2 suggested for GLM-5.3), parsers `glm45`
  (reasoning) and `glm47` (tools). The `nvidia/GLM-5.3-NVFP4` model card gives only an SGLang command
  (`--tp-size 8 --quantization modelopt_fp4 --tool-call-parser glm47 --reasoning-parser glm45`), no
  MTP, and FP8→NVFP4 scores of MMLU-Pro 87.81→87.88, GPQA 91.79→92.68, LiveCodeBench 52.95→53.55,
  AIME25 99.38→99.53.

### zai-org

`github.com/zai-org/GLM-5` has no reference inference code (no `inference/model.py` as DeepSeek
ships), only examples and serving recipes. It names the cross-layer sharing "IndexShare" (one indexer
per four sparse layers) and says GLM-5.3 is built on the GLM-5.2 base.

### Which to take as the reference

- **Architecture and bf16 math: transformers 5.15** (`modeling_glm_moe_dsa.py`). It is the only
  complete, readable definition, it runs on CPU, and its indexer is unquantised (fp32 scores), so it
  defines the top-k the model "means". It lacks the MTP layer, the quantised GEMMs and an absorbed
  MLA, and it is slow (it caches expanded K/V).
- **Quantised numerics: ModelOpt's convention as vLLM implements it** (§5.2–5.3; `break_fp4_bytes`,
  `ref_nvfp4_quant`, `scaled_fp4_quant`), since nothing in transformers runs this checkpoint's NVFP4.
- **MTP: vLLM's `DeepseekV32MTP`** (post-norm hidden, position-0 zeroing, `index_share_for_mtp_iteration`).
- **Not as a numerical reference: the serving engines' indexers**, which quantise q and k to FP8
  (vLLM with e8m0 scales, no Hadamard). An engine that keeps the indexer in bf16/fp32 will match
  transformers' top-k more closely than vLLM does; golden data for the top-k should come from the
  transformers formulas, with a tolerance for ties and near-ties at the 2048th score.

## 9. Surprises, in one list

- `head_dim: 64` is only the rotary dimension (overwritten in `__post_init__`).
- `q_a_layernorm` and `kv_a_layernorm` use eps **1e-6**, not `rms_norm_eps` 1e-5.
- RMSNorm multiplies by the weight in bf16 after rounding (two roundings).
- RoPE is interleaved but the reference writes its output de-interleaved, and computes in bf16 with
  bf16 cos/sin.
- The indexer's `k_norm` is a LayerNorm with bias; its rope is on the first 64 dims, the attention's
  on the last 64.
- `index_topk_freq`/`index_skip_topk_offset` are **per layer**: 21 full-indexer layers
  (0, 1, 2, 6, 10, …, 74), 57 sharing the last full layer's top-k.
- Attention is dense for every query at position < 2048, by construction of `min(2048, T)` and the
  causal mask, with no special case.
- `e_score_correction_bias` is ~34, so the selection must be fp32.
- Layer 78's experts are **BF16**, unquantised (~18 GiB): the MTP layer was excluded from NVFP4.
- All 256 experts of a layer share one `input_scale` per projection; gate and up share
  `weight_scale_2`.
- The KV cache is declared FP8 with no scales shipped.
- vLLM and SGLang quantise the indexer's q and k to FP8; transformers keeps them bf16 with fp32 scores, so
  the engines' top-k is not the reference's.
- zai-org publishes no reference inference code; the MTP forward is defined only by the engines, which feed it the
  post-final-norm hidden state.
- The template always prepends `Reasoning Effort: Max` (only `low`/`high` override), always opens
  `<think>`, has no `enable_thinking`, and keeps past reasoning by default.
- `tokenizer.json` uses `ignore_merges: true` and array-form merges.
