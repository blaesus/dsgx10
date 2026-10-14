# altar-1: the model, exactly

The math of altar-1 (`AikidoSec/altar-1` at `5d591cd98958e4e7e429517887e80ca3bb7ce2c4`) as a **delta from
GLM-5.3**, whose exact math is in [glm53's model.md](../../glm53-4xgx10-switchless/docs/model.md) (cited below as
*glm53 § n*). The new part is the INT4 format: its storage, its dequantisation to the bit, what transformers and
vLLM compute from it, and so what "the model" is for this configuration's exactness standard. Then the routing over
168 experts, the BF16 layers, the MTP layer, the chat template and generation config, and the reference
implementation. Written 2026-10-14.

Sources, cited below by short name:

| Short name | What |
|---|---|
| `ct` | compressed-tensors **0.18.0** (PyPI wheel, sha256 `91168237…`, 2026-08-08), the newest release before transformers 5.16 (2026-08-26): `compressors/pack_quantized/base.py`, `compressors/pack_quantized/helpers.py`, `quantization/lifecycle/forward.py`, `quantization/lifecycle/forward_helpers.py`, `compressors/base.py`, `compressors/model_compressors/model_compressor.py`. Compared with 0.15.0, 0.16.0, 0.17.1 and 0.19.0 (§4.1). |
| `tf` | transformers **5.16.1** (PyPI wheel, 2026-08-26): `quantizers/quantizer_compressed_tensors.py`, `integrations/compressed_tensors.py`, `models/glm_moe_dsa/modeling_glm_moe_dsa.py`, `conversion_mapping.py`. The checkpoint says `transformers_version 5.16.0.dev0`; transformers requires compressed-tensors ≥ 0.15.0 (`utils/import_utils.py` line 151) and pins no upper version. Compared with 5.16.0, 5.17.0, 5.18.0, 5.19.0 and main at `34f41636` (2026-10-14) |
| `vllm` | vLLM main at `bb1438a93f8873c30176d93267d0647f8e4c7b83` (2026-10-14) and release v0.31.0 (`db9527a4`), read on GitHub, not run |
| checkpoint | altar-1's `config.json`, `generation_config.json`, `tokenizer_config.json`, `chat_template.jinja`, `reap_prune_plan.json`, the 39 shard headers (range requests), and sampled tensor data compared with `cyankiwi/GLM-5.3-AWQ-INT4` (`20f265b6`), `zai-org/GLM-5.3` (`c5087a76`) and `zai-org/GLM-5.3-BF16` (`9d2398f4`) on 2026-10-14 (scripts kept out of the repo) |

Anything not read directly from one of these is marked *inferred*.

## Contents

1. [What is identical to GLM-5.3](#1-what-is-identical-to-glm-53)
2. [Dimensions that change](#2-dimensions-that-change)
3. [The tensors and their formats](#3-the-tensors-and-their-formats)
4. [The INT4 format, exactly](#4-the-int4-format-exactly)
5. [The BF16 tensors are cyankiwi's, not GLM-5.3's](#5-the-bf16-tensors-are-cyankiwis-not-glm-53s)
6. [Routing over 168 experts](#6-routing-over-168-experts)
7. [Layers 3 and 77, and the MTP layer](#7-layers-3-and-77-and-the-mtp-layer)
8. [Tokenizer, chat template, generation config](#8-tokenizer-chat-template-generation-config)
9. [The reference implementation](#9-the-reference-implementation)
10. [Surprises, in one list](#10-surprises-in-one-list)

## 1. What is identical to GLM-5.3

`config.json` equals GLM-5.3's (`zai-org/GLM-5.3` at `c5087a76`) except for `n_routed_experts` (168 against 256),
`quantization_config`, and fields that transformers' serialisation writes out without changing the math:
`head_dim` 64 (zai's file says 192; `__post_init__` sets it to `qk_rope_head_dim` = 64 either way, glm53 § 1),
`bos_token_id` 0, `mlp_bias` false, an explicit `layer_types` list, and `transformers_version` (checked 2026-10-14;
the configuration's own reading of the config on 2026-10-14 found the same). transformers' modeling file
is the same math in 5.15 (glm53's reference) and 5.16.1: the diff is a restructuring of the `q_proj`/`q_a_proj`
construction and of the MoE/MLP choice, and one `input.to(weight.dtype)` cast in the `grouped_mm` fallback, a no-op
for bf16. So these sections of glm53's model.md hold unchanged, line numbers shifted by +10 after `modeling` line
339:

- glm53 § 1, dimensions, except the MoE row (§2 below).
- glm53 § 2.1–2.6 and 2.8–2.9: embedding, RMSNorm (eps 1e-5, latent norms 1e-6), the decoder layer, MLA, the
  interleaved RoPE, the DSA indexer and its 21/57 full/shared pattern, the gathering, the dense MLP and shared
  expert formula, final norm and head (bf16 logits, rows 154,856–154,879 padding).
- glm53 § 2.7's formulas, with 168 in place of 256 (§6).
- glm53 § 3's MTP forward and § 4's per-token cache (1152 B a layer, + 256 B on full-indexer layers).
- glm53 § 7.1: `tokenizer.json` is byte-identical to GLM-5.3's (our reading), so the tokenizer, its
  `ignore_merges`, its pre-tokeniser and the 36 added tokens are unchanged; `tokenizer_config.json` differs from
  `zai-org/GLM-5.3`'s only by an added `"local_files_only": false` line, which does not affect encoding.
- glm53 § 8's notes on vLLM's and SGLang's attention, indexer and MTP code.

What does change: glm53 § 5 (the checkpoint's formats: NVFP4 and FP8 there, INT4 and BF16 here, §3–5), the number of
experts (§2, §6), which layers are BF16 (§7), the template (§8.1) and generation config (§8.2).

## 2. Dimensions that change

| Symbol | GLM-5.3 | altar-1 | Field |
|---|---:|---:|---|
| E (routed experts a MoE layer, and in the MTP layer) | 256 | **168** | `n_routed_experts` (`num_local_experts` via the config's `attribute_map`, `tf` `configuration_glm_moe_dsa.py` line 91) |
| Router weight | [256, 6144] BF16 | **[168, 6144]** BF16 | `mlp.gate.weight` |
| Router bias | [256] F32 | **[168]** F32 | `mlp.gate.e_score_correction_bias` |
| Experts a token, shared experts | 8, 1 | 8, 1 | `num_experts_per_tok`, `n_shared_experts` |

Expert ids are 0..167 in every MoE layer and the MTP layer. `reap_prune_plan.json` `layers["L"][j]` is the original
GLM-5.3 id of altar-1's expert j in layer L, for L = 3..78; the lists are sorted, so the renumbering keeps the
original order. The plan does not change the math; it matters only for comparisons with GLM-5.3 (§5).

## 3. The tensors and their formats

From the 39 shard headers (*checked*; 305.42 GiB of tensor data):

| Class | Layers | Format | Tensors (shapes for one module) |
|---|---|---|---|
| Routed experts | 4–76 | **INT4** | `experts.j.{gate,up}_proj.{weight_packed I32 [2048, 768], weight_scale BF16 [2048, 192], weight_zero_point I32 [256, 192], weight_shape I64 [2] = [2048, 6144]}`; `down_proj.{weight_packed [6144, 256], weight_scale [6144, 64], weight_zero_point [768, 64], weight_shape [6144, 2048]}` |
| `q_b_proj`, `kv_b_proj`, `o_proj` | 4–76 | **INT4** | `q_b_proj` packed [16384, 256], scale [16384, 64], zero point [2048, 64]; `kv_b_proj` packed [28672, 64], scale [28672, 16], zero point [3584, 16]; `o_proj` packed [6144, 2048], scale [6144, 512], zero point [768, 512] |
| Routed experts | 3, 77, 78 | BF16 | `experts.j.{gate,up}_proj.weight` [2048, 6144], `down_proj.weight` [6144, 2048], j = 0..167 |
| Attention | 0–3, 77, 78: all; 4–76: `q_a_proj`, `kv_a_proj_with_mqa`, norms, indexer | BF16 | as GLM-5.3 |
| Indexer | the 21 full layers and 78 | BF16 | `wq_b`, `wk`, `k_norm` (weight and bias), `weights_proj` |
| Dense MLPs | 0–2 | BF16 | [12288, 6144] / [6144, 12288] (FP8 per-tensor in NVIDIA's checkpoint, glm53 § 5.3) |
| Shared experts, routers, norms, head, embeddings | all | BF16 (bias F32) | as GLM-5.3 |
| MTP extras | 78 | BF16 | `eh_proj` [6144, 12288], `enorm`, `hnorm`, `shared_head.norm` [6144]; no `embed_tokens` or `shared_head.head` (shared with the main model, as glm53 § 3) |

The INT4 cost is 4 bits + 16/32 (scale) + 4/32 (zero point) = **4.625 bits a weight**, plus 16 bytes of
`weight_shape` a module. There is no global scale, no `weight_g_idx` (`actorder` null), no activation quantisation
(`input_activations: null`, so activations stay bf16: W4A16), and no KV-cache scheme (`kv_cache_scheme: null`).

## 4. The INT4 format, exactly

### 4.1 Storage

**Weights.** `pack_to_int32` (`ct` `helpers.py`) packs a signed int8 tensor q ∈ [−8, 7] (the range `calculate_range`
gives for a 4-bit INT type: `q_min = −2^3`, `q_max = 2^3 − 1`, `utils/helpers.py` lines 208–211) along the last
dimension (K, the input dimension), after adding an offset:

```
offset = 1 << (num_bits - 1)                                   # helpers.py 54: 8
value  = value.to(torch.int32) + offset                         # helpers.py 55: stored nibble = q + 8, in 0..15
bit_starts = elem_i * num_bits                                  # helpers.py 75: element i of each 32 at bit 4·i of the 4 words
word_idx   = (bit_starts // 32).long()                          # helpers.py 76
bit_offset = bit_starts % 32                                    # helpers.py 77
```

and `unpack_from_int32` undoes it, ending with `return (output - offset).to(torch.int8)` (helpers.py 179–180). For
4 bits no element crosses a word boundary, so:

```
nib(n, k) = (weight_packed[n, k // 8] >> (4 · (k mod 8))) & 0xF      # lowest nibble = lowest k
q(n, k)   = nib(n, k) − 8                                            # signed, −8..7
```

The int32 words are read as unsigned bit patterns (the top nibble of a word sets the sign bit of the I32). The
layout is the same in compressed-tensors 0.15.0, 0.16.0 and 0.17.1, whose `pack_to_int32` adds the same offset
(`(value + offset).to(torch.uint8)`) and packs groups of `pack_factor = 8` consecutive values with
`bit_shifts = arange(8) · 4` (0.17.1 `helpers.py` 62–83), and in 0.19.0. The checkpoint was written by a
development build (`0.1.dev537+g1cee215`) whose source we did not find; the data confirm the layout (*checked*, layer
4, expert 4 = GLM-5.3 expert 6, 16 rows): dequantised with low-nibble-first, gate_proj correlates with GLM-5.3's BF16
weight at 0.996 (after dividing out AWQ's column factor, §5) and down_proj at 0.962 (its AWQ factor, from
`up_proj`, left in); with high-nibble-first, at −0.003 and −0.006. Every nibble value 0–15 occurs.

**Zero points.** Stored only because the scheme is asymmetric with a group strategy (`ct` `base.py` 29–32, 51–52,
107–110):

```
if not weights.symmetric and weights.strategy in PACK_ZP_STRATS:
    packed_zp = pack_to_int32(zero_point, weights.num_bits, packed_dim=0)     # base.py 109
```

so the zero point z ∈ [−8, 7] (int8, `zp_dtype torch.int8`) gets the same +8 offset and is packed **along N, eight
output rows per int32**, lowest nibble first:

```
znib(n, g) = (weight_zero_point[n // 8, g] >> (4 · (n mod 8))) & 0xF   # g = k // 32, the group index
z(n, g)    = znib(n, g) − 8
```

In the data (*checked*, 16 rows of one expert's gate and down) `znib` spans 2–12 with mean 7.5. The zero points' row
order is confirmed the same way: with the eight rows of a word taken high-nibble-first instead, the correlations fall
from 0.996 / 0.962 to 0.851 / 0.770.

**Scales.** `weight_scale` is **BF16** [N, K/32], one per output row and group of 32 consecutive k, not packed
(`scale_dtype: null`, so the model's dtype). `weight_shape` is the unpacked [N, K] as I64.

### 4.2 Dequantisation

`PackedQuantizationCompressor.decompress` (`ct` `base.py` 117–163):

```
if not weights.symmetric and weights.strategy in PACK_ZP_STRATS:
    assert zero_point is not None, "Asymmetric quant requires zero-point values"   # 148
    original_zp_shape = (*original_shape[:-1], scale.shape[-1])                      # 149
    zero_point = unpack_from_int32(zero_point, weights.num_bits, original_zp_shape, packed_dim=0)
unpacked = unpack_from_int32(packed, weights.num_bits, original_shape)               # 155
state_dict["weight"] = dequantize(x_q=unpacked, scale=scale, zero_point=zero_point, g_idx=g_idx)   # 156–161
```

`dequantize` (`forward.py` 77–145) infers the group strategy from the scale's shape (lines 112–116: scale rows equal
the weight's rows, so `group_size = K / scale.shape[1]` = 32), sets **`dtype = scale.dtype`** when none is given (lines
132–133: bf16 here), and reaches `_process_group` (`forward_helpers.py` 118–177), which views the weight as
[N, K/32, 32] and calls `_dequantize` with the scale and zero point unsqueezed to [N, K/32, 1]:

```
def _dequantize(x_q, scale, zero_point=None, dtype=None, global_scale=None):          # forward_helpers.py 550
    if global_scale is not None: scale = scale / global_scale                         # 559–560: not here
    dequant_value = x_q.to(scale.dtype)                                               # 562: int8 → bf16, exact
    if zero_point is not None:
        dequant_value = dequant_value - zero_point.to(scale.dtype)                    # 564–565: bf16 − bf16
    dequant_value = dequant_value * scale                                             # 567: bf16 × bf16
    if dtype is not None: dequant_value = dequant_value.to(dtype)                     # 569–570: not passed
    return dequant_value
```

and `_process_group` ends with `output.flatten(start_dim=-2).to(output_dtype)` (line 171), a no-op since the output
dtype is the scale's. **So the weight that defines altar-1 is**

```
W(n, k) = bf16_rne( (nib(n,k) − znib(n, k//32)) · s(n, k//32) )          s = weight_scale, BF16
        = bf16_rne( (q − z) · s )
```

with exactly **one rounding**:
- `x_q.to(bf16)` and `zero_point.to(bf16)` are exact (integers in [−8, 7]).
- The difference q − z is an integer in [−15, 15], exact in bf16. It is the same as the difference of the stored
  nibbles, since the two +8 offsets cancel.
- The product is computed by torch as an f32 multiply rounded to bf16, round-to-nearest-even. The f32 product of a
  4-bit integer and an 8-bit significand is exact, so the only rounding is the final bf16 one. Checked exhaustively
  on 2026-10-14 with torch 2.14.1+cpu: for every finite bf16 scale (65,280 bit patterns) and every q, z ∈ [−8, 7],
  `(q.to(bf16) − z.to(bf16)) * s` equals the round-to-nearest-even bf16 of the exact product in all 16,711,680 cases.
- The rounding is real: 7,133,948 of those products (43 %) are not representable in bf16, since (q − z) needs up to
  4 significant bits and s has 8. A dequantiser that computes `q·s − z·s`, or `fma(q, s, −z·s)`, or works in f16,
  can differ in the last bit.
- The zero points are signed int8 values stored with the same +8 offset as the weights. Neither weights nor zero
  points are interleaved: AutoAWQ's `[0, 2, 4, 6, 1, 3, 5, 7]` nibble order is absent despite the "AWQ" in the
  name. Only the calibration was AWQ; the storage is compressed-tensors'.

The expression is the same for the INT4 attention projections (dense `Linear`s) and the routed experts; the
difference between them is only in how transformers reaches it (§4.4).

### 4.3 The GEMM

transformers then runs the dequantised bf16 weight through its ordinary code: `F.linear(x_bf16, W_bf16)` for the
attention projections, and for the experts the `grouped_mm` path over the fused `gate_up_proj` [168, 4096, 6144]
and `down_proj` [168, 6144, 2048] (glm53 § 2.7). The INT4 format adds no activation quantisation and no other
rounding. The engine's numerics contract (glm53 DESIGN decisions 3 and 8: bf16 activations, f32 accumulation, its
own combine order) applies unchanged with W as above.

### 4.4 What transformers does when it loads this checkpoint

`CompressedTensorsHfQuantizer` (`tf` `quantizer_compressed_tensors.py`) has two routes, and altar-1 uses both:

1. **The INT4 `Linear`s (`q_b_proj`, `kv_b_proj`, `o_proj` of layers 4–76).** Before loading, the model is wrapped
   by `apply_quantization_config(model, remaining_config, run_compressed=False)` and
   `self.compressor.compress_model(model=model)` (`quantizer_compressed_tensors.py` lines 109–118) so that each targeted `Linear` holds
   `weight_packed`, `weight_scale`, `weight_zero_point` and `weight_shape`. The default is `dequantize=False`
   (`utils/quantization_config.py` line 1138): the weights stay packed, and `compress_model` registers a forward
   pre-hook that runs `decompress_model` on the first forward pass (`ct` `model_compressor.py` 178–181, 246–260).
   With `dequantize=True` they are decompressed at load (lines 127–128). Either way each module gets
   `weight = dequantize(...)` as in §4.2, bf16, its status becomes `DECOMPRESSED`, and the quantised forward wrapper
   then skips weight fake-quantisation (`forward.py` 278: `status < QuantizationStatus.COMPRESSED` is false, since
   `DECOMPRESSED` comes after `COMPRESSED` in `LIFECYCLE_ORDER`). So the `Linear` computes `F.linear` with W.
2. **The routed experts.** transformers 5 stores a layer's experts fused (`GlmMoeDsaExperts.gate_up_proj`
   [E, 2·2048, 6144], `down_proj` [E, 6144, 2048]); `glm_moe_dsa` uses the `qwen2_moe` conversion
   (`conversion_mapping.py` line 55, 1019–1032), which merges `mlp.experts.*.{gate,up,down}_proj.weight`. For a
   compressed checkpoint `update_weight_conversions` (lines 182–222) rewrites the sources and prepends
   `DecompressExperts`:

   ```
   packed_weight = [p + "_packed$" for p in weight_sources]                     # 210
   shape_sources = [p + "_shape$" for p in weight_sources]                      # 211
   new_sources = packed_weight + scale_sources + shape_sources + other          # 212
   ```

   and `DecompressExperts.convert` (`integrations/compressed_tensors.py` 62–165) builds, per expert, a module with
   only `weight_packed`, `weight_scale` and `weight_shape` (lines 108–113) and calls
   `compressor.decompress_module(module)` (line 146). **The expert zero points are neither among the sources nor on
   the module.** For this asymmetric scheme `decompress` then stops at the assertion
   `"Asymmetric quant requires zero-point values"` (`ct` `base.py` 148; the same assertion is in 0.15.0, 0.16.0,
   0.17.1 and 0.19.0). The code is unchanged in transformers 5.16.0, 5.17.0, 5.18.0, 5.19.0 and main at `34f41636`.

   **Run on 2026-10-14** (CPU, torch 2.14.1+cpu, transformers 5.16.1, compressed-tensors 0.18.0): a two-layer
   `glm_moe_dsa` model with 128-wide hidden state, 4 experts and altar-1's `quantization_config`, written in
   altar-1's tensor layout with `pack_to_int32`:
   - **INT4 experts.** `from_pretrained` fails with `AssertionError: Asymmetric quant requires zero-point values`
     inside `DecompressExperts`, reported as `RuntimeError: We encountered some issues during automatic conversion of
     the weights`.
   - **BF16 experts in a compressed checkpoint.** With only the attention INT4 and the MoE layer's experts left BF16
     (as altar-1's layers 3 and 77 are), loading succeeds but the report lists every
     `mlp.experts.N.{gate,up,down}_proj.weight` as UNEXPECTED and the fused `gate_up_proj` and `down_proj` as
     MISSING, so **they are left at their random initialisation**. The rewritten converter looks for
     `weight_packed` in every expert converter, whether or not those experts are quantised.
   - **INT4 `Linear`s.** In the same run, after the first forward pass, `q_b_proj`, `kv_b_proj` and `o_proj` hold
     bf16 weights **bit-equal to `bf16_rne((q − z)·s)`** computed independently (f32 product, one rounding), which
     confirms §4.2 end to end for route 1.

   So transformers 5.16.1 cannot load altar-1. With a fix that passes the zero points it would still need the BF16
   experts of layers 3 and 77 to be loaded, which it currently drops without an error.

So the exact definition comes from compressed-tensors' `decompress` (§4.2), which transformers would apply to the
experts too if it passed the zero points. The engine's reference implements §4.2 directly (§9).

### 4.5 What vLLM computes

From `vllm` (main `bb1438a9`; v0.31.0 the same in the CUDA sources; read, not run; paths relative to the repository):

- **Which kernel.** The MoE method is `CompressedTensorsWNA16MoEMethod`
  (`compressed_tensors_moe/compressed_tensors_moe.py` 76–129); asymmetric gives `scalar_types.uint4` with zero points
  and the key `kInt4Static32Asym` (`compressed_tensors_moe_wna16.py` 75–99). The backend oracle
  (`fused_moe/oracle/int_wna16.py` 126–160) picks **Marlin** on sm_120/sm_121 (GB10). On Hopper, main now ranks
  "Humming" first (`quantization/utils/humming/priority.py`), while v0.31.0 uses Marlin. The INT4 `Linear`s use
  `CompressedTensorsWNA16` (`schemes/compressed_tensors_wNa16.py` 105–149) with **`MarlinLinearKernel`**. Machete
  is excluded by group 32 (`machete_utils.py` 31–32) and CUTLASS W4A8 by the zero points.
- **Repacking.** The nibbles are read low-first as raw `uint4` (`csrc/quantization/marlin/gptq_marlin_repack.cu`
  125–134), with no −8 applied. The zero points are unpacked as standard (non-interleaved) column packing
  (`moe_packed_to_marlin_zero_points`, `marlin_utils.py` 650–671) and stay integers. The scales stay bf16, and are
  permuted.
- **Dequantisation in the kernel** (`csrc/quantization/marlin/dequant.h` 173–205, `marlin_template.h` 104–139,
  320–324, 990–1074):
  - `lop3` with `0x43004300` turns each nibble into the bf16 value 128 + nib, exactly.
  - The zero point gets the same treatment, and `__hsub2` gives nib − znib = q − z, exactly.
  - `__hmul2(frag_b, s)` multiplies by the bf16 scale with **one round-to-nearest to bf16**. The `-z·s` FMA branch
    is skipped for integer zero points (`dequant_skip_flop`).
  - **So vLLM's weight equals W of §4.2 bit for bit** (the checkpoint's scales are already bf16, so vLLM's cast of them to `params_dtype`
    changes nothing; an f32 scale would be rounded to bf16 first).
- **Where the outputs part** from a "dequantise, then bf16 GEMM with f32 accumulation" reference, and from each
  other between batch sizes:
  - The `mma.sync … f32.bf16.bf16.f32` order (f32 accumulation) differs from a CPU or cuBLAS GEMM's.
  - The in-block and cross-block K reductions are in f32 (`use_fp32_reduce=True` for both dense and MoE), but their
    tiling depends on the row count M.
  - The single bf16 rounding of the output is the same as the reference's.
  - In the MoE, the router weight is rounded to bf16 and applied to the already-rounded bf16 expert output
    (`__hmul2`, MoE `marlin_template.h` 489–497 and 1670–1680), a second rounding. The 8 slots are then summed in f32
    in slot order and rounded once (`moe_sum`, `csrc/moe/moe_align_sum_kernels.cu` 420–460).
  - transformers' default path keeps the router weight in f32 and multiplies the bf16 expert output in f32
    (glm53 § 2.7).
  - **So vLLM's logits are not transformers' bit for bit**, by summation order and the router-weight rounding,
    though the weights are the same.
- `kv_b_proj` for MLA is turned back into dense bf16 at load by running the Marlin layer on an identity matrix
  (`layers/attention/mla_attention.py` 407–422, `quant_utils.py` 511–571). Each output there is one exact product
  plus zeros, so W_UK and W_UV are exactly W.
- With `VLLM_BATCH_INVARIANT=1` (main only) the INT4 `Linear`s dequantise to `(w − zp)·scale` in the scale dtype
  and use a batch-invariant matmul (`compressed_tensors_wNa16.py` 268–288).

## 5. The BF16 tensors are cyankiwi's, not GLM-5.3's

*Checked* on 2026-10-14 by range reads (details in [scouting § 3–4](scouting.md#3-the-int4-base-cyankiwiglm-53-awq-int4)):

- **Everything in altar-1 is cyankiwi's**, sliced to the kept experts. Checked on layers 4 and 40 for experts 0,
  100 and 167: `weight_packed`, `weight_scale` and `weight_zero_point` of the kept experts, `o_proj`, the norms and
  the router are byte-identical to `cyankiwi/GLM-5.3-AWQ-INT4` at the plan's original ids.
- **cyankiwi's BF16 values are `bf16_rne(E4M3 × block scale)` of the FP8 release** (`zai-org/GLM-5.3`
  `c5087a76`), not the BF16 master. 100 % of the sampled values of `q_a_proj`, `kv_a_proj_with_mqa` and the shared
  `down_proj` (layer 4, 128 rows each) and of the BF16 experts of layers 3, 77 and 78 match.
- **AWQ's smoothing is folded in** on the INT4 layers: `post_attention_layernorm`, `q_a_layernorm` and
  `kv_a_layernorm` are divided by per-channel factors, and the consumers' input columns are multiplied by them.
  - The consumers are the router and the experts' and shared expert's `gate_proj`/`up_proj` for the first
    (checked on the router and on one expert's `gate_proj`, which correlates 0.996 with GLM-5.3's only after the
    factor is divided out), and `q_b_proj` and `kv_b_proj` for the other two (*inferred* from the norms).
  - The router columns match GLM-5.3's times the factor to within bf16 rounding (median 0.13 %).
  - `input_layernorm`, `model.norm`, all norms of layers 0–3, 77 and 78, and the MTP layer's `eh_proj`, `enorm`,
    `hnorm` and `shared_head.norm` are the release's own.

Consequences:
1. **altar-1 is its own function.** It differs from "GLM-5.3 with 88 experts deleted" in every INT4 layer, by the
   quantisation and by bf16 rounding of the rescaled norms and router columns. Golden data must come from altar-1's
   tensors; none of glm53's golden data or rank files carry over.
2. **Layers 0–3 are the FP8 release's values**, dequantised to bf16, with nothing rescaled. Up to layer 3's MoE,
   altar-1 computes what a bf16 run of the FP8 release computes; layer 3's MoE differs only by the missing experts.
   From layer 4 on, the quantisation and the smoothing change every layer, and layer 77 (BF16 again) sees a
   different residual stream.
3. The norm weights of layers 4–76 span a wider range than GLM-5.3's (factors 0.27–6.1 in layer 4). This does not
   change any formula (RMSNorm still rounds `x·rsqrt(...)` to bf16 then multiplies by the weight, glm53 § 2.2).

## 6. Routing over 168 experts

The router is glm53 § 2.7's with E = 168 (`tf` `modeling` 499–537, `GlmMoeDsaTopkRouter`; line numbers below are 5.16.1's):

```
logits = fp32(x) · fp32(gate.weight)ᵀ                     # gate.weight BF16 [168, 6144]                 (514)
s      = sigmoid(logits)                                   # fp32 [168]                                    (515)
sel    = s + e_score_correction_bias                       # F32 [168]                                     (516)
         # n_group = topk_group = 1: view(-1, 1, 168); the group step keeps all 168                        (517–530)
idx    = top-8 of sel                                       # sorted=False                                  (531)
w      = s[idx] / (Σ s[idx] + 1e-20) · 2.5                  # norm_topk_prob, routed_scaling_factor 2.5     (532–536)
```

- **What pruning did** (*checked*, layers 3, 4, 40, 77, 78): `gate.weight` rows and `e_score_correction_bias` entries
  are the source's at the kept ids, in the kept order; the removed experts' rows and entries are gone; nothing is
  re-centred or rescaled. This is REAP's `prune.py` slicing (scouting § 4). Since the scores are independent
  sigmoids (not a softmax over experts), removing experts changes no kept expert's score `s` or selection key `sel`.
  It changes which eight are selected (the removed ones can no longer win) and so the normalisation denominator.
- The bias is ~28 in layer 4 (28.10–28.65 over the kept experts) and ~34 in layer 3 (glm53 § 2.7), so the
  selection sum must stay fp32, as before.
- The expert computation, the f32 sum over the 8 slots, the shared expert and the combine are glm53 § 2.7's. The
  engine's contract (glm53 DESIGN decision 8: f32 across experts and ranks, one rounding) is unchanged.
- **Tie-breaking.** With 168 candidates the router's eighth-place margin is a property of altar-1 to be measured
  again (glm53 found a quarter of its cases under 1e-3, DESIGN decision 8).

## 7. Layers 3 and 77, and the MTP layer

- **Layers 3 and 77**: MoE layers whose 168 experts are BF16 (`experts.j.{gate,up,down}_proj.weight`), with all
  their attention BF16 (`q_b_proj`, `kv_b_proj`, `o_proj` included). The math is glm53 § 2.7 with plain bf16
  weights. They cost 11.8 GiB each.
- **The MTP layer (78)**: as glm53 § 3, with **168 BF16 experts** chosen by expert weight norm, not by routing
  observations (`reap_prune_plan.json` `mtp_ranking: "weight magnitude from …/mtp_expert_norms.json"`). Its router
  [168, 6144] and bias [168] are sliced like the others (*checked*: equal to the FP8 release's rows at the kept
  ids). `eh_proj` [6144, 12288], `enorm`, `hnorm`, `shared_head.norm` and its whole attention and indexer are BF16
  and equal to the release's own (*checked*). transformers still ignores `model.layers.78.*`
  (`_keys_to_ignore_on_load_unexpected`, `modeling` line 666); the MTP forward is vLLM's (glm53 § 3, § 8). Its
  draft acceptance against the 168-expert main model is unmeasured anywhere.

## 8. Tokenizer, chat template, generation config

### 8.1 The chat template

altar-1's `chat_template.jinja` (sha256 `69bb3ab5…`) is **byte-identical to `zai-org/GLM-5.3` at its first two
revisions, `7cda8193` (2026-08-27) and `30333038` (2026-08-28)**, and to cyankiwi's. It predates zai's fix of
2026-09-04 (`aca966e4`, "chat template: early break in tool result reordering check"), whose template (sha256
`3740abce…`) is current at `c5087a76` and is the one NVIDIA's NVFP4 checkpoint and the BF16 master ship, and so the one
glmfront ports (glm53 § 7.2, [protocol.md](../../glm53-4xgx10-switchless/docs/protocol.md)). The diff, altar-1
against current:

```
@@ visible_text(content) @@
-    {%- else -%}
+    {%- elif content is not none -%}
         {{- content }}
@@ tool-result reordering check @@
         {%- for k in range(block_start, ns_blk.end + 1) -%}
+            {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
                 {%- for entry in m.content -%}
+                    {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
         {%- for i in range(ns_a.tool_calls | length) -%}
+            {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
                 {%- if id_of(ns_a.tool_calls[j]) == tc_id -%}
                     {%- set ns_chk.can_sort = false -%}
+                    {%- break -%}
```

What the differences mean for a byte-exact port:

1. **`null` content renders as `None`.** In altar-1's `visible_text`, a content that is neither a string nor a
   non-mapping iterable falls to `{{ content }}`, and Jinja renders Python's `None` as the text `None`. So a message
   with `"content": null` renders `None` where the current template renders nothing. The common case is an
   assistant turn that only calls tools, as OpenAI-style clients send it. A missing `content` key is unaffected,
   since Jinja's `Undefined` is iterable and renders nothing. Numbers and other non-iterables render as `str()` under
   both templates.

   Rendered with transformers 5.16.1's `apply_chat_template` on 2026-10-14:

   | Message | altar-1's template | zai's current template |
   |---|---|---|
   | assistant `{"content": null, "tool_calls": [...]}` | `<think></think>None<tool_call>f…` | `<think></think><tool_call>f…` |
   | user `null` | `<\|user\|>None` | `<\|user\|>` |
   | tool result `null` | `<tool_response>None</tool_response>` | `<tool_response></tool_response>` |
   | duplicate ids; reordered results | identical | identical |

   glmfront's rule "null nothing" ([protocol.md](../../glm53-4xgx10-switchless/docs/protocol.md), branch 3) becomes
   "null → `None`" for altar-1, in every role that calls `visible_text`: system, user, assistant content, tool
   results and their `output` entries.
2. **The `break`s change nothing in the output.** Each one only fires once `ns_chk.can_sort` is false, and no branch
   sets it back to true, so the loops' outcome (`can_sort`) is the same. The current template merely stops early.
   Loop controls need the `loopcontrols` extension, which transformers enables; altar-1's template does not use them.

So the port is glmfront's encoder with one rule changed. Whether to serve altar-1 with its own template (exact to the
checkpoint) or zai's fixed one (exact to GLM-5.3's behaviour on null content) is a front-end decision. What the
vendor's own serving passed for a `null` content (vLLM may normalise it before rendering) was not checked.

### 8.2 generation_config.json

```
{"_from_model_config": true, "do_sample": true, "eos_token_id": [154820, 154827, 154829],
 "pad_token_id": 154820, "temperature": 1.0, "top_p": 0.95, "transformers_version": "5.16.0.dev0"}
```

Identical to cyankiwi's. Against `zai-org/GLM-5.3` (`c5087a76`, and the same at `7cda8193`): the same eos, pad,
temperature 1.0 and top-p 0.95, plus **`do_sample: true`**. NVIDIA's NVFP4 copy (the one glm53's scouting § 1 describes as holding only the eos ids) has no
sampling fields. No change to the stop ids.

## 9. The reference implementation

- **The definition:** transformers 5.16.1's `modeling_glm_moe_dsa.py` (identical math to glm53's 5.15 reference, §1)
  with every INT4 weight replaced by W of §4.2, and the MTP layer per glm53 § 3. No released transformers runs the
  checkpoint's experts as shipped (§4.4). The reference therefore does what glm53's `glmref.py` did for NVFP4:
  read the tensors itself and dequantise with the §4.2 expression, in torch on the CPU
  (`(q.to(bf16) − z.to(bf16)) * s` with q, z from the nibbles; or in numpy as f32 products rounded to nearest-even
  bf16). Its output is checked against `compressed_tensors`' own `PackedQuantizationCompressor.decompress` on real
  tensors (a module-level call, which works on the CPU and needs no transformers model).
- **Packages:** torch (CPU wheel), `safetensors`, `compressed-tensors` 0.18.0 (for the cross-check; it pulls
  `transformers ≥ 4.45`, `pydantic`, `loguru`, `psutil`, and needs torch ≥ 2.10), `transformers` 5.16.1 for the
  one-layer check against `GlmMoeDsaDecoderLayer` (as glm53's `check_layer.py`, with dequantised weights loaded into
  the fused expert tensors directly).
- **CPU decompression is supported**, and ran on 2026-10-14 (§4.4's tiny model: torch 2.14.1+cpu, no triton
  installed). Nothing in `pack_quantized` or `_dequantize` needs CUDA. compressed-tensors imports triton through a
  shim (`utils/triton`) and uses it only for quantisation (`_quantize_triton`), not for `_dequantize`.
- **Memory per layer** (*inferred* from §3):

  | Layer | As stored | Dequantised to bf16 |
  |---|---:|---:|
  | INT4 layer (4–76) | 3.41 GiB of experts + 0.08 GiB of INT4 attention + ~0.12 GiB of BF16 tensors | 11.81 GiB of experts + 0.28 GiB of attention projections + the same ~0.12 GiB |
  | Layers 3 and 77 | 11.81 GiB of experts + ~0.4 GiB of attention and shared expert, all BF16 | the same |

  Dequantising the whole model would be ~0.9 TiB, so a layer-streaming reference, as glm53's, is the only CPU
  option on a 121 GiB box. One layer dequantised costs ~12.2 GiB, more than glm53's 3 GB per-pass ceiling on a box
  that also runs a ring job (glm53 serving § 5.1). Two ways to stay small:
  - Dequantise only the experts a sequence routes to, one expert at a time (each is 72 MiB in bf16 against 21 MiB
    packed).
  - Run the expert GEMM over row blocks of the dequantised weight.

  Either way W is the same bits.

## 10. Surprises, in one list

- **No released transformers loads the routed experts as shipped.** `DecompressExperts` drops `weight_zero_point`
  (5.16.0–5.19.0 and main of 2026-10-14), so compressed-tensors' asymmetric decompression asserts. In the same
  loader **the BF16 experts of layers 3 and 77 are silently left at random initialisation**, because the converter
  expects `weight_packed` for every expert. Both were reproduced on a tiny model on 2026-10-14.
- **The BF16 weights are not GLM-5.3's master values.** They are cyankiwi's: dequantised from the FP8 release,
  with AWQ's per-channel smoothing folded into three norms of every INT4 layer and into the router's and the
  experts' input columns.
- **The card's "only the routed experts are 4-bit" is wrong**: `q_b_proj`, `kv_b_proj` and `o_proj` of layers
  4–76 are INT4 too.
- **The INT4 weight is `bf16_rne((q − z)·s)`, one rounding**, with q and z each stored as nibble + 8. The zero
  points are packed along N, eight rows to a word. vLLM's Marlin dequantises to the same bits; its MoE output does
  not match transformers' because it rounds the router weight and the weighted product to bf16.
- **The MTP layer's experts were chosen by weight norm**, not by any routing statistic.
- **altar-1's template is zai's pre-2026-09-04 one**, and renders `"content": null` as `None`.
- `generation_config.json` sets `do_sample: true`, which GLM-5.3's does not.
- The `quantization_config.ignore` list still names experts 168–255 of layers 3, 77 and 78, which do not exist
  (harmless: an ignore entry that matches nothing).
- REAP's name notwithstanding, the kept experts were chosen by routed mass (the "massmax" criterion), not by REAP's
  saliency (scouting § 4).
