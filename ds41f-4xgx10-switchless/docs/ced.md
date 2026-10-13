# CED prefill: semantics, reference and measurements

DESIGN decision 5 says prefill uses CED (Causal Encoder-Decoder) with Decoder SWA Bounded Replay.
This page pins down what that means for this checkpoint, where the reference for it lives, how far
it moves the model from the exact forward, and what the engine has to do. Its measurements ran
dealignai's variant of the checkpoint, the one the engine serves ([README](../README.md)); the official
set has the same `config.json`, so the structure below holds for both.

## Sources

- Tech report §2.2 (CED) and §3.2.2 (SWA Bounded Replay), `ref/techreport.txt`.
- The publisher's code has none of it. In `model.py`, prefill is one `start_pos` 0 chunk through all
  40 layers. `generate.py` prefills once, then decodes one token at a time. `config.json` has no
  encoder or replay fields.
- The model was post-trained with the replay simulated (§3.2.2). So CED is a sanctioned way to run
  the model, but it is not bit-compatible with `model.py`.
- [`ref/torchref/ced_run.py`](../ref/torchref/ced_run.py) is the reference for it. It emulates CED
  around the unchanged `model.py`, next to the exact forward.

## Semantics

These use model.py's 0-indexed layers, with L/2 = 20 and n_win = W = 128.

- **Encoder.** Layers 0–19 run over every prompt position, exactly as now. This includes Engram
  (layers 1 and 14) and the encoder's CSA (ratio 2, kv sources 2/8/14), so CED leaves both alone.
- **Decoder global KV.** It comes from layer 20's compressor and index keys. The compressor is ratio
  1, `norm(wkv(x))`, then RoPE, then fp4. The index keys are `k_norm(wk(latent))`, then RoPE, then
  fp4.
  - Here x = `attn_norm(hc_pre(h, layer 19's FFN pre-mix))`, a function of the encoder output only.
    This is the report's "projected from H_{L/2}".
  - The report gives each decoder layer its own W^KV. In this checkpoint that collapses to layer 20's
    projection, because `kv_source_layers` = [2, 8, 14, 20]. Layers 21–39 read layer 20's cache, and
    the Reindex layers 24/28/32/36 re-score inside layer 20's candidate pool.
  - It is built for all P prompt positions, and it is bit-identical to the exact forward's. The
    reference checks this per run: the compressed KV, the index keys, the candidate pool and layer
    20's top-k all match.
- **Decoder replay.** Layers 20–39 run only over [s, P), where s = max(0, P − 128).
  - Every decoder layer's sliding window is truncated at s, so a query at i sees SWA keys in
    [max(s, i−127), i]. That includes layer 20, even though its keys before s could be projected from
    the encoder output.
  - Since the replay is 128 tokens, every replay query's window is the whole replay prefix [s, i]. The
    SWA part of the replay is therefore plain causal attention within the chunk.
  - Global attention is unchanged: each query takes its top 512 of the compressed positions [0, i]
    through layer 20's pool, so it reaches the whole prompt.
- **Output.** Logits are needed at P−1 only. Nothing in the decoder exists for positions before s.
- **Decode.** Each decoder ring now holds the replay's keys for [P−128, P), which is a full window. A
  decode step at t ≥ P sees [t−127, t], which lies inside [s, t], so decode is unchanged.
  - The approximation survives only in hidden states. Through the chain of windows, layer 39 at t
    still depends on the replay for t ≤ P − 2 + 19·127 ≈ P + 2,400.
  - The global KV is exact throughout.
- **Short prompts.** If P ≤ 128, then s = 0 and CED is the exact forward (for example, prose's
  39-token prompt).
- **DSpark.** `main_hidden` (the inputs of layers 37–39) exists only for the replay rows [s, P). Those
  are exactly the 128 positions DSpark's prefill seeds, since `DSparkAttention` keeps the last 128
  `main_kv`. The values are the replay's.
- **Engram.** It is encoder-only and per position, so it is unaffected.

## How close CED is to exact

The reference emulates CED in one layer-streaming pass. The decoder runs over the whole sequence,
but every query at or after s is denied the window keys before s. Positions ≥ s then compute
exactly what CED computes: their SWA is truncated, their global KV is the encoder's (identical in
both modes), and every other op is per token. Teacher-forced decode is the same prefill run past P,
as in `ref_run.py`.

Two checks back the emulation:

- **Isolation.** A copy of the CED stream has its decoder rows before s scrambled after layer 20. In
  prose's check, its layers 21–22 come out bit-identical. Beyond that, only the MoE GEMM shapes
  differ, so the copy differs at the numerical-noise level. That copy is the "iso" yardstick.
- **Plumbing.** The exact stream reproduces `golden/prose` bit for bit. It reproduces `golden/long`
  to 1e-4 in the logsumexp, which differs only by the head's chunking. A rerun on the same box is
  bit-identical.

There are two yardsticks for what counts as noise. "iso" is the CED stream with different GEMM
shapes. "split-K" is the exact forward with every decoder GEMM summed in two K halves
(`kernel.py`'s splitk). The golden README shows that perturbations like these already move
`model.py`'s own logits as far as SGLang does.

Data is in `rank 0's /data/dsgb10/golden/ced/`, with `manifest.json` there listing the commands and the
comparisons. The runs:

- `prose`.
- `long8k`: the first 8,936 tokens of `golden/long.tokens.json`.
- `long28k`: all of it.
- `long8k-noise` and `long28k-noise`: the split-K twins.
  - `long28k-noise` keeps only the twin. Its exact stream picked up the twin's layer-20 candidate
    pool through a bug in the script's first version, which is fixed. So compare the twin with
    `long28k/exact`.

Prompts end at `<｜Assistant｜><think>` of a turn. KL is KL(exact ‖ CED) in nats.

| run | P | replay | first token (P−1) | teacher-forced decode after P | split-K twin, same positions |
|---|---|---|---|---|---|
| prose | 39 | 128 (s = 0) | identical | identical (96 tokens) | – |
| long8k ced128 | 8,680 | 128 | same argmax, KL 0.0025 (iso 0.0002) | 256 tokens: top-1 248/255, mean \|Δlp\| 0.079 | 248/255, 0.091 (iso 245/255, 0.065) |
| long8k ced256 | 8,680 | 256 | same, KL 0.0021 | 248/255, 0.094 | same as above |
| long8k ced512 | 8,680 | 512 | same, KL 0.0004 | 250/255, 0.076 | same as above |
| long8k ced128at4728 | 4,728 | 128 | same, KL 0.012 (split-K 0.0008) | 4,207 tokens: top-1 95.0 %, 0.162, mean lp −1.600 vs −1.601 | 92.8 %, 0.229 |
| long28k ced128at12453 | 12,453 | 128 | same, KL 0.0065 (split-K 0.0014) | 16,035 tokens: top-1 95.9 %, 0.120, mean lp −3.309 vs −3.313 | 89.1 %, 0.326 |
| long28k ced128 | 28,489 | 128 | same, KL 0.0036 (iso 0.0007) | – (prompt only; replay rows: top-1 118/127, 0.145) | replay rows: 124/127, 0.057 |

- **The first sampled token.** It is unchanged in all six long cases. The KL is 0.0004–0.012 nats,
  3–15× a numerics twin's: measurable, but small.
- **Decode.** CED stays inside the numerics envelope. Top-1 agreement with exact is 95–98 %, with no
  bias in the mean log-prob, and it is closer to exact than the split-K twin over the same
  positions.
  - The difference does not die out after the ~2,400-token SWA reach. From there on it is the same
    chaotic bf16/fp8 drift that any perturbation starts.
- **The replay positions' own logits.** These are perturbed (top-1 57–93 %, KL 0.06–1.5), but CED
  never uses them.
  - The replay's SWA KV, which is what decode reads, is exact at layer 20 (projected from the encoder
    output) and off by 5–60 % relative per layer at layers 21–39 (`kvdiff`).
  - DSpark's `main_hidden` over the replay is off by ~0.5 relative on average and 0.06–0.12 at P−1
    (`main_hidden_diff`).
- **Longer replays.** R = 256 or 512 buys nothing measurable in decode, so 128 (what the model was
  trained with) is the right length.
- **The massive activation.** In `long28k`'s replay, positions 28,420 and 28,423 grow it at layers
  36–38 (relative difference ~300×) where the exact forward does not. This is the same fragile spot
  the golden README found split-K flipping (position 28,423).

## What the engine has to do

### Prefill of P prompt tokens (new session)

1. **Encoder.** Run the encoder chunks over [0, P) as now (layers 0–19).
2. **Layer 20's KV projection, for every row.**
   - `hc_pre` with layer 19's FFN pre-mix, then `attn_norm`.
   - The compressor: `wkv`, then its norm. RoPE at the row's position, then fp4 (16-groups, e4m3
     scales) into the decoder's global KV.
   - The index key: from the same latent *before* RoPE, `wk`, then `k_norm`, then RoPE, then fp4
     (32-groups, e8m0) into the index-key cache.
   - Nothing else of layer 20 is needed for rows before s: no q, no indexer scores, no window KV.
   - Keep the layer-20 input (h, 4 × 5120, and the pre-mix) of the last 128 rows; they can span two
     encoder chunks.
3. **Decoder replay.** Run layers 20–39 once over the 128-row chunk [s, P), starting from empty
   decoder rings.
   - The SWA is causal attention inside the chunk, which is exactly the truncation.
   - The global part is the decode-style one: layer 20's indexer over all P index keys, then the
     pool (2,048 blocks of 8) and top-512. Layers 24/28/32/36 re-index; the rest reuse.
   - The chunk's window KV becomes the rings' contents (positions [P−128, P)).
   - When P ≤ 128 this is the whole prompt, and the result is exact.
4. **Head.** Final `hc_pre`, then norm, then head, for row P−1 only.
5. **DSpark.** Take the replay rows' layer 37–39 inputs as `main_hidden` and `accept()` them, 8 rows a
   call (engine/dspark/README.md). Then decode as now; nothing in decode changes.

### Cost

- Encoder: P rows × 20 layers.
- Layer-20 projection: a 5120→512 and a 512→128 GEMM per row.
- Decoder: a fixed 128 rows × 20 layers.
- At 128 rows the decoder MoE still touches ~86 % of each layer's experts (128 rows × 6 picks over
  384 experts), so the replay costs about 20 expert sweeps (7.7 ms each at TP=4, DESIGN's table)
  whatever P is.
- CED therefore halves prefill once chunks are compute-bound. For short prompts it saves little.

### Turns in a live session (decision 6)

A turn appends n tokens at [P₀, P).

1. The encoder continues from its live state, exactly.
2. Layer 20 projects the n rows.
3. The decoder depends on n:
   - **n ≥ 128:** replay [P−128, P) as above. The live decoder rings are discarded.
   - **n < 128:** run the decoder over the n rows from the live rings, a multi-row decode step. This
     is exact continuation: it is cheaper than a 128-row replay and needs no encoder output from the
     previous turn.

The report replays at every prefill (§3.2.2), which for n < 128 would need the previous turn's
layer-20 inputs. Exact continuation is the cheaper side of the same trained behaviour.

### Prefix reuse across sessions (decision 6's fallback)

Encoder SWA Bounded Replay re-runs the 128 tokens before the cut through the encoder with the same
truncation. Those tokens regenerate only SWA KV and keep their cached global KV. The decoder side is
then the replay above. It is not built: the server restarts exactly from the live rings or a window
snapshot instead ([engine § Serving](engine.md#serving), "Exact reuse").

### Testing

- **Exact mode.** It stays the bit-level check against `golden/` (`prose`, `long`).
- **CED mode.** Compare against `long8k/ced128.*` (prefill 8,680 tokens, then 256 teacher-forced) and
  `long28k/ced128.*` (the whole transcript).
  - First-token logits: `full_logits`.
  - `next_logprob` and top-32.
  - `trace`: the residual after each decoder layer at s, s+1, s+2, s+63, P−1 and L−1.
  - `main_hidden` (the DSpark seed) and `final_hidden`.
- **Pass bar.** End-to-end comparisons are statistical, because iso and split-K show the noise. Pass
  if the engine in CED mode sits inside the split-K envelope against `ced128` in the replay, the
  first token and decode.
