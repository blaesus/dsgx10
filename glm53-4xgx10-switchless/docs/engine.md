# The layer runner and the TP=4 step (engine/model)

G4: GLM-5.3 end to end on the four ranks (two ASUS Ascent GX10, two DGX Spark): the embedding, 78 layers, the
final norm, the vocabulary-split head, the sampler and the MTP layer, over the kernels of `engine/attn`,
`engine/experts` and `engine/comm`. Everything here was run on the ring on 2026-10-13/14, borrowed from the
DeepSeek service through its `ring.sh` (jobs of 15 s to 2 min).

**Status.** Runs on four ranks on synthetic weights and on the real non-expert rank files (G2) with synthetic
routed experts, layer 3 also with its real NVFP4 experts. Decode step at 1 row **46.4 ms** (21.6 tok/s) against
a 34.2 ms bandwidth floor (1.35×); prefill **400–434 tok/s** (GEMV-bound, see "What remains"). Against the
reference's golden data the residual stream after layers 0–3 agrees to cosine 0.9998 (mean), relative error
1.2–2.1 %, the size the FP8 requantisation predicts (each FP8 tensor 2.6 % relative error). Every
self-consistency check passes (below).

## Code

| File | What |
|---|---|
| `model.h` | the API: `Model` (bind a rank file or synthetic weights, `launch` a step of 1–16 rows, `wait`, `prefill`, `profile`), `Batch` (rows with token, position, slot, session; per-session `Sampling`), `SampleOut` |
| `config.h` | the model's constants beyond attn.h/experts.h: the MTP layer 78, 79 caches, dense layers 0–2, vocabulary 154,856 used of 154,880, stop ids 154820/154827/154829 (protocol.md v2) |
| `model.cu` | the decode step: one layer, the forward, one CUDA graph per row count, launch and wait, profiling, cache hashes |
| `mtp.cu` | the MTP layer's draft passes (decode) and its eh_proj |
| `prefill.cu` | prefill chunks of up to 8,192 rows, the MTP cache fill |
| `ops.cu`, `sample.cu` | the step-input ring, the vocabulary-split embedding, the top-1 for drafts, publishing; the DeepSeek sampler ported (38,720 ids a rank, ids ≥ 154,856 masked) |
| `l2pf.cu` | the paced L2 prefetch (DeepSeek's l2pf): the next weights during collectives and attention |
| `bind.cu` | binding a `.glw` rank file (names, sizes and layouts checked), synthetic weights, synthetic routed experts for a file whose pools are not filled |
| `tools/glm_run.cu` | the driver: `bench`, `l2pf`, `prefill`, `check`, `tf`, `refhash` |
| `tools/compare_golden.py` | the `tf` dumps against the golden files (reference.md § 7) |
| `install.sh`, `run4.sh`, `run.sh` | out-of-tree build (-j4) with the sources' hash; the ring launcher (comm's pattern, binaries under `~/dsgb10-data/glm53-model/`); the resumable ring steps |

## The decode step

A layer, R ≤ 16 rows (each rank holds 16 of 64 heads, a quarter of every expert, a quarter of the vocabulary):

1. x-projection GEMV (q_a, kv_a, k_pe and on full-indexer layers the indexer's k: 688 rows a rank, 656 on
   shared layers) and the indexer head weights → **all-gather** (`allgather_xproj`);
2. `attn_prep`: q_a norm, kv norm, RoPE, the latent cache write (and the index key on full layers);
3. q_b (+ the indexer's wq_b on full layers) → **all-gather** of the indexer queries → `index_select`
   (full layers; shared layers reuse the top-2048 of the last full layer);
4. absorbed W_UK → `sparse_attn` (split-K 16 for 1–2 rows, 8 above) → absorbed W_UV → o_proj (f32 partials);
5. **all-reduce + residual + ffn_norm** (`allreduce_add_norm`, the partials rounded once to bf16 after the sum);
6. dense layers: gate_up with the SwiGLU epilogue, down (f32) → **all-reduce + residual + next norm**; MoE
   layers: the router GEMV (BF16, f32 out), `route`, `moe_decode` (routed and shared experts in one kernel
   family, f32 partials for each) → **`allreduce2_add_norm`** (new, below): both partials in one collective,
   y = bf16(bf16(Σ routed) + bf16(Σ shared)), the reference's order of roundings.

So a step has 3 collectives a layer plus one on full layers: 255 for 78 layers, plus the embedding's row
all-gather. The head is a BF16 GEMV to f32 logits (the golden form); the sampler exchanges each rank's top
candidates (`topk_exchange`), never the logits. A step's inputs go through a ring of host slots read by the
first kernel (`k_step_in`), so the graph never changes; a token −1−r means "row r of the previous step's
sample", so the next step can be queued before the previous sample is known. Each step publishes its
`SampleOut` to mapped host memory with a sequence number; `wait` spins on it.

**MTP.** With `drafts` ≥ 1 each step ends with the draft passes: pass 1 at each row's position p with the token
after it (the next row's input if the same session continues, else the sampled token; its embedding zeroed at
position 0), x = eh_proj([enorm(e) | hnorm(h_p)]) (h_p after `model.norm`), layer 78 with its own cache and
indexer, the shared head norm, the head, the masked greedy top-1. Pass k > 1 chains on the MTP's own output
and the top-2048 of pass 1. Chained passes over multi-row sessions are refused (the pass would overwrite the
next row's MTP entry before its verify; G7).

## Prefill

Chunks of ≤ 8,192 rows (`Options::prefill_rows`): the embedding all-gathered in 1,024-row pieces; the attention
projections as 16-row GEMV launches; the indexer for rows at position ≥ 2,048 in batches sized to a 512 MiB
score scratch (rows below attend densely); `sparse_attn` without split; each sublayer ends with an f32
reduce-scatter by rows, the residual add and the next norm on the rank's quarter of the rows and one
all-gather of h and the normed rows; the MoE through `moe_prefill`. The MTP cache fill runs after each chunk
(eh_proj, layer 78's norm, x-projection, all-gather and `attn_prep` only — no MoE), the chunk's last row kept
pending until the next chunk or the first decode step brings the token after it. With a `Sampling`, the last
row is sampled as a one-row step (head, sampler, draft passes).

## Checks

All on the ring, four ranks (`run.sh check`, synthetic weights, 2,600-token prompt crossing 2,048; split-K 1 so
decode and prefill sum the attention alike):

| Check | Result |
|---|---|
| one prefill chunk = two chunks (1,300 + 1,300) | all 79 caches bit-identical |
| decode steps of 16 rows = prefill | caches of layers 0–3 bit-identical; from layer 4 on the routed sums differ in order (`moe_decode` vs `moe_prefill`), as designed |
| 16 rows in one step = 16 one-row steps | token and logit bits of all 16 rows, all 78 caches |
| graph replay = eager | all rows and caches |
| the four ranks | identical samples and cache hashes |
| temperature 0.8, top-p 0.9, 20 logprobs | ids in the used vocabulary, probabilities ≤ 1, identical on every rank |
| comm `coll_tests` with `allreduce2_add_norm` | exact against the host for 1–16 rows, with and without PDL, identical on all ranks |

Batch invariance holds only at a fixed split-K: the default (16 up to 2 rows, 8 above) makes a row's attention
sum depend on the step's row count. Decided by the serving layer, not here; the check pins it.

**Against the golden data** (`run.sh tf-prefill tf-decode`; the 291-token `short` sequence; the real rank files
with routed experts synthetic, then with layer 3's real NVFP4 experts filled into copies of all four files):

| After layer | cosine min / mean | relative error |
|---|---|---|
| 0 | 0.99953 / 0.99993 | 1.19 % |
| 1 | 0.99947 / 0.99980 | 2.00 % |
| 2 | 0.99947 / 0.99978 | 2.09 % |
| 3 (real experts) | 0.99895 / 0.99977 | 2.14 % |
| 3 (synthetic experts) | 0.95159 / 0.99606 | 8.56 % |

Prefill and decode give the same numbers to five digits. The reference attends with BF16 weights, this engine
with FP8 (2.6 % relative error a tensor in the converter's manifest), so this is an envelope, not equality;
logits and top-1 cannot be compared until the expert pools are filled. `refhash` (synthetic, 300-token prompt,
64 greedy tokens fed back one step ahead): the same hash on all four ranks.

The weight files' first GPU run (`run.sh weights`): every rank loads its file with the device copies matching
their CRCs (0.8–2.9 s); `glw_kcheck --gpu` on rank 0's layers 0, 3, 10, 78 and the head: 105,136 checks ok,
**1 bad**: one of 12,288 SwiGLU outputs of `layers.0.mlp.gate_up` outside kcheck's 2-ulp bound (the kernels'
own tests are bit-exact against torch; likely the bound near a cancelling gate, not yet looked into).

## Measurements

Decode, synthetic weights (16 routed-expert pools of all 256 NVFP4 experts, layer l on pool l mod 16), context
4,096, CUDA graphs with PDL and the L2 prefetch, the median of the last 4 of 32 steps (GPU time between events).
The floor is the step's bytes at 235 GB/s: every weight read once, the routed experts at the expected number of
distinct experts for R rows routed independently (8 at 1 row, 46 at 6).

| Rows | GPU ms | floor ms | × floor | tok/s | host launch |
|---|---|---|---|---|---|
| 1 | 46.4 | 34.2 | 1.35 | 21.6 | 6.8 µs |
| 2 | 60.8 | 47.4 | 1.28 | 33.2 | 6.6 µs |
| 4 | 81.4 | 72.4 | 1.13 | 49.1 | 6.5 µs |
| 6 | 97.2 | 95.9 | 1.01 | 61.7 | 6.8 µs |
| 8 | 110.8 | 118.0 | 0.94 | 72.4 | 6.6 µs |
| 16 | 146.8 | 193.4 | 0.76 | 111.0 | 6.7 µs |

Above 6 rows the synthetic router's picks overlap more than independent routing would, so the "floor" there
overstates the bytes. With one MTP draft pass: 50.4 / 64.6 / 85.5 ms at 1 / 2 / 4 rows (+3.6–4.0 ms).

Where the time goes (one eager step without PDL, events between kernel classes; the collectives include
waiting for the slowest rank):

| Class | 1 row | 6 rows | 16 rows |
|---|---|---|---|
| MLP / MoE (router, routed + shared experts) | 21.1 | 61.8 | 93.7 |
| o_proj | 9.0 | 9.1 | 9.2 |
| q_b + wq_b | 3.6 | 3.7 | 3.8 |
| x-projection | 2.2 | 2.4 | 2.6 |
| W_UK + W_UV | 2.4 | 2.5 | 2.6 |
| attn_prep + index_select + sparse_attn | 4.1 | 5.5 | 9.2 |
| head + sample | 2.6 | 2.2 | 2.5 |
| collectives (4 kinds) | 9.7 | 19.4 | 28.5 |
| total (eager) | 54.7 | 106.6 | 152.1 |

Per layer class at 1 row: dense 0.74 ms, MoE with full indexer 0.71, MoE with shared indexer 0.65. The graph
with PDL is 8 ms faster than eager at 1 row. The collectives' own cost at 1 row is 4.4 ms (ring.md); the rest
of the 12 ms gap to the floor is the MoE at ~83 % of its bytes' floor at one row and the launch tails.

**L2 prefetch, A/B and B/A** (`run.sh l2pf`): 1 row +0.1 %, 6 rows +4.1 %, 16 rows +2.8 %. Kept on.

**Prefill** (synthetic, from position 0): 414 / 434 / 419 / 403 / 400 tok/s for chunks of 512 / 1,024 / 2,048
/ 4,096 / 8,192 rows (1.24 s to 20.5 s a chunk), after the expert prefill tuning (6cb6fce); 397–416 before.

## What remains

- **Prefill GEMMs.** The attention projections, o_proj, the dense MLP, the router and the head run as 16-row
  GEMVs, so each weight is read once every 16 rows (~24 GB a layer a rank at 8,192 rows): prefill is bound by
  that, not by the experts. Tensor-core FP8 GEMMs for these (DeepSeek's prefill kernels as the model) are the
  next step; the target is the chunk's expert time.
- **The decode gap** (1.35× at 1 row against G4's 1.25×): fewer collectives (merge the x-projection
  all-gather into the previous all-reduce's epilogue, the indexer queries' all-gather into it on full layers),
  the embedding all-gather onto the sampler's exchange, and the MoE's one-row efficiency.
- **Real routed experts**: the rank files' pools are reserved (the fill needs the user's disk decision); then
  `tf` against the golden logits (top-1, top-20, KL) and the MTP's drafts against `mtp/` (G7's verify).
- `glw_kcheck --gpu`'s one SwiGLU output outside its bound.
- Batch invariance under the default split-K (above).
