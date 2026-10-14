# The layer runner and the TP=4 step (engine/model)

G4: GLM-5.3 end to end on the four ranks (two ASUS Ascent GX10, two DGX Spark): the embedding, 78 layers, the
final norm, the vocabulary-split head, the sampler and the MTP layer, over the kernels of `engine/attn`,
`engine/experts` and `engine/comm`. Everything here was run on the ring on 2026-10-13/14, borrowed from the
DeepSeek service through its `ring.sh` (jobs of 15 s to 2 min).

**Status.** Runs on four ranks on synthetic weights and on the real non-expert rank files (G2) with synthetic
routed experts, layer 3 also with its real NVFP4 experts. Decode step at 1 row **46.4 ms** (21.6 tok/s) against
a 34.2 ms bandwidth floor (1.35×); after the 2026-10-14 decode-gap pass 43.6 ms (1.27×) in the interleaved A/B, ~44 ms
on undisturbed ranks, the MoE at its floor and the rest the collectives' latency and attention ([Closing the
decode gap](#closing-the-decode-gap-2026-10-14)); prefill **400–434 tok/s** (GEMV-bound, see "What remains"). Against the
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
| `oproj.cu` | o_proj as the producer half of its all-reduce (`kTuneOproj`, off by default; [Closing the decode gap](#closing-the-decode-gap-2026-10-14)) |
| `tools/step_profile.py` | the per-kernel critical-path profile of `glm_run trace`'s nsys reports, every rank |
| `l2pf.cu` | the paced L2 prefetch (DeepSeek's l2pf): the next weights during collectives and attention |
| `bind.cu` | binding a `.glw` rank file (names, sizes and layouts checked), synthetic weights, synthetic routed experts for a file whose pools are not filled |
| `tools/glm_run.cu` | the driver: `bench`, `l2pf`, `ab` (two lever sets, interleaved, timing and output bits), `trace` (under nsys), `prefill`, `check`, `tf`, `refhash` |
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

## Closing the decode gap (2026-10-14)

A pass on the decode step's gap to its bandwidth floor with every output unchanged bit for bit. Five ring jobs of
4–9 minutes on the four ranks, synthetic weights at context 4,096 as above; the merged tree of 2ec6f1c (the
attention-side tuning and the server) for the final numbers.

**Method.** `glm_run trace` replays the step's graph under nsys on every rank (`run.sh trace`: CUDA trace with
graph nodes, `--capture-range=cudaProfilerApi`); `tools/step_profile.py` reads the four reports and gives, per
kernel kind, its *critical-path increment*: the end of the kernel minus the end of the one before it on the
stream, i.e. what it adds to the step with PDL overlapping the launches (the eager profile above, without PDL,
overstates every boundary). The levers are switches (`Options::tune`, `Model::set_tune`, `glm_run --tune`); `glm_run
ab` (`run.sh ab`) times two lever sets in one process, ten rounds of 12 graph steps in the order A B, then B A
(each lever set keeps its own graphs), and checks that both give the same bits: the final normed rows, the
sampled tokens and logit values, and every layer's cache, on every rank.

**Where the 1-row step goes** (ms a step, the sum of the increments over the layers; ranks 1 and 2: rank 1 is
undisturbed, rank 2 the slowest, below):

| Kernel | before, rank 1 | before, rank 2 | after, rank 1 | after, rank 2 |
|---|---:|---:|---:|---:|
| `k_up` (shared + routed gate/up) | 11.49 | 13.29 | 11.92 | 13.32 |
| o_proj | 7.30 | 8.46 | 8.48¹ | 9.37¹ |
| `k_down` | 4.77 | 5.59 | 4.98 | 5.55 |
| all-reduce after the MoE / MLP | 5.35 | 2.15 | 7.70² | 3.82² |
| all-reduce after o_proj | 4.83 | 1.90 | (above) | (above) |
| `k_attn` (attn/) | 3.00 | 3.43 | 2.28 | 2.58 |
| head | 2.09 | 2.50 | 2.11 | 2.42 |
| x-projection all-gather (+ indexer queries') | 2.10 | 1.81 | 2.14 | 1.73 |
| q_b (+ wq_b) | 2.05 | 2.50 | 2.04 | 2.30 |
| router GEMV | 0.67 | 1.30 | 0.69 | 1.06 |
| `k_route` | 0.48 | 0.49 | — | — |
| step | 48.0 | | 45.9 | |

"Before": the tree before this pass (the attention-side tuning not yet merged), "after": all levers on (¹ this trace
ran o_proj's posting variant, off by default: see below; ² both all-reduces as one kind). The two traces ran with
the same disturbance on rank 2 (below); one trace of the merged tree with the levers off, with rank 2 undisturbed for its
length, gave **44.4 ms** at 1 row and **91.5 ms** at 6.

What the profile says:
- **The MoE is at its floor inside the graph.** `k_up` + `k_down` on an undisturbed rank: 16.3 ms for 75 layers,
  217 µs a layer against 221 µs for its bytes at 235 GB/s (the shared expert's 9.4 MB and eight routed experts'
  42.5 MB; the kernels preload the shared expert before their PDL wait, so a little lands inside the all-reduce
  before). The eager profile's 21.1 ms was launch gaps without PDL, plus `route` (7 µs a layer in the graph:
  0.5 ms a step) and the router GEMV (13–17 µs a layer, at its floor).
- **o_proj, the head, q_b are at their floors** (o_proj 7.3 ms against 8.4 for 78 × 25.2 MB: its first chunks
  load before the PDL wait, during the all-reduce; the head 2.1 against 2.0).
- **The waiting is one slow rank, and the slowness is not ours.** Rank 2's memory-bound kernels ran 8–16 %
  slower than ranks 1 and 3 (`k_up` 13.3 against 11.5 ms; at 6 rows 43.2 against 37.3), so the other ranks wait
  for it in every all-reduce (5.3 + 4.8 ms on rank 1, 2.2 + 1.9 on rank 2). The cause: the reference's CPU run
  (experiment 2, `glmref.py`) on rank 2's box at ~600 % CPU, on cores 5–9 among others, i.e. on the cores pinned
  for the ring's proxies, and drawing on the same LPDDR5X as the GPU. While it paused between runs the step was
  44.4 ms at 1 row and 91.5 at 6 (the trace above) against 46–48 and 97–99 with it. Every A/B below ran with it
  (A/B rounds interleaved so that both sides see the same mix); the absolute step times move with it by ±2 ms.
- **What is left on the slowest rank** at 1 row is the collectives' own time (the two all-reduces ~24 µs each
  inside the step, 156 of them: 3.8 ms; the all-gathers 1.7 ms), the attention kernels (`k_attn`, the indexer,
  prep, combine: ~4 ms, attn/), the sampler and the launch boundaries.

**The levers** (all exact; leave-one-out: all on against all but one, at 1 / 6 rows, gain in %):

| Lever (`Tune` bit) | What | Leave-one-out 1 / 6 rows |
|---|---|---|
| `kTuneRoute` (1) | the routing inside `k_up` and `k_down` (`DecodeCfg::logits`; experts.md § 4): no `route()` launch | +0.1 / −0.0 |
| `kTunePost` (2) | `k_down` writes both partials into the ring's send buffer and posts them from its epilogue; the all-reduce is its consumer half (`allreduce2_add_norm_posted`) | +0.7 / +1.4 |
| `kTuneIw` (4) | full layers: the indexer's head weights in the x-projection's launch (attn's `GemvIw`) | +0.0 / +0.0 |
| `kTuneOproj` (8) | o_proj as the producer half of its all-reduce (`model/oproj.cu`: attn's FP8 GEMV arithmetic, the partials into the send buffer, posted) | −0.2 / +0.5 |
| `kTuneArRow` (16) | the fused all-reduces with one block of 768 threads a row: no barrier between blocks (`slices = 1`; ring.md) | +1.0 / +0.9 |

Leave-one-out spreads (min .. max of the ten rounds' ratios) are ±3–9 %, so single levers below ~1 % are not
resolved. The default is `kTuneDefault` = 1 | 2 | 4 | 16: the o_proj variant is off, since its kernel is slower than
attn's (its tail writes 24 KB a row to host-mapped memory and fences at system scope: 8.5 against 8.1 ms a
step) and buys back about that in the all-reduce.

**Step time, before and after** (GPU ms, medians of ten interleaved rounds, the same job; outputs the same bits
at every row count):

| Rows | none | default (17) | gain | all (1f) | gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.28 | 43.60 | 1.2 % | 45.59 (A 46.99) | 3.0 % |
| 2 | 58.96 | 57.96 | 0.6 % | 60.87 (A 61.85) | 2.0 % |
| 4 | 79.48 | 78.01 | 0.6 % | 80.65 (A 80.45) | 1.3 % |
| 6 | 96.48 | 94.08 | 0.2 % | 94.30 (A 94.54) | 0.4 % |
| 16 | 151.64 | 152.53 | −0.6 % | 146.37 (A 145.94) | −0.3 % |
| 1, one draft pass | 48.57 | 48.09 | 1.0 % | | |
| 2, one draft pass | 64.59 | 64.50 | 0.4 % | | |

(The gain column is the median of the rounds' B/A ratios, not the ratio of the medians; the "all" columns are a
second job, its own A in brackets.) Against the 34.2 ms floor: 1.27× at 1 row in the best interleaved
measurement, 1.30× with the gap measured from the earlier `bench` (47.7 ms before this pass on the same
disturbed ring). The levers are worth ~1–3 % at 1–4 rows and nothing at 16 (where the collectives are 98 µs
apiece and bound by their bytes, not their latency).

**The L2 prefetch with the levers on** (`run.sh l2pf`, A/B and B/A): +3.0 % at 1 row, +0.6 % at 6, +1.7 % at 16
(G4: +0.1 / +4.1 / +2.8 %). Kept as it is: on the slowest rank its windows are only the collectives' own ~20 µs,
and the next GEMVs already load their first chunks before their PDL wait inside the same window, so a schedule
change has at most that window to gain.

**Exactness.** `ab`: every row count, the default and the full set, with and without a draft pass: the same
bits as no levers on all four ranks. `tf` via decode with `--tune 0` and `--tune 1f`: all 20 dump files
(argmax, top-20, log-probabilities, lse, the logits of the sampled rows, the hidden rows after layers 0–4 and
77) byte-identical, on the real rank files (layers 0–3 real, routed experts synthetic) and on synthetic weights.
`run.sh check` passes with the defaults (chunking, decode = prefill, 16 rows = 16 one-row steps, graph = eager,
ranks agree). `coll_tests test` passes with the new `slices = 1` cases (1–16 rows, with and without PDL, both
add-norm variants, against the host reference and attn's rmsnorm). `test_experts` passes (route through
`route_warp`). The server's ring tests (`server/tests/ring_job.sh`: serve, batch, front, kept) pass on the merged
binary. No summation order changed: the all-reduce sums ranks 0–3 in order as before; the one-block norm does
rmsnorm's 256 runs of 24, its eight xor-trees and its warp-order sum; the posted partials are the same values.

**What failed or did not pay.** o_proj posting (above: neutral, off). Fusing the index weights into the
x-projection (attn's new `GemvIw`): nothing measurable, kept since it is one launch fewer on 21 layers. The
first A/B runs (one block of 32 steps a side, A B B A) could not resolve 1 % against rank 2's disturbance (the
same configuration measured 43.0–47.3 ms in one job): replaced by the interleaved rounds.

**What remains for the gap.** On undisturbed ranks the step is ~44 ms at 1 row (1.29×). The rest is the
collectives' own latency on the critical path (~5.5 ms at 1 row: 156 f32 all-reduces at ~24 µs and 99
all-gathers at ~17 µs) and the attention side (~4 ms, attn/). Exact ways to take more: fewer collectives (the
x-projection all-gather merged into the all-reduce before it needs the x-projection replicated or a
reduce-scatter design, ring.md), and the ring's per-collective latency itself (the host proxies' relay; not broken down
here). Operationally: keep CPU jobs off the ring's boxes (or off cores 5–8) while
measuring or serving; a busy box costs 2–4 ms at 1 row and ~6 ms at 6.

## What remains

- **Prefill GEMMs.** The attention projections, o_proj, the dense MLP, the router and the head run as 16-row
  GEMVs, so each weight is read once every 16 rows (~24 GB a layer a rank at 8,192 rows): prefill is bound by
  that, not by the experts. Tensor-core FP8 GEMMs for these (DeepSeek's prefill kernels as the model) are the
  next step; the target is the chunk's expert time.
- **The decode gap** (1.27–1.30× at 1 row after the 2026-10-14 pass, ~1.29× on undisturbed ranks; G4's target
  1.25×): fewer collectives (merge the x-projection all-gather into the previous all-reduce's epilogue, the
  indexer queries' all-gather into it on full layers), the embedding all-gather onto the sampler's exchange, the
  ring's per-collective latency; the MoE is at its floor ([Closing the decode gap](#closing-the-decode-gap-2026-10-14)).
- **Real routed experts**: the rank files' pools are reserved (the fill needs the user's disk decision); then
  `tf` against the golden logits (top-1, top-20, KL) and the MTP's drafts against `mtp/` (G7's verify).
- `glw_kcheck --gpu`'s one SwiGLU output outside its bound.
- Batch invariance under the default split-K (above).
