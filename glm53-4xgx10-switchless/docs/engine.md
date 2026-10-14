# The layer runner and the TP=4 step (engine/model)

G4: GLM-5.3 end to end on the four ranks (two ASUS Ascent GX10, two DGX Spark): the embedding, 78 layers, the
final norm, the vocabulary-split head, the sampler and the MTP layer, over the kernels of `engine/attn`,
`engine/experts` and `engine/comm`. Everything here was run on the ring on 2026-10-13/14, borrowed from the
DeepSeek service through its `ring.sh` (jobs of 15 s to 2 min).

**Status.** Runs on four ranks on synthetic weights and on the real non-expert rank files (G2) with synthetic
routed experts, layer 3 also with its real NVFP4 experts. Decode step at 1 row **46.4 ms** (21.6 tok/s) against
a 34.2 ms bandwidth floor (1.35×); after the 2026-10-14 decode-gap pass 43.6 ms (1.27×) in the interleaved A/B, ~44 ms
on undisturbed ranks, the MoE at its floor and the rest the collectives' latency and attention ([Closing the
decode gap](#closing-the-decode-gap-2026-10-14)); prefill **1,084 / 1,086 / 860 tok/s** for chunks of 2,600 and
8,192 from 0 and the last 8,192 of 32,768 tokens, **968 tok/s** for a whole 32,768-token prompt, with the ring's
collectives overlapped with the compute ([Prefill overlap](#prefill-overlap-2026-10-14); 854 / 794 / 698 and 753
before, after the tensor-core GEMMs of [Prefill GEMMs](#prefill-gemms-2026-10-14); 381 / 374 / 350 on the 16-row
GEMVs), 79 % of the published 1,374 tok/s at 8k. Against the
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
| `prefill.cu` | prefill chunks of up to 8,192 rows (the projections through attn's `gemm()` / `gemm_bf16()`, `pf_proj`; the layers in two pieces of rows, each piece's collectives on a second stream beside the other's compute, `kTunePfOverlap`), the MTP cache fill |
| `ops.cu`, `sample.cu` | the step-input ring, the vocabulary-split embedding, the top-1 for drafts, publishing; the DeepSeek sampler ported (38,720 ids a rank, ids ≥ 154,856 masked) |
| `oproj.cu` | o_proj as the producer half of its all-reduce (`kTuneOproj`, off by default; [Closing the decode gap](#closing-the-decode-gap-2026-10-14)) |
| `tools/step_profile.py` | the per-kernel critical-path profile of `glm_run trace`'s nsys reports, every rank; `--prefill`: one prefill chunk's (`glm_run pftrace`), per kernel kind and per class, and per stream (busy, idle, the side stream under the main one) |
| `l2pf.cu` | the paced L2 prefetch (DeepSeek's l2pf): the next weights during collectives and attention |
| `bind.cu` | binding a `.glw` rank file (names, sizes and layouts checked), synthetic weights, synthetic routed experts for a file whose pools are not filled |
| `tools/glm_run.cu` | the driver: `bench`, `l2pf`, `ab` (two lever sets, interleaved, timing and output bits), `trace` (under nsys), `prefill`, `pfab` (prefill, two lever sets A B B A, timing and output bits; chunks or whole prompts), `pftrace` (one chunk under nsys), `check`, `tf`, `refhash` |
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

Chunks of ≤ 8,192 rows (`Options::prefill_rows`): the embedding all-gathered in 1,024-row pieces; every FP8
projection (x-projection, q_b + wq_b, W_UK, W_UV, o_proj, the dense gate/up and down, the MTP fill's eh_proj and
x-projection) one launch of attn's tensor-core `gemm()` over the chunk, the BF16 router one launch of
`gemm_bf16()` (both the 16-row GEMV's bits; since 2026-10-14, [Prefill GEMMs](#prefill-gemms-2026-10-14)); the
indexer for rows at position ≥ 2,048 through `index_select`'s prefill path (staged queries, `k_score_pf`, the
merged `k_sel_thresh` + `k_sel_compact` selection) in batches sized to a 512 MiB score scratch (rows below attend
densely); `sparse_attn` without split; each sublayer ends with an f32
reduce-scatter by rows, the residual add and the next norm on the rank's quarter of the rows and one
all-gather of h and the normed rows; the MoE through `moe_prefill`. The layers run on two pieces of the chunk's
rows, each piece's sublayer ends on a second stream while the other piece computes (chunks of 1,024 rows and more;
the same bits; since 2026-10-14, [Prefill overlap](#prefill-overlap-2026-10-14)). The MTP cache fill runs after each chunk
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
/ 4,096 / 8,192 rows (1.24 s to 20.5 s a chunk), after the expert prefill tuning (6cb6fce); 397–416 before. With
the prefill GEMMs (2026-10-14): 854 / 802 / 704 tok/s at 2,600 / 8,192 / 32,768 tokens; with the collectives overlapped (T5, 2026-10-14) 1,084 / 1,086 / 860, a whole 32,768-token prompt 968; below.

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

## Prefill GEMMs (2026-10-14)

The prefill path's projections moved from 16-row GEMV launches to the attention side's tuned tensor-core GEMM
(T2, [attention.md § 5](attention.md)), outputs unchanged bit for bit. Seven ring jobs of 1–6 minutes on the four
ranks and one on rank 0 (attn's tests), 01:11–01:49 (+0100); synthetic weights except `tf`.

**What changed.** `prefill.cu`'s `pf_proj` runs an FP8 projection over the whole chunk as one `gemm()` launch at
the GEMV's split-K: the x-projection (split-K 4; also the MTP fill's), q_b with the indexer's wq_b (two segments),
W_UK and W_UV (grouped), o_proj, the dense gate/up (SwiGLU) and down, and the MTP layer's eh_proj (`mtp.cu`;
1–16 rows at decode stay one GEMV). attn's test P shows each of these shapes equal to the 16-row GEMVs in all six
tile configurations (eh_proj, 1536 × 12288, is not in P; the runner's checks below cover it). The router is BF16,
which `gemm()` does not take: attn's `gemm()` gained a BF16-weight form, `gemm_bf16()` (gemm.cu: the BF16 GEMV's
arithmetic, the four MMAs of each 64-k chunk accumulating straight into the sum, the weights staged as each lane's
32 bytes; one 128 × 64 tile shape, 3 stages, 72 KiB), tested equal to the 16-row BF16 GEMV in three new P checks
at 4,600 rows (the router's shape at split-K 4, the whole head, a 1000-row bf16-out case). Two `Tune` bits switch
them, both in `kTuneDefault` (now 0x77): `kTunePfGemm` (32) the FP8 projections, `kTunePfRouter` (64) the router.
Kept as GEMVs: the head (BF16, only for `kFlagLogits`, i.e. `tf`; its logits buffer holds 16 rows, so it stays
16 rows a launch), `index_weights` (BF16, 8 rows, its own kernel: 10 ms a chunk) and any chunk of ≤ 16 rows. The
shared expert is not attn's: `moe_prefill` already runs it as a dense FP8 GEMM in the experts' own layout. **The
indexer** already took attn's prefill path: `index_select` with more than 16 rows stages the queries and runs
`k_score_pf` and the merged `k_sel_thresh` + `k_sel_compact` (4 launches a call, T2), the trace below shows it, so
nothing changed there; a batch tail of ≤ 16 rows takes the decode kernels, the same bits (attn test E). The
batches are 512 MiB of scores: 3,971 rows at `max_ctx` 33,792, 512 at 256k (attn measured calls of 512 rows
~10 % slower than of 2,048 at 32k).

**Method.** `glm_run pfab` (`run.sh pfab`): one process, two lever sets, for each chunk A B B A (A = 0x17, the
16-row GEMVs; B = 0x77), the slot's caches before the chunk filled with hashed values (`fill_cache`), a greedy
`Sampling` on the last row and one MTP draft (so that the MTP layer's cache holds every row's final normed hidden),
`prefill_ms` (events around the whole call: the chunk, the MTP fill, the last row's sample and draft pass); both
lever sets warmed with a 512-row prefill first. After each run: the hash of every layer's cache (KV, index keys,
the MTP layer's) over positions 0 to the chunk's end and the sampled token and value; A and B must agree in all four
runs and on every rank. "2,600 / 8,192 / 32,768 tokens" are one chunk each: 2,600 rows from 0, 8,192 from 0, and
the last 8,192-row chunk of a 32,768-token prompt (at position 24,576). `--max-ctx 33792`. The breakdown: `glm_run
pftrace` (`run.sh pftrace`: one 8,192-row chunk at 24,576 after a warm one, inside the profiler range, no drafts)
under nsys on every rank, read by `tools/step_profile.py --prefill` (per kernel kind and class, the critical-path
increments of the main stream as in [Closing the decode gap](#closing-the-decode-gap-2026-10-14)).

**Prefill throughput** (sources sha256 225b0ca7cb2def35, 01:28–01:31; ms of the two runs of each side):

| Tokens (chunk at) | 16-row GEMVs (A) | tok/s | gemm + gemm_bf16 (B) | tok/s | B/A speed |
|---|---:|---:|---:|---:|---:|
| 2,600 (0) | 6,836.7 / 6,801.0 | 381 | 3,043.6 / 3,043.3 | **854** | 2.24× |
| 8,192 (0) | 21,742.2 / 22,007.7 | 374 | 10,217.3 / 10,222.0 | **802** | 2.14× |
| 32,768 (8,192 at 24,576) | 23,236.4 / 23,560.6 | 350 | 11,622.5 / 11,659.9 | **704** | 2.01× |

The two levers apart: the FP8 projections alone (B = 0x37, an earlier build of the same code, 01:11–01:15)
768 / 725 / 641 tok/s (2.02 / 1.94 / 1.84×); the router's `gemm_bf16` on top (0x37 against 0x77, 8,192 from 0,
01:31) 723 → 799 tok/s, 1.105×. Repeats agree within 1.5 %. **Against the published bar** (1,374 tok/s cold at 8k,
DESIGN.md): 802 tok/s is 58 % of it; the 8,192-row chunk takes 10.2 s where the bar needs 6.0.

**Where a chunk goes** (8,192 rows at 24,576; ms, rank 0, the range over the four ranks in brackets; "before":
0x17, 23.47–23.49 s; "after": 0x77, 11.67–11.71 s):

| Class (kernels) | before | after | share after |
|---|---:|---:|---:|
| collectives, waiting included: all-gather of h + normed rows 1,380, reduce-scatter after o_proj / dense down 1,018, of the routed partials 937, of the shared partials 809, x-projection all-gather 175, the rest 110 | 4,927 | **4,429** (4,155–4,754) | 37.8 % |
| attention: `k_attn` (33 ms a layer, no split) 2,592, `k_prep` 32 | 2,620 | **2,624** (2,512–2,721) | 22.4 % |
| MoE prefill: `k_pdown_c` 1,220, `k_pup_ws` 996 + 180 (two launch shapes a layer), `k_pdown` 175 (four a layer), the rest 9 | 2,545 | **2,580** (2,454–2,686) | 22.0 % |
| projections: o_proj + dense down 645, q_b (+ wq_b) 182, x-projection 109, W_UV 72, W_UK 64, router 48, dense gate/up 35 | 12,484 | **1,154** (1,117–1,192) | 9.9 % |
| indexer: `k_score_pf` 497, `k_sel_compact` 179, `k_sel_thresh` 64, `k_q_stage` 15, index weights 10, the rest 10 | 770 | **775** (712–775) | 6.6 % |
| norms, residual adds, embedding | 149 | **149** | 1.3 % |

Before, the projections were 53 % of the chunk (the router's 16-row BF16 GEMVs alone 1,132 ms); after, they are
10 % (o_proj 8.0 ms at 8,192 rows: 52 TFLOPS, as attn measured it). **What is left is the ring and two kernels.**
A MoE layer's collectives move three 201 MB f32 reduce-scatters (o_proj's, the routed and the shared partials) and
two 201 MB bf16 all-gathers (h and the normed rows, after each sublayer). The shared partials' reduce-scatter
follows the routed one, so the ranks arrive together: 10.8 ms on every rank, ~14 GB/s a rank of the 151 MB it
sends. So ~4 s of a chunk is the ring's bandwidth, not its latency, and it runs on the compute stream: nothing
overlaps it. Then sparse attention without split (33 ms a layer at 8,192 rows: ~18 TFLOPS of its MMAs) and the
expert GEMMs at T1's level. For 1,374 tok/s the 8k chunk must lose 4.3 s of its 10.2. Overlapping the collectives
with the compute (a second stream and sub-chunks; rows are independent, so outputs stay exact) bounds the traced
chunk by its compute, ~7.3 s (~1,120 tok/s; the chunk from position 0, not traced, is 1.4 s shorter); the rest has
to come from attention and the experts.

**Exactness.** `pfab`: A and B the same bits at all three chunks, in all four runs and on every rank (hashes
f561aaa161f5703e, b5d2be0e85c00e60, 01f4fa4803ca445f; the same in the earlier build), and 0x37 against 0x77 too.
`tf` via prefill on the real rank files (`run.sh tf-prefill`, the 291-token `short` sequence, layers 0–3 real,
routed experts synthetic): all 20 dump files (argmax, top-20, log-probabilities, lse, logits, the sampled rows,
the hidden rows after layers 0–4 and 77) byte-identical between the base build (067ba12, its own default 0x17),
this build's default and this build with `--tune 17`, and identical to the `short-prefill` dump of 2026-10-13.
`run.sh check` passes (chunking, decode = prefill, 16 rows = 16 one-row steps, graph = eager, ranks agree,
sampling), run on both builds. attn's tests: 44 pass at S = 300, 96 at S = 4,600 (93 before, plus the three BF16
checks), memcheck on A C B F P at S = 300 clean (41 pass; the BF16 checks need the head data of S = 4,600 and were
not run under memcheck). The server's ring tests (`server/tests/ring_job.sh`: serve, batch, front, kept) pass on a
`glm53_rank` built from this tree (sources b8ba34daeb3d6f71). On the final sources (merged with glm53 at 05cf496,
comments changed since; `glm_run` d32601a9e8517dd6, `glm53_rank` 2930a17b4553e7e6, 01:45–01:49): `check` passes,
`pfab` at 2,600 rows the same bits and hash (384 → 855 tok/s), the server's ring tests pass. Decode is unchanged: the step's code is untouched
except eh_proj's call, which at 1–16 rows is the same single GEMV launch.

**What failed or was not done.** Nothing failed. The first A/B job overlapped a CPU build on rank 0 (nice 19);
the table's job ran alone. Not measured: a whole 32,768-token prompt (four chunks), `pfab` without the MTP draft;
not run: memcheck over `gemm_bf16`.

## Prefill overlap (2026-10-14)

After T4 a chunk spent 38 % of its time in the ring's collectives, bandwidth-bound and on the compute stream, so
nothing overlapped them. T5 runs them on a stream of their own beside the compute, outputs unchanged bit for bit.
Ten ring jobs of 1–7 minutes (one of them, attn's tests, on rank 0 only), 02:00–03:07 (+0100); synthetic weights
except `tf` and the server's tests.

**What changed** (`prefill.cu`, one `Tune` bit `kTunePfOverlap` = 0x80, now in `kTuneDefault` = 0xf7). The chunk's
rows are split into pieces (`Options::prefill_pieces`, default 2; a multiple of 64 rows each, at least
`prefill_min_piece` = 512, so chunks under 1,024 rows stay one piece). Each sublayer computes the pieces in order on
the compute stream; a piece's sublayer end (the f32 reduce-scatter by rows, the residual add and norm on the rank's
quarter of the piece, the all-gather of h and the normed rows) runs on a second stream, created at the highest
priority so that a collective's blocks get the next free SMs, while the next piece computes. Events carry the
dependencies: a piece's sublayer end waits for its compute, the piece's next sublayer waits for its end. There is
one ring channel, so every collective goes to the second stream in the same order on every rank. The x-projection
and indexer-query all-gathers are short and the compute waits for them, so a long sublayer end is queued only
behind the next piece's short ones (deferred); the last piece's MoE end waits behind the next layer's first piece's.
With two pieces a layer runs: piece 0's attention while the layer before's piece-1 end runs; piece 1's attention
while piece 0's attention end runs; piece 0's MoE while piece 1's attention end runs; piece 1's MoE while piece 0's
MoE end runs. The overlapped collectives use `prefill_comm_blocks` (8) blocks, not 16. Bits: a row's arithmetic is
the whole chunk's. The projections, the router, the indexer and the attention compute a row alike whatever rows
share the launch (attn's tests P and E), and so do the experts (experts.h: a call's results do not depend on the
other rows). A piece's attention reads the cache entries its own and the earlier pieces' `attn_prep` wrote. The
reduce-scatter adds the four ranks' partials in rank order wherever the piece boundaries fall, and the add and the
norm go row by row. With the bit off (or one piece) the launches are the unpipelined chunk's, in the same order.
The embedding's all-gathers stay on the compute stream before the layers; the chunk ends by joining the second
stream, so the MTP fill, the sample and the decode steps after it see a quiet ring. `glm_run` gained
`--pf-pieces`, `--pf-blocks`, `--pf-min-piece`, and `pfab` takes a whole prompt longer than a chunk;
`step_profile.py --prefill` prints a per-stream summary (busy time, the main stream's idle time, the side stream's
time under the main stream's kernels).

**Choosing pieces and blocks** (sources 0d9ee41d6b627474, jobs 1–4, 02:00–02:29; `pfab` 0x77 against 0xf7, 8,192
rows at 24,576, tok/s of the overlapped side; the unpipelined side 699–713 in every job):

| Pieces | comm blocks 2 | 4 | 8 | 12 | 16 |
|---:|---:|---:|---:|---:|---:|
| 2 | 869 | 882 | **892** | 849 | |
| 3 | | | 887 | | |
| 4 | | 852 | | | 827 |
| 8 | | 811 | | | |

At 8,192 from 0: 2 pieces 1,106, 3 pieces 1,060, 2 pieces with 12 blocks 1,058. More pieces mean more short
all-gathers to wait for and smaller GEMMs; more blocks take more SMs from the compute. Repeats of one setting
differ by a few per cent (2 × 8 at 24,576: 892 in job 3, 860 in job 5's `pfab` on the final sources, 866 in its
`pftrace`).

**Prefill throughput** (final sources: `glm_run` 139d188c604ea91a, merged with glm53 at 3b2ae08; job 5,
02:34–02:42, and job 6, 02:44–02:49; ms of the two runs of each side, A = 0x77, B = 0xf7):

| Tokens (chunk at) | unpipelined (A) | tok/s | overlapped (B) | tok/s | B/A speed |
|---|---:|---:|---:|---:|---:|
| 2,600 (0) | 3,046.1 / 3,041.6 | 854 | 2,394.8 / 2,403.2 | **1,084** | 1.27× |
| 8,192 (0) | 10,301.5 / 10,343.7 | 794 | 7,544.4 / 7,545.6 | **1,086** | 1.37× |
| 8,192 at 24,576 (the last chunk of 32,768) | 11,717.8 / 11,749.2 | 698 | 9,525.8 / 9,518.7 | **860** | 1.23× |
| a whole 32,768-token prompt (four chunks) | 43,791.8 / 43,274.3 | 753 | 33,959.1 / 33,738.5 | **968** | 1.29× |

**Against the published bar** (1,374 tok/s cold at 8k): 1,086 tok/s is 79 % of it (T4: 58 %); the 8,192-row chunk
takes 7.5 s where the bar needs 6.0.

**Where a chunk goes now** (`pftrace`, 8,192 rows at 24,576, the same binary with 0x77 and with the default, job 5;
ms; rank 0, the four ranks' range in brackets):

| | before (0x77): 11,469–11,507 ms | after (0xf7): 9,446–9,485 ms |
|---|---:|---:|
| compute stream busy | 11,476 (all of it, collectives included) | **8,745** (8,371–9,225) |
| … attention (`k_attn`, `k_prep`) | 2,651 (2,510–2,651) | 3,239 (3,146–3,383) |
| … MoE prefill | 2,565 (2,462–2,580) | 3,264 (3,059–3,415) |
| … projections | 1,152 (1,122–1,174) | 1,288 (1,253–1,409) |
| … indexer | 774 (714–774) | 934 (877–966) |
| … collectives on the compute stream | 4,185 (4,185–4,505) | 19 (the embedding's; 19–59) |
| compute stream idle (waiting for the other stream) | 8 | **700** (253–1,088) |
| collectives' stream busy | – | 5,322 (4,665–5,589), of it under compute 4,610 (4,403–4,610) |

So 87 % of the ring's time now runs under compute, and it takes longer there (5.3 s against 4.2 s: 8 blocks, and
the compute beside it). **The cost is the compute itself:** beside the ring's copies the compute kernels run 20 %
slower on rank 0 (7,290 → 8,745 ms busy) and 27 % on rank 2 (7,265 → 9,225), every class alike (attention +22 %,
MoE +27 %, indexer +21 % on rank 0), more than the 8 SMs the collectives take (17 %). The ring's traffic goes
through the same memory: a reduce-scatter of 201 MB copies its input into the send buffers, the NICs read those and
write the neighbours' quarters, and the reduction reads four quarters; by that count (an estimate, not measured) on
the order of a gigabyte of DRAM traffic in ~10 ms. Rank 2, the slowest under it, sets the chunk: its compute stream
is idle only 253 ms. The idle time of the other ranks is mostly the x-projection all-gather's wait (the `k_prep`
after it: 545–715 ms of increments on ranks 0, 1, 3 against 36 ms of kernel) and the next layer's first GEMM
waiting for the MoE's end (`k_gemm g11x32 <k_pdown_c`: 66–336 against 48–57).

**Exactness.** `pfab`: A and B the same bits in all four runs, at every size, on every rank and at every piece
count and block count tried (hashes f561aaa161f5703e, b5d2be0e85c00e60, 01f4fa4803ca445f: T4's, and 8ee634566afdf576
for the whole 32,768-token prompt). `run.sh check` passes with the overlap on (its 2,048- and 1,300-row chunks run
as two pieces): chunking, decode = prefill, 16 rows = 16 one-row steps, graph = eager, ranks agree, sampling. `tf`
via prefill on the real rank files (`run.sh tf-prefill`, the 291-token `short` sequence): all 20 dump files
byte-identical to the `short-prefill` dump of 2026-10-13, with the default (291 rows: one piece) and with
`--pf-min-piece 64` (pieces of 192 and 99 rows, the overlap on). attn's tests from this tree: 44 pass at S = 300, 96
at S = 4,600 (attn is unchanged). The server's ring tests (`server/tests/ring_job.sh`: serve, batch, front, kept) pass
on a `glm53_rank` built from this tree (sources 98817b82c9ab36db). **Decode is unchanged**: the step's code is
untouched; `ab` 0x77 against 0xf7 43.612 / 43.549 ms at 1 row, 95.482 / 95.432 at 6, the same bits; `bench` of the
T4 build (d32601a9e8517dd6) against this one, A B B A, wall ms a step over 32 steps: 45.21 / 45.43 against 45.38 /
45.48 at 1 row, 96.83 / 97.24 against 97.46 / 96.97 at 6 (the GPU medians of 4 steps, 44.4–49.5 ms, are too noisy
to compare).

**What failed or was not done.** Evict-first cache hints (`__ldcs` / `__stcs`) on the bulk reduce-scatter's and the
multi all-gather's copies, meant to keep the ring's streams out of the compute's L2 (sources 39643ccbf8d251aa, job
3): 892 → 866 tok/s at 24,576, 1,106 → 1,088 at 8,192 from 0, no gain; reverted. The server's front phase failed
once (job 9, 03:02) because the worktree had no `glmfront` build; run again with the main checkout's (the frontend
is unchanged), it passes. Not tried: the MoE's shared expert first so that its reduce-scatter starts during the
routed experts (with two pieces the MoE end already runs during the other piece's MoE), a fused residual + norm +
all-gather after the reduce-scatter, the ring's copies with fewer bytes (the producer writing its partials straight
into the send buffers, as decode's posted partials do; a chunk's partials are far beyond the 4 MiB buffers).

## What remains

- **Prefill** (1,086 tok/s at 8k after the 2026-10-14 overlap, 79 % of the published 1,374): the compute now
  runs 20–27 % slower beside the ring's copies (their memory traffic; [Prefill
  overlap](#prefill-overlap-2026-10-14)): fewer bytes through memory per collective (a fused residual + norm +
  all-gather, the producers writing into the send buffers) would give that back; the compute stream still waits
  0.25–1.1 s a chunk, mostly for the short x-projection all-gathers; then `sparse_attn` at prefill (no split,
  ~18 TFLOPS alone) and the expert GEMMs, a third of the chunk each. The slowest rank (2) sets the chunk. The
  indexer's batches shrink to 512 rows at a 256k `max_ctx` (512 MiB scratch).
- **The decode gap** (1.27–1.30× at 1 row after the 2026-10-14 pass, ~1.29× on undisturbed ranks; G4's target
  1.25×): fewer collectives (merge the x-projection all-gather into the previous all-reduce's epilogue, the
  indexer queries' all-gather into it on full layers), the embedding all-gather onto the sampler's exchange, the
  ring's per-collective latency; the MoE is at its floor ([Closing the decode gap](#closing-the-decode-gap-2026-10-14)).
- **Real routed experts**: the rank files' pools are reserved (the fill needs the user's disk decision); then
  `tf` against the golden logits (top-1, top-20, KL) and the MTP's drafts against `mtp/` (G7's verify).
- `glw_kcheck --gpu`'s one SwiGLU output outside its bound.
- Batch invariance under the default split-K (above).
