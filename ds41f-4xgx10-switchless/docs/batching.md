# Several sessions in one decode step (design note)

This page gives what batching would take, what it would buy by the measured row costs, and a cheaper
step that gets most of the latency without the kernel work. The owner said go on 2026-10-12. The runner
leads, and [Plan](#plan) at the end gives the steps, who builds what, and the interfaces, fixed before
any kernel was written. When it was written the engine ran one request at a time.

## Status (2026-10-12)

- **B1, time-slicing,** has served in production since 12:59 (master cbb5f61). Behind a request
  decoding 600 tokens, a 16k request's first token comes after 1.9 s instead of 7.1 s, and a 2k
  one's after 0.48 s instead of 2.1 s. The last of the three ends 0.66 s (5 %) later.
- **B2, rows of several sessions in one step,** is merged (master d23a332) and passes on the ring
  with `--slots 4` ([engine.md, Scheduling](engine.md#scheduling-time-slicing-and-batching)). Every
  request's tokens and log-probabilities equal its run alone: in batches of two to four, staggered,
  with more requests than slots, cancelled or stopped inside a batch, and kept across a planned stop.
  `--slots` stays 1 by default, so nothing changes until a service asks for slots.
  - The soak with four and six clients: 499 requests, none failed, memory flat on every rank and in
    the front-end, a small turn's first token in 0.39–0.67 s (median)
    ([The soak with four and six clients](#the-soak-with-four-and-six-clients)).
  - A batch step takes 23.2 ms at two rows, 27.6 at three and 33.6 at four (medians), where
    [Estimates](#estimates) said 23.2, 27.8 and 32.4.
  - A batch has no drafts. So B2 buys fairness and first tokens, not throughput, as the estimates
    said: four requests together made 91.9 tok/s from the first submit, prefills included, and one
    after another, alone with drafts, they decode at 93.4.
  - Departures from the plan:
    - A new request's prefill runs whole while the others wait (prefill first), not in chunks of
      2,048 rows between their steps.
    - The last request decoding in a batch leaves it, to go on alone with drafts.
    - With more requests than slots, they take turns of `--slice` tokens, as in B1.
  - Open:
    - A request's drafter windows miss the positions it decoded in a batch. When it goes on alone, its
      first drafts see stale windows: they are still verified, so only acceptance suffers, until the
      windows fill again.
    - The 2,048-row prefill interleave.
- **B3** was estimated from measured step times and acceptance ([B3 from
  measurements](#b3-from-measurements)). With 2–4 agent sessions decoding at once, it would decode
  18–32 % more in all than the better of B1 and B2, and one session alone would not change. But the
  owner runs several agents at once only occasionally, so it gains only in those minutes, 8–14 % end
  to end: it is not built for now, and production stays on `--slots 1`.
- **B4:** the kernels agent's half (GEMVs and the MoE up to 16 rows, the same bits) and the attention
  agent's half (9–16 rows in groups of 8) are on master. The runner's half (step buffers and graphs
  past 8 rows) is not wired for B3's sake. Drafts of one session past 8 rows (prompt lookup) would
  need it, and `kMaxSessRows` past 8, which decode_prep's race at 16 rows of one session forbids today.
- **B3 is built and measured** (2026-10-13, [B3: the build](#b3-the-build-2026-10-13)): the owner asked
  for 2–4 streams at once, with one stream kept as fast as it is, within 2 %. B4's runner half goes with
  it. Four different streams decode 131 tok/s in all (one alone 97; B2's undrafted batches 122), two
  107 (B2: 88), each stream with its drafts; one session is unchanged (A/B/B/A, six runs each: 8.322
  against 8.322 ms a token with drafts). Its phase 2a (one draft pass over the sessions, the drafts bit
  for bit) is merged too: 109 / 127 / 136 tok/s in all at two, three and four streams. A gate across the
  sessions is not built.

## Why

In the [soak](engine.md#soak-2026-10-12), three agent clients sent requests back to back. A turn
adding under 1k positions waited 3.0–3.6 s (median) for its first token at the front-end. Rank 0
needed about 0.25 s for it; the rest was waiting for the requests ahead to finish, their decode
included. The SGLang kit batched concurrent requests. The engine was busy about 57 of the 60
minutes: 12.9M positions prefilled at about 8k tok/s (~27 min) and 191k tokens decoded at about
107 tok/s (~30 min).

## What a row costs

Measured on one session at 1k of context ([Drafts, measured](engine.md#drafts-dspark-measured)):
- a plain step (one row): 18.55 ms; with the draft pass: 21.10 ms;
- each further verify row: 4.28, 3.88, 3.50, 2.85 and 3.09 ms for rows 2–6. On bench.py's short
  prompts, 3.4, 3.5, 3.2, 3.3 and 2.9 ms (the steps the gate chose, in the gates' A/B);
- at six rows the step's 36.3 ms are the MoE 21.4 ms, the other GEMVs 9.6 (read once whatever the
  rows), collectives 4.0, attention 1.0 and mixes 0.3 ([Step times](engine.md#step-times)).

A row's cost is mostly its routed experts streamed from memory. A routed expert's slice is 4.7 MB
per rank and layer (MXFP4 with scales: 67.2 GiB for 384 experts in 40 layers,
[layout.md](layout.md)). A row's six experts over 40 layers are then ~1.13 GB, ~4.9 ms at the MoE
kernels' ~228 GB/s. Verify rows are consecutive tokens of one sequence, and they share experts with
the rows before: the second row costs 3.4–4.3 ms and later ones 2.9–3.9.

**A row of another session** was measured with `--mode bench --bench-stride 160` (2026-10-12, at 1k
and 64k of context, the median of 37 graph replays per row count). For each row count, after the
consecutive rows the bench runs as many steps of three other kinds:
- unrelated rows: tokens from scattered golden positions, the rows 160 positions apart so that their
  windows do not overlap;
- the same rows with row r's router logits rotated by 64 r experts (bench only,
  `Model::set_route_shift`), so that they share almost no experts;
- a few eager steps of each kind, which count the distinct experts per layer.

| rows | consecutive | unrelated | rotated | distinct experts per layer: consecutive / unrelated / rotated |
|---|---|---|---|---|
| 1 | 18.56 ms | 18.57 ms | 18.55 ms | 6.0 / 6.0 / 6.0 |
| 2 | 22.71 | 22.29 | 24.04 | 9.8 / 9.6 / 11.9 |
| 3 | 26.50 | 25.34 | 29.60 | 13.8 / 12.7 / 17.7 |
| 4 | 30.45 | 27.66 | 34.57 | 16.2 / 15.3 / 23.5 |
| 5 | 33.67 | 30.49 | 39.47 | 18.9 / 17.2 / 29.1 |
| 6 | 37.68 | 32.43 | 44.31 | 21.1 / 20.1 / 34.8 |

The table is at 1k. At 64k the six-row steps take 36.86, 32.71 and 44.45 ms.
- Unrelated rows cost less than consecutive ones, 2.8 ms per row. They share more experts than a
  sequence's own rows do: a token out of its context routes to common experts. So they do not stand
  for another session.
- Rows that share almost no experts cost 5.5, 5.6, 5.0, 4.9 and 4.8 ms for rows 2–6: 5.15 ms each on
  average at 1k and 5.10 at 64k. That bounds another session's row from above.
- Over the 36 kinds and row counts, a step costs 14.5 ms + 0.72 ms per distinct expert per layer +
  1.1 ms per row (rms 0.8 ms). Suppose rows route independently, each expert as often as in 126 rows
  of this text. Rows 2–6 would then reach 10.7, 14.8, 18.4, 21.8 and 24.8 distinct experts, and cost
  4.5, 4.0, 3.7, 3.5 and 3.2 ms (3.8 mean).

So a row of another session costs 3.8–5.1 ms. The low end holds if sessions overlap in their experts
as much as random rows of one text do; the high end holds if they never overlap. Where real
sessions sit is not measured; sessions on unrelated work should sit nearer the high end. The
estimates below keep 4.6 ms:
- at 5.1 ms, the totals fall by at most 2.5 % with drafts, and by 4.4 % without (four sessions: 118
  tok/s instead of 124);
- at 3.8 ms, they rise by 2–5 % with drafts, and by 8 % without.

The draft pass handles one anchor today (2.55 ms). For B sessions it would handle B anchors in one
pass; the estimates add 0.8 ms per further anchor (a guess: its weights are read once, its rows grow).

Attention grows with context: a one-row step at 256k takes 0.38 ms more than at 1k, and every row
of a batch pays that for its own session.

## Estimates

These came from a model, before B2. [B3 from measurements](#b3-from-measurements) replaces them for
B3.

**Without drafts**, B sessions, one row each: a step takes 18.55 + 4.6 (B − 1) ms and emits B tokens.

| sessions | step | tokens per s, all sessions | per session |
|---|---|---|---|
| 1 (today, k = 1) | 18.55 ms | 53.9 | 53.9 |
| 2 | 23.2 ms | 86 | 43 |
| 3 | 27.8 ms | 108 | 36 |
| 4 | 32.4 ms | 124 | 31 |

**With drafts**, B sessions share a budget of rows: 8 (the graphs' and step buffers' limit today) or 12.
- Drafts are spread by survival, so every session gets about the same number.
- A session's own draft rows cost the measured 4.28, 3.88, … ms.
- Another session's first row costs 4.6 ms plus 0.8 for its anchor in the draft pass.
- Each draft is accepted with probability p given the one before it (survival p, p², …).
  - At p = 0.87, one session alone gets 113 tok/s. That is about the soak's 110 (4.1 tokens per step at
    37.5 ms).
  - p = 0.8 is weaker content: about what one session gets at 1–2 drafts on the golden run. Prose is
    weaker still (69 tok/s in bench.py).

The one-session row is the same model, so every row compares with it.

| sessions | drafts per session | rows | step | tok/s, all sessions, p = 0.8 | p = 0.87 | per session, p = 0.87 |
|---|---|---|---|---|---|---|
| 1 | 5 | 6 | 38.7 ms | 95 | 113 | 113 |
| 2 | 3, 3 | 8 | 49.8 ms | 119 | 132 | 66 |
| 3 | 2, 2, 1 | 8 | 52.5 ms | 127 | 136 | 45 |
| 4 | 1, 1, 1, 1 | 8 | 54.4 ms | 132 | 137 | 34 |
| 2 | 5, 5 | 12 | 61.7 ms | 120 | 141 | 71 |
| 4 | 2, 2, 2, 2 | 12 | 69.9 ms | 140 | 150 | 38 |

With drafts, then, batching adds 17–21 % to the total at 8 rows when drafts are good (p = 0.87) and
25–39 % when they are weaker (p = 0.8). A budget of 12 rows adds up to 33 % and 47 %.
- A draft row and another session's row cost about the same (3–5 ms of experts). A draft yields p,
  p², … tokens and another session's row yields one, so the total rises as rows move from weak drafts
  to sessions.
- Without drafts, batching more than doubles the total at four sessions, but drafts already take
  most of that gain.
- Each session's own rate falls: 113 alone, 66 with two sessions, 34 with four (p = 0.87).

**Time to first token** of a new request, a turn adding under 1k positions while others decode:
- Today: the requests ahead finish first, 3.0–3.6 s with three busy clients (soak).
- Batched, a new request's prefill goes first: one step (≤ 55 ms), then its own prefill, about
  0.25 s, so ~0.3 s.
  - A 15k-token tool output takes ~2 s, as alone.
  - The decoding sessions pause for that prefill, or run between its chunks.
  - Two prefills still queue: prefill is compute-bound, about 8k tok/s whatever the batch.

## What it would take

1. **The step takes rows from different sessions.** Each row brings its session's state, not only
   its position:
   - the window ring of each layer and the compressor's carries;
   - the four compressed caches and their index keys;
   - the draft module's windows.

   A per-step table of live slots in `StepIn` would do it, with a slot index per row. Kernels then
   index the slot's buffers instead of taking one `Window`, `Compressed` and `IndexKeys`:
   - `kv_write` and `decode_prep` (`compressor_emit`, into the row's session);
   - `index_dense` and `index_fused` (the row's keys, scanned to the row's position, as
     `index_dense` already reads the scan's length from the positions);
   - `sparse_attn` (the row's window and compressed entries);
   - `kv_write_sets` (DSpark's windows; it already takes four sets).

   Sampling takes its parameters per row (temperature, top-p, top-k, seed, top logprobs), not per
   step. Engram hashes each row's n-grams from its own session's history.
2. **Live sessions in device memory.** Today one working state is live, and a switch copies pages in
   and out of the store. A resume from a session's end, with one new position, took ~27 ms in the
   soak. Batching needs B working states at once. Each is ~890 B per position for the compressed
   caches (233 MB at 262,144), 5.4 MB of rings, the carries and the draft windows. Four slots sized
   for 262k are ~0.95 GiB per rank, from the store's pool or the 18.7–20.2 GiB left free. The
   alternative is attention reading the store's pages through a per-session page table: no copies,
   but more work in the index scans and the sparse gathers.
3. **Prefill of one session beside the others' decode.** The simplest is to alternate: a prefill
   chunk, then a decode step for the batch.
   - Chunks of 8,192 rows take ~1 s, so with sessions decoding, smaller chunks (2,048 rows, ~0.3 s)
     keep their pauses short, at some loss of prefill speed.
   - Running both at once on two streams is not worth it. They compete for the same bandwidth, and
     their collectives would need separate ring channels to keep every rank's order.
   - The prefill path writes the live working state today. It would write the new session's slot.
4. **Drafts with a row budget shared across sessions.** One draft pass for all anchors. Then one
   gate over every session's survivals: start from one row per session, and add drafts in order of
   expected tokens per ms while the next one beats the batch's rate. That is the `marginal` gate
   (`--spec-gate marginal`), applied across sessions; `k_spec_next` becomes one block over the
   batch. The spec state (`SpecState`: count, position, bonus, the next rows) becomes per session.
5. **Graphs by row count.** One graph per total row count stays enough if everything per row comes
   through `StepIn` and the slot table. The number of sessions needs no graphs of its own. A budget
   past 8 rows needs graphs for 9–16 rows and step buffers for 16. The step buffers hold the
   activations, the split-K partials and the gathered logits (4.1 MB at 8 rows).
6. **The loop and the scheduler.** `kCmdDecode` runs one session's loop, driven by the device's
   samples on every rank. A batch loop must re-plan when a session ends, is cancelled (rank 0's
   control word stops the whole loop today) or joins after its prefill. Each change is a command
   from rank 0. Admission is prefill-first, with at most B sessions decoding.

### Files and owners

| where | what changes | whose |
|---|---|---|
| `engine/server/serve.cpp`, `cmd.cpp`, `cmd.h` | scheduler, the batch loop, per-session cancel and ends | runner |
| `engine/model/model.cu`, `impl.h`, `ops.h`, `ops.cu`, `state.cu` | live slots, `StepIn`'s slot table, per-row histories for Engram, save and restore into a slot | runner |
| `engine/model/sample.cu` | per-row sampling parameters | runner |
| `engine/model/spec.cu`, `engine/dspark/dspark.cu` | per-session spec state, the draft pass over B anchors, the shared budget | runner (the draft-pass work is the runner's) |
| `engine/attn/`: `kv_write`, `decode_prep` / `compressor_emit`, `index_dense`, `index_fused`, `sparse_attn`, `kv_write_sets` | per-row session buffers | attention agent |
| `engine/kernels/`: `moe.cu` (`k_moe_up`, `k_moe_down`), `gemv_fp8.cu`, `gemv16.cu` | rows from different sessions route to more experts (up to 6 per row plus the shared one); more than 8 rows if the budget grows | kernels agent |
| `engine/model/prefill.cu` | nothing: the runner points the eager paths at a session's slot ([Plan](#plan)); smaller chunks are the scheduler's choice | |
| `frontend/` | nothing required: it already sends concurrent requests | |

### Checks it would need

- Every row computes as it would alone. No reduction may mix rows, so greedy output stays
  independent of the batch: the same hashes as today per session (1k febc7dc54ee40a39, 16k CED
  81f772ab79a950e4). Today's tokens are identical with and without drafts, so the verify rows
  already compute independently.
- The golden checks per session, inside a batch.
- The soak with three and more clients: time to first token, per-session decode, no errors.

## A cheaper step first

Preemptive time-slicing in `serve.cpp`, with no kernel change, gets most of the latency:
- A new request ends the decode loop running at its next step. The session keeps its state, as on
  a cancel, but the client is not told. The scheduler runs the new request's prefill and first
  token, then gives the decoding sessions turns of, say, 64 tokens (~0.6 s) each.
- A switch is a resume from the session's end, ~25 ms (the soak's one-position resumes took ~27 ms),
  so 64-token turns lose ~4 % to switches.
- A small turn's first token would come in ~0.3–0.5 s instead of 3 s. The total stays at today's
  ~110 tok/s, shared by the sessions, less those switches. A decoding session's tokens come in
  bursts, a turn's worth every B turns.
- Its files: `serve.cpp` and the decode loop's early exit (`cmd.cpp`), the runner's.

Batching then adds 17–39 % to the total with drafts (8 rows), and more without them, for the kernel
work above.

## Plan

Four steps, each useful on its own. B1 needs no kernel work, and B2's kernels can be written while
B1 is built.

| step | what | whose | what it buys |
|---|---|---|---|
| B1 | Time-slicing ([A cheaper step first](#a-cheaper-step-first)). A request that arrives while another decodes pauses that decode at its next step and is prefilled. Then the decoding sessions take turns of 64 tokens, switching through the store as today (~25 ms a switch) | runner | the first token of a small turn behind busy clients in ~0.3–0.5 s instead of 3 s |
| B2 | Rows of several sessions in one step, without drafts: slots for 4 live sessions, the slot table, sampling and Engram per row, and the batch loop | attention agent: slot-aware kernels; runner: the rest | sessions decode together, without turns or switches; the base for B3 |
| B3 | Drafts across sessions: one draft pass over every session's anchor, and one gate over all of their survivals ([What it would take](#what-it-would-take), item 4) | runner | measured: 18–32 % more tokens per second in all with 2–4 agent sessions, at up to 16 rows ([B3 from measurements](#b3-from-measurements)); not built for now |
| B4 | A budget of 12–16 rows | kernels agent: dsk to 16 rows; attention agent: `kMaxDecodeRows` 16, with `index_dense`'s shared memory per group of at most 8 rows ([Interfaces](#interfaces)); runner: step buffers and graphs | up to 33–47 % more |

### Interfaces

**attn (the attention agent).** A state struct can stand for an array of session slots: slot s is at
base + s · stride, and a table on the device gives each row its slot. With no table (`slot` null),
every row uses slot 0. So the new fields change no existing call:

```cpp
// attn.h: fields added to the state structs; everything else as it is
struct Window          { ...; const int* slot = nullptr; size_t kv_stride = 0, sc_stride = 0; };  // bytes
struct Compressed      { ...; const int* slot = nullptr; size_t kv_stride = 0, sc_stride = 0; };  // bytes
struct IndexKeys       { ...; const int* slot = nullptr; size_t k_stride = 0, sc_stride = 0; };   // bytes, whole entries
struct CompressorCarry { ...; const int* slot = nullptr; size_t stride = 0; };  // floats, kv and score alike
```

- `slot[r]` (device memory, `[rows]`) is the slot of row r's session.
  - A session's rows are consecutive and ascend in position (one row each in B2, an anchor and its
    drafts in B3).
  - Sessions follow one another in any order, and their positions are unrelated: one at 200k, the
    next at 1k.
- The strides keep every load aligned. Each slot-aware entry point checks them on the host and dies on
  a bad one:
  - `Window` and `Compressed`: bytes, multiples of 16.
  - `CompressorCarry`: floats, a multiple of 4, one stride for kv and score.
  - `IndexKeys`: whole entries, k_stride = 64 · entries = 16 · sc_stride, at most 2^24 entries a slot.
    The indexer reaches a row's keys at a 32-bit entry offset from the pointers, slot · entries.
- `kv_write`, `decode_prep` (kv, `compressor_emit` with its carry), `kv_write_sets` and `sparse_attn`:
  row r reads and writes its own slot's buffers.
  - `compressor_emit` takes at most 16 rows with a table. Its carry records a call's last 16 rows, so
    up to 16 every row's projections reach its own slot's carry. A step has at most 8 rows, 16 in B4.
  - `kv_write_sets` also takes each row's own position (`pos`, device, `[max_rows]`, negative for
    none) in place of `*pos0 + r`, so that several sessions' accepted rows go in one call (B3).
- The indexer (`index_dense`, `index_dense_blocks`, `index_fused`, `index_merge2`):
  - Row r scans its own slot's keys, as far as its own length, (pos[r] + 1) / ratio.
  - Rows of one session still share each key tile; a CTA's group of rows never spans two sessions.
  - The grid stays fixed, so one graph per row count serves any mix of sessions and contexts.
  - Layer 20's pool is per row, as today.
  - With no table, `index_dense` and `index_fused` run instances compiled for one session: today's
    code, unchanged.
    - **How that came about.** With a session's keys as an entry offset (623031a), index_dense was one
      ~2 µs timer tick slower than master at 16k with no table. bench_index then had it 2–8 % (1–3
      µs) slower with no table (5a06cd0's message). The causes were the key offset's signed
      arithmetic (19 instructions more a chunk) and the CTAs' partition from dependent global loads.
      The one-session instances (5a06cd0) remove both.
    - **Measured** in B2's last job (a5264cf, 2026-10-12 10:39). bench_index took interquartile means
      of 200 launches after an L2 flush, two runs each, over 18 cases: 16k and 131k; 1, 4 and 8 rows;
      layers 2/8/14, 20 with the pool, and the pool scans.
      - Without a table, summed over the cases: −0.7 % of master's for index_fused, −0.4 % for
        index_dense. No case was more than 0.9 µs slower.
      - With a one-session table: −0.3 % and +1.1 %, at most +2.4 µs (131k, 8 rows, layers 2/8/14).
      - bench_step's attention side at 8 rows and 131k: master 2712 µs, no table 2697, a table 2715.
  - `index_score` takes no table and dies on one. It is the reference for prefill's scan
    (`kernels/prefill_index.cu`), and prefill runs one session at a time through `use_slot`, with a
    null table.
- B4 (16 rows): `index_dense` keeps each of a CTA's rows' queries and two 4 KiB histograms in shared
  memory: 82 KiB at 8 rows, 164 KiB at 16, over the 99 KiB a CTA may have. Past 8 rows its shared
  memory must go per row group, a CTA taking at most 8 rows (it already splits the rows into groups
  for short scans). A session with more than 8 rows then reads its keys once per group.
- Every row computes, bit for bit, what it computes alone in a one-session step at the same position.
  The tests: a step of 2–4 sessions in shuffled slots, against each session's rows run alone. They
  compare outputs, window writes, compressed entries and keys, carries, the top-k and the pool.

**dsk and prefill.cu (the kernels agent).**
- Nothing changes up to 8 rows. `moe_routed` already takes rows routed anywhere, up to `kMaxUnion` =
  48 experts; the bench's rotated rows ran it at 35 distinct experts per layer.
- An optional look: the MoE at 20–35 distinct experts per layer. A row that shares no experts cost
  4.8–5.6 ms there, against 2.9–4.3 ms for a verify row.
- B4 raises `dsk::kMaxRows` to 16: the MoE's union to 96 experts, its row masks, the GEMVs' row tiling.
- prefill.cu needs no change. The runner points the eager paths at the session's slot before a chunk
  (`use_slot`, below), and the chunk size is the scheduler's.

**model, cmd and serve (the runner).**
- `Options::slots` (1 by default; serve takes 4).
  - Every per-session buffer becomes an array of slots: the 40 window rings, the four sources'
    compressed caches, keys and carries, and the draft module's windows.
  - That is ~240 MB a slot at 262k positions, ~0.7 GiB more for 4 slots than today's one. It is
    allocated before the session store, which then reserves that much less.
- `Model::use_slot(s)`: the eager paths act on slot s. Those are prefill chunks, state operations,
  snapshots and the golden modes; Impl's `ring_kv`, `kvs` and the draft windows point into the slot.
  Captured steps go through the slot table instead.
- `ops::StepIn` gains `slot[kMaxRows]`, `sess[kMaxRows]` (the row's session within the step) and
  `nsess`. The sampling parameters (seed, temperature, top-p, top-k, top logprobs) move from the
  step to `samp[kMaxSess]`, one per session. The sampler reads row r's from `samp[sess[r]]`, and its
  noise stays a function of (seed, position).
- Engram keeps a history per slot; a row hashes its own session's tokens, including the late lookups
  of a step queued before its tokens are known.
- cmd.h:
  - `kCmdBatch` carries an entry per session: slot, handle, first position, first token, tokens to
    sample, stop ids, sampling and the snapshot interval.
  - `batch_loop` runs on every rank alike, as `decode_loop` does.
  - Rank 0's control word: bit s ends slot s's session after the step that reads it (a cancel);
    bit 31 pauses the whole loop so that a new request can be prefilled.
  - A session that ends keeps its row in the step already queued, which only writes one more
    position, as a cancel does today.
- serve.cpp, the scheduler:
  - Admission is prefill-first, with at most `slots` sessions decoding.
  - A new request's chunks run between steps: the loop pauses, the chunk runs, and the loop resumes,
    with the new session once its first token is out. While others decode, chunks are 2,048 rows.
  - A session's slot is its live state: `live_` becomes a slot table, and persisting into the store
    happens only when a slot is wanted for another session.

**Checks, at every step:**
- Greedy hashes per session inside a batch, the same as alone: 1k febc7dc54ee40a39 and 16k CED
  81f772ab79a950e4, for each slot and in mixed batches.
- `serve_test.py` (exact and `--ced`) and `disk_test.py`.
- The soak with three and more clients: time to first token, decode per session, memory, no errors.

## The soak with four and six clients

B2 (engine 51c2dab0, `--slots 4`, serve's defaults) behind the release's dsfront, under soak.py's
mix: agent turns, regenerates, cancels, forks, disconnects, long sessions, JSON and golden replays
(2026-10-12). Two parts of 6 minutes, each from the engine's start to a planned stop inside
ring.sh's 10 minutes: 4 clients, then 6.

| | 4 clients | 6 clients |
|---|---|---|
| requests, failed | 248, 0 | 251, 0 |
| batch runs, by sessions at their start | 91 of 4, 1 of 3 | 150 of 4, 34 of 3, 5 of 2 |
| rows a batch step, on average | 3.12 | 3.38 |
| a batch step, on average | 27.7 ms | 29.0 ms (runs of 4) |
| decoded in batches | 20,515 tokens, 112 tok/s in all | 20,425 tokens, 116 tok/s |
| a request's decode (engine, ≥ 32 tokens, by 2-minute window) | 31.8, 26.8, 31.3 tok/s | 32.1, 31.7, 31.2 tok/s |
| first token of a turn adding < 1k (front-end, median by window) | 0.59, 0.47, 0.39 s | 0.67, 0.64, 0.41 s |
| the same, p90 | 1.6–2.6 s | 1.5–2.5 s |
| prefill of turns adding ≥ 4k | 7.9k–8.1k tok/s | 7.9k–8.2k tok/s |
| the engine's time: prefill, decode in batches | 172 s, 182 s | 183 s, 175 s |
| the store's peak, evictions | 3.9 GiB, 0 | 4.7 GiB, 0 |
| regenerates identical (greedy) | 11 of 12 | 16 of 16 |

- Memory stayed flat on every rank and in the front-end. Each rank's device memory peaked at
  93,020 MiB, and resident sets grew only by the Engram cache (to 0.35–1.1 GiB a rank). The
  front-end's table was still filling (61 and 70 sessions).
- The golden replay gave the same 13 answers in both parts (f5459970df999).
- The regenerate that differed resumed from a snapshot at 2,048 the first time and from 2,732 the
  second: CED's known effect ([engine.md, Soak](engine.md#soak-2026-10-12)).
- The engine was busy ~99 % of each part, and prefill took 49 % and 51 % of that. The soak before B1
  (three clients for an hour, one request at a time with drafts) decoded 107 tok/s in all, and a small
  turn's first token came in 3.0–3.6 s (median). The runs and mixes differ.
- **The verdicts.**
  - soak.py judged both parts FAIL on its eviction rule: the store must evict. In 6 minutes it
    reached only 3.9 and 4.7 of its 11.25 GiB. soak.py now judges evictions only when the store
    passes 90 % of its budget (`--store-gib`). With that, `--report` passes the 6-client part.
  - The 4-client part fails one more rule: the small turns' prefill, 231.7 ms in the first two
    windows and 257.9 in the last two (+11.3 %). That is the mix, not drift. Turns adding under 300
    tokens take ~30 ms, and those adding 300–999 take 270–290 ms. The last window had more of the
    larger ones: 26 of 36, against 13 of 35 before. By size, the windows were flat: 29.1, 30.9 and
    30.6 ms under 300 tokens; 292, 270 and 275 ms from 300 to 999.

## B3 from measurements

[Estimates](#estimates) modelled B3. Before any of its code, it was estimated again from measured
step times and acceptance (2026-10-12), in three ring parts plus the attention agent's index sweep.
- **a:** the engine's steps at 1k, from 1 to 8 rows (`--mode bench`, engine 55ea9b9a: B2 with
  master's B4 GEMV/MoE half and the bench's `--bench-groups`, runner-b3est bb895b3). The kinds of
  rows were:
  - one session's consecutive rows;
  - rows of separate sessions, rotated as in [What a row costs](#what-a-row-costs);
  - S sessions' runs of 2–4 rows, from one text and rotated.

  Eager steps counted each kind's distinct experts per layer.
- **b:** bench_chain (the kernels agent's, built at bb895b3 too: ee2b1f17): the step's kernels at
  1–16 rows and 6–86 distinct experts per layer.
- **c:** serve (engine 51c2dab0) with bench.py's four decode prompts, 512 tokens each, the gate capped
  at d = 1–5 drafts. It logged tokens, verify rows and GPU time per step, and the rate.
- **The index sweep:** the dense index scans per step at 16k, 128k and 250k, for 1–16 rows of 1, 2
  or 4 sessions.

### Step time by rows

GPU medians at 1k in ms, with the distinct experts per layer in brackets:

| rows | one session's rows | separate sessions, rotated | runs of S sessions: one text / rotated | B2 in serve | bench_chain |
|---|---|---|---|---|---|
| 1 | 18.66 (6.0) | 18.74 (6.0) | | 17.9 | 18.02 (6) |
| 2 | 23.02 (9.8) | 24.05 (11.9) | | 23.2 | 22.97 (12) |
| 3 | 26.59 (13.9) | 29.47 (17.7) | | 27.6 | 28.20 (18) |
| 4 | 30.37 (16.3) | 34.63 (23.6) | 2 × 2: 28.24 (16.1) / 31.15 (18.5) | 33.6 | 27.01, 30.22, 33.76 (16, 20, 24) |
| 6 | 37.65 (21.8) | 44.73 (34.6) | 3 × 2: 33.52 (20.1) / 39.14 (28.2); 2 × 3: 33.50 (21.6) / 37.37 (24.8) | | 32.21, 37.73, 42.78 (21, 28, 34) |
| 8 | 42.03 (27.3) | 55.51 (45.6) | 4 × 2: 38.26 (25.3) / 48.12 (36.3); 2 × 4: 38.94 (26.2) / 43.45 (32.1) | | 38.07, 42.18, 47.86, 55.29 (27, 32, 39, 48) |
| 12 | | | | | 54.39, 59.97, 65.18, 74.28 (42, 49, 55, 66) |
| 16 | | | | | 68.00, 76.02, 84.50, 93.70 (55, 65, 75, 86) |

- **The model.** Over the engine's 31 steps, a step costs:
  - 13.47 ms;
  - plus 0.825 ms per distinct expert per layer: the MoE streams 40 layers' 4.7 MB slices at
    ~228 GB/s;
  - plus 0.463 ms per row;
  - plus 0.179 ms per row after its session's first.

  The rms error is 0.71 ms, and the worst is +2.5 ms (six rows of one session).
- **Past 8 rows** the kernels take the rows in two groups of 8, and bench_chain rises above the model.
  Its excess there, less its excess at 1–8 rows (−1.4 ms), is 1.6 ms at 9 rows, 1.7 at 10, 2.1 at 12,
  2.6 at 15 and 3.1 at 16. The estimate adds that past 8 rows, for each step's own row count. The
  engine runs at most 8 rows until B4's runner half.
- **Sessions share experts.** Runs of one text reached 0.80, 0.68 and 0.65 of the sum of the runs' own
  distinct experts (2, 3 and 4 sessions). Rotated, they reached 0.94, 0.96 and 0.93.
  - The soak's batch steps ([above](#the-soak-with-four-and-six-clients)) sit between: 27.7 ms at
    3.12 rows on average. The model gives 25.4 ms at one text's overlap and 29.7 rotated.
  - So agent sessions share some experts, but less than one text's runs do.
- **Long context** lengthens only the dense index scans ([engine.md, Long
  contexts](engine.md#long-contexts)). Per step:

  | rows (sessions) | 16k | 128k | 250k |
  |---|---|---|---|
  | 1 (1) | 0.15 ms | 0.25 ms | 0.32 ms |
  | 4 (1 / 4) | 0.18 / 0.25 | 0.34 / 0.52 | 0.44 / 0.83 |
  | 8 (1 / 4) | 0.22 / 0.28 | 0.48 / 0.62 | 0.68 / 0.98 |
  | 16 (1 / 2 / 4) | 0.28 / 0.31 / 0.33 | 0.66 / 0.75 / 0.81 | 1.01 / 1.09 / 1.29 |

  Added to B1's, B2's and B3's steps, they move each comparison below by at most a point. For
  example, agentic with four sessions gains +18 … +31 % at 1k and +19 … +32 % at 128k and 250k.

### Tokens per step

One session alone in serve, the gate capped at d drafts. Each cell gives the tokens per step, the
verify rows per step in brackets, and bench.py's rate:

| content | d = 1 | d = 2 | d = 3 | d = 4 | d = 5 |
|---|---|---|---|---|---|
| prose | 1.57 (1.93), 65.1 tok/s | 1.76 (2.34), 69.3 | 1.80 (2.42), 70.2 | 1.81 (2.43), 70.3 | 1.81 (2.43), 70.6 |
| code | 1.83 (2.00), 74.7 | 2.44 (2.88), 88.7 | 2.88 (3.46), 98.4 | 3.13 (3.86), 102.6 | 3.19 (4.03), 103.3 |
| agentic | 1.83 (2.00), 74.6 | 2.43 (2.84), 88.4 | 2.77 (3.38), 95.0 | 2.98 (3.67), 98.9 | 3.11 (3.91), 101.0 |
| JSON | 1.95 (2.00), 79.1 | 2.79 (2.97), 99.3 | 3.62 (3.87), 116.7 | 4.40 (4.68), 132.0 | 4.90 (5.43), 137.4 |

- At d = 5 they match DSpark's measured 1.81, 3.17, 3.09 and 4.90 tokens per step, to within 0.02.
- In a B3 step, each session keeps its own drafter and gate. So each is assumed to accept as it does
  alone at the same cap.
- The rows are capped at 16: d ≤ 5 with two sessions, d ≤ 4 with three and d ≤ 3 with four. A
  session's own rows stay within `kMaxSessRows` = 8.

### Tokens per second, all sessions

B3's step for S sessions at cap d is the model's step at S times a session's mean rows. Its terms:
- the distinct experts: the overlap above, times S, times a session's mean distinct experts (over its
  gate's histogram of verify rows);
- past 8 rows, the increment above, averaged over the step's row counts (the sessions' histograms,
  taken as independent);
- the draft pass for S anchors: between one pass of 2.55 ms (one anchor's today, the floor for a pass
  over all anchors) and S passes (the drafter run once per session, no new kernel);
- the host's part of a step, 0.1–0.4 ms (B1's wall time per step less its GPU time).

The baselines:
- B1 is bench.py's rate at d = 5, one session at a time, the sessions taking turns.
- B2 is serve's medians: 86.2, 108.7 and 119.0 tok/s at 2–4 sessions.

**Agentic.** "Rotated" means sessions that share no experts, compared with the better of B1 and B2.
"One text" means the overlap of one text's runs, compared with B2 at that overlap (88, 118 and 141
tok/s by the model). Each range runs from S draft passes to one.

| sessions | d | rows | experts per layer: rotated / one text | step, rotated | tok/s, rotated | against the better of B1, B2 | tok/s, one text | per session, rotated |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 4.00 | 18.4 / 15.7 | 33.8–36.3 ms | 101–108 | 0 … +7 % | 107–116 | 50–54 |
| 2 | 2 | 5.68 | 24.9 / 21.2 | 40.2–42.7 | 114–121 | +13 … +20 % | 123–131 | 57–60 |
| 2 | 3 | 6.75 | 27.2 / 23.2 | 42.8–45.4 | 122–129 | +21 … +28 % | 132–140 | 61–65 |
| 2 | 4 | 7.33 | 29.8 / 25.4 | 45.7–48.3 | 124–130 | +22 … +29 % | 134–142 | 62–65 |
| 2 | 5 | 7.82 | 30.2 / 25.8 | 46.6–49.2 | 127–133 | +25 … +32 % | 137–145 | 63–67 |
| 3 | 1 | 6.00 | 28.2 / 20.1 | 42.9–48.0 | 114–128 | +5 … +18 % | 133–151 | 38–43 |
| 3 | 2 | 8.51 | 38.1 / 27.2 | 53.7–58.8 | 124–136 | +14 … +25 % | 147–163 | 41–45 |
| 3 | 3 | 10.12 | 41.7 / 29.7 | 58.3–63.4 | 131–143 | +21 … +31 % | 155–172 | 44–48 |
| 3 | 4 | 11.00 | 45.6 / 32.5 | 62.3–67.4 | 133–144 | +22 … +32 % | 158–174 | 44–48 |
| 4 | 1 | 8.00 | 36.3 / 25.3 | 50.7–58.4 | 125–144 | +5 … +21 % | 149–176 | 31–36 |
| 4 | 2 | 11.35 | 49.0 / 34.2 | 65.3–73.0 | 133–149 | +12 … +25 % | 160–183 | 33–37 |
| 4 | 3 | 13.50 | 53.6 / 37.4 | 70.9–78.6 | 141–156 | +18 … +31 % | 170–193 | 35–39 |

**Every content, at the best d** (rotated, against the better of B1 and B2):

| content | B1 | 2 sessions | 3 sessions | 4 sessions |
|---|---|---|---|---|
| prose | 70.6 tok/s | +7 … +15 % | −5 … +5 % | −8 … +7 % |
| code | 103.3 | +23 … +30 % | +25 … +35 % | +22 … +35 % |
| agentic | 101.0 | +25 … +32 % | +22 … +32 % | +18 … +31 % |
| JSON | 137.4 | +21 … +27 % | +22 … +31 % | +24 … +37 % |
| agentic, one text's overlap (against B2 at that overlap) | | +35 … +43 % | +34 … +47 % | +21 … +37 % |
| agentic, at most 8 rows | | +21 … +28 % (d = 3) | +5 … +18 % (d = 1) | +5 … +21 % (d = 1) |

The 1k golden text decodes 108.5 tok/s alone with drafts (9.22 ms per token), between agentic and
JSON here. Each content is compared with its own B1.

### The verdict

**Against the 10 % bar, B3 clears it.** With 2–4 agent sessions decoding at once, it decodes 25–32 %,
22–32 % and 18–31 % more in all than the better of B1 and B2. That holds even when sessions share no
experts. Code and JSON gain as much. Prose does not: its drafts carry only 0.8 tokens a step.
- **The worry was 16 rows reading 86 experts.** B3's steps hold fewer.
  - At four sessions and d = 3, the gate verifies 2.4 drafts a session on average, so a step has
    13.5 rows.
  - A session's drafts share its experts: 53.6 distinct per layer when sessions share none, and 37.4
    at one text's overlap.
  - Such a step takes 71–79 ms and yields 11.1 tokens. B2's takes 33.6 ms for 4.
- **These figures lean low.**
  - The model is fitted on bench steps over the golden text. It puts one session's steps in serve
    0.8–1.5 ms above what serve measured: 31.6 ms against 30.5 for agentic at d = 5.
  - B1's switches (~4 % with 64-token turns) are not charged to B1.
- **Past 8 rows needs B4's runner half** (step buffers and graphs to 16 rows). Within 8 rows, two
  sessions still gain 21–28 % (d = 3), but three or four gain only 5–21 % (d = 1).
- **Prefill is outside B3.** Where prefill takes half the engine's time, as in both soaks, a decode
  gain of 18–32 % is 8–14 % end to end.

**A session's own speed while others decode** (agentic at the best d; no shared experts … one text's
overlap):
- 63–72 tok/s with one other session, 44–58 with two and 35–48 with three, against 101 alone.
- That is more than B2 gives it (43, 36 and 30 tok/s, no drafts). It is also more than B1's turns
  average (101 / S: 50, 34 and 25 tok/s, in bursts).
- A new request's prefill waits for the step in flight: up to 47–79 ms in B3, against 23–34 in B2.
- Every row computes as it would alone, and drafts never change tokens. So outputs stay
  bit-identical to each session's run alone, tokens and log-probabilities, as in B2.

**Against the owner's use, B3 is not worth building now.** The owner runs several agents at once only
occasionally (2026-10-12), so one session is the common case. The coordinator's test: build B3 only
if it gains a lot when sessions overlap and costs nothing for one session.
- **One session:** built as B2 was, B3 costs it nothing. A lone session leaves the batch and runs
  today's loop, and B2's ABBA found no change at k = 1. Its only cost is the store's ~0.24 GiB per
  extra slot.
- **Sessions overlapping:** the gain is real, but moderate, and only in those minutes. With two
  sessions decode gains 25–32 %, which is 11–14 % end to end where prefill takes half the engine's
  time. B1 already gives the fairness and first tokens.

So:
- B3 is parked.
- B4's runner half is not wired for B3's sake.
- Production stays on `--slots 1`. With two sessions, B2 decodes 86 tok/s in all, against B1's 101.

If several agents at once become common, the first step is B3 for two sessions within 8 rows
(d ≤ 3): 21–28 % more, with no runner work past 8 rows.
That came on 2026-10-13: [B3: the build](#b3-the-build-2026-10-13).

## B3: the build (2026-10-13)

The owner asked for it on 2026-10-13: "Optimise for small multi-stream (up to 4). I want to keep single
stream as fast as well. Be careful if single stream suffers more than 2%." So B3 is built, for two to
four sessions, under two bars:
- **One session's speed.** The project's bar stays: the 1k golden run at k = 1 and with drafts, and
  bench.py's prompts, within the run-to-run spread of master in an A/B run in both orders (ABBA). The
  owner's limit is 2 %: a build that costs a lone session more than that is not merged, whatever it
  gains for several.
- **Outputs.** Bit-identical to each session's run alone, tokens and log-probabilities, in every batch,
  and every reference hash ([golden/README.md](../golden/README.md)).

### Design

1. **A path of its own.** The one-session speculative path (`spec_graph`, `k_spec_in`, `k_spec_next`,
   `launch_spec` / `launch_verify`, `spec_loop`) stays as it is, and a lone session never runs the
   batch code: that is how one stream keeps its speed. The batch path adds `spec_batch_graph(ctx)`,
   `launch_spec_batch` / `launch_verify_batch` (the model) and `spec_batch_loop` (cmd.cpp). serve's
   `decode_batch` takes it when drafts are on and the budget gives every session at least one draft;
   else B2's plain batch. When one session is left decoding, it pauses and goes on alone with drafts,
   as B2's `kBatchSolo` does.
2. **Spec state per entry.** `SpecState ss[kMaxSess]` on the device, one per batch entry, and each
   entry's first row in the step (`row0[kMaxSess]`, device, written by the step's `k_spec_in_batch`).
   `StepIn` gains, per entry: whether it is in the step (`sess_on`, host-written for every launch; an
   entry that ended is off from the step after the one already queued), its slot (`sess_slot`), its
   gate's threshold and cap (`sthresh`, `scap`). A verify step's rows are laid out on the device: the
   entries on, in order, each its `next_rows` rows (its bonus token, then its drafts), with `slot[r]`
   and `sess[r]` per row; the switch takes the total.
3. **The forward.** The spec switch's bodies, unchanged: `forward_rows` reads the slot table from
   `StepIn`, as B2's batch steps do. A session's rows stay within `kMaxSessRows`.
4. **Verdicts per entry** (`k_sample`). Entry k's rows are [row0, row0 + n): row row0 is final, and row
   row0 + j + 1 is accepted while its token is row row0 + j's sample; the entry's count, pos0 and bonus
   go to `ss[k]`. `SampleOut` keeps `row[r]` as row r's sample (today's layout) and adds, per entry,
   the rows it emits (`sess_rows`), its first row (`sess_row0`), and the next step's rows, pos0 and
   tokens.
5. **The draft passes.** The accept of each entry on, of its final rows from its row0 (`Draft::accept`
   takes a device row offset) into its own slot's windows, under a conditional node of the graph
   (`sess_on[k]`) with a per-entry slot table (device, host-filled when the loop starts); then one draft
   pass over the entries on (phase 2a, `Draft::draft_batch`): a switch over their number S, body S − 1
   running passes of at most three entries (15 rows within the kernels' 16) over a device-side list of
   the entries on, each entry's five rows from its bonus token in its slot, the weights read once a
   pass. Every row computes as in a pass of its own (the kernels' per-row contract,
   [kernels § 9–16 rows](../engine/kernels/README.md#916-rows-b4); the entries' Markov chains and
   confidence per entry), so the drafts are the same bits as alone; `--bgraph-check` compares them on
   the device. The experts' reads shrink only where rows share an expert.
6. **The gate.** Per entry, today's gates (thresh, ratio, marginal) under its cap. The caps come from
   the budget: cap = min(5, (kMaxRows − S) / S) with S entries on. The marginal gate across sessions
   (the batch's tokens per ms against a row's marginal cost) is phase 2.
7. **Engram.** Late lookups per entry, each in its own slot, as `engram_late_batch` does for B2.
8. **The loop** (`spec_batch_loop`). Per entry: the tokens it emits, its stop ids, max_new, its slot's
   cancel bit, `kCtlPause`, the context's end. An entry that ends keeps its rows in the step already
   queued (their tokens dropped, their KV written past its end, as today), and is off from the step
   after. `SpecRate` per entry; `Flight` carries it across runs, as it does for slices.
9. **The budget** is `kMaxRows`: 8 today, 16 with B4's runner half.

### B4's runner half

`kMaxRows` 16 (config.h) with `kMaxSessRows` still 8: the step buffers (activations, split-K partials,
the gathered logits), graphs for 9–16 rows, the `StepIn` / `SampleOut` rings, the sampler's
candidates, the Engram segments. The attention and kernel halves are on master. The bench mode covers
9–16 rows as runs of 2–4 sessions (`--bench-groups`); a step with more than 8 rows of one session is
refused.

### Checks

- The reference hashes in batches with drafts (`--mode batch --spec`): 1k febc7dc54ee40a39, 16k CED
  81f772ab79a950e4 and sampled a8f7a08f8cd1c5e8, in every slot and mixed.
- batch_test.py with drafts (greedy with top logprobs, sampled, staggered, cancels, stop ids, kept
  sessions): 0 failed. serve_test `--ced` 17 of 17, slice_test, disk_test, kept_test.
- Throughput at 2, 3 and 4 sessions, bench.py's four contents, against B1 (101 tok/s agentic) and B2
  (86.2 / 108.7 / 119.0 in all).
- One session, ABBA against master: the 1k golden run at k = 1 and with drafts (ms per token) and
  bench.py's prompts. Within the spread, and never more than 2 % slower.

### Where it is

- 2026-10-13 14:10: the design above; the runner builds B3 and B4's runner half on two branches, with no
  ring time until both are built (the kit serves an agent; its interruptions come last).
- 2026-10-13, B4's runner half (branch runner-b4rows, built, untested on the ring): `kMaxRows` 16 with
  `kMaxSessRows` 8. Everything sized by `kMaxRows` follows it (the step buffers, the `StepIn` / `SampleOut`
  rings, the sampler's candidates, the Engram segments, `Cmd::tokens`, the graphs' `warm`); the runner's own
  kernels whose launch holds the rows come in two instances by a row budget of 8 or 16 (`ops::row_budget`:
  the sampler's grid and posted bytes, `allreduce_hc`'s and `mix_allreduce`'s block sizes), so a step of at
  most 8 rows launches the kernels it launched before, and the one-session spec graph keeps its 8 bodies
  (serve's small turns run spec steps of up to `--small-rows` = 8 rows: no change to its startup or memory).
  A step with more than 8 rows of one session is refused with a message (`next_in`, `launch_batch`,
  `launch_spec`; main.cpp's `--chunk`, `--k`, `--small-rows`, `--batch-rows`, `--route-every`, spectf's
  chunks). New: `kCmdBatchTf` / `batch_tf_loop` (teacher-forced rows of several sessions, G of each per
  step; `Batch::flags` / `sink` for an eager dumped step), `--mode bench` at 9–16 rows as runs of 2–4
  sessions in slots (`--bench-groups`, `--slots`), `--mode batch --batch-tf [--batch-rows G]` whose
  per-session hash must equal the session's run alone and tf mode's. Untested: every GPU run (the
  equalities above, the ABBA at one session, the 12- and 16-row step times).
- 2026-10-13 15:10, branch `runner-b3`: built, warning-free, not yet run on the ring. The model: the
  batch spec step graph (`spec_batch_graph`: `k_spec_in_batch` lays the entries' rows out on the device
  and sets the switch's and the ifs' values, the spec switch's bodies, `k_sample`'s verdicts per entry,
  one if node per entry with its accept (from its `row0`), draft and gate in its own slot table, then
  `k_spec_publish_batch`), `SpecState[kMaxSess]`, `SpecBatch`, `launch_spec_batch` /
  `launch_verify_batch` / `engram_late_rows`; `spec_graphs` captures the batch graphs too and prints
  their own line. The loop: `spec_batch_loop` (`kCmdSpecBatch`; cmd.h has its semantics), serve's
  `decode_batch` taking it for batches of at most kMaxRows / 2 with the drafts' figures on the batch log
  line, `--mode batch --spec` for the reference hashes in drafted batches. Checks written:
  batch_test.py's scenarios 7–8 (pairs, triples, a pause, a cancel and a stop id in drafted batches),
  bench/multistream.py, bench/abba.sh. The layout and the verdicts per entry live in
  [spec_batch.h](../engine/model/spec_batch.h), one code for the device and the host:
  `spec_batch_test` (host only, no GPU) passes, 656k checks over 20k random steps against the
  one-session rule per entry. **Untested on a GPU**: everything else, first of all the batch graph's
  capture (if nodes whose bodies hold the draft module's collectives), then the reference hashes in
  drafted batches, batch_test with drafts, the throughput at 2–4 sessions and the one-session ABBA.
  Two choices to note: a paused entry's last token is not fed by an extra step (its next run feeds it,
  as B2's pause does), and the ahead launch waits for the step before when any entry's worst-case rows
  would pass its `feed` (the speculative loop's rule, applied to the batch as a whole).
- 2026-10-13 15:25, the integration branch `b3-int` (B4's runner half merged first, then B3; one closing brace
  between `batch_tf_loop` and `spec_batch_loop` added by hand). On the ring, preemptible, the owner's agent
  paused: the merged engine gives every reference hash (tf 842278b28d87dad0, k1 / nol2pf / fp8 / bf16
  febc7dc54ee40a39, temp a8f7a08f8cd1c5e8, ced16 81f772ab79a950e4, ced28 8f08a153d415c5c8; golden/refhash.sh
  8 of 8, 52–65 s a run), so one session computes as before. The first drafted batch died at the batch
  graph's capture: `dspark::main_hidden` refused more than `kMaxAcc` = 8 rows, while the batch graph's bodies
  go to 16; it now takes `kMaxHidden` = 16 (accept still reads at most 8, from its entry's row0). The
  drafted-batch hashes, the one-session ABBA and the serve tests follow.
- 2026-10-13 15:35, the second fault, found by the next run: `cudaGraphInstantiate` refused the batch graph
  (invalid argument) because a conditional handle was created for every `kMaxSess` entry while only `bents`
  if nodes used one. One handle per node (ee748c1), and the graph of 4 entries instantiates: 14,584 nodes in
  its bodies, 483–490 MiB of device memory, 0.29 s. Then the checks below passed.
- 2026-10-13, branch `runner-b3p2`, **phase 2a, the plan** (one draft pass over the entries of a batch step;
  written before the code). The measurement above says a drafted batch step pays about 3 ms per further entry
  for its own draft pass, accept and gate, each in its own if node. The change: the accepts stay per entry
  (each writes its entry's accepted rows into its slot's windows, and every draft row reads them), then one
  pass over all the entries on, then the gates. Where it goes, and why every row keeps its bits:
  - `engine/dspark/dspark.h`, `dspark.cu`: `Draft::draft_batch(const BatchPass&, stream)`, an entry point
    beside `draft()`, which is not changed. Rows [5q, 5q + 5) of a pass are entry `ent[q]`'s (q < n): its
    token `tok[ent[q] * tok_stride]` (entry k's bonus in `ss[k]`), its anchor `start[ent[q]]` (`accept()`
    gains an optional `start_out`, the entry's anchor slot; null, today's one-session `b.start`), and the
    rows' slot table (device). Its results per entry: `batch_ids(k)`, `batch_survival(k)` and the
    confidence logits. The pass has buffers of its own (`enable_batch()`, about 10 MB, sized for
    `kBatchRows` = 15 rows), so `draft()`, `accept()` and the one-session arena are as they were and the
    one-session graph captures the launches it captured. A pass takes at most `kBatchEnts` = 3 entries:
    15 rows, within the kernels' 16 (`dsk::kMaxRows`, `attn::kMaxDecodeRows`); S entries on run
    ceil(S / 3) passes one after the other (S = 4: a pass of 15 rows, then one of 5).
  - The pass's kernels, each over rows = 5 n, and why a row's bits are `draft()`'s for it alone:
    - `draft_prep` (dspark): a block per row: its entry's token (row 0) or the noise token, embedded and
      copied four times; its positions and window bounds from its entry's anchor; the identity pre-mix.
    - `hc_project`, `hc_mix` (attn, per sublayer): a row's 24 dot products are its own lane-strided FMA
      chain and warp sum per chunk (rows in groups of 8 share a CTA, not an accumulator); `hc_mix` is a
      CTA per row: its chunks summed in chunk order, its Sinkhorn, pre-mix, norm and act_quant.
    - the GEMVs (kernels): `gemv_fp8_multi` (wq_a + wkv), `gemv_fp8` (wq_b, wo_b), `gemv_woa` (wo_a with
      its act_quant), `gemv_bf16` (the router gate), the head's `gemv_fp8w_f32` (`--draft-fp8`),
      `gemv_bf16x` or `gemv_bf16`. An activation row is one lane group of the MMA tiles; an output's k
      order, its split-K slices (`wk`) and the fold over the slices come from the launch config, not from
      M. At 10 and 15 rows the `_r16` builds run (rows 8-15 in the upper half of the same MMAs), whose
      contract is each row's bits of an 8-row call (engine/kernels/README.md, "9-16 rows"; test_kernels
      `rows16`; B4's batch steps at 12-16 rows give the reference hashes). The persistent kernels run at
      one row only (`persist_rows()`), in neither path. The MXFP4 head (`--draft-head4`,
      `gemv_fp4w_f32`, at most 8 rows) runs per entry: today's call on its 5 rows.
    - `decode_prep` (attn): a CTA per row: q_norm and act_quant of its qa, kv_write of its key at its
      position into its slot's window (the slot table; from 9 rows the kernel requires one).
    - `sparse_attn` (attn): grid (3 splits, rows): each row's online softmax over its own window
      [max(0, anchor - 127), anchor + 5] in block order, its splits merged in split order by the second
      launch.
    - `hc_post`, `hc_pre` (attn): per row.
    - `router3` (dspark): a warp per row, a block per entry.
    - `dsk::moe`: the union of the rows' picks says which experts run; a row's output is the sum over its
      own picks in expert-id order, then the shared expert, whatever the other rows picked; from 9 rows
      the `_r16` builds (every up16 x down16 pair is checked against 8-row calls in test_kernels). The
      experts a pass reads are the union over its rows, up to 3 a row: the pass shares the dense weights
      (the blocks' projections, the head, the Markov head), the launches and the collectives, not the
      experts.
    - `markov` (dspark): grid (505 CTAs, n entries). Entry q's CTAs compute its row j's bias and biased
      logits with the lane arithmetic of today, and reduce their candidates through entry q's own counter
      and buffer in today's order (the reduction is a max over a total order, the value then the lower
      id, so any order gives the same candidate; the order is kept anyway). The gathered argmax takes
      the n candidates in one `comm::argmax_rows` (per row the best of the four ranks; rows are
      independent) instead of n calls.
    - `conf` (dspark): a block per entry, a warp per row: its dot product, its survival products in order.
    - The collectives: the all-gathers carry 5 n rows per segment (rank order per row, as today's 5), the
      two all-reduces per block 5 n x 5120 (an fp32 sum in rank order per element). `RingComm` takes up
      to 15 x 5120 (`f32_` sized for it) and gets `argmax_rows` for several candidates. A step's
      collectives drop from 16 per entry to 16 per pass.
  - `engine/model/spec_batch.h`: `spec_batch_passes(StepIn, bent, bstab)`: the entries on in entry order
    (the compact list) and their rows' slot tables for the passes; host and device, checked by
    `spec_batch_test` against a host reference (a new check).
  - `engine/model/spec.cu`: `k_spec_in_batch` also writes the compact list and the pass tables and sets a
    switch over the entries on (one more handle, used by its node). `spec_batch_graph`: the if nodes keep
    the accepts (each from its `row0` into its slot table, its anchor into `batch_start() + k`); then a
    switch node over S whose body S - 1 runs the passes over the first S of the compact list (the number
    of entries on is the device's, so the body for any S exists); then one gate kernel over the entries
    (`k_spec_gate_batch_all`: a thread per entry on, today's three gates under its cap, into its
    SpecState and the step's SampleOut); then the publication. Still one graph per context bucket.
    DIAGNOSTICS, `--bgraph-check`: after the pass, every entry's `draft()` as today (its accept once more,
    so that its anchor is in the one-session buffer) and a compare kernel counting the words of its ids
    and confidence that differ from the pass's, printed by `--mode batch`
    (`Model::spec_batch_mismatches()`): the bit check of the pass on the ring. `--bgraph-noif` and
    `--bgraph-bodies` stay.
  - `engine/server/cmd.cpp`: nothing; the protocol (ranks follow rank 0, one step queued ahead, ended
    entries keep their rows in the queued step), the batch log line and `SpecBatchStats` stay. `main.cpp`:
    the `--bgraph-check` flag and its line.
  - Expected: 3, 6 and 9 ms of a 41, 57 and 67 ms step at 2, 3 and 4 streams, less the experts' share,
    which the pass does not share; four streams run two passes (15 rows, then 5).
  - To check on the ring, in this order: the batch graph instantiates (a switch inside it); `--bgraph-check`
    with 0 mismatches at 2 and 4 sessions, with `--draft-fp8` and without, greedy and sampled; the reference
    hashes in drafted batches (`--mode batch --golden G --big --slots 4 --spec --draft-fp8 --batch 1024,1024
    --steps 256` and `--batch 1024,1024,1024,1024`); batch_test.py `--slots 4`; bench/multistream.py; and
    the batch line's steps, rows and tokens a step equal to the table below (the same drafts give the same
    steps).
- 2026-10-13, B3 phase 2a built, not run (branch runner-b3p2 from b3-int e1072d0; commits fa317f0 the plan,
  e5f63d2 the draft module, 6c9096d the model layer and the flag; the whole engine builds with no warnings,
  spec_batch_test 0 failed over 20000 iterations; no ring time). What is there:
  - `engine/dspark`: `Draft::draft_batch(BatchPass, stream)` runs draft()'s launches over 5 n rows (n ≤ 3
    entries): `draft_prep_batch_kernel` (grid (5, n): row 5 q + j takes entry ent[q]'s token at j = 0 and
    the noise token after, its anchor from `batch_start()[ent[q]]`), `bmix` (hc_project and hc_mix over
    the rows; the chunk gather 1,000 B a row a rank), the GEMVs on 5 n rows (from 9 rows the `_r16`
    builds), `decode_prep` and `sparse_attn` with the pass's slot table, the MoE over the rows' union
    (from 9 rows up16 4 and down16 2, which test_kernels rows16 checks row by row against two 8-row calls
    at up 0 and down 4: draft()'s configuration), the FP4 head per entry (8 rows a call), the five Markov
    kernels with a grid over the entries (each entry its own CTA candidates and counter, its ids; the
    pass's candidates [5][4]) and one `argmax_rows` collective a step (an even count of candidates; the
    unused ones stay zero), `conf` per entry into `batch_ids(k)` [8] and `batch_confidence(k)` [16]
    (survival at + 8). The pass has an arena of its own (`enable_batch()`, about 10 MB); the one-session
    kernels are the `<false>` instances of the same templates (every offset folded to zero), and draft()
    does not change. `Comm::argmax_rows` (the default: `argmax` per row); `accept()` takes `start_out`.
  - `engine/model`: `spec_batch_passes` (spec_batch.h: the compact list and the slot tables, checked by
    spec_batch_test); `k_spec_in_batch` sets the switch over the entries on; the batch graph: the accepts
    under their if nodes (each with its anchor into `batch_start()[k]`), the switch (body S − 1 runs
    ceil(S / 3) passes), `k_spec_gate_batch_all` (a thread per entry on; the three gates and the caps as
    before), the publication. RingComm: `argmax_rows`, the all-reduce buffer at 15 rows. `--bgraph-check`:
    after the pass, every entry's accept and draft() as before under an if node of its own, and
    `k_spec_draft_check` counts the words of its ids, confidence and survival that differ
    (`Model::spec_batch_mismatches()`, the batch mode's last line). The loop, its protocol, the batch line
    and `SpecBatchStats` are not touched.
  - Not verified without a GPU: that the batch graph instantiates with the second switch and the check's
    if nodes (every handle has one node); the `_r16` GEMV builds at the drafter's shapes and the MoE at 10
    and 15 rows of the drafter's experts (the per-row contract is tested at the main model's shapes); the
    bit check itself; the hashes and the step times. The checks run in the plan's order above.

### Measured (2026-10-13, engine ee748c1 on the integration branch)

**Every reference hash in drafted batches** (`--mode batch --spec --draft-fp8 --slots 4`, 256 steps a
session, 128 sampled; the same text in every 1k session, so these sessions share experts):

| sessions (context) | hashes | steps, rows a step | tokens a step | ms a step | tok/s in all, per session |
|---|---|---|---|---|---|
| 1k, 1k | febc7dc54ee40a39 twice | 74, 8.30 | 6.92 | 37.0 | 187.1, 93.6 |
| 1k, 16k CED, 1k | febc7dc54ee40a39, 81f772ab79a950e4, febc7dc54ee40a39 | 88, 10.47 | 8.73 | 51.1 | 170.8, 56.9 |
| 1k ×4 | febc7dc54ee40a39 four times | 89, 13.66 | 11.51 | 44.3 | 259.7, 64.9 |
| sampled (T 0.8, top-p 0.95, top-k 40, seed 7): 1k, 4k, 1k | a8f7a08f8cd1c5e8 twice (the 4k one has no reference) | 54, 10.43 | 7.11 | 49.1 | 144.7, 48.2 |

- A step's GPU time by its rows, 4 sessions of one text: 8 rows 32.8 ms (16 steps), 12 rows 40.4 (7), 16
  rows 47.5 (60). Two sessions: 4 rows 25.9, 6 rows 30.1, 8 rows 33.2, 12 rows 44.6 (33 of 74 steps).
  Three sessions with the 16k one: 9–15 rows 49.5–62.4 ms.
- One session alone decodes this text at 8.29 ms a token with drafts (120.7 tok/s; the ABBA below), so two
  sessions of it decode 55 % more in all and four 115 % more.

**One session, A/B/B/A** (bench/abba.sh: master 4e2bae9 against the integration engine, the 1k golden run,
two runs each per variant, both orders; every run the reference hash):

| variant | master, ms a token | integration | B over A |
|---|---|---|---|
| k = 1 | 16.730 (16.677, 16.784) | 16.768 (16.743, 16.794) | +0.23 % |
| drafts (fp8) | 8.285 (8.270, 8.301) | 8.352 (8.313, 8.391) | +0.80 % |

Both within the owner's 2 %. The drafted figure was not clearly inside the run-to-run spread with two runs
each, so three more rounds of it ran after the serve tests (A B B A, A B B A, A B B A; six runs each):
master's median 8.322 ms a token (8.301–8.362), the integration engine's 8.322 (8.310–8.362), B over A
−0.01 %, every run the reference hash. One stream is unchanged.

**Serving** (`--mode serve --slots 4 --max-ctx 131072`): batch_test.py `--slots 4` 0 failed in its eight
scenarios, the two drafted ones included (pairs A+C and B+D, triples C+E+F and A+B+D, every request's
tokens and log-probabilities its run's alone; then a pause inside a drafted pair, E's first token 0.39 s
after its submit, 0.11 s later than alone, a cancel at its 40th token and a stop id); serve_test.py
`--ced` 17 of 17; slice_test.py 0 failed. Over that run, with batch_test's six prompts (different texts,
0.8k–4k positions), a step's GPU time by its rows: 3 rows 30.3 ms, 4: 33.8, 6: 39.9, 8: 46.1, 10: 58.4,
12: 66.2, 14: 75.6, 16: 82.0; its batches of two ran 47.3 ms a step at 7.97 rows and 6.39 tokens (76 % of
the drafts accepted), of three 54.9 ms at 9.37 rows and 7.28 tokens (71 %), of four 64.4 ms at 11.12 rows
and 8.57 tokens (68 %).

**Streams** ([bench/multistream.py](../bench/multistream.py) `--socket`: S prompts of 1,024 tokens cut from
the synthetic 270k text at different offsets, submitted at once, 512 new tokens each, greedy; two
repetitions, which agreed to 0.1 tok/s; the step figures from the engine's log):

| streams | tok/s in all, from the first submit to the last end | decoding, from the first token | each stream, tok/s | a step: ms (median), tokens, rows | drafts accepted |
|---|---|---|---|---|---|
| 1 | 97.3 | 102.7 | 102.7 | the speculative loop alone | |
| 2 | 106.9 | 110.0 | 65.0, 56.8 | 41.2–41.4, 4.93, 6.51 | 65 % |
| 3 | 121.9 | 124.4 | 46.8, 42.5, 49.5 | 56.7–56.9, 7.68, 9.69 | 71 % |
| 4 | 131.3 | 133.5 | 36.0, 34.0, 37.6, 38.4 | 67.1–67.3, 9.78, 12.01 | 73 % |

The first tokens came 0.29 s apart (the prefills run one after another); the streams' ends spread
by 0.7–1.7 s as their drafts fared. Four different streams decode 35 % more in all than one alone, three
25 %, two 10 %; one stream at a time (B1) would give each of them the 102.7 of one alone, in turn.
Master's engine (B2: batches without drafts, 4e2bae9) on the same prompts and bench, the same day: one
stream 95.8–97.0 tok/s in all (102.4–102.5 decoding), two 87.7 (89.7; 21.5 ms a step), three 106.9–107.1
(108.8–109.0; 26.2 ms), four 121.4–121.6 (123.3–123.5; 30.5 ms). So drafts across sessions add 22 % at
two streams, 14 % at three and 8 % at four over B2, and at two streams the batch beats one stream at a
time for the first time (B2's pair decoded 10 % less in all than one stream alone).

**What a step costs.** A drafted batch's step is one session's step at the same number of rows plus
about 3 ms for every further session: one session's steps are 26.4 ms at 3 rows and 36.2 at 6 (3.3 ms a
row over 16.6), and 16.6 + 3.3 × rows + 3 × (S − 1) gives 41.1, 54.6 and 65.2 ms for the three batches
above against the measured 41.2, 56.8 and 67.2. The rows' cost is their experts: every row routes to its
own six, and the same text at the same positions in four slots (P2's 1k ×4) ran 16 rows in 47.5 ms where
four different texts take 82.0. The 3 ms a session is the drafter's pass, the accept and the gate, run
once per entry in its own conditional node ([design item 5](#design)). So the two items phase 2 named
are worth, at the measured rows: one draft pass over all anchors, 3, 6 and 9 ms of a 41, 57 and 67 ms
step (7, 11 and 13 %); and a gate across the sessions, which today verifies 2.3 drafts a session in a
pair whose cap is 5, judging each entry's drafts as its one-session gate would.

**Phase 2a, measured** (2026-10-13, engine 6da6160, branch `runner-b3p2`; the build is described under
[Where it is](#where-it-is)): one draft pass over up to three entries (15 rows), the accepts still per
entry. `--bgraph-check`, which runs each entry's one-session draft after the pass and counts the words of
the drafts and confidence that differ: 0 in the four drafted batches above (1k ×2, 1k ×4, 1k + 16k CED +
1k, the sampled trio), every hash as before, batch_test `--slots 4` 0 failed. The multistream bench, the
same steps and rows as phase 1 (the drafts are the same bits): two streams 39.9–40.1 ms a step (41.2–41.4
before), three 54.1 (56.7–56.9), four 64.5 (67.1–67.3): 109.4 / 126.6 / 135.5 tok/s in all (106.9 / 121.9
/ 131.3), one stream 96.8 (97.3). The same text in two and four slots: 35.2 and 40.4 ms a step (37.0,
44.3), 196.4 and 284.9 tok/s in all (187.1, 259.7). So the pass saves 1.3–2.7 ms a step with different
texts and up to 3.9 with one text, less than the 3 ms a further session that S separate passes cost: a
pass over 10 or 15 rows reads the union of its rows' drafter experts and runs the GEMVs' 16-row builds,
so it is not the 2.55 ms of one pass over 5.

**B4's runner half** (`--batch-tf`, [engine.md](engine.md#scheduling-time-slicing-and-batching)): a
session's 256 golden positions through batch steps of 1, 4 and 8 rows give the teacher-forced hash of
`--mode tf` (1k: cbf28c20d4a29e35; 2k: c612bb419cd6aeb4; 4k: d095975dd8969f11), and so do four sessions
(1k, 2k, 4k, 1k) at 4 rows each (16 rows a step, 57.8 ms median), two (1k, 2k) at 8 rows each (16 rows,
63.0 ms) and three (1k, 2k, 4k) at 4 rows each (12 rows, 54.5 ms): every session its own hash. One
session's steps: 16.76 ms at 1 row, 27.5 at 4, 37.4 at 8. `--mode bench --ks 8,12,16 --bench-groups 4,8`
(context 1k, 40 steps a k): 8 rows as two sessions of 4, 35.6 ms (26.1 experts a layer; the sessions'
routing rotated, 39.5); 12 rows as three of 4, 53.1 (39.4 experts; rotated 62.6); 16 rows as four of 4,
62.6 (50.9; rotated 78.0) and as two of 8, 62.2 (rotated 70.3); one session's 8 consecutive rows 38.1.

**Everything else as before.** The final integration engine (ee748c1) gives every reference hash alone
(tf, k = 1, the drafted variants, sampled, 16k CED, 28k CED: 8 of 8), as it did before the two fixes.
