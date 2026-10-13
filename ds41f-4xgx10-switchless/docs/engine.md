# The TP=4 engine (engine/model, engine/server)

`dsgb10_rank` runs DeepSeek-V4.1-Flash on the four boxes of the ring: one process per box, each
holding a quarter of every layer ([layout.md](layout.md)), all four running the same steps in the
same order. This page covers how the processes run, what each rank holds, what a step does, how to
start and stop it, and how it is checked against the reference. Every measurement on it ran the
weights production serves, dealignai's variant of the checkpoint ([README](../README.md)); deepseek-ai's
official weights have not been run through the engine.

The current figures are in [Where it stands on release candidate 5](#where-it-stands-on-release-candidate-5-2026-10-13)
(2026-10-13); the status below is as of 2026-10-12.

**Status (2026-10-12).** The engine serves: `--mode serve` speaks [protocol.md](protocol.md) to the
front-end (`dsfront`), with sessions that keep their state across requests and reuse the longest
common prefix exactly ([Serving](#serving)), a sequence-parallel prefill with CED, the default
([Prefill](#prefill)), greedy and sampled decode (temperature, top-k, top-p, seed, top logprobs:
[Sampling](#sampling)) with DSpark drafts ([Drafts](#drafts-dspark)), stop ids, max tokens and
cancel. End to end through the front-end with drafts ([Measured](#drafts-dspark-measured)): decode
67.0 prose, 99.0 code, 96.5 agentic and 133.2 JSON tok/s (the SGLang kit with its MTP drafts: 46.8,
77.4, 73.6, 106.9; the engine without drafts: 53.6–53.7), time to first token 128–150 ms (kit
244–280), agentic turns of +2k tokens on 16–25k contexts 0.42–0.45 s (kit 0.71–0.79), the whole
previous turn reused every time. Decode with drafts emits the tokens decode without them emits, bit
for bit, greedy or sampled, after an exact or a CED prefill. Decode matches model.py up to bf16
numerics and, on the 28,489-token agentic transcript, reproduces model.py's collapse after token
16,864; the prefill path agrees with the step path ([Measured](#measured-2026-10-11)). Next: the
step's weight kernels and the prefill's collectives.

## Process model

The same binary and arguments run on every box ([run4.sh](../engine/server/run4.sh)), with the fleet
file beside it as `fleet.conf` ([`fleet/homelab.conf`](../fleet/homelab.conf); `--fleet FILE` reads
another): the ring's devices and addresses, and each rank's files and Engram read batch. The rank
comes from `--rank` (the scripts pass it) or from the host name, as the file's `RANK<r>_HOSTNAME` gives
it. Each rank logs the file it read, its weight file and its
read batch. Startup ([main.cpp](../engine/server/main.cpp)), in order:

1. CUDA, and every allocation except the weights' arena: `model::Model`'s constructor makes the KV
   state, the RoPE tables, the step buffers and the pinned staging. On GB10 CUDA takes only free pages,
   which the page cache fills ([findings.md](findings.md#environment-gotchas)), so `run4.sh` first
   drops the model files' cache on every box ([evict_cache.py](../engine/server/tools/evict_cache.py)),
   and a server allocates everything it needs by the end of startup (6.).
2. The ring (`comm::Ring`, [ring.md](ring.md)): waits for the other three ranks.
3. The weights (`dsw::WeightFile::load`, O_DIRECT; a file cut for another rank is refused): 70.49 GiB per rank, skipping `mtp.*` (the DSpark
   blocks, not used yet). Rank 0 reads at 2.0 GB/s (37 s), rank 1 at 5.0 GB/s, ranks 2 and 3 at 9.4–9.6.
   `Model::bind` then re-lays anything the file holds untiled, all-gathers the router gate (every
   rank keeps all 384 rows) and the mHC projections, and multiplies Engram's q and k weights.
4. The Engram store ([engram.md](engram.md)): the hash map, this rank's shards of both tables on its
   SSD, an io_uring thread pinned to CPU 9, a row cache of `--engram-cache-mib` (default 4096).
5. The prefill's buffers and gathered weights (serve, `--big`); rank 0 prints every rank's free
   device memory after them.
6. Serve only: the spec step's graph, one for every context ([Drafts](#drafts-dspark)), then the
   session store's pool ([Serving](#serving)). Each rank offers its free device memory less
   `--session-margin-gib` (default 2) and its MemAvailable less the margin and the Engram row cache
   still to grow; one all-gather, and every rank reserves the smallest offer, at most `--session-gib`
   (default 12). Rank 0 logs the size and every rank's offer. Then the store's disk tier: each rank
   offers its spill directory's free space less 32 GiB, at most `--disk-gib` (default 32), and every
   rank makes a file of the smallest offer.
7. A barrier, then the main thread pins itself to CPU 15, and each rank logs its free device memory,
   MemAvailable and resident size (again at `done` / `stopped`). A watchdog thread kills the process
   when one step runs longer than `--step-timeout` seconds (a collective whose peer died spins
   forever).

All four are ready about 43 s after start with the full model. A server at its defaults then has
18.7–20.2 GiB of device memory free per rank (2026-10-12, after the model files' cache was dropped),
and nothing takes device memory after startup: free memory at ready and at stop differs by less than
0.05 GiB on every rank.

**Who decides.** Rank 0 decides, by commands ([cmd.h](../engine/server/cmd.h)): a `Cmd` of ~470
bytes goes to all ranks through `Ring::allgather_host` on the bootstrap TCP sockets, and ranks 1–3
loop on it (`follow()`); `kCmdStop` ends them. A command carries up to 12 session-state operations,
applied first on every rank, then one of: `kCmdTokens` (a prompt's token ids, sent after it in pieces
of 64 KiB), `kCmdPrefill` (a chunk of the many-row path, optionally starting the next chunk's
Engram lookups and sampling the last row), `kCmdRows` (≤ 8 rows through the step), `kCmdDecode` (a
whole decode loop: first token, sampling, stop ids, max tokens, the context limit, window snapshots
on the way), `kCmdState` (operations only) and `kCmdStep` (the golden modes' checked steps). Every
rank runs the identical kernel sequence, so the residual stream, routing, index choices and samples
are bit-identical on all four (sums are in rank order).

**Decode drives itself.** Each step's sampling hands every rank the chosen tokens
([Sampling](#sampling)), so in a decode loop every rank launches the next step from its own copy of
the sample, with no message from rank 0 per token, and every rank stops at the same step (same
tokens, same stop ids, same limits). Steps are queued one ahead: step n + 1 is launched before step n
has ended, its token left to the device (`-1`: row 0 of the previous step's sample, fed from device
memory by `k_step_in`), and its Engram lookups submitted once the host sees the token
(`engram_late`, 1 µs; the GPU waits for them only at layer 1). Inputs reach the device through a ring
of 4 pinned `StepIn` slots, results come back in a ring of 4 pinned `SampleOut` slots that the host
spins on (`Model::wait`). Rank 0 cancels by setting a control word that the next sampling carries to
every rank (`SampleOut::ctl`): all four leave the loop after the same step.

## Memory per rank

GB10's 128 GB of LPDDR5X is shared by CPU and GPU (121 GiB visible). With `--max-ctx 65536`:

| what | size | notes |
|---|---|---|
| weights | 70.49 GiB | the rank's file minus `mtp.*` ([layout.md](layout.md)) |
| router gate, all 384 experts | 157 MB | 40 × 384 × 5120 bf16; the rank's own 96 rows stay in the arena |
| mHC projections, whole | 157 MB | 40 × 2 × [24][20480] fp32, all-gathered at bind |
| compressed KV + index keys | 58 MB | layers 2, 8, 14 (ratio 2) and 20 (ratio 1); 356 B per entry: 256 KV, 32 scales, 64 key, 4 key scale; replicated |
| window rings | 5.4 MB | 40 layers × 256 slots × (512 + 16) B, FP8 |
| RoPE tables | 34 MB | plain and YaRN, max_ctx × 32 × float2 |
| step buffers | ~12 MB | R ≤ 8: activations, split-K partials, gathered logits (4.1 MB) |
| Engram row cache | 4 GiB (1 GiB in the runs below) | host memory, plus pinned staging |
| prefill buffers and gathered weights | 3.76 GiB (4096-row chunks), 4.89 GiB (8192) | made at startup (serve, `--big`); wo_b's slices are 1.68 GB of it. The dense index scans' key scratch is one batch of rows since 2026-10-12, not every row: about 0.2 GiB less at 65536; at 266240 (8192) 4.69 GiB, from 5.77 |
| ring region | 48 MiB | pinned host memory, 12 × `--ring-cap` (default 4 MiB) |
| DSpark blocks `mtp.*` | 2.0 GB | with drafts (serve's default; [Drafts](#drafts-dspark)); plus 16 MB for their mHC projections and router rows whole, and 165 MB for the FP8 copy of the head (serve's default) |
| spec step graph | 0.24 GiB | one for every context (index_dense reads the scan's length from the positions), captured at startup: its conditional bodies (8 row counts) live in device memory. It was one per 4096-position context bucket, ~210 MiB each: 3.33 GiB at 65536, about 13 GiB at 262144 |
| session store | 12 GiB | serve: one pool reserved at startup, the same size on every rank (`--session-gib`, [Serving](#serving)) |
| the store's disk tier | 528 MiB | serve: pinned host memory, 512 MiB of staging for spills in flight and two 8 MiB read buffers |

Each process holds 71.1 GiB of GPU memory once its graphs exist (`nvidia-smi`: 72,830 MiB). Context
costs little memory: about 890 B per position for all 40 layers, since the four compressed caches
serve every layer, so 1M positions would need 0.9 GB. Nor do the graphs grow with it (one per row
count). Context length is bounded by the indexer's scan time instead: a one-row step at 256k takes
0.38 ms longer than at 1k ([Long contexts](#long-contexts)).

## The step

A step is R ≤ 8 consecutive positions of one sequence: a decode step (R = 1), a verify step (R ≤ 6)
or a short stretch of a prompt (long prompts take the [prefill path](#prefill)). On every rank:

**Host** (`Model::launch`, then `Model::wait`). Append the rows' token ids to the history, hash their
n-grams and submit both Engram layers' row requests to the store (the GPU waits for them only when it
reaches layer 1 or 14), write the next pinned `StepIn` slot (token ids or "the previous step's sample",
positions, the store's sequence numbers, the sampling parameters), launch the step's graph and
return; `wait` spins on the step's `SampleOut` slot in pinned memory.

**Device**, on the main stream unless marked (sa, sb: side streams at the highest priority):

```
k_step_in (StepIn to device) · embed (4 copies of the token's embedding)
per layer L = 0..39:
  Engram (L = 1, 14): wait for the store · all-gather the rows (fixed R×24×264 B per rank) · pack ·
      wkv GEMV (FP8, 6400 × 6144) · partial dots · all-gather the partials · combine into x
  if x changed since the last collapse (first layer, Engram, injection): yraw = hc_pre(x, pre_mix)
  attention
    sa: hc_project of this rank's copy of x (columns [5120 rank, +5120) of hc_fn·x, and the sum
        of squares: 25 partial sums per row)
    attn_norm + wq_a + wkv of this rank (one FP8 GEMV launch, 320 + 128 rows)
    kv sources: compressor_project_rows (this rank's 128 rows of the compressor's wkv/wgate)
    index layers: sb: the indexer's weights_proj (32 heads)
    allgather_attn: the q_a, kv and compressor slices into full rows, the mix partials along
        (one collective)
    sa: mix_ranks: the four ranks' partials added in rank order, sinkhorn -> a_pre, a_post, a_comb
    sb: kv_write (kv_norm, RoPE, FP8 into the window ring) · kv sources: compressor_emit
    q_norm + wq_b (+ the indexer's wq_b on index layers) (one FP8 GEMV launch, 8192 + 4096 rows)
    index layers: index_dense on the dense scans of layers 2, 8, 14 and 20 (all rows in one grid of
        one CTA per SM: scores, top-512; layer 20 also the 2048-block pool), index_dense_blocks on
        the Reindex layers' 16,384 pool positions (the same grid, a row per group); every rank
        scans everything, so no merge
    sparse_attn (window + 512 selected compressed entries, split-K)
    wo_a (grouped FP8, output quantised) · wo_b (FP8, row-parallel partial sums)
    allreduce_hc (40 blocks): fp32 sum in rank order -> bf16 · hc_post into mid ·
        yraw = hc_pre(mid, a_pre)
  FFN
    sa: hc_project of this rank's copy of mid · mix_allreduce (one block: the partials of the four
        ranks added in rank order, sinkhorn) -> next pre_mix, f_post, f_comb
    ffn_norm + router logits (all 384, bf16) in one launch; its CTA 0 also quantises the input
    moe_routed: top-6 with bias, the shared expert and 6 routed experts, this rank's 576 columns
    allreduce_hc: fp32 sum -> hc_post into x · yraw = hc_pre(x, next pre_mix)
head: final norm + this rank's 32,320 vocabulary rows (one bf16 GEMV launch) · sample: each rank's
  summary of its logits exchanged, the same token picked on every rank (Sampling), straight into
  pinned memory and the next step's feedback
```

The mHC is single-pass (model.py `Block.forward`): attention collapses the stream with the previous
FFN's pre mix, the FFN with attention's. Each collapse therefore runs in the epilogue of the
all-reduce that ends the sublayer before it, and the next sublayer starts from `yraw`. A sublayer's
own mixes are needed only there, at its end, so they are worked out off the path: each rank
projects only its copy of the stream (a quarter of hc_fn, 0.5 MB per sublayer) and the ranks'
partial sums are added in rank order, so every rank gets the same bits. The attention's ride its
all-gather; the FFN has no collective before its all-reduce, so its take a one-block collective of
their own on `sa`.

**Collectives.** Per decode step: 40 × (allgather_attn + 2 allreduce_hc), 4 for the two Engram
layers, and the sampling: 125 on the main stream, in the same order on every rank, plus 40
`mix_allreduce` on `sa`, each forked after the attention's all-reduce and joined before the FFN's,
so nothing else takes a sequence number in between. They are kernels over the ring's device
channel ([ring.md](ring.md)): at most 48 blocks when they wait (the all-reduce has 40; a mix
collective has ended before it starts), sizes in multiples of 16 bytes.

**Streams, PDL and graphs.** The mixes run on `sa` and the KV writes and the indexer's head weights
on `sb`, forked from and joined to the main stream by events; their priority puts their few CTAs
ahead of the queued GEMV CTAs. The GEMVs, the attention kernels and the runner's two fused
collectives on the main stream (`ops::allreduce_hc`, `ops::allgather_attn_mix` in
[fused.cu](../engine/model/fused.cu)) are chained with programmatic dependent launch: a collective
kernel calls `griddepcontrol.wait` before taking its ring sequence number and `launch_dependents`
once it has posted its send, so the next kernel's prologue (and its weight prefetch) overlaps the
wait. comm's plain all-gathers (Engram), the argmax and the side stream's kernels (`mix_ranks`,
`mix_allreduce`) are ordinary launches. A step without golden checks replays a CUDA graph: one per R
for every context (index_dense takes the scan's length from the step's positions), captured on
first use with `cudaStreamCaptureModeThreadLocal` and instantiated with
`cudaGraphInstantiateFlagUseNodePriority` (otherwise the side streams lose their priority). Each R's
first step runs eagerly (kernels set their shared-memory attributes once, outside any capture).
Inputs reach a replay only through the pinned `StepIn`, so graphs are never updated. Steps with
injection or logits run eagerly on every rank; rank 0's dumps add no collectives, so the others
still replay.

## Sampling

`ops::sample` ([sample.cu](../engine/model/sample.cu)) ends every step: one kernel that is also the
step's collective. Each of the R blocks reduces this rank's 32,320 logits of its row to a 544-byte
summary, the four summaries are exchanged (one all-gather of R × 4 summaries), and every rank picks
the same token from them in rank order:

- the maximum and its lowest id (greedy: ties go to the lower id, as a single argmax would);
- the partition sum of exp((l − max) / T), so the chosen token's log-probability (after the
  temperature) is exact;
- the Gumbel-max winner, with noise hashed from (seed, position, token id): temperature without
  truncation samples the whole vocabulary exactly, and a seed reproduces a run on any rank count;
- for top-k / top-p / top logprobs, up to 64 candidates per rank: whole bins of width 0.25 below the
  rank's maximum (at most 64 units down), plus the largest logit left out. The union (≤ 256) is
  sorted; temperature, then top-k (renormalised), then top-p are applied over it, and one uniform from
  (seed, position) picks. The result is exact whenever every kept token's logit is above the
  largest one left out of the union; otherwise `SampleRow::flags` says approximate.

The step's `SampleOut` (token, log-probability, the value that won, up to 20 top ids and logprobs per
row, rank 0's control word) lands in pinned memory on every rank, and the tokens also in device memory
for the next step's feedback. `Sampling` comes with every step (`StepIn`), so requests with different
settings need no graph of their own.

## Drafts (DSpark)

With drafts (serve's default; `--spec` in decode and generate) every rank also loads the model's own
drafter, the three DSpark blocks `mtp.*` (2.0 GB
per rank, [engine/dspark](../engine/dspark/README.md)), and decode runs as speculative steps
([spec.cu](../engine/model/spec.cu), DESIGN.md decision 7).

**A spec step** is one CUDA graph for every context (a server captures it at startup: 0.2 s and
0.24 GiB, since the switch's bodies live in device memory; with a graph per 4096-position bucket it was
16 at `--max-ctx 65536`, 2.4 s and 3.33 GiB). The row count is a device decision, through a
conditional switch node:

```
k_spec_in    the StepIn slot (k_step_in's work); a verify step's rows, tokens and positions from the
             device's SpecState; the switch's value: rows - 1
switch       body R - 1 (R = 1..8): the R rows through the model as a plain step runs them, plus
             main_hidden (the bf16 mean of the 4 copies) at the inputs of layers 37, 38 and 39
sample       every row as in a plain step; a verify step's drafts judged; its final rows and the token
             after them (the bonus) into SpecState and SampleOut
accept       the final rows' main_hidden into the 3 blocks' windows
draft        from the bonus: 5 drafts and the confidence head's survival
k_spec_next  the next step's rows: the bonus, then d_1..d_k while survival[j - 1] >= thresh (k <= the
             max drafts, rows below the request's end); then the step is published
```

**Verify.** Row j of a verify step (position P + j) predicts the draft in row j + 1, which is
accepted when it equals row j's sample. Greedy, this accepts the longest prefix of drafts that equal
the model's argmax. Sampled, it is speculative sampling with the greedy draft as a one-hot proposal
q. Accepting d with probability min(1, p(d) / q(d)) = p(d), and otherwise drawing from p without d,
gives the same law as drawing from p and accepting when the draw is d. The draw is the plain
sampler's, with noise from the seed and position. So the step emits the accepted drafts and then row
n's own sample, and every token is the one plain decode would sample. A row's arithmetic does not
depend on the step's row count (tf with k = 1, 2, 6 gives one hash). Speculative decode therefore
emits the plain decode's tokens bit for bit, greedy or sampled ([Measured](#drafts-dspark-measured)).

**The loop** (`spec_loop` in [cmd.cpp](../engine/server/cmd.cpp)) runs on every rank alike:

1. Step 0 feeds tok_0 (`launch_spec`, the host's row).
2. Step i + 1 is queued before step i ends (`launch_verify`). Its rows and positions are left to the
   device (the host passes a bound on them, next_pos0 + next_rows + 5, which one graph covers).
3. Once step i is in, `SampleOut::next_*` names step i + 1's rows, and their Engram lookups go to the
   store (`engram_late`).
4. Step i's tokens are emitted, and stops apply per token: what follows a stop token in its step is
   dropped.
5. Step i + 2 is queued.

Positions written past the history (rejected drafts, the queued step, the drafter's block keys at
anchor + 1..5) are reported as `DecodeEnd::written`, for the restart rule below.

**The verify length.** The step verifies d_j while survival[j - 1] >= λ·ΔC
([README](../engine/dspark/README.md#the-confidence-head-for-the-verify-length-policy-design-decision-7)).
- λ is the tokens per ms so far: an EMA (1/16) of each step's emitted tokens over its modelled time
  C1 + (R - 1)·ΔC + C_draft.
- `--spec-cost C1,dC,Cdraft` sets the three costs. The default is 19.4, 3.6 and 3.3 ms, measured
  before the decode fusion and the draft-pass cut. The costs measured now (18.6, 3.7 and 2.5)
  decode up to 1.6 % slower ([Step cost by rows](#drafts-dspark-measured)).
- `--spec-thresh` fixes the threshold instead, and `--spec-drafts` caps k.
- `--spec-gate ratio` or `marginal` prices each row count instead (`--spec-row-ms`: the ms of a step
  verifying 0..5 drafts). Neither beat the default ([the A/B](#drafts-dspark-measured), "Gates by a
  cost per row count").

Each rank computes the threshold from the same SampleOuts, so every rank makes the same choice. The
decode report lists each row count's measured GPU time, which shows whether the costs still hold.

**Prompt and sessions.**
- A prefill chunk sends main_hidden of its last 256 rows to `accept` (8 per call). That is every
  window slot, so the drafter's windows are exact wherever the main ring is. (The drafter attends to
  128 of them; the other 128 cover restarts up to 128 positions back, as the main ring does.)
- It also sends the 128 rows before each snapshot inside the chunk, whose window copies go with the
  snapshot.
- The chunk's rows are row-split ([Prefill](#prefill)), so each rank takes main_hidden of the seed
  rows it owns (the inputs of layers 37–39) into its block, one all-gather shares the blocks, and
  `accept` takes groups of up to 8 that stay within one rank's rows. This runs after the chunk's
  sample, and the server sends the first token before it drains the chunk, so the first token waits
  only for the last chunk's main_hidden (earlier chunks' seeding is on the path).
- Under CED only the replay runs layers 37–39, so a CED turn seeds the windows with the replay's 128
  rows [n − 128, n): all the drafter attends to, and where the decoder's rings are valid. A snapshot at
  n carries them; the encoder's snapshots inside CED chunks carry no window copies, as they carry no
  decoder rings.
- A step of explicit rows (a short prompt remainder) is a spec step, and accepts its rows.
- Window snapshots carry the drafter's [pos − 128, pos), the end snapshot carries the whole windows,
  and restores bring them back.

Positions written past the history (rejected drafts, the queued step, the block keys) count in w, the
restart rule's bound ([Serving](#serving)): the ring is exact for [max(lo, w − 256), end) and the
carries for [max(clo, w − 16), end). The rule used to assume the ring exact back to end − 256, which
a short continuation after a snapshot restore already broke, and drafts would have.

## Prefill

The many-row path ([prefill.cu](../engine/model/prefill.cu), `Model::prefill`) runs chunks of up to
`--prefill-rows` positions (default 4096; 8192 for the server's CED turns) with the prefill GEMMs
([kernels/README.md](../engine/kernels/README.md)) and the multi-row forms of the attention kernels.
It does the decode step's arithmetic (the same weights, bf16 roundings at the same points, sums in
rank order), so a chunk's state equals the 8-row steps' up to summation order, and splitting a prompt
into different chunks gives bit-identical results.

It is sequence parallel around the residual stream. Rank r owns rows [r·q, r·q + n) of a chunk of M
rows (q is M/4 rounded up to a multiple of 8). Only those rows have their residual stream (four
copies of 5120), their mixes and their sublayer boundaries (`dsk::hc_boundary_p`: hc_post, the next
sublayer's mixes, hc_pre, its norm and act_quant in one pass). The tensor-parallel GEMMs need every
row:
- their inputs arrive by all-gather, in FP8 with its scales where the GEMM reads FP8;
- their partial sums leave by a reduce-scatter by rows (`comm::reduce_scatter_rows_*`). Each rank
  gets its own rows' sums, added in rank order as the all-reduce added them.
- wo_b is the exception. An all-to-all (`comm::alltoall_rows`) brings every rank's wo_a outputs for
  own rows (FP8, 2,112 B per row), and wo_b runs on own rows with every rank's slice of it (gathered
  at the first prefill). Its four partials are added in rank order (`dsk::sum4_bf16_p`): the
  reduce-scatter's bits for a fifth of its traffic.

Per layer:

```
attention: wq_a + wkv (and the compressor's wkv / wgate) for own rows with every rank's weights ·
    one all-gather of the results
    window: the 127 positions before the chunk from the ring into a linear buffer, the chunk's keys
    after them (kv_write); compressor_emit for the whole chunk (entries, index keys, the carry)
    q_norm · act_quant · wq_b (this rank's heads) · index layers: the indexer for own rows (the
    dense scans: index_q_prep, then dsk::index_topk_p in batches of 128 rows; the Reindex layers'
    pool scans: index_fused) ·
    all-gather of the top-k (layer 20: also the candidate pool)
    sparse_attn_p (window + 512 selected entries) · the last 256 positions back into the ring ·
    window snapshots inside the chunk
    wo_a · all-to-all of its FP8 outputs under wo_b with this rank's slice on own rows · wo_b with
    the other three slices · the four partials added in rank order · hc_boundary_p on own rows into
    the FFN
FFN: router for own rows · all-gather of the input's act_quant and the routes · moe_prefill:
    bucketing, shared expert and up over every row (this rank's 576 columns of every expert), then
    down in four column groups, each group's combine inside its reduce-scatter (fp32, model.py's;
    dsk::moe_combine_rs) under the next group's down. Chunks up to 2,520 rows combine inside the
    down kernel and reduce-scatter its fp32 outputs · hc_boundary_p on own rows into the next layer
Engram (layers 1 and 14): own rows with every rank's wkv
```

The reduce-scatters run on a high-priority stream with 8 blocks each, so the GEMMs keep most of the
GPU. The main stream issues no collective until it has joined them, so every rank still sees one
sequence of collectives.

`sparse_attn_p` stages each row's RoPE table row with its q, and reads the positions a row ahead
(a4d174d, 2026-10-11). Before that, the q RoPE read the table from global memory inside the q load,
an L2 round trip per dependent group, and the window's prefetch waited on the position. bench_pattn
at 4096 rows: 2.78 → 2.47 ms, and with the window only, 1.24 → 0.79 ms.

**CED** ([ced.md](ced.md), `Model::prefill(…, ced)`; the server's default). A CED chunk does three
things:
- It runs the encoder, layers 0–19.
- It runs layer 20's compressor for its rows. That gives the decoder's global KV and index keys, bit
  for bit the exact path's.
- It keeps the layer-20 input of its last 128 positions: x, y with its act_quant, and the mixes,
  55 KB per position. They go in slot position mod 128, so the 128 can span two chunks.

The chunk that samples then replays layers 20–39 over [P − 128, P) as one chunk of 128 rows: 32 per
rank, the records all-gathered.
- Its window starts at P − 128 (`sparse_attn_p`'s window floor).
- Layer 20's compressor does not run again.
- The replay writes the decoder layers' rings, and the head reads its last row.
- Window snapshots inside CED chunks hold the encoder layers' rings only, except one at P, which the
  replay fills.
- At P = 128 CED is the exact forward, and the engine gives the same bits whether it runs exact, CED
  in one chunk, or CED in chunks of 48.

Engram's lookups for a chunk go to the store before the chunk starts (the server starts the next
chunk's while this one runs: `Cmd::next`); two sets of buffers alternate. The last row of the last
chunk (of the replay, with CED) runs through the head and the sampling as a one-row step, so the
first token comes out of the prefill's own stream. The prefill buffers and the weights gathered for
them take 3.76 GiB per rank at 4096 rows and 4.89 GiB at 8192, of which 1.68 GB is wo_b's four
slices for all 40 layers. Serve and the golden modes with `--big` make them at startup
(`Model::prefill_init`), other callers on the first prefill. Every rank allocates its share first and
the ranks meet at a host barrier before the weights are gathered: a rank that cannot allocate dies
with CUDA's error and the others with "peer closed" at the barrier, instead of waiting in a
collective. Rank 0 then logs every rank's free device memory (`free device memory after the prefill
buffers, ranks 0-3: …`).

A 4096-row chunk of every layer, on the GPU (nsys, rank 0), by stage:

| stage | time | breakdown |
|---|---|---|
| bulk all-reduces, every rank on every row (engine/attn's attention, the step's mHC) | 1.94 s | all-reduces 724 ms, MoE 490, mHC 251, attention 168 |
| sequence parallel | 1.26 s | MoE 504 ms, reduce-scatters 318, all-gathers 111, GEMMs 151, attention 96, mHC 54 |
| reduce-scatters under the column groups | 1.12 s | |

A warm chunk end to end (`--mode tf --big`) takes 1.06 s exact and 0.54 s with CED (20 layers),
7.5k tok/s.

**How the collectives were set (2026-10-11).** Three measurements from building this path, which until now
were only in commit messages:
- **Copies ahead of the posts** (2a3aa6b, kept). The bulk reduce-scatter and the multi-buffer all-gather copy
  the next post's data in while the current one travels, and post it the moment the current one has
  arrived. `ring_test`, at a cap of 4 MiB and 64 MiB a rank:
  - the reduce-scatter moves 18.4 GB/s of input, 9.2 GB/s of it on L0 (75 % of its 12.25). It moved
    16.1 GB/s before the copies went ahead;
  - the all-gather moves 22.5 GB/s of output, 8.4 of it on L0;
  - the bulk all-reduce moves 9.3 GB/s, 8.1 of it on L0.
- **The slow link's share** (`comm::Options::rs_slow_pct`, b7ad036). This is the share of the opposite
  rank's quarter that the reduce-scatter routes across the slow link, L0.
  - 25 would balance the links' loads, but it measured slower: 14.4 GB/s against 50's 18.4, at 64 MiB.
  - Relays are store-and-forward, so a post's time follows its larger relayed part over two hops, not
    the links' loads.
  - The default stays 50.
- **Blocks per reduce-scatter under the column groups** (72860c6; `kRsBlocks` in model/prefill.h). A
  4096-row chunk, with rank 0 under nsys, timed over the GPU window:

  | reduce-scatters | chunk |
  |---|---|
  | sequence parallel, not overlapped | 1262.8 ms |
  | overlapped, 16 blocks each | 1245.6 ms: the contention took `moe_down` from 176 to 240 ms |
  | overlapped, 4 blocks | 1223.4 ms |
  | overlapped, 8 blocks (kept) | 1117.1 ms |

  The teacher-forced hash was the same in all four runs (4eff0352e144e2e7).

**The shared expert on a side stream** (`MoePCfg::side`, 2026-10-11; off, kept as an option).
- **The claim.** The kernels README says it "measured slower, because the routed experts already fill the
  GPU". No commit gave figures.
- **Where the figures come from.** The kernels line's session log kept bench_kernels prefill's output, so
  the figures below are recovered from it, not re-measured.
  - Two rows of that output are compared: "moe, uniform routing", with the shared expert in line, and
    "moe, uniform, shared beside", with it on the side stream.
  - Each row is the median of 7 interleaved rounds, with weights cold in L2.
  - There are 34 pairs from 11 runs, from 2026-10-11 16:54 to 2026-10-12 00:17. The kernels changed
    between runs, so only the pairs within a run compare.
- **The side stream against in line:**
  - median +0.85 %, mean +0.6 %, ranging from −3.5 % to +3.4 %; slower in 21 pairs of 34;
  - by chunk:
    - 128 rows −0.9 % (3 pairs), 256 rows +0.1 % (1);
    - 512 rows +2.9 % (3), 1024 rows +1.8 % (2);
    - 2048 rows +1.7 % (9), 4096 rows +1.1 % (7), 8192 rows −0.5 % (9).
- **Verdict.** It gains nothing beyond the ±3 % spread between runs, so it stays off.

## Serving

`--mode serve` makes rank 0 a server ([serve.cpp](../engine/server/serve.cpp)) on a Unix socket
(`--socket`, default `/tmp/dsgb10-engine.sock`, mode 0600) speaking [protocol.md](protocol.md): HELLO,
SUBMIT, CANCEL, RELEASE in; HELLO, TOKENS, DONE out. A reader thread takes frames (a new front-end
connection replaces the old one, whose requests then count as cancelled), a writer thread sends them,
so the engine never blocks on the socket; both run on CPU 12. One thread runs the requests: a new one is
prefilled while the others pause, and up to `--slots` of them decode together, taking turns when there
are more ([Scheduling](#scheduling-time-slicing-and-batching)).

Prefill chunks are 8192 rows with CED (`--prefill-rows`; 4096 with `--exact`): 13 % less time per
token than 4096, at 0.85 s per chunk. A new request's prefill runs whole while the requests decoding
wait (prefill first). Were their decode steps to run between its chunks (not built), each of them would
wait for a whole chunk, and 4096 rows (0.49 s) might then be the better trade.

**A request.** Validate (prompt within `--max-ctx`, sampling parameters, ≤ 32 stop ids); find the
session or its base; take a slot ([Scheduling](#scheduling-time-slicing-and-batching)), moving the
state of the session that held it into the store; choose where to restart (below); send the state
operations and the prompt's tokens from three positions before the restart (Engram hashes three back);
prefill the rest in chunks of `--prefill-rows` (a remainder of ≤ `--small-rows` = 8 positions through
the step); send the first token; decode, alone or with others, with the request's sampling, stop ids
and `max_new_tokens`, sending each token as it comes; DONE with the reason, `cached_tokens` and the
queue / prefill / decode times. A CANCEL is checked between prefill chunks and, in decode, through the
control word. Without the KEEP flag the session is freed afterwards; with it, it stays.

**Sessions.** The model's working state (window rings, compressor carries, compressed caches) is laid
out once per slot (`--slots`), and each slot's belongs to one session, live there. Every other session
lives in the store
([state.cu](../engine/model/state.cu)): its compressed caches in pages of 4096 positions (3.6 MB),
window snapshots (each layer's ring slots of positions [s − 128, s), with the drafter's windows:
2.9 MB) at the end of its last prompt and every `--snap-every` = 1024 positions, thinned further back
(below), and an end snapshot (every layer's whole ring and the carries, with the drafter's windows:
6.1 MB). A 57k-token session holds ~110 MiB (~210 MiB with every snapshot).

**The store's memory** is one pool reserved at startup (step 6 of [Process model](#process-model):
the smallest rank's free memory less a margin, at most `--session-gib`, default 12 GiB), so a page
cache that grows during serving cannot fail a session's allocation. Blocks are 64 KiB multiples,
handed out best fit with coalescing, and every rank runs the same operations in the same order, so
the ranks hold the same blocks. Between requests the server keeps at most the pool less room for one
long request (max(256 MiB, 1/16 of it)), evicting least recently used sessions. Inside a request, an
allocation that finds the pool full evicts the least recently used session that the current command
does not use (the sessions its state operations name, its own and the slots' are pinned), the same one on every
rank; rank 0's server then forgets it (`serve: session … evicted to make room`). What still does not
fit is skipped, never a failure: a snapshot not taken, pages saved only as far as they go (the
session then restarts at most there), no end snapshot. Checked with a 25 MiB store (`--session-gib
0.025`, a session of 3k positions holds ~19 MiB): `serve_test.py --ced` ran with 13 evictions inside
requests and one snapshot skipped, every comparison but one equal, and the two failing checks are the
short fork's, whose base had been evicted, so it ran cold (same tokens, `cached` 0, a cold CED
prefill's log-probability); at the defaults it passes 17 of 17, and `--exact` 9 of 9.

**The disk tier.** A session that leaves the store is spilled to disk rather than freed (unless
`--disk-gib 0`). Each rank writes its own slice to its own disk: `rank<r>.spill` in `--disk-dir`,
by default the fleet file's `RANK<r>_SPILL`: `/data/dsgb10/spill` on rank 0 (its USB4 SSD) and `~/dsgb10/spill` on the others (NVMe).
- At startup each rank removes any earlier file and offers its free space less 32 GiB, at most
  `--disk-gib` (default 32). Every rank makes a file of the smallest offer (none under 1 GiB) with
  `posix_fallocate`, opened `O_DIRECT`, mode 0600 in a directory of mode 0700. The file is removed
  at exit, unless a planned stop keeps it, with its sessions, for the next start, which then opens it
  as it is ([Kept sessions](#kept-sessions)).
  On 2026-10-12 the boxes had 209 GiB, 178 GiB, 1.3 TiB and 831 GiB free, so the cap could go to
  about 145 GiB.
  32 GiB holds about 170 thinned sessions of 128k, or 110 of 250k.
- The file's room is a second pool of 64 KiB blocks. Every rank runs the same operations on it, so
  every slice lands at the same offsets. When it is full, the least recently used spilled sessions
  are dropped from it (`state: session … dropped from the disk tier`), and the server forgets them.
- Writes stay off the critical path. A spill queues each block's copy to the pinned staging on the
  model's stream, after whatever wrote the block, and frees the block in the store at once. A writer
  thread waits for the copies' event and writes them out. A spill waits for the disk only when the
  staging is full.
- Eviction is otherwise unchanged. Between requests, the server spills least recently used sessions
  down to its budget (`serve: spill session …`). Inside a request, an allocation that finds the store
  full spills its victim.
- A request whose session (or fork base) is on disk brings it back before it plans its restart: its
  pages, then its end snapshot, then its window snapshots from the latest down, as far as the store has
  room. The slots' sessions stay (every command pins them), so the room comes from other sessions. Reads go through two 8 MiB pinned buffers, each `pread`
  overlapping the previous buffer's copy to the device. A session read back right after its spill
  first waits for its own writes. From then on it is a stored session like any other.

[Long contexts](#long-contexts) has the times.

**Snapshot thinning.** With every snapshot kept, a long session's window snapshots outweighed its
caches three to one, while restarts far back are rare. In soak1, every restart of a session from its
own window snapshot was within 4,096 positions of its end (21 within 128, 242 within 1,024, 44 within
4,096). Only forks reach further back, to a tool result anywhere in the session. So after each
request the server drops the snapshots that `keep_snap` ([serve.cpp](../engine/server/serve.cpp)) no
longer keeps, by their distance d from the history's end E:
- all of them while d < 8,192 (8 × `--snap-every`);
- further back, multiples of 2^k × 1,024 while d < 2^k × 8,192: every 2,048 positions out to 16k,
  every 4,096 out to 32k, and so on, four per doubling of d;
- and the one at the last prompt's end.

A prefill takes only the snapshots that its own prompt's end would keep, and encoder-only snapshots
are thinned with the rest. So:
- a restart d back re-prefills at most 2d / 7 positions from the snapshot below it (under 1,024
  within the last 8,192);
- a session keeps O(log E) snapshots: 16 at 32k, 24 at 128k, 28 at 250k (32, 128 and 245 before);
- as E grows, each position's spacing only coarsens, by powers of two, so a dropped snapshot is never
  wanted again.

`--snap-all` keeps every one. [Long contexts](#long-contexts) has the sizes and the cost of forks far
back.

**Exact reuse.** A request continues from the longest common prefix p of the session's history (the
tokens whose state it holds, generated ones included) and its prompt, capped at n − 1 so that the last
prompt position runs again for its logits. It restarts at the largest q ≤ p from which the state can
be rebuilt exactly:

- the live ring: a restart needs the 128 positions before it in the ring, which is exact for
  [max(lo, w − 256), E) of the history's end E. lo is where the state's lineage begins (0, or a
  restored snapshot's s − 128) and w is past the last position anything wrote: the history, a step
  queued ahead, rejected drafts, the drafter's block keys. So q = p when q − 128 ≥ max(lo, w − 256).
  A ratio-2 compressed entry completing at an odd q needs q − 1's projection, which the carries hold
  for [max(clo, w − 16), E) (clo: the lineage's restart point), else q − 1;
- the end snapshot: the same about its position, with the ranges it was taken with;
- a window snapshot at s: q = s, or s − 1 when s is odd (no carry is kept with it);
- otherwise 0.

Compressed caches are per position and append-only, so entries below q are kept as they are (loaded
from the store's pages when the session was not live) and entries from q on are overwritten as the
prefill passes them; no row reads an entry beyond its own position. A fork (`base_session_id`, the
session new) starts from the base's state the same way and copies its snapshots up to q. So a reused
request computes exactly what a cold one computes, bit for bit, except where the step and the prefill
path meet, whose summation orders differ:
- positions first computed by the decode step keep the decode step's roundings: a continuation over
  generated tokens equals a replay of the same history, not a cold prefill of it (serve_test.py checks
  both);
- a restart that reruns prefilled positions through the step (a remainder of at most `--small-rows`,
  as a regenerate's last positions are) matches the prefill path to numerics, not bits: serve_test.py
  checks the same first token and its log-probability within 2e-3 (3.6e-4 now).

**CED turns** (the default, DESIGN decision 5; `--exact` makes every turn exact). The rule depends
on how many positions the turn adds after its restart point q:

- **n − q ≥ 128: CED.** The turn runs CED (above). Its chunks run the encoder and layer 20's
  compressor over [q, n). The last chunk replays the decoder over [n − 128, n) from fresh rings. So
  it needs only the encoder's state at q: the encoder layers' rings and carries, and the compressed
  caches (layer 20's included, which the compressor writes for every position as the exact path
  does).
- **n − q < 128: exact continuation.** A shorter turn runs every layer over its rows. That needs the
  decoder layers' rings at q too, and they are not valid everywhere:
  - after a CED turn, the live rings hold the replay's keys and are valid from n − 128 on
    (`Session::dec_lo`); decode extends them;
  - a CED prefill cancelled between chunks leaves them valid nowhere;
  - window snapshots taken inside a CED turn's chunks hold the encoder layers' rings only
    (`Session::enc_snaps`, pruned with the history and copied to forks), except the one at n;
  - the end snapshot keeps the live rings' validity (`end_dec_lo`).

So a short turn restarts only where the decoder's window [q − 127, q) is valid. Where it is not,
`choose()` takes the latest restart point that leaves at least 128 positions, and the turn runs CED.

Either way the result is bit-reproducible:
- A CED turn computes what the same prompt prefilled cold computes, bit for bit, wherever it restarts,
  as long as every position before its restart came from the prefill path. Positions the decode step
  computed (an earlier answer the turn keeps) keep its roundings, as above: then the turn computes
  what the same history replayed computes, and a repeat that restarts before them (at an earlier
  window snapshot) can differ from it at a near-tie ([correctness review](review/correctness.md), P3).
- An exact continuation computes what the same history replayed in a fresh session computes.

`serve_test.py --ced` checks both, against a CED server. It covers:
- a short turn after a CED turn;
- long turns from the live ring;
- restarts from encoder-only snapshots, which a long turn takes and a short one may not;
- long and short forks;
- a turn after a cancelled CED prefill.

### Scheduling: time-slicing and batching

One thread runs the requests ([serve.cpp](../engine/server/serve.cpp); [batching.md](batching.md) has
the design, steps B1 and B2).

- **Slots.** `--slots N` (1 to 8, 1 by default) lays out the per-session state N times (the window
  rings, the compressed caches, index keys and carries, the drafter's windows: ~240 MB a slot at
  262,144 positions, allocated before the store). Each slot's state is one session's. A request takes
  its session's slot; else a free one; else the least recently used one whose session no request uses;
  else, with turns (below), the slot of the request whose turn comes last. The state of the session that
  held it goes to the store: its pages and an end snapshot, as a switch of the one live session did.
  Every command pins the slots' sessions in the store.
- **Prefill first.** A request that comes while others decode pauses them at their next step, through
  the control word, if it can have a slot. It is prefilled whole and sends its first token; then it
  decodes with them. A request for a session in flight waits for that request's end, as before.
- **Runs.** Requests between their first token and their end wait in turn. A run takes those at the
  front, as many as there are slots. Two or more decode together, a row of each per step
  ([`batch_loop`](../engine/server/cmd.cpp), no drafts); one alone runs the speculative loop with
  drafts. With drafts, the last request decoding in a batch pauses there, to go on alone with drafts. A
  request joins a run at its next token, the last one sent, at position prompt + sent − 1. The step
  queued behind a pause may have written that position's KV already; it is written again, with the same
  values.
- **Drafts across sessions** (B3, [batching.md](batching.md#b3-the-build-2026-10-13), 2026-10-13).
  With drafts, a run of up to kMaxRows / 2 requests (eight at kMaxRows 16) decodes with drafts across
  them ([`spec_batch_loop`](../engine/server/cmd.cpp), `kCmdSpecBatch`): a batch spec step holds each
  request's bonus token and its drafts, at most (kMaxRows − S) / S each for S requests and never more
  than the five the drafter makes (5 for two, 4 for three, 3 for four, 2 for five, 1 for six to
  eight), laid out on the device in request order. The verdicts are per
  request and the draft passes run one per request in its own slot, so every request's tokens and
  log-probabilities are its run's alone. Its end, stop ids, cancel (its slot's bit) and a pause act per
  request as in the speculative loop; a request that ends keeps its rows in the step already queued
  (their KV counts, their tokens do not); the last one decoding pauses to go on alone with its five
  drafts. Larger runs take the plain batch loop. The batch log line adds `drafts: T tokens and R rows
  per step, A of V drafts accepted; GPU ms by rows: …`.
- **Turns.** While more requests decode than there are slots, a run is a turn of `--slice` tokens
  (64). A request whose turn is over goes to the back of the queue; paused ones go back to the front,
  in order. A request whose slot went to another resumes from the store when its turn comes: its prompt
  and the tokens sent so far, restarted at its end snapshot (one row, ~25 ms, the state's load
  included). With `--slice 0` there are no turns: a request that would need the slot of another
  request in flight waits until one ends.
- **Exact.** Every row computes what it would alone, and the sampler's noise is a function of the seed
  and the position. A request therefore gives the same tokens and log-probabilities batched, sliced,
  resumed or alone. [slice_test.py](../engine/server/tests/slice_test.py) and
  [batch_test.py](../engine/server/tests/batch_test.py) check that.
- **Logs.** Each batched run prints `serve: batch of N: S steps in T ms, M ms a step (median); req,
  tokens, end: …`. A request's line adds `K slices, paused P s, resumes R ms, B tokens in batches`.

**Measured on the ring, 2026-10-12** (engine 33de65fd: B2 before master's B4 GEMV/MoE half).
- **Every request equals its run alone.** [batch_test.py](../engine/server/tests/batch_test.py) with
  `--slots 4`, at `--slice 64` and at 0: six requests (prompts of 0.8k–4k positions; greedy, with the
  top 5 log-probabilities, and sampled at temperatures 0.6–0.9 with top-p), each first alone; then four
  together, four staggered, all six over the four slots, a cancel and a stop id inside a batch, and two
  kept sessions back from the store beside another's run. Every request's tokens and log-probabilities
  equal its run alone: 0 failed. With one slot and with four: serve_test.py `--ced` 17 of 17,
  slice_test.py 0 failed, and a planned stop with two requests in flight kept both sessions, which
  went on after the restart with every position cached (1,028 and 2,002). The reference hashes hold
  in batches (`--mode batch`): four sessions in slots 3–0 (1k, 16k CED, 1k, 1k) give 1k's
  febc7dc54ee40a39 three times and 16k CED's 81f772ab79a950e4; two 1k sessions with the draft module
  bound, febc7dc54ee40a39 twice; sampled sessions (1k, 4k, 1k), the hash of the 1k one alone
  (a8f7a08f8cd1c5e8) twice. One session alone gives the reference hashes (tf 842278b28d87dad0, 1k
  febc7dc54ee40a39 at k = 1 and with drafts, 16k CED 81f772ab79a950e4, long28k CED 14650fb0739d0383, until
  2026-10-13 the hash's seed, see [Golden check](#golden-check); now 8f08a153d415c5c8) at 17.90, 9.21 and
  18.00 ms per token (1k at k = 1, 1k with drafts, 16k CED), against the release's
  17.93, 9.21 and 17.96. With `--slots 4`, kept sessions come back after a planned stop and not after a
  crash, an age past `--keep-days` or another run id (kept_test, as with one slot); and disk_test
  passes on a spilling server (14 sessions spilled, 10 back from the disk) and a store-only one, 4 of 4
  checks against cold each, their 12 requests equal.
- **Steps.** A batch step's median: 23.0–23.4 ms at two rows, 27.1–28.2 at three, 33.2–33.9 at four,
  against 17.9 ms for one ([batching.md](batching.md) estimated 23.2, 27.8 and 32.4).
- **Throughput.** Four requests together made 1,056 tokens in 11.49 s from the first submit, their
  prefills included: 91.9 tok/s, each request ~28–30 tok/s while all four ran. One after another,
  alone, with drafts, they decode at 93.4 tok/s. A batch has no drafts: it buys fairness and first
  tokens, not tokens per second.
- **First tokens.** X (1k, 600 tokens) decodes; Y (16k) comes at X's 30th token, and Z (2k) at Y's
  first. Alone they take 0.32, 1.84 and 0.40 s.

  | | X | Y | Z | done: X, Y, Z |
  |---|---|---|---|---|
  | one at a time (B1 with `--slice 0`, engine 51f08122) | 0.33 s | 7.12 s | 2.12 s | 5.9, 9.5, 12.9 s |
  | one slot, turns of 64 | 0.32 | 1.91 | 0.40 | 13.5, 8.2, 10.7 |
  | four slots | 0.32 | 1.91 | 0.40 | 13.2, 10.8, 10.8 |
  | four slots, `--slice 0` | 0.32 | 1.92 | 0.40 | 14.0, 10.8, 10.8 |

  Staggered (B at A's 30th token, C at B's first, D at C's), B's first token came 0.09–0.11 s later
  than alone and C's and D's as alone.
- **Again at the branch's head** (engine 51c2dab0: B2 with master's B4 GEMV/MoE half). The batches'
  reference hashes, serve_test `--ced` (17 of 17), slice_test and batch_test (0 failed) and the
  planned stop pass again. One session alone gives every reference hash, at 18.19, 9.22 and 17.98 ms
  per token (1k at k = 1, with drafts, 16k CED). The 1k run at k = 1 had a slow tail, not a slower
  step: its median token took 17.88 ms (17.80 before), its p90 19.01 ms (18.23).
  - The tail was a one-off. An ABBA against 33de65fd (B2 before master's half), 1k at k = 1, two runs
    each: 17.91 and 17.91 ms per token (median 17.80, p90 18.26), against 17.98 and 17.93 (median
    17.83–17.87, p90 18.26–18.29).
  - The golden modes' first token after prefill came 9–16 ms later than with 33de65fd (37.5–44.2 ms
    against 28.6–28.7), once per process: they capture a row count's graph on first use. serve
    captures its graphs at startup. Six requests in a row in one serve process gave the same first
    tokens with both builds: the first request's in 342 ms (382 with 33de65fd), the later five's in
    328.8 ms (328.1), median.
- **With drafts across sessions** (B3 and B4's runner half, engine ee748c1, 2026-10-13;
  [batching.md](batching.md#measured-2026-10-13-engine-ee748c1-on-the-integration-branch)). With
  `--slots 4`: batch_test (its two drafted scenarios included) 0 failed, serve_test `--ced` 17 of 17,
  slice_test 0 failed; the reference hashes hold in drafted batches (`--mode batch --spec`: 1k ×2, 1k +
  16k CED + 1k, 1k ×4, the sampled trio) and every session's teacher-forced hash holds through batch
  steps of 12 and 16 rows of 2–4 sessions (`--batch-tf`); alone, 8 of 8 reference hashes. Four streams
  of different text (1k prompts, 512 tokens each) decode 131.3 tok/s in all, three 121.9, two 106.9, one
  alone 97.3; a drafted batch's step is one session's step at the same rows plus about 3 ms a further
  session. One session against master, A/B/B/A: +0.2 % at k = 1, +0.8 % with drafts (two runs each).

## Launch and stop

```
cmake --build engine/build -j                       # engine/build/server/dsgb10_rank
engine/server/run4.sh --mode prefill                # prose golden, free-running
engine/server/run4.sh --mode prefill --inject       # every block fed golden inputs
engine/server/run4.sh --mode generate               # greedy after 39 prompt tokens
engine/server/run4.sh --mode bench --golden /data/dsgb10/golden/long --ctx 16384 --ks 1,6
engine/server/run4.sh --mode decode --golden /data/dsgb10/golden/long --ctx 16384 --steps 256 --big
engine/server/run4.sh --mode tf --golden /data/dsgb10/golden/long --prompt 4096 --limit 4608 --big
engine/server/run4.sh --mode decode --golden /data/dsgb10/golden/long --ctx 1024 --steps 256 --big --spec
engine/server/run4.sh --mode spectf --golden /data/dsgb10/golden/prose --prompt 39   # drafts vs engine/dspark's reference
engine/server/run4.sh --mode tf --golden <tf golden> --prompt 8680 --limit 8936 --big --ced   # CED prefill
OWNER=me engine/server/run4.sh --mode serve        # the server (CED turns and drafts; --exact: all exact, --no-spec: no drafts); then the front-end:
frontend/target/release/dsfront serve --listen 127.0.0.1:18893 --engine-socket /tmp/dsgb10-engine.sock \
    --api-key-file ~/dsgb10-data/dev-api-key
python3 engine/server/tests/serve_test.py           # the protocol and exact reuse, against an --exact engine
python3 engine/server/tests/serve_test.py --ced     # CED's turn rule, against the default engine
python3 engine/server/tests/serve_test.py --speed   # cold prefill 1k..16k and decode through the server
engine/server/run4.sh stop                          # rank 0 first (it stops the others), then whatever is left
```

`run4.sh` takes the ring lock (the fleet file's `RING_LOCK`, `rank 0's /data/dsgb10/ring.lock`, holder in `ring.owner`; set `OWNER`)
and waits for it: one big job on the boxes at a time ([findings.md](findings.md#environment-gotchas)).
While the service holds the ring, a waiting run borrows it once the service has been idle for
`DSGB10_YIELD_IDLE_S` (10 minutes by default), and is preempted (exit 76) when requests come ([Borrowing
the ring](#borrowing-the-ring)); other jobs take the lock with `ring.sh`. A server holds
the lock for as long as it runs, so stop it when idle. Before it starts the ranks,
`run4.sh` drops the model files' page cache on every box (`engine/server/tools/evict_cache.py`, given each rank's files from the fleet file; no root needed); a
service start script (M9) has to run the same eviction first, or the free memory that sizes the
session store can be a few GiB. `run4.sh stop` sends SIGTERM to rank 0 only and waits up to 30 s: a
server ends the request in flight (its DONE is an error, "the server is stopping"), keeps its
sessions for the next start when `--keep-gib` is above 0 ([Kept sessions](#kept-sessions)), sends
`kCmdStop`, and all four ranks leave the ring together; then whatever is still running on any box is
killed. Killing a worker first
failed the others in the middle of that (`socket read: peer closed`, a core dump, in every server log
before 2026-10-12). Workers get a timeout a minute longer than rank 0's for the same reason. `--big`
sends the prompt of the
golden modes through the prefill path instead of 8-row steps. `--ring-cap` is one rank's largest
contribution to a collective (default 4 MiB, comm's own 1 MiB); the prefill's bulk collectives
pipeline pieces of a quarter of it.

`run4.sh` reads the fleet file (`FLEET`, by default `fleet/homelab.conf`), copies the binary and the file, as `fleet.conf`, to
its `BIN_DIR` (`~/dsgb10/bin/`) on each box, starts ranks 1–3 detached by SSH (`RANK<r>_SSH`; `setsid nohup
timeout $((TIMEOUT + 60))`, a minute past rank 0's `TIMEOUT`, default 3600 s) and rank 0 in the foreground, then prints the workers' log tails;
logs are `LOG_DIR/dsgb10_rank.<rank>.log` (`~/dsgb10/logs/`). `WRAP="nsys profile …"` runs rank 0 under a
profiler (`WRAP_RANK=r` rank r instead); bench mode opens the capture range on every rank at the
same step (`kFlagProfile`). The step breakdowns below come from:

```
N=/usr/local/cuda-13.0/bin/nsys
WRAP="$N profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -s none -f true -o /tmp/p" \
  engine/server/run4.sh --mode bench --golden /data/dsgb10/golden/long --ctx 1024 --ks 1,6 --steps 40
$N export -t sqlite -f true -o /tmp/p.sqlite /tmp/p.nsys-rep
python3 engine/server/tools/kprof.py /tmp/p.sqlite    # per step: each kernel class's exclusive and busy time
python3 engine/server/tools/kwall.py /tmp/p.sqlite    # step walls; each collective's p10 / median / p90
```

Modes: `prefill` (the golden sequence in chunks of `--chunk` rows, every layer compared, logits at
every position), `tf` (teacher-forced decode of `--k` rows after `--prompt` tokens), `generate`
(`--gen` tokens, greedy or `--temp`/`--seed`), `bench` (`--ctx` positions of prefill, then `--steps`
steps for each k of `--ks`; median, mean, min and p90 of the host and GPU step times and of the
broadcast, dropping each k's first 3 steps), `spectf` (the drafter teacher-forced on a golden run: a
step per position after `--prompt` − 1 tokens, drafting from the golden next token; each step's
drafts, and how many lead with the golden tokens; `--big` takes the prompt through the prefill path).
`--spec` turns on DSpark drafts in decode and generate; serve runs them unless `--no-spec`, since they
change the speed and never the tokens (`--spec-drafts`, `--spec-thresh`, `--spec-cost`, `--spec-gate`,
`--spec-row-ms`:
[Drafts](#drafts-dspark)). `--draft-fp8` runs the drafter's head GEMV on an FP8 copy of the head, 0.75 ms
less per step; serve does unless `--draft-bf16` ([The draft pass](#the-draft-pass)). `--profile` makes
a decode loop nsys's capture range ([specprof.py](../engine/server/tools/specprof.py): the spec steps'
draft pass by kernel). `--setenv NAME=VALUE` sets a variable in every rank's environment (experiments'
knobs). `--layers N` / `--first N` run only blocks [first, N) (half-model tests with injection),
`--no-graphs` runs every step eagerly, `--no-engram` skips the store.

### As a service: service.sh

[service.sh](../engine/server/service.sh) runs the engine and the front-end as a service: the four
ranks in serve mode and dsfront, under a supervisor on rank 0 that restarts what fails. It installs no
systemd unit and changes no system configuration. The supervisor is a detached bash process, started
and stopped by hand, and not brought back by a reboot. Run it on
rank 0:

```
engine/server/service.sh start      # returns once it serves (or failed); START_WAIT=0 returns at once
engine/server/service.sh status     # exit 0 serving, 1 starting / restarting / lent / not answering, 3 not running
engine/server/service.sh stop       # rank 0 first (it stops ranks 1-3), then the front-end
engine/server/service.sh restart    # stop, then start with this environment's settings
engine/server/service.sh stop --forget   # stop, then remove the kept sessions on every box and dsfront's table
engine/server/service.sh forget     # the same, while stopped
DSGB10_PORT=18894 DSGB10_ENGINE_ARGS="--max-ctx 131072 --exact" DSGB10_NOTE="until 18:00" engine/server/service.sh start
```

Settings come from the environment of `start`, which writes them to `run/settings` for `status` and
`stop`:

| variable | default | |
|---|---|---|
| `DSGB10_ADDR`, `DSGB10_PORT` | `127.0.0.1`, `18893` | the front-end's addresses and port: one or more specific IPv4 addresses, separated by spaces or commas (`"127.0.0.1 <address>"` in production: loopback and rank 0's private-network address), each passed to dsfront as a `--listen` of its own; `status` checks `/health` on every one. 18893 is a test port in the 18xxx range of rank 0's own services; 8888 is production's, taken only with `DSGB10_PROD=1` |
| `DSGB10_KEY_FILE` | `~/dsgb10-data/dev-api-key` | the API key, read by dsfront from the file. The file must be readable by its owner only; the key is on no command line |
| `DSGB10_ENGINE_ARGS` | `--max-ctx 262144` | the engine's arguments after `--mode serve --socket …`; the rest are serve's defaults (CED, drafts with the FP8 drafter head) |
| `DSGB10_FRONT_ARGS`, `DSGB10_NOTE` | | more dsfront arguments; text for `ring.owner` (how long the service will hold the ring) |
| `DSGB10_BIN`, `DSGB10_FRONT_BIN` | `engine/build/server/dsgb10_rank`, `frontend/target/release/dsfront` | copied at start to `~/dsgb10/service/bin`, and from there to the workers once the service has the ring, so a later build does not change a running service |
| `MIN_AVAIL_GIB` | 100 GiB | the MemAvailable every box needs before the engine starts |
| `DSGB10_YIELD_IDLE_S` | 600 s | the ring is lent to a job that waits for it once dsfront has had no request for this long ([Borrowing the ring](#borrowing-the-ring)) |
| `DSGB10_HOLD_S` | 600 s | while the engine is down (lent, or restarting), dsfront holds a request this long, then answers 503 with `Retry-After: 60` |
| `DSGB10_KEEP_GIB` | 4 | an engine stop (`stop`, `restart`, a lend) keeps the sessions for the next start: those in the disk tier, and those in device memory, the most recently used first, up to this many GiB written at the stop. 0: nothing is kept, nor taken back ([Kept sessions](#kept-sessions)) |
| `FLEET` | `fleet/homelab.conf` | the fleet file: the workers, the ring lock, `SERVICE_DIR` and each rank's files. It is read from the environment of every command, `stop` and `status` too, not from `run/settings`; `start` copies it beside the engine on every box as `bin/fleet.conf` |
| `DSGB10_HOME`, `DSGB10_MOCK` | the fleet's `SERVICE_DIR` (`~/dsgb10/service`), 0 | a test instance: `DSGB10_MOCK=1` runs dsfront's mock engine in place of the four ranks, on rank 0 alone, and touches no rank, worker or page cache. It must have its own `DSGB10_HOME` and `RING_LOCK` (the script refuses otherwise) and a free port, set for every command, `stop` and `status` too |

What it does:
- **The ring.** The supervisor takes the ring lock and holds it while it runs, with `ring.owner`
  saying `service` (and `DSGB10_NOTE`). A start that finds a job on the ring waits for it as during
  a lend, with dsfront up ([The start](#borrowing-the-ring)). Rank 0 and the front-end do not inherit
  the lock's file descriptor. So a supervisor that dies releases the ring, and the next job's stray
  reap clears its ranks. `service.sh stop` also cleans up after a dead supervisor, but only if it can
  take the lock (otherwise the ranks may belong to another job).
- **Each start of the engine**, like run4.sh:
  - stray ranks are stopped;
  - the model files' page cache is dropped on every box (`evict_cache.py`), so that the free memory
    that sizes the session store is not a few GiB short;
  - every box must then show `MIN_AVAIL_GIB` of MemAvailable, or the start counts as a failure. An
    OOM can hang a box, so another model left loaded must stop the start;
  - ranks 1–3 start detached on the workers, then rank 0 on rank 0.
- **OOM.** Everything started over SSH on rank 0 inherits sshd's `oom_score_adj` of −1000, which makes
  it unkillable by both OOM killers. The supervisor writes 0 to its own,
  and the ranks and the front-end inherit that. (run4.sh does not, so the runs above start rank 0 at
  −1000.)
- **The front-end** starts at once, before the supervisor takes the ring, and answers `/health` with
  503 until the engine connects. It reconnects to a restarted engine by itself; requests in flight
  when the engine goes end with an error. A request that comes while the engine is down waits at
  dsfront's gate, up to `DSGB10_HOLD_S`, and then gets 503 with `Retry-After`
  ([Borrowing the ring](#borrowing-the-ring)).
- **Failures.**
  - Rank 0 is the engine's health: it exits when a peer dies or a step hangs (the step watchdog).
  - A worker that dies between requests would fail rank 0 only at the next request, so the supervisor
    checks ranks 1–3 every minute and stops rank 0 (SIGTERM; `kill -9` 30 s later) when one is gone.
  - Rank 0 must listen within 600 s of its start (`READY_S`).
  - The engine restarts, all four ranks, 10, 30, 60, 120, then every 300 s after a failure. A run that
    lasted 10 minutes resets the delay. The front-end restarts alone, after 2, 5, 10, 30, then every 60 s.
  - More than 5 failures within an hour (`MAX_FAILS`, `FAIL_WINDOW`): the supervisor stops
    everything, frees the ring and stays `failed`.
- **Network and secrets.**
  - The script refuses `0.0.0.0`, and dsfront refuses
    an unspecified address too: `0.0.0.0`, `::` and `::ffff:0.0.0.0`, in any of its `--listen` options. It binds
    every address before it serves any, and one that cannot be bound stops the start.
  - Port 8888 is production's (M9, the owner's go-ahead on 2026-10-12): the script takes it only with
    `DSGB10_PROD=1`.
  - The engine's socket is `~/dsgb10/service/run/engine.sock`, mode 0600 in a 0700 directory. Nothing
    of the engine listens on a TCP port.
- **Files.** On rank 0, `~/dsgb10/service/run` holds the state, settings, pids, the engine's socket
  and dsfront's admin socket (`front.admin`), and between a stop and the next start dsfront's session
  table (`front.table`). The kept sessions are in the engine's spill directory on every box
  ([Kept sessions](#kept-sessions)).
  `~/dsgb10/service/logs` holds `service.log` (the supervisor: every start, exit, failure and state),
  `rank.0.log` and `front.log`; each worker has `logs/rank.<r>.log`. Every (re)start rotates its log,
  and on rank 0 a log past 64 MiB is rotated in place, copied and truncated (the processes append).
  Five old logs are kept. Rank 0 writes three lines per request (~550 B), dsfront two.
- **Where a turn started.** Both logs say it in counts, never text or ids, under the same request
  number. dsfront's `req N plan:` names the session the request continues in place, the one it
  forks from, or a new one, with the bytes of text and the ids it shares with the best match.
  Rank 0's `serve: req N restart:` says where the state came from, and for a cold start why:
  - no base named: the best match shared under 64 bytes;
  - a session or base the engine does not hold: lost at an engine restart, or evicted;
  - a common prefix short of the base's first restart point.

Checked on 2026-10-12 (the [soak](#soak-2026-10-12) ran under it):
- `start` waited 46–53 s for the ring while another job held it.
- The engine was ready 44 s after its ranks started.
- rank 0, dsfront and the supervisor run at `oom_score_adj` 0; neither rank 0 nor dsfront holds a lock
  descriptor.
- `kill -9` of rank 3 on rank 3, with the engine idle:
  - the minute's check saw it 13 s later and stopped rank 0;
  - rank 0 aborted on `socket read: peer closed`, rc 134 (apport keeps no core: `ulimit -c` is 0);
  - the engine was back 10 s later and serving 74 s after the kill.
- `kill -9` of dsfront: restarted 2 s later.
- `stop` took 3 s: rank 0 exited 0, the workers logged `done`, no rank was left on any box, and the
  next job took the ring the same second. After the hour's soak it took 4 s, with the same result.
- `stop` with requests in flight, twice, each followed by a start:
  - first a decode 10 s in, with a 120k-token prefill queued behind it; then, once the next start had
    answered a short request, that prefill 7 s in, with a decode queued behind it;
  - the request in flight ended at its next step or chunk. Its stream ended with an error event, "the
    server is stopping", then `[DONE]`. The queued one got "engine connection lost" from dsfront,
    then `[DONE]`;
  - `stop` took 3.5 s each time. No rank, dsfront, pid file or socket was left on any box, and the
    ring lock was free.

  Rank 0 used to end a request cut short by its own stop as cancelled, which dsfront passes on as a
  finished answer (finish "stop"). Now it sends an error (`Io::stopping` in serve.cpp); a cancel by
  the client is still a cancel.

Production (M9) has served since 2026-10-12 06:39 on 127.0.0.1:8888 (since 06:48 on rank 0's private-network address too),
from the pinned worktree `~/dsgb10/prod` (`DSGB10_PROD=1`, the kit's API key). Since 11:14 that was
master 9d2bea4 with kept sessions (engine c2dd1421, built from 29ae206); since 12:59 it is cbb5f61,
with B1, the ring queue and the SSE keep-alive (engine 51f08122 and dsfront d5ba1e6a, built from fd3d0e4, in
`~/dsgb10/prodbin`; [operations.md](operations.md) has the command). Since 13:13 its service.sh is 426c87b's: a start behind a busy ring
serves at once (on 2026-10-12, :8888 had no listener from 11:09:11 to 11:13:32 and from 12:57:58 to 12:58:43 before it).
Since 2026-10-13 04:13 it runs release candidate 5 (c247a78; engine b352b870…, dsfront ce116cea…):
[Where it stands on release candidate 5](#where-it-stands-on-release-candidate-5-2026-10-13).

Each deploy is checked on both addresses with the API key: the model list, a short answer ("pong"), thinking at max, a
tool call, streaming, and 401 without the key. At the 12:59 deploy a pong sent while a 600-token answer decoded came
back in 0.17 s (B1; f9b2ed8). The address list's rules (every address bound before any serves; 0.0.0.0, :: and
::ffff:0.0.0.0 refused; /health 200 on each for `status`) were tested against the mock engine on 127.0.0.1 and
127.0.0.2: 25 of 25 (cefed2a). Still open:
- Nothing brings the service back after a reboot. A systemd unit would change system configuration,
  so that is the owner's call.
- The engine time-slices requests (B1, since 12:59) but does not batch them: one session's rows per
  step. Under the soak's three clients a small turn used to wait about 3 s for its first token. Several
  sessions per step (B2) is merged but off (`--slots 1`): it buys fairness and first-token time, not
  throughput, and drafts across sessions (B3) are estimated and parked, as the owner runs several agents
  only occasionally ([batching.md](batching.md)).
- After a worker dies, the engine serves again about 74 s later. The requests in flight fail.
- The hours-long soak, on a free ring. It should also show where the front-end's memory levels off.

### Borrowing the ring

**Fixed 2026-10-12 (11f8343).** A request that waited during a lend and left before the engine was back
(a pi request, after 55 s) did not count as activity: dsfront's gate updated its idle clock only when a
request passed or ended. At 10:03:55 the service lent the ring again 2 s after its engine was ready. Now a
request's arrival and its leaving both count. service.sh also lends no sooner than `DSGB10_YIELD_IDLE_S`
after the engine's last start (from its next restart). Production's dsfront was swapped in place at 10:12
(the supervisor restarts it; the engine stayed up). 38/38 against the mock.

Development jobs borrow the ring from the service while it is idle. Requests come first: a request
that comes during a lend preempts the job.

```
OWNER=me engine/server/run4.sh --mode bench …     # waits for the ring; borrows it once the service is idle
OWNER=me engine/server/ring.sh <command> [args]   # the same for any other job that needs the boxes
PREEMPT=0 OWNER=me engine/server/run4.sh …        # not preempted: requests wait for its end
engine/server/service.sh status                   # the state (lent?), dsfront's gate, the jobs waiting
```

- **Asking.** A job that finds the ring busy registers a want, `/data/dsgb10/ring.wants/<pid>`, and
  then tries the lock every 2 s, for up to `LOCK_WAIT` (7200 s) ([ringlock.sh](../engine/server/ringlock.sh)).
  - It holds the want file flock'd while it waits. A want whose lock is free belongs to a dead job,
    and the supervisor deletes it.
  - Jobs take the ring in the order they asked for it. A want's second line says when it was made
    (ns since the epoch) and its pid. A job tries the lock only while no live want older than its own
    is queued. A job that finds the lock free while wants are queued queues behind them ("jobs wait
    for the ring before this one").
  - Until 2026-10-12 whoever polled first won. A driver that ran its parts back to back took the ring
    again and again, ahead of jobs that had waited longer. The order holds only between jobs whose
    ringlock.sh has it: a job from an older checkout neither defers nor has a place in the queue.
  - run4.sh asks this way. Any other job uses [ring.sh](../engine/server/ring.sh), which runs its
    command holding the lock. A bare `flock` on the lock registers no want, so it waits for as long
    as the service serves.
- **Lending.** The supervisor looks for wants every 2 s.
  - It lends the ring only once dsfront has had no request for `DSGB10_YIELD_IDLE_S` (600 s): none
    in flight, none waiting, and the last one ended at least that long ago.
  - The check and the hold are one step inside dsfront. `hold IDLE_S` on its admin socket
    (`run/front.admin`, mode 600) succeeds only when the service is that idle, and from then on new
    requests wait. A request in flight is never cut: no hold succeeds while it runs.
  - The supervisor then stops the engine the way `stop` does: rank 0 first, then strays on every box.
    It unlocks the ring (`flock -u`), and the state becomes `lent`. An engine that is still starting
    finishes its start first.
- **Holding requests.** While the engine is down, lent or restarting after a failure, dsfront holds
  each new request at its gate (`frontend/src/gate.rs`).
  - The request goes on once the engine is connected and the gate is released.
  - After `DSGB10_HOLD_S` (600 s) it is answered 503 with `Retry-After: 60`.
  - A streamed request (`"stream": true`) that has to wait is answered at once instead, so that a
    client's timeout sees bytes: 200 and an SSE comment, `:waiting for the engine (the ring is lent
    to a development job), up to 600 s`. A comment follows every 5 s without an event, as it does on
    every stream (a long prefill before the first token). The answer then streams as usual. Past the
    hold limit, or at dsfront's stop, the stream ends with an error event (`"code": 503`) and
    `[DONE]`. A client that goes away while it waits is counted out at once.
  - While the gate is held, `/health` answers 503, "the ring is lent to a development job".
  - When dsfront stops, it answers the requests it holds at once, with 503.
- **Back.** When the job releases the lock, the supervisor retakes it within 2 s, releases the gate,
  and starts the engine (page cache, memory check, four ranks: about 45 s). Held requests go through
  as soon as dsfront connects.
  - If requests wait during a lend, the supervisor holds `ring.reclaim` flock'd. The job on the ring
    is preempted (below), and no queued job takes the ring before the service has it back: jobs check
    the flag before and after they take the lock.
  - If no request waits and another job is queued, that job goes next, with no engine start in
    between. The service retakes the ring once no job waits.
- **The start.** A `start` or `restart` while a job holds the ring waits for it as during a lend:
  - dsfront starts at once and holds requests, up to `DSGB10_HOLD_S` each. `/health` and a waiting
    stream's comment say the ring is lent.
  - A request that waits takes the ring back: the job is preempted (unless `PREEMPT=0`), and the
    engine starts as soon as the job is gone, before any other job.
  - With no request, the engine starts once no job waits, a driver's held want included.
  - There is no bound on the wait. `start` returns once the engine serves, or after `START_WAIT`
    (900 s) with the state `waiting`, while the supervisor goes on. `stop` meanwhile stops dsfront and
    leaves the job alone.
  - Nothing touches the boxes before the ring is the service's: the binaries are copied to
    `~/dsgb10/service/bin` here at the start, and to the workers before the engine's first start.
  - Until 2026-10-12 the supervisor started dsfront only once it had the ring, and waited at most
    `LOCK_WAIT` (600 s) for it, preempting nothing: nothing listened on the port meanwhile, and past
    `LOCK_WAIT` the start failed.
- **Drivers.** A driver is a script that runs ring.sh part after part, each a few minutes, to stay
  under the hold or preemptible. It gives the ring back between parts.
  - Before held wants, no want was queued in the gap between parts. The supervisor took the ring back,
    started the engine (~47 s) and lent it again `DSGB10_YIELD_IDLE_S` later: about 3 minutes of ring
    per part in which nobody was served.
  - A driver now holds a want across its parts:

    ```
    . engine/server/ringlock.sh
    ring_want_hold "my parts"    # ring.wants/<pid>.held, until ring_want_drop or the driver's end
    for p in 1 2 3; do sleep 5; OWNER=me PREEMPT=0 engine/server/ring.sh bash part.sh $p; done
    ring_want_drop
    ```
  - The supervisor counts a held want only to stay lent: it does not take the ring back while one is
    queued. A held want never makes it lend. While the service serves, each part asks and waits like
    any job.
  - A held want has no place in the queue. Each part waits in turn with its own want, and a job that
    asks during the gap goes first. Leave a few seconds between parts.
  - A watcher subshell holds the want's lock. It removes the want once the driver is gone, even after
    SIGKILL (within 2 s).
  - While a want is held and no part runs, the ring stays lent and idle; a request still reclaims it at
    once. Drop the want before anything long that does not use the ring.
- **Preempting.** ring.sh and run4.sh run the job as a process group of its own, and a watcher
  checks the reclaim flag every second (`ring_run` in ringlock.sh).
  - Once the flag is held, the watcher sends TERM to the group and stops any ranks on every box, in
    `run4.sh stop`'s order: rank 0 first, then whatever is left. While the ring is lent, every rank is
    the job's. After `PREEMPT_GRACE` (10 s) it kills what is left of the group.
  - The job then exits 76: run it again later. It no longer has a want, so it queues afresh behind
    the jobs that wait.
  - The request waits for the detection (up to 3 s), the stop (seconds; at most the grace) and the
    engine's start (~45 s): about a minute, not the rest of the job.
  - `PREEMPT=0` opts out, for a job that must not be cut. It runs to its end while requests wait, and
    past `DSGB10_HOLD_S` (10 minutes by default) users get 503s: keep such jobs shorter than that,
    less the stop and the engine's start (about a minute).
  - A request in flight is still never cut: the ring is lent only while the service is idle, and
    preempting stops the job, not a request.
  - A job inside a job, such as ring.sh around a script that runs run4.sh, runs at once. The lock
    comes with it (fd 9, and `RING_HOLDER` in its environment), and the outer job's watcher stops it
    with the rest.
- **Stopping.**
  - `service.sh stop` while the ring is lent stops only the front-end: the ranks are the job's.
  - `run4.sh stop` refuses while the service holds the ring (`ring.owner` says `service:` and the
    lock is taken): the service's ranks are its supervisor's to stop.
- **Production** runs `~/dsgb10/prod/ds41f-4xgx10-switchless/engine/server/service.sh` (since
  2026-10-12 08:16, 8f3d16b; since 11:14, 9d2bea4, with `DSGB10_YIELD_IDLE_S=120` and
  `DSGB10_HOLD_S=900`: development first, the owner's call). The repo moved into `ds41f-4xgx10-switchless/` (master b9bf35d), and
  its `SRC` is two levels up, so the binaries' default paths follow the move.
  - The dsfront it copies at start must be built from the lending change or later (`cargo build
    --release` in `frontend/`). An older dsfront ignores `--admin-socket`, and the service then never
    lends.
  - Preemption is the job's side. A job run from a checkout with `ring_run` is preempted by any
    supervisor that lends, production's of 08:16 included. A job from an older checkout runs to its
    end.

Tested against the mock engine on 2026-10-12 ([tests/yield_test.sh](../engine/server/tests/yield_test.sh)).
The test instance had `DSGB10_MOCK=1` and its own home, port 18897 and `RING_LOCK`, an idle time of 4 s,
a hold limit of 10 s and a grace of 2 s. `ssh` and `pkill` were stand-ins that only log, and none was
called. All 38 checks passed, in two runs. Two runs between them failed one check each: the check that
the service was untouched compared its state, which changed when other jobs borrowed the ring. It now
compares the service's supervisor and restart count.
- 12 requests over 12 s with a job waiting: nothing was lent. The job got the ring 7.3 s after the
  last request.
- A 2,400-token request in flight (15 s) was not cut. The job waiting behind it started 5.7 s after
  it ended.
- A request that came during a lend preempted the job: three processes, one of which ignored SIGTERM
  and needed the KILL after the grace.
  - The job exited 76, and nothing of its process group was left.
  - The request was answered after 6.5 s.
  - The job queued behind did not go first: it got the ring 7.3 s after that request ended, and ran
    to its end.
- A job with `PREEMPT=0` ran to its end while a request waited past the hold limit. The request got
  503 after 10.1 s, with `Retry-After: 60`, and the next request after the job was answered.
- Two jobs queued with no request: the second started 2 s after the first ended, and the engine
  started once, after both.
- A stale want (its job killed with `-9`) was deleted, and nothing was lent.
- `stop` while lent took 1.0 s. The held request got its 503 at once. The job (`PREEMPT=0`) ran on with
  the lock and ended by itself.
- A ring.sh inside a ring job ran 0.0 s after the outer one got the ring; only the outer one waited.
- The service's supervisor, state and restart count were unchanged.

The queue's order and the drivers' held wants, against the mock on 2026-10-12: 52 of 52 checks.
- A driver held its want, with parts 5 s apart.
  - A job that had waited behind its first part got the ring 1.2 s after that part ended.
  - The second part got it 2.8 s after that job.
  - The ring came back once, after the last part, with one engine start.
- A driver killed with SIGKILL lost its want within 6 s.
- Three jobs ran one after another in the order they asked. The youngest found the lock free between
  the others' polls, and still waited for them.
- A dead want and a held one, both older, held no job back: it got the ring after 0.0 s.

The start behind a busy ring, against the mock on 2026-10-12: 73 of 73 checks, the 17 new ones
among them.
- A start while a job held the ring (8 s), with no request:
  - dsfront answered 0.3 s after the start. Once the supervisor held its gate, `/health` said the ring
    is lent; `ring.owner` stayed the job's.
  - The job ran to its end, and the service served 2.5 s after it, with one engine start.
- A start behind a job of three processes, one of which ignored SIGTERM, then a request and a stream:
  - the request was answered after 7.3 s (the grace: 2 s);
  - the stream got its first bytes at once and two comments while it waited, then its answer;
  - the job was preempted (exit 76); "the ring is ours after 8 s of waiting at the start (the job is
    done; requests wait)", and one engine start.
- A stop while the start waited took 1.0 s. dsfront was gone, and the job ran on with the ring to its
  end, with no engine start.

The holder in `status` while the start waits, against the mock on 2026-10-12: 76 of 76 checks, the 3
new ones among them. The first line used to name the job that held the ring when the wait began. It
now reads `ring.owner` when `status` runs, as the `ring:` line does. A start behind a job (6 s) with a
second one queued: the first line named the first job's pid while it ran, then the second's; one
engine start after the second.

The rank stop, run alone with stand-ins for `bash`, `pkill` and `ssh`: `pkill -x dsgb10_rank` on rank 0
first, then the reaps (rank 0 10 s, each worker 5 s) at once; exit 76.

**On the ring, 2026-10-12, before preemption** (production's supervisor of 08:16, a run4.sh job from
before `ring_run`):
- A job (decode 1k, OWNER=runner: tokens hash febc7dc54ee40a39) asked at 08:23:06.
- The service lent the ring at 08:26:59, 600 s after its last request, and its engine was down 2 s
  later.
- A request sent 20 s into the job waited at the gate. The supervisor held the reclaim flag 2 s later.
- The job ended at 08:27:57, 56 s on the ring, and the service had the ring back 1 s after.
- The engine was up 44 s after that, and the request was answered: 200 after 82 s. Preempted, it
  would have waited for the stop and the start alone.

**Preempted on the ring, 2026-10-12** (run4.sh of 784de56; production's supervisor of 08:16):
- A run4.sh job (decode: 16k prefilled, then 12,000 steps, ~4.5 min alone) had the ring from 08:41:55.
- A request came 73 s in. The supervisor held the reclaim flag the same second, and the watcher
  preempted the job at once.
- run4.sh exited 76 seven seconds later, and no dsgb10_rank was left on any box.
- The service had the ring back 1 s after, and its engine was ready 46 s later. The request got 200
  after 55 s: the stop 7 s, the retake 1 s, the engine's start 46 s. Without preemption it would have
  waited about 3.5 minutes more, for the rest of the job.

### Kept sessions

A planned stop of the engine keeps its sessions for the next start. After a lend, a `restart` or a
deploy of the same build, a conversation's next turn finds its state on disk rather than prefilling
the whole conversation again. A crash keeps nothing.

```
engine/server/service.sh stop --forget                  # stop, then remove what stops kept, on every box, and dsfront's table
engine/server/service.sh forget                         # the same while stopped (refused while it runs: its stop would keep them again)
DSGB10_KEEP_GIB=0 engine/server/service.sh restart      # from then on nothing is kept, nor taken back
```

**The stop.** On SIGTERM (`service.sh stop` and `restart`, a lend, `run4.sh stop`), with `--keep-gib
N` above 0 and a disk tier, rank 0's server ends the requests in flight (paused ones and those
waiting for their turn too: "the server is stopping"), then:
- moves every slot's session's state into the store, as taking its slot for another session would;
- frees the sessions unused for `--keep-days` (7);
- spills the sessions still in device memory to the disk tier, the most recently used first, as far
  as N GiB (the service's `DSGB10_KEEP_GIB`, 4), and frees the rest. That bounds the stop's writes.
  Sessions already on disk stay. When the file is full, a spill drops its least recently used
  sessions, as any spill does;
- sends `kStKeep` with a random generation. Every rank waits for its writes, `fdatasync`s its file
  and writes its index, `rank<r>.<tag>.index`: each session's blocks and their checksums, the store's
  layout, the run id (below), the generation and the time;
- writes the kept sessions' histories, `server.<tag>.index`: each session's id, the tokens its state
  holds, its restart ranges and when it was last used. Same generation. Then the ranks stop together
  (`kCmdStop`).

dsfront saves its session table at its own stop (`--table-file`; the service's is `run/front.table`):
each session's text and token ids, by which a restarted dsfront sends a conversation's next turn to
the same session. During a lend dsfront keeps running, and its table stays in memory.

**The start.** Each rank reads its index and checks:
- the magic, its rank, the run id, the store's layout, and a hash over the records;
- the age: an index written more than `--keep-days` ago is cleared;
- every record: handles unique, blocks on the 64 KiB grain, inside the file, none overlapping;
- the file beside it: a regular file of this user, mode 0600, of the size the index names.

The ranks then share their generations and file sizes (an allgather). If every rank's index checks
out with the same generation, and the file fits every rank's offer, every rank opens its file as it
is. Its sessions become spilled sessions of the disk tier. They come back on demand like any other,
each block checked against its checksum. Rank 0's server takes the histories of the same generation,
and frees any session of the store they do not name. So the front-end's session ids find their
sessions as before the stop. The indexes are removed once read, so a crash from then on starts clean.

Anything else starts clean, and every rank removes the kept files:
- **A new build.** The run id is a hash of the binary's bytes, of the weights' and the Engram files'
  paths, sizes and times, and of the options that change what a request computes (`--max-ctx`,
  `--exact`, drafts, the drafter head's FP8, `--prefill-rows`, `--small-rows`, `--snap-every`,
  `--no-engram`, `--layers`, `--first`, `--ring-cap`, `--draft-head4`, `--draft-markov8`). So a deploy with a new engine binary fails the
  check and starts clean. That is meant: the old build's state need not be the new one's.
- **A crash** (`kill -9`, a dead peer, an OOM) writes nothing. The indexes of the last planned stop
  went when that start read them.
- **A stop that failed partway**: a rank died, or could not write its index. The generations then
  differ, or an index is missing.
- **A damaged or foreign file**, or one older than `--keep-days`.
- **`--keep-gib 0`**: nothing is kept, and nothing is taken back.

A kept block that comes back different from its checksum stops the engine (`state: session …, kept
by an earlier run, came back different from the disk`), so that no turn continues from a wrong state.
The next start is clean.

**What is kept, where, and for how long.** The kept files hold what the conversations left: the token
histories are the conversations themselves, and dsfront's table holds their text.

| file | where | holds |
|---|---|---|
| `rank<r>.service.spill` | every box's spill directory (`RANK<r>_SPILL`): `/data/dsgb10/spill` on rank 0, `~/dsgb10/spill` on the others | rank r's slice of each session's state: compressed caches, window and end snapshots. It is the disk tier's file (`--disk-gib`, 32 GiB), reused as it is |
| `rank<r>.service.index` | beside it | each session's blocks and checksums |
| `server.service.index` | beside rank 0's, on rank 0 | each session's id and token history |
| `front.table` | `~/dsgb10/service/run` on rank 0 | each dsfront session's text and token ids |

- **Private.** Every one is mode 0600, in a directory of mode 0700. The engine makes its spill
  directory 0700, and refuses a kept file that is not a regular file of its user, mode 0600; dsfront
  does the same with its table. Each is written under a temporary name (`.new`, created exclusively,
  following no link), synced and renamed into place, so it is whole or absent.
- **Never in a repository.** The engine refuses a disk tier inside a git work tree (a `.git` in the
  directory or above it: `no disk tier: … is in a git tree`). The service's run directory is outside
  any checkout.
- **Never in the logs.** They give counts, sizes and times, never tokens or text.
- **The service's own.** The service runs the engine with `--disk-tag service`, so its files are
  `rank<r>.service.*` and `server.service.index`. A development run uses no tag (`rank<r>.spill`) or
  one of its own, and never touches them.
- **How long.**
  - The indexes and dsfront's table last until the next start, which reads and removes them.
  - The sessions last as long as the disk tier keeps them. A session unused for `--keep-days` (7) is
    neither kept at a stop nor taken back at a start; dsfront's `--table-days` is 7 too. When the
    file is full, its least recently used sessions are dropped.
  - An index written more than 7 days before the start is cleared whole.
  - `forget` removes them all.
- **Forgetting.** `service.sh forget`, or `stop --forget`, removes the four kinds of file on every
  box. It refuses while the service runs, since the service would keep them again at its stop. It
  says so when files are left (a box unreachable): run it again. By hand, with the service stopped:

  ```
  rm -f /data/dsgb10/spill/*.service.* ~/dsgb10/service/run/front.table*; for h in 10.0.0.4 10.0.0.3 10.0.0.2; do ssh $h 'rm -f ~/dsgb10/spill/*.service.*'; done
  ```

Tested against the mock engine ([tests/yield_test.sh](../engine/server/tests/yield_test.sh), 42
of 42): dsfront saved its table at the stop (mode 600) and took it back at the next start, and the file
was gone; `forget` was refused while the service ran, and `stop --forget` stopped it and removed the
table. The table's round trip and its refusals (damaged, mode 0644, too old) are dsfront's unit tests.

**Measured on the ring, 2026-10-12.** The engine was f737e73's build (sha256 9203dcdf…, which master
d7b33ce builds the same), driven by `tests/kept_test.py` through rank 0's socket (`--session-gib 0.4
--disk-gib 8`). Its files had their own tags, kref and ktest, and were wiped afterwards. Production's
request preempted the first window after three steps at 10:02:58; the rest ran in the next lend.
- **The stop.** Three sessions were kept: A (64.6k positions), B (20k) and C (2k). Two were written at
  the stop (0.13 GiB); the third was on disk already. The ranks' files were synced and indexed in 0.12 s,
  and the histories (338 KiB) were written in 2.4 ms. The whole stop took 5.1 s; the window's six stops
  took 4.8–5.5 s.
- **The start.** Rank 0 put the kept sessions in its store in 196–201 ms: the index read and checked,
  and the file opened as it is. The run id took 23 ms; it hashes the binary and the weights. The
  server's histories took 0.1–0.5 ms. The engine was up in 46.7 s, the same as a clean start
  (46.2–47.2 s).
- **A turn after the restart**, against the same turns on one engine with no restart (kept_test's
  after-phase, client TTFT):

  | turn | after the restart | no restart | cold |
  |---|---|---|---|
  | A: 64.6k + 300 (CED) | 0.347 s | 0.230 s | 6.89 s |
  | B: 20k + 40 (exact) | 0.224 s | 0.227 s | |
  | C: 2k, regenerated | 0.358 s | 0.358 s | |
  | D: a fork of B at 10k | 0.454 s | 0.428 s | |

  - All four turns were equal to the run with no restart, bit for bit: tokens, log-probabilities, and
    the prefix each reused.
  - A came back from disk in 108 ms (126 ms without the restart, where it had been spilled too). The
    restart's extra 0.12 s is in A's prefill, 347 ms against 229. A was the new process's first request.
- **The bound.** A stop with `--keep-gib 0.05` spilled the two most recently used sessions (0.05 GiB)
  and freed the third past it. The session already on disk stayed.
- **Clean starts.**
  - Another run id (`--max-ctx 131072`): `none reused (…: of another engine binary, weights or
    options)`. C came back cold.
  - `kill -9` of rank 0: no index was left, because the start had read and removed it. The next start
    was clean, and C cold.
  - Too old: `--keep-days 0.0001` (8.6 s) against an index written 68 s before. Clean, and C cold.
  - The next planned stop and start with the usual settings: C came back warm (cached 1,024, the
    snapshot a regeneration restarts from).
- No file was left on any box after the wipe.
- The reference hashes from the same binary: decode 1k febc7dc54ee40a39 with and without drafts, 16k
  through CED 81f772ab79a950e4, and long28k with CED (tf) 14650fb0739d0383 (until 2026-10-13 the hash's
  seed, see [Golden check](#golden-check); now 8f08a153d415c5c8).

## Golden check

The golden records (golden/README.md) hold, from model.py: per-layer tensors at saved positions
(`positions.npy`), index top-k and layer 20's pool at record positions, the top-32 logits at every
position, the full logits at the last few, and the token sequence. Rank 0 compares as it runs
([golden.h](../engine/server/golden.h)); steps that need it run eagerly with:

- `kFlagDump`: every block's `engram_out`, `attn_in`, `attn_out`, `mid`, `ffn_in`, `ffn_out`, `out`,
  the six mixes, routing and index choices, relative L2 error per position.
- `kFlagLogits`: the full logits all-gathered to rank 0: top-1 against golden's argmax, top-5
  overlap, the largest difference over golden's top 32, accuracy against the next token, and
  golden/cmp_runs.py's measures where the full logits exist.
- `kFlagInject` (`--inject`): each block's input stream and pre mix replaced by golden's, so every
  block is checked on its own instead of accumulating its predecessors' differences;
  `--inject-sub` also replaces the FFN's input.
- `--onset P`: the per-layer table again for positions ≥ P, and top-1 / next-token accuracy per
  window (4096 positions, split at P; the first 4096 also in doubling windows). `--per-position`:
  each saved position's error in every fifth block's output and in the final hidden state.

Logits can only be compared statistically end to end (golden/README.md): two correct bf16
implementations already disagree at near-ties. Exactness is checked per block with `--inject`.

Bench mode prints a hash of every chosen token and its winning logit: graph replays and eager steps
give the same hash, bit for bit. With `--big` (every reference hash) the prompt goes through the
prefill path, which the hash does not cover, not even the prompt's first token: it covers the rows
after it (decode steps, or `--mode tf`'s teacher-forced positions). A `--mode tf` run whose `--limit`
equals `--prompt` (long28k with CED) has no such row: until 2026-10-13 it hashed nothing and printed
the hash's seed, `14650fb0739d0383`, whatever the logits
([review/correctness.md](review/correctness.md), M1). Since b47c510 such a run hashes the prompt's
sample instead: its token, the bits of its logit and logprob, and its top 20 logprobs' ids and bits.
long28k with CED now gives `8f08a153d415c5c8`, which replaces the seed wherever this page cites it, and
without `--ced` the same run gives another hash, `87f42fa9f94f13de`
([golden/README.md](../golden/README.md#checking-a-build) has the commands; `golden/refhash.sh` runs
them).

## Measured (2026-10-11)

Full model on all four boxes. During the timing and profile runs nothing else ran on any GPU
(sampled every 10 s on all four).

### Correctness

These are the engine of 2026-10-11, before ec22622 moved the decode step's arithmetic on purpose (the
mixes summed in rank order, a new tf hash). Release candidate 5 gives slightly different figures, inside
the same envelope: [review/correctness.md](review/correctness.md).

- **Prose, each block fed golden's inputs** (`--inject`, run as layers 0–19 and 20–39 while rank 2
  was short of memory): block output relative L2 0.5–1.3e-2 per layer (max 0.15, only at routing
  flips); with the FFN's input injected too, attention 3–5e-3, MoE 2.7–3.0e-3, Engram 6.5e-6 (layer
  1) and 1.1e-6 (layer 14), mixes ≤ 5e-3. Routing ≥ 806/810 per layer, index top-k identical at all 8
  index layers, top-1 134/135 (the miss has golden margin 0.002).
- **Prose, free-running**: top-1 124/135 with no clear flips (both margins > 1 in log-probability);
  over the 96 completion positions 88/96, \|Δ log p\| over golden's top 5 mean 0.60, KL 0.059.
  model.py's own numerics variants differ from it as much (golden/README.md: split-K 87/96 with one
  clear flip, g128 84/96, bf16 86/96). Against SGLang's tokens: ours 105/134, model.py 102/134.
  Teacher-forced decode at k = 1 and k = 6 gives the chunked prefill's logits (every measure
  identical over the 96 completion positions), and graph replays give the eager steps' tokens and
  logits bit for bit (bench hash).
- **Generation**: SGLang's first 12 tokens, then model.py's token where model.py itself leaves
  SGLang's sequence.
- **Long transcript** (28,489 tokens, `--onset 16864 --per-position`), free-running:

| positions | top-1 vs golden's argmax | clear flips | argmax = next token: ours / model.py |
|---|---|---|---|
| 0–511 | 0.90 (460/512) | 9 | 0.65 / 0.62 |
| 512–1,023 | 0.83 | 8 | 0.56 / 0.56 |
| 1,024–4,095 | 0.71 | 200 | 0.57 / 0.57 |
| 4,096–16,863 | 0.80 | 521 | 0.68 / 0.68 |
| 16,864–28,488 (after the onset) | 0.70 (8,079/11,625) | 602 | 0.44 / 0.44 |
| all | 0.75 | 1,340 | 16,210 / 16,122 of 28,488 |

  The engine reproduces the collapse: next-token accuracy falls from 0.68 to 0.44 as
  model.py's does, window by window, and over the whole sequence it predicts the text as well as
  model.py (golden/README.md has the cause: layer 1's Engram in this context). Free-running, the
  residual streams drift apart over thousands of positions: block-output error at layer 39 is 0.01
  at position 0, 0.2–0.3 by positions 256–1,024 and 0.05–1.0 after, varying with the content rather
  than growing steadily, and top-1 agreement falls from 0.90 to 0.70–0.83. Fed golden's block inputs
  at the last 504 positions (`--inject`; the only whole chunks golden saved, with the engine's own
  caches before them) it is back to 0.94 (476/505) with no clear flips; per block, attention differs
  by 3–15e-2 and the block output by 2–7e-2 there, from the drifted caches, and index top-k keeps
  0.83–0.97 of golden's. How far model.py's own numerics variants drift apart at this length is not
  measured (golden/README.md compares them on prose).

### Prefill path

The prefill path against the 8-row step path on golden/long: the first 4096 positions prefilled,
then 512 teacher-forced steps with every check (`--mode tf --big`):

| prompt through | top-1 vs model.py (positions 4096–4607) | argmax = next token: ours / model.py |
|---|---|---|
| 512 steps of 8 rows | 392/512 | 336 / 325 |
| chunks of 4096, 2048 or 1001 (engine/attn's attention, the step's mHC) | 383, 383, 400/512 | 313, 316, 339 / 325 |
| chunks of 4096, 2048 or 1001 (`sparse_attn_p`, `hc_boundary_p`): the same logits | 382/512 | 345 / 325 |
| the same, with the decode step's mixes summed in rank order (ec22622) | 385/512 | 338 / 325 |

`--mode tf` prints a hash of the teacher-forced logits, which is the bit-identity check for changes
that should not move the arithmetic: every chunking of the prompt gives the same one. On master since
ec22622 it is `842278b28d87dad0` (4096 prefilled, 512 teacher-forced, exact mode); before that it was
`39b5c5c52a00dd73`. A change that moves it on purpose says so and gives the new value.

Per-layer errors after the prefill match the step path's (layer 2 attention 4.8–5.3e-2; deep
layers 0.3–0.5, the long-context drift above). Through the server, splitting a prompt differently
(a restart from a snapshot at 2048, a fork, a stored session) gives bit-identical logits.

One bug found on the way: `dalloc`'s `cudaMemset` runs on the legacy stream, which the model's
non-blocking streams do not wait for, and with 4096-row buffers the zeroing of the re-tiled compressor
weights landed after `tile_fp8` had filled them. Every compressed entry the prefill wrote was then
garbage, and the teacher-forced steps after it predicted the transcript's next token 498 times in 512
(model.py: 325), most likely because the transcript was itself generated by an engine whose long
attention misbehaved alike; `dalloc` now waits for the zeroing.

### CED

`--mode tf --big --ced` on golden/ced's prompts (tf goldens made from its streams, with model.py's
logits in the CED or the exact mode), the engine's exact mode beside it:

| check | engine | model.py (golden/ced) |
|---|---|---|
| P = 128, where CED is exact: exact, CED, CED in chunks of 48 | one hash | |
| long8k in chunks of 4096 or 4300 (the replay's rows from two chunks) | one hash | |
| long8k first token (position 8,679) | argmax 8197 in both modes; lp −0.117 CED, −0.115 exact | the same argmax; −0.139, −0.116 |
| long8k decode, 255 positions: CED against exact | top-1 247/255, mean \|Δlp\| 0.100 | 248/255, 0.079 (split-K twin 248/255, 0.091) |
| long8k decode against model.py's stream of the same mode | CED 240/255, 0.259; exact 241/255, 0.251 | |
| long28k first token (position 28,488) | argmax 21750 in both modes | the same |

CED moves the engine's output about as far as it moves model.py's. The engine is as close to
model.py's CED stream as its exact mode is to the exact stream.

Position by position, the two CED effects do not correlate (0.01). At 8k positions both sit on top
of the engine's own drift from model.py (|Δlp| 0.25), which is larger than the effect itself.

Prefill times (`--mode tf --big`, chunks of 4096, rank 0's wall clock):

| prompt | exact | CED |
|---|---|---|
| a warm 4096-row chunk | 1.06 s | 0.54 s |
| long8k, 8,680 positions | 2.61 s | 1.51 s |
| long28k, 28,489 positions | 7.64 s (3.7k tok/s) | 4.08 s (7.0k tok/s) |

The first chunk of a process takes about 0.19 s more, because it allocates the prefill buffers and
gathers the weights. The replay, 128 rows through layers 20–39, adds about 0.1 s to the last chunk.

wo_b by all-to-all and the MoE's combine inside its reduce-scatter (2026-10-12), long28k with CED,
`--ring-cap 4194304` (now the default). Every run gives the hash `14650fb0739d0383` (the hash's seed,
which shows only that the run completed: until 2026-10-13 this run hashed no row, see
[Golden check](#golden-check); it now gives `8f08a153d415c5c8`):

| chunks | master fbb3e10: warm chunk, long28k | now: warm chunk, long28k | first chunk now |
|---|---|---|---|
| 4096 rows | 0.54 s (7.5k tok/s), 4.09 s | 0.49 s (8.3k tok/s), 3.83 s (7.4k tok/s) | 0.78 s |
| 8192 rows | 0.95 s (8.6k tok/s), 3.77 s | 0.85 s (9.6k tok/s), 3.52 s (8.1k tok/s) | 1.23 s |

8192-row chunks are 13 % faster per token than 4096-row ones: the routed experts' weights (1.8 GB
per layer per rank) stream once per chunk whatever its size. The first chunk of a process paid
0.29–0.38 s of buffers and weight gathers. Startup makes them now (about 0.5 s more before
"ready"): the first 8192-row CED chunk takes 0.91 s and long28k 3.21 s; the first exact 4096-row
chunk (tf golden) takes 1.03 s instead of 1.30 s.

### The official checkpoint's weights (2026-10-13)

From 2026-10-13, measurements to be published run on the official checkpoint's files. Production
keeps serving the variant's. [layout.md § The official checkpoint](layout.md#the-official-checkpoint-converted-from-one-boxs-copy-2026-10-13)
covers the files: how they were made, and how they differ from the variant's. The difference is 54
FP8 tensors in layers 10–36, each layer's `wo_b` and shared-expert `w2`. Every rank gets the files
with `--weights ~/dsgb10/weights-official/rank.dsw`.

The binary is release candidate 5's engine code, built from f7adc24. In the variant column, the
first four rows are the documented reference hashes. Release candidate 5 reproduced them on
2026-10-13, and this binary reproduced tf and 1k the same day. The other rows are this binary's runs
on the variant's files that day; the long28k first token comes from the correctness review's run
with release candidate 5.

| check: `run4.sh --big` with | variant (production's files) | official |
|---|---|---|
| tf, 4096 prefilled and 512 teacher-forced: `--mode tf --golden /data/dsgb10/golden/long --prompt 4096 --limit 4608 --prefill-rows 4096` | `842278b28d87dad0` | `c7605e0b6ba8d9d2` |
| 1k at k = 1: `--mode decode --golden /data/dsgb10/golden/long --ctx 1024 --steps 256`; the same with `--no-l2pf`, `--spec --draft-fp8` or `--spec --draft-bf16` | `febc7dc54ee40a39` | `7941647b162df1a8`, all four |
| sampled 1k: the same with `--steps 128 --temp 0.8 --top-p 0.95 --top-k 40 --seed 7` | `a8f7a08f8cd1c5e8` | `02ef3e0858d7eb45` |
| 16k CED: `--mode decode --golden /data/dsgb10/golden/long --ctx 16384 --steps 256 --ced` | `81f772ab79a950e4` | `8790ee9388f4abd8` |
| long28k CED, the first token: `--mode tf --golden /data/dsgb10/kernels/cedcheck/long28k-ced128 --prompt 28489 --limit 28489 --prefill-rows 8192 --ced` | 21750, log p −0.09542 | 21750, log p −0.08909 |
| long28k CED with the last 128 positions teacher-forced: the same with `--prompt 28361` (one run each) | `4a5933120ce54307`; first token 16224, log p −0.24755 | `e1df17aac1a8df4c`; 16224, log p −0.32933 |
| spectf on prose, the leading drafts equal to golden's tokens 0..5: `--mode spectf --golden /data/dsgb10/golden/prose --prompt 39` | 49 32 11 3 1 0 (0.698 per step) | 49 31 10 4 1 1 (0.750 per step) |

**The long28k CED hash checked nothing until 2026-10-13.** The documented run teacher-forces no
positions, because `--prompt` equals `--limit`. The hash it printed, `14650fb0739d0383`, was therefore
the hash's seed (1469598103934665603), and every such run printed it whatever it computed; both weight
sets printed it here. Where this page quotes that hash as a bit-identity check (§ CED's chunking table,
the reference hash lists), it shows only that the runs finished. What tells two such runs apart is the
first token's log-probability, which the run prints beside it. The run with 128 positions teacher-forced
gives a hash that covers what the prefill left behind. Since b47c510 a run that teacher-forces nothing
hashes the prompt's sample instead (its token, the bits of its logit and logprob, and its top 20
logprobs' ids and bits; [Golden check](#golden-check)): long28k with CED gives `8f08a153d415c5c8` on the
variant's files (three runs) and `de99bfce6ea96ba8` on the official ones (two runs), on 2026-10-13 with
`golden/refhash.sh ced28` (for the official files with `EXPECT=0` and the `--weights` above).

**Serving.** A server on the official files ran production's arguments, with `--disk-gib 8
--keep-gib 0` and a spill directory of its own. `serve_test.py --ced` passed 17 of 17.

**Greedy completions for the benchmark.** The same server produced greedy completions of
`bench/golden_sglang.py`'s four prompts (chat, thinking off, each in a fresh session, 256 new tokens
or EOS):

- math answers `391` and stops;
- the story and the code run to 256 tokens;
- the sky-is-blue answer stops after 75 tokens.

They are saved with every token's log-probability, the top 2 and the text in
`/data/dsgb10/golden-official/greedy/`.

**Against model.py on the official weights.** golden/README.md's prose run, with
`CKPT=/data/models/DeepSeek-V4.1-Flash`, wrote `/data/dsgb10/golden-official/prose` (1.4 GB) in 265 s
on rank 0:

```bash
golden/run_ref.sh run prose golden/prose.tokens.json --positions all --record auto --full-logits all
```

Its tokens are the variant golden's: the 39-token prompt and SGLang's 96-token completion, made on
the variant. The engine ran § Correctness's checks against it, and the same binary ran them on the
variant the same day:

| prose, 135 positions | variant, 2026-10-11 (§ Correctness) | variant, this binary | official |
|---|---|---|---|
| free-running: top-1 against golden's argmax | 124/135 | 121/135 | 123/135 |
| all 134 rows (`cmp_runs.py`): argmax agreement, near-tie / clear flips; mean \|Δ log p\| over golden's top 5; KL | | 120/134, 13 / 1; 0.49; 0.048 | 122/134, 12 / 0; 0.51; 0.050 |
| the 96 completion positions, the same | 88/96; 0.60; 0.059 | 89/96, 6 / 1; 0.59; 0.053 | 86/96, 10 / 0; 0.62; 0.057 |
| argmax = the sequence's next token: ours / model.py, of 134 | | 102 / 102 | 99 / 102 |
| every block fed golden's inputs (`--inject`): top-1; routing per layer; index top-k at the 8 index layers; block output relative L2 per layer | 134/135; ≥ 806/810; identical; 0.5–1.3e-2 (max 0.15) | | 134/135; ≥ 805/810; identical; 0.4–1.3e-2 (max 0.12) |
| teacher-forced decode at k = 1 and k = 6 | the prefill's logits | | the prefill's logits: one hash (`70fdbc5578d81406`) and every measure equal |

- **Generation.** Free-running from the prompt, the engine reproduces SGLang's first 4 completion
  tokens. At position 43, where official model.py also leaves that sequence clearly (margin 3.4), it
  takes model.py's token. On the variant it followed SGLang for 12 tokens.
- **The engine's distance to model.py** is the same on the official weights as on the variant's.
- **The two checkpoints' model.py runs** are further apart from each other than the engine is from
  the model.py run on its own weights (variant as reference): 110/134, with 18 near-tie and 6 clear
  flips, \|Δ log p\| 0.92 and KL 0.19; over the completion, 85/96, 1.01 and 0.17.
- **Against SGLang's completion,** which was made on the variant, official model.py scores 83/96 and
  the variant's 88/96.

**The long goldens: estimated, not run.**

- **The 28,489-token agentic transcript** (golden/README.md's `long`) took 1,038 s on rank 2 on the
  variant. rank 2 holds the official checkpoint too, so the official run would take about as long
  there. On rank 0 it would take about 18–25 minutes:
  - official prose took 265 s on rank 0 against the variant's 238 s on rank 2;
  - CED's long28k, with three decoder streams beside the exact one, took 1,496 s on rank 0.
- **The six long CED cases** ([ced.md](ced.md): long8k and long28k) took 1,054 + 1,496 s on rank 0,
  43 minutes, or 68 with the split-K twins. The official runs would take the same.
- Either job holds the ring (`PREEMPT=0`) throughout: model.py needs up to 23 GiB of GPU memory.

### What remains for prefill

A warm 8192-row CED chunk on rank 0 (nsys, master 3c36d66, the second of two chunks) spends 42.9 ms
per encoder layer; rank 2 shows the same structure:

| part | ms per layer | |
|---|---|---|
| attention side | 14.2 | all-gather of the qa / kv / compressor rows 1.33 (the main stream waits), wq_b 1.15, `sparse_attn_p` 4.79, wo_a 1.80, wo_b 2.59 of which 0.8 waits for the all-to-all, sum4 0.50, boundary 1.31 |
| FFN | 26.8 | router 0.33, all-gather of the MoE's inputs 1.98 (the main stream waits), shared expert 1.26, bucketing 0.19, up 8.45, down and combine-RS 13.2, boundary 1.34 |

The FFN's critical path is the chain of the four column groups' combine-RS kernels (11.4 ms) after
the first group's down (1.78 ms). The other three downs run under it, slowed to 2.85 ms each, and
the main stream waits 2.9 ms for the last link. Down and combine-RS together move ~2.5 GB per layer
(the bf16 outputs' round trip 1.0, the ring's buffers 0.67, w2 0.57) at ~190 GB/s, so the phase is
bound by DRAM, not by the wire: a combine-RS alone reaches 73 % of L0's line rate, beside a down
57 %. Keeping the bits, the MoE moves ~4.0 GB per layer, a floor of ~17 ms against the 21.7 its up,
down and combine-RS take. Standalone at 8192 rows (`bench_kernels prefill`, one box): shared expert
1.12 ms, up 7.38 (203 GB/s), down 6.61 (~167 GB/s), the separate combine 3.46.

16384-row chunks are not worth their 3.2 GiB per rank: 83.5 ms per layer, 2.6 % less per token than
two chunks of 8192. The up kernel amortizes its weights (13.3 ms against 16.9), but the combine-RS
chain doubles (21.6 ms) and the main stream waits 6.6 ms for its last link.

In order of gain per effort:

- **Down kernel**: one CTA per (tile, column group). The tile's h rows load once per group instead
  of once per 128 columns, w2 streams without a break between column blocks, and each block's
  outputs are stored from registers under the next block's loads. The fused kernel for small chunks
  stays. Target 6.6 → ~5 ms standalone at 8192, which also leaves DRAM to the combine-RS.
- **Up kernel**: 7.4 → ~6.5 ms: a deeper weight prefetch (246 registers and 44.5 KB of shared
  memory hold it to 2 CTAs per SM).
- **Column groups**: a smaller first group starts the combine-RS chain earlier. A model fitted to the
  measured rates gives at best −0.45 ms per layer (six groups of 768–1024 columns).
- **Three small items, ~2 ms per layer**: own-row work under the two all-gathers the main stream
  waits for (~0.7 ms); wo_a and its all-to-all in two halves of rows (~0.5 ms); `hc_boundary_p`,
  which runs at ~67 % of DRAM bandwidth (~0.8 ms).

**Built, off by default** (2026-10-12; measured the same day, below):
- the down kernel above, `k_moe_down_g`, fused or not (`PrefillTuning::dgrp`, column blocks per CTA,
  and `dring`, weight steps in flight); up's L2 weight prefetch (`upf`); a tile order that interleaves
  the most- and fewest-picked experts (`tord`). Defaults `kDownGroup`, `kDownGroupF`, `kUpPrefetch`,
  `kTileOrder` are 0. Each tile's expert, rows and base are now one 16-byte entry (the default path too;
  the same arithmetic).
- `hc_boundary_p` with 4, 8 or 16 rows per CTA for small chunks and an L2 prefetch of the block a few
  ahead (`hc_boundary_tuning`; `kHcRowsByChunk`, `kHcPrefetch`). A replay's 32 rows per rank run in
  one CTA, ~0.235 ms per boundary cold, 40 boundaries (~9.4 ms) per replay, and as much again in a +200
  turn's encoder. A row's sums are the same at every R.
- Checks: `test_kernels p_moe` (every variant against the reference, bit for bit), `test_pattn
  mhcrows`; sweeps: `bench_kernels moereal` on dumped routes (`PF_DGRP`, `PF_DRING`, `PF_UPF`,
  `PF_TORD`), `bench_pattn --only mhcr`.
- Measured on the GPU (2026-10-12):
  - Bits: the bit tests pass. The engine gives the defaults' hashes with every knob on (`dgrp=4 dgrpf=3
    dring=9 upf=2 tord=2 hc=4 hcpf=3`) and with the picks below: tf 842278b28d87dad0, CED
    14650fb0739d0383 (until 2026-10-13 the hash's seed, see [Golden check](#golden-check); now
    8f08a153d415c5c8), decode 81f772ab79a950e4. `serve_test.py --ced` passes 17 of 17.
  - The picks, each on the cases it governs: `dgrp` 10 at 0.948 of `k_moe_down_p` (8,000 and 8,192
    rows); `tord` 1 at 0.957; the boundary by the chunk with prefetch 2 at a geomean of 0.823 (0.66 at
    32 rows, 0.92 at 500, 0.99 at 8,192).
  - Turns through the server, the three picks against the defaults: prefill 0.989 in geomean. Cold
    16k–128k 0.982–0.988, +8000 0.980–0.985, +200 0.977–1.000, +2000 0.994–1.001. That is short of the
    2 % a default needs, so the defaults stay off.
  - For a later sweep: the boundary's prefetch per rows per CTA (none at 4 rows: 0.809 on the same
    table), and the knobs at 2,000 rows, where the +2000 turns' layers 0–19 ran 0.7–1.2 % slower.

Prefill is a small share of agentic serving (a cold system prompt about once, then ~0.4 s per
turn), so these wait behind decode.

**A two-chunk wavefront: probed, not built (2026-10-12).** The idea was to run chunk c's FFN beside
chunk c + 1's attention side, which needs only chunk c's layer-L keys. It hides little: the two sides
compete for the SMs as well as for DRAM.

The probe is `--wave-probe` (engine/model/wave_probe.cu). At layer L of one chunk, it runs that
chunk's FFN on one stream beside a second attention side of layer L on another, then the two in
sequence. The second side stands in for the next chunk's: the same positions, from a copy of the
chunk's activations, with its collectives on a second ring (the next port, proxy core 6). Kernels,
weights and wire are the real ones.

Rank 0's numbers, in ms per layer, are the median of 7 interleaved rounds. The other ranks are
within 0.6 ms. The chunk starts at position 16,384 of a 32k CED prefill, or at 114,688 of a 128k
one. "Together" has both main streams at equal priority. Raising the FFN's priority helped layer 8
by 0.4–0.7 ms and slowed the other layers by up to 0.9 ms; raising the attention's changed little
(−0.2 to +0.6 ms).

| layer | rows | FFN | attention | in sequence | together | together / in sequence |
|---|---|---|---|---|---|---|
| 1 (window only) | 8192 | 26.7 | 10.8 | 37.6 | 35.3 | 0.94 |
| 8 (compressor, indexer) | 8192 | 26.8 | 19.2 | 46.0 | 42.8 | 0.93 |
| 10 | 8192 | 26.7 | 14.3 | 41.3 | 38.3 | 0.93 |
| 16 | 8192 | 26.6 | 14.1 | 41.1 | 38.4 | 0.94 |
| 8, chunk at 114,688 | 8192 | 27.1 | 22.9 | 50.0 | 47.2 | 0.94 |
| 10, chunk at 114,688 | 8192 | 27.2 | 14.1 | 41.7 | 38.9 | 0.93 |
| 1 | 4096 | 16.5 | 5.6 | 22.3 | 21.0 | 0.94 |
| 8 | 4096 | 16.3 | 10.0 | 25.9 | 24.9 | 0.96 |
| 10 | 4096 | 16.2 | 7.6 | 23.6 | 22.7 | 0.96 |
| 16 | 4096 | 16.2 | 7.2 | 23.4 | 22.0 | 0.94 |

At 8192 rows, running the two sides together hides 2.2–3.2 ms of the attention side. That is less
than half of the ~7 ms per layer the two sides spend waiting for collectives. At 4096 rows it hides
0.8–1.4 ms. Weighted over the encoder (2 window-only layers, 3 like layer 8, 15 like layers 10
and 16):

- Two chunks of 8192 rows in flight take 0.93 of today's time per layer (38.7 against 41.6 ms).
  For a 28k compaction (3.2 s), that saves at most 0.2 s.
- Two chunks of 4096 rows take 1.09 of one 8192-row chunk's time (45.3 ms). The routed experts
  stream once per chunk, so the FFN of 4096 rows takes 16.2 ms against 26.7 for 8192 rows.
  The overlap does not win back that 5.7 ms per 8192 rows.

Neither clears the 15 % that would pay for what building it takes:

- a second ring;
- pairs of chunks through the server and Engram's lookup sets;
- per-chunk top-k slots;
- the cross-stream hazards in between.

CTAs that the attention side takes cost the FFN about as much as they gain. The MoE's kernels
cover every SM (the up kernel holds 2 CTAs per SM at 246 registers), and priorities do not move the
sum.

The second set of activations would have been small, because the attention's and the MoE's scratch
serve the two chunks in turn. It holds x, mid, y, yq / ys, the mixes and the encoder's three top-k
slots: ~0.3 GiB per rank at 8192 rows, computed from the buffer sizes.

Re-probe if the MoE's kernels come to reach their rate on fewer SMs:
`--wave-probe 66818 --wave-probe-at 16384` with `--mode decode --big --ced --ctxs 32768` probes
layers 1, 8, 10 and 16. A probe run's outputs are not the model's.

### Step times

The decode loop (`--mode decode`, 256 greedy tokens, mean over the last 247; each step queued behind
the last, no broadcast), before and after the mixes moved off the path (single-pass mixes from a
quarter of hc_fn, summed across the ranks on the side stream; allreduce_hc over 40 SMs):

| context | before (master 2f8acae) | now |
|---|---|---|
| 1,024 | 18.90 ms, 52.9 tok/s | 18.56 ms, 53.9 tok/s (three runs: 18.55–18.57) |
| 16,384 | 18.98 ms, 52.7 tok/s | 18.63 ms, 53.7 tok/s |

Graph replays (`--mode bench`, 40 steps per k, the first 3 dropped; GPU time from the step's first
kernel to its last, on rank 0). Two runs of one build differ by up to 0.6 ms, so "now" is the mean of
two runs' medians (before: one run):

| rows k | context | step: median, now | per row | before |
|---|---|---|---|---|
| 1 | 1,024 | 18.77 ms | 18.8 ms | 18.90 ms |
| 1 | 16,384 | 19.39 ms | 19.4 ms | 19.76 ms |
| 6 | 1,024 | 36.19 ms | 6.0 ms | 36.95 ms |
| 6 | 16,384 | 37.71 ms | 6.3 ms | 38.35 ms |

These are bench mode's steps, each broadcast from rank 0 (0.8–1.9 ms between steps); the decode loop
above has no broadcast. Eager steps take 21.9 and 40.4 ms at 1k. Prefill through this path (chunks
of 8 rows) takes 42–44 ms per chunk, about 190 tokens/s; it is not the prefill path.

Where a k = 1 step goes at 1k (nsys on rank 0, graph nodes, 34 steps; exclusive is the time a class
alone advanced the step's end, busy the summed kernel time):

| class | before: exclusive (busy) ms | now: exclusive (busy) ms | floor |
|---|---|---|---|
| weights (GEMVs, MoE) | 15.64 (22.19) | 15.97 (22.60) | 14.4: 3.37 GB at 234 GB/s |
| collectives | 2.37 (3.69) | 2.34 (5.02; 2.11 of it the side stream's mix collective) | ~1.25: 125 on the path at ~10 µs |
| attention | 0.75 (1.89) | 0.74 (1.86) | |
| mHC mixes | 0.30 (3.81) | 0.01 (0.82) | 0 (side stream) |
| other, gaps | 0.04 | 0.04 | |
| kernel span: median (min) | 19.04 (18.71) | 18.96 (18.40) | |

Exclusive time follows whichever kernel started first (chainprof's rule then; `chainprof.py --first`
keeps it, and since 693d278 a side-stream kernel gets only the time no main-stream kernel covers), and
with PDL the GEMV after a collective starts early: when the collective ends sooner, that GEMV is left
more time alone and the saving shows up as weights. Per sublayer (from the previous all-reduce's end
to its own, median over 39 layers × 34 steps) the attention went from 196.3 to 192.6 µs and the FFN from 219.4 to 214.5 µs,
8.6 µs a layer or 0.34 ms a step, which is the decode loop's gain: the attention's all-gather ends
3.7 µs sooner (its GEMV now shares the bandwidth with 0.5 MB of hc_fn reads instead of 2) and the
FFN's all-reduce takes 18.1 µs after the last MoE kernel instead of 22.1.

The weight kernels stream at 214 GB/s overall (exclusive time): `k_gemv_fp8` (wq_a + wkv, wq_b,
wo_b, Engram's wkv) ~198 GB/s, `k_gemv16` (wo_a, router, head) ~233, `k_moe_up` 212, `k_moe_down`
228. Each `allreduce_hc` adds 15.6 µs exclusive (80 per step; 17.8 before; the ring's 10 KB
all-reduce takes 12). At six rows the step's 36.3 ms (36.7 before) are the MoE 21.4 ms (up to 36
experts per layer), the other GEMVs 9.6, collectives 4.0, attention 1.0 and mixes 0.3 (0.9 before).

### Long contexts

Agentic sessions reach 50–150k tokens. These runs use `golden/synth270k`: the golden texts (long,
doc-textwrap, doc-novel) end to end, repeated to 270,000 tokens. Each run is at `--max-ctx 266240`
and reaches its contexts through the exact prefill path. "Before" is the build this work started from
(c553aff: index_fused on every index layer, a graph per 4096-position bucket). "After" is index_dense
on one CTA per SM. Measured 2026-10-12, before and after within two hours of each other.

**Decode loop.** `--mode decode --ctxs …`, 256 greedy tokens per context, mean ms per token over the
last 247. Each context goes on from where the last one's decode ended. The tokens are the same before
and after.

| context | before | after | tokens hash |
|---|---|---|---|
| 1,024 | 18.556 | 18.545 | febc7dc54ee40a39 |
| 32,768 | 18.589 | 18.651 | 3d6712294df0c472 |
| 65,536 | 18.745 | 18.697 | b2c1a8fbe964cf3d |
| 131,072 | 18.854 | 18.725 | b1673ad731d7a705 |
| 262,144 | 19.158 | 18.927 | fbcec718984007a6 |

128k is now 0.18 ms above 1k (0.30 before), and 256k is 0.38 above (0.60).

**Graph replays.** `--mode bench`, GPU ms, the median of 37 steps. Two runs of the before build
(second in brackets) differ by up to 0.75 ms, more than the indexer's share. The k = 6 steps also
change with the context through the MoE: other tokens, other experts. So the profile below is what
separates the attention.

| context | k = 1 before | after | k = 6 before | after |
|---|---|---|---|---|
| 1,024 | 18.48 (19.22) | 18.95 | 36.37 (37.16) | 36.41 |
| 32,768 | 19.27 (19.51) | 19.24 | 37.97 (38.11) | 37.65 |
| 65,536 | 19.31 (19.60) | 19.44 | 39.46 (39.12) | 38.78 |
| 131,072 | 19.49 (19.79) | 19.23 | 38.89 (38.70) | 37.95 |
| 262,144 | 19.69 (19.98) | 19.40 | 39.44 (39.51) | 38.52 |

**Where the context went.** nsys at 1k and 128k: graph nodes, 34–40 steps, medians.

| | k = 1, 1k | k = 1, 128k | k = 6, 1k | k = 6, 128k |
|---|---|---|---|---|
| attention, exclusive ms per step: before | 0.752 | 0.970 | 1.064 | 1.516 |
| after | 0.724 | 0.843 | 1.029 | 1.259 |
| the eight index kernels, µs per step: before | 176 | 398 | 189 | 616 |
| after | 157 | 259 | 164 | 365 |

- Only the index layers' scans grow with context. Window attention and the 512 selected entries are
  fixed. The Reindex layers (24–36) score layer 20's pool of 16,384 positions at any context: 20–31 µs
  each with index_fused, 13–14 at one row and 21–22 at six with index_dense_blocks (below).
- At 128k, k = 1:
  - layers 2, 8 and 14 (65,536 keys each) take 62 µs before and 32–37 after;
  - layer 20 (131,072 keys and the pool) takes 123 before and 63 after.
- At six rows the same layers take 101–104 → 54–55 µs and 203 → 94.
- The keys are read from DRAM once per step: 4.5 MB at layers 2/8/14 and 8.9 at layer 20, at 235 GB/s.
  That puts a floor of about 95 µs a step under the dense scans at 128k (they take 165 µs).
- These used chainprof's first-started rule. "The attention side per step" (below) re-measures them on
  09847af with the side-stream rule: at 128k, 0.947 ms at k = 1 and 1.222 at k = 6. At k = 1 most of
  the difference is the compressor's emit, which the main stream now waits for.

**index_dense** ([indexer.cu](../engine/attn/indexer.cu)) is one launch per dense index layer for all
of a step's rows. index_fused, which it replaces there, ran one CTA per (part, row), so a six-row step
read the keys six times.

- **Grid.** One CTA per SM (1024 threads). The CTAs take the scan's 512-position chunks in turn.
  - They load each chunk's keys once, the next chunk's while they score this one.
  - They score each chunk for every row of their row group.
- **Row groups.** A scan with fewer than two chunks per SM splits its rows into groups, so each CTA
  preps the queries of fewer rows.
- **Selection.**
  - Each CTA adds its histograms of key>>5 (from shared memory) to global ones.
  - The last R CTAs to arrive select a row each from those histograms.
  - They gather the candidates in shared memory.
  - Past 32k keys, a selection reads only the blocks of 8 keys whose maxima reach the threshold bin.
- **One graph.** The CTAs take their roles from the scan length the positions give. So the launch, and
  each row count's graph, is the same at every context.
- **Exact.** The outputs are the same, bit for bit, as index_fused's. The query's fp4 codes divide by
  a power of two as a multiply by its exact inverse.

**Graphs and memory.** Before, a graph sweep (one step at every bucket from 1k to 266k, `--graph-sweep`)
captured 64 new graphs per row count: 131 in all, 26.9 MiB each, 3.4 GiB. A step that captured one
took 25.8 ms at k = 1 and 44.4 at k = 6, against 18.5 and 36.4. The spec step's graph is ~210 MiB, and
a server captures one per bucket at startup: 16 at 65,536, 32 at 131,072 and 64 at 262,144 (about
13 GiB). After, the sweep captures no graph, and a server captures one spec graph at any limit: 0.24
GiB and 0.2 s at both 131,072 and 266,240. The session store keeps its 12 GiB.

**Session store.** One session of n positions holds:

- its compressed caches, 3.48 MiB per 4096 positions;
- window snapshots, 2.81 MiB each with the drafter's windows (2.58 without): every 1024 positions over
  the last 8192, four per doubling of the distance beyond ([Serving](#serving), snapshot thinning);
  with every one kept they were three quarters of the session;
- the end state, 6.1 MiB (5.4 without the drafter's windows).

Measured on 2026-10-12 with `snap_test.py --far 32768,131072,250000`: a cold session of each length,
then forks of it, each of which parks it in the store. A parked session's size is the store's total
less the live fork's snapshots.

| positions | window snapshots | store, every snapshot | store, thinned |
|---|---|---|---|
| 32,776 | 32 → 16 | 127.3 MiB | 82.3 MiB |
| 131,080 | 128 → 24 | 481.3 MiB | 188.8 MiB |
| 250,008 | 245 → 28 | 911.9 MiB | 301.6 MiB |

The 12 GiB store now holds about 65 sessions of 128k (25 before) or 40 of 250k (13). Before the
drafter's windows were stored, every snapshot came to 455 MiB at 132,072 and 855 MiB at 251,000.

Each fork adds 300 tokens d positions before its base's end. "Re-prefilled" counts the positions from
its restart to its prompt's end; the times are the engine's prefill.

| session | d | every snapshot: re-prefilled, ms | thinned: re-prefilled, ms |
|---|---|---|---|
| 32,768 | 1,000 / 8,000 / 20,000 | 324 / 492 / 780; 245 / 271 / 304 | the same; 242 / 268 / 302 |
| 131,072 | 1,000 / 8,000 / 20,000 | 324 / 492 / 780; 260 / 265 / 305 | the same; 254 / 266 / 300 |
| 131,072 | 50,000 | 476; 285 | 7,644; 944 |
| 131,072 | 100,000 | 652; 310 | 14,988; 1,724 |
| 250,000 | 1,000 / 8,000 / 20,000 | 468 / 636 / 924; 286 / 303 / 336 | the same; 277 / 292 / 329 |
| 250,000 | 50,000 | 620; 294 | 3,692; 593 |
| 250,000 | 100,000 | 796; 322 | 2,844; 522 |
| 250,000 | 200,000 | 1,148; 336 | 17,532; 2,072 |

The forks 20,000 back each found a kept multiple of 4,096 within 1,024 of the fork point. Over fork
points spread evenly through a session, thinning adds 0.7k positions to the re-prefill on average at
24k, 1.5k at 40k, 7k at 150k and 10.5k at 250k: 0.1, 0.2, 0.8 and 1.2 s at 8.5k tok/s. A cold
prefill of an evicted 150k session takes 16–20 s.

Checked against the same prompts prefilled cold:
- `snap_test.py`: a 40k session; forks at d = 300 to 37,000 with 300 new tokens each, each restarting
  where `keep_snap` says; a 50-token fork 25k back; the session diverging 18k back, then a turn at that
  prompt's end. 19 of 19 with CED and 19 of 19 at `--exact`.
- `serve_test.py --ced`: 17 of 17. At `--exact`: 9 of 9.
- `lst.py 131072`, the long-session checks: 6 of 6.

**The disk tier** ([Serving](#serving)) was measured on 2026-10-12 through the front-end with
`revisit.py`:
- two conversations, of ~37.5k and ~180k tokens, each pushed out of a 0.25 GiB store
  (`--session-gib 0.25`, where nothing stays between requests) by another request;
- each then revisited with a few new tokens, against the same prompt cold in a new session.

| prompt | spilled | first token, from disk | of which reading back | first token, cold |
|---|---|---|---|---|
| 37,504 | 88.6 MiB | 0.197 s | 100 ms | 4.112 s |
| 180,300 | 236.4 MiB | 0.368 s | 265 ms | 19.288 s |

The rest of a revisit is what one from the store costs, about 0.1 s: the 9 new positions run as one
prefill chunk, 86 ms at either length. Rank 0 read back at 0.64–1.42 GB/s (25 reads of 18–236 MiB),
in 8 MiB `pread`s one at a time from its USB4 SSD; a deeper queue would be the next gain. A session
revisited right after its spill waited up to 120 ms for its own writes. The spills' copies to staging
were not timed on their own.

Checked on the 0.25 GiB store (56 spills, 25 reads back), against the same prompts cold:
- `disk_test.py`: sessions of 20k and 64k, each turn preceded by a request in another session, so
  that every turn's session comes back from disk. The turns: +300 (CED), +40 (exact), a regenerate, a
  divergence 8,000 back and a fork 15,000 back. All 12 requests' tokens and log-probabilities equal a
  default server's, where nothing spills (`--compare`), and the 4 checks against cold pass. The +300
  turn is not checked against cold: its history holds decoded positions ([Serving](#serving), exact
  reuse).
- `serve_test.py --ced`: 17 of 17. `snap_test.py --ced`: 19 of 19. `lst.py 131072`: 6 of 6.
- At the defaults (12 GiB, the disk tier on, nothing spilled): `serve_test.py --ced` 17 of 17; tf
  842278b28d87dad0 and decode at 1k febc7dc54ee40a39, unchanged.
- No spill file was left on any box after the servers stopped.

**Under load** (`soak.py --disk`, 2026-10-12 09:37–09:57). The binary was 7af6ade's, with c5ff9c7's
checksums: each block is summed on the device before it leaves and again after it comes back. Three
clients ran for 20 minutes through the front-end, with a 3 GiB store (`--session-gib 3`) and an 8 GiB
file:
- 948 requests, none failed or malformed, and no error line in any rank's log.
- 371 spills (27.1 GiB) and 183 read-backs (14.3 GiB). 75 sessions were dropped from the full file
  (at most 7.95 of 8 GiB).
- 4,583 blocks were checked against their checksums, all equal. Every probe of a session put aside
  answered its revisit the same: 70 read back from disk, 59 from the store.
- Read-backs of 32 MiB or more: 180, with a median of 103 ms for 79 MiB (0.80 GB/s; p10 0.65). Three
  waited for their own writes, at most 164 ms.
- The copy into staging, on the model's stream: a median of 1.48 ms for 72.5 MiB (51.6 GB/s), after
  0.57 ms of checksums. The host's part of a spill took 0.19 ms; the file was written at 0.84 GB/s.
- Rank 0's time to the first token, a revisit against the same prompt cold:

  | prompt | from disk (of it reading back) | cold |
  |---|---|---|
  | 33k | 314 ms (81) | 3,662 ms |
  | 36k | 120 ms (90) | 3,978 ms |
  | 79k | 454 ms (172) | 8,520 ms |

- Memory stayed flat: GPU memory, the resident sets apart from the Engram cache (at most +49 MiB, on
  rank 0), and pinned memory (774 MiB). The spill file never sat in the page cache, and each box's
  dirty pages stayed under 28 MiB.
- The golden replays, 10 minutes apart, gave the same answers at 107.4 tok/s both times.
- Regenerates were identical in all 22 pairs that resumed at the same position. Of the 8 pairs that
  resumed at different positions, 2 differed, which CED explains.
- The run's own verdict was FAIL, on one rule: the front-end's resident set went from 121 MiB (the
  first half's peak) to 222 MiB (the second half's).
  - That was its session table filling: 226 of 256 sessions and 7.3M of 16M tokens at the end. The
    resident set fits 51 MiB + 25 bytes per token held, to within 9 MiB.
  - The rule had judged the two halves as if the table were already full. It now judges a table that
    is still filling against what the table holds, and `--report` passes the run.

**Long sessions through the server.** `lst.py` (scratch) runs these, each checked against the same
prompt cold in a fresh session (the same ids and top-2 logprobs):

- a cold prompt of n;
- n + 1000 in the same session, which reuses all n;
- a prompt diverging 3000 positions back, which restarts from the window snapshot at the multiple of 1024
  below.

The results:

- **Before the store was reserved at startup** (exact-path chunks of 4096): 8 of 8 checks at n =
  131,072 and 250,000.
  - Cold prefill runs at 6.7–6.8k tok/s: 19.4 s and 37.3 s.
  - The appended 1000 take 0.33–0.35 s.
  - Decode runs at 50–51 tok/s.
- **At serve's current defaults** (CED chunks of 8192, drafts), n = 131,072 at `--max-ctx 266240`: 4 of 4.
  - Cold prefill takes 14.1 s.
  - Decode runs at 105–112 tok/s on this repeated text.
  - `serve_test.py --ced` gives 17 of 17 on the same server.

**Agentic turns.** A turn appends a tool result (200 to 8k tokens) to a live session, prefills it
(CED, chunks of 8192), replays the decoder over the last 128 positions, and decodes. `turns.py`
(scratch) drives a server at `--max-ctx 266240` (drafts on) with synth270k. For each n it sends:

- a cold prompt of n;
- three turns appended to the same session, each the last prompt, its output and new tokens: +200,
  +2000 and +8000;
- another session's 1000-token request, then +2000 more, whose state comes back from the store.

Rank 0 logs each request's split (`serve: req N turn:`): the state operations (host and GPU), the
chunks' GPU time in the encoder (layers 0–19) and in the replay (layers 20–39), the head, and the
first decode step. "Before" is master 2979972 with that log line; "after" has the prefill's dense
index scans on dsk::index_topk_p (below). Time to first token at the client, ms, measured 2026-10-12:

| turn | 16k | 64k | 128k | 250k |
|---|---|---|---|---|
| +200, before → after | 220 → 224 | 216 → 217 | 241 → 235 | 228 → 218 |
| +2000 | 420 → 416 | 420 → 421 | 423 → 418 | 435 → 423 |
| +8000 | 998 → 1002 | 965 → 953 | 991 → 955 | 1026 → 987 |
| +2000 after another session's request | 433 → 433 | 420 → 415 | 433 → 420 | 435 → 420 |

Repeat runs of one build differ by up to about 30 ms at +8000. Cold prefill, CED in chunks of 8192:

| n | before | after |
|---|---|---|
| 16,384 | 1.90 s | 1.89 s |
| 65,536 | 7.02 s | 6.97 s |
| 131,072 | 14.13 s (9.27k tok/s) | 13.91 s (9.42k tok/s) |
| 250,000 | 27.62 s (9.05k tok/s) | 26.82 s (9.32k tok/s) |

Where a +2000 turn's time goes, after, rank 0, ms:

| | 16k | 128k | 250k |
|---|---|---|---|
| state operations, host (GPU) | 1.0 (0.1) | 2.6 (0.1) | 3.1 (0.1) |
| encoder: 2000 rows through layers 0–19 | 313 | 312 | 321 |
| replay: 128 rows through layers 20–39 | 98 | 99 | 94 |
| head and sampling | 1.4 | 1.5 | 1.5 |
| time to first token, at the client | 416 | 418 | 423 |
| after it: the draft windows, then the first decode step | 2.7 + 25.2 | 2.7 + 26.0 | 2.7 + 25.5 |

- **Session state.** On a live session the state operations cost nothing on the GPU. Bringing a
  250k session back from the store after another request costs 6.3 ms on the GPU and 4.7 ms on the
  host, and saving it when the other request arrives costs 6.2 ms.
- **What does not grow.** nsys over the +2000 chunk (a build with NVTX ranges per layer,
  `--profile-at`) shows the same at 16k, 128k and 256k.
  - The encoder's 2000 rows spend about 180 ms in the MoE (up 102, down 76), 40 in the attention's
    FP8 GEMMs, 23 in sparse attention and 14 in mHC. Its reduce-scatters and all-gathers keep 86 ms
    busy, mostly overlapped.
  - The replay spends 64 of its 95 ms in the MoE. For 128 rows the MoE streams nearly every expert's
    weights.
- **What grows: the index scans.** The index kernels in the +2000 chunk, ms:

| | 16k | 128k before | after | 256k before | after |
|---|---|---|---|---|---|
| encoder: layers 2, 8, 14 | 1.91 | 12.36 | 4.86 | 23.75 | 10.62 |
| replay: layer 20 and the Reindex layers | 0.47 | 1.03 | 0.69 | 2.05 | 0.88 |

**The prefill's dense scans.**

- **Before.** Layers 2, 8 and 14 in every chunk, and layer 20 in the replay and in exact chunks,
  ran index_fused: a CTA per (2048 positions, row). It loaded each key once per row and prepped the
  row's queries in every CTA.
- **After.** They run `attn::index_q_prep` once per row, then the kernels line's
  `dsk::index_topk_p` ([prefill_index.cu](../engine/kernels/prefill_index.cu), M5).
  - A CTA keeps 256 positions of keys in registers for every row of a batch of 128.
  - Each batch is selected while its keys are still in L2.
  - The keys and selections are the same, bit for bit.
- **Speed.** bench_pattn, per rank, for a 2048-row chunk at 256k:
  - 512 rows over 131,072 keys take 3.0–3.2 ms (the scores at 235 TFLOPS), against 7.8 ms with
    index_fused (bench_index);
  - layer 20's 32 replay rows over 262,144 keys take 0.45 ms, against 1.66 ms for index_fused in the
    profile above.
- **Memory.** The keys scratch shrinks from every row's to one batch's (or the Reindex scans' rows):
  1.08 GiB less per rank at 266240.

**Reindex layers in decode.** index_dense_blocks runs the pool scans of layers 24–36 on index_dense's
grid. A row's blocks are its own, so each row group is one row. Its selection starts from the
histograms, as index_dense's does.

- **Per layer at 128k** (nsys): 21.8–25.4 µs → 13.3–13.7 at one row, and 25.7–31.5 → 21.0–22.4 at six.
- **The eight index kernels per step:** 259 → 221 µs at k = 1, and 365 → 346 at k = 6. At 1k:
  157 → 115 and 164 → 143.
- **Decode loop** (ms per token, the same token hashes): 18.510, 18.630, 18.701, 18.698 and 18.876
  at 1k, 32k, 64k, 128k and 256k.

**The attention side per step, re-measured (2026-10-12).** Master 09847af, `--mode bench --big` on
synth270k at `--max-ctx 266240`, 40 steps at k = 1 and 40 at k = 6. Rank 0 is under nsys and the medians
come from `chainprof.py` (693d278). A kernel on the side stream is credited only with the time no
main-stream kernel covers: emit, kv_write, the indexer's wproj GEMV and the mixes. The attention class
counts the attention kernel and its combine, index_dense(_blocks), kv_write, the compressor's emit,
carry and projection rows, and the merges.

| context | k = 1: exclusive ms | share of the step | k = 6: exclusive ms | share of the step |
|---|---|---|---|---|
| 1,024 (an earlier profile) | 0.681 | 3.6 % | 1.010 | 2.8 % |
| 16,384 | 0.830 | 4.5 % | 1.047 | 2.8 % |
| 65,536 | 0.889 | 4.8 % | 1.159 | 3.0 % |
| 131,072 | 0.947 | 5.2 % | 1.222 | 3.3 % |
| 250,000 | 1.055 | 5.8 % | 1.366 | 3.7 % |

The share is of the profile's wall median per step: 18.29–18.40 ms at k = 1, 36.9–38.1 at k = 6.

- **What grows with the context.** Only index_dense:
  - at k = 1: 0.158, 0.206, 0.257 and 0.357 ms per step at 16k, 64k, 128k and 250k;
  - at k = 6: 0.162, 0.259, 0.323 and 0.454 ms per step.
  - The attention kernel stays at 0.48–0.50 ms (k = 1) and 0.75–0.77 ms (k = 6), and the combine at
    0.10 and 0.14 ms.
  - Against 1k, 64k and 128k add 0.21 and 0.27 ms at k = 1 (1.1–1.5 % of the step), and 0.15 and
    0.21 ms at k = 6 (0.4–0.6 %).
- **At 128k, k = 1** (0.947 ms):
  - the attention kernel 0.492 ms (12.3 µs × 40);
  - index_dense 0.257 ms: layers 2, 8 and 14 take 36.5–36.8 µs each, layer 20 66.9, and the Reindex
    layers 19.5–20.4;
  - the combine 0.098 ms;
  - emit 0.088 ms;
  - the rest (kv_write, carry, compressor_project_rows) 0.025 ms.
- **At 128k, k = 6** (1.222 ms): the attention kernel 0.749 ms, index_dense 0.323 (layers 2/8/14:
  53.5–55.6 µs, layer 20: 95.3, Reindex: 21.5–22.6) and the combine 0.147.
- **Why the emit shows at k = 1.**
  - The side stream runs the indexer's wproj GEMV (`gemv_bf16`, 4 CTAs, ~76 µs at one row) before
    kv_write and the emit, so they start as wq_b ends (median 1.8 µs before).
  - The emit then takes a median 33 µs. It is one CTA, which reads the 128 KB index-key projection
    when its row completes an entry.
  - At each kv-source layer, the main stream waits a median 34 µs between wq_b's end and index_dense's
    start: ~0.1 ms a step.
- **The Reindex layers: an open finding.** They took 19.5–21.7 µs each at one row in these runs
  (16k–250k). When index_dense_blocks went in, 00dc97c's nsys run had them at 13.3–13.7 µs. At six
  rows the two agree (21.5–22.6 against 21.0–22.4).
  - bench_index flushes L2 before each launch. It puts the pool scan at 20.3 µs at 131k and one row,
    on master's build of that morning and on B2's slot kernels alike (B2's last job, 10:39). So the
    later figure is what the kernel takes cold, and B2's slot work did not move it.
  - Why the first profile saw 13.3–13.7 was not looked into. The gap is ~7 µs a scan, ~0.03 ms over a
    step's four scans.
- **Decode loop.** `--mode decode --ctxs 16384,65536,131072,250000 --steps 256`, each context reached
  from the last one's end through the prefill path. Two runs each, k = 1 and `--spec --draft-fp8`, in
  alternating order. The token hashes are the same in all four runs:

  | context | k = 1, ms per token | `--spec`, ms per token | drafts accepted, tokens per step | tokens hash |
  |---|---|---|---|---|
  | 16,384 | 17.90 / 17.93 (55.8 tok/s) | 10.24 / 10.27 (97.5 tok/s) | 71.1 %, 3.00 | bbe79f6ad5f1be26 |
  | 65,536 | 18.17 / 18.18 (55.0) | 7.03 / 6.97 (143) | 96.8 %, 5.67 | 2da9e9876746bc61 |
  | 131,072 | 18.21 / 18.20 (54.9) | 7.53 / 7.53 (132.8) | 90.3 %, 5.00 | b8ba97b573341811 |
  | 250,000 | 18.33 / 18.39 (54.5) | 7.39 / 7.39 (135.3) | 88.8 %, 5.20 | 0cbd7c0b32277edd |

  At k = 1, 128k costs 0.29 ms (1.6 %) more than 16k. With drafts, the time per token follows
  acceptance, not the context. Past 16k, synth270k repeats its texts, so the drafts are accepted more
  often.
- **What is reachable without changing bits.** About 1–1.5 % of the step at k = 1, in three separate
  pieces. At k = 6 it is under 1 %. None of the pieces reaches the 2 % a change needs, so none is
  pursued:
  - **index_dense down to its keys' bandwidth.** The dense scans read ~22 MB a step at 128k: ~95 µs
    at 235 GB/s, against 177 µs at k = 1 and 259 at k = 6. Add the Reindex layers' ~60 µs over their
    4.5 MB. That is about 0.15 ms, 0.8 % at k = 1.
  - **The emit's wait at k = 1**, ~0.1 ms (0.5 %). Two ways out, either of which may suffice:
    - kv_write and the emit ahead of the wproj GEMV, or that GEMV on its own stream (model.cu);
    - wk in L2 before the emit, as decode_prep prefetches it.
  - **The attention kernel and the combine** (0.59 ms at k = 1, 0.90 at k = 6). They are latency, not
    bandwidth: about 9 MB of keys a step at k = 1. Their 64-key blocks, reduction trees and split
    order are what the bits fix.
    - **Merging the splits inside the attention launch was tried and dropped** (2026-10-11, d5f568a).
      Each row's last CTA merged the splits, in place of the combine kernel. It measured slower than
      a separate combine kernel launched with PDL, since one CTA then reads every split (attn.h).
      The figures were not kept: the commit and attn.h record only "measured slower".
- **With batching (B4).** index_dense scans each session's own keys, so the context-dependent part
  above grows with sessions × context: four sessions at 128k read about four times the keys of one.
  The time grows far less, because the rows of one session already share every key they read.
  index_dense a step (layers 2, 8 and 14, layer 20 with the pool, the four block scans) in µs, from
  bench_index: interquartile means of 100 launches, each after an L2 flush (B4's part b, 2026-10-12,
  one run):

  | rows | sessions | 16,384 | 131,072 | 250,000 |
  |---|---|---|---|---|
  | 1 | 1 | 154 | 245 | 322 |
  | 4 | 1 | 183 | 340 | 444 |
  | 4 | 4 | 249 | 522 | 831 |
  | 8 | 1 | 218 | 475 | 679 |
  | 8 | 4 | 277 | 624 | 976 |
  | 16 | 1 | 280 | 650 | 1,032 |
  | 16 | 2 | 311 | 757 | 1,127 |
  | 16 | 4 | 334 | 781 | 1,367 |

  At 250k, four sessions of four rows cost 1.32 times one session of 16. Four sessions of one row
  each cost 2.6 times one row, not four. It is the attention side's next likely target.
  - **One DIFFERENT, unexplained.** B4's part b printed it once, for index_dense_blocks against
    index_fused at 8 rows, one session and 250,000, with the fills from seed 441:
    `250000 8 1 24-36 (pool) 61.0 48.3 | 6.0 40.6 14.2 0.0 | same; DIFFERENT`. Nothing has
    reproduced it:
    - Part b's command again, 4 times (B4's binary 3 times, a diagnostic build once): 540 cases on
      the same data, all `same; same`.
    - Seed 441's case alone, 6 times (master's library 3 times, B4's 3 times): the same outputs
      every time (fingerprint 0b664e577fdceacc).
    - Master's and B4's binaries at 131k–262k (216 cases), 36 block scans checked against a host
      top-k, and 200 fresh data sets: all `same; same`.
    - On master's library: 1,600 fresh data sets of part b's kind (1,000 at 250,000, 300 each at
      131,072 and 262,144). Each kernel's first launch after the previous case's timed launches
      matched a host top-k, and 20 relaunches of each matched its first: none of 64,000 differed.
    - A reading of the code found no race: index_fused's last-CTA hand-off, index_dense's ticket and
      acquire load, both selection paths, and the harness's syncs.

    If it was a race, it is rare: none in 33,600 launches of each kernel, so at 95 % confidence it
    fires less than once in about 11,000. bench_index now keeps the outputs of a case that differs
    (`--dump`): the first differing row's keys and selections from both kernels, the host top-k of
    each, and the arguments that make the case again. The next one can be diagnosed.
  - **k = 1: +0.06 ms, not pursued.** B4's engine took 17.949 / 17.958 ms per token at 1k against
    master's 17.883 / 17.896, consistent in two ABBA pairs of one job (+0.064 ms, 0.36 %). With
    drafts there was no difference: 9.018 / 8.932 against 9.013 / 8.958. It is not attributable
    under nsys:
    - all 191 kernels both engines run have the same SASS, grids, shared memory and registers;
    - B4's decode loop is not slower there: 36 µs a step less GPU-busy time (one run each).

    The suspect is allocation and layout: B4 doubled index_dense's scratch (128 to 256 KiB, for 16
    rows), the one runtime difference found. If it ever matters, the follow-up is three builds
    (master; kMaxDecodeRows 8; 8 with the 16-row scratch), measured at k = 1 in alternating order.

    One job does not settle a difference of this size. The same kind of measurement put master's
    session-slot work at +0.12 ms at k = 1: c2dd1421 17.850 / 17.845 against 51f08122 17.988 /
    17.949, two rounds agreeing (+0.138, +0.104). In another job the same day (four runs each in two
    ABBA rounds) 51f08122 was the faster, 17.896 against 17.952 (−0.062, −0.050); over both jobs the
    two engines are 0.003 ms apart. One engine's runs spread over 0.09 ms within a job and 0.14 ms
    across the two. A fix that gives the one-session path the pre-slot instructions (attn-k1, not
    merged) ran at 17.932 in the second job, between the two; all 12 runs had the reference hash.
    Under nsys (one run each) 51f08122's decode loop had 32 µs a step less GPU-busy time than
    c2dd1421's.

    Two branches from this are kept, not merged:
    - **attn-k1** (18f903d; 57720b4 adds attn-scratch's change). The session-slot work gave the
      one-session kernels runtime table checks and struct copies. attn-k1 makes each slot-aware
      kernel a template on kSlots: attn_kernel, decode_prep, emit, carry, kv_write, kv_write_sets,
      load_tile, index_fused, merge, select and index_dense. The instance without a table compiles
      to the pre-slot code (c2dd1421's SASS, parameter loads aside), and the host picks it when no
      session table is given. Measured in the second job, built on production's source (fd3d0e4):
      - 17.951, 17.944, 17.908 and 17.924 ms per token, a mean of 17.932, each run with the
        reference hash; 51f08122 17.896 and c2dd1421 17.952 in the same rounds;
      - under nsys (one run each), the least GPU-busy time of the three: 17,999 µs a step, against
        18,035 for 51f08122 and 18,067 for c2dd1421.

      Not merged: the +0.12 ms it was built to recover did not reproduce, so there is nothing for it
      to win back.
    - **attn-scratch** (54b00a1), the follow-up's first build: index_dense's scratch sized for a
      step's rows (model::kMaxRows, 8), 128 KiB again, as before B4. Host code only: every kernel
      is unchanged, and the tests and bench_index keep the 16-row size. Not timed, and not merged,
      since B4's +0.06 ms was not pursued.

**What's left for turns.**

- **What still grows.** The dense scans: 10.6 ms at 256k for +2000, four times that for +8000. The
  scorer runs at 235 of the ~500 TFLOPS that mxf4nvf4 reaches.
- **The fixed cost.** About 200 ms of a turn's time to first token does not depend on the context or
  the turn's size:
  - the encoder streams layers 0–19's weights, about 100 ms at 200 rows;
  - the replay streams layers 20–39's, about 95 ms.
- **The first decode step** after the first token takes 25 ms, against 18.5–19 in steady decode.
- **Per token.** Each new token adds about 0.1 ms (about 10k tok/s), mostly MoE and collectives.
- These costs come from the prefill GEMM and MoE kernels and from CED's shape, not from the context.

**Correctness past golden/long's 28.5k.**

- fused_tests checks index_dense against the separate kernels, bit for bit, in 696 checks:
  - keys, top-512, top keys, block maxima and pool;
  - at 1k to 140k positions;
  - for 1, 6 and 8 rows;
  - with candidate caps that force each selection path.
- bench_index finds index_dense = index_fused = a host top-k (torch.topk's order) from 1k to 256k, at
  one and six rows.
- The tf hash is still 842278b28d87dad0, and the decode hashes are as above.
- 131,072 positions prefilled in chunks of 4096 or 1001 rows decode the same 64 tokens
  (75144a159d57121f).
- With drafts, the tokens match k = 1:
  - golden/long at 16,384: bbe79f6ad5f1be26, 2.93 tokens per step;
  - synth270k at 131,072: 094fe2b89b2793a4, 4.90 tokens per step, 7.78 ms per token against 18.73.
- The prefill's dense scans on dsk::index_topk_p and the Reindex layers on index_dense_blocks
  (2026-10-12) leave every result the same:
  - the tf hash is 842278b28d87dad0 and long28k with CED is 14650fb0739d0383 (until 2026-10-13 the
    hash's seed, see [Golden check](#golden-check); now 8f08a153d415c5c8);
  - the decode hashes above are unchanged at 1k–256k, with their contexts reached through the exact
    prefill path;
  - fused_tests passes 744 checks, 48 of them new for index_dense_blocks;
  - `serve_test.py --ced` passes 17 of 17.
- `lst.py` passes 12 of 12 at 131,072 and 250,000. It now also forks the long session: a new
  session based on it, diverging 500 positions past n, which restarts from the window snapshot
  below (249,856 at 250k).
  - The appended, forked and diverging turns each equal the same prompt run cold.
  - At 250k they take 330, 368 and 316 ms.

### End to end through the front-end

`bench/bench.py` (decode, prefill and agentic modes, as DESIGN.md's baseline) against `dsfront` and
the engine on 127.0.0.1, k = 1 (no drafts), greedy, nothing else on the GPUs. Prefill and agentic
turns are with CED (the default) on the sequence-parallel prefill. Decode is from the run in
[Drafts, measured](#drafts-dspark-measured), after the decode-path fusion; the short prompts' TTFT
from the run before this one.

| | dsgb10 | SGLang kit (DESIGN.md) |
|---|---|---|
| decode tok/s: prose / code / agentic / json | 53.7 / 53.7 / 53.6 / 53.7 | 46.8 / 77.4 / 73.6 / 106.9 |
| time to first token, short prompts | 116–145 ms | 244–280 ms |
| cold prefill tok/s at 945 / 3,747 / 16,419 / 28,654 / 57,542 tokens | 2,738 / 5,645 / 6,906 / 7,221 / 7,221 | 1.9k / 3.7k / 4.3k / 4.2k / 4.25k |
| the same, exact (`--exact`) | 2,125 / 3,575 / 3,710 / 3,777 / 3,737 | |
| the same, the first prefill path (exact, bulk all-reduces) | 1,528 / 2,103 / 2,112 / 2,133 / 2,114 | |
| agentic: first turn (15.8k tokens) | 2.25 s (first prefill path: 7.47 s) | 3.73 s |
| agentic: turns 1–5 (+1.7–2.1k tokens on 17.9–25.2k), TTFT | 0.41–0.44 s (first prefill path: 0.92–1.06 s) | 0.71–0.79 s |

Every agentic turn reused the whole previous turn (`cached_tokens` = the previous prompt plus its
completion), and each added about 2k positions, so each ran CED. Decode holds 51–52.5 tok/s from 1k
to 16k of context (serve_test.py `--speed`). The kit's decode includes its MTP drafts, which pay off
on predictable text; at k = 1 the engine leads only on prose, with drafts on all four
([Drafts, measured](#drafts-dspark-measured)).

Also checked through the front-end: the smoke test's protocol checks (models, auth, streaming
framing, a tool call with its id, a tool-result turn reusing the whole previous turn, max_tokens,
the 400s, a client disconnecting mid-stream) pass; its content checks expect the mock engine's
replies. Restated for a real model ([frontend_e2e.py](../engine/server/tests/frontend_e2e.py)): a
non-streamed answer, streamed reasoning and content with reasoning-token usage, a tool call and the
answer after its result, a stop string, an answer right after a client disconnects, four concurrent
streams, all pass. pi (the coding agent, with an isolated config) completes multi-turn tool tasks
(writing and running files) at thinking low and high, each turn reusing the last.

A longer pi session (2026-10-12, the server at its defaults: CED, drafts with the FP8 head; thinking
off, read and bash on a copy of this repository): six turns of file reads grew one session to 52k
tokens in 18 requests, then pi compacted its context (a 1.4k-token summary request, then a 28k-token
context that only a 1k snapshot covered, 3.0 s to its first token). Every request reused the whole
previous one. Time to first token: 0.11–0.22 s for a tool result of under 1k tokens at 27–47k of
context; 0.54–1.78 s for a turn that appends a file of 3–15k tokens (7–8k tok/s); 0.72 s for +5k at
52k. Decode: answers of 400–650 tokens at 83–109 tok/s (83 at 52k), tool calls of 40–170 tokens at
104–143 tok/s. The store held 294 MiB at most, and the server stopped cleanly.

#### The front-end against its references (2026-10-11)

dsfront's tokeniser, its chat encoding and its prompts were each checked against a reference as they
were written. The results were in the commit messages only, and fe_bench.py's in the front-end
agent's log.

- **The tokeniser** (01ffa42) equals the Hugging Face tokenizer, encoding and decoding, on 1,536,769
  cases:
  - 574 corpus texts: Python sources, the model's README and tech report, multilingual prose and the
    special tokens;
  - 60,000 fuzz strings over a mixed alphabet (whitespace, digits of many scripts, CJK, marks, emoji,
    special tokens);
  - every assigned code point, and every 97th unassigned one, in five contexts (1,476,195 cases).
    The sweep first failed on characters new in Unicode 16.0, the version the reference's Oniguruma
    uses, so the character classes are generated from UnicodeData 16.0.0.

  On 4.9 MB of text (1.2M tokens) it encodes 13.5–16.4 M tok/s on one thread and 67.8–77.4 M on 16,
  and the parallel output equals one pass's. It loads in about 0.1 s.
- **The chat encoding and the output parser** (f41b9a6) are a port of the publisher's encoding.py.
  They give 0 mismatches on 40,201 encode cases (2,014 of them raise) and 30,009 parse cases (6,577
  raise), and 23,432 of the parses are also streamed in random pieces. tools/gen_encoding_cases.py
  makes the cases: the publisher's five golden examples, cases for each feature and quirk, random
  conversations, and completions both well formed and broken.
- **The kit's SGLang path** (af08426). `dsfront serve --dump-dir` saved 33 requests: pi at all seven
  thinking levels with read and bash tool loops, and the smoke test. tools/check_against_sglang.py
  renders them through SGLang's DSV4.1 path with the kit's patches. All 33 prompts are byte-identical,
  and all 33 id lists equal SGLang's tokenisation.
- **The front-end's own cost** (fe_bench.py, 77df0c6), against `dsfront mock-engine --delay-ms 0
  --prefill-tps 0`:
  - 32,768 tokens streamed in 0.18 s, 182k tok/s with the SSE parsing;
  - an agent conversation of 135,294 tokens (40 tool rounds of ~2.5k tokens) has its first event
    10 ms after the request, cold;
  - its next turn, 7 ms, with 135,345 of 135,611 tokens reused.

  These numbers come from the agent's log of the run, 6 minutes before the commit.
- **Rechecked on 2026-10-12** with master's dsfront (front-end sources as of a2e18c2): the
  tokeniser's three sets and the encoding's cases still give 0 mismatches. The SGLang check was not
  rerun, since it needs requests dumped from a session and the kit's image.

```
dsfront tokcheck ~/dsgb10-data/frontend/tok_corpus.jsonl  # also tok_fuzz and tok_sweep: tools/gen_tokenizer_cases.py
dsfront enccheck ~/dsgb10-data/frontend/enc_cases.jsonl   # tools/gen_encoding_cases.py
dsfront tokbench FILE                                     # tok/s at 1, 4 and 16 threads
frontend/tools/fe_bench.py http://127.0.0.1:PORT KEYFILE  # against the mock engine, as above
```

The case generators and check_against_sglang.py run in the kit's canary-roce image; each one's
header gives its command.

#### Against the kit with its own benchmark (sparkDash 1.8.8, 2026-10-12)

**Verdict: the engine leads the published SGLang kit on every comparison.** The comparison uses the
kit's own benchmark tool.

- **c1 decode:** ahead on all four of sparkDash's types, by 3.5–16 % over the kit's latest public
  numbers (v2.4).
- **Prompt catalogs** (many prompts at c1): ahead by 16–25 %.
- **Cold prefill on real text:** 1.3–1.7× faster at 4k tokens, 1.7–1.9× from 15k to 120k.

**The tool.** [sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8 is commit
`754f40a7454fed1d26d47d1d68dd7bef930ce5ca`, its release commit; the repository has no tag. Its
production dependencies come from its lockfile (undici 8.9.0).

- **sparkDash's own modules make every request and take every time;**
  [`bench/sparkdash/sdrun.mjs`](../bench/sparkdash/sdrun.mjs) only calls them:
  - `DecodeBenchManager` for its decode bench;
  - `runStreamingRequest` for the catalogs and the real text;
  - `PrefillBenchManager` for its filler bench.
- **The protocol is the kit's** (its `docs/tp4.md`, "Measuring"):
  - 256 new tokens, temperature 0, top_p 1, thinking off;
  - a 32-token warm-up per job;
  - two jobs discarded after the boot;
  - then five rounds of the four types, each type's median reported.
- **sparkDash's waves send `min_tokens`, `ignore_eos` and `stop: []`.** dsfront honours
  `ignore_eos`, so every request returned 256 tokens with finish "length".
- **The setup:**
  - a dev engine at serve's defaults (`--slots 1`, CED, drafts with the FP8 drafter head,
    `--max-ctx 262144`) and dsfront on 127.0.0.1, nothing else on the ring;
  - the binaries are aa27273's build: master's engine and dsfront plus the end-of-turn timers,
    which only log;
  - rank 0 runs `gpu-clock-cap` at 2200 MHz, the cap the kit's v2.4 fleet
    runs at.
- **How to run it:** [`bench/sparkdash/part.sh`](../bench/sparkdash/part.sh) runs each mode as a
  ring part, and [`sdreport.py`](../bench/sparkdash/sdreport.py) joins the records with rank 0's
  log.

**What the kit's figures are.**

- **v2.1:** the local copy of the kit, `docs/tp4.md`, "Results", 2026-09-25, on a switched RoCE
  fabric.
- **v2.4:** knapcio's DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 at 27616be, 2026-10-12. Each cell is
  the better of two sweeps, with c1 the median of three.
- Both were taken from the kit's documentation, not measured here.

| | dsgb10 | kit v2.1 | kit v2.4 |
|---|---|---|---|
| c1 prose | **94.9** (94.4–95.1) | 87.7 | 89.9 |
| c1 code | **135.6** (135.3–136.1) | 124.8 | 131.0 |
| c1 structured | **171.6** (171.5–171.8) | 152.4 | 158.7 |
| c1 json | **149.3** (148.9–149.5) | 118.9 | 128.4 |
| catalogs at c1: prose / structured | **72.9 / 115.3** | 58.4 / 94.3 (v2.2: 61.1 / 99.0) | |
| decode step on prose | 30.1 ms at 2.83 tokens a step (c1 prompt); 26.8 ms at 1.93 (catalog) | 32.4–33.6 ms at 2.27 | |
| decode step on code | 38.0 ms at 5.10 tokens a step (c1 prompt); 36.9 ms at 4.97 (catalog) | 38.9–39.4 ms at 3.87 | |
| cold prefill on real text, ~4k / 15k / 32k / 62k / 120k tokens | **6.2–7.0k / 8.5–8.9k / 9.1k / 9.1–9.2k / 9.3–9.4k** | 4.2–4.7k / 4.8–4.9k / 4.8k / 4.9–5.0k / 4.8–4.9k | 5.0k (one 388k prompt) |

- **c1 against v2.4:** prose +5.6 %, code +3.5 %, structured +8.1 %, json +16.3 %.
  - Against v2.1: +8 %, +9 %, +13 %, +26 %.
  - The greedy text was the same in all five runs of each type, and the runs agree within 0.7 %.
  - The time to the first token was 27–35 ms, from the warm-up's kept session. Code's was 156–159 ms,
    because sparkDash warms code with a different prompt.
- **Our step times come from rank 0's log.** The wall time is the decode time over the verify
  steps, including the end-of-turn step. The GPU times were 29.5 ms (prose c1) and 37.2 ms (code
  c1).
  - The kit's step times come from its own probe, on six prose and six code prompts of its own.
  - Per token on prose, the kit's probe gives 14.6 ms. We give 10.6 ms on the c1 prompt and 13.9 ms
    over the catalog.
- **The catalogs.** sparkDash 1.8.8 has no many-prompt bench, so its catalogs were run at c1, its
  bench's way (the 32-token warm-up, then 256 tokens):
  - TEXT_PROMPTS (16 prose prompts): 62.1–85.3 tok/s, mean 72.9;
  - STRUCTURAL_PROMPTS (18): 81.2–153.3, mean 115.3;
  - DECODE_CODE_TASKS (32): 122.3–148.9, mean 136.4, 4.97 tokens a step.
- **The kit's "45 varied prompts" come from its own scripts.**
  - Its doc lists the set as "prose, structured and other catalogs", with three figures, read here
    in that order.
  - knapcio's `ab_inboot.py --prompts sparkdash` uses exactly these two catalogs.
  - The 11 others are not published, and the kit's runs stop where the model does (no
    `ignore_eos`).
- **Real text** is this repository's documentation and source code at 078837d (`git archive`,
  [`corpus.py`](../bench/sparkdash/corpus.py): a seeded shuffle of the files, 2.82 characters a
  token).
  - Each prompt has a unique header (a UUID) and a slice of text no other prompt uses, in
    sparkDash's prefill request: 8 new tokens, tok/s = prompt tokens / time to the first token.
  - Two passes. Every prompt was cold (0 reused), and rank 0's prefill time is within 1 % of
    sparkDash's time to the first token.
  - The kit used its own docs and code, so prompt sizes match only roughly. Our "62k" came out at
    56.7k and 58.6k tokens.
- **sparkDash's own prefill bench** fills with one repeated token, so every filler token hits the
  same Engram row, here as in the kit. It is reported apart: 8.2k / 9.8k / 10.2k / 10.4k / 10.5k
  tok/s at 4k / 16k / 32k / 64k / 128k (two runs within 0.4 %).
  - The kit's v2.1 gives 4.1–4.8k / 5.8–5.9k / 5.9k / 5.9k / 5.8k, and v2.4 4.8k / 5.8k / 5.9k /
    6.0k / 5.9k.
  - At 128k the filler runs 12–13 % faster than real text here.

**Caveats.**

- **Greedy text depends on summation order.** A stack with another fabric or another all-reduce
  follows a different greedy text on the same prompt, so the single-prompt c1 figures compare two
  different texts. The catalogs and the step times are the fairer comparison, as the kit's own doc
  says.
- **The fabrics differ:** the kit runs over a switched RoCE fabric, we run over a switchless ring.
- **The weights differ:** the kit's published runs use deepseek-ai's official checkpoint (its scripts'
  default; v2.4's README says "native weights"), and these runs used dealignai's variant
  ([README](../README.md)). Greedy texts and draft acceptance
  depend on the weights, so this is another reason the catalogs and step times are the fairer measures.
- **Only c1 is compared.** The kit also publishes c2–c16. Production here runs `--slots 1`, and
  batching is merged but off ([Scheduling](#scheduling-time-slicing-and-batching)).

The raw data are in rank 0:~/dsgb10/editruns/sd1/{decode,prefill}: sdrun's records, rank 0's and
dsfront's logs, and sparkDash's own history files.

### Agentic edits through pi's tools (2026-10-12)

`bench.py --modes edit` ([editbench.py](../bench/editbench.py); tasks in
[edit_bench.json](../bench/edit_bench.json)) drives twelve small coding tasks the way pi 0.84.3 does:
pi's system prompt, its read, edit and write tools, and their result and error texts (no bash;
[THIRD_PARTY.md](../bench/THIRD_PARTY.md)). The tasks: rename a helper, add a parameter, fix a comment,
let the environment override three addresses, print a usage error, turn a loop into a helper and a
comprehension, name six constants, extract a function, add docstrings, rename a constant across two files
(a 38 KB read), and rewrite two files whole with write. The calls apply to scratch copies of this
repository's files at e522153; each task ends with regex checks and, for Python, a `compile()`. Greedy,
one request at a time, the engine at serve's defaults (`--slots 1`, CED, drafts) behind dsfront with
`--dump-dir`. The decode rate and the drafts come from rank 0's per-request lines; tokens per step is 1
plus the drafts accepted per step.

Thinking off, all twelve tasks: 39 requests, 4,855 output tokens, every task's checks passed.

| kind of call | turns | output tokens | decode tok/s | tokens/step | drafts verified/step (accepted) | steps verifying 5 | TTFT median |
|---|---|---|---|---|---|---|---|
| edit | 10 | 2,165 | 141.5 | 5.52 | 4.74 (95 %) | 87 % | 351 ms |
| write | 2 | 1,231 | 138.2 | 5.23 | 4.55 (93 %) | 76 % | 255 ms |
| read | 15 | 806 | 125.7 | 5.60 | 4.97 (93 %) | 96 % | 247 ms |
| prose (the closing answers) | 12 | 653 | 104.6 | 4.09 | 4.19 (74 %) | 58 % | 115 ms |
| all | 39 | 4,855 | 131.8 | 5.22 | 4.63 | 81 % | 244 ms |

Thinking max (the default of the pi setup this engine serves), four of the tasks (the rename, the
comment, the loop, the small rewrite): 15 requests, 2,133 output tokens, at most 131 reasoning tokens in
a turn; every check passed.

| kind of call | turns | output tokens | decode tok/s | tokens/step | drafts verified/step (accepted) | steps verifying 5 | TTFT median |
|---|---|---|---|---|---|---|---|
| edit | 4 | 909 | 128.9 | 4.84 | 4.37 (88 %) | 71 % | 315 ms |
| write | 1 | 423 | 131.6 | 4.96 | 4.45 (89 %) | 71 % | 243 ms |
| read | 6 | 518 | 117.7 | 4.66 | 4.41 (83 %) | 69 % | 250 ms |
| prose | 4 | 283 | 103.2 | 3.81 | 3.89 (72 %) | 48 % | 136 ms |
| all | 15 | 2,133 | 122.6 | 4.65 | 4.32 | 67 % | 243 ms |

Edits do not run below the JSON prompt's 136 tok/s: at thinking off they decode at 141.5 tok/s, with
5.52 of the 6 tokens a step can give at five drafts (a 6-row step: 38.6 ms of GPU time). Within five
drafts, prompt-lookup drafts can add at most 0.48 tokens a step on edits (+9 %). The copies are long,
though: 48 % of the edit calls' tokens and 65 % of the writes' lie in verbatim runs of 8 or more tokens
from the prompt. Drafts from the prompt pay off only past five tokens, that is with verify steps of 7–8
rows (`kMaxSessRows` ≤ 8). Time to first token is the prefill of the last tool result: 0.11–1.46 s. The
read turns include each task's first request.

The traces (each request's prompt and output ids, as golden-style directories for `spectf --prompt N`)
are in rank 0's /data/dsgb10/bench/edit/r1, listed in its `index.jsonl`. That run had this machine's paths
in pi's system prompt (the docs' install path and the working directory); edit_bench.json now templates
both with neutral paths, so a new run's prompts differ by those few tokens. pi sends no temperature, so
its own requests sample at dsfront's defaults (T 1.0, top-p 0.95): [At dsfront's sampling
defaults](#at-dsfronts-sampling-defaults). To run it: dsfront with `--dump-dir D`, then
`bench.py --modes edit --edit-out DIR [--edit-thinking off,max] [--edit-tasks ...] [--edit-repeats N]
[--edit-temperature none]`, then
`editbench.py report DIR --dump-dir D --log RANK0_LOG [--traces T] [--baseline JOINED.jsonl]`.
`edit_pi_check.py` checks the ported tool texts against pi's own code under node (43 cases, all
identical).

#### At dsfront's sampling defaults

The same bench as pi sends it: no temperature and no seed (`--edit-temperature none`), so dsfront
samples at T 1.0 and top-p 0.95 and the engine draws a seed per request; the neutral paths in the
prompt. Thinking off, all twelve tasks three times: 113 requests, 14,883 output tokens. Thinking max,
the same four tasks once: 13 requests. Every task's checks passed (36 of 36, 4 of 4); three repeats
took 3, 4 and 5 turns for the extraction, 3 or 4 for the two-file rename. The engine and dsfront were
master's (ffdaab5's build), as in the greedy run. Greedy → sampled:

| thinking off | turns | decode tok/s | tokens/step | drafts accepted | drafts verified/step | steps verifying 5 | gate |
|---|---|---|---|---|---|---|---|
| edit | 10 → 31 | 141.5 → 139.6 | 5.52 → 5.48 | 95 % → 95 % | 4.74 → 4.73 | 87 % → 87 % | 0.394 → 0.396 |
| write | 2 → 6 | 138.2 → 136.1 | 5.23 → 5.12 | 93 % → 93 % | 4.55 → 4.44 | 76 % → 73 % | 0.447 → 0.441 |
| read | 15 → 40 | 125.7 → 120.5 | 5.60 → 5.37 | 93 % → 88 % | 4.97 → 4.97 | 96 % → 97 % | 0.251 → 0.245 |
| prose | 12 → 36 | 104.6 → 96.7 | 4.09 → 3.73 | 74 % → 68 % | 4.19 → 4.03 | 58 % → 52 % | 0.254 → 0.240 |
| all | 39 → 113 | 131.8 → 128.7 | 5.22 → 5.08 | 91 % → 89 % | 4.63 → 4.57 | 81 % → 79 % | 0.362 → 0.361 |

| thinking max | turns | decode tok/s | tokens/step | drafts accepted | drafts verified/step | steps verifying 5 | gate |
|---|---|---|---|---|---|---|---|
| edit | 4 → 3 | 128.9 → 121.7 | 4.84 → 4.57 | 88 % → 80 % | 4.37 → 4.46 | 71 % → 74 % | 0.359 → 0.358 |
| write | 1 → 1 | 131.6 → 123.7 | 4.96 → 4.57 | 89 % → 83 % | 4.45 → 4.28 | 71 % → 67 % | 0.405 → 0.381 |
| read | 6 → 5 | 117.7 → 122.4 | 4.66 → 5.10 | 83 % → 84 % | 4.41 → 4.89 | 69 % → 91 % | 0.273 → 0.282 |
| prose | 4 → 4 | 103.2 → 86.8 | 3.81 → 3.01 | 72 % → 58 % | 3.89 → 3.45 | 48 % → 33 % | 0.264 → 0.250 |
| all | 15 → 13 | 122.6 → 114.2 | 4.65 → 4.26 | 85 % → 77 % | 4.32 → 4.24 | 67 % → 65 % | 0.331 → 0.325 |

- **Edits and writes keep their speed.** By repeat, edits decode at 139.3, 141.0 and 138.5 tok/s
  (5.47, 5.55, 5.43 tokens a step; greedy 141.5 and 5.52), writes at 134.1–138.7 (greedy 138.2).
  Copied text is low-entropy: the sampled token is the one greedy would take, and the drafts are
  accepted as often (95 % and 93 %).
- **Prose loses 3–10 %.** 95.2, 101.9 and 93.7 tok/s by repeat against 104.6: at T 1.0 a draft is
  accepted only when the sample equals it, 65.9–70.9 % of the time against 73.7 %. Reads lose 3–6 %
  (88 % accepted against 93 %): a read call is ~55 tokens, and its sentence before the call is prose.
- **Thinking max** is one run of 13 requests, so its per-kind figures move by several percent between
  runs. Its reasoning is prose: edit turns, whose reasoning comes before the call, lose 6 % (80 %
  accepted against 88 %), the closing answers 16 %. On the client's clock the edit turns' reasoning
  came at 105.9 tok/s and their answers at 139.6.
- **The gate's choices barely move.** Its threshold (λ·ΔC, the mean over the steps) stays within
  0.006 of greedy on edits, writes and reads, and eases on prose (0.254 → 0.240) as the rate λ falls
  with the acceptance. Steps verifying all five drafts: 87 % on edits either way; on prose 58 % → 52 %
  (47–65 % by repeat).
- Time to first token is unchanged: 241 ms median against 244.

The traces are in rank 0's /data/dsgb10/bench/edit/r2 (126 turns, `index.jsonl`; directories
`r2-off/off/rep{0,1,2}/TASK/tNN` and `r2-max/max/rep0/TASK/tNN`). Their outputs are the sampled
tokens, which a sampled verify step compares its drafts with, so `spectf` on them gives the
acceptance of sampled decoding.

### Where it stands on release candidate 5 (2026-10-13)

This is the variant's record. README's tables now give the official checkpoint's figures, measured the same day
([Where it stands on the official checkpoint](#where-it-stands-on-the-official-checkpoint-2026-10-13)).

README's two "Where it stands" tables were measured on 10-11 and 10-12, before the exact weight codes (2812072)
and B′, the paced L2 prefetch (ae57e04), which production now runs. Here every row of ours is measured again on
release candidate 5, production's build since 10-13 04:13 (c247a78; engine b352b870…, dsfront ce116cea…). Each
row keeps the tool, prompts, token counts, greedy settings and streams it was first measured with. The kits'
columns are their documents' figures and stay as recorded. The agentic-edit bench at dsfront's sampling
defaults, which is what pi gets, is measured as well.

**Verdict: no row got slower.**

- **Decode with drafts** is 9.1–10.1 % faster on bench.py's four types, 7.7–9.2 % on sparkDash's c1 types and
  7.6–8.9 % on its catalogs. A step takes 7–8 % less time at the same tokens a step.
- **Decode without drafts** (k = 1): 59.1 tok/s against 53.6–53.7, +10.1 %.
- **Cold prefill** is 23–31 % faster from 16k to 57k tokens on bench.py's prompts, whose originals predate the
  prefill work of 10-12 00:43–01:00. On sparkDash's real text, first measured after that work, it is 0.7–1.6 %
  faster.
- **bench.py's agentic turns:** the first (15.8k tokens) takes 1.80 s against 2.25–2.38 s. The turns that add
  ~2k tokens take 0.39–0.42 s against 0.41–0.45 s, a smaller gain (median −6 to −8 %).
- **As pi gets it** (the edit bench at T 1.0 and top-p 0.95, thinking off, 112–113 requests): 131.4 and 134.2 s
  end to end against 143.7 s (−7.6 %), with 6.8 % more output tokens per second of it. Decode runs at
  138.9–139.9 tok/s against 128.7, and the median time to the first token is 230–231 ms against 241.
- **The greedy outputs are the originals'.**
  - bench.py keeps no text, but its eight decode requests match the original's request for request: the
    verify steps, the drafts accepted per step, and how many steps verified 0 to 5 drafts.
  - sparkDash's 20 c1 requests and 66 catalog prompts return the original's text, in both sessions.
  - So the greedy decode rows compare the same tokens.

**What release candidate 5 has that the originals lacked** (master's first-parent history):

- **Since bench.py's decode run** (10-12 02:14, master after 1f09107):
  - the persistent normed GEMVs (e40eabc) and the router's loads in one round trip (6a224ac), each within 0.4 %
    with drafts in its own A/B;
  - the exact weight codes (2812072), whose A/B gave bench decode 2.30–4.13 % less time a token with drafts;
  - B′ (ae57e04): −3.54 and −3.59 % with drafts in its two A/B jobs;
  - the reduce-scatters' race fix (1128401);
  - merges that are off by default or that their own measurements found neutral at one slot and up to six rows:
    B2, B4, bundle B's stage 1, the split mHC boundary, the small-M tiles.

  With drafts the A/Bs predict 6–8 % on bench decode, and the bench gained 9–10 %. The remaining 1–2 % is not
  attributed here. At k = 1 they predict about 10 % (−1.4, −1.9, −2.0 and −4.2 to −4.5 % a token), and the
  bench gained 10.1 %.
- **Since the prefill and agentic runs**, also:
  - the prefill's wo_b by all-to-all, with the MoE combine inside its reduce-scatter (59b67ef; README's "8.1k in
    the engine since");
  - 8192-row CED chunks by default (aa05c03);
  - the prefill's dense index scans on `index_topk_p` (2e92dd7).

  Those runs were a k = 1 session with CED at 4096-row chunks and `--max-ctx 65536` (10-11 23:12) and the M7
  branch with `--spec` (10-12 00:33).
- **Since sparkDash's run and the sampled edit bench r2** (10-12, aa27273's and ffdaab5's builds): 2812072,
  ae57e04, 1128401 and the merges that are off by default.

**How it was measured.**

- **The binaries** are release candidate 5's, from ~/dsgb10/candbin5 (byte-identical to production's), used as
  they are. Each ring part checked their hashes first.
- **The server** has production's arguments: `run4.sh --mode serve --max-ctx 262144`, and serve's defaults
  otherwise (`--slots 1`, CED at 8192-row chunks, drafts with the FP8 drafter head, B′ on, a 12 GiB session
  store). The k = 1 parts add `--no-spec`. dsfront listens on 127.0.0.1 with `--hold-s 120`, plus `--dump-dir`
  for the edit bench.
- **Each session is one ring part** (`ring.sh` with `OWNER=bench PREEMPT=0`), with a fresh engine on the ring
  that the idle service lent:
  - b1, sparkDash's decode and prefill (sd1), e1, b2, sd2, e2 and b3 ran from 05:26 to 05:55, 2–5 minutes
    each;
  - k1a and k1b ran from 05:57 to 06:02.

  No other job ran on the ring. rank 0 ran `gpu-clock-cap` at 2200 MHz, as for every original.
- **The tools are the originals':**
  - **bench.py.** Its decode, prefill and agentic modes have not changed since the baseline (6d35638); the edit
    mode was added beside them. Each session ran `--modes decode,prefill,agentic` at the defaults: two
    repetitions of 512 tokens, greedy, and prompts of 1, 4, 16, 32 and 64k.
  - **sparkDash** 1.8.8 at 754f40a, with undici 8.9.0, run through `sdrun.mjs` (unchanged since 10-12 but for
    a comment). Its corpus was regenerated from 078837d with the same sha256 (298391b7…), and the kit's
    protocol is as in [Against the kit](#against-the-kit-with-its-own-benchmark-sparkdash-188-2026-10-12).
  - **The edit bench** used r2's settings: `--modes edit --edit-temperature none`, thinking off with three
    repeats of the twelve tasks, thinking max on four tasks once. `editbench.py report` then compared each
    run with r2.
- **The analysis** reads the originals' raw logs and RC5's with the same parsers:
  - `rc5table.py` takes bench.py's lines, with the unrounded rates from its json, and rank 0's per-request
    lines;
  - it joins sparkDash's records as `sdreport.py` does, and aggregates the edit bench as `editbench.py` does;
  - `changes.py` makes the tables below.

**bench.py**, README's first table. The originals:

- decode: the run with the FP8 drafter head (10-12 02:14; [Drafts, measured](#drafts-dspark-measured));
- the rest: the M7 run with `--spec` (10-12 00:33), and in brackets the k = 1 CED run (10-11 23:12);
- without drafts: the M7 build's session without `--spec`, run right after it (10-12 00:38).

Change is RC5's mean against the original. Spread is (largest − smallest) / mean over RC5's sessions.

| | original | RC5: b1, b2, b3 | spread | change |
|---|---|---|---|---|
| decode tok/s, prose | 69.4 | 75.85, 75.67, 75.51 | 0.5 % | +9.1 % |
| decode tok/s, code | 101.9 (101.85 unrounded) | 111.84, 111.66, 111.47 | 0.3 % | +9.6 % |
| decode tok/s, agentic | 99.8 | 109.35, 109.20, 108.95 | 0.4 % | +9.4 % |
| decode tok/s, JSON | 136.0 | 150.09, 149.74, 149.51 | 0.4 % | +10.1 % |
| time to first token, short prompts | 128–150 ms, median 146 | 123–144, 125–147, 123–143 ms; medians 142, 144, 141 | 2.1 % (medians) | −2.7 % (medians) |
| cold prefill, 16,419 tokens | 6,494 tok/s (6,906) | 8,534, 8,558, 8,502 | 0.7 % | +31.4 % (+23.5 %) |
| cold prefill, 28,654 tokens | (7,221) | 8,909, 8,882, 8,914 | 0.4 % | +23.3 % |
| cold prefill, 57,542 tokens | (7,221) | 9,239, 9,201, 9,200 | 0.4 % | +27.6 % |
| agentic: first turn (15,774 tokens) | 2.38 s (2.25 s) | 1.80, 1.80, 1.81 s | 0.4 % | −24.4 % (−20.1 %) |
| agentic: turns 1–5 (+1.7–2.0k tokens on 17.8–25.1k), to first token | 0.42–0.45 s, median 0.43 (0.41–0.44, median 0.42) | 0.39–0.42, 0.39–0.42, 0.39–0.41 s; medians 0.394, 0.396, 0.399 | 1.3 % (medians) | −7.9 % (−5.6 %) (medians) |
| decode tok/s without drafts (k = 1), all eight requests | 53.6–53.7, mean 53.63 | 59.1–59.4, mean 59.21 (k1a); 58.0–59.3, mean 58.90 (k1b) | 0.5 % | +10.1 % |

- **Smaller prompts.** The cold prefill at 945 and 3,748 tokens ran at 2,906–2,933 and 6,275–6,310 tok/s, against
  2,725 and 5,489 in the M7 run (2,738 and 5,645 at k = 1).
- **The first request after each start** is left out of the short prompts' time to first token, as before:
  165–168 ms on RC5, 310 in the M7 run.
- **Every agentic turn reused the whole previous turn,** as in the originals.
- **The engine's own prefill** (rank 0's log) was 1–3 % faster than bench.py's figures at 1k–4k and within
  0.6 % from 16k up (8,548–8,604 tok/s at 16k, 9,215–9,257 at 57k).

**sparkDash 1.8.8**, README's second table. The original is 10-12's run (rank 0:~/dsgb10/editruns/sd1). c1 is
the median of five rounds (their range in brackets), the catalogs are the mean over their prompts, and a step's
time is rank 0's decode time over its verify steps.

| | original | RC5: sd1, sd2 | spread | change |
|---|---|---|---|---|
| c1 prose | 94.9 (94.4–95.1) | 102.2 (102.1–102.3), 102.1 (102.0–102.2) | 0.1 % | +7.7 % |
| c1 code | 135.6 (135.3–136.1) | 148.1 (147.9–148.4), 148.1 (147.9–148.4) | 0.0 % | +9.2 % |
| c1 structured | 171.6 (171.5–171.8) | 186.3 (185.6–186.7), 186.4 (185.8–186.6) | 0.1 % | +8.6 % |
| c1 JSON | 149.3 (148.9–149.5) | 162.5 (162.3–162.9), 162.6 (162.4–162.7) | 0.0 % | +8.9 % |
| catalogs at c1: prose / structured / code | 72.9 / 115.3 / 136.4 | 78.5 / 124.9 / 148.6, 78.5 / 125.0 / 148.6 | ≤ 0.1 % | +7.6 / +8.4 / +8.9 % |
| decode step, prose (c1 prompt) | 30.11 ms at 2.83 tokens | 27.94, 27.96 ms at 2.83 | 0.1 % | −7.2 % |
| decode step, code (c1 prompt) | 37.99 ms at 5.10 tokens | 34.81, 34.83 ms at 5.10 | 0.0 % | −8.3 % |
| decode step, catalogs: prose / code | 26.77 ms at 1.93 / 36.91 at 4.97 | 24.87, 24.88 / 33.90, 33.90 | 0.0 % | −7.1 / −8.1 % |
| cold prefill, real text, tok/s: ~4k / 15k / 32k / 62k / 120k | 6,245, 6,950 / 8,519, 8,866 / 9,090, 9,103 / 9,204, 9,099 / 9,313, 9,362 | 6,296, 7,062 / 8,684, 9,020 / 9,180, 9,176 / 9,304, 9,232 / 9,396, 9,445 (sd1); 6,303, 7,086 / 8,624, 8,997 / 9,130, 9,158 / 9,295, 9,206 / 9,406, 9,437 (sd2) | ≤ 0.5 % | +1.3 / +1.6 / +0.7 / +1.2 / +0.9 % |
| sparkDash's filler bench, tok/s at 4k / 16k / 32k / 64k / 128k | 8.2k / 9.8k / 10.2k / 10.4k / 10.5k | 8.2k / 9.8k / 10.2k / 10.4k / 10.5k in both | ≤ 0.3 % | −0.2 to +0.4 % |

- **The real text runs 8.6k–9.4k tok/s from 15k to 120k tokens,** against 8.5k–9.4k. Every prompt was cold, and
  rank 0's prefill time stays within 1 % of sparkDash's time to the first token.
- **Per token on prose** a step now costs 9.9 ms on the c1 prompt and 12.9 ms over the catalog (10.6 and 13.9
  before).
- **The time to the first token at c1** was 26–31 ms, and 152–153 ms on code, for the reason given in [Against
  the kit](#against-the-kit-with-its-own-benchmark-sparkdash-188-2026-10-12): sparkDash warms code with a
  different prompt.

**The edit bench at dsfront's sampling defaults**, against r2 (10-12, ffdaab5's build;
[At dsfront's sampling defaults](#at-dsfronts-sampling-defaults)):

- End to end is the sum of the requests' client times, from sending to the stream's end. The tasks' wall times
  agree within 0.4 s, and so do dsfront's request times.
- The time to the first token is the median of dsfront's per-request `ttft_s`.
- The decode rate is rank 0's tokens over its decode time.

| thinking off: twelve tasks, three times | r2 | RC5: e1, e2 | spread | change |
|---|---|---|---|---|
| requests; output tokens | 113; 14,883 | 112; 14,571 and 113; 14,805 | | |
| end to end | 143.7 s | 131.4, 134.2 s | 2.1 % | −7.6 % |
| output tokens per second of it | 103.6 | 110.9, 110.4 | 0.5 % | +6.8 % |
| decode tok/s, tokens a step | 128.7 at 5.08 | 139.9 at 5.09, 138.9 at 5.04 | 0.7 % | +8.3 % |
| time to first token, median | 241 ms (client 246) | 230, 231 ms (client 235, 236) | 0.4 % | −4.2 % |
| prefill and decode time (rank 0) | 28.7 s and 114.7 s | 27.6 and 103.4 s; 27.9 and 105.8 s | | −3.4 % and −8.8 % |
| checks passed | 36 of 36 | 36 of 36, 36 of 36 | | |

- **Thinking max** (four tasks once) is one run of 13–14 requests, so its figures move by several percent between
  runs.
  - r2: 13 requests, 2,202 output tokens, 22.2 s end to end (99.0 tokens a second), decode 114.2 tok/s at 4.26
    tokens a step, 242 ms to the first token.
  - RC5: 14 and 13 requests, 2,320 and 1,867 tokens, 21.9 and 17.3 s (105.9 and 107.7 tokens a second), decode
    122.9 at 4.17 and 129.0 at 4.51, 229 and 236 ms.
  - Every check passed (4 of 4, twice).
- **The fairer measure is output per second.** The engine draws a seed per request, so every run's answers
  differ: e1 and e2 wrote 2.1 % and 0.5 % fewer tokens than r2. Tokens a step are within 0.8 % of r2's.

**Where it differs from the originals.**

- **The disk tier** was 8 GiB in ~/dsgb10/bspill, where production has 32 GiB in /data/dsgb10/spill.
  Production's `--disk-tag service --keep-gib 4` act only at a stop or a start, and were left out. No request
  reached the tier: the store held at most 1.8 GiB of its 12 (the 131k filler prompt), with no spill and no
  eviction in any part.
- **README's first table mixed three sessions:**
  - decode from a decode-only session (10-12 02:14);
  - time to first token, the 16k prefill and the 2.38 s first turn from the M7 branch with `--spec` (10-12
    00:33, prefill lengths 1, 4 and 16k);
  - the 28k and 57k prefill and the 2.25 s first turn from a k = 1 session (10-11 23:12).

  Each RC5 session ran all three modes at production's arguments.
- **sparkDash's real-text prompts** are the same slices, but the warm-up's UUID header moves their token counts by
  up to 0.04 % (122,963 → 123,022 and 122,915 tokens at "120k").
- **bench.py's prefill and agentic prompts carry a random nonce,** so their sizes move by a token between runs
  (944 or 945, 28,653 or 28,654 tokens). The agentic answers' lengths change with it, and so do the later
  turns' sizes (turn 5: 25,079–25,094 tokens against 24,991 and 25,078).

**Two smaller findings.**

- **k1b's last two requests** (JSON) ran at 58.4 and 58.0 tok/s, 1.3–1.9 % below the six before them and below
  k1a's JSON (59.2, 59.1). A token at k = 1 is the same work whatever the prompt, and no other job held the
  ring, so this is the machine's spread, not the prompt's. Likewise bench decode drifted down 0.45 % from b1 to
  b3 over 27 minutes.
- **The agentic turns that add ~2k tokens gain less** (6–8 %) than the first turn (20–24 %). Where a small turn's
  time goes is in [A small turn's time to first token](#a-small-turns-time-to-first-token-2026-10-12).

**The raw data** are in rank 0's /data/dsgb10/bench/rc5:

- b1, b2, b3: bench.py's log and json, and rank 0's and dsfront's logs;
- sd1, sd2: each {decode,prefill} holds sdrun's records, sparkDash's history files and the logs, and
  sdreport.md is `sdreport.py`'s report;
- e1, e2: the edit bench's turns and tasks, dsfront's dumps, and editbench.py's report;
- k1a, k1b: the k = 1 sessions;
- logs/: each ring part's own log;
- originals/: the originals' bench.py, rank 0 and dsfront logs; 10-12's sparkDash run and r2 stay in
  ~/dsgb10/editruns;
- scripts/: `bpart.sh`, one ring part; `drive.sh`, the driver; `rc5table.py` and `changes.py`, the analysis;
- table.md and rc5table.txt: their output.

### Where it stands on the official checkpoint (2026-10-13)

From 2026-10-13 the figures to publish are measured on deepseek-ai's official checkpoint, DeepSeek-V4.1-Flash at
`fb2764a5` ([The official checkpoint's weights](#the-official-checkpoints-weights-2026-10-13)). This section
measures every row of README's two "Where it stands" tables again on its files, with release candidate 5's engine
code. The SGLang kit was measured on the same checkpoint, ring and client the same day, between our first and
second sets of bench.py sessions:

- O1: three sessions of ours, 10:54–11:10;
- K: the kit's window, 11:13–11:32;
- O2: three more of ours, 11:39–11:54;
- then the rest of ours, 11:57–12:37.

Production kept serving the variant's files and lent the ring for all of it. Times are the boxes' local time.

**Verdict: on the same checkpoint, the same day, one request at a time, the engine leads the kit on every
bench.py row.**

- **bench.py:**
  - decode with drafts 77.4 / 115.4 / 102.9 / 153.7 tok/s on prose / code / agentic / JSON, against the kit's
    50.9 / 85.6 / 76.3 / 107.7: 1.52, 1.35, 1.35 and 1.43 times;
  - cold prefill from 16k to 57k tokens 8.5k–9.2k tok/s, against 4.3k: 2.0–2.2 times;
  - the agentic transcript's first turn 1.81 s against 3.72 s, and its turns that add ~2k tokens 0.39–0.42 s
    against 0.68–0.79 s;
  - the time to the first token on short prompts, a median of 143 ms against 230.
- **The steps cost what they cost on the variant; the tokens a step differ.** Against release candidate 5 on the
  variant the same morning ([Where it stands on release candidate 5](#where-it-stands-on-release-candidate-5-2026-10-13)):
  - prefill, the agentic turns, the time to the first token and decode without drafts are the same within
    0.7 %, and so are the times of sparkDash's decode steps (within 0.8 %);
  - decode with drafts follows the drafts' acceptance on each checkpoint's greedy texts: −5.7 to +3.4 % on
    bench.py's four prompts, −7.6 to +2.0 % on sparkDash's four, −2.1 to −0.3 % over its catalogs.
- **Against the MiaAI kit's published figures** (sparkDash 1.8.8; theirs on the official checkpoint too):
  c1 decode 95.0 / 151.0 / 185.8 / 150.2 tok/s against v2.4's 89.9 / 131.0 / 158.7 / 128.4 (+5.7 to +17.1 %),
  the catalogs 78.1 / 122.3 against v2.2's 61.1 / 99.0 (+28 and +24 %), real text 8.7k–9.5k tok/s from 15k to
  120k tokens against 4.8k–5.0k.
- **As pi gets it** (the edit bench at dsfront's sampling defaults, thinking off): 138.5 and 134.2 s end to end
  for 15,284 and 15,011 output tokens, 110.4 and 111.9 tokens a second of it (on the variant that morning 110.9
  and 110.4), decoding at 137.4 and 138.5 tok/s, with 229–230 ms to the first token (median); 36 of 36 checks in
  both runs.
- **Correctness on the official files:**
  - GSM8K 99 of 100; the kit, on the same checkpoint the same day, 100 of 100;
  - passkey retrieval 18 of 18 at 33k–220k tokens;
  - teacher-forced along the kit's greedy outputs on four prompts, the argmax is the kit's token at 260 of 271
    steps (on the variant: 256 of 267).
- **O2 against O1:** 0.997 to 1.004 on the four decode types, within the method's 1 %.

**How it was measured.**

- **The binaries** were built from 530af7f (master at 10:35) and frozen in ~/dsgb10/candbin6: engine
  `dab79c98…`, dsfront `971a6298…`. Their engine and front-end code are release candidate 5's (c247a78):
  since then the engine's sources gained only ring_test's stall hook, compiled out of the engine, and
  comments; dsfront's gained comments. They differ from candbin5's binaries in their embedded build paths.
  Each ring part checked their hashes first.
- **Part 0 ran the reference hashes** with them before any benchmark (10:42–10:48, `run4.sh --big` as in [The
  official checkpoint's weights](#the-official-checkpoints-weights-2026-10-13)). Each equals its documented
  value:
  - on the official files: tf `c7605e0b6ba8d9d2`, 1k at k = 1 `7941647b162df1a8`, sampled 1k
    `02ef3e0858d7eb45`, 16k CED `8790ee9388f4abd8`;
  - on the variant's: tf `842278b28d87dad0`, 1k at k = 1 `febc7dc54ee40a39`.

  The long28k CED hash was not run: candbin6 (530af7f) predates b47c510, before which that run printed the
  hash's seed. Since b47c510 it is `de99bfce6ea96ba8` on the official files.
- **The weights** are `--weights ~/dsgb10/weights-official/rank.dsw` on every box. Each part read every rank
  file's header first: rank r of 4, 1,609 tensors, 72.38 GiB, made from `/data/models/DeepSeek-V4.1-Flash`.
- **The server** has production's arguments: `run4.sh --mode serve --max-ctx 262144`, and serve's defaults
  otherwise (`--slots 1`, CED at 8192-row chunks, drafts with the FP8 drafter head, B′ on, a 12 GiB session
  store). The k = 1 parts add `--no-spec`. dsfront listens on 127.0.0.1 with `--hold-s 120`, `--model-name
  deepseek-v4.1-flash` and the official checkpoint's `tokenizer.json` (the variant's, byte for byte), plus
  `--dump-dir` for the edit bench.
- **Each session is one ring part** (`ring.sh` with `OWNER=bench-official PREEMPT=0`), with a fresh engine.
  Other agents' ring jobs ran between the parts. rank 0 ran `gpu-clock-cap`, as for every original.
- **The tools** are RC5's ([Where it stands on release candidate 5](#where-it-stands-on-release-candidate-5-2026-10-13)),
  except that bench.py's sessions run through `bench/bench_record.py`: bench.py unchanged, with every request
  it sends recorded (the body, every SSE line with its arrival time, bench.py's result). The kit's sessions
  ran through it too, from the same checkout.
- **The analysis** is RC5's (`rc5table.py` and `changes.py`, their paths changed), plus `bench_record_report.py`
  (each side's session statistics, the token accounting and the texts) and `steps.py` (tokens and time a step
  from each server's own log, and the kit's conditions).

**bench.py: the two servers on the official checkpoint.** Ours is six sessions, each a fresh engine: O1 at
10:54, 11:02 and 11:08, O2 at 11:39, 11:43 and 11:52, 117–119 s each on the ring. The kit's is three sessions,
k1–k3, after a warm-up session in the same start. A decode figure is the mean of the sessions' means, each the
mean of its two requests. In brackets: the sessions' range, then the spread of all the requests ((largest −
smallest) / mean). The ratio's range comes from the extreme requests.

| | dsgb10: O1 and O2 | SGLang kit: k1–k3 | dsgb10 / kit |
|---|---|---|---|
| decode tok/s, prose | **77.41** (76.99–77.59; 1.3 %) | 50.90 (50.83–50.99; 2.2 %) | 1.52 (1.49–1.55) |
| decode tok/s, code | **115.40** (115.24–115.54; 0.5 %) | 85.63 (85.34–86.03; 0.9 %) | 1.35 (1.34–1.36) |
| decode tok/s, agentic | **102.93** (102.60–103.23; 0.9 %) | 76.25 (76.07–76.56; 0.9 %) | 1.35 (1.34–1.36) |
| decode tok/s, JSON | **153.68** (153.43–154.06; 0.9 %) | 107.71 (107.44–107.97; 0.7 %) | 1.43 (1.42–1.44) |
| time to first token, short prompts, each session's first request apart | **median 143 ms, 124–156** (42 requests) | median 230 ms, 219–253 (21) | |
| each session's first request | 164–168 ms | 215–218 ms | |
| cold prefill, 945 tokens | **2,908 tok/s** (1.7 %) | 2,090 (1.1 %) | 1.39 |
| cold prefill, 3,747–3,748 tokens | **6,288** (1.3 %) | 4,103 (19 %: 3,651, 4,219, 4,440) | 1.53 |
| cold prefill, 16,418–16,419 tokens | **8,534** (1.3 %) | 4,298 (0.2 %) | 1.99 |
| cold prefill, 28,653–28,654 tokens | **8,898** (1.1 %) | 4,264 (0.8 %) | 2.09 |
| cold prefill, 57,541–57,542 tokens | **9,201** (0.7 %) | 4,274 (0.6 %) | 2.15 |
| agentic: first turn (15,774 tokens) | **1.805 s** (1.801–1.813) | 3.722 s (3.718–3.725) | 0.49 |
| agentic: turns 1–5, to first token | **median 0.395 s, 0.386–0.424** (30 turns) | median 0.730 s, 0.681–0.788 (15) | 0.54 (medians) |
| agentic: new tokens a turn | 1,675–2,035 | 1,705–2,209 | |

- **The prefill spreads** are over the sessions, one request each. The kit's 4k prefill moved by 19 % between its
  sessions (and ran at 2.2k tok/s in the warm-up); its other lengths moved by at most 1.1 %.
- **The agentic turns' sizes** follow each server's answers, so the new tokens a turn differ between the sides.
  Every turn on both sides reused the whole previous turn.
- **O2 / O1** is 0.997 (prose), 0.998 (code), 1.004 (agentic) and 1.001 (JSON), with the extreme requests' ratios
  0.988–1.009. The prefill and agentic figures of O2 are O1's within 0.6 %.
- **The kit on the variant**, its one session of 2026-10-11 ([baseline](../DESIGN.md#baseline)), decoded 46.8 /
  77.4 / 73.6 / 106.9 tok/s: today's run on the official checkpoint is 0.8–10.6 % faster, its first agentic turn
  the same (3.72 s against 3.73), its time to the first token lower (219–253 ms against 244–280). README's
  earlier ratios (1.40–1.62) set our sessions of 2026-10-13 on the variant against that run; this table's are
  of one checkpoint and one day.

**The kit's window.**

- **The kit** is [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)
  at `cad252b` (v2.1) as it served on this ring: the `dsv4.1` branch of SGLang in its `canary-roce` image, TP=4
  over its patched NCCL (2.30.7) on the switchless ring, DSpark drafts of 5, up to 16 running requests,
  4096-token prefill chunks. An overlay of its env file pointed it at the official checkpoint, which rank 0
  shared to the other three boxes over NFS, served under the name `deepseek-v4.1-flash`, on port 18888.
- **From its start to Ready** took 8.4 minutes (11:13:40–11:22:03), its smoke test and warm-up included. Then:
  a warm-up session of bench.py (k0, left out), three sessions (k1–k3, 11:23:41–11:28:23, 93–95 s each),
  `golden_sglang.py`'s four greedy completions (now `bench/golden_sglang_official.json`), and the GSM8K harness
  (220 s). The window held the ring for 19 minutes.
- **Its conditions**, from its own log over k1–k3 (`steps.py`, 281 s): bench.py's 57 requests and 9 health
  checks, all from 127.0.0.1; at most one request running on any log line, none queued. rank 0's GPU ran at
  2,184–2,190 MHz (its clock cap) and the other three at 2,398–2,411 MHz, as the window's checks between
  sessions read them.
- **k0** decoded 0.3–2.1 % slower than k1–k3, and prefilled 4k and 16k at 2.2k and 3.4k tok/s; its code,
  agentic and JSON texts were k1–k3's.
- **Afterwards** the kit's own state and log files had the hashes, sizes and times they had before, and its
  default model link still named the variant. Its stop left its NFS server container running; a second
  `docker rm -f`, four minutes later, removed it.

**Steps, token accounting and texts.**

| tokens a step; ms a step | dsgb10 (rank 0's log; the mean of 12 requests) | SGLang kit (its log; the median of its lines) |
|---|---|---|
| prose | 1.87; **24.2** | 1.96; 38.2 |
| code | 3.38; **29.3** | 3.41; 43.7 |
| agentic | 2.91; **28.3** | 3.19; 44.3 |
| JSON | 5.05; **32.9** | 5.09; 46.6 |

- **Our steps are 29–37 % shorter**, at 1–9 % fewer tokens a step on the two servers' different texts.
- **Token accounting.** Each of the kit's 42 texts (k1–k3) and our 84 re-tokenises with dsfront's tokenizer to
  the server's `completion_tokens`, less a stop's end id: the two servers count alike.
- **Greedy texts.**
  - Ours repeat across the six sessions, request for request.
  - The kit's code, agentic and JSON texts repeat across its four sessions, k0 included. Its prose texts
    agree for their first 478 of 512 tokens and then part: its eight (k0 included) end in four ways, parting
    at tokens 478, 489 and 500, and k1–k3's six in three.
  - The two servers' texts part after 9 tokens on prose, 36 on code, 69 on agentic and 49 on JSON, so
    the decode rows compare different texts after their first tens of tokens.
  - Each first content chunk carries one token on both sides. The kit then sends a draft step's tokens in
    one chunk (103–266 chunks for 512 tokens), ours one chunk a token.

**k = 1** (`--no-spec`, bench.py's decode mode, two sessions, 12:09–12:14). Without drafts a step makes one token
whatever the text, so k = 1 measures the step alone.

| | RC5 on the variant: k1a, k1b | official: k1a, k1b | spread | change |
|---|---|---|---|---|
| decode tok/s, all eight requests | 59.21, 58.90 (58.02–59.41) | **59.20, 59.06** (58.83–59.36) | 0.2 % | +0.1 % |

- **The texts are the drafted decode's.** Every k = 1 request's text is O1's and O2's on its prompt
  (`report_spec_k1.md`): on the official files too, drafts change the speed, never the tokens.

**sparkDash 1.8.8 against the MiaAI kit's published figures.** Ours is the mean of two sessions, sd1 and sd2
(12:14–12:30), each a fresh engine for decode and another for prefill, with RC5's protocol. c1 is the median of
five rounds (two sessions' ranges in brackets), the catalogs are the mean over their prompts, and a step's time
is rank 0's decode time over its verify steps. The MiaAI kit's columns are its documents' figures, as README
cites them, on deepseek-ai's official checkpoint on its authors' switched RoCE fabric; not re-measured here.

| | RC5 on the variant | official: sd1, sd2 | MiaAI kit, published |
|---|---|---|---|
| c1 prose | 102.2 | **95.0** (94.8–95.3), 94.9 (94.9–95.0) | 89.9 (v2.4) |
| c1 code | 148.1 | **151.0** (150.8–152.0), 151.0 (150.5–151.6) | 131.0 (v2.4) |
| c1 structured | 186.4 | **185.6** (185.3–186.8), 186.0 (184.9–186.3) | 158.7 (v2.4) |
| c1 JSON | 162.6 | **150.3** (149.7–150.9), 150.1 (149.5–150.7) | 128.4 (v2.4) |
| catalogs at c1: prose / structural / code | 78.5 / 125.0 / 148.6 | **78.2 / 122.4 / 148.2**, 78.1 / 122.2 / 148.0 | 61.1 / 99.0 (v2.2) |
| decode step, prose (c1 prompt) | 27.95 ms at 2.83 tokens | **27.86, 27.89 ms at 2.63** | 31.2–32.5 ms at 2.27 (v2.2, its own probe prompts) |
| decode step, code (c1 prompt) | 34.82 ms at 5.10 tokens | **34.80, 34.87 ms at 5.20** | 37.4–37.8 ms at 3.87 (v2.2, as above) |
| decode step, catalogs: prose / code | 24.88 ms at 1.93 / 33.90 at 4.97 | 24.92, 24.94 at 1.92 / 34.00, 34.03 at 4.97 | |
| cold prefill, real text, tok/s: ~4k / 15k / 32k / 62k / 120k | 6.3–7.1k / 8.6–9.0k / 9.1–9.2k / 9.2–9.3k / 9.4k | **6.3–7.1k / 8.7–9.1k / 9.2k / 9.2–9.3k / 9.4–9.5k** | 4.2–4.7k / 4.8–4.9k / 4.8k / 4.9–5.0k / 4.8–4.9k (v2.1) |
| sparkDash's filler bench, tok/s at 4k / 16k / 32k / 64k / 128k | 8.2k / 9.8k / 10.2k / 10.4k / 10.5k | 8.2k / 9.8k / 10.2k / 10.4k / 10.5k in both | 4.8k / 5.8k / 5.9k / 6.0k / 5.9k (v2.4) |

- **c1 against v2.4:** prose +5.7 %, code +15.2 %, structured +17.1 %, JSON +17.0 %. Against the best published
  per type (90.2 / 133.2 / 158.7 / 135.3; [survey](review/external.md#survey)): +5.3 to +17.1 %.
- **Against RC5 on the variant** the c1 rates change by −7.0 / +2.0 / −0.3 / −7.6 %, the catalogs by −0.5 / −2.1 /
  −0.3 %, at step times within 0.8 % of the variant's: the tokens a step move with the texts (prose 2.83 →
  2.63, JSON 5.43 → 5.00, code 5.10 → 5.20).
- **The texts:** sd2's c1 texts and its 66 catalog texts are sd1's.
- **Per token on prose** a step costs 10.6 ms on the c1 prompt and 13.0 ms over the catalog; the kit's probe gives
  13.7–14.3 ms.
- **The nearest published code step** is still spark-ds41f's 34.25 ms at 3.37 tokens (its own prompt, reasoning on;
  [survey](review/external.md#survey)): 1.7 % shorter than ours, carrying 35 % fewer tokens.
- **Real text** runs 8.7k–9.5k tok/s from 15k to 120k tokens, every prompt cold. The best published, spark-ds41f's,
  is 5.7k–5.9k.

**The edit bench at dsfront's sampling defaults**, r2's settings, against release candidate 5 on the variant (e1
and e2 of that morning) and r2:

| thinking off: twelve tasks, three times | r2 (the variant, 10-12) | RC5 on the variant: e1, e2 | official: e1, e2 | spread | change against RC5 |
|---|---|---|---|---|---|
| requests; output tokens | 113; 14,883 | 112; 14,571 and 113; 14,805 | 113; 15,284 and 112; 15,011 | | +3.1 % (tokens) |
| end to end | 143.7 s | 131.4, 134.2 s | **138.5, 134.2 s** | 3.1 % | +2.7 % |
| output tokens per second of it | 103.6 | 110.9, 110.4 | **110.4, 111.9** | 1.3 % | +0.5 % |
| decode tok/s, tokens a step | 128.7 at 5.08 | 139.9 at 5.09, 138.9 at 5.04 | **137.4 at 4.96, 138.5 at 5.03** | 0.8 % | −1.0 % |
| time to first token, median | 241 ms (client 246) | 230, 231 ms (client 235, 236) | **229, 230 ms** (client 231, 234) | 0.3 % | −0.6 % |
| prefill and decode time (rank 0) | 28.7 s and 114.7 s | 27.6 and 103.4 s; 27.9 and 105.8 s | 27.7 and 110.4 s; 26.2 and 107.6 s | | −2.9 % and +4.2 % |
| checks passed | 36 of 36 | 36 of 36, 36 of 36 | 36 of 36, 36 of 36 | | |

- **Thinking max** (four tasks once) is one run of 13–15 requests, so its figures move by several percent between
  runs. Official: 14 and 15 requests, 2,034 and 2,056 output tokens, 19.4 and 19.7 s (104.8 and 104.2 tokens a
  second), decode 124.2 tok/s at 4.30 and 4.37 tokens a step, 236 and 230 ms to the first token; 4 of 4 checks
  twice. RC5 on the variant: 21.9 and 17.3 s (105.9 and 107.7 tokens a second), decode 122.9 and 129.0.
- **The fairer measure is output per second.** The engine draws a seed per request, so every run's answers
  differ: the official runs wrote 3.1 % more tokens than RC5's on the variant and wrote them as fast (+0.5 %).

**Correctness on the official files.** The correctness review's harness ([correctness.md](review/correctness.md),
whose figures are the variant's) on this binary at production's arguments. The kit's greedy outputs are
`bench/golden_sglang_official.json`, made in the kit's window: `golden_sglang.py`'s four prompts, chat, thinking off,
96 new tokens or the end.

| check | reference | official, 2026-10-13 | variant (correctness review) |
|---|---|---|---|
| Q: GSM8K, the first 100 test problems, greedy, thinking off, zero-shot, ending with "#### <number>" | the dataset's answers | **99 of 100**; every answer on a "####" line, none cut; 18,165 completion tokens in 160 s. The miss (problem 93) reads a 10 % gain in speed as the time ÷ 1.1 (36.36 s), where the dataset takes the time × 0.9 (36 s) | 100 of 100 (18,612 tokens, 210 s) |
| the same harness on the kit, in its window | the dataset's answers | 100 of 100; 18,127 completion tokens in 220 s | |
| N1: passkey, classic filler, 32.8k / 131k / 200k tokens, the key at depths 0.1, 0.5, 0.9 | the key | **9 of 9** | 9 of 9 |
| N2: passkey, this repository's docs as filler, 33.7k / 139k / 220k tokens | the key | **9 of 9** | 9 of 9 |
| S1: the kit's four prompts free-running through rank 0's socket (CED, FP8 drafts), twice | the kit's greedy tokens | math the kit's 2 of 2; code its first 36 tokens, where the kit's top two tie (margin 0.0); prose its first 8, where ours is a near tie (margin 0.016, the kit's 0.75); fact its first 7 (the kit's margin 1.0, ours 0.87). Both runs identical | math and code identical; prose 12; fact 5 |
| S2: the same, teacher-forced along the kit's tokens (top 20) | the kit's | argmax = the kit's token at **260 of 271** steps: math 2/2; prose 89/96 (\|Δ log p\| over the kit's top 5: mean 0.59, median 0.41, max 3.73); code 95/96 (0.36; the miss at the tie); fact 74/77 (0.66) | 256 of 267: 2/2, 89/96, 96/96, 69/73 |
| A1: the same prompts through dsfront, each twice | the kit's; S1 | `prompt_tokens` 18, 39, 36, 14: the kit's; the first answers are S1's texts; `completion_tokens` 2, 96, 96 and 75 (the kit's 77, for its other text). The second answers repeat math and code and differ for prose and fact, as on the variant (m7) | the same against SGLang's outputs on the variant (fact 61 tokens against its 73) |

- **On the side, the passkey prompts** give the cold prefill through the API, each request's few decoded tokens
  included: 9.6k–10.1k tok/s on the classic filler from 32.8k to 200k tokens and 8.8k–9.4k on the docs from 33.7k
  to 220k, as on the variant (9.6k–10.1k and 8.9k–9.5k).

**Where it differs from release candidate 5's campaign.**

- **The binaries** are candbin6 (530af7f), not candbin5: the same engine and front-end code, other build paths.
  Part 0's hashes are RC5's on the variant and, on the official files, those of [The official checkpoint's
  weights](#the-official-checkpoints-weights-2026-10-13).
- **bench_record.py** ran bench.py's sessions and the k = 1 parts; bench.py itself is unchanged.
- **The model name.** dsfront serves `deepseek-v4.1-flash` with the official `tokenizer.json`, and bench.py and
  sparkDash ask for it. The edit bench sends its spec's name (`deepseek-v4.1-flash-uncensored`), which dsfront does
  not check.
- **Other agents' jobs** (fleet, refhash) held the ring between our parts; RC5's campaign had the ring alone.
- **The kit's window** sat between O1 and O2. Its stop left the NFS server container running, and the driver,
  which stops at a failed check, stopped; the rest ran from O2 on, from a second driver five minutes later.
- **The correctness checks** ran in two ring parts (c1: the socket's S1 and S2, A1 and GSM8K; c2: the passkeys),
  each under ten minutes.
- **The disk tier** was 8 GiB in ~/dsgb10/ospill (tag `bof`), as RC5's 8 GiB in ~/dsgb10/bspill; production has 32
  GiB. No request reached the tier: the store held at most 3.1 GiB of its 12 (c2's 220k-token
  passkey prompts; at most 1.8 GiB in the other parts, at sparkDash's 131k filler prompt), with no spill and no
  eviction in any part.

**The raw data** are in rank 0's /data/dsgb10/bench:

- wave2/o1/s1–s3 and wave2/o2/s1–s3: bench.py's log and json, its records (bench.jsonl), rank 0's and dsfront's
  logs;
- wave2/kit: the kit's records and logs (k0–k3), its own log of the window (logs/dsv41.log), the greedy outputs, the
  GSM8K run, its start, stop and share logs, and before/ (its files before the window);
- wave2/logs: each window part's log (K.log's own lines are in UTC, an hour behind the local times given
  here); K.check.log has the kit's files after the window;
- wave2/tools: the correctness harness;
- official/h0: part 0's hash runs; official/c1 and c2: the correctness checks; official/k1a and k1b; official/sd1
  and sd2 (as RC5's); official/e1 and e2;
- official/logs: each part's log and the two drivers' (drive.log, drive2.log);
- official/report_kit_ours.md, report_o1_o2.md, report_k0_kit.md, report_spec_k1.md, steps_kit.txt,
  steps_kit_k0.txt, steps_ours.txt, kit_texts.txt, offtable.txt and offchanges.md: the analysis;
- official/scripts: `opart.sh` (one ring part), `odrive.sh` (the driver), `kitpart.sh` (the kit's window),
  `stats.sh`, `offtable.py` and `offchanges.py` (the analysis), `kittexts.py` (where the kit's texts part).

### The end of a turn and the host's share of a step (2026-10-12)

**Verdict: nothing to build.** The host leaves the GPU no idle worth taking. The end of a turn is
GPU work that the session's state needs. Sending DONE before that work is the one fix that keeps the
bits, and on the edit bench it is worth about 2 ms a turn (0.2 %), far under the 2 % bar. Through pi
itself it is worth 9–11 ms a turn (0.6–0.7 %; "pi's turnaround", below), still under it.

**The question.** In agentic use the end of every turn gates the client's next request. Two things
were measured:

- how long the engine's end of a turn is, from the last token to the end of the stream;
- how much of a decode step the host leaves the GPU idle.

Either would be worth a fix only if it kept outputs and session state bit-identical, the cancel path
clean, and long turns no slower. The bar was ≥ 2 % on the edit bench's per-turn end-to-end time.

**The method.**

- **Timers** (aa27273, branch runner-edit, unmerged):
  - rank 0 logs, per request, the time from the last token's step to DONE, and the GPU time of the
    step after it;
  - dsfront logs the time from its stop to the engine's DONE, and to its `[DONE]`.
- **Edit bench:** greedy, thinking off, all 12 tasks twice (76 turns), on a dev engine at serve's
  defaults with those builds.
- **nsys over the decode loop** (graph nodes traced): k = 1 and drafts, 1k context, 256 steps,
  sampled at T 1 / top-p 0.95, seed 7.
- **The same two runs without nsys**, with the same tokens (hash 9cc97f5db64afd70), for their own
  step times.

**The end of a turn.**

| kind | turns | end to end per turn, mean | last token to DONE, median (p10–p90) | the step after it, GPU | stop to `[DONE]` at dsfront | tail / end to end |
|---|---|---|---|---|---|---|
| edit | 20 | 1924 ms | 31.3 ms (20.4–40.2) | 31.3 ms | 31.5 ms | 1.6 % |
| write | 4 | 4936 ms | 20.4 ms (20.4–20.7) | 20.4 ms | 20.6 ms | 0.4 % |
| read | 28 | 686 ms | 32.5 ms (29.2–40.4) | 32.5 ms | 32.8 ms | 4.9 % |
| prose | 24 | 551 ms | 24.9 ms (20.3–34.0) | 24.9 ms | 25.2 ms | 4.8 % |
| all | 76 | 1193 ms | 29.8 ms (20.4–39.2) | 29.8 ms | 30.2 ms | 2.5 % |

- **The tail is one step's GPU time.** From the last token to DONE takes exactly the GPU time of the
  step queued behind the last token (median difference 0.00 ms, max 0.10).
  - The loop runs one step ahead, and at a stop it waits for that step.
  - That step writes the stop token's KV on the decode path and the rejected drafts' rows. The rows
    raise `DecodeEnd::written`, and the next request's restart point depends on it.
  - dsfront adds 0.2 ms.
- **The bench's client is quick.** It sends its next request 2.0 ms (median; p10 1.0, p90 2.2) after
  a turn's end (the dump files' times).

**Fixes considered.**

- **Skip the trailing step: rejected.** The stop token's KV would then come from the next request's
  prefill, which rounds differently from the decode path. The written rows would change, and with
  them the next request's restart point and state. Not bit-identical.
- **Shrink it (fewer rows, no draft pass): rejected.** It changes the written rows and the drafter's
  windows, so the same objection applies.
- **Send DONE before waiting for the step: bit-safe, but small.**
  - The step still runs. The next request on the session starts after it, as today, and cancel is
    unaffected.
  - It saves min(tail, the client's turnaround) per turn. On the edit bench that is about 2 ms, or
    0.2 % of a turn.
  - A client that takes 30 ms or more between turns would save up to the whole tail: 2.5 % of an
    average turn, and 4.8–4.9 % of read and prose turns.
  - pi's turnaround was measured next ("pi's turnaround", below): 9–10 ms, never the whole tail.

**Per-step idle** (`idleprof.py` on the nsys capture: steps split at `k_step_in` / `k_spec_in`, the
first 3 dropped):

| | steps | span per step | GPU idle between steps | idle inside a step | Engram waits beyond their cost | steps launched after the previous one ended | host's lead |
|---|---|---|---|---|---|---|---|
| k = 1 | 253 | 17.95 ms | 11.3 µs | 14.5 µs | 0.9 µs | 0 | 17.35 ms |
| drafts, 1–6 rows | 95 | 23.2–41.3 ms | 4.1 µs | 1931 µs (nsys's, below) | 0.3 µs | 0 | 33.4 ms |

- **The host never makes the GPU wait.** It is a whole step ahead, and no step was launched after
  the one before it had ended.
  - The idle it causes is the Engram waits' excess: 0.9 µs a step at k = 1 (0.005 %), 0.3 µs with
    drafts.
  - The 26 µs of gaps between and inside k = 1 steps (0.14 %) are the graph's own kernel boundaries.
- **The 1.93 ms inside each draft step is nsys's own.** It sits between `k_spec_in` and the first
  kernel of the conditional body.
  - Without nsys the same steps take 1.8–2.6 ms less. The engine's own timers give GPU ms by rows
    1–6 of 21.38 / 25.19 / 28.72 / 32.01 / 35.33 / 38.67, against 23.23 / 27.25 / 30.97 / 34.22 /
    37.91 / 40.99 under nsys.
  - k = 1, with no conditional node, differs by only 0.16 ms (17.91 against 18.08 ms a token).

The timers stay on runner-edit, unmerged: aa27273, with their report `tools/turnend.py` in c2a3666.
They add a field to `DecodeEnd` and one line to the "turn:" log, and are cheap to merge if the tail
needs watching again.
[`engine/server/tools/idleprof.py`](../engine/server/tools/idleprof.py) gives the per-step table
from any capture of the decode loop. The runs are in rank 0:~/dsgb10/editruns/t1 (the bench), n1
(nsys) and p1 (without nsys).

#### pi's turnaround (2026-10-12)

**Verdict: DONE-early is not built.** pi sends its next request 9–10 ms (median) after `[DONE]`.
That never covered the tail behind the last token: 0 of 76 turnarounds did, against a tail of
31–33 ms (median). Sending DONE before that step would hide 9–11 ms a turn, 0.6–0.7 % of a turn
through pi, under the 2 % bar.

**The question.** DONE-early saves min(tail, the client's turnaround) each turn. The bench's client
turns around in 2 ms, but pi runs its tools in between. If pi's turnaround usually covered the
~30 ms tail, DONE-early would save ~2.5 % a turn and would be built. The test is the time of a turn
through pi itself.

**The method.** One ring part, [`bench/piturn/part.sh`](../bench/piturn/part.sh):

- **The engine and dsfront:** the timer builds (aa27273; sha256 18ef2194… for the engine,
  d2eef9ff… for dsfront) as a dev engine at serve's defaults. dsfront runs with `--dump-dir`, and its
  stderr goes through `stamp.py`, which stamps each line as it arrives (to 0.1 ms).
- **pi 0.84.3 itself,** driven by `pidrive.py`:
  - the edit bench's 12 tasks, one pi process each, with read/edit/write only, thinking off and pi's
    own sampling;
  - isolated: its own `PI_CODING_AGENT_DIR` holding only `models.json` (the key comes from the
    environment), `--no-session`, `--offline`, and no context files, extensions, skills, templates
    or themes;
  - three modes: print (`-p`), the TUI on a pseudo-terminal (120×40), and JSON (`--mode json`, each
    event stamped as it arrives).
- **The gap** runs from dsfront's "end" line, logged once `[DONE]` is sent, to the next request's
  "plan" line, logged once dsfront has parsed, tokenized and submitted it. It is pi's turnaround
  plus dsfront's intake.
- **The tail** is the timers' "last token to DONE" for the turn before the gap.
- **A turn through pi** runs from one request's intake to the next one's.

**The numbers.** All 36 pi runs passed their checks: 112 requests, 76 turnarounds.

| mode | turnarounds | gap, median (p10–p90) | gap, max | tail, median | gap ≥ tail | hidden by DONE-early, mean | a turn through pi, mean | hidden / turn |
|---|---|---|---|---|---|---|---|---|
| print | 26 | 9.0 ms (6.4–10.9) | 14.6 ms | 32.2 ms | 0 | 8.8 ms | 1555 ms | 0.57 % |
| TUI | 25 | 10.4 ms (7.6–12.9) | 23.4 ms | 30.7 ms | 0 | 10.7 ms | 1518 ms | 0.70 % |
| JSON | 25 | 9.5 ms (7.3–12.6) | 14.3 ms | 32.6 ms | 0 | 9.4 ms | 1528 ms | 0.61 % |

- **By the tool call before the gap** (the range of the three modes' medians):
  - after a read, 8.7–10.1 ms. These turns are short (658–796 ms), so that is 1.1–1.4 % of them;
  - after an edit, 9.2–11.0 ms, or 0.5–0.7 % of a turn;
  - after a write, 6.5–8.4 ms, or 0.1–0.2 % of a turn.
- **The largest gap** was 23.4 ms, after an edit in the TUI, with a 13.5k-token prompt. Its tail was
  longer still.

**Where pi's time goes** (JSON mode, from its events' arrival; medians and p10–p90, in ms):

| `[DONE]` sent → pi's `message_end` | → first tool starts | tools run | last tool ends → next request submitted | gap |
|---|---|---|---|---|
| 1.3 (0.7–1.8) | 0.1 (0.0–0.2) | 5.1 (3.2–7.7) | 2.5 (2.1–3.3) | 9.5 (7.3–12.6) |

- Half the gap is pi running its tools.
- The fourth column is pi building and sending the request, plus dsfront's intake (parse, template,
  tokenize, plan, submit). dsfront doesn't time its intake on its own.
- On dsfront's mock engine (`mock.sh`), the same harness measures 6–15 ms (6 turnarounds). So the gap
  is pi's and dsfront's, not the engine's.

**What it means.**

- DONE-early would save the gap, not the tail: 9–11 ms a turn through pi. The tail itself is about
  2 % of a turn through pi (31–33 ms of ~1.5 s). Only a client that took as long as the tail between
  turns would save all of it.
- If DONE-early is ever wanted, it stays bit-safe this way:
  - a stop hook in the decode and draft loops sends DONE before waiting on the queued step;
  - `finish()` skips its own DONE. It still keeps the session, holds the store to its budget and
    releases the session's held requests after the wait, as today;
  - so the next request on the session starts after the step, and kept sessions and spills see only
    finished state.

The scripts are in [`bench/piturn/`](../bench/piturn/):

- `part.sh` is the part's script with the repo's paths;
- `pidrive.py` and `piturn.py`, with `stamp.py`, produced the tables above;
- `mock.sh` runs the same harness on dsfront's mock engine.

The run is in rank 0:~/dsgb10/editruns/pi1.

### Soak (2026-10-12)

[soak.py](../engine/server/tests/soak.py) replays agentic sessions through the front-end. It records
what each client sees, rank 0's log line for each request and every rank's memory. It ran for an
hour against [service.sh](#as-a-service-servicesh) at its defaults: CED, drafts with the FP8 drafter
head, `--max-ctx 262144`, and a session store of 12 GiB per rank, 11.25 GiB of it kept between
requests. The binary was master a3aff57, and the run came right after the service's failure tests.
Three clients sent requests back to back. The engine runs one request at a time, so a client's time
to first token includes its wait behind the other two.

```
engine/server/service.sh start
python3 engine/server/tests/soak.py --minutes 60 --out soak1    # later: --report soak1
engine/server/service.sh stop
```

- **Sessions.** Six sessions grow to 12–60k tokens each, and three long ones to ~150k (a turn every
  90 s at most). Each turn appends the last answer, then tool outputs cut from this repository's
  files. Half of these are 200–2k tokens (log-uniform), half 2–15k (10–15k for the long sessions).
  The requests offer the golden transcript's four tools. Answers are 64–256 tokens: 30 % greedy, the
  rest at temperature 0.7 and top-p 0.95, and 15 % think.
- **Kinds of request.**
  - turn (55 %); json, not streamed (5 %).
  - regenerate (6 %): the last turn twice more, greedy.
  - fork (6 %): a session's history up to an earlier tool result, then a different result.
  - cancel (6 %): the client closes after 1–30 events.
  - disconnect (5 %): the client resets the connection during the prefill of a tool output of at
    least 6k tokens. The session's next turn sends the output again.
  - reset (3 %): a reset mid-stream.
  - compact (4 %): the history replaced by a 2–6k-token summary, the system prompt kept.
  - revisit (5 %): a turn on a session put aside long ago.
  - The golden agentic transcript: 13 turns of 0.7–28.3k tokens, greedy with thinking, at minute 2
    and then every 10 minutes.
- **Recorded.** Per request: what the client saw (status, events, finish, usage, time to first
  token) and rank 0's line. Every 30 s: each rank's GPU memory (nvidia-smi, per process), its
  resident set and the box's MemAvailable.
- **Pass.** Every response well formed, no failed request and no error in the logs. GPU memory and
  resident sets flat, the Engram row cache aside (below). Decode and prefill speeds of the last ten
  minutes within 10 % of the first ten. The store evicting. Then a clean stop.

It passed. There were 2,183 requests: turn 1,137, regenerate 266, cancel 128, fork 128, turns of the
long sessions 111, revisit 90, json 88, disconnect 85, golden 78, reset 58 and compact 14. The engine
reused 39.0M of the prompts' 51.8M positions, prefilled 12.9M and decoded 191k tokens. Medians per 10
minutes, decode over answers of at least 32 tokens at contexts under 32k:

| minutes | requests | decode tok/s | ms per step | drafts accepted per step | prefill ms, turns adding < 1k | prefill tok/s, turns adding ≥ 4k | time to first token (front-end), turns adding < 1k |
|---|---|---|---|---|---|---|---|
| 0–10 | 405 | 111.7 | 37.5 | 3.13 | 252 | 8,005 | 3.24 s |
| 10–20 | 362 | 108.2 | 37.1 | 3.00 | 242 | 8,212 | 3.22 s |
| 20–30 | 347 | 111.4 | 37.2 | 3.13 | 242 | 8,273 | 3.19 s |
| 30–40 | 335 | 109.7 | 37.6 | 3.13 | 257 | 8,274 | 3.42 s |
| 40–50 | 372 | 107.5 | 36.9 | 2.88 | 251 | 8,183 | 2.99 s |
| 50–60 | 362 | 104.3 | 37.0 | 2.88 | 248 | 8,147 | 3.59 s |

Memory, with the store's largest occupancy and the evictions in each window, each rank's largest
resident set and the median MemAvailable of its box. GPU memory was 93,352 MiB on every rank in every
sample (everything is allocated at startup).

| minutes | store GiB | evictions | resident set MiB, ranks 0–3 | MemAvailable GiB, ranks 0–3 |
|---|---|---|---|---|
| start | 0 | | 758 / 771 / 771 / 772 | 21.4 / 23.2 / 23.1 / 23.2 |
| 0–10 | 9.8 | 0 | 1,226 / 1,553 / 1,847 / 2,091 | 21.1 / 22.8 / 22.5 / 22.4 |
| 10–20 | 11.6 | 77 | 1,337 / 1,781 / 2,187 / 2,533 | 20.8 / 22.4 / 21.9 / 21.7 |
| 20–30 | 11.6 | 92 | 1,386 / 1,893 / 2,357 / 2,759 | 20.7 / 22.2 / 21.7 / 21.3 |
| 30–40 | 11.4 | 91 | 1,427 / 1,985 / 2,499 / 2,945 | 20.6 / 22.1 / 21.5 / 21.1 |
| 40–50 | 11.4 | 86 | 1,448 / 2,031 / 2,569 / 3,037 | 20.6 / 22.1 / 21.4 / 21.0 |
| 50–60 | 11.3 | 84 | 1,466 / 2,073 / 2,635 / 3,129 | 20.6 / 22.0 / 21.3 / 20.9 |

- **Errors: none.** Every response was well formed: status, JSON events, one finish reason, `[DONE]`
  and usage. All 1,405 tool calls had JSON arguments and named an offered tool. Neither log has an
  error line, and the front-end stopped no repetition loop.
- **Latency: flat.**
  - The step time held at 36.9–37.6 ms.
  - Decode fell from 111.7 to 104.3 tok/s in the last 20 minutes because fewer drafts were accepted
    (3.13 → 2.88 per step), in every context band. The answers changed, not the engine.
  - The golden replays, the same prompts every 10 minutes, decoded at 106.5, 107.1, 107.0, 106.9,
    107.1 and 107.0 tok/s. Their prefill varied with what the store still held of them (reused
    158k, 25k, 0, 49k, 0 and 37k of 186k positions).
  - Prefill held: 242–257 ms for turns adding under 1k positions, 8.0–8.3k tok/s for larger ones.
  - The time to first token of small turns, 3.0–3.6 s at the front-end, is mostly the wait behind
    the other clients' requests.
- **Memory: flat, apart from the Engram row cache.** The resident sets grew, and all of the growth
  is the Engram row cache. This is a 4 GiB anonymous mapping that fills as rows are first read; the
  store's sizing reserves its full size. Split from smaps:

  | minute | 12 | 29 | 58 |
  |---|---|---|---|
  | Engram row cache, MiB, ranks 0–3 | 472 / 844 / 1,164 / 1,432 | 594 / 1,118 / 1,580 / 1,978 | 672 / 1,300 / 1,860 / 2,350 |
  | the rest of the resident set | 787 / 769 / 769 / 769 | 790 / 769 / 769 / 769 | 792 / 769 / 769 / 769 |

  MemAvailable fell with the cache on the workers. On rank 0, which other agents share, it dipped in
  two single samples (19.5 GiB at minute 17, 18.8 at minute 32) and was back 30 s later; rank 0's
  resident set did not move with it. The front-end's resident set was 231 MiB at minute 34, 248 at
  minute 39 and 270 at minute 58. Its session table is bounded (256 sessions and 16M tokens, about 16
  bytes per token), so it should level off below about 0.5 GiB. The hours-long soak should show
  where.
- **Store.** It filled in the first 10 minutes, then held 11.3–11.6 GiB while evicting 432 sessions,
  of 470 in all. The longest prompt was 158,341 tokens. 33 prompts were over 100k, and their answers
  decoded at 102 tok/s (median).
- **Cancels, resets and disconnects** (118, 52 and 85 aborted requests):
  - Rank 0 ends a request at its next decode step or prefill chunk (8,192 positions) after the
    front-end's `CANCEL`. It served the next request every time.
  - A disconnect during a prefill lets the chunk finish, and the session keeps what it computed. The
    session's next turn, which sent the same tool output again, reused all but under 1k positions in
    63 of 81 cases (a median of one new position).
- **Forks** (128): 113 resumed from a window snapshot of the session they branched from, 14 from its
  end. They reused 70 % of the prompt (median).
- **Compaction turns** (14): each new session resumed the old one's system prompt from a snapshot
  (36 % of its prompt, median) and prefilled the summary.
- **Revisits** (90): 41 found their session evicted and started cold, 28 resumed from a snapshot, 21
  from the end.
- **Regenerates** (133 pairs):
  - When both requests resumed at the same position, the greedy answers were identical: 98 of 98
    without tool calls.
  - When they resumed at different positions (as when the first resumed from a snapshot and the
    second from the end of the first's prompt), 6 of 12 differed. That follows from CED. The
    decoder replays the last 128 positions of each prefill from fresh rings ([CED](ced.md)), so the
    state an answer starts from depends on where its prefill started.
  - The 23 pairs with tool calls could not be compared (below).
- **Stop.** `service.sh stop` took 4 s. Rank 0 logged `serve: stopped` and exited 0, and the workers
  logged `done`. No rank was left on any box, the pid files were gone, and the kernels' next job
  took the ring the same second.

The run's own verdict was FAIL, on two memory rules that this run showed to be wrong. With the
corrected rules, `--report` passes it. Changed in soak.py during and after the run (the engine and
the service needed no fix):
- The resident set's growth was judged against 256 MiB, and the Engram cache's fill exceeded that.
  The cache is now sampled apart (from smaps), and only the rest must stay flat.
- MemAvailable belongs to the box, and rank 0 is shared. Its fall was judged at its lowest sample, a
  single dip on rank 0. It is now judged from the first window's median to the last's.
- A regenerate with tool calls compared the calls' ids, which the front-end draws at random for each
  response, so such pairs always differed. The comparison now takes the text and the calls' names and
  arguments, and a call without an id counts as malformed.
- The front-end's resident set is sampled (by the service's pid file).
- A compaction needed a session of 40k tokens, which few reached (14 in the hour). It needs 24k now.

### Drafts (DSpark), measured

On the build with the decode-path fusion (ec22622, [Step times](#step-times)). The fusion moved the
arithmetic, so the hashes and the drafts accepted are not those of the runs before it.

**The drafter against engine/dspark's reference** (`--mode spectf`, prose: 39-token prompt, 96 steps
teacher-forced). Leading drafts equal to the golden tokens, 0..5 per step:

| | 0 | 1 | 2 | 3 | 4 | 5 | accepted per step |
|---|---|---|---|---|---|---|---|
| reference (model.py's hiddens) | 47 | 31 | 12 | 3 | 2 | 1 | 0.80 |
| engine, prompt through the prefill path (`--big`) | 49 | 31 | 12 | 2 | 1 | 1 | 0.73 |
| engine, prompt in 8-row steps | 49 | 32 | 11 | 3 | 1 | 0 | 0.70 |

- d1 equals the reference's d1 in 87 and 84 of the 96 steps (85 and 87 before the fusion).
- Every difference is at a step where the drafter is unsure: survival[0] below 0.48 in both. The
  engine's main hidden differs from model.py's by bf16 numerics, and the drafter amplifies that where
  it is unsure. engine/dspark's own free pass, from the reference's hiddens, already differs in 3.
- The two prompt paths differ from each other in the same way: d1 equal in 85 of 96.
- The confidence head stays calibrated (the prefill path):
  - d1: mean survival 0.51, accepted 0.50;
  - d2 given d1: 0.38 and 0.35 (46 steps);
  - d3 given d1 and d2: 0.36 and 0.27 (15 steps).

Through a CED prefill (`--big --ced`, the long golden: a 999-position prompt, then 256 steps), where
the windows start from the replay's 128 rows only, the drafter accepts 0.566 drafts per step against
0.605 through the exact prefill path (0.28 against 0.22 in the first 32 steps, while the windows still
hold mostly seeded rows). d1 agrees between the two in 10 of the first 32 steps and 27 of the last 32:
CED moves the main model's early positions, and the windows follow it. On this text the numerics
alone move the rate about as much: before the fusion the two runs gave 0.633 and 0.648, and the exact
prompt in 8-row steps 0.551.

**Same tokens.** Decode after the long golden's first 1,024 or 16,384 positions (prefill path), 256
tokens, without and with `--spec`. The hash covers every token and its winning logit's bits, and each
pair of runs gives one hash:

| context, sampling | tokens hash | tok/s, k = 1 | tok/s, `--spec` | tokens per step |
|---|---|---|---|---|
| 1,024, greedy | febc7dc54ee40a39 | 53.8 | 107.5 | 3.54 |
| 1,024, T 0.7, seed 5 | 3634863d9905862b | 53.9 | 94.1 | 2.90 |
| 1,024, T 0.7, top-p 0.9, seed 5 | 1addb1547d6cc71e | 53.8 | 78.5 | 2.32 |
| 16,384, greedy | bbe79f6ad5f1be26 | 53.9 | 90.9 | 2.93 |
| 16,384 through CED (`--ced`), greedy | 81f772ab79a950e4 | 53.5 | 92.5 | 3.07 |

- Sampled, fewer drafts are accepted (1.90 and 1.32 per step, against 2.54 greedy): a draft is the
  drafter's argmax, and the token it must equal is a sample.
- serve_test.py prints the same with `--spec` as without, timings aside: against `--exact` every
  continuation, fork, snapshot, stored-session, regenerate and replay check (9 of 9), and against the
  default every check of the CED turn rule (17 of 17).
- A request restarted from a window snapshot, a stored session's end or a fork's base verifies and
  accepts the same drafts, step for step, as its cold twin: the drafter's windows travel with the
  session state exactly.
- `--mode tf` (4,096 prefilled, 512 teacher-forced) gives the reference hash, 842278b28d87dad0, with
  `--spec` as without: seeding the drafter's windows in the prefill leaves the main model alone.

**Step cost by rows.** GPU ms of a spec step, including the draft pass; the mean over each row
count's steps in the runs above:

| rows | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| context 1,024 (three runs) | 21.95 | 25.85 | 29.51 | 32.62 | 37.01 | 39.84 |
| context 16,384 (two runs) | 22.05 | 25.90 | 29.98 | 33.81 | 37.37 | 40.74 |
| context 1,024, before the fusion | 22.35 | 26.03 | 30.78 | 33.30 | 36.20 | 39.92 |
| context 1,024, FP8 drafter head, the row count forced (2026-10-12) | 21.10 | 25.38 | 29.26 | 32.76 | 35.61 | 38.70 |

- A plain one-row step is 18.8 ms at 1k ([Step times](#step-times)), so the draft pass (accept,
  draft, spec_next) costs 3.2 ms.
- Each further row costs 3.6 ms, mostly the MoE's extra experts.
- The default costs, 19.4, 3.6 and 3.3 ms, were measured before the fusion: a one-row step is now
  22.0 ms against their 22.7, and a further row the same 3.6 (3.7 at 16k). The gate depends only on
  their ratios: λ·ΔC is tokens over modelled time, times ΔC, so a common factor cancels.

**The costs re-measured (2026-10-12)**, after the draft-pass cut, on master a3aff57. The last row
of the table forces each row count (`--spec-thresh 2`: no draft verified; `--spec-thresh 0
--spec-drafts k`: k drafts every step), one run each. The same runs give a plain step of 18.55 ms
per token (`decode` without `--spec`, hash febc7dc54ee40a39 in all seven).
- C1 + C_draft is 21.10 ms: 18.55 for the step and 2.55 for the draft pass, as specprof measures it.
- A further row costs 3.67 ms on average (least squares through the one-row step), but the cost
  falls with the row count: 4.28, 3.88, 3.50, 2.85 and 3.09 ms for the 2nd to 6th row. Rows that
  route to the same experts stream them once.
- The defaults set from this, 18.6, 3.7 and 2.5 ms, raise the gate by 6–9 % (prose 0.229 → 0.249,
  code 0.324 → 0.343, agentic 0.318 → 0.343, JSON 0.415 → 0.438). bench.py decode, tok/s, two
  server sessions each (each the mean of bench's two runs), interleaved:

| | prose | code | agentic | JSON |
|---|---|---|---|---|
| 19.4, 3.6, 3.3 (session 1, session 2) | 69.5, 69.8 | 102.1, 102.4 | 99.9, 100.2 | 135.9, 136.8 |
| 18.6, 3.7, 2.5 | 69.4, 69.5 | 100.5, 100.6 | 99.8, 100.0 | 135.4, 135.4 |
| change | −0.3 % | −1.6 % | −0.2 % | −0.7 % |

So the defaults stay at 19.4, 3.6 and 3.3: the gate does better with a lower threshold than the
measured costs give. Why was not measured. The row costs suggest one reason: they fall with the row
count, so a single ΔC overstates what the later drafts cost. The old costs' larger one-row step (22.7
ms against 21.1) lowers λ, and with it the threshold. A cost per row count models this directly
(below).

**Gates by a cost per row count (2026-10-12).** `--spec-gate` (`ops.h` `kGate*`, `k_spec_next`), with
T(n) the ms of a step verifying n drafts, its draft pass included (`--spec-row-ms`, default the
measured 21.10, 25.38, 29.26, 32.76, 35.61 and 38.70 above):
- `ratio` verifies the prefix of n drafts with the most expected tokens per ms,
  (1 + survival[0] + … + survival[n − 1]) / T(n).
- `marginal` verifies d_j while survival[j − 1] ≥ λ·(T(j) − T(j − 1)), with λ the tokens per ms so
  far by T. A draft is verified when it yields more tokens per ms than the loop averages. `ratio`
  maximises each step's own rate instead, which verifies more on a weak step and less on a strong one.

Tokens are identical under each gate: 1k febc7dc54ee40a39, 16k CED 81f772ab79a950e4. On the golden
decode (256 tokens) every gate is within a step or two of the others:

| gate | 1k: tokens per step | tok/s | 16k CED: tokens per step | tok/s |
|---|---|---|---|---|
| thresh | 3.54 | 110.6 | 3.07 | 94.3 (an earlier run, same code) |
| ratio | 3.59 | 111.4 | 3.04 | 93.7 |
| marginal | 3.49 | 110.0 | 3.04 | 94.2 |

bench.py decode, tok/s, two server sessions per gate (each the mean of bench's two runs), started in
turn thresh, ratio, marginal, thresh, ratio, marginal:

| gate | prose | code | agentic | JSON |
|---|---|---|---|---|
| thresh (session 1, session 2) | 69.5, 69.4 | 102.1, 101.9 | 99.9, 99.7 | 136.3, 136.0 |
| ratio | 70.0, 70.0 | 102.7, 102.5 | 100.2, 99.9 | 135.4, 135.6 |
| marginal | 69.2, 69.4 | 100.5, 100.5 | 99.2, 99.1 | 137.0, 137.0 |
| ratio against thresh | +0.9 % | +0.7 % | +0.3 % | −0.4 % |
| marginal against thresh | −0.2 % | −1.4 % | −0.6 % | +0.7 % |

Drafts verified and accepted per step (a session's two runs are identical):

| gate | prose | code | agentic | JSON |
|---|---|---|---|---|
| thresh | 1.43, 0.81 | 3.03, 2.19 | 2.91, 2.11 | 4.43, 3.90 |
| ratio | 1.44, 0.83 | 3.22, 2.27 | 3.11, 2.19 | 4.63, 3.95 |
| marginal | 1.31, 0.78 | 3.11, 2.17 | 2.92, 2.09 | 4.38, 3.90 |

The step's GPU time by rows was the same under each gate (±0.1 ms): 21.1, 24.5, 28.0, 31.2, 34.5
and 37.3 ms for 1–6 rows. From 2 rows on that is 0.9–1.6 ms below T: on these short prompts a
further row costs 3.4, 3.5, 3.2, 3.3 and 2.9 ms, against the golden run's 4.28, 3.88, 3.50, 2.85
and 3.09. The cost per row count depends on the content, as the experts the rows share do.

So the thresh gate stays the default, and the other two stay behind the flag (for A/B runs, and for
a row budget shared across sessions, [batching.md](batching.md)):
- `ratio` gains 0.3–0.9 % on prose, code and agentic, and loses 0.4 % on JSON, where it verifies 0.2
  more drafts per step for 0.05 more accepted. Over the four it is +0.2 %. The thresh gate's own
  four-type mean ranged over 101.7–102.3 tok/s in four sessions that day.
- `marginal` loses 0.3 % over the four. Its first draft is held to λ·4.28 ms, against the thresh
  gate's λ·3.6 with a lower λ: 50 of its 287 prose steps verify no draft, against 29 of 282.

#### The draft pass

nsys over the 1k greedy run's spec loop (`decode --spec --profile`, then
[specprof.py](../engine/server/tools/specprof.py)): the pass is everything after the verify's sample,
the median over each row count's steps.

| rows verified (steps) | 2 (16) | 3 (7) | 6 (30) | kernels |
|---|---|---|---|---|
| the drafter's mHC projections and router rows sliced by rank, gathered per sublayer | 3.34 ms | 3.41 | 3.42 | 121 |
| whole on every rank (gathered once at bind) | 3.27 | 3.31 | 3.36 | 94 |
| and wq_b, wo_b with two MMA tiles per warp | 3.22 | 3.23 | 3.34 | 94 |
| and the head GEMV on an FP8 copy of the head (serve's default) | 2.48 | 2.50 | 2.61 | 94 |

- **Whole on every rank** (`BlockW::*_all`, [engine/dspark](../engine/dspark/README.md#on-a-rank-tp4),
  15.7 MB per rank): every rank computes the 40 projection chunks and the 128 router logits itself,
  the bits the gathers brought, so 9 of the pass's 25 collectives go with their pack and unpack
  launches. The projections read four times the bytes (2 MB a sublayer, ~9 µs more each), so the pass
  gains 60–100 µs.
- **Two tiles per warp** for wq_b and wo_b: half the CTAs (512 and 320). A row's MMA chain and the K
  split are those of one tile, so the bits are the same; wo_b takes ~50 µs instead of ~60.
- These two keep the tokens and the drafts: 1k febc7dc54ee40a39 with 72 verify steps (0..5 drafts:
  1, 19, 10, 7, 2, 33), 16k CED 81f772ab79a950e4 (1, 17, 20, 10, 12, 23), spectf 49, 31, 12, 2, 1, 1.
- **The FP8 head** (`--draft-fp8`; serve's default, `--draft-bf16` opts out). At bind the rank's head
  slice is quantised to e4m3 with a power-of-two scale per 32×32 block (amax / 448 rounded up,
  act_quant's rule): 165 MB beside the bf16 slice the verify reads. The GEMV widens it to bf16 in
  registers, exactly (wo_a's path, `dsk::gemv_fp8w_f32`), against the bf16 activations with fp32 sums:
  0.66 ms instead of 1.40. The drafts' arithmetic moves; the tokens cannot, and the acceptance barely
  does:
  - 1k: the same hash, and the same drafts as the bf16 head (72 verify steps, 1, 19, 10, 7, 2, 33);
  - 16k CED: the same hash and 2.072 accepted per step, the verify lengths a little different
    (1, 17, 18, 10, 15, 22);
  - spectf: 49, 32, 11, 2, 1, 1 (0.72 per step against 0.73);
  - bench.py's prompts: the same accepted per step on prose and JSON (0.81, 3.90), 2.19 and 2.11
    on code and agentic against 2.17 and 2.09.
- **Through the front-end**, `bench.py` decode (as in the table below: greedy, 512 tokens, two
  repetitions each, which agree within 1 %; the server at its defaults, one session per build):

  | decode tok/s | prose | code | agentic | json |
  |---|---|---|---|---|
  | gathers per sublayer (two sessions) | 67.1, 67.1 | 98.8, 98.8 | 96.2, 96.2 | 133.0, 133.1 |
  | whole, two tiles per warp, bf16 head (`--draft-bf16`) | 67.4 | 98.9 | 96.5 | 133.5 |
  | and the FP8 head (the default) | **69.4** | **101.9** | **99.8** | **136.0** |

  A session with only the whole projections measured 66.5, 97.2, 95.1 and 132.2: the 0.1 ms they
  save per step is below what separates two sessions.
- **What it is now** (the FP8 head, 6 rows, 22 µs of it idle between launches):
  - accept, 0.15 ms: main_proj 87 µs, two all-gathers ~35 µs with their pack and unpack, wkv, the
    window writes;
  - the three blocks, 1.55 ms: per block wq_b ~50 µs, wo_a 49, wo_b ~50, attention 28, the MoE ~150
    after its router, the mHC fronts ~55 (attention) and ~36 (FFN), a bf16 all-reduce 14–26 and an
    fp32 one 29–39;
  - hc_pre and the head GEMV, 0.70 ms;
  - the Markov chain, 0.21 ms: markov[0] ~90 µs (the 16.5 MB Markov head from DRAM), the other four
    ~19 µs each from L2, and five argmaxes of 4–10 µs;
  - conf and spec_next, 18 µs.
- **What is left.** The blocks are 60 % of the pass. Their weights (~75 MB a block) stream near full
  speed; the rest is the ring's latency (16 collectives, ~0.3 ms in all, the fp32 all-reduce the
  longest at ~35 µs) and small launches. Then: wo_a at ~175 GB/s (its fused act_quant fixes 64 CTAs);
  markov[0]'s DRAM read (~70 µs if the Markov head stayed in L2, ~45 µs from an FP8 copy); and accept,
  which needs `count` only for its window writes, so most of it could run beside the verify's tail.
  The head is a quarter of the pass now; a vocabulary subset would cut it further, at a cost in
  acceptance on rare tokens. [The draft pass's approximations](#the-draft-passs-approximations-a-study-2026-10-12)
  measures these levers and cheaper formats for the drafter's weights.

**Through the front-end.** `bench.py` as in [End to end](#end-to-end-through-the-front-end) (greedy,
512 tokens, two repetitions per prompt type that agree within 1.2 %), the server at its defaults (CED,
a 4 GiB Engram row cache) without and with `--spec`, one build, run back to back:

| | `--spec` | k = 1 | SGLang kit |
|---|---|---|---|
| decode tok/s: prose | 67.0 | 53.7 | 46.8 |
| decode tok/s: code | 99.0 | 53.7 | 77.4 |
| decode tok/s: agentic | 96.5 | 53.6 | 73.6 |
| decode tok/s: json | 133.2 | 53.7 | 106.9 |
| time to first token, short prompts | 128–150 ms (310 the first after start) | 125–147 ms (353) | 244–280 ms |
| cold prefill tok/s at 945 / 3,748 / 16,419 tokens | 2,725 / 5,489 / 6,494 | 2,761 / 5,485 / 6,510 | 1.9k / 3.7k / 4.3k |
| agentic: first turn (15.8k tokens), TTFT | 2.38 s | 2.39 s | 3.73 s |
| agentic: turns 1–5 (+1.7–2.1k tokens on 17.9–25.2k), TTFT | 0.42–0.45 s | 0.42–0.45 s | 0.71–0.79 s |
| agentic: decode of the 35–64-token answers | 84–112 tok/s | 52–53 tok/s | |

A 1 GiB row cache (`--engram-cache-mib 1024`) gives the same within 2 %.

Per prompt type, from the engine's report of each request (the two repetitions identical, step for
step, and identical to the run with the 1 GiB cache):

| | verify steps | drafts verified per step | accepted per step | tokens per step | ms per step | gate λ·ΔC: mean (min–max) | steps verifying 0, 1, 2, 3, 4, 5 drafts |
|---|---|---|---|---|---|---|---|
| prose | 282 | 1.43 | 0.81 | 1.81 | 27.1 | 0.229 (0.159–0.274) | 29, 131, 97, 21, 4, 0 |
| code | 161 | 3.01 | 2.17 | 3.17 | 32.1 | 0.323 (0.151–0.400) | 2, 25, 42, 29, 27, 36 |
| agentic | 165 | 2.93 | 2.09 | 3.09 | 32.1 | 0.316 (0.159–0.404) | 1, 32, 46, 27, 17, 42 |
| json | 104 | 4.43 | 3.90 | 4.90 | 36.9 | 0.415 (0.159–0.492) | 0, 8, 3, 5, 8, 80 |

- The verify length follows the text: on prose most steps verify one or two drafts, on JSON 80 of
  104 verify all five. The cost model (19.4 ms, 3.6 per draft verified, 3.3 for the draft pass) gives
  27.9, 33.5, 33.3 and 38.7 ms per step; the measured steps are 3–5 % shorter (2–4 % before the
  fusion).
- The gate starts at λ·ΔC = 0.159 (one token per 22.7 ms) and follows the rate: 0.23 on prose, where
  DESIGN.md's ~0.21 came from, 0.32 on code and agentic, 0.42 on JSON. The faster the decode, the
  more an extra row's 3.6 ms is worth, so the more sure of a draft the gate wants to be. The agentic
  turns' short answers stay at 0.20–0.24.
- Against the kit on the same prompts (with its MTP drafts): +43 % prose, +28 % code, +31 % agentic,
  +25 % JSON; against k = 1, 1.25, 1.84, 1.80 and 2.48 times. The best published GB10 numbers (~88
  prose, ~125 code: DESIGN.md) come from other prompts, so they compare by time per step and tokens
  per step rather than tok/s.
- The fusion moved the arithmetic, and the drafts accepted moved with it: code accepts 2.17 per step
  where it accepted 2.45 (102.2 tok/s before the fusion), prose and agentic a little more than before
  (0.81 and 2.09 against 0.77 and 2.05), JSON the same 3.90.
- The agentic answers are not the same text in the runs: bench.py puts a random nonce in the system
  prompt.
- Seeding the drafter's windows does not show under CED: the replay's 128 rows, once per turn. The
  cold prefills and the agentic turns take the time they take without drafts. With `--exact` every
  chunk seeds its last 256 rows and the 128 before each snapshot. That costs ~12 ms per chunk that
  is not the last (before the fusion, the 16k prompt: 4.79 s against 4.74 s): main_hidden of the
  seed rows, the all-gather, and `accept` in groups of 8 (96 calls for a 4096-row chunk with
  snapshots every 1024). The last chunk's seeding runs after its sample, and the server sends the
  first token before it drains the chunk.

### What remains for ~15 ms at k = 1

- **Collectives, 2.3 ms.** Done: `allreduce_hc`'s epilogue (hc_post and the collapse, 40 KB per row)
  runs on 40 blocks instead of 5 per row; on the ring, back to back, 15.8 → 14.2 µs at R = 1 and
  33.8 → 30.3 µs at R = 6, against 13.9 and 30.8 for comm's plain all-reduce. Tried and dropped:
  - posting `allgather_attn` from the wq_a/wkv GEMV's epilogue (0453c3f, reverted in bc4c4ca): the
    host-memory stores and each CTA's fence and count cost what the kernel boundary saved (GEMV +
    all-gather on the ring 16.6 → 16.6 µs at R = 1, 20.1 → 21.0 µs with the compressor's rows);
  - Sinkhorn in the all-reduce's epilogue (`allreduce_hc_mix`, the mixes riding the sublayer's own
    all-reduce): +3.4 µs at R = 1 and +4.0 µs at R = 6 on the path, mostly 40 dependent divisions.
    The FFN's mixes take a one-block collective on `sa` instead (`mix_allreduce`, 10.6 µs off the
    path); `allreduce_hc_mix` stays in fused.cu, tested, unused. The attention's ride its all-gather
    for 0.8 µs (R = 1) and 1.1 µs (R = 6) more.
  - bf16 MoE partials were not tried: model.py sums the routed experts in fp32 and all-reduces that
    (`MoE.forward`), so they would add a rounding the reference does not have, to send 10 KB less
    per row through a latency-bound collective.

  What is left is the ring's own time and waiting for the slowest rank: per step 125 collectives on
  the path whose p10 durations add up to 1.37 ms at k = 1.
- **Weights, 15.7 → 14.4 ms.** `k_gemv_fp8` is the slowest at ~85 %: wq_a + wkv (2.3 MB) is
  latency-bound, and wq_b / wo_b run below the bandwidth. A single-box copy of the decode chain
  (`engine/kernels/tests/bench_chain.cu`, real weight kernels and timed stand-ins; 18.63 ms at
  k = 1) puts the call sites at, in GB/s: wq_a + wkv 190, wo_b 217, wo_a 212, router gate 207, moe_up
  215, the rest 230–240 (`chainprof.py`). Back-to-back weight kernels alone reach ~225 GB/s.
  - Splitting the router gate across ranks costs more than it saves (+0.5 ms at k = 1, +0.15 ms at
    k = 6), so it stays replicated.
  - L2 prefetch in the PDL window does not work on GB10 ([findings](findings.md#memory-bandwidth)).
  - **Persistent GEMV: on for one row.** `k_gemv_fp8p` (gemv_fp8.cu) and `k_gemv16p` (gemv16.cu,
    the router gate) run one CTA per SM, each over a contiguous run of 8-row tiles. Warp w streams
    K-slice w of every tile through a ring of up to 16 `cp.async` stages of 512 bytes in shared
    memory (~64 KB a CTA, ~3 MB on the GPU), requested before `griddepcontrol.wait`, so the ring
    fills while the collective before it runs. Per tile there is one barrier and the classic
    kernel's slice-order sum, so the outputs are the classic kernel's bit for bit (`test_kernels
    pers`: wq_a + wkv, wq_b, wq_b + the indexer's, wo_b, Engram's wkv and the router gate at 1, 3,
    6 and 8 rows, every ring depth and grid). The normed GEMVs (wq_a + wkv, wq_b, the gate) take
    it at one row (`DSK_PERSIST`, dsk.h; other sites opt in with `GemvCfg::ring`). Decode loop at
    1k, k = 1, alternating against master a3aff57: 18.58, 18.57 → 18.31, 18.32 ms (54.6 tok/s);
    `--spec` unchanged (9.30 → 9.32 ms per token, steps of 2–6 rows within ±0.5 ms); the hashes
    hold (1k febc7dc54ee40a39 with and without drafts, tf 842278b28d87dad0).
    - What it gains is the windows: in bench_chain at one row (exclusive ms per step) the router
      gate 0.78 → 0.42, wq_b 1.40 → 1.14, wq_b + the indexer's 0.52 → 0.02, wq_a + wkv
      0.46 → 0.36.
    - Some of it moves elsewhere. A persistent CTA holds its SM, so `k_moe_up` (a whole SM's
      registers per CTA) no longer starts beside the router GEMV's tail. At index layers the
      indexer's 4-block wproj GEMV on the side stream becomes the path. wo_b, which has no window,
      is 5 µs a layer slower persistent than its 640-CTA classic kernel behind wo_a: CTAs that
      start late on SMs wo_a still holds finish late. So wo_b and Engram stay classic.
    - With more rows it loses, and the shared memory the normed input takes from the rings (the gate
      at 6 rows: 62 KB of the 80) is not why. With the input in registers instead (each lane
      normalizes and quantizes the rows of its own MMA fragments, the same bits, `test_kernels pers` at
      1–6 and 8 rows; ba1b161 on the kernels branch worktree-agent-a1102a1670c38111d, not merged) it
      loses more. bench_chain, step median in ms over two alternating reps, each site persistent from
      two rows on, input in shared memory / in registers:

      | rows | classic | wq_a + wkv | router gate | wq_b | all three |
      |---|---|---|---|---|---|
      | 2 | 23.05 | +0.31 / +0.29 | +0.12 / +0.38 | +0.37 / +0.59 | +0.39 / +1.11 |
      | 3 | 28.27 | +0.07 / +0.23 | +0.37 / +0.40 | +0.70 / +0.80 | +1.04 / +1.55 |
      | 4 | 33.73 | +0.15 / +0.25 | +0.13 / +0.28 | +0.57 / +0.89 | +1.05 / +1.62 |
      | 5 | 36.57 | +0.17 / +0.28 | +0.37 / +0.08 | +0.56 / +0.89 | +1.00 / +1.49 |
      | 6 | 37.08 | +0.14 / +0.30 | +0.36 / +0.06 | +0.49 / +0.76 | +0.92 / +1.41 |

      So the normed GEMVs stay persistent at one row only (`DSK_PERSIST` unchanged), and the register
      path was not merged (the gate's instantiation took 213 registers, bounded to 256 threads, since
      under a 512-thread bound ptxas held it to 64 and spilled). At six rows two nsys captures trace
      the loss to the kernels after them, not the sites themselves (below, "The persistent normed
      GEMVs at six rows").
    - Its rings need the SM in a large-shared configuration, which a kernel can only change on an
      idle SM. Before the kernels asked for it (`cudaSharedmemCarveoutMaxShared`), wq_a + wkv's CTAs
      could not start during the all-reduce after `k_moe_down`, and the prefetch did not happen.
      Asking for it on every launch (`DSK_CARVEOUT=100`) made bench_chain's base 0.2 ms faster
      and the decode loop at k = 1 0.1 ms faster, but spec steps of 2–6 rows 0.3–1.9 ms slower
      (9.29 → 9.61 ms per token): the 28 KB of L1 left beside 100 KB of shared memory does not
      hold their activations. So it is off.
  - **`k_moe_up`'s start: the router's loads in one round trip.** A per-CTA `globaltimer` trace of
    `k_moe_up` and `k_moe_down` in bench_chain (a scratch build: entry, past the wait, past the
    router, each task's start and end, exit) put 10.6 µs between `k_moe_up`'s `griddepcontrol.wait`
    and its first task. On the logits path every CTA runs `router_row`, which used each group of 32
    experts' logit and bias as soon as it had loaded them: 12 dependent round trips at E = 384
    (5.8 µs with the bias already in L1). It now issues every load before any math: 5.1 µs, and
    `k_moe_up` ends 4.4 µs earlier per layer (123.7 → 119.3 µs after the gate). Same arithmetic and
    order: the router and `moe_routed` tests are bit-exact. Decode loop at 1k, alternating against
    master 8ec7de7: k = 1 18.27, 18.40 → 17.99, 18.00 ms; `--spec` 9.29, 9.31 → 9.23, 9.30 ms per token
    (steps of 2–6 rows between 0.43 ms faster and 0.16 ms slower, pair by pair); the hashes hold
    (1k febc7dc54ee40a39 with and without drafts, 16k CED 81f772ab79a950e4, tf 842278b28d87dad0).
    - Past its start `k_moe_up` streams at the bandwidth: in the trace 47–48 CTAs are in a task
      until ~95 µs after the gate, then the last wave of 30 routed tasks runs on ~30 CTAs at ~220
      GB/s; in bench_chain its exclusive time runs at 233 GB/s (chainprof). The CTA whose SM the
      mix collective holds starts its first task ~50 µs late and still finishes with the rest
      (without the mixes `k_moe_up` ends at the same time, 123.5 µs).
    - A task queue (`MoeCfg::dyn`: past the first wave a task goes to whichever CTA is free)
      measured the same as the static schedule (bench_chain at one row 18.685 → 18.685 and
      18.427 → 18.411 ms; `k_moe_up` ends 125.4 against 125.6 µs after the gate over 600 layers).
      It stays behind the option, off, tested bit-exact at 1–8 rows.
- **Mixes: done.** Each rank projects its 5,120 columns and the partial sums are added across the
  ranks off the path ([The step](#the-step)): exclusive time 0.30 → 0.01 ms at k = 1 and 0.9 → 0.3
  ms at k = 6, hc_fn reads 157 → 39 MB per step, the decode loop 0.34 ms faster. The 0.5 MB a
  sublayer still reads lands in the weight kernels' time: against the old mixes the router GEMV ends
  0.9 µs later and `k_moe_up` takes 2.8 µs longer, while `k_moe_down`, which used to share its time
  with hc_finish, takes 2.9 µs less. The side stream's priority places hc_project's 10 blocks and the
  mix collective's spinning block ahead of `k_moe_up`'s CTAs, which take a whole SM's registers
  each; forking the FFN's mixes after the router instead (so they run under `k_moe_down`) measured
  the same in the decode loop (18.57 against 18.56 ms over three runs each at 1k) and was dropped.
  Folding each projection into the epilogue of the all-reduce before it was considered and dropped.
  Each block would load its slice of hc_fn while it waits. That takes the side kernels out of
  `k_moe_up`'s way, though not the bytes.
  - The ceiling, measured: bench_chain with every mix off (`--variants base,nomix`: no hc_project,
    mix_ranks or mix collective), 40 interleaved rounds, two runs, step median in ms. Master 09847af,
    2026-10-12:

    | rows | base | no mixes |
    |---|---|---|
    | 1 | 18.03 / 18.30 | 17.87 / 18.12 |
    | 2 | 22.92 / 22.92 | 22.95 / 22.97 |
    | 4 | 30.09 / 30.10 | 30.13 / 30.14 |
    | 6 | 36.91 / 36.93 | 36.86 / 36.88 |

  - Taking every mix away gains 0.16–0.18 ms at one row (0.9 %), nothing at two and four rows, and
    0.05 ms at six (0.14 %).
  - A fold could take only part of that, and it would have to permute allreduce_hc's columns to keep
    the bits.
- **Host: done.** Every rank drives its own decode loop from the sampled tokens and queues step
  n + 1 before step n ends ([Process model](#process-model)); the host adds 1 µs (Engram submit)
  and 5 µs (launch) per token, off the GPU's path.
- **What is left over the floor.** In bench_chain at one row after the router change (chainprof)
  the weight sites' 3,358.6 MB per step stream at 234.6 GB/s in their exclusive time, 14.32 ms: the
  floor. The rest of the 18.3 ms is time when no weights stream: collectives 2.71 ms
  (`allreduce_hc` 1.31, the all-gathers 0.83, the mix collective 0.57), attention 0.79 ms (the
  attention kernel 0.44, the indexer 0.19, the combine 0.17) and the indexer's wproj GEMV on the
  side stream 0.46 ms. The largest piece a weight kernel can still take is attention's. wo_a's CTAs
  are already resident during it (`attn_kernel` and `attn_combine_kernel` trigger their dependents
  at their start, and in the engine wo_a's first CTA starts 3 µs after the attention kernel's), but
  the classic kernel loads only its first batch before the wait, and wo_a is 8.4 MB (36 µs). A
  persistent ring for its format (`k_gemv16p` with FP8 rows and block scales) would fill ~3 MB in
  the ~14 µs of attention at k = 1.
- **What is left over the floor at six rows.** bench_chain at six rows with 34 experts a layer
  (chainprof as of 693d278, 15 steps; the classic normed GEMVs, as the engine runs them from two rows
  on): the weight sites' 8,622.8 MB per step stream at 224.0 GB/s in 38.50 ms of exclusive time,
  against a floor of 36.69 ms at 235 GB/s. The step takes 42.55 ms, so 5.9 ms are over the floor:
  - collectives 3.39 ms: `allreduce_hc` 2.57 (80 calls at 32 µs, its whole back-to-back cost at R = 6;
    1.31 at one row) and the all-gathers 0.83 (as at one row);
  - weights below the bandwidth 1.8 ms: `k_moe_up` 20.47 ms at 220 GB/s (1.3 over its 19.14) and the
    normed GEMVs after the collectives, the router gate 0.94 ms at 167 GB/s, wq_a + wkv 0.67 at 138,
    wq_b 1.81 at 186 and with the indexer's 0.63 at 201 (floors 0.67, 0.39, 1.43 and 0.54);
    `k_moe_down` 9.36 ms at 240 GB/s, part of its reads under `k_moe_up`'s tail; wo_a, overlapping
    the attention, takes 1.20 ms exclusive, under its 1.43 ms of bytes;
  - attention 0.74 ms.

  The side stream takes nothing off the path: `hc_project` 0.005 ms and the mix collective ~0
  exclusive (bench_chain base 37.380 against nomix 37.346 ms at six rows). An earlier chainprof gave
  overlapped time to whichever kernel started first, so the FFN's `hc_project`, which starts 1.4 µs
  before `k_moe_up`, took `k_moe_up`'s first ~17 µs a layer (0.66 ms a step) and the mix collective
  0.29 ms of the MoE's; this paragraph had those figures until 2026-10-12. The indexer's wproj GEMV,
  0.46 ms of the path at one row, is off it at six (0.007 ms). Each distinct expert adds ~4.7 MB per
  layer and rank to the MoE's reads (`k_moe_up` 24.70 → 112.44 MB a call and `k_moe_down` 12.35 →
  56.22 MB from 6 to 34 experts), ~0.8 ms per step at the bandwidth, so a row that shares no experts
  with the others costs ~4.8 ms.
- **The persistent normed GEMVs at six rows, and the static-tile guess: dropped.** The guess was that
  the classic normed GEMVs (one tile per CTA, assigned up front) lose time when side-stream kernels
  hold SMs, so a persistent GEMV that splits its work as CTAs free up would recover it. Two nsys
  captures of bench_chain at six rows (34 experts, 15 steps each) say no:
  - wq_b's calls (74 µs, 186 GB/s) overlap no side-stream kernel at all, and the router gate's calls
    with more overlap take 43.5 µs against 42.8 µs for those with less (10.4 against 7.1 µs of
    overlap): the side kernels cost the gate ~0.2 µs per µs of overlap, under 0.1 ms a step (the
    step without the mixes is 0.03 ms faster).
  - The persistent kernels on the gate and wq_b (`pgate+pqb`) make the step 1.22 ms slower (42.55 →
    43.77 ms). wq_b's sites get faster (with the indexer's, 2.44 → 2.19 ms exclusive; the gate 0.94 →
    0.97) but in the same steps `k_moe_down` takes 0.60 ms more and wo_b 0.54 ms more: they ask for
    the large-shared carveout, which leaves 28 KB of L1 to the kernels after them, as
    `DSK_CARVEOUT=100` measured for spec steps.

  So the GEMVs' time below the bandwidth at six rows is their own, not the side kernels', and the
  dynamic-split persistent GEMV was not written. The persistent normed GEMVs stay at one row
  (`persist_rows()`).
- **wo_a's weights during the attention, and the normed GEMVs' second batch: measured, not merged.**
  Four changes on the kernels branch (worktree-agent-a1102a1670c38111d at 2999809), each the same
  bits, timed in one ring job on 2026-10-12: the decode loop at 1k twice per build, alternating
  (`--spec` and k = 1, ms per token), and bench_chain (step medians over 30 rounds; 27 experts a
  layer from two rows on). None is 2 % faster, the bar for merging a speed change:

  | change | `--spec` | k = 1 | bench_chain, 1 row / 2–6 rows |
  |---|---|---|---|
  | wo_a's classic kernel requests both weight batches before its PDL wait (f4bd98e) | 9.232 → 9.247 | 17.946 → 17.925 | −0.00 / −0.06 to −0.14 ms |
  | `k_woa_p`: wo_a on a persistent kernel whose `cp.async` ring fills during the attention, at 1–8 rows (4d726cc, 03b1d96, 82e61d6) | 9.247 → 9.372 (+1.4 %) | 17.925 → 18.177 (+1.4 %) | +0.04 / +0.71 to +0.85 ms |
  | the normed GEMVs (wq_a + wkv, wq_b, the router gate) request their second batch before the wait too, at 2–8 rows (8dd693a) | 9.247 → 9.155 (−1.0 %) | one row: unchanged | −0.00 / −0.20 to −0.24 ms |
  | the indexer's wproj on the compressor's row kernel (`attn::compressor_project_rows`, bench_chain's `wrow`, 875b956) | 9.247 → 9.259 | 17.925 → 17.864 | +0.07 / −0.02 to +0.01 ms |

  The baseline of each line but the first is the branch as committed, with f4bd98e's prefetch. The
  1k hash held in every run (febc7dc54ee40a39). With `k_woa_p`, the wproj swap and the prefetch
  together: tf 842278b28d87dad0, 16k CED 81f772ab79a950e4, long28k CED 14650fb0739d0383 (until
  2026-10-13 the hash's seed, see [Golden check](#golden-check); now 8f08a153d415c5c8), and 1k
  febc7dc54ee40a39 with FP8 drafts. With the normed prefetch: tf 842278b28d87dad0. `k_woa_p`'s loss
  was not traced, and is still untraced (2026-10-12): the kernel was not taken further. It asks for the
  large-shared carveout, which cost the persistent normed GEMVs their gain at six rows (above). The code
  stays on that branch: `k_woa_p` is 4d726cc, 03b1d96 and 82e61d6, and the branch's head is 2999809.

### The collectives' time: a proposal (2026-10-12)

In the decode loop at k = 1, rank 0's step (17.9 ms) runs 3.6 ms over the 14.3 ms weight floor:

- 2.0 ms of collectives;
- 0.7 ms of attention;
- 0.8 ms of weights below the bandwidth;
- 0.2 ms of gaps.

At six rows, the collectives take 3.45 ms. This section covers where that time goes, its floor, what
could hide or remove it with the same bits, and why wo_a's prefetch came out neutral. Nothing in it
is built.

Sources:

- nsys of rank 0 in the decode loop at 1k, master 123f959's engine (`--profile`): 126 steps at
  k = 1, and the 13 six-row steps of a `--spec` run;
- three k = 1 captures that change rank 0 only (run4.sh's `WRAP`): `DSK_PERSIST=0`,
  `DSK_CARVEOUT=100`, and master again;
- [ring.md](ring.md) for the floors.

A main-stream kernel's charge is the time it extends the path, credited as chainprof.py does. It
is measured on rank 0 only, so a collective's charge includes waiting for a slower rank.

**Budget.** The table gives the median charge per call (µs) and the charge per step (ms). The floor
is [ring.md](ring.md)'s back-to-back time for the size: two hops of ~4.4 µs, the 3.3–3.7 µs GPU side,
and the bytes. The floor includes a launch that PDL hides here, which is how a charge can come out
under it.

| collective (calls a step) | k = 1: bytes a rank, µs, ms | floor, µs | six rows: bytes, µs, ms | floor, µs |
|---|---|---|---|---|
| `allgather_attn` (40) | ~1–1.5 K, 17.6, 0.73 | 10.6–11.6 | ~6–9 K, 11.8, 0.61 | ~12.5 |
| `allreduce_hc` after the MoE (40; fp32: the MoE is tensor-parallel, this is its only exchange) | 20 K, 17.2, 0.70 | 15.5 | 120 K, 43.1, 1.71 | ~36.6 |
| `allreduce_hc` after the attention (40; bf16) | 10 K, 10.5, 0.48 | 12.1 | 60 K, 16.0, 0.72 | 24.5 |
| Engram's all-gathers (4) | 12.5 µs, 0.05 | ~10–12 | 0.09 | |
| sampling, `k_sample` (1) | 16.6 µs, 0.02 | 8.1 + the argmax | 23.5 µs, 0.03 | |
| the drafter: 6 all-reduces (~26 µs), 5 all-gathers, 5 argmax | – | | 0.29 | |
| total | **1.99 ms** | ~1.6 ms | **3.45 ms** | ~3.3 ms |

- The head is weights (1.40 ms at 236 GB/s), not a collective.
- The 2.71 ms of collectives in [What remains](#what-remains-for-15-ms-at-k--1) is bench_chain's.
  Its stand-ins hold for the engine's durations, not their charges, and it charged 0.57 ms to the
  mix collective, which the engine runs off the path.
- At six rows, the collectives add up to about their floor. Rank 0's attention all-reduce is under its
  floor and its MoE all-reduce over it by about as much: skew.
- At k = 1 they are 0.4 ms over. 0.28 ms of that is `allgather_attn`'s ~7 µs a call, which looks
  like the cost of wq_b's ring filling under it (below).
- Under the same bits, the only thing that makes a collective itself faster is relay cut-through
  (F). The rest of their time can only be hidden: weights streaming while they wait.

**The windows.** With PDL, a collective lets the next weight kernel launch once it has posted. That
kernel can load weights from then until the collective ends, before its `griddepcontrol.wait`. The
tail is how long it runs after the collective ends. Medians, µs:

| collective → next weight kernel | k = 1: window, tail, GB/s of the tail | six rows: window, tail, GB/s |
|---|---|---|
| `allreduce_hc` (attention) → router gate (3.93 MB) | 9.4, 17.9, 219 | 14.8, 26.2, 150 |
| `allreduce_hc` (MoE) → wq_a + wkv (2.30 MB) | 16.2, 15.0, 153 | 41.4, 17.9, 129 |
| `allgather_attn` → wq_b (10.5 MB) | 16.1, 34.8, **302** | 9.9, 60.4, 174 |

- At k = 1, only wq_b uses its window. Its persistent ring takes ~2.3 MB during the all-gather, so
  its tail runs above the DRAM's 235 GB/s.
- The gate's and wq_a + wkv's rings stay empty. Their tails are a cold kernel's. Without its ring,
  the persistent wq_a + wkv is slower than the classic kernel: with `DSK_PERSIST=0`, rank 0's tail
  is 12.5 µs against 15.0, ~0.1 ms a step.
- bench_chain shows the rings can fill there (captures of 2026-10-12):
  - With wo_b on the persistent kernel, the gate's exclusive time falls 0.67–0.69 → 0.42 ms a step.
  - With `DSK_CARVEOUT=100` as well, wq_a + wkv's falls 0.63–0.68 → 0.36.
  - Its collectives are stand-ins. They took DSK_CARVEOUT's attribute, and behind the persistent
    wo_b, which lets its dependents launch at its start, they started beside its CTAs.
- In the engine, `DSK_CARVEOUT=100` on rank 0 leaves both tails as they were (18.0 and 15.0 µs).
  So the likely blocker is the all-reduce kernels themselves. They take no carveout attribute
  (`attn::launch`), and their 40 blocks land on SMs that wo_b and `k_moe_down` have left, so they
  set the split there. A ring needs the large-shared split, and a split changes only on an idle SM
  (above). The GEMV's CTAs then fit on the ~8 other SMs.
- The all-gather has 4 blocks, so it leaves 44 SMs in the split of the persistent wq_a + wkv before
  it.
- Registers are tight as well. The fp32 all-reduce's block is sized for 8 rows (288 threads × 56
  registers); beside wq_a + wkv's CTA (512 × 96) the two use 65,280 of the SM's 65,536.

**A filling ring costs the collective under it.** wq_b's ring fills during the all-gather, and the
all-gather is the one collective over its floor. With `DSK_PERSIST=0` on rank 0, rank 0's all-gather
took 14.6 µs instead of 17.5. That was although rank 0 then posted ~2.5 µs earlier, its classic
wq_a + wkv being quicker. At two rows, with every rank on the classic kernels, the same all-gather
takes 12.4 µs with twice the bytes. So ~3–5 µs of it is the fill. The ring's buffers, its NIC's DMA
and the proxy's relays live in the LPDDR5X that the fill saturates. Tests that change rank 0 alone
cannot separate this from skew, so this is the least certain number here. Every estimate below pays
it per collective.

**What could hide or remove it.** Every option keeps the bits. Percentages are of 17.9 ms (k = 1)
and 41 ms (six rows).

| | k = 1 | six rows | confidence | size |
|---|---|---|---|---|
| A. let the gate's and wq_a + wkv's rings fill during the all-reduces | −0.17 to −0.57 ms (1–3.2 %) | 0 (no rings there) | low-medium | small |
| B. weights into L2 from the collective's own blocks while they wait | about A's, instead of it | −0.3 to −1.1 ms (0.7–2.7 %) | low | medium |
| C. `allreduce_hc` pipelined with the kernel before it | ~0 | −0.3 to −0.6 ms (0.7–1.5 %) | low-medium | large |
| D. PDL for what is left of the GPU side | < 0.05 ms | < 0.1 ms | high | small–medium |
| E. fusing collectives | 0 | ≤ 0.15 ms (the drafter's) | high | medium |
| F. relay cut-through | ~0.05 ms (0.3 %) | ~0.3 ms (0.7 %) | medium | medium |

- **A.**
  - The change: at one row, the two `allreduce_hc` launches ask for the large-shared carveout
    (`cudaLaunchAttributePreferredSharedMemoryCarveout` in fused.cu), and so do wo_b and
    `k_moe_down` (DSK_CARVEOUT's attribute, at one row only).
  - At one row, `allreduce_hc` also launches 32 + R × items threads (64), not the 8-row block.
  - That is launch configuration only, and each item's arithmetic and order stay as they are. Two
    and more rows keep today's launches: the large-shared split cost those steps 0.3–1.9 ms under
    `DSK_CARVEOUT=100`.
  - The gain is bench_chain's on the two GEMVs, −0.30 and −0.27 ms a step, less what the fill
    costs the 80 all-reduces: −0.57 ms if nothing, −0.17 ms at 5 µs each.
  - If the rings still do not fill, run wq_a + wkv on the classic kernel at one row: −0.1 ms.
- **B.**
  - While they wait, the all-reduce's 40 blocks (or the all-gather's 4) read the next weight
    kernel's first bytes with `ld.global.cg`. Size the read to finish before the contributions
    arrive: ~2 MB at one row, where two hops are ~8.8 µs, and more at six rows.
  - GB10's L2 keeps such lines: findings re-read them at 1.4–1.6 TB/s 300 µs later. It did not keep
    the prefetch instructions' lines, and a GEMV prefetch into L2 in the PDL window lost.
  - The runner passes the next GEMV's weight pointer and byte count. Unlike A, B needs no
    co-resident CTAs and works at every row count.
  - At six rows the windows add up to 66 µs a layer, 2.6 ms a step. The MoE all-reduce's 41 µs
    holds all of wq_a + wkv and wq_b's first ~7 MB, within the 24 MB L2.
  - Unmeasured: how much faster the six-row classic GEMVs (130–175 GB/s) take bytes from L2. The
    range runs from "no faster" to "twice as fast", less the same ~3–5 µs a collective as A.
- **C.**
  - Post `allreduce_hc` in column chunks as `k_moe_down` and wo_b finish them, so the bytes travel
    during the kernel.
  - At one row, the 10–20 KB are all latency: the last chunk still takes two hops. At six rows,
    ~24 µs of the MoE all-reduce's 36.6 is bytes, and ~12 µs of the attention all-reduce's 24.5.
  - It needs the producers to finish their columns in order (the work is kernels and comm), one
    sequence number and flag per chunk in comm, and fused.cu.
  - It shortens the window that B would fill, so it comes after B, not with it.
- **D.**
  - PDL already chains the GEMVs, the attention and both fused collectives. ring.md's open item
    on this is done for the main stream.
  - Plain launches remain: Engram's 4 all-gathers, the sampling and the drafter's 16 collectives,
    at ~0.5–2 µs of launch gap each.
  - The rest of the GPU side is reading the contributions from host memory, after all six flags.
    Reading each one as it lands, into registers, and summing them in rank order afterwards would
    hide ~0.5 µs a collective (< 0.1 ms a step).
- **E.** No two collectives on the path are adjacent: a GEMV that needs the first one's result sits
  between them. The attention's mixes already ride its all-gather, and the FFN's are off the path.
  Posting the all-gather from the GEMV's epilogue was tried and reverted (0453c3f). That leaves the
  drafter's 0.29 ms.
- **F.** ring.md's ~0.4 µs at 10 KB, scaled by the bytes relayed: ~5 µs on the MoE's 120 KB and
  ~2.4 µs on the attention's 60 KB at six rows.

**Why wo_a's prefetch came out neutral: it landed, and the attention paid for it.** Rank 0 nsys at
k = 1 of the A/B build, with the prefetch off and then on:

- wo_a's exclusive time fell 1.335 → 1.185 ms a step: −3.75 µs a call, ~0.9 MB of its second batch
  at 235 GB/s.
- In the same window, `attn_kernel` went 17.7 → 21.6 µs a call (exclusive 0.472 → 0.577 ms) and
  `index_dense` 17.7 → 20.8 µs.
- The step went 17.857 → 17.851 ms.

The attention's window is not idle DRAM time. The attention kernel is bound by the latency of its
own loads, and wo_a's 0.9 MB queued ahead of them. A collective's window is the same kind of window,
only quieter: the all-gather above pays ~3–5 µs for wq_b's 2.3 MB.

**Verdict.** Nothing here clears 2 % with confidence.

- A (k = 1) and B (six rows) might, at the top of their ranges. Both depend on what a filling ring
  costs the collective under it, which only rank-0 tests have touched.
- D, E and F are each under 1 %. C reaches 1.5 % at six rows at most, is large, and competes with B
  for the same window.

Proposed order, each step a measurement first:

1. A, in the engine at k = 1: nsys on rank 0 for the two tails and the all-reduces' charges, then
   the decode loop at k = 1 and `--spec`, alternating against master. It is small, and it measures
   the cost per collective that B also pays.
2. B on one box, if A shows that cost is ≤ 3 µs. bench_chain's collective stand-ins read the next
   GEMV's first window × 235 GB/s with `ld.global.cg` while they hold, and the step at six rows has
   to gain ≥ 0.8 ms before the reads go into fused.cu.
3. C only if B fails, or leaves the MoE all-reduce's window idle.

### Experiment A: the large-shared carveout at one row (2026-10-12)

The proposal's step 1. Three knobs, off by default, each launch configuration only (kernels/dsk.h),
on branch kernels-a1 (85c7314). They were not merged: the result is under the bar, so master has none of them:

- `DSK_CARVE1`: at one row, the two `allreduce_hc` launches, wo_b and `k_moe_down` ask for the
  large-shared carveout.
- `DSK_ARSLIM`: at one row, `allreduce_hc` launches 64 threads, not the 8-row block.
- `DSK_QA_CLASSIC` (S): at one row, wq_a + wkv runs on the one-tile-per-CTA kernel.

A is the first two together. One binary, the variants set by `--setenv`. The decode loop at 1k ran
256 steps, in three alternating rounds. The table gives ms a token over the last 247: the mean of
three runs, then the change against the knobs off.

| | k = 1 | with drafts |
|---|---|---|
| off (runs 17.915–17.999 and 9.163–9.236) | 17.944 | 9.192 |
| A | 17.632, −1.74 % | 9.204, +0.13 % |
| S | 17.834, −0.61 % | 9.234, +0.46 % |
| A + S | 17.625, −1.78 % | 9.215, +0.25 % |

Every 1k run gave febc7dc54ee40a39. With A + S on, tf gave 842278b28d87dad0 and 16k CED
81f772ab79a950e4.

Rank-0 nsys at k = 1, with A on against the knobs off. The values are medians over 40 calls × 126
steps:

| site | all-reduce charge, µs | next GEMV's tail, µs (GB/s) | ms a step |
|---|---|---|---|
| attention all-reduce → gate | 10.8 → 17.0 | 18.0 → 10.2 (218 → 385) | +0.22, −0.29 |
| MoE all-reduce → wq_a + wkv | 18.5 → 13.9 | 15.1 → 8.8 (152 → 260) | −0.11, −0.26 |
| all-gather → wq_b | 17.4 → 17.7 | 35.1 → 34.7 | 0 |

The other kernels added 0.08 ms. The step went 17.829 → 17.497 ms (−1.86 %).

- **The rings fill now.** Both tails run above the DRAM's 235 GB/s.
- **The fill costs the collective under it.** The attention all-reduce paid +6.2 µs, more than the
  3–5 µs estimated above. Its ~10 µs window is shorter than the gate's ~3 MB fill, so the step
  keeps only 0.07 ms of that site's 0.29 ms.
- **The MoE all-reduce came out 4.6 µs faster.** Its window is 17 µs. That figure nets the fill's
  cost against what its launch configuration gained, and an nsys of A + S would separate them:
  with S, wq_a + wkv fills nothing.
- **S helps only without A.** With A, the persistent wq_a + wkv's ring fills, and that kernel beats
  the classic one.
- **Verdict.**
  - The bundle is −1.78 % at k = 1 and neutral with drafts. That is under the 2 % bar, so the
    defaults stay off.
  - At the attention site the fill cost is twice the 3 µs that B needed (the order above). B pays it
    on every collective, so it does not go ahead as proposed.
- **Not measured yet (launch configuration).**
  - A at the MoE site only: −0.37 ms by the charges above.
  - A shallower ring for the gate at one row, so that its fill ends inside the attention
    all-reduce's window.

### Draft trees: a feasibility study (2026-10-12)

The question: would a tree of drafts, several candidates per depth verified in one step, beat today's
chain by 10 % or more on prose and code? Verification is exact, so any tree keeps the outputs.

This was measurement and analysis only. The probe is bench-only and off by default, and it stays on
the branch kernels-tree (154f0c1, not merged: trees are not being built). Serving is unchanged. The analysis is engine/dspark/ref/tree_study.py, and
tree_costs.py fits its costs.

**Verdict: no.**
- **Siblings add about 1 % at most.** At the measured cost of a verify row, the best tree beats the
  best gate on the chain alone by +0.4 % on prose, +1.0 % on code, +1.2 % on agentic and 0 on JSON
  (α tuned per type). With one setting for every type, the margin is at most 0.4 %.
- **The larger gains are the gate's.** A policy that gains more against today's chain does so by
  verifying fewer or more chain drafts. On code that is +3.8 %, and the same setting costs agentic
  1.7 %.
- **A sibling row costs 2.4–3.5 ms.** That is 1.1 ms for the row itself, plus the experts it adds
  (1.8–3.3 per layer that the chain does not route) at 0.71 ms each.
- **10 % on prose and code needs a sibling row of about 0.5 ms**, measured against today's chain. A
  row's own share is 1.1 ms before any expert.
- **The ceiling.** With free siblings, the drafter's ranks allow +22 % on prose and +19 % on code
  over the best chain gate.

**How DSpark proposes** ([engine/dspark](../engine/dspark/README.md)):
- **The positions are parallel, from one pass.**
  - The three blocks run once over a block of 5 rows: the bonus token, then the noise token 128799
    four times.
  - The block attends to itself non-causally, so all 5 depths' hidden states come from that pass.
  - No draft enters the blocks.
  - The head gives every depth's logits.
- **The Markov head chains the depths.**
  - Depth j + 1 sees depth j's draft only through bias_j = markov_head(markov_embed(ids[j])). That is
    a rank-256 lookup, then a GEMV over the vocabulary slice (16.5 MB a rank, which stays in L2).
  - ids[j + 1] = argmax(logits[j] + bias_j), with one cross-rank argmax per depth.
- **The confidence head scores the parent.**
  - conf[j] = proj([x_j, markov_embed(ids[j])]) is the logit of P(d_{j+1} accepted | d_1..d_j
    accepted).
  - It does not depend on the child, so siblings under one parent share it.
  - Today's gate verifies d_j while the product of the first j sigmoids is ≥ λ·ΔC.
- **What top-k or a tree would take.**
  - The pass and the head are shared.
  - Per depth: the Markov bias of every parent ([256] × [P] instead of [256] × [1], over the same
    matrix), a top-k per parent instead of the argmax, one gather of P·k candidates, and the
    confidence per parent. A rough estimate: +0.1–0.2 ms on the 2.55 ms pass.
  - The depth stays at 5. A deeper tree needs a second pass, and JSON already takes 4.9 of the 6
    tokens a pass can give.

**Method.**
- **The true path.** The target's own greedy k = 1 generation, 512 tokens from each bench prompt.
  None of the four hits EOS.
- **The probe.** `spectf --tree-probe FILE` (EXPERIMENT, default off):
  - It runs the target teacher-forced along the path: one 1-row spec step per position, with the
    drafter's bonus forced to the path's token.
  - After draft(), `Draft::probe` reruns the Markov chain with every depth's parent forced to the
    path.
  - It records at each depth 1–5: the top 8 tokens with their probabilities (softmax over the whole
    vocabulary, by a cross-rank log-sum-exp), the path token's rank and probability, and the
    confidence logit.
- **Sibling routing.** `--tree-sib 16`: every 16 positions, the probe's 4 most likely depth-1 tokens are
  routed, one eager 1-row step each at the true context. The 4 path rows are routed in one 4-row
  step. That gives 32 samples a type.
- **Checks (every type).**
  - At every position, the probe's top 1 is draft()'s d1, and sigmoid(conf[0]) is its survival[0].
  - After a matching prefix, the chain's next draft is the probe's top 1.
  - The leading matches are the leading rank-0 count.
  - The step's own sample is the path's token at 483, 502, 496 and 509 of 512 (prose, code, agentic,
    JSON). spectf prefills the prompt less one token and runs that token as a step, so the prompt's
    KV differs from the generation's in summation order, and greedy near-ties flip, most often on
    prose. The probe stays conditioned on the path.
- **The walk** (tree_study.py) steps through the anchors as spec_loop does.
  - Step 0 is one row.
  - A step that accepts a drafts emits a + 1 tokens, and the next anchor is A + 1 + a.
  - λ is the rate after the step two back, since one step is queued ahead.
  - Acceptance is exact for any tree given as rank tuples. A node (k_1, …, k_j) is accepted when the
    path's ranks start with it.
  - Per-anchor trees choose nodes by a score. A node whose parent is off the path is scored with its
    depth's distribution from the path. That approximation only decides which nodes compete for the
    rows.

Today's gate, replayed over the records, reproduces the bench (the engine's report in brackets):

| | tok/step | drafts verified per step | steps verifying 0, 1, 2, 3, 4, 5 drafts | tok/s by the cost model |
|---|---|---|---|---|
| prose | 1.82 (1.81) | 1.47 (1.43) | 27, 122, 102, 24, 3, 0 (29, 131, 97, 21, 4, 0) | 69.6 (69.5) |
| code | 3.07 (3.17) | 3.01 (3.03) | 1, 29, 38, 30, 29, 37 (2, 25, 42, 29, 27, 36) | 98.8 (102.1) |
| agentic | 3.11 (3.09) | 2.89 (2.91) | 1, 33, 43, 28, 17, 40 (1, 32, 46, 27, 17, 42) | 101.3 (99.9) |
| JSON | 4.92 (4.90) | 4.34 (4.43) | 0, 9, 4, 4, 8, 78 (0, 8, 3, 5, 8, 80) | 139.7 (136.3) |

**Where the path's token ranks.** Each row is conditioned on the path above that depth being in the
top 8:

| | depth | 0 | 1 | 2 | 3 | 4–7 | ≥ 8 |
|---|---|---|---|---|---|---|---|
| prose | 1 | 0.569 | 0.111 | 0.059 | 0.026 | 0.077 | 0.158 |
| | 2 | 0.444 | 0.124 | 0.054 | 0.052 | 0.066 | 0.261 |
| code | 1 | 0.830 | 0.081 | 0.028 | 0.022 | 0.022 | 0.018 |
| | 2 | 0.757 | 0.109 | 0.034 | 0.016 | 0.034 | 0.050 |
| agentic | 1 | 0.822 | 0.095 | 0.026 | 0.012 | 0.022 | 0.024 |
| | 2 | 0.741 | 0.103 | 0.049 | 0.016 | 0.036 | 0.055 |
| JSON | 1 | 0.957 | 0.024 | 0.004 | 0.004 | 0.006 | 0.006 |
| | 2 | 0.942 | 0.024 | 0.006 | 0.006 | 0.012 | 0.010 |

- A rank-1 sibling catches about one anchor in nine on prose, and one in twelve on code.
- On prose, rank 0 falls to 0.29 by depth 5, and 0.16–0.45 of the true tokens are not in the top 8 at
  all.
- The drafter's own top-1 probability overstates prose. At depth 1, p ≥ 0.8 is right 82 % of the time,
  and p 0.6–0.8 is right 40 %. The confidence head is calibrated: a mean of 0.567 against 0.569
  accepted.

**The siblings' experts.** Distinct routed experts per layer, the mean over 40 layers and 32
samples:

| | 1–4 chain rows | 1–4 depth-1 siblings | added to 4 chain rows by the 1st, 2nd, 3rd sibling | a sibling row, ms |
|---|---|---|---|---|
| prose | 6.0 9.8 13.0 15.7 | 6.0 9.4 11.8 14.2 | 2.30 1.87 2.06 | 2.74, 2.43, 2.57 |
| code | 6.0 10.0 13.3 16.3 | 6.0 9.5 12.3 14.6 | 2.43 2.18 1.88 | 2.84, 2.66, 2.44 |
| agentic | 6.0 10.0 13.5 16.8 | 6.0 9.5 12.2 14.4 | 2.30 2.09 1.77 | 2.74, 2.59, 2.36 |
| JSON | 6.0 10.0 13.1 16.3 | 6.0 10.0 12.8 15.1 | 3.31 2.34 2.04 | 3.46, 2.77, 2.55 |

Siblings share more experts with each other than consecutive chain rows do. Each one still routes
about two per layer that the chain's 4 rows do not.

**Costs** (ms of a step, the draft pass included):
- **The chain.** The measured spec step on the bench prompts: 21.1, 24.5, 28.0, 31.2, 34.5 and 37.3
  at 1–6 rows.
- **A further node of a depth.** It costs b + c × the experts it adds. b and c come from a fit over
  batching.md's 1k table and this job's engine bench (1k; k = 1, 6, 7, 8; consecutive, unrelated and
  rotated rows): T = 18.64 + 1.10 (R − 1) + 0.713 (U − 6), over 30 steps with rms 1.05.
- **9–16 rows.** bench_chain's 16-row kernels cost +1.8 at 9 rows, +2.6–2.7 at 12 and +3.8–4.2 at
  16, over a fit of its own ≤ 8-row points. The mean, +3.09 ms, is charged above 8 rows. The runner's
  half of B4 is not built.
- **Drafting.** One pass for any shape. A tree adds +0.15 ms (a rough estimate).

**Acceptance by shape.** E[a] is the drafts accepted per step over every anchor, with the gate off.
The tree is the best of three kinds:
- static trees from the other three types' rank-prefix counts;
- per-anchor trees by the drafter's probability (EAGLE-2-like);
- per-anchor trees by the confidence head.

A type's own static tree does less well than the per-anchor trees.

| | chain, 5 drafts (6 rows) | tree ≤ 6 rows | ≤ 8 | ≤ 12 | ≤ 16 | today's gated chain (rows) |
|---|---|---|---|---|---|---|
| prose | 0.99 | 1.20 | 1.31 | 1.45 | 1.57 | 0.82 (2.47) |
| code | 2.58 | 2.71 | 2.93 | 3.18 | 3.33 | 2.07 (4.01) |
| agentic | 2.63 | 2.78 | 3.04 | 3.34 | 3.47 | 2.11 (3.89) |
| JSON | 4.23 | 4.24 | 4.37 | 4.53 | 4.61 | 3.92 (5.34) |

Without a gate, a tree verifies all its rows every step (tok/s for prose, code, agentic, JSON):
- at 6 rows, 61.3, 99.7, 98.9 and 136.3;
- at 16 rows, 39.4, 64.2, 66.8 and 80.3;
- today's chain, 69.6, 98.8, 101.3 and 139.7.

**Tok/s.** Three policies:
- today's chain;
- the chain under a per-node gate, which verifies a node while its score ≥ α·λ·(its row's ms). The
  score is the drafter's probability of the path to the node, or the confidence head's survival;
- trees under the same per-node gate, within 6, 8, 12 and 16 rows.

Each column is one score and one α for every type, chosen by the best sum of log gains over the four.
The brackets give tokens per step and rows.

| | today's chain | chain, per-node gate | tree ≤ 6 rows | ≤ 8 | ≤ 12 | ≤ 16 |
|---|---|---|---|---|---|---|
| prose | 69.6 (1.82, 2.47) | 69.5, −0.1 % (1.88, 2.76) | 69.7, +0.1 % (1.90, 2.84) | 69.6, +0.1 % | 69.6, +0.1 % | 69.6, +0.1 % |
| code | 98.8 (3.07, 4.01) | 102.6, +3.8 % (2.98, 3.41) | 102.5, +3.8 % (2.98, 3.41) | 102.4, +3.7 % | 102.4, +3.7 % | 102.4, +3.7 % |
| agentic | 101.3 (3.11, 3.89) | 99.6, −1.7 % (2.86, 3.32) | 99.9, −1.3 % (2.88, 3.34) | 99.7, −1.6 % | 99.7, −1.6 % | 99.7, −1.6 % |
| JSON | 139.7 (4.92, 5.34) | 139.8, 0.0 % (4.70, 4.84) | 139.8, 0.0 % (4.70, 4.84) | 139.8, 0.0 % | 139.8, 0.0 % | 139.8, 0.0 % |

- **The gate makes the gains, not the tree.**
  - The probability score with α = 2 shortens code's steps (3.41 rows against 4.01), for +3.8 %. The
    same setting costs agentic 1.7 %.
  - Under that gate, siblings add at most 0.4 %.
  - Gates measured on the ring moved by −1.4 to +0.9 % ("Gates by a cost per row count" above), so a
    ±4 % model estimate for a gate is a lead, not a result.
- **α tuned per type flatters the tree.** In-sample, the best tree gains +0.7 % on prose, +4.9 % on
  code, +2.5 % on agentic and 0 on JSON. The chain gate alone gains +0.2, +3.8, +1.3 and 0.
- **Siblings priced as chain rows (3.3 ms)** gain at most +0.8 % over the chain gate.
- **Siblings that route no new expert** cost only b = 1.1 ms. Over the best chain gate, trees then
  gain +5.9 % on prose, +3.1 % on code, +4.9 % on agentic and +0.9 % on JSON.

**Sensitivity.** Every further node of a depth costs a flat X ms, with α per type. The table gives the
best tree against today's chain, and in brackets against the best chain gate:

| X, ms | prose | code | agentic | JSON |
|---|---|---|---|---|
| 0 | +16.1 % (+15.8 %) | +13.5 % (+9.3 %) | +13.9 % (+12.4 %) | +1.9 % (+1.9 %) |
| 0.5 | +10.6 % (+10.4 %) | +10.7 % (+6.6 %) | +9.1 % (+7.6 %) | +1.6 % (+1.6 %) |
| 1.0 | +6.2 % (+6.0 %) | +7.3 % (+3.3 %) | +7.1 % (+5.8 %) | +1.1 % (+1.0 %) |
| 1.5 | +3.1 % (+2.9 %) | +6.5 % (+2.6 %) | +5.8 % (+4.4 %) | +1.2 % (+1.2 %) |
| 2.0 | +1.7 % (+1.5 %) | +6.8 % (+2.8 %) | +4.3 % (+2.9 %) | 0.0 % (−0.1 %) |
| 2.5 | +1.2 % (+1.0 %) | +5.2 % (+1.3 %) | +3.0 % (+1.7 %) | 0.0 % (0.0 %) |
| 3.0 | 0.0 % (−0.2 %) | +5.0 % (+1.1 %) | +2.0 % (+0.7 %) | 0.0 % (0.0 %) |
| free | +22.3 % (+22.0 %) | +23.6 % (+19.0 %) | +24.2 % (+22.6 %) | +8.3 % (+8.2 %) |

"free" also drops the 16-row overhead and the tree's drafting cost. A row costs at least b = 1.1 ms,
so 10 % on both prose and code is out of reach. Code's +5 % at 3 ms comes from the gate, as above.

**What it would take to build.** This is not proposed, given the verdict. The sizes are rough
estimates.
- **Attention** (the attention agent), the largest part:
  - Tree masks. A row attends to the committed window and its own ancestors in the step, not to its
    siblings.
  - Siblings share a position, so the step's rows need KV slots of their own rather than
    position & 255 in the window ring. sparse_attn then needs a per-row ancestor mask over those
    slots; WinBounds gives only a range today.
  - A compressor group that completes inside the tree differs per branch. So do the compressor's
    carries, its emitted entries and the index keys, and later rows read them.
  - Committing the accepted branch's KV, carries, entries and keys.
- **Runner:**
  - The tree in SpecState (tokens, parents, depths), and the per-node gate in k_spec_next.
  - The judge: from the step's first row, follow the child whose token equals the node's sample.
    The next bonus is the sample at the deepest accepted node. Sampled decoding stays exact as well,
    since the plain sampler's draw is fixed by the seed and the position.
  - Committing the accepted branch.
  - The drafter's accept() of a branch, where today it takes a prefix of rows.
  - Engram n-gram histories per branch (engram_prepare and engram_late).
  - Rows above 8 (B4's runner half).
- **Drafter** (the kernels agent), small:
  - The per-parent Markov bias, top-k and confidence, as above.
  - The tree's output into SpecState.
  - About +0.1–0.2 ms a pass.

**What would move the verdict.**
- A drafter whose top 1 is right more often, since the ranks cap even free trees at +22 %.
- A verify row well under 1 ms.
- Neither is in reach here. The probability gate on the chain (code +3.8 %, agentic −1.7 % in the
  model) is a separate and cheap lead. It would need a ring A/B.

**To rerun** (kernels-tree 154f0c1; on master since 631cf3c, measurement only, off by default):
1. Generate the path: `--mode generate --golden P --prompt N --gen 512 --big` on each bench prompt.
2. Make a golden directory of the prompt and those tokens.
3. Run the probe: `--mode spectf --golden G --prompt N --big --draft-fp8 --tree-probe tp_<type>.jsonl
   --tree-sib 16`. Each type takes about 70 s.
4. Fit the costs: `tree_costs.py costs.json ENGINE_BENCH_LOG BENCH_CHAIN_LOGS…`.
5. Analyse: `tree_study.py DIR --costs costs.json [--sweep | --tables | --sib-new chain|zero]`.

This study's records, its costs.json and its engine bench log are in rank 0's /data/dsgb10/kernels/tree.

The hooks are bench-only and default-off:
- ops.h: StepIn.probe, SampleOut.probe;
- model.h: SpecCfg.probe, Options.tree_probe;
- spec.cu: the probe and k_probe_out, behind opt.tree_probe;
- cmd.h and cmd.cpp: Cmd.spec_probe;
- main.cpp: the spectf flags.

### Prompt-lookup drafts: a feasibility study (2026-10-12)

The question: would prompt-lookup drafts (PLD), alone or added to today's chain, beat the chain by 10 % or
more on agentic traces, with no bench type slower? PLD proposes the tokens that followed the latest earlier
occurrence of the context's last n tokens (n = 8 down to 2). Verification is exact, so any proposal keeps the
outputs.

This was measurement and analysis only. The hooks are bench-only and default-off, all in main.cpp
(they stay on the branch kernels-pld, 49c60b8, not merged: PLD is not being built), and serving is unchanged. The analysis is engine/dspark/ref/pld_study.py.
pld_inputs.py makes its inputs, and pld_match.cpp times a host-side matcher.

**Verdict: no. The long copies are there, but no gate finds where they end. PLD is dropped, and so is
wiring 16 rows for it.**
- **The edit traces copy a lot.** These are the runner's 54 turns of read/edit/write calls.
  - 55 % of their output is a right PLD proposal, and 30 % lies in right runs of 16 tokens or more.
  - Edit calls: 62 % and 36 %. Write calls: 61 % and 41 %.
- **A perfect gate would pass; no real gate does.** On the edit traces:

  | | 8 rows | 16 rows |
  |---|---|---|
  | the chain, then exactly PLD's right continuation (bound) | +8.9 % | +17.8 % |
  | the best gate | +4.3 % | +5.0 % |
  | lengths chosen in hindsight, per match length and per whether the chain verified all 5 drafts | +5.9 % | +8.3 % |

  - On edit and write calls alone, the best gate gains +6.7 % against a bound of +22.1 %.
  - So the runs exist, and the shortfall is the gate's. Nothing cheap in the context says where a copy ends,
    which is where the model's edit begins.
  - Even inside a copy matched 32–63 tokens back, a 15-token proposal is wholly right only 48 % of the time.
- **Today's chain already takes most of a copy.**
  - On the edit traces the chain verifies all 5 drafts on 72 % of steps and runs at 128.8 tok/s.
  - PLD can only add rows 7–16. Each costs 2.5–3.1 ms, and the 9th adds a further 3.09 ms.
  - A 16-row step must emit 10.6 tokens to match a 6-row step that emits 6.
- **Cheaper rows are not enough.**
  - Without the 16-row overhead, the best gate gains +6.1 % at 16 rows.
  - With rows at half today's cost as well, it gains +9.6 %.
  - It passes only if a row's experts cost nothing (+10.6 %).
- **The bench and the transcript gain less.**
  - The best gate gains +1.9–2.2 % on the transcript and +3.2–3.6 % on its tool-call turns.
  - It moves the bench types by −0.3 to +0.7 %.
  - The bound is +8.9 % on the transcript at 16 rows.
- **The 16-row rule.** The rule was to drop PLD and the 16-row wiring if even the bound stayed under 10 %.
  The bound clears it (+17.8 %), so the long runs are real. The gate's ceiling fails the bar, though: +8.3 %,
  even when tuned in hindsight.
  - Without PLD, nothing needs 16 rows of one session. The chain fills 6, and trees did no better (the tree
    study above), so kMaxSessRows stays 8.
  - Whether kMaxRows goes to 16 for steps of several sessions is a separate question.

**PLD here.**
- **The matcher.** The longest suffix match, n = 8 down to 2, over the prompt and the output so far. The
  proposal is up to 15 tokens after the match's latest occurrence.
- **The escaped matcher.**
  - A tool call's JSON-valued parameters (DSML `string="false"`, such as an edit's oldText) carry file text
    JSON-escaped. The tool results carry it raw.
  - So each `<tool_result>` is also JSON-escaped (ensure_ascii off), tokenized by dsfront and indexed as a
    copy after the result. No match crosses a copy's boundaries.
  - An anchor inside a JSON-valued parameter searches the history and the copies. Any other anchor searches
    the history alone.
  - Searching both everywhere let the escaped copies win "latest" in raw contexts, which made write calls
    worse.
  - The escaped matcher raises the edit calls' copied share from 58 % to 62 %, and their share in runs of
    16 or more from 29 % to 36 %.

**The policies.** Each verifies rows as spec_loop does: a step accepts the rows its samples confirm.
- **PLD alone,** with no draft pass:
  - the whole proposal;
  - gated, where q(n) is P(the first token is right) and c is P(each further one is right), both EMAs of
    earlier steps, and a row is verified while the product ≥ λ·ΔC, as today's gate does;
  - exactly its right run (a bound).
- **PLD or the chain:**
  - PLD when its match has n ≥ 4 (or 8), else the chain. A PLD step skips the draft pass, though accept still
    runs;
  - per step, whichever expects more tokens per ms.
- **PLD, then the chain's drafts** beyond it where they agree.
- **The chain, then PLD's continuation.** Today's gated chain, then PLD's proposal looked up after the
  verified drafts (the history plus d1..dk), up to the row cap. It comes in four kinds:
  - ungated;
  - gated, while survival[k] · q(n) · c^j ≥ λ·ΔC;
  - value-gated, taking the length that maximises Σp − λ·(ΔC a row, plus 3.09 at the 9th row), with q keyed
    by n, and optionally also by whether the drafts equal PLD's own proposal;
  - keyed by the match's uncapped length, with q and c by n below 8, and above by how far back the match
    goes (8, 16, 32, 64 or 128 tokens and more).
- **Bounds:**
  - the chain, then exactly PLD's right continuation when the chain's drafts are all right;
  - a perfect gate on the drafts alone;
  - a perfect gate on both, which per step takes the best of the right drafts, those plus PLD's right
    continuation, and PLD's right run alone.
- **Hindsight.** A fixed extension length per key, where the key is the match key and whether the chain
  verified all 5 drafts. The lengths are chosen on the edit traces themselves by coordinate descent, so this
  is in-sample and optimistic.

**Method.**
- **The outputs** (12,274 tokens, in contexts of up to 29k):
  - **The bench:** the target's greedy 512 tokens from each bench prompt, the tree study's.
  - **The transcript:** golden/long.tokens.json's 14 assistant turns, each regenerated by the target from
    that turn's context. This was pass 1, with today's drafts: greedy, ended by EOS, at most 2048 tokens.
    12 turns hold a tool call.
  - **The edit traces:** the runner's editbench.py r1, in rank 0's /data/dsgb10/bench/edit/r1.
    - 54 turns of pi-style read/edit/write calls on scratch copies of this repo's files, greedy.
    - Thinking is off on 39 turns and max on 15.
    - Each turn comes with its served decode rate and steps histogram.
- **The records** (passes 2 and 3: spectf --seq-list, the FP8 drafter head as served):
  - Every output teacher-forced, one 1-row spec step per position, records the drafts d1..d5 and their
    survival at each anchor.
  - `--route-file` records each position's own routed experts (40 layers × top 6).
  - It also routes, in eager steps at the true context, the drafts' rows at every 8th anchor and a third of
    the anchors whose PLD proposal holds a wrong token (up to 14 rows, `--props`). These fit β, the experts
    a rejected row adds: drafts −0.02 to +0.05, PLD −0.02 to +0.19.
  - With pass 1, that is 12 ring steps of 1–3 min, 24 min in all.
- **The walk** is the tree study's.
  - λ lags two steps, and acceptance is exact.
  - A step's ms is T = c0 + 1.10 (R − 1) + 0.713 (U − 6), plus 3.09 above 8 rows, plus 2.55 for the draft
    pass (0.15 for accept alone).
  - U is the exact union of the step's own rows' experts per layer, plus β per rejected row.
  - c0 is offset per output so that today's chain matches the measured rate:
    - bench.py's decode for the bench (−0.46 to −1.62 ms);
    - pass 1's GPU ms by row count for the transcript's turns (−0.8 to −1.9 ms);
    - one offset for the edit traces from their served decode (+0.07 ms).
- **Checks.**
  - The records and routes cover every position of every output.
  - The bench records equal the tree study's.
  - The replayed chain's steps match the engine's:
    - per turn for the transcript (turn 10: verifying 0–5 drafts, replay 0, 38, 72, 55, 30, 40 against the
      engine's 0, 38, 75, 55, 26, 38);
    - in sum for the edit traces (replay 3, 24, 78, 104, 140, 1042 against served 1, 22, 70, 112, 124,
      1048).

**Q1: how much PLD would copy.**
- The first column, P(next k right), is over every anchor with any match.
- "Copied" is the share of the output that a PLD-only walk gets from PLD, at a proposal cap of 7, 15 or 63
  tokens.
- "In runs" is the share inside maximal right runs of ≥ 8 or ≥ 16 tokens.
- The raw matcher has no escaped copies.

| | output | P(next 1, 2, 4, 8, 15 right), % | copied at caps 7, 15, 63, % | in runs ≥ 8, ≥ 16, % | tok/step, PLD alone | raw matcher: copied, in runs ≥ 16, % |
|---|---|---|---|---|---|---|
| prose | 512 | 1.2, 0.0, 0.0, 0.0, 0.0 | 1.2, 1.2, 1.2 | 0.0, 0.0 | 1.01 | 1.2, 0.0 |
| code | 512 | 8.6, 5.1, 2.2, 0.2, 0.0 | 8.4, 8.6, 8.6 | 1.6, 0.0 | 1.10 | 8.6, 0.0 |
| agentic | 512 | 6.5, 2.5, 0.2, 0.0, 0.0 | 6.4, 6.4, 6.4 | 0.0, 0.0 | 1.07 | 6.4, 0.0 |
| JSON | 512 | 42.5, 32.9, 23.7, 12.5, 2.7 | 39.5, 41.0, 42.4 | 23.4, 21.9 | 1.70 | 41.0, 21.9 |
| the transcript | 3238 | 36.6, 28.2, 19.4, 11.6, 5.5 | 34.4, 35.9, 36.4 | 19.6, 11.6 | 1.56 | 35.9, 11.6 |
| its tool-call turns | 2067 | 43.3, 34.6, 25.0, 15.7, 7.6 | 40.4, 42.3, 43.1 | 26.1, 15.1 | 1.74 | 42.3, 15.1 |
| its other turns | 1171 | 24.7, 17.0, 9.4, 4.4, 1.9 | 23.8, 24.4, 24.7 | 8.0, 5.4 | 1.33 | 24.4, 5.4 |
| edit traces | 6988 | 56.8, 47.9, 36.7, 25.1, 15.7 | 52.1, 55.0, 56.4 | 39.7, 29.6 | 2.25 | 53.2, 26.6 |
| their edit calls | 3074 | 64.3, 54.7, 43.1, 30.3, 19.3 | 58.9, 62.2, 64.0 | 47.1, 36.0 | 2.65 | 58.2, 29.4 |
| their write calls | 1654 | 63.2, 56.8, 47.3, 34.9, 22.3 | 57.3, 61.2, 63.1 | 54.0, 40.8 | 2.58 | 61.2, 40.8 |
| their read calls | 1324 | 51.8, 39.8, 24.6, 12.3, 6.2 | 48.3, 50.4, 50.9 | 23.0, 15.3 | 2.07 | 50.4, 15.3 |
| their prose answers | 936 | 27.7, 20.8, 13.4, 8.2, 5.5 | 25.6, 26.8, 27.2 | 14.0, 8.4 | 1.40 | 26.8, 8.4 |
| thinking off | 4855 | 57.8, 48.5, 36.6, 24.4, 14.6 | 53.1, 56.0, 57.3 | 39.6, 28.4 | 2.30 | 54.0, 25.2 |
| thinking max | 2133 | 54.6, 46.6, 36.8, 26.5, 18.3 | 49.7, 52.7, 54.2 | 40.0, 32.2 | 2.13 | 51.5, 29.9 |

The PLD-only walk's steps by the length of their right run (cap 15):

| | 0 | 1 | 2–3 | 4–7 | 8–15 |
|---|---|---|---|---|---|
| prose | 499 | 6 | 0 | 0 | 0 |
| code | 449 | 8 | 6 | 3 | 1 |
| agentic | 458 | 12 | 7 | 1 | 0 |
| JSON | 252 | 19 | 13 | 9 | 8 |
| the transcript | 1788 | 104 | 80 | 56 | 46 |
| its tool-call turns | 998 | 66 | 50 | 38 | 39 |
| its other turns | 790 | 38 | 30 | 18 | 7 |
| edit traces | 2401 | 172 | 207 | 128 | 204 |
| their edit calls | 818 | 83 | 98 | 59 | 103 |
| their write calls | 505 | 22 | 32 | 19 | 63 |
| their read calls | 477 | 45 | 54 | 38 | 27 |
| their prose answers | 601 | 22 | 23 | 12 | 11 |
| thinking off | 1601 | 116 | 155 | 97 | 142 |
| thinking max | 800 | 56 | 52 | 31 | 62 |

- The runs are bimodal: most PLD-alone steps get nothing, and a copy, once found, often runs long. On the
  edit traces, 204 steps take 8–15 tokens against 128 that take 4–7.
- **The match length helps, but not enough to see a copy's end.** By the length of the match behind an edit
  trace anchor's proposal (n below 8, else how far back the match goes; pld_study.py --why):

  | match | anchors | P(first right) | mean right run (of up to 15) | whole proposal right |
  |---|---|---|---|---|
  | 2 | 15 % | 0.50 | 2.2 | 0.05 |
  | 3 | 11 % | 0.73 | 3.5 | 0.10 |
  | 4 | 9 % | 0.68 | 3.8 | 0.12 |
  | 5 | 7 % | 0.70 | 4.8 | 0.16 |
  | 6 | 5 % | 0.88 | 6.2 | 0.21 |
  | 7 | 5 % | 0.81 | 6.0 | 0.21 |
  | 8–15 | 23 % | 0.90 | 7.5 | 0.24 |
  | 16–31 | 15 % | 0.93 | 8.5 | 0.38 |
  | 32–63 | 8 % | 0.96 | 10.6 | 0.48 |
  | 64 and more | 2 % | 0.93 | 9.4 | 0.40 |

**Q2: tok/s, the edit traces** (% over today's chain, the policies as above):

| 8 rows at most | edit traces | edit, write | read, prose | thinking off | thinking max |
|---|---|---|---|---|---|
| today's chain (tok/s) | 128.8 | 134.8 | 117.7 | 131.5 | 122.8 |
| PLD alone, the whole proposal | −43.6 % | −41.4 % | −47.1 % | −44.6 % | −41.2 % |
| PLD alone, gated | −42.2 % | −39.5 % | −46.5 % | −43.1 % | −40.1 % |
| PLD alone, exactly its right run (bound) | −28.9 % | −25.6 % | −34.1 % | −29.4 % | −27.7 % |
| PLD when n ≥ 4, else the chain; no draft pass on PLD steps | −4.6 % | −3.2 % | −7.0 % | −5.9 % | −1.6 % |
| the same with n ≥ 8 | −0.3 % | +0.0 % | −0.9 % | −0.7 % | +0.6 % |
| PLD, then the chain's drafts where they agree | −12.9 % | −10.7 % | −16.7 % | −13.7 % | −11.0 % |
| per step, PLD or the chain by expected tokens per ms | −0.6 % | −0.2 % | −1.3 % | −0.3 % | −1.3 % |
| the chain, then PLD's continuation after its drafts, ungated | +4.2 % | +6.2 % | +0.7 % | +4.9 % | +2.7 % |
| the chain, then PLD's continuation, gated (n ≥ 2) | +4.3 % | +5.5 % | +2.2 % | +4.9 % | +3.1 % |
| the same, n ≥ 8 | +3.6 % | +4.6 % | +1.7 % | +3.6 % | +3.5 % |
| the same, value-gated (n = 8) | +3.6 % | +4.6 % | +1.7 % | +3.6 % | +3.5 % |
| the same, q also by agreement | +3.5 % | +4.6 % | +1.7 % | +3.5 % | +3.6 % |
| the same, q and c by the match's uncapped length (n ≥ 2) | +4.1 % | +5.1 % | +2.1 % | +4.5 % | +3.0 % |
| the same, n ≥ 8 | +3.3 % | +4.2 % | +1.7 % | +3.3 % | +3.3 % |
| bound: the chain, then exactly PLD's right continuation | +8.9 % | +10.5 % | +5.9 % | +9.4 % | +7.8 % |
| bound: a perfect gate on the drafts alone | +5.8 % | +4.1 % | +9.2 % | +4.9 % | +7.9 % |
| bound: a perfect gate on the drafts and PLD | +18.7 % | +18.7 % | +18.5 % | +17.6 % | +20.9 % |

| 16 rows at most | edit traces | edit, write | read, prose | thinking off | thinking max |
|---|---|---|---|---|---|
| today's chain (tok/s) | 128.8 | 134.8 | 117.7 | 131.5 | 122.8 |
| PLD alone, the whole proposal | −53.2 % | −51.1 % | −56.6 % | −54.4 % | −50.6 % |
| PLD alone, gated | −44.0 % | −40.8 % | −49.0 % | −44.8 % | −42.0 % |
| PLD alone, exactly its right run (bound) | −25.9 % | −21.3 % | −33.0 % | −26.4 % | −24.7 % |
| PLD when n ≥ 4, else the chain; no draft pass on PLD steps | −5.4 % | −3.5 % | −8.8 % | −7.2 % | −1.3 % |
| the same with n ≥ 8 | −0.9 % | −0.4 % | −1.7 % | −1.3 % | −0.0 % |
| PLD, then the chain's drafts where they agree | −14.4 % | −11.4 % | −19.4 % | −15.1 % | −12.9 % |
| per step, PLD or the chain by expected tokens per ms | −0.5 % | −0.2 % | −1.1 % | −0.3 % | −1.0 % |
| the chain, then PLD's continuation after its drafts, ungated | −7.7 % | −3.3 % | −14.8 % | −7.8 % | −7.7 % |
| the chain, then PLD's continuation, gated (n ≥ 2) | +0.5 % | +3.9 % | −5.1 % | +0.1 % | +1.4 % |
| the same, n ≥ 8 | +4.8 % | +6.5 % | +1.7 % | +5.1 % | +4.0 % |
| the same, value-gated (n = 8) | +5.0 % | +6.7 % | +2.0 % | +5.4 % | +4.2 % |
| the same, q also by agreement | +4.7 % | +6.3 % | +2.0 % | +5.2 % | +3.8 % |
| the same, q and c by the match's uncapped length (n ≥ 2) | +0.3 % | +4.1 % | −6.0 % | −0.3 % | +1.4 % |
| the same, n ≥ 8 | +4.0 % | +5.3 % | +1.8 % | +4.1 % | +3.8 % |
| bound: the chain, then exactly PLD's right continuation | +17.8 % | +22.1 % | +10.6 % | +18.6 % | +16.0 % |
| bound: a perfect gate on the drafts alone | +5.8 % | +4.1 % | +9.2 % | +4.9 % | +7.9 % |
| bound: a perfect gate on the drafts and PLD | +28.1 % | +30.7 % | +23.5 % | +26.7 % | +31.0 % |

- **Of the eight gated extensions, the best gains +5.0 % at 16 rows** (value-gated, n = 8) and +4.3 % at 8
  (gated, n ≥ 2).
- **Lengths chosen in hindsight** (pld_study.py --hindsight) gain +5.9 % at 8 rows and +8.3 % at 16. At 16
  rows they extend by 6–14 tokens after a match of 8 or more.
- **By task** (thinking, task; the served rate in brackets):

| thinking, task | turns | output | copied (cap 15) | in runs ≥ 16 | today's chain, tok/s | gated (n ≥ 2), 8 rows | gated (n ≥ 8), 16 rows | bound, 16 rows |
|---|---|---|---|---|---|---|---|---|
| off, rename_helper | 3 | 187 | 47 % | 11 % | 127.6 (124.0) | +4.4 % | +4.9 % | +15.7 % |
| off, add_param | 3 | 210 | 56 % | 33 % | 132.0 (133.6) | +6.6 % | +6.7 % | +17.0 % |
| off, fix_comment | 3 | 233 | 48 % | 25 % | 129.5 (125.9) | +3.7 % | +4.5 % | +15.0 % |
| off, env_override | 3 | 355 | 55 % | 41 % | 131.1 (130.2) | +5.8 % | +7.5 % | +19.8 % |
| off, usage_error | 4 | 298 | 48 % | 30 % | 122.9 (121.5) | +2.1 % | +4.0 % | +12.3 % |
| off, convert_loop | 3 | 313 | 64 % | 45 % | 131.3 (132.9) | +7.6 % | +10.3 % | +20.9 % |
| off, rust_consts | 3 | 455 | 54 % | 25 % | 136.2 (134.7) | +4.4 % | +7.7 % | +15.5 % |
| off, extract_function | 4 | 445 | 62 % | 39 % | 134.3 (129.5) | +2.4 % | +4.0 % | +21.9 % |
| off, docstrings | 3 | 383 | 52 % | 7 % | 126.6 (130.7) | +3.8 % | +1.6 % | +13.5 % |
| off, two_file_rename | 4 | 535 | 60 % | 10 % | 134.6 (137.0) | +6.0 % | +3.2 % | +21.5 % |
| off, rewrite_small | 3 | 476 | 52 % | 32 % | 130.8 (129.1) | +4.8 % | +4.4 % | +15.1 % |
| off, rewrite_medium | 3 | 965 | 60 % | 35 % | 133.1 (136.7) | +6.1 % | +5.3 % | +24.1 % |
| max, rename_helper | 4 | 473 | 35 % | 9 % | 107.7 (106.3) | +1.9 % | +1.0 % | +8.6 % |
| max, fix_comment | 3 | 355 | 51 % | 34 % | 122.4 (122.8) | +1.9 % | +1.9 % | +14.4 % |
| max, convert_loop | 5 | 768 | 66 % | 44 % | 129.3 (130.6) | +5.2 % | +7.8 % | +24.0 % |
| max, rewrite_small | 3 | 537 | 50 % | 35 % | 129.9 (128.1) | +2.2 % | +3.5 % | +14.9 % |

**Q2: tok/s, the bench and the transcript** (the full tables: pld_study.py --tables):

| | prose | code | agentic | JSON | transcript | its tool-call turns | its other turns |
|---|---|---|---|---|---|---|---|
| today's chain (tok/s) | 69.4 | 101.9 | 99.8 | 136.0 | 104.3 | 111.0 | 94.3 |
| the chain, then PLD's continuation, gated (n ≥ 2), 8 rows | −1.2 % | −0.6 % | −0.1 % | +3.7 % | +1.6 % | +2.5 % | +0.3 % |
| the same, n ≥ 8, 16 rows | +0.0 % | −0.3 % | +0.0 % | +0.7 % | +1.9 % | +3.2 % | +0.1 % |
| the same, value-gated (n = 8, q also by agreement), 16 rows | +0.0 % | −0.1 % | +0.0 % | −0.4 % | +2.2 % | +3.6 % | +0.2 % |
| PLD when n ≥ 4, else the chain; no draft pass on PLD steps | +0.0 % | −1.7 % | −0.1 % | −6.2 % | −3.9 % | −4.9 % | −2.2 % |
| bound: the chain, then exactly PLD's right continuation, 8 rows | +0.0 % | +1.0 % | +1.5 % | +8.1 % | +5.5 % | +6.1 % | +4.7 % |
| the same, 16 rows | +0.0 % | +1.0 % | +1.5 % | +14.1 % | +8.9 % | +11.0 % | +6.0 % |
| bound: a perfect gate on the drafts alone | +19.9 % | +20.7 % | +17.4 % | +5.6 % | +15.7 % | +13.3 % | +19.6 % |
| bound: a perfect gate on the drafts and PLD, 16 rows | +24.5 % | +25.3 % | +19.1 % | +21.9 % | +29.6 % | +30.2 % | +28.5 % |

- Short tool-call turns gain the most: +36.6 % at 16 rows on turn 7, which has 48 tokens, 90 % copied. The
  long turns (394–668 tokens) gain under 3 %.
- PLD on its own loses everywhere (−20 to −58 %). The drafts beat it on most anchors, and a PLD-only step
  pays the full step for one token whenever PLD has nothing.

**Why the gates fall short** (pld_study.py --why). Walks of the edit traces at 16 rows:

| | tok/step | rows/step | steps verifying all 5 drafts | steps extended by PLD | PLD rows/step: accepted, rejected | steps past 8 rows | tok/s |
|---|---|---|---|---|---|---|---|
| today's chain | 4.84 | 5.33 | 72 % | | | | 128.8 |
| gated (n ≥ 8) | 5.52 | 6.37 | 70 % | 18 % | 0.76, 0.34 | 16 % | +4.8 % |
| bound | 6.49 | 7.03 | 67 % | 35 % | 1.87, 0 | 23 % | +17.8 % |

- **The gate extends half as often as the bound, and a third of its PLD rows are wrong.** A wrong row costs as
  much as a right one.
- **The cost of a row past the 6th.** Each of rows 7 to 16 adds 2.9 down to 1.9 distinct experts per layer,
  so it costs 1.1 + 0.713 × that, 3.1 down to 2.5 ms. The 9th row also brings the 3.09 ms of a step past 8
  rows.
- **A step with every row accepted:**
  - 6 rows: 40.0 ms, 150 tok/s;
  - 8 rows: 46.1 ms, 174 tok/s;
  - 9 rows: 52.1 ms, 173 tok/s;
  - 16 rows: 70.7 ms, 226 tok/s.
  
  So 16 rows pay only when PLD's 10 extra rows yield at least 4.6 tokens.
- **Cheaper rows.** With the gate's ΔC set to match, at 16 rows:

  | | the best gate | the bound |
  |---|---|---|
  | as measured | +5.0 % | +17.8 % |
  | without the 16-row overhead | +6.1 % | +19.8 % |
  | without it, rows at half cost | +9.6 % | +24.1 % |
  | without it, a row's experts free | +10.6 % | +23.9 % |

**Q3: the cost of proposing, and where it would run.**
- **On the host it is cheap.** pld_match.cpp's index holds hash tables for n = 2..8: 86 MiB at 250k tokens of
  history, built at 200–370 ns a token. A step appends 1–16 tokens and looks up once.
  - Median: 0.8–1.1 µs.
  - p99: 3–6 µs.
  
  These are two runs on rank 0 with production up.
- **But the host is on the wrong side of the step.**
  - Step i + 1's rows are fixed on the device by k_spec_next when step i ends. k_spec_in reads them from
    SpecState.
  - The host sees step i's tokens only after that, when it submits engram_late, which the GPU awaits at
    layer 1.
  - The row count and the tokens are needed from the first layer on. So a host proposal would make every
    step wait for the host.
- **It belongs on the device.**
  - The index update would sit beside accept, adding the emitted tokens' n-grams, off the critical path.
  - The lookup would sit in k_spec_next, after the drafts and their gate: a hash probe per n, a few µs.
  - Each session holds its own index.
- **The draft pass is not skipped.**
  - The policies that skip it on PLD steps lose: −0.3 to −5.4 % on the edit traces (n ≥ 8 or 4), −1.4 to
    −4.4 % on the transcript, and up to −6.2 % on JSON.
  - Every policy that gains keeps the pass, since PLD extends the chain.
  - accept() (0.15 ms, the drafter's window) runs every step either way.

**What it would take to build.** This is not proposed, given the verdict.
- **Kernels (engine/dspark):**
  - the device index and its update after accept;
  - the escaped copies of tool results, which the front end would tokenize, since the engine has no
    tokenizer;
  - the mode switch on DSML parameters;
  - the lookup.
- **Runner (spec.cu):**
  - k_spec_next's extension and its gate (q, c, λ as today);
  - SpecState's next_tok to 16 rows.
  - The judge is unchanged, since PLD's rows are verified as further chain rows.
- **16 rows a step:**
  - B4's attention half is merged (2a408dd), but kMaxRows and kMaxSessRows stay 8.
  - attn::decode_prep races at exactly 16 rows of one session (engine/attn).
  - accept() takes 8 rows a call, so it needs two calls.
  - The 3.09 ms above 8 rows comes from bench_chain's kernels and would need measuring on the built path.

**What would move the verdict.**
- **A predictor of where a copy ends.** Cheap features cap the gain at +8.3 % at 16 rows, even when tuned in
  hindsight. The drafter's survival does not reach past depth 5.
- **Rows past the 6th at well under half today's cost,** together with no 16-row overhead.
- **Sampling will not help.** The runner's sampled traces (bench/edit/r2) accept copies as greedy does (edit
  calls 95 % in both), and prose falls. PLD's case is the copies, so it carries over.

**To rerun** (kernels-pld 49c60b8, `main.cpp`'s flags on the engine binary; on master since 631cf3c):
1. Make pass 1's inputs: `pld_inputs.py turns DIR golden/long.tokens.json`.
2. Run pass 1: `--mode generate --loop --big --spec --draft-fp8 --golden DIR/long --seq-list DIR/gen_{a,b,c}.txt`,
   one ring step each.
3. Take in its logs: `pld_inputs.py pass1 DIR LOG…`.
4. Make the runs' inputs: `pld_inputs.py runs DIR --dsfront BIN --bench NAME=GOLDEN:PROMPT…
   --turns RUN:K-K… --edit INDEX.jsonl`.
   - The bench's golden directories are the tree study's.
   - `--dsfront` makes the escaped copies (esc.json) first, so the routed proposals are the escaped
     matcher's.
5. Run passes 2 and 3, one ring step per run, each run's output in DIR/tf_RUN.log:
   `--mode spectf --big --draft-fp8 --golden G --seq-list DIR/sl_RUN.txt --route-file DIR/rt_RUN.bin
   --route-every 8 --chain-every 8 --props DIR/props_RUN.txt`. G is the list's first directory.
6. Analyse: `pld_study.py DIR --check --copy --cal --replay --policies --tables --hindsight --why` (a few
   minutes).
7. Time the host matcher: `pld_match TOKENS.bin 250000 20000`.

The study's data are in rank 0's /data/dsgb10/kernels/pld. That covers the transcript turns' golden
directories, every run's records and routes, and esc.json. The edit traces stay in the runner's
/data/dsgb10/bench/edit/r1.

The hooks are bench-only and default-off, all in main.cpp:
- `--seq-list`: the list's sequences in one process, for spectf and generate --loop;
- `--route-file`, `--route-every` and `--chain-every`: the routed experts of eager steps;
- `--props`: the rows to route.

### The draft gate under sampling (2026-10-12)

The question: pi's requests decode at dsfront's defaults (T 1.0, top-p 0.95), and today's verify-length gate
was tuned under greedy decoding. Would another gate gain 2 % or more on the edit bench decoded that way,
pi's workload, with no bench type or catalog more than 2 % slower? A gate only picks how many drafts
a step verifies, so no gate can change an output.

This was measurement and analysis only, and serving is unchanged. The records come from a bench-only
recording build, runner-gate-rec 22ec1d9 (merged with the draft pass study, 631cf3c, measurement only and off by default): spectf's
`--tree-probe` now combines with `--seq-list`, and each record also holds the target's own top 20
logprobs. The analysis is engine/dspark/ref/gate_study.py, and gate_inputs.py makes its inputs.

**Verdict: no. The best candidate gains +1.1 % on the held-out repeats, and a gate that knew the
sampler's own probability of each draft would gain only +3.5 %. Nothing is built, and there was no
ring A/B to run.**
- **Under sampling the head overstates, but only where few drafts are.**
  - At depth 1, drafts with a confidence of 0.5–0.7 are accepted 45 % of the time, against a mean
    confidence of 63 %. For 0.7–0.8 it is 61 % against 76 %, and for 0.8–0.9, 78 % against 85 %.
  - Those buckets hold 8 % of the depth-1 drafts. 87 % have a confidence of 0.98 or more, and 99.9 %
    of those are accepted.
  - At depth 2 it overstates less (0.5–0.7: 52 % accepted against 62 %), and at depth 3 it is
    close (60 % against 61 %). The drafter's own probability overstates more: 0.70 and 0.75 in the
    same buckets.
- **No candidate comes near 2 %.** Each was trained on r2's repeat 0 and tested on the rest of r2.
  The table gives each at its best setting on repeat 0, as % over today's gate:

  | candidate (its setting) | repeat 0 (train) | the rest (test) | edit calls | write calls | read calls | prose | thinking max | r1, greedy |
  |---|---|---|---|---|---|---|---|---|
  | today's gate, tok/s | 126.3 | 126.7 | 134.9 | 130.9 | 128.0 | 97.3 | 113.1 | 128.8 |
  | today's rule, threshold ×1.5 | +0.6 % | −0.1 % | −0.3 % | +0.3 % | −0.6 % | +0.4 % | +1.0 % | −0.3 % |
  | `--spec-gate ratio` | +0.3 % | −0.0 % | +0.1 % | −0.4 % | −0.9 % | +1.0 % | +0.1 % | +0.5 % |
  | 1: the drafter's probability, threshold ×1.5 | +1.2 % | +1.1 % | +0.5 % | +0.8 % | +1.2 % | +2.7 % | +1.9 % | +0.3 % |
  | 1: the same, the tree study's rule (α = 2) | +1.5 % | +0.8 % | −0.0 % | +0.9 % | +1.3 % | +2.3 % | +1.3 % | −0.0 % |
  | 1: the same, by expected tokens per ms | +1.3 % | +0.7 % | +0.2 % | +0.7 % | +1.1 % | +1.8 % | +0.9 % | +0.3 % |
  | 2: learned per session (56 settings) | −0.1 to +0.5 % | −0.4 to +0.4 % | | | | | | −0.4 to +0.6 % |
  | 3: the head, recalibrated (threshold ×1.2) | +0.6 % | −0.1 % | −0.2 % | +0.3 % | −0.9 % | +0.1 % | +0.3 % | today's |
  | 3: the head and the drafter's probability, recalibrated (threshold ×1.0) | +1.3 % | +0.9 % | +0.5 % | +0.8 % | +0.7 % | +2.6 % | +1.6 % | today's |
  | 3: those and the target's confidence at the bonus (threshold ×1.2) | +1.3 % | +1.1 % | +0.3 % | +1.0 % | +1.0 % | +3.5 % | +2.0 % | today's |
  | bound: the sampler's probability of each draft | +4.8 % | +3.5 % | +2.1 % | +4.2 % | +2.7 % | +7.1 % | +6.1 % | +4.6 % |
  | bound: a perfect gate | +7.1 % | +6.4 % | +3.9 % | +4.9 % | +5.5 % | +17.8 % | +11.0 % | +5.8 % |

  - The kinds and thinking max are of the test set. r1 is the greedy edit bench, on which the
    recalibrated gates are today's by construction (they apply only to sampled requests).
  - The tree study's probability gate (α = 2, code +3.8 % and agentic −1.7 % there) gains +0.8 % on
    the sampled bench and nothing on the greedy one.
  - Nothing is worth a ring A/B. Gates measured on the ring moved by −1.4 to +0.9 % ("Gates by a cost
    per row count" above), so a modelled +1.1 % could not be confirmed by an A/B of two jobs, and it
    would stay under the bar if it were.
- **The gate decides only at the margin.**
  - Today's gate verifies 4.5 drafts a step, and 87 % of them are accepted. The bound verifies 4.2 and
    has 95 % accepted.
  - A draft row costs about 3.5 ms of a 37 ms step: 1.1 ms for the row and 0.71 ms for each of the
    experts it adds. It pays when its chance of being accepted is above about 0.45. Rows near that
    line are the only ones a gate can change, and each moves little.
- **The bound needs what only the verify computes.** The sampler's probability of a draft is the
  target's distribution at the draft's own position.
  - The only distribution of the target that a step has before its verify is the bonus's. It adds
    nothing: in the fit, the target's P(bonus) and its top P get weights of 0.06 or less at depths
    1–3, of opposite signs, next to 0.30–0.34 for the head and 0.57–0.60 for the drafter.
  - Learning acceptance from the session's own steps (candidate 2) moves the result by −0.4 to
    +0.4 %.
- **The gains are prose's and thinking's.**
  - The candidates that use the drafter's probability gain most where the sampler is least sure:
    prose +1.8 to +3.5 % and thinking max +0.9 to +2.0 %, against edit calls at −0.0 to +0.5 %.
  - The bar is the whole bench, and 87 % of its output is tool calls (edit, write and read).
- **Not recorded.** sparkDash's catalogs, the bench types and the transcript turns were not decoded at
  pi's sampling or recorded.
  - They decide only the bar's no-regression half, and no candidate passes the first half.
  - Under greedy decoding the tree study modelled the probability gate at +3.8 % on code and −1.7 % on
    agentic.
  - If a later gate needs them, they take about four ring parts.
- **An aside: speculative sampling with a sampled draft.**
  - Today's verify is speculative sampling with the drafter's top token as a one-hot proposal
    ([Drafts](#drafts-dspark)), so a draft is accepted with the sampler's probability of it.
  - Sampling the draft from the drafter's distribution, and keeping it with probability min(1, the
    sampler's over the drafter's), keeps the output's distribution exact. It does not keep the plain
    decode's tokens for a seed. A draft is then accepted with probability Σ min(drafter, sampler)
    over the vocabulary.
  - Along these records, with the draft sampled from the drafter's top 8 at temperature 0.5–2 and
    every step verifying all 5, the best temperature (0.7) gives +0.8 % tokens a step on r2. That is
    +0.3 % on edit calls, +4.5 % on prose and +2.3 % on thinking max. (The same estimate for today's
    drafts gives the 4.34 drafts a step that were accepted.)
  - Not proposed: it would give up bit-for-bit spec decode for under 1 % on pi's workload.

**Method.**
- **The outputs.** These are the edit bench's served turns, each with its decode rate and steps
  histogram (rank 0's /data/dsgb10/bench/edit).
  - r2, sampled at dsfront's defaults. Repeat 0 at thinking off is the training set (38 turns, 4,974
    tokens). Repeats 1 and 2 at thinking off, and repeat 0 at thinking max, are the test set (88
    turns, 12,111 tokens).
  - r1, greedy: 54 turns and 6,988 tokens, thinking off on 39 and max on 15.
- **The records.** Four ring parts of 220–310 s each: spectf with `--seq-list`, `--tree-probe` and
  `--route-file --route-every 8`, and the FP8 drafter as served. Each output is teacher-forced, one
  1-row spec step per position. Each position records:
  - the drafts d1..d5 and their survival;
  - the drafter's top 8 per depth, with its parents forced to the output (the chain's own through the
    first rejected draft);
  - the target's top 20 logprobs, which give the sampler's probability of each draft (T 1, top-p 0.95
    over those 20);
  - its routed experts, plus the drafts' rows at every 8th anchor.
- **Acceptance is exact for any gate.** A sampled verify accepts d_j when the plain sampler's draw at
  its position is d_j, and that draw is the served token. So the served output fixes every acceptance.
- **The walk and the step cost** are pld_study.py's.
  - λ lags two steps.
  - A step's ms is c0 + 1.10 (R − 1) + 0.713 (U − 6) + 2.55. U is the union of the step's own rows'
    experts, plus β (−0.002) per rejected row.
  - c0 is offset (+0.08 ms) so that today's gate replays r2's served rate.
- **Checks.**
  - Every position of every output has its record, its probe and its routes.
  - The probe matches the log's drafts and survival everywhere, and its top 1 is the chain's draft
    through the first rejected one.
  - The served token is in the sampler's kept set at 99.9 % of positions. The rest lie outside the
    target's top 20.
  - Today's gate replays r2's served decode: 126.6 tok/s both. The steps verifying 0..5 drafts are 5,
    76, 179, 241, 297 and 2654, against 5, 74, 180, 251, 279 and 2627 served.
  - By kind the replay is within 3.3 % (edit −1.2 %, write −3.3 %, prose +2.0 %), except for read
    calls (+6.7 %). Those are short turns, whose per-turn overheads the walk leaves out.
  - It replays r1's too: 128.8 tok/s both (offset +0.07 ms), and steps of 3, 24, 78, 104, 140 and
    1042 against 1, 22, 70, 112, 124 and 1048 served, as in the PLD study. On r1 the record's own greedy
    token differs from the served one at 51 of 6,934 positions, where spectf's prompt, summed in
    another order, flips a near-tie (the tree study saw the same). Acceptance comes from the served
    tokens, so this moves only the bound.
- **The candidates.** Each family was run at up to 10 settings (the per-session learners at 56), and
  the best on repeat 0 is reported on the rest.
  - Today's rule with its threshold scaled (×0.6–2), `--spec-gate ratio` and `marginal`, for
    reference.
  - **1: the drafter's probability.** Verify d_j while the product of the drafter's probabilities
    (also under top-p 0.95 on its top 8) is at least k·λ·ΔC. Or use the tree study's rule, the
    product at least α·λ·(the chain's next row's ms). Or take the length that maximises the
    expected tokens less k·λ times its rows' ms (`--spec-row-ms`).
  - **2: learned per session.** Acceptance by depth and confidence bucket, learned from the session's
    own steps two steps late and carried across its turns, at η 0.02–0.2. It is used as an offset on
    the head's logit or as a table, then picked by threshold, by the expected tokens less λ times the
    ms, or as `ratio` does.
  - **3: sampling-aware.** Logistic fits per depth on repeat 0 of acceptance against logit(c); against
    logit(c) and logit(the drafter's p); and against those plus the target's P(bonus) and its top P.
    At depth 1 they are sig(−0.59 + 1.06 logit c) and sig(−0.47 + 0.34 logit c + 0.58 logit p). They
    apply only to sampled requests.
  - **The bounds.** The sampler's own probability of each draft, from the target's logprobs at the
    drafts' positions (known only after the verify), under the same rules. And a perfect gate, which
    verifies exactly the drafts that will be accepted.

**To rerun** (the records need runner-gate-rec 22ec1d9's flags, on master since 631cf3c):
1. `gate_inputs.py edit DIR s0 /data/dsgb10/bench/edit/r2/index.jsonl --sel off:0 --tag r2`. s1 is
   `--sel off:1`, and s2 is `--sel off:2,max:0`. g1 is r1's index with `--tag r1 --sampling greedy`.
2. For each run: `run4.sh --mode spectf --big --draft-fp8 --golden FIRST --seq-list DIR/sl_RUN.txt
   --tree-probe DIR/tp_RUN.jsonl --route-file DIR/rt_RUN.bin --route-every 8`, where FIRST is the run's
   first golden directory.
3. `gate_study.py DIR --check --cal --replay --sweep --tables --ss`.

The records and the analysis's output (gate_study_6f6a6f9.txt) are in rank 0's /data/dsgb10/bench/gate.

### Lossless weight compression: a feasibility study (2026-10-12)

The question: decode is bound by streaming weights; at k = 1, 14.3 of the 17.9 ms is weights. Suppose
the stored weights took fewer bytes losslessly and the GEMVs decoded them inline at the same bandwidth.
Every step would then be faster with the same bits: the same FP8, FP4 and bf16 values going into the
same arithmetic in the same order. Is that worth 10 % of a decode step?

This was measurement and analysis only: the GPU bench and the CPU tools are bench-only, and serving is
unchanged.

**Verdict: not 10 %. Only one fixed-length code, the expert scales, can clear the ordinary 2 % bar by
itself.**
- **Entropy-coded FP8: no.** Two exact fused Huffman decoders are 10–58 % slower than today's kernels
  at one row (Part 2), and the published decoders agree. GB10 has no hardware decompression.
- **Fixed-length codes** (Part 3), all exact by construction:
  - the expert scales at 1 bit, which the kernels agent is prototyping: −0.67 to −0.86 ms at 6 rows
    (1.8–2.3 %);
  - the head's exponent at 4 bits from a tile base: ~0.3 ms on every step, below the bar on its own;
  - FP8 with a 3-bit code: skipped, at 2.0 % or less at 6 rows even with free decoding.

**Part 1: what the codes hold** (CPU only; [engine/weights/wcomp](../engine/weights/wcomp/README.md)).
- **The sample.** `wcomp` streamed named tensors of rank 0's file and dropped the pages behind it:
  - layers 0, 1, 2, 7, 14, 20, 21, 24, 31 and 39, with every tensor kind;
  - the experts at a stride of 8 (48 of 384 a layer);
  - the head whole, and the embed in 64 MB;
  - mtp.0–2.

  It read 3.64 GB with one thread at nice 19 and idle I/O priority, while rank 0 served.
- **What `analyze.py` computes:**
  - order-0 and conditional entropies;
  - Huffman codes, both optimal and limited to 12 or 8 bits by package-merge (a 12-bit limit is
    the longest code a single 4096-entry table lookup decodes);
  - DFloat11's split, which codes the exponent and leaves the sign and mantissa raw.
- **What `project.py` does:** it gives every tensor of the file its kind's bits, then converts the
  bytes saved into ms at 235 GB/s.
- **How much depth matters.** Across the sampled layers, a tensor kind's entropy moves by at most
  0.1 bits for FP8 and 0.02 bits for FP4.

Bits per code. A range spans tensor kinds; "saved" is net of 0.25 % for offsets (below).

| class | GB a rank per k = 1 step | order-0 entropy | given context | Huffman ≤ 12 bits | ≤ 8 bits | saved |
|---|---|---|---|---|---|---|
| FP8 e4m3, 32×32 block scales: attention, shared experts, indexer, Engram | 1.741 | 6.48–6.69 (Engram's wkv 6.95) | its block scale: −0.002 to −0.06; the previous byte: −0.005 to −0.04 (Engram −0.15) | 6.52–6.73 | 6.59–6.76 (exponent coded) | 17.3 % |
| bf16 head | 0.331 | 10.49 (exponent 2.57, sign and mantissa 7.92) | | 10.62 (exponent coded) | 10.69 | 33.4 % |
| bf16 router, compressor, indexer | 0.052 | 10.47–10.67 | | 10.59–10.79 | | 33 % |
| MXFP4 expert values, 6 experts a layer | 1.062 | 3.894 | its block scale: 3.77 | 3.80–3.81 (a nibble table per scale value) | 4 (raw) | 4.6 % |
| their ue8m0 scales | 0.066 | 0.99 | | 1.10 (two scales as one symbol) | 1.43 | 86 % |
| f32 mHC | 0.039 | exponent 2.88 | | 26.9 of 32 | | 15.7 % |

- **FP8.** With a 32×32 block scale, each block's amax lands near 448, so every block's codes follow
  nearly the same distribution: a 1-bit sign, a ~3-bit mantissa and a ~2.55-bit exponent. Neither the
  scale nor the neighbouring byte says much. wo_a is FP8 like the other projections: 0.336 GB a step,
  17.3 %.
- **bf16.** The redundancy is in the exponent: 2.6 of its 8 bits. The sign and mantissa do not
  compress, so DFloat11's split comes within 0.13 bits of the word's entropy.
- **MXFP4.** The 16 codes are nearly uniform. The scales are not:
  - in every sampled layer nearly every block scale is 2^-7 or 2^-6 (120 or 121, 59 / 41 %), so
    each byte per 32 values carries about one bit;
  - values at the higher scale rarely reach ±6, so coding the values given their scale saves
    another 0.12 bits.

  The MTP blocks' experts are similar (3.83; w2 of blocks 1 and 2, 3.64–3.71).
- **Tables.** One table per kind costs at most 0.02 bits against per-tensor tables, so a few shared
  tables cover everything. A 12-bit table is 8 KB of shared memory.
- **Offsets.** A parallel decoder has to know where each lane starts. Suppose a warp's lanes decode
  in lockstep from one interleaved stream, which the encoder writes in the order the lanes consume it
  (DietGPU's arrangement for ANS). Then a warp needs one offset per run and flushes up to 31 bits per
  lane at the run's end. That is 0.25 % for runs of ~32 KB; one u32 per 512-byte tile would be 0.8 %.

**Projection.** Bytes saved per rank at 235 GB/s, with decoding assumed free, so these are ceilings:

| scheme | k = 1 (17.9 ms) | 6 rows (37.3 ms) | +200 turn (~220 ms) | +2000 turn (~416 ms) |
|---|---|---|---|---|
| order-0 entropy (what ANS approaches) | −2.40 ms (13.4 %) | −3.91 (10.5 %) | −21 (9.5 %) | −23 (5.6 %) |
| Huffman ≤ 12 bits | −2.30 (12.8 %) | −3.69 (9.9 %) | −19 (8.7 %) | −21 (5.1 %) |
| ≤ 8 bits (FP8 and bf16 exponents; FP4 values raw) | −2.02 (11.3 %) | −2.83 (7.6 %) | −11 (4.8 %) | −11 (2.7 %) |

- **k = 1** reads 3.29 GB a rank: 2.166 GB of the layers' non-expert tensors and the head, plus
  6 experts a layer. One expert in every layer is 0.188 GB.
  - Of the 2.30 ms, the head saves 0.47, wo_b 0.32, wq_b 0.31, the shared experts 0.26, wo_a 0.25,
    the expert scales 0.24 and the expert values 0.21; the rest saves 0.24.
- **6 rows** reads 21 distinct experts a layer: the trees study measured 15.7–16.8 at 4 chain rows,
  and the step fit gives 37.3 ms at 6.
  - The draft pass adds 308 MB of FP8 (the FP8 head copy, attention, shared experts and main_proj)
    and ~22 experts.
  - At DESIGN's 27 experts a layer, the saving is −4.1 ms.
- **Turns.**
  - +200 streams ~195 ms of weights, ~187 ms of them experts.
  - +2000's MoE takes 180 ms in the encoder and 64 ms in the replay. 154 ms of the 180 is the
    experts' bytes, at 7.7 ms a layer; its attention GEMMs are compute-bound.
  - Turn savings need the prefill MoE GEMMs to decode in their producer warps and stay bound by the
    bytes they read.
- **Memory.** The rank file would shrink from 72.4 to 65.1 GiB.

Part 1 clears the 8 % gate at k = 1 and at 6 rows, but only if decoding costs nothing. Part 2
measures whether it does.

**Part 2: decoding inside the GEMV** ([bench_wcomp.cu](../engine/kernels/tests/bench_wcomp.cu); bench
only, on the GPU through `ring.sh`).
- **The kernel.** `k_gemv_fp8z` is the persistent GEMV (`k_gemv_fp8p`) with coded words in its ring
  instead of tiles.
  - Each site has one canonical Huffman code of at most 12 bits (package-merge). A 4096-entry u16
    table in shared memory decodes it: (length << 8) | byte, 8 KB.
  - Each (CTA, warp) decodes its own run, which starts 16-byte aligned. The offsets are [CTAs][warps]
    u32.
  - A warp's lanes decode in lockstep from one interleaved stream, DietGPU's arrangement. Before every
    pair of bytes, each lane holding 32 bits or fewer takes the next word, in lane order by ballot and
    popc.
  - The 16 decoded bytes go into `k_gemv_fp8p`'s own MMA code, so the per-lane partials and the
    reduction are today's.
- **v1, then v2.** v1's loop is 723 SASS instructions a warp-step (512 bytes of weights). Much of that
  is the collectives the compiler puts around ballots in divergent code. v2 manages the ring once a step
  and refills without branches, in 497 instructions. It can also run two CTAs an SM.
- **Exact.** Both decode every layer's bytes exactly: 40 layers and every grid, against the file's
  bytes. Both give outputs bit-identical to today's kernel; v2 checked 192, 280 and 192 cases over 1–8
  rows, grids and ring depths.
- **Compression as coded**, including flushes, offsets and the table:
  - wo_b 17.2 % at 48 CTAs, 16.1 % at 96;
  - wq_b 17.2 % and 16.6 %;
  - wq_a + wkv 15.8 % and 15.4 % (56 CTAs).

  More CTAs cut more runs, so there are more flushes.

| site | rows | today µs | persistent | v1 fused, 48 CTAs | v2 fused (CTAs, ring) | v2 GB/s, raw / coded | realised, v1 / v2 |
|---|---|---|---|---|---|---|---|
| wo_b | 1 | 42.9 | 44.6 | 59.8 | 47.2 (96, 4) | 222 / 187 | −244 % / −63 % |
| wo_b | 6 | 43.5 | 45.2 | 60.5 | 47.8 (96, 4) | 220 / 184 | −248 % / −61 % |
| wo_b | 8 | 42.7 | 47.0 | 61.3 | 47.5 (96, 4) | 221 / 185 | −258 % / −69 % |
| wq_b | 1 | 44.5 | 44.5 | 109.4 | 55.1 (96, 8) | 190 / 159 | −858 % / −142 % |
| wq_b | 6 | 51.5 | 45.8 | 112.6 | 55.0 (96, 4) | 191 / 159 | −693 % / −42 % |
| wq_b | 8 | 59.2 | 46.1 | 113.3 | 55.7 (96, 4) | 188 / 157 | −536 % / 36 % |
| wq_a + wkv | 1 | 12.0 | 12.3 | 35.9 | 19.0 (56, 4) | 121 / 102 | −1273 % / −379 % |
| wq_a + wkv | 6 | 17.3 | 18.1 | 42.4 | 33.0 (48, 8) | 69 / 58 | −915 % / −577 % |
| wq_a + wkv | 8 | 20.1 | 20.6 | 44.3 | 36.1 (48, 8) | 63 / 53 | −808 % / −502 % |

- **How to read it.**
  - Each time is µs a call: the median over launches that rotate through ≥ 256 MB of copies of the
    site, so L2 holds none of it.
  - "today" is the engine's classic kernel, from v2's job. At one row the engine runs wq_b and
    wq_a + wkv persistent instead: the next column.
  - "realised" is the share of the projected saving, today's time × the coded fraction of the bytes.
    Each version is measured against its own job's today.
  - Two and four rows fall between the rows shown.
  - wq_b's 36 % at 8 rows is against the classic kernel; the persistent kernel alone is 9.7 µs faster
    still. In the decode chain, though, the persistent kernel's carveout costs the kernels after it
    more ("The persistent normed GEMVs at six rows", above).
- **Where the time goes.**
  - Today's kernels stream at 235–245 GB/s, with 41 instructions a warp-step (classic) or 157
    (persistent).
  - v2's 497 decode 190–222 G bytes/s at 16 and 32 warps an SM, which is 4.0–4.6 G an SM. Realising
    ~16 % at 235 GB/s of coded bytes needs ~280 G/s, ~5.8 G an SM.
  - Take its coded bytes at the persistent kernel's rate as the baseline. Over that, v2 adds 0.48 ns
    (wo_b) and 0.88 ns (wq_b) a warp-step for its 340 extra instructions: 1.4–2.6 ps an instruction.
  - wq_a + wkv adds 1.9 ns a step. Its 4,480 steps a call can't amortise each CTA's table load and
    ring fill.
- **Per step**, on these three sites alone: v2 adds 0.88 ms at k = 1 and 0.94 ms at 6 rows over 40
  layers. The projection said −0.65 ms at k = 1.

**Prior art.** No published system makes batch-1 decode faster with variable-length codes:
- [DFloat11](https://arxiv.org/abs/2504.11651) Huffman-codes the bf16 exponent. It decompresses a block
  into HBM before cuBLAS runs, and is "approximately 2× slower" at batch 1.
- [NeuZip](https://arxiv.org/abs/2410.20650) ANS-codes bf16 exponents, layer by layer: −33 % it/s.
- [Unweight](https://research.cloudflare.com/papers/unweight-2026.pdf) (Cloudflare, 2026)
  Huffman-codes a 16-exponent palette in a separate kernel, at 420.8 GiB/s on H100. Only the
  fixed-length rebuild is fused. At batch 1 on H100, Llama-3.1-8B drops from 168 to 100 tok/s (−41 %).
- [Tan et al.](https://arxiv.org/abs/2606.15789) fuse rANS into the GEMM. It is "decoder-bound" at
  small batch, and they give no batch-1 numbers.
- [ECF8](https://arxiv.org/abs/2510.02676) Huffman-codes FP8 exponents and saves 9.8–14.8 %. It
  decompresses layer by layer, not fused, and gains only from larger batches.
- [DietGPU](https://github.com/facebookresearch/dietgpu) is the lane arrangement used here (rANS,
  refills by ballot and popc). It runs global to global: 250–600 GB/s on A100, and 592.5 GB/s on H100
  ([2606.18900](https://arxiv.org/abs/2606.18900)).
- nvCOMP's SM codecs on H100: ANS ~370 GB/s median, Bitcomp ~350, GDeflate ~70
  ([benchmarks](https://docs.nvidia.com/cuda/nvcomp/benchmarks.html)). Those numbers use whole H100s,
  memory to memory.
- [ZipNN](https://arxiv.org/abs/2411.05239) runs only on the CPU, for storage and transfer.
  [Huff-LLM](https://arxiv.org/abs/2502.00922) is accelerator hardware, and its simulated gain is gone
  at 256 GB/s.
- The one fused system that wins, [ZipServ](https://arxiv.org/abs/2603.17435) (ASPLOS '26), uses a
  fixed-length code. It codes the 7 most frequent bf16 exponents in 3 bits, as bitplanes, with a bf16
  fallback, and decodes branch-free into `mma.sync`. It is 1.31–1.87× faster than cuBLAS at batch
  8–32, with no batch-1 figures.
- [Heilper and Singer](https://arxiv.org/abs/2508.19263) find that FP8 exponents compress to 0.20–0.30
  of their size. FP4 values are "statistically uniform", so only the scales compress, as Part 1 found.

**Hardware decompression: none on GB10.**
- `CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK` reads 0 on rank 0 (driver 580.178.04, CUDA 13.0).
  The check is a driver query that creates no context: [dequery.c](../engine/weights/wcomp/dequery.c).
- Blackwell's Decompression Engine (Snappy, LZ4, Deflate; "up to 600 GB/s") is documented only for
  B200, B300, GB200 and GB300 ([FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)).
- It writes memory to memory. Even on those parts it could shorten loading, but it can't feed a GEMV.

**Verdict on FP8 Huffman: no.**
- v2 is exact, and per SM it decodes about as fast as the best published decoders.
- At one row it is still 10–58 % slower than streaming the raw bytes, on every site.
- So whether to encode at load time or in a new file format (per-grid streams, tables and offsets) no
  longer matters.

**Part 3: fixed-length codes.** This part costs two of Part 1's cheap wins and FP8's bulk against the
ordinary merge bar: the same bits, and ≥ 2 % end to end.
- **Bytes.** What each layout saves, at 235 GB/s. The counts come from per-group and per-chunk
  statistics on Part 1's sample ([wfix.cpp](../engine/weights/wcomp/wfix.cpp)).
- **Decode cost.** Each decoder is written and compiled
  ([fixdec.cu](../engine/weights/wcomp/fixdec.cu); SASS only, never run). Its loop is counted against a
  raw load's. The cost lies between two bounds:
  - hidden: the kernels it joins use 13–157 instructions a step, where v2's measured rate puts the
    limit at ~400–470;
  - v2's exposure: 1.4–2.6 ps for each added instruction a warp-step.

  The ranges below run from the second bound to the first
  ([fixcost.py](../engine/weights/wcomp/fixcost.py)).

| option | saves | instructions a step: added (the kernel's own) | k = 1 (17.9 ms) | 6 rows (37.3 ms) |
|---|---|---|---|---|
| expert scales: 1 bit, raw groups as fallback (the kernels agent's) | 85 % of the scales, 5.0 % of expert bytes | 4–10 (92) | −0.19 to −0.24 ms (1.0–1.3 %) | −0.67 to −0.86 (1.8–2.3 %) |
| the head's exponent: 4 bits from a tile base | 24.7 % of 0.331 GB | 31 (13) | −0.30 to −0.35 (1.7–1.9 %) | the same ms (0.8–0.9 %) |
| the head's exponent: 5 bits | 18.8 % | 44 (13) | −0.19 to −0.26 (1.1–1.5 %) | the same ms (0.5–0.7 %) |
| FP8: a 3-bit exponent code, with a lane fallback | 8.5 % of 1.741 GB, and of the draft pass's 0.308 | 85 (41–157) | +0.12 to −0.63 (−0.7 to 3.5 %) | +0.14 to −0.74 (−0.4 to 2.0 %) |

- **Expert scales.** Built and merged (master 2812072; [below](#the-experts-scales-coded-and-the-exact-compression-bundles-ab-2026-10-12)).
  Each scale takes 1 bit over {120, 121}. The 2.3 % of groups that leave that pair are read raw. A +200 turn gains ~9 ms (4 %)
  if the prefill MoE kernels decode the scales too.
- **The head's exponent.** One site, read once a step, so it saves the same ms at any row count.
  - **The code.**
    - The head slice's exponents span 98–127, so 5 bits cover every weight.
    - 98.5 % of its 512-byte tiles span 16 exponents or fewer. A tile is one warp-step: 8 rows × 32
      values. Those tiles take 4 bits from their own base; the other 1.5 % take 5 bits.
  - **The layout**, in tile order:
    - the sign-and-mantissa bytes, 256 B a tile;
    - the 4-bit indexes, 128 B a tile, with nibbles ordered so that a shift and a mask give a word's
      two;
    - a byte per tile: its base, and a flag for a 5-bit tile. A 5-bit tile's fifth bits go in a side
      array, found by a running count along the warp's run.
  - **The decode.**
    - Each pair of weights takes one PRMT, whose sign-replicate mode puts each sign at bit 15 or 31.
      Then (t & 0x807F807F) | (index × 128 + the base pair) builds the word.
    - That is 31 instructions a step, all ALU, no tables, on top of k_gemv16's 13.
    - The words are today's bf16 bits, into the same HMMA in the same order.
    - The 5-bit tiles take a warp-uniform branch.
  - **Sites.**
    - `gemv_bf16_norm` for the head, as a new format beside the router gate's. It is called for decode
      and verify in model.cu, and for the prefill's last token in prefill.cu.
    - Its 9–16-row twin.
    - dspark makes its FP8 copy from the raw head at bind, before the transform. `--draft-bf16` would
      read the coded head through the same kernel.
  - **Loader.** A load-time transform: a GPU kernel at bind, which takes milliseconds. After it the raw
    head can go, freeing 0.08 GB. The rank file doesn't change.
  - **Effort and risk.** ~1–1.5 days, low risk.
  - **Payoff.** On its own it misses the bar: 1.7–1.9 % at k = 1, 0.8–0.9 % at 6 rows, and nothing in
    turns.
  - **5 bits** needs no tile bases and has no exceptions. It saves less, though, and the fifth bit's
    plane costs more instructions (44).
- **FP8 with a 3-bit exponent code.** It applies to every dense FP8 site.
  - **The code.**
    - With the 32×32 block scales, every FP8 tensor's exponents sit in nearly the same window.
    - The best 8 consecutive exponents hold 98.25 % of values; in the worst tensor, 97.3 %.
    - A 7-value window with an escape code would let 3.0 % escape.
  - **The layout.** Per lane chunk (16 values, 16 bytes):
    - a nibble plane of sign and mantissa (8 B);
    - three bitplanes of codes (6 B), which makes 7 bits a value;
    - a flag bit per lane (a u32 a warp-step), set for chunks with a value outside the window;
    - for flagged chunks, a 4th exponent plane (2 B) in a side array, found by popc.

    24 % of lane chunks are flagged (measured; 31 % in the indexer's wq_b), which leaves 8.5 % saved.
  - **The decode.** 85 instructions a step, on top of 41 (classic) or 157 (persistent). It is
    branch-free, because the fallback is too common to branch on: nearly every warp-step has a flagged
    lane.
  - **Sites.**
    - `gemv_fp8`, `_norm` and `_multi`, in their classic, persistent and 9–16-row kernels;
    - `gemv_woa` (wo_a, through gemv16's FP8 widening);
    - `gemv_fp8w_f32` (the drafter's FP8 head);
    - the shared expert inside `k_moe_up` and `k_moe_down`;
    - the drafter's calls to all of these.

    The persistent ring's 512-byte stages would become a fixed part plus side data.
  - **Loader.** A load-time transform. The prefill GEMMs (`gemm_fp8_p`) read the raw FP8, so either the
    raw stays (+1.6 GB a rank) or their producer warps decode too.
  - **Effort and risk.** 1–2 weeks, with medium-high risk. The gain depends on the decode staying hidden
    in the persistent kernel, and v1 and v2 show that's not a given.
  - **Payoff.** With free decoding it would save 0.63 ms at k = 1 (3.5 %), but only 0.74 ms at 6 rows
    (2.0 %). At v2's exposure it saves nothing.

**Arithmetic order**, for all of them:
- The decoders rebuild the stored bytes exactly and hand them to today's MMA instructions in today's
  order. So the outputs are the same bits by construction.
- Each would be checked the way v2 was: decoded bytes against the file, outputs against today's kernel
  bit for bit, then the decode hashes.

**Build order.**
1. **The expert scales** (the kernels agent's). It is the only option that can clear 2 % at 6 rows by
   itself: 2.3 % if its decode stays hidden, which the MoE kernel's slack (92 instructions a step)
   suggests.
2. **The head's exponent, 4 bits from a tile base.**
   - It saves ~0.3 ms on every step, at one site, in ~1–1.5 days, with low risk.
   - On its own it misses the bar.
   - In one exact-compression merge with (1), the pair saves 0.49–0.59 ms at k = 1 (2.7–3.3 %) and
     0.97–1.21 ms at 6 rows (2.6–3.2 %).
   - So build it in that merge, or skip it if each change has to clear the bar alone.
3. **Skip FP8 with a 3-bit code.**
   - It touches every dense site.
   - Even with free decoding, 6 rows gain only 2.0 %; at v2's exposure, nothing.
   - If k = 1 alone ever matters, first spend a day on a fixed-length kernel in bench_wcomp, on wo_b and
     wq_b, to settle the exposure.
4. **Skip FP8 Huffman** (measured above), and the head's 5-bit variant: the tile base saves more for
   fewer instructions.

The rank file stays as it is: everything kept is a load-time transform.

### The head's exponent coded: built (2026-10-12)

Item 2 of the build order above, built: the vocab head's bf16 exponents as 4-bit offsets from a base
per tile, coded at load. The code is on branch attn-head4 (b6d0773; master merged as 8358020). It
went into one exact-compression bundle with the kernels agent's expert scales (kernels-scales), and
the bundle's engine A/B decided it: the same bits, at least 2 % end to end, and no case more than 2 %
slower. The bundle passed and was merged as master 2812072
([its A/B](#the-experts-scales-coded-and-the-exact-compression-bundles-ab-2026-10-12)).

**Result: the same bits everywhere, and the head's GEMV 25–27 % faster, −0.35 to −0.38 ms a call at
1–16 rows** (the study estimated −0.30 to −0.35). The head is one call a step, so that is 2.1 % of a
17.9 ms step at k = 1 and 1.0 % of a 37.3 ms step at 6 rows, if the step gives all of it back.

**The code.**
- A tile is one warp-step of the tile_fp8 layout: 8 rows × 32 values, 512 bytes. A tile whose
  exponents span 16 values or fewer keeps each value's sign and mantissa (a byte), and its exponent as
  a 4-bit offset from the tile's least: 388 bytes with the tile's header.
- **Raw tiles instead of 5-bit ones, a deviation from the costing.** A tile spanning more than 16
  exponents stays raw: its 512 bytes sit in a side array, found by the index in its header. The study
  costed 5-bit offsets for these. The raw fallback takes any spread, with zeros, subnormals, inf and
  NaN, and needs no second decode path. It reads 0.9 MB a call more than 5 bits would: 0.3 % of the
  head. Stored, it takes 4.7 MB more (1.4 %), since a raw tile's coded slots stay empty, which costs
  nothing in place.
- **Rank 0's slice:** 9,706 of 646,400 tiles raw (1.50 %). Stored, 255.8 MB in place of 331.0
  (0.773). Read a call, 252.0 MB (0.761).
- **The layout**, for T tiles of which R are raw:
  - [T][256 B] sign and mantissa bytes (sign << 7 | mantissa), lane l's 8 values at 8l;
  - [T][128 B] offsets, lane l's u32 at 4l: value 2j's in nibble j and value 2j + 1's in nibble
    4 + j, so that one mask gives a bf16 pair's two;
  - [T] u32 headers: (b << 23) | (b << 7) for a coded tile whose least exponent is b, 2^31 + r for
    raw tile r;
  - [R][512 B] the raw tiles, from the next 256-byte boundary.

**The decode**, a third gemv16 format beside kBF16 and kWOA (kernels/gemv16.cu).
- Every lane reads its tile's header; one L1 line holds 32 tiles' headers. The loads are predicated
  on the header's sign, with no branch:
  - a raw tile's lane loads its 16 bytes as kBF16 does, at an address one IMAD.WIDE makes from the
    header;
  - a coded tile's lane loads 8 bytes of signs and mantissas and 4 of offsets.
- Each bf16 pair of a coded tile takes four instructions:
  - a PRMT, which puts its two bytes in the two halves;
  - a mask of its two offsets;
  - an IMAD, the offsets × 128 plus the header's base pair, which puts each exponent in bits 7–14;
  - a LOP3, which keeps sign and mantissa and adds the exponents.

  That rebuilds the raw step's 16 bytes in registers. They go into kBF16's MMAs in kBF16's order, so
  the outputs are the same bits at any row count.
- The main loop up to 8 rows (mt 2) takes 45 instructions a warp-step against kBF16's 13: +32, where
  the study estimated +31. The 9–16-row kernel takes 84 against 51.
- The head's two kernels use kBF16's registers (122 and 128) and don't spill. The drafter's plain
  instance uses 126 against kBF16's 98.
- **The first version** kept the headers in registers, shuffled them across the lanes and branched on
  a raw tile: 64 instructions a step, and the 16-row kernel spilled. Reading the header again from L1
  at the decode and predicating the loads brought it to 45.

**Where it runs.**
- `dsk::bf16x_encode` and `bf16x_check`; `gemv_bf16x` and `gemv_bf16x_norm` (tensor cores, mt 1 or 2).
- `Model::code_head` runs after `bind` and `bind_spec`, so DSpark's FP8 head copy is made from the raw
  head first. It codes the head on the GPU, checks every tile against the raw head, and copies the
  code over the raw head in its arena slot. That takes 0.03 s, with 256 MB more for that moment. The
  arena is one allocation, so the slot's last 75 MB stay allocated.
- The readers:
  - the head GEMV of decode and verify (model.cu);
  - the prefill's last token (prefill.cu);
  - the drafter's bf16 head (`--draft-bf16`).

  The server codes the head unless `--head-raw`.

**Measured** (one job, binaries from 8358020).
- **Bits.** test_kernels head4 runs on rank 0's head slice, and on a copy with planted tiles: ±0,
  subnormals, exponents 0–15, 240–255 with inf and NaN, a span of 15, a span of 16 and a lone zero
  (both go raw), and the last tile at ±max.
  - Every tile decodes to the raw bytes, and one flipped mantissa bit is caught.
  - The raw count equals a host scan.
  - `gemv_bf16x_norm` and `gemv_bf16x` equal `gemv_bf16_norm` and `gemv_bf16` bit for bit, at 1–8
    and 9–16 rows, at mt 2 and 1.
  - Today's gemv16 tests (head, woa, normgemv, g16p, rows16) pass.
- **The GEMV** (bench_kernels head4). It runs on the real head, cold: the head is 6–8 times L2 and
  SLC together. The figures are µs a call, the median of 30 replays of 4 calls each, with the
  variants interleaved, at chead (mt 2, wk 16) as the engine calls it:

  | rows | bf16 | coded | change | coded at mt 1 |
  |---|---|---|---|---|
  | 1 | 1,390.6 | 1,013.6 | −377.1 (−27.1 %) | 1,005.1 |
  | 2 | 1,391.4 | 1,016.6 | −374.8 (−26.9 %) | 1,007.4 |
  | 4 | 1,392.4 | 1,022.0 | −370.5 (−26.6 %) | 1,059.9 |
  | 6 | 1,400.6 | 1,040.2 | −360.5 (−25.7 %) | 1,173.2 |
  | 8 | 1,400.7 | 1,055.8 | −344.9 (−24.6 %) | 1,234.0 |
  | 12 | 1,398.3 | 1,039.5 | −358.8 (−25.7 %) | 1,229.7 |
  | 16 | 1,397.8 | 1,045.1 | −352.7 (−25.2 %) | 1,326.4 |
  | 5, the drafter's plain GEMV | 1,393.8 | 1,034.5 | −359.3 (−25.8 %) | |

  - The bf16 head streams at 236–238 GB/s. The coded one reads its 252.0 MB at 239–249 GB/s: 0.73–0.75
    of bf16's time for 0.761 of the bytes, so the decode is hidden.
  - mt 1 is 8 µs faster at 1–2 rows and slower from 4 rows. Not taken.
- **The engine**, with the coded head (the default):
  - tf 842278b28d87dad0;
  - 1k at k = 1, febc7dc54ee40a39;
  - 1k with drafts on the FP8 drafter head, febc7dc54ee40a39;
  - 1k with drafts on the bf16 drafter head, which reads the coded head, febc7dc54ee40a39;
  - sampled 1k (T 0.8, top-p 0.95, top-k 40, seed 7), a8f7a08f8cd1c5e8;
  - 16k through CED, 81f772ab79a950e4;
  - long28k through CED, 14650fb0739d0383 (until 2026-10-13 the hash's seed, see
    [Golden check](#golden-check); now 8f08a153d415c5c8);
  - `serve_test.py --ced` against a server with serve's defaults (the coded head, drafts on the FP8
    drafter head): 17 of 17.
  - With the bf16 drafter head, the coded head and `--head-raw` give the same drafts line. With the
    FP8 one, the drafts are those of the engine before the coding (72 verify steps, 2.542 drafts
    accepted a step).
- **Memory per rank: no change**, as built. The free device memory after the prefill buffers, at 1k
  and k = 1, was 35.60, 38.13, 38.58 and 38.68 GiB on ranks 0–3 with the coded head, and 36.13,
  38.16, 38.57 and 38.66 with `--head-raw`. Ranks 1–3 are within 0.03 GiB. Rank 0's figure moves by
  ±0.4 GiB between runs of one binary (rank 0 serves and builds), and it read 36.31 in the coded tf
  run. Coding the head freed nothing, since the slot keeps its 331 MB. Freeing the 75 MB would need
  the head in an allocation of its own.
- **Time in the engine.** The end-to-end figure is the bundle A/B's. The first job's single runs:
  - 1k at k = 1: 17.603 ms per token with the coded head, 17.909 with `--head-raw` (−0.31 ms);
  - 16k through CED: 17.574, against 17.977 in the morning's job.

  A side probe ran 1k with drafts in the order coded, raw, raw, coded (raw: `--head-raw`), one part
  for each drafter head:
  - **The bf16 drafter head**, which reads the coded head too: coded 8.996 and 8.983 ms per token,
    raw 9.202 and 9.162, so −0.19 ms (−2.1 %). The median step is 0.64 ms shorter; two head calls
    at −0.36 ms each would make 0.72.
  - **The FP8 drafter head**, serve's default, where only the main head is coded: coded 8.898 and
    8.865, raw 8.942 and 8.955, so −0.07 ms (−0.7 %). At 3.54 tokens a step that is 0.24 ms a step,
    of the 0.36 expected.
  - The first job's single runs had put the bf16 drafter the other way round: 9.200 coded against
    9.158 raw. Runs of one binary spread by up to 0.22 ms per token, so one pair says nothing at
    this size. The drafter's coded plain kernel uses 126 registers against 98, which costs nothing
    measurable here.

### The experts' scales coded, and the exact-compression bundle's A/B (2026-10-12)

Item 1 of Part 3's build order ([above](#lossless-weight-compression-a-feasibility-study-2026-10-12)),
built: the routed experts' ue8m0 block scales as a 1-bit code over each tensor's pair {b, b + 1}, with
the groups that leave the pair read raw. With [the head's exponent](#the-heads-exponent-coded-built-2026-10-12)
(item 2) it makes one exact-compression bundle, and the bundle's engine A/B decides both.

**The question.** Built and run in the engine, do the two codes keep every bit, and is the pair at least
2 % faster end to end, with no case more than 2 % slower?

**Verdict.** Passed, and merged as master 2812072 (2026-10-12).
- **Bits.** The tokens were the same in all 328 requests, and every edit turn matched its trace.
- **Speed.** The bar's 36 cases were 2.61 % faster in geometric mean, and none was slower: the smallest
  gain was 1.41 %.
  - With drafts, serve's default: 3.24 % faster.
  - At k = 1: 1.97 % faster.
- **End to end**, the edit replay took 3.83 % less time with drafts and 1.65 % less at k = 1.
- **Memory.** The bundle uses 0.61 GiB a rank more.

**Where the code is.** Master 2812072 merged branch kernels-bundle, which holds:
- from branch kernels-scales:
  - 6c5f6f7: [escales.cpp](../engine/kernels/tests/escales.cpp), the census (CPU only);
  - 2b0661c: the bench prototype (`bench_kernels mscale`);
  - e8cd5e6: the engine build. `dsk::scale_code` and the coded kernels are in
    [moe.cu](../engine/kernels/moe.cu), [prefill.cu](../engine/kernels/prefill.cu), `common.cuh` and
    `dsk.h`. The codes are made at bind in [model.cu](../engine/model/model.cu), and
    `test_kernels mcode` checks them.
- bcd6197, which merges attn-head4 (8358020, which is b6d0773 with master);
- 55f7cbd, which merges master 984b688. The A/B's bundle binaries were built from this commit;
- c1b82a3, which adds the A/B's tools: [abserve.sh](../engine/kernels/tests/abserve.sh),
  [abload.py](../engine/kernels/tests/abload.py) and [abcmp.py](../engine/kernels/tests/abcmp.py).

**What the scales hold** (escales, on rank 0's file; ranks 1–3 are within 0.5 points a tensor).
- **The values.** A rank holds 4.35 GB of expert scales, 4.25 GB of them in the 40 layers.
  - 53.5 % of the values are 120 and 44.2 % are 121.
  - 1.7 % are 122, 0.2 % are 123, and 0.4 % are 115–119.
- **A group** is the 32 scales that one 16-row tile reads in one step of 64 K values: one warp-step.
  - In the 40 layers, 98.11 % of groups hold only their tensor's pair. The pair is {120, 121} in all 120
    tensors.
  - Per layer and matrix the share runs 93.8–99.1 %, except layer 39's w2 at 85.4 %.
  - The MTP blocks' experts sit at {122, 123} or {121, 122}, with 0.3–61 % of their groups in the pair.
    They stay raw, and dspark reads them as before.
- **The bytes.** Under the code, a step of the 40 layers reads 15.0 % of the scale bytes it read raw,
  counting the code words, the flags and the raw groups. That saves 5.0 % of an expert's bytes, as Part 3
  costed.
- **The raw groups are spread out.**
  - 68.6 % of w1 tiles, 62.8 % of w3 tiles and 11.1 % of w2 tiles hold at least one raw group.
  - 29.5 % of `moe_up`'s 20-step warp slices of w1 hold one, and 24.0 % of w3's.
  - So the raw path can't be a rare branch in `moe_up`. In `moe_down`, 89 % of units never take it.

**The format** (`dsk::scale_code`, made at bind: 579.4 MiB a rank).
- **The base.** Each tensor (a layer's w1, w3 or w2) has a base b. b is its commonest value, or the value
  below it when that one is more common than the value above. A scale is coded as the one bit s − b.
- **The block.** Each (expert, tile) has a block of P code words: P = 80 steps for w1 and w3, and 9 for w2.
  - Bit 8j + g of word p is byte j of word g of step p's group, minus b.
  - ⌈P / 32⌉ flag words follow. Each flag bit marks a step whose group holds a value off the pair.
  - The block is padded to 16 B: 84 words for w1 and w3, and 12 for w2. Raw, the same scales take 640
    and 72 words.
- **The decode.** `((w >> g) & 0x01010101) + b · 0x01010101` gives word g of the group, in 3
  instructions.
- **The raw groups.** A flagged group is read from the raw scales, which stay.

**The kernels.** In every variant the same scale bytes go into the same MMAs in the same order. A kernel
takes the code when `MoeWeights` carries it (`c1`, `c3` and `c2`). dspark's experts carry none.
- **Up:** `k_moe_up` <1, 4, 4> at 1–8 rows, and `k_moe_up_r16` <2, 4, 2> at 9–16 rows.
  - A tile's flags are one 16-byte load from its block. A funnel shift gives the 32 steps from any step
    on.
  - The first two load groups are issued from the code words. A fix-up reloads any raw ones.
  - Later steps choose code or raw by lane, in a predicated pair of loads (`ldg_stream32_sel`).
- **Down:** `k_moe_down` <4, 1> at one row and <16, 4> at 2–8 rows, and `k_moe_down_r16` <8, 4> at
  9–16 rows.
  - Each warp reads its units' flags into shared memory before the PDL wait, while `moe_up` finishes.
  - Each lane holds one code word. A unit's first four raw groups are held 8 lanes each: lane / 8 is the
    raw group's rank in the unit, and lane % 8 its word. Both are shuffled out at each step.
  - A unit with no raw group takes a branch-free path. A fifth or later raw group is read directly.
- **Prefill:** `k_moe_up_p` and `k_moe_down_p` select each step's code word or raw group, with the raw
  path's L2 hints.
  - The tiles' flags are read beside the token ids (up) or the h rows (down), so their latency hides.
  - `k_moe_down_g`, a tuning that is off by default, reads the raw scales.
- **Spills found and fixed.**
  - `k_moe_down` <16, 4> at first held a unit's nine words a buffer. It spilled 64 B right after its
    loads and stalled on them. The shuffles above hold two.
  - `k_moe_up_p` with its flags in registers took 255 registers and 48–56 B of stack. With the flags in
    shared memory and the code addressed by a word offset, it takes 249–250 and no stack.
  - In the bundle's SASS, the coded decode kernels have no local-memory traffic, except `k_moe_up`
    <1, 4, 4>: 3 local stores and 2 loads. Its raw build has 3 and 3.

**Bits.**
- **test_kernels `mcode`** (in `moe`) compares the coded run with the raw one, bit for bit:
  - `moe()` at 1, 6, 8, 9, 12 and 16 rows, under 3 routings, checking y and up's output in the
    workspace;
  - `moe_prefill` at 256, 200, 129, 64 and 1 rows, fused and separate, with and without hints,
    checking y.

  It runs on the reference experts and on a pool of 128 random experts. In the pool, expert e's groups
  leave the pair with probability (e % 12) / 11, so every path of the decode runs.
- **kernels-scales e8cd5e6 and the bundle 55f7cbd both passed**, in one ring job:
  - test_kernels, with head4 on the bundle;
  - every reference hash: tf 842278b28d87dad0; 1k febc7dc54ee40a39 at k = 1, with FP8 drafts and with
    bf16 drafts; sampled 1k a8f7a08f8cd1c5e8; 16k CED 81f772ab79a950e4; long28k CED 14650fb0739d0383
    (until 2026-10-13 the hash's seed, see [Golden check](#golden-check); now 8f08a153d415c5c8);
  - `serve_test.py --ced`, 17 of 17.
- **The A/B below** compared every request's tokens between master and the bundle.

**The kernels' time** (bench_kernels `mscale`).
- **The setup.** One layer's MoE, with 96 experts' real scales from rank 0's file and random weights,
  and a synthetic layer in which half the groups are raw. Raw and coded run in turn over 15 rounds, and
  the times are medians.
- **The columns.** "Real." is the time saved as a share of "proj.", the raw call's time scaled by the
  bytes the code saves. The bits were the same in every case.
- **The runs.** The prototype (2b0661c) ran twice, in separate jobs; the engine build (e8cd5e6) ran
  once. Its µs are shown. The 12- and 16-row cases exist only in the engine build.

| case | layer 20: raw → coded µs | real.: engine (prototype) | layer 39 | real. | synthetic, half raw | real. |
|---|---|---|---|---|---|---|
| 1 row, 6 experts | 164.8 → 161.3 | 55 % (101, 103) | 165.0 → 162.0 | 52 % (93, 88) | 165.0 → 165.7 | −22 % (35, 38) |
| 6 rows, 12 experts | 290.7 → 273.8 | 134 % (124, 125) | 289.5 → 274.5 | 130 % (131, 124) | 290.7 → 283.1 | 121 % (105, 106) |
| 6 rows, 21 experts | 476.0 → 446.1 | 136 % (131, 134) | 474.5 → 447.4 | 135 % (133, 129) | 475.5 → 460.3 | 138 % (134, 121) |
| 6 rows, 36 experts | 778.6 → 731.1 | 128 % (125, 129) | 778.7 → 735.7 | 126 % (129, 124) | 779.5 → 756.4 | 124 % (127, 118) |
| 8 rows, 21 experts | 476.6 → 445.9 | 140 % (132, 137) | 474.6 → 447.8 | 134 % (132, 130) | 476.0 → 460.7 | 140 % (129, 126) |
| 8 rows, 48 experts | 1029.6 → 962.3 | 135 % (132, 131) | 1030.0 → 969.2 | 133 % (137, 129) | 1034.7 → 1002.8 | 127 % (129, 130) |
| 12 rows, 36 experts | 783.0 → 730.1 | 142 % | 783.1 → 735.2 | 140 % | 782.7 → 755.9 | 143 % |
| 16 rows, 64 experts | 1359.7 → 1262.6 | 147 % | 1360.5 → 1272.3 | 145 % | 1361.3 → 1312.3 | 147 % |

- **6–16 rows save 21–47 % more than their bytes.** One guess is that the raw scale loads cost more than
  their bytes: a raw step issues a small load a group, where the code issues one word a lane for the
  whole block. That is not verified. On rank 0, ncu needs admin (`RmProfilingAdminOnly` 1), and nsys
  counts no bytes a kernel.
- **One row lost half its gain between the prototype and the engine build.**
  - The coded `k_moe_down` <4, 1> took 53.1, 54.3 and 56.0 µs (layer 20, layer 39, synthetic), against
    the prototype's 51.9, 52.8–53.1 and 54.5. Raw took 53.3–53.8 in all three runs.
  - The engine build's whole coded call is also 1.5 µs longer than its two phases (161.3 against
    106.7 + 53.1). The prototype's added up (158.3).
  - **The likely cause is registers.** The engine build moved the down kernel's code into shared helpers
    (`code_load_down`, `code_steps_down`) with no change to what it computes. Its coded <4, 1> build
    went from 128 registers (and 8 B of stack) to 132.
    - At 128 threads a block, 128 registers allow 4 blocks an SM, and 132 allow 3: registers go by 256
      a warp.
    - Raw has 158, so 3 blocks. If this is the cause, the prototype's 1-row lead came from running 16
      warps an SM against raw's 12.
  - This is not confirmed by a run. The check is the coded <4, 1> build at `__launch_bounds__(128, 4)`,
    which changes registers only, then mscale. At 2.2–3.0 µs a layer, the gap is worth 0.09–0.12 ms a
    token at k = 1 (0.5–0.7 %). With drafts it is worth almost nothing: a verify step runs one row only
    when it verifies no draft (1 step of 72 in the 1k hash run).
  - **Left on the table** (2026-10-12): it is a k = 1 gain only, and production decodes with drafts.
- **Over 40 layers**, the engine build saves 0.12–0.14 ms a token at one row and 1.1–1.2 ms a step at
  6 rows with 21 experts. Part 3 put the scales at −0.19 to −0.24 ms and −0.67 to −0.86 ms.

**The head's code** is written up [above](#the-heads-exponent-coded-built-2026-10-12): the head GEMV
25–27 % faster, −0.35 to −0.38 ms a call, the same bits. It went into the bundle unchanged.

**The bundle A/B.**
- **The jobs.** Two ring jobs, one server session per part, with master 984b688 and the bundle 55f7cbd
  alternating. Each part stayed under 10 minutes and ran with `PREEMPT=0`, and each job held its want
  across its parts.
  - Job 1 ran: master with drafts, the bundle with drafts, the bundle at k = 1, master at k = 1.
  - Job 2 ran the same in reverse.
- **The servers** ran production's arguments (`--max-ctx 262144`, and serve's defaults: CED, and drafts
  on the FP8 drafter head) without the disk tier (`--disk-gib 0`). The k = 1 sessions added
  `--no-spec`.
- **The client** (`abload.py`) spoke to rank 0's socket directly, with no front-end:
  - **bench:** bench.py's four decode prompts as dsfront renders them with thinking off. Greedy, 512
    tokens, twice each, in a fresh session each.
  - **edit:** the edit bench's r1 traces (54 turns of 16 tasks, thinking off and max), replayed in order.
    - Each task is one session, and its first turn is forked from the previous task's session, as
      served.
    - Each turn asks for its trace's number of tokens, and every turn's tokens were checked against the
      trace.
  - **turns:** at 16,384 and 25,984 positions of golden/long, three times each, in a fresh session:
    - a cold prompt;
    - then +200 golden tokens;
    - then +2000 golden tokens.

    Each turn comes after the previous answer and answers 64 tokens. Both turns run CED, since each is
    at least 128 new positions. The golden text has 28,489 tokens, so the deeper turns start at
    25,984 and end at 28,376 positions.
- **The bar's cases** are what a user waits for:
  - the bench's four decode rates;
  - the edit replay's 54 turns, end to end;
  - the +200 and +2000 turns at both depths, end to end.

  The change is the bundle's time against master's, so negative is faster. The figure is a geometric
  mean.

| case | job 1: master → bundle | change | job 2: master → bundle | change | mean |
|---|---|---|---|---|---|
| **with drafts** (serve's default) | | | | | |
| **bench prose, tok/s** | 70.8 → 72.5 | −2.31 % | 70.4 → 72.1 | −2.29 % | −2.30 % |
| **bench code, tok/s** | 103.6 → 106.8 | −2.96 % | 103.0 → 106.3 | −3.12 % | −3.04 % |
| **bench agentic, tok/s** | 101.4 → 104.6 | −3.00 % | 100.9 → 104.2 | −3.19 % | −3.10 % |
| **bench json, tok/s** | 138.0 → 143.9 | −4.09 % | 137.3 → 143.3 | −4.16 % | −4.13 % |
| **edit replay, 54 turns, s** | 66.4 → 63.7 | −4.08 % | 66.5 → 64.1 | −3.57 % | −3.83 % |
| edit replay, decode, tok/s | 131.2 → 136.8 | −4.13 % | 131.1 → 136.0 | −3.64 % | −3.88 % |
| edit replay, prefill, s | 13.6 → 13.0 | −3.91 % | 13.6 → 13.1 | −3.32 % | −3.62 % |
| 16k cold, TTFT ms | 1807.4 → 1795.4 | −0.66 % | 1826.7 → 1790.6 | −1.98 % | −1.32 % |
| 16k +200, TTFT ms | 220.4 → 211.3 | −4.14 % | 220.6 → 213.6 | −3.19 % | −3.67 % |
| **16k +200 turn, ms** | 812.4 → 781.9 | −3.76 % | 810.8 → 789.2 | −2.67 % | −3.22 % |
| 16k +2000, TTFT ms | 411.8 → 397.3 | −3.51 % | 410.4 → 404.5 | −1.46 % | −2.48 % |
| **16k +2000 turn, ms** | 1016.2 → 982.3 | −3.34 % | 1024.3 → 988.0 | −3.54 % | −3.44 % |
| 26k cold, TTFT ms | 2928.1 → 2889.3 | −1.33 % | 2947.9 → 2902.2 | −1.55 % | −1.44 % |
| 26k +200, TTFT ms | 212.4 → 203.8 | −4.05 % | 213.0 → 204.4 | −4.04 % | −4.05 % |
| **26k +200 turn, ms** | 804.7 → 780.0 | −3.06 % | 807.2 → 783.8 | −2.91 % | −2.99 % |
| 26k +2000, TTFT ms | 400.3 → 385.4 | −3.71 % | 403.1 → 391.6 | −2.87 % | −3.29 % |
| **26k +2000 turn, ms** | 1206.3 → 1169.6 | −3.05 % | 1217.3 → 1178.4 | −3.19 % | −3.12 % |
| **k = 1** (`--no-spec`) | | | | | |
| **bench prose, tok/s** | 55.2 → 56.5 | −2.26 % | 55.3 → 56.2 | −1.66 % | −1.96 % |
| **bench code, tok/s** | 55.2 → 56.5 | −2.28 % | 55.3 → 56.1 | −1.55 % | −1.91 % |
| **bench agentic, tok/s** | 55.1 → 56.4 | −2.38 % | 55.3 → 56.1 | −1.55 % | −1.96 % |
| **bench json, tok/s** | 55.1 → 56.4 | −2.34 % | 55.3 → 56.2 | −1.66 % | −2.00 % |
| **edit replay, 54 turns, s** | 141.0 → 139.1 | −1.41 % | 140.3 → 137.6 | −1.90 % | −1.65 % |
| edit replay, decode, tok/s | 54.4 → 55.1 | −1.17 % | 54.7 → 55.7 | −1.71 % | −1.44 % |
| edit replay, prefill, s | 13.6 → 13.1 | −3.60 % | 13.6 → 13.1 | −3.60 % | −3.60 % |
| 16k cold, TTFT ms | 1824.7 → 1804.1 | −1.13 % | 1821.3 → 1779.2 | −2.31 % | −1.72 % |
| 16k +200, TTFT ms | 221.1 → 209.1 | −5.41 % | 221.3 → 209.4 | −5.37 % | −5.39 % |
| **16k +200 turn, ms** | 1386.5 → 1362.7 | −1.72 % | 1385.5 → 1359.1 | −1.91 % | −1.81 % |
| 16k +2000, TTFT ms | 412.0 → 400.8 | −2.71 % | 412.4 → 400.0 | −3.00 % | −2.86 % |
| **16k +2000 turn, ms** | 1584.7 → 1547.0 | −2.38 % | 1573.7 → 1549.1 | −1.56 % | −1.97 % |
| 26k cold, TTFT ms | 2940.5 → 2879.6 | −2.07 % | 2934.3 → 2891.6 | −1.46 % | −1.76 % |
| 26k +200, TTFT ms | 209.6 → 201.5 | −3.88 % | 211.4 → 201.3 | −4.76 % | −4.32 % |
| **26k +200 turn, ms** | 1375.4 → 1346.7 | −2.09 % | 1380.2 → 1349.0 | −2.26 % | −2.17 % |
| 26k +2000, TTFT ms | 398.9 → 388.8 | −2.54 % | 399.4 → 390.3 | −2.27 % | −2.40 % |
| **26k +2000 turn, ms** | 1573.8 → 1534.3 | −2.51 % | 1565.2 → 1533.4 | −2.03 % | −2.27 % |

**What the table says.**
- **The bar.** The geometric means over the 9 bar cases:
  - with drafts, −3.30 % in job 1 and −3.18 % in job 2;
  - at k = 1, −2.15 % and −1.79 %;
  - over all 36, −2.61 %.

  Every case was faster in both jobs. The smallest gain was the k = 1 edit replay's in job 1, 1.41 %.
- **Per token** (the decode time over the tokens after the first, job 1 and job 2):
  - with drafts: −0.30 and −0.31 ms on the bench, −0.31 and −0.28 on the edit replay, and −0.36 and
    −0.34 in the turns;
  - at k = 1: −0.42 and −0.29 ms on the bench, −0.22 and −0.31 on the edit replay, and −0.22 and −0.29
    in the turns.
- **Against the kernels' figures.** These projected about −0.5 ms a token at k = 1: the head's GEMV
  −0.35 to −0.38 ms, and the scales −0.12 to −0.14 at one row. The engine kept about −0.3.
  - The head alone measured −0.31 ms a token in the engine, in its own write-up's single runs.
  - So within the jobs' spread (about ±0.1 ms a token), the k = 1 gain is the head's. That fits the
    one-row finding above.
  - With drafts, where steps run 1–6 rows, the scales add more. The head's −0.36 ms a step is about
    −0.10 ms a token at about 3.5 tokens a step, which leaves about −0.2 ms a token, −0.7 ms a step, to
    the scales.
- **Prefill** reads the code too.
  - The edit replay's prefill took 3.6 % less in both settings (13.6 → 13.1 s).
  - A +200 turn's TTFT fell by 7.0–12.0 ms (3.2–5.4 %). Part 3 estimated ~9 ms if the prefill MoE read
    the code.
  - A +2000 turn's TTFT fell 1.5–3.7 %, and a cold prompt's 0.7–2.3 %.
- **The spread between the jobs.** Master's k = 1 bench decoded 55.2 and 55.3 tok/s, and the bundle's 56.5
  and 56.2. Job 2 ran each setting's pair in the reverse order of job 1.

**Memory.** Ranks 1–3 had 0.58–0.64 GiB less device memory free with the bundle, after the prefill buffers,
in all four same-job pairs.
- That is the codes' 579.4 MiB, in 40 allocations of 14.5 MiB. cudaMalloc most likely rounds each one up
  to 16 MiB, which makes 0.63 GiB.
- The head is coded in place, so it costs nothing.
- With drafts, ranks 1–3 had 34.60, 35.03 and 35.36 GiB free under master, and 33.98, 34.42 and 34.75
  under the bundle.
- Rank 0's difference ran from +1.27 to −3.24 GiB with rank 0's other use, so it says nothing here.

**Option, not built: an exceptions table** (~4.2 GB a rank back).
- **The change.** The raw scales would stay only for the raw groups.
  - The raw scales take 4.25 GB a rank in the 40 layers, but only 1.9 % of their groups are read raw.
  - A table of just those groups takes ~80 MB.
  - A block's offset into the table fits in its padding words: w1 and w3 blocks have one spare word of
    84, and w2 blocks two of 12. So the offsets cost nothing.
- **The saving.** About 4.2 GB a rank, since the raw scales' 4.25 GB would give way to the table's
  ~0.08 GB. Net of the codes' 579.4 MiB (0.61 GB), that is ~3.6 GB below the engine before the codes.
  - That is room for more sessions' caches.
  - Or for prefill chunks of 16,384 rows, which take 2.6 % less time a token than two chunks of 8,192
    and need 3.2 GiB a rank ([What remains for prefill](#what-remains-for-prefill)).
- **What it needs:**
  - **Every reader on the codes.** The decode and prefill MoE kernels read them already. `k_moe_down_g`
    would need the code or would have to go. The test and bench paths that compare against raw scales
    would keep their own raw copies.
  - **The raw groups addressed by rank.** A raw group would be found at its block's offset plus its rank
    among the block's raw steps, a popc of the flags. Down already computes that rank. Up would add a
    popc for each raw step.
  - **A loader or format change.** The weights sit in one arena laid out like the rank file, which
    cannot give back part of a slot (the head's 75 MB stay allocated for that reason). Freeing the
    scales needs a compacted arena, in which the loader stages the scales, codes them and leaves their
    slots out. Or it needs rank files that carry the code and the table in place of the raw scales.

### The prefill MoE at small chunks (2026-10-12)

A small agentic turn spends most of its prefill in the routed experts. Its passes depend on its size:
- **Under 128 new positions** (`serve` takes CED only when a turn adds 128 or more), the turn is an exact
  continuation: all 40 layers over its new rows. A +44 turn is one.
- **From 128 on**, the encoder runs layers 0–19 over the new rows, and the replay runs layers 20–39 over the
  last 128 ([CED](#ced)).

**The question.**
- At these chunk sizes, are the experts' kernels bound by the bytes of the experts they touch?
- What does reading the scales' 1-bit code save there?
- The bundle's A/B moved a +200 turn's time to first token by −5.4 % and the k = 1 edit prefill by
  −3.6 % ([the A/B](#the-experts-scales-coded-and-the-exact-compression-bundles-ab-2026-10-12)). Where
  did that come from?

**Verdict.** A finding. There is no kernel lever left at small chunks.
- **At the floor.** Up to 728 rows, up and down each run at 91–99 % of the touched experts' bytes at
  235 GB/s (213–233 GB/s), in the mean over a case's layers. The whole MoE call runs at 90–97 %.
- **What is left.**
  - In the engine's configuration (the code, down in 4 column groups), the MoE sits 6–23 ms a turn above
    its floor, from +44 to +1,368. That is 3.6–6.7 % of a turn's time to first token.
  - Prefill is 12–18 % of the edit bench's time, so even the whole gap is under 1.2 % end to end.
  - No kernel closes all of it.
- **The time is the experts touched.**
  - A layer's MoE time follows the number of experts its rows pick, not the rows.
  - The replay's 128 rows cost ~60 ms a turn, whatever the turn's size.
- **The code is where the bundle's prefill gain came from.** The prefill kernels have read it since
  e8cd5e6.
  - It saves 6.4–7.6 % of the MoE's time at these chunks, against 4.9 % of its bytes.
  - For a +200 turn that is 9.1 ms. The head's last-token GEMV adds 0.36 ms, for 9.5 ms in all.
  - The A/B measured −7.0 to −12.0 ms in its eight +200 times to first token (−3.2 to −5.4 %), a mean
    of −9.4 ms.
  - The k = 1 edit prefill's −3.6 % is −491 ms over 54 turns of mixed sizes, −9.1 ms a turn.
- **Left on the table:** the last layer for the last row only. It is exact, but worth ~2–4 ms a turn,
  ≤ 0.4 % end to end (below).

**Where the code is.** Branch kernels-pmoe 175d59a and b63b3b4 extend `bench_kernels moereal`
([bench_kernels.cu](../engine/kernels/tests/bench_kernels.cu)):
- `PF_SCALES=<layer>` takes that layer's real scales for all 384 experts;
- `PF_CODE="0 1"` runs raw against coded, with the bits compared;
- the floors are per expert;
- it reports GB/s and the MoE's time summed over a case's layers;
- `PF_CGRP` splits down into the engine's column groups;
- `PF_TURNS` reads the engine's own route dumps (`DSGB10_ROUTE_DUMP` on branch runner-ttft).

The engine is unchanged.

**The method.**
- **The runs.** Ring parts on rank 0 alone, PREEMPT=0, from bins built at b63b3b4. A run takes 15–40 s.
- **The routes.** Rank 0's dumped routes of golden/long (`/data/dsgb10/kernels/routes/long`):
  - the encoder: rows [0, M) of the second 8192-row chunk, in layers 0–19;
  - the replay: its 128 rows, in layers 20–39;
  - an exact continuation of M ≤ 128 rows: the encoder's rows [0, M) in layers 0–19, and the replay's
    first M rows in layers 20–39.
- **The weights.** The weights are random. The scales are layer 20's real ones, for every layer, from
  rank 0's file. Their code holds 97.56 % of w1's groups, 98.38 % of w3's and 98.87 % of w2's.
- **The timing.** Per layer, rounds of bucketing (untimed), up, down and the whole `moe_prefill` call
  (the shared expert inline), each timed by events. The figures are medians of 7.
  - The combine runs inside down at every M up to 2,000 (`moe_prefill_fuses`).
  - The engine's ffn_p splits down into 4 column groups, with the reduce-scatter under them. The main
    table runs down as one call; the column groups are measured apart (below).
- **The floor.** The touched experts' bytes at 235 GB/s:
  - raw, 4.70 MB an expert on a rank: w1 + w3 3.13 MB, w2 1.57 MB;
  - coded, 4.47 MB: the code blocks and the raw groups in place of the scales;
  - the MoE's floor adds the shared expert's 8.9 MB.

**The numbers** (ms a layer, the mean over the case's 20 layers; "sum" is over them):

| case | M | experts a layer | up raw → coded | down raw → coded | MoE raw → coded | % of floor, raw / coded | GB/s, raw / coded | sum raw → coded |
|---|---|---|---|---|---|---|---|---|
| replay | 128 | 151.6 | 2.109 → 1.957 | 1.048 → 0.969 | 3.252 → 3.024 | 94 / 97 | 222 / 227 | 65.0 → 60.5 |
| encoder | 44 | 77.5 | 1.092 → 1.015 | 0.560 → 0.503 | 1.739 → 1.606 | 91 / 94 | 215 / 221 | 34.8 → 32.1 |
| encoder | 128 | 141.2 | 1.968 → 1.826 | 0.985 → 0.925 | 3.047 → 2.846 | 94 / 96 | 221 / 225 | 61.0 → 56.9 |
| encoder | 200 | 161.9 | 2.251 → 2.095 | 1.131 → 1.056 | 3.489 → 3.260 | 94 / 96 | 221 / 225 | 69.8 → 65.2 |
| encoder | 350 | 186.2 | 2.624 → 2.442 | 1.323 → 1.222 | 4.051 → 3.768 | 93 / 95 | 218 / 223 | 81.0 → 75.4 |
| encoder | 728 | 257.9 | 3.676 → 3.420 | 1.898 → 1.765 | 5.749 → 5.353 | 90 / 92 | 212 / 217 | 115.0 → 107.1 |
| encoder | 1,368 | 297.1 | 4.391 → 4.083 | 2.396 → 2.249 | 7.057 → 6.604 | 85 / 86 | 199 / 202 | 141.2 → 132.1 |
| encoder | 2,000 | 319.5 | 4.882 → 4.580 | 2.758 → 2.629 | 7.968 → 7.524 | 81 / 81 | 190 / 191 | 159.4 → 150.5 |
| encoder | 4,096 | 335.3 | 6.069 → 5.761 | 4.148 → 4.072 | 12.697 → 12.321 | 53 / 52 | 125 / 122 | 253.9 → 246.4 |
| encoder | 8,192 | 358.5 | 8.414 → 8.133 | 6.301 → 6.269 | 19.375 → 19.318 | 37 / 35 | 87 / 83 | 387.5 → 386.4 |

The bits were the same in every case. Each case's first layer was compared with the default kernels on
the raw scales.

**Down in the engine's 4 column groups** (`PF_CGRP="1 4"`, coded, a second run).
- The groups cost down 0.04–0.08 ms a layer, and the MoE call 0.04–0.09 ms:
  - replay: 3.038 → 3.087 ms;
  - 44 rows: 1.616 → 1.660;
  - 200 rows: 3.270 → 3.319;
  - 728 rows: 5.391 → 5.443;
  - 2,000 rows: 7.508 → 7.594.
- Down then runs at 89–94 % of its floor up to 728 rows, against 92–99 % as one call. That is ~2 ms of
  a +200 turn.
- The groups are what lets each group's reduce-scatter run under the next group's down. One call
  would leave the reduce-scatter exposed instead, so the groups are not a lever by themselves.
- The bits were the same.

**Exact continuations: 40 layers at 1, 44 and 100 rows** (a third run). Layers 0–19 take the encoder's
routes and layers 20–39 the replay's first M rows. The three variants are the engine's (coded, down in
4 column groups), raw in the same groups, and coded with down as one call.

| rows | experts a layer | MoE ms a layer: coded / raw / coded, one call | % of floor (coded) | sum over 40 layers: coded / raw |
|---|---|---|---|---|
| 1 | 6.0 | 0.262 / 0.269 / 0.241 | 58 | 10.5 / 10.8 |
| 44 | 84.6 | 1.809 / 1.943 / 1.753 | 91 | 72.4 / 77.7 |
| 100 | 131.8 | 2.720 / 2.904 / 2.668 | 94 | 108.8 / 116.2 |

- At one row the prefill kernels are bound by latency, not bytes:
  - up runs at 80 % of its floor and down at 45 % (60 % as one call);
  - the call takes 0.26 ms against 0.15 ms of bytes.
- From 44 rows the exact passes stream as the encoder's do.
- The bits were the same.

**What the numbers say.**
- **Up to 728 rows, the bytes set the time.**
  - Layer by layer, up runs at 90–99 % of its floor, down at 80–100 % and the whole call at 81–97 %.
  - The whole call takes ~0.1 ms a layer more than up and down: the bucketing, the shared expert and the
    gaps between launches.
- **Beyond 728 rows, the bytes stop setting the time.**
  - From 1,368 rows, down falls to 84 % and then 77 % of its floor. The floor does not count the fused
    combine's traffic: at 2,000 rows a layer writes 41 MB of fp32 output and reads 20 MB of the shared
    expert's rows (~0.26 ms).
  - At 4,096 and 8,192 rows the combine runs apart, and the MoE is in the long-prefill regime of
    [What remains for prefill](#what-remains-for-prefill).
- **The experts touched.** On this text a chunk's rows pick, of 384 experts a layer:
  - 78 at 44 rows, 141–162 at 128–200, 258 at 728 and 320 at 2,000, in layers 0–19;
  - 92 at 44 rows and 152 at 128 in layers 20–39 (the replay's rows).

  The time follows that count, not the rows: 44 rows of layers 0–19 cost half of 200 rows (34.8
  against 69.8 ms).
- **The code saves more than its bytes,** as it does at 6–16 rows in decode
  ([mscale](#the-experts-scales-coded-and-the-exact-compression-bundles-ab-2026-10-12)).
  - It saves 6.4–7.6 % of the MoE's time at 44–1,368 rows, against 4.9 % of the floor.
  - The coded call runs closer to its floor: 92–97 % of it up to 728 rows, against 90–94 % raw.
  - One guess, unchecked (ncu needs admin rights here): a raw step reads a lone 32-byte sector of
    scales a warp from a separate array, which may cost the DRAM more than its bytes. A code word
    shares its sector with 7 other steps.
- **Against the runner's turns.** These are its pre-bundle build (bff5d5d,
  `/data/dsgb10/bench/ttft/bff5d5d/`), so the raw column applies. The time is the server's split of
  each turn, as the median (range) of 8 turns. "Rest" is the time left beside the MoE.

  | turn | pass | pass time ms | MoE ms | MoE share | rest ms | time to first token ms |
  |---|---|---|---|---|---|---|
  | +44 | exact, 40 layers × 44 rows | 136.9 (131.6–146.5) | 77.7 (in 4 groups) | 57 % | 59.2 | 144.1 (136.3–154.6) |
  | +200 | encoder, 200 rows | 119.8 (113.8–126.3) | 69.8 | 58 % | 50.0 | 217.4 (212.9–232.2) |
  | +350 | encoder, 350 rows | 143.4 (136.9–152.5) | 81.0 | 56 % | 62.4 | 248.4 (228.1–255.3) |
  | +728 | encoder, 728 rows | 191.1 (179.8–195.2) | 115.0 | 60 % | 76.1 | 298.6 (275.7–303.2) |
  | +1,368 | encoder, 1,368 rows | 251.8 (242.7–256.2) | 141.2 | 56 % | 110.6 | 356.6 (343.4–361.7) |
  | each turn of 128 or more | the replay, 128 rows | 93.2–99.2 (85.0–104.3) | 65.0 | 66–70 % | 28–34 | |

  - Beside the MoE, a layer takes ~1.5 ms at 44 rows (exact) and ~2.5 ms at 200 rows, rising to ~5.5 ms
    at 1,368. The replay's layers take 1.4–1.7 ms each.
  - That share is the attention side, the boundaries and the collectives: the runner's breakdown.
- **What the code saves a turn** (encoder + replay, or the exact pass): 5.3 ms at +44, 9.1 at +200, 10.2 at
  +350, 12.5 at +728 and 13.6 at +1,368.

**On the turns' own routes** (a fourth run). The runner dumped the routes of its edit turns from a
post-bundle build (8805de4: master 0eb3a7d with NVTX ranges; `/data/dsgb10/bench/ttft/routes`, rows
24–53 of its `turns_routes.log`: the long phase at 16–32k, 6 turns of each size). `moereal` ran each
pass on its own routes (`PF_TURNS`, 5 rounds), in the three variants of the exact passes. The pass times
are the runner's medians on that build. The MoE is summed over the pass's layers, the mean over the 6
turns, in ms; the experts a layer on the long text are in brackets.

| turn | pass | experts a layer | MoE coded, 4 groups | raw, 4 groups | coded, one call | floor, coded | % of floor | pass time | MoE share |
|---|---|---|---|---|---|---|---|---|---|
| +44 | exact, 40 layers | 92.3 (84.6) | 78.7 | 84.3 | 76.3 | 71.7 | 91 | 136.0 | 58 % |
| +200 | encoder | 190.9 (161.9) | 76.6 | 82.5 | 75.6 | 73.4 | 96 | 117.8 | 65 % |
| +350 | encoder | 227.3 (186.2) | 91.7 | 98.5 | 90.5 | 87.2 | 95 | 138.8 | 66 % |
| +728 | encoder | 282.5 (257.9) | 116.7 | 124.6 | 115.3 | 108.2 | 93 | 183.9 | 63 % |
| +1,368 | encoder | 299.0 (297.1) | 133.7 | 141.2 | 131.8 | 114.5 | 86 | 242.5 | 55 % |
| 128 or more | the replay | 145.8–151.7 (151.6) | 59.2–61.3 | 63.5–65.8 | 58.3–60.3 | 56.2–58.4 | 95 | 92.6–95.0 | 64–65 % |

- **An edit turn's rows pick more experts than the long text's.** That is 9 % more at +44, 18 % at +200
  and 22 % at +350, then 10 % at +728 and none at +1,368, where both near 300 of 384. The replay's 128
  rows pick as many as before.
- **So the MoE is a larger share of a pass:** 63–66 % of an encoder pass of 200–728 rows, against 56–60 %
  estimated on the long text. Beside it, a layer takes 1.4 ms at +44 and 2.1–3.4 ms at 200–728 rows. The
  replay's layers take 1.6–1.7 ms each.
- **Still at the floor:** 93–96 % of the coded floor up to 728 rows, 91 % for the exact +44 pass and 86 %
  at 1,368 rows. The verdict stands: no kernel lever.
- **The code saves 5.6 ms a +44 turn,** 10.1 at +200, 11.1 at +350, 12.5 at +728 and 12.0 at +1,368, close
  to the long text's 5.3–13.6.
- **Over the floor:** 7.0 ms a +44 turn (4.9 % of its 142.7 ms to first token), 6.2 at +200 (2.9 %), 7.5 at
  +350 (3.1 %), 11.3 at +728 (3.9 %) and 22.3 at +1,368 (6.5 %).
- **Down in one call** would save another 2.0–2.9 ms a turn (0.8–1.7 % of a time to first token), but only
  if the reduce-scatter that the groups hide were hidden some other way.
- **The bits were the same** in every pass.

**The last layer for the last row only: left on the table.**
- **Who reads layer 39's output.** Only the head reads it, and only for the pass's last row.
  - DSpark's `main_hidden` takes the inputs of layers 37–39 (`kSpecLayer0`), not layer 39's output.
  - Layer 39's keys come from its input. So its attention keys stay as they are, but its queries,
    attention, wo_a, wo_b, router, MoE and boundary could run for one row.
- **Bits.** It is exact if a row's arithmetic does not depend on the chunk's other rows.
  - Every chunking of a prompt gives the same tf hash, which says it does not. But no chunking goes
    down to one row.
  - So a test would compare a pass's last row with the same row run alone: the shared expert's GEMMs,
    the router and the attention included.
  - The one-row call must stay on the prefill kernels. The decode path's are faster at one row (0.16 ms
    in mscale), but they sum in another order.
- **Size.**
  - The MoE alone: layer 39 takes 1.88 ms at 44 rows and ~3.3 ms at the replay's 128, against 0.27 ms
    at one row. That saves 1.6 ms a +44 turn and ~3.0 ms a turn of 128 or more.
  - With layer 39's attention side for the other rows, ~2–4 ms a turn: 1–2 % of a time to first token,
    ≤ 0.4 % end to end.
- **Why not.** Under the 2 % bar.

**What would make the MoE cheaper at small chunks.** Every lever touches fewer bytes, and none of them
is a kernel's:
- **Fewer bytes an expert.** The scales are already coded. The FP4 values compress by 4.6 % at best
  ([Lossless weight compression](#lossless-weight-compression-a-feasibility-study-2026-10-12)). At the
  stream's rate a decoder would need ~470 G symbols/s, about twice what the FP8 decoder reached there
  (4.0–4.6 G/s an SM).
- **Fewer experts.** The routing is the model's.
- **Fewer rows in the replay.** The replay's 128 rows are CED's definition.

### A small turn's time to first token (2026-10-12)

**The question.** A small agentic turn's time to first token: what it is made of, how far each part is from its
floor, and what could come off it with the same bits.

The bar has three parts:
- a lever or bundle that projects 2 % or more end to end on the sampled edit bench;
- no bench type or sparkDash catalog more than 2 % slower;
- the reference hashes and `serve_test.py --ced` unchanged.

Why it matters: the prefill turns of the sampled edit bench (r2: 126 requests, 165.8 s of server time) are
- 80 CED turns of 159–1,851 new tokens (median 369): 22.6 s of prefill, 13.6 % of the bench;
- 40 exact continuations of 24–86 (median 27): 4.8 s, 2.9 %;
- 3 turns of 11,632 tokens (2.6 %) and 3 of 8 rows or fewer.

2 % of the bench is 3.3 s, or ~27 ms a prefill turn.

**Verdict.** A finding. No lever reaches the bar alone. A bundle of two kernels and two collective changes
reaches it only if both kernels hit their targets.

- **Where the time goes.** A +200 turn at 26k context takes 214 ms to its first token.
  - 140 ms of it is the routed experts, at 93 % of their byte floor
    ([the kernels line's study](#the-prefill-moe-at-small-chunks-2026-10-12)).
  - The other 74 ms:

    | part | ms |
    |---|---|
    | the dense weights' GEMMs | 30 |
    | the mHC boundaries | 18 |
    | the collectives' exposed waits | 10 |
    | sparse attention | 4.5 |
    | Engram, small kernels and launch gaps | 2 each |
    | the head | 1 |
    | host work around the GPU's | ~5.6 |
- **Two parts are far from their floors.** The levers are there.
  - **The boundaries** run at 14 % of their floor. Each of a turn's 80 takes ~0.22 ms, whatever the rows up to
    ~350 a rank. It is a chain of 80 dependent block steps in one CTA or a few.
  - **The dense GEMMs** stream at 33–190 GB/s at these rows:
    - the router gate at 33 GB/s, on 3 CTAs;
    - the shared expert's w1/w3 at 84, on 9;
    - wq_a + wkv, wq_b, wo_a and wo_b at 75–188.
- **Each lever end to end on r2** (all requests; thinking off / max in brackets):
  - L1, the boundary's built tuning: 0.39 % (0.41 / 0.31);
  - L2, a boundary that streams at DRAM speed with the same bits: 0.86–0.98 %;
  - L3, small-M tiles for the dense GEMMs: 0.67–0.82 %;
  - L9, each rank's own weight slices read once at small M: ~0.34 % on top of L3;
  - the replay's records by all-to-all: 0.05 %;
  - eight column groups from 512 rows: 0.10 %;
  - launch gaps: ≤ 0.07 %;
  - the last layer for the last row only (the kernels line's): 0.19 %.
- **The bundles.** L2, L3 and the two collective changes together project 1.68–1.95 % (2.01 % with thinking off
  at the targets). Adding L9 brings that to 2.29 %.
  - At the targets, a +200 turn's first token comes 27 ms sooner: 214 → ~187 ms, −12 %.
  - A +44 turn's comes 23.5 ms sooner: −17 %.
- **What would settle it.** The two kernels' figures are targets, not measurements. Before the bundle is built,
  each kernel needs a prototype on one box, its bits checked. The bundle reaches the bar if:
  - the boundary takes ≤ 60 µs at 32 rows;
  - the small-M GEMMs reach ≥ 80 % of DRAM bandwidth.

  Both are the kernels line's kernels.

**The method.**

The engine is master 0eb3a7d (the coded scales) with profiling hooks, built from 8805de4 (2c85d77 on branch
runner-ttft, after a rebase onto c748dd4). The hooks are compiled only with `-DDSGB10_NVTX=ON`:
- NVTX ranges per turn, chunk, layer and part;
- `DSGB10_ROUTE_DUMP`: rank 0's routes for every pass;
- serve's `--profile-from-req N`: request N opens nsys's capture range on every rank.

The turns:
- `ttft_turns.py` drives a test server at production's `--max-ctx 262144`. Each ring part runs through
  `ttft_part.sh`, with PREEMPT=0.
- The new tokens are the edit bench's own: r2's dumped prompts, each request's new tokens laid end to end as a
  stream.
- Each part sends 54 requests with 4 tokens out each:
  - a cold first prompt of 1,364 tokens;
  - 11 forks from its snapshot (the other tasks' first prompts);
  - 2 rounds of +44, +200, +350, +728 and +1,368 at 1.4–10k context;
  - a cold 16k prompt of synthetic text;
  - 6 rounds of each size at 16–33k ("long": the tables below);
  - one +11,632 turn at 44k.

Five parts ran on 2026-10-12, between 20:28 and 20:40:
- the route dump;
- nsys on rank 0;
- the uninstrumented build ("base");
- nsys on rank 2;
- the NVTX build without nsys.

In the long phase, against base's median time to first token, NVTX adds 0.3–1.7 ms a turn, nsys on rank 0
1.9–5.4 ms and nsys on rank 2 0.8–5.4 ms.

The tools, in engine/server/tools:
- `ttft_join.py` joins the client's rows with rank 0's split of each turn.
- `ttft_prof.py` splits each turn by part and kernel class.
- `ttft_roles.py` names each main-stream operation by its role, with its own time and the gap before it. It splits
  each gap into waiting on the comm stream and the rest.
- `ttft_routes.py` lists the experts each pass touched, and their bytes at 235 GB/s.
- `ttft_breakdown.py` makes the table below, and `ttft_levers.py` the projection.

The data is in /data/dsgb10/bench/ttft, with the tables in its ana/ directory.

An operation's own time runs from the later of its start and the previous main-stream operation's end, so PDL's
overlap counts once. A turn's own times and gaps add up to the main stream's span, from its first operation to the
token.

**The breakdown.** ms a turn on rank 0, long phase, the medians of 6 turns.
- The GPU rows come from the nsys part; the time to first token and the host rows from base.
- Rank 2 is within 3 ms of rank 0 in every row except the two collective rows (below).

| part | +44 (exact, 40 layers) | +200 | +350 | +728 | +1,368 | floor at +200 |
|---|---|---|---|---|---|---|
| time to first token | 137.8 | 214.1 | 234.9 | 284.5 | 340.5 | |
| routed experts: up, down, bucketing | 79.8 | 140.3 | 157.2 | 183.1 | 199.4 | 130.2 (the touched experts) |
| wq_a + wkv, wq_b, wo_a, wo_b; the compressor's and indexer's GEMMs | 17.3 | 20.9 | 21.4 | 25.1 | 33.4 | 11.9 |
| router gate, shared expert | 8.7 | 8.8 | 9.0 | 10.1 | 12.4 | 2.2 |
| mHC boundaries (80 a turn) | 18.0 | 17.9 | 18.0 | 18.3 | 19.1 | 2.5 |
| sparse attention | 1.3 | 4.5 | 6.1 | 10.4 | 17.3 | |
| index scans | 0.4 | 0.6 | 0.7 | 0.9 | 1.2 | |
| collectives on the main stream | 2.2 | 6.0 | 7.0 | 11.4 | 17.7 | |
| waits for the comm stream's tails | 1.5 | 3.6 | 5.1 | 9.3 | 15.3 | |
| Engram (layers 1 and 14) | 1.7 | 1.9 | 2.0 | 4.1 | 10.8 | 1.3 (its weights) |
| small kernels | 1.6 | 2.0 | 2.1 | 2.8 | 4.2 | |
| launch gaps | 2.2 | 2.2 | 2.2 | 2.3 | 2.3 | 0 |
| head | 1.1 | 1.1 | 1.2 | 1.1 | 1.1 | ~1.1 (its coded bytes) |
| state operations (host) | 2.5 | 2.8 | 2.5 | 2.5 | 2.5 | |
| the rest of the server's time (host) and the client | 2.5 | 2.8 | 3.1 | 3.1 | 3.2 | |

**A layer.** µs, rank 0 / rank 2; the dense GEMMs also in GB/s of their bytes.
- Every turn of 128 or more pays for the replay's layers. They are its fixed cost.
- The +44 column is an exact layer over 44 rows. The encoder column is the +200 turn's.

| role | exact +44 | replay | encoder +200 |
|---|---|---|---|
| routed experts: up, down, bucketing | 1,995 / 1,957 | 3,053 / 3,033 | 3,939 / 3,937 |
| router gate | 119 / 106 (33 GB/s) | 119 / 106 (33) | 119 / 113 (33) |
| shared expert: w1 + w3, w2 | 92 / 84 (80, 162) | 90 / 82 (84, 145) | 98 / 92 (81, 113) |
| wq_a + wkv | 65 / 61 (142) | 68 / 63 (136) | 68 / 65 (135) |
| wq_b | 62 / 60 (170) | 76 / 75 (138) | 86 / 81 (122) |
| wo_a | 48 / 48 (176) | 60 / 56 (140) | 112 / 101 (75) |
| wo_b: every rank's slice, for own rows | 223 / 219 (188) | 246 / 240 (171) | 258 / 256 (163) |
| sparse attention | 31 / 29 | 88 / 78 | 135 / 123 |
| the two boundaries | 447 / 411 | 444 / 401 | 447 / 416 |
| the two all-gathers | 54 / 145 | 43 / 158 | 139 / 304 |
| comm-stream tails: the last reduce-scatter, the all-to-all | 38 / 60 | 52 / 87 | 127 / 151 |
| everything else, gaps included (rank 0) | ~220 | ~135 | ~315 |
| the layer | 3,393 / 3,381 | 4,473 / 4,502 | 5,844 / 5,950 |

**Each part against its floor.**

**The routed experts** run at 87–93 % of the touched experts' bytes at 235 GB/s up to 728 rows, and at 84 % at
1,368. That is in place; standalone, the kernels line measured 90–97 %. Their verdict holds: there is no kernel
lever at small chunks, and the whole gap is ≤ 1.2 % end to end. That gap is all `dgrp` and `tord` could take at
small chunks.

**The boundaries** take 0.22–0.25 ms each on rank 0 (0.20 on rank 2), at 8–344 rows a rank.
- The floor at 32 rows is 0.03 ms: 3.3 MB of activations and 2.9 MB of hc_fn's parts at 235 GB/s.
- Why: a CTA walks the 80 blocks of 64 columns in order, and every CTA reads all of hc_fn's parts. Each block's
  parts are loaded one block ahead, into registers. The walk's latency sets the time, not the bytes.
- At 32 rows a rank, one CTA does the whole boundary.
- A +200 turn's boundaries run at 14 % of their floor.

**The dense GEMMs** stream at 33–188 GB/s:
- the router gate is a 128×128 tile over 384 outputs, so 3 CTAs do it;
- the shared expert's w1/w3 make 9 tiles of 64 SwiGLU columns;
- wq_a + wkv is every rank's 1,792 rows for own rows, in 14 tiles;
- wo_b reads every rank's slice: 42 MB a layer, against 10.5 MB for one slice.

At +200 they take 29.7 ms against 14.1 ms of bytes: 47 % of their floor.

**The collectives.** On rank 0, a replay layer exposes ~100 µs of collective time:
- the two all-gathers, 27 and 16 µs;
- the index's top-k all-gather, 7 µs;
- the comm stream's tails (the last column group's reduce-scatter, the all-to-all), 52 µs.

That is about what the ring takes for collectives of these sizes ([ring.md](ring.md)'s large payloads: an
all-gather with 256 KB of output takes 27 µs).

Rank 2 waits 2.5 times as long, 258 µs. It reaches each collective earlier and waits there for rank 0, whose
kernels run 5–10 % slower: the boundary takes 225 against 203 µs, the gate 119 against 106. A turn ends when its
slowest rank does, so rank 0's waits are the ones that cost.

Two larger items:
- The replay's records go to every rank by all-gather: 7 MB, 1.45 ms a turn of 128 or more. Each rank needs a
  quarter of them.
- At +728 and +1,368, the last column group's reduce-scatter is exposed for 0.37 and 0.66 ms a layer. At 728
  rows a group is 3.7 MB of fp32.

**Sparse attention** takes 88 µs a replay layer, 135 at 200 rows and 431 at 728. It is the attention line's.

**The launch gaps** come to 2.2 ms a turn over ~1,450 main-stream operations, 1.5 µs each. About 0.5 ms of that
sits around the window copies, which are `cudaMemcpyAsync` calls that PDL does not chain.

**The head** takes 1.1 ms: its coded bytes.

**Engram.**
- At small turns its time is its weights. Each rank reads every rank's wkv slice for its own rows: 157 MB a layer,
  0.67 ms at 235 GB/s.
- From +728, layer 1's all-gather waits for the slowest rank's store lookups: 5.8 ms on rank 0 at +1,368.

**The host.** The state operations take 2.5–2.8 ms, from the request to the tokens' broadcast. The rest of the
server's time is 1.7–2.4 ms, and the client adds 0.7–0.8 ms.

**The encoder/replay boundary** has nothing to overlap:
- the replay needs the encoder's layer 19 for its rows, and layer 20's compressor for every row;
- in between there is only `ced_project` (0.2 ms) and the records' all-gather.

**The levers.** ms a turn at +44 / +200 / +350 / +728 / +1,368. End to end on r2: all requests, with thinking
off / max in brackets.

| lever | why the bits stay | ms a turn | end to end | owner, state |
|---|---|---|---|---|
| L1: the boundary's rows per CTA by the chunk, prefetch 2 (`hc_boundary_tuning`; bench ratios 0.66 at 32 rows, 0.92 at 500) | the same per-row sums at every R (`test_pattn mhcrows`) | 6.1 / 5.6 / 5.2 / 4.6 / 4.2 | 0.39 % (0.41 / 0.31) | kernels line; built (with the MoE knobs), off |
| L2: a boundary at 85 % of DRAM bandwidth, or no faster than 60 µs; up to 512 rows a rank | split by column halves and copies. Each chain of sums stays in one thread. The lanes' and warps' combines keep their order, done by the last CTA | 13–16 / 13–14 / 13–14 / 11–12 / 8–9 | 0.86–0.98 % | kernels line; not built |
| L3: small-M tiles for the dense GEMMs: enough CTAs for every SM, at 75–85 % of DRAM or their MMA floor | a tile's shape does not change any output's K chain: one thread's accumulator through the k32 blocks, in order. `test_kernels p_fp8` already checks wo_b in every tile variant | 5.6–7.8 / 9.2–11.3 / 9.2–11.3 / 10.4–12.3 / 14.8–16.5 | 0.67–0.82 % | kernels line; not built |
| L3a: the router gate alone | as L3 | 4.0 a turn | 0.29 % | part of L3 |
| L9: at small M, each rank reads its own slice once, instead of every rank's for its own rows: wq_a + wkv, the gate, wo_b (with the reduce-scatter that `sum4_bf16_p` stands for) and Engram's wkv | the same sums in rank order ([Prefill](#prefill)) | 5.4 / 5.4 / 5.4 / 5.4 / 2.2 (estimates) | 0.34 %, on top of L3 | not built; the reduce-scatter at these sizes needs measuring |
| L4a: the replay's records sent only to the ranks that run them (an all-to-all, not an all-gather) | the same bytes to the same rows | 0 / 1 / 1 / 1 / 1 | 0.05 % | runner; not built |
| L4b: 8 column groups instead of 4 from 512 rows. The last reduce-scatter is half as long; down runs ~0.06 ms a layer slower | the same sums | 0 / 0 / 0 / 2.5 / 5.4 | 0.10 % | not built |
| L5: the launch gaps (graphs, PDL'd window copies) | the same kernels, launched differently | ≤ 1 | ≤ 0.07 % | not built |
| L6: the shared expert beside the routed up (`MoePCfg::side`) | the same kernels, on a side stream | 2 / 2 / 2 / 2 / 1 at most | 0.13 % at most. The kernels line measured −0.9 % of the MoE call at 128 rows, inside the ±3 % spread | merged, off; no gain |
| L7: the last layer for the last row only | see [the kernels line's study](#the-prefill-moe-at-small-chunks-2026-10-12) | 1.6 / 3 / 3 / 3 / 3 | 0.19 % | kernels line; not built |

The targets:
- **L2:** the boundary's bytes at 85 % of 235 GB/s, plus 5 µs. The bytes are ~110 KB a row of activations, plus
  hc_fn's parts. The low end puts no boundary under 60 µs.
- **L3:** for each GEMM, the larger of its bytes at 85 % (or 75 %) of 235 GB/s and its MMA work at 150 TFLOPS FP8
  or 75 bf16. Against the measured sum of the gate, shared expert, wq_a + wkv, wq_b, wo_a and wo_b, ms a layer
  (the saving tapers to nothing at 4,096 rows):

  | rows | measured | saved at 85 % | saved at 75 % |
  |---|---|---|---|
  | 44 | 0.61 | 0.19 | 0.14 |
  | the replay (128) | 0.66 | 0.25 | 0.19 |
  | 200 | 0.74 | 0.32 | 0.27 |
  | 350 | 0.77 | 0.32 | 0.27 |
  | 728 | 0.99 | 0.37 | 0.33 |
  | 1,368 | 1.53 | 0.58 | 0.55 |
- **Which turns each lever touches.** L1, L2 and L3 act on every prefill pass. L4a acts on turns of 128 or more,
  and L4b on chunks of 512–2,520 rows.

The bundles, end to end on r2:

| bundle | all | off | max |
|---|---|---|---|
| L2 + L3 + L4a + L4b, at the targets | 1.95 % | 2.01 % | 1.56 % |
| the same at the low ends (L2 ≥ 60 µs, L3 at 75 %) | 1.68 % | 1.73 % | 1.35 % |
| L2 + L3 + L9 + L4a + L4b | 2.29 % | 2.36 % | 1.83 % |
| L2 + L3 + L4a + L4b + L5 + L7 | 2.21 % | 2.28 % | 1.77 % |
| L1 + L3 + L4a + L4b (the built boundary tuning instead of a new kernel) | 1.36 % | 1.40 % | 1.10 % |

Each lever is bounded by rows:
- a long prefill's 8,192-row chunks and the decode step keep today's kernels;
- cold prompts short enough for chunks of ≤ 2,048 rows would take the new ones, which an A/B must cover.

**Dead ends and corrections.**
- **PDL overlap counted twice.** ttft_prof.py's first split summed each kernel's whole duration. With PDL a kernel
  starts while the one before it runs, so the classes summed to more than the main stream's busy time: 163
  against 135 ms at +44. It now counts own time.
- **Rank 2 as one turn.** Its split of a worker rank's turns cut them at the tokens command. With CED a worker's
  turn is the prefill command and what follows it, so rank 2's first analysis came out as one 16-second "turn".
  The split now cuts at the prefill command.
- **The join's arguments.** ttft_join.py took `--json OUT` for a log when it came first. Fixed.
- **A stale build.** The first parts ran a build from before the coded scales (bff5d5d), so the study reran on
  0eb3a7d. The old data is in /data/dsgb10/bench/ttft/bff5d5d; the kernels line's comparison uses it.
- **One late all-gather.** One replay layer on rank 0 showed a 255 µs all-gather, a rank arriving late. The
  medians over turns leave it out.

**Reproduce.**
1. Build with `-DDSGB10_NVTX=ON`, and once without.
2. Run the five parts, each as `OWNER=runner PREEMPT=0 engine/server/ring.sh bash
   engine/server/tools/ttft_part.sh LABEL OUT BIN`:
   - the route dump: `ROUTES=OUT/routes`;
   - rank 0's profile: `WRAP="nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi
     --capture-range-end=stop -s none -o OUT/nsys0"`, with `--profile-from-req 1`;
   - rank 2's profile: as rank 0's, with `WRAP_RANK=2` and `NSYS_REMOTE=...`.
3. Export each report with `nsys export -t sqlite`.
4. Run, in order:
   - `ttft_join.py`;
   - `ttft_roles.py CAPTURE --turns turns_LABEL.log --json`;
   - `ttft_routes.py OUT/routes turns_routes.log --json`;
   - `ttft_breakdown.py OUT/ana`;
   - `ttft_levers.py`.

### The split mHC boundary: a prototype (2026-10-12)

**The question.** Can the mHC boundary at small chunks run near its byte floor with the same bits? This is
lever L2 of [the time-to-first-token study](#a-small-turns-time-to-first-token-2026-10-12). The target was
≤ 60 µs at 32 rows a rank (225 µs today) and ≥ 85 % of DRAM bandwidth where the bytes bind. Only if this
and the small-M GEMM tiles both reached their targets was the bundle to be built.

**Verdict.** It misses the target: 88 µs at 32 rows, not ≤ 60. The bits are the same at every row count
tested, and it is 2.5 times faster than today's kernel up to 56 rows. So the gate stopped the work here,
and the small-M GEMM prototype was not started.

- **End to end on r2:** 0.69 % (0.72 % thinking off, 0.55 % max), against 0.86–0.98 % at the target.
- **The bundles with it as measured** (the GEMM tiles still at their targets):
  - L2 + L3 + L4a + L4b: 1.66 % (1.71 % off);
  - with L9: 2.00 % (2.06 % off);
  - with L9, L5 and L7: 2.27 %.
- **Merged, off:** `hc_boundary_split(max_rows, rows_per_cta)` turns it on, for chunks of up to 512 rows a
  rank. Today's kernel stays the default.

**The kernel** (`hc_bsplit_kernel` in prefill_mhc.cu, beside the default one):
- **Its split.** Each row group of R = 4, 8 or 16 rows runs on two CTAs, one per half of each block's 64
  columns. The default kernel runs it on one CTA of 32 rows.
- **Its loads.** The stream and hc_fn's parts come through shared memory 3–4 blocks ahead (cp.async). The
  default kernel loads the parts into registers one block ahead.
- **Why the bits stay.** Every chain stays in one thread, in the default kernel's order:
  - the projection's sums for each (copy, half) warp slot, over the 80 blocks;
  - y² and x² for each row and 8-column slot.
- **Its finish.** Each CTA leaves its partials in a scratch. The second CTA of a group to finish sums them
  in the default kernel's orders (warps 0..7, the lanes' butterflies), then runs the Sinkhorn and the norm.
- **The test.** `test_pattn mhcrows` compares every output bit for bit with the default kernel at
  1–512 rows, R = 4, 8, 16 and by-rows, in its four option sets. It passes.

**Measured.** `bench_pattn --only mhcs` (e0c3d24) on rank 0, under the ring lock. µs per boundary,
cold L2; in brackets, GB/s of its bytes:

| rows a rank | default (32 rows a CTA) | built tuning (R4, prefetch 2) | split R4 | split R8 | split R16 |
|---|---|---|---|---|---|
| 8 | 220.9 (18) | 141.0 | 83.7 (47) | 96.0 | 114.8 |
| 16 (+44 exact) | 225.3 (22) | 143.1 | 85.6 (58) | 98.0 | 134.9 |
| 32 (the replay) | 222.9 (32) | 145.4 | **87.7 (80)** | 98.1 | 136.9 |
| 56 (+200) | 225.1 (45) | 147.1 | 93.9 (108) | 104.0 | 137.2 |
| 88 (+350) | 225.3 (63) | 150.2 | 110.3 (129) | 120.5 | 145.5 |
| 128 | 232.9 (83) | 153.3 | 171.7 | 126.7 (153) | 153.3 |
| 184 (+728) | 229.2 (116) | 162.5 | 179.9 | 143.5 (185) | 167.6 |
| 344 (+1,368) | 256.1 (184) | 305.8 | 324.3 | 247.5 | 240.2 (196) |
| 512 | 350.8 (195) | 447.2 | 477.8 | 393.9 | 388.9 (176) |

**Where it stops.**
- **Up to 56 rows, the time hardly moves (84–94 µs).** That is ~1.05 µs for each of the 80 block steps. A
  CTA's step is still a chain of dependent phases, two barriers apart: the y tile, then the chains and the
  projection's MMAs.
- **More CTAs help (R4 beats R8 and R16), but not enough.** Each CTA streams its half of hc_fn's parts,
  18 KB a block, from L2.
- **From 344 rows the bytes bind.** It runs at 196 GB/s there, 83 % of 235. At 512 rows it is slower than
  the default, so it should not be used above ~344.

**What could close the gap** (not tried):
- also split by copy: 8 CTAs per row group, 4.6 KB of parts a block each, so the pipeline can run 8 or
  more blocks deep;
- software-pipeline the phases across blocks: the y tile of block cb + 1 beside block cb's MMAs;
- give the MMAs the parts through ldmatrix from shared memory.

**Reproduce:**
1. Build test_pattn and bench_pattn from runner-proto e0c3d24.
2. Under the ring lock, run each: `test_pattn mhcrows`, then `bench_pattn --only mhcs`.

### Small-M tiles for the dense prefill GEMMs: a measurement (2026-10-12)

**The question.** L3 of [the time-to-first-token study](#a-small-turns-time-to-first-token-2026-10-12): can
the six dense prefill GEMMs reach ≥ 80 % of DRAM bandwidth at a small turn's rows, with the same bits?
The six are the router gate, the shared expert's w1/w3, wq_a + wkv, wq_b, wo_a and wo_b.

**Verdict.** Missed. No GEMM reaches 80 % (188 GB/s) at any row count measured.
- **The router gate gains 2.6 times:** 118.5 → 43–45 µs at 16–32 rows, 69 at 184.
- **The shared expert's w1/w3 gains 16–21 %.**
- **wq_a + wkv and wo_b gain 2–7 %** up to 32 rows and lose beyond.
- **wq_b and wo_a are slower** in the small tiles.
- **Bits:** every variant gives the default tiles' bits at every row count tested.
- **Measured L3:** ~4.3 ms a +44 turn, 3.8 at +200, 3.4–3.5 at +350 and +728. That is about a third of
  the target's 7.8–12.3 ms.

**The variants** (`PrefillTuning::small`, off by default):
- **The router gate** (N = 384) was a 128 × 128 tile: 3 CTAs whatever the rows. In 32 × 32 tiles it gets
  12 CTAs per 32 rows; in 16 × 16 tiles, 24 per 16 rows (48 at the replay's 32).
- **The FP8 GEMMs and wo_a** get 32 × 64 or 32 × 32 tiles; **the shared expert's w1/w3** 32 × 64 (32 SwiGLU
  columns a tile, one act_quant group).
- **No split-K.** Each output stays one thread's accumulator through the K blocks in order, as in the
  default tiles.
- **The test.** `small_gemms.h` runs every output against the default tiles on random data, bit for bit,
  at 1–728 rows (`test_kernels p_fp8`, `p_small`); it passes.

**Measured.** `bench_kernels psmall` (289ad70) on rank 0, under the ring lock. µs per GEMM, median of 9 after a
64 MB memset of L2; in brackets, GB/s of the weights. The memset leaves dirty lines that the GEMM's loads
write back, so these times run 10–40 % above the engine's (wo_b 265 against 223 in place).

| GEMM, rows | default | small 1 | small 2 |
|---|---|---|---|
| router gate, 16 | 118.5 (33) | 44.8 (88) | 42.7 (92) |
| router gate, 32 | 118.5 (33) | 44.8 (88) | 50.9 (77) |
| router gate, 184 | 122.6 (32) | 69.3 (57) | 84.7 (46) |
| shared w1/w3, 16 | 77.5 (76) | 61.2 (96) | 61.2 (96) |
| shared w1/w3, 184 | 98.0 (60) | 81.5 (72) | 80.6 (73) |
| wq_a + wkv, 16 | 87.8 (105) | 81.6 (112) | 84.7 (108) |
| wq_a + wkv, 184 | 108.3 (85) | 103.1 (89) | 127.7 (72) |
| wo_b (4 slices), 16 | 264.9 (158) | 257.8 (163) | 256.7 (163) |
| wo_b (4 slices), 184 | 281.3 (149) | 411.3 (102) | 474.0 (88) |
| wq_b, 44 | 97.0 (108) | 108.3 (97) | 112.4 (93) |
| wq_b, 200 | 125.7 (83) | 151.3 (69) | 171.7 (61) |
| wo_a, 128 | 103.0 (81) | 103.4 (81) | 126.7 (66) |
| wo_a, 200 | 160.4 (52) | 152.3 (55) | 169.7 (49) |

**Why it stops.**
- **The gate:** with 12–24 CTAs, each still streams its whole K = 5,120 in order. It is bound by one CTA's
  pipeline depth, not by the bytes.
- **The others:** narrower tiles give more CTAs but make each read its activations once per N tile, and run
  fewer MMAs per byte. Past ~56 rows that costs more than the parallelism gains.
- **Reaching 80 %** would need deeper per-CTA pipelines (more K stages in flight), not just more CTAs.

**L9 and the bundle, from measured numbers.**
- **L9's collectives** (`ring_test --cap 4194304 latency`, 289ad70, the ring): a bf16 reduce-scatter takes
  38 µs at 451 KB (+44's rows), 88 µs at 1.3 MB (the replay) and 127 µs at 2 MB.
- **L9's net per layer:** +176 µs at 44 rows, +52 µs at 128, and a loss from 200 rows. It saves 3/4 of the
  wo_b, wq_a + wkv and gate times, against the reduce-scatter and two all-gathers.
- **End to end on r2:**

| lever | end to end | status |
|---|---|---|
| L2 (88 µs) | 0.69 % | measured |
| L3 | 0.26 % | measured |
| L9 | 0.24 % | measured parts |
| L4a | 0.05 % | projected |
| L4b | 0.10 % | projected |
| L2 + L3 + L4a + L4b | 1.11 % (1.14 % off) | |
| + L9 | 1.34 % (1.39 % off) | |
| + L5 and L7 (estimates) | 1.61 % (1.66 % off) | |

**Verdict:** under 2 % even with every estimate.

**Reproduce:** build test_kernels and bench_kernels from 289ad70, then run under the ring lock:
- `test_kernels '' p_fp8`
- `bench_kernels psmall`

### The draft pass's approximations: a study (2026-10-12)

The question: the draft pass costs 2.5–2.6 ms of a 26–38 ms step ([The draft pass](#the-draft-pass)). The
verify judges the drafts by the target's own samples, so a draft never changes a token, and the drafter can be
approximated for speed, as the FP8 head copy already is. Which approximations pay, net of the acceptance they
cost? The bar: 2 % or more end to end on the sampled edit bench, no bench type or sparkDash catalog more than
2 % slower, and no language's text much slower.

The levers studied:
1. a vocabulary subset for the drafter's head GEMV and Markov chain;
2. the Markov head from an FP8 copy;
3. the FP8 head copy as MXFP4;
4. the blocks' dense FP8 weights and main_proj as MXFP4;
5. accept beside the verify's tail, and wo_a at full speed (modelled only).

This was measurement and analysis only, and serving is unchanged. The hooks (attn-dpstudy 23beba7) are on master since
631cf3c, off by default. The analysis is engine/dspark/ref/dp_study.py. vocab_freq.py, vocab_study.py and
vocab_script.py build and score the subsets.

**Verdict: one bundle clears the bar in the replay, with little margin.** (2026-10-13: built, it failed its A/B,
+1.50 % end to end. The replay's gains are decode tok/s, and prefill is 24 % of the sampled edit bench's time:
[Bundle B's A/B](#bundle-bs-ab-under-the-bar-2026-10-13).)
- The bundle has six parts:
  - the blocks and main_proj as MXFP4;
  - the head copy as MXFP4;
  - the Markov head as FP8;
  - accept beside the verify;
  - the wo_a fix;
  - the Latin-script subset, used only while the context's last 128 tokens are all in it.
- Its projected gains:
  - +2.36 % on the sampled edit bench;
  - +2.19 to +3.65 % on the bench types;
  - +0.41 to +2.18 % on the catalogs;
  - +2.11 to +3.83 % on Chinese, Japanese and Russian prose.
- Two of its savings are modelled: the FP4 blocks, and the accept and wo_a pair.
- It is built in stages, each measured before the next (attn-dpbuild).
- Without the FP4 blocks, no bundle reaches 2 % except one with an interleaved 32k subset (+2.13 %).
- Every static subset fails on non-Latin text.

- **How it was measured.**
  - `DSPARK_SIM` (spec.cu, at bind) rounds the drafter's weights in place to the values a cheaper format
    holds:
    - `fp4blocks`: wq_a, wkv, wq_b, wo_a, wo_b and the shared expert of each block, and main_proj, become
      MXFP4. Each row's groups of 32 take e2m1 under a power-of-two scale (amax / 6, rounded up), stored as
      e4m3 under the block's scale doubled.
    - `fp4head`: the FP8 head copy, the same way.
    - `fp8markov`: the Markov head, to e4m3 under a power-of-two scale per 32×32 block, as the head copy is
      made.

    So spectf measures each variant's own drafts and survival with today's kernels. Block 0 of every
    rounded tensor is checked against a host reference at bind (1024 of 1024 values, every run).
  - `DSPARK_VROWS` and `bench_dspark --fp8 1 --vrows` time the pass with its head GEMV and Markov chain
    over fewer vocabulary rows. Half the rows reads an FP4 head's bytes, and an FP8 Markov head's.
  - **The sequences.** One ring job (dpjob: seven parts, about 35 minutes on the ring) ran spectf over 66 sequences and 13,166
    positions:
    - the sampled edit bench's repeat 0 (38 turns, 4,974 positions);
    - the four bench types (2,048);
    - 20 sparkDash catalog prompts, spread over each catalog: 6 of TEXT_PROMPTS, 6 of STRUCTURAL_PROMPTS
      and 8 of DECODE_CODE_TASKS;
    - four non-Latin prose prompts written for the study: two Chinese, one Japanese, one Russian.

    The new prompts were encoded as dsfront serves a request with thinking off. The pipeline gives
    bench.py's prose prompt the bench trace's 39 ids exactly. They were generated greedily to 256 tokens,
    as sparkDash's catalogs are, with no early EOS.
  - **A check of the build.** The study build's records equal the earlier runs' on all 7,022 positions of
    the same sequences: drafts and survival, step for step.
  - **The replay.** Today's chain gate and step costs (pld_study.py's walk and the tree study's fit, with
    per-trace offsets to the served rates) are replayed on each variant's records, with the pass shortened
    by the variant's saving.
    - The edit bench and the bench types use the gate and PLD studies' routes and offsets. The new traces
      take the model's experts and no offset; only their ratios matter.
    - ± is the gain's standard deviation over sequences resampled within the group (4 or more sequences).
  - **Subsets** are scored on the probe records (the drafter's top 8 along the target's tokens), as
    vocab_study.py does: a subset drafter's top-1 is the first of the top 8 inside the subset. This is
    exact while its drafts are right. Its saving is the pass at the subset's largest rank slice.

- **The time each lever saves.** bench_dspark: one rank, stand-in collectives, the FP8 head, ms.

  | vocabulary rows a rank | the pass | accept 1 + the pass | accept 6 + the pass | the head GEMV alone |
  |---|---|---|---|---|
  | 32,320 (today) | 2.068 | 2.200 | 2.210 | 0.699 |
  | 21,376 (the Latin subset's largest slice, rank 0) | 1.761 | 1.886 | 1.888 | 0.464 |
  | 19,520 (its balanced share) | 1.719 | 1.843 | 1.842 | 0.428 |
  | 16,192 (half: an FP4 head's bytes) | 1.629 | 1.752 | 1.755 | 0.357 |
  | 10,688 (half the Latin slice) | 1.478 | 1.604 | 1.602 | 0.238 |
  | 8,192 (32k tokens interleaved) | 1.420 | 1.544 | 1.542 | 0.182 |
  | 4,096 | 1.325 | 1.448 | 1.451 | 0.090 |

  - **The FP4 head: −0.342 ms.** The FP8 head at half the rows takes 0.357 ms against 0.699. This assumes an
    FP4 GEMV streams as the FP8 one does. No such kernel exists yet; the MoE's FP4 decode is the model.
  - **The FP8 Markov head: −0.097 ms.** This is the rest of the pass at half the rows (1.272 against 1.369
    ms). The proxy halves all five steps' reads and the chain's CTAs, so it may overstate. The earlier
    estimate for markov[0] alone was −0.045.
  - **The Latin subset: −0.307 ms**, or −0.151 on the FP4 head and FP8 Markov head (10,688 rows against
    16,192). The 32k most frequent tokens, interleaved over the ranks: −0.648 and −0.304. A row of head and
    chain costs 26 ns a rank.
  - **The FP4 blocks, modelled: −0.30 ms.** The blocks' dense FP8 is 40.5 MB a block a rank, and main_proj
    19.66 MB. As MXFP4 they take 17 bytes a group of 32 in place of 32, so 66 MB less a pass, at the 220 GB/s
    the blocks' GEMVs stream.
  - **Modelled:**
    - accept beside the verify: −0.05 ms (main_proj's DRAM read stays; its collectives' latency and the
      launches hide);
    - wo_a without its act_quant's fixed 64 CTAs: −0.02 ms on FP4's bytes, −0.04 on FP8's.

- **What each format costs in acceptance.** The cells are: accepted drafts a step; the gain in tok/s with
  the variant's saving; in brackets, the gain at no saving (its acceptance alone). The drafts-differ row
  counts the steps where any of the five drafts differs from today's.

  | | edit r2 | bench prose | code | agentic | json | catalog prose | structural | code | zh | ja | ru |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | today: accepted a step, tok/s | 3.75, 126.4 | 0.81, 69.4 | 2.07, 101.9 | 2.10, 99.8 | 3.88, 136.0 | 1.00, 74.4 | 2.30, 103.7 | 3.81, 132.7 | 1.06, 75.9 | 0.72, 67.1 | 1.17, 80.4 |
  | fp4blocks (−0.30 ms) | 3.73, +0.63 % (−0.18 ± 0.27) | +1.99 (+0.82) | +1.39 (+0.37) | −1.21 (−2.18) | +0.63 (−0.21) | −0.36 (−1.47 ± 0.69) | −0.87 (−1.81 ± 0.52) | −1.46 (−2.29 ± 0.70) | +0.83 (−0.27) | +1.22 (+0.06) | +1.02 (−0.11) |
  | its drafts differ | 13.9 % | 61.9 % | 39.8 % | 39.3 % | 10.2 % | 59.2 % | 29.9 % | 12.0 % | 55.3 % | 52.0 % | 55.5 % |
  | fp4head (−0.34 ms) | 3.74, +0.73 % (−0.18 ± 0.14) | +2.35 (+1.00) | +2.45 (+1.28) | +0.06 (−1.04) | +0.94 (−0.02) | +1.61 (+0.32 ± 0.31) | +1.21 (+0.12 ± 0.46) | +0.59 (−0.36 ± 0.44) | −0.03 (−1.28) | +1.26 (−0.09) | +1.06 (−0.21) |
  | its drafts differ | 4.4 % | 27.5 % | 20.9 % | 18.4 % | 5.5 % | 30.3 % | 13.0 % | 4.2 % | 31.8 % | 24.2 % | 25.4 % |
  | fp8markov (−0.10 ms) | 3.76, +0.41 % (+0.15 ± 0.09) | +0.38 (+0.01) | +1.55 (+1.22) | +0.25 (−0.06) | +0.27 (+0.00) | +0.43 (+0.07 ± 0.23) | +0.33 (+0.02 ± 0.24) | +0.27 (+0.00) | +0.51 (+0.15) | +0.38 (+0.00) | +0.36 (+0.00) |
  | its drafts differ | 1.1 % | 6.4 % | 3.7 % | 5.1 % | 2.0 % | 7.7 % | 3.3 % | 0.9 % | 7.2 % | 9.0 % | 9.0 % |
  | all three (−0.74 ms) | 3.72, +1.76 % (−0.25 ± 0.31) | +2.75 (−0.16) | +1.74 (−0.77) | +1.45 (−0.99) | +1.89 (−0.21) | +1.33 (−1.46 ± 0.54) | −0.01 (−2.32 ± 0.59) | −0.21 (−2.26 ± 0.62) | +1.84 (−0.88) | +3.54 (+0.62) | +3.50 (+0.67) |

  - **The drafts move a lot, and acceptance moves little.** The drafts move most where acceptance is
    lowest (prose).
  - **The FP4 blocks are the only format that costs acceptance clearly.** On the greedy catalogs, −1.5 to
    −2.3 % at no saving, 2–4 standard deviations. On the sampled edit bench, −0.18 ± 0.27 %.
  - **The FP4 head and the FP8 Markov head cost nothing measurable.** Their single-trace swings (±1.3 %)
    go both ways.
  - The bench types are one 512-token trace each, so their cells move by ±1–2 % on drafts alone.

- **Vocabulary subsets: every static one fails on some language.** Gain in tok/s with each subset on today's
  drafter (none's probe records):

  | subset (its saving) | edit r2 | bench prose | code | agentic | json | catalog prose | structural | code | zh | ja | ru |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | the Latin script, 78,026 ids (−0.31 ms) | +0.82 % | +1.19 | +1.03 | +1.00 | +0.86 | +1.15 | +0.97 | +0.85 | −40.9 | −36.1 | −51.8 |
  | the same, with each session's context added | +0.82 | +1.19 | +1.03 | +1.00 | +0.86 | +1.15 | +0.97 | +0.85 | −31.1 | −13.4 | −35.1 |
  | the same, used only while the last 128 tokens are in it | +0.79 | +1.19 | +1.02 | +0.99 | +0.86 | +1.14 | +0.96 | +0.84 | 0.00 | 0.00 | 0.00 |
  | the 32k most frequent, interleaved (−0.65 ms) | +1.57 | +1.63 | +2.20 | +0.89 | −15.6 | +0.72 | −0.32 | +1.07 | −37.7 | −31.2 | −49.4 |
  | the same, with the context | +1.68 | +1.68 | +2.20 | +2.11 | −9.1 | +0.75 | +0.14 | +1.57 | −29.1 | −10.2 | −32.9 |
  | the same, only while the last 128 tokens are in it | +1.54 | +0.56 | +1.63 | +0.28 | +0.09 | +0.08 | +0.68 | +1.35 | 0.00 | 0.00 | 0.00 |
  | the 64k most frequent, interleaved (−0.43 ms) | +1.09 | +1.97 | +1.46 | +1.41 | −6.5 | +0.99 | +0.17 | +1.21 | −3.9 | −3.1 | −10.0 |
  | the same, with the context | +1.09 | +1.97 | +1.46 | +1.41 | −4.6 | +0.99 | +0.34 | +1.21 | −3.7 | −1.2 | −9.8 |

  - **The frequency counts** are public text only:
    - this repository, the golden transcripts, and the bench and catalog prompts;
    - engram's public code and prose streams: the Python stdlib, /usr/share/doc and the licences, 5.5M
      tokens in all.

    They see 34,168 distinct tokens. Past those, the order is BPE merge order.
  - **The Latin subset** (vocab_script.py) keeps:
    - Latin, digits, punctuation, symbols, emoji and fullwidth ASCII;
    - every added token;
    - byte pieces by their lead byte's block.

    0.000 % of the Latin workloads' outputs fall outside it. Its rows on the head's own slices are 21,318,
    19,719, 18,543 and 18,446, so no rank needs another's rows. The frequency subsets are lopsided on the
    slices (32k: 15,621 to 3,245 rows) and need the rows interleaved.
  - **Adding the context** (the prompt and the output so far) costs under 0.01 ms a pass in rows, and
    still loses 10–35 % on CJK and Cyrillic. A new character, or a new inflection, is a first occurrence.
  - **The adaptive rule passes every language.**
    - The rule: the subset only while the context's last 128 tokens are all inside it, the whole
      vocabulary otherwise.
    - The Latin subset is then in use at 100 % of steps on every Latin workload and at 0 % on the zh, ja
      and ru traces, which see today's drafter exactly.
    - The 32k subset falls back much more: 96 % of steps use it on the edit bench, 8 % on bench prose and
      4 % on json.

- **Bundles.** Gain in tok/s on all three formats' records, with the modelled levers, and subsets (adaptive)
  from the same run's probe records:

  | bundle (its saving a pass) | edit r2 | bench prose | code | agentic | json | catalog prose | structural | code | zh | ja | ru |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | all three (−0.74 ms) | +1.76 % | +2.75 | +1.74 | +1.45 | +1.89 | +1.33 | −0.01 | −0.21 | +1.84 | +3.54 | +3.50 |
  | A: and accept beside the verify, wo_a (−0.81) | +1.95 | +3.04 | +1.98 | +1.68 | +2.09 | +1.60 | +0.22 | −0.02 | +2.11 | +3.83 | +3.78 |
  | **B: A and the Latin subset, adaptive (−0.96 on its steps)** | **+2.36** | +3.65 | +2.51 | +2.19 | +2.53 | +2.18 | +0.70 | +0.41 | +2.11 | +3.83 | +3.78 |
  | C: A and the 32k most frequent, interleaved, adaptive (−1.11) | +2.71 | +3.15 | +2.76 | +1.82 | +2.13 | +1.48 | +0.54 | +0.75 | +2.11 | +3.83 | +3.78 |
  | A and the 64k most frequent, interleaved, adaptive (−1.02) | +2.58 | +3.22 | +2.52 | +2.10 | +1.48 | +1.69 | +0.15 | +0.57 | +2.11 | +3.83 | +3.78 |
  | D: no FP4 blocks; the FP4 head, the FP8 Markov head, accept, wo_a (−0.53) | +1.24 | +3.10 | +3.10 | +0.68 | +1.48 | +2.34 | +1.81 | +1.12 | +0.67 | +2.02 | +1.78 |
  | D and the Latin subset, adaptive (−0.68) | +1.83 | +2.68 | +2.31 | +2.23 | +1.93 | +2.59 | +2.18 | +1.90 | +0.67 | +2.02 | +1.78 |
  | D and the 32k, interleaved, adaptive (−0.83) | +2.13 | +2.83 | +2.72 | +0.81 | +1.52 | +2.24 | +1.61 | +1.67 | +0.67 | +2.02 | +1.78 |

  - D's rows use fp4head's records and take fp8markov's acceptance as today's (+0.15 ± 0.09 % on edit).
    Their subset steps come from today's probe records. The single-trace columns (bench types, zh, ja,
    ru) carry that mix's noise.
  - **B is the bundle to build.** It is the best with natural slices, and no type is slower.
    - Its uncertainty is about ±0.3 % from acceptance on the edit bench, plus whatever the two modelled
      savings and the two proxies miss.
    - C gains 0.35 % more but needs other ranks' head rows on each rank.
    - Without the FP4 blocks, only D with the 32k interleaved subset passes, and only just.
  - **The build goes in stages, each measured before the next** (the coordinator's plan; results in
    [Bundle B, built in stages](#bundle-b-built-in-stages-2026-10-12)):
    1. the FP8 Markov head, then the FP4 head;
    2. the adaptive Latin subset: two copies of the head and Markov rows, and a device-side count of the
       tokens outside it over the last 128 (built as one permuted copy, for Latin and for the 32k most
       frequent ids; the 32k were picked);
    3. accept beside the verify, and wo_a;
    4. the FP4 blocks last, after a one-site prototype confirms the modelled saving within a third.

    Otherwise it falls back to D with the 32k subset, the other ranks' head rows gathered once at bind.
    Then an A/B against master decides.

- **Files.**
  - The hooks: attn-dpstudy 23beba7. bench_dspark's head timing over the rows: c3b6956. The binaries were
    built at c3b6956.
  - The tools: engine/dspark/ref/dp_study.py, vocab_freq.py, vocab_study.py and vocab_script.py.
  - The data: rank 0's /data/dsgb10/bench/dpstudy.
    - The new traces' dirs, and enc_prompts.py, which made them.
    - job/: the parts' scripts; tf_V.log for each variant; tp_none.jsonl and tp_all.jsonl; b_vrows.log; the
      generate log; study.txt.
  - The figures rerun with:

    ```
    dp_study.py /data/dsgb10/bench/dpstudy/job --latin LATIN.json --counts COUNTS.json
    ```

    vocab_script.py writes LATIN.json, and vocab_freq.py writes COUNTS.json, from the sources above.

### Bundle B, built in stages (2026-10-12)

The study's bundle B ([above](#the-draft-passs-approximations-a-study-2026-10-12)) is being built on
attn-dpbuild, one stage at a time. Every piece is behind a flag, and today's drafter stays the default.
Each stage's measured saving is set against the study's model before the next one starts.
`dp_stage.py` replays a stage's spectf records through today's chain with the measured saving, next to
the study's model of the same levers.

The outcome (2026-10-13): the four stages were built and measured, and the bundle failed its A/B (+1.50 % end to
end on the sampled edit bench against the 2 % bar; [below](#bundle-bs-ab-under-the-bar-2026-10-13)). Only stage 1's
flags are on master (`--draft-markov8`, `--draft-head4`, off by default, ccec056). Stages 2–4's code, their tests
and the A/B's tools stay on the branch attn-dpbuild (55b35a1, not merged and not pushed), and the wiring that would
have made the bundle the default on attn-dpwire (1932c22). Master carries these sections only.

#### Stage 1: the FP8 Markov head and the MXFP4 head

**Verdict: both build bit-exact to the study's simulations and save 90 % of the modelled time.**
- The pass is 0.397 ms shorter (modelled 0.439).
- Replayed, that is +0.98 % on the sampled edit bench (the study's model of the pair: +1.00 %).
- The tokens do not change: the 7 reference hashes, serve_test, and a drafted decode with both flags on
  all match.

- **The flags.**
  - `--draft-markov8`: the Markov head as FP8.
  - `--draft-head4`: the FP8 head copy as MXFP4 (it implies the copy; the copy is freed once the MXFP4
    head is made).
  - Both copies are made at bind from today's weights.
- **The FP8 Markov head.**
  - markov_fp8() lays the bf16 Markov head out as e4m3, with a ue8m0 scale per 32×32 block. These are
    sim_fp8_l2's values, in the markov kernel's L2 order: a lane loads 4 × 16 bytes in place of 8.
  - `markov_kernel<true>` decodes each value to fp32 times its block's scale (exact) and multiplies in
    today's order. Its bias is therefore the study's simulated one, bit for bit.
- **The MXFP4 head.**
  - head_fp4() makes it from the FP8 copy. Each group of 32 takes sim_fp4's values: e2m1 under the
    group's MXFP4 scale, then the FP8 copy's e4m3 rounding of the halved value.
  - Each group is stored as e2m1 under the least power-of-two scale that holds all 32 values. That is
    sim_fp4's own scale unless the e4m3 rounding moved a value: then it is one step up. On the real head
    no group moves; on random weights only groups whose largest value is 3 × 2^-9 of the block scale do.
  - gemv_fp4w_f32 (kernels/gemv_fp4.cu, new) reads it. It widens e2m1 into bf16 in registers and runs
    gemv_fp8w_f32's MMAs, split-K and reduction. So it gives the FP8 kernel's sums on those values, bit
    for bit, at the same wk.
  - 88 MB a rank, against the FP8 copy's 165 MB.
- **The checks** (lite_tests, a new test binary; 714f3a2).
  - The GEMV on a random head-shaped FP8 matrix: subnormal rows, and 103,360 groups whose scale moves,
    all found. FP4 against FP8 on the same values at mt/wn 1/1, 2/1, 4/1, 1/2 and 2/2, and 1, 5 and 8
    rows: bit-identical.
  - The module on rank 0's real weights over 16 golden steps, against the module on DSPARK_SIM's
    rounded weights: the logits, the bias, the biased logits, the drafts and the confidences are
    bit-identical.
- **The pass** (bench_dspark, one rank, stand-in collectives, median of 50 × 5 replays), ms:

  | drafter | draft pass | accept 1 + draft | the head GEMV alone |
  |---|---|---|---|
  | today's (the FP8 head copy) | 2.063 | 2.193 | 0.699 |
  | the FP8 Markov head | 1.976 | 2.101 | 0.695 |
  | the MXFP4 head | 1.762 | 1.891 | 0.341 |
  | both | 1.666 | 1.789 | 0.341 |

  - The savings against the model:

    | | measured | modelled |
    |---|---|---|
    | the FP8 Markov head | 0.087 | 0.097 |
    | the MXFP4 head | 0.301 (the GEMV alone 0.358) | 0.342 |
    | both | 0.397 | 0.439 |

    The model's 0.097 for the Markov head holds within 10 %; the earlier estimate from markov[0] alone
    (0.045) was too low.
  - The MXFP4 head GEMV streams 88 MB in 0.341 ms (258 GB/s). The other tile shapes (mt/wn 1/1, 4/1,
    1/2 and 2/2) are no faster (0.345 to 0.355 ms).
  - With 64 vocabulary rows, the pass is 1.22 ms with either head. So the chain's rows cost about
    0.15 ms on the bf16 Markov head, and 0.07 to 0.11 ms on the FP8 one.
- **Acceptance.** spectf ran with both flags over the study's 66 sequences (13,166 positions). The probe
  records came along (tp_s1.jsonl), as stage 2's model needs them.
  - The drafts differ from today's at 4.4 to 31.6 % of positions. fp4head's differ at 4.2 to 31.8 %.
  - Drafts right from the first, a position (teacher-forced):

    | | edit-r2 | bench-prose | bench-code | bench-agentic | bench-json | cat-prose | cat-structural | cat-code | zh | ja | ru |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | today's | 4.254 | 0.980 | 2.564 | 2.617 | 4.211 | 1.297 | 2.973 | 4.154 | 1.471 | 0.898 | 1.551 |
    | fp4head (DSPARK_SIM) | 4.252 | 0.973 | 2.555 | 2.611 | 4.215 | 1.298 | 2.960 | 4.139 | 1.443 | 0.891 | 1.547 |
    | stage 1 | 4.254 | 0.971 | 2.537 | 2.619 | 4.199 | 1.298 | 2.969 | 4.138 | 1.445 | 0.883 | 1.539 |

    Stage 1 is at fp4head's level in every group. Against it, only 1 to 28 positions per group are
    worse or better.
  - Today's chain replayed (dp_stage.py), the gain in tok/s:

    | | edit-r2 | bench-prose | bench-code | bench-agentic | bench-json | cat-prose | cat-structural | cat-code | zh | ja | ru |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | stage 1, no saving | −0.08 ± 0.17 % | +0.61 % | +0.57 % | −1.76 % | −0.98 % | −0.04 ± 0.33 % | +0.28 ± 0.43 % | −0.36 ± 0.44 % | −1.29 % | −0.09 % | −0.21 % |
    | stage 1, −0.397 ms (measured) | +0.98 % | +2.17 % | +1.92 % | −0.49 % | +0.13 % | +1.46 % | +1.55 % | +0.74 % | +0.17 % | +1.48 % | +1.27 % |
    | the study's model: fp4head's records, −0.439 ms | +1.00 % | +2.74 % | +2.78 % | +0.38 % | +1.22 % | +1.99 % | +1.52 % | +0.86 % | +0.34 % | +1.65 % | +1.43 % |

    - On the edit bench and the catalogs, the replay matches the model within its spread.
    - The bench types and the languages are one or two traces each, 256 to 512 positions. Their
      shortfall against the model (up to 1.1 %) comes with no loss in the teacher-forced counts, so it
      is the walk's noise on a few changed positions.
- **The tokens.**
  - With every new flag off, the 7 reference hashes match (tf, k1, fp8, bf16, temp, ced16, ced28), and
    serve_test --ced passes (17 of 17).
  - k1's decode with `--spec --draft-fp8 --draft-head4 --draft-markov8` gives k1's hash, with the same
    2.542 drafts accepted per step. It ran 8.614 ms a token against 8.701 with the FP8 drafter alone
    (one run each: −1.0 %, about 0.31 ms a step).
- **A Markov-bias cache** was weighed (the coordinator's suggestion): the bias rows of the 4,096 most
  frequent previous tokens, precomputed by the same kernel.
  - It would cost ~265 MB a rank.
  - After the FP8 Markov head, the chain's rows cost 0.07 to 0.11 ms a pass. A hit would still read the
    logits and run the argmax, so a cache saves less than that, and only on hits: at most ~0.3 % end to
    end. It is not worth building now.
- **Files.**
  - attn-dpbuild: the stage 5127ae7; head_fp4's counters and the test's scales efd2d7c; the test's
    moved-scale rows 714f3a2; dp_stage.py 958c98b and 8abd097.
  - The job: rank 0's /data/dsgb10/bench/dpbuild/stage1. It holds the parts' scripts, lt.log, b_*.log,
    tf_s1.log, tp_s1.jsonl, h_*.log, s_*.log and stage1.txt. The binaries were dsgb10_rank and
    bench_dspark at efd2d7c, and lite_tests at 714f3a2.
  - The figures rerun with:

    ```
    dp_stage.py /data/dsgb10/bench/dpstudy/job /data/dsgb10/bench/dpbuild/stage1 s1 --saving 0.397 \
      --model 0.439 --model-recs fp4head
    ```

#### Stage 2: the vocabulary subset, as one permuted copy

**Verdict: both subsets build bit-exact to the study's adaptive rule; the bundle takes the 32k most
frequent ids.**
- On the steps that use it, the pass is 0.336 ms shorter with the 32k most frequent ids (modelled 0.304)
  and 0.156 ms with the Latin ids (0.151).
- Replayed with stage 1, the sampled edit bench gains +1.78 % with top-32k (the study's model +1.68 %) and
  +1.39 % with Latin (+1.38 %). No group is more than 0.5 % slower than today's drafter with either.
- The pick follows the coordinator's rule: best on the sampled edit bench, no bench type, catalog or
  language more than 2 % slower, and the memory a rank. The production workload is agentic coding, so the
  edit bench is the metric that counts.
  - The tradeoff: top-32k is 0.4 to 0.7 points weaker than Latin on bench-prose, bench-json and
    cat-prose, where its subset serves only 4 to 8 % of the steps, and better on code.
- The tokens do not change.

- **The flag.** `--draft-subset FILE` (with `--draft-head4` and `--draft-markov8`, without
  `--tree-probe`).
  - The file is what dspark/ref/subset_file.py writes: an int32 .npy of 129,280 entries, each id's rank
    or −1.
  - latin.npy holds the study's 78,026 Latin-script ids on their own ranks (21,318, 19,719, 18,543 and
    18,446).
  - top32k.npy holds the study's 32,768 most frequent ids, 8,192 a rank, interleaved by frequency.
- **One permuted copy per rank.** Each rank's MXFP4 head and FP8 Markov head rows become one copy.
  - Its rows [0, n_sub) are the rank's subset ids in id order. Then come its own slice's other ids, then
    the ids that other ranks have no room for. Every id appears once over the four ranks.
  - The copy is made at bind from stage 1's natural copies by moving rows, so every id's values are the
    natural copy's bits. The natural copies are freed afterwards.
  - Rows of other ranks' slices travel once, as packed rows of 2,992 bytes (head 2,560, its scales 160,
    the Markov row 256, its 8 scales), in one all-gather. Bind checks that the rows cover every id
    exactly once.
  - row_id maps a row to its id. The argmax breaks ties by id, as the natural rows do by position.
  - The Markov head's scales become per row ([32,320][8]), since a 32×32 block no longer holds
    neighbouring ids.
- **The device-side skip.** A draft's choice is a flag in device memory, written by draft_prep.
  - With the flag set, the head GEMV's and the Markov kernel's CTAs past roundup64(n_sub) return after
    the PDL wait. CTAs past that point load their first weights after the wait instead of before it.
  - Rows in [n_sub, roundup64(n_sub)) score −∞.
  - One step graph serves both choices.
- **The rule and its state.** Each session slot keeps 64 bytes: a ring of 256 bits marking which
  positions took a token outside the subset, and the end it reaches.
  - accept() notes the accepted tokens, and draft_prep notes the bonus. A position that was passed over
    counts as outside.
  - A draft uses the subset when its context's last 128 tokens, through the bonus, were all inside it.
    This is dp_study's adaptive rule.
  - The prefill notes each chunk's tokens. A CED chunk that is not sampled notes its last 256.
  - Snapshots keep the bits of the 128 positions before them (16 bytes). Swaps keep the whole state
    beside the slot's windows. A restored window marks every earlier position as outside.
- **Memory a rank.** The steady state is the same for both subsets: the copy replaces the natural rows,
  plus row_id (129 KB) and the subset's bits (16 KB).
  - At bind, both copies are held until the natural one is freed (96 MB).
  - top-32k's rows from other slices also need the all-gather's buffers: (1 + 4) × the most rows a rank
    sends × 2,992 bytes, 175 MB (rank 0 sends 11,717). Latin's ids are all on their own ranks, so it
    needs none.
  - All of it is freed inside the model's bind, before the Engram store, the prefill buffers and the
    session store size themselves from free memory.
- **The checks** (lite_tests; cc835b1).
  - The subset files: Latin's ids all on their own slices; top-32k 8,192 a rank, with 11,717, 6,372,
    6,844 and 7,381 rows from other slices; every id once.
  - The GEMV on a random head-shaped matrix, permuted with 10,710 of 32,320 rows in the subset: with the
    skip set and clear, against the natural rows.
  - The copy on each rank against the natural copies, every row's bytes (a random subset, up to 14,966
    rows from other slices).
  - The module on rank 0's real weights over 16 golden steps. With the subset forced off: stage 1's
    bits. Forced on: the argmax of stage 1's chain over the subset, bit for bit (0 of 80 drafts differ).
  - The state: 100 steps of a Latin session (70 on the subset) and 80 of a zh-like one (none). No choice
    goes against the rule.
- **The pass** (bench_dspark, one rank, stand-in collectives, median of 50 × 5 replays), ms:

  | subset (rank 0's rows) | stage 1 (natural rows) | the copy, subset off | the copy, subset on | saving | modelled |
  |---|---|---|---|---|---|
  | Latin (21,318 of 32,320, none from other slices) | 1.675 | 1.688 | 1.519 | 0.156 | 0.151 |
  | top-32k (8,192; 11,717 rows from other slices) | 1.684 | 1.675 | 1.348 | 0.336 | 0.304 |

  The copy with the subset off is within 0.013 ms of stage 1, either way: noise.
- **Acceptance.** spectf ran with each subset over the study's 66 sequences (tf_s2.log, tf_s2c.log).
  `dp_subset.py` checks every step against stage 1's records:
  - the choice is the rule's at every step;
  - off the subset, the drafts are stage 1's;
  - on it, every draft depth stage 1's records reach is the argmax over the subset (a few depths past
    them, 2 with Latin and 272 with top-32k, are not known).
  - 0 differences with either subset.
- **The replay** (today's chain, as stage 1's), the gain in tok/s:

  | | edit-r2 | bench-prose | bench-code | bench-agentic | bench-json | cat-prose | cat-structural | cat-code | zh | ja | ru |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | stage 1, −0.397 ms | +0.98 % | +2.17 % | +1.92 % | −0.49 % | +0.13 % | +1.46 % | +1.55 % | +0.74 % | +0.17 % | +1.48 % | +1.27 % |
  | Latin, −0.156 ms more on its steps (measured) | +1.39 % | +2.79 % | +2.45 % | +0.02 % | +0.57 % | +2.06 % | +2.05 % | +1.18 % | +0.17 % | +1.48 % | +1.27 % |
  | Latin, the study's model (−0.151 ms) | +1.38 % | +2.77 % | +2.44 % | +0.00 % | +0.55 % | +2.04 % | +2.03 % | +1.16 % | +0.17 % | +1.48 % | +1.27 % |
  | top-32k, −0.336 ms more on its steps (measured) | +1.78 % | +2.29 % | +2.77 % | −0.34 % | +0.18 % | +1.36 % | +1.91 % | +1.33 % | +0.17 % | +1.48 % | +1.27 % |
  | top-32k, the study's model (−0.304 ms) | +1.68 % | +2.28 % | +2.69 % | −0.36 % | +0.17 % | +1.36 % | +1.87 % | +1.25 % | +0.17 % | +1.48 % | +1.27 % |
  | top-32k's share of the steps | 95.7 % | 8.4 % | 75.0 % | 15.0 % | 3.9 % | 4.7 % | 41.1 % | 88.7 % | 0 | 0 | 0 |

  - Latin serves every step of the Latin-script groups and none of the languages'.
  - The study's model replays its adaptive records on stage 1's records. Each subset's own records give
    the same steps, with gains within 0.1 point of it.
- **The tokens.**
  - With every new flag off, the 7 reference hashes match.
  - k1's decode with stage 1's flags and either subset gives k1's hash.
  - serve_test --ced passes (17 of 17) with the flags off and with stage 1's formats and the Latin
    subset. slice_test (time-slicing's swaps) passes with the subset too. top-32k's server run is stage
    4's s4.
- **Files.**
  - attn-dpbuild: the copy 8b3e662, the flag f4bb8b0, lite_tests fd310b2, spectf's subset choice and
    dp_subset.py b37d0a9; the test's zh-like session ec56619 (the first lite_tests run failed on it: a
    test bug).
  - The job: rank 0's /data/dsgb10/bench/dpbuild/stage2. It holds the parts' scripts, lt.log, b_*.log,
    tf_s2.log, tf_s2c.log, h_*.log, s*_*.log, subset_latin.txt and subset_top.txt. The binaries were
    dsgb10_rank and bench_dspark at b37d0a9 and lite_tests at cc835b1. The subset files are in
    ../subsets.
  - The figures rerun with:

    ```
    dp_subset.py /data/dsgb10/bench/dpstudy/job /data/dsgb10/bench/dpbuild/stage1 \
      /data/dsgb10/bench/dpbuild/stage2 s2c --ids /data/dsgb10/bench/dpbuild/subsets/top32k_ids.json \
      --s1 0.397 --saving 0.336 --model 0.304
    ```

#### Stage 3: wo_a's tile shape, and accept beside the verify

**Verdict: the same bits, and less saved than modelled: about −0.025 ms a step against −0.09.** Both
levers stay in the bundle; its A/B decides.
- wo_a at other tile shapes saves nothing: the draft pass is within 0.004 ms of today's at every shape
  (modelled −0.04 ms on FP8's bytes).
- Accept beside the verify takes 0.124 ms off the draft path after the verdict. It costs the verify's
  head GEMV 0.099 ms to the join beside it: −0.025 ms a step (modelled −0.05).
- Every bit is unchanged: spectf's 13,166 records equal stage 2's, the module equals stage 1's at every
  shape, and k1's decode gives k1's hash in all 10 runs.

- **The flags.** Both keep the same bits and are off by default.
  - `--draft-woa MT,WN`: the drafter's wo_a GEMV at that tile shape, with its act_quant as a kernel of its
    own.
  - `--draft-accept-side`: a spec step's accept beside the verify's head GEMV.
- **wo_a.**
  - gemv_woa's fused act_quant needs 32 rows a CTA, so the drafter's wo_a (2,048 rows a rank, 8.4 MB)
    runs as 64 CTAs of 16 warps on 48 SMs.
  - With `--draft-woa`, gemv_woa runs at mt, wn with wk = 4, so every warp sums the same k range in the
    same order and the reduction is unchanged. It writes bf16.
  - A PDL kernel then makes the act_quant with the fused epilogue's own functions (dsk's
    act_scale_byte, quant_e4m3 and warp_max on the bf16 value).
  - At 1,1 that is 256 CTAs of 4 warps.
- **Accept beside the verify.**
  - Draft::accept splits in two:
    - accept_rows needs no count: act_quant of every main_hidden row, main_proj, the norm and the blocks'
      wkv, and their two all-gathers;
    - accept_write needs the count: the anchor, the subset state, and rows [0, count) into the windows.
  - Rows past the count are computed and never written. The GEMVs, the norm and the act_quant work row
    by row, so the rows that are written keep accept()'s bits.
  - In a spec step's capture, forward_rows forks accept_rows onto a high-priority side stream just
    before the head GEMV, and joins it just after.
  - Every rank must take the ring's sequence numbers in the same order. The head GEMV is the one stretch
    of the step with no collective on the main stream: the side stream's two all-gathers come after the
    last layer's all-reduce and before the sampling's.
  - accept_write runs after the verdict, followed by the draft.
  - main_hidden is zeroed at bind, since accept_rows reads every row.
- **The checks** (lite_tests --stage3; 04bdf00). The module on rank 0's real weights over 16 golden
  steps, with wo_a at 1,4, 1,1, 1,2 and 2,1, and with accept in two parts at today's wo_a and at 1,1:
  every bit stage 1's. The subset's state with accept in two parts: no choice against the rule.
- **The pass** (bench_dspark --stage3, one rank, stand-in collectives, median of 50 × 5 replays), ms:

  | variant | draft | accept 1 + draft | accept 6 + draft |
  |---|---|---|---|
  | wo_a fused at 1,4 (today's) | 1.666 | 1.796 | 1.797 |
  | wo_a at 1,4, act_quant apart | 1.662 | 1.792 | 1.792 |
  | wo_a at 1,1 | 1.667 | 1.798 | 1.800 |
  | wo_a at 1,2 | 1.663 | 1.792 | 1.795 |
  | wo_a at 2,1 | 1.668 | 1.798 | 1.800 |
  | wo_a at 2,2 | 1.667 | 1.797 | 1.798 |
  | accept in two parts | 1.670 | 1.800 | 1.798 |
  | accept_write 6 + draft (after the verdict) | | | 1.673 (−0.124) |

  - The verify's head GEMV (6 rows, the coded head's 255.8 MB): 1.042 ms alone; with accept_rows
    beside it, the head 1.128 (+0.086) and the join 1.140 (+0.099). accept_rows alone takes 0.079 ms.
  - The fused kernel's 64 CTAs cost nothing measurable: at 8.4 MB, wo_a takes the same time at every
    shape.
- **Under nsys** (p3: rank 0, k1's decode, 128 steps, wo_a at 1,1, graph nodes traced one by one, so
  indicative only), medians a step, µs, without | with accept beside the head:

  | rows | steps | the head GEMV | the draft pass after it | accept_rows beside (its span) |
  |---|---|---|---|---|
  | 2 | 12 / 12 | 1,099.9 / 1,159.5 (+59.6) | 2,017.3 / 1,867.2 (−150.1) | 300.3 |
  | 3 | 4 / 4 | 1,119.9 / 1,135.5 (+15.6) | 2,039.6 / 1,899.6 (−140.1) | 295.1 |
  | 6 | 13 / 13 | 1,135.8 / 1,118.0 (−17.8) | 2,073.1 / 1,883.7 (−189.4) | 290.7 |

  The side stream slows the head GEMV by less than it saves after it, so it stays (the coordinator's
  condition).
- **The decode** (k1: 1,024 positions, 256 steps; two rounds, the variants alternating; every run k1's
  hash and 2.542 drafts accepted a step), the step's GPU time weighted by the steps of each row count, ms
  a step against stage 2's drafter:

  | variant | round 1 | round 2 | mean |
  |---|---|---|---|
  | wo_a at 1,1 | +0.021 | −0.008 | +0.006 |
  | wo_a at 1,2 | +0.090 | −0.010 | +0.040 |
  | accept beside the verify | +0.138 | −0.210 | −0.036 |
  | both | +0.246 | −0.160 | +0.043 |

  Stage 2's drafter itself moved 0.27 ms a step between the rounds (31.18 and 31.46), so these are
  inside the noise.
- **The tokens.** spectf over the study's 66 sequences with both flags on (tf_s3.log): 13,166 records, 0
  differ from stage 2's (drafts, survival, accepted, subset choice). serve_test --ced (17 of 17) and
  slice_test pass with both flags on.
- **Files.**
  - attn-dpbuild: accept in two parts and wo_a's shapes 93e6cb3, the flags 7fe8f8a, the tests 04bdf00,
    the bench cbbb5e7, dp_stage3.py 137e27c and 2f9fea0.
  - The job: rank 0's /data/dsgb10/bench/dpbuild/stage3. It holds the parts' scripts, lt.log, b3.log, the
    decode runs d{1,2}_*.log, tf_s3.log, s3_*.log, p_{noside,side}.nsys-rep and stage3.txt. The binaries
    were at cbbb5e7.
  - The figures rerun with `dp_stage3.py /data/dsgb10/bench/dpbuild/stage3`.

#### Stage 4: the FP4 blocks

**Verdict: the FP4 blocks give the simulation's drafts and save more than modelled in the decode.**
- The draft pass is 0.174 ms shorter in the bench (0.207 with accept; modelled 0.244). k1's decode is
  0.372 ms a step faster.
- The kernels give `DSPARK_SIM=fp4blocks`'s values, and its sums except a rare 1-ulp difference in a draft
  GEMV output. The MMA with B in e2m1 does not always give the bits of the MMA with B in e4m3 on the same
  products (below). On the study's sequences no draft differs: 0 of 13,166 records.
- Replayed, the bundle (stages 1 to 4) gains +2.20 % on the sampled edit bench at the bench's savings, or
  +2.76 % with the decode's. No group is more than 0.11 % slower. Its A/B against master decides.
- Only drafts change, so the tokens do not.

- **The flag.** `--draft-blocks4` (off by default; drafts only): the drafter's blocks and main_proj as
  MXFP4.
  - Bind first rounds, in place, the 25 tensors that `DSPARK_SIM=fp4blocks` rounds (round_fp4blocks, by
    sim_fp4): main_proj, and each block's wq_a, wkv, wq_b, wo_a, wo_b and the shared expert's w1, w3 and
    w2. The flag and the variable refuse each other.
  - blocks4() then makes MXFP4 copies of main_proj and of each block's wq_a, wkv, wq_b, wo_a and wo_b
    (fp4_tiles): 61 MB a rank beside the FP8 tensors, which stay.
  - So the drafts take the simulation's values, which the study measured. The stage only makes them
    cheaper.
- **The copies.** fp4_tiles stores each row's group of 32 as e2m1 under the least power-of-two scale (from
  its largest value's, at most 8 times it) that holds all 32 rounded values exactly, as head_fp4 does for
  the head. A group of zeros keeps its block's scale.
  - It dies on a group with no such scale, or on a scale out of the kernel's range (the MMA's ue8m0,
    1..254; gemv_fp4w's bf16 widening, at most 128).
  - The real tensors' FP8 block scales are 115..120, so their MXFP4 scales stay at most 127.
- **The kernels** (kernels/gemv_fp4.cu).
  - gemv_fp4x: gemv_fp8's e4m3 activations against MXFP4 weights. It runs wq_a + wkv (one launch, two
    segments), wq_b, wo_b and main_proj.
    - It keeps gemv_fp8's 64-k steps, each warp's step range, the MMA order (two 32-k blocks a step) and
      the split-K reduction.
    - Each tile and block is one m16n8k32 block-scaled MMA whose B is e2m1 (e2m1x4_expand), under the
      row's own scale.
    - A lane holds its row's 8 values a block at gemv_fp8's offsets after its shuffle
      (16(t & 1) + 8(t >> 1)), so the activations load as there. A lane reads 8 bytes a step.
  - gemv_fp4w (wo_a): stage 1's k_gemv_fp4w with gemv_woa's block-diagonal input (a CTA's rows read their
    group's 4,096 columns) and bf16 out. Its act_quant follows as stage 3's kernel.
- **The MMA's sums depend on B's format** (a hardware finding, 2026-10-12).
  - The plan assumed that the block-scaled MMA
    (`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`) gives the same bits for the
    same products whether B comes as e4m3 under one scale or as e2m1 under a scale 2^k higher. Then
    gemv_fp4x would give gemv_fp8's sums on the same values, bit for bit.
  - lite_tests' test_mma_pair checks that on 8,192 random tiles: A in e4m3 under random scales, B's values
    exact in both formats, two k-blocks chained onto a random C. **1,034 of the 1,048,576 outputs
    differ.** The products are the same, so the accumulation differs (in its order or its width; the PTX
    ISA specifies neither).
  - On the drafter's shapes with random weights, gemv_fp4x gives gemv_fp8's bits except one output of
    wq_b, 1 bf16 ulp off (51,044 against 51,045): 1 of 8,192 with one row, 1 of 40,960 with five, at
    both tile shapes. wq_a + wkv, wo_b and main_proj are exact. gemv_fp4w widens e2m1 to bf16 and runs
    gemv_fp8w's MMAs, so it is exact.
  - On rank 0's real weights, the module gives `DSPARK_SIM=fp4blocks`'s bits over 16 steps.
  - So the stage's claim is: the simulation's values; its sums except a rare 1-ulp difference in a draft
    GEMV output. A different draft can cost acceptance, never a token: the verify decides every token.
- **The checks** (lite_tests --stage4, 12f2187: the coordinator's rule after the premise failed).
  - gemv_fp4x against gemv_fp8 on the same values: at most one bf16 ulp apart, in at most 1e-4 of the
    outputs per site, pooled over configs. Each configuration's count is printed.
    - The caps: wq_a + wkv 0 of 5,376 outputs, main_proj 2 of 23,040, wo_b 6 of 61,440 and wq_b 9 of
      98,304. The counts: wq_b 4 (one in each of its four configurations), the others 0.
    - Judged per call at first, wq_b's one output was 1.2e-4 of a one-row call's 8,192 outputs, and the
      second lt4 failed on it.
  - The same call again, and on another stream: every bit. Approximate drafts are fine; nondeterministic
    ones are not.
  - gemv_fp4w against gemv_woa, and the module against `DSPARK_SIM=fp4blocks`'s: every bit.
  - The random rows include subnormal rows, groups whose scale moves and groups of zeros.
- **Left on the table** (the coordinator's call: dsk::moe is the main model's MoE kernel, in B′'s A/B).
  - The shared expert's w1, w3 and w2 (8.85 MB a block a rank, 26.5 MB in all) stay on their rounded FP8
    in dsk::moe: about −0.056 ms a pass.
  - accept's wkv (gemv_fp8_norm, 2 MB) stays on its rounded FP8: about −0.004 ms.
  - So stage 4's model is −0.244 ms, not the study's −0.30. They are revisited only if the bundle lands
    just under 2 %.
- **The one-site prototype** (stage 2's job, b_fp4.log; f4ed2c7): wq_b's three GEMVs alone took 0.120 ms
  as FP8 and 0.041 as MXFP4 (−0.079), and the draft pass 0.049 ms less. wo_b's alone: −0.075 ms.
- **The GEMVs and the pass** (bench_dspark --blocks4, one rank, stand-in collectives, median of 50 × 5
  replays, all on the rounded weights), ms:

  | site (3 blocks) | FP8 | MXFP4 | change | MB |
  |---|---|---|---|---|
  | wq_a + wkv | 0.0143 | 0.0186 | +0.0042 | 6.88 → 3.66 |
  | wq_b | 0.1271 | 0.0389 | −0.0882 | 31.46 → 16.71 |
  | wo_a | 0.0660 | 0.0306 | −0.0354 | 25.17 → 13.37 |
  | wo_b | 0.1224 | 0.0462 | −0.0762 | 31.46 → 16.71 |
  | main_proj (accept's) | 0.0385 | 0.0263 | −0.0122 | 19.66 → 10.44 |
  | all | 0.3684 | 0.1606 | −0.2078 | 114.62 → 60.89 |

  | | draft | accept 1 + draft | accept 6 + draft |
  |---|---|---|---|
  | FP8, wo_a fused at 1,4 | 1.612 | 1.743 | 1.747 |
  | the FP4 blocks, wo_a at 1,4 apart | 1.434 (−0.178) | 1.536 (−0.207) | 1.541 (−0.206) |
  | FP8, wo_a at 1,1 apart | 1.606 | 1.743 | 1.744 |
  | the FP4 blocks, wo_a at 1,1 apart | 1.432 (−0.174) | 1.535 (−0.208) | 1.538 (−0.207) |

  - The pass with accept is 0.207 ms shorter (modelled 0.244): 85 % of the model.
  - With stage 3, accept_rows runs beside the verify's head GEMV, so the draft alone (−0.174 at wo_a 1,1)
    is the part that surely leaves the step's path. main_proj's −0.012 shows only as a lighter load beside
    the head. The replay below uses −0.174.
  - wq_a + wkv (448 rows) is slower as MXFP4, by 0.004 ms. It could stay on its rounded FP8 (exact, too).
  - The FP8 pass ran 0.06 ms faster on the rounded weights than on the real ones (1.612 against 1.675).
    The likely cause is the drafter's routed experts, which follow the values (the stand-in routes 9, 9
    and 6 distinct experts per block before the rounding). Each comparison above is on the same rounded
    weights.
- **Acceptance.** spectf ran over the study's 66 sequences on stage 3's drafter with the picked subset
  (top-32k), with `--draft-blocks4` (t5, tf_s4.log) and with `DSPARK_SIM=fp4blocks` on the FP8 kernels in
  its place (t6, tf_s4sim.log).
  - **0 of the 13,166 records differ** (drafts, survival, accepted, subset choice), in every group. On real
    weights and activations, the rare 1-ulp difference changed no draft, so it costs no acceptance.
  - The rounding itself moves acceptance, as the study's simulation did. Against stages 2 and 3's records
    at the same savings: edit-r2 −0.13 points; cat-structural −2.65, cat-code −1.67, cat-prose −1.49,
    bench-code −1.37 and bench-prose −0.78; bench-agentic +0.79, bench-json +0.77, and zh, ja and ru
    +0.38, +0.69 and +0.90.
- **The replay** (dp_stage4.py: today's chain over each stage's records, with each stage's measured saving),
  the gain in tok/s:

  | | edit-r2 | bench-prose | bench-code | bench-agentic | bench-json | cat-prose | cat-structural | cat-code | zh | ja | ru |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | stages 1 and 2 (top-32k) | +1.78 % | +2.29 % | +2.77 % | −0.34 % | +0.18 % | +1.36 % | +1.91 % | +1.33 % | +0.17 % | +1.48 % | +1.27 % |
  | stages 1 to 3 (−0.025 ms more) | +1.85 % | +2.39 % | +2.86 % | −0.26 % | +0.25 % | +1.46 % | +1.99 % | +1.40 % | +0.27 % | +1.58 % | +1.37 % |
  | stages 1 to 4, stage 4's records, no stage-4 saving | +1.72 % | +1.61 % | +1.49 % | +0.53 % | +1.02 % | −0.03 % | −0.66 % | −0.27 % | +0.65 % | +2.27 % | +2.27 % |
  | stages 1 to 4, −0.174 ms more (measured) | +2.20 % | +2.30 % | +2.09 % | +1.11 % | +1.52 % | +0.63 % | −0.11 % | +0.22 % | +1.30 % | +2.96 % | +2.94 % |
  | stages 1 to 4, −0.244 ms more (modelled) | +2.40 % | +2.58 % | +2.34 % | +1.35 % | +1.72 % | +0.90 % | +0.11 % | +0.42 % | +1.56 % | +3.24 % | +3.21 % |

  - The whole bundle replays at +2.20 % on the sampled edit bench, with no group slower than −0.11 %. The
    margin over the 2 % bar is thin; the bundle's A/B decides.
  - The study's model of the same levers (bundle C, −1.11 ms) was +2.71 %. The shortfall is in the
    savings: stage 1 −0.397 ms (modelled −0.439), stage 3 −0.025 (−0.09) and stage 4 −0.174 (−0.30: the
    shared expert and accept's wkv left out, and main_proj beside the head). Stage 2 saved more than
    modelled (−0.336 against −0.304).
- **The tokens** (s4). A CED server with every bundle-B flag on (top-32k): serve_test --ced passes (17 of
  17) and slice_test 0 failed.
  - Its start: free device memory after the prefill buffers 34.89, 34.23, 34.67 and 35.11 GiB on ranks
    0 to 3 (34.55, 34.25, 34.65 and 35.15 with every flag off), and a 12.00 GiB session store on every
    rank (rank 0 had 34.10 GiB free before it), as with the flags off.
- **The decode** (d4a to d4c): k1's decode (1,024 positions, 256 steps) on stage 3's drafter without the
  FP4 blocks (s3) and with them (b4), alternating over three rounds. Every run gives k1's hash. The step's
  GPU time is weighted by s3's steps of each row count, in ms:

  | round | s3, a step | b4, a step at s3's rows | change | s3, a token | b4, a token |
  |---|---|---|---|---|---|
  | 1 | 31.223 | 30.782 | −0.441 | 8.484 | 8.125 |
  | 2 | 31.155 | 30.807 | −0.347 | 8.467 | 8.135 |
  | 3 | 31.141 | 30.813 | −0.329 | 8.466 | 8.129 |

  - b4 is 0.372 ms a step faster at s3's mix of rows (0.407 at six rows). That is more than the bench's
    0.174 (the draft alone) or 0.207 (with accept), and more than the model's 0.244. The bench runs one
    rank on stand-in collectives, without the main model's weights beside the drafter's in L2 and DRAM.
    Why the real step gains more is not measured.
  - b4's drafts are the rounding's, so k1's steps verify a different mix of rows (65 steps against 68).
    Per token b4 is 0.343 ms faster, partly from that.
  - Replayed with −0.372 in place of −0.174, the bundle gains +2.76 % on the sampled edit bench, and its
    worst group is cat-structural at +0.53 %.
  - The figures rerun with:

    ```
    dp_stage4.py /data/dsgb10/bench/dpstudy/job /data/dsgb10/bench/dpbuild/stage2 \
      /data/dsgb10/bench/dpbuild/stage4 --s1 0.397 --s2 0.336 --s3 0.025 --saving 0.174 --model 0.244 \
      --s2-tag s2c
    ```
- **Files.**
  - attn-dpbuild: the premise's test 5c19390; the kernels c6502ec; fp4_tiles, round_fp4blocks and blocks4
    9f81842; the tests c21d68d, d100132 and 12f2187; the flag 8de8b7c; dp_stage4.py 45d8296 and 2f9fea0.
  - The job: rank 0's /data/dsgb10/bench/dpbuild/stage4. It holds the parts' scripts, lt4.log, b4.log,
    tf_s4.log, tf_s4sim.log, s4_*.log, the decode runs d{1,2,3}_{s3,b4}.log and stage4.txt. try1/ and try2/
    hold the first two lt4s, which failed on the premise and on the threshold judged per call. The binaries: dsgb10_rank and bench_dspark at
    d100132, lite_tests at 12f2187.

#### Bundle B's A/B: under the bar (2026-10-13)

Bundle B against today's drafter on production's server, both on a binary with B′ (attn-dpbuild 4804db6, master
ce75a2d merged).

**Verdict: it fails. +1.50 % end to end on the sampled edit bench (ab1 +1.09 %, ab2 +1.91 %) against the 2 % bar.**
- Every request's tokens are equal in both arms.
- One other group misses as well: bench-code, −2.16 % in ab2 (−1.35 % in ab1, −1.76 % mean). It loses 6.9 % of
  its accepted drafts a step live, on the study's own tokens.
- Nothing more is merged: stage 1's flags were already on master, off by default (ccec056). The wiring that would make the bundle the default is kept on attn-dpwire, unbuilt. It holds
  `--no-draft-approx`, the subset compiled in, and DESIGN.md decision 12, marked not adopted.

- **How it ran** (dpab_serve.sh, dpab_load.py and dpab_cmp.py):
  - Each arm got a server session per part, with production's arguments (`--max-ctx 262144 --disk-gib 0`). On:
    every bundle-B flag, with top-32k. Off: none.
  - edit2: the sampled edit bench (126 turns: r2-off repeats 0–2 and r2-max repeat 0), replayed in order at
    T 1.0 and top-p 0.95. Each request is seeded by its case, so both arms sample the same tokens.
  - rest: the greedy r1 edit bench (54 turns), the four bench types twice, and the study's 20 catalog and 4
    language prompts, 512 tokens each.
  - Two jobs in opposite orders: ab1 off then on, ab2 on then off. End to end is the engine's prefill plus
    decode.
  - Sanity first (sa, every flag on): the 7 reference hashes, serve_test --ced (17 of 17) and slice_test (0
    failed).
- **Results** (dpab_cmp.py): the edit benches' end-to-end time, the other groups' tok/s.

  | group | ab1 off | ab1 on | ab1 gain | ab2 off | ab2 on | ab2 gain | gain, mean |
  |---|---|---|---|---|---|---|---|
  | edit-r2 (126) | 146.1 s | 144.5 s | +1.09 % | 146.8 s | 144.0 s | +1.91 % | +1.50 % |
  | edit-r1 (54) | 62.1 s | 61.0 s | +1.77 % | 62.0 s | 61.0 s | +1.62 % | +1.69 % |
  | bench-prose (2) | 75.6 tok/s | 76.9 tok/s | +1.69 % | 75.6 tok/s | 76.0 tok/s | +0.54 % | +1.11 % |
  | bench-code (2) | 111.4 tok/s | 109.9 tok/s | -1.35 % | 111.5 tok/s | 109.1 tok/s | -2.16 % | -1.76 % |
  | bench-agentic (2) | 109.0 tok/s | 111.4 tok/s | +2.17 % | 108.9 tok/s | 111.5 tok/s | +2.35 % | +2.26 % |
  | bench-json (2) | 149.5 tok/s | 153.7 tok/s | +2.80 % | 149.6 tok/s | 152.0 tok/s | +1.63 % | +2.22 % |
  | cat-prose (6) | 82.9 tok/s | 84.3 tok/s | +1.65 % | 82.9 tok/s | 82.9 tok/s | +0.04 % | +0.85 % |
  | cat-structural (6) | 116.9 tok/s | 117.9 tok/s | +0.92 % | 115.8 tok/s | 116.1 tok/s | +0.22 % | +0.57 % |
  | cat-code (8) | 147.2 tok/s | 150.9 tok/s | +2.46 % | 145.4 tok/s | 148.4 tok/s | +2.03 % | +2.24 % |
  | zh (2) | 86.3 tok/s | 88.5 tok/s | +2.62 % | 85.7 tok/s | 88.5 tok/s | +3.31 % | +2.96 % |
  | ja (1) | 77.6 tok/s | 78.9 tok/s | +1.68 % | 77.7 tok/s | 79.0 tok/s | +1.66 % | +1.67 % |
  | ru (1) | 90.4 tok/s | 93.5 tok/s | +3.40 % | 90.5 tok/s | 93.6 tok/s | +3.44 % | +3.42 % |

- **The order matters.** The two jobs differ by up to 1.6 points in a group (cat-prose: +1.65 and +0.04 %).
  - On the sampled edit bench, the arm run second was the slower one in each job. Prefill: +0.53 % (ab1's on arm)
    and +1.18 % (ab2's off arm). Decode: by about 0.3 %.
  - The two jobs' mean cancels a drift that follows the order. One job alone would have been off by about 0.4
    points on the sampled edit bench.
- **Where the time went** (edit2, both jobs pooled; dpab_acc.py and dpab_proj.py):
  - **Decode +1.88 %.** At the off arm's mix of rows, the step's GPU time is 0.88 ms shorter (33.83 → 32.95 ms).
    Fewer drafts verified take 0.12 ms more off, and the wall time a step falls 1.11 ms. The stages benched
    0.918 ms together (−0.397, −0.322 on the subset's steps, −0.025, −0.174).
  - **But 1.29 % more steps.** Accepted drafts a step fell from 3.920 to 3.859 (−1.55 % ± 0.45 over sessions
    resampled). The drafts are deterministic, so both jobs give the same steps.
  - **Prefill −0.32 %:** the order's effect, above.
- **The bar is end to end; the replays were decode only.** dp_study.py and the dp_stage scripts replay decode
  tok/s.
  - Prefill is 24 % of the sampled edit bench's time (the off arm). A decode gain g is about 0.76 g end to end,
    and the bar needs +2.65 % decode.
  - Stages 1–4 replayed +2.20 % decode at the bench's savings, which is +1.67 % end to end. The bundle was under
    the bar before its A/B.
  - The study's +2.36 % (bundle B) and +2.71 % (C) were decode gains too: +1.8 and +2.0 % end to end. The
    stages' verdicts missed this.
- **Why acceptance fell 1.55 %, against the replay's 0.8 %.** This uses the data at hand, with no new runs
  (dp_acc.py for the replay, dpab_acc.py for the A/B).
  - **The levels.**
    - The replay's 3.75 counts each sequence's one-row step 0, which the server's verify steps do not. Without
      it, 3.89.
    - The A/B's 3.858 counts the two warm-ups. Over the 126 turns it is 3.920: r2-off repeats 0, 1 and 2 at 4.02,
      4.07 and 4.10, and r2-max at 2.97.
    - The original r2 run's live rate on repeat 0's own samples was 3.96.
  - **The workloads differ in their samples, not their tasks.**
    - t5's records are repeat 0's 38 turns (12 tasks), teacher-forced along that run's samples (4,974 positions).
    - The A/B takes the same 38 prompts, plus repeats 1 and 2 (75 turns) and r2-max (13). It samples every
      continuation anew under its own seed (15,515 tokens).
  - **The drop is broad, and the replay's was small-sample.**
    - 29 of the 40 sessions lose acceptance, 7 gain and 4 are unchanged.
    - Each repeat loses about as much. Repeat 0's prompts: −1.50 % ± 0.73. Repeats 1 and 2: −1.41 ± 0.76 and
      −2.29 ± 0.45. r2-max: −0.25 ± 2.12.
    - The replay's −0.74 % (without step 0) is ± 0.30 over its 12 tasks. The gap, 0.8 ± 0.55 points, is 1.5
      standard deviations.
  - **The walk is close to live, but not exact.** The bench types are the study's sequences token for token.
    - Live and replayed steps agree within one step in 7 of the 8 arm-sequences. The exception is bench-code's
      off arm: 160 steps live against 166 replayed. So bench-code loses 6.9 % of its accepted drafts a step live,
      and 1.8 % replayed.
    - Over the four, live steps rise 1.55 % and replayed ones 0.70 %.
    - Elsewhere the replay errs both ways. The catalogs lose 0.8–2.0 % live against 3.1–3.5 % replayed (on their
      first 256 tokens). So the replay is not biased one way.
  - **Not T 1.0.** The greedy r1 edit bench loses as much live: −1.51 % ± 0.62. The replay's trace was sampled
    at T 1.0 too, and there the subset cost 0.13 points beyond stage 1 (−0.25 % against −0.12 %).
  - **The FP4 blocks carry most of the replayed cost.**
    - On the edit bench they cost −0.49 % of the −0.74 % (−0.62 % ± 0.30 alone on today's drafter). On the
      catalogs they cost −2.5 to −3.8 %.
    - If the live excess sits in one lever, they are the likeliest. This A/B cannot split it.
  - **The likeliest reading:** the bundle costs about 1.5 % of acceptance on edit work, and the replay's 12 tasks
    understated it.
- **Sub-bundles, projected** (dpab_proj.py; no runs). The model:
  - Decode is steps × the wall time a step. The tokens are fixed, so the steps follow the accepted drafts a step.
  - Each sub-bundle's acceptance comes from its own replayed records: stage 1; stage 2's; DSPARK_SIM=all for
    "no subset"; fp4blocks; stage 4's. It is taken as replayed, or ×2.1 (the A/B's −1.55 % over the replay's
    −0.74 %).
  - The saving: the stages' bench savings scaled ×1.07 to the A/B's 0.98 ms a step. As a variant, stage 4 at
    its decode's −0.372 and the rest scaled to fill the 0.98 ms.
  - The rows' share is set by the replay's change in drafts verified. Prefill is 24 %, unchanged.

  | sub-bundle | saving a step (ms) | acceptance: replayed; ×2.1 | end to end, central (range) | with no acceptance cost |
  |---|---|---|---|---|
  | stages 1 and 3 (head4, markov8, woa 1,1, accept-side) | 0.45 (0.35) | −0.12 % ± 0.23; −0.25 % | +0.8 % (+0.6 to +0.9) | +1.0 % |
  | stages 1–3 (no FP4 blocks) | 0.80 (0.61) | −0.25 % ± 0.25; −0.52 % | +1.4 % (+1.0 to +1.6) | +1.7 % |
  | stages 1, 3 and 4 (no subset) | 0.64 (0.72) | −0.74 % ± 0.40; −1.55 % | +0.7 % (+0.7 to +1.2) | +1.6 % |
  | stages 3 and 4 (the FP4 blocks) | 0.21 (0.39) | −0.62 % ± 0.30; −1.30 % | −0.1 % (−0.1 to +0.3) | +0.8 % |
  | the bundle | 0.98 (measured) | −0.74 % ± 0.30; −1.55 % (measured) | +1.45 % (+1.50 % measured) | +2.1 % |

  - **None reaches 2 %.** The bar needs a step 0.92 ms shorter at no acceptance cost, or 1.35 ms shorter at the
    bundle's −1.55 %. The bundle saves 0.98 ms.
  - Only the whole bundle at no acceptance cost would clear it (+2.1 %).
  - Stages 1–3 (no FP4 blocks) project about as well as the whole bundle: +1.4 % (+1.0 to +1.6 %). The FP4
    blocks bring most of the replayed acceptance cost along with their 0.2 ms.
  - The A/B's ±0.45 % in acceptance adds about ±0.3 points to each projection.
- **Files.**
  - The A/B's binary: dsgb10_rank at attn-dpbuild 4804db6.
  - The job: rank 0's /data/dsgb10/bench/dpbuild/ab. It holds the scripts (go.sh, sa.sh, p.sh, chain.sh); sa_*.log
    and h_*.log; and ab_{ab1,ab2}_{off,on}_{edit2,rest}.json, .log and _server.log. try1/ holds the first sanity
    try, which failed to start (sa.sh not executable).
  - The analysis: engine/dspark/tests/dpab_acc.py, engine/dspark/ref/dp_acc.py and engine/dspark/ref/dpab_proj.py
    (attn-dpbuild 10664ca).
  - The figures rerun with:

    ```
    dpab_cmp.py /data/dsgb10/bench/dpbuild/ab ab1 ab2
    dpab_acc.py /data/dsgb10/bench/dpbuild/ab ab1 edit2            # and rest; --key task per session
    dp_acc.py edit-r2 bench-prose bench-code bench-agentic bench-json cat-prose cat-structural cat-code
    dpab_proj.py /data/dsgb10/bench/dpbuild/ab edit2 edit-r2 ab1 ab2
    ```

### A paced L2 prefetch in the step's windows (2026-10-12)

B′: [the collectives' proposal](#the-collectives-time-a-proposal-2026-10-12)'s option B (read the next weights
while a collective waits) from a side stream instead of the collective's blocks. While a collective, or the
attention, leaves DRAM idle, one warp prefetches into L2 the weights the step reads next. It needs no whole SM,
and it writes nothing the model reads, so the bits cannot change. The bar: ≥ 2 % end to end on the sampled edit
bench, no bench type or case > 2 % slower, both A/B jobs agreeing, the hashes unchanged.

**Verdict: adopted.** The engine A/B (two jobs in reversed order, the flag off and on in one binary) gives
−3.95 % over the bar's 36 cases (geometric mean; job 1 and job 2: −4.16 and −4.50 % at k = 1, −3.54 and −3.59 %
with drafts). Every case is faster, the least by 2.64 %, and the sampled edit bench gains 3.0 % with drafts and
4.0 % at k = 1. Every request's tokens are equal, and the 7 hashes and serve_test hold with the flag on.

It is now `model::Options::l2pf`, on by default (`--no-l2pf` turns it off). Built from that commit (ce55e73,
bins `*_l2f`), the 7 hashes hold with the default, serve_test `--ced` passes 17 of 17, and `--no-l2pf` gives
k = 1's hash at 17.511 ms a token, against 16.687 with the default (FP8 drafts 8.291, bf16 drafts 8.404).

Sources:

- bench_chain at six rows on the edit bench's routes (`rt_g1.bin`, 24.3 experts a layer), and at one and three
  rows;
- `micro/l2_retention.cu`;
- rank 0 of the engine under nsys in the 1k decode loop, with drafts and at k = 1;
- the engine with the hook, off and on: the hashes, serve_test, and abserve.sh's A/B.

**Where the six-row step goes.** bench_chain's chainprof at six rows (bins at fae253c): 33.3 ms a step.

| class | exclusive ms | |
|---|---|---|
| weights | 28.93 | 6.50 GB a step at 225 GB/s |
| collectives (stand-ins) | 3.40 | |
| attention | 0.79 | |
| mHC, gaps | 0.04 | |

- The weight sites below the DRAM's ~235 GB/s are the small ones between collectives: wq_a + wkv at 137 GB/s
  (0.67 ms), the gate at 163 (0.96 ms), wq_b at 200 (1.68 + 0.63 ms with the indexer's). The MoE runs at 224
  (up) and 249 (down), wo_b at 246, wo_a at 214.
- So the room is the windows, not the GEMVs' rate: at six rows the stand-ins add up to 3.4 ms a step of idle DRAM.
- The coded bundle holds here: raw experts and head cost 34.98 against 33.25 ms (+5.2 %); on the bench's routes
  (22.1 experts) 33.31 against 31.65. The old case, 34 experts a layer with raw experts and head: 42.5 ms.

**What L2 does with a paced prefetch** (`micro/l2_retention.cu`, beyond 10-12's retention test):

- *Latency beside it* (`lat`): a chain of dependent 128-byte loads takes 427 ns a link alone; 436 beside a
  prefetch paced in waves of 8 chunks at 175 GB/s; 558 in waves of 32 and 684 in waves of 128 (230 GB/s); 2049
  beside four blocks issuing at once. Small waves cost the GEMVs nothing; bursts do.
- *Reads between* (`between`): 3.9 or 10.5 MiB prefetched, then other bytes read by 48 CTAs, then the
  prefetched bytes: reads with `ld.global.nc.L1::no_allocate` (the GEMVs' weight loads) evict none of it up to
  32 MiB between; `ld.global.cg` reads of 16 MiB keep 59–83 %, of 32 MiB 1–3 %.
- *Eviction* (`evict`): a 4 MiB working set (written or read by `ld.cg`) keeps ≥ 94 % of its L2 benefit after
  a 10.5 MiB paced prefetch, 76 % after 16 MiB; a 1 MiB set 89–99 %.
- So the windows' prefetch, ≤ ~10 MB between two GEMVs, leaves the KV and activations alone, and the GEMVs'
  streaming reads do not push it out before it is used.

**The windows.** Rank 0's step in the 1k decode loop with drafts (the earlier e6 capture), from the last weight
kernel's end to the window's end, median (5th percentile), µs:

| rows | W1 MoE all-reduce | W2 all-gather | W3 attention | W4 attention all-reduce |
|---|---|---|---|---|
| 2 to 6 | 33.1 (12.1), 75th 44.3 | 14.2 (7.9), 75th 19.1 | 19.1 (13.5) | 13.6 (7.4), 75th 18.0 |
| 1 | 15.5 (5.9) | 17.6 (15.2) | 12.5 (10.1) | 13.1 (8.1) |

They are shorter and spread wider than bench_chain's stand-ins (the all-reduces 14 + 1.2 (R − 1) and
18 + 5.1 (R − 1) µs, the all-gather 18, the attention 14, 37 on the indexer's layers), and a slower rank
waits less. Hence the stop on arrival below.

**The variants** (bench_chain, step medians; base about 33.1 ms at six rows, 17.5–17.8 at one, 24.2 at three):

1. *At the GEMV sites* (`pf`, `pfa`, `pfb<KB>`, `pfo`, `pfg`, `pfq`): the next GEMV's bytes from the GEMV
   before it. Best at six rows `pf+pfa+pfb8192` −1.9 %, at one row `pf+pfa+pfb3072` −2.8 %; `pfq` +0.3 %;
   bigger waves (`pfw16`) or a lower pace (`pfr170`) worse. Under the bar.
2. *A queue over the windows* (`pfT<targets>`, `pfW<windows>`, `pff<fill %>`): each window prefetches the
   next bytes of the layer's targets that no earlier window took, as many as its stand-in's time holds at the
   pace. Unpaced, `pfTgox+pfW1234` (wo_a, wo_b, the gate in all four windows) −3.5 % at six rows,
   `pfTqbgox+pfW1234` −4.5 % at one. Paced in waves of 8 at 170 GB/s only −1.8 % and −0.7 %: the waves
   reached ~133 GB/s in place, fell behind, and ran past their windows.
   - *An overrun is expensive.* A 5.8 µs overrun of W4 cost `k_moe_up` +0.76 ms a step: a prefetch block
     still resident when wo_b, the MoE kernels or the one-row persistent GEMVs launch takes the one SM their
     512-thread, ~125-register CTAs need whole.
   - *wq_b's targets (`q`) lose at six rows:* the classic FP8 GEMV from two rows gains nothing from L2, and
     the prefetch moved wq_b's tail and so wo_a's CTA placement, which left wo_b no whole SM to start on (nsys
     of `pfq` against `pfq+pfe`, which launches the same graph and issues nothing: the topology is not it).
3. *A token bucket* (`pfk`: one warp, each poll issues the chunks due by now, ≤ 32): at six rows
   `pfk+pfTgox+pfW1234+pfr200+pff85` −3.6 %, at one row with wq_b (`pfTbgox`) −6.1 %, at three rows −5.6 %.
   Fill 80–90 % and pace 170–230 all land within 0.25 ms of it.
4. *The engine's plan* (`pfx`, kernels/l2pf.cu itself, stopped at each stand-in collective's arrival; `pfxn`
   the same on its budgets alone), bins at 7c16c3a:

   | rows | base, ms | pfx | pfxn | pfk bound |
   |---|---|---|---|---|
   | 6 | 33.141 | 31.701 (−4.3 %) | 32.455 (−2.1 %) | 31.540 (−4.8 %) |
   | 1 | 17.667 | 16.773 (−5.1 %) | 17.174 (−2.8 %) | 16.569 (−6.2 %) |
   | 3 | 24.145 | 23.105 (−4.3 %) | 23.023 (−4.6 %) | 22.962 (−4.9 %) |

   Under nsys, `pfx`'s launches end 0.7–1.2 µs before the stand-in ends (its arrival is 2 µs before): the stop
   works. But 18 % of the all-gather's launches at six rows (8 % at one) start more than 5 µs late: the
   all-gather's PDL lets wq_b's CTAs fill every SM while it waits, and a prefetch launched after them gets an
   SM only when the all-gather's own blocks leave. Started after its collective finished, it cannot tell that
   from the collective before, runs its budget into wq_b (90th percentile 27 µs past the window), and is slowed
   by it.

- *The eviction gap.* The stand-ins read no KV and no activations, so the bench cannot see what the
  prefetch evicts there. `evict` bounds it: the attention's KV reads are top-k bounded (a few MB a layer),
  and ≤ 10.5 MiB of prefetch keeps ≥ 94 % of such a set. The engine A/B is the measure.

**The hook** (kernels/l2pf.cu; model.cu's `Model::Impl::l2pf`). In the checks and the A/B below it was off by
default and turned on by `DSK_L2PF=1`; it is now `Options::l2pf`, on by default.

- One warp on `sc`, a high-priority stream forked from the main stream after the window's last weight
  kernel and joined after the layers, inside the captured graphs. Off, there is no stream and no launch.
- The plan: wo_a, wo_b and the router gate; at one row wq_b (and the indexer's wq_b) too. 8 KiB chunks of
  `cp.async.bulk.prefetch.L2`, a token bucket at 200 GB/s. A layer's windows run in order and share its
  queue through `L2pfState` (device memory), so each starts where the last ended.
- Budgets (µs at the pace): W1 18 + 5.1 (r − 1), W2 18, W3 11 (30 on the indexer's layers), W4 14 + 1.2 (r − 1),
  r = min(R, 8): about the 75th percentile of the engine's collective windows, and the 5th of the attention's.
- The stop: W1, W2 and W4 end once their collective has arrived. The ring's `DevChannel::ready` (the last
  collective whose arrival block 0 saw, set by multi-block grids) changed since the launch, or is ahead of
  `seq` (the last finished: collectives run one at a time, so the one in flight has arrived, perhaps before
  the launch). W3 has no collective and runs its budget.
- Code: 9246d07 (l2pf.cu), 2abc647 (the hook), b05f018 (bench_chain `pfx`, `pfxn`; the other variants 1ea3f1d,
  6318e2e, 15783de, 88f00c4, `micro/l2_retention.cu` 1ea3f1d and 6318e2e), 7c16c3a
  (abserve.sh's server args); ce55e73 (the engine option).

**The engine, off and on** (bins `*_l2c` from 7c16c3a, one binary, `--setenv DSK_L2PF=1`):

- All 7 hashes, off and on: tf 842278b28d87dad0, 1k k = 1 febc7dc54ee40a39, FP8 and bf16 drafts
  febc7dc54ee40a39 (the spec decode with the flag on matches k = 1's), temp a8f7a08f8cd1c5e8, 16k CED
  81f772ab79a950e4, 28k CED 14650fb0739d0383 (until 2026-10-13 the hash's seed, see
  [Golden check](#golden-check); now 8f08a153d415c5c8). serve_test `--ced`: 17 of 17, off and on.
- The single runs' ms a token, off → on: k = 1 17.470 → 16.737 (−4.2 %); FP8 drafts 8.631 → 8.260 (−4.3 %);
  bf16 drafts 8.714 → 8.407 (−3.5 %); temp 17.523 → 16.778 (−4.3 %); 16k CED 17.605 → 16.802 (−4.6 %).
  The drafts are the same (72 verify steps, 2.542 accepted a step).
- Rank 0 under nsys, 1k decode: k = 1 steps 17.546 → 16.788 ms (median of 128, −4.3 %); spec steps of two to
  six rows (the drafter's pass after the join) 31.28 → 30.13 ms (median of 40, −3.7 %; mean −2.2 %).
  At one row the persistent wq_a + wkv and wq_b go 47.7 → 38.3 µs each, wo_a 48.4 → 44.5, wo_b 94.5 → 87.7;
  the attention's kernels 14.9 → 17.0 (`attn_kernel`), 16.8 → 21.2 (`index_dense`), 15.5 → 16.1 (combine).

**Overruns** (the same captures; a launch's end against its collective's end, which comes a few µs after the
arrival; the next GEMV's CTAs launch early under PDL and wait, so its start says nothing):

| window | rows | launches | ends after the collective | > 2 µs after | starts after it ended | end − collective end, median (p90) |
|---|---|---|---|---|---|---|
| W1 | 2–6 | 1558 | 2.4 % | 0.2 % | 0 | −2.6 (−1.3) µs |
| W2 | 2–6 | 1600 | 2.6 % | 0.1 % | 0.1 % | −1.5 (−0.7) |
| W4 | 2–6 | 1602 | 1.8 % | 0.9 % | 0.1 % | −2.4 (−1.1) |
| W1 | 1 | 5029 | 1.4 % | 0.4 % | 0 | −1.2 (−1.0) |
| W2 | 1 | 5160 | 0.2 % | 0.0 % | 0 | −1.4 (−0.6) |
| W4 | 1 | 5162 | 4.0 % | 0.4 % | 0.0 % | −2.1 (−0.4) |

- The stop holds in the engine: the late starts that broke it in bench_chain's all-gather are 0.1 % here (the
  engine's all-gather runs 24–42 µs, the stand-in 18). From ready to the block's exit: in bench_chain, where the
  arrival is 2 µs before the stand-in's end, the prefetch ends 0.7–1.2 µs before it, so ~1 µs.
- *W3 overruns.* The attention's window has no stop, and its prefetch outlives its 11 µs budget: 17.2 µs at the
  median at two to six rows, 17.4 at one. Beside the attention's reads its 268 chunks issue at ~130 GB/s, not
  200, and the kernel issues all of them however late. It ends after the attention in 66 % of launches at two to
  six rows (+1.8 µs at the median, +9.2 at the 90th percentile) and in 78 % at one row (+7.2 µs), into wo_a's
  reads. At one row the attention's kernels' spans (which include their PDL wait) grow 0.5–4.3 µs.

**The A/B** (abserve.sh and abload.py, bins `dsgb10_rank_l2c`, the flag off as the baseline; job 1 off then on for
k = 1, on then off with drafts, job 2 the reverse; results in rank 0's /data/dsgb10/kernels/abload/l2pf). Change in
time, the flag on against off (negative: faster), %:

| case | k = 1, job 1 | job 2 | drafts, job 1 | job 2 |
|---|---|---|---|---|
| bench prose | −5.38 | −6.26 | −4.32 | −4.37 |
| bench code | −4.84 | −5.56 | −4.21 | −4.23 |
| bench agentic | −4.47 | −5.04 | −4.25 | −4.20 |
| bench json | −4.45 | −5.88 | −3.95 | −4.11 |
| sampled edit bench, 54 turns | −4.11 | −3.87 | −2.96 | −3.04 |
| (its decode) | −4.59 | −4.33 | −3.65 | −3.64 |
| 16k +200 turn | −3.77 | −3.74 | −3.42 | −3.07 |
| 16k +2000 turn | −3.25 | −3.14 | −2.64 | −2.78 |
| 26k +200 turn | −3.63 | −3.71 | −3.17 | −3.49 |
| 26k +2000 turn | −3.55 | −3.25 | −2.91 | −2.98 |
| the bar's 9 cases, geometric mean | −4.16 | −4.50 | −3.54 | −3.59 |

- Every request's tokens are equal (82 of 82 in each of the four pairs), and the edit bench's 54 turns equal its
  trace in all eight sessions. Prefill does not move: the edit bench's prefill and every cold, +200 and +2000
  TTFT are within ±2 % (the most +1.37 %).
- With drafts the sampled edit bench gains less than the bench types (−3.0 % against −4.0 to −4.4 %): its turns
  carry prefill, and its decode gains −3.6 %.
- *Other jobs inside the pairs.* ringlock.sh's waiting wants go oldest first, so a held want does not keep other
  agents' jobs out of the gaps between parts. ring.owner and the drive logs show the attention line's jobs inside
  three of the four pairs:
  - b.sh (22:20) inside job 1's pair with drafts;
  - h.sh and stage 3's d.sh (22:33, 22:38) inside job 2's pair with drafts;
  - s.sh (22:59) inside job 2's k = 1 pair.

  Others ran between pairs (lt.sh at 22:00, during the checks, and 22:16; t.sh at 22:23; h.sh at 22:46).
- *The jobs agree, with one flag.* With drafts they agree within 0.35 points a case. At k = 1 their geometric
  means are 0.34 points apart, but job 2's k = 1 pair, the one with s.sh inside it, is where they differ most:
  bench json −4.45 against −5.88 %, prose −5.38 against −6.26 %. Its off part ran 0.4–0.8 tok/s slower than
  job 1's, so its bench gains may be up to ~1 point high. Job 1's k = 1 pair had no other job inside it and
  clears the bar alone (−4.16 %, the least −3.25 %), and the edit bench agrees across the jobs (−4.11 and −3.87
  %).

**Not done, or tried and dropped.**

- *A cap at the budget's time, and W3's budget* (measured after the A/B, 2026-10-13). The experiment's knobs
  `DSK_L2PF_CAP` (each launch ends at its budget's time, not when its chunks are issued) and `DSK_L2PF_W3` (W3's
  budget in percent) were committed on branch kernels-l2cap as c2b1978 and reverted in 1ea18c8 (bins `*_l2h`).
  The 1k decode loop ran three rounds, the order reversed in the middle one. The table gives each variant's
  median ms a token against today's default (16.72–16.77 at k = 1, 8.28–8.29 with FP8 drafts). The hashes are
  equal in every run.

  | variant | k = 1 | drafts |
  |---|---|---|
  | the cap | +0.38 % | +0.34 % |
  | the cap, W3 at 50 % | +0.91 % | +0.62 % |
  | the cap, no W3 | +1.87 % | +1.00 % |
  | W3 at 150 % | −0.07 % | −0.06 % |
  | W3 at 200 % | −0.24 % | −0.23 % |
  | W3 at 300 % | +0.31 % | −0.42 % |

  - W3 is worth ~1.9 % at k = 1 and ~1.0 % with drafts, its overrun included: the bytes it brings for wo_a and
    wo_b outweigh what it costs the attention. A larger budget is within the noise; the default's own rounds
    spread 0.45 %.
  - The cap barely shortens W3. Under nsys (rank 0, the cap on), W3 lasts 15.0 µs at the median at two to six
    rows (17.2 without) and 16.7 at one row (17.4). It ends after the attention in 50 % and 78 % of launches.
  - A capped launch still ends 4–6 µs after its cap. That would be its last poll's issues (up to 32 chunks)
    and perhaps the prefetches in flight; this was not measured. The cap's loss is then most likely the late
    chunks it cuts from the collectives' windows.
  - Dropped: the default stays (no cap; W3 at 11 µs, 30 on the indexer's layers).
- *Late starts numbered.* A version that numbers each window's collective (the previous window's plus the
  collectives between, re-anchored on each arrival seen) and issues nothing when its collective has already
  finished. Written and compiled, dropped: the engine's late starts are 0.1 %.
- *The eviction gap* stays a bound (above), but the A/B's equal-or-better attention-heavy cases (the 26k turns)
  show nothing it missed.
