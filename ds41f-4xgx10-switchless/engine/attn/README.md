# engine/attn: attention-side kernels (M3, part 2)

One rank at TP=4 (16 of the 64 query heads), sm_121a, for decode (1–6 rows), verify rows and prefill
chunks. The FP8 projections around these kernels (wq_a, wq_b, wkv, the indexer's wq_b, wo_a, wo_b) are
engine/kernels'; everything here starts from their bf16 outputs. API and layouts: [attn.h](attn.h).

| File | Kernels |
|---|---|
| rope.cu, prep.cu | RoPE tables (theta 1e4; 1.6e5 + YaRN), bit-identical to model.py's freqs_cis; q_norm; kv_write (kv_norm, RoPE, fp8 e4m3 + ue8m0/32 into the ring) |
| gemv.cu | bf16 GEMV (compressor projections, weights_proj) |
| compressor.cu | pooling (ratio 2, carry of the last 16 rows), RMSNorm, index key (wk, k_norm, RoPE, fp4 + ue8m0/32), KV entry (RoPE, fp4 + e4m3/16) |
| indexer.cu | q prep (RoPE, fp4); scores with `mma.sync…kind::mxf4nvf4.block_scale` m16n8k64; torch.topk-exact select (histogram + ordered gather); layer 20's pool; Reindex over the pool; rank merge |
| sparse_attn.cu | window + selected entries, online softmax (P in bf16, sink in the denominator), split-K + combine, fused q RoPE and output inverse RoPE; DSpark via per-row window bounds |
| mhc.cu | hc_fn projection (column range per rank), Sinkhorn (torch's op order, 20 iters), hc_pre (+ norm), hc_post |

State: window ring of 256 slots per layer (`slot = (pos - base) & mask`; a prefill chunk can use a
linear buffer with mask -1 holding the 127 positions before it plus the chunk). Compressed KV
[entries][256 B e2m1 + 32 B e4m3] and index keys [entries][64 B e2m1, words transposed 4x4 + 4 B
ue8m0], one cache per kv source (2, 8, 14, 20), replicated on every rank. DSpark drafts are written
into their stage's ring at their own positions; the draft rows attend to `[max(0, start-127), start+5]`.

Session slots (batching, [batching.md](../../docs/batching.md)): each of these state structs can stand for
several sessions' states, slot s at its pointers + s × stride, with a device table giving each row its
session's slot (a null table: slot 0, the one-session calls). A session's rows are consecutive and ascend in
position; sessions follow one another in any order, at unrelated positions. Every row computes, bit for bit,
what it computes in a step of its session alone.
- The indexer scans each session's own keys as far as its last row's length. A CTA's rows never span two
  sessions, and index_dense shares its fixed grid among the sessions by rows × key chunks, so one graph per
  row count serves any mix. IndexKeys' strides are whole entries (k_stride = 16 × sc_stride): a row's keys
  are an entry offset from the pointers. Without a table, index_dense and index_fused run instances compiled
  for one session; index_score (prefill's separate path) takes none.
- compressor_emit takes at most 16 rows with a table; kv_write_sets also takes each row's own position (B3).
- `slot_tests` (no data): steps of 2–4 sessions in shuffled slots, up to 8 rows, against each session alone
  (outputs, window writes, compressed entries and keys, carries, the top-k and the pool), and that each run
  wrote; `--no-table` is its negative control. Every comparison identical on rank 0 (2026-10-12). B4 adds
  steps of 9–16 rows in 2–4 sessions.
  `bench_index --sessions S` times the dense scans with the rows split into S sessions (`--table`: one too).

Up to 16 rows a step (B4, `kMaxDecodeRows`). Up to 8 rows every launch runs the kernel it ran before, with
the same SASS. From 9 rows, the kernels that hold the rows in shared memory or registers take them in groups
of up to 8 (`kDecodeGroup`), and each row computes what it computes in a call of at most 8 rows:
- `compressor_project_rows`: a second pass of 8 rows through the shared memory, over the weight row the
  warp holds in registers.
- `decode_prep`: the head weights in two groups.
- `index_dense`: a CTA's group has at most 8 rows, with their queries and histograms in shared memory
  (82 KiB). A session of more than 8 rows reads its keys once per group.

A session has at most 8 rows in a call. `decode_prep` runs the emit and the carry in one launch, so with 16
rows of a session its first row's read of a carry slot would race with its 16th row's write of the same
slot (attn.h). `decode_prep` dies past 8 rows without a slot table; with one, the step's builder checks.
Everything else takes 16 rows as it took 8 (the grids follow the rows).

## Running

```
cmake -S engine -B engine/build-attn -DCMAKE_BUILD_TYPE=Release && cmake --build engine/build-attn -j 16
bash engine/attn/tests/gen.sh gen_attn.py l20 --layers 20,21,24,25 --T 600 --decode 4 --index-topk 64 --cand-blocks 16 --check
bash engine/attn/tests/gen.sh gen_attn.py long32k --layers 2,3,20,21,24,25 --T 32000 --decode 3 --state-prefill --own-index-k
bash engine/attn/tests/gen.sh gen_dspark.py dspark --stages 0,2 --T 200 --decode 3 --check
bash engine/attn/tests/gen.sh gen_mhc.py mhc --layers 2,30 --T 64
./engine/build-attn/attn/attn_tests ~/dsgb10-data/attntests/<case>
./engine/build-attn/attn/slot_tests                                          # several sessions per step
./engine/build-attn/attn/bench_index --iqm [--sessions 1,2,4,8] 16384 131072  # the decode indexer
./engine/build-attn/attn/bench_attn [--detail] 1024 16384 131072 1048576     # decode, per rank
./engine/build-attn/attn/bench_attn --prefill 2048 --detail 16384            # one prefill chunk
```

Generators run model.py with torchref in the canary-roce image on the real checkpoint (`--check`
asserts the traced copy equals model.py). Cases in `~/dsgb10-data/attntests/`: l0, l2, l20 (T=300/600,
topk 64), long32k (T=32000, topk 512, 2048-block pool, decode only), dspark, mhc, stale_index_k.
`attn_tests` on `stale_index_k` fails two checks by design (`L2_c1_iscore`, `L2_c1_itopk`): the case
reproduces [model.py's bug](#accuracy-against-modelpy), which the kernels do not copy.

## Jobs on the ring (tools/)

Scripts for measurements that need the ring, from the prefill MoE and mHC validation (2026-10-12). Each
script's header has its usage.
- `ringparts.sh DIR NAME=CMD...`: a job in parts, each its own `ring.sh` call (its own borrowed window). A
  part that a request preempted (exit 76) or whose wait was given up (75) runs again; finished parts are
  recorded in DIR and skipped when the job runs again. Between parts it waits `GAP` s (5), so that other
  jobs waiting for the ring get it in between. `steps.sh` (sourced) does the same for the steps inside a
  part.
- `run_hash.sh LOG HASH run4-args...`: an engine run checked for its tokens hash (`SETENV` for a knob on
  every rank).
- `serve_turns.sh LABEL DIR [n...]`: a test server, `turns.py` through it (a cold prompt of n tokens, then
  +200, +2000, +8000, another session's turn and +2000), optionally `serve_test.py --ced`, and `tjoin.py`'s
  table of the turns with rank 0's turn lines. `turns_cmp.py BASE NEW` compares two tables by the rule a
  default change must pass (no turn 2 % slower, the geomean 2 % faster).
- `moe_sweep.sh DIR KBUILD`: the prefill MoE and mHC boundary knobs' bit tests and sweeps, then `pick.py`'s
  `DSGB10_MOE_TUNE` line (`--later`: also the candidates kept for a later sweep). That line names the knobs
  to set; the engine at master has no runtime reader for it, since the knobs stay off by default, and
  `bench_kernels` takes them as `PF_DGRP`, `PF_DRING`, `PF_UPF` and `PF_TORD`.

```
J=/data/dsgb10/jobs/x; mkdir -p $J   # or any directory outside the repo
setsid nohup engine/attn/tools/ringparts.sh $J sweep="engine/attn/tools/moe_sweep.sh $J engine/build/kernels" \
  > $J/drive.log 2>&1 < /dev/null &   # PREEMPT=0: not preemptible; progress: $J/parts.log, $J/steps.log
```

## Accuracy against model.py

- Bit-exact: RoPE tables, q_norm, kv_write, compressor latent / KV entry / index key (fed the
  reference's projections), indexer q, weights, scores, top-k, pool, Reindex, the 4-rank split + merge
  (every row equal, also at T=32000 with the real topk 512 / 2048 blocks), Sinkhorn, hc_pre, hc_post,
  DSpark window and draft writes.
- The RMSNorm fused into hc_mix and hc_pre was exact on these cases. It is not exact everywhere: fed
  model.py's 5-row DSpark blocks (engine/dspark's test), 2–6e-6 of its outputs differ by one bf16 ulp.
  That is the sum of squares in another fp32 order; hc_pre's collapse itself is exact there.
- Summation order only: compressor projections (fp32, max abs 6e-8), weights_proj, hc projection
  (≤ 4e-6 relative).
- Attention (online softmax, not bit-exact): relative L2 error per head vs fp64 equals the reference's
  own, mean 1.7e-3, max ≤ 2.6e-3, prefill and decode, split-K, DSpark, long32k.
- long32k's caches after a 32000-row prefill differ from model.py in 0–1 KV entries and 6–10 index keys
  per cache, and 19–53 latents by 1 ulp: model.py's own results depend on its batch size there
  (tests/probe_batch_order.py: its index keys differ in 334–823 rows between one batch and slices).

**model.py bug.** `Indexer.forward` publishes `shared_attn.index_k` only when a group completed, so at
even decode positions the ratio-2 sources (2, 8, 14) score against the last publisher's cache — layer
20's from the previous step. The `stale_index_k` case reproduces it (layer 2, position 300: scores 0%
equal, wrong top-k). The kernels use each source's own keys; `gen_attn.py --own-index-k` does the same.

## Decode timing, per rank (GB10, 1 row, CUDA graphs, L2 flushed before each replay)

µs per layer (attention path without mHC; mHC = both sublayers):

| context | window 0–1 | r2 src+Full 2,8,14 | r2 reuse ×15 | r1 src+Full+pool 20 | r1 reuse ×15 | Reindex ×4 | mHC ×40 | step |
|---|---|---|---|---|---|---|---|---|
| 1k | 25 | 238 | 53 | 184 | 49 | 78 | 88 | 6.31 ms |
| 16k | 27 | 218 | 49 | 188 | 49 | 86 | 88 | 6.24 ms |
| 128k | 27 | 227 | 51 | 216 | 50 | 88 | 89 | 6.35 ms |
| 1M | 26 | 299 | 49 | 374 | 50 | 86 | 93 | 6.87 ms |

"step" sums 80 separate graphs; one graph of the whole step measured 5.3–6.0 ms when the GPU was
otherwise idle. Components at 1M: layer 20 score 146, select 88, merge 29, compressor 88, attention 43;
layers 2/8/14 score 88, select 41, compressor 136. The step is ~610 small dependent kernels (launch
floor 0.9 µs each, ~3–9 µs each in practice): mHC alone is 10 kernels per layer, 3.5 ms. The fused
decode path below replaces them.

Prefill chunk of 2048 rows ending at 16k: 455 ms for the 40 layers per rank: attention 5.9 ms/layer
(one CTA per row re-dequantises its keys), mHC 4.8 ms/layer (CUDA-core fp32), compressor 8.9 / 4.6 ms
(GEMV re-reads the weights per 8 rows), indexer 0.35–0.8 ms per index layer. Prefill needs a multi-row
attention kernel and tensor-core GEMMs for hc_fn (fp32 weights as bf16 parts) and the compressor.

## Fused decode path (PDL chain)

Decode and verify steps (rows ≤ 16) use fused launches, each bit-identical to the separate kernels it
replaces: tests/fused_tests.cu checks every output against them, and the separate kernels themselves
match the pre-fusion ones (dfaeab9) byte for byte. Every launch is a programmatic dependent launch
(launch.h): it lets the next launch start, loads its constant weights, then waits
(`griddepcontrol.wait`) before reading its predecessor's outputs, so with engine/kernels' GEMVs a step
is one graph of PDL launches. One layer on one rank, in stream order (G: an engine/kernels GEMV;
C: a collective, the layer runner's):

| launch | reads | writes |
|---|---|---|
| C all-reduce + `hc_post` (runner) | sublayer output, stream, post, comb | stream x [rows][20480] bf16 |
| `hc_project(x, rows, fn, k0, k1, chunks)` | x, hc_fn [24][20480] f32, columns [k0, k1) | chunks [(k1-k0)/512][rows][25] f32 |
| C (split projection only) | each rank's 10 chunks | all 40 in rank order (all-gather; same bits as one rank) or their sum |
| `hc_mix(x, chunks, nchunks, rows, scale, base, norm_w, pre, post, comb, y, q, qs, pdl, s, pre_in)` | x, chunks, hc_scale [3], hc_base [24], the sublayer's norm, pre_in: the previous sublayer's pre mix | pre, post [rows][4], comb [rows][16] f32 (this sublayer's); y [rows][5120] bf16 collapsed with pre_in; q e4m3 + qs ue8m0/32 |
| G wq_a + wkv | q, qs | this rank's qa, kv_pre columns |
| `compressor_project_rows(y, cw, n0, n, kv, score)` (layers 2, 8, 14, 20) | y, rows [n0, n0+n) of wkv (+ wgate) | kv, score [rows][n] f32 |
| C all-gather | qa, kv_pre, compressor slices | qa [rows][1280], kv_pre [rows][512], kv/score [rows][512] |
| `decode_prep(DecodePrep)` | the gathered three, q_norm/kv_norm, pos, table; cw (norm, wk, knorm), carry; y + weights_proj (index layers) | qq/qs (wq_b's input), the ring slot, the compressed entry and index key (+ carry), wp [rows][32] |
| G wq_b (+ the indexer's wq_b as a second segment) | qq, qs | q_pre [rows][16][512], this rank's iq heads |
| C all-gather (index layers) | iq | iq_pre [rows][32][128] |
| `index_fused(iq_pre, wp, …, scan, top, pool, counters)` (2, 8, 14, 20, 24–36) | iq_pre, wp, index keys; Full: this rank's quarter, Reindex: layer 20's pool blocks | top-512 positions + keys (local on Full layers, final on Reindex); layer 20's 2048 pool blocks |
| C all-gather (2, 8, 14, 20) | candidates (+ pool) | 4 ranks' candidates |
| `index_merge2(rows, cand…, top, pcand…, pool)` | gathered candidates | top-512 (+ pool), the same on every rank |
| `sparse_attn(…, scratch, s, true)`: attention + split-K combine | q_pre, ring, compressed KV and top-k of the layer's sources, sink | o [rows][16][512] bf16, inverse RoPE applied |
| G wo_a + quant, wo_b; C all-reduce + `hc_post`; `hc_project`, `hc_mix` (FFN norm) | | |
| G gate, top-6, moe_up, moe_down; C all-reduce | | |

**model.py's single-pass mHC.** `Block.forward` collapses the stream for the attention with the
*previous* sublayer's pre mix (the previous layer's FFN's; the identity before layer 0) and for the FFN
with this layer's attention's. The mixes a sublayer computes go to its `hc_post` and to the next
collapse. Against that:

- `hc_mix` without `pre_in` collapses with the pre mix it has just computed. That is not model.py.
  With `pre_in` (since 2a47c3b) it is, bit for bit, `hc_pre(x, pre_in)` + norm + act_quant, and `pre`
  receives this sublayer's own pre mix (`pre_in` may be `pre`: DSpark keeps one buffer). fused_tests
  checks both forms.
- Everything else in the chain is unaffected by the ordering and matches the separate kernels as it is:
  - `hc_project` (also on a rank's own [24][5120] slice of hc_fn: `fn_ld`);
  - `hc_post`, `hc_pre` (both with `pdl`);
  - `compressor_project_rows`, `decode_prep`, `index_fused`, `index_merge2`, `sparse_attn`;
  - `kv_write_sets` (DSpark's window writes).
- The head: model.py has no hc_head projection. The final collapse is `hc_pre` with the last FFN's pre
  mix, then the final norm. bench_step's head mixes a stand-in mHC there instead, one `hc_project` +
  `hc_mix` more than the model needs.
- engine/dspark runs the DSpark blocks this way (`hc_mix` with `pre_in`). Its test reproduces model.py's
  drafts and acceptance.

`index_fused`'s counters ([rows] ints) start at zero and are left zero. `hc_post` takes `pdl` for the
runner's chain. Launches per layer: the attention side 3 (`decode_prep`, attention, combine), 4 on
Reindex layers, 6 on layers 2, 8, 14, 20; mHC 2 per sublayer plus the runner's 2 `hc_post`; the GEMVs 8.
Per step 699 launches without the collectives, 378 of them attention side and mHC (734 as separate
kernels).

Step times, one rank (tests/bench_step.cu: 40 layers and the head in one graph, synthetic weights and
state, collectives replaced by local STAND-IN kernels at each point above; variants interleaved, median of
21 replays, µs, rank 0 shared):

| rows | context | GEMV chain alone | step | step without stand-ins | attention side + mHC alone | the same, separate kernels |
|---|---|---|---|---|---|---|
| 1 | 1k | 13852 | 15283 | 15033 | 1593 | 4745 |
| 1 | 16k | 13901 | 15287 | 15090 | 1640 | 4768 |
| 1 | 128k | 13899 | 15418 | 15122 | 1674 | 4792 |
| 6 | 1k | 31377 | 34783 | 33853 | 2394 | 5215 |
| 6 | 16k | 31398 | 34951 | 33912 | 2448 | 5250 |
| 6 | 128k | 31416 | 34939 | 34006 | 2503 | 5289 |

The replicated hc projection (no mix collectives) measured 16.6–16.7 ms at 1 row. Per launch at 1 row,
16k (nsys, time each adds to the chain): `hc_mix` 6.8 µs ×81, attention 10.6 ×40, `decode_prep` 4.7 ×40,
`index_fused` 23 ×8, `hc_project` 2.2 ×81, combine 3.4 ×40, `hc_post` 1.2 ×80, `index_merge2` 13 ×4,
`compressor_project_rows` 6.2 ×4. Left: Sinkhorn is serial (~2.5 µs of each `hc_mix`); the compressor
projection could become a bf16 segment of the wq_a + wkv launch (needs a bf16 segment in dsk's
multi-segment GEMV); the selection in `index_fused` / `index_merge2` is one CTA per row at the end; the
bench's head stands in for hc_head with an mHC mix.

## Cross-rank interfaces (for engine/comm)

- Decode, Full layers 2, 8, 14, 20: all-gather of each rank's local top-512 (i32 position + u16 key,
  3 KB per row), concatenated in rank order, into `index_merge`; layer 20 adds its local 2048 pool
  blocks (12 KB per row) in the same all-gather. Reindex layers (24–36) and DSpark: replicated, no comm.
- Prefill (rows split): all-gather of each rank's `[rows/4][512]` i32 top-k for the 4 Full and 4
  Reindex layers; the pool stays local.
- mHC: per sublayer the 25 floats per row of `hc_partial` (rank r: columns of copy r), all-reduced in
  fp32 in rank order so ranks stay bit-identical (80 per step). It cannot ride the sublayer's own
  all-reduce (the stream it projects comes out of that). Or all-gather the 10 `hc_project` chunks per
  rank (1 KB per row) and let `hc_mix` add all 40: the same bits as one rank's projection. Replicating
  the projection instead measured +1.1–1.4 ms per step at 1 row in bench_step (against stand-ins).
- With the compressor split below: each rank's projection slice (ratio 2: 128 kv + 128 score fp32,
  1 KB per row; ratio 1: 512 B) joins decision 1's wq_a / wkv all-gather of the same layer.
