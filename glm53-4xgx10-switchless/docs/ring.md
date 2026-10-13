# Ring collectives (engine/comm)

The four-box ring of this engine: the DeepSeek engine's ibverbs ring with CPU proxies, copied as this engine's
own, plus GLM-5.3's collectives at their message sizes. Everything here was measured on the ring itself, ranks
0–3, on 2026-10-13 (22:49–22:58), borrowed from the DeepSeek service through its `ring.sh`.

**Status.** Ported and measured. `ring_test` (the DeepSeek engine's tests, with GLM's sizes) and `coll_tests`
(GLM's collectives, 191 cases) pass on the four ranks: every result exact against a host reference, the same
bits on every rank and run to run; 300,000 random-size collectives with every payload word checked, 0
mismatches; the write-after-read race test exact with a block stalled 2 ms, for the copied pipelined kernels
and the new row all-gather. A decode step's collectives, back to back in one graph with nothing between them,
take **4.40 ms at 1 row** (256 collectives, 17.2 µs each) against DESIGN's budget of 2.9 ms; the gap is the f32
all-reduces (below).

## What is the same as the DeepSeek ring

All of it except what the next section lists: the cabling and fleet file, the protocol (host-NUMA region, three
buffer sets, doorbells, the proxy's relays, rank-order sums, flow control), the API halves in `ring_dev.cuh`,
the standalone and bulk collectives in `collectives.cu`, the hazards and the write-after-read fix
(`comm::wait_posted`). See the DeepSeek engine's
[ring.md](../../ds41f-4xgx10-switchless/docs/ring.md): [Protocol](../../ds41f-4xgx10-switchless/docs/ring.md#protocol),
[API](../../ds41f-4xgx10-switchless/docs/ring.md#api),
[the race](../../ds41f-4xgx10-switchless/docs/ring.md#the-pipelined-reduce-scatters-send-buffer-race-2026-10-13),
[Hazards](../../ds41f-4xgx10-switchless/docs/ring.md#hazards). Its measured latencies reproduce here within
~0.5 µs (64 B all-reduce 7.89 µs, 10 KB 12.56, 60 KB 24.84; the full table is in `ring_test latency`'s output).

Differences, all of identity:
- library `glm53_comm` (tests: `glm53_comm_test`, hooks `GLM_COMM_TEST_HOOKS`), namespace still `comm`;
- the bootstrap port is 18660 (DeepSeek's 18620), and the hello carries a magic (`GLMR`, version 1): a stray
  rank of either engine can never bootstrap with the other's;
- the fleet file: this configuration has none of its own yet and reads the DeepSeek engine's
  (`ds41f-4xgx10-switchless/fleet/homelab.conf`, the same boxes); the engine's CMake copies it beside the
  binaries as `fleet.conf`, `run4.sh` installs it beside its copies. Its DeepSeek-only keys (weights, Engram)
  are read and unused.

## GLM's collectives (`comm/glmcoll.h`)

All take 1–16 rows at decode (any row count for the all-gathers), sizes from `attn/attn.h` as compile-time
constants, are CUDA-graph capturable, and take `pdl` (wait for the predecessor, let dependents launch once
posted).

| Collective | Per rank a row | What it serves |
|---|---:|---|
| `allreduce_hidden` | 24,576 B (f32 [6144]) | o_proj and the MoE/MLP down partials: sums in rank order, f32 and/or bf16 out; the same bits as `comm::allreduce_f32` |
| `allreduce_add_norm` | 24,576 B | the same with the residual add and the next RMSNorm fused into its end: `h = bf16(res + bf16(sum))`, `xn = rmsnorm(h)`, **bit for bit attn's `rmsnorm` kernel** (its sum-of-squares order rebuilt across blocks) |
| `allgather_xproj` | 1,408 B (full layer), 1,312 (shared) | in place into attn's `xg [R][4][688]` (q_a 512 + kv latent 128 + rope 16 + wk 32 bf16) plus the indexer head weights `[R][4][8]` f32: one collective |
| `allgather_idxq` | 2,048 B | the indexer's queries (wq_b output, 8 heads a rank) in place into `[R][32][128]`, full layers |
| `allgather_cand` | 16,384 B | the position-split indexer's candidates (2048 positions + 2048 order keys) → `[R][4][2048]` each, `index_merge`'s input |
| `allgather_logits_f32/_bf16` | 154,880 / 77,440 B | the head's vocabulary-split logits → `[R][154,880]` |
| `topk_exchange` | 8 B × K | each rank's own top-K (value, index), merged: the global top-K, best first, ties to the lower index (K = 1: argmax; up to 64) |
| `allgather_rows` | any | the general row-interleaved all-gather under all of the above: rank k's row r at `out + r·ld + k·rb`, segments, in place, posts of up to `chunk` pipelined like `allgather_multi` (copies two ahead; each send-buffer unit has one thread, so no `wait_posted`; the race test covers it) |

The fused all-reduce's grid: S column slices (8, 16 or 32; the same bits; default 8) × row groups, at most 48
blocks. Each thread reduces 8 columns; the owners of rmsnorm's 24-column runs sum squares from shared memory
into a device array, the S blocks of a row group meet at a counter barrier, and each row's first warp redoes
rmsnorm's eight xor-trees and warp-order sum. At most one fused all-reduce runs at a time (stream order).

**Tests** (`coll_tests test`, four ranks): `allreduce_hidden` and `allreduce_add_norm` at 1, 2, 3, 4, 5, 6, 7,
8, 12, 13, 16 rows × S 8/16/32 × with/without PDL, against a host reference (exact f32, no FMA contraction) and
against `comm::allreduce_f32` + residual add + attn's `rmsnorm`, in place twice for run-to-run identity; the
all-gathers at 1–16 rows, 300 and 2048 rows (pipelined), 64 KiB chunks, in place and not; `topk_exchange` with
deliberate value ties across ranks, K 1/2/7/64; eight layers' collectives in one PDL graph against the same one
by one. All exact and identical on all ranks.

## Measurements

µs per collective, the median of 3 trials of 20 replays of a graph of 50 back-to-back collectives (logits:
10 × 10), cap 4 MiB, proxies on core 5, PDL on. Sources sha256 `816f43a6367e8351` (S given explicitly).

| Rows | AR f32 generic | AR bf16 generic | `allreduce_hidden` S8 | `allreduce_add_norm` S8 | xproj full | xproj shared | idxq | cand | logits f32 | logits bf16 | argmax | top-k K1 | top-k K64 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.66 | 13.35 | 18.52 | **21.50** | **10.13** | 10.02 | 10.50 | 15.11 | 46.3 | 29.0 | 8.43² | 8.11 | 10.62 |
| 2 | 22.86 | 17.09 | 24.54 | 27.17 | 11.10 | 10.78 | 11.34 | 20.59 | 75.8 | 45.4 | 8.42 | 9.45 | 12.92 |
| 4 | 32.56 | 22.83 | 34.61 | 37.14 | 11.89 | 11.68 | 12.69 | 27.02 | 129.3 | 75.9 | 8.35 | 10.03 | 13.23 |
| 6 | 42.20 | 27.42 | 46.11 | 48.50 | 12.78 | 12.65 | 13.70 | 33.04 | 185.9 | 104.0 | 8.37 | 10.64 | 13.69 |
| 8 | 52.02 | 31.90 | 57.76 | 59.07 | 13.64 | 13.27 | 15.63 | 39.64 | 238.4 | 128.7 | 8.39 | 11.52 | 14.10 |
| 12 | 70.76 | 42.23 | 78.44 | 79.45 | 15.93 | 14.76 | 17.72 | 53.36 | 352.5 | 185.7 | 8.79 | 12.12 | 14.64 |
| 16 | 88.16 | 52.09 | 98.21 | 98.57 | 17.34 | 16.85 | 20.21 | 65.11 | 471.1 | 238.0 | 8.88 | 12.73 | 15.47 |

(² `comm::argmax_rows` needs an even row count: 2 rows.) S16 and S32 are within ±1 µs of S8 everywhere
(slightly worse at 1–2 rows). The logits all-gather in 256 KiB posts is slower than one post (619.6 µs at 16
rows f32): at cap 4 MiB a single post is best. The x-projection all-gather (10.1 µs) is the floor of an
all-exchange (DeepSeek's 64 B: 7.9 µs, plus ~2 µs for its row-strided copies at 1.4 KB).

Readings:
- **The f32 all-reduce is the decode cost.** 24 KB at one row costs 17.7 µs generic, 21.5 fused, against 13.4
  for bf16 partials. bf16 partials would change outputs (each rank's partial rounded before the sum), which
  the exactness rule excludes; they stay a measured number only.
- **The fused epilogue costs ~3 µs on the collective** (21.5 against 18.5 for the same kernel without it: the
  barrier and the second pass) and saves the residual-add + RMSNorm launch it replaces (not measured
  separately; attn's rmsnorm is one 256-thread CTA a row). Past 8 rows the fusion is free (98.6 against 98.2).
- My `allreduce_hidden` reads host memory with half the threads of `comm::allreduce_f32`'s grid and is 0.9 µs
  slower at 1 row, 10 µs at 16 (98.2 against 88.2).
- **The head:** exchange top-K, never logits: K = 64 costs 10.6–15.5 µs against 46–471 for f32 logits.
- **The indexer merge** (position split): 15.1 µs at 1 row, 65 at 16, per full layer.

### A decode step's collectives (`coll_tests step`)

Per layer the x-projection all-gather and two `allreduce_add_norm`s; on the 21 full-indexer layers also
`allgather_idxq` (and with the position-split indexer `allgather_cand`); then `topk_exchange` K = 64. One graph,
PDL, nothing between the collectives:

| Rows | replicated indexer: 256 collectives | position-split: 277 |
|---:|---:|---:|
| 1 | **4.40 ms** (17.2 µs each) | 4.69 ms |
| 2 | 5.31 | 5.77 |
| 4 | 6.98 | 7.61 |
| 6 | 8.93 | 9.57 |
| 8 | 10.67 | 11.50 |
| 16 | 17.58 | 19.00 |

### Prefill chunks (`coll_tests prefill`, 16 blocks)

| Rows | reduce-scatter by rows, f32 partials | all-gather of the reduced rows, bf16 | together | bulk all-reduce f32 | xproj all-gather |
|---:|---:|---:|---:|---:|---:|
| 256 | 369 µs (17.1 GB/s of input) | 167 | 536 | 671 | 88 |
| 1024 | 1,417 (17.8) | 637 | 2,054 | 2,738 | 276 |
| 2048 | 2,756 (18.3) | 1,174 | **3,930** | 5,497 | 557 |

So a 2048-row chunk's two all-reduces a layer, as reduce-scatter + residual/norm on the rank's quarter +
all-gather of h (bf16), cost ~7.9 ms a layer, ~610 ms over 78 layers: ~3.3k tokens/s if nothing overlaps
(the published bar is 1,374 tok/s cold prefill at 8k). The bulk all-reduce costs 1.4× that.

## The per-step collective budget

| | Collectives a decode step | Measured at 1 row |
|---|---:|---:|
| 78 layers × 3 (x-projection all-gather, o_proj all-reduce, MLP/MoE down all-reduce) | 234 | 78 × (10.1 + 2 × 21.5) = 4.14 ms |
| 21 full-indexer layers: the indexer queries' all-gather | +21 | 0.22 ms |
| the head: top-64 exchange | +1 | 0.01 ms |
| **Replicated indexer** | **256** | **4.40 ms** in one graph |
| position-split indexer: candidates' all-gather on full layers | +21 | +0.29 ms → 4.69 ms |
| MTP draft pass (a full layer: xproj, idxq, 2 all-reduces; its top-k) | +5 a pass | ~0.07 ms a pass |
| the router (`mlp.gate` replicated, as DeepSeek's), the embedding (replicated) | 0 | |

DESIGN's budget (3 × 78 at ~12 µs = 2.9 ms) assumed DeepSeek's 10 KB bf16 all-reduce; the f32 hidden all-reduce
is 24 KB and 17.7–21.5 µs. Replicating `wq_b` instead of gathering its output would cost 6.3 MB a full layer
(~0.56 ms a token) against 0.22 ms of all-gathers: the split stays.

## Running it

GPU and ring code only through the DeepSeek engine's `ring.sh`. `engine/comm/install.sh` builds the tests on
the CPU (-j4, an out-of-tree wrapper with `comm/` and `attn/`) into `~/dsgb10-data/glm53-comm/bin` with the
sources' hash; `engine/comm/run.sh` runs the steps not yet marked done for that hash (resumable after a
preemption, exit 76), each through `engine/comm/run4.sh`, which copies the binary and fleet file to
`~/dsgb10-data/glm53-comm/run/` on every box, starts ranks 1–3 detached and rank 0 in the foreground, and on
TERM stops the workers' ranks (ring.sh only reaps the DeepSeek engine's):

```
OWNER=glm53-comm ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/comm/run.sh [ring|test|bench]
```

The job takes ~4 minutes once the ring is lent. Results: `~/dsgb10-data/glm53-comm/results.tsv` and `.log`.
Two failed first attempts (2026-10-13 22:41, 22:44): the workers' `pkill -f` matched its own remote shell,
whose command line named the binary's path, and killed it before the launch; `run4.sh` now pkills in a call of
its own with a `[d]` pattern.

## What remains

- **Halve the all-reduce's cost or its count.** Options, all exact: (1) a GPU-side reduce-scatter + all-gather
  at decode sizes (0.875× instead of 1.5× the payload a link, but two hops of latency: DeepSeek's 24 KB
  reduce-scatter is 12.5 µs, so only past ~4 rows); (2) overlap: post the o_proj partials from the GEMV epilogue
  (the `ring_dev.cuh` producer half) so the collective's network time hides behind the shared-expert / router
  work, as DeepSeek's fused kernels do; (3) merge the MLP down all-reduce with the next layer's x-projection
  all-gather (the x-projection needs the normed h, so this needs the x-projection replicated or a
  reduce-scatter + local norm + all-gather of x-projection outputs: three-way design, G4).
- `allreduce_hidden` reads host memory with too few threads (10 µs slower than the generic kernel at 16 rows):
  two 16-byte items a thread, or more blocks per row group.
- The fused all-reduce's ~3 µs barrier: compute the row statistic redundantly per block from all columns of
  the row (more host reads) or with a single-block-per-row variant at 1 row; measure against the separate
  rmsnorm launch inside a real layer graph.
- The prefill path's residual + norm on the reduce-scattered quarter, and the all-gather of h, as one fused
  kernel (`reduce_scatter_rows` already lands the quarter in place).
- The position-split cache (two more collectives a layer, DESIGN decision 5) is not built.
- This configuration's own fleet file (and dropping the DeepSeek-only keys) when it gets its own service.
