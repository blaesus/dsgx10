# Serving altar-1 (milestones A6, A7)

Rank 0 of the four-rank engine as a server behind the front-end `altarfront`, the store, the service and the
drafts, as a delta from GLM-5.3's ([glm53 serving.md](../../glm53-4xgx10-switchless/docs/serving.md), "glm53"
below). Written 2026-10-14. Every number names its job; jobs are ring jobs through the DeepSeek engine's `ring.sh`
(`OWNER=altar1-serve`), results in `~/dsgb10-data/altar1-serve/results.tsv` and the job logs beside it (not in the
repo).

## Contents

1. [What is identical](#1-what-is-identical)
2. [What differs: names, ports, paths](#2-what-differs-names-ports-paths)
3. [Memory: the guard and the pool](#3-memory-the-guard-and-the-pool)
4. [The disk tier and kept sessions](#4-the-disk-tier-and-kept-sessions)
5. [Drafts](#5-drafts)
6. [Tests and results](#6-tests-and-results)
7. [The service](#7-the-service)
8. [Failures and what remains](#8-failures-and-what-remains)

## 1. What is identical

The protocol (version 2, [protocol.md](protocol.md)), sessions and exact prefix reuse (q = min(LCP, n − 1)), the
scheduling (B1 turns, B2 up to `--slots` sessions a step, the control word), batch invariance at the pinned split-K
(`--splits 16`), the paged pool (pages of 1,024 positions, rank 0 hands them out, compaction, eviction to disk,
`--grow`), the disk tier's file format, hashes and O_DIRECT staging, kept sessions across a planned stop (the run id
now hashes `altar1 kept sessions`), the drafts' verify loop `spec_loop` with chained draft passes, the gate, spec
graphs and the MTP entries carried in every page, and the service's supervisor, ring borrowing, yielding and
reclaiming: glm53 § 1–5 hold as written, with the values below. A position still costs **96,640 B** a rank
(`Model::cache_pos_bytes`, unchanged: altar-1's cache is GLM-5.3's), a page 94.4 MiB, a 256k session 23.59 GiB.

## 2. What differs: names, ports, paths

altar-1 coexists with glm53 on disk and on the boxes; only one engine is ever on the ring (the one ring lock).

| | glm53 | altar-1 |
|---|---|---|
| rank binary (`engine/server/`) | `glm53_rank` | **`altar1_rank`** |
| HELLO's model name, served name | `glm-5.3` | **`altar-1`** (altarfront's default `--model-name`) |
| engine socket default | `/tmp/dsgb10-glm53-engine.sock` | **`/tmp/dsgb10-altar1-engine.sock`** (altarfront's default) |
| front-end, its session table magic | `glmfront`, `GLMT0001` | **`altarfront`**, `ALTR0001` |
| service port | 18894 | **18895** (8888, the DeepSeek service's, and 18894 refused) |
| service settings | `GLM53_*` | **`ALTAR1_*`** (`ALTAR1_ADDR`, `_PORT`, `_KEY_FILE`, `_ENGINE_ARGS`, `_FRONT_ARGS`, `_NOTE`, `_BIN`, `_FRONT_BIN`, `_YIELD_IDLE_S`, `_HOLD_S`, `_KEEP_GIB`, `_HOME`) |
| fleet `SERVICE_DIR`, `BIN_DIR`, `LOG_DIR` | `~/glm53/…` | **`~/altar1/service`, `~/altar1/bin`, `~/altar1/logs`** |
| spill directories (`RANK<r>_SPILL`) | `/data/dsgb10/glm53/spill`, `~/glm53/spill` | **`/data/dsgb10/altar1/spill`** (rank 0), **`~/altar1/spill`** (ranks 1–3) |
| rank files (`RANK<r>_WEIGHTS`) | glm53's | `/data/dsgb10/altar1/weights/rank0.glw`, `~/altar1/weights/rank{1,2,3}.glw` |
| binaries and test data | `~/dsgb10-data/glm53-server` | **`~/dsgb10-data/altar1-serve`** (`bin/altar1_rank`, `bin/altarfront`, `bin/SOURCES` with the sources hash and the commit) |
| test sockets, ring-job owner | `/tmp/glm53-test.*`, `glm53-server` | `/tmp/altar1-test.*`, `altar1-serve` |
| ranks a preempted job's reap stops (`ringlock.sh`) | `dsgb10_rank`, `glm53_rank` | `dsgb10_rank`, `glm53_rank`, **`altar1_rank`** |

The DeepSeek engine's `ring.sh`, through which every job here runs, reaps only `dsgb10_rank` on preemption; the job
scripts (`ranks.sh`'s TERM trap, `run4.sh stop`) stop altar-1's ranks themselves, as glm53's did.

## 3. Memory: the guard and the pool

The rank files are 76.6 GiB a rank (82,411,745,280 B on disk), against glm53's 101.4; what is left is the cache's
(DESIGN decision 4: one 256k BF16 session in the pool when memory allows). What changed in `main.cpp`:

- **The guard** (new, glm_run's `--min-free-mib` in the server): after the weights and the prefill buffers every rank
  reports its free device memory and MemAvailable, rank 0 logs them (`memory after the weights …`), and **all ranks
  stop when one has less than `--min-free-gib` (8) plus one page**: the pool could not then be had without going under
  the fleet's 8 GiB.
- **Margins 8 GiB on every rank** (glm53: 4 on rank 0, 2 on the workers; the margin is at least `--min-free-gib`): the
  pool is min over ranks of (min(free device, MemAvailable) − 8 GiB), capped by `--kv-gib`. Rank 0 logs the free device
  memory and MemAvailable **after** the pool too, and whether a session of `--max-ctx` positions fits.
- **Every launcher with `oom_score_adj` 1000**: `ranks.sh` (run4.sh, the tests), `g8_job.sh`, and now the service's
  ranks (glm53's service left them at the supervisor's 0); the supervisor and the front-end keep 0.
- The service's default engine arguments are the memory rule's `--chunk 2048 --index-scratch-mib 128` while the CPU
  reference (rank 2) and the converter (rank 3) share the boxes (prefill 11–15 % slower than at 8,192 rows, A5);
  `ALTAR1_ENGINE_ARGS=""` gives the 8,192-row chunks. `MIN_AVAIL_GIB` (the service's check before a start) is 90 GiB.

**Measured on synthetic weights** (22.09 GiB a rank bound: the server's `--synthetic` with 16 INT4 pools; chunk 2,048,
scratch 128 MiB; free device / MemAvailable, GiB):

| Job (2026-10-14) | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---|---|---|---|
| 12:11 main, after weights and buffers | 56.05 / 88.32 | 89.73 / 92.26 | 70.58 / 88.31 | 54.97 / 88.43 |
| 12:11 main, after a 1.2 GiB pool | 54.69 / 87.13 | 88.54 / 91.06 | 69.42 / 87.16 | 53.69 / 87.31 |

**The pool with the real files, estimated**: the real files bind 54.5 GiB more than these synthetic weights, so
MemAvailable after the load would be ~33.8 / 37.8 / 33.8 / 33.9 GiB at these settings, a pool of ~25.8 GiB
(the smallest rank less 8): **273 pages, one 256k session (256 pages) fits**, with ~1.6 GiB to spare. Rank 2's
reference pass (7.1 GB RSS at 12:28) is already inside its figure; anything more there and the session no longer fits
(the pool then caps the context: HELLO's `max_context` is the pool's positions). The real figure is § 6's real-file
job.

## 4. The disk tier and kept sessions

Every rank holds the whole cache, so the tier is as large as the smallest rank's offer. Rank 0's data disk holds its
rank file with **33 GB free** (12:28, after the file landed): glm53's rule (free space less 32 GiB) gives it 0
pages, and so the tier and with it the kept sessions are **off by default**. New options make it explicit:
`--disk-rank0-gib N` (default 0: rank 0 offers at most N GiB; set it once rank 0's disk has room, e.g. 32) and
`--disk-margin-gib` (32: the free space every rank keeps). Ranks 1–3 have 421, 449 and 225 GB free. The tests run a
tier of 8 GiB with `--disk-rank0-gib 8 --disk-margin-gib 8` and remove rank 0's test files at the end of the job.

## 5. Drafts

The machinery is glm53's G7 (glm53 § 3.1; its engine.md, "Drafts"); altar-1's MTP layer has **BF16 experts**
(DESIGN decision 5: exact by construction, the NVFP4 drafter a later lever), bound by `bind.cu` like layers 3 and 77,
so a draft pass reads 0.77 GB a rank.

**The gate's cost model, refitted.** `tests/spec_cost.py` takes rank 0's log, the median step time of every
one-session run of ≥ 8 steps by server start, and fits c(k) = c1 + k · drow over the starts with k drafts forced
(`--spec-drafts k --spec-force k`: every step verifies k drafts with k passes). Synthetic weights in the real formats,
split-K 16, the spec jobs of 12:16 and 12:25 (server sources 2d4c15173bd3e49c, commit c418d3c):

| Server | one-session runs | step ms (median of medians; range) | tokens a step | tok/s |
|---|---:|---:|---:|---:|
| plain (`--no-spec`) | 28 | 47.45 (44.77–48.97) | 0.98 | 20.6 |
| gate, k ≤ 3 (all MTP drafts rejected: synthetic text) | 28 | 51.92 (49.72–62.90) | 1.00 | 19.3 |
| k 1 forced (+ oracle drafts on the repeat pass) | 26 | 65.38 (60.32–68.13) | 1.29 | 19.7 |
| k 2 forced | 23 | 78.60 (76.29–81.51) | 1.40 | 17.8 |
| k 3 forced | 23 | 91.47 (85.80–95.37) | 1.46 | 15.9 |

Fit: **c(k) = 52.40 + 13.04 k ms** (residuals ≤ 0.12 ms); the default `--spec-cost` is now **52.4,13.0** (glm53:
44.5,15.5, GPU time of glm_run's verify steps). These are server wall times a step, host synchronisation of the
one-step-at-a-time verify loop included (the plain loop queues a step ahead: 47.45 ms against glm_run's 46.27 GPU
ms, A4); the k = 0 intercept (52.4) matches the gate's measured k = 0 steps (51.92). A draft pays when accepted
more often than drow / c ≈ 13.0 / 52.4 = 25 % (glm53: 35 %). With every MTP draft rejected the gate costs **9.4 %**
against plain (51.92 / 47.45; glm53 9–13 %). On real files the routed rows read real experts, so the fit is
repeated there (§ 8).

## 6. Tests and results

`engine/server/tests/ring_job.sh` (glm53's, re-specialised): `WEIGHTS` default `--synthetic` (every weight synthetic
in the real formats; `""` for the fleet's rank files), a pool of 1.2 GiB (13 pages) so the eviction test evicts, a
disk tier of 8 GiB (tag `test`, rank 0 included), `--slots 4 --slice 16 --chunk 2048 --index-scratch-mib 128`; the
spec phase now runs `--spec-drafts k --spec-force k --spec-oracle` for every k of `SPEC_KS` (1 2 3) and the gate.
Two jobs keep each under ten minutes:

```
OWNER=altar1-serve ds41f-4xgx10-switchless/engine/server/ring.sh altar1-4xgx10-switchless/engine/server/tests/ring_job.sh   # PHASES="main batch front kept"
PHASES=spec SPEC_KS="1 2" OWNER=altar1-serve ds41f-4xgx10-switchless/engine/server/ring.sh altar1-4xgx10-switchless/engine/server/tests/ring_job.sh
```

**Synthetic weights** (binary sources 2d4c15173bd3e49c, commit c418d3c; altarfront from commit 7a8aab4):

| Test | Job (2026-10-14) | Checks | Result |
|---|---|---|---|
| `serve_test.py`: HELLO (altar-1, vocab 154,856), cold 2,600-token prompt, resume at 2,599 same tokens, fork at 1,500 = cold, a turn's suffix (2,632 cached), eviction to disk and back at the full prefix (2,691), seeded repeats, logprobs, cancel, ids ≥ 154,856 refused | 12:11:22 | 16 | **passed** |
| `batch_test.py`: 2, 3, 4, 6 sessions (B1 turns at 6) = alone; a mid-decode arrival | 12:11:55 | 5 | **passed** |
| `front_test.py`: altarfront on 127.0.0.1:18895: same text twice (632 of 633 cached), streamed, a second turn (657 cached), a client leaving mid-stream | 12:12:52 | 6 | **passed** |
| `serve_test.py --kept`: planned stop (4 sessions, 0.55 GiB, 0.51 s), start, the turn at its full prefix (2,691 of 2,692) from disk, same tokens | 12:13:07 | 3 | **passed** |
| `spec_test.py` record (`--no-spec`, 26 requests: 1 session greedy and seeded, stops inside the text and turns after them, 2/3/4 sessions at once, a mid-decode arrival, a turn after eviction, turns past/below 2,048 positions, a fork, a 160-token request) | 12:16:30, 12:25:28 | 1 + 1 | **passed** |
| `spec_test.py` k 1 forced, two passes (MTP drafts, then oracle drafts): tokens, logprob bits and ends = `--no-spec` | 12:17:47 | 52 | **passed** |
| k 2 forced | 12:20:15 | 52 | **passed** |
| k 3 forced (oracle pass: 912 of 1,394 drafts accepted in 600 steps) | 12:26:46 | 52 | **passed** |
| the gate (k ≤ 3), against each job's record | 12:22:50, 12:29:28 | 26 + 26 | **passed** |

263 checks, 0 failures, in three ring jobs (12:11–12:13:30; 12:16–12:24; 12:25–12:30:50). Throughput on these
synthetic weights is not a result (the oracle's acceptance is made up; the router routes noise).

**Real rank files**: pending (§ 8).

## 7. The service

`engine/server/service.sh` is glm53's (§ 5.2 there) with § 2's names and § 3's memory: it borrows the ring through
the queue of wants, the DeepSeek service lends it once idle for its `DSGB10_YIELD_IDLE_S`, it gives the ring back on
a reclaim or after `ALTAR1_YIELD_IDLE_S` (600 s) without a request, altarfront holds requests meanwhile
(`ALTAR1_HOLD_S`, 900 s). Defaults: `ALTAR1_ADDR` 127.0.0.1, `ALTAR1_PORT` 18895, `ALTAR1_ENGINE_ARGS`
`--chunk 2048 --index-scratch-mib 128`, `ALTAR1_KEEP_GIB` 8 (kept sessions need the disk tier: § 4), `MIN_AVAIL_GIB`
90, binaries from `~/dsgb10-data/altar1-serve/bin`.

```
engine/server/install.sh                                     # altar1_rank (-j4) into ~/dsgb10-data/altar1-serve/bin
(cd frontend && cargo build --release -j4) && cp …/altarfront ~/dsgb10-data/altar1-serve/bin/
engine/server/service.sh start | stop [--forget] | status | restart | forget
```

**First start on the ring** (glm53's G6b protocol): pending the real files (§ 8).

## 8. Failures and what remains

**Failures.** The first install from a clean tree failed (12:10): `install.sh`'s new commit line ended in a `grep -q`
that fails on a clean tree under `set -e`; fixed in c418d3c (`|| true`) and rebuilt from that commit.

**What remains** (all wait for the real rank files and the converter's loader on `altar1`: the runner's loader is
glm53's and rejects the layouts 4 and 5):
- the four test phases and the spec phase on the real files (`WEIGHTS=""`), the post-load MemAvailable and the real
  pool (one 256k session or not);
- the drafts on real text through altarfront (`tests/g8_job.sh` plain, k1, k2, k3, gate; `g8_front.py compare`):
  acceptance and tok/s against plain decode, and the cost fit repeated on real routing;
- the service's first start beside the DeepSeek service (start, a request, a reclaim by a DeepSeek request, an
  altar-1 request while DeepSeek holds the ring, stop);
- rank 0's disk tier (the user's to free space), the NVFP4 drafter (decision 5's lever).
