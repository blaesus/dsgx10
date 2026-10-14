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
  pool is min over ranks of min(free device − `--device-margin-gib` (1), MemAvailable − 8 − 1 GiB), capped by
  `--kv-gib`: the second 1 GiB is for what CUDA allocates after the pool (graphs, spec buffers, ~0.7 GiB; without it
  the service's start at 16:55 left rank 2 at 7.36 GiB, 9d10f8a).
  Free device memory binds, not MemAvailable: see "the page cache" below.
- **The rank file's page cache is dropped before the load** (`posix_fadvise(DONTNEED)`; the loader reads with
  O_DIRECT, so a copy or an earlier `cat` left up to 24 GiB of it cached on ranks 1 and 2), and **before the pool every
  rank touches anonymous memory** up to what the pool wants, staying above the 8 GiB margin, and lets it go
  (`--no-reclaim`: not), so the kernel reclaims other files' page cache that CUDA will not take. Rank 0 logs the free device
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

**With the real files** (main job 15:40, commit af3d5d4; free device / MemAvailable after the weights and the
buffers, GiB):

| Job (2026-10-14) | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---|---|---|---|
| 15:40 main, after weights and buffers | 2.02 / 33.44 | 34.02 / 36.97 | 23.03 / 33.21 | 35.21 / 36.82 |

MemAvailable is as estimated from the synthetic run (33–37 GiB; less 8: a 25 GiB pool, a 256k session's 23.59 GiB
fits), but **free device memory on rank 0 is 1.2–7.4 GiB** across the day's jobs (1.24 at 14:35, 7.38 at 16:34) and
rank 2's 16–26: CUDA on GB10 allocates free pages only and does not reclaim the page cache, and rank 0's cache holds
the DeepSeek service's Engram shards (two files of 23.60 GiB, read at each of its starts), rank 2's the CPU
reference's files. **The page cache, measured three ways:**

1. **A pool sized from MemAvailable** (d3bc4e5, 13:33): rank 0 offered 25.7 GiB with 6.3 GiB free; the pool's
   allocation failed in the driver (`NV_ERR_NO_MEMORY`), rank 0 hung, and the kernel's OOM killer took `altar1_rank`
   at 13:38:39 (its `oom_score_adj` 1000), at the moment the whole session on rank 0 was interrupted (§ 8). Reverted:
   the pool takes the smaller of free device and MemAvailable (694c448).
2. **Touching only the shortfall** (14ebc35: anonymous memory for want − free): 0.50 GiB at 15:24 and 0.25 at 15:40
   reclaimed nothing; free device fell by what was touched (2.54 → 2.04, 2.00 → 1.74 GiB): the kernel hands out its
   free pages first and reclaims the cache only below its watermarks.
3. **Touching all the pool wants** (742c581): 9.00 GiB at 16:42 (g8 job, `--kv-gib 8`) took rank 0 from 3.62 to
   10.39 GiB free; the pool got its 86 pages (7.93 GiB), the tokens the same as plain decode's (`g8_front.py
   compare`). Uncapped (§ 7 and the 16:57 job) it touched 26.00–27.00 GiB on rank 0 and took it from 9–15 to
27.3–27.7 GiB free; **the pool is then bound by rank 2's MemAvailable** (30.35–32.55 GiB while the CPU reference
runs there): **231 pages (236,544 positions, 21.29 GiB) at 9d10f8a, 16:57**, every rank at ≥ 8.55 GiB of MemAvailable
after it (6.57 / 13.14, 12.42 / 15.41, 5.50 / 8.55, 14.90 / 15.31 GiB free device / MemAvailable). **One 256k session
(256 pages) fits only when every rank has ≥ 32.6 GiB available before the pool**, i.e. without the reference's ~3 GiB; until then HELLO's
`max_context` is the pool's positions.

The tests keep their own `--kv-gib` caps (1.2 GiB in `ring_job.sh`, 2 in `g8_job.sh`); with rank 0's free device
low they got less (8 pages at 15:40, 4 at 15:45, 11 at 15:24), which § 8's k = 2 failures came from.

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

Fit: **c(k) = 52.40 + 13.04 k ms** (residuals ≤ 0.12 ms), the default until the real fit below (glm53:
44.5,15.5, GPU time of glm_run's verify steps). These are server wall times a step, host synchronisation of the
one-step-at-a-time verify loop included (the plain loop queues a step ahead: 47.45 ms against glm_run's 46.27 GPU
ms, A4); the k = 0 intercept (52.4) matches the gate's measured k = 0 steps (51.92). A draft pays when accepted
more often than drow / c ≈ 13.0 / 52.4 = 25 % (glm53: 35 %). With every MTP draft rejected the gate costs **9.4 %**
against plain (51.92 / 47.45; glm53 9–13 %). On real files the routed rows read real experts, so the fit is
repeated there.

**On the real files** (server commit af3d5d4, sources fe90f03ad5242f43: decision 8's combine and residual rounding;
the spec jobs of 15:45, 15:55 and 16:02):

| Server | one-session runs | step ms (median of medians; range) | tokens a step | tok/s |
|---|---:|---:|---:|---:|
| plain (`--no-spec`) | 36 | 46.13 (44.88–47.38) | 0.98 | 21.2 |
| gate, k ≤ 3 (cost 52.4,13.0) | 12 | 63.30 (52.09–67.28) | 1.56 | 24.6 |
| k 1 forced | 24 | 63.30 (60.02–67.69) | 1.68 | 26.5 |
| k 2 forced | 46 | 78.62 (72.11–81.92) | 2.02 | 25.7 |
| k 3 forced | 24 | 91.77 (79.94–95.84) | 2.21 | 24.0 |

Fit: **c(k) = 49.43 + 14.23 k ms** (residuals ≤ 0.72 ms); **the default `--spec-cost` is now 49.4,14.2** (commit
42691b8). These runs are `spec_test.py`'s short requests (40 tokens), so the tokens a step are not real text's.

**Real text through altarfront** (`tests/g8_job.sh` CFG, then `g8_front.py compare plain CFG`; four prompts of
glm53's G8 — `short` 291 tokens, `prose` 4,762, `python` 6,106, `agentic` 4,265 with tools — 512 tokens each,
greedy then seeded (temperature 1, seed 7), then two pairs at once; server commit 42691b8, sources 6df4c61ee1ce3bfd,
altarfront 7a8aab4, a 2 GiB pool, jobs 16:12–16:38):

| Server | single decode tok/s, greedy: short / prose / python / agentic (mean) | seeded (mean) | pair python + prose, together | pair short + agentic, together | drafts accepted (rate) | tokens a step |
|---|---|---|---:|---:|---:|---:|
| plain (16:12) | 21.6 / 21.3 / 20.9 / 20.7 (**21.1**) | 21.9 / 21.1 / 20.9 / 21.0 (21.2) | 30.8 | 24.1 | — | 1 |
| k 1 (16:17) | 28.0 / 28.9 / 28.5 / 26.3 (**27.9**, +32 %) | 26.0 / 28.2 / 27.2 / 25.3 (26.7) | 39.1 | 30.0 | 1,449 of 1,825 (0.794) | 1.80 |
| k 2 (16:23) | 28.9 / 31.5 / 29.9 / 29.4 (**29.9**, +42 %) | 25.2 / 30.2 / 27.0 / 23.0 (26.4) | **41.0** | 30.8 | 1,846 of 2,860 (0.645) | 2.29 |
| k 3 (16:29) | 26.9 / 30.7 / 29.9 / 30.3 (**29.5**, +39 %) | 22.6 / 29.6 / 26.2 / 21.2 (24.9) | 39.9 | 28.8 | 2,017 of 3,776 (0.534) | 2.60 |
| gate, k ≤ 3, 49.4,14.2 (16:34) | 28.4 / 28.5 / 29.8 / 28.9 (**28.9**, +37 %) | 26.5 / 27.9 / 27.2 / 24.6 (26.6) | 38.7 | 30.1 | 1,561 of 2,062 (0.757) | 1.91 |

**Every configuration gave the same token ids as plain decode** on all 12 requests (`compare: all the same`, four
times; again at 16:42 for k 2 with an 8 GiB pool). Acceptance and tokens a step are over the eight single requests
(3,280 tokens). The one-draft acceptance on generated text is **0.794** (greedy 0.846: short 0.778, prose 0.893,
python 0.872, agentic 0.840; seeded 0.746), against **0.492** teacher-forced on the prompts' own text
(reference.md § 6.1, the bar): the model's own continuations are far easier to draft than the given text. Forced
k = 2 is the fastest here (+42 % single greedy, +33 % for the python + prose pair); the gate, with the real fit,
stays near k = 1's choice (1.91 tokens a step) and gives up 3 % of k = 2's single-session speed on this text; it is
the default because it closes on text where drafts miss (synthetic: 9.4 % behind plain with every draft rejected, not
the 40 % of k = 2 forced). The `short + agentic` pair is bound by `short`'s 512 tokens (agentic stops at 93).

Before the decision-8 merge (a44ead0, the bf16 combine; jobs 14:53–15:28) the same runs gave plain 21.3, k 1 28.0,
k 2 29.5, k 3 28.0, gate 29.3 tok/s single greedy, acceptance 0.774 / 0.620 / 0.502 / 0.719, all the same ids as
that commit's plain; its gate pair was not comparable (rank 0's free device gave an 11-page pool, so the pair's
sessions were evicted and prefilled again).

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

**Real rank files** (`WEIGHTS=""`: the fleet's rank files, 82,411,745,280 B each; rank 0's test tier 4 GiB; the
servers ready 45–50 s after the start, the load from page-cache-dropped files):

| Phase | Final commit af3d5d4 (sources fe90f03ad5242f43) | Checks | Earlier commits | Checks |
|---|---|---:|---|---:|
| main, batch, front, kept | 15:40 job: serve 15:42:42, batch 15:43:16, front 15:44:15, kept 15:45:11, **passed** (pool 8 pages: rank 0's free device) | 30 | 14:17 (694c448), passed | 30 |
| spec record + k 1 forced (MTP, then oracle drafts) | 15:45 job, **passed** | 1 + 52 | 14:22 (694c448), passed | 53 |
| k 2 forced | 15:45 job: **failed** (a 4-page pool: § 8); 15:55 job **passed** | 1 + 52 | 14:35 (694c448) **failed** (§ 8); 14:38 (a44ead0) passed | 53 |
| k 3 forced, the gate | 16:02 job, **passed** | 1 + 52 + 26 | 14:44 (a44ead0), passed | 79 |

On af3d5d4: **268 checks passed, 2 failed** (the 15:45 k = 2 phase, a pool too small for its 2,050-token fork; the
same phase passed at 15:55). Real decode at one session: ~21 tok/s plain (46.13 ms a step, § 5). The real-text runs
of § 5 add 5 × 12 requests compared by ids (all the same). The kept-sessions phase (planned stop, start, the turn
from disk at its full prefix) is the stop/start test.

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

**First start on the ring** (A6b, glm53's G6b protocol: `tests/handover.sh`, run on rank 0 outside `ring.sh`;
2026-10-14 16:49–16:57, server commit 742c581, the real files; the DeepSeek service had restarted at 16:48:50):

| Time | Event |
|---|---|
| 16:49:02 | `service.sh start`: supervisor, altarfront on 127.0.0.1:18895; the want registered, requests held |
| 16:51:37 | the DeepSeek service lends the ring after 120 s without a request (its idle time when a service waits) |
| 16:51:40 → 16:52:38 | the ring is altar-1's; engine started 16:51:46, serving at 16:52:38 (**58 s**: load, page-cache drop, reclaim of 26.00 GiB on rank 0 and 22.75 on rank 2); pool **246 pages (251,904 positions, 22.67 GiB)**, rank 2 bound; a 256k session did NOT fit |
| 16:52:42 → :45 | an altar-1 request, 64 tokens: **3.01 s** |
| 16:52:45 | a DeepSeek request: altar-1 gives the ring back at 16:52:46 (rank 0 exited 16:52:48), the DeepSeek engine starts 16:52:49, ready 16:53:36; its answer after **51.6 s** |
| 16:53:45 | an altar-1 request while the DeepSeek service holds the ring: it waits for that service's 120 s idle, the lend at 16:55:37, altar-1's ring at 16:55:41, serving at 16:56:34 (53 s); the answer after **172.7 s** (decode 34.9 tok/s, 40 of 46 drafts accepted) |
| 16:56:38 → :42 | `service.sh stop`: **3.6 s**, no `altar1_rank` left |

The second start's pool was 266 pages (24.52 GiB: a 256k session fitted, context 262,139) but left rank 2 at 7.36
GiB of MemAvailable after it, and the first start's ranks showed rank 2 at 7 GiB in `status`: under the fleet's 8.
Fixed in 9d10f8a (§ 3: the pool leaves 1 GiB more of MemAvailable); checked uncapped at 16:57 (a g8 job with
`--kv-gib 1000 --max-ctx 262144`, the gate: 231 pages, every rank ≥ 8.55 GiB, the ids the same as plain decode's).
The disk tier was 0 pages (rank 0's `--disk-rank0-gib` 0), so no sessions were kept across the stop.

## 8. Failures and what remains

**Failures** (2026-10-14):

1. **12:10, install**: `install.sh`'s new commit line ended in a `grep -q` that fails on a clean tree under `set -e`;
   fixed in c418d3c (`|| true`) and rebuilt from that commit.
2. **13:24, the first real start: no pool.** Rank 0 had 6.31 GiB of free device memory against 32.93 of MemAvailable
   (offers 0 / 2.96 / 6.01 / 25.74 GiB); the job stopped with "no memory for the cache pool". Cause: other files'
   page cache (§ 3) and the rank file's own on ranks 1 and 2 (dropped before the load since d3bc4e5).
3. **13:33, a pool sized from MemAvailable (d3bc4e5): rank 0 ran out of memory.** The pool's allocation failed in
   the driver (`NVRM: … NV_ERR_NO_MEMORY`), rank 0 hung, and at 13:38:39 the kernel's OOM killer killed `altar1_rank`
   (`oom_score_adj` 1000). At the same second the whole session on rank 0 was interrupted (a journald watchdog
   failure; the DeepSeek service's supervisor died and was restarted by hand). Most likely this job's memory pressure
   caused it. Reverted in 694c448 (free device binds; `--device-margin-gib` 1; the guard checks free device too).
4. **14:35, k = 2 forced on real files: the server died** ("model: row 2: position 1023, slot 0, session 0") with a
   one-page pool (rank 0's free device 1.24 GiB): the model's row check was one short (pos + passes < pool, while the
   last position written is pos + passes − 1). Fixed in 14ebc35 (`pos + max(passes, 1) <= pool`); the rerun at 14:38
   passed.
5. **15:24 and 15:40, the reclaim reclaimed nothing**: it touched only want − free (0.50, 0.25 GiB) and free device
   fell by that (§ 3). Fixed in 742c581 (touch all the pool wants).
6. **15:45, k = 2 forced on real files: 2 checks failed**: rank 0's free device gave a 4-page pool and the fork test's
   2,050-token prompt did not fit beside the other sessions ("does not fit the KV pool (4096 positions)"); an
   environment failure, not a bit mismatch. The rerun at 15:55 passed.
7. **16:52 and 16:55, the service's starts left rank 2 under 8 GiB** of MemAvailable (7 GiB in `status`, 7.36 after
   the second pool): the offer kept 8 GiB but CUDA allocates ~0.7 GiB more after the pool. Fixed in 9d10f8a (§ 3).

**What remains:**
- one 256k session in the service: rank 2's MemAvailable binds while the CPU reference runs there (§ 3); the
  service's start with 9d10f8a's margin is checked by a ring job (16:57), not by a second service start;
- rank 0's disk tier and with it kept sessions in the service (the user's to free space on rank 0's data disk);
- the gate's choice on real text (k = 2 forced is 3 % faster on § 5's prompts; the gate is kept for text where drafts
  miss), the NVFP4 drafter (decision 5's lever).
