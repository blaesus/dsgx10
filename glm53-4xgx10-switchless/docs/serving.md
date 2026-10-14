# Serving GLM-5.3 (G6's engine half)

Rank 0 of the four-rank engine as a server: the front-end, `glmfront` ([protocol.md](protocol.md)), talks to it
over a Unix socket in protocol version 2; sessions keep their caches across requests and resume at any common
prefix; prompts are prefilled in chunks and decoded with the sampler; several sessions share decode steps; the
caches live in a paged pool with a disk tier behind it; a planned stop keeps the sessions for the next start.
It is the DeepSeek engine's server (`ds41f-4xgx10-switchless/engine/server/`) re-specialised, simpler where
GLM-5.3 is simpler: its state at a position is the cache entries below it, so there are no snapshots.

Code: `engine/server/` (binary `glm53_rank`, mode `serve`). The tests are in `engine/server/tests/`.

## 1. The protocol as implemented

The frames are DeepSeek's version 1 frames with GLM-5.3's values (`engine/server/protocol.h`, a copy of
`ds41f-4xgx10-switchless/engine/server/frontend_protocol.h` with `kVersion = 2`); [protocol.md](protocol.md)
describes them from the front-end's side.

- **HELLO** (engine → front-end, once a connection): version 2, name `glm-5.3`, `max_context` = the positions a
  session may reach (`--max-ctx` less the draft rows, at most the pool's positions), vocabulary **154,856**
  (`kVocabUsed`: ids at or above it are never sampled and are refused in prompts), eos **154,820**
  (`<|user|>`; one of the three stop ids), `max_sessions` = the sessions the pool and the disk tier can hold, a page
  each.
- **SUBMIT**: request id, session id, base session id (a fork: the new session starts from the base's common
  prefix), prompt ids, `max_new`, sampling (temperature, top-p, top-k, seed), logprobs (`n_top` ≤ 20), stop ids
  (the front-end sends 154820, 154827 and 154829; any id in the list ends the request with reason `stop`), keep
  (whether the session outlives the request).
- **TOKENS** (streamed): ids as they are sampled, with their logprobs and top logprobs when asked.
- **DONE**: reason (`stop`, `length`, `context`, `cancelled`, `error` with a message), prompt tokens, **cached**
  tokens (the positions reused), completion tokens, queue, prefill and decode times.
- **CANCEL** (request id) ends a request after the step in flight; **RELEASE** (session id) frees a session's pages.
- A new connection replaces the old one; the old one's requests count as cancelled.

## 2. Sessions

A session is a token history and the caches of its positions on every rank: a run of pages in the device pool,
or pages in the disk tier when spilled, or nothing.

- **Exact prefix reuse.** GLM-5.3's state at position q is exactly the cache entries below q: no sliding
  windows, no compressed carries, no snapshots. A request therefore continues its session from **q = min(LCP,
  n − 1)**, where LCP is the longest common prefix of the session's history and the prompt; the last prompt
  position always runs again for its logits. The entries from q on are rewritten as the prefill passes them. A
  fork copies its base's first q positions into the new session's pages (`Model::cache_copy`).
- **Prompts go through prefill, outputs through decode.** A prompt is prefilled in chunks of `--chunk` rows
  (8,192); the chunking changes no bit, a one-row chunk included. Generated positions are written by decode
  steps. A request sent twice therefore gives the same tokens, and a fork at any q gives the tokens of the same
  prompt prefilled cold (both tested).
- **What differs.** A position written by a decode step is not bit-identical to the same token prefilled (the
  rounding of the one-row path differs from the chunked path). A turn that extends a session (its history, its
  last output and a new user message) continues exactly from the cached decode-written entries, which is what
  serving wants; it is not equal to a cold prefill of the whole turn. The front-end's cached-token counts are
  the truth either way.
- **Batch invariance.** Sparse attention runs with a fixed split-K (`--splits 16`) at every row count, so a
  session's tokens do not depend on the others in its steps (batch_test).

## 3. Scheduling (B1, B2)

- A request's prompt is prefilled at once, pausing the requests that decode: rank 0's control word
  (`SampleOut::ctl`, sent with the sampled tokens) stops every rank after the same step. Its first token is
  sent, then it joins the decoding requests.
- **B2**: up to `--slots` (4) sessions decode together, a row each per step.
- **B1**: while more requests decode than slots, they take turns of `--slice` (64) tokens.
- A request whose session is in flight waits for it. A request that cannot get pages while others are in flight
  waits for one of them to end.
- A session that reaches the end of its pages before `--max-ctx` is given more (`--grow`, 1,024 positions beyond
  its need) and requeued, without losing a token.
- **B3**: drafts, on by default (§ 3.1).

Measured on the test job (one MoE layer's experts from the file, the others synthetic; `--splits 16`): a decode
step of **45 ms at 1 row, 60 ms at 2, 72 ms at 3, 85 ms at 4** (22, 33, 41, 47 tokens a second in all).

### 3.1 Drafts (G7, 2026-10-14)

With drafts on (the default; `--no-spec` turns them off) a run decodes with `spec_loop` (`cmd.cpp`) instead of
`batch_loop`: one verify step at a time ([engine.md, Drafts](engine.md#drafts-g7-2026-10-14)), each session
feeding its next token and the k drafts its gate chooses (0 .. `--spec-drafts`, default 3), all sessions of the
run in one step of at most 16 rows (a session gets at least its own row). The step's samples up to the first
draft that differs are the session's tokens, so the tokens, logprobs and top logprobs are `batch_loop`'s, bit for
bit, greedy or seeded; the ends are the same too:

- a stop id or `max_new` inside a step's accepted rows ends the request there (the rows after it are not sent);
- as `batch_loop`'s queued row does, the request's last token is fed at its position before the run ends (by the
  verify row after it when its draft held, else by one more one-row step), so the session's history and its
  caches are the plain loop's: a later turn on the session reuses the same prefix and prefills the same rows
  (without this a turn after a request differed at its first token: the plain loop had decoded one position
  more, which the turn then reused while the drafts' run left it to prefill);
- cancel and pause are read from the control word after each step, as in `batch_loop`;
- rows and chained draft passes stay below the session's pages (`limit`): a session near the end of its pages
  verifies fewer drafts, and is given more pages (`run` places 2·kmax positions more than plain decoding does).

The drafts after a run (the MTP's for the token after the last one sent) go back to the request (`Flight::drafts`)
for its next run, the gate's state to the session (`Session::gate`, in memory only: a kept session's gate starts
afresh); a request's first drafts come from its prefill's sampled step. The MTP cache entries are part of every
page (`cache_copy`, `cache_get`/`cache_put`: the 79 layers' caches include layer 78's and its index cache), so they
move with a session's pages on a move, a spill and a read back, a fork's copy and a planned stop; the one entry
not exact after a turn is the position before the turn's first prefilled row (written with the token the last
step sampled, which the turn's prompt may not continue with; prefill's pending row is not kept across requests):
it only costs acceptance. Options: `--spec-drafts K` (1-5), `--spec-cost C1,DROW` (the gate's cost model, ms),
`--no-spec`; for tests `--spec-force N` (N drafts every step) and `--spec-oracle` (a request identical to an
earlier one drafts that one's tokens, about one in four changed). A verify step needs a fixed split-K
(`--splits`, default 16; `--splits 0` with drafts is refused).

The run's log line gives the tokens a step; a request's line its drafts accepted of verified.

## 4. The store

### 4.1 The device pool

- The caches of all slots are one pool of `pages × --page` positions (1,024 a page). A session gets a contiguous
  run of pages: position p of a session whose run starts at page s is pool position s × page + p
  (`Model::Options::page`, `resize_slots`).
- Rank 0 hands out the pages, the same on every rank by command. A session that outgrows its run extends it in
  place, or moves to a free run (`cache_copy`), after compacting the pool if the free pages are scattered, after
  evicting the least recently used idle sessions. A session in flight is never evicted.
- **Budget.** Every rank offers min(free device memory, MemAvailable) after the weights and the prefill buffers,
  less a margin: **4 GiB on rank 0, 2 GiB on the workers** (`--margin-gib`, `--worker-margin-gib`; the DeepSeek
  store's margins, [quantisation.md § 1.2](quantisation.md#12-glm-53s-non-weight-memory)). All ranks take the
  smallest offer, capped by `--kv-gib`.
- **A position** costs **96,640 B** a rank (`Model::cache_pos_bytes`): the BF16 latent and rope key (1,152 B) on
  79 layers (78 and the MTP layer), and the BF16 indexer key (256 B) on 22 layers (21 and the MTP layer). The
  cache is replicated, not split by position. A page is 94.4 MiB; **a 256k session (262,144 positions) is 23.59
  GiB a rank**.

### 4.2 How many 256k sessions it holds

| Weights a rank | Pool a rank (121.6 − 10 system − 4 margin − ~7 buffers − weights) | Positions | 256k sessions |
|---|---:|---:|---:|
| 101.4 GiB: the rank files today at their reserved size (logged as `file_gib`) | none (−0.7) | 0 | **0** (it does not start) |
| 87 GiB: quantisation.md's budget | ~13.6 GiB | ~151k | **0** (0.58 of one) |
| 77 GiB | ~23.6 GiB | ~262k | 1 |

So with a replicated BF16 cache **no 256k session fits in device memory** at the planned weights; a 256k session
needs the FP8 cache (54,728 B a position: 13.4 GiB, one session at 87 GiB of weights) or the cache split by
position across ranks (DCP), both later work ([quantisation.md § 1.2](quantisation.md#12-glm-53s-non-weight-memory)).
Sessions of agentic length (tens of thousands of positions) fit several at a time: 13.6 GiB holds ~147 pages.

### 4.3 The disk tier

- Every rank keeps a spill file in its spill directory (the fleet's `RANK<r>_SPILL`, `--disk-dir`), of pages of
  the pool's page size: `rank<r>[.<tag>].spill` (`--disk-tag`: `test` for the tests, `service` for the service).
- Size: every rank offers min(`--disk-gib`, the directory's free space less 32 GiB, its own file counting as
  free); all take the smallest offer.
- An evicted idle session is written to disk (`kCmdSpill`; the least recently used disk sessions are dropped to
  make room) and read back when a request names it (`kCmdUnspill`). Every page carries a 64-bit hash, checked
  on read: a mismatch stops the engine rather than serve a wrong cache.
- Writes and reads go through pinned staging with O_DIRECT. A 2,716-position session (3 pages, 283 MiB) spilled
  in 192 ms and came back in 171 ms; one page in 63–111 ms.
- A 256k session is 23.6 GiB on every rank's disk. The spill directories had 33, 57, 662 and 797 GiB free on
  2026-10-14 (ranks 0–3); rank 0's 33 GiB less the 32 GiB margin leaves the tier **about 1 GiB today**, so rank
  0's disk binds: the disk tier holds no 256k session until rank 0's spill directory has room.

### 4.4 Kept sessions across a planned stop

On SIGTERM, rank 0 ends the requests in flight, spills the pool's sessions to the disk tier, the most recently
used first, up to `--keep-gib` (0: none kept), and every rank fdatasyncs its file and writes an index
(`rank<r>[.tag].index`: the generation, the run id, the page size, every page's hash). Rank 0 writes the
histories beside it (`server[.tag].index`). The next start takes them back when every rank finds the same
generation and the same run id (a hash of the binary, the weights' path, size and mtime, the synthetic options,
the page size and the bytes a position): the front-end's session ids find their sessions as before the stop.
Sessions unused for `--keep-days` (7) are dropped. An unplanned stop keeps nothing.

## 5. Running it

### 5.1 By hand

```
engine/server/install.sh                       # builds glm53_rank (-j4) into ~/dsgb10-data/glm53-server/bin
engine/server/run4.sh --mode serve --fill-experts [options]   # in the foreground; Ctrl-C stops all four
engine/server/run4.sh stop
```

`run4.sh` ships the binary to every box and starts ranks 1–3, then rank 0. Run it only through the ring lock:
`ring.sh` (the DeepSeek engine's) for a job of a few minutes.

Options (`glm53_rank --mode serve`):

| Option | Default | |
|---|---|---|
| `--weights fleet\|PATH` | fleet | the rank files (`RANK<r>_WEIGHTS`); `%d` in PATH is the rank |
| `--fill-experts` / `--synthetic` | | routed-expert pools the file has not filled yet run synthetic / no file at all |
| `--socket PATH` | /tmp/dsgb10-glm53-engine.sock | the front-end's socket (mode 600) |
| `--max-ctx N` | 262,144 | positions a session may reach |
| `--page N` | 1,024 | positions a page |
| `--slots N`, `--slice N` | 4, 64 | B2's rows a step, B1's turns |
| `--chunk N`, `--splits N` | 8,192, 16 | prefill rows a command, sparse attention's fixed split-K |
| `--kv-gib`, `--margin-gib`, `--worker-margin-gib` | none, 4, 2 | the pool's cap and margins |
| `--disk-gib`, `--disk-dir`, `--disk-tag` | 32, `RANK<r>_SPILL`, none | the disk tier |
| `--keep-gib`, `--keep-days` | 0, 7 | kept sessions |
| `--step-timeout S` | 120 | the watchdog's limit for one step or chunk |
| `--no-spec`, `--spec-drafts K`, `--spec-cost C1,DROW` | drafts on, 3, 44.5,15.5 | drafts (§ 3.1) |
| `--spec-force N`, `--spec-oracle` | off | tests only (§ 3.1) |

### 5.2 As a service

`engine/server/service.sh` runs the four ranks and `glmfront` under a supervisor on rank 0 that restarts what
fails, with bounded backoff (the DeepSeek service's `service.sh` re-specialised).

```
engine/server/service.sh start | stop [--forget] | status | restart | forget
```

- **Settings** come from the environment at start (written to `run/settings`): `GLM53_ADDR` (127.0.0.1; specific
  IPv4 addresses only, never 0.0.0.0), `GLM53_PORT` (**18894**; 8888 is the DeepSeek service's and is refused),
  `GLM53_KEY_FILE` (mode 600), `GLM53_ENGINE_ARGS` (default `--fill-experts`), `GLM53_FRONT_ARGS`,
  `GLM53_YIELD_IDLE_S` (600), `GLM53_HOLD_S` (900), `GLM53_KEEP_GIB` (8), `MIN_AVAIL_GIB` (20). No address is
  written in the script.
- **Directories** come from the GLM fleet file (`fleet/homelab.conf`): `SERVICE_DIR` (`bin/`, `logs/` on every
  box, `run/` on rank 0), the spill directories, the workers.
- **The ring.** Both services and every job take the one ring lock (`RING_LOCK`), so the GLM engine never runs
  while the DeepSeek service or a job holds the ring, and the other way round. The GLM service is the ring's
  **borrower**: it registers a want (as a job does) when it starts and while requests wait at its front-end, and
  the DeepSeek service lends it the ring as it lends it to a job. It never takes the reclaim flag. It gives the
  ring back, its engine stopped with its sessions kept, as soon as the ring is reclaimed or once it has had no
  request for `GLM53_YIELD_IDLE_S`. Meanwhile `glmfront` holds requests (`GLM53_HOLD_S` each), and they bring
  the engine back through a want.
- **First supervised start (2026-10-14, G6b)**, on the engine built from 94ecdd5 (sources sha256
  d79152856ea93a98), default settings, the rank files with `--fill-experts`, the DeepSeek service serving in
  production (its `DSGB10_YIELD_IDLE_S` 120 s) and idle; about 4 minutes of ring time, every step as designed, no
  fix needed:
  - **Start.** `service.sh start` at 01:07:41: glmfront up, the want registered at 01:07:42; the DeepSeek service,
    idle for more than 120 s, lent the ring at once (its engine stopped in 3 s), the ring was ours at 01:07:48,
    the four ranks ready 8 s after their start (rank 0's file from /data, ranks 1–3's from their home
    directories; a pool of 575 pages, 52.99 GiB a rank), serving at 01:08:01: 20 s from `start`. A 17-token
    request with 64 tokens out: 3.1 s (prefill 339 ms, decode 23.0 tok/s).
  - **Reclaim by the DeepSeek service.** One DeepSeek request (32 tokens at most) at 01:08:25.9: its supervisor
    took the reclaim flag at 01:08:27, this supervisor saw it at 01:08:28 (its 2 s poll) and gave the ring back,
    rank 0 out at 01:08:29 and ranks 1–3 with it, the lock free and the DeepSeek service's at 01:08:31. **The GLM
    engine was off the ring 3 s after the flag (5 s after the request).** The DeepSeek request completed in
    51.9 s, 44 s of it the DeepSeek engine's own start.
  - **A GLM request while the DeepSeek service holds the ring** (01:09:30): glmfront held it, the supervisor
    registered a want (“1 requests wait”) at once, the DeepSeek service lent the ring once it had had no request
    for its 120 s (01:11:18, 120 s after its engine was ready), the ring was ours at 01:11:22, the engine ready in
    10 s and the request answered at 01:11:38: **128 s of waiting, set by the DeepSeek service's idle time**, not by
    this service. A ring job that queued meanwhile (01:10:59) waited behind this service's older want.
  - **Stop.** `service.sh stop` at 01:11:47 returned in 3 s (rank 0 out at 01:11:49, the front-end saved its
    session table); no `glm53_rank` on any rank; the lock files as before (ring.lock, ring.reclaim untouched,
    ring.owner rewritten by the next owner); the queued ring job took the ring at 01:11:51, and the DeepSeek
    service stayed lent to it (a request reclaims it, as above).
  - **Seen, not fixed.** Nothing was kept across the stops ("kept sessions: none"): rank 0's disk tier offers 0
    pages while /data has 33 GB free (§ 4.3, § 7). While it holds the ring this service does not yield to a ring
    job that waits: it gives the ring back only when reclaimed or after `GLM53_YIELD_IDLE_S` (600 s) without a
    request, so a waiting job can wait up to that long; set `GLM53_YIELD_IDLE_S` lower for a window shared with
    jobs.

## 6. Tests and results

`engine/server/tests/ring_job.sh` runs everything in one ring job of about three minutes, through the DeepSeek
engine's `ring.sh`, on the fleet's rank files with `--fill-experts` (one MoE layer's experts from the file, the
rest synthetic: the text means nothing, the checks are exact), a pool of 1.2 GiB (13 pages) so that the eviction
test evicts, a disk tier of 8 GiB (tag `test`), `--slots 4 --slice 16`:

```
OWNER=glm53-server ds41f-4xgx10-switchless/engine/server/ring.sh glm53-4xgx10-switchless/engine/server/tests/ring_job.sh
```

| Test | Checks | Result (2026-10-14) |
|---|---|---|
| `serve_test.py` | HELLO's values; a 2,600-token prompt cold; the same request again resumes at 2,599 with the same tokens; a fork at 1,500 gives the tokens of the same prompt prefilled cold; a turn prefills only its suffix (cached 2,632 = prompt + output), and again at the full prefix with the same tokens; two 5,500-token fillers evict it to disk, and it comes back at the full prefix with the same tokens; seeded sampling repeats ids and logprobs, 5 top logprobs a token, ids in the used vocabulary; cancel; an id ≥ 154,856 refused | passed (16 checks) |
| `batch_test.py` | 2, 3, 4 and 6 sessions at once (greedy and seeded sampling alternating; 6 means B1 turns) give each the tokens it gives alone; a request arriving mid-decode leaves the others unchanged | passed (5 checks) |
| `front_test.py` | `glmfront` on 127.0.0.1:18894 over the engine: the same chat completion twice gives the same text with 632 of 633 prompt tokens cached; streamed, the same text; a second turn reuses the first (657 cached); a client that leaves mid-stream is cancelled and the next request is served with the same text | passed (6 checks) |
| `serve_test.py --kept` | a planned stop (SIGTERM: 4 sessions kept, 0.55 GiB, in 0.44 s) and a start: the turn resumes at its full prefix (2,691 of 2,692) from disk with the same tokens | passed (3 checks) |
| `frontend/tools/smoke_test.py` | the front-end against its mock engine, unchanged | passed (29 checks, 0 failures) |
| `spec_test.py` (phase `spec`, G7) | 26 requests (one session greedy and seeded with logprobs; a stop id inside the text and a turn after it, greedy and seeded; 2, 3, 4 sessions at once; a request arriving mid-decode; a turn after eviction to disk; turns past and below 2,048 positions; a fork; a 160-token request) recorded on a server with `--no-spec`, then compared on one with `--spec-force 3 --spec-oracle` (twice: MTP drafts, all rejected, then oracle drafts: 26 of 39 accepted in 14 steps on a 40-token request, 34 tok/s against 21 plain) and on one with the gate: tokens, logprob bits and ends identical | passed 2026-10-14 03:33 on the merged code (1 + 52 + 26 checks) |

The ring job was run twice: on the engine before and after merging glm53 at dbc9a88 (the attention tuning),
both all passed; [experiments.md](experiments.md) lists them. With drafts (G7, 2026-10-14, the merged code at
5141dd8): every phase passed with drafts on (the default), in two ring jobs (main to kept 03:25-03:27, spec
03:27-03:33), on copies of the four filled rank files (`WEIGHTS='--weights ~/dsgb10-data/glm53-g7/weights/rank%d.glw
--fill-experts'`, `SERVER_DATA` for the job's own data tree). The phase took three tries: the first two found the
history difference of § 3.1 (a turn after a request differed at its first token), fixed by feeding the last
token.

## 7. What remains

- **The FP8 cache or DCP**, for a 256k session at the planned weights (§ 4.2); the store's budget follows
  `cache_pos_bytes`, so neither changes the server.
- **Drafts on real text** (G7): acceptance and tok/s end to end through glmfront wait for the routed experts of
  layers 3-77; the verify loop runs a step at a time (no step queued ahead); the gate's cost model is one
  session's.
- **The service on the ring**: started, lent, reclaimed and stopped (§ 5.2, 2026-10-14); a planned stop with
  kept sessions waits for rank 0's disk tier.
- **pi against the test port** with the real experts (G6's acceptance test, once the rank files are filled).
- **Rank 0's disk**: the spill directory needs room for a disk tier worth having.
- Spills on demand are synchronous: a request that needs pages waits for the evicted session's write (~100 ms a
  page).
