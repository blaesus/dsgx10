# Review: the method of the comparison with the SGLang kit (2026-10-13)

Pre-release review, question 4: are the measurements objective and fair, especially against other
kits' numbers? This part covers the **method** of the `bench/bench.py` comparison with the SGLang TP=4
kit that served on this ring: the rows of the README's "Where it stands". MiaAI's published figures,
other stacks and precision parity are other reviewers' parts. No GPU job ran for this review; it read
data, logs and tools. A same-day re-measurement on the official checkpoint (wave 2) is specified at
the end for the orchestrator to run.

Sources:
- ours: release candidate 5's three sessions b1–b3 in rank 0's /data/dsgb10/bench/rc5 (`bench.json`,
  `bench.log`, `run4.log`, `front.log`; `scripts/bpart.sh`, `scripts/rc5table.py`), and the edit-bench
  dumps e1, e2 there for token accounting;
- the kit: `bench/baseline-sglang-{decode,prefill}.json` and `bench/golden_sglang.json` (2026-10-11), its
  launch record `~/ai/dsv41-4x-ring/state-tp4/launch.json`, its scripts (`start.sh`, `start-tp4.sh`,
  `boot.py`, `stop.sh`, `files/nfs-share.sh`, `scripts/remote.py`), its settings (`.env.tp4`, never its key
  lines) and its engine log `logs-tp4/dsv41.log`, of which only timestamps, counts and rates were read,
  never a prompt or an answer.

## Verdict

The comparison is fair in direction and in size, but its statistics fall short of objective. Both
sides ran the same instrument (bench.py's decode, prefill and agentic code is unchanged since the kit's
run) and the same prompts. The decode prompts came to 39/36/55/51 tokens on both servers, and the
prefill prompts, whose nonce differs per request, to within one token (944–57,542). Both ran greedy
with thinking off, on the same weights and the same boxes, and nothing changed on the boxes between the
two days. No other traffic reached the kit during its run. Both servers count tokens the same way,
end-of-sentence id included, and ours re-tokenises exactly (33,311 tokens in 252 requests). The
advantage is far
outside every dispersion seen. Decode is 1.40–1.62× the kit's (1.38–1.63× between the extreme
requests), against spreads of at most 0.6 % for ours and 0.8–2.2 % between the kit's two requests.
Cold prefill runs at 1.5–2.2× the kit's rate. The agentic first turn is 2.1× as fast and the later
turns 1.8× (medians). Per decode step, ours accepts within about 3 % as many tokens as the kit on every
prompt type; the whole gain is the step time, 23.9–32.7 ms against 37.8–47.0, so it does not come from
a luckier text. The kit also runs an approximation that favours its speed (an FP4 indexer), while
ours is exact.

The presentation is where it falls short:
- the kit's column is one unreplicated session from 2026-10-11;
- the kit's time to first token was never kept where it can be checked;
- every figure is single-stream while the kit is set up to batch 16;
- our TTFT range leaves out each start's first request.

The READMEs said none of this. These are majors, not blockers. Commit 56dbbce states the first,
third and fourth in both READMEs; only a re-measurement can fix the second.

One thing does block release: the user's rule that every published figure be on the official
checkpoint. Both columns ran the community variant. So wave 2 is needed in any case. It also gives the
kit three sessions after a warm-up on the same day as ours (ours–kit–ours), records TTFT, texts,
chunking and step logs on both sides, and takes the kit's greedy logprobs and GSM8K-style accuracy on
the official weights for the correctness review.

## Findings

Severity: a **blocker** must not be published as it stands. A **major** claim stands, but a reader
lacks a condition needed to judge it, or a published figure cannot be checked. A **minor** finding is
an imprecision or gap that changes no conclusion.

| # | Severity | Finding | Fix |
|---|---|---|---|
| B1 | blocker (release rule, not method) | Both columns were measured on dealignai's DeepSeek-V4.1-Flash-UNCENSORED-FP8 (`d61c59ea`). The user requires the official checkpoint for every published figure. | wave 2, both sides on rank 0's /data/models/DeepSeek-V4.1-Flash |
| M1 | major | The kit's column is one session on 2026-10-11; ours is the mean of three on 2026-10-13. The configuration README gave neither the kit's date nor its N, and neither README gave a spread. | README (56dbbce); wave 2: three kit sessions, same day as ours |
| M2 | major | The kit's TTFT, 244–280 ms, cannot be checked. bench.py's JSON stores no decode TTFT, its console output from 2026-10-11 was not kept (no file under rank 0's /data/dsgb10 from 13:00–15:00 that day), and the kit's log stamps whole seconds. | wave 2 (`bench_record.py` keeps every request's TTFT and SSE lines) |
| M3 | major | Scope. Every figure is one request at a time. The kit was launched for batches (`--max-running-requests 16`, `--cuda-graph-max-bs-decode 16`). Its throughput under concurrency is not compared, and neither the tables nor the root README's summary row said "one stream". | README (56dbbce) |
| m1 | minor | Our TTFT range, 123–147 ms, leaves out each session's first request after the engine start (165, 167, 168 ms). rc5table.py does this and docs/engine.md says so; the README did not. The kit had been serving for three days. | README (56dbbce); wave 2 reports each session's first request apart, on both sides |
| m2 | minor | The kit's decode figures are means of bench.py's printed one-decimal values, rounded again: 77.4, 73.6 and 106.9 against exact means of 77.32, 73.55 and 106.85 (prose 46.8 = 46.79). This favours the kit by ≤ 0.1 tok/s. | wave 2 replaces them |
| m3 | minor | The kit's second request of each prompt was faster than its first on all four types (+1.25, +1.18, +0.84, +2.25 %), and the mean of the two is published. Ours: two requests within 0.15 % (once 0.4 %). Against the kit's faster request alone the ratios are 1.39–1.61, not 1.40–1.62. | wave 2: a warm-up session before the kit's three |
| m4 | minor | bench.py's JSON keeps only rates: no texts, per-request usage, TTFT, chunking or wall-clock times. Neither side's greedy text or token count can be re-checked from the files. | `bench/bench_record.py` (727230d) records every request; `bench_record_report.py`, `steps.py` |
| m5 | minor | The rate is (completion_tokens − 1) / (t_end − t_first), from the first content chunk on. dsfront sends one token per chunk. A server whose first chunk holds k tokens gets the k − 1 extra counted as decoded after it: +(k − 1)/(512 − k), at most 0.8 % for k = 5. How the kit chunked was not recorded. | wave 2: `bench_record_report.py` prints the first chunk's tokens on both sides |
| m6 | minor | Agentic turns. The kit's prefix cache matches whole 256-token pages, so its turns 1–5 prefilled 1–10 % more new tokens than ours: 2209, 2009, 1873, 1835, 1770 against 2035, 1822, 1858, 1705, 1675 (turn 0: 15,774 tokens on both). Prefix reuse is part of what is compared, but "turns adding ~2k tokens" means 1.7–2.2k. | note; wave 2 records the new tokens on both sides |
| m7 | minor | The root README's summary row paired our cold-prefill range from two instruments (8.5–9.4k; the 9.4k is sparkDash's, to 120k tokens) with the kit's bench.py figure (4.2k). | README (56dbbce): 8.5–9.2k against 4.2–4.3k |
| i1 | info | Model parity on 2026-10-11/13 holds. The kit's head mounted `/var/tmp/DeepSeek-V4.1-Flash`, a link to /data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8, and its three workers mount the volume `dsv41-tp4-weights`, a bind of `~/DeepSeek-V4.1-Flash-UNCENSORED-FP8` on each box. Our converter read the same directories. | — |
| i2 | info | launch.json's `revision` fb2764a5 is `boot.py`'s `REVISION` constant: the Hugging Face revision of the official checkpoint that its prepare step would download, written whatever directory is mounted. cad252b is the kit repository's HEAD (`git log`), which DESIGN.md cites. They name different things. | — |
| i3 | info | Image. `.env.tp4` sets `IMAGE` twice, and line 433, `dsv41-4x-spark:canary-roce`, wins over line 81. All four boxes hold it, created 2026-10-08T16:24:3x UTC, before the kit's 10-08 start and unchanged since. Its ID is 7d01ae1caf6f on rank 0, rank 1 and rank 2 and 95338888b189 on rank 3. | — |
| i4 | info | Conditions, 2026-10-11 13:54:09–13:55:49 UTC. Only bench.py's two invocations ran, from 127.0.0.1 (19 chat completions, plus `/v1/models` once and `/health` 4 times), with never more than one request running and none queued. About 3 h idle came before; the golden recording followed at 13:56:03–08 and the stop at 13:56:42. Between the days no box rebooted (boots 2026-09-25 and 09-29), the driver stayed 580.178.04, and rank 0's 2200 MHz clock cap was active (the workers have none). The kit's docs find `--sleep-on-idle` changes no decode. | — |
| i5 | info | The kit computes the sparse-attention indexer in FP4 (`--enable-deepseek-v4-fp4-indexer`), an approximation for speed; ours is exact. What this and the kit's other flags do to outputs is the precision review's part. It ran without RoCEnante (`SGLANG_ROCE_ALLREDUCE=0`; a ring has no opposite-node RDMA path), so "as it served on this ring" is accurate, and its author's switched-fabric figures are higher (DESIGN.md; the MiaAI table is that comparison). Both sides use Decoder SWA Bounded Replay in prefill (`--enable-decoder-swa-bounded-replay`; ours, CED). | — |

## Dispersion

Decode, tok/s (bench.py's rate, 512 tokens, two requests per prompt type per session):

| decode tok/s | ours: session means, b1 / b2 / b3 | ours: mean | ours: 6 requests | spread | kit: request 1 / 2 | kit: mean | kit: spread | ratio | ratio, extreme requests |
|---|---|---|---|---|---|---|---|---|---|
| prose | 75.85 / 75.67 / 75.51 | **75.68** | 75.45–75.89 | 0.6 % | 46.50 / 47.08 | **46.79** | 1.2 % | 1.617 | 1.603–1.632 |
| code | 111.84 / 111.66 / 111.47 | **111.66** | 111.46–111.85 | 0.3 % | 76.87 / 77.78 | **77.32** | 1.2 % | 1.444 | 1.433–1.455 |
| agentic | 109.35 / 109.20 / 108.95 | **109.17** | 108.90–109.35 | 0.4 % | 73.24 / 73.85 | **73.55** | 0.8 % | 1.484 | 1.475–1.493 |
| json | 150.09 / 149.74 / 149.51 | **149.78** | 149.23–150.13 | 0.6 % | 105.66 / 108.04 | **106.85** | 2.2 % | 1.402 | 1.381–1.421 |

Spread is (largest − smallest) / mean. Our session means fall by 0.2 % per session, b1 > b2 > b3 on
every type; the standard deviation of the three means is 0.17–0.29 tok/s. The kit's two requests
are its only replication.

Cold prefill, tok/s (prompt_tokens / time to the one token):

| cold prefill | prompt tokens | ours: b1 / b2 / b3 | ours: mean | spread | kit: one request | ratio |
|---|---|---|---|---|---|---|
| ~1k | 944/945 (kit 944) | 2931 / 2906 / 2933 | 2923 | 0.9 % | 1930 | 1.51 |
| ~4k | 3748 (kit 3747) | 6310 / 6276 / 6275 | 6287 | 0.5 % | 3674 | 1.71 |
| ~16k | 16419 (kit 16419) | 8534 / 8558 / 8502 | 8531 | 0.7 % | 4311 | 1.98 |
| ~32k | 28653/28654 (kit 28654) | 8909 / 8882 / 8914 | 8902 | 0.4 % | 4171 | 2.13 |
| ~64k | 57542 (kit 57542) | 9239 / 9201 / 9200 | 9213 | 0.4 % | 4254 | 2.17 |

Agentic:
- first turn (15,774 tokens, cold): ours 1.798 / 1.800 / 1.806 s (mean 1.802); the kit 3.728 s, one
  request;
- turns 1–5: ours 0.387–0.419 s over 15 turns (median 0.396); the kit 0.786, 0.744, 0.730, 0.715,
  0.709 s (median 0.730), one each.

Time to first token, the eight decode requests of each session, in ms (bench.log):

| session | requests in order: prose, prose, code, code, agentic, agentic, json, json |
|---|---|
| b1 | **168**, 132, 125, 123, 144, 142, 144, 142 |
| b2 | **167**, 135, 126, 125, 146, 144, 144, 147 |
| b3 | **165**, 131, 123, 123, 141, 142, 142, 143 |

The first of each session (bold) came right after the engine started. The other 21 are 123–147 ms,
median 142. The kit's 244–280 ms has no surviving per-request record (M2).

## Token accounting

- **Ours.** `usage.completion_tokens` is the number of ids dsfront kept, end-of-sentence id included
  (`frontend/src/server.rs`). Checked on RC5's edit-bench dumps e1 and e2: 252 of 252 requests had
  usage equal to the kept ids, every stop counted its end-of-sentence id, and the 33,311 generated
  tokens re-tokenised (`dsfront tokenize`, the checkpoint's tokenizer.json) to exactly 33,311, request by
  request. dsfront sends one SSE chunk per token.
- **The kit** counts the end-of-sentence id too. In `golden_sglang.json`, `math` has 2 completion tokens
  for "391" plus the end, and `fact` 73 for a 72-token text plus the end; the 96-token
  length-capped ones re-tokenise to 96. (The tokenizer.json files of the two checkpoints are the same
  file.)
- bench.py's decode requests end by length at 512 on our side (run4.log). Either way the rate uses
  the count each server reports, so a stop could not inflate it.

## The instrument on both sides

| | ours (2026-10-13) | the kit (2026-10-11) |
|---|---|---|
| client | bench.py from rank 0, through dsfront on 127.0.0.1 | bench.py from rank 0, to the kit's API on 127.0.0.1:8888 |
| bench.py | 191c30f; decode, prefill and agentic code identical to 6d35638 (the diff adds only the edit mode, ffdaab5) | 6d35638 |
| prompts | the four decode prompts: 39, 36, 55, 51 prompt tokens | the same: 39, 36, 55, 51 (its prefill lines) |
| sampling | `temperature` 0, `chat_template_kwargs: {"thinking": false}` (dsfront honours it) | the same request; the kit's own launch example turns thinking off the same way |
| length | `max_tokens` 512, all requests ended by length | 512 |
| rate | (completion_tokens − 1) / (t_end − t_first) | the same code |
| TTFT | to the first SSE `data:` event with content, reasoning or tool calls; role-only chunks and SSE comments do not count | the same code |
| warm-up | none: each session a fresh engine, so its first request is the first after start | none in bench.py; the kit had warmed itself at its start on 2026-10-08 (prompts of 12–3,600 words and a batch of 16) and served since |
| prefix cache | decode prompts: "reused 0 (cold)"; prefill: a nonce prefix each request, cold; agentic: each turn reuses the previous | decode prompts are shorter than its 256-token page, so not cacheable; prefill cached 0 each; agentic turns reuse whole pages (m6) |
| model | the variant (i1) | the same files (i1) |
| conditions | the ring held by the job, production stopped, nothing else running | only bench.py, ≤ 1 running, 0 queued (i4) |

## Where the advantage comes from

Tokens per step and time per step. Ours comes from rank 0's per-request report in run4.log; b1–b3
agree to 0.2 ms. The kit's comes from the decode-batch lines of its log during the bench: accept
length per logging interval, and ms per step = 1000 × accept length / generation rate.

| | ours: tokens per step | ours: ms per step | kit: tokens per step | kit: ms per step (median) |
|---|---|---|---|---|
| prose | 1.81 | 23.9 | 1.79 | 37.8 |
| code | 3.19 | 28.6 | 3.27 | 43.5 |
| agentic | 3.11 | 28.5 | 3.21 | 43.5 |
| json | 4.90 | 32.7 | 4.98 | 47.0 |

The two stacks' greedy texts differ, since they sum in different orders. Even so, our tokens per step
are within about 3 % of the kit's, slightly fewer on three types. The whole advantage is the step
time, 0.63–0.70× the kit's. The kit's figures sample its text at a few log lines per request, so they hold to
a few per cent. In wave 2 `bench/steps.py` derives both sides' figures from the recorded requests.

## Wave 2: a same-day re-measurement on the official checkpoint

### What and why

- **Why.** B1: the user's rule requires the official checkpoint, rank 0's /data/models/DeepSeek-V4.1-Flash
  (476 GiB with the Engram tables; `config.json` identical to the variant's). The run also fixes M1,
  M2, m1, m3, m4 and m5 at their source. It records the kit's greedy logprobs and GSM8K-style accuracy
  on the official weights for the correctness review.
- **Order, same day:**
  1. O1: ours, three sessions;
  2. K: the kit, one warm-up session and three measured, then the golden recording, then the GSM8K
     harness;
  3. O2: ours, three sessions.

  Each step is its own ring window. Ours–kit–ours shows drift as O2 − O1. A kit start costs 6–10
  minutes, which rules out more alternation.
- **Each session:** `bench.py --modes decode,prefill,agentic` at its defaults, through `bench_record.py`.
  That is greedy, thinking off, 512 tokens and 2 requests per prompt type; cold prefill at ~1, 4, 16,
  32 and 64k; agentic 16k then 5 turns of +2k, 64 tokens each.
- **Model name `deepseek-v4.1-flash` on both:** the kit's `SERVED_MODEL_NAME`, and ours
  `dsfront --model-name deepseek-v4.1-flash`.
- **Keys from files only.** The kit's is `~/ai/dsv41-4x-ring/state-tp4/api-key`, the same value as its
  `.env.tp4`, which the kit loads. Ours is the file its dsfront was started with. Never print either,
  and never run a `docker inspect` that shows the head container's environment, which holds the key.
- **Data:** rank 0's /data/dsgb10/bench/wave2/{kit,o1,o2}, mode 700. Run the tools from a checkout at a
  recorded commit that contains 727230d (`$W` below is its configuration directory).

### One launch of the kit on the official checkpoint, its files unchanged

`start-tp4.sh` hard-sets `ENV_FILE=.env.tp4`. So the launch calls `start.sh` with the variables
`start-tp4.sh` exports, and `ENV_FILE` names an overlay. `start.sh` sources the overlay with `set -a`
before it applies its defaults and resolves `MODEL_DIR`, so the overlay's values reach everything below.

| variable | the kit's setting (`.env.tp4`) | effect | for this launch |
|---|---|---|---|
| `MODEL_DIR` | line 428: the variant | the checkpoint; `start.sh` resolves it (`readlink -f`), `serve` downloads only if it has no `config.json`, `dsv41-nfs` exports it | /data/models/DeepSeek-V4.1-Flash |
| `COMMON_MODEL` | line 78: /var/tmp/DeepSeek-V4.1-Flash | `serve` and `share` run `ln -sfn MODEL_DIR COMMON_MODEL`; the head mounts its target at /models/DeepSeek-V4.1-Flash | /var/tmp/DeepSeek-V4.1-Flash-official, a new link, so the kit's own link keeps pointing at the variant |
| `SERVED_MODEL_NAME` | line 429: deepseek-v4.1-flash-uncensored | `--served-model-name` | deepseek-v4.1-flash (the file's own comment for the original) |
| `PORT` | line 437: 8888 | the API port (`SERVER_PORT` in boot.py) | 18888: production's dsfront keeps 8888 during a lend |
| `NFS_SHARE`, `NFS_VOLUME` | lines 426–427: 0, `dsv41-tp4-weights` (each worker's bind of the variant) | the volume the worker containers mount read-only at /models/DeepSeek-V4.1-Flash | 1, `dsv41-tp4-official` |
| `NFS_SERVER_IPS`, `NFS_CLIENTS` | lines 92–93: the TP=3 Sparks' 10.0.22.x | the address workers mount; the export's clients | 10.0.0.1; 10.0.0.2,10.0.0.3,10.0.0.4 |
| `STATE_DIR`, `LOG_DIR`, `SERVE_LOG` | `start-tp4.sh`: `state-tp4`, `logs-tp4`, `logs-tp4/dsv41.log` | launch.json, smoke.json and api-key are written to `STATE_DIR`; `serve` truncates `SERVE_LOG` | new directories under wave2/kit, so the 2026-10-08/11 records stay |
| `SKIP_PREPARE`, `SKIP_VERIFY` | lines 186–187: 1, 1 | see below | unchanged, 1 and 1 |

**How the workers get the weights.** ranks 2 and 3 hold the official copy at
`~/DeepSeek-V4.1-Flash` (48 shards; their variant hardlinks the 21 shared shards). rank 1 holds only the
variant, and the 27 shards that differ (13–39, 199.6 GB) do not fit in its 146 GB free. So all three
workers read rank 0's copy over NFS, the kit's own path (`NFS_SHARE=1`). `serve` runs `share` first:
- `nfs_ensure_server` starts the container `dsv41-nfs` on rank 0. It uses the image `dsv41-nfs:local`
  (present, built 2026-09-19), runs the kernel nfsd on the host network with `--restart unless-stopped`,
  and exports **`$MODEL_DIR`** read-only as /export (`fsid=0`) to `NFS_CLIENTS`. The overlay's
  `MODEL_DIR` reaches it.
- On each worker, `nfs_ensure_worker_volume` removes and recreates `NFS_VOLUME` as an NFS volume
  (`addr=10.0.0.1,nfsvers=4.2,ro,nconnect=8,rsize=wsize=1 MiB,hard`, device `:/`). A throwaway container
  of the serving image then checks that `config.json` and 48 non-empty shards are readable through it,
  or `serve` stops. The variant's volume `dsv41-tp4-weights` is not touched.
- Checked from here, without starting anything:
  - rank 0's firewall passes TCP 2049 and 111 from all three workers: a connect to a closed port was
    refused, not dropped, as for the kit's port 20000;
  - each worker routes to 10.0.0.1 from its own 10.0.0.x (rank 2 through a neighbour);
  - nothing listens on 2049 or 111 on rank 0, and no host nfsd or rpcbind runs;
  - the `nfsd` module is installed but not loaded. The container's `mount -t nfsd` should load it; if
    the export does not come up, run `sudo modprobe nfsd` first.

**What `SKIP_PREPARE=1` and `SKIP_VERIFY=1` skip, and why no prepare step is needed.** With
`SKIP_PREPARE=1`, boot.py's `prepare` only asserts `/models/DeepSeek-V4.1-Flash/config.json` on each
rank. Without it, rank 0 would download revision fb2764a5 from Hugging Face into the mounted directory,
which the head mounts read-write. `SKIP_VERIFY=1` skips re-hashing it. Both must stay 1. Nothing else
depends on the checkpoint:
- **DSpark tables.** There are no SPS/STS tables. `state-tp4` holds none, and the 2026-10-08 start
  logged "not found; staying on the verify-all schedule". `push_spec_tables` clears stale copies on the
  workers, as on 2026-10-11.
- **Engram and draft.** The packed Engram rows (rank 0 `~/dsv41-engram-tp4`, the workers
  `~/dsv41-4x-ring/engram`) come from shards 47–48 and the draft from shards 44–46. These are the same
  files in both checkpoints (the same inodes on rank 0, ranks 2 and 3), so no `pack` is needed.
- **The served name** is the setting above.
- **The revision.** `launch.json` will again say revision fb2764a5, boot.py's constant, whatever
  revision rank 0's copy is; the official-checkpoint review's hashes say which.

**First start.** On 2026-10-08 the kit went from container start (18:19:2x UTC) to Ready in about
6 min: 154.7 s of weights (18:19:57–18:22:32), 11.6 s of draft, API up at 18:24:25, then the smoke
test and an 11-s warm-up. This time the official shards are cold in every page cache, and all four ranks read
rank 0's /data, a USB4 SSD (`Extreme Pro USB4`), three of them over NFS. A new `STATE_DIR` also makes rank 0
compile its b12x kernels again, as the 10-08 start did. Expect 6–10 min to Ready; the job allows 20.

**Fallbacks, decided before the window.**
- **H.** If NFS through a neighbour fails for rank 2 or rank 3, use `NFS_SHARE=0` with
  `dsv41-tp4-official` made by hand:
  - on ranks 2 and 3, `docker volume create --driver local --opt type=none --opt o=bind
    --opt device=$HOME/DeepSeek-V4.1-Flash dsv41-tp4-official`, as `dsv41-tp4-weights` binds the variant;
  - on rank 1, an NFS volume to rank 0's export, over the direct 100G link;
  - on rank 0, `dsv41-nfs` started by hand with `NFS_CLIENTS=10.0.0.4`.
- **L.** A local copy on rank 1 needs about 60 GB freed there first.

### Before the window (production up, no GPU)

1. Image on every box. Run `docker images dsv41-4x-spark:canary-roce` on rank 0, rank 1 (10.0.0.4),
   rank 2 (10.0.0.3) and rank 3 (10.0.0.2). Expect 7d01ae1caf6f on the first three, 95338888b189 on
   rank 3, all created 2026-10-08T16:24:3x. Stop if one is missing: `serve` would build.
2. Weights. Check `cmp /data/models/DeepSeek-V4.1-Flash/config.json
   /data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/config.json` and 48 shards in the official
   directory, and that the packed Engram directories are present.
3. Record the kit's own state, so it can be shown unchanged afterwards:
   - `sha256sum ~/ai/dsv41-4x-ring/state-tp4/{launch,smoke}.json`;
   - `stat -c '%s %y' ~/ai/dsv41-4x-ring/logs-tp4/dsv41.log`;
   - `readlink /var/tmp/DeepSeek-V4.1-Flash`;
   - `docker volume ls` on each worker;
   - `sha256sum ~/dsv41-4x-ring/state/launch.json` on each worker. Every rank saves its own
     `launch.json` at a start, so this launch rewrites the workers' copies (from 2026-10-08): the one
     record of the kit it changes. Step 4 keeps copies.
4. `mkdir -m 700 -p /data/dsgb10/bench/wave2/{kit,o1,o2}`, then
   `mkdir -p /data/dsgb10/bench/wave2/kit/{state,logs,before}`. Copy each worker's `launch.json` to
   `before/<address>.launch.json`. Write `/data/dsgb10/bench/wave2/kit/env.official`:

   ```bash
   # The kit's own settings, then one launch on the official checkpoint. start.sh and stop.sh source
   # this file with set -a; scripts/remote.py skips the source line and keeps the exported values.
   source "$HOME/ai/dsv41-4x-ring/.env.tp4"
   MODEL_DIR=/data/models/DeepSeek-V4.1-Flash
   COMMON_MODEL=/var/tmp/DeepSeek-V4.1-Flash-official
   SERVED_MODEL_NAME=deepseek-v4.1-flash
   PORT=18888
   NFS_SHARE=1
   NFS_VOLUME=dsv41-tp4-official
   NFS_SERVER_IPS=10.0.0.1
   NFS_CLIENTS=10.0.0.2,10.0.0.3,10.0.0.4
   ```

5. Dry-run the weight path, which touches no GPU:

   ```bash
   O=/data/dsgb10/bench/wave2/kit
   env ENV_FILE=$O/env.official ENV_EXAMPLE=$HOME/ai/dsv41-4x-ring/.env.tp4.example STATE_DIR=$O/state \
       LOG_DIR=$O/logs SERVE_LOG=$O/logs/dsv41.log DSV41_LAUNCHER=tp4 \
       ~/ai/dsv41-4x-ring/start.sh share > $O/share.log 2>&1
   ```

   This starts `dsv41-nfs`, makes `dsv41-tp4-official` on the three workers, and checks the 48 shards
   through each. Leave them up: `serve` reuses the export and remakes the volumes. If it fails, use
   fallback H, or tear it down with step 8 below.

### The kit's window

Run it under the ring lock, not preemptible:

```bash
OWNER=wave2-kit PREEMPT=0 $W/engine/server/ring.sh bash /data/dsgb10/bench/wave2/kitpart.sh
```

`PREEMPT=0` matters. A preemption kills the job's processes but not the kit's containers, which would
keep the GPUs while the service restarts its engine. `kitpart.sh` is in the style of RC5's `bpart.sh`.
`W`, `SCR` and the trap are the orchestrator's to fill in:

```bash
#!/bin/bash
# wave 2, the kit on the official checkpoint: start, warm-up + 3 sessions, golden, GSM8K, stop.
set -u
W=...                                      # the configuration directory of a checkout containing 727230d
SCR=...                                    # the review's scratchpad (the correctness harness)
K=$HOME/ai/dsv41-4x-ring                   # the kit, unchanged
O=/data/dsgb10/bench/wave2/kit
KEY=$HOME/ai/dsv41-4x-ring/state-tp4/api-key
URL=http://127.0.0.1:18888/v1
MODEL=deepseek-v4.1-flash
KENV=(ENV_FILE=$O/env.official ENV_EXAMPLE=$K/.env.tp4.example STATE_DIR=$O/state LOG_DIR=$O/logs
      SERVE_LOG=$O/logs/dsv41.log DSV41_LAUNCHER=tp4)
WS=(10.0.0.4 10.0.0.3 10.0.0.2)
SSH=(timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=10)
log() { echo "$(date -u '+%F %T') kitpart: $*"; }
[ -n "${RING_HOLDER:-}" ] || { log "run me under ring.sh"; exit 2; }
boxes() {  # the state of every box, for the record
  for h in rank 0 "${WS[@]}"; do
    c="pgrep -xc dsgb10_rank; docker ps --filter name=dsv41- --format '{{.Names}}' | tr '\n' ' ';
       nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader;
       awk '/MemAvailable/ {printf \"MemAvailable %.0f GiB\n\", \$2 / 1048576}' /proc/meminfo"
    if [ $h = rank 0 ]; then echo "$h: $(bash -c "$c" | tr '\n' ' ')"
    else echo "$h: $("${SSH[@]}" -- $h "$c" | tr '\n' ' ')"; fi
  done
}
STARTED=0
cleanup() {
  [ $STARTED = 1 ] || return 0
  STARTED=0
  log "stop"; env "${KENV[@]}" "$K/stop.sh" --unmount > "$O/stop.log" 2>&1
  rm -f /var/tmp/DeepSeek-V4.1-Flash-official
  log "after the stop:"; boxes
  for h in "${WS[@]}"; do log "$h volume left: $("${SSH[@]}" -- $h "docker volume ls -q | grep -x dsv41-tp4-official")"; done
  ss -ltn | grep -q ':18888 ' && log "18888 STILL OPEN"
  log "/var/tmp/DeepSeek-V4.1-Flash -> $(readlink /var/tmp/DeepSeek-V4.1-Flash)"
  # the root-owned copy of the key that boot.py wrote into the new STATE_DIR
  docker run --rm --network none -v "$O/state:/s" --entrypoint /bin/rm dsv41-4x-spark:canary-roce -f /s/api-key
}
trap cleanup EXIT
trap 'log "signal"; exit 143' TERM INT
log "the ring: $(cat /data/dsgb10/ring.owner)"; boxes
for h in rank 0 "${WS[@]}"; do  # no rank of ours may be left
  if [ $h = rank 0 ]; then n=$(pgrep -xc dsgb10_rank); else n=$("${SSH[@]}" -- $h "pgrep -xc dsgb10_rank"); fi
  [ "${n:-0}" = 0 ] || { log "$h still runs dsgb10_rank"; exit 1; }
done
STARTED=1
log "start"
env "${KENV[@]}" READY_TIMEOUT=120 "$K/start.sh" serve > "$O/start.log" 2>&1 < /dev/null || { log "start failed"; exit 1; }
log "ready: $(python3 -c "import json,os,urllib.request as u; k=open(os.path.expanduser('$KEY')).read().strip();
print(json.load(u.urlopen(u.Request('$URL/models', headers={'Authorization': 'Bearer ' + k})))['data'][0]['id'])")"
boxes
for s in k0 k1 k2 k3; do  # k0 is the warm-up, left out of the statistics
  timeout 600 python3 "$W/bench/bench_record.py" --record "$O/$s.jsonl" --url "$URL" --key-file "$KEY" \
      --model "$MODEL" --modes decode,prefill,agentic --out "$O/$s.json" > "$O/$s.log" 2>&1
  log "$s: exit $?"; boxes
done
timeout 300 python3 "$W/bench/golden_sglang.py" "$KEY" --url "$URL" --model "$MODEL" \
    --out "$O/golden_sglang_official.json" > "$O/golden.log" 2>&1
log "golden: exit $?"
timeout 1800 python3 "$SCR/review/correct/gsm8k_eval.py" --base-url "$URL" --key-file "$KEY" --model "$MODEL" \
    --data "$SCR/review/correct/gsm8k/test.jsonl" --out "$O/gsm8k" --n 100 --max-tokens 2048 \
    --no-ring-guard > "$O/gsm8k.log" 2>&1
log "gsm8k: exit $?"
```

Notes on the script:
- `start.sh serve` returns once boot.py prints `Ready: API on port 18888`, after its own smoke test
  and warm-up (prompts of 12–3,600 words and a batch of 16). It exits 1 if the head dies, and gives up
  after `READY_TIMEOUT` × 10 s. It also runs the NCCL preflight and refuses a busy GPU.
- The correctness harness needs `--no-ring-guard`: the ring is lent to this job, and the guard would
  wait for it.
- `stop.sh --unmount` reads `ENV_FILE`. With the overlay it removes every `dsv41-*` container (head,
  `dsv41-nfs`, workers) and only the `dsv41-tp4-official` volumes. **Never run it with `--unmount`
  under the kit's default env file**, which would drop `dsv41-tp4-weights`.
- Afterwards, compare the records of step 3: `state-tp4` and `logs-tp4` unchanged, and
  `/var/tmp/DeepSeek-V4.1-Flash` still pointing at the variant. The workers' `launch.json` now
  describe this launch; `before/` keeps those of 2026-10-08.
- `golden_sglang.py` refuses an existing `--out` file, so `bench/golden_sglang.json` is safe. Commit the
  new file as `bench/golden_sglang_official.json`.

**8. Teardown without a window** (if step 5 ran but the window does not):
`env ENV_FILE=... LOG_DIR=... ~/ai/dsv41-4x-ring/stop.sh --unmount` with the same variables as step 5,
then `rm /var/tmp/DeepSeek-V4.1-Flash-official`.

### Ours, before and after (O1, O2)

Use the engine on the official weights at production's arguments, as the official-checkpoint work sets
it up, with dsfront `--model-name deepseek-v4.1-flash`. Each session is a fresh engine, as RC5's
`bpart.sh` does, with `bench_record.py` in place of `bench.py`:

```bash
timeout 480 python3 "$W/bench/bench_record.py" --record "$D/bench.jsonl" --url "http://127.0.0.1:$PORT/v1" \
    --key-file "$KEY" --model deepseek-v4.1-flash --modes decode,prefill,agentic --out "$D/bench.json" \
    > "$D/bench.log" 2>&1
```

Run three sessions in O1 and three in O2, each bracket one ring job (`OWNER=wave2-ours PREEMPT=0
ring.sh`), and keep each session's `run4.log` beside its records.

### Statistics, and what to publish

```bash
B=/data/dsgb10/bench/wave2
python3 $W/bench/bench_record_report.py --side kit $B/kit/k1.jsonl $B/kit/k2.jsonl $B/kit/k3.jsonl \
    --side ours $B/o1/s1/bench.jsonl $B/o1/s2/bench.jsonl $B/o1/s3/bench.jsonl \
                $B/o2/s1/bench.jsonl $B/o2/s2/bench.jsonl $B/o2/s3/bench.jsonl \
    --dsfront ~/dsgb10/prodbin/dsfront --tokenizer /data/models/DeepSeek-V4.1-Flash/tokenizer.json
python3 $W/bench/bench_record_report.py --side o1 $B/o1/s*/bench.jsonl --side o2 $B/o2/s*/bench.jsonl
python3 $W/bench/steps.py kit $B/kit/logs/dsv41.log $B/kit/k1.jsonl $B/kit/k2.jsonl $B/kit/k3.jsonl
python3 $W/bench/steps.py ours $B/o1/s1/run4.log $B/o1/s1/bench.jsonl $B/o1/s2/run4.log $B/o1/s2/bench.jsonl ...
```

- **Decode.** Per prompt type, publish the mean of the session means (each session the mean of its two
  requests), with the sessions' range and the spread of the requests. Publish the same on both sides,
  with the date and N.
  - The ratio's range comes from the extreme requests (the report prints both).
  - Ours is the mean of O1 and O2. Report O2/O1; above 1 % look for the cause before publishing.
  - The kit's sessions should agree within 1–2 % after the warm-up. If they do not, check `steps.py`'s
    conditions block (clients, running and queued requests) and the `boxes` lines.
- **TTFT.** Publish the median and range of the decode requests other than each session's first, on
  both sides, and each session's first apart (the report prints both).
- **Prefill.** Per length, the mean of three sessions and the spread, on both sides.
- **Agentic.** Turn 0, the mean of three; turns 1–5, the median and range of 15 values, with the new
  tokens per turn on each side (m6).
- **Accounting and texts.** Report the token accounting on both sides, which should be 100 % (a
  failure on the kit's side means its count differs from its text). Report whether each side's greedy
  texts repeat across sessions, and where the two sides' texts first part (the report prints the token
  index). Report the first chunk's tokens on each side (m5).
- **Steps.** Tokens and time per step on both sides (`steps.py`), as above.
- **Correctness.** Send the golden file and the GSM8K summary to the correctness review. These are not
  speed figures.

### Time with production down

| window | step | minutes |
|---|---|---|
| O1 | three fresh-engine sessions (engine start ~45 s, bench ~75 s, stop) and the service's restart | ~9 |
| K | the lend: the service stops its engine | < 1 |
| | kit start to Ready (cold official weights over NFS) | 6–10 (allowed 20) |
| | warm-up and three sessions, ~100–110 s each (the 2026-10-11 run took 100 s) | ~7 |
| | golden, 4 × 96 tokens | < 1 |
| | GSM8K harness, 100 problems at a few seconds each | 5–10 |
| | stop, checks, and the service's restart (~45 s) | ~2 |
| | the kit's window in all | **~22–31, at most ~40** |
| O2 | as O1 | ~9 |
| **day** | O1 + K + O2 | **~40–50, at most ~60** |

Without the GSM8K harness, K is ~17–21 min. Each window starts only once dsfront has been idle for
10 min, and the service lends no sooner than 10 min after an engine start. Submitted one at a time,
the windows therefore come at least ~10 min apart, with production serving in between. Queued
together, they run back to back while no request waits (the supervisor hands the ring to the next job
with no engine start in between): one stretch of ~40–50 min. A request that comes during a window
waits at dsfront's gate and is answered 503 after 600 s, since these jobs are not preempted. The
pre-window steps 1–5 do not stop production.
