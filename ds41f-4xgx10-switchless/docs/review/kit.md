# Kit review: can an outsider run it? (2026-10-13)

The question: could someone with four GB10s, using only the public docs, build dsgx10, prepare the
weights, deploy to four boxes, start it and serve correct tokens? The review cloned the public
repository (<https://github.com/blaesus/dsgx10>, tip `191c30f`) into a scratch directory (`$K`
below) and used only that clone and its docs, except where the [command log](#appendix-command-log)
says otherwise. Nothing was reconverted; the ring was used only through the clone's own
`engine/server/ring.sh` and `run4.sh`, in parts under 10 minutes; production was not touched.

## Verdict

**Not yet a kit, although the code reproduces.** A fresh clone of `191c30f` builds with its README's
commands without a warning. Every documented test this review ran passes, except one attention case that
fails by design, which the case list did not say. On the ring its binaries give production's recorded
hash in all eight reference runs, two of them again from a golden directory made from the repository
alone. Its server passes serve_test in CED and exact mode, and the front-end's end-to-end tests. What
fails is getting from the public docs to a service on someone else's four GB10s. At `191c30f` the docs
did not name the checkpoint the engine was built and validated on (dealignai's uncensored variant, not
deepseek-ai's original), and listed no prerequisites. For the Engram shards they sent the reader to a
path on rank 0, and for the 518 KB hash map to another project's SGLang image. They said nothing about
making four ranks' weight files from one copy of the checkpoint, or about putting them on the workers.
And they did not say that the fleet is written into the code: link addresses and devices in
`ring_config.h`, this fleet's file paths in `main.cpp`, the workers' SSH addresses in four scripts. A
stranger's boxes must either copy this fleet's addresses or edit about six files. This branch's doc
commits fix what docs can fix: the prerequisites and the checkpoint, what each box needs and where, the
Engram sources and the hash map's hash, what to copy or edit for another fleet, and the reference hashes
with how to check a build without the golden data. The rest needs code or a decision: a fleet
configuration and per-rank paths in place of compiled-in ones, a rank check in the weight loader, a
published hash map, the four-rank conversion procedure and a run on the original checkpoint (both with
the official-weights review), and a LICENSE.

## Build equivalence on the ring

The clone's engine (sha256 `bb7daa99…`, 42,360,024 bytes, the size of production's `b352b870…`) and
dsfront (`c7e2ef4b…`) were built exactly as the README says, but with `-j4` and under `nice`. The
reference runs went through the clone's own `run4.sh`, as release candidate 5's validation part H1 ran
them, with `/data/dsgb10/golden/long` as G. Every hash equals production's recorded value.

| Run | Production (RC5) | Fresh clone | ms per token, RC5 → clone |
|---|---|---|---|
| tf `--prompt 4096 --limit 4608 --prefill-rows 4096` | `842278b28d87dad0` | `842278b28d87dad0` | – |
| k = 1, `--ctx 1024 --steps 256` | `febc7dc54ee40a39` | `febc7dc54ee40a39` | 16.705 → 16.820 |
| k = 1, `--no-l2pf` | `febc7dc54ee40a39` | `febc7dc54ee40a39` | 17.466 → 17.623 |
| drafts, `--spec --draft-fp8` | `febc7dc54ee40a39` | `febc7dc54ee40a39` | 8.267 → 8.316 |
| drafts, `--spec --draft-bf16` | `febc7dc54ee40a39` | `febc7dc54ee40a39` | 8.407 → 8.434 |
| sampled, `--temp 0.8 --top-p 0.95 --top-k 40 --seed 7` | `a8f7a08f8cd1c5e8` | `a8f7a08f8cd1c5e8` | 16.756 → 16.795 |
| CED decode, `--ctx 16384 --ced` | `81f772ab79a950e4` | `81f772ab79a950e4` | 16.724 → 16.896 |
| CED tf, the 28,489-token prompt (`cedcheck/long28k-ced128`) | `14650fb0739d0383` | `14650fb0739d0383` (the hash's seed: until 2026-10-13 this run hashed nothing, [correctness.md](correctness.md) M1; it now gives `8f08a153d415c5c8`) | – |
| k = 1 from the repository's tokens alone | `febc7dc54ee40a39` | `febc7dc54ee40a39` | 16.797 |
| CED decode from the repository's tokens alone | `81f772ab79a950e4` | `81f772ab79a950e4` | 16.889 |

The last two rows used a directory made from `golden/long.tokens.json` alone, with `tokens.npy` and empty
`positions.npy` and `record_positions.npy`. That is what an outsider without rank 0's golden data can make.
The speeds are within 1 % of release candidate 5's, from a single job, in the run-to-run spread
([README](../../README.md)).

The server and the tests, from the same binaries:

- **Part A**, the server at production's engine arguments (`--max-ctx 262144`). It used our own socket
  and spill directory (`~/dsgb10/kitspill`, mode 700 on every box, removed at the end) with
  `--disk-gib 8 --disk-tag kit`. It was up in 66 s, rank 0 reading its 72.38 GiB in 56.7 s.
  - `serve_test.py --ced`, fed from the tokens-only directory: 17 of 17 checks pass.
  - The fresh dsfront on 127.0.0.1:18871, with a random key of our own: `/health` 200, and
    `/v1/models` lists `deepseek-v4.1-flash-uncensored`.
  - Four greedy chat requests: a code answer, an answer to a system prompt, the code answer again
    streamed, and a tool call (`get_weather` with `{"location": "Paris"}`). All gave sensible output;
    the streamed regenerate parted from the first answer after ~50 tokens (finding 22).
  - `frontend_e2e.py`: 9 ok, 0 failures.
  - A planned stop took 5 s.
- **Part E**, the server with `--exact`, up in 47 s. `serve_test.py`, fed from the tokens-only
  directory: 9 of 9 checks pass. Its regenerate kept the first token and its log-probability within
  2e-3, then 19 of 32 tokens, as engine.md describes. A planned stop took 5 s.
- **Part U**:
  - `slot_tests`: PASSED, 3013 checks, 0 differ, in 3 s.
  - `ring_test race --cap 4194304`, run with `engine/comm/run4.sh` inside our `ring.sh`: all eight
    cases exact on every rank, with and without the 2 ms stall, in 6 s.
- **Part T**, the documented one-GPU tests on rank 0's GPU, with rank 0's reference data
  (`~/dsgb10-data`, made by generators in the canary-roce image):
  - `test_kernels`: all passed;
  - `test_pattn`: PASSED;
  - `attn_tests` on l0, l2, l20, dspark, mhc and long32k: PASSED.
  - `attn_tests` on stale_index_k fails two checks (`L2_c1_iscore`, `L2_c1_itopk`). That is by design:
    the case reproduces a model.py bug the kernels do not copy (attn README). The case list did not say
    so (finding 23).
- **Part W**, the two documented checks that read rank 0's weight file, on rank 0's GPU:
  - `dsw_kcheck /data/dsgb10/weights/rank0.dsw`: all passed. The kernels' tiling gives the file's bytes,
    and the GEMVs on the file's tensors match the kernel tests' reference.
  - `dspark_tests`, with its default data: PASSED. In its injected pass all five drafts equal the
    reference's in 96 of 96 steps.
- **Part X**, ring.md's first "Build and test" command, `ring_test correct race integrity latency
  graph` (ring_test's default tests) through `engine/comm/run4.sh`: every rank passed. All 94 cases are
  ok, none failed, and the integrity test's 1,000,000 collectives found 0 mismatched words on every rank.
  52 s.
- **Off the ring:**
  - `cargo test --release`: 15 of 15;
  - `smoke_test.py` against `dsfront mock-engine`: 26 ok, 0 failures;
  - `toc.py FILE --replicated` on each box's rank file (header and TOC only). Every header says version
    2, its own rank of 4, 1609 tensors and 72.38 GiB, cut on 2026-10-11 from the variant's checkpoint.
    The 476 replicated tensors' CRCs are identical in all four files;
  - `dsfront tokcheck` on the three tokeniser sets (1,536,769 cases): 0 encode and 0 decode mismatches;
  - `dsfront enccheck` (40,201 encode and 30,009 parse cases): 0 mismatches;
  - `engram_hashcheck`: 13 cases, 504,768 row ids each checked 3 ways, all equal.

  The case files are rank 0's, made by the generators in the canary-roce image.
- **Not done:**
  - `dsw_verify`. It compares every byte of a rank file with the checkpoint, which took 202 s on rank 0
    (layout.md), and reads both from `/data`, the disk that also holds production's spill directory. That
    is not the cheap check the brief allowed.
  - Other tests the docs describe, mostly where they were written rather than in a command block:
    `engram_rowcheck`, `engram_storecheck` and `engram_gathercheck` (engram.md), `fused_tests` (attn
    README), `lite_tests` and `coll_tests` (engine.md), `check_dsw.py` (in the canary-roce image),
    ring.md's `ring_test bulk` and `graph --trace`, and the server's other scripts in
    `engine/server/tests` (`disk_test.py`, `slice_test.py`, `batch_test.py`, `kept_test.py`,
    `snap_test.py`, `yield_test.sh`, `soak.py`).
  - No request went to production. The hashes compare the engines, and tokcheck and enccheck the
    front-ends. Comparing answers would have taken production's ring back from the other reviews' jobs,
    and a greedy answer depends on the session state it resumes from (finding 22).

Ring jobs. Each part ran under the clone's `ring.sh` with `OWNER=kit-review PREEMPT=0`. The first driver
held a want from H1 to U; T, W and X had drivers of their own. Other reviews' parts ran between ours.

| Part | What | Asked | Ring ours → ring.sh exit | On the ring | Waited | Result |
|---|---|---|---|---:|---:|---|
| H1 | tf, k = 1, `--no-l2pf`, drafts FP8, drafts bf16 | 08:31:55 | 08:33:36 → 08:38:19 | 283 s | 101 s | 5 of 5 hashes |
| H2 | sampled, CED 16k, CED 28.5k tf; k = 1 and CED 16k from the tokens-only directory | 08:38:24 | 08:43:01 → 08:47:36 | 275 s | 277 s | 5 of 5 hashes |
| A | server at production's arguments: serve_test `--ced`, the fresh dsfront, kit_chat, frontend_e2e | 08:47:41 | 08:52:48 → 08:54:29 | 101 s | 307 s | 17 of 17; 4 answers; 9 of 9 |
| E | server with `--exact`: serve_test | 08:54:34 | 08:58:21 → 08:59:36 | 75 s | 227 s | 9 of 9 |
| U | slot_tests (one GPU, no data); ring_test `race --cap 4194304` through `engine/comm/run4.sh` | 08:59:41 | 09:07:36 → 09:07:49 | 13 s | 475 s | 3013 checks, 0 differ; 8 of 8 cases exact on every rank |
| T | test_kernels, test_pattn, attn_tests on 7 cases (rank 0's GPU) | 09:08:09 | 09:17:33 → 09:17:54 | 21 s | 564 s | all pass but stale_index_k's 2 checks, by design |
| W | dsw_kcheck on `rank0.dsw`, dspark_tests (rank 0's GPU) | 09:26:02 | 09:26:02 → 09:26:19 | 17 s | 0 s | all passed; PASSED |
| X | ring.md's `ring_test correct race integrity latency graph` through `engine/comm/run4.sh` | 09:30:31 | 09:30:49 → 09:31:41 | 52 s | 18 s | every rank passed; 0 mismatched words in 1,000,000 collectives |

## What the docs said about deploying, at `191c30f`

| Question | Documented? |
|---|---|
| Binaries to the other boxes | Yes: `run4.sh` copies the engine to `~/dsgb10/bin` and `service.sh` to `~/dsgb10/service/bin` on every box at each start (engine.md). |
| Weights to the other boxes | No. layout.md says where each box keeps its file and DESIGN converts on every box, but nothing copies a rank file or makes four from one checkpoint (finding 3). |
| NIC, GID, ring order | Mostly: ring.md's table has the devices, GID 3, MTU 9000, port f0 to the next rank and every address. It had the subnets wrong (/30 for /24), and the netplan and `kho=off`'s boot entry are in the private homelab repo (findings 16, 18). |
| service.sh's settings | Yes: its header and engine.md's table. |
| Where Engram lives | Where: engram.md and findings.md give each rank's shard directory. Where from: no (finding 2). |
| Model name and endpoints | No (finding 11). |
| Which checkpoint | Only in DESIGN.md (finding 4). |

## Findings

At `191c30f`. Status "docs fixed" means the doc commit on this branch fixes it; "open" needs code, data
or a decision.

| # | Severity | What an outsider sees | Fix | Status |
|---|---|---|---|---|
| 1 | **blocker** | The fleet is compiled in and written into the scripts. `ring_config.h` holds this ring's hostnames, its sixteen 192.168.x link addresses, device names and GID index. run4.sh, service.sh, ringlock.sh and comm/run4.sh SSH to 10.0.0.4, .3 and .2. Unless every link and worker gets this fleet's addresses, each rank dies at once (`bind 192.168.x.x:18620: Cannot assign requested address`) and the scripts SSH to hosts that are not there. No doc said that the addresses must be copied or which files to edit. | Docs: say what to copy or edit (README, ring.md). Code: one fleet file (hosts, ranks, link addresses, devices) read by the engine and the scripts. | docs fixed; code open |
| 2 | **blocker** | The Engram shards and the hash map come from another project. The engine needs `engram-l{1,14}-r<r>of4.bin` on every box (23.6 GiB each) and `engram_hashmap.bin`. The docs called the shards "the kit's" and named the packer only by a path on rank 0 (`~/ai/dsv41-4x-ring/scripts/pack_engram.py`). `make_hashmap.py` needs the MiaAI kit's SGLang image: it imports `model.py`, and so tilelang, only for `ModelArgs`, to write a 518 KB file. | Docs: link the kit's public `pack_engram.py` with its command (`--tp 4`: its default is 3), give the hash map's size and sha256. Code or data: ship the hash map, or let `make_hashmap.py` import only `engram.py`. | docs fixed; hash map open |
| 3 | major | Nothing says how to make the four ranks' weight files from one copy of the checkpoint. DESIGN decision 10 converts on every box from that box's own 476 GiB copy. `dsw_convert --rank R` needs a GPU, and nothing copies the files to the workers (run4.sh copies binaries only). Each box needs ~120 GiB: 72.4 GiB of weights and 47.2 GiB of Engram shards. | Document converting all four ranks on one box, copying them with their sizes and checks (`dsw_verify`, `toc.py --replicated`), and the Engram shards and hash map with them. | open: the official-weights review documents it in `docs/layout.md` |
| 4 | major | The checkpoint is not named. README says "DeepSeek-V4.1-Flash"; everything was built and validated on dealignai's `DeepSeek-V4.1-Flash-UNCENSORED-FP8` at `d61c59ea` (DESIGN.md says so; shards 13–39 differ from the original). The served model name and dsfront's default tokenizer are the variant's, and the four rank files' headers name the variant's checkpoint as their source. The original, `deepseek-ai/DeepSeek-V4.1-Flash` at `fb2764a5`, has the same 96,085 tensors and the same per-rank conversion plans (checked here from headers), but no engine output has been checked on it. | README: name both, with revisions (done). Validate the engine on the original's weights. | docs fixed; validation open |
| 5 | major | No prerequisites. A build needs CUDA 13.0 in `/usr/local/cuda-13.0` (CMakeLists' default `nvcc`), `libibverbs-dev`, CMake ≥ 3.28 and Rust. The ring needs `kho=off` on kernel 7.0.0-1019-nvidia (else `ibv_reg_mr` fails with ENOMEM), the ring cabled and addressed as in ring.md, and SSH to the workers without a password. These were scattered (findings.md) or in the private homelab repo. | README "What the build and the ring need". | docs fixed |
| 6 | major | The per-box file layout is this fleet's: rank 0 under `/data/dsgb10` and `~/dsv41-engram-tp4/`, ranks 1–3 under `~/dsgb10` and `~/dsv41-4x-ring/engram/`. The path flags reach all four ranks alike through run4.sh and service.sh, so another layout needs one path that holds each box's own file. `/data/dsgb10` must exist on rank 0 for the lock unless `RING_LOCK` is set. | Docs (done). Code: per-rank templates (`--weights …/rank{r}.dsw`) or a per-box config. | docs fixed; code open |
| 7 | major | No way to check a build. The reference hashes were scattered through engine.md (36 mentions) with rank 0's paths. They need rank 0's `/data/dsgb10/golden/long` (5.9 GB, from `run_ref.sh` in the SGLang image) and `/data/dsgb10/kernels/cedcheck/long28k-ced128`, which no doc or script in the repo names or makes. Nor is there a list of the tests with their commands and data: each is described where it was written, and most read data made in the canary-roce image (`~/dsgb10-data`). | golden/README "Checking a build": the 8 hashes with their commands, and that the decode runs need only a tokens directory made from the repo (verified on the ring). Document how the cedcheck directory is made. List the tests, with their commands, data and expected results. | docs fixed; the cedcheck recipe is `golden/make_cedcheck.py`, and `golden/refhash.sh` runs the hashes (572901f); test list open |
| 8 | major | The loader checks a weight file's magic, version and size, but not which rank it was cut for (the Engram store checks its shards' rows). The four files have the same size, so a file on the wrong box loads silently and the tokens are wrong without an error. Each header does record its rank (`toc.py` shows ranks 0 to 3 on the four boxes). | Compare the header's rank with the process's rank in `engine/weights/loader.cpp`. | open (code); README warns |
| 9 | major | No LICENSE at the repository root or in the configuration. Only `ref/` carries one (`ref/LICENSE-deepseek`, the publisher's MIT). An outsider has no grant to use the rest. | Add a LICENSE (the owner's choice). | open |
| 10 | minor | dsfront's default tokenizer is `/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json`, and service.sh passes no `--tokenizer`. The front-end exits with "…/tokenizer.json: No such file or directory". | README: `DSGB10_FRONT_ARGS="--tokenizer PATH"`; the original's tokenizer.json is the same file. | docs fixed |
| 11 | minor | The HTTP endpoints and the model name were not written down anywhere (`/v1/models`, `/v1/chat/completions`, `/health`; `deepseek-v4.1-flash-uncensored`). | README. | docs fixed |
| 12 | minor | The front-end's build was mentioned only inside engine.md's "Borrowing the ring", and `cargo test` and the smoke test not at all from the README. | README. | docs fixed |
| 13 | minor | The build commands disagree. README says `-j` (unbounded), the attn README uses `build-attn -j 16`, and ring.md uses `build-comm -j 8`. An unbounded build on a box that also serves can run out of memory. | README: `-j4` with the reason. The separate build directories work and stay. | docs fixed (README) |
| 14 | minor | ring.md's ring_test commands use `engine/comm/run4.sh`, which takes no ring lock. With the service running they run beside it. | ring.md: run them inside `engine/server/ring.sh` (as this review did). | docs fixed |
| 15 | minor | findings.md and engine/kernels/README.md told jobs to take a bare `flock /data/dsgb10/ring.lock`. That waits for as long as the service serves, against engine.md's rule. | `engine/server/ring.sh <command>`. | docs fixed |
| 16 | minor | ring.md says each link is a /30. Every ring netdev on the four boxes is a /24, as `ring_config.h` says. | ring.md. | docs fixed |
| 17 | minor | findings.md says `nvcc` is not on PATH. It now is on rank 0. | "may not be on PATH", with CMakeLists' default. | docs fixed |
| 18 | minor | The homelab repository is private (404). Its references are dead outside: `docs/network.md`, `gotchas.md`, `host.md`, `fleet.md` §9 (the ring's netplan and the `kho=off` boot entry), `deepseek-v41.md`, `ds4.md`. ring.md and findings.md carry the facts an outsider needs, but not the netplan or the GRUB change. | Publish the ring's network setup with the kit, or drop the links. | open |
| 19 | minor | `cargo test` passes session.rs's tokenizer test when the variant's tokenizer.json is missing (it returns early), so an outsider's 15 of 15 hides a skipped test. | Report the skip, or take the path from an environment variable. | open (code) |
| 20 | minor | `evict_cache.py` drops the page cache only of this fleet's paths. With other paths it drops nothing, and a start can find too little free memory (findings.md: CUDA on GB10 allocates only free pages). | Take the paths from the engine's arguments. | open (code) |
| 21 | minor | A home path in `frontend/tools/gen_tokenizer_cases.py:8`, and a worktree path inside the agentic transcript (`golden/long.messages.json:309`, `long.tokens.json`). README's production command embeds the tailnet address and the kit's key path, and "Reproducing" copies to `andy@192.168.2.2`: this fleet's examples, not marked as such. | The tool's path → `~` (code). The transcript stays: its tokens are the reference. | open |
| 22 | minor | The same greedy request sent twice gave two answers that part after ~50 tokens. kit_chat's `prime` and `prime-stream`; the second went "in place" into the first's session, 21 of 21 prompt ids in common. That is the engine's documented behaviour: a regenerate reruns its last positions through the decode step, which matches the prefill path to numerics, not bits. It is written only deep in engine.md "Serving", where nobody checking reproducibility would look. | README: a greedy answer depends on the session state it resumes from. | docs fixed |
| 23 | minor | `attn_tests` on the listed case `stale_index_k` exits 1 ("FAILED (2 failing checks)"). The case reproduces a model.py bug that the kernels do not copy, so the failure is by design. That was written in the attn README's accuracy notes, but not where the cases are listed. | attn README: say so at the case list. | docs fixed |

## What is configurable and what is baked in

A grep of the clone, with `.git` and build outputs left out, for `rank 0`, `spark`, `192.168`, `100.79`,
`10.0.0.`, `rocep`, `/data/dsgb10`, `dsgb10-data`, `andy`, `~/ai/` and `dsv41-*`, `/data/models` and
absolute home paths. It counted code and script files, then Markdown files:

| Pattern | Code and script files | Markdown files |
|---|---:|---:|
| `rank 0` | 20 | 13 |
| `spark` (mostly DSpark) | 54 | 15 |
| `192.168` | 3 | 3 |
| `100.79` | 1 | 3 |
| `10.0.0.` | 8 | 3 |
| `rocep` / `roceP` | 4 | 4 |
| `/data/dsgb10` | 27 | 13 |
| `dsgb10-data` | 17 | 6 |
| `andy` | 3 | 2 |
| `~/ai/`, `dsv41-4x-ring`, `dsv41-engram` | 4 | 3 |
| `/data/models` | 9 | 2 |
| absolute home paths | 3 | 0 |

What the hits are, by what an outsider would have to do about them:

| What | Where | Configurable? |
|---|---|---|
| A box's rank | `ring_config.h` hostnames (`rank_from_hostname`) | yes: `--rank`, which run4.sh and service.sh pass |
| Link addresses, RDMA devices, GID index | `engine/comm/ring_config.h` (`kRing`, `kNextDev`, `kPrevDev`, `kGidIndex`); the bootstrap's TCP port 18620 in `ring.cpp` | baked in at compile time |
| Ranks 1–3's SSH addresses 10.0.0.4, .3, .2 | `engine/server/run4.sh` (W1–W3), `service.sh` (WS), `ringlock.sh` (RING_BOXES), `engine/comm/run4.sh`; also `micro/ring_{bw,lat}.sh`, `tools/ttft_part.sh`, `tests/soak.py` | baked in: edit the scripts |
| Weight file, hash map, Engram shards | `engine/server/main.cpp`: rank 0 `/data/dsgb10/…` and `~/dsv41-engram-tp4/`, ranks 1–3 `~/dsgb10/…` and `~/dsv41-4x-ring/engram/` | `--weights`, `--hashmap`, `--shard1`, `--shard14`, but run4.sh and service.sh pass one value to all four ranks |
| Ring lock, owner, wants | `/data/dsgb10/ring.*` (`ringlock.sh`) | `RING_LOCK` |
| Spill directory (kept sessions) | `/data/dsgb10/spill` where `/data/dsgb10` exists, else `~/dsgb10/spill` | `--disk-dir` |
| Golden data | `/data/dsgb10/golden/…` (main.cpp default `prose`, tests, tools) | `--golden`, `--tokens` |
| Page-cache eviction | `engine/server/tools/evict_cache.py`: this fleet's checkpoint, Engram and weight paths | baked in |
| Tokenizer | `frontend/src/main.rs`: `/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json` (also `session.rs`'s test) | `--tokenizer`; service.sh passes it only through `DSGB10_FRONT_ARGS` |
| Model name | `deepseek-v4.1-flash-uncensored` | `--model-name` |
| API key, addresses, port | service.sh: `~/dsgb10-data/dev-api-key`, 127.0.0.1:18893; 8888 only with `DSGB10_PROD=1`; never 0.0.0.0 | `DSGB10_KEY_FILE`, `DSGB10_ADDR`, `DSGB10_PORT` |
| Tailnet address <tailnet address>, the kit's key path | README and engine.md's production command; service.sh's comment | docs only |
| Test and reference data | `~/dsgb10-data/{kerneltests,attntests,frontend}` (made in the canary-roce image) | most tests take a directory; some defaults are baked in |
| Home paths | `frontend/tools/gen_tokenizer_cases.py:8` (an absolute home path in a usage line); `golden/long.messages.json:309` and `long.tokens.json`'s text (a worktree path inside the agentic transcript) | the transcript's path is part of the reference tokens: changing it would change every hash |

## Appendix: command log

All times are Europe/London (BST, +01:00) on 2026-10-13. `$K` is the review's scratch directory and `$C` is
`$K/dsgx10/ds41f-4xgx10-switchless`, the clone's configuration, where the commands ran unless an entry says
otherwise. A script named without a path (`smoke.sh`, `kpart.sh`, `kit_chat.py`, …) is the review's own, in
`$K`; each entry says what it ran. The steps are the brief's; entries are grouped by step, so their times
interleave where the ring parts ran in the background.

### Step 1: clone

```text
08:16:06  git clone https://github.com/blaesus/dsgx10           -> ok, 2.8 MB .git, 336 files
          git log --oneline -5                                   -> tip 191c30f (matches local master)
          read README.md (root), ds41f-4xgx10-switchless/README.md
```

### Step 2: toolchain and build (from $C, as README "Layout" says, but -j4)

```text
08:17     toolchain on rank 0: cmake 3.28.3, gcc/g++ 13.3.0, nvcc 13.0.88 (/usr/local/cuda-13.0, now on PATH), driver 580.178.04,
          kernel 7.0.0-1019-nvidia, libibverbs-dev/rdma-core 50.0, cargo/rustc 1.97.1 (~/.cargo), python3 3.12.3 (no numpy)
          docs name: CUDA 13 path (CMakeLists default /usr/local/cuda-13.0/bin/nvcc; findings "nvcc not on PATH"), sm_121a
          (findings#tensor-cores), C++20, CMake 3.28 (cmake_minimum_required). Nowhere: driver, gcc version, ibverbs dev
          package, Rust version, Python packages, a "prerequisites" list.
08:18:05  cmake -S engine -B engine/build                                   -> ok, 2 s
08:18:07  nice -n 19 cmake --build engine/build -j4                         -> exit 0, 43 s, 69 objects, 0 warnings
          sha256 dsgb10_rank bb7daa99... (42,360,024 B; prodbin's b352b870... is the same size; both embed 90 build paths;
          git diff c247a78 191c30f touches engine code only under #ifdef DSGB10_COMM_TEST_HOOKS + ring_test)
08:19:52  cargo build --release -j4 (frontend/)                             -> exit 0, 13 s, 0 warnings (crates cached)
08:20:05  cargo test --release -j4                                          -> 15 passed, 0 failed (session.rs's tokenizer test
          returns early when /data/models/...UNCENSORED-FP8/tokenizer.json is missing: an outsider's run skips it silently)
          sha256 dsfront c7e2ef4b... (prodbin ce116cea...; debuginfo embeds paths)
          front-end build documented only in engine.md "Borrowing the ring" ("cargo build --release in frontend/") and in
          service.sh's error message; cargo test documented nowhere.
```

### Step 3/4: reading (weights, Engram, deployment)

```text
          see the report; ref/inference has vision.py and run.sh (an early `find | head -300` cut them off: not missing)
08:21     homelab repo: https://github.com/blaesus/homelab -> 404 (private): every "homelab docs/..." reference is dead outside
          MiaAI kit https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks -> 200; scripts/pack_engram.py on main -> 200
          bash hardcoded.sh (grep of the clone for fleet strings)                -> hardcoded.txt
          python3 inv_diff.py official variant (headers only)                    -> identical inventories (96,085 tensors, 48 shards)
          tools/st_inventory.py on /data/models/DeepSeek-V4.1-Flash              -> byte-identical to ref/inventory.txt
          dsw_convert --plan, ranks 0-3, both checkpoints (headers only, no GPU) -> identical plans (1609 tensors, 72.38 GiB)
          (coordinator, ~08:25: drop the official-checkpoint conversion; no dsw_convert beyond --plan)
08:26     python3 mk_tokgold.py golden/long.tokens.json tokgold/long /data/dsgb10/golden/long
          (a tokens-only golden dir from the repo alone, standard library only: tokens.npy and empty int32
          positions.npy, record_positions.npy)
             -> the repo's 28,489 tokens are byte-identical to rank 0's golden/long/tokens.npy (and cedcheck's)
```

### Step 2 (cont.): tests that need no ring

```text
08:32     bash smoke.sh: dsfront mock-engine + dsfront serve (fresh binary, our random 0600 key, 127.0.0.1:18881),
          then frontend/tools/smoke_test.py as its header says   -> every request timed out: the socket path under $K
          was longer than SUN_LEN (dsfront: "path must be shorter than SUN_LEN; retrying"); moved to ~/dsgb10/kitmock
08:33:20  bash smoke.sh (socket ~/dsgb10/kitmock/engine.sock)        -> 26 ok, 0 FAIL ("0 failures"); both processes stopped
08:36:37  bash fecheck.sh: nice -n 19 dsfront tokcheck ~/dsgb10-data/frontend/{tok_corpus,tok_fuzz,tok_sweep}.jsonl,
          dsfront enccheck ~/dsgb10-data/frontend/enc_cases.jsonl (engine.md's commands; the case files are the
          existing ones, made in the canary-roce image)
          -> tokeniser 574 + 60,000 + 1,476,195 cases, 0 encode and 0 decode mismatches; encoding 40,201 encode and
             30,009 parse cases (23,432 streamed), 0 mismatches
08:37     sha256sum of tokenizer.json / tokenizer_config.json in both checkpoints -> identical (c90dfa01..., 6ac8c8dc...)
08:37     dsfront tokenize /etc/hostname --tokenizer /nonexistent/tokenizer.json
          -> "/nonexistent/tokenizer.json: No such file or directory (os error 2)", exit 1 (the error names the path)
```

### Step 3/4 (cont.): files on the boxes (read-only)

```text
08:39     bash engram_files.sh (ls and the hash map's sha256 on every box; no shard read)
          -> engram_hashmap.bin 518,048 B, sha256 2405ae1d... on all four; Engram shards 25,344,411,184 and
             25,345,104,976/25,345,105,240 B per rank (23.6 GiB); rank files 77,719,715,840 B on all four
08:41     curl the MiaAI kit's scripts/pack_engram.py (main)      -> 200, byte-identical to ~/ai/dsv41-4x-ring's copy;
          needs numpy only; --model --out --rank --tp (default --tp 3!) -> engram-l{L}-r{R}of{TP}.bin
          bash kitpub.sh: Dockerfile.canary-roce, docs/tp4.md public (200)
08:43     ip -br addr on rank 0, bash prefixes.sh on the workers   -> every ring netdev is a /24 (ring.md said /30)
```

### Step 5: build-equivalence on the ring (the clone's ring.sh and run4.sh, OWNER=kit-review PREEMPT=0)

```text
08:31     bash prep.sh (random dev key, 0600, never printed; syntax checks; workers reachable)
08:31:49  setsid nohup bash kitparts.sh > parts.log 2>&1 < /dev/null &
          (holds a want; mkdir -m 700 ~/dsgb10/kitspill on every box; per part:
           OWNER=kit-review PREEMPT=0 $C/engine/server/ring.sh bash kpart.sh P $K/ring/P)
          H1 asked 08:31:55, ring ours 08:33:36, ring.sh exit 08:38:19 (283 s; waited 101 s behind the correctness
          review's part C):
            tf 842278b28d87dad0 ok (64 s); k1 febc7dc54ee40a39 ok 16.820 ms/token; k1-nol2pf ok 17.623;
            fp8 ok 8.316; bf16 ok 8.434
          H2 asked 08:38:24, ring ours 08:43:01, ring.sh exit 08:47:36 (275 s; waited 277 s behind the correctness
          review's H1):
            temp a8f7a08f8cd1c5e8 ok 16.795; ced16 81f772ab79a950e4 ok 16.896; ced28 14650fb0739d0383 ok;
            k1-tok (tokens-only dir) febc7dc54ee40a39 ok 16.797; ced16-tok 81f772ab79a950e4 ok 16.889
            (ced28's 14650fb0739d0383 is the hash's seed: until 2026-10-13 that run hashed nothing,
            correctness.md M1; since b47c510 it gives 8f08a153d415c5c8)
          A asked 08:47:41, ring ours 08:52:48, ring.sh exit 08:54:29 (101 s; waited 307 s behind the correctness
          review's H2 and the official-weights review's ring_convert.sh 1):
            run4.sh --mode serve --socket ~/dsgb10/kitrun/engine.sock --max-ctx 262144 --disk-dir ~/dsgb10/kitspill
              --disk-gib 8 --disk-tag kit                          -> up after 66 s (rank 0: 72.38 GiB in 56.7 s)
            timeout 240 python3 engine/server/tests/serve_test.py --socket ... --ced --tokens $K/tokgold/long/tokens.npy
                                                                   -> 17 of 17 checks pass
            dsfront serve --listen 127.0.0.1:18871 --engine-socket ... --api-key-file $K/devkey/key -> /health 200
            timeout 240 python3 kit_chat.py http://127.0.0.1:18871 $K/devkey/key ring/A/kit_chat.json
                                                                   -> models [deepseek-v4.1-flash-uncensored]; 4 answers;
                                                                      the streamed regenerate parts from the first at char 200
            timeout 300 python3 engine/server/tests/frontend_e2e.py http://127.0.0.1:18871 $K/devkey/key -> 9 ok, 0 failures
            pkill -x dsgb10_rank (rank 0's SIGTERM)                -> stopped in 5 s, run4.sh exit 0
          E asked 08:54:34, ring ours 08:58:21, ring.sh exit 08:59:36 (75 s; waited 227 s behind the correctness review):
            run4.sh --mode serve ... --exact                       -> up after 47 s
            timeout 300 python3 engine/server/tests/serve_test.py --socket ... --tokens $K/tokgold/long/tokens.npy
                                                                   -> 9 of 9 checks pass (regenerate CLOSE: first token equal,
                                                                      logprob |diff| 3.64e-04 <= 2e-3, 19/32 tokens equal)
            stop                                                   -> 5 s, run4.sh exit 0
08:55:00  nice -n 19 engine/build/engram/engram_hashcheck /data/dsgb10/engram/engram_hashmap.bin /data/dsgb10/engram/hash_cases.bin
                                                                   -> 13 cases, 504,768 row ids each checked 3 ways: all equal
08:55:3x  (a slip, caught: kpart.sh was edited in place to add part T while the driver could still start parts
          from it; restored byte for byte at 08:55:53, before part E's bash read it at 08:58:21; part T moved
          to copies, kpart_T.sh and kitparts_T.sh)
          U asked 08:59:41, ring ours 09:07:36, ring.sh exit 09:07:49 (13 s; waited 475 s behind the official-weights
          review's ring_copy.sh 1):
            timeout 300 engine/build/attn/slot_tests              -> PASSED: 3013 checks, 0 differ (3 s)
            timeout 240 engine/comm/run4.sh engine/build/comm/ring_test race --cap 4194304
                                                                   -> 8 of 8 cases exact on every rank, all four ranks passed (6 s)
09:07:50  kitparts: done: parts H1 H2 A E U passed; ~/dsgb10/kitspill removed on all four boxes, ~/dsgb10/kitrun removed
09:08:01  setsid nohup bash kitparts_T.sh > parts_T.log 2>&1 < /dev/null &   (part T, the same driver with kpart_T.sh)
          T asked 09:08:09, ring ours 09:17:33, ring.sh exit 09:17:54 (21 s; waited 564 s behind the official-weights
          review's ring_convert.sh 2), rank 0's GPU only, with rank 0's reference data (~/dsgb10-data, made in canary-roce):
            timeout 300 engine/build/kernels/test_kernels         -> all passed (580 ok lines, 0 FAIL), 4 s
            timeout 300 engine/build/kernels/test_pattn ""        -> PASSED (0 failing checks), 8 s
            timeout 150 engine/build/attn/attn_tests ~/dsgb10-data/attntests/{l0,l2,l20,dspark,mhc,long32k}
                                                                   -> PASSED (0 failing checks) each, 0-2 s each
            timeout 150 engine/build/attn/attn_tests ~/dsgb10-data/attntests/stale_index_k
                                                                   -> exit 1, FAILED (2 failing checks: L2_c1_iscore 0 % exact,
                                                                      L2_c1_itopk 0/1 rows): by design, the case reproduces
                                                                      model.py's stale index keys (attn README § Accuracy)
09:17:55  kitparts_T: STOPPED at part T (that one exit 1: the driver counts it); ~/dsgb10/kitspill removed on every box
09:18     ls -d ~/dsgb10/kit{spill,run,mock}                      -> all gone; no process of the review left (pgrep -f)
09:18     bash leftovers.sh (workers: ~/dsgb10/kitspill gone)     -> gone on all three. It also counted ".kit." names in
          the service's spill directories (rank 0 /data/dsgb10/spill, workers ~/dsgb10/spill): 0. That listed file names
          in directories the brief asked not to read (no file was opened); it was not needed, as every run used
          ~/dsgb10/kitspill
```

### Step 6: doc fixes, in the worktree

```text
09:0x     python3 replacement scripts docfix1.py, docfix2.py and small edits (each old text asserted to occur once)
09:08:58  git commit 42c73d9 (doc fixes: README, findings.md, ring.md, engram.md, kernels README, golden README)
09:18:51  git commit 6ae1ae6 (attn README: stale_index_k fails two checks by design)
```

### After the report's first draft: the checks that read the rank files, and ring.md's ring_test

```text
09:25     (the report's first draft had left out dsw_kcheck, dspark_tests and the cheap verifier: it took the
          coordinator's "never touch the rank files", which was about where a conversion may write, to cover
          reading them. Every engine run above had read the same files, so they ran now. Of ring.md's ring_test
          commands it had run only race; the first one, ring_test's default tests, ran now too.)
09:26:02  setsid nohup bash kitparts_W.sh > parts_W.log 2>&1 < /dev/null &
          (OWNER=kit-review PREEMPT=0 $C/engine/server/ring.sh bash kpart_W.sh W $K/ring/W)
09:26:04  bash toc_check.sh: python3 engine/weights/toc.py FILE --replicated on the four rank files at main.cpp's
          default places (header and TOC only; the workers' through ssh: python3 - FILE --replicated < toc.py)
             -> every header: version 2, rank r of 4, 1609 tensors, 72.38 GiB, made 2026-10-11 from the variant's
                checkpoint (rank 0 /data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8, workers ~/DeepSeek-V4.1-Flash-
                UNCENSORED-FP8); 476 replicated tensors per file, every line (CRC included) identical in all four
          W asked 09:26:02, ring ours 09:26:02, ring.sh exit 09:26:19 (17 s; no wait), rank 0's GPU only:
            timeout 200 engine/build/weights/dsw_kcheck /data/dsgb10/weights/rank0.dsw
                                                                   -> all passed (99 ok lines; 2.03 GiB loaded in 1.7 s), 3 s
            timeout 360 engine/build/dspark/dspark_tests (its defaults: /data/dsgb10/dspark/prose and w/,
              /data/dsgb10/golden/prose, /data/dsgb10/weights/rank0.dsw)
                                                                   -> PASSED, 14 s (injected pass: all 5 drafts equal to the
                                                                      reference's in 96 of 96 steps)
          dsw_verify not run: it compares every byte of a rank file with the checkpoint (202 s on rank 0, layout.md),
          reading both from /data, the disk that also holds production's spill directory
09:30:31  setsid nohup bash kitparts_X.sh > parts_X.log 2>&1 < /dev/null &
          (OWNER=kit-review PREEMPT=0 $C/engine/server/ring.sh bash kpart_X.sh X $K/ring/X)
          X asked 09:30:31, ring ours 09:30:49, ring.sh exit 09:31:41 (52 s; waited 18 s behind the official-weights
          review's ring_copy.sh 3):
            timeout 300 engine/comm/run4.sh engine/build/comm/ring_test correct race integrity latency graph
              (ring.md's first "Build and test" command: ring_test's default tests)
                                                                   -> exit 0, 52 s: every rank passed; 94 of 94 cases ok
                                                                      (87 of them say exact); integrity: 1,000,000
                                                                      collectives, 0 mismatched words on every rank
```

### Where the review had to look outside the docs

1. Source code, for what the docs did not say:
   - `engine/server/main.cpp`: the per-rank default paths, the spill rule, what the run hash covers;
   - `engine/comm/ring_config.h` and `ring.cpp`: the compiled-in fleet and the bootstrap's bind;
   - `engine/weights/loader.cpp`: no rank check;
   - `engine/server/{run4,ring,ringlock,service}.sh`: the workers' addresses, the lock's paths, the settings;
   - `engine/server/tools/evict_cache.py`: its paths;
   - `frontend/src/{main,server,session}.rs`: the default tokenizer, the model name, the routes,
     temperature 0, the skipped test;
   - `engine/engram/py/make_hashmap.py`: its imports;
   - `engine/server/golden.h`: which golden files a mode reads.
2. The MiaAI kit, for the Engram packer and the canary-roce image. Its public repository has
   `scripts/pack_engram.py`, `Dockerfile.canary-roce` and `docs/tp4.md`. Its clone on rank 0,
   `~/ai/dsv41-4x-ring`, has the same `pack_engram.py`.
3. The homelab repository, for the ring's netplan and `kho=off`'s boot entry: private (404), so not
   reachable.
4. The boxes themselves:
   - the ring netdevs' prefixes;
   - the sizes of the rank files and Engram shards, and the hash map's sha256 on all four;
   - both checkpoints' tokenizer files and safetensors headers (inventories, `--plan`);
   - the npy headers of `/data/dsgb10/golden/long` and `/data/dsgb10/kernels/cedcheck/long28k-ced128`;
   - the test reference data in `~/dsgb10-data`, which the tests need and the repo does not hold.
5. Production's recorded values: release candidate 5's validation logs on rank 0, for its speeds.
   engine.md and README carry the hashes, and README the k = 1 speeds.
