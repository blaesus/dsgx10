# Fleet: the boxes in a file (2026-10-13)

The [kit review](kit.md#findings) found the fleet compiled into the engine and written into the
scripts: the link addresses, devices and GID index in `ring_config.h`, each rank's files in
`main.cpp`, the workers' SSH addresses in four scripts. This work, by a one-off agent (fleet) on a
branch from master de274e7, moves all of it into one runtime file that the engine and the scripts read
alike, without changing what the engine computes: this project's four boxes are the committed default,
so nothing changes on them. It also adds per-rank paths, a rank check in the weight loader, a check
of every link's address before the ring starts, and tools of our own for the Engram shards and the
hash map.

## What changed

| Commit | What |
|---|---|
| d7264e5 | **The fleet file**, [`fleet/homelab.conf`](../../fleet/homelab.conf): `KEY=value` lines and `#` comments. `engine/comm/fleet.cpp` reads it for the engine (`dsgb10_rank --fleet FILE`, by default `fleet.conf` beside the binary), `engine/server/fleet.sh` for the scripts (`FLEET=FILE`, by default this one), with the same rules: an unknown key, a key twice or any other line is an error naming the file and line. `ring_config.h` declares the loaded fleet instead of the table. Before the bootstrap every rank checks that each of its four devices' GID at `GID_INDEX` is `::ffff:` and the address the file gives that link, and stops otherwise, naming the host, the device, the GID, the key and the file. engine/server/run4.sh, ring.sh and service.sh (through ringlock.sh) and engine/comm/run4.sh take the workers, directories and lock from the file, and both run4.sh and service.sh install it beside the engine on every box. `evict_cache.py` also drops the pages of the fleet's files for the box's rank (kit review minor 20). CMake copies the file beside `dsgb10_rank`, `ring_test` and `coll_tests` in the build tree. |
| b5a8854 | **Per-rank paths.** `{rank}` in `--weights`, `--hashmap`, `--shard1`, `--shard14`, `--disk-dir` and in the fleet's paths is the rank, and a leading `~/` the home directory of the box that reads it, so one set of arguments names each rank's own files (`--weights '~/models/rank{rank}.dsw'`). |
| 39b0b80 | **The loader's rank check.** `dsw::WeightFile(path, rank, world)` refuses a file whose header names another rank or world: `FILE: the file was cut for rank 1 of 4, but this process is rank 0 of 4 (a weight file serves only the rank it was cut for)`. The engine passes its rank; `dsw_load --rank R` does the same; `dsw_rank_test` tests it (kit review major 8). |
| fd97737 | `--fleet` expands a leading `~/` as the path flags do. |
| 007d3cd | **Engram tools of our own.** `engram_pack` (`engine/engram/tools/pack.cpp`) writes a rank's two shards from the checkpoint, the MiaAI kit's layout byte for byte; `engine/engram/py/make_hashmap_stdlib.py` writes the 518,048-byte hash map from the checkpoint's tokenizer and config with Python's standard library alone, and fails unless its sha256 is `2405ae1d…` (kit review blocker 2). NOTICE gives the algorithms' provenance: DeepSeek's `engram.py` (MIT) and numpy's SeedSequence, PCG64 and bounded draw (BSD-3-Clause). |
| eeb9e8b | master merged (the reviews, the official weights, the specifics pass, docs/operations.md). |
| 1af9493 | **The Engram read batch** as the per-rank key `RANK<r>_ENGRAM_READ_BATCH` (1 if absent) instead of `rank >= 2 ? 8 : 1` in `main.cpp`, which assumed the boxes with the coalescing SSD are ranks 2 and 3; `homelab.conf` gives 1, 1, 8, 8. The comments of the files it touched lost this fleet's particulars (box names where the drive model is what matters, how the boxes are reached, the kit's key path), as the [specifics pass](specifics.md) did for the docs. |
| c30048b | README "Running on your own four GB10s" (since replaced by "How to use"); ring.md, engine.md, engram.md, layout.md and operations.md where they described the compiled-in fleet or the missing checks; this page. |
| cb7da4a | master merged (4047770): the long28k CED reference hash fix (b47c510), golden/refhash.sh and golden/make_cedcheck.py. |
| 252a177 | **README "How to use"**, the owner's request: from four GB10 boxes to an OpenAI-compatible endpoint in nine steps, each command checked here but those it marks (below). It replaces "Running on your own four GB10s"; "Reproducing what exists" gains the commands of bench_record.py, the edit bench and the sparkDash driver, and the root README's row points at the section. |
| this one | this page: the How-to's checks, the merged build's hashes, the GID check's run. |

The default file reproduces what was compiled in: the same devices, GID index, hostnames and sixteen
link addresses as `ring_config.h` at de274e7, and for every rank the same default path strings as
`main.cpp` had, spill directories included. The strings matter beyond the files they name: a kept
session's run id hashes them.

**Each rank's own file, three ways.** The official weights' runs give every rank one path,
`~/dsgb10/weights-official/rank.dsw`, which on each box is a link to that box's rank file
([layout.md](../layout.md#one-path-on-every-box)). This work adds two ways without links: `{rank}` in
a path flag, where the files differ only by their number, and a fleet file whose `RANK<r>_WEIGHTS`
name each rank's file wherever it is. All three stay, and whichever is used, the loader now refuses a
file cut for another rank.

## The keys

[README § How to use](../../README.md#how-to-use), step 3, describes each key, and
[`fleet/homelab.conf`](../../fleet/homelab.conf) holds this project's values. In short:

| Key | Read by | What |
|---|---|---|
| `NEXT_DEV0`, `NEXT_DEV1`, `PREV_DEV0`, `PREV_DEV1` | engine | the RDMA devices facing rank r + 1 (port f0) and r − 1 (port f1), PCI domains 0 and 2, alike on every box |
| `GID_INDEX` | engine | the RoCE v2 GID index, alike on every device |
| `RANK<r>_NEXT_IP0`, `_NEXT_IP1`, `_PREV_IP0`, `_PREV_IP1` | engine | the rank's addresses on its two links, per PCI domain; `PREV_IP0` also carries the bootstrap |
| `RANK<r>_HOSTNAME`, `RANK<r>_NAME` | engine | the rank by hostname, when no `--rank` is given; the box's name in logs (optional) |
| `RANK<r>_WEIGHTS`, `_HASHMAP`, `_ENGRAM`, `_SPILL` | engine, scripts | the rank's weight file, hash map, shard directory and spill directory: the defaults of `--weights`, `--hashmap`, `--shard1`/`--shard14` and `--disk-dir`; the scripts drop their pages and `service.sh forget` clears the spill directories |
| `RANK<r>_ENGRAM_READ_BATCH` | engine | optional, 1 if absent: the Engram store's SSD reads in multiples of this |
| `RANK<r>_SSH` | scripts | how rank 0 reaches ranks 1–3 |
| `BIN_DIR`, `LOG_DIR`, `SERVICE_DIR`, `RING_LOCK` | scripts | the engine's and the logs' directories on every box, service.sh's directory, the ring lock on rank 0 |

A value has no spaces or quotes (letters, digits and `. _ - / ~ : , @ + % { }`), so the scripts read it
without evaluating it. The engine checks the keys it needs, and the scripts theirs; both accept the
other's.

## What is still this project's own

Nothing below changes a result.

- **In the engine.** `kSlowLink` in `engine/comm/ring_config.h` names L0, the 100G link: it routes the
  reduce-scatter's relays, the sums are in rank order whatever the route, and at `rs_slow_pct` 50 it
  decides only which way a quarter's rounding remainder goes. The bootstrap's TCP port, 18620
  (`comm::Options::port`). The golden modes' default `--golden` directory, `/data/dsgb10/golden/prose`.
- **In the front-end.** dsfront's default `--tokenizer`, this project's checkpoint's path.
- **In the scripts.** `service.sh`'s `DSGB10_KEY_FILE` default (`~/dsgb10-data/dev-api-key`) and its
  ports (18893 for tests, 8888 for production with `DSGB10_PROD=1`). `evict_cache.py`'s built-in
  globs (this project's checkpoint and shard paths), beside the fleet's files that run4.sh and
  service.sh pass it.
- **In test and measurement tools** that are not on the engine's path: `micro/ring_lat.sh` and
  `micro/ring_bw.sh` (the workers' and links' addresses), `engine/server/tools/ttft_part.sh` and
  `engine/server/tests/soak.py` (the workers' addresses), `golden/run_ref.sh` (the checkpoint's
  path), and the data directories of tests and tools (golden data and kernel-test references under
  `/data/dsgb10` and `~/dsgb10-data`).

`service.sh` reads `FLEET` from the environment of every command, not from `run/settings`, so `status`
and `stop` need the same `FLEET` as `start` (only the default file has been used with it).

**Deploying.** An engine built from these commits reads `fleet.conf` beside it, which only the
scripts of these commits install. Under an older `service.sh` or `run4.sh` it stops at its start
("cannot read the fleet file"), so a binary goes with its own commit's scripts. An older engine
ignores `fleet.conf`.

## Validation

Every GPU job went through the branch's own `engine/server/ring.sh` and `run4.sh`, `OWNER=fleet
PREEMPT=0`, in parts under 10 minutes, with the binary built from the branch and frozen with its
commit noted. Production was not touched.

**Off the ring.**

- **The parsers.** `homelab.conf` read by `fleet.cpp` gives exactly de274e7's table, and for every
  rank the same default path strings as `main.cpp` had and the read batches 1, 1, 8, 8 (a file without
  those keys: 1 everywhere); read by `fleet.sh`, the same workers, lock, directories and files. Both
  readers accept the same 6 variants (CRLF line ends, a comment after a value, tabs, no final newline,
  an indented comment, no read batch keys) and refuse the same 12 bad files, each with its file and
  line (an unknown key, a misspelt key, a misspelt read batch key, a repeated key, a space, quotes or
  `$` in a value, `#` with no blank before it, a lowercase or indented key, a line without `=`, a
  missing file). The engine alone refuses 8 more (no `GID_INDEX`, `GID_INDEX=abc`, a missing address,
  a missing device, read batches 0, 129, −8 and `8x`), and the scripts alone 3 (no `RANK3_SSH`,
  `BIN_DIR` or `RING_LOCK`). `{rank}` and `~/` expand alike in both.
- **The engine's arguments**, with no GPU visible: `dsgb10_rank` stops in its argument parsing, with
  a message naming the file and the key, for a missing fleet file, a misspelt key, a hostname the file
  does not have (without `--rank`), `--rank 5`, a file without `RANK0_WEIGHTS`, and serve mode with a
  file without `RANK0_SPILL`. Each rank's startup line names its fleet file, its weight file and its
  read batch (none with `--no-engram`).
- **The rank check.** `dsw_rank_test` passes on synthetic files of every rank, and on the four real
  rank files on their boxes: each opens as its own rank and is refused as each of the three others,
  with the file and both ranks in the message.
- **The Engram tools.** `engram_pack` against the shards in use, packed by the kit on all four boxes:
  every file's size (25,344,411,184 bytes for layer 1, 25,345,104,976 or 25,345,105,240 for layer 14),
  and the header and three windows of 4,096 records (start, middle, end) of both layers of every rank,
  packed from the official checkpoint, are identical, 24 of 24; so is a window of 1,000,000 records
  (four of the packer's steps). `make_hashmap_stdlib.py` writes the map in use byte for byte from the
  official checkpoint and from the variant, in 0.35 s.

**On the ring.** The reference hashes of [golden/README.md](../../golden/README.md#checking-a-build) on
the variant's weights at their default paths, `serve_test.py --ced`, a run with the fleet file copied
elsewhere and passed explicitly, and the rank check's negative test, in four parts, once with the
engine of fd97737 and again with that of 1af9493, which adds the read batch key:

| Part | Run | fd97737 (engine `4aa1632d…`) | 1af9493 (engine `eb7a53db…`) |
|---|---|---|---|
| H1 | teacher-forced | `842278b28d87dad0`, 63 s | `842278b28d87dad0`, 64 s |
| | decode, k = 1 | `febc7dc54ee40a39`, 55 s | `febc7dc54ee40a39`, 54 s |
| H2 | sampled | `a8f7a08f8cd1c5e8`, 52 s | `a8f7a08f8cd1c5e8`, 53 s |
| | CED, decode at 16k | `81f772ab79a950e4`, 56 s | `81f772ab79a950e4`, 56 s |
| | CED, the 28.5k prompt | `14650fb0739d0383` (the seed: below), 53 s | `14650fb0739d0383`, 53 s |
| S | a server at production's engine arguments (`--max-ctx 262144`; its own socket and spill directory, `--disk-gib 8`): `serve_test.py --ced` | 17 of 17; up in 47 s | 17 of 17; up in 48 s |
| X | decode, k = 1, with the fleet file elsewhere | `febc7dc54ee40a39`, 54 s | `febc7dc54ee40a39`, 55 s |
| | every rank given another rank's weights | 4 of 4 refused | 4 of 4 refused |
| | time on the ring: H1, H2, S, X | 118, 161, 73, 68 s | 118, 162, 73, 69 s |

Every hash is the reference's of the time; the decode runs took 16.71–16.90 ms a step. The 28.5k
prompt's, `14650fb0739d0383`, was the hash's seed until 2026-10-13: that run teacher-forces nothing,
and before b47c510 (merged here in cb7da4a) such a run hashed nothing, so in these two columns it
shows only that the run finished. The merged build gives the run's own hash, `8f08a153d415c5c8` on
these files ([below](#the-how-to-checked-2026-10-13)). The parts waited from nothing to 5.5 minutes for
the ring, which the service and two other agents' jobs shared.

- **Elsewhere and explicit:** `FLEET=~/dsgb10/fleettest/copy.conf` for the scripts and `--fleet` with
  the same path for the engine: every rank's startup line names that file (with 1af9493 also its read
  batch, 1, 1, 8, 8), and each worker's `fleet.conf` beside the engine is that file, as run4.sh
  installed it.
- **Another rank's weights:** `--weights '~/dsgb10/fleettest/wrong{rank}.dsw'`, where each box's file
  is a small one that says rank r + 1 mod 4: every rank stopped at its weight file with
  `…/wrong0.dsw: the file was cut for rank 1 of 4, but this process is rank 0 of 4 (a weight file
  serves only the rank it was cut for)` and its like on ranks 1–3, and run4.sh exited 134 within 9 s,
  in both runs.
- **The GID check and the ring test** (1af9493's engine `eb7a53db…` and ring_test `d18c10fc…`; 68 s on
  the ring): ring_test's defaults through engine/comm/run4.sh passed on all four ranks in 52 s, its
  integrity test's 1,000,000 collectives with 0 mismatched words. The default fleet file with
  `GID_INDEX=1`, given to run4.sh, stopped every rank at its first device before the bootstrap, and
  run4.sh exited 134 after 9 s: `comm: on <hostname>, rocep1s0f0's GID 1 is fe80::…, but the fleet file
  …/fleet.conf has RANK<r>_NEXT_IP0=<address>: this box's link is not addressed as the fleet file says,
  or this is not that rank's box`. No rank was left running, and the default file went back beside the
  engine on every box.

### The How-to, checked (2026-10-13)

The README's [How to use](../../README.md#how-to-use) ran here as written, with other paths, in seven
ring parts (`OWNER=fleet PREEMPT=0`) and on the build box. The builds of step 2, from fresh `git
archive` exports of 1af9493 and of cb7da4a: cmake 1.5–1.6 s, the engine 53.2–54.0 s at `-j4` with no
warning; cargo 13.1 s for 50 crates at `-j4`.

| Part | Binaries | What ran | Result | On the ring |
|---|---|---|---|---|
| U1 | engine `fd5fb085…`, a fresh export of 1af9493 | step 8's runs on the official files, named by a fleet file (tf, k1, ced16) and by `--weights '~/…/official{rank}.dsw'` (temp) | tf `c7605e0b6ba8d9d2` 64 s, k1 `7941647b162df1a8` 55 s, temp `02ef3e0858d7eb45` 52 s, ced16 `8790ee9388f4abd8` 57 s | 230 s |
| U2 | the same; dsfront `f0185c8d…` | k1 on a directory of the transcript's tokens alone; a server on the official files and `serve_test.py --ced`; dsfront on a test port with a throwaway key: API checks, step 6's curl lines, pi | `7941647b162df1a8` 54 s; up in 48 s; 17 of 17 in 17 s; 8 of 8 API checks (the model list, 401 without the key, a chat, a stream, a tool call, thinking by `reasoning_effort` and by `chat_template_kwargs`, a second turn reusing 2,283 of its 2,295 prompt tokens); the curl lines' answers; pi 0.84.3 answered; stopped in 6 s | 131 s |
| U3 | the tools of U1's export | steps 4 and 5 for rank 2, on its own box from that box's copy of the checkpoint (NVMe) | `--plan` 0.5 s; convert 22.9 s; verify 39.2 s, 0 bad; the TOC that of the file in use; `dsw_load --verify-crc --rank 2` 8.4 s + 4.4 s; `engram_pack` 62.8 s, both shards' sha256 those of the shards in use; the hash map 0.3 s, the map in use | 161 s |
| R1 | engine `8a6d2072…`, a fresh export of cb7da4a (master merged) | step 8 as written: `make_cedcheck.py --tokens-only`, then `EXPECT=official golden/refhash.sh tf k1 nol2pf fp8`, the official files named by a fleet file | `c7605e0b6ba8d9d2` 63 s; `7941647b162df1a8` 54, 54 and 53 s | 224 s |
| R2 | the same | `… refhash.sh bf16 temp ced16 ced28 -- --weights '~/…/official{rank}.dsw'` | `7941647b162df1a8` 53 s, `02ef3e0858d7eb45` 53 s, `8790ee9388f4abd8` 56 s, `de99bfce6ea96ba8` 53 s | 216 s |
| R3 | the same | `refhash.sh ced28` on the variant's files (the default fleet file) | `8f08a153d415c5c8` 52 s | 52 s |
| V | the same; ring_test `d4d8da85…` | step 3's ring test; step 8's sessions with the official files, then step 6's curl lines and step 7's pi (its provider taken from bench/piturn/models.json) against dsfront on a test port; step 8's `run4.sh stop` | 4 of 4 ranks passed in 54 s; listening after 48 s; 17 of 17 in 17 s; the curl lines' answers; pi answered; stopped in 5 s | 129 s |

Every hash is the reference's: the official checkpoint's in engine.md and refhash.sh, and for R3 the
variant's in [golden/README.md](../../golden/README.md#checking-a-build). The parts' waits for the ring
were up to 4 minutes. Not run on this fleet: setting `kho=off` (all four boxes boot with it), the copy
of the weight files to their boxes ([layout.md](../layout.md#converted-verified-copied-loaded) did it),
`service.sh` (production's service owns its directory here) and the benches.

## The kit review's findings

| # | Finding | Now |
|---|---|---|
| 1 | blocker: the fleet compiled in and written into the scripts | closed: the fleet file |
| 2 | blocker: the Engram shards and the hash map from another project | closed: `engram_pack` and `make_hashmap_stdlib.py`, byte for byte the files in use |
| 6 | major: this fleet's per-box file layout | closed: the fleet's per-rank paths, and `{rank}` in the path flags |
| 8 | major: the loader does not check a file's rank | closed: it refuses another rank's file |
| 20 | minor: `evict_cache.py` drops only this fleet's paths | closed: run4.sh and service.sh pass each rank's files from the fleet |
