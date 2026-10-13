# Specifics: a pass over the public documentation (2026-10-13)

The owner asked: "Remove unnecessary specifics from the documentation (e.g. that we have two ASUS ascent and
two DGX Sparks - they are all just GB10s)". This is that pass, made by a one-off agent at word and sentence
level. No number, code, script, file name, command path or fenced block changed, and no heading, so no anchor
moved. The checks are under [Check](#check).

*Later the same day the owner asked for the box labels to go too, with the hardware named generically: the
docs now say rank 0–3 and state once that the boxes are two ASUS Ascent GX10 (ranks 0 and 1) and two NVIDIA DGX
Spark (ranks 2 and 3). That pass replaced the labels throughout the docs, this report's quotations included, so
the "before" texts below show rank numbers where the labels stood.*

## Rules applied

- **The boxes are GB10 boxes.** The product names that told them apart (ASUS Ascent GX10, DGX Spark, "the
  Sparks", "the gx10s") are gone. The boxes' own labels stayed in this pass (they went in the later one, above),
  introduced once in the configuration README. Where a measurement differs between boxes because their SSDs differ, the boxes
  are named, or the drive models from engram.md's table.
- **Other people's GB10s are GB10s too** ("four Sparks", "2 Sparks", "on Spark", "one Spark"). Repository
  names and URLs that contain "DGX-Spark" are unchanged.
- **The homelab** and its private repository (`homelab/docs/*.md`, "homelab 4a73014") are gone. A figure that
  came from there stays, marked as measured outside this project.
- **The owner's devices and network.** The Mac's tunnel is gone, and the tailnet address is "rank 0's
  private-network address". The placeholder `<tailnet address>` survives only in the production command, a
  fenced block in docs/operations.md.
- **"The user"** is now "the owner" where a decision was taken (the go-ahead for production, development
  first, the precision rule, B3 parked, the release), "pi" where it meant whose requests are sampled at
  dsfront's defaults, and "requests" where it meant a client of the service. Generic uses ("a user", "users
  get 503s", user turns) stay.
- **Disk-free figures** and "`/` has too little space" are gone. The USB4 SSD stays wherever it explains a
  speed: rank 0's conversion, loading, spills and read-backs.
- **The kit's local paths** (`~/ai/dsv41-4x-ring`, its key file) are gone from prose. The shard directories
  the engine reads by default (`~/dsv41-engram-tp4/`, `~/dsv41-4x-ring/engram/`) stay.
- **Operations** (the pinned worktree, the binaries' directories, restart rules, bringing the kit back,
  "ready in ~6 min") moved to docs/operations.md, every sentence kept.

## The runbook's move (the first commit)

The configuration README's paragraphs after its Status table are now [docs/operations.md](../operations.md),
as the docs review proposed ([docs.md](docs.md#left-for-the-orchestrator-with-proposed-text), item 5): what
production runs and how that build was validated, the production command, the binaries' directories,
restarting around held wants, and bringing the SGLang kit back. The README keeps the reviewer's one line and
its link. Every sentence moved unchanged except for two references that had to follow it: "(dealignai's
variant, as above)" now links the README, and the ring link is relative to docs/. engine.md's "README.md has
the command" now names operations.md. The second commit changed only operations.md's two "the user"
sentences (edits 8 and 9).

## Edits (the second commit)

65 edits: README.md (root) 1, README.md 6, docs/operations.md 2, DESIGN.md 8, docs/findings.md 14, docs/layout.md 3, docs/engram.md 2, docs/engine.md 17, docs/batching.md 3, docs/experiments.md 8, golden/README.md 1.

| # | File | Before | After | Why |
|---:|---|---|---|---|
| 1 | README.md (root) | From-scratch inference engines for the homelab's GB10 boxes (two ASUS Ascent GX10s and two DGX Sparks). Each directory | From-scratch inference engines for NVIDIA GB10 boxes. Each directory | the homelab; the boxes' product names (they are all GB10s) |
| 2 | README.md | on the homelab's **four GB10 boxes** | on **four GB10 boxes** | the homelab |
| 3 | README.md | **Terms used below.** *rank 0* and *rank 1* are ASUS Ascent GX10s, *rank 3* and *rank 2* DGX Sparks; ranks 0–3 are rank 0, rank 1, ranks 2 and 3, and *the ring* | **Terms used below.** The four boxes are named *rank 0*, *rank 1*, *rank 2* and *rank 3* here (ranks 0–3), and *the ring* | the boxes' product names; the box names introduced once as names and ranks |
| 4 | README.md | *The user* is the homelab's owner, who decides what ships. | *The owner* owns the boxes and decides what ships. | "the user" as a character: the owner, who decides |
| 5 | README.md | parked, as the user runs several agents only occasionally | parked, as the owner runs several agents only occasionally | "the user": the owner (B3 parked on the owner's use) |
| 6 | README.md | **The draft gate under the user's sampling (T 1.0, top-p 0.95) | **The draft gate under pi's sampling (T 1.0, top-p 0.95) | "the user's sampling": what pi's requests get |
| 7 | README.md | on rank 0 under `/data/dsgb10/` (the root filesystem has only 88 G free), on the other boxes | on rank 0 under `/data/dsgb10/`, on the other boxes | disk-free trivia |
| 8 | docs/operations.md | (M9, the user's go-ahead), in | (M9, the owner's go-ahead), in | "the user's go-ahead": the owner's (a decision) |
| 9 | docs/operations.md | The user put development first (2026-10-12) | The owner put development first (2026-10-12) | "the user put development first": the owner (a decision) |
| 10 | DESIGN.md | on rank 0:8888, so pi and the tunnel need no change. | on rank 0:8888, so pi needs no change. | the owner's tunnel |
| 11 | DESIGN.md | (the user, 2026-10-12: "i want the precision | (the owner, 2026-10-12: "i want the precision | "the user": the owner (a decision) |
| 12 | DESIGN.md | \| M9 \| Ship \| `bench/bench.py` against both stacks; homelab docs updated \| | \| M9 \| Ship \| `bench/bench.py` against both stacks \| | the homelab's private documentation |
| 13 | DESIGN.md | 127.0.0.1:8888 (what the Mac's tunnel reaches) and, since 06:48, its tailnet address &lt;tailnet address>:8888, by the user's go-ahead: `service.sh` from a pinned worktree (`~/dsgb10/prod`), the kit's key and model id | 127.0.0.1:8888 and, since 06:48, its private-network address, by the owner's go-ahead: `service.sh` with the kit's key and model id | the Mac's tunnel, the tailnet address, "the user", the pinned worktree (now in operations.md) |
| 14 | DESIGN.md | (production since 08:16, [engine.md]\(docs/engine.md#borrowing-the-ring)); the homelab docs are updated (homelab 4a73014). Since 11:14 | (production since 08:16, [engine.md]\(docs/engine.md#borrowing-the-ring)). Since 11:14 | the homelab repository and its commit |
| 15 | DESIGN.md | M9 took the tailnet address and :8888 (2026-10-12); whether a systemd unit should start it after a reboot is still the user's call. | M9 took :8888 (2026-10-12); whether a systemd unit should start it after a reboot is still the owner's call. | the tailnet address; "the user's call": the owner's |
| 16 | DESIGN.md | it is not built, as the user runs several agents at | it is not built, as the owner runs several agents at | "the user": the owner (B3 parked on the owner's use) |
| 17 | DESIGN.md | dsgb10 in [README.md]\(README.md#where-it-stands-2026-10-13)), and the homelab docs updated (models, ports, how to switch back). | dsgb10 in [README.md]\(README.md#where-it-stands-2026-10-13)); how to switch back is in [docs/operations.md]\(docs/operations.md). | the homelab's private documentation; how to switch back is now in operations.md |
| 18 | docs/findings.md | `~/ai/dsv41-4x-ring` @ `cad252b`: SGLang | The kit @ `cad252b`: SGLang | the kit's local checkout path |
| 19 | docs/findings.md | The homelab's own record of the kit on the ring (`homelab/docs/deepseek-v41.md`): 53–55 prose and | An earlier record of the kit on the ring, from outside this project: 53–55 prose and | the homelab and its private document; the figures and their source kept |
| 20 | docs/findings.md | On one Spark it manages single-digit tok/s, and it wedged rank 0 on load before (`homelab/docs/ds4.md`). | On one GB10 it manages single-digit tok/s, and it wedged rank 0 on load before. | product name; the homelab's private document |
| 21 | docs/findings.md | On four Sparks in a cable loop | On four GB10s in a cable loop | product name (a third party's GB10s) |
| 22 | docs/findings.md | errors on Spark ([forum] | errors on GB10 ([forum] | product name |
| 23 | docs/findings.md | \| V4 Flash on 2 Sparks ([forum] | \| V4 Flash on 2 GB10s ([forum] | product name (a third party's GB10s) |
| 24 | docs/findings.md | so ~86 % is achieved. `homelab/docs/fleet.md` states "546 GB/s theoretical", which is twice the real figure. This is the number that sets the decode ceiling. | so ~86 % is achieved. This is the number that sets the decode ceiling. | a correction of the homelab's private document |
| 25 | docs/findings.md | reach ~185 Gb/s (homelab measurement). | reach ~185 Gb/s (measured on these boxes outside this project). | "homelab measurement": its provenance kept |
| 26 | docs/findings.md | (`homelab/docs/fleet.md` §9; rank 0's `/proc/cmdline` confirms) | (rank 0's `/proc/cmdline` confirms) | the homelab's private document (rank 0's /proc/cmdline is the evidence) |
| 27 | docs/findings.md | \| `…r3of4.bin` \| \| Free disk \| `/` 88 G, `/data` 308 G \| 250 G \| 1.4 T \| 903 G \| | \| `…r3of4.bin` \| | the fleet table's free-disk row |
| 28 | docs/findings.md | /scripts/pack_engram.py), here `~/ai/dsv41-4x-ring/scripts/pack_engram.py`): a 4096-byte header | /scripts/pack_engram.py)): a 4096-byte header | the kit's local path (the script is linked) |
| 29 | docs/findings.md | costs 1.1 ms (gx10s) to 9.5 ms (Sparks), | costs 1.1 ms (rank 0, rank 1) to 9.5 ms (ranks 2, 3), | product names; the boxes named instead |
| 30 | docs/findings.md | ranks 2 and 3's Samsung drives hold completions until | rank 2's and rank 3's Samsung drives hold completions until | product name; the boxes named instead |
| 31 | docs/findings.md | - The kit's API key is `~/ai/dsv41-4x-ring/state-tp4/api-key` (mode 600); pass it by file, never | - The kit's API key is in a file of mode 600; pass it by file, never | the kit's key path |
| 32 | docs/layout.md | (USB drive; `/` has too little space) | (an external USB4 SSD) | disk-space trivia; the drive kept, as it explains rank 0's conversion and load times |
| 33 | docs/layout.md | and on ranks 2 and 3 conversion takes no longer | and on ranks 2 and 3 conversion takes no longer | product name; the boxes named instead |
| 34 | docs/layout.md | reads the checkpoint from the USB drive and writes | reads the checkpoint from the USB4 SSD and writes | the drive named as above |
| 35 | docs/engram.md | costs 1.1 ms on the gx10s' drives and 9.5 ms on ranks 2 and 3's. After 2 s it costs 10–20 ms. The gx10s' drive also slows | costs 1.1 ms on the ESL01TBTLCZ (rank 0, rank 1) and 9.5 ms on the PM9C1a (ranks 2, 3). After 2 s it costs 10–20 ms. The ESL01TBTLCZ also slows | product names; the drives (from the table above) and their boxes named instead |
| 36 | docs/engram.md | 3. **Interrupt coalescing on ranks 2 and 3's PM9C1a.** | 3. **Interrupt coalescing on the PM9C1a (ranks 2, 3).** | product name; the boxes named instead |
| 37 | docs/engine.md | rank 1 at 5.0 GB/s, ranks 2 and 3 at 9.4–9.6. | rank 1 at 5.0 GB/s, ranks 2 and 3 at 9.4–9.6. | product name; the boxes named instead |
| 38 | docs/engine.md | and stopped by hand like the homelab's other model servers, and not brought back | and stopped by hand, and not brought back | the homelab |
| 39 | docs/engine.md | (`"127.0.0.1 &lt;tailnet address>"`: loopback for the Mac's tunnel, and rank 0's tailnet address) | (`"127.0.0.1 &lt;address>"` in production: loopback and rank 0's private-network address) | the Mac's tunnel and the tailnet |
| 40 | docs/engine.md | it unkillable by both OOM killers (homelab `docs/gotchas.md`). | it unkillable by both OOM killers. | the homelab's private document |
| 41 | docs/engine.md | The script refuses `0.0.0.0` (HTTP is tailnet-only: homelab `docs/network.md`), and dsfront refuses | The script refuses `0.0.0.0`, and dsfront refuses | the tailnet; the homelab's private document |
| 42 | docs/engine.md | (M9, the user's go-ahead on 2026-10-12) | (M9, the owner's go-ahead on 2026-10-12) | "the user's go-ahead": the owner's (a decision) |
| 43 | docs/engine.md | (since 06:48 on rank 0's tailnet address too) | (since 06:48 on rank 0's private-network address too) | the tailnet |
| 44 | docs/engine.md | so that is the user's call. | so that is the owner's call. | "the user's call": the owner's (a decision) |
| 45 | docs/engine.md | as the user runs several agents | as the owner runs several agents | "the user": the owner (B3 parked on the owner's use) |
| 46 | docs/engine.md | The user comes first: a request | Requests come first: a request | "the user": a client of the service |
| 47 | docs/engine.md | - The user waits for the detection (up to 3 s) | - The request waits for the detection (up to 3 s) | "the user": the request that preempts the job |
| 48 | docs/engine.md | development first, the user's call) | development first, the owner's call) | "the user's call": the owner's (a decision) |
| 49 | docs/engine.md | rank 0 runs `gpu-clock-cap` at 2200 MHz (homelab `docs/host.md`), the cap | rank 0 runs `gpu-clock-cap` at 2200 MHz, the cap | the homelab's private document |
| 50 | docs/engine.md | The question: the user decodes at dsfront's defaults | The question: pi's requests decode at dsfront's defaults | "the user decodes": pi's requests (sampled at dsfront's defaults) |
| 51 | docs/engine.md | decoded that way, the user's workload, with no bench type | decoded that way, pi's workload, with no bench type | "the user's workload": pi's |
| 52 | docs/engine.md | decoded at the user's sampling or recorded. | decoded at pi's sampling or recorded. | "the user's sampling": pi's |
| 53 | docs/engine.md | for under 1 % on the user's workload. | for under 1 % on pi's workload. | "the user's workload": pi's |
| 54 | docs/batching.md | The user said go on 2026-10-12. | The owner said go on 2026-10-12. | "the user said go": the owner (a decision) |
| 55 | docs/batching.md | But the user runs several agents at once only occasionally | But the owner runs several agents at once only occasionally | "the user": the owner (B3 parked on the owner's use) |
| 56 | docs/batching.md | **Against the user's use, B3 is not worth building now.** The user runs several agents at once only | **Against the owner's use, B3 is not worth building now.** The owner runs several agents at once only | "the user": the owner (B3 parked on the owner's use) |
| 57 | docs/experiments.md | \| the user runs several agents only occasionally \| | \| the owner runs several agents only occasionally \| | "the user": the owner (B3 parked on the owner's use) |
| 58 | docs/experiments.md | under 1 % on the user's workload, and it would give up | under 1 % on pi's workload, and it would give up | "the user's workload": pi's |
| 59 | docs/experiments.md | The draft gate under the user's sampling (T 1.0, top-p 0.95) | The draft gate under pi's sampling (T 1.0, top-p 0.95) | "the user's sampling": pi's |
| 60 | docs/experiments.md | Parked: "the user runs several agents at once only occasionally" | Parked: "the owner runs several agents at once only occasionally" | "the user": the owner (the quote follows batching.md) |
| 61 | docs/experiments.md | after 100 ms idle 1.1 ms on the gx10s' drives, 9.5 ms on ranks 2 and 3's | after 100 ms idle 1.1 ms on rank 0's and rank 1's drives, 9.5 ms on rank 2's and rank 3's | product names; the boxes named instead |
| 62 | docs/experiments.md | \| Interrupt coalescing on ranks 2 and 3's drives \| | \| Interrupt coalescing on rank 2's and rank 3's drives \| | product name; the boxes named instead |
| 63 | docs/experiments.md | the user is about to release the repository | the owner is about to release the repository | "the user": the owner (a decision) |
| 64 | docs/experiments.md | **Direction change (user, 2026-10-13 ~07:10): | **Direction change (owner, 2026-10-13 ~07:10): | "user": the owner (a decision) |
| 65 | golden/README.md | and from the homelab only the absolute path | and from the fleet only the absolute path | the homelab |

## Kept on purpose

- The box names, as rank labels, host names and paths.
- Hardware facts that explain results: GB10's bandwidth, the ring's topology and its 100G rank 0 ↔ rank 1 leg,
  the link speeds, rank 0's USB4 SSD (findings' fleet table with its ~1.7 GB/s, layout.md, engine.md's spill
  and revisit figures), and the SSD models in engram.md's table (ESL01TBTLCZ, Samsung PM9C1a) with their
  measured latencies, which explain the per-box differences.
- The addresses the scripts use: the 10.0.0.x SSH addresses (findings' "Overlay / SSH from rank 0" row,
  engram.md's table) and the ring's link addresses.
- engine.md's production history in its service section (`~/dsgb10/prod`, `~/dsgb10/prodbin`, the deploys'
  times): only its tailnet mention changed.
- "The kit's API key": production uses that key so that pi works unchanged. Only its path went.
- The owner's words in DESIGN's decision 11 ("i want the precision to be kept"), the decision's record.
- docs/operations.md as moved: the pinned worktree, `~/dsgb10/prodbin` and its `prev-*` directories, the
  production command with its key path and `<tailnet address>`, and the command that brings the kit back.
- Every fenced block, command path and file name; the reviewers' records in docs/review/.

## Left

- Code and scripts were outside this pass and still carry the same trivia in comments:
  `engine/server/service.sh` (lines 17–19 and 111 on the tailnet, 347 on the homelab), `.gitignore:18` (the
  kit's key path and "the homelab .env"), `engine/engram/engram.h:99–100` ("gx10s", "Sparks") and
  `engine/server/main.cpp:582` ("ranks 2 and 3's SSDs"). That last line also pads Engram reads by rank
  (`a.rank >= 2`), not by drive: an assumption about this fleet, for whoever makes the fleet configurable.
- findings.md's Environment gotchas: "the short host names do not resolve from this session" is an aside from
  the development sessions. It is left, since it is not clear whether it holds outside them.
- engine.md: "18893 is a test port in the 18xxx range of rank 0's own services", a homelab convention outside
  the patterns.
- The configuration README's Layout listing is a fenced block, so it does not name docs/operations.md.
- docs/review/*.md keep "the user", the homelab and the tailnet: they are the reviewers' records.

## Check

- Every edit matched its text exactly once (the script asserts it). The move is exact: the old README equals
  the new one with the pointer replaced by the moved block, less the two reference fixes.
- Links: 694 checked in the repo's markdown. 3 are broken, all in docs/review/external.md and all
  broken before this pass (relative links that repeat their own directory).
- Scanning the documentation for ASUS, Ascent, DGX Spark, Spark(s), gx10s, homelab, Mac, tunnel, tailnet,
  Tailscale, "the user", USB, Extreme, free disk, `~/ai/`, state-tp4 and pinned now finds only what is listed
  under Kept and Left, plus URLs.
