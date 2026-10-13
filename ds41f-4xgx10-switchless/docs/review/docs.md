# Pre-release review: is it correctly documented? (2026-10-13)

The docs reviewer's report, one of five (kit, correctness, docs, method, external; experiments.md, In progress 4).
It covers the public repository at master 191c30f: both READMEs, DESIGN.md, docs/, the component READMEs
and the comments that document flags. The fixes are on the branch `worktree-agent-ae8f58b0062ed5459`, which
also merges master's f6a90a7.

## Verdict

**Mostly yes after this branch's fixes, with one blocker left: the repository has no licence.**

- The published figures are right. Rerunning `rc5table.py` on the raw data reproduces `table.md` and
  `rc5table.txt` byte for byte, and every figure in the two READMEs' "Where it stands" tables and DESIGN's
  M5 and M7 traces to them.
- The links were already sound: all 530 links and anchors resolved at 191c30f.
- What was wrong came in three kinds:
  - **Which weights.** The docs did not say which checkpoint everything ran on. This is fixed: dealignai's
    variant is named wherever a figure or a claim about the model's behaviour depends on it.
  - **Stale statements.** Statements that later work superseded were still written as current: DESIGN's "one
    request at a time", the README's bundle B "being built", M7's remaining levers, and experiments.md
    placing merged code on branches. These are fixed.
  - **Written for the homelab.** The configuration README addressed its owner, not an outsider. This is
    partly fixed: there is now a glossary, a note on the status table's order, and commands that run as
    written.
- **Left, besides the licence, are two majors:**
  - The evidence for the headline figures exists only on rank 0. It should be published with its scripts.
  - An outsider has no path from a clone to a running endpoint. That is the kit review's subject.
- **Privacy:** no secret, key or conversation is in the tree or its history. What remains is homelab detail:
  a tailnet address, the account name inside the golden transcript, and the key file's path. Each needs a
  decision.

Counts:
- **Links:** 0 were broken, so none needed fixing. Two links to `docs/review/kit.md` resolve once the kit
  review merges.
- **Numbers:** 11 mismatches, 9 of them fixed.
- **Flags, defaults and paths:** 13 corrected in cb754c5. Also corrected: the golden-data command,
  findings' ring lock, and DESIGN's defaults and ports.
- **Findings:** 2 blockers (one fixed), 4 majors (two fixed, one partly fixed, one left), and the minors below.

## What was checked, and how

- **Links.** A checker covered inline, reference and image links outside code, and anchors under GitHub's
  slug rules (`<a name>` targets and `#Lnn` on code files included). It also checked that every target file
  is tracked. All 56 external links (50 distinct URLs) returned 200 on 2026-10-13.
  - At 191c30f: 530 links, 0 broken.
  - After the fixes: 561 links. 2 are broken on purpose, both pointing to `docs/review/kit.md`.
- **Numbers.**
  - **Release candidate 5.** I reran rank 0's /data/dsgb10/bench/rc5/scripts/rc5table.py and changes.py from
    the raw logs and compared them with table.md and rc5table.txt. Then I traced the root README, the
    configuration README's "Where it stands", DESIGN's M5 and M7, and engine.md's release-candidate-5
    section to them.
  - **sparkDash means** were recomputed unrounded from the sd1 and sd2 records.
  - **The baseline** was recomputed from `bench/baseline-sglang-decode.json` and `-prefill.json`.
- **Flags and defaults.** Every flag, environment variable and default the docs name was checked against the
  code: `main.cpp`, `service.sh`, `run4.sh`, `ring.sh` and `ringlock.sh`, `bench.py`, the tests and tools.
- **Outsider reading.** I read both READMEs as someone who has never seen the homelab, noting homelab
  references, undefined terms, superseded rows and commands that do not run as written.
- **Consistency.**
  - experiments.md's rows against engine.md's dated sections and the README rows; 15 rows were spot-checked
    in full.
  - Every commit the index names was checked against master's history.
  - DESIGN's milestones against the README; the root summary row against the configuration README.
- **Privacy.** A scan of the tree and the history covered home paths, user@host, e-mail addresses, private,
  tailnet and other IPv4 addresses, key-shaped strings and homelab directories.
  - The provenance of the committed data files: golden/long.messages.json, golden/prose_src.json,
    bench/edit_bench.json, bench/piturn/models.json and bench/sparkdash/corpus.py.
  - The API key file was never opened. I searched the tree and every commit for its presence by other means.
  - No conversation data, spill directory or service log was read.
- **Licensing.** Every licence or notice file was checked, along with the third-party material in ref/,
  bench/ and frontend/, sparkDash and the Cargo dependencies.

## Findings

### Blockers

**B1. No licence for the project's own code (left).**
- There is no LICENSE, COPYING or licence statement at the root or in the configuration. The only licence
  files are `ref/LICENSE-deepseek` (DeepSeek's MIT, for their reference code) and `bench/THIRD_PARTY.md`
  (pi's MIT).
- An outsider would conclude that the author keeps all rights. GitHub's terms let them view and fork it on
  GitHub, but they may not use, copy, modify or redistribute the engine. For a repository released as a kit
  that is the opposite of the intent.
- A reader may also wrongly take the MIT file in ref/ as covering the whole repository.
- Proposed: [Left for the orchestrator](#left-for-the-orchestrator-with-proposed-text), item 1.

**B2. The checkpoint was not named where it mattered (fixed, 5ccebcf).**
- The engine was built, validated and measured on dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 at
  `d61c59ea`. Only DESIGN.md:8 said so.
- Both READMEs said "for DeepSeek-V4.1-Flash", and the SGLang baseline ran the same variant. Claims about the
  model's behaviour did not say which weights they were seen on, such as golden/README's Engram collapse,
  "the model is numerically sensitive" and the long-transcript accuracy.
- Now:
  - the root summary row and the configuration README's first paragraph name the checkpoint, its revision,
    what it shares with the official one and why the served name is `deepseek-v4.1-flash-uncensored`;
  - engine.md, ced.md and golden/README say which weights their measurements ran;
  - the collapse is attributed to both weight sets, the official one under `model.py` on the first 5,120
    tokens;
  - the sparkDash comparison notes that the kit published on the official weights.
- Left to the orchestrator: the sentence "the official checkpoint not yet run through the engine (see
  docs/review/kit.md)". Master's direction change (experiments.md, In progress 4) will turn these sentences
  into history once the figures are re-measured on the official weights.

### Major

**M1. Superseded statements written as current (fixed, 2cd55e2).**
- DESIGN.md:
  - Its "One request at a time" bullet said the engine serves one session per step and "the user decides
    whether to build it". In fact B1 has served in production since 2026-10-12 12:59 and B2 is merged.
    It now reads "Several requests at once: B1 in production, B2 merged off, B3 not built".
  - M7 named "the 18.6 ms one-row step" as a remaining lever and "~1.8 tokens in ~26 ms" as the prose step.
    It now gives the sparkDash comparison with the best published figures.
  - M8 said "standalone"; the store runs inside the engine. M5 said "larger chunks next"; 8192-row chunks
    have been the default since aa05c03.
  - Also corrected: M6, M9 (now on release candidate 5), the defaults bullet (the FP8 drafter head and B′),
    the service's ports and systemd note, "Both stacks benchmarked", the NVMe IOPS budget and the survey's
    date.
- The configuration README:
  - The last Status row said the draft pass's approximations were "being built". They were built in four
    stages and failed their A/B: +1.50 % against the 2 % bar.
  - There were no rows for B′ (adopted, ae57e04), the reduce-scatters' race (fixed, 1128401) or a small
    turn's time to first token. All three are added.
- engine.md:
  - Its opening status (2026-10-12) gave 67.0 / 99.0 / 96.5 / 133.2 tok/s and "Next: ..." with no pointer to
    the current figures. A line now points to release candidate 5's section.
  - "Nothing is merged" under bundle B's A/B contradicted its own stage-1 paragraph, which says the flags are
    merged (ccec056).
  - The production paragraph gave 06:48 as the start and stopped before release candidate 5.

**M2. experiments.md's "Code" column and verdicts (fixed, 422293a and 889741d).** Rows are cited by
their line on this branch.
- Rows placed code on branches when it is on master:
  - B′'s rows, lines 140, 141, 143 and 144 (merges ae57e04 and c4ac842);
  - the prefill MoE, line 192 (7d00d5e);
  - the small turn's first-token line, lines 193–195 (3cbd594, c7c1192, 4cb4209);
  - bundle B's stage 1, line 218 (ccec056);
  - the race's fix, test and audit, lines 307–311 (1128401, 0b0604f).
- Lines 221 and 222 and engine.md said nothing of bundle B was merged, but stage 1 is.
- Line 290 cited c3a49d9, which is in no branch: a rebase replaced it with 4c9536d.
- Line 123 was still "In progress" though B′ was adopted. Line 288 gave production's start as 06:48; the
  service took 127.0.0.1:8888 at 06:39 and the tailnet address at 06:48.
- Seven rows used verdicts the legend did not define (Done, Failed, Missed), one row's line ("release") was
  not in the legend, and line 141 had no verdict. The legend now defines them and says why dates are not
  always in order.
- Added: a row for `engram_iobench`'s random reads (engram.md's table had none), and line 113 now links the
  section that holds its 16.6–16.7 ms.

**M3. The evidence for the headline figures is not public (left).**
- The root README says "The raw data and the scripts are in rank 0's /data/dsgb10/bench/rc5" (README.md:66).
  That is 14 MB of logs plus `rc5table.py`, `changes.py` and the part drivers. An outsider cannot check any
  of the tables.
- The SGLang kit's time to first token (244–280 ms) has no raw record anywhere. `bench/baseline-sglang-*.json`
  hold decode, prefill and the agentic turns only.
- Proposed: [Left for the orchestrator](#left-for-the-orchestrator-with-proposed-text), item 3.

**M4. Written for its homelab (partly fixed, 5a4cffe; the rest left).**
- Fixed:
  - **A glossary.** The configuration README now has a "Terms used below" paragraph covering the boxes and
    ranks, the ring, TP and EP, and the SGLang kit and the MiaAI kit as one project measured in two places.
    It also defines RoCEnante, pi, dsfront, CED, SWA, mHC, Engram, DSpark and MTP, k = 1, c1, B1–B4, B′,
    release candidate 5 and its commit, M0–M9 and "the user".
  - **The status table.** A sentence says its rows run in time order, so later rows supersede earlier ones.
  - **bench.py.** The command now names a key file of the reader's own, and the docs say why the model is
    `deepseek-v4.1-flash-uncensored`.
  - **The golden-data command** now runs. The old form exited with status 2.
  - **The Layout** lists docs/, the edit bench and engine/dspark/.
- Left:
  - There is no path from a clone to a serving endpoint. The configuration README has one cmake line
    (README.md:192) and no word on dsfront's build, the converter, distributing the rank files, run4.sh or
    the ring's addressing. See [Left for the orchestrator](#left-for-the-orchestrator-with-proposed-text), item 4, and docs/review/kit.md.
  - README.md:129–162 is homelab operations: the production command (with the tailnet address and the kit's
    key path), the binaries' directories, and how to bring the kit back. Proposed: [Left for the orchestrator](#left-for-the-orchestrator-with-proposed-text), item 5.

### Minor, fixed

- **99da249, privacy:**
  - the account name in the README's ssh example and in findings' overlay-address note;
  - a home path in `gen_tokenizer_cases.py`'s usage line.
- **99da249, golden/README:**
  - the long transcript has 11 tool calls, not 13;
  - its provenance is now stated: CPython 3.12 sources, real command output, DESIGN.md as of 4a544f1, and
    user turns and answers written in the script. No secret and no real conversation is in it, and the only
    homelab string is the worktree's path, which names the account.
- **cb754c5, flags, defaults and paths, as the code has them:**
  - engine.md: `engine/server/tools/evict_cache.py`; the workers' `timeout $((TIMEOUT + 60))`; a mock instance
    needs its own `DSGB10_HOME`, `RING_LOCK` and a free port; the run id also hashes `--draft-head4` and
    `--draft-markov8`;
  - protocol.md: session reuse as built (the disk tier, `--disk-gib` 32 by default);
  - ring.md and engine/kernels/README: tests go through `ring.sh`, not a bare `flock`;
  - engine/attn/README: `DSGB10_MOE_TUNE` has no reader, and `bench_kernels`' `PF_*` knobs are the ones in
    use;
  - golden/README and check_prose.py: the paths;
  - run4.sh's header: `DSGB10_YIELD_IDLE_S` defaults to 600 s, and production uses 120.
- **5a4cffe, numbers:**
  - The MiaAI kit's step times were given as 31–34 and 37–39 ms. Its own probe gives 32.4–33.6 ms at 2.27
    tokens and 38.9–39.4 ms at 3.87.
  - The root README's "9–10 % faster" was true of bench.py only. sparkDash's single prompts gained
    7.7–9.2 %.
  - Both READMEs now name the release candidate's commit, and say which kit and which weights each table
    compares with.
- **2cd55e2, the root summary row's prefill:**
  - "8.5–9.4k (kit: 4.2k)" paired bench.py's 16k–57k figure with sparkDash's 120k one. It now reads
    "8.5–9.2k at 16k–57k (kit: 4.2–4.3k) and 9.4k at 120k".
  - "Each dsgb10 figure is the mean of its sessions" does not hold for the edit bench, whose end-to-end
    times are given per session.
  - The 2026-10-12 column's 6.5k came from the M7 run, where DESIGN's M5 quotes the k = 1 run's 6.9k. A note
    now says so.
- **2cd55e2, DESIGN's Engram budget:** it said the NVMe gives "~112k IOPS at QD 64", which is the kit's
  figure. The drives measured 377k–854k at QD 64 (engram.md), and DESIGN now says so.
- **2cd55e2, findings:**
  - "what served here until today" is now dated 2026-10-11 (README's link follows the new anchor);
  - its ring-lock note said to use a bare `flock`, which registers no want and so waits as long as the
    service serves. It now says `ring.sh`.

### Minor, left

- **The SGLang kit's decode figures are 0.1 tok/s high, in its favour.**
  - 77.4 / 73.6 / 106.9 are means of bench.py's printed one-decimal rates. The JSON gives 77.32 / 73.55 /
    106.85, that is 77.3 / 73.5 / 106.8; prose's 46.8 agrees.
  - They appear in 12 lines: README.md:24; the configuration README:48 and 106; DESIGN.md:26 and 216;
    engine.md:20, 2085 and 3110–3112; experiments.md:207 and 232; findings.md:136.
  - Proposed: take the means from the JSON when the kit is re-measured, as master's direction change plans.
    Until then the difference only flatters the kit.
- **The tailnet address.** rank 0's Tailscale address (a carrier-grade NAT address) appears in DESIGN.md:218, the
  configuration README.md:150–151, engine.md:791 and service.sh:17, and in the public history.
  - It is reachable only inside the owner's tailnet, so the risk is low, but it identifies the homelab.
  - Proposed: write `<tailnet address>` or "rank 0's tailnet address", or move the lines to the operations
    note (item 5 below).
- **The account name in the golden transcript.** golden/long.messages.json:309, 321 and 348, and their
  tokens in long.tokens.json, hold the absolute path of the worktree the agentic script ran in. That path
  names the account, and it is already public.
  - Removing it changes the transcript's tokens, so every golden array and comparison made from it would
    have to be remade.
  - Proposed: accept it, as golden/README now states, or change it at the next golden regeneration.
- **The API key.** The key is in no file and no commit. Its file's path appears in .gitignore:18, the
  configuration README.md:150 and findings.md:452. These are paths only; the operations note (item 5 below) could
  take them.
- **Licence notices for third-party material.** Each is short, but each is owed:
  - `frontend/src/encoding.rs` is a port of DeepSeek's `encoding.py` (MIT) and does not say so under that
    licence;
  - `frontend/src/tokenizer/unicode_tables.rs` is generated from the Unicode Character Database 16.0.0
    (Unicode License v3) with no notice; its header comes from `tools/gen_unicode_tables.py`;
  - `golden/long.messages.json` embeds CPython 3.12 standard-library sources (PSF-2.0). The two upstream
    e-mail addresses the scan finds are from those files' headers;
  - ref/ does not say which files are DeepSeek's (inference/, encoding/, config.json,
    tokenizer_config.json) and which are this project's (torchref/, inventory.txt).
  - These are covered correctly: pi's MIT notice (bench/THIRD_PARTY.md). Not vendored, so they need no
    notice: sparkDash (Apache-2.0, loaded from `SD_ROOT` at run time); bench/sparkdash/corpus.py's corpus
    (built from this repository); the Rust crates (fetched by Cargo).
  - Proposed: [Left for the orchestrator](#left-for-the-orchestrator-with-proposed-text), item 2.
- **Commit metadata.** Every commit, the published ones included, carries the author's personal name and
  e-mail address. This is the owner's choice; it is noted only because the review was asked about personal
  data.
- **A stale public branch.** origin carries `attn-wcomp` (c24b5ec), already merged into master as 9eb3b56,
  while experiments.md's legend says the public repository carries master only. Proposed: delete the remote
  branch.
- **Two builds outside the repository.** bff5d5d and 8805de4 (experiments.md:192; engine.md:5216–5236, 5358
  and 5558) are builds that a rebase replaced, and no branch holds them. engine.md:5358 maps 8805de4 to
  2c85d77. An outsider cannot look them up, but they name the data directories, so they stay.
- **The "In progress" heading.** experiments.md's "In progress (2026-10-12, evening)" holds items of
  2026-10-13. It was not renamed here, to avoid a conflict with master's item 4. Proposed: "In progress
  (2026-10-12 to 2026-10-13)".
- **"today" in dated sections.** 74 lines in engine.md, plus experiments.md lines 88, 157, 217 and 278. Each
  is anchored by its section's or row's date, so none is wrong.
- **Older "best published" figures.** findings § The MiaAI-Lab kit and § Other recipes, and DESIGN.md:31–37,
  carry the 2026-10-11 survey: 87.7 / 124.8 tok/s, "the bar ~88 / ~125–132". The READMEs use the kit's v2.4
  figures, 89.9 / 131.0 of 2026-10-12. Both are dated now, and § sudoingX (2026-10-12) compares with that
  day's figures.
  - Proposed, under § Other recipes: "The kit's later documents (v2.4, 2026-10-12) give 89.9 prose / 131.0
    code: README.md." The external review covers these figures' provenance.
- **Unreferenced file.** `golden/prose_src.json` (a prompt and a completion, committed in 6d35638) is named by
  no doc or script. Proposed: say what it is in golden/README, or drop it.
- **M7's state.** DESIGN's M7 is still "beats the baseline on every prompt type". Its second condition, the
  best published, is now met on sparkDash, so the orchestrator may mark it done.
- **No experiments row for the sudoingX evaluation.** findings § sudoingX has no row in the index. That is
  consistent: the index covers experiments on this engine, and evaluations of other projects live in
  findings.md, so no row was added.

## Left for the orchestrator, with proposed text

1. **Licence (B1).** The owner chooses.
   - MIT would match the DeepSeek reference code and pi: a root `LICENSE` with the standard MIT text and
     "Copyright (c) 2026 <owner>".
   - Then a root README section:

     ```markdown
     ## Licence

     The code is under the MIT licence (the LICENSE file). `ds41f-4xgx10-switchless/ref/inference` and
     `ref/encoding` are DeepSeek's reference code under their MIT licence (`ref/LICENSE-deepseek`); the
     agentic-edit bench carries pi's prompt and tools (MIT, `bench/THIRD_PARTY.md`); other third-party
     material is listed in NOTICE. No model weights are included.
     ```
2. **Notices.** A `NOTICE` at the root, or a THIRD_PARTY.md, listing:
   - encoding.rs: a port of DeepSeek's encoding.py, MIT, Copyright (c) 2023 DeepSeek;
   - unicode_tables.rs: derived from the Unicode Character Database 16.0.0 under the Unicode License v3;
   - golden/long.messages.json: excerpts of CPython 3.12's `json`, `textwrap` and `heapq`, PSF-2.0;
   - ref/: which files are DeepSeek's.
   - Add a one-line pointer in encoding.rs's header and in gen_unicode_tables.py's generated header.
3. **Publish the evidence (M3).**
   - Commit `rc5table.py`, `changes.py` and the part drivers (`bpart.sh`, `drive.sh`, `mtest.sh`,
     `ref_editpart_r2.sh`) under `bench/rc5/`. First delete rc5table.py:17, an unused scratch path that
     names the account.
   - Publish the logs (14 MB) as a release asset, after a privacy scan of `front.log` and the dumps.
   - In README.md:66, replace "The raw data and the scripts are in rank 0's /data/dsgb10/bench/rc5." with
     `The scripts are in [bench/rc5](ds41f-4xgx10-switchless/bench/rc5) and the raw logs in <the release asset>.`
   - Do the same for the re-measurement on the official weights, and have bench.py record the time to first
     token in its JSON.
4. **A quick start (M4).** Add a "From a clone to a serving endpoint" section to the configuration README:
   - the hardware and the ring's addressing (docs/ring.md);
   - CUDA 13.0 for sm_121a;
   - `cmake -S engine -B engine/build && cmake --build engine/build -j`;
   - `cargo build --release` in frontend/;
   - converting and verifying the rank files (docs/layout.md), distributing them, then `run4.sh` and
     `service.sh`.
   - The kit review (docs/review/kit.md) has the tested sequence.
5. **Operations out of the README (M4).** Move the configuration README.md:129–162 to
   `docs/operations.md`: the production command, the binaries' directories, restarting around held wants,
   and bringing the kit back. Leave one line in the README:

   ```markdown
   dsgb10 serves dealignai's variant on rank 0:8888 (since 2026-10-12 06:39; on release candidate 5 since
   2026-10-13 04:13); how it runs there: [docs/operations.md](../operations.md).
   ```

   This also moves the tailnet address and the key path out of the main page.
6. **The tailnet address**, if the operations note keeps it: write `<tailnet address>` in the five places.
   History keeps it unless rewritten, which is not worth doing for this address.
7. **The golden transcript's path:** accept it, as documented, or change it at the next golden regeneration.
8. **The kit's decode means:** 77.3 / 73.5 / 106.8 from the JSON, or the re-measured figures, in the 12 lines.
9. **Delete the public `attn-wcomp` branch:** `git push origin --delete attn-wcomp`.
10. **Once the kit review lands:** finalise the two sentences that link `docs/review/kit.md`, in the root
    summary row and the configuration README's paragraph. When the official weights are measured, turn the
    variant's figures into history, as master's direction change says.
11. **Small items:** the "In progress" heading's date; the findings line on v2.4's figures; golden/prose_src.json;
    marking M7 done.

## Checked and found right

These are the figures and claims that held up:
- **The headline tables.** Every figure in the root README, the configuration README's "Where it stands",
  DESIGN's M5 and M7, and engine.md's release-candidate-5 section matches table.md and rc5table.txt. Those
  two reproduce exactly from the raw logs.
- **The kit's baseline.** Its prefill (1.9k / 3.7k / 4.3k / 4.2k / 4.25k) and agentic turns (3.73 s;
  0.71–0.79 s) match the JSON. The JSON has no field for the time to first token, so that figure could not
  be checked.
- **"16.71 ms a token at k = 1 against 17.47 with `--no-l2pf`"** (the configuration README) matches release
  candidate 5's own validation record: 16.705 and 17.466 ms, in the production binaries' `ENGINE_BUILT_FROM`.
  A cross-check had flagged it against engine.md's B′ figures (16.687–16.737 against 17.470–17.511), but
  those are other runs.
- **moe_sweep.sh's ROUTES default** is a file prefix, as bench_kernels' `PF_ROUTES` expects, and its files
  exist on rank 0.
- **experiments.md lines 120, 129 and 157** take their figures from the named commits' messages, as the legend
  allows.
- **No secrets.** No API key, token or credential is in the tree or the history. bench/piturn/models.json
  reads its key from `$DSGB10_DEV_KEY`. The key-shaped strings in ref/encoding/encoding.py are variable
  names.
- **The data files** contain no personal data:
  - bench/edit_bench.json: pi's prompt and tools, and tasks written for the bench;
  - bench/golden_sglang.json: the SGLang kit's greedy completions on the bench prompts;
  - golden/prose_src.json: a story prompt and its completion;
  - bench/sparkdash/corpus.py's corpus: this repository at a pinned commit.
- **The model name** `deepseek-v4.1-flash-uncensored` is dsfront's default `--model-name` and what the kit
  served, so the bench command works against both.

## Commits on this branch

| commit | what |
|---|---|
| 5ccebcf | the checkpoint named, and which weights each claim about the model's behaviour was seen on (B2) |
| 99da249 | no account name or home path in the README's ssh lines, findings or gen_tokenizer_cases.py; golden/README's transcript (11 tool calls; provenance) |
| cb754c5 | flags, defaults and paths as the code has them (13 corrections) |
| 5a4cffe | both READMEs readable for an outsider: the glossary, the table's order, the kit's step times, the commands |
| 422293a | experiments and ring: the race's fix and test as merged (1128401, 0b0604f) |
| 70618ea | merge of master (f6a90a7) |
| 889741d | experiments: code on master named with its merge, stage 1 merged off, the legend's verdicts, lines and order, the engram_iobench row |
| 2cd55e2 | superseded statements in README, DESIGN, engine.md and findings brought up to date; the root summary's prefill pairs |
| (this file) | the review |
