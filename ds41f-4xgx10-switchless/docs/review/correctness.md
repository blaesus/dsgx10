# Correctness review: release candidate 5 (2026-10-13)

The correctness part of the pre-release review: does the engine produce correct results? Every claim
of exact output was re-tested on 2026-10-13 with production's own binaries (release candidate 5:
`dsgb10_rank` sha256 b352b870…, `dsfront` ce116cea…, built from c247a78, never rebuilt here), the
repository at 191c30f, and production's API.

## Verdict

**What "correct" means here.** Every reference this review could use was computed on dealignai's
community variant `DeepSeek-V4.1-Flash-UNCENSORED-FP8` (`d61c59ea`: deepseek-ai's FP8 release with
shards 13–39, layers 10–36, changed), the weights every box holds and production serves
([DESIGN.md](../../DESIGN.md#target)). "Correct" below means **"matches the publisher's `model.py`
(run by this project's torch reference) and SGLang (the TP=4 kit) on dealignai's variant
`d61c59ea`"**. Nothing here ran on deepseek-ai's official checkpoint (rank 0's /data/models/DeepSeek-V4.1-Flash),
and the GSM8K and needle figures are the variant's, not the official model's (abliteration can lower
accuracy).

**Verified**, on that variant:
- The engine computes `model.py`'s model. Fed golden's block inputs, every block matches `model.py` to
  `model.py`'s own numerics (prose: top-1 134/135, KL 0.00017; the 28.5k-token transcript's last 505
  positions: 0.96, no clear flip).
- End to end it stays inside `model.py`'s numerics envelope: on prose it is as far from `model.py` as
  `model.py` with re-ordered sums (89/96, the same single clear flip); over the 28,489-token transcript
  it agrees on 75 % of argmaxes and predicts the text as well as `model.py` (16,238 against 16,122 next
  tokens), as documented.
- It agrees with SGLang, an independent implementation, at least as well as `model.py` does:
  teacher-forced along SGLang's tokens, 256 of 267 argmaxes (prose 89/96 against `model.py`'s 88/96,
  \|Δlp\| 0.66 against 0.744); free-running, math and code are SGLang's text token for token, and prose
  leaves SGLang's text only where `model.py` does. Production's API returns the same texts on a first
  request, with SGLang's prompt token counts.
- The tokeniser and chat encoding give the reference's tokens.
- Drafts change the speed, never the tokens: greedy and three sampled settings give the same hashes with
  and without drafts, and the server gives bit-identical tokens and log-probabilities either way.
- The speed-ups changed no output: all 7 reference hashes reproduce on release candidate 5 (one of them
  checked nothing, M1, fixed in b47c510), and CED's turn rule holds bit for bit (17 of 17).
- Production is deterministic for a given server state: B1 time-slicing changed no token (2 of 2),
  and repeats of a request gave the first answer wherever CED's turn rule promises the same bits.
  A repeat answered from another state of the same prompt can differ at a near-tie (4 of 10 repeats),
  as engine.md says but the README does not (m7).
- On tasks with known answers, through production: GSM8K's first 100 test problems 100 of 100
  (greedy, thinking off, zero-shot; the variant's figure), and passkey retrieval 18 of 18 at 33k–220k
  tokens, the key at depths 0.1, 0.5 and 0.9, in repeated sentences and in natural text.

**Not verified:** anything on deepseek-ai's official checkpoint; CED against an independent CED
implementation (only the project's emulation exists; the SGLang kit serves the same approximation, but
no long-prompt SGLang reference was recorded), and the 28.5k-token CED prefill bit for bit (M1); the
sampler's distribution (temperature, top-p, top-k) against a reference, beyond regression hashes and
draft equality; thinking mode, tool calls, `--slots` > 1 (off in production) and kept sessions across a
restart, which no check here exercised; the kernels' unit tests (not re-run; the inject check covers
their composition).

**Wrong:** no output. One reference hash was vacuous (M1, fixed in b47c510); the READMEs named the
official model (M2); a few sentences were stale or overstated (m1–m4, m7). The docs commit beside this
report fixes the sentences.

## The checks

Ring parts (G, C, H, S) ran production's binary through `run4.sh` under `ring.sh` (`PREEMPT=0`), one
part at a time between the other reviewers' parts; the commands are under "Repeating it". The API
checks (A, P, Q, N) went to production on rank 0:8888. "Documented" is the figure the docs give (the
golden check's are 2026-10-11's engine, finding m1).

| id | check | reference | release candidate 5 | documented |
|---|---|---|---|---|
| T | tokeniser and chat encoding: SGLang's four prompts rendered by the publisher's `encoding.py`, tokenised by `dsfront tokenize` | SGLang's prompt_tokens; transformers' ids (`golden/prose.tokens.json`) | 18, 39, 36 and 14 tokens, SGLang's counts; prose's 135 ids identical | "exact vs the reference" |
| G1a | prose (135 tokens, every position), each block fed golden's inputs (`--inject`) | `model.py` (`golden/prose`) | top-1 134/135 (the miss: golden margin 0.002); KL 0.00017; block output 0.43–1.3e-2 per layer (max 0.15, at routing flips); routing ≥ 806/810 per layer | 134/135; 0.5–1.3e-2 (max 0.15); ≥ 806/810 |
| G1b | prose free-running | `model.py`; its numerics variants as the envelope (split-K: 87/96, \|Δlp\| 0.59, one clear flip with margins 2.04 / 1.24) | top-1 121/135; completion 89/96, \|Δlp\| 0.59, KL 0.053; one clear flip (position 66, margins 2.04 / 1.22) | 124/135; 88/96, 0.60, 0.059; no clear flip |
| G1c | prose: argmax = the next token of SGLang's text | SGLang | 102/134, as `model.py` (102/134) | 105/134 |
| G1d | greedy generation from prose's 39-token prompt (`--mode generate`: the prompt through the step path in 8-row chunks) | SGLang's tokens | equal for 1 token; the 2nd is a near-tie that this path breaks the other way (our margin 0.015, `model.py`'s 0.243). The server's path (S1) gives SGLang's first 12, as documented | "SGLang's first 12 tokens" |
| G2 | the 28,489-token agentic transcript, free-running (bands below) | `model.py` (`golden/long`) | top-1 0.749 (21,340); argmax = next token 16,238 of 28,488 (`model.py` 16,122) | 0.75; 16,210 / 16,122 |
| G3 | the same, its last 505 positions fed golden's block inputs | `model.py` | top-1 485/505 = 0.96, no clear flip; attention 0.003–0.15, block output 0.006–0.064, index top-k 0.82–0.96 of golden's | 0.94 (476/505), none; 3–15e-2, 2–7e-2, 0.83–0.97 |
| C1 | long8k: an 8,680-token prompt with CED, then 256 teacher-forced | the CED reference (`golden/ced`, ced128) | first token 8197 at −0.1173; top-1 241/256; next token 221/255 (reference 220/255) | 8197 at −0.117; 240/255 |
| C2 | long8k in exact mode | exact `model.py` | first token 8197 at −0.1154; top-1 244/256; next token 219/255 (reference 221/255) | 8197 at −0.115; 241/255 |
| C3 | long8k with CED, against exact `model.py` | exact `model.py` | top-1 242/256 (`model.py`'s own CED against exact: 248/255) | – |
| C4 | long28k (the whole transcript): the first token, exact and CED | – | 21750 at −0.09748 (exact) and −0.09542 (CED) | 21750 in both modes |
| H1 | reference hashes: tf; 1k decode at k = 1, with `--no-l2pf`, with FP8 drafts, with bf16 drafts | the same runs on earlier builds (regression) | tf 842278b28d87dad0; the four 1k runs febc7dc54ee40a39: all equal | the same |
| H2 | reference hashes: sampled 1k; 16k CED decode; long28k CED | the same | a8f7a08f8cd1c5e8 and 81f772ab79a950e4 equal; 14650fb0739d0383 equal but covers no row (M1; fixed in b47c510: the run now gives 8f08a153d415c5c8) | the same |
| H3 | drafts under sampling: each run with and without FP8 drafts | the run without drafts | T 0.8 / top-p 0.95 / top-k 40: a8f7a08f8cd1c5e8 both; 16k CED: 81f772ab79a950e4 both; T 0.7 seed 5: 3634863d9905862b both | 3634863d9905862b both |
| S1 | SGLang's four prompts free-running (96 tokens) through rank 0's socket, at production's computation arguments (CED, FP8 drafts) | SGLang's greedy tokens | math and code identical (2/2, 96/96); prose SGLang's first 12, then " was" where SGLang has " beam" (its margin 2.75), which is `model.py`'s choice too (margin 3.03, golden/README.md); fact SGLang's first 5, then "," where SGLang has " of" (its margin 1.0, ours 0.30) | prose: "SGLang's first 12 tokens, then `model.py`'s token" |
| S2 | the same, teacher-forced along SGLang's tokens (each request continues its session by one token), top 20 | SGLang; `model.py` on prose: 88/96, \|Δlp\| over SGLang's top 5 mean 0.744 (median 0.518, p95 2.16, max 7.47) | argmax = SGLang's token at 256 of 267 steps: prose 89/96 (\|Δlp\| mean 0.66, median 0.44, p95 2.09, max 4.45; six of its seven misses are `model.py`'s too), code 96/96 (0.33), fact 69/73 (0.72), math 2/2 | – |
| S3 | S1 again on a server without drafts (`--no-spec`); each prompt twice | the same engine | tokens, log-probabilities and top 5 bit-identical with and without drafts, and run to run | "never the tokens" |
| S4 | `serve_test.py --ced`: CED's turn rule, every reuse case (short and long turns, snapshots, forks, a cancelled prefill) against the same prompt cold or the same history replayed, bit for bit | the same engine | 17 of 17 | 17 of 17 |
| A1 | SGLang's four prompts through production's API (greedy, thinking off, 96 tokens), each sent twice | SGLang; S1 | `prompt_tokens` 18, 39, 36, 14: SGLang's. The first answers are S1's texts character for character (math "391", prose SGLang's first 12 tokens, code all 96, fact SGLang's first 5); `completion_tokens` count the end token as SGLang does (math 2, as SGLang; fact 61 against SGLang's 73, for its other text). The second answers (regenerates: the kept session's state reused for all but the prompt's last 1–2 positions) repeat math and code and differ for prose (from its 5th token: " black", SGLang's second choice at margin 0.125) and fact (its last sentence): m7 | – |
| P1 | the same greedy request three times through the API (each prompt new to the server): a 2-token answer, a 400-token answer, and a 200-token summary of a 25,589-token document | the first answer | 2-token: identical (the repeats reuse 29 of 30 positions); 400-token: identical (the repeats ran cold: the window had moved past the prompt); 25.6k: the two repeats (25,588 positions reused, the last rerun through the decode step) equal each other and differ from the cold answer from its 3rd token: m7 | – |
| P2 | B1 time-slicing in production: a 600-token streamed answer X, and a 400-token Y sent while X decodes (Y answered in 9.2 s, X done 12.1 s after Y was sent); then X and Y again | each request alone (both repeats ran cold) | X and Y identical to their runs alone | "the same tokens … batched, sliced, resumed or alone" (engine.md) |
| P3 | a three-turn conversation over engine.md: turns of 25,585, 28,586 and 73,632 tokens (turns 2 and 3 CED turns from the session: 2,869 and 44,942 new positions), then each turn's whole prompt again | the first answers | turn 1 again (restarted at a window snapshot, 24,576) and turn 3 again (72,704): identical. Turn 2 again restarted at 24,576, before turn 1's answer, so it recomputed that answer through the prefill path, where the first turn 2 had kept the decode step's computation of it; it differs from char 33: m7 | CED's turn rule (engine.md) |
| Q | GSM8K, the first 100 test problems (openai/grade-school-math `test.jsonl`, sha256 3730d312…), greedy, thinking off, zero-shot, asked to end with "#### <number>" | the dataset's answers | 100 of 100 (one written "57.00" for 57, counted equal); every answer on a "####" line, none cut by `max_tokens` (2,048) or the loop guard; 18,612 completion tokens (median 176) in 210 s. The variant's figure: not comparable with the official model's | – |
| N1 | passkey retrieval (Mohtashami & Jaggi's test: a 6-digit key in filler, "What is the pass key?"), classic filler, 32.8k, 131k and 200k tokens, the key at depths 0.1, 0.5, 0.9; greedy, thinking off, each prompt cold | the key | 9 of 9 exact | – |
| N2 | the same with natural text as filler (this repository's DESIGN.md and docs, 663 KB, repeated past ~200k tokens): 33.7k, 139k and 220k tokens | the key | 9 of 9 exact | – |

On the side, the needle prompts give production's cold prefill through the API (each request's time,
its few decoded tokens included): 9.6k–10.1k tok/s on the classic filler from 32.8k to 200k tokens,
and 8.9k–9.5k on the docs from 33.7k to 220k, in line with the README's 8.6k–9.4k on real text.

The transcript by bands (G2):

| positions | top-1 vs `model.py`'s argmax | documented | clear flips | documented | argmax = next token, ours / `model.py` | documented |
|---|---|---|---|---|---|---|
| 0–511 | 0.91 (465/512) | 0.90 (460/512) | 1 | 9 | 0.64 / 0.62 | 0.65 / 0.62 |
| 512–1,023 | 0.84 | 0.83 | 11 | 8 | 0.56 / 0.56 | 0.56 / 0.56 |
| 1,024–4,095 | 0.72 | 0.71 | 176 | 200 | 0.59 / 0.57 | 0.57 / 0.57 |
| 4,096–16,863 | 0.80 | 0.80 | 541 | 521 | 0.68 / 0.68 | 0.68 / 0.68 |
| 16,864–28,488 | 0.69 (7,985/11,625) | 0.70 (8,079/11,625) | 617 | 602 | 0.44 / 0.44 | 0.44 / 0.44 |
| all | 0.749 | 0.75 | 1,346 | 1,340 | 16,238 / 16,122 | 16,210 / 16,122 |

As in 2026-10-11's run, the engine follows `model.py`'s collapse in the last band (next-token accuracy
0.68 → 0.44 on both sides; golden/README.md has the cause, layer 1's Engram on this text). G2 and G3
gave identical figures over the windows they share before G3's injection, as a deterministic engine must.

Ring time per part (the part alone; with the wait behind other reviewers' parts in brackets): G1 154 s,
G2 218 s, C 230 s, H1 280 s (560 s), H2 161 s (433 s), G3 229 s (477 s), H3 217 s (549 s), S 162 s
(589 s): about 28 minutes on the ring in all.

## Findings

No blocker: nothing here shows a wrong output. The sentences marked "fixed" were corrected in the docs
commit next to this report; no code was changed.

**M1 (major; fixed in b47c510): the long28k CED reference hash checked nothing.** One of the 7 hashes
every change is held to, `14650fb0739d0383`, comes from `--mode tf … --golden …/long28k-ced128 --prompt
28489 --limit 28489 --prefill-rows 8192 --ced`. `--mode tf` hashes only the rows after `--prompt`
(main.cpp, the teacher-forced loop), and this run has none, so it prints the hash's seed (main.cpp:719,
`chosen_hash = 1469598103934665603ull`, which is 0x14650fb0739d0383) whatever the logits.
Reproduction: the same command in exact mode (no `--ced`, `--prefill-rows 4096`) prints the same hash
although its first token's log-probability differs (21750 at −0.09748 against CED's −0.09542; part C
and H2 here). The docs cite it as bit-level evidence about ten times (README's prefill-MoE row,
ring.md's check table, engine.md: "Every run gives the hash `14650fb0739d0383`" for the 28.5k-token
prefill changes). The kernels it was meant to cover are also under the tf hash (exact, 4096 rows) and
the 16k CED decode hash (81f772ab79a950e4), so the "same bits" claims still have support, but nothing
has checked the 28.5k CED prefill (four 8192-row chunks and the replay) bit for bit. Fix (code, not done
here): fold the prompt's sampled token and its logit bits into `chosen_hash`, or teacher-force a few
positions (e.g. `--prompt 28233 --limit 28489` on the same golden). Docs annotated (fixed). (The seed
is FNV-1a's offset basis 14695981039346656037 with its last digit dropped: harmless for a regression
hash.) **Fixed** in b47c510 (2026-10-13): a `--mode tf --big` run that teacher-forces nothing now folds
the prompt's sample into the hash: its token, the bits of its logit and logprob, and its top 20
logprobs' ids and bits. long28k with CED gives `8f08a153d415c5c8` (three runs on the ring), the same
command in exact mode `87f42fa9f94f13de`, and the six other hashes are unchanged; golden/README.md
("Checking a build") has them and `golden/refhash.sh`, which runs them all.

**M2 (major, docs): the READMEs named the official model.** The root README's table and the
configuration README said "DeepSeek-V4.1-Flash"; the engine serves dealignai's
`DeepSeek-V4.1-Flash-UNCENSORED-FP8` (`d61c59ea`), and every correctness and quality figure (golden
data, SGLang, GSM8K) is that variant's. Only DESIGN.md said so. A reader would take the comparisons as
the official model's. Fixed in both READMEs.

**m1: the golden-check figures in engine.md were 2026-10-11's.** They predate ec22622, which changed
the decode step's arithmetic on purpose. Release candidate 5 gives prose free-running 121/135 (doc 124),
one clear flip at position 66 (doc: none; it is the flip `model.py`'s own split-K and bf16 variants
make, golden margin 2.04), `--mode generate` on SGLang's tokens for 1 token (doc 12; a near-tie at the
second, our margin 0.015; the server's path still gives the documented 12, S1), the long transcript
0.749 (doc 0.75). All inside `model.py`'s numerics envelope. A pointer to these figures was added
(fixed).

**m2: "CED matches model.py's CED".** The publisher's code has no CED; the reference is the project's
`ref/torchref/ced_run.py` around `model.py`. The README row now says so and that CED is an
approximation the SGLang kit serves too (fixed).

**m3: DESIGN decision 6 promised an approximate fallback that is not built.** It said an evicted or
divergent prefix falls back to Encoder SWA Bounded Replay; the server restarts exactly from the live
rings or a window snapshot instead (engine.md, "Exact reuse"). protocol.md said the same of a resumed
session, and that an evicted session is "simply cold" (it comes back from the disk tier). Fixed in
DESIGN.md, ced.md and protocol.md.

**m4: golden/README.md's rerun of `check_prose.py`** pointed at `/work/golden/golden_sglang.json`; the
file is `bench/golden_sglang.json` (fixed; the same path in `check_prose.py`'s usage line is code and
left alone).

**m5: the reference checks are not reproducible from the repository alone.** The 7 hashes' commands
are not written down in it (only tf's, in `engine/attn/tools/run_hash.sh`), and the engine-format CED
goldens (`/data/dsgb10/kernels/cedcheck`) come from a script outside it. "Repeating it" below lists
the commands. **Fixed** in 572901f (2026-10-13): `golden/refhash.sh` runs every reference hash and
prints a table, and `golden/make_cedcheck.py` makes the `cedcheck` directories from `ced_run.py`'s
streams (on rank 0 it reproduces them byte for byte) or, from the repository alone, a directory every hash
run can read (golden/README.md, "Checking a build").

**m6: two front-end behaviours that change what an API client sees are documented only in passing.**
dsfront's loop guard (on by default, with the kit's loop-abort defaults; `--no-loop-guard` in its
`--help`) cuts a repetitive generation short and reports `finish_reason` "stop"; protocol.md says only
that the front-end cancels a repetition loop. The API accepts `logprobs`/`top_logprobs` but always
returns `null` (dsfront never asks the engine for them, although the protocol and the engine carry
them). Neither changes a token that is sent; both deserve a line where API users look.

**m7: the same greedy request sent twice can come back different.** dsfront keeps each conversation's
session and matches the next request by its text, so a repeat is answered from the state the server
holds, and two states of the same prompt can differ in their last bits:
- a regenerate reuses the state up to the prompt's last position or two and reruns those through the
  decode step (a remainder of ≤ `--small-rows`), where the first request ran them through the prefill
  path;
- a turn keeps the decode step's computation of the previous answers, and a repeat that restarts
  before them (at an earlier window snapshot) recomputes them through the prefill path.

The two paths sum in different orders and agree "to numerics, not bits", which engine.md (Exact reuse)
states, so a near-tie can flip. Through production, 4 of 10 repeats differed: prose (from its 5th
token, " slab" → " black", SGLang's margin between them 0.125) and fact (its last sentence) in A1, the
25.6k summary (P1, from its 3rd token) and turn 2 of P3; the 6 others, and both B1 cases, were
identical. Both answers are the same model's greedy continuation, summed in two orders, and the engine
is deterministic given the state (the two 25.6k repeats equal each other), so no output is wrong. But
neither the README nor protocol.md says it, and a user checking that temperature 0 repeats will see
it. The root README's "every greedy run gave the earlier runs' tokens" stands: it compares the same
request sequence on fresh servers. One sentence was overstated: engine.md's turn rule said a CED turn
equals the same prompt prefilled cold "wherever it restarts", which P3's turn 2 contradicts; it holds
when every position before the restart came from the prefill path (serve_test.py's long turns say
so), and the sentence now names that condition (fixed). The rest is not fixed: it needs a line where
API users look; a code change could run a regenerate's remainder through the prefill path (if a short
chunk there gives the cold run's bits), and the second case is inherent in reusing generated tokens'
state.

## CED: an approximation, and quantified

CED (DESIGN decision 5, [docs/ced.md](../ced.md)) is not exact by construction. For a prompt of P > 128
tokens the decoder (layers 20–39) runs only over the last 128 positions, with its sliding windows cut at
s = P − 128 (Decoder SWA Bounded Replay). The decoder's global KV (layer 20's compressor and index keys,
this checkpoint's only decoder KV source) is bit-identical to the exact forward's, and decode after the
prompt reads it and the replay's window, but that window's KV at layers 21–39 is off by 5–60 %
relative, so every token after a long prompt is computed from a slightly different state. At P ≤ 128
(every prompt of `bench/golden_sglang.json`) CED is the exact forward.

The docs quantify it, against the project's CED reference (`ref/torchref/ced_run.py`, which emulates
CED around the unchanged `model.py`; the publisher's code has no CED): the first token is unchanged in
six long cases with KL 0.0004–0.012 nats, 3–15× a numerics twin's; teacher-forced decode agrees with
exact on 95–98 % of tokens with no bias in the mean log-probability, inside the split-K envelope. The
tech report says the model was post-trained with the replay simulated, and the SGLang kit served with
the same approximation (`--enable-decoder-swa-bounded-replay` in its `boot.py`), so CED is the
sanctioned way to run the model and the kit comparison is like for like. `--exact` opts out.

What this review re-measured is in the table (C rows): the engine's CED sits as close to the CED
reference as its exact mode sits to exact `model.py`, and the first tokens equal the documented ones.
The README never calls CED exact; one phrase overstated its reference ("model.py's CED", fixed in the
docs commit beside this report).

A consequence for determinism, documented as CED's turn rule (docs/engine.md): a turn of ≥ 128 new
positions replays the decoder and equals the same prompt prefilled cold, bit for bit, when everything
before its restart was prefilled; a shorter turn continues exactly from the live rings, so it equals
the same history replayed, not the same prompt cold. Positions the decode step computed (earlier
answers) keep its roundings either way. The same final prompt can therefore give different (equally
valid) outputs depending on how the conversation reached it. That is inherent in CED and in reusing
generated tokens' state, and the engine is deterministic within each rule (S4, P1–P3; m7).

## What each claim rests on

An independent reference is another implementation of the model: the publisher's `model.py`, SGLang,
or transformers' tokenizer. The engine's own hashes and A/B comparisons only show that a change left
the output alone; they are regression checks, not evidence of correctness.

| claim (where) | reference | independent? | re-tested |
|---|---|---|---|
| Every block computes `model.py`'s function; the numerics follow it (DESIGN decision 4; README rows on the kernels) | the golden data: the publisher's `model.py` run by `ref/torchref`, with torch stand-ins for its tilelang kernels (bit-equal to them on random data, golden/README.md) | yes, the publisher's code, run by this project | G1 inject, G3 |
| Weights in the checkpoint's own formats, re-coded only losslessly (DESIGN decisions 3 and 11) | load-time checks of the codes; the same hashes before and after (regression) | indirectly: a lossy weight would show in the inject check against `model.py` | G1 inject |
| End to end the engine is as close to `model.py` as `model.py`'s own numerics variants (engine.md, Golden check) | golden prose and long; the variants (split-K, g128, bf16) as the envelope | yes | G1, G2 |
| It agrees with SGLang (bench/golden_sglang.json) | SGLang's greedy completions with top-5 log-probabilities, from the kit on 2026-10-11 | yes, another implementation | S1, S2, A1 |
| Tokeniser and chat encoding exact (README) | transformers' tokenizer (golden/prose.tokens.json); SGLang's prompt_tokens | yes | T |
| CED matches its reference; first token unchanged; decode inside the envelope (README, ced.md) | `ref/torchref/ced_run.py` around `model.py` | partly: the project's emulation of the tech report's method around the publisher's code; no other CED implementation was compared | C |
| Drafts change the speed, never the tokens, greedy or sampled (README) | the same engine without drafts | a self-match, which is what this claim is about | H1, H3, S3 |
| The speed-ups changed no output (root README) | the 7 reference hashes and A/B request comparisons, recorded on earlier builds | regression only | H1, H2 |
| Exact prefix reuse, CED's turn rule, B1 time-slicing (engine.md, Serving) | the same prompt cold, or the same history replayed, on the same engine | self-match | S4, P1–P3 |

The checkpoint is the same everywhere ("correct" is about one set of weights). On all four boxes:
the list of shard sizes hashes alike (`44f1c08e198667ee`), `.expected.sha256` hashes alike
(`8609aaa6635b3be1`), shard 13 (one of dealignai's) has the sha256 the manifest lists (`4c2bcdf8…`) and
shard 1 (an original) is identical (`886aebda…`). The kit's Docker volume binds the same directory
(`~/DeepSeek-V4.1-Flash-UNCENSORED-FP8`), and `golden/run_ref.sh` defaults to it.

## Repeating it

**On the ring.** From `ds41f-4xgx10-switchless/`, each line one engine run through `run4.sh`, grouped
into parts of under 10 minutes, each part under `OWNER=… PREEMPT=0 engine/server/ring.sh bash part.sh`
and the parts behind a held want (`ring_want_hold`, ringlock.sh). `BIN` picks the binary under test
(here production's, never rebuilt); every run takes ~50 s, of which ~40 s load the weights.

```
export BIN=~/dsgb10/prodbin/dsgb10_rank
G=/data/dsgb10/golden C=/data/dsgb10/kernels/cedcheck
# G1: prose, 135 tokens, every position (golden/prose is --golden's default)
TIMEOUT=300 engine/server/run4.sh --mode prefill --quiet
TIMEOUT=300 engine/server/run4.sh --mode prefill --inject --quiet
TIMEOUT=300 engine/server/run4.sh --mode generate
# G2, G3: the 28,489-token transcript
TIMEOUT=520 engine/server/run4.sh --mode prefill --golden $G/long --quiet --onset 16864 --per-position
TIMEOUT=520 engine/server/run4.sh --mode prefill --golden $G/long --quiet --inject --onset 27984
# C: CED against the CED reference, exact against exact
TIMEOUT=240 engine/server/run4.sh --mode tf --big --golden $C/long8k-ced128 --prompt 8680 --limit 8936 --prefill-rows 4096 --ced
TIMEOUT=240 engine/server/run4.sh --mode tf --big --golden $C/long8k-exact --prompt 8680 --limit 8936 --prefill-rows 4096
TIMEOUT=240 engine/server/run4.sh --mode tf --big --golden $C/long8k-exact --prompt 8680 --limit 8936 --prefill-rows 4096 --ced
TIMEOUT=240 engine/server/run4.sh --mode tf --big --golden $C/long28k-ced128 --prompt 28489 --limit 28489 --prefill-rows 4096
# H1, H2: the 7 reference hashes (expected values in the table; golden/refhash.sh runs them all)
engine/server/run4.sh --mode tf --big --golden $G/long --prompt 4096 --limit 4608 --prefill-rows 4096         # tf
engine/server/run4.sh --mode decode --golden $G/long --ctx 1024 --steps 256 --big                           # k1
engine/server/run4.sh --mode decode --golden $G/long --ctx 1024 --steps 256 --big --no-l2pf                 # k1, no L2 prefetch
engine/server/run4.sh --mode decode --golden $G/long --ctx 1024 --steps 256 --big --spec --draft-fp8        # fp8
engine/server/run4.sh --mode decode --golden $G/long --ctx 1024 --steps 256 --big --spec --draft-bf16       # bf16
engine/server/run4.sh --mode decode --golden $G/long --ctx 1024 --steps 128 --big --temp 0.8 --top-p 0.95 --top-k 40 --seed 7   # temp
engine/server/run4.sh --mode decode --golden $G/long --ctx 16384 --steps 256 --big --ced                    # ced16
engine/server/run4.sh --mode tf --big --golden $C/long28k-ced128 --prompt 28489 --limit 28489 --prefill-rows 8192 --ced   # ced28 (until b47c510: the seed, M1)
# H3: drafts under sampling (each pair must give one hash)
engine/server/run4.sh --mode decode --golden $G/long --ctx 1024 --steps 128 --big --temp 0.8 --top-p 0.95 --top-k 40 --seed 7 --spec --draft-fp8
engine/server/run4.sh --mode decode --golden $G/long --ctx 16384 --steps 256 --big --ced --spec --draft-fp8
engine/server/run4.sh --mode decode --golden $G/long --ctx 1024 --steps 256 --big --temp 0.7 --seed 5
engine/server/run4.sh --mode decode --golden $G/long --ctx 1024 --steps 256 --big --temp 0.7 --seed 5 --spec --draft-fp8
# S: rank 0's server at production's computation arguments, no disk tier, our own 0700 directories
TIMEOUT=420 engine/server/run4.sh --mode serve --socket ~/dsgb10/rcrun/engine.sock --max-ctx 262144 \
    --disk-dir ~/dsgb10/rcspill --disk-gib 0 --keep-gib 0 &       # then, once the socket exists:
python3 sglang_sock.py --socket ~/dsgb10/rcrun/engine.sock --out free_spec.json --phase free
python3 sglang_sock.py --socket ~/dsgb10/rcrun/engine.sock --out tf.json --phase tf
python3 engine/server/tests/serve_test.py --socket ~/dsgb10/rcrun/engine.sock --ced
#   stop (pkill -x dsgb10_rank on rank 0: rank 0 stops the others), start again with --no-spec, then
python3 sglang_sock.py --socket ~/dsgb10/rcrun/engine.sock --out free_nospec.json --phase free
```

Reading them: G1–G3 print the per-block table, `logits (positions >= 0): top-1 …`, `argmax == the next
token …` and cmp_runs.py's measures (`--onset` adds the per-window table); C prints `first token
(position …)` and `logits (positions >= P): top-1 …`; H prints `hash …` (tf: `teacher-forced: … hash`).

**With other weights** (`--weights PATH`; one path for every rank, so each box needs its own rank file at
the same path): the golden data, the CED reference and the hashes above are dealignai's variant's and
do not carry over. Regenerate golden prose and long with `CKPT=<checkpoint> golden/run_ref.sh run …`
(golden/README.md, Rerun), the CED streams with `ref/torchref/ced_run.py` (golden/ced/manifest.json
lists the commands), and the engine-format `cedcheck` directories from them (`golden/make_cedcheck.py`,
since 572901f; finding m5); SGLang's completions with `bench/golden_sglang.py` on a server with
those weights. What does carry over is every equality among the hashes: k1 = --no-l2pf = fp8 = bf16;
temp with and without drafts; T 0.7 seed 5 with and without drafts; tf at every `--prefill-rows`
(engine.md: "every chunking of the prompt gives the same one").

**Through the API.** The scripts that ran the production checks (standard library only; the key is read
from a file and never printed):

```
python3 sglang_agree.py --base-url URL --key-file KEYFILE --model NAME --golden bench/golden_sglang.json --out DIR   # A1
python3 det_prod.py     --base-url URL --key-file KEYFILE --model NAME --out DIR --doc docs/engine.md            # P1-P3
python3 gsm8k_eval.py   --base-url URL --key-file KEYFILE --model NAME --data test.jsonl --out DIR --n 100       # Q
python3 needle.py       --base-url URL --key-file KEYFILE --model NAME --out DIR                                # N1
python3 needle.py       --base-url URL --key-file KEYFILE --model NAME --out DIR --filler-file DOCS.txt --seed 2  # N2
```

`test.jsonl` is openai/grade-school-math's `grade_school_math/data/test.jsonl`; `DOCS.txt` is DESIGN.md
and docs/{findings,ced,ring,batching,engram,layout,protocol,engine}.md concatenated. `sglang_agree.py
--record FILE` records a new reference in golden_sglang.json's format from another server. On rank 0
each request first waits for the ring to be the service's; `--no-ring-guard` skips that (for any other
server). The scripts and their raw outputs are in the review's scratch directory on rank 0 (not
committed). Through production the steps took 55 s (A1), 116 s (P), 210 s (Q), 158 s and 175 s (N1,
N2), each started in a gap between other reviewers' ring jobs.
