# External review: the comparisons with other kits (2026-10-13)

Pre-release audit, question 4, external side: are dsgb10's measurements objective and fair in relation to
other kits' numbers? This page checks every number the documentation attributes to someone else, the
precision each side runs at, the checkpoint each side runs, and whether "the best published four-GB10
numbers" is true on 2026-10-13. How our own SGLang-kit baseline was measured (`bench.py`, its logs) is the
sibling review's question and is not repeated here.

**Method.** Every external figure was read from its source at a pinned commit (cloned, or fetched from the
GitHub and Hugging Face APIs), not from summaries. The two checkpoints were compared on rank 0 from their files
and from the Hugging Face hashes. GPU clocks were read on all four boxes with `nvidia-smi`. No GPU job was
run. The survey also used research agents: GitHub with its forks, issues and PRs; the NVIDIA forums,
Hugging Face cards and other sites; and SGLang's numerics. Every figure in the survey that is near or above
one of ours was re-read from its source; the others are as the agents cited them.

## Verdict

The comparisons are honest in substance, and on method they mostly favour the other side:
- Every external figure traces to a public document, and the version labels on the README's MiaAI column are
  right (c1 from v2.4, catalogs from v2.2, prefill from v2.1).
- The statistic favours the kit: each of its cells is the better of two sweeps, ours the mean of two sessions.
- The fabric favours the kit: its figures are from a switched fabric. The same kit on a switchless ring
  publishes 80.9–84.4 tok/s on prose c1, against 89.9 switched.
- The precision favours the kit slightly. Both stacks run the checkpoint's own weight formats, quantise
  activations by the reference's rule, verify drafts exactly and use the same trained-in prefill
  approximation (CED). But on GB10 the kit's attention also runs its query and attention weights in FP8,
  where the reference and dsgb10 keep bf16.

As of 2026-10-13, no published result on four GB10s at the checkpoint's native precision beats dsgb10 on
any README row. The nearest is a code step 1.6 % shorter than ours that carries a third fewer tokens
(spark-ds41f r6h: 34.25 ms at 3.37 tokens, against 34.8 ms at 5.10). Outside those terms some results do
beat it:
- **Eight GB10s.** rhys101's TP8 line of the same kit beats every c1 row on the same benchmark: 111.9 /
  184.9 / 203.9 / 177.2 ([MiaAI PR #39](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/39),
  2026-09-26).
- **Lower precision.** TensorFold TP4, on four GB10s with 2.9-bpw EXL3 experts, publishes 160.3 code and
  219.5 counting tok/s on its own prompts (2026-10-09). That is above our code and structured rows (148.1,
  186.4); its prose is below (95.4).
- **Unverifiable.** One user reports 195–198 tok/s single-stream on the kit, on prompts that are not named.

The prefill lead survives every caveat below. Against the best published real-text figure it is ~1.6×, not
the 1.7–1.9× the README shows against the kit's 4.8k–5.0k: christopherowen's spark-ds41f (vLLM, four Sparks
in a ring) publishes 5.7k–5.9k at 32K–128K.

The README is not release-ready on four counts. All four are major and none is a blocker:
1. **The checkpoint.** Our figures are on dealignai's abliterated variant; the kit's are on the official
   weights. The same bytes stream per token, but the greedy texts and the draft acceptance differ, and neither
   README says so.
2. **The step rows.** They compare our sparkDash c1 prompt at 2.83 tokens a step with the kit's own,
   unpublished probe prompts at 2.27. They also pair a v2.1 step range with a v2/v2.2 token count, although
   v2.2 published faster steps.
3. **The catalog row.** It compares sparkDash runs forced to 256 tokens (our mean) with the kit's own,
   unpublished harness, whose length, stop rule and statistic differ.
4. **"Best published".** The column holds the kit's v2.4 figures, which are not the best per row. Higher
   ones are published for prose (90.2) and code (133.2), both v2.2 in MiaAI's PR #37, and for JSON (135.3, a
   community ring run of v2.3). Real-text prefill (5.7k–5.9k) and the code step (34.25 ms, at fewer tokens)
   are best in spark-ds41f. The claim also carries no date and no precision, and v2.4 lives in a repository
   the README never names.

With the replacements below the comparison is fair. The decode leads are probably real: +11–20 % per c1 type
against the best published, and +26–28 % on the catalogs. Prefill is ~1.6×. The exact size of the decode
leads waits for a measurement on the official checkpoint.

## Must change before release

| # | Severity | In one line |
|---|---|---|
| [M1](#m1-the-checkpoint-is-not-disclosed) | major | Say that dsgb10 (and the local SGLang baseline) ran dealignai's abliterated variant and the MiaAI kit the official weights, or re-measure on the official weights. |
| [M2](#m2-the-step-rows-mix-versions-and-prompt-sets) | major | Step rows: use v2.2's 31.2–32.5 / 37.4–37.8 ms, say the prompts differ, and drop "the fairer measures". |
| [M3](#m3-the-catalog-row-compares-two-harnesses) | major | Catalog row: say the kit's figures come from its own unpublished harness (length, stop rule, statistic unknown). |
| [M4](#m4-best-published-is-not-the-best-per-row-and-has-no-date) | major | "Best published": date it, say "at the checkpoint's native precision", name knapcio's repository, and give the per-row best (prose 90.2, code 133.2, JSON 135.3, real-text prefill 5.9k). |
| [m1](#m1-single-prompt-rows-understated) … [m11](#m11-the-kits-per-token-step-is-v21s) | minor | Wording, dating and attribution (below). The small, certain ones are applied on this branch ([list](#applied-on-this-branch)). |

## Provenance of every external number

### The README's comparison table

The root `README.md` (lines 29–56) and the configuration `README.md` (lines 36–50) carry the same rows.
"Ours" is release candidate 5 (RC5, 2026-10-13: rank 0's /data/dsgb10/bench/rc5/sd1 and sd2).

| Row | Ours | The kit, as cited | Where the kit's figure is from | Like for like? | Verdict |
|---|---|---|---|---|---|
| c1 decode, tok/s: prose / code / structured / JSON | 102.2 / 148.1 / 186.4 / 162.6 | 89.9 / 131.0 / 158.7 / 128.4 (v2.4) | [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) README, "Current results", at `27616be` (2026-10-12). sparkDash 1.8.8 DecodeBench: 256 tokens, T 0, thinking off, two discarded warm-ups. c1 is the median of three; each cell is the better of two sweeps on two boots. Four DGX Sparks, switched RoCE, RoCEnante, all four capped at 2200 MHz, EP 1. Official weights (`HF_REVISION=fb2764a5…`) | Same tool, version, prompts, length and sampling. Different: weights ([M1](#m1-the-checkpoint-is-not-disclosed)), fabric (their switched against our ring), clocks (ours: only rank 0 capped, [m2](#m2-the-clock-caps-are-not-equal)), statistic (their better of two, our mean of two), greedy text | Figures exact. JSON's listed runs (123.4 / 129.5 / 128.5) have median 128.5, not their table's 128.4, which does not matter. Not the best per row ([M4](#m4-best-published-is-not-the-best-per-row-and-has-no-date)). |
| Prompt catalogs, prose / structural | 78.5 / 125.0 (mean of 16 and 18 prompts) | 61.1 / 99.0 (v2.2) | knapcio `docs/results/validation-20260925-v22.txt`: `{"prose": 61.11, "structural": 98.96, "other": 74.71}`. The kit calls these "45 varied prompts (prose, structured and other catalogs, c1 greedy)". The harness is in neither repository | Probably the same prompts (sparkDash's 16 TEXT_PROMPTS and 18 STRUCTURAL_PROMPTS, as knapcio's `ab_inboot.py --prompts sparkdash` loads them). Length, stop rule and statistic: ours forced 256 tokens (`ignore_eos`), mean; theirs unpublished | Figures exact, and the right (newest) version. The comparison is not like for like ([M3](#m3-the-catalog-row-compares-two-harnesses)). |
| A decode step, prose | 27.9 ms at 2.83 tokens | 32.4–33.6 ms at 2.27 | The ms are v2.1's (`docs/tp4.md` of MiaAI-Lab's kit at `cad252b`, 2026-09-25). The 2.27 tokens are v2.2's probe (`tok_per_step` 2.274 in the v2.2 file) or v2's switched column in knapcio `docs/switchless-ring.md` ("33.0 ms (2.27 tok/step)") | No. Ours is sparkDash's c1 prompt; theirs is the kit's own step probe, whose prompts its documentation says are "not published". The tokens per step differ too | Version mix. v2.2's probe gives 31.24–32.5 ms at 2.274 ([M2](#m2-the-step-rows-mix-versions-and-prompt-sets)). |
| A decode step, code | 34.8 ms at 5.10 tokens | 38.9–39.4 ms at 3.87 | As above. v2.2 gives 37.4–37.84 ms at 3.865 | No, as above | As above. Not the shortest published code step either: spark-ds41f r6h takes 34.25 ms at 3.37 tokens (its own prompt, reasoning on; [survey](#survey)). |
| Cold prefill, real text, 15k–120k tokens | 8.6k–9.4k tok/s (8,624–9,445) | 4.8k–5.0k tok/s (v2.1) | MiaAI-Lab `docs/tp4.md` (v2.1, 2026-09-25): 4824–4922 / 4827–4830 / 4942–4970 / 4822–4871 tok/s at ~15k / 32k / 62k / 120k, two passes, chunk 4096, clocks then uncapped. v2.2 measured 4743 / 4822 / 4946 / 4800 | Same method class: cold, a unique prefix, prompt tokens / time to first token. Each side used its own documentation and code as text. v2.4 moved to 8192-row chunks and published real text only at 388K (4,976 tok/s) | Exact and correctly labelled, but not the best published. christopherowen's spark-ds41f r6h (vLLM, four Sparks in a ring, 2026-10-12) measured 5,870 tok/s at 32K and 5,726 at 128K, cold, on Python source. So the lead over the best is ~1.6×, not 1.7–1.9× ([M4](#m4-best-published-is-not-the-best-per-row-and-has-no-date)). Corpora differ ([m8](#m8-the-prefill-row-uses-different-texts)). |
| Configuration README: "the kit's 31–34 and 37–39 ms" | as above | v2.1 ∪ v2.2 | As above | As above | [M2](#m2-the-step-rows-mix-versions-and-prompt-sets). |
| Configuration README, status: "(RoCEnante: 16.9)" | 12.3 µs at 10 KB (ring.md: 12.2) | 16.9 µs | christopherowen/dgx-spark-networking README: "One-shot, bidirectional host relay", four Sparks in a cable loop, BF16, the median of the slowest rank per sample of 256 graph-replayed calls | Close. Ours is back to back in a graph; theirs is the median of the slowest rank | The figure is right but misattributed: it is christopherowen's host relay ("ring4"), not RoCEnante, and the same package later measured 16.2 µs ([m3](#m3-the-collective-latency-is-misattributed)). |
| Configuration README, status: "the best published ~4.9k" | 6.9k–7.2k then; 8.5k–9.2k now | ~4.9k | v2.1's real text (4.8–5.0k) | — | Wrong, and already wrong on 2026-10-11: spark-ds41f published 5,836 tok/s at 32K on real source text on 2026-10-05 (r6), and 5,870 on 2026-10-12 (r6h). Fixed on this branch. |

### Elsewhere in the documentation

| Where | Claim | Source and check | Verdict |
|---|---|---|---|
| DESIGN.md 31–32 | "the kit's author reports 87.7 prose / 124.8 code tok/s and ~4.9k tok/s real-text prefill on a *switched* fabric with RoCEnante" | v2.1, `docs/tp4.md` (2026-09-25) | Exact. The version and date were missing; added on this branch. |
| DESIGN.md 34–35, findings.md 147 | "the best published … ~88 prose / ~125–132 code tok/s and ~4.9k tok/s prefill, and eight boxes do no better than four" | Already published by 2026-10-11: v2.2's 89.8 / 132.4 (and 90.2 / 133.2 in MiaAI PR #37); spark-ds41f r6's 5,836 tok/s on real text (2026-10-05); and on eight boxes rhys101's TP8 line, 111.9 / 184.9 (MiaAI PR #39, 2026-09-26), beside his SG18 at 87.30 / 131.72 | The decode bar was about right for four boxes. The prefill bar and the eight-box clause were wrong when written ([m7](#m7-bars-that-were-incomplete-when-written)). Re-worded on this branch. |
| DESIGN.md 72 | "the kit sees ~4.9 tokens per step" (code) | The kit's probe gives 3.87 (v2.2); its Known limits say "~6 on code" | Matches neither ([m5](#m5-the-kits-tokens-per-step-are-quoted-three-ways)). |
| DESIGN.md 73 | "The kit's measured steps (33 ms prose, 39 ms code)" | v2 / v2.1 (33.0 / 39.2 ms switched; 32.4–33.6 / 38.9–39.4) | Right for v2.1; v2.2 is 31–32 / 37–38. A design-time figure, fine as history. |
| DESIGN.md 48 | "the best prior implementation takes 16.9 µs at 10 KiB" | christopherowen's host relay: 16.9 µs (2026-10-03), then 16.2 µs with the package's CuTe kernel (2026-10-05, `docs/oneshot.md`) | Best prior is 16.2 µs; fixed on this branch ([m3](#m3-the-collective-latency-is-misattributed)). |
| DESIGN.md 109–110 | "RoCEnante's pattern with the host relay of christopherowen's 'ring4', which measure 16.9 and 16.2 µs" | Both figures are ring4's: two campaigns of the same package | Fixed on this branch. |
| DESIGN.md 60–62, findings.md 261 | vLLM's DSpark acceptance on GSM8K: 0.724 / 0.516 / 0.301 / 0.180 / 0.096; 2.82 tokens a step (reference 3.51) | [vllm#56797](https://github.com/vllm-project/vllm/issues/56797) (closed 2026-09-18; a maintainer pointed to #57432) | Exact. |
| findings.md 99–106 | The kit: 87.7 / 124.8 c1; 45 varied prompts 58.4 / 94.3 / 71.8; steps 32–34 ms prose, 39 code; filler prefill 4.0–5.9k; real text 4.8–5.0k; 1,011,084-token needle; DSpark ~3 / ~6 tokens a step | `docs/tp4.md` at `cad252b` (v2.1); "~3 … ~6" is from its Known limits | Exact for v2.1. The v2.1 label was missing (the heading names `cad252b`); added on this branch. |
| findings.md 135–142 (survey of 2026-10-11) | rhys101 87.3 / 131.7, 4.4k at 4K, 3.5k at 299K; yunwei37 50.1 / 68.1; 0xSero 200.9 / 7.4k; sparkring 57.0 / 4.4k at 64K; antirez/ds4 18.1 / 826 | rhys101 README and progress.md (2026-09-14): exact. yunwei37 README (2026-09-11): exact. 0xSero (four RTX PRO 6000, labelled): exact. sparkring has since published 70.6 tok/s at 32K context and a 102.4 coding peak on four Sparks (release 2026.10.2–10.3). ds4 now runs V4.1 on one or two Sparks (21.8 tok/s on two) | Dated, so not wrong; sparkring's and ds4's rows are stale ([m7](#m7-bars-that-were-incomplete-when-written)). |
| findings.md 150–151 | "knapcio's TP4 decode step on the ring was within 3 % of the switched one: 33.3 / 38.1 ms against 33.0 / 39.2 ms" | knapcio `docs/switchless-ring.md` and [PR #8](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/8) by rchmagos: 33.3 ms at 2.0 tokens a step against 33.0 at 2.27, and 38.1 at 3.74 against 39.2 at 3.87, on different prompts | Figures exact. The work is rchmagos's, and the tokens a step differ ([m4](#m4-the-ring-step-comparison-omits-tokens-per-step)). Attribution fixed on this branch. |
| findings.md 152–155 | rhys101: a ~1 MB collective 1366.8 → 360.2 µs; RoCEnante took code decode from 111.3 to 134.9 | rhys101 `docs/mia-improvements.md` (1366.8 / 360.2 at 983,040 bytes), `docs/progress.md` (134.91) | Exact. |
| findings.md 166–200 | sudoingX v1.1: 2.70 bpw EXL3 experts; KL 0.0243, top-1 95.57 %; 44.6 tok/s without drafts; verify steps 35.9 ms at 2.09 / 43.3 ms at 4.45; 2.1k prefill; 159.6 s at 256K; "not a peer" | sudoingX/dgx-spark-deepseek-v41 at `cc59c36` (2026-10-12) | Figures exact. The step figures are derived here from its raw JSON and should say so ([m6](#m6-sudoingxs-step-times-are-ours-not-its)). "Not a peer" is fair: lossy weights on half the boxes. |
| findings.md 218–231 | "RoCEnante … On the forwarded 4-ring, BF16: 16.9 µs at 10 KiB (NCCL ~91), 28.0 µs at 60 KiB (~112), 307.7 µs at 2 MiB (~360)"; host relay "16.2 µs at 10 KiB, 271.9 µs at 2 MiB" | christopherowen README and `docs/oneshot.md`: all five are the one-shot host relay on a cable loop. The NIC-forwarded "mesh4" was "measured slower … not recommended" | Misattributed: these were not RoCEnante on a forwarded ring. Fixed on this branch. |
| engine.md 2190–2191 | "rank 0 runs `gpu-clock-cap` at 2200 MHz …, the cap the kit's v2.4 fleet runs at" | Read on 2026-10-13: rank 0 at 2184 MHz under load (`gpu-clock-cap.service` active); rank 1 2489, rank 3 2463 and rank 2 2489 MHz, with no cap unit installed | Implies parity that is not there ([m2](#m2-the-clock-caps-are-not-equal)). |
| engine.md 2235–2236 | "the kit's runs stop where the model does (no `ignore_eos`)" | The 45-prompt harness is unpublished. knapcio's `ab_inboot.py` sends no `ignore_eos` (`max_tokens` 400 by default) | Plausible, not shown ([m10](#m10-an-unverified-claim-about-the-kits-catalog-runs)). |
| engine.md 2224 | "Per token on prose, the kit's probe gives 14.6 ms" | v2.1: 33.26 ms / 2.27. v2.2: 13.7–14.3 ms | v2.1's; label it ([m11](#m11-the-kits-per-token-step-is-v21s)). |

## Survey

### The best published figure on four GB10s at native precision, row by row (as of 2026-10-13)

| README row | dsgb10 (RC5) | Best published | Whose | Lead |
|---|---|---|---|---|
| c1 prose, sparkDash 1.8.8 | 102.2 | 90.2 | knapcio's v2.2 port in [MiaAI PR #37](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/37) (open), 2026-09-25, switched: the median of 90.48 / 89.19 / 90.17. v2.4: 89.9 | +13.3 % |
| c1 code | 148.1 | 133.2 | The same PR, the better of two runs (126.3 / 133.2). v2.2's own table: 132.37, a median of three, with clocks at 3003 MHz (no effective cap) | +11.2 % |
| c1 structured | 186.4 | 158.7 | knapcio v2.4, 2026-10-12, switched, 2200 MHz | +17.5 % |
| c1 JSON | 162.6 | 135.26 | xueyinmei-dot's "R5": v2.3 code on a switchless ring with NIC-forwarded diagonals, measured 2026-10-03 ([PR #11](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/11), closed unmerged and linked from knapcio's `docs/community-results.md`) | +20.2 % |
| Catalogs, prose / structural | 78.5 / 125.0 | 61.11 / 98.96 | knapcio v2.2 (own harness) | +28 % / +26 % |
| A prose step | 27.9 ms at 2.83 tokens (9.9 ms a token) | 31.0 ms at 2.13 (14.6 ms a token); the kit's v2.2 probe, 31.24–32.5 ms at 2.27 (13.7–14.3 ms a token) | [spark-ds41f](https://github.com/christopherowen/spark-ds41f) r6h, its prose prompt with reasoning on; the kit's own probe prompts | not like for like |
| A code step | 34.8 ms at 5.10 tokens (6.8 ms a token) | 34.25 ms at 3.37 (10.2 ms a token); the kit's v2.2 probe, 37.4–37.84 ms at 3.87 (9.7–9.8 ms a token) | spark-ds41f r6h, its code prompt with reasoning on | not like for like: its step is 1.6 % shorter and carries a third fewer tokens |
| Cold prefill, real text, 15k–120k | 8.6k–9.4k (9,130–9,180 at 32k; 9,396–9,445 at 120k) | 5,870 at 30.7k tokens; 5,726 at 113.8k | christopherowen's spark-ds41f r6h, 2026-10-12: vLLM + TileLang on four Sparks in a ring, Python standard-library source, cold and unique, the mean of three. The kit: 4.8k–5.0k (v2.1) | 1.56× at 32k; 1.64–1.65× at 120k |
| sparkDash's own filler prefill, 4k–128k (not in the README) | 8.2k–10.5k | 4.76k–5.97k | knapcio v2.4 (R5: 3.1k–5.9k) | 1.7× |

### Results that beat a README row outside its terms

| Result | Terms it differs in | Figures | Against ours |
|---|---|---|---|
| rhys101's TP8 line of the MiaAI kit ([MiaAI PR #39](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/39), open, 2026-09-26) | **Eight** GB10s, switched 200G RoCE on two rails, TP8/EP2. Same benchmark (sparkDash 1.8.8, greedy, 256 tokens) and official weights | c1 111.9 / 184.9 / 203.9 / 177.2. Prefill: sparkDash filler 6.1k–6.2k at 16k–128k; distinct tokens 5.6k–5.8k at 32k–128k | +9.5 / +24.8 / +9.4 / +9.0 % above ours on c1; below on prefill |
| [TensorFold TP4](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10) (bertholomus, recipe v0.1.1, `6033ea4`, 2026-10-09; [card](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10)) | Four GB10s on "one RoCE network", a Zig engine with exact DSpark. **Lossy weights**: Mia-AiLab's EXL3 2.9 bpw. Its own `kit_bench.py`: greedy, thinking off, 384 or 512 tokens with no forced length, the median of three, and the first token excluded | Set b: code 160.3, prose 95.4, "count from 1 to 200, one per line" 219.5. Set a (512 tokens): 138.1 / 87.2 / 165.6, where the last is a JSON array. Prefill 2.7k–2.9k at 8K–128K | Above our code (148.1), structured (186.4) and JSON (162.6) rows by 8.2 %, 17.8 % and 1.8 %, on other prompts. Below on prose and far below on prefill |
| hushengkai on the kit, four GB10s switched ([MiaAI PR #36 comment](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/36#issuecomment-5930311857), 2026-10-01; [#44](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/44) and [#43](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/43), 2026-10-06) | Prompts, tool and lengths are not named. The same fleet's own 5 × 256-token battery read 134.2 / 137.2 three days earlier ([knapcio #10](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/10)) | "prose 92.4 / code 198.1 tok/s at c1" on "different prompts"; "single-stream decode 195.0 tok/s" | Cannot be compared. A highly repetitive prompt is plausible but not stated |

### Other published results on GB10

Figures as each project publishes them, on its own harness unless the benchmark column says sparkDash.

| Project (source) | Date, version | Boxes, fabric, stack | Weights | One stream, decode | Cold prefill | Above dsgb10 on a README row? |
|---|---|---|---|---|---|---|
| [knapcio TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) (the MiaAI kit's TP4 line) | v2.4, 2026-10-12 (`27616be`); v2.3 2026-09-29; v2.2 2026-09-25 | 4, switched RoCE, SGLang + b12x + RoCEnante | official, native | sparkDash c1: v2.4 89.9 / 131.0 / 158.7 / 128.4; v2.3 89.7 / 131.9 / 157.8 / 130.4; v2.2 89.8 / 132.4 / 156.9 / 124.3 | filler 4.8k–6.0k; real text 4.7k–4.9k (v2.2); 388K 4,976 (v2.4) | no |
| knapcio's v2.2 port to MiaAI ([PR #37](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/37), open) | 2026-09-25 | as above | official, native | sparkDash c1 prose 90.2 (median of three), code 126.3 / 133.2 (two runs); steps 31.25 / 37.45 ms warm | sparkDash 5,775 at 128k | no |
| [MiaAI-Lab kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) `docs/tp4.md` | v2.1, 2026-09-25 (`cad252b`); no newer TP4 results on its main branch | as above | official, native | sparkDash c1 87.7 / 124.8 / 152.4 / 118.9 | real text 4.2k–5.0k | no |
| rchmagos, ring ([PR #8](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/8), `docs/switchless-ring.md`) | 2026-09-25, v2 line | 4, switchless ring, RoCEnante over NIC hairpin forwarding | official | sparkDash c1 80.9 / 120.5 / 150.2 / 134.2 | filler 5.4k–5.6k | no |
| xueyinmei-dot "R5" ([PR #11](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/11)) | measured 2026-10-03, v2.3 | 4, switchless ring, NIC-forwarded diagonals | official | sparkDash c1 84.44 / 129.86 / 156.14 / 135.26 | filler 3.1k–5.9k | no |
| voktolom ([forum 382725/196](https://forums.developer.nvidia.com/t/382725/196)) | 2026-09-30, knapcio v2.3 | 4, MikroTik CRS504 switch at 100G, one HCA a node | official | sparkDash prose c1 86.3 (v2.1 on the same boxes: 82.7); own prompts: prose 72.3 greedy, code 127.0, code edit 135.1, JSON rewrite 136.1 | unique text 4.6k–4.8k at 17K–138K | no |
| hushengkai ([knapcio #10](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/10)) | 2026-10-03, v2.3 | 4, switched | not stated (the kit's default is the official) | own battery: 134.2 / 137.2 (5 × 256 tokens); see the table above for the figures on unnamed prompts | 199K cold 4,623–4,689 | no |
| ecohash-co ([MiaAI #38](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/38)) | 2026-09-25, v2.1 line | 4, switched | not stated (as above) | own prompts: prose 57.0, code 131.5 | — | no |
| [ZackO2o](https://github.com/ZackO2o/DeepSeek-V4.1-Flash-4x-GB10-1M-Full-Recipe/blob/main/results/2026-09-28-sglang-switchless-ring-tp4.md) | 2026-09-28, canary-roce line | 4, switchless ring, clocks pinned at 2400 MHz | official | `decode_window.py`, greedy code, ~3.2k-token answer: 104.10 (~1k-token answer: 86.0–87.5) | random words 4.6k–4.7k at 16k–64k | no: a code prompt, below our code row (148.1) |
| [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f) | r6h, 2026-10-12 (`e9d79b5`); r6, 2026-10-05 | 4, switchless ring, vLLM + TileLang + one-shot RoCE collectives | official, native (FP32 logits and mHC as in the reference) | own bench, temperature 0, 256 tokens, reasoning on: prose 65.9 at 31.0 ms and 2.13 tokens a step; code 92.3 at 34.25 ms and 3.37 | Python source, cold: 5,870 / 5,726 / 5,496 / 5,074 / 4,391 at 31K / 114K / 231K / 482K / 904K | no; it has the best published real-text prefill and the shortest code step (above) |
| [nero-](https://github.com/nero-/deepseek-v41-flash-gb10-ring) (`364eb87`, `results/20260927/final-ring.json`) | 2026-09-27, the MiaAI line (since replaced by vLLM) | 4 (two Sparks, two GX10s), switchless ring with NIC-forwarded diagonals | official (`fb2764a5`) | own `ringbench.py` (greedy, thinking off, 256 tokens, median of 5): 60.7 / 72.2 / 77.1 / 85.0 | real text after an untimed warm-up pass: 5.0k–5.4k at 13K–100K | no |
| [LuZ](https://github.com/luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring) | 0.2.8–0.2.9 | 4, switchless ring, SGLang + b12x | official | own DE bench: C1 peak 89.19 (code) | pure prefill 5.0k–5.8k (5,823 peak at 64K × 2 streams; 5,042 at 512K) | no |
| [sparkring](https://github.com/FujitsuPolycom/sparkring) | release 2026.10.2–10.3 (records 2026-10-12) | 4 (ring of four), 8 (ring of eight) | FP8 / MXFP4 | llm-inference-bench at T 1, 32K context: 70.6 (four), 73.7 (eight); coding peak 102.4 (four), 133.8 (eight) | 32K: 4,388 (four), 4,851 (eight) | no |
| [yunwei37](https://github.com/yunwei37/dgx-spark-4-ring-no-switch) | 2026-09-11 | 4, ring, vLLM | official | 50.14 mean, 68.06 code | — | no |
| ChrisLou-bioinfo ([MiaAI #40](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/40)) | 2026-09-27 | 4, ring, the MiaAI line | not stated | "decode 54–56 tok/s" end to end | ~4,470 on a 283k cold prompt | no |
| [rhys101 SG18](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8) | 2026-09-14 | **8**, switched, SGLang TP8/EP4 + RoCEnante | official | coding 131.72 (200 tokens), prose 87.30; SG17 record 134.91 code | 4,428 at 4K … 3,458 at 299K | no |
| [rhys101 TP8](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/39) | 2026-09-26 | **8**, switched | official | sparkDash c1 111.9 / 184.9 / 203.9 / 177.2 | sparkDash filler 6.1k–6.2k | **yes, on eight boxes** (above) |
| [TensorFold TP4](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10) | 2026-10-09 | 4, RoCE | **lossy**: EXL3 2.9 bpw | own set b: code 160.3, prose 95.4, count 219.5 | 2.7k–2.9k | **yes, at lower precision on its own prompts** (above) |
| [sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41) | v1.1, 2026-10-12 | **2**, one cable, vLLM + own kernels | **lossy**: EXL3 experts at 2.70 bpw (KL 0.0243) | own set b, 384 tokens: code 103.0, prose 58.3, structured 141.3; 44.6 without drafts | 2.1k at 8K–32K | no |
| [MiaAI EXL3 2×](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks); [TensorFold](https://github.com/MiaAI-Lab/TensorFold) v0.6.0 with that 2.9 bpw pack, as measured in sudoingX's README | 2026-09-19; 2026-10-12 | **2**, ExLlamaV3; TensorFold | **lossy**: EXL3 2.9 bpw | EXL3 kit C1 31.45–40.23; TensorFold prose 62.1, code 100.5, structured 139.0 | EXL3 kit 32K 1,138; TensorFold 2,163 / 2,276 at 8K / 32K | no |
| [antirez/ds4](https://github.com/antirez/ds4/blob/main/docs/DGX_SPARK.md) | 2026-09 | 1–2, RDMA | **lossy**: Q2 GGUF | 21.8 on two Sparks | 32K 411 | no |
| [0xSero](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000); [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt) | — | **4 × RTX PRO 6000**; **2 RTX PRO 6000 + 4 Sparks** | native | 200.9; code 179.86 | 7.4k at 32K; 8,355 | not GB10 only |

Far below every row, and not tabled: forks and recipes of the kit (aotsukiqx's sparkDash 1.8.9 re-implementation, 83.7 prose / 125.2 code and 4,973 real-text at 139K; cqingwang 82.4 prose; colonel-otto 126.7 / 132.0 per stream on a sampled word list; Carlos Duque 74.0; koldfrontier 48.6 / 97.3 / 90.3); vLLM and SGLang recipes (AidenLab 110 / 94 / 37 counting / code / prose, 0xTank, bertholomus, tonyd2wild); [Spark Arena](https://spark-arena.com/) entries (tg128 27.9–43.7); OpenZeka's whitepapers (29.5 at C=1); Patrick Moorhead's four-Spark ring as reported by [Wccftech](https://wccftech.com/the-552b-deepseek-v4-1-flash-model-offers-a-peak-output-of-494-tokens-second-when-powered-by-an-at-home-rig-spanning-4x-nvidia-dgx-spark-units/) (~58 prose, ~96 code, ~4.8k prefill); neko-legends' pruned-EXL3 TensorFold (129.1 code); and the other reports in the NVIDIA forum's [DeepSeek V4.1 Flash thread](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash/382725). NVIDIA publishes no V4.1 figure for DGX Spark: its only V4.1 release is the [NVFP4 checkpoint](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4) for GB300, with accuracy only, and the Spark playbooks do not cover V4.1. ZackO2o's [cross-project table](https://github.com/ZackO2o/DeepSeek-V4.1-Flash-4x-GB10-1M-Full-Recipe/blob/main/results/2026-09-28-cross-project-4x-gb10-reference.md) credits nero- with 81.8 / 76.7 / 78.8 tok/s; nero-'s archive (`RESULTS-SGLANG-ARCHIVE.md`) says those are Qwen3.8 TP2 numbers.

**Is "the best published four-GB10 numbers" true today?** Only with qualifiers. On sparkDash 1.8.8 at the
checkpoint's native precision, the MiaAI kit's TP4 line holds the best published four-GB10 c1 and catalog
figures, and dsgb10 is above all of them. But:
- the README's column is v2.4's figures, not the best per row (prose and code from v2.2 in PR #37, JSON from
  R5);
- the best published real-text prefill and the shortest code step are spark-ds41f's, not the kit's;
- at lower precision, TensorFold TP4 publishes higher code and counting rates on four GB10s, on its own
  prompts;
- the claim carries no date ([M4](#m4-best-published-is-not-the-best-per-row-and-has-no-date)).

On eight GB10s, rhys101's TP8 line of the same kit beats every c1 row. The README makes no eight-box claim,
but DESIGN.md and findings.md did ([m7](#m7-bars-that-were-incomplete-when-written)).

**The newest MiaAI version** is v2.4 (knapcio's repository, `27616be`, 2026-10-12, "decode unchanged" against
v2.3), and the README cites its numbers exactly. MiaAI-Lab's own main branch still holds v2.1 (2026-09-25);
the v2.2 port (PR #37) and the TP8 line (PR #39) are open pull requests there. No v2.5 or v3 exists in either
repository, their forks or the forum.

## Precision parity

**What the two stacks compute.** Both serve the checkpoint's own formats: MXFP4 routed experts, FP8 e4m3
dense weights with 32×32 E8M0 block scales, a bf16 head. Neither re-quantises a weight.

Both quantise activations as DeepSeek's reference kernel does: e4m3 per 32 values, with the power-of-two
scale rounded up.
- FlashInfer's MXFP8 quantiser (v0.6.18) and b12x's W4A8 MoE (`a7d7d29`) both use the ceil rule. The kit's
  block-to-MX scale conversion is bit-exact.
- b12x applies the route weights after the FC2 activation's quantisation, not before as the reference does;
  this is a rounding difference with no systematic direction.

Both verify drafts exactly under greedy decoding, the kit through SGLang's `AcceptGreedy` and a certified head
that returns the stock argmax.

Both use the Decoder SWA Bounded Replay that DeepSeek post-trained the model for: layers 21–39 run only over
a prompt's last 128 tokens. The kit's `boot.py` always passes `--enable-decoder-swa-bounded-replay`; CED is
dsgb10's serving default ([docs/ced.md](../ced.md)). So the prefill rows compare the same approximation.

The kit's `--enable-deepseek-v4-fp4-indexer` does nothing on V4.1. Only ratio-4 layers read it, and V4.1's
indexer is FP4 by design, as the kit's `docs/tp4.md` says.

**Where they differ, the kit is the less exact one.** On GB10, SGLang (`f80c91a4`) runs attention through
FlashInfer's SM120 sparse-MLA kernel, which quantises the attention query (batches of up to 64 query tokens,
so every decode and verify step) and the attention weights P (always) to FP8. The reference and dsgb10 keep
both in bf16. The same path re-rounds the FP4 compressed KV into an FP8 layout. That costs the kit memory
(584 against 288 bytes a token) and is a second rounding the design does not have.

The FP8 attention makes the kit faster. At sparkDash's short prompts, attention is a few percent of a step,
so it moves the decode rows by a few percent at most, in the kit's favour. dsgb10 matches the reference's
numerics except for summation order; its exact weight codes change no bit. **On precision the comparison is
fair, and slightly conservative for dsgb10.**

**Not precision peers:**
- TensorFold TP4 on four GB10s, and MiaAI's two-box EXL3 kit and TensorFold TP2: Mia-AiLab's EXL3 at
  2.9 bpw. On four GB10s TensorFold's weight file is 56.2 GB a node, against the native checkpoint's
  ~72 GiB (~77 GB) a rank at TP=4 (DESIGN.md), so each step reads fewer bytes.
- sudoingX v1.1: EXL3 experts at 2.70 bpw, KL 0.0243 against the native model.
- neko-legends' TensorFold lane: EXL3 with experts pruned at top-p 0.85.
- antirez/ds4: Q2.
- sparkring's optional QAD checkpoint, which is a fine-tuned model.

spark-ds41f is a precision peer. It follows the reference more closely than the kit: FP32 logits, an exact
FP32 mHC projection, and the shared expert's SwiGLU in FP32. The deviation it declares is WO-A's FP8
activations.

**The checkpoint: dealignai's variant.** dsgb10's figures, and those of the SGLang kit measured on this ring
(its `.env.tp4` points `MODEL_DIR` at the same directory), are on
[`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8)
at `d61c59ea` (2026-09-11, MIT licence like the base). The MiaAI kit, and every native-precision figure
surveyed above that names its weights, use [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
knapcio pins revision `fb2764a5` (2026-09-10), and rank 0's `/data/models/DeepSeek-V4.1-Flash` is that
revision. One far slower recipe uses dealignai's variant too, and says so
([0xTank R37](https://huggingface.co/0xTank/DeepSeek-V4.1-Flash-FP8-R37-4x-GB10): "The benchmarks on this
card used the dealignai checkpoint").

What dealignai changed:
- **Method.** The card calls it "permanent weight-level abliteration … No custom `model.py`, no runtime
  hooks, no steering vectors". The commit that made it is "CRACK v1: MPOA per-layer refusal ablation"
  (`d7d98e1`, 2026-09-10).
- **The cost.** The card reports MMLU-14k 86.96 → 82.74 % (−4.22 pp; −1.1 pp outside its ethics subjects) and
  HarmBench-320 compliance at 100 %.
- **"FP8".** The card's quantisation line is "FP8 (`e4m3fn`) weights with E8M0 block-scale [32, 32], FP4
  routed experts — native, unchanged". The suffix names the native format, which the official repository
  ships too; nothing was converted to FP8.

What the bytes show:
- **Shards.** By the Hugging Face hashes, 21 of the 48 shards (1–12 and 40–48) are the official's, byte for
  byte, and the 27 others (13–39) differ. `config.json`, the shard index and the tokenizer files are
  identical. rank 0's variant is the official copy's 21 shards (hard links) plus dealignai's 27, whose sha256
  match `d61c59ea`.
- **Tensors.** In shards 13–39, which hold layers 10–36, the safetensors headers are identical: names, dtypes,
  shapes, offsets.
- **Values.** Every byte was compared except the routed experts' FP4 values, which were sampled; all of their
  scales were compared. Exactly two tensor kinds differ: each layer's attention output projection
  `attn.wo_b.weight` (53.7 % of its bytes) and its shared expert's `ffn.shared_experts.w2.weight` (48.5 %).
  Both are FP8, and both changed in all 27 layers. Their E8M0 scales are unchanged, so the edit was re-rounded
  into e4m3 at the original block scales.
- **Unchanged.** Every other tensor is identical: all 384 experts' scales in every layer, the sampled expert
  values, the router, the norms, the mHC parameters, the indexer, the compressor, and `wq_a`, `wq_b`, `wkv`
  and `wo_a`. So are the drafter (`mtp.*`), the embeddings, the head, the Engram tables and the vision
  tower, which sit in unchanged shards.

No speed-relevant structure changed: the expert count (384, top-6, one shared), the vocabulary and the drafter
are the official's. So **this is not a blocker**.

**Why the speeds still compare.** The same tensors in the same formats and shapes mean the same bytes are
streamed per token and the same kernels run. dsgb10's exact weight codes apply only to the experts' scales and
the head's exponents, which are unchanged, so even its coded bytes are the same.

**What does not carry over.** The greedy text does not, and with it the drafts accepted per step:
- The drafter was trained on the official model.
- Its input is the target's hidden state, which the edits in layers 10–36 shift.
- Tokens per step drive most of each decode figure (prose: 2.83 tokens a step for ours, 2.27 for the kit's
  probe).

So the decode rows carry a confound of unknown sign and size until dsgb10 is measured on the official
weights. Prefill does not use drafts, and its rows should carry over within noise.

## Findings

Severity: **blocker**, a comparison that is false or a variant that changes speed-relevant structure (none
found); **major**, a claim that stands but is presented unfairly, which must change before release;
**minor**, wording, dating or attribution. "Now" quotes master at `191c30f`.

### M1. The checkpoint is not disclosed

Major. Neither README says which weights the figures are on. DESIGN.md's Target names the variant, but no
comparison links to it ([Precision parity](#precision-parity)).

- **Root `README.md` 54–56. Now:** "The MiaAI kit's figures are from its own documents, not measured here.
  A stack that sums in another order follows another greedy text on the same prompt, so the single-prompt rows
  compare slightly different texts; the catalogs and the step times are the fairer measures."

  **Replace with:** "The MiaAI kit's figures are from its own documents, not measured here.
  dsgb10's figures, and the SGLang kit's measured here, are on dealignai's DeepSeek-V4.1-Flash-UNCENSORED-FP8
  (d61c59ea), an abliterated copy of the official checkpoint with the same tensors, formats and shapes (only
  the FP8 values of attention's wo_b and the shared expert's w2 in layers 10–36 differ), so the same bytes
  stream per token; the MiaAI kit's figures are on the official weights (fb2764a5), whose greedy texts and
  draft acceptance differ." The rest of the bullet is [m1](#m1-single-prompt-rows-understated)'s and
  [M2](#m2-the-step-rows-mix-versions-and-prompt-sets)'s.
- **Configuration `README.md` 46:** the same sentence, after "not re-measured here".
- **Once dsgb10 is re-measured on the official weights,** the sentence becomes: "dsgb10's figures are on
  deepseek-ai/DeepSeek-V4.1-Flash at revision fb2764a5, the revision the MiaAI kit pins. The SGLang kit's
  baseline above was measured on dealignai's abliterated copy of it (same tensors and formats)."
- **`DESIGN.md` 8–10. Now:** "It is the same architecture and format as the original; only shards 13–39
  differ."

  **Replace with:** "It is the same architecture and format as the original: only shards 13–39 differ, and in
  them only the FP8 values of `attn.wo_b` and `ffn.shared_experts.w2` of layers 10–36 (scales, shapes, the
  drafter and every other tensor byte-identical; [docs/review/external.md](#precision-parity))."
- **Also:** the configuration README's Reproduce section benchmarks `--model deepseek-v4.1-flash-uncensored`;
  an outsider reproducing the table needs to know which weights that is.

### M2. The step rows mix versions and prompt sets

Major. The kit's milliseconds are v2.1's (`docs/tp4.md`) and its tokens a step are v2's or v2.2's. v2.2's
probe is faster: 31.24–32.5 ms at 2.274 on prose and 37.4–37.84 ms at 3.865 on code (knapcio
`validation-20260925-v22.txt`; CHANGELOG: "Decode step c1 prose 33.26 → 31.24–32.5 ms, code 38.88 →
37.4–37.84 ms"). Using v2.1's figure flatters our lead by 3–4 %.

The prompts also differ. Ours is sparkDash's single c1 prompt. The kit's is its own step probe, "3 prompts ×
400 tokens" per its `history.md`, and its own `switchless-ring.md` says those prompts are "not published".
More tokens a step means more verify rows, so a step comparison at 2.83 against 2.27 tokens is not like for
like in either direction. The kit's are not the shortest published steps either. spark-ds41f r6h takes 31.0 ms
at 2.13 tokens on prose and 34.25 ms at 3.37 on code, on its own prompts with reasoning on. Its code step is
shorter than ours (34.8 ms at 5.10), and per token it is 1.5× slower (10.2 against 6.8 ms).

- **Root `README.md` 37–38. Now:** `| a decode step, prose | **27.9 ms at 2.83 tokens** | 30.1 ms at 2.83 | 32.4–33.6 ms at 2.27 |`
  and `| a decode step, code | **34.8 ms at 5.10 tokens** | 38.0 ms at 5.10 | 38.9–39.4 ms at 3.87 |`

  **Replace with:**
  `| a decode step, prose: ours on sparkDash's prompt, the kit's on its own probe prompts | **27.9 ms at 2.83 tokens** | 30.1 ms at 2.83 | 31.2–32.5 ms at 2.27 (v2.2) |`
  and `| a decode step, code: likewise | **34.8 ms at 5.10 tokens** | 38.0 ms at 5.10 | 37.4–37.8 ms at 3.87 (v2.2) |`
- **Root `README.md` 56. Now:** "the catalogs and the step times are the fairer measures."

  **Replace with:** "the kit's documents recommend comparing step times and many-prompt runs across stacks,
  but its step probe and its catalog harness are its own and unpublished, and its steps carry fewer tokens,
  so no decode row is like for like; the prefill row is the one least affected."
- **Configuration `README.md` 48–49. Now:** "the catalogs and the time per step (27.9 ms at 2.83 tokens on
  prose, 34.8 ms at 5.10 on code, against the kit's 31–34 and 37–39 ms) are the fairer measures"

  **Replace with:** "the time per step is 27.9 ms at 2.83 tokens on sparkDash's prose prompt and 34.8 ms at
  5.10 on its code prompt, against the kit's 31.2–32.5 ms at 2.27 and 37.4–37.8 ms at 3.87 on its own,
  unpublished probe prompts (v2.2); neither that nor the catalogs is like for like"
- **Worth doing:** knapcio's `ab_inboot.py` publishes three prose and three code prompts ("the step_probe.py
  way"). If those are the probe's prompts, running them through dsgb10 at 400 tokens would give a step
  comparison on the same prompts.

### M3. The catalog row compares two harnesses

Major. The README introduces the table as measured "with that project's own benchmark (sparkDash 1.8.8 …
256 tokens…)". That is true of our catalog runs, but not of the kit's 61.1 / 99.0. Those come from the kit's
own "45 varied prompts" harness, which neither repository publishes.

Our runs force 256 tokens with `ignore_eos` and report the mean of per-prompt rates. For the kit's harness,
only knapcio's in-boot A/B tool is public: it stops at EOS, defaults to 400 tokens, and reports medians. Our
structural medians are 128.1 and 128.3 against means of 124.9 and 125.0 (RC5's two sessions), so the statistic
alone is worth ~3 % on that row.

- **Root `README.md` 36. Now:** `| its prompt catalogs: prose / structured | **78.5 / 125.0** | 72.9 / 115.3 | 61.1 / 99.0 (v2.2) |`

  **Replace with:** `| sparkDash's prose / structural catalogs, one stream (ours: 256 tokens each, the mean; the kit's: its own harness) | **78.5 / 125.0** | 72.9 / 115.3 | 61.1 / 99.0 (v2.2) |`
- **Configuration `README.md` 43**, which has no local-kit column. **Now:**
  `| its prompt catalogs at one stream: prose / structured | **78.5 / 125.0** | 61.1 / 99.0 (v2.2) |`

  **Replace with:** `| sparkDash's prose / structural catalogs, one stream (ours: 256 tokens each, the mean; the kit's: its own harness) | **78.5 / 125.0** | 61.1 / 99.0 (v2.2) |`

### M4. "Best published" is not the best per row, and has no date

Major. The column is the kit's v2.4 figures, but the best published figures on four GB10s at native
precision are, row by row ([survey](#survey)):
- **c1 on sparkDash 1.8.8:** **90.2** (v2.2, MiaAI PR #37) / **133.2** (the same PR) / 158.7 (v2.4) /
  **135.26** (R5, a community ring run of v2.3, 2026-10-03). Against those our lead is +13.3 % on prose (not
  +13.7 %), +11.2 % on code (not +13.1 %), +17.5 % on structured and +20.2 % on JSON (not +26.6 %).
- **Cold prefill on real text:** **5,870 tok/s at 30.7k tokens and 5,726 at 113.8k**, from christopherowen's
  spark-ds41f r6h (2026-10-12, vLLM on four Sparks in a ring), not the kit's 4.8k–5.0k. Our lead is ~1.6×,
  not 1.7–1.9×.
- **The code step:** spark-ds41f r6h takes 34.25 ms, shorter than both the kit's and ours, though at a third
  fewer tokens ([M2](#m2-the-step-rows-mix-versions-and-prompt-sets)).

Two terms are also missing:
- **Precision.** The claim holds only at the checkpoint's native precision. TensorFold TP4 publishes 160.3
  code and 219.5 counting tok/s on four GB10s, with 2.9-bpw EXL3 experts and its own prompts.
- **A date.** The v2.4 figures are in knapcio's repository, which the README never links; MiaAI-Lab's main
  branch stops at v2.1.

The fixes:
- **Root `README.md` 29–31. Now:** "Against the best published four-GB10 numbers, the MiaAI kit on a switched
  RoCE fabric, with that project's own benchmark ([sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8:
  its prompts, 256 tokens, greedy, one stream):"

  **Replace with:** "The table compares dsgb10 with the MiaAI kit's TP4 line on a switched RoCE fabric
  ([knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4),
  v2.4 of 2026-10-12 unless marked), whose decode figures were the best published for four GB10s at the
  checkpoint's native precision on 2026-10-13. It uses that project's own benchmark
  ([sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8: its prompts, 256 tokens, greedy, one stream).
  No published native-precision four-GB10 figure we found beats dsgb10's on any row
  ([survey](#survey)); the nearest is a code step 1.6 %
  shorter than ours that carries a third fewer tokens. The best per c1 type are 90.2 / 133.2 (v2.2) / 158.7 /
  135.3 (a community ring run of v2.3), and the best real-text prefill is 5.7k–5.9k (christopherowen's
  spark-ds41f, vLLM). A 2.9-bpw quantisation publishes higher code and counting rates on its own prompts."
- **Root `README.md` 39 and configuration `README.md` 44** (with [m8](#m8-the-prefill-row-uses-different-texts)):
  in the kit's column, "4.8k–5.0k tok/s (v2.1)" becomes "4.8k–5.0k tok/s (v2.1; the best published,
  spark-ds41f's: 5.7k–5.9k)".
- **Configuration `README.md` 36–38. Now:** "Against the best published four-GB10 numbers (the MiaAI kit on a
  switched RoCE fabric), measured with that project's own benchmark,
  [sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8, on its own prompts (256 tokens, greedy, one
  stream; ours the mean of two sessions):"

  **Replace with:** "The table compares dsgb10 with the MiaAI kit's TP4 line
  ([knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4),
  on a switched RoCE fabric), whose decode figures were the best published for four GB10s at the
  checkpoint's native precision on 2026-10-13. It uses that project's own benchmark,
  [sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8, on its own prompts (256 tokens, greedy, one
  stream; ours the mean of two sessions, the kit's the better of two sweeps). No published native-precision
  four-GB10 figure we found beats ours on any row; the nearest is spark-ds41f's code step, 1.6 % shorter but
  carrying a third fewer tokens, and its 5.7k–5.9k is the best real-text prefill
  ([survey](#survey))."
- **Configuration `README.md` 74:** "the best published ~4.9k" is fixed on this branch
  ([m7](#m7-bars-that-were-incomplete-when-written)).

### m1. Single-prompt rows understated

Root `README.md` 55–56. **Now:** "so the single-prompt rows compare slightly different texts". **Replace with:**
"so the single-prompt rows compare different texts (the kit's README: a rounding change 'can flip a near-tie
early in the 256 tokens and move c1 by ±10 % without changing speed')". The same applies in the configuration
README, lines 47–48.

### m2. The clock caps are not equal

`docs/engine.md` 2190–2191. **Now:** "rank 0 runs `gpu-clock-cap` at 2200 MHz (homelab `docs/host.md`), the cap
the kit's v2.4 fleet runs at."

**Replace with:** "rank 0 runs `gpu-clock-cap` at 2200 MHz (homelab `docs/host.md`), the cap all four of the
kit's v2.4 boxes run at. rank 1, ranks 2 and 3 have no cap and run at 2.46–2.49 GHz under load (read
2026-10-13). On four Sparks the cap cost 2.2 % of single-stream decode
([hushengkai, knapcio#10](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/issues/10)), so
this favours us by at most that, and less if capped rank 0 sets the ring's pace."

Prefill runs the other way: the kit's v2.1 prefill figures were at uncapped clocks and our real-text runs had
rank 0 capped.

### m3. The collective latency is misattributed

All five figures are christopherowen's one-shot all-reduce with a bidirectional host relay ("ring4") on four
Sparks in a cable loop:
- 16.9 µs at 10 KiB, 28.0 at 60 KiB and 307.7 at 2 MiB (README, "Measured"; spark-ds41f, 2026-10-03);
- 16.2 µs at 10 KiB and 271.9 at 2 MiB (`docs/oneshot.md`; the package's CuTe kernel, 2026-10-05).

Its NIC-forwarded "mesh4" was slower and is not recommended. RoCEnante's protocol is the same, but these
measurements are not RoCEnante on a forwarded ring. Fixed on this branch: configuration `README.md` 66,
`DESIGN.md` 48 and 109–110, `docs/ring.md` 10–11, `docs/findings.md` 218–231.

Not changed: our 12.2–12.3 µs is "back to back in a graph", and theirs is the median of the slowest rank per
sample of 256 graph replays. These are close, not identical. README and DESIGN say 12.3, ring.md 12.2.

### m4. The ring step comparison omits tokens per step

`docs/findings.md` 150–151. The ring work is rchmagos's
[PR #8](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/8); the attribution is fixed on
this branch. **Still to add:** after "33.0 / 39.2 ms", add "(2.0 and 3.74 tokens a step on the ring against
2.27 and 3.87 switched, on different prompts; per token the ring's prose step was ~15 % slower)".

### m5. The kit's tokens per step are quoted three ways

`DESIGN.md` 72 says "~4.9 tokens per step" on code. The README says 3.87 (the kit's probe, v2.2), and
`findings.md` 105 says "~6 on code" (the kit's Known limits). **Replace DESIGN's with:** "(the kit's probe
sees 3.87 tokens per step on code, v2.2; its docs give ~6 as the bound)".

### m6. sudoingX's step times are ours, not its

`docs/findings.md` 181. **Now:** "| a verify step, prose / code | 35.9 ms at 2.09 tokens / 43.3 ms at 4.45 |
…". **Replace the cell with:** "35.9 ms at 2.09 tokens / 43.3 ms at 4.45 (derived here from its raw JSON)".

### m7. Bars that were incomplete when written

These lines all quote the bar of the 2026-10-11 survey:
- DESIGN.md 34–35: "The best published results on GB10s are ~88 prose / ~125–132 code tok/s and ~4.9k tok/s
  prefill, and eight boxes do no better than four";
- DESIGN.md 77–78: "1.1–1.3× the best published numbers";
- DESIGN.md 175: "Decode headroom over the best published numbers is modest";
- DESIGN.md 213: "the best published ~4.9k";
- DESIGN.md 215: "the best published ~88 prose / ~125 code tok/s";
- findings.md 147–148;
- the configuration README, line 74: "the best published ~4.9k";
- engine.md 3129.

Even on 2026-10-11 that bar was incomplete:
- v2.2 (89.8 / 132.4) and its port in PR #37 (90.2 / 133.2) were out on 2026-09-25.
- spark-ds41f r6 had 5,836 tok/s on real text on 2026-10-05.
- On eight boxes, rhys101's TP8 line had 111.9 / 184.9 on 2026-09-26, so "eight boxes do no better than four"
  was wrong.

These are re-worded on this branch where they state the bar: DESIGN.md 34–35, findings.md 147–148 and the
configuration README's line 74.

**For the milestone rows (M5, M7) and DESIGN.md 77–78 and 175, add after "the best published":** "(the best we
had found by 2026-10-11; on 2026-10-13 the best published on four GB10s at native precision are 90.2 prose /
133.2 code on sparkDash c1 and ~5.9k real-text prefill)".

findings.md 135–142's survey is dated 2026-10-11. Its sparkring and ds4 rows are stale:
- sparkring has since published 70.6 tok/s and a 102.4 coding peak on four Sparks, and 133.8 on eight;
- ds4 now runs V4.1 on one or two Sparks (21.8 tok/s on two).

It also lacks spark-ds41f and rhys101's TP8 line.

### m8. The prefill row uses different texts

Root `README.md` 39 and configuration `README.md` 44. **Now:** "| cold prefill, real text, 15k–120k tokens |".
**Replace with:** "| cold prefill, real text (each side's own docs and code), 15k–120k tokens |".

Also worth one clause: v2.4 moved to 8192-row chunks (+20 % at 388K: 4,976 tok/s) but published no real-text
figure at 15k–120k, so the kit's column there is v2.1 at 4096-row chunks.

### m9. The root summary row loses "as it served here"

Root `README.md` 10. **Now:** "decode 76–150 tok/s with speculative drafts (the SGLang kit: 47–107), cold
prefill 8.5–9.4k tok/s (kit: 4.2k)".

**Replace with:** "… (the SGLang kit as it served on this ring, NCCL without RoCEnante: 47–107) … (that kit:
4.2k)".

The same kit with RoCEnante on a switchless ring publishes 80.9–84.4 tok/s on prose c1 (PR #8, R5), so the
local baseline is not the kit's best ring configuration.

### m10. An unverified claim about the kit's catalog runs

`docs/engine.md` 2235–2236. **Now:** "The 11 others are not published, and the kit's runs stop where the model
does (no `ignore_eos`)."

**Replace with:** "The 11 others are not published, and neither is the harness; knapcio's `ab_inboot.py`,
which loads these two catalogs, sends no `ignore_eos` and stops at 400 tokens by default."

### m11. The kit's per-token step is v2.1's

`docs/engine.md` 2224. **Now:** "Per token on prose, the kit's probe gives 14.6 ms." **Replace with:** "Per
token on prose, the kit's probe gives 14.6 ms (v2.1; v2.2: 13.7–14.3 ms)."

## Applied on this branch

Two commits, separate from this report; the line numbers are master's at `191c30f`. These are small and
certain: versions, dates, attributions, and two bars the survey showed to be wrong. The comparisons
themselves are untouched; the majors and the other minors are left to the orchestrator.

Commit `034cdd9`:
- `DESIGN.md` 31: "the kit's author reports" → "the kit's documentation (v2.1, 2026-09-25) reports".
- `DESIGN.md` 48: "the best prior implementation takes 16.9 µs at 10 KiB" → "the best prior one we found
  takes 16.2 µs at 10 KiB (christopherowen's host relay)".
- `DESIGN.md` 109–110: the sentence names both figures as ring4's, with their dates.
- Configuration `README.md` 66: "(RoCEnante: 16.9)" → "(christopherowen's host relay: 16.2–16.9)".
- `docs/ring.md` 10–11: the same attribution.
- `docs/findings.md`:
  - 102 and 137 get the "v2.1" label;
  - 150 credits rchmagos's PR #8;
  - 218–231 move the 16.9 / 28.0 / 307.7 µs figures to the host relay, date both campaigns, and credit the
    proxy idle-spin fix (−8.6 % per step) to PR #8, where it was measured.

Commit `c121b99` ([m7](#m7-bars-that-were-incomplete-when-written)):
- `DESIGN.md` 34–35 and `docs/findings.md` 147–148: the bar becomes "the best … that we had found by
  2026-10-11", and the eight-box clause goes. Each now points to this survey for the higher figures. (The
  first commit had dated them "as of 2026-10-11", which the survey then showed to be wrong.)
- Configuration `README.md` 74: "the best published ~4.9k" → "the best published on four GB10s as of
  2026-10-13, ~5.9k on real text".

## Sources

Read on 2026-10-13. Commits are the ones read. At the end of the review (08:42 UTC) every cloned
repository's HEAD was checked against GitHub again, and none had moved.

- knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 at `27616be` (2026-10-12): README "Current results", CHANGELOG,
  `docs/history.md`, `docs/switchless-ring.md`, `docs/community-results.md`, `docs/results/validation-2026092{5,9}-v2*.txt`
  and `validation-20261012-v24.txt`, `scripts/ab_inboot.py`, `.env.tp4.example` (`HF_REVISION`). Issues and PRs
  #8, #10, #11.
- MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks at `cad252b` (2026-09-25): `docs/tp4.md`, `boot.py`, adapters,
  `benchmarks/decode_window.py`. Issues #21, #38, #40, #41, #43, #44; pull requests #36, #37 (knapcio's v2.2
  port) and #39 (rhys101's TP8 line), with their comments.
- MiaAI-Lab/sparkDash: 1.8.8 = `754f40a` (2026-09-22). The current release is 2.40.0 (2026-10-11); its prompt
  file is unchanged since 1.8.8, and its prefill bench was rewritten, so 2.x prefill is not comparable with
  1.8.8.
- christopherowen/spark-ds41f at `e9d79b5` (2026-10-13): README "Performance" and "Benchmarking",
  `manifests/benchmarks/2026-10-12-karmic-kraken-r6h-tp4.json`, `experiments/2026-10-13-r6i` and
  `-kernel-names` (r6i was not promoted; r6j has no results yet). Its author's forum posts
  [384438/6](https://forums.developer.nvidia.com/t/384438/6) (r6) and
  [384438/15](https://forums.developer.nvidia.com/t/384438/15) (r6h).
- christopherowen/dgx-spark-networking at `b61660f` (2026-10-07): README, `docs/oneshot.md`, `docs/design.md`.
- bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10 at `6033ea4` (2026-10-09) and its Hugging Face card at
  `f8ee0e77`; `kit_bench.py` at bertholomus/TensorFold `fe6f9f7` (sha256 `83a4fbeb…`, the file sudoingX's
  README pins).
- nero-/deepseek-v41-flash-gb10-ring at `364eb87`: `RESULTS.md`, `RESULTS-SGLANG-ARCHIVE.md`,
  `results/20260927/final-ring.json`.
- NVIDIA forum thread 382725, posts /196 (voktolom) and /226 (christopher_owen).
- rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8 at `3ac7546` (2026-09-14); sudoingX/dgx-spark-deepseek-v41
  at `cc59c36` (v1.1); ZackO2o/DeepSeek-V4.1-Flash-4x-GB10-1M-Full-Recipe at `ce69944` (results of
  2026-09-28); luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring; FujitsuPolycom/sparkring (README,
  release 2026.10.3); yunwei37/dgx-spark-4-ring-no-switch; antirez/ds4 (`docs/DGX_SPARK.md`);
  0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000; vllm-project/vllm#56797; the 0xTank R37 card.
- Hugging Face API: deepseek-ai/DeepSeek-V4.1-Flash at `fb2764a5cf321eaa5070ca8f9e892818f477c16d`;
  dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 at `d61c59ea5e514e25d305b5850e8a432f7a9969f2` (its model card
  and commit history).
- SGLang `f80c91a4b97e52e17fbcff088aa6b47b6270f6b9`, FlashInfer v0.6.18 (`69ff11fc`), b12x `a7d7d29b`,
  and DeepSeek's reference `inference/` at `5254448`. The research agent's citations are by file and line at
  these commits.
- On rank 0: the checkpoint comparison, the clocks (`nvidia-smi` on all four boxes), and RC5's
  `/data/dsgb10/bench/rc5/sd{1,2}/sdreport.md`.
