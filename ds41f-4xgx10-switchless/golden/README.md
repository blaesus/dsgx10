# Golden data (M1)

Reference tensors for checking the engine. Each sequence is one teacher-forced prefill (start_pos 0,
batch 1, world size 1) through the publisher's `ref/inference/model.py`, unchanged, with our
pure-torch stand-ins for its tilelang kernels (`ref/torchref/kernel.py`). `ref/torchref/ref_run.py`
builds and loads one layer at a time, so a GB10 holds it with room to spare (peak 9 GiB for prose,
23 GiB for 28.5k tokens).

The weights are the checkpoint the engine serves, dealignai's variant DeepSeek-V4.1-Flash-UNCENSORED-FP8
(`run_ref.sh`'s default `CKPT`; [README](../README.md)). Only the collapse study's `orig-*` runs below use
deepseek-ai's official weights.

## Sequences

| name | tokens | what | saved positions P | runtime (rank 2) | size |
|---|---|---|---|---|---|
| `prose` | 135 = 39 prompt + 96 completion | the `prose` prompt of `bench/golden_sglang.json` (chat, thinking off), completed with SGLang's own greedy tokens | all 135 | 238 s | 1.4 GB |
| `long` | 28,489 (prompt only) | agentic transcript: system prompt, 4 tools, 3 tasks, 11 tool calls with real results, thinking mode (effort prefix 50), ends at `<｜Assistant｜><think>` | every 256th + last 512 (622) | 1,038 s | 5.9 GB |

`long` comes from `build_agentic.py` (sources: CPython 3.12 `json`, `textwrap`, `heapq`, real command
output, this repo's `DESIGN.md` as of 4a544f1; no secret, no real conversation, and from the fleet only
the absolute path of the worktree the design document was read from, which names the account; the user
turns and the answers are written in the script) → `long.messages.json` →
`make_tokens.py messages` → `long.tokens.json`. It reaches layer 20's candidate pool (2,048 blocks of 8
positions = 16,384) from position 16,384 on, and every index layer's top-512 from position 1,023
(ratio 2) or 511 (ratio 1).

## Where

- rank 2: `~/dsgb10-data/golden/<name>/` (+ `<name>.log`), where they were made
- rank 0: `/data/dsgb10/golden/<name>/` (+ `<name>.log`), the copy to use

## Checking a build

A build computes what production computes when `engine/server/run4.sh` gives these hashes, release
candidate 5's (a build from a fresh clone of the public repository gave each of them again on
2026-10-13). The 28.5k CED prompt's is the exception: release candidate 5 prints the hash's seed there
(below), and its hash is b47c510's, whose engine computes what release candidate 5 computes. A run's
last lines end `tokens hash …` (decode) or `… hash …` (teacher-forced): FNV-1a over the engine's own
outputs, never the reference's. A decode or teacher-forced step hashes each token it chooses and that
token's logit; the 28.5k prompt, which teacher-forces nothing, hashes the prompt's sample: its token,
its logit and logprob, and its top 20 logprobs' ids and values.

| Run | `golden/refhash.sh` | Hash | `engine/server/run4.sh` arguments |
|---|---|---|---|
| teacher-forced | `tf` | `842278b28d87dad0` | `--mode tf --big --golden G --prompt 4096 --limit 4608 --prefill-rows 4096` |
| decode, k = 1 | `k1` | `febc7dc54ee40a39` | `--mode decode --golden G --ctx 1024 --steps 256 --big` |
| the same without the L2 prefetch | `nol2pf` | `febc7dc54ee40a39` | the same and `--no-l2pf` |
| the same with drafts | `fp8`, `bf16` | `febc7dc54ee40a39` | the same and `--spec --draft-fp8`, or `--spec --draft-bf16` |
| sampled | `temp` | `a8f7a08f8cd1c5e8` | `--mode decode --golden G --ctx 1024 --steps 128 --big --temp 0.8 --top-p 0.95 --top-k 40 --seed 7` |
| CED, decode at 16k | `ced16` | `81f772ab79a950e4` | `--mode decode --golden G --ctx 16384 --steps 256 --big --ced` |
| CED, the 28.5k prompt | `ced28` | `8f08a153d415c5c8` (before 2026-10-13, printed the seed `14650fb0739d0383`) | `--mode tf --big --golden C --prompt 28489 --limit 28489 --prefill-rows 8192 --ced` |

Before 2026-10-13 (b47c510) the last run printed `14650fb0739d0383` whatever the engine computed:
`--mode tf` hashed only teacher-forced rows, and this run has none, so it printed the hash's seed
([correctness review](../docs/review/correctness.md), M1). Its hash now covers the 28.5k-token CED
prefill's output at the prompt's last position (four chunks of up to 8,192 rows, then the decoder's
replay of the last 128 positions). The same command in exact mode (no `--ced`, `--prefill-rows 4096`:
`exact28`, a control, not a reference) gives another hash, `87f42fa9f94f13de`.

`golden/refhash.sh` runs them, four runs to a ring job through `engine/server/ring.sh` (a run takes
about 55 s, 40 of them loading the weights), and prints each run's hash, the hash above, a check and
the seconds; `golden/refhash.sh --help` lists its settings:

    golden/refhash.sh                             # the eight runs; BIN=... picks the binary
    golden/refhash.sh ced28 ced28 exact28         # some of them, repeats too
    EXPECT=official golden/refhash.sh -- --weights PATH   # deepseek-ai's checkpoint: its own hashes (below)
    EXPECT=0 golden/refhash.sh -- --weights PATH          # other weights: their equalities only

G is `long`'s directory and C rank 0's `/data/dsgb10/kernels/cedcheck/long28k-ced128`. These runs read
only the transcript's tokens (the teacher-forced one at 4,096 also opens G's top logits to compare
with, whatever they hold), so a directory made from this repository alone does for both:

    python3 golden/make_cedcheck.py --tokens-only --out DIR    # DIR/long-tokens: long.tokens.json's ids, zero logits
    G=DIR/long-tokens C=DIR/long-tokens golden/refhash.sh

With that directory `tf` and `ced28` gave their hashes on 2026-10-13, and a directory of the tokens
alone gave `k1`'s and `ced16`'s ([kit review](../docs/review/kit.md)). The comparisons with the
references need the real data: `long` as `run_ref.sh` makes it (Rerun, below), and for the CED checks
([correctness review](../docs/review/correctness.md), part C) the `cedcheck` directories, which
`golden/make_cedcheck.py` makes from `ref/torchref/ced_run.py`'s streams. Its `--help` gives the two
`ced_run.py` commands, and on rank 0 it writes `/data/dsgb10/kernels/cedcheck` again byte for byte:

    python3 golden/make_cedcheck.py --src /data/dsgb10/golden/ced --out DIR --tokenizer CKPT/tokenizer.json

On other weights, deepseek-ai's official checkpoint for one, no hash above carries over, only their
equalities (`k1` = `nol2pf` = `fp8` = `bf16`, and each run with itself), and the CED streams must come
from `ced_run.py` on those weights (`CKPT=<checkpoint> OUT=<dir> golden/run_ref.sh py ced_run.py …`,
then `--src <dir>/golden/ced`). The token directories carry over: both checkpoints ship the same
`tokenizer.json` (sha256 c90dfa01…), which `--tokenizer` checks against the transcript's text.
The official files' own values (`EXPECT=official`), `ced28`'s `de99bfce6ea96ba8` among them, are in
[engine.md § The official checkpoint's weights](../docs/engine.md#the-official-checkpoints-weights-2026-10-13):
`tf` c7605e0b6ba8d9d2, the four 1k decodes 7941647b162df1a8, `temp` 02ef3e0858d7eb45, `ced16` 8790ee9388f4abd8.

## Files

`manifest.json` lists every array with shape, dtype, positions and meaning. bf16 tensors are stored as
fp32 (exact), ids as int32, hash ids as int64. `LNN` is the block index 00..39; `[P, ...]` arrays hold
the saved positions `positions.npy`, `[R, ...]` arrays the record positions `record_positions.npy`.

| file | shape | content |
|---|---|---|
| `tokens`, `positions`, `record_positions` | [L], [P], [R] | ids; saved positions; indexer record positions |
| `hashes` | [L, 2, 24] | Engram hash row ids for layers 1 and 14 |
| `embed` | [P, 5120] | token embedding |
| `LNN.engram_out` | [P, 4, 5120] | layers 1, 14: residual after Engram, before the block |
| `LNN.attn_in` / `attn_out` | [P, 5120] | attention input (hc_pre + attn_norm) / output (after wo_b) |
| `LNN.mid` | [P, 4, 5120] | residual after the attention sublayer |
| `LNN.ffn_in` / `ffn_out` | [P, 5120] | MoE input (hc_pre + ffn_norm) / output incl. shared expert |
| `LNN.route_idx` / `route_w` | [P, 6] | chosen experts in top-k order / their weights (x route_scale) |
| `LNN.hc_{attn,ffn}_{pre,post,comb}` | [P, 4], [P, 4], [P, 4, 4] | mHC coefficients after Sinkhorn |
| `LNN.out`, `LNN.pre_mix` | [P, 4, 5120], [P, 4] | block output and the pre_mix it hands on |
| `LNN.index_topk` | [R, 512] | index layers 2, 8, 14, 20, 24, 28, 32, 36: chosen compressed positions, ascending, -1 = none |
| `L20.pool_blocks` | [R, ceil(T/8)] | layer 20's candidate pool: block b (compressed positions 8b..8b+7) kept |
| `final_pre_norm`, `final_hidden` | [P, 5120] | hc_pre with the last pre_mix; after the final norm (head input) |
| `main_hidden` | [P, 15360] | DSpark input: mean over copies of the inputs of layers 37, 38, 39 |
| `logits_topv` / `logits_topi` | [L, 32] | top-32 fp32 logits and ids, every position |
| `logits_lse`, `next_logprob` | [L], [L-1] | logsumexp of the full logits; log-probability of the actual next token |
| `logits_full` | [N, 129280] | full fp32 logits of the last N positions (prose: all 135; long: last 8) |

## Rerun

```
golden/run_ref.sh run prose golden/prose.tokens.json --positions all --record auto --full-logits all
golden/run_ref.sh run long golden/long.tokens.json --positions sparse:256:512 --record auto --full-logits 8
golden/run_ref.sh py /work/ref/torchref/check_kernels.py
golden/run_ref.sh py /work/golden/check_prose.py /out/prose /work/bench/golden_sglang.json prose
golden/run_ref.sh py /work/golden/check_records.py /out/long
golden/run_ref.sh py /work/golden/cmp_runs.py /out/prose /out/<variant>      # two runs, position by position
TORCHREF_ACT=splitk golden/run_ref.sh run long-splitk golden/long.tokens.json --positions sparse:256:512 --record none --full-logits 8
golden/run_ref.sh py /work/golden/cmp_drift.py /out/long /out/long-splitk --onset 16864   # drift by band
```

The collapse runs: `python3 golden/cut_long.py golden/long.tokens.json golden/long-cut5120.tokens.json 5120`,
then `golden/engram_batch.sh probe|ablate|orig|dense|exact|docs|conv` (it runs from a checkout of this
configuration at `~/dsgb10/golden-harness`, and its `orig` runs need deepseek-ai's official weights in
`~/DeepSeek-V4.1-Flash`); read them with `collapse_table.py`,
`cmp_windows.py`, `cmp_positions.py`, `cmp_doc_context.py`, `engram_report.py` and `engram_rowfreq.py`.

Launch long runs detached (`setsid nohup … < /dev/null &`) and poll the log. The script runs the
`dsv41-4x-spark:canary-roce` image against the box's checkpoint copy.

## Checks

**Kernel stand-ins vs the publisher's tilelang kernels** (`check_kernels.py`, random data, rank 2):

| op | bit-equal | max abs diff | rel-l2 |
|---|---|---|---|
| act_quant in place (ue8m0 / 32) | 100% | 0 | 0 |
| fp4_act_quant e8m0 / 32, e4m3 / 16 | 100% | 0 | 0 |
| fp8_gemm | 99.994% | 3.1e-2 | 3.9e-5 |
| fp4_gemm | 99.998% | 3.9e-3 | 8.3e-7 |
| sparse_attn (16 heads) | 99.739% | 4.9e-4 | 1.5e-4 |
| hc_split_sinkhorn | | ≤ 1.2e-7 | |

The tilelang sparse_attn needs 141,312 B of shared memory at 64 heads and GB10 allows ~100 KB, so it is
checked at 16 heads (one TP=4 rank's share).

**Wiring.** Every Block parameter (layers 0, 1, 2, 14, 20, 39 checked) maps to a checkpoint tensor and
nothing is left unread except the Engram tables, which are read row by row. All dtype conversions are
exact except `wo_a`, which is dequantized to bf16 as convert.py does. Engram hash ids equal SGLang's
hasher (`sglang.srt.layers.engram`) on 7,400 transcript tokens: identical token map (99,092 ids), and row
ids stay below the table sizes, which equal the sum of the primes.

**Prefill only.** Every run is one start_pos-0 forward per layer over the whole sequence. Decode-only
state therefore never comes into play: for example, model.py publishes `index_k` only when a ratio-2 group
completes, so layers 2/8/14 read stale keys at some decode positions. `check_records.py` confirms that
each index layer picked min(512, ⌊(p+1)/ratio⌋) positions at every record position.

**prose vs SGLang** (`check_prose.py`). The argmax equals SGLang's greedy token at 88/96 completion
steps. Over SGLang's top-5 tokens, |Δ logprob| has mean 0.744, median 0.518, p95 2.16 and max 7.47
(480 pairs). SGLang's logprobs come from bf16 logits: its margins are multiples of 1/8.

The 8 misses, as step: SGLang's token / ours (SGLang margin / our margin):

- 4: " slab" / " black" (0.125 / 0.60)
- 12: " beam" / " was" (2.75 / 3.03)
- 28: " the" / " Black" (0.25 / 0.81)
- 34: " St" / " Car" (0.125 / 0.009)
- 40: " life" / " world" (1.25 / 0.51)
- 49: " two" / " three" (0.5 / 0.27)
- 53: " three" / " four" (0.0, a tie / 2.68)
- 60: " language" / " steady" (0.5 / 0.59)

Steps 12 and 53 are clear misses on one or both sides. To calibrate, the same harness was rerun with only
the numerics changed (`TORCHREF_ACT`), comparing the logits at all 135 positions (`cmp_runs.py`):

| variant vs reference | completion argmax | top-5 mean abs Δ | clear flips | same vs SGLang |
|---|---|---|---|---|
| each GEMM summed in two K halves (`splitk`) | 87/96 | 0.59 | 1 (margins 2.04 / 1.24) | 0.745 |
| activations per 128 with fp32 scales (`g128`) | 84/96 | 0.71 | 0 | 0.83 |
| no fp8 rounding of GEMM activations (`bf16`) | 86/96 | 0.57 | 1 (2.04 / 1.36) | 0.62 |

Reordering fp32 sums alone moves the prose logits as far as SGLang does, clear flips included. Each fp8
re-rounding and each top-6-of-384 routing flip amplifies a perturbation, and over 40 layers that adds up.
SGLang's misses lie within that envelope, so they do not show the harness is wrong.

For the engine, this means end-to-end logits can only be compared statistically. Exactness has to be
checked per layer, feeding each layer the golden inputs; `prose` has every position for that.

The same holds along `long` (`cmp_drift.py long long-splitk`; split-K is the only change). Relative error
of the block output against golden at the saved positions, median (max) per band:

| tensor | pos 0 | 256–1,024 | 1,280–4,096 | 4,352–16,863 | 16,864–27,976 | last 512 |
|---|---|---|---|---|---|---|
| L01.out | 0.001 | 0.008 (0.010) | 0.009 (0.028) | 0.008 (0.032) | 0.009 (0.021) | 0.008 (0.063) |
| L09.out | 0.019 | 0.079 (0.103) | 0.077 (0.103) | 0.074 (0.156) | 0.083 (0.193) | 0.083 (0.335) |
| L19.out | 0.014 | 0.148 (0.209) | 0.224 (0.782) | 0.138 (0.870) | 0.161 (0.913) | 0.149 (0.990) |
| L29.out | 0.005 | 0.274 (0.610) | 0.389 (0.735) | 0.391 (1.201) | 0.436 (1.034) | 0.375 (1.129) |
| L39.out | 0.005 | 0.387 (0.632) | 0.497 (0.631) | 0.330 (0.914) | 0.406 (0.905) | 0.389 (9.23) |
| final_hidden | 0.005 | 0.421 (0.517) | 0.628 (0.742) | 0.390 (0.817) | 0.447 (0.778) | 0.439 (1.34) |

The 9.23 (and 378 at L37) is position 28,423, where one run has a massive activation that the other does
not. Top-1 agreement with golden's argmax is 0.902 on the first 512 positions, 0.796 before the onset
(16,864) and 0.701 after. Of the flips, 1,708 before and 1,649 after are clear, meaning each run puts its
own pick more than 1 nat above the other's. split-K gets the next token right at 16,355 positions, golden
at 16,122. An engine that drifts by that much is inside model.py's own envelope.

**long.** `check_records.py`: layer 20's pool kept exactly min(2048, reachable blocks) at all 20 record
positions. Past position 16,383 that is 2,048 of 2,049–3,562 blocks, and the dropped blocks are scattered
old ones. Every index layer picked min(512, reachable), and the reindex layers 24/28/32/36 stayed inside
the pool. Over the whole sequence, argmax == next token at 16,122/28,488 positions, and the mean
next-token logprob is −2.86.

By segment, the json files score 0.57–0.70 (logprob −1.8 to −2.8) and tool calls 0.88–1.0. From token
16,864 on, the code and prose score 0.2–0.45 (logprob −4 to −6), and the predictions become non-code
tokens ("Now", `{"`, "conc"). Token 16,864 is textwrap.py's `# -- Public interface ---` line; the last
~600 tokens of json/encoder.py look the same.

Content vs position: the drop follows the text and comes with Engram. Removing the json task
(`long.tokens.json[:566] + [12903:]`) moves the same text to token 4,527, nowhere near the 16,384 pool
limit, and the drop moves with it. A 7,400-token prefix shows it too. Ablations of that prefix:

| ablation | effect on the drop |
|---|---|
| pool off (`--set candidate_topk_blocks=4096`, full sequence) | none (16,092 vs 16,122 hits) |
| YaRN off (`--set original_seq_len=0`) | none |
| both Engram layers skipped (`--ablate engram`) | gone: the four 256-token windows after the onset score 0.85/0.80/0.61/0.71 instead of 0.33/0.25/0.27/0.21 |

**Collapse: cause.** Not the checkpoint or the numerics, and no fault in the Engram code: layer 1's Engram, working
as trained, derails the model in this agentic context. Runs on the first 5,120 tokens of long-cut
(`engram_batch.sh`, onset ~4,527): top-1 accuracy in the windows after the onset, mean next-token logprob.

| run | change | 4,608+256 | 4,864+256 | logprob 0–4,479 | 0–5,118 |
|---|---|---|---|---|---|
| orig-base | deepseek-ai's own weights | 0.35 | 0.27 | −1.89 | −2.21 |
| pre-base | the uncensored weights (7,400-token run) | 0.33 | 0.25 | −1.96 | −2.26 |
| exact | no rounding of activations, KV caches or indexer (`TORCHREF_ACT=exact`) | 0.35 | 0.26 | −1.93 | −2.24 |
| dense | every compressed entry selected (`--set index_topk=8192`) | 0.45 | 0.31 | −1.82 | −2.02 |
| off-l14 | layer 14's Engram off | 0.37 | 0.30 | −1.71 | −1.98 |
| off-onset | both Engram layers off at 4,480–4,559 only | 0.61 | 0.27 | −1.97 | −2.12 |
| orig-noengram | both skipped, original weights | 0.85 | 0.80 | −1.51 | −1.41 |
| off-before | both off at 0–4,479 | 0.95 | 0.73 | −1.53 | −1.43 |
| off-l01 | layer 1's Engram off | 0.88 | 0.78 | −1.29 | −1.21 |
| off-l01-conv | layer 1's off at 0–1,162 only (the turns before the file) | 0.49 | 0.32 | −1.13 | −1.38 |

- Checkpoint: all `engram.*` tensors sit in shards 47/48, outside dealignai's 27 changed shards (13–39,
  layers 10–36), and on the boxes they are hard links to the original's SHA-verified files.
  `tools/hf_rows.py` fetched the onset rows (positions 4,520–4,527, 184 rows per layer, weights and
  scales) from deepseek-ai's repo by range requests, 98 KB in all: identical. orig-base collapses too.
- Rows (`engram_probe.py`, `engram_report.py`): no e4m3 NaN codes, no 0xFF scales, exponents −11..−6 as in
  random rows of the same sub-tables. Engram's writes at the onset are ordinary: at most 6× the median
  |Δh| at layer 14 and 3× at layer 1, against 11× (and 33× at BOS) elsewhere.
- Hash: as in the Engram demo (compressed tokenizer, odd multipliers from rng(10007·layer), XOR, prime
  moduli, offsets); the primes sum to the table sizes and token × multiplier stays below 2^63. The rows it
  picks for the sequence's 20 most frequent 2- and 3-grams per head have norms at the 0–4th percentile of
  random rows (`engram_rowfreq.py`): trained rows, so the hash agrees with training.
- Engram works on documents: BOS + textwrap.py (2,048 tokens) gets 2,037/2,047 top-1 at logprob −0.03, and
  1,518 at −1.08 with layer 1's Engram off. On never-published prose (docs/findings.md) layer 1 helps a
  little (−2.03 against −2.13).
- Mechanism: layers 0–1 see 128-token windows and Engram is local, so from 255 tokens into the file the
  stream entering layer 2 is the same as in the plain document (up to rounding). Yet the transcript puts −15 to −20 on
  tokens the document predicts at −0.00 (`cmp_doc_context.py`), predicting words of the conversation
  (" Let", " Looking", " read", " heapq") inside the tool result. Layer 1 off over the turns before the
  file removes those errors, so they come through long-range attention to layer-1 writes at the
  conversation positions. The collapse itself needs the writes inside the file (off-l01-conv still
  collapses, off-before does not). No single position starts it.

So, as far as model.py's forward goes, this is how the weights behave: deepseek-ai's official set
(orig-base) as well as dealignai's variant, which the golden data and the engine use. SGLang's prompt
logprobs on `long.tokens.json` (one `/generate` call with `input_ids`, `return_logprob` and
`logprob_start_len: 0`) would confirm it for the serving path. For the engine: reproduce it, since the
golden data is what model.py computes, and do not read long-transcript accuracy past such an onset as an
engine fault. Turning layer 1's Engram off is no fix: it costs a nat per token on memorized text.

**GB10 notes.**
- A run started while the page cache was full failed at its first CUDA allocation. ref_run.py now drops
  checkpoint pages after reading them.
- At 28k tokens each Engram layer takes ~3 min: 684k random row reads from the 98 GB table. With the
  mmap's default readahead those reads put ~70 GB into the page cache; ref_run.py now sets MADV_RANDOM
  (100k rows: 0.4 GB of cache instead of 11.5 GB, and 30% faster) and drops the pages after each lookup.
- The experiment runs (`long-cut`, `long-nopool`, `pre-base` with all tensors at cut positions 4,380–4,720,
  `pre-noengram`, `pre-noyarn`, `prose-act-g128`, `prose-act-bf16`, `prose-splitk`, `long-splitk`, and the
  collapse runs above with `probe5120.npz`) sit beside the golden runs in both locations. The `--set`, `--ablate`, `ranges:` and `TORCHREF_ACT` options exist for such
  experiments only.
