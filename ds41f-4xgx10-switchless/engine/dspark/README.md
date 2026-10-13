# engine/dspark: the DSpark draft module (M7, part 1)

DeepSeek-V4.1-Flash's own drafter, the three DSpark blocks stored as `mtp.0`–`mtp.2`, runs here on one
rank at TP=4. This is the runner's part of a step. The module takes the newly accepted positions' layer
37–39 inputs and returns 5 greedy draft tokens with the confidence head's survival probabilities.

| file | what |
|---|---|
| dspark.h / dspark.cu | `Draft` (`accept`, `draft`), `weights_from`, `main_hidden`, the `Comm` interface, the stand-in collectives |
| ref/dspark_ref.py | the reference: model.py's DSpark (`Transformer.forward_spec`) over a golden run's main hiddens, every intermediate saved |
| tests/dspark_tests.cu | TP=4 on one GPU (four threads, emulated collectives) against the reference |
| tests/bench_dspark.cu | one rank's pass as a CUDA graph with stand-in collectives |

```
OUT=/data/dsgb10 golden/run_ref.sh py /work/engine/dspark/ref/dspark_ref.py --golden /out/golden/prose --out-dir /out/dspark/prose
engine/build/dspark/dspark_tests   # --ref /data/dsgb10/dspark/prose --golden /data/dsgb10/golden/prose --rank0 … --wdir …
engine/build/dspark/bench_dspark
```

The test reads ranks 1–3 from small files cut with `dsw_convert --only mtp.` and `--only head.weight`
(`/data/dsgb10/dspark/w/`). Rank 0 is a subset of `rank0.dsw`. About 11 GB in all.

## How model.py drives the draft

- **Main hidden.** At the input of layers 37, 38 and 39 (after Engram, before the attention), the model
  takes the mean of the 4 hc copies. Torch's bf16 mean is an fp32 sum in copy order, times 1/4, then
  rounded. The three means are concatenated: `main_hidden` [15360] per position.
- **main_x.** `main_x = main_norm(main_proj(main_hidden))` is the FP8 linear 15360 → 5120, then RMSNorm.
- **Window.** Each block's window holds `main_kv = kv_norm(wkv(main_x))` with RoPE at the position
  (theta 1e4, no YaRN) and the in-place FP8 round trip. Slots run `position % 128`.
  - The prefill call (`start_pos` 0) only writes these entries.
  - A decode call at position `i` writes `i`'s entry, then drafts.
  - The window holds only accepted positions. The block's own keys are concatenated per call and never
    stored.
- **The block.** The block's 5 rows sit at positions `i+1 … i+5`.
  - Input ids are the token at `i+1` (from the main model), then `dspark_noise_token_id` 128799 four
    times.
  - The ids go through the main model's `embed`, which is copied into the 4 hc copies.
  - The initial pre-mix is the identity `[1, 0, 0, 0]`.
  - Every row attends to the window's last `min(128, i+1)` entries plus all 5 block keys
    (`get_dspark_topk_idxs`: within the block it is non-causal). The same 64 heads, sink and scale apply.
- **The blocks are whole `Block`s.** Each has mHC around both sublayers and a 128-expert MoE.
  - Routing is top-3 of `sqrt(softplus(l)) + bias`, with weights `score / (sum + 1e-20) * 1.5`.
  - The shared expert is added.
  - mHC is single-pass. The attention collapses the stream with the previous sublayer's pre mix (the
    identity for the first block, else the previous block's FFN's). The FFN collapses with its own
    attention's pre mix. The mixes a sublayer computes go to `hc_post` and to the next collapse.
  - The stream passes through the three blocks in turn.
- **Head.**
  - `x = hc_pre(stream, last FFN's pre)`. There is no hc_head projection.
  - `logits = head(mtp.2.norm(x))` in fp32 over the main model's vocabulary head.
  - Then the Markov chain runs for j = 0..4:
    - `bias_j = markov_head.head(markov_head.embed(ids[j]))`, a rank-256 lookup and an fp32 GEMV over
      the vocabulary.
    - `logits[j] += bias_j`, then `ids[j+1] = sample(logits[j])`.
  - `confidence = proj([x_j, markov_embed_j])` in fp32. It is 5 logits with no sigmoid; `confidence[j]`
    belongs to `ids[j+1]`.
- **Sampling.** `sample` is Gumbel-max at `args.temperature`, default 1. At 0 it is argmax. The reference
  and this module draft greedily.
  - For a sampled request, the verify can treat each greedy draft as a one-hot proposal: accept `d`
    with probability p(d), else resample from p with d removed.
- **Advancing.** model.py has no speculative loop. Its `__main__` calls `forward_spec` once per decoded
  position: every accepted position's main hidden enters the window once, and the next block starts
  after the last accepted position.
- **The reference run.** It is teacher-forced on the prose golden run: a 39-token prompt, then 96 steps.
  Its leading drafts equal the golden tokens 0..5 times in 47, 31, 12, 3, 2, 1 steps (0.80 per step).
  d1 equals the main model's own argmax in 49 of 96 steps.

## On a rank (TP=4)

Per step, after the verify: `accept()` the newly accepted positions, then `draft()`. Every launch is a
PDL launch. Counts, positions and the token live on the device, so one graph of 89 nodes serves every
step. COLLECTIVE marks the runner's cross-rank points: 25 per step, all in the same order on every rank.
Given the whole mHC projections and router rows (`BlockW::attn_fn_all`, `ffn_fn_all`, `gate_all`, which
the runner gathers once at bind), every rank computes all 40 projection chunks and all 128 router logits
itself, the same bits the gathers bring, and the 9 gathers marked (*) drop out: 16 per step.

| launch | what |
|---|---|
| `main_hidden` (the runner, at layers 37, 38, 39) | the mean of the 4 copies → `mh` [8][15360] |
| `accept_prep` | act_quant of `mh` rows [0, count) and the anchor `pos0 + count - 1` |
| G main_proj | this rank's 1280 of 5120 outputs |
| COLLECTIVE all-gather | 8 × 2560 B per rank → main_proj's output |
| G main_norm + 3 × wkv | one `gemv_fp8_norm`: this rank's 128 of 512 per block |
| COLLECTIVE all-gather | 3 × 8 × 256 B per rank |
| `kv_write_sets` | kv_norm, RoPE, FP8 into the 3 windows (ring slot = position & 255) |
| `draft_prep` | the block's stream (token + 4 noise, 4 copies each), positions, window bounds, the identity pre mix |
| per block: `hc_project` | this rank's copy of the stream → 10 chunks (whole: all 4 copies → the 40) |
| COLLECTIVE all-gather (*) | 5000 B per rank: the 40 chunks in rank order, one rank's bits |
| `hc_mix` (pre_in) | the mixes, the collapse with the previous pre mix, attn_norm, act_quant |
| G wq_a + wkv | this rank's 320 + 128 outputs |
| COLLECTIVE all-gather | 5 × (640 + 256) B per rank |
| `decode_prep` | q_norm + act_quant, and the block's keys written at p+1..p+5 |
| G wq_b | this rank's 16 heads |
| `sparse_attn` (2 launches) | window bounds [max(0, p-127), p+5], 3 splits |
| G wo_a, wo_b | grouped wo_a with act_quant, then wo_b's partial sum |
| COLLECTIVE all-reduce | 5 × 5120, bf16 partials, fp32 sum in rank order |
| `hc_post` | |
| `hc_project`, COLLECTIVE all-gather (*), `hc_mix` (pre_in) | the FFN's mixes, the collapse with attention's pre mix, ffn_norm, act_quant |
| G gate | this rank's 32 of 128 router logits (bf16 GEMV, fp32 out; whole: all 128) |
| COLLECTIVE all-gather (*) | 5 × 128 B per rank |
| `router3` | top-3, torch's 3-value sum order (p0+p2)+p1; picks 4–6 repeat 1–3 for `dsk::moe` |
| `dsk::moe` (2 launches) | the routed experts' and the shared expert's 576 columns, fp32 partial |
| COLLECTIVE all-reduce | 5 × 5120, fp32 partials |
| `hc_post` | |
| `hc_pre` | the last pre mix + mtp.2.norm; also returns x before the norm |
| G head | this rank's 32320 vocabulary rows, fp32 (with `Weights::head8`: its FP8 copy, `head_fp8()`) |
| 5 × `markov` + COLLECTIVE argmax | step j: the Markov bias of ids[j] on this rank's slice, biased logits, local argmax (ties: lowest id) |
| `conf` | the 5 confidence logits, the survival products, ids[1..5] |

## API for the runner

```cpp
dspark::Weights w = dspark::weights_from({&rank_file});   // mtp.*, embed, head (the main model's)
dspark::Draft d(w, rank, &comm, rope_table_no_yarn);       // the runner implements dspark::Comm over the ring
// at the input of layers 37, 38, 39 of a verify (rows = its rows):
dspark::main_hidden(stream, rows, layer - 37, mh, true, s); // mh: bf16 [8][15360], the runner's buffer
// after the verify, k drafts accepted: positions p+1 .. p+1+k are final, bonus token t at p+2+k
d.accept(mh_rows_of_the_accepted, &count /* k+1 */, &pos0 /* p+1 */, s);
d.draft(&t, s);
// d.tokens() [5], d.survival() [5], d.confidence() [5]: device memory, the same bits on every rank
```

- **Prompt.** `accept` its last 128 positions (8 per call: `main_hidden` of a prefill chunk's rows), then
  `draft` the first generated token.
- **Rows.** `mh` rows are the accepted positions in order; `count` and `pos0` are device ints.
- **The `Comm` contract.**
  - `all_gather`: up to 4 segments in one collective, rank order.
  - `all_reduce`: an fp32 sum in rank order, rounded to bf16.
  - `argmax`: the highest `{v, id}`, ties to the lowest id.
  - Every rank must get the same bits back. `standin_comm()` times one rank alone.
- **Memory per rank.** 2.0 GB of `mtp.*` (+ the main model's embed and head) and 3 MB of buffers. The
  windows hold 3 × 135 KB.
- **`taps` / `inject`.** These copy out or replace intermediates for the test only. Keep them null in
  the runner.

## Accuracy (tests/dspark_tests.cu, prose, 96 steps, TP=4)

The test makes two passes.

**Injected pass.** Every sublayer starts from the reference's stream and pre mix, and the Markov chain
from its tokens.
- Bit-exact in every element:
  - main_hidden (3.5 M values), the 4 copies' collapse (`hc_pre`), the Markov embeddings, the expert
    choices (1440/1440 per block);
  - all 5 drafts in all 96 steps;
  - every rank's drafts and confidence identical.
- RMSNorms and FP8 GEMVs agree in ≥ 99.84 % of elements, with relative RMS error ≤ 3e-4. These are
  main_x, attn_in, ffn_in, the head's input, qa, q_pre and the window entries. The remaining elements
  are bf16 rounding flips: fp32 sums in another order, and a flipped input moves its whole row. The
  main path has the same class: engine/kernels' FP8 GEMVs are 99.9–100 % exact.
- TP=4 values sit at relative RMS error 2–5e-3 against model.py's single GPU: wo_b's and the MoE's
  partial sums, and with them mid and the block outputs. Attention (online softmax) is at 4e-4.
- The mixes, router weights, logits, Markov biases and confidence agree to relative RMS error ≤ 3e-6
  (fp32 sum order).

**Free pass.** The whole chain runs from the token alone.
- The drafts equal the reference's in all 5 positions in 79 of 96 steps (d1..d5: 93, 89, 85, 87, 87).
- Leading drafts equal to the golden tokens are **identical to the reference's**: 47, 31, 12, 3, 2, 1.
- At the three d1 differences, the reference's top-2 biased logits differ by 0.02, 0.19 and 0.75.
  - The drift also changes 1, 5 and 8 % of the expert choices in blocks 0, 1 and 2.
  - Confidence for the same input token: max diff 0.32 logits.

## Timing (tests/bench_dspark.cu, one rank, stand-in collectives, graph replays)

| pass | ms |
|---|---|
| draft: 3 blocks, head, 5-step Markov chain, confidence (83 nodes) | 2.77 |
| accept 1 + draft (89 nodes) | 2.90 |
| accept 6 + draft | 2.91 |
| of which the head GEMV (bf16 32320 × 5120, 331 MB) | 1.39 |

The pass reads about 590 MB per rank:
- the head slice, 331 MB;
- per block, attention 32 MB + shared expert 9 MB + about 9 routed experts × 4.4 MB (the reference
  averages 9.4, 7.0 and 5.3 distinct experts in blocks 0, 1, 2);
- main_proj, 20 MB;
- the Markov head, 16.5 MB, which stays in L2 across the chain's 5 steps.

At the head GEMV's 236 GB/s that is 2.5 ms, so the pass runs at about 86 % of streaming speed. Add the
runner's 25 collectives (≈ 0.4 ms at ~15 µs) for ≈ 3.3 ms per step. On the ring the pass takes 3.42 ms
at 6 rows, and 3.36 with the whole projections and router rows (16 collectives; the projections read
4 times the bytes): [engine.md](../../docs/engine.md#the-draft-pass).

**The FP8 head.** `head_fp8()` quantises the head slice to e4m3 with a power-of-two scale per 32×32 block
(act_quant's rule), and `Weights::head8` makes the pass read it (`dsk::gemv_fp8w_f32`: the weights
widened to bf16 exactly, bf16 activations, fp32 sums): 165 MB instead of 331, 0.66 ms instead of 1.40,
and 2.61 ms for the pass at 6 rows on the ring. The drafts' arithmetic changes; on the ring's runs the
drafts accepted per step stay the same or within one draft in a hundred steps (engine.md). The runner
makes it at bind when asked (`--draft-fp8`, serve's default).

What would cut it further: a vocabulary subset for the draft head (an approximation, FR-Spec style); the
head is a quarter of the pass with the FP8 copy.

## The confidence head, for the verify-length policy (DESIGN decision 7)

- **`confidence[j]` is the logit of a conditional probability**: P(d_{j+1} accepted | d_1..d_j were).
  On the reference run, mean sigmoid against the measured conditional rate, by position:

  | position | mean sigmoid | measured rate | n |
  |---|---|---|---|
  | d1 | 0.52 | 0.52 | 95 |
  | d2 | 0.39 | 0.38 | 48 |
  | d3 | 0.35 | 0.35 | 17 |

  Read as an unconditional "d_j accepted", it overestimates: 0.38 against 0.19 at d2, and 0.28 against
  0.065 at d3. So `survival()[j] = prod_{i<=j} sigmoid(confidence[i])` is P(the first j+1 drafts all
  accepted). The expected tokens from verifying k drafts is 1 + Σ_{j<k} survival[j].
- **Policy simulation.** This uses the reference's confidences and acceptances (prose), with the runner's
  19.4 ms (1 row) and 37.3 ms (6 rows), linear between them, plus 3.3 ms per draft pass.

  | policy | tok/s |
  |---|---|
  | no drafting | 51.5 |
  | fixed 1 draft | 57.5 |
  | fixed 2 drafts | 56.9 |
  | fixed 5 drafts | 44.4 |
  | verify while survival ≥ 0.3–0.4 | 59.2–59.5 (0.75–1 drafts per step) |
  | an oracle | 70.5 |

  The script is `ref/verify_policy.py REF_DIR DRAFT_MS`.
- **The rule.** Verify d_j while survival[j-1] ≥ λ·ΔC, with λ the recent tokens per ms and ΔC one more
  row's cost: 3.6 ms here, so a threshold near 0.21. The best fixed threshold on this run is 0.3–0.4,
  and the curve is flat from 0.25. Code accepts far more, so it will verify longer blocks under the
  same rule.
