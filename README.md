# dsgx10

From-scratch inference engines for NVIDIA GB10 boxes.
Each directory is one configuration — a model, a set of boxes, a topology — and is
specialised to it: speed over generality. Each is self-contained, with its own engine, front-end,
docs and tools, and its README is its index.

| Configuration | Model | Boxes and topology | State |
|---|---|---|---|
| [ds41f-4xgx10-switchless](ds41f-4xgx10-switchless/) ([how to use](ds41f-4xgx10-switchless/README.md#how-to-use)) | DeepSeek-V4.1-Flash, built on the community variant [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8) at `d61c59ea` (the official architecture and tensor formats; the values of shards 13–39 differ; hence the served name `deepseek-v4.1-flash-uncensored`); measured since 2026-10-13 on deepseek-ai's official checkpoint ([DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) at `fb2764a5`) | four GB10 boxes (two ASUS Ascent GX10 and two NVIDIA DGX Spark, ranks 0–3) in a switchless ConnectX-7 ring, TP=4 | **serving** the variant on rank 0:8888 from 2026-10-12, and [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) since 2026-10-18 (`dsfront --encoding 1017`). On the official checkpoint: single-stream decode 77–154 tok/s with speculative drafts (the SGLang kit on it the same day: 51–108), cold prefill 8.5–9.2k tok/s at 16k–57k tokens (kit: 4.3k) and 9.4–9.5k at 120k; on the variant, agentic turns flat to 250k tokens of context |
| [altar1-4xgx10-switchless](altar1-4xgx10-switchless/) | [AikidoSec/altar-1](https://huggingface.co/AikidoSec/altar-1) at `5d591cd9` (GLM-5.3 pruned to 168 of 256 routed experts, compressed-tensors INT4 g32 experts and attention projections, the rest BF16; nothing requantised) | the same four boxes, TP=4 | **Serving (2026-10-14)**: A0–A8 done on the four installed rank files, within the reference's own order variance on all four golden sets, bit-stable, 45.7 ms a step at one row (1.23× the byte floor), drafts exact at 28–30 tok/s, the service handing the ring over with the DeepSeek service → [where it stands](altar1-4xgx10-switchless/README.md#where-it-stands-2026-10-14-evening) |

## ds41f-4xgx10-switchless: where it stands (2026-10-13)

On deepseek-ai's official checkpoint ([DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
at `fb2764a5`), with release candidate 5's engine code (built from 530af7f), each figure measured with its
original tool, prompts and settings. Production serves the community variant the engine was built on; on it,
release candidate 5 decoded 75.7 / 111.7 / 109.2 / 149.8 tok/s on bench.py the same morning, at the same prefill
and agentic times ([history](ds41f-4xgx10-switchless/docs/engine.md#where-it-stands-on-release-candidate-5-2026-10-13)).

Against the SGLang TP=4 kit that served on this ring before
([MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) at
`cad252b`, v2.1, over NCCL), on the same checkpoint and ring the same day, with the same benchmark
(`bench/bench.py`: speculative drafts on both, greedy, 512 tokens, one request at a time):

| | dsgb10 | SGLang kit |
|---|---|---|
| decode tok/s: prose / code / agentic / JSON | **77.4 / 115.4 / 102.9 / 153.7** | 50.9 / 85.6 / 76.3 / 107.7 |
| time to first token, short prompts | **124–156 ms** | 219–253 ms |
| cold prefill, 16k–57k tokens | **8.5k–9.2k tok/s** | 4.3k tok/s |
| agentic: first turn (15.8k tokens) | **1.81 s** | 3.72 s |
| agentic: turns adding ~2k tokens, to first token | **0.39–0.42 s** | 0.68–0.79 s |

The table compares dsgb10 with the MiaAI kit's TP4 line on a switched RoCE fabric
([knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4),
v2.4 of 2026-10-12 unless marked), whose decode figures were the best published for four GB10s at the
checkpoint's native precision on 2026-10-13; both run deepseek-ai's official checkpoint. It uses that project's
own benchmark ([sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8: its prompts, 256 tokens, greedy, one
stream). No published native-precision four-GB10 figure we found beats dsgb10's on any row
([survey](ds41f-4xgx10-switchless/docs/review/external.md#survey)); the nearest is a code step 1.7 %
shorter than ours that carries a third fewer tokens. The best per c1 type are 90.2 / 133.2 (v2.2) / 158.7 /
135.3 (a community ring run of v2.3), and the best real-text prefill is 5.7k–5.9k (christopherowen's
spark-ds41f, vLLM). A 2.9-bpw quantisation publishes higher code and counting rates on its own prompts.

| | dsgb10 | MiaAI kit |
|---|---|---|
| decode tok/s: prose / code / structured / JSON | **95.0 / 151.0 / 185.8 / 150.2** | 89.9 / 131.0 / 158.7 / 128.4 (v2.4) |
| sparkDash's prose / structural catalogs, one stream (ours: 256 tokens each, the mean; the kit's: its own harness) | **78.1 / 122.3** | 61.1 / 99.0 (v2.2) |
| a decode step, prose: ours on sparkDash's prompt, the kit's on its own probe prompts | **27.9 ms at 2.63 tokens** | 31.2–32.5 ms at 2.27 (v2.2) |
| a decode step, code: likewise | **34.8 ms at 5.20 tokens** | 37.4–37.8 ms at 3.87 (v2.2) |
| cold prefill, real text, 15k–120k tokens | **8.7k–9.5k tok/s** | 4.8k–5.0k tok/s (v2.1; the best published, spark-ds41f's: 5.7k–5.9k) |

As the coding agent [pi](https://github.com/earendil-works/pi) uses it, on the agentic-edit bench
(twelve pi-style coding tasks, three times, at dsfront's sampling defaults, thinking off): 138.5 and 134.2 s
end to end, 110.4 and 111.9 output tokens a second of it, decoding at 137.4 and 138.5 tok/s, with 229–230 ms to
the first token (median). All 36 task checks passed in both runs.

On the same checkpoint, dsgb10 answers 99 of GSM8K's first 100 test problems (the SGLang kit, the same day: 100),
finds the passkey in 18 of 18 prompts of 33k–220k tokens, and, teacher-forced along the kit's greedy outputs on
four prompts, picks the kit's token at 260 of 271 steps.

How it was measured:

- Each dsgb10 figure is the mean of its sessions on 2026-10-13: six of bench.py (three before the kit's
  sessions, three after: O2 / O1 0.997–1.004 on decode), two of sparkDash, two of the edit bench (its
  end-to-end times are given for each). Each session was a fresh engine at production's arguments on the
  official files; other agents' short jobs held the ring between sessions, never during one. bench.py's
  sessions agreed within 0.8 % on decode (their requests within 1.3 %), sparkDash's within 0.2 %. Each
  side's time to first token leaves out each session's first request (ours 164–168 ms, after an engine
  start; the kit's 215–218 ms).
- The SGLang kit was measured on this ring on 2026-10-13 from 11:13 to 11:32, in one start: a warm-up session,
  left out, then three sessions, with at most one request running and none queued in its own log. Each decode
  figure is the mean of three session means, its requests 0.7–2.2 % apart; each prefill figure and the first
  agentic turn the mean of three requests. It was set up to batch up to 16 requests; every figure here is one
  request at a time, so throughput under concurrency is not compared. On the variant, on 2026-10-11, it decoded
  46.8 / 77.4 / 73.6 / 106.9 tok/s ([its baseline](ds41f-4xgx10-switchless/DESIGN.md#baseline)).
- The MiaAI kit's figures are from its own documents, not measured here. A stack that sums in
  another order follows another greedy text on the same prompt, so the single-prompt rows compare
  slightly different texts. The kit's documents recommend comparing step times and many-prompt runs across
  stacks, but its step probe and its catalog harness are its own and unpublished, and its steps carry fewer
  tokens, so no decode row is like for like; the prefill row is the one least affected.
- Against the variant, a step costs the same on the official checkpoint (prefill, the agentic turns and decode
  without drafts within 0.7 %); decode with drafts follows the drafts' acceptance on each checkpoint's greedy
  texts: −5.7 to +3.4 % on bench.py, −7.6 to +2.0 % on sparkDash's single prompts.
- The write-up, with every session's figures, the kit's conditions, the correctness checks and how the runs
  differ from release candidate 5's:
  [docs/engine.md](ds41f-4xgx10-switchless/docs/engine.md#where-it-stands-on-the-official-checkpoint-2026-10-13).
  Every experiment, failed ones included, is indexed in
  [docs/experiments.md](ds41f-4xgx10-switchless/docs/experiments.md). The raw data and the scripts
  are in rank 0's /data/dsgb10/bench/wave2 and rank 0's /data/dsgb10/bench/official.

Where to read, inside a configuration:

- `README.md`: where it stands, the status of every part, the layout, how to build and run;
- `DESIGN.md`: the decisions, budgets and milestones;
- `docs/`: what was measured and learned, with the evidence.

Large data (weights, golden activations, logs) never goes in the repo; each configuration's
README says where it lives on the boxes.

## Licence

The code is under the MIT licence (the [LICENSE](LICENSE) file). `ds41f-4xgx10-switchless/ref/inference`
and `ref/encoding` are DeepSeek's reference code under their MIT licence (`ref/LICENSE-deepseek`); the
agentic-edit bench carries the coding agent pi's prompt and tools (MIT, `bench/THIRD_PARTY.md`); other
third-party material is listed in [NOTICE](NOTICE). No model weights are included.
