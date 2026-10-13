# dsgb10 design

The plan for a DeepSeek-V4.1-Flash engine on 4× GB10 in a ring. The evidence behind every number
here is in [docs/findings.md](docs/findings.md).

## Target

- **Model:** `DeepSeek-V4.1-Flash-UNCENSORED-FP8` (dealignai, `d61c59ea`), the set every box holds
  and the one pi uses. It is the same architecture and format as the original; only shards 13–39
  differ. Served name stays `deepseek-v4.1-flash-uncensored` on rank 0:8888, so pi needs
  no change.
- **Fleet:** ranks **0 rank 0, 1 rank 1, 2 rank 2, 3 rank 3**, the directed ring
  rank 0 → rank 1 → rank 2 → rank 3 → rank 0. The rank 0 ↔ rank 1 leg runs at 100G, the others at 200G.
  There is no cable between opposite boxes.
- **Workload:** agentic. One to a few concurrent sessions, long and growing contexts, short new
  suffixes per turn (tool results), tool calls, thinking on or off. That means single-stream decode
  latency, prefill throughput, and exact prefix reuse across turns.

## Baseline

The SGLang TP=4 kit as it served on the ring (MiaAI-Lab @ `cad252b`, NCCL `switchless` ring, no
RoCEnante, DSpark k=5). Measured 2026-10-11 from rank 0 with `bench/bench.py`, greedy, thinking off.

| | |
|---|---|
| Decode, 512 tokens, one stream | prose **46.8**, code **77.4**, agentic **73.6**, json **106.9** tok/s |
| TTFT, short prompt | 244–280 ms |
| Cold prefill, unique prefix, real Python source | 1k **1.9k**, 4k **3.7k**, 16k **4.3k**, 32k **4.2k**, 64k **4.25k** tok/s |
| Agentic turn, 16k context + ~2k new tokens per turn, prefix cached | TTFT **0.71–0.79 s** |

For comparison, the kit's documentation (v2.1, 2026-09-25) reports 87.7 prose / 124.8 code tok/s and
~4.9k tok/s real-text prefill on a *switched* fabric with RoCEnante.

Those numbers are the real bar. The best published results on four GB10s that we had found by
2026-10-11 were **~88 prose / ~125–132 code tok/s and ~4.9k tok/s prefill**
([docs/findings.md](docs/findings.md#other-recipes)). The review of 2026-10-13 found higher ones:
90.2 / 133.2 on sparkDash and ~5.9k on real text on four boxes, and 111.9 / 184.9 on eight
([docs/review/external.md](docs/review/external.md#survey)). Projects use different prompts, so
comparisons also look at time per step and tokens accepted per step, not at tok/s alone.

## Budgets that drive the design

| | |
|---|---|
| Streaming read bandwidth per box | ~235 GB/s measured (273 GB/s peak) |
| Bytes read per decode token, whole model | ~13.0 GB: routed experts 4.5 GB, attention 5.2 (`wo_a` read as FP8), shared experts 1.4, head 1.3 (bf16), engram wkv 0.3, gate + mHC 0.3 |
| → decode floor at TP=4, perfect split, no comms | 3.26 GB/rank → **~13.9 ms/token, ~72 tok/s** before speculation |
| A verify of k rows | ~2.1 GB/rank of non-expert reads (~9.0 ms), plus ~0.19 GB/rank (~0.8 ms) per distinct expert per layer; 6 rows touch ~25–30 experts per layer → **~29–33 ms/step** |
| Prefill compute | ~17 GFLOP/token through the 20 encoder layers (the GEMMs are 15.3 of it); block-scaled MMA ~214 TFLOPS per box measured, ~250 reached elsewhere. In practice prefill MoE is bound by reading each layer's experts once per chunk (1.8 GB/rank, 7.7 ms), so chunks should be ≥ 4096 rows |
| Collective latency | GPU→GPU one ring hop via CPU proxy: 2.7 µs (64 B) – 5.1 µs (10 KB); our 4-ring all-reduce: 7.7 µs (64 B), **12.3 µs (10 KB)**, 24.5 µs (60 KB); the best prior one we found takes 16.2 µs at 10 KiB (christopherowen's host relay) |
| Per-rank resident weights at TP=4 | ~72 GiB without Engram; Engram adds 47 GiB/rank, so it **must stay on NVMe** |
| Engram reads in prefill | 2 layers × 24 rows per token ≈ 12 random reads per token per box: ~120k IOPS per box at 10k tok/s, more than the ~112k the kit's documents give for the NVMe at QD 64, so the hot-row cache is required. Measured later on these drives: 377k–854k IOPS at QD 64 and 543k–1,102k at QD 128 with 768-byte reads, the slowest over four times the need at QD 128 ([docs/engram.md](docs/engram.md#random-record-reads-engram_iobench-264-byte-records-768-bytes-per-read)) |

Decode is a memory-bandwidth problem; prefill is a compute problem with an NVMe side-channel (Engram).

Measured on the full TP=4 engine (2026-10-11): a one-row decode step takes 19.4 ms against a floor
of ~15.6 ms (weights at 214 GB/s overall: 15.7 ms; collectives 2.3 ms; attention 0.8 ms; mHC 0.4 ms),
and a six-row step 37.3 ms. Known savings: collectives ~1.1 ms, weight streaming ~1.3 ms, mHC
~0.85 ms, and 1–2 ms per token of host time outside the GPU step ([docs/engine.md](docs/engine.md)).

### What the budgets allow

Speculation multiplies the tokens per step, but every verified row adds experts to read. With the
DSpark acceptance vLLM measured on GSM8K (drafts 1–5 survive with probability 0.72 / 0.52 / 0.30 /
0.18 / 0.10), the bandwidth floor gives:

| Verify rows | 1 | 2 | 3 | 4 | 6 |
|---|---:|---:|---:|---:|---:|
| Tokens per step | 1.00 | 1.72 | 2.24 | 2.54 | 2.82 |
| Distinct experts per layer (estimate) | 6 | ~11 | ~15 | ~19 | ~27 |
| Floor per step | 13.8 ms | 17.8 ms | 21.0 ms | 24.2 ms | 30.6 ms |
| Ceiling | 72 tok/s | 97 | **107** | 105 | 92 |

On code nearly every draft survives (the kit sees ~4.9 tokens per step), and six rows give a ceiling
near 160 tok/s. The kit's measured steps (33 ms prose, 39 ms code) are 1.1–1.6× these floors,
depending on how many rows its confidence cap leaves. Decode gains come from two places: running
close to the bandwidth floor (fused kernels, collectives at ~15 µs instead of NCCL's ~90), and
picking the number of verify rows per step from the confidence head. Neither reduces the bytes read,
so a realistic decode target is **1.1–1.3× the best published numbers** (roughly 95–110 prose,
140–160 code), which is about 2× what the kit did on this ring.

Prefill is different. The baseline's 4.2k tok/s at ~17 GFLOP per token is ~18 TFLOPS per box, under
10 % of what the tensor cores do. Prefill and exact prefix reuse are where the large gains for
agentic use are, until attention at long context and Engram reads set the limit.

## Decisions

1. **TP=4 for everything on the decode path**, with no replicated weight reads:
   - attention: 16 heads per rank; `wq_b` column-split, `wo_a` 2 groups per rank, `wo_b` row-split;
     `wq_a`, `wkv`, the router gate, `hc_fn` and Engram `wkv` split by output and all-gathered,
     instead of every rank reading them in full;
   - routed experts **tensor-split** (576 of the 2304 intermediate columns per rank). Every rank
     reads exactly a quarter of every selected expert, so load is balanced and there is no
     all-to-all; `w2`'s activation blocks (32) align with 576;
   - vocab-split head with **distributed Gumbel-max sampling** (each rank argmaxes its slice with
     its own noise, then one tiny all-gather).
   As converted ([docs/layout.md](docs/layout.md)): the embedding is replicated, since a token reads
   one 10 KiB row of it; the compressor and indexer weights are replicated for now (~0.35 ms of reads
   per step). The attention work settled them: the compressor's `wkv`/`wgate` split by output rows
   (128 per rank, exact; its slice rides the `wq_a`/`wkv` all-gather), the indexer stays replicated
   because decode splits compressed *positions* across ranks (local top-512, then a merge of 4 × 512
   candidates) and every rank needs all 32 heads. mHC is split by columns with a 25-float all-reduce
   per sublayer, summed in rank order; the projected stream exists only after the sublayer's own
   all-reduce, so it cannot share it.
2. **Our own collectives over ibverbs.** Buffers are `cudaHostAlloc` memory registered with
   `ibv_reg_mr`: GB10 has no GPUDirect RDMA, but host memory *is* GPU memory, and GPU kernels can poll
   and write it directly. One CPU proxy thread per NIC port posts RDMA writes when a GPU flag flips.
   The 4-ring all-reduce takes two hops: each rank writes its vector to both neighbours, then each
   neighbour's *proxy* relays the opposite rank's vector without a GPU round trip. Kernels spin on
   flags, so a whole decode step can be one CUDA graph.
   This is RoCEnante's pattern with the host relay of christopherowen's "ring4", which measured 16.9 µs
   per 10 KiB all-reduce on a 4-ring (2026-10-03) and 16.2 µs with its CuTe kernel (2026-10-05). Two
   of their details are requirements here. The
   flag travels as a second, inline RDMA WRITE on the data's RC queue pair, which delivers in order.
   And every rank sums the four partials **in fp32 in the same rank order**, so all ranks hold
   bit-identical residuals: replicated work that makes discrete choices (indexer top-k, routing,
   sampling) must choose the same on every rank. Large payloads split the relayed vector across both
   directions of the ring.
   As built ([docs/ring.md](docs/ring.md)): one spinning proxy thread per rank drives all four
   queue pairs (two per neighbour, one per PCI domain, each message split across both), buffers
   are host-NUMA `cuMemCreate` memory (the GPU reads it at 236 GB/s against 178 for
   `cudaHostAlloc`), and large payloads use reduce-scatter + all-gather. A 10 KB bf16 all-reduce
   takes 12.3 µs back to back in a graph, so a decode step's 80 cost ~1 ms.
3. **Kernels in CUDA C++ with inline PTX**, built for `compute_121a`/`sm_121a`:
   `mma.sync … kind::mxf8f6f4.block_scale` for FP8×FP4 (experts) and FP8×FP8 (dense) with ue8m0
   scales per 32. That is exactly the checkpoint's format, so there is no repacking of values, only of
   layout. Decode GEMVs are written to saturate bandwidth; prefill GEMMs to saturate the tensor cores.
   `wo_a` stays FP8 in memory and is widened to bf16 in registers. An e4m3 value times a
   power-of-two scale is exact in bf16, so these are the reference's bf16 weights at half the bytes
   (the bf16 einsum cost the kit 11 ms per step at TP=3). Warps that wait on a collective prefetch
   the next weights into L2, which took 13.5 % off a decode step elsewhere.
4. **Numerics follow the publisher's `model.py`** (FP8 activation quantisation with pow-2 scales
   per 32, FP4/FP8 KV formats, fp32 router and head, Sinkhorn mHC). Exact bit-equality with the
   reference is not a goal, but every kernel is checked against golden activations.
5. **Prefill uses CED** ([docs/ced.md](docs/ced.md)). The 20 encoder layers run over every prompt
   token, plus layer 20's KV projection (its ratio-1 compressor and index keys, the decoder's only
   global KV source in this checkpoint). The decoder runs once over the last 128 prompt tokens with
   its windows truncated there (Decoder SWA Bounded Replay, which the model was post-trained for);
   its global attention still reaches the whole prompt, and decode afterwards is unchanged. Against
   the exact forward the first token is unchanged in every long case tried (KL ≤ 0.012 nats) and
   decode stays inside `model.py`'s own numerics envelope (95–98 % top-1), so CED is the default and
   an "exact" mode that runs the decoder over everything is kept for validation. The replay costs
   about 20 expert sweeps (~150 ms at TP=4) whatever the prompt length, so prefill time halves once
   chunks are compute-bound. A turn that adds fewer than 128 tokens continues the decoder exactly
   from its live rings (a multi-row decode step) instead of replaying. The indexer's query rows are
   split across ranks (21–28 % off long prefills for rhys101).
6. **Prefix reuse per session is exact where it can be.** Global KV is 890 B/token and SWA state is
   ~3 MB per session, so recent sessions keep their full state and a new turn continues from it. A
   divergent prefix restarts exactly, from the live rings or the latest window snapshot before the
   divergence (one every 1,024 positions, thinned further back), and an evicted session comes back
   from the disk tier (or, once dropped from it, is prefilled cold). Encoder SWA Bounded Replay, the
   approximate fallback first planned here, is not built ([engine § Serving](docs/engine.md#serving)).
7. **Speculation with DSpark, gated by its confidence head.** The verify length is chosen per step
   from the predicted survival probabilities and the measured cost of verifying k tokens (which grows
   with distinct experts). The confidence head's j-th output is the logit of P(draft j+1 accepted |
   drafts 1..j accepted), well calibrated on the prose run, so survival is the running product of
   sigmoids; verify draft j while survival ≥ (recent tokens per ms × one extra row's cost), ≈ 0.21 at
   the measured step times. Sampled requests use standard speculative sampling. Kernels read the
   step's row count from device memory, so one CUDA graph serves every length and the host can queue
   the next step before the current one ends (others measured a 0.318 ms gap between graph replays).
   On prose, acceptance is low (0.80 accepted drafts per step on our prompt, which is also what the
   kit's 46.8 tok/s implies), so most of DSpark's gain will come on code, JSON and agentic output.
8. **Engram rows live on NVMe** (the kit's packed per-rank shards are reused as-is), read with
   O_DIRECT through io_uring, behind a small hot-row cache. Hashes depend only on token ids, so reads
   start the moment a token is known: before layer 0 in decode, a whole chunk ahead in prefill.
9. **Front-end in Rust** on rank 0: OpenAI chat completions (streaming, tools, thinking, reasoning
   effort), a port of the publisher's `encoding.py`, BPE tokenisation, and session tracking for
   prefix reuse. It talks to rank 0 over a Unix socket.
10. **Weights are converted once per box** into a per-rank file laid out for the kernels, from that
    box's local copy of the checkpoint (rank 0: `/data`; others: `~`). Loading is then a sequential read.
11. **Outputs stay exact; weights may only shrink losslessly** (the owner, 2026-10-12: "i want the precision
    to be kept"). No lower-bit copies of weights the outputs depend on: the main head stays bf16, the
    experts MXFP4. Weights decode exactly in the kernels instead: since 2812072 the experts' ue8m0 scales
    as a 1-bit code over two neighbours and the head's exponents as 4-bit offsets per tile, both made
    and checked at load ([engine § Lossless weight compression](docs/engine.md#lossless-weight-compression-a-feasibility-study-2026-10-12)).
    Only the drafter may be approximated (its FP8 head copy), since every draft is verified.

## Risks and open questions

- **Decode headroom over the best published numbers is modest** (see above). Prefill and prefix
  reuse are where the large gains are.
- **Kernel count.** The attention side alone, unfused, is ~610 kernels and 5.3–6.9 ms per decode
  step, almost all launch and drain latency (context length adds only ~0.6 ms from 1k to 1M); mHC is
  10 kernels per layer. The layer runner has to fuse down towards DeepSeek's 11 kernels per layer,
  and prefill needs its own tensor-core attention (455 ms per rank for a 2,048-row chunk unfused).
- **Collective count.** Two all-reduces per layer plus the all-gathers of decision 1 come to up to
  ~240 collectives per step; at 5–17 µs each that is 1–4 ms of a 14–31 ms step. Merge the
  all-gathers, and replicate a small projection instead wherever its all-gather costs more than the
  reads it saves.
- **The slowest box sets the pace.** One node elsewhere measured 166 GB/s against 235 for its peers.
- **Proxy memory ordering on AArch64.** NCCL needed a fence patch on these CPUs. The collective tests
  check payload contents under load, not only flags.
- **The model's next-token accuracy collapses on parts of long agentic transcripts**, and this is
  the model's own behaviour, not the variant's ([golden/README.md](golden/README.md), "Collapse: cause"):
  in the 28.5k-token transcript, run with dealignai's weights like all golden data, top-1 accuracy falls
  from ~0.7 to 0.2–0.45 inside a tool output and stays there. deepseek-ai's official weights do the same
  under `model.py` (on the transcript's first 5,120 tokens; the engine has not run them); the Engram
  tables are byte-identical to theirs; the hashing, rows and numerics check out. The trigger is layer 1's
  Engram, through long-range attention to its writes at many earlier positions (switching it off
  gives 0.88/0.78 and −1.21 instead of −2.21 nats per token there, but costs ~1 nat per token on
  memorised documents). The engine reproduces the model, so this is a quality question for serving,
  not an engine bug; SGLang's prompt logprobs would confirm the serving path shows it too.
- **The model is numerically sensitive.** Changing only the GEMM summation order in the reference
  moves its argmax on prose (the variant's weights) about as far from itself as it is from SGLang
  (87–88/96 agreement, one or two clear flips), so the engine is validated layer by layer from
  golden inputs and end to end only statistically.
- **DSpark acceptance in practice** is below the paper's on natural text (2.82 tokens per step in
  vLLM on GSM8K). The gate is tuned on agentic traffic, not on benchmarks.

## Milestones

| # | Milestone | Done when | State |
|---|---|---|---|
| M0 | Research, baseline, microbenchmarks | numbers above recorded | **done** |
| M1 | Golden data | per-layer residuals + logits for a short (135-token) and a long (~20k, agentic) sequence, argmax agrees with SGLang's greedy text | **done** ([golden/README.md](golden/README.md)); 88/96 argmax vs SGLang, within what numeric noise alone produces, so the engine is checked layer by layer; the long sequence's accuracy collapse is the model's own behaviour (the variant's weights, and deepseek-ai's official ones under `model.py`) |
| M2 | Converter | per-rank files on all four boxes, round-trip checked against the checkpoint | **done**, format v2 with the kernels' layouts ([docs/layout.md](docs/layout.md)) |
| M3 | Kernels, one box | each layer type matches golden within tolerance (layers 0–3, 14, 20, 24 at TP=1) | **met at TP=4 instead**: every layer, fed golden inputs, matches at bf16 level (block rel-L2 0.5–1.3e-2, routing ≥ 806/810, index top-k 100 %); attention side fused (1.6 ms per step at 1 row) |
| M4 | RDMA collectives + TP=4 decode | full-model greedy decode matches golden logits; one CUDA graph per step; bandwidth measured on all four boxes | **done** ([docs/engine.md](docs/engine.md)): matches golden within the reference's envelope; one graph per step; 19.4 ms at k=1 (weights 15.7, collectives 2.3, attention 0.8, mHC 0.4), 37.3 ms at k=6 |
| M5 | Prefill | CED + bounded replay, chunked, tensor-core GEMMs; beats the baseline's 4.2k tok/s and the best published ~4.9k | **done**: on deepseek-ai's official checkpoint (2026-10-13, release candidate 5's code), through the front-end, cold prefill 8.5k–9.2k tok/s from 16k to 57k tokens against the SGLang kit's 4.3k on the same checkpoint the same day, agentic turns of +2k tokens 0.39–0.42 s to first token (kit 0.68–0.79), the 15.8k-token first turn 1.81 s (kit 3.72) ([engine.md](docs/engine.md#where-it-stands-on-the-official-checkpoint-2026-10-13)); when M5 was met, on the variant, 6.9k–7.2k tok/s (kit 4.2k–4.3k), 0.41–0.44 s and 2.25 s (kit 0.71–0.79 and 3.73). CED is the serving default (`--exact` opts out). Such a turn reads nearly every expert of all 40 layers (encoder chunk + 128-row replay), ~0.31 s at 235 GB/s, so turn latency is close to its floor; 8192-row chunks are the default since aa05c03 |
| M6 | Front-end | pi works unchanged against rank 0:8888: tools, thinking, streaming | **done** on a test port: pi completes multi-turn tool tasks against the engine, every turn reusing the whole previous one ([docs/engine.md](docs/engine.md#serving)); the move to :8888 came with M9 |
| M7 | DSpark | confidence-gated speculation; beats the baseline on every prompt type, and the best published ~88 prose / ~125 code tok/s | **beats the baseline on every prompt type**: on deepseek-ai's official checkpoint (2026-10-13, release candidate 5's code) 77.4 prose, 115.4 code, 102.9 agentic, 153.7 JSON tok/s against the SGLang kit's 50.9, 85.6, 76.3, 107.7 on the same checkpoint the same day, the tokens identical to decode without drafts; when M7 was met, on the variant, 69.4, 101.9, 99.8 and 136.0 (kit 46.8, 77.4, 73.6, 106.9). Against the best published GB10 numbers on their own benchmark (sparkDash 1.8.8: its prompts, one stream, greedy; both on the official checkpoint), it decodes 95.0 prose / 151.0 code tok/s against the MiaAI kit's 89.9 / 131.0 (v2.4, on its authors' switched fabric), a prose step taking 27.9 ms at 2.63 tokens against its 31.2–32.5 ms at 2.27 (v2.2) ([README](README.md#where-it-stands-2026-10-13)) |
| M8 | Engram store | NVMe rows with prefetch; prefill not I/O bound | **done** ([docs/engram.md](docs/engram.md)), and in the engine: each rank's store serves its own rows of both Engram layers, all-gathered over the ring at layers 1 and 14 ([docs/engine.md](docs/engine.md#process-model)) |
| M9 | Ship | `bench/bench.py` against both stacks | **live** since 2026-10-12 06:39 on rank 0's 127.0.0.1:8888 and, since 06:48, its private-network address, by the owner's go-ahead: `service.sh` with the kit's key and model id, so pi works unchanged. Checked: models, a plain answer, thinking at `max`, a tool call, streaming, 401 without the key. Development jobs borrow the ring from the service while it is idle (production since 08:16, [engine.md](docs/engine.md#borrowing-the-ring)). Since 11:14 it keeps sessions across stops and lends the ring after 2 idle minutes; since 12:59 it runs B1 time-slicing, the ring queue and the SSE keep-alive (engine 51f08122 from fd3d0e4), and since 13:13 a start behind a busy ring serves at once (426c87b); since 2026-10-13 04:13 it runs release candidate 5 (c247a78) |

Before M9 (shipping on rank 0:8888 in place of the kit), the hardening list:

- **Device memory reserved at startup: done.** On GB10, CUDA allocations see only free pages, not
  reclaimable page cache ([findings](docs/findings.md#environment-gotchas)), and most of the workers'
  cache was the model's own files: `run4.sh` drops it before starting (the service's start script
  must too). The prefill buffers, the spec step graph and the session store's pool (the smallest
  rank's free memory less 2 GiB, at most `--session-gib`; 12 GiB per rank now) are made at startup,
  and nothing allocates after it. A full store spills sessions, least recently used first, to a
  file on each rank's own disk and reads them back on a revisit (a 180k-token session's first token
  in 0.37 s against 19.3 s cold); it never crashes. Window snapshots thin with their distance from a
  session's end, so the store holds ~65 sessions of 128k tokens (25 before).
- **Clean shutdown: done.** `run4.sh stop` takes rank 0 down first; no more "peer closed" or core dumps.
- **The defaults a user gets are the fast ones: done.** CED, DSpark drafts with their head on an FP8
  copy, 8192-row chunks, the exact weight codes and B′'s L2 prefetch (`--exact`, `--no-spec`,
  `--draft-bf16` and `--no-l2pf` opt out).
- **Graph memory at long context limits: done.** The dense index layers' kernel takes its split of
  the work from the scan length in device memory, so one graph per row count serves every context:
  the spec step's graph is 0.24 GiB at any limit (it was one per 4096-position bucket, ~13 GiB at a
  256k limit), and a decode step no longer captures a graph as the context grows
  ([engine.md](docs/engine.md#long-contexts)).
- **Hours of real use: one hour done, hours to go.** `engine/server/tests/soak.py` drives agentic
  sessions through the front-end: turns, regenerates, forks, cancels, client disconnects, compactions,
  sessions to 158k tokens, more sessions than the store holds. One hour from three clients (2,183
  requests, [engine.md](docs/engine.md#soak-2026-10-12)): no errors, GPU memory and step time flat,
  every tool call's arguments valid JSON, the store full from minute 10 and evicting. The
  hours-long run waits for a free ring; it should also show where dsfront's memory levels off
  (231 → 270 MiB over the hour, its session table bounded).
- **Run as a service: the script is done.** `engine/server/service.sh` (start / stop / restart /
  status) supervises the four ranks and dsfront: it holds the ring lock, drops the page cache,
  restarts with backoff (a dead worker costs ~74 s, dsfront 2 s), rotates the logs, keeps the
  engine on a private unix socket and refuses 0.0.0.0, and port 8888 without `DSGB10_PROD=1`
  ([engine.md](docs/engine.md#as-a-service-servicesh)). M9 took :8888
  (2026-10-12); whether a systemd unit should start it after a reboot is still the owner's call.
- **Several requests at once: B1 in production, B2 merged off, B3 not built.** Since 2026-10-12 12:59
  production time-slices requests (B1): behind a request decoding 600 tokens, a 16k request's first
  token comes after 1.9 s instead of 7.1 s. B2, the rows of several sessions in one step, is merged
  (`--slots N`, 1 by default); each request in a batch equals its run alone, and a batch runs without
  drafts, so it buys fairness and first tokens, not throughput. B3, drafts across sessions, would
  decode 18–32 % more in all with 2–4 sessions; it is not built, as the owner runs several agents at
  once only occasionally ([docs/batching.md](docs/batching.md#status-2026-10-12)). A stop with
  requests in flight is clean (the cut request ends with an error event).
- **Both stacks benchmarked: done** with `bench/bench.py` (the kit on 2026-10-11, [Baseline](#baseline);
  dsgb10 in [README.md](README.md#where-it-stands-2026-10-13)); how to switch back is in
  [docs/operations.md](docs/operations.md).

Order: M1 → M2 → M3 → M4 is the critical path to a correct engine; M5 and M7 are where the speed
comes from; M6 makes it usable; M8 can start any time after M2.
