# Findings

What was learned before writing the engine, with the evidence. Dates are 2026-10-11 unless stated.

## The model

Sources: the publisher's tech report (`DeepSeek_V41_Tech_Report.pdf` in the checkpoint; text via
`pdftotext -layout` → `ref/techreport.txt`, not committed), `inference/model.py`, `kernel.py`,
`engram.py`, `convert.py` (copied to `ref/inference/`), and the checkpoint headers
(`ref/inventory.txt`).

### Shape

| | |
|---|---|
| Layers | 40 = **20-layer causal encoder + 20-layer decoder** (CED), hidden 5120, + 3 DSpark draft blocks (`mtp.0–2`) |
| Attention | 64 query heads × 512, one shared 512-wide latent KV (MQA), RoPE on the last 64 dims; q LoRA 1280; output projected per group: 8 groups × 1024 (`wo_a`, block-diagonal), then `wo_b` 8192 → 5120; learned attention sink per head |
| SWA | every layer, window 128, KV kept **FP8 e4m3 with a ue8m0 scale per 32**, after RoPE |
| Compressed (global) KV | layers 2–19 ratio 2 (encoder), 20–39 ratio 1 (decoder); stored **FP4 e2m1 with an e4m3 scale per 16** (no global scale), after RoPE; 890 B/token in all |
| CSA2 modes | Full (own KV + indexer): **2, 8, 14** (encoder), **20** (decoder, also builds the candidate pool). Reindex (own indexer query, shared KV): **24, 28, 32, 36**. Reuse (shared KV and top-k): all others with a ratio. Layers 0–1: SWA only |
| Indexer | 32 heads × 128, FP4 q and k (ue8m0 per 32), `relu(q·k)` weighted by a per-head projection of x, top-512; layer 20 also keeps the best 2048 blocks of 8 as a 16,384-position candidate pool that 24–36 search within |
| MoE | every layer: 384 routed experts top-6 + 1 shared, intermediate 2304, SwiGLU with clamps (up ±10, gate ≤10); routed weights **MXFP4** (e2m1 + ue8m0 per 32 along K), shared FP8 with 32×32 ue8m0 blocks; router fp32 `sqrt(softplus)`, bias for selection only, normalised, ×1.5; a second bias for image tokens |
| mHC | residual carried as 4 copies; per sublayer 24 coefficients from one fp32 projection of the RMS-normalised 20480-wide stream → pre (4), post (4), comb (4×4, Sinkhorn 20 iters). **Single-pass**: a sublayer's input mix uses the coefficients computed by the *previous* sublayer |
| Engram | layers **1 and 14**; per position 3 n-gram orders × 8 heads = 24 rows of 256 FP8 values (+ ue8m0 per 32) from 384M-row tables, hashed from "compressed" token ids (NFKC/lowercase-normalised vocab of 99,092); `wkv` 6144 → 25600 = 4 keys + 1 value; sigmoid gate per hc copy |
| DSpark | 3 blocks, SWA-only, 128 experts top-3; input = `main_proj` of the mean-over-copies inputs of layers 37, 38, 39; drafts 5 tokens in one pass, Markov head (rank 256) adds token-to-token bias, confidence head predicts acceptance |
| Vision | 32-layer ViT (not planned for the first version) |
| Norm eps | 1e-20 (RMSNorm), hc eps 1e-6 |

### What makes it fast to serve, per the report

- **CED**: the decoder's global KV comes from the encoder output (layer 20's input), so prefill
  needs only the encoder plus layer 20's compressor for most tokens: "8B active parameters per
  token in prefill, 16B in decode".
- **Decoder SWA Bounded Replay**: at every prefill the decoder runs only over the last 128 prompt
  tokens, with SWA truncated to that segment (a query at i sees SWA keys in [max(s, i−127), i]). The
  result is approximate, and the model was trained with the same replay "for train-aware adaptation".
- **Encoder SWA Bounded Replay**: with a cached prefix but no SWA state, replay the last 128 tokens
  of the prefix together with the new suffix. Also approximate, and reported harmless.
- **The publisher's code has none of it**: `model.py` prefills all 40 layers over every position,
  `generate.py` prefills once, and `config.json` has no encoder or replay fields. Our reference
  (`ref/torchref/ced_run.py`, [docs/ced.md](ced.md)) emulates the decoder replay around `model.py`.
  In this checkpoint the decoder's only global KV source is layer 20 (`kv_source_layers` =
  [2, 8, 14, 20]), so its compressed KV, index keys and candidate pool are bit-identical to the
  exact forward's. Measured against exact: the first sampled token is unchanged in all six long
  cases (P = 4.7k–28.5k, KL 0.0004–0.012 nats), and teacher-forced decode over 255–16,035 tokens
  agrees on top-1 95–98 % with no bias in mean log-prob, closer to exact than `model.py` with its
  decoder GEMMs merely summed in two K halves. Replays of 256 or 512 tokens buy nothing measurable.
- Reuse-mode layers run in 15 kernels in prefill and 11 in decode in DeepSeek's own stack.

### Numerics of the reference (what the engine has to reproduce)

- Activations entering any FP8/FP4 GEMM are quantised to **e4m3 per 32 elements with a power-of-two
  scale** `2^ceil(log2(amax · fp32(1/448)))`, amax clamped at 1e-4; GEMMs accumulate in fp32 and
  emit bf16.
- `wo_a` is dequantised to bf16 by `convert.py` and applied as a bf16 einsum; no activation quant.
- The router, the head (bf16 weights promoted), the ratio-2 compressor (`wkv`, `wgate`) and mHC
  run in fp32. The ratio-1 compressor `wkv` is bf16. RMSNorm computes in fp32 and rounds to bf16.
- Expert outputs are rounded to bf16 before the fp32 sum; the shared expert is added after.
  Tensor-splitting experts across ranks rounds partial sums instead: a deliberate, small deviation.
- `sparse_attn` is online softmax over 64-wide blocks, P cast to bf16 for the PV product, sink added
  to the denominator, an all-invalid row gives zeros.
- **`model.py` has a decode-only bug**: `Indexer.forward` sets `shared_attn.index_k` only when its
  compression group completes (`if self.owns_k and latent is not None`). In decode, at positions where
  a ratio-2 group is still filling, layers 2, 8 and 14 therefore score against the previously
  published keys, which are layer 20's from the step before; at layer 2, position 300, none of the
  scores match and the top-k is wrong. Prefill from position 0 is unaffected. The engine uses each
  layer's own keys (the fix is to publish whenever `owns_k`); golden data from the reference in
  decode mode would be wrong at those positions for layers 2–19.
- `model.py`'s batched prefill varies slightly with batch size, so comparisons with golden prefill
  data need a small tolerance.

### Checkpoint

475.2 GiB in 48 shards; `config.json`, tokenizer and index are identical between the original and the
uncensored sets.

| Component | GiB |
|---|---:|
| Routed experts (fp4 + scales), 40 layers | 268.95 |
| Engram tables (2 layers) | 188.83 |
| DSpark routed experts | 6.72 |
| Attention incl. compressor/indexer | 4.80 |
| Embed + head (bf16) | 2.47 |
| Shared experts | 1.32 |
| Vision | 0.90 |
| DSpark dense | 0.66 |
| Engram `wkv`/gates, router, mHC | 0.59 |

Per rank at TP=4 that is ~72 GiB without Engram. Engram would add 47 GiB per rank, which does not
fit beside it in 121 GiB, so the tables stay on NVMe.

Tensor names are already in the reference's naming (`layers.N.attn.wq_a.weight`, `.scale`),
experts are stored as `I8 [out, in/2]` (two e2m1 per byte, low nibble first), scales as `F8_E8M0`.

## Prior work

### The MiaAI-Lab kit (what served here until 2026-10-11)

The kit @ `cad252b`: SGLang `dsv4.1` branch + FlashInfer/b12x + ~25 adapters, and the
TP4 production line contributed from knapcio's fork. Its README and `docs/tp4.md` record:

- **Numbers on a switched fabric** (v2.1, 2026-09-25): 87.7 prose / 124.8 code tok/s c1 (sparkDash,
  greedy); over 45 varied prompts 58.4 / 94.3 / 71.8; decode step 32–34 ms prose, 39 ms code. Cold
  prefill 4.0–5.9k tok/s on one repeated filler token, **4.8–5.0k on real text**. A 1,011,084-token
  needle passes.
- **DSpark** accepts ~3 tokens per step on prose and ~6 on code. k=5 with a confidence cap
  (`conf:0.1`) was their best. Verify rows past the cap reuse the anchor row's experts ("router live").
- **Experts tensor-split (EP=1, 576 wide per rank)** beat expert parallelism: the EP straggler at the
  MoE all-reduce cost more than it saved (NCCL 5.9 → 2.6 ms per step).
- **Engram**: rows from NVMe through a host row store (O_DIRECT, 96 threads; the NVMe does ~112k
  IOPS at QD 64). A 4 GiB row cache hits 67–76 %. Lookups moved to a side stream saved 2.3–2.9 ms per
  step. Real-text prefill is slower than the repeated-token filler because of these reads.
- **Time per decode step at TP=3 (~59 ms)**: routed MoE ~21–27 ms (scales with the 6-token verify),
  dense FP8 GEMMs ~17 ms (b12x warp-MMA kernel; Triton/CUTLASS were 50+), `wo_a` bf16 einsum 11 ms
  until an fp8 twin, NCCL 13–16 ms for 104 collectives, ~2000 small kernels ~10 ms.
- **RoCEnante**, b12x's one-shot RDMA all-reduce, writes into every peer, so on a ring it needs
  opposite-node paths forwarded in the neighbours' ConnectX-7 (sparkring's hairpin). Without it the
  kit falls back to patched NCCL, which is how it ran here.
- Memory lessons: NCCL's default buffers pinned 4.7 GiB per node; PyTorch's allocator fragments
  on long prefills; `expandable_segments` produced NaNs. None of this applies to an engine that
  allocates once.
- Serving lessons worth copying: the kit's reasoning-effort table is low/high/max = 50/75/100,
  default 75; cap `max_tokens` when a client omits it; abort on n-gram loops. The publisher's
  `encoding.py` has a different table (low 25, high 50, xhigh 75, max 100, default high = 50); the
  front-end follows the kit, which pi was tuned against.

An earlier record of the kit on the ring, from outside this project: 53–55 prose and
94–97 code tok/s on one prompt each, ~6 minutes to ready.

### Other recipes

From a web survey on 2026-10-11. Figures are single-stream, as each project reports them on its own
prompts, so compare them loosely.

| Recipe | Hardware and stack | Decode, one stream (tok/s) | Cold prefill (tok/s) |
|---|---|---|---|
| MiaAI kit, as it ran here | 4× GB10 ring, SGLang TP4, NCCL | 46.8 prose, 77.4 code | ~4.2k |
| MiaAI kit v2.1, its author's fabric | 4× GB10 switched, RoCEnante | 87.7 prose, 124.8 code | 4.8–5.0k real text |
| [rhys101](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8) ("SG18") | 8× GB10 switched, SGLang TP8/EP4, RoCEnante | 87.3 prose, 131.7 code | 4.4k at 4K, 3.5k at 299K |
| [sparkring](https://github.com/FujitsuPolycom/sparkring) | 4× GB10 ring, NIC hairpin forwarding | 57.0 | 4.4k at 64K |
| [yunwei37](https://github.com/yunwei37/dgx-spark-4-ring-no-switch) | 4× GB10 ring, vLLM TP4, NCCL mesh plugin | 50.1 mean, 68.1 code | 1.0–1.6k |
| [0xSero](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) | 4× RTX PRO 6000 on PCIe, SGLang TP4/EP4 | 200.9 | 7.4k peak, at 32K |
| [antirez/ds4](https://github.com/antirez/ds4) | 1× GB10, **V4** Flash, Q2 GGUF | 18.1 | 826 at 2K |
| llama.cpp [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696), open | 1× GB10, Q2_K | 2.8 | — |

What carries over:

- **The bar we had found by 2026-10-11 was ~88 prose / ~125–132 code tok/s and ~4.9k tok/s
  prefill.** rhys101's eight-box SG18 did not raise single-stream speed. The review of 2026-10-13
  found higher figures: 90.2 / 133.2 on sparkDash and ~5.9k on real text on four boxes, and 111.9 /
  184.9 on eight with rhys101's TP8 line ([survey](review/external.md#survey)).
- **A ring costs little once collectives are fast.** With hairpin forwarding and RoCEnante (rchmagos's
  ring work for knapcio's repo), the TP4 decode step on the ring was within 3 % of the switched one:
  33.3 / 38.1 ms against 33.0 / 39.2 ms, prose / code
  ([PR #8](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/8)).
  Prefill was 4–5 % slower, since a 4-ring's bisection is a single link.
- **Collectives dominate the stock stacks.** rhys101 cut a ~1 MB NCCL collective from 1366.8 to
  360.2 µs with 1 MiB buffers, protocols `^LL128` and 8 channels (which also freed 11.18 GiB of
  pinned memory per node), and RoCEnante took code decode from 111.3 to 134.9 tok/s
  ([progress.md](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/main/docs/progress.md)).
- **The few rows of a verify step need their own kernels.** b12x's warp-MMA path was 10–13× faster
  than CUTLASS for MXFP8 projections at M = 5–6.
- **Splitting the FP4 indexer's query rows across TP ranks** cut long-prompt prefill time by 21–28 %
  ([mia-improvements.md](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8/blob/main/docs/mia-improvements.md)).
- `antirez/ds4` has run V4.1 since 2026-09-13 (Metal, and CUDA text-only), from a 151.8 GiB Q2 GGUF
  plus the 188.8 GiB FP8 Engram tables streamed from SSD. On one GB10 it manages single-digit
  tok/s, and it wedged rank 0 on load before. ik_llama.cpp merged V4.1 on
  2026-09-21. Nothing in the llama.cpp family runs across several GB10s.

### sudoingX/dgx-spark-deepseek-v41 (evaluated 2026-10-12)

[sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) v1.1
(cc59c36, created 2026-10-12, Apache-2.0) serves V4.1-Flash on **two** GB10s, one ConnectX-7 cable,
TP=2: upstream vLLM (6e517b15) with 21 patches and an out-of-tree plugin of its own kernels, some
from bertholomus's TensorFold. The weights are re-quantized to fit two boxes: routed experts to EXL3
at 2.70 bpw (2–5 bits by measured sensitivity), dense linears and the head to 4–5 bits, the drafter
to 3–5. Its outputs are not the model's: KL 0.0243 against the native model, top-1 95.57 %. So it is
not a peer under this engine's exactness rule; it buys half the boxes with lossy weights.

Where its numbers compare (its `kit_bench.py` set b: one stream, greedy, thinking off, 384 tokens):

| | it (2 boxes, 2.7 bpw) | this engine (4 boxes, exact) |
|---|---|---|
| decode without drafts | 44.6 tok/s (22.4 ms a token) | ~56 tok/s (17.5–18 ms) |
| a verify step, prose / code | 35.9 ms at 2.09 tokens / 43.3 ms at 4.45 | 26.8–30.1 ms at 1.93–2.83 / 36.9–38.0 ms at 4.97–5.10 (sparkDash) |
| cold prefill | 2.1k tok/s at 8K–32K, random words | 8.5k–9.4k tok/s on real text, 15k–250k |
| first token, 256K cold | 159.6 s | ~27 s |

- Per token this engine is ~1.25× faster without drafts and 1.24–1.31× with them, at similar tokens
  a step, reading about the same weight bytes per box per token (its 2.7 bpw over two boxes against
  MXFP4 over four), so the gap is engine efficiency. Prefill is ~4× (2× per box).
- Its headline "structured" 141.3 tok/s is "Count from 1 to 200", where every draft is accepted, and
  its short-context runs skip the indexer (patch 0019), a saving that ends past 1,025 tokens.
- Its per-lever gains (+2 % from a paced L2 prefetch, +5 % from its ring, +18 % prose from a depth
  policy) cite a private log; the release-level speeds recompute from its raw JSON.

Worth taking, all exact: a **paced L2 prefetch** of the next dense weights during collectives and
attention (`exl3_l2pf.cu`: a one-block side-stream kernel, 8 KiB chunks paced to ~230 GB/s). It is
this engine's collectives option B by another route, without B's SMs and registers. Tested here the
same day, it overturned [Memory bandwidth](#memory-bandwidth)'s "L2 prefetch does not pay": the cause
was the issue pattern (all at once), not the gap. Paced, the lines stay for at least 300 µs.
Also a precomputed Markov-bias cache (≤ 0.15 ms a step) and per-expert flags between MoE up and down
(≤ 1 %, our MoE being at 96–97 % of its byte floor). The rest it has, this engine has (one-hop RDMA
all-reduce, per-shard draft argmax, a cost-based draft depth) or excludes (EXL3, a TF32 mHC).

### Collectives on a switchless ring

Opposite boxes have no cable, so their traffic is either forwarded by a neighbour's NIC or relayed
by a neighbour's CPU.

- **NIC forwarding** (sparkring; MiaAI's [switchless-ring.md](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/docs/switchless-ring.md)).
  A marker process on the sender (`mlx5-rdma-tx-rewrite-probe`, which must keep running) installs an
  mlx5 RDMA-TX steering rule that re-tags opposite-node queue pairs (flow label 16383, i.e. UDP
  source port 65535) from EtherType 0x0800 to 0x88b5, and a `/32` route sends them to a neighbour.
  There a `skip_sw` tc flower rule restores 0x0800, rewrites the destination MAC and sends the
  packet out of the other port through mlx5 hairpin queues, with no CPU involved. It needs devlink
  `flow_steering_mode=hmfs`, a legacy eswitch, `hairpin_num_queues=4`, `hairpin_queue_size=8192`
  (the default 1024 overflows above ~100 KB bursts), `hw-tc-offload on`, four RDMA functions per
  node across both PCI domains, and MTU 9000. An opposite-node RDMA write takes ~10 µs against
  8.5 µs direct (61 KB); an all-reduce takes 24 / 35 / 40 / 60 / 95 µs at 60 / 100 / 120 / 240 /
  480 KB. Sparkring calls it research-only.
- **RoCEnante** (b12x, [_roce_proxy.c](https://github.com/local-inference-lab/b12x/blob/master/b12x/comm/roce/_roce_proxy.c);
  [design.md](https://github.com/christopherowen/dgx-spark-networking/blob/main/docs/design.md)).
  A GPU kernel copies its input into a `cudaHostAlloc` slot and writes a sequence number to a
  control word. A CPU proxy busy-polls that word (20M spins before it sleeps) and, per peer, posts
  an RDMA WRITE followed by an inline, signalled 4-byte flag WRITE on the same RC queue pair, so the
  flag cannot overtake the data. The kernel spins on the flags with `ld.acquire.sys` and sums in
  fp32 **in fixed rank order, so every rank gets bit-identical output**. It all captures into CUDA
  graphs. Tuning the proxy's idle spins alone cut the single-stream decode step 8.6 %
  ([PR #8](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/pull/8), on the forwarded
  ring). The repo was archived on 2026-10-06; work continues in `flashinfer/experimental/b12x`.
- **Host relay** ("ring4" in [christopherowen/dgx-spark-networking](https://github.com/christopherowen/dgx-spark-networking)),
  the same one-shot protocol. The opposite rank's data goes half clockwise and half counter-clockwise,
  forwarded by the middle hosts. On four GB10s in a cable loop, BF16, the median of the slowest rank
  per sample of 256 graph replays: **16.9 µs at 10 KiB** (NCCL ~91), 28.0 µs at 60 KiB (~112),
  307.7 µs at 2 MiB (~360) on 2026-10-03, and **16.2 µs at 10 KiB**, 271.9 µs at 2 MiB with its CuTe
  kernel on 2026-10-05. That is as fast as NIC forwarding at small sizes, with none of its
  configuration (its author measured NIC forwarding, "mesh4", slower).
- **GPU-initiated RDMA** ([roadmap](https://github.com/christopherowen/dgx-spark-networking/blob/main/docs/gpudirect-roadmap.md)).
  With the GPU writing work requests and the host ringing the doorbell, half a round trip took
  4.1–4.5 µs. The GPU ringing the doorbell itself failed and rebooted a node, and DOCA GPUNetIO
  errors on GB10 ([forum](https://forums.developer.nvidia.com/t/dgx-spark-gpunetio-support/370367)).
  **Treat IBGDA as unavailable.**
- christopherowen's NCCL 2.30.7 build marks an AArch64 send-path fence patch as mandatory, or NCCL
  hangs. A proxy on GB10's Arm cores needs an explicit barrier between seeing the GPU's flag and
  handing the data to the NIC.

### DeepSeek's own kernels on SM12x

None of the publisher's optimised stack runs on GB10 as released, which is why the kits are
patchworks.

- FlashMLA: SM100/SM103 only.
- DeepGEMM: SM90/SM100 officially. SM120 kernels (GEMMs, einsum, HC prenorm, V4.1 MXFP8/MXFP4
  sparse MQA) exist only on the `nv_dev` branch ([PR #447](https://github.com/deepseek-ai/DeepGEMM/pull/447)),
  with no Mega-MoE, Mega-Gate or Mega-mHC and SM121 unqualified. A silent-corruption bug was
  reported on GB10 ([#417](https://github.com/deepseek-ai/DeepGEMM/issues/417)).
- TileKernels: SM90/SM100 and Ascend. DeepSelect: no SM12x; a draft port in vLLM's fork is slower
  than `torch.topk` in 39 of 54 cases and never ran on SM121.
- Community substitutes, tested on RTX cards only: FlashInfer's SM120/121 NVFP4 sparse-MLA decode
  ([#5914](https://github.com/flashinfer-ai/flashinfer/pull/5914)) and its W4A8 MegaMoE
  ([#4632](https://github.com/flashinfer-ai/flashinfer/pull/4632)).

### DSpark acceptance elsewhere

| Where | Setting | Acceptance |
|---|---|---|
| vLLM ([#56797](https://github.com/vllm-project/vllm/issues/56797)) | V4.1, k = 5, greedy, GSM8K | 2.82 tokens per step (reference 3.51; fixed by PR #57432); drafts 1–5 survive with probability 0.724 / 0.516 / 0.301 / 0.180 / 0.096 |
| llama.cpp PR #28696 | block 5 / block 3 | 43.6 % / 59.1 % |
| ik_llama.cpp | block 3 | 60 %, 1.41× speed-up |
| 0xSero | synthetic text, forced 8,192-token outputs | 78.9–88.4 % |
| V4 Flash on 2 GB10s ([forum](https://forums.developer.nvidia.com/t/deepseek-v4-flash-0731-dspark-1m-nvfp4-kv-2x-dgx-spark/378824)) | after a weight-mapping fix | 25.7 → 60.2 %, decode 32.7 → 55.4 tok/s |
| MiaAI kit | k = 5, confidence cap 0.1 | ~3 tokens per step on prose, ~6 on code |

On natural text acceptance falls off quickly along the draft, while every extra verified row brings
new experts to read (the verify budget in [DESIGN.md](../DESIGN.md#what-the-budgets-allow)). The
[DSpark paper](https://arxiv.org/html/2607.05147v1) reports 60–85 % faster per-user decode than MTP-1
for V4-Flash in production.

## Hardware, measured here

The four boxes are two ASUS Ascent GX10 (ranks 0 and 1) and two NVIDIA DGX Spark (ranks 2 and 3); every box is a
GB10 with 128 GB of unified memory, and the docs name them by rank.

All on rank 0 (and rank 3 for RDMA), GPUs idle, with `micro/`.

### Memory bandwidth

| | |
|---|---|
| Streaming 128-bit reads, 96–384 blocks | **232–238 GB/s** |
| `cudaMemcpy` D2D | 111 GB/s copied (222 GB/s of traffic) |

GB10's LPDDR5X is 256-bit at 8533 MT/s: **273 GB/s peak**, so ~86 % is achieved.
This is the number that sets the decode ceiling.

Others measure the same: 231–234 GB/s ([ds4#773](https://github.com/antirez/ds4/issues/773)), 235.6
([bloomery#17](https://github.com/midagedev/bloomery/issues/17)) and ~231
([Chips and Cheese](https://chipsandcheese.com/p/inside-nvidia-gb10s-memory-subsystem)). Real GEMV
kernels reach ~177. One of four nodes in [vLLM PR #60843](https://github.com/vllm-project/vllm/pull/60843)
measured 166 GB/s against 235–258 for the rest. A tensor-parallel step runs at the pace of the
slowest box, so all four get measured.

**L2 prefetch pays on GB10 if it is paced** (`micro/l2_retention.cu`; corrected 2026-10-12, master
6c3be80). The first test (2026-10-11) concluded "L2 prefetch does not pay on GB10". It issued each
CTA's whole 87–350 KB slice at once, as one `cp.async.bulk.prefetch.L2` or as `prefetch.global.L2` per
line, and the hardware drops much of that: read 300 µs later, 4 MB came back at 284–438 GB/s and
16 MB at 257–269 GB/s, near DRAM speed. The lines were mostly never brought in; they were not evicted.

- **Paced, they stay.** Issued as TensorFold issues them (one block, 8 KiB bulk prefetches in waves of
  128 at ~230 GB/s, so DRAM's queue holds about a wave), 99–100 % of 2–16 MiB reads back from L2 at
  every gap tested from 0 to 300 µs, read by the GEMVs' own loads (`ld.global.nc.L1::no_allocate`,
  48 CTAs): 16 MiB in 11.9 µs, against 71.0 cold and 11.2 from a fully resident buffer. An
  `evict_last` policy makes no difference.
- **At once, about half is lost.** The same bytes prefetched at once from 4 blocks read back 51–65 %
  from L2 at any gap, from 4 MiB up (all of it at 1.9 MiB).
- **Beside a collective.** With 40 CTAs spinning as a collective would, a paced side prefetch started
  with the window turns 77–92 % of the window into reads. Window plus read: 41.2 → 25.9 µs (4 MiB, a
  20 µs window), 68.3 → 41.6 (8 MiB, 30 µs), 135.5 → 80.3 (16 MiB, 60 µs).
- **Still true from the first test.** Loads with `.L1::no_allocate`, which is what the GEMVs stream
  weights with, leave nothing in L2 for a re-read; `ld.cg` and `ld.nc` without it do (16 MB re-read at
  1.4–1.5 TB/s). A decode step whose GEMVs prefetched their own next weights into L2 during the PDL
  window got slower (19.34 against 18.63 ms).

Whether a paced side prefetch of the next layer's weights during the collectives pays in the engine
is being measured (the six-row step study, [experiments § In progress](experiments.md)). It was
projected at −1.5 to −3.0 ms a six-row step, depending on what the prefetch costs the collective.

### Tensor cores

- `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0`
  and the `e2m1.e4m3` form produce **bit-identical results to a CPU reference** (random operands and
  scales), at **~214 TFLOPS** with 4 independent chains per warp.
- It must be built with `-gencode arch=compute_121a,code=sm_121a`. Plain `-arch=sm_121a` emitted
  PTX for `sm_121` and ptxas rejected `.block_scale`.
- Operand facts (from CUTLASS `mma_traits_sm120.hpp`, confirmed by the test):
  - A 16×32 row-major: lane = 4g+t holds bytes [g][4t..], [g+8][4t..], [g][16+4t..], [g+8][16+4t..];
  - B is stored n-major: [g][4t..] and [g][16+4t..];
  - C: the lane holds (g, 2t..2t+1) and (g+8, 2t..2t+1);
  - an **e2m1 value sits in bits 5:2 of its byte** (`0b00ABCD00`), so packed nibbles must be expanded
    and shifted before the MMA;
  - the A scale a lane supplies is the scale of row `g + 8·(lane & 1)`; the B scale is that of column
    `g`. If every lane holds its own row's and column's scale, the thread/byte selectors can be 0.
- The 214 TFLOPS is a 4-chain microbenchmark, not the ceiling.
  [nvfp4bench](https://github.com/secYOUre/nvfp4bench) reaches **~248–256 TFLOPS** dense with
  `kind::mxf8f6f4` on sm_121a and 486–512 with `kind::mxf4nvf4` (2:4 sparse ~1,014–1,022); a
  CUTLASS NVFP4 GEMM reaches ~375.
- Elsewhere the e4m3 × e2m1 form had been checked bitwise against CUTLASS on SM120 only
  ([tilelang#3099](https://github.com/tile-ai/tilelang/pull/3099), FlashInfer #4632), and nobody had
  timed it on GB10. CUTLASS DSL 4.5.0 emitted block-scale PTX that ptxas rejected for sm_121a (fixed
  in 4.5.1, [cutlass#3227](https://github.com/NVIDIA/cutlass/issues/3227)); hand-written PTX was fine.

### Ring RDMA (rank 0 `rocep1s0f1` ↔ rank 3 `rocep1s0f0`, 192.168.2.0/24, GID index 3, RC, MTU 4096)

| Payload | CPU ping-pong, one way | GPU→GPU via CPU proxy, one way |
|---|---:|---:|
| 64 B | 1.53 µs | **2.74 µs** |
| 10 KB | 3.81 µs | **5.10 µs** |
| 40 KB | 6.71 µs | **8.29 µs** |

GPU→GPU path: the GPU writes the payload and a flag into `cudaHostAlloc` memory and issues
`__threadfence_system()`; a CPU thread spinning on the flag posts one RDMA WRITE (payload + flag);
the remote GPU spins on the flag in its own host-pinned buffer. `ibv_reg_mr` on `cudaHostAlloc`
memory works, and the GPU sees NIC writes. One PCIe domain was used; both domains of a port together
reach ~185 Gb/s (measured on these boxes outside this project).

So a TP all-reduce of one token's hidden state (10–20 KB) should take ~10 µs over two hops, and the
80 per decoded token ~1 ms. The best implementations elsewhere take 16–17 µs at 10 KiB
([above](#collectives-on-a-switchless-ring)), so ~15 µs is the safer budget.

Host memory and the NIC, from NVIDIA's
[porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html) and others:

- GPUDirect RDMA, nvidia-peermem, dma-buf and GDRCopy are not supported, and `ibv_reg_mr` on
  `cudaMalloc` memory fails with EFAULT. Registered buffers are host allocations, which kernels can
  use directly.
- GPU reads of `cudaHostAlloc` memory are 7–11 % slower than of `cudaMalloc` memory. `cuMemCreate`
  with a `HOST_NUMA` location reads as fast and registers fine
  ([vLLM#60844](https://github.com/vllm-project/vllm/issues/60844)). `cudaHostRegister` on 4 KiB
  pages halves write bandwidth and can stall ~11 µs per page on first touch
  ([LTX-2#318](https://github.com/Lightricks/LTX-2/issues/318)). Weights stay in `cudaMalloc`; only
  the small collective buffers are registered.
- Whether the GPU caches `cudaHostAlloc` memory in L2 is unmeasured anywhere. RoCEnante uses
  system-scope loads for every access to it.
- On kernel `7.0.0-1019-nvidia`, `ibv_reg_mr` fails with ENOMEM without `kho=off` (~9.3 GiB of CMA
  pageblocks the NIC cannot pin;
  [MiaAI#259](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/259)). All
  four boxes boot with it (rank 0's `/proc/cmdline` confirms).
- Each ConnectX-7 function sits on PCIe Gen5 x4, ~13.6 GB/s (109 Gb/s), so a 200G port needs both
  PCI domains.

### Platform facts from elsewhere

- GB10's GPU has 48 SMs at up to 2.55 GHz, a 24 MB L2 and a 16 MB system-level cache. There is no
  `wgmma` or `tcgen05`: the tensor cores are reached through `mma.sync` only.
- CUDA graphs ([bloomery#17](https://github.com/midagedev/bloomery/issues/17)): in a 520-node decode
  graph the gaps between kernels were 0.128 µs median and 0.608 µs p99, but there was a **0.318 ms
  gap between graph replays**. A warp-level L2 prefetch of upcoming weight rows cut the decode step
  13.5 %.

## Fleet state that the engine reuses

| | rank 0 | rank 1 (1) | rank 2 (2) | rank 3 (3) |
|---|---|---|---|---|
| Overlay / SSH from rank 0 | 10.0.0.1 | 10.0.0.4 | 10.0.0.3 | 10.0.0.2 |
| Uncensored weights | `/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8` (USB4, ~1.7 GB/s) | `~/DeepSeek-V4.1-Flash-UNCENSORED-FP8` | same | same |
| Engram shards (kit format) | `~/dsv41-engram-tp4/engram-l{1,14}-r0of4.bin` | `~/dsv41-4x-ring/engram/…r1of4.bin` | `…r2of4.bin` | `…r3of4.bin` |
| RDMA devices | `rocep1s0f{0,1}`, `roceP2p1s0f{0,1}` on every box ||||

Engram shard format (the MiaAI kit's [`scripts/pack_engram.py`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/scripts/pack_engram.py)): a 4096-byte header
(`<6Q`: magic `DSV41EN1`, layer, lo, hi, rows, 264), then rows `lo..hi` as **264-byte records** (256 B
weight + 8 B scale). Rank r owns rows `[rows·r/4, rows·(r+1)/4)` with the kit's rank order, which is
also ours. A record can straddle a 4 KiB page.

## Environment gotchas

- **CUDA can fail to start when the page cache is full** (found by the collectives work): on
  rank 2, `cudaFree(nullptr)` returned out-of-memory with 0.9 GB free and over 90 GB of
  reclaimable cache. The engine initialises CUDA and allocates before reading weights, and reads
  them with `O_DIRECT`. CUDA allocations on GB10 see only free pages, so the cache also caps what the
  engine can reserve: after its prefill buffers rank 2 had 4.5–11.5 GiB free depending on the cache.
  Most of that cache was the model's own files: the checkpoint's last shard alone held 27–29 GB on
  rank 2 (`posix_fadvise(DONTNEED)` on it took MemFree from 12.7 to 41.6 GB), and rank 1 and rank 3
  freed 26 and 18 GB the same way. `run4.sh` now drops the model files' cache on every box before
  starting the ranks (`engine/server/tools/evict_cache.py`, no root needed).
- **Any other job on any GPU slows the whole ring**: with the golden reference running on rank 2, a
  10 KB all-reduce took 34 µs instead of 12. Benchmarks need all four GPUs quiet.
- **One big job on the ring at a time.** An engine run holds ~72 GB on every box, and rank 0 can hang
  under OOM (anything started over SSH there gets `oom_score_adj` −1000). A second `run4.sh` would
  also kill the first one's ranks and overwrite its binaries. So `engine/server/run4.sh` takes
  `rank 0's /data/dsgb10/ring.lock` (holder in `ring.owner`) and waits for it. Other jobs that need
  more than ~8 GB of GPU memory on any box, or that time anything, take the same lock through
  `engine/server/ring.sh <command>`; a bare `flock` registers no want, so it waits as long as the
  service serves ([engine.md](engine.md#borrowing-the-ring)).
- Kernels that spin on collective flags must have at most 48 blocks (one per SM), or they deadlock.
- SSH from rank 0 to the workers uses the overlay addresses (`10.0.0.4` rank 1, `.3` rank 2,
  `.2` rank 3); the short host names do not resolve from this session.

- **Latency on these boxes hides in power management** (found by the Engram store,
  [docs/engram.md](engram.md)): a CPU waking from idle can take up to 433 µs, so latency-critical
  threads spin on a pinned big core (CPUs 5–9, 15–19); the SSDs drop into power states after
  ~100 ms idle and the next read costs 1.1 ms (rank 0, rank 1) to 9.5 ms (ranks 2, 3), 10–20 ms after 2 s, so the
  store keeps them awake with one read every 5 ms; rank 2's and rank 3's Samsung drives hold completions until
  8 are pending or 200 µs pass, so read bursts are padded to 8 there.

- **A job started with `nohup … &` inside one tool call was killed when the call returned** (the
  golden run's log ends `Terminated` after layer 0). Long jobs need `setsid … < /dev/null &` or the
  harness's background mode.
- The reference's first 49 s per layer came from torch copying out of the safetensors mmap
  (~140 MB/s on GB10, from NVMe too), not from the USB drive; plain reads bring it to ~3.5 s per
  layer. The 135-token prose pass takes 238 s on rank 2, the 28.5k-token one 1,038 s.
- The publisher's tilelang `sparse_attn` cannot run 64 heads on GB10: it needs 141 KB of shared
  memory against a limit of about 100 KB.
- `nvcc` may not be on PATH: it is `/usr/local/cuda-13.0/bin/nvcc`, which engine/CMakeLists.txt uses
  unless `CMAKE_CUDA_COMPILER` is set. CUDA 13 removed
  `cudaDeviceProp::clockRate`.
- The SGLang image `dsv41-4x-spark:canary-roce` has torch 2.13 (cu130) with `float8_e8m0fnu` and
  `float4_e2m1fn_x2`, transformers 5.12, safetensors 0.8, and CUTLASS headers under
  `/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include`. It is the golden-data
  runtime.
- The kit's API key is in a file of mode 600; pass it by file, never
  on a command line.
