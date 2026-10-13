# ds41f-4xgx10-switchless

One configuration of [dsgx10](../README.md). It was developed as **dsgb10**, and that name stays on its binaries (`dsgb10_rank`), on its files on the boxes (`/data/dsgb10`, `~/dsgb10`) and on the service. Commands below run from this directory.

A from-scratch inference engine for **DeepSeek-V4.1-Flash** on **four GB10 boxes**
(rank 0, ranks 1, 2, 3) cabled in a switchless ConnectX-7 ring. It is specialised to that
one model and that one fleet, and it aims at prefill and single-stream decode speed for agentic use.
The checkpoint it was built and validated on is a community variant,
[dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8)
at revision `d61c59ea` (MIT, derived from deepseek-ai/DeepSeek-V4.1-Flash): it shares the official checkpoint's
architecture, config, tokenizer and tensor formats, and differs in the values of shards 13–39 (layers 10–36).
Production serves that variant, and its served model name, `deepseek-v4.1-flash-uncensored`, reflects it.
Since 2026-10-13 the figures to publish are measured on deepseek-ai's official checkpoint,
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) at `fb2764a5`: "Where it stands"
below, with the SGLang kit measured on the same checkpoint the same day, and the reference hashes
([engine.md](docs/engine.md#the-official-checkpoints-weights-2026-10-13)). The other figures in these docs,
correctness included (the golden data from `model.py`, the SGLang comparisons), are the variant's unless they name
the official checkpoint ([docs/review/kit.md](docs/review/kit.md), [docs/review/correctness.md](docs/review/correctness.md)).

"From scratch" means C++/CUDA (runtime API + inline PTX), ibverbs, and plain Linux I/O for the engine,
with a Rust front-end for HTTP, tokenisation and chat encoding. It does not build on vLLM, SGLang,
llama.cpp, NCCL, CUTLASS or FlashInfer. Python appears only in offline tooling: the golden-data
reference and the benchmark client.

**Terms used below.** The four boxes are *ranks 0–3*: two ASUS Ascent GX10 (ranks 0 and 1) and two NVIDIA
DGX Spark (ranks 2 and 3), each a GB10 with 128 GB; rank 0 runs the scripts and the front-end. *The ring* is their ConnectX-7 cabling, which one
job holds at a time. *TP=4*: tensor parallelism over the four boxes (*EP*, expert parallelism, is not
used). *The SGLang kit*: MiaAI-Lab's SGLang-based deployment of the model
([findings](docs/findings.md#the-miaai-lab-kit-what-served-here-until-2026-10-11)), which served on this ring
before this engine, over NCCL; *the MiaAI kit* is the same kit as its authors publish it, on a switched
RoCE fabric with *RoCEnante* (b12x's one-shot RDMA all-reduce). *pi*: the coding agent
([earendil-works/pi](https://github.com/earendil-works/pi)) that uses the service. *dsfront* is the
front-end, *dsgb10_rank* the engine's process on each box. From the model: *CED*, the causal
encoder-decoder, whose prefill runs the 20 encoder layers over the prompt and the decoder over its last
128 tokens ([ced.md](docs/ced.md)); *SWA*, sliding-window attention; *mHC*, the hyper-connections that
carry the residual as four mixed copies; *Engram*, the n-gram lookup tables of layers 1 and 14
([engram.md](docs/engram.md)); *DSpark*, the model's own draft module for speculative decoding (what
SGLang calls its MTP drafts). From the measurements: *k = 1*, decode without drafts (one token a step);
*c1*, one stream at a time; *B1–B4*, the batching steps of [batching.md](docs/batching.md#plan); *B′*, the
paced L2 prefetch; *RC5*, release candidate 5, the binaries production runs, built from c247a78;
*M0–M9*, the milestones of [DESIGN.md](DESIGN.md#milestones), M9 being the move to production. *The
owner* owns the boxes and decides what ships.

## Where it stands (2026-10-13)

On deepseek-ai's official checkpoint, with release candidate 5's engine code (built from 530af7f), through the
front-end, against the SGLang kit
([MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) at
`cad252b`, v2.1, TP=4 over NCCL, as it served on this ring before this engine) on the same checkpoint and ring the
same day, with the same benchmark (`bench/bench.py`: both with their speculative drafts, greedy, 512 tokens, one
request at a time). Ours is the mean of six sessions on 2026-10-13, three before the kit's and three after, each a
fresh engine; their decode requests lay within 1.3 % of each other on every prompt type. The kit's is the mean of
three sessions after a warm-up session, in one start; its decode requests lay within 0.7–2.2 %. The kit was set
up to batch up to 16 requests; throughput under concurrency is not compared.

| | dsgb10 | SGLang kit |
|---|---|---|
| decode tok/s: prose / code / agentic / JSON | **77.4 / 115.4 / 102.9 / 153.7** | 50.9 / 85.6 / 76.3 / 107.7 |
| time to first token, short prompts | **124–156 ms** | 219–253 ms |
| cold prefill, 16k–57k tokens | **8.5k–9.2k tok/s** | 4.3k tok/s |
| agentic: first turn (15.8k tokens) | **1.81 s** | 3.72 s |
| agentic: turns adding ~2k tokens, to first token | **0.39–0.42 s** | 0.68–0.79 s |

Both columns' time to first token leave out each session's first request (ours 164–168 ms, after an engine
start; the kit's 215–218 ms).

Without drafts the engine decodes 59.1 tok/s on every prompt type; greedy or sampled, drafts change
the speed, never the tokens. On this checkpoint it answers 99 of GSM8K's first 100 test problems (the kit, the
same day: 100) and finds the passkey in 18 of 18 prompts of 33k–220k tokens. Greedy output is deterministic for
a given server state, and a repeat answered from another state of the same prompt can differ at a near-tie (4 of
10 repeats on some prompts), as it does between any two stacks that sum in another order
([correctness review](docs/review/correctness.md)). As pi uses it (the agentic-edit
bench at dsfront's sampling defaults, thinking off, twelve tasks three times): 138.5 and 134.2 s end to end,
decoding at 137.4 and 138.5 tok/s, with 229–230 ms to the first token (median). On the variant the engine was
built on, release candidate 5 decoded 75.7 / 111.7 / 109.2 / 149.8 tok/s that morning, at the same prefill and
agentic times ([history](docs/engine.md#where-it-stands-on-release-candidate-5-2026-10-13)). Details in
[docs/engine.md](docs/engine.md#where-it-stands-on-the-official-checkpoint-2026-10-13).

The table compares dsgb10 with the MiaAI kit's TP4 line
([knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4),
on a switched RoCE fabric), whose decode figures were the best published for four GB10s at the
checkpoint's native precision on 2026-10-13; both run deepseek-ai's official checkpoint. It uses that project's
own benchmark, [sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8, on its own prompts (256 tokens, greedy,
one stream; ours the mean of two sessions, the kit's the better of two sweeps). No published native-precision
four-GB10 figure we found beats ours on any row; the nearest is spark-ds41f's code step, 1.7 % shorter but
carrying a third fewer tokens, and its 5.7k–5.9k is the best real-text prefill
([survey](docs/review/external.md#survey)).

| | dsgb10 | MiaAI kit |
|---|---|---|
| decode tok/s: prose / code / structured / JSON | **95.0 / 151.0 / 185.8 / 150.2** | 89.9 / 131.0 / 158.7 / 128.4 (v2.4, 2026-10-12) |
| sparkDash's prose / structural catalogs, one stream (ours: 256 tokens each, the mean; the kit's: its own harness) | **78.1 / 122.3** | 61.1 / 99.0 (v2.2) |
| cold prefill, real text, 15k–120k tokens | **8.7k–9.5k tok/s** | 4.8k–5.0k tok/s (v2.1; the best published, spark-ds41f's: 5.7k–5.9k) |

The kit's figures are from its documents, not re-measured here, and a stack that sums in another order
follows another greedy text on the same prompt, so the single-prompt rows compare slightly different
texts. The time per step is 27.9 ms at 2.63 tokens on sparkDash's prose prompt and 34.8 ms at 5.20 on its
code prompt, against the kit's 31.2–32.5 ms at 2.27 and 37.4–37.8 ms at 3.87 on its own, unpublished probe
prompts (v2.2); neither that nor the catalogs is like for like
([details](docs/engine.md#where-it-stands-on-the-official-checkpoint-2026-10-13)).

**Several streams at once** (2026-10-13, on the variant weights; [batching.md](docs/batching.md#b3-the-build-2026-10-13)):
with four slots, up to eight requests decode together, each with its own drafts. Four streams of different text
(1k-token prompts, 512 tokens each, `bench/multistream.py`) decode 136 tok/s in all, three 127, two 109, one
alone 97; batches without drafts (the step before) gave 121 / 107 / 88, and one stream at a time gives 97
whatever their number. One stream is unchanged: A/B/B/A against the engine before, six runs each, 8.322 against
8.322 ms a token with drafts. Every request's tokens and log-probabilities are its run's alone.

## How to use

From four GB10 boxes to an OpenAI-compatible endpoint. Commands run on rank 0's box, from this
directory, unless a step says otherwise; the times are this project's. Each command ran here on
2026-10-13, on deepseek-ai's checkpoint, with other paths, unless marked *not run on this fleet during the review*.

**1. What you need.**

- Four GB10 boxes (128 GB each), ranks 0–3, rank 0 reaching the others by SSH without a password,
  cabled as a directed ring of ConnectX-7 links: port f0 of rank r (the fleet file's `NEXT_DEV0` and
  `NEXT_DEV1`, its RDMA devices on PCI domains 0 and 2) to port f1 of rank r + 1 (`PREV_DEV0`,
  `PREV_DEV1`), and rank 3's f0 to rank 0's f1 ([ring.md](docs/ring.md#the-ring)). A link's ends have IPv4
  addresses in a /24 of their own, with RoCE v2 GIDs at one index everywhere (`bash micro/ring_links.sh`).
- Every box as here: Ubuntu 24.04, kernel 7.0.0-1019-nvidia, driver 580.178.04, rdma-core 50, booted
  with `kho=off`, without which `ibv_reg_mr` fails (`grep -o kho=off /proc/cmdline` shows it; to set it,
  not run on this fleet during the review: `GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT kho=off"`
  in a file in `/etc/default/grub.d/`, `sudo update-grub`, a reboot).
- To build, on rank 0: CUDA 13.0 in `/usr/local/cuda-13.0`, `libibverbs-dev`, CMake 3.28 or later,
  g++ 13.3 (C++20), the Rust toolchain (cargo 1.97) and Python 3.12.
- Disk, best NVMe: on each box its weight file (77.7 GB), its Engram shards (50.7 GB) and 64 GiB to spill
  to; on rank 0 also the checkpoint (510 GB) and room to make one rank's files. A rank takes ~90 GiB of memory.
- The checkpoint, [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  at revision `fb2764a5`: measurements from 2026-10-13 on use it, and step 8's hashes are its. The
  engine converts its FP8 and FP4 weights to its own layout and re-quantises nothing.

**2. Build** on rank 0, for `sm_121a` (GB10 alone) with CUDA 13.0's `nvcc`; the scripts copy the binaries
to the other boxes. More jobs take more memory, and an OOM can hang a GB10:

```bash
cmake -S engine -B engine/build && cmake --build engine/build -j4    # 2 s + 54 s
(cd frontend && cargo build --release -j4)                           # 13 s once its crates are fetched
```

**3. The fleet file.** [`fleet/homelab.conf`](fleet/homelab.conf) describes this project's boxes. Copy it,
give its keys your values, and `export FLEET=~/fleet.conf` (your copy) wherever engine/server/run4.sh,
ring.sh, service.sh (`status` and `stop` too) or engine/comm/run4.sh run; all but ring.sh install it beside
the engine on every box as `fleet.conf`, which the engine reads. Lines are `KEY=value` or `#` comments,
values have no spaces or quotes, and in a path a leading `~/` is the home on the box using it, `{rank}` the rank.

| Key | What it is |
|---|---|
| `NEXT_DEV0`, `NEXT_DEV1`, `PREV_DEV0`, `PREV_DEV1`, `GID_INDEX` | port f0's RDMA devices on PCI domains 0 and 2 (to rank r + 1), and port f1's (from rank r − 1), the same names on every box; the RoCE v2 GID index of the links' addresses, the same on every device |
| `BIN_DIR`, `LOG_DIR`, `SERVICE_DIR`, `RING_LOCK` | on every box: the engine and `fleet.conf`, the ranks' logs, service.sh's state; on rank 0, in a directory that exists: the lock every job on the ring holds |
| `RANK<r>_NAME`, `RANK<r>_HOSTNAME`, `RANK<r>_SSH` | the box's name in logs (optional); its `hostname`, by which a rank started without `--rank` finds itself; how rank 0 reaches ranks 1–3 (an address, or `user@host`) |
| `RANK<r>_NEXT_IP0`, `RANK<r>_NEXT_IP1`, `RANK<r>_PREV_IP0`, `RANK<r>_PREV_IP1` | its addresses on the links to rank r + 1 and from rank r − 1, PCI domains 0 and 2; `PREV_IP0` also takes the start's TCP handshake (port 18620) |
| `RANK<r>_WEIGHTS`, `RANK<r>_HASHMAP`, `RANK<r>_ENGRAM` | its weight file, the Engram hash map, the directory of its two Engram shards (steps 4 and 5); `--weights`, `--hashmap`, `--shard1` and `--shard14` override them |
| `RANK<r>_ENGRAM_READ_BATCH` | optional, 1 if absent: the Engram store's SSD reads go in multiples of this, for an SSD that signals completions several at a time; no result depends on it |
| `RANK<r>_SPILL` | its spill directory: the session store's disk tier and the kept sessions (`--disk-dir`) |

Both refuse a line they cannot read and an unknown or repeated key, naming the file and the line. Before
the ring starts, each rank checks the GID at `GID_INDEX` of its four devices against the addresses the
file gives their links, and stops at a mismatch, naming the device, its GID, the key and the file. Then
`engine/server/ring.sh engine/comm/run4.sh engine/build/comm/ring_test` checks the links and the
collectives (52 s; each rank prints `passed`).

**4. The four weight files**, cut on rank 0 from the one checkpoint
([layout.md](docs/layout.md#repeating-it-with-one-copy-of-the-checkpoint) has the whole procedure):

```bash
B=engine/build/weights CKPT=~/DeepSeek-V4.1-Flash OUT=~/rankfiles && mkdir -p $OUT
$B/dsw_convert --ckpt $CKPT --rank 0 --plan                     # the headers alone: 1,609 tensors, 72.38 GiB a rank
r=1                                                             # then 2, 3 and 0
$B/dsw_convert --ckpt $CKPT --rank $r --out $OUT/rank$r.dsw     # 23 s from NVMe, 128-150 s from USB4; 1.2 GB of GPU
$B/dsw_verify --ckpt $CKPT --file $OUT/rank$r.dsw --threads 4   # every byte un-tiled and compared: 39 s; 200-399 s
```

Copy the file and `dsw_load` to rank r's box, where `RANK<r>_WEIGHTS` says (not run on this fleet during
the review; layout.md's copies took 117–164 s), and load it there once, which checks every tensor's CRC on
the GPU and that the file is rank r's (8.4 s, then 4.4 s): `./dsw_load ~/models/rank1.dsw --verify-crc --rank 1`.
The paths can differ by box or be one with `{rank}` (`~/models/rank{rank}.dsw` in the fleet file, or
`--weights '~/models/rank{rank}.dsw'` to the scripts). A rank stops at another rank's file, naming both.

**5. The Engram store**: each rank's two shards of the Engram tables, and one hash map for all:

```bash
engine/build/engram/engram_pack $CKPT $OUT $r      # engram-l1-r${r}of4.bin, engram-l14-r${r}of4.bin: 50.7 GB, 63 s
python3 engine/engram/py/make_hashmap_stdlib.py $CKPT $OUT/engram_hashmap.bin    # 518,048 bytes, 0.3 s
```

They go where `RANK<r>_ENGRAM` (the shards' directory) and `RANK<r>_HASHMAP` say. The shards are what the
MiaAI kit's [`pack_engram.py`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/scripts/pack_engram.py)
`--tp 4` writes, byte for byte; the map script fails unless its map is the one the engine was validated
with. At its start the engine checks each shard's header and the map's constants, naming a bad file.

**6. Start serving** (not run on this fleet during the review, where the service is production's):

```bash
(umask 077; openssl rand -hex 24 > ~/api-key)
DSGB10_KEY_FILE=~/api-key DSGB10_FRONT_ARGS="--tokenizer $HOME/DeepSeek-V4.1-Flash/tokenizer.json --model-name deepseek-v4.1-flash" \
    engine/server/service.sh start      # returns once it serves: ~47 s
engine/server/service.sh status        # exit 0: serving
```

`start` takes the ring lock, copies the engine, dsfront (`DSGB10_BIN`, `DSGB10_FRONT_BIN`; step 2's by
default) and `fleet.conf` to every box, drops the model files' page cache, wants 100 GiB available on each
box, starts the ranks and dsfront on `DSGB10_ADDR`:`DSGB10_PORT` (127.0.0.1:18893; never 0.0.0.0) and
restarts what fails. `DSGB10_ENGINE_ARGS` (`--max-ctx 262144`) takes engine flags such as `--weights`
([all settings](docs/engine.md#as-a-service-servicesh)). To check it (these ran against a test port):

```bash
(umask 077; { printf 'Authorization: Bearer '; cat ~/api-key; } > ~/api-key.header)
URL=http://127.0.0.1:18893/v1 H=~/api-key.header
curl -s $URL/models -H @$H                  # {"object":"list","data":[{"id":"deepseek-v4.1-flash",…
curl -s $URL/chat/completions -H @$H -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Reply with one word: pong"}], "max_tokens": 20}'
curl -sN $URL/chat/completions -H @$H -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Count to five."}], "stream": true}'    # data: lines, data: [DONE]
```

`service.sh stop` stops it in about 5 s; `restart` starts it again with the shell's settings. A planned
stop keeps the sessions on each box's disk for the next start of the same build (`stop --forget` drops
them). A job through ring.sh or run4.sh borrows the ring once the service has had no request for
`DSGB10_YIELD_IDLE_S` (600 s); a request stops it (exit 76) unless it has `PREEMPT=0` ([operations.md](docs/operations.md)).

**7. Use it.** dsfront serves OpenAI's chat completions (`POST /v1/chat/completions`, streamed with
`"stream": true`, `tools` answered by `tool_calls`), `GET /v1/models` and `GET /health`, under the name
`--model-name` gives (`deepseek-v4.1-flash-uncensored` by default; a request's `model` is ignored).
Thinking is off unless asked for, by `"chat_template_kwargs": {"thinking": true}` or `"reasoning_effort"`
(`"low"`, `"high"`, `"max"`); the reasoning comes back in `reasoning_content`. Sampling defaults to
temperature 1.0 and top-p 0.95 (0 is greedy; `top_k` and `seed` are taken), `max_tokens` to its cap,
32,768. With no session id, a request that extends an earlier one's messages resumes from its state;
`usage.prompt_tokens_details.cached_tokens` counts the prompt tokens reused. For pi (0.84.3 here), put
[bench/piturn/models.json](bench/piturn/models.json)'s `dsgb10` provider in `~/.pi/agent/models.json`
with the service's `baseUrl` and the served name as its model's `id`, then run
`DSGB10_DEV_KEY=$(cat ~/api-key) pi --provider dsgb10 --model deepseek-v4.1-flash`.

**8. Check a build:**

```bash
python3 golden/make_cedcheck.py --tokens-only --out ~/golden    # ~/golden/long-tokens: the 28,489-token transcript
G=~/golden/long-tokens C=~/golden/long-tokens EXPECT=official golden/refhash.sh
```

Eight runs through run4.sh, four to a ring job (4–5 minutes each), print their hashes beside deepseek-ai's
checkpoint's: `tf` c7605e0b6ba8d9d2; `k1`, `nol2pf`, `fp8` and `bf16` 7941647b162df1a8; `temp`
02ef3e0858d7eb45; `ced16` 8790ee9388f4abd8; `ced28` de99bfce6ea96ba8; a build that gives them computes
what this one does. Weight files elsewhere than the fleet file says: add `-- --weights '~/models/rank{rank}.dsw'`
([golden/README.md](golden/README.md#checking-a-build) has the runs' arguments and the variant's hashes).
Then 17 checks that resumed sessions give what cold runs give, on an engine with its own spill directory:

```bash
rm -f ~/test.sock; engine/server/run4.sh --mode serve --socket ~/test.sock --disk-dir '~/test-spill' > ~/test.log 2>&1 &
until [ -S ~/test.sock ]; do sleep 1; done                             # ~48 s
python3 engine/server/tests/serve_test.py --socket ~/test.sock --ced    # serve_test --ced: 17 of 17 checks pass
engine/server/run4.sh stop
```

**9. Measure.** `bench/bench.py` (decode by prompt type, cold prefill, agentic turns) against any
OpenAI-compatible endpoint, `bench/bench_record.py` (it, with every request recorded), `bench.py --modes
edit` (pi-style coding tasks) and `bench/sparkdash/part.sh` (MiaAI-Lab's sparkDash): their commands are
in [Reproducing what exists](#reproducing-what-exists); not run on this fleet during the review.

## Status

The rows are in the order the work was done, and each gives its figures as they were measured then, so
a later row supersedes an earlier one (the DSpark module "not yet wired into the decode loop" is wired
a few rows down; the Serving row's 52.8 tok/s and 2.1k tok/s prefill are the first end-to-end figures).
The current figures are those of "Where it stands" above.

| Step | State |
|---|---|
| Every experiment, failed and dropped ones included: question, numbers, verdict, where the code lives | index → [docs/experiments.md](docs/experiments.md) |
| Research: model spec, tech report, prior recipes | done → [docs/findings.md](docs/findings.md) |
| Baseline: the SGLang TP=4 kit as it served on the ring | measured → [DESIGN.md](DESIGN.md#baseline) |
| Hardware microbenchmarks (bandwidth, block-scaled MMA, RDMA latency) | done → [docs/findings.md](docs/findings.md#hardware-measured-here) |
| Golden-data reference (pure torch, layer streaming) | done → [golden/README.md](golden/README.md): prose (135 tokens, all positions) and an agentic transcript (28,489 tokens) in rank 0's /data/dsgb10/golden/; an accuracy collapse in the long one turned out to be the model's own behaviour (layer 1's Engram), seen with the variant's weights and, under `model.py`, with deepseek-ai's official weights too |
| Per-rank weight files: partition, converter, loader, verifier | done → [docs/layout.md](docs/layout.md); format v2 (the kernels' tile layouts, compressor and mHC splits) verified byte for byte on all four boxes and read correctly by the kernels |
| Front-end: BPE tokeniser, chat encoding, OpenAI server, rank-0 protocol | done against a mock engine → [docs/protocol.md](docs/protocol.md); tokeniser and encoding exact vs the reference, pi works unchanged |
| Engram store: exact hashing, io_uring row store, cache, prefetch | done → [docs/engram.md](docs/engram.md); decode rows ready in < 0.5 ms worst case, prefill reads ≥ 6× faster than 10k tok/s needs |
| Attention-side kernels: RoPE, KV caches, compressor, indexer, sparse attention, mHC, DSpark attention | done and fused → `engine/attn/README.md`; exact vs `model.py` except summation order; the decode attention side + hc pre takes 1.6–1.7 ms per step (unfused: 4.75), 3–6 launches per layer |
| Decode weight-streaming kernels: FP8/bf16 GEMVs, MXFP4 MoE, wo_a, head, norm, router | done → `engine/kernels/README.md`; 95–108 % of 235 GB/s, all exact or within 1 bf16 ulp; a k=1 step's weight reads take 13.9 ms per rank |
| Ring collectives over ibverbs with a CPU proxy | done → [docs/ring.md](docs/ring.md); 10 KB all-reduce 12.3 µs (christopherowen's host relay: 16.2–16.9), 80 per step 0.98 ms, bit-identical on all ranks, 0 errors in 3M randomised runs |
| TP=4 engine (layer runner, rank binary, golden checks, decode graphs) | done → [docs/engine.md](docs/engine.md); matches the golden data within the reference's own numerics (on the 28.5k-token transcript its top-1 agreement equals that of `model.py`'s own split-K variant within 0.01 in every band, and it drifts less); **decode step 19.4 ms at 1 row, 37.3 ms at 6** (1k context), plus 0.8–1.9 ms per token on the host |
| Prefill GEMMs on tensor cores (dense FP8, bf16 wo_a, grouped MXFP4 MoE) | done → `engine/kernels/README.md`; 20 encoder layers' GEMMs: 9.7k tok/s at 2048-row chunks, 13.5k at 4096, 17k at 8192 (bound by reading each layer's experts once per chunk) |
| Single-rank decode step (fused attention + GEMV chain, collectives stubbed) | 15.0–15.4 ms at 1 row, 34–35 ms at 6 rows, flat from 1k to 128k context |
| DSpark draft module (3 blocks, Markov and confidence heads) | done → `engine/dspark/README.md`; matches `model.py` (identical acceptance on the prose run), 2.8 ms per draft pass per rank; not yet wired into the decode loop |
| CED prefill reference (Decoder SWA Bounded Replay around `model.py`) | done → [docs/ced.md](docs/ced.md); this checkpoint's decoder global KV is exact under CED; the first token is unchanged in all six long cases and decode stays inside `model.py`'s own numerics envelope; golden data in rank 0's /data/dsgb10/golden/ced/ |
| Prefill attention side: chunk sparse attention, indexer, compressor, mHC; CED's window cut and 128-row decoder chunk | done → `engine/kernels/README.md`; checked against `model.py` (indexer scores and compressor entries bit-identical, attention 90–95 % bit-exact); 4096-row encoder chunks take 80–88 ms per rank for 20 layers' attention side, **~10.5k tok/s with the GEMMs** from 16k to 128k context (before collectives); the decoder replay's attention side is 7–9 ms next to 144 ms of GEMMs |
| Serving: rank-0 sessions with exact prefix reuse, sampling, a self-driving decode loop, the many-row prefill path, front-end end to end | done → [docs/engine.md](docs/engine.md); pi completes multi-turn tool tasks through `dsfront`, every turn reusing the whole previous one. Against the SGLang kit through the same benchmark: decode **52.8 tok/s** at k = 1 on every prompt type (kit 46.8 prose, 73.6–106.9 elsewhere with its MTP drafts), time to first token **116–145 ms** (kit 244–280), cold prefill 2.1k tok/s (kit ~4.2k: ours is still exact mode with the collectives at 37 % of a chunk and nothing overlapped), agentic turns of +2k tokens 0.92–1.06 s (kit 0.71–0.79) |
| Prefill: sequence-parallel, collectives under compute, CED as the serving default | **done** → [docs/engine.md](docs/engine.md#prefill), [docs/ced.md](docs/ced.md). Through the front-end against the kit: cold prefill **6.9k–7.2k tok/s** from 16k to 57k tokens (kit 4.2k–4.3k; the best published on four GB10s as of 2026-10-13, ~5.9k on real text), the 15.8k-token first agentic turn **2.25 s** (kit 3.73), turns adding ~2k tokens **0.41–0.44 s** to first token (kit 0.71–0.79). Since then wo_b by all-to-all and the MoE combine inside its reduce-scatter (same bits) take the 28.5k-token transcript to 3.52 s in the engine (8.1k tok/s, 8192-row chunks). CED matches model.py's CED as closely as exact mode matches exact; `--exact` opts out |
| Decode-path fusion: mixes off the critical path, all-reduce epilogue over 40 SMs | done → [docs/engine.md](docs/engine.md); decode loop 18.90 → **18.56 ms/token** at 1k (53.9 tok/s), 18.98 → 18.63 at 16k; the producer-side all-gather and the mixes on the all-reduce's path measured and dropped. At one row a step is ~16 ms of weight streaming (floor 14.4) plus ~2.3 ms of collectives |
| DSpark in the decode loop: confidence-gated verify steps, seeding from the prompt (CED's replay rows), drafter windows in the session state | **done** → [docs/engine.md](docs/engine.md#drafts-dspark); emits exactly the tokens decode without drafts emits (greedy and sampled); 1.81 / 3.17 / 3.09 / 4.90 tokens per step on prose / code / agentic / JSON; the draft pass costs 2.6 ms a step (3.4 before 9 of its 25 collectives went and its head moved to an FP8 copy, which changes which drafts are proposed, never the tokens) and each verified row 3.6 ms |
| Long-context decode, 32k–256k: the dense index layers in one launch per step, one step graph per row count at any context limit | done → [docs/engine.md](docs/engine.md#long-contexts); a one-row step is 0.18 ms slower at 128k than at 1k (was 0.30) and 0.38 ms at 256k (was 0.60): decode 18.73 / 18.93 ms per token; the spec step's graph is 0.24 GiB at any limit (was ~13 GiB at 256k); same tokens at every context, 250k-token sessions extend and fork like short ones |
| Service and soak: `service.sh` (supervised ranks and front-end, ring lock, restarts, rotated logs; never 0.0.0.0 or :8888 before M9), an agentic soak through the front-end | done → [docs/engine.md](docs/engine.md#as-a-service-servicesh); one hour from three clients, 2,183 requests (regenerates, forks, cancels, disconnects, compactions, sessions to 158k tokens): no errors, memory and step time flat |
| Persistent GEMV: the normed GEMVs at one row (wq_a + wkv, wq_b, the router gate) stream their weights through a `cp.async` ring that fills while the collective before them runs | done → [docs/engine.md](docs/engine.md#what-remains-for-15-ms-at-k--1); decode loop **18.60 → 18.31 ms/token** at k = 1 (54.6 tok/s), spec steps unchanged to slightly faster; bit-exact against the classic kernels at every decode shape, the same hashes everywhere. Then the MoE router's loads issued all at once (each `k_moe_up` CTA waited 10.6 µs for 12 dependent round trips): **18.00 ms/token** (55.6 tok/s). While weights stream they now run at 234.6 GB/s, the floor; what is left is the ~3.9 ms of a step with no weights streaming (collectives 2.7, attention 0.8, the indexer's side GEMV 0.5) |
| Agentic turns at long context (prefill side): the dense index scans on the tensor-core top-k kernel, a per-request time split in rank 0's log | done → [docs/engine.md](docs/engine.md#long-contexts); time to first token is flat from 16k to 250k: **~0.22 s** for +200 tokens, **~0.42 s** for +2,000, **~1.0 s** for +8,000; cold prefill **9.4k tok/s at 128k, 9.3k at 250k**. A turn's ~0.2 s floor is streaming the weights (encoder chunk + the 128-row replay), not the context |
| Long sessions: window snapshots thinned with their distance from a session's end; evicted sessions spilled to each rank's disk and read back on a revisit | done → [docs/engine.md](docs/engine.md); 128k sessions take 189 MiB of the store (481 before), so it holds ~65 of them; a 180k-token session evicted and revisited gives its first token in **0.37 s** (19.3 s cold), the same tokens as if it had never left |
| Two-chunk prefill wavefront | probed, not built: 7 % at 8192 rows, below its 15 % bar ([What remains for prefill](docs/engine.md#what-remains-for-prefill)) |
| Decode: wo_a loading before attention ends (the classic kernel's two batches, and a persistent `k_woa_p`), the indexer's wproj on the row kernel, the pre-norm GEMVs at 2–8 rows (PRE2) | measured and not merged → [docs/engine.md](docs/engine.md): an A/B on the ring (2026-10-12) gave every reference hash, but none 2 % faster (the wo_a prefetch neutral, `k_woa_p` 1.4 % slower, PRE2 1.0 % faster with drafts, the wproj swap neutral). They stay, off, on branch `worktree-agent-a1102a1670c38111d`. The persistent normed GEMVs at 2–6 rows and the dynamic-split persistent GEMV lose and are dropped |
| Prefill MoE kernels at turn sizes: `k_moe_down_g` (a CTA per tile and column group), up's L2 weight prefetch, a tile order mixing busy and quiet experts, a 16-byte tile entry (all off by default); the mHC boundary's rows per CTA chosen by the chunk's rows (same bits by construction) | merged with the defaults off (7862d51): the same bits on the GPU (tf 842278b28d87dad0, decode 81f772ab79a950e4 after a 16k CED prefill, with every knob on; the long28k CED hash 14650fb0739d0383 was the hash's seed: until 2026-10-13 that run printed it whatever it computed ([docs/review/correctness.md](docs/review/correctness.md), M1; it now gives 8f08a153d415c5c8, [golden/README.md](golden/README.md#checking-a-build)); serve_test --ced 17/17). The kernels gain alone (geomean 0.948 / 0.957 / 0.823 per knob), but agentic turns with the tuned knobs gain only 1.1 % (geomean 0.989, none slower), under the 2 % bar, so they stay off ([docs/engine.md](docs/engine.md)) |
| Production sharing the ring with development: a job that waits for the ring borrows it once the service has had no request for 2 minutes (10 until 2026-10-12 11:14) and not within 2 minutes of its engine's start; dsfront holds requests meanwhile (503 with `Retry-After` past 15 minutes). A request that comes during a borrowed run preempts it (TERM, ranks stopped, KILL after 10 s, exit 76; `PREEMPT=0` opts out), and the ring comes back at once | done → [docs/engine.md](docs/engine.md#borrowing-the-ring); 42 checks against the mock engine (`tests/yield_test.sh`); in production since 2026-10-12 08:16. On the ring (before preemption): lent 600 s after the last request, the job's hashes as usual, a request held during it answered 82 s after it came, once the engine was back. Fixed at 10:12 (11f8343): a request that waited and left did not count as activity, so the ring was lent again 2 s after it came back while pi waited. Jobs use `run4.sh` or `engine/server/ring.sh`, never a bare `flock`. Since 2026-10-12 12:59 in production: jobs take the ring in the order they asked, a driver of parts holds a want (`ring_want_hold`) so the service stays lent between them, and dsfront answers a stream that has to wait at the gate at once and keeps it alive with SSE comments (`yield_test.sh` 56 of 56) |
| Kept sessions: a planned stop (`stop`, `restart`, a lend) keeps the sessions on each box's disk tier and dsfront's table, and the next start of the same build takes them back; a crash or another build starts clean | done → [docs/engine.md](docs/engine.md#kept-sessions); on the ring: a stop takes 4.8–5.5 s, a start is unchanged (46.7 s, of which 196–201 ms take the sessions back), and a 64.6k-token session's next turn after a restart gives its first token in 0.347 s (6.89 s cold), bit-identical to the same turn with no restart. The files hold the conversations: mode 0600 in 0700 directories, never in a repository or the logs; `service.sh forget` removes them. In production since 2026-10-12 11:14 (up to 4 GiB per stop) |
| Several sessions per step | B1 in production, B2 merged (off), B3 estimated and parked, B4's halves merged: plan in [docs/batching.md](docs/batching.md#plan) (B1 time-slicing, B2 several sessions per step, B3 drafts across sessions, B4 16 rows). B2's attention kernels take a session slot table, merged (1b79a71): 1656 checks of 2–4 sessions in shuffled slots identical to each alone, the no-table path equal to master within noise. **B1 in production since 2026-10-12 12:59**: several requests in flight, each decoding in slices of 64 tokens while others wait (a lone request is not sliced). Validated on the ring with kept sessions: serve_test --ced 17 of 17, three requests in flight each give their tokens alone, kept sessions with requests in flight come back and take turns; behind a 1k request that decodes, a 16k request's first token comes 5.2 s sooner than queued (1.9 s), at 5 % on the last request's end. decode speed is unchanged from c2dd1421 (a first job measured k = 1 0.12 ms/token slower, a second 0.056 ms faster: differences of ~0.1 ms need two jobs to count); B2 merged (d23a332, `--slots N`, default 1: production unchanged): every request in a batch equals its run alone, tokens and log-probabilities, with kept sessions and the disk tier; it buys fairness and first-token time, not throughput (four requests together 91.9 tok/s against 93.4 one at a time with drafts), so B3 (drafts across sessions) was estimated from measurements ([B3 from measurements](docs/batching.md), edeefb2): +25–32 % / +22–32 % / +18–31 % decode for 2 / 3 / 4 overlapping agentic sessions over the better of B1 and B2 (+8–14 % end to end, prefill being half the engine time), nothing for a lone session; parked, as the owner runs several agents only occasionally (2026-10-12), with production on `--slots 1`. The cheap first step if that changes: B3 for 2 sessions within 8 rows (+21–28 %). The B2 soak (4 and 6 clients, 499 requests): none failed, memory flat, regenerates identical. B4's GEMV/MoE half merged (defc73d: every test_kernels filter passes, the same SASS and speed up to 8 rows, the engine's hashes unchanged; at 9/12/16 rows the tuned MoE pair takes 1.5/1.9/2.8 % off the step); its attention half merged too (2a408dd: 3013 slot and 1359 fused checks identical, every engine hash unchanged, k = 1 within the run-to-run spread (+0.06 ms in one job, not attributable under nsys) and the same with drafts; dense index scans of 4 sessions at 128k–250k take 1.24× one session's time for 4× the keys). The engine still runs at most 8 rows: 16 are wired only if prompt-lookup drafts need them |
| The attention side per step, re-measured from 1k to 250k with chainprof's side-stream rule | done → [docs/engine.md](docs/engine.md#long-contexts); 0.68–1.06 ms at one row (3.6–5.8 % of the step), 1.01–1.37 ms at six (2.8–3.7 %); only the dense index scan grows with context. Reachable without changing bits: ~1–1.5 % at one row in three pieces, none ≥ 2 %, so not pursued. Removing every mHC mix gains 0.9 % at most, so the fold stays dropped. Decode 17.9 ms/token at 16k, 18.3 at 250k (k = 1); 97–143 tok/s with drafts |
| Draft trees: several candidates per depth, verified in one step | studied, not built → [docs/engine.md](docs/engine.md#draft-trees-a-feasibility-study-2026-10-12): at the measured cost of a sibling row (2.4–3.5 ms: 1.1 for the row, the rest the experts it adds) the best tree beats the best chain gate by +0.4 % (prose), +1.0 % (code), +1.2 % (agentic) and 0 (JSON); 10 % would need ~0.5 ms a sibling row, and even free siblings cap at +22 % (prose) and +19 % (code). The probe stays on the branch kernels-tree |
| Agentic edits through pi's own tools (`bench.py --modes edit`: 12 pi-style read/edit/write tasks on scratch copies of repo files, pi 0.84.3's prompt and tools, [bench/THIRD_PARTY.md](bench/THIRD_PARTY.md)) | done → [docs/engine.md](docs/engine.md#agentic-edits-through-pis-tools-2026-10-12): greedy, thinking off, all 12 tasks' checks pass; edit calls decode at **141.5 tok/s** (5.52 of a 5-draft step's 6 tokens, 95 % of drafts accepted), writes 138.2, reads 125.7, prose 104.6; first token 0.11–1.46 s (the last tool result's prefill). Thinking max: 122.6 tok/s overall, 4/4 checks. At dsfront's sampling defaults (T 1.0, top-p 0.95: what pi gets, as it sends no temperature; 3 repeats): edits **139.6 tok/s** (95 % accepted, unchanged), writes 136.1, reads 120.5, prose 96.7 (−8 %), 36/36 checks; thinking max 114.2 overall. Traces in rank 0's /data/dsgb10/bench/edit/r1 (greedy) and r2 (sampled) |
| Exact-output decode studies after the trees | **Prompt-lookup drafts: studied, dropped** → [docs/engine.md](docs/engine.md#prompt-lookup-drafts-a-feasibility-study-2026-10-12): on the pi-style edit traces 55 % of the output is a right n-gram proposal and 30 % lies in right runs of ≥ 16 tokens, but the best realistic gate gains +4.3 % at 8 rows and +5.0 % at 16 (the bound +8.9 / +17.8 %), since the chain already verifies all 5 drafts on 72 % of steps and copy ends are unpredictable; +2 % on the transcript, ±0.7 % on the bench types. With it goes any need for 16 rows of one session. **Lossless weight compression: studied** → [docs/engine.md](docs/engine.md#lossless-weight-compression-a-feasibility-study-2026-10-12): ≤ 12-bit Huffman would save −2.30 ms at k = 1 (12.8 %) if decoded for free, but a fused FP8 GEMV decoding it inline, exact on every site, stays slower than today's kernels (4.0–4.6 G symbols/s an SM against ~5.8 needed; v2 +0.88 ms a step), and GB10 has no hardware decompression. Fixed-length codes: **merged** (2812072, 2026-10-12 evening), the experts' ue8m0 scales at 1 bit over {120, 121} (kernels line) and the head's bf16 exponent as 4-bit offsets per 8×32 tile (attention line). Exact: tokens equal in 328 of 328 A/B requests, every reference hash. Against master in two alternating jobs at production's arguments: with drafts −3.24 % over the bar's cases (the edit replay −3.83 %, bench decode −2.30 to −4.13 %), k = 1 −1.97 %, no case slower; the codes take 0.58–0.64 GiB a rank; FP8 at 3-bit exponents skipped (≤ 2 % at 6 rows for every dense site). **The draft gate under pi's sampling (T 1.0, top-p 0.95): studied, not built** → [docs/engine.md](docs/engine.md#the-draft-gate-under-sampling-2026-10-12): replayed offline along the sampled edit bench, the best gate gains +1.1 % held out (the bound with the sampler's own probabilities +3.5 %, a perfect gate +6.4 %), since 87 % of drafts are ≥ 0.98 confident and 99.9 % of those are accepted; under the 2 % bar. **The draft pass's approximations: studied, built, not adopted** → [docs/engine.md](docs/engine.md#the-draft-passs-approximations-a-study-2026-10-12): drafts are verified, so the drafter may be approximated with the outputs unchanged; the FP4 head, the FP8 Markov head, MXFP4 blocks and a Latin vocabulary subset used only while the last 128 tokens are in it (every static subset loses 36–52 % on Chinese, Japanese and Russian) project +2.36 % on the sampled edit bench with no type slower. Built in four measured stages, the bundle gained +1.50 % end to end on the sampled edit bench in its A/B, under the 2 % bar, and bench-code lost 2.16 % in one of its two jobs, so it is not merged but for stage 1's two flags, off by default ([Bundle B's A/B](docs/engine.md#bundle-bs-ab-under-the-bar-2026-10-13)) |
| A small agentic turn's time to first token, by part, against its floors; then a split mHC boundary and small-M tiles for the dense prefill GEMMs | studied → [docs/engine.md](docs/engine.md#a-small-turns-time-to-first-token-2026-10-12): no lever alone reaches 2 % end to end. Both prototypes missed their targets (the boundary 223 → 88 µs at 32 rows against 60; the tiles' bundle 1.11–1.61 %, under the bar) and are merged, off by default (c7c1192, 4cb4209) |
| B′: a paced L2 prefetch of the next weights while the decode step's collectives and attention leave DRAM idle (one warp on a side stream) | **adopted** (ae57e04, 2026-10-12; `--no-l2pf` turns it off) → [docs/engine.md](docs/engine.md#a-paced-l2-prefetch-in-the-steps-windows-2026-10-12): it writes nothing the model reads, so the bits stay; −3.95 % end to end over the bar's 36 cases, every case faster, and 3.0 % on the sampled edit bench with drafts. A time cap on each launch and W3's budget from 0 to 300 % were measured and dropped (c4ac842) |
| The pipelined reduce-scatters' send-buffer race, seen once as a wrong first-token logprob in a cold prefill (release candidate 4's first validation run) | **fixed** → [docs/ring.md](docs/ring.md#the-pipelined-reduce-scatters-send-buffer-race-2026-10-13): a write-after-read race in the three pipelined reduce-scatters, at 4 or more pieces with a short last one (prefill chunks of 2,433 rows or more); forced, 42 of 42 wrong unfixed and 0 of 176 fixed. The fix (1128401) is in release candidate 5; `ring_test race` forces the window as a regression test, an audit found no other kernel with the hazard (0b0604f), and release candidate 5 gave 0 faults in 1,370 repeats |

dsgb10 serves dealignai's variant on rank 0:8888 (since 2026-10-12 06:39; on release candidate 5 since
2026-10-13 04:13); how it runs there: [docs/operations.md](docs/operations.md).

## Layout

```
DESIGN.md              the plan: architecture decisions, budgets, milestones
docs/                  what was learned and measured, with the evidence: engine.md (the engine and its
                       measurements), experiments.md (the index of every experiment), findings.md (the
                       research), ring.md, layout.md, protocol.md, ced.md, engram.md, batching.md
bench/bench.py         OpenAI-API benchmark: decode by prompt type, cold prefill, agentic multi-turn
bench/editbench.py     the agentic-edit bench (pi's prompt and tools); sparkdash/, piturn/: their drivers
bench/bench_record.py  bench.py unchanged, with every request recorded: body, SSE lines and their arrival, usage, text
bench/bench_record_report.py   statistics of recorded sessions, one server or two side by side
bench/steps.py         tokens and time per decode step for recorded requests, from each server's own log
bench/golden_sglang.*  greedy completions + top-5 logprobs recorded from the SGLang baseline (--url/--model/--out for another)
micro/                 microbenchmarks: bw.cu (memory), mma.cu (block-scaled MMA), rdma_lat.cu (ring RDMA)
ref/torchref/          pure-torch kernel.py + layer-streaming driver around the publisher's model.py
ref/inference, ref/encoding   copies of the publisher's reference code (MIT), the spec
ref/inventory.txt      every checkpoint tensor, aggregated by layer (tools/st_inventory.py)
golden/run_ref.sh      produce golden activations/logits in the SGLang image (torch 2.13)
tools/                 small helpers
fleet/homelab.conf     the fleet file: the four boxes, how the ring links them, where each keeps its files
engine/                the engine, C++20/CUDA for sm_121a (CMake; one subdirectory per part, picked up when present)
  common/              shared basics (error checks)
  weights/             per-rank weight files: converter, loader, verifier (docs/layout.md)
  comm/                ring collectives over ibverbs with CPU proxies (docs/ring.md)
  kernels/             decode/prefill GEMV and GEMM kernels: FP8, MXFP4, bf16
  attn/                attention, indexer, compressor, mHC kernels
  dspark/              the DSpark draft module (engine/dspark/README.md)
  engram/              Engram hashing and the NVMe row store
  model/, server/      the layer runner and the rank-0 socket server
frontend/              Rust: OpenAI API, chat encoding, BPE tokeniser (docs/protocol.md)
```

Build the engine with `cmake -S engine -B engine/build && cmake --build engine/build -j4` (more jobs need
more memory, and an OOM can hang a GB10), and the front-end, `dsfront`, with `cargo build --release` in
`frontend/`. `cargo test --release` there runs its unit tests, and `frontend/tools/smoke_test.py` (its
header gives the commands) checks it end to end against `dsfront mock-engine`, with no GPU.

**What the build and the ring need**, as on rank 0 (Ubuntu 24.04.5, kernel 7.0.0-1019-nvidia):

- CUDA 13.0 in `/usr/local/cuda-13.0` (engine/CMakeLists.txt uses its `nvcc` unless `-DCMAKE_CUDA_COMPILER`
  names another) and driver 580.178.04. The code targets `sm_121a`, GB10 alone.
- CMake 3.28 or later, g++ 13.3 (C++20) and `libibverbs-dev` (rdma-core 50.0).
- Rust for `frontend/` (cargo 1.97.1 here; edition 2021).
- Python 3: the standard library for the tests and tools under `engine/server/` and for the Engram hash
  map (`engine/engram/py/make_hashmap_stdlib.py`). The golden data, `make_hashmap.py` (the hash map and
  the hashing test cases, from the reference's own code) and `check_dsw.py` run in the SGLang image that
  the MiaAI kit builds from its `Dockerfile.canary-roce`, `dsv41-4x-spark:canary-roce`
  ([findings](docs/findings.md#environment-gotchas)).
- On kernel 7.0.0-1019-nvidia, every box booted with `kho=off`: without it `ibv_reg_mr` fails
  ([findings](docs/findings.md#hardware-measured-here)). The ring is cabled as in
  [docs/ring.md](docs/ring.md#the-ring) and addressed as its fleet file says (below), and rank 0 reaches
  ranks 1–3 by SSH without a password.

**The weights.** Everything here was built and validated on dealignai's
`DeepSeek-V4.1-Flash-UNCENSORED-FP8` at `d61c59ea` ([DESIGN](DESIGN.md#target)). The original,
`deepseek-ai/DeepSeek-V4.1-Flash` at `fb2764a5`, has the same 96,085 tensors and gives the same conversion
plan for every rank (`dsw_convert --plan`, 2026-10-13). Each box needs:

- its rank's weight file, 72.4 GiB, made by `dsw_convert`, which needs the GPU
  ([docs/layout.md](docs/layout.md#tools));
- its rank's two Engram shards, `engram-l1-r<r>of4.bin` and `engram-l14-r<r>of4.bin`, 23.6 GiB each, made
  from the checkpoint by `engram_pack` (engine/engram/tools/pack.cpp): byte for byte the shards of the
  MiaAI kit's [`scripts/pack_engram.py`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/scripts/pack_engram.py)
  with `--tp 4` (its default `--tp` is 3);
- the Engram hash map, `engram_hashmap.bin`, 518,048 bytes, made from the checkpoint's tokenizer and
  config by `engine/engram/py/make_hashmap_stdlib.py` ([docs/engram.md](docs/engram.md#hashing)).

The engine looks for them where the fleet file says (`RANK<r>_WEIGHTS`, `RANK<r>_HASHMAP`,
`RANK<r>_ENGRAM`): here rank 0's are `/data/dsgb10/weights/rank0.dsw`,
`/data/dsgb10/engram/engram_hashmap.bin` and `~/dsv41-engram-tp4/`, and ranks 1–3's
`~/dsgb10/weights/rank<r>.dsw`, `~/dsgb10/engram/engram_hashmap.bin` and `~/dsv41-4x-ring/engram/`.
`--weights`, `--hashmap`, `--shard1` and `--shard14` name other files. `run4.sh` and `service.sh` give
all four ranks the same arguments, so in such a path `{rank}` stands for the rank and a leading `~/` for
each box's home directory (`--weights '~/models/rank{rank}.dsw'`, quoted). The loader refuses a weight
file cut for another rank, naming the file and both ranks.

**The fleet is a file**, [`fleet/homelab.conf`](fleet/homelab.conf): the four boxes' hostnames and SSH
addresses, the ring's RDMA devices, GID index and link addresses, and each rank's files and
directories. The engine reads it (`--fleet FILE`; by default `fleet.conf` beside the binary, where
`run4.sh` and `service.sh` install it), and so do engine/server/run4.sh, ring.sh and service.sh, and
engine/comm/run4.sh (`FLEET=FILE`, by default this one). Other boxes need only a copy with their values
([How to use](#how-to-use), step 3); [docs/review/fleet.md](docs/review/fleet.md) lists what is still
this project's own in the code and the scripts. An engine binary needs the scripts of its own commit,
which install `fleet.conf` beside it.

**The front-end** reads the tokenizer from `/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json`
unless given `--tokenizer PATH` (through service.sh: `DSGB10_FRONT_ARGS="--tokenizer PATH"`); the
original's tokenizer.json is the same file. It serves `GET /v1/models`, `POST /v1/chat/completions`
(OpenAI's API, with the key as a bearer token) and `GET /health`, and names the model
`deepseek-v4.1-flash-uncensored` (`--model-name`). A greedy answer depends on the session state its
request resumes from: sent twice, the same request can differ after its first tokens, because a
regenerate reruns its last positions through the decode step ([engine § Serving](docs/engine.md#serving)).

Large data never goes in the repo. It lives on each box: on rank 0 under `/data/dsgb10/`,
on the other boxes under `~/dsgb10/` (`bin/`, `weights/`, `logs/`),
and scratch outputs under `~/dsgb10-data/`.

## Reproducing what exists

```bash
# bench.py against any OpenAI-compatible endpoint (--url, default http://127.0.0.1:8888/v1): dsfront
# serves the model as deepseek-v4.1-flash-uncensored unless its --model-name says otherwise (the SGLang
# kit used the same name); the API key is read from a file, never from the command line
python3 bench/bench.py --key-file <file with the API key> \
    --model deepseek-v4.1-flash-uncensored --modes decode,prefill,agentic

# the same, unchanged, with every request recorded (body, SSE lines and their arrival, usage, text);
# bench/bench_record_report.py reads the records
python3 bench/bench_record.py --record ~/bench.jsonl --key-file <file with the API key> \
    --model deepseek-v4.1-flash-uncensored --modes decode,prefill,agentic

# the agentic-edit bench: pi-style read/edit/write tasks on scratch copies of repo files (bench/editbench.py)
python3 bench/bench.py --modes edit --key-file <file with the API key> \
    --model deepseek-v4.1-flash-uncensored --edit-out ~/edit

# sparkDash 1.8.8's own benchmark on an engine of its own, as one ring job (bench/sparkdash/part.sh's header)
BIN=engine/build/server/dsgb10_rank FE=frontend/target/release/dsfront SD_ROOT=<sparkDash at 754f40a, npm ci --omit=dev> \
    engine/server/ring.sh bash bench/sparkdash/part.sh decode ~/sparkdash

# microbenchmarks (rank 0); the arch flag matters, see docs/findings.md
cd micro && /usr/local/cuda-13.0/bin/nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o mma mma.cu && ./mma
/usr/local/cuda-13.0/bin/nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o rdma_lat rdma_lat.cu -libverbs
scp rdma_lat 192.168.2.2:/tmp/ && ssh 192.168.2.2 /tmp/rdma_lat server 192.168.2.2 rocep1s0f0 3 64 &
./rdma_lat client 192.168.2.2 rocep1s0f1 3 64

# golden data (4 min for prose, 17 for the long transcript on a GB10; run detached, see
# docs/findings.md#environment-gotchas; every sequence's command is in golden/README.md)
setsid nohup golden/run_ref.sh run prose golden/prose.tokens.json --positions all --record auto --full-logits all \
    > ~/dsgb10-data/golden/prose.log 2>&1 < /dev/null &
```

