# Ring collectives (engine/comm)

The collectives of DESIGN.md decision 2: our own all-reduce, all-gather and reduce-scatter over
ibverbs for the four-box ring, driven by GPU kernels that only touch memory, so that a whole decode
step, collectives included, is one CUDA graph. Everything here was measured on the ring itself
(ranks 0 rank 0, 1 rank 1, 2 rank 2, 3 rank 3).

**Status.** Built and tested across the four boxes: sums exact against a host reference and
bit-identical on every rank; 3 million random-size collectives with every payload word checked,
no mismatches; a 10 KB bf16 all-reduce takes **12.2 µs** back to back in a graph (christopherowen's
host relay, "ring4": 16.9 µs at 10 KiB on 2026-10-03, 16.2 µs with its CuTe kernel on 2026-10-05);
large all-reduces run at 9.3–9.6 GB/s.

## The ring

The four boxes are two ASUS Ascent GX10 (ranks 0 and 1) and two NVIDIA DGX Spark (ranks 2 and 3), all GB10
systems. Each box has one ConnectX-7 with two QSFP ports. Each port appears as two RDMA devices, one per PCI
domain (`rocep1s0f*` on domain 0000, `roceP2p1s0f*` on domain 0002), each a PCIe Gen5 x4 function
with its own netdev and /24 subnet. Port f0 faces the next rank, f1 the previous one. RoCE v2 is
GID index 3 on every device; netdev MTU 9000, RDMA active MTU 4096. The fleet file,
[`fleet/homelab.conf`](../fleet/homelab.conf), holds this table (`engine/comm/fleet.cpp` reads it for
the engine, `engine/server/fleet.sh` for the scripts); `micro/ring_links.sh` prints it from sysfs. On
other boxes, a copy with their devices, GID index, addresses and hostnames, passed as `FLEET=FILE`, is
all it takes: the scripts install it beside the engine as `fleet.conf` (`--fleet FILE` reads another;
run4.sh and service.sh pass `--rank`, so the hostnames matter only without it). Before the bootstrap,
each rank checks that the GID at `GID_INDEX` of each of its four devices is the IPv4-mapped address
the file gives that link, and stops otherwise, naming the device, the GID, the key and the file. The
scripts reach ranks 1–3 by SSH at the file's `RANK<r>_SSH` (10.0.0.4, .3 and .2 here). Only
`kSlowLink` stays compiled in (`engine/comm/ring_config.h`: L0, the 100G link): it routes the
reduce-scatter's relays, and the sums are in rank order whatever the route ([Protocol](#protocol)).

| Link | Speed | Domain 0: rocep1s0f0 → rocep1s0f1 | Domain 2: roceP2p1s0f0 → roceP2p1s0f1 |
|---|---|---|---|
| L0 rank 0 → rank 1 | **100G** | 192.168.6.1 → 192.168.6.2 (enp1s0f0np0 → enp1s0f1np1) | 192.168.7.1 → 192.168.7.2 |
| L1 rank 1 → rank 2 | 200G | 192.168.0.1 → 192.168.0.2 | 192.168.1.1 → 192.168.1.2 |
| L2 rank 2 → rank 3 | 200G | 192.168.4.1 → 192.168.4.2 | 192.168.5.1 → 192.168.5.2 |
| L3 rank 3 → rank 0 | 200G | 192.168.2.2 → 192.168.2.1 | 192.168.3.2 → 192.168.3.1 |

On every link the first box uses its f0 (next-facing) devices and the second box its f1 devices.

### Measured per box and per link

Streaming read bandwidth (`micro/bw.cu`), best GB/s: rank 0 236.2, rank 1 238.1, rank 2 235.9,
rank 3 236.8. There is no slow box.

GPU access to the host-memory kinds a collective buffer can use (`micro/hostmem.cu`):

| Memory | Read GB/s | Dependent load | RDMA-registrable |
|---|---|---|---|
| cudaMalloc (reference) | 236.4 | 0.33 µs | – |
| cudaHostAlloc | 178.4 | 0.41 µs | yes |
| cuMemCreate, `CU_MEM_LOCATION_TYPE_HOST_NUMA` | **236.0** | 0.42 µs | yes |

One-way latency per link on domain 0 (`micro/rdma_lat.cu`, run by `micro/ring_lat.sh`): CPU
ping-pong / GPU ping-pong through a CPU proxy, in µs:

| Payload | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| 64 B | 1.53 / 2.65 | 1.54 / 2.76 | 1.54 / 2.81 | 1.53 / 2.77 |
| 10 KB | 4.25 / 5.44 | 3.72 / 5.05 | 3.85 / 5.02 | 3.86 / 5.13 |
| 60 KB | 8.55 / 10.51 | 8.32 / 10.29 | 8.32 / 10.26 | 8.30 / 10.59 |

RDMA write bandwidth (`ib_write_bw`, run by `micro/ring_bw.sh`), the same at 1 MB and 8 MB. On L1,
one PCI domain gives 109 Gb/s and both at once 196 Gb/s (98 + 98): **one PCIe x4 function cannot
fill a 200G port**. On L0 the 100G cable is the limit: 98 Gb/s with one domain or both (49 + 49).
L0 is the ring's bandwidth bottleneck: 12.25 GB/s per direction.

## Protocol

**Region.** Each rank has one host region from `cuMemCreate` on the host NUMA node (2 MiB
granules), mapped for the GPU and the CPU at the same address and registered with all four RDMA
devices. Collective s uses buffer set p = s % 3 (`kSlots`); each set holds `send[p]`,
`from_prev[p]`, `from_next[p]` and `opp[p]` (cap bytes each) and eight 64-byte flags. The layout is identical on every rank, so a remote
address is the peer's base plus the same offset. Kernels find rank k's contribution in `send`
(their own), `from_prev`, `from_next` or `opp`.

**Post.** A kernel writes its contribution to `send[p]`, then the doorbell flag {seq, nbytes, op}
with a release store at system scope. The proxy sees it with an acquire load (LDAR) and issues
`dmb osh`, so the NIC's DMA read cannot see older data. Then it posts RDMA WRITEs to both
neighbours. Each neighbour gets part A (what it relays on to the opposite rank) and part B. Each
part is followed by its own 16-byte inline flag WRITE on the same RC queue pair, which the NIC
delivers after the data. From 4 KiB up, A goes on the domain-0 QP and B on the domain-1 QP at the
same time.

| op | to next: A / B | to prev: A / B | per link and direction |
|---|---|---|---|
| all-exchange (all-reduce, all-gather), n bytes | [h, n) / [0, h) | [0, h) / [h, n) | 1.5 n |
| reduce-scatter, quarter k (q = n/4) for rank k | half 1 of quarter r+2 / quarter r+1 | half 0 of quarter r+2 / quarter r−1 | 0.5 n |

(h is n/2 rounded down to 64 bytes.) Everything lands at the same offset as in the sender's buffer.

**Relay.** When part A from next has arrived (its flag), the proxy forwards it to prev's `opp`
buffer; part A from prev goes on to next. So the opposite rank's data crosses two links without a
GPU round trip, half each way round the ring. Consumers wait for six flags: A and B from each
neighbour and the two relayed halves (`wait_all`).

**Reduction.** Consumers sum the four contributions in fp32 in rank order, ((c0 + c1) + c2) + c3,
and round once (bf16: round to nearest even). Every rank therefore holds bit-identical results, and
a bulk all-reduce gives the same bits as an all-exchange one.

**Flow control** needs no credits and no resets. Every rank runs the same collectives in the same
order, and posts s only once it has (1) consumed s − 2 and (2) seen every rank's s − 1 arrive.
Plain collectives consume s − 1 before they post s, which is stricter; the bulk all-reduce uses
the slack to pipeline.

- A rank that writes s into a peer's buffer set (last used by s − 3) has seen the peer's s − 1,
  which the peer posted only after consuming s − 3.
- The opposite rank's `opp` buffer follows by the same argument, through the relaying neighbour.
- The own send buffer is free again once the neighbours have consumed its last use, because our NIC
  has then finished reading it.
- The relay sources are the exception, because the NIC reads them asynchronously. A relay of s
  towards prev is posted only after the relay of s − 1 towards next has completed (CQE), and vice
  versa. Before next can write s + 3 into our `from_next` it must have consumed s + 1, which needs
  our relay of s + 1 towards next, which we post only after our relay of s towards prev (the one
  that reads `from_next`) has completed.
- The own send buffer has readers inside the grid too. The pipelined reduce-scatters (the bulk
  reduce-scatter, the all-to-all by rows, the MoE's combine) have each block read its part of the
  own quarter as it reduces s, and write s + 2's send buffer (s − 1's) while s + 1 travels. Quarter k
  sits at k × the piece's length, so with a shorter last piece one block's copy lands on another
  block's part, and a block still reducing s − 1 would read the new piece. The copy therefore waits
  until every block has posted s + 1 (`wait_posted`), which each does only after reducing s − 1.
  The other kernels write only what the same block reads, or after a `wait_all` that every block
  passes only after its reads.

**Sequence numbers live in device memory** (`ch.seq`). Kernels read `next_seq(ch)`, which is the
last finished collective + 1, and the last block of `finish_grid` advances it. Graph replays
therefore advance by themselves; nothing is reset between replays or steps.

**Grids.**

- *Publishing.* A single-block producer publishes with one release store. With more blocks, each
  block fences at GPU scope and counts itself in; the last one fences at system scope and publishes.
- *Waiting.* In `wait_all` only block 0 polls host memory (the six arrival flags and the own
  doorbell), then passes the news on through device memory. When every block polled host flags
  and fenced at system scope, each block of grid size added ~2.4 µs.
- *Grid barrier.* Checking the own doorbell makes `wait_all` a grid barrier, which kernels that run
  several collectives in a row (the bulk all-reduce) need.
- *Residency.* Blocks that wait must all be resident, so a waiting kernel has at most 48 blocks.

**Bulk all-reduce.** Payloads are cut into chunks of up to cap bytes. Each chunk is a
reduce-scatter (rank r sums its quarter straight into its send buffer) followed by an all-gather of
the reduced quarters, all in one kernel. That moves 0.875 n per link and direction instead of 1.5 n.
The kernel pipelines one deep: as soon as chunk i's all-gather has arrived it copies chunk i + 1 in
and posts its reduce-scatter, then copies chunk i out while that travels.

## API

**Host** (`ring.h`). `comm::Ring ring(opts)` does the setup:

- bootstraps over TCP on the ring-link addresses (port 18620, bound to the specific address);
- connects two RC QPs to each neighbour, one per PCI domain;
- allocates and registers the region;
- starts the proxy thread, pinned to core 5. X925 cores 5–8 are for comm proxies (9 is the Engram
  worker's, 15 the engine's main thread's). The proxy spins: a sleeping core takes up to 433 µs to
  wake. After 1 s without traffic it polls every 50 µs instead, so an idle engine does not burn a
  core. That costs nothing measurable: a decode-step graph (80 × 10 KB) right after 2 s idle took
  1.05 ms (median of 5, worst 1.36), the same as one right after another; with constant spinning
  1.10 ms (`ring_test wake`).

`ring.dev()` is the `DevChannel` that kernels take by value. `Options`:

| Field | Default | Meaning |
|---|---|---|
| `rank` | from the hostname | this box's rank |
| `cap` | 1 MiB | bytes per contribution; the region is 12 × cap |
| `proxy_cpu` | 5 | core the proxy is pinned to (use 5–8) |
| `idle_backoff_ms`, `idle_poll_us` | 1000, 50 | back off to polling after this much idle time (0: always spin) |
| `bootstrap_timeout_s` | 600 | give up if a neighbour has not connected by then |
| `split_domains_from` | 4096 | send the two halves on both PCI domains from this size up |
| `relay_domain` | 1 | PCI domain for all-exchange relays |
| `trace` | 0 | record the first N collectives' timelines |
| loopback modes | off | single-box diagnostics |

`barrier()` and `allgather_host()` run over the bootstrap sockets; they are for setup and tests
only.

**Device**, for fusion (`ring_dev.cuh`):

```cpp
uint64_t s = comm::next_seq(ch);              // every block, kernel start
... write this rank's part to comm::send_buf(ch, s) ...      // e.g. a GEMV epilogue
comm::post_grid(ch, s, nbytes);               // every block; the last rings the proxy
// --- same or a later kernel (which reads the same next_seq: nothing called finish_grid) ---
comm::wait_all(ch, s);                        // every block
... comm::sum8_bf16(ch, s, off, f) / comm::src(ch, s, k) ...  // e.g. the next kernel's prologue
comm::finish_grid(ch, s);                     // every block; the last advances ch.seq
```

**Standalone kernels** (`collectives.h`), all graph-capturable:

- `allreduce_bf16/f32`;
- `allgather(ch, in, out, nbytes, stream)`: out = [in_0 | in_1 | in_2 | in_3], any multiple of
  16 bytes; above cap it runs in pipelined chunks. This is the call Engram's hand-off makes with its
  packed row segments (`engine/engram/engram.h`, docs/engram.md);
- `reduce_scatter_bf16/f32`;
- `allreduce_bulk_bf16/f32`: any size that is a multiple of 64 bytes; a `blocks` argument lets it
  share the GPU with other work;
- `argmax_rows`: per row, the best (value, index) of the four ranks, ties to the lower index. This
  is the all-gather for distributed Gumbel-max sampling.

**Build and test** on rank 0. `engine/comm/run4.sh` copies the binary, and the fleet file (`FLEET`) as
`fleet.conf`, to the fleet's `BIN_DIR` (`~/dsgb10/bin`) on every box,
starts ranks 1–3 detached and rank 0 in the foreground, then prints the worker log tails. It takes no
ring lock of its own, so it runs under `engine/server/ring.sh` like any job on the ring ([Borrowing the
ring](engine.md#borrowing-the-ring)), with `PREEMPT=0`, since a preemption stops only `dsgb10_rank`
processes on the workers:

```
cmake -S engine -B engine/build-comm -DCMAKE_BUILD_TYPE=Release && cmake --build engine/build-comm -j 8
OWNER=me PREEMPT=0 engine/server/ring.sh engine/comm/run4.sh engine/build-comm/comm/ring_test correct race integrity latency graph
OWNER=me PREEMPT=0 engine/server/ring.sh engine/comm/run4.sh engine/build-comm/comm/ring_test bulk --cap 4194304
OWNER=me PREEMPT=0 engine/server/ring.sh engine/comm/run4.sh engine/build-comm/comm/ring_test graph --trace 4000   # per-flag arrival times
OWNER=me PREEMPT=0 engine/server/ring.sh engine/comm/run4.sh engine/build-comm/comm/ring_test race --cap 4194304   # the engine's shapes, ~6 s
./engine/build-comm/comm/ring_test --mode proxy loop                       # one box, no network
```

This `run4.sh` takes no ring lock. While the service runs, run it inside `engine/server/ring.sh`
(`OWNER=me engine/server/ring.sh engine/comm/run4.sh …`), which waits for the ring as the engine's
`run4.sh` does.

`race` is the regression test of [the pipelined reduce-scatters'
race](#the-pipelined-reduce-scatters-send-buffer-race-2026-10-13), in ring_test's defaults. It runs
with every ring change. ring_test links its own copy of the comm library, built with
`DSGB10_COMM_TEST_HOOKS`. There, block 1 of the pipelined reduce-scatter, all-to-all and
multi-buffer all-gather can be stalled (`comm::test_stall`) after posting piece i + 1 and before
reading piece i, at the last i whose send buffer the short last piece reuses. The engine's
`dsgb10_comm` has no hook: its SASS is the same as without the test, but for the anonymous
namespace's hash in 13 names. Each case runs without and with a 2 ms stall and must be exact on
every rank. The shapes follow `--cap`; at the engine's 4 MiB they are the MoE's rows reduce-scatter
at 776 × 1280 fp32 (4 pieces, as a 3,100-row chunk) and 903 rows (5 pieces), attention's all-to-all
at 1,602 rows of 2,048 + 64 B (4 pieces), and the all-gather at 2,859 rows of 5,120 + 160 B (4
chunks).

## Results

**Correctness.** Tested from 16 B to 1 MiB: all-reduce (bf16, fp32), all-gather, reduce-scatter,
the split producer/consumer pattern and argmax. The bulk all-reduce was tested up to 6 MiB in
1 MiB chunks, including a partial chunk. All results are exact against a host reference with the
same summation order and identical on all ranks.

**Integrity: 0 mismatches on all four ranks**, twice: 3,000,000 collectives in graphs of 250,
sizes log-uniform from 16 B to 1 MiB (mean 101 KB), a third of them reduce-scatters (219 s with two
buffer sets on an idle ring; 166 s with three, while rank 2's GPU ran another job). And 100,000
bulk all-reduces of 64 B to 8 MiB (mean 390 KB, 1 MiB chunks) checked on the GPU, every element a
small integer that depends on the round, rank and index: 0 wrong sums. Every rank checks every word it receives against a pattern of (seq, sender,
word), so stale, torn or misplaced data would show. So would the NIC reading the send buffer before
the GPU's stores landed.

**Latency**, µs per collective, back to back in a graph (cap 2 MiB; redscat16 is a bf16
reduce-scatter of a vector of that size):

| bytes | allreduce16 | allreduce32 | allgather | redscat16 |
|---:|---:|---:|---:|---:|
| 64 | 7.69 | 7.56 | 8.77 | 7.45 |
| 1 K | 9.15 | 9.17 | 10.56 | 7.95 |
| 4 K | 10.24 | 10.27 | 11.64 | 9.30 |
| 10 K | 12.08 | 12.06 | 13.24 | 9.88 |
| 20 K | 15.48 | 15.47 | 16.75 | 11.67 |
| 60 K | 24.46 | 24.43 | 25.57 | 14.60 |
| 128 K | 38.22 | 38.19 | 39.20 | 19.01 |
| 256 K | 63.68 | 63.89 | 64.22 | 25.91 |
| 512 K | 110.11 | 110.23 | 110.16 | 43.03 |
| 1 M | 200.58 | 200.73 | 202.40 | 74.02 |
| 2 M | 383.61 | 383.66 | 402.88 | 129.54 |

**Decode step**: 80 all-reduces of 5120 elements, back to back in one graph.

| Variant | bf16, µs each | fp32, µs each |
|---|---:|---:|
| **Default** | **12.28 (0.98 ms per step)** | 15.61 (1.25 ms per step) |
| Proxy on core 19 instead of 5 | 12.17 | 15.54 |
| Published by one kernel, reduced by the next (the fusion pattern) | 13.14 per pair | – |
| Relays on domain 0 instead of 1 (no difference) | 12.07 | 15.38 |
| Without the domain split | 12.77 | 16.57 |
| First version: four flags, one domain, 2–3 chunks per thread, a system fence per block | 14.2 | 17.1 |

**Where the 12 µs go** (`--trace`: medians in µs after the rank's own doorbell, on each box's
clock):

| Event | µs |
|---|---|
| Neighbours' data arrives | 3.6–5.3 |
| Relays posted | 3.4–5.2 |
| Opposite rank's halves arrive | 8.0–8.8 |
| All six flags in | 8.8–9.2 |
| Last arrival → next doorbell | 3.3–3.7 |

The last interval is all GPU: notice the flag, reduce from host memory, finish, launch the next
kernel, copy in, publish. The sampling all-gather of (value, index) rows takes 8.1 µs. Two hops of ~4.4 µs each are the floor of this algorithm. The GPU side is
where fusion helps.

**Single box, no network.** GPU → proxy → GPU takes 3.1 µs at 64 B and 3.9 µs at 10 KB. Other
costs:

- an empty graph node: 0.5 µs;
- ibv_post_send: 0.11–0.14 µs per call;
- per thread on the GPU: ld.acquire.sys of host memory 0.43 µs, st.release.sys 0.32 µs,
  fence.acq_rel.sys 0.28 µs, fence.acq_rel.gpu 0.16 µs.

**Large payloads** (bf16, cap 4 MiB, 16 blocks): the bulk all-reduce against cap-sized all-exchange
all-reduces in a row, and an all-gather of a quarter of the bytes per rank (so its output is the
same size):

| bytes | bulk µs | GB/s | chained all-exchange µs | GB/s | all-gather µs | GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 64 K | 29.9 | 2.19 | 25.8 | 2.54 | 15.5 | 4.24 |
| 256 K | 51.7 | 5.07 | 64.6 | 4.06 | 26.8 | 9.78 |
| 1 M | 137.6 | 7.62 | 201.3 | 5.21 | 65.1 | 16.12 |
| 4 M | 429.4 | 9.77 | 755.0 | 5.56 | 204.8 | 20.48 |
| 16 M | 1819 | 9.23 | 3173 | 5.29 | 840 | 19.97 |
| 32 M | 3621 | 9.27 | 6349 | 5.29 | 1595 | 21.04 |

L0 caps the bulk path at 14.0 GB/s and the all-exchange at 8.2 GB/s. A 4 MiB chunk takes ~450 µs
against ~300 µs of L0 time. Without the one-deep pipelining, 32 MiB took 3738 µs. 32 blocks
instead of 16 change nothing (memory-bound GPU copies of ~75–105 µs per chunk). Two channels (two
`Ring`s, half the payload each, `ring_test bulk2`) are worse: 4.9 GB/s. The all-gather moves
1.5 × its per-rank bytes per link and direction, so it too runs at ~65 % of L0's bound; Engram's
largest prefill segments (6.5 MB per rank) take ~1.3 ms.

For prefill: two all-reduces per layer of a 10 KB residual per token, over the 20 encoder layers,
come to ~400 KB per prompt token. At 9.4 GB/s that is ~43 µs per token, a ceiling of ~23k tokens/s
if nothing overlaps (the bar is ~4.9k).

## The pipelined reduce-scatters' send-buffer race (2026-10-13)

The determinism line's write-up. The fix (det-fix 83f7b72) is merged as 1128401 and is in release
candidate 5; the regression test and the audit are merged as 0b0604f; the diagnostics stay on det-diag
(not for master).

**The fault.** In release candidate 4's first validation run (`serve_test.py --ced` on candbin4,
master 9610544, at production's arguments but for `--disk-gib 8`), "encoder snapshot vs cold"
differed once. Request 13, the 3,100-token prompt prefilled cold, gave a first-token logprob of
−0.16151 where every other run gives −0.27298. The 8 tokens were the same; the draft statistics were
not. Six fresh-server reruns were exact ([experiments §
Numerics](experiments.md#numerics-and-correctness)).

**Its rate.** A stress client drove fresh servers at RC4's arguments, 400 s a part, alternating the
binaries over six parts. The client is the det agent's (in its scratchpad, not committed) and uses
`serve_test.py`'s engine client unchanged. Each part ran three loops:

- `--ced`'s 25-request sequence, with fresh session ids each round;
- "session to 3,000, restart at 2,048, the 3,100-token prompt cold";
- the 3,100-token prompt cold.

A fault is a request whose tokens, logprobs, first-token top-2 or draft statistics differ from the
modal result of its group (the same prompt, max_new and cached length).

| binary | requests | faults | the 3,100-token prompt cold | 95 % bound a cold prefill |
|---|---|---|---|---|
| candbin4 (RC4, B′ on) | 1,059 | 0 | 0 of 417 | < 1 in 139 |
| candbin4 `--no-l2pf` | 1,048 | 0 | 0 of 410 | < 1 in 137 |
| candbin3 (production, 198923bb) | 1,049 | 0 | 0 of 411 | < 1 in 137 |
| all three | 3,156 | 0 | 0 of 1,238 | < 1 in 413 |

A seventh part on candbin4, cold prompts of 3,100, 3,600, 4,096 and 2,400 tokens (70 each), also
had no fault. So the fault is rare. B′ made no difference at this resolution, and production's binary
behaved the same.

**The race.** A read of the cold prefill's path for the suspects (PDL edges, the comm stream's
joins, state reused across requests) found nothing. A read of the ring kernels it uses found a
write-after-read race in the three pipelined reduce-scatters:

- `k_moe_combine_rs`, the MoE's combine fused into its reduce-scatter (chunks of more than 2,520
  rows);
- `reduce_scatter_bulk_kernel`, behind `reduce_scatter_rows_f32` (the MoE's reduce-scatter up to
  2,520 rows);
- `alltoall_rows_kernel` (attention's output all-to-all).

Each kernel posts piece i of every quarter as collective s. After reducing s, it copies piece i + 2
into s + 2's send buffer, which is s − 1's (three buffer sets). `wait_all(s)` tells a block that
every block has *posted* s, not that every block has finished *reading* s − 1, because a block
posts s before it reduces s − 1, and while it reduces s − 1 it reads its share of the own quarter
in that send buffer.

Threads stride over the 16-byte units, and quarter k of a piece starts at k × the piece's length.
With full pieces (65,536 units, a multiple of the grid's threads), every unit of the buffer is
written and read by the same thread, so nothing collides. A shorter last piece shifts quarter k by
k × its length, so one block's copy lands on units that other blocks read in the previous piece. It
matters only if such a block is still reducing s − 1 after its siblings have seen s arrive from the
ring and have reduced it. That takes a stall of about one piece's transfer: a full piece moves
2 MiB per link and direction, ~0.17 ms at L0's 12.5 GB/s.

The race is exposed when a kernel has 4 or more pieces and the last piece's 16-byte count is not a
multiple of the grid's threads (Q = `q_rows(M)`, a rank's rows of an M-row chunk):

| kernel | threads | a quarter | exposed chunks |
|---|---|---|---|
| the MoE's reduce-scatter (combine or rows) | 8,192 | Q × 5,120 B | M ≥ 2,433, unless Q is a multiple of 128 (4,096 and 8,192 are safe) |
| attention's all-to-all | 2,048 | Q × 2,112 B | M ≥ 5,953, unless Q is 1,536 or 2,048 (8,192 is safe) |

The cold 3,100-token prefill (Q = 776) runs 80 exposed combines: 20 encoder layers × 4 column
groups, each with 4 pieces, the last of 827,392 B. Its all-to-all has 2 pieces. The snapshot side's
1,052-row chunk has 2 pieces, a 2,400-row chunk 3 and a 4,096-row chunk 5 equal ones. The race came
in with the bulk reduce-scatter's copies ahead (2a3aa6b, 2026-10-11) and the two kernels built on
its pipeline (39ba0fd, 77d1c82, 2026-10-12), so production's binaries have it too.

**The race forced.** A diagnostic build (det-diag 944d93e, unfixed) adds two knobs, off by default:

- `DSGB10_RSDELAY_US` / `_TAG`: block 1 of one combine waits after posting piece 1, before
  reducing piece 0.
- `DSGB10_RSCHECK`: every block compares its own contribution, as read from the send buffer, with
  the sum recomputed from the source, and counts the mismatching units in each prefill.

With a 2 ms delay at layer 0's first column group, each build ran 150 s of the four cold prompts in
turn. The fixed build is det-diagfix 904db34, the same plus the fix.

| prompt | pieces (the last, B) | unfixed | fixed |
|---|---|---|---|
| 3,100 | 4 (827,392) | 42 of 42 wrong; mismatches per prefill: rank 0 256, rank 1 2,048, rank 2 2,048, rank 3 512 | 44 of 44 exact, no mismatch |
| 3,600 | 5 (434,176) | 42 of 42 exact, no mismatch | 44 of 44 exact, no mismatch |
| 4,096 | 5 (1,048,576) | 42 of 42 exact, no mismatch | 44 of 44 exact, no mismatch |
| 2,400 | 3 (974,848) | 42 of 42 exact, no mismatch | 44 of 44 exact, no mismatch |

The wrong result was the same every time: a first-token logprob of −0.0703582763671875 against
−0.2729816436767578, with the first 7 of 32 tokens the same. The incident's value differs, as a
natural stall would hit other blocks, layers or units: only the kind of fault compares. Without the
delay, the check alone gave the right result with no mismatch. The mismatch counts are exactly the
predicted overlap:

- On rank r, block 1 reads 2,048 units of its quarter of piece 0: 8 runs of 256 units, one every
  8,192, between units r × 65,536 and (r + 1) × 65,536.
- The other blocks' copy of piece 3 writes quarter k at k × 51,712.
- On rank 0, only the 256 of those units past 51,712 are overwritten; on rank 3, only the 512 below
  4 × 51,712. Ranks 1 and 2 lose all 2,048.

3,600 stays exact because the buffer that the delayed block still reads (s0's) receives piece 3,
which is full at 5 pieces. The short piece 4 goes into s0 + 1's buffer, which only a delay at piece
1 would expose.

**Verdict.** The race gives this kind of fault on the incident's path, and the reads found no other
cause. That it caused the incident is likely but not proven: at under 1 in 413 cold prefills, no
stress run can show the natural rate fall.

**The fix** (det-fix 83f7b72) applies to all three kernels. Before the copy of piece i + 2 (for
i > 0), every block waits until the own doorbell shows s + 1 (`comm::wait_posted`: thread 0 polls
with a system-scope acquire, then `__syncthreads`). `post_grid` publishes s + 1 only after every
block has counted in, and each block counts in only after its reduce of s − 1. So every read of
s − 1's send buffer happens before any write into it. The rule is in `ring_dev.cuh` and in
[Flow control](#protocol) above.

| check (bins/fix1, master 76361eb + the fix) | result |
|---|---|
| stress, 476 requests (ced 168, snapshot 129, cold 107, cold lengths 72) | every one identical to candbin4's, draft statistics included |
| the 7 hashes | tf 842278b28d87dad0; k1, fp8 and bf16 febc7dc54ee40a39; temp a8f7a08f8cd1c5e8; ced16 81f772ab79a950e4; ced28 14650fb0739d0383 (the hash's seed: that run teacher-forces no position, and until 2026-10-13 it hashed nothing, so this shows only that the run completed; it now gives 8f08a153d415c5c8, [golden/README.md](../golden/README.md#checking-a-build)) |
| the forced race | 0 of 176 wrong (above) |
| cold prefill, median ms (fix1 against candbin4) | 3,100: 503.1 / 501.8; 3,600: 535.7 / 536.1; 4,096: 579.0 / 578.3; 2,400 (no wait at 3 pieces): 427.2 / 426.9 |

These kernels run only in prefill. Their cost is within the parts' spread of about ±0.3 %.

**Why the tests missed it.**

- ring_test's bulk reduce-scatter cases include a short last piece after six full ones (6 cap +
  192 B), but each case runs once at an even pace, and nothing stalls one block by a piece's transfer.
- The integrity runs cover the single-shot collectives and the bulk all-reduce, which is not exposed.
- The hashes run each prompt once.
- The stress client's 3,436 requests had no natural fault.

**The regression test** (2026-10-13, `ring_test race`, [Build and test](#api)) forces the window at the
engine's shapes, so the race no longer depends on chance to show. Two parts on the ring at the
engine's 4 MiB cap gave:

| comm code | rs, 776 rows, 4 pieces | rs, 903 rows, 5 pieces | all-to-all, 1,602 rows, 4 pieces | all-gather, 2,859 rows, 4 chunks |
|---|---|---|---|---|
| 9610544's (no `wait_posted`) with the hook | wrong in 6 of 6 runs, the same counts each time; per rank 4,095 / 32,755 / 22,517 / 4,094 of 993,280 | wrong in 6 of 6; 16,375 / 20,468 / 0 / 0 of 1,155,840 | wrong in 6 of 6; 21,504 / 0 / 0 / 0 of 3,383,424 | exact in 3 of 3 |
| master's | exact in 7 of 7 | exact in 7 of 7 | exact in 7 of 7 | exact in 3 of 3 |

Without the stall, every case was exact on both. The counts are again the predicted overlap, now
with blocks of 1,024 threads: rank 0, for example, loses one run of 1,024 units (4,096 floats; a
sum comes out the same when the overwritten contribution is too small to change it). At ring_test's
default 1 MiB cap (194, 237 and 399 rows; 714 for the all-gather), the result was the same: the
three cases wrong in 3 of 3 runs without the fix, again with the same counts each time, and every
case exact in 5 of 5 with it. A `ring_test race` takes ~6 s on the ring, mostly run4.sh's copies
and the bootstrap. The default suite with `--integrity 300000` passed in 31 s.

The all-gather case has no race to find. `allgather_multi_kernel` also copies chunk j + 2 into
s − 1's send buffer with no wait. But its copy and its read of the rank's own contribution (which
`src` points at the send buffer) give unit i to the same thread: both loops start at the thread's
index in the grid and step by the grid's size. So each thread reads its units of s − 1 before it
overwrites them. The case guards that property. If a change gave the copy another layout, the other
blocks would overwrite block 1's units while it stalls.

**The audit** (2026-10-13) looked for the hazard beyond the three kernels: any multi-block kernel
that rewrites a buffer that other blocks of its grid may still read, guarded only by a posted or
arrived count. It read every user of `send_buf`, `post_grid`, `wait_all`, `finish_grid` and
`next_seq`, and every counter and slot ring, on master c247a78. A model of the pipelined kernels' op
orders, `engine/comm/tests/race_model.py` (~1 s), checks every read of a slot's old data against
every rewrite of that slot by another block. It covers 1 to 8 pieces (the three slots wrapped more
than twice), last pieces that are full, of 1 unit, or multiples and non-multiples of the grid's
threads, in 4 grid shapes. It finds the race in the unfixed order, at 4 or more pieces with a last
piece that is not a multiple of the grid's threads, and nothing else. No other kernel is suspect:

| kernel | verdict | why |
|---|---|---|
| `reduce_scatter_bulk_kernel`, `alltoall_rows_kernel`, `k_moe_combine_rs` | safe with the fix | `wait_posted(s + 1)` comes before each copy into s − 1's buffer. The copies before the loop and at i = 0 touch only slots that earlier kernels are done with. |
| `allgather_multi_kernel` | safe, but fragile | It copies two chunks ahead into s − 1's buffer with no wait, but each unit has one thread (above). A `wait_posted` there would remove the dependence; that is a proposal, not done. |
| `allgather_bulk_kernel` (`comm::allgather` above the cap) | safe | It is one deep: the copy into s − 2's buffer follows `wait_all(s)`, which every block reaches only after its copy-out of s − 2. Each unit also has one thread. |
| `allreduce_bulk_kernel` (ring_test only) | safe | Each rewrite follows a `wait_all` that every block reaches only after its reads of the old data. |
| the single-shot collectives: `allreduce_kernel`, `reduce_scatter_kernel`, `allgather_kernel` and `argmax_kernel`; decode's `k_allreduce_hc` (`allreduce_hc`, `allreduce_hc_mix`), `k_allgather_attn` and `k_mix_allreduce`; `k_sample`; the drafter's `RingComm` | safe | One collective a kernel: the send buffer is written before `post_grid(s)` and read after `wait_all(s)`, and its slot is next written at s + 3, by a later kernel. Every PDL-launched ring kernel waits (`griddepcontrol.wait`) before `next_seq`, so no ring kernel overlaps the one before it. |
| Engram's gathers | safe | In decode, the host rewrites the pinned send buffer only after `wait_done` of the step that read it. In prefill there are two send buffers: the next chunk's lookup writes the other one, and `prefill()` drains before each chunk. |
| the StepIn and SampleOut slot rings (4 slots) | safe | The host writes slot n mod 4 only after `wait_done(n − 4)`. Outputs are published with a system fence and a release store and read with acquire, and an overwrite is caught (`wait()` dies). |
| the persistent GEMVs' shared-memory rings (`gemv_fp8.cu`, `gemv16.cu`) | safe | Each warp's ring is private: a lane reads only its own 16 B of a stage and refills it after `__syncwarp`. |
| the indexer's counters and the MoE fused combine's `done` counters | safe | They are last-arriver counters with fences on both sides. The indexer's last block resets them after every poller has left; the combine's are zeroed once a layer on the stream. |
| the L2 prefetch (B′) | safe | It only reads `ch.ready` and `ch.seq`. |

The streams keep the ring kernels in order. Ring kernels run on `st`, except prefill's all-to-all,
MoE reduce-scatter and combine. Those run on the comm stream `p.cs`, between `cs_after` and
`rs_join`, with no `st` ring kernel in between. Remote flow control holds too: every pipelined kernel
posts s + 1 only after `wait_all` on s, and each block counts in only after its reads of s − 1.

The proxy's segment arithmetic (`ring.cpp`'s `split_at`, `rs_split`, `direct_segs` and
`relay_seg`) was checked by enumeration, `engine/comm/tests/seg_model.py` (~6 s). For every post
size up to 4 MiB (16-byte steps for the all-exchange, 64 for the reduce-scatter), every sender and
six slow-link shares, each rank gets exactly the bytes it needs of the post, once and 16-byte
aligned. Each relay forwards the part its neighbour sent it as A. That is 2.6 M cases, with 0 bad.
The non-ring PDL dependents were not audited kernel by kernel; their work before the wait should be
weight loads only.

**The wider sweep** (2026-10-13) ran master's fixed build, release candidate 5 (c247a78,
`~/dsgb10/candbin5`, engine b352b870), and looked for any repeat that differs from its first. It ran
in four parts of about 7 min 20 s on the ring, each on a fresh server:

| what | requests | repeats compared | faults |
|---|---|---|---|
| cold prefills at 94 lengths from 1,024 to 9,088 tokens: every 128, plus each edge where the MoE reduce-scatter's or the all-to-all's piece count changes or where their last piece turns full or exposed, the fused combine's switch (2,520 / 2,521), 8,193 and 9,000; 8 tokens with log-probs, 10 times each | 940 | 846 | 0 |
| cold prefills of 16,384 and 28,000 tokens (two and four chunks) | 80 | 78 | 0 |
| spec decode at the sampled settings (temperature 1.0, top-p 0.95, seeds 11, 22 and 33) on prompts of 3,000, 1,200 and 600 tokens, up to 128 tokens | 270 | 261 | 0, with the draft statistics equal too |
| B1 slicing: a 192-token answer (A), with a 48-token one (B) sent after A's 40th token, A greedy or sampled, and each of them alone | 192 | 185 | 0, and overlapped equal to alone |

The lengths cover every MoE piece count from 1 to 10 and the all-to-all's 1 to 5. From 4 pieces on,
each count has a short last piece that the race exposed and one that it did not. The full last
pieces, which occur only at 5 and 10, are covered too. In the slicing rounds, B ran inside A's
decode: A's decode took 2.9 s, against 1.8 s alone. In all, 1,482 requests gave 1,370 repeats
with no fault (95 % bound: under 1 in 457 repeats).

**Open.** It is not known what could stall one block of a prefill's reduce-scatter for ~0.17 ms. A third
knob, `DSGB10_RSWIN` (every block's progress word and the least margin before each copy), was meant
to measure how close blocks come to that window. It hung the engine, with and without the fix and
on its own (a bisect: the delay alone and the check alone ran). The watchdog showed every rank's
proxy at posted 323, next 324. The kernel code adds no wait, and the cause was not found, so the
natural margins were not measured. The hang is left open: the audit above does not point at it. Its
host side uses the same synchronous copies as `DSGB10_RSCHECK`, which ran.

**Failures and dead ends.**

- The RSWIN hang, above.
- The plan expected 3,600 to fail under the delay. It could not, for the reason above.
- A build without PDL (det-diag 0c18d86) was built and never run, because the race was found first.
  So was a build with the comm stream's collectives on the main stream and per-layer hashes of the
  prefill's buffers (5df0add), and the Engram check, `DSGB10_ENGCHECK` (b66064f).
- The first start of the verification parts failed within 4 s: a binary name with a slash made
  its log paths into directories.

## Hazards

- **Page cache vs CUDA on GB10.** CUDA context creation fails (`cudaFree(nullptr): out of memory`)
  when MemFree is low, even with 100+ GB MemAvailable in page cache. It happened on rank 2 with
  MemFree at 11 GB and at 0.9 GB while another job read the checkpoint. Start CUDA before reading
  big files, read weights with O_DIRECT (or drop them from the cache), and check MemFree in
  launchers.
- **A rank that dies at start-up** leaves the others waiting in the bootstrap until
  `Options::bootstrap_timeout_s` (default 600 s; ring_test uses 120 s). A rank that dies later
  leaves the others spinning in `wait_all`. Launchers must time out and `pkill -x`
  the survivors (run4.sh does). `ring_test` prints `describe()` and exits after 20–60 s.
- **A shared GPU slows every rank.** With another job on rank 2's GPU (the golden reference), the
  10 KB all-reduce took 34 µs instead of 12. The ring runs at the pace of its slowest box, so the
  engine must own all four GPUs and the proxies' cores (5–8).
- **CPU idle states** take up to 433 µs to exit (docs/engram.md). Proxies spin on pinned X925 cores
  and only back off after 1 s without traffic.
- **L0 is 100G**, and each PCI domain is PCIe Gen5 x4 (~13.6 GB/s). Bulk traffic is bound by L0.
- **AArch64 ordering.** Four rules keep data and flags in order:
  - the proxy reads the doorbell with an acquire load and issues `dmb osh` before posting (NCCL
    needed a fence patch on these CPUs);
  - the GPU publishes with a system-scope release after the data;
  - received data is read with `ld.global.cv`, so no stale line from the parity buffer's previous
    use is served;
  - the integrity test checks payloads, not just flags.
- **Co-residency.** A kernel that waits must have at most 48 blocks (one per SM) or it deadlocks.
  The collectives use 1–48 blocks of up to 1024 threads.
- **Readers inside the grid.** A pipelined kernel that rewrites its own send buffer must know that
  every block of its grid is done reading it, not only that the ring is. A short last piece put one
  block's copy on another block's reads: a rare wrong prefill, fixed on 2026-10-13 (`wait_posted`;
  [the race](#the-pipelined-reduce-scatters-send-buffer-race-2026-10-13)).
- **cudaHostAlloc** memory reads 25 % slower on the GPU than cuMemCreate host-NUMA memory.
- **perftest** servers listen on 0.0.0.0 unless given `--bind_source_ip` (`micro/ring_bw.sh` passes
  it).

## Deviations from DESIGN.md decision 2

- One proxy thread per rank (core 5) drives all four QPs, not one thread per port. Posting costs 0.13 µs,
  and a single thread keeps the relay ordering rule (the opposite direction's CQE) free of locks.
- Buffers are cuMemCreate host-NUMA memory, not cudaHostAlloc: the same registration, faster reads.
- Each neighbour gets two RC QPs, one per PCI domain; the halves of a contribution travel on both.
- Reduce-scatter and the bulk all-reduce were added for large payloads.

## Open

- The bulk path reaches ~65 % of L0's bound. With fusion its copies mostly disappear; finer
  chunks within a reduce-scatter (relay cut-through) would shorten the two-hop store-and-forward.
- Programmatic dependent launch in the decode graph could hide part of the 3.2–3.6 µs GPU side.
- Relay cut-through (forwarding part A in pieces) would save ~0.4 µs at 10 KB.
- The engine needs a watchdog around steps (describe, then exit): a dead peer is otherwise a hang.
- What could stall one block of a prefill's reduce-scatter for ~0.17 ms, the race's natural window,
  is not known; the diagnostic meant to measure it hung the engine
  ([the race § Open](#the-pipelined-reduce-scatters-send-buffer-race-2026-10-13)). That hang is left
  open, since the audit of the hazard class does not point at it.
