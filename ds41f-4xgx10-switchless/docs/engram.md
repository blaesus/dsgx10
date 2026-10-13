# Engram row store

Engram adds to layers 1 and 14, for every position, 24 rows of 264 bytes looked up by n-gram hashes
in two tables of 384M rows (188.8 GiB together, so they stay on NVMe: DESIGN.md decision 8). This
part turns token ids into those rows on every rank: hashing (`engine/engram/hash.cpp`), the row
store (`store.cpp`, `uring.h`, `cache.h`), the hand-off to the all-gather (`CellMap` in `engram.h`)
and the tools that check and measure them (`engine/engram/tools/`).

## At a glance

- **Hashing** is exact against `ref/inference/engram.py`: 13 synthetic and text cases plus the golden
  long transcript (28,489 positions), 1.87M row ids, each checked whole, one position at a time and in
  random chunks. 162–173 ns per position on one core.
- **Reads**: rank r reads only its quarter of each table, from the kit's shard on its own SSD, with
  O_DIRECT through io_uring on raw syscalls, behind a CLOCK cache. The SSDs give 0.54–0.58M
  (rank 0, rank 1) and 1.06–1.10M (ranks 2, 3) random records per second at queue depth 128.
- **Decode**: layer 1's rows are packed on this rank p99 123–226 µs after the token ids are known
  for one position, and 215–363 µs for six, on every box. The largest of 4,800 steps was 0.50 ms.
- **Prefill**: a 4096-token chunk's rows take a median 10–65 ms per box (61k–260k tok/s with no
  overlap; the slowest chunk took 97 ms). At 10k tok/s that is at most 16% of the chunk's time,
  and it overlaps by reading a chunk ahead.
- **Cache**: 4 GiB per rank (4.4 GB with its index).

## Hashing

`hash_positions(map, tokens, live, first, n, out)` is the reference's `NgramHashState`. Token ids
map to compressed ids (129,280 → 99,092; empty n-gram slots use the pad id 2). For each layer and
n-gram size k = 2..4, a rolling XOR of `t[s] * mult[layer][s]` over the last k tokens gives, for
each of 8 heads, `row = rolling % prime + offset`. Look-back stops at the start of the sequence and
at a position whose `live` is 0 (image spans; text never has any). Only `tokens[first - 3 ..]` is
read, so a chunk, a decode step and a whole sequence hash identically (checked).

The constants live in `engram_hashmap.bin`, written from the checkpoint's tokenizer and config by
`py/make_hashmap_stdlib.py` (Python's standard library alone, under a second) or by
`py/make_hashmap.py` in the canary-roce image (with the reference's own code, and the hashing test
cases): token map, multipliers, primes and offsets.
`HashMap::load` recomputes the primes (the next unused primes above 15,999,999, drawn layer, then
n-gram size, then head) and the offsets, and refuses a file that disagrees. The file is 518,048
bytes, sha256 `2405ae1d345ec2f76f637e2de71918060c2160f24bc7ed5a0e0ac4f9e123b02e`, on all four boxes;
the original checkpoint's tokenizer and config are the uncensored set's. `make_hashmap_stdlib.py`
writes this file byte for byte from either checkpoint (2026-10-13) and fails on any other result. It
re-implements engram.py's token map (each token's byte-level decode, then NFKC, NFD, the marks
stripped, lowercase, whitespace collapsed, a lone space kept) and, for the multipliers, numpy's
SeedSequence, PCG64 and bounded draw. Python's Unicode database can differ between versions; 3.12
(Unicode 15.0) made it.

## Rows, shards and owners

The shards (`engram-l{1,14}-r{r}of4.bin`, 23.6 GiB each) have the MiaAI kit's layout: a 4096-byte
header (`<6Q`: magic `DSV41EN1`, layer, lo, hi, rows, 264), then the records of rows lo..hi, 256 e4m3
values and 8 ue8m0 scales each. Rank r owns rows [rows·r/4, rows·(r+1)/4) of each table (`Split`).
The shards in use were made by the kit's
[`scripts/pack_engram.py`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/main/scripts/pack_engram.py)
(numpy: `--model CKPT --out DIR --rank r --tp 4`, its default `--tp` being 3). `engram_pack CKPT DIR r`
(`tools/pack.cpp`) makes the same files from the checkpoint, reading only the rank's rows (about 51
GB) and writing through a `.part` file. Against the shards in use on all four boxes (2026-10-13), every
file's size, and the header and three windows of 4,096 records (start, middle, end) of every rank's
both layers, packed from the official checkpoint, are identical, as is a window of 1,000,000 records.

Each table is 24 consecutive column segments of ~16.0M rows, one per (n-gram size, head), so this
split is in effect by column: rank 0 has bigram heads 0–5, rank 1 bigram heads 6–7 and trigram heads
0–3, rank 2 trigram heads 4–7 and 4-gram heads 0–1, rank 3 4-gram heads 2–7. Ranks 0–2 also hold the
first 947–1,344 rows of the next rank's first column. Every rank therefore owns 6 of the 24 rows per
position and layer (124,990–124,994 lookups per rank over the same 10,416 positions × 2 layers), but
how often rows come back differs a lot: bigrams repeat far more than 4-grams (hit rates below).

| Rank | Box | Shards | SSD |
|---|---|---|---|
| 0 | rank 0 | `~/dsv41-engram-tp4/` | ESL01TBTLCZ, PCIe 4 x4 |
| 1 | rank 1 (10.0.0.4) | `~/dsv41-4x-ring/engram/` | ESL01TBTLCZ, PCIe 4 x4 |
| 2 | rank 2 (10.0.0.3) | `~/dsv41-4x-ring/engram/` | Samsung PM9C1a (MZALC4T0HBL1), PCIe 5 x4 |
| 3 | rank 3 (10.0.0.2) | `~/dsv41-4x-ring/engram/` | Samsung PM9C1a (MZALC4T0HBL1), PCIe 5 x4 |

`engram_rowcheck`: 3,003 rows of each shard (both layers, all four ranks) equal the checkpoint's
weight and scale.

## The store

A `Store` runs one worker thread. It owns an io_uring (`io_uring_setup`/`enter`/`register`, mmapped
rings, registered files and one registered buffer, `READ_FIXED`), the cache, and the bookkeeping of
reads in flight. The control thread hands it `Request`s through a single-producer ring 256 deep;
requests complete in any order.

For each cell it owns, in (position, column) order, the worker copies a cached row, joins a read of
the same row already in flight, or queues a miss. Misses go out while read slots are free
(`queue_depth`, 128): the smallest 512-byte-aligned window around the record (512 or 1024 bytes,
768 on average) into a bounce slot. A finished read goes into the cache and to every cell waiting
for it. When a request's last cell lands, the worker stores `seq` to `*done` with release order.
A window that runs past the end of a shard (whose size is not a multiple of 512) comes back short.
That is accepted as long as the record is complete.

The cache is CLOCK over fixed 264-byte slots, found through an open-addressing index (linear
probing, backward-shift deletion), in huge-page anonymous memory. New rows enter unreferenced, so a
row seen once leaves before a row seen twice. `engram_cachesim` replays the same index.

`engram_storecheck` compares every record the store returns with `pread`. The kit shards of every
rank were checked with 200 requests of 1–2048 positions (~792k records per rank), with a 64 MiB
cache so that eviction and joins run (rank 0 again with read padding on). The checkpoint was
checked at rank −1 with 16 requests (253k records). All records were equal.

### What decode latency depends on

The flash itself is not what limits decode latency. Three other things are, and all three are
handled in the store without root:

1. **CPU idle states.** GB10 cores have idle states with 231 and 433 µs exit latency. A worker that
   sleeps between decode steps (15 ms apart in the tests) wakes slowly: p50 was 230 µs per step even
   when every row was cached. The worker therefore spins for `spin_us` (default 1 s) after its last
   work before it sleeps on a futex. Pin it to an otherwise idle X925 core (CPUs 5–9, 15–19). The
   SSD's completion interrupt arrives on the submitting CPU (one NVMe queue per CPU), so a pinned,
   spinning worker sees it at once. With these settings, steps whose rows are all cached take
   4–16 µs (p50; up to 45 µs on the shared rank 2).
2. **SSD power states.** Idle time raises the latency of the next read (table below). After 100 ms
   idle, APST (the kernel's `apst_primary_timeout_ms` = 100) costs 1.1 ms on the ESL01TBTLCZ (rank 0, rank 1) and
   9.5 ms on the PM9C1a (ranks 2, 3). After 2 s it costs 10–20 ms. The ESL01TBTLCZ also slows by ~50 µs after
   ~10 ms idle. Decode reads come in bursts with long gaps, so the drive kept dropping into these
   states, which gave rank 0 a ~1.1 ms p99 for steps that read. Now, whenever 5 ms pass without a
   read (`keepalive_ms`), the worker reads one random record, waking from its sleep to do so. That
   is 200 reads/s, about 100 KB/s.
3. **Interrupt coalescing on the PM9C1a (ranks 2, 3).** The drive signals a completion once 8 are
   pending or 200 µs have passed, so 1–7 reads take 200 µs (p50 199.8 µs at queue depths 1, 2 and
   4). With `read_batch` = 8, whenever the worker has nothing more to issue, it rounds the reads
   outstanding up to a multiple of 8 with discarded reads. On rank 3, steps that read went from
   p50 237 / p99 256 µs to 86 / 169 µs. Deep prefill queues need no padding: storecheck issued 684
   padding reads against 642k real ones.

With root, the causes themselves could be switched off: APST with
`nvme_core.default_ps_max_latency_us=0`, and coalescing by setting NVMe feature 0x08 to 0 on
ranks 2 and 3. The store does not need either.

## Interface

```c++
engram::HashMap map = engram::HashMap::load("engram_hashmap.bin");
engram::StoreConfig cfg;            // rank r, its two shards, cache_bytes, cpu, read_batch
engram::Store store(cfg, map);

engram::hash_positions(map, tokens, nullptr, first, n, ids);   // ids: [n][2][24]
engram::CellMap cm;
cm.build(store.split(), layer, ids, n);                        // identical on every rank
engram::Request q{layer, ids, n, segment, done, ++seq};        // segment: cm.count[r] * 264 bytes
store.submit(q);                                               // returns at once
```

### Hand-off to the all-gather

- Every rank hashes the same tokens and builds the same `CellMap`, so every rank knows every
  segment's size without asking.
- Rank r's store packs its `count[r]` records into the start of its send segment, in (position,
  column) order. `cm.seg_bytes()` is the largest count × 264 rounded up to 16 bytes, and it is the
  one size every rank passes to `comm::allgather(ch, segment, gathered, cm.seg_bytes(), stream)`.
- The gathered buffer is `[seg_0 | seg_1 | seg_2 | seg_3]`. The row of cell (p, c) is at
  `cm.offsets()[p * 24 + c]` = `owner * seg_bytes + index * 264`. That table (n × 24 uint32) goes
  to the Engram kernel with the rows.
- The segment and `done` live in `cudaHostAlloc` memory that is registered for RDMA. Before the
  all-gather, the GPU stream waits on `done == seq` with a flag-spin kernel, like the collectives
  do.
- `engram_gathercheck` runs this hand-off on one box. Four stores, one per rank, read the
  checkpoint into one gather buffer. Over 20 requests of 1–2048 positions (499,968 cells), every row
  found through `offsets()` equals the single-box store's row, and nothing lands in the padding.
- Segment sizes per layer and rank: ~1.6 KB for 1 position, ~9.5 KB for 6, ~6.5 MB for 4096 (the
  largest seen at 2048 positions was 12,290 records, two boundary rows over 6 per position).
- The worker copies each row once, into the segment, and the GPU never reads the cache. The segment
  has to be packed for the all-gather anyway (at most 13 MB per 4096-token chunk per rank), and a
  private cache can evict a slot without coordinating with the GPU.

### Decode step

1. When the step's token ids are known (one, or up to six with speculation), the control thread
   hashes them (1–3 µs), builds both layers' maps and submits two requests.
2. The GPU runs layer 0 in the meantime. Before layer 1's Engram, its stream waits on layer 1's
   `done`, all-gathers, and reads the rows through the offsets. Layer 14's rows are not needed until
   13 layers later.

### Prefill

Submit chunk i+1's requests when chunk i starts. Hashing 4096 tokens takes ~0.7 ms on one core.
Reading the rows takes a median 10–65 ms, against ~410 ms of compute per 4096 tokens at 10k tok/s.

### Single-box mode

With `rank = -1` and `Source::Checkpoint`, the store serves every row from the checkpoint's
safetensors (weight and scale are two reads per row) into the full `[n][24][264]` grid in cell
order. It is checked by storecheck and serves as the reference in gathercheck.

## Measurements

All measurements below ran with the worker spinning on CPU 7, a 4 GiB cache and 5 ms keep-alive,
and with `read_batch` 8 on ranks 2 and 3. rank 2 was shared with the golden harness at the time
(load ~4), which shows in its CPU-side numbers.

### Random record reads (`engram_iobench`, 264-byte records, 768 bytes per read)

| QD | rank 0 IOPS / p50 / p99 µs | rank 1 | rank 2 | rank 3 |
|---:|---|---|---|---|
| 1 | 16.4k / 54 / 98 | 16.9k / 52 / 94 | 5.0k / 200 / 300 | 5.0k / 200 / 201 |
| 8 | 102k / 69 / 168 | 104k / 66 / 170 | 89k / 76 / 158 | 96k / 73 / 134 |
| 32 | 230k / 139 / 240 | 232k / 139 / 235 | 421k / 72 / 158 | 467k / 66 / 139 |
| 64 | 377k / 162 / 319 | 395k / 156 / 289 | 774k / 74 / 193 | 854k / 69 / 163 |
| 128 | 543k / 206 / 610 | 580k / 194 / 533 | 1,057k / 107 / 269 | 1,102k / 108 / 225 |

At 10k tok/s, a box needs 120k lookups/s (12 per token) before the cache. That is under a quarter of
what the slower drives give at QD 128.

### One read after an idle gap (`engram_iobench … gap`, QD 1, p50 µs)

| Gap | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---:|---:|---:|---:|
| ≤ 9 ms | 54–58 | 52 | 200 | 200 |
| 11–15 ms | 109–111 | 105–106 | 200 | 200 |
| 50 ms | 107 | 107 | 200 | 596 |
| 120 ms | 1,109 | 1,067 | 199 (busy box) | 9,497 |
| 2.5 s | 1,218 (p90 19,382) | 1,274 (p90 19,425) | 9,500 | 9,488 |

rank 3, 1 ms gap, one read sent with others: batches of 1–7 take 200 µs (p50). A batch of 8 takes
74 / 146 µs (p50 / p99), and a batch of 16 takes 64 / 138 µs.

### Hit rates (`engram_cachesim`, each rank's cache seeing only its own rows)

Streams:
- code: 2.81M tokens of Python stdlib.
- prose: 1.40M tokens of the tech report, licences and `/usr/share/doc`.
- golden: the golden harness's long transcript, 28,489 tokens.

| Stream, cache | kit split, ranks 0 / 1 / 2 / 3 | column split (every rank) |
|---|---|---|
| code, 4 GiB | 89.3 / 77.9 / 67.5 / 58.1 | 73.2 |
| prose, 4 GiB | 81.4 / 67.0 / 55.5 / 47.1 | 62.8 |
| code + prose + golden, 256 MiB | 82.3 / 65.5 / 52.3 / 42.7 | 60.4 |
| code + prose + golden, 1 GiB | 88.0 / 73.8 / 61.1 / 50.6 | 68.6 |
| code + prose + golden, 2 GiB | 88.3 / 75.6 / 64.1 / 54.1 | 70.8 |
| code + prose + golden, 4 GiB | 88.3 / 76.0 / 65.3 / 55.9 | 71.5 |
| golden alone, cold, any size | 59.2 / 45.5 / 35.0 / 27.7 | 41.9 |
| golden after code + prose, 2 GiB | 94.0 / 75.0 / 49.1 / 34.6 | 66.8 |
| golden after code + prose, 4 GiB | 94.0 / 89.1 / 85.7 / 57.8 | 88.3 |

Distinct rows over code + prose + golden: 1.5 / 3.0 / 4.3 / 5.5 GiB per rank with the kit split, and
3.6 GiB on every rank with the column split. A 4 GiB cache holds the whole 4.2M-token history of
ranks 0 and 1 (rank 2's is 4.3 GiB), and of every rank with the column split.

### End to end (`engram_e2e`: token ids → this rank's rows packed and ready)

Each run first warms the cache with 1M tokens as prefill (half the stream for prose). It then runs
300 decode steps 15 ms apart, first with 1 position per step and then with 6, followed by 8 fresh
4096-token chunks with a warm cache and 8 with a cold one. Times are from hashing to `done`.

| Code stream | rank 0 (r0) | rank 1 (r1) | rank 2 (r2) | rank 3 (r3) |
|---|---|---|---|---|
| decode hits, 1 position | 89.9% | 79.6% | 69.6% | 59.9% |
| layer 1 ready, 1 position, p50 / p99 µs | 3 / 123 | 6 / 127 | 15 / 226 | 7 / 159 |
| layer 1 ready, 6 positions, p50 / p99 µs | 8 / 215 | 113 / 247 | 139 / 363 | 132 / 217 |
| steps that read, 6 positions, p50 / p99 / max µs | 162 / 300 / 312 | 192 / 338 / 348 | 160 / 372 / 458 | 147 / 218 / 256 |
| prefill chunk, warm cache: ms / tok/s | 18.1 / 216k | 29.7 / 129k | 39.2 / 96k | 30.4 / 129k |
| prefill chunk, cold cache: ms / tok/s | 10.1 / 260k | 15.7 / 186k | 20.8 / 150k | 16.2 / 206k |
| prose: prefill warm / cold, tok/s | 202k / 89k | 121k / 72k | 61k / 64k | 131k / 104k |

Hashing took 0.8–2.4 µs per step for one position and 2.0–5.1 µs for six. On prose, layer 1 p99 was
79–134 µs for one position and 235–275 µs for six (max 501 µs, on rank 2).

## Configuration

- `cache_bytes`: 4 GiB per rank. On 3–4M-token streams hit rates flatten beyond 2 GiB, but 4 GiB
  keeps a whole history of that size, which lifted the golden transcript's hit rate on ranks 1–2 by
  14–37 points over 2 GiB. With a 72.4 GiB rank file, rank 3 (4-grams, the largest working set) is
  where extra memory helps most.
- `cpu`: an X925 core used by nothing else, and not the comm proxy's. `spin_us` must exceed the gap
  between decode steps (default 1 s).
- `keepalive_ms` 5 (default) on every box. `read_batch` 8 on ranks 2 and 3, 1 (default) on ranks 0 and 1:
  the engine takes each rank's from the fleet file's `RANK<r>_ENGRAM_READ_BATCH`, 1 where it has none.
- `queue_depth` 128.

## Column split (measured, not adopted)

If rank r owned columns c with c % 4 = r (two heads of each n-gram size), the following would change:
- Every rank would hit equally: 71.5% on code + prose + golden, and 88.3% on golden after the
  warm-up.
- The worst rank's misses would fall by about a third: 29% against 44%, and on golden 12% against
  42%.
- The gathered layout would be fixed (cell (p, c) is record p·6 + c/4 of rank c % 4), so no
  `CellMap` would be needed.

It would mean re-packing ~51 GB per rank, or serving from each box's checkpoint copy (weight and
scale in separate tensors, so scales would best be held in RAM at ~1.5 GB per rank). The kit shards
also back the SGLang restart path. Since reads are not the limit at 10k tok/s, the kit split stays.

## Files and tools

- `engine/engram/`: `engram.h` (API), `hash.cpp`, `store.cpp`, `uring.h`, `cache.h`; `measure.sh`
  runs the per-box checks and measurements.
- `py/make_hashmap_stdlib.py`: writes the hash map with the standard library alone.
  `py/make_hashmap.py`: writes the hash map and 13 reference cases. `py/tokenize_corpus.py`: writes
  the code and prose streams. `py/golden_case.py`: turns a golden case into a hash case and a stream.
- Tools (`engine/build/engram/engram_*`):
  - `hashcheck <map> <cases>`
  - `iobench <shard> [secs] [depths] [wait]` and `iobench <shard> gap [gaps-us] [reads] [batch]`
  - `pack <ckpt-dir> <out-dir> <rank> [--layer L] [--rows A:B]`
  - `rowcheck <shard> <ckpt-dir> [n]`
  - `storecheck <map> rank <r> <l1> <l14> [n]` and `storecheck <map> ckpt <r|-1> <dir> [n]`
  - `gathercheck <map> <ckpt-dir> <stream> [n]`
  - `cachesim <map> [split=kit|col] [caps=MiB,…] [from=pos|last] <streams…>`
  - `e2e <map> <rank> <l1> <l14> <stream> [cache-MiB] [spin-us] [cpu] [keepalive-ms] [read-batch]`
- Data:
  - rank 0 `/data/dsgb10/engram/`: `engram_hashmap.bin`, `hash_cases.bin`, `streams/{code,prose}.i32`,
    `golden-long/` and `logs/`.
  - Workers `~/dsgb10/engram/` (hash map, streams), binaries in
    `~/dsgb10/bin/engram/`, logs in `~/dsgb10/logs/engram-*.log`.
