// Collectives for the four-box ring (DESIGN.md decision 2, docs/ring.md): host side.
//
// Every collective is an all-exchange: each rank's contribution reaches every other rank, the two
// neighbours directly and the opposite rank through a neighbour's CPU proxy (half of it through
// each neighbour). Consumers then reduce in fixed rank order 0,1,2,3 (all-reduce) or read the four
// contributions in place (all-gather), so every rank computes bit-identical results. A
// reduce-scatter post sends each rank only its own quarter of the contribution, the same way; large
// all-reduces are a reduce-scatter and an all-gather of the reduced quarters (0.875 instead of 1.5
// times the payload per link and direction).
//
// Buffers live in one host region (cuMemCreate on the host NUMA node: CPU- and GPU-mapped at the
// same address, registered with every RDMA device). Collective s uses buffer set s % kSlots. A rank
// may post s once it has consumed s - (kSlots - 1) and seen every rank's s - 1 arrive; plain
// collectives (consume s - 1, then post s) always do, and the bulk all-reduce pipelines one deep
// within that rule (docs/ring.md has the argument). A CPU proxy thread
// turns GPU doorbells into RDMA writes and relays neighbours' data; GPUs only touch memory, so a
// whole step, collectives included, can be one CUDA graph.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "comm/ring_config.h"

namespace comm {

// One cache line per flag. Writers set nbytes and op, then seq; waiters compare seq for equality.
struct alignas(64) Flag {
  uint64_t seq;
  uint32_t nbytes;
  uint32_t op;  // Op
  uint32_t pad[12];
};

// What a post sends. kOpAllExchange: the whole contribution to every rank. kOpReduceScatter: the
// contribution is four quarters, quarter k for rank k; rank k receives quarter k of every other
// rank's contribution, at the same offset as in the sender's buffer.
enum Op : uint32_t { kOpAllExchange = 0, kOpReduceScatter = 1 };

// The flags a rank waits on before reading a collective's contributions: both halves of each
// neighbour's direct write (they may travel on different PCI domains) and the two relayed halves.
enum WaitFlag { kWaitPrevA, kWaitPrevB, kWaitNextA, kWaitNextB, kWaitOpp0, kWaitOpp1, kNumWait };

// Buffer sets: collective s uses set s % kSlots.
constexpr int kSlots = 3;

// What a kernel needs to take part in collectives. Plain data, passed by value.
struct DevChannel {
  int rank;
  uint32_t cap;                        // max bytes of one rank's contribution
  uint8_t* src[kSlots][kRanks];        // [slot][rank]: that rank's contribution (own: send buffer)
  Flag* wait[kSlots][kNumWait];        // [slot][WaitFlag]: all equal seq ⇒ src[slot][*] complete
  Flag* post[kSlots];                  // [slot]: GPU → proxy doorbell
  unsigned long long* seq;             // device memory: sequence number of the last finished collective
  unsigned long long* ready;           // device memory: last collective whose arrival block 0 has seen
  unsigned int* arrive;                // device memory: grid arrival counter for post_grid
  unsigned int* depart;                // device memory: grid departure counter for finish_grid
};

struct Options {
  std::string fleet;                   // the fleet file (ring_config.h); "": fleet.conf beside the executable
  int rank = -1;                       // -1: from the hostname (the fleet's RANK<r>_HOSTNAME)
  size_t cap = 1 << 20;                // max bytes of one rank's contribution (region: 4 * kSlots * cap)
  int port = 18620;                    // TCP bootstrap port on the ring-link addresses
  int bootstrap_timeout_s = 600;       // give up if a neighbour has not connected by then
  // Pin the proxy thread. X925 cores 5-8 are for comm proxies (9: Engram worker, 15: the engine's
  // main thread); a sleeping core takes up to 433 us to wake, so the proxy spins. -1: no pin.
  int proxy_cpu = 5;
  // After this long without traffic the proxy polls every idle_poll_us instead of spinning, so an
  // idle engine does not burn a core; the first collective after that pays ~0.5 ms. 0: always spin.
  int idle_backoff_ms = 1000;
  int idle_poll_us = 50;
  bool verbose = false;
  // Single-box diagnostics, no network: kLoopProxy - the proxy answers each post by setting the
  // local wait flags itself (GPU → proxy → GPU); kLoopNone - no proxy at all (kernels fake arrival).
  enum Mode { kRing, kLoopProxy, kLoopNone } mode = kRing;
  int trace = 0;  // record proxy-side timestamps for the first `trace` collectives (diagnostics)
  // Contributions of at least this many bytes send their two halves on the two PCI domains at once
  // (each NIC function DMA-reads its half in parallel); smaller ones use domain 0 only.
  uint32_t split_domains_from = 4096;
  int relay_domain = 1;  // the PCI domain all-exchange relays travel on (reduce-scatter: the other)
  // Reduce-scatter: percent of the opposite rank's quarter that takes the path across the slow link
  // (kSlowLink); the rest goes the other way round. 50 splits it evenly. 25 would load L0 with 1.5
  // instead of 2 quarters per direction, as much as the busier PCI domain of a 200G link, but measured
  // slower (14.4 against 18.4 GB/s at 64 MiB): relays are store-and-forward, so a post's time follows
  // its larger relayed part over two hops, not the links' loads.
  int rs_slow_pct = 50;
};

// Proxy-side timeline of one collective, CLOCK_MONOTONIC_RAW ns on this box (0: not seen).
struct TraceRec {
  uint64_t post;          // GPU doorbell seen
  uint64_t relay[2];      // relay posted: [0] to prev, [1] to next
  uint64_t arrive[kNumWait];  // wait flags seen by the CPU, WaitFlag order
};

struct ProxyStats {
  uint64_t posted;        // collectives sent for the local GPU
  uint64_t relayed[2];    // [0] next's data to prev, [1] prev's data to next
  uint64_t relay_waits;   // relays held back until the other direction's previous relay completed
  uint64_t post_retries;  // send-queue full
  uint64_t post_calls;    // ibv_post_send calls
  uint64_t post_ns;       // time spent in them
};

class Ring {
 public:
  explicit Ring(const Options& o);
  ~Ring();
  Ring(const Ring&) = delete;
  Ring& operator=(const Ring&) = delete;

  int rank() const;
  const DevChannel& dev() const;
  ProxyStats stats() const;
  const TraceRec* trace() const;       // Options::trace records, indexed by seq (valid once idle)
  std::string describe() const;        // proxy and flag state, for diagnosing a hang

  // Host-side helpers over the bootstrap sockets (setup and tests only, ~100 µs).
  void barrier();
  void allgather_host(const void* mine, void* all, size_t n);  // all = 4 × n bytes, rank order

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

int rank_from_hostname(const std::string& fleet_file = "");  // -1 if this box is not in the fleet

}  // namespace comm
