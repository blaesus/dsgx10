// The decode / verify step as engine/model's runner chains it (model.cu), on one rank with synthetic
// weights: this directory's GEMVs and MoE and attn's compressor rows, indexer head weights and mHC
// projections run for real; collectives, attention and the indexer are STAND-INs that hold the stream
// as long as the 4-rank engine measured them (docs/engine.md, "Step times"). A stand-in waits for its
// predecessor, lets the next kernel launch (PDL, as the fused collectives do once they have posted)
// and holds for its time, so the weight kernels get the same windows to load ahead in as in the
// engine. The side stream has the engine's mixes: hc_project of this rank's 5120 columns per
// sublayer, the attention's partials riding its all-gather and summed by a one-block kernel
// (mix_ranks; here hc_finish), the FFN's by a one-block collective that spins for as long as the
// engine's (mix_allreduce; here a stand-in).
//
//   bench_chain [--rows R (1..16)] [--experts U] [--routes FILE [--windows N]] [--scales L] [--head NPY]
//               [--rounds N] [--variants a,b,...] [--ar-us 14] [--arf-us 18] [--ag-us 18] [--mix-us 49.5]
//               [--prefer-shared (cudaFuncCachePreferShared, device-wide)]
//
// The bundle (master 2812072) as the engine runs it: --scales L gives the pool's experts layer L's real
// scales (its first 96 experts' in MS_FILE, default rank 0's file), which the MoE reads through their 1-bit
// code (dsk::scale_code); --head NPY the real head slice (kerneltests' head_w.npy), coded (bf16x_encode).
// Variant parts raw and hraw read the raw scales and the raw head instead. Without the options: random
// scales and a random head, raw (the chain before the bundle).
// --routes FILE: the rows pick the experts that R consecutive positions of a target's output picked, from a
// --route-file (main.cpp; the edit bench's are in rank 0's /data/dsgb10/bench/gate/rt_*.bin): N windows (64)
// spread evenly over the file's, one per round in turn (every variant of a round the same), each layer's
// distinct experts on pool slots of their own, as --experts places its U.
//
// Prints the median and mean step time per variant (interleaved graph replays), and a "chainprof" line with
// the bytes a call site reads (the experts' mean count, coded or raw). For the per-kernel breakdown run it
// under nsys (--cuda-graph-trace=node) and feed the export to chainprof.py (--meta: this output): each step
// starts with k_step_in (a block per row) and the all-reduce stand-ins have the fused all-reduce's grid (40
// blocks).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"
#include "kernels/tests/dsw_read.h"
#include "kernels/tests/npy.h"

namespace dsk {  // kernel names as kprof.py shortens them (dsk::<unnamed>::)
namespace {

constexpr int kL = 40, kD = 5120, kCopies = 4, kPool = 96, kMR = dsk::kMaxRows;  // kMR: buffers' rows
constexpr int kQaPart = 320, kKvPart = 128, kCompPart = 128, kQ = 8192, kQLora = 1280, kIdxQ = 4096;
constexpr int kOPart = 2048, kWoaK = 4096, kOGroup = 1024, kExperts = 384, kGatePart = 96, kVocabPart = 32320;
constexpr int kEngN = 6400, kEngK = 6144, kHcDim = 4 * kD;

constexpr bool kv_src(int L) { return L == 2 || L == 8 || L == 14 || L == 20; }
constexpr bool idx_src(int L) { return kv_src(L) || (L >= 24 && L % 4 == 0); }
constexpr bool engram(int L) { return L == 1 || L == 14; }

__global__ void k_fill(uint8_t* p, size_t n, uint32_t seed, uint8_t lo, uint8_t span) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    const uint8_t v = (uint8_t)h;
    p[i] = span ? (uint8_t)(lo + v % span) : (uint8_t)(v & 0x77);
  }
}
uint8_t* dev_fill(size_t n, uint32_t seed, uint8_t lo = 0, uint8_t span = 0) {
  uint8_t* p;
  CUDA_CHECK(cudaMalloc(&p, n));
  k_fill<<<512, 256>>>(p, n, seed, lo, span);
  CUDA_CHECK(cudaGetLastError());
  return p;
}
template <class T>
T* dev_zero(size_t n) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
  return p;
}
const uint16_t* U(const void* p) { return static_cast<const uint16_t*>(p); }

// ---- STAND-INs ------------------------------------------------------------------------------------
// Wait for the predecessor, hold for ns. Collectives let the next kernel launch once they have posted
// (right away here: early 1); the attention kernels at their start, before their wait (early 2, as
// sparse_attn.cu's and indexer.cu's kernels do, so wo_a's CTAs are resident during the attention); the mix
// collective not at all. The names keep kprof.py's classes.
// The collectives' block 0 counts their arrivals in g_arrived (pfx's stop, as comm::DevChannel::ready): 2 us
// before the end, about when a collective's contributions are in and it adds and writes them; and their ends in
// g_finished (as ::seq).
__device__ unsigned long long g_arrived, g_finished;
__device__ __forceinline__ void hold(uint32_t ns, int early) {
  if (early == 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (early == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  bool tell = early == 1 && blockIdx.x == 0 && threadIdx.x == 0;
  uint64_t t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (tell && t - t0 + 2000 >= ns) atomicAdd(&g_arrived, 1ull), tell = false;
  } while (t - t0 < ns);
  if (early == 1 && blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&g_finished, 1ull);
}
__global__ void k_allreduce_hc_standin(uint32_t ns) { hold(ns, 1); }
__global__ void k_allgather_standin(uint32_t ns) { hold(ns, 1); }
__global__ void k_mix_allreduce_standin(uint32_t ns) { hold(ns, 0); }
__global__ void k_attn_kernel_standin(uint32_t ns) { hold(ns, 2); }
__global__ void k_attn_combine_standin(uint32_t ns) { hold(ns, 2); }
__global__ void k_index_fused_standin(uint32_t ns) { hold(ns, 2); }
__global__ void k_step_in() {}

// The pf variants' paced L2 prefetch: TensorFold's issue pattern (bertholomus/TensorFold's l2_prefetch.cu, as
// sudoingX/dgx-spark-deepseek-v41's exl3_l2pf.cu runs it; Apache-2.0, this code is ours). One block: each thread
// issues a cp.async.bulk.prefetch.L2 of an 8 KiB chunk, the ranges in order (each from a chunk of its own), a wave
// of blockDim chunks every wave_ns; then (pfs) it stays until linger_ns after its start. It writes nothing, so no
// output changes. No ranges (pfe): it only starts and ends.
constexpr int kPfRanges = 16;
constexpr uint32_t kPfChunk = 8192;
struct PfRanges {
  const uint8_t* p[kPfRanges];
  uint32_t n[kPfRanges];    // bytes (multiples of 16)
  uint32_t end[kPfRanges];  // chunks up to and including range i
  int count;
};
__global__ void __launch_bounds__(128) k_l2pf(const __grid_constant__ PfRanges r, uint32_t wave_ns, uint32_t linger_ns) {
  __shared__ uint64_t t0s;
  if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0s));
  __syncthreads();
  const uint64_t t0 = t0s;
  const uint32_t total = r.count ? r.end[r.count - 1] : 0;
  uint32_t wave = 0;
  for (uint32_t c0 = 0; c0 < total; c0 += blockDim.x, ++wave) {
    if (wave) {
      if (threadIdx.x == 0) {
        uint64_t t;
        do {
          __nanosleep(wave_ns >= 2048 ? 256 : 32);  // waves of 8 or 16 chunks come every 0.3-0.6 us
          asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        } while (t - t0 < (uint64_t)wave * wave_ns);
      }
      __syncthreads();
    }
    const uint32_t c = c0 + threadIdx.x;
    if (c >= total) continue;
    int i = 0;
    while (c >= r.end[i]) ++i;
    const uint32_t off = (c - (i ? r.end[i - 1] : 0)) * kPfChunk, left = r.n[i] - off;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(r.p[i] + off), "r"(left < kPfChunk ? left : kPfChunk)
                 : "memory");
  }
  if (linger_ns && threadIdx.x == 0) {
    uint64_t t;
    do {
      __nanosleep(1000);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < linger_ns);
  }
}

// pfk: the same prefetches from one warp at a token bucket's pace: each poll issues the chunks due by the clock
// (ns_chunk apart; up to 32), so the rate holds however long a poll takes beside other blocks (k_l2pf's waves of 8
// fell to ~133 GB/s of 170 there) and a burst is what came due since the last poll; it ends with the last chunk.
__global__ void __launch_bounds__(32) k_l2pf_tb(const __grid_constant__ PfRanges r, uint32_t ns_chunk) {
  uint64_t t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  t0 = __shfl_sync(~0u, t0, 0);
  const uint32_t total = r.count ? r.end[r.count - 1] : 0;
  for (uint32_t done = 0; done < total;) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    t = __shfl_sync(~0u, t, 0);
    uint64_t due = (t - t0) / ns_chunk + 1;
    if (due > min(total, done + 32)) due = min(total, done + 32);
    const uint32_t c = done + threadIdx.x;
    if (c < due) {
      int i = 0;
      while (c >= r.end[i]) ++i;
      const uint32_t off = (c - (i ? r.end[i - 1] : 0)) * kPfChunk, left = r.n[i] - off;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(r.p[i] + off),
                   "r"(left < kPfChunk ? left : kPfChunk)
                   : "memory");
    }
    if (due > done) done = (uint32_t)due;
    else __nanosleep(64);
  }
}

// Stand-in times at one row, from the engine's decode loop at 1k (nsys on rank 0): the attention's
// all-reduce (+1.2 us per further row), the FFN's (fp32, +5.1 us per row; median 18.0 us after
// k_moe_down), the attention's all-gather, the FFN's mix collective (busy 49.5 us at one row, 41.4 at
// six), attention, its combine, the indexer, Engram's all-gathers and the sampling.
struct Opts {
  int rows = 1, experts = 0, rounds = 15, windows = 64, scales = -1;
  float ar_us = 14, arf_us = 18, ag_us = 18, mix_us = 49.5f, attn_us = 10.6f, comb_us = 3.4f, idx_us = 23, eng_us = 20,
        argmax_us = 12;
  std::string routes, head;
  std::vector<std::string> variants = {"base"};
};

// What a variant changes in the chain.
struct Var {
  std::string name;
  bool gate_split = false;  // the router gate on this rank's 96 rows, then an all-gather of the logits
  bool mixes = true;        // the side stream's mixes (hc_project, mix_ranks, the mix collective)
  bool standins = true;
  bool moe_dyn = false;  // moe_up's task queue (MoeCfg::dyn)
  int moe_up = -1, moe_down = -1;  // MoE variants (MoeCfg::up/down, from 9 rows up16/down16); -1: the defaults
  bool head_sep = false;  // the head as rmsnorm + gemv_bf16 instead of the normed GEMV
  bool head1 = false;     // the head on one row (its cost at R rows against one)
  bool raw = false;       // with --scales: the MoE on the raw scales, not their code
  bool head_raw = false;  // with --head: the raw head, not the coded one
  // pf: a paced L2 prefetch on a side stream (k_l2pf) of what the next weight kernels read: 1 at the FFN's
  // all-reduce (the next layer's wq_a + wkv, then its wq_b and the indexer's wq), 2 at the attention's all-reduce
  // (the router gate), 4 at the attention (wo_a), 8 at the attention's all-gather (what 1 left of wq_b and the
  // indexer's wq); pf_gbps its pace, pf_wave the chunks a wave, pf_kb a launch's budget (0: all), pf_linger_us
  // how long a launch's block stays (0: until its last wave). The queue variants instead: pf_t the targets (q: wq_a
  // + wkv, b: wq_b + the indexer's wq, o: wo_a, x: wo_b, g: the router gate), pf_w the windows (1: the FFN's
  // all-reduce, for the next layer; 2: the all-gather; 3: the attention; 4: the attention's all-reduce), pf_fill a
  // window's budget as a share (%) of what its stand-in time holds at the pace; pf_empty: launches that issue nothing
  int pf = 0, pf_wave = 128, pf_kb = 0, pf_linger_us = 0, pf_fill = 100;
  float pf_gbps = 230;
  std::string pf_t, pf_w;
  bool pf_empty = false, pf_tb = false;  // pf_tb: k_l2pf_tb's token bucket instead of k_l2pf's waves
  bool pf_x = false, pf_x_stop = true;    // the engine's plan (dsk::l2pf_window), stopped by the arrivals or not
  // per call site: wq_a + wkv, wq_b (+ the indexer's), wo_b, Engram's wkv (FP8, tiled as model.cu has
  // them), wo_a, router gate, head
  dsk::GemvCfg cqa{1, 1, 8, true, true}, cqb{1, 1, 8, true, true}, cob{1, 1, 16, true, true},
      ceng{1, 1, 8, true, true}, cwoa{1, 4, 4}, cgate{1, 1, 8}, chead{2, 1, 16};
};
// A variant is a '+'-joined list of: base, split (router gate split + logits all-gather), nomix (no
// mixes), bare (no mixes, no stand-ins), pers (the persistent FP8 GEMV at every FP8 site; pqa, pqb,
// pob, peng: at one), pgate (the persistent bf16 GEMV for the router gate), r<S> (their rings: S
// stages per warp), c<N> (their grids), mdyn (moe_up's task queue: tasks past the first wave to whichever CTA is free),
// mu<N>, md<N> (the MoE's up and down variants: MoeCfg::up/down, from 9 rows up16/down16), hw2 (the head with
// two warps along N, 32 rows a CTA: half the CTAs, each normalising x), hsep (the head as rmsnorm + gemv_bf16:
// x normalised once), h1 (the head on one row), raw (with --scales: the experts' raw scales), hraw (with
// --head: the raw head), pfq, pfg, pfo, pfa (a paced L2 prefetch on a side stream: at the FFN's all-reduce the
// next layer's wq_a + wkv and wq_b, at the attention's all-reduce the router gate, at the attention wo_a, at the
// all-gather what pfq's budget left of wq_b; pf: pfq + pfg), pfr<G> (its pace, GB/s; 230), pfw<N> (chunks of 8
// KiB a wave, up to 128), pfb<KB> (a launch's budget), pfs<us> (its block stays resident that long); the queue
// variants pfT<targets> (q, b, o, x, g: wq_a + wkv, wq_b + the indexer's, wo_a, wo_b, the gate) at pfW<windows> (1:
// the FFN's all-reduce, for the next layer; 2: the all-gather; 3: the attention; 4: the attention's all-reduce):
// each window prefetches the next bytes of its layer's targets the chain will read and no earlier window took, as
// many as the window's stand-in time holds at the pace, pff<percent> of it (100); pfe (every pf launch issues
// nothing: the graph's topology alone), pfk (one warp at a token bucket's pace instead of waves), pfx (the
// engine's L2 prefetch: kernels/l2pf.cu's plan at the four windows, the collectives' ends by their arrivals), pfxn
// (the same by its budgets alone). (The reverted GEMV prefetch, pf<KB>,
// issued its prefetches unpaced in the PDL window; micro/l2_retention.cu finds unpaced prefetches partly dropped and
// paced ones kept.)
Var make_var(const std::string& n) {
  Var v;
  v.name = n;
  for (size_t p = 0; p <= n.size();) {
    size_t q = n.find('+', p);
    if (q == std::string::npos) q = n.size();
    const std::string f = n.substr(p, q - p);
    p = q + 1;
    if (f == "base") continue;
    if (f == "split") v.gate_split = true;
    else if (f == "nomix") v.mixes = false;
    else if (f == "bare") v.mixes = false, v.standins = false;
    else if (f == "pers") v.cqa.ring = v.cqb.ring = v.cob.ring = v.ceng.ring = 0;
    else if (f == "pqa") v.cqa.ring = 0;
    else if (f == "pqb") v.cqb.ring = 0;
    else if (f == "pob") v.cob.ring = 0;
    else if (f == "peng") v.ceng.ring = 0;
    else if (f == "pgate") v.cgate.ring = 0;
    else if (f == "mdyn") v.moe_dyn = true;
    else if (f == "hw2") v.chead.wn = 2, v.chead.wk = 8;
    else if (f == "hsep") v.head_sep = true;
    else if (f == "h1") v.head1 = true;
    else if (f == "raw") v.raw = true;
    else if (f == "hraw") v.head_raw = true;
    else if (f == "pf") v.pf |= 3;
    else if (f == "pfq") v.pf |= 1;
    else if (f == "pfg") v.pf |= 2;
    else if (f == "pfo") v.pf |= 4;
    else if (f == "pfa") v.pf |= 8;
    else if (f == "pfe") v.pf_empty = true;
    else if (f == "pfk") v.pf_tb = true;
    else if (f == "pfx") v.pf_x = true;
    else if (f == "pfxn") v.pf_x = true, v.pf_x_stop = false;
    else if (f.rfind("pfT", 0) == 0 && f.size() > 3 && f.find_first_not_of("qboxg", 3) == std::string::npos)
      v.pf_t = f.substr(3);
    else if (f.rfind("pfW", 0) == 0 && f.size() > 3 && f.find_first_not_of("1234", 3) == std::string::npos)
      v.pf_w = f.substr(3);
    else if (f.rfind("pff", 0) == 0 && f.size() > 3 && std::isdigit((unsigned char)f[3])) v.pf_fill = std::atoi(f.c_str() + 3);
    else if (f.rfind("pfb", 0) == 0 && f.size() > 3 && std::isdigit((unsigned char)f[3])) v.pf_kb = std::atoi(f.c_str() + 3);
    else if (f.rfind("pfs", 0) == 0 && f.size() > 3 && std::isdigit((unsigned char)f[3]))
      v.pf_linger_us = std::atoi(f.c_str() + 3);
    else if (f.rfind("pfr", 0) == 0 && f.size() > 3 && std::isdigit((unsigned char)f[3])) v.pf_gbps = std::atof(f.c_str() + 3);
    else if (f.rfind("pfw", 0) == 0 && f.size() > 3 && std::isdigit((unsigned char)f[3]))
      v.pf_wave = std::max(1, std::min(128, std::atoi(f.c_str() + 3)));
    else if ((f.rfind("mu", 0) == 0 || f.rfind("md", 0) == 0) && f.size() > 2 && std::isdigit((unsigned char)f[2]))
      (f[1] == 'u' ? v.moe_up : v.moe_down) = std::atoi(f.c_str() + 2);
    else if ((f[0] == 'r' || f[0] == 'c') && f.size() > 1 && std::isdigit((unsigned char)f[1])) {
      for (dsk::GemvCfg* c : {&v.cqa, &v.cqb, &v.cob, &v.ceng, &v.cgate})
        if (c->ring >= 0) (f[0] == 'r' ? c->ring : c->ctas) = std::atoi(f.c_str() + 1);
    } else die("bench_chain: unknown variant part %s", f.c_str());
  }
  return v;
}

// --routes: n windows of R consecutive positions of one sequence, [n][kL][R][6] experts, spread evenly over a
// --route-file's windows in (sequence, position) order. The file's records: int32 {seq, kind, anchor, pos0, nr},
// int32 tokens[nr], int16 experts [kL][nr][6]; kind 0 holds positions' own experts (other kinds: drafts' rows).
std::vector<int16_t> route_windows(const std::string& path, int R, int n) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) die("bench_chain: cannot open %s", path.c_str());
  std::map<std::pair<int, int>, std::vector<int16_t>> pos;  // (seq, position) -> [kL][6]
  int32_t h[5];
  while (std::fread(h, 4, 5, f) == 5) {
    const int nr = h[4];
    std::vector<int32_t> tok(nr);
    std::vector<int16_t> e((size_t)kL * nr * 6);
    if (std::fread(tok.data(), 4, nr, f) != (size_t)nr || std::fread(e.data(), 2, e.size(), f) != e.size())
      die("bench_chain: %s is short", path.c_str());
    if (h[1] != 0) continue;
    for (int r = 0; r < nr; ++r) {
      std::vector<int16_t>& v = pos[{h[0], h[3] + r}];
      v.resize(kL * 6);
      for (int l = 0; l < kL; ++l)
        for (int k = 0; k < 6; ++k) v[l * 6 + k] = e[((size_t)l * nr + r) * 6 + k];
    }
  }
  std::fclose(f);
  std::vector<std::pair<int, int>> starts;
  for (const auto& p : pos) {
    bool ok = true;
    for (int k = 1; k < R && ok; ++k) ok = pos.count({p.first.first, p.first.second + k}) > 0;
    if (ok) starts.push_back(p.first);
  }
  if (starts.empty()) die("bench_chain: no %d-row windows in %s", R, path.c_str());
  n = std::min(n, (int)starts.size());
  std::vector<int16_t> out((size_t)n * kL * R * 6);
  for (int w = 0; w < n; ++w) {
    const std::pair<int, int> s = starts[(size_t)w * starts.size() / n];
    for (int m = 0; m < R; ++m) {
      const std::vector<int16_t>& v = pos[{s.first, s.second + m}];
      for (int l = 0; l < kL; ++l)
        for (int k = 0; k < 6; ++k) out[(((size_t)w * kL + l) * R + m) * 6 + k] = v[l * 6 + k];
    }
  }
  std::printf("bench_chain: %d of the %zu %d-row windows of %s's %zu positions\n", n, starts.size(), R, path.c_str(),
              pos.size());
  return out;
}

struct Layer {  // one copy of a layer's dense weights (layers rotate over kCopies copies)
  uint8_t *wqa, *sqa, *wkv, *skv, *wqb, *sqb, *wix, *six, *woa, *soa, *wob, *sob, *gate, *cwkv, *cwg, *wproj;
  __nv_bfloat16 *n_attn, *n_q, *n_ffn;
};

void run(const Opts& o) {
  // experts per layer: a verify step's rows share most of theirs (27 at 2..8 rows); from 9 rows (batched
  // steps of separate sequences) as many as R independent uniform picks of 6 of 384 touch: 51 at 9 rows,
  // 66 at 12, 86 at 16
  const int R = o.rows,
            nu = o.experts ? o.experts
                 : R == 1  ? 6
                 : R <= 8  ? 27
                           : std::min(kPool, (int)std::lround(kExperts * (1 - std::pow(1 - 6.0 / kExperts, R))));
  // ---- weights
  auto fp8 = [&](size_t n, size_t k, uint8_t*& w, uint8_t*& s, uint32_t seed) {
    w = dev_fill(n * k, seed), s = dev_fill(n / 32 * (k / 32), seed + 1, 118, 16);
  };
  std::vector<Layer> Ls(kCopies);
  for (int c = 0; c < kCopies; ++c) {
    Layer& d = Ls[c];
    const uint32_t b = 100 + 40 * c;
    fp8(kQaPart, kD, d.wqa, d.sqa, b);
    fp8(kKvPart, kD, d.wkv, d.skv, b + 2);
    fp8(kQ, kQLora, d.wqb, d.sqb, b + 4);
    fp8(kIdxQ, kQLora, d.wix, d.six, b + 6);
    fp8(kOPart, kWoaK, d.woa, d.soa, b + 8);
    fp8(kD, kOPart, d.wob, d.sob, b + 10);
    d.gate = dev_fill((size_t)kExperts * kD * 2, b + 12, 0x3c, 2);  // the tile_fp8 layout of bf16 rows
    d.cwkv = dev_fill((size_t)kCompPart * kD * 2, b + 13, 0x3c, 2);
    d.cwg = dev_fill((size_t)kCompPart * kD * 2, b + 14, 0x3c, 2);
    d.wproj = dev_fill((size_t)32 * kD * 2, b + 15, 0x3c, 2);
    d.n_attn = reinterpret_cast<__nv_bfloat16*>(dev_fill(kD * 2, b + 16, 0x3f, 1));
    d.n_q = reinterpret_cast<__nv_bfloat16*>(dev_fill(kQLora * 2, b + 17, 0x3f, 1));
    d.n_ffn = reinterpret_cast<__nv_bfloat16*>(dev_fill(kD * 2, b + 18, 0x3f, 1));
  }
  const size_t w13 = (size_t)576 * kD / 2, s13 = (size_t)576 * kD / 32, w2 = (size_t)kD * 576 / 2, s2 = (size_t)kD * 576 / 32;
  dsk::MoeWeights base{};
  base.w1 = dev_fill(kPool * w13, 1), base.w3 = dev_fill(kPool * w13, 3), base.w2 = dev_fill(kPool * w2, 5);
  // the bytes a call reads per routed expert in moe_up (w1, w3) and moe_down (w2): raw, and under the code
  // (its words, and the groups off the pair, read raw)
  double up_b[2] = {2.0 * (w13 + s13), 0}, down_b[2] = {(double)(w2 + s2), 0};
  dsk::ScaleCode code[3];
  if (o.scales < 0) {
    base.s1 = dev_fill(kPool * s13, 2, 120, 8), base.s3 = dev_fill(kPool * s13, 4, 120, 8);
    base.s2 = dev_fill(kPool * s2, 6, 120, 8);
  } else {
    const char* file = std::getenv("MS_FILE") ? std::getenv("MS_FILE") : "/data/dsgb10/weights/rank0.dsw";
    std::vector<uint8_t> h[3];
    read_expert_scales(file, o.scales, kPool, h[0], h[1], h[2]);
    uint8_t* d[3];
    for (int i = 0; i < 3; ++i) {
      CUDA_CHECK(cudaMalloc(&d[i], h[i].size()));
      CUDA_CHECK(cudaMemcpy(d[i], h[i].data(), h[i].size(), cudaMemcpyHostToDevice));
    }
    base.s1 = d[0], base.s3 = d[1], base.s2 = d[2];
    const size_t nc13 = dsk::scale_code_words(kPool, 576, kD / 32), nc2 = dsk::scale_code_words(kPool, kD, 576 / 32);
    uint32_t* dc;
    CUDA_CHECK(cudaMalloc(&dc, (2 * nc13 + nc2) * 4));
    int b[3];
    double f[3];
    for (int i = 0; i < 3; ++i) {
      f[i] = dsk::scale_code(d[i], kPool, i < 2 ? 576 : kD, i < 2 ? kD / 32 : 576 / 32, dc + i * nc13, &b[i], 0);
      code[i].code = dc + i * nc13, code[i].base4 = (uint32_t)b[i] * 0x01010101u;
    }
    size_t raw[3] = {0, 0, 0};  // groups (a step's 32 bytes of an L3 tile) holding a value off {b, b + 1}
    for (int i = 0; i < 3; ++i)
      for (size_t g = 0; g < h[i].size(); g += 32)
        for (int j = 0; j < 32; ++j)
          if ((unsigned)(h[i][g + j] - b[i]) > 1u) {
            ++raw[i];
            break;
          }
    up_b[1] = 2.0 * w13 + 8.0 * nc13 / kPool + 32.0 * (raw[0] + raw[1]) / kPool;
    down_b[1] = (double)w2 + 4.0 * nc2 / kPool + 32.0 * raw[2] / kPool;
    std::printf("bench_chain: layer %d's scales from %s, coded: groups w1 %.2f %%, w3 %.2f %%, w2 %.2f %%; an expert "
                "reads %.0f + %.0f bytes (raw %.0f + %.0f)\n",
                o.scales, file, 100 * f[0], 100 * f[1], 100 * f[2], up_b[1], down_b[1], up_b[0], down_b[0]);
  }
  std::vector<dsk::MoeWeights> MW(kCopies, base);
  for (int c = 0; c < kCopies; ++c) {
    MW[c].sh_w1 = dev_fill((size_t)576 * kD, 200 + c), MW[c].sh_s1 = dev_fill(18 * 160, 210 + c, 118, 16);
    MW[c].sh_w3 = dev_fill((size_t)576 * kD, 220 + c), MW[c].sh_s3 = dev_fill(18 * 160, 230 + c, 118, 16);
    MW[c].sh_w2 = dev_fill((size_t)kD * 576, 240 + c), MW[c].sh_s2 = dev_fill(160 * 18, 250 + c, 118, 16);
  }
  std::vector<dsk::MoeWeights> MWc = MW;  // with the code (--scales)
  for (auto& m : MWc) m.c1 = code[0], m.c3 = code[1], m.c2 = code[2];
  uint8_t *ewkv[2], *eskv[2];
  for (int e = 0; e < 2; ++e) fp8(kEngN, kEngK, ewkv[e], eskv[e], 300 + 2 * e);
  // the head: random, or the real slice (tiled) and its code
  const size_t head_raw_b = (size_t)kVocabPart * kD * 2;
  size_t head_x_b = 0;
  uint8_t *head, *headx = nullptr;
  if (o.head.empty()) {
    head = dev_fill(head_raw_b, 7, 0x3c, 2);
  } else {
    const Npy w = npy_load(o.head);
    if (w.shape.size() != 2 || w.shape[0] != (size_t)kVocabPart || w.shape[1] != (size_t)kD || w.data.size() != head_raw_b)
      die("bench_chain: %s is not a [%d][%d] bf16 head slice", o.head.c_str(), kVocabPart, kD);
    uint8_t* nat;
    CUDA_CHECK(cudaMalloc(&nat, head_raw_b));
    CUDA_CHECK(cudaMalloc(&head, head_raw_b));
    CUDA_CHECK(cudaMemcpy(nat, w.data.data(), head_raw_b, cudaMemcpyHostToDevice));
    dsk::tile_fp8(nat, head, kVocabPart, 2 * kD, 0);
    CUDA_CHECK(cudaFree(nat));
    int raw = 0;
    headx = dsk::bf16x_encode(head, kVocabPart, kD, &head_x_b, &raw, 0);
    if (dsk::bf16x_check(headx, head, kVocabPart, kD, 0)) die("bench_chain: the coded head does not decode to the head");
    std::printf("bench_chain: the head from %s, coded in %zu bytes (%.3f of raw), %d tiles raw\n", o.head.c_str(),
                head_x_b, (double)head_x_b / head_raw_b, raw);
  }
  __nv_bfloat16* n_out = reinterpret_cast<__nv_bfloat16*>(dev_fill(kD * 2, 8, 0x3f, 1));
  // every sublayer its own hc_fn, as the model (a rank reads its quarter: 39 MB per step)
  std::vector<float*> hc_fn(2 * kL);
  for (int i = 0; i < 2 * kL; ++i) {
    hc_fn[i] = dev_zero<float>((size_t)24 * kHcDim);
    k_fill<<<512, 256>>>(reinterpret_cast<uint8_t*>(hc_fn[i]), (size_t)24 * kHcDim * 4, 400 + i, 0x30, 8);
  }
  float* hc_sc = dev_zero<float>(3);
  float* hc_base = dev_zero<float>(24);

  // ---- activations and scratch (contents do not matter)
  __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(dev_fill((size_t)kMR * kHcDim * 2, 9, 0x3c, 2));
  __nv_bfloat16* yraw = reinterpret_cast<__nv_bfloat16*>(dev_fill((size_t)kMR * kD * 2, 10, 0x3c, 2));
  __nv_bfloat16 *y = dev_zero<__nv_bfloat16>(kMR * kD), *qa = dev_zero<__nv_bfloat16>(kMR * kQLora),
                *kvx = dev_zero<__nv_bfloat16>(kMR * 512), *q = dev_zero<__nv_bfloat16>(kMR * kQ),
                *iq = dev_zero<__nv_bfloat16>(kMR * kIdxQ), *part = dev_zero<__nv_bfloat16>(kMR * kD),
                *ekv = dev_zero<__nv_bfloat16>(kMR * kEngN), *ao = dev_zero<__nv_bfloat16>(kMR * 16 * 512);
  CUDA_CHECK(cudaMemset(qa, 0x3c, kMR * kQLora * 2));
  CUDA_CHECK(cudaMemset(ao, 0x3c, kMR * 16 * 512 * 2));
  uint8_t *oq = dev_zero<uint8_t>(kMR * kOPart), *os = dev_zero<uint8_t>(kMR * kOPart / 32), *yq = dev_zero<uint8_t>(kMR * kD),
          *ys = dev_zero<uint8_t>(kMR * kD / 32), *eq = dev_fill(kMR * kEngK, 11), *es = dev_fill(kMR * kEngK / 32, 12, 120, 8);
  float *pck = dev_zero<float>(kMR * kCompPart), *pcs = dev_zero<float>(kMR * kCompPart), *wproj = dev_zero<float>(kMR * 32),
        *gl = dev_zero<float>(kMR * kExperts), *bias = dev_zero<float>(kExperts), *ymoe = dev_zero<float>(kMR * kD),
        *logits = dev_zero<float>((size_t)kMR * kVocabPart);
  const int nch = attn::hc_chunks(0, kHcDim);
  float *chunks = dev_zero<float>((size_t)nch * kMR * 25), *hpart = dev_zero<float>(kMR * 25), *pre = dev_zero<float>(kMR * 4),
        *post = dev_zero<float>(kMR * 4), *comb = dev_zero<float>(kMR * 16);
  CUDA_CHECK(cudaMemset(hpart, 0x3c, kMR * 25 * 4));
  uint8_t* ws = dev_zero<uint8_t>(dsk::moe_workspace_bytes());
  // logits that make the MoE pick nu fresh experts per layer from the pool (the gate's own output is
  // computed but not used: its weights are random), or a route window's: each layer's distinct experts on
  // the pool slots after the previous layer's. Window w's logits are in lwin; a round copies its own to lsel.
  const std::vector<int16_t> routes = o.routes.empty() ? std::vector<int16_t>() : route_windows(o.routes, R, o.windows);
  const int nwin = o.routes.empty() ? 1 : (int)(routes.size() / ((size_t)kL * R * 6));
  const size_t lsel_n = (size_t)kL * kMR * kExperts;
  float* lsel = dev_zero<float>(lsel_n);
  float* lwin = dev_zero<float>(nwin * lsel_n);
  std::vector<int> wu(nwin, 0);  // experts a window touches over the 40 layers
  {
    std::vector<float> hl(nwin * lsel_n, -10.f);
    int slot = 0;
    for (int w = 0; w < nwin; ++w)
      for (int l = 0; l < kL; ++l) {
        float* hw = &hl[w * lsel_n + (size_t)l * kMR * kExperts];
        if (o.routes.empty()) {
          for (int m = 0; m < R; ++m)
            for (int k = 0; k < 6; ++k) hw[(size_t)m * kExperts + (l * nu + (m * 6 + k) % nu) % kPool] = 5.f - 0.1f * k;
          wu[w] += nu;
          continue;
        }
        std::map<int, int> at;  // the layer's experts -> pool slots
        for (int m = 0; m < R; ++m)
          for (int k = 0; k < 6; ++k) {
            const int e = routes[(((size_t)w * kL + l) * R + m) * 6 + k];
            if (e < 0) continue;
            auto it = at.find(e);
            if (it == at.end()) it = at.emplace(e, slot++ % kPool).first;
            hw[(size_t)m * kExperts + it->second] = 5.f - 0.1f * k;
          }
        wu[w] += (int)at.size();
      }
    CUDA_CHECK(cudaMemcpy(lwin, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lsel, lwin, lsel_n * 4, cudaMemcpyDeviceToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t sa, sb, sc;
  int lo = 0, hi = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&sa, cudaStreamNonBlocking, hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&sb, cudaStreamNonBlocking, hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&sc, cudaStreamNonBlocking, hi));  // the pf variants' prefetches
  unsigned long long *arrived, *finished;  // pfx's stop
  CUDA_CHECK(cudaGetSymbolAddress((void**)&arrived, g_arrived));
  CUDA_CHECK(cudaGetSymbolAddress((void**)&finished, g_finished));
  dsk::L2pfState* xstate = dev_zero<dsk::L2pfState>(1);
  std::vector<cudaEvent_t> evs(64);
  for (auto& e : evs) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

  auto step = [&](const Var& v, cudaStream_t st) {
    int ne = 0;
    auto fork = [&](cudaStream_t side) {
      CUDA_CHECK(cudaEventRecord(evs[ne % 64], st));
      CUDA_CHECK(cudaStreamWaitEvent(side, evs[ne % 64]));
      ++ne;
    };
    auto join = [&](cudaStream_t side) {
      CUDA_CHECK(cudaEventRecord(evs[ne % 64], side));
      CUDA_CHECK(cudaStreamWaitEvent(st, evs[ne % 64]));
      ++ne;
    };
    auto standin = [&](void (*k)(uint32_t), int blocks, int threads, float us) {
      if (v.standins) dsk::launch_pdl(k, dim3(blocks), dim3(threads), 0, st, true, (uint32_t)(us * 1000));
    };
    const float ar_us = o.ar_us + 1.2f * (R - 1), arf_us = o.arf_us + 5.1f * (R - 1),
                mix_us = o.mix_us - 1.6f * (R - 1);
    // a pf prefetch on sc, from the main stream's last kernel on: these weights (and scales) in order, from `skip`
    // bytes into them, up to `cap` bytes (the variant's budget); the bytes it takes
    const size_t budget = v.pf_kb ? (size_t)v.pf_kb << 10 : ~(size_t)0;
    auto l2pf = [&](const std::vector<std::pair<const uint8_t*, size_t>>& ws, size_t skip = 0,
                    size_t cap = ~(size_t)0) -> size_t {
      PfRanges r{};
      uint32_t c = 0;
      size_t left = std::min(cap, budget);
      const size_t start = left;
      for (const auto& w : ws) {
        size_t a = std::min(skip, w.second);
        skip -= a;
        const size_t n = std::min(w.second - a, left) & ~(size_t)15;
        if (r.count == kPfRanges || n < 16) continue;
        r.p[r.count] = w.first + a;
        r.n[r.count] = (uint32_t)n;
        left -= n;
        c += (r.n[r.count] + kPfChunk - 1) / kPfChunk;
        r.end[r.count++] = c;
      }
      if (!r.count) return 0;
      if (v.pf_empty) r.count = 0;
      fork(sc);
      if (v.pf_tb) k_l2pf_tb<<<1, 32, 0, sc>>>(r, (uint32_t)std::lround(kPfChunk / v.pf_gbps));
      else k_l2pf<<<1, v.pf_wave, 0, sc>>>(r, (uint32_t)(v.pf_wave * kPfChunk / v.pf_gbps), v.pf_linger_us * 1000u);
      CUDA_CHECK(cudaGetLastError());
      return start - left;
    };
    const size_t sc32 = 32 * 32;  // fp8's scales: a byte per 32 x 32 block
    const size_t qakv = (size_t)(kQaPart + kKvPart) * kD * (sc32 + 1) / sc32;  // wq_a + wkv with their scales
    // the queue variants: window w (pfW) of layer Lq's chain prefetches its enabled targets (pfT) in the order the
    // chain reads them, from the first one it comes before and past what earlier windows took, up to what its
    // stand-in time (us) holds at the pace
    int q_layer = -1;
    size_t q_done = 0;
    auto window = [&](int w, int Lq, float us) {
      if (v.pf_w.find((char)('0' + w)) == std::string::npos || Lq >= kL) return;
      if (q_layer != Lq) q_layer = Lq, q_done = 0;
      const Layer& e = Ls[Lq % kCopies];
      const size_t ix = idx_src(Lq) ? (size_t)kIdxQ * kQLora : 0;
      static const char kOrder[] = "qboxg";
      static const int kFrom[5] = {0, 0, 1, 2, 4};  // window w's first target in kOrder (the earlier ones are read)
      std::vector<std::pair<const uint8_t*, size_t>> ws;
      size_t first = 0;
      for (int t = 0; t < 5; ++t) {
        if (v.pf_t.find(kOrder[t]) == std::string::npos) continue;
        const size_t n0 = ws.size();
        if (t == 0)
          ws.insert(ws.end(), {{e.wqa, (size_t)kQaPart * kD}, {e.sqa, (size_t)kQaPart * kD / sc32},
                               {e.wkv, (size_t)kKvPart * kD}, {e.skv, (size_t)kKvPart * kD / sc32}});
        else if (t == 1)
          ws.insert(ws.end(), {{e.wqb, (size_t)kQ * kQLora}, {e.sqb, (size_t)kQ * kQLora / sc32}, {e.wix, ix},
                               {e.six, ix / sc32}});
        else if (t == 2)
          ws.insert(ws.end(), {{e.woa, (size_t)kOPart * kWoaK}, {e.soa, (size_t)kOPart * kWoaK / sc32}});
        else if (t == 3)
          ws.insert(ws.end(), {{e.wob, (size_t)kD * kOPart}, {e.sob, (size_t)kD * kOPart / sc32}});
        else
          ws.push_back({e.gate, (size_t)kExperts * kD * 2});
        if (t < kFrom[w])
          for (size_t i = n0; i < ws.size(); ++i) first += ws[i].second;
      }
      const size_t skip = std::max(q_done, first);
      q_done = skip + l2pf(ws, skip, (size_t)(us * v.pf_gbps * 10 * v.pf_fill));  // us x GB/s = KB
    };
    // pfx: the engine's window w of layer Lw (dsk::l2pf_window)
    auto xwin = [&](int w, int Lw) {
      if (!v.pf_x || Lw >= kL) return;
      const Layer& e = Ls[Lw % kCopies];
      const size_t ix = idx_src(Lw) ? (size_t)kIdxQ * kQLora : 0;
      const dsk::L2pfTarget t[dsk::kL2pfTargets] = {
          {{e.wqa, e.sqa, e.wkv, e.skv},
           {(size_t)kQaPart * kD, (size_t)kQaPart * kD / sc32, (size_t)kKvPart * kD, (size_t)kKvPart * kD / sc32}},
          {{e.wqb, e.sqb, e.wix, e.six}, {(size_t)kQ * kQLora, (size_t)kQ * kQLora / sc32, ix, ix / sc32}},
          {{e.woa, e.soa}, {(size_t)kOPart * kWoaK, (size_t)kOPart * kWoaK / sc32}},
          {{e.wob, e.sob}, {(size_t)kD * kOPart, (size_t)kD * kOPart / sc32}},
          {{e.gate}, {(size_t)kExperts * kD * 2}}};
      fork(sc);
      const bool stop = w != 3 && v.pf_x_stop;
      dsk::l2pf_window(t, Lw, w, R, stop ? arrived : nullptr, stop ? finished : nullptr, xstate, sc);
    };
    // a sublayer's mixes: this rank's quarter of hc_fn . stream on the side stream
    auto project = [&](int i) {
      if (!v.mixes) return;
      fork(sa);
      attn::hc_project(U(x), R, hc_fn[i], 0, kD, chunks, false, sa);
    };
    k_step_in<<<R, 32, 0, st>>>();
    for (int L = 0; L < kL; ++L) {
      const Layer& d = Ls[L % kCopies];
      if (engram(L)) {  // store rows all-gathered, wkv, the partial dots all-gathered
        standin(k_allgather_standin, 4, 256, o.eng_us);
        dsk::gemv_fp8(eq, es, R, ewkv[L == 14], eskv[L == 14], kEngN, kEngK, ekv, v.ceng, st);
        standin(k_allgather_standin, 4, 256, o.eng_us);
      }
      // ---- attention
      project(2 * L);
      const dsk::Fp8Seg seg[2] = {{d.wqa, d.sqa, qa, kQaPart}, {d.wkv, d.skv, kvx, kKvPart}};
      dsk::gemv_fp8_norm(yraw, d.n_attn, 1e-20f, R, seg, 2, kD, v.cqa, st, y);
      if (kv_src(L)) {
        const attn::CompressorWeights cw{L < 20 ? 2 : 1, U(d.cwkv), U(d.cwg), nullptr, nullptr, nullptr};
        for (int r0 = 0; r0 < R; r0 += attn::kMaxDecodeRows)  // attn's rows per call
          attn::compressor_project_rows(U(y) + (size_t)r0 * kD, cw, 0, kCompPart, pck + r0 * kCompPart,
                                        pcs + r0 * kCompPart, std::min(R - r0, attn::kMaxDecodeRows), true, st);
      }
      if (idx_src(L)) {
        fork(sb);
        attn::gemv_bf16(U(y), R, kD, U(d.wproj), nullptr, 32, 32, wproj, nullptr, true, sb);
      }
      if (v.pf & 8) {  // pfq had wq_a + wkv first, then as much of these as its budget left
        const size_t ix = idx_src(L) ? (size_t)kIdxQ * kQLora : 0;
        const size_t done = (v.pf & 1) && L > 0 ? (budget > qakv ? budget - qakv : 0) : 0;
        if (done < ((size_t)kQ * kQLora + ix) * (sc32 + 1) / sc32)
          l2pf({{d.wqb, (size_t)kQ * kQLora}, {d.sqb, (size_t)kQ * kQLora / sc32}, {d.wix, ix}, {d.six, ix / sc32}},
               done);
      }
      window(2, L, o.ag_us);
      xwin(2, L);
      if (v.mixes) join(sa);  // the partials ride the all-gather
      standin(k_allgather_standin, 4, 256, o.ag_us);
      if (v.mixes) {  // mix_ranks
        fork(sa);
        attn::hc_finish(hpart, R, hc_sc, hc_base, pre, post, comb, sa);
      }
      const dsk::Fp8Seg qseg[2] = {{d.wqb, d.sqb, q, kQ}, {d.wix, d.six, iq, kIdxQ}};
      dsk::gemv_fp8_norm(qa, d.n_q, 1e-20f, R, qseg, idx_src(L) ? 2 : 1, kQLora, v.cqb, st);
      if (v.pf & 4) l2pf({{d.woa, (size_t)kOPart * kWoaK}, {d.soa, (size_t)kOPart * kWoaK / sc32}});
      window(3, L, o.attn_us + o.comb_us + (idx_src(L) ? o.idx_us : 0));
      xwin(3, L);
      if (idx_src(L)) {
        join(sb);
        standin(k_index_fused_standin, 40, 256, o.idx_us);
      }
      standin(k_attn_kernel_standin, 2 + (L < 2 ? 0 : 8), 256, o.attn_us);
      standin(k_attn_combine_standin, R, 256, o.comb_us);
      dsk::gemv_woa(ao, R, d.woa, d.soa, kOPart, kWoaK, kOGroup, nullptr, v.cwoa, false, st, oq, os);
      dsk::gemv_fp8(oq, os, R, d.wob, d.sob, kD, kOPart, part, v.cob, st);
      if (v.pf & 2) l2pf({{d.gate, (size_t)kExperts * kD * 2}});
      window(4, L, ar_us);
      xwin(4, L);
      if (v.mixes) join(sa);
      standin(k_allreduce_hc_standin, 40, 256, ar_us);
      // ---- FFN
      project(2 * L + 1);
      if (v.mixes && v.standins)
        k_mix_allreduce_standin<<<1, 256, 0, sa>>>((uint32_t)(mix_us * 1000));
      if (v.gate_split) {
        dsk::gemv_bf16_norm(yraw, d.n_ffn, 1e-20f, R, d.gate, kGatePart, kD, gl, nullptr, v.cgate, st, yq, ys);
        standin(k_allgather_standin, 4, 256, o.ag_us);
      } else {
        dsk::gemv_bf16_norm(yraw, d.n_ffn, 1e-20f, R, d.gate, kExperts, kD, gl, nullptr, v.cgate, st, yq, ys);
      }
      dsk::MoeCfg mc{};
      mc.down = R == 1 ? 2 : 4;
      mc.dyn = v.moe_dyn;
      if (v.moe_up >= 0) mc.up = mc.up16 = v.moe_up;
      if (v.moe_down >= 0) mc.down = mc.down16 = v.moe_down;
      const dsk::MoeWeights& mw = o.scales >= 0 && !v.raw ? MWc[L % kCopies] : MW[L % kCopies];
      dsk::moe_routed(yq, ys, R, lsel + (size_t)L * kMR * kExperts, bias, kExperts, 1.5f, mw, ws, ymoe, mc, st);
      if ((v.pf & 1) && L + 1 < kL) {
        const Layer& e = Ls[(L + 1) % kCopies];
        const size_t ix = idx_src(L + 1) ? (size_t)kIdxQ * kQLora : 0;
        l2pf({{e.wqa, (size_t)kQaPart * kD}, {e.sqa, (size_t)kQaPart * kD / sc32}, {e.wkv, (size_t)kKvPart * kD},
              {e.skv, (size_t)kKvPart * kD / sc32}, {e.wqb, (size_t)kQ * kQLora}, {e.sqb, (size_t)kQ * kQLora / sc32},
              {e.wix, ix}, {e.six, ix / sc32}});
      }
      window(1, L + 1, arf_us);
      xwin(1, L + 1);
      if (v.mixes) join(sa);
      standin(k_allreduce_hc_standin, 40, 288, arf_us);
    }
    if (v.pf || !v.pf_w.empty() || v.pf_x) join(sc);
    if (v.head_sep) {
      dsk::rmsnorm(yraw, n_out, R, kD, 1e-20f, y, nullptr, nullptr, st);
      dsk::gemv_bf16(y, R, head, kVocabPart, kD, logits, nullptr, v.chead, false, st);
    } else if (headx && !v.head_raw) {
      dsk::gemv_bf16x_norm(yraw, n_out, 1e-20f, v.head1 ? 1 : R, headx, kVocabPart, kD, logits, nullptr, v.chead, st);
    } else {
      dsk::gemv_bf16_norm(yraw, n_out, 1e-20f, v.head1 ? 1 : R, head, kVocabPart, kD, logits, nullptr, v.chead, st);
    }
    standin(k_allgather_standin, 4, 256, o.argmax_us);
  };

  std::vector<Var> vars;
  for (auto& n : o.variants) vars.push_back(make_var(n));
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  std::vector<cudaGraphExec_t> ex;
  for (auto& v : vars) {
    step(v, st);  // eager once: kernels set their attributes outside capture
    CUDA_CHECK(cudaStreamSynchronize(st));
    cudaGraph_t g;
    cudaGraphExec_t e;
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    step(v, st);
    CUDA_CHECK(cudaStreamEndCapture(st, &g));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&e, g, cudaGraphInstantiateFlagUseNodePriority));
    CUDA_CHECK(cudaGraphDestroy(g));
    ex.push_back(e);
  }
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<std::vector<float>> ts(vars.size());
  double usum = 0;  // the experts of the timed rounds' windows (rounds 2 on: chainprof's steps)
  for (int r = 0; r < o.rounds + 2; ++r) {
    if (nwin > 1)
      CUDA_CHECK(cudaMemcpyAsync(lsel, lwin + (size_t)(r % nwin) * lsel_n, lsel_n * 4, cudaMemcpyDeviceToDevice, st));
    if (r >= 2) usum += wu[r % nwin];
    for (size_t i = 0; i < ex.size(); ++i) {
      CUDA_CHECK(cudaEventRecord(a, st));
      CUDA_CHECK(cudaGraphLaunch(ex[i], st));
      CUDA_CHECK(cudaEventRecord(b, st));
      CUDA_CHECK(cudaEventSynchronize(b));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      if (r >= 2) ts[i].push_back(ms);
    }
  }
  const double nu_mean = usum / ((double)o.rounds * kL);
  char ul[32];  // dspark/ref/tree_costs.py reads "rows R, U experts/layer, base step median T" (U an integer)
  if (o.routes.empty()) std::snprintf(ul, sizeof ul, "%2d", nu);
  else std::snprintf(ul, sizeof ul, "%5.2f", nu_mean);
  for (size_t i = 0; i < vars.size(); ++i) {
    auto& t = ts[i];
    double mean = 0;
    for (float x : t) mean += x / t.size();
    std::sort(t.begin(), t.end());
    std::printf("rows %d, %s experts/layer, %-14s step median %7.3f ms (min %7.3f, max %7.3f), mean %7.3f\n", R, ul,
                vars[i].name.c_str(), t[t.size() / 2], t.front(), t.back(), mean);
  }
  // the bytes a step's weight sites read, for chainprof.py --meta: a routed expert's in moe_up and moe_down, the
  // head's
  for (const Var& v : vars) {
    const int c = o.scales >= 0 && !v.raw;
    std::printf("chainprof %s: nu %.4f up_b %.1f down_b %.1f head_b %zu\n", v.name.c_str(), nu_mean, up_b[c], down_b[c],
                headx && !v.head_raw && !v.head_sep ? head_x_b : head_raw_b);
  }
}

}  // namespace
}  // namespace dsk

int main(int argc, char** argv) {
  using namespace dsk;
  Opts o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return i + 1 < argc ? argv[++i] : (die("bench_chain: %s needs a value", a.c_str()), ""); };
    if (a == "--rows") o.rows = std::atoi(next());
    else if (a == "--experts") o.experts = std::atoi(next());
    else if (a == "--routes") o.routes = next();
    else if (a == "--windows") o.windows = std::atoi(next());
    else if (a == "--scales") o.scales = std::atoi(next());
    else if (a == "--head") o.head = next();
    else if (a == "--rounds") o.rounds = std::atoi(next());
    else if (a == "--ar-us") o.ar_us = std::atof(next());
    else if (a == "--arf-us") o.arf_us = std::atof(next());
    else if (a == "--ag-us") o.ag_us = std::atof(next());
    else if (a == "--mix-us") o.mix_us = std::atof(next());
    else if (a == "--prefer-shared") CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));  // EXPERIMENT
    else if (a == "--variants") {
      o.variants.clear();
      std::string s = next();
      for (size_t p = 0; p <= s.size();) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) o.variants.push_back(s.substr(p, q - p));
        p = q + 1;
      }
    } else die("bench_chain: unknown argument %s", a.c_str());
  }
  if (o.rows < 1 || o.rows > dsk::kMaxRows) die("bench_chain: rows %d", o.rows);
  if (o.windows < 1 || (!o.routes.empty() && o.experts)) die("bench_chain: --windows %d, or --routes with --experts", o.windows);
  run(o);
  return 0;
}
