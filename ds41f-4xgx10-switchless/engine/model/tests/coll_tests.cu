// The runner's fused collectives (model/fused.cu) against the separate kernels they replace, and their
// timing.
//   coll_tests [--fleet FILE] [--mode proxy|ring] [--rows 1,6,8] [--n 80] [--rounds 21] [--proxy-cpu C] test|bench ...
// test: one box, the ring in loop mode (the proxy answers every post itself); before each collective the
// other three ranks' contributions are written where theirs would arrive, and everything runs once as
// each of the four ranks:
//   - allreduce_hc (bf16 and fp32 partials, with and without yraw / sub) against comm::allreduce_bf16 /
//     allreduce_f32 (+ f32_to_bf16), attn::hc_post and attn::hc_pre, bit for bit, and against the kernel
//     it replaced (5 blocks of 128 threads per row, kept below as legacy::allreduce_hc);
//   - allreduce_hc_mix against the same chain with the mixes from attn::hc_finish of the four ranks'
//     chunk sums added in rank order, bit for bit; and against the chain before it, the mixes of the
//     whole stream's projection (attn::hc_partial: 40 chunks in order), with the differences printed;
//   - allgather_attn_mix (the attention's mixes riding its all-gather) against allgather_attn, and
//     mix_ranks of the gathered sums against the same hc_finish of the rank-order sums, bit for bit;
//   - mix_allreduce (the mixes by a collective of their own) against the same, bit for bit;
//   - every output is the same bits whichever rank computes it.
// bench: graphs of --n collectives back to back, bf16 and fp32 partials alternating (attention's and
// the FFN's), µs per collective, the variants interleaved, median of --rounds. With --mode ring on all
// four boxes (engine/comm/run4.sh build/model/coll_tests --mode ring bench) every rank's outputs are
// also hashed and compared.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "attn/launch.h"
#include "comm/collectives.h"
#include "comm/ring.h"
#include "comm/ring_dev.cuh"
#include "fill.cuh"
#include "model/config.h"
#include "model/ops.h"
#include "testutil.h"

using namespace attn_test;
using model::kDim;
using model::kHc;
using model::kHcDim;
using model::kRanks;
using bf16 = __nv_bfloat16;

namespace {

// ---- the kernel allreduce_hc replaced (fused.cu at c10c867), for comparison -----------------------

namespace legacy {

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

constexpr int kArChunks = kDim / 8;  // 8 columns per thread
constexpr int kArThreads = 128;
constexpr int kArBlocks = kArChunks / kArThreads;  // per row

template <bool kF32>
__global__ void __launch_bounds__(kArThreads) k_allreduce_hc(comm::DevChannel ch, const uint8_t* part, int R,
                                                             const uint16_t* res, const float* post, const float* comb,
                                                             const float* pre, uint16_t* stream, uint16_t* yraw,
                                                             uint16_t* sub) {
  constexpr int CB = kF32 ? 32 : 16;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint64_t s = comm::next_seq(ch);
  const int r = blockIdx.x / kArBlocks, c = (blockIdx.x % kArBlocks) * kArThreads + threadIdx.x;
  const size_t off = ((size_t)r * kArChunks + c) * CB;
  uint8_t* sb = comm::send_buf(ch, s);
  *reinterpret_cast<uint4*>(sb + off) = *reinterpret_cast<const uint4*>(part + off);
  if (kF32) *reinterpret_cast<uint4*>(sb + off + 16) = *reinterpret_cast<const uint4*>(part + off + 16);
  comm::post_grid(ch, s, (uint32_t)(R * kDim * (kF32 ? 4 : 2)));
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  comm::wait_all(ch, s);
  float v[8];
  if constexpr (kF32) {
    const float4 a = comm::sum4_f32(ch, s, off), b = comm::sum4_f32(ch, s, off + 16);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
  } else {
    comm::sum8_bf16(ch, s, off, v);
  }
  comm::finish_grid(ch, s);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = bf2f(f2bf(v[j]));
  const int d0 = 8 * c;
  if (sub) {
    uint4 o;
    uint32_t* ow = &o.x;
#pragma unroll
    for (int j = 0; j < 4; ++j) ow[j] = (uint32_t)f2bf(v[2 * j]) | ((uint32_t)f2bf(v[2 * j + 1]) << 16);
    *reinterpret_cast<uint4*>(sub + (size_t)r * kDim + d0) = o;
  }
  const uint16_t* rr = res + (size_t)r * kHcDim;
  uint4 xin[kHc];
#pragma unroll
  for (int i = 0; i < kHc; ++i) xin[i] = *reinterpret_cast<const uint4*>(rr + (size_t)i * kDim + d0);
  const float* cb = comb + r * 16;
  const float* po = post + r * 4;
  float y[kHc][8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float x[kHc];
#pragma unroll
    for (int i = 0; i < kHc; ++i) x[i] = bf2f(reinterpret_cast<const uint16_t*>(&xin[i])[j]);
#pragma unroll
    for (int i = 0; i < kHc; ++i) {
      float a = __fadd_rn(__fmul_rn(cb[i], x[0]), __fmul_rn(cb[4 + i], x[1]));
      a = __fadd_rn(a, __fmul_rn(cb[8 + i], x[2]));
      a = __fadd_rn(a, __fmul_rn(cb[12 + i], x[3]));
      y[i][j] = bf2f(f2bf(__fadd_rn(__fmul_rn(po[i], v[j]), a)));
    }
  }
  uint16_t* so = stream + (size_t)r * kHcDim;
#pragma unroll
  for (int i = 0; i < kHc; ++i) {
    uint4 o;
    uint32_t* ow = &o.x;
#pragma unroll
    for (int j = 0; j < 4; ++j) ow[j] = (uint32_t)f2bf(y[i][2 * j]) | ((uint32_t)f2bf(y[i][2 * j + 1]) << 16);
    *reinterpret_cast<uint4*>(so + (size_t)i * kDim + d0) = o;
  }
  if (!yraw) return;
  const float* pr = pre + r * 4;
  uint4 o;
  uint32_t* ow = &o.x;
#pragma unroll
  for (int j = 0; j < 8; j += 2) {
    float a[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float t = __fadd_rn(__fmul_rn(pr[0], y[0][j + h]), __fmul_rn(pr[1], y[1][j + h]));
      t = __fadd_rn(t, __fmul_rn(pr[2], y[2][j + h]));
      a[h] = __fadd_rn(t, __fmul_rn(pr[3], y[3][j + h]));
    }
    ow[j / 2] = (uint32_t)f2bf(a[0]) | ((uint32_t)f2bf(a[1]) << 16);
  }
  *reinterpret_cast<uint4*>(yraw + (size_t)r * kDim + d0) = o;
}

void allreduce_hc(const comm::DevChannel& ch, const void* part, bool f32, int R, const uint16_t* res, const float* post,
                  const float* comb, const float* pre, uint16_t* stream, uint16_t* yraw, uint16_t* sub, cudaStream_t s) {
  const uint8_t* p = static_cast<const uint8_t*>(part);
  if (f32)
    attn::launch(k_allreduce_hc<true>, dim3(kArBlocks * R), dim3(kArThreads), 0, s, true, ch, p, R, res, post, comb,
                 pre, stream, yraw, sub);
  else
    attn::launch(k_allreduce_hc<false>, dim3(kArBlocks * R), dim3(kArThreads), 0, s, true, ch, p, R, res, post, comb,
                 pre, stream, yraw, sub);
}

}  // namespace legacy

// ---- harness ----------------------------------------------------------------------------------------

constexpr int kMixStride = 32;  // fused.cu's mix block: floats per row
const int kChunks = attn::hc_chunks(0, kDim);  // hc_project chunks of one copy

cudaStream_t g_s;
int g_checks = 0, g_fail = 0;

void sync() { CUDA_CHECK(cudaStreamSynchronize(g_s)); }
template <class T>
std::vector<T> host(const Dev<T>& d) {
  sync();
  return d.host();
}
bf16* B(uint16_t* p) { return reinterpret_cast<bf16*>(p); }

template <class T>
bool same(const std::string& what, const std::vector<T>& a, const std::vector<T>& b) {
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(T))) {
      if (!bad) first = i;
      ++bad;
    }
  ++g_checks;
  if (bad || a.size() != b.size()) {
    ++g_fail;
    std::printf("  %s differs in %zu of %zu (first at %zu)\n", what.c_str(), bad, a.size(), first);
  }
  return !bad;
}
template <class T>
bool same(const std::string& what, const Dev<T>& a, const Dev<T>& b) {
  return same(what, host(a), host(b));
}

// The inputs of one row count, the same on every rank (and every box: the fills are seeded in order).
struct Inputs {
  int R;
  Dev<uint16_t> x;                    // the stream, the all-reduce's res
  Dev<uint16_t> pb[kRanks];           // each rank's bf16 partial sums [R][5120]
  Dev<float> pf[kRanks];              // ... fp32
  Dev<float> post, comb, pre;         // mixes, as allreduce_hc's inputs
  Dev<float> fn, scale, base;         // hc_fn [24][20480], hc_scale, hc_base
  Dev<float> chunks[kRanks];          // hc_project of copy k [kChunks][R][25]
  std::vector<float> psum[kRanks];    // [R][32]: rank k's chunk sums, the mix block it sends
  Dev<float> msum;                    // [R][25]: the four added in rank order
  explicit Inputs(int rows) : R(rows) {
    x = Dev<uint16_t>((size_t)R * kHcDim);
    fill::bf16(x, 1.f);
    for (int k = 0; k < kRanks; ++k) {
      pb[k] = Dev<uint16_t>((size_t)R * kDim);
      pf[k] = Dev<float>((size_t)R * kDim);
      fill::bf16(pb[k], 0.5f);
      fill::f32(pf[k], 0.5f);
    }
    post = Dev<float>(R * 4), comb = Dev<float>(R * 16), pre = Dev<float>(R * 4);
    fill::f32_range(post, 0.f, 2.f);
    fill::f32_range(comb, 0.f, 1.f);
    fill::f32_range(pre, 0.f, 1.f);
    fn = Dev<float>((size_t)attn::kMix * kHcDim), scale = Dev<float>(3), base = Dev<float>(attn::kMix);
    fill::f32(fn, 0.01f);
    fill::f32_range(scale, 0.5f, 1.5f);
    fill::f32(base, 1.f);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> m((size_t)R * 25);
    for (int k = 0; k < kRanks; ++k) {
      chunks[k] = Dev<float>((size_t)kChunks * R * 25);
      attn::hc_project(x.p, R, fn.p, k * kDim, (k + 1) * kDim, chunks[k].p, false, g_s);
      const std::vector<float> h = host(chunks[k]);
      psum[k].assign((size_t)R * kMixStride, 0.f);
      for (int r = 0; r < R; ++r)
        for (int i = 0; i < 25; ++i) {
          float s = 0.f;
          for (int c = 0; c < kChunks; ++c) s += h[((size_t)c * R + r) * 25 + i];
          psum[k][r * kMixStride + i] = s;
          m[r * 25 + i] = k ? m[r * 25 + i] + s : s;
        }
    }
    msum = Dev<float>(m.data(), m.size());
    for (int k = 0; k < kRanks; ++k) {  // any bytes: the all-gather only copies them
      pack[k] = Dev<uint8_t>(model::ops::attn_pack_bytes(R, 2));
      fill::u8(pack[k], 0xFF, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  Dev<uint8_t> pack[kRanks];  // each rank's attention pack (ops::attn_pack_bytes(R, 2))
  // what rank k sends to the attention all-gather: its pack, then (mixes) its mix block
  std::vector<uint8_t> attn_contribution(int k, int ncomp, bool mixes) const {
    std::vector<uint8_t> c(model::ops::attn_pack_bytes(R, ncomp));
    CUDA_CHECK(cudaMemcpy(c.data(), pack[k].p, c.size(), cudaMemcpyDeviceToHost));
    if (mixes) {
      const uint8_t* p = reinterpret_cast<const uint8_t*>(psum[k].data());
      c.insert(c.end(), p, p + psum[k].size() * 4);
    }
    return c;
  }
  size_t part_bytes(bool f32) const { return (size_t)R * kDim * (f32 ? 4 : 2); }
  const void* part(int k, bool f32) const { return f32 ? (const void*)pf[k].p : (const void*)pb[k].p; }
  // what rank k sends: its partial sums, then (mixes) its mix block
  std::vector<uint8_t> contribution(int k, bool f32, bool mixes) const {
    std::vector<uint8_t> c(part_bytes(f32));
    CUDA_CHECK(cudaMemcpy(c.data(), part(k, f32), c.size(), cudaMemcpyDeviceToHost));
    if (mixes) {
      const uint8_t* p = reinterpret_cast<const uint8_t*>(psum[k].data());
      c.insert(c.end(), p, p + psum[k].size() * 4);
    }
    return c;
  }
};

// Loop mode: the other ranks' contributions to the next collective, written where theirs would arrive.
void prime(const comm::Ring& ring, const std::function<std::vector<uint8_t>(int)>& contribution) {
  const comm::DevChannel& ch = ring.dev();
  sync();
  unsigned long long seq = 0;
  CUDA_CHECK(cudaMemcpy(&seq, ch.seq, sizeof seq, cudaMemcpyDeviceToHost));
  const int slot = (int)((seq + 1) % comm::kSlots);
  for (int k = 0; k < kRanks; ++k)
    if (k != ch.rank) {
      const std::vector<uint8_t> c = contribution(k);
      std::memcpy(ch.src[slot][k], c.data(), c.size());
    }
}
void prime(const comm::Ring& ring, const Inputs& in, bool f32, bool mixes) {
  prime(ring, [&](int k) { return in.contribution(k, f32, mixes); });
}

struct Outs {
  Dev<uint16_t> stream, yraw, sub;
  Dev<float> pre, post, comb;
  explicit Outs(int R)
      : stream((size_t)R * kHcDim), yraw((size_t)R * kDim), sub((size_t)R * kDim), pre(R * 4), post(R * 4),
        comb(R * 16) {}
};

struct AttnOut {  // the attention all-gather's outputs
  Dev<uint16_t> qa, kv;
  Dev<float> ckv, csc;
  explicit AttnOut(int R)
      : qa((size_t)R * model::kQLora), kv((size_t)R * model::kHead), ckv((size_t)R * model::kHead),
        csc((size_t)R * model::kHead) {}
};

// The separate kernels: the standalone all-reduce (rounded to bf16), hc_post, hc_pre without the norm.
void chain(const comm::Ring& ring, const Inputs& in, bool f32, const float* post, const float* comb, const float* pre,
           Outs& o) {
  const int R = in.R, rank = ring.rank();
  prime(ring, in, f32, false);
  if (f32) {
    Dev<float> s((size_t)R * kDim);
    comm::allreduce_f32(ring.dev(), in.pf[rank].p, s.p, (size_t)R * kDim, g_s);
    model::ops::f32_to_bf16(s.p, B(o.sub.p), R * kDim, g_s);
    sync();
  } else {
    comm::allreduce_bf16(ring.dev(), B(in.pb[rank].p), B(o.sub.p), (size_t)R * kDim, g_s);
  }
  attn::hc_post(o.sub.p, in.x.p, post, comb, R, o.stream.p, g_s);
  attn::hc_pre(o.stream.p, pre, R, nullptr, o.yraw.p, nullptr, g_s);
}

struct Diff {
  double mix = 0;           // max |a - b| over the mixes
  size_t stream = 0, yraw = 0;  // bf16 elements that differ
  int ulp = 0;              // the largest difference in units of the last place
};
Diff diff(const Outs& a, const Outs& b) {
  Diff d;
  for (auto p : {&Outs::pre, &Outs::post, &Outs::comb}) {
    const auto x = host(a.*p), y = host(b.*p);
    for (size_t i = 0; i < x.size(); ++i) d.mix = std::max(d.mix, (double)std::fabs(x[i] - y[i]));
  }
  for (auto p : {&Outs::stream, &Outs::yraw}) {
    const auto x = host(a.*p), y = host(b.*p);
    size_t n = 0;
    for (size_t i = 0; i < x.size(); ++i)
      if (x[i] != y[i]) ++n, d.ulp = std::max(d.ulp, bf_ulp(x[i], y[i]));
    (p == &Outs::stream ? d.stream : d.yraw) = n;
  }
  return d;
}

// Everything one rank computes for one row count, for the comparison across ranks.
struct RankOut {
  std::vector<std::vector<uint8_t>> bufs;
  void add(const Outs& o) {
    for (auto p : {&Outs::stream, &Outs::yraw, &Outs::sub}) {
      const auto h = host(o.*p);
      bufs.emplace_back(reinterpret_cast<const uint8_t*>(h.data()), reinterpret_cast<const uint8_t*>(h.data() + h.size()));
    }
    for (auto p : {&Outs::pre, &Outs::post, &Outs::comb}) {
      const auto h = host(o.*p);
      bufs.emplace_back(reinterpret_cast<const uint8_t*>(h.data()), reinterpret_cast<const uint8_t*>(h.data() + h.size()));
    }
  }
};

RankOut test_rank(const comm::Ring& ring, const Inputs& in) {
  const int R = in.R, rank = ring.rank();
  const comm::DevChannel& ch = ring.dev();
  const std::string tag = "R=" + std::to_string(R) + " rank " + std::to_string(rank) + ": ";
  RankOut ro;
  for (bool f32 : {false, true}) {
    const std::string t = tag + (f32 ? "fp32 " : "bf16 ");
    // allreduce_hc: the mixes are inputs
    Outs ref(R), got(R), old(R);
    chain(ring, in, f32, in.post.p, in.comb.p, in.pre.p, ref);
    prime(ring, in, f32, false);
    model::ops::allreduce_hc(ch, in.part(rank, f32), f32, R, B(in.x.p), in.post.p, in.comb.p, in.pre.p,
                             B(got.stream.p), B(got.yraw.p), B(got.sub.p), g_s);
    same(t + "allreduce_hc stream", ref.stream, got.stream);
    same(t + "allreduce_hc yraw", ref.yraw, got.yraw);
    same(t + "allreduce_hc sub", ref.sub, got.sub);
    prime(ring, in, f32, false);
    legacy::allreduce_hc(ch, in.part(rank, f32), f32, R, in.x.p, in.post.p, in.comb.p, in.pre.p, old.stream.p,
                         old.yraw.p, old.sub.p, g_s);
    same(t + "legacy stream", old.stream, got.stream);
    same(t + "legacy yraw", old.yraw, got.yraw);
    ro.add(got);
    // without yraw and sub: those buffers are left alone
    Outs bare(R);
    CUDA_CHECK(cudaMemsetAsync(bare.yraw.p, 0x5a, bare.yraw.n * 2, g_s));
    const auto before = host(bare.yraw);
    prime(ring, in, f32, false);
    model::ops::allreduce_hc(ch, in.part(rank, f32), f32, R, B(in.x.p), in.post.p, in.comb.p, in.pre.p,
                             B(bare.stream.p), nullptr, nullptr, g_s);
    same(t + "allreduce_hc stream (no yraw)", ref.stream, bare.stream);
    same(t + "allreduce_hc yraw untouched", before, host(bare.yraw));
    // allreduce_hc_mix: the mixes ride along
    Outs mref(R), mgot(R), before_chain(R);
    attn::hc_finish(in.msum.p, R, in.scale.p, in.base.p, mref.pre.p, mref.post.p, mref.comb.p, g_s);
    chain(ring, in, f32, mref.post.p, mref.comb.p, mref.pre.p, mref);
    prime(ring, in, f32, true);
    model::ops::allreduce_hc_mix(ch, in.part(rank, f32), f32, R, in.chunks[rank].p, kChunks, in.scale.p, in.base.p,
                                 B(in.x.p), mgot.pre.p, mgot.post.p, mgot.comb.p, B(mgot.stream.p), B(mgot.yraw.p),
                                 B(mgot.sub.p), g_s);
    same(t + "mix pre", mref.pre, mgot.pre);
    same(t + "mix post", mref.post, mgot.post);
    same(t + "mix comb", mref.comb, mgot.comb);
    same(t + "mix stream", mref.stream, mgot.stream);
    same(t + "mix yraw", mref.yraw, mgot.yraw);
    same(t + "mix sub", mref.sub, mgot.sub);
    ro.add(mgot);
    // the chain before: the whole stream projected by one rank (40 chunks summed in order)
    {
      Dev<float> ch40((size_t)attn::hc_chunks(0, kHcDim) * R * 25), p40((size_t)R * 25);
      attn::hc_partial(in.x.p, R, in.fn.p, 0, kHcDim, ch40.p, p40.p, g_s);
      attn::hc_finish(p40.p, R, in.scale.p, in.base.p, before_chain.pre.p, before_chain.post.p, before_chain.comb.p,
                      g_s);
      chain(ring, in, f32, before_chain.post.p, before_chain.comb.p, before_chain.pre.p, before_chain);
      if (rank == 0) {
        const Diff d = diff(mgot, before_chain);
        std::printf("  %sagainst the 40-chunk order: mixes max |d| %.3g, stream %zu of %d and yraw %zu of %d "
                    "bf16 differ (max %d ulp)\n",
                    t.c_str(), d.mix, d.stream, R * kHcDim, d.yraw, R * kDim, d.ulp);
      }
    }
  }
  // allgather_attn_mix + mix_ranks: the attention's mixes riding its all-gather instead, against
  // allgather_attn and hc_finish of the rank-order sums (allreduce_hc_mix's mixes, checked above)
  Outs mref(R);
  attn::hc_finish(in.msum.p, R, in.scale.p, in.base.p, mref.pre.p, mref.post.p, mref.comb.p, g_s);
  for (int ncomp : {0, 2}) {
    const std::string t = tag + "attention all-gather, ncomp " + std::to_string(ncomp) + ": ";
    AttnOut ref(R), got(R);
    prime(ring, [&](int k) { return in.attn_contribution(k, ncomp, false); });
    model::ops::allgather_attn(ch, in.pack[rank].p, R, ncomp, B(ref.qa.p), B(ref.kv.p), ref.ckv.p, ref.csc.p, g_s);
    Dev<float> mix((size_t)kRanks * R * kMixStride);
    prime(ring, [&](int k) { return in.attn_contribution(k, ncomp, true); });
    model::ops::allgather_attn_mix(ch, in.pack[rank].p, R, ncomp, in.chunks[rank].p, kChunks, B(got.qa.p),
                                   B(got.kv.p), got.ckv.p, got.csc.p, mix.p, g_s);
    Outs m(R);
    model::ops::mix_ranks(mix.p, R, in.scale.p, in.base.p, m.pre.p, m.post.p, m.comb.p, g_s);
    same(t + "qa", ref.qa, got.qa);
    same(t + "kv", ref.kv, got.kv);
    same(t + "ckv", ref.ckv, got.ckv);
    same(t + "csc", ref.csc, got.csc);
    const std::vector<float> h = host(mix);
    for (int k = 0; k < kRanks; ++k)
      same(t + "rank " + std::to_string(k) + "'s mix partials",
           in.psum[k], std::vector<float>(h.begin() + (size_t)k * R * kMixStride, h.begin() + (size_t)(k + 1) * R * kMixStride));
    same(t + "pre", mref.pre, m.pre);
    same(t + "post", mref.post, m.post);
    same(t + "comb", mref.comb, m.comb);
  }
  {  // mix_allreduce: the mixes by a collective of their own (the FFN's, on the side stream)
    const std::string t = tag + "mix_allreduce: ";
    prime(ring, [&](int k) {
      const uint8_t* p = reinterpret_cast<const uint8_t*>(in.psum[k].data());
      return std::vector<uint8_t>(p, p + in.psum[k].size() * 4);
    });
    Outs m(R);
    model::ops::mix_allreduce(ch, in.chunks[rank].p, kChunks, R, in.scale.p, in.base.p, m.pre.p, m.post.p, m.comb.p,
                              g_s);
    same(t + "pre", mref.pre, m.pre);
    same(t + "post", mref.post, m.post);
    same(t + "comb", mref.comb, m.comb);
  }
  return ro;
}

void test(const std::vector<int>& rows, comm::Options o) {
  for (int R : rows) {
    const int f0 = g_fail, c0 = g_checks;
    Inputs in(R);
    std::vector<RankOut> outs;
    for (int rank = 0; rank < kRanks; ++rank) {
      o.rank = rank;
      comm::Ring ring(o);
      outs.push_back(test_rank(ring, in));
    }
    for (int rank = 1; rank < kRanks; ++rank)
      for (size_t b = 0; b < outs[0].bufs.size(); ++b)
        same("R=" + std::to_string(R) + " rank " + std::to_string(rank) + " against rank 0, output " +
                 std::to_string(b),
             outs[rank].bufs[b], outs[0].bufs[b]);
    std::printf("rows %d: %d checks, %s\n", R, g_checks - c0, g_fail == f0 ? "all identical" : "DIFFERENT");
  }
}

// ---- bench --------------------------------------------------------------------------------------------

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

void bench(comm::Ring& ring, const std::vector<int>& rows, int n, int rounds, bool loop) {
  const comm::DevChannel& ch = ring.dev();
  const int rank = ring.rank();
  if (loop)  // nobody writes the other ranks' buffers: zeros keep the mixes finite
    for (int p = 0; p < comm::kSlots; ++p)
      for (int k = 0; k < kRanks; ++k)
        if (k != rank) std::memset(ch.src[p][k], 0, ch.cap);
  for (int R : rows) {
    Inputs in(R);
    Outs o(R);
    Dev<float> sumf((size_t)R * kDim), amix((size_t)kRanks * R * kMixStride);
    AttnOut ao(R);
    struct V {
      const char* name;
      std::function<void(bool)> run;  // one collective: f32 = the FFN's partials
    };
    const std::vector<V> vs = {
        {"comm::allreduce (no hc)",
         [&](bool f32) {
           if (f32) comm::allreduce_f32(ch, in.pf[rank].p, sumf.p, (size_t)R * kDim, g_s);
           else comm::allreduce_bf16(ch, B(in.pb[rank].p), B(o.sub.p), (size_t)R * kDim, g_s);
         }},
        {"legacy allreduce_hc (5 blocks/row)",
         [&](bool f32) {
           legacy::allreduce_hc(ch, in.part(rank, f32), f32, R, in.x.p, in.post.p, in.comb.p, in.pre.p, o.stream.p,
                                o.yraw.p, nullptr, g_s);
         }},
        {"allreduce_hc (40 blocks)",
         [&](bool f32) {
           model::ops::allreduce_hc(ch, in.part(rank, f32), f32, R, B(in.x.p), in.post.p, in.comb.p, in.pre.p,
                                    B(o.stream.p), B(o.yraw.p), nullptr, g_s);
         }},
        {"allreduce_hc_mix (40 blocks)",
         [&](bool f32) {
           model::ops::allreduce_hc_mix(ch, in.part(rank, f32), f32, R, in.chunks[rank].p, kChunks, in.scale.p,
                                        in.base.p, B(in.x.p), o.pre.p, o.post.p, o.comb.p, B(o.stream.p),
                                        B(o.yraw.p), nullptr, g_s);
         }},
        {"allgather_attn",
         [&](bool) {
           model::ops::allgather_attn(ch, in.pack[rank].p, R, 0, B(ao.qa.p), B(ao.kv.p), ao.ckv.p, ao.csc.p, g_s);
         }},
        {"allgather_attn_mix",
         [&](bool) {
           model::ops::allgather_attn_mix(ch, in.pack[rank].p, R, 0, in.chunks[rank].p, kChunks, B(ao.qa.p),
                                          B(ao.kv.p), ao.ckv.p, ao.csc.p, amix.p, g_s);
         }},
        {"mix_allreduce (1 block)",
         [&](bool) {
           model::ops::mix_allreduce(ch, in.chunks[rank].p, kChunks, R, in.scale.p, in.base.p, o.pre.p, o.post.p,
                                     o.comb.p, g_s);
         }},
    };
    std::vector<cudaGraphExec_t> ge(vs.size());
    for (size_t v = 0; v < vs.size(); ++v) {
      cudaGraph_t g;
      CUDA_CHECK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < n; ++i) vs[v].run(i % 2 == 1);
      CUDA_CHECK(cudaStreamEndCapture(g_s, &g));
      CUDA_CHECK(cudaGraphInstantiate(&ge[v], g, 0));
      CUDA_CHECK(cudaGraphDestroy(g));
    }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<std::vector<float>> us(vs.size());
    std::vector<uint64_t> hash(vs.size());
    ring.barrier();
    for (int it = 0; it < rounds + 1; ++it)
      for (size_t v = 0; v < vs.size(); ++v) {
        CUDA_CHECK(cudaEventRecord(e0, g_s));
        CUDA_CHECK(cudaGraphLaunch(ge[v], g_s));
        CUDA_CHECK(cudaEventRecord(e1, g_s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (it) us[v].push_back(ms * 1000.f / n);
        if (it == rounds) {  // the last replay's outputs (the last collective is an FFN one)
          const auto a = host(o.stream), b = host(o.yraw), c = host(o.sub);
          const auto d = host(o.pre), e = host(o.post), f = host(o.comb), g = host(sumf);
          uint64_t h = fnv(a.data(), a.size() * 2);
          h = fnv(b.data(), b.size() * 2, h);
          h = fnv(c.data(), c.size() * 2, h);
          h = fnv(d.data(), d.size() * 4, h);
          h = fnv(e.data(), e.size() * 4, h);
          h = fnv(f.data(), f.size() * 4, h);
          h = fnv(g.data(), g.size() * 4, h);
          const auto qa = host(ao.qa), kv = host(ao.kv);
          const auto mx = host(amix);
          h = fnv(qa.data(), qa.size() * 2, h);
          h = fnv(kv.data(), kv.size() * 2, h);
          hash[v] = fnv(mx.data(), mx.size() * 4, h);
        }
      }
    std::vector<uint64_t> all(vs.size() * kRanks);
    ring.allgather_host(hash.data(), all.data(), hash.size() * sizeof(uint64_t));
    for (size_t v = 0; v < vs.size(); ++v) {
      std::sort(us[v].begin(), us[v].end());
      bool agree = true;
      for (int k = 0; k < kRanks; ++k) agree = agree && all[k * vs.size() + v] == hash[v];
      std::printf("rank %d rows %d %-36s %7.2f us per collective (p10 %.2f, p90 %.2f)%s\n", rank, R, vs[v].name,
                  us[v][us[v].size() / 2], us[v][us[v].size() / 10], us[v][us[v].size() * 9 / 10],
                  loop ? "" : agree ? ", outputs identical on all ranks" : ", OUTPUTS DIFFER BETWEEN RANKS");
      if (!loop && !agree) ++g_fail;
      CUDA_CHECK(cudaGraphExecDestroy(ge[v]));
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
  }
}

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    v.push_back(std::atoi(s.substr(p, q - p).c_str()));
    p = q + 1;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  comm::Options o;
  o.mode = comm::Options::kLoopProxy;
  o.proxy_cpu = 7;  // not core 5: a running engine's proxy spins there
  std::vector<int> rows;
  int n = 80, rounds = 21;
  std::vector<std::string> what;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&] {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return std::string(argv[++i]);
    };
    if (a == "--mode") {
      const std::string m = val();
      o.mode = m == "ring" ? comm::Options::kRing : comm::Options::kLoopProxy;
      if (m == "ring") o.proxy_cpu = 5;
    } else if (a == "--rank") o.rank = std::atoi(val().c_str());
    else if (a == "--fleet") o.fleet = val();
    else if (a == "--rows") rows = parse_list(val());
    else if (a == "--n") n = std::atoi(val().c_str());
    else if (a == "--rounds") rounds = std::atoi(val().c_str());
    else if (a == "--proxy-cpu") o.proxy_cpu = std::atoi(val().c_str());
    else what.push_back(a);
  }
  if (what.empty()) what = {"test"};
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
  for (const std::string& w : what) {
    if (w == "test") {
      if (o.mode == comm::Options::kRing) die("test runs in loop mode (one box)");
      test(rows.empty() ? std::vector<int>{1, 6, 8} : rows, o);
    } else if (w == "bench") {
      comm::Ring ring(o);
      bench(ring, rows.empty() ? std::vector<int>{1, 6} : rows, n, rounds, o.mode != comm::Options::kRing);
    } else {
      die("unknown: %s", w.c_str());
    }
  }
  std::printf("%d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
