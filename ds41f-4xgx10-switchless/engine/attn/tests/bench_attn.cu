// Per-rank timing of the attention-side kernels (engine/attn) on synthetic state: all 40 layers with their
// own weights, window rings, compressed caches and index keys, as one rank at TP=4 runs them (16 heads).
// Decode: the indexer scans this rank's quarter of the positions, selects locally and merges the four
// ranks' lists (the other three parts are precomputed in setup, as the all-gather would deliver them).
// Prefill (--prefill CHUNK): the chunk's rows at the end of the context; the indexer runs this rank's
// quarter of the rows (the last, longest one) over all positions. Collectives and the FP8 projections
// (engine/kernels) are not included. Every timed replay is a CUDA graph launched after an L2 flush.
//   bench_attn [--rows R] [--splits S] [--iters N] [--prefill CHUNK] [--detail] ctx...
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "common/check.h"
#include "testutil.h"

using namespace attn_test;
using attn::kDim;

namespace {

__device__ __forceinline__ uint32_t hash32(uint64_t i, uint32_t seed) {
  uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t)seed * 0xD1B54A32D192ED03ull;
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29;
  return (uint32_t)(x >> 16);
}
__global__ void fill_u8(uint8_t* p, size_t n, uint32_t seed, uint32_t mask, uint32_t add) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (uint8_t)((hash32(i, seed) & mask) + add);
}
__global__ void fill_bf16(uint16_t* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float v = ((hash32(i, seed) & 0xFFFFFF) * (1.0f / 8388608.0f) - 1.0f) * scale;
    uint32_t b = __float_as_uint(v);
    p[i] = (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16);
  }
}
__global__ void fill_f32(float* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = ((hash32(i, seed) & 0xFFFFFF) * (1.0f / 8388608.0f) - 1.0f) * scale;
}
__global__ void empty_kernel() {}
// ascending, evenly spread selections for rows whose indices come from other ranks
__global__ void fill_topk(int* top, int rows, int kmax, const int* pos, int ratio) {
  int r = blockIdx.x;
  if (r >= rows) return;
  int n = (pos[r] + 1) / ratio, k = min(kmax, n);
  for (int i = threadIdx.x; i < kmax; i += blockDim.x) top[(size_t)r * kmax + i] = i < k ? (int)((long)i * n / k) : -1;
}

uint32_t g_seed = 1;
void rnd_u8(Dev<uint8_t>& d, uint32_t mask, uint32_t add) { fill_u8<<<1024, 256>>>(d.p, d.n, g_seed++, mask, add); }
void rnd_bf16(Dev<uint16_t>& d, float scale) { fill_bf16<<<1024, 256>>>(d.p, d.n, g_seed++, scale); }
void rnd_f32(Dev<float>& d, float scale) { fill_f32<<<1024, 256>>>(d.p, d.n, g_seed++, scale); }

constexpr int kLayers = 40, kTopk = 512, kPoolBlocks = 2048, kParts = 4;
int ratio_of(int L) { return L < 2 ? 0 : L < 20 ? 2 : 1; }
bool kv_src(int L) { return L == 2 || L == 8 || L == 14 || L == 20; }
int kv_source(int L) { return L < 2 ? -1 : L < 8 ? 2 : L < 14 ? 8 : L < 20 ? 14 : 20; }
bool idx_src(int L) { return kv_src(L) || (L >= 24 && L % 4 == 0); }
int index_source(int L) { return L < 20 ? kv_source(L) : 20 + (L - 20) / 4 * 4; }
std::string type_of(int L) {
  if (L < 2) return "window only (0-1)";
  if (L < 20) return kv_src(L) ? "r2 compressor+Full indexer (2,8,14)" : "r2 reuse (3-7,9-13,15-19)";
  if (L == 20) return "r1 compressor+Full+pool (20)";
  return idx_src(L) ? "r1 Reindex (24,28,32,36)" : "r1 reuse (21-23,25-27,..)";
}

struct Layer {
  int L = 0, ratio = 0, cap = 0, pmax = 0, kstride = 0;
  Dev<uint8_t> ring_kv, ring_sc, ckv_kv, ckv_sc, ik_k, ik_sc;
  Dev<uint16_t> qnw, kvw, hnorm[2], wkv, wgate, cnorm, wk, knorm, wproj, keys, bkeys, cand_key, topkey, pcand_key,
      poolkey;
  Dev<float> sink, fn[2], hsc[2], hbase[2], carry_kv, carry_sc;
  Dev<int> cand, top, pcand, pool;
  attn::CompressorWeights cw() const { return {ratio, wkv.p, wgate.p, cnorm.p, wk.p, knorm.p}; }
  attn::Compressed ckv() const { return {ckv_kv.p, ckv_sc.p, cap}; }
  attn::IndexKeys ik() const { return {ik_k.p, ik_sc.p, cap}; }
};

enum : unsigned { kPrep = 1, kComp = 2, kIdxQ = 4, kScore = 8, kSelect = 16, kMerge = 32, kAttn = 64, kAll = 127 };

struct Bench {
  int R, ctx, splits, iters, ir0;  // ir0: first indexer row (prefill: this rank's quarter)
  bool prefill, hc_full;
  cudaStream_t s;
  std::vector<Layer> ly;
  Dev<float2> tab_plain, tab_yarn;
  Dev<uint16_t> stream[2], y, sub, qa, qr, kvp, qpre, iqpre, out;
  Dev<float> chunks, partial, pre, post, comb, cproj, cscore, wp, iqw, po, pml;
  Dev<uint8_t> iq, lin_kv, lin_sc, flush;
  Dev<uint32_t> iqsc;
  Dev<int> pos;

  attn::Window window(Layer& l) {
    if (prefill) return attn::Window{lin_kv.p, lin_sc.p, -1, ctx - R - attn::kWindow};
    return attn::Window{l.ring_kv.p, l.ring_sc.p};
  }
  const float2* table(int L) { return L < 2 ? tab_plain.p : tab_yarn.p; }

  void setup() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    ir0 = prefill ? R - R / kParts : 0;
    int maxpos = ctx + 16;
    tab_plain = Dev<float2>((size_t)maxpos * 32);
    tab_yarn = Dev<float2>((size_t)maxpos * 32);
    attn::rope_table_build(tab_plain.p, maxpos, false, 0);
    attn::rope_table_build(tab_yarn.p, maxpos, true, 0);
    std::vector<int> ph(R);
    for (int r = 0; r < R; ++r) ph[r] = ctx - R + r;
    pos = Dev<int>(ph.data(), R);
    for (auto* d : {&stream[0], &stream[1]}) *d = Dev<uint16_t>((size_t)R * attn::kHcDim);
    y = Dev<uint16_t>((size_t)R * kDim);
    sub = Dev<uint16_t>((size_t)R * kDim);
    qa = Dev<uint16_t>((size_t)R * attn::kQLora);
    qr = Dev<uint16_t>((size_t)R * attn::kQLora);
    kvp = Dev<uint16_t>((size_t)R * attn::kHead);
    qpre = Dev<uint16_t>((size_t)R * attn::kHeads * attn::kHead);
    out = Dev<uint16_t>((size_t)R * attn::kHeads * attn::kHead);
    iqpre = Dev<uint16_t>((size_t)R * attn::kIdxHeads * attn::kIdxDim);
    for (auto* d : {&stream[0], &y, &sub, &qa, &kvp, &qpre, &iqpre}) rnd_bf16(*d, 1.0f);
    chunks = Dev<float>((size_t)attn::hc_chunks(0, attn::kHcDim) * R * 25);
    partial = Dev<float>((size_t)R * 25);
    pre = Dev<float>((size_t)R * 4);
    post = Dev<float>((size_t)R * 4);
    comb = Dev<float>((size_t)R * 16);
    cproj = Dev<float>((size_t)R * 512);
    cscore = Dev<float>((size_t)R * 512);
    wp = Dev<float>((size_t)R * 32);
    iq = Dev<uint8_t>((size_t)R * 32 * 64);
    iqsc = Dev<uint32_t>((size_t)R * 32);
    iqw = Dev<float>((size_t)R * 32);
    int max_splits = 16;
    po = Dev<float>((size_t)R * max_splits * attn::kHeads * attn::kHead);
    pml = Dev<float>((size_t)R * max_splits * attn::kHeads * 2);
    if (prefill) {
      lin_kv = Dev<uint8_t>((size_t)(R + attn::kWindow) * 512);
      lin_sc = Dev<uint8_t>((size_t)(R + attn::kWindow) * 16);
      rnd_u8(lin_kv, 0x77, 0);
      rnd_u8(lin_sc, 7, 120);
    }
    flush = Dev<uint8_t>((size_t)256 << 20);
    ly.resize(kLayers);
    for (int L = 0; L < kLayers; ++L) {
      Layer& l = ly[L];
      l.L = L;
      l.ratio = ratio_of(L);
      l.ring_kv = Dev<uint8_t>((size_t)attn::kRing * 512);
      l.ring_sc = Dev<uint8_t>((size_t)attn::kRing * 16);
      rnd_u8(l.ring_kv, 0x77, 0);
      rnd_u8(l.ring_sc, 7, 120);
      l.qnw = Dev<uint16_t>(attn::kQLora);
      l.kvw = Dev<uint16_t>(512);
      rnd_bf16(l.qnw, 1.0f);
      rnd_bf16(l.kvw, 1.0f);
      l.sink = Dev<float>(16);
      rnd_f32(l.sink, 1.0f);
      for (int h = 0; h < 2; ++h) {
        l.hnorm[h] = Dev<uint16_t>(kDim);
        l.fn[h] = Dev<float>((size_t)attn::kMix * attn::kHcDim);
        l.hsc[h] = Dev<float>(3);
        l.hbase[h] = Dev<float>(attn::kMix);
        rnd_bf16(l.hnorm[h], 1.0f);
        rnd_f32(l.fn[h], 0.01f);
        rnd_f32(l.hsc[h], 0.1f);
        rnd_f32(l.hbase[h], 1.0f);
      }
      if (kv_src(L)) {
        l.cap = ctx / l.ratio + 64;
        l.wkv = Dev<uint16_t>((size_t)512 * kDim);
        rnd_bf16(l.wkv, 0.02f);
        if (l.ratio > 1) {
          l.wgate = Dev<uint16_t>((size_t)512 * kDim);
          rnd_bf16(l.wgate, 0.02f);
        }
        l.cnorm = Dev<uint16_t>(512);
        l.wk = Dev<uint16_t>((size_t)128 * 512);
        l.knorm = Dev<uint16_t>(128);
        rnd_bf16(l.cnorm, 1.0f);
        rnd_bf16(l.wk, 0.05f);
        rnd_bf16(l.knorm, 1.0f);
        l.carry_kv = Dev<float>(16 * 512);
        l.carry_sc = Dev<float>(16 * 512);
        l.ckv_kv = Dev<uint8_t>((size_t)l.cap * 256);
        l.ckv_sc = Dev<uint8_t>((size_t)l.cap * 32);
        l.ik_k = Dev<uint8_t>((size_t)l.cap * 64);
        l.ik_sc = Dev<uint8_t>((size_t)l.cap * 4);
        rnd_u8(l.ckv_kv, 0xFF, 0);
        rnd_u8(l.ckv_sc, 0x1F, 0x28);
        rnd_u8(l.ik_k, 0xFF, 0);
        rnd_u8(l.ik_sc, 7, 120);
      }
      if (idx_src(L)) {
        l.wproj = Dev<uint16_t>((size_t)32 * kDim);
        rnd_bf16(l.wproj, 0.02f);
        int n = ctx / ratio_of(L);  // positions the last row sees
        l.pmax = prefill ? (n + 63) / 64 * 64 : ((n + kParts - 1) / kParts + 63) / 64 * 64;
        if (L >= 24) l.pmax = kPoolBlocks * 8;
        l.kstride = (l.pmax + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
        l.keys = Dev<uint16_t>((size_t)R * l.kstride);
        if (L == 20) l.bkeys = Dev<uint16_t>((size_t)R * l.kstride / 8);
        l.cand = Dev<int>((size_t)R * kParts * kTopk);
        l.cand_key = Dev<uint16_t>((size_t)R * kParts * kTopk);
        l.top = Dev<int>((size_t)R * kTopk);
        l.topkey = Dev<uint16_t>((size_t)R * kTopk);
        fill_topk<<<R, 256>>>(l.top.p, R, kTopk, pos.p, l.ratio);
        if (L == 20) {
          l.pcand = Dev<int>((size_t)R * kParts * kPoolBlocks);
          l.pcand_key = Dev<uint16_t>((size_t)R * kParts * kPoolBlocks);
          l.pool = Dev<int>((size_t)R * kPoolBlocks);
          l.poolkey = Dev<uint16_t>((size_t)R * kPoolBlocks);
          fill_topk<<<R, 256>>>(l.pool.p, R, kPoolBlocks, pos.p, 8);
        }
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!prefill) {  // the other ranks' local selections, as the all-gather would deliver them
      for (int L : {2, 8, 14, 20}) {
        Layer& l = ly[L];
        prep_q(l);
        for (int part = 1; part < kParts; ++part) {
          attn::IndexScan sc;
          sc.part = part;
          sc.nparts = kParts;
          attn::index_score(q(), pos.p, R, l.ratio, l.ik(), sc, l.pmax, l.keys.p, l.kstride,
                            L == 20 ? l.bkeys.p : nullptr, l.kstride / 8, s);
          attn::index_topk(l.keys.p, l.kstride, pos.p, R, l.ratio, sc, l.pmax, false, kTopk, l.cand.p + part * kTopk,
                           l.cand_key.p + part * kTopk, kParts * kTopk, s);
          if (L == 20)
            attn::index_topk(l.bkeys.p, l.kstride / 8, pos.p, R, 1, sc, l.pmax, true, kPoolBlocks,
                             l.pcand.p + part * kPoolBlocks, l.pcand_key.p + part * kPoolBlocks, kParts * kPoolBlocks, s);
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(s));
    }
  }

  attn::IndexQ q() { return attn::IndexQ{iq.p, iqsc.p, iqw.p}; }
  void prep_q(Layer& l) {
    int iR = R - ir0;
    attn::gemv_bf16(y.p + (size_t)ir0 * kDim, iR, kDim, l.wproj.p, nullptr, 32, 32, wp.p, nullptr, true, s);
    attn::index_q_prep(iqpre.p + (size_t)ir0 * 32 * 128, wp.p, pos.p + ir0, tab_yarn.p, q(), nullptr, iR, s);
  }

  // one layer's attention path (without mHC), the parts in `mask`
  void attn_path(int L, unsigned mask) {
    Layer& l = ly[L];
    attn::Window win = window(l);
    if (mask & kPrep) {
      attn::q_norm(qa.p, l.qnw.p, qr.p, R, s);
      attn::kv_write(kvp.p, l.kvw.p, pos.p, table(L), win, nullptr, R, s);
    }
    if (kv_src(L) && (mask & kComp)) {
      attn::compressor_project(y.p, l.cw(), cproj.p, cscore.p, R, s);
      attn::compressor_emit(cproj.p, cscore.p, pos.p, R, l.cw(), {l.carry_kv.p, l.carry_sc.p}, tab_yarn.p, l.ckv(),
                            l.ik(), nullptr, 0, s);
    }
    if (idx_src(L)) {
      int iR = R - ir0;
      const int* ip = pos.p + ir0;
      if (mask & kIdxQ) prep_q(l);
      attn::IndexScan sc;
      bool pool = L == 20;
      if (L >= 24) {
        sc.blocks = ly[20].pool.p + (size_t)ir0 * kPoolBlocks;
        sc.nblocks = kPoolBlocks;
      } else if (!prefill) {
        sc.part = 0;
        sc.nparts = kParts;
      }
      const Layer& src = ly[kv_source(L)];
      if (mask & kScore)
        attn::index_score(q(), ip, iR, l.ratio, src.ik(), sc, l.pmax, l.keys.p, l.kstride, pool ? l.bkeys.p : nullptr,
                          l.kstride / 8, s);
      bool merged = !prefill && L < 24;  // decode Full layers: local lists, then the 4-rank merge
      if (mask & kSelect) {
        if (merged) {
          attn::index_topk(l.keys.p, l.kstride, ip, iR, l.ratio, sc, l.pmax, false, kTopk, l.cand.p, l.cand_key.p,
                           kParts * kTopk, s);
          if (pool)
            attn::index_topk(l.bkeys.p, l.kstride / 8, ip, iR, 1, sc, l.pmax, true, kPoolBlocks, l.pcand.p,
                             l.pcand_key.p, kParts * kPoolBlocks, s);
        } else {
          attn::index_topk(l.keys.p, l.kstride, ip, iR, l.ratio, sc, l.pmax, false, kTopk, l.top.p + (size_t)ir0 * kTopk,
                           l.topkey.p + (size_t)ir0 * kTopk, kTopk, s);
          if (pool)
            attn::index_topk(l.bkeys.p, l.kstride / 8, ip, iR, 1, sc, l.pmax, true, kPoolBlocks,
                             l.pool.p + (size_t)ir0 * kPoolBlocks, l.poolkey.p + (size_t)ir0 * kPoolBlocks, kPoolBlocks, s);
        }
      }
      if (merged && (mask & kMerge)) {
        attn::index_merge(l.cand.p, l.cand_key.p, kParts * kTopk, R, kTopk, l.top.p, l.topkey.p, kTopk, s);
        if (pool)
          attn::index_merge(l.pcand.p, l.pcand_key.p, kParts * kPoolBlocks, R, kPoolBlocks, l.pool.p, l.poolkey.p,
                            kPoolBlocks, s);
      }
    }
    if (mask & kAttn) {
      int kmax = L < 2 ? 0 : kTopk;
      const int* topk = L < 2 ? nullptr : ly[index_source(L)].top.p;
      attn::Compressed ckv = L < 2 ? attn::Compressed{nullptr, nullptr, 0} : ly[kv_source(L)].ckv();
      int nblocks = 2 + kmax / 64;
      int sp = prefill ? 1 : splits > 0 ? splits : nblocks;
      attn::sparse_attn(qpre.p, pos.p, {}, R, table(L), win, ckv, topk, kmax, l.sink.p, out.p, sp,
                        attn::AttnScratch{po.p, pml.p, 16}, s);
    }
  }
  // both sublayers' mHC of a layer: mixes (this rank's copy of the stream, or all of it), Sinkhorn, pre-mix
  // with the norm, and the post/comb mix of a sublayer output back into the stream
  void mhc(int L) {
    Layer& l = ly[L];
    int k0 = hc_full ? 0 : 0, k1 = hc_full ? attn::kHcDim : kDim;
    for (int h = 0; h < 2; ++h) {
      attn::hc_partial(stream[h].p, R, l.fn[h].p, k0, k1, chunks.p, partial.p, s);
      attn::hc_finish(partial.p, R, l.hsc[h].p, l.hbase[h].p, pre.p, post.p, comb.p, s);
      attn::hc_pre(stream[h].p, pre.p, R, l.hnorm[h].p, y.p, nullptr, s);
      attn::hc_post(sub.p, stream[h].p, post.p, comb.p, R, stream[h ^ 1].p, s);
    }
  }

  template <class F>
  float time_us(F&& f) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    f();
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<float> t;
    for (int i = 0; i < iters + 2; ++i) {
      CUDA_CHECK(cudaMemsetAsync(flush.p, i & 0xff, flush.n, s));  // evict L2: weights and caches stream from DRAM
      CUDA_CHECK(cudaEventRecord(a, s));
      CUDA_CHECK(cudaGraphLaunch(ge, s));
      CUDA_CHECK(cudaEventRecord(b, s));
      CUDA_CHECK(cudaEventSynchronize(b));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      if (i >= 2) t.push_back(ms * 1000.f);
    }
    std::sort(t.begin(), t.end());
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    CUDA_CHECK(cudaGraphExecDestroy(ge));
    CUDA_CHECK(cudaGraphDestroy(g));
    return t[t.size() / 2];
  }
};

}  // namespace

int main(int argc, char** argv) {
  int rows = 1, splits = 0, iters = 30, prefill = 0;
  bool detail = false;
  std::vector<int> ctxs;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--rows") rows = std::atoi(argv[++i]);
    else if (a == "--splits") splits = std::atoi(argv[++i]);
    else if (a == "--iters") iters = std::atoi(argv[++i]);
    else if (a == "--prefill") prefill = std::atoi(argv[++i]);
    else if (a == "--detail") detail = true;
    else ctxs.push_back(std::atoi(argv[i]));
  }
  if (ctxs.empty()) ctxs = {1024, 16384, 131072, 1048576};
  int dev;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("%s, %d SMs, L2 %d MB; %s, %d row(s), attention splits %s, median of %d replays after an L2 flush\n",
              prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, prefill ? "prefill chunk" : "decode",
              prefill ? prefill : rows, splits ? std::to_string(splits).c_str() : "all blocks", iters);
  for (int ctx : ctxs) {
    Bench b;
    b.R = prefill ? prefill : rows;
    b.ctx = ctx;
    b.splits = splits;
    b.iters = iters;
    b.prefill = prefill > 0;
    b.hc_full = b.prefill;
    b.setup();
    for (int L = 0; L < kLayers; ++L) b.attn_path(L, kAll);  // eager warm-up (also sets kernel attributes)
    for (int L = 0; L < kLayers; ++L) b.mhc(L);
    CUDA_CHECK(cudaStreamSynchronize(b.s));
    if (ctx == ctxs[0]) {
      float e = b.time_us([&] {
        for (int i = 0; i < 64; ++i) empty_kernel<<<1, 32, 0, b.s>>>();
      });
      float e1 = b.time_us([&] { empty_kernel<<<1, 32, 0, b.s>>>(); });
      std::printf("launch floor: a graph of 1 empty kernel %.1f us; of 64, %.2f us per kernel\n", e1, e / 64);
    }
    std::printf("\n== context %d tokens%s ==\n", ctx, b.prefill ? " (chunk ends there)" : "");
    std::map<std::string, std::pair<float, int>> by_type;
    std::vector<std::string> order;
    float sum_attn = 0, sum_mhc = 0;
    for (int L = 0; L < kLayers; ++L) {
      float t = b.time_us([&] { b.attn_path(L, kAll); });
      std::string ty = type_of(L);
      if (!by_type.count(ty)) order.push_back(ty);
      by_type[ty].first += t;
      by_type[ty].second += 1;
      sum_attn += t;
    }
    for (int L = 0; L < kLayers; L += 13) sum_mhc += b.time_us([&] { b.mhc(L); });
    sum_mhc = sum_mhc / 4 * kLayers;  // layers 0, 13, 26, 39 sampled; identical work in every layer
    std::printf("%-38s %6s %10s %10s\n", "layer type", "layers", "us/layer", "us/step");
    for (auto& ty : order) {
      auto [t, n] = by_type[ty];
      std::printf("%-38s %6d %10.1f %10.1f\n", ty.c_str(), n, t / n, t);
    }
    std::printf("%-38s %6d %10.1f %10.1f   (%s)\n", "mHC, both sublayers", kLayers, sum_mhc / kLayers, sum_mhc,
                b.hc_full ? "full 20480-column projection" : "this rank's 5120 columns");
    float whole = b.time_us([&] {
      for (int L = 0; L < kLayers; ++L) {
        b.attn_path(L, kAll);
        b.mhc(L);
      }
    });
    std::printf("%-38s %6s %10s %10.1f   (sum of the per-layer graphs: %.1f)\n", "whole step, one graph", "", "", whole,
                sum_attn + sum_mhc);
    if (detail) {
      std::printf("components (us):            prep  compr  idx_q  score select  merge   attn\n");
      for (int L : {0, 2, 3, 20, 21, 24}) {
        std::printf("  layer %2d %-17s", L, type_of(L).substr(0, 17).c_str());
        for (unsigned m : {kPrep, kComp, kIdxQ, kScore, kSelect, kMerge, kAttn}) {
          bool any = m == kPrep || m == kAttn || (m == kComp && kv_src(L)) ||
                     ((m == kIdxQ || m == kScore || m == kSelect) && idx_src(L)) ||
                     (m == kMerge && idx_src(L) && L < 24 && !b.prefill);
          if (any)
            std::printf(" %6.1f", b.time_us([&] { b.attn_path(L, m); }));
          else
            std::printf(" %6s", "-");
        }
        std::printf("\n");
      }
      b.hc_full = !b.hc_full;
      std::printf("  mHC per layer with %s: %.1f us\n", b.hc_full ? "the full projection" : "this rank's columns",
                  b.time_us([&] { b.mhc(5); }));
      b.hc_full = !b.hc_full;
      if (!b.prefill)
        for (int sp : {1, 2, 5, 10}) {
          b.splits = sp;
          std::printf("  attention layer 20 with %2d split(s): %.1f us\n", sp, b.time_us([&] { b.attn_path(20, kAttn); }));
        }
      b.splits = splits;
    }
  }
  return 0;
}
