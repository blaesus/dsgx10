// Per-rank timing of the prefill attention-side kernels (prefill_attn.h) against engine/attn's
// decode-shaped kernels, on synthetic state: a chunk of R rows ending at a context of ctx tokens, 16 heads
// (TP=4). Every timed replay is a CUDA graph launched after an L2 flush (median of N), as attn's bench.
//   bench_pattn [--rows R] [--iters N] [--only attn|index|comp|mhc|mhcr|mhcs|dec|layer] [--ref] [--kmax 0,512]
//               [--batch 64,128] [--gemm-ms G] ctx...
// --ref also times engine/attn's kernels on the same state. Without --only (or with layer) it ends with the
// per-layer sums by layer type for each context and the chunk's projected prefill rate with G ms of GEMMs
// per layer (default 15.22, kernels/README.md at M = 4096), then CED's decoder replay (docs/ced.md: the
// 128-row chunk through layers 20-39, attention side) at each context.
#include <algorithm>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "attn/tests/testutil.h"
#include "common/check.h"
#include "kernels/dsk.h"
#include "kernels/prefill_attn.h"

using namespace attn_test;

namespace {

std::map<std::string, float> g_t;  // results by name, for the per-layer sums
std::string key(const char* what, int a, int b = -1) {
  return std::string(what) + "/" + std::to_string(a) + (b >= 0 ? "/" + std::to_string(b) : "");
}

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
// Ascending selections of k = min(kmax, n) of row r's n = (pos+1)/ratio entries, one per stratum of n/k;
// rows r with equal r >> share pick the same offsets (share 0: every row its own, 3: groups of 8 rows).
__global__ void fill_topk(int* top, int rows, int kmax, const int* pos, int ratio, int share, uint32_t seed) {
  int r = blockIdx.x;
  if (r >= rows) return;
  int n = (pos[r] + 1) / ratio, k = min(kmax, n);
  for (int i = threadIdx.x; i < kmax; i += blockDim.x) {
    int v = -1;
    if (i < k) {
      long a = (long)i * n / k, b = (long)(i + 1) * n / k;
      v = (int)(a + hash32((uint64_t)(r >> share) * kmax + i, seed) % (uint32_t)(b - a));
    }
    top[(size_t)r * kmax + i] = v;
  }
}

uint32_t g_seed = 1;
void rnd_u8(Dev<uint8_t>& d, uint32_t mask, uint32_t add) { fill_u8<<<1024, 256>>>(d.p, d.n, g_seed++, mask, add); }
void rnd_bf16(Dev<uint16_t>& d, float scale) { fill_bf16<<<1024, 256>>>(d.p, d.n, g_seed++, scale); }
void rnd_f32(Dev<float>& d, float scale) { fill_f32<<<1024, 256>>>(d.p, d.n, g_seed++, scale); }

struct Timer {
  cudaStream_t s;
  int iters;
  Dev<uint8_t> flush;
  Timer(int it) : iters(it), flush((size_t)256 << 20) { CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
  template <class F>
  float us(F&& f) {
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
      CUDA_CHECK(cudaMemsetAsync(flush.p, i & 0xff, flush.n, s));
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

// ---- sparse attention -------------------------------------------------------------------------------
void bench_attn(Timer& T, int R, int ctx, bool ref, const std::vector<int>& kmaxs) {
  const int cap = ctx / 2 + 64;
  std::vector<int> ph(R);
  for (int r = 0; r < R; ++r) ph[r] = ctx - R + r;
  Dev<int> pos(ph.data(), R);
  Dev<float2> tab((size_t)(ctx + 16) * 32);
  attn::rope_table_build(tab.p, ctx + 16, true, 0);
  Dev<uint16_t> q((size_t)R * 16 * 512), out((size_t)R * 16 * 512);
  Dev<uint8_t> wkv((size_t)(R + 128) * 512), wsc((size_t)(R + 128) * 16), ckv((size_t)cap * 256), csc((size_t)cap * 32);
  Dev<float> sink(16);
  Dev<int> topk((size_t)R * 512);
  rnd_bf16(q, 1.0f);
  rnd_u8(wkv, 0x77, 0);
  rnd_u8(wsc, 7, 120);
  rnd_u8(ckv, 0xFF, 0);
  rnd_u8(csc, 0x1F, 0x28);
  rnd_f32(sink, 1.0f);
  attn::Window win{wkv.p, wsc.p, -1, ctx - R - 128};
  attn::Compressed cc{ckv.p, csc.p, cap};
  for (int share : {0, 3}) {
    fill_topk<<<R, 256>>>(topk.p, R, 512, pos.p, 2, share, 77);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int kmax : kmaxs) {
      if (kmax != 512 && share) continue;
      float t = T.us([&] { dsk::sparse_attn_p(q.p, pos.p, R, tab.p, win, cc, topk.p, kmax, sink.p, out.p, T.s); });
      float tr = !ref ? 0.f : T.us([&] {
        attn::sparse_attn(q.p, pos.p, {}, R, tab.p, win, cc, kmax ? topk.p : nullptr, kmax, sink.p, out.p, 1,
                          attn::AttnScratch{nullptr, nullptr, 0}, T.s);
      });
      if (!share) g_t[key("attn", ctx, kmax)] = t;
      double gflop = 2.0 * 2 * 16 * 512 * (128.0 + kmax) * R * 1e-9;
      std::printf("attn  ctx %6d rows %5d %-10s %-12s %8.1f us  %6.1f TFLOPS", ctx, R, ("window+" + std::to_string(kmax)).c_str(),
                  kmax ? (share ? "topk shared/8" : "topk own") : "", t, gflop / t * 1e3);
      if (ref) std::printf("  | attn::sparse_attn %8.1f us (x%.1f)", tr, tr / t);
      std::printf("\n");
    }
  }
}

// ---- indexer ----------------------------------------------------------------------------------------
// head weights as index_q_prep makes them: bf16 values times 2^-6
__global__ void fill_w(float* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float v = ((hash32(i, seed) & 0xFFFFFF) * (1.0f / 8388608.0f) - 1.0f) * 2.0f;
    p[i] = __uint_as_float(__float_as_uint(v) & 0xffff0000u) * 0x1p-6f;
  }
}
// One rank's share of a Full layer's indexer for a chunk of R rows ending at ctx: rows interleaved across
// the 4 ranks (this is the last rank, the longest rows), scores of all compressed positions so far and
// the top-512 per row. Also layer 20's (ratio 1, with the pool) for the decoder's 128 rows split 4 ways.
void bench_index(Timer& T, int R, int ctx, bool ref, const std::vector<int>& batches) {
  for (int which = 0; which < 2; ++which) {
    const int ratio = which ? 1 : 2, rows = which ? 128 / 4 : R / 4, K = 512, KB = 2048;
    const int n = ctx / ratio, cap = n + 64, stride = (n + 255) / 256 * 256;
    const bool pool = which == 1;
    std::vector<int> ph(rows);
    const int span = which ? 128 : R;
    for (int i = 0; i < rows; ++i) ph[i] = ctx - span + 4 * i + 3;
    Dev<int> pos(ph.data(), rows);
    Dev<uint8_t> qq((size_t)rows * 32 * 64), kk((size_t)cap * 64), ksc((size_t)cap * 4), qsc8((size_t)rows * 32 * 4);
    Dev<float> qw((size_t)rows * 32);
    rnd_u8(qq, 0xFF, 0);
    rnd_u8(kk, 0xFF, 0);
    rnd_u8(ksc, 7, 120);
    rnd_u8(qsc8, 7, 122);
    fill_w<<<512, 256>>>(qw.p, qw.n, g_seed++);
    attn::IndexQ q{qq.p, reinterpret_cast<uint32_t*>(qsc8.p), qw.p};
    attn::IndexKeys ik{kk.p, ksc.p, cap};
    Dev<uint16_t> keys((size_t)rows * stride), bkeys((size_t)rows * stride / 8), ky((size_t)rows * K),
        pk((size_t)rows * KB);
    Dev<int> top((size_t)rows * K), pl((size_t)rows * KB);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double gflop = 2.0 * 32 * 128 * rows * ((double)n - span / 2.0 / ratio) * 1e-9;
    const char* nm = which ? "layer 20 (pool), 32 decoder rows" : "Full layer";
    float ts = T.us([&] {
      dsk::index_score_p(q, pos.p, rows, ratio, ik, n, keys.p, stride, pool ? bkeys.p : nullptr, stride / 8, T.s);
    });
    std::printf("index ctx %6d %-34s %5d rows: scores %8.1f us (%5.1f TFLOPS)", ctx, nm, rows, ts, gflop / ts * 1e3);
    if (ref) {
      float tr = T.us([&] {
        attn::index_score(q, pos.p, rows, ratio, ik, attn::IndexScan{}, n, keys.p, stride, pool ? bkeys.p : nullptr,
                          stride / 8, T.s);
      });
      float tt = T.us([&] {
        attn::index_topk(keys.p, stride, pos.p, rows, ratio, attn::IndexScan{}, n, false, K, top.p, ky.p, K, T.s);
        if (pool)
          attn::index_topk(bkeys.p, stride / 8, pos.p, rows, ratio, attn::IndexScan{}, n, true, KB, pl.p, pk.p, KB, T.s);
      });
      std::printf(" | attn::index_score %8.1f us, select from DRAM %8.1f us", tr, tt);
    }
    std::printf("\n");
    for (int b : batches) {
      if (b > rows && b != batches.front()) continue;
      float tb = T.us([&] {
        dsk::index_topk_p(q, pos.p, rows, ratio, ik, n, b, keys.p, stride, K, top.p, ky.p, pool ? bkeys.p : nullptr, KB,
                          pl.p, pk.p, T.s);
      });
      std::printf("index ctx %6d %-34s       scores + top-k in batches of %4d rows: %8.1f us\n", ctx, nm, b, tb);
      const std::string k = key(which ? "pool" : "index", ctx);
      if (!g_t.count(k) || tb < g_t[k]) g_t[k] = tb;
    }
  }
}

// ---- compressor ------------------------------------------------------------------------------------
// The kv-source layers' compressor for a chunk: ratio 2 (layers 2, 8, 14) and ratio 1 (layer 20), on this
// rank's quarter of the rows (rows split) and on all of them (replicated).
void bench_comp(Timer& T, int R, bool ref) {
  for (int ratio : {2, 1})
    for (int rows : {R / 4, R}) {
      const int p0 = 2 * R, maxpos = p0 + rows + 1, cap = maxpos / ratio + 1;
      Dev<float2> table((size_t)maxpos * 32);
      attn::rope_table_build(table.p, maxpos, true, 0);
      Dev<uint16_t> x((size_t)rows * 5120), wkv(512 * 5120), wgate(512 * 5120), norm(512), wk(128 * 512), knorm(128);
      rnd_bf16(x, 2.0f);
      rnd_bf16(wkv, 0.03f);
      rnd_bf16(wgate, 0.05f);
      rnd_bf16(wk, 0.08f);
      rnd_bf16(norm, 1.0f);
      rnd_bf16(knorm, 1.0f);
      attn::CompressorWeights w{ratio, wkv.p, ratio > 1 ? wgate.p : nullptr, norm.p, wk.p, knorm.p};
      std::vector<int> ph(rows);
      for (int r = 0; r < rows; ++r) ph[r] = p0 + r;
      Dev<int> pos(ph.data(), rows);
      Dev<float> kv((size_t)rows * 512), score((size_t)rows * 512), ck(16 * 512), cs(16 * 512);
      Dev<uint8_t> ckv((size_t)cap * 256), csc((size_t)cap * 32), ik((size_t)cap * 64), isc((size_t)cap * 4);
      attn::Compressed cc{ckv.p, csc.p, cap};
      attn::IndexKeys ii{ik.p, isc.p, cap};
      CUDA_CHECK(cudaDeviceSynchronize());
      dsk::compressor_project_p(x.p, w, kv.p, score.p, rows, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      const float tp = T.us([&] { dsk::compressor_project_p(x.p, w, kv.p, score.p, rows, T.s); });
      const float te = T.us([&] { dsk::compressor_emit_p(kv.p, score.p, pos.p, rows, w, {ck.p, cs.p}, table.p, cc, ii, T.s); });
      const double gflop = 2.0 * rows * 512 * ratio * 5120 * 1e-9;
      std::printf("comp  ratio %d rows %5d: projection %7.1f us (%5.1f TFLOPS), %5d entries %7.1f us", ratio, rows, tp,
                  gflop / tp * 1e3, rows / ratio, te);
      if (ref) {
        const float tpr = T.us([&] { attn::compressor_project(x.p, w, kv.p, score.p, rows, T.s); });
        const float ter = T.us([&] {
          attn::compressor_emit(kv.p, score.p, pos.p, rows, w, {ck.p, cs.p}, table.p, cc, ii, nullptr, 0, T.s);
        });
        std::printf(" | attn: %8.1f + %7.1f us", tpr, ter);
      }
      std::printf("\n");
      g_t[key("comp", ratio, rows)] = tp + te;
    }
}

// ---- mHC --------------------------------------------------------------------------------------------
// One sublayer boundary (hc_post, projection + Sinkhorn, hc_pre + norm, act_quant) on this rank's quarter
// of the rows and on all of them.
void bench_mhc(Timer& T, const std::vector<int>& rowsv, bool ref) {
  Dev<float> fn(24 * 20480), scale(3), base(24);
  rnd_f32(fn, 0.01f);
  rnd_f32(scale, 0.5f);
  rnd_f32(base, 1.0f);
  Dev<uint32_t> parts(dsk::hc_fn_parts_bytes() / 4);
  dsk::hc_fn_parts(fn.p, parts.p, 0);
  for (int rows : rowsv) {
    Dev<uint16_t> x((size_t)rows * 20480), y((size_t)rows * 20480), sub((size_t)rows * 5120), xo((size_t)rows * 5120),
        nw(5120);
    Dev<uint8_t> xq((size_t)rows * 5120), xs((size_t)rows * 160);
    Dev<float> pre(rows * 4), post(rows * 4), comb(rows * 16), po(rows * 4), qo(rows * 4), co(rows * 16);
    rnd_bf16(x, 4.0f);
    rnd_bf16(sub, 1.0f);
    rnd_bf16(nw, 1.0f);
    rnd_f32(pre, 0.5f);
    rnd_f32(post, 1.0f);
    rnd_f32(comb, 0.5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    dsk::HcBoundary a;
    a.sub = sub.p, a.res = x.p, a.pre = pre.p, a.post = post.p, a.comb = comb.p, a.fn_parts = parts.p;
    a.scale = scale.p, a.base = base.p, a.norm_w = nw.p, a.y = y.p, a.x_out = xo.p, a.xq = xq.p, a.xs = xs.p;
    a.pre_out = po.p, a.post_out = qo.p, a.comb_out = co.p;
    const float tq = T.us([&] { dsk::hc_boundary_p(a, rows, T.s); });
    dsk::HcBoundary b = a;
    b.xq = b.xs = nullptr;
    const float tb = T.us([&] { dsk::hc_boundary_p(b, rows, T.s); });
    b.y = nullptr;  // CED: layer 20's projection of the rows before the replay needs x only
    const float tn = T.us([&] { dsk::hc_boundary_p(b, rows, T.s); });
    const double mb = rows * (10240.0 + 2 * 40960 + 10240 + 5280) * 1e-6;
    std::printf("mhc   rows %5d: boundary %7.1f us (%5.0f GB/s), without act_quant %7.1f us, nor y %7.1f us", rows, tq,
                mb / tq * 1e3, tb, tn);
    g_t[key("mhc_x", rows)] = tn;
    if (ref) {
      Dev<float> chunks((size_t)attn::hc_chunks(0, 20480) * rows * 25), part(rows * 25);
      const float tr = T.us([&] {
        attn::hc_post(sub.p, x.p, post.p, comb.p, rows, y.p, T.s);
        attn::hc_partial(y.p, rows, fn.p, 0, 20480, chunks.p, part.p, T.s);
        attn::hc_finish(part.p, rows, scale.p, base.p, po.p, qo.p, co.p, T.s);
        attn::hc_pre(y.p, pre.p, rows, nw.p, xo.p, nullptr, T.s);
        dsk::act_quant(reinterpret_cast<const __nv_bfloat16*>(xo.p), rows, 5120, xq.p, xs.p, T.s);
      });
      std::printf(" | attn's kernels %8.1f us", tr);
      g_t[key("mhc_attn", rows)] = tr;
    }
    std::printf("\n");
    g_t[key("mhc", rows)] = tq;
  }
}

// hc_boundary_p's rows per CTA and L2 prefetch distance (hc_boundary_tuning) on a whole boundary (hc_post,
// y, act_quant, the projection), cold L2 as every boundary has its own hc_fn: us per variant and rows
void bench_mhc_rows(Timer& T, const std::vector<int>& rowsv) {
  Dev<float> fn(24 * 20480), scale(3), base(24);
  rnd_f32(fn, 0.01f);
  rnd_f32(scale, 0.5f);
  rnd_f32(base, 1.0f);
  Dev<uint32_t> parts(dsk::hc_fn_parts_bytes() / 4);
  dsk::hc_fn_parts(fn.p, parts.p, 0);
  const int Rs[] = {32, 16, 8, 4}, pfs[] = {0, 2, 4, 8};
  std::printf("mhc boundary by rows per CTA R and L2 prefetch distance pf (blocks), us:\n   rows");
  for (int R : Rs)
    for (int pf : pfs) std::printf("  R%-2d pf%d", R, pf);
  std::printf("\n");
  for (int rows : rowsv) {
    Dev<uint16_t> x((size_t)rows * 20480), y((size_t)rows * 20480), sub((size_t)rows * 5120), xo((size_t)rows * 5120),
        nw(5120);
    Dev<uint8_t> xq((size_t)rows * 5120), xs((size_t)rows * 160);
    Dev<float> pre(rows * 4), post(rows * 4), comb(rows * 16), po(rows * 4), qo(rows * 4), co(rows * 16);
    rnd_bf16(x, 4.0f);
    rnd_bf16(sub, 1.0f);
    rnd_bf16(nw, 1.0f);
    rnd_f32(pre, 0.5f);
    rnd_f32(post, 1.0f);
    rnd_f32(comb, 0.5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    dsk::HcBoundary a;
    a.sub = sub.p, a.res = x.p, a.pre = pre.p, a.post = post.p, a.comb = comb.p, a.fn_parts = parts.p;
    a.scale = scale.p, a.base = base.p, a.norm_w = nw.p, a.y = y.p, a.x_out = xo.p, a.xq = xq.p, a.xs = xs.p;
    a.pre_out = po.p, a.post_out = qo.p, a.comb_out = co.p;
    std::printf("  %5d", rows);
    for (int R : Rs)
      for (int pf : pfs) {
        dsk::hc_boundary_tuning(R, pf);
        std::printf(" %8.1f", T.us([&] { dsk::hc_boundary_p(a, rows, T.s); }));
        std::fflush(stdout);
      }
    std::printf("\n");
  }
  dsk::hc_boundary_tuning(0, -1);
}

// the split boundary (hc_boundary_split) against the default and the by-chunk tuning: a whole boundary as
// bench_mhc_rows, cold L2; GB/s of its bytes (the stream read, y, x written, re-read and written, act_quant,
// hc_fn's parts)
void bench_mhc_split(Timer& T, const std::vector<int>& rowsv) {
  Dev<float> fn(24 * 20480), scale(3), base(24);
  rnd_f32(fn, 0.01f);
  rnd_f32(scale, 0.5f);
  rnd_f32(base, 1.0f);
  Dev<uint32_t> parts(dsk::hc_fn_parts_bytes() / 4);
  dsk::hc_fn_parts(fn.p, parts.p, 0);
  std::printf("mhc split boundary, us (GB/s): default R32, R4 pf2, split R4 / R8 / R16 / by rows\n");
  for (int rows : rowsv) {
    Dev<uint16_t> x((size_t)rows * 20480), y((size_t)rows * 20480), sub((size_t)rows * 5120), xo((size_t)rows * 5120),
        nw(5120);
    Dev<uint8_t> xq((size_t)rows * 5120), xs((size_t)rows * 160);
    Dev<float> pre(rows * 4), post(rows * 4), comb(rows * 16), po(rows * 4), qo(rows * 4), co(rows * 16);
    rnd_bf16(x, 4.0f);
    rnd_bf16(sub, 1.0f);
    rnd_bf16(nw, 1.0f);
    rnd_f32(pre, 0.5f);
    rnd_f32(post, 1.0f);
    rnd_f32(comb, 0.5f);
    CUDA_CHECK(cudaDeviceSynchronize());
    dsk::HcBoundary a;
    a.sub = sub.p, a.res = x.p, a.pre = pre.p, a.post = post.p, a.comb = comb.p, a.fn_parts = parts.p;
    a.scale = scale.p, a.base = base.p, a.norm_w = nw.p, a.y = y.p, a.x_out = xo.p, a.xq = xq.p, a.xs = xs.p;
    a.pre_out = po.p, a.post_out = qo.p, a.comb_out = co.p;
    const double bytes = (double)rows * (10240 + 40960 + 40960 + 3 * 10240 + 5120 + 160) + dsk::hc_fn_parts_bytes();
    std::printf("  %5d", rows);
    auto one = [&](int R, int pf, int split) {
      dsk::hc_boundary_tuning(R, pf);
      dsk::hc_boundary_split(split ? 512 : 0, split > 0 ? split : 0);
      const double us = T.us([&] { dsk::hc_boundary_p(a, rows, T.s); });
      std::printf(" %7.1f (%3.0f)", us, bytes / us * 1e-3);
      std::fflush(stdout);
    };
    one(32, 0, 0);
    one(4, 2, 0);
    for (int sp : {4, 8, 16, -1}) one(32, 0, sp);
    std::printf("\n");
  }
  dsk::hc_boundary_tuning(0, -1);
  dsk::hc_boundary_split(0);
}

// ---- the rest of the attention side (attn's row-wise kernels, already fine for many rows) ----------------
void bench_misc(Timer& T, int R, int Q, const std::string& tag) {
  Dev<float2> tab((size_t)(2 * R + 16) * 32);
  attn::rope_table_build(tab.p, 2 * R + 16, true, 0);
  std::vector<int> ph(R);
  for (int r = 0; r < R; ++r) ph[r] = R + r;
  Dev<int> pos(ph.data(), R);
  Dev<uint16_t> qa((size_t)R * 1280), qw(1280), qr((size_t)R * 1280), kvp((size_t)R * 512), kvw(512), xin((size_t)Q * 5120),
      wp(32 * 5120), iq((size_t)Q * 32 * 128);
  Dev<uint8_t> win((size_t)R * 512), wsc((size_t)R * 16), iqq((size_t)Q * 32 * 64);
  Dev<uint32_t> iqs((size_t)Q * 32);
  Dev<float> iw((size_t)Q * 32), wpo((size_t)Q * 32);
  rnd_bf16(qa, 1.0f);
  rnd_bf16(qw, 1.0f);
  rnd_bf16(kvp, 1.0f);
  rnd_bf16(kvw, 1.0f);
  rnd_bf16(xin, 1.0f);
  rnd_bf16(wp, 0.05f);
  rnd_bf16(iq, 1.0f);
  CUDA_CHECK(cudaDeviceSynchronize());
  const float t1 = T.us([&] { attn::q_norm(qa.p, qw.p, qr.p, R, T.s); });
  const float t2 = T.us([&] { attn::kv_write(kvp.p, kvw.p, pos.p, tab.p, attn::Window{win.p, wsc.p, -1, R}, nullptr, R, T.s); });
  const float t3 = T.us([&] { attn::gemv_bf16(xin.p, Q, 5120, wp.p, nullptr, 32, 32, wpo.p, nullptr, true, T.s); });
  const float t4 = T.us([&] { attn::index_q_prep(iq.p, wpo.p, pos.p, tab.p, attn::IndexQ{iqq.p, iqs.p, iw.p}, nullptr, Q, T.s); });
  std::printf("misc  q_norm %d rows %6.1f us, kv_write %d rows %6.1f us, weights_proj %d rows %6.1f us, index_q_prep %d "
              "rows %6.1f us (attn's kernels)\n", R, t1, R, t2, Q, t3, Q, t4);
  g_t[tag + "/all"] = t1 + t2;
  g_t[tag + "/index"] = t3 + t4;
}

// ---- CED decoder replay (docs/ced.md) ----------------------------------------------------------------
// Layers 20-39 over one chunk of 128 rows [P-128, P), per rank: attention on all 128 rows (16 heads, the
// window causal inside the chunk, top-512 of layer 20's ratio-1 KV), layer 20's indexer over all P index
// keys with the pool (2048 blocks of 8) and top-512, Reindex (24, 28, 32, 36: the row's pool, top-512).
// The indexer and Reindex on 32 rows (split 4 ways) and on all 128.
void bench_decoder(Timer& T, int P) {
  const int R = 128, K = 512, KB = 2048, s = P - R, cap = P + 64;
  {
    std::vector<int> ph(R), lh(R, s);
    for (int r = 0; r < R; ++r) ph[r] = s + r;
    Dev<int> pos(ph.data(), R), lo(lh.data(), R);
    Dev<float2> tab((size_t)(P + 16) * 32);
    attn::rope_table_build(tab.p, P + 16, true, 0);
    Dev<uint16_t> q((size_t)R * 16 * 512), out((size_t)R * 16 * 512);
    Dev<uint8_t> wkv((size_t)R * 512), wsc((size_t)R * 16), ckv((size_t)cap * 256), csc((size_t)cap * 32);
    Dev<float> sink(16), so((size_t)R * 10 * 16 * 512), sml((size_t)R * 10 * 16 * 2);
    Dev<int> topk((size_t)R * K);
    rnd_bf16(q, 1.0f);
    rnd_u8(wkv, 0x77, 0);
    rnd_u8(wsc, 7, 120);
    rnd_u8(ckv, 0xFF, 0);
    rnd_u8(csc, 0x1F, 0x28);
    rnd_f32(sink, 1.0f);
    fill_topk<<<R, 256>>>(topk.p, R, K, pos.p, 1, 0, 78);
    CUDA_CHECK(cudaDeviceSynchronize());
    attn::Window win{wkv.p, wsc.p, -1, s};
    attn::Compressed cc{ckv.p, csc.p, cap};
    const float ta = T.us([&] { dsk::sparse_attn_p(q.p, pos.p, R, tab.p, win, cc, topk.p, K, sink.p, out.p, T.s, s); });
    const float tr = T.us([&] {
      attn::sparse_attn(q.p, pos.p, attn::WinBounds{lo.p, nullptr}, R, tab.p, win, cc, topk.p, K, sink.p, out.p, 10,
                        attn::AttnScratch{so.p, sml.p, 10}, T.s);
    });
    std::printf("dec   P %6d attention 128 rows: sparse_attn_p %7.1f us | attn::sparse_attn, 10 splits %7.1f us\n", P, ta,
                tr);
    g_t[key("dattn", P)] = std::min(ta, tr);
  }
  for (int nr : {32, 128}) {
    const int stride = (P + 255) / 256 * 256, ms = KB * 8;
    std::vector<int> ph(nr);
    for (int i = 0; i < nr; ++i) ph[i] = nr == R ? s + i : s + 4 * i + 3;
    Dev<int> pos(ph.data(), nr);
    Dev<uint8_t> qq((size_t)nr * 32 * 64), kk((size_t)cap * 64), ksc((size_t)cap * 4), qsc8((size_t)nr * 32 * 4);
    Dev<float> qw((size_t)nr * 32);
    rnd_u8(qq, 0xFF, 0);
    rnd_u8(kk, 0xFF, 0);
    rnd_u8(ksc, 7, 120);
    rnd_u8(qsc8, 7, 122);
    fill_w<<<512, 256>>>(qw.p, qw.n, g_seed++);
    attn::IndexQ q{qq.p, reinterpret_cast<uint32_t*>(qsc8.p), qw.p};
    attn::IndexKeys ik{kk.p, ksc.p, cap};
    Dev<uint16_t> keys((size_t)nr * stride), bkeys((size_t)nr * stride / 8), ky((size_t)nr * K), pk((size_t)nr * KB),
        rkeys((size_t)nr * ms);
    Dev<int> top((size_t)nr * K), pl((size_t)nr * KB);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float ti = T.us([&] {
      dsk::index_topk_p(q, pos.p, nr, 1, ik, P, nr, keys.p, stride, K, top.p, ky.p, bkeys.p, KB, pl.p, pk.p, T.s);
    });
    attn::IndexScan sc;  // the pool just picked, scored again (another layer's queries in the engine)
    sc.blocks = pl.p, sc.nblocks = KB;
    const float tre = T.us([&] {
      attn::index_score(q, pos.p, nr, 1, ik, sc, ms, rkeys.p, ms, nullptr, 0, T.s);
      attn::index_topk(rkeys.p, ms, pos.p, nr, 1, sc, ms, false, K, top.p, ky.p, K, T.s);
    });
    std::printf("dec   P %6d indexer %3d rows: layer 20 (all P keys, pool, top-512) %7.1f us, Reindex (pool of %d, "
                "top-512, attn's kernels) %7.1f us\n", P, nr, ti, ms, tre);
    g_t[key("dindex", P, nr)] = ti;
    g_t[key("dreindex", P, nr)] = tre;
  }
}

void decoder_table(const std::vector<int>& ctxs) {
  auto get = [&](const std::string& k) {
    auto it = g_t.find(k);
    return it == g_t.end() ? NAN : it->second * 1e-3;
  };
  std::printf("\nCED decoder replay, per rank, ms (128 rows through layers 20-39; attention on all rows):\n");
  std::printf("%7s %5s | %-24s | %-24s | %8s | %10s\n", "P", "split", "layer 20", "Reindex 24, 28, 32, 36", "the rest",
              "20 layers");
  for (int P : ctxs)
    for (int nr : {32, 128}) {
      const double at = get(key("dattn", P)), mh = 2 * std::min(get(key("mhc", nr)), get(key("mhc_attn", nr)));
      const double misc = get("dmisc/all"), misci = get("dmisc/index");
      const double l20 = at + get(key("dindex", P, nr)) + mh + misc + misci, lre = at + get(key("dreindex", P, nr)) + mh +
                                                                                  misc + misci,
                   lr = at + mh + misc, tot = l20 + 4 * lre + 15 * lr;
      char a[64], b[64];
      std::snprintf(a, sizeof a, "%.3f (idx %.3f)", l20, get(key("dindex", P, nr)));
      std::snprintf(b, sizeof b, "%.3f (reidx %.3f)", lre, get(key("dreindex", P, nr)));
      std::printf("%7d %5s | %-24s | %-24s | %8.3f | %10.2f\n", P, nr == 32 ? "rows" : "none", a, b, lr, tot);
    }
  std::printf("(each layer: attention %s, 2 mHC boundaries on the split's rows, q_norm + kv_write; indexer layers also "
              "weights_proj + index_q_prep; no GEMMs, no collectives)\n", "min(sparse_attn_p, attn's split kernel)");
}

// Per-layer sums by type at R rows, per rank: attention (all rows, 16 heads), the indexer (R/4 rows, rows
// split), the compressor (R/4 rows), two mHC boundaries (R/4 rows), and attn's row-wise kernels.
void layer_table(int R, const std::vector<int>& ctxs, double gemm_ms) {
  auto get = [&](const std::string& k) {
    auto it = g_t.find(k);
    return it == g_t.end() ? NAN : it->second * 1e-3;
  };
  const int Q = R / 4;
  const double mhc = 2 * get(key("mhc", Q)), misc = get("misc/all"), misci = get("misc/index");
  const double comp2 = get(key("comp", 2, Q)), comp1 = get(key("comp", 1, Q)) + get(key("mhc_x", Q));
  std::printf("\nper rank, ms per layer at %d rows (attention on all rows; indexer, compressor and mHC on %d rows):\n", R, Q);
  std::printf("%7s | %-22s | %-22s | %-22s | %8s | %9s | %8s | %9s\n", "ctx", "window 0-1", "Full 2, 8, 14",
              "reuse 3-7, 9-13, 15-19", "layer 20", "20 layers", "+ GEMMs", "tokens/s");
  for (int ctx : ctxs) {
    const double aw = get(key("attn", ctx, 0)), as = get(key("attn", ctx, 512)), ix = get(key("index", ctx));
    const double lw = aw + mhc + misc, lf = as + ix + comp2 + mhc + misc + misci, lr = as + mhc + misc;
    const double tot = 2 * lw + 3 * lf + 15 * lr + comp1, all = tot + 20 * gemm_ms;
    char a[64], b[64], c[64];
    std::snprintf(a, sizeof a, "%.2f (attn %.2f)", lw, aw);
    std::snprintf(b, sizeof b, "%.2f (idx %.2f)", lf, ix);
    std::snprintf(c, sizeof c, "%.2f (attn %.2f)", lr, as);
    std::printf("%7d | %-22s | %-22s | %-22s | %8.2f | %9.1f | %8.1f | %9.0f\n", ctx, a, b, c, comp1, tot, all,
                R / (all * 1e-3));
  }
  std::printf("(each layer: 2 mHC boundaries %.2f, q_norm + kv_write %.2f; Full layers also the compressor %.2f and "
              "weights_proj + index_q_prep %.2f; layer 20 (CED: x for every row): one boundary without y and its "
              "compressor; GEMMs %.2f ms per layer, 20 layers; no collectives)\n",
              mhc, misc, comp2, misci, gemm_ms);
}

}  // namespace

int main(int argc, char** argv) {
  int rows = 4096, iters = 20;
  double gemm_ms = 15.22;
  bool ref = false;
  std::string only;
  std::vector<int> ctxs, kmaxs = {0, 512}, batches = {64, 128, 256};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--rows") rows = std::atoi(argv[++i]);
    else if (a == "--iters") iters = std::atoi(argv[++i]);
    else if (a == "--only") only = argv[++i];
    else if (a == "--ref") ref = true;
    else if (a == "--gemm-ms") gemm_ms = std::atof(argv[++i]);
    else if (a == "--batch") {
      batches.clear();
      for (char* t = std::strtok(argv[++i], ","); t; t = std::strtok(nullptr, ",")) batches.push_back(std::atoi(t));
    } else if (a == "--kmax") {
      kmaxs.clear();
      for (char* t = std::strtok(argv[++i], ","); t; t = std::strtok(nullptr, ",")) kmaxs.push_back(std::atoi(t));
    }
    else ctxs.push_back(std::atoi(a.c_str()));
  }
  if (ctxs.empty()) ctxs = {16384, 65536, 131072};
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("%s, %d SMs, L2 %d MB; chunk of %d rows ending at the context, median of %d replays after an L2 flush\n",
              prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, rows, iters);
  Timer T(iters);
  const bool layer = only.empty() || only == "layer";
  if (layer) kmaxs = {0, 512};
  for (int ctx : ctxs) {
    if (layer || only == "attn") bench_attn(T, rows, ctx, ref, kmaxs);
    if (layer || only == "index") bench_index(T, rows, ctx, ref, batches);
  }
  if (layer || only == "comp") bench_comp(T, rows, ref);
  if (layer || only == "mhc") bench_mhc(T, {rows / 4, rows}, ref);
  if (only == "mhcr") bench_mhc_rows(T, {32, 50, 128, 500, 2000, 2048, 8192});
  if (only == "mhcs") bench_mhc_split(T, {8, 16, 32, 56, 88, 128, 184, 344, 512});
  if (layer) {
    bench_misc(T, rows, rows / 4, "misc");
    layer_table(rows, ctxs, gemm_ms);
  }
  if (layer || only == "dec") {
    std::printf("\nCED decoder replay (one chunk of 128 rows)\n");
    bench_mhc(T, {32, 128}, true);
    bench_misc(T, 128, 32, "dmisc");
    for (int ctx : ctxs) bench_decoder(T, ctx);
    decoder_table(ctxs);
  }
  return 0;
}
