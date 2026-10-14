// Timings of engine/attn on one GB10 (this rank's shapes at TP=4). Weights and caches are cycled through pools
// several times larger than L2 (24 MB), so every timed launch reads from DRAM, as a step's would. Each figure is
// the median over replays of a CUDA graph of one pass over the pool, divided by the launches in it.
// usage: bench_attn [--out FILE] [--only bw,agemv,agemvcfg,apgemm,gemv,index,attn,prefill,pgemm,pattn] [--ctx 32768,131072,262144]
#include <algorithm>
#include <cstdarg>
#include <random>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "testutil.h"

using namespace glm;
using tu::Dev;

namespace {

FILE* g_out = nullptr;
void result(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  printf("%s\n", buf);
  fflush(stdout);
  if (g_out) { fprintf(g_out, "%s\n", buf); fflush(g_out); }
}

// median µs per step of `steps` steps captured in one graph (each a lambda launching that step's kernels)
template <class F>
double time_graph(int steps, F launch_step, int reps = 7) {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t gr;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int s = 0; s < steps; ++s) launch_step(s, st);
  CUDA_CHECK(cudaStreamEndCapture(st, &gr));
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
  CUDA_CHECK(cudaGraphLaunch(ge, st));  // warm-up
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<float> t;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaEventRecord(a, st));
    CUDA_CHECK(cudaGraphLaunch(ge, st));
    CUDA_CHECK(cudaEventRecord(b, st));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    t.push_back(ms * 1000.f / steps);
  }
  std::sort(t.begin(), t.end());
  CUDA_CHECK(cudaGraphExecDestroy(ge));
  CUDA_CHECK(cudaGraphDestroy(gr));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  CUDA_CHECK(cudaStreamDestroy(st));
  return t[t.size() / 2];
}

__global__ void k_fill(uint32_t* p, size_t n, uint32_t seed, uint32_t mode) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    if (mode == 0) p[i] = x & 0x7E7E7E7Eu;                      // e4m3 bytes, no NaN
    else if (mode == 1) p[i] = (x & 0x807F807Fu) | 0x3C003C00u;  // bf16 pairs in +-[2^-7, 2^-6)
    else p[i] = 0x3F800000u | (x & 0x007FFFFFu);                  // f32 in [1, 2)
  }
}
void fill(void* p, size_t bytes, uint32_t seed, uint32_t mode) {
  k_fill<<<480, 256>>>(static_cast<uint32_t*>(p), bytes / 4, seed, mode);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void k_read(const int4* __restrict__ p, size_t n, int* out) {
  int4 acc = make_int4(0, 0, 0, 0);
  size_t stride = (size_t)gridDim.x * blockDim.x, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    int4 a = __ldcs(p + i), b = __ldcs(p + i + stride), c = __ldcs(p + i + 2 * stride), d = __ldcs(p + i + 3 * stride);
    acc.x ^= a.x ^ b.x ^ c.x ^ d.x; acc.y ^= a.y ^ b.y ^ c.y ^ d.y; acc.z ^= a.z ^ b.z ^ c.z ^ d.z; acc.w ^= a.w ^ b.w ^ c.w ^ d.w;
  }
  for (; i < n; i += stride) { int4 v = p[i]; acc.x ^= v.x; }
  if (acc.x == 0x12345678 && acc.y == 1) out[0] = acc.z + acc.w;
}

double g_bw = 0;
void bench_bw() {
  const size_t bytes = (size_t)1 << 30;
  Dev<uint8_t> buf(bytes);
  Dev<int> o(1);
  double best = 0;
  for (int grid : {96, 192, 384}) {
    double us = time_graph(4, [&](int, cudaStream_t s) { k_read<<<grid, 512, 0, s>>>((const int4*)buf.p, bytes / 16, o.p); });
    best = std::max(best, bytes / us / 1e3);
  }
  g_bw = best;
  result("bw\tstreaming read\t%.1f GB/s", best);
}

// ---- GEMVs ---------------------------------------------------------------------------------------------------
struct GemvCase {
  const char* name;
  bool fp8;
  int K;
  std::vector<int> n;  // segments' rows
  int rs, ks, group_rows, x_group, ldx;
  GemvOut out;
  int splitk;
};
void bench_gemv() {
  const std::vector<GemvCase> cases = {
      {"x-projection (q_a+kv_a+wk) 688x6144", true, kDim, {kXRows}, 16, 128, 0, 0, kDim, GemvOut::kBF16, 4},
      {"x-projection, shared layer 656x6144", true, kDim, {kXRowsShared}, 16, 128, 0, 0, kDim, GemvOut::kBF16, 4},
      {"q_b + indexer wq_b (4096+1024)x2048", true, kQLora, {kOProjK, 1024}, 128, 128, 0, 0, kQLora, GemvOut::kBF16, 1},
      {"q_b 4096x2048", true, kQLora, {kOProjK}, 128, 128, 0, 0, kQLora, GemvOut::kBF16, 1},
      {"W_UK^T 16x(512x192) grouped", true, kNope, {kHeads * kKvLora}, 128, 64, kKvLora, kQkDim, kOProjK, GemvOut::kBF16, 1},
      {"W_UV 16x(256x512) grouped", true, kKvLora, {kOProjK}, 64, 128, kVDim, kKvLora, kHeads * kKvLora, GemvOut::kBF16, 1},
      {"o_proj 6144x4096 (f32 partial)", true, kOProjK, {kDim}, 128, 128, 0, 0, kOProjK, GemvOut::kF32, 1},
      {"dense gate+up 6144x6144 (SwiGLU)", true, kDim, {2 * kDenseInterRank}, 128, 128, 0, 0, kDim, GemvOut::kSwiGLU, 1},
      {"dense down 6144x3072 (f32 partial)", true, kDenseInterRank, {kDim}, 128, 128, 0, 0, kDenseInterRank, GemvOut::kF32, 1},
      {"shared gate+up 1024x6144 (SwiGLU)", true, kDim, {2 * kSharedInterRank}, 128, 128, 0, 0, kDim, GemvOut::kSwiGLU, 4},
      {"shared down 6144x512 (f32 partial)", true, kSharedInterRank, {kDim}, 128, 128, 0, 0, kSharedInterRank, GemvOut::kF32, 1},
      {"head BF16 38720x6144", false, kDim, {kVocabRank}, 0, 0, 0, 0, kDim, GemvOut::kBF16, 1},
  };
  Dev<uint16_t> x(16 * 16384);
  fill(x.p, x.n * 2, 7, 1);
  Dev<uint8_t> y(16 * 40000 * 4);
  for (const auto& c : cases) {
    size_t wbytes = 0, sbytes = 0;
    for (int n : c.n) {
      wbytes += (size_t)n * c.K * (c.fp8 ? 1 : 2);
      if (c.fp8) sbytes += (size_t)(c.out == GemvOut::kSwiGLU ? n : n) / c.rs * (c.K / c.ks) * 4;
    }
    const size_t per = wbytes + sbytes;
    const int L = (int)std::max<size_t>(4, ((size_t)768 << 20) / per);
    Dev<uint8_t> pool(per * L);
    fill(pool.p, wbytes * L, 11, c.fp8 ? 0 : 1);
    if (sbytes) fill(pool.p + wbytes * L, sbytes * L, 13, 2);
    for (int rows : {1, 2, 4, 6, 8, 16})
     for (int pdl = 0; pdl < 2; ++pdl) {
      const double us = time_graph(L, [&](int l, cudaStream_t s) {
        GemvSeg sg[2];
        size_t wo = 0, so = 0;
        for (size_t i = 0; i < c.n.size(); ++i) {
          sg[i].w = pool.p + wbytes * l + wo;
          sg[i].s = c.fp8 ? reinterpret_cast<const float*>(pool.p + wbytes * L + sbytes * l + so) : nullptr;
          sg[i].n = c.n[i];
          sg[i].rs = c.rs ? c.rs : 128;
          sg[i].ks = c.ks ? c.ks : 128;
          sg[i].y = y.p + wo / c.K * 4;
          sg[i].ldy = 40000;
          sg[i].group_rows = c.group_rows;
          sg[i].x_group = c.x_group;
          wo += (size_t)c.n[i] * c.K * (c.fp8 ? 1 : 2);
          so += (size_t)c.n[i] / std::max(c.rs, 1) * (c.K / std::max(c.ks, 1)) * 4;
        }
        if (c.out == GemvOut::kSwiGLU) sg[0].ldy = 20000;
        gemv(c.fp8, x.p, c.ldx, rows, c.K, sg, (int)c.n.size(), c.out, c.splitk, s, pdl != 0);
      });
      result("gemv\t%s\t%s\trows %d\t%.2f MB\t%.1f us\t%.1f GB/s\t%.1f %% of 235\t%.1f %% of window", c.name, pdl ? "PDL chain" : "plain", rows, per / 1e6, us,
             per / us / 1e3, per / us / 1e3 / 2.35, g_bw > 0 ? per / us / 1e3 / g_bw * 100 : 0.0);
    }
  }
  // launch-configuration sweep (bit-neutral): the FP8 GEMVs at 1 and 6 rows in a PDL chain
  for (const auto& c : cases) {
    if (!c.fp8) continue;
    size_t wbytes = 0, sbytes = 0;
    for (int n : c.n) {
      wbytes += (size_t)n * c.K;
      sbytes += (size_t)n / c.rs * (c.K / c.ks) * 4;
    }
    const size_t per = wbytes + sbytes;
    const int L = (int)std::max<size_t>(4, ((size_t)768 << 20) / per);
    Dev<uint8_t> pool(per * L);
    fill(pool.p, wbytes * L, 11, 0);
    fill(pool.p + wbytes * L, sbytes * L, 13, 2);
    for (int cfg = -1; cfg < 6; ++cfg)
      for (int rows : {1, 6}) {
        gemv_config(cfg);
        const double us = time_graph(L, [&](int l, cudaStream_t s) {
          GemvSeg sg[2];
          size_t wo = 0, so = 0;
          for (size_t i = 0; i < c.n.size(); ++i) {
            sg[i].w = pool.p + wbytes * l + wo;
            sg[i].s = reinterpret_cast<const float*>(pool.p + wbytes * L + sbytes * l + so);
            sg[i].n = c.n[i];
            sg[i].rs = c.rs;
            sg[i].ks = c.ks;
            sg[i].y = y.p + wo / c.K * 4;
            sg[i].ldy = c.out == GemvOut::kSwiGLU ? 20000 : 40000;
            sg[i].group_rows = c.group_rows;
            sg[i].x_group = c.x_group;
            wo += (size_t)c.n[i] * c.K;
            so += (size_t)c.n[i] / c.rs * (c.K / c.ks) * 4;
          }
          gemv(true, x.p, c.ldx, rows, c.K, sg, (int)c.n.size(), c.out, c.splitk, s, true);
        });
        result("gemvcfg\t%s\tcfg %d\trows %d\t%.1f us\t%.1f %% of 235", c.name, cfg, rows, us, per / us / 1e3 / 2.35);
      }
    gemv_config(-1);
  }
  // the x-projection and the index weights: two launches against one (fused)
  {
    const size_t wb = (size_t)kXRows * kDim, sb = (size_t)kXRows / 16 * (kDim / 128) * 4, ib = (size_t)kIdxHeadsRank * kDim * 2;
    const size_t per = wb + sb + ib;
    const int L = (int)std::max<size_t>(4, ((size_t)768 << 20) / per);
    Dev<uint8_t> pool(per * L);
    fill(pool.p, wb * L, 11, 0);
    fill(pool.p + wb * L, sb * L, 13, 2);
    fill(pool.p + (wb + sb) * L, ib * L, 3, 1);
    Dev<float> w(16 * kIdxHeadsRank);
    for (int fused = 0; fused < 2; ++fused)
      for (int rows : {1, 6}) {
        const double us = time_graph(L, [&](int l, cudaStream_t s) {
          GemvSeg sg;
          sg.w = pool.p + wb * l;
          sg.s = reinterpret_cast<const float*>(pool.p + wb * L + sb * l);
          sg.n = kXRows;
          sg.rs = 16;
          sg.y = y.p;
          sg.ldy = 40000;
          const uint16_t* wp = reinterpret_cast<const uint16_t*>(pool.p + (wb + sb) * L + ib * l);
          if (fused) {
            GemvIw iw;
            iw.wproj = wp;
            iw.nh = kIdxHeadsRank;
            iw.w = w.p;
            iw.ldw = kIdxHeadsRank;
            gemv(true, x.p, kDim, rows, kDim, &sg, 1, GemvOut::kBF16, 4, s, true, &iw);
          } else {
            gemv(true, x.p, kDim, rows, kDim, &sg, 1, GemvOut::kBF16, 4, s, true);
            index_weights(x.p, rows, wp, kIdxHeadsRank, w.p, kIdxHeadsRank, s, true);
          }
        });
        result("gemv\tx-projection + index weights_proj (%s)\tPDL chain\trows %d\t%.2f MB\t%.1f us\t%.1f %% of 235", fused ? "one launch" : "two launches",
               rows, per / 1e6, us, per / us / 1e3 / 2.35);
      }
  }
  // index weights (8 heads a rank, BF16 [8][6144], f32 out)
  {
    const size_t per = (size_t)kIdxHeadsRank * kDim * 2;
    const int L = 256;
    Dev<uint16_t> pool(per / 2 * L);
    fill(pool.p, per * L, 3, 1);
    Dev<float> w(16 * kIdxHeadsRank);
    for (int rows : {1, 6, 16}) {
      const double us = time_graph(L, [&](int l, cudaStream_t s) {
        index_weights(x.p, rows, pool.p + per / 2 * l, kIdxHeadsRank, w.p, kIdxHeadsRank, s);
      });
      result("gemv\tindex weights_proj 8x6144 bf16 (f32)\trows %d\t%.2f MB\t%.1f us\t%.1f GB/s", rows, per / 1e6, us, per / us / 1e3);
    }
  }
}

// ---- prefill projections: the GEMM against the 16-row GEMV launches, chunks of 2048-8192 rows -----------------
void bench_pgemm() {
  struct PCase {
    const char* name;
    int K;
    std::vector<int> n;
    int rs, ks, group_rows, x_group, ldx;
    GemvOut out;
    int splitk;
  };
  const std::vector<PCase> cases = {
      {"x-projection 688x6144 (split-K 4)", kDim, {kXRows}, 16, 128, 0, 0, kDim, GemvOut::kBF16, 4},
      {"q_b + wq_b (4096+1024)x2048", kQLora, {kOProjK, 1024}, 128, 128, 0, 0, kQLora, GemvOut::kBF16, 1},
      {"W_UK^T 16x(512x192) grouped", kNope, {kHeads * kKvLora}, 128, 64, kKvLora, kQkDim, kOProjK, GemvOut::kBF16, 1},
      {"W_UV 16x(256x512) grouped", kKvLora, {kOProjK}, 64, 128, kVDim, kKvLora, kHeads * kKvLora, GemvOut::kBF16, 1},
      {"o_proj 6144x4096 (f32)", kOProjK, {kDim}, 128, 128, 0, 0, kOProjK, GemvOut::kF32, 1},
      {"dense gate+up 6144x6144 (SwiGLU)", kDim, {2 * kDenseInterRank}, 128, 128, 0, 0, kDim, GemvOut::kSwiGLU, 1},
      {"dense down 6144x3072 (f32)", kDenseInterRank, {kDim}, 128, 128, 0, 0, kDenseInterRank, GemvOut::kF32, 1},
      {"shared gate+up 1024x6144 (SwiGLU, split-K 4)", kDim, {2 * kSharedInterRank}, 128, 128, 0, 0, kDim, GemvOut::kSwiGLU, 4},
      {"shared down 6144x512 (f32)", kSharedInterRank, {kDim}, 128, 128, 0, 0, kSharedInterRank, GemvOut::kF32, 1},
  };
  const int RMAX = 8192;
  Dev<uint16_t> x((size_t)RMAX * kHeads * kKvLora);
  fill(x.p, x.n * 2, 7, 1);
  Dev<uint8_t> y((size_t)RMAX * kDim * 4);
  for (const auto& c : cases) {
    size_t wbytes = 0, sbytes = 0;
    int N = 0;
    for (int n : c.n) {
      wbytes += (size_t)n * c.K;
      sbytes += (size_t)n / c.rs * (c.K / c.ks) * 4;
      N += n;
    }
    Dev<uint8_t> w(wbytes + sbytes);
    fill(w.p, wbytes, 11, 0);
    fill(w.p + wbytes, sbytes, 13, 2);
    std::vector<GemvSeg> sg(c.n.size());
    size_t wo = 0, so = 0, yo = 0;
    for (size_t i = 0; i < c.n.size(); ++i) {
      sg[i].fmt = WFmt::kFP8;
      sg[i].w = w.p + wo;
      sg[i].s = reinterpret_cast<const float*>(w.p + wbytes + so);
      sg[i].n = c.n[i];
      sg[i].rs = c.rs;
      sg[i].ks = c.ks;
      sg[i].group_rows = c.group_rows;
      sg[i].x_group = c.x_group;
      sg[i].ldy = c.out == GemvOut::kSwiGLU ? N / 2 : N;
      sg[i].y = y.p + yo * (c.out == GemvOut::kF32 ? 4 : 2);
      wo += (size_t)c.n[i] * c.K;
      so += (size_t)c.n[i] / c.rs * (c.K / c.ks) * 4;
      yo += c.n[i];
    }
    for (int rows : {2048, 8192}) {
      const double fl = 2.0 * rows * N * c.K;
      double best = 1e30;
      int bcfg = 0;
      for (int cfg = 0; cfg < 6; ++cfg) {
        gemm_config(cfg);
        const double us = time_graph(1, [&](int, cudaStream_t s) {
          gemm(x.p, c.ldx, rows, c.K, sg.data(), (int)sg.size(), c.out, c.splitk, s);
        }, 5);
        result("pgemm\t%s\trows %d\tGEMM cfg %d\t%.1f us\t%.1f TFLOPS\t%.1f %% of 107", c.name, rows, cfg, us, fl / us / 1e6,
               fl / us / 1e6 / 1.07);
        if (us < best) best = us, bcfg = cfg;
      }
      gemm_config(-1);
      const double gus = time_graph(1, [&](int, cudaStream_t s) {
        for (int r0 = 0; r0 < rows; r0 += kMaxRows) {
          std::vector<GemvSeg> s2 = sg;
          for (auto& g : s2) g.y = static_cast<char*>(g.y) + (size_t)r0 * g.ldy * (c.out == GemvOut::kF32 ? 4 : 2);
          gemv(true, x.p + (size_t)r0 * c.ldx, c.ldx, kMaxRows, c.K, s2.data(), (int)s2.size(), c.out, c.splitk, s, true);
        }
      }, 3);
      result("pgemm\t%s\trows %d\t16-row GEMVs\t%.1f us\t%.1f TFLOPS\tGEMM best cfg %d %.1f us (%.1fx)", c.name, rows, gus,
             fl / gus / 1e6, bcfg, best, gus / best);
    }
  }
}
// ---- altar-1's projections: BF16 and INT4 GEMVs (decode) and GEMMs (prefill) -----------------------------------
struct ACase {
  const char* name;
  std::vector<std::pair<WFmt, int>> seg;  // (format, rows) a segment
  int K, group_rows, x_group, ldx;
  GemvOut out;
  int splitk;
};
const std::vector<ACase>& altar_cases() {
  static const std::vector<ACase> cs = {
      {"x-projection BF16 688x6144", {{WFmt::kBF16, kXRows}}, kDim, 0, 0, kDim, GemvOut::kBF16, 4},
      {"x-projection BF16, shared layer 656x6144", {{WFmt::kBF16, kXRowsShared}}, kDim, 0, 0, kDim, GemvOut::kBF16, 4},
      {"q_b INT4 4096x2048", {{WFmt::kINT4, kOProjK}}, kQLora, 0, 0, kQLora, GemvOut::kBF16, 1},
      {"indexer wq_b BF16 1024x2048", {{WFmt::kBF16, 1024}}, kQLora, 0, 0, kQLora, GemvOut::kBF16, 1},
      {"q_b INT4 + wq_b BF16 (one call, two launches)", {{WFmt::kINT4, kOProjK}, {WFmt::kBF16, 1024}}, kQLora, 0, 0, kQLora,
       GemvOut::kBF16, 1},
      {"W_UK^T BF16 16x(512x192) grouped", {{WFmt::kBF16, kHeads * kKvLora}}, kNope, kKvLora, kQkDim, kOProjK, GemvOut::kBF16, 1},
      {"W_UV INT4 16x(256x512) grouped", {{WFmt::kINT4, kOProjK}}, kKvLora, kVDim, kKvLora, kHeads * kKvLora, GemvOut::kBF16, 1},
      {"o_proj INT4 6144x4096 (f32)", {{WFmt::kINT4, kDim}}, kOProjK, 0, 0, kOProjK, GemvOut::kF32, 1},
      {"q_b BF16 4096x2048 (BF16 layers)", {{WFmt::kBF16, kOProjK}}, kQLora, 0, 0, kQLora, GemvOut::kBF16, 1},
      {"q_b BF16 + wq_b BF16 (BF16 layers)", {{WFmt::kBF16, kOProjK}, {WFmt::kBF16, 1024}}, kQLora, 0, 0, kQLora, GemvOut::kBF16, 1},
      {"W_UV BF16 grouped (BF16 layers)", {{WFmt::kBF16, kOProjK}}, kKvLora, kVDim, kKvLora, kHeads * kKvLora, GemvOut::kBF16, 1},
      {"o_proj BF16 6144x4096 (f32, BF16 layers)", {{WFmt::kBF16, kDim}}, kOProjK, 0, 0, kOProjK, GemvOut::kF32, 1},
      {"dense gate+up BF16 6144x6144 (SwiGLU)", {{WFmt::kBF16, 2 * kDenseInterRank}}, kDim, 0, 0, kDim, GemvOut::kSwiGLU, 1},
      {"dense down BF16 6144x3072 (f32)", {{WFmt::kBF16, kDim}}, kDenseInterRank, 0, 0, kDenseInterRank, GemvOut::kF32, 1},
      {"shared gate+up BF16 1024x6144 (SwiGLU)", {{WFmt::kBF16, 2 * kSharedInterRank}}, kDim, 0, 0, kDim, GemvOut::kSwiGLU, 4},
      {"shared down BF16 6144x512 (f32)", {{WFmt::kBF16, kDim}}, kSharedInterRank, 0, 0, kSharedInterRank, GemvOut::kF32, 1},
      {"head BF16 38720x6144", {{WFmt::kBF16, kVocabRank}}, kDim, 0, 0, kDim, GemvOut::kBF16, 1},
      // the same BF16 shapes in the GEMV's lane order (kBF16T)
      {"x-projection BF16T 688x6144", {{WFmt::kBF16T, kXRows}}, kDim, 0, 0, kDim, GemvOut::kBF16, 4},
      {"indexer wq_b BF16T 1024x2048", {{WFmt::kBF16T, 1024}}, kQLora, 0, 0, kQLora, GemvOut::kBF16, 1},
      {"q_b INT4 + wq_b BF16T (one call, two launches)", {{WFmt::kINT4, kOProjK}, {WFmt::kBF16T, 1024}}, kQLora, 0, 0,
       kQLora, GemvOut::kBF16, 1},
      {"W_UK^T BF16T 16x(512x192) grouped", {{WFmt::kBF16T, kHeads * kKvLora}}, kNope, kKvLora, kQkDim, kOProjK,
       GemvOut::kBF16, 1},
      {"q_b BF16T + wq_b BF16T (BF16 layers)", {{WFmt::kBF16T, kOProjK}, {WFmt::kBF16T, 1024}}, kQLora, 0, 0, kQLora,
       GemvOut::kBF16, 1},
      {"W_UV BF16T grouped (BF16 layers)", {{WFmt::kBF16T, kOProjK}}, kKvLora, kVDim, kKvLora, kHeads * kKvLora,
       GemvOut::kBF16, 1},
      {"o_proj BF16T 6144x4096 (f32, BF16 layers)", {{WFmt::kBF16T, kDim}}, kOProjK, 0, 0, kOProjK, GemvOut::kF32, 1},
      {"dense gate+up BF16T 6144x6144 (SwiGLU)", {{WFmt::kBF16T, 2 * kDenseInterRank}}, kDim, 0, 0, kDim,
       GemvOut::kSwiGLU, 1},
      {"dense down BF16T 6144x3072 (f32)", {{WFmt::kBF16T, kDim}}, kDenseInterRank, 0, 0, kDenseInterRank, GemvOut::kF32, 1},
      {"shared gate+up BF16T 1024x6144 (SwiGLU)", {{WFmt::kBF16T, 2 * kSharedInterRank}}, kDim, 0, 0, kDim,
       GemvOut::kSwiGLU, 4},
      {"shared down BF16T 6144x512 (f32)", {{WFmt::kBF16T, kDim}}, kSharedInterRank, 0, 0, kSharedInterRank,
       GemvOut::kF32, 1},
      {"head BF16T 38720x6144", {{WFmt::kBF16T, kVocabRank}}, kDim, 0, 0, kDim, GemvOut::kBF16, 1},
  };
  return cs;
}
size_t acase_bytes(const ACase& c) {
  size_t b = 0;
  for (auto [f, n] : c.seg) b += weight_bytes(f, n, c.K);
  return b;
}
// a step's segments over pool slot l (weights back to back); y shared
std::vector<GemvSeg> acase_segs(const ACase& c, const uint8_t* pool, size_t per, int l, uint8_t* y, int N) {
  std::vector<GemvSeg> sg(c.seg.size());
  size_t wo = 0, yo = 0;
  for (size_t i = 0; i < c.seg.size(); ++i) {
    sg[i].fmt = c.seg[i].first;
    sg[i].w = pool + per * l + wo;
    sg[i].n = c.seg[i].second;
    sg[i].group_rows = c.group_rows;
    sg[i].x_group = c.x_group;
    sg[i].ldy = c.out == GemvOut::kSwiGLU ? N / 2 : N;
    sg[i].y = y + yo * (c.out == GemvOut::kF32 ? 4 : 2);
    wo += weight_bytes(c.seg[i].first, c.seg[i].second, c.K);
    yo += c.seg[i].second;
  }
  return sg;
}
// decode: every case at 1, 6, 16 rows (PDL chain; 1 row also plain), then the launch shapes at 1, 6, 16 rows
void bench_agemv(bool sweep) {
  Dev<uint16_t> x(16 * 16384);
  fill(x.p, x.n * 2, 7, 1);
  Dev<uint8_t> y(16 * 40000 * 4);
  for (const auto& c : altar_cases()) {
    const size_t per = acase_bytes(c);
    int N = 0;
    for (auto& s : c.seg) N += s.second;
    const int L = (int)std::max<size_t>(4, ((size_t)768 << 20) / per);
    Dev<uint8_t> pool(per * L);
    fill(pool.p, per * L, 11, 1);  // bf16 pairs: valid values, scales and nibbles alike
    auto run = [&](int rows, bool pdl) {
      return time_graph(L, [&](int l, cudaStream_t s) {
        auto sg = acase_segs(c, pool.p, per, l, y.p, N);
        gemv(x.p, c.ldx, rows, c.K, sg.data(), (int)sg.size(), c.out, c.splitk, s, pdl);
      });
    };
    for (int rows : {1, 6, 16})
      for (int pdl = 0; pdl < 2; ++pdl) {
        if (!pdl && rows != 1) continue;
        const double us = run(rows, pdl != 0);
        result("agemv\t%s\t%s\trows %d\t%.2f MB\t%.1f us\t%.1f GB/s\t%.1f %% of 235", c.name, pdl ? "PDL chain" : "plain", rows,
               per / 1e6, us, per / us / 1e3, per / us / 1e3 / 2.35);
      }
    if (!sweep || c.seg.size() > 1) continue;
    for (int cfg = 0; cfg < 6; ++cfg)
      for (int rows : {1, 6, 16}) {
        gemv_config(cfg);
        const double us = run(rows, true);
        result("agemvcfg\t%s\tcfg %d\trows %d\t%.1f us\t%.1f %% of 235", c.name, cfg, rows, us, per / us / 1e3 / 2.35);
      }
    gemv_config(-1);
  }
  // the BF16 x-projection and the index weights: two launches against one
  {
    const size_t wb = (size_t)kXRows * kDim * 2, ib = (size_t)kIdxHeadsRank * kDim * 2, per = wb + ib;
    const int L = (int)std::max<size_t>(4, ((size_t)768 << 20) / per);
    Dev<uint8_t> pool(per * L);
    fill(pool.p, per * L, 11, 1);
    Dev<float> w(16 * kIdxHeadsRank);
    for (int fused = 0; fused < 2; ++fused)
      for (int rows : {1, 6, 16}) {
        const double us = time_graph(L, [&](int l, cudaStream_t s) {
          GemvSeg sg;
          sg.fmt = WFmt::kBF16;
          sg.w = pool.p + per * l;
          sg.n = kXRows;
          sg.y = y.p;
          sg.ldy = 40000;
          const uint16_t* wp = reinterpret_cast<const uint16_t*>(pool.p + per * l + wb);
          if (fused) {
            GemvIw iw;
            iw.wproj = wp, iw.nh = kIdxHeadsRank, iw.w = w.p, iw.ldw = kIdxHeadsRank;
            gemv(x.p, kDim, rows, kDim, &sg, 1, GemvOut::kBF16, 4, s, true, &iw);
          } else {
            gemv(x.p, kDim, rows, kDim, &sg, 1, GemvOut::kBF16, 4, s, true);
            index_weights(x.p, rows, wp, kIdxHeadsRank, w.p, kIdxHeadsRank, s, true);
          }
        });
        result("agemv\tx-projection BF16 + index weights_proj (%s)\tPDL chain\trows %d\t%.2f MB\t%.1f us\t%.1f %% of 235",
               fused ? "one launch" : "two launches", rows, per / 1e6, us, per / us / 1e3 / 2.35);
      }
  }
}

// prefill: the GEMMs at 2,600 and 8,192 rows (INT4: the dequantisation into the scratch included, and alone)
void bench_apgemm(bool rsweep) {
  const int RMAX = 8192;
  Dev<uint16_t> x((size_t)RMAX * kHeads * kKvLora);
  fill(x.p, x.n * 2, 7, 1);
  Dev<uint8_t> y((size_t)2600 * kVocabRank * 2);  // the largest output: the head at 2,600 rows (bf16)
  const size_t scratch_elems = (size_t)kDim * kOProjK;
  Dev<uint16_t> scratch(scratch_elems);
  for (const auto& c : altar_cases()) {
    const size_t per = acase_bytes(c);
    int N = 0;
    bool int4 = false;
    for (auto& s : c.seg) N += s.second, int4 = int4 || s.first == WFmt::kINT4;
    Dev<uint8_t> w(per);
    fill(w.p, per, 11, 1);
    auto sg = acase_segs(c, w.p, per, 0, y.p, N);
    for (int rows : {2600, 8192}) {
      const size_t es = c.out == GemvOut::kF32 ? 4 : 2;
      if ((size_t)rows * (c.out == GemvOut::kSwiGLU ? N / 2 : N) * es > y.n) continue;
      const double fl = 2.0 * rows * N * c.K;
      const double us = time_graph(1, [&](int, cudaStream_t s) {
        gemm(x.p, c.ldx, rows, c.K, sg.data(), (int)sg.size(), c.out, c.splitk, s, false, scratch.p, scratch_elems);
      }, 5);
      result("apgemm\t%s\trows %d\t%.1f us\t%.1f TFLOPS\t%.1f %% of 107", c.name, rows, us, fl / us / 1e6, fl / us / 1e6 / 1.07);
      if (!rsweep) continue;
      for (int gm : {0, 2, 4, 8, 16, 16, 8, 4, 2, 0}) {  // the raster, up and back down (A/B in both orders)
        gemm_raster(gm);
        const double ur = time_graph(1, [&](int, cudaStream_t s) {
          gemm(x.p, c.ldx, rows, c.K, sg.data(), (int)sg.size(), c.out, c.splitk, s, false, scratch.p, scratch_elems);
        }, 5);
        result("apgemmr\t%s\trows %d\traster %d\t%.1f us\t%.1f TFLOPS\t%.1f %% of 107", c.name, rows, gm, ur, fl / ur / 1e6,
               fl / ur / 1e6 / 1.07);
      }
      gemm_raster(-1);
    }
    if (int4)
      for (auto& g : sg) {
        if (g.fmt != WFmt::kINT4) continue;
        const double us = time_graph(1, [&](int, cudaStream_t s) { dequant_int4(g.w, g.n, c.K, scratch.p, c.K, s); }, 5);
        const double b = weight_bytes(WFmt::kINT4, g.n, c.K) + (double)g.n * c.K * 2;
        result("apgemm\t%s\tdequant_int4 alone [%d][%d]\t%.1f us\t%.1f GB/s (read + write)", c.name, g.n, c.K, us, b / us / 1e3);
      }
  }
}


// ---- the indexer at decode ------------------------------------------------------------------------------------
void bench_index(const std::vector<int>& ctxs) {
  std::vector<uint32_t> rope_h((size_t)(ctxs.back() + 64) * 32, 0x3F800000u >> 16);  // cos 1, sin 0 (timing only)
  Dev<uint32_t> rope(rope_h);
  for (int T : ctxs) {
    for (bool f8 : {false, true}) {
      const size_t ent = f8 ? kIdxDim : kIdxDim * 2;
      const size_t per = (size_t)T * ent;
      const int L = (int)std::max<size_t>(3, ((size_t)256 << 20) / per);
      Dev<uint8_t> keys(per * L);
      Dev<float> ksc(f8 ? (size_t)T * L : 1);
      fill(keys.p, per * L, 5, f8 ? 0 : 1);
      if (f8) fill(ksc.p, (size_t)T * L * 4, 9, 2);
      for (int rows : {1, 6}) {
        std::vector<int> hp(rows);
        for (int i = 0; i < rows; ++i) hp[i] = T - rows + i;
        Dev<int> pos(hp);
        Dev<uint16_t> q((size_t)rows * kIdxHeads * kIdxDim);
        fill(q.p, q.n * 2, 17, 1);
        Dev<float> w((size_t)rows * kIdxHeads);
        fill(w.p, w.n * 4, 19, 2);
        const size_t ld = (size_t)(T + 1023) / 1024 * 1024;
        Dev<float> scores((size_t)rows * ld);
        Dev<uint32_t> work(index_scratch_words(rows));
        Dev<int> tk((size_t)rows * kTopK), cp((size_t)rows * kTP * kTopK);
        Dev<uint32_t> ck((size_t)rows * kTP * kTopK);
        auto ik_at = [&](int l) {
          IdxCache ik;
          ik.k = keys.p + per * l;
          ik.ks = f8 ? ksc.p + (size_t)T * l : nullptr;
          ik.fp8 = f8;
          return ik;
        };
        const double rep = time_graph(L, [&](int l, cudaStream_t s) {
          index_select(IdxQuery{q.p, w.p}, pos.p, nullptr, rows, rope.p, ik_at(l), 0, 0, IdxScratch{scores.p, ld, work.p}, tk.p,
                       nullptr, s);
        });
        for (int v = 0; v < 5; ++v) {  // the scoring kernel's key-load variants
          index_score_config(v);
          const double us = time_graph(L, [&](int l, cudaStream_t s) {
            index_select(IdxQuery{q.p, w.p}, pos.p, nullptr, rows, rope.p, ik_at(l), 0, 0, IdxScratch{scores.p, ld, work.p},
                         tk.p, nullptr, s);
          });
          index_score_config(-1);
          result("index	ctx %d	%s keys	rows %d	key loads %d	replicated %.1f us (%.1f GB/s of keys)", T, f8 ? "FP8" : "BF16",
                 rows, v, us, per / us / 1e3);
        }
        // a rank's quarter of the positions (the last quarter: every quarter has the same work), then the merge
        const int part = (T + kTP - 1) / kTP;
        const size_t ldp = (size_t)(part + 1023) / 1024 * 1024;
        const double quarter = time_graph(L, [&](int l, cudaStream_t s) {
          index_select(IdxQuery{q.p, w.p}, pos.p, nullptr, rows, rope.p, ik_at(l), (kTP - 1) * part, kTP * part,
                       IdxScratch{scores.p, ldp, work.p}, cp.p, ck.p, s);
        });
        const double merge = time_graph(16, [&](int, cudaStream_t s) { index_merge(cp.p, ck.p, pos.p, rows, tk.p, s); });
        result("index\tctx %d\t%s keys\trows %d\treplicated %.1f us (%.1f GB/s of keys)\tquarter %.1f us + merge %.1f us",
               T, f8 ? "FP8" : "BF16", rows, rep, per / rep / 1e3, quarter, merge);
      }
    }
  }
}

// ---- attention at decode ---------------------------------------------------------------------------------------
void bench_attn(const std::vector<int>& ctxs) {
  std::vector<uint32_t> rope_h((size_t)(ctxs.back() + 64) * 32, 0x3F80u);
  Dev<uint32_t> rope(rope_h);
  std::mt19937 rng(1);
  for (int T : ctxs) {
    const size_t per = (size_t)T * kLat * 2;
    const int L = (int)std::max<size_t>(2, ((size_t)600 << 20) / per);
    Dev<uint16_t> cache(per / 2 * L);
    fill(cache.p, per * L, 23, 1);
    for (int rows : {1, 6}) {
      std::vector<int> hp(rows), ht((size_t)rows * kTopK);
      for (int i = 0; i < rows; ++i) {
        hp[i] = T - rows + i;
        std::vector<int> all(hp[i] + 1);
        for (int j = 0; j <= hp[i]; ++j) all[j] = j;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + kTopK);
        std::copy(all.begin(), all.begin() + kTopK, ht.begin() + (size_t)i * kTopK);
      }
      Dev<int> pos(hp), tk(ht);
      Dev<uint16_t> ql((size_t)rows * kHeads * kKvLora), q((size_t)rows * kHeads * kQkDim), u((size_t)rows * kHeads * kKvLora);
      fill(ql.p, ql.n * 2, 29, 1);
      fill(q.p, q.n * 2, 31, 1);
      Dev<float> so((size_t)rows * 32 * kHeads * kKvLora), sml((size_t)rows * 32 * kHeads * 2);
      for (int splits : {8, 16, 32}) {
        const double us = time_graph(L * 4, [&](int l, cudaStream_t s) {
          KvCache kv;
          kv.kv = cache.p + per / 2 * (l % L);
          sparse_attn(ql.p, q.p, pos.p, nullptr, rope.p, kv, tk.p, rows, splits, AttnScratch{so.p, sml.p, 32}, u.p, s);
        });
        result("attn\tctx %d\trows %d\tsplits %d\t%.1f us\t(%.2f MB of entries gathered, %.1f GB/s)", T, rows, splits, us,
               rows * kTopK * kLat * 2 / 1e6, rows * kTopK * kLat * 2 / us / 1e3);
      }
    }
  }
}

// ---- prefill: a chunk of rows -----------------------------------------------------------------------------------
void bench_prefill(const std::vector<int>& ctxs) {
  const int R = 2048;  // rows a chunk
  std::vector<uint32_t> rope_h((size_t)(ctxs.back() + 64) * 32, 0x3F80u);
  Dev<uint32_t> rope(rope_h);
  for (int T : ctxs) {
    Dev<uint16_t> keys((size_t)T * kIdxDim), cache((size_t)T * kLat);
    fill(keys.p, keys.n * 2, 41, 1);
    fill(cache.p, cache.n * 2, 43, 1);
    std::vector<int> hp(R);
    for (int i = 0; i < R; ++i) hp[i] = T - R + i;
    Dev<int> pos(hp);
    Dev<uint16_t> q((size_t)R * kIdxHeads * kIdxDim), ql((size_t)R * kHeads * kKvLora), qa((size_t)R * kHeads * kQkDim),
        u((size_t)R * kHeads * kKvLora);
    fill(q.p, q.n * 2, 47, 1);
    fill(ql.p, ql.n * 2, 53, 1);
    fill(qa.p, qa.n * 2, 59, 1);
    Dev<float> w((size_t)R * kIdxHeads);
    fill(w.p, w.n * 4, 61, 2);
    const size_t ld = (size_t)(T + 1023) / 1024 * 1024;
    Dev<int> tk((size_t)R * kTopK);
    IdxCache ik;
    ik.k = reinterpret_cast<uint8_t*>(keys.p);
    double ius = 0;
    for (int sub : {512, 2048}) {  // rows a selection call (scores [sub][T] f32)
      Dev<float> scores((size_t)sub * ld);
      Dev<uint32_t> work(index_scratch_words(sub));
      const double us = time_graph(1, [&](int, cudaStream_t s) {
        for (int a = 0; a < R; a += sub)
          index_select(IdxQuery{q.p + (size_t)a * kIdxHeads * kIdxDim, w.p + (size_t)a * kIdxHeads}, pos.p + a, nullptr, sub, rope.p,
                       ik, 0, 0, IdxScratch{scores.p, ld, work.p}, tk.p + (size_t)a * kTopK, nullptr, s);
      }, 3);
      result("prefill\tchunk %d rows ending at %d\tindexer, calls of %d rows\t%.2f ms\t(%.1f TFLOPS of score MMAs)", R, T, sub,
             us / 1e3, 2.0 * R * (T - R / 2.0) * kIdxHeads * kIdxDim / us / 1e6);
      if (sub == 512) ius = us;
    }
    KvCache kv;
    kv.kv = cache.p;
    const double aus = time_graph(1, [&](int, cudaStream_t s) {
      sparse_attn(ql.p, qa.p, pos.p, nullptr, rope.p, kv, tk.p, R, 1, AttnScratch{}, u.p, s);
    }, 3);
    const double flops_i = 2.0 * R * (T - R / 2.0) * kIdxHeads * kIdxDim, flops_a = 2.0 * R * kTopK * kHeads * (kLat + kKvLora);
    result("prefill\tchunk %d rows ending at %d\tindexer %.2f ms (%.1f TFLOPS)\tattention %.2f ms (%.1f TFLOPS)", R, T, ius / 1e3,
           flops_i / ius / 1e6, aus / 1e3, flops_a / aus / 1e6);
  }
}

// ---- prefill attention at the runner's shape: a chunk of 8,192 rows at 24,576 over a 32,768-entry cache ------
// The gather alone (cp.async of whole 64-key blocks into shared memory, CTAs persistent over the rows) and
// sparse_attn, for three key patterns: "rand" (each row a sorted random 2048 of 0..p, as the indexer picks on random
// data: the whole 37.7 MB cache in play, over L2's 24 MB), "win" (a random 2048 of the last 4,096 positions: 4.7 MB
// in play) and "same" (every row the first row's set); and the dense rows (positions 0..2047).
__global__ void k_gather(const uint16_t* cache, const int* topk, int rows, int kpi, int* sink) {
  extern __shared__ __align__(16) uint8_t gsm[];
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    const int* tk = topk + (size_t)r * kTopK;
    for (int b = 0; b < kTopK; b += kpi) {
      __syncthreads();
      for (int i = threadIdx.x; i < kpi * (kLat / 8); i += blockDim.x) {
        const int key = i / (kLat / 8), part = i % (kLat / 8);
        glm::cp_async16(gsm + (size_t)i * 16, cache + (size_t)tk[b + key] * kLat + part * 8);
      }
      glm::cp_async_wait_all();
      __syncthreads();
    }
  }
  if (threadIdx.x == 0 && gsm[5] == 123 && gsm[77] == 1) sink[0] = 1;
}

// The gather alone as a ring of NS slots of 16 keys a CTA (one CTA an SM), each slot regathered the moment it lands:
// the gather rate at NS x 18.4 KB in flight an SM. TMA: warp 0 alone, one bulk copy a key on the slot's mbarrier;
// otherwise warp s owns slot s and gathers it with cp.async (36 pieces a lane), then waits for its own pieces.
__device__ __forceinline__ uint32_t s32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
template <bool TMA>
__global__ void k_gather_ring(const uint16_t* cache, const int* topk, int rows, int ns, int pfd, int* sink) {
  extern __shared__ __align__(16) uint8_t gsm[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(gsm + (size_t)ns * 16 * kLat * 2);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int per = (rows - blockIdx.x + gridDim.x - 1) / gridDim.x, total = per * (kTopK / 16);
  auto src = [&](int n, int k) {
    const int r = blockIdx.x + (n / (kTopK / 16)) * gridDim.x;
    return cache + (size_t)topk[(size_t)r * kTopK + (n % (kTopK / 16)) * 16 + k] * kLat;
  };
  if (TMA) {
    if (warp) return;
    if (lane == 0)
      for (int s = 0; s < ns; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(s32(bar + s)) : "memory");
    __syncwarp();
    auto issue = [&](int n) {
      const int s = n % ns;
      if (lane == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(s32(bar + s)), "r"(16 * kLat * 2) : "memory");
      __syncwarp();
      if (lane < 16)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                         s32(gsm + ((size_t)s * 16 + lane) * kLat * 2)), "l"(src(n, lane)), "r"(kLat * 2), "r"(s32(bar + s))
                     : "memory");
    };
    for (int n = 0; n < ns && n < total; ++n) issue(n);
    for (int n = 0; n < total; ++n) {
      const uint32_t par = (n / ns) & 1;
      asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" ::"r"(
                       s32(bar + n % ns)), "r"(par) : "memory");
      if (n + ns < total) issue(n + ns);
    }
  } else {
    if (warp >= ns) return;
    for (int n = warp; n < total; n += ns) {
      if (pfd && n + pfd < total)  // L2 prefetch of the slice pfd ahead: 16 keys x 9 lines of 128 bytes
        for (int i = lane; i < 16 * 9; i += 32)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(src(n + pfd, i / 9) + (i % 9) * 64));
      for (int i = lane; i < 16 * (kLat / 8); i += 32)
        glm::cp_async16(gsm + ((size_t)warp * 16 * kLat + (i / (kLat / 8)) * kLat + (i % (kLat / 8)) * 8) * 2,
                        src(n, i / (kLat / 8)) + (i % (kLat / 8)) * 8);
      glm::cp_async_wait_all();
    }
  }
  if (threadIdx.x == 0 && gsm[5] == 123 && gsm[77] == 1) sink[0] = 1;
}

// a copy kernel to run beside the attention (as the ring's copies run beside the prefill's compute): blocks of 512
// threads bouncing 16-byte words through shared memory, `iters` passes
__global__ void __launch_bounds__(512) k_stress(int4* dst, const int4* src, size_t n, int iters) {
  __shared__ int4 b[2048];
  for (int it = 0; it < iters; ++it)
    for (size_t i = blockIdx.x * (size_t)2048; i < n; i += (size_t)gridDim.x * 2048) {
      for (int j = threadIdx.x; j < 2048; j += 512) b[j] = src[i + j];
      __syncthreads();
      for (int j = threadIdx.x; j < 2048; j += 512) dst[i + j] = b[(j + 512) & 2047];
      __syncthreads();
    }
}

__global__ void k_sleep(long long ns) {
  const long long t0 = clock64();
  while (clock64() - t0 < ns * 2) __nanosleep(1000);  // ~2 GHz
}

void bench_pstress() {
  // the runner's first chunk shape (rows 0..2599 in two launches, a slot table) with copy kernels of 8 blocks on a
  // second stream (low or high priority) arriving while the attention runs (after a sleeping kernel of d us): k_attn
  // once alone, then k_attn and k_attn_pf beside the copies
  const int M = 2600, P1 = 1344, T = 8192;
  const size_t per = (size_t)T * kLat;
  Dev<uint16_t> cache(per);
  fill(cache.p, per * 2, 43, 1);
  std::vector<uint32_t> rope_h((size_t)(T + 64) * 32, 0x3F80u);
  Dev<uint32_t> rope(rope_h);
  Dev<uint16_t> ql((size_t)M * kHeads * kKvLora), qa((size_t)M * kHeads * kQkDim), u((size_t)M * kHeads * kKvLora);
  fill(ql.p, ql.n * 2, 53, 1);
  fill(qa.p, qa.n * 2, 59, 1);
  const size_t sn = (size_t)1 << 20;  // 16 MB each
  Dev<int4> sa(sn), sb(sn);
  std::vector<int> mp(M), ms(M, 1), mt((size_t)M * kTopK, -7);
  std::mt19937 rng2(11);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  for (int i = 0; i < M; ++i) {
    mp[i] = i;
    if (i < kTopK) continue;
    int need = kTopK, left = i + 1, k = 0;
    for (int v = 0; need > 0; ++v, --left)
      if (U(rng2) * left < need) mt[(size_t)i * kTopK + k++] = v, --need;
  }
  Dev<int> dpos(mp), dslot(ms), dtk(mt);
  KvCache kv;
  kv.kv = cache.p;
  kv.slot_stride = 4096;
  int lo = 0, hi = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  cudaStream_t s1, s2[2];
  CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithPriority(&s2[0], cudaStreamNonBlocking, lo));
  CUDA_CHECK(cudaStreamCreateWithPriority(&s2[1], cudaStreamNonBlocking, hi));
  std::vector<uint16_t> ref;
  auto run = [&](int cfg, int pri, int d) {
    sparse_attn_config(cfg);
    u.zero();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (d >= 0) {
      k_sleep<<<1, 32, 0, s2[pri]>>>((long long)d * 1000);
      for (int i = 0; i < 40; ++i) k_stress<<<8, 512, 0, s2[pri]>>>(sb.p, sa.p, sn, 1);
      CUDA_CHECK(cudaGetLastError());
    }
    for (int pc = 0; pc < 2; ++pc) {
      const int a0 = pc ? P1 : 0, n = pc ? M - P1 : P1;
      sparse_attn(ql.p + (size_t)a0 * kHeads * kKvLora, qa.p + (size_t)a0 * kHeads * kQkDim, dpos.p + a0, dslot.p + a0, rope.p,
                  kv, dtk.p + (size_t)a0 * kTopK, n, 1, AttnScratch{}, u.p + (size_t)a0 * kHeads * kKvLora, s1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return u.down((size_t)M * kHeads * kKvLora);
  };
  ref = run(0, 0, -1);
  for (int pri : {0, 1})
    for (int cfg : {0, -1}) {
      int same = 0, runs = 0;
      for (int d : {0, 200, 500, 1000, 2000, 3000})
        for (int rep = 0; rep < 2; ++rep) {
          const std::vector<uint16_t> got = run(cfg, pri, d);
          ++runs;
          if (got == ref) { ++same; continue; }
          size_t first = 0, nd = 0, last = 0;
          for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != ref[i]) { if (!nd) first = i; last = i; ++nd; }
          result("pstress\t%s %s priority, copies after %d us: %zu values differ, rows %zu..%zu (head %zu dim %zu first)",
                 cfg ? "k_attn_pf" : "k_attn", pri ? "high" : "low", d, nd, first / (kHeads * kKvLora), last / (kHeads * kKvLora),
                 first / kKvLora % kHeads, first % kKvLora);
        }
      result("pstress\t%s beside 40 copy kernels of 8 blocks (%s priority): the same bits as k_attn alone in %d of %d runs",
             cfg ? "k_attn_pf" : "k_attn", pri ? "high" : "low", same, runs);
    }
  sparse_attn_config(-1);
  CUDA_CHECK(cudaStreamDestroy(s1));
  CUDA_CHECK(cudaStreamDestroy(s2[0]));
  CUDA_CHECK(cudaStreamDestroy(s2[1]));
}

void bench_pattn() {
  const int T = 32768, R = 8192, P0 = T - R;
  const size_t per = (size_t)T * kLat;
  Dev<uint16_t> cache(per);
  fill(cache.p, per * 2, 43, 1);
  std::vector<uint32_t> rope_h((size_t)(T + 64) * 32, 0x3F80u);
  Dev<uint32_t> rope(rope_h);
  Dev<uint16_t> ql((size_t)R * kHeads * kKvLora), qa((size_t)R * kHeads * kQkDim), u((size_t)R * kHeads * kKvLora);
  fill(ql.p, ql.n * 2, 53, 1);
  fill(qa.p, qa.n * 2, 59, 1);
  Dev<int> sink(1);
  std::mt19937 rng(7);
  std::vector<int> hp(R), ht((size_t)R * kTopK);
  for (int i = 0; i < R; ++i) hp[i] = P0 + i;
  Dev<int> pos(hp);
  const double flops = 2.0 * R * kTopK * kHeads * (kLat + kKvLora), bytes = (double)R * kTopK * kLat * 2;
  for (const char* pat : {"rand", "win", "same"}) {
    for (int i = 0; i < R; ++i) {
      int* d = ht.data() + (size_t)i * kTopK;
      const std::string p(pat);
      if (p == "same" && i > 0) { std::copy(ht.begin(), ht.begin() + kTopK, d); continue; }
      const int lo = p == "win" ? hp[i] - 4095 : 0;
      // selection sampling (Knuth's algorithm S): kTopK of [lo, p], ascending
      int need = kTopK, left = hp[i] + 1 - lo, k = 0;
      std::uniform_real_distribution<double> U(0.0, 1.0);
      for (int v = lo; need > 0; ++v, --left)
        if (U(rng) * left < need) d[k++] = v, --need;
    }
    Dev<int> tk(ht);
    KvCache kv;
    kv.kv = cache.p;
    std::vector<uint16_t> ref;
    for (int cfg : {0, -1, 0, -1}) {  // k_attn at one split, k_attn_pf; A B A B
      sparse_attn_config(cfg);
      u.zero();
      const double aus = time_graph(1, [&](int, cudaStream_t s) {
        sparse_attn(ql.p, qa.p, pos.p, nullptr, rope.p, kv, tk.p, R, 1, AttnScratch{}, u.p, s);
      }, 3);
      const std::vector<uint16_t> got = u.down();
      if (ref.empty()) ref = got;
      result("pattn\t%s\tsparse_attn %s %d rows at %d\t%.2f ms\t(%.1f TFLOPS, %.0f GB/s of entries gathered)\t%s", pat,
             cfg ? "k_attn_pf" : "k_attn", R, P0, aus / 1e3, flops / aus / 1e6, bytes / aus / 1e3,
             got == ref ? "bits = k_attn" : "BITS DIFFER");
    }
    sparse_attn_config(-1);
    for (int ns : {4}) {
      const size_t sm = (size_t)ns * 16 * kLat * 2 + 64;
      CUDA_CHECK(cudaFuncSetAttribute(k_gather_ring<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
      CUDA_CHECK(cudaFuncSetAttribute(k_gather_ring<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
      const double t1 = time_graph(1, [&](int, cudaStream_t s) { k_gather_ring<true><<<48, 32, sm, s>>>(cache.p, tk.p, R, ns, 0, sink.p); }, 3);
      const double t2 = time_graph(1, [&](int, cudaStream_t s) { k_gather_ring<false><<<48, 32 * ns, sm, s>>>(cache.p, tk.p, R, ns, 0, sink.p); }, 3);
      result("pattn\t%s\tgather ring of %d slots of 16 keys, 48 CTAs: bulk copies %.2f ms (%.0f GB/s), cp.async %.2f ms (%.0f GB/s)", pat, ns,
             t1 / 1e3, bytes / t1 / 1e3, t2 / 1e3, bytes / t2 / 1e3);
    }
    for (int cfg = 0; cfg < (std::string(pat) == "rand" ? 3 : 0); ++cfg) {
      const int kpi = 64 >> cfg, ctas = 48 << cfg;
      const size_t sm = (size_t)kpi * kLat * 2;
      CUDA_CHECK(cudaFuncSetAttribute(k_gather, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
      const double gus = time_graph(1, [&](int, cudaStream_t s) {
        k_gather<<<ctas, 128, sm, s>>>(cache.p, tk.p, R, kpi, sink.p);
      }, 3);
      result("pattn\t%s\tgather only, %d keys a step, %d CTAs of 128\t%.2f ms\t(%.0f GB/s)", pat, kpi, ctas, gus / 1e3, bytes / gus / 1e3);
    }
  }
  {  // dense rows 0..2047
    std::vector<int> dp(2048);
    for (int i = 0; i < 2048; ++i) dp[i] = i;
    Dev<int> dpos(dp);
    KvCache kv;
    kv.kv = cache.p;
    std::vector<uint16_t> ref;
    for (int cfg : {0, -1, 0, -1}) {
      sparse_attn_config(cfg);
      u.zero();
      const double aus = time_graph(1, [&](int, cudaStream_t s) {
        sparse_attn(ql.p, qa.p, dpos.p, nullptr, rope.p, kv, nullptr, 2048, 1, AttnScratch{}, u.p, s);
      }, 3);
      const std::vector<uint16_t> got = u.down((size_t)2048 * kHeads * kKvLora);
      if (ref.empty()) ref = got;
      const double df = 2.0 * (2048.0 * 2049 / 2) * kHeads * (kLat + kKvLora);
      result("pattn\tdense\tsparse_attn %s rows 0..2047\t%.2f ms\t(%.1f TFLOPS of the causal work)\t%s",
             cfg ? "k_attn_pf" : "k_attn", aus / 1e3, df / aus / 1e6, got == ref ? "bits = k_attn" : "BITS DIFFER");
    }
    sparse_attn_config(-1);
  }
  {  // the runner's first chunk: rows 0..2599 (dense below 2048, then random selections), a slot table, in two
     // launches as the two pieces (0..1343, 1344..2599); k_attn once, then k_attn_pf 6 times: the same bits each time?
    const int M = 2600, P1 = 1344;
    std::vector<int> mp(M), ms(M, 1), mt((size_t)M * kTopK, -7);
    std::mt19937 rng2(11);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int i = 0; i < M; ++i) {
      mp[i] = i;
      if (i < kTopK) continue;
      int need = kTopK, left = i + 1, k = 0;
      for (int v = 0; need > 0; ++v, --left)
        if (U(rng2) * left < need) mt[(size_t)i * kTopK + k++] = v, --need;
    }
    Dev<int> dpos(mp), dslot(ms), dtk(mt);
    KvCache kv;
    kv.kv = cache.p;
    kv.slot_stride = 4096;  // slot 1 at entries 4096..
    std::vector<uint16_t> ref;
    int same = 0;
    for (int it = 0; it < 7; ++it) {
      sparse_attn_config(it ? -1 : 0);
      u.zero();
      for (int pc = 0; pc < 2; ++pc) {
        const int a0 = pc ? P1 : 0, n = pc ? M - P1 : P1;
        sparse_attn(ql.p + (size_t)a0 * kHeads * kKvLora, qa.p + (size_t)a0 * kHeads * kQkDim, dpos.p + a0, dslot.p + a0, rope.p,
                    kv, dtk.p + (size_t)a0 * kTopK, n, 1, AttnScratch{}, u.p + (size_t)a0 * kHeads * kKvLora, 0);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<uint16_t> got = u.down((size_t)M * kHeads * kKvLora);
      if (ref.empty()) ref = got;
      else if (got == ref) ++same;
      else {
        size_t first = 0, nd = 0;
        for (size_t i = 0; i < got.size(); ++i)
          if (got[i] != ref[i]) { if (!nd) first = i; ++nd; }
        result("pattn\tchunk0\trun %d: %zu values differ, the first at row %zu head %zu dim %zu", it, nd, first / (kHeads * kKvLora),
               first / kKvLora % kHeads, first % kKvLora);
      }
    }
    sparse_attn_config(-1);
    result("pattn\tchunk0\tk_attn_pf = k_attn in %d of 6 runs (rows 0..2599 in two launches, a slot table)", same);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string only = "bw,agemv,apgemm,index,attn,prefill", out;  // glm53 FP8 gemv/pgemm, agemvcfg, pattn: when asked
  std::vector<int> ctxs = {32768, 131072, 262144};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--ctx" && i + 1 < argc) {
      ctxs.clear();
      std::string s = argv[++i];
      size_t p = 0;
      while (p < s.size()) { ctxs.push_back(std::stoi(s.substr(p))); p = s.find(',', p); if (p == std::string::npos) break; ++p; }
    } else die("usage: bench_attn [--out FILE] [--only bw,agemv,agemvcfg,apgemm,gemv,index,attn,prefill,pgemm,pattn] [--ctx a,b,c]");
  }
  if (!out.empty()) g_out = fopen(out.c_str(), "a");
  cudaDeviceProp pr;
  CUDA_CHECK(cudaGetDeviceProperties(&pr, 0));
  result("device\t%s\t%d SMs\tsmem/block optin %zu", pr.name, pr.multiProcessorCount, pr.sharedMemPerBlockOptin);
  auto want = [&](const char* k) { return ("," + only + ",").find("," + std::string(k) + ",") != std::string::npos; };
  if (want("bw")) bench_bw();
  if (want("gemv")) bench_gemv();
  if (want("index")) bench_index(ctxs);
  if (want("attn")) bench_attn(ctxs);
  if (want("prefill")) bench_prefill(ctxs);
  if (want("pgemm")) bench_pgemm();
  if (want("agemv") || want("agemvcfg")) bench_agemv(want("agemvcfg"));
  if (want("apgemm") || want("apgemmr")) bench_apgemm(want("apgemmr"));
  if (want("pattn")) bench_pattn();
  if (want("pstress")) bench_pstress();
  return 0;
}
