// Timings of engine/attn on one GB10 (this rank's shapes at TP=4). Weights and caches are cycled through pools
// several times larger than L2 (24 MB), so every timed launch reads from DRAM, as a step's would. Each figure is
// the median over replays of a CUDA graph of one pass over the pool, divided by the launches in it.
// usage: bench_attn [--out FILE] [--only bw,gemv,index,attn,prefill,pgemm] [--ctx 32768,131072,262144]
#include <algorithm>
#include <cstdarg>
#include <random>
#include <string>
#include <vector>

#include "attn/attn.h"
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

}  // namespace

int main(int argc, char** argv) {
  std::string only = "bw,gemv,index,attn,prefill,pgemm", out;
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
    } else die("usage: bench_attn [--out FILE] [--only bw,gemv,index,attn,prefill,pgemm] [--ctx a,b,c]");
  }
  if (!out.empty()) g_out = fopen(out.c_str(), "a");
  cudaDeviceProp pr;
  CUDA_CHECK(cudaGetDeviceProperties(&pr, 0));
  result("device\t%s\t%d SMs\tsmem/block optin %zu", pr.name, pr.multiProcessorCount, pr.sharedMemPerBlockOptin);
  auto want = [&](const char* k) { return only.find(k) != std::string::npos; };
  if (want("bw")) bench_bw();
  if (want("gemv")) bench_gemv();
  if (want("index")) bench_index(ctxs);
  if (want("attn")) bench_attn(ctxs);
  if (want("prefill")) bench_prefill(ctxs);
  if (want("pgemm")) bench_pgemm();
  return 0;
}
