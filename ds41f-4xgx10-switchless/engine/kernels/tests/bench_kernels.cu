// Bandwidth of the decode kernels on per-rank shapes, with weights cold in L2: every launch reads
// a different copy of the weights (copies total > 256 MB, L2 is 24 MB plus a 16 MB SLC). Launches
// are captured in a CUDA graph; the figure is the median over repeats of (graph time / launches).
// usage: bench_kernels [filter]  (head4: the head coded against the bf16 head, on the real head slice)
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"
#include "kernels/tests/dsw_read.h"
#include "kernels/tests/npy.h"
#include "kernels/tests/small_gemms.h"

static const double kPeak = 235.0;  // GB/s, measured streaming read on rank 0's GB10

__global__ void k_fill(uint8_t* p, size_t n, uint32_t seed, uint8_t lo, uint8_t span) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    uint8_t v = (uint8_t)h;
    p[i] = span ? (uint8_t)(lo + v % span) : (uint8_t)(v & 0x77);  // e4m3 without NaN, |v| < 2^?
  }
}

static uint8_t* dev_fill(size_t n, uint32_t seed, uint8_t lo = 0, uint8_t span = 0) {
  uint8_t* p;
  CUDA_CHECK(cudaMalloc(&p, n));
  k_fill<<<512, 256>>>(p, n, seed, lo, span);
  CUDA_CHECK(cudaGetLastError());
  return p;
}

// Median time (us) per launch of fn(copy) over `copies` rotating copies.
static double time_us(int copies, const std::function<void(int, cudaStream_t)>& fn, int launches = 0) {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  if (!launches) launches = std::max(copies, 40);
  for (int i = 0; i < copies; ++i) fn(i, st);  // warm-up (and lazy attributes)
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < launches; ++i) fn(i % copies, st);
  CUDA_CHECK(cudaStreamEndCapture(st, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<double> ts;
  for (int r = 0; r < 9; ++r) {
    CUDA_CHECK(cudaEventRecord(a, st));
    CUDA_CHECK(cudaGraphLaunch(exec, st));
    CUDA_CHECK(cudaEventRecord(b, st));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms * 1000.0 / launches);
  }
  std::sort(ts.begin(), ts.end());
  cudaGraphExecDestroy(exec), cudaGraphDestroy(graph), cudaEventDestroy(a), cudaEventDestroy(b), cudaStreamDestroy(st);
  return ts[ts.size() / 2];
}

// Several graph-captured functions timed round-robin, one replay each per round, so that other work
// on the shared GPU hits them alike. Returns each one's replay times in us, sorted.
static std::vector<std::vector<double>> time_interleaved(const std::vector<std::function<void(cudaStream_t)>>& fns,
                                                         int rounds) {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  std::vector<cudaGraphExec_t> execs;
  for (auto& fn : fns) {
    fn(st);  // warm-up (and lazy attributes)
    CUDA_CHECK(cudaStreamSynchronize(st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    fn(st);
    CUDA_CHECK(cudaStreamEndCapture(st, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    execs.push_back(exec);
  }
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<std::vector<double>> ts(fns.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t i = 0; i < execs.size(); ++i) {
      CUDA_CHECK(cudaEventRecord(a, st));
      CUDA_CHECK(cudaGraphLaunch(execs[i], st));
      CUDA_CHECK(cudaEventRecord(b, st));
      CUDA_CHECK(cudaEventSynchronize(b));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      ts[i].push_back(ms * 1000.0);
    }
  for (auto& v : ts) std::sort(v.begin(), v.end());
  for (auto e : execs) cudaGraphExecDestroy(e);
  cudaEventDestroy(a), cudaEventDestroy(b), cudaStreamDestroy(st);
  return ts;
}

static int copies_for(size_t bytes) { return (int)std::max<size_t>(4, (256u << 20) / bytes + 1); }

struct Shape {
  const char* name;
  int N, K;
};

static void bench_fp8(const Shape& s, bool sweep) {
  const size_t wb = (size_t)s.N * s.K, sb = (size_t)(s.N / 32) * (s.K / 32), bytes = wb + sb;
  const int copies = copies_for(bytes);
  std::vector<uint8_t*> W(copies), S(copies);
  for (int i = 0; i < copies; ++i) W[i] = dev_fill(wb, 17 + i), S[i] = dev_fill(sb, 99 + i, 118, 16);
  uint8_t* xq = dev_fill(8 * s.K, 5);
  uint8_t* xs = dev_fill(8 * s.K / 32, 6, 118, 16);
  __nv_bfloat16* y;
  CUDA_CHECK(cudaMalloc(&y, 8 * (size_t)s.N * 2));
  printf("%-8s N=%-5d K=%-5d %6.2f MB\n", s.name, s.N, s.K, bytes / 1e6);
  for (int M : {1, 6}) {
    double best = 1e30;
    dsk::GemvCfg bc;
    std::vector<dsk::GemvCfg> cfgs;
    if (sweep) {
      for (int mt : {1, 2, 4})
        for (int wn : {1, 2, 4})
          for (int wk : {1, 2, 4, 8, 16})
            if (wn * wk <= 16 && s.N % (8 * mt * wn) == 0 && (s.K / 64) >= wk) cfgs.push_back({mt, wn, wk});
    } else {
      cfgs.push_back({});
    }
    for (auto c : cfgs) {
      double us = time_us(copies, [&](int i, cudaStream_t st) { dsk::gemv_fp8(xq, xs, M, W[i], S[i], s.N, s.K, y, c, st); });
      if (us < best) best = us, bc = c;
      dsk::GemvCfg np = c;
      np.pdl = false;
      double us_np = time_us(copies, [&](int i, cudaStream_t st) { dsk::gemv_fp8(xq, xs, M, W[i], S[i], s.N, s.K, y, np, st); });
      if (!sweep) printf("  M=%d  without PDL %7.2f us %6.1f GB/s (%3.0f%%)\n", M, us_np, bytes / us_np / 1e3, 100 * bytes / us_np / 1e3 / kPeak);
    }
    double bt = 1e30;
    dsk::GemvCfg btc;
    for (auto c : cfgs) {
      c.tiled = true;  // same bytes, tiled layout (contents do not matter for timing)
      double us = time_us(copies, [&](int i, cudaStream_t st) { dsk::gemv_fp8(xq, xs, M, W[i], S[i], s.N, s.K, y, c, st); });
      if (us < bt) bt = us, btc = c;
    }
    printf("  M=%d  tiled %7.2f us %6.1f GB/s (%3.0f%%) [mt %d wn %d wk %d]\n", M, bt, bytes / bt / 1e3,
           100 * bytes / bt / 1e3 / kPeak, btc.mt, btc.wn, btc.wk);
    double cc = time_us(copies, [&](int i, cudaStream_t st) { dsk::gemv_fp8_cc(xq, xs, M, W[i], S[i], s.N, s.K, y, st); });
    printf("  M=%d  mma %7.2f us %6.1f GB/s (%3.0f%%) [mt %d wn %d wk %d]   cuda-core %7.2f us %6.1f GB/s (%3.0f%%)\n", M,
           best, bytes / best / 1e3, 100 * bytes / best / 1e3 / kPeak, bc.mt, bc.wn, bc.wk, cc, bytes / cc / 1e3,
           100 * bytes / cc / 1e3 / kPeak);
  }
  for (int i = 0; i < copies; ++i) cudaFree(W[i]), cudaFree(S[i]);
  cudaFree(xq), cudaFree(xs), cudaFree(y);
}

// wq_a and wkv share their input: one launch over both (320 + 128 rows per rank).
static void bench_fp8_pair(bool sweep) {
  const int K = 5120, N1 = 320, N2 = 128;
  const size_t bytes = (size_t)(N1 + N2) * K + (size_t)(N1 + N2) / 32 * (K / 32);
  const int copies = copies_for(bytes);
  std::vector<uint8_t*> W1(copies), S1(copies), W2(copies), S2(copies);
  for (int i = 0; i < copies; ++i) {
    W1[i] = dev_fill((size_t)N1 * K, 7 + i), S1[i] = dev_fill(N1 / 32 * (K / 32), 70 + i, 118, 16);
    W2[i] = dev_fill((size_t)N2 * K, 9 + i), S2[i] = dev_fill(N2 / 32 * (K / 32), 90 + i, 118, 16);
  }
  uint8_t* xq = dev_fill(8 * K, 5);
  uint8_t* xs = dev_fill(8 * K / 32, 6, 118, 16);
  __nv_bfloat16 *y1, *y2;
  CUDA_CHECK(cudaMalloc(&y1, 8 * N1 * 2));
  CUDA_CHECK(cudaMalloc(&y2, 8 * N2 * 2));
  printf("%-8s N=%-5d K=%-5d %6.2f MB (two segments, one launch)\n", "wq_a+wkv", N1 + N2, K, bytes / 1e6);
  for (int M : {1, 6}) {
    double best = 1e30;
    dsk::GemvCfg bc;
    std::vector<dsk::GemvCfg> cfgs;
    for (int mt : {1, 2})
      for (int wn : {1, 2})
        for (int wk : {4, 8, 16})
          if (wn * wk <= 16 && (sweep || (mt == 1 && wn == 1 && wk == 16))) cfgs.push_back({mt, wn, wk});
    for (int tiled = 0; tiled < 2; ++tiled) {
      best = 1e30;
      for (auto c : cfgs) {
        c.tiled = tiled;
        double us = time_us(copies, [&](int i, cudaStream_t st) {
          dsk::Fp8Seg seg[2] = {{W1[i], S1[i], y1, N1}, {W2[i], S2[i], y2, N2}};
          dsk::gemv_fp8_multi(xq, xs, M, seg, 2, K, c, st);
        });
        if (us < best) best = us, bc = c;
      }
      printf("  M=%d  mma %s %7.2f us %6.1f GB/s (%3.0f%%) [mt %d wn %d wk %d]\n", M, tiled ? "tiled  " : "natural", best,
             bytes / best / 1e3, 100 * bytes / best / 1e3 / kPeak, bc.mt, bc.wn, bc.wk);
    }
  }
}

// 16-bit activations: bf16 weights (head, gate; fp32 out) or wo_a (FP8 widened; bf16 out).
static void bench_w16(const char* name, int N, int K, bool woa, bool sweep) {
  const size_t wb = woa ? (size_t)N * K : (size_t)N * K * 2, sb = woa ? (size_t)(N / 32) * (K / 32) : 0, bytes = wb + sb;
  const int copies = copies_for(bytes), G = woa ? 2 : 1;
  std::vector<uint8_t*> W(copies), S(copies, nullptr);
  for (int i = 0; i < copies; ++i) {
    W[i] = dev_fill(wb, 31 + i);
    if (woa) S[i] = dev_fill(sb, 41 + i, 110, 12);
  }
  __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(dev_fill((size_t)8 * K * G * 2, 5, 0x3c, 2));
  float* yf;
  __nv_bfloat16* yb;
  CUDA_CHECK(cudaMalloc(&yf, (size_t)8 * N * 4));
  CUDA_CHECK(cudaMalloc(&yb, (size_t)8 * N * 2));
  printf("%-8s N=%-5d K=%-5d %7.2f MB  (%s)\n", name, N, K, bytes / 1e6, woa ? "FP8 widened to bf16" : "bf16");
  for (int M : {1, 6}) {
    double best[2] = {1e30, 1e30};
    dsk::GemvCfg bc[2];
    for (int cc = 0; cc < 2; ++cc)
      for (int mt : {1, 2, 4})
        for (int wn : {1, 2, 4})
          for (int wk : {2, 4, 8, 16, 32}) {
            if (cc && mt == 4) continue;
            if (wn * wk > 16 || N % (8 * mt * wn) || (woa && (N / G) % (8 * mt * wn)) || (woa ? K : 2 * K) / 64 < wk) continue;
            if (!sweep && !(mt == 1 && wn == 1 && wk == 16)) continue;
            dsk::GemvCfg c{mt, wn, wk};
            c.tiled = true;
            double us = time_us(copies, [&](int i, cudaStream_t st) {
              if (woa) dsk::gemv_woa(x, M, W[i], S[i], N, K, N / G, yb, c, cc, st);
              else dsk::gemv_bf16(x, M, W[i], N, K, yf, nullptr, c, cc, st);
            });
            if (us < best[cc]) best[cc] = us, bc[cc] = c;
          }
    printf("  M=%d  mma %8.2f us %6.1f GB/s (%3.0f%%) [mt %d wn %d wk %d]   cuda-core %8.2f us %6.1f GB/s (%3.0f%%) [mt %d wn %d wk %d]\n",
           M, best[0], bytes / best[0] / 1e3, 100 * bytes / best[0] / 1e3 / kPeak, bc[0].mt, bc[0].wn, bc[0].wk, best[1],
           bytes / best[1] / 1e3, 100 * bytes / best[1] / 1e3 / kPeak, bc[1].mt, bc[1].wn, bc[1].wk);
  }
  for (int i = 0; i < copies; ++i) cudaFree(W[i]), cudaFree(S[i]);
  cudaFree(x), cudaFree(yf), cudaFree(yb);
}

// Small latency-bound kernels: RMSNorm (+ fused act_quant) of one 5120 row set, router top-6.
static void bench_small() {
  const int D = 5120, E = 384;
  __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(dev_fill((size_t)8 * D * 2, 5, 0x3c, 2));
  __nv_bfloat16* w = reinterpret_cast<__nv_bfloat16*>(dev_fill((size_t)D * 2, 6, 0x3c, 2));
  __nv_bfloat16* y;
  uint8_t *q, *sc;
  CUDA_CHECK(cudaMalloc(&y, 8 * D * 2));
  CUDA_CHECK(cudaMalloc(&q, 8 * D));
  CUDA_CHECK(cudaMalloc(&sc, 8 * D / 32));
  float *lg, *bias, *wts;
  int* idx;
  CUDA_CHECK(cudaMalloc(&lg, 8 * E * 4));
  CUDA_CHECK(cudaMalloc(&bias, E * 4));
  CUDA_CHECK(cudaMemset(lg, 0, 8 * E * 4));
  CUDA_CHECK(cudaMemset(bias, 0, E * 4));
  CUDA_CHECK(cudaMalloc(&wts, 8 * 6 * 4));
  CUDA_CHECK(cudaMalloc(&idx, 8 * 6 * 4));
  for (int M : {1, 6}) {
    double tn = time_us(1, [&](int, cudaStream_t st) { dsk::rmsnorm(x, w, M, D, 1e-20f, y, nullptr, nullptr, st); }, 200);
    double tq = time_us(1, [&](int, cudaStream_t st) { dsk::rmsnorm(x, w, M, D, 1e-20f, y, q, sc, st); }, 200);
    double tr = time_us(1, [&](int, cudaStream_t st) { dsk::router_topk(lg, bias, M, E, idx, wts, 1.5f, st); }, 200);
    printf("  M=%d  rmsnorm %5.2f us, + act_quant %5.2f us, router top-6 %5.2f us (back-to-back launches)\n", M, tn, tq, tr);
  }
}

// MoE: NE experts of random MXFP4 weights (tiled layout; contents do not matter for timing), the
// shared expert in rotating copies; every launch picks a fresh set of nU experts so weights are cold.
static void bench_moe(bool sweep) {
  const int NE = 96, kD = 5120, kI = 576;
  const size_t w13 = (size_t)kI * kD / 2, s13 = (size_t)kI * kD / 32, w2 = (size_t)kD * kI / 2, s2 = (size_t)kD * kI / 32;
  const size_t per_expert = 2 * (w13 + s13) + w2 + s2;
  const size_t sh_bytes = 2 * ((size_t)kI * kD + kI / 32 * (kD / 32)) + (size_t)kD * kI + kD / 32 * (kI / 32);
  dsk::MoeWeights base{};
  base.w1 = dev_fill(NE * w13, 1), base.s1 = dev_fill(NE * s13, 2, 120, 8);
  base.w3 = dev_fill(NE * w13, 3), base.s3 = dev_fill(NE * s13, 4, 120, 8);
  base.w2 = dev_fill(NE * w2, 5), base.s2 = dev_fill(NE * s2, 6, 120, 8);
  const int nsh = 8;
  std::vector<dsk::MoeWeights> W(nsh, base);
  for (int i = 0; i < nsh; ++i) {
    W[i].sh_w1 = dev_fill((size_t)kI * kD, 10 + i), W[i].sh_s1 = dev_fill(kI / 32 * (kD / 32), 20 + i, 118, 16);
    W[i].sh_w3 = dev_fill((size_t)kI * kD, 30 + i), W[i].sh_s3 = dev_fill(kI / 32 * (kD / 32), 40 + i, 118, 16);
    W[i].sh_w2 = dev_fill((size_t)kD * kI, 50 + i), W[i].sh_s2 = dev_fill(kD / 32 * (kI / 32), 60 + i, 118, 16);
  }
  uint8_t* xq = dev_fill(8 * kD, 7);
  uint8_t* xs = dev_fill(8 * kD / 32, 8, 118, 16);
  uint8_t* ws;
  float* y;
  CUDA_CHECK(cudaMalloc(&ws, dsk::moe_workspace_bytes()));
  CUDA_CHECK(cudaMemset(ws, 0, dsk::moe_workspace_bytes()));
  CUDA_CHECK(cudaMalloc(&y, 8 * kD * 4));
  const int copies = 48;
  int* idx;
  float* wts;
  CUDA_CHECK(cudaMalloc(&idx, copies * 48 * 4));
  CUDA_CHECK(cudaMalloc(&wts, copies * 48 * 4));
  std::vector<float> hw(copies * 48, 0.25f);
  CUDA_CHECK(cudaMemcpy(wts, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  printf("moe: %.2f MB per routed expert, %.2f MB shared expert (per rank)\n", per_expert / 1e6, sh_bytes / 1e6);
  auto set_union = [&](int M, int nu) {  // launch i: rows pick 6 each from experts base_i .. base_i + nu - 1
    std::vector<int> h(copies * 48);
    for (int i = 0; i < copies; ++i)
      for (int m = 0; m < M; ++m)
        for (int k = 0; k < 6; ++k) h[i * 48 + m * 6 + k] = (i * nu + (m * 6 + k) % nu) % NE;
    CUDA_CHECK(cudaMemcpy(idx, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  };
  auto t = [&](int M, dsk::MoeCfg c) {
    return time_us(copies, [&](int i, cudaStream_t st) {
      dsk::moe(xq, xs, M, idx + i * 48, wts + i * 48, W[i % nsh], ws, y, c, st);
    });
  };
  struct Case { int M, nu; };
  std::vector<Case> cases = {{1, 6}, {2, 12}, {4, 24}, {6, 6}, {6, 12}, {6, 18}, {6, 24}, {6, 30}, {6, 36}, {8, 48}};
  for (auto cs : cases) {
    set_union(cs.M, cs.nu);
    const double bytes = cs.nu * (double)per_expert + sh_bytes;
    dsk::MoeCfg best{};
    double tb = 1e30;
    if (sweep) {
      for (int up = 0; up < 6; ++up)
        for (int down = 0; down < 5; ++down) {
          dsk::MoeCfg c{up, down};
          double us = t(cs.M, c);
          if (us < tb) tb = us, best = c;
        }
    } else {
      best.down = cs.M == 1 ? 2 : 4;
      tb = t(cs.M, best);
    }
    dsk::MoeCfg cu = best, cd = best;
    cu.phases = 1, cd.phases = 2;
    double tu = t(cs.M, cu), td = t(cs.M, cd);
    dsk::MoeCfg cuq = cu;
    cuq.dyn = true;
    const double tuq = t(cs.M, cuq);  // moe_up with the task queue
    double tcc = 1e30;  // moe_up with the routed dot products on CUDA cores
    int ucc = 0;
    for (int up = 6; up < 10; ++up) {
      dsk::MoeCfg c = cu;
      c.up = up;
      double us = t(cs.M, c);
      if (us < tcc) tcc = us, ucc = up;
    }
    const double bu = cs.nu * 2.0 * (w13 + s13) + 2 * ((double)kI * kD + kI / 32 * (kD / 32));
    const double bd = bytes - bu;
    printf("  M=%d experts %2d  %7.1f MB  %7.1f us %6.1f GB/s (%3.0f%%)   up %6.1f us (%3.0f%%; queue %6.1f us)  down %6.1f us (%3.0f%%)  [up %d down %d]   up on CUDA cores %6.1f us (%3.0f%%) [%d]\n",
           cs.M, cs.nu, bytes / 1e6, tb, bytes / tb / 1e3, 100 * bytes / tb / 1e3 / kPeak, tu, 100 * bu / tu / 1e3 / kPeak,
           tuq, td, 100 * bd / td / 1e3 / kPeak, best.up, best.down, tcc, 100 * bu / tcc / 1e3 / kPeak, ucc);
  }
}

// The weight-streaming part of one decode step on one rank (TP=4 shapes of DESIGN decision 1), as
// one CUDA graph of PDL launches: per layer attn_norm(+quant), wq_a+wkv, q_norm(+quant), wq_b,
// [indexer wq_b], wo_a(+quant), wo_b, ffn_norm(+quant), gate (96 rows) + top-6, moe_up, moe_down;
// then the final norm and the head. Not included: attention itself, the compressor/indexer
// beyond wq_b, mHC, Engram, collectives. Weights rotate over 4 copies of the dense layer and a
// pool of 96 experts, every layer picking a fresh expert set, so nothing is reused from cache.
static void bench_step() {
  const int kL = 40, kCopies = 4, NE = 96, kD = 5120;
  const int index_layers[] = {2, 8, 14, 20, 24, 28, 32, 36};
  struct Dense {
    uint8_t *wqa, *sqa, *wkv, *skv, *wqb, *sqb, *wix, *six, *woa, *soa, *wob, *sob;
    uint8_t* gate;  // bf16 [96][5120]
    __nv_bfloat16 *n_attn, *n_q, *n_ffn;
  };
  std::vector<Dense> L(kCopies);
  double dense_bytes = 0;
  auto fp8 = [&](size_t n, size_t k, uint8_t*& w, uint8_t*& sc, uint32_t seed) {
    w = dev_fill(n * k, seed), sc = dev_fill(n / 32 * (k / 32), seed + 1, 118, 16);
    return (double)n * k + n / 32 * (k / 32);
  };
  for (int c = 0; c < kCopies; ++c) {
    Dense& d = L[c];
    double b = 0;
    b += fp8(320, kD, d.wqa, d.sqa, 100 + 20 * c);
    b += fp8(128, kD, d.wkv, d.skv, 102 + 20 * c);
    b += fp8(8192, 1280, d.wqb, d.sqb, 104 + 20 * c);
    fp8(1024, 1280, d.wix, d.six, 106 + 20 * c);
    b += fp8(2048, 4096, d.woa, d.soa, 108 + 20 * c);
    b += fp8(kD, 2048, d.wob, d.sob, 110 + 20 * c);
    d.gate = dev_fill((size_t)96 * kD * 2, 112 + 20 * c, 0x3c, 2);
    b += 96.0 * kD * 2;
    d.n_attn = reinterpret_cast<__nv_bfloat16*>(dev_fill(kD * 2, 113 + 20 * c, 0x3f, 1));
    d.n_q = reinterpret_cast<__nv_bfloat16*>(dev_fill(1280 * 2, 114 + 20 * c, 0x3f, 1));
    d.n_ffn = reinterpret_cast<__nv_bfloat16*>(dev_fill(kD * 2, 115 + 20 * c, 0x3f, 1));
    dense_bytes = b;
  }
  const size_t w13 = (size_t)576 * kD / 2, s13 = (size_t)576 * kD / 32, w2 = (size_t)kD * 576 / 2, s2 = (size_t)kD * 576 / 32;
  const double per_expert = 2.0 * (w13 + s13) + w2 + s2;
  const double sh_bytes = 2.0 * (576.0 * kD + 18 * 160) + (double)kD * 576 + 160 * 18;
  dsk::MoeWeights base{};
  base.w1 = dev_fill(NE * w13, 1), base.s1 = dev_fill(NE * s13, 2, 120, 8);
  base.w3 = dev_fill(NE * w13, 3), base.s3 = dev_fill(NE * s13, 4, 120, 8);
  base.w2 = dev_fill(NE * w2, 5), base.s2 = dev_fill(NE * s2, 6, 120, 8);
  std::vector<dsk::MoeWeights> MW(kCopies, base);
  for (int c = 0; c < kCopies; ++c) {
    MW[c].sh_w1 = dev_fill((size_t)576 * kD, 200 + c), MW[c].sh_s1 = dev_fill(18 * 160, 210 + c, 118, 16);
    MW[c].sh_w3 = dev_fill((size_t)576 * kD, 220 + c), MW[c].sh_s3 = dev_fill(18 * 160, 230 + c, 118, 16);
    MW[c].sh_w2 = dev_fill((size_t)kD * 576, 240 + c), MW[c].sh_s2 = dev_fill(160 * 18, 250 + c, 118, 16);
  }
  uint8_t* head = dev_fill((size_t)32320 * kD * 2, 7, 0x3c, 2);
  __nv_bfloat16* n_out = reinterpret_cast<__nv_bfloat16*>(dev_fill(kD * 2, 8, 0x3f, 1));
  // activations (contents irrelevant)
  __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(dev_fill((size_t)8 * 8192 * 2, 9, 0x3c, 2));
  __nv_bfloat16 *xn, *qa, *kv, *qn, *qb, *ix, *ob;
  uint8_t *xq, *xs, *qq, *qs, *oq, *os;
  float *lg, *bias, *wts, *y, *logits;
  int *idx_dummy, *idx;
  CUDA_CHECK(cudaMalloc(&xn, 8 * kD * 2));
  CUDA_CHECK(cudaMalloc(&qa, 8 * 1280 * 2));  // this rank's wq_a out; q_norm reads the all-gathered 1280
  CUDA_CHECK(cudaMalloc(&kv, 8 * 128 * 2));
  CUDA_CHECK(cudaMalloc(&qn, 8 * 1280 * 2));
  CUDA_CHECK(cudaMalloc(&qb, 8 * 8192 * 2));
  CUDA_CHECK(cudaMalloc(&ix, 8 * 1024 * 2));
  CUDA_CHECK(cudaMalloc(&ob, 8 * kD * 2));
  CUDA_CHECK(cudaMalloc(&xq, 8 * kD));
  CUDA_CHECK(cudaMalloc(&xs, 8 * kD / 32));
  CUDA_CHECK(cudaMalloc(&qq, 8 * 1280));
  CUDA_CHECK(cudaMalloc(&qs, 8 * 1280 / 32));
  CUDA_CHECK(cudaMalloc(&oq, 8 * 2048));
  CUDA_CHECK(cudaMalloc(&os, 8 * 2048 / 32));
  CUDA_CHECK(cudaMalloc(&lg, 8 * 384 * 4));
  float* lsel;
  CUDA_CHECK(cudaMalloc(&lsel, (size_t)kL * 8 * 384 * 4));
  CUDA_CHECK(cudaMemset(lg, 0, 8 * 384 * 4));
  CUDA_CHECK(cudaMalloc(&bias, 384 * 4));
  CUDA_CHECK(cudaMemset(bias, 0, 384 * 4));
  CUDA_CHECK(cudaMalloc(&wts, kL * 48 * 4));
  CUDA_CHECK(cudaMalloc(&idx_dummy, 48 * 4));
  CUDA_CHECK(cudaMalloc(&idx, kL * 48 * 4));
  CUDA_CHECK(cudaMalloc(&y, 8 * kD * 4));
  CUDA_CHECK(cudaMalloc(&logits, (size_t)8 * 32320 * 4));
  std::vector<float> hw(kL * 48, 0.25f);
  CUDA_CHECK(cudaMemcpy(wts, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  uint8_t* ws;
  CUDA_CHECK(cudaMalloc(&ws, dsk::moe_workspace_bytes()));
  CUDA_CHECK(cudaMemset(ws, 0, dsk::moe_workspace_bytes()));
  const double head_bytes = 32320.0 * kD * 2, ix_bytes = 1024.0 * 1280 + 32 * 40;

  // STEP_FUSE: the variants to time, interleaved, as bit sets (1: attn norm into wq_a+wkv, 2: ffn norm
  // into the gate, 4: q norm into wq_b, 8: top-6 into the MoE; -1 all); default "0 2" (2 = what
  // measured best). STEP_ROUNDS replays each (default 15).
  std::vector<int> fuses;
  {
    const char* fu = getenv("STEP_FUSE");
    std::string str = fu ? fu : "0 2";
    for (size_t i = 0; i < str.size();) {
      size_t j = str.find_first_of(" ,", i);
      if (j == std::string::npos) j = str.size();
      if (j > i) fuses.push_back(atoi(str.substr(i, j - i).c_str()));
      i = j + 1;
    }
  }
  const int rounds = getenv("STEP_ROUNDS") ? atoi(getenv("STEP_ROUNDS")) : 15;
  const char* cases_env = getenv("STEP_CASES");  // e.g. "1" for M=1 only
  struct Case { int M, nu; };
  for (Case cs : {Case{1, 6}, Case{6, 27}, Case{6, 36}}) {
    if (cases_env && cs.M != atoi(cases_env)) continue;
    const int M = cs.M, nu = cs.nu;
    std::vector<int> h(kL * 48);
    for (int l = 0; l < kL; ++l)
      for (int m = 0; m < M; ++m)
        for (int k = 0; k < 6; ++k) h[l * 48 + m * 6 + k] = (l * nu + (m * 6 + k) % nu) % NE;
    CUDA_CHECK(cudaMemcpy(idx, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    {  // logits that make the fused router pick the same experts
      std::vector<float> hl((size_t)kL * 8 * 384, -10.f);
      for (int l = 0; l < kL; ++l)
        for (int m = 0; m < M; ++m)
          for (int k = 0; k < 6; ++k) hl[((size_t)l * 8 + m) * 384 + h[l * 48 + m * 6 + k]] = 5.f - 0.1f * k;
      CUDA_CHECK(cudaMemcpy(lsel, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
    }
    dsk::GemvCfg c8{1, 1, 16}, c8k{1, 1, 8}, cw{1, 4, 4}, cg{1, 1, 8}, ch{2, 1, 16};
    dsk::GemvCfg cqn{getenv("QN_MT") ? atoi(getenv("QN_MT")) : 1, 1, getenv("QN_WK") ? atoi(getenv("QN_WK")) : 8};
    c8.tiled = c8k.tiled = cqn.tiled = true;
    dsk::MoeCfg mc{};
    mc.down = M == 1 ? 2 : 4;
    const char* sk = getenv("STEP_SKIP");  // bits: 1 norms, 2 gate+top-6, 4 MoE, 8 attention GEMVs
    const int skip = sk ? atoi(sk) : 0;
    auto step = [&](int fuse, cudaStream_t st) {
      for (int l = 0; l < kL; ++l) {
        const Dense& d = L[l % kCopies];
        dsk::Fp8Seg seg[2] = {{d.wqa, d.sqa, qa, 320}, {d.wkv, d.skv, kv, 128}};
        dsk::Fp8Seg segq = {d.wqb, d.sqb, qb, 8192};
        if (fuse & 1) {  // the normed x goes on to the compressor/indexer
          if (!(skip & 8)) dsk::gemv_fp8_norm(x, d.n_attn, 1e-20f, M, seg, 2, kD, c8k, st, xn);
        } else {
          if (!(skip & 1)) dsk::rmsnorm(x, d.n_attn, M, kD, 1e-20f, xn, xq, xs, st);
          if (!(skip & 8)) dsk::gemv_fp8_multi(xq, xs, M, seg, 2, kD, c8k, st);
        }
        if (fuse & 4) {  // q's quant goes on to the indexer's wq_b
          if (!(skip & 8)) dsk::gemv_fp8_norm(qa, d.n_q, 1e-20f, M, &segq, 1, 1280, cqn, st, nullptr, qq, qs);
        } else {
          if (!(skip & 1)) dsk::rmsnorm(qa, d.n_q, M, 1280, 1e-20f, qn, qq, qs, st);  // on the all-gathered q (stand-in)
          if (!(skip & 8)) dsk::gemv_fp8(qq, qs, M, d.wqb, d.sqb, 8192, 1280, qb, c8k, st);
        }
        for (int il : index_layers)
          if (il == l && !(skip & 8)) dsk::gemv_fp8(qq, qs, M, d.wix, d.six, 1024, 1280, ix, c8k, st);
        if (!(skip & 8)) dsk::gemv_woa(x, M, d.woa, d.soa, 2048, 4096, 1024, nullptr, cw, false, st, oq, os);
        if (!(skip & 8)) dsk::gemv_fp8(oq, os, M, d.wob, d.sob, kD, 2048, ob, c8, st);
        if (fuse & 2) {  // the gate's CTA 0 quantises the normed x for the MoE
          if (!(skip & 2)) dsk::gemv_bf16_norm(x, d.n_ffn, 1e-20f, M, d.gate, 96, kD, lg, nullptr, cg, st, xq, xs);
        } else {
          if (!(skip & 1)) dsk::rmsnorm(x, d.n_ffn, M, kD, 1e-20f, xn, xq, xs, st);
          if (!(skip & 2)) dsk::gemv_bf16(xn, M, d.gate, 96, kD, lg, nullptr, cg, false, st);
        }
        if (fuse & 8) {  // top-6 inside the MoE kernels (from logits standing in for this layer's gate)
          if (!(skip & 4))
            dsk::moe_routed(xq, xs, M, lsel + (size_t)l * 8 * 384, bias, 384, 1.5f, MW[l % kCopies], ws, y, mc, st);
        } else {
          if (!(skip & 2)) dsk::router_topk(lg, bias, M, 384, idx_dummy, wts + l * 48, 1.5f, st);
          if (!(skip & 4)) dsk::moe(xq, xs, M, idx + l * 48, wts + l * 48, MW[l % kCopies], ws, y, mc, st);
        }
      }
      dsk::rmsnorm(x, n_out, M, kD, 1e-20f, xn, nullptr, nullptr, st);
      dsk::gemv_bf16(xn, M, head, 32320, kD, logits, nullptr, ch, false, st);
    };
    const double bytes = kL * (dense_bytes + sh_bytes + nu * per_expert) + 8 * ix_bytes + head_bytes;
    std::vector<std::function<void(cudaStream_t)>> fns;
    for (int f : fuses) fns.push_back([&step, f](cudaStream_t st) { step(f, st); });
    const auto ts = time_interleaved(fns, rounds);
    for (size_t i = 0; i < fuses.size(); ++i) {
      const double us = ts[i][ts[i].size() / 2];
      printf("decode step, M=%d, %2d experts/layer, fuse %2d: %7.1f MB/rank  median %6.0f us (min %6.0f)  %6.1f GB/s "
             "(%3.0f%% of 235)   at 235 GB/s: %6.0f us\n",
             M, nu, fuses[i], bytes / 1e6, us, ts[i][0], bytes / us / 1e3, 100 * bytes / us / 1e3 / kPeak, bytes / 235e3);
    }
  }
}

// ---- prefill (prefill.cu): tensor-core GEMMs on chunks of M rows.
__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    p[i] = __float2bfloat16((float)(h & 0xFFFF) / 32768.0f - 1.0f);
  }
}

// Tensor-core throughput: 8 independent accumulation chains per warp, 8 warps per CTA, 4 CTAs per SM.
template <bool BF16>
__global__ void __launch_bounds__(256) k_mma_peak(float* out, int iters) {
  float acc[8][4] = {};
  const uint32_t a0 = threadIdx.x * 0x01010101u & 0x37373737u, a1 = a0 ^ 0x01000100u, a2 = a0 ^ 0x00010001u;
  const uint32_t a3 = a0 ^ 0x01010000u, b0 = a0 ^ 0x00000101u, b1 = a0 ^ 0x01000001u;
  for (int i = 0; i < iters; ++i)
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      if constexpr (BF16) dsk::mma_bf16(acc[c], a0, a1, a2, a3, b0, b1);
      else dsk::mma_e4m3_e4m3(acc[c], a0, a1, a2, a3, b0, b1, 127, 127);
    }
  float s = 0;
  for (int c = 0; c < 8; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
  if (s == 1.2345f) out[0] = s;
}

// L2 -> SM read bandwidth: every CTA streams an L2-resident buffer (16-byte .cg loads).
__global__ void __launch_bounds__(256) k_l2_read(const uint4* p, size_t n, int iters, float* out) {
  uint32_t acc = 0;
  for (int it = 0; it < iters; ++it)
    for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x + (size_t)it * 4096) % n, c = 0; c < 64;
         ++c, i = (i + (size_t)gridDim.x * blockDim.x) % n) {
      uint4 v;
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  if (acc == 0x12345678u) out[0] = 1.f;
}

// Routing of M rows: 6 distinct experts per row, uniform or Zipf-skewed (weight ~ 1/(rank+1)^0.5
// over a random expert order).
static std::vector<int> make_routing(int M, int E, bool skewed, uint32_t seed) {
  std::vector<double> w(E, 1.0);
  std::vector<int> order(E);
  for (int e = 0; e < E; ++e) order[e] = e;
  uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  auto rnd = [&]() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return (s >> 11) * (1.0 / 9007199254740992.0);
  };
  for (int e = E - 1; e > 0; --e) std::swap(order[e], order[(int)(rnd() * (e + 1))]);
  if (skewed)
    for (int r = 0; r < E; ++r) w[order[r]] = 1.0 / std::sqrt(r + 1.0);
  std::vector<int> idx((size_t)M * 6);
  for (int m = 0; m < M; ++m) {
    std::vector<double> ww = w;
    for (int k = 0; k < 6; ++k) {
      double tot = 0;
      for (double v : ww) tot += v;
      double u = rnd() * tot;
      int e = 0;
      while (e < E - 1 && (u -= ww[e]) > 0) ++e;
      while (ww[e] == 0) e = (e + 1) % E;
      idx[(size_t)m * 6 + k] = e;
      ww[e] = 0;
    }
  }
  return idx;
}

static void bench_prefill() {
  std::vector<int> Ms;
  if (const char* e = getenv("PF_M"))
    for (const char* p = e; *p;) {
      char* q;
      const long v = strtol(p, &q, 10);
      if (q == p) break;
      Ms.push_back((int)v), p = q;
    }
  if (Ms.empty()) Ms = {2048, 4096};
  const int rounds = getenv("PF_ROUNDS") ? atoi(getenv("PF_ROUNDS")) : 11;
  dsk::PrefillTuning base;  // PF_L2: the MoE's L2 hint bits for everything but the hint variants
  if (const char* e = getenv("PF_L2")) base.l2 = atoi(e);
  dsk::prefill_tuning(base);
  const int Mmax = *std::max_element(Ms.begin(), Ms.end());
  const int kD = 5120, kI = 576, E = 384;
  const double kPeakMma = 250.0;  // TFLOPS, mxf8f6f4 dense (docs/findings.md)
  float* sink;
  CUDA_CHECK(cudaMalloc(&sink, 4));
  {
    const int iters = 1024, grid = 48 * 4;
    auto ts = time_interleaved({[&](cudaStream_t st) { k_mma_peak<false><<<grid, 256, 0, st>>>(sink, iters); },
                                [&](cudaStream_t st) { k_mma_peak<true><<<grid, 256, 0, st>>>(sink, iters); }},
                               5);
    const double mmas = (double)grid * 8 * iters * 8;
    printf("mma.sync peak (8 chains/warp, 32 warps/SM): e4m3 m16n8k32 %.0f TFLOPS, bf16 m16n8k16 %.0f TFLOPS\n",
           mmas * 8192 / ts[0][2] / 1e6, mmas * 4096 / ts[1][2] / 1e6);
    for (size_t mb : {4, 8, 16}) {
      const size_t n = (mb << 20) / 16;
      uint8_t* buf = dev_fill(n * 16, 77);
      const int grid2 = 48 * 4, it2 = 256;
      auto t2 = time_interleaved({[&](cudaStream_t st) {
                                   k_l2_read<<<grid2, 256, 0, st>>>(reinterpret_cast<const uint4*>(buf), n, it2, sink);
                                 }},
                                 5);
      printf("L2 read, %zu MB resident: %.0f GB/s\n", mb, (double)grid2 * 256 * it2 * 64 * 16 / t2[0][2] / 1e3);
      cudaFree(buf);
    }
    int maxp = 0;
    size_t lim = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxp, cudaDevAttrMaxPersistingL2CacheSize, 0));
    CUDA_CHECK(cudaDeviceGetLimit(&lim, cudaLimitPersistingL2CacheSize));
    printf("L2 set-aside for persisting (evict_last) lines: %zu of at most %d bytes\n", lim, maxp);
  }

  // Dense FP8 matrices (per rank); copies rotate so that every launch starts with cold weights.
  struct D8 {
    const char* name;
    std::vector<int> N;
    int K;
    int copies = 0;
    std::vector<std::vector<uint8_t*>> W, S;  // [copy][segment]
  };
  std::vector<D8> d8 = {{"wq_a+wkv", {320, 128}, 5120}, {"wq_b", {8192}, 1280}, {"ix_wq_b", {1024}, 1280},
                        {"wo_b", {5120}, 2048},         {"sh_w1w3", {576, 576}, 5120}, {"sh_w2", {5120}, 576}};
  uint32_t seed = 100;
  for (auto& d : d8) {
    size_t bytes = 0;
    for (int n : d.N) bytes += (size_t)n * d.K;
    d.copies = std::max<int>(2, (int)((64u << 20) / bytes + 1));
    d.W.resize(d.copies), d.S.resize(d.copies);
    for (int c = 0; c < d.copies; ++c)
      for (int n : d.N)
        d.W[c].push_back(dev_fill((size_t)n * d.K, seed++)), d.S[c].push_back(dev_fill((size_t)n / 32 * (d.K / 32), seed++, 118, 16));
  }
  const int woa_copies = 9;
  std::vector<uint8_t*> woaW(woa_copies), woaS(woa_copies);
  for (int c = 0; c < woa_copies; ++c) woaW[c] = dev_fill((size_t)2048 * 4096, seed++), woaS[c] = dev_fill(64 * 128, seed++, 118, 16);
  const int gate_copies = 8;
  std::vector<uint8_t*> gateW(gate_copies);
  for (int c = 0; c < gate_copies; ++c) {
    __nv_bfloat16* p;
    CUDA_CHECK(cudaMalloc(&p, (size_t)96 * kD * 2));
    k_fill_bf16<<<256, 256>>>(p, (size_t)96 * kD, seed++);
    gateW[c] = reinterpret_cast<uint8_t*>(p);
  }
  uint8_t* xq = dev_fill((size_t)Mmax * 8192, 7);
  uint8_t* xs = dev_fill((size_t)Mmax * 256, 8, 118, 16);
  __nv_bfloat16 *xb, *yb;
  CUDA_CHECK(cudaMalloc(&xb, (size_t)Mmax * 8192 * 2));
  k_fill_bf16<<<1024, 256>>>(xb, (size_t)Mmax * 8192, 9);
  CUDA_CHECK(cudaMalloc(&yb, (size_t)Mmax * 8192 * 2));
  uint8_t *q2, *qs2, *hq, *hs;
  CUDA_CHECK(cudaMalloc(&q2, (size_t)Mmax * 2048));
  CUDA_CHECK(cudaMalloc(&qs2, (size_t)Mmax * 64));
  CUDA_CHECK(cudaMalloc(&hq, (size_t)Mmax * kI));
  CUDA_CHECK(cudaMalloc(&hs, (size_t)Mmax * kI / 32));
  float* yf;
  CUDA_CHECK(cudaMalloc(&yf, (size_t)Mmax * kD * 4));

  // Routed experts of one layer (384 x 4.7 MB per rank: never in L2) and the shared expert.
  const size_t w13 = (size_t)kI * kD / 2, s13 = (size_t)kI * kD / 32, w2b = (size_t)kD * kI / 2, s2b = (size_t)kD * kI / 32;
  dsk::MoeWeights mw{};
  mw.w1 = dev_fill(E * w13, 1), mw.s1 = dev_fill(E * s13, 2, 120, 8);
  mw.w3 = dev_fill(E * w13, 3), mw.s3 = dev_fill(E * s13, 4, 120, 8);
  mw.w2 = dev_fill(E * w2b, 5), mw.s2 = dev_fill(E * s2b, 6, 120, 8);
  mw.sh_w1 = d8[4].W[0][0], mw.sh_s1 = d8[4].S[0][0], mw.sh_w3 = d8[4].W[0][1], mw.sh_s3 = d8[4].S[0][1];
  mw.sh_w2 = d8[5].W[0][0], mw.sh_s2 = d8[5].S[0][0];
  int *idx_u, *idx_s;
  float* wts;
  CUDA_CHECK(cudaMalloc(&idx_u, (size_t)Mmax * 24));
  CUDA_CHECK(cudaMalloc(&idx_s, (size_t)Mmax * 24));
  CUDA_CHECK(cudaMalloc(&wts, (size_t)Mmax * 24));
  {
    auto hu = make_routing(Mmax, E, false, 1), hsk = make_routing(Mmax, E, true, 2);
    std::vector<float> hw((size_t)Mmax * 6, 0.25f);
    CUDA_CHECK(cudaMemcpy(idx_u, hu.data(), hu.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(idx_s, hsk.data(), hsk.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wts, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  }
  void* ws;
  CUDA_CHECK(cudaMalloc(&ws, dsk::moe_prefill_workspace_bytes(Mmax, E)));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double routed_bytes = (double)E * (2 * (w13 + s13) + w2b + s2b);
  cudaStream_t side;
  CUDA_CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));

  if (getenv("PF_SWEEP")) {  // tile variants of the dense GEMMs, interleaved per shape
    for (const int M : Ms)
      for (int k = 0; k <= (int)d8.size(); ++k) {
        std::vector<std::function<void(cudaStream_t)>> fns;
        const int nv = k < (int)d8.size() ? 6 : 4;
        double fl = 0;
        int reps = woa_copies;
        if (k < (int)d8.size()) {
          for (int n : d8[k].N) fl += 2.0 * M * n * d8[k].K;
          reps = d8[k].copies;
        } else {
          fl = 2.0 * M * 2048 * 4096;
        }
        for (int v = 0; v < nv; ++v)
          fns.push_back([&, M, k, v](cudaStream_t st) {
            dsk::PrefillTuning t;
            if (k < (int)d8.size()) {
              t.fp8 = v;
              dsk::prefill_tuning(t);
              const D8& d = d8[k];
              for (int c = 0; c < d.copies; ++c) {
                if (d.N.size() == 2 && d.N[0] == 576) {
                  dsk::gemm_fp8_swiglu_p(xq, xs, M, d.W[c][0], d.S[c][0], d.W[c][1], d.S[c][1], 576, d.K, hq, hs, st);
                  continue;
                }
                dsk::Fp8Seg segs[2];
                size_t off = 0;
                for (size_t i = 0; i < d.N.size(); ++i)
                  segs[i] = {d.W[c][i], d.S[c][i], yb + off, d.N[i]}, off += (size_t)M * d.N[i];
                dsk::gemm_fp8_p(xq, xs, M, segs, (int)d.N.size(), d.K, st);
              }
            } else {
              t.w16 = v;
              dsk::prefill_tuning(t);
              for (int c = 0; c < woa_copies; ++c)
                dsk::gemm_woa_p(xb, M, woaW[c], woaS[c], 2048, 4096, 1024, nullptr, q2, qs2, st);
            }
            dsk::prefill_tuning(dsk::PrefillTuning{});
          });
        auto ts = time_interleaved(fns, rounds);
        printf("M=%d %-10s", M, k < (int)d8.size() ? d8[k].name : "wo_a");
        for (int v = 0; v < nv; ++v) printf("  v%d %6.1f", v, fl / (ts[v][ts[v].size() / 2] / reps) / 1e6);
        printf("  TFLOPS\n");
      }
    return;
  }

  for (const int M : Ms) {
    std::vector<std::string> names;
    std::vector<double> flops, reps;
    std::vector<std::function<void(cudaStream_t)>> fns;
    auto dense = [&](const D8& d, int c, cudaStream_t st) {
      if (d.N.size() == 2 && d.N[0] == 576) {
        dsk::gemm_fp8_swiglu_p(xq, xs, M, d.W[c][0], d.S[c][0], d.W[c][1], d.S[c][1], 576, d.K, hq, hs, st);
        return;
      }
      dsk::Fp8Seg segs[2];
      size_t off = 0;
      for (size_t i = 0; i < d.N.size(); ++i) segs[i] = {d.W[c][i], d.S[c][i], yb + off, d.N[i]}, off += (size_t)M * d.N[i];
      dsk::gemm_fp8_p(xq, xs, M, segs, (int)d.N.size(), d.K, st);
    };
    for (const auto& d : d8) {
      double n = 0;
      for (int v : d.N) n += v;
      names.push_back(d.name), flops.push_back(2.0 * M * n * d.K), reps.push_back(d.copies);
      fns.push_back([&, M](cudaStream_t st) {
        for (int c = 0; c < d.copies; ++c) dense(d, c, st);
      });
    }
    names.push_back("wo_a (bf16)"), flops.push_back(2.0 * M * 2048 * 4096), reps.push_back(woa_copies);
    fns.push_back([&, M](cudaStream_t st) {
      for (int c = 0; c < woa_copies; ++c) dsk::gemm_woa_p(xb, M, woaW[c], woaS[c], 2048, 4096, 1024, nullptr, q2, qs2, st);
    });
    names.push_back("gate (bf16)"), flops.push_back(2.0 * M * 96 * kD), reps.push_back(gate_copies);
    fns.push_back([&, M](cudaStream_t st) {
      for (int c = 0; c < gate_copies; ++c) dsk::gemm_bf16_p(xb, M, gateW[c], 96, kD, yf, st);
    });
    const double moe_flops = 2.0 * M * 6 * (3.0 * kI * kD) + 2.0 * M * 3.0 * kI * kD;
    for (int sk = 0; sk < 6; ++sk) {
      const char* nm[] = {"moe, uniform routing", "moe, skewed routing", "moe, uniform, shared beside",
                          M > 2560 ? "moe, uniform, fused comb." : "moe, uniform, comb. apart",
                          "moe, uniform, no L2 hints", "moe, uniform, L2 all normal"};
      names.push_back(nm[sk]), flops.push_back(moe_flops), reps.push_back(1);
      fns.push_back([&, M, sk](cudaStream_t st) {
        dsk::MoePCfg cfg;
        if (sk == 2) cfg.side = side;
        dsk::PrefillTuning t = base;
        t.down = sk == 3 ? (M > 2560 ? 2 : 1) : 0;
        t.l2 = sk == 4 ? 0 : sk == 5 ? 8 : base.l2;
        dsk::prefill_tuning(t);
        dsk::moe_prefill(xq, xs, M, sk == 1 ? idx_s : idx_u, wts, E, mw, ws, yf, st, cfg);
        dsk::prefill_tuning(base);
      });
    }
    const char* phase_names[] = {"  bucketing", "  shared expert", "  routed up", "  routed down (+combine)",
                                 "  combine"};
    for (int ph = 0; ph < 5; ++ph) {
      names.push_back(phase_names[ph]);
      flops.push_back(ph == 1 ? 2.0 * M * 3.0 * kI * kD : ph == 2 ? 2.0 * M * 6 * 2.0 * kI * kD : ph == 3 ? 2.0 * M * 6 * kI * kD : 0);
      reps.push_back(1);
      fns.push_back([&, M, ph](cudaStream_t st) {
        dsk::MoePCfg cfg;
        cfg.phases = 1 << ph;
        dsk::moe_prefill(xq, xs, M, idx_u, wts, E, mw, ws, yf, st, cfg);
      });
    }
    // one layer's GEMMs for the chunk (indexer included, MoE with uniform routing)
    double layer_flops = 0;  // attention-side GEMMs, gate, MoE with the shared expert (uniform routing)
    for (size_t i : {0, 1, 2, 3, 6, 7, 8}) layer_flops += flops[i];
    names.push_back("layer GEMMs"), flops.push_back(layer_flops), reps.push_back(1);
    fns.push_back([&, M](cudaStream_t st) {
      dense(d8[0], 0, st), dense(d8[1], 0, st), dense(d8[2], 0, st);
      dsk::gemm_woa_p(xb, M, woaW[0], woaS[0], 2048, 4096, 1024, nullptr, q2, qs2, st);
      dense(d8[3], 0, st);
      dsk::gemm_bf16_p(xb, M, gateW[0], 96, kD, yf, st);
      dsk::moe_prefill(xq, xs, M, idx_u, wts, E, mw, ws, yf, st);
    });
    auto ts = time_interleaved(fns, rounds);
    printf("\nprefill chunk M=%d: median of %d interleaved rounds (min); weights cold in L2; per rank (TP=4)\n", M, rounds);
    printf("  %-22s %8s %10s %10s %8s\n", "kernel", "GFLOP", "us", "(min)", "TFLOPS");
    for (size_t i = 0; i < fns.size(); ++i) {
      const double us = ts[i][ts[i].size() / 2] / reps[i], mn = ts[i][0] / reps[i];
      printf("  %-22s %8.2f %10.1f %10.1f %8.1f", names[i].c_str(), flops[i] / 1e9, us, mn, flops[i] / us / 1e6);
      if (names[i].rfind("moe", 0) == 0) printf("   routed weights %.0f GB/s", routed_bytes / us / 1e3);
      if (names[i] == "layer GEMMs")
        printf("\n  x20 encoder layers: %.1f ms per chunk = %.0f tok/s (GEMMs only, %.0f%% of %.0f TFLOPS)",
               20 * us / 1e3, M / (20 * us / 1e6), 100 * flops[i] / us / 1e6 / kPeakMma, kPeakMma);
      printf("\n");
    }
  }
}

// ---- prefill MoE on real routes (moereal): each kernel against the streaming floor.
// PF_ROUTES=<prefix> names routes dumped from the engine (rank 0's ffn_p, every row of a chunk:
// <prefix>.L<LL>.c<call>.m<M>.i32, int32 [M][6]). The replay case reads layers 20-39 of call 0 at 128
// rows (CED's replay); the encoder cases read layers 0-19 of call PF_CALL (default 1, the second 8192-row
// chunk), rows [0, M). PF_M (default "200 512 2000 8000 8192") the encoder chunks, PF_ROUNDS (default 5).
// Per layer: the shared expert once, then rounds of bucketing (untimed: it also resets the fused
// combine's counters), up, down (with the combine when the MoE fuses it), the separate combine, and the
// whole MoE (every phase, the shared expert inline, PDL overlaps included), timed apart by events. The
// floor is the touched experts' bytes (this rank's slices, scales included) at 235 GB/s: up w1 + w3,
// down w2; the MoE's adds the shared expert's 8.9 MB. GB/s: those bytes over the time; "sum": the MoE's
// time summed over the case's layers. PF_TURNS="<dir>/a<pos>.m<M> ..." adds a case for each pass the
// engine dumped (DSGB10_ROUTE_DUMP in profiling builds: <prefix>.L<LL>.i32, the layers found); PF_ROUTES
// may then be unset. The scales are random (120..127) unless PF_SCALES=<layer> takes that layer's real
// ones (all 384 experts, from MS_FILE or rank 0's file) for every layer. Variants: PF_DGRP="0 5 10" the
// down kernel by these column blocks per CTA (PrefillTuning::dgrp, 0: k_moe_down_p; PF_DRING its weight
// steps in flight), then PF_UPF="0 1 2" the up kernel's L2 prefetch distances (::upf), then PF_TORD="0 2"
// tile orders (::tord), then PF_CODE="0 1" the scales raw or read from their 1-bit code (dsk::scale_code;
// real scales only; its floor counts the code blocks and the raw groups in place of the scales), then
// PF_CGRP="1 4" down in this many column groups, a call each (the engine's ffn_p: 4; the MoE then runs
// bucketing, shared expert and up as one call, as ffn_p does), each beside the first of the others. With
// more than one variant, the first layer's outputs are compared with the default kernels' on the raw
// scales, bit for bit.
static bool load_routes(const std::string& path, int M, std::vector<int>& idx) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  idx.resize((size_t)M * 6);
  const size_t got = fread(idx.data(), 4, idx.size(), f);
  fclose(f);
  return got == idx.size();
}

static std::vector<int> env_ints(const char* name) {
  std::vector<int> v;
  if (const char* e = getenv(name))
    for (const char* p = e; *p;) {
      char* q;
      const long x = strtol(p, &q, 10);
      if (q == p) break;
      v.push_back((int)x), p = q;
    }
  return v;
}

static size_t check_code(const std::vector<uint8_t>& s, const std::vector<uint32_t>& code, int E, int N, int KB, int base,
                         std::vector<int>& raw);

static void bench_moe_real() {
  const char *pre = getenv("PF_ROUTES"), *turns = getenv("PF_TURNS");
  if (!pre && !turns) die("moereal: set PF_ROUTES or PF_TURNS");
  std::vector<int> Ms = env_ints("PF_M"), dg = env_ints("PF_DGRP"), uf = env_ints("PF_UPF"), to = env_ints("PF_TORD"),
                   cd = env_ints("PF_CODE"), cg = env_ints("PF_CGRP");
  if (Ms.empty()) Ms = {200, 512, 2000, 8000, 8192};
  if (dg.empty()) dg = {-1};
  if (uf.empty()) uf = {-1};
  if (to.empty()) to = {-1};
  if (cd.empty()) cd = {0};
  if (cg.empty()) cg = {1};
  for (int K : cg)
    if (K < 1 || 5120 % (K * 128)) die("moereal: PF_CGRP %d: 5120 columns in groups of a multiple of 128", K);
  struct Var {
    int G, U, T, C, K;
  };
  std::vector<Var> var;
  for (int G : dg) var.push_back({G, uf[0], to[0], cd[0], cg[0]});
  for (size_t k = 1; k < uf.size(); ++k) var.push_back({dg[0], uf[k], to[0], cd[0], cg[0]});
  for (size_t k = 1; k < to.size(); ++k) var.push_back({dg[0], uf[0], to[k], cd[0], cg[0]});
  for (size_t k = 1; k < cd.size(); ++k) var.push_back({dg[0], uf[0], to[0], cd[k], cg[0]});
  for (size_t k = 1; k < cg.size(); ++k) var.push_back({dg[0], uf[0], to[0], cd[0], cg[k]});
  const int rounds = getenv("PF_ROUNDS") ? atoi(getenv("PF_ROUNDS")) : 5;
  const int call = getenv("PF_CALL") ? atoi(getenv("PF_CALL")) : 1;
  const bool verbose = getenv("PF_VERBOSE") != nullptr;
  const int sl = getenv("PF_SCALES") ? atoi(getenv("PF_SCALES")) : -1;
  bool code = false;
  for (const Var& v : var) code = code || v.C;
  if (code && sl < 0) die("moereal: PF_CODE 1 needs real scales (PF_SCALES=<layer>)");
  dsk::PrefillTuning tune;
  if (getenv("PF_DRING")) tune.dring = atoi(getenv("PF_DRING"));
  const int kD = 5120, kI = 576, E = 384, Mmax = 8192;
  const double kBw = 235.0;  // GB/s, the streaming floor's rate
  const size_t w13 = (size_t)kI * kD / 2, s13 = (size_t)kI * kD / 32, w2b = (size_t)kD * kI / 2, s2b = (size_t)kD * kI / 32;
  // bytes a touched expert's up (w1 + w3) and down (w2) read: [raw, code][expert]
  std::vector<double> up_b[2], down_b[2];
  up_b[0].assign(E, 2.0 * (w13 + s13)), down_b[0].assign(E, (double)(w2b + s2b));
  up_b[1] = up_b[0], down_b[1] = down_b[0];
  dsk::MoeWeights mw{};
  mw.w1 = dev_fill(E * w13, 1), mw.w3 = dev_fill(E * w13, 3), mw.w2 = dev_fill(E * w2b, 5);
  mw.sh_w1 = dev_fill((size_t)kI * kD, 7), mw.sh_s1 = dev_fill((size_t)kI / 32 * (kD / 32), 8, 118, 16);
  mw.sh_w3 = dev_fill((size_t)kI * kD, 9), mw.sh_s3 = dev_fill((size_t)kI / 32 * (kD / 32), 10, 118, 16);
  mw.sh_w2 = dev_fill((size_t)kD * kI, 11), mw.sh_s2 = dev_fill((size_t)kD / 32 * (kI / 32), 12, 118, 16);
  const double sh_b = 3.0 * kI * kD + 3 * (kI / 32) * (kD / 32);  // the shared expert's bytes
  if (sl < 0) {
    mw.s1 = dev_fill(E * s13, 2, 120, 8), mw.s3 = dev_fill(E * s13, 4, 120, 8), mw.s2 = dev_fill(E * s2b, 6, 120, 8);
  } else {
    const char* file = getenv("MS_FILE") ? getenv("MS_FILE") : "/data/dsgb10/weights/rank0.dsw";
    std::vector<uint8_t> h1, h3, h2;
    read_expert_scales(file, sl, E, h1, h3, h2);
    uint8_t* d[3];
    for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaMalloc(&d[i], i < 2 ? E * s13 : E * s2b));
    CUDA_CHECK(cudaMemcpy(d[0], h1.data(), h1.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d[1], h3.data(), h3.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d[2], h2.data(), h2.size(), cudaMemcpyHostToDevice));
    mw.s1 = d[0], mw.s3 = d[1], mw.s2 = d[2];
    printf("moereal: layer %d's scales from %s\n", sl, file);
  }
  dsk::MoeWeights mwc = mw;  // mw with the scales' code (PF_CODE 1)
  if (code) {
    const size_t nc13 = dsk::scale_code_words(E, kI, kD / 32), nc2 = dsk::scale_code_words(E, kD, kI / 32);
    uint32_t* dc;
    CUDA_CHECK(cudaMalloc(&dc, (2 * nc13 + nc2) * 4));
    int b[3];
    const double f1 = dsk::scale_code(mw.s1, E, kI, kD / 32, dc, &b[0], 0);
    const double f3 = dsk::scale_code(mw.s3, E, kI, kD / 32, dc + nc13, &b[1], 0);
    const double f2 = dsk::scale_code(mw.s2, E, kD, kI / 32, dc + 2 * nc13, &b[2], 0);
    mwc.c1 = {dc, (uint32_t)b[0] * 0x01010101u};
    mwc.c3 = {dc + nc13, (uint32_t)b[1] * 0x01010101u};
    mwc.c2 = {dc + 2 * nc13, (uint32_t)b[2] * 0x01010101u};
    // the code against the scales, and each expert's raw groups
    std::vector<uint8_t> h[3];
    std::vector<uint32_t> hc[3];
    std::vector<int> raw[3];
    size_t bad = 0;
    for (int i = 0; i < 3; ++i) {
      const int N = i < 2 ? kI : kD, KB = i < 2 ? kD / 32 : kI / 32;
      h[i].resize((size_t)E * N * KB), hc[i].resize(i < 2 ? nc13 : nc2);
      CUDA_CHECK(cudaMemcpy(h[i].data(), i == 0 ? mw.s1 : i == 1 ? mw.s3 : mw.s2, h[i].size(), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hc[i].data(), dc + (size_t)i * nc13, hc[i].size() * 4, cudaMemcpyDeviceToHost));
      bad += check_code(h[i], hc[i], E, N, KB, b[i], raw[i]);
    }
    for (int e = 0; e < E; ++e) {
      up_b[1][e] = 2.0 * w13 + 2 * nc13 / E * 4.0 + 32.0 * (raw[0][e] + raw[1][e]);
      down_b[1][e] = w2b + nc2 / E * 4.0 + 32.0 * raw[2][e];
    }
    printf("moereal: the code: groups coded w1 %.2f %%, w3 %.2f %%, w2 %.2f %%; decode %s (%zu words wrong)\n", 100 * f1,
           100 * f3, 100 * f2, bad ? "WRONG" : "exact", bad);
  }
  const dsk::MoeWeights* cur = &mw;
  auto use = [&](Var v) {
    tune.dgrp = v.G, tune.upf = v.U, tune.tord = v.T;
    dsk::prefill_tuning(tune);
    cur = v.C ? &mwc : &mw;
  };
  uint8_t* xq = dev_fill((size_t)Mmax * kD, 13);
  uint8_t* xs = dev_fill((size_t)Mmax * kD / 32, 14, 118, 16);
  float *yf, *wts;
  int* idx_d;
  CUDA_CHECK(cudaMalloc(&yf, (size_t)Mmax * kD * 4));
  CUDA_CHECK(cudaMalloc(&wts, (size_t)Mmax * 24));
  CUDA_CHECK(cudaMalloc(&idx_d, (size_t)Mmax * 24));
  {
    std::vector<float> hw((size_t)Mmax * 6, 1.0f / 6);
    CUDA_CHECK(cudaMemcpy(wts, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  }
  void* ws;
  CUDA_CHECK(cudaMalloc(&ws, dsk::moe_prefill_workspace_bytes(Mmax, E)));
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaEvent_t ev[5];
  for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaDeviceSynchronize());

  struct Case {
    std::string label;
    int M, L0, L1, call, file_m;
    std::string turn;  // an engine dump's prefix (PF_TURNS); empty: PF_ROUTES's files
  };
  std::vector<Case> cases;
  if (pre) {
    cases.push_back({"replay", 128, 20, 40, 0, 128, ""});
    for (int M : Ms) cases.push_back({"encoder", M, 0, 20, call, 8192, ""});
  }
  for (const char* p = turns; p && *p;) {
    while (*p == ' ') ++p;
    const char* q = p;
    while (*q && *q != ' ') ++q;
    if (q == p) break;
    const std::string t(p, q);
    const size_t m = t.rfind(".m");
    const int M = m == std::string::npos ? 0 : atoi(t.c_str() + m + 2);
    if (M < 1 || M > Mmax) die("moereal: %s: no .m<M> in 1..%d", t.c_str(), Mmax);
    cases.push_back({t.substr(t.rfind('/') + 1), M, 0, 40, 0, M, t});
    p = q;
  }
  printf("prefill MoE on real routes (%s%s%s): ms per layer, mean over the layers; floor = touched experts' bytes at %.0f"
         " GB/s; down ring %d\n", pre ? pre : "", pre && turns ? ", " : "", turns ? turns : "", kBw, tune.dring);
  printf("  %-12s %5s %3s %3s %2s %2s %1s %2s %7s %6s | %7s %7s %4s %5s | %7s %7s %4s %5s | %7s | %7s %7s %4s %5s %7s\n", "case",
         "M", "lay", "G", "U", "T", "C", "K", "experts", "tiles", "up", "floor", "%", "GB/s", "down", "floor", "%", "GB/s",
         "combine", "MoE", "floor", "%", "GB/s", "sum");
  for (const Case& c : cases) {
    const bool fused = dsk::moe_prefill_fuses(c.M);
    const size_t ny = (size_t)c.M * kD;
    struct Sum {
      double up = 0, down = 0, comb = 0, all = 0;
    };
    std::vector<Sum> sum(var.size());
    std::vector<int> same(var.size(), -1);  // the first layer's outputs against the default kernels'
    double s_n = 0, s_tiles = 0, sb[2][2] = {};  // sb: the touched experts' bytes, [raw, code][up, down]
    int nl = 0;
    for (int L = c.L0; L < c.L1; ++L) {
      char path[600];
      if (c.turn.empty())
        snprintf(path, sizeof path, "%s.L%02d.c%02d.m%d.i32", pre, L, c.call, c.file_m);
      else
        snprintf(path, sizeof path, "%s.L%02d.i32", c.turn.c_str(), L);
      std::vector<int> h;
      if (!load_routes(path, c.file_m, h)) {
        if (c.turn.empty()) printf("  (no routes: %s)\n", path);
        continue;
      }
      h.resize((size_t)c.M * 6);
      std::vector<int> cnt(E, 0);
      for (int v : h) {
        if (v < 0 || v >= E) die("moereal: expert %d in %s", v, path);
        ++cnt[v];
      }
      int n = 0, tiles = 0;
      for (int k : cnt) n += k > 0, tiles += (k + 63) / 64;
      double lb[2][2] = {};  // this layer's
      for (int e = 0; e < E; ++e)
        for (int C = 0; C < 2 && cnt[e]; ++C) lb[C][0] += up_b[C][e], lb[C][1] += down_b[C][e];
      CUDA_CHECK(cudaMemcpy(idx_d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
      dsk::MoePCfg cfg;
      cfg.phases = 2;
      dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st, cfg);
      const bool check = var.size() > 1 && nl == 0;
      std::vector<float> ref;
      if (check) {
        use({0, 0, 0, 0, 1});
        dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        ref.resize(ny);
        CUDA_CHECK(cudaMemcpy(ref.data(), yf, ny * 4, cudaMemcpyDeviceToHost));
      }
      for (size_t k = 0; k < var.size(); ++k) {
        use(var[k]);
        const int K = var[k].K;
        auto down_k = [&](int phases) {  // down in the variant's column groups
          for (int g = 0; g < K; ++g) {
            cfg.phases = phases, cfg.col0 = g * (kD / K), cfg.cols = K > 1 ? kD / K : 0;
            dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st, cfg);
          }
          cfg.col0 = 0, cfg.cols = 0;
        };
        std::vector<double> tu, td, tc, ta;
        for (int r = 0; r < rounds; ++r) {
          cfg.phases = 1;
          dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st, cfg);
          CUDA_CHECK(cudaEventRecord(ev[0], st));
          cfg.phases = 4;
          dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st, cfg);
          CUDA_CHECK(cudaEventRecord(ev[1], st));
          down_k(8);
          CUDA_CHECK(cudaEventRecord(ev[2], st));
          if (!fused) {
            cfg.phases = 16;
            dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st, cfg);
          }
          CUDA_CHECK(cudaEventRecord(ev[3], st));
          if (K == 1) {
            dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st);
          } else {
            cfg.phases = 1 | 2 | 4;
            dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st, cfg);
            down_k(fused ? 8 | 16 : 8);
            if (!fused) {
              cfg.phases = 16;
              dsk::moe_prefill(xq, xs, c.M, idx_d, wts, E, *cur, ws, yf, st, cfg);
            }
          }
          CUDA_CHECK(cudaEventRecord(ev[4], st));
          CUDA_CHECK(cudaEventSynchronize(ev[4]));
          float ms[4];
          for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
          tu.push_back(ms[0]), td.push_back(ms[1]), tc.push_back(ms[2]), ta.push_back(ms[3]);
        }
        if (check) {
          std::vector<float> y(ny);
          CUDA_CHECK(cudaMemcpy(y.data(), yf, ny * 4, cudaMemcpyDeviceToHost));
          same[k] = memcmp(y.data(), ref.data(), ny * 4) == 0;
        }
        auto med = [](std::vector<double>& v) {
          std::sort(v.begin(), v.end());
          return v[v.size() / 2];
        };
        const double up = med(tu), down = med(td), comb = med(tc), all = med(ta);
        const double *bl = lb[var[k].C], ab = bl[0] + bl[1] + sh_b;
        if (verbose)
          printf("    L%02d %12d %3s %3d %2d %2d %1d %2d %7d %6d | %7.3f %7.3f %4.0f %5.0f | %7.3f %7.3f %4.0f %5.0f | %7.3f | %7.3f"
                 " %7.3f %4.0f %5.0f\n",
                 L, c.M, "", var[k].G, var[k].U, var[k].T, var[k].C, var[k].K, n, tiles, up, bl[0] / kBw / 1e6,
                 100 * bl[0] / kBw / 1e6 / up, bl[0] / up / 1e6, down, bl[1] / kBw / 1e6, 100 * bl[1] / kBw / 1e6 / down,
                 bl[1] / down / 1e6, comb, all, ab / kBw / 1e6, 100 * ab / kBw / 1e6 / all, ab / all / 1e6);
        sum[k].up += up, sum[k].down += down, sum[k].comb += comb, sum[k].all += all;
      }
      s_n += n, s_tiles += tiles, ++nl;
      for (int C = 0; C < 2; ++C) sb[C][0] += lb[C][0], sb[C][1] += lb[C][1];
    }
    if (!nl) continue;
    for (size_t k = 0; k < var.size(); ++k) {
      const Sum& s = sum[k];
      const double *b = sb[var[k].C], ab = b[0] + b[1] + nl * sh_b;  // summed over the layers
      const double fu = b[0] / kBw / 1e6 / nl, fd = b[1] / kBw / 1e6 / nl, fa = ab / kBw / 1e6 / nl;
      printf("  %-12s %5d %3d %3d %2d %2d %1d %2d %7.1f %6.1f | %7.3f %7.3f %4.0f %5.0f | %7.3f %7.3f %4.0f %5.0f | %7.3f | %7.3f"
             " %7.3f %4.0f %5.0f %7.2f%s%s\n",
             c.label.c_str(), c.M, nl, var[k].G, var[k].U, var[k].T, var[k].C, var[k].K, s_n / nl, s_tiles / nl, s.up / nl, fu,
             100 * fu * nl / s.up, b[0] / s.up / 1e6, s.down / nl, fd, 100 * fd * nl / s.down, b[1] / s.down / 1e6,
             s.comb / nl, s.all / nl, fa, 100 * fa * nl / s.all, ab / s.all / 1e6, s.all, fused ? "  fused combine" : "",
             same[k] < 0 ? "" : same[k] ? "  bits: same" : "  bits: DIFFER");
    }
  }
}

// ---- the experts' 1-bit scale code (docs/engine.md, "Compact expert scales"; bench only): moe() on the raw
// scales against the same call reading scale_code's code, interleaved over MS_ROUNDS (default 15) replays a
// case, with the outputs (y, and moe_up's in the workspace) compared bit for bit. The scales are real: the
// first 96 experts of each layer in MS_LAYERS (default "20 39 -1"; -1: synth_scales) of MS_FILE (default rank
// 0's file), read once with their pages dropped; the code's decode is checked against them on the host. The
// weights are random, as in the moe bench. "projected": the raw call's time scaled by the bytes the code saves
// on the call's experts (a group's 32 bytes against its code word, or the raw group when flagged, plus the
// flags); "realised": the measured share of it. The engine's variants: up 0, down 2 at one row and 4 above,
// up16 4 and down16 2 at 9..16 rows (the MoeCfg defaults).

// Synthetic scales for the decode's corners (MS_LAYERS entry -1): expert e's 32-scale groups raw (one scale off
// the pair {120, 121}) with probability (e % 12) / 11, so tiles carry from none to all of their groups raw; the
// other scales 120 or 121.
static void synth_scales(std::vector<uint8_t>& s, int E, uint64_t seed) {
  const size_t per = s.size() / E;
  uint64_t x = seed;
  auto next = [&] {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(x >> 33);
  };
  for (int e = 0; e < E; ++e)
    for (size_t gi = 0; gi < per / 32; ++gi) {
      uint8_t* grp = &s[e * per + gi * 32];
      for (int i = 0; i < 32; ++i) grp[i] = next() % 5 < 3 ? 120 : 121;
      if (next() % 11 < (uint32_t)(e % 12)) {
        const int i = next() % 32;
        grp[i] = (uint8_t)(116 + next() % 4 + (next() & 1) * 6);  // 116 .. 119 or 122 .. 125
      }
    }
}

// The code against the raw scales s [E][N][KB] (L3): mismatched words (0 when exact), and per expert its raw groups.
static size_t check_code(const std::vector<uint8_t>& s, const std::vector<uint32_t>& code, int E, int N, int KB, int base,
                         std::vector<int>& raw) {
  const int T = N / 16, P = KB / 2, BW = dsk::code_block_words(P);
  const uint32_t b4 = (uint32_t)base * 0x01010101u;
  size_t bad = 0;
  raw.assign(E, 0);
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < T; ++t) {
      const uint32_t* blk = &code[((size_t)e * T + t) * BW];
      for (int p = 0; p < P; ++p) {
        const bool r = blk[P + p / 32] >> (p % 32) & 1u;
        raw[e] += r;
        bad += r && blk[p] != 0;
        for (int g = 0; g < 8 && !r; ++g) {
          uint32_t want;
          memcpy(&want, &s[(((size_t)e * T + t) * P + p) * 32 + g * 4], 4);
          bad += (((blk[p] >> g) & 0x01010101u) + b4) != want;
        }
      }
    }
  return bad;
}

static void bench_mscale() {
  const char* file = getenv("MS_FILE") ? getenv("MS_FILE") : "/data/dsgb10/weights/rank0.dsw";
  std::vector<int> layers = env_ints("MS_LAYERS"), rr = env_ints("MS_ROUNDS");
  if (layers.empty()) layers = {20, 39, -1};
  const int rounds = rr.empty() ? 15 : rr[0];
  const int NE = 96, kD = 5120, kI = 576, nsh = 8, copies = 48;
  const size_t w13 = (size_t)kI * kD / 2, s13 = (size_t)kI * kD / 32, w2 = (size_t)kD * kI / 2, s2 = (size_t)kD * kI / 32;
  const size_t per_expert = 2 * (w13 + s13) + w2 + s2;
  const size_t sh_bytes = 2 * ((size_t)kI * kD + kI / 32 * (kD / 32)) + (size_t)kD * kI + kD / 32 * (kI / 32);

  dsk::MoeWeights base{};
  base.w1 = dev_fill(NE * w13, 1), base.w3 = dev_fill(NE * w13, 3), base.w2 = dev_fill(NE * w2, 5);
  uint8_t *ds1, *ds3, *ds2;
  uint32_t *dc1, *dc3, *dc2;
  const size_t nc13 = dsk::scale_code_words(NE, kI, kD / 32), nc2 = dsk::scale_code_words(NE, kD, kI / 32);
  CUDA_CHECK(cudaMalloc(&ds1, NE * s13));
  CUDA_CHECK(cudaMalloc(&ds3, NE * s13));
  CUDA_CHECK(cudaMalloc(&ds2, NE * s2));
  CUDA_CHECK(cudaMalloc(&dc1, nc13 * 4));
  CUDA_CHECK(cudaMalloc(&dc3, nc13 * 4));
  CUDA_CHECK(cudaMalloc(&dc2, nc2 * 4));
  base.s1 = ds1, base.s3 = ds3, base.s2 = ds2;
  std::vector<dsk::MoeWeights> W(nsh, base);
  for (int i = 0; i < nsh; ++i) {
    W[i].sh_w1 = dev_fill((size_t)kI * kD, 10 + i), W[i].sh_s1 = dev_fill(kI / 32 * (kD / 32), 20 + i, 118, 16);
    W[i].sh_w3 = dev_fill((size_t)kI * kD, 30 + i), W[i].sh_s3 = dev_fill(kI / 32 * (kD / 32), 40 + i, 118, 16);
    W[i].sh_w2 = dev_fill((size_t)kD * kI, 50 + i), W[i].sh_s2 = dev_fill(kD / 32 * (kI / 32), 60 + i, 118, 16);
  }
  std::vector<dsk::MoeWeights> WC = W;
  uint8_t* xq = dev_fill(16 * kD, 7);
  uint8_t* xs = dev_fill(16 * kD / 32, 8, 118, 16);
  uint8_t* ws;
  float* y;
  const size_t wsb = dsk::moe_workspace_bytes();
  CUDA_CHECK(cudaMalloc(&ws, wsb));
  CUDA_CHECK(cudaMemset(ws, 0, wsb));
  CUDA_CHECK(cudaMalloc(&y, 16 * kD * 4));
  int* idx;
  float* wts;
  CUDA_CHECK(cudaMalloc(&idx, copies * 96 * 4));
  CUDA_CHECK(cudaMalloc(&wts, copies * 96 * 4));
  std::vector<float> hw(copies * 96, 0.25f);
  CUDA_CHECK(cudaMemcpy(wts, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  printf("mscale: %s, %d experts a layer, the weights random; moe() raw scales against the 1-bit code, %d rounds\n",
         file, NE, rounds);

  for (int L : layers) {
    std::vector<uint8_t> h1(NE * s13), h3(NE * s13), h2(NE * s2);
    if (L < 0)
      synth_scales(h1, NE, 1), synth_scales(h3, NE, 3), synth_scales(h2, NE, 5);
    else
      read_expert_scales(file, L, NE, h1, h3, h2);
    CUDA_CHECK(cudaMemcpy(ds1, h1.data(), h1.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds3, h3.data(), h3.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds2, h2.data(), h2.size(), cudaMemcpyHostToDevice));
    int b1, b3, b2;
    const double f1 = dsk::scale_code(ds1, NE, kI, kD / 32, dc1, &b1, st);
    const double f3 = dsk::scale_code(ds3, NE, kI, kD / 32, dc3, &b3, st);
    const double f2 = dsk::scale_code(ds2, NE, kD, kI / 32, dc2, &b2, st);
    std::vector<uint32_t> c1(nc13), c3(nc13), c2(nc2);
    CUDA_CHECK(cudaMemcpy(c1.data(), dc1, nc13 * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c3.data(), dc3, nc13 * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c2.data(), dc2, nc2 * 4, cudaMemcpyDeviceToHost));
    std::vector<int> r1, r3, r2;
    const size_t bad = check_code(h1, c1, NE, kI, kD / 32, b1, r1) + check_code(h3, c3, NE, kI, kD / 32, b3, r3) +
                       check_code(h2, c2, NE, kD, kI / 32, b2, r2);
    // bytes the code saves an expert: 28 a coded group, less the flags (16 B a w1/w3 tile, 4 a w2 tile)
    std::vector<double> saved(NE);
    const int G = 36 * 80;  // groups a matrix: w1/w3 36 tiles x 80 steps; w2 320 x 9 = 2880 too
    for (int e = 0; e < NE; ++e)
      saved[e] = 28.0 * (3 * G - r1[e] - r3[e] - r2[e]) - 2 * 36 * 16 - 320 * 4;
    printf("%s %d: groups coded w1 %.2f %% (pair {%d, %d}), w3 %.2f %% ({%d, %d}), w2 %.2f %% ({%d, %d}); decode %s"
           " (%zu words wrong)\n",
           L < 0 ? "synthetic" : "layer", L, 100 * f1, b1, b1 + 1, 100 * f3, b3, b3 + 1, 100 * f2, b2, b2 + 1, bad ? "WRONG" : "exact", bad);
    for (auto& w : WC) {
      w.c1 = {dc1, (uint32_t)b1 * 0x01010101u};
      w.c3 = {dc3, (uint32_t)b3 * 0x01010101u};
      w.c2 = {dc2, (uint32_t)b2 * 0x01010101u};
    }
    struct Case { int M, nu; };
    const Case cases[] = {{1, 6}, {6, 12}, {6, 21}, {6, 36}, {8, 21}, {8, 48}, {12, 36}, {16, 64}};
    printf("  %-14s %8s %8s %7s %7s %6s | %8s %8s | %8s %8s | %s\n", "case", "raw us", "code us", "saved", "proj.",
           "real.", "up raw", "up code", "dn raw", "dn code", "bits");
    for (const Case cs : cases) {
      std::vector<int> h(copies * 96);
      double sv = 0;
      for (int i = 0; i < copies; ++i) {
        for (int m = 0; m < cs.M; ++m)
          for (int k = 0; k < 6; ++k) h[i * 96 + m * 6 + k] = (i * cs.nu + (m * 6 + k) % cs.nu) % NE;
        for (int j = 0; j < cs.nu; ++j) sv += saved[(i * cs.nu + j) % NE] / copies;
      }
      CUDA_CHECK(cudaMemcpy(idx, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
      dsk::MoeCfg cfg{};
      cfg.down = cs.M == 1 ? 2 : 4;
      // bits: every launch's y and moe_up's output, raw then coded
      const int RB = dsk::row_budget(cs.M);
      const size_t yb = (size_t)cs.M * kD * 4, hb = dsk::moe_hs_offset(cs.M) + (size_t)(RB * 6 + 1) * RB * 18;
      std::vector<uint8_t> ya(yb), yc(yb), wa(hb), wc(hb);
      bool same = true;
      for (int i = 0; i < copies; ++i) {
        dsk::moe(xq, xs, cs.M, idx + i * 96, wts + i * 96, W[i % nsh], ws, y, cfg, st);
        CUDA_CHECK(cudaMemcpyAsync(ya.data(), y, yb, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(wa.data(), ws, hb, cudaMemcpyDeviceToHost, st));
        dsk::moe(xq, xs, cs.M, idx + i * 96, wts + i * 96, WC[i % nsh], ws, y, cfg, st);
        CUDA_CHECK(cudaMemcpyAsync(yc.data(), y, yb, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(wc.data(), ws, hb, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        same = same && ya == yc && wa == wc;
      }
      auto run = [&](const std::vector<dsk::MoeWeights>& WW, int phases) {
        return [&, phases, cfg](cudaStream_t s) {
          dsk::MoeCfg c = cfg;
          c.phases = phases;
          for (int i = 0; i < copies; ++i) dsk::moe(xq, xs, cs.M, idx + i * 96, wts + i * 96, WW[i % nsh], ws, y, c, s);
        };
      };
      const auto ts = time_interleaved({run(W, 3), run(WC, 3), run(W, 1), run(WC, 1), run(W, 2), run(WC, 2)}, rounds);
      double med[6];
      for (int k = 0; k < 6; ++k) med[k] = ts[k][ts[k].size() / 2] / copies;
      const double bytes = cs.nu * (double)per_expert + sh_bytes, proj = med[0] * sv / bytes;
      char name[32];
      snprintf(name, sizeof name, "M=%d experts %d", cs.M, cs.nu);
      printf("  %-14s %8.1f %8.1f %7.2f %7.2f %5.0f%% | %8.1f %8.1f | %8.1f %8.1f | %s\n", name, med[0], med[1],
             med[0] - med[1], proj, 100 * (med[0] - med[1]) / proj, med[2], med[3], med[4], med[5],
             same ? "same" : "DIFFER");
    }
  }
}

// The head coded (dsk::bf16x_encode) against the bf16 head, as the model calls them: gemv_bf16_norm and
// gemv_bf16x_norm with chead (mt 2, wk 16), the coded one also at mt 1; and DSpark's plain head GEMV at its 5
// rows. On the real rank-0 head slice, since the coding's size depends on the values: HEAD_W (default
// ~/dsgb10-data/kerneltests/head_w.npy). A replay is 4 calls (cold weights: the head is 10x L2 + SLC); the
// variants interleave, H4_ROUNDS replays each (30); us a call.
static void bench_head4() {
  const char* hw = getenv("HEAD_W");
  const Npy w = npy_load(hw ? hw : std::string(getenv("HOME")) + "/dsgb10-data/kerneltests/head_w.npy");
  const int N = (int)w.shape[0], K = (int)w.shape[1];
  const size_t rb = (size_t)N * K * 2;
  uint8_t *nat, *W;
  CUDA_CHECK(cudaMalloc(&nat, rb));
  CUDA_CHECK(cudaMalloc(&W, rb));
  CUDA_CHECK(cudaMemcpy(nat, w.data.data(), rb, cudaMemcpyHostToDevice));
  dsk::tile_fp8(nat, W, N, 2 * K, 0);
  CUDA_CHECK(cudaFree(nat));
  size_t xb = 0;
  int raw = 0;
  uint8_t* X = dsk::bf16x_encode(W, N, K, &xb, &raw, 0);
  if (dsk::bf16x_check(X, W, N, K, 0)) die("head4: the coded head does not decode to the head");
  auto* x = reinterpret_cast<__nv_bfloat16*>(dev_fill((size_t)16 * K * 2, 5, 0x3c, 2));
  auto* nw = reinterpret_cast<__nv_bfloat16*>(dev_fill((size_t)K * 2, 6, 0x3c, 2));
  float* y;
  CUDA_CHECK(cudaMalloc(&y, (size_t)16 * N * 4));
  const int rounds = getenv("H4_ROUNDS") ? atoi(getenv("H4_ROUNDS")) : 30;
  printf("head4 [%d x %d]: bf16 %.2f MB, coded %.2f MB (%.4f; %d tiles raw); us a call: median [p10 p90] of %d replays\n",
         N, K, rb / 1e6, xb / 1e6, (double)xb / rb, raw, rounds);
  const dsk::GemvCfg ch{2, 1, 16}, c1{1, 1, 16};
  auto row = [&](const char* what, int M, const std::vector<std::function<void(cudaStream_t)>>& fns) {
    const auto ts = time_interleaved(fns, rounds);
    auto q = [&](int v, double f) { return ts[v][(size_t)(f * (ts[v].size() - 1))] / 4; };
    printf("  %-5s M=%-2d bf16 %7.1f [%7.1f %7.1f] %5.1f GB/s | coded %7.1f [%7.1f %7.1f] %5.1f GB/s, %5.1f eff | %+6.1f us "
           "%+5.1f%%",
           what, M, q(0, .5), q(0, .1), q(0, .9), rb / q(0, .5) / 1e3, q(1, .5), q(1, .1), q(1, .9), xb / q(1, .5) / 1e3,
           rb / q(1, .5) / 1e3, q(1, .5) - q(0, .5), 100 * (q(1, .5) / q(0, .5) - 1));
    if (fns.size() > 2) printf(" | coded mt 1 %7.1f", q(2, .5));
    printf("\n");
  };
  for (int M : {1, 2, 4, 6, 8, 12, 16})
    row("norm", M,
        {[&](cudaStream_t st) {
           for (int i = 0; i < 4; ++i) dsk::gemv_bf16_norm(x, nw, 1e-6f, M, W, N, K, y, nullptr, ch, st);
         },
         [&](cudaStream_t st) {
           for (int i = 0; i < 4; ++i) dsk::gemv_bf16x_norm(x, nw, 1e-6f, M, X, N, K, y, nullptr, ch, st);
         },
         [&](cudaStream_t st) {
           for (int i = 0; i < 4; ++i) dsk::gemv_bf16x_norm(x, nw, 1e-6f, M, X, N, K, y, nullptr, c1, st);
         }});
  row("plain", 5,
      {[&](cudaStream_t st) {
         for (int i = 0; i < 4; ++i) dsk::gemv_bf16(x, 5, W, N, K, y, nullptr, ch, false, st);
       },
       [&](cudaStream_t st) {
         for (int i = 0; i < 4; ++i) dsk::gemv_bf16x(x, 5, X, N, K, y, nullptr, ch, st);
       }});
  cudaFree(W), cudaFree(X), cudaFree(x), cudaFree(nw), cudaFree(y);
}

int main(int argc, char** argv) {
  std::string filter = argc > 1 ? argv[1] : "";
  bool sweep = getenv("SWEEP") != nullptr;
  const Shape fp8_shapes[] = {
      {"wq_a", 320, 5120}, {"wkv", 128, 5120},   {"wq_b", 8192, 1280}, {"wo_b", 5120, 2048},
      {"sh_w13", 1152, 5120}, {"sh_w2", 5120, 576}, {"ix_wq_b", 4096, 1280},
  };
  for (auto& s : fp8_shapes)
    if (filter.empty() || filter == "fp8" || filter == s.name) bench_fp8(s, sweep);
  if (filter.empty() || filter == "fp8" || filter == "pair") bench_fp8_pair(sweep);
  if (filter.empty() || filter == "moe") bench_moe(sweep);
  if (filter.empty() || filter == "w16" || filter == "woa") bench_w16("wo_a", 2048, 4096, true, sweep);
  if (filter.empty() || filter == "w16" || filter == "gate") bench_w16("gate", 384, 5120, false, sweep);
  if (filter.empty() || filter == "w16" || filter == "head") bench_w16("head", 32320, 5120, false, sweep);
  if (filter.empty() || filter == "small") bench_small();
  if (filter == "step") bench_step();
  if (filter == "prefill") bench_prefill();  // PF_M="2048 4096", PF_ROUNDS
  if (filter == "psmall") small_gemms::run(true);  // the dense GEMMs' small-M tiles, cold L2
  if (filter == "moereal") bench_moe_real();  // PF_ROUTES=<prefix> or PF_TURNS, PF_M, PF_SCALES, PF_CODE, PF_CGRP, ...
  if (filter == "mscale") bench_mscale();     // MS_FILE, MS_LAYERS, MS_ROUNDS
  if (filter == "headsz")  // bandwidth vs matrix size
    for (int n : {1024, 4096, 8192, 16384, 32320}) bench_w16("head", n, 5120, false, false);
  if (filter == "head4") bench_head4();  // HEAD_W, H4_ROUNDS
  return 0;
}
