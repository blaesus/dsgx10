// The dense prefill GEMMs' small-M tiles (PrefillTuning::small) against the default tiles, on random data at
// the row counts of +44 to +728 turns (docs/engine.md, "A small turn's time to first token"): every output
// bit for bit (test_kernels p_fp8, p_small) and, with timing, us and GB/s of the weights from a cold L2
// (bench_kernels psmall). Own rows (16, 32, 56, 88, 184 a rank): the router gate, the shared expert's
// w1/w3, wq_a + wkv, wo_b (every rank's slice, 4 launches); all rows (44 .. 728): wq_b, wo_a.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/dsk.h"

namespace small_gemms {

inline uint8_t* fp8_dev(size_t n, uint32_t seed) {  // e4m3 without NaN
  std::vector<uint8_t> h(n);
  std::mt19937 g(seed);
  for (auto& b : h) b = (uint8_t)(g() & 0xF7);
  uint8_t* d;
  cudaMalloc(&d, n);
  cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice);
  return d;
}
inline uint8_t* scale_dev(size_t n, uint32_t seed) {  // ue8m0 around 1
  std::vector<uint8_t> h(n);
  std::mt19937 g(seed);
  for (auto& b : h) b = (uint8_t)(122 + g() % 10);
  uint8_t* d;
  cudaMalloc(&d, n);
  cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice);
  return d;
}
inline uint16_t* bf16_dev(size_t n, uint32_t seed) {
  std::vector<uint16_t> h(n);
  std::mt19937 g(seed);
  for (auto& v : h) v = (uint16_t)(((g() & 1u) << 15) | ((118u + g() % 12) << 7) | (g() & 0x7Fu));
  uint16_t* d;
  cudaMalloc(&d, n * 2);
  cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice);
  return d;
}
template <class T>
inline T* zeros(size_t n) {
  T* d;
  cudaMalloc(&d, n * sizeof(T));
  cudaMemset(d, 0, n * sizeof(T));
  return d;
}

// returns the number of variants whose outputs differ from the default tiles'
inline int run(bool timing) {
  constexpr int MX = 728, D = 5120;
  auto* xq = fp8_dev((size_t)MX * D, 1);
  auto* xs = scale_dev((size_t)MX * D / 32, 2);
  auto* x16 = reinterpret_cast<__nv_bfloat16*>(bf16_dev((size_t)MX * D, 3));
  auto* wg = reinterpret_cast<uint8_t*>(bf16_dev((size_t)384 * D, 4));
  auto* yg = zeros<float>((size_t)MX * 384);
  auto *w1 = fp8_dev((size_t)576 * D, 5), *w3 = fp8_dev((size_t)576 * D, 6);
  auto *s1 = scale_dev((size_t)576 * D / 32, 7), *s3 = scale_dev((size_t)576 * D / 32, 8);
  auto* hq = zeros<uint8_t>((size_t)MX * 576);
  auto* hs = zeros<uint8_t>((size_t)MX * 576 / 32);
  auto *wa = fp8_dev((size_t)1280 * D, 9), *wk = fp8_dev((size_t)512 * D, 10);
  auto *sa = scale_dev((size_t)1280 * D / 32, 11), *sk = scale_dev((size_t)512 * D / 32, 12);
  auto* ya = zeros<__nv_bfloat16>((size_t)MX * 1280);
  auto* yk = zeros<__nv_bfloat16>((size_t)MX * 512);
  auto* qq = fp8_dev((size_t)MX * 1280, 13);
  auto* qs = scale_dev((size_t)MX * 1280 / 32, 14);
  auto* wqb = fp8_dev((size_t)8192 * 1280, 15);
  auto* sqb = scale_dev((size_t)8192 * 1280 / 32, 16);
  auto* yqb = zeros<__nv_bfloat16>((size_t)MX * 8192);
  auto* xo = reinterpret_cast<__nv_bfloat16*>(bf16_dev((size_t)MX * 8192, 17));
  auto* woa = fp8_dev((size_t)2048 * 4096, 18);
  auto* soa = scale_dev((size_t)2048 * 4096 / 32, 19);
  auto* oq = zeros<uint8_t>((size_t)MX * 2048);
  auto* os = zeros<uint8_t>((size_t)MX * 64);
  auto* bq = fp8_dev((size_t)MX * 2048, 20);
  auto* bs = scale_dev((size_t)MX * 64, 21);
  uint8_t *wob[4], *sob[4];
  __nv_bfloat16* yob[4];
  for (int i = 0; i < 4; ++i)
    wob[i] = fp8_dev((size_t)D * 2048, 22 + i), sob[i] = scale_dev((size_t)D * 2048 / 32, 26 + i),
    yob[i] = zeros<__nv_bfloat16>((size_t)MX * D);
  struct Out {
    const void* p;
    size_t row;
  };
  struct G {
    const char* name;
    bool own;
    double wbytes;
    std::vector<Out> outs;
  };
  const G gs[] = {
      {"router gate (bf16, N 384)", true, 384.0 * D * 2, {{yg, 384 * 4}}},
      {"shared w1/w3 + SwiGLU", true, 2.0 * 576 * D, {{hq, 576}, {hs, 576 / 32}}},
      {"wq_a + wkv", true, 1792.0 * D, {{ya, 1280 * 2}, {yk, 512 * 2}}},
      {"wo_b (4 slices)", true, 4.0 * D * 2048, {{yob[0], D * 2}, {yob[1], D * 2}, {yob[2], D * 2}, {yob[3], D * 2}}},
      {"wq_b", false, 8192.0 * 1280, {{yqb, 8192 * 2}}},
      {"wo_a", false, 2048.0 * 4096, {{oq, 2048}, {os, 64}}},
  };
  auto gemm = [&](int which, int M) {
    if (which == 0) dsk::gemm_bf16_p(x16, M, wg, 384, D, yg, 0);
    if (which == 1) dsk::gemm_fp8_swiglu_p(xq, xs, M, w1, s1, w3, s3, 576, D, hq, hs, 0);
    if (which == 2) {
      dsk::Fp8Seg segs[2] = {{wa, sa, ya, 1280}, {wk, sk, yk, 512}};
      dsk::gemm_fp8_p(xq, xs, M, segs, 2, D, 0);
    }
    if (which == 3)
      for (int i = 0; i < 4; ++i) {
        dsk::Fp8Seg seg{wob[i], sob[i], yob[i], D};
        dsk::gemm_fp8_p(bq, bs, M, &seg, 1, 2048, 0);
      }
    if (which == 4) {
      dsk::Fp8Seg seg{wqb, sqb, yqb, 8192};
      dsk::gemm_fp8_p(qq, qs, M, &seg, 1, 1280, 0);
    }
    if (which == 5) dsk::gemm_woa_p(xo, M, woa, soa, 2048, 4096, 1024, nullptr, oq, os, 0);
  };
  auto tune = [](int small) {
    dsk::PrefillTuning t;
    t.small = small;
    dsk::prefill_tuning(t);
  };
  auto grab = [&](const G& g, int M) {
    std::vector<uint8_t> h;
    for (const Out& o : g.outs) {
      const size_t at = h.size(), n = o.row * M;
      h.resize(at + n);
      cudaMemcpy(h.data() + at, o.p, n, cudaMemcpyDeviceToHost);
    }
    return h;
  };
  uint8_t* flush = zeros<uint8_t>((size_t)64 << 20);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  auto time_us = [&](int which, int M) {
    std::vector<float> v;
    for (int rep = 0; rep < 9; ++rep) {
      cudaMemsetAsync(flush, rep, (size_t)64 << 20, 0);
      cudaEventRecord(e0, 0);
      gemm(which, M);
      cudaEventRecord(e1, 0);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      v.push_back(ms * 1e3f);
    }
    std::sort(v.begin(), v.end());
    return (double)v[v.size() / 2];
  };
  int bad = 0;
  const int own[] = {1, 16, 32, 56, 88, 129, 184}, all[] = {1, 44, 128, 129, 200, 350, 728};
  std::printf("small-M tiles (PrefillTuning::small 1, 2) against the default tiles%s\n",
              timing ? ": us (GB/s of the weights), cold L2, median of 9" : "");
  for (int w = 0; w < 6; ++w) {
    const G& g = gs[w];
    for (int M : g.own ? own : all) {
      tune(0);
      gemm(w, M);
      cudaDeviceSynchronize();
      const std::vector<uint8_t> ref = grab(g, M);
      double t[3] = {0, 0, 0};
      if (timing) t[0] = time_us(w, M);
      std::string res;
      for (int s = 1; s <= 2; ++s) {
        tune(s);
        for (const Out& o : g.outs) cudaMemset(const_cast<void*>(o.p), 0xA5, o.row * M);
        gemm(w, M);
        const cudaError_t err = cudaDeviceSynchronize();
        const bool same = err == cudaSuccess && grab(g, M) == ref;
        if (!same) ++bad;
        res += same ? " same" : err != cudaSuccess ? " ERROR" : " DIFFER";
        if (timing) t[s] = time_us(w, M);
      }
      tune(0);
      if (timing)
        std::printf("  %-26s M %4d: default %7.1f (%3.0f)  small1 %7.1f (%3.0f)  small2 %7.1f (%3.0f)  bits:%s\n",
                    g.name, M, t[0], g.wbytes / t[0] * 1e-3, t[1], g.wbytes / t[1] * 1e-3, t[2], g.wbytes / t[2] * 1e-3,
                    res.c_str());
      else
        std::printf("  %-26s M %4d: small 1, 2 against the default:%s %s\n", g.name, M, res.c_str(),
                    res.find('D') == std::string::npos && res.find('E') == std::string::npos ? "ok" : "FAIL");
    }
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  void* bufs[] = {xq, xs, x16, wg, yg, w1, w3, s1, s3, hq, hs, wa, wk, sa, sk, ya, yk, qq, qs, wqb, sqb, yqb, xo,
                  woa, soa, oq, os, bq, bs, flush};
  for (void* b : bufs) cudaFree(b);
  for (int i = 0; i < 4; ++i) cudaFree(wob[i]), cudaFree(sob[i]), cudaFree(yob[i]);
  return bad;
}

}  // namespace small_gemms
