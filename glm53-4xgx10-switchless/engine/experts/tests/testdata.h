// Random expert layers in the real formats for the tests and the bench: canonical tensors (as the checkpoint
// and the quantiser write them), packed with layout.h's packers, uploaded with their descriptors.
#pragma once
#include <algorithm>
#include <cstdio>
#include <vector>

#include "common/check.h"
#include "experts/layout.h"

namespace gx {

static const float kSubA[4] = {0.5f, 2.f, 4.f, 6.f};  // quantisation.md 3.3's best single subset
static const float kSubB[4] = {0.f, 1.f, 3.f, 6.f};   // a second one, with 0

struct HostMat {
  uint32_t fmt = kNVFP4;
  int N = 0, K = 0;
  std::vector<uint8_t> w, s;  // NVFP4: [N][K/2], [N][K/16]; S3: codes [N][K/16][6], scales [N][K/16]
  float s2 = 1.f;
  int nsub = 1;
  float mags[2][4] = {};
  std::vector<uint8_t> packed;
  float value(int n, int k) const {  // without s2
    return fmt == kNVFP4 ? canon_q4(w.data(), s.data(), K, n, k) : canon_s3(w.data(), s.data(), mags, K, n, k);
  }
};

// scale bytes: e4m3 in [2^-3, 2^3), positive
GX_HD uint8_t rnd_scale(uint64_t h) { return (uint8_t)(0x20 + h % 0x30); }

inline HostMat make_q4(int N, int K, uint64_t seed, float s2) {
  HostMat m;
  m.fmt = kNVFP4, m.N = N, m.K = K, m.s2 = s2;
  m.w.resize((size_t)N * K / 2);
  m.s.resize((size_t)N * K / 16);
  for (size_t i = 0; i < m.w.size(); ++i) m.w[i] = (uint8_t)mix64(seed * 1000003 + i);
  for (size_t i = 0; i < m.s.size(); ++i) m.s[i] = rnd_scale(mix64(seed * 7919 + i + (1ull << 40)));
  m.packed.resize(mat_bytes(kNVFP4, N, K));
  pack_nvfp4(m.w.data(), m.s.data(), N, K, m.packed.data());
  return m;
}
inline HostMat make_s3(int N, int K, uint64_t seed, float s2, int nsub, const float* a, const float* b) {
  HostMat m;
  m.fmt = kS3, m.N = N, m.K = K, m.s2 = s2, m.nsub = nsub;
  for (int i = 0; i < 4; ++i) m.mags[0][i] = a[i], m.mags[1][i] = nsub > 1 ? b[i] : a[i];
  const size_t G = (size_t)N * K / 16;
  m.w.resize(G * 6);
  m.s.resize(G);
  for (size_t gi = 0; gi < G; ++gi) {
    const uint64_t h = mix64(seed * 1000003 + gi);
    for (int i = 0; i < 6; ++i) m.w[gi * 6 + i] = (uint8_t)(h >> (8 * i));
    const uint64_t h2 = mix64(seed * 7919 + gi + (1ull << 41));
    m.s[gi] = rnd_scale(h2) | (nsub > 1 && (h2 >> 32) & 1 ? 0x80 : 0);
  }
  m.packed.resize(mat_bytes(kS3, N, K));
  pack_s3(m.w.data(), m.s.data(), N, K, m.packed.data());
  return m;
}

// the shared expert: canonical e4m3 [N][K] and f32 block scales [N/128][K/128]
struct HostFp8 {
  int N = 0, K = 0;
  std::vector<uint8_t> w, packed;
  std::vector<float> s;
  float value(int n, int k) const { return e4m3_to_f(w[(size_t)n * K + k]) * s[(n / 128) * (K / 128) + k / 128]; }
};
inline HostFp8 make_fp8(int N, int K, uint64_t seed, float scale) {
  HostFp8 m;
  m.N = N, m.K = K;
  m.w.resize((size_t)N * K);
  for (size_t i = 0; i < m.w.size(); ++i) {
    uint8_t v = (uint8_t)mix64(seed * 1000003 + i);
    if ((v & 0x78) > 0x48) v &= 0xC7;  // magnitudes below 2^2-ish, no NaN
    m.w[i] = v;
  }
  m.s.resize((N / 128) * (K / 128));
  for (size_t i = 0; i < m.s.size(); ++i) m.s[i] = scale * (0.5f + (float)(mix64(seed + i) % 1000) / 1000.f);
  m.packed.resize(m.w.size());
  pack_fp8(m.w.data(), N, K, m.packed.data());
  return m;
}

// the stored experts' format patterns: gate, up, down
struct Pattern { uint32_t f[3]; int nsub[3]; };
static const Pattern kPatterns[] = {
    {{kNVFP4, kNVFP4, kNVFP4}, {1, 1, 1}},  // NVFP4 throughout
    {{kS3, kS3, kS3}, {1, 1, 1}},           // S3, one subset
    {{kS3, kS3, kS3}, {2, 2, 2}},           // S3, two subsets
    {{kS3, kS3, kNVFP4}, {2, 2, 1}},        // the mixture: 3 bits on gate/up, NVFP4 on down
    {{kNVFP4, kS3, kS3}, {1, 2, 1}},        // odd pairs
};
constexpr float kS2Up = 1.f / 192, kS2Down = 1.f / 64;

// A layer of NS stored experts (pattern p_of(e) for stored expert e); expert id e of the layer uses stored
// expert map[e]. Device copies on demand.
struct TestLayer {
  int NS = 0;
  std::vector<HostMat> m;  // [NS * 3]
  HostFp8 sg, su, sd;
  std::vector<uint64_t> off;  // [NS * 3]
  size_t pool_bytes = 0;
  uint8_t* d_pool = nullptr;
  uint8_t *d_sg = nullptr, *d_su = nullptr, *d_sd = nullptr;
  float *d_sgs = nullptr, *d_sus = nullptr, *d_sds = nullptr;
  MatDesc* d_desc = nullptr;
  std::vector<int> map;  // [kE]

  void build(int ns, uint64_t seed, int pattern = -1) {
    NS = ns;
    m.clear();
    for (int e = 0; e < NS; ++e) {
      const Pattern& p = kPatterns[pattern >= 0 ? pattern : e % 5];
      for (int j = 0; j < 3; ++j) {
        const int N = j < 2 ? kI : kH, K = j < 2 ? kH : kI;
        const uint64_t sd = seed * 131 + e * 3 + j;
        const float s2 = j < 2 ? kS2Up : kS2Down;
        m.push_back(p.f[j] == kNVFP4 ? make_q4(N, K, sd, s2) : make_s3(N, K, sd, s2, p.nsub[j], kSubA, kSubB));
      }
    }
    sg = make_fp8(kI, kH, seed * 17 + 1, 1.f / 1024);
    su = make_fp8(kI, kH, seed * 17 + 2, 1.f / 1024);
    sd = make_fp8(kH, kI, seed * 17 + 3, 1.f / 256);
    off.clear();
    pool_bytes = 0;
    for (auto& x : m) {
      off.push_back(pool_bytes);
      pool_bytes += (x.packed.size() + 255) / 256 * 256;
    }
    map.resize(kE);
    for (int e = 0; e < kE; ++e) map[e] = (int)(mix64(seed + 99 + e) % NS);
  }
  std::vector<MatDesc> descs(const std::vector<int>& mp) const {
    std::vector<MatDesc> d(kE * 3);
    for (int e = 0; e < kE; ++e)
      for (int j = 0; j < 3; ++j) {
        const HostMat& x = m[mp[e] * 3 + j];
        d[e * 3 + j] = make_desc(off[mp[e] * 3 + j], x.s2, x.fmt, x.nsub, x.mags);
      }
    return d;
  }
  void upload() {
    CUDA_CHECK(cudaMalloc(&d_pool, pool_bytes));
    for (size_t i = 0; i < m.size(); ++i)
      CUDA_CHECK(cudaMemcpy(d_pool + off[i], m[i].packed.data(), m[i].packed.size(), cudaMemcpyHostToDevice));
    auto up8 = [](const HostFp8& h, uint8_t*& w, float*& s) {
      CUDA_CHECK(cudaMalloc(&w, h.packed.size()));
      CUDA_CHECK(cudaMemcpy(w, h.packed.data(), h.packed.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(&s, h.s.size() * 4));
      CUDA_CHECK(cudaMemcpy(s, h.s.data(), h.s.size() * 4, cudaMemcpyHostToDevice));
    };
    up8(sg, d_sg, d_sgs);
    up8(su, d_su, d_sus);
    up8(sd, d_sd, d_sds);
    CUDA_CHECK(cudaMalloc(&d_desc, kE * 3 * sizeof(MatDesc)));
    auto d = descs(map);
    CUDA_CHECK(cudaMemcpy(d_desc, d.data(), d.size() * sizeof(MatDesc), cudaMemcpyHostToDevice));
  }
  LayerW layer() const { return LayerW{d_pool, d_desc, SharedW{d_sg, d_su, d_sd, d_sgs, d_sus, d_sds}}; }
  void free_dev() {
    for (void* p : {(void*)d_pool, (void*)d_sg, (void*)d_su, (void*)d_sd, (void*)d_sgs, (void*)d_sus, (void*)d_sds, (void*)d_desc})
      if (p) CUDA_CHECK(cudaFree(p));
    d_pool = nullptr;
  }
};

// bf16 activations, roughly unit RMS (a normed hidden state)
inline void rnd_x(std::vector<__nv_bfloat16>& x, size_t n, uint64_t seed) {
  x.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t h = mix64(seed * 7 + i);
    const float u = ((float)(h >> 40) * 0x1p-24f) * 2.f - 1.f;  // (-1, 1)
    x[i] = __float2bfloat16(u * 1.7f);
  }
}
// router logits: ~N(0, 1)-ish, with engineered ties when ties > 0
inline void rnd_logits(std::vector<float>& l, int M, uint64_t seed, bool ties) {
  l.resize((size_t)M * kE);
  for (size_t i = 0; i < l.size(); ++i) {
    const uint64_t h = mix64(seed * 13 + i);
    l[i] = ((float)(h >> 40) * 0x1p-24f - 0.5f) * 6.f;
    if (ties && (h & 7) == 0) l[i] = 1.25f;  // many equal logits
  }
}

// the CPU routing reference (route.cu's math, the same helper functions)
inline void route_ref(const float* l, const float* bias, int M, int* idx, float* w) {
  for (int m = 0; m < M; ++m) {
    float s[kE], key[kE];
    for (int e = 0; e < kE; ++e) s[e] = route_sigmoid(l[m * kE + e]), key[e] = s[e] + bias[e];
    float p[kTopK];
    for (int k = 0; k < kTopK; ++k) {
      int b = -1;
      for (int e = 0; e < kE; ++e)
        if (key[e] != -INFINITY && (b < 0 || key[e] > key[b])) b = e;
      idx[m * kTopK + k] = b;
      p[k] = s[b];
      key[b] = -INFINITY;
    }
    const float sum = route_sum8(p) + 1e-20f;
    for (int k = 0; k < kTopK; ++k) w[m * kTopK + k] = p[k] / sum * kRouteScale;
  }
}

}  // namespace gx
