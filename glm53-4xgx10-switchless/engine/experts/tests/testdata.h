// Random expert layers in the real formats for the tests and the bench: canonical tensors (as the checkpoint
// and the quantiser write them), packed with layout.h's packers, uploaded with their descriptors.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "experts/layout.h"

namespace gx {

static const float kSubA[4] = {0.5f, 2.f, 4.f, 6.f};  // S3x2 subset A (kS3x2Mags[0]), the best single subset
static const float kSubB[4] = {0.5f, 1.5f, 3.f, 6.f};  // S3x2 subset B (kS3x2Mags[1]), the measured partner (Q1)
static const float kSubC[4] = {0.f, 1.f, 3.f, 6.f};    // the tests' second subset before E1 (2026-10-14), with a 0

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
        // the odd pairs (pattern 4) keep the pre-E1 second subset with a 0: any pair decodes the same way
        const bool odd = &p == &kPatterns[4];
        m.push_back(p.f[j] == kNVFP4 ? make_q4(N, K, sd, s2) : make_s3(N, K, sd, s2, p.nsub[j], kSubA, odd ? kSubC : kSubB));
      }
    }
    finish(seed);
  }
  // the shared expert (synthetic FP8), the pool offsets and the id -> stored expert map, after m is filled
  void finish(uint64_t seed) {
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

// Real (or random) experts written by engine/weights/s3x2_pack.py dump / dump-random: index.tsv has a line
// "S recipe expert" per stored expert, then three "M proj fmt s2-bits-hex N K prefix"; each matrix's files are
// prefix.w / .s (the canonical slice), .packed (the Python packer's blocks), .desc (its MatDesc at off 0) and .val
// (the torch model's values without s2, bf16 [N][K]).
struct RealMat {
  std::string recipe, proj, prefix;
  int expert = 0;
  std::vector<uint8_t> desc;
  std::vector<uint16_t> val;
};
inline std::vector<uint8_t> slurp(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) die("%s: cannot open", p.c_str());
  std::vector<uint8_t> v;
  uint8_t buf[1 << 16];
  for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) v.insert(v.end(), buf, buf + n);
  fclose(f);
  return v;
}
inline void load_real(TestLayer& L, std::vector<RealMat>& rm, const std::string& dir, uint64_t seed) {
  FILE* f = fopen((dir + "/index.tsv").c_str(), "r");
  if (!f) die("%s/index.tsv: cannot open", dir.c_str());
  L.m.clear();
  rm.clear();
  char line[1024], a[64], b[64], c[512];
  std::string recipe;
  int expert = 0;
  while (fgets(line, sizeof line, f)) {
    if (line[0] == 'S') {
      if (sscanf(line, "S %63s %d", a, &expert) != 2) die("%s: bad line %s", dir.c_str(), line);
      recipe = a;
      continue;
    }
    unsigned fmt, s2b;
    int N, K;
    if (sscanf(line, "M %63s %u %x %d %d %511s", b, &fmt, &s2b, &N, &K, c) != 6) die("%s: bad line %s", dir.c_str(), line);
    HostMat m;
    m.fmt = fmt, m.N = N, m.K = K;
    memcpy(&m.s2, &s2b, 4);
    m.nsub = fmt == kS3 ? 2 : 1;
    memcpy(m.mags, kS3x2Mags, sizeof m.mags);
    const std::string p = dir + "/" + c;
    m.w = slurp(p + ".w"), m.s = slurp(p + ".s"), m.packed = slurp(p + ".packed");
    RealMat r;
    r.recipe = recipe, r.proj = b, r.prefix = c, r.expert = expert, r.desc = slurp(p + ".desc");
    const std::vector<uint8_t> v = slurp(p + ".val");
    r.val.resize(v.size() / 2);
    memcpy(r.val.data(), v.data(), v.size());
    const size_t wb = fmt == kS3 ? (size_t)N * K / 16 * 6 : (size_t)N * K / 2;
    if (m.w.size() != wb || m.s.size() != (size_t)N * K / 16 || m.packed.size() != mat_bytes(fmt, N, K) ||
        r.desc.size() != sizeof(MatDesc) || r.val.size() != (size_t)N * K)
      die("%s: sizes", p.c_str());
    L.m.push_back(std::move(m));
    rm.push_back(std::move(r));
  }
  fclose(f);
  if (L.m.empty() || L.m.size() % 3) die("%s: %zu matrices", dir.c_str(), L.m.size());
  L.NS = (int)L.m.size() / 3;
  L.finish(seed);
}

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
