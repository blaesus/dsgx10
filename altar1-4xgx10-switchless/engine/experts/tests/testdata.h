// Random expert layers in the real formats for the tests and the bench: canonical tensors (as the checkpoint
// stores them), packed with layout.h's packers, uploaded with their descriptors. A layer has one format
// (LayerW::fmt): INT4 (altar-1's layers 4-76), BF16 (3, 77, 78) or NVFP4 (kept for the MTP layer's option).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "experts/layout.h"

namespace gx {

struct HostMat {
  uint32_t fmt = kINT4;
  int N = 0, K = 0;
  // NVFP4: w [N][K/2] e2m1, s [N][K/16] e4m3. INT4: w = the stored nibbles' bytes [N][K/2], s = bf16 scales
  // [N][K/32] (2 B each), zp [N/8][K/32]. BF16: w = bf16 bits [N][K] (2 B each).
  std::vector<uint8_t> w, s;
  std::vector<uint32_t> zp;
  float s2 = 1.f;
  std::vector<uint8_t> packed;
  const uint16_t* s16() const { return reinterpret_cast<const uint16_t*>(s.data()); }
  const uint16_t* w16() const { return reinterpret_cast<const uint16_t*>(w.data()); }
  float value(int n, int k) const {  // without s2; exact (every format's value is a bf16)
    switch (fmt) {
      case kNVFP4: return canon_q4(w.data(), s.data(), K, n, k);
      case kINT4: return bf16_bits_to_f(canon_i4(w.data(), s16(), zp.data(), K, n, k));
      case kBF16: return bf16_bits_to_f(w16()[(size_t)n * K + k]);
    }
    die("HostMat: format %u", fmt);
  }
  uint16_t value_bits(int n, int k) const {  // INT4 / BF16: the contract's bf16 bits
    if (fmt == kINT4) return canon_i4(w.data(), s16(), zp.data(), K, n, k);
    if (fmt == kBF16) return w16()[(size_t)n * K + k];
    die("HostMat::value_bits: format %u", fmt);
  }
};

// NVFP4 scale bytes: e4m3 in [2^-3, 2^3), positive
GX_HD uint8_t rnd_scale(uint64_t h) { return (uint8_t)(0x20 + h % 0x30); }
// INT4 scale: a positive bf16 in [2^-9, 2^-5) with a random mantissa (the checkpoint's experts' range is
// 2^-12 .. 2^-4: docs/findings.md A0), so values (u - u_z) s are ~1e-2 and the products are inexact as often as
// the real ones
GX_HD uint16_t rnd_i4_scale(uint64_t h) { return (uint16_t)(((118 + h % 4) << 7) | ((h >> 8) & 0x7F)); }
// BF16 value: sign, exponent 2^-12 .. 2^-4, random mantissa (no zero, NaN or infinity)
GX_HD uint16_t rnd_b16(uint64_t h) { return (uint16_t)(((h >> 20) & 1) << 15 | ((115 + h % 8) << 7) | ((h >> 8) & 0x7F)); }

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
inline void repack(HostMat& m) {
  m.packed.assign(mat_bytes(m.fmt, m.N, m.K), 0);
  if (m.fmt == kINT4) pack_int4(m.w.data(), m.s16(), m.zp.data(), m.N, m.K, m.packed.data());
  else if (m.fmt == kBF16) pack_bf16(m.w16(), m.N, m.K, m.packed.data());
  else pack_nvfp4(m.w.data(), m.s.data(), m.N, m.K, m.packed.data());
}
// INT4: random nibbles and zero points (all 16 values each), scales from rnd_i4_scale (or `scale` when non-zero)
inline HostMat make_i4(int N, int K, uint64_t seed, uint16_t scale = 0) {
  HostMat m;
  m.fmt = kINT4, m.N = N, m.K = K;
  m.w.resize((size_t)N * K / 2);
  m.s.resize((size_t)N * K / 32 * 2);
  m.zp.resize((size_t)N / 8 * K / 32);
  for (size_t i = 0; i < m.w.size(); ++i) m.w[i] = (uint8_t)mix64(seed * 1000003 + i);
  uint16_t* s = reinterpret_cast<uint16_t*>(m.s.data());
  for (size_t i = 0; i < m.s.size() / 2; ++i) s[i] = scale ? scale : rnd_i4_scale(mix64(seed * 7919 + i + (1ull << 40)));
  for (size_t i = 0; i < m.zp.size(); ++i) m.zp[i] = (uint32_t)mix64(seed * 104729 + i + (1ull << 41));
  repack(m);
  return m;
}
inline HostMat make_b16(int N, int K, uint64_t seed) {
  HostMat m;
  m.fmt = kBF16, m.N = N, m.K = K;
  m.w.resize((size_t)N * K * 2);
  uint16_t* w = reinterpret_cast<uint16_t*>(m.w.data());
  for (size_t i = 0; i < (size_t)N * K; ++i) w[i] = rnd_b16(mix64(seed * 1000003 + i));
  repack(m);
  return m;
}
// The BF16 matrix of an INT4 matrix's contract values (the dequantised checkpoint tensor)
inline HostMat b16_of(const HostMat& a) {
  HostMat m;
  m.fmt = kBF16, m.N = a.N, m.K = a.K;
  m.w.resize((size_t)a.N * a.K * 2);
  uint16_t* w = reinterpret_cast<uint16_t*>(m.w.data());
  for (int n = 0; n < a.N; ++n)
    for (int k = 0; k < a.K; ++k) w[(size_t)n * a.K + k] = a.value_bits(n, k);
  repack(m);
  return m;
}
inline HostMat make_mat(uint32_t fmt, int N, int K, uint64_t seed, float s2) {
  switch (fmt) {
    case kNVFP4: return make_q4(N, K, seed, s2);
    case kINT4: return make_i4(N, K, seed);
    case kBF16: return make_b16(N, K, seed);
  }
  die("make_mat: format %u", fmt);
}
inline uint32_t fmt_of(const std::string& s) {  // "i4", "b16", "q4"
  if (s == "i4") return kINT4;
  if (s == "b16") return kBF16;
  if (s == "q4") return kNVFP4;
  die("format %s: i4, b16 or q4", s.c_str());
}
inline const char* fmt_tag(uint32_t f) { return f == kINT4 ? "i4" : f == kBF16 ? "b16" : f == kNVFP4 ? "q4" : "?"; }

// a fused FP8 shared expert (GLM-5.3's; the kernels keep it): canonical e4m3 [N][K] and f32 block scales [N/128][K/128]
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

constexpr float kS2Up = 1.f / 192, kS2Down = 1.f / 64;  // NVFP4 layers' s2

// A layer of NS stored experts of one format; expert id e of the layer uses stored expert map[e]. With fused,
// a synthetic FP8 shared expert runs inside the kernels (GLM-5.3's arrangement, still supported); without it
// (altar-1: the shared expert is BF16 and runs through attn's GEMV) the kernels run none and ys is an input.
// Device copies on demand.
struct TestLayer {
  int NS = 0;
  uint32_t fmt = kINT4;
  bool fused = false;
  std::vector<HostMat> m;  // [NS * 3]
  HostFp8 sg, su, sd;
  std::vector<uint64_t> off;  // [NS * 3]
  size_t pool_bytes = 0;
  uint8_t* d_pool = nullptr;
  uint8_t *d_sg = nullptr, *d_su = nullptr, *d_sd = nullptr;
  float *d_sgs = nullptr, *d_sus = nullptr, *d_sds = nullptr;
  MatDesc* d_desc = nullptr;
  std::vector<int> map;  // [kE]

  void build(uint32_t f, int ns, uint64_t seed, bool fused_shared) {
    NS = ns, fmt = f, fused = fused_shared;
    m.clear();
    for (int e = 0; e < NS; ++e)
      for (int j = 0; j < 3; ++j) {
        const int N = j < 2 ? kI : kH, K = j < 2 ? kH : kI;
        m.push_back(make_mat(f, N, K, seed * 131 + e * 3 + j, j < 2 ? kS2Up : kS2Down));
      }
    finish(seed);
  }
  // the same layer with every matrix replaced by the BF16 matrix of its contract values (INT4 -> BF16)
  TestLayer dequantised() const {
    TestLayer b;
    b.NS = NS, b.fmt = kBF16, b.fused = fused;
    for (const HostMat& x : m) b.m.push_back(b16_of(x));
    b.sg = sg, b.su = su, b.sd = sd;
    b.map = map;
    b.layout_pool();
    return b;
  }
  void layout_pool() {
    off.clear();
    pool_bytes = 0;
    for (auto& x : m) {
      off.push_back(pool_bytes);
      pool_bytes += (x.packed.size() + 255) / 256 * 256;
    }
  }
  // the shared expert (synthetic FP8), the pool offsets and the id -> stored expert map, after m is filled
  void finish(uint64_t seed) {
    sg = make_fp8(kI, kH, seed * 17 + 1, 1.f / 1024);
    su = make_fp8(kI, kH, seed * 17 + 2, 1.f / 1024);
    sd = make_fp8(kH, kI, seed * 17 + 3, 1.f / 256);
    layout_pool();
    map.resize(kE);
    for (int e = 0; e < kE; ++e) map[e] = (int)(mix64(seed + 99 + e) % NS);
  }
  std::vector<MatDesc> descs(const std::vector<int>& mp) const {
    std::vector<MatDesc> d(kE * 3);
    for (int e = 0; e < kE; ++e)
      for (int j = 0; j < 3; ++j) {
        const HostMat& x = m[mp[e] * 3 + j];
        d[e * 3 + j] = make_desc(off[mp[e] * 3 + j], x.s2, x.fmt);
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
    if (fused) {
      up8(sg, d_sg, d_sgs);
      up8(su, d_su, d_sus);
      up8(sd, d_sd, d_sds);
    }
    CUDA_CHECK(cudaMalloc(&d_desc, kE * 3 * sizeof(MatDesc)));
    auto d = descs(map);
    CUDA_CHECK(cudaMemcpy(d_desc, d.data(), d.size() * sizeof(MatDesc), cudaMemcpyHostToDevice));
  }
  LayerW layer() const {
    LayerW L;
    L.pool = d_pool, L.desc = d_desc, L.fmt = fmt;
    if (fused) L.sh = SharedW{d_sg, d_su, d_sd, d_sgs, d_sus, d_sds};
    return L;
  }
  void free_dev() {
    for (void* p : {(void*)d_pool, (void*)d_sg, (void*)d_su, (void*)d_sd, (void*)d_sgs, (void*)d_sus, (void*)d_sds, (void*)d_desc})
      if (p) CUDA_CHECK(cudaFree(p));
    d_pool = nullptr, d_sg = d_su = d_sd = nullptr, d_sgs = d_sus = d_sds = nullptr, d_desc = nullptr;
  }
};

// Real (or random) experts written by engine/weights/int4_pack.py dump / dump-random: index.tsv has a line
// "S tag expert" per stored expert, then three "M proj fmt s2-bits-hex N K prefix"; each matrix's files are
// prefix.wp / .s / .zp (INT4) or .w (BF16) (the rank's canonical slice), .packed (the Python packer's blocks), .desc
// (its MatDesc at off 0) and .val (the contract's values, bf16 [N][K], from int4_pack.value_bits).
struct RealMat {
  std::string tag, proj, prefix;
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
// All matrices of the dump in index order (any mix of formats)
inline void load_real(std::vector<HostMat>& hm, std::vector<RealMat>& rm, const std::string& dir) {
  FILE* f = fopen((dir + "/index.tsv").c_str(), "r");
  if (!f) die("%s/index.tsv: cannot open", dir.c_str());
  hm.clear();
  rm.clear();
  char line[1024], a[64], b[64], c[512];
  std::string tag;
  int expert = 0;
  while (fgets(line, sizeof line, f)) {
    if (line[0] == 'S') {
      if (sscanf(line, "S %63s %d", a, &expert) != 2) die("%s: bad line %s", dir.c_str(), line);
      tag = a;
      continue;
    }
    unsigned fmt, s2b;
    int N, K;
    if (sscanf(line, "M %63s %u %x %d %d %511s", b, &fmt, &s2b, &N, &K, c) != 6) die("%s: bad line %s", dir.c_str(), line);
    if (fmt != kINT4 && fmt != kBF16) die("%s: format %u (INT4 2 or BF16 3)", dir.c_str(), fmt);
    HostMat m;
    m.fmt = fmt, m.N = N, m.K = K;
    memcpy(&m.s2, &s2b, 4);
    const std::string p = dir + "/" + c;
    if (fmt == kINT4) {
      m.w = slurp(p + ".wp"), m.s = slurp(p + ".s");
      const std::vector<uint8_t> z = slurp(p + ".zp");
      m.zp.resize(z.size() / 4);
      memcpy(m.zp.data(), z.data(), z.size());
      if (m.w.size() != (size_t)N * K / 2 || m.s.size() != (size_t)N * K / 32 * 2 || m.zp.size() != (size_t)N / 8 * K / 32)
        die("%s: INT4 sizes", p.c_str());
    } else {
      m.w = slurp(p + ".w");
      if (m.w.size() != (size_t)N * K * 2) die("%s: BF16 size", p.c_str());
    }
    m.packed = slurp(p + ".packed");
    RealMat r;
    r.tag = tag, r.proj = b, r.prefix = c, r.expert = expert, r.desc = slurp(p + ".desc");
    const std::vector<uint8_t> v = slurp(p + ".val");
    r.val.resize(v.size() / 2);
    memcpy(r.val.data(), v.data(), v.size());
    if (m.packed.size() != mat_bytes(fmt, N, K) || r.desc.size() != sizeof(MatDesc) || r.val.size() != (size_t)N * K)
      die("%s: sizes", p.c_str());
    hm.push_back(std::move(m));
    rm.push_back(std::move(r));
  }
  fclose(f);
  if (hm.empty() || hm.size() % 3) die("%s: %zu matrices", dir.c_str(), hm.size());
}
// A layer of the dump's experts of format fmt (whole experts: gate, up, down), or NS = 0 if it has none
inline void real_layer(TestLayer& L, const std::vector<HostMat>& hm, uint32_t fmt, uint64_t seed) {
  L.m.clear();
  for (size_t i = 0; i + 2 < hm.size(); i += 3)
    if (hm[i].fmt == fmt) {
      if (hm[i + 1].fmt != fmt || hm[i + 2].fmt != fmt) die("real_layer: an expert of mixed formats");
      for (int j = 0; j < 3; ++j) L.m.push_back(hm[i + j]);
    }
  L.NS = (int)L.m.size() / 3, L.fmt = fmt, L.fused = false;
  if (L.NS) L.finish(seed);
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
