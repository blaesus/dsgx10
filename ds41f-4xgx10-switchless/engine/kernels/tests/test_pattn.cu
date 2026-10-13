// Checks the prefill attention-side kernels (prefill_attn/index/comp/mhc.cu) against engine/attn's kernels
// on synthetic long-context state, and against the reference cases of engine/attn/tests (gen_attn.py,
// gen_mhc.py) like attn_tests does.
// usage: test_pattn [case root, default ~/dsgb10-data/attntests ("" for it)] [only: attn|index|comp|mhc]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "attn/tests/testutil.h"
#include "kernels/dsk.h"
#include "kernels/prefill_attn.h"

using namespace attn_test;

namespace {

std::string g_root;

// ---- host formats ---------------------------------------------------------------------------------
float h_e4m3(uint8_t b) {
  int e = (b >> 3) & 15, m = b & 7;
  float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
  return (b & 0x80) ? -v : v;
}
float h_e2m1(uint32_t nib) {
  static const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (nib & 8) ? -t[nib & 7] : t[nib & 7];
}
uint16_t h_f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float h_bfround(float f) { return bf2f(h_f2bf(f)); }
float h_rope_re(float x0, float x1, float c, float s) { return std::fmaf(x0, c, -(x1 * s)); }
float h_rope_im(float x0, float x1, float c, float s) { return std::fmaf(x1, c, x0 * s); }

// ---- synthetic data -------------------------------------------------------------------------------
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
uint32_t g_seed = 7;
void rnd_u8(Dev<uint8_t>& d, uint32_t mask, uint32_t add) { fill_u8<<<512, 256>>>(d.p, d.n, g_seed++, mask, add); }
void rnd_bf16(Dev<uint16_t>& d, float scale) { fill_bf16<<<512, 256>>>(d.p, d.n, g_seed++, scale); }
void rnd_f32(Dev<float>& d, float scale) { fill_f32<<<512, 256>>>(d.p, d.n, g_seed++, scale); }

// ---- attention yardstick ----------------------------------------------------------------------------
// relative L2 error per (row, head) against fp64, ours and a second result (attn's kernel or the reference)
struct Truth {
  double max_a = 0, max_b = 0, sum_a = 0, sum_b = 0;
  int n = 0;
  void add(const std::vector<double>& tr, const uint16_t* a, const uint16_t* b) {
    double nt = 0, ea = 0, eb = 0;
    for (size_t d = 0; d < tr.size(); ++d) {
      nt += tr[d] * tr[d];
      ea += (bf2f(a[d]) - tr[d]) * (bf2f(a[d]) - tr[d]);
      eb += (bf2f(b[d]) - tr[d]) * (bf2f(b[d]) - tr[d]);
    }
    ea = std::sqrt(ea / nt);
    eb = std::sqrt(eb / nt);
    max_a = std::max(max_a, ea);
    max_b = std::max(max_b, eb);
    sum_a += ea;
    sum_b += eb;
    ++n;
  }
  void report(const std::string& name, const char* other) const {
    bool ok = max_a <= std::max(1.5 * max_b, 2e-3);
    if (!ok) ++g_failures;
    std::printf("%-40s vs fp64 (rel L2 per head): ours max %.2e mean %.2e | %s max %.2e mean %.2e  %s\n", name.c_str(),
                max_a, sum_a / std::max(n, 1), other, max_b, sum_b / std::max(n, 1), ok ? "ok" : "FAIL");
  }
};
// one head of one row in fp64: q (bf16, RoPE applied), keys (dequantised), sink, then the inverse RoPE
void attn_truth(const float* q, const std::vector<std::vector<float>>& keys, double sink, const float* freqs,
                std::vector<double>& o) {
  const double scale = std::pow(512.0, -0.5);
  size_t n = keys.size();
  std::vector<double> s(n);
  double m = -1e300;
  for (size_t j = 0; j < n; ++j) {
    double acc = 0;
    for (int d = 0; d < 512; ++d) acc += (double)q[d] * keys[j][d];
    s[j] = acc * scale;
    m = std::max(m, s[j]);
  }
  double den = std::exp(sink - m);
  o.assign(512, 0.0);
  for (size_t j = 0; j < n; ++j) {
    double p = std::exp(s[j] - m);
    den += p;
    for (int d = 0; d < 512; ++d) o[d] += p * keys[j][d];
  }
  for (int d = 0; d < 512; ++d) o[d] /= den;
  for (int i = 0; i < 32; ++i) {
    double c = freqs[2 * i], sn = freqs[2 * i + 1], x0 = o[448 + 2 * i], x1 = o[449 + 2 * i];
    o[448 + 2 * i] = x0 * c + x1 * sn;
    o[449 + 2 * i] = x1 * c - x0 * sn;
  }
}
std::vector<float> win_key(const std::vector<uint8_t>& kv, const std::vector<uint8_t>& sc, int slot) {
  std::vector<float> k(512);
  for (int d = 0; d < 512; ++d) k[d] = h_e4m3(kv[(size_t)slot * 512 + d]) * std::ldexp(1.0f, sc[(size_t)slot * 16 + d / 32] - 127);
  return k;
}
std::vector<float> ckv_key(const std::vector<uint8_t>& kv, const std::vector<uint8_t>& sc, int e) {
  std::vector<float> k(512);
  for (int d = 0; d < 512; ++d) {
    uint8_t b = kv[(size_t)e * 256 + d / 2];
    k[d] = h_e2m1(d & 1 ? b >> 4 : b & 15) * h_e4m3(sc[(size_t)e * 32 + d / 16]);
  }
  return k;
}
// q of one head with RoPE at the position (table row tb), rounded to bf16 as the kernels do
std::vector<float> q_rope(const uint16_t* qh, const float* tb) {
  std::vector<float> q(512);
  for (int d = 0; d < 512; ++d) q[d] = bf2f(qh[d]);
  for (int i = 0; i < 32; ++i) {
    float x0 = q[448 + 2 * i], x1 = q[449 + 2 * i], c = tb[2 * i], s = tb[2 * i + 1];
    q[448 + 2 * i] = h_bfround(h_rope_re(x0, x1, c, s));
    q[449 + 2 * i] = h_bfround(h_rope_im(x0, x1, c, s));
  }
  return q;
}

// ---- sparse attention, synthetic -------------------------------------------------------------------------
// R rows ending at position ctx-1 of a ratio-2 layer, every row with kmax random entries of its causal range
// (rows near the start of the sequence keep fewer, padded with -1); both kernels against fp64.
void test_attn_synth(int R, int ctx, int kmax, int winscale_lo, int win_lo = 0) {
  char name[96];
  std::snprintf(name, sizeof name, "attn synth R=%d ctx=%d kmax=%d%s", R, ctx, kmax, win_lo ? " win_lo" : "");
  const int p0 = ctx - R, maxpos = ctx + 16, ncap = ctx / 2 + 1;
  Dev<float2> table((size_t)maxpos * 32);
  attn::rope_table_build(table.p, maxpos, true, 0);
  std::vector<int> ph(R);
  for (int r = 0; r < R; ++r) ph[r] = p0 + r;
  Dev<int> pos(ph.data(), R);
  Dev<uint16_t> q((size_t)R * 16 * 512), out_a((size_t)R * 16 * 512), out_p((size_t)R * 16 * 512);
  rnd_bf16(q, 3.0f);
  const int slots = R + 127;
  Dev<uint8_t> wkv((size_t)slots * 512), wsc((size_t)slots * 16), ckv((size_t)ncap * 256), csc((size_t)ncap * 32);
  rnd_u8(wkv, 0x77, 0);
  rnd_u8(wsc, 7, winscale_lo);
  rnd_u8(ckv, 0xFF, 0);
  rnd_u8(csc, 0x1F, 0x10);
  Dev<float> sink(16);
  rnd_f32(sink, 2.0f);
  std::vector<int> th((size_t)R * std::max(kmax, 1), -1);
  std::mt19937 rng(1234 + R);
  for (int r = 0; r < R && kmax > 0; ++r) {
    int n = (ph[r] + 1) / 2, k = std::min(kmax, n);
    if (r % 97 == 5) k = std::min(k, 37);  // a few short lists
    std::vector<int> pick;
    std::uniform_int_distribution<int> u(0, n - 1);
    std::vector<uint8_t> used(n, 0);
    while ((int)pick.size() < k) {
      int x = u(rng);
      if (!used[x]) used[x] = 1, pick.push_back(x);
    }
    std::sort(pick.begin(), pick.end());
    for (int i = 0; i < k; ++i) th[(size_t)r * kmax + i] = pick[i];
  }
  Dev<int> topk(th.data(), th.size());
  attn::Window win{wkv.p, wsc.p, -1, p0 - 127};
  attn::Compressed cc{ckv.p, csc.p, ncap};
  Dev<float> po((size_t)16 * 16 * 512), pml((size_t)16 * 16 * 2);
  std::vector<int> lh(R);
  for (int r = 0; r < R; ++r) lh[r] = std::max(win_lo, ph[r] - 127);
  Dev<int> lo(lh.data(), R);
  attn::sparse_attn(q.p, pos.p, attn::WinBounds{win_lo ? lo.p : nullptr, nullptr}, R, table.p, win, cc,
                    kmax ? topk.p : nullptr, kmax, sink.p, out_a.p, 1, attn::AttnScratch{po.p, pml.p, 16}, 0);
  dsk::sparse_attn_p(q.p, pos.p, R, table.p, win, cc, kmax ? topk.p : nullptr, kmax, sink.p, out_p.p, 0, win_lo);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto ha = out_a.host(), hp = out_p.host(), hq = q.host();
  report(std::string(name) + " vs attn", cmp_bf16(hp.data(), ha.data(), ha.size()), false, false);
  auto hk = wkv.host(), hs = wsc.host(), hc = ckv.host(), hcs = csc.host();
  auto ht = table.host();
  auto hsink = sink.host();
  Truth tr;
  std::vector<double> o;
  for (int r : {0, 1, 5, 63, 64, 127, 128, 200, R / 2, R - 2, R - 1}) {
    if (r < 0 || r >= R) continue;
    int p = ph[r];
    std::vector<std::vector<float>> keys;
    for (int kp = std::max(win_lo, p - 127); kp <= p; ++kp) keys.push_back(win_key(hk, hs, kp - (p0 - 127)));
    for (int i = 0; i < kmax; ++i) {
      int e = th[(size_t)r * kmax + i];
      if (e >= 0) keys.push_back(ckv_key(hc, hcs, e));
    }
    const float* tb = reinterpret_cast<const float*>(ht.data()) + (size_t)p * 64;
    for (int h = 0; h < 16; ++h) {
      size_t off = ((size_t)r * 16 + h) * 512;
      auto qr = q_rope(hq.data() + off, tb);
      attn_truth(qr.data(), keys, hsink[h], tb, o);
      tr.add(o, hp.data() + off, ha.data() + off);
    }
  }
  tr.report(name, "attn");
}

// ---- reference cases (gen_attn.py) -------------------------------------------------------------------
struct Case {
  std::string dir, json;
  std::vector<int> layers, ratios;
  int T = 0, D = 0, index_topk = 0, cand_blocks = 0;
  bool has(const std::string& n) const { return npy_exists(dir + "/" + n + ".npy"); }
  Npy get(const std::string& n) const { return npy_load(dir + "/" + n + ".npy"); }
  int ratio(int L) const {
    for (size_t i = 0; i < layers.size(); ++i)
      if (layers[i] == L) return ratios[i];
    return 0;
  }
};
int json_int(const std::string& j, const std::string& key) {
  auto p = j.find("\"" + key + "\":");
  if (p == std::string::npos) die("manifest: no %s", key.c_str());
  return std::stoi(j.substr(p + key.size() + 3));
}
std::vector<int> json_ints(const std::string& j, const std::string& key) {
  auto p = j.find("\"" + key + "\":");
  if (p == std::string::npos) die("manifest: no %s", key.c_str());
  p = j.find('[', p) + 1;
  auto e = j.find(']', p);
  std::vector<int> v;
  std::stringstream ss(j.substr(p, e - p));
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::stoi(tok));
  return v;
}
Case load_case(const std::string& dir) {
  Case c;
  c.dir = dir;
  std::ifstream f(dir + "/manifest.json");
  if (!f) die("no manifest in %s", dir.c_str());
  std::stringstream ss;
  ss << f.rdbuf();
  c.json = ss.str();
  c.layers = json_ints(c.json, "layers");
  c.ratios = json_ints(c.json, "compress_ratios");
  c.T = json_int(c.json, "T");
  c.D = json_int(c.json, "decode");
  c.index_topk = json_int(c.json, "index_topk");
  c.cand_blocks = json_int(c.json, "cand_blocks");
  return c;
}
std::string P(int L, int call, const char* what) { return "L" + std::to_string(L) + "_c" + std::to_string(call) + "_" + what; }
std::string W(int L, const char* what) { return "L" + std::to_string(L) + "_" + what; }
int kv_source(int L) { return L < 2 ? -1 : L < 8 ? 2 : L < 14 ? 8 : L < 20 ? 14 : 20; }
int index_source(int L) { return L < 20 ? kv_source(L) : 20 + (L - 20) / 4 * 4; }

// a kv-source layer's caches after the prefill call, from the reference's projections (attn's compressor)
struct CompState {
  int ratio = 0, cap = 0;
  Dev<float2> table;
  Dev<uint16_t> wkv, wgate, norm, wk, knorm;
  Dev<float> carry_kv, carry_score;
  Dev<uint8_t> ckv_kv, ckv_sc, ik_k, ik_sc;
  void build(const Case& cs, int L, const int* pos, int rows) {
    ratio = cs.ratio(L);
    int maxpos = cs.T + cs.D;
    cap = maxpos / ratio + 1;
    table = Dev<float2>((size_t)maxpos * 32);
    attn::rope_table_build(table.p, maxpos, true, 0);
    wkv = to_dev<uint16_t>(cs.get(W(L, "w_c_wkv")));
    if (ratio > 1) wgate = to_dev<uint16_t>(cs.get(W(L, "w_c_wgate")));
    norm = to_dev<uint16_t>(cs.get(W(L, "w_c_norm")));
    wk = to_dev<uint16_t>(cs.get(W(L, "w_i_wk")));
    knorm = to_dev<uint16_t>(cs.get(W(L, "w_i_knorm")));
    carry_kv = Dev<float>(16 * 512);
    carry_score = Dev<float>(16 * 512);
    ckv_kv = Dev<uint8_t>((size_t)cap * 256);
    ckv_sc = Dev<uint8_t>((size_t)cap * 32);
    ik_k = Dev<uint8_t>((size_t)cap * 64);
    ik_sc = Dev<uint8_t>((size_t)cap * 4);
    Dev<float> kv((size_t)rows * 512), score((size_t)rows * 512);
    if (ratio > 1) {
      auto rk = cs.get(P(L, 0, "c_kv")), rs = cs.get(P(L, 0, "c_score"));
      CUDA_CHECK(cudaMemcpy(kv.p, rk.as<float>(), rk.data.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(score.p, rs.as<float>(), rs.data.size(), cudaMemcpyHostToDevice));
    } else {
      auto ref = cs.get(P(L, 0, "c_proj"));
      std::vector<float> f(ref.numel());
      for (size_t i = 0; i < f.size(); ++i) f[i] = bf2f(ref.as<uint16_t>()[i]);
      CUDA_CHECK(cudaMemcpy(kv.p, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    attn::CompressorWeights w{ratio, wkv.p, wgate.p, norm.p, wk.p, knorm.p};
    attn::compressor_emit(kv.p, score.p, pos, rows, w, {carry_kv.p, carry_score.p}, table.p,
                          {ckv_kv.p, ckv_sc.p, cap}, {ik_k.p, ik_sc.p, cap}, nullptr, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

// The prefill call (call 0) of every layer of a case: window from kv_write into a linear buffer, entries
// from the compressor, indices from the reference's itopk; the four 16-head slices of the 64 heads.
void test_attn_case(const std::string& dir) {
  Case cs = load_case(dir);
  const int rows = cs.T;
  std::vector<int> ph(rows);
  for (int i = 0; i < rows; ++i) ph[i] = i;
  Dev<int> pos(ph.data(), rows);
  std::map<int, CompState> comp;
  for (int L : cs.layers)
    if (L == kv_source(L) && cs.has(W(L, "w_c_wkv"))) comp[L].build(cs, L, pos.p, rows);
  const int kmax_c = (cs.index_topk + 63) / 64 * 64;
  for (int L : cs.layers) {
    if (!cs.has(P(L, 0, "q_pre"))) continue;
    int maxpos = cs.T + cs.D;
    Dev<float2> table((size_t)maxpos * 32);
    attn::rope_table_build(table.p, maxpos, cs.ratio(L) > 0, 0);
    auto kvw = to_dev<uint16_t>(cs.get(W(L, "w_kv_norm")));
    auto sink_h = cs.get(W(L, "attn_sink"));
    auto sink = to_dev<float>(sink_h);
    auto kvp = to_dev<uint16_t>(cs.get(P(L, 0, "kv_pre")));
    Dev<uint8_t> lin_kv((size_t)rows * 512), lin_sc((size_t)rows * 16);
    attn::Window win{lin_kv.p, lin_sc.p, -1, 0};
    attn::kv_write(kvp.p, kvw.p, pos.p, table.p, win, nullptr, rows, 0);
    attn::Compressed ckv{nullptr, nullptr, 0};
    Dev<int> topk;
    int kmax = 0;
    if (cs.ratio(L) > 0) {
      CompState& src = comp.at(kv_source(L));
      ckv = attn::Compressed{src.ckv_kv.p, src.ckv_sc.p, src.cap};
      auto it = cs.get(P(index_source(L), 0, "itopk"));
      int kr = (int)it.shape[1];
      kmax = kmax_c;
      std::vector<int> h((size_t)rows * kmax, -1);
      for (int r = 0; r < rows; ++r)
        for (int i = 0; i < kr; ++i) h[(size_t)r * kmax + i] = it.as<int>()[(size_t)r * kr + i];
      topk = Dev<int>(h.data(), h.size());
    }
    auto qpre = cs.get(P(L, 0, "q_pre"));
    auto oref = cs.get(P(L, 0, "o"));
    std::vector<uint16_t> got(oref.numel());
    for (int hs = 0; hs < 4; ++hs) {
      std::vector<uint16_t> qh((size_t)rows * 16 * 512);
      for (int r = 0; r < rows; ++r)
        std::memcpy(qh.data() + (size_t)r * 16 * 512, qpre.as<uint16_t>() + ((size_t)r * 64 + hs * 16) * 512, 16 * 512 * 2);
      Dev<uint16_t> q(qh.data(), qh.size()), out(qh.size());
      dsk::sparse_attn_p(q.p, pos.p, rows, table.p, win, ckv, topk.p, kmax, sink.p + hs * 16, out.p, 0);
      auto ho = out.host();
      for (int r = 0; r < rows; ++r)
        std::memcpy(got.data() + ((size_t)r * 64 + hs * 16) * 512, ho.data() + (size_t)r * 16 * 512, 16 * 512 * 2);
    }
    std::string nm = dir.substr(dir.find_last_of('/') + 1) + " " + P(L, 0, "attn_o");
    report(nm + " vs model.py", cmp_bf16(got.data(), oref.as<uint16_t>(), oref.numel()), false, false);
    auto qref = cs.get(P(L, 0, "q"));
    auto freqs = cs.get(W(L, "freqs"));
    auto wkv = lin_kv.host(), wsc = lin_sc.host();
    std::vector<uint8_t> hk, hsc;
    std::vector<int> ht;
    if (kmax) {
      hk = comp.at(kv_source(L)).ckv_kv.host();
      hsc = comp.at(kv_source(L)).ckv_sc.host();
      ht = topk.host();
    }
    Truth tr;
    std::vector<double> o;
    for (int r : {0, 1, 63, 127, 128, 129, 200, 299, 450, 599}) {
      if (r >= rows) continue;
      int p = ph[r];
      std::vector<std::vector<float>> keys;
      for (int kp = std::max(0, p - 127); kp <= p; ++kp) keys.push_back(win_key(wkv, wsc, kp));
      for (int i = 0; i < kmax; ++i) {
        int e = ht[(size_t)r * kmax + i];
        if (e >= 0) keys.push_back(ckv_key(hk, hsc, e));
      }
      for (int h = 0; h < 64; ++h) {
        size_t off = ((size_t)r * 64 + h) * 512;
        std::vector<float> qv(512);
        for (int d = 0; d < 512; ++d) qv[d] = bf2f(qref.as<uint16_t>()[off + d]);
        attn_truth(qv.data(), keys, sink_h.as<float>()[h], freqs.as<float>() + (size_t)p * 64, o);
        tr.add(o, got.data() + off, oref.as<uint16_t>() + off);
      }
    }
    tr.report(nm, "model.py");
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

// ---- indexer -----------------------------------------------------------------------------------------
// head weights as index_q_prep makes them: bf16 values times 2^-6
__global__ void fill_w(float* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float v = ((hash32(i, seed) & 0xFFFFFF) * (1.0f / 8388608.0f) - 1.0f) * 2.0f;
    p[i] = __uint_as_float(__float_as_uint(v) & 0xffff0000u) * 0x1p-6f;
  }
}
int rows_equal(const std::vector<int>& a, const std::vector<int>& b, int rows, int k, int* first_bad) {
  int ok = 0;
  *first_bad = -1;
  for (int r = 0; r < rows; ++r) {
    bool eq = std::equal(a.begin() + (size_t)r * k, a.begin() + (size_t)(r + 1) * k, b.begin() + (size_t)r * k);
    ok += eq;
    if (!eq && *first_bad < 0) *first_bad = r;
  }
  return ok;
}
void report_rows(const std::string& name, int ok, int rows) {
  bool pass = ok == rows;
  if (!pass) ++g_failures;
  std::printf("%-44s rows equal %d/%d %s\n", name.c_str(), ok, rows, pass ? "ok" : "FAIL");
}
// keys over each row's [0, n) against attn's: element-wise equality
void report_keys(const std::string& name, const Dev<uint16_t>& got, const Dev<uint16_t>& ref, int rows, int stride, int n) {
  auto g = got.host(), r = ref.host();
  long bad = 0;
  int first = -1;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < n; ++j)
      if (g[(size_t)i * stride + j] != r[(size_t)i * stride + j]) {
        if (first < 0) first = i * 0 + j, std::printf("  %s: row %d pos %d: %04x vs %04x\n", name.c_str(), i, j,
                                                      g[(size_t)i * stride + j], r[(size_t)i * stride + j]);
        ++bad;
      }
  if (bad) ++g_failures;
  std::printf("%-44s %ld of %ld keys differ from attn::index_score %s\n", name.c_str(), bad, (long)rows * n,
              bad ? "FAIL" : "ok");
}

// R rows ending at ctx on random queries and keys: keys (and block maxima) bit-identical to attn's dense
// scan, the batched selection equal to attn::index_topk's
void test_index_synth(int R, int ctx, int ratio, bool pool, int batch) {
  const int n = ctx / ratio, cap = n + 64, stride = (n + 255) / 256 * 256, K = 512, KB = 2048;
  std::vector<int> ph(R);
  for (int r = 0; r < R; ++r) ph[r] = ctx - R + r;
  Dev<int> pos(ph.data(), R);
  Dev<uint8_t> qq((size_t)R * 32 * 64), kk((size_t)cap * 64), ksc((size_t)cap * 4), qsc8((size_t)R * 32 * 4);
  Dev<float> qw((size_t)R * 32);
  rnd_u8(qq, 0xFF, 0);
  rnd_u8(kk, 0xFF, 0);
  rnd_u8(ksc, 7, 120);
  rnd_u8(qsc8, 7, 122);
  fill_w<<<512, 256>>>(qw.p, qw.n, g_seed++);
  attn::IndexQ q{qq.p, reinterpret_cast<uint32_t*>(qsc8.p), qw.p};
  attn::IndexKeys ik{kk.p, ksc.p, cap};
  Dev<uint16_t> kref((size_t)R * stride), kgot((size_t)R * stride), bref((size_t)R * stride / 8),
      bgot((size_t)R * stride / 8);
  attn::index_score(q, pos.p, R, ratio, ik, attn::IndexScan{}, n, kref.p, stride, pool ? bref.p : nullptr, stride / 8, 0);
  dsk::index_score_p(q, pos.p, R, ratio, ik, n, kgot.p, stride, pool ? bgot.p : nullptr, stride / 8, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  char nm[96];
  std::snprintf(nm, sizeof nm, "index synth R=%d ctx=%d ratio=%d%s", R, ctx, ratio, pool ? " pool" : "");
  report_keys(std::string(nm) + " keys", kgot, kref, R, stride, n);
  if (pool) report_keys(std::string(nm) + " bkeys", bgot, bref, R, stride / 8, (n + 7) / 8);
  Dev<int> tref((size_t)R * K), tgot((size_t)R * K), pref((size_t)R * KB), pgot((size_t)R * KB);
  Dev<uint16_t> kyr((size_t)R * K), kyg((size_t)R * K), pkr((size_t)R * KB), pkg((size_t)R * KB);
  attn::index_topk(kref.p, stride, pos.p, R, ratio, attn::IndexScan{}, n, false, K, tref.p, kyr.p, K, 0);
  if (pool) attn::index_topk(bref.p, stride / 8, pos.p, R, ratio, attn::IndexScan{}, n, true, KB, pref.p, pkr.p, KB, 0);
  Dev<uint16_t> scratch((size_t)batch * stride), bscratch((size_t)batch * stride / 8);
  dsk::index_topk_p(q, pos.p, R, ratio, ik, n, batch, scratch.p, stride, K, tgot.p, kyg.p, pool ? bscratch.p : nullptr,
                    KB, pgot.p, pkg.p, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  int bad;
  report_rows(std::string(nm) + " topk", rows_equal(tgot.host(), tref.host(), R, K, &bad), R);
  if (pool) report_rows(std::string(nm) + " pool", rows_equal(pgot.host(), pref.host(), R, KB, &bad), R);
}

// The prefill call of the case's index-source layers: keys against attn's, top-k against model.py's
// (and layer 20's pool against its candidate mask)
void test_index_case(const std::string& dir) {
  Case cs = load_case(dir);
  const int rows = cs.T, K = cs.index_topk, KB = cs.cand_blocks;
  std::vector<int> ph(rows);
  for (int i = 0; i < rows; ++i) ph[i] = i;
  Dev<int> pos(ph.data(), rows);
  std::map<int, CompState> comp;
  for (int L : cs.layers)
    if (L == kv_source(L) && cs.has(W(L, "w_c_wkv"))) comp[L].build(cs, L, pos.p, rows);
  const std::string cname = dir.substr(dir.find_last_of('/') + 1);
  for (int L : cs.layers) {
    if (L >= 24 || index_source(L) != L || !comp.count(kv_source(L)) || !cs.has(P(L, 0, "iq_pre"))) continue;
    CompState& src = comp.at(kv_source(L));
    const int ratio = src.ratio, maxc = (cs.T + cs.D) / ratio + 1, n = rows / ratio;
    const int stride = (maxc + 255) / 256 * 256;
    auto rwp = cs.get(P(L, 0, "iw_proj"));
    std::vector<float> wph(rwp.numel());
    for (size_t i = 0; i < wph.size(); ++i) wph[i] = bf2f(rwp.as<uint16_t>()[i]);
    Dev<float> wp(wph.data(), wph.size());
    auto iq_pre = to_dev<uint16_t>(cs.get(P(L, 0, "iq_pre")));
    Dev<uint8_t> qq((size_t)rows * 32 * 64);
    Dev<uint32_t> qsc((size_t)rows * 32);
    Dev<float> qw((size_t)rows * 32);
    attn::IndexQ q{qq.p, qsc.p, qw.p};
    attn::index_q_prep(iq_pre.p, wp.p, pos.p, src.table.p, q, nullptr, rows, 0);
    const bool pool = L == 20;
    attn::IndexKeys ik{src.ik_k.p, src.ik_sc.p, src.cap};
    Dev<uint16_t> kref((size_t)rows * stride), kgot((size_t)rows * stride), bref((size_t)rows * stride / 8),
        bgot((size_t)rows * stride / 8);
    attn::index_score(q, pos.p, rows, ratio, ik, attn::IndexScan{}, maxc, kref.p, stride, pool ? bref.p : nullptr,
                      stride / 8, 0);
    dsk::index_score_p(q, pos.p, rows, ratio, ik, maxc, kgot.p, stride, pool ? bgot.p : nullptr, stride / 8, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::string nm = cname + " " + P(L, 0, "index");
    report_keys(nm + " keys", kgot, kref, rows, stride, n);
    if (pool) report_keys(nm + " bkeys", bgot, bref, rows, stride / 8, (n + 7) / 8);
    Dev<int> top((size_t)rows * K), pl((size_t)rows * KB);
    Dev<uint16_t> topkey((size_t)rows * K), plkey((size_t)rows * KB);
    Dev<uint16_t> scratch((size_t)64 * stride), bscratch((size_t)64 * stride / 8);
    dsk::index_topk_p(q, pos.p, rows, ratio, ik, maxc, 64, scratch.p, stride, K, top.p, topkey.p,
                      pool ? bscratch.p : nullptr, KB, pl.p, plkey.p, 0);
    auto ref = cs.get(P(L, 0, "itopk"));
    auto ht = top.host();
    int kref_w = (int)ref.shape[1], ok = 0;
    for (int r = 0; r < rows; ++r) {
      bool eq = true;
      for (int i = 0; i < K; ++i) eq &= ht[(size_t)r * K + i] == (i < kref_w ? ref.as<int>()[(size_t)r * kref_w + i] : -1);
      ok += eq;
    }
    report_rows(nm + " itopk vs model.py", ok, rows);
    if (pool) {
      auto hp = pl.host();
      auto cand = cs.get(P(L, 0, "icand"));
      int w = (int)cand.shape[1], okp = 0;
      for (int r = 0; r < rows; ++r) {
        std::vector<uint8_t> m(w, 0);
        for (int i = 0; i < KB; ++i) {
          int b = hp[(size_t)r * KB + i];
          for (int p = b * 8; b >= 0 && p < b * 8 + 8 && p < w; ++p) m[p] = 1;
        }
        okp += std::memcmp(m.data(), cand.as<uint8_t>() + (size_t)r * w, w) == 0;
      }
      report_rows(nm + " icand vs model.py", okp, rows);
    }
  }
}

// ---- compressor --------------------------------------------------------------------------------------
std::vector<uint16_t> host_bf16(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = h_f2bf(u(rng));
  return v;
}
// entries of width w (bytes) that differ between two caches
int entries_differ(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int n, int w, int* first) {
  int bad = 0;
  *first = -1;
  for (int e = 0; e < n; ++e)
    if (std::memcmp(a.data() + (size_t)e * w, b.data() + (size_t)e * w, w)) {
      if (*first < 0) *first = e;
      ++bad;
    }
  return bad;
}
void report_entries(const std::string& name, int bad, int n, int per_mille) {
  const bool ok = (int64_t)bad * 1000 <= (int64_t)n * per_mille;
  if (!ok) ++g_failures;
  std::printf("%-44s entries differing from attn's %d/%d (allowed %d per mille) %s\n", name.c_str(), bad, n, per_mille,
              ok ? "ok" : "FAIL");
}
struct CompCaches {
  int cap;
  Dev<uint8_t> kv, sc, ik, isc;
  Dev<float> ckv, csc;
  explicit CompCaches(int c) : cap(c), kv((size_t)c * 256), sc((size_t)c * 32), ik((size_t)c * 64), isc((size_t)c * 4),
                               ckv(16 * 512), csc(16 * 512) {}
  attn::Compressed ckvs() const { return {kv.p, sc.p, cap}; }
  attn::IndexKeys iks() const { return {ik.p, isc.p, cap}; }
  attn::CompressorCarry carry() const { return {ckv.p, csc.p}; }
};
// compressor_emit_p against attn::compressor_emit on the same projections and carry: KV entries and the
// carry bit-identical, index keys equal but for rare entries (wk's summation order)
void compare_emit(const std::string& nm, const float* kv, const float* score, const int* pos, int rows,
                  const attn::CompressorWeights& w, const float2* table, int cap, const std::vector<float>& carry_kv,
                  const std::vector<float>& carry_sc) {
  CompCaches A(cap), D(cap);
  for (CompCaches* c : {&A, &D}) {
    CUDA_CHECK(cudaMemcpy(c->ckv.p, carry_kv.data(), carry_kv.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c->csc.p, carry_sc.data(), carry_sc.size() * 4, cudaMemcpyHostToDevice));
  }
  attn::compressor_emit(kv, score, pos, rows, w, A.carry(), table, A.ckvs(), A.iks(), nullptr, 0, 0);
  dsk::compressor_emit_p(kv, score, pos, rows, w, D.carry(), table, D.ckvs(), D.iks(), 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  int f1, f2, f3, f4;
  const int bkv = entries_differ(A.kv.host(), D.kv.host(), cap, 256, &f1) + entries_differ(A.sc.host(), D.sc.host(), cap, 32, &f2);
  report_entries(nm + " ckv entries", bkv, cap, 0);
  const int bik = entries_differ(A.ik.host(), D.ik.host(), cap, 64, &f3) + entries_differ(A.isc.host(), D.isc.host(), cap, 4, &f4);
  report_entries(nm + " index keys", bik, cap, 10);
  if (w.ratio > 1) {
    const bool ok = A.ckv.host() == D.ckv.host() && A.csc.host() == D.csc.host();
    if (!ok) ++g_failures;
    std::printf("%-44s carry %s\n", (nm + " carry").c_str(), ok ? "equal ok" : "differs FAIL");
  }
}
// rows consecutive positions from p0 on random weights: projections against attn::compressor_project, the
// entries from attn's projections against attn::compressor_emit
void test_comp_synth(int rows, int ratio, int p0) {
  const int maxpos = p0 + rows + 1, cap = maxpos / ratio + 1;
  char nm[64];
  std::snprintf(nm, sizeof nm, "comp synth rows=%d ratio=%d p0=%d", rows, ratio, p0);
  Dev<float2> table((size_t)maxpos * 32);
  attn::rope_table_build(table.p, maxpos, true, 0);
  Dev<uint16_t> x((size_t)rows * 5120), wkv(512 * 5120), wgate(512 * 5120), wk(128 * 512);
  rnd_bf16(x, 2.0f);
  rnd_bf16(wkv, 0.03f);
  rnd_bf16(wgate, 0.05f);
  rnd_bf16(wk, 0.08f);
  auto hn = host_bf16(512, 11 + rows, 0.5f, 1.5f), hk = host_bf16(128, 12 + rows, 0.5f, 1.5f);
  Dev<uint16_t> norm(hn.data(), hn.size()), knorm(hk.data(), hk.size());
  attn::CompressorWeights w{ratio, wkv.p, ratio > 1 ? wgate.p : nullptr, norm.p, wk.p, knorm.p};
  std::vector<int> ph(rows);
  for (int r = 0; r < rows; ++r) ph[r] = p0 + r;
  Dev<int> pos(ph.data(), rows);
  Dev<float> kva((size_t)rows * 512), sca((size_t)rows * 512), kvd((size_t)rows * 512), scd((size_t)rows * 512);
  attn::compressor_project(x.p, w, kva.p, sca.p, rows, 0);
  dsk::compressor_project_p(x.p, w, kvd.p, scd.p, rows, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  if (ratio > 1) {
    report(std::string(nm) + " kv proj", cmp_f32(kvd.host().data(), kva.host().data(), kva.n, 1e-5, 1e-5));
    report(std::string(nm) + " score proj", cmp_f32(scd.host().data(), sca.host().data(), sca.n, 1e-5, 1e-5));
  } else {
    report(std::string(nm) + " kv proj (bf16)", cmp_f32(kvd.host().data(), kva.host().data(), kva.n, 1e-2, 1e-4));
  }
  std::vector<float> ck(16 * 512), cs(16 * 512);
  std::mt19937 rng(5 + rows);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : ck) v = nd(rng);
  for (auto& v : cs) v = nd(rng);
  compare_emit(nm, kva.p, sca.p, pos.p, rows, w, table.p, cap, ck, cs);
}
// the prefill call of a case's kv-source layers: projections against model.py's, the entries (from
// model.py's projections) against attn::compressor_emit's, which equal model.py's
void test_comp_case(const std::string& dir) {
  Case cs = load_case(dir);
  const std::string cname = dir.substr(dir.find_last_of('/') + 1);
  const int rows = cs.T;
  std::vector<int> ph(rows);
  for (int i = 0; i < rows; ++i) ph[i] = i;
  Dev<int> pos(ph.data(), rows);
  for (int L : cs.layers) {
    if (L != kv_source(L) || !cs.has(W(L, "w_c_wkv")) || !cs.has(P(L, 0, "x"))) continue;
    const int ratio = cs.ratio(L), maxpos = cs.T + cs.D, cap = maxpos / ratio + 1;
    Dev<float2> table((size_t)maxpos * 32);
    attn::rope_table_build(table.p, maxpos, true, 0);
    auto wkv = to_dev<uint16_t>(cs.get(W(L, "w_c_wkv")));
    auto wgate = ratio > 1 ? to_dev<uint16_t>(cs.get(W(L, "w_c_wgate"))) : Dev<uint16_t>();
    auto norm = to_dev<uint16_t>(cs.get(W(L, "w_c_norm")));
    auto wk = to_dev<uint16_t>(cs.get(W(L, "w_i_wk")));
    auto knorm = to_dev<uint16_t>(cs.get(W(L, "w_i_knorm")));
    attn::CompressorWeights w{ratio, wkv.p, wgate.p, norm.p, wk.p, knorm.p};
    auto x = to_dev<uint16_t>(cs.get(P(L, 0, "x")));
    Dev<float> kv((size_t)rows * 512), score((size_t)rows * 512);
    dsk::compressor_project_p(x.p, w, kv.p, score.p, rows, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::string nm = cname + " " + P(L, 0, "comp");
    if (ratio > 1) {
      auto rk = cs.get(P(L, 0, "c_kv")), rs = cs.get(P(L, 0, "c_score"));
      report(nm + " kv proj vs model.py", cmp_f32(kv.host().data(), rk.as<float>(), rk.numel(), 1e-5, 1e-4));
      report(nm + " score proj vs model.py", cmp_f32(score.host().data(), rs.as<float>(), rs.numel(), 1e-5, 1e-4));
      CUDA_CHECK(cudaMemcpy(kv.p, rk.as<float>(), rk.data.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(score.p, rs.as<float>(), rs.data.size(), cudaMemcpyHostToDevice));
    } else {
      auto ref = cs.get(P(L, 0, "c_proj"));
      auto got = kv.host();
      std::vector<uint16_t> gb(got.size());
      for (size_t i = 0; i < got.size(); ++i) gb[i] = h_f2bf(got[i]);
      report(nm + " proj vs model.py", cmp_bf16(gb.data(), ref.as<uint16_t>(), ref.numel()));
      std::vector<float> f(ref.numel());
      for (size_t i = 0; i < f.size(); ++i) f[i] = bf2f(ref.as<uint16_t>()[i]);
      CUDA_CHECK(cudaMemcpy(kv.p, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    compare_emit(nm, kv.p, score.p, pos.p, rows, w, table.p, cap, std::vector<float>(16 * 512, 0.f),
                 std::vector<float>(16 * 512, 0.f));
  }
}

// ---- mHC ---------------------------------------------------------------------------------------------
void report_u8(const std::string& name, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
  if (bad) ++g_failures;
  std::printf("%-44s %zu of %zu bytes differ %s\n", name.c_str(), bad, a.size(), bad ? "FAIL" : "ok");
}
// One boundary on (res, sub, post, comb, pre) against attn's kernels: y, x and act_quant bit-identical,
// the next coefficients from attn::hc_partial + hc_finish within the projection's summation order.
void check_boundary_vs_attn(const std::string& nm, int T, const Dev<uint16_t>& res, const Dev<uint16_t>& sub,
                            const Dev<float>& pre, const Dev<float>& post, const Dev<float>& comb, const Dev<float>& fn,
                            const Dev<float>& scale, const Dev<float>& base, const Dev<uint16_t>& nw) {
  Dev<uint32_t> parts(dsk::hc_fn_parts_bytes() / 4);
  dsk::hc_fn_parts(fn.p, parts.p, 0);
  Dev<uint16_t> y((size_t)T * attn::kHcDim), x((size_t)T * attn::kDim);
  Dev<uint8_t> xq((size_t)T * attn::kDim), xs((size_t)T * attn::kDim / 32);
  Dev<float> po(T * 4), qo(T * 4), co(T * 16);
  dsk::HcBoundary a;
  a.sub = sub.p, a.res = res.p, a.pre = pre.p, a.post = post.p, a.comb = comb.p, a.fn_parts = parts.p;
  a.scale = scale.p, a.base = base.p, a.norm_w = nw.p, a.y = y.p, a.x_out = x.p, a.xq = xq.p, a.xs = xs.p;
  a.pre_out = po.p, a.post_out = qo.p, a.comb_out = co.p;
  dsk::hc_boundary_p(a, T, 0);
  Dev<uint16_t> yr((size_t)T * attn::kHcDim), xr((size_t)T * attn::kDim);
  Dev<uint8_t> xqr((size_t)T * attn::kDim), xsr((size_t)T * attn::kDim / 32);
  Dev<float> chunks((size_t)attn::hc_chunks(0, attn::kHcDim) * T * 25), part(T * 25), pr(T * 4), qr(T * 4), cr(T * 16);
  attn::hc_post(sub.p, res.p, post.p, comb.p, T, yr.p, 0);
  attn::hc_pre(yr.p, pre.p, T, nw.p, xr.p, nullptr, 0);
  dsk::act_quant(reinterpret_cast<const __nv_bfloat16*>(xr.p), T, attn::kDim, xqr.p, xsr.p, 0);
  attn::hc_partial(yr.p, T, fn.p, 0, attn::kHcDim, chunks.p, part.p, 0);
  attn::hc_finish(part.p, T, scale.p, base.p, pr.p, qr.p, cr.p, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  report(nm + " y = hc_post", cmp_bf16(y.host().data(), yr.host().data(), yr.n), true);
  report(nm + " x = norm(hc_pre)", cmp_bf16(x.host().data(), xr.host().data(), xr.n), true);
  report_u8(nm + " act_quant e4m3", xq.host(), xqr.host());
  report_u8(nm + " act_quant ue8m0", xs.host(), xsr.host());
  report(nm + " pre vs attn", cmp_f32(po.host().data(), pr.host().data(), pr.n, 1e-5));
  report(nm + " post vs attn", cmp_f32(qo.host().data(), qr.host().data(), qr.n, 1e-5));
  report(nm + " comb vs attn", cmp_f32(co.host().data(), cr.host().data(), cr.n, 1e-4));
  // without y and act_quant (layer 20's projection before the replay): the same x and mixes
  Dev<uint16_t> x2((size_t)T * attn::kDim);
  Dev<float> po2(T * 4), qo2(T * 4), co2(T * 16);
  a.y = nullptr, a.xq = a.xs = nullptr, a.x_out = x2.p, a.pre_out = po2.p, a.post_out = qo2.p, a.comb_out = co2.p;
  dsk::hc_boundary_p(a, T, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  report(nm + " without y: x", cmp_bf16(x2.host().data(), x.host().data(), x.n), true);
  report(nm + " without y: pre", cmp_f32(po2.host().data(), po.host().data(), po.n, 0), true);
  report(nm + " without y: comb", cmp_f32(co2.host().data(), co.host().data(), co.n, 0), true);
}
// gen_mhc.py's case: on the reference's stream x (no hc_post) the projection, the Sinkhorn and hc_pre
// against model.py's; with its sublayer output, hc_post against model.py's and the rest against attn's
void test_mhc_case(const std::string& dir) {
  std::ifstream f(dir + "/manifest.json");
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string json = ss.str();
  const auto layers = json_ints(json, "layers");
  const int T = json_int(json, "T");
  auto get = [&](const std::string& n) { return npy_load(dir + "/" + n + ".npy"); };
  for (int L : layers)
    for (const char* which : {"attn", "ffn"}) {
      const std::string p = "L" + std::to_string(L) + "_" + which + "_", nm = "mhc " + p;
      auto x = to_dev<uint16_t>(get(p + "x"));
      auto fn = to_dev<float>(get(p + "fn"));
      auto scale = to_dev<float>(get(p + "scale")), base = to_dev<float>(get(p + "base"));
      auto rpre = get(p + "pre"), rpost = get(p + "post"), rcomb = get(p + "comb"), rraw = get(p + "raw");
      auto rrs = get(p + "rsqrt"), ryp = get(p + "y_pre"), rypo = get(p + "y_post");
      auto pre = to_dev<float>(rpre), post = to_dev<float>(rpost), comb = to_dev<float>(rcomb);
      Dev<uint32_t> parts(dsk::hc_fn_parts_bytes() / 4);
      dsk::hc_fn_parts(fn.p, parts.p, 0);
      Dev<uint16_t> xo((size_t)T * attn::kDim);
      Dev<float> po(T * 4), qo(T * 4), co(T * 16), proj(T * 25);
      dsk::HcBoundary a;
      a.res = x.p, a.pre = pre.p, a.fn_parts = parts.p, a.scale = scale.p, a.base = base.p, a.x_out = xo.p;
      a.pre_out = po.p, a.post_out = qo.p, a.comb_out = co.p, a.proj_out = proj.p;
      dsk::hc_boundary_p(a, T, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      report(nm + "hc_pre vs model.py", cmp_bf16(xo.host().data(), ryp.as<uint16_t>(), ryp.numel()), true);
      auto hp = proj.host();
      std::vector<float> raw((size_t)T * 24), rs(T);
      for (int r = 0; r < T; ++r) {
        for (int i = 0; i < 24; ++i) raw[(size_t)r * 24 + i] = hp[(size_t)r * 25 + i];
        rs[r] = 1.0f / std::sqrt(hp[(size_t)r * 25 + 24] / attn::kHcDim);
      }
      report(nm + "projection vs model.py", cmp_f32(raw.data(), rraw.as<float>(), rraw.numel(), 1e-4, 1e-4));
      report(nm + "rsqrt vs model.py", cmp_f32(rs.data(), rrs.as<float>(), rrs.numel(), 1e-5));
      report(nm + "pre vs model.py", cmp_f32(po.host().data(), rpre.as<float>(), rpre.numel(), 1e-5));
      report(nm + "post vs model.py", cmp_f32(qo.host().data(), rpost.as<float>(), rpost.numel(), 1e-5));
      report(nm + "comb vs model.py", cmp_f32(co.host().data(), rcomb.as<float>(), rcomb.numel(), 1e-4));
      // hc_post into the stream
      auto sub = to_dev<uint16_t>(get(p + "sub"));
      Dev<uint16_t> y((size_t)T * attn::kHcDim);
      a.sub = sub.p, a.post = post.p, a.comb = comb.p, a.y = y.p, a.proj_out = nullptr;
      dsk::hc_boundary_p(a, T, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      report(nm + "hc_post vs model.py", cmp_bf16(y.host().data(), rypo.as<uint16_t>(), rypo.numel()), true);
      auto hn = host_bf16(attn::kDim, 3 + L, 0.5f, 1.5f);
      Dev<uint16_t> nw(hn.data(), hn.size());
      check_boundary_vs_attn(nm + "boundary", T, x, sub, pre, post, comb, fn, scale, base, nw);
    }
}
// many rows of a random stream (per-copy scales, outlier channels) through one boundary against attn's
// kernels, with the case's real hc weights when there are some
void test_mhc_synth(int T, const std::string& mhc_dir) {
  std::vector<float> hfn(24 * attn::kHcDim), hsc = {0.3f, 0.2f, 0.4f}, hb(24);
  std::mt19937 rng(77);
  std::normal_distribution<float> nd(0.f, 1.f);
  if (npy_exists(mhc_dir + "/L2_ffn_fn.npy")) {
    auto a = npy_load(mhc_dir + "/L2_ffn_fn.npy"), b = npy_load(mhc_dir + "/L2_ffn_scale.npy"),
         c = npy_load(mhc_dir + "/L2_ffn_base.npy");
    std::memcpy(hfn.data(), a.as<float>(), hfn.size() * 4);
    std::memcpy(hsc.data(), b.as<float>(), 12);
    std::memcpy(hb.data(), c.as<float>(), 96);
  } else {
    for (auto& v : hfn) v = nd(rng) * 0.01f;
    for (auto& v : hb) v = nd(rng);
  }
  Dev<float> fn(hfn.data(), hfn.size()), scale(hsc.data(), 3), base(hb.data(), 24);
  std::vector<uint16_t> hx((size_t)T * attn::kHcDim), hs((size_t)T * attn::kDim);
  const float cs[4] = {0.5f, 1.f, 2.f, 4.f};
  for (int r = 0; r < T; ++r)
    for (int c = 0; c < 4; ++c)
      for (int d = 0; d < attn::kDim; ++d)
        hx[((size_t)r * 4 + c) * attn::kDim + d] = h_f2bf(nd(rng) * cs[c] * (d % 640 == 17 ? 20.f : 1.f));
  for (auto& v : hs) v = h_f2bf(nd(rng) * 0.7f);
  Dev<uint16_t> x(hx.data(), hx.size()), sub(hs.data(), hs.size());
  // coefficients as a previous boundary makes them: a Sinkhorn of random mixes
  std::vector<float> hpart((size_t)T * 25);
  for (int r = 0; r < T; ++r) {
    for (int i = 0; i < 24; ++i) hpart[(size_t)r * 25 + i] = nd(rng) * 3.f;
    hpart[(size_t)r * 25 + 24] = (float)attn::kHcDim;
  }
  Dev<float> part(hpart.data(), hpart.size()), pre(T * 4), post(T * 4), comb(T * 16);
  attn::hc_finish(part.p, T, scale.p, base.p, pre.p, post.p, comb.p, 0);
  auto hn = host_bf16(attn::kDim, 9, 0.5f, 1.5f);
  Dev<uint16_t> nw(hn.data(), hn.size());
  check_boundary_vs_attn("mhc synth T=" + std::to_string(T), T, x, sub, pre, post, comb, fn, scale, base, nw);
}

// hc_boundary_p's rows per CTA (4, 8, 16, 32), L2 prefetch and the split kernel (up to 512 rows) against its
// default (32 rows, none), bit for
// bit: every output of four option sets, on chunks of one row to many CTAs (partial CTAs included)
void test_mhc_rows() {
  constexpr int kTmax = 2048;
  std::mt19937 rng(91);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hfn(24 * attn::kHcDim), hsc = {0.3f, 0.2f, 0.4f}, hb(24), hpart((size_t)kTmax * 25);
  for (auto& v : hfn) v = nd(rng) * 0.01f;
  for (auto& v : hb) v = nd(rng);
  for (int r = 0; r < kTmax; ++r) {
    for (int i = 0; i < 24; ++i) hpart[(size_t)r * 25 + i] = nd(rng) * 3.f;
    hpart[(size_t)r * 25 + 24] = (float)attn::kHcDim;
  }
  Dev<float> fn(hfn.data(), hfn.size()), scale(hsc.data(), 3), base(hb.data(), 24), part(hpart.data(), hpart.size());
  Dev<uint32_t> parts(dsk::hc_fn_parts_bytes() / 4);
  dsk::hc_fn_parts(fn.p, parts.p, 0);
  Dev<uint16_t> res((size_t)kTmax * attn::kHcDim), sub((size_t)kTmax * attn::kDim);
  rnd_bf16(res, 2.0f);
  rnd_bf16(sub, 0.7f);
  Dev<float> pre(kTmax * 4), post(kTmax * 4), comb(kTmax * 16);
  attn::hc_finish(part.p, kTmax, scale.p, base.p, pre.p, post.p, comb.p, 0);
  auto hn = host_bf16(attn::kDim, 9, 0.5f, 1.5f);
  Dev<uint16_t> nw(hn.data(), hn.size()), y((size_t)kTmax * attn::kHcDim), x((size_t)kTmax * attn::kDim);
  Dev<uint8_t> xq((size_t)kTmax * attn::kDim), xs((size_t)kTmax * attn::kDim / 32);
  Dev<float> po(kTmax * 4), qo(kTmax * 4), co(kTmax * 16), proj(kTmax * 25);
  struct Out {
    void* p;
    size_t row_bytes;
  };
  const Out outs[] = {{y.p, attn::kHcDim * 2}, {x.p, attn::kDim * 2}, {xq.p, attn::kDim}, {xs.p, attn::kDim / 32},
                      {po.p, 16},              {qo.p, 16},             {co.p, 64},        {proj.p, 100}};
  // every output buffer's first T rows (bytes), poisoned before each run so that a row left unwritten shows
  auto run = [&](const dsk::HcBoundary& a, int T, int R, int pf) {
    for (const Out& o : outs) CUDA_CHECK(cudaMemset(o.p, 0xA5, o.row_bytes * kTmax));
    dsk::hc_boundary_tuning(R, pf);
    dsk::hc_boundary_p(a, T, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> h;
    for (const Out& o : outs) {
      const size_t n = o.row_bytes * T, at = h.size();
      h.resize(at + n);
      CUDA_CHECK(cudaMemcpy(h.data() + at, o.p, n, cudaMemcpyDeviceToHost));
    }
    return h;
  };
  for (int set = 0; set < 4; ++set) {
    dsk::HcBoundary a;
    a.res = res.p, a.pre = pre.p, a.x_out = x.p, a.norm_w = nw.p;
    a.fn_parts = parts.p, a.scale = scale.p, a.base = base.p, a.pre_out = po.p, a.post_out = qo.p, a.comb_out = co.p;
    const char* name = "";
    if (set == 0) {  // a boundary: hc_post, the stream, act_quant, the projection out too
      name = "post, y, act_quant";
      a.sub = sub.p, a.post = post.p, a.comb = comb.p, a.y = y.p, a.xq = xq.p, a.xs = xs.p, a.proj_out = proj.p;
    } else if (set == 1) {  // the first layer's hc_pre: no sublayer, x not normalised
      name = "pre only, no norm";
      a.norm_w = nullptr, a.proj_out = proj.p;
    } else if (set == 2) {  // CED: layer 20's projection of the rows before the replay
      name = "post, no y";
      a.sub = sub.p, a.post = post.p, a.comb = comb.p;
    } else {  // no projection: the last boundary
      name = "post, no projection";
      a.sub = sub.p, a.post = post.p, a.comb = comb.p, a.y = y.p, a.xq = xq.p, a.xs = xs.p;
      a.fn_parts = nullptr, a.scale = a.base = nullptr, a.pre_out = a.post_out = a.comb_out = nullptr;
    }
    for (int T : {1, 5, 16, 31, 32, 33, 56, 88, 128, 184, 344, 500, 512, kTmax}) {
      const std::vector<uint8_t> ref = run(a, T, 32, 0);
      std::string bad;
      for (int R : {4, 8, 16, 32})
        for (int pf : {0, 3}) {
          if (R == 32 && pf == 0) continue;
          if (run(a, T, R, pf) != ref) bad += " R" + std::to_string(R) + "/pf" + std::to_string(pf);
        }
      if (T <= 512)  // the split kernel (hc_boundary_split), rows per CTA 4, 8, 16 and by the rows
        for (int R : {4, 8, 16, 0}) {
          dsk::hc_boundary_split(512, R);
          if (run(a, T, 32, 0) != ref) bad += " split" + std::to_string(R);
          dsk::hc_boundary_split(0);
        }
      if (!bad.empty()) ++g_failures;
      std::printf("mhc rows per CTA, %-20s T=%4d: R 4..32, prefetch 0/3, split: %s\n", name, T,
                  bad.empty() ? "same bits ok" : ("DIFFER" + bad + " FAIL").c_str());
    }
  }
  dsk::hc_boundary_tuning(0, -1);
}

}  // namespace

int main(int argc, char** argv) {
  g_root = argc > 1 && argv[1][0] ? argv[1] : std::string(getenv("HOME")) + "/dsgb10-data/attntests";
  std::string only = argc > 2 ? argv[2] : "";
  auto want = [&](const char* s) { return only.empty() || only == s; };
  if (want("attn")) {
    test_attn_synth(300, 4096, 512, 112);
    test_attn_synth(1024, 65536, 512, 116);
    test_attn_synth(700, 20000, 0, 118);
    test_attn_synth(200, 200, 512, 112);  // the start of a sequence: short windows and lists
    test_attn_synth(128, 65536, 512, 114, 65536 - 128);  // CED's decoder replay: the window inside the chunk
    for (const char* c : {"l0", "l2", "l20"})
      if (npy_exists(g_root + "/" + c + "/manifest.json")) test_attn_case(g_root + "/" + c);
  }
  if (want("index")) {
    test_index_synth(300, 4096, 2, false, 64);
    test_index_synth(96, 2000, 1, true, 40);
    test_index_synth(1024, 131072, 2, false, 96);
    test_index_synth(128, 65536, 1, true, 128);
    for (const char* c : {"l2", "l20"})
      if (npy_exists(g_root + "/" + c + "/manifest.json")) test_index_case(g_root + "/" + c);
  }
  if (want("comp")) {
    test_comp_synth(300, 2, 0);
    test_comp_synth(1001, 2, 4097);  // odd start: row 0 completes the carry's group
    test_comp_synth(1024, 1, 12288);
    test_comp_synth(37, 2, 15);
    for (const char* c : {"l2", "l20"})
      if (npy_exists(g_root + "/" + c + "/manifest.json")) test_comp_case(g_root + "/" + c);
  }
  if (want("mhc")) {
    if (npy_exists(g_root + "/mhc/manifest.json")) test_mhc_case(g_root + "/mhc");
    test_mhc_synth(1000, g_root + "/mhc");
    test_mhc_synth(37, g_root + "/mhc");
  }
  if (want("mhc") || want("mhcrows")) test_mhc_rows();
  std::printf("%s (%d failing checks)\n", g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
