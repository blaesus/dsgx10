// Checks the attention-side kernels against reference data from gen_attn.py.
// usage: attn_tests <case dir>   (e.g. ~/dsgb10-data/attntests/l2)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "tests/testutil.h"

using namespace attn_test;

namespace {

struct Case {
  std::string dir;
  std::vector<int> layers, ratios;
  int T = 0, D = 0, index_topk = 0, cand_blocks = 0;
  bool state_prefill = false;  // call 0 recorded only what the caches are rebuilt from (gen_attn --state-prefill)
  std::string json;
  bool has(const std::string& name) const { return npy_exists(dir + "/" + name + ".npy"); }
  Npy get(const std::string& name) const { return npy_load(dir + "/" + name + ".npy"); }
  int ratio(int L) const {
    for (size_t i = 0; i < layers.size(); ++i)
      if (layers[i] == L) return ratios[i];
    return 0;
  }
  int rows(int call) const { return call == 0 ? T : 1; }
  std::vector<int> pos(int call) const {
    std::vector<int> p;
    if (call == 0)
      for (int i = 0; i < T; ++i) p.push_back(i);
    else
      p.push_back(T + call - 1);
    return p;
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
  c.state_prefill = c.json.find("\"state_prefill\": 1") != std::string::npos;
  return c;
}

std::string P(int L, int call, const char* what) { return "L" + std::to_string(L) + "_c" + std::to_string(call) + "_" + what; }
std::string W(int L, const char* what) { return "L" + std::to_string(L) + "_" + what; }

void test_prep(const Case& cs, int L) {
  int maxpos = cs.T + cs.D;
  bool comp = cs.ratio(L) > 0;
  Dev<float2> table((size_t)maxpos * 32);
  attn::rope_table_build(table.p, maxpos, comp, 0);
  {
    auto ref = cs.get(W(L, "freqs"));
    auto got = table.host();
    report(W(L, "rope_table"), cmp_f32(reinterpret_cast<const float*>(got.data()), ref.as<float>(), ref.numel()), true);
  }
  auto qnw = to_dev<uint16_t>(cs.get(W(L, "w_q_norm")));
  auto kvw = to_dev<uint16_t>(cs.get(W(L, "w_kv_norm")));
  Dev<uint8_t> ring_kv(attn::kRing * 512), ring_sc(attn::kRing * 16);
  attn::Window win{ring_kv.p, ring_sc.p};
  for (int call = 0; call <= cs.D; ++call) {
    if (!cs.has(P(L, call, "qa"))) continue;  // a --state-prefill call
    int rows = cs.rows(call);
    auto pos_h = cs.pos(call);
    Dev<int> pos(pos_h.data(), pos_h.size());
    // q_norm
    {
      auto qa = to_dev<uint16_t>(cs.get(P(L, call, "qa")));
      Dev<uint16_t> qr((size_t)rows * attn::kQLora);
      attn::q_norm(qa.p, qnw.p, qr.p, rows, 0);
      auto ref = cs.get(P(L, call, "qr"));
      report(P(L, call, "q_norm"), cmp_bf16(qr.host().data(), ref.as<uint16_t>(), ref.numel()));
    }
    // q rope (64 heads in the reference)
    {
      auto q = to_dev<uint16_t>(cs.get(P(L, call, "q_pre")));
      attn::rope_rows(q.p, rows, 64, 512, pos.p, table.p, false, 0);
      auto ref = cs.get(P(L, call, "q"));
      report(P(L, call, "q_rope"), cmp_bf16(q.host().data(), ref.as<uint16_t>(), ref.numel()));
    }
    // o inverse rope
    {
      auto o = to_dev<uint16_t>(cs.get(P(L, call, "o_raw")));
      attn::rope_rows(o.p, rows, 64, 512, pos.p, table.p, true, 0);
      auto ref = cs.get(P(L, call, "o"));
      report(P(L, call, "o_inverse_rope"), cmp_bf16(o.host().data(), ref.as<uint16_t>(), ref.numel()));
    }
    // window KV
    {
      auto kvp = to_dev<uint16_t>(cs.get(P(L, call, "kv_pre")));
      Dev<uint16_t> kv((size_t)rows * 512);
      attn::kv_write(kvp.p, kvw.p, pos.p, table.p, win, kv.p, rows, 0);
      auto ref = cs.get(P(L, call, "kv"));
      report(P(L, call, "kv_write"), cmp_bf16(kv.host().data(), ref.as<uint16_t>(), ref.numel()));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

// host decoders for the packed caches
float host_e2m1(uint32_t nib) {
  static const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (nib & 8) ? -t[nib & 7] : t[nib & 7];
}
float host_e4m3(uint8_t b) {
  int e = (b >> 3) & 15, m = b & 7;
  float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
  return (b & 0x80) ? -v : v;
}
uint16_t host_f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
int host_fp4x128_byte(int cb) {  // attn_common.cuh fp4x128_byte
  int cw = cb >> 2;
  return (((cw & 3) << 2) | (cw >> 2)) * 4 + (cb & 3);
}
// entry j of a compressed-KV cache / index-key cache, dequantised to bf16
void dequant_ckv(const std::vector<uint8_t>& kv, const std::vector<uint8_t>& sc, int j, uint16_t* out) {
  for (int c = 0; c < 512; ++c) {
    uint8_t b = kv[(size_t)j * 256 + c / 2];
    out[c] = host_f2bf(host_e2m1(c & 1 ? b >> 4 : b & 15) * host_e4m3(sc[(size_t)j * 32 + c / 16]));
  }
}
void dequant_ik(const std::vector<uint8_t>& k, const std::vector<uint8_t>& sc, int j, uint16_t* out) {
  for (int c = 0; c < 128; ++c) {
    uint8_t b = k[(size_t)j * 64 + host_fp4x128_byte(c / 2)];
    out[c] = host_f2bf(host_e2m1(c & 1 ? b >> 4 : b & 15) * std::ldexp(1.0f, sc[(size_t)j * 4 + c / 32] - 127));
  }
}

// The compressor of a kv-source layer, call by call: projections, latents, and the cache entries
// (dequantised from the packed caches, so the layout is checked too).
void test_compressor(const Case& cs, int L) {
  if (!cs.has(W(L, "w_c_wkv"))) return;
  int ratio = cs.ratio(L), maxpos = cs.T + cs.D, cap = maxpos / ratio + 1;
  Dev<float2> table((size_t)maxpos * 32);
  attn::rope_table_build(table.p, maxpos, true, 0);
  auto wkv = to_dev<uint16_t>(cs.get(W(L, "w_c_wkv")));
  auto wgate = ratio > 1 ? to_dev<uint16_t>(cs.get(W(L, "w_c_wgate"))) : Dev<uint16_t>();
  auto norm = to_dev<uint16_t>(cs.get(W(L, "w_c_norm")));
  auto wk = to_dev<uint16_t>(cs.get(W(L, "w_i_wk")));
  auto knorm = to_dev<uint16_t>(cs.get(W(L, "w_i_knorm")));
  attn::CompressorWeights w{ratio, wkv.p, wgate.p, norm.p, wk.p, knorm.p};
  Dev<float> carry_kv(16 * 512), carry_score(16 * 512);
  Dev<uint8_t> ckv_kv((size_t)cap * 256), ckv_sc((size_t)cap * 32), ik_k((size_t)cap * 64), ik_sc((size_t)cap * 4);
  attn::Compressed ckv{ckv_kv.p, ckv_sc.p, cap};
  attn::IndexKeys ik{ik_k.p, ik_sc.p, cap};
  for (int call = 0; call <= cs.D; ++call) {
    int rows = cs.rows(call);
    auto pos_h = cs.pos(call);
    Dev<int> pos(pos_h.data(), pos_h.size());
    bool state = !cs.has(P(L, call, "x"));  // a --state-prefill call: only the projections are recorded
    Dev<float> kv((size_t)rows * 512), score((size_t)rows * 512);
    if (!state) {
      auto x = to_dev<uint16_t>(cs.get(P(L, call, "x")));
      attn::compressor_project(x.p, w, kv.p, score.p, rows, 0);
    }
    // The reference's GEMMs (cuBLAS) sum in another order, so the stages after the projection are
    // checked from the reference's own projections: emit then sees exactly the reference's input.
    if (ratio > 1) {
      auto rk = cs.get(P(L, call, "c_kv")), rs = cs.get(P(L, call, "c_score"));
      if (!state) {
        report(P(L, call, "c_kv"), cmp_f32(kv.host().data(), rk.as<float>(), rk.numel(), 1e-5, 1e-4));
        report(P(L, call, "c_score"), cmp_f32(score.host().data(), rs.as<float>(), rs.numel(), 1e-5, 1e-4));
      }
      CUDA_CHECK(cudaMemcpy(kv.p, rk.as<float>(), rk.data.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(score.p, rs.as<float>(), rs.data.size(), cudaMemcpyHostToDevice));
    } else {
      auto ref = cs.get(P(L, call, "c_proj"));
      std::vector<float> got(ref.numel());
      if (!state) {
        got = kv.host();
        std::vector<uint16_t> gb(got.size());
        for (size_t i = 0; i < got.size(); ++i) gb[i] = host_f2bf(got[i]);
        report(P(L, call, "c_proj"), cmp_bf16(gb.data(), ref.as<uint16_t>(), ref.numel()));
      }
      for (size_t i = 0; i < got.size(); ++i) got[i] = bf2f(ref.as<uint16_t>()[i]);
      CUDA_CHECK(cudaMemcpy(kv.p, got.data(), got.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    int first = -1, n = 0;  // the entries this call completes
    for (int p : pos_h)
      if ((p + 1) % ratio == 0) {
        if (first < 0) first = p / ratio;
        ++n;
      }
    Dev<uint16_t> lat((size_t)std::max(n, 1) * 512);
    attn::compressor_emit(kv.p, score.p, pos.p, rows, w, {carry_kv.p, carry_score.p}, table.p, ckv, ik, lat.p, first, 0);
    if (n == 0) continue;
    if (state) {  // a long batched prefill: the reference's RMSNorm and wk GEMM sum in a batch-dependent
                  // order (tests/probe_batch_order.py), so a few entries may differ; judged by count
      auto hk = ckv_kv.host(), hs = ckv_sc.host(), hik = ik_k.host(), his = ik_sc.host();
      std::vector<uint16_t> dk((size_t)n * 512), di((size_t)n * 128);
      for (int e = 0; e < n; ++e) {
        dequant_ckv(hk, hs, first + e, dk.data() + (size_t)e * 512);
        dequant_ik(hik, his, first + e, di.data() + (size_t)e * 128);
      }
      auto hl = lat.host();
      auto rl = cs.get(P(L, call, "latent")), rk = cs.get(W(L, "ckv_cache")), ri = cs.get(W(L, "ik_cache"));
      struct Arr {
        const char* name;
        const uint16_t *got, *ref;
        int w, per_mille;  // allowed fraction of differing entries
      } arrs[3] = {{"prefill_latent", hl.data(), rl.as<uint16_t>(), 512, 10},  // 1 ulp, the batch's RMSNorm order
                   {"ckv_cache", dk.data(), rk.as<uint16_t>(), 512, 1},
                   {"ik_cache", di.data(), ri.as<uint16_t>(), 128, 1}};
      for (const Arr& a : arrs) {
        int nd = 0, first_bad = -1;
        for (int e = 0; e < n; ++e)
          if (std::memcmp(a.got + (size_t)a.w * e, a.ref + (size_t)a.w * e, a.w * 2)) {
            if (first_bad < 0) first_bad = e;
            ++nd;
          }
        bool ok = (int64_t)nd * 1000 <= (int64_t)n * a.per_mille;
        if (!ok) ++g_failures;
        std::printf("%-44s entries differing %d/%d (first %d)  %s\n", W(L, a.name).c_str(), nd, n, first_bad,
                    ok ? "ok" : "FAIL");
      }
      continue;
    }
    auto rl = cs.get(P(L, call, "latent"));
    report(P(L, call, "latent"), cmp_bf16(lat.host().data(), rl.as<uint16_t>(), rl.numel()));
    auto hk = ckv_kv.host(), hs = ckv_sc.host(), hik = ik_k.host(), his = ik_sc.host();
    std::vector<uint16_t> dk((size_t)n * 512), di((size_t)n * 128);
    for (int e = 0; e < n; ++e) {
      dequant_ckv(hk, hs, first + e, dk.data() + (size_t)e * 512);
      dequant_ik(hik, his, first + e, di.data() + (size_t)e * 128);
    }
    auto rk = cs.get(P(L, call, "ckv")), ri = cs.get(P(L, call, "ik"));
    report(P(L, call, "ckv_entry"), cmp_bf16(dk.data(), rk.as<uint16_t>(), rk.numel()));
    report(P(L, call, "index_key"), cmp_bf16(di.data(), ri.as<uint16_t>(), ri.numel()));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ---- indexer ------------------------------------------------------------------------------------

int kv_source(int L) { return L < 2 ? -1 : L < 8 ? 2 : L < 14 ? 8 : L < 20 ? 14 : 20; }
int index_source(int L) { return L < 20 ? kv_source(L) : 20 + (L - 20) / 4 * 4; }
uint16_t host_bf16_from_key(uint16_t k) { return k == 0 ? 0xFF80 : (k & 0x8000) ? (k & 0x7FFF) : (uint16_t)~k; }

// A kv-source layer's compressor, driven by the reference's projections (so its caches are exact).
struct CompState {
  int L = -1, ratio = 0, cap = 0;
  Dev<float2> table;
  Dev<uint16_t> wkv, wgate, norm, wk, knorm;
  Dev<float> carry_kv, carry_score;
  Dev<uint8_t> ckv_kv, ckv_sc, ik_k, ik_sc;
  void init(const Case& cs, int layer) {
    L = layer;
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
  }
  attn::IndexKeys ik() const { return {ik_k.p, ik_sc.p, cap}; }
  attn::Compressed ckv() const { return {ckv_kv.p, ckv_sc.p, cap}; }
  void step(const Case& cs, int call, const int* pos, int rows) {
    Dev<float> kv((size_t)rows * 512), score((size_t)rows * 512);
    if (ratio > 1) {
      auto rk = cs.get(P(L, call, "c_kv")), rs = cs.get(P(L, call, "c_score"));
      CUDA_CHECK(cudaMemcpy(kv.p, rk.as<float>(), rk.data.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(score.p, rs.as<float>(), rs.data.size(), cudaMemcpyHostToDevice));
    } else {
      auto ref = cs.get(P(L, call, "c_proj"));
      std::vector<float> f(ref.numel());
      for (size_t i = 0; i < f.size(); ++i) f[i] = bf2f(ref.as<uint16_t>()[i]);
      CUDA_CHECK(cudaMemcpy(kv.p, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    attn::CompressorWeights w{ratio, wkv.p, wgate.p, norm.p, wk.p, knorm.p};
    attn::compressor_emit(kv.p, score.p, pos, rows, w, {carry_kv.p, carry_score.p}, table.p, ckv(), ik(), nullptr, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

// rows of `got` (width kgot) against the reference's (width kref): same entries, the rest -1
int topk_rows_equal(const std::vector<int>& got, int kgot, const Npy& ref, int rows, int* first_bad) {
  int kref = (int)ref.shape[1], ok = 0;
  *first_bad = -1;
  for (int r = 0; r < rows; ++r) {
    bool eq = true;
    for (int i = 0; i < kgot; ++i) eq &= got[(size_t)r * kgot + i] == (i < kref ? ref.as<int>()[(size_t)r * kref + i] : -1);
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

// The indexers of the case's index-source layers, call by call, with the compressor feeding their keys.
// Decode calls also run the 4-rank split (dense parts, local top-k, merge) and must match the unsplit result.
void test_indexer(const Case& cs) {
  std::map<int, CompState> comp;
  for (int L : cs.layers)
    if (L == kv_source(L) && cs.has(W(L, "w_c_wkv"))) comp[L].init(cs, L);
  const int K = cs.index_topk, KB = cs.cand_blocks;
  std::map<int, Dev<float2>> tables;
  for (int call = 0; call <= cs.D; ++call) {
    int rows = cs.rows(call);
    auto pos_h = cs.pos(call);
    Dev<int> pos(pos_h.data(), pos_h.size());
    Dev<int> pool;  // layer 20's candidate blocks for this call, [rows][KB]
    for (int L : cs.layers) {
      if (comp.count(L)) comp[L].step(cs, call, pos.p, rows);
      if (index_source(L) != L || !comp.count(kv_source(L))) continue;
      if (!cs.has(P(L, call, "iq_pre"))) continue;  // a --state-prefill call
      CompState& src = comp[kv_source(L)];
      int ratio = src.ratio, maxc = (cs.T + cs.D) / ratio + 1;
      int stride = (maxc + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
      // queries and head weights, from the reference's projections
      auto x = to_dev<uint16_t>(cs.get(P(L, call, "x")));
      auto wp_w = to_dev<uint16_t>(cs.get(W(L, "w_i_wproj")));
      Dev<float> wp((size_t)rows * 32);
      attn::gemv_bf16(x.p, rows, attn::kDim, wp_w.p, nullptr, 32, 32, wp.p, nullptr, true, 0);
      auto rwp = cs.get(P(L, call, "iw_proj"));
      {
        auto got = wp.host();
        std::vector<uint16_t> gb(got.size());
        for (size_t i = 0; i < got.size(); ++i) gb[i] = host_f2bf(got[i]);
        report(P(L, call, "iw_proj"), cmp_bf16(gb.data(), rwp.as<uint16_t>(), rwp.numel()));
        for (size_t i = 0; i < got.size(); ++i) got[i] = bf2f(rwp.as<uint16_t>()[i]);
        CUDA_CHECK(cudaMemcpy(wp.p, got.data(), got.size() * sizeof(float), cudaMemcpyHostToDevice));
      }
      auto iq_pre = to_dev<uint16_t>(cs.get(P(L, call, "iq_pre")));
      Dev<uint8_t> qq((size_t)rows * 32 * 64);
      Dev<uint32_t> qsc((size_t)rows * 32);
      Dev<float> qw((size_t)rows * 32);
      Dev<uint16_t> qdbg((size_t)rows * 32 * 128);
      attn::IndexQ q{qq.p, qsc.p, qw.p};
      attn::index_q_prep(iq_pre.p, wp.p, pos.p, src.table.p, q, qdbg.p, rows, 0);
      {
        auto ref = cs.get(P(L, call, "iq"));
        report(P(L, call, "iq"), cmp_bf16(qdbg.host().data(), ref.as<uint16_t>(), ref.numel()), true);
        auto riw = cs.get(P(L, call, "iw"));
        auto got = qw.host();
        std::vector<uint16_t> gb(got.size());
        for (size_t i = 0; i < got.size(); ++i) gb[i] = host_f2bf(got[i]);
        report(P(L, call, "iw"), cmp_bf16(gb.data(), riw.as<uint16_t>(), riw.numel()), true);
      }
      bool reindex = L >= 24;
      attn::IndexScan scan;
      if (reindex) {
        if (!pool.p) die("layer %d: no candidate pool", L);
        scan.blocks = pool.p;
        scan.nblocks = KB;
      }
      int max_scan = reindex ? KB * 8 : maxc;
      int kstride = reindex ? (KB * 8 + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign : stride;
      Dev<uint16_t> keys((size_t)rows * kstride), bkeys((size_t)rows * kstride / 8);
      bool is_pool = L == 20;
      attn::index_score(q, pos.p, rows, ratio, src.ik(), scan, max_scan, keys.p, kstride, is_pool ? bkeys.p : nullptr,
                        kstride / 8, 0);
      // scores against the reference (Reindex: its masked scores, ours expanded from the pool's blocks)
      auto rsc = cs.get(P(L, call, reindex ? "iscore_masked" : "iscore"));
      int width = (int)rsc.shape[1];
      {
        auto hk = keys.host();
        std::vector<uint16_t> dense((size_t)rows * width, 0xFF80);
        std::vector<int> hp = reindex ? pool.host() : std::vector<int>();
        for (int r = 0; r < rows; ++r)
          for (int i = 0; i < (reindex ? KB * 8 : width); ++i) {
            int p = i;
            if (reindex) {
              int b = hp[(size_t)r * KB + i / 8];
              p = b < 0 ? -1 : b * 8 + i % 8;
            }
            if (p >= 0 && p < width) dense[(size_t)r * width + p] = host_bf16_from_key(hk[(size_t)r * kstride + i]);
          }
        // after a --state-prefill a few keys differ by the reference's batch order: scores are then information
        report(P(L, call, reindex ? "iscore_masked" : "iscore"), cmp_bf16(dense.data(), rsc.as<uint16_t>(), rsc.numel()),
               false, !cs.state_prefill);
      }
      Dev<int> top((size_t)rows * K);
      Dev<uint16_t> topkey((size_t)rows * K);
      attn::index_topk(keys.p, kstride, pos.p, rows, ratio, scan, max_scan, false, K, top.p, topkey.p, K, 0);
      auto htop = top.host();
      int bad;
      report_rows(P(L, call, "itopk"), topk_rows_equal(htop, K, cs.get(P(L, call, "itopk")), rows, &bad), rows);
      if (bad >= 0) {
        auto ref = cs.get(P(L, call, "itopk"));
        std::printf("  row %d got:", bad);
        for (int i = 0; i < std::min(K, 12); ++i) std::printf(" %d", htop[(size_t)bad * K + i]);
        std::printf("\n  row %d ref:", bad);
        for (int i = 0; i < std::min((int)ref.shape[1], 12); ++i) std::printf(" %d", ref.as<int>()[(size_t)bad * ref.shape[1] + i]);
        std::printf("\n");
      }
      Dev<int> pool_new;
      if (is_pool) {
        pool_new = Dev<int>((size_t)rows * KB);
        Dev<uint16_t> pkey((size_t)rows * KB);
        attn::index_topk(bkeys.p, kstride / 8, pos.p, rows, ratio, scan, max_scan, true, KB, pool_new.p, pkey.p, KB, 0);
        auto hp = pool_new.host();
        auto ref = cs.get(P(L, call, "icand"));
        int w = (int)ref.shape[1], ok = 0;
        for (int r = 0; r < rows; ++r) {
          std::vector<uint8_t> m(w, 0);
          for (int i = 0; i < KB; ++i) {
            int b = hp[(size_t)r * KB + i];
            for (int p = b * 8; b >= 0 && p < b * 8 + 8 && p < w; ++p) m[p] = 1;
          }
          ok += std::memcmp(m.data(), ref.as<uint8_t>() + (size_t)r * w, w) == 0;
        }
        report_rows(P(L, call, "icand"), ok, rows);
      }
      // decode: the 4-rank split must give exactly the unsplit selection
      if (rows == 1 && !reindex) {
        const int NP = 4;
        Dev<int> cpos((size_t)NP * K), ppos((size_t)NP * KB);
        Dev<uint16_t> ckey((size_t)NP * K), pkey((size_t)NP * KB);
        for (int part = 0; part < NP; ++part) {
          attn::IndexScan ps;
          ps.part = part;
          ps.nparts = NP;
          int pmax = ((maxc + NP - 1) / NP + 63) / 64 * 64;
          attn::index_score(q, pos.p, rows, ratio, src.ik(), ps, pmax, keys.p, kstride, is_pool ? bkeys.p : nullptr,
                            kstride / 8, 0);
          attn::index_topk(keys.p, kstride, pos.p, rows, ratio, ps, pmax, false, K, cpos.p + part * K,
                           ckey.p + part * K, K, 0);
          if (is_pool)
            attn::index_topk(bkeys.p, kstride / 8, pos.p, rows, ratio, ps, pmax, true, KB, ppos.p + part * KB,
                             pkey.p + part * KB, KB, 0);
        }
        Dev<int> mpos(K);
        Dev<uint16_t> mkey(K);
        attn::index_merge(cpos.p, ckey.p, NP * K, 1, K, mpos.p, mkey.p, K, 0);
        report_rows(P(L, call, "itopk_split4"), mpos.host() == htop ? 1 : 0, 1);
        if (is_pool) {
          Dev<int> mp(KB);
          Dev<uint16_t> mk(KB);
          attn::index_merge(ppos.p, pkey.p, NP * KB, 1, KB, mp.p, mk.p, KB, 0);
          report_rows(P(L, call, "icand_split4"), mp.host() == pool_new.host() ? 1 : 0, 1);
        }
      }
      if (is_pool) pool = std::move(pool_new);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

// ---- attention ----------------------------------------------------------------------------------

// fp64 attention of one row (all 64 heads) from the bf16 q (after RoPE) and the dequantised keys, with
// the inverse RoPE on the output; the yardstick for both our kernel and the reference's own rounding.
struct Truth {
  double max_got = 0, max_ref = 0, sum_got = 0, sum_ref = 0;
  int n = 0;
  void add(const std::vector<double>& truth, const uint16_t* got, const uint16_t* ref) {
    double nt = 0, eg = 0, er = 0;
    for (size_t d = 0; d < truth.size(); ++d) {
      nt += truth[d] * truth[d];
      eg += (bf2f(got[d]) - truth[d]) * (bf2f(got[d]) - truth[d]);
      er += (bf2f(ref[d]) - truth[d]) * (bf2f(ref[d]) - truth[d]);
    }
    eg = std::sqrt(eg / nt);
    er = std::sqrt(er / nt);
    max_got = std::max(max_got, eg);
    max_ref = std::max(max_ref, er);
    sum_got += eg;
    sum_ref += er;
    ++n;
  }
  void report(const std::string& name) const {
    bool ok = max_got <= std::max(1.5 * max_ref, 2e-3);
    if (!ok) ++g_failures;
    std::printf("%-44s vs fp64 (rel L2 per head): ours max %.2e mean %.2e | reference max %.2e mean %.2e  %s\n",
                name.c_str(), max_got, sum_got / std::max(n, 1), max_ref, sum_ref / std::max(n, 1), ok ? "ok" : "FAIL");
  }
};
void attn_truth_row(const uint16_t* q, const std::vector<std::vector<float>>& keys, const float* sink,
                    const float* freqs, float scale, int head, std::vector<double>& o) {
  size_t n = keys.size();
  std::vector<double> s(n);
  double m = -1e300;
  for (size_t j = 0; j < n; ++j) {
    double acc = 0;
    for (int d = 0; d < 512; ++d) acc += (double)bf2f(q[d]) * keys[j][d];
    s[j] = acc * scale;
    m = std::max(m, s[j]);
  }
  double den = std::exp((double)sink[head] - m);
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

// Every layer's attention, call by call: the window from kv_write (a linear buffer for the prefill call,
// the ring for decode), compressed entries from the compressor, indices from the reference's itopk of
// the index source. Runs the four 16-head slices of the reference's 64 heads; decode also split-K.
void test_attention(const Case& cs) {
  std::map<int, CompState> comp;
  for (int L : cs.layers)
    if (L == kv_source(L) && cs.has(W(L, "w_c_wkv"))) comp[L].init(cs, L);
  int maxpos = cs.T + cs.D;
  struct LayerState {
    Dev<float2> table;
    Dev<uint16_t> kvw;
    Dev<uint8_t> ring_kv, ring_sc;
    Dev<float> sink;
  };
  std::map<int, LayerState> ls;
  for (int L : cs.layers) {
    auto& s = ls[L];
    s.table = Dev<float2>((size_t)maxpos * 32);
    attn::rope_table_build(s.table.p, maxpos, cs.ratio(L) > 0, 0);
    s.kvw = to_dev<uint16_t>(cs.get(W(L, "w_kv_norm")));
    s.ring_kv = Dev<uint8_t>((size_t)attn::kRing * 512);
    s.ring_sc = Dev<uint8_t>((size_t)attn::kRing * 16);
    s.sink = to_dev<float>(cs.get(W(L, "attn_sink")));
  }
  const int kmax_c = (cs.index_topk + 63) / 64 * 64, max_splits = 16;
  Dev<float> po((size_t)max_splits * 16 * 512), pml((size_t)max_splits * 16 * 2);
  attn::AttnScratch scratch{po.p, pml.p, max_splits};
  for (int call = 0; call <= cs.D; ++call) {
    int rows = cs.rows(call);
    auto pos_h = cs.pos(call);
    Dev<int> pos(pos_h.data(), pos_h.size());
    for (int L : cs.layers)
      if (comp.count(L)) comp[L].step(cs, call, pos.p, rows);
    for (int L : cs.layers) {
      auto& s = ls[L];
      if (!cs.has(P(L, call, "q_pre"))) {  // a --state-prefill call: the ring gets the recorded last rows
        auto tail = cs.get(P(L, call, "kv_pre_tail"));
        int n = (int)tail.shape[0];
        auto kvt = to_dev<uint16_t>(tail);
        attn::kv_write(kvt.p, s.kvw.p, pos.p + (rows - n), s.table.p, attn::Window{s.ring_kv.p, s.ring_sc.p}, nullptr,
                       n, 0);
        continue;
      }
      auto kvp = to_dev<uint16_t>(cs.get(P(L, call, "kv_pre")));
      attn::Window ring{s.ring_kv.p, s.ring_sc.p}, win = ring;
      Dev<uint8_t> lin_kv, lin_sc;
      if (call == 0) {  // prefill: a linear window buffer; the ring keeps the last kRing rows for decode
        lin_kv = Dev<uint8_t>((size_t)rows * 512);
        lin_sc = Dev<uint8_t>((size_t)rows * 16);
        win = attn::Window{lin_kv.p, lin_sc.p, -1, pos_h[0]};
        attn::kv_write(kvp.p, s.kvw.p, pos.p, s.table.p, win, nullptr, rows, 0);
        int n = std::min(rows, attn::kRing), off = rows - n;
        attn::kv_write(kvp.p + (size_t)off * 512, s.kvw.p, pos.p + off, s.table.p, ring, nullptr, n, 0);
      } else {
        attn::kv_write(kvp.p, s.kvw.p, pos.p, s.table.p, ring, nullptr, rows, 0);
      }
      attn::Compressed ckv{nullptr, nullptr, 0};
      Dev<int> topk;
      int kmax = 0;
      if (cs.ratio(L) > 0) {
        ckv = comp.at(kv_source(L)).ckv();
        auto it = cs.get(P(index_source(L), call, "itopk"));
        int kr = (int)it.shape[1];
        kmax = kmax_c;
        std::vector<int> h((size_t)rows * kmax, -1);
        for (int r = 0; r < rows; ++r)
          for (int i = 0; i < kr; ++i) h[(size_t)r * kmax + i] = it.as<int>()[(size_t)r * kr + i];
        topk = Dev<int>(h.data(), h.size());
      }
      auto qpre = cs.get(P(L, call, "q_pre"));
      auto oref = cs.get(P(L, call, "o"));
      std::vector<uint16_t> got(oref.numel()), got_split(oref.numel());
      for (int hs = 0; hs < 4; ++hs) {
        std::vector<uint16_t> qh((size_t)rows * 16 * 512);
        for (int r = 0; r < rows; ++r)
          std::memcpy(qh.data() + (size_t)r * 16 * 512, qpre.as<uint16_t>() + ((size_t)r * 64 + hs * 16) * 512,
                      16 * 512 * 2);
        Dev<uint16_t> q(qh.data(), qh.size()), out(qh.size());
        attn::sparse_attn(q.p, pos.p, {}, rows, s.table.p, win, ckv, topk.p, kmax, s.sink.p + hs * 16, out.p, 1,
                          scratch, 0);
        auto ho = out.host();
        for (int r = 0; r < rows; ++r)
          std::memcpy(got.data() + ((size_t)r * 64 + hs * 16) * 512, ho.data() + (size_t)r * 16 * 512, 16 * 512 * 2);
        if (rows == 1) {
          attn::sparse_attn(q.p, pos.p, {}, rows, s.table.p, win, ckv, topk.p, kmax, s.sink.p + hs * 16, out.p,
                            max_splits, scratch, 0);
          ho = out.host();
          std::memcpy(got_split.data() + (size_t)hs * 16 * 512, ho.data(), 16 * 512 * 2);
        }
      }
      // not bit-exact (fp32 GEMM order, online vs global max): element-wise for information, judged against fp64
      report(P(L, call, "attn_o"), cmp_bf16(got.data(), oref.as<uint16_t>(), oref.numel()), false, false);
      if (rows == 1)
        report(P(L, call, "attn_o_splitk"), cmp_bf16(got_split.data(), oref.as<uint16_t>(), oref.numel()), false, false);
      {
        auto qref = cs.get(P(L, call, "q"));
        auto freqs = cs.get(W(L, "freqs"));
        auto sink = cs.get(W(L, "attn_sink"));
        auto wkv = call == 0 ? lin_kv.host() : s.ring_kv.host();
        auto wsc = call == 0 ? lin_sc.host() : s.ring_sc.host();
        std::vector<uint8_t> hk, hs;
        std::vector<int> ht;
        if (kmax) {
          hk = comp.at(kv_source(L)).ckv_kv.host();
          hs = comp.at(kv_source(L)).ckv_sc.host();
          ht = topk.host();
        }
        Truth tr, trs;
        std::vector<double> o;
        for (int r : {0, 1, 63, 127, 128, 129, 200, 299, 599}) {
          if (r >= rows) continue;
          int p = pos_h[r];
          std::vector<std::vector<float>> keys;
          for (int kp = std::max(0, p - 127); kp <= p; ++kp) {
            int slot = (kp - win.base) & win.mask;
            std::vector<float> k(512);
            for (int d = 0; d < 512; ++d)
              k[d] = host_e4m3(wkv[(size_t)slot * 512 + d]) * std::ldexp(1.0f, wsc[(size_t)slot * 16 + d / 32] - 127);
            keys.push_back(k);
          }
          for (int i = 0; i < kmax; ++i) {
            int idx = ht[(size_t)r * kmax + i];
            if (idx < 0) continue;
            uint16_t kb[512];
            dequant_ckv(hk, hs, idx, kb);
            std::vector<float> k(512);
            for (int d = 0; d < 512; ++d) k[d] = bf2f(kb[d]);
            keys.push_back(k);
          }
          for (int h = 0; h < 64; ++h) {
            size_t off = ((size_t)r * 64 + h) * 512;
            attn_truth_row(qref.as<uint16_t>() + off, keys, sink.as<float>(), freqs.as<float>() + (size_t)p * 64,
                           (float)std::pow(512.0, -0.5), h, o);
            tr.add(o, got.data() + off, oref.as<uint16_t>() + off);
            if (rows == 1) trs.add(o, got_split.data() + off, oref.as<uint16_t>() + off);
          }
        }
        tr.report(P(L, call, "attn_o"));
        if (rows == 1) trs.report(P(L, call, "attn_o_splitk"));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

// ---- mHC ----------------------------------------------------------------------------------------

// A gen_mhc.py case: the sinkhorn, hc_pre and hc_post from the reference's own inputs must be bit-exact;
// the projection (fp32, K = 20480) differs from cuBLAS in summation order only, also split by copy.
void test_mhc(const std::string& dir, const std::string& json) {
  auto layers = json_ints(json, "layers");
  int T = json_int(json, "T");
  auto get = [&](const std::string& n) { return npy_load(dir + "/" + n + ".npy"); };
  std::printf("mhc case %s: T=%d\n", dir.c_str(), T);
  for (int L : layers)
    for (const char* which : {"attn", "ffn"}) {
      std::string p = "L" + std::to_string(L) + "_" + which + "_";
      auto x = to_dev<uint16_t>(get(p + "x"));
      auto fn = to_dev<float>(get(p + "fn"));
      auto scale = to_dev<float>(get(p + "scale")), base = to_dev<float>(get(p + "base"));
      auto rpre = get(p + "pre"), rpost = get(p + "post"), rcomb = get(p + "comb"), rmix = get(p + "mixes");
      Dev<float> pre(T * 4), post(T * 4), comb(T * 16);
      // sinkhorn from the reference's mixes: partial = (mixes, 20480) makes rsqrt exactly 1
      std::vector<float> hp((size_t)T * 25);
      for (int r = 0; r < T; ++r) {
        for (int i = 0; i < 24; ++i) hp[(size_t)r * 25 + i] = rmix.as<float>()[(size_t)r * 24 + i];
        hp[(size_t)r * 25 + 24] = (float)attn::kHcDim;
      }
      Dev<float> part(hp.data(), hp.size());
      attn::hc_finish(part.p, T, scale.p, base.p, pre.p, post.p, comb.p, 0);
      report(p + "pre", cmp_f32(pre.host().data(), rpre.as<float>(), rpre.numel()), true);
      report(p + "post", cmp_f32(post.host().data(), rpost.as<float>(), rpost.numel()), true);
      report(p + "comb", cmp_f32(comb.host().data(), rcomb.as<float>(), rcomb.numel()), true);
      // hc_pre / hc_post from the reference's coefficients
      auto rp = to_dev<float>(rpre), rpo = to_dev<float>(rpost), rc = to_dev<float>(rcomb);
      Dev<uint16_t> y(T * attn::kDim), yraw(T * attn::kDim), ypost((size_t)T * attn::kHcDim);
      auto ryp = get(p + "y_pre"), rypo = get(p + "y_post");
      attn::hc_pre(x.p, rp.p, T, nullptr, y.p, nullptr, 0);
      report(p + "hc_pre", cmp_bf16(y.host().data(), ryp.as<uint16_t>(), ryp.numel()), true);
      auto sub = to_dev<uint16_t>(get(p + "sub"));
      attn::hc_post(sub.p, x.p, rpo.p, rc.p, T, ypost.p, 0);
      report(p + "hc_post", cmp_bf16(ypost.host().data(), rypo.as<uint16_t>(), rypo.numel()), true);
      // hc_pre with the fused RMSNorm (weights 1): y_raw is hc_pre, y its norm (checked in fp64, 1 ulp)
      {
        std::vector<uint16_t> ones(attn::kDim, 0x3f80);
        Dev<uint16_t> w(ones.data(), ones.size());
        attn::hc_pre(x.p, rp.p, T, w.p, y.p, yraw.p, 0);
        report(p + "hc_pre_raw", cmp_bf16(yraw.host().data(), ryp.as<uint16_t>(), ryp.numel()), true);
        auto hy = y.host();
        std::vector<uint16_t> want(hy.size());
        for (int r = 0; r < T; ++r) {
          double ss = 0;
          for (int d = 0; d < attn::kDim; ++d) ss += (double)bf2f(ryp.as<uint16_t>()[(size_t)r * attn::kDim + d]) * bf2f(ryp.as<uint16_t>()[(size_t)r * attn::kDim + d]);
          double rs = 1.0 / std::sqrt(ss / attn::kDim + 1e-20);
          for (int d = 0; d < attn::kDim; ++d) want[(size_t)r * attn::kDim + d] = host_f2bf((float)(bf2f(ryp.as<uint16_t>()[(size_t)r * attn::kDim + d]) * rs));
        }
        auto c = cmp_bf16(hy.data(), want.data(), want.size());
        report(p + "hc_pre_norm", c);
      }
      // the projection: one rank, and split by copy (4 partials added in rank order)
      int nch = attn::hc_chunks(0, attn::kHcDim);
      Dev<float> chunks((size_t)nch * T * 25), full(T * 25);
      attn::hc_partial(x.p, T, fn.p, 0, attn::kHcDim, chunks.p, full.p, 0);
      auto hf = full.host();
      auto rraw = get(p + "raw"), rrs = get(p + "rsqrt");
      {
        std::vector<float> got((size_t)T * 24), rs(T);
        for (int r = 0; r < T; ++r) {
          for (int i = 0; i < 24; ++i) got[(size_t)r * 24 + i] = hf[(size_t)r * 25 + i];
          rs[r] = 1.0f / std::sqrt(hf[(size_t)r * 25 + 24] / attn::kHcDim);
        }
        report(p + "hc_proj", cmp_f32(got.data(), rraw.as<float>(), rraw.numel(), 1e-4, 1e-4));
        report(p + "hc_rsqrt", cmp_f32(rs.data(), rrs.as<float>(), rrs.numel(), 1e-5));
      }
      std::vector<float> split((size_t)T * 25, 0.f);
      for (int c = 0; c < 4; ++c) {
        int n4 = attn::hc_chunks(c * attn::kDim, (c + 1) * attn::kDim);
        Dev<float> ch4((size_t)n4 * T * 25), out4(T * 25);
        attn::hc_partial(x.p, T, fn.p, c * attn::kDim, (c + 1) * attn::kDim, ch4.p, out4.p, 0);
        auto h4 = out4.host();
        for (size_t i = 0; i < split.size(); ++i) split[i] += h4[i];
      }
      for (auto* src : {&hf, &split}) {
        Dev<float> pp(src->data(), src->size());
        attn::hc_finish(pp.p, T, scale.p, base.p, pre.p, post.p, comb.p, 0);
        std::string tag = src == &hf ? "e2e_" : "e2e_split4_";
        report(p + tag + "pre", cmp_f32(pre.host().data(), rpre.as<float>(), rpre.numel(), 1e-5));
        report(p + tag + "post", cmp_f32(post.host().data(), rpost.as<float>(), rpost.numel(), 1e-5));
        report(p + tag + "comb", cmp_f32(comb.host().data(), rcomb.as<float>(), rcomb.numel(), 1e-4));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }
}

}  // namespace

// ---- DSpark -------------------------------------------------------------------------------------

// A gen_dspark.py case, per stage: the main stream seeds the ring (call 0, its last kRing rows); each decode
// call writes the main token at start = T + d and the block's 5 drafts at start+1..+5 to the ring, then runs
// the draft rows over [max(0, start-127), start+5]. kv_write must be exact; attention is judged against fp64.
void test_dspark(const std::string& dir, const std::string& json) {
  Case cs;
  cs.dir = dir;
  auto stages = json_ints(json, "dspark");
  int T = json_int(json, "T"), D = json_int(json, "decode"), B = json_int(json, "block");
  std::printf("dspark case %s: T=%d, decode=%d, block=%d\n", dir.c_str(), T, D, B);
  int maxpos = T + D + B;
  Dev<float2> table((size_t)maxpos * 32);
  attn::rope_table_build(table.p, maxpos, false, 0);
  const int max_splits = 4;
  Dev<float> po((size_t)B * max_splits * 16 * 512), pml((size_t)B * max_splits * 16 * 2);
  attn::AttnScratch scratch{po.p, pml.p, max_splits};
  for (int st : stages) {
    std::string S = "S" + std::to_string(st) + "_";
    auto freqs = cs.get(S + "freqs");
    report(S + "rope_table", cmp_f32(reinterpret_cast<const float*>(table.host().data()), freqs.as<float>(),
                                     freqs.numel()), true);
    auto kvw = to_dev<uint16_t>(cs.get(S + "w_kv_norm"));
    auto sink = cs.get(S + "attn_sink");
    auto sink_d = to_dev<float>(sink);
    Dev<uint8_t> ring_kv((size_t)attn::kRing * 512), ring_sc((size_t)attn::kRing * 16);
    attn::Window ring{ring_kv.p, ring_sc.p};
    {
      auto kvp = to_dev<uint16_t>(cs.get(S + "c0_main_kv_pre"));
      int n = std::min(T, attn::kRing), off = T - n;
      std::vector<int> ph(n);
      for (int i = 0; i < n; ++i) ph[i] = off + i;
      Dev<int> pos(ph.data(), n);
      Dev<uint16_t> kv((size_t)n * 512);
      attn::kv_write(kvp.p + (size_t)off * 512, kvw.p, pos.p, table.p, ring, kv.p, n, 0);
      auto ref = cs.get(S + "c0_main_kv");
      report(S + "c0_main_kv_write", cmp_bf16(kv.host().data(), ref.as<uint16_t>() + (size_t)off * 512, (size_t)n * 512));
    }
    for (int call = 1; call <= D; ++call) {
      int start = T + call - 1;
      std::string Pc = S + "c" + std::to_string(call) + "_";
      {
        auto mkv = to_dev<uint16_t>(cs.get(Pc + "main_kv_pre"));
        Dev<int> mpos(&start, 1);
        Dev<uint16_t> kv(512);
        attn::kv_write(mkv.p, kvw.p, mpos.p, table.p, ring, kv.p, 1, 0);
        auto ref = cs.get(Pc + "main_kv");
        report(Pc + "main_kv_write", cmp_bf16(kv.host().data(), ref.as<uint16_t>(), 512));
      }
      std::vector<int> dp(B), lo(B), hi(B);
      for (int i = 0; i < B; ++i) {
        dp[i] = start + 1 + i;
        lo[i] = std::max(0, start - 127);
        hi[i] = start + B;
      }
      Dev<int> dpos(dp.data(), B), dlo(lo.data(), B), dhi(hi.data(), B);
      {
        auto dkv = to_dev<uint16_t>(cs.get(Pc + "kv_pre"));
        Dev<uint16_t> kv((size_t)B * 512);
        attn::kv_write(dkv.p, kvw.p, dpos.p, table.p, ring, kv.p, B, 0);
        auto ref = cs.get(Pc + "kv");
        report(Pc + "draft_kv_write", cmp_bf16(kv.host().data(), ref.as<uint16_t>(), ref.numel()));
      }
      auto qpre = cs.get(Pc + "q_pre");
      auto oref = cs.get(Pc + "o");
      std::vector<uint16_t> got(oref.numel()), got_split(oref.numel());
      for (int hs = 0; hs < 4; ++hs) {
        std::vector<uint16_t> qh((size_t)B * 16 * 512);
        for (int r = 0; r < B; ++r)
          std::memcpy(qh.data() + (size_t)r * 16 * 512, qpre.as<uint16_t>() + ((size_t)r * 64 + hs * 16) * 512,
                      16 * 512 * 2);
        Dev<uint16_t> q(qh.data(), qh.size()), out(qh.size());
        for (int sp : {1, max_splits}) {
          attn::sparse_attn(q.p, dpos.p, {dlo.p, dhi.p}, B, table.p, ring, {nullptr, nullptr, 0}, nullptr, 0,
                            sink_d.p + hs * 16, out.p, sp, scratch, 0);
          auto ho = out.host();
          auto& dst = sp == 1 ? got : got_split;
          for (int r = 0; r < B; ++r)
            std::memcpy(dst.data() + ((size_t)r * 64 + hs * 16) * 512, ho.data() + (size_t)r * 16 * 512, 16 * 512 * 2);
        }
      }
      report(Pc + "attn_o", cmp_bf16(got.data(), oref.as<uint16_t>(), oref.numel()), false, false);
      report(Pc + "attn_o_splitk", cmp_bf16(got_split.data(), oref.as<uint16_t>(), oref.numel()), false, false);
      auto qref = cs.get(Pc + "q");
      auto wkv = ring_kv.host();
      auto wsc = ring_sc.host();
      std::vector<std::vector<float>> keys;
      for (int kp = lo[0]; kp <= hi[0]; ++kp) {
        int slot = kp & (attn::kRing - 1);
        std::vector<float> k(512);
        for (int d = 0; d < 512; ++d)
          k[d] = host_e4m3(wkv[(size_t)slot * 512 + d]) * std::ldexp(1.0f, wsc[(size_t)slot * 16 + d / 32] - 127);
        keys.push_back(k);
      }
      Truth tr, trs;
      std::vector<double> o;
      for (int r = 0; r < B; ++r)
        for (int h = 0; h < 64; ++h) {
          size_t off = ((size_t)r * 64 + h) * 512;
          attn_truth_row(qref.as<uint16_t>() + off, keys, sink.as<float>(), freqs.as<float>() + (size_t)dp[r] * 64,
                         (float)std::pow(512.0, -0.5), h, o);
          tr.add(o, got.data() + off, oref.as<uint16_t>() + off);
          trs.add(o, got_split.data() + off, oref.as<uint16_t>() + off);
        }
      tr.report(Pc + "attn_o");
      trs.report(Pc + "attn_o_splitk");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

int main(int argc, char** argv) {
  if (argc < 2) die("usage: attn_tests <case dir> [test...]");
  {
    std::ifstream f(std::string(argv[1]) + "/manifest.json");
    std::stringstream ss;
    ss << f.rdbuf();
    if (ss.str().find("\"dspark\"") != std::string::npos) {
      test_dspark(argv[1], ss.str());
      std::printf("%s (%d failing checks)\n", g_failures ? "FAILED" : "PASSED", g_failures);
      return g_failures ? 1 : 0;
    }
    if (ss.str().find("\"decode\"") == std::string::npos) {
      test_mhc(argv[1], ss.str());
      std::printf("%s (%d failing checks)\n", g_failures ? "FAILED" : "PASSED", g_failures);
      return g_failures ? 1 : 0;
    }
  }
  Case cs = load_case(argv[1]);
  std::printf("case %s: layers", cs.dir.c_str());
  for (int L : cs.layers) std::printf(" %d", L);
  std::printf(", T=%d, decode=%d\n", cs.T, cs.D);
  for (int L : cs.layers) test_prep(cs, L);
  for (int L : cs.layers) test_compressor(cs, L);
  test_indexer(cs);
  test_attention(cs);
  std::printf("%s (%d failing checks)\n", g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
