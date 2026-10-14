// engine/attn against gen_ref.py's golden data (one full-indexer layer at the real dimensions, random weights).
// usage: attn_tests DIR [--only A,B,...]. Rank r's kernels run for r = 0..3 on this one GPU; the all-gathers and
// the o_proj/down all-reduce are host copies (in rank order). Every check prints PASS/FAIL with its numbers.
#include <cstdarg>
#include <map>
#include <set>

#include "attn/attn.h"
#include "testutil.h"

using namespace glm;
using tu::Dev;

namespace {

std::string g_dir;
tu::Tally T;
std::map<std::string, tu::Npy> g_cache;
const tu::Npy& L(const std::string& n) {
  auto it = g_cache.find(n);
  if (it == g_cache.end()) it = g_cache.emplace(n, tu::load(g_dir + "/" + n + ".npy")).first;
  return it->second;
}
bool has(const std::string& n) { return tu::exists(g_dir + "/" + n + ".npy"); }
int manifest_int(const char* key) {
  std::ifstream f(g_dir + "/manifest.json");
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto p = s.find(std::string("\"") + key + "\":");
  if (p == std::string::npos) die("manifest: no %s", key);
  return std::stoi(s.substr(p + std::strlen(key) + 3));
}
float e4m3_host(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
  return s ? -v : v;
}
void sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

struct FP8W {
  Dev<uint8_t> w;
  Dev<float> s;
  int rs, ks;
};
FP8W fp8w(const std::string& name, int rs, int ks) {
  return FP8W{tu::to_dev<uint8_t>(L(name + "_w")), tu::to_dev<float>(L(name + "_s")), rs, ks};
}
struct RankW {
  FP8W x, qb, iq, uk, uv, o;
};
std::vector<RankW> g_rw;
void load_weights() {
  for (int r = 0; r < kTP; ++r) {
    const std::string p = "r" + std::to_string(r) + "_";
    g_rw.push_back(RankW{fp8w(p + "xproj", 16, 128), fp8w(p + "q_b", 128, 128), fp8w(p + "wq_b", 128, 128),
                         fp8w(p + "w_uk", 128, 64), fp8w(p + "w_uv", 64, 128), fp8w(p + "o_proj", 128, 128)});
  }
}
GemvSeg seg(const FP8W& w, int n, void* y, int ldy, int group_rows = 0, int x_group = 0) {
  GemvSeg g;
  g.w = w.w.p;
  g.s = w.s.p;
  g.n = n;
  g.rs = w.rs;
  g.ks = w.ks;
  g.y = y;
  g.ldy = ldy;
  g.group_rows = group_rows;
  g.x_group = x_group;
  return g;
}
// a GEMV over any number of rows, 16 at a time
void gemv_rows(bool fp8, const uint16_t* x, int ldx, int rows, int K, std::vector<GemvSeg> segs, GemvOut out,
               int splitk = 1, int max_rows = 16) {
  const size_t es = out == GemvOut::kF32 ? 4 : 2;
  for (int r0 = 0; r0 < rows; r0 += max_rows) {
    const int n = std::min(max_rows, rows - r0);
    std::vector<GemvSeg> s = segs;
    for (auto& g : s) g.y = static_cast<char*>(g.y) + (size_t)r0 * g.ldy * es;
    gemv(fp8, x + (size_t)r0 * ldx, ldx, n, K, s.data(), (int)s.size(), out, splitk, 0);
  }
}

// a projection as the engine runs it: the GEMM for prefill chunks (more than 16 rows), else the GEMV
void proj_rows(const uint16_t* x, int ldx, int rows, int K, std::vector<GemvSeg> segs, GemvOut out, int splitk = 1) {
  if (rows <= kMaxRows) gemv(true, x, ldx, rows, K, segs.data(), (int)segs.size(), out, splitk, 0);
  else gemm(x, ldx, rows, K, segs.data(), (int)segs.size(), out, splitk, 0);
}

// ---- A: the tiles decode to zai's values through the kernel's own fragment mapping ---------------------------
void test_dequant() {
  const auto& qc = L("zai_q_a_codes");
  const auto& qs = L("zai_q_a_scale");
  const auto* codes = qc.as<uint8_t>();
  const auto* sc = qs.as<float>();
  // rank 0's x-projection rows 0..511 are q_a rows 0..511: one-hot x rows give W[n][k] * scale exactly
  Dev<uint16_t> x(16 * kDim);
  Dev<float> y(16 * kXRows);
  size_t bad = 0, n = 0;
  for (int b = 0; b < 4; ++b) {
    std::vector<uint16_t> hx(16 * kDim, 0);
    int ks[16];
    for (int i = 0; i < 16; ++i) {
      ks[i] = (b * 16 + i) * 97 % kDim;
      hx[i * kDim + ks[i]] = 0x3F80;
    }
    x.up(hx);
    GemvSeg s = seg(g_rw[0].x, kXRows, y.p, kXRows);
    gemv(true, x.p, kDim, 16, kDim, &s, 1, GemvOut::kF32, 4, 0);
    auto hy = y.down();
    for (int i = 0; i < 16; ++i)
      for (int r = 0; r < kXQa; ++r) {
        const float want = e4m3_host(codes[(size_t)r * kDim + ks[i]]) * sc[(r / 128) * (kDim / 128) + ks[i] / 128];
        ++n;
        bad += !(want == hy[i * kXRows + r]);  // -0 (code 0x80) sums to +0
      }
  }
  T(bad == 0, "A dequant: x-projection tile (q_a rows of rank 0) = e4m3 x scale (as floats), %zu of %zu differ", bad, n);
  // W_UK^T of rank 0: one-hot d in every head's q_nope
  const auto& kc = L("zai_kv_b_codes");
  const auto& ksc = L("zai_kv_b_scale");
  Dev<uint16_t> xq(16 * kOProjK);
  Dev<float> yq(16 * kHeads * kKvLora);
  bad = n = 0;
  for (int b = 0; b < 2; ++b) {
    std::vector<uint16_t> hx(16 * kOProjK, 0);
    int ds[16];
    for (int i = 0; i < 16; ++i) {
      ds[i] = (b * 16 + i) * 11 % kNope;
      for (int h = 0; h < kHeads; ++h) hx[i * kOProjK + h * kQkDim + ds[i]] = 0x3F80;
    }
    xq.up(hx);
    GemvSeg s = seg(g_rw[0].uk, kHeads * kKvLora, yq.p, kHeads * kKvLora, kKvLora, kQkDim);
    gemv(true, xq.p, kOProjK, 16, kNope, &s, 1, GemvOut::kF32, 1, 0);
    auto hy = yq.down();
    for (int i = 0; i < 16; ++i)
      for (int h = 0; h < kHeads; ++h)
        for (int c = 0; c < kKvLora; ++c) {
          const int row = 448 * h + ds[i];
          const float want = e4m3_host(kc.as<uint8_t>()[(size_t)row * kKvLora + c]) * ksc.as<float>()[(row / 128) * 4 + c / 128];
          ++n;
          bad += !(want == hy[(size_t)i * kHeads * kKvLora + h * kKvLora + c]);
        }
  }
  T(bad == 0, "A dequant: W_UK^T tile (rank 0, transposed kv_b rows) = e4m3 x scale (as floats), %zu of %zu differ", bad, n);
}

// bf16 comparison with a summary line: pass when no value is off by more than one ulp in more than frac of them
bool check_bf16(const char* what, const uint16_t* a, const uint16_t* b, size_t rows, size_t cols, size_t lda, size_t ldb,
                double frac = 1e-4, bool exact = false) {
  auto c = tu::cmp_bf16(a, b, rows, cols, lda, ldb);
  const bool ok = exact ? c.diff == 0 : (double)c.over1ulp <= frac * c.n && (double)c.diff <= 0.02 * c.n;
  return T(ok, "%s: %zu values, %zu differ (%.2e), %zu by > 1 ulp, max |d| %.3g", what, c.n, c.diff,
           (double)c.diff / std::max<size_t>(c.n, 1), c.over1ulp, c.max_abs);
}

// ---- B: projection GEMVs against the reference, 1-16 rows ---------------------------------------------------
void test_gemvs(int S, int nd) {
  const auto& x = L("x");
  const auto* hx = x.as<uint16_t>();
  const auto* qa = L("qa").as<uint16_t>();
  const auto* ckv = L("ckv").as<uint16_t>();
  const auto* kpre = L("kpre").as<uint16_t>();
  const auto* qr = L("qr").as<uint16_t>();
  const auto* q = L("q").as<uint16_t>();
  const auto* iq = L("iq").as<uint16_t>();
  const int r0 = S - nd;
  Dev<uint16_t> dx(nd * kDim), dqr(nd * kQLora);
  dx.up(hx + (size_t)r0 * kDim, (size_t)nd * kDim);
  dqr.up(qr + (size_t)r0 * kQLora, (size_t)nd * kQLora);
  Dev<uint16_t> g(nd * kTP * kXRows), qo(nd * kOProjK), io(nd * kIdxHeadsRank * kIdxDim);
  std::vector<uint16_t> first;
  for (int rows : {16, 6, 1}) {
    if (rows > nd) continue;
    for (int r = 0; r < kTP; ++r) {
      GemvSeg s = seg(g_rw[r].x, kXRows, g.p + r * kXRows, kTP * kXRows);
      gemv(true, dx.p, kDim, rows, kDim, &s, 1, GemvOut::kBF16, 4, 0);
    }
    auto hg = g.down();
    if (rows == 16) first = hg;
    else
      T(std::memcmp(hg.data(), first.data(), (size_t)rows * kTP * kXRows * 2) == 0,
        "B x-projection at %d rows equals its rows at 16 rows, bit for bit", rows);
    if (rows != 16) continue;
    // compare the gathered parts
    std::vector<uint16_t> eq(rows * kQLora), ec(rows * (kLat)), ek(rows * kIdxDim);
    for (int i = 0; i < rows; ++i)
      for (int r = 0; r < kTP; ++r) {
        const uint16_t* b = hg.data() + (size_t)i * kTP * kXRows + r * kXRows;
        std::memcpy(&eq[i * kQLora + r * kXQa], b, kXQa * 2);
        std::memcpy(&ec[i * kLat + r * kXKv], b + kXQa, kXKv * 2);
        std::memcpy(&ec[i * kLat + kKvLora + r * kXRope], b + kXQa + kXKv, kXRope * 2);
        std::memcpy(&ek[i * kIdxDim + r * kXWk], b + kXRowsShared, kXWk * 2);
      }
    check_bf16("B q_a (FP8 GEMV, 16 rows, 4 ranks' slices gathered)", eq.data(), qa + (size_t)r0 * kQLora, rows, kQLora, kQLora, kQLora);
    check_bf16("B kv_a_proj_with_mqa", ec.data(), ckv + (size_t)r0 * kLat, rows, kLat, kLat, kLat);
    check_bf16("B indexer wk", ek.data(), kpre + (size_t)r0 * kIdxDim, rows, kIdxDim, kIdxDim, kIdxDim);
  }
  // the index weights fused into the x-projection's launch = the two launches; the GEMV's launch configurations
  // (loads in flight, warps a CTA) agree, bit for bit
  {
    Dev<uint16_t> g1((size_t)nd * kXRows), g2((size_t)nd * kXRows);
    Dev<float> w1((size_t)nd * kIdxHeadsRank), w2((size_t)nd * kIdxHeadsRank);
    GemvSeg s1 = seg(g_rw[1].x, kXRows, g1.p, kXRows), s2 = seg(g_rw[1].x, kXRows, g2.p, kXRows);
    Dev<uint16_t> wpd = tu::to_dev<uint16_t>(L("wp"));
    const uint16_t* wp1 = wpd.p + (size_t)kIdxHeadsRank * kDim;  // rank 1's 8 heads
    gemv(true, dx.p, kDim, nd, kDim, &s1, 1, GemvOut::kBF16, 4, 0);
    index_weights(dx.p, nd, wp1, kIdxHeadsRank, w1.p, kIdxHeadsRank, 0);
    GemvIw iw;
    iw.wproj = wp1;
    iw.nh = kIdxHeadsRank;
    iw.w = w2.p;
    iw.ldw = kIdxHeadsRank;
    gemv(true, dx.p, kDim, nd, kDim, &s2, 1, GemvOut::kBF16, 4, 0, false, &iw);
    sync();
    T(g1.down() == g2.down() && w1.down() == w2.down(),
      "B x-projection with the index weights in its launch = the two launches, bit for bit (%d rows)", nd);
    Dev<uint16_t> q1((size_t)nd * kOProjK), q2((size_t)nd * kOProjK);
    GemvSeg b1 = seg(g_rw[1].qb, kOProjK, q1.p, kOProjK), b2 = seg(g_rw[1].qb, kOProjK, q2.p, kOProjK);
    gemv(true, dqr.p, kQLora, nd, kQLora, &b1, 1, GemvOut::kBF16, 1, 0);
    bool same = true;
    for (int cfg = 0; cfg < 6; ++cfg) {
      gemv_config(cfg);
      gemv(true, dx.p, kDim, nd, kDim, &s2, 1, GemvOut::kBF16, 4, 0);
      gemv(true, dqr.p, kQLora, nd, kQLora, &b2, 1, GemvOut::kBF16, 1, 0);
      sync();
      same = same && g1.down() == g2.down() && q1.down() == q2.down();
    }
    gemv_config(-1);
    T(same, "B GEMV launch configurations 0-5 = the default (by shape), bit for bit (x-projection split-K 4, q_b)");
  }
  // q_b and the indexer's wq_b in one launch (two segments), every rank
  for (int r = 0; r < kTP; ++r) {
    GemvSeg s[2] = {seg(g_rw[r].qb, kOProjK, qo.p, kOProjK), seg(g_rw[r].iq, kIdxHeadsRank * kIdxDim, io.p, kIdxHeadsRank * kIdxDim)};
    gemv(true, dqr.p, kQLora, nd, kQLora, s, 2, GemvOut::kBF16, 1, 0);
    auto hq = qo.down(), hi = io.down();
    char w[128];
    std::snprintf(w, sizeof w, "B q_b rank %d (16 heads)", r);
    check_bf16(w, hq.data(), q + (size_t)r0 * kHeadsAll * kQkDim + r * kOProjK, nd, kOProjK, kOProjK, kHeadsAll * kQkDim);
    std::snprintf(w, sizeof w, "B indexer wq_b rank %d (8 heads)", r);
    check_bf16(w, hi.data(), iq + (size_t)r0 * kIdxHeads * kIdxDim + r * 1024, nd, 1024, 1024, kIdxHeads * kIdxDim);
  }
}

// gathered x-projection outputs [rows][4][688] made from the reference's q_a, kv_a and wk outputs
std::vector<uint16_t> gathered_from_ref(int S) {
  const auto* qa = L("qa").as<uint16_t>();
  const auto* ckv = L("ckv").as<uint16_t>();
  const auto* kpre = L("kpre").as<uint16_t>();
  std::vector<uint16_t> g((size_t)S * kTP * kXRows);
  for (int i = 0; i < S; ++i)
    for (int r = 0; r < kTP; ++r) {
      uint16_t* b = g.data() + ((size_t)i * kTP + r) * kXRows;
      std::memcpy(b, qa + (size_t)i * kQLora + r * kXQa, kXQa * 2);
      std::memcpy(b + kXQa, ckv + (size_t)i * kLat + r * kXKv, kXKv * 2);
      std::memcpy(b + kXQa + kXKv, ckv + (size_t)i * kLat + kKvLora + r * kXRope, kXRope * 2);
      std::memcpy(b + kXRowsShared, kpre + (size_t)i * kIdxDim + r * kXWk, kXWk * 2);
    }
  return g;
}

struct Common {
  Dev<uint16_t> w_in, w_qn, w_kvn, kn_w, kn_b, wp;
  Dev<uint32_t> rope;
  Dev<int> pos;  // 0..S-1 (and more)
};
Common* C;

// ---- C: prep (norms, RoPE, cache entries) from the reference's projections ---------------------------------
void test_prep(int S) {
  auto hg = gathered_from_ref(S);
  Dev<uint16_t> g(hg), qr((size_t)S * kQLora), kv((size_t)S * kLat), ik((size_t)S * kIdxDim);
  PrepArgs a;
  a.xg = g.p;
  a.pos = C->pos.p;
  a.rope = C->rope.p;
  a.q_norm = C->w_qn.p;
  a.kv_norm = C->w_kvn.p;
  a.k_norm_w = C->kn_w.p;
  a.k_norm_b = C->kn_b.p;
  a.qr = qr.p;
  a.kv.kv = kv.p;
  a.ik.k = reinterpret_cast<uint8_t*>(ik.p);
  attn_prep(a, S, 0);
  sync();
  auto hq = qr.down(), hkv = kv.down(), hk = ik.down();
  check_bf16("C q_a_layernorm (RMSNorm 1e-6, two roundings)", hq.data(), L("qr").as<uint16_t>(), S, kQLora, kQLora, kQLora);
  check_bf16("C kv_a_layernorm into the cache", hkv.data(), L("c").as<uint16_t>(), S, kKvLora, kLat, kKvLora);
  check_bf16("C k_rope RoPE (bf16 arithmetic, de-interleaved) into the cache", hkv.data() + kKvLora, L("kr").as<uint16_t>(), S,
             kRope, kLat, kRope, 0, true);
  check_bf16("C index key (LayerNorm with bias, RoPE on 0..63)", hk.data(), L("kidx").as<uint16_t>(), S, kIdxDim, kIdxDim, kIdxDim);
  // the input RMSNorm
  Dev<uint16_t> xr = tu::to_dev<uint16_t>(L("x_raw")), xn((size_t)S * kDim);
  rmsnorm(xr.p, kDim, C->w_in.p, kNormEps, kDim, xn.p, kDim, S, 0);
  auto hx = xn.down();
  check_bf16("C input_layernorm (RMSNorm 1e-5)", hx.data(), L("x").as<uint16_t>(), S, kDim, kDim, kDim);
  // index weights (all 32 heads)
  Dev<uint16_t> x = tu::to_dev<uint16_t>(L("x"));
  Dev<float> w((size_t)S * kIdxHeads);
  index_weights(x.p, S, C->wp.p, kIdxHeads, w.p, kIdxHeads, 0);
  auto hw = w.down();
  const float* rw = L("iw").as<float>();
  // f32 dots of 6144 terms in a different order than torch's: relative to the row's largest weight
  double mx = 0;
  for (int r = 0; r < S; ++r) {
    double rmax = 1e-30;
    for (int h = 0; h < kIdxHeads; ++h) rmax = std::max(rmax, (double)std::fabs(rw[(size_t)r * kIdxHeads + h]));
    for (int h = 0; h < kIdxHeads; ++h) {
      const size_t i = (size_t)r * kIdxHeads + h;
      mx = std::max(mx, (double)std::fabs(hw[i] - rw[i]) / rmax);
    }
  }
  T(mx < 1e-5, "C index weights (f32 dot of bf16, x 32^-0.5): max |d| / row max %.2e", mx);
}

// ---- E: the indexer ------------------------------------------------------------------------------------------
struct IdxSetup {
  Dev<uint16_t> iq, kidx;
  Dev<float> w, scores;
  Dev<uint32_t> work;
  size_t ld;
};
// near-tie check of a selection against the reference's: each difference is within tol of the 2048th score
struct TopkCmp {
  int rows = 0, equal_rows = 0, diff_elems = 0, far = 0;
};
TopkCmp cmp_topk(const int* got, const int* ref, const float* refsc, int rows, int S, double tol) {
  TopkCmp c;
  for (int i = 0; i < rows; ++i) {
    const int* a = got + (size_t)i * kTopK;
    const int* b = ref + (size_t)i * kTopK;
    ++c.rows;
    if (std::equal(a, a + kTopK, b)) { ++c.equal_rows; continue; }
    const float* sc = refsc + (size_t)i * S;
    std::vector<float> v(b, b + kTopK);
    float thr = INFINITY;
    for (int k = 0; k < kTopK; ++k) thr = std::min(thr, sc[b[k]]);
    std::set<int> A(a, a + kTopK), B(b, b + kTopK);
    for (int p : A)
      if (!B.count(p)) { ++c.diff_elems; if (p < 0 || std::fabs(sc[p] - thr) > tol * (std::fabs(thr) + 1e-6)) ++c.far; }
    for (int p : B)
      if (!A.count(p)) { ++c.diff_elems; if (std::fabs(sc[p] - thr) > tol * (std::fabs(thr) + 1e-6)) ++c.far; }
  }
  return c;
}

void test_indexer(int S, int nd) {
  if (S <= kTopK) { std::printf("skip E: S %d <= 2048\n", S); return; }
  const int R = S - kTopK;  // rows that select
  const size_t ld = (S + 1023) / 1024 * 1024;
  Dev<uint16_t> iq = tu::to_dev<uint16_t>(L("iq")), kidx = tu::to_dev<uint16_t>(L("kidx"));
  Dev<float> w = tu::to_dev<float>(L("iw"));
  Dev<float> scores((size_t)R * ld);
  Dev<uint32_t> work(index_scratch_words(R));
  Dev<int> tk((size_t)R * kTopK);
  IdxCache ik;
  ik.k = reinterpret_cast<uint8_t*>(kidx.p);
  IdxQuery q{iq.p + (size_t)kTopK * kIdxHeads * kIdxDim, w.p + (size_t)kTopK * kIdxHeads};
  const int* pos = C->pos.p + kTopK;
  index_select(q, pos, nullptr, R, C->rope.p, ik, 0, 0, IdxScratch{scores.p, ld, work.p}, tk.p, nullptr, 0);
  sync();
  auto hs = scores.down();
  auto ht = tk.down();
  const float* rs = L("iscores").as<float>();
  double mx = 0;
  for (int i = 0; i < R; ++i) {
    double rmax = 0;
    for (int j = 0; j <= kTopK + i; ++j) rmax = std::max(rmax, (double)std::fabs(rs[(size_t)i * S + j]));
    for (int j = 0; j <= kTopK + i; ++j)
      mx = std::max(mx, std::fabs((double)hs[(size_t)i * ld + j] - rs[(size_t)i * S + j]) / (rmax + 1e-12));
  }
  T(mx < 1e-5, "E indexer scores (prefill, %d rows x up to %d keys) vs the reference: max |d| / row max %.2e", R, S, mx);
  auto c = cmp_topk(ht.data(), L("topk").as<int>(), rs, R, S, 1e-5);
  T(c.far == 0, "E top-2048 vs the reference's torch.topk: %d of %d rows identical, %d elements differ, %d not near-ties",
    c.equal_rows, c.rows, c.diff_elems, c.far);
  // the first-level histogram (the first 8192 words of each row's scratch) and the row's counters (words 8196-8198)
  // must be left zero; the other params, chunk segments and candidates are rewritten by every call
  auto wz = work.down();
  const size_t rw_words = wz.size() / R;
  bool hz = true;
  for (int r = 0; r < R; ++r) {
    for (size_t i = 0; i < 8192; ++i) hz = hz && wz[r * rw_words + i] == 0;
    for (size_t i = 8196; i < 8199; ++i) hz = hz && wz[r * rw_words + i] == 0;
  }
  T(hz, "E selection histograms and counters left zeroed");
  // the fallback (more candidates than the cap: the row selected from its scores directly) = the candidates' path
  for (int cap : {0, 2100}) {
    Dev<int> tf((size_t)R * kTopK);
    index_candidate_cap(cap);
    index_select(q, pos, nullptr, R, C->rope.p, ik, 0, 0, IdxScratch{scores.p, ld, work.p}, tf.p, nullptr, 0);
    sync();
    index_candidate_cap(1 << 30);
    T(tf.down() == ht, "E selection with a candidate cap of %d (rows over it select from their scores) = the default, bit for bit",
      cap);
  }
  // decode rows (1 row, 6 rows, 16 rows) = the prefill rows, bit for bit (scores and top-2048), with each of the
  // decode scoring kernel's key-load variants
  for (int rows : {1, 6, 16}) {
    if (rows > nd || rows > R) continue;
    const int a0 = R - rows;
    for (int ldv = 0; ldv < 5; ++ldv) {
      Dev<int> t2((size_t)rows * kTopK);
      IdxQuery q2{q.q + (size_t)a0 * kIdxHeads * kIdxDim, q.w + (size_t)a0 * kIdxHeads};
      index_score_config(ldv);
      index_select(q2, pos + a0, nullptr, rows, C->rope.p, ik, 0, 0, IdxScratch{scores.p, ld, work.p}, t2.p, nullptr, 0);
      sync();
      index_score_config(-1);
      auto h2 = t2.down();
      auto s2 = scores.down();
      bool se = true;
      for (int i = 0; i < rows; ++i) {
        const int n = kTopK + a0 + i + 1;  // row a0 + i's keys
        se = se && std::memcmp(s2.data() + (size_t)i * ld, hs.data() + (size_t)(a0 + i) * ld, (size_t)n * 4) == 0;
      }
      T(se && std::memcmp(h2.data(), ht.data() + (size_t)a0 * kTopK, h2.size() * 4) == 0,
        "E decode call of %d rows (key loads %d) = the prefill rows' scores and top-2048, bit for bit", rows, ldv);
    }
  }
  // chunking: rows in two calls
  {
    Dev<int> t3((size_t)R * kTopK);
    const int h = R / 3;
    index_select(q, pos, nullptr, h, C->rope.p, ik, 0, 0, IdxScratch{scores.p, ld, work.p}, t3.p, nullptr, 0);
    IdxQuery q3{q.q + (size_t)h * kIdxHeads * kIdxDim, q.w + (size_t)h * kIdxHeads};
    index_select(q3, pos + h, nullptr, R - h, C->rope.p, ik, 0, 0, IdxScratch{scores.p, ld, work.p}, t3.p + (size_t)h * kTopK,
                 nullptr, 0);
    sync();
    auto h3 = t3.down();
    T(h3 == ht, "E prefill in two chunks (%d + %d rows) = one chunk, bit for bit", h, R - h);
  }
  // position split: 4 parts, then the merge
  {
    const int rows = std::min(nd, R), a0 = R - rows;
    Dev<int> cp((size_t)rows * kTP * kTopK), out((size_t)rows * kTopK);
    Dev<uint32_t> ck((size_t)rows * kTP * kTopK);
    const int part = (S + kTP - 1) / kTP / 64 * 64 + 64;
    std::vector<int> hcp((size_t)rows * kTP * kTopK);
    std::vector<uint32_t> hck(hcp.size());
    Dev<int> tp((size_t)rows * kTopK);
    Dev<uint32_t> kp((size_t)rows * kTopK);
    IdxQuery q2{q.q + (size_t)a0 * kIdxHeads * kIdxDim, q.w + (size_t)a0 * kIdxHeads};
    for (int r = 0; r < kTP; ++r) {
      index_select(q2, pos + a0, nullptr, rows, C->rope.p, ik, r * part, (r + 1) * part, IdxScratch{scores.p, ld, work.p}, tp.p,
                   kp.p, 0);
      auto a = tp.down();
      auto b = kp.down();
      for (int i = 0; i < rows; ++i) {
        std::memcpy(&hcp[((size_t)i * kTP + r) * kTopK], &a[(size_t)i * kTopK], kTopK * 4);
        std::memcpy(&hck[((size_t)i * kTP + r) * kTopK], &b[(size_t)i * kTopK], kTopK * 4);
      }
    }
    cp.up(hcp);
    ck.up(hck);
    index_merge(cp.p, ck.p, pos + a0, rows, out.p, 0);
    sync();
    auto ho = out.down();
    T(std::memcmp(ho.data(), ht.data() + (size_t)a0 * kTopK, ho.size() * 4) == 0,
      "E position split (4 parts of %d positions) + merge = the replicated top-2048, %d rows, bit for bit", part, rows);
  }
  // FP8 index keys (written by attn_prep from the reference's projections)
  {
    auto hg = gathered_from_ref(S);
    Dev<uint16_t> g(hg), qr((size_t)S * kQLora), kv((size_t)S * kLat);
    Dev<uint8_t> k8((size_t)S * kIdxDim);
    Dev<float> s8(S);
    PrepArgs a;
    a.xg = g.p; a.pos = C->pos.p; a.rope = C->rope.p; a.q_norm = C->w_qn.p; a.kv_norm = C->w_kvn.p;
    a.k_norm_w = C->kn_w.p; a.k_norm_b = C->kn_b.p; a.qr = qr.p; a.kv.kv = kv.p;
    a.ik.k = k8.p; a.ik.ks = s8.p; a.ik.fp8 = true;
    attn_prep(a, S, 0);
    IdxCache i8;
    i8.k = k8.p; i8.ks = s8.p; i8.fp8 = true;
    Dev<int> t8((size_t)R * kTopK);
    index_select(q, pos, nullptr, R, C->rope.p, i8, 0, 0, IdxScratch{scores.p, ld, work.p}, t8.p, nullptr, 0);
    sync();
    auto h8 = t8.down();
    auto hs8 = scores.down();
    // FP8 decode rows (each key-load variant) = the FP8 prefill rows, bit for bit
    for (int rows : {1, 6}) {
      if (rows > nd || rows > R) continue;
      const int a0 = R - rows;
      IdxQuery q2{q.q + (size_t)a0 * kIdxHeads * kIdxDim, q.w + (size_t)a0 * kIdxHeads};
      for (int ldv = 0; ldv < 5; ++ldv) {
        Dev<int> t2((size_t)rows * kTopK);
        index_score_config(ldv);
        index_select(q2, pos + a0, nullptr, rows, C->rope.p, i8, 0, 0, IdxScratch{scores.p, ld, work.p}, t2.p, nullptr, 0);
        sync();
        index_score_config(-1);
        auto h2 = t2.down();
        auto s2 = scores.down();
        bool se = true;
        for (int i = 0; i < rows; ++i)
          se = se && std::memcmp(s2.data() + (size_t)i * ld, hs8.data() + (size_t)(a0 + i) * ld,
                                 (size_t)(kTopK + a0 + i + 1) * 4) == 0;
        T(se && std::memcmp(h2.data(), h8.data() + (size_t)a0 * kTopK, h2.size() * 4) == 0,
          "E FP8 keys: decode call of %d rows (key loads %d) = the prefill rows' scores and top-2048, bit for bit", rows,
          ldv);
      }
    }
    double ov = 0, mn = 1;
    for (int i = 0; i < R; ++i) {
      std::set<int> A(ht.begin() + (size_t)i * kTopK, ht.begin() + (size_t)(i + 1) * kTopK);
      int n = 0;
      for (int k = 0; k < kTopK; ++k) n += A.count(h8[(size_t)i * kTopK + k]);
      ov += n / (double)kTopK;
      mn = std::min(mn, n / (double)kTopK);
    }
    T(true, "E FP8 index keys (128 e4m3 + f32 amax/448): top-2048 overlap with the BF16 keys' mean %.4f, min %.4f (%d rows)",
      ov / R, mn, R);
  }
}

// ---- F: attention from the reference's q, cache and top-k; G: the whole chain on the GPU ------------------------
struct Layer {  // one rank-emulated layer's device state for S rows
  int S;
  Dev<uint16_t> cache;  // [S][576]
  Dev<uint16_t> kidx;   // [S][128]
  Dev<int> topk;        // [S][2048] (rows >= 2048)
  Dev<uint16_t> o;      // [S][64 * 256] attention output before o_proj
  Dev<float> out;       // [S][6144] o_proj summed over ranks
};

// attention for rows [a, b) of positions pos (q: [S][16384] all heads), cache, topk [S][2048]: o [S][16384]
void run_attention(const Dev<uint16_t>& q, const Dev<uint16_t>& cache, const Dev<int>& topk, int a, int b,
                   Dev<uint16_t>& o, int splits) {
  const int rows = b - a;
  Dev<uint16_t> qr((size_t)rows * kOProjK), ql((size_t)rows * kHeads * kKvLora), u((size_t)rows * kHeads * kKvLora),
      orr((size_t)rows * kOProjK);
  AttnScratch sc;
  Dev<float> so, sml;
  if (splits > 1) {
    so = Dev<float>((size_t)rows * splits * kHeads * kKvLora);
    sml = Dev<float>((size_t)rows * splits * kHeads * 2);
    sc = AttnScratch{so.p, sml.p, splits};
  }
  KvCache kv;
  kv.kv = cache.p;
  for (int r = 0; r < kTP; ++r) {
    CUDA_CHECK(cudaMemcpy2D(qr.p, kOProjK * 2, q.p + (size_t)a * kHeadsAll * kQkDim + r * kOProjK, kHeadsAll * kQkDim * 2,
                            kOProjK * 2, rows, cudaMemcpyDeviceToDevice));
    proj_rows(qr.p, kOProjK, rows, kNope, {seg(g_rw[r].uk, kHeads * kKvLora, ql.p, kHeads * kKvLora, kKvLora, kQkDim)},
              GemvOut::kBF16);
    sparse_attn(ql.p, qr.p, C->pos.p + a, nullptr, C->rope.p, kv, topk.p + (size_t)a * kTopK, rows, splits, sc, u.p, 0);
    proj_rows(u.p, kHeads * kKvLora, rows, kKvLora, {seg(g_rw[r].uv, kOProjK, orr.p, kOProjK, kVDim, kKvLora)},
              GemvOut::kBF16);
    CUDA_CHECK(cudaMemcpy2D(o.p + (size_t)a * kHeadsAll * kVDim + r * kOProjK, kHeadsAll * kVDim * 2, orr.p, kOProjK * 2,
                            kOProjK * 2, rows, cudaMemcpyDeviceToDevice));
  }
  sync();
}
// o_proj: the four ranks' f32 partials added in rank order, rounded to bf16 (host)
std::vector<uint16_t> run_oproj(const Dev<uint16_t>& o, int S) {
  Dev<float> part((size_t)S * kDim);
  Dev<uint16_t> orr((size_t)S * kOProjK);
  std::vector<float> acc((size_t)S * kDim, 0.f);
  for (int r = 0; r < kTP; ++r) {
    CUDA_CHECK(cudaMemcpy2D(orr.p, kOProjK * 2, o.p + r * kOProjK, kHeadsAll * kVDim * 2, kOProjK * 2, S, cudaMemcpyDeviceToDevice));
    proj_rows(orr.p, kOProjK, S, kOProjK, {seg(g_rw[r].o, kDim, part.p, kDim)}, GemvOut::kF32);
    auto h = part.down();
    for (size_t i = 0; i < acc.size(); ++i) acc[i] += h[i];
  }
  std::vector<uint16_t> out(acc.size());
  for (size_t i = 0; i < acc.size(); ++i) out[i] = tu::f2bf(acc[i]);
  return out;
}

// relative L2 per head (over rows) of a against the exact, aggregated: mean and max over heads
void head_err(const uint16_t* a, const float* ex, int S, double& mean, double& mx, bool a_is_bf16 = true) {
  mean = mx = 0;
  for (int h = 0; h < kHeadsAll; ++h) {
    const double e = tu::rel_l2(
        (size_t)S * kVDim, [&](size_t i) { return (double)tu::bf(a[(i / kVDim) * kHeadsAll * kVDim + h * kVDim + i % kVDim]); },
        [&](size_t i) { return (double)ex[(i / kVDim) * kHeadsAll * kVDim + h * kVDim + i % kVDim]; });
    mean += e / kHeadsAll;
    mx = std::max(mx, e);
  }
  (void)a_is_bf16;
}

void test_attention(int S, int nd) {
  Dev<uint16_t> q = tu::to_dev<uint16_t>(L("q"));
  // cache entries from the reference's c and k_rope; the reference's top-k at rows >= 2048
  std::vector<uint16_t> hc((size_t)S * kLat);
  const auto* c = L("c").as<uint16_t>();
  const auto* kr = L("kr").as<uint16_t>();
  for (int i = 0; i < S; ++i) {
    std::memcpy(&hc[(size_t)i * kLat], c + (size_t)i * kKvLora, kKvLora * 2);
    std::memcpy(&hc[(size_t)i * kLat + kKvLora], kr + (size_t)i * kRope, kRope * 2);
  }
  Dev<uint16_t> cache(hc);
  Dev<int> topk((size_t)S * kTopK);
  if (S > kTopK) CUDA_CHECK(cudaMemcpy(topk.p + (size_t)kTopK * kTopK, L("topk").as<int>(), (size_t)(S - kTopK) * kTopK * 4, cudaMemcpyHostToDevice));
  Dev<uint16_t> o1((size_t)S * kHeadsAll * kVDim), o32((size_t)S * kHeadsAll * kVDim);
  run_attention(q, cache, topk, 0, S, o1, 1);
  auto h1 = o1.down();
  const float* ex = L("o_exact").as<float>();
  const uint16_t* oref = L("o_ref").as<uint16_t>();
  double em, emx, rm, rmx;
  head_err(h1.data(), ex, S, em, emx);
  head_err(oref, ex, S, rm, rmx);
  T(em <= 2 * rm && emx <= 2 * rmx,
    "F attention output (absorbed, prefill, %d rows) vs f64 exact: rel L2 per head mean %.2e max %.2e; the reference's own "
    "mean %.2e max %.2e", S, em, emx, rm, rmx);
  // rows >= 2048 alone (the sparse rows)
  if (S > kTopK) {
    double a = 0, b = 0;
    const int R = S - kTopK;
    a = tu::rel_l2((size_t)R * kHeadsAll * kVDim, [&](size_t i) { return (double)tu::bf(h1[(size_t)kTopK * kHeadsAll * kVDim + i]); },
                   [&](size_t i) { return (double)ex[(size_t)kTopK * kHeadsAll * kVDim + i]; });
    b = tu::rel_l2((size_t)R * kHeadsAll * kVDim, [&](size_t i) { return (double)tu::bf(oref[(size_t)kTopK * kHeadsAll * kVDim + i]); },
                   [&](size_t i) { return (double)ex[(size_t)kTopK * kHeadsAll * kVDim + i]; });
    T(a <= 2 * b, "F sparse rows (p >= 2048, 2048 selected keys): rel L2 vs exact %.2e, the reference's %.2e", a, b);
  }
  // split-K (decode's 32 splits) agrees within the tolerance; decode rows = prefill rows bit for bit at equal splits
  run_attention(q, cache, topk, 0, S, o32, 32);
  auto h32 = o32.down();
  double e32m, e32x;
  head_err(h32.data(), ex, S, e32m, e32x);
  T(e32m <= 2 * rm, "F 32 splits (decode's split-K): rel L2 vs exact mean %.2e max %.2e", e32m, e32x);
  for (int rows : {1, 6, 16}) {
    if (rows > nd) continue;
    Dev<uint16_t> od((size_t)S * kHeadsAll * kVDim);
    for (int a = S - nd; a < S; a += rows) run_attention(q, cache, topk, a, std::min(S, a + rows), od, 32);
    auto hd = od.down();
    const size_t off = (size_t)(S - nd) * kHeadsAll * kVDim;
    T(std::memcmp(hd.data() + off, h32.data() + off, (size_t)nd * kHeadsAll * kVDim * 2) == 0,
      "F decode calls of %d rows (32 splits) = the prefill rows, bit for bit", rows);
  }
  // o_proj
  auto out = run_oproj(o1, S);
  const auto* oref2 = L("out_ref").as<uint16_t>();
  const double e = tu::rel_l2((size_t)S * kDim, [&](size_t i) { return (double)tu::bf(out[i]); },
                              [&](size_t i) { return (double)tu::bf(oref2[i]); });
  const double eo = tu::rel_l2((size_t)S * kHeadsAll * kVDim, [&](size_t i) { return (double)tu::bf(h1[i]); },
                               [&](size_t i) { return (double)tu::bf(oref[i]); });
  T(e <= 1.5 * eo + 1e-3, "F o_proj (4 ranks' partials, f32 sum) vs the reference's out: rel L2 %.2e (attention outputs differ by %.2e)",
    e, eo);
}

// G: the chain on the GPU from x_raw (rank-emulated); rows [a, b) of the sequence in one call each
struct Chain {
  int S;
  Dev<uint16_t> x, xg, qr, q, iq, cache, kidx, o;
  Dev<float> iw, scores;
  Dev<uint32_t> work;
  Dev<int> topk;
  size_t ld;
  explicit Chain(int S_)
      : S(S_), x((size_t)S_ * kDim), xg((size_t)S_ * kTP * kXRows), qr((size_t)S_ * kQLora), q((size_t)S_ * kHeadsAll * kQkDim),
        iq((size_t)S_ * kIdxHeads * kIdxDim), cache((size_t)S_ * kLat), kidx((size_t)S_ * kIdxDim),
        o((size_t)S_ * kHeadsAll * kVDim), iw((size_t)S_ * kIdxHeads), topk((size_t)S_ * kTopK) {
    ld = (S + 1023) / 1024 * 1024;
    const int R = std::max(1, S - kTopK);
    scores = Dev<float>((size_t)R * ld);
    work = Dev<uint32_t>(index_scratch_words(R));
  }
  void run(const Dev<uint16_t>& x_raw, int a, int b) {
    const int rows = b - a;
    rmsnorm(x_raw.p + (size_t)a * kDim, kDim, C->w_in.p, kNormEps, kDim, x.p + (size_t)a * kDim, kDim, rows, 0);
    for (int r = 0; r < kTP; ++r)
      proj_rows(x.p + (size_t)a * kDim, kDim, rows, kDim,
                {seg(g_rw[r].x, kXRows, xg.p + (size_t)a * kTP * kXRows + r * kXRows, kTP * kXRows)}, GemvOut::kBF16, 4);
    PrepArgs p;
    p.xg = xg.p + (size_t)a * kTP * kXRows;
    p.pos = C->pos.p + a;
    p.rope = C->rope.p;
    p.q_norm = C->w_qn.p; p.kv_norm = C->w_kvn.p; p.k_norm_w = C->kn_w.p; p.k_norm_b = C->kn_b.p;
    p.qr = qr.p + (size_t)a * kQLora;
    p.kv.kv = cache.p;
    p.ik.k = reinterpret_cast<uint8_t*>(kidx.p);
    attn_prep(p, rows, 0);
    for (int r = 0; r < kTP; ++r)
      proj_rows(qr.p + (size_t)a * kQLora, kQLora, rows, kQLora,
                {seg(g_rw[r].qb, kOProjK, q.p + (size_t)a * kHeadsAll * kQkDim + r * kOProjK, kHeadsAll * kQkDim),
                 seg(g_rw[r].iq, 1024, iq.p + (size_t)a * kIdxHeads * kIdxDim + r * 1024, kIdxHeads * kIdxDim)},
                GemvOut::kBF16);
    index_weights(x.p + (size_t)a * kDim, rows, C->wp.p, kIdxHeads, iw.p + (size_t)a * kIdxHeads, kIdxHeads, 0);
    const int s0 = std::max(a, kTopK);
    if (b > s0) {
      IdxCache ik;
      ik.k = reinterpret_cast<uint8_t*>(kidx.p);
      index_select(IdxQuery{iq.p + (size_t)s0 * kIdxHeads * kIdxDim, iw.p + (size_t)s0 * kIdxHeads}, C->pos.p + s0, nullptr, b - s0,
                   C->rope.p, ik, 0, 0, IdxScratch{scores.p, ld, work.p}, topk.p + (size_t)s0 * kTopK, nullptr, 0);
    }
    run_attention(q, cache, topk, a, b, o, 1);
  }
};

void test_chain(int S) {
  Dev<uint16_t> x_raw = tu::to_dev<uint16_t>(L("x_raw"));
  Chain one(S);
  one.run(x_raw, 0, S);
  auto out = run_oproj(one.o, S);
  const auto* oref = L("out_ref").as<uint16_t>();
  const double e = tu::rel_l2((size_t)S * kDim, [&](size_t i) { return (double)tu::bf(out[i]); },
                              [&](size_t i) { return (double)tu::bf(oref[i]); });
  // the reference's chain is bf16 torch (its attention alone is 2e-2 rel L2 from f64 exact, test F; ours 5e-3)
  T(e < 4e-2, "G the chain from the layer input (norm, projections, prep, indexer, attention, o_proj): out rel L2 vs the "
    "reference %.2e", e);
  if (S > kTopK) {
    auto ht = one.topk.down();
    auto c = cmp_topk(ht.data() + (size_t)kTopK * kTopK, L("topk").as<int>(), L("iscores").as<float>(), S - kTopK, S, 1e-2);
    T(true, "G chain top-2048 vs the reference's (own projections: not bitwise inputs): %d of %d rows identical, %d elements "
      "differ, %d beyond 1e-2 of the threshold", c.equal_rows, c.rows, c.diff_elems, c.far);
  }
  // chunked prefill (two chunks) = one chunk, bit for bit; and a second run = the first (determinism)
  Chain two(S);
  const int cut = S > kTopK ? kTopK + (S - kTopK) / 2 : S / 2;
  two.run(x_raw, 0, cut);
  two.run(x_raw, cut, S);
  auto eqd = [](const auto& A, const auto& B) { auto a = A.down(); auto b = B.down(); return a == b; };
  T(eqd(one.cache, two.cache) && eqd(one.kidx, two.kidx) && eqd(one.topk, two.topk) && eqd(one.o, two.o),
    "G prefill in two chunks (%d + %d rows) = one chunk: cache, index keys, top-k, attention output bit for bit", cut, S - cut);
  Chain again(S);
  again.run(x_raw, 0, S);
  T(eqd(one.cache, again.cache) && eqd(one.topk, again.topk) && eqd(one.o, again.o) && eqd(one.q, again.q),
    "G determinism: a second run is bit-identical");
}

// slot tables: rows of two sessions in one call (decode prep + indexer + attention) = each alone
void test_slots(int S) {
  if (S <= kTopK + 8) return;
  Chain one(S);
  Dev<uint16_t> x_raw = tu::to_dev<uint16_t>(L("x_raw"));
  one.run(x_raw, 0, S);
  sync();
  // two slots holding the same sequence; rows: slot 1 at S-2, S-1, then slot 0 at S-4, S-3
  const size_t cap = S;
  Dev<uint16_t> cache2(2 * cap * kLat), kidx2(2 * cap * kIdxDim);
  for (int s = 0; s < 2; ++s) {
    CUDA_CHECK(cudaMemcpy(cache2.p + s * cap * kLat, one.cache.p, cap * kLat * 2, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(kidx2.p + s * cap * kIdxDim, one.kidx.p, cap * kIdxDim * 2, cudaMemcpyDeviceToDevice));
  }
  const int rows_pos[4] = {S - 2, S - 1, S - 4, S - 3}, rows_slot[4] = {1, 1, 0, 0};
  Dev<int> pos(std::vector<int>(rows_pos, rows_pos + 4)), slot(std::vector<int>(rows_slot, rows_slot + 4));
  // gather the rows' inputs
  Dev<uint16_t> iq(4 * kIdxHeads * kIdxDim), q(4 * kHeadsAll * kQkDim);
  Dev<float> iw(4 * kIdxHeads);
  for (int i = 0; i < 4; ++i) {
    CUDA_CHECK(cudaMemcpy(iq.p + i * kIdxHeads * kIdxDim, one.iq.p + (size_t)rows_pos[i] * kIdxHeads * kIdxDim, kIdxHeads * kIdxDim * 2, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(iw.p + i * kIdxHeads, one.iw.p + (size_t)rows_pos[i] * kIdxHeads, kIdxHeads * 4, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(q.p + i * kHeadsAll * kQkDim, one.q.p + (size_t)rows_pos[i] * kHeadsAll * kQkDim, kHeadsAll * kQkDim * 2, cudaMemcpyDeviceToDevice));
  }
  IdxCache ik;
  ik.k = reinterpret_cast<uint8_t*>(kidx2.p);
  ik.slot_stride = cap;
  Dev<int> tk(4 * kTopK);
  Dev<float> sc(4 * one.ld);
  Dev<uint32_t> work(index_scratch_words(4));
  index_select(IdxQuery{iq.p, iw.p}, pos.p, slot.p, 4, C->rope.p, ik, 0, 0, IdxScratch{sc.p, one.ld, work.p}, tk.p, nullptr, 0);
  sync();
  auto ht = tk.down();
  auto hone = one.topk.down();
  bool ok = true;
  for (int i = 0; i < 4; ++i) ok &= std::memcmp(&ht[i * kTopK], &hone[(size_t)rows_pos[i] * kTopK], kTopK * 4) == 0;
  T(ok, "S indexer with a slot table (2 sessions, rows 2 + 2 in one call) = each row alone, bit for bit");
  // attention, rank 0, 32 splits
  KvCache kv;
  kv.kv = cache2.p;
  kv.slot_stride = cap;
  Dev<uint16_t> qr(4 * kOProjK), ql(4 * kHeads * kKvLora), u(4 * kHeads * kKvLora), u1(kHeads * kKvLora);
  CUDA_CHECK(cudaMemcpy2D(qr.p, kOProjK * 2, q.p, kHeadsAll * kQkDim * 2, kOProjK * 2, 4, cudaMemcpyDeviceToDevice));
  GemvSeg s = seg(g_rw[0].uk, kHeads * kKvLora, ql.p, kHeads * kKvLora, kKvLora, kQkDim);
  gemv(true, qr.p, kOProjK, 4, kNope, &s, 1, GemvOut::kBF16, 1, 0);
  Dev<float> so(4 * 32 * kHeads * kKvLora), sml(4 * 32 * kHeads * 2);
  sparse_attn(ql.p, qr.p, pos.p, slot.p, C->rope.p, kv, tk.p, 4, 32, AttnScratch{so.p, sml.p, 32}, u.p, 0);
  auto hu = u.down();
  ok = true;
  KvCache kv0;
  kv0.kv = one.cache.p;
  for (int i = 0; i < 4; ++i) {
    sparse_attn(ql.p + i * kHeads * kKvLora, qr.p + i * kOProjK, pos.p + i, nullptr, C->rope.p, kv0, tk.p + i * kTopK, 1, 32,
                AttnScratch{so.p, sml.p, 32}, u1.p, 0);
    auto h1 = u1.down();
    ok &= std::memcmp(h1.data(), &hu[i * kHeads * kKvLora], kHeads * kKvLora * 2) == 0;
  }
  T(ok, "S attention with a slot table (2 sessions in one call) = each row alone, bit for bit");
  // cache writes with a slot table: rows of slot 1 and slot 0 into a zeroed two-slot cache = the one-session entries
  {
    Dev<uint16_t> c2(2 * cap * kLat), k2(2 * cap * kIdxDim), qr(4 * kQLora);
    Dev<uint16_t> xg(4 * kTP * kXRows);
    for (int i = 0; i < 4; ++i)
      CUDA_CHECK(cudaMemcpy(xg.p + i * kTP * kXRows, one.xg.p + (size_t)rows_pos[i] * kTP * kXRows, kTP * kXRows * 2,
                            cudaMemcpyDeviceToDevice));
    PrepArgs p;
    p.xg = xg.p; p.pos = pos.p; p.slot = slot.p; p.rope = C->rope.p;
    p.q_norm = C->w_qn.p; p.kv_norm = C->w_kvn.p; p.k_norm_w = C->kn_w.p; p.k_norm_b = C->kn_b.p;
    p.qr = qr.p; p.kv.kv = c2.p; p.kv.slot_stride = cap; p.ik.k = reinterpret_cast<uint8_t*>(k2.p); p.ik.slot_stride = cap;
    attn_prep(p, 4, 0);
    sync();
    auto hc = c2.down();
    auto hk = k2.down();
    auto oc = one.cache.down();
    auto ok2 = one.kidx.down();
    bool okp = true;
    for (int i = 0; i < 4; ++i) {
      const size_t e = (size_t)rows_slot[i] * cap + rows_pos[i];
      okp &= std::memcmp(&hc[e * kLat], &oc[(size_t)rows_pos[i] * kLat], kLat * 2) == 0;
      okp &= std::memcmp(&hk[e * kIdxDim], &ok2[(size_t)rows_pos[i] * kIdxDim], kIdxDim * 2) == 0;
    }
    T(okp, "S cache entries and index keys written through a slot table = the one-session entries, bit for bit");
  }
}

// ---- H: dense MLP, shared expert, head --------------------------------------------------------------------
void test_mlp(int S, int nd) {
  if (!has("dense_y")) { std::printf("skip H: no MLP data\n"); return; }
  Dev<uint16_t> x(nd * kDim);
  x.up(L("x").as<uint16_t>() + (size_t)(S - nd) * kDim, (size_t)nd * kDim);
  for (auto [nm, inter] : {std::pair<const char*, int>{"dense", kDenseInterRank}, {"shared", kSharedInterRank}}) {
    std::vector<float> acc((size_t)nd * kDim, 0.f);
    std::vector<uint16_t> hh((size_t)nd * inter * kTP);
    Dev<uint16_t> h((size_t)nd * inter);
    Dev<float> y((size_t)nd * kDim);
    for (int r = 0; r < kTP; ++r) {
      const std::string p = "r" + std::to_string(r) + "_" + nm;
      FP8W gu = fp8w(p + "_gu", 128, 128), dn = fp8w(p + "_down", 128, 128);
      GemvSeg s = seg(gu, 2 * inter, h.p, inter);
      gemv(true, x.p, kDim, nd, kDim, &s, 1, GemvOut::kSwiGLU, inter >= 2048 ? 1 : 4, 0);
      GemvSeg d = seg(dn, kDim, y.p, kDim);
      gemv(true, h.p, inter, nd, inter, &d, 1, GemvOut::kF32, 1, 0);
      auto hy = y.down();
      auto hr = h.down();
      for (size_t i = 0; i < acc.size(); ++i) acc[i] += hy[i];
      for (int i = 0; i < nd; ++i) std::memcpy(&hh[(size_t)i * inter * kTP + r * inter], &hr[(size_t)i * inter], inter * 2);
    }
    std::vector<uint16_t> out(acc.size());
    for (size_t i = 0; i < acc.size(); ++i) out[i] = tu::f2bf(acc[i]);
    char w[160];
    std::snprintf(w, sizeof w, "H %s gate/up (FP8, interleaved, SwiGLU epilogue) at %d rows", nm, nd);
    check_bf16(w, hh.data(), L(std::string(nm) + "_h").as<uint16_t>(), nd, (size_t)inter * kTP, (size_t)inter * kTP, (size_t)inter * kTP, 2e-3);
    // the chained output (our h in): h's 1-ulp differences propagate through 4 x inter terms, so a rel L2
    const uint16_t* yref = L(std::string(nm) + "_y").as<uint16_t>();
    const double e = tu::rel_l2(out.size(), [&](size_t i) { return (double)tu::bf(out[i]); },
                                [&](size_t i) { return (double)tu::bf(yref[i]); });
    T(e < 1e-2, "H %s MLP chained (our gate/up then down): out rel L2 vs the reference %.2e", nm, e);
    // the down GEMV alone: the reference's h in
    std::fill(acc.begin(), acc.end(), 0.f);
    const uint16_t* href = L(std::string(nm) + "_h").as<uint16_t>();
    for (int r = 0; r < kTP; ++r) {
      std::vector<uint16_t> hs((size_t)nd * inter);
      for (int i = 0; i < nd; ++i) std::memcpy(&hs[(size_t)i * inter], href + (size_t)i * inter * kTP + r * inter, inter * 2);
      h.up(hs.data(), hs.size());
      FP8W dn = fp8w("r" + std::to_string(r) + "_" + nm + "_down", 128, 128);
      GemvSeg d = seg(dn, kDim, y.p, kDim);
      gemv(true, h.p, inter, nd, inter, &d, 1, GemvOut::kF32, 1, 0);
      auto hy = y.down();
      for (size_t i = 0; i < acc.size(); ++i) acc[i] += hy[i];
    }
    for (size_t i = 0; i < acc.size(); ++i) out[i] = tu::f2bf(acc[i]);
    std::snprintf(w, sizeof w, "H %s down alone (the reference's h; 4 ranks' f32 partials)", nm);
    check_bf16(w, out.data(), yref, nd, kDim, kDim, kDim, 2e-3);
  }
  Dev<uint16_t> head = tu::to_dev<uint16_t>(L("head_r0"));
  Dev<uint16_t> lg((size_t)nd * kVocabRank);
  GemvSeg s;
  s.w = head.p; s.n = kVocabRank; s.y = lg.p; s.ldy = kVocabRank;
  gemv(false, x.p, kDim, nd, kDim, &s, 1, GemvOut::kBF16, 1, 0);
  auto hl = lg.down();
  check_bf16("H head (BF16 GEMV, rank 0's 38,720 rows)", hl.data(), L("logits_r0").as<uint16_t>(), nd, kVocabRank, kVocabRank, kVocabRank);
}

// ---- P: the prefill GEMM = the 16-row GEMV, bit for bit, every projection and tile configuration --------------
void test_gemm(int S) {
  struct Case {
    std::string name;
    const uint16_t* x;
    int ldx, K;
    std::vector<GemvSeg> segs;  // y set below
    std::vector<size_t> ysz;    // elements a row
    GemvOut out;
    int splitk;
  };
  Dev<uint16_t> x = tu::to_dev<uint16_t>(L("x")), q = tu::to_dev<uint16_t>(L("q")), qr = tu::to_dev<uint16_t>(L("qr"));
  // W_UK / W_UV / o_proj inputs: rank 0's 16 heads of q; a latent-like input (q's values, 512 a head); o rows
  Dev<uint16_t> q0((size_t)S * kOProjK), lat((size_t)S * kHeads * kKvLora);
  CUDA_CHECK(cudaMemcpy2D(q0.p, kOProjK * 2, q.p, kHeadsAll * kQkDim * 2, kOProjK * 2, S, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2D(lat.p, kHeads * kKvLora * 2, q.p, kHeadsAll * kQkDim * 2, kHeads * kKvLora * 2, S,
                          cudaMemcpyDeviceToDevice));
  std::vector<Case> cs;
  std::vector<FP8W> keep;
  for (int r = 0; r < kTP; r += 3) {
    const std::string rn = " rank " + std::to_string(r);
    cs.push_back({"x-projection 688x6144 split-K 4" + rn, x.p, kDim, kDim, {seg(g_rw[r].x, kXRows, nullptr, kXRows)}, {kXRows}, GemvOut::kBF16, 4});
    cs.push_back({"x-projection, shared-layer rows 656" + rn, x.p, kDim, kDim, {seg(g_rw[r].x, kXRowsShared, nullptr, kXRowsShared)},
                  {kXRowsShared}, GemvOut::kBF16, 4});
    cs.push_back({"q_b + wq_b (two segments)" + rn, qr.p, kQLora, kQLora,
                  {seg(g_rw[r].qb, kOProjK, nullptr, kOProjK), seg(g_rw[r].iq, 1024, nullptr, 1024)}, {kOProjK, 1024}, GemvOut::kBF16, 1});
    cs.push_back({"W_UK^T grouped" + rn, q0.p, kOProjK, kNope, {seg(g_rw[r].uk, kHeads * kKvLora, nullptr, kHeads * kKvLora, kKvLora, kQkDim)},
                  {(size_t)kHeads * kKvLora}, GemvOut::kBF16, 1});
    cs.push_back({"W_UV grouped" + rn, lat.p, kHeads * kKvLora, kKvLora, {seg(g_rw[r].uv, kOProjK, nullptr, kOProjK, kVDim, kKvLora)},
                  {kOProjK}, GemvOut::kBF16, 1});
    cs.push_back({"o_proj (f32)" + rn, q0.p, kOProjK, kOProjK, {seg(g_rw[r].o, kDim, nullptr, kDim)}, {kDim}, GemvOut::kF32, 1});
  }
  if (has("dense_y"))
    for (auto [nm, inter] : {std::pair<const char*, int>{"dense", kDenseInterRank}, {"shared", kSharedInterRank}}) {
      keep.push_back(fp8w(std::string("r0_") + nm + "_gu", 128, 128));
      keep.push_back(fp8w(std::string("r0_") + nm + "_down", 128, 128));
      const FP8W& gu = keep[keep.size() - 2];
      const FP8W& dn = keep.back();
      const int sk = inter >= 2048 ? 1 : 4;
      cs.push_back({std::string(nm) + " gate/up SwiGLU split-K " + std::to_string(sk), x.p, kDim, kDim, {seg(gu, 2 * inter, nullptr, inter)},
                    {(size_t)inter}, GemvOut::kSwiGLU, sk});
      cs.push_back({std::string(nm) + " down (f32)", x.p, kDim, inter, {seg(dn, kDim, nullptr, kDim)}, {kDim}, GemvOut::kF32, 1});
    }
  for (auto& c : cs) {
    const size_t es = c.out == GemvOut::kF32 ? 4 : 2;
    std::vector<Dev<uint8_t>> ya, yb;
    for (size_t i = 0; i < c.segs.size(); ++i) {
      ya.emplace_back((size_t)S * c.ysz[i] * es);
      yb.emplace_back((size_t)S * c.ysz[i] * es);
    }
    std::vector<GemvSeg> sa = c.segs, sb = c.segs;
    for (size_t i = 0; i < c.segs.size(); ++i) { sa[i].y = ya[i].p; sb[i].y = yb[i].p; }
    gemv_rows(true, c.x, c.ldx, S, c.K, sa, c.out, c.splitk);
    sync();
    std::vector<std::vector<uint8_t>> ref;
    for (auto& y : ya) ref.push_back(y.down());
    bool ok = true;
    std::string which;
    for (int cfg = 0; cfg < 6; ++cfg) {
      for (auto& y : yb) y.zero();
      gemm_config(cfg);
      gemm(c.x, c.ldx, S, c.K, sb.data(), (int)sb.size(), c.out, c.splitk, 0);
      sync();
      bool e = true;
      for (size_t i = 0; i < yb.size(); ++i) e = e && yb[i].down() == ref[i];
      if (!e) which += " " + std::to_string(cfg);
      ok = ok && e;
    }
    gemm_config(-1);
    T(ok, "P GEMM %s, %d rows: = the 16-row GEMV bit for bit in tile configurations 0-5%s%s", c.name.c_str(), S,
      ok ? "" : "; differ in", which.c_str());
  }
  // BF16 weights (gemm_bf16 = the BF16 GEMV): rank 0's head rows, whole and in the router's shape (256 rows, split-K 4)
  if (has("head_r0")) {
    Dev<uint16_t> head = tu::to_dev<uint16_t>(L("head_r0"));
    struct B {
      const char* name;
      int n, splitk;
      GemvOut out;
    };
    for (const B& b : {B{"router shape 256x6144 split-K 4 (f32)", 256, 4, GemvOut::kF32},
                       B{"head 38720x6144 (f32)", kVocabRank, 1, GemvOut::kF32},
                       B{"1000x6144 split-K 2 (bf16 out)", 1000, 2, GemvOut::kBF16}}) {
      const size_t es = b.out == GemvOut::kF32 ? 4 : 2;
      Dev<uint8_t> ya((size_t)S * b.n * es), yb((size_t)S * b.n * es);
      GemvSeg g;
      g.w = head.p, g.n = b.n, g.y = ya.p, g.ldy = b.n;
      gemv_rows(false, x.p, kDim, S, kDim, {g}, b.out, b.splitk);
      sync();
      const std::vector<uint8_t> ref = ya.down();
      g.y = yb.p;
      gemm_bf16(x.p, kDim, S, kDim, &g, 1, b.out, b.splitk, 0);
      sync();
      T(yb.down() == ref, "P GEMM BF16 %s, %d rows: = the 16-row BF16 GEMV bit for bit", b.name, S);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) die("usage: attn_tests DIR [--only ABCPEFGHS]");
  g_dir = argv[1];
  std::string only = argc > 3 && std::string(argv[2]) == "--only" ? argv[3] : "ACBPEFGSH";
  const int S = manifest_int("S"), nd = manifest_int("decode");
  std::printf("case %s: S %d, decode rows %d\n", g_dir.c_str(), S, nd);
  load_weights();
  Common c{tu::to_dev<uint16_t>(L("w_in")), tu::to_dev<uint16_t>(L("w_qn")), tu::to_dev<uint16_t>(L("w_kvn")),
           tu::to_dev<uint16_t>(L("kn_w")), tu::to_dev<uint16_t>(L("kn_b")), tu::to_dev<uint16_t>(L("wp")),
           tu::to_dev<uint32_t>(L("rope")), Dev<int>()};
  std::vector<int> hp(S + 64);
  for (size_t i = 0; i < hp.size(); ++i) hp[i] = (int)i;
  c.pos = Dev<int>(hp);
  C = &c;
  for (char t : only) {
    switch (t) {
      case 'A': test_dequant(); break;
      case 'B': test_gemvs(S, nd); break;
      case 'C': test_prep(S); break;
      case 'E': test_indexer(S, nd); break;
      case 'F': test_attention(S, nd); break;
      case 'G': test_chain(S); break;
      case 'S': test_slots(S); break;
      case 'H': test_mlp(S, nd); break;
      case 'P': test_gemm(S); break;
    }
  }
  std::printf("%d passed, %d failed\n", T.pass, T.fail);
  return T.fail ? 1 : 0;
}
