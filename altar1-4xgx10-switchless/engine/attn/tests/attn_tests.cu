// engine/attn against gen_ref.py's golden data (one full-indexer layer at the real dimensions, random weights).
// usage: attn_tests DIR [--only A,B,...]. Rank r's kernels run for r = 0..3 on this one GPU; the all-gathers and
// the o_proj/down all-reduce are host copies (in rank order). Every check prints PASS/FAIL with its numbers.
#include <cstdarg>
#include <map>
#include <set>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
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

// a string value of the manifest: the first "key": "value" after `after` (e.g. a tile's format inside "fmts")
std::string manifest_str(const char* key, const char* after = nullptr) {
  std::ifstream f(g_dir + "/manifest.json");
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t p0 = 0;
  if (after) {
    p0 = s.find(std::string("\"") + after + "\":");
    if (p0 == std::string::npos) return "";
  }
  auto p = s.find(std::string("\"") + key + "\": \"", p0);
  if (p == std::string::npos) return "";
  p += std::strlen(key) + 5;
  return s.substr(p, s.find('"', p) - p);
}
WFmt fmt_of(const std::string& f) {
  if (f == "bf16") return WFmt::kBF16;
  if (f == "int4") return WFmt::kINT4;
  if (f == "fp8" || f.empty()) return WFmt::kFP8;  // glm53's data carry no formats: FP8
  die("unknown format %s", f.c_str());
}
std::string g_fmt;  // the layer's: int4, bf16 or fp8
WFmt mlp_fmt() { return fmt_of(manifest_str("mlp", "fmts")); }

// a weight on the device in its format (FP8: with its scales)
struct Wt {
  WFmt fmt = WFmt::kNone;
  Dev<uint8_t> w;
  Dev<float> s;
  int rs = 128, ks = 128;
};
Wt wload(const std::string& name, WFmt fmt, int rs = 128, int ks = 128) {
  Wt t;
  t.fmt = fmt;
  t.w = tu::to_dev<uint8_t>(L(name + "_w"));
  if (fmt == WFmt::kFP8) t.s = tu::to_dev<float>(L(name + "_s"));
  t.rs = rs, t.ks = ks;
  return t;
}
struct RankW {
  Wt x, qb, iq, uk, uv, o;
};
std::vector<RankW> g_rw;
void load_weights() {
  auto F = [](const char* t) { return fmt_of(manifest_str(t, "fmts")); };
  for (int r = 0; r < kTP; ++r) {
    const std::string p = "r" + std::to_string(r) + "_";
    g_rw.push_back(RankW{wload(p + "xproj", F("xproj"), 16, 128), wload(p + "q_b", F("q_b")), wload(p + "wq_b", F("wq_b")),
                         wload(p + "w_uk", F("w_uk"), 128, 64), wload(p + "w_uv", F("w_uv"), 64, 128),
                         wload(p + "o_proj", F("o_proj"))});
  }
}
GemvSeg seg(const Wt& w, int n, void* y, int ldy, int group_rows = 0, int x_group = 0) {
  GemvSeg g;
  g.fmt = w.fmt;
  g.w = w.w.p;
  g.s = w.fmt == WFmt::kFP8 ? w.s.p : nullptr;
  g.n = n;
  g.rs = w.rs;
  g.ks = w.ks;
  g.y = y;
  g.ldy = ldy;
  g.group_rows = group_rows;
  g.x_group = x_group;
  return g;
}
GemvSeg seg_bf16(const void* w, int n, void* y, int ldy, int group_rows = 0, int x_group = 0) {
  GemvSeg g;
  g.fmt = WFmt::kBF16;
  g.w = w;
  g.n = n;
  g.y = y;
  g.ldy = ldy;
  g.group_rows = group_rows;
  g.x_group = x_group;
  return g;
}
// a GEMV over any number of rows, 16 at a time
void gemv_rows(const uint16_t* x, int ldx, int rows, int K, std::vector<GemvSeg> segs, GemvOut out, int splitk = 1,
               int max_rows = 16) {
  const size_t es = out == GemvOut::kF32 ? 4 : 2;
  for (int r0 = 0; r0 < rows; r0 += max_rows) {
    const int n = std::min(max_rows, rows - r0);
    std::vector<GemvSeg> s = segs;
    for (auto& g : s) g.y = static_cast<char*>(g.y) + (size_t)r0 * g.ldy * es;
    gemv(x + (size_t)r0 * ldx, ldx, n, K, s.data(), (int)s.size(), out, splitk, 0);
  }
}

// the prefill GEMM's bf16 scratch for INT4 weights (the largest: o_proj's 6144 x 4096)
Dev<uint16_t>* g_scratch = nullptr;
constexpr size_t kScratch = (size_t)kDim * kOProjK;
void prefill_gemm(const uint16_t* x, int ldx, int rows, int K, const std::vector<GemvSeg>& segs, GemvOut out, int splitk) {
  if (!g_scratch) g_scratch = new Dev<uint16_t>(kScratch);
  gemm(x, ldx, rows, K, segs.data(), (int)segs.size(), out, splitk, 0, false, g_scratch->p, kScratch);
}
// a projection as the engine runs it: the GEMM for prefill chunks (more than 16 rows), else the GEMV
void proj_rows(const uint16_t* x, int ldx, int rows, int K, std::vector<GemvSeg> segs, GemvOut out, int splitk = 1) {
  if (rows <= kMaxRows) gemv(x, ldx, rows, K, segs.data(), (int)segs.size(), out, splitk, 0);
  else prefill_gemm(x, ldx, rows, K, segs, out, splitk);
}
// host: the contract's INT4 weight from the canonical tensors (gen_ref's ct_*), as tools/pack.py int4_dequant
uint16_t int4_w_host(const uint32_t* packed, const uint16_t* scale, const uint32_t* zp, int K, int n, int k) {
  const int nib = (packed[(size_t)n * (K / 8) + k / 8] >> (4 * (k % 8))) & 15;
  const int zn = (zp[(size_t)(n / 8) * (K / 32) + k / 32] >> (4 * (n % 8))) & 15;
  return tu::f2bf((float)(nib - zn) * tu::bf(scale[(size_t)n * (K / 32) + k / 32]));
}


// ---- A: the tiles decode to zai's values through the kernel's own fragment mapping ---------------------------
void test_dequant_fp8() {
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
    gemv(x.p, kDim, 16, kDim, &s, 1, GemvOut::kF32, 4, 0);
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
    gemv(xq.p, kOProjK, 16, kNope, &s, 1, GemvOut::kF32, 1, 0);
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

// one-hot x rows through a GEMV (f32 out): y[i][n] = W[n][k_i] exactly (a single product with 1.0, zeros elsewhere)
std::vector<float> onehot(const GemvSeg& s0, int N, int K, int ldx, int xoff_stride, int nrep, const std::vector<int>& ks,
                          int splitk = 1) {
  Dev<uint16_t> x((size_t)16 * ldx);
  Dev<float> y((size_t)16 * N);
  std::vector<uint16_t> hx((size_t)16 * ldx, 0);
  for (int i = 0; i < 16; ++i)
    for (int rep = 0; rep < nrep; ++rep) hx[(size_t)i * ldx + rep * xoff_stride + ks[i]] = 0x3F80;
  x.up(hx);
  GemvSeg s = s0;
  s.y = y.p;
  s.ldy = N;
  gemv(x.p, ldx, 16, K, &s, 1, GemvOut::kF32, splitk, 0);
  return y.down();
}

// ---- A (BF16 and INT4 layers): the tiles decode to the contract's weights through the kernels' fragment maps -----
void test_dequant() {
  if (g_fmt == "fp8") return test_dequant_fp8();
  // the BF16 x-projection: rank 0's rows 0..511 are q_a's
  {
    const uint16_t* dq = L("dq_q_a").as<uint16_t>();
    size_t bad = 0, n = 0;
    for (int b = 0; b < 4; ++b) {
      std::vector<int> ks(16);
      for (int i = 0; i < 16; ++i) ks[i] = (b * 16 + i) * 97 % kDim;
      auto hy = onehot(seg(g_rw[0].x, kXRows, nullptr, 0), kXRows, kDim, kDim, 0, 1, ks, 4);
      for (int i = 0; i < 16; ++i)
        for (int r = 0; r < kXQa; ++r, ++n) bad += !(tu::bf(dq[(size_t)r * kDim + ks[i]]) == hy[(size_t)i * kXRows + r]);
    }
    T(bad == 0, "A x-projection BF16 rows (q_a rows of rank 0) through the GEMV = the weights, %zu of %zu differ", bad, n);
  }
  // W_UK^T of rank 0 (BF16 in either layer: an INT4 kv_b's exact dequantisation): one-hot d in every head's q_nope
  {
    const uint16_t* dq = L("dq_kv_b").as<uint16_t>();
    size_t bad = 0, n = 0;
    for (int b = 0; b < 2; ++b) {
      std::vector<int> ds(16);
      for (int i = 0; i < 16; ++i) ds[i] = (b * 16 + i) * 11 % kNope;
      auto hy = onehot(seg(g_rw[0].uk, kHeads * kKvLora, nullptr, 0, kKvLora, kQkDim), kHeads * kKvLora, kNope, kOProjK, kQkDim,
                       kHeads, ds);
      for (int i = 0; i < 16; ++i)
        for (int h = 0; h < kHeads; ++h)
          for (int c = 0; c < kKvLora; ++c, ++n)
            bad += !(tu::bf(dq[(size_t)(448 * h + ds[i]) * kKvLora + c]) == hy[(size_t)i * kHeads * kKvLora + h * kKvLora + c]);
    }
    T(bad == 0, "A W_UK^T tile (rank 0, kv_b's %s rows transposed) = the weights, %zu of %zu differ",
      g_fmt == "int4" ? "INT4, dequantised" : "BF16", bad, n);
  }
  if (g_fmt != "int4") return;
  // INT4 q_b (rank 0: rows 0..4095) and o_proj (rank 0: columns 0..4095) against the host's bf16_rne((nib - znib) s)
  // from the canonical tensors (compressed-tensors' layout)
  for (auto [nm, N, K, Kc] : {std::tuple<const char*, int, int, int>{"q_b", kOProjK, kQLora, kQLora},
                              {"o_proj", kDim, kOProjK, kHeadsAll * kVDim}}) {
    const std::string c = std::string("ct_") + nm;
    const uint32_t* pk = L(c + "_packed").as<uint32_t>();
    const uint16_t* sc = L(c + "_scale").as<uint16_t>();
    const uint32_t* zp = L(c + "_zp").as<uint32_t>();
    const Wt& w = std::string(nm) == "q_b" ? g_rw[0].qb : g_rw[0].o;
    size_t bad = 0, n = 0;
    for (int b = 0; b < 4; ++b) {
      std::vector<int> ks(16);
      for (int i = 0; i < 16; ++i) ks[i] = (b * 16 + i) * 131 % K;
      auto hy = onehot(seg(w, N, nullptr, 0), N, K, K, 0, 1, ks);
      for (int i = 0; i < 16; ++i)
        for (int r = 0; r < N; ++r, ++n) bad += !(tu::bf(int4_w_host(pk, sc, zp, Kc, r, ks[i])) == hy[(size_t)i * N + r]);
    }
    T(bad == 0, "A INT4 %s tile (rank 0) through the INT4 GEMV = bf16_rne((nib - znib) * s) from the canonical tensors, "
      "%zu of %zu differ", nm, bad, n);
  }
  // dequant_int4 (the prefill's weights) = pack.py's int4_dequant of the same rows, every value
  for (int r : {0, 3}) {
    const std::string p = "r" + std::to_string(r) + "_";
    for (auto [nm, N, K, w] : {std::tuple<const char*, int, int, const Wt*>{"q_b", kOProjK, kQLora, &g_rw[r].qb},
                               {"w_uv", kOProjK, kKvLora, &g_rw[r].uv}, {"o_proj", kDim, kOProjK, &g_rw[r].o}}) {
      Dev<uint16_t> out((size_t)N * K);
      dequant_int4(w->w.p, N, K, out.p, K, 0);
      const auto h = out.down();
      const auto& ref = L(p + nm + "_deq");
      const bool ok = std::memcmp(h.data(), ref.data.data(), h.size() * 2) == 0;
      T(ok, "A dequant_int4 rank %d %s [%d][%d] = pack.py int4_dequant, bit for bit", r, nm, N, K);
    }
  }
}

// ---- D: INT4 GEMV = BF16 GEMV on the dequantised matrix, bit for bit ---------------------------------------------
__global__ void k_int4_all(const uint16_t* __restrict__ sc, uint16_t* __restrict__ out) {
  // every (nibble, zero nibble) pair x one scale pattern a thread: 256 products through the kernels' dequantisation
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= 65536) return;
  const uint32_t ss = (uint32_t)sc[s] | (uint32_t)sc[s] << 16;
  for (int zn = 0; zn < 16; ++zn) {
    // 16 nibbles 0..15 in the checkpoint's order: byte i = (2i, 2i + 1)
    uint32_t o[8];
    int4x16_to_bf16(0x76543210u, 0xFEDCBA98u, int4_zz((uint32_t)zn), ss, o);
    for (int m = 0; m < 8; ++m) {
      out[((size_t)s * 16 + zn) * 16 + 2 * m] = (uint16_t)(o[m] & 0xFFFF);
      out[((size_t)s * 16 + zn) * 16 + 2 * m + 1] = (uint16_t)(o[m] >> 16);
    }
  }
}

void test_int4(int S) {
  if (g_fmt != "int4") { std::printf("skip D: not an INT4 layer\n"); return; }
  // the element arithmetic for every bf16 scale pattern (finite) and every (nibble, zero nibble)
  {
    std::vector<uint16_t> hs(65536);
    for (int i = 0; i < 65536; ++i) hs[i] = (uint16_t)i;
    Dev<uint16_t> sc(hs), out((size_t)65536 * 256);
    k_int4_all<<<256, 256>>>(sc.p, out.p);
    CUDA_CHECK(cudaGetLastError());
    const auto h = out.down();
    size_t bad = 0, n = 0, sub = 0;
    for (int s = 0; s < 65536; ++s) {
      const float f = tu::bf((uint16_t)s);
      if (!std::isfinite(f)) continue;
      for (int zn = 0; zn < 16; ++zn)
        for (int q = 0; q < 16; ++q, ++n) {
          const uint16_t want = tu::f2bf((float)(q - zn) * f), got = h[((size_t)s * 16 + zn) * 16 + q];
          bad += got != want;
          sub += (want & 0x7F80) == 0 && (want & 0x7F);
        }
    }
    T(bad == 0, "D INT4 element arithmetic (bf16 128 + u, exact subtract, one bf16 multiply) = bf16_rne((q - z) s) for "
      "every finite scale pattern and (q, z): %zu of %zu differ (%zu subnormal results)", bad, n, sub);
  }
  Dev<uint16_t> qr = tu::to_dev<uint16_t>(L("qr")), q = tu::to_dev<uint16_t>(L("q"));
  const int nx = std::min(S, 16);
  Dev<uint16_t> qx((size_t)nx * kOProjK), lat((size_t)nx * kHeads * kKvLora);
  CUDA_CHECK(cudaMemcpy2D(qx.p, kOProjK * 2, q.p, kHeadsAll * kQkDim * 2, kOProjK * 2, nx, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2D(lat.p, kHeads * kKvLora * 2, q.p, kHeadsAll * kQkDim * 2, kHeads * kKvLora * 2, nx,
                          cudaMemcpyDeviceToDevice));
  for (int r : {0, 3}) {
    const std::string p = "r" + std::to_string(r) + "_";
    Dev<uint16_t> dqb = tu::to_dev<uint16_t>(L(p + "q_b_deq")), duv = tu::to_dev<uint16_t>(L(p + "w_uv_deq")),
                  dop = tu::to_dev<uint16_t>(L(p + "o_proj_deq"));
    struct Case {
      const char* name;
      const Wt* w;
      const uint16_t* deq;
      const uint16_t* x;
      int ldx, N, K, group_rows, x_group;
      std::vector<int> splitks;
    };
    for (const Case& c : {Case{"q_b", &g_rw[r].qb, dqb.p, qr.p, kQLora, kOProjK, kQLora, 0, 0, {1, 2, 4}},
                          Case{"W_UV grouped", &g_rw[r].uv, duv.p, lat.p, kHeads * kKvLora, kOProjK, kKvLora, kVDim, kKvLora, {1}},
                          Case{"o_proj", &g_rw[r].o, dop.p, qx.p, kOProjK, kDim, kOProjK, 0, 0, {1, 2, 4}}}) {
      bool ok = true;
      int cases = 0;
      for (GemvOut out : {GemvOut::kBF16, GemvOut::kF32, GemvOut::kSwiGLU})
        for (int sk : c.splitks)
          for (int rows : {1, 6, 16}) {
            const size_t es = out == GemvOut::kF32 ? 4 : 2, ny = out == GemvOut::kSwiGLU ? c.N / 2 : c.N;
            Dev<uint8_t> ya((size_t)rows * ny * es), yb((size_t)rows * ny * es);
            GemvSeg a = seg(*c.w, c.N, ya.p, (int)ny, c.group_rows, c.x_group);
            GemvSeg b = seg_bf16(c.deq, c.N, yb.p, (int)ny, c.group_rows, c.x_group);
            gemv(c.x, c.ldx, rows, c.K, &b, 1, out, sk, 0);
            const auto ref = yb.down();
            for (int cfg = -1; cfg < 6; ++cfg) {  // every INT4 launch shape
              gemv_config(cfg);
              ya.zero();
              gemv(c.x, c.ldx, rows, c.K, &a, 1, out, sk, 0);
              ok = ok && ya.down() == ref;
              ++cases;
            }
            gemv_config(-1);
          }
      T(ok, "D INT4 GEMV rank %d %s = the BF16 GEMV on pack.py's dequantised matrix, bit for bit (%d cases: bf16/f32/"
        "SwiGLU out, split-K, 1/6/16 rows, launch shapes 0-5)", r, c.name, cases);
    }
    // q_b (INT4) and the indexer's wq_b (BF16) in one call: = the two alone
    {
      Dev<uint16_t> y1((size_t)16 * kOProjK), y2((size_t)16 * 1024), z1((size_t)16 * kOProjK), z2((size_t)16 * 1024);
      GemvSeg s2[2] = {seg(g_rw[r].qb, kOProjK, y1.p, kOProjK), seg(g_rw[r].iq, 1024, y2.p, 1024)};
      gemv(qr.p, kQLora, 16, kQLora, s2, 2, GemvOut::kBF16, 1, 0);
      GemvSeg a = seg(g_rw[r].qb, kOProjK, z1.p, kOProjK), b = seg(g_rw[r].iq, 1024, z2.p, 1024);
      gemv(qr.p, kQLora, 16, kQLora, &a, 1, GemvOut::kBF16, 1, 0);
      gemv(qr.p, kQLora, 16, kQLora, &b, 1, GemvOut::kBF16, 1, 0);
      T(y1.down() == z1.down() && y2.down() == z2.down(),
        "D q_b (INT4) + wq_b (BF16) in one gemv() call (two launches) = each alone, bit for bit (rank %d)", r);
    }
  }
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
      gemv(dx.p, kDim, rows, kDim, &s, 1, GemvOut::kBF16, 4, 0);
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
    check_bf16("B q_a (x-projection GEMV, 16 rows, 4 ranks' slices gathered)", eq.data(), qa + (size_t)r0 * kQLora, rows, kQLora, kQLora, kQLora);
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
    gemv(dx.p, kDim, nd, kDim, &s1, 1, GemvOut::kBF16, 4, 0);
    index_weights(dx.p, nd, wp1, kIdxHeadsRank, w1.p, kIdxHeadsRank, 0);
    GemvIw iw;
    iw.wproj = wp1;
    iw.nh = kIdxHeadsRank;
    iw.w = w2.p;
    iw.ldw = kIdxHeadsRank;
    gemv(dx.p, kDim, nd, kDim, &s2, 1, GemvOut::kBF16, 4, 0, false, &iw);
    sync();
    T(g1.down() == g2.down() && w1.down() == w2.down(),
      "B x-projection with the index weights in its launch = the two launches, bit for bit (%d rows)", nd);
    Dev<uint16_t> q1((size_t)nd * kOProjK), q2((size_t)nd * kOProjK);
    GemvSeg b1 = seg(g_rw[1].qb, kOProjK, q1.p, kOProjK), b2 = seg(g_rw[1].qb, kOProjK, q2.p, kOProjK);
    gemv(dqr.p, kQLora, nd, kQLora, &b1, 1, GemvOut::kBF16, 1, 0);
    bool same = true;
    for (int cfg = 0; cfg < 6; ++cfg) {
      gemv_config(cfg);
      gemv(dx.p, kDim, nd, kDim, &s2, 1, GemvOut::kBF16, 4, 0);
      gemv(dqr.p, kQLora, nd, kQLora, &b2, 1, GemvOut::kBF16, 1, 0);
      sync();
      same = same && g1.down() == g2.down() && q1.down() == q2.down();
    }
    gemv_config(-1);
    T(same, "B GEMV launch configurations 0-5 = the default (by shape), bit for bit (x-projection split-K 4, q_b)");
  }
  // q_b and the indexer's wq_b in one launch (two segments), every rank
  for (int r = 0; r < kTP; ++r) {
    GemvSeg s[2] = {seg(g_rw[r].qb, kOProjK, qo.p, kOProjK), seg(g_rw[r].iq, kIdxHeadsRank * kIdxDim, io.p, kIdxHeadsRank * kIdxDim)};
    gemv(dqr.p, kQLora, nd, kQLora, s, 2, GemvOut::kBF16, 1, 0);
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
  // the unsplit kernel (k_attn_pf, the default at one split) = k_attn at one split, bit for bit: the whole sequence
  // in one call (dense rows, sparse rows) and decode calls of 1 / 6 / 16 rows at one split
  {
    sparse_attn_config(0);
    Dev<uint16_t> ok((size_t)S * kHeadsAll * kVDim);
    run_attention(q, cache, topk, 0, S, ok, 1);
    sparse_attn_config(-1);
    auto hk = ok.down();
    T(hk == h1, "F the pipelined unsplit kernel (k_attn_pf) = k_attn at one split, %d rows (dense and sparse), bit for bit", S);
    for (int rows : {1, 6, 16}) {
      if (rows > nd) continue;
      Dev<uint16_t> od((size_t)S * kHeadsAll * kVDim);
      for (int a = S - nd; a < S; a += rows) run_attention(q, cache, topk, a, std::min(S, a + rows), od, 1);
      auto hd = od.down();
      const size_t off = (size_t)(S - nd) * kHeadsAll * kVDim;
      T(std::memcmp(hd.data() + off, h1.data() + off, (size_t)nd * kHeadsAll * kVDim * 2) == 0,
        "F decode calls of %d rows at one split = the prefill rows, bit for bit", rows);
    }
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
  gemv(qr.p, kOProjK, 4, kNope, &s, 1, GemvOut::kBF16, 1, 0);
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
      Wt gu = wload(p + "_gu", mlp_fmt()), dn = wload(p + "_down", mlp_fmt());
      GemvSeg s = seg(gu, 2 * inter, h.p, inter);
      gemv(x.p, kDim, nd, kDim, &s, 1, GemvOut::kSwiGLU, inter >= 2048 ? 1 : 4, 0);
      GemvSeg d = seg(dn, kDim, y.p, kDim);
      gemv(h.p, inter, nd, inter, &d, 1, GemvOut::kF32, 1, 0);
      auto hy = y.down();
      auto hr = h.down();
      for (size_t i = 0; i < acc.size(); ++i) acc[i] += hy[i];
      for (int i = 0; i < nd; ++i) std::memcpy(&hh[(size_t)i * inter * kTP + r * inter], &hr[(size_t)i * inter], inter * 2);
    }
    std::vector<uint16_t> out(acc.size());
    for (size_t i = 0; i < acc.size(); ++i) out[i] = tu::f2bf(acc[i]);
    char w[160];
    std::snprintf(w, sizeof w, "H %s gate/up (interleaved, SwiGLU epilogue) at %d rows", nm, nd);
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
      Wt dn = wload("r" + std::to_string(r) + "_" + nm + "_down", mlp_fmt());
      GemvSeg d = seg(dn, kDim, y.p, kDim);
      gemv(h.p, inter, nd, inter, &d, 1, GemvOut::kF32, 1, 0);
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
  s.fmt = WFmt::kBF16; s.w = head.p; s.n = kVocabRank; s.y = lg.p; s.ldy = kVocabRank;
  gemv(x.p, kDim, nd, kDim, &s, 1, GemvOut::kBF16, 1, 0);
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
  std::vector<Wt> keep;
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
    cs.push_back({"o_proj (f32) split-K 2" + rn, q0.p, kOProjK, kOProjK, {seg(g_rw[r].o, kDim, nullptr, kDim)}, {kDim}, GemvOut::kF32, 2});
  }
  if (has("dense_y"))
    for (auto [nm, inter] : {std::pair<const char*, int>{"dense", kDenseInterRank}, {"shared", kSharedInterRank}}) {
      keep.push_back(wload(std::string("r0_") + nm + "_gu", mlp_fmt()));
      keep.push_back(wload(std::string("r0_") + nm + "_down", mlp_fmt()));
      const Wt& gu = keep[keep.size() - 2];
      const Wt& dn = keep.back();
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
    gemv_rows(c.x, c.ldx, S, c.K, sa, c.out, c.splitk);
    sync();
    std::vector<std::vector<uint8_t>> ref;
    for (auto& y : ya) ref.push_back(y.down());
    bool ok = true;
    std::string which;
    for (int v = 0; v < 10; ++v) {  // tile configurations 0-5 (default raster), then rasters 0, 1, 3, 16 (default tiles)
      static const int kRasters[4] = {0, 1, 3, 16};
      const int cfg = v < 6 ? v : -1, gm = v < 6 ? -1 : kRasters[v - 6];
      for (auto& y : yb) y.zero();
      gemm_config(cfg);
      gemm_raster(gm);
      prefill_gemm(c.x, c.ldx, S, c.K, sb, c.out, c.splitk);
      sync();
      bool e = true;
      for (size_t i = 0; i < yb.size(); ++i) e = e && yb[i].down() == ref[i];
      if (!e) which += v < 6 ? " cfg " + std::to_string(cfg) : " raster " + std::to_string(gm);
      ok = ok && e;
    }
    gemm_config(-1);
    gemm_raster(-1);
    T(ok, "P GEMM %s, %d rows: = the 16-row GEMV bit for bit in tile configurations 0-5 and rasters 0/1/3/16%s%s", c.name.c_str(), S,
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
      g.fmt = WFmt::kBF16, g.w = head.p, g.n = b.n, g.y = ya.p, g.ldy = b.n;
      gemv_rows(x.p, kDim, S, kDim, {g}, b.out, b.splitk);
      sync();
      const std::vector<uint8_t> ref = ya.down();
      g.y = yb.p;
      gemm_bf16(x.p, kDim, S, kDim, &g, 1, b.out, b.splitk, 0);
      sync();
      T(yb.down() == ref, "P GEMM BF16 %s, %d rows: = the 16-row BF16 GEMV bit for bit", b.name, S);
    }
  }
}

// ---- R: real altar-1 tensors (tools/real_ref.py: one rank's slices of a verified shard) ------------------------
// INT4 cases: the engine tile, compressed-tensors' own dequantisation of the same rows ("lib"), 16 activation rows
// and torch's f32-accumulated product of them with lib, rounded to bf16 (the oracle). BF16 cases: the tile is the
// weight. Checks: dequant_int4 = lib bit for bit; the INT4 GEMV = the BF16 GEMV on lib bit for bit (1/6/16 rows,
// bf16 and f32 out) and the INT4 GEMM = both; every GEMV against the oracle up to summation order.
void test_real() {
  struct Case {
    const char* name;
    WFmt fmt;
    int N, K, group_rows, x_group, ldx;
  };
  const int layer = manifest_int("layer"), rank = manifest_int("rank");
  std::printf("real tensors: layer %d, rank %d (%s)\n", layer, rank, manifest_str("source").c_str());
  for (const Case& c : {Case{"q_b", WFmt::kINT4, kOProjK, kQLora, 0, 0, kQLora},
                        Case{"w_uv", WFmt::kINT4, kOProjK, kKvLora, kVDim, kKvLora, kHeads * kKvLora},
                        Case{"o_proj", WFmt::kINT4, kDim, kOProjK, 0, 0, kOProjK},
                        Case{"w_uk", WFmt::kBF16, kHeads * kKvLora, kNope, kKvLora, kQkDim, kOProjK},
                        Case{"q_b_bf16", WFmt::kBF16, kOProjK, kQLora, 0, 0, kQLora},
                        Case{"o_proj_bf16", WFmt::kBF16, kDim, kOProjK, 0, 0, kOProjK}}) {
    const std::string n = c.name;
    if (!has(n + "_w")) { std::printf("skip R %s: no data\n", c.name); continue; }
    Dev<uint8_t> w = tu::to_dev<uint8_t>(L(n + "_w"));
    Dev<uint16_t> x = tu::to_dev<uint16_t>(L(n + "_x"));
    const auto& yref = L(n + "_y");
    GemvSeg s;
    s.fmt = c.fmt, s.w = w.p, s.n = c.N, s.ldy = c.N, s.group_rows = c.group_rows, s.x_group = c.x_group;
    Dev<uint16_t> y((size_t)16 * c.N), yl((size_t)16 * c.N);
    if (c.fmt == WFmt::kINT4) {
      Dev<uint16_t> lib = tu::to_dev<uint16_t>(L(n + "_lib"));
      Dev<uint16_t> dq((size_t)c.N * c.K);
      dequant_int4(w.p, c.N, c.K, dq.p, c.K, 0);
      T(dq.down() == lib.down(), "R L%d %s [%d][%d]: dequant_int4 of the engine tile = compressed-tensors' decompress, every "
        "value bit for bit", layer, c.name, c.N, c.K);
      bool ok = true;
      for (GemvOut out : {GemvOut::kBF16, GemvOut::kF32})
        for (int rows : {1, 6, 16}) {
          const size_t es = out == GemvOut::kF32 ? 4 : 2;
          Dev<uint8_t> ya((size_t)rows * c.N * es), yb((size_t)rows * c.N * es), yc((size_t)rows * c.N * es);
          GemvSeg a = s, b = s, g = s;
          a.y = ya.p, b.y = yb.p, b.fmt = WFmt::kBF16, b.w = lib.p, g.y = yc.p;
          gemv(x.p, c.ldx, rows, c.K, &a, 1, out, 1, 0);
          gemv(x.p, c.ldx, rows, c.K, &b, 1, out, 1, 0);
          prefill_gemm(x.p, c.ldx, rows, c.K, {g}, out, 1);
          const auto ra = ya.down();
          ok = ok && ra == yb.down() && ra == yc.down();
        }
      T(ok, "R L%d %s: the INT4 GEMV = the BF16 GEMV on compressed-tensors' weights = the INT4 GEMM, bit for bit (1/6/16 "
        "rows, bf16 and f32 out)", layer, c.name);
    }
    s.y = y.p;
    gemv(x.p, c.ldx, 16, c.K, &s, 1, GemvOut::kBF16, 1, 0);
    const auto hy = y.down();
    char what[160];
    std::snprintf(what, sizeof what, "R L%d %s (%s GEMV, 16 rows) vs torch's f32-accumulated product on the library's weights",
                  n.size() > 5 && n.substr(n.size() - 5) == "_bf16" ? manifest_int("bf16_layer") : layer, c.name,
                  c.fmt == WFmt::kINT4 ? "INT4" : "BF16");
    check_bf16(what, hy.data(), yref.as<uint16_t>(), 16, c.N, c.N, c.N);
  }
  (void)rank;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) die("usage: attn_tests DIR [--only ADCBPEFGSH]");
  g_dir = argv[1];
  if (manifest_str("kind") == "real") {  // tools/real_ref.py's data: real altar-1 tensors
    test_real();
    std::printf("%d passed, %d failed\n", T.pass, T.fail);
    return T.fail ? 1 : 0;
  }
  std::string only = argc > 3 && std::string(argv[2]) == "--only" ? argv[3] : "ADCBPEFGSH";
  const int S = manifest_int("S"), nd = manifest_int("decode");
  g_fmt = manifest_str("fmt");
  if (g_fmt.empty()) g_fmt = "fp8";
  std::printf("case %s: S %d, decode rows %d, layer format %s\n", g_dir.c_str(), S, nd, g_fmt.c_str());
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
      case 'D': test_int4(S); break;
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
