// glw_kcheck: does a rank's .glw file hold what the kernels read? (the DeepSeek engine's dsw_kcheck, for GLM-5.3;
// the routed experts' part for altar-1: 168 experts a layer, INT4 or BF16 by layer, experts.h's formats)
//
//   glw_kcheck FILE --rank R                 host only, no device: every layer's tensors against engine/attn/attn.h's
//                                            constants and GemvSeg conventions (shapes, dtypes, layouts, the scale
//                                            block rs x ks and the scale table's shape, the 8 x 64 chunk inside one
//                                            block), the full-indexer pattern, the BF16 head/embedding/norms, the RoPE
//                                            table, and the routed experts' reserved entries against
//                                            engine/experts/experts.h (block bytes, MatDesc); binds every FP8 tensor
//                                            into a glm::GemvSeg as the layer runner will
//   glw_kcheck FILE --rank R --partial --layers 3
//                                            a test file holding some layers (convert.py --only): those only
//   ... --blocks                             also reads every filled expert entry and checks its scale fields
//                                            (INT4: bf16 finite and positive; NVFP4: no e4m3 NaN, positive); every
//                                            descriptor is checked with layout.h desc_ok (tag, alignment, s2, reserved)
//   glw_kcheck FILE --rank R --gpu [--layers 0,3,10,78] [--rows 4]
//                                            also loads those layers (and the head) and runs glm::gemv on each FP8
//                                            projection and the BF16 head with random bf16 inputs, against a host
//                                            reference that un-tiles the file's own bytes with glw.h's formula and
//                                            dequantises in double: kF32 outputs within 1e-4 of sum |x w|, the
//                                            SwiGLU pairs within 2 bf16 ulps plus the spread the kernel's f32 sums'
//                                            rounding can carry through the bf16 SwiGLU (swiglu_tol below). Needs
//                                            the GPU (a ring job).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "experts/experts.h"
#include "experts/layout.h"
#include "common/check.h"
#include "weights/loader.h"

using namespace glw;

namespace {

int g_bad = 0, g_ok = 0;
bool g_blocks = false;                    // --blocks: scan the filled experts' scale bytes
size_t g_fmt_mats[4] = {};                // filled matrices by format tag
size_t g_scales = 0;                       // --blocks: scale fields scanned
void expect(bool ok, const std::string& what) {
  if (ok) {
    ++g_ok;
    return;
  }
  ++g_bad;
  std::printf("BAD %s\n", what.c_str());
}

struct Want {
  DType dt;
  std::vector<int64_t> shape;
  Layout layout;
  int rs = 0, ks = 0;
  bool interleaved = false;
};

std::string shp(const int64_t* s, int n) {
  std::ostringstream o;
  for (int i = 0; i < n; i++) o << (i ? "x" : "") << s[i];
  return o.str();
}

const DeviceTensor* need(const WeightFile& wf, const std::string& name, const Want& w) {
  const DeviceTensor* t = wf.find(name);
  if (!t) {
    expect(false, name + ": missing");
    return nullptr;
  }
  bool ok = t->dtype == w.dt && t->layout == w.layout && t->ndim == int(w.shape.size());
  for (size_t i = 0; ok && i < w.shape.size(); i++) ok = t->shape[i] == w.shape[i];
  expect(ok, name + ": " + dtype_name(t->dtype) + " " + shp(t->shape, t->ndim) + " " + layout_name(t->layout) +
                 ", the kernels read " + dtype_name(w.dt) + " " + shp(w.shape.data(), int(w.shape.size())) + " " +
                 layout_name(w.layout));
  return t;
}

// An FP8 GEMV weight [N][K] in T64 tiles and its scale table, as attn.h's GemvSeg reads them.
glm::GemvSeg fp8(const WeightFile& wf, const std::string& base, int N, int K, int rs, int ks, bool inter = false) {
  glm::GemvSeg s;
  const DeviceTensor* w = need(wf, base + ".weight", {DType::F8_E4M3, {N, K}, Layout::T64});
  std::vector<int64_t> ss = inter ? std::vector<int64_t>{N / 2 / rs, 2, K / ks} : std::vector<int64_t>{N / rs, K / ks};
  const DeviceTensor* sc = need(wf, base + ".scale", {DType::F32, ss, Layout::Natural});
  expect((rs == 16 || rs == 64 || rs == 128) && (ks == 64 || ks == 128) && N % 8 == 0 && K % 64 == 0,
         base + ": rs, ks in {16, 64, 128} and 8 x 64 chunks (attn.h)");
  if (w && sc) {
    expect(w->blk_rows == uint32_t(rs) && w->blk_k == uint32_t(ks) && sc->blk_rows == uint32_t(rs) &&
               sc->blk_k == uint32_t(ks),
           base + ": the TOC's scale block is " + std::to_string(w->blk_rows) + "x" + std::to_string(w->blk_k) +
               ", the kernels' " + std::to_string(rs) + "x" + std::to_string(ks));
    expect(bool(w->flags & kInterleaved) == inter && bool(sc->flags & kScale), base + ": flags");
    expect(w->rel_err > 0 && w->rel_err < 0.05f, base + ": relative error " + std::to_string(w->rel_err));
    s.w = w->data ? w->data : reinterpret_cast<void*>(w->offset);  // host check: the arena offset stands in
    s.s = static_cast<const float*>(sc->data ? sc->data : reinterpret_cast<void*>(sc->offset));
  }
  s.n = N;
  s.rs = rs;
  s.ks = ks;
  return s;
}

// The format of layer L's routed experts (glw.py layer_expert_fmt): BF16 on 3, 77 and the MTP layer 78, INT4 on 4-76;
// the MTP layer may also hold NVFP4 (DESIGN decision 5's option).
bool layer_fmt_ok(int L, uint32_t f) {
  if (L == 78) return f == gx::kBF16 || f == gx::kNVFP4;
  return f == (L == 3 || L == 77 ? gx::kBF16 : gx::kINT4);
}

// The routed experts of a MoE layer: the desc entry and 168 expert entries, each sized by its format tag with
// experts.h's mat_bytes; one format for the whole layer (LayerW::fmt: the kernels are instantiated per format), the
// layer's own; once filled, the entries are back to back (the layer's pool) and every MatDesc points at its matrix
// inside its expert's entry with the entry's format and passes layout.h's desc_ok.
void check_experts(const WeightFile& wf, int L) {
  const std::string p = "layers." + std::to_string(L) + ".experts.";
  const DeviceTensor* ed = wf.find(p + "desc");
  expect(ed && ed->ndim == 3 && ed->shape[0] == gx::kE && ed->shape[1] == 3 &&
             ed->shape[2] == int64_t(sizeof(gx::MatDesc)),
         p + "desc: [168][3][sizeof(gx::MatDesc)]");
  if (!ed) return;
  std::vector<uint8_t> db;
  if (!ed->reserved()) db = wf.read_host(*ed);
  const auto* desc = reinterpret_cast<const gx::MatDesc*>(db.data());
  uint64_t base = 0, next = 0;
  int filled = 0;
  uint32_t layer_fmt = 0xFFFFFFFFu;
  for (int e = 0; e < gx::kE; e++) {
    const DeviceTensor* t = wf.find(p + std::to_string(e));
    if (!t) {
      expect(false, p + std::to_string(e) + ": missing");
      continue;
    }
    const int N[3] = {gx::kI, gx::kI, gx::kH}, K[3] = {gx::kH, gx::kH, gx::kI};
    uint64_t want = 0, moff[3];
    for (int m = 0; m < 3; m++) {
      const uint32_t f = expert_matrix_fmt(t->fmt, m);
      expect(gx::fmt_known(f) && layer_fmt_ok(L, f), t->name + ": format tag " + std::to_string(f) + " of matrix " + std::to_string(m));
      if (layer_fmt == 0xFFFFFFFFu) layer_fmt = f;
      expect(f == layer_fmt, t->name + ": format " + std::to_string(f) + ", the layer's is " + std::to_string(layer_fmt));
      moff[m] = want;
      want += gx::mat_bytes(f, N[m], K[m]);
    }
    expect(t->layout == Layout::ExpertPool && t->dtype == DType::U8 && want % kAlign == 0,
           t->name + ": expert-pool U8, 4 KiB multiple");
    if (t->reserved()) {
      expect(t->reserve_bytes == want, t->name + ": reserved for its format's bytes");
      continue;
    }
    filled++;
    expect(t->bytes == want, t->name + ": " + std::to_string(t->bytes) + " B, its format needs " + std::to_string(want));
    if (e == 0) base = t->file_offset, next = base;
    expect(t->file_offset == next, t->name + ": contiguous with the previous expert");
    next = t->file_offset + t->bytes;
    if (desc)
      for (int m = 0; m < 3; m++) {
        const gx::MatDesc& d = desc[3 * e + m];
        const uint32_t f = expert_matrix_fmt(t->fmt, m);
        expect(d.off == t->file_offset - base + moff[m] && gx::fmt_known(f) && gx::desc_ok(d, f),
               t->name + ": descriptor " + std::to_string(m));
        if (gx::fmt_known(f)) ++g_fmt_mats[f];
      }
    // --blocks: every stored scale field of the expert's matrices (INT4: bf16, finite and positive; NVFP4: not the
    // e4m3 NaN, positive), so a damaged or misaligned fill is found before the kernels read it
    if (g_blocks) {
      const std::vector<uint8_t> ent = wf.read_host(*t);
      for (int m = 0; m < 3; m++) {
        const uint32_t f = expert_matrix_fmt(t->fmt, m);
        if (!gx::fmt_known(f)) continue;
        const size_t bad = gx::scale_bytes_check(ent.data() + moff[m], f, N[m], K[m]);
        expect(bad == 0, t->name + ": matrix " + std::to_string(m) + " has " + std::to_string(bad) + " bad scales");
        g_scales += f == gx::kINT4 ? size_t(N[m]) * K[m] / 32 : f == gx::kNVFP4 ? size_t(N[m]) * K[m] / 16 : 0;
      }
    }
  }
  expect(filled == 0 || (filled == gx::kE && !ed->reserved()), p + ": all 168 experts and the desc filled, or none");
}

void print_expert_summary() {
  if (g_fmt_mats[gx::kNVFP4] + g_fmt_mats[gx::kINT4] + g_fmt_mats[gx::kBF16])
    std::printf("expert matrices: %zu INT4, %zu BF16, %zu NVFP4\n", g_fmt_mats[gx::kINT4], g_fmt_mats[gx::kBF16],
                g_fmt_mats[gx::kNVFP4]);
  if (g_blocks) std::printf("--blocks: %zu scale fields scanned\n", g_scales);
}

struct LayerSegs {
  glm::GemvSeg xproj, q_b, wq_b, w_uk, w_uv, o_proj, gate_up, down;
  bool full = false, dense = false, t128 = false;
  int inter = 0;
};

LayerSegs bind_layer(const WeightFile& wf, int L) {
  using namespace glm;
  LayerSegs g;
  const std::string p = "layers." + std::to_string(L) + ".";
  g.full = L == kLayers ? true : full_indexer(L);  // the MTP layer (78) has its own full indexer
  g.dense = L < 3;
  for (auto [n, d] : {std::pair{"attn_norm.weight", kDim}, {"ffn_norm.weight", kDim}, {"attn.q_norm.weight", kQLora},
                      {"attn.kv_norm.weight", kKvLora}})
    need(wf, p + n, {DType::BF16, {d}, Layout::Natural});
  g.xproj = fp8(wf, p + "attn.xproj", g.full ? kXRows : kXRowsShared, kDim, 16, 128);
  g.q_b = fp8(wf, p + "attn.q_b", kHeads * kQkDim, kQLora, 128, 128);
  g.w_uk = fp8(wf, p + "attn.w_uk", kHeads * kKvLora, kNope, 128, 64);
  g.w_uk.group_rows = kKvLora, g.w_uk.x_group = kQkDim;
  g.w_uv = fp8(wf, p + "attn.w_uv", kHeads * kVDim, kKvLora, 64, 128);
  g.w_uv.group_rows = kVDim, g.w_uv.x_group = kKvLora;
  g.o_proj = fp8(wf, p + "attn.o_proj", kDim, kOProjK, 128, 128);
  const bool has_idx = wf.find(p + "indexer.wq_b.weight") != nullptr;
  expect(has_idx == g.full, p + ": indexer tensors present exactly on full-indexer layers");
  if (g.full) {
    g.wq_b = fp8(wf, p + "indexer.wq_b", kIdxHeadsRank * kIdxDim, kQLora, 128, 128);
    need(wf, p + "indexer.weights_proj.weight", {DType::BF16, {kIdxHeadsRank, kDim}, Layout::Natural});
    need(wf, p + "indexer.k_norm.weight", {DType::BF16, {kIdxDim}, Layout::Natural});
    need(wf, p + "indexer.k_norm.bias", {DType::BF16, {kIdxDim}, Layout::Natural});
  }
  if (g.dense) {
    g.inter = kDenseInterRank;
    g.gate_up = fp8(wf, p + "mlp.gate_up", 2 * kDenseInterRank, kDim, 128, 128, true);
    g.down = fp8(wf, p + "mlp.down", kDim, kDenseInterRank, 128, 128);
  } else {
    g.inter = kSharedInterRank;
    need(wf, p + "router.weight", {DType::BF16, {gx::kE, kDim}, Layout::Natural});
    need(wf, p + "router.bias", {DType::F32, {gx::kE}, Layout::Natural});
    g.t128 = wf.find(p + "shared.gate.weight") != nullptr;
    if (g.t128) {  // the expert kernels' SharedW: gate, up [512][6144], down [6144][512] in tile_fp8, scales [N/128][K/128]
      for (auto n : {"gate", "up"}) {
        need(wf, p + "shared." + n + ".weight", {DType::F8_E4M3, {gx::kI, gx::kH}, Layout::T128});
        need(wf, p + "shared." + n + ".scale", {DType::F32, {gx::kI / 128, gx::kH / 128}, Layout::Natural});
      }
      need(wf, p + "shared.down.weight", {DType::F8_E4M3, {gx::kH, gx::kI}, Layout::T128});
      need(wf, p + "shared.down.scale", {DType::F32, {gx::kH / 128, gx::kI / 128}, Layout::Natural});
    } else {
      g.gate_up = fp8(wf, p + "shared.gate_up", 2 * kSharedInterRank, kDim, 128, 128, true);
      g.down = fp8(wf, p + "shared.down", kDim, kSharedInterRank, 128, 128);
    }
    check_experts(wf, L);
  }
  if (L == kLayers) {
    for (auto n : {"mtp.enorm.weight", "mtp.hnorm.weight", "mtp.head_norm.weight"})
      need(wf, p + n, {DType::BF16, {kDim}, Layout::Natural});
    fp8(wf, p + "mtp.eh_proj", kDim / kTP, 2 * kDim, 128, 128);
  }
  return g;
}

// ---- the device check -------------------------------------------------------------------------------------------
float e4m3(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  if (e == 15 && m == 7) return NAN;
  const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
  return s ? -v : v;
}
float bf(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t to_bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}

// W(n, k) of an FP8 tensor from the file's own bytes (glw.h's T64 formula; the scale at [n/rs][k/ks], or
// [n/2/rs][n%2][k/ks] for an interleaved pair)
struct HostFp8 {
  std::vector<uint8_t> w;
  std::vector<float> s;
  int N, K, rs, ks;
  bool inter;
  double at(int n, int k) const {
    const float sc = inter ? s[((n / 2 / rs) * 2 + n % 2) * (K / ks) + k / ks] : s[(n / rs) * (K / ks) + k / ks];
    return double(e4m3(w[t64_offset(n, k, K)])) * sc;
  }
};
HostFp8 host_fp8(const WeightFile& wf, const std::string& base, const glm::GemvSeg& g, int K) {
  HostFp8 h;
  h.w = wf.read_host(*wf.find(base + ".weight"));
  auto sb = wf.read_host(*wf.find(base + ".scale"));
  h.s.resize(sb.size() / 4);
  std::memcpy(h.s.data(), sb.data(), sb.size());
  h.N = g.n, h.K = K, h.rs = g.rs, h.ks = g.ks, h.inter = bool(wf.find(base + ".weight")->flags & kInterleaved);
  return h;
}

// The SwiGLU bound. The kernel sums each gate/up row in f32: per 64-k chunk an MMA of exact bf16 x bf16 products
// (internal error <= 2^-22 of the chunk's sum |x w|), times the chunk's scale (<= 2^-24 of |s c|), added to the
// running sum (<= 2^-24 of it). To first order its g is within E = 2^-22 sum|x w s| + 2^-24 sum_j (|S_j| + |s_j c_j|)
// of the exact one (S_j the exact running sum after chunk j). The reference's bf16(silu(bf16 g)) * bf16 u is then
// recomputed at the four corners g +- E_g, u +- E_u, and the output may sit 2 bf16 ulps of ref beyond the widest
// corner. For most outputs E moves nothing (the corners round to ref); it matters where the gate's sum cancels
// (2026-10-14: layers.0.mlp.gate_up row 1 intermediate 659, |g| = 3.3e-6 of sum |x w| = 52.4, the kernel's f32 g
// 1.8 % from the exact one, 3 ulps on the output; docs/layout.md) or a value sits on a bf16 rounding boundary.
struct RowSum {
  double v = 0, mag = 0, err = 0;  // the exact sum, sum |x w s|, E
};
float swiglu_ref(double g, double u) {
  const float gb = bf(to_bf(float(g))), ub = bf(to_bf(float(u)));
  return bf(to_bf(bf(to_bf(gb / (1.f + std::exp(-gb)))) * ub));
}
float ulp_bf(float v) { return std::ldexp(1.f, std::ilogb(std::fabs(v) + 1e-30f) - 7); }
float swiglu_tol(const RowSum& g, const RowSum& u, float ref) {
  float spread = 0;
  for (int a = -1; a <= 1; a += 2)
    for (int b = -1; b <= 1; b += 2)
      spread = std::max(spread, std::fabs(swiglu_ref(g.v + a * g.err, u.v + b * u.err) - ref));
  return 2 * ulp_bf(ref) + spread;
}

template <class T>
T* dev(const std::vector<T>& v) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, v.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

// One FP8 GEMV (or a grouped one) against the host reference; kF32 out, or kSwiGLU for an interleaved pair.
void run_fp8(const WeightFile& wf, const std::string& base, glm::GemvSeg g, int K, int ldx, int rows, std::mt19937& rng) {
  const HostFp8 h = host_fp8(wf, base, g, K);
  const bool swiglu = h.inter;
  const int nout = swiglu ? g.n / 2 : g.n;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<uint16_t> x(size_t(rows) * ldx);
  for (auto& v : x) v = to_bf(nd(rng));
  uint16_t* dx = dev(x);
  void* dy;
  const size_t ysz = size_t(rows) * nout * (swiglu ? 2 : 4);
  CUDA_CHECK(cudaMalloc(&dy, ysz));
  g.y = dy;
  g.ldy = nout;
  glm::gemv(true, dx, ldx, rows, K, &g, 1, swiglu ? glm::GemvOut::kSwiGLU : glm::GemvOut::kF32, 1, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> y(ysz);
  CUDA_CHECK(cudaMemcpy(y.data(), dy, ysz, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int r = 0; r < rows; r++)
    for (int n = 0; n < g.n; n++) {
      if (swiglu && n % 2) continue;
      auto dot = [&](int row, double& mag) {
        const int xo = g.group_rows ? (row / g.group_rows) * g.x_group : 0;
        double acc = 0;
        mag = 0;
        for (int k = 0; k < K; k++) {
          const double p = double(bf(x[size_t(r) * ldx + xo + k])) * h.at(row, k);
          acc += p, mag += std::fabs(p);
        }
        return acc;
      };
      double mag;
      if (!swiglu) {
        const double ref = dot(n, mag);
        float got;
        std::memcpy(&got, &y[(size_t(r) * nout + n) * 4], 4);
        if (std::fabs(got - ref) > 1e-4 * mag + 1e-30) bad++;
      } else {
        auto rsum = [&](int row) {  // in the kernel's 64-k chunks, for E
          RowSum s;
          for (int k0 = 0; k0 < K; k0 += 64) {
            double c = 0, cm = 0;
            for (int k = k0; k < k0 + 64; k++) {
              const double p = double(bf(x[size_t(r) * ldx + k])) * h.at(row, k);
              c += p, cm += std::fabs(p);
            }
            s.v += c, s.mag += cm;
            s.err += std::fabs(s.v) + std::fabs(c);
          }
          s.err = std::ldexp(s.mag, -22) + std::ldexp(s.err, -24);
          return s;
        };
        const RowSum gs = rsum(n), us = rsum(n + 1);
        const float ref = swiglu_ref(gs.v, us.v);
        uint16_t got;
        std::memcpy(&got, &y[(size_t(r) * nout + n / 2) * 2], 2);
        const float d = std::fabs(bf(got) - ref), tol = swiglu_tol(gs, us, ref);
        if (d > tol) bad++;
        if (d > 2 * ulp_bf(ref))  // outside the plain 2-ulp bound: say why
          std::printf("    row %d intermediate %d: got %.9g ref %.9g (exact %.9g), %.1f ulps; gate %.9g of sum|xw| "
                      "%.4g (E %.3g), up %.9g (E %.3g); bound %.3g: %s\n",
                      r, n / 2, bf(got), ref, gs.v / (1 + std::exp(-gs.v)) * us.v, d / ulp_bf(ref), gs.v, gs.mag,
                      gs.err, us.v, us.err, tol, d > tol ? "BAD" : "inside");
      }
    }
  expect(bad == 0, base + ": " + std::to_string(bad) + " of " + std::to_string(rows * nout) + " GEMV outputs off");
  std::printf("  %-28s %5d x %-5d %d rows: %s\n", base.c_str(), g.n, K, rows, bad ? "BAD" : "matches the file's bytes");
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) die("usage: glw_kcheck FILE --rank R [--gpu] [--layers 0,3,10,78] [--rows 4]");
  const std::string file = argv[1];
  int rank = -1, rows = 4;
  bool gpu = false, partial = false;
  std::vector<int> layers = {0, 3, 10, 78};
  for (int i = 2; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--rank" && i + 1 < argc) rank = std::atoi(argv[++i]);
    else if (a == "--gpu") gpu = true;
    else if (a == "--partial") partial = true;  // a test file (convert.py --only): check just --layers
    else if (a == "--blocks") g_blocks = true;
    else if (a == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
    else if (a == "--layers" && i + 1 < argc) {
      layers.clear();
      std::stringstream ss(argv[++i]);
      for (std::string t; std::getline(ss, t, ',');) layers.push_back(std::stoi(t));
    } else die("unknown argument %s", a.c_str());
  }
  if (rank < 0) die("--rank R is required: the check binds the file as that rank's, as the engine does");
  WeightFile wf(file, rank, glm::kTP);  // refuses another rank's file before anything else

  // host: the whole file against the kernels' constants
  using namespace glm;
  if (partial) {
    for (int L : layers) bind_layer(wf, L);
    std::printf("%s: rank %d, %zu entries; host binding of layers", file.c_str(), rank, wf.tensors().size());
    for (int L : layers) std::printf(" %d", L);
    std::printf(": %d checks ok, %d bad\n", g_ok, g_bad);
    print_expert_summary();
    return g_bad ? 1 : 0;
  }
  need(wf, "embed.weight", {DType::BF16, {kVocabRank, kDim}, Layout::Natural});
  need(wf, "head.weight", {DType::BF16, {kVocabRank, kDim}, Layout::Natural});
  need(wf, "norm.weight", {DType::BF16, {kDim}, Layout::Natural});
  need(wf, "rope.table", {DType::U32, {1 << 20, kRope / 2}, Layout::Natural});
  int full = 0;
  for (int L = 0; L <= kLayers; L++) full += bind_layer(wf, L).full;
  expect(full == 22, "full-indexer layers: " + std::to_string(full) + " (21 + the MTP layer's)");
  std::printf("%s: rank %d, %zu entries; host binding: %d checks ok, %d bad\n", file.c_str(), rank,
              wf.tensors().size(), g_ok, g_bad);
  print_expert_summary();
  if (!gpu) return g_bad ? 1 : 0;

  // device: load the chosen layers and the head, run the GEMVs
  std::set<std::string> pre;
  for (int L : layers) pre.insert("layers." + std::to_string(L) + ".");
  LoadOptions opt;
  opt.select = [&](const DeviceTensor& t) {
    if (t.name == "head.weight") return true;
    for (auto& p : pre)
      if (t.name.rfind(p, 0) == 0) return true;
    return false;
  };
  CUDA_CHECK(cudaSetDevice(0));
  const LoadStats st = wf.load(opt);
  std::printf("loaded %.2f GiB of %zu layers and the head in %.2f s\n", st.bytes / 1073741824.0, layers.size(),
              st.seconds);
  std::mt19937 rng(1234);
  for (int L : layers) {
    const std::string p = "layers." + std::to_string(L) + ".";
    LayerSegs g = bind_layer(wf, L);
    run_fp8(wf, p + "attn.xproj", g.xproj, kDim, kDim, rows, rng);
    run_fp8(wf, p + "attn.q_b", g.q_b, kQLora, kQLora, rows, rng);
    if (g.full) run_fp8(wf, p + "indexer.wq_b", g.wq_b, kQLora, kQLora, rows, rng);
    run_fp8(wf, p + "attn.w_uk", g.w_uk, kNope, kOProjK, rows, rng);
    run_fp8(wf, p + "attn.w_uv", g.w_uv, kKvLora, kHeads * kKvLora, rows, rng);
    run_fp8(wf, p + "attn.o_proj", g.o_proj, kOProjK, kOProjK, rows, rng);
    if (!g.t128) {
      const std::string m = p + (g.dense ? "mlp" : "shared");
      run_fp8(wf, m + ".gate_up", g.gate_up, kDim, kDim, rows, rng);
      run_fp8(wf, m + ".down", g.down, g.inter, g.inter, rows, rng);
    }
  }
  {  // the BF16 head, row-major
    const DeviceTensor& h = wf["head.weight"];
    std::vector<uint8_t> hb = wf.read_host(h);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x(size_t(rows) * kDim);
    for (auto& v : x) v = to_bf(nd(rng));
    uint16_t* dx = dev(x);
    float* dy;
    CUDA_CHECK(cudaMalloc(&dy, size_t(rows) * kVocabRank * 4));
    GemvSeg s;
    s.w = h.data, s.n = kVocabRank, s.y = dy, s.ldy = kVocabRank;
    gemv(false, dx, kDim, rows, kDim, &s, 1, GemvOut::kF32, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> y(size_t(rows) * kVocabRank);
    CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    const uint16_t* w = reinterpret_cast<const uint16_t*>(hb.data());
    for (int r = 0; r < rows; r++)
      for (int n = 0; n < kVocabRank; n += 7) {
        double acc = 0, mag = 0;
        for (int k = 0; k < kDim; k++) {
          const double p = double(bf(x[size_t(r) * kDim + k])) * bf(w[size_t(n) * kDim + k]);
          acc += p, mag += std::fabs(p);
        }
        if (std::fabs(y[size_t(r) * kVocabRank + n] - acc) > 1e-4 * mag + 1e-30) bad++;
      }
    expect(bad == 0, "head: " + std::to_string(bad) + " sampled outputs off");
    std::printf("  %-28s %5d x %-5d %d rows: %s\n", "head.weight (BF16)", kVocabRank, kDim, rows,
                bad ? "BAD" : "matches the file's bytes");
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
  }
  std::printf("%s: %d checks ok, %d bad\n", g_bad ? "FAIL" : "PASS", g_ok, g_bad);
  return g_bad ? 1 : 0;
}
