// dsw_kcheck: proves that a version-2 weight file holds what engine/kernels reads. On rank 0's file it
// takes the kernel tests' reference data (engine/kernels/tests/gen_ref.py: rank 0's slices of real
// layer-3 weights, inputs and outputs) and
//   1. tiles the reference weight slices with the kernels' own dsk::tile_fp8 / dsk::tile_fp4 and
//      compares the result byte for byte with the file's tensors (also with layout.h, the host
//      implementation dsw_verify un-tiles with);
//   2. runs the kernels' GEMVs straight on the file's tensors and checks their outputs against the
//      reference outputs with test_kernels' criteria.
//
//   dsw_kcheck FILE [REFDIR, default ~/dsgb10-data/kerneltests]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels/dsk.h"
#include "kernels/tests/npy.h"  // also brings die() and CUDA_CHECK (common/check.h)
#include "weights/layout.h"
#include "weights/loader.h"

using namespace dsw;

namespace {

constexpr int kLayer = 3;  // gen_ref.py's LAYER
std::string g_dir;
int g_fail = 0;

Npy L(const std::string& name) { return npy_load(g_dir + "/" + name + ".npy"); }

template <class T = uint8_t>
T* dev_copy(const void* h, size_t bytes) {
  void* p;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  CUDA_CHECK(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice));
  return static_cast<T*>(p);
}
template <class T>
T* dev_zero(size_t n) {
  void* p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
  return static_cast<T*>(p);
}
template <class T>
std::vector<T> to_host(const void* d, size_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

float bf2f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void result(bool ok, const char* what, const std::string& detail) {
  g_fail += !ok;
  std::printf("  %-34s %s %s\n", what, ok ? "ok  " : "FAIL", detail.c_str());
}

// ---- part 1: bytes

// The kernels' tilings of natural host data.
std::vector<uint8_t> kernel_tile8(const uint8_t* nat, int N, int K) {
  uint8_t *d = dev_copy(nat, size_t(N) * K), *t = dev_zero<uint8_t>(size_t(N) * K);
  dsk::tile_fp8(d, t, N, K, 0);
  auto h = to_host<uint8_t>(t, size_t(N) * K);
  cudaFree(d), cudaFree(t);
  return h;
}
void kernel_tile4(const uint8_t* w, const uint8_t* s, int N, int K, std::vector<uint8_t>& wt, std::vector<uint8_t>& st) {
  const size_t wb = size_t(N) * K / 2, sb = size_t(N) * K / 32;
  uint8_t *dw = dev_copy(w, wb), *ds = dev_copy(s, sb), *tw = dev_zero<uint8_t>(wb), *ts = dev_zero<uint8_t>(sb);
  dsk::tile_fp4(dw, ds, tw, ts, N, K, 0);
  wt = to_host<uint8_t>(tw, wb), st = to_host<uint8_t>(ts, sb);
  cudaFree(dw), cudaFree(ds), cudaFree(tw), cudaFree(ts);
}

// File bytes [off, off + want.size()) of tensor t equal `want`.
void same_bytes(const char* what, const DeviceTensor& t, uint64_t off, const std::vector<uint8_t>& want) {
  if (off + want.size() > t.bytes) die("%s: %s is too small", what, t.name.c_str());
  auto got = to_host<uint8_t>(static_cast<const uint8_t*>(t.data) + off, want.size());
  size_t diff = 0;
  for (size_t i = 0; i < want.size(); i++) diff += got[i] != want[i];
  char d[160];
  std::snprintf(d, sizeof d, "%zu of %zu bytes differ (%s, %s)", diff, want.size(), t.name.c_str(), layout_name(t.layout));
  result(diff == 0, what, d);
}

// An FP8 GEMV weight (L1) and its natural scales, or a bf16 one (L2, no scales), against a natural
// reference slice: kernel tiling == host tiling == file.
void bytes_dense(const char* what, const Npy& w, const Npy* s, const DeviceTensor& tw, const DeviceTensor* ts, int rows) {
  const int K = int(w.shape[1]) * (w.descr == "<u2" ? 2 : 1);  // bytes per row
  const uint8_t* nat = w.data.data();
  auto kt = kernel_tile8(nat, rows, K);
  std::vector<uint8_t> ht(kt.size());
  TileGeom g{uint64_t(rows), uint64_t(K), uint64_t(rows)};
  retile(Layout::L1, DType::F8_E4M3, g, nat, ht.data(), false);
  result(kt == ht, (std::string(what) + " host == kernel tiling").c_str(), "");
  if (tw.bytes != kt.size()) die("%s: %s has %llu bytes, the reference slice %zu", what, tw.name.c_str(),
                                 (unsigned long long)tw.bytes, kt.size());
  same_bytes(what, tw, 0, kt);
  if (s && ts) {
    const size_t sb = size_t(rows / 32) * (s->shape[1]);
    if (ts->bytes != sb) die("%s: %s has %llu bytes, expected %zu", what, ts->name.c_str(), (unsigned long long)ts->bytes, sb);
    same_bytes((std::string(what) + " scales").c_str(), *ts, 0, std::vector<uint8_t>(s->data.begin(), s->data.begin() + sb));
  }
}

// ---- part 2: GEMVs on the file's tensors, judged as engine/kernels/tests/test_kernels.cu does

// bf16 outputs: no element further than 1 bf16 ulp from the reference (or within fp32 noise of the
// row), and at least 98 % bit-exact.
void check_bf16(const char* what, const uint16_t* got, const uint16_t* exp, int rows, int cols, int ld_exp) {
  size_t exact = 0, n = 0, far = 0;
  for (int r = 0; r < rows; ++r) {
    double scale = 0;
    for (int c = 0; c < cols; ++c) scale = std::fmax(scale, std::fabs(bf2f(exp[size_t(r) * ld_exp + c])));
    for (int c = 0; c < cols; ++c) {
      uint16_t e = exp[size_t(r) * ld_exp + c], gt = got[size_t(r) * cols + c];
      double d = std::fabs(double(bf2f(gt)) - bf2f(e));
      double ulp = std::ldexp(1.0, int(std::floor(std::log2(std::fmax(std::fabs(double(bf2f(e))), 1e-30)))) - 7);
      exact += gt == e, ++n;
      far += d > ulp * 1.0001 && d > 2e-5 * scale;
    }
  }
  char d[96];
  std::snprintf(d, sizeof d, "exact %6.2f%%  far %zu", 100.0 * exact / n, far);
  result(far == 0 && exact >= 0.98 * n, what, d);
}

// fp32 outputs with a different summation order: max error relative to the row's largest value.
void check_f32(const char* what, const float* got, const float* exp, int rows, int cols, int ld_exp, double tol) {
  double max_rel = 0;
  for (int r = 0; r < rows; ++r) {
    double scale = 0;
    for (int c = 0; c < cols; ++c) scale = std::fmax(scale, std::fabs(exp[size_t(r) * ld_exp + c]));
    for (int c = 0; c < cols; ++c)
      max_rel = std::fmax(max_rel, std::fabs(double(got[size_t(r) * cols + c]) - exp[size_t(r) * ld_exp + c]) / scale);
  }
  char d[96];
  std::snprintf(d, sizeof d, "max|d|/rowmax %.3g (tol %.1g)", max_rel, tol);
  result(max_rel <= tol, what, d);
}

// fp32 sums of bf16 values: at least 95 % bit-exact, none off by more than far_rel of the row's maximum.
void check_f32_rows(const char* what, const float* got, const float* exp, int rows, int cols, double far_rel) {
  size_t exact = 0, far = 0, n = size_t(rows) * cols;
  for (int r = 0; r < rows; ++r) {
    double scale = 0;
    for (int c = 0; c < cols; ++c) scale = std::fmax(scale, std::fabs(exp[size_t(r) * cols + c]));
    for (int c = 0; c < cols; ++c) {
      const size_t i = size_t(r) * cols + c;
      exact += got[i] == exp[i];
      far += std::fabs(double(got[i]) - exp[i]) > far_rel * scale;
    }
  }
  char d[96];
  std::snprintf(d, sizeof d, "exact %6.2f%%  far %zu", 100.0 * exact / n, far);
  result(far == 0 && exact >= 0.95 * n, what, d);
}

void gemv_fp8_check(const char* ref, const DeviceTensor& W, const DeviceTensor& S) {
  Npy x = L(std::string(ref) + "_x"), y = L(std::string(ref) + "_y");
  const int N = int(W.shape[0]), K = int(W.shape[1]), M = int(x.shape[0]);
  auto* dx = dev_copy<__nv_bfloat16>(x.data.data(), x.data.size());
  auto *dq = dev_zero<uint8_t>(size_t(M) * K), *dqs = dev_zero<uint8_t>(size_t(M) * K / 32);
  auto* dy = dev_zero<__nv_bfloat16>(size_t(M) * N);
  dsk::act_quant(dx, M, K, dq, dqs, 0);
  for (int m : {8, 1})
    for (int mt : {1, 2, 4}) {
      dsk::GemvCfg cfg{mt, 1, 16};
      cfg.tiled = true;
      dsk::gemv_fp8(dq, dqs, m, static_cast<const uint8_t*>(W.data), static_cast<const uint8_t*>(S.data), N, K, dy, cfg, 0);
      auto hy = to_host<uint16_t>(dy, size_t(m) * N);
      char what[96];
      std::snprintf(what, sizeof what, "gemv_fp8 %s M=%d mt=%d", ref, m, mt);
      check_bf16(what, hy.data(), y.as<uint16_t>(), m, N, N);
    }
  cudaFree(dx), cudaFree(dq), cudaFree(dqs), cudaFree(dy);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) die("usage: dsw_kcheck FILE [REFDIR]");
  g_dir = argc > 2 ? argv[2] : std::string(getenv("HOME")) + "/dsgb10-data/kerneltests";
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  WeightFile wf(argv[1]);
  const Header& h = wf.header();
  if (h.version < 2 || h.rank != 0 || h.world != 4) die("%s: need a version-2 file of rank 0 of 4", argv[1]);
  const std::string P = "layers." + std::to_string(kLayer) + ".";
  LoadOptions opt;
  opt.select = [&](const DeviceTensor& t) {
    return t.name.rfind(P, 0) == 0 || t.name == "head.weight" || t.name.rfind("layers.2.attn.indexer.wq_b.", 0) == 0;
  };
  LoadStats ls = wf.load(opt);
  std::printf("%s: loaded %.2f GiB (layer %d, head, layer 2's indexer wq_b) in %.1f s; reference %s\n", argv[1],
              ls.bytes / 1073741824.0, kLayer, ls.seconds, g_dir.c_str());
  auto T = [&](const std::string& n) -> const DeviceTensor& { return wf[n]; };

  std::printf("bytes: the kernels' tiling of the reference slices == the file\n");
  struct Dense {
    const char* ref;
    std::string tensor;
  };
  const Dense dense[] = {
      {"wq_a", P + "attn.wq_a"},          {"wkv", P + "attn.wkv"},
      {"wq_b", P + "attn.wq_b"},          {"wo_b", P + "attn.wo_b"},
      {"sh_w1", P + "ffn.shared_experts.w1"}, {"sh_w2", P + "ffn.shared_experts.w2"},
      {"ix_wq_b", "layers.2.attn.indexer.wq_b"}, {"woa", P + "attn.wo_a"},
  };
  for (auto& d : dense) {
    Npy w = L(std::string(d.ref) + "_w"), s = L(std::string(d.ref) + "_s");
    bytes_dense(d.ref, w, &s, T(d.tensor + ".weight"), &T(d.tensor + ".scale"), int(w.shape[0]));
  }
  {
    Npy w = L("moe_shw3"), s = L("moe_shs3");
    bytes_dense("sh_w3", w, &s, T(P + "ffn.shared_experts.w3.weight"), &T(P + "ffn.shared_experts.w3.scale"),
                int(w.shape[0]));
  }
  Npy head_w = L("head_w"), gate_w = L("gate_w");
  bytes_dense("head (L2)", head_w, nullptr, T("head.weight"), nullptr, int(head_w.shape[0]));
  const int gate_rows = int(T(P + "ffn.gate.weight").shape[0]);  // rank 0's rows of the full reference gate
  bytes_dense("gate (L2)", gate_w, nullptr, T(P + "ffn.gate.weight"), nullptr, gate_rows);
  {
    Npy ids = L("moe_ids");
    const char* mats[3] = {"w1", "w3", "w2"};
    for (const char* m : mats) {
      Npy w = L(std::string("moe_") + m), s = L(std::string("moe_s") + (m + 1));
      const int E = int(w.shape[0]), N = int(w.shape[1]), K = int(w.shape[2]) * 2;
      const size_t wb = size_t(N) * K / 2, sb = size_t(N) * K / 32;
      const DeviceTensor &tw = T(P + "ffn.experts." + m + ".weight"), &ts = T(P + "ffn.experts." + m + ".scale");
      size_t bad = 0, host_bad = 0;
      for (int i = 0; i < E; i++) {
        std::vector<uint8_t> kw, ks, hw(wb), hs(sb);
        kernel_tile4(w.data.data() + i * wb, s.data.data() + i * sb, N, K, kw, ks);
        TileGeom g{uint64_t(N), uint64_t(K), uint64_t(N)};
        retile(Layout::L3, DType::E2M1X2, g, w.data.data() + i * wb, hw.data(), false);
        retile(Layout::L3, DType::F8_E8M0, g, s.data.data() + i * sb, hs.data(), false);
        host_bad += (hw != kw) + (hs != ks);
        const uint64_t e = uint64_t(ids.as<int>()[i]);
        auto fw = to_host<uint8_t>(static_cast<const uint8_t*>(tw.data) + e * wb, wb);
        auto fs = to_host<uint8_t>(static_cast<const uint8_t*>(ts.data) + e * sb, sb);
        bad += (fw != kw) + (fs != ks);
      }
      char what[64], d[96];
      std::snprintf(what, sizeof what, "experts %s (L3) host == kernel", m);
      result(host_bad == 0, what, "");
      std::snprintf(what, sizeof what, "experts %s (L3) values+scales", m);
      std::snprintf(d, sizeof d, "%zu of %d experts' tensors differ (%s)", bad, 2 * E, tw.name.c_str());
      result(bad == 0, what, d);
    }
  }
  {
    Npy nw = L("norm_w"), gb = L("gate_b");
    same_bytes("attn_norm (natural)", T(P + "attn_norm.weight"), 0, nw.data);
    same_bytes("gate bias (natural)", T(P + "ffn.gate.bias"), 0, gb.data);
  }

  std::printf("GEMVs on the file's tensors vs the reference outputs\n");
  for (auto& d : dense) {
    if (std::string(d.ref) == "woa") continue;
    gemv_fp8_check(d.ref, T(d.tensor + ".weight"), T(d.tensor + ".scale"));
  }
  {  // wo_a: block-diagonal over rank 0's 2 groups, FP8 widened to bf16
    Npy x = L("woa_x"), y = L("woa_y");
    const DeviceTensor &W = T(P + "attn.wo_a.weight"), &S = T(P + "attn.wo_a.scale");
    const int N = int(W.shape[0]), K = int(W.shape[1]), M = int(x.shape[0]), G = int(x.shape[1]) / K;
    auto* dx = dev_copy<__nv_bfloat16>(x.data.data(), x.data.size());
    auto* dy = dev_zero<__nv_bfloat16>(size_t(M) * N);
    for (int m : {8, 1})
      for (int mt : {1, 2, 4}) {
        dsk::GemvCfg cfg{mt, 1, 16};
        dsk::gemv_woa(dx, m, static_cast<const uint8_t*>(W.data), static_cast<const uint8_t*>(S.data), N, K, N / G, dy,
                      cfg, false, 0);
        auto hy = to_host<uint16_t>(dy, size_t(m) * N);
        char what[64];
        std::snprintf(what, sizeof what, "gemv_woa M=%d mt=%d", m, mt);
        check_bf16(what, hy.data(), y.as<uint16_t>(), m, N, N);
      }
    cudaFree(dx), cudaFree(dy);
  }
  {  // head and router gate: bf16 (L2), fp32 out
    Npy hx = L("head_x"), hy = L("head_y"), gx = L("gate_x"), gl = L("gate_logits");
    struct B {
      const char* what;
      const DeviceTensor& W;
      const Npy &x, &y;
    };
    for (const B& b : {B{"head", T("head.weight"), hx, hy}, B{"gate", T(P + "ffn.gate.weight"), gx, gl}}) {
      const int N = int(b.W.shape[0]), K = int(b.W.shape[1]), M = int(b.x.shape[0]), ld = int(b.y.shape[1]);
      auto* dx = dev_copy<__nv_bfloat16>(b.x.data.data(), b.x.data.size());
      auto* dy = dev_zero<float>(size_t(M) * N);
      for (int m : {8, 1})
        for (int mt : {1, 2}) {
          dsk::GemvCfg cfg{mt, 1, 16};
          dsk::gemv_bf16(dx, m, static_cast<const uint8_t*>(b.W.data), N, K, dy, nullptr, cfg, false, 0);
          auto got = to_host<float>(dy, size_t(m) * N);
          char what[64];
          std::snprintf(what, sizeof what, "gemv_bf16 %s M=%d mt=%d", b.what, m, mt);
          check_f32(what, got.data(), b.y.as<float>(), m, N, ld, 2e-6);  // rank 0's columns [0, N)
        }
      cudaFree(dx), cudaFree(dy);
    }
  }
  {  // MoE: the reference's 12 experts by their global ids in the file's stacks, plus the shared expert
    Npy x = L("moe_x"), ids = L("moe_ids"), sel = L("moe_sel"), rw = L("moe_rw"), yr = L("moe_y_routed"),
        ys = L("moe_y_shared");
    const int M8 = int(x.shape[0]), D = int(x.shape[1]);
    std::vector<int> idx(sel.count());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = ids.as<int>()[sel.as<int>()[i]];
    auto ptr = [&](const std::string& n) { return static_cast<const uint8_t*>(T(P + n).data); };
    dsk::MoeWeights mw{ptr("ffn.experts.w1.weight"), ptr("ffn.experts.w1.scale"), ptr("ffn.experts.w3.weight"),
                       ptr("ffn.experts.w3.scale"), ptr("ffn.experts.w2.weight"), ptr("ffn.experts.w2.scale"),
                       ptr("ffn.shared_experts.w1.weight"), ptr("ffn.shared_experts.w1.scale"),
                       ptr("ffn.shared_experts.w3.weight"), ptr("ffn.shared_experts.w3.scale"),
                       ptr("ffn.shared_experts.w2.weight"), ptr("ffn.shared_experts.w2.scale")};
    auto* dx = dev_copy<__nv_bfloat16>(x.data.data(), x.data.size());
    auto *dq = dev_zero<uint8_t>(size_t(M8) * D), *dqs = dev_zero<uint8_t>(size_t(M8) * D / 32);
    dsk::act_quant(dx, M8, D, dq, dqs, 0);
    int* didx = dev_copy<int>(idx.data(), idx.size() * sizeof(int));
    float* dwts = dev_copy<float>(rw.data.data(), rw.data.size());
    auto* ws = dev_zero<uint8_t>(dsk::moe_workspace_bytes());
    auto* dy = dev_zero<float>(size_t(M8) * D);
    std::vector<float> want(size_t(M8) * D);
    for (size_t i = 0; i < want.size(); i++) want[i] = yr.as<float>()[i] + ys.as<float>()[i];
    for (int m : {8, 6, 1}) {
      dsk::moe(dq, dqs, m, didx, dwts, mw, ws, dy, dsk::MoeCfg{}, 0);
      auto got = to_host<float>(dy, size_t(m) * D);
      char what[64];
      std::snprintf(what, sizeof what, "moe (12 experts + shared) M=%d", m);
      check_f32_rows(what, got.data(), want.data(), m, D, 1.0 / 64);
    }
    cudaFree(dx), cudaFree(dq), cudaFree(dqs), cudaFree(didx), cudaFree(dwts), cudaFree(ws), cudaFree(dy);
  }
  {  // a natural tensor through a kernel too: RMSNorm with the file's attn_norm
    Npy x = L("norm_x"), y = L("norm_y");
    const int M8 = int(x.shape[0]), D = int(x.shape[1]);
    auto* dx = dev_copy<__nv_bfloat16>(x.data.data(), x.data.size());
    auto* dy = dev_zero<__nv_bfloat16>(size_t(M8) * D);
    dsk::rmsnorm(dx, static_cast<const __nv_bfloat16*>(T(P + "attn_norm.weight").data), M8, D, 1e-20f, dy, nullptr,
                 nullptr, 0);
    auto got = to_host<uint16_t>(dy, size_t(M8) * D);
    check_bf16("rmsnorm attn_norm", got.data(), y.as<uint16_t>(), M8, D, D);
    cudaFree(dx), cudaFree(dy);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf(g_fail ? "%d FAILED\n" : "all passed: the file's layouts are what the kernels read\n", g_fail);
  return g_fail != 0;
}
