// Correctness of the decode kernels against reference data from tests/gen_ref.py.
// usage: test_kernels [data dir, default ~/dsgb10-data/kerneltests ("" for it)]
//                     [only: act_quant|fp8|<fp8 name>|moe|mcode|woa|head|head4|norm|normgemv|router|rows16|prefill|p_fp8|p_small|p_16|p_moe]
#include <algorithm>
#include <string>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kernels/dsk.h"
#include "kernels/tests/npy.h"

static std::string g_dir;
#include <functional>
static std::function<void(int, int, float, float)> g_detail;  // called for elements >1 ulp off
#include "kernels/tests/small_gemms.h"
static int g_fail = 0;

static Npy L(const std::string& name) { return npy_load(g_dir + "/" + name + ".npy"); }

template <class T = uint8_t>
static T* dev(const Npy& n) {
  void* p;
  CUDA_CHECK(cudaMalloc(&p, n.data.size()));
  CUDA_CHECK(cudaMemcpy(p, n.data.data(), n.data.size(), cudaMemcpyHostToDevice));
  return static_cast<T*>(p);
}
template <class T>
static T* dev_alloc(size_t n) {
  void* p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
  return static_cast<T*>(p);
}
template <class T>
static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// got vs expected, both bf16 bits, over `rows` x `cols` (expected has stride ld_exp).
static void check_bf16(const char* what, const uint16_t* got, const uint16_t* exp, int rows, int cols, int ld_exp) {
  size_t exact = 0, n = 0, over1ulp = 0;
  double max_rel = 0, max_abs = 0;
  for (int r = 0; r < rows; ++r) {
    double scale = 0;
    for (int c = 0; c < cols; ++c) scale = fmax(scale, fabs(bf2f(exp[(size_t)r * ld_exp + c])));
    for (int c = 0; c < cols; ++c) {
      uint16_t e = exp[(size_t)r * ld_exp + c], gt = got[(size_t)r * cols + c];
      double d = fabs((double)bf2f(gt) - bf2f(e));
      exact += gt == e;
      ++n;
      // within 1 bf16 ulp of the reference, or within fp32 accumulation noise of the row
      double ulp = ldexp(1.0, (int)floor(log2(fmax(fabs((double)bf2f(e)), 1e-30))) - 7);
      bool far = d > ulp * 1.0001 && d > 2e-5 * scale;
      over1ulp += far;
      if (far && g_detail) g_detail(r, c, bf2f(gt), bf2f(e));
      max_abs = fmax(max_abs, d);
      max_rel = fmax(max_rel, d / fmax(scale, 1e-30));
    }
  }
  bool ok = over1ulp == 0 && exact >= 0.98 * n;
  g_fail += !ok;
  printf("  %-28s %s exact %6.2f%%  far %zu  max|d| %.3g  max|d|/rowmax %.3g\n", what, ok ? "ok  " : "FAIL",
         100.0 * exact / n, over1ulp, max_abs, max_rel);
}

// fp32 results whose summation order differs from the reference: max error relative to the row's
// largest value, plus the bit-exact share.
static void check_f32(const char* what, const float* got, const float* exp, int rows, int cols, double tol_rel) {
  double max_rel = 0;
  size_t exact = 0;
  for (int r = 0; r < rows; ++r) {
    double scale = 0;
    for (int c = 0; c < cols; ++c) scale = fmax(scale, fabs(exp[(size_t)r * cols + c]));
    for (int c = 0; c < cols; ++c) {
      size_t i = (size_t)r * cols + c;
      max_rel = fmax(max_rel, fabs((double)got[i] - exp[i]) / scale);
      exact += got[i] == exp[i];
    }
  }
  bool ok = max_rel <= tol_rel;
  g_fail += !ok;
  printf("  %-28s %s exact %6.2f%%  max|d|/rowmax %.3g (tol %.1g)\n", what, ok ? "ok  " : "FAIL",
         100.0 * exact / ((size_t)rows * cols), max_rel, tol_rel);
}

// Informational bf16 comparison (does not fail the run).
static void compare_bf16(const char* what, const uint16_t* got, const uint16_t* exp, size_t n) {
  size_t exact = 0;
  double max_d = 0;
  for (size_t i = 0; i < n; ++i) exact += got[i] == exp[i], max_d = fmax(max_d, fabs((double)bf2f(got[i]) - bf2f(exp[i])));
  printf("  %-28s info exact %6.2f%%  max|d| %.3g\n", what, 100.0 * exact / n, max_d);
}

// fp32 outputs: bit-exact share, and "far" = off by more than far_rel of the row's largest value.
static void check_f32_rows(const char* what, const float* got, const float* exp, int rows, int cols, double far_rel) {
  size_t exact = 0, far = 0, n = (size_t)rows * cols;
  double max_abs = 0, max_rel = 0;
  for (int r = 0; r < rows; ++r) {
    double scale = 0;
    for (int c = 0; c < cols; ++c) scale = fmax(scale, fabs(exp[(size_t)r * cols + c]));
    for (int c = 0; c < cols; ++c) {
      double d = fabs((double)got[(size_t)r * cols + c] - exp[(size_t)r * cols + c]);
      exact += got[(size_t)r * cols + c] == exp[(size_t)r * cols + c];
      far += d > far_rel * scale;
      max_abs = fmax(max_abs, d);
      max_rel = fmax(max_rel, d / fmax(scale, 1e-30));
    }
  }
  bool ok = far == 0 && exact >= 0.95 * n;
  g_fail += !ok;
  printf("  %-28s %s exact %6.2f%%  far %zu  max|d| %.3g  max|d|/rowmax %.3g\n", what, ok ? "ok  " : "FAIL",
         100.0 * exact / n, far, max_abs, max_rel);
}

static double e4m3_val(uint8_t b) {
  int s = b >> 7, e = (b >> 3) & 15, f = b & 7;
  double v = e ? (1 + f / 8.0) * ldexp(1, e - 7) : f / 8.0 * ldexp(1, -6);
  return s ? -v : v;
}
static double e2m1_val(uint8_t nib) {
  static const double t[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  return (nib & 8) ? -t[nib & 7] : t[nib & 7];
}
static float bf16_round(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
  memcpy(&x, &u, 4);
  return x;
}

static void test_act_quant() {
  printf("act_quant\n");
  Npy x = L("aq_x"), q = L("aq_q"), s = L("aq_s");
  int M = x.shape[0], K = x.shape[1];
  auto* dx = dev<__nv_bfloat16>(x);
  auto* dq = dev_alloc<uint8_t>(M * K);
  auto* ds = dev_alloc<uint8_t>(M * K / 32);
  dsk::act_quant(dx, M, K, dq, ds, 0);
  auto hq = host(dq, M * K), hs = host(ds, M * K / 32);
  size_t bad_q = 0, bad_s = 0;
  for (int i = 0; i < M * K; ++i) bad_q += hq[i] != q.as<uint8_t>()[i];
  for (int i = 0; i < M * K / 32; ++i) bad_s += hs[i] != s.as<uint8_t>()[i];
  bool ok = !bad_q && !bad_s;
  g_fail += !ok;
  printf("  %-28s %s mismatched values %zu / %d, scales %zu / %d\n", "bit-exact vs reference", ok ? "ok  " : "FAIL",
         bad_q, M * K, bad_s, M * K / 32);
  cudaFree(dx), cudaFree(dq), cudaFree(ds);
}

static void test_gemv_fp8(const std::string& name) {
  Npy w = L(name + "_w"), s = L(name + "_s"), x = L(name + "_x"), y = L(name + "_y");
  int N = w.shape[0], K = w.shape[1], M = x.shape[0];
  printf("gemv_fp8 %s [%d x %d]\n", name.c_str(), N, K);
  auto *dw = dev(w), *ds = dev(s);
  auto* dx = dev<__nv_bfloat16>(x);
  auto* dq = dev_alloc<uint8_t>(M * K);
  auto* dqs = dev_alloc<uint8_t>(M * K / 32);
  auto* dy = dev_alloc<__nv_bfloat16>((size_t)M * N);
  dsk::act_quant(dx, M, K, dq, dqs, 0);
  const uint16_t* ye = y.as<uint16_t>();
  auto hq = host(dq, (size_t)M * K), hqs = host(dqs, (size_t)M * K / 32);
  g_detail = [&](int m, int n, float got, float exp) {
    double acc = 0;  // exact: products of e4m3 values times power-of-two scales, summed in double
    for (int k = 0; k < K; ++k) {
      auto e4 = [](uint8_t b) { int s = b >> 7, e = (b >> 3) & 15, f = b & 7; double v = e ? (1 + f / 8.0) * ldexp(1, e - 7) : f / 8.0 * ldexp(1, -6); return s ? -v : v; };
      double a = e4(hq[(size_t)m * K + k]) * ldexp(1, hqs[(size_t)m * (K / 32) + k / 32] - 127);
      double wv = e4(w.as<uint8_t>()[(size_t)n * K + k]) * ldexp(1, s.as<uint8_t>()[(size_t)(n / 32) * (K / 32) + k / 32] - 127);
      acc += a * wv;
    }
    printf("    m=%d n=%d got %.6g ref %.6g exact %.9g\n", m, n, got, exp, acc);
  };
  for (int m : {8, 1}) {
    for (int mt : {1, 2, 4}) {
      dsk::GemvCfg cfg{mt, 1, 16};
      dsk::gemv_fp8(dq, dqs, m, dw, ds, N, K, dy, cfg, 0);
      auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)m * N);
      char what[64];
      snprintf(what, sizeof what, "mma M=%d mt=%d", m, mt);
      check_bf16(what, hy.data(), ye, m, N, N);
    }
    {
      uint8_t* dwt = dev_alloc<uint8_t>((size_t)N * K);
      dsk::tile_fp8(dw, dwt, N, K, 0);
      dsk::GemvCfg cfg{1, 1, 16};
      cfg.tiled = true;
      dsk::gemv_fp8(dq, dqs, m, dwt, ds, N, K, dy, cfg, 0);
      auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)m * N);
      char what[64];
      snprintf(what, sizeof what, "mma tiled M=%d", m);
      check_bf16(what, hy.data(), ye, m, N, N);
      cudaFree(dwt);
    }
    dsk::gemv_fp8_cc(dq, dqs, m, dw, ds, N, K, dy, 0);
    auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)m * N);
    char what[64];
    snprintf(what, sizeof what, "cuda-core M=%d", m);
    check_bf16(what, hy.data(), ye, m, N, N);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  g_detail = nullptr;
  cudaFree(dw), cudaFree(ds), cudaFree(dx), cudaFree(dq), cudaFree(dqs), cudaFree(dy);
}

static void test_moe() {
  Npy x = L("moe_x"), sel = L("moe_sel"), rw = L("moe_rw");
  Npy w1 = L("moe_w1"), s1 = L("moe_s1"), w3 = L("moe_w3"), s3 = L("moe_s3"), w2 = L("moe_w2"), s2 = L("moe_s2");
  Npy shw1 = L("moe_shw1"), shs1 = L("moe_shs1"), shw3 = L("moe_shw3"), shs3 = L("moe_shs3");
  Npy shw2 = L("moe_shw2"), shs2 = L("moe_shs2"), yr = L("moe_y_routed"), ysh = L("moe_y_shared");
  const int E = w1.shape[0], I = w1.shape[1], D = x.shape[1], M8 = x.shape[0];
  printf("moe: %d routed experts (layer 3, rank 0 slices) + shared expert\n", E);
  auto tile4 = [&](const Npy& w, const Npy& s, int N, int K, const uint8_t*& wt, const uint8_t*& st) {
    uint8_t *dw = dev(w), *ds = dev(s), *t = dev_alloc<uint8_t>(w.data.size()), *ts = dev_alloc<uint8_t>(s.data.size());
    dsk::tile_fp4(dw, ds, t, ts, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw), cudaFree(ds);
    wt = t, st = ts;
  };
  auto tile8 = [&](const Npy& w, int N, int K) {
    uint8_t *dw = dev(w), *t = dev_alloc<uint8_t>(w.data.size());
    dsk::tile_fp8(dw, t, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw);
    return (const uint8_t*)t;
  };
  dsk::MoeWeights mw{};
  tile4(w1, s1, E * I, D, mw.w1, mw.s1);
  tile4(w3, s3, E * I, D, mw.w3, mw.s3);
  tile4(w2, s2, E * D, I, mw.w2, mw.s2);
  mw.sh_w1 = tile8(shw1, I, D), mw.sh_s1 = dev(shs1);
  mw.sh_w3 = tile8(shw3, I, D), mw.sh_s3 = dev(shs3);
  mw.sh_w2 = tile8(shw2, D, I), mw.sh_s2 = dev(shs2);
  auto* dx = dev<__nv_bfloat16>(x);
  auto* dq = dev_alloc<uint8_t>((size_t)M8 * D);
  auto* dqs = dev_alloc<uint8_t>((size_t)M8 * D / 32);
  dsk::act_quant(dx, M8, D, dq, dqs, 0);
  int* didx = dev<int>(sel);
  float* dwts = dev<float>(rw);
  uint8_t* ws = dev_alloc<uint8_t>(dsk::moe_workspace_bytes());
  float* dy = dev_alloc<float>((size_t)M8 * D);
  std::vector<float> yexp((size_t)M8 * D);
  for (size_t i = 0; i < yexp.size(); ++i) yexp[i] = yr.as<float>()[i] + ysh.as<float>()[i];

  auto run = [&](int m, dsk::MoeCfg cfg) {
    CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)M8 * D * 4));
    dsk::moe(dq, dqs, m, didx, dwts, mw, ws, dy, cfg, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto hy = host(dy, (size_t)m * D);
    char what[64];
    snprintf(what, sizeof what, "M=%d up %d down %d", m, cfg.up, cfg.down);
    check_f32_rows(what, hy.data(), yexp.data(), m, D, 1.0 / 64);
  };
  for (int m : {8, 6, 1}) {
    for (int up = 0; up < 10; ++up) run(m, {up, 0});
    for (int down = 1; down < 5; ++down) run(m, {0, down});
  }

  // moe_up's task queue (dyn) against the static schedule, bit for bit: y and the intermediate
  {
    const size_t wsb = dsk::moe_workspace_bytes(), hqb = wsb - 2 * sizeof(int);  // hq, hs; then the counters
    int* didx2 = dev_alloc<int>(48);
    float* dwts2 = dev_alloc<float>(48);
    float* dy2 = dev_alloc<float>((size_t)M8 * D);
    auto once = [&](int m, dsk::MoeCfg cfg, float* yo) {
      CUDA_CHECK(cudaMemset(ws, 0xAB, hqb));  // the counters (zero) stay
      CUDA_CHECK(cudaMemset(yo, 0xFF, (size_t)M8 * D * 4));
      dsk::moe(dq, dqs, m, didx2, dwts2, mw, ws, yo, cfg, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      return std::make_pair(host(yo, (size_t)m * D), host(ws, wsb));
    };
    const char* rname[] = {"reference picks", "all rows one set", "all rows distinct"};
    for (int routing = 0; routing < 3; ++routing)
      for (int m : {1, 2, 3, 6, 8}) {
        std::vector<int> hi(48);
        std::vector<float> hw(48);
        for (int r = 0; r < m; ++r)
          for (int k = 0; k < 6; ++k) {
            const int i = r * 6 + k;
            hi[i] = routing == 0 ? sel.as<int>()[i] : routing == 1 ? sel.as<int>()[k] : (i * 7 + 3) % E;
            hw[i] = rw.as<float>()[i];
          }
        if (routing == 2)  // distinct ids within each row too (the router never repeats one)
          for (int r = 0; r < m; ++r) std::sort(hi.begin() + r * 6, hi.begin() + r * 6 + 6);
        CUDA_CHECK(cudaMemcpy(didx2, hi.data(), 48 * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dwts2, hw.data(), 48 * 4, cudaMemcpyHostToDevice));
        size_t runs = 0, bad = 0;
        for (int up : {0, 2, 6})
          for (int grid : {0, 1, 7, 13, 48, 200}) {
            dsk::MoeCfg cs{up, 0};
            cs.up_grid = grid, cs.dyn = false;
            dsk::MoeCfg cd = cs;
            cd.dyn = true;
            const auto a = once(m, cs, dy2);
            for (int rep = 0; rep < 2; ++rep) {  // twice: the queue's counters must be back at zero
              const auto b = once(m, cd, dy2);
              ++runs;
              bad += memcmp(a.first.data(), b.first.data(), a.first.size() * 4) != 0 || a.second != b.second;
            }
          }
        g_fail += bad != 0;
        printf("  queue == static, %-18s M=%d  %s %zu runs (up 0/2/6, grids 48/1/7/13/48/200)%s\n", rname[routing], m,
               bad ? "FAIL" : "ok  ", runs, bad ? ", outputs or intermediate DIFFER" : "");
      }
    cudaFree(didx2), cudaFree(dwts2), cudaFree(dy2);
  }

  // moe_up's intermediate (quantised SwiGLU output) for M=8 against an fp64 host reference
  run(8, {0, 0});
  auto hq = host(ws, dsk::moe_workspace_bytes());
  const uint8_t* hs = hq.data() + dsk::moe_hs_offset(8);
  auto xq = host(dq, (size_t)M8 * D), xs = host(dqs, (size_t)M8 * D / 32);
  std::vector<int> uni;
  for (size_t i = 0; i < sel.count(); ++i) uni.push_back(sel.as<int>()[i]);
  std::sort(uni.begin(), uni.end());
  uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
  size_t nq = 0, bad_q = 0, ns = 0, bad_s = 0;
  std::vector<double> xd((size_t)M8 * D);
  for (int m = 0; m < M8; ++m)
    for (int k = 0; k < D; ++k) xd[(size_t)m * D + k] = e4m3_val(xq[(size_t)m * D + k]) * ldexp(1, xs[(size_t)m * (D / 32) + k / 32] - 127);
  for (size_t u = 0; u < uni.size(); ++u) {
    const int e = uni[u];
    for (int m = 0; m < M8; ++m) {
      int pick = -1;
      for (int k = 0; k < 6; ++k) pick = sel.as<int>()[m * 6 + k] == e ? k : pick;
      if (pick < 0) continue;
      float h[576];
      for (int j = 0; j < I; ++j) {
        double acc[2] = {0, 0};
        const Npy* W[2] = {&w1, &w3};
        const Npy* S[2] = {&s1, &s3};
        for (int f = 0; f < 2; ++f) {
          const uint8_t* wr = W[f]->as<uint8_t>() + ((size_t)e * I + j) * (D / 2);
          const uint8_t* sr = S[f]->as<uint8_t>() + ((size_t)e * I + j) * (D / 32);
          for (int k = 0; k < D; ++k)
            acc[f] += xd[(size_t)m * D + k] * e2m1_val(k & 1 ? wr[k / 2] >> 4 : wr[k / 2] & 15) * ldexp(1, sr[k / 32] - 127);
        }
        float gt = bf16_round((float)acc[0]), up = bf16_round((float)acc[1]);
        up = fminf(fmaxf(up, -10.f), 10.f);
        gt = fminf(gt, 10.f);
        float v = gt / (1.0f + expf(-gt)) * up;
        h[j] = bf16_round(rw.as<float>()[m * 6 + pick] * v);
      }
      for (int jb = 0; jb < I / 32; ++jb) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(h[32 * jb + j]));
        float v = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
        uint32_t b;
        memcpy(&b, &v, 4);
        uint32_t eb = ((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0);
        ++ns;
        bad_s += hs[(u * 8 + m) * (I / 32) + jb] != eb;
        for (int j = 0; j < 32; ++j) {
          double q = e4m3_val(hq[(u * 8 + m) * I + 32 * jb + j]) * ldexp(1, (int)eb - 127);
          ++nq;
          // the stored e4m3 value must be the nearest to h (ties may go either way)
          double err = fabs(q - h[32 * jb + j]), step = ldexp(1, (int)floor(log2(fmax(fabs(h[32 * jb + j]), 1e-30))) - 3);
          bad_q += err > 0.5001 * fmax(step, ldexp(1, (int)eb - 127 - 9));
        }
      }
    }
  }
  bool ok = bad_q * 1000 <= nq && bad_s * 100 <= ns;
  g_fail += !ok;
  printf("  %-28s %s values off %zu / %zu, scales off %zu / %zu (vs fp64 host)\n", "moe_up output M=8", ok ? "ok  " : "FAIL",
         bad_q, nq, bad_s, ns);

  // moe_routed (top-6 from logits inside the MoE kernels) == router_topk then moe, bit for bit.
  // Synthetic logits over the 12 test experts, with ties in the keys to exercise the tie rule.
  std::vector<float> hl((size_t)M8 * E), hb(E);
  uint32_t r = 12345;
  auto rnd = [&]() { r = r * 1664525u + 1013904223u; return (float)(r >> 8) / (float)(1u << 24); };
  for (auto& v : hl) v = 6.f * rnd() - 3.f;
  for (auto& v : hb) v = 0.2f * rnd() - 0.1f;
  hl[3] = hl[5], hb[3] = hb[5];  // row 0: experts 3 and 5 tie
  float *dl = dev_alloc<float>(hl.size()), *db = dev_alloc<float>(E), *dw1 = dev_alloc<float>(M8 * 6), *dw2 = dev_alloc<float>(M8 * 6);
  int *di1 = dev_alloc<int>(M8 * 6), *di2 = dev_alloc<int>(M8 * 6);
  float* dy2 = dev_alloc<float>((size_t)M8 * D);
  CUDA_CHECK(cudaMemcpy(dl, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
  for (int m : {8, 6, 1})
    for (int down : {2, 4}) {
      dsk::MoeCfg cfg{};
      cfg.down = down;
      dsk::router_topk(dl, db, m, E, di1, dw1, 1.5f, 0);
      dsk::moe(dq, dqs, m, di1, dw1, mw, ws, dy, cfg, 0);
      dsk::moe_routed(dq, dqs, m, dl, db, E, 1.5f, mw, ws, dy2, cfg, 0, di2, dw2);
      const bool same_sel = host(di1, m * 6) == host(di2, m * 6) && host(dw1, m * 6) == host(dw2, m * 6);
      auto a = host(dy, (size_t)m * D), b = host(dy2, (size_t)m * D);
      size_t diff = 0;
      for (size_t j = 0; j < a.size(); ++j) diff += memcmp(&a[j], &b[j], 4) != 0;
      const bool ok2 = same_sel && diff == 0;
      g_fail += !ok2;
      char what[64];
      snprintf(what, sizeof what, "moe_routed M=%d down %d", m, down);
      printf("  %-28s %s outputs differing %zu, selection %s\n", what, ok2 ? "ok  " : "FAIL", diff,
             same_sel ? "equal" : "DIFFERS");
    }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(dl), cudaFree(db), cudaFree(dw1), cudaFree(dw2), cudaFree(di1), cudaFree(di2), cudaFree(dy2);
}

static void test_woa() {
  Npy w = L("woa_w"), sc = L("woa_s"), x = L("woa_x"), y = L("woa_y"), ye = L("woa_y_einsum");
  const int N = w.shape[0], K = w.shape[1], M8 = x.shape[0], G = x.shape[1] / K;
  printf("gemv_woa [%d x %d], %d groups of %d rows\n", N, K, G, N / G);
  uint8_t *dw = dev(w), *ds = dev(sc), *dwt = dev_alloc<uint8_t>(w.data.size());
  dsk::tile_fp8(dw, dwt, N, K, 0);
  auto* dx = dev<__nv_bfloat16>(x);
  auto* dy = dev_alloc<__nv_bfloat16>((size_t)M8 * N);
  for (int m : {8, 1})
    for (int cc = 0; cc < 2; ++cc)
      for (int mt : {1, 2, 4}) {
        if (cc && mt == 4) continue;
        dsk::GemvCfg cfg{mt, 1, 16};
        dsk::gemv_woa(dx, m, dwt, ds, N, K, N / G, dy, cfg, cc, 0);
        auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)m * N);
        char what[64];
        snprintf(what, sizeof what, "%s M=%d mt=%d", cc ? "cuda-core" : "mma", m, mt);
        check_bf16(what, hy.data(), y.as<uint16_t>(), m, N, N);
        if (mt == 1) {
          snprintf(what, sizeof what, "  vs torch bf16 einsum");
          compare_bf16(what, hy.data(), ye.as<uint16_t>(), (size_t)m * N);
        }
      }
  {
    char what[64] = "fp32-math ref vs einsum";
    compare_bf16(what, y.as<uint16_t>(), ye.as<uint16_t>(), (size_t)M8 * N);
  }
  {  // fused act_quant of the output (wo_b's input) == act_quant of the bf16 output
    auto *dq = dev_alloc<uint8_t>((size_t)M8 * N), *dqs = dev_alloc<uint8_t>((size_t)M8 * N / 32);
    auto *dq2 = dev_alloc<uint8_t>((size_t)M8 * N), *dqs2 = dev_alloc<uint8_t>((size_t)M8 * N / 32);
    dsk::GemvCfg cfg{1, 4, 4};
    dsk::gemv_woa(dx, M8, dwt, ds, N, K, N / G, dy, cfg, false, 0, dq, dqs);
    auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)M8 * N);
    check_bf16("mma M=8 + fused act_quant", hy.data(), y.as<uint16_t>(), M8, N, N);
    dsk::act_quant(dy, M8, N, dq2, dqs2, 0);
    bool ok = host(dq, (size_t)M8 * N) == host(dq2, (size_t)M8 * N) && host(dqs, (size_t)M8 * N / 32) == host(dqs2, (size_t)M8 * N / 32);
    g_fail += !ok;
    printf("  %-28s %s\n", "fused quant == act_quant(y)", ok ? "ok  " : "FAIL");
    cudaFree(dq), cudaFree(dqs), cudaFree(dq2), cudaFree(dqs2);
  }
  cudaFree(dw), cudaFree(ds), cudaFree(dwt), cudaFree(dx), cudaFree(dy);
}

static void test_head() {
  Npy w = L("head_w"), x = L("head_x"), y = L("head_y");
  const int N = w.shape[0], K = w.shape[1], M8 = x.shape[0];
  printf("gemv_bf16 head [%d x %d], fp32 logits\n", N, K);
  uint8_t *dw = dev(w), *dwt = dev_alloc<uint8_t>(w.data.size());
  dsk::tile_fp8(dw, dwt, N, 2 * K, 0);
  cudaFree(dw);
  auto* dx = dev<__nv_bfloat16>(x);
  float* dy = dev_alloc<float>((size_t)M8 * N);
  for (int m : {8, 1})
    for (int cc = 0; cc < 2; ++cc)
      for (int mt : {1, 2}) {
        dsk::GemvCfg cfg{mt, 1, 16};
        dsk::gemv_bf16(dx, m, dwt, N, K, dy, nullptr, cfg, cc, 0);
        auto hy = host(dy, (size_t)m * N);
        char what[64];
        snprintf(what, sizeof what, "%s M=%d mt=%d", cc ? "cuda-core" : "mma", m, mt);
        check_f32(what, hy.data(), y.as<float>(), m, N, 2e-6);
        if (m == 1 && mt == 1) {  // both against an fp64 dot product, first 4096 logits of row 0
          double e_ours = 0, e_ref = 0, scale = 0;
          for (int n = 0; n < 4096; ++n) {
            double ex = 0;
            for (int k = 0; k < K; ++k)
              ex += (double)bf2f(x.as<uint16_t>()[k]) * bf2f(w.as<uint16_t>()[(size_t)n * K + k]);
            e_ours = fmax(e_ours, fabs(hy[n] - ex)), e_ref = fmax(e_ref, fabs(y.as<float>()[n] - ex));
            scale = fmax(scale, fabs(ex));
          }
          printf("    vs fp64: max err/rowmax ours %.3g, torch's %.3g\n", e_ours / scale, e_ref / scale);
        }
      }
  cudaFree(dwt), cudaFree(dx), cudaFree(dy);
}

static void test_norm() {
  Npy w = L("norm_w"), x = L("norm_x"), y = L("norm_y");
  const int D = w.shape[0], M8 = x.shape[0];
  printf("rmsnorm [%d x %d] (+ fused act_quant)\n", M8, D);
  auto *dw = dev<__nv_bfloat16>(w), *dx = dev<__nv_bfloat16>(x);
  auto* dy = dev_alloc<__nv_bfloat16>((size_t)M8 * D);
  auto *dq = dev_alloc<uint8_t>((size_t)M8 * D), *ds = dev_alloc<uint8_t>((size_t)M8 * D / 32);
  auto *dq2 = dev_alloc<uint8_t>((size_t)M8 * D), *ds2 = dev_alloc<uint8_t>((size_t)M8 * D / 32);
  dsk::rmsnorm(dx, dw, M8, D, 1e-20f, dy, dq, ds, 0);
  auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)M8 * D);
  check_bf16("bf16 out", hy.data(), y.as<uint16_t>(), M8, D, D);
  dsk::act_quant(dy, M8, D, dq2, ds2, 0);
  auto q1 = host(dq, (size_t)M8 * D), q2 = host(dq2, (size_t)M8 * D);
  auto s1 = host(ds, (size_t)M8 * D / 32), s2 = host(ds2, (size_t)M8 * D / 32);
  bool ok = q1 == q2 && s1 == s2;
  g_fail += !ok;
  printf("  %-28s %s\n", "fused quant == act_quant(y)", ok ? "ok  " : "FAIL");
  cudaFree(dw), cudaFree(dx), cudaFree(dy), cudaFree(dq), cudaFree(ds), cudaFree(dq2), cudaFree(ds2);
}

// RMSNorm fused into the GEMV prologues: must equal rmsnorm(+quant) followed by the GEMV bit for bit
// (outputs and CTA 0's copies of the normed activations), so it inherits their match with the
// reference. Shapes: attn_norm -> wq_a+wkv (K 5120), q_norm -> wq_b (K 1280), ffn_norm -> gate.
static void test_norm_gemv() {
  Npy nw = L("norm_w"), nx = L("norm_x");
  const int M8 = nx.shape[0];
  printf("rmsnorm fused into gemv_fp8 (== rmsnorm + gemv_fp8)\n");
  struct Shape { int K; std::vector<std::string> mats; };
  for (const Shape& sh : {Shape{5120, {"wq_a", "wkv"}}, Shape{1280, {"wq_b"}}}) {
    const int K = sh.K, nseg = (int)sh.mats.size();
    std::vector<uint16_t> hx((size_t)M8 * K);  // the first K columns of the norm test rows
    for (int m = 0; m < M8; ++m) memcpy(&hx[(size_t)m * K], nx.as<uint16_t>() + (size_t)m * 5120, K * 2);
    __nv_bfloat16 *dx = dev_alloc<__nv_bfloat16>((size_t)M8 * K), *dw = dev<__nv_bfloat16>(nw);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    dsk::Fp8Seg seg[2], seg2[2];
    std::vector<void*> frees;
    int Ntot = 0;
    for (int i = 0; i < nseg; ++i) {
      Npy w = L(sh.mats[i] + "_w"), sc = L(sh.mats[i] + "_s");
      const int N = w.shape[0];
      if (w.shape[1] != K) {
        printf("  %s has K=%d, expected %d\n", sh.mats[i].c_str(), (int)w.shape[1], K);
        ++g_fail;
        return;
      }
      uint8_t *dwn = dev(w), *dwt = dev_alloc<uint8_t>((size_t)N * K), *ds = dev(sc);
      dsk::tile_fp8(dwn, dwt, N, K, 0);
      seg[i] = {dwt, ds, dev_alloc<__nv_bfloat16>((size_t)M8 * N), N};
      seg2[i] = {dwt, ds, dev_alloc<__nv_bfloat16>((size_t)M8 * N), N};
      frees.insert(frees.end(), {dwn, dwt, ds, seg[i].y, seg2[i].y});
      Ntot += N;
    }
    auto* y1 = dev_alloc<__nv_bfloat16>((size_t)M8 * K);
    auto* y2 = dev_alloc<__nv_bfloat16>((size_t)M8 * K);
    auto *q1 = dev_alloc<uint8_t>((size_t)M8 * K), *s1 = dev_alloc<uint8_t>((size_t)M8 * K / 32);
    auto *q2 = dev_alloc<uint8_t>((size_t)M8 * K), *s2 = dev_alloc<uint8_t>((size_t)M8 * K / 32);
    for (int m : {8, 6, 1})
      for (dsk::GemvCfg cfg : {dsk::GemvCfg{1, 1, 8}, dsk::GemvCfg{1, 1, 16}, dsk::GemvCfg{2, 1, 8}}) {
        cfg.tiled = true;
        dsk::rmsnorm(dx, dw, m, K, 1e-20f, y1, q1, s1, 0);
        dsk::gemv_fp8_multi(q1, s1, m, seg, nseg, K, cfg, 0);
        CUDA_CHECK(cudaMemset(y2, 0, (size_t)M8 * K * 2));
        dsk::gemv_fp8_norm(dx, dw, 1e-20f, m, seg2, nseg, K, cfg, 0, y2, q2, s2);
        bool same = host(y1, (size_t)m * K) == host(y2, (size_t)m * K) &&
                    host(q1, (size_t)m * K) == host(q2, (size_t)m * K) &&
                    host(s1, (size_t)m * K / 32) == host(s2, (size_t)m * K / 32);
        size_t diff = 0;
        for (int i = 0; i < nseg; ++i) {
          auto a = host(reinterpret_cast<uint16_t*>(seg[i].y), (size_t)m * seg[i].N);
          auto b = host(reinterpret_cast<uint16_t*>(seg2[i].y), (size_t)m * seg[i].N);
          for (size_t j = 0; j < a.size(); ++j) diff += a[j] != b[j];
        }
        const bool ok = same && diff == 0;
        g_fail += !ok;
        char what[96];
        snprintf(what, sizeof what, "K=%d N=%d M=%d mt=%d wk=%d", K, Ntot, m, cfg.mt, cfg.wk);
        printf("  %-28s %s outputs differing %zu, normed copies %s\n", what, ok ? "ok  " : "FAIL", diff,
               same ? "equal" : "DIFFER");
      }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (void* p : frees) cudaFree(p);
    cudaFree(dx), cudaFree(dw), cudaFree(y1), cudaFree(y2), cudaFree(q1), cudaFree(s1), cudaFree(q2), cudaFree(s2);
  }
  {  // the router gate (bf16 weights, fp32 logits), its CTA 0 also producing the MoE's input
    Npy gw = L("gate_w");
    const int E = gw.shape[0], K = gw.shape[1];
    uint8_t *dgn = dev(gw), *dgt = dev_alloc<uint8_t>(gw.data.size());
    dsk::tile_fp8(dgn, dgt, E, 2 * K, 0);
    __nv_bfloat16 *dx = dev<__nv_bfloat16>(nx), *dw = dev<__nv_bfloat16>(nw);
    auto* y1 = dev_alloc<__nv_bfloat16>((size_t)M8 * K);
    auto *q1 = dev_alloc<uint8_t>((size_t)M8 * K), *s1 = dev_alloc<uint8_t>((size_t)M8 * K / 32);
    auto *q2 = dev_alloc<uint8_t>((size_t)M8 * K), *s2 = dev_alloc<uint8_t>((size_t)M8 * K / 32);
    float *l1 = dev_alloc<float>((size_t)M8 * E), *l2 = dev_alloc<float>((size_t)M8 * E);
    for (int m : {8, 6, 1})
      for (dsk::GemvCfg cfg : {dsk::GemvCfg{1, 1, 8}, dsk::GemvCfg{1, 1, 16}, dsk::GemvCfg{2, 1, 8}}) {
        dsk::rmsnorm(dx, dw, m, K, 1e-20f, y1, q1, s1, 0);
        dsk::gemv_bf16(y1, m, dgt, E, K, l1, nullptr, cfg, false, 0);
        CUDA_CHECK(cudaMemset(q2, 0, (size_t)M8 * K));
        dsk::gemv_bf16_norm(dx, dw, 1e-20f, m, dgt, E, K, l2, nullptr, cfg, 0, q2, s2);
        const bool same = host(q1, (size_t)m * K) == host(q2, (size_t)m * K) &&
                          host(s1, (size_t)m * K / 32) == host(s2, (size_t)m * K / 32);
        auto a = host(l1, (size_t)m * E), b = host(l2, (size_t)m * E);
        size_t diff = 0;
        for (size_t j = 0; j < a.size(); ++j) diff += a[j] != b[j];
        const bool ok = same && diff == 0;
        g_fail += !ok;
        char what[96];
        snprintf(what, sizeof what, "gate K=%d N=%d M=%d mt=%d wk=%d", K, E, m, cfg.mt, cfg.wk);
        printf("  %-28s %s outputs differing %zu, quantised copy %s\n", what, ok ? "ok  " : "FAIL", diff,
               same ? "equal" : "DIFFERS");
      }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dgn), cudaFree(dgt), cudaFree(dx), cudaFree(dw), cudaFree(y1), cudaFree(q1), cudaFree(s1), cudaFree(q2);
    cudaFree(s2), cudaFree(l1), cudaFree(l2);
  }
}

// The persistent FP8 GEMV (GemvCfg::ring >= 0) against the one-tile-per-CTA kernel with the same wk,
// bit for bit, on the decode call sites' shapes: attn_norm + wq_a + wkv (K 5120), q_norm + wq_b +
// the indexer's wq_b (K 1280), wo_b (K 2048, wk 16) and wq_b without the norm; rings of every depth,
// grids from one CTA to more CTAs than SMs (runs of uneven length, segments split across CTAs).
static void test_gemv_fp8p() {
  Npy nw = L("norm_w"), nx = L("norm_x");
  const int M8 = nx.shape[0];
  printf("persistent gemv_fp8 (== gemv_fp8, tiled, same wk)\n");
  struct Site { const char* name; int K, wk; bool norm; std::vector<std::string> mats; };
  for (const Site& sh : {Site{"wq_a+wkv", 5120, 8, true, {"wq_a", "wkv"}}, Site{"wq_b+ix", 1280, 8, true, {"wq_b", "ix_wq_b"}},
                         Site{"wo_b", 2048, 16, false, {"wo_b"}}, Site{"wq_b", 1280, 8, false, {"wq_b"}},
                         Site{"engram wkv", 6144, 8, false, {"rand:6400"}}}) {
    const int K = sh.K, nseg = (int)sh.mats.size();
    std::vector<uint16_t> hx((size_t)M8 * K);  // the norm test rows' first K columns (also the plain input)
    for (int m = 0; m < M8; ++m)
      for (int k = 0; k < K; ++k) hx[(size_t)m * K + k] = nx.as<uint16_t>()[(size_t)m * 5120 + k % 5120];
    __nv_bfloat16 *dx = dev_alloc<__nv_bfloat16>((size_t)M8 * K), *dw = dev<__nv_bfloat16>(nw);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    auto *xq = dev_alloc<uint8_t>((size_t)M8 * K), *xs = dev_alloc<uint8_t>((size_t)M8 * K / 32);
    dsk::act_quant(dx, M8, K, xq, xs, 0);
    dsk::Fp8Seg seg[2], seg2[2];
    std::vector<void*> frees;
    int Ntot = 0;
    for (int i = 0; i < nseg; ++i) {
      int N;
      uint8_t *dwn, *ds;
      if (sh.mats[i].rfind("rand:", 0) == 0) {  // synthetic: e4m3 bytes without NaN, ue8m0 scales 2^-9..2^6
        N = std::atoi(sh.mats[i].c_str() + 5);
        std::vector<uint8_t> hw((size_t)N * K), hs((size_t)(N / 32) * (K / 32));
        uint32_t r = 12345;
        for (auto& b : hw) r = r * 1664525u + 1013904223u, b = (uint8_t)((r >> 24) & 0x77);
        for (auto& b : hs) r = r * 1664525u + 1013904223u, b = (uint8_t)(118 + (r >> 24) % 16);
        dwn = dev_alloc<uint8_t>(hw.size()), ds = dev_alloc<uint8_t>(hs.size());
        CUDA_CHECK(cudaMemcpy(dwn, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice));
      } else {
        Npy w = L(sh.mats[i] + "_w"), sc = L(sh.mats[i] + "_s");
        N = w.shape[0], dwn = dev(w), ds = dev(sc);
      }
      uint8_t* dwt = dev_alloc<uint8_t>((size_t)N * K);
      dsk::tile_fp8(dwn, dwt, N, K, 0);
      seg[i] = {dwt, ds, dev_alloc<__nv_bfloat16>((size_t)M8 * N), N};
      seg2[i] = {dwt, ds, dev_alloc<__nv_bfloat16>((size_t)M8 * N), N};
      frees.insert(frees.end(), {dwn, dwt, ds, seg[i].y, seg2[i].y});
      Ntot += N;
    }
    auto *y1 = dev_alloc<__nv_bfloat16>((size_t)M8 * K), *y2 = dev_alloc<__nv_bfloat16>((size_t)M8 * K);
    auto *q1 = dev_alloc<uint8_t>((size_t)M8 * K), *s1 = dev_alloc<uint8_t>((size_t)M8 * K / 32);
    auto *q2 = dev_alloc<uint8_t>((size_t)M8 * K), *s2 = dev_alloc<uint8_t>((size_t)M8 * K / 32);
    auto run = [&](const dsk::Fp8Seg* sg, int m, const dsk::GemvCfg& cfg, __nv_bfloat16* yo, uint8_t* qo, uint8_t* so) {
      for (int i = 0; i < nseg; ++i) CUDA_CHECK(cudaMemset(sg[i].y, 0xFF, (size_t)M8 * sg[i].N * 2));
      if (sh.norm) dsk::gemv_fp8_norm(dx, dw, 1e-20f, m, sg, nseg, K, cfg, 0, yo, qo, so);
      else dsk::gemv_fp8_multi(xq, xs, m, sg, nseg, K, cfg, 0);
    };
    for (int m : {8, 6, 3, 1}) {
      dsk::GemvCfg ref{1, 1, sh.wk};
      ref.tiled = true;
      run(seg, m, ref, y1, q1, s1);
      size_t diff = 0, runs = 0;
      bool copies = true;
      for (int ring : {0, 4, 8, 12, 16})
        for (int ctas : {0, 1, 5, 48, 96, 333}) {
          if (ring && ctas != 0 && ctas != 5) continue;  // every depth on the default grid, every grid on the auto ring
          dsk::GemvCfg cfg = ref;
          cfg.ring = ring, cfg.ctas = ctas;
          const size_t smem = (size_t)sh.wk * ring * 516 + (size_t)2 * sh.wk * 256 + (sh.norm ? (size_t)m * (K + 32 + K / 32) : 0);
          if (smem > 99 * 1024) continue;  // the kernel's shared memory with this ring
          run(seg2, m, cfg, y2, q2, s2);
          ++runs;
          for (int i = 0; i < nseg; ++i) {
            auto a = host(reinterpret_cast<uint16_t*>(seg[i].y), (size_t)m * seg[i].N);
            auto b = host(reinterpret_cast<uint16_t*>(seg2[i].y), (size_t)m * seg[i].N);
            for (size_t j = 0; j < a.size(); ++j) diff += a[j] != b[j];
          }
          if (sh.norm)
            copies = copies && host(y1, (size_t)m * K) == host(y2, (size_t)m * K) &&
                     host(q1, (size_t)m * K) == host(q2, (size_t)m * K) &&
                     host(s1, (size_t)m * K / 32) == host(s2, (size_t)m * K / 32);
        }
      const bool ok = diff == 0 && copies;
      g_fail += !ok;
      char what[96];
      snprintf(what, sizeof what, "%s N=%d K=%d M=%d", sh.name, Ntot, K, m);
      printf("  %-28s %s %zu runs, outputs differing %zu%s\n", what, ok ? "ok  " : "FAIL", runs, diff,
             sh.norm ? (copies ? ", normed copies equal" : ", normed copies DIFFER") : "");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (void* f : frees) cudaFree(f);
    cudaFree(dx), cudaFree(dw), cudaFree(xq), cudaFree(xs), cudaFree(y1), cudaFree(y2), cudaFree(q1), cudaFree(s1),
        cudaFree(q2), cudaFree(s2);
  }
}

// The persistent bf16 GEMV against k_gemv16 with the same wk, bit for bit: the router gate (384 x 5120)
// with its fused norm (fp32 logits and CTA 0's act_quant of the normed input) and without.
static void test_gemv16p() {
  Npy nw = L("norm_w"), nx = L("norm_x"), gw = L("gate_w");
  const int M8 = nx.shape[0], E = gw.shape[0], K = gw.shape[1];
  printf("persistent gemv_bf16 (== k_gemv16, same wk)\n");
  uint8_t *dgn = dev(gw), *dgt = dev_alloc<uint8_t>(gw.data.size());
  dsk::tile_fp8(dgn, dgt, E, 2 * K, 0);
  __nv_bfloat16 *dx = dev<__nv_bfloat16>(nx), *dw = dev<__nv_bfloat16>(nw);
  auto* y = dev_alloc<__nv_bfloat16>((size_t)M8 * K);
  dsk::rmsnorm(dx, dw, M8, K, 1e-20f, y, nullptr, nullptr, 0);
  auto *q1 = dev_alloc<uint8_t>((size_t)M8 * K), *s1 = dev_alloc<uint8_t>((size_t)M8 * K / 32);
  auto *q2 = dev_alloc<uint8_t>((size_t)M8 * K), *s2 = dev_alloc<uint8_t>((size_t)M8 * K / 32);
  float *l1 = dev_alloc<float>((size_t)M8 * E), *l2 = dev_alloc<float>((size_t)M8 * E);
  for (bool norm : {true, false})
    for (int m : {8, 6, 3, 1})
      for (int wk : {8, 16}) {
        const dsk::GemvCfg ref{1, 1, wk};
        if (norm) dsk::gemv_bf16_norm(dx, dw, 1e-20f, m, dgt, E, K, l1, nullptr, ref, 0, q1, s1);
        else dsk::gemv_bf16(y, m, dgt, E, K, l1, nullptr, ref, false, 0);
        const auto a = host(l1, (size_t)m * E);
        size_t diff = 0, runs = 0;
        bool copies = true;
        for (int ring : {0, 4, 8, 12, 16})
          for (int ctas : {0, 1, 5, 48, 96}) {
            if (ring && ctas != 0 && ctas != 5) continue;
            const size_t smem = (size_t)wk * ring * 512 + (size_t)std::max(2 * wk * 64, 264) * 4 + (norm ? (size_t)m * (K + 32) * 2 : 0);
            if (smem > 99 * 1024) continue;
            dsk::GemvCfg cfg = ref;
            cfg.ring = ring, cfg.ctas = ctas;
            CUDA_CHECK(cudaMemset(l2, 0xFF, (size_t)M8 * E * 4));
            if (norm) dsk::gemv_bf16_norm(dx, dw, 1e-20f, m, dgt, E, K, l2, nullptr, cfg, 0, q2, s2);
            else dsk::gemv_bf16(y, m, dgt, E, K, l2, nullptr, cfg, false, 0);
            ++runs;
            const auto b = host(l2, (size_t)m * E);
            for (size_t j = 0; j < a.size(); ++j) diff += memcmp(&a[j], &b[j], 4) != 0;
            if (norm)
              copies = copies && host(q1, (size_t)m * K) == host(q2, (size_t)m * K) &&
                       host(s1, (size_t)m * K / 32) == host(s2, (size_t)m * K / 32);
          }
        const bool ok = diff == 0 && copies;
        g_fail += !ok;
        char what[96];
        snprintf(what, sizeof what, "gate%s M=%d wk=%d", norm ? " (norm)" : "", m, wk);
        printf("  %-28s %s %zu runs, logits differing %zu%s\n", what, ok ? "ok  " : "FAIL", runs, diff,
               norm ? (copies ? ", act_quant copies equal" : ", act_quant copies DIFFER") : "");
      }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(dgn), cudaFree(dgt), cudaFree(dx), cudaFree(dw), cudaFree(y), cudaFree(q1), cudaFree(s1), cudaFree(q2),
      cudaFree(s2), cudaFree(l1), cudaFree(l2);
}

static void test_router() {
  Npy w = L("gate_w"), b = L("gate_b"), x = L("gate_x"), lg = L("gate_logits"), ix = L("gate_idx"), wt = L("gate_wts");
  const int E = w.shape[0], K = w.shape[1], M8 = x.shape[0];
  printf("router: gate [%d x %d] fp32 logits, sqrt(softplus), top-6\n", E, K);
  uint8_t *dw = dev(w), *dwt = dev_alloc<uint8_t>(w.data.size());
  dsk::tile_fp8(dw, dwt, E, 2 * K, 0);
  auto* dx = dev<__nv_bfloat16>(x);
  float* db = dev<float>(b);
  float* dl = dev_alloc<float>((size_t)M8 * E);
  float* dlr = dev<float>(lg);
  int* di = dev_alloc<int>(M8 * 6);
  float* dwts = dev_alloc<float>(M8 * 6);
  for (int cc = 0; cc < 2; ++cc) {
    dsk::GemvCfg cfg{1, 1, 16};
    dsk::gemv_bf16(dx, M8, dwt, E, K, dl, nullptr, cfg, cc, 0);
    auto hl = host(dl, (size_t)M8 * E);
    check_f32(cc ? "logits cuda-core" : "logits mma", hl.data(), lg.as<float>(), M8, E, 2e-6);
  }
  for (int src = 0; src < 2; ++src) {  // selection from our logits, then from the reference's
    dsk::router_topk(src ? dlr : dl, db, M8, E, di, dwts, 1.5f, 0);
    auto hi = host(di, M8 * 6);
    auto hw = host(dwts, M8 * 6);
    size_t bad_i = 0, exact_w = 0;
    double max_w = 0;
    for (int i = 0; i < M8 * 6; ++i) {
      bad_i += hi[i] != ix.as<int>()[i];
      exact_w += hw[i] == wt.as<float>()[i];
      max_w = fmax(max_w, fabs((double)hw[i] - wt.as<float>()[i]));
    }
    bool ok = bad_i == 0 && max_w < 1e-6;
    g_fail += !ok;
    printf("  %-28s %s ids off %zu / %d, weights exact %zu / %d, max|d| %.3g\n",
           src ? "top-6 from reference logits" : "top-6 from our logits", ok ? "ok  " : "FAIL", bad_i, M8 * 6, exact_w,
           M8 * 6, max_w);
  }
  cudaFree(dw), cudaFree(dwt), cudaFree(dx), cudaFree(db), cudaFree(dl), cudaFree(dlr), cudaFree(di), cudaFree(dwts);
}

// ---- prefill GEMMs (prefill.cu) against gen_ref_prefill.py's 256-row references; row prefixes
// of 200, 129 and 1 exercise partial tiles.
static const uint8_t* tiled_fp8(const Npy& w, int N, int K) {
  uint8_t *dw = dev(w), *t = dev_alloc<uint8_t>(w.data.size());
  dsk::tile_fp8(dw, t, N, K, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(dw);
  return t;
}

struct QAct {
  uint8_t *q, *s;
};
static QAct quantised(const Npy& x) {
  const int M = x.shape[0], K = x.shape[1];
  auto* dx = dev<__nv_bfloat16>(x);
  QAct a{dev_alloc<uint8_t>((size_t)M * K), dev_alloc<uint8_t>((size_t)M * K / 32)};
  dsk::act_quant(dx, M, K, a.q, a.s, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(dx);
  return a;
}

static void test_prefill_fp8() {
  printf("prefill gemm_fp8_p (128x128 / 128x64 tiles, mxf8f6f4 MMA)\n");
  const int ms[] = {256, 200, 129, 1};
  for (const char* name : {"wq_b", "wo_b", "sh_w2", "ix_wq_b"}) {
    Npy w = L(std::string(name) + "_w"), s = L(std::string(name) + "_s");
    Npy x = L(std::string("p_") + name + "_x"), y = L(std::string("p_") + name + "_y");
    const int N = w.shape[0], K = w.shape[1];
    const uint8_t *dw = tiled_fp8(w, N, K), *ds = dev(s);
    QAct a = quantised(x);
    auto* dy = dev_alloc<__nv_bfloat16>((size_t)x.shape[0] * N);
    for (int m : ms)
      for (int v = 0; v < (std::string(name) == "wo_b" ? 6 : 1); ++v) {  // wo_b: every tile variant
        dsk::PrefillTuning tn;
        tn.fp8 = v;
        dsk::prefill_tuning(tn);
        dsk::Fp8Seg seg{dw, ds, dy, N};
        dsk::gemm_fp8_p(a.q, a.s, m, &seg, 1, K, 0);
        dsk::prefill_tuning(dsk::PrefillTuning{});
        auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)m * N);
        char what[64];
        snprintf(what, sizeof what, "%s [%dx%d] M=%d v%d", name, N, K, m, v);
        check_bf16(what, hy.data(), y.as<uint16_t>(), m, N, N);
      }
    cudaFree((void*)dw), cudaFree((void*)ds), cudaFree(a.q), cudaFree(a.s), cudaFree(dy);
  }
  {  // wq_a + wkv: one launch, two segments sharing the input (N = 320 + 128: 64-wide tiles)
    Npy wa = L("wq_a_w"), sa = L("wq_a_s"), wk = L("wkv_w"), sk = L("wkv_s");
    Npy x = L("p_wq_a_x"), ya = L("p_wq_a_y"), yk = L("p_wkv_y");
    const int Na = wa.shape[0], Nk = wk.shape[0], K = wa.shape[1];
    const uint8_t *dwa = tiled_fp8(wa, Na, K), *dsa = dev(sa), *dwk = tiled_fp8(wk, Nk, K), *dsk_ = dev(sk);
    QAct a = quantised(x);
    auto *dya = dev_alloc<__nv_bfloat16>((size_t)x.shape[0] * Na), *dyk = dev_alloc<__nv_bfloat16>((size_t)x.shape[0] * Nk);
    for (int m : ms) {
      dsk::Fp8Seg segs[2] = {{dwa, dsa, dya, Na}, {dwk, dsk_, dyk, Nk}};
      dsk::gemm_fp8_p(a.q, a.s, m, segs, 2, K, 0);
      auto ha = host(reinterpret_cast<uint16_t*>(dya), (size_t)m * Na);
      auto hk = host(reinterpret_cast<uint16_t*>(dyk), (size_t)m * Nk);
      char what[64];
      snprintf(what, sizeof what, "wq_a (+wkv) M=%d", m);
      check_bf16(what, ha.data(), ya.as<uint16_t>(), m, Na, Na);
      snprintf(what, sizeof what, "wkv (+wq_a) M=%d", m);
      check_bf16(what, hk.data(), yk.as<uint16_t>(), m, Nk, Nk);
    }
    cudaFree((void*)dwa), cudaFree((void*)dsa), cudaFree((void*)dwk), cudaFree((void*)dsk_), cudaFree(a.q);
    cudaFree(a.s), cudaFree(dya), cudaFree(dyk);
  }
  {  // shared expert: w1/w3 + SwiGLU + act_quant, then w2 on the reference's quantised h
    Npy w1 = L("moe_shw1"), s1 = L("moe_shs1"), w3 = L("moe_shw3"), s3 = L("moe_shs3"), w2 = L("moe_shw2"), s2 = L("moe_shs2");
    Npy x = L("p_sh_x"), hq = L("p_sh_hq"), hs = L("p_sh_hs"), y = L("p_sh_y");
    const int I = w1.shape[0], D = w1.shape[1], M = x.shape[0];
    const uint8_t *dw1 = tiled_fp8(w1, I, D), *dw3 = tiled_fp8(w3, I, D), *dw2 = tiled_fp8(w2, D, I);
    const uint8_t *ds1 = dev(s1), *ds3 = dev(s3), *ds2 = dev(s2);
    QAct a = quantised(x);
    auto *dq = dev_alloc<uint8_t>((size_t)M * I), *dqs = dev_alloc<uint8_t>((size_t)M * I / 32);
    for (int m : ms) {
      dsk::gemm_fp8_swiglu_p(a.q, a.s, m, dw1, ds1, dw3, ds3, I, D, dq, dqs, 0);
      auto gq = host(dq, (size_t)m * I), gs = host(dqs, (size_t)m * I / 32);
      size_t exact_q = 0, exact_s = 0, far = 0;
      for (int r = 0; r < m; ++r) {
        double rowmax = 0;
        std::vector<double> ge(I), re(I);
        for (int c = 0; c < I; ++c) {
          const size_t i = (size_t)r * I + c, j = (size_t)r * (I / 32) + c / 32;
          ge[c] = e4m3_val(gq[i]) * ldexp(1, gs[j] - 127);
          re[c] = e4m3_val(hq.as<uint8_t>()[i]) * ldexp(1, hs.as<uint8_t>()[j] - 127);
          rowmax = fmax(rowmax, fabs(re[c]));
          exact_q += gq[i] == hq.as<uint8_t>()[i];
        }
        for (int c = 0; c < I; ++c) far += fabs(ge[c] - re[c]) > rowmax / 64;
        for (int c = 0; c < I / 32; ++c) exact_s += gs[(size_t)r * (I / 32) + c] == hs.as<uint8_t>()[(size_t)r * (I / 32) + c];
      }
      const bool ok = far == 0 && exact_q >= 0.99 * m * I;
      g_fail += !ok;
      char what[64];
      snprintf(what, sizeof what, "shared w1/w3 SwiGLU M=%d", m);
      printf("  %-28s %s h exact %6.2f%%, scales exact %6.2f%%, far %zu\n", what, ok ? "ok  " : "FAIL",
             100.0 * exact_q / ((size_t)m * I), 100.0 * exact_s / ((size_t)m * I / 32), far);
    }
    uint8_t *rq = dev(hq), *rs = dev(hs);
    auto* dy = dev_alloc<__nv_bfloat16>((size_t)M * D);
    for (int m : {256, 1}) {
      dsk::Fp8Seg seg{dw2, ds2, dy, D};
      dsk::gemm_fp8_p(rq, rs, m, &seg, 1, I, 0);
      auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)m * D);
      char what[64];
      snprintf(what, sizeof what, "shared w2 (K=%d) M=%d", I, m);
      check_bf16(what, hy.data(), y.as<uint16_t>(), m, D, D);
    }
    for (auto p : {dw1, dw3, dw2, ds1, ds3, ds2}) cudaFree((void*)p);
    cudaFree(a.q), cudaFree(a.s), cudaFree(dq), cudaFree(dqs), cudaFree(rq), cudaFree(rs), cudaFree(dy);
  }
}

static void test_prefill_16() {
  printf("prefill gemm_woa_p / gemm_bf16_p (bf16 MMA)\n");
  {
    Npy w = L("woa_w"), sc = L("woa_s"), x = L("p_woa_x"), y = L("p_woa_y"), q = L("p_woa_q"), qs = L("p_woa_qs");
    const int N = w.shape[0], K = w.shape[1], M = x.shape[0], G = x.shape[1] / K;
    const uint8_t *dw = tiled_fp8(w, N, K), *ds = dev(sc);
    auto* dx = dev<__nv_bfloat16>(x);
    auto* dy = dev_alloc<__nv_bfloat16>((size_t)M * N);
    auto *dq = dev_alloc<uint8_t>((size_t)M * N), *dqs = dev_alloc<uint8_t>((size_t)M * N / 32);
    auto *dq2 = dev_alloc<uint8_t>((size_t)M * N), *dqs2 = dev_alloc<uint8_t>((size_t)M * N / 32);
    for (int m : {256, 200, 129, 1, -256, -200, -1})
     for (int v = m < 0 ? 1 : 0; v < (m < 0 ? 3 : 1); ++v) {  // M < 0: the other tile shapes at |M|
      dsk::PrefillTuning tn;
      tn.w16 = m < 0 ? v : 0;
      dsk::prefill_tuning(tn);
      if (m < 0) m = -m;
      dsk::gemm_woa_p(dx, m, dw, ds, N, K, N / G, dy, dq, dqs, 0);
      dsk::prefill_tuning(dsk::PrefillTuning{});
      auto hy = host(reinterpret_cast<uint16_t*>(dy), (size_t)m * N);
      char what[64];
      snprintf(what, sizeof what, "wo_a [%dx%d]x%d M=%d tiles %d", N / G, K, G, m, tn.w16);
      check_bf16(what, hy.data(), y.as<uint16_t>(), m, N, N);
      dsk::act_quant(dy, m, N, dq2, dqs2, 0);
      const bool ok = host(dq, (size_t)m * N) == host(dq2, (size_t)m * N) &&
                      host(dqs, (size_t)m * N / 32) == host(dqs2, (size_t)m * N / 32);
      g_fail += !ok;
      auto gq = host(dq, (size_t)m * N);
      size_t exact = 0;
      for (size_t i = 0; i < gq.size(); ++i) exact += gq[i] == q.as<uint8_t>()[i];
      printf("  %-28s %s (vs reference's quantised: %.2f%% exact)\n", "  fused quant == act_quant(y)", ok ? "ok  " : "FAIL",
             100.0 * exact / gq.size());
      if (tn.w16) m = -m;
    }
    cudaFree((void*)dw), cudaFree((void*)ds), cudaFree(dx), cudaFree(dy), cudaFree(dq), cudaFree(dqs), cudaFree(dq2);
    cudaFree(dqs2);
    (void)qs;
  }
  {
    Npy w = L("gate_w"), x = L("p_gate_x"), lg = L("p_gate_logits");
    const int E = w.shape[0], K = w.shape[1], M = x.shape[0];
    const uint8_t* dw = tiled_fp8(w, E, 2 * K);
    auto* dx = dev<__nv_bfloat16>(x);
    float* dl = dev_alloc<float>((size_t)M * E);
    for (int N : {384, 96, 64})
      for (int m : {256, 200, 1}) {
        dsk::gemm_bf16_p(dx, m, dw, N, K, dl, 0);
        auto hl = host(dl, (size_t)m * N);
        std::vector<float> ex((size_t)m * N);
        for (int r = 0; r < m; ++r)
          for (int c = 0; c < N; ++c) ex[(size_t)r * N + c] = lg.as<float>()[(size_t)r * E + c];
        char what[64];
        snprintf(what, sizeof what, "gate N=%d M=%d", N, m);
        check_f32(what, hl.data(), ex.data(), m, N, 2e-6);
        if (N == 384 && m == 256) {  // both against fp64 dot products, rows 0..15
          double e_ours = 0, e_ref = 0;
          for (int r = 0; r < 16; ++r) {
            std::vector<double> exd(N);
            double scale = 0;
            for (int n = 0; n < N; ++n) {
              double acc = 0;
              for (int k = 0; k < K; ++k)
                acc += (double)bf2f(x.as<uint16_t>()[(size_t)r * K + k]) * bf2f(w.as<uint16_t>()[(size_t)n * K + k]);
              exd[n] = acc, scale = fmax(scale, fabs(acc));
            }
            for (int n = 0; n < N; ++n) {
              e_ours = fmax(e_ours, fabs(hl[(size_t)r * N + n] - exd[n]) / scale);
              e_ref = fmax(e_ref, fabs(lg.as<float>()[(size_t)r * E + n] - exd[n]) / scale);
            }
          }
          printf("    vs fp64: max err/rowmax ours %.3g, torch's %.3g\n", e_ours, e_ref);
        }
      }
    cudaFree((void*)dw), cudaFree(dx), cudaFree(dl);
  }
}

static void test_prefill_moe() {
  Npy x = L("p_moe_x"), sel = L("p_moe_sel"), rw = L("p_moe_rw"), yr = L("p_moe_y_routed"), ysh = L("p_moe_y_shared");
  Npy w1 = L("moe_w1"), s1 = L("moe_s1"), w3 = L("moe_w3"), s3 = L("moe_s3"), w2 = L("moe_w2"), s2 = L("moe_s2");
  Npy shw1 = L("moe_shw1"), shs1 = L("moe_shs1"), shw3 = L("moe_shw3"), shs3 = L("moe_shs3");
  Npy shw2 = L("moe_shw2"), shs2 = L("moe_shs2");
  const int E = w1.shape[0], I = w1.shape[1], D = x.shape[1], M = x.shape[0];
  std::vector<int> cnt(E);
  for (size_t i = 0; i < sel.count(); ++i) ++cnt[sel.as<int>()[i]];
  printf("prefill moe_prefill: %d routed experts, %d rows, picks per expert:", E, M);
  for (int c : cnt) printf(" %d", c);
  printf("\n");
  auto tile4 = [&](const Npy& w, const Npy& s, int N, int K, const uint8_t*& wt, const uint8_t*& st) {
    uint8_t *dw = dev(w), *ds = dev(s), *t = dev_alloc<uint8_t>(w.data.size()), *ts = dev_alloc<uint8_t>(s.data.size());
    dsk::tile_fp4(dw, ds, t, ts, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw), cudaFree(ds);
    wt = t, st = ts;
  };
  dsk::MoeWeights mw{};
  tile4(w1, s1, E * I, D, mw.w1, mw.s1);
  tile4(w3, s3, E * I, D, mw.w3, mw.s3);
  tile4(w2, s2, E * D, I, mw.w2, mw.s2);
  mw.sh_w1 = tiled_fp8(shw1, I, D), mw.sh_s1 = dev(shs1);
  mw.sh_w3 = tiled_fp8(shw3, I, D), mw.sh_s3 = dev(shs3);
  mw.sh_w2 = tiled_fp8(shw2, D, I), mw.sh_s2 = dev(shs2);
  QAct a = quantised(x);
  int* didx = dev<int>(sel);
  float* dwts = dev<float>(rw);
  uint8_t* ws = dev_alloc<uint8_t>(dsk::moe_prefill_workspace_bytes(M, E));
  float *dy = dev_alloc<float>((size_t)M * D), *dy2 = dev_alloc<float>((size_t)M * D);
  std::vector<float> yexp((size_t)M * D);
  for (size_t i = 0; i < yexp.size(); ++i) yexp[i] = yr.as<float>()[i] + ysh.as<float>()[i];
  for (int m : {256, 200, 129, 1}) {
    CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)M * D * 4));
    dsk::moe_prefill(a.q, a.s, m, didx, dwts, E, mw, ws, dy, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto hy = host(dy, (size_t)m * D);
    char what[64];
    snprintf(what, sizeof what, "moe_prefill M=%d", m);
    check_f32_rows(what, hy.data(), yexp.data(), m, D, 1.0 / 64);
  }
  {  // run-to-run: the bucket order varies (atomics), the result must not
    dsk::moe_prefill(a.q, a.s, M, didx, dwts, E, mw, ws, dy, 0);
    dsk::moe_prefill(a.q, a.s, M, didx, dwts, E, mw, ws, dy2, 0);
    const bool ok = host(dy, (size_t)M * D) == host(dy2, (size_t)M * D);
    g_fail += !ok;
    printf("  %-28s %s\n", "deterministic", ok ? "ok  " : "FAIL");
    dsk::PrefillTuning t;  // the combine fused into the down kernel or separate, L2 hints: the same bits
    t.down = 1, t.l2 = 0, t.dgrp = 0;
    dsk::prefill_tuning(t);
    dsk::moe_prefill(a.q, a.s, M, didx, dwts, E, mw, ws, dy, 0);
    const auto ref = host(dy, (size_t)M * D);
    for (int v = 0; v < 3; ++v) {
      t.down = v == 1 ? 1 : 2, t.l2 = v == 0 ? 0 : 7;
      dsk::prefill_tuning(t);
      dsk::moe_prefill(a.q, a.s, M, didx, dwts, E, mw, ws, dy2, 0);
      const bool ok2 = ref == host(dy2, (size_t)M * D);
      g_fail += !ok2;
      const char* nm[] = {"fused combine == separate", "L2 hints, separate: same", "L2 hints, fused: same"};
      printf("  %-28s %s\n", nm[v], ok2 ? "ok  " : "FAIL");
    }
    // variants that only change the schedule, each from a workspace and output of 0xFF bytes (so that
    // no tile or row group can be missed unnoticed), the combine separate (1) or fused (2)
    auto variant = [&](const char* what, dsk::PrefillTuning v) {
      dsk::prefill_tuning(v);
      CUDA_CHECK(cudaMemset(ws, 0xFF, dsk::moe_prefill_workspace_bytes(M, E)));
      CUDA_CHECK(cudaMemset(dy2, 0xFF, (size_t)M * D * 4));
      dsk::moe_prefill(a.q, a.s, M, didx, dwts, E, mw, ws, dy2, 0);
      const bool ok2 = ref == host(dy2, (size_t)M * D);
      g_fail += !ok2;
      printf("  %-28s %s\n", what, ok2 ? "ok  " : "FAIL");
    };
    char what[64];
    for (int fz : {1, 2}) {
      dsk::PrefillTuning v;
      v.down = fz;
      const char* comb = fz == 2 ? ", fused" : "";
      for (int G : {1, 3, 10, 40})  // the down kernel a CTA per (tile, G column blocks)
        for (int ring : {3, 9}) {
          v.l2 = G == 3 ? 0 : 7, v.dgrp = G, v.dring = ring;
          snprintf(what, sizeof what, "down by %d blocks, ring %d%s: same", G, ring, comb);
          variant(what, v);
        }
      v.l2 = 7, v.dgrp = 0, v.dring = 3;
      for (int u : {1, 3}) {  // the up kernel's L2 prefetch
        v.upf = u;
        snprintf(what, sizeof what, "up prefetch %d ahead%s: same", u, comb);
        variant(what, v);
      }
      v.upf = 0;
      for (int k : {1, 2, 5}) {  // the tile order, also with the down kernel by 4 blocks
        v.tord = k, v.dgrp = k == 2 ? 4 : 0;
        snprintf(what, sizeof what, "tiles 1:%d%s%s: same", k, k == 2 ? ", down by 4" : "", comb);
        variant(what, v);
      }
    }
    dsk::prefill_tuning(dsk::PrefillTuning{});
  }
  {  // the decode kernel on the same picks, 8 rows at a time: same numerics, so (nearly) equal
    uint8_t* wsd = dev_alloc<uint8_t>(dsk::moe_workspace_bytes());
    for (int r = 0; r < M; r += 8)
      dsk::moe(a.q + (size_t)r * D, a.s + (size_t)r * D / 32, 8, didx + r * 6, dwts + r * 6, mw, wsd,
               dy2 + (size_t)r * D, dsk::MoeCfg{}, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto hy = host(dy, (size_t)M * D), hd = host(dy2, (size_t)M * D);
    check_f32_rows("vs decode moe(), 8-row steps", hy.data(), hd.data(), M, D, 1.0 / 64);
    cudaFree(wsd);
  }
  for (auto p : {mw.w1, mw.s1, mw.w3, mw.s3, mw.w2, mw.s2, mw.sh_w1, mw.sh_s1, mw.sh_w3, mw.sh_s3, mw.sh_w2, mw.sh_s2})
    cudaFree((void*)p);
  cudaFree(a.q), cudaFree(a.s), cudaFree(didx), cudaFree(dwts), cudaFree(ws), cudaFree(dy), cudaFree(dy2);
}

// ---- 9..16 rows (docs/batching.md, B4: the _r16 kernels, rows 8..15 in the MMAs' upper half). The
// 16-row inputs are the 8 reference rows and then the same rows reversed (row r >= 8 is row 15 - r). At
// 9, 12 and 16 rows every output row must equal, bit for bit, that input row's output from an 8-row call
// (which the tests above hold to the reference), and the fused paths their unfused sequences.
static int src8(int r) { return r < 8 ? r : 15 - r; }
template <class T>
static T* dev_rows16(const T* h8, size_t cols) {  // [16][cols] on the device from 8 host rows
  std::vector<T> v(16 * cols);
  for (int r = 0; r < 16; ++r) memcpy(&v[r * cols], h8 + src8(r) * cols, cols * sizeof(T));
  T* d = dev_alloc<T>(v.size());
  CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
// elements of rows [0, m) of d [m][cols] whose bits differ from row src8(r) of h8 [8][cols]
template <class T>
static size_t diff8(const T* d, const std::vector<T>& h8, int m, size_t cols) {
  auto h = host(d, (size_t)m * cols);
  size_t n = 0;
  for (int r = 0; r < m; ++r)
    for (size_t c = 0; c < cols; ++c) n += memcmp(&h[r * cols + c], &h8[src8(r) * cols + c], sizeof(T)) != 0;
  return n;
}
template <class T>
static size_t diffd(const T* a, const T* b, size_t n) {  // device arrays, bitwise
  auto ha = host(a, n), hb = host(b, n);
  size_t d = 0;
  for (size_t i = 0; i < n; ++i) d += memcmp(&ha[i], &hb[i], sizeof(T)) != 0;
  return d;
}
static void verdict(const char* what, size_t d8, long dref = -1) {  // dref < 0: no unfused counterpart
  const bool ok = d8 == 0 && dref <= 0;
  g_fail += !ok;
  if (dref < 0) printf("  %-40s %s %zu differ from the 8-row calls\n", what, ok ? "ok  " : "FAIL", d8);
  else printf("  %-40s %s %zu differ from the 8-row calls, %ld from the unfused path\n", what, ok ? "ok  " : "FAIL", d8, dref);
}

static void test_rows16_gemv() {
  printf("9..16 rows: GEMVs (rows 8..15 = rows 7..0) against 8-row calls and unfused paths\n");
  for (const std::string name : {"wq_a", "wo_b"}) {  // FP8, plain and tiled
    Npy w = L(name + "_w"), s = L(name + "_s"), x = L(name + "_x");
    const int N = w.shape[0], K = w.shape[1];
    if (x.shape[0] != 8) {
      printf("  %s: %d reference rows, expected 8\n", name.c_str(), (int)x.shape[0]);
      ++g_fail;
      continue;
    }
    uint8_t *dw = dev(w), *ds = dev(s), *dwt = dev_alloc<uint8_t>((size_t)N * K);
    dsk::tile_fp8(dw, dwt, N, K, 0);
    auto* dx = reinterpret_cast<__nv_bfloat16*>(dev_rows16(x.as<uint16_t>(), K));
    uint8_t *dq = dev_alloc<uint8_t>((size_t)16 * K), *dqs = dev_alloc<uint8_t>((size_t)16 * K / 32);
    dsk::act_quant(dx, 16, K, dq, dqs, 0);
    auto* dy = dev_alloc<uint16_t>((size_t)16 * N);
    for (int tiled = 0; tiled < 2; ++tiled)
      for (dsk::GemvCfg cfg : {dsk::GemvCfg{1, 1, 16}, dsk::GemvCfg{2, 1, 8}, dsk::GemvCfg{1, 2, 8}}) {
        cfg.tiled = tiled;
        const uint8_t* W = tiled ? dwt : dw;
        dsk::gemv_fp8(dq, dqs, 8, W, ds, N, K, reinterpret_cast<__nv_bfloat16*>(dy), cfg, 0);
        const auto y8 = host(dy, (size_t)8 * N);
        for (int m : {9, 12, 16}) {
          CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)16 * N * 2));
          dsk::gemv_fp8(dq, dqs, m, W, ds, N, K, reinterpret_cast<__nv_bfloat16*>(dy), cfg, 0);
          char what[96];
          snprintf(what, sizeof what, "fp8 %s%s M=%d mt=%d wn=%d wk=%d", name.c_str(), tiled ? " tiled" : "", m, cfg.mt,
                   cfg.wn, cfg.wk);
          verdict(what, diff8(dy, y8, m, N));
        }
      }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw), cudaFree(ds), cudaFree(dwt), cudaFree(dx), cudaFree(dq), cudaFree(dqs), cudaFree(dy);
  }

  Npy nw = L("norm_w"), nx = L("norm_x");
  struct Shape { int K; std::vector<std::string> mats; };
  for (const Shape& sh : {Shape{5120, {"wq_a", "wkv"}}, Shape{1280, {"wq_b"}}}) {  // gemv_fp8_norm
    const int K = sh.K, nseg = (int)sh.mats.size();
    std::vector<uint16_t> hx((size_t)8 * K);
    for (int m = 0; m < 8; ++m) memcpy(&hx[(size_t)m * K], nx.as<uint16_t>() + (size_t)m * 5120, K * 2);
    auto* dx = reinterpret_cast<__nv_bfloat16*>(dev_rows16(hx.data(), K));
    auto* dnw = dev<__nv_bfloat16>(nw);
    dsk::Fp8Seg seg[2], seg2[2];
    std::vector<void*> frees;
    for (int i = 0; i < nseg; ++i) {
      Npy w = L(sh.mats[i] + "_w"), sc = L(sh.mats[i] + "_s");
      const int N = w.shape[0];
      uint8_t *dwn = dev(w), *dwt = dev_alloc<uint8_t>((size_t)N * K), *ds = dev(sc);
      dsk::tile_fp8(dwn, dwt, N, K, 0);
      seg[i] = {dwt, ds, dev_alloc<__nv_bfloat16>((size_t)16 * N), N};
      seg2[i] = {dwt, ds, dev_alloc<__nv_bfloat16>((size_t)16 * N), N};
      frees.insert(frees.end(), {dwn, dwt, ds, seg[i].y, seg2[i].y});
    }
    auto *y1 = dev_alloc<__nv_bfloat16>((size_t)16 * K), *y2 = dev_alloc<__nv_bfloat16>((size_t)16 * K);
    auto *q1 = dev_alloc<uint8_t>((size_t)16 * K), *s1 = dev_alloc<uint8_t>((size_t)16 * K / 32);
    auto *q2 = dev_alloc<uint8_t>((size_t)16 * K), *s2 = dev_alloc<uint8_t>((size_t)16 * K / 32);
    for (dsk::GemvCfg cfg : {dsk::GemvCfg{1, 1, 8}, dsk::GemvCfg{1, 1, 16}, dsk::GemvCfg{2, 1, 8}}) {
      cfg.tiled = true;
      dsk::gemv_fp8_norm(dx, dnw, 1e-20f, 8, seg2, nseg, K, cfg, 0, y2, q2, s2);
      std::vector<std::vector<__nv_bfloat16>> y8(nseg);
      for (int i = 0; i < nseg; ++i) y8[i] = host(seg2[i].y, (size_t)8 * seg2[i].N);
      const auto n8 = host(y2, (size_t)8 * K);
      const auto q8 = host(q2, (size_t)8 * K), s8 = host(s2, (size_t)8 * K / 32);
      for (int m : {9, 12, 16}) {
        dsk::rmsnorm(dx, dnw, m, K, 1e-20f, y1, q1, s1, 0);
        dsk::gemv_fp8_multi(q1, s1, m, seg, nseg, K, cfg, 0);
        CUDA_CHECK(cudaMemset(y2, 0xFF, (size_t)16 * K * 2));
        CUDA_CHECK(cudaMemset(q2, 0xFF, (size_t)16 * K));
        CUDA_CHECK(cudaMemset(s2, 0xFF, (size_t)16 * K / 32));
        for (int i = 0; i < nseg; ++i) CUDA_CHECK(cudaMemset(seg2[i].y, 0xFF, (size_t)16 * seg2[i].N * 2));
        dsk::gemv_fp8_norm(dx, dnw, 1e-20f, m, seg2, nseg, K, cfg, 0, y2, q2, s2);
        size_t d8 = diff8(y2, n8, m, K) + diff8(q2, q8, m, K) + diff8(s2, s8, m, K / 32), dref = 0;
        dref += diffd(y1, y2, (size_t)m * K) + diffd(q1, q2, (size_t)m * K) + diffd(s1, s2, (size_t)m * K / 32);
        for (int i = 0; i < nseg; ++i) {
          d8 += diff8(seg2[i].y, y8[i], m, seg2[i].N);
          dref += diffd(seg[i].y, seg2[i].y, (size_t)m * seg[i].N);
        }
        char what[96];
        snprintf(what, sizeof what, "fp8 norm K=%d M=%d mt=%d wk=%d", K, m, cfg.mt, cfg.wk);
        verdict(what, d8, (long)dref);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (void* p : frees) cudaFree(p);
    cudaFree(dx), cudaFree(dnw), cudaFree(y1), cudaFree(y2), cudaFree(q1), cudaFree(s1), cudaFree(q2), cudaFree(s2);
  }

  {  // the head (bf16 weights, fp32 logits)
    Npy w = L("head_w"), x = L("head_x");
    const int N = w.shape[0], K = w.shape[1];
    uint8_t *dw = dev(w), *dwt = dev_alloc<uint8_t>(w.data.size());
    dsk::tile_fp8(dw, dwt, N, 2 * K, 0);
    cudaFree(dw);
    auto* dx = reinterpret_cast<__nv_bfloat16*>(dev_rows16(x.as<uint16_t>(), K));
    float* dy = dev_alloc<float>((size_t)16 * N);
    for (int mt : {1, 2}) {
      dsk::GemvCfg cfg{mt, 1, 16};
      dsk::gemv_bf16(dx, 8, dwt, N, K, dy, nullptr, cfg, false, 0);
      const auto y8 = host(dy, (size_t)8 * N);
      for (int m : {9, 12, 16}) {
        CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)16 * N * 4));
        dsk::gemv_bf16(dx, m, dwt, N, K, dy, nullptr, cfg, false, 0);
        char what[96];
        snprintf(what, sizeof what, "head M=%d mt=%d", m, mt);
        verdict(what, diff8(dy, y8, m, N));
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dwt), cudaFree(dx), cudaFree(dy);
  }

  {  // the router gate with the norm fused (and CTA 0's act_quant for the MoE) == rmsnorm + gemv_bf16
    Npy gw = L("gate_w");
    const int E = gw.shape[0], K = gw.shape[1];
    uint8_t *dgn = dev(gw), *dgt = dev_alloc<uint8_t>(gw.data.size());
    dsk::tile_fp8(dgn, dgt, E, 2 * K, 0);
    auto* dx = reinterpret_cast<__nv_bfloat16*>(dev_rows16(nx.as<uint16_t>(), K));
    auto* dnw = dev<__nv_bfloat16>(nw);
    auto* y1 = dev_alloc<__nv_bfloat16>((size_t)16 * K);
    auto *q1 = dev_alloc<uint8_t>((size_t)16 * K), *s1 = dev_alloc<uint8_t>((size_t)16 * K / 32);
    auto *q2 = dev_alloc<uint8_t>((size_t)16 * K), *s2 = dev_alloc<uint8_t>((size_t)16 * K / 32);
    float *l1 = dev_alloc<float>((size_t)16 * E), *l2 = dev_alloc<float>((size_t)16 * E);
    for (dsk::GemvCfg cfg : {dsk::GemvCfg{1, 1, 8}, dsk::GemvCfg{1, 1, 16}, dsk::GemvCfg{2, 1, 8}, dsk::GemvCfg{2, 1, 16}}) {
      dsk::gemv_bf16_norm(dx, dnw, 1e-20f, 8, dgt, E, K, l2, nullptr, cfg, 0, q2, s2);
      const auto l8 = host(l2, (size_t)8 * E);
      const auto q8 = host(q2, (size_t)8 * K), s8 = host(s2, (size_t)8 * K / 32);
      for (int m : {9, 12, 16}) {
        dsk::rmsnorm(dx, dnw, m, K, 1e-20f, y1, q1, s1, 0);
        dsk::gemv_bf16(y1, m, dgt, E, K, l1, nullptr, cfg, false, 0);
        CUDA_CHECK(cudaMemset(l2, 0xFF, (size_t)16 * E * 4));
        CUDA_CHECK(cudaMemset(q2, 0xFF, (size_t)16 * K));
        CUDA_CHECK(cudaMemset(s2, 0xFF, (size_t)16 * K / 32));
        dsk::gemv_bf16_norm(dx, dnw, 1e-20f, m, dgt, E, K, l2, nullptr, cfg, 0, q2, s2);
        const size_t d8 = diff8(l2, l8, m, E) + diff8(q2, q8, m, K) + diff8(s2, s8, m, K / 32);
        const size_t dref = diffd(l1, l2, (size_t)m * E) + diffd(q1, q2, (size_t)m * K) + diffd(s1, s2, (size_t)m * K / 32);
        char what[96];
        snprintf(what, sizeof what, "gate norm M=%d mt=%d wk=%d", m, cfg.mt, cfg.wk);
        verdict(what, d8, (long)dref);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dgn), cudaFree(dgt), cudaFree(dx), cudaFree(dnw), cudaFree(y1), cudaFree(q1), cudaFree(s1), cudaFree(q2);
    cudaFree(s2), cudaFree(l1), cudaFree(l2);
  }

  {  // wo_a (FP8 widened to bf16, block-diagonal), its fused act_quant, and DSpark's FP8 head (one group)
    Npy w = L("woa_w"), sc = L("woa_s"), x = L("woa_x");
    const int N = w.shape[0], K = w.shape[1], G = x.shape[1] / K;
    uint8_t *dw = dev(w), *ds = dev(sc), *dwt = dev_alloc<uint8_t>(w.data.size());
    dsk::tile_fp8(dw, dwt, N, K, 0);
    auto* dx = reinterpret_cast<__nv_bfloat16*>(dev_rows16(x.as<uint16_t>(), (size_t)G * K));
    auto* dy = dev_alloc<__nv_bfloat16>((size_t)16 * N);
    for (int mt : {1, 2}) {
      dsk::GemvCfg cfg{mt, 1, 16};
      dsk::gemv_woa(dx, 8, dwt, ds, N, K, N / G, dy, cfg, false, 0);
      const auto y8 = host(dy, (size_t)8 * N);
      for (int m : {9, 12, 16}) {
        CUDA_CHECK(cudaMemset(dy, 0xFF, (size_t)16 * N * 2));
        dsk::gemv_woa(dx, m, dwt, ds, N, K, N / G, dy, cfg, false, 0);
        char what[96];
        snprintf(what, sizeof what, "woa M=%d mt=%d", m, mt);
        verdict(what, diff8(dy, y8, m, N));
      }
    }
    auto *dq = dev_alloc<uint8_t>((size_t)16 * N), *dqs = dev_alloc<uint8_t>((size_t)16 * N / 32);
    auto *dq2 = dev_alloc<uint8_t>((size_t)16 * N), *dqs2 = dev_alloc<uint8_t>((size_t)16 * N / 32);
    for (dsk::GemvCfg cfg : {dsk::GemvCfg{1, 4, 4}, dsk::GemvCfg{2, 2, 4}}) {  // mt * wn = 4
      dsk::gemv_woa(dx, 8, dwt, ds, N, K, N / G, dy, cfg, false, 0, dq, dqs);
      const auto y8 = host(dy, (size_t)8 * N);
      const auto q8 = host(dq, (size_t)8 * N), s8 = host(dqs, (size_t)8 * N / 32);
      for (int m : {9, 12, 16}) {
        CUDA_CHECK(cudaMemset(dq, 0xFF, (size_t)16 * N));
        CUDA_CHECK(cudaMemset(dqs, 0xFF, (size_t)16 * N / 32));
        dsk::gemv_woa(dx, m, dwt, ds, N, K, N / G, dy, cfg, false, 0, dq, dqs);
        dsk::act_quant(dy, m, N, dq2, dqs2, 0);
        const size_t d8 = diff8(dy, y8, m, N) + diff8(dq, q8, m, N) + diff8(dqs, s8, m, N / 32);
        const size_t dref = diffd(dq, dq2, (size_t)m * N) + diffd(dqs, dqs2, (size_t)m * N / 32);
        char what[96];
        snprintf(what, sizeof what, "woa + act_quant M=%d mt=%d wn=%d", m, cfg.mt, cfg.wn);
        verdict(what, d8, (long)dref);
      }
    }
    {  // gemv_fp8w_f32: every row reads all of x [M][K] (the first K columns of the wo_a rows)
      std::vector<uint16_t> hx((size_t)8 * K);
      for (int m = 0; m < 8; ++m) memcpy(&hx[(size_t)m * K], x.as<uint16_t>() + (size_t)m * G * K, K * 2);
      auto* dx1 = reinterpret_cast<__nv_bfloat16*>(dev_rows16(hx.data(), K));
      float* dyf = dev_alloc<float>((size_t)16 * N);
      for (int mt : {1, 2}) {
        dsk::GemvCfg cfg{mt, 1, 16};
        dsk::gemv_fp8w_f32(dx1, 8, dwt, ds, N, K, dyf, cfg, 0);
        const auto y8 = host(dyf, (size_t)8 * N);
        for (int m : {9, 12, 16}) {
          CUDA_CHECK(cudaMemset(dyf, 0xFF, (size_t)16 * N * 4));
          dsk::gemv_fp8w_f32(dx1, m, dwt, ds, N, K, dyf, cfg, 0);
          char what[96];
          snprintf(what, sizeof what, "fp8w_f32 M=%d mt=%d", m, mt);
          verdict(what, diff8(dyf, y8, m, N));
        }
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaFree(dx1), cudaFree(dyf);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw), cudaFree(ds), cudaFree(dwt), cudaFree(dx), cudaFree(dy), cudaFree(dq), cudaFree(dqs), cudaFree(dq2);
    cudaFree(dqs2);
  }
}

__global__ void k_fill16(uint8_t* p, size_t n, uint32_t seed, int lo, int span) {  // span 0: any byte
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;
    p[i] = span ? (uint8_t)(lo + h % span) : (uint8_t)h;
  }
}
static uint8_t* dev_fill16(size_t n, uint32_t seed, int lo = 0, int span = 0) {
  uint8_t* p = dev_alloc<uint8_t>(n);
  k_fill16<<<1024, 256>>>(p, n, seed, lo, span);
  CUDA_CHECK(cudaGetLastError());
  return p;
}

// The MoE at 9..16 rows: the reference layer's experts (12) and a synthetic pool of 128 (random MXFP4
// weights, the reference's shared expert) with routings whose unions reach 96 experts. Every 16-row
// variant (up16 x down16) against two 8-row calls (rows 0..7, rows 8..m-1), bit for bit: y, and moe_up's
// intermediate (hq/hs) row by row through the unions' slots. moe_up's bits depend only on its warps
// along K (WK): the 8-row calls use up 0 (WK 4) or 4 (WK 8) to match. Then moe_routed == router_topk +
// moe, and the task queue == the static schedule, at 16 rows.
static void test_rows16_moe() {
  Npy x = L("moe_x"), sel = L("moe_sel"), rw = L("moe_rw");
  Npy w1 = L("moe_w1"), s1 = L("moe_s1"), w3 = L("moe_w3"), s3 = L("moe_s3"), w2 = L("moe_w2"), s2 = L("moe_s2");
  Npy shw1 = L("moe_shw1"), shs1 = L("moe_shs1"), shw3 = L("moe_shw3"), shs3 = L("moe_shs3");
  Npy shw2 = L("moe_shw2"), shs2 = L("moe_shs2");
  const int E0 = w1.shape[0], I = w1.shape[1], D = x.shape[1];
  printf("9..16 rows: moe (%d reference experts; a pool of 128 random ones) against 8-row calls\n", E0);
  if (x.shape[0] != 8 || I != 576 || D != 5120) {
    printf("  moe reference: %d x %d rows, I=%d, expected 8 x 5120, 576\n", (int)x.shape[0], D, I);
    ++g_fail;
    return;
  }
  auto tile4 = [&](const Npy& w, const Npy& s, int N, int K, const uint8_t*& wt, const uint8_t*& st) {
    uint8_t *dw = dev(w), *ds = dev(s), *t = dev_alloc<uint8_t>(w.data.size()), *ts = dev_alloc<uint8_t>(s.data.size());
    dsk::tile_fp4(dw, ds, t, ts, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw), cudaFree(ds);
    wt = t, st = ts;
  };
  auto tile8 = [&](const Npy& w, int N, int K) {
    uint8_t *dw = dev(w), *t = dev_alloc<uint8_t>(w.data.size());
    dsk::tile_fp8(dw, t, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw);
    return (const uint8_t*)t;
  };
  dsk::MoeWeights ref{};
  tile4(w1, s1, E0 * I, D, ref.w1, ref.s1);
  tile4(w3, s3, E0 * I, D, ref.w3, ref.s3);
  tile4(w2, s2, E0 * D, I, ref.w2, ref.s2);
  ref.sh_w1 = tile8(shw1, I, D), ref.sh_s1 = dev(shs1);
  ref.sh_w3 = tile8(shw3, I, D), ref.sh_s3 = dev(shs3);
  ref.sh_w2 = tile8(shw2, D, I), ref.sh_s2 = dev(shs2);
  constexpr int kPoolE = 128;
  const size_t w13 = (size_t)I * D / 2, s13 = (size_t)I * D / 32, w2b = (size_t)D * I / 2, s2b = (size_t)D * I / 32;
  dsk::MoeWeights pool = ref;  // already in the tiled layouts: random bytes in any layout are random
  pool.w1 = dev_fill16(kPoolE * w13, 1), pool.s1 = dev_fill16(kPoolE * s13, 2, 118, 8);
  pool.w3 = dev_fill16(kPoolE * w13, 3), pool.s3 = dev_fill16(kPoolE * s13, 4, 118, 8);
  pool.w2 = dev_fill16(kPoolE * w2b, 5), pool.s2 = dev_fill16(kPoolE * s2b, 6, 118, 8);

  auto* dx = reinterpret_cast<__nv_bfloat16*>(dev_rows16(x.as<uint16_t>(), D));
  uint8_t *dq = dev_alloc<uint8_t>((size_t)16 * D), *dqs = dev_alloc<uint8_t>((size_t)16 * D / 32);
  dsk::act_quant(dx, 16, D, dq, dqs, 0);
  const size_t wsb = dsk::moe_workspace_bytes();
  uint8_t *ws16 = dev_alloc<uint8_t>(wsb), *wsA = dev_alloc<uint8_t>(wsb), *wsB = dev_alloc<uint8_t>(wsb);
  float *y16 = dev_alloc<float>((size_t)16 * D), *yA = dev_alloc<float>((size_t)8 * D), *yB = dev_alloc<float>((size_t)8 * D);
  int* didx = dev_alloc<int>(96);
  float* dwts = dev_alloc<float>(96);

  // routings: 0 the reference picks (rows 8..15 those of rows 7..0), 1 96 distinct experts of the pool,
  // 2 rows sharing experts (ids < 40)
  const char* rname[] = {"ref", "96 experts", "shared"};
  for (int routing = 0; routing < 3; ++routing) {
    const dsk::MoeWeights& mw = routing == 0 ? ref : pool;
    std::vector<int> hi(96);
    std::vector<float> hw(96);
    for (int r = 0; r < 16; ++r)
      for (int k = 0; k < 6; ++k) {
        const int i = r * 6 + k;
        hi[i] = routing == 0 ? sel.as<int>()[src8(r) * 6 + k] : routing == 1 ? (i * 37 + 11) % kPoolE : (r * 3 + k * 5) % 40;
        hw[i] = routing == 0 ? rw.as<float>()[src8(r) * 6 + k] : 0.05f + 0.9f * (float)((i * 7919) % 101) / 101.f;
      }
    CUDA_CHECK(cudaMemcpy(didx, hi.data(), 96 * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dwts, hw.data(), 96 * 4, cudaMemcpyHostToDevice));
    auto uni = [&](int r0, int r1) {  // the sorted union of rows [r0, r1)'s picks
      std::vector<int> u(hi.begin() + r0 * 6, hi.begin() + r1 * 6);
      std::sort(u.begin(), u.end());
      u.erase(std::unique(u.begin(), u.end()), u.end());
      return u;
    };
    for (int m : {9, 12, 16}) {
      const auto u16 = uni(0, m), uA = uni(0, 8), uB = uni(8, m);
      for (int wk8 = 0; wk8 < 2; ++wk8) {
        dsk::MoeCfg c8{};
        c8.up = wk8 ? 4 : 0;
        CUDA_CHECK(cudaMemset(wsA, 0xAB, wsb - 8));
        CUDA_CHECK(cudaMemset(wsB, 0xAB, wsb - 8));
        dsk::moe(dq, dqs, 8, didx, dwts, mw, wsA, yA, c8, 0);
        dsk::moe(dq + (size_t)8 * D, dqs + (size_t)8 * D / 32, m - 8, didx + 48, dwts + 48, mw, wsB, yB, c8, 0);
        const auto ya = host(yA, (size_t)8 * D), yb = host(yB, (size_t)(m - 8) * D);
        const auto hA = host(wsA, wsb), hB = host(wsB, wsb);
        for (int up16 = 0; up16 < 5; ++up16) {
          if ((up16 == 2 || up16 == 3) != (wk8 == 1)) continue;  // up16 2 and 3 have WK 8
          size_t dy = 0, dh = 0;
          for (int down16 = 0; down16 < 5; ++down16) {
            dsk::MoeCfg c{};
            c.up16 = up16, c.down16 = down16;
            CUDA_CHECK(cudaMemset(ws16, 0xAB, wsb - 8));
            CUDA_CHECK(cudaMemset(y16, 0xFF, (size_t)16 * D * 4));
            dsk::moe(dq, dqs, m, didx, dwts, mw, ws16, y16, c, 0);
            const auto h = host(y16, (size_t)m * D);
            for (int r = 0; r < m; ++r)
              dy += memcmp(&h[(size_t)r * D], r < 8 ? &ya[(size_t)r * D] : &yb[(size_t)(r - 8) * D], (size_t)D * 4) != 0;
            if (down16) continue;
            // moe_up's output, row by row through the unions' slots (the shared expert's: 96 and 48)
            const auto h16 = host(ws16, wsb);
            const uint8_t* hs16 = h16.data() + dsk::moe_hs_offset(m);
            for (int r = 0; r < m; ++r) {
              const bool a = r < 8;
              const auto& h8 = a ? hA : hB;
              const auto& u8 = a ? uA : uB;
              const size_t r8 = a ? r : r - 8;
              const uint8_t* hs8 = h8.data() + dsk::moe_hs_offset(a ? 8 : m - 8);
              for (int k = 0; k <= 6; ++k) {
                const int e = k < 6 ? hi[r * 6 + k] : -1;
                const size_t s16 = k < 6 ? std::lower_bound(u16.begin(), u16.end(), e) - u16.begin() : 96;
                const size_t s8 = k < 6 ? std::lower_bound(u8.begin(), u8.end(), e) - u8.begin() : 48;
                dh += memcmp(&h16[(s16 * 16 + r) * I], &h8[(s8 * 8 + r8) * I], I) != 0;
                dh += memcmp(&hs16[(s16 * 16 + r) * (I / 32)], &hs8[(s8 * 8 + r8) * (I / 32)], I / 32) != 0;
              }
            }
          }
          const bool ok = dy == 0 && dh == 0;
          g_fail += !ok;
          char what[96];
          snprintf(what, sizeof what, "moe %s M=%d up16 %d down16 0..4", rname[routing], m, up16);
          printf("  %-40s %s union %3zu: y rows differing %zu, intermediate rows %zu\n", what, ok ? "ok  " : "FAIL",
                 u16.size(), dy, dh);
        }
      }
    }
  }

  // moe_routed (the top-6 from the logits inside) == router_topk + moe; the task queue == the static
  // schedule (y and the workspace, the counters back at zero: twice), at 16 rows
  std::vector<float> hl((size_t)16 * kPoolE), hb(kPoolE);
  uint32_t r = 4321;
  auto rnd = [&]() { r = r * 1664525u + 1013904223u; return (float)(r >> 8) / (float)(1u << 24); };
  for (auto& v : hl) v = 6.f * rnd() - 3.f;
  for (auto& v : hb) v = 0.2f * rnd() - 0.1f;
  float *dl = dev_alloc<float>(hl.size()), *db = dev_alloc<float>(kPoolE), *dw2 = dev_alloc<float>(96);
  float* y2 = dev_alloc<float>((size_t)16 * D);
  int* di2 = dev_alloc<int>(96);
  CUDA_CHECK(cudaMemcpy(dl, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
  for (int m : {9, 16})
    for (int up16 : {0, 2}) {
      dsk::MoeCfg cfg{};
      cfg.up16 = up16;
      dsk::router_topk(dl, db, m, kPoolE, didx, dwts, 1.5f, 0);
      dsk::moe(dq, dqs, m, didx, dwts, pool, ws16, y16, cfg, 0);
      dsk::moe_routed(dq, dqs, m, dl, db, kPoolE, 1.5f, pool, wsA, y2, cfg, 0, di2, dw2);
      const bool same_sel = host(didx, m * 6) == host(di2, m * 6) && host(dwts, m * 6) == host(dw2, m * 6);
      const size_t diff = diffd(y16, y2, (size_t)m * D);
      const bool ok = same_sel && diff == 0;
      g_fail += !ok;
      char what[96];
      snprintf(what, sizeof what, "moe_routed M=%d up16 %d", m, up16);
      printf("  %-40s %s outputs differing %zu, selection %s\n", what, ok ? "ok  " : "FAIL", diff,
             same_sel ? "equal" : "DIFFERS");
    }
  auto once = [&](const dsk::MoeCfg& c, float* yo) {  // didx/dwts: the router's picks for 16 rows
    CUDA_CHECK(cudaMemset(ws16, 0xAB, wsb - 8));     // the counters (zero) stay
    CUDA_CHECK(cudaMemset(yo, 0xFF, (size_t)16 * D * 4));
    dsk::moe(dq, dqs, 16, didx, dwts, pool, ws16, yo, c, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    return std::make_pair(host(yo, (size_t)16 * D), host(ws16, wsb));
  };
  for (int up16 : {0, 2}) {
    size_t runs = 0, bad = 0;
    for (int grid : {0, 1, 7, 13, 48, 200}) {
      dsk::MoeCfg cs{};
      cs.up16 = up16, cs.up_grid = grid;
      dsk::MoeCfg cd = cs;
      cd.dyn = true;
      const auto a = once(cs, y16);
      for (int rep = 0; rep < 2; ++rep) {
        const auto b = once(cd, y2);
        ++runs;
        bad += memcmp(a.first.data(), b.first.data(), a.first.size() * 4) != 0 || a.second != b.second;
      }
    }
    g_fail += bad != 0;
    char what[96];
    snprintf(what, sizeof what, "moe queue == static M=16 up16 %d", up16);
    printf("  %-40s %s %zu runs (grids 0/1/7/13/48/200)%s\n", what, bad ? "FAIL" : "ok  ", runs,
           bad ? ", outputs or intermediate DIFFER" : "");
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(dl), cudaFree(db), cudaFree(dw2), cudaFree(y2), cudaFree(di2);
  cudaFree(const_cast<uint8_t*>(pool.w1)), cudaFree(const_cast<uint8_t*>(pool.s1));
  cudaFree(const_cast<uint8_t*>(pool.w3)), cudaFree(const_cast<uint8_t*>(pool.s3));
  cudaFree(const_cast<uint8_t*>(pool.w2)), cudaFree(const_cast<uint8_t*>(pool.s2));
  cudaFree(dx), cudaFree(dq), cudaFree(dqs), cudaFree(ws16), cudaFree(wsA), cudaFree(wsB), cudaFree(y16), cudaFree(yA);
  cudaFree(yB), cudaFree(didx), cudaFree(dwts);
}

// The experts' 1-bit scale code (dsk::scale_code; MoeWeights c1, c3, c2): moe() reading it == moe() on the
// raw scales, bit for bit (y, and moe_up's output in the workspace), with the engine's variants: up 0, down 2
// at one row and 4 above, up16 4 and down16 2 at 9..16 rows; and moe_prefill likewise (y), its combine fused or
// not, with and without L2 hints. Two sets of experts: the reference ones (layer 3: few raw groups) and a pool
// of 128 random ones whose scales put expert e's 32-scale groups off the pair with probability (e % 12) / 11,
// so units carry from none to all of their groups raw (every path of the decode).
static void test_moe_code() {
  Npy x = L("moe_x"), sel = L("moe_sel");
  Npy w1 = L("moe_w1"), s1 = L("moe_s1"), w3 = L("moe_w3"), s3 = L("moe_s3"), w2 = L("moe_w2"), s2 = L("moe_s2");
  Npy shw1 = L("moe_shw1"), shs1 = L("moe_shs1"), shw3 = L("moe_shw3"), shs3 = L("moe_shs3");
  Npy shw2 = L("moe_shw2"), shs2 = L("moe_shs2");
  const int E0 = w1.shape[0], I = w1.shape[1], D = x.shape[1];
  printf("moe scale code: moe() on the code against the raw scales, %d reference experts and 128 random ones\n", E0);
  if (x.shape[0] != 8 || I != 576 || D != 5120) {
    printf("  moe reference: %d x %d rows, I=%d, expected 8 x 5120, 576\n", (int)x.shape[0], D, I);
    ++g_fail;
    return;
  }
  auto tile4 = [&](const Npy& w, const Npy& s, int N, int K, const uint8_t*& wt, const uint8_t*& st) {
    uint8_t *dw = dev(w), *ds = dev(s), *t = dev_alloc<uint8_t>(w.data.size()), *ts = dev_alloc<uint8_t>(s.data.size());
    dsk::tile_fp4(dw, ds, t, ts, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw), cudaFree(ds);
    wt = t, st = ts;
  };
  auto tile8 = [&](const Npy& w, int N, int K) {
    uint8_t *dw = dev(w), *t = dev_alloc<uint8_t>(w.data.size());
    dsk::tile_fp8(dw, t, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dw);
    return (const uint8_t*)t;
  };
  dsk::MoeWeights ref{};
  tile4(w1, s1, E0 * I, D, ref.w1, ref.s1);
  tile4(w3, s3, E0 * I, D, ref.w3, ref.s3);
  tile4(w2, s2, E0 * D, I, ref.w2, ref.s2);
  ref.sh_w1 = tile8(shw1, I, D), ref.sh_s1 = dev(shs1);
  ref.sh_w3 = tile8(shw3, I, D), ref.sh_s3 = dev(shs3);
  ref.sh_w2 = tile8(shw2, D, I), ref.sh_s2 = dev(shs2);
  constexpr int kPoolE = 128;
  const size_t w13 = (size_t)I * D / 2, s13 = (size_t)I * D / 32, w2b = (size_t)D * I / 2, s2b = (size_t)D * I / 32;
  uint32_t r = 2468;
  auto rnd = [&]() { r = r * 1664525u + 1013904223u; return r >> 8; };
  auto synth = [&](size_t per) {  // [kPoolE][per] scales, already tiled: groups are 32 consecutive bytes
    std::vector<uint8_t> h(kPoolE * per);
    for (int e = 0; e < kPoolE; ++e)
      for (size_t gi = 0; gi < per / 32; ++gi) {
        uint8_t* grp = &h[e * per + gi * 32];
        for (int i = 0; i < 32; ++i) grp[i] = rnd() % 5 < 3 ? 120 : 121;
        if (rnd() % 11 < (uint32_t)(e % 12)) grp[rnd() % 32] = (uint8_t)(116 + rnd() % 4 + (rnd() & 1) * 6);
      }
    uint8_t* d = dev_alloc<uint8_t>(h.size());
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    return (const uint8_t*)d;
  };
  dsk::MoeWeights pool = ref;  // the shared expert's stays
  pool.w1 = dev_fill16(kPoolE * w13, 1), pool.s1 = synth(s13);
  pool.w3 = dev_fill16(kPoolE * w13, 3), pool.s3 = synth(s13);
  pool.w2 = dev_fill16(kPoolE * w2b, 5), pool.s2 = synth(s2b);
  auto coded = [&](const dsk::MoeWeights& w, int E, const char* name) {
    dsk::MoeWeights c = w;
    const size_t n13 = dsk::scale_code_words(E, I, D / 32), n2 = dsk::scale_code_words(E, D, I / 32);
    uint32_t* code = dev_alloc<uint32_t>(2 * n13 + n2);
    int b1, b3, b2;
    const double f1 = dsk::scale_code(w.s1, E, I, D / 32, code, &b1, 0);
    const double f3 = dsk::scale_code(w.s3, E, I, D / 32, code + n13, &b3, 0);
    const double f2 = dsk::scale_code(w.s2, E, D, I / 32, code + 2 * n13, &b2, 0);
    c.c1 = {code, (uint32_t)b1 * 0x01010101u};
    c.c3 = {code + n13, (uint32_t)b3 * 0x01010101u};
    c.c2 = {code + 2 * n13, (uint32_t)b2 * 0x01010101u};
    printf("  %-10s groups coded: w1 %.2f %% ({%d, %d}), w3 %.2f %% ({%d, %d}), w2 %.2f %% ({%d, %d})\n", name, 100 * f1,
           b1, b1 + 1, 100 * f3, b3, b3 + 1, 100 * f2, b2, b2 + 1);
    return c;
  };
  const dsk::MoeWeights refc = coded(ref, E0, "reference"), poolc = coded(pool, kPoolE, "pool");

  auto* dx = reinterpret_cast<__nv_bfloat16*>(dev_rows16(x.as<uint16_t>(), D));
  uint8_t *dq = dev_alloc<uint8_t>((size_t)16 * D), *dqs = dev_alloc<uint8_t>((size_t)16 * D / 32);
  dsk::act_quant(dx, 16, D, dq, dqs, 0);
  const size_t wsb = dsk::moe_workspace_bytes();
  uint8_t *wsA = dev_alloc<uint8_t>(wsb), *wsB = dev_alloc<uint8_t>(wsb);
  float *yA = dev_alloc<float>((size_t)16 * D), *yB = dev_alloc<float>((size_t)16 * D);
  int* didx = dev_alloc<int>(96);
  float* dwts = dev_alloc<float>(96);
  // routings: the reference picks (rows 8..15 those of rows 7..0); 96 distinct experts of the pool; rows
  // sharing the pool's experts (ids < 40)
  const char* rname[] = {"ref", "96 experts", "shared"};
  for (int routing = 0; routing < 3; ++routing) {
    const dsk::MoeWeights &mw = routing == 0 ? ref : pool, &mc = routing == 0 ? refc : poolc;
    std::vector<int> hi(96);
    std::vector<float> hw(96);
    for (int rr = 0; rr < 16; ++rr)
      for (int k = 0; k < 6; ++k) {
        const int i = rr * 6 + k;
        hi[i] = routing == 0 ? sel.as<int>()[src8(rr) * 6 + k] : routing == 1 ? (i * 37 + 11) % kPoolE : (rr * 3 + k * 5) % 40;
        hw[i] = 0.05f + 0.9f * (float)((i * 7919) % 101) / 101.f;
      }
    CUDA_CHECK(cudaMemcpy(didx, hi.data(), 96 * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dwts, hw.data(), 96 * 4, cudaMemcpyHostToDevice));
    for (int m : {1, 6, 8, 9, 12, 16}) {
      dsk::MoeCfg cfg{};
      cfg.down = m == 1 ? 2 : 4;
      CUDA_CHECK(cudaMemset(wsA, 0xAB, wsb - 8));
      CUDA_CHECK(cudaMemset(wsB, 0xAB, wsb - 8));
      CUDA_CHECK(cudaMemset(yA, 0xFF, (size_t)16 * D * 4));
      CUDA_CHECK(cudaMemset(yB, 0xFF, (size_t)16 * D * 4));
      dsk::moe(dq, dqs, m, didx, dwts, mw, wsA, yA, cfg, 0);
      dsk::moe(dq, dqs, m, didx, dwts, mc, wsB, yB, cfg, 0);
      const auto ya = host(yA, (size_t)m * D), yb = host(yB, (size_t)m * D);
      const auto ha = host(wsA, wsb - 8), hb = host(wsB, wsb - 8);
      size_t dy = 0, dh = 0;
      for (size_t i = 0; i < ya.size(); ++i) dy += memcmp(&ya[i], &yb[i], 4) != 0;
      for (size_t i = 0; i < ha.size(); ++i) dh += ha[i] != hb[i];
      const bool ok = dy == 0 && dh == 0;
      g_fail += !ok;
      char what[96];
      snprintf(what, sizeof what, "moe code %s M=%d", rname[routing], m);
      printf("  %-40s %s y values differing %zu, workspace bytes %zu\n", what, ok ? "ok  " : "FAIL", dy, dh);
    }
  }

  // moe_prefill: the reference experts with p_moe's picks, and the pool with 256 rows of random picks
  Npy px = L("p_moe_x"), psel = L("p_moe_sel"), prw = L("p_moe_rw");
  const int PM = px.shape[0];
  QAct pa = quantised(px);
  std::vector<int> pi(PM * 6);
  std::vector<float> pw(PM * 6);
  for (int rr = 0; rr < PM; ++rr)
    for (int k = 0; k < 6; ++k) {  // six distinct experts a row
      int id;
      bool dup;
      do {
        id = (int)(rnd() % kPoolE);
        dup = false;
        for (int j = 0; j < k; ++j) dup = dup || pi[rr * 6 + j] == id;
      } while (dup);
      pi[rr * 6 + k] = id;
      pw[rr * 6 + k] = 0.05f + 0.9f * (float)(rnd() % 101) / 101.f;
    }
  int* pidx = dev_alloc<int>(PM * 6);
  float* pwts = dev_alloc<float>(PM * 6);
  CUDA_CHECK(cudaMemcpy(pidx, pi.data(), pi.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pwts, pw.data(), pw.size() * 4, cudaMemcpyHostToDevice));
  int* ridx = dev<int>(psel);
  float* rwts = dev<float>(prw);
  const size_t pwsb = dsk::moe_prefill_workspace_bytes(PM, kPoolE);
  uint8_t* pws = dev_alloc<uint8_t>(pwsb);
  float *py = dev_alloc<float>((size_t)PM * D), *py2 = dev_alloc<float>((size_t)PM * D);
  for (int set = 0; set < 2; ++set) {
    const dsk::MoeWeights &mw = set ? pool : ref, &mc = set ? poolc : refc;
    const int E = set ? kPoolE : E0;
    const int* di = set ? pidx : ridx;
    const float* dw = set ? pwts : rwts;
    for (int v = 0; v < 3; ++v) {  // the default; the combine separate, no L2 hints; fused, hints
      dsk::PrefillTuning t;
      if (v) t.down = v, t.l2 = v == 1 ? 0 : 7, t.dgrp = 0;
      dsk::prefill_tuning(t);
      for (int m : {256, 200, 129, 64, 1}) {
        if (m > PM) continue;
        CUDA_CHECK(cudaMemset(py, 0xFF, (size_t)PM * D * 4));
        CUDA_CHECK(cudaMemset(py2, 0xFF, (size_t)PM * D * 4));
        dsk::moe_prefill(pa.q, pa.s, m, di, dw, E, mw, pws, py, 0);
        dsk::moe_prefill(pa.q, pa.s, m, di, dw, E, mc, pws, py2, 0);
        const auto ya = host(py, (size_t)m * D), yb = host(py2, (size_t)m * D);
        size_t dy = 0;
        for (size_t i = 0; i < ya.size(); ++i) dy += memcmp(&ya[i], &yb[i], 4) != 0;
        const bool ok = dy == 0;
        g_fail += !ok;
        char what[96];
        const char* vn[] = {"", ", separate, no hints", ", fused, hints"};
        snprintf(what, sizeof what, "moe_prefill code %s M=%d%s", set ? "pool" : "ref", m, vn[v]);
        printf("  %-52s %s y values differing %zu\n", what, ok ? "ok  " : "FAIL", dy);
      }
    }
    dsk::prefill_tuning(dsk::PrefillTuning{});
  }
}

// The head coded (dsk::bf16x_encode: each tile of 8 rows x 32 values with its exponents as 4-bit offsets
// from its least, a tile spanning more than 16 exponents kept raw): it decodes to the head exactly
// (bf16x_check, which must also see a flipped bit), it keeps raw the tiles a host scan says, and its GEMVs
// give the bf16 head's bits at 1..16 rows (rows 8..15 = rows 7..0), mt 2 (the model's) and 1, normed (the
// model's head: gemv_bf16_norm) and plain (DSpark's: gemv_bf16). On the rank-0 head slice, and on a copy
// with planted tiles of every kind: zeros, -0, subnormals, inf and NaNs, exponent spans of 15 and 16.
static uint16_t bf16_bits(int s, int e, int m) { return (uint16_t)((s << 15) | (e << 7) | m); }

static void test_head4() {
  Npy w = L("head_w"), x8 = L("head_x"), nw = L("norm_w");
  const int N = w.shape[0], K = w.shape[1], P = 2 * K / 64;
  if (x8.shape[0] != 8 || (int)x8.shape[1] != K || (int)nw.shape[0] != K) {
    printf("head4: head_x / norm_w do not match the head [%d x %d]\n", N, K);
    ++g_fail;
    return;
  }
  printf("head coded (bf16x): [%d x %d], %d tiles of 8 x 32\n", N, K, N / 8 * P);
  // the planted copy, in the natural layout: tile (rb, p) is rows 8 rb .. +7, values 32 p .. +31
  std::vector<uint16_t> pl(w.as<uint16_t>(), w.as<uint16_t>() + (size_t)N * K);
  auto put = [&](int rb, int p, const std::function<uint16_t(int, int)>& f) {
    for (int r = 0; r < 8; ++r)
      for (int k = 0; k < 32; ++k) pl[(size_t)(8 * rb + r) * K + 32 * p + k] = f(r, k);
  };
  put(0, 0, [](int, int) { return bf16_bits(0, 0, 0); });                                  // +0
  put(1, 1, [](int r, int k) { return bf16_bits((r + k) & 1, 0, 0); });                     // +0, -0
  put(2, 2, [](int r, int k) { return bf16_bits(k & 1, 0, (r * 32 + k) % 127 + 1); });     // subnormals
  put(3, 3, [](int r, int k) { return bf16_bits(r & 1, k % 16, (r * 7 + k) % 128); });     // exponents 0..15
  put(4, 4, [](int r, int k) {  // 240..255: large values, inf and NaNs
    const int e = 240 + (r * 32 + k) % 16;
    return bf16_bits(k & 1, e, e < 255 ? (k * 5) % 128 : k % 3 ? (r * 32 + k) % 127 + 1 : 0);
  });
  put(5, 5, [](int r, int k) { return bf16_bits(r & 1, 100 + (r + k) % 16, k); });        // a span of 15
  put(6, 6, [](int r, int k) { return bf16_bits(r & 1, 100 + (r + k) % 17, k); });        // 16: raw
  pl[(size_t)(8 * 7 + 3) * K + 32 * 7 + 5] = 0;                                            // a zero: raw
  put(2000, 80, [](int r, int k) { return bf16_bits(k & 1, 1 + (r * 3 + k) % 16, r * 16 + k % 16); });
  put(N / 8 - 1, P - 1, [](int r, int k) { return bf16_bits((r ^ k) & 1, 254, 127); });   // +-max
  auto host_raw = [&](const uint16_t* v) {  // the tiles whose exponents span more than 16 values
    int n = 0;
    for (int rb = 0; rb < N / 8; ++rb)
      for (int p = 0; p < P; ++p) {
        int lo = 255, hi = 0;
        for (int r = 0; r < 8; ++r)
          for (int k = 0; k < 32; ++k) {
            const int e = (v[(size_t)(8 * rb + r) * K + 32 * p + k] >> 7) & 0xFF;
            lo = std::min(lo, e), hi = std::max(hi, e);
          }
        n += hi - lo > 15;
      }
    return n;
  };
  uint16_t* x = dev_rows16(x8.as<uint16_t>(), K);
  const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x);
  auto* dnw = dev<__nv_bfloat16>(nw);
  float *y1 = dev_alloc<float>((size_t)16 * N), *y2 = dev_alloc<float>((size_t)16 * N);
  uint8_t* nat = dev_alloc<uint8_t>((size_t)N * K * 2);
  for (int c = 0; c < 2; ++c) {
    const uint16_t* hv = c ? pl.data() : w.as<uint16_t>();
    CUDA_CHECK(cudaMemcpy(nat, hv, (size_t)N * K * 2, cudaMemcpyHostToDevice));
    uint8_t* W = dev_alloc<uint8_t>((size_t)N * K * 2);
    dsk::tile_fp8(nat, W, N, 2 * K, 0);
    size_t bytes = 0;
    int raw = 0;
    uint8_t* X = dsk::bf16x_encode(W, N, K, &bytes, &raw, 0);
    const uint64_t bad = dsk::bf16x_check(X, W, N, K, 0);
    const int hraw = host_raw(hv);
    // one bit of a coded tile's mantissa flipped: one lane's 16 bytes decode wrong
    std::vector<uint32_t> hd = host(reinterpret_cast<const uint32_t*>(X + (size_t)N / 8 * P * 384), 64);
    const int t = (int)(std::find_if(hd.begin(), hd.end(), [](uint32_t h) { return h >> 31 == 0; }) - hd.begin());
    uint8_t b8;
    uint8_t* at = X + (size_t)t * 256 + 8 * 5 + 2;
    CUDA_CHECK(cudaMemcpy(&b8, at, 1, cudaMemcpyDeviceToHost));
    b8 ^= 1;
    CUDA_CHECK(cudaMemcpy(at, &b8, 1, cudaMemcpyHostToDevice));
    const uint64_t bad1 = dsk::bf16x_check(X, W, N, K, 0);
    b8 ^= 1;
    CUDA_CHECK(cudaMemcpy(at, &b8, 1, cudaMemcpyHostToDevice));
    const bool ok = bad == 0 && raw == hraw && bad1 == 1 && t < 64;
    g_fail += !ok;
    printf("  %-8s %s %.1f of %.1f MB (%.4f), %d tiles raw (host scan: %d); decoded wrong: %llu pieces, %llu with a bit "
           "flipped\n",
           c ? "planted" : "head", ok ? "ok  " : "FAIL", bytes / 1e6, (double)N * K * 2 / 1e6,
           (double)bytes / ((double)N * K * 2), raw, hraw, (unsigned long long)bad, (unsigned long long)bad1);
    for (int norm = 1; norm >= 0; --norm)
      for (int mt : {2, 1}) {
        size_t d[2] = {0, 0};
        for (int M = 1; M <= 16; ++M) {
          const dsk::GemvCfg cfg{mt, 1, 16};
          CUDA_CHECK(cudaMemset(y2, 0xFF, (size_t)16 * N * 4));
          if (norm) {
            dsk::gemv_bf16_norm(xb, dnw, 1e-6f, M, W, N, K, y1, nullptr, cfg, 0);
            dsk::gemv_bf16x_norm(xb, dnw, 1e-6f, M, X, N, K, y2, nullptr, cfg, 0);
          } else {
            dsk::gemv_bf16(xb, M, W, N, K, y1, nullptr, cfg, false, 0);
            dsk::gemv_bf16x(xb, M, X, N, K, y2, nullptr, cfg, 0);
          }
          d[M > 8] += diffd(y1, y2, (size_t)M * N);
        }
        const bool okg = d[0] == 0 && d[1] == 0;
        g_fail += !okg;
        printf("    %-16s mt=%d %s %zu logits differ at 1..8 rows, %zu at 9..16\n", norm ? "gemv_bf16x_norm" : "gemv_bf16x",
               mt, okg ? "ok  " : "FAIL", d[0], d[1]);
      }
    cudaFree(X), cudaFree(W);
  }
  cudaFree(nat), cudaFree(x), cudaFree(dnw), cudaFree(y1), cudaFree(y2);
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // lines reach a log file even if a check aborts
  g_dir = argc > 1 && argv[1][0] ? argv[1] : std::string(getenv("HOME")) + "/dsgb10-data/kerneltests";
  std::string only = argc > 2 ? argv[2] : "";
  if (only.empty() || only == "act_quant") test_act_quant();
  for (const char* n : {"wq_a", "wkv", "wq_b", "wo_b", "sh_w1", "sh_w2", "ix_wq_b"})
    if (only.empty() || only == "fp8" || only == n) test_gemv_fp8(n);
  if (only.empty() || only == "moe") test_moe();
  if (only.empty() || only == "moe" || only == "mcode") test_moe_code();
  if (only.empty() || only == "woa") test_woa();
  if (only.empty() || only == "head") test_head();
  if (only.empty() || only == "head4") test_head4();
  if (only.empty() || only == "norm") test_norm();
  if (only.empty() || only == "norm" || only == "normgemv") test_norm_gemv();
  if (only.empty() || only == "fp8" || only == "fp8p" || only == "pers") test_gemv_fp8p();
  if (only.empty() || only == "g16p" || only == "pers") test_gemv16p();
  if (only.empty() || only == "router") test_router();
  if (only.empty() || only == "rows16") test_rows16_gemv();
  if (only.empty() || only == "rows16" || only == "rows16moe") test_rows16_moe();
  if (only.empty() || only == "prefill" || only == "p_fp8") test_prefill_fp8();
  if (only.empty() || only == "prefill" || only == "p_fp8" || only == "p_small") g_fail += small_gemms::run(false);
  if (only.empty() || only == "prefill" || only == "p_16") test_prefill_16();
  if (only.empty() || only == "prefill" || only == "p_moe") test_prefill_moe();
  printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
  return g_fail != 0;
}
