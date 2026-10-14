// Tests of the routed-expert kernels against CPU references on random data in the real formats.
//   test_experts --cpu            host only: decoders, packers and readers, the routing reference
//   test_experts [what...]        GPU (only through ring.sh): deq route decode prefill (default: all)
// Exit status 0 when every check passes. Every check prints one "CHECK name ok|FAIL detail" line.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <omp.h>

#include "experts/formats.cuh"
#include "experts/tests/testdata.h"

using namespace gx;

static int g_fail = 0;
static FILE* g_out = nullptr;
static std::string g_tag;  // prefixed to every check's name (the real-expert runs: "real l3r1: ")
static void check(const char* name0, bool ok, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  const std::string nm = g_tag + name0;
  const char* name = nm.c_str();
  printf("CHECK %-44s %s  %s\n", name, ok ? "ok  " : "FAIL", buf);
  fflush(stdout);
  if (g_out) fprintf(g_out, "CHECK\t%s\t%s\t%s\n", name, ok ? "ok" : "FAIL", buf), fflush(g_out);
  if (!ok) ++g_fail;
}
template <class T>
static std::vector<T> d2h(const T* p, size_t n) {
  std::vector<T> v(n);
  CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
template <class T>
static T* h2d(const std::vector<T>& v) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, v.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
static float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
static uint16_t bits(__nv_bfloat16 v) { uint16_t b; memcpy(&b, &v, 2); return b; }
static uint32_t fbits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

// ------------------------------------------------------------------------------------------------------------------
// host-only checks
static void cpu_checks() {
  {  // NVFP4 widen: every nibble, scales 0x08..0x7e
    int bad = 0;
    for (int sc = 0x08; sc < 0x7F; sc += 3)
      for (int r = 0; r < 64; ++r) {
        const uint32_t lo = (uint32_t)mix64(r * 7 + sc), hi = (uint32_t)mix64(r * 7 + sc + 1000);
        uint32_t o[8];
        q4_widen16(lo, hi, q4_scale_f2_host((uint8_t)sc), o);
        for (int i = 0; i < 16; ++i) {
          const int nib = ((i < 8 ? lo : hi) >> 4 * (i % 8)) & 15;
          const uint16_t want = f_to_bf16_bits(e2m1_to_f(nib) * e4m3_to_f((uint8_t)sc) * 0x1p-30f);
          const uint16_t got = (o[i / 2] >> 16 * (i & 1)) & 0xFFFF;
          bad += want != got && !(nib == 8 && got == 0x8000);
        }
      }
    check("cpu nvfp4 widen", !bad, "%d mismatches", bad);
  }
  {  // S3: pack a lane, decode through selectors + prmt, both subsets
    int bad = 0;
    for (int sub = 0; sub < 3; ++sub) {
      const float* mg = sub == 2 ? kSubC : sub ? kSubB : kSubA;
      const S3Tab T = s3_table(mg);
      for (int r = 0; r < 2048; ++r) {
        uint8_t v[32];
        for (int i = 0; i < 32; ++i) v[i] = (uint8_t)(mix64(r * 33 + i + sub) & 7);
        uint32_t P0, P1, Q;
        s3_pack_lane(v, P0, P1, Q);
        const uint8_t sc = (uint8_t)(0x30 + r % 0x18);
        for (int q = 0; q < 2; ++q) {
          uint32_t o[8];
          s3_group(P0, P1, Q, q, T, s3_scale_f2_host(sc), o);
          for (int i = 0; i < 16; ++i) {
            const int c = v[16 * q + i];
            const uint16_t want = f_to_bf16_bits((c & 4 ? -1.f : 1.f) * mg[c & 3] * e4m3_to_f(sc));
            bad += want != ((o[i / 2] >> 16 * (i & 1)) & 0xFFFF);
          }
        }
      }
    }
    check("cpu s3 pack + prmt decode (subsets A, B, C)", !bad, "%d mismatches", bad);
  }
  {  // FP8 widen, every non-NaN byte
    int bad = 0;
    for (int v = 0; v < 256; ++v) {
      if ((v & 0x7F) == 0x7F) continue;
      uint32_t o0, o1;
      fp8x4_to_bf16((uint32_t)v * 0x01010101u, o0, o1);
      const uint16_t want = f_to_bf16_bits(e4m3_to_f((uint8_t)v));
      bad += (o0 & 0xFFFF) != want || (o0 >> 16) != want || (o1 & 0xFFFF) != want || (o1 >> 16) != want;
    }
    check("cpu fp8 widen (254 values)", !bad, "%d mismatches", bad);
  }
  {  // the S3 nibble table (FP4 path): the nibble's e2m1 value equals the code's value
    int bad = 0;
    for (int sub = 0; sub < 3; ++sub) {
      const float* mg = sub == 2 ? kSubC : sub ? kSubB : kSubA;
      const uint32_t t = s3_nibble_table(mg);
      for (int c = 0; c < 8; ++c) {
        const float want = (c & 4 ? -1.f : 1.f) * mg[c & 3];
        bad += e2m1_to_f((t >> 4 * c) & 15) != want;
      }
    }
    check("cpu s3 code -> e2m1 nibble table", !bad, "%d mismatches", bad);
  }
  {  // the loader's descriptor check (glw_kcheck): S3x2 accepted as the pair, another pair accepted as not the pair,
     // a damaged table or nibble word, a non-E2M1 or descending subset refused
    int bad = 0;
    float mg[2][4];
    bool pair = false;
    const MatDesc a = make_desc_s3x2(4096, 0.01f);
    bad += !s3_desc_check(a, mg, &pair) || !pair || memcmp(mg, kS3x2Mags, sizeof mg);
    const float ac[2][4] = {{kSubA[0], kSubA[1], kSubA[2], kSubA[3]}, {kSubC[0], kSubC[1], kSubC[2], kSubC[3]}};
    bad += !s3_desc_check(make_desc(0, 1.f, kS3, 2, ac), mg, &pair) || pair;
    MatDesc d = a;
    d.tab[5] ^= 0x100;  // a low byte of subset B
    bad += s3_desc_check(d, mg);
    d = a;
    d.nib[1] ^= 1;
    bad += s3_desc_check(d, mg);
    const float desc_[2][4] = {{0.5f, 4.f, 2.f, 6.f}, {0.5f, 1.5f, 3.f, 6.f}}, offg[2][4] = {{0.5f, 2.f, 4.f, 5.f}, {0.5f, 1.5f, 3.f, 6.f}};
    bad += s3_desc_check(make_desc(0, 1.f, kS3, 2, desc_), mg) + s3_desc_check(make_desc(0, 1.f, kS3, 2, offg), mg);
    bad += s3_desc_check(make_desc(0, 1.f, kNVFP4), mg);
    check("cpu s3 descriptor check (loader)", !bad, "%d wrong verdicts of 7", bad);
  }
  {  // packers against the canonical decode, through the layout readers
    int bad = 0;
    const int N = 24, K = 1024;
    HostMat q = make_q4(N, K, 5, 1.f), s = make_s3(N, K, 6, 1.f, 2, kSubA, kSubB);
    HostFp8 f = make_fp8(128, 256, 7, 1.f);
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        bad += read_q4(q.packed.data(), K, n, k) != q.value(n, k);
        bad += read_s3(s.packed.data(), s.mags, K, n, k) != s.value(n, k);
      }
    for (int n = 0; n < 128; ++n)
      for (int k = 0; k < 256; ++k) bad += read_fp8(f.packed.data(), 256, n, k) != e4m3_to_f(f.w[(size_t)n * 256 + k]);
    check("cpu packers vs canonical (nvfp4, s3, fp8)", !bad, "%d mismatches", bad);
  }
  {  // routing reference: ties go to the lower id, weights sum to 2.5
    std::vector<float> l, bias(kE);
    rnd_logits(l, 64, 3, true);
    for (int e = 0; e < kE; ++e) bias[e] = 34.2f + 0.001f * (float)(e % 7);
    std::vector<int> idx(64 * kTopK);
    std::vector<float> w(64 * kTopK);
    route_ref(l.data(), bias.data(), 64, idx.data(), w.data());
    int bad = 0;
    for (int m = 0; m < 64; ++m) {
      double s = 0;
      for (int k = 0; k < kTopK; ++k) {
        s += w[m * kTopK + k];
        if (k) {
          const int a = idx[m * kTopK + k - 1], b = idx[m * kTopK + k];
          const float ka = route_sigmoid(l[m * kE + a]) + bias[a], kb = route_sigmoid(l[m * kE + b]) + bias[b];
          bad += !(ka > kb || (ka == kb && a < b));
        }
      }
      bad += fabs(s - 2.5) > 1e-5;
    }
    check("cpu routing reference (order, ties, sum)", !bad, "%d bad rows", bad);
  }
}

// ------------------------------------------------------------------------------------------------------------------
// GPU: the decode through the kernels' own fragment mapping, bit for bit
__global__ void k_deq(const uint8_t* m, MatDesc d, int N, int K, uint16_t* out) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), B = K / kBlkK;
  if (gw >= N / 8 * B) return;
  const int n8 = gw / B, b = gw % B;
  Raw r;
  load_blk(r, m + (size_t)gw * (d.fmt == kNVFP4 ? kQ4Blk : kS3Blk), d.fmt, lane);
  uint32_t tab[8];
  for (int j = 0; j < 8; ++j) tab[j] = d.tab[j];
  for (int jj = 0; jj < 4; ++jj)
    for (int q = 0; q < 2; ++q) {
      uint32_t o[8];
      dec_group(r, jj, q, d.fmt, tab, o);
      for (int mm = 0; mm < 8; ++mm) {
        const size_t base = (size_t)(8 * n8 + g) * K + 512 * b + 128 * jj + 64 * q + 16 * t + 2 * mm;
        out[base] = o[mm] & 0xFFFF;
        out[base + 1] = o[mm] >> 16;
      }
    }
}
static void test_deq() {
  for (int pat = 0; pat < 3; ++pat)
    for (int j = 0; j < 3; j += 2) {  // gate shape and down shape
      const int N = j < 2 ? kI : kH, K = j < 2 ? kH : kI;
      HostMat m = pat == 0 ? make_q4(N, K, 40 + j, 1.f) : make_s3(N, K, 50 + j + pat, 1.f, pat, kSubA, kSubB);
      const MatDesc d = make_desc(0, 1.f, m.fmt, m.nsub, m.mags);
      uint8_t* dm = h2d(m.packed);
      uint16_t* dout;
      CUDA_CHECK(cudaMalloc(&dout, (size_t)N * K * 2));
      const int warps = N / 8 * (K / kBlkK);
      k_deq<<<(warps + 7) / 8, 256>>>(dm, d, N, K, dout);
      CUDA_CHECK(cudaDeviceSynchronize());
      auto got = d2h(dout, (size_t)N * K);
      size_t bad = 0;
      for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
          const float want = m.value(n, k) * (m.fmt == kNVFP4 ? 0x1p-30f : 1.f);
          const uint16_t wb = f_to_bf16_bits(want), gb = got[(size_t)n * K + k];
          bad += wb != gb && !(want == 0 && (gb & 0x7FFF) == 0);
        }
      char name[64];
      snprintf(name, sizeof name, "deq %s%s [%d x %d]", m.fmt == kNVFP4 ? "nvfp4" : "s3", pat == 2 ? " 2 subsets" : "", N, K);
      check(name, !bad, "%zu of %zu mismatches (bf16 of value x scale%s)", bad, (size_t)N * K, m.fmt == kNVFP4 ? " x 2^-30" : "");
      CUDA_CHECK(cudaFree(dm));
      CUDA_CHECK(cudaFree(dout));
    }
}

static void test_route() {
  for (int ties = 0; ties < 2; ++ties) {
    const int M = 16 + 3 * ties;  // more rows than one step: the kernel takes any M
    std::vector<float> l, bias(kE);
    rnd_logits(l, M, 11 + ties, ties);
    for (int e = 0; e < kE; ++e) bias[e] = ties ? 34.25f : 34.2f + 0.37f * (float)((mix64(e) % 1000) / 1000.0);
    std::vector<int> ri(M * kTopK);
    std::vector<float> rw(M * kTopK);
    route_ref(l.data(), bias.data(), M, ri.data(), rw.data());
    float *dl = h2d(l), *db = h2d(bias), *dw;
    int* di;
    CUDA_CHECK(cudaMalloc(&di, M * kTopK * 4));
    CUDA_CHECK(cudaMalloc(&dw, M * kTopK * 4));
    route(dl, db, M, di, dw, 0, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gi = d2h(di, M * kTopK);
    auto gw = d2h(dw, M * kTopK);
    int bad = 0;
    for (int i = 0; i < M * kTopK; ++i) bad += gi[i] != ri[i] || fbits(gw[i]) != fbits(rw[i]);
    check(ties ? "route (many tied logits, equal bias)" : "route (random)", !bad, "%d of %d picks differ (ids and weight bits)", bad,
          M * kTopK);
    cudaFree(dl), cudaFree(db), cudaFree(di), cudaFree(dw);
  }
}

// ------------------------------------------------------------------------------------------------------------------
// decode: CPU reference in double
struct Ref {
  const TestLayer& L;
  std::map<int, std::vector<float>> deq;  // stored matrix index -> values without s2 [N][K]
  std::vector<float> sg, su, sd;          // shared, with block scales
  explicit Ref(const TestLayer& l) : L(l) {
    auto fp = [](const HostFp8& h, std::vector<float>& v) {
      v.resize((size_t)h.N * h.K);
#pragma omp parallel for
      for (int n = 0; n < h.N; ++n)
        for (int k = 0; k < h.K; ++k) v[(size_t)n * h.K + k] = h.value(n, k);
    };
    fp(L.sg, sg), fp(L.su, su), fp(L.sd, sd);
  }
  const std::vector<float>& mat(int e, int j) {  // expert id e, projection j
    const int si = L.map[e] * 3 + j;
    auto it = deq.find(si);
    if (it != deq.end()) return it->second;
    const HostMat& m = L.m[si];
    std::vector<float> v((size_t)m.N * m.K);
#pragma omp parallel for
    for (int n = 0; n < m.N; ++n)
      for (int k = 0; k < m.K; ++k) v[(size_t)n * m.K + k] = m.value(n, k);
    return deq.emplace(si, std::move(v)).first->second;
  }
  float s2(int e, int j) const { return L.m[L.map[e] * 3 + j].s2; }
};
static float silu_mul_ref(float g, float u) { return bf16r(bf16r(g / (1.f + expf(-g))) * u); }

struct ErrStat {
  double worst = 0;
  size_t n = 0, bad = 0;
  void add(double got, double ref, double mag, double tol) {
    const double e = fabs(got - ref) / (mag > 0 ? mag : 1);
    worst = std::max(worst, e);
    bad += !(e <= tol);
    ++n;
  }
};

// h (bf16) of every slot for M rows: gate/up from x in double, rounded as the kernel does
// lo/hi: the range h can take when gate and up each move by kSumTol x their sum of |terms| (the f32 sums of the
// kernel in another order), each end rounded as the kernel rounds
constexpr double kSumTol = 2e-5;
struct HRef {
  std::vector<float> h, lo, hi;
  bool ok(size_t i, float v) const { return v >= lo[i] && v <= hi[i]; }
};
static void ref_h(Ref& R, const std::vector<__nv_bfloat16>& x, int M, const std::vector<int>& uni, HRef& out) {
  const int U = (int)uni.size();
  std::vector<float>& h = out.h;
  h.assign((size_t)(U + 1) * M * kI, 0.f);
  out.lo.assign(h.size(), 0.f);
  out.hi.assign(h.size(), 0.f);
  for (int u = 0; u <= U; ++u) {
    const bool sh = u == U;
    const std::vector<float>& G = sh ? R.sg : R.mat(uni[u], 0);
    const std::vector<float>& Uw = sh ? R.su : R.mat(uni[u], 1);
    const double kg = sh ? 1.0 : R.s2(uni[u], 0), ku = sh ? 1.0 : R.s2(uni[u], 1);
#pragma omp parallel for
    for (int c = 0; c < kI; ++c)
      for (int m = 0; m < M; ++m) {
        double ga = 0, ua = 0, gm = 0, um = 0;
        for (int k = 0; k < kH; ++k) {
          const double xv = bf(x[(size_t)m * kH + k]);
          ga += xv * G[(size_t)c * kH + k];
          ua += xv * Uw[(size_t)c * kH + k];
          gm += fabs(xv * G[(size_t)c * kH + k]);
          um += fabs(xv * Uw[(size_t)c * kH + k]);
        }
        const size_t i = ((size_t)u * M + m) * kI + c;
        h[i] = silu_mul_ref(bf16r((float)(ga * kg)), bf16r((float)(ua * ku)));
        float lo = h[i], hi = h[i];
        const double g0 = (ga - kSumTol * gm) * kg, g1 = (ga + kSumTol * gm) * kg;
        const double u0 = (ua - kSumTol * um) * ku, u1 = (ua + kSumTol * um) * ku;
        const double gs[3] = {g0, g1, std::clamp(-1.2785, g0, g1)};  // the ends and silu's minimum when inside
        for (double gg : gs)
          for (double uu : {u0, u1}) {
            const float v = silu_mul_ref(bf16r((float)gg), bf16r((float)uu));
            lo = std::min(lo, v), hi = std::max(hi, v);
          }
        out.lo[i] = lo, out.hi[i] = hi;
      }
  }
}

struct DecodeRun {
  std::vector<int> uni;
  std::vector<__nv_bfloat16> h;  // [U + 1][M][kI], the shared expert last
  std::vector<float> yr, ys;
};
static DecodeRun run_decode(const TestLayer& L, const __nv_bfloat16* dx, const int* didx, const float* dw, int M,
                            const std::vector<int>& idx, void* ws, float* dyr, float* dys, const DecodeCfg& cfg) {
  CUDA_CHECK(cudaMemset(dyr, 0xFF, (size_t)M * kH * 4));
  CUDA_CHECK(cudaMemset(dys, 0xFF, (size_t)M * kH * 4));
  moe_decode(L.layer(), dx, didx, dw, M, ws, dyr, dys, 0, cfg);
  CUDA_CHECK(cudaDeviceSynchronize());
  DecodeRun r;
  std::set<int> s(idx.begin(), idx.begin() + M * kTopK);
  r.uni.assign(s.begin(), s.end());
  const int U = (int)r.uni.size();
  auto hb = d2h(reinterpret_cast<const __nv_bfloat16*>(ws), (size_t)(kMaxUnion + 1) * kMaxRows * kI);
  r.h.resize((size_t)(U + 1) * M * kI);
  for (int u = 0; u <= U; ++u) {
    const int slot = u == U ? kShSlot : u;
    memcpy(&r.h[(size_t)u * M * kI], &hb[(size_t)slot * M * kI], (size_t)M * kI * 2);
  }
  r.yr = d2h(dyr, (size_t)M * kH);
  r.ys = d2h(dys, (size_t)M * kH);
  return r;
}

static void test_decode(TestLayer* ext = nullptr) {  // ext: an uploaded layer (the real experts), else synthetic
  TestLayer syn;
  if (!ext) syn.build(10, 1), syn.upload();
  TestLayer& L = ext ? *ext : syn;
  Ref R(L);
  void* ws;
  CUDA_CHECK(cudaMalloc(&ws, decode_ws_bytes()));
  CUDA_CHECK(cudaMemset(ws, 0, decode_ws_bytes()));
  float *dyr, *dys, *dl, *db, *dw;
  int* di;
  __nv_bfloat16* dx;
  CUDA_CHECK(cudaMalloc(&dyr, (size_t)kMaxRows * kH * 4));
  CUDA_CHECK(cudaMalloc(&dys, (size_t)kMaxRows * kH * 4));
  CUDA_CHECK(cudaMalloc(&dl, (size_t)kMaxRows * kE * 4));
  CUDA_CHECK(cudaMalloc(&db, kE * 4));
  CUDA_CHECK(cudaMalloc(&di, kMaxRows * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dw, kMaxRows * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dx, (size_t)kMaxRows * kH * 2));
  std::vector<float> bias(kE);
  for (int e = 0; e < kE; ++e) bias[e] = 34.2f + 0.37f * (float)((mix64(e + 5) % 1000) / 1000.0);
  CUDA_CHECK(cudaMemcpy(db, bias.data(), kE * 4, cudaMemcpyHostToDevice));
  // 16 rows of x and their routing
  std::vector<__nv_bfloat16> x;
  rnd_x(x, (size_t)kMaxRows * kH, 21);
  std::vector<float> lg;
  rnd_logits(lg, kMaxRows, 22, false);
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dl, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
  route(dl, db, kMaxRows, di, dw, 0, false);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto idx = d2h(di, kMaxRows * kTopK);
  const auto wts = d2h(dw, kMaxRows * kTopK);

  // full check against the CPU at several row counts
  std::map<int, DecodeRun> runs;
  std::map<int, HRef> refs;
  for (int M : {1, 3, 5, 6, 8, 9, 16}) {
    DecodeRun r = run_decode(L, dx, di, dw, M, idx, ws, dyr, dys, DecodeCfg());
    const int U = (int)r.uni.size();
    HRef& hr = refs[M];
    ref_h(R, x, M, r.uni, hr);
    const std::vector<float>& h = hr.h;
    size_t exact = 0, far = 0;
    for (size_t i = 0; i < h.size(); ++i) {
      const float gv = bf(r.h[i]);
      exact += fbits(gv) == fbits(h[i]) || (gv == 0 && h[i] == 0);
      far += !hr.ok(i, gv);
    }
    char name[64];
    snprintf(name, sizeof name, "decode M=%d h (%d experts + shared)", M, U);
    check(name, far == 0 && exact >= h.size() * 0.995, "%zu of %zu bit-exact, %zu outside the range of f32 sums", exact, h.size(),
          far);
    // down from the GPU's h, in double
    ErrStat er, es;
    std::vector<double> yr((size_t)M * kH, 0), yrm((size_t)M * kH, 0), ys((size_t)M * kH, 0), ysm((size_t)M * kH, 0);
    for (int u = 0; u <= U; ++u) {
      const bool sh = u == U;
      const std::vector<float>& D = sh ? R.sd : R.mat(r.uni[u], 2);
      std::vector<double> wr(M, 0.0);
      for (int m = 0; m < M; ++m)
        for (int k = 0; k < kTopK; ++k)
          if (!sh && idx[m * kTopK + k] == r.uni[u]) wr[m] = wts[m * kTopK + k] * (double)R.s2(r.uni[u], 2);
#pragma omp parallel for
      for (int n = 0; n < kH; ++n)
        for (int m = 0; m < M; ++m) {
          if (!sh && wr[m] == 0) continue;
          double a = 0, am = 0;
          for (int k = 0; k < kI; ++k) {
            const double t = (double)bf(r.h[((size_t)u * M + m) * kI + k]) * D[(size_t)n * kI + k];
            a += t, am += fabs(t);
          }
          if (sh) ys[(size_t)m * kH + n] += a, ysm[(size_t)m * kH + n] += am;
          else yr[(size_t)m * kH + n] += a * wr[m], yrm[(size_t)m * kH + n] += am * fabs(wr[m]);
        }
    }
    for (size_t i = 0; i < yr.size(); ++i) er.add(r.yr[i], yr[i], yrm[i], 1e-4), es.add(r.ys[i], ys[i], ysm[i], 1e-4);
    snprintf(name, sizeof name, "decode M=%d routed out vs double (from GPU h)", M);
    check(name, !er.bad, "max |err|/sum|terms| %.2e over %zu", er.worst, er.n);
    snprintf(name, sizeof name, "decode M=%d shared out vs double (from GPU h)", M);
    check(name, !es.bad, "max |err|/sum|terms| %.2e over %zu", es.worst, es.n);
    runs[M] = std::move(r);
  }
  // a row's bits do not depend on the other rows: each row alone against the 16-row step
  {
    size_t bad = 0;
    for (int m = 0; m < kMaxRows; ++m) {
      DecodeRun r = run_decode(L, dx + (size_t)m * kH, di + m * kTopK, dw + m * kTopK, 1,
                               std::vector<int>(idx.begin() + m * kTopK, idx.end()), ws, dyr, dys, DecodeCfg());
      for (int n = 0; n < kH; ++n)
        bad += fbits(r.yr[n]) != fbits(runs[16].yr[(size_t)m * kH + n]) || fbits(r.ys[n]) != fbits(runs[16].ys[(size_t)m * kH + n]);
    }
    check("decode batch invariance (16 rows vs each alone)", !bad, "%zu of %d outputs differ", bad, kMaxRows * kH * 2);
  }
  // determinism, and every variant: down variants and up variants of the same K split give the same bits
  {
    size_t bad = 0;
    DecodeRun a = run_decode(L, dx, di, dw, 16, idx, ws, dyr, dys, DecodeCfg());
    for (size_t i = 0; i < a.yr.size(); ++i) bad += fbits(a.yr[i]) != fbits(runs[16].yr[i]) || fbits(a.ys[i]) != fbits(runs[16].ys[i]);
    check("decode deterministic (M=16 twice)", !bad, "%zu differ", bad);
  }
  for (int M : {5, 16})
    for (int uv = 0; uv < decode_up_variants(); ++uv)
      for (int dv = 0; dv < decode_down_variants(); ++dv) {
        if (uv && dv) continue;
        DecodeCfg c;
        c.up = uv, c.down = dv;
        DecodeRun b = run_decode(L, dx, di, dw, M, idx, ws, dyr, dys, c);
        DecodeRun base = run_decode(L, dx, di, dw, M, idx, ws, dyr, dys, DecodeCfg());
        const auto wk = [](const char* n) { return std::string(n).substr(std::string(n).find("wk"), 3); };
        const bool same_k = wk(decode_up_name(uv)) == wk(decode_up_name(0));
        size_t hb = 0, yb = 0, hfar = 0;
        for (size_t i = 0; i < b.h.size(); ++i) hb += bits(b.h[i]) != bits(base.h[i]), hfar += !refs[M].ok(i, bf(b.h[i]));
        for (size_t i = 0; i < b.yr.size(); ++i) yb += fbits(b.yr[i]) != fbits(base.yr[i]) || fbits(b.ys[i]) != fbits(base.ys[i]);
        char name[96];
        snprintf(name, sizeof name, "decode M=%d variant up %s / down %s", M, decode_up_name(uv), decode_down_name(dv));
        if (same_k) check(name, hb == 0 && yb == 0, "bit-identical to the default: h %zu, out %zu differ", hb, yb);
        else check(name, hfar == 0, "another K split: h %zu of %zu differ from the default, %zu outside the range of f32 sums", hb,
                   b.h.size(), hfar);
      }
  cudaFree(ws), cudaFree(dyr), cudaFree(dys), cudaFree(dl), cudaFree(db), cudaFree(di), cudaFree(dw), cudaFree(dx);
  if (!ext) L.free_dev();
}

void test_prefill(TestLayer* ext = nullptr, bool light = false);  // test_prefill.inc

// ------------------------------------------------------------------------------------------------------------------
// S3x2 and the mixed expert on the quantiser's own bytes (engine/weights/s3x2_pack.py dump: real BF16 experts of
// layers 3 and 40 cut to a rank; dump-random: random canonical bits). Host: the Python packer's blocks and
// descriptors equal layout.h's byte for byte; every value of the torch model of the format (levelset3_dequant /
// nvfp4_dequant without s2) equals layout.h's canonical reader and its reader of the Python-packed blocks; the
// loader's checks pass. GPU: the kernels' decode of every value equals the torch model's; then the decode and
// prefill tests on a layer of these experts.
static void real_host(const TestLayer& L, const std::vector<RealMat>& rm) {
  size_t pb = 0, db = 0, vb = 0, rb = 0, sb = 0, nv = 0, nb = 0, nsc = 0, nds = 0;
  int s3 = 0, q4 = 0;
  for (size_t i = 0; i < L.m.size(); ++i) {
    const HostMat& m = L.m[i];
    const RealMat& r = rm[i];
    std::vector<uint8_t> pk(m.packed.size());
    if (m.fmt == kS3) pack_s3(m.w.data(), m.s.data(), m.N, m.K, pk.data()), ++s3;
    else pack_nvfp4(m.w.data(), m.s.data(), m.N, m.K, pk.data()), ++q4;
    for (size_t j = 0; j < pk.size(); ++j) pb += pk[j] != m.packed[j];
    const MatDesc d = m.fmt == kS3 ? make_desc_s3x2(0, m.s2) : make_desc(0, m.s2, kNVFP4);
    db += memcmp(&d, r.desc.data(), sizeof d) != 0;
    MatDesc rd;
    memcpy(&rd, r.desc.data(), sizeof rd);
    float mg[2][4];
    bool pair = false;
    if (m.fmt == kS3) nds += !s3_desc_check(rd, mg, &pair) || !pair;
    size_t nbm = 0;
    sb += scale_bytes_check(m.packed.data(), m.fmt, m.N, m.K, &nbm);
    nb += nbm, nsc += m.fmt == kS3 ? (size_t)m.N * m.K / 16 : 0;
    size_t v1 = 0, v2 = 0;
#pragma omp parallel for reduction(+ : v1, v2)
    for (int n = 0; n < m.N; ++n)
      for (int k = 0; k < m.K; ++k) {
        const float want = bf16_bits_to_f(r.val[(size_t)n * m.K + k]);
        const float c = m.value(n, k);
        const float g = m.fmt == kS3 ? read_s3(m.packed.data(), m.mags, m.K, n, k) : read_q4(m.packed.data(), m.K, n, k);
        v1 += fbits(c) != fbits(want);
        v2 += fbits(g) != fbits(want);
      }
    vb += v1, rb += v2, nv += (size_t)m.N * m.K;
  }
  check("real: python packer = layout.h packer", !pb, "%zu bytes differ (%d S3x2, %d NVFP4 matrices)", pb, s3, q4);
  check("real: python MatDesc = make_desc", !db && !nds, "%zu of %zu descriptors differ; %zu S3 not the S3x2 pair", db, L.m.size(), nds);
  check("real: torch model = canonical reader", !vb, "%zu of %zu values differ (bits, without s2)", vb, nv);
  check("real: torch model = reader of the packed blocks", !rb, "%zu of %zu values differ", rb, nv);
  check("real: scale bytes (loader check)", !sb, "%zu bad; subset B in %.1f %% of S3 groups", sb, nsc ? 100.0 * nb / nsc : 0.0);
}
static void real_deq(const TestLayer& L, const std::vector<RealMat>& rm) {
  size_t bad = 0, n = 0;
  for (size_t i = 0; i < L.m.size(); ++i) {
    const HostMat& m = L.m[i];
    const MatDesc d = m.fmt == kS3 ? make_desc_s3x2(0, m.s2) : make_desc(0, m.s2, kNVFP4);
    uint8_t* dm = h2d(m.packed);
    uint16_t* dout;
    CUDA_CHECK(cudaMalloc(&dout, (size_t)m.N * m.K * 2));
    const int warps = m.N / 8 * (m.K / kBlkK);
    k_deq<<<(warps + 7) / 8, 256>>>(dm, d, m.N, m.K, dout);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto got = d2h(dout, (size_t)m.N * m.K);
    for (size_t j = 0; j < got.size(); ++j) {
      const float v = bf16_bits_to_f(rm[i].val[j]);
      const uint16_t want = m.fmt == kS3 ? rm[i].val[j] : f_to_bf16_bits(v * 0x1p-30f);
      bad += got[j] != want && !(v == 0 && (got[j] & 0x7FFF) == 0);
    }
    n += got.size();
    CUDA_CHECK(cudaFree(dm));
    CUDA_CHECK(cudaFree(dout));
  }
  check("real: GPU decode = torch model", !bad, "%zu of %zu values differ (bf16 bits; NVFP4 x 2^-30)", bad, n);
}
static void test_real(const std::string& dir, bool gpu) {
  const std::string tag = dir.substr(dir.find_last_of('/') + 1);
  g_tag = "[" + tag + "] ";
  TestLayer L;
  std::vector<RealMat> rm;
  load_real(L, rm, dir, 5);
  printf("== %s: %d stored experts (%zu matrices)\n", dir.c_str(), L.NS, L.m.size());
  real_host(L, rm);
  if (gpu) {
    real_deq(L, rm);
    L.upload();
    test_decode(&L);
    test_prefill(&L, true);
    L.free_dev();
  }
  g_tag.clear();
}

int main(int argc, char** argv) {
  std::set<std::string> what;
  std::vector<std::string> real;
  bool cpu = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--cpu")) cpu = true;
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) g_out = fopen(argv[++i], "a");
    else if (!strcmp(argv[i], "--real") && i + 1 < argc) real.push_back(argv[++i]);
    else what.insert(argv[i]);
  }
  if (!real.empty()) {  // --real DIR ...: those layers only (with --cpu, their host checks only)
    if (cpu) cpu_checks();
    for (const auto& d : real) test_real(d, !cpu);
  } else if (cpu) {
    cpu_checks();
  } else {
    auto want = [&](const char* w) { return what.empty() || what.count(w); };
    if (want("cpu")) cpu_checks();
    if (want("deq")) test_deq();
    if (want("route")) test_route();
    if (want("decode")) test_decode();
    if (want("prefill")) test_prefill();
  }
  printf("%s: %d failed\n", g_fail ? "FAILED" : "all ok", g_fail);
  return g_fail ? 1 : 0;
}

#include "experts/tests/test_prefill.inc"
