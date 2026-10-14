// Tests of the routed-expert kernels against CPU references on random data in the real formats.
//   test_experts --cpu                 host only: decoders, packers and readers, descriptors, the routing reference
//   test_experts [--fmt i4,b16,q4] [what...]
//                                      GPU (only through ring.sh): deq route decode prefill equiv (default: all);
//                                      decode and prefill on a synthetic layer of each format in --fmt (default
//                                      i4,b16: altar-1's, no fused shared expert; q4 runs with GLM-5.3's fused FP8
//                                      shared expert, the arrangement the kernels keep); equiv: an INT4 layer against
//                                      the BF16 layer of its dequantised values, every output bit for bit
//   test_experts --real DIR [--cpu]    int4_pack.py dump / dump-random: host checks, GPU decode of every value, and the
//                                      decode, prefill and equiv tests on the dump's experts
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
static std::string g_tag;  // prefixed to every check's name ("[i4] ", "[l4r0] ")
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

// INT4 scales for the decode checks: random normal ones, the extremes of the normal range, subnormals (the
// checkpoint has none: its scales are 2^-15 .. 2^-5, docs/findings.md A0), and overflow (15 x the largest)
static std::vector<uint16_t> i4_test_scales() {
  std::vector<uint16_t> v = {0x0001, 0x0002, 0x0017, 0x0040, 0x007F, 0x0080, 0x0081, 0x00FF, 0x0100, 0x3F80, 0x3F81,
                             0x3FFF, 0x7F00, 0x7F7F, 0x3B80, 0x3C00, 0x3D00};
  for (int i = 0; i < 64; ++i) v.push_back(rnd_i4_scale(mix64(i + 5)));
  for (int i = 0; i < 64; ++i) v.push_back((uint16_t)(mix64(i + 99) % 0x7F7F + 1));  // any positive finite bf16
  return v;
}

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
  {  // INT4: i4_dec16's arithmetic (host emulation of the bf16 sub and mul) against the contract, every (u, u_z)
     // with each test scale, and the nibble order of 16 values
    size_t bad = 0, n = 0, inexact = 0;
    for (uint16_t s : i4_test_scales())
      for (int uz = 0; uz < 16; ++uz)
        for (int r = 0; r < 2; ++r) {
          // 16 values: u = (i + r * 7 + uz) % 16 covers every u for each u_z over i
          uint32_t lo = 0, hi = 0;
          int u[16];
          for (int i = 0; i < 16; ++i) {
            u[i] = (i + r * 7 + uz) & 15;
            (i < 8 ? lo : hi) |= (uint32_t)u[i] << (4 * (i % 8));
          }
          uint32_t o[8];
          i4_dec16(lo, hi, i4_zz(uz), bf16x2_bcast(s), o);
          for (int i = 0; i < 16; ++i) {
            const uint16_t got = (o[i / 2] >> 16 * (i & 1)) & 0xFFFF, want = i4_value_bits(u[i], uz, s);
            bad += got != want, ++n;
            inexact += (double)(u[i] - uz) * bf16_bits_to_f(s) != (double)bf16_bits_to_f(want);
          }
        }
    check("cpu int4 dec16 (host bf16 ops) vs contract", !bad, "%zu of %zu mismatches (%zu products inexact in bf16)", bad, n, inexact);
  }
  {  // INT4: the contract's rounding is one rounding of the exact product (not f32 kept, not a rounded difference)
    size_t one = 0, n = 0;
    for (uint16_t s : i4_test_scales())
      for (int u = 0; u < 16; ++u)
        for (int uz = 0; uz < 16; ++uz) {
          const double exact = (double)(u - uz) * bf16_bits_to_f(s);
          const float f = (float)exact;  // exact in f32 (<= 12 significant bits)
          one += ((double)f == exact || std::isinf(f)) && i4_value_bits(u, uz, s) == f_to_bf16_bits(f);  // inf: past f32 (and bf16) range
          ++n;
        }
    check("cpu int4 contract = bf16_rne(exact (u - u_z) s)", one == n, "%zu of %zu", one, n);
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
  {  // the loader's descriptor check (glw_kcheck): each format accepted at its own tag, refused at another; the
     // retired S3 tag and unknown tags refused; misaligned offsets, s2 != 1 on INT4 / BF16, reserved words
    int bad = 0, n = 0;
    for (uint32_t f : {kNVFP4, kINT4, kBF16}) {
      const MatDesc d = make_desc(4096, f == kNVFP4 ? 0.01f : 1.f, f);
      bad += !desc_ok(d, f), ++n;
      for (uint32_t g : {(uint32_t)kNVFP4, (uint32_t)kINT4, (uint32_t)kBF16, 1u, 4u})
        if (g != f) bad += desc_ok(d, g), ++n;
      MatDesc e = d;
      e.off += 128;
      bad += desc_ok(e, f), ++n;
      e = d;
      e.rsv[7] = 1;
      bad += desc_ok(e, f), ++n;
      e = d;
      e.s2 = f == kNVFP4 ? -1.f : 2.f;
      bad += desc_ok(e, f), ++n;
      e = d;
      e.s2 = NAN;
      bad += desc_ok(e, f), ++n;
    }
    bad += mat_bytes(1, kI, kH) != 0 || mat_bytes(7, kI, kH) != 0, ++n;
    check("cpu descriptor check (loader)", !bad, "%d wrong verdicts of %d", bad, n);
  }
  {  // packers against the canonical decode, through the layout readers (every element)
    size_t bad = 0;
    const int N = 24, K = 1024;
    HostMat q = make_q4(N, K, 5, 1.f), a = make_i4(N, K, 6), b = make_b16(N, K, 7), ab = b16_of(a);
    HostFp8 f = make_fp8(128, 256, 7, 1.f);
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        bad += read_q4(q.packed.data(), K, n, k) != q.value(n, k);
        bad += read_i4(a.packed.data(), K, n, k) != a.value_bits(n, k);
        bad += read_b16(b.packed.data(), K, n, k) != b.value_bits(n, k);
        bad += read_b16(ab.packed.data(), K, n, k) != a.value_bits(n, k);
        const I4Parts p = read_i4_parts(a.packed.data(), K, n, k);
        uint16_t sc;
        memcpy(&sc, &a.s[((size_t)n * (K / 32) + k / 32) * 2], 2);
        bad += p.u != q4_nibble(a.w.data(), K, n, k) || p.uz != i4_zp(a.zp.data(), K, n, k / 32) || p.s != sc;
        for (uint32_t fm : {kNVFP4, kINT4, kBF16}) {
          const HostMat& m = fm == kNVFP4 ? q : fm == kINT4 ? a : b;
          bad += fbits(read_value(m.packed.data(), fm, K, n, k)) != fbits(m.value(n, k));
        }
      }
    for (int n = 0; n < 128; ++n)
      for (int k = 0; k < 256; ++k) bad += read_fp8(f.packed.data(), 256, n, k) != e4m3_to_f(f.w[(size_t)n * 256 + k]);
    size_t sbad = scale_bytes_check(q.packed.data(), kNVFP4, N, K) + scale_bytes_check(a.packed.data(), kINT4, N, K);
    HostMat z = make_i4(N, K, 8, 0x8001);  // a negative scale: refused
    sbad += scale_bytes_check(z.packed.data(), kINT4, N, K) != (size_t)N * K / 32;
    check("cpu packers vs canonical (nvfp4, int4, bf16, fp8, int4 -> bf16)", !bad, "%zu mismatches", bad);
    check("cpu scale check (nvfp4, int4; a negative int4 scale refused)", !sbad, "%zu wrong", sbad);
  }
  {  // routing reference at 168: ties go to the lower id, weights sum to 2.5
    std::vector<float> l, bias(kE);
    rnd_logits(l, 64, 3, true);
    for (int e = 0; e < kE; ++e) bias[e] = 28.25f;
    for (int m = 0; m < 64; ++m)  // three experts tied above the rest (|l| < 3), so the tie lies inside the top 8
      for (int j = 0; j < 3; ++j) l[m * kE + (m * 29 + 53 * j) % kE] = 3.5f;
    std::vector<int> idx(64 * kTopK);
    std::vector<float> w(64 * kTopK);
    route_ref(l.data(), bias.data(), 64, idx.data(), w.data());
    int bad = 0, ties = 0;
    for (int m = 0; m < 64; ++m) {
      double s = 0;
      for (int k = 0; k < kTopK; ++k) {
        s += w[m * kTopK + k];
        bad += idx[m * kTopK + k] < 0 || idx[m * kTopK + k] >= kE;
        if (k) {
          const int a = idx[m * kTopK + k - 1], b = idx[m * kTopK + k];
          const float ka = route_sigmoid(l[m * kE + a]) + bias[a], kb = route_sigmoid(l[m * kE + b]) + bias[b];
          bad += !(ka > kb || (ka == kb && a < b));
          ties += ka == kb;
        }
      }
      bad += fabs(s - 2.5) > 1e-5;
    }
    check("cpu routing reference (order, ties, sum)", !bad && ties > 0, "%d bad rows, %d tied neighbours", bad, ties);
  }
}

// ------------------------------------------------------------------------------------------------------------------
// GPU: the decode through the kernels' own fragment mapping, bit for bit
template <class F>
__global__ void k_deq(const uint8_t* m, int N, int K, uint16_t* out) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), B = K / F::kBK;
  if (gw >= N / 8 * B) return;
  const int n8 = gw / B, b = gw % B;
  typename F::Raw r;
  F::load(r, m + (size_t)gw * F::kBlk, lane);
  for (int jj = 0; jj < F::kU; ++jj)
    for (int q = 0; q < 2; ++q) {
      uint32_t o[8];
      F::dec(r, jj, q, o);
      for (int mm = 0; mm < 8; ++mm) {
        const size_t base = (size_t)(8 * n8 + g) * K + F::kBK * b + 128 * jj + 64 * q + 16 * t + 2 * mm;
        out[base] = o[mm] & 0xFFFF;
        out[base + 1] = o[mm] >> 16;
      }
    }
}
// the GPU's decoded bf16 bits of a packed matrix [N][K]
static std::vector<uint16_t> gpu_deq(const HostMat& m) {
  uint8_t* dm = h2d(m.packed);
  uint16_t* dout;
  CUDA_CHECK(cudaMalloc(&dout, (size_t)m.N * m.K * 2));
  const int warps = m.N / 8 * (m.K / blk_k(m.fmt));
  switch (m.fmt) {
    case kNVFP4: k_deq<FmtNVFP4><<<(warps + 7) / 8, 256>>>(dm, m.N, m.K, dout); break;
    case kINT4: k_deq<FmtINT4><<<(warps + 7) / 8, 256>>>(dm, m.N, m.K, dout); break;
    case kBF16: k_deq<FmtBF16><<<(warps + 7) / 8, 256>>>(dm, m.N, m.K, dout); break;
    default: die("gpu_deq: format %u", m.fmt);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto got = d2h(dout, (size_t)m.N * m.K);
  CUDA_CHECK(cudaFree(dm));
  CUDA_CHECK(cudaFree(dout));
  return got;
}
// mismatches of the GPU's decode against the format's exact bf16 value (NVFP4: x 2^-30, its kernels' carry)
static size_t deq_mismatches(const HostMat& m, const std::vector<uint16_t>& got) {
  size_t bad = 0;
  for (int n = 0; n < m.N; ++n)
    for (int k = 0; k < m.K; ++k) {
      const uint16_t gb = got[(size_t)n * m.K + k];
      if (m.fmt == kNVFP4) {
        const float want = m.value(n, k) * 0x1p-30f;
        bad += f_to_bf16_bits(want) != gb && !(want == 0 && (gb & 0x7FFF) == 0);
      } else {
        bad += m.value_bits(n, k) != gb;
      }
    }
  return bad;
}
static void test_deq() {
  for (uint32_t f : {kINT4, kBF16, kNVFP4})
    for (int j = 0; j < 3; j += 2) {  // gate shape and down shape
      const int N = j < 2 ? kI : kH, K = j < 2 ? kH : kI;
      HostMat m = make_mat(f, N, K, 40 + j + 10 * f, 1.f);
      const size_t bad = deq_mismatches(m, gpu_deq(m));
      char name[64];
      snprintf(name, sizeof name, "deq %s [%d x %d]", fmt_name(f), N, K);
      check(name, !bad, "%zu of %zu mismatches (bf16 bits%s)", bad, (size_t)N * K, f == kNVFP4 ? " of value x 2^-30" : " of the contract's value");
    }
  {  // INT4 with every test scale (subnormal, extreme, overflowing): the device's bf16 sub and mul = the contract
    size_t bad = 0, n = 0;
    for (uint16_t s : i4_test_scales()) {
      HostMat m = make_i4(64, 1024, s, s);
      bad += deq_mismatches(m, gpu_deq(m)), n += (size_t)64 * 1024;
    }
    check("deq INT4, 145 scales (subnormal, extreme, overflow)", !bad, "%zu of %zu mismatches", bad, n);
  }
}

static void test_route() {
  for (int ties = 0; ties < 3; ++ties) {
    const int M = 16 + 3 * (ties > 0);  // more rows than one step: the kernel takes any M
    std::vector<float> l, bias(kE);
    rnd_logits(l, M, 11 + ties, ties == 1);
    for (int e = 0; e < kE; ++e) bias[e] = ties ? 28.25f : 28.2f + 0.37f * (float)((mix64(e) % 1000) / 1000.0);
    if (ties == 2)  // ties at the 8th place: three experts share the 8th key, the rest distinct
      for (int m = 0; m < M; ++m) {
        for (int e = 0; e < kE; ++e) l[m * kE + e] = -4.f + 0.01f * (float)((e * 37 + m * 11) % kE);
        const int a = (m * 13) % kE, b = (m * 13 + 50) % kE, c = (m * 13 + 101) % kE, top = (m * 13 + 7) % kE;
        for (int r = 0; r < 7; ++r) l[m * kE + (top + 3 * r) % kE] = 5.f + r;
        l[m * kE + a] = l[m * kE + b] = l[m * kE + c] = 4.5f;
      }
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
    const char* nm = ties == 0 ? "route 168 (random)" : ties == 1 ? "route 168 (many tied logits, equal bias)" : "route 168 (3-way tie at the 8th place)";
    check(nm, !bad, "%d of %d picks differ (ids and weight bits)", bad, M * kTopK);
    cudaFree(dl), cudaFree(db), cudaFree(di), cudaFree(dw);
  }
}

// ------------------------------------------------------------------------------------------------------------------
// decode: CPU reference in double
struct Ref {
  const TestLayer& L;
  std::map<int, std::vector<float>> deq;  // stored matrix index -> values without s2 [N][K]
  std::vector<float> sg, su, sd;          // fused shared, with block scales
  explicit Ref(const TestLayer& l) : L(l) {
    if (!L.fused) return;
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
  const int U = (int)uni.size(), US = U + (R.L.fused ? 1 : 0);
  std::vector<float>& h = out.h;
  h.assign((size_t)US * M * kI, 0.f);
  out.lo.assign(h.size(), 0.f);
  out.hi.assign(h.size(), 0.f);
  for (int u = 0; u < US; ++u) {
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
  std::vector<__nv_bfloat16> h;  // [U (+ 1)][M][kI], the fused shared expert last
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
  const int U = (int)r.uni.size(), US = U + (L.fused ? 1 : 0);
  auto hb = d2h(reinterpret_cast<const __nv_bfloat16*>(ws), (size_t)(kMaxUnion + 1) * kMaxRows * kI);
  r.h.resize((size_t)US * M * kI);
  for (int u = 0; u < US; ++u) {
    const int slot = u == U ? kShSlot : u;
    memcpy(&r.h[(size_t)u * M * kI], &hb[(size_t)slot * M * kI], (size_t)M * kI * 2);
  }
  r.yr = d2h(dyr, (size_t)M * kH);
  r.ys = d2h(dys, (size_t)M * kH);
  return r;
}
static size_t ys_touched(const std::vector<float>& ys) {  // without a fused shared expert ys is left as it was (0xFF)
  size_t n = 0;
  for (float v : ys) n += fbits(v) != 0xFFFFFFFFu;
  return n;
}

// the routing of the decode tests: 16 rows of x, idx and w from route() on random logits
struct Rows {
  std::vector<__nv_bfloat16> x;
  std::vector<int> idx;
  std::vector<float> wts;
  __nv_bfloat16* dx = nullptr;
  int* di = nullptr;
  float* dw = nullptr;
  void make(int n, uint64_t seed) {
    std::vector<float> bias(kE), lg;
    for (int e = 0; e < kE; ++e) bias[e] = 28.2f + 0.37f * (float)((mix64(e + 5) % 1000) / 1000.0);
    rnd_x(x, (size_t)n * kH, seed);
    rnd_logits(lg, n, seed + 1, false);
    float *dl = h2d(lg), *db = h2d(bias);
    dx = h2d(x);
    CUDA_CHECK(cudaMalloc(&di, (size_t)n * kTopK * 4));
    CUDA_CHECK(cudaMalloc(&dw, (size_t)n * kTopK * 4));
    for (int r0 = 0; r0 < n; r0 += 1024) route(dl + (size_t)r0 * kE, db, std::min(1024, n - r0), di + r0 * kTopK, dw + r0 * kTopK, 0, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    idx = d2h(di, (size_t)n * kTopK);
    wts = d2h(dw, (size_t)n * kTopK);
    cudaFree(dl), cudaFree(db);
  }
  void free_dev() { cudaFree(dx), cudaFree(di), cudaFree(dw); }
};

static void test_decode(TestLayer& L) {
  Ref R(L);
  void* ws;
  CUDA_CHECK(cudaMalloc(&ws, decode_ws_bytes()));
  CUDA_CHECK(cudaMemset(ws, 0, decode_ws_bytes()));
  float *dyr, *dys;
  CUDA_CHECK(cudaMalloc(&dyr, (size_t)kMaxRows * kH * 4));
  CUDA_CHECK(cudaMalloc(&dys, (size_t)kMaxRows * kH * 4));
  Rows rw;
  rw.make(kMaxRows, 21);
  const auto& x = rw.x;
  const auto& idx = rw.idx;
  const auto& wts = rw.wts;
  __nv_bfloat16* dx = rw.dx;
  int* di = rw.di;
  float* dw = rw.dw;

  // full check against the CPU at several row counts
  std::map<int, DecodeRun> runs;
  std::map<int, HRef> refs;
  for (int M : {1, 3, 5, 6, 8, 9, 16}) {
    DecodeRun r = run_decode(L, dx, di, dw, M, idx, ws, dyr, dys, DecodeCfg());
    const int U = (int)r.uni.size(), US = U + (L.fused ? 1 : 0);
    HRef& hr = refs[M];
    ref_h(R, x, M, r.uni, hr);
    const std::vector<float>& h = hr.h;
    size_t exact = 0, far = 0;
    for (size_t i = 0; i < h.size(); ++i) {
      const float gv = bf(r.h[i]);
      exact += fbits(gv) == fbits(h[i]) || (gv == 0 && h[i] == 0);
      far += !hr.ok(i, gv);
    }
    char name[96];
    snprintf(name, sizeof name, "decode M=%d h (%d experts%s)", M, U, L.fused ? " + shared" : "");
    check(name, far == 0 && exact >= h.size() * 0.995, "%zu of %zu bit-exact, %zu outside the range of f32 sums", exact, h.size(),
          far);
    // down from the GPU's h, in double
    ErrStat er, es;
    std::vector<double> yr((size_t)M * kH, 0), yrm((size_t)M * kH, 0), ys((size_t)M * kH, 0), ysm((size_t)M * kH, 0);
    for (int u = 0; u < US; ++u) {
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
    for (size_t i = 0; i < yr.size(); ++i) er.add(r.yr[i], yr[i], yrm[i], 1e-4);
    snprintf(name, sizeof name, "decode M=%d routed out vs double (from GPU h)", M);
    check(name, !er.bad, "max |err|/sum|terms| %.2e over %zu", er.worst, er.n);
    if (L.fused) {
      for (size_t i = 0; i < ys.size(); ++i) es.add(r.ys[i], ys[i], ysm[i], 1e-4);
      snprintf(name, sizeof name, "decode M=%d shared out vs double (from GPU h)", M);
      check(name, !es.bad, "max |err|/sum|terms| %.2e over %zu", es.worst, es.n);
    } else {
      const size_t t = ys_touched(r.ys);
      snprintf(name, sizeof name, "decode M=%d ys untouched (no fused shared expert)", M);
      check(name, !t, "%zu of %zu written", t, r.ys.size());
    }
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
  cudaFree(ws), cudaFree(dyr), cudaFree(dys);
  rw.free_dev();
}

void test_prefill(TestLayer& L, bool light);  // test_prefill.inc
void test_equiv(TestLayer& A, bool light);    // test_prefill.inc: INT4 layer A against its dequantised BF16 layer

// ------------------------------------------------------------------------------------------------------------------
// The converter's own bytes (engine/weights/int4_pack.py dump: a rank's slice of real experts; dump-random: random
// canonical bits). Host: the Python packer's blocks and descriptors equal layout.h's byte for byte; the contract's
// values the Python wrote (.val, value_bits) equal layout.h's canonical reader and its reader of the Python-packed
// blocks; the loader's checks pass. GPU: the kernels' decode of every value equals .val; then the decode, prefill and
// equivalence tests on a layer of each format's experts.
static void real_host(const std::vector<HostMat>& hm, const std::vector<RealMat>& rm) {
  size_t pb = 0, db = 0, vb = 0, rb = 0, sb = 0, nv = 0, nds = 0;
  int ni4 = 0, nb16 = 0;
  for (size_t i = 0; i < hm.size(); ++i) {
    const HostMat& m = hm[i];
    const RealMat& r = rm[i];
    HostMat c = m;
    repack(c);
    (m.fmt == kINT4 ? ni4 : nb16)++;
    for (size_t j = 0; j < c.packed.size(); ++j) pb += c.packed[j] != m.packed[j];
    const MatDesc d = make_desc(0, 1.f, m.fmt);
    db += memcmp(&d, r.desc.data(), sizeof d) != 0;
    MatDesc rd;
    memcpy(&rd, r.desc.data(), sizeof rd);
    nds += !desc_ok(rd, m.fmt);
    sb += scale_bytes_check(m.packed.data(), m.fmt, m.N, m.K);
    size_t v1 = 0, v2 = 0;
#pragma omp parallel for reduction(+ : v1, v2)
    for (int n = 0; n < m.N; ++n)
      for (int k = 0; k < m.K; ++k) {
        const uint16_t want = r.val[(size_t)n * m.K + k];
        v1 += m.value_bits(n, k) != want;
        v2 += (m.fmt == kINT4 ? read_i4(m.packed.data(), m.K, n, k) : read_b16(m.packed.data(), m.K, n, k)) != want;
      }
    vb += v1, rb += v2, nv += (size_t)m.N * m.K;
  }
  check("real: python packer = layout.h packer", !pb, "%zu bytes differ (%d INT4, %d BF16 matrices)", pb, ni4, nb16);
  check("real: python MatDesc = make_desc, desc_ok", !db && !nds, "%zu of %zu descriptors differ; %zu refused", db, hm.size(), nds);
  check("real: python contract values = canonical reader", !vb, "%zu of %zu values differ (bf16 bits)", vb, nv);
  check("real: python contract values = reader of the packed blocks", !rb, "%zu of %zu values differ", rb, nv);
  check("real: scales (loader check)", !sb, "%zu bad", sb);
}
static void real_deq(const std::vector<HostMat>& hm, const std::vector<RealMat>& rm) {
  size_t bad = 0, n = 0;
  for (size_t i = 0; i < hm.size(); ++i) {
    const auto got = gpu_deq(hm[i]);
    for (size_t j = 0; j < got.size(); ++j) bad += got[j] != rm[i].val[j];
    n += got.size();
  }
  check("real: GPU decode = the contract's values", !bad, "%zu of %zu values differ (bf16 bits)", bad, n);
}
static void test_real(const std::string& dir, bool gpu) {
  const std::string tag = dir.substr(dir.find_last_of('/') + 1);
  g_tag = "[" + tag + "] ";
  std::vector<HostMat> hm;
  std::vector<RealMat> rm;
  load_real(hm, rm, dir);
  printf("== %s: %zu matrices\n", dir.c_str(), hm.size());
  real_host(hm, rm);
  if (gpu) {
    real_deq(hm, rm);
    for (uint32_t f : {kINT4, kBF16}) {
      TestLayer L;
      real_layer(L, hm, f, 5);
      if (!L.NS) continue;
      g_tag = "[" + tag + " " + fmt_tag(f) + "] ";
      printf("== %s: %d stored %s experts\n", dir.c_str(), L.NS, fmt_name(f));
      L.upload();
      test_decode(L);
      test_prefill(L, true);
      if (f == kINT4) test_equiv(L, true);
      L.free_dev();
    }
  }
  g_tag.clear();
}

int main(int argc, char** argv) {
  std::set<std::string> what;
  std::vector<std::string> real, fmts = {"i4", "b16"};
  bool cpu = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--cpu")) cpu = true;
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) g_out = fopen(argv[++i], "a");
    else if (!strcmp(argv[i], "--real") && i + 1 < argc) real.push_back(argv[++i]);
    else if (!strcmp(argv[i], "--fmt") && i + 1 < argc) {
      fmts.clear();
      for (std::string s = argv[++i]; !s.empty();) {
        const size_t c = s.find(',');
        fmts.push_back(s.substr(0, c));
        s = c == std::string::npos ? "" : s.substr(c + 1);
      }
    } else what.insert(argv[i]);
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
    for (const auto& fs : fmts) {
      if (!want("decode") && !want("prefill")) break;
      const uint32_t f = fmt_of(fs);
      TestLayer L;
      L.build(f, 10, f == kNVFP4 ? 1 : 2 + f, f == kNVFP4);  // NVFP4 with GLM-5.3's fused FP8 shared expert
      L.upload();
      g_tag = "[" + fs + "] ";
      if (want("decode")) test_decode(L);
      if (want("prefill")) test_prefill(L, false);
      L.free_dev();
      g_tag.clear();
    }
    if (want("equiv")) {
      TestLayer A;
      A.build(kINT4, 10, 7, false);
      A.upload();
      g_tag = "[i4 = b16(deq)] ";
      test_equiv(A, false);
      A.free_dev();
      g_tag.clear();
    }
  }
  printf("%s: %d failed\n", g_fail ? "FAILED" : "all ok", g_fail);
  return g_fail ? 1 : 0;
}

#include "experts/tests/test_prefill.inc"
