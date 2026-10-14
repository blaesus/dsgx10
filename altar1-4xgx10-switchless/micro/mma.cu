// glm53 experiment 1, tensor cores on sm_121a (extends ds41f micro/mma.cu):
//   1. kind::mxf4nvf4 scale_vec::4X m16n8k64 e2m1 x e2m1 with ue4m3 scales (NVFP4, group 16): bit-exactness against a
//      CPU reference, the operand / scale-register layout (a hypothesis search if the expected one fails), the rate
//   2. kind::mxf4nvf4 scale_vec::2X ue8m0 (MXFP4, group 32; what ds41f's indexer runs): exactness and rate
//   3. kind::mxf8f6f4 scale_vec::1X m16n8k32 e4m3 x e4m3 and e4m3 x e2m1 (ue8m0): exactness and rate
//   4. m16n8k16 bf16 and f16 (f32 accumulate) and f16 (f16 accumulate): rates
// Usage: mma [--out results.tsv] [--dump raw.bin]     (GPU)
//        mma --analyse raw.bin                         (CPU only: re-classify dumped 4X trials)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstdarg>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

static FILE* g_out = nullptr;
static void result(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  printf("%s\n", buf);
  if (g_out) { fprintf(g_out, "%s\n", buf); fflush(g_out); }
}
static double e2m1_val(int c) { static const double t[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0., -0.5, -1, -1.5, -2, -3, -4, -6}; return t[c & 15]; }
static double e4m3_val(uint8_t b) { __nv_fp8_e4m3 v; memcpy(&v, &b, 1); return (double)float(v); }

// ---- the instructions ----
__device__ __forceinline__ void mma_nvf4(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sa, uint32_t sb) {
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_mxf4(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sa, uint32_t sb) {
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_e4e4(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sa, uint32_t sb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_e4e2(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sa, uint32_t sb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_bf16(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16f32(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16f16(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t) {
  uint32_t* c = reinterpret_cast<uint32_t*>(d);  // two f16x2 accumulators in d[0], d[1]
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- one MMA on raw per-lane registers: A [32][4], B [32][2], SA [32], SB [32], C in [32][4] -> D [32][4] ----
template <int KIND>
__global__ void k_one(const uint32_t* A, const uint32_t* B, const uint32_t* SA, const uint32_t* SB, const float* C, float* D) {
  const int l = threadIdx.x;
  uint32_t a[4] = {A[4 * l], A[4 * l + 1], A[4 * l + 2], A[4 * l + 3]}, b[2] = {B[2 * l], B[2 * l + 1]};
  float d[4] = {C[4 * l], C[4 * l + 1], C[4 * l + 2], C[4 * l + 3]};
  if (KIND == 0) mma_nvf4(d, a, b, SA[l], SB[l]);
  if (KIND == 1) mma_mxf4(d, a, b, SA[l], SB[l]);
  if (KIND == 2) mma_e4e4(d, a, b, SA[l], SB[l]);
  if (KIND == 3) mma_e4e2(d, a, b, SA[l], SB[l]);
  for (int i = 0; i < 4; ++i) D[4 * l + i] = d[i];
}

// ---- throughput: CH independent accumulator chains a warp ----
template <int KIND, int CH>
__global__ void k_thru(float* out, int iters, uint32_t seed) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i) a[i] = (seed * (threadIdx.x + 1) * 2654435761u + i) & 0x77777777u;
  b[0] = a[1] ^ 0x11111111u; b[1] = a[2] ^ 0x22222222u;
  if (KIND >= 4) { for (int i = 0; i < 4; ++i) a[i] = 0x3C003C00u; b[0] = b[1] = 0x3C003C00u; }  // 1.0 in f16 / small in bf16
  const uint32_t sa = 0x38383838u, sb = 0x38383838u;  // ue4m3 1.0 (and a ue8m0 of 2^-71 for the other kinds: fine)
  float d[CH][4] = {};
  for (int it = 0; it < iters; ++it)
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      if (KIND == 0) mma_nvf4(d[c], a, b, sa, sb);
      if (KIND == 1) mma_mxf4(d[c], a, b, 0x7F7F7F7Fu, 0x7F7F7F7Fu);
      if (KIND == 2) mma_e4e4(d[c], a, b, 0x7F7F7F7Fu, 0x7F7F7F7Fu);
      if (KIND == 3) mma_e4e2(d[c], a, b, 0x7F7F7F7Fu, 0x7F7F7F7Fu);
      if (KIND == 4) mma_bf16(d[c], a, b, 0, 0);
      if (KIND == 5) mma_f16f32(d[c], a, b, 0, 0);
      if (KIND == 6) mma_f16f16(d[c], a, b, 0, 0);
    }
  float s = 0;
  for (int c = 0; c < CH; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
  if (s == 1.2345f) out[0] = s;
}
template <int KIND, int CH>
static double rate(double flop_per_mma) {
  float* o;
  CK(cudaMalloc(&o, 4));
  const int blocks = 48 * 8, iters = 2000;
  k_thru<KIND, CH><<<blocks, 128>>>(o, 10, 1);
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  double best = 0;
  for (int r = 0; r < 3; ++r) {
    cudaEventRecord(a);
    k_thru<KIND, CH><<<blocks, 128>>>(o, iters, 1);
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    best = std::max(best, flop_per_mma * iters * CH * blocks * 4 / (ms * 1e9));
  }
  CK(cudaGetLastError());
  cudaFree(o);
  return best;
}

// ---- operand layout model: (lane, register, element) -> (row, k) and the scale bytes ----
struct Layout {
  int nib_hi_first;      // 0: element 0 in bits 3:0 (low nibble first)
  int pa[4], pb[4];      // scale byte j of a lane's register holds the scale of K block pa[j] (A), pb[j] (B)
};
// FP4 m16n8k64: a lane (g, t) holds A rows g (regs 0, 2), g + 8 (1, 3), k = 32 (reg / 2) + 8t + i; B column g, k = 32 reg + 8t + i
static void unpack_fp4(const uint32_t* A, const uint32_t* B, const Layout& L, int Ak[16][64], int Bk[8][64]) {
  for (int l = 0; l < 32; ++l) {
    const int g = l >> 2, t = l & 3;
    for (int r = 0; r < 4; ++r)
      for (int i = 0; i < 8; ++i) {
        const int nib = (A[4 * l + r] >> 4 * (L.nib_hi_first ? 7 - i : i)) & 15;
        Ak[g + 8 * (r & 1)][32 * (r >> 1) + 8 * t + i] = nib;
      }
    for (int r = 0; r < 2; ++r)
      for (int i = 0; i < 8; ++i) Bk[g][32 * r + 8 * t + i] = (B[2 * l + r] >> 4 * (L.nib_hi_first ? 7 - i : i)) & 15;
  }
}
// reference D (16 x 8, f32 rounding of the exact sum + C) of the 4X form under a layout hypothesis
static void ref_nvf4(const uint32_t* A, const uint32_t* B, const uint32_t* SA, const uint32_t* SB, const float* C, const Layout& L,
                     float D[16][8]) {
  int Ak[16][64], Bk[8][64];
  unpack_fp4(A, B, L, Ak, Bk);
  double sa[16][4], sb[8][4];
  for (int l = 0; l < 32; ++l) {
    const int g = l >> 2, t = l & 3;
    for (int j = 0; j < 4; ++j) {
      if (t < 2) sa[g + 8 * t][L.pa[j]] = e4m3_val((SA[l] >> 8 * j) & 0x7F);
      if (t == 0) sb[g][L.pb[j]] = e4m3_val((SB[l] >> 8 * j) & 0x7F);
    }
  }
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n) {
      double s = 0;
      for (int k = 0; k < 64; ++k) s += e2m1_val(Ak[m][k]) * e2m1_val(Bk[n][k]) * sa[m][k / 16] * sb[n][k / 16];
      const int l = 4 * (m % 8) + n / 2, r = 2 * (m / 8) + (n & 1);
      D[m][n] = (float)(s + (double)C[4 * l + r]);
    }
}
static void lane_to_D(const float* Dl, float D[16][8]) {
  for (int l = 0; l < 32; ++l) {
    const int g = l >> 2, t = l & 3;
    D[g][2 * t] = Dl[4 * l]; D[g][2 * t + 1] = Dl[4 * l + 1]; D[g + 8][2 * t] = Dl[4 * l + 2]; D[g + 8][2 * t + 1] = Dl[4 * l + 3];
  }
}

struct Trial { std::vector<uint32_t> A, B, SA, SB; std::vector<float> C, D; };
template <int KIND>
static Trial run_one(std::mt19937& rng, bool withC, int scale_mode) {
  Trial T;
  T.A.resize(128); T.B.resize(64); T.SA.resize(32); T.SB.resize(32); T.C.assign(128, 0.f); T.D.resize(128);
  auto r32 = [&]() { return (uint32_t)rng(); };
  auto fp8w = [&]() { uint32_t v = 0; for (int j = 0; j < 4; ++j) { uint32_t b; do { b = rng() & 0xFF; } while ((b & 0x7F) == 0x7F); v |= b << 8 * j; } return v; };
  for (auto& x : T.A) x = KIND == 2 || KIND == 3 ? fp8w() : r32();
  for (auto& x : T.B) x = KIND == 2 ? fp8w() : r32();
  if (KIND == 3) for (auto& x : T.B) { uint32_t v = 0; for (int j = 0; j < 4; ++j) v |= ((x >> 8 * j) & 15) << (8 * j + 2); x = v; }  // e2m1 in bits 5:2
  // per row / column scales, one register a lane holding its own row's (A: row g + 8 (t & 1)) and column's (B: g)
  std::vector<uint32_t> rowS(16), colS(8);
  auto sbyte = [&]() -> uint32_t {
    if (KIND == 0) { uint32_t b; do { b = 0x20 + rng() % 0x30; } while ((b & 0x7F) == 0x7F); return scale_mode ? b : b; }
    return 127 - 6 + rng() % 13;  // ue8m0 near 1
  };
  for (auto& s : rowS) { s = 0; for (int j = 0; j < 4; ++j) s |= sbyte() << 8 * j; }
  for (auto& s : colS) { s = 0; for (int j = 0; j < 4; ++j) s |= sbyte() << 8 * j; }
  for (int l = 0; l < 32; ++l) { const int g = l >> 2, t = l & 3; T.SA[l] = rowS[g + 8 * (t & 1)]; T.SB[l] = colS[g]; }
  if (withC) for (auto& c : T.C) c = (float)((int)(rng() % 2001) - 1000) / 64.f;
  uint32_t *dA, *dB, *dSA, *dSB;
  float *dC, *dD;
  CK(cudaMalloc(&dA, 512)); CK(cudaMalloc(&dB, 256)); CK(cudaMalloc(&dSA, 128)); CK(cudaMalloc(&dSB, 128)); CK(cudaMalloc(&dC, 512)); CK(cudaMalloc(&dD, 512));
  CK(cudaMemcpy(dA, T.A.data(), 512, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, T.B.data(), 256, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSA, T.SA.data(), 128, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dSB, T.SB.data(), 128, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, T.C.data(), 512, cudaMemcpyHostToDevice));
  k_one<KIND><<<1, 32>>>(dA, dB, dSA, dSB, dC, dD);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(T.D.data(), dD, 512, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dC); cudaFree(dD);
  return T;
}
// classify each D value against the exact sum (in double, exact for these operands) under the expected layout:
// equal to the round-to-nearest f32 of it, equal to its round-toward-zero f32 (they differ), or neither
struct Cls { long exact = 0, rz = 0, other = 0, inexact_sums = 0; };
static void classify(const Trial& T, Cls& c) {
  const Layout L{0, {0, 1, 2, 3}, {0, 1, 2, 3}};
  int Ak[16][64], Bk[8][64];
  unpack_fp4(T.A.data(), T.B.data(), L, Ak, Bk);
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n) {
      const int l = 4 * (m % 8) + n / 2, r = 2 * (m / 8) + (n & 1);
      double s = T.C[4 * l + r];
      for (int k = 0; k < 64; ++k)
        s += e2m1_val(Ak[m][k]) * e2m1_val(Bk[n][k]) * e4m3_val((T.SA[4 * (m % 8) + m / 8] >> 8 * (k / 16)) & 0x7F) *
             e4m3_val((T.SB[4 * n] >> 8 * (k / 16)) & 0x7F);
      const float rn = (float)s;
      float rz = rn;
      if ((double)rn != s) { ++c.inexact_sums; if (fabs((double)rn) > fabs(s)) rz = nextafterf(rn, 0.f); }
      const float g = T.D[4 * l + r];
      if (!memcmp(&g, &rn, 4) || (g == 0 && rn == 0)) ++c.exact;
      else if (!memcmp(&g, &rz, 4)) ++c.rz;
      else ++c.other;
    }
}
static void report_cls(const std::vector<Trial>& tr, const char* where) {
  Cls c;
  for (auto& t : tr) classify(t, c);
  result("MMA\tmxf4nvf4 4X ue4m3 m16n8k64 e2m1xe2m1 (%s)\tlayout: ds41f's 2X operand layout, scale byte j = K block j of 16\t%s\t"
         "%ld values: %ld = round-to-nearest of the exact sum, %ld = round-toward-zero (where they differ), %ld neither; %ld exact sums not representable in f32",
         where, c.other ? "LAYOUT FAIL" : "layout confirmed", c.exact + c.rz + c.other, c.exact, c.rz, c.other, c.inexact_sums);
}
static int count_bad(const Trial& T, const Layout& L) {
  float R[16][8], G[16][8];
  ref_nvf4(T.A.data(), T.B.data(), T.SA.data(), T.SB.data(), T.C.data(), L, R);
  lane_to_D(T.D.data(), G);
  int bad = 0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n) bad += memcmp(&R[m][n], &G[m][n], 4) != 0 && !(R[m][n] == 0 && G[m][n] == 0);
  return bad;
}
// the 2X / mxf8f6f4 references with the ds41f-proven layouts
static int check_other(int kind, const Trial& T, double* maxrel) {
  float G[16][8];
  lane_to_D(T.D.data(), G);
  int bad = 0;
  *maxrel = 0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n) {
      double s = 0;
      const int la = 4 * (m % 8), lb = 4 * n;
      for (int k = 0; k < (kind == 1 ? 64 : 32); ++k) {
        double av, bv, sa, sb;
        const int t = kind == 1 ? (k % 32) / 8 : (k % 16) / 4, half = kind == 1 ? k / 32 : k / 16;
        const int ra = 2 * half + (m / 8), rb = half;
        const uint32_t aw = T.A[4 * (la + t) + ra], bw = T.B[2 * (lb + t) + rb];
        if (kind == 1) { av = e2m1_val((aw >> 4 * (k % 8)) & 15); bv = e2m1_val((bw >> 4 * (k % 8)) & 15); }
        else {
          av = e4m3_val((aw >> 8 * (k % 4)) & 0xFF);
          bv = kind == 2 ? e4m3_val((bw >> 8 * (k % 4)) & 0xFF) : e2m1_val(((bw >> 8 * (k % 4)) >> 2) & 15);
        }
        const uint32_t rs = T.SA[4 * (m % 8) + (m / 8)], cs = T.SB[4 * n];
        sa = ldexp(1.0, (int)((rs >> (kind == 1 ? 8 * (k / 32) : 0)) & 0xFF) - 127);
        sb = ldexp(1.0, (int)((cs >> (kind == 1 ? 8 * (k / 32) : 0)) & 0xFF) - 127);
        s += av * bv * sa * sb;
      }
      const int l = 4 * (m % 8) + n / 2, r = 2 * (m / 8) + (n & 1);
      const float want = (float)(s + (double)T.C[4 * l + r]);
      if (memcmp(&want, &G[m][n], 4) != 0 && !(want == 0 && G[m][n] == 0)) ++bad;
      *maxrel = std::max(*maxrel, fabs(G[m][n] - want) / (fabs(want) + 1e-30));
    }
  return bad;
}

int main(int argc, char** argv) {
  std::string dump, analyse;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--analyse") && i + 1 < argc) { analyse = argv[++i]; continue; }
    if (!strcmp(argv[i], "--out") && i + 1 < argc) g_out = fopen(argv[++i], "a");
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
  }
  if (!analyse.empty()) {  // CPU only: re-read a --dump file of 4X trials
    FILE* f = fopen(analyse.c_str(), "rb");
    if (!f) { perror(analyse.c_str()); return 2; }
    std::vector<Trial> tr;
    for (;;) {
      Trial t;
      t.A.resize(128); t.B.resize(64); t.SA.resize(32); t.SB.resize(32); t.C.resize(128); t.D.resize(128);
      if (fread(t.A.data(), 4, 128, f) != 128) break;
      if (fread(t.B.data(), 4, 64, f) != 64 || fread(t.SA.data(), 4, 32, f) != 32 || fread(t.SB.data(), 4, 32, f) != 32 ||
          fread(t.C.data(), 4, 128, f) != 128 || fread(t.D.data(), 4, 128, f) != 128) break;
      tr.push_back(t);
    }
    fclose(f);
    report_cls(tr, analyse.c_str());
    return 0;
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  int clk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  result("DEVICE\t%s\t%d SMs\tclock %d MHz", prop.name, prop.multiProcessorCount, clk / 1000);
  std::mt19937 rng(12345);

  // ---- 1. NVFP4 4X: bit-exactness under the expected layout, then a search if it fails ----
  std::vector<Trial> tr;
  for (int i = 0; i < 64; ++i) tr.push_back(run_one<0>(rng, i >= 48, 0));
  Layout L0{0, {0, 1, 2, 3}, {0, 1, 2, 3}};
  int bad0 = 0, badC = 0;
  for (int i = 0; i < 64; ++i) (i >= 48 ? badC : bad0) += count_bad(tr[i], L0);
  report_cls(tr, "GPU");
  result("MMA\tmxf4nvf4 4X ue4m3 m16n8k64 e2m1xe2m1\texpected layout (ds41f's 2X operand layout, scale byte j = K block j)\t%s\t%d of %d values differ (C=0), %d of %d (C!=0)",
         bad0 + badC ? "FAIL" : "bit-exact", bad0, 48 * 128, badC, 16 * 128);
  Cls cl;
  for (auto& t : tr) classify(t, cl);
  if (cl.other) {  // a layout error, not rounding: search the alternatives
    std::vector<int> perm = {0, 1, 2, 3};
    int found = 0;
    for (int nib = 0; nib < 2 && !found; ++nib) {
      std::vector<int> pa = perm;
      do {
        std::vector<int> pb = perm;
        do {
          Layout L{nib, {pa[0], pa[1], pa[2], pa[3]}, {pb[0], pb[1], pb[2], pb[3]}};
          int b = 0;
          for (int i = 0; i < 48 && b == 0; ++i) b += count_bad(tr[i], L);
          if (!b) {
            result("MMA\tmxf4nvf4 4X layout search\tfound: nibble %s first, A scale bytes -> blocks %d%d%d%d, B %d%d%d%d\tok", nib ? "high" : "low",
                   pa[0], pa[1], pa[2], pa[3], pb[0], pb[1], pb[2], pb[3]);
            found = 1;
          }
        } while (std::next_permutation(pb.begin(), pb.end()));
      } while (std::next_permutation(pa.begin(), pa.end()));
    }
    if (!found) result("MMA\tmxf4nvf4 4X layout search\tno hypothesis matches; raw trials dumped for offline analysis\tFAIL");
  }
  if (!dump.empty()) {
    FILE* f = fopen(dump.c_str(), "wb");
    for (auto& t : tr) {
      fwrite(t.A.data(), 4, 128, f); fwrite(t.B.data(), 4, 64, f); fwrite(t.SA.data(), 4, 32, f); fwrite(t.SB.data(), 4, 32, f);
      fwrite(t.C.data(), 4, 128, f); fwrite(t.D.data(), 4, 128, f);
    }
    fclose(f);
  }
  // the scale-lane rule: put a different (wrong) scale in the lanes the expected rule says are not read (t = 2, 3 for A;
  // t != 0 for B) and see whether the result moves
  {
    Trial t = tr[0];
    for (int l = 0; l < 32; ++l) { if ((l & 3) >= 2) t.SA[l] = 0x30303030u; if ((l & 3) != 0) t.SB[l] = 0x30303030u; }
    uint32_t *dA, *dB, *dSA, *dSB; float *dC, *dD;
    CK(cudaMalloc(&dA, 512)); CK(cudaMalloc(&dB, 256)); CK(cudaMalloc(&dSA, 128)); CK(cudaMalloc(&dSB, 128)); CK(cudaMalloc(&dC, 512)); CK(cudaMalloc(&dD, 512));
    cudaMemcpy(dA, t.A.data(), 512, cudaMemcpyHostToDevice); cudaMemcpy(dB, t.B.data(), 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dSA, t.SA.data(), 128, cudaMemcpyHostToDevice); cudaMemcpy(dSB, t.SB.data(), 128, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, t.C.data(), 512, cudaMemcpyHostToDevice);
    k_one<0><<<1, 32>>>(dA, dB, dSA, dSB, dC, dD);
    CK(cudaDeviceSynchronize());
    std::vector<float> D2(128);
    cudaMemcpy(D2.data(), dD, 512, cudaMemcpyDeviceToHost);
    const bool same = !memcmp(D2.data(), tr[0].D.data(), 512);
    result("MMA\tmxf4nvf4 4X scale lanes (selectors 0)\tA scales read from lanes t = 0 (rows g) and t = 1 (rows g + 8), B from t = 0: other lanes' registers ignored\t%s",
           same ? "confirmed" : "NOT confirmed (other lanes' scales matter)");
    cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dC); cudaFree(dD);
  }

  // ---- 2./3. the other block-scaled kinds, ds41f's layouts ----
  {
    const char* names[4] = {"", "mxf4nvf4 2X ue8m0 m16n8k64 e2m1xe2m1", "mxf8f6f4 1X ue8m0 m16n8k32 e4m3xe4m3", "mxf8f6f4 1X ue8m0 m16n8k32 e4m3xe2m1"};
    for (int kind = 1; kind <= 3; ++kind) {
      int bad = 0;
      double mr = 0, m1;
      for (int i = 0; i < 32; ++i) {
        Trial t = kind == 1 ? run_one<1>(rng, i >= 24, 0) : kind == 2 ? run_one<2>(rng, i >= 24, 0) : run_one<3>(rng, i >= 24, 0);
        bad += check_other(kind, t, &m1);
        mr = std::max(mr, m1);
      }
      result("MMA\t%s\tds41f's layout, 32 trials (8 with C != 0)\t%s\t%d of %d values differ, max rel %.2e", names[kind], bad ? "FAIL" : "bit-exact", bad, 32 * 128, mr);
    }
  }

  // ---- 4. rates ----
  struct R { const char* name; double tf4, tf8; };
  std::vector<R> rs;
  rs.push_back({"mxf4nvf4 4X ue4m3 m16n8k64 (NVFP4 x NVFP4)", rate<0, 4>(16384), rate<0, 8>(16384)});
  rs.push_back({"mxf4nvf4 2X ue8m0 m16n8k64 (MXFP4 x MXFP4)", rate<1, 4>(16384), rate<1, 8>(16384)});
  rs.push_back({"mxf8f6f4 1X m16n8k32 e4m3 x e4m3 (FP8 x FP8)", rate<2, 4>(8192), rate<2, 8>(8192)});
  rs.push_back({"mxf8f6f4 1X m16n8k32 e4m3 x e2m1", rate<3, 4>(8192), rate<3, 8>(8192)});
  rs.push_back({"m16n8k16 bf16, f32 accumulate", rate<4, 4>(4096), rate<4, 8>(4096)});
  rs.push_back({"m16n8k16 f16, f32 accumulate", rate<5, 4>(4096), rate<5, 8>(4096)});
  rs.push_back({"m16n8k16 f16, f16 accumulate", rate<6, 4>(4096), rate<6, 8>(4096)});
  for (auto& r : rs) result("RATE\t%s\t4 chains %.1f TFLOPS\t8 chains %.1f TFLOPS", r.name, r.tf4, r.tf8);
  result("DONE\tmma");
  if (g_out) fclose(g_out);
  return 0;
}
