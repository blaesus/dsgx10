// Checks the operand layout of the block-scaled FP4 MMA the indexer uses:
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
// against a CPU reference, on random e2m1 operands and ue8m0 scales (one per 32 of K).
// Layout under test (lane = 4g + t, nibble i of a register = element 8t+i, low nibble first):
//   A (16x64): a0 = [g][8t..], a1 = [g+8][8t..], a2 = [g][32+8t..], a3 = [g+8][32+8t..]
//   B (8x64, n-major): b0 = [g][8t..], b1 = [g][32+8t..]
//   scales: a lane's A register holds row g+8(t&1)'s two scales in bytes 0,1; B: column g's
//   C: (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/check.h"

__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sa,
                                        uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}

// A: [16][64] codes, B: [8][64] codes, SA: [16][2], SB: [8][2]; out D [16][8]
__global__ void probe(const uint8_t* A, const uint8_t* B, const uint8_t* SA, const uint8_t* SB, float* D) {
  int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  auto pack = [](const uint8_t* row) {
    uint32_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint32_t)(row[i] & 15) << (4 * i);
    return v;
  };
  uint32_t a[4] = {pack(A + g * 64 + 8 * t), pack(A + (g + 8) * 64 + 8 * t), pack(A + g * 64 + 32 + 8 * t),
                   pack(A + (g + 8) * 64 + 32 + 8 * t)};
  uint32_t b[2] = {pack(B + g * 64 + 8 * t), pack(B + g * 64 + 32 + 8 * t)};
  int ra = g + 8 * (t & 1);
  uint32_t sa = SA[ra * 2] | (uint32_t)SA[ra * 2 + 1] << 8;
  uint32_t sb = SB[g * 2] | (uint32_t)SB[g * 2 + 1] << 8;
  float d[4] = {0, 0, 0, 0};
  mma_fp4(d, a, b, sa, sb);
  D[g * 8 + 2 * t] = d[0];
  D[g * 8 + 2 * t + 1] = d[1];
  D[(g + 8) * 8 + 2 * t] = d[2];
  D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

static double e2m1(uint8_t c) {
  static const double t[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  return (c & 8) ? -t[c & 7] : t[c & 7];
}

int main(int argc, char** argv) {
  int spread = argc > 1 ? atoi(argv[1]) : 3;  // scale exponents in 127 +- spread
  srand(1);
  int bad = 0, trials = 200;
  uint8_t *A, *B, *SA, *SB;
  float* D;
  CUDA_CHECK(cudaMallocManaged(&A, 16 * 64));
  CUDA_CHECK(cudaMallocManaged(&B, 8 * 64));
  CUDA_CHECK(cudaMallocManaged(&SA, 32));
  CUDA_CHECK(cudaMallocManaged(&SB, 16));
  CUDA_CHECK(cudaMallocManaged(&D, 16 * 8 * 4));
  for (int trial = 0; trial < trials; ++trial) {
    for (int i = 0; i < 16 * 64; ++i) A[i] = rand() & 15;
    for (int i = 0; i < 8 * 64; ++i) B[i] = rand() & 15;
    for (int i = 0; i < 32; ++i) SA[i] = 127 - spread + rand() % (2 * spread + 1);
    for (int i = 0; i < 16; ++i) SB[i] = 127 - spread + rand() % (2 * spread + 1);
    probe<<<1, 32>>>(A, B, SA, SB, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int m = 0; m < 16; ++m)
      for (int n = 0; n < 8; ++n) {
        double ref = 0;
        for (int k = 0; k < 64; ++k)
          ref += e2m1(A[m * 64 + k]) * std::ldexp(1.0, SA[m * 2 + k / 32] - 127) * e2m1(B[n * 64 + k]) *
                 std::ldexp(1.0, SB[n * 2 + k / 32] - 127);
        if ((float)ref != D[m * 8 + n]) {
          if (bad < 5) std::printf("trial %d m %d n %d: got %.9g want %.9g\n", trial, m, n, D[m * 8 + n], ref);
          ++bad;
        }
      }
  }
  std::printf("%s: %d mismatches in %d values\n", bad ? "FAILED" : "PASSED", bad, trials * 128);
  return bad ? 1 : 0;
}
