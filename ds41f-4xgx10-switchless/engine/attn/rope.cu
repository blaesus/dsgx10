// RoPE tables and the standalone RoPE kernel.
#include <cmath>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "common/check.h"

namespace attn {
namespace {

// model.py precompute_freqs_cis, op by op in fp32 on the GPU (torch computes it on the GPU too):
//   freqs = 1 / base**(arange(0,64,2)/64); YaRN: freqs/factor*(1-smooth) + freqs*smooth;
//   angle = fp32(t) * freqs; (cos, sin) = polar(1, angle).
// torch divides a tensor by a Python scalar as a multiply by the scalar's fp32 reciprocal, which
// matters for the ramp: (i - low) * fp32(1/span), not (i - low) / span.
__global__ void rope_table_kernel(float2* table, int max_pos, float base, int yarn, float low, float inv_span) {
  int i = threadIdx.x;  // 0..31
  int t = blockIdx.x * blockDim.y + threadIdx.y;
  if (t >= max_pos) return;
  float e = __fmul_rn((float)(2 * i), 1.0f / 64.0f);
  float f = __fdiv_rn(1.0f, powf(base, e));
  if (yarn) {
    float ramp = fminf(fmaxf(__fmul_rn(__fsub_rn((float)i, low), inv_span), 0.0f), 1.0f);
    float smooth = __fsub_rn(1.0f, ramp);
    float a = __fmul_rn(__fmul_rn(f, 1.0f / 16.0f), __fsub_rn(1.0f, smooth));
    float b = __fmul_rn(f, smooth);
    f = __fadd_rn(a, b);
  }
  float ang = __fmul_rn((float)t, f);
  table[(size_t)t * 32 + i] = make_float2(cosf(ang), sinf(ang));
}

__global__ void rope_rows_kernel(uint16_t* x, int rows, int heads, int dim, const int* pos,
                                 const float2* table, int inverse) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;  // over rows * heads * 32
  if (pair >= rows * heads * 32) return;
  int i = pair & 31, rh = pair >> 5, r = rh / heads;
  uint16_t* p = x + (size_t)rh * dim + (dim - kRope) + 2 * i;
  float2 cs = table[(size_t)pos[r] * 32 + i];
  float x0 = bf2f(p[0]), x1 = bf2f(p[1]);
  rope_pair(x0, x1, cs.x, cs.y, inverse);
  p[0] = f2bf(x0);
  p[1] = f2bf(x1);
}

}  // namespace

void rope_table_build(float2* table, int max_pos, bool compressed, cudaStream_t s) {
  float base = compressed ? 160000.0f : 10000.0f;
  float low = 0.f, inv_span = 1.f;
  if (compressed) {  // corrected_dim() in Python doubles, exactly as model.py
    const double dim = 64, orig = 65536, b = 160000.0;
    auto corrected = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log(b)); };
    double lo = std::max(std::floor(corrected(32)), 0.0);
    double hi = std::min(std::ceil(corrected(1)), dim - 1);
    low = (float)lo;
    inv_span = 1.0f / (float)std::max(hi - lo, 1e-3);
  }
  dim3 block(32, 8);
  rope_table_kernel<<<(max_pos + 7) / 8, block, 0, s>>>(table, max_pos, base, compressed, low, inv_span);
  CUDA_CHECK(cudaGetLastError());
}

void rope_rows(uint16_t* x, int rows, int heads, int dim, const int* pos, const float2* table, bool inverse,
               cudaStream_t s) {
  int n = rows * heads * 32;
  rope_rows_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, rows, heads, dim, pos, table, inverse);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace attn
