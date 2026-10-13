// Absorbed sparse MLA attention: MQA over the 576-wide cache entries for this rank's 16 heads. A row's keys
// are its 2048 selected positions (p >= 2048) or positions 0..p, in 32 blocks of 64; a CTA takes per_split
// blocks of one row: it gathers each block's entries into shared memory with cp.async, computes S = Q K^T
// (m16n8k16, the 16 heads as M, 36 k-steps over q_lat ++ q_rope), the online softmax in f32 (P rounded to bf16),
// and O += P V over the 512 latent dims (each warp owns 128). With splits > 1 the CTAs' (m, l, O) are merged by
// a second kernel in split order. Structure from the DeepSeek engine's sparse_attn (no window, no sink).
#include <cmath>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"

namespace glm {
namespace {

constexpr int kBlk = 64;          // keys a block
constexpr int kBlocks = kTopK / kBlk;  // 32
constexpr int kRowS = kLat + 8;   // padded smem row (bf16): 16 B shift a row, no ldmatrix conflicts
constexpr int kPS = kBlk + 8;
constexpr size_t kAttnSmem = (size_t)(kHeads + kBlk) * kRowS * 2 + kHeads * kPS * 2 + 2 * 4 * kHeads * 4 + kBlk * 4;

struct AttnArgs {
  const uint16_t* q_lat;
  const uint16_t* q;
  const int* pos;
  const int* slot;
  const uint32_t* rope;
  KvCache kv;
  const int* topk;
  int per_split, splits;
  uint16_t* out;
  float* po;
  float* pml;
};

__global__ void __launch_bounds__(128) k_attn(AttnArgs a) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  uint16_t* qs = reinterpret_cast<uint16_t*>(smem_raw);       // [16][kRowS]
  uint16_t* ks = qs + kHeads * kRowS;                         // [64][kRowS]
  uint16_t* ps = ks + kBlk * kRowS;                           // [16][kPS]
  float* red = reinterpret_cast<float*>(ps + kHeads * kPS);   // [2][4][16]
  int* kpos = reinterpret_cast<int*>(red + 2 * 4 * kHeads);   // [64]
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y, split = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int p = a.pos[r];
  const bool dense = p < kTopK;
  const uint16_t* cache = a.kv.kv + (a.slot ? (size_t)a.slot[r] * a.kv.slot_stride : 0) * kLat;
  // q: the absorbed latent query, then the RoPE'd rope part (dims 192..255 of q_b's head)
  const uint16_t* ql = a.q_lat + (size_t)r * kHeads * kKvLora;
  for (int i = threadIdx.x; i < kHeads * kKvLora / 8; i += 128) {
    const int h = i / (kKvLora / 8), c = (i % (kKvLora / 8)) * 8;
    *reinterpret_cast<uint4*>(qs + h * kRowS + c) = *reinterpret_cast<const uint4*>(ql + h * kKvLora + c);
  }
  const uint32_t* cs = a.rope + (size_t)p * (kRope / 2);
  for (int i = threadIdx.x; i < kHeads * 32; i += 128) {
    const int h = i / 32, pi = i % 32;
    const uint32_t v = *reinterpret_cast<const uint32_t*>(a.q + ((size_t)r * kHeads + h) * kQkDim + kNope + 2 * pi);
    float lo, hi;
    rope_pair(bf2f(v), bf2f_hi(v), cs[pi], lo, hi);
    qs[h * kRowS + kKvLora + pi] = f2bf(lo);
    qs[h * kRowS + kKvLora + 32 + pi] = f2bf(hi);
  }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;  // heads g and g+8
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  const int b0 = split * a.per_split, b1 = min(kBlocks, b0 + a.per_split);
  for (int blk = b0; blk < b1; ++blk) {
    if (dense && blk * kBlk > p) break;  // uniform in the CTA
    __syncthreads();  // the previous block's ks / ps are consumed, qs is ready
    if (threadIdx.x < kBlk) {
      const int i = blk * kBlk + threadIdx.x;
      kpos[threadIdx.x] = dense ? (i <= p ? i : -1) : a.topk[(size_t)r * kTopK + i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kBlk * (kLat / 8); i += 128) {
      const int key = i / (kLat / 8), part = i % (kLat / 8);
      const int kp = kpos[key];
      uint16_t* dst = ks + key * kRowS + part * 8;
      if (kp >= 0)
        cp_async16(dst, cache + (size_t)kp * kLat + part * 8);
      else
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
    cp_async_wait_all();
    __syncthreads();
    // S = q . k for this warp's 16 keys (two n-tiles), all 16 heads
    float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 4
    for (int kk = 0; kk < kLat / 16; ++kk) {
      uint32_t af[4], bf[4];
      ldsm_x4(af, qs + (lane & 15) * kRowS + kk * 16 + (lane >> 4) * 8);
      ldsm_x4(bf, ks + (warp * 16 + (lane & 7) + ((lane >> 4) << 3)) * kRowS + kk * 16 + ((lane >> 3) & 1) * 8);
      mma_bf16(s[0], af[0], af[1], af[2], af[3], bf[0], bf[1]);
      mma_bf16(s[1], af[0], af[1], af[2], af[3], bf[2], bf[3]);
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int key = warp * 16 + nt * 8 + 2 * t + (j & 1);
        s[nt][j] = kpos[key] >= 0 ? __fmul_rn(s[nt][j], kSoftmaxScale) : -INFINITY;
        if (j < 2) mx0 = fmaxf(mx0, s[nt][j]);
        else mx1 = fmaxf(mx1, s[nt][j]);
      }
#pragma unroll
    for (int o2 = 1; o2 < 4; o2 <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o2));
    }
    if (t == 0) {
      red[warp * kHeads + g] = mx0;
      red[warp * kHeads + g + 8] = mx1;
    }
    __syncthreads();
    float mn0 = m0, mn1 = m1;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      mn0 = fmaxf(mn0, red[w * kHeads + g]);
      mn1 = fmaxf(mn1, red[w * kHeads + g + 8]);
    }
    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const float p0 = expf(__fsub_rn(s[nt][0], mn0)), p1 = expf(__fsub_rn(s[nt][1], mn0));
      const float p2 = expf(__fsub_rn(s[nt][2], mn1)), p3 = expf(__fsub_rn(s[nt][3], mn1));
      sum0 = __fadd_rn(__fadd_rn(sum0, p0), p1);
      sum1 = __fadd_rn(__fadd_rn(sum1, p2), p3);
      const int col = warp * 16 + nt * 8 + 2 * t;
      *reinterpret_cast<uint32_t*>(ps + g * kPS + col) = pack_bf16(p0, p1);
      *reinterpret_cast<uint32_t*>(ps + (g + 8) * kPS + col) = pack_bf16(p2, p3);
    }
#pragma unroll
    for (int o2 = 1; o2 < 4; o2 <<= 1) {
      sum0 = __fadd_rn(sum0, __shfl_xor_sync(0xffffffffu, sum0, o2));
      sum1 = __fadd_rn(sum1, __shfl_xor_sync(0xffffffffu, sum1, o2));
    }
    if (t == 0) {
      red[4 * kHeads + warp * kHeads + g] = sum0;
      red[4 * kHeads + warp * kHeads + g + 8] = sum1;
    }
    __syncthreads();
    const float al0 = expf(__fsub_rn(m0, mn0)), al1 = expf(__fsub_rn(m1, mn1));
    float bs0 = 0.f, bs1 = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      bs0 = __fadd_rn(bs0, red[4 * kHeads + w * kHeads + g]);
      bs1 = __fadd_rn(bs1, red[4 * kHeads + w * kHeads + g + 8]);
    }
    l0 = __fadd_rn(__fmul_rn(l0, al0), bs0);
    l1 = __fadd_rn(__fmul_rn(l1, al1), bs1);
    m0 = mn0;
    m1 = mn1;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      o[i][0] = __fmul_rn(o[i][0], al0);
      o[i][1] = __fmul_rn(o[i][1], al0);
      o[i][2] = __fmul_rn(o[i][2], al1);
      o[i][3] = __fmul_rn(o[i][3], al1);
    }
    // O += P V over the block's 64 keys, this warp's 128 latent dims
#pragma unroll
    for (int kk = 0; kk < kBlk / 16; ++kk) {
      uint32_t af[4];
      ldsm_x4(af, ps + (lane & 15) * kPS + kk * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, ks + (kk * 16 + (lane & 15)) * kRowS + warp * 128 + np * 16 + (lane >> 4) * 8);
        mma_bf16(o[2 * np], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        mma_bf16(o[2 * np + 1], af[0], af[1], af[2], af[3], bf[2], bf[3]);
      }
    }
  }
  if (a.splits == 1) {
    uint16_t* orow = a.out + (size_t)r * kHeads * kKvLora;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int d = warp * 128 + i * 8 + 2 * t;
      *reinterpret_cast<uint32_t*>(orow + g * kKvLora + d) = pack_bf16(__fdiv_rn(o[i][0], l0), __fdiv_rn(o[i][1], l0));
      *reinterpret_cast<uint32_t*>(orow + (g + 8) * kKvLora + d) =
          pack_bf16(__fdiv_rn(o[i][2], l1), __fdiv_rn(o[i][3], l1));
    }
  } else {
    const size_t base = ((size_t)r * a.splits + split) * kHeads;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int d = warp * 128 + i * 8 + 2 * t;
      *reinterpret_cast<float2*>(a.po + (base + g) * kKvLora + d) = make_float2(o[i][0], o[i][1]);
      *reinterpret_cast<float2*>(a.po + (base + g + 8) * kKvLora + d) = make_float2(o[i][2], o[i][3]);
    }
    if (warp == 0 && t == 0) {
      *reinterpret_cast<float2*>(a.pml + (base + g) * 2) = make_float2(m0, l0);
      *reinterpret_cast<float2*>(a.pml + (base + g + 8) * 2) = make_float2(m1, l1);
    }
  }
}

// one CTA per (row, head): the splits' (m, l, O) merged in split order, normalised, rounded to bf16
__global__ void __launch_bounds__(128) k_attn_combine(const float* po, const float* pml, int splits, uint16_t* out) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x * 4;
  const size_t base = (size_t)r * splits * kHeads + h;
  float m = -1e30f;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, pml[(base + (size_t)s * kHeads) * 2]);
  float l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < splits; ++s) {
    const float2 ml = *reinterpret_cast<const float2*>(pml + (base + (size_t)s * kHeads) * 2);
    if (ml.y == 0.f) continue;  // a split with no keys
    const float f = expf(__fsub_rn(ml.x, m));
    l = __fadd_rn(l, __fmul_rn(ml.y, f));
    const float4 v = *reinterpret_cast<const float4*>(po + (base + (size_t)s * kHeads) * kKvLora + d);
    acc[0] = __fadd_rn(acc[0], __fmul_rn(v.x, f));
    acc[1] = __fadd_rn(acc[1], __fmul_rn(v.y, f));
    acc[2] = __fadd_rn(acc[2], __fmul_rn(v.z, f));
    acc[3] = __fadd_rn(acc[3], __fmul_rn(v.w, f));
  }
  *reinterpret_cast<uint2*>(out + ((size_t)r * kHeads + h) * kKvLora + d) =
      make_uint2(pack_bf16(__fdiv_rn(acc[0], l), __fdiv_rn(acc[1], l)), pack_bf16(__fdiv_rn(acc[2], l), __fdiv_rn(acc[3], l)));
}

}  // namespace

void sparse_attn(const uint16_t* q_lat, const uint16_t* q, const int* pos, const int* slot, const uint32_t* rope,
                 KvCache kv, const int* topk, int rows, int splits, AttnScratch sc, uint16_t* out, cudaStream_t s,
                 bool pdl) {
  static bool once = [] {
    CUDA_CHECK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kAttnSmem));
    return true;
  }();
  (void)once;
  if (splits < 1 || kBlocks % splits) die("sparse_attn: splits %d (a divisor of 32)", splits);
  if (splits > 1 && splits > sc.max_splits) die("sparse_attn: %d splits, scratch for %d", splits, sc.max_splits);
  AttnArgs a{q_lat, q, pos, slot, rope, kv, topk, kBlocks / splits, splits, out, sc.o, sc.ml};
  launch(k_attn, dim3(splits, rows), dim3(128), kAttnSmem, s, pdl, a);
  CUDA_CHECK(cudaGetLastError());
  if (splits > 1) {
    launch(k_attn_combine, dim3(rows, kHeads), dim3(kKvLora / 4), 0, s, pdl, (const float*)sc.o, (const float*)sc.ml,
           splits, out);
    CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace glm
