// Sparse attention over a row's window and its selected compressed entries (MQA: one 512-wide latent
// per key serves all 16 local heads as both K and V). Online softmax in blocks of 64 keys as the
// reference's tilelang kernel: running max from -1e30, fp32 P summed, P rounded to bf16 for PV, the
// sink added to the denominator at the end. Keys are dequantised to bf16 in shared memory (exactly:
// e4m3 x 2^k and e2m1 x e4m3 both fit bf16) and multiplied with bf16 mma.sync m16n8k16.
#include <cmath>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"
#include "common/check.h"

namespace attn {
namespace {

constexpr int kBlk = 64;          // keys per block
constexpr int kRowS = kHead + 8;  // padded smem row (bf16 elements): 16 B shift per row, no ldmatrix conflicts
constexpr int kPS = kBlk + 8;
constexpr int kWinBlocks = kWindow / kBlk;
constexpr int kWinBlocksMax = 3;     // with explicit bounds: up to 192 window keys (DSpark: 128 main + the 5 drafts)
constexpr size_t kAttnSmem = (size_t)(kHeads + kBlk) * kRowS * 2 + kHeads * kPS * 2 + 2 * 4 * kHeads * 4 + kBlk * 4;

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { return (uint32_t)f2bf(lo) | (uint32_t)f2bf(hi) << 16; }

struct AttnArgs {
  const uint16_t* q;
  const int* pos;
  WinBounds wb;
  const float2* table;
  Window win;
  Compressed ckv;
  const int* topk;
  int kmax;
  const float* sink;
  float scale;
  int nwin, nblocks, per_split, splits;
  uint16_t* out;
  float* po;
  float* pml;
};

// o (bf16) of heads h, dims (d, d+1), with the inverse RoPE on dims 448..511 applied to the bf16 values
__device__ __forceinline__ uint32_t finish_pair(float x0, float x1, bool rope, float2 cs) {
  x0 = bfround(x0);
  x1 = bfround(x1);
  if (rope) rope_pair(x0, x1, cs.x, cs.y, true);
  return pack_bf16(x0, x1);
}
__device__ __forceinline__ uint32_t finish_pair(float x0, float x1, int d, const float2* tb) {
  bool rope = d >= kHead - kRope;
  return finish_pair(x0, x1, rope, rope ? tb[(d - (kHead - kRope)) >> 1] : make_float2(0.f, 0.f));
}

// grid (splits, rows), 4 warps. Block b of a row: b < nwin are window positions wlo + 64b + i (up to whi),
// the rest compressed entries topk[r][64(b-nwin) + i]. Warp w scores keys 16w..16w+15 and owns output dims
// 128w..128w+127.
__global__ void __launch_bounds__(128) attn_kernel(AttnArgs a) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  __shared__ float e2m1_lut[16];  // e2m1_to_f by table: the compressed blocks' decode is ALU-bound otherwise
  pdl_trigger();
  if (threadIdx.x < 16) e2m1_lut[threadIdx.x] = e2m1_to_f(threadIdx.x);
  pdl_wait();
  uint16_t* qs = reinterpret_cast<uint16_t*>(smem_raw);  // [16][kRowS]
  uint16_t* ks = qs + kHeads * kRowS;                    // [64][kRowS]
  uint16_t* ps = ks + kBlk * kRowS;                      // [16][kPS]
  float* red = reinterpret_cast<float*>(ps + kHeads * kPS);  // [2][4][16]
  int* kvalid = reinterpret_cast<int*>(red + 2 * 4 * kHeads);
  int r = blockIdx.y, split = blockIdx.x;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  int p = a.pos[r];
  const Window win = win_at(a.win, r);  // the row's session's
  const Compressed ckv = ckv_at(a.ckv, r);
  int whi = a.wb.hi ? a.wb.hi[r] : p, wlo = max(0, a.wb.lo ? a.wb.lo[r] : whi - (kWindow - 1));
  const float2* tb = a.table + (size_t)p * 32;
  const uint16_t* qrow = a.q + (size_t)r * kHeads * kHead;
#pragma unroll
  for (int it = 0; it < kHeads * kHead / 8 / 128; ++it) {  // every load in flight at once
    int i = threadIdx.x + 128 * it;
    int h = i >> 6, c = (i & 63) * 8;
    uint4 v = *reinterpret_cast<const uint4*>(qrow + h * kHead + c);
    if (c >= kHead - kRope) {
      uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float x0 = __uint_as_float(w[j] << 16), x1 = __uint_as_float(w[j] & 0xffff0000u);
        float2 cs = tb[((c - (kHead - kRope)) >> 1) + j];
        rope_pair(x0, x1, cs.x, cs.y, false);
        w[j] = pack_bf16(x0, x1);
      }
      v = make_uint4(w[0], w[1], w[2], w[3]);
    }
    *reinterpret_cast<uint4*>(qs + h * kRowS + c) = v;
  }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;  // heads g and g+8 (every warp keeps the same copy)
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  int b0 = split * a.per_split, b1 = min(a.nblocks, b0 + a.per_split);
  for (int blk = b0; blk < b1; ++blk) {
    if (blk < a.nwin && wlo + blk * kBlk > whi) continue;  // past this row's window (uniform in the CTA)
    __syncthreads();  // the previous block's ks/ps are consumed (and qs is ready)
    // This warp's 16 keys: the positions first (lane k < 16 holds key k's, -1 = none), then every key's
    // codes and scale in flight at once (decode is latency-bound), then dequantised into ks.
    int my_kp = -1;
    if (lane < 16) {
      int key = warp * 16 + lane;
      if (blk < a.nwin) {
        int kp = wlo + blk * kBlk + key;
        my_kp = kp <= whi ? kp : -1;
      } else {
        my_kp = a.topk[(size_t)r * a.kmax + (blk - a.nwin) * kBlk + key];
      }
      kvalid[key] = my_kp >= 0;
    }
    if (blk < a.nwin) {  // e4m3 x ue8m0 per 32: lane holds elements 16 lane .. +15
      uint4 c[16];
      uint32_t e[16];
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        int kp = __shfl_sync(0xffffffffu, my_kp, kk), slot = (kp - win.base) & win.mask;
        c[kk] = kp >= 0 ? *reinterpret_cast<const uint4*>(win.kv + (size_t)slot * kHead + lane * 16)
                        : make_uint4(0, 0, 0, 0);
        e[kk] = kp >= 0 ? win.sc[slot * 16 + lane / 2] : 127u;
      }
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        float s = ue8m0_to_f(e[kk]);
        const uint32_t cw[4] = {c[kk].x, c[kk].y, c[kk].z, c[kk].w};
        uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          uint32_t b = cw[j >> 1] >> (16 * (j & 1));
          w[j] = pack_bf16(__fmul_rn(e4m3_to_f(b & 0xff), s), __fmul_rn(e4m3_to_f((b >> 8) & 0xff), s));
        }
        uint4* dst = reinterpret_cast<uint4*>(ks + (warp * 16 + kk) * kRowS + lane * 16);
        dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
      }
    } else {  // e2m1 x e4m3 per 16: lane holds elements 16 lane .. +15
      uint2 c[16];
      uint32_t e[16];
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        int kp = __shfl_sync(0xffffffffu, my_kp, kk);
        c[kk] = kp >= 0 ? *reinterpret_cast<const uint2*>(ckv.kv + (size_t)kp * (kHead / 2) + lane * 8)
                        : make_uint2(0, 0);
        e[kk] = kp >= 0 ? ckv.sc[(size_t)kp * 32 + lane] : 0u;
      }
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        float s = e4m3_to_f(e[kk]);
        const uint32_t cw[2] = {c[kk].x, c[kk].y};
        uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          uint32_t b = cw[j >> 2] >> (8 * (j & 3));
          w[j] = pack_bf16(__fmul_rn(e2m1_lut[b & 15], s), __fmul_rn(e2m1_lut[(b >> 4) & 15], s));
        }
        uint4* dst = reinterpret_cast<uint4*>(ks + (warp * 16 + kk) * kRowS + lane * 16);
        dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
      }
    }
    __syncthreads();
    // S = q . k for this warp's 16 keys (two n-tiles), all 16 heads
    float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 8
    for (int kk = 0; kk < kHead / 16; ++kk) {
      uint32_t af[4], bf[4];
      ldsm_x4(af, qs + (lane & 15) * kRowS + kk * 16 + (lane >> 4) * 8);
      ldsm_x4(bf, ks + (warp * 16 + (lane & 7) + ((lane >> 4) << 3)) * kRowS + kk * 16 + ((lane >> 3) & 1) * 8);
      mma_bf16(s[0], af, bf[0], bf[1]);
      mma_bf16(s[1], af, bf[2], bf[3]);
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int key = warp * 16 + nt * 8 + 2 * t + (j & 1);
        s[nt][j] = kvalid[key] ? __fmul_rn(s[nt][j], a.scale) : -INFINITY;
        if (j < 2)
          mx0 = fmaxf(mx0, s[nt][j]);
        else
          mx1 = fmaxf(mx1, s[nt][j]);
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
      float p0 = expf(__fsub_rn(s[nt][0], mn0)), p1 = expf(__fsub_rn(s[nt][1], mn0));
      float p2 = expf(__fsub_rn(s[nt][2], mn1)), p3 = expf(__fsub_rn(s[nt][3], mn1));
      sum0 = __fadd_rn(__fadd_rn(sum0, p0), p1);
      sum1 = __fadd_rn(__fadd_rn(sum1, p2), p3);
      int col = warp * 16 + nt * 8 + 2 * t;
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
    float al0 = expf(__fsub_rn(m0, mn0)), al1 = expf(__fsub_rn(m1, mn1));
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
    // O += P V over the block's 64 keys, this warp's 128 dims
#pragma unroll
    for (int kk = 0; kk < kBlk / 16; ++kk) {
      uint32_t af[4];
      ldsm_x4(af, ps + (lane & 15) * kPS + kk * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, ks + (kk * 16 + (lane & 15)) * kRowS + warp * 128 + np * 16 + (lane >> 4) * 8);
        mma_bf16(o[2 * np], af, bf[0], bf[1]);
        mma_bf16(o[2 * np + 1], af, bf[2], bf[3]);
      }
    }
  }
  if (a.splits == 1) {
    float d0 = __fadd_rn(l0, expf(__fsub_rn(a.sink[g], m0))), d1 = __fadd_rn(l1, expf(__fsub_rn(a.sink[g + 8], m1)));
    uint16_t* orow = a.out + (size_t)r * kHeads * kHead;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      int d = warp * 128 + i * 8 + 2 * t;
      *reinterpret_cast<uint32_t*>(orow + g * kHead + d) =
          finish_pair(__fdiv_rn(o[i][0], d0), __fdiv_rn(o[i][1], d0), d, tb);
      *reinterpret_cast<uint32_t*>(orow + (g + 8) * kHead + d) =
          finish_pair(__fdiv_rn(o[i][2], d1), __fdiv_rn(o[i][3], d1), d, tb);
    }
  } else {
    size_t base = ((size_t)r * a.splits + split) * kHeads;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      int d = warp * 128 + i * 8 + 2 * t;
      *reinterpret_cast<float2*>(a.po + (base + g) * kHead + d) = make_float2(o[i][0], o[i][1]);
      *reinterpret_cast<float2*>(a.po + (base + g + 8) * kHead + d) = make_float2(o[i][2], o[i][3]);
    }
    if (warp == 0 && t == 0) {
      *reinterpret_cast<float2*>(a.pml + (base + g) * 2) = make_float2(m0, l0);
      *reinterpret_cast<float2*>(a.pml + (base + g + 8) * 2) = make_float2(m1, l1);
    }
  }
}

// one CTA per (row, head): merge the splits' (m, l, O), add the sink, normalise, inverse RoPE (the sink loads
// before the PDL wait)
__global__ void __launch_bounds__(128) attn_combine_kernel(const float* po, const float* pml, const float* sink,
                                                           const int* pos, const float2* table, int splits,
                                                           uint16_t* out) {
  int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x * 4;
  pdl_trigger();
  float sk = sink[h];
  pdl_wait();
  const float2* tb = table + (size_t)pos[r] * 32;
  bool rope = d >= kHead - kRope;
  float2 cs0 = make_float2(0.f, 0.f), cs1 = cs0;
  if (rope) {
    cs0 = tb[(d - (kHead - kRope)) >> 1];
    cs1 = tb[(d + 2 - (kHead - kRope)) >> 1];
  }
  size_t base = (size_t)r * splits * kHeads + h;
  float m = -1e30f;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, pml[(base + (size_t)s * kHeads) * 2]);
  float l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < splits; ++s) {
    float2 ml = *reinterpret_cast<const float2*>(pml + (base + (size_t)s * kHeads) * 2);
    float f = expf(__fsub_rn(ml.x, m));
    l = __fadd_rn(l, __fmul_rn(ml.y, f));
    float4 v = *reinterpret_cast<const float4*>(po + (base + (size_t)s * kHeads) * kHead + d);
    acc[0] = __fadd_rn(acc[0], __fmul_rn(v.x, f));
    acc[1] = __fadd_rn(acc[1], __fmul_rn(v.y, f));
    acc[2] = __fadd_rn(acc[2], __fmul_rn(v.z, f));
    acc[3] = __fadd_rn(acc[3], __fmul_rn(v.w, f));
  }
  l = __fadd_rn(l, expf(__fsub_rn(sk, m)));
  uint32_t w0 = finish_pair(__fdiv_rn(acc[0], l), __fdiv_rn(acc[1], l), rope, cs0);
  uint32_t w1 = finish_pair(__fdiv_rn(acc[2], l), __fdiv_rn(acc[3], l), rope, cs1);
  *reinterpret_cast<uint2*>(out + ((size_t)r * kHeads + h) * kHead + d) = make_uint2(w0, w1);
}

}  // namespace

void sparse_attn(const uint16_t* q_pre, const int* pos, WinBounds wb, int rows, const float2* table, Window win,
                 Compressed ckv, const int* topk, int kmax, const float* sink, uint16_t* out, int splits,
                 AttnScratch scratch, cudaStream_t s, bool pdl) {
  static bool once = [] {
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kAttnSmem));
    return true;
  }();
  (void)once;
  if (kmax % kBlk) die("sparse_attn: kmax %d", kmax);
  check_slots(win, "sparse_attn");
  check_slots(ckv, "sparse_attn");
  AttnArgs a{};
  a.q = q_pre;
  a.pos = pos;
  a.wb = wb;
  a.nwin = wb.lo || wb.hi ? kWinBlocksMax : kWinBlocks;
  a.table = table;
  a.win = win;
  a.ckv = ckv;
  a.topk = topk;
  a.kmax = kmax;
  a.sink = sink;
  a.scale = (float)std::pow(512.0, -0.5);
  a.nblocks = a.nwin + kmax / kBlk;
  splits = std::max(1, std::min(splits, a.nblocks));
  a.per_split = (a.nblocks + splits - 1) / splits;
  a.splits = (a.nblocks + a.per_split - 1) / a.per_split;
  if (a.splits > 1 && a.splits > scratch.max_splits) die("sparse_attn: %d splits, scratch for %d", a.splits, scratch.max_splits);
  a.out = out;
  a.po = scratch.o;
  a.pml = scratch.ml;
  launch(attn_kernel, dim3(a.splits, rows), dim3(128), kAttnSmem, s, pdl, a);
  CUDA_CHECK(cudaGetLastError());
  if (a.splits > 1) {
    launch(attn_combine_kernel, dim3(rows, kHeads), dim3(kHead / 4), 0, s, pdl, (const float*)scratch.o,
           (const float*)scratch.ml, sink, pos, table, a.splits, out);
    CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace attn
