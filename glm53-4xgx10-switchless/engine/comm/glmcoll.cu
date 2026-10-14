// GLM-5.3's collectives (glmcoll.h), from the ring_dev.cuh halves.
#include "comm/glmcoll.h"
#include "comm/collectives.h"
#include "comm/ring_dev.cuh"
#include "common/check.h"

namespace comm::glm {
namespace {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float bf2f(uint32_t b) { return __uint_as_float((b & 0xFFFFu) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfround(float f) { return bf2f(f2bf(f)); }

template <class... KArgs, class... Args>
void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t st, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

#ifdef GLM_COMM_TEST_HOOKS
struct TestStall {
  uint32_t ns;
  int block, piece;
};
__device__ TestStall g_test_stall;
__device__ __forceinline__ void stall_for_test(size_t i, size_t npieces) {
  const TestStall t = g_test_stall;
  if (!t.ns || (int)blockIdx.x != t.block || (long)i != (t.piece < 0 ? (long)npieces - 4 : (long)t.piece)) return;
  if (threadIdx.x == 0) {
    uint64_t t0, tn;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tn));
    while (tn - t0 < t.ns);
  }
  __syncthreads();
}
#define TEST_STALL(i, npieces) stall_for_test(i, npieces)
#else
#define TEST_STALL(i, npieces) ((void)0)
#endif

// ---- (a) the hidden-size all-reduce ----------------------------------------------------------------------------
//
// Grid: S column slices of 6144 / S columns (S = 8, 16 or 32) times G row groups of RG rows, at most 48 blocks
// (a kernel that waits must have them all resident). A row's slice is 768 / S items of 8 columns, one a thread
// (TPR threads a row, a multiple of 32; the spare ones only help with the norm). Each item is 32 bytes of
// every contribution (two 16-byte loads of host memory).
//
// The norm's statistic in attn's rmsnorm order: its thread u (of 256) sums the squares of columns
// [24u, 24u + 24) in order; the 256 sums go through eight warp xor-trees, added in warp order. Here a slice
// holds 256 / S of those threads' columns (3 items each): their owners sum them from shared memory and leave the
// sums in g_ss; once every slice of the row group has (a counter barrier over the group's S blocks), the row's
// first warp redoes rmsnorm's trees and warp-order sum over the row's 256 sums.
constexpr int kMaxGroups = 16;
__device__ float g_ss[kMaxRows][256];
__device__ unsigned g_bar_count[kMaxGroups];
__device__ unsigned long long g_bar_gen[kMaxGroups];

struct ArShape {
  int S, items, tpr, RG, G;
};
ArShape ar_shape(int R, int S) {
  ArShape a;
  a.S = S;
  a.items = 768 / S;
  a.tpr = (a.items + 31) / 32 * 32;
  const int gmax = 48 / S;
  a.RG = (R + gmax - 1) / gmax;
  a.G = (R + a.RG - 1) / a.RG;
  return a;
}

__device__ __forceinline__ float rms_rstd(float ss, int D, float eps) {
  const float var = __fdiv_rn(ss, (float)D);
  return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(var, eps)));
}
__device__ __forceinline__ float rms_out(float x, float rstd, float w) {
  return bfround(__fmul_rn(w, bfround(__fmul_rn(x, rstd))));
}

// kTwo (allreduce2_add_norm): two partials, part then part2, sent as one contribution of 2 R rows; the value added
// to the residual is bf16(bf16(sum) + bf16(sum2)), the experts' finalize (the routed and the shared expert's sums).
// kPosted (allreduce2_add_norm_posted): the producer kernel wrote this rank's contribution into the send buffer and
// posted it (the producer half in its epilogue); this kernel is the consumer half only.
template <bool kNorm, bool kTwo = false, bool kPosted = false>
__global__ void __launch_bounds__(512) k_allreduce_hidden(DevChannel ch, const float* part, int R, int S, int RG,
                                                          float* out_f32, uint16_t* out_bf16, const uint16_t* res,
                                                          uint16_t* res_out, const uint16_t* w, float eps, uint16_t* xn,
                                                          const float* part2 = nullptr) {
  extern __shared__ float sh[];  // kNorm: [RG][768 / S * 8] the slice's h values; then [RG] rstd
  const int items = 768 / S, tpr = (items + 31) / 32 * 32;
  const int slice = blockIdx.x % S, grp = blockIdx.x / S;
  const int rr = threadIdx.x / tpr, it = threadIdx.x % tpr, row = grp * RG + rr;
  const bool item = row < R && it < items;
  const int c0 = slice * (kDim / S) + it * 8;
  const size_t off = ((size_t)row * kDim + c0) * 4;
  pdl_wait();
  const uint64_t s = next_seq(ch);
  uint8_t* sb = send_buf(ch, s);
  if (!kPosted && item) {
    const uint4* p = reinterpret_cast<const uint4*>(part + (size_t)row * kDim + c0);
    const uint4 a = p[0], b = p[1];
    *reinterpret_cast<uint4*>(sb + off) = a;
    *reinterpret_cast<uint4*>(sb + off + 16) = b;
    if constexpr (kTwo) {
      const uint4* p2 = reinterpret_cast<const uint4*>(part2 + (size_t)row * kDim + c0);
      const uint4 a2 = p2[0], b2 = p2[1];
      *reinterpret_cast<uint4*>(sb + (size_t)R * kHiddenF32Row + off) = a2;
      *reinterpret_cast<uint4*>(sb + (size_t)R * kHiddenF32Row + off + 16) = b2;
    }
  }
  if (!kPosted) post_grid(ch, s, (uint32_t)((size_t)R * kDim * 4 * (kTwo ? 2 : 1)));
  pdl_launch();
  uint4 rv{}, wv{};
  if (kNorm && item) {  // while the contributions travel
    rv = *reinterpret_cast<const uint4*>(res + (size_t)row * kDim + c0);
    wv = *reinterpret_cast<const uint4*>(w + c0);
  }
  wait_all(ch, s);
  float v[8];
  if (item) {
    const float4 a = sum4_f32(ch, s, off), b = sum4_f32(ch, s, off + 16);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
    if constexpr (kTwo) {
      const size_t o2 = (size_t)R * kHiddenF32Row + off;
      const float4 a2 = sum4_f32(ch, s, o2), b2 = sum4_f32(ch, s, o2 + 16);
      const float u[8] = {a2.x, a2.y, a2.z, a2.w, b2.x, b2.y, b2.z, b2.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) v[j] = bfround(__fadd_rn(bfround(v[j]), bfround(u[j])));
    }
  }
  finish_grid(ch, s);
  if constexpr (!kNorm) {
    if (!item) return;
    if (out_f32) {
      float4* o = reinterpret_cast<float4*>(out_f32 + (size_t)row * kDim + c0);
      o[0] = make_float4(v[0], v[1], v[2], v[3]);
      o[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
    if (out_bf16) {
      uint32_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) q[j] = (uint32_t)f2bf(v[2 * j]) | (uint32_t)f2bf(v[2 * j + 1]) << 16;
      *reinterpret_cast<uint4*>(out_bf16 + (size_t)row * kDim + c0) = make_uint4(q[0], q[1], q[2], q[3]);
    }
  } else {
    // h = bf16(res + bf16(sum)): the new residual
    float h[8];
    if (item) {
      const uint32_t rw[4] = {rv.x, rv.y, rv.z, rv.w};
      uint32_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        h[2 * j] = bfround(__fadd_rn(bf2f(rw[j]), bfround(v[2 * j])));
        h[2 * j + 1] = bfround(__fadd_rn(bf2f(rw[j] >> 16), bfround(v[2 * j + 1])));
        q[j] = (uint32_t)f2bf(h[2 * j]) | (uint32_t)f2bf(h[2 * j + 1]) << 16;
      }
      *reinterpret_cast<uint4*>(res_out + (size_t)row * kDim + c0) = make_uint4(q[0], q[1], q[2], q[3]);
      float* hs = sh + (size_t)rr * items * 8 + it * 8;
#pragma unroll
      for (int j = 0; j < 8; ++j) hs[j] = h[j];
    }
    __syncthreads();
    // rmsnorm's thread u of this slice: columns [24u, 24u + 24) of the slice
    const int owners = 256 / S;
    if (row < R && it < owners) {
      const float* hs = sh + (size_t)rr * items * 8 + it * 24;
      float ss = 0.f;
#pragma unroll
      for (int j = 0; j < 24; ++j) ss = __fadd_rn(ss, __fmul_rn(hs[j], hs[j]));
      g_ss[row][slice * owners + it] = ss;
    }
    __syncthreads();
    // the S blocks of this row group meet
    if (threadIdx.x == 0) {
      const unsigned long long gen = *(volatile unsigned long long*)&g_bar_gen[grp];
      __threadfence();
      if (atomicAdd(&g_bar_count[grp], 1u) == (unsigned)S - 1) {
        g_bar_count[grp] = 0;
        __threadfence();
        st_release_gpu(&g_bar_gen[grp], gen + 1);
      } else {
        while (ld_acquire_gpu(&g_bar_gen[grp]) == gen) {
        }
      }
    }
    __syncthreads();
    // the row's first warp: rmsnorm's block_sum over the 256 sums (eight xor-trees, then the warps in order)
    float* rstd_sh = sh + (size_t)RG * items * 8;
    if (row < R && it < 32) {
      float tot = 0.f;
#pragma unroll
      for (int wp = 0; wp < 8; ++wp) {
        float x = __ldcg(&g_ss[row][wp * 32 + it]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, o));
        tot = __fadd_rn(tot, x);
      }
      if (it == 0) rstd_sh[rr] = rms_rstd(tot, kDim, eps);
    }
    __syncthreads();
    if (!item) return;
    const float rstd = rstd_sh[rr];
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
    uint32_t q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      q[j] = (uint32_t)f2bf(rms_out(h[2 * j], rstd, bf2f(ww[j]))) |
             (uint32_t)f2bf(rms_out(h[2 * j + 1], rstd, bf2f(ww[j] >> 16))) << 16;
    *reinterpret_cast<uint4*>(xn + (size_t)row * kDim + c0) = make_uint4(q[0], q[1], q[2], q[3]);
  }
}

// The fused all-reduce with one block a row (slices = 1): 768 threads of 8 columns, the whole row in one block, so
// rmsnorm's statistic needs no barrier across blocks (its 256 run sums go through shared memory, then the same
// eight xor-trees and warp-order sum); a single-block grid's wait_all also has no release hop. The same bits as
// the sliced kernel (the same sums in the same order).
template <bool kTwo, bool kPosted>
__global__ void __launch_bounds__(768) k_ar_norm_row(DevChannel ch, const float* part, int R, const uint16_t* res,
                                                     uint16_t* res_out, const uint16_t* w, float eps, uint16_t* xn,
                                                     const float* part2) {
  __shared__ float hs[kDim];
  __shared__ float ss[256];
  __shared__ float rstd_sh;
  const int row = blockIdx.x, c0 = threadIdx.x * 8;
  const size_t off = ((size_t)row * kDim + c0) * 4;
  pdl_wait();
  const uint64_t s = next_seq(ch);
  if (!kPosted) {
    uint8_t* sb = send_buf(ch, s);
    const uint4* p = reinterpret_cast<const uint4*>(part + (size_t)row * kDim + c0);
    const uint4 a = p[0], b = p[1];
    *reinterpret_cast<uint4*>(sb + off) = a;
    *reinterpret_cast<uint4*>(sb + off + 16) = b;
    if constexpr (kTwo) {
      const uint4* p2 = reinterpret_cast<const uint4*>(part2 + (size_t)row * kDim + c0);
      const uint4 a2 = p2[0], b2 = p2[1];
      *reinterpret_cast<uint4*>(sb + (size_t)R * kHiddenF32Row + off) = a2;
      *reinterpret_cast<uint4*>(sb + (size_t)R * kHiddenF32Row + off + 16) = b2;
    }
    post_grid(ch, s, (uint32_t)((size_t)R * kDim * 4 * (kTwo ? 2 : 1)));
  }
  pdl_launch();
  const uint4 rv = *reinterpret_cast<const uint4*>(res + (size_t)row * kDim + c0);
  const uint4 wv = *reinterpret_cast<const uint4*>(w + c0);
  wait_all(ch, s);
  float v[8];
  {
    const float4 a = sum4_f32(ch, s, off), b = sum4_f32(ch, s, off + 16);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
    if constexpr (kTwo) {
      const size_t o2 = (size_t)R * kHiddenF32Row + off;
      const float4 a2 = sum4_f32(ch, s, o2), b2 = sum4_f32(ch, s, o2 + 16);
      const float u[8] = {a2.x, a2.y, a2.z, a2.w, b2.x, b2.y, b2.z, b2.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) v[j] = bfround(__fadd_rn(bfround(v[j]), bfround(u[j])));
    }
  }
  finish_grid(ch, s);
  float h[8];
  {
    const uint32_t rw[4] = {rv.x, rv.y, rv.z, rv.w};
    uint32_t q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      h[2 * j] = bfround(__fadd_rn(bf2f(rw[j]), bfround(v[2 * j])));
      h[2 * j + 1] = bfround(__fadd_rn(bf2f(rw[j] >> 16), bfround(v[2 * j + 1])));
      q[j] = (uint32_t)f2bf(h[2 * j]) | (uint32_t)f2bf(h[2 * j + 1]) << 16;
    }
    *reinterpret_cast<uint4*>(res_out + (size_t)row * kDim + c0) = make_uint4(q[0], q[1], q[2], q[3]);
#pragma unroll
    for (int j = 0; j < 8; ++j) hs[c0 + j] = h[j];
  }
  __syncthreads();
  if (threadIdx.x < 256) {  // rmsnorm's thread u: columns [24u, 24u + 24) in order
    const float* x = hs + threadIdx.x * 24;
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 24; ++j) a = __fadd_rn(a, __fmul_rn(x[j], x[j]));
    ss[threadIdx.x] = a;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float tot = 0.f;
#pragma unroll
    for (int wp = 0; wp < 8; ++wp) {
      float x = ss[wp * 32 + threadIdx.x];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, o));
      tot = __fadd_rn(tot, x);
    }
    if (threadIdx.x == 0) rstd_sh = rms_rstd(tot, kDim, eps);
  }
  __syncthreads();
  const float rstd = rstd_sh;
  const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
  uint32_t q[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    q[j] = (uint32_t)f2bf(rms_out(h[2 * j], rstd, bf2f(ww[j]))) |
           (uint32_t)f2bf(rms_out(h[2 * j + 1], rstd, bf2f(ww[j] >> 16))) << 16;
  *reinterpret_cast<uint4*>(xn + (size_t)row * kDim + c0) = make_uint4(q[0], q[1], q[2], q[3]);
}

template <bool kNorm, bool kTwo = false, bool kPosted = false>
void launch_ar(const DevChannel& ch, const float* part, int R, float* out_f32, uint16_t* out_bf16, const uint16_t* res,
               uint16_t* res_out, const uint16_t* w, float eps, uint16_t* xn, cudaStream_t st, bool pdl, int S,
               const float* part2 = nullptr) {
  if (S == 0) S = 8;  // measured best or equal at 1-16 rows (docs/ring.md)
  if constexpr (kNorm) {
    if (S == 1) {  // one block a row
      if (R < 1 || R > kMaxRows || (size_t)R * kHiddenF32Row * (kTwo ? 2 : 1) > ch.cap)
        die("comm::glm: all-reduce of %d rows (1 to %d; cap %u)", R, kMaxRows, ch.cap);
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(R);
      cfg.blockDim = dim3(768);
      cfg.stream = st;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = attr;
      cfg.numAttrs = pdl ? 1 : 0;
      CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_ar_norm_row<kTwo, kPosted>, ch, part, R, res, res_out, w, eps, xn, part2));
      return;
    }
  }
  if (S != 8 && S != 16 && S != 32) die("comm::glm: all-reduce over %d slices (8, 16 or 32)", S);
  if (R < 1 || R > kMaxRows || (size_t)R * kHiddenF32Row * (kTwo ? 2 : 1) > ch.cap)
    die("comm::glm: all-reduce of %d rows of %zu B (1 to %d rows; cap %u)", R, kHiddenF32Row, kMaxRows, ch.cap);
  const ArShape a = ar_shape(R, S);
  if (a.G > kMaxGroups || a.S * a.G > 48 || a.RG * a.tpr > 512) die("comm::glm: all-reduce shape");
  const size_t smem = kNorm ? ((size_t)a.RG * a.items * 8 + a.RG) * 4 : 0;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(a.S * a.G);
  cfg.blockDim = dim3(a.RG * a.tpr);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_allreduce_hidden<kNorm, kTwo, kPosted>, ch, part, R, a.S, a.RG, out_f32, out_bf16, res,
                                res_out, w, eps, xn, part2));
}

// ---- (b) all-gathers by rows -----------------------------------------------------------------------------------
struct RowArgs {
  const uint8_t* in[kMaxRowSegs];
  uint8_t* out[kMaxRowSegs];
  size_t ld_in[kMaxRowSegs], ld_out[kMaxRowSegs], rb[kMaxRowSegs];
  size_t off[kMaxRowSegs + 1];  // stream offset of each segment's rows; off[nseg] = a rank's bytes
  int own[kMaxRowSegs];         // in place: this rank's block is not rewritten
  int nseg;
};

// comm::allgather_multi's pipeline (copies two ahead, posts one ahead), with row-strided ends.
__global__ void __launch_bounds__(1024) k_allgather_rows(DevChannel ch, RowArgs a, uint32_t chunk) {
  pdl_wait();
  const uint64_t s0 = next_seq(ch);
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const size_t total = a.off[a.nseg], nchunks = (total + chunk - 1) / chunk;
  auto len = [&](size_t j) { return (uint32_t)(total - j * chunk < chunk ? total - j * chunk : chunk); };
  // stream offset x -> segment g, row, column
  auto where = [&](size_t x, int& g, size_t& row, size_t& col) {
    g = 0;
    while (g + 1 < a.nseg && x >= a.off[g + 1]) ++g;
    const size_t l = x - a.off[g];
    row = l / a.rb[g], col = l - row * a.rb[g];
  };
  auto copy = [&](size_t j, uint64_t s) {
    const uint32_t units = len(j) / 16;
    uint4* sb = (uint4*)send_buf(ch, s);
    for (uint32_t i = t0; i < units; i += stride) {
      int g;
      size_t row, col;
      where(j * chunk + (size_t)i * 16, g, row, col);
      sb[i] = *(const uint4*)(a.in[g] + row * a.ld_in[g] + col);
    }
  };
  copy(0, s0);
  post_grid(ch, s0, len(0));
  pdl_launch();
  if (nchunks > 1) copy(1, s0 + 1);
  for (size_t j = 0; j < nchunks; j++) {
    const uint64_t s = s0 + j;
    wait_all(ch, s);
    if (j + 1 < nchunks) post_grid(ch, s + 1, len(j + 1));
    TEST_STALL(j, nchunks);
    const uint32_t units = len(j) / 16;
    for (uint32_t i = t0; i < units; i += stride) {
      int g;
      size_t row, col;
      where(j * chunk + (size_t)i * 16, g, row, col);
      uint8_t* o = a.out[g] + row * a.ld_out[g] + col;
      uint4 v[kRanks];
#pragma unroll
      for (int k = 0; k < kRanks; k++) v[k] = ld_cv16(src(ch, s, k) + (size_t)i * 16);
#pragma unroll
      for (int k = 0; k < kRanks; k++)
        if (!(a.own[g] && k == ch.rank)) *(uint4*)(o + k * a.rb[g]) = v[k];
    }
    if (j + 2 < nchunks) copy(j + 2, s + 2);
  }
  finish_grid(ch, s0 + nchunks - 1);
}

// ---- (d) the top-k exchange ------------------------------------------------------------------------------------
// One block a row (R <= 16 blocks), 4K threads: candidate j is rank j / K's j % K; its place is the number of
// candidates better than it (value greater, or equal and index lower).
__global__ void __launch_bounds__(256) k_topk(DevChannel ch, const Cand* local, Cand* out, int R, int K, int Kp) {
  __shared__ Cand c[4 * kMaxTopK];
  pdl_wait();
  const uint64_t s = next_seq(ch);
  Cand* sb = (Cand*)send_buf(ch, s);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < R * Kp; i += gridDim.x * blockDim.x) {
    const int r = i / Kp, j = i % Kp;
    sb[i] = j < K ? local[(size_t)r * K + j] : Cand{0.f, 0};
  }
  post_grid(ch, s, (uint32_t)(R * Kp * sizeof(Cand)));
  pdl_launch();
  wait_all(ch, s);
  const int r = blockIdx.x, t = threadIdx.x;
  if (t < 4 * K) {
    const uint2 w = __ldcv((const uint2*)src(ch, s, t / K) + (size_t)r * Kp + t % K);
    c[t] = Cand{__uint_as_float(w.x), (int)w.y};
  }
  finish_grid(ch, s);
  __syncthreads();
  if (t >= 4 * K) return;
  const Cand me = c[t];
  int place = 0;
  for (int j = 0; j < 4 * K; ++j) {
    const Cand o = c[j];
    place += o.v > me.v || (o.v == me.v && o.i < me.i);
  }
  if (place < K) out[(size_t)r * K + place] = me;
}

}  // namespace

void allreduce_hidden(const DevChannel& ch, const float* part, int R, float* out_f32, uint16_t* out_bf16,
                      cudaStream_t st, bool pdl, int slices) {
  launch_ar<false>(ch, part, R, out_f32, out_bf16, nullptr, nullptr, nullptr, 0.f, nullptr, st, pdl, slices);
}

void allreduce_add_norm(const DevChannel& ch, const float* part, int R, const uint16_t* res, uint16_t* res_out,
                        const uint16_t* w, float eps, uint16_t* xn, cudaStream_t st, bool pdl, int slices) {
  launch_ar<true>(ch, part, R, nullptr, nullptr, res, res_out, w, eps, xn, st, pdl, slices);
}

void allreduce2_add_norm(const DevChannel& ch, const float* part, const float* part2, int R, const uint16_t* res,
                         uint16_t* res_out, const uint16_t* w, float eps, uint16_t* xn, cudaStream_t st, bool pdl,
                         int slices) {
  launch_ar<true, true>(ch, part, R, nullptr, nullptr, res, res_out, w, eps, xn, st, pdl, slices, part2);
}

void allreduce_add_norm_posted(const DevChannel& ch, int R, const uint16_t* res, uint16_t* res_out, const uint16_t* w,
                               float eps, uint16_t* xn, cudaStream_t st, bool pdl, int slices) {
  launch_ar<true, false, true>(ch, nullptr, R, nullptr, nullptr, res, res_out, w, eps, xn, st, pdl, slices);
}

void allreduce2_add_norm_posted(const DevChannel& ch, int R, const uint16_t* res, uint16_t* res_out, const uint16_t* w,
                                float eps, uint16_t* xn, cudaStream_t st, bool pdl, int slices) {
  launch_ar<true, true, true>(ch, nullptr, R, nullptr, nullptr, res, res_out, w, eps, xn, st, pdl, slices, nullptr);
}

void allgather_rows(const DevChannel& ch, const RowSeg* segs, int nseg, int rows, cudaStream_t st, int blocks,
                    size_t chunk, bool pdl) {
  if (nseg < 1 || nseg > kMaxRowSegs || rows < 1) die("comm::glm: all-gather of %d segments of %d rows", nseg, rows);
  RowArgs a{};
  a.nseg = nseg;
  for (int g = 0; g < nseg; ++g) {
    const RowSeg& e = segs[g];
    if (!e.rb || e.rb % 16 || e.ld_in % 16 || e.ld_out % 16 || ((uintptr_t)e.in | (uintptr_t)e.out) % 16 ||
        (rows > 1 && (e.ld_in < e.rb || e.ld_out < 4 * e.rb)))
      die("comm::glm: all-gather segment of %zu-byte rows (strides %zu, %zu; multiples of 16)", e.rb, e.ld_in, e.ld_out);
    a.in[g] = (const uint8_t*)e.in, a.out[g] = (uint8_t*)e.out;
    a.ld_in[g] = e.ld_in, a.ld_out[g] = e.ld_out, a.rb[g] = e.rb;
    a.own[g] = e.in == (const uint8_t*)e.out + ch.rank * e.rb && e.ld_in == e.ld_out;
    a.off[g + 1] = a.off[g] + (size_t)rows * e.rb;
  }
  const size_t c = (chunk ? chunk : ch.cap) & ~(size_t)63;
  if (!c || c > ch.cap) die("comm::glm: all-gather chunk of %zu B (cap %u)", chunk, ch.cap);
  const size_t first = a.off[nseg] < c ? a.off[nseg] : c;
  int threads = block_for(first), grid = blocks;
  if (grid <= 0) {
    grid = (int)((first / 16 + threads - 1) / threads);
    grid = grid < 1 ? 1 : grid > 16 ? 16 : grid;
  }
  if (grid > 48) grid = 48;
  launch(k_allgather_rows, dim3(grid), dim3(threads), st, pdl, ch, a, (uint32_t)c);
}

void allgather_xproj(const DevChannel& ch, uint16_t* xg, float* iw, int R, bool full, cudaStream_t st, bool pdl) {
  const int xr = full ? kXRows : kXRowsShared;
  const size_t ld = (size_t)4 * xr * 2;
  RowSeg s[2] = {{xg + (size_t)ch.rank * xr, ld, xg, ld, (size_t)xr * 2},
                 {iw + ch.rank * kIdxHeadsRank, (size_t)kIdxHeads * 4, iw, (size_t)kIdxHeads * 4, (size_t)kIdxHeadsRank * 4}};
  allgather_rows(ch, s, full ? 2 : 1, R, st, 0, 0, pdl);
}

void allgather_idxq(const DevChannel& ch, uint16_t* iq, int R, cudaStream_t st, bool pdl) {
  const size_t rb = kIdxQRow, ld = 4 * rb;
  RowSeg s{(const uint8_t*)iq + ch.rank * rb, ld, iq, ld, rb};
  allgather_rows(ch, &s, 1, R, st, 0, 0, pdl);
}

void allgather_cand(const DevChannel& ch, const int* pos, const uint32_t* key, int* pos_all, uint32_t* key_all, int R,
                    cudaStream_t st, bool pdl) {
  const size_t rb = (size_t)kTopK * 4;
  RowSeg s[2] = {{pos, rb, pos_all, 4 * rb, rb}, {key, rb, key_all, 4 * rb, rb}};
  allgather_rows(ch, s, 2, R, st, 0, 0, pdl);
}

void allgather_logits_f32(const DevChannel& ch, const float* local, size_t ld_local, float* all, int R, cudaStream_t st,
                          bool pdl) {
  RowSeg s{local, ld_local * 4, all, (size_t)kVocab * 4, kLogitsF32Row};
  allgather_rows(ch, &s, 1, R, st, 0, 0, pdl);
}

void allgather_logits_bf16(const DevChannel& ch, const uint16_t* local, size_t ld_local, uint16_t* all, int R,
                           cudaStream_t st, bool pdl) {
  RowSeg s{local, ld_local * 2, all, (size_t)kVocab * 2, kLogitsBf16Row};
  allgather_rows(ch, &s, 1, R, st, 0, 0, pdl);
}

void topk_exchange(const DevChannel& ch, const Cand* local, Cand* out, int R, int K, cudaStream_t st, bool pdl) {
  const int Kp = (K + 1) & ~1;  // rows of whole 16-byte units
  if (R < 1 || R > kMaxRows || K < 1 || K > kMaxTopK || (size_t)R * Kp * sizeof(Cand) > ch.cap)
    die("comm::glm: top-%d exchange of %d rows", K, R);
  launch(k_topk, dim3(R), dim3(4 * K < 32 ? 32 : (4 * K + 31) / 32 * 32), st, pdl, ch, local, out, R, K, Kp);
}

#ifdef GLM_COMM_TEST_HOOKS
void test_stall_rows(uint32_t ns, int block, int piece, cudaStream_t st) {
  const TestStall t{ns, block, piece};
  CUDA_CHECK(cudaMemcpyToSymbolAsync(g_test_stall, &t, sizeof t, 0, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
}
#endif

}  // namespace comm::glm
