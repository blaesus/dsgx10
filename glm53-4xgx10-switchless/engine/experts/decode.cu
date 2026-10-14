// The decode expert step for 1-16 rows (docs/experts.md, "Decode"): two PDL kernels.
//
// k_up: a persistent grid walks tasks in a fixed order: the shared expert's (FP8, gate and up strips of 8
// channels: they need nothing from the router, so the first wave loads them before griddepcontrol.wait), then
// for every expert of the union (the rows' picks, each read once, in expert-id order) its gate and up strips.
// A task is one CTA: WK warps split K, each streams its blocks (both formats, chosen per matrix at run time)
// into bf16 B fragments of m16n8k16 MMAs whose A rows are the step's rows (rows 8..15 in the upper half from
// 9 rows on); the partials meet in shared memory and one warp a strip pair applies the per-tensor scales,
// rounds gate and up to bf16, SwiGLU, and writes h (bf16) to the slot's workspace.
//
// k_down: warps of NT output strips, every slot of the union in expert-id order, acc += (s2 * w_row) * c, the
// routing weight 0 for rows that did not pick the expert, so a row's bits do not depend on the other rows;
// each warp first does the shared expert's down projection for its strips (weights loaded before the PDL
// wait) into its own output. Both outputs f32 partial sums.
#include <cstdio>

#include "comm/ring_dev.cuh"
#include "common/check.h"
#include "experts/formats.cuh"

namespace gx {

namespace {

constexpr int kStrips = kH / 8;  // 768 strips of down
constexpr int kUpB = kH / kBlkK;   // 12 blocks of gate/up along K
constexpr int kShU = kH / 128;     // 48 FP8 units of the shared gate/up along K

// workspace: h [kMaxUnion + 1][kMaxRows][kI] bf16, then k_up's task counters (claimed, CTAs done)
constexpr size_t kWsH = (size_t)(kMaxUnion + 1) * kMaxRows * kI * 2;
constexpr size_t kWsBytes = kWsH + 256;

struct Params {
  const __nv_bfloat16* x;
  int M;
  const int* idx;
  const float* w;
  const float* logits;  // fused routing (DecodeCfg::logits): idx, w unused
  const float* bias;
  LayerW L;
  __nv_bfloat16* h;  // [slot][M][kI]
  float *yr, *ys;
  int* ctr;          // k_up's task queue (null: static order); zero between launches
  bool post;         // k_down: yr, ys into ch's send buffer, posted (DecodeCfg::post)
  comm::DevChannel ch;
};

// The union of the rows' picks, sorted by expert id. All threads of the CTA (>= 128). W: also each slot's
// per-row routing weights (0 where the row did not pick it) and the down matrices' descriptors.
template <bool W>
struct Union {
  int n;
  uint32_t mask[8];
  int pre[8];
  int ri[kMaxRows * kTopK];    // fused routing: the rows' picks and weights
  float rw[kMaxRows * kTopK];
  int id[kMaxUnion];
  float w[W ? kMaxUnion : 1][kMaxRows];
  MatDesc d[W ? kMaxUnion : 1];
};
template <bool W>
__device__ void build_union(Union<W>& U, const Params& a) {
  const int tid = threadIdx.x, ne = a.M * kTopK;
  if (tid < 8) U.mask[tid] = 0;
  if (W)
    for (int i = tid; i < kMaxUnion * kMaxRows; i += blockDim.x) (&U.w[0][0])[i] = 0.f;
  const bool fused = a.logits != nullptr;
  if (fused)  // a warp a row, route()'s code
    for (int row = tid >> 5; row < a.M; row += blockDim.x >> 5)
      route_warp(a.logits + (size_t)row * kE, a.bias, tid & 31, U.ri + row * kTopK, U.rw + row * kTopK);
  __syncthreads();
  for (int i = tid; i < ne; i += blockDim.x) {
    const int e = fused ? U.ri[i] : __ldcg(a.idx + i);
    atomicOr(&U.mask[e >> 5], 1u << (e & 31));
  }
  __syncthreads();
  if (tid < 32) {
    const int c = tid < 8 ? __popc(U.mask[tid]) : 0;
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    if (tid < 8) U.pre[tid] = incl - c;
    if (tid == 31) U.n = incl;
  }
  __syncthreads();
  for (int i = tid; i < ne; i += blockDim.x) {
    const int e = fused ? U.ri[i] : __ldcg(a.idx + i);
    const int r = U.pre[e >> 5] + __popc(U.mask[e >> 5] & ((1u << (e & 31)) - 1));
    U.id[r] = e;
    if (W) {
      U.w[r][i / kTopK] = fused ? U.rw[i] : __ldcg(a.w + i);
      U.d[r] = a.L.desc[e * 3 + 2];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float silu_mul(float g, float u) {  // h = bf16(bf16(silu(g)) * u), g and u bf16 values
  const float s = __bfloat162float(__float2bfloat16_rn(g / (1.f + expf(-g))));
  return s * u;
}

// ---------------------------------------------------------------------------------------------------------------
// k_up. NT strips of gate and of up a task (64 / NT tasks an expert), WK warps along K, P blocks in flight a warp.
template <int NT, int WK, int P, bool R16>
__global__ void __launch_bounds__(32 * WK) k_up(Params a) {
  constexpr int NTL = 2 * NT, TPE = 64 / NT, PS = 2;  // tiles a task, tasks an expert, FP8 units in flight
  __shared__ Union<false> U;
  __shared__ float red[WK][NTL][32][4];
  __shared__ int s_next[2];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int M = a.M;
  const __nv_bfloat16* x0 = a.x + (size_t)min(g, M - 1) * kH + 16 * t;
  const __nv_bfloat16* x1 = a.x + (size_t)min(g + 8, M - 1) * kH + 16 * t;

  // the shared expert's tasks: units [u0, u1) of the strips
  const int u0 = kShU * warp / WK, u1 = kShU * (warp + 1) / WK, nu = u1 - u0;
  Raw8 sb[PS][NTL];
  auto sh_ptr = [&](int task, int i) {
    const uint8_t* m = i < NT ? a.L.sh.g : a.L.sh.u;
    return m + ((size_t)(task * NT + i % NT) * kShU) * 1024 + 32 * lane;
  };
  auto sh_load = [&](int task, int s, int u) {
#pragma unroll
    for (int i = 0; i < NTL; ++i) {
      const uint8_t* p = sh_ptr(task, i) + (size_t)(u0 + u) * 1024;
      sb[s][i].q0 = ldg_stream128(p);
      sb[s][i].q1 = ldg_stream128(p + 16);
    }
  };
  int task = blockIdx.x;
  bool pre = task < TPE;
  if (pre)
#pragma unroll
    for (int s = 0; s < PS; ++s)
      if (s < nu) sh_load(task, s, s);
  pdl_wait();
  build_union(U, a);
  pdl_launch_dependents();
  const int ntask = TPE * (1 + U.n);

  // The first wave is static (CTA b: task b, so the shared expert's tasks load before the wait); then, with a
  // queue, each CTA claims the next task when it starts one, so CTAs that finish early take more.
#pragma unroll 1
  for (int it = 0; task < ntask; ++it) {
    if (a.ctr && threadIdx.x == 0) s_next[it & 1] = (int)gridDim.x + atomicAdd(a.ctr, 1);
    float c[NTL][4];
#pragma unroll
    for (int i = 0; i < NTL; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
    const bool sh = task < TPE;
    const int slot = sh ? kShSlot : task / TPE - 1, st = sh ? task : task % TPE;  // st: strip group
    if (sh) {
      if (!pre)
#pragma unroll
        for (int s = 0; s < PS; ++s)
          if (s < nu) sh_load(task, s, s);
      const float* sc[NTL];
#pragma unroll
      for (int i = 0; i < NTL; ++i) sc[i] = (i < NT ? a.L.sh.gs : a.L.sh.us) + ((st * NT + i % NT) / 16) * kShU;
      for (int u = 0; u < nu; u += PS) {
#pragma unroll
        for (int s = 0; s < PS; ++s) {
          if (u + s >= nu) break;
          float cu[NTL][4];
#pragma unroll
          for (int i = 0; i < NTL; ++i) cu[i][0] = cu[i][1] = cu[i][2] = cu[i][3] = 0.f;
          const int k = 128 * (u0 + u + s);
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            XF f;
            load_x<R16>(f, x0, x1, k + 64 * q);
#pragma unroll
            for (int i = 0; i < NTL; ++i) {
              uint32_t o[8];
              dec8(q ? sb[s][i].q1 : sb[s][i].q0, o);
              mma4(cu[i], f, o);
            }
          }
#pragma unroll
          for (int i = 0; i < NTL; ++i) {
            const float s8 = __ldg(sc[i] + u0 + u + s);
#pragma unroll
            for (int e = 0; e < 4; ++e) c[i][e] = fmaf(s8, cu[i][e], c[i][e]);
          }
          if (u + s + PS < nu) sh_load(task, s, u + s + PS);
        }
      }
    } else {
      const int ex = U.id[slot];
      const MatDesc* dg = a.L.desc + ex * 3;
      uint32_t fm[2], tab[2][8];
      const uint8_t* base[NTL];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        fm[m] = __ldg(&dg[m].fmt);
#pragma unroll
        for (int j = 0; j < 8; ++j) tab[m][j] = __ldg(&dg[m].tab[j]);
      }
#pragma unroll
      for (int i = 0; i < NTL; ++i) {
        const int m = i / NT;
        base[i] = a.L.pool + __ldg(&dg[m].off) + (size_t)(st * NT + i % NT) * kUpB * (fm[m] == kNVFP4 ? kQ4Blk : kS3Blk);
      }
      const int b0 = kUpB * warp / WK, nb = kUpB * (warp + 1) / WK - b0;
      Raw rb[P][NTL];
      auto load = [&](int s, int b) {
#pragma unroll
        for (int i = 0; i < NTL; ++i) {
          const int m = i / NT;
          load_blk(rb[s][i], base[i] + (size_t)(b0 + b) * (fm[m] == kNVFP4 ? kQ4Blk : kS3Blk), fm[m], lane);
        }
      };
#pragma unroll
      for (int s = 0; s < P; ++s)
        if (s < nb) load(s, s);
      for (int b = 0; b < nb; b += P) {
#pragma unroll
        for (int s = 0; s < P; ++s) {
          if (b + s >= nb) break;
          const int k = kBlkK * (b0 + b + s);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
              XF f;
              load_x<R16>(f, x0, x1, k + 128 * jj + 64 * q);
#pragma unroll
              for (int i = 0; i < NTL; ++i) {
                uint32_t o[8];
                dec_group(rb[s][i], jj, q, fm[i / NT], tab[i / NT], o);
                mma4(c[i], f, o);
              }
            }
          if (b + s + P < nb) load(s, b + s + P);
        }
      }
    }
    pre = false;

    // partials -> shared memory; warp i % WK finishes strip pair i (gate tile i, up tile NT + i)
#pragma unroll
    for (int i = 0; i < NTL; ++i)
      *reinterpret_cast<float4*>(&red[warp][i][lane][0]) = make_float4(c[i][0], c[i][1], c[i][2], c[i][3]);
    __syncthreads();
    float kg = 1.f, ku = 1.f;
    if (!sh) {
      const MatDesc* dg = a.L.desc + U.id[slot] * 3;
      kg = __ldg(&dg[0].s2) * fmt_k(__ldg(&dg[0].fmt));
      ku = __ldg(&dg[1].s2) * fmt_k(__ldg(&dg[1].fmt));
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      if (i % WK != warp) continue;
      float gs[4], us[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) gs[e] = us[e] = 0.f;
#pragma unroll
      for (int w = 0; w < WK; ++w)
#pragma unroll
        for (int e = 0; e < 4; ++e) gs[e] += red[w][i][lane][e], us[e] += red[w][NT + i][lane][e];
      float hv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float gt = __bfloat162float(__float2bfloat16_rn(gs[e] * kg));
        const float ut = __bfloat162float(__float2bfloat16_rn(us[e] * ku));
        hv[e] = silu_mul(gt, ut);
      }
      const int ch = 8 * (st * NT + i) + 2 * t;
      if (g < M)
        *reinterpret_cast<__nv_bfloat162*>(a.h + ((size_t)slot * M + g) * kI + ch) = __floats2bfloat162_rn(hv[0], hv[1]);
      if (R16 && g + 8 < M)
        *reinterpret_cast<__nv_bfloat162*>(a.h + ((size_t)slot * M + g + 8) * kI + ch) = __floats2bfloat162_rn(hv[2], hv[3]);
    }
    __syncthreads();
    task = a.ctr ? s_next[it & 1] : task + (int)gridDim.x;
  }
  if (a.ctr && threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(a.ctr + 1, 1) == (int)gridDim.x - 1) atomicExch(a.ctr, 0), atomicExch(a.ctr + 1, 0);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// k_down. A warp owns NT output strips (8 channels each): the shared expert's down projection for them (FP8, its
// weights loaded before griddepcontrol.wait, since they need nothing upstream) into ys, then every routed slot
// in expert-id order, P slots in flight, acc += (s2 * w_row) * c into yr. One wave: kStrips / NT / 4 CTAs.
template <int NT, int P, bool R16>
__global__ void __launch_bounds__(128) k_down(Params a) {
  __shared__ Union<true> U;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int M = a.M;
  const int r0 = min(g, M - 1), r1 = min(g + 8, M - 1);
  const int s0 = (blockIdx.x * 4 + warp) * NT;
  Raw8 sb[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const uint8_t* p = a.L.sh.d + ((size_t)(s0 + i) * 4) * 1024 + 32 * lane;
#pragma unroll
    for (int u = 0; u < 4; ++u) sb[i][u].q0 = ldg_stream128(p + u * 1024), sb[i][u].q1 = ldg_stream128(p + u * 1024 + 16);
  }
  // the union from the router's output (complete: k_up passed its wait before this grid launched)
  build_union(U, a);
  const int n = U.n;
  Raw rb[P][NT];
  auto load = [&](int s, int u) {
    const MatDesc& d = U.d[u];
    const int bs = d.fmt == kNVFP4 ? kQ4Blk : kS3Blk;
#pragma unroll
    for (int i = 0; i < NT; ++i) load_blk(rb[s][i], a.L.pool + d.off + (size_t)(s0 + i) * bs, d.fmt, lane);
  };
#pragma unroll
  for (int s = 0; s < P; ++s)
    if (s < n) load(s, s);
  pdl_wait();
  pdl_launch_dependents();
  // posted: the outputs go into the send buffer of the collective about to run (every earlier collective has
  // finished: this grid is past its wait, so the send buffer's last reader, three collectives back, is done)
  uint64_t seq = 0;
  float *yr = a.yr, *ys = a.ys;
  if (a.post) {
    seq = comm::next_seq(a.ch);
    yr = reinterpret_cast<float*>(comm::send_buf(a.ch, seq));
    ys = yr + (size_t)M * kH;
  }
  {
    const __nv_bfloat16* h0 = a.h + ((size_t)kShSlot * M + r0) * kI + 16 * t;
    const __nv_bfloat16* h1 = a.h + ((size_t)kShSlot * M + r1) * kI + 16 * t;
    float acc[NT][4], cu[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
#pragma unroll
      for (int i = 0; i < NT; ++i) cu[i][0] = cu[i][1] = cu[i][2] = cu[i][3] = 0.f;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        XF f;
        load_x<R16>(f, h0, h1, 128 * u + 64 * q);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          uint32_t o[8];
          dec8(q ? sb[i][u].q1 : sb[i][u].q0, o);
          mma4(cu[i], f, o);
        }
      }
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const float s8 = __ldg(a.L.sh.ds + ((s0 + i) / 16) * 4 + u);
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][e] = fmaf(s8, cu[i][e], acc[i][e]);
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int ch = 8 * (s0 + i) + 2 * t;
      if (g < M) *reinterpret_cast<float2*>(ys + (size_t)g * kH + ch) = make_float2(acc[i][0], acc[i][1]);
      if (R16 && g + 8 < M) *reinterpret_cast<float2*>(ys + (size_t)(g + 8) * kH + ch) = make_float2(acc[i][2], acc[i][3]);
    }
  }
  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  for (int u = 0; u < n; u += P) {
#pragma unroll
    for (int s = 0; s < P; ++s) {
      const int uu = u + s;
      if (uu >= n) break;
      const uint32_t fmt = U.d[uu].fmt;
      uint32_t tab[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) tab[j] = U.d[uu].tab[j];
      const __nv_bfloat16* h0 = a.h + ((size_t)uu * M + r0) * kI + 16 * t;
      const __nv_bfloat16* h1 = a.h + ((size_t)uu * M + r1) * kI + 16 * t;
      float c[NT][4];
#pragma unroll
      for (int i = 0; i < NT; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          XF f;
          load_x<R16>(f, h0, h1, 128 * jj + 64 * q);
#pragma unroll
          for (int i = 0; i < NT; ++i) {
            uint32_t o[8];
            dec_group(rb[s][i], jj, q, fmt, tab, o);
            mma4(c[i], f, o);
          }
        }
      if (uu + P < n) load(s, uu + P);
      const float k = U.d[uu].s2 * fmt_k(fmt);
      const float k0 = k * U.w[uu][g], k1 = R16 ? k * U.w[uu][g + 8] : 0.f;
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        acc[i][0] = fmaf(k0, c[i][0], acc[i][0]);
        acc[i][1] = fmaf(k0, c[i][1], acc[i][1]);
        if (R16) {
          acc[i][2] = fmaf(k1, c[i][2], acc[i][2]);
          acc[i][3] = fmaf(k1, c[i][3], acc[i][3]);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int ch = 8 * (s0 + i) + 2 * t;
    if (g < M) *reinterpret_cast<float2*>(yr + (size_t)g * kH + ch) = make_float2(acc[i][0], acc[i][1]);
    if (R16 && g + 8 < M) *reinterpret_cast<float2*>(yr + (size_t)(g + 8) * kH + ch) = make_float2(acc[i][2], acc[i][3]);
  }
  if (a.post) comm::post_grid(a.ch, seq, (uint32_t)((size_t)2 * M * kH * 4));
}

__global__ void k_finalize(const float* __restrict__ R, const float* __restrict__ S, int n, __nv_bfloat16* __restrict__ out) {
  pdl_launch_dependents();
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float r = __bfloat162float(__float2bfloat16_rn(R[i])), s = __bfloat162float(__float2bfloat16_rn(S[i]));
  out[i] = __float2bfloat16_rn(r + s);
}

template <class... KArgs, class... Args>
void launch(void (*kernel)(KArgs...), int grid, int block, cudaStream_t st, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

int num_sms() {
  static int n = 0;
  if (!n) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  }
  return n;
}

// the variants: up (NT, WK, P, CTAs an SM), down (NT, P)
struct UpV { int nt, wk, p; const char* name; };
constexpr UpV kUp[] = {
    {1, 6, 2, "nt1 wk6 p2"}, {1, 4, 2, "nt1 wk4 p2"}, {1, 4, 1, "nt1 wk4 p1"},
    {2, 4, 1, "nt2 wk4 p1"}, {1, 2, 2, "nt1 wk2 p2"}, {2, 2, 1, "nt2 wk2 p1"},
};
struct DownV { int nt, p; const char* name; };
constexpr DownV kDown[] = {{2, 1, "nt2 p1"}, {1, 2, "nt1 p2"}, {1, 3, "nt1 p3"}, {2, 2, "nt2 p2"}, {1, 4, "nt1 p4"}};

// cps <= 0: as many CTAs an SM as fit (the static task order assumes every CTA is resident)
template <class K>
int fit(K kernel, int threads, int cps) {
  int n = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, 0));
  return cps > 0 && cps < n ? cps : n;
}
template <int NT, int WK, int P>
void up_launch(const Params& p, int cps, cudaStream_t st, bool pdl) {
  if (p.M > 8) launch(k_up<NT, WK, P, true>, num_sms() * fit(k_up<NT, WK, P, true>, 32 * WK, cps), 32 * WK, st, pdl, p);
  else launch(k_up<NT, WK, P, false>, num_sms() * fit(k_up<NT, WK, P, false>, 32 * WK, cps), 32 * WK, st, pdl, p);
}
template <int NT, int P>
void down_launch(const Params& p, cudaStream_t st, bool pdl) {
  const int grid = kStrips / NT / 4;
  if (p.M > 8) launch(k_down<NT, P, true>, grid, 128, st, pdl, p);
  else launch(k_down<NT, P, false>, grid, 128, st, pdl, p);
}

}  // namespace

size_t decode_ws_bytes() { return kWsBytes; }
int decode_up_variants() { return (int)(sizeof kUp / sizeof kUp[0]); }
int decode_down_variants() { return (int)(sizeof kDown / sizeof kDown[0]); }
const char* decode_up_name(int v) { return kUp[v].name; }
const char* decode_down_name(int v) { return kDown[v].name; }

void moe_decode(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int M, void* ws, float* yr,
                float* ys, cudaStream_t st, const DecodeCfg& cfg) {
  if (M < 1 || M > kMaxRows) die("moe_decode: M=%d", M);
  if (!cfg.logits != !cfg.bias || (!cfg.logits && (!idx || !w))) die("moe_decode: idx and w, or logits and bias");
  Params p{x, M, idx, w, cfg.logits, cfg.bias, L, reinterpret_cast<__nv_bfloat16*>(ws), yr, ys,
           cfg.dyn ? reinterpret_cast<int*>(static_cast<uint8_t*>(ws) + kWsH) : nullptr, cfg.post != nullptr, {}};
  if (cfg.post) {
    p.ch = *cfg.post;
    if ((size_t)2 * M * kH * 4 > p.ch.cap) die("moe_decode: %d rows' two partials exceed the channel (%u B)", M, p.ch.cap);
    if (!(cfg.parts & 2)) die("moe_decode: posted outputs need the down kernel");
  }
  const int cps = cfg.up_ctas_per_sm;
  if (cfg.parts & 1) switch (cfg.up) {
    case 0: up_launch<1, 6, 2>(p, cps, st, cfg.pdl); break;
    case 1: up_launch<1, 4, 2>(p, cps, st, cfg.pdl); break;
    case 2: up_launch<1, 4, 1>(p, cps, st, cfg.pdl); break;
    case 3: up_launch<2, 4, 1>(p, cps, st, cfg.pdl); break;
    case 4: up_launch<1, 2, 2>(p, cps, st, cfg.pdl); break;
    case 5: up_launch<2, 2, 1>(p, cps, st, cfg.pdl); break;
    default: die("moe_decode: up variant %d", cfg.up);
  }
  if (cfg.parts & 2) switch (cfg.down) {
    case 0: down_launch<2, 1>(p, st, cfg.pdl); break;
    case 1: down_launch<1, 2>(p, st, cfg.pdl); break;
    case 2: down_launch<1, 3>(p, st, cfg.pdl); break;
    case 3: down_launch<2, 2>(p, st, cfg.pdl); break;
    case 4: down_launch<1, 4>(p, st, cfg.pdl); break;
    default: die("moe_decode: down variant %d", cfg.down);
  }
}

void finalize(const float* R, const float* S, int M, __nv_bfloat16* out, cudaStream_t st, bool pdl) {
  const int n = M * kH;
  launch(k_finalize, (n + 255) / 256, 256, st, pdl, R, S, n, out);
}

}  // namespace gx
