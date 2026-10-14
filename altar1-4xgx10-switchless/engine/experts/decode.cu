// The decode expert step for 1-16 rows (docs/experts.md, "Decode"): two PDL kernels, templated on the layer's
// expert format F (FmtINT4, FmtBF16, FmtNVFP4: formats.cuh), chosen by the host from LayerW::fmt.
//
// k_up: a persistent grid walks tasks in a fixed order: a fused FP8 shared expert's if the layer has one (gate
// and up strips of 8 channels: they need nothing from the router, so the first wave loads them before
// griddepcontrol.wait; altar-1 has none), then for every expert of the union (the rows' picks, each read once,
// in expert-id order) its gate and up strips. A task is one CTA: WK warps split K, each streams its blocks into
// bf16 B fragments of m16n8k16 MMAs whose A rows are the step's rows (rows 8..15 in the upper half from 9 rows
// on); the partials meet in shared memory and one warp a strip pair applies the per-tensor scales, rounds gate
// and up to bf16, SwiGLU, and writes h (bf16) to the slot's workspace. A warp's K range and MMA order do not
// depend on the format (a BF16 block is 128 k, an INT4 one 512: the same 1,024 k a warp at WK 6), so INT4's
// sums equal BF16's on the dequantised matrix bit for bit.
//
// k_down: warps of NT output strips, every slot of the union in expert-id order, acc += (s2 * w_row) * c, the
// routing weight 0 for rows that did not pick the expert, so a row's bits do not depend on the other rows;
// with a fused shared expert each warp first does its down projection for its strips (weights loaded before
// the PDL wait) into ys. Both outputs f32 partial sums. Without one, ys is the caller's (the BF16 shared
// expert's f32 output, made before this step) and k_down only forwards it into a posted collective's buffer.
#include <cstdio>

#ifndef KUP_XPF
#define KUP_XPF true
#endif

#include "comm/ring_dev.cuh"
#include "common/check.h"
#include "experts/formats.cuh"

namespace gx {

namespace {

constexpr int kStrips = kH / 8;  // 768 strips of down
constexpr int kShU = kH / 128;     // 48 FP8 units of the shared gate/up along K

// workspace: h [kMaxUnion + 1][kMaxRows][kI] bf16, then k_up's task counters (claimed, CTAs done)
constexpr size_t kWsH = (size_t)(kMaxUnion + 1) * kMaxRows * kI * 2;
constexpr size_t kWsBytes = kWsH + 256;

struct Params {
  const __nv_bfloat16* x;
  bool sh;              // a fused FP8 shared expert (LayerW::sh.fused())
  int M;
  const int* idx;
  const float* w;
  const float* logits;  // fused routing (DecodeCfg::logits): idx, w unused
  const float* bias;
  LayerW L;
  __nv_bfloat16* h;  // [slot][M][kI]
  float* yr;
  float* ys;         // fused shared: written; else read (forwarded when posting)
  int* ctr;          // k_up's task queue (null: static order); zero between launches
  bool post;         // k_down: yr, ys into ch's send buffer, posted (DecodeCfg::post)
  comm::DevChannel ch;
};

// The union of the rows' picks, sorted by expert id. All threads of the CTA (>= 128). W: also each slot's
// per-row routing weights (0 where the row did not pick it) and the down matrices' descriptors.
template <bool W>
struct Union {
  int n;
  uint32_t mask[kEWords];
  int pre[kEWords];
  int ri[kMaxRows * kTopK];    // fused routing: the rows' picks and weights
  float rw[kMaxRows * kTopK];
  int id[kMaxUnion];
  float w[W ? kMaxUnion : 1][kMaxRows];
  MatDesc d[W ? kMaxUnion : 1];
};
template <bool W>
__device__ void build_union(Union<W>& U, const Params& a) {
  const int tid = threadIdx.x, ne = a.M * kTopK;
  if (tid < kEWords) U.mask[tid] = 0;
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
    const int c = tid < kEWords ? __popc(U.mask[tid]) : 0;
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    if (tid < kEWords) U.pre[tid] = incl - c;
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

// k_up: load the next routed task's first blocks under the current task's epilogue (bit-neutral; docs/engine.md, "Tuning")
constexpr bool kUpXpf = KUP_XPF;

__device__ __forceinline__ float bf_round(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }

__device__ __forceinline__ float silu_mul(float g, float u) {  // h = bf16(bf16(silu(g)) * u), g and u bf16 values
  const float s = __bfloat162float(__float2bfloat16_rn(g / (1.f + expf(-g))));
  return s * u;
}

// ---------------------------------------------------------------------------------------------------------------
// k_up. NT strips of gate and of up a task (64 / NT tasks an expert), WK warps along K, P blocks in flight a warp.
template <class F, int NT, int WK, int P, bool R16, bool SHB>
__global__ void __launch_bounds__(32 * WK) k_up(Params a) {
  constexpr int NTL = 2 * NT, TPE = 64 / NT, PS = 2;  // tiles a task, tasks an expert, FP8 units in flight
  constexpr int kUpB = kH / F::kBK;                   // blocks of gate/up along K (12 INT4/NVFP4, 48 BF16)
  using Raw = typename F::Raw;
  // The BF16 shared expert (a.shb): a task is BT 8-row tiles of the interleaved gate/up, each tile's K in kShParts
  // parts of BC 64-k chunks as attn's gemv at split-K 4; warp w takes BU consecutive parts of one tile (BT * 4 = BU *
  // WK), BPS chunks in flight a lane.
  constexpr int BT = WK % 4 == 0 ? WK / 4 : WK % 2 == 0 ? WK / 2 : WK, BU = kShParts * BT / WK;
  constexpr int BC = kH / 64 / kShParts, BPS = 8;
  static_assert(BU * WK == kShParts * BT && BC % BPS == 0, "the BF16 shared expert's split");
  static_assert(WK == 1 || WK * 2 * NT >= kShParts * BT, "red holds the BF16 shared expert's partials");
  const int nsh = a.sh ? TPE : SHB ? (kShTiles + BT - 1) / BT : 0;  // the fused shared expert's tasks
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
  // the BF16 shared expert's place in a task: tile task * BT + bto, parts bp0 .. bp0 + BU - 1
  const int bto = BU * warp / kShParts, bp0 = BU * warp % kShParts;
  auto bptr = [&](int tk) {
    return a.L.sh.bgu + ((size_t)(tk * BT + bto) * (kH / 64) + bp0 * BC) * 1024 + 16 * lane;
  };
  int task = blockIdx.x;
  bool pre = task < nsh;
  if (pre && !SHB)  // (SHB: no FP8 shared expert; moe_decode refuses both)
#pragma unroll
    for (int s = 0; s < PS; ++s)
      if (s < nu) sh_load(task, s, s);
  if (pre && SHB && task * BT + bto < kShTiles) {  // the warp's BU parts (contiguous) into L2 before the wait
    const uint8_t* p = bptr(task) - 16 * lane;
#pragma unroll
    for (int i = 0; i < BU * BC * 1024 / 32 / 128; ++i)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p + 128 * (lane + 32 * i)));
  }
  pdl_wait();
  build_union(U, a);
  pdl_launch_dependents();
  const int ntask = nsh + TPE * U.n;
  // the routed tasks' blocks: warp's K range [b0, b0 + nb) of the 12 (48) blocks; rb the P blocks in flight. With
  // XPF (kUpXpf, WK > 1) the next routed task's first P blocks are loaded under the current task's epilogue (pf: that
  // task); the loads, their order and the MMAs are the same, so the same bits.
  constexpr bool XPF = kUpXpf && WK > 1;
  const int b0 = kUpB * warp / WK, nb = kUpB * (warp + 1) / WK - b0;
  Raw rb[P][NTL];
  int pf = -1;
  auto rbase = [&](int tk, const uint8_t* (&bs)[NTL]) {
    const int sl = (tk - nsh) / TPE, sg = (tk - nsh) % TPE;
    const MatDesc* d = a.L.desc + U.id[sl] * 3;
#pragma unroll
    for (int i = 0; i < NTL; ++i) bs[i] = a.L.pool + __ldg(&d[i / NT].off) + (size_t)(sg * NT + i % NT) * kUpB * F::kBlk;
  };

  // The first wave is static (CTA b: task b, so the shared expert's tasks load before the wait); then, with a
  // queue, each CTA claims the next task when it starts one, so CTAs that finish early take more.
#pragma unroll 1
  for (int it = 0; task < ntask; ++it) {
    if (a.ctr && threadIdx.x == 0) s_next[it & 1] = (int)gridDim.x + atomicAdd(a.ctr, 1);
    if constexpr (SHB) if (task < nsh) {
      // the BF16 shared expert: attn's gemv(kSwiGLU, split-K 4) for BT tiles, bit for bit
      const bool bv = task * BT + bto < kShTiles;
      const uint8_t* bp = bptr(task);
      uint4 bb[BPS][2];  // BPS chunks in flight a lane (only here: no registers held across the routed tasks)
      auto bload = [&](int s, int c) {
        bb[s][0] = ldg_stream128(bp + (size_t)c * 1024);
        bb[s][1] = ldg_stream128(bp + (size_t)c * 1024 + 512);
      };
      if (bv)
#pragma unroll
        for (int s = 0; s < BPS; ++s) bload(s, s);
      float cs[BU][4];
#pragma unroll
      for (int u = 0; u < BU; ++u) cs[u][0] = cs[u][1] = cs[u][2] = cs[u][3] = 0.f;
      if (bv) {
#pragma unroll
        for (int u = 0; u < BU; ++u)
#pragma unroll 1
          for (int c0 = 0; c0 < BC; c0 += BPS)
#pragma unroll
            for (int s = 0; s < BPS; ++s) {
              const int cc = u * BC + c0 + s;
              const uint32_t o[8] = {bb[s][0].x, bb[s][0].y, bb[s][0].z, bb[s][0].w,
                                     bb[s][1].x, bb[s][1].y, bb[s][1].z, bb[s][1].w};
              if (cc + BPS < BU * BC) bload(s, cc + BPS);
              XF f;
              load_x<R16>(f, x0, x1, 64 * (bp0 * BC + cc));
              mma4(cs[u], f, o);
            }
      }
      // the parts added in part order (gemv's finish), then its SwiGLU epilogue
      auto out = [&](int tile, const float (&acc)[4]) {
        const int col = 4 * tile + t;  // the intermediate of columns 2t (gate) and 2t + 1 (up)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r = g + 8 * hh;
          if (hh ? !(R16 && r < M) : r >= M) continue;
          const float gt = bf_round(acc[2 * hh]), up = bf_round(acc[2 * hh + 1]);
          const float sl = bf_round(__fdiv_rn(gt, __fadd_rn(1.f, expf(-gt))));
          a.h[((size_t)kShSlot * M + r) * kI + col] = __float2bfloat16_rn(__fmul_rn(sl, up));
        }
      };
      if constexpr (WK == 1) {
        float acc[4] = {cs[0][0], cs[0][1], cs[0][2], cs[0][3]};
#pragma unroll
        for (int k = 1; k < kShParts; ++k)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[e] = __fadd_rn(acc[e], cs[k][e]);
        if (bv) out(task * BT + bto, acc);
      } else {
        float(*rp)[32][4] = reinterpret_cast<float(*)[32][4]>(&red[0][0][0][0]);
#pragma unroll
        for (int u = 0; u < BU; ++u)
          *reinterpret_cast<float4*>(&rp[BU * warp + u][lane][0]) = make_float4(cs[u][0], cs[u][1], cs[u][2], cs[u][3]);
        __syncthreads();
        const int tile = task * BT + warp;
        if (warp < BT && tile < kShTiles) {
          float acc[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[e] = rp[kShParts * warp][lane][e];
#pragma unroll
          for (int k = 1; k < kShParts; ++k)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[e] = __fadd_rn(acc[e], rp[kShParts * warp + k][lane][e]);
          out(tile, acc);
        }
      }
      pre = false;
      __syncthreads();
      task = a.ctr ? s_next[it & 1] : task + (int)gridDim.x;
      continue;
    }
    float c[NTL][4];
#pragma unroll
    for (int i = 0; i < NTL; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
    const bool sh = task < nsh;
    const int slot = sh ? kShSlot : (task - nsh) / TPE, st = sh ? task : (task - nsh) % TPE;  // st: strip group
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
      if (__ldg(&dg[0].fmt) != F::kFmt || __ldg(&dg[1].fmt) != F::kFmt) __trap();  // the layer's format, every matrix
      const uint8_t* base[NTL];
      rbase(task, base);
      auto load = [&](int s, int b) {
#pragma unroll
        for (int i = 0; i < NTL; ++i) F::load(rb[s][i], base[i] + (size_t)(b0 + b) * F::kBlk, lane);
      };
      if (pf != task)
#pragma unroll
        for (int s = 0; s < P; ++s)
          if (s < nb) load(s, s);
      pf = -1;
      for (int b = 0; b < nb; b += P) {
#pragma unroll
        for (int s = 0; s < P; ++s) {
          if (b + s >= nb) break;
          const int k = F::kBK * (b0 + b + s);
#pragma unroll
          for (int jj = 0; jj < F::kU; ++jj)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
              XF f;
              load_x<R16>(f, x0, x1, k + 128 * jj + 64 * q);
#pragma unroll
              for (int i = 0; i < NTL; ++i) {
                uint32_t o[8];
                F::dec(rb[s][i], jj, q, o);
                mma4(c[i], f, o);
              }
            }
          if (b + s + P < nb) load(s, b + s + P);
        }
      }
    }
    pre = false;

    // partials -> shared memory; warp i % WK finishes strip pair i (gate tile i, up tile NT + i)
    if (WK > 1) {
#pragma unroll
      for (int i = 0; i < NTL; ++i)
        *reinterpret_cast<float4*>(&red[warp][i][lane][0]) = make_float4(c[i][0], c[i][1], c[i][2], c[i][3]);
      __syncthreads();
      if constexpr (XPF) {  // the next routed task's first blocks, loaded now, under this task's epilogue
        const int nx = a.ctr ? s_next[it & 1] : task + (int)gridDim.x;
        if (nx < ntask && nx >= nsh) {
          const uint8_t* nbase[NTL];
          rbase(nx, nbase);
#pragma unroll
          for (int s = 0; s < P; ++s)
            if (s < nb)
#pragma unroll
              for (int i = 0; i < NTL; ++i) F::load(rb[s][i], nbase[i] + (size_t)(b0 + s) * F::kBlk, lane);
          pf = nx;
        }
      }
    }
    float kg = 1.f, ku = 1.f;
    if (!sh) {
      const MatDesc* dg = a.L.desc + U.id[slot] * 3;
      kg = __ldg(&dg[0].s2) * F::kMul;
      ku = __ldg(&dg[1].s2) * F::kMul;
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      if (i % WK != warp) continue;
      float gs[4], us[4];
      if (WK == 1) {  // one warp summed all of K, in prefill's order: its sums as they are (no 0 + c, which
                      // would turn a -0 into +0), so h equals moe_prefill's bit for bit
#pragma unroll
        for (int e = 0; e < 4; ++e) gs[e] = c[i][e], us[e] = c[NT + i][e];
      } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) gs[e] = us[e] = 0.f;
#pragma unroll
        for (int w = 0; w < WK; ++w)
#pragma unroll
          for (int e = 0; e < 4; ++e) gs[e] += red[w][i][lane][e], us[e] += red[w][NT + i][lane][e];
      }
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
// The routed part runs as one pipeline over (slot, block) steps: a strip's K = 512 is kDB blocks (1 INT4/NVFP4,
// 4 BF16); a slot's sums c restart at its first block and go into acc after its last, so the MMA order a slot is
// the same for every format.
template <class F, int NT, int P, bool R16, bool SHB>
__global__ void __launch_bounds__(128) k_down(Params a) {
  __shared__ Union<true> U;
  constexpr int kDB = kI / F::kBK;
  using Raw = typename F::Raw;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int M = a.M;
  const int r0 = min(g, M - 1), r1 = min(g + 8, M - 1);
  const int s0 = (blockIdx.x * 4 + warp) * NT;
  Raw8 sb[NT][4];
  if constexpr (SHB)  // the BF16 shared down's NT strips (8 KB each, contiguous) into L2 now, read after the wait
#pragma unroll
    for (int i = 0; i < 2 * NT; ++i)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(a.L.sh.bd + (size_t)s0 * 8192 + 128 * (lane + 32 * i)));
  if (a.sh)
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const uint8_t* p = a.L.sh.d + ((size_t)(s0 + i) * 4) * 1024 + 32 * lane;
#pragma unroll
      for (int u = 0; u < 4; ++u) sb[i][u].q0 = ldg_stream128(p + u * 1024), sb[i][u].q1 = ldg_stream128(p + u * 1024 + 16);
    }
  // the union from the router's output (complete: k_up passed its wait before this grid launched)
  build_union(U, a);
  const int n = U.n, nst = n * kDB;
  if (threadIdx.x < n && U.d[threadIdx.x].fmt != F::kFmt) __trap();  // the layer's format, every matrix
  Raw rb[P][NT];
  auto load = [&](int s, int u) {  // step u: slot u / kDB, block u % kDB of the strips
    const MatDesc& d = U.d[u / kDB];
#pragma unroll
    for (int i = 0; i < NT; ++i) F::load(rb[s][i], a.L.pool + d.off + ((size_t)(s0 + i) * kDB + u % kDB) * F::kBlk, lane);
  };
#pragma unroll
  for (int s = 0; s < P; ++s)
    if (s < nst) load(s, s);
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
  if (a.sh) {
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
  } else if (SHB) {  // the BF16 shared down: attn's gemv(kF32, split-K 1), one MMA chain over the 512 k a strip
    const __nv_bfloat16* h0 = a.h + ((size_t)kShSlot * M + r0) * kI + 16 * t;
    const __nv_bfloat16* h1 = a.h + ((size_t)kShSlot * M + r1) * kI + 16 * t;
#pragma unroll 1
    for (int i = 0; i < NT; ++i) {
      const uint8_t* p = a.L.sh.bd + (size_t)(s0 + i) * 8192 + 16 * lane;
      uint4 v[kI / 64][2];
#pragma unroll
      for (int j = 0; j < kI / 64; ++j) v[j][0] = ldg_stream128(p + 1024 * j), v[j][1] = ldg_stream128(p + 1024 * j + 512);
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < kI / 64; ++j) {
        const uint32_t o[8] = {v[j][0].x, v[j][0].y, v[j][0].z, v[j][0].w, v[j][1].x, v[j][1].y, v[j][1].z, v[j][1].w};
        XF f;
        load_x<R16>(f, h0, h1, 64 * j);
        mma4(acc, f, o);
      }
      const int ch = 8 * (s0 + i) + 2 * t;
      if (g < M) *reinterpret_cast<float2*>(ys + (size_t)g * kH + ch) = make_float2(acc[0], acc[1]);
      if (R16 && g + 8 < M) *reinterpret_cast<float2*>(ys + (size_t)(g + 8) * kH + ch) = make_float2(acc[2], acc[3]);
    }
  } else if (a.post) {  // the shared expert ran outside (its f32 output in a.ys): forward the warp's columns
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int ch = 8 * (s0 + i) + 2 * t;
      if (g < M) *reinterpret_cast<float2*>(ys + (size_t)g * kH + ch) = *reinterpret_cast<const float2*>(a.ys + (size_t)g * kH + ch);
      if (R16 && g + 8 < M)
        *reinterpret_cast<float2*>(ys + (size_t)(g + 8) * kH + ch) = *reinterpret_cast<const float2*>(a.ys + (size_t)(g + 8) * kH + ch);
    }
  }
  float acc[NT][4], c[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  for (int u = 0; u < nst; u += P) {
#pragma unroll
    for (int s = 0; s < P; ++s) {
      const int us = u + s, uu = us / kDB, bb = us % kDB;
      if (us >= nst) break;
      const __nv_bfloat16* h0 = a.h + ((size_t)uu * M + r0) * kI + 16 * t + F::kBK * bb;
      const __nv_bfloat16* h1 = a.h + ((size_t)uu * M + r1) * kI + 16 * t + F::kBK * bb;
      if (bb == 0)
#pragma unroll
        for (int i = 0; i < NT; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
#pragma unroll
      for (int jj = 0; jj < F::kU; ++jj)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          XF f;
          load_x<R16>(f, h0, h1, 128 * jj + 64 * q);
#pragma unroll
          for (int i = 0; i < NT; ++i) {
            uint32_t o[8];
            F::dec(rb[s][i], jj, q, o);
            mma4(c[i], f, o);
          }
        }
      if (us + P < nst) load(s, us + P);
      if (bb != kDB - 1) continue;
      const float k = U.d[uu].s2 * F::kMul;
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
    // one warp a task, K in prefill's order (kUpPrefillOrder = 6): decode = prefill bit for bit
    {1, 1, 2, "nt1 wk1 p2"}, {1, 1, 4, "nt1 wk1 p4"}, {2, 1, 2, "nt2 wk1 p2"},
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
template <class F, int NT, int WK, int P, bool SHB>
void up_launch2(const Params& p, int cps, cudaStream_t st, bool pdl) {
  if (p.M > 8)
    launch(k_up<F, NT, WK, P, true, SHB>, num_sms() * fit(k_up<F, NT, WK, P, true, SHB>, 32 * WK, cps), 32 * WK, st, pdl, p);
  else
    launch(k_up<F, NT, WK, P, false, SHB>, num_sms() * fit(k_up<F, NT, WK, P, false, SHB>, 32 * WK, cps), 32 * WK, st, pdl, p);
}
template <class F, int NT, int WK, int P>
void up_launch(const Params& p, int cps, cudaStream_t st, bool pdl) {
  if (p.L.sh.bf16()) up_launch2<F, NT, WK, P, true>(p, cps, st, pdl);
  else up_launch2<F, NT, WK, P, false>(p, cps, st, pdl);
}
template <class F, int NT, int P>
void down_launch(const Params& p, cudaStream_t st, bool pdl) {
  const int grid = kStrips / NT / 4;
  if (p.L.sh.bf16()) {
    if (p.M > 8) launch(k_down<F, NT, P, true, true>, grid, 128, st, pdl, p);
    else launch(k_down<F, NT, P, false, true>, grid, 128, st, pdl, p);
  } else {
    if (p.M > 8) launch(k_down<F, NT, P, true, false>, grid, 128, st, pdl, p);
    else launch(k_down<F, NT, P, false, false>, grid, 128, st, pdl, p);
  }
}

template <class F>
void decode_fmt(const Params& p, const DecodeCfg& cfg, cudaStream_t st) {
  const int cps = cfg.up_ctas_per_sm;
  if (cfg.parts & 1) switch (cfg.up) {
    case 0: up_launch<F, 1, 6, 2>(p, cps, st, cfg.pdl); break;
    case 1: up_launch<F, 1, 4, 2>(p, cps, st, cfg.pdl); break;
    case 2: up_launch<F, 1, 4, 1>(p, cps, st, cfg.pdl); break;
    case 3: up_launch<F, 2, 4, 1>(p, cps, st, cfg.pdl); break;
    case 4: up_launch<F, 1, 2, 2>(p, cps, st, cfg.pdl); break;
    case 5: up_launch<F, 2, 2, 1>(p, cps, st, cfg.pdl); break;
    case 6: up_launch<F, 1, 1, 2>(p, cps, st, cfg.pdl); break;
    case 7: up_launch<F, 1, 1, 4>(p, cps, st, cfg.pdl); break;
    case 8: up_launch<F, 2, 1, 2>(p, cps, st, cfg.pdl); break;
    default: die("moe_decode: up variant %d", cfg.up);
  }
  if (cfg.parts & 2) switch (cfg.down) {
    case 0: down_launch<F, 2, 1>(p, st, cfg.pdl); break;
    case 1: down_launch<F, 1, 2>(p, st, cfg.pdl); break;
    case 2: down_launch<F, 1, 3>(p, st, cfg.pdl); break;
    case 3: down_launch<F, 2, 2>(p, st, cfg.pdl); break;
    case 4: down_launch<F, 1, 4>(p, st, cfg.pdl); break;
    default: die("moe_decode: down variant %d", cfg.down);
  }
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
  Params p{x, L.sh.fused(), M, idx, w, cfg.logits, cfg.bias, L, reinterpret_cast<__nv_bfloat16*>(ws), yr, ys,
           cfg.dyn ? reinterpret_cast<int*>(static_cast<uint8_t*>(ws) + kWsH) : nullptr, cfg.post != nullptr, {}};
  if (L.sh.fused() && L.sh.bf16()) die("moe_decode: an FP8 and a BF16 shared expert");
  if (L.sh.bf16() && (!L.sh.bd || (!cfg.post && (cfg.parts & 2) && !ys)))
    die("moe_decode: the BF16 shared expert needs its down and ys");
  if (cfg.post) {
    p.ch = *cfg.post;
    if ((size_t)2 * M * kH * 4 > p.ch.cap) die("moe_decode: %d rows' two partials exceed the channel (%u B)", M, p.ch.cap);
    if (!(cfg.parts & 2)) die("moe_decode: posted outputs need the down kernel");
  }
  if (!L.sh.fused() && !L.sh.bf16() && (cfg.parts & 2) && cfg.post && !ys) die("moe_decode: posting without a fused shared expert needs its output in ys");
  switch (L.fmt) {
    case kINT4: decode_fmt<FmtINT4>(p, cfg, st); break;
    case kBF16: decode_fmt<FmtBF16>(p, cfg, st); break;
    case kNVFP4: decode_fmt<FmtNVFP4>(p, cfg, st); break;
    default: die("moe_decode: layer expert format %u is not NVFP4 (0), INT4 (2) or BF16 (3)", L.fmt);
  }
}

void finalize(const float* R, const float* S, int M, __nv_bfloat16* out, cudaStream_t st, bool pdl) {
  const int n = M * kH;
  launch(k_finalize, (n + 255) / 256, 256, st, pdl, R, S, n, out);
}

}  // namespace gx
