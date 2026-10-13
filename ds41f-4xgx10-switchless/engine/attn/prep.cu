// q_norm and the window KV write (kv_norm -> RoPE -> fp8 -> ring), and the fused decode prep.
#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/compressor.cuh"
#include "attn/gemv_row.cuh"
#include "attn/launch.h"
#include "common/check.h"

namespace attn {
namespace {

// RMSNorm as model.py: fp32 sum of squares, mean = sum * fp32(1/n), rsqrt, y = bf16(w * (x * r)).
// Lane l holds elements l + 32j of v and w (w: the weights as floats).
template <int N>
__device__ __forceinline__ void rmsnorm_warp(float (&v)[N / 32], const float (&w)[N / 32]) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < N / 32; ++j) ss = __fadd_rn(ss, __fmul_rn(v[j], v[j]));
  ss = warp_sum(ss);
  float r = rsqrtf(__fadd_rn(__fmul_rn(ss, 1.0f / N), kNormEps));
#pragma unroll
  for (int j = 0; j < N / 32; ++j) v[j] = bfround(__fmul_rn(w[j], __fmul_rn(v[j], r)));
}

// one warp per row; lane l owns elements l + 32*j (strided, so loads coalesce)
__global__ void q_norm_kernel(const uint16_t* qa, const uint16_t* w, uint16_t* qr, int rows) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  constexpr int N = kQLora;
  float v[N / 32], wf[N / 32];
  const uint16_t* x = qa + (size_t)row * N;
#pragma unroll
  for (int j = 0; j < N / 32; ++j) {
    v[j] = bf2f(x[lane + 32 * j]);
    wf[j] = bf2f(w[lane + 32 * j]);
  }
  rmsnorm_warp<N>(v, wf);
  uint16_t* y = qr + (size_t)row * N;
#pragma unroll
  for (int j = 0; j < N / 32; ++j) y[lane + 32 * j] = f2bf(v[j]);
}

// One row's window KV by one warp; lane l owns elements [16l, 16l+16): half an fp8 block, and lanes 28..31
// the RoPE dims. nw: the lane's 16 kv_norm weights.
__device__ __forceinline__ void kv_write_warp(const uint16_t* src_row, const float (&nw)[16], int p,
                                              const float2* table, const Window& win, uint16_t* kv_out_row,
                                              int lane) {
  float v[16];
  const uint4* src = reinterpret_cast<const uint4*>(src_row + lane * 16);
  uint4 a = src[0], b = src[1];
  const uint32_t wds[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    v[2 * j] = __uint_as_float(wds[j] << 16);
    v[2 * j + 1] = __uint_as_float(wds[j] & 0xffff0000u);
  }
  // kv_norm
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) ss = __fadd_rn(ss, __fmul_rn(v[j], v[j]));
  ss = warp_sum(ss);
  float r = rsqrtf(__fadd_rn(__fmul_rn(ss, 1.0f / kHead), kNormEps));
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = bfround(__fmul_rn(nw[j], __fmul_rn(v[j], r)));
  // RoPE on dims 448..511
  if (lane >= 28) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float2 cs = table[(size_t)p * 32 + (lane - 28) * 8 + j];
      rope_pair(v[2 * j], v[2 * j + 1], cs.x, cs.y, false);
      v[2 * j] = bfround(v[2 * j]);
      v[2 * j + 1] = bfround(v[2 * j + 1]);
    }
  }
  // fp8 per 32: lanes 2b, 2b+1 share block b
  float am = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) am = fmaxf(am, fabsf(v[j]));
  am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
  float sc = fp8_block_scale(am);
  int slot = (p - win.base) & win.mask;
  uint32_t q[4] = {0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < 16; ++j) q[j / 4] |= (uint32_t)fp8_code(v[j], sc) << (8 * (j & 3));
  *reinterpret_cast<uint4*>(win.kv + (size_t)slot * kHead + lane * 16) = make_uint4(q[0], q[1], q[2], q[3]);
  if ((lane & 1) == 0) win.sc[slot * 16 + lane / 2] = f_to_ue8m0(sc);
  if (kv_out_row) {
    uint32_t o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float d0 = __fmul_rn(e4m3_to_f((uint8_t)(q[(2 * j) / 4] >> (8 * ((2 * j) & 3)))), sc);
      float d1 = __fmul_rn(e4m3_to_f((uint8_t)(q[(2 * j + 1) / 4] >> (8 * ((2 * j + 1) & 3)))), sc);
      o[j] = (uint32_t)f2bf(d0) | ((uint32_t)f2bf(d1) << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(kv_out_row + lane * 16);
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// one warp per row
__global__ void kv_write_kernel(const uint16_t* kv_pre, const uint16_t* norm_w, const int* pos, const float2* table,
                                Window win, uint16_t* kv_out, int rows) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  float nw[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) nw[j] = bf2f(norm_w[lane * 16 + j]);
  kv_write_warp(kv_pre + (size_t)row * kHead, nw, pos[row], table, win_at(win, row),
                kv_out ? kv_out + (size_t)row * kHead : nullptr, lane);
}

// DSpark's window writes: set blockIdx.x, warp r: row r < *count at position pos[r] (pos null: *pos0 + r;
// negative: none), into its slot's window (norm weights load before the PDL wait)
__global__ void kv_write_sets_kernel(KvSets a, const int* count, const int* pos0, const int* pos, const float2* table) {
  pdl_trigger();
  const KvSet& k = a.s[blockIdx.x];
  int r = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float nw[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) nw[j] = bf2f(k.norm_w[lane * 16 + j]);
  pdl_wait();
  if (r >= *count) return;
  const int p = pos ? pos[r] : *pos0 + r;
  if (p < 0) return;
  kv_write_warp(k.kv_pre + (size_t)r * kHead, nw, p, table, win_at(k.win, r), nullptr, lane);
}

// ---- the fused decode path ----------------------------------------------------------------------
// The rest of a decode step's preparation in one launch, by CTA: [0, rows): q_norm with act_quant (warp 0)
// and kv_write (warp 1) of row b; [rows, 2 rows): compressor emit and carry of row b - rows (kv-source
// layers); then 8 CTAs of 4 warps for the indexer's 32 head weights (index layers). Norm and projection
// weights load before the PDL wait. k16: 9..16 rows, the head weights in two groups (rows 0..7, then the rest).
template <bool k16>
__global__ void __launch_bounds__(128) decode_prep_kernel(DecodePrep a, int rows) {
  pdl_trigger();
  int b = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (b < rows) {
    if (warp == 0) {
      constexpr int N = kQLora / 32;
      float w[N], v[N];
#pragma unroll
      for (int j = 0; j < N; ++j) w[j] = bf2f(a.q_norm[lane + 32 * j]);
      pdl_wait();
      const uint16_t* x = a.qa + (size_t)b * kQLora;
#pragma unroll
      for (int j = 0; j < N; ++j) v[j] = bf2f(x[lane + 32 * j]);
      rmsnorm_warp<kQLora>(v, w);
      float am[N];  // act_quant: block j is element j of every lane; all blocks' warp_max at once
#pragma unroll
      for (int j = 0; j < N; ++j) am[j] = fabsf(v[j]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int j = 0; j < N; ++j) am[j] = fmaxf(am[j], __shfl_xor_sync(0xffffffffu, am[j], o));
#pragma unroll
      for (int j = 0; j < N; ++j) {
        if (a.qr) a.qr[(size_t)b * kQLora + lane + 32 * j] = f2bf(v[j]);
        float s = fp8_block_scale(am[j]);
        a.qq[(size_t)b * kQLora + 32 * j + lane] = fp8_code(v[j], s);
        if (lane == 0) a.qs[(size_t)b * N + j] = f_to_ue8m0(s);
      }
    } else if (warp == 1) {
      float nw[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) nw[j] = bf2f(a.kv_norm[lane * 16 + j]);
      pdl_wait();
      kv_write_warp(a.kv_pre + (size_t)b * kHead, nw, a.pos[b], a.table, win_at(a.win, b), nullptr, lane);
    } else {
      pdl_wait();
    }
    return;
  }
  b -= rows;
  if (a.ckv_proj) {
    if (b < rows) {  // the index key projection's weights head for L2 during the wait
      for (int i = threadIdx.x; i < kIdxDim * kHead * 2 / 128; i += blockDim.x)
        prefetch_l2(reinterpret_cast<const uint8_t*>(a.cw.wk) + 128 * i);
      pdl_wait();
      const CompressorCarry carry = carry_at(a.carry, b);
      emit_row(b, b > 0 && same_session(a.carry.slot, b - 1, b), a.ckv_proj, a.cscore, a.pos, a.cw, carry, a.table,
               ckv_at(a.ckv, b), ik_at(a.ik, b), nullptr, 0);
      if (a.cw.ratio > 1) carry_row(b, a.ckv_proj, a.cscore, a.pos, carry);
      return;
    }
    b -= rows;
  }
  int o = 4 * b + warp;  // head weight o = bf16(x . wproj[o]), as gemv_bf16
  uint4 wv[kRowWords];
  gemv_row_load(a.wproj + (size_t)o * kDim, lane, wv);
  pdl_wait();
  float acc[8];
  gemv_row_dot(wv, a.x, k16 ? kDecodeGroup : rows, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 8; ++r)
      if (r < rows) a.wp[r * kIdxHeads + o] = bfround(acc[r]);
  }
  if (k16) {
    gemv_row_dot(wv, a.x + (size_t)kDecodeGroup * kDim, rows - kDecodeGroup, lane, acc);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < 8; ++r)
        if (r < rows - kDecodeGroup) a.wp[(kDecodeGroup + r) * kIdxHeads + o] = bfround(acc[r]);
    }
  }
}

}  // namespace

void q_norm(const uint16_t* qa, const uint16_t* w, uint16_t* qr, int rows, cudaStream_t s) {
  q_norm_kernel<<<(rows + 3) / 4, 128, 0, s>>>(qa, w, qr, rows);
  CUDA_CHECK(cudaGetLastError());
}

void kv_write(const uint16_t* kv_pre, const uint16_t* norm_w, const int* pos, const float2* table, Window win,
              uint16_t* kv_out, int rows, cudaStream_t s) {
  check_slots(win, "kv_write");
  kv_write_kernel<<<(rows + 3) / 4, 128, 0, s>>>(kv_pre, norm_w, pos, table, win, kv_out, rows);
  CUDA_CHECK(cudaGetLastError());
}

void kv_write_sets(const KvSets& sets, int max_rows, const int* count, const int* pos0, const float2* table, bool pdl,
                   cudaStream_t s, const int* pos) {
  if (sets.n < 1 || sets.n > kMaxKvSets || max_rows < 1 || max_rows > 32 || !(pos0 || pos))
    die("kv_write_sets: %d x %d", sets.n, max_rows);
  for (int i = 0; i < sets.n; ++i) check_slots(sets.s[i].win, "kv_write_sets");
  launch(kv_write_sets_kernel, dim3(sets.n), dim3(32 * max_rows), 0, s, pdl, sets, count, pos0, pos, table);
}

void decode_prep(const DecodePrep& a, int rows, bool pdl, cudaStream_t s) {
  if (rows < 1 || rows > kMaxDecodeRows) die("decode_prep: %d rows", rows);
  // At most kDecodeGroup rows a session (attn.h DecodePrep: 16 would race on the carry). Without a table, the rows
  // are one session; with one, its builder checks.
  if (rows > kDecodeGroup && !(a.win.slot || a.carry.slot || a.ckv.slot || a.ik.slot))
    die("decode_prep: %d rows of one session, at most %d", rows, kDecodeGroup);
  check_slots(a.win, "decode_prep");
  check_slots(a.carry, "decode_prep");
  check_slots(a.ckv, "decode_prep");
  check_slots(a.ik, "decode_prep");
  int grid = rows + (a.ckv_proj ? rows : 0) + (a.wproj ? kIdxHeads / 4 : 0);
  launch(rows > kDecodeGroup ? decode_prep_kernel<true> : decode_prep_kernel<false>, dim3(grid), dim3(128), 0, s, pdl,
         a, rows);
}

}  // namespace attn
