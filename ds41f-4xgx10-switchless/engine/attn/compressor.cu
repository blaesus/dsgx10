// The compressor of the kv-source layers: projections, pooling, norm, and the two things each
// completed entry produces (an index key for the indexer and an fp4 KV entry for attention).
#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/compressor.cuh"
#include "common/check.h"

namespace attn {
namespace {

// One 128-thread CTA per row (compressor.cuh emit_row), into the row's session's state.
__global__ void __launch_bounds__(128) emit_kernel(const float* kv, const float* score, const int* pos, int rows,
                                                   CompressorWeights w, CompressorCarry carry, const float2* table,
                                                   Compressed ckv, IndexKeys ik, uint16_t* dbg_latent, int dbg_base) {
  const int r = blockIdx.x;
  emit_row(r, r > 0 && same_session(carry.slot, r - 1, r), kv, score, pos, w, carry_at(carry, r), table, ckv_at(ckv, r),
           ik_at(ik, r), dbg_latent, dbg_base);
}

// keep the last (up to) 16 rows' ratio-2 projections, by position
__global__ void carry_kernel(const float* kv, const float* score, const int* pos, int rows, CompressorCarry carry) {
  int r = rows - 1 - blockIdx.x;
  if (r < 0) return;
  carry_row(r, kv, score, pos, carry_at(carry, r));
}

}  // namespace

void compressor_project(const uint16_t* x, const CompressorWeights& w, float* kv, float* score, int rows,
                        cudaStream_t s) {
  gemv_bf16(x, rows, kDim, w.wkv, w.wgate, kHead, w.ratio > 1 ? 2 * kHead : kHead, kv, score, w.ratio == 1, s);
}

void compressor_emit(const float* kv, const float* score, const int* pos, int rows, const CompressorWeights& w,
                     CompressorCarry carry, const float2* table, Compressed ckv, IndexKeys ik,
                     uint16_t* debug_latent, int debug_base, cudaStream_t s) {
  // the carry keeps a call's last 16 rows, so with session slots every row (no two rows of a session on one slot)
  if (carry.slot && rows > 16) die("compressor_emit: %d rows with session slots", rows);
  check_slots(carry, "compressor_emit");
  check_slots(ckv, "compressor_emit");
  check_slots(ik, "compressor_emit");
  emit_kernel<<<rows, 128, 0, s>>>(kv, score, pos, rows, w, carry, table, ckv, ik, debug_latent, debug_base);
  CUDA_CHECK(cudaGetLastError());
  if (w.ratio > 1) {
    carry_kernel<<<rows < 16 ? rows : 16, 128, 0, s>>>(kv, score, pos, rows, carry);
    CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace attn
