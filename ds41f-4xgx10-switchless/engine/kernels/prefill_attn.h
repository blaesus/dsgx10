// The attention side of prefill (M5, part 2): sparse attention, the indexer, the compressor and mHC
// for chunks of 2048-8192 rows at per-rank shapes (TP=4, 16 heads). State, formats and numerics are
// engine/attn's (attn/attn.h): these are the many-row versions of its decode-shaped kernels and are
// tested against them. Rows are a chunk's query tokens in ascending position order.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#include "attn/attn.h"

namespace dsk {

// ---- sparse attention (prefill_attn.cu) ---------------------------------------------------------
// attn::sparse_attn for many rows: row r attends to positions [max(win_lo, pos-127), pos] of `win` (CED's
// decoder replay truncates the window at its first row: win_lo = P - 128, docs/ced.md) and to the
// compressed entries topk[r][0..kmax) of `ckv` (-1 = none; kmax a multiple of 64, at most 512, 0 for the
// window-only layers 0-1). q_pre [rows][16][512] bf16 (RoPE at pos applied on load), sink [16], out
// [rows][16][512] bf16 with the inverse RoPE applied. Online softmax over blocks of 64 keys with P rounded
// to bf16 for PV and the sink in the denominator, as the reference; not bit-exact (summation order).
void sparse_attn_p(const uint16_t* q_pre, const int* pos, int rows, const float2* table, attn::Window win,
                   attn::Compressed ckv, const int* topk, int kmax, const float* sink, uint16_t* out, cudaStream_t s,
                   int win_lo = 0);

// ---- indexer (prefill_index.cu) ---------------------------------------------------------------------
// attn::index_score's dense scan (one part) for many rows, bit-identical: keys[r][i] for positions
// i < n = (pos[rows-1]+1)/ratio (0 past row r's own length), and with bkeys the block maxima (the row's
// newest block pinned to 0xFFFF). max_n >= n sizes the grid; keys_stride >= max_n rounded up to 256.
void index_score_p(const attn::IndexQ& q, const int* pos, int rows, int ratio, attn::IndexKeys ik, int max_n,
                   uint16_t* keys, int keys_stride, uint16_t* bkeys, int bkeys_stride, cudaStream_t s);
// The indexer's selection for a prefill chunk's rows on this rank (ascending positions): batches of
// `batch` rows, each scored (index_score_p) and then selected by attn::index_topk (torch.topk-exact: the
// min(kmax, len) best positions ascending, -1 padded; out [rows][kmax]) while the batch's keys
// [batch][keys_stride] are still in L2. With bkeys ([batch][keys_stride/8] scratch) also the pool of
// layer 20: the best pool_k blocks of 8 positions [rows][pool_k].
void index_topk_p(const attn::IndexQ& q, const int* pos, int rows, int ratio, attn::IndexKeys ik, int max_n, int batch,
                  uint16_t* keys, int keys_stride, int kmax, int* out_pos, uint16_t* out_key, uint16_t* bkeys,
                  int pool_k, int* pool_pos, uint16_t* pool_key, cudaStream_t s);

// ---- compressor (prefill_comp.cu) -------------------------------------------------------------------
// attn::compressor_project as a tensor-core GEMM (x [rows][5120] bf16 -> kv, score fp32 [rows][512];
// ratio 1 rounds kv to bf16 values), summation order only.
void compressor_project_p(const uint16_t* x, const attn::CompressorWeights& w, float* kv, float* score, int rows,
                          cudaStream_t s);
// attn::compressor_emit for many rows, with the same contract (row r completes entry pos[r]/ratio when
// (pos[r]+1) % ratio == 0; ratio 2 pools rows r-1 and r, or the carry for r = 0; then the carry keeps the
// last 16 rows). Rows: consecutive positions, or pairs of them for ratio 2. KV entries bit-identical; wk
// runs on tensor cores in another summation order (rare one-step differences in index-key elements).
void compressor_emit_p(const float* kv, const float* score, const int* pos, int rows, const attn::CompressorWeights& w,
                       attn::CompressorCarry carry, const float2* table, attn::Compressed ckv, attn::IndexKeys ik,
                       cudaStream_t s);

// ---- mHC (prefill_mhc.cu) ---------------------------------------------------------------------------
// hc_fn [24][20480] fp32 as three bf16 parts in MMA fragment order (exact), made once per weight.
size_t hc_fn_parts_bytes();
void hc_fn_parts(const float* fn, uint32_t* parts, cudaStream_t s);
// One sublayer boundary of the residual stream for many rows, in one pass over it (model.py's single-pass
// mHC: the coefficients come from the previous boundary):
//   y = hc_post(sub, res, post, comb)                    (no sub: y = res, nothing written)
//   x_out = norm_w * RMSNorm(hc_pre(y, pre)), optional act_quant -> xq / xs (no norm_w: hc_pre only)
//   pre_out, post_out, comb_out = Sinkhorn(hc_fn . y * rsqrt(mean(y^2) + eps))   (with fn_parts)
// hc_post, hc_pre, the norm, act_quant and the Sinkhorn are bit-identical to attn's / dsk::act_quant;
// the projection (bf16 MMA on the exact parts, fp32 sums per 64 columns of a copy) differs in summation
// order. A CTA takes 32 rows: under ~1500 rows the grid leaves SMs idle.
struct HcBoundary {
  const uint16_t* sub = nullptr;      // [rows][5120] the sublayer's output
  const uint16_t* res = nullptr;      // [rows][4][5120] the stream before it
  const float *pre = nullptr, *post = nullptr, *comb = nullptr;  // [rows][4], [rows][4], [rows][16]
  const uint32_t* fn_parts = nullptr; // the next sublayer's hc_fn (hc_fn_parts), with its scale [3], base [24]
  const float *scale = nullptr, *base = nullptr;
  const uint16_t* norm_w = nullptr;   // [5120] the next sublayer's attn_norm / ffn_norm
  uint16_t* y = nullptr;              // [rows][4][5120]; null: not written (layer 20's projection of
                                      // the rows before the CED replay needs only x)
  uint16_t* x_out = nullptr;          // [rows][5120]
  uint8_t *xq = nullptr, *xs = nullptr;  // e4m3 [rows][5120], ue8m0 [rows][160]
  float *pre_out = nullptr, *post_out = nullptr, *comb_out = nullptr;
  float* proj_out = nullptr;          // optional [rows][25]: hc_fn . y and sum(y^2) (tests)
};
void hc_boundary_p(const HcBoundary& a, int rows, cudaStream_t s);
// hc_boundary_p's rows per CTA (4, 8, 16, 32; -1: by the chunk's rows; 0: the default) and L2 prefetch
// distance in blocks of 64 columns (0: none; -1: the default). Benches and tests; the same bits.
void hc_boundary_tuning(int rows_per_cta, int prefetch_blocks = -1);
// The split boundary for chunks of up to max_rows rows (at most 512; 0: off, the default): each row group's
// columns in two halves on two CTAs, the stream and hc_fn's parts S blocks ahead, the last CTA of a group
// summing the partials in the default kernel's orders: the same bits. rows_per_cta 4, 8, 16 (0: by the
// chunk's rows). Allocates its scratch on the first call that turns it on (not during a graph capture).
void hc_boundary_split(int max_rows, int rows_per_cta = 0);

}  // namespace dsk
