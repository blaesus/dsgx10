// GLM-5.3's collectives at TP=4 (docs/ring.md, "GLM's collectives"), built from the ring_dev.cuh halves:
// the per-layer all-reduce of hidden-size f32 partials (plain, or fused with the residual add and the next
// RMSNorm), the row-interleaved all-gathers the attention side reads in place (the x-projection with the
// indexer's head weights, the indexer queries, the position-split indexer's candidates), the head's
// vocabulary-split logits (all-gathered, or a top-k exchange), at 1-16 rows; and the same all-gathers
// pipelined for prefill chunks. The sizes are the model's (attn/attn.h), compile-time.
//
// Every rank must call the same collectives in the same order (as comm::). All are CUDA-graph capturable.
// With pdl, a kernel waits for its predecessor (griddepcontrol.wait) before it touches anything, and lets its
// dependents launch once it has posted; launched without PDL the wait is a no-op.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "attn/attn.h"
#include "comm/ring.h"

namespace comm::glm {

using ::glm::kDim;           // 6144: hidden
using ::glm::kMaxRows;       // 16: decode/verify rows a step
using ::glm::kXRows;         // 688: a rank's x-projection rows on a full-indexer layer (q_a, kv_a, wk)
using ::glm::kXRowsShared;   // 656: the same on a shared-indexer layer (no wk)
using ::glm::kIdxHeadsRank;  // 8: a rank's indexer heads (weights_proj rows, wq_b heads)
using ::glm::kIdxHeads;      // 32
using ::glm::kIdxDim;        // 128
using ::glm::kTopK;          // 2048: the indexer's top-k
using ::glm::kVocab;         // 154,880
using ::glm::kVocabRank;     // 38,720 head rows a rank
inline constexpr float kNormEps = ::glm::kNormEps;

// Bytes a rank contributes per row.
inline constexpr size_t kHiddenF32Row = (size_t)kDim * 4;                           // 24,576
inline constexpr size_t kXprojRow = (size_t)kXRows * 2 + kIdxHeadsRank * 4;         // 1,408 (full layer)
inline constexpr size_t kXprojRowShared = (size_t)kXRowsShared * 2;                 // 1,312
inline constexpr size_t kIdxQRow = (size_t)kIdxHeadsRank * kIdxDim * 2;             // 2,048
inline constexpr size_t kCandRow = (size_t)kTopK * 8;                               // 16,384 (pos + key)
inline constexpr size_t kLogitsF32Row = (size_t)kVocabRank * 4;                     // 154,880
inline constexpr size_t kLogitsBf16Row = (size_t)kVocabRank * 2;                    // 77,440
static_assert(kXRows * 2 % 16 == 0 && kXRowsShared * 2 % 16 == 0 && kLogitsBf16Row % 16 == 0, "16-byte rows");

// ---- (a) the all-reduce of hidden-size f32 partials (o_proj; the MoE/MLP down) -----------------------------
// part [R][6144] f32, this rank's partial sums (R in 1..16). sum = ((p0 + p1) + p2) + p3 in f32, the four ranks'
// partials in rank order: the same bits as comm::allreduce_f32, on every rank, run to run.
// allreduce_hidden: out_f32 [R][6144] the sums and/or out_bf16 [R][6144] the sums rounded to bf16 (either null).
// allreduce_add_norm, the residual add and the next RMSNorm fused into the end (the reference's arithmetic, and
// bit for bit attn's rmsnorm kernel):
//   y = bf16(sum); h = bf16(res + y) -> res_out (may be res); xn = bf16(w * bf16(h * rsqrt(mean(h^2) + eps)))
// with mean(h^2) summed in rmsnorm's order (256 threads of 24 contiguous columns, a warp xor-tree, the warps in
// order). xn [R][6144] bf16, w [6144] bf16. slices: the grid's column split (8, 16 or 32: the same bits; 0: 8, measured best).
// The fused kernel keeps a row sum per thread in device memory across its blocks: at most one runs at a time
// (stream order or PDL guarantees it in a step's graph).
void allreduce_hidden(const DevChannel& ch, const float* part, int R, float* out_f32, uint16_t* out_bf16,
                      cudaStream_t st, bool pdl = false, int slices = 0);
void allreduce_add_norm(const DevChannel& ch, const float* part, int R, const uint16_t* res, uint16_t* res_out,
                        const uint16_t* w, float eps, uint16_t* xn, cudaStream_t st, bool pdl = false, int slices = 0);

// ---- (b) all-gathers by rows ---------------------------------------------------------------------------------
// Segment g: this rank's rows of rb bytes at in + r ld_in (r < rows); rank k's row r lands at
// out + r ld_out + k rb. rb, the strides and the addresses are multiples of 16. In place when
// in == out + rank rb and ld_in == ld_out (this rank's block is then not rewritten). The segments of a rank
// form one stream, sent in posts of up to `chunk` bytes (0: ch.cap), pipelined one deep with the copies two
// ahead (as comm::allgather_multi; each 16-byte unit of a send buffer is written and read by one thread, so
// no wait_posted is needed). blocks 0: by size (1-16, 1024 threads at most).
struct RowSeg {
  const void* in;
  size_t ld_in;
  void* out;
  size_t ld_out;
  size_t rb;
};
inline constexpr int kMaxRowSegs = 4;
void allgather_rows(const DevChannel& ch, const RowSeg* segs, int nseg, int rows, cudaStream_t st, int blocks = 0,
                    size_t chunk = 0, bool pdl = false);

// The x-projection's outputs, in place (docs/attention.md § 1): xg [R][4][xrows] bf16, rank r's GEMV having
// written its xrows at xg + r xrows (ldy 4 xrows); on full-indexer layers also the indexer's head weights,
// iw [R][32] f32, rank r's 8 at iw + 8 r (index_weights' ldw 32). full: xrows = 688 and iw; else 656, iw unused.
// Any row count (prefill chunks pipeline).
void allgather_xproj(const DevChannel& ch, uint16_t* xg, float* iw, int R, bool full, cudaStream_t st, bool pdl = false);
// The indexer's queries (wq_b's output), in place: iq [R][32][128] bf16, rank r's 8 heads at iq + 1024 r.
void allgather_idxq(const DevChannel& ch, uint16_t* iq, int R, cudaStream_t st, bool pdl = false);
// The position-split indexer's candidates (index_select over a quarter): pos [R][2048] i32 and key [R][2048]
// u32 -> pos_all, key_all [R][4][2048] (index_merge's input).
void allgather_cand(const DevChannel& ch, const int* pos, const uint32_t* key, int* pos_all, uint32_t* key_all, int R,
                    cudaStream_t st, bool pdl = false);

// ---- (d) the head: vocabulary-split logits ------------------------------------------------------------------
// local [R][ld_local] this rank's 38,720 logits a row (f32 or bf16) -> all [R][154,880], rank k's at 38,720 k.
// In place when local == all + 38,720 rank and ld_local == 154,880.
void allgather_logits_f32(const DevChannel& ch, const float* local, size_t ld_local, float* all, int R, cudaStream_t st,
                          bool pdl = false);
void allgather_logits_bf16(const DevChannel& ch, const uint16_t* local, size_t ld_local, uint16_t* all, int R,
                           cudaStream_t st, bool pdl = false);
// Or each rank's own top-K (value, global vocabulary index) a row, local [R][K], exchanged and merged: out [R][K]
// the best K of the 4K by value, ties to the lower index, best first (K = 1: the argmax). The same on every
// rank. Values must not be NaN. K in 1..64.
struct Cand {
  float v;
  int i;
};
inline constexpr int kMaxTopK = 64;
void topk_exchange(const DevChannel& ch, const Cand* local, Cand* out, int R, int K, cudaStream_t st, bool pdl = false);

#ifdef GLM_COMM_TEST_HOOKS
// Test only (glm53_comm_test): the TEST_STALL of collectives.cu for allgather_rows, the same settings
// (comm::test_stall sets both).
void test_stall_rows(uint32_t ns, int block, int piece, cudaStream_t st);
#endif

}  // namespace comm::glm
