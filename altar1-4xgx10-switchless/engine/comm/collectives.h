// Standalone collectives on the ring: one kernel each, producer and consumer halves together.
// Each rank must call the same collectives in the same order. All are CUDA-graph capturable.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "comm/ring.h"

namespace comm {

// out = in_0 + in_1 + in_2 + in_3 summed in fp32 in rank order (bit-identical on every rank).
// n elements; n * sizeof(element) must be a multiple of 16 and at most ch.cap. in and out may alias.
void allreduce_bf16(const DevChannel& ch, const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t st);
void allreduce_f32(const DevChannel& ch, const float* in, float* out, size_t n, cudaStream_t st);

// out = [in_0 | in_1 | in_2 | in_3], each nbytes (a multiple of 16). Above ch.cap it runs in
// pipelined chunks (16 blocks).
void allgather(const DevChannel& ch, const void* in, void* out, size_t nbytes, cudaStream_t st);

// out = quarter `rank` of in_0 + in_1 + in_2 + in_3 (same order and rounding as the all-reduce): in
// has n elements on every rank, out n / 4. n * sizeof(element) must be a multiple of 64, at most ch.cap.
void reduce_scatter_bf16(const DevChannel& ch, const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t st);
void reduce_scatter_f32(const DevChannel& ch, const float* in, float* out, size_t n, cudaStream_t st);

// All-reduce of any size, bit-identical to allreduce_bf16/f32: chunks of up to ch.cap bytes, each a
// reduce-scatter followed by an all-gather of the reduced quarters (0.875 instead of 1.5 times the
// payload per link and direction), in one kernel of `blocks` blocks (few, so that other work can run
// alongside). n * sizeof(element) must be a multiple of 64; in and out may alias.
void allreduce_bulk_bf16(const DevChannel& ch, const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t st,
                         int blocks = 16);
void allreduce_bulk_f32(const DevChannel& ch, const float* in, float* out, size_t n, cudaStream_t st, int blocks = 16);

// Reduce-scatter of any size: in has n elements on every rank, four quarters (quarter k for rank k),
// out gets this rank's quarter summed as reduce_scatter_bf16/f32 does (bit-identical). Pipelined one
// deep in posts of up to ch.cap bytes (a piece of every quarter each), in one kernel of `blocks`
// blocks. n * sizeof(element) must be a multiple of 64; out must not overlap in except at this rank's
// own quarter (out == in + rank * n / 4 works: each piece is read before its sum is written).
void reduce_scatter_bulk_bf16(const DevChannel& ch, const __nv_bfloat16* in, __nv_bfloat16* out, size_t n,
                              cudaStream_t st, int blocks = 16);
void reduce_scatter_bulk_f32(const DevChannel& ch, const float* in, float* out, size_t n, cudaStream_t st,
                             int blocks = 16);

// The same by rows of a column block: in is [4 rows][ld_in] elements and quarter k its rows
// [k rows, (k + 1) rows), columns [0, cols) (offset in to the block's first column); out is [rows][ld_out].
// cols * sizeof(element) must be a multiple of 16; the sums are those of reduce_scatter_bulk_*.
void reduce_scatter_rows_bf16(const DevChannel& ch, const __nv_bfloat16* in, size_t ld_in, __nv_bfloat16* out,
                              size_t ld_out, int rows, int cols, cudaStream_t st, int blocks = 16);
void reduce_scatter_rows_f32(const DevChannel& ch, const float* in, size_t ld_in, float* out, size_t ld_out, int rows,
                             int cols, cudaStream_t st, int blocks = 16);

// All-gather of several buffers in one pipelined collective: for each segment, out = [in_0 | in_1 |
// in_2 | in_3] of `bytes` each (a multiple of 16). in may be out + rank * bytes (in place).
struct AgSeg {
  const void* in;
  void* out;
  size_t bytes;
};
constexpr int kMaxAgSegs = 6;
void allgather_multi(const DevChannel& ch, const AgSeg* segs, int nseg, cudaStream_t st, int blocks = 16);

// All-to-all by rows: quarter k of each segment (rows [k rows, (k + 1) rows) of in, row stride ld_in
// bytes) goes to rank k, and rank k's quarter for this rank lands at out + (k rows + r) ld_out. Rows are
// rb bytes (multiples of 16, as the strides). Per quarter the segments form one stream, sent in posts of
// up to ch.cap bytes on the reduce-scatter's transport (0.5 n per link and direction, n = the bytes of
// all four quarters), in one kernel of `blocks` blocks of 256 threads (small, so that work can share
// their SMs).
struct A2aSeg {
  const void* in;
  size_t ld_in;
  void* out;
  size_t ld_out;
  size_t rb;
};
constexpr int kMaxA2aSegs = 4;
void alltoall_rows(const DevChannel& ch, const A2aSeg* segs, int nseg, int rows, cudaStream_t st, int blocks = 32);

#ifdef GLM_COMM_TEST_HOOKS
// Test only (ring_test's copy of this library; compiled out of the engine): for the launches that follow on st,
// block `block` of every pipelined reduce-scatter, all-to-all and multi-buffer all-gather stalls for ns after
// posting piece i + 1 and before reading piece i, at i = piece (negative: npieces - 4, where the short last
// piece reuses i's send buffer). ns 0: off. Waits for st.
void test_stall(uint32_t ns, int block, int piece, cudaStream_t st);
#endif

// Per row, the best (value, index) of the four ranks: larger value wins, ties go to the lower index.
// For distributed argmax / Gumbel-max sampling over a vocab-split head. rows * 8 must be a multiple of 16.
struct ArgMax {
  float v;
  int i;
};
void argmax_rows(const DevChannel& ch, const ArgMax* in, ArgMax* out, int rows, cudaStream_t st);

// Launch shape the collectives use (one 16-byte chunk per thread where possible). All blocks of a
// kernel that waits must be resident at once, hence at most 48 blocks.
int grid_for(size_t bytes);
int block_for(size_t bytes);

}  // namespace comm
