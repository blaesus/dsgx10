// dsgb10 kernels: the weight-streaming GEMVs of one decode/verify step (M <= 16 activation rows) and
// the prefill GEMMs (chunks of M rows, prefill.cu), on per-rank (TP=4) shapes. All tensors are device
// pointers in the checkpoint's own formats: e4m3 (uint8), packed e2m1 (uint8, low nibble first),
// ue8m0 scales (uint8), bf16.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace comm {
struct DevChannel;
}

namespace dsk {

// Activation rows per call: a verify step has up to 6, a batched step (docs/batching.md, B4) up to 16.
// Up to kLowRows the kernels run as they did before B4: an activation row per lane group, in the lower
// half of the m16 MMA tiles (the upper half zero). From 9 rows on they fill the upper half as well (rows
// 8..15): the same weight loads and MMAs, the activation loads doubled.
constexpr int kLowRows = 8;
constexpr int kMaxRows = 16;
constexpr int kTopK = 6;                      // routed experts per row
constexpr int kMaxUnion = kMaxRows * kTopK;  // distinct experts one MoE call can touch
// The rows a call's kernels are built for: 8 up to 8 rows, 16 above.
constexpr int row_budget(int M) { return M <= kLowRows ? kLowRows : kMaxRows; }

// bf16 x [M][K] -> e4m3 q [M][K] + ue8m0 s [M][K/32], bit-exact with the reference's act_quant.
void act_quant(const __nv_bfloat16* x, int M, int K, uint8_t* q, uint8_t* s, cudaStream_t st);

// Launch shape of the GEMV kernels: n-tiles of 8 rows per warp (template-selected: 1, 2 or 4),
// warps along N and along K per CTA.
struct GemvCfg {
  int mt = 1, wn = 1, wk = 16;
  bool pdl = true;    // launch with programmatic stream serialization (overlap with the previous kernel)
  bool tiled = false; // weights in the tiled layout of tile_fp8()
  // gemv_fp8 with tiled weights and mt = wn = 1: the persistent kernel (gemv_fp8.cu), the same bits.
  // ring: steps of 512 bytes each warp keeps in flight (4, 8, 12 or 16; 0: the deepest that fits;
  // -1: the one-tile-per-CTA kernel), ctas: its grid (0: one CTA per SM)
  int ring = -1, ctas = 0;
};

// EXPERIMENT: DSK_CARVEOUT=<percent> gives every launch a shared-memory carveout preference (off by
// default). At 100 the SMs keep one L1 / shared split across the step, so the next kernel's CTAs can
// start beside the previous one's (PDL) without the SM draining to change it: bench_chain at one row
// 18.21 -> 18.04 ms, the decode loop at k = 1 18.56 -> 18.46 ms; but spec steps of 2-6 rows took
// 0.3-1.9 ms longer (the 28 KB of L1 left does not hold their activations), so it stays off.
inline int carveout_pref() {
  static const int c = getenv("DSK_CARVEOUT") ? atoi(getenv("DSK_CARVEOUT")) : -1;
  return c;
}

// The normed GEMVs (wq_a + wkv, wq_b (+ the indexer's), the router gate: each follows a collective)
// take the persistent kernels (GemvCfg::ring = 0, the same bits) up to this many rows, whatever their
// GemvCfg says: their rings load while the collective runs. One row by default (the decode loop at
// k = 1, 1k: 18.58 -> 18.32 ms; spec steps of 2-6 rows unchanged; with more rows the normed input
// takes the rings' shared memory). DSK_PERSIST=<rows> sets another bound, 0 turns it off.
inline int persist_rows() {
  static const int r = getenv("DSK_PERSIST") ? atoi(getenv("DSK_PERSIST")) : 1;
  return r;
}

// cudaLaunchKernelEx with the PDL attribute when asked.
template <class... KArgs, class... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, bool pdl,
                       Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (pdl) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na++].val.programmaticStreamSerializationAllowed = 1;
  }
  if (carveout_pref() >= 0) {
    attr[na].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
    attr[na++].val.sharedMemCarveout = carveout_pref();
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
  if (e != cudaSuccess) {
    fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(e));
    abort();
  }
}

// One weight matrix of a multi-matrix GEMV: rows N, output y [M][N] bf16 (gemm_fp8_p: rows of ld
// elements when ld is set).
struct Fp8Seg {
  const uint8_t* W;
  const uint8_t* S;
  __nv_bfloat16* y;
  int N;
  int ld = 0;
};
constexpr int kMaxSegs = 4;

// Re-lay an FP8 weight [N][K] for the GEMV: [N/8][K/64][8 rows][64 bytes] (a byte permutation;
// scales are unchanged). src and dst must not overlap.
void tile_fp8(const uint8_t* src, uint8_t* dst, int N, int K, cudaStream_t st);

// FP8 block-scaled GEMV, tensor cores: y[M][N] = bf16(sum_k (q*s_a)[m][k] (W*S)[n][k]).
// W e4m3 [N][K] (row stride K), S ue8m0 [N/32][K/32] (32x32 blocks), xq/xs from act_quant.
void gemv_fp8(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* W, const uint8_t* S, int N, int K,
              __nv_bfloat16* y, GemvCfg cfg, cudaStream_t st);
// Several matrices with the same input and K in one launch (e.g. wq_a + wkv, w1 + w3).
// Every N must be a multiple of the rows per CTA (8 * mt * wn).
void gemv_fp8_multi(const uint8_t* xq, const uint8_t* xs, int M, const Fp8Seg* segs, int nseg, int K, GemvCfg cfg,
                    cudaStream_t st);
// With the RMSNorm in front fused in: every CTA normalises bf16 x [M][K] (K a multiple of 256) with
// weight norm_w and quantises it into shared memory while its first weights load (no separate norm
// or act_quant launch). Optionally CTA 0 also writes the normed bf16 (y_out) and its act_quant.
void gemv_fp8_norm(const __nv_bfloat16* x, const __nv_bfloat16* norm_w, float eps, int M, const Fp8Seg* segs, int nseg,
                   int K, GemvCfg cfg, cudaStream_t st, __nv_bfloat16* y_out = nullptr, uint8_t* q_out = nullptr,
                   uint8_t* qs_out = nullptr);
// The same on CUDA cores (fp32 FMAs), for comparison.
void gemv_fp8_cc(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* W, const uint8_t* S, int N, int K,
                 __nv_bfloat16* y, cudaStream_t st);

// ---- 16-bit activations (gemv16.cu). Weights in the tile_fp8 byte layout (bf16 rows: tile_fp8 with
// 2K bytes per row). cuda_cores: fp32 FMAs instead of tensor cores, for comparison (mt 1 or 2).
// bf16 W [N][K], x bf16 [M][K] -> y = x W^T in fp32 math; written fp32 (yf) or rounded to bf16 (yb).
void gemv_bf16(const __nv_bfloat16* x, int M, const uint8_t* W, int N, int K, float* yf, __nv_bfloat16* yb, GemvCfg cfg,
               bool cuda_cores, cudaStream_t st);
// The same with the RMSNorm in front fused in (tensor cores): every CTA normalises bf16 x [M][K] (K a
// multiple of 256) with norm_w into shared memory; CTA 0 also writes its act_quant (q_out [M][K],
// qs_out [M][K/32]) when given, e.g. the router gate's GEMV producing the MoE's input on the way.
void gemv_bf16_norm(const __nv_bfloat16* x, const __nv_bfloat16* norm_w, float eps, int M, const uint8_t* W, int N, int K,
                    float* yf, __nv_bfloat16* yb, GemvCfg cfg, cudaStream_t st, uint8_t* q_out = nullptr,
                    uint8_t* qs_out = nullptr);
// bf16 weights coded exactly in fewer bytes (the vocab head: 3/4 of them): W as gemv_bf16 reads it, each
// tile of 8 rows x 32 values with its exponents as 4-bit offsets from the tile's least (a tile whose
// exponents span more than 16 values stays raw). bf16x_encode returns the coded copy (cudaMalloc'd, *bytes;
// *raw tiles kept raw; synchronises); bf16x_check counts the 16-byte pieces of W that X does not decode
// to (0: exact). gemv_bf16x / gemv_bf16x_norm read X and give gemv_bf16's / gemv_bf16_norm's bits (tensor
// cores, mt 1 or 2; the one-tile-per-CTA kernel whatever cfg.ring says).
uint8_t* bf16x_encode(const uint8_t* W, int N, int K, size_t* bytes, int* raw, cudaStream_t st);
uint64_t bf16x_check(const uint8_t* X, const uint8_t* W, int N, int K, cudaStream_t st);
void gemv_bf16x(const __nv_bfloat16* x, int M, const uint8_t* X, int N, int K, float* yf, __nv_bfloat16* yb, GemvCfg cfg,
                cudaStream_t st);
void gemv_bf16x_norm(const __nv_bfloat16* x, const __nv_bfloat16* norm_w, float eps, int M, const uint8_t* X, int N, int K,
                     float* yf, __nv_bfloat16* yb, GemvCfg cfg, cudaStream_t st);
// wo_a: FP8 W [N][K] with [N/32][K/32] scales, widened to bf16 (exact; block scales <= 2^7);
// block-diagonal: rows [g*group_rows, (g+1)*group_rows) read x[:, g*K:(g+1)*K]. x [M][N/group_rows*K].
// With q/qs also act_quant(y) (wo_b's input; y may then be null); needs mt * wn = 4.
void gemv_woa(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, int group_rows,
              __nv_bfloat16* y, GemvCfg cfg, bool cuda_cores, cudaStream_t st, uint8_t* q = nullptr,
              uint8_t* qs = nullptr);
// The same widening with one group (every row reads all of x [M][K]) and fp32 out: DSpark's FP8 head.
void gemv_fp8w_f32(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, float* yf,
                   GemvCfg cfg, cudaStream_t st);
// ---- gemv_fp4.cu: the same with MXFP4 weights (DSpark's head as MXFP4, dspark::head_fp4): W e2m1 [N/8][K/128]
// [8 rows][4][16 B] (lane (g, t) of a tile: row g's k = 128j + 16t + i, then 128j + 64 + 16t + i, i = 0..15, low
// nibble first), S ue8m0 [N/8][K/128][8 rows][4] (per row, one per 32 values; at most 128). M <= 8, mt 1, 2 or
// 4, K a multiple of 128 * wk. Each output is gemv_fp8w_f32's with the same wk on the same weight values.
void gemv_fp4w_f32(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, float* yf,
                   GemvCfg cfg, cudaStream_t st);

// ---- norm_router.cu
// y = bf16(w * (x * rsqrt(mean(x^2) + eps))), fp32 math, rows of D (multiple of 256, <= 8192);
// with q/s non-null also act_quant(y) into them.
void rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, int M, int D, float eps, __nv_bfloat16* y, uint8_t* q,
             uint8_t* s, cudaStream_t st);
// Router selection from fp32 logits [M][E] (E <= 512): scores sqrt(softplus(l)), top-6 of
// scores + bias (descending), weights = scores / (sum + 1e-20) * route_scale.
void router_topk(const float* logits, const float* bias, int M, int E, int* idx, float* wts, float route_scale,
                 cudaStream_t st, const float* bias_vl = nullptr, const int* image_mask = nullptr);

// ---- MoE (routed MXFP4 experts + FP8 shared expert), per rank at TP=4: dim 5120, 576 intermediate

// Re-lay an MXFP4 weight for the MoE kernels: w [N][K/2] (e2m1 pairs, low nibble first) and s
// [N][K/32] -> wt [N/16][K/64][32 lanes][16 B], st [N/16][K/64][8][4 B]. Lane (g, t) of a 16-row
// tile holds, per 64 k-values, the 8 values k = 8t..8t+7 of blocks 2p and 2p+1 for rows g and g+8.
void tile_fp4(const uint8_t* w, const uint8_t* s, uint8_t* wt, uint8_t* st, int N, int K, cudaStream_t stream);

// The 1-bit code of a tiled (L3) expert scale tensor, made at load time by scale_code: nearly every
// block scale of a layer's experts is one of two neighbours {b, b + 1}, so per 32-scale group (a tile's
// step: the 32 bytes one warp-step reads) a 32-bit word holds them, bit 8j + g for byte j of group g's
// word, and ((word >> g) & 0x01010101) + b * 0x01010101 rebuilds that word. A group with another value
// is flagged and read raw, from the scales themselves, which stay. Per (expert, 16-row tile), in
// code_block_words(P) words: the P steps' code words (0 for a raw group), then the raw flags (bit p of
// word p / 32: step p), zero-padded to 16 bytes. code null: the kernels read the raw scales.
struct ScaleCode {
  const uint32_t* code = nullptr;
  uint32_t base4 = 0;  // b * 0x01010101
};
constexpr int code_block_words(int P) { return (P + (P + 31) / 32 + 3) / 4 * 4; }
// Words of the code of scales [E][N][KB] (N % 16 == 0, KB even).
size_t scale_code_words(int E, int N, int KB);
// Writes the code of the L3 scales s [E][N][KB] to code (scale_code_words words) with the pair {b, b + 1}
// that holds the most groups: b next to the commonest value. Returns the share of groups in the pair (the
// rest are read raw) and b in *base. Synchronous: load time.
double scale_code(const uint8_t* s, int E, int N, int KB, uint32_t* code, int* base, cudaStream_t st);

// Weights of one layer on this rank. Routed experts are stacked by expert id ([E][rows][...], so
// expert e is at e times the per-expert size), each matrix tiled by tile_fp4: w1, w3 [576][5120]
// and w2 [5120][576]. The shared expert is FP8, tiled by tile_fp8, with its [N/32][K/32] scales.
// c1, c3, c2: the 1-bit codes of s1, s3, s2 when made; the decode and prefill MoE kernels then read them
// instead, for the same scale bytes (up takes c1 and c3 only together; k_moe_down_g, a prefill tuning, reads
// the raw scales).
struct MoeWeights {
  const uint8_t *w1, *s1, *w3, *s3, *w2, *s2;
  const uint8_t *sh_w1, *sh_s1, *sh_w3, *sh_s3, *sh_w2, *sh_s2;
  ScaleCode c1{}, c3{}, c2{};
};

struct MoeCfg {
  int up = 0, down = 4;  // kernel variants (moe.cu); up 6..9 do the routed dot products on CUDA cores
  int up_grid = 0;       // CTAs of the persistent first kernel; 0 = all that fit
  bool pdl = true;
  int phases = 3;        // bit 0: moe_up, bit 1: moe_down (benchmarks)
  bool dyn = false;      // true: moe_up's tasks past the first wave go to whichever CTA is free (a counter
                         // in the workspace); false: CTA b takes tasks b, b + grid, ... Same bits; the queue
                         // measured the same as the static schedule in the decode chain (docs/engine.md).
  int up16 = 4, down16 = 2;  // 9..16 rows: variants from moe.cu's run_moe16 lists (tensor cores only); the
                             // fastest pair at 9, 12 and 16 rows (README, 9-16 rows)
};

size_t moe_workspace_bytes();
// The workspace of a call of M rows, with RB = row_budget(M) (tests read moe_up's output from it): hq
// [RB * kTopK + 1][RB][576] e4m3 (the union's slots, then the shared expert's), at moe_hs_offset(M) hs
// [..][RB][18] ue8m0; moe_up's task counters at the end.
constexpr size_t moe_hs_offset(int M) { return (size_t)(row_budget(M) * kTopK + 1) * row_budget(M) * 576; }

// y [M][5120] fp32 = sum over each row's picks, in expert-id order, of bf16(expert(x) * weight),
// plus the shared expert: this rank's partial, before the all-reduce. xq/xs = act_quant(x);
// idx/wts [M][6] from the router (expert ids, fp32 weights); ws of moe_workspace_bytes(), zeroed before
// its first use (the kernels leave its task counters at zero).
void moe(const uint8_t* xq, const uint8_t* xs, int M, const int* idx, const float* wts, const MoeWeights& w, void* ws,
         float* y, MoeCfg cfg, cudaStream_t st);
// The same with the router's selection fused in: both kernels run router_topk's selection on the
// gate's logits [M][E] (fp32, E <= 512) themselves, so no top-k launch sits between gate and MoE;
// idx_out/wts_out [M][6] (optional) receive it.
void moe_routed(const uint8_t* xq, const uint8_t* xs, int M, const float* logits, const float* bias, int E,
                float route_scale, const MoeWeights& w, void* ws, float* y, MoeCfg cfg, cudaStream_t st,
                int* idx_out = nullptr, float* wts_out = nullptr);

// ---- Prefill (prefill.cu): GEMMs for chunks of M rows (any M >= 1; built for 128..8192) on tensor
// cores, reading the same weight layouts as the decode kernels. Row-major activations and outputs.

// FP8 x FP8 block-scaled GEMM: y = bf16(act_quant(x) W^T) per segment (several matrices sharing
// the input, e.g. wq_a + wkv). Every N must be a multiple of 64, K of 64. Weights tiled by tile_fp8.
void gemm_fp8_p(const uint8_t* xq, const uint8_t* xs, int M, const Fp8Seg* segs, int nseg, int K, cudaStream_t st);
// out = bf16(((p0 + p1) + p2) + p3), fp32 sums of the bf16 partials p_k = part + k * slot (n elements
// each, n a multiple of 8), rounded once: the reduce-scatter's sum, for partials made on one rank.
void sum4_bf16_p(const __nv_bfloat16* part, size_t slot, size_t n, __nv_bfloat16* out, cudaStream_t st);
// The shared expert's w1/w3 (each [N][K], N a multiple of 64) with SwiGLU, bf16 rounding and act_quant
// fused: hq [M][N] e4m3, hs [M][N/32] ue8m0, ready for w2's gemm_fp8_p.
void gemm_fp8_swiglu_p(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w1, const uint8_t* s1,
                       const uint8_t* w3, const uint8_t* s3, int N, int K, uint8_t* hq, uint8_t* hs, cudaStream_t st);
// bf16 x bf16 GEMM, fp32 out: y [M][N] = x W^T, W in the bf16 GEMV layout (N a multiple of 32).
void gemm_bf16_p(const __nv_bfloat16* x, int M, const uint8_t* W, int N, int K, float* y, cudaStream_t st);
// The same on plain row-major bf16 weights (attn::gemv_bf16's contract, e.g. the compressor's wkv /
// wgate): rows n < N0 from W0 [N0][K] into y0 [M][N0], the rest from W1 into y1 [M][N - N0]; fp32 sums
// per pipeline stage as gemm_bf16_p; round_bf16 rounds the outputs to bf16 values. N0, N multiples of 64.
void gemm_bf16_rows_p(const __nv_bfloat16* x, int M, int K, const uint16_t* W0, const uint16_t* W1, int N0, int N,
                      float* y0, float* y1, bool round_bf16, cudaStream_t st);
// wo_a as in gemv_woa (FP8 widened exactly to bf16, block-diagonal, bf16 MMA): y bf16 [M][N] and/or
// act_quant(y) into q/qs. N and group_rows multiples of 128.
void gemm_woa_p(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, int group_rows,
                __nv_bfloat16* y, uint8_t* q, uint8_t* qs, cudaStream_t st);

// Routed experts (E <= 1024 stacked as in MoeWeights) plus the shared expert for a chunk of M rows,
// with the same result as moe(): picks bucketed by expert, grouped GEMMs per 64-row tile, each
// pick's output rounded to bf16, then per row summed in expert-id order in fp32, shared expert last.
// ws: moe_prefill_workspace_bytes(M, E) bytes (no initialisation needed).
size_t moe_prefill_workspace_bytes(int M, int E);
// Benchmark hook: tile-configuration variants of the prefill kernels (0 = the defaults; down: 1 =
// separate combine kernel, 2 = combine fused into the down kernel, 0 = by size; l2: the MoE's L2
// eviction priorities as bits, see MoePArgs::l2 in prefill.cu, -1 = the default).
struct PrefillTuning {
  int fp8 = 0, w16 = 0, up = 0, down = 0, l2 = -1;
  // the dense GEMMs' small-M tiles (0: off): narrower N tiles, so that a short chunk gives every SM work;
  // never split-K, so each output's K chain is the default tiles' (FP8 and wo_a: 1 32x64, 2 32x32; the
  // router gate: 1 32x32, 2 16x16; the shared expert's w1/w3: 32x64 in both)
  int small = 0;
  int tord = -1;  // tile order: k > 0 one tile of the most-picked experts, then k of the fewest (0: descending; -1: the default)
  int upf = -1;  // up: weights prefetched into L2 this many chunks of 256 k ahead (0: none; -1: the default)
  int dgrp = -1, dring = 3;  // down: column blocks per CTA (0: one, k_moe_down_p; -1: the default), weight steps in flight (3 or 9)
};
void prefill_tuning(const PrefillTuning& t);
struct MoePCfg {
  int phases = 31;               // 1 bucketing, 2 shared expert, 4 routed up, 8 routed down, 16 combine
  cudaStream_t side = nullptr;   // if set, the shared expert's GEMMs run there, beside the routed experts
  int col0 = 0, cols = 0;        // down and combine: output columns [col0, col0 + cols) (0: all), multiples of 128
};
void moe_prefill(const uint8_t* xq, const uint8_t* xs, int M, const int* idx, const float* wts, int E,
                 const MoeWeights& w, void* ws, float* y, cudaStream_t st, MoePCfg cfg = MoePCfg{});
// Whether moe_prefill runs the combine inside the down kernel at M rows (outputs kept in L2).
bool moe_prefill_fuses(int M);
// For chunks it does not fuse: the combine of output columns [col0, col0 + cols) (multiples of 128;
// down done for them) fused into their reduce-scatter by rows over the ring. Rank k gets rows
// [k rq, (k + 1) rq) (rows past M count as zeros): the four ranks' sums added in rank order and
// rounded to bf16 into out [rq][ld_out], at column col0 - the bits of k_moe_combine, comm's
// reduce_scatter_rows_f32 and an fp32 -> bf16 rounding, without the fp32 outputs' trip through
// memory. One kernel of `blocks` blocks of 256 threads on st, every rank in the same order.
void moe_combine_rs(const comm::DevChannel& ch, void* ws, int M, int E, int rq, int col0, int cols,
                    __nv_bfloat16* out, size_t ld_out, cudaStream_t st, int blocks = 32);

// ---- the decode step's L2 prefetch (l2pf.cu) ------------------------------------------------------
// While a collective, or the attention, leaves DRAM idle, a side stream prefetches into L2 the weights the
// step reads next (model::Options::l2pf, on by default). It writes nothing the model reads.
// A layer's weights the plan may prefetch, in the order its step reads them: wq_a + wkv, wq_b (+ the indexer's
// wq_b), wo_a, wo_b, the router gate; each up to four tensors (weights, scales; n 0: none).
enum { kL2pfWqa, kL2pfWqb, kL2pfWoa, kL2pfWob, kL2pfGate, kL2pfTargets };
struct L2pfTarget {
  const void* p[4];
  size_t n[4];
};
struct L2pfState {  // device memory: how far a layer's windows got (they run in order on one stream)
  int layer;
  uint32_t chunk;
};
// Window `window` of layer `layer`'s step at R rows, on s (forked from the main stream after the window's last
// weight kernel): 1 the FFN's all-reduce (layer: the next one), 2 the attention's all-gather, 3 the attention,
// 4 its all-reduce. arrived, finished: the ring's counters (comm::DevChannel::ready and ::seq), which end the
// prefetch once the window's collective has arrived; nullptr for the attention.
void l2pf_window(const L2pfTarget (&t)[kL2pfTargets], int layer, int window, int R, const unsigned long long* arrived,
                 const unsigned long long* finished, L2pfState* state, cudaStream_t s);

}  // namespace dsk
