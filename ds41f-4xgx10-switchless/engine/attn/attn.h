// Attention-side kernels for one rank at TP=4: RoPE, q/kv norms, the sliding-window and compressed
// KV state, the compressor, the indexer (scores, top-k, layer 20's candidate pool), sparse attention
// and mHC. The FP8 projections around them (wq_a, wq_b, wkv, the indexer's wq_b, wo_a, wo_b) belong
// to engine/kernels; these kernels start from their bf16 outputs.
//
// Numerics mirror ref/inference/model.py with kernel.py (see docs/findings.md#numerics-of-the-reference).
// Rows are query tokens; every row carries its absolute position, so one call serves a decode step,
// a multi-row verify step or a prefill chunk. Everything is asynchronous on the given stream.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace attn {

inline constexpr int kDim = 5120;      // model width
inline constexpr int kHead = 512;      // latent / head dim (MQA: one latent shared by all heads)
inline constexpr int kRope = 64;       // rotary dims, the last 64 of a head
inline constexpr int kHeads = 16;      // query heads per rank (64 / TP4)
inline constexpr int kQLora = 1280;
inline constexpr int kWindow = 128;
inline constexpr int kRing = 256;      // window ring slots: window + room for verify rows
inline constexpr int kIdxHeads = 32, kIdxDim = 128;
inline constexpr int kHc = 4, kMix = 24, kHcDim = kHc * kDim;
inline constexpr float kNormEps = 1e-20f;  // rms_norm_eps
inline constexpr float kHcEps = 1e-6f;

// ---- RoPE ---------------------------------------------------------------------------------------
// Table of [max_pos][32] (cos, sin), built with the reference's fp32 operation order, so it is
// bit-identical to model.py's freqs_cis. compressed=false: theta 1e4 (layers 0-1, DSpark);
// compressed=true: theta 1.6e5 with YaRN (factor 16, original 65536, beta 32/1), every ratio>0 layer.
void rope_table_build(float2* table, int max_pos, bool compressed, cudaStream_t s);

// ---- state --------------------------------------------------------------------------------------
// Session slots (batching, docs/batching.md): a state struct can stand for an array of sessions' states,
// slot s at its pointers + s * stride (bytes, multiples of 16; CompressorCarry: floats, multiples of 4;
// IndexKeys: below), with `slot` (device, [rows]) giving each row its session's slot. A session's rows are consecutive and ascend in position; sessions follow one
// another in any order, at unrelated positions. A null table: every row in slot 0, i.e. at the pointers.
// Every row computes, bit for bit, what it computes in a step of its session alone.
struct Window {          // sliding-window KV, slot = (position - base) & mask: a layer's ring (the
  uint8_t* kv;           // defaults), or a prefill chunk's linear buffer (mask -1, base = first position)
  uint8_t* sc;           // kv [slots][512] e4m3, sc [slots][16] ue8m0 (one per 32)
  int mask = kRing - 1;
  int base = 0;
  const int* slot = nullptr;  // session slots (above)
  size_t kv_stride = 0, sc_stride = 0;
};
struct Compressed {      // a kv-source layer's compressed KV: entry j stands for tokens [j*ratio, (j+1)*ratio)
  uint8_t* kv;           // [cap][256] e2m1 pairs, low nibble = even element
  uint8_t* sc;           // [cap][32]  e4m3, one per 16
  int cap;
  const int* slot = nullptr;  // session slots (above)
  size_t kv_stride = 0, sc_stride = 0;
};
struct IndexKeys {       // the indexer keys of the same entries
  uint8_t* k;            // [cap][64] e2m1 pairs, words transposed 4x4 (attn_common.cuh fp4x128_byte)
  uint8_t* sc;           // [cap][4]  ue8m0, one per 32
  int cap;
  const int* slot = nullptr;  // session slots (above); the indexer's sessions are this table's
  size_t k_stride = 0, sc_stride = 0;  // whole entries apart: k_stride = 16 * sc_stride
};

// ---- q / kv preparation -------------------------------------------------------------------------
// RMSNorm(1280) of wq_a's output: qr = q_norm(qa). bf16 in and out.
void q_norm(const uint16_t* qa, const uint16_t* w, uint16_t* qr, int rows, cudaStream_t s);
// RoPE on the last 64 dims of each of `heads` heads of width `dim`, in place (bf16), at pos[row].
// The attention kernels fuse this; it is exposed for tests and for the DSpark path.
void rope_rows(uint16_t* x, int rows, int heads, int dim, const int* pos, const float2* table,
               bool inverse, cudaStream_t s);
// The window KV of `rows` new tokens: kv_norm(wkv output) -> RoPE at pos -> fp8 per 32 with a
// power-of-two scale -> ring slot pos & (kRing-1). Optionally also returns the dequantised bf16.
void kv_write(const uint16_t* kv_pre, const uint16_t* norm_w, const int* pos, const float2* table,
              Window win, uint16_t* kv_out_or_null, int rows, cudaStream_t s);

// y[r][n] = x[r] . W[n] for bf16 x [rows][K] and bf16 W [N][K] (rows n < N0 from W0 into y0 [rows][N0],
// the rest from W1 into y1 [rows][N-N0]), fp32 out; round_bf16 rounds to bf16 values (a bf16 F.linear).
void gemv_bf16(const uint16_t* x, int rows, int K, const uint16_t* W0, const uint16_t* W1, int N0, int N, float* y0,
               float* y1, bool round_bf16, cudaStream_t s);

// ---- compressor (kv-source layers 2, 8, 14: ratio 2; layer 20: ratio 1) ---------------------------
struct CompressorWeights {
  int ratio;                 // 2 or 1
  const uint16_t* wkv;       // [512][5120] bf16 (ratio 2 runs it in fp32, as model.py)
  const uint16_t* wgate;     // [512][5120] bf16, ratio 2 only
  const uint16_t* norm;      // [512] compressor RMSNorm
  const uint16_t* wk;        // [128][512] indexer key projection
  const uint16_t* knorm;     // [128] indexer key RMSNorm
};
struct CompressorCarry {     // ratio 2: the last 16 positions' projections, slot = position & 15,
  float* kv;                 // so a group can complete in a later call (and after rejected verify rows)
  float* score;              // [16][512] each
  const int* slot = nullptr;  // session slots (above), stride in floats (kv and score alike)
  size_t stride = 0;
};
// Projections of `rows` consecutive positions: kv (and score for ratio 2) in fp32, [rows][512] each.
// For ratio 1, kv holds the bf16-rounded projection (model.py runs that one in bf16).
void compressor_project(const uint16_t* x, const CompressorWeights& w, float* kv, float* score, int rows,
                        cudaStream_t s);
// Emit every entry completed by these rows (row r completes entry pos[r]/ratio when (pos[r]+1) % ratio
// == 0): pool (ratio 2), RMSNorm, then the index key (wk, k_norm, RoPE at entry*ratio, fp4/e8m0) and the
// KV entry (RoPE at entry*ratio, fp4/e4m3). Then records the rows' projections in the carry.
// Tests can pass debug_latent to get each entry's normalised latent (pre-RoPE) at [entry - debug_base].
void compressor_emit(const float* kv, const float* score, const int* pos, int rows, const CompressorWeights& w,
                     CompressorCarry carry, const float2* table, Compressed ckv, IndexKeys ik,
                     uint16_t* debug_latent, int debug_base, cudaStream_t s);

// ---- indexer ------------------------------------------------------------------------------------
// Full layers (2, 8, 14, 20) score every compressed position a row can see; layer 20 also picks the
// candidate pool (the best blocks of 8 by block maximum); Reindex layers (24, 28, 32, 36) score only
// a row's pool; the other layers reuse their source's indices. Scores are handled as 16-bit order
// keys: the bf16 score made sortable (key_from_bf16), 0 for "not a candidate".
//
// Rows are in ascending position order (a decode step, verify rows, or a prefill chunk); row r sees
// compressed positions [0, (pos[r]+1)/ratio). Splitting across ranks: decode splits the positions
// (IndexScan part/nparts, then index_merge over the gathered candidates), prefill splits the rows.
// With session slots (ik.slot), a row scans its own slot's keys and "the last row" is its session's.
struct IndexQ {              // one call's prepared queries
  uint8_t* q;                // [rows][32][64] e2m1, same word order as the keys
  uint32_t* sc;              // [rows][32] four ue8m0 scales
  float* w;                  // [rows][32] head weights: bf16(weights_proj(x)) * 2^-6
};
// RoPE (dims 64..127 at pos) -> fp4 with e8m0 per 32 of the indexer queries iq_pre [rows][32][128]
// (bf16, from the indexer's wq_b), and the head weights from wproj = gemv_bf16(x, weights_proj)
// [rows][32]. debug_q (optional) gets the dequantised queries.
void index_q_prep(const uint16_t* iq_pre, const float* wproj, const int* pos, const float2* table, IndexQ q,
                  uint16_t* debug_q, int rows, cudaStream_t s);

struct IndexScan {           // which compressed positions a call scores ("scan index" i -> position)
  int part = 0, nparts = 1;  // dense: part `part` of `nparts` (64-aligned) of [0, n), n = the last row's length
  const int* blocks = nullptr;  // or each row's candidate blocks [rows][nblocks] (ascending, -1 padded):
  int nblocks = 0;              //    i -> blocks[r][i/8]*8 + i%8
};
inline constexpr int kScoreAlign = 256;  // pad keys_stride to this; dense max_scan = positions per part
// index_score and index_topk take one session (ik.slot null); the fused kernels below take several.
// Scores -> keys[r][i] for the scan; with bkeys (dense only), also block maxima bkeys[r][i/8] with the
// block of each row's newest position pinned to 0xFFFF (the reference's +inf). max_scan bounds the scan
// length (dense: the largest part; blocks: nblocks*8) and sizes the grid, so a captured graph covers it.
void index_score(const IndexQ& q, const int* pos, int rows, int ratio, IndexKeys ik, const IndexScan& scan,
                 int max_scan, uint16_t* keys, int keys_stride, uint16_t* bkeys, int bkeys_stride, cudaStream_t s);
// Top-k per row as torch.topk picks it (highest key, ties to the lowest position): the
// min(kmax, candidates) chosen positions in ascending order with their keys, then -1 / 0 padding.
// From a scan's keys: positions. From block maxima (pool = true): block indices.
void index_topk(const uint16_t* keys, int keys_stride, const int* pos, int rows, int ratio, const IndexScan& scan,
                int max_scan, bool pool, int kmax, int* out_pos, uint16_t* out_key, int out_stride, cudaStream_t s);
// The same selection over gathered candidates: cand_pos/cand_key [rows][n] (each part's list in rank
// order, so positions ascend), e.g. 4 ranks x 512 positions or 4 x 2048 pool blocks.
void index_merge(const int* cand_pos, const uint16_t* cand_key, int n, int rows, int kmax, int* out_pos,
                 uint16_t* out_key, int out_stride, cudaStream_t s);

// ---- sparse attention ---------------------------------------------------------------------------
struct AttnScratch {         // split-K partials, for splits > 1
  float* o;                  // [rows][max_splits][16][512]
  float* ml;                 // [rows][max_splits][16][2]
  int max_splits;
};
// Per-row window bounds, inclusive positions of `win` (device arrays [rows]); nullptr = the default,
// hi = pos and lo = hi - 127. With either given, hi - lo < 192 (3 blocks). DSpark draft rows (block of 5
// at start+1..start+5, their keys written to the ring at those positions) pass lo = max(0, start-127),
// hi = start+5: the last 128 main positions and the whole block, non-causal, as get_dspark_topk_idxs.
struct WinBounds {
  const int* lo = nullptr;
  const int* hi = nullptr;
};
// One layer's attention for `rows` query rows on this rank's 16 heads. Row r attends to its window,
// positions max(0, lo)..hi of `win` (above), and to the compressed entries topk[r][0..kmax) of `ckv`
// (-1 = none; kmax a multiple of 64, 0 for layers without compression). q_pre is wq_b's output
// [rows][16][512] (RoPE at pos is applied on load); attn_sink [16]; out [rows][16][512] bf16 gets o with
// the inverse RoPE applied, ready for wo_a. `splits` CTAs share a row's 64-key blocks (decode: 2 window
// blocks + kmax/64), merged by a second kernel; prefill uses 1. pdl: both are PDL launches (see the decode
// path). (Merging in the main launch, by each row's last CTA, measured slower: one CTA reads every split.)
void sparse_attn(const uint16_t* q_pre, const int* pos, WinBounds wb, int rows, const float2* table, Window win,
                 Compressed ckv, const int* topk, int kmax, const float* sink, uint16_t* out, int splits,
                 AttnScratch scratch, cudaStream_t s, bool pdl = false);

// ---- mHC ----------------------------------------------------------------------------------------
// The residual stream is x [rows][4][5120] bf16 (kHcDim = 20480 flattened). A sublayer's mixes are
// sinkhorn(hc_fn . x * rsqrt(mean(x^2) + eps)), computed in two steps so the ranks can share the
// projection: hc_partial sums columns [k0, k1) of the flattened stream into out[rows][25] (the 24 dot
// products with hc_fn [24][20480] fp32, then the sum of squares) through per-chunk partials (chunks
// [hc_chunks(k0,k1)][rows][25]) added in a fixed order. One rank alone: [0, 20480). Split: rank r takes
// copy r, [5120r, 5120r + 5120), and the 25 floats per row ride the next sublayer's all-reduce.
int hc_chunks(int k0, int k1);
void hc_partial(const uint16_t* x, int rows, const float* fn, int k0, int k1, float* chunks, float* out,
                cudaStream_t s);
// Summed partials -> pre [rows][4], post [rows][4], comb [rows][4][4] (torchref hc_split_sinkhorn, 20 iters).
void hc_finish(const float* partial, int rows, const float* scale, const float* base, float* pre, float* post,
               float* comb, cudaStream_t s);
// y = bf16(sum_c pre[c] x[c]) [rows][5120], then RMSNorm(5120) with norm_w (attn_norm / ffn_norm) when
// given; y_raw (optional) gets the value before the norm.
void hc_pre(const uint16_t* x, const float* pre, int rows, const uint16_t* norm_w, uint16_t* y, uint16_t* y_raw,
            cudaStream_t s, bool pdl = false);
// y[j] = bf16(post[j] sub + sum_i comb[i][j] res[i]): the sublayer output sub [rows][5120] back into the stream.
// pdl: a PDL launch (the decode path; the layer runner fuses this with the sublayer's all-reduce).
void hc_post(const uint16_t* sub, const uint16_t* res, const float* post, const float* comb, int rows, uint16_t* y,
             cudaStream_t s, bool pdl = false);

// ---- the fused decode path ----------------------------------------------------------------------
// A decode or verify step (rows <= kMaxDecodeRows) in few launches, each bit-identical to the kernels above
// that it replaces (tests/fused_tests.cu). With pdl, a launch is a programmatic dependent launch: it loads
// its weights, then waits for the previous kernel in the stream (griddepcontrol.wait), so a chain of
// them overlaps launch latency with work. Per layer: hc_project + hc_mix (hc pre) per sublayer, and
// [compressor_project_rows] -> decode_prep -> [index_fused -> [index_merge2]] -> sparse_attn (2 launches),
// around the FP8 GEMVs of engine/kernels, the collectives and hc_post (the runner fuses it with the
// sublayer's all-reduce). tests/bench_step.cu times the whole step; README.md lists every launch's I/O.
// Up to kDecodeGroup rows a launch runs the kernels it ran before B4; from 9 rows the ones whose shared memory
// or register tiles hold the rows (compressor_project_rows, decode_prep's head weights, index_dense) take them
// in groups of kDecodeGroup, each row computing what it computes in a call of at most kDecodeGroup rows.
// A session has at most kDecodeGroup rows in a call (decode_prep below).
inline constexpr int kMaxDecodeRows = 16;
inline constexpr int kDecodeGroup = 8;

// mHC, kernel 1 of a sublayer: hc_partial's chunks [(k1 - k0) / 512][rows][25] of the stream x over columns
// [k0, k1) (multiples of 512; one rank's share or all), not summed. fn: the whole [24][20480] (fn_ld 0), or
// only columns [k0, k1) as [24][fn_ld] (a rank's slice in the weight files: fn_ld = 5120).
void hc_project(const uint16_t* x, int rows, const float* fn, int k0, int k1, float* chunks, bool pdl,
                cudaStream_t s, int fn_ld = 0);
// mHC, kernel 2: the sum of nchunks partials (hc_partial's order; 1 when summed already, e.g. all-reduced),
// hc_finish, hc_pre with the norm into y [rows][5120], and when q is given its act_quant: q [rows][5120]
// e4m3, qs [rows][160] ue8m0 (the FP8 GEMVs' input). pre_in: the pre-mix y collapses the stream with.
// model.py's Block uses the previous sublayer's (the attention the previous layer's FFN's, the FFN this
// layer's attention's; the first sublayer the identity [1, 0, 0, 0]); null = the one computed here. pre gets
// this sublayer's own, for the next; pre_in may be pre itself (each is read before it is written).
void hc_mix(const uint16_t* x, const float* chunks, int nchunks, int rows, const float* scale, const float* base,
            const uint16_t* norm_w, float* pre, float* post, float* comb, uint16_t* y, uint8_t* q, uint8_t* qs,
            bool pdl, cudaStream_t s, const float* pre_in = nullptr);
// compressor_project for output rows [n0, n0 + n) of wkv (and the same rows of wgate, ratio 2), e.g. one
// rank's share: kv/score [rows][n].
void compressor_project_rows(const uint16_t* x, const CompressorWeights& w, int n0, int n, float* kv, float* score,
                             int rows, bool pdl, cudaStream_t s);
// The rest of a step's preparation in one launch: q_norm of the gathered qa with its act_quant (wq_b's
// input; qr optional), kv_write of the gathered kv_pre, compressor_emit (and carry) of the gathered
// projections (ckv_proj null: none; cw.wkv/wgate unused), and the indexer's head weights wp = bf16(x . wproj)
// (wproj null: none).
// A session's rows in one call: at most kDecodeGroup. A kv-source layer's emit and carry run in one launch, so
// with 16 rows of a session its first row's read of the carry slot (p0 - 1) & 15 would race with its 16th
// row's write of the same slot ((p0 + 15) & 15). Without a slot table every row is one session and
// decode_prep dies past kDecodeGroup rows; with one, the step's builder (the runner's StepIn) must check it.
struct DecodePrep {
  const int* pos = nullptr;
  const float2* table = nullptr;
  const uint16_t* qa = nullptr;  // [rows][1280]
  const uint16_t* q_norm = nullptr;
  uint8_t* qq = nullptr;  // [rows][1280] e4m3
  uint8_t* qs = nullptr;  // [rows][40] ue8m0
  uint16_t* qr = nullptr;
  const uint16_t* kv_pre = nullptr;  // [rows][512]
  const uint16_t* kv_norm = nullptr;
  Window win{};
  const float* ckv_proj = nullptr;  // [rows][512] each
  const float* cscore = nullptr;
  CompressorWeights cw{};
  CompressorCarry carry{};
  Compressed ckv{};
  IndexKeys ik{};
  const uint16_t* x = nullptr;  // [rows][5120], the normed sublayer input
  const uint16_t* wproj = nullptr;
  float* wp = nullptr;  // [rows][32]
};
void decode_prep(const DecodePrep& a, int rows, bool pdl, cudaStream_t s);
// DSpark's window writes of newly accepted positions into up to 4 windows at once: rows [0, *count) of each
// set's kv_pre ([max_rows][512] bf16, wkv's output) at positions *pos0 + r, as kv_write (count and pos0 live
// on the device, so one graph serves any count).
// pos (device, [max_rows]): each row's own position instead, negative for none, e.g. several sessions' accepted
// rows in one call (B3). With session slots (win.slot), row r goes into its slot's window.
struct KvSet {
  const uint16_t* kv_pre = nullptr;
  const uint16_t* norm_w = nullptr;
  Window win{};
};
inline constexpr int kMaxKvSets = 4;
struct KvSets {
  KvSet s[kMaxKvSets];
  int n = 0;
};
void kv_write_sets(const KvSets& sets, int max_rows, const int* count, const int* pos0, const float2* table, bool pdl,
                   cudaStream_t s, const int* pos = nullptr);
// One selection's output: the top kmax (positions, or pool blocks) and their keys, rows `stride` apart.
struct IndexSel {
  int kmax = 0;
  int* pos = nullptr;
  uint16_t* key = nullptr;
  int stride = 0;
};
// index_q_prep, index_score and index_topk in one launch (the last CTA of each row selects): top from the
// scan's keys; pool (pos non-null; dense scans with bkeys) from the block maxima. counters: [rows] ints,
// zero before and left zero.
void index_fused(const uint16_t* iq_pre, const float* wproj, const int* pos, const float2* table, int rows, int ratio,
                 IndexKeys ik, const IndexScan& scan, int max_scan, uint16_t* keys, int keys_stride, uint16_t* bkeys,
                 int bkeys_stride, IndexSel top, IndexSel pool, int* counters, bool pdl, cudaStream_t s);
// index_fused of a whole dense scan (IndexScan{}) for all rows at once, sized for long contexts: the same
// outputs, bit for bit. One CTA per SM at most, each scoring its share of the positions for every row; the
// selections start from histograms the scoring CTAs make. The grid is one CTA per SM and the scan follows
// pos, so the launch is the same for every context; max_scan only checks the strides. bkeys: required, the block maxima
// are written for every scan (the top's selection starts from them). scratch: index_dense_scratch() bytes,
// zero before and left zero. cand: the most candidates a selection gathers in shared memory (-1: all it has
// room for).
size_t index_dense_scratch();
void index_dense(const uint16_t* iq_pre, const float* wproj, const int* pos, const float2* table, int rows, int ratio,
                 IndexKeys ik, int max_scan, uint16_t* keys, int keys_stride, uint16_t* bkeys, int bkeys_stride,
                 IndexSel top, IndexSel pool, uint32_t* scratch, bool pdl, cudaStream_t s, int cand = -1);
// index_fused of a block-list scan (layer 20's pool: each row's nblocks blocks of 8, nblocks * 8 <= 32768)
// on index_dense's grid, the same outputs bit for bit: a row's blocks are its own, so a row group is one row.
void index_dense_blocks(const uint16_t* iq_pre, const float* wproj, const int* pos, const float2* table, int rows,
                        int ratio, IndexKeys ik, const int* blocks, int nblocks, uint16_t* keys, int keys_stride,
                        IndexSel top, uint32_t* scratch, bool pdl, cudaStream_t s, int cand = -1);
// index_merge of the gathered candidates, and of the gathered pool candidates when pool.pos is given, in
// one launch.
void index_merge2(int rows, const int* cand_pos, const uint16_t* cand_key, int n, IndexSel top, const int* pcand_pos,
                  const uint16_t* pcand_key, int pn, IndexSel pool, bool pdl, cudaStream_t s);

}  // namespace attn
