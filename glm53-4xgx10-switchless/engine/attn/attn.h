// The attention side of GLM-5.3 for one rank at TP=4 (16 of the 64 heads), sm_121a: the FP8 projections
// (zai's 128x128 block FP8, BF16 activations, f32 accumulation), norms, RoPE, the cache writes, the DSA
// indexer (f32 scores, exact top-2048), absorbed sparse MLA attention, and the BF16 head and FP8 MLP GEMVs.
// Layouts, the per-rank split and the measurements: docs/attention.md. The math: docs/model.md § 2.
//
// Rows are query tokens. Every row carries its absolute position (pos[r]) and its session's cache slot
// (slot[r]; a null table: every row in slot 0). A session's rows are consecutive and ascend in position, so
// one call serves a decode step (1-16 rows, any mix of sessions), a verify step, or a prefill chunk. Every
// row computes, bit for bit, what it computes in any other call (no result depends on which other rows share
// the call). bf16 values are passed as uint16_t bit patterns. Everything is asynchronous on the stream.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace glm {

// ---- the model (config.json of zai-org/GLM-5.3) and the split at TP=4 -----------------------------------
inline constexpr int kTP = 4;
inline constexpr int kDim = 6144;           // hidden
inline constexpr int kHeadsAll = 64;
inline constexpr int kHeads = kHeadsAll / kTP;  // 16 a rank
inline constexpr int kQLora = 2048;         // q_lora_rank
inline constexpr int kKvLora = 512;         // kv_lora_rank
inline constexpr int kRope = 64;            // qk_rope_head_dim (the rotary dim; "head_dim" in config)
inline constexpr int kNope = 192;           // qk_nope_head_dim
inline constexpr int kQkDim = kNope + kRope;  // 256 a query head
inline constexpr int kVDim = 256;           // v_head_dim
inline constexpr int kLat = kKvLora + kRope;  // 576: one cache entry (latent then roped k_rope)
inline constexpr int kIdxHeads = 32, kIdxDim = 128, kIdxHeadsRank = kIdxHeads / kTP;
inline constexpr int kTopK = 2048;          // index_topk
inline constexpr int kVocab = 154880, kVocabRank = kVocab / kTP;  // 38,720 head rows a rank
inline constexpr int kDenseInter = 12288, kDenseInterRank = kDenseInter / kTP;    // layers 0-2
inline constexpr int kSharedInter = 2048, kSharedInterRank = kSharedInter / kTP;  // the shared expert
inline constexpr int kOProjK = kHeads * kVDim;  // 4096: this rank's columns of o_proj
inline constexpr int kLayers = 78;
inline constexpr float kNormEps = 1e-5f;    // input/post-attention/final RMSNorm
inline constexpr float kLatEps = 1e-6f;     // q_a_layernorm, kv_a_layernorm (class default)
inline constexpr float kIdxLnEps = 1e-6f;   // the indexer's k_norm LayerNorm
inline constexpr float kSoftmaxScale = 0.0625f;  // 256^-0.5
inline constexpr float kIdxScale = 0.08838834764831845f;    // f32(128^-0.5), the indexer's q.k scale
inline constexpr float kIdxWScale = 0.17677669529663687f;   // f32(32^-0.5), the head weights' scale
inline constexpr int kMaxRows = 16;         // decode/verify rows a call (GEMVs, decode indexer)

// full(l) <=> max(l - 2, 0) % 4 == 0 (index_topk_freq 4, index_skip_topk_offset 3): 0, 1, 2, 6, 10, ..., 74.
constexpr bool full_indexer(int l) { return ((l - 2 > 0 ? l - 2 : 0) % 4) == 0; }
// the full layer whose top-k layer l uses (itself when full)
constexpr int index_source(int l) { return full_indexer(l) ? l : index_source(l - 1); }

// ---- this rank's x-projection: one FP8 GEMV of the normed layer input (docs/attention.md § split) -----------
// Rank r's rows, in order: q_a rows [512r, +512), kv_a latent rows [128r, +128), kv_a rope rows [512+16r, +16),
// and on full-indexer layers the indexer's wk rows [32r, +32). After the all-gather the four ranks' blocks lie
// in rank order: [rows][4][kXRows] (shared layers: kXRowsShared, the same without wk).
inline constexpr int kXQa = kQLora / kTP, kXKv = kKvLora / kTP, kXRope = kRope / kTP, kXWk = kIdxDim / kTP;
inline constexpr int kXRowsShared = kXQa + kXKv + kXRope;  // 656
inline constexpr int kXRows = kXRowsShared + kXWk;         // 688

// ---- FP8 / BF16 GEMV, 1..16 rows ---------------------------------------------------------------------------
// y[r][n] = sum_k x[r][k] W[n][k]. FP8 weights in the engine tile layout [N/8][K/64][32 lanes][16 B] (lane
// (g, t) holds row 8i+g, k = 64j + 16t + b at byte b), e4m3 values; f32 scales [N/rs][K/ks] (rs, ks in
// {16, 64, 128}: zai's 128x128 blocks re-cut so that every 8-row x 64-k chunk lies in one block). BF16 weights
// row-major [N][K] (no scales). x is bf16 [rows][ldx]. Accumulation: bf16 x bf16 MMAs with f32 accumulation
// over each 64-k chunk, times the chunk's scale, added in k order; split-K partials added in a fixed order.
enum class GemvOut : int {
  kBF16 = 0,    // y bf16 [rows][ldy]: a bf16 F.linear
  kF32 = 1,     // y f32 [rows][ldy] (o_proj and down partials before the all-reduce; weights_proj)
  kSwiGLU = 2,  // rows interleaved gate/up in pairs (2i gate, 2i+1 up): y bf16 [rows][ldy] at column i of
                // bf16(bf16(silu(bf16 g)) * bf16 u), the reference's down input. FP8 scales [N/2/rs][2][K/ks]:
                // rs counts intermediates (gate block row, then up's)
};
struct GemvSeg {
  const void* w = nullptr;   // e4m3 tiles or bf16 rows
  const float* s = nullptr;  // FP8 scales (null for BF16)
  int n = 0;                 // rows (multiple of 8)
  int rs = 128, ks = 128;    // FP8 scale block (rows, k)
  void* y = nullptr;
  int ldy = 0;               // y's row stride in elements
  // grouped input (the absorbed W_UK / W_UV): row block n / group_rows reads x + (n / group_rows) * x_group
  int group_rows = 0, x_group = 0;
};
inline constexpr int kMaxSegs = 4;
// fp8: true for FP8 tiles, false for BF16 rows. K: multiple of 64. splitk in {1, 2, 4}: warps of a CTA share a
// tile's k range (for small N). All segments share K, x and the output kind.
// The index weights (index_weights() below) fused into the x-projection's launch: its own CTAs after the
// GEMV's, the same arithmetic bit for bit. FP8 bf16-out GEMV of K = 6144 only (the x-projection of a full layer).
struct GemvIw {
  const uint16_t* wproj = nullptr;  // BF16 [nh][6144]
  int nh = 0;
  float* w = nullptr;               // [rows][ldw] f32
  int ldw = 0;
};
void gemv(bool fp8, const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out,
          int splitk, cudaStream_t s, bool pdl = false, const GemvIw* iw = nullptr);
// bench knob: the FP8 GEMV's loads in flight a lane and warps a CTA (-1, the default: by shape, 1 for the grouped
// W_UV and 4 otherwise; 0: 8 and 4; 1: 16, 4; 2: 24, 4; 3: 8, 2; 4: 8, 8; 5: 16, 2; configurations a split-K does
// not fit use 0). Outputs do not depend on it.
void gemv_config(int cfg);

// FP8 GEMM for prefill chunks (any number of rows; the BF16 path stays GEMV-only): the same segments, output
// kinds and split-K as gemv(), and every output equals gemv()'s at the same splitk bit for bit (the same MMAs from
// zero over each 64-k chunk, acc += s * c in k order, split-K parts added in part order). Tiles of 128 x 128 rows
// x columns (split-K: 128 x 64); a group (group_rows) must hold whole column tiles.
void gemm(const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out, int splitk,
          cudaStream_t s, bool pdl = false);
// bench knob: the GEMM tile configuration (-1, the default: 4 for a split-K, else 0; 0-5: gemm.cu). Outputs do not
// depend on it.
void gemm_config(int cfg);

// ---- norms, RoPE, cache entries ------------------------------------------------------------------------------
// RMSNorm as the reference: bf16(w * bf16(x * rsqrt(mean(x^2) + eps))), fp32 statistics. [rows][D] bf16.
void rmsnorm(const uint16_t* x, int ldx, const uint16_t* w, float eps, int D, uint16_t* y, int ldy, int rows,
             cudaStream_t s, bool pdl = false);

// RoPE table: [max_pos][32] u32, bf16 cos in the low half, bf16 sin in the high (rotary.forward's values:
// fp32 angle p * inv_freq, cos/sin in fp32, rounded to bf16). Built by tools/rope_table.py with torch itself
// (a double-precision host build differs from torch's in 29 of 2^20 x 32 x 2 values) and uploaded.

// The cache of one layer: entry [slot][pos] = 576 bf16 (latent after kv_a_layernorm, then k_rope after RoPE,
// de-interleaved as the reference writes it), 1152 B. Slot s at kv + s * slot_stride entries.
struct KvCache {
  uint16_t* kv = nullptr;
  size_t slot_stride = 0;  // entries
};
// The indexer keys of a full layer: BF16 [slot][pos][128] (256 B), or FP8 [slot][pos] = 128 e4m3 + f32 scale
// (132 B, amax/448 per key, vLLM's form). Dims 0..63 hold the RoPE'd half, de-interleaved.
struct IdxCache {
  uint8_t* k = nullptr;    // BF16: [slots][cap][128] bf16; FP8: [slots][cap][128] e4m3
  float* ks = nullptr;     // FP8: [slots][cap] f32 scales (same slot stride)
  size_t slot_stride = 0;  // entries
  bool fp8 = false;
};
inline constexpr int kIdxEntryBf16 = kIdxDim * 2, kIdxEntryFp8 = kIdxDim + 4;

// After the x-projection's all-gather: xg [rows][4][xrows] bf16 (xrows = kXRows on full-indexer layers, else
// kXRowsShared). Per row: qr = q_a_layernorm(q_a) [rows][2048]; the cache entry at (slot, pos): latent =
// kv_a_layernorm(c), k_rope = RoPE(k_rope); on full layers the index key = RoPE_0..63(LayerNorm(wk x) * w + b).
struct PrepArgs {
  const uint16_t* xg = nullptr;
  int xrows = kXRows;
  const int* pos = nullptr;
  const int* slot = nullptr;
  const uint32_t* rope = nullptr;
  const uint16_t* q_norm = nullptr;   // [2048]
  const uint16_t* kv_norm = nullptr;  // [512]
  const uint16_t* k_norm_w = nullptr; // [128], null: not a full layer
  const uint16_t* k_norm_b = nullptr;
  uint16_t* qr = nullptr;             // [rows][2048]
  KvCache kv{};
  IdxCache ik{};
};
void attn_prep(const PrepArgs& a, int rows, cudaStream_t s, bool pdl = false);

// ---- the indexer ---------------------------------------------------------------------------------------------
// Row r (position p, p >= 2048) scores every key j in [lo, hi) with j <= p of its slot:
//   I(r, j) = sum_h w[r][h] * relu(f32(q_h . k_j) * 128^-0.5)
// with q = iq [rows][32][128] (the indexer wq_b output, RoPE on dims 0..63 applied on load), w [rows][32] f32
// (x . weights_proj^T in f32, times 32^-0.5: index_weights). The q . k are bf16 MMAs with f32 accumulation in
// one fixed order (dims permuted identically for q and k), the head sum in one fixed order: decode and prefill
// produce identical scores. Then the exact top-2048 of the row's keys by (score desc, position asc): ties
// (e.g. the keys whose 32 terms are all zero) go to the lowest positions. Output: positions ascending
// [rows][2048] i32. Rows with p < 2048 get no selection (dense attention reads positions 0..p).
struct IdxQuery {
  const uint16_t* q = nullptr;  // [rows][32][128] bf16, before RoPE
  const float* w = nullptr;     // [rows][32] f32, index weights (already * 32^-0.5)
};
// x . weights_proj^T * 32^-0.5 in f32 for 32 heads (a rank's 8 heads: hd0, nh): a small kernel of its own.
void index_weights(const uint16_t* x, int rows, const uint16_t* wproj, int nh, float* w, int ldw, cudaStream_t s,
                   bool pdl = false);
// Scratch for the selection: scores [rows][max_t] f32 and the histograms/counters.
struct IdxScratch {
  float* scores = nullptr;
  size_t ld = 0;               // >= the longest scan, a multiple of 256
  uint32_t* work = nullptr;    // index_scratch_words(rows) words, zero before the first call; left zero
};
size_t index_scratch_words(int rows);
// test knob: how many candidates a row's selection keeps (8192, the default and the maximum) before it selects from
// the row's scores directly instead; results do not depend on it.
void index_candidate_cap(int cap);
// bench knob: how the decode scoring kernel loads its key tiles (-1, the default: 2 for BF16 keys, 0 for FP8; 0-4:
// indexer.cu). Results do not depend on it.
void index_score_config(int cfg);
// Decode (rows <= 16) or prefill (any rows): positions [lo, hi) of each row's scan (hi = 0: up to p + 1).
// topk [rows][2048]: ascending positions; with lo/hi a part of the positions (a rank's quarter), the part's
// own top-2048 (fewer: -1 padded), and cand_key [rows][2048] u32 the order keys for index_merge.
void index_select(const IdxQuery& q, const int* pos, const int* slot, int rows, const uint32_t* rope, IdxCache ik,
                  int lo, int hi, IdxScratch sc, int* topk, uint32_t* cand_key, cudaStream_t s, bool pdl = false);
// The merge of 4 ranks' candidates, gathered in rank order: cand_pos/cand_key [rows][4][2048] -> topk [rows][2048].
void index_merge(const int* cand_pos, const uint32_t* cand_key, const int* pos, int rows, int* topk, cudaStream_t s,
                 bool pdl = false);

// ---- absorbed sparse MLA attention ------------------------------------------------------------------------
// q_lat [rows][16][512] bf16 = W_UK_h^T q_nope_h (the absorbed W_UK GEMV), q [rows][16][256] bf16 (q_b's output;
// RoPE on its last 64 dims is applied on load). Row r attends to its slot's entries at topk[r][0..2048) when
// p >= 2048, else to positions 0..p: s = (q_lat . c + q_rope . k_rope) * 0.0625, softmax in f32, P in bf16,
// u_h = sum P c (f32 accumulation) -> bf16 out [rows][16][512] (W_UV's input). The 2048 keys go in 32 blocks of
// 64; `splits` CTAs a row share them (split-K, then a combine). The same call serves decode and prefill.
struct AttnScratch {
  float* o = nullptr;   // [rows][splits][16][512]
  float* ml = nullptr;  // [rows][splits][16][2]
  int max_splits = 0;
};
void sparse_attn(const uint16_t* q_lat, const uint16_t* q, const int* pos, const int* slot, const uint32_t* rope,
                 KvCache kv, const int* topk, int rows, int splits, AttnScratch sc, uint16_t* out, cudaStream_t s,
                 bool pdl = false);

}  // namespace glm
