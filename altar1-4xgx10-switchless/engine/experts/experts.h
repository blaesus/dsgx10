// AikidoSec/altar-1 routed experts at TP=4, one rank (milestone A3e; the GLM-5.3 engine's expert half, G3,
// specialised): routing over 168 experts, the decode expert step for 1-16 rows and the prefill expert GEMMs.
// Every dimension is altar-1's per-rank shape; nothing else is supported. Formats, layouts and numerics:
// docs/experts.md (a delta from glm53's).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace comm {
struct DevChannel;
}

namespace gx {

// ---- shapes (per rank at TP=4) ----------------------------------------------------------------------------
constexpr int kH = 6144;          // hidden (K of gate/up, N of down)
constexpr int kI = 512;           // a rank's slice of the 2048 intermediate (N of gate/up, K of down)
constexpr int kE = 168;           // routed experts a layer (altar-1: REAP-pruned from GLM-5.3's 256)
constexpr int kEWords = (kE + 31) / 32;  // words of an expert bitmap (6)
constexpr int kTopK = 8;
constexpr float kRouteScale = 2.5f;
constexpr int kMaxRows = 16;      // decode rows a step
constexpr int kMaxUnion = kMaxRows * kTopK;  // 128
constexpr int kShSlot = kMaxUnion;           // the shared expert's slot in the decode h workspace

// ---- the routed formats -----------------------------------------------------------------------------------
// A matrix [N][K] (N outputs, K inputs) is stored as blocks of 8 rows x kBK k, [N/8][K/kBK], each block a
// lossless re-tiling of the checkpoint's tensor (docs/experts.md section 2; layout.h is the executable spec):
//   NVFP4 block (8 x 512 k) 2304 B = 4 x [32 lanes][16 B] e2m1 values + [32 lanes][8 B] e4m3 scales
//   INT4  block (8 x 512 k) 2368 B = 4 x [32 lanes][16 B] stored nibbles + [8 rows][2][8] bf16 scales
//                                    + [8 rows][2][8 nibbles] zero points (compressed-tensors pack-quantized,
//                                    group 32 along K, asymmetric: W = bf16((u_w - u_z) * s))
//   BF16  block (8 x 128 k) 2048 B = 4 x [32 lanes][16 B] bf16 values
// The format is a property of a layer (every matrix of a layer's 168 experts has it): altar-1 stores INT4 on
// layers 4-76 and BF16 on 3, 77 and the MTP layer 78; NVFP4 is kept for the MTP layer's later option (DESIGN
// decision 5). Tag 1 was GLM-5.3's S3 code, retired: every function here refuses it, as any other unknown tag.
enum Fmt : uint32_t { kNVFP4 = 0, kINT4 = 2, kBF16 = 3 };
constexpr int kBlkK = 512;  // NVFP4 and INT4 blocks
constexpr int kBlkKBF16 = 128;
constexpr int kQ4Blk = 2304, kI4Blk = 2368, kB16Blk = 2048;
constexpr int kI4Group = 32;  // INT4: k a scale and zero point
constexpr bool fmt_known(uint32_t fmt) { return fmt == kNVFP4 || fmt == kINT4 || fmt == kBF16; }
constexpr int blk_k(uint32_t fmt) { return fmt == kBF16 ? kBlkKBF16 : fmt_known(fmt) ? kBlkK : 0; }
constexpr int blk_bytes(uint32_t fmt) { return fmt == kNVFP4 ? kQ4Blk : fmt == kINT4 ? kI4Blk : fmt == kBF16 ? kB16Blk : 0; }
// Bytes of a [N][K] matrix in the format; 0 for an unknown format (every caller refuses 0).
constexpr size_t mat_bytes(uint32_t fmt, int N, int K) {
  return fmt_known(fmt) ? (size_t)(N / 8) * (K / blk_k(fmt)) * blk_bytes(fmt) : 0;
}
static_assert(mat_bytes(kINT4, kI, kH) == 444 * 4096 && mat_bytes(kBF16, kI, kH) == (size_t)kI * kH * 2, "4 KiB pages");
constexpr const char* fmt_name(uint32_t fmt) {
  return fmt == kNVFP4 ? "NVFP4" : fmt == kINT4 ? "INT4" : fmt == kBF16 ? "BF16" : "unknown";
}

// One routed matrix of one expert: where its blocks are in the layer's pool, its per-tensor F32 scale (NVFP4's
// weight_scale_2; exactly 1 for INT4 and BF16) and its format; the rest is zero.
struct alignas(64) MatDesc {
  uint64_t off;      // byte offset of block (0, 0) in the pool, a multiple of 256
  float s2;          // NVFP4: weight_scale_2; INT4, BF16: 1.0
  uint32_t fmt;      // Fmt
  uint32_t rsv[12];  // zero
};
static_assert(sizeof(MatDesc) == 64, "MatDesc is 64 B");

// An FP8 shared expert fused into the routed kernels (GLM-5.3's: zai's 128x128 block-scaled e4m3, the rank's
// slices): gate, up [512][6144] and down [6144][512] in the "tile_fp8" layout [N/8][K/128][32 lanes][32 B]
// (lane (g, t): row 8 n8 + g, k = 128 j + 64 q + 16 t + i at byte 16 q + i), scales F32 row-major
// [N/128][K/128]. altar-1's shared experts are BF16 and run outside these kernels (attn's BF16 GEMV / GEMM),
// so for altar-1 every pointer is null: decode and prefill then run no shared expert, and ys is an input
// (the shared expert's f32 output) that k_down only forwards into a posted collective's buffer.
//
// A BF16 shared expert fused into the decode kernels (altar-1's, DecodeCfg-independent; moe_decode only, never
// moe_prefill): bgu the interleaved gate/up [1024][6144] (rows 2i gate, 2i + 1 up) and bd the down [6144][512],
// both in attn's kBF16T lane order ([N/8][K/64][2 halves][32 lanes][8 values], byte for byte the BF16 expert
// block's order). k_up computes its h exactly as attn's gemv(kSwiGLU, split-K 4) does (each 8-row tile's K in four
// parts of 1536 k, each part one MMA chain in k order, the parts added in part order, the same SwiGLU epilogue)
// and k_down its ys as gemv(kF32, split-K 1) (one MMA chain over the 512 k): the same bits as the two GEMVs.
// ys is its own f32 partial (the all-reduce's second contribution, y = bf16(bf16(sum yr) + bf16(sum ys))), or with
// DecodeCfg::ysum added into yr in f32 after the routed experts (decision 8; the runner's default).
struct SharedW {
  const uint8_t *g = nullptr, *u = nullptr, *d = nullptr;
  const float *gs = nullptr, *us = nullptr, *ds = nullptr;
  const uint8_t *bgu = nullptr, *bd = nullptr;  // the BF16 shared expert (decode only)
  bool fused() const { return g != nullptr; }
  bool bf16() const { return bgu != nullptr; }
};
constexpr int kShTiles = 2 * kI / 8;  // the BF16 shared gate/up's 8-row tiles (128)
constexpr int kShParts = 4;           // its split-K (attn's gemv at splitk 4)

// One MoE layer's weights on a rank.
struct LayerW {
  const uint8_t* pool = nullptr;  // the routed experts' blocks
  const MatDesc* desc = nullptr;  // [kE][3]: gate, up, down
  SharedW sh;
  uint32_t fmt = kINT4;           // the layer's expert format: every desc[e][m].fmt equals it (checked)
};

// ---- routing ------------------------------------------------------------------------------------------------
// logits f32 [M][168] (the router GEMV's, fp32(x) . fp32(gate.weight)^T), bias f32 [168] ->
// idx [M][8] (descending selection score, ties to the lower id), w [M][8] (s / (sum + 1e-20) * 2.5).
void route(const float* logits, const float* bias, int M, int* idx, float* w, cudaStream_t st, bool pdl = true);

// ---- decode -------------------------------------------------------------------------------------------------
// The expert step for M <= 16 rows: x bf16 [M][6144] (the post-attention-normed input), idx/w from route().
// ws: decode_ws_bytes(), zeroed once before the first call (its task counters reset themselves). With
// DecodeCfg::logits, the routing runs inside the kernels and idx, w are ignored.
// Writes yr = sum over the rows' experts in expert-id order of w * down(h) and ys = the shared expert's
// output, both f32 [M][6144] partial sums of this rank (the ring all-reduces them), and h's in ws.
struct DecodeCfg {
  // up-kernel variant (decode.cu, kUp). 0 (the default) splits K over 6 warps; kUpPrefillOrder and the other
  // "wk1" variants sum each strip's K in one warp, in prefill's order: their h, yr and ys equal moe_prefill's bit
  // for bit (tested; docs/experts.md section 13).
  int up = 0;
  int down = 0;   // down-kernel variant
  int up_ctas_per_sm = 0;  // 0: as many as fit
  bool pdl = true;
  bool dyn = true;         // k_up claims tasks past the first wave from a counter (load balance)
  // k_up (WK > 1 variants): load the next routed task's first blocks under the current task's epilogue (after its
  // partials' barrier) instead of at its start; the same loads, order and MMAs, so the same bits
  bool xpf = true;
  int parts = 3;           // bench: 1 up only, 2 down only, 3 both
  // Fused routing: with logits (the router GEMV's f32 [M][168]) and bias set, each CTA of both kernels routes the
  // rows itself (route()'s code, the same bits) and idx / w are not read (may be null): no route() launch between
  // the router GEMV and the experts.
  const float* logits = nullptr;
  const float* bias = nullptr;
  // Posted outputs: with a ring channel, k_down writes yr and ys into the send buffer of the channel's next
  // collective (yr's M rows, then ys's) instead of the yr / ys arguments, and posts it from its epilogue, the
  // producer half of comm::glm::allreduce2_add_norm (its consumer half: allreduce2_add_norm_posted). The same bits.
  const comm::DevChannel* post = nullptr;
  // ysum (glm53 DESIGN decision 8, the contract): k_down adds the shared expert's f32 output ys (fused: computed in
  // k_down; else the caller's, e.g. the BF16 GEMV's) into yr after the routed experts, one __fadd_rn a value, and
  // yr is the only output: posted, M rows (the producer half of comm::glm::allreduce_add_norm). ys must be given
  // (the fused shared expert's scratch). false: ys a separate partial, y = bf16(bf16(sum yr) + bf16(sum ys)).
  bool ysum = false;
};
constexpr int kUpPrefillOrder = 6;  // DecodeCfg::up: "nt1 wk1 p2", decode = prefill bit for bit
size_t decode_ws_bytes();
void moe_decode(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int M, void* ws, float* yr,
                float* ys, cudaStream_t st, const DecodeCfg& cfg = DecodeCfg());
int decode_up_variants();
int decode_down_variants();
const char* decode_up_name(int v);
const char* decode_down_name(int v);
// The output after the all-reduce: out = bf16(bf16(R) + bf16(S)), as the reference adds the shared expert.
void finalize(const float* R, const float* S, int M, __nv_bfloat16* out, cudaStream_t st, bool pdl = true);

// ---- prefill ------------------------------------------------------------------------------------------------
// A chunk of T rows (T <= kMaxChunk): the rows' picks bucketed by expert, gate/up and down as grouped GEMMs on
// the bf16 tensor cores with the weights dequantised to bf16 in shared memory by the decode kernels' own
// functions (BF16: copied), each pick's down output kept in f32 and combined per row in expert-id order (the
// decode step's order). A fused FP8 shared expert (LayerW::sh) as a dense FP8 GEMM. Outputs as moe_decode:
// yr, ys f32 [T][6144]. (GLM-5.3's FP4-activation path, which changed outputs, is not in this engine.)
constexpr int kMaxChunk = 8192;
struct PrefillCfg {
  bool shared = true;                          // run the fused shared expert (if the layer has one)
  // bench diagnostics, INT4 layers only (wrong results): the gate/up kernel (up 1) with 1 no weights, 2 no MMAs, 4 no
  // A loads; the chained down (down 1) with 8 no adds into yr, 16 no waits (bench_experts --diag; glm53's other
  // combinations are not built here)
  int diag = 0;
  int ddiag = 0;
  // The routed down and combine: 1 (default) the chained down, which adds each pick's f32 sums into yr in the
  // row's expert-id order as its tile finishes (no per-pick buffer); 0 GLM-5.3's kernels of 2026-10-13
  // (per-pick f32 sums to a buffer, then a combine). The same bits (tested). (glm53's 2, the warp-specialised
  // chained down, measured slower, is not in this engine.)
  int down = 1;
  // The routed gate/up: 1 (default) warp-specialised (8 MMA warps, 8 warps loading A and decoding B), 0 the
  // kernel of 2026-10-13. The same bits (tested). (glm53's 2, 64 x 64 warps, measured slower, is not here.)
  int up = 1;
  // The shared expert's down: 128 (default) rows a CTA (k_pdown<true, 4>), or 64, that of 2026-10-14 (more FP8
  // decodes a row). The same bits (tested).
  int sh_rows = 128;
  int yr_kb = 0;               // chained down: the L2 budget of the yr columns in flight, KB (0: the default)
  int chain_q = 4;             // chained down: passes claimed round-robin (fewer waits, h re-read from L2)
  cudaStream_t side = nullptr;  // if set, the shared expert runs on it (forked and joined with events)
};
// ws: prefill_ws_bytes(T) for the largest chunk, zeroed once before the first call (its counters reset
// themselves); a call's results do not depend on the other rows of the chunk.
size_t prefill_ws_bytes(int T);
void moe_prefill(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int T, void* ws, float* yr,
                 float* ys, cudaStream_t st, const PrefillCfg& cfg = PrefillCfg());
// Per-phase timing hooks for the bench: the same work in separate calls on the same workspace.
// The workspace's intermediates after a call (tests): per row t its picks in expert-id order (rpos: bucket
// positions, re: experts), h [position][512] bf16 and the fused shared expert's hsh [T][512].
struct PrefillView {
  const int *rpos, *re;
  const __nv_bfloat16 *h, *hsh;
};
PrefillView prefill_view(void* ws, int T);
enum PrefillPhase { kPhBucket = 1, kPhUp = 2, kPhDown = 4, kPhCombine = 8, kPhShared = 16, kPhAll = 31 };
void moe_prefill_phases(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int T, void* ws,
                        float* yr, float* ys, cudaStream_t st, const PrefillCfg& cfg, int phases);

}  // namespace gx
