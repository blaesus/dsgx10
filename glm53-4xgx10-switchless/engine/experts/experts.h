// GLM-5.3 routed experts at TP=4, one rank (milestone G3, the expert half): routing, the decode expert step
// for 1-16 rows and the prefill expert GEMMs. Every dimension is GLM-5.3's per-rank shape; nothing else is
// supported. Formats, layouts and numerics: docs/experts.md.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gx {

// ---- shapes (per rank at TP=4) ----------------------------------------------------------------------------
constexpr int kH = 6144;          // hidden (K of gate/up, N of down)
constexpr int kI = 512;           // a rank's slice of the 2048 intermediate (N of gate/up, K of down)
constexpr int kE = 256;           // routed experts a layer
constexpr int kTopK = 8;
constexpr float kRouteScale = 2.5f;
constexpr int kMaxRows = 16;      // decode rows a step
constexpr int kMaxUnion = kMaxRows * kTopK;  // 128
constexpr int kShSlot = kMaxUnion;           // the shared expert's slot in the decode h workspace

// ---- the routed formats -----------------------------------------------------------------------------------
// A matrix [N][K] (N outputs, K inputs) is stored as blocks of 8 rows x 512 k, [N/8][K/512], each block a
// lossless re-tiling of the canonical tensor (docs/experts.md, "The layout the converter emits"):
//   NVFP4 block 2304 B = 4 x [32 lanes][16 B] e2m1 values + [32 lanes][8 B] e4m3 scales
//   S3    block 1792 B = 3 planes (P0, P1, Q) x [32 lanes][16 B] + [32 lanes][8 B] scales (bit 7: subset)
enum Fmt : uint32_t { kNVFP4 = 0, kS3 = 1 };
constexpr int kBlkK = 512;
constexpr int kQ4Blk = 2304, kS3Blk = 1792;
constexpr size_t mat_bytes(uint32_t fmt, int N, int K) {
  return (size_t)(N / 8) * (K / kBlkK) * (fmt == kNVFP4 ? kQ4Blk : kS3Blk);
}

// One routed matrix of one expert: where its blocks are in the layer's pool, its per-tensor F32 scale, its
// format and, for S3, the bf16 values of its two subsets' eight codes as prmt tables (tab[4s .. 4s + 3] =
// low bytes of codes 0-3, of 4-7, high bytes of 0-3, of 4-7; the loader builds them, s3_tables()).
struct alignas(64) MatDesc {
  uint64_t off;      // byte offset of block (0, 0) in the pool, a multiple of 256
  float s2;          // weight_scale_2
  uint32_t fmt;      // Fmt
  uint32_t tab[8];   // S3 only
  uint32_t nib[2];   // S3 only: code -> e2m1 nibble of each subset, 4 bits a code (the FP4 prefill path)
  uint32_t pad[2];
};
static_assert(sizeof(MatDesc) == 64, "MatDesc is 64 B");

// The FP8 shared expert, zai's 128x128 block-scaled e4m3 (the rank's slices): gate, up [512][6144] and down
// [6144][512] in the "tile_fp8" layout [N/8][K/128][32 lanes][32 B] (lane (g, t): row 8 n8 + g,
// k = 128 j + 64 q + 16 t + i at byte 16 q + i), scales F32 row-major [N/128][K/128] (weight_scale_inv).
struct SharedW {
  const uint8_t *g, *u, *d;
  const float *gs, *us, *ds;
};

// One MoE layer's weights on a rank.
struct LayerW {
  const uint8_t* pool;    // the routed experts' blocks
  const MatDesc* desc;    // [kE][3]: gate, up, down
  SharedW sh;
};

// ---- routing ------------------------------------------------------------------------------------------------
// logits f32 [M][256] (the router GEMV's, fp32(x) . fp32(gate.weight)^T), bias f32 [256] ->
// idx [M][8] (descending selection score, ties to the lower id), w [M][8] (s / (sum + 1e-20) * 2.5).
void route(const float* logits, const float* bias, int M, int* idx, float* w, cudaStream_t st, bool pdl = true);

// ---- decode -------------------------------------------------------------------------------------------------
// The expert step for M <= 16 rows: x bf16 [M][6144] (the post-attention-normed input), idx/w from route().
// ws: decode_ws_bytes(), zeroed once before the first call (its task counters reset themselves).
// Writes yr = sum over the rows' experts in expert-id order of w * down(h) and ys = the shared expert's
// output, both f32 [M][6144] partial sums of this rank (the ring all-reduces them), and h's in ws.
struct DecodeCfg {
  int up = 0;     // up-kernel variant (decode.cu, kUpCfgs)
  int down = 0;   // down-kernel variant
  int up_ctas_per_sm = 0;  // 0: as many as fit
  bool pdl = true;
  bool dyn = true;         // k_up claims tasks past the first wave from a counter (load balance)
  int parts = 3;           // bench: 1 up only, 2 down only, 3 both
};
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
// the bf16 tensor cores with the weights dequantised to bf16 in shared memory, each pick's down output kept
// in f32 and combined per row in expert-id order (the decode step's order). The shared expert as a dense
// FP8 GEMM. Outputs as moe_decode: yr, ys f32 [T][6144].
// fp4: the activations are quantised to NVFP4 per 16 with the layer's static input scales (gate/up input,
// down input) and the GEMMs run on mxf4nvf4; this changes outputs (experiment 4).
constexpr int kMaxChunk = 8192;
struct PrefillCfg {
  bool fp4 = false;
  float in_scale_gu = 1.f, in_scale_dn = 1.f;  // input_scale of gate/up and of down (fp4 only)
  bool shared = true;                          // run the shared expert
  int diag = 0;  // bench diagnostics of the gate/up kernel (wrong results): 1 no weights, 2 no MMAs, 4 no A loads
};
// ws: prefill_ws_bytes(T) for the largest chunk, zeroed once before the first call (its counters reset
// themselves); a call's results do not depend on the other rows of the chunk.
size_t prefill_ws_bytes(int T);
void moe_prefill(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int T, void* ws, float* yr,
                 float* ys, cudaStream_t st, const PrefillCfg& cfg = PrefillCfg());
// Per-phase timing hooks for the bench: the same work in separate calls on the same workspace.
// The workspace's intermediates after a call (tests): per row t its picks in expert-id order (rpos: bucket
// positions, re: experts), h [position][512] bf16, the shared expert's hsh [T][512], and the FP4 path's activations (xq [T][3072], xs [T][384],
// hq [position][256], hs [position][32]).
struct PrefillView {
  const int *rpos, *re;
  const __nv_bfloat16 *h, *hsh;
  const uint8_t *xq, *xs, *hq, *hs;
};
PrefillView prefill_view(void* ws, int T);
enum PrefillPhase { kPhBucket = 1, kPhUp = 2, kPhDown = 4, kPhCombine = 8, kPhShared = 16, kPhAll = 31 };
void moe_prefill_phases(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int T, void* ws,
                        float* yr, float* ys, cudaStream_t st, const PrefillCfg& cfg, int phases);

}  // namespace gx
