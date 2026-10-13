// DeepSeek-V4.1-Flash at TP=4 on the GB10 ring: the shapes and the per-layer roles the runner needs.
// Everything here is fixed by the checkpoint (config.json) and DESIGN.md decision 1.
#pragma once
#include <cstdint>

namespace model {

constexpr int kRanks = 4;
constexpr int kLayers = 40;
constexpr int kDim = 5120;
constexpr int kHc = 4;                      // residual copies (mHC)
constexpr int kHcDim = kHc * kDim;          // 20480
constexpr int kMix = 24;                    // (2 + hc) * hc mixes per sublayer
constexpr int kVocab = 129280;
constexpr int kVocabPart = kVocab / kRanks;  // 32320 head rows per rank
constexpr int kMaxRows = 8;                 // rows per step (decode 1, verify <= 6, prefill chunks <= 8)
constexpr int kMaxSlots = 8;                // session slots (Options::slots): live sessions' states, at most
// Rows of one session in a step, at most, whatever kMaxRows becomes (B4): the attention kernels' contract
// for a step's slot table. attn::decode_prep's fused compressor emit and carry race across CTAs at 16
// rows of one session (the step runs kv_write and compressor_emit, safe at 16, but the contract holds for
// any caller), and attn can check only steps without a table: the model checks every step it builds.
constexpr int kMaxSessRows = 8;

// attention, per rank
constexpr int kQLora = 1280, kQaPart = kQLora / kRanks;  // wq_a: 320 rows per rank
constexpr int kHead = 512, kKvPart = kHead / kRanks;     // wkv: 128 rows per rank
constexpr int kCompPart = kHead / kRanks;                // compressor wkv / wgate: 128 rows per rank
constexpr int kHeads = 16;                               // query heads per rank
constexpr int kQ = kHeads * kHead;                       // wq_b: 8192 rows per rank
constexpr int kOGroupRows = 1024;                        // wo_a rows per group (2 groups per rank)
constexpr int kOPart = 2 * kOGroupRows;                  // 2048 = wo_a rows = wo_b columns per rank
constexpr int kWoaK = 4096;                              // wo_a input per group (8 heads x 512)
constexpr int kIdxHeads = 32, kIdxDim = 128;             // indexer (replicated)
constexpr int kIdxQ = kIdxHeads * kIdxDim;               // 4096
constexpr int kIdxTopk = 512;
constexpr int kPoolBlocks = 2048;                        // layer 20's candidate pool (blocks of 8)

// MoE
constexpr int kExperts = 384, kGatePart = kExperts / kRanks;  // gate: 96 rows per rank
constexpr int kTopK = 6;
constexpr float kRouteScale = 1.5f;
constexpr int kInterPart = 576;  // expert and shared-expert intermediate columns per rank (2304 / 4)

constexpr float kNormEps = 1e-20f;

// Engram (layers 1 and 14): 24 rows of 256 e4m3 + 8 ue8m0 per position; wkv per rank is 5 segments
// of 1280 output rows: keys 0..3 and the value, each this rank's quarter of the 5120 columns.
constexpr int kEngramCols = 24, kEngramRow = 264, kEngramK = kEngramCols * 256;  // 6144
constexpr int kEngramSeg = kDim / kRanks;                                       // 1280
constexpr int kEngramN = 5 * kEngramSeg;                                        // 6400
constexpr int kEngramPartBytes = 8 * 4 + kEngramSeg * 2;  // per row: 4 dots, 4 key sums of squares, value slice

// Layer roles (config.json: compress_ratios, kv_source_layers, index_source_layers).
constexpr int ratio_of(int L) { return L < 2 ? 0 : L < 20 ? 2 : 1; }
constexpr bool kv_src(int L) { return L == 2 || L == 8 || L == 14 || L == 20; }
constexpr int kv_source(int L) { return L < 2 ? -1 : L < 8 ? 2 : L < 14 ? 8 : L < 20 ? 14 : 20; }
constexpr bool idx_src(int L) { return kv_src(L) || (L >= 24 && L % 4 == 0); }
// The index layer whose top-k this layer attends with (itself when it is one).
constexpr int index_source(int L) { return L < 2 ? -1 : L < 20 ? kv_source(L) : 20 + (L - 20) / 4 * 4; }
constexpr bool reindex(int L) { return L > 20 && idx_src(L); }  // scores only layer 20's pool
constexpr bool engram_layer(int L) { return L == 1 || L == 14; }
constexpr int engram_index(int L) { return L == 1 ? 0 : 1; }
// Per-step index results: one slot per index source, in layer order; compressed caches: one per kv source.
constexpr int kIdxSlots = 8;
constexpr int idx_slot(int L) { return L == 2 ? 0 : L == 8 ? 1 : L == 14 ? 2 : L == 20 ? 3 : 4 + (L - 24) / 4; }
constexpr int kKvSlots = 4;
constexpr int kv_slot(int L) { return L == 2 ? 0 : L == 8 ? 1 : L == 14 ? 2 : 3; }

}  // namespace model
