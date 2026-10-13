// GLM-5.3 at TP=4: the runner's constants and layer roles (docs/engine.md). The kernels' shapes are attn/attn.h's
// and experts/experts.h's; this adds what only the runner needs: the layer roles, the MTP layer, the vocabulary
// the sampler may choose from, the stop ids, the step's row and session limits.
#pragma once
#include <cstdint>

#include "attn/attn.h"
#include "experts/experts.h"

namespace glm::model {

using ::glm::kDim;
using ::glm::kLayers;  // 78 main layers
using ::glm::kTP;
using ::glm::kVocab;
using ::glm::kVocabRank;
inline constexpr int kRanks = kTP;
inline constexpr int kMtpLayer = 78;          // the MTP layer's index: its weights are layers.78.*
inline constexpr int kCacheLayers = kLayers + 1;  // every main layer and the MTP layer have a KV cache
inline constexpr int kDenseLayers = 3;        // layers 0-2: dense MLP (first_k_dense_replace)
inline constexpr int kMaxRows = ::glm::kMaxRows;  // 16 rows a step (attn and experts agree)
static_assert(kMaxRows == gx::kMaxRows, "attn and experts serve the same rows a step");
inline constexpr int kMaxSess = kMaxRows;     // sessions in a step, at most
inline constexpr int kMaxChunk = gx::kMaxChunk;  // 8192 rows a prefill chunk
inline constexpr int kEmbRow = kDim * 2;      // 12,288 B: one embedding row
inline constexpr int kEhRows = kDim / kTP;    // 1536: a rank's rows of the MTP's eh_proj
inline constexpr int kEhK = 2 * kDim;         // 12,288: its input, [enorm(e) | hnorm(h)]

// The vocabulary: the head has 154,880 rows, the tokenizer uses 0..154,855 (docs/protocol.md, version 2): ids
// from kVocabUsed on are masked before sampling. The stop ids end a generation (generation_config.json).
inline constexpr int kVocabUsed = 154856;
inline constexpr int32_t kStopIds[3] = {154820, 154827, 154829};
constexpr bool is_stop(int32_t t) { return t == kStopIds[0] || t == kStopIds[1] || t == kStopIds[2]; }

constexpr bool dense_layer(int l) { return l < kDenseLayers; }
// full(l) and its source for the main layers (attn.h); the MTP layer has its own indexer (a full layer)
constexpr bool full_layer(int l) { return l == kMtpLayer || ::glm::full_indexer(l); }
// The 22 index-key caches: the 21 full main layers and the MTP layer, numbered in layer order.
constexpr int idx_slot(int l) {
  int n = 0;
  for (int i = 0; i < l && i < kLayers; ++i) n += ::glm::full_indexer(i);
  return n;
}
inline constexpr int kIdxCaches = idx_slot(kLayers) + 1;
static_assert(kIdxCaches == 22, "21 full layers and the MTP layer");

// Layer classes for the per-layer timing.
enum LayerClass : int { kClsDense = 0, kClsMoeFull = 1, kClsMoeShared = 2, kClsMtp = 3, kClasses = 4 };
constexpr int layer_class(int l) {
  return l == kMtpLayer ? kClsMtp : dense_layer(l) ? kClsDense : ::glm::full_indexer(l) ? kClsMoeFull : kClsMoeShared;
}

}  // namespace glm::model
