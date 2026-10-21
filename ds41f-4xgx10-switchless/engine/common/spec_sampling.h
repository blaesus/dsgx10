// Probabilistic speculative-sampling additions (2026): reimplemented for
// dsgx10's CUDA/ring execution path after consulting DeepInfer, revision
// 8b16149741a4935e1dec8cea27853f273b374c5b. Relevant upstream portions retain
// their MIT notices. DeepInfer-derived additions are provided under Apache-2.0;
// see repository NOTICE, LICENSES/Apache-2.0-DeepInfer.txt, and
// docs/probabilistic-speculation.md for source mapping and changes.
// Probability proposal sampling shared by DSpark and the main-model verifier.
// Coordinates are (request seed, predicted absolute token position, stream, token id).
// They do not depend on batch entry, CUDA launch, rank, or the number of accepted drafts.
#pragma once
#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define SPEC_HD __host__ __device__
#else
#define SPEC_HD
#endif

namespace spec_sampling {
struct Params {
  uint64_t seed;
  float temperature, top_p;
  int32_t top_k, n_top;
};
struct ProposalMeta {
  uint64_t seed;
  float temperature;
  int32_t first_pos;
  int32_t valid;
  int32_t reserved[3];
};
static_assert(sizeof(Params) == 24 && sizeof(ProposalMeta) == 32, "sampling ABI");

constexpr uint64_t kDraftStream = 0xD2B74407B1CE6E93ull;
constexpr uint64_t kAcceptStream = 0xCA5A826395121157ull;
constexpr uint64_t kResidualStream = 0x9E6C63D0676A9A99ull;
SPEC_HD inline uint64_t mix(uint64_t x) {
  x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29; x *= 0x94D049BB133111EBull;
  x ^= x >> 32; return x;
}
// All 23-bit midpoints are representable and strictly inside (0,1).
SPEC_HD inline float unit(uint64_t h) {
  return (static_cast<float>(h >> 41) + 0.5f) * (1.0f / 8388608.0f);
}
SPEC_HD inline float uniform(uint64_t seed, int pos, uint64_t stream, int token = 0) {
  return unit(mix(mix(seed ^ stream) ^ (uint64_t(uint32_t(pos)) * 0x9E3779B97F4A7C15ull)
                  ^ (uint64_t(uint32_t(token)) * 0xD1B54A32D192ED03ull)));
}
SPEC_HD inline bool eligible(const Params& p) {
  return p.temperature > 0.f && p.temperature < INFINITY && p.top_p == 1.f && p.top_k == 0;
}
SPEC_HD inline bool matches(const ProposalMeta& m, const Params& p, int predicted_pos) {
  return m.valid == 1 && eligible(p) && m.seed == p.seed && m.temperature == p.temperature
      && predicted_pos >= m.first_pos && predicted_pos < m.first_pos + 5;
}
SPEC_HD inline float log_positive_difference(float lp, float lq) {
  // log(max(p-q,0)), without exponentiating two nearly equal normalised probabilities.
  return lp > lq ? lp + logf(-expm1f(lq - lp)) : -INFINITY;
}
}  // namespace spec_sampling
#undef SPEC_HD
