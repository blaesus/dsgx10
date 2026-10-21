// Replay lengths are a request policy, not the number of rows whose old local state is reusable.
// The tail policy removes the finite sliding-window dependency on the truncated prefix for the
// final sample and the last 128 decoder/DSpark KV rows. Different GEMM shapes can still round
// differently; this is an architectural dependency bound, not a bit-exact numerical guarantee.
#pragma once

namespace model {
inline constexpr int kCedReplayRows = 128;
inline constexpr int kCedTailRows = 2560;
inline constexpr int kCedDecoderLayers = 20;
inline constexpr int kCedWindowRows = 128;
inline constexpr int kCedWindowReach = kCedWindowRows - 1;
inline constexpr int kCedLastInputReach = (kCedDecoderLayers - 1) * kCedWindowReach;
inline constexpr int kCedMinimumTailRows = kCedLastInputReach + kCedWindowRows;
static_assert(kCedTailRows >= kCedMinimumTailRows, "the last decoder and DSpark KV window is covered");
static_assert(kCedTailRows >= kCedDecoderLayers * kCedWindowReach + 1, "the last sampled output is covered");

constexpr bool ced_replay_supported(int rows) {
  return rows == kCedReplayRows || rows == kCedTailRows;
}
// Earliest replay position whose input to every decoder layer (and DSpark's 37..39 taps) is valid.
// Existing legacy replay retains its trained approximate semantics. Tail replay only exposes the
// suffix beyond its dependency cone. A replay starting at zero has no truncated history.
constexpr int ced_valid_from(int end, int rows) {
  return rows == kCedReplayRows ? end - rows
       : end <= rows ? 0
       : end - rows + kCedLastInputReach;
}
}  // namespace model
