// CPU-only structural oracle for model/ced_policy.h. No model weights or GPU.
// Built by engine/tests as a host-only target; no CUDA toolchain is required.
#include "model/ced_policy.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// Independent architecture fixture: 20 decoder layers, causal 128-row windows.
// A row is valid only if EVERY reachable prior-layer row is available. This is
// conservative: it does not rely on an attention score being zero or a route
// cancelling a dependency. Complete encoder/compressed-memory state is assumed.
constexpr int decoder_layers = 20;
constexpr int window_rows = 128;
using Bits = std::vector<std::uint8_t>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct Reach {
  int begin;
  std::vector<Bits> inputs;
  Bits final_output;

  bool all_layer_inputs_valid(int position) const {
    for (const auto& input : inputs) if (!input.at(position)) return false;
    return true;
  }
  int first_all_layer_input() const {
    for (int p = begin; p < static_cast<int>(final_output.size()); ++p)
      if (all_layer_inputs_valid(p)) return p;
    return static_cast<int>(final_output.size());
  }
};

Reach propagate(int end, int replay_rows) {
  require(end >= 0 && replay_rows > 0, "invalid synthetic graph size");
  Reach result{std::max(0, end - replay_rows), {}, {}};
  Bits previous(end, 0);
  // These are exact encoder outputs saved from the complete original history.
  std::fill(previous.begin() + result.begin, previous.end(), 1);
  for (int layer = 0; layer < decoder_layers; ++layer) {
    result.inputs.push_back(previous);
    Bits next(end, 0);
    for (int p = result.begin; p < end; ++p) {
      bool available = true;
      // Explicit edges, including self/residual. At absolute position zero the
      // window clips to zero; at a truncated replay boundary it must NOT clip
      // to replay begin, since those missing edges are the issue being tested.
      for (int dependency = std::max(0, p - window_rows + 1);
           dependency <= p; ++dependency) {
        if (!previous[dependency]) { available = false; break; }
      }
      next[p] = available;
    }
    previous = std::move(next);
  }
  result.final_output = std::move(previous);
  return result;
}

bool last_kv_window_valid(const Reach& graph) {
  const int end = static_cast<int>(graph.final_output.size());
  for (int p = std::max(0, end - window_rows); p < end; ++p)
    if (!graph.all_layer_inputs_valid(p)) return false;
  return true;
}

void policy_support() {
  require(model::kCedDecoderLayers == decoder_layers, "decoder architecture changed");
  require(model::kCedWindowRows == window_rows, "window architecture changed");
  require(model::kCedReplayRows == 128 && model::kCedTailRows == 2560,
          "published policy lengths changed");
  for (int rows : {std::numeric_limits<int>::min(), -1, 0, 1, 127, 128,
                   129, 2540, 2541, 2559, 2560, 2561, 8192,
                   std::numeric_limits<int>::max()}) {
    require(model::ced_replay_supported(rows) == (rows == 128 || rows == 2560),
            "unsupported replay length accepted: " + std::to_string(rows));
  }
}

void exact_tail() {
  for (int end : {2561, 4096, 8223}) {
    const auto graph = propagate(end, model::kCedTailRows);
    const int helper = model::ced_valid_from(end, model::kCedTailRows);
    const int first = graph.first_all_layer_input();
    require(helper == first, "helper differs from independently propagated input cone");
    require(graph.final_output.back(), "final sampled output reaches omitted history");
    require(last_kv_window_valid(graph), "last 128 layer-input/KV rows reach omitted history");
    for (int p = helper; p < end; ++p)
      require(graph.all_layer_inputs_valid(p), "advertised valid suffix has a hole");
    require(helper > graph.begin && !graph.all_layer_inputs_valid(helper - 1),
            "helper must not expose a row before the valid input suffix");
    // Last-layer *outputs* have one more dependency step than its KV inputs.
    // This policy must not be misread as proving all last 128 final outputs exact.
    require(!graph.final_output[end - window_rows],
            "synthetic oracle should distinguish KV inputs from final outputs");
    require(!graph.all_layer_inputs_valid(graph.begin),
            "entire 2560-row replay must not be advertised as reusable state");
  }
}

void tight_bound_control() {
  // Adjacent independently propagated graphs detect an inclusive-window
  // off-by-one. These diagnostic lengths are not accepted runtime policies.
  const auto short_graph = propagate(4096, 2540);
  const auto sufficient = propagate(4096, 2541);
  require(!short_graph.final_output.back() && !last_kv_window_valid(short_graph),
          "one-row-short counterexample did not fail both contracts");
  require(sufficient.final_output.back() && last_kv_window_valid(sufficient),
          "minimum finite dependency graph should satisfy both contracts");
  require(model::kCedMinimumTailRows == 2541, "declared minimum differs from graph boundary");
  require(!model::ced_replay_supported(2540) && !model::ced_replay_supported(2541),
          "structural diagnostic lengths must not silently enable a runtime policy");
}

void absolute_prefix_zero() {
  // For short requests this describes the full-history path / helper contract,
  // not permission to invoke a 2560-row CED kernel with too few input rows.
  for (int end : {0, 1, 127, 128, 2559, 2560}) {
    const auto graph = propagate(end, model::kCedTailRows);
    require(graph.begin == 0 && model::ced_valid_from(end, model::kCedTailRows) == 0,
            "absolute-prefix-zero must not invent truncated dependencies");
    require(graph.first_all_layer_input() == 0 && last_kv_window_valid(graph),
            "full-history inputs should remain valid");
    require(std::all_of(graph.final_output.begin(), graph.final_output.end(),
                       [](auto bit) { return bit != 0; }),
            "full-history final outputs should remain structurally valid");
  }
}

void legacy_unchanged() {
  for (int end : {128, 129, 2560, 4096, 8223})
    require(model::ced_valid_from(end, model::kCedReplayRows) == end - 128,
            "legacy approximate suffix policy changed");
  const auto truncated = propagate(4096, model::kCedReplayRows);
  require(!truncated.final_output.back() && !last_kv_window_valid(truncated),
          "legacy replay must not be mistaken for the exact structural tail bound");
  const auto full = propagate(128, model::kCedReplayRows);
  require(full.final_output.back() && last_kv_window_valid(full),
          "legacy replay beginning at absolute zero has no omitted prefix");
}
}  // namespace

int main() {
  try {
    policy_support();
    exact_tail();
    tight_bound_control();
    absolute_prefix_zero();
    legacy_unchanged();
    std::cout << "PASS: 5 CPU structural policy controls; no GPU or numerical-equivalence claim\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
