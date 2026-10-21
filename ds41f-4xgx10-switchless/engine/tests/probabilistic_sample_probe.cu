// Actual ops::sample, actual four-rank comm channel; no model weights required.
// All expected probability masses are from the independent Fraction oracle.
// --fleet FILE --rank 0..3 --port UNIQUE --trials 4096 --deadline 120
// Run only in a separately authorized, bounded four-rank diagnostic window.
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "comm/ring.h"
#include "common/check.h"
#include "common/spec_sampling.h"
#include "model/ops.h"
#include "distribution_fixtures.h"

namespace {
using namespace model;
using namespace model::ops;
constexpr int NS = 8, SLOTS = 16, ROWS = NS * 2;
constexpr int SLOT_IDS[NS] = {0, 3, 7, 2, 14, 10, 5, 12};
constexpr float NEG = -std::numeric_limits<float>::infinity();
using Clock = std::chrono::steady_clock;
Clock::time_point deadline;
int checks = 0;

void require(bool ok, const char* why) {
  ++checks;
  if (!ok) die("independent sampler check failed: %s", why);
}
void time_check() {
  if (Clock::now() > deadline) die("independent sampler wall deadline exceeded");
}
void sync(cudaStream_t stream) {
  for (;;) {
    auto rc = cudaStreamQuery(stream);
    if (rc == cudaSuccess) return;
    if (rc != cudaErrorNotReady) die("sampler CUDA failure: %s", cudaGetErrorString(rc));
    time_check();
    usleep(50);
  }
}
template<class T> struct Device {
  T* p = nullptr;
  size_t n;
  explicit Device(size_t count) : n(count) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
  ~Device() { if (p) cudaFree(p); }
  Device(const Device&) = delete;
  void put(const T* source, cudaStream_t s) {
    CUDA_CHECK(cudaMemcpyAsync(p, source, n * sizeof(T), cudaMemcpyHostToDevice, s));
  }
};
uint64_t hash_bytes(const void* source, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ ((const unsigned char*)source)[i]) * 1099511628211ull;
  return h;
}
int global_id(int i) { return (i % 4) * kVocabPart + 17 + i / 4; }
int support_id(const OracleCase& c, int token) {
  for (int i = 0; i < c.n; ++i) if (global_id(i) == token) return i;
  return -1;
}
int independent_proposal(const OracleCase& c, uint64_t seed, int slot, int index) {
  // This RNG is deliberately unrelated to the candidate's hash/Gumbel code.
  std::seed_seq words{uint32_t(seed), uint32_t(seed >> 32), uint32_t(slot), uint32_t(index), 0x993a12u};
  std::mt19937_64 random(words);
  const long double u = std::generate_canonical<long double, 64>(random);
  long double cumulative = 0;
  int last = -1;
  for (int i = 0; i < c.n; ++i) {
    if (c.q[i] > 0) last = i;
    cumulative += c.q[i];
    if (c.q[i] > 0 && u < cumulative) return global_id(i);
  }
  require(last >= 0, "proposal support empty");
  return global_id(last);
}
struct Harness {
  comm::Ring& ring;
  cudaStream_t stream{};
  Device<float> logits{ROWS * kVocabPart}, q{SLOTS * kDrafts * kVocabPart};
  Device<spec_sampling::ProposalMeta> meta{SLOTS};
  Device<StepIn> input{1};
  Device<uint32_t> ctl{1}, counter{1};
  Device<int32_t> feedback{kMaxRows};
  Device<SpecState> state{kMaxSess};
  Device<SampleOut> output{kOutSlots};
  cudaGraphExec_t graph{}, graph_low{};
  std::vector<float> hlogits, hq;
  std::array<spec_sampling::ProposalMeta, SLOTS> hmeta{};
  StepIn hin{};
  explicit Harness(comm::Ring& r) : ring(r) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    uint32_t zero = 0; ctl.put(&zero, stream); counter.put(&zero, stream);
    CUDA_CHECK(cudaMemsetAsync(output.p, 0, sizeof(SampleOut) * kOutSlots, stream));
    CUDA_CHECK(cudaMemsetAsync(state.p, 0, sizeof(SpecState) * kMaxSess, stream));
    sync(stream);
    for (int rows : {kLowRows, kMaxRows}) {
      cudaGraph_t captured{};
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      sample(ring.dev(), logits.p, ring.rank() * kVocabPart, input.p, ctl.p,
             output.p, feedback.p, counter.p, stream, state.p, rows,
             q.p, meta.p, SLOTS);
      CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
      CUDA_CHECK(cudaGraphInstantiate(rows == kLowRows ? &graph_low : &graph, captured, 0));
      CUDA_CHECK(cudaGraphDestroy(captured));
    }
  }
  ~Harness() {
    if (graph) cudaGraphExecDestroy(graph);
    if (graph_low) cudaGraphExecDestroy(graph_low);
    cudaStreamDestroy(stream);
  }
  void probabilities(const OracleCase& c) {
    hlogits.assign(ROWS * kVocabPart, NEG);
    hq.assign(SLOTS * kDrafts * kVocabPart, NEG);
    const int rank = ring.rank();
    for (int i = 0; i < c.n; ++i) {
      const int id = global_id(i);
      if (id / kVocabPart != rank) continue;
      const int local = id % kVocabPart;
      for (int row = 0; row < ROWS; ++row)
        hlogits[row * kVocabPart + local] = c.p[i] > 0 ? float(std::log(c.p[i])) : NEG;
      for (int slot = 0; slot < SLOTS; ++slot)
        for (int j = 0; j < kDrafts; ++j)
          hq[(slot * kDrafts + j) * kVocabPart + local] = c.q[i] > 0 ? float(std::log(c.q[i])) : NEG;
    }
    logits.put(hlogits.data(), stream); q.put(hq.data(), stream);
    sync(stream);
  }
  void inputs(const OracleCase& c, int trial, bool reverse = false, int drafts = 1) {
    hin = StepIn{};
    hmeta.fill(spec_sampling::ProposalMeta{});
    const int ns = drafts == 1 ? NS : 1;
    hin.rows = ns * (drafts + 1); hin.nsess = ns; hin.step = 0;
    hin.spec = kSpecVerify; hin.sbatch = ns > 1; hin.force = -1;
    hin.max_drafts = drafts; hin.pos_end = 1000000;
    for (int entry = 0; entry < ns; ++entry) {
      const int logical = reverse ? ns - 1 - entry : entry;
      const int slot = SLOT_IDS[logical];
      const int pos = 1024 + trial * 8 + slot * 1024;
      const uint64_t seed = 0x46532d91b172aa01ull + uint64_t(trial) * 0x9e3779b97f4a7c15ull
                            + uint64_t(slot) * 0xc2b2ae3d27d4eb4full;
      hin.samp[entry] = StepSamp{seed, 1.0f, 1.0f, 0, 0};
      hin.sess_on[entry] = 1; hin.sess_slot[entry] = slot;
      hin.scap[entry] = drafts; hin.spos_end[entry] = 1000000;
      for (int row = 0; row <= drafts; ++row) {
        const int at = entry * (drafts + 1) + row;
        hin.pos[at] = pos + row; hin.slot[at] = slot; hin.sess[at] = entry;
        hin.tokens[at] = row == 0 ? global_id(0) : independent_proposal(c, seed, slot, row);
      }
      hmeta[slot].seed = seed; hmeta[slot].temperature = 1.0f;
      hmeta[slot].first_pos = pos + 1; hmeta[slot].valid = 1;
    }
  }
  SampleOut run(bool use_graph, bool proposal = true, bool low_rows = false) {
    time_check();
    require(!low_rows || hin.rows <= kLowRows, "low-row launch overflow");
    input.put(&hin, stream); meta.put(hmeta.data(), stream);
    uint32_t zero = 0; counter.put(&zero, stream);
    CUDA_CHECK(cudaMemsetAsync(output.p, 0, sizeof(SampleOut) * kOutSlots, stream));
    if (use_graph) {
      require(proposal, "graph without proposal uses a different entry point");
      CUDA_CHECK(cudaGraphLaunch(low_rows ? graph_low : graph, stream));
    } else {
      sample(ring.dev(), logits.p, ring.rank() * kVocabPart, input.p, ctl.p,
             output.p, feedback.p, counter.p, stream, state.p, low_rows ? kLowRows : kMaxRows,
             proposal ? q.p : nullptr, proposal ? meta.p : nullptr, proposal ? SLOTS : 0);
    }
    sync(stream);
    SampleOut out{};
    CUDA_CHECK(cudaMemcpy(&out, output.p, sizeof(out), cudaMemcpyDeviceToHost));
    return out;
  }
  void all_rank_same(const SampleOut& o) {
    const uint64_t mine = hash_bytes(&o, sizeof(o));
    uint64_t all[4]{}; ring.allgather_host(&mine, all, sizeof(mine));
    require(all[0] == all[1] && all[1] == all[2] && all[2] == all[3], "four-rank outputs differ");
  }
};
void same_rows(const SampleOut& a, const SampleOut& b, int rows, bool exact_flags = true) {
  for (int row = 0; row < rows; ++row) {
    require(a.row[row].token == b.row[row].token, "same input produced a different token");
    require(a.row[row].logprob == b.row[row].logprob, "same input produced different target logprob");
    if (exact_flags) require(a.row[row].flags == b.row[row].flags, "same input flags differ");
  }
}
void controls(Harness& h) {
  const auto& equal = kOracleCases[0];
  h.probabilities(equal); h.inputs(equal, 7);
  const auto eager = h.run(false), graph = h.run(true);
  same_rows(eager, graph, ROWS); h.all_rank_same(graph);
  for (int k = 0; k < NS; ++k) {
    require(graph.sess_rows[k] == 2, "p=q must accept every proposal");
    require(graph.row[2*k].flags & kSampProb, "p=q did not use probability verifier");
    require(!(graph.row[2*k+1].flags & kSampProb), "bonus incorrectly used proposal verifier");
  }
  h.inputs(equal, 7, true);
  const auto swapped = h.run(true);
  for (int k = 0; k < NS; ++k)
    for (int r = 0; r < 2; ++r)
      require(graph.row[2*k+r].token == swapped.row[2*(NS-1-k)+r].token,
              "batch reorder changed per-session token");

  // Existing captured graph must read the new distributions and metadata.
  const auto& disjoint = kOracleCases[1];
  h.probabilities(disjoint); h.inputs(disjoint, 8);
  const auto rejected = h.run(true);
  for (int k = 0; k < NS; ++k) {
    require(rejected.sess_rows[k] == 1, "disjoint support must reject first draft");
    require(rejected.row[2*k].flags & kSampProbReject, "missing reject flag");
    const int support = support_id(disjoint, rejected.row[2*k].token);
    require(support >= 0 && disjoint.p[support] > 0, "residual outside target support");
  }
  // Five proposals: accepted-prefix count and bonus-row contract, no model body.
  h.probabilities(equal); h.inputs(equal, 9, false, 5);
  const auto five = h.run(true, true, true);
  require(five.rows == 6, "all five proposals must leave one bonus");
  require(!(five.row[5].flags & kSampProb), "five-draft bonus used q");
  h.probabilities(disjoint); h.inputs(disjoint, 9, false, 5);
  const auto first = h.run(true, true, true);
  require(first.rows == 1, "first rejection failed to discard later proposals");

  // Unsupported sampling and stale metadata compare with the original path.
  for (int mode = 0; mode < 8; ++mode) {
    h.probabilities(equal); h.inputs(equal, 11 + mode);
    for (int k = 0; k < NS; ++k) {
      const int slot = h.hin.sess_slot[k];
      if (mode == 0) h.hin.samp[k].temperature = 0;
      if (mode == 1) h.hin.samp[k].top_p = 0.8f;
      if (mode == 2) h.hin.samp[k].top_k = 2;
      if (mode == 3) ++h.hmeta[slot].seed;
      if (mode == 4) ++h.hmeta[slot].first_pos;
      if (mode == 5) h.hmeta[slot].valid = 0;
      if (mode == 6) h.hmeta[slot].temperature = 0.5f;
      if (mode == 7) h.hin.gate = kGateRatio;
    }
    const auto candidate = h.run(true), original = h.run(false, false);
    same_rows(candidate, original, ROWS, false);
    for (int row = 0; row < ROWS; ++row)
      require(!(candidate.row[row].flags & kSampProb), "unsupported/stale q was used");
  }
}
void statistics(Harness& h, const OracleCase& c, int trials) {
  h.probabilities(c);
  std::array<int, 16> counts{}, bonus{};
  int accepted = 0, total = 0, zero_residual = 0;
  for (int batch = 0; total < trials; ++batch) {
    h.inputs(c, 100 + batch);
    const auto out = h.run(true);
    if (batch == 0) h.all_rank_same(out);
    for (int k = 0; k < NS && total < trials; ++k) {
      const auto& row = out.row[2*k];
      const int id = support_id(c, row.token), b = support_id(c, out.row[2*k+1].token);
      require(id >= 0 && b >= 0, "output outside fixture support");
      require(c.p[id] > 0 && c.p[b] > 0, "output on zero target probability");
      require(row.flags & kSampProb, "valid q did not use probability branch");
      require(!(row.flags & kSampProbFallback), "unexpected metadata/probability fallback");
      zero_residual += bool(row.flags & kSampResidualZero);
      accepted += out.sess_rows[k] == 2;
      ++counts[id]; ++bonus[b]; ++total;
    }
  }
  require(zero_residual == 0, "unexpected numerical zero-residual fallback");
  auto near = [=](int count, double expected) {
    const double bound = 7 * std::sqrt(trials * expected * (1 - expected)) + 8;
    require(std::abs(count - trials * expected) <= bound, "empirical mass outside fixed 7-sigma+8 bound");
  };
  for (int i = 0; i < c.n; ++i) { near(counts[i], c.p[i]); near(bonus[i], c.p[i]); }
  near(accepted, c.accept);
  if (c.accept == 1) require(accepted == total, "exact all-accept case rejected");
  if (c.accept == 0) require(accepted == 0, "exact all-reject case accepted");
  if (h.ring.rank() == 0) {
    std::printf("{\"case\":\"%s\",\"trials\":%d,\"accepted\":%d,\"expected_accept\":%.12g,"
                "\"residual_zero\":%d,\"counts\":[", c.name, total, accepted, c.accept, zero_residual);
    for (int i = 0; i < c.n; ++i) std::printf("%s%d", i ? "," : "", counts[i]);
    std::printf("],\"bonus_counts\":[");
    for (int i = 0; i < c.n; ++i) std::printf("%s%d", i ? "," : "", bonus[i]);
    std::puts("],\"passed\":true}"); std::fflush(stdout);
  }
}
}  // namespace
int main(int argc, char** argv) {
  comm::Options options;
  options.cap = 1 << 20; options.proxy_cpu = -1; options.bootstrap_timeout_s = 20;
  int trials = 4096, seconds = 120; bool explicit_port = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* { if (++i >= argc) die("missing option value"); return argv[i]; };
    if (arg == "--fleet") options.fleet = value();
    else if (arg == "--rank") options.rank = std::atoi(value());
    else if (arg == "--port") options.port = std::atoi(value()), explicit_port = true;
    else if (arg == "--trials") trials = std::atoi(value());
    else if (arg == "--deadline") seconds = std::atoi(value());
    else die("unknown argument %s", arg.c_str());
  }
  require(explicit_port && options.port >= 20000 && options.port <= 65000, "fresh explicit diagnostic port required");
  require(options.rank >= 0 && options.rank < 4 && !options.fleet.empty(), "explicit rank/fleet required");
  require(trials >= 1024 && trials <= 16384 && seconds >= 10 && seconds <= 180, "finite probe budgets required");
  deadline = Clock::now() + std::chrono::seconds(seconds);
  CUDA_CHECK(cudaSetDevice(0));
  comm::Ring ring(options);
  Harness harness(ring);
  controls(harness);
  for (const auto& c : kOracleCases) statistics(harness, c, trials);
  ring.barrier();
  if (ring.rank() == 0)
    std::printf("{\"passed\":true,\"scope\":\"actual-four-rank-ops-sample-not-full-model\","
                "\"cases\":%zu,\"trials_per_case\":%d,\"checks\":%d}\n", kOracleCaseCount, trials, checks);
  return 0;
}
