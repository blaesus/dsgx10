// The model's driver on the ring (docs/engine.md, "The driver"): every rank runs it with the same arguments
// (run4.sh); rank 0 prints and writes.
//   glm_run [options] mode...
// Weights: --weights PATH (a rank file; "%d" in PATH is the rank; "fleet": the fleet file's RANK<r>_WEIGHTS) or --synthetic [--pools N] [--s3] [--seed S].
// Modes:
//   bench     decode steps as graphs at --rows (default 1,2,4,6,8,16) at context --ctx: GPU ms a step, wall ms, host
//             ms a launch; the floor; then one profiled eager step a row count (ms per kernel class, per layer class)
//   l2pf      the paced L2 prefetch off and on, A/B and B/A, at --rows
//   ab        the speed levers --ab-base MASK (default none) against --ab MASK (default all; Tune bits, hex), A/B/B/A
//             at --rows, --ab-rounds (10) rounds of 12 steps each, A B then B A: GPU ms a step (the medians, the ratio
//             of each round's pair) and whether the outputs are the same bits
//             (the final normed rows, the sampled tokens and values, every layer's cache)
//   trace     graph steps at --rows inside cudaProfilerStart/Stop (run under nsys; tools/step_profile.py)
//   prefill   prefill chunks of --chunks (default 512,1024,2048,4096,8192) rows at position --ctx: tok/s
//   pfab      prefill, the lever sets --ab-base and --ab, A B B A for each chunk of --chunks rows at position --at (a
//             list beside --chunks; default --ctx; a "chunk" above --chunk is a whole prompt in chunks), the caches
//             before it filled with hashed values: ms and tok/s,
//             and whether A and B give the same bits (every layer's cache, the last row's sample; with --drafts the
//             MTP layer's cache holds every row's final normed hidden), every rank
//   pftrace   one prefill chunk of --chunks[0] rows at --at[0] (after a warm one) inside cudaProfilerStart/Stop (nsys;
//             tools/step_profile.py --prefill)
//   check     the state checks on synthetic weights (chunking, prefill vs decode, batch invariance, determinism)
//   tf        teacher-forced over --tokens FILE (--via prefill|decode): argmax, top-20, lse, next logprob, sampled
//             rows' logits and hidden into --dir (the reference's golden format, docs/reference.md § 7)
//   refhash   prefill --tokens (or --prompt N synthetic ids) and decode --steps greedy tokens (and --temp T): the hash
//             of the tokens and their logit bits, every rank
//   spec      speculative decoding (G7) against plain decoding on the golden prompts (--gold PREFIX, --seqs), --sess
//             sessions, --ks drafts a step (-1: the gate), --steps tokens a session: the same tokens bit for bit, and
//             ms a step, tok/s, acceptance (docs/engine.md, "Drafts")
//   mtpgold   the MTP layer teacher-forced on the golden final hidden (--gold PREFIX on every rank, --seqs, --passes):
//             its drafts against the reference's MTP argmax, the main argmax and the text
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include <deque>

#include <cuda_profiler_api.h>

#include "comm/ring.h"
#include "comm/ring_config.h"
#include "model/model.h"
#include "model/spec.h"
#include "model/tools/io.h"
#include "weights/loader.h"

using namespace glm::model;
using glmrun::npy_write;

namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  size_t p = 0;
  while (p < s.size()) {
    const size_t e = s.find(',', p);
    v.push_back(std::stoi(s.substr(p, e == std::string::npos ? std::string::npos : e - p)));
    if (e == std::string::npos) break;
    p = e + 1;
  }
  return v;
}
uint64_t fnv(const void* p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
  return h;
}
int32_t synth_token(uint64_t seed, int i) {
  uint64_t x = seed * 0x9E3779B97F4A7C15ull + (uint64_t)i * 0xD1B54A32D192ED03ull;
  x = (x ^ (x >> 31)) * 0xBF58476D1CE4E5B9ull;
  return (int32_t)((x >> 20) % 154820);
}

// a leading "~/" as the home directory (paths given to every rank alike)
std::string expand(const std::string& p) {
  const char* h = std::getenv("HOME");
  return p.rfind("~/", 0) == 0 && h ? std::string(h) + p.substr(1) : p;
}

// a tokens file read on rank 0 and sent to every rank (the workers have no copy)
std::vector<int32_t> ring_tokens(comm::Ring& ring, const std::string& path) {
  std::vector<int32_t> t;
  if (ring.rank() == 0) t = glmrun::read_tokens(path);
  int64_t n = (int64_t)t.size(), all[4];
  ring.allgather_host(&n, all, 8);
  t.resize(all[0]);
  std::vector<int32_t> buf(4 * all[0]);
  if (all[0]) ring.allgather_host(t.data(), buf.data(), (size_t)all[0] * 4);
  std::copy(buf.begin(), buf.begin() + all[0], t.begin());
  return t;
}

struct Args {
  int rank = -1;
  std::string fleet, weights, out, tag, dir, tokens, sample_rows, via = "prefill";
  bool synthetic = false, s3 = false, fill_experts = false;
  int pools = 16;
  uint64_t seed = 1;
  int ctx = 4096, steps = 32, prompt = 0;
  float temp = 0.f, top_p = 1.f;
  std::vector<int> rows = {1, 2, 4, 6, 8, 16}, chunks = {512, 1024, 2048, 4096, 8192}, ats;
  std::vector<int> dump_layers;
  uint32_t ab_base = 0, ab = 0xffffffffu;
  int ab_rounds = 10;
  std::string gold, seqs;  // spec, mtpgold: the golden set's raw copies (PREFIX.tokens.i32 ...), the sequences
  int passes = kMaxDrafts, kmax = 3, spec_samp = 3, block = 16;
  std::vector<int> sess = {1}, ks = {1, 2, 3, -1};
  std::vector<float> spec_cost;
  Options mo;
  std::vector<std::string> modes;
};

FILE* g_out = nullptr;
Args A;
int g_fail = 0;

void record(int rank, const char* fmt, ...) {
  if (rank != 0) return;
  char line[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(line, sizeof line, fmt, ap);
  va_end(ap);
  std::printf("%s\n", line);
  std::fflush(stdout);
  if (g_out) {
    std::fprintf(g_out, "%s\t%s\n", A.tag.c_str(), line);
    std::fflush(g_out);
  }
}
void check(int rank, bool ok, const char* fmt, ...) {
  char line[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(line, sizeof line, fmt, ap);
  va_end(ap);
  if (!ok) ++g_fail;
  if (rank == 0) record(0, "%s\t%s", ok ? "PASS" : "FAIL", line);
}

Batch one_session(int R, int pos0, const int32_t* tok, int slot, const Sampling& sp) {
  Batch b;
  b.R = R, b.nsess = 1;
  for (int r = 0; r < R; ++r) b.tokens[r] = tok[r], b.pos[r] = pos0 + r, b.slot[r] = slot, b.sess[r] = 0;
  b.sp[0] = sp;
  return b;
}

// expected distinct experts of R rows routed independently and uniformly (synthetic weights route about so)
double union_of(int R) { return 256.0 * (1.0 - std::pow(1.0 - 8.0 / 256.0, R)); }

// ---- bench ------------------------------------------------------------------------------------------------------

void bench(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  m.fill_cache(0, A.ctx + 64, 7);
  Sampling sp;
  for (int R : A.rows) {
    std::vector<int32_t> tok(R);
    for (int r = 0; r < R; ++r) tok[r] = synth_token(A.seed, r);
    const Batch b = one_session(R, A.ctx, tok.data(), 0, sp);
    for (int i = 0; i < 3; ++i) m.wait(m.launch(b));  // eager once, then the graph's capture and a replay
    ring.barrier();
    const int N = A.steps;
    std::vector<uint64_t> ns(N);
    double host = 0;
    const double t0 = now_s();
    for (int i = 0; i < N; ++i) {
      ns[i] = m.launch(b);
      host += m.launch_us() * 1e-6;
    }
    m.wait(ns[N - 1]);
    const double wall = (now_s() - t0) * 1e3 / N;
    std::vector<double> g;
    for (int i = std::max(0, N - 4); i < N; ++i) g.push_back(m.gpu_ms(ns[i]));
    std::sort(g.begin(), g.end());
    const double gpu = g[g.size() / 2];
    const double bytes = m.step_bytes(union_of(R), m.options().drafts > 0);
    const double floor = bytes / 235e6;
    record(rk, "bench\tstep\trows %d\tctx %d\tdrafts %d\tgpu %.3f ms\twall %.3f ms\thost launch %.1f us\tbytes %.3f GB\t"
           "floor %.2f ms\tx floor %.3f\ttok/s %.1f",
           R, A.ctx, m.options().drafts, gpu, wall, host * 1e6 / N, bytes / 1e9, floor, gpu / floor, R * 1e3 / wall);
    ring.barrier();
    // per kernel class and layer class, one eager step without PDL
    const Profile p = m.profile(b);
    for (int c = 0; c < kKClasses; ++c)
      if (p.ms[c] > 0) record(rk, "profile\trows %d\tclass\t%s\t%.3f ms", R, kclass_name(c), p.ms[c]);
    double cls[kClasses] = {}, cnt[kClasses] = {};
    for (int l = 0; l < kCacheLayers; ++l)
      if (p.layer_ms[l] > 0) cls[layer_class(l)] += p.layer_ms[l], cnt[layer_class(l)] += 1;
    static const char* cn[kClasses] = {"dense (0-2)", "MoE, full indexer", "MoE, shared indexer", "MTP passes"};
    for (int c = 0; c < kClasses; ++c)
      if (cnt[c] > 0) record(rk, "profile\trows %d\tlayer class\t%s\t%d\t%.3f ms a layer\t%.3f ms", R, cn[c], (int)cnt[c],
                             cls[c] / cnt[c], cls[c]);
    const double coll = p.ms[kKAgX] + p.ms[kKAgQ] + p.ms[kKAr1] + p.ms[kKAr2];
    record(rk, "profile\trows %d\ttotal %.3f ms (eager, no PDL)\tcollectives %.3f ms\tattention compute %.3f ms", R,
           p.total_ms, coll, p.ms[kKPrep] + p.ms[kKIndex] + p.ms[kKAttn]);
    ring.barrier();
  }
}

// ab: two lever sets, A/B/B/A in one process, timing and output bits
uint64_t step_hash(Model& m, const SampleOut& o, int R, int n_cache) {
  const std::vector<uint16_t> h = m.last_hidden(R);
  uint64_t x = fnv(h.data(), h.size() * 2);
  for (int r = 0; r < R; ++r) x = fnv(&o.row[r].token, 4, x), x = fnv(&o.row[r].value, 4, x);
  const std::vector<uint64_t> c = m.cache_hashes(0, n_cache);
  return fnv(c.data(), c.size() * 8, x);
}

void ab(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  Sampling sp;
  const uint32_t cfg[2] = {A.ab_base, A.ab};
  const int rounds = A.ab_rounds, per = 12, keep = 8;
  for (int R : A.rows) {
    std::vector<int32_t> tok(R);
    for (int r = 0; r < R; ++r) tok[r] = synth_token(A.seed, r);
    const Batch b = one_session(R, A.ctx, tok.data(), 0, sp);
    m.fill_cache(0, A.ctx + 64, 7);
    uint64_t hs[2] = {};
    for (int k = 0; k < 2; ++k) {  // warm both (eager, capture, replay), and each one's output bits
      m.set_tune(cfg[k]);
      SampleOut o{};
      for (int i = 0; i < 3; ++i) o = m.wait(m.launch(b));
      hs[k] = step_hash(m, o, R, A.ctx + R);
    }
    ring.barrier();
    std::vector<double> ms[2], ratio;
    for (int round = 0; round < rounds; ++round) {
      double blk[2] = {};
      for (int q = 0; q < 2; ++q) {
        const int k = (q ^ round) & 1;  // A B, then B A
        m.set_tune(cfg[k]);
        std::vector<uint64_t> ns(per);
        for (int i = 0; i < per; ++i) ns[i] = m.launch(b);
        m.wait(ns.back());
        std::vector<double> g;
        for (int i = per - keep; i < per; ++i) g.push_back(m.gpu_ms(ns[i]));
        std::sort(g.begin(), g.end());
        blk[k] = (g[keep / 2 - 1] + g[keep / 2]) / 2;
        ms[k].push_back(blk[k]);
      }
      ratio.push_back(blk[1] / blk[0]);
    }
    const SampleOut o = m.wait(m.launch(b));  // B's bits again after the rounds
    const uint64_t hb = step_hash(m, o, R, A.ctx + R);
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
    };
    const double ma = med(ms[0]), mb = med(ms[1]), mr = med(ratio);
    double lo = 1e9, hi = 0;
    for (double r : ratio) lo = std::min(lo, r), hi = std::max(hi, r);
    uint64_t mine = hs[1], all[4];
    ring.allgather_host(&mine, all, 8);
    bool ranks = true;
    for (int r = 1; r < 4; ++r) ranks &= all[r] == all[0];
    record(rk, "ab\trows %d\tA %08x %.3f ms\tB %08x %.3f ms\t(medians of %d rounds, A B / B A)\tB/A median %.4f "
           "(%.4f .. %.4f)\tgain %.2f %%", R, cfg[0], ma, cfg[1], mb, rounds, mr, lo, hi, 100.0 * (1.0 - mr));
    check(rk, hs[0] == hs[1] && hb == hs[1] && ranks, "ab rows %d: the outputs of A and B the same bits (hash %016llx), every "
          "rank the same", R, (unsigned long long)hs[1]);
    ring.barrier();
  }
  m.set_tune(A.mo.tune);
}

// trace: inside one cudaProfilerStart/Stop (nsys --capture-range=cudaProfilerApi), for each of --rows: an eager step,
// the graph's capture, then 9 graph steps (tools/step_profile.py takes the last 4 of each row count's 10 steps)
void trace(Model& m, comm::Ring& ring) {
  m.fill_cache(0, A.ctx + 64, 7);
  Sampling sp;
  CUDA_CHECK(cudaProfilerStart());
  for (int R : A.rows) {
    std::vector<int32_t> tok(R);
    for (int r = 0; r < R; ++r) tok[r] = synth_token(A.seed, r);
    const Batch b = one_session(R, A.ctx, tok.data(), 0, sp);
    for (int i = 0; i < 2; ++i) m.wait(m.launch(b));
    m.drain();
    ring.barrier();
    std::vector<uint64_t> ns(9);
    for (int i = 0; i < 9; ++i) ns[i] = m.launch(b);
    m.wait(ns.back());
    m.drain();
    record(ring.rank(), "trace\trows %d\tgpu %.3f ms (last step)", R, m.gpu_ms(ns.back()));
    ring.barrier();
  }
  CUDA_CHECK(cudaProfilerStop());
}

void l2pf_ab(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  m.fill_cache(0, A.ctx + 64, 7);
  Sampling sp;
  for (int R : A.rows) {
    std::vector<int32_t> tok(R);
    for (int r = 0; r < R; ++r) tok[r] = synth_token(A.seed, r);
    const Batch b = one_session(R, A.ctx, tok.data(), 0, sp);
    double ms[2][2] = {};
    for (int order = 0; order < 2; ++order)
      for (int k = 0; k < 2; ++k) {
        const bool on = (k ^ order) != 0;
        m.set_l2pf(on);
        for (int i = 0; i < 3; ++i) m.wait(m.launch(b));
        ring.barrier();
        std::vector<uint64_t> ns(A.steps);
        for (int i = 0; i < A.steps; ++i) ns[i] = m.launch(b);
        m.wait(ns.back());
        std::vector<double> g;
        for (int i = std::max(0, A.steps - 4); i < A.steps; ++i) g.push_back(m.gpu_ms(ns[i]));
        std::sort(g.begin(), g.end());
        ms[order][on] = g[g.size() / 2];
        ring.barrier();
      }
    record(rk, "l2pf\trows %d\toff %.3f / %.3f ms\ton %.3f / %.3f ms\t(A/B, B/A)\tgain %.2f %%", R, ms[0][0], ms[1][0],
           ms[0][1], ms[1][1], 100.0 * (1.0 - (ms[0][1] + ms[1][1]) / (ms[0][0] + ms[1][0])));
  }
  m.set_l2pf(true);
}

void bench_prefill(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  for (int C : A.chunks) {
    if (C > m.options().prefill_rows) continue;
    std::vector<int32_t> tok(C);
    for (int i = 0; i < C; ++i) tok[i] = synth_token(A.seed + 3, i);
    m.fill_cache(0, A.ctx, 9);
    m.prefill(0, A.ctx, C, tok.data());  // warm
    m.drain();
    ring.barrier();
    m.reset_slot(0);
    m.prefill(0, A.ctx, C, tok.data());
    const float ms = m.prefill_ms();
    record(rk, "prefill\tchunk %d\tat %d\t%.1f ms\t%.0f tok/s", C, A.ctx, ms, C * 1e3 / ms);
    ring.barrier();
  }
}

// pfab: prefill A B B A at each (chunk, position), timing and output bits
void prefill_ab(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  const uint32_t cfg[2] = {A.ab_base, A.ab};
  Sampling sp;  // greedy
  // warm both lever sets (lazy module loads, the prefill buffers)
  {
    std::vector<int32_t> tok(512);
    for (int j = 0; j < 512; ++j) tok[j] = synth_token(A.seed + 5, j);
    for (int k = 0; k < 2; ++k) {
      m.set_tune(cfg[k]);
      m.reset_slot(0);
      m.wait(m.prefill(0, 0, std::min(512, m.options().prefill_rows), tok.data(), &sp));
      m.drain();
    }
  }
  for (size_t i = 0; i < A.chunks.size(); ++i) {
    const int C = A.chunks[i], at = i < A.ats.size() ? A.ats[i] : A.ctx;
    // a prompt longer than a chunk runs as prefill() splits it (chunks of prefill_rows; the timing is the whole call)
    std::vector<int32_t> tok(C);
    for (int j = 0; j < C; ++j) tok[j] = synth_token(A.seed + 3, at + j);
    double ms[2][2] = {};
    uint64_t h[2][2] = {};
    std::vector<uint64_t> lh[4];  // per layer, each run
    for (int q = 0; q < 4; ++q) {  // A B B A
      const int k = (q == 1 || q == 2) ? 1 : 0, o = q / 2;
      m.set_tune(cfg[k]);
      m.fill_cache(0, at, 9);
      m.reset_slot(0);
      ring.barrier();
      const SampleOut& so = m.wait(m.prefill(0, at, C, tok.data(), &sp));
      const int32_t t = so.row[0].token;
      const float v = so.row[0].value;
      ms[k][o] = m.prefill_ms();
      const std::vector<uint64_t> c = m.cache_hashes(0, at + C);
      lh[q] = c;
      uint64_t x = fnv(c.data(), c.size() * 8);
      x = fnv(&t, 4, x), x = fnv(&v, 4, x);
      h[k][o] = x;
      ring.barrier();
    }
    uint64_t mine = h[1][0], all[4];
    ring.allgather_host(&mine, all, 8);
    bool ranks = true;
    for (int r = 1; r < 4; ++r) ranks &= all[r] == all[0];
    const double a = (ms[0][0] + ms[0][1]) / 2, b = (ms[1][0] + ms[1][1]) / 2;
    record(rk, "pfab\tchunk %d\tat %d\tA %08x %.1f / %.1f ms\t%.0f tok/s\tB %08x %.1f / %.1f ms\t%.0f tok/s\t(A B B A)\t"
           "B/A %.3f\tdrafts %d", C, at, cfg[0], ms[0][0], ms[0][1], C * 1e3 / a, cfg[1], ms[1][0], ms[1][1], C * 1e3 / b,
           a / b, m.options().drafts);
    check(rk, h[0][0] == h[0][1] && h[1][0] == h[1][1] && h[0][0] == h[1][0] && ranks,
          "pfab chunk %d at %d: A and B the same bits (caches of positions 0-%d, the last row's sample; hash %016llx), "
          "every rank the same", C, at, at + C - 1, (unsigned long long)h[1][0]);
    if (!(h[0][0] == h[0][1] && h[1][0] == h[1][1] && h[0][0] == h[1][0])) {
      int fl[4] = {-1, -1, -1, -1};  // each run's first cache layer that differs from run 0's (-1: none)
      for (int q = 1; q < 4; ++q)
        for (size_t l = 0; l < lh[q].size(); ++l)
          if (lh[q][l] != lh[0][l]) { fl[q] = (int)l; break; }
      record(rk, "pfab\tchunk %d\tat %d\thashes A %016llx %016llx B %016llx %016llx\tfirst layer differing from run 0: %d %d %d", C,
             at, (unsigned long long)h[0][0], (unsigned long long)h[0][1], (unsigned long long)h[1][0],
             (unsigned long long)h[1][1], fl[1], fl[2], fl[3]);
    }
  }
  m.set_tune(A.mo.tune);
}

// pftrace: one prefill chunk inside the profiler range (nsys --capture-range=cudaProfilerApi)
void prefill_trace(Model& m, comm::Ring& ring) {
  const int C = A.chunks.at(0), at = A.ats.empty() ? A.ctx : A.ats[0];
  std::vector<int32_t> tok(C);
  for (int j = 0; j < C; ++j) tok[j] = synth_token(A.seed + 3, at + j);
  m.fill_cache(0, at, 9);
  m.reset_slot(0);
  m.prefill(0, at, C, tok.data());  // warm
  m.drain();
  m.fill_cache(0, at, 9);
  m.reset_slot(0);
  ring.barrier();
  CUDA_CHECK(cudaProfilerStart());
  m.prefill(0, at, C, tok.data());
  const float ms = m.prefill_ms();
  m.drain();
  CUDA_CHECK(cudaProfilerStop());
  record(ring.rank(), "pftrace\tchunk %d\tat %d\t%.1f ms\t%.0f tok/s", C, at, ms, C * 1e3 / ms);
  ring.barrier();
}

// ---- check (synthetic) ------------------------------------------------------------------------------------------

void check_state(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  const Options& o = m.options();
  if (o.slots < 6) die("check: --slots 6 at least");
  const int P = A.prompt ? A.prompt : 2600;  // crosses 2048: rows that select
  std::vector<int32_t> tok(P + 64);
  for (int i = 0; i < P + 64; ++i) tok[i] = synth_token(A.seed + 11, i);
  Sampling sp;
  // (1) one chunk vs two
  m.prefill(0, 0, P, tok.data());
  m.prefill(1, 0, P / 2, tok.data());
  m.prefill(1, P / 2, P - P / 2, tok.data() + P / 2);
  m.drain();
  const auto h0 = m.cache_hashes(0, P), h1 = m.cache_hashes(1, P);
  int same = 0;
  for (int l = 0; l < kCacheLayers; ++l) same += h0[l] == h1[l];
  check(rk, same == kCacheLayers, "prefill in one chunk = two chunks (%d + %d rows): %d of %d layers' caches identical", P / 2,
        P - P / 2, same, kCacheLayers);
  // (2) the same prompt by decode steps of 16 rows (each step repeats the row before it, so that its MTP entry is
  // written with the next token): caches against the prefill's
  int pos = 0;
  while (pos < P) {
    const int start = std::max(0, pos - 1), R = std::min(kMaxRows, P - start);
    m.launch(one_session(R, start, tok.data() + start, 2, sp));
    pos = start + R;
  }
  m.drain();
  const auto h2 = m.cache_hashes(2, P);
  int lead = 0;
  while (lead < kLayers && h2[lead] == h0[lead]) ++lead;
  same = 0;
  for (int l = 0; l < kCacheLayers; ++l) same += h2[l] == h0[l];
  check(rk, lead >= kDenseLayers + 1, "decode steps (16 rows) = prefill: the caches of layers 0-%d identical (the first "
        "MoE output feeds layer 4: routed sums in another order from there; %d of %d layers identical)", lead - 1, same,
        kCacheLayers);
  // (3) batch invariance and determinism: 16 rows in one step (eager, then graph) vs 16 steps of one row, after the
  // same prefix
  m.prefill(3, 0, P, tok.data());
  m.prefill(4, 0, P, tok.data());
  m.prefill(5, 0, P, tok.data());
  const SampleOut a = m.wait(m.launch(one_session(16, P, tok.data() + P, 3, sp)));
  const SampleOut c = m.wait(m.launch(one_session(16, P, tok.data() + P, 5, sp)));
  std::vector<SampleRow> one(16);
  for (int r = 0; r < 16; ++r) one[r] = m.wait(m.launch(one_session(1, P + r, tok.data() + P + r, 4, sp))).row[0];
  m.drain();
  int eq = 0, eqd = 0;
  for (int r = 0; r < 16; ++r) {
    eq += a.row[r].token == one[r].token && std::memcmp(&a.row[r].value, &one[r].value, 4) == 0;
    eqd += a.row[r].token == c.row[r].token && std::memcmp(&a.row[r].value, &c.row[r].value, 4) == 0;
  }
  const auto h3 = m.cache_hashes(3, P + 16), h4 = m.cache_hashes(4, P + 16), h5 = m.cache_hashes(5, P + 16);
  int s34 = 0, s35 = 0;
  for (int l = 0; l < kLayers; ++l) s34 += h3[l] == h4[l], s35 += h3[l] == h5[l];
  check(rk, eq == 16 && s34 == kLayers, "16 rows in one step = 16 steps of one row: %d of 16 rows' token and logit bits, "
        "%d of %d layers' caches", eq, s34, kLayers);
  check(rk, eqd == 16 && s35 == kLayers, "the 16-row step again (graph replay vs eager): %d of 16 rows, %d of %d layers' caches",
        eqd, s35, kLayers);
  // (4) every rank the same: the tokens and the cache hashes, all-gathered
  uint64_t mine[2] = {fnv(&a, sizeof a), fnv(h3.data(), h3.size() * 8)}, all[4][2];
  ring.allgather_host(mine, all, sizeof mine);
  bool agree = true;
  for (int r = 1; r < 4; ++r) agree &= all[r][0] == all[0][0] && all[r][1] == all[0][1];
  check(rk, agree, "the four ranks' samples and caches identical");
  // (5) sampling: temperature / top-p with top logprobs; every token a used id, every rank the same
  Sampling ts;
  ts.temperature = 0.8f, ts.top_p = 0.9f, ts.n_top = 20, ts.seed = 5;
  const SampleOut t = m.wait(m.launch(one_session(16, P + 16, tok.data() + P + 16, 3, ts)));
  bool ok = true;
  for (int r = 0; r < 16; ++r) {
    ok &= t.row[r].token >= 0 && t.row[r].token < kVocabUsed;
    double sum = 0;
    for (int i = 0; i < 20; ++i) ok &= t.top_id[r][i] < kVocabUsed, sum += std::exp(t.top_lp[r][i]);
    ok &= sum <= 1.0001 && std::isfinite(t.row[r].logprob);
  }
  uint64_t th = fnv(&t, sizeof t), tall[4];
  ring.allgather_host(&th, tall, 8);
  for (int r = 1; r < 4; ++r) ok &= tall[r] == tall[0];
  check(rk, ok, "sampled rows (T 0.8, top-p 0.9, 20 logprobs): used ids, probabilities <= 1, the same on every rank");
}

// ---- tf: teacher-forced dumps -----------------------------------------------------------------------------------

struct TfSink : Sink {
  int N = 0;
  std::vector<int32_t> tokens;
  std::vector<int32_t> argmax, top_id;
  std::vector<float> top_lp, lse, next_lp;
  std::vector<uint8_t> seen;
  std::set<int> logit_rows;
  std::map<int, std::vector<float>> logit_keep;
  std::set<int> sample_rows;
  std::map<int, std::map<int, std::vector<float>>> hidden;  // layer -> row -> values
  std::set<int> dump_layers;
  bool mtp_mode = false;
  // the MTP's: row t (hidden t, token t + 1) predicts token t + 2
  std::vector<int32_t> m_argmax, m_top_id;
  std::vector<float> m_top_lp;
  std::vector<uint8_t> m_seen;
  explicit TfSink(const std::vector<int32_t>& t) : N((int)t.size()), tokens(t) {
    argmax.assign(N, -1), top_id.assign((size_t)N * 20, -1), top_lp.assign((size_t)N * 20, 0.f);
    lse.assign(N, NAN), next_lp.assign(N, NAN), seen.assign(N, 0);
    m_argmax.assign(N, -1), m_top_id.assign((size_t)N * 20, -1), m_top_lp.assign((size_t)N * 20, 0.f), m_seen.assign(N, 0);
  }
  void logits(const int32_t* pos, int R, const float* full, bool mtp) override {
    for (int r = 0; r < R; ++r) {
      const int p = pos[r];
      if (p < 0 || p >= N) continue;
      if (mtp ? m_seen[p] : seen[p]) continue;
      const float* l = full + (size_t)r * kVocab;
      double mx = -INFINITY;
      for (int i = 0; i < kVocabUsed; ++i) mx = std::max(mx, (double)l[i]);
      double z = 0;
      for (int i = 0; i < kVocabUsed; ++i) z += std::exp((double)l[i] - mx);
      const double ls = mx + std::log(z);
      std::vector<int> idx(kVocabUsed);
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(),
                        [&](int x, int y) { return l[x] > l[y] || (l[x] == l[y] && x < y); });
      if (mtp) {
        m_seen[p] = 1, m_argmax[p] = idx[0];
        for (int i = 0; i < 20; ++i) m_top_id[(size_t)p * 20 + i] = idx[i], m_top_lp[(size_t)p * 20 + i] = (float)(l[idx[i]] - ls);
        continue;
      }
      seen[p] = 1, argmax[p] = idx[0], lse[p] = (float)ls;
      for (int i = 0; i < 20; ++i) top_id[(size_t)p * 20 + i] = idx[i], top_lp[(size_t)p * 20 + i] = (float)(l[idx[i]] - ls);
      if (p + 1 < N) next_lp[p] = (float)(l[tokens[p + 1]] - ls);
      if (logit_rows.count(p)) logit_keep[p] = std::vector<float>(l, l + kVocab);
    }
  }
  void tensor(int L, const char* name, const int32_t* pos, int R, const void* data, size_t row_bytes) override {
    // the MTP layer's diagnostics ("x", "attn", "router") are not the golden format's hidden rows
    if (!dump_layers.count(L) || !std::strcmp(name, "x") || !std::strcmp(name, "attn") || !std::strcmp(name, "router"))
      return;
    const uint16_t* b = static_cast<const uint16_t*>(data);
    for (int r = 0; r < R; ++r) {
      if (!sample_rows.count(pos[r]) || hidden[L].count(pos[r])) continue;
      std::vector<float> v(kDim);
      for (int c = 0; c < kDim; ++c) {
        const uint32_t u = (uint32_t)b[(size_t)r * kDim + c] << 16;
        std::memcpy(&v[c], &u, 4);
      }
      hidden[L][pos[r]] = v;
    }
    (void)row_bytes;
  }
  void write(const std::string& dir) {
    const std::string f = dir + "/final", mt = dir + "/mtp";
    std::filesystem::create_directories(f), std::filesystem::create_directories(mt);
    npy_write(f + "/argmax.npy", "<i4", {(size_t)N}, argmax.data(), 4);
    npy_write(f + "/top20_ids.npy", "<i4", {(size_t)N, 20}, top_id.data(), 4);
    npy_write(f + "/top20_logprobs.npy", "<f4", {(size_t)N, 20}, top_lp.data(), 4);
    npy_write(f + "/lse.npy", "<f4", {(size_t)N}, lse.data(), 4);
    npy_write(f + "/next_logprob.npy", "<f4", {(size_t)N}, next_lp.data(), 4);
    std::vector<int32_t> rows;
    std::vector<float> lg;
    for (auto& kv : logit_keep) rows.push_back(kv.first), lg.insert(lg.end(), kv.second.begin(), kv.second.end());
    npy_write(f + "/logits_rows.npy", "<i4", {rows.size()}, rows.data(), 4);
    npy_write(f + "/logits.npy", "<f4", {rows.size(), (size_t)kVocab}, lg.data(), 4);
    npy_write(dir + "/tokens.npy", "<i4", {(size_t)N}, tokens.data(), 4);
    if (mtp_mode) {
      npy_write(mt + "/argmax.npy", "<i4", {(size_t)N}, m_argmax.data(), 4);
      npy_write(mt + "/top20_ids.npy", "<i4", {(size_t)N, 20}, m_top_id.data(), 4);
      npy_write(mt + "/top20_logprobs.npy", "<f4", {(size_t)N, 20}, m_top_lp.data(), 4);
    }
    for (auto& L : hidden) {
      char d[64];
      std::snprintf(d, sizeof d, L.first < 0 ? "/final_hidden" : "/layers/L%02d", L.first);
      std::filesystem::create_directories(dir + d);
      std::vector<int32_t> rr;
      std::vector<float> v;
      for (auto& kv : L.second) rr.push_back(kv.first), v.insert(v.end(), kv.second.begin(), kv.second.end());
      npy_write(dir + d + "/hidden_rows.npy", "<i4", {rr.size()}, rr.data(), 4);
      npy_write(dir + d + "/hidden.npy", "<f4", {rr.size(), (size_t)kDim}, v.data(), 4);
    }
  }
};

void tf(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  if (A.tokens.empty()) die("tf: --tokens FILE");
  const std::vector<int32_t> tok = ring_tokens(ring, A.tokens);
  const int N = (int)tok.size();
  if (N + kMaxDrafts + 1 > m.options().max_ctx) die("tf: %d tokens, max_ctx %d", N, m.options().max_ctx);
  TfSink sink(tok);
  if (rk == 0 && !A.sample_rows.empty())
    for (int64_t r : glmrun::npy_read_ints(A.sample_rows)) sink.sample_rows.insert((int)r), sink.logit_rows.insert((int)r);
  else
    for (int r = 0; r < N; r += 256) sink.sample_rows.insert(r), sink.logit_rows.insert(r);
  for (int l : A.dump_layers) sink.dump_layers.insert(l);
  uint32_t flags = kFlagLogits | (A.dump_layers.empty() ? 0u : kFlagDump);
  const double t0 = now_s();
  if (A.via == "prefill") {
    m.prefill(0, 0, N, tok.data(), nullptr, flags, &sink);
    m.drain();
  } else {  // decode steps of 16 rows, each repeating the row before it (its MTP entry gets the next token)
    sink.mtp_mode = m.options().drafts > 0;
    if (sink.mtp_mode) flags |= kFlagMtpLogits;
    Sampling sp;
    int pos = 0;
    while (pos < N) {
      const int start = std::max(0, pos - 1), R = std::min(kMaxRows, N - start);
      Batch b = one_session(R, start, tok.data() + start, 0, sp);
      b.flags = flags;
      m.launch(b, &sink);
      pos = start + R;
    }
    m.drain();
  }
  const double sec = now_s() - t0;
  if (rk == 0) {
    if (A.dir.empty()) die("tf: --dir DIR");
    sink.write(A.dir);
    int agree = 0, cnt = 0;
    for (int p = 0; p + 1 < N; ++p) cnt += sink.seen[p], agree += sink.seen[p] && sink.argmax[p] == tok[p + 1];
    double lp = 0;
    for (int p = 0; p + 1 < N; ++p) lp += std::isfinite(sink.next_lp[p]) ? sink.next_lp[p] : 0;
    record(0, "tf\t%s\tvia %s\t%d tokens\t%.1f s\ttop-1 vs the next token %.4f\tmean logprob %.4f", A.tokens.c_str(),
           A.via.c_str(), N, sec, cnt ? (double)agree / cnt : 0.0, lp / std::max(1, N - 1));
  }
}

// ---- refhash: the engine's own outputs ---------------------------------------------------------------------------

void refhash(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  std::vector<int32_t> prompt;
  if (!A.tokens.empty()) prompt = ring_tokens(ring, A.tokens);
  else
    for (int i = 0; i < (A.prompt ? A.prompt : 300); ++i) prompt.push_back(synth_token(A.seed + 21, i));
  Sampling sp;
  sp.temperature = A.temp, sp.top_p = A.top_p, sp.seed = A.seed;
  const int P = (int)prompt.size();
  m.reset_slot(0);
  const double t0 = now_s();
  uint64_t n = m.prefill(0, 0, P, prompt.data(), &sp);
  SampleOut o = m.wait(n);
  const double tp = now_s() - t0;
  std::vector<int32_t> out;
  uint64_t h = 0xcbf29ce484222325ull;
  auto take = [&](const SampleOut& s) {
    out.push_back(s.row[0].token);
    h = fnv(&s.row[0].token, 4, h);
    h = fnv(&s.row[0].value, 4, h);
    for (int k = 0; k < s.drafts; ++k) h = fnv(&s.draft[0][k], 4, h);
  };
  take(o);
  const double t1 = now_s();
  int32_t fbk = -1;  // the previous step's row 0, filled in by the device
  uint64_t prev = 0;
  bool pending = false;
  int pos = P;
  for (int i = 1; i < A.steps && !is_stop(out.back()); ++i) {
    // queue the next step before the previous one's sample is known (one ahead), as the server does
    if (pending) {
      take(m.wait(prev));
      if (is_stop(out.back())) break;
    }
    prev = m.launch(1, pos++, &fbk, sp, 0);
    pending = true;
  }
  if (pending) take(m.wait(prev));
  const double td = now_s() - t1;
  uint64_t all[4];
  ring.allgather_host(&h, all, 8);
  bool agree = true;
  for (int r = 1; r < 4; ++r) agree &= all[r] == all[0];
  if (rk == 0) {
    std::string s;
    for (size_t i = 0; i < std::min<size_t>(out.size(), 24); ++i) s += std::to_string(out[i]) + " ";
    record(0, "refhash\tprompt %d\tT %.2f top-p %.2f\tdrafts %d\t%zu tokens\thash %016llx\tranks %s\tprefill %.1f ms\t"
           "decode %.2f ms a token\tfirst: %s", P, A.temp, A.top_p, m.options().drafts, out.size(), (unsigned long long)h,
           agree ? "agree" : "DIFFER", tp * 1e3, td * 1e3 / std::max<size_t>(1, out.size() - 1), s.c_str());
  }
  if (!agree) ++g_fail;
}

// ---- the golden prompts (docs/reference.md § 7: raw copies made by tools/gold_prep.py) ----------------------------

struct GoldSeq {
  std::string name;
  int off = 0, n = 0;
};
std::vector<GoldSeq> gold_seqs(const std::string& prefix) {
  std::vector<GoldSeq> v;
  FILE* f = std::fopen((prefix + ".seqs.txt").c_str(), "r");
  if (!f) die("cannot read %s.seqs.txt", prefix.c_str());
  char name[64];
  int off, n;
  while (std::fscanf(f, "%63s %d %d", name, &off, &n) == 3) v.push_back({name, off, n});
  std::fclose(f);
  return v;
}
template <class T>
std::vector<T> read_raw(const std::string& path, size_t off_elems = 0, size_t n_elems = SIZE_MAX) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) die("cannot read %s", path.c_str());
  std::fseek(f, 0, SEEK_END);
  const size_t all = (size_t)std::ftell(f) / sizeof(T);
  if (off_elems > all) die("%s: %zu elements, offset %zu", path.c_str(), all, off_elems);
  n_elems = std::min(n_elems, all - off_elems);
  std::vector<T> v(n_elems);
  std::fseek(f, (long)(off_elems * sizeof(T)), SEEK_SET);
  if (std::fread(v.data(), sizeof(T), n_elems, f) != n_elems) die("%s: short read", path.c_str());
  std::fclose(f);
  return v;
}
std::vector<std::string> split_names(const std::string& s) {
  std::vector<std::string> v;
  size_t p = 0;
  while (p <= s.size()) {
    const size_t e = s.find(',', p);
    v.push_back(s.substr(p, e == std::string::npos ? std::string::npos : e - p));
    if (e == std::string::npos) break;
    p = e + 1;
  }
  return v;
}

// ---- mtpgold: the MTP layer teacher-forced on the golden final hidden ---------------------------------------------
// Every rank reads the same files (--gold PREFIX on every box). Each sequence of --seqs: its MTP rows t = 0 .. n - 2
// (hidden t after model.norm, token t + 1) in blocks of 16 rows from position 0 of slot 0 (Model::mtp_forced), the
// pass-1 draft of every row, and --passes - 1 chained drafts from each block's last row. Against the golden data:
// the draft against the reference's MTP argmax (mtp/argmax.npy), against the main model's argmax at t + 1 (the
// one-draft greedy acceptance on the model's own text) and the text's token t + 2; the chains' accepted lengths
// against the main argmax and against the text.
// rank 0: layer 78's output rows against the golden ones (PREFIX.l78.bf16: by main row, zeros where not saved)
struct L78Sink : Sink {
  const uint16_t* gold = nullptr;  // the sequence's rows
  int n = 0;
  double cos_sum = 0, cos_min = 1, rel_sum = 0;
  long rows = 0;
  std::vector<uint16_t> gx;  // PREFIX.x16.bf16: the layer's input of the first rows (diagnostics), if there
  std::string first;         // the first rows' cosines
  std::vector<uint16_t> ga;  // PREFIX.row0.bf16: row 0's residual after attention and output (diagnostics), if there
  static double bf(uint16_t u) {
    const uint32_t x = (uint32_t)u << 16;
    float v;
    std::memcpy(&v, &x, 4);
    return (double)v;
  }
  void tensor(int L, const char* name, const int32_t* pos, int R, const void* data, size_t) override {
    if (L != kMtpLayer || !gold) return;
    if (!std::strcmp(name, "router")) {  // row 0's top-12 logits
      const float* lg = static_cast<const float*>(data);
      for (int r = 0; r < R; ++r) {
        if (pos[r] != 0) continue;
        const float* l = lg + (size_t)r * 256;
        std::vector<int> id(256);
        std::iota(id.begin(), id.end(), 0);
        std::partial_sort(id.begin(), id.begin() + 12, id.end(), [&](int a, int b) { return l[a] > l[b]; });
        first += " router0:";
        for (int i = 0; i < 12; ++i) {
          char b[32];
          std::snprintf(b, sizeof b, " %d=%.3f", id[i], l[id[i]]);
          first += b;
        }
      }
      return;
    }
    if (!std::strcmp(name, "attn")) {  // row 0 against PREFIX.row0.bf16's first row (the residual after attention)
      if (ga.size() < (size_t)kDim) return;
      for (int r = 0; r < R; ++r) {
        if (pos[r] != 0) continue;
        const uint16_t* b = static_cast<const uint16_t*>(data) + (size_t)r * kDim;
        double gg = 0, mm = 0, gm = 0, dd = 0;
        for (int c = 0; c < kDim; ++c) {
          const double x = bf(ga[c]), y = bf(b[c]);
          gg += x * x, mm += y * y, gm += x * y, dd += (x - y) * (x - y);
        }
        char t[96];
        std::snprintf(t, sizeof t, " attn0 %.5f/%.4f", gm / std::sqrt(gg * mm + 1e-30), std::sqrt(dd / gg));
        first += t;
      }
      return;
    }
    const bool is_x = std::strcmp(name, "x") == 0;
    if (is_x && gx.empty()) return;
    const uint16_t* b = static_cast<const uint16_t*>(data);
    auto f = [](uint16_t u) {
      const uint32_t x = (uint32_t)u << 16;
      float v;
      std::memcpy(&v, &x, 4);
      return (double)v;
    };
    for (int r = 0; r < R; ++r) {
      if (pos[r] < 0 || pos[r] >= n) continue;
      if (is_x && (size_t)(pos[r] + 1) * kDim > gx.size()) continue;
      const uint16_t* g = is_x ? gx.data() + (size_t)pos[r] * kDim : gold + (size_t)pos[r] * kDim;
      double gg = 0, mm = 0, gm = 0, dd = 0;
      for (int c = 0; c < kDim; ++c) {
        const double x = f(g[c]), y = f(b[(size_t)r * kDim + c]);
        gg += x * x, mm += y * y, gm += x * y, dd += (x - y) * (x - y);
      }
      if (gg == 0) continue;  // not saved
      const double cs = gm / std::sqrt(gg * mm + 1e-30);
      if (pos[r] < 8) {
        char b[96];
        std::snprintf(b, sizeof b, " %s%d %.4f/%.3f", is_x ? "x" : "h", pos[r], cs, std::sqrt(dd / gg));
        first += b;
      }
      if (is_x) continue;
      cos_sum += cs, cos_min = std::min(cos_min, cs), rel_sum += std::sqrt(dd / gg), ++rows;
    }
  }
};

void mtpgold(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  const bool have_l78 = rk == 0 && std::filesystem::exists(A.gold + ".l78.bf16");
  if (A.gold.empty()) die("mtpgold: --gold PREFIX");
  const std::vector<GoldSeq> seqs = gold_seqs(A.gold);
  const std::vector<int32_t> tok = read_raw<int32_t>(A.gold + ".tokens.i32");
  const std::vector<int32_t> am = read_raw<int32_t>(A.gold + ".argmax.i32");
  const std::vector<int32_t> mam = read_raw<int32_t>(A.gold + ".mtp_argmax.i32");
  const std::vector<std::string> want = split_names(A.seqs.empty() ? "short,prose,python,agentic" : A.seqs);
  const int passes = std::max(1, std::min(m.options().drafts, A.passes));
  const int B = std::clamp(A.block, 1, kMaxRows);  // rows a teacher-forced block; a chain from each block's last row
  struct Tot {
    long rows = 0, ref = 0, main = 0, text = 0, chains = 0;
    long chain_main[kMaxDrafts + 1] = {}, chain_text[kMaxDrafts + 1] = {};
  } all;
  for (const GoldSeq& s : seqs) {
    if (std::find(want.begin(), want.end(), s.name) == want.end()) continue;
    const int n = s.n;
    if (n + kMaxDrafts + 2 > m.options().max_ctx) die("mtpgold: %s has %d tokens, max_ctx %d", s.name.c_str(), n, m.options().max_ctx);
    const std::vector<uint16_t> hid = read_raw<uint16_t>(A.gold + ".hfinal.bf16", (size_t)s.off * kDim, (size_t)n * kDim);
    std::vector<uint16_t> l78;
    L78Sink sink;
    if (have_l78) {
      l78 = read_raw<uint16_t>(A.gold + ".l78.bf16", (size_t)s.off * kDim, (size_t)n * kDim);
      sink.gold = l78.data(), sink.n = n;
      if (std::filesystem::exists(A.gold + ".x16.bf16")) sink.gx = read_raw<uint16_t>(A.gold + ".x16.bf16");
      if (std::filesystem::exists(A.gold + ".row0.bf16")) sink.ga = read_raw<uint16_t>(A.gold + ".row0.bf16");
    }
    m.reset_slot(0);
    Tot t;
    std::vector<int32_t> d1(B), chain(kMaxDrafts);
    const double t0 = now_s();
    for (int r0 = 0; r0 + 1 < n; r0 += B) {  // MTP rows r0 .. r0 + R - 1: hidden r, token r + 1
      const int R = std::min(B, n - 1 - r0);
      m.mtp_forced(0, r0, R, tok.data() + s.off + r0 + 1, hid.data() + (size_t)r0 * kDim, passes, d1.data(), chain.data(),
                   have_l78 ? &sink : nullptr);
      for (int r = 0; r < R; ++r) {
        const int g = s.off + r0 + r;  // the main row
        if (r0 + r + 2 >= n) continue;  // no token t + 2
        ++t.rows;
        t.ref += mam[g] >= 0 && d1[r] == mam[g];
        t.main += d1[r] == am[g + 1];
        t.text += d1[r] == tok[g + 2];
      }
      // the chain from the block's last row t: draft j (1 .. passes) is the token at t + j + 1
      const int tl = r0 + R - 1;
      if (tl + passes + 1 < n) {
        ++t.chains;
        int lm = 0, lt = 0;
        bool om = true, ot = true;
        for (int j = 1; j <= passes; ++j) {
          const int32_t d = j == 1 ? d1[R - 1] : chain[j - 2];
          const int g = s.off + tl;
          om = om && d == am[g + j];
          ot = ot && d == tok[g + j + 1];
          lm += om, lt += ot;
        }
        ++t.chain_main[lm], ++t.chain_text[lt];
      }
    }
    const double sec = now_s() - t0;
    auto frac = [](long a, long b) { return b ? (double)a / b : 0.0; };
    auto lens = [&](const long* h) {
      std::string o;
      double mean = 0;
      for (int j = 0; j <= passes; ++j) {
        char b[32];
        std::snprintf(b, sizeof b, "%s%.3f", j ? " " : "", frac(h[j], t.chains));
        o += b;
        mean += j * frac(h[j], t.chains);
      }
      char b[48];
      std::snprintf(b, sizeof b, " (mean %.3f)", mean);
      o += b;
      // a verify step of k drafts accepts min(L, k): tokens a step 1 + sum_{j <= k} P(L >= j)
      o += "; tokens a step at k = 1..";
      o += std::to_string(passes) + ":";
      double tps = 1;
      for (int k = 1; k <= passes; ++k) {
        double ge = 0;
        for (int j = k; j <= passes; ++j) ge += frac(h[j], t.chains);
        tps += ge;
        std::snprintf(b, sizeof b, " %.3f", tps);
        o += b;
      }
      return o;
    };
    record(rk, "mtpgold\t%s\t%ld rows\t%.1f s\tdraft = reference MTP argmax %.4f\tdraft = main argmax at t+1 %.4f\t"
           "draft = token t+2 %.4f\tchains %ld (passes %d): accepted 0..%d vs main argmax %s\tvs text %s",
           s.name.c_str(), t.rows, sec, frac(t.ref, t.rows), frac(t.main, t.rows), frac(t.text, t.rows), t.chains, passes,
           passes, lens(t.chain_main).c_str(), lens(t.chain_text).c_str());
    if (sink.rows)
      record(rk, "mtpgold\t%s\tlayer 78's output against the golden rows\t%ld rows\tcosine min %.5f mean %.5f\trelative error "
             "%.4f\tfirst rows (cosine/relative error):%s", s.name.c_str(), sink.rows, sink.cos_min, sink.cos_sum / sink.rows,
             sink.rel_sum / sink.rows, sink.first.c_str());
    all.rows += t.rows, all.ref += t.ref, all.main += t.main, all.text += t.text, all.chains += t.chains;
    for (int j = 0; j <= kMaxDrafts; ++j) all.chain_main[j] += t.chain_main[j], all.chain_text[j] += t.chain_text[j];
  }
  record(rk, "mtpgold\tall\t%ld rows\tdraft = reference MTP argmax %.4f\tdraft = main argmax at t+1 %.4f\tdraft = token t+2 "
         "%.4f", all.rows, all.rows ? (double)all.ref / all.rows : 0.0, all.rows ? (double)all.main / all.rows : 0.0,
         all.rows ? (double)all.text / all.rows : 0.0);
  // every rank the same drafts: the counts all-gathered
  int64_t mine[3] = {all.ref, all.main, all.text}, every[4][3];
  ring.allgather_host(mine, every, sizeof mine);
  bool agree = true;
  for (int r = 1; r < 4; ++r) agree &= std::memcmp(every[r], every[0], sizeof mine) == 0;
  check(rk, agree, "mtpgold: the four ranks' drafts agree");
}

// ---- spec: speculative decoding against plain decoding ------------------------------------------------------------
// The prompts of --seqs from the golden set (--gold PREFIX on rank 0), each prefilled once into a slot of its own
// with a greedy sample (its first token and drafts). A run copies the prompts' caches into working slots and decodes
// --gen tokens a session for --sess sessions (session j on prompt (t + j) mod prompts), greedy or seeded sampling
// (T 0.8, top-p 0.95, seed 1000 + j): plain (one row a session a step, no MTP, queued one step ahead) and speculative
// at each of --ks (k drafts verified a step, k passes; -1: the gate, SpecGate with --spec-cost), in the order
// plain, ks, ks reversed, plain (both orders). Every speculative run's tokens, logit values and log-probabilities
// against the plain run's, bit for bit; GPU ms a step, wall tok/s, acceptance, tokens a step. k = -3, the oracle:
// --kmax drafts a step taken from the plain run's tokens with about one in four changed (so the verify steps accept
// 0 .. kmax drafts whatever the MTP's quality: with synthetic routed experts the MTP's own drafts are never accepted).
struct Stream {
  std::vector<int32_t> tok;
  std::vector<uint32_t> bits;  // value and logprob bits of each token
};
struct RunStat {
  int steps = 0;
  long emitted = 0, verified = 0, accepted = 0, rows = 0;
  double gpu_ms = 0, wall_s = 0;
  long hist_acc[kMaxDrafts + 2] = {};  // steps by drafts accepted (single-session runs)
};
struct SpecPrompt {
  int slot = 0, P = 0;
  int32_t x0 = 0;
  std::vector<int32_t> d0;
  std::string name;
};

void take(Stream& s, const SampleRow& r) {
  s.tok.push_back(r.token);
  uint32_t v, l;
  std::memcpy(&v, &r.value, 4), std::memcpy(&l, &r.logprob, 4);
  s.bits.push_back(v), s.bits.push_back(l);
}

void spec(Model& m, comm::Ring& ring) {
  const int rk = ring.rank();
  if (A.gold.empty()) die("spec: --gold PREFIX");
  std::vector<GoldSeq> seqs;
  std::vector<int32_t> all_tok;
  if (rk == 0) seqs = gold_seqs(A.gold), all_tok = read_raw<int32_t>(A.gold + ".tokens.i32");
  const std::vector<std::string> want = split_names(A.seqs.empty() ? "short,python" : A.seqs);
  const Options& o = m.options();
  const int np = (int)want.size(), nmax = *std::max_element(A.sess.begin(), A.sess.end());
  if (np + nmax > o.slots) die("spec: %d prompts and %d sessions need %d slots (--slots)", np, nmax, np + nmax);
  if (o.drafts < 1) die("spec: --drafts K (the most draft passes)");
  // the prompts: rank 0's tokens to every rank, prefilled, sampled greedily (their first token and drafts)
  std::vector<SpecPrompt> pr(np);
  Sampling greedy;
  for (int i = 0; i < np; ++i) {
    std::vector<int32_t> t;
    if (rk == 0) {
      for (const GoldSeq& s : seqs)
        if (s.name == want[i]) t.assign(all_tok.begin() + s.off, all_tok.begin() + s.off + s.n);
      if (t.empty()) die("spec: no sequence %s in %s", want[i].c_str(), A.gold.c_str());
    }
    int64_t n = (int64_t)t.size(), every[4];
    ring.allgather_host(&n, every, 8);
    t.resize(every[0]);
    for (int64_t o = 0; o < every[0]; o += 16384) {  // pieces of 64 KiB (the bootstrap sockets' limit)
      const int64_t m = std::min<int64_t>(16384, every[0] - o);
      std::vector<int32_t> buf(4 * m);
      ring.allgather_host(t.data() + o, buf.data(), (size_t)m * 4);
      std::copy(buf.begin(), buf.begin() + m, t.begin() + o);
    }
    SpecPrompt& p = pr[i];
    p.slot = i, p.P = (int)t.size(), p.name = want[i];
    if (p.P + A.steps + 4 * kMaxDrafts + 8 > o.max_ctx) die("spec: %s: %d tokens + %d, max_ctx %d", want[i].c_str(), p.P, A.steps, o.max_ctx);
    m.reset_slot(p.slot);
    const double t0 = now_s();
    const SampleOut so = m.wait(m.prefill(p.slot, 0, p.P, t.data(), &greedy));
    m.drain();
    p.x0 = so.row[0].token;
    for (int k = 0; k < so.drafts; ++k) p.d0.push_back(so.draft[0][k]);
    record(rk, "spec\tprompt %s\t%d tokens\tprefill %.1f s\tfirst token %d\tdrafts %d", p.name.c_str(), p.P, now_s() - t0,
           p.x0, (int)p.d0.size());
  }
  const int N = A.steps;
  SpecCost cost;
  if (A.spec_cost.size() == 2) cost.c1 = A.spec_cost[0], cost.drow = A.spec_cost[1];
  // one run: n sessions on prompts t0 + j; k: -2 plain, -1 gate, >= 0 k drafts verified a step
  auto run = [&](int n, int t0, bool sampled, int k, std::vector<Stream>& out, RunStat& st, const std::vector<Stream>* oref) {
    out.assign(n, Stream{});
    st = RunStat{};
    std::vector<int> pos(n);
    std::vector<int32_t> x(n);
    std::vector<std::vector<int32_t>> D(n);
    std::vector<SpecGate> gate(n, SpecGate(A.kmax));
    std::vector<Sampling> sp(n);
    for (int j = 0; j < n; ++j) {
      const SpecPrompt& p = pr[(t0 + j) % np];
      m.cache_copy(np + j, p.slot, p.P);
      pos[j] = p.P, x[j] = p.x0, D[j] = p.d0;
      if (sampled) sp[j].temperature = 0.8f, sp[j].top_p = 0.95f, sp[j].seed = 1000 + j;
    }
    m.drain();
    ring.barrier();
    const double w0 = now_s();
    if (k == -2) {  // plain: every session a row a step, the next step queued before the last one's sample is known
      std::deque<uint64_t> q;
      for (int i = 0; i < N; ++i) {
        Batch b;
        b.R = n, b.nsess = n, b.passes = 0;
        for (int j = 0; j < n; ++j)
          b.tokens[j] = i == 0 ? x[j] : -1 - j, b.pos[j] = pos[j] + i, b.slot[j] = np + j, b.sess[j] = j, b.sp[j] = sp[j];
        q.push_back(m.launch(b));
        if (q.size() == 2) {
          const SampleOut& so = m.wait(q.front());
          st.gpu_ms += m.gpu_ms(q.front());
          for (int j = 0; j < n; ++j) take(out[j], so.row[j]);
          q.pop_front();
          ++st.steps;
        }
      }
      while (!q.empty()) {
        const SampleOut& so = m.wait(q.front());
        st.gpu_ms += m.gpu_ms(q.front());
        for (int j = 0; j < n; ++j) take(out[j], so.row[j]);
        q.pop_front();
        ++st.steps;
      }
      st.emitted = (long)n * N, st.rows = (long)n * N;
    } else {  // speculative: a verify step at a time
      int passes = k > 0 ? k : 1;
      for (;;) {
        Batch b;
        b.spec = 1;
        std::vector<int> first(n, -1), kk(n, 0), sess_of(n, -1);
        int ns = 0;
        for (int j = 0; j < n; ++j) {
          if ((int)out[j].tok.size() >= N) continue;
          if (k == -3) {  // the oracle: the plain run's next tokens, one in four changed (by a hash of the position)
            const int e = (int)out[j].tok.size();
            D[j].clear();
            for (int r = 1; r <= A.kmax && e + r - 1 < N; ++r) {
              int32_t t = (*oref)[j].tok[e + r - 1];
              if (oracle_changed((uint32_t)(e + r))) t = (t + 1) % kVocabUsed;
              D[j].push_back(t);
            }
          }
          int kj = k == -1 ? gate[j].choose(cost, (int)D[j].size()) : k == -3 ? (int)D[j].size() : std::min<int>(k, (int)D[j].size());
          kj = std::min(kj, kMaxRows - b.R - 1);
          first[j] = b.R, kk[j] = kj, sess_of[j] = ns;
          for (int r = 0; r <= kj; ++r) {
            b.tokens[b.R] = r == 0 ? x[j] : D[j][r - 1];
            b.pos[b.R] = pos[j] + r, b.slot[b.R] = np + j, b.sess[b.R] = ns;
            ++b.R;
          }
          b.sp[ns++] = sp[j];
        }
        if (!ns) break;
        b.nsess = ns;
        if (k == -3) passes = A.kmax;
        if (k == -1) {  // the drafts the next step may verify
          passes = 1;
          for (int j = 0; j < n; ++j)
            if (first[j] >= 0) passes = std::max(passes, gate[j].choose(cost, A.kmax));
        }
        b.passes = passes;
        const uint64_t id = m.launch(b);
        const SampleOut so = m.wait(id);
        st.gpu_ms += m.gpu_ms(id);
        ++st.steps;
        st.rows += b.R;
        for (int j = 0; j < n; ++j) {
          if (first[j] < 0) continue;
          const int s = sess_of[j], a = so.accepted[s];
          if (a < 0 || a > kk[j]) die("spec: session %d accepted %d of %d", j, a, kk[j]);
          for (int i = 0; i <= a && (int)out[j].tok.size() < N; ++i) take(out[j], so.row[first[j] + i]), ++st.emitted;
          st.verified += kk[j], st.accepted += a;
          if (n == 1) ++st.hist_acc[std::min(a, kMaxDrafts + 1)];
          gate[j].update(kk[j], a, kk[j] == 0 && !D[j].empty() ? (int)(D[j][0] == so.row[first[j]].token) : -1);
          x[j] = so.row[first[j] + a].token, pos[j] += a + 1;
          D[j].assign(so.draft[s], so.draft[s] + so.drafts);
        }
      }
    }
    st.wall_s = now_s() - w0;
  };
  const std::vector<int> ks = A.ks;
  for (int n : A.sess) {
    const int types = n == 1 ? np : 1;  // one session: each prompt; more: prompts mixed, from the first
    for (int t0 = 0; t0 < types; ++t0)
      for (int sampled = 0; sampled < 2; ++sampled) {
        if (!(A.spec_samp & (1 << sampled))) continue;
        // plain, ks, ks reversed, plain
        std::vector<int> order = {-2};
        for (int k : ks) order.push_back(k);
        for (auto it = ks.rbegin(); it != ks.rend(); ++it) order.push_back(*it);
        order.push_back(-2);
        std::map<int, std::vector<RunStat>> stats;
        std::vector<Stream> ref;
        bool all_same = true;
        std::string names;
        for (int j = 0; j < n; ++j) names += (j ? "+" : "") + pr[(t0 + j) % np].name;
        for (size_t oi = 0; oi < order.size(); ++oi) {
          const int k = order[oi];
          std::vector<Stream> out;
          RunStat st;
          run(n, t0, sampled != 0, k, out, st, &ref);
          stats[k].push_back(st);
          if (oi == 0) {
            ref = out;
            continue;
          }
          int same = 0;
          for (int j = 0; j < n; ++j) same += out[j].tok == ref[j].tok && out[j].bits == ref[j].bits;
          all_same &= same == n;
          if (same != n && rk == 0)
            for (int j = 0; j < n; ++j)
              if (out[j].tok != ref[j].tok || out[j].bits != ref[j].bits) {
                size_t i = 0;
                while (i < out[j].tok.size() && i < ref[j].tok.size() && out[j].tok[i] == ref[j].tok[i] &&
                       out[j].bits[2 * i] == ref[j].bits[2 * i] && out[j].bits[2 * i + 1] == ref[j].bits[2 * i + 1])
                  ++i;
                record(0, "spec\tDIFFER\t%s\tsession %d\tk %d\tfirst difference at token %zu of %zu", names.c_str(), j, k, i,
                       out[j].tok.size());
              }
        }
        uint64_t h = 0xcbf29ce484222325ull;
        for (const Stream& s : ref) h = fnv(s.tok.data(), s.tok.size() * 4, h);
        uint64_t every[4];
        ring.allgather_host(&h, every, 8);
        bool ranks = true;
        for (int r = 1; r < 4; ++r) ranks &= every[r] == every[0];
        check(rk, all_same && ranks, "spec %s, %d session%s, %s: every speculative run (k %s) emits the plain run's tokens, "
              "logit values and logprobs bit for bit (%d tokens a session; hash %016llx, every rank the same)",
              names.c_str(), n, n > 1 ? "s" : "", sampled ? "sampled T 0.8 top-p 0.95" : "greedy", [&] {
                std::string s;
                for (int k : ks) s += (s.empty() ? "" : ",") + (k == -1 ? std::string("gate") : k == -3 ? std::string("oracle") : std::to_string(k));
                return s;
              }().c_str(), N, (unsigned long long)h);
        // the measurements: both runs of each configuration
        const std::vector<RunStat>& pl = stats[-2];
        const double plain_ms = (pl[0].gpu_ms / pl[0].steps + pl[1].gpu_ms / pl[1].steps) / 2;
        const double plain_tps = (pl[0].emitted / pl[0].wall_s + pl[1].emitted / pl[1].wall_s) / 2;
        record(rk, "spec\t%s\tn %d\t%s\tplain\tsteps %d\tgpu %.2f / %.2f ms a step\twall tok/s %.2f / %.2f (all sessions)",
               names.c_str(), n, sampled ? "sampled" : "greedy", pl[0].steps, pl[0].gpu_ms / pl[0].steps,
               pl[1].gpu_ms / pl[1].steps, pl[0].emitted / pl[0].wall_s, pl[1].emitted / pl[1].wall_s);
        for (int k : ks) {
          const std::vector<RunStat>& v = stats[k];
          double tps = 0, ms = 0, tpst = 0, acc = 0, rows = 0;
          for (const RunStat& s : v) {
            tps += s.emitted / s.wall_s / v.size(), ms += s.gpu_ms / s.steps / v.size();
            tpst += (double)s.emitted / s.steps / n / v.size(), rows += (double)s.rows / s.steps / v.size();
            acc += (s.verified ? (double)s.accepted / s.verified : 0.0) / v.size();
          }
          std::string hist;
          if (n == 1)
            for (int a = 0; a <= kMaxDrafts; ++a) {
              char b[24];
              std::snprintf(b, sizeof b, "%s%ld", a ? "/" : "", v[0].hist_acc[a] + v[1].hist_acc[a]);
              hist += b;
            }
          record(rk, "spec\t%s\tn %d\t%s\tk %s\tsteps %d / %d\tgpu %.2f ms a step (%.2f rows)\ttokens a step a session %.3f\t"
                 "accepted %.3f of drafts\twall tok/s %.2f (plain %.2f, x %.3f)\tgpu-time tok/s %.2f (plain %.2f, x %.3f)\t"
                 "accepted 0..%d: %s",
                 names.c_str(), n, sampled ? "sampled" : "greedy", k == -1 ? "gate" : k == -3 ? "oracle" : std::to_string(k).c_str(), v[0].steps,
                 v[1].steps, ms, rows, tpst, acc, tps, plain_tps, tps / plain_tps, 1e3 * tpst * n / ms, 1e3 * n / plain_ms,
                 (tpst * n / ms) / (n / plain_ms), kMaxDrafts, hist.c_str());
        }
        ring.barrier();
      }
  }
}

}  // namespace

int main(int argc, char** argv) {
  A.mo.slots = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() -> std::string {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--rank") A.rank = std::stoi(nx());
    else if (a == "--fleet") A.fleet = nx();
    else if (a == "--weights") A.weights = expand(nx());
    else if (a == "--synthetic") A.synthetic = true;
    else if (a == "--fill-experts") A.fill_experts = true;
    else if (a == "--pools") A.pools = std::stoi(nx());
    else if (a == "--s3") A.s3 = true;
    else if (a == "--seed") A.seed = std::stoull(nx());
    else if (a == "--ctx") A.ctx = std::stoi(nx());
    else if (a == "--max-ctx") A.mo.max_ctx = std::stoi(nx());
    else if (a == "--slots") A.mo.slots = std::stoi(nx());
    else if (a == "--drafts") A.mo.drafts = std::stoi(nx());
    else if (a == "--no-l2pf") A.mo.l2pf = false;
    else if (a == "--no-graphs") A.mo.graphs = false;
    else if (a == "--no-pdl") A.mo.pdl = false;
    else if (a == "--fp8-keys") A.mo.fp8_index_keys = true;
    else if (a == "--chunk") A.mo.prefill_rows = std::stoi(nx());
    else if (a == "--pf-pieces") A.mo.prefill_pieces = std::stoi(nx());
    else if (a == "--pf-blocks") A.mo.prefill_comm_blocks = std::stoi(nx());
    else if (a == "--pf-min-piece") A.mo.prefill_min_piece = std::stoi(nx());
    else if (a == "--splits") A.mo.attn_splits = std::stoi(nx());
    else if (a == "--rows") A.rows = parse_list(nx());
    else if (a == "--tune") A.mo.tune = (uint32_t)std::stoul(nx(), nullptr, 16);
    else if (a == "--ab") A.ab = (uint32_t)std::stoul(nx(), nullptr, 16);
    else if (a == "--ab-rounds") A.ab_rounds = std::stoi(nx());
    else if (a == "--ab-base") A.ab_base = (uint32_t)std::stoul(nx(), nullptr, 16);
    else if (a == "--chunks") A.chunks = parse_list(nx());
    else if (a == "--at") A.ats = parse_list(nx());
    else if (a == "--steps") A.steps = std::stoi(nx());
    else if (a == "--prompt") A.prompt = std::stoi(nx());
    else if (a == "--temp") A.temp = std::stof(nx());
    else if (a == "--top-p") A.top_p = std::stof(nx());
    else if (a == "--out") A.out = expand(nx());
    else if (a == "--tag") A.tag = nx();
    else if (a == "--dir") A.dir = expand(nx());
    else if (a == "--tokens") A.tokens = expand(nx());
    else if (a == "--sample-rows") A.sample_rows = expand(nx());
    else if (a == "--via") A.via = nx();
    else if (a == "--dump-layers") A.dump_layers = parse_list(nx());
    else if (a == "--gold") A.gold = expand(nx());
    else if (a == "--seqs") A.seqs = nx();
    else if (a == "--passes") A.passes = std::stoi(nx());
    else if (a == "--sess") A.sess = parse_list(nx());
    else if (a == "--ks") A.ks = parse_list(nx());
    else if (a == "--kmax") A.kmax = std::stoi(nx());
    else if (a == "--block") A.block = std::stoi(nx());
    else if (a == "--spec-samp") A.spec_samp = std::stoi(nx());
    else if (a == "--spec-cost") {
      for (const int v : parse_list(nx())) A.spec_cost.push_back((float)v);
    }
    else if (a[0] == '-') die("unknown option %s", a.c_str());
    else A.modes.push_back(a);
  }
  if (A.modes.empty()) die("glm_run: a mode (bench l2pf prefill check tf refhash)");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  comm::Options co;
  co.fleet = A.fleet;
  co.rank = A.rank;
  co.cap = 4 << 20;
  co.bootstrap_timeout_s = 120;
  comm::Ring ring(co);
  const int rk = ring.rank();
  if (rk == 0 && !A.out.empty() && !(g_out = std::fopen(A.out.c_str(), "a"))) die("cannot append to %s", A.out.c_str());
  if (A.tag.empty()) A.tag = "glm_run";
  const double t0 = now_s();
  Model m(rk, A.mo);
  std::unique_ptr<glw::WeightFile> wf;
  if (A.synthetic) {
    SynthOptions so;
    so.seed = A.seed, so.expert_pools = A.pools, so.s3 = A.s3;
    m.bind_synthetic(so, ring);
  } else {
    if (A.weights.empty()) die("glm_run: --weights PATH or --synthetic");
    char path[1024];
    if (A.weights == "fleet")  // the fleet file's RANK<r>_WEIGHTS
      std::snprintf(path, sizeof path, "%s", comm::fleet_path(comm::fleet(A.fleet).rank[rk].weights, rk).c_str());
    else
      std::snprintf(path, sizeof path, A.weights.c_str(), rk);
    wf = std::make_unique<glw::WeightFile>(path, rk, kRanks);
    wf->load();
    SynthOptions so;
    so.seed = A.seed, so.expert_pools = A.pools, so.s3 = A.s3;
    m.bind(*wf, ring, A.fill_experts ? &so : nullptr);
  }
  size_t fr = 0, tot = 0;
  CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
  record(rk, "tune\t%08x\tprefill pieces %d comm blocks %d", m.options().tune, m.options().prefill_pieces,
         m.options().prefill_comm_blocks);
  record(rk, "start\t%s weights (%d MoE layers' routed experts from the file) %.2f GB\tmax_ctx %d slots %d drafts %d l2pf %d\tbound in %.1f s\tfree %.1f GB",
         A.synthetic ? "synthetic" : "file", m.file_expert_layers(), m.weight_bytes() / 1e9, m.options().max_ctx, m.options().slots,
         m.options().drafts, (int)m.options().l2pf, now_s() - t0, fr / 1e9);
  for (const std::string& mode : A.modes) {
    ring.barrier();
    if (mode == "bench") bench(m, ring);
    else if (mode == "l2pf") l2pf_ab(m, ring);
    else if (mode == "trace") trace(m, ring);
    else if (mode == "ab") ab(m, ring);
    else if (mode == "prefill") bench_prefill(m, ring);
    else if (mode == "pfab") prefill_ab(m, ring);
    else if (mode == "pftrace") prefill_trace(m, ring);
    else if (mode == "check") check_state(m, ring);
    else if (mode == "tf") tf(m, ring);
    else if (mode == "refhash") refhash(m, ring);
    else if (mode == "spec") spec(m, ring);
    else if (mode == "mtpgold") mtpgold(m, ring);
    else die("unknown mode %s", mode.c_str());
  }
  m.drain();
  ring.barrier();
  std::printf("rank %d: %s; %d failures\n", rk, g_fail ? "FAILED" : "passed", g_fail);
  if (g_out) std::fclose(g_out);
  return g_fail ? 1 : 0;
}
