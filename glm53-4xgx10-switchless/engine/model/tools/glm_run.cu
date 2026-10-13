// The model's driver on the ring (docs/engine.md, "The driver"): every rank runs it with the same arguments
// (run4.sh); rank 0 prints and writes.
//   glm_run [options] mode...
// Weights: --weights PATH (a rank file; "%d" in PATH is the rank; "fleet": the fleet file's RANK<r>_WEIGHTS) or --synthetic [--pools N] [--s3] [--seed S].
// Modes:
//   bench     decode steps as graphs at --rows (default 1,2,4,6,8,16) at context --ctx: GPU ms a step, wall ms, host
//             ms a launch; the floor; then one profiled eager step a row count (ms per kernel class, per layer class)
//   l2pf      the paced L2 prefetch off and on, A/B and B/A, at --rows
//   prefill   prefill chunks of --chunks (default 512,1024,2048,4096,8192) rows at position --ctx: tok/s
//   check     the state checks on synthetic weights (chunking, prefill vs decode, batch invariance, determinism)
//   tf        teacher-forced over --tokens FILE (--via prefill|decode): argmax, top-20, lse, next logprob, sampled
//             rows' logits and hidden into --dir (the reference's golden format, docs/reference.md § 7)
//   refhash   prefill --tokens (or --prompt N synthetic ids) and decode --steps greedy tokens (and --temp T): the hash
//             of the tokens and their logit bits, every rank
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

#include "comm/ring.h"
#include "comm/ring_config.h"
#include "model/model.h"
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
  std::vector<int> rows = {1, 2, 4, 6, 8, 16}, chunks = {512, 1024, 2048, 4096, 8192};
  std::vector<int> dump_layers;
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
    if (!dump_layers.count(L)) return;
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
    (void)row_bytes, (void)name;
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
    else if (a == "--splits") A.mo.attn_splits = std::stoi(nx());
    else if (a == "--rows") A.rows = parse_list(nx());
    else if (a == "--chunks") A.chunks = parse_list(nx());
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
  record(rk, "start\t%s weights (%d MoE layers' routed experts from the file) %.2f GB\tmax_ctx %d slots %d drafts %d l2pf %d\tbound in %.1f s\tfree %.1f GB",
         A.synthetic ? "synthetic" : "file", m.file_expert_layers(), m.weight_bytes() / 1e9, m.options().max_ctx, m.options().slots,
         m.options().drafts, (int)m.options().l2pf, now_s() - t0, fr / 1e9);
  for (const std::string& mode : A.modes) {
    ring.barrier();
    if (mode == "bench") bench(m, ring);
    else if (mode == "l2pf") l2pf_ab(m, ring);
    else if (mode == "prefill") bench_prefill(m, ring);
    else if (mode == "check") check_state(m, ring);
    else if (mode == "tf") tf(m, ring);
    else if (mode == "refhash") refhash(m, ring);
    else die("unknown mode %s", mode.c_str());
  }
  m.drain();
  ring.barrier();
  std::printf("rank %d: %s; %d failures\n", rk, g_fail ? "FAILED" : "passed", g_fail);
  if (g_out) std::fclose(g_out);
  return g_fail ? 1 : 0;
}
