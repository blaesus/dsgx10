// dsgb10_rank: the engine process. Every box runs it with the same arguments (engine/server/run4.sh);
// rank 0 decides and sends commands, ranks 1-3 follow (docs/engine.md). In decode every rank drives
// itself from the tokens each step's sampling hands to all of them.
//
//   dsgb10_rank [--fleet FILE] [--rank R] --mode prefill|tf|generate|bench|decode|batch [options]
//     --fleet FILE: the fleet file (fleet/homelab.conf; default: fleet.conf beside this executable, where
//                run4.sh and service.sh install theirs): the ring's devices and addresses, this box's
//                rank by its hostname unless --rank, and this rank's files: --weights, --hashmap,
//                --shard1 and --shard14 (in RANK<r>_ENGRAM) and --disk-dir default to its values. In
//                these paths, given or the fleet's, {rank} is the rank and a leading ~/ this box's home:
//                --weights '~/w/rank{rank}.dsw' names each rank's own file. Also this rank's Engram
//                read batch (RANK<r>_ENGRAM_READ_BATCH, 1 when absent)
//     prefill    the golden sequence in chunks of --chunk rows, every layer compared (--inject: each
//                layer fed the golden input), logits at every position
//     tf         teacher-forced decode: the first --prompt tokens as prefill, then steps of --k rows
//     generate   greedy (or --temp) generation of --gen tokens after --prompt tokens, against golden
//     bench      step times: the golden tokens' first --ctx positions as prefill, then for each k of
//                --ks (e.g. 1,6) --steps teacher-forced steps of k rows, no checks (graph replays
//                unless --no-graphs); the hash of every chosen token and its logit compares runs;
//                --bench-stride S: then as many steps of k unrelated rows, as other sessions' would
//                be: tokens from scattered golden positions, the rows S positions apart (S >= 128:
//                apart in the window too), then the same with row r's router logits rotated 48 r
//                experts (rows that share few experts), and each kind's distinct experts per layer;
//                --bench-groups G,...: then, for each G dividing k (1 < G < k), steps of k / G runs of
//                G consecutive rows (B3's verify rows of k / G sessions: docs/batching.md), each run
//                from its own scattered golden position, the runs S apart; natural, then each run's
//                routing rotated 48 experts further than the run before. A k past 8 (B4: up to 16) runs
//                the grouped kinds only, as a batch of k / G sessions (--slots at least that; session g
//                in slot g, G consecutive positions per step, the sessions S positions apart): one
//                session's step holds at most model::kMaxSessRows rows
//     decode     end-to-end decode speed: --ctx golden positions as prefill, then --steps greedy (or
//                --temp, --top-p, --top-k) tokens through the self-driving decode loop
//     batch      sessions decoding together (docs/batching.md, B2; --big): --batch 1024,16384c,1024
//                prefills each of these golden contexts (c: with CED) in a session slot of its own
//                (--slots; the last slot first), then decodes --steps tokens of each in one batch loop.
//                Each session's tokens hash is its decode-mode run's at that context. --batch-tf
//                [--batch-rows G] (B4): instead, each session's golden continuation teacher-forced, G
//                (1..8) consecutive rows of each per step for --steps steps (a step of at most 16 rows in
//                all); its samples hash is the same whatever the other sessions and G, and tf mode's
//                over the same positions (--big --prompt C --k G --limit C + G * steps)
//     serve      rank 0 serves the front-end (docs/protocol.md) on --socket (default
//                /tmp/dsgb10-engine.sock): sessions with exact prefix reuse (--snap-every positions
//                between window snapshots, thinned with the distance from a session's end unless
//                --snap-all; saved state in a pool reserved at startup: the smallest
//                rank's free memory less --session-margin-gib, at most --session-gib; LRU, the evicted
//                spilled to a file of --disk-gib (default 32; 0: none) in --disk-dir (default the
//                fleet file's RANK<r>_SPILL), prefill in
//                chunks of --prefill-rows (default 8192 with CED, else 4096 as in the other modes),
//                sampled decode. --keep-gib N: a planned stop (SIGTERM) keeps the sessions for the
//                next start (docs/engine.md, "Kept sessions"): those in the file, and those in device
//                memory, the most recently used first, up to N GiB written at the stop (default 0:
//                nothing is kept), none unused for --keep-days (7); --disk-tag T names the files
//                rank<r>.T.spill and .index (rank 0's server.T.index), so that runs of another tag
//                never touch them (the service's: service). --slots N: up to N requests decode
//                together, a row each per step (one alone: with drafts). --slice N: a request that
//                comes while others decode pauses them and is prefilled; while more requests decode
//                than there are slots, they take turns of N tokens (default 64; 0: no turns, and a
//                request that would need the slot of another in flight waits for one to free)
//   generate --loop: the generation runs through the decode loop (same tokens as without)
//   --spec: DSpark drafts (mtp.* loaded) in the decode loop (decode, generate --loop; serve runs them
//     unless --no-spec: they change the speed, never the tokens), at most
//     --spec-drafts per step, verified while their survival >= --spec-thresh (default: lambda * dC from
//     --spec-cost C1,dC,Cdraft ms); --spec-gate ratio: the prefix of drafts with the most expected tokens
//     per ms, by --spec-row-ms (the ms of a step verifying 0..5 drafts, its draft pass included);
//     --spec-gate marginal: draft j while survival[j - 1] >= lambda * its row's ms by --spec-row-ms;
//     --spec-gate thresh: the default above; --mode spectf: the draft module teacher-forced on the golden run (a
//     step per position after the first --prompt - 1, drafting from the golden next token), its leading
//     drafts that equal the golden tokens against engine/dspark's reference; with --big the prompt goes
//     through the prefill path and its seeding of the windows; --draft-fp8: the drafter's head GEMV on an
//     FP8 copy of the vocab head, 0.75 ms less per step (the drafts' arithmetic changes, never the tokens;
//     serve's default, --draft-bf16 opts out); --draft-head4: that head as MXFP4 instead (made from the FP8
//     copy, which goes), --draft-markov8: the Markov head as FP8 (docs/engine.md, "The draft pass's
//     approximations": bundle B's formats, the study's simulated drafts bit for bit; off by default)
//   --tree-probe FILE (spectf; EXPERIMENT, docs/engine.md "Draft trees"): each step also runs the drafter's
//     probe along the golden tokens after the bonus (dspark::ProbeOut), a JSON line per step into FILE, with
//     the target's own top logprobs for the bonus's position (and with --seq-list, the sequence);
//     --tree-sib N: at every Nth step i, the routed experts of the probe's 4 most likely tokens for
//     position i + 2 (an eager step each after step i + 1, as a tree's siblings there would route) and of
//     the golden rows i + 2 .. i + 5 (one eager step, a chain's rows), a JSON line each
//   --head-raw: the vocab head's GEMVs on the bf16 head as loaded, rather than on its exact coding (every
//     tile's exponents as 4-bit offsets, 3/4 of the bytes: model::Model::code_head; the same bits)
//   --no-l2pf: no L2 prefetch in the decode step's windows (model::Options::l2pf, kernels/l2pf.cu: while a
//     collective or the attention leaves DRAM idle, a side stream prefetches the next weights; the same bits)
//   --setenv NAME=VALUE: set in every rank's environment before anything else (experiments' knobs)
//   --profile-from-req N (serve): request N's first command opens nsys's capture range on every rank
//     (--capture-range=cudaProfilerApi) until the server stops; builds with -DDSGB10_NVTX add NVTX ranges
//     per turn, chunk, layer and part (model/nvtx.h)
//   --wave-probe MASK --wave-probe-at POS (with --big): in the prefill chunk at POS, for each layer L of
//     bit L, time layer L's FFN beside a second attention side (model/wave_probe.cu); a second ring on
//     the next port, its proxy on core 6. The run's outputs are not the model's.
//   --ced: --big prefills of at least 128 positions run CED (docs/ced.md): the encoder over every
//   position, the decoder replaying the last 128. serve runs CED by default (turns of at least 128 new
//   positions); --exact makes every turn exact (the bit-level checks)
//   --slots N: N session slots (docs/batching.md, B2): live sessions' states side by side, every step through
//   the attention kernels' slot tables (serve: N requests decode together); --slot S: the golden modes run
//   in slot S (the same bits in any slot)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "comm/ring.h"
#include "common/check.h"
#include "common/keepfile.h"
#include "dspark/dspark.h"
#include "engram/engram.h"
#include "model/model.h"
#include "server/cmd.h"
#include "server/golden.h"
#include "server/serve.h"
#include "weights/loader.h"

namespace {

using namespace srv;

// Bench: the distinct routed experts per layer of a step, from rank 0's route_idx dumps, and how often
// each expert of each layer was routed to (rows: the rows seen per layer).
struct RouteSink : model::Sink {
  double distinct = 0;
  int n = 0;
  std::vector<std::vector<int>> freq = std::vector<std::vector<int>>(model::kLayers, std::vector<int>(model::kExperts));
  std::vector<int> rows = std::vector<int>(model::kLayers);
  void fill(int, const char*, int, int, void*, size_t) override {}
  void tensor(int L, const char* name, int, int R, const void* data, size_t) override {
    if (std::strcmp(name, "route_idx") != 0 || L < 0 || L >= model::kLayers) return;
    const auto* e = static_cast<const int32_t*>(data);
    distinct += (double)std::set<int32_t>(e, e + (size_t)R * model::kTopK).size();
    ++n;
    for (int i = 0; i < R * model::kTopK; ++i)
      if (e[i] >= 0 && e[i] < model::kExperts) ++freq[L][e[i]];
    rows[L] += R;
  }
  void logits(int, int, const float*) override {}
  // the distinct experts per layer that B rows routed independently with these frequencies would reach
  double independent(int B) const {
    double s = 0;
    int nl = 0;
    for (int L = 0; L < model::kLayers; ++L) {
      if (!rows[L]) continue;
      for (int x : freq[L]) s += 1.0 - std::pow(1.0 - (double)x / rows[L], B);
      ++nl;
    }
    return nl ? s / nl : 0;
  }
};

// EXPERIMENT (--tree-sib): each layer's routed experts in one eager step, [R][kTopK] per layer
struct ExpertSink : model::Sink {
  std::vector<std::vector<int32_t>> e = std::vector<std::vector<int32_t>>(model::kLayers);
  void fill(int, const char*, int, int, void*, size_t) override {}
  void tensor(int L, const char* name, int, int R, const void* data, size_t) override {
    if (std::strcmp(name, "route_idx") != 0 || L < 0 || L >= model::kLayers) return;
    const auto* p = static_cast<const int32_t*>(data);
    e[L].assign(p, p + (size_t)R * model::kTopK);
  }
  void logits(int, int, const float*) override {}
};

// EXPERIMENT (--seq-list; docs/engine.md "Prompt-lookup drafts"): sequences of their own in one process, a
// line "DIR PROMPT [N]" each: DIR's tokens.npy, its first PROMPT tokens the prompt; N: generate's most
// tokens, spectf's end (0: the whole sequence)
struct SeqLine {
  std::string dir;
  int prompt = 0, n = 0;
  std::vector<int32_t> toks;
};

std::vector<SeqLine> read_seq_list(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) die("cannot read %s", path.c_str());
  std::vector<SeqLine> v;
  char line[8192], dir[4096];
  while (std::fgets(line, sizeof line, f)) {
    SeqLine s;
    if (line[0] == '#' || std::sscanf(line, "%4095s %d %d", dir, &s.prompt, &s.n) < 2) continue;
    s.dir = dir;
    golden::Golden g(s.dir);
    for (int p = 0; p < g.length(); ++p) s.toks.push_back(g.token(p));
    v.push_back(std::move(s));
  }
  std::fclose(f);
  if (v.empty()) die("%s: no sequences", path.c_str());
  return v;
}

// EXPERIMENT (--props): rows to route, a line "SEQ ANCHOR t1 .. tk" each: the anchor's own token, then t1 .. tk
using Props = std::map<std::pair<int, int>, std::vector<std::vector<int32_t>>>;
Props read_props(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) die("cannot read %s", path.c_str());
  Props m;
  char line[8192];
  while (std::fgets(line, sizeof line, f)) {
    char* q = line;
    char* e;
    const long sq = std::strtol(q, &e, 10);
    if (e == q) continue;
    q = e;
    const long anchor = std::strtol(q, &e, 10);
    if (e == q) continue;
    std::vector<int32_t> t;
    for (q = e;; q = e) {
      const long x = std::strtol(q, &e, 10);
      if (e == q) break;
      t.push_back((int32_t)x);
    }
    m[{(int)sq, (int)anchor}].push_back(t);
  }
  std::fclose(f);
  return m;
}

// Free bytes in dir (made if missing; mode 0700: the sessions' state is what conversations left), this
// rank's file (base.spill) counted as free: it is either reused or made again.
uint64_t spill_free(const std::string& dir, const std::string& base) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  struct stat s;
  if (stat(dir.c_str(), &s) == 0 && s.st_uid == getuid() && (s.st_mode & 077)) chmod(dir.c_str(), 0700);
  struct statvfs v;
  if (statvfs(dir.c_str(), &v) != 0) return 0;
  uint64_t fr = (uint64_t)v.f_bavail * v.f_frsize;
  if (lstat((base + ".spill").c_str(), &s) == 0 && S_ISREG(s.st_mode)) fr += (uint64_t)s.st_blocks * 512;
  return fr;
}

// dir, or a directory above it, holds .git: a repository or a worktree, where no session's state goes
bool in_git_tree(const std::string& dir) {
  std::error_code ec;
  for (std::filesystem::path p = std::filesystem::absolute(dir, ec); !ec && !p.empty(); p = p.parent_path()) {
    if (std::filesystem::exists(p / ".git", ec)) return true;
    if (p == p.parent_path()) break;
  }
  return false;
}

struct Args {
  std::string fleet;  // the fleet file ("": fleet.conf beside this executable)
  int rank = -1;
  std::string mode = "prefill";
  std::string golden = "/data/dsgb10/golden/prose";
  std::string weights, hashmap, shard[2];
  bool inject = false, inject_sub = false, no_engram = false, quiet = false, graphs = true, per_position = false;
  bool profile = false;  // decode: the decode loop is nsys's capture range (--capture-range=cudaProfilerApi)
  bool loop = false, big = false, ced = false, exact = false, graph_sweep = false;
  int ahead = 1, prefill_rows = 0;  // 0: 8192 for serve with CED, else 4096
  int wave_probe = 0, wave_probe_at = -1;
  int profile_at = -1;  // --big: the prefill chunk at this position is nsys's capture range
  uint64_t profile_from_req = 0;  // serve: request N's first command opens it on every rank (0: none)
  int chunk = 8, k = 1, prompt = -1, limit = -1, gen = 96, max_ctx = 65536, threads = 4, layers = model::kLayers, first = 0;
  int ctx = 1024, steps = 64, onset = -1, bench_stride = 0;
  std::vector<int> ks{1}, ctxs;  // ctxs: decode and bench at each of these contexts in turn (else ctx)
  std::vector<int> bench_groups;  // bench: runs of consecutive rows (--bench-groups)
  std::string batch;             // --mode batch: the sessions' contexts
  bool batch_tf = false;         // --mode batch: teacher-forced, batch_rows rows of each session per step
  int batch_rows = 1;
  std::string tree_probe;        // --mode spectf, EXPERIMENT: the probe's records
  int tree_sib = 0;
  std::string seq_list, route_file, props;  // EXPERIMENT (spectf, generate --loop): prompt-lookup drafts
  int route_every = 0, chain_every = 0;
  int step_timeout_s = 120, engram_cache_mib = 4096;
  int engram_read_batch = 1;  // the fleet file's RANK<r>_ENGRAM_READ_BATCH
  float temp = 0.f, top_p = 1.f;
  int top_k = 0;
  uint32_t seed = 1;
  std::string socket = "/tmp/dsgb10-engine.sock";
  double session_gib = 12, session_margin_gib = 2;
  double disk_gib = 32;  // serve: the store's disk tier (0: none)
  std::string disk_dir;  // its directory (default: the fleet's RANK<r>_SPILL)
  std::string disk_tag;  // its files' tag: rank<r>.<tag>.spill (none: rank<r>.spill)
  double keep_gib = 0, keep_days = 7;  // kept sessions: written at a planned stop, at most; their age
  int snap_every = 1024, small_rows = 8;
  int slots = 1, slot = 0;  // session slots (model::Options::slots); the golden modes' slot
  int bgraph_bodies = 0;      // DIAGNOSTICS (spec.cu, the batch graph): switch bodies to capture (0: kMaxRows)
  bool bgraph_noif = false;   // DIAGNOSTICS: the entries' accepts without if nodes
  bool bgraph_check = false;  // DIAGNOSTICS (B3 phase 2a): the one-session draft of every entry after the batch
                              // step's pass over them, and the words that differ, printed by --mode batch
  int slice = 64;  // serve: time-slicing's turns, in tokens (0: none)
  bool snap_all = false;
  size_t ring_cap = 4u << 20;  // the ring's per-rank collective size (prefill's bulk collectives post cap / 4)
  bool spec = false, no_spec = false, draft_fp8 = false, draft_bf16 = false, draft_head4 = false, draft_markov8 = false;
  bool head_raw = false, no_l2pf = false;
  int spec_drafts = 5;
  float spec_thresh = -1.f;
  float spec_cost[3] = {19.4f, 3.6f, 3.3f};
  int spec_gate = model::ops::kGateThresh;
  float spec_row_ms[model::ops::kDrafts + 1] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f};
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) die("missing value for %s", s.c_str());
      return argv[++i];
    };
    if (s == "--rank") a.rank = std::atoi(val().c_str());
    else if (s == "--fleet") a.fleet = comm::fleet_path(val());  // a leading ~/ is $HOME, as in the paths below
    else if (s == "--mode") a.mode = val();
    else if (s == "--golden") a.golden = val();
    else if (s == "--weights") a.weights = val();
    else if (s == "--hashmap") a.hashmap = val();
    else if (s == "--shard1") a.shard[0] = val();
    else if (s == "--shard14") a.shard[1] = val();
    else if (s == "--inject") a.inject = true;
    else if (s == "--inject-sub") a.inject = a.inject_sub = true;
    else if (s == "--no-engram") a.no_engram = true;
    else if (s == "--quiet") a.quiet = true;
    else if (s == "--profile") a.profile = true;
    else if (s == "--profile-at") a.profile_at = std::atoi(val().c_str());
    else if (s == "--profile-from-req") a.profile_from_req = std::strtoull(val().c_str(), nullptr, 10);
    else if (s == "--wave-probe") a.wave_probe = std::atoi(val().c_str());
    else if (s == "--wave-probe-at") a.wave_probe_at = std::atoi(val().c_str());
    else if (s == "--chunk") a.chunk = std::atoi(val().c_str());
    else if (s == "--k") a.k = std::atoi(val().c_str());
    else if (s == "--prompt") a.prompt = std::atoi(val().c_str());
    else if (s == "--limit") a.limit = std::atoi(val().c_str());
    else if (s == "--gen") a.gen = std::atoi(val().c_str());
    else if (s == "--max-ctx") a.max_ctx = std::atoi(val().c_str());
    else if (s == "--threads") a.threads = std::atoi(val().c_str());
    else if (s == "--layers") a.layers = std::atoi(val().c_str());
    else if (s == "--first") a.first = std::atoi(val().c_str());
    else if (s == "--step-timeout") a.step_timeout_s = std::atoi(val().c_str());
    else if (s == "--engram-cache-mib") a.engram_cache_mib = std::atoi(val().c_str());
    else if (s == "--temp") a.temp = std::atof(val().c_str());
    else if (s == "--seed") a.seed = (uint32_t)std::atoll(val().c_str());
    else if (s == "--ctx") a.ctx = std::atoi(val().c_str());
    else if (s == "--steps") a.steps = std::atoi(val().c_str());
    else if (s == "--bench-stride") a.bench_stride = std::atoi(val().c_str());
    else if (s == "--ks" || s == "--ctxs" || s == "--bench-groups") {
      std::vector<int>& list = s == "--ks" ? a.ks : s == "--ctxs" ? a.ctxs : a.bench_groups;
      list.clear();
      const std::string v = val();
      for (const char* q = v.c_str(); *q;) {
        list.push_back(std::atoi(q));
        while (*q && *q != ',') ++q;
        if (*q) ++q;
      }
    }
    else if (s == "--no-graphs") a.graphs = false;
    else if (s == "--onset") a.onset = std::atoi(val().c_str());
    else if (s == "--per-position") a.per_position = true;
    else if (s == "--loop") a.loop = true;
    else if (s == "--ahead") a.ahead = std::atoi(val().c_str());
    else if (s == "--big") a.big = true;
    else if (s == "--graph-sweep") a.graph_sweep = true;
    else if (s == "--ced") a.ced = true;
    else if (s == "--exact") a.exact = true;
    else if (s == "--prefill-rows") a.prefill_rows = std::atoi(val().c_str());
    else if (s == "--top-p") a.top_p = std::atof(val().c_str());
    else if (s == "--top-k") a.top_k = std::atoi(val().c_str());
    else if (s == "--socket") a.socket = val();
    else if (s == "--session-gib") a.session_gib = std::atof(val().c_str());
    else if (s == "--session-margin-gib") a.session_margin_gib = std::atof(val().c_str());
    else if (s == "--disk-gib") a.disk_gib = std::atof(val().c_str());
    else if (s == "--disk-dir") a.disk_dir = val();
    else if (s == "--disk-tag") a.disk_tag = val();
    else if (s == "--keep-gib") a.keep_gib = std::atof(val().c_str());
    else if (s == "--keep-days") a.keep_days = std::atof(val().c_str());
    else if (s == "--snap-every") a.snap_every = std::atoi(val().c_str());
    else if (s == "--slots") a.slots = std::atoi(val().c_str());
    else if (s == "--bgraph-bodies") a.bgraph_bodies = std::atoi(val().c_str());
    else if (s == "--bgraph-noif") a.bgraph_noif = true;
    else if (s == "--bgraph-check") a.bgraph_check = true;
    else if (s == "--slot") a.slot = std::atoi(val().c_str());
    else if (s == "--batch") a.batch = val();
    else if (s == "--batch-tf") a.batch_tf = true;
    else if (s == "--batch-rows") a.batch_rows = std::atoi(val().c_str());
    else if (s == "--slice") a.slice = std::atoi(val().c_str());
    else if (s == "--snap-all") a.snap_all = true;
    else if (s == "--small-rows") a.small_rows = std::atoi(val().c_str());
    else if (s == "--ring-cap") a.ring_cap = std::strtoull(val().c_str(), nullptr, 10);
    else if (s == "--spec") a.spec = true;
    else if (s == "--no-spec") a.no_spec = true;
    else if (s == "--draft-fp8") a.draft_fp8 = true;
    else if (s == "--draft-bf16") a.draft_bf16 = true;
    else if (s == "--draft-head4") a.draft_head4 = true;
    else if (s == "--draft-markov8") a.draft_markov8 = true;
    else if (s == "--tree-probe") a.tree_probe = val();
    else if (s == "--tree-sib") a.tree_sib = std::atoi(val().c_str());
    else if (s == "--seq-list") a.seq_list = val();
    else if (s == "--route-file") a.route_file = val();
    else if (s == "--props") a.props = val();
    else if (s == "--route-every") a.route_every = std::atoi(val().c_str());
    else if (s == "--chain-every") a.chain_every = std::atoi(val().c_str());
    else if (s == "--head-raw") a.head_raw = true;
    else if (s == "--no-l2pf") a.no_l2pf = true;
    else if (s == "--setenv") {
      const std::string v = val();
      const size_t eq = v.find('=');
      if (eq == std::string::npos || eq == 0) die("--setenv takes NAME=VALUE");
      setenv(v.substr(0, eq).c_str(), v.substr(eq + 1).c_str(), 1);
    }
    else if (s == "--spec-drafts") a.spec_drafts = std::atoi(val().c_str());
    else if (s == "--spec-thresh") a.spec_thresh = std::atof(val().c_str());
    else if (s == "--spec-cost") {
      const std::string v = val();
      if (std::sscanf(v.c_str(), "%f,%f,%f", &a.spec_cost[0], &a.spec_cost[1], &a.spec_cost[2]) != 3)
        die("--spec-cost takes C1,dC,Cdraft (ms)");
    }
    else if (s == "--spec-gate") {
      const std::string v = val();
      if (v == "thresh") a.spec_gate = model::ops::kGateThresh;
      else if (v == "ratio") a.spec_gate = model::ops::kGateRatio;
      else if (v == "marginal") a.spec_gate = model::ops::kGateMarginal;
      else die("--spec-gate takes thresh, ratio or marginal");
    }
    else if (s == "--spec-row-ms") {
      const std::string v = val();
      float* r = a.spec_row_ms;
      if (std::sscanf(v.c_str(), "%f,%f,%f,%f,%f,%f", &r[0], &r[1], &r[2], &r[3], &r[4], &r[5]) != 6)
        die("--spec-row-ms takes the ms of a step verifying 0..5 drafts, six of them");
    }
    else die("unknown argument %s", s.c_str());
  }
  if (!a.prefill_rows) a.prefill_rows = a.mode == "serve" && !a.exact ? 8192 : 4096;
  if (a.disk_tag.size() > 32 || a.disk_tag.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
    die("--disk-tag: at most 32 of a-z, 0-9, _ and -");
  if (a.keep_gib < 0 || !(a.keep_days > 0)) die("--keep-gib must be >= 0 and --keep-days > 0");
  if (a.slice < 0) die("--slice must be >= 0");
  // this box's rank and files: the fleet file's, unless given
  const comm::Fleet& fl = comm::fleet(a.fleet);
  if (a.rank < 0) a.rank = comm::rank_from_hostname(a.fleet);
  if (a.rank < 0)
    die("cannot tell this box's rank: its hostname is no RANK<r>_HOSTNAME in the fleet file %s; pass --rank",
        fl.file.c_str());
  if (a.rank > 3) die("--rank %d: the ring's ranks are 0-3", a.rank);
  const comm::RankAddr& me = fl.rank[a.rank];
  const std::string r = std::to_string(a.rank);
  // a path flag's value, else the fleet's: {rank} is this rank, a leading ~/ this box's home (the same
  // arguments then name each rank's own files)
  auto path = [&](std::string& v, const std::string& value, const char* key, const char* flag) {
    v = comm::fleet_path(v.empty() ? value : v, a.rank);
    if (v.empty()) die("no %s, and the fleet file %s has no RANK%d_%s", flag, fl.file.c_str(), a.rank, key);
  };
  path(a.weights, me.weights, "WEIGHTS", "--weights");
  if (!a.no_engram) {
    path(a.hashmap, me.hashmap, "HASHMAP", "--hashmap");
    std::string dir = me.engram;
    while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
    path(a.shard[0], dir.empty() ? "" : dir + "/engram-l1-r" + r + "of4.bin", "ENGRAM", "--shard1");
    path(a.shard[1], dir.empty() ? "" : dir + "/engram-l14-r" + r + "of4.bin", "ENGRAM", "--shard14");
  }
  if (a.mode == "serve") path(a.disk_dir, me.spill, "SPILL", "--disk-dir");
  a.engram_read_batch = me.engram_read_batch;
  // steps of one session hold at most kMaxSessRows rows (config.h); a step of up to kMaxRows takes several sessions
  if (a.chunk < 1 || a.chunk > model::kMaxSessRows || a.k < 1 || a.k > model::kMaxSessRows)
    die("--chunk and --k must be 1..%d (one session's rows in a step)", model::kMaxSessRows);
  if (a.small_rows < 0 || a.small_rows > model::kMaxSessRows)
    die("--small-rows must be 0..%d (a small turn runs as one step of one session)", model::kMaxSessRows);
  if (a.batch_rows < 1 || a.batch_rows > model::kMaxSessRows)
    die("--batch-rows must be 1..%d (one session's rows in a step)", model::kMaxSessRows);
  if (a.mode == "serve") a.spec = !a.no_spec;  // the drafts change the speed, never the tokens
  if (a.mode == "serve") a.draft_fp8 = !a.draft_bf16;  // nor does the drafter's FP8 head
  if (a.draft_fp8 && a.draft_bf16) die("--draft-fp8 or --draft-bf16");
  if (a.draft_head4 && a.draft_bf16) die("--draft-head4 or --draft-bf16");
  if (a.slots < 1 || a.slots > model::kMaxSlots || a.slot < 0 || a.slot >= a.slots)
    die("--slots %d --slot %d: 1 to %d slots, a slot below their number", a.slots, a.slot, model::kMaxSlots);
  if (a.mode == "spectf") a.spec = true;
  if ((!a.tree_probe.empty() || a.tree_sib) && (a.mode != "spectf" || a.tree_probe.empty() || a.tree_sib < 0))
    die("--tree-probe FILE [--tree-sib N]: --mode spectf only");
  if (!a.seq_list.empty() && !(a.mode == "spectf" || (a.mode == "generate" && a.loop && a.big)))
    die("--seq-list FILE: --mode spectf, or generate --loop --big");
  if ((!a.route_file.empty() || !a.props.empty() || a.route_every || a.chain_every) &&
      (a.mode != "spectf" || a.route_file.empty() || a.route_every < 0 || a.route_every > model::kMaxSessRows ||
       a.chain_every < 0))
    die("--route-file FILE [--route-every 1..%d] [--chain-every N] [--props FILE]: --mode spectf only",
        model::kMaxSessRows);
  if (a.spec_drafts < 0 || a.spec_drafts > model::ops::kDrafts) die("--spec-drafts must be 0..5");
  if (a.spec && (a.mode == "decode" || a.mode == "generate") && !a.big)
    die("--spec: the prompt goes through the prefill path (--big), which fills the draft module's windows");
  for (int k : a.ks) {
    if (k < 1 || k > model::kMaxRows) die("--ks: %d is not 1..%d", k, model::kMaxRows);
    if (k <= model::kMaxSessRows) continue;
    // past one session's rows: runs of several sessions only (bench), each a slot of its own
    bool ok = false;
    for (int G : a.bench_groups) ok |= G > 1 && G <= model::kMaxSessRows && G < k && k % G == 0 && k / G <= a.slots;
    if (a.mode != "bench" || a.bench_stride <= 0 || !ok)
      die("--ks %d: more than %d rows take runs of several sessions: --mode bench --bench-stride S --bench-groups G "
          "with G <= %d dividing %d, and --slots at least %d / G", k, model::kMaxSessRows, model::kMaxSessRows, k, k);
  }
  return a;
}

// What a kept session's state depends on besides the store's layout (state.cu checks that): this
// binary's bytes, the weights' and the Engram files' paths, sizes and times, and the options that change
// what a request computes. A start whose id differs from the stop's starts clean (a new build does).
uint64_t run_id(const Args& a) {
  uint64_t h = keepfile::fnv("dsgb10 kept sessions", 20);
  const int fd = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
  if (fd < 0) die("run id: /proc/self/exe: %s", std::strerror(errno));
  std::vector<uint8_t> buf(1 << 20);
  for (ssize_t r; (r = read(fd, buf.data(), buf.size())) > 0;) h = keepfile::fnv(buf.data(), (size_t)r, h);
  close(fd);
  auto file = [&](const std::string& f) {
    struct stat s{};
    const int64_t v[3] = {stat(f.c_str(), &s) == 0 ? (int64_t)s.st_size : -1, (int64_t)s.st_mtim.tv_sec,
                          (int64_t)s.st_mtim.tv_nsec};
    h = keepfile::fnv(f.data(), f.size(), h);
    h = keepfile::fnv(v, sizeof v, h);
  };
  file(a.weights);
  if (!a.no_engram) file(a.hashmap), file(a.shard[0]), file(a.shard[1]);
  const int64_t o[] = {a.max_ctx, a.exact, a.spec, a.draft_fp8, a.prefill_rows, a.small_rows, a.snap_every,
                       a.no_engram, a.layers, a.first, (int64_t)a.ring_cap, a.draft_head4, a.draft_markov8};
  return keepfile::fnv(o, sizeof o, h);
}

// /proc/meminfo's MemAvailable (free pages and the page cache the kernel can reclaim), bytes
size_t mem_available() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  size_t kb = 0;
  while (std::fgets(line, sizeof line, f))
    if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
  std::fclose(f);
  return kb << 10;
}

// This process's resident host memory, bytes
size_t resident() {
  FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0;
  size_t pages = 0, rss = 0;
  if (std::fscanf(f, "%zu %zu", &pages, &rss) != 2) rss = 0;
  std::fclose(f);
  return rss * (size_t)sysconf(_SC_PAGESIZE);
}

void mem_report(int rank, const char* when) {
  size_t dfree = 0, dtotal = 0;
  CUDA_CHECK(cudaMemGetInfo(&dfree, &dtotal));
  std::printf("rank %d: %s; free device memory %.2f GiB, available host memory %.2f GiB, resident %.2f GiB\n", rank,
              when, dfree / 1073741824.0, mem_available() / 1073741824.0, resident() / 1073741824.0);
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const Args a = parse(argc, argv);
  const double t0 = now_s();
  std::printf("rank %d (%s): the fleet file %s; weights %s", a.rank, comm::fleet(a.fleet).rank[a.rank].name.c_str(),
              comm::fleet(a.fleet).file.c_str(), a.weights.c_str());
  if (!a.no_engram) std::printf("; Engram reads in batches of %d", a.engram_read_batch);
  std::printf("\n");
  // 1. CUDA, and every allocation but the weights' arena, before any file is read (docs/findings.md)
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  model::Options mo;
  mo.max_ctx = a.max_ctx;
  mo.n_layers = a.layers;
  mo.first_layer = a.first;
  mo.graphs = a.graphs;
  mo.prefill_rows = a.prefill_rows;
  mo.wave_probe = a.wave_probe, mo.wave_probe_at = a.wave_probe_at;
  mo.draft_fp8 = a.draft_fp8;
  mo.draft_head4 = a.draft_head4, mo.draft_markov8 = a.draft_markov8;
  mo.tree_probe = !a.tree_probe.empty();
  mo.slots = a.slots;
  mo.bgraph_bodies = a.bgraph_bodies, mo.bgraph_noif = a.bgraph_noif, mo.bgraph_check = a.bgraph_check;
  mo.l2pf = !a.no_l2pf;
  model::Model m(a.rank, mo);
  m.use_slot(a.slot);
  // 2. the ring (waits for the other three ranks)
  comm::Options ro;
  ro.fleet = a.fleet;
  ro.rank = a.rank;
  if (a.ring_cap) ro.cap = a.ring_cap;
  comm::Ring ring(ro);
  std::unique_ptr<comm::Ring> ring2;
  if (a.wave_probe) {
    comm::Options r2 = ro;
    r2.port = ro.port + 1, r2.proxy_cpu = 6;
    ring2 = std::make_unique<comm::Ring>(r2);
  }
  std::printf("rank %d: ring up after %.1f s\n", a.rank, now_s() - t0);
  // 3. weights (the DSpark blocks with --spec)
  dsw::WeightFile wf(a.weights, a.rank, comm::kRanks);  // refuses another rank's file
  dsw::LoadOptions lo;
  lo.threads = a.threads;
  lo.select = [&a](const dsw::DeviceTensor& t) {
    if (t.name.rfind("mtp.", 0) == 0) return a.spec;
    if (t.name.rfind("layers.", 0) == 0) {
      const int L = std::atoi(t.name.c_str() + 7);
      return L >= a.first && L < a.layers;
    }
    return true;
  };
  const dsw::LoadStats ls = wf.load(lo);
  std::printf("rank %d: %.2f GiB of weights in %.1f s (%.2f GB/s)\n", a.rank, ls.bytes / 1073741824.0, ls.seconds, ls.gbps);
  m.bind(wf, ring);
  if (ring2) m.bind_ring2(*ring2);
  if (a.spec) m.bind_spec(wf);
  if (!a.head_raw) {  // after bind_spec: DSpark's FP8 head comes from the raw one
    const double t1 = now_s();
    int raw = 0;
    const size_t hb = m.code_head(&raw);
    std::printf("rank %d: the head coded in %.2f s: %.1f MB in place of %.1f (%d tiles raw)\n", a.rank, now_s() - t1,
                hb / 1e6, (double)model::kVocabPart * model::kDim * 2 / 1e6, raw);
  }
  // 4. Engram
  std::unique_ptr<engram::HashMap> emap;
  std::unique_ptr<engram::Store> store;
  if (!a.no_engram) {
    emap = std::make_unique<engram::HashMap>(engram::HashMap::load(a.hashmap));
    engram::StoreConfig sc;
    sc.rank = a.rank;
    sc.shard[0] = a.shard[0];
    sc.shard[1] = a.shard[1];
    sc.cpu = 9;
    sc.cache_bytes = (size_t)a.engram_cache_mib << 20;
    sc.read_batch = a.engram_read_batch;  // the fleet file's, for this rank's SSD
    store = std::make_unique<engram::Store>(sc, *emap);
    m.set_engram(emap.get(), store.get());
  }
  // 5. the prefill path's buffers and gathered weights now, not on the first request (serve, and the
  // modes that prefill through it with --big); every rank's free device memory afterwards goes to
  // rank 0's log
  if (a.mode == "serve" || (a.big && a.mode != "prefill" && a.mode != "bench")) {
    const uint64_t mine = m.prefill_init(ring);
    uint64_t all[4];
    ring.allgather_host(&mine, all, sizeof mine);
    if (a.rank == 0)
      std::printf("free device memory after the prefill buffers, ranks 0-3: %.2f %.2f %.2f %.2f GiB\n",
                  all[0] / 1073741824.0, all[1] / 1073741824.0, all[2] / 1073741824.0, all[3] / 1073741824.0);
  }
  // 6. serve: nothing takes device memory after startup (DESIGN.md's hardening list: CUDA takes only free
  // pages, and the page cache grows into them). The spec step's graphs of every context bucket first, then
  // the sessions' store: each rank offers its free device memory and its available host memory
  // (MemAvailable: free pages and page cache, less the Engram row cache still to grow), each less
  // --session-margin-gib; every rank reserves the smallest offer, at most --session-gib.
  const std::string disk_dir = a.disk_dir;  // serve: from the fleet file unless given
  const std::string disk_tag = a.disk_tag.empty() ? "" : "." + a.disk_tag;
  const std::string disk_base = disk_dir + "/rank" + std::to_string(a.rank) + disk_tag;
  if (a.mode == "serve") {
    if (a.spec) m.spec_graphs();
    const double gib = 1073741824.0;
    // the store's disk tier first, as its staging is pinned host memory: each rank offers its directory's
    // free space less 32 GiB (its own file counted as free), at most --disk-gib; every rank takes the
    // smallest offer (under 1 GiB: none). Kept sessions: when every rank's index of an earlier planned
    // stop checks out (state_kept_check) with the same generation, and its file fits every offer, every
    // rank opens that file as it is; anything else starts clean and removes the kept files, as does
    // --keep-gib 0 (nothing kept, nothing taken back).
    const bool disk = a.disk_gib > 0 && !in_git_tree(disk_dir);
    if (a.disk_gib > 0 && !disk) std::printf("rank %d: no disk tier: %s is in a git tree\n", a.rank, disk_dir.c_str());
    if (disk) {
      const uint64_t fr = spill_free(disk_dir, disk_base), margin = 32ull << 30;
      size_t kbytes = 0;
      std::string why;
      const double tk0 = now_s();
      const uint64_t id = run_id(a);
      const double tk1 = now_s();
      const uint64_t gen = a.keep_gib > 0 ? m.state_kept_check(disk_base, id, a.keep_days * 86400, &kbytes, &why) : 0;
      const double tk2 = now_s();
      if (!why.empty()) std::printf("rank %d: kept sessions: none reused (%s)\n", a.rank, why.c_str());
      const uint64_t mine[3] = {std::min<uint64_t>((uint64_t)(a.disk_gib * gib), fr > margin ? fr - margin : 0), gen,
                                kbytes};
      uint64_t all[4][3];
      ring.allgather_host(mine, all, sizeof mine);
      uint64_t bytes = mine[0];
      bool reuse = gen != 0;
      for (const auto& x : all) bytes = std::min(bytes, x[0]), reuse = reuse && x[1] == gen && x[2] == kbytes;
      reuse = reuse && kbytes <= bytes;
      if (reuse) bytes = kbytes;
      if (bytes >= (1ull << 30)) m.state_disk(disk_base, bytes, 512ull << 20, reuse, id);
      std::printf("session store's disk tier: %.1f GiB in %s (at most %.1f; free space less 32 GiB, ranks 0-3: "
                  "%.0f %.0f %.0f %.0f GiB)\n", m.state_disk_capacity() / gib,
                  m.state_disk_capacity() ? m.state_disk_path().c_str() : "(none)", a.disk_gib, all[0][0] / gib,
                  all[1][0] / gib, all[2][0] / gib, all[3][0] / gib);
      if (a.rank == 0)
        std::printf("kept sessions: %s (generations of ranks 0-3: %016llx %016llx %016llx %016llx; the run id %016llx "
                    "in %.1f ms, this rank's index checked in %.1f ms)\n",
                    reuse && m.state_disk_capacity() ? "the files of the last planned stop are reused" : "none, a clean start",
                    (unsigned long long)all[0][1], (unsigned long long)all[1][1], (unsigned long long)all[2][1],
                    (unsigned long long)all[3][1], (unsigned long long)id, 1e3 * (tk1 - tk0), 1e3 * (tk2 - tk1));
    }
    if (!m.state_disk_capacity()) {  // no disk tier: nothing kept stays
      std::error_code ec;
      std::filesystem::remove(disk_base + ".spill", ec);
      std::filesystem::remove(disk_base + ".index", ec);
      std::filesystem::remove(disk_base + ".index.new", ec);
    }
    size_t dfree = 0, dtotal = 0;
    CUDA_CHECK(cudaMemGetInfo(&dfree, &dtotal));
    const size_t avail = mem_available(), margin = (size_t)(a.session_margin_gib * gib);
    const size_t cache = a.no_engram ? 0 : (size_t)a.engram_cache_mib << 20;
    const uint64_t mine[3] = {dfree, avail,
                              std::min(dfree > margin ? dfree - margin : 0, avail > margin + cache ? avail - margin - cache : 0)};
    uint64_t all[4][3];
    ring.allgather_host(mine, all, sizeof mine);
    uint64_t pool = (uint64_t)(a.session_gib * gib);
    for (const auto& r : all) pool = std::min(pool, r[2]);
    m.state_reserve(pool);
    if (a.rank == 0) {
      std::printf("session store: %.2f GiB on every rank (at most %.1f; free device / available host memory before "
                  "it, ranks 0-3:", m.state_capacity() / gib, a.session_gib);
      for (const auto& r : all) std::printf(" %.2f / %.2f", r[0] / gib, r[1] / gib);
      std::printf(" GiB, less a margin of %.1f and the Engram cache's %.1f)\n", margin / gib, cache / gib);
    }
  }
  ring.barrier();
  pin_self(15);
  mem_report(a.rank, ("ready after " + std::to_string((int)(now_s() - t0)) + " s").c_str());

  Watchdog wd(a.step_timeout_s, ring);
  if (a.rank != 0) {  // follow rank 0
    follow(m, ring, wd, a.rank);
    mem_report(a.rank, "done");
    return 0;
  }

  if (a.mode == "serve") {
    ServeOptions so;
    so.socket = a.socket;
    so.max_ctx = a.max_ctx;
    so.prefill_rows = a.prefill_rows;
    so.small_rows = a.small_rows;
    so.snap_every = a.snap_every;
    so.slice = a.slice;
    so.snap_thin = !a.snap_all;
    so.budget = (size_t)(a.session_gib * 1073741824.0);
    so.keep_index = disk_dir + "/server" + disk_tag + ".index";
    so.keep_bytes = std::min(m.state_disk_capacity(), (size_t)(a.keep_gib * 1073741824.0));  // none without a file
    so.keep_age_s = a.keep_days * 86400;
    so.ced = !a.exact;
    so.spec = a.spec;
    so.spec_drafts = a.spec_drafts;
    so.spec_thresh = a.spec_thresh;
    std::copy(a.spec_cost, a.spec_cost + 3, so.spec_cost);
    so.spec_gate = a.spec_gate;
    so.profile_from_req = a.profile_from_req;
    std::copy(a.spec_row_ms, a.spec_row_ms + model::ops::kDrafts + 1, so.spec_row_ms);
    const int rc = serve(m, ring, wd, so);
    mem_report(a.rank, "stopped");
    return rc;
  }

  // ---- rank 0: drive ----
  int32_t sampled[model::kMaxRows];
  golden::Golden g(a.golden);
  if (a.onset >= 0) g.set_split(a.onset);
  g.set_per_position(a.per_position);
  const int L = a.limit > 0 ? std::min(a.limit, g.length()) : g.length();
  const int prompt = a.prompt > 0 ? a.prompt : a.mode == "prefill" ? L : 39;
  const uint32_t check = model::kFlagDump | model::kFlagLogits | (a.inject ? model::kFlagInject : 0) |
                         (a.inject_sub ? model::kFlagInjectSub : 0);
  double t_pref = 0, t_dec = 0, t_bcast = 0;
  int n_pref = 0, n_dec = 0;
  uint64_t chosen_hash = 1469598103934665603ull;  // FNV-1a over every row's chosen token (graphs vs eager)
  auto run = [&](int pos0, int R, const int32_t* toks, uint32_t flags) {
    Cmd c{};
    c.op = kCmdStep;
    c.rows = R;
    c.pos0 = pos0;
    c.flags = flags;
    for (int r = 0; r < R; ++r) c.tokens[r] = toks[r];
    c.sp.temperature = a.temp;
    c.sp.seed = a.seed;
    const double tb = now_s();
    bcast(ring, c);
    const double ts = now_s();
    t_bcast = ts - tb;
    wd.started = ts;
    m.step(to_ctl(c), sampled, &g);
    wd.started = 0;
    const double t = now_s() - ts;
    for (int r = 0; r < R; ++r) {
      float v = m.last_logit(r);
      uint32_t bits;
      std::memcpy(&bits, &v, 4);
      chosen_hash = (chosen_hash ^ (uint32_t)sampled[r]) * 1099511628211ull;
      chosen_hash = (chosen_hash ^ bits) * 1099511628211ull;
    }
    return t;
  };
  // The decode loop from rank 0: tok_0 at position pos0, n tokens; times[i] when tok_{i+1} arrived
  // (step_times: when each step's first token did).
  std::vector<double> step_times;
  auto run_decode = [&](int pos0, int32_t tok0, int n, std::vector<int32_t>& out, std::vector<double>& times) {
    Cmd c{};
    c.op = kCmdDecode;
    c.pos0 = pos0;
    c.tokens[0] = tok0;
    c.sp.temperature = a.temp;
    c.sp.top_p = a.top_p;
    c.sp.top_k = a.top_k;
    c.sp.seed = a.seed;
    c.max_new = n;
    c.limit = a.max_ctx;
    c.ahead = a.ahead;
    c.spec = a.spec;
    c.spec_drafts = a.spec_drafts;
    c.spec_thresh = a.spec_thresh;
    std::copy(a.spec_cost, a.spec_cost + 3, c.spec_cost);
    c.spec_gate = a.spec_gate;
    std::copy(a.spec_row_ms, a.spec_row_ms + model::ops::kDrafts + 1, c.spec_row_ms);
    if (a.profile) c.flags |= model::kFlagProfile;
    if (!a.seq_list.empty()) c.n_stop = 1, c.stop[0] = 1;  // EXPERIMENT (--seq-list): a sequence ends at EOS
    bcast(ring, c);
    step_times.clear();
    if (a.profile) CUDA_CHECK(cudaProfilerStart());
    const DecodeEnd e = decode_loop(m, c, wd, [&](int, const model::SampleOut& o, int row) {
      times.push_back(now_s());
      if (row == 0) step_times.push_back(times.back());
      out.push_back(o.row[row].token);
      uint32_t bits;
      std::memcpy(&bits, &o.row[row].value, 4);
      chosen_hash = (chosen_hash ^ (uint32_t)o.row[row].token) * 1099511628211ull;
      chosen_hash = (chosen_hash ^ bits) * 1099511628211ull;
    });
    if (a.profile) CUDA_CHECK(cudaProfilerStop());
    return e;
  };
  auto spec_report = [&](const DecodeEnd& e) {
    if (!e.vsteps) return;
    std::printf("  drafts: %d verify steps, %.3f drafts verified and %.3f accepted per step (%.1f%%), "
                "%.3f tokens per step; steps verifying 0..5 drafts: %d %d %d %d %d %d\n",
                e.vsteps, (double)e.verified / e.vsteps, (double)e.accepted / e.vsteps,
                e.verified ? 100.0 * e.accepted / e.verified : 0.0, 1.0 + (double)e.accepted / e.vsteps, e.rows_hist[0],
                e.rows_hist[1], e.rows_hist[2], e.rows_hist[3], e.rows_hist[4], e.rows_hist[5]);
    std::printf("  verify-length gate (%s): mean %.3f, min %.3f, max %.3f over %d steps\n",
                a.spec_gate == model::ops::kGateRatio      ? "ratio; lambda, tokens per ms"
                : a.spec_gate == model::ops::kGateMarginal ? "marginal; lambda, tokens per ms"
                                                           : "lambda * dC",
                e.gate_sum / std::max(1, e.gate_n), e.gate_min, e.gate_max, e.gate_n);
    std::printf("  spec step GPU ms (rows, draft pass) by rows:");
    for (int r = 1; r <= model::kMaxRows; ++r)
      if (e.gpu_n[r]) std::printf(" %d: %.2f (%d)", r, e.gpu_ms[r] / e.gpu_n[r], e.gpu_n[r]);
    std::printf("\n");
  };
  // Prefill positions [p0, p0 + n) through the many-row path, in chunks of --prefill-rows; returns the
  // last row's sample.
  bool ced_now = a.ced;  // big_prefill's CED (--mode batch: the session's)
  model::SampleOut prompt_out{};  // big_prefill's last sample (tf hashes it when it teacher-forces nothing)
  auto big_prefill = [&](int p0, int n, const int32_t* t) {
    Cmd c{};
    c.op = kCmdTokens;
    c.pos0 = std::max(0, p0 - 3);
    c.n = p0 + n - c.pos0;
    bcast(ring, c);
    std::vector<int32_t> v(t - (p0 - c.pos0), t + n);
    bcast_payload(ring, v.data(), v.size() * 4, 0);
    m.set_tokens(c.pos0, c.n, v.data());
    int32_t last = -1;
    for (int q = p0; q < p0 + n;) {
      Cmd k{};
      k.op = kCmdPrefill;
      k.pos0 = q;
      k.rows = std::min(a.prefill_rows, p0 + n - q);
      k.next = std::min(a.prefill_rows, p0 + n - q - k.rows);
      k.flags = (q + k.rows == p0 + n ? kPfSample : 0) | (ced_now && n >= 128 ? kPfCed : 0);
      k.sp.temperature = a.temp;
      k.sp.seed = a.seed;
      k.sp.n_top = model::ops::kTopLogprobs;
      const bool prof = q == a.profile_at;  // rank 0's (nsys wraps it)
      if (prof) CUDA_CHECK(cudaProfilerStart());
      const double tb = now_s();
      bcast(ring, k);
      wd.started = now_s();
      const uint64_t s = run_prefill(m, k);
      if (s != ~0ull) {
        const model::SampleOut& o = m.wait(s);
        last = o.row[0].token;
        prompt_out = o;
        std::printf("first token (position %d%s): %d, logprob %.5f; top logprobs:", q + k.rows - 1,
                    (k.flags & kPfCed) ? ", CED" : "", last, o.row[0].logprob);
        for (int j = 0; j < o.n_top; ++j) std::printf(" %d:%.5f", o.top_id[0][j], o.top_lp[0][j]);
        std::printf("\n");
      }
      m.drain();
      if (prof) CUDA_CHECK(cudaProfilerStop());
      wd.started = 0;
      const double dt = now_s() - tb;
      t_pref += dt;
      ++n_pref;
      if (!a.quiet) std::printf("prefill [%d, %d): %.1f ms, %.0f tok/s\n", q, q + k.rows, 1e3 * dt, k.rows / dt);
      q += k.rows;
    }
    return last;
  };
  std::vector<int32_t> toks(L);
  for (int p = 0; p < L; ++p) toks[p] = g.token(p);
  // decode and bench run at each of --ctxs in turn (ascending), each reached from where the last ended
  const std::vector<int> ctxs = a.ctxs.empty() ? std::vector<int>{a.ctx} : a.ctxs;
  // positions [p, ctx) of the golden tokens: through the prefill path (--big) or 8-row steps; returns
  // the last row's sample
  auto reach = [&](int p, int ctx) {
    t_pref = 0;
    n_pref = 0;
    if (a.big) return big_prefill(p, ctx - p, &toks[p]);
    for (int q = p; q < ctx; q += a.chunk) {
      t_pref += run(q, std::min(a.chunk, ctx - q), &toks[q], 0);
      if (!a.quiet && ++n_pref % 512 == 0) std::printf("prefill %d/%d positions\n", q + a.chunk, ctx);
    }
    n_pref = (ctx - p + a.chunk - 1) / a.chunk;
    return (int32_t)sampled[(ctx - 1 - p) % a.chunk];
  };

  if (a.mode == "decode") {
    for (size_t c = 0, p = 0; c < ctxs.size(); ++c) {
      const int ctx = ctxs[c];
      if (ctx <= (int)p || ctx + a.steps >= L)
        die("decode: context %d after %zu positions, %d steps, %d golden tokens", ctx, p, a.steps, L);
      const int32_t first = reach((int)p, ctx);
      std::printf("prefill: %d positions in %.2f s (%.0f tok/s); first token %d\n", ctx - (int)p, t_pref,
                  (ctx - (int)p) / t_pref, first);
      std::vector<int32_t> out;
      std::vector<double> times;
      const double t0d = now_s();
      const DecodeEnd e = run_decode(ctx, first, a.steps, out, times);
      if (c + 1 == ctxs.size()) bcast(ring, Cmd{});  // stop
      const int n = (int)times.size(), skip = std::min(8, n / 4);  // the first steps capture graphs
      std::printf("decode: context %d, %d tokens (%s, T %.2f top-p %.2f top-k %d%s), %d positions fed, end %u; "
                  "tokens hash %016llx\n",
                  ctx, n, a.graphs ? "graphs" : "eager", a.temp, a.top_p, a.top_k, a.spec ? ", DSpark drafts" : "",
                  e.fed, e.why, (unsigned long long)chosen_hash);
      spec_report(e);
      std::printf("  first token after %.2f ms; then %.3f ms per token (%.2f tok/s) over the last %d\n",
                  1e3 * (times[0] - t0d), 1e3 * (times[n - 1] - times[skip]) / (n - 1 - skip),
                  (n - 1 - skip) / (times[n - 1] - times[skip]), n - 1 - skip);
      {  // per step: with drafts a step emits several tokens at once
        const std::vector<double>& t = a.spec ? step_times : times;
        const int ns = (int)t.size(), sk = std::min(8, ns / 4);
        std::vector<double> d;
        for (int i = sk + 1; i < ns; ++i) d.push_back(1e3 * (t[i] - t[i - 1]));
        std::sort(d.begin(), d.end());
        if (!d.empty())
          std::printf("  per %s (ms): median %.3f min %.3f p90 %.3f max %.3f\n", a.spec ? "step" : "token",
                      d[d.size() / 2], d[0], d[d.size() * 9 / 10], d.back());
      }
      int same = 0;
      while (same < n && ctx + 1 + same < L && out[same] == toks[ctx + 1 + same]) ++same;
      std::printf("  the first %d tokens equal the golden continuation\n", same);
      {
        std::vector<double> late, lau;
        for (size_t i = skip; i + 1 < g_t_launch.size(); ++i) {
          late.push_back(1e3 * (g_t_late[i] - g_t_wait[i]));
          lau.push_back(1e3 * (g_t_launch[i] - g_t_late[i]));
        }
        std::sort(late.begin(), late.end());
        std::sort(lau.begin(), lau.end());
        if (!late.empty())
          std::printf("  host: sample -> lookups submitted %.3f ms (p90 %.3f), -> next launched %.3f ms (p90 %.3f, max %.3f)\n",
                      late[late.size() / 2], late[late.size() * 9 / 10], lau[lau.size() / 2], lau[lau.size() * 9 / 10],
                      lau.back());
      }
      std::printf("generated:");
      for (int t : out) std::printf(" %d", t);
      std::printf("\n");
      // the history now holds what the loop fed (tok_0 .. tok_{fed-1}): the next context goes on from it
      for (int i = 0; i < e.fed; ++i) toks[ctx + i] = i ? out[i - 1] : first;
      p = ctx + e.fed;
    }
    return 0;
  }

  if (a.mode == "batch") {
    struct Sess {
      int ctx = 0, slot = 0;
      bool ced = false;
      int32_t first = 0;
      uint64_t hash = 1469598103934665603ull;  // as decode mode's chosen_hash over its decode
      std::vector<int32_t> out;
    };
    std::vector<Sess> ss;
    for (const char* q = a.batch.c_str(); *q;) {
      Sess s;
      s.ctx = std::atoi(q);
      while (*q && *q != ',') s.ced |= *q++ == 'c';
      if (*q) ++q;
      ss.push_back(s);
    }
    if (!a.big || ss.empty() || (int)ss.size() > a.slots)
      die("batch: %zu sessions (--batch) in %d slots (--slots), through the prefill path (--big)", ss.size(), a.slots);
    const int per = a.batch_tf ? a.steps * a.batch_rows : a.steps;  // positions each session goes on by
    if (a.batch_tf && (int)ss.size() * a.batch_rows > model::kMaxRows)
      die("batch: %zu sessions of %d rows a step: at most %d rows", ss.size(), a.batch_rows, model::kMaxRows);
    if (a.spec && (int)ss.size() > model::kMaxRows / 2)  // B3: a draft each at least
      die("batch: %zu sessions with drafts (--spec): at most %d (kMaxRows / 2)", ss.size(), model::kMaxRows / 2);
    for (size_t k = 0; k < ss.size(); ++k) {  // each session's prompt in its slot
      Sess& s = ss[k];
      s.slot = a.slots - 1 - (int)k;
      if (s.ctx < 128 || s.ctx + per >= L) die("batch: context %d and %d positions, %d golden tokens", s.ctx, per, L);
      Cmd c{};
      c.op = kCmdState;
      c.slot = s.slot;
      bcast(ring, c);
      m.use_slot(s.slot);
      ced_now = s.ced;
      t_pref = 0, n_pref = 0;
      s.first = big_prefill(0, s.ctx, toks.data());
      std::printf("batch: session %zu in slot %d: context %d%s, prefill %.2f s, first token %d\n", k, s.slot, s.ctx,
                  s.ced ? " (CED)" : "", t_pref, s.first);
    }
    if (a.batch_tf) {
      // B4's check (docs/batching.md): every session's golden continuation teacher-forced, --batch-rows rows of
      // each per step (kCmdBatchTf). Each row computes what it computes alone, so a session's samples hash is
      // the same whatever the other sessions and the rows per step, and tf mode's over the same positions
      // (--big --prompt C --k G --limit C + G * steps: the same (token, logit) folds in the same order).
      std::vector<BatchEntry> e(ss.size());
      std::vector<std::vector<int32_t>> t(ss.size());
      for (size_t k = 0; k < ss.size(); ++k) {
        BatchEntry& x = e[k];
        x.slot = ss[k].slot, x.pos0 = ss[k].ctx, x.max_new = per, x.limit = a.max_ctx;
        x.sp.temperature = a.temp, x.sp.top_p = a.top_p, x.sp.top_k = a.top_k, x.sp.seed = a.seed;
        t[k].assign(toks.begin() + ss[k].ctx, toks.begin() + ss[k].ctx + per);
      }
      Cmd c{};
      c.op = kCmdBatchTf;
      c.n = (int)e.size();
      c.rows = a.batch_rows;
      bcast(ring, c);
      bcast_tf(ring, c, e, t, 0);
      const std::vector<float> ms = batch_tf_loop(m, c, e, t, wd, [&](int k, const model::SampleOut& o, int row) {
        Sess& s = ss[k];
        s.out.push_back(o.row[row].token);  // the sample at its position: the golden token after it when equal
        uint32_t bits;
        std::memcpy(&bits, &o.row[row].value, 4);
        s.hash = (s.hash ^ (uint32_t)o.row[row].token) * 1099511628211ull;
        s.hash = (s.hash ^ bits) * 1099511628211ull;
      });
      bcast(ring, Cmd{});  // stop
      for (size_t k = 0; k < ss.size(); ++k) {
        const Sess& s = ss[k];
        int same = 0;
        for (size_t i = 0; i < s.out.size(); ++i) same += s.out[i] == toks[s.ctx + 1 + i];
        std::printf("batch-tf: session %zu (slot %d, context %d%s): %zu rows in steps of %d, samples hash %016llx; %d of "
                    "them equal the golden next token\n",
                    k, s.slot, s.ctx, s.ced ? ", CED" : "", s.out.size(), a.batch_rows, (unsigned long long)s.hash, same);
      }
      const int skip = std::min(3, (int)ms.size() / 2);  // the first steps run eagerly or capture
      std::vector<double> d(ms.begin() + skip, ms.end());
      std::sort(d.begin(), d.end());
      if (!d.empty())
        std::printf("batch-tf: %zu steps of %zu sessions, %d rows each: gpu (ms) median %.3f min %.3f p90 %.3f max %.3f\n",
                    ms.size(), ss.size(), (int)ss.size() * a.batch_rows, d[d.size() / 2], d[0], d[d.size() * 9 / 10],
                    d.back());
      return 0;
    }
    std::vector<BatchEntry> e(ss.size());
    for (size_t k = 0; k < ss.size(); ++k) {
      BatchEntry& x = e[k];
      x.slot = ss[k].slot, x.pos0 = ss[k].ctx, x.token = ss[k].first;
      x.max_new = a.steps, x.limit = a.max_ctx;
      x.sp.temperature = a.temp, x.sp.top_p = a.top_p, x.sp.top_k = a.top_k, x.sp.seed = a.seed;
    }
    Cmd c{};
    c.op = a.spec ? kCmdSpecBatch : kCmdBatch;  // B3 (--spec): drafts across the sessions, every one to its end
    c.n = (int)e.size();
    if (a.spec) {
      c.spec = 1;
      c.spec_drafts = a.spec_drafts;
      c.spec_thresh = a.spec_thresh;
      std::copy(a.spec_cost, a.spec_cost + 3, c.spec_cost);
      c.spec_gate = a.spec_gate;
      std::copy(a.spec_row_ms, a.spec_row_ms + model::ops::kDrafts + 1, c.spec_row_ms);
    }
    bcast(ring, c);
    bcast_payload(ring, e.data(), e.size() * sizeof(BatchEntry), 0);
    const OnBatch cb = [&](int k, const model::SampleOut& o, int row) {
      Sess& s = ss[k];
      s.out.push_back(o.row[row].token);
      uint32_t bits;
      std::memcpy(&bits, &o.row[row].value, 4);
      s.hash = (s.hash ^ (uint32_t)o.row[row].token) * 1099511628211ull;
      s.hash = (s.hash ^ bits) * 1099511628211ull;
    };
    std::vector<BatchEnd> ends(ss.size());
    std::vector<DecodeEnd> dends;
    SpecBatchStats st;
    if (a.spec) {
      dends = spec_batch_loop(m, c, e, wd, cb, &st);
      for (size_t k = 0; k < ss.size(); ++k)
        ends[k].sampled = dends[k].sampled, ends[k].fed = dends[k].fed, ends[k].why = dends[k].why;
    } else {
      ends = batch_loop(m, c, e, wd, cb);
    }
    bcast(ring, Cmd{});  // stop
    for (size_t k = 0; k < ss.size(); ++k) {
      const Sess& s = ss[k];
      int same = 0;
      while (same < (int)s.out.size() && s.ctx + 1 + same < L && s.out[same] == toks[s.ctx + 1 + same]) ++same;
      std::printf("batch: session %zu (slot %d, context %d%s): %zu tokens, %d positions fed, end %u; tokens hash "
                  "%016llx; the first %d tokens equal the golden continuation\n",
                  k, s.slot, s.ctx, s.ced ? ", CED" : "", s.out.size(), ends[k].fed, ends[k].why,
                  (unsigned long long)s.hash, same);
      if (a.spec) spec_report(dends[k]);
    }
    // the steps' times, by when each one's samples came (the first ones run eagerly and capture graphs)
    const std::vector<double>& t = g_t_wait;
    const int ns = (int)t.size(), skip = std::min(8, ns / 4);
    if (ns - 1 - skip > 0) {
      const double ms = 1e3 * (t[ns - 1] - t[skip]) / (ns - 1 - skip);
      std::vector<double> d;
      for (int i = skip + 1; i < ns; ++i) d.push_back(1e3 * (t[i] - t[i - 1]));
      std::sort(d.begin(), d.end());
      if (!a.spec) {
        std::printf("batch: %d steps of %zu rows: %.3f ms per step (median %.3f, p90 %.3f, max %.3f), %.1f tok/s in "
                    "all, %.1f per session\n",
                    ns, ss.size(), ms, d[d.size() / 2], d[d.size() * 9 / 10], d.back(), 1e3 * ss.size() / ms, 1e3 / ms);
      } else {
        const double tps = st.steps ? (double)st.emitted / st.steps : 0.0;  // tokens per step, every session's
        std::printf("batch: %d steps of %.2f rows (%zu sessions with drafts): %.3f ms per step (median %.3f, p90 %.3f, "
                    "max %.3f), %.2f tokens per step, %.1f tok/s in all, %.1f per session; spec step GPU ms by rows:",
                    ns, st.steps ? (double)st.rows / st.steps : 0.0, ss.size(), ms, d[d.size() / 2],
                    d[d.size() * 9 / 10], d.back(), tps, 1e3 * tps / ms, 1e3 * tps / ms / (double)ss.size());
        for (int r = 1; r <= model::kMaxRows; ++r)
          if (st.gpu_n[r]) std::printf(" %d: %.2f (%d)", r, st.gpu_ms[r] / st.gpu_n[r], st.gpu_n[r]);
        std::printf("\n");
      }
    }
    if (a.spec && a.bgraph_check)  // B3 phase 2a: the passes' drafts against each entry's one-session draft
      std::printf("batch: draft check: %u words of the entries' drafts and confidence differ from their one-session "
                  "drafts (0: the pass over the entries is bit for bit each entry's own)\n",
                  m.spec_batch_mismatches());
    return 0;
  }

  if (a.mode == "spectf") {
    // engine/dspark's reference run: step i (i >= the prompt's last position) adds position i to the
    // windows and drafts from the golden token at i + 1; d_j is for position i + 1 + j.
    // EXPERIMENT (--seq-list): the list's sequences one after another, each from position 0
    std::vector<SeqLine> seqs;
    if (a.seq_list.empty()) seqs.push_back(SeqLine{a.golden, prompt, L, toks});
    else seqs = read_seq_list(a.seq_list);
    const int32_t* tk = nullptr;  // the sequence's tokens
    auto rows = [&](int pos0, int R, int force, const int32_t* probe = nullptr) {
      Cmd c{};
      c.op = kCmdRows;
      c.pos0 = pos0;
      c.rows = R;
      std::copy(tk + pos0, tk + pos0 + R, c.tokens);
      c.spec_force = force;
      if (probe) std::copy(probe, probe + model::ops::kDrafts, c.spec_probe);
      if (probe) c.sp.n_top = model::ops::kTopLogprobs;  // the target's own top logprobs too (the draft gate study)
      bcast(ring, c);
      wd.started = now_s();
      const model::SampleOut& o = m.wait(run_rows(m, c));
      wd.started = 0;
      return o;
    };
    // EXPERIMENT (--tree-probe, --tree-sib): the probe's records, and the routed experts of eager steps
    std::FILE* tp = nullptr;
    if (!a.tree_probe.empty() && !(tp = std::fopen(a.tree_probe.c_str(), "w")))
      die("spectf: cannot write %s", a.tree_probe.c_str());
    auto ints = [](const int32_t* v, int k) {
      std::string s = "[";
      for (int j = 0; j < k; ++j) s += (j ? "," : "") + std::to_string(v[j]);
      return s + "]";
    };
    auto flts = [](const float* v, int k) {
      std::string s = "[";
      char b[32];
      for (int j = 0; j < k; ++j) std::snprintf(b, sizeof b, "%s%.6g", j ? "," : "", v[j]), s += b;
      return s + "]";
    };
    auto route = [&](int pos0, int R, const int32_t* t, ExpertSink& es) {  // an eager step's routed experts
      Cmd c{};
      c.op = kCmdStep;
      c.rows = R;
      c.pos0 = pos0;
      c.flags = model::kFlagDump;
      std::copy(t, t + R, c.tokens);
      bcast(ring, c);
      wd.started = now_s();
      m.step(to_ctl(c), sampled, &es);
      wd.started = 0;
    };
    auto experts = [&](int pos0, int R, const int32_t* t) {  // the same, per layer, as JSON
      ExpertSink es;
      route(pos0, R, t, es);
      std::string s = "[";
      for (int l = 0; l < model::kLayers; ++l) s += (l ? "," : "") + ints(es.e[l].data(), (int)es.e[l].size());
      return s + "]";
    };
    // EXPERIMENT (--route-file; docs/engine.md "Prompt-lookup drafts"): the routed experts of eager steps on
    // the way, a record each: int32 {sequence, kind, anchor, pos0, R}, the R rows' tokens, then int16 experts
    // [kLayers][R][kTopK] (-1: a layer without). Kinds: 0 the sequence's own tokens from the anchor on
    // (--route-every R: R rows at every R-th anchor), 1 the anchor's token and the step's drafts (--chain-every
    // N: at every N-th anchor), 2 the anchor's token and a --props line's tokens. Rows past 8 go in further
    // steps behind the first ones (a row computes the same in a step of any size).
    std::FILE* rf = nullptr;
    if (!a.route_file.empty() && !(rf = std::fopen(a.route_file.c_str(), "wb")))
      die("spectf: cannot write %s", a.route_file.c_str());
    const Props props = a.props.empty() ? Props{} : read_props(a.props);
    double t_route = 0;
    int n_route = 0;
    auto route_rec = [&](int sq, int kind, int anchor, int R, const int32_t* t) {
      // rows at [anchor, anchor + R) read the positions before the anchor; what they write past the history
      // must leave the carries' slot of anchor - 1 alone, for the next spectf step (16 slots: R <= 15)
      if (R < 1 || R > 15) die("spectf: %d rows to route at %d (1..15)", R, anchor);
      for (int q = 0; q < R; q += model::kMaxSessRows) {  // steps of one session: at most kMaxSessRows rows
        const int nr = std::min(model::kMaxSessRows, R - q);
        ExpertSink es;
        const double t0 = now_s();
        route(anchor + q, nr, t + q, es);
        t_route += now_s() - t0, ++n_route;
        const int32_t h[5] = {sq, kind, anchor, anchor + q, nr};
        std::fwrite(h, 4, 5, rf);
        std::fwrite(t + q, 4, nr, rf);
        std::vector<int16_t> e((size_t)model::kLayers * nr * model::kTopK, -1);
        for (int l = 0; l < model::kLayers; ++l)
          if ((int)es.e[l].size() == nr * model::kTopK)
            for (int j = 0; j < nr * model::kTopK; ++j) e[(size_t)l * nr * model::kTopK + j] = (int16_t)es.e[l][j];
        std::fwrite(e.data(), 2, e.size(), rf);
      }
    };
    int sib_at = -1;  // the step whose probe's 4 most likely tokens for position sib_at + 2 are routed next
    int32_t sib[4] = {};
    int hist[model::ops::kDrafts + 1] = {}, n = 0, first = -1;
    double conf[model::ops::kDrafts] = {}, rate[model::ops::kDrafts] = {};
    int runs[model::ops::kDrafts] = {};
    for (size_t sq = 0; sq < seqs.size(); ++sq) {
      tk = seqs[sq].toks.data();
      const int SL = std::min(seqs[sq].n > 0 ? seqs[sq].n : INT32_MAX, (int)seqs[sq].toks.size());
      first = seqs[sq].prompt - 1;
      if (first < 1 || first + 2 > SL) die("spectf: --prompt %d does not fit the %d golden tokens", first + 1, SL);
      char pre[32] = "";
      if (!a.seq_list.empty()) std::snprintf(pre, sizeof pre, "seq %zu ", sq);
      if (a.big)  // the prompt through the prefill path: its seeding of the windows
        big_prefill(0, first, tk);
      else  // spec steps of one session: at most kMaxSessRows rows each
        for (int p = 0; p < first; p += model::kMaxSessRows) rows(p, std::min(model::kMaxSessRows, first - p), -1);
      sib_at = -1;
      for (int i = first; i + 1 < SL; ++i, ++n) {
        int32_t path[model::ops::kDrafts];
        for (int j = 0; j < model::ops::kDrafts; ++j) path[j] = i + 2 + j < SL ? tk[i + 2 + j] : -1;
        const model::SampleOut& o = rows(i, 1, tk[i + 1], tp ? path : nullptr);
        int32_t drafts[model::ops::kDrafts];  // o's slot is reused 4 steps on
        std::copy(o.drafts, o.drafts + model::ops::kDrafts, drafts);
        if (tp) {
          dspark::ProbeOut p;
          static_assert(sizeof p == sizeof o.probe, "ProbeOut");
          std::memcpy(&p, o.probe, sizeof p);
          std::string id = "[", pr = "[";
          for (int j = 0; j < model::ops::kDrafts; ++j) {
            id += (j ? "," : "") + ints(p.id[j], dspark::kProbeK);
            pr += (j ? "," : "") + flts(p.p[j], dspark::kProbeK);
          }
          // with --seq-list, the sequence; tid / tlp: the target's own most likely ids for position i + 1 and their
          // logprobs (T = 1, before top-k / top-p), which the draft gate study turns into P(a sample takes d_j)
          const std::string sqf = a.seq_list.empty() ? "" : "\"seq\":" + std::to_string(sq) + ",";
          std::fprintf(tp, "{%s\"i\":%d,\"bonus\":%d,\"sample\":%d,\"d\":%s,\"s\":%s,\"path\":%s,\"id\":%s],"
                       "\"p\":%s],\"pp\":%s,\"rank\":%s,\"conf\":%s,\"lse\":%s,\"tid\":%s,\"tlp\":%s}\n",
                       sqf.c_str(), i, tk[i + 1], o.row[0].token, ints(o.drafts, 5).c_str(), flts(o.survival, 5).c_str(),
                       ints(path, 5).c_str(), id.c_str(), pr.c_str(), flts(p.p_path, 5).c_str(), ints(p.rank, 5).c_str(),
                       flts(p.conf, 5).c_str(), flts(p.lse, 5).c_str(), ints(o.top_id[0], o.n_top).c_str(),
                       flts(o.top_lp[0], o.n_top).c_str());
        }
        int acc = 0;
        while (acc < model::ops::kDrafts && i + 2 + acc < SL && o.drafts[acc] == tk[i + 2 + acc]) ++acc;
        std::printf("%sspectf %d: drafts %d %d %d %d %d, survival %.4f %.4f %.4f %.4f %.4f, %d equal\n", pre, i,
                    o.drafts[0], o.drafts[1], o.drafts[2], o.drafts[3], o.drafts[4], o.survival[0], o.survival[1],
                    o.survival[2], o.survival[3], o.survival[4], acc);
        ++hist[acc];
        for (int j = 0; j < model::ops::kDrafts && j <= acc && i + 2 + j < SL; ++j) {  // P(d_j+1 | d_1..d_j accepted)
          const double prev = j ? o.survival[j - 1] : 1.0;
          conf[j] += prev > 0 ? o.survival[j] / prev : 0.0;
          rate[j] += j < acc;
          ++runs[j];
        }
        if (tp && sib_at == i - 1 && i + 4 < SL) {  // position i + 1 has its KV of the golden token: route there
          std::string s = "[";
          for (int k = 0; k < 4; ++k) s += (k ? "," : "") + experts(i + 1, 1, &sib[k]);
          const std::string ch = experts(i + 1, 4, &tk[i + 1]);
          std::fprintf(tp, "{\"sib_at\":%d,\"pos\":%d,\"tokens\":%s,\"golden\":%s,\"sib\":%s],\"chain\":%s}\n", sib_at,
                       i + 1, ints(sib, 4).c_str(), ints(&tk[i + 1], 4).c_str(), s.c_str(), ch.c_str());
        }
        sib_at = -1;
        if (tp && a.tree_sib > 0 && (i - first) % a.tree_sib == 0) {
          sib_at = i;
          dspark::ProbeOut p;
          std::memcpy(&p, o.probe, sizeof p);
          std::copy(p.id[0], p.id[0] + 4, sib);
        }
        if (rf) {  // the next step's anchor i + 1: rows from there (the history before it is the sequence's)
          const int A = i + 1;
          if (a.route_every > 0 && (i - first) % a.route_every == 0)
            route_rec((int)sq, 0, A, std::min(a.route_every, SL - A), tk + A);
          if (a.chain_every > 0 && (i - first) % a.chain_every == 0) {
            int32_t r[1 + model::ops::kDrafts] = {tk[A]};
            std::copy(drafts, drafts + model::ops::kDrafts, r + 1);
            route_rec((int)sq, 1, A, 1 + model::ops::kDrafts, r);
          }
          const auto it = props.find({(int)sq, A});
          if (it != props.end())
            for (const std::vector<int32_t>& p : it->second) {
              std::vector<int32_t> r{tk[A]};
              r.insert(r.end(), p.begin(), p.end());
              route_rec((int)sq, 2, A, (int)r.size(), r.data());
            }
        }
      }
      if (!a.seq_list.empty()) std::printf("seq %zu: steps from position %d to %d\n", sq, first, SL - 2);
    }
    bcast(ring, Cmd{});  // stop
    if (tp) std::fclose(tp);
    if (rf) {
      std::fclose(rf);
      std::printf("spectf: %d eager steps routed, %.1f ms each\n", n_route, n_route ? 1e3 * t_route / n_route : 0.0);
    }
    std::printf("spectf: %d steps from position %d; leading drafts equal to the golden tokens 0..5: %d %d %d %d %d %d "
                "(%.3f per step; engine/dspark's reference on prose: 47 31 12 3 2 1, 0.80)\n",
                n, first, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5],
                (double)(hist[1] + 2 * hist[2] + 3 * hist[3] + 4 * hist[4] + 5 * hist[5]) / n);
    for (int j = 0; j < 3; ++j)
      if (runs[j])
        std::printf("  d%d: mean confidence %.3f, accepted %.3f, of %d steps%s\n", j + 1, conf[j] / runs[j],
                    rate[j] / runs[j], runs[j], j == 1 ? " with d1 accepted" : j ? " with d1..d2 accepted" : "");
    return 0;
  }

  if (a.mode == "bench") {
    int need = ctxs.back();
    for (int k : a.ks) {
      const bool multi = k > model::kMaxSessRows;  // runs of several sessions only (B4)
      if (!multi) need += k * (a.bench_stride > 0 ? 3 * a.steps + 18 : a.steps);
      for (int G : a.bench_groups)
        if (a.bench_stride > 0 && G > 1 && G < k && k % G == 0 && (!multi || (G <= model::kMaxSessRows && k / G <= a.slots)))
          need += k * (2 * a.steps + 12);
    }
    // the rotated kinds: row r's (or run r's) router logits rotated kRouteShift experts further than the one
    // before (model.h set_route_shift), kExperts / 8 as measured since 2026-10-12, whatever kMaxRows
    constexpr int kRouteShift = model::kExperts / model::kMaxSessRows;
    if (ctxs[0] < 1 || a.steps < 4 || need > L)
      die("bench: context %d plus %d steps per k do not fit the %d golden tokens", ctxs.back(), a.steps, L);
    auto stat = [](std::vector<double> v) {  // the first 3 steps run eagerly or capture
      v.erase(v.begin(), v.begin() + 3);
      std::sort(v.begin(), v.end());
      double s = 0;
      for (double x : v) s += x;
      char b[128];
      std::snprintf(b, sizeof b, "median %.3f mean %.3f min %.3f p90 %.3f", v[v.size() / 2], s / v.size(), v[0],
                    v[v.size() * 9 / 10]);
      return std::string(b);
    };
    for (size_t c = 0, p = 0; c < ctxs.size(); ++c) {
      const int ctx = ctxs[c];
      if (ctx < (int)p) die("bench: context %d after %zu positions (--ctxs ascend by the steps at least)", ctx, p);
      n_pref = 0;
      if (ctx > (int)p) reach((int)p, ctx);
      std::vector<std::vector<double>> host(a.ks.size()), gpu(a.ks.size()), bc(a.ks.size()), ugpu(a.ks.size()),
          xgpu(a.ks.size());
      RouteSink route_all;  // every consecutive row routed in this context's eager steps
      p = ctx;
      for (size_t j = 0; j < a.ks.size(); ++j) {
        if (a.ks[j] > model::kMaxSessRows) {
          // B4: a step of 9..16 rows holds runs of several sessions only (one session's at most kMaxSessRows: the
          // one-session kinds below are skipped), through Model::launch_batch (kCmdBatchTf) with a session slot per
          // run: session g in slot g, G consecutive positions per step from base - (n - 1 - g) * S (S = --bench-stride:
          // the sessions as far apart as a one-session step's runs), its tokens from its own scattered golden
          // position (run_tok, as the runs below); natural, then each session's routing rotated kRouteShift experts
          // further than the one before (rank 0's, as the kinds below); then a few eager steps of each kind for
          // their distinct experts. The slots past the context's hold no state (their rows attend to empty windows
          // and caches: the same work), and every slot's Engram history is the golden tokens (set_tokens), so that
          // each session's rows hash their n-grams from it.
          const int k = a.ks[j];
          std::printf("  k %d: past one session's %d rows; runs of several sessions only\n", k, model::kMaxSessRows);
          const int base = (int)p;
          for (int G : a.bench_groups) {
            if (G <= 1 || G > model::kMaxSessRows || G >= k || k % G || k / G > a.slots) continue;
            const int n = k / G;
            if (base < (n - 1) * a.bench_stride)
              die("bench: k %d in %d runs %d apart needs a context of at least %d", k, n, a.bench_stride, (n - 1) * a.bench_stride);
            for (int s = 0; s < n; ++s) {  // the sessions' Engram histories: the golden tokens, in every slot
              Cmd c{};
              c.op = kCmdTokens;
              c.slot = s;
              c.pos0 = 0;
              c.n = L;
              bcast(ring, c);
              bcast_payload(ring, toks.data(), toks.size() * 4, 0);
              m.use_slot(s);
              m.set_tokens(0, L, toks.data());
            }
            {
              Cmd c{};
              c.op = kCmdState;
              c.slot = a.slot;
              bcast(ring, c);
              m.use_slot(a.slot);
            }
            auto run_tok = [&](size_t q, int r) {
              const uint64_t g = (uint64_t)(r / G);
              return toks[(size_t)(((q + g) * 2654435761ull + g * 40503ull) % (uint64_t)(L - G)) + r % G];
            };
            // one step: the sessions' step i (their positions go on by G each), tokens drawn at q; eager with its
            // routed experts into rs when given; its GPU ms
            auto tf_step = [&](size_t q, int i, RouteSink* rs) {
              Cmd c{};
              c.op = kCmdBatchTf;
              c.n = n;
              c.rows = G;
              c.flags = rs ? model::kFlagDump : 0;
              std::vector<BatchEntry> e(n);
              std::vector<std::vector<int32_t>> t(n);
              for (int g = 0; g < n; ++g) {
                e[g].slot = g;
                e[g].pos0 = base - (n - 1 - g) * a.bench_stride + i * G;
                e[g].max_new = G;
                e[g].limit = a.max_ctx;
                e[g].sp.temperature = a.temp, e[g].sp.seed = a.seed;
                for (int r = 0; r < G; ++r) t[g].push_back(run_tok(q, g * G + r));
              }
              bcast(ring, c);
              bcast_tf(ring, c, e, t, 0);
              const std::vector<float> ms =
                  batch_tf_loop(m, c, e, t, wd, [&](int, const model::SampleOut& o, int row) {
                    uint32_t bits;
                    std::memcpy(&bits, &o.row[row].value, 4);
                    chosen_hash = (chosen_hash ^ (uint32_t)o.row[row].token) * 1099511628211ull;
                    chosen_hash = (chosen_hash ^ bits) * 1099511628211ull;
                  }, rs);
              return ms.empty() ? 0.0 : (double)ms[0];
            };
            std::vector<double> gg[2];
            RouteSink gs[2];
            int i = 0;
            for (int v = 0; v < 2; ++v) {
              m.set_route_shift(v ? kRouteShift : 0, G);
              for (int s = 0; s < a.steps; ++s, ++i, p += k) gg[v].push_back(tf_step(p, i, nullptr));
              for (int s = 0; s < 6; ++s, ++i, p += k) tf_step(p, i, a.rank == 0 ? &gs[v] : nullptr);
            }
            m.set_route_shift(0);
            std::printf("  k %d in %d runs of %d rows %d apart (a session a slot, B4), gpu (ms): natural %s, %.1f experts per "
                        "layer; rotated %s, %.1f\n",
                        k, n, G, a.bench_stride, stat(gg[0]).c_str(), gs[0].distinct / std::max(1, gs[0].n),
                        stat(gg[1]).c_str(), gs[1].distinct / std::max(1, gs[1].n));
          }
          continue;
        }
        for (int i = 0; i < a.steps; ++i, p += a.ks[j]) {
          // nsys --capture-range=cudaProfilerApi: the first context's replays from the first k's on (tools:
          // kprof drops each k's first 3 steps, which run eagerly or capture)
          const bool start = c == 0 && j == 0 && i == 3;
          if (start) CUDA_CHECK(cudaProfilerStart());
          host[j].push_back(1e3 * run((int)p, a.ks[j], &toks[p], start ? model::kFlagProfile : 0));
          gpu[j].push_back(m.last_gpu_ms());
          bc[j].push_back(1e3 * t_bcast);
        }
        if (a.bench_stride <= 0) continue;
        // the same row count, unrelated rows: golden tokens from scattered positions, the rows
        // bench_stride positions apart (every rank draws the same); then the same with each row's router
        // logits rotated kRouteShift experts further than the last row's, so the rows share few experts
        auto scattered = [&](size_t q, int r) {
          return toks[(size_t)(((uint64_t)(q + r) * 2654435761ull + (uint64_t)r * 40503ull) % (uint64_t)L)];
        };
        m.set_pos_stride(a.bench_stride);
        for (int v = 1; v <= 2; ++v) {
          m.set_route_shift(v == 2 ? kRouteShift : 0);
          for (int i = 0; i < a.steps; ++i, p += a.ks[j]) {
            int32_t u[model::kMaxRows];
            for (int r = 0; r < a.ks[j]; ++r) u[r] = scattered(p, r);
            run((int)p, a.ks[j], u, 0);
            (v == 1 ? ugpu : xgpu)[j].push_back(m.last_gpu_ms());
          }
        }
        // then a few eager steps of each kind for their routing: distinct experts per layer
        RouteSink rs[3];
        for (int v = 0; v < 3; ++v) {
          m.set_pos_stride(v ? a.bench_stride : 1);
          m.set_route_shift(v == 2 ? kRouteShift : 0);
          for (int i = 0; i < 6; ++i, p += a.ks[j]) {
            Cmd c{};
            c.op = kCmdStep;
            c.rows = a.ks[j];
            c.pos0 = (int)p;
            c.flags = model::kFlagDump;
            for (int r = 0; r < a.ks[j]; ++r) c.tokens[r] = v ? scattered(p, r) : toks[p + r];
            bcast(ring, c);
            m.step(to_ctl(c), sampled, a.rank == 0 ? &rs[v] : nullptr);
          }
        }
        m.set_pos_stride(1);
        m.set_route_shift(0);
        for (int L = 0; L < model::kLayers; ++L) {
          for (int e = 0; e < model::kExperts; ++e) route_all.freq[L][e] += rs[0].freq[L][e];
          route_all.rows[L] += rs[0].rows[L];
        }
        if (a.rank == 0)
          std::printf("  k %d routed experts per layer (distinct, of %d): consecutive %.1f, unrelated %.1f, rotated %.1f\n",
                      a.ks[j], a.ks[j] * model::kTopK, rs[0].distinct / std::max(1, rs[0].n),
                      rs[1].distinct / std::max(1, rs[1].n), rs[2].distinct / std::max(1, rs[2].n));
        // then runs of G consecutive rows: k / G sessions' verify rows, each run from its own scattered golden
        // position, the runs bench_stride apart; natural, then each run's routing rotated kRouteShift experts
        // further than the run before (sessions that share few experts); a few eager steps of each for their
        // distinct experts
        for (int G : a.bench_groups) {
          if (G <= 1 || G >= a.ks[j] || a.ks[j] % G) continue;
          auto run_tok = [&](size_t q, int r) {
            const uint64_t g = (uint64_t)(r / G);
            return toks[(size_t)(((q + g) * 2654435761ull + g * 40503ull) % (uint64_t)(L - G)) + r % G];
          };
          std::vector<double> gg[2];
          RouteSink gs[2];
          m.set_pos_stride(a.bench_stride, G);
          for (int v = 0; v < 2; ++v) {
            m.set_route_shift(v ? kRouteShift : 0, G);
            int32_t u[model::kMaxRows];
            for (int i = 0; i < a.steps; ++i, p += a.ks[j]) {
              for (int r = 0; r < a.ks[j]; ++r) u[r] = run_tok(p, r);
              run((int)p, a.ks[j], u, 0);
              gg[v].push_back(m.last_gpu_ms());
            }
            for (int i = 0; i < 6; ++i, p += a.ks[j]) {
              Cmd c{};
              c.op = kCmdStep;
              c.rows = a.ks[j];
              c.pos0 = (int)p;
              c.flags = model::kFlagDump;
              for (int r = 0; r < a.ks[j]; ++r) c.tokens[r] = run_tok(p, r);
              bcast(ring, c);
              m.step(to_ctl(c), sampled, &gs[v]);
            }
          }
          m.set_pos_stride(1);
          m.set_route_shift(0);
          std::printf("  k %d in %d runs of %d rows %d apart, gpu (ms): natural %s, %.1f experts per layer; rotated %s, "
                      "%.1f\n",
                      a.ks[j], a.ks[j] / G, G, a.bench_stride, stat(gg[0]).c_str(), gs[0].distinct / std::max(1, gs[0].n),
                      stat(gg[1]).c_str(), gs[1].distinct / std::max(1, gs[1].n));
        }
      }
      if (a.bench_stride > 0 && a.rank == 0) {
        // rows routed independently, each as often as the consecutive rows above routed to each expert
        std::printf("  independent rows by the experts' frequencies over %d rows, distinct per layer:", route_all.rows[0]);
        for (int B = 1; B <= 6; ++B) std::printf(" %d: %.1f", B, route_all.independent(B));
        std::printf("\n");
      }
      if (c == 0) CUDA_CHECK(cudaProfilerStop());
      if (c + 1 == ctxs.size() && !a.graph_sweep) bcast(ring, Cmd{});  // stop
      std::printf("bench: layers [%d, %d), %s, context %d, %d steps per k; chosen tokens hash %016llx\n", a.first,
                  a.layers, a.graphs ? "graphs" : "eager", ctx, a.steps, (unsigned long long)chosen_hash);
      if (n_pref && a.big)
        std::printf("  prefill: %d chunks of up to %d rows, %.2f s\n", n_pref, a.prefill_rows, t_pref);
      else if (n_pref)
        std::printf("  prefill: %d chunks of %d rows, %.2f ms each\n", n_pref, a.chunk, 1e3 * t_pref / n_pref);
      for (size_t j = 0; j < a.ks.size(); ++j) {
        if (host[j].empty()) continue;  // a k past kMaxSessRows: its runs of several sessions are printed above
        std::printf("  k %d step, host (ms): %s\n", a.ks[j], stat(host[j]).c_str());
        std::printf("  k %d step, gpu  (ms): %s\n", a.ks[j], stat(gpu[j]).c_str());
        std::printf("  k %d bcast      (ms): %s\n", a.ks[j], stat(bc[j]).c_str());
        if (!ugpu[j].empty())
          std::printf("  k %d unrelated rows %d apart, gpu (ms): %s\n", a.ks[j], a.bench_stride, stat(ugpu[j]).c_str());
        if (!xgpu[j].empty())
          std::printf("  k %d the same, routing rotated, gpu (ms): %s\n", a.ks[j], stat(xgpu[j]).c_str());
      }
    }
    if (a.graph_sweep) {
      // a step ending at the last position of every context bucket up to --max-ctx, for each k: each one
      // without a graph yet captures and instantiates it, as a decode loop does on entering a bucket. The
      // rows overwrite positions the benches have passed (their state is not used again).
      size_t free0 = 0, free1 = 0, total = 0;
      const int g0 = m.graph_count();
      CUDA_CHECK(cudaMemGetInfo(&free0, &total));
      for (int k : a.ks) {
        if (k > model::kMaxSessRows) continue;  // one session's steps only
        std::vector<double> t;
        for (int b = model::kCtxBucket; b <= a.max_ctx; b += model::kCtxBucket) {
          const int n0 = m.graph_count();
          const double dt = run(b - k, k, &toks[b - k], 0);
          if (m.graph_count() > n0) t.push_back(1e3 * dt);
        }
        std::sort(t.begin(), t.end());
        if (!t.empty())
          std::printf("graph sweep: k %d, %zu new graphs; a step that captures one takes %.1f ms median, %.1f max\n", k,
                      t.size(), t[t.size() / 2], t.back());
      }
      CUDA_CHECK(cudaMemGetInfo(&free1, &total));
      bcast(ring, Cmd{});  // stop
      const int ng = m.graph_count() - g0;
      std::printf("graph sweep: %d graphs in all; device memory free %.0f -> %.0f MiB, %.2f MiB per new graph\n",
                  m.graph_count(), free0 / 1048576.0, free1 / 1048576.0,
                  ng ? ((double)free0 - (double)free1) / 1048576.0 / ng : 0.0);
    }
    return 0;
  }

  // EXPERIMENT (--seq-list, generate --loop; docs/engine.md "Prompt-lookup drafts"): the list's sequences one
  // after another, each prompt from position 0 through the prefill path, then up to N tokens through the
  // decode loop (with --spec, today's drafts: their speed on that output), ending at EOS
  if (a.mode == "generate" && !a.seq_list.empty()) {
    const std::vector<SeqLine> seqs = read_seq_list(a.seq_list);
    for (size_t sq = 0; sq < seqs.size(); ++sq) {
      const SeqLine& s = seqs[sq];
      if (s.prompt < 1 || s.prompt > (int)s.toks.size() || s.n < 1 || s.prompt + s.n + 8 > a.max_ctx)
        die("--seq-list: %s: prompt %d of %zu tokens, %d to generate, context %d", s.dir.c_str(), s.prompt,
            s.toks.size(), s.n, a.max_ctx);
      t_pref = 0, n_pref = 0;
      const int32_t tok0 = big_prefill(0, s.prompt, s.toks.data());
      const double tp0 = t_pref;
      std::vector<int32_t> out{tok0}, rest;
      std::vector<double> times;
      chosen_hash = 1469598103934665603ull;
      const double td = now_s();
      const DecodeEnd e = run_decode(s.prompt, tok0, s.n - 1, rest, times);
      const double dt = now_s() - td;
      out.insert(out.end(), rest.begin(), rest.end());
      const int nt = (int)times.size(), skip = std::min(8, nt / 4);
      std::printf("seq %zu (%s): prompt %d in %.2f s; %zu tokens in %.2f s, %.1f tok/s after the first %d, end %u; "
                  "tokens hash %016llx\n",
                  sq, s.dir.c_str(), s.prompt, tp0, out.size(), dt,
                  nt - 1 - skip > 0 ? (nt - 1 - skip) / (times[nt - 1] - times[skip]) : 0.0, skip, e.why,
                  (unsigned long long)chosen_hash);
      spec_report(e);
      std::printf("seq %zu generated:", sq);
      for (int t : out) std::printf(" %d", t);
      std::printf("\n");
      std::fflush(stdout);
    }
    bcast(ring, Cmd{});  // stop
    return 0;
  }

  // prompt / prefill in chunks
  const uint32_t pflags = a.mode == "generate" ? model::kFlagLogits : check;
  if (a.big && a.mode != "prefill") {
    sampled[(prompt - 1) % a.chunk] = big_prefill(0, prompt, toks.data());
    std::printf("prompt: %d positions through the prefill path in %.2f s\n", prompt, t_pref);
  }
  for (int p = 0; p < prompt && !(a.big && a.mode != "prefill"); p += a.chunk) {
    const int R = std::min(a.chunk, prompt - p);
    uint32_t f = pflags;  // dump and inject only where the golden run saved the positions
    bool any = false, all = true;
    for (int r = 0; r < R; ++r) any |= g.saved(p + r), all &= g.saved(p + r);
    if (!any) f &= ~model::kFlagDump;
    if (!all) f &= ~(model::kFlagInject | model::kFlagInjectSub);
    t_pref += run(p, R, &toks[p], f);
    ++n_pref;
    if (!a.quiet && (n_pref % 256 == 0 || p + R == prompt))
      std::printf("prefill %d/%d positions, %.1f ms per chunk\n", p + R, prompt, 1e3 * t_pref / n_pref);
  }
  int first_miss = -1;
  if (a.mode == "tf") {
    for (int p = prompt; p < L; p += a.k) {
      const int R = std::min(a.k, L - p);
      t_dec += run(p, R, &toks[p], check);
      ++n_dec;
    }
    if (a.big && n_dec == 0) {
      // Nothing teacher-forced (--limit = --prompt, as long28k's CED check): the hash covers the prompt's
      // sample from the prefill path instead: its token, logit and logprob, then its top logprobs' ids and
      // values. (Until 2026-10-13 such a run hashed no row and printed the hash's seed, 14650fb0739d0383,
      // whatever the engine computed.) A run with steps hashes them alone, as before.
      const model::SampleOut& o = prompt_out;
      const int nt = std::min(o.n_top, model::ops::kTopLogprobs);
      const auto fold = [&](uint32_t x) { chosen_hash = (chosen_hash ^ x) * 1099511628211ull; };
      const auto bits = [](float v) {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        return b;
      };
      fold((uint32_t)o.row[0].token);
      fold(bits(o.row[0].value));
      fold(bits(o.row[0].logprob));
      for (int j = 0; j < nt; ++j) fold((uint32_t)o.top_id[0][j]), fold(bits(o.top_lp[0][j]));
      // equal hashes: the same sample after the prompt, bit for bit
      std::printf("teacher-forced: 0 positions; the prompt's sample (token, logit, logprob, top %d logprobs) hash "
                  "%016llx\n", nt, (unsigned long long)chosen_hash);
    } else {
      // equal hashes: the same state after the prompt, bit for bit (e.g. prefill chunks of another size)
      std::printf("teacher-forced: %d positions, chosen tokens and logits hash %016llx\n", L - prompt,
                  (unsigned long long)chosen_hash);
    }
  } else if (a.mode == "generate" && a.loop) {
    // the same generation through the decode loop: tok_0 is the prompt's sample, the loop samples the rest
    std::vector<int32_t> out{sampled[(prompt - 1) % a.chunk]}, rest;
    std::vector<double> times;
    const double td = now_s();
    spec_report(run_decode(prompt, out[0], a.gen - 1, rest, times));
    t_dec = now_s() - td;
    n_dec = (int)rest.size();
    out.insert(out.end(), rest.begin(), rest.end());
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    int same = 0;
    while (same < (int)out.size() && prompt + same < L && out[same] == toks[prompt + same]) ++same;
    std::printf("generation equals the golden (SGLang) tokens for %d of %zu; tokens hash %016llx\n", same,
                out.size(), (unsigned long long)chosen_hash);
  } else if (a.mode == "generate") {
    int next = sampled[(prompt - 1) % a.chunk];
    std::vector<int32_t> out;
    for (int i = 0; i < a.gen && prompt + i < a.max_ctx; ++i) {
      out.push_back(next);
      const int p = prompt + i;
      if (first_miss < 0 && p < L && next != toks[p]) first_miss = p;
      t_dec += run(p, 1, &next, model::kFlagLogits);
      ++n_dec;
      next = sampled[0];
    }
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    int same = 0;
    while (same < (int)out.size() && prompt + same < L && out[same] == toks[prompt + same]) ++same;
    std::printf("generation equals the golden (SGLang) tokens for %d of %zu; first difference at position %d\n", same,
                out.size(), first_miss);
  } else if (a.mode != "prefill") {
    die("unknown mode %s", a.mode.c_str());
  }
  bcast(ring, Cmd{});  // stop
  if (n_pref) std::printf("prefill: %d chunks, %.2f ms each\n", n_pref, 1e3 * t_pref / n_pref);
  if (n_dec) std::printf("decode: %d steps, %.2f ms each\n", n_dec, 1e3 * t_dec / n_dec);
  if (a.mode != "generate") g.report(stdout);
  if (a.mode != "generate" && a.onset >= 0) g.report(stdout, true);
  g.report_logits(stdout, a.mode == "tf" ? prompt : 0);
  g.report_cmp(stdout, 0, L - 1);
  g.report_cmp(stdout, std::min(prompt, 39) - 1, L - 1);
  if (a.onset >= 0) {  // windows of 4096 positions, split at --onset; then everything from the onset on
    std::vector<int> e;
    for (int p = 0; p < L; p += 4096) {
      if (a.onset > p && a.onset < p + 4096) e.push_back(p), p = a.onset - 4096;
      else e.push_back(p);
    }
    e.push_back(L);
    g.report_windows(stdout, e);
    g.report_windows(stdout, {0, a.onset, L});
    g.report_windows(stdout, {0, 128, 256, 512, 1024, 2048, 4096});
  }
  if (a.per_position) g.report_positions(stdout);
  if (a.mode == "generate" && first_miss >= 0) {
    for (const golden::LogitRec& r : g.logit_recs())
      if (r.pos == first_miss - 1)
        std::printf("at the first difference: ours %d (margin %.3f), golden %d (golden margin %.3f)\n", r.ours,
                    r.our_margin, r.gold, r.gold_margin);
  }
  return 0;
}
