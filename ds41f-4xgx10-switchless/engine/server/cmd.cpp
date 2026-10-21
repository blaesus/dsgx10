// Rank 0's commands and what every rank runs alike (cmd.h).
#include "server/cmd.h"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "common/check.h"
#include "model/nvtx.h"

namespace srv {

std::vector<double> g_t_wait, g_t_late, g_t_launch;

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void pin_self(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

Watchdog::Watchdog(int timeout_s, comm::Ring& ring) {
  th = std::thread([this, timeout_s, &ring] {
    while (!stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      const double t = started;
      if (t > 0 && now_s() - t > timeout_s) {
        std::fprintf(stderr, "watchdog: step running for %d s, aborting\n%s\n", timeout_s, ring.describe().c_str());
        std::fflush(stderr);
        _exit(3);
      }
    }
  });
}

Watchdog::~Watchdog() {
  stop = true;
  th.join();
}

void apply_state(model::Model& m, const StateOp& o) {
  switch (o.op) {
    case kStSave: m.state_save(o.h, o.a, o.b); break;
    case kStSnapEnd: m.state_snap_end(o.h, o.a); break;
    case kStLoad: m.state_load(o.h, o.a); break;
    case kStRestore: m.state_restore(o.h, o.a); break;
    case kStRestoreEnd: m.state_restore_end(o.h); break;
    case kStDrop: m.state_drop(o.h, o.a); break;
    case kStFree: m.state_free(o.h); break;
    case kStCopySnaps: m.state_copy_snaps(o.h, o.a, o.b); break;
    case kStDropSnap: m.state_drop_snap(o.h, o.a); break;
    case kStSnap: m.state_snap(o.h, o.a); break;
    case kStSpill: m.state_spill(o.h); break;
    case kStUnspill: m.state_unspill(o.h); break;
    case kStKeep: m.state_keep((uint64_t)(uint32_t)o.a | (uint64_t)(uint32_t)o.b << 32); break;
    default: die("state: unknown operation %d", o.op);
  }
}

void apply_ops(model::Model& m, const Cmd& c) {
  if (c.slot >= 0) m.use_slot(c.slot);
  std::vector<int> pin;
  for (int i = 0; i < c.n_ops; ++i) {
    pin.push_back(c.ops[i].h);
    if (c.ops[i].op == kStCopySnaps) pin.push_back(c.ops[i].a);
  }
  if (c.handle >= 0) pin.push_back(c.handle);
  for (const int32_t h : c.live)
    if (h > 0) pin.push_back(h);
  m.state_pin(pin);
  for (int i = 0; i < c.n_ops; ++i) apply_state(m, c.ops[i]);
}

Cmd bcast(comm::Ring& ring, const Cmd& mine) {
  Cmd all[4];
  ring.allgather_host(&mine, all, sizeof(Cmd));
  return all[0];
}

void bcast_payload(comm::Ring& ring, void* data, size_t nbytes, int rank) {
  constexpr size_t kPiece = 65536;  // allgather_host's limit
  std::vector<uint8_t> all(4 * kPiece), mine(kPiece);
  for (size_t off = 0; off < nbytes; off += kPiece) {
    const size_t n = std::min(kPiece, nbytes - off);
    if (rank == 0) std::memcpy(mine.data(), static_cast<uint8_t*>(data) + off, n);
    ring.allgather_host(mine.data(), all.data(), n);
    if (rank != 0) std::memcpy(static_cast<uint8_t*>(data) + off, all.data(), n);
  }
}

model::Ctl to_ctl(const Cmd& c) {
  model::Ctl t{};
  t.op = model::kOpStep;
  t.rows = c.rows;
  t.pos0 = c.pos0;
  t.flags = c.flags;
  std::memcpy(t.tokens, c.tokens, sizeof t.tokens);
  t.temperature = c.sp.temperature;
  t.seed = (uint32_t)c.sp.seed;
  return t;
}

uint64_t run_prefill(model::Model& m, const Cmd& c) {
  const std::vector<int> snaps(c.snaps, c.snaps + c.n_snaps);
  const uint64_t n =
      m.prefill(c.pos0, c.rows, (c.flags & kPfSample) ? &c.sp : nullptr, c.handle, snaps, (c.flags & kPfCed) != 0);
  if (c.next > 0) m.prefill_lookup(c.pos0 + c.rows, c.next);
  return n;
}

uint64_t run_rows(model::Model& m, const Cmd& c) {
  uint64_t n;
  if (m.spec()) {  // the rows go into the draft module's windows too
    model::SpecCfg sc;
    sc.max_drafts = 0;
    sc.pos_end = c.pos0 + c.rows + 1;
    sc.force = c.spec_force;
    std::copy(c.spec_probe, c.spec_probe + model::ops::kDrafts, sc.probe);
    n = m.launch_spec(c.rows, c.pos0, c.tokens, c.sp, sc);
  } else {
    n = m.launch(c.rows, c.pos0, c.tokens, c.sp);
  }
  for (int i = 0; i < c.n_snaps; ++i) m.state_snap(c.handle, c.snaps[i]);
  return n;
}

void SpecRate::add(const Cmd& c, int emitted, int rows) {
  constexpr double a = 1.0 / 16;
  if (c.spec_gate != model::ops::kGateThresh) {
    if (ms == 0) ms = c.spec_row_ms[0];
    tok += a * (emitted - tok);
    ms += a * (c.spec_row_ms[std::clamp(rows - 1, 0, model::ops::kDrafts)] - ms);
    return;
  }
  if (ms == 0) ms = c.spec_cost[0] + c.spec_cost[2];
  tok += a * (emitted - tok);
  ms += a * (c.spec_cost[0] + (rows - 1) * c.spec_cost[1] + c.spec_cost[2] - ms);
}

float SpecRate::thresh(const Cmd& c) const {
  if (c.spec_gate != model::ops::kGateThresh) return (float)(tok / (ms > 0 ? ms : c.spec_row_ms[0]));
  if (c.spec_thresh >= 0) return c.spec_thresh;
  const double m = ms > 0 ? ms : c.spec_cost[0] + c.spec_cost[2];
  return (float)(tok / m * c.spec_cost[1]);
}

namespace {

// decode_loop with DSpark (cmd.h)
DecodeEnd spec_loop(model::Model& m, const Cmd& c, Watchdog& wd, const OnSample& on) {
  constexpr int kD = model::ops::kDrafts;
  DecodeEnd e;
  e.written = c.pos0;
  // verify rows stay below pos_end: the tokens they emit stay within max_new and the context. As in
  // plain decode, the step after the last token still runs (up to `feed`), so the KV of every emitted
  // token is there for the session's next request.
  const int pos_end = (int)std::min<int64_t>(c.limit, (int64_t)c.pos0 + c.max_new);
  const int feed = std::min(pos_end, c.limit - 1);  // the last position a step may start at
  e.spec_tok = c.spec_tok, e.spec_ms = c.spec_ms;
  if (c.max_new < 1 || c.pos0 >= c.limit) {
    e.why = c.max_new < 1 ? kEndLength : kEndContext;
    return e;
  }
  SpecRate rate{c.spec_tok, c.spec_ms};
  model::SpecCfg cfg;
  cfg.max_drafts = c.spec_drafts;
  cfg.pos_end = pos_end;
  cfg.gate = c.spec_gate;
  std::copy(c.spec_row_ms, c.spec_row_ms + model::ops::kDrafts + 1, cfg.step_ms);
  auto verify = [&](int bound) {
    cfg.thresh = rate.thresh(c);
    e.gate(cfg.thresh);
    return m.launch_verify(c.sp, cfg, std::min(bound, c.limit - 1));
  };
  std::vector<int32_t> toks{c.tokens[0]};  // tok_0, tok_1, ...: positions pos0, pos0 + 1, ...
  int next_snap = c.snap_every > 0 ? (c.pos0 / c.snap_every + 1) * c.snap_every : INT32_MAX;
  cfg.thresh = rate.thresh(c);
  e.gate(cfg.thresh);
  const uint64_t first = m.launch_spec(1, c.pos0, c.tokens, c.sp, cfg);
  int launched = 1, prev_rows = 1;
  if (c.ahead && c.pos0 + 1 <= feed) verify(c.pos0 + 1 + kD), ++launched;
  for (int i = 0;; ++i) {
    wd.started = now_s();
    const model::SampleOut& o = m.wait(first + i);
    wd.started = 0;
    g_t_wait.push_back(now_s());
    if (i + 1 < launched) m.engram_late(o.next_pos0, o.next_rows, o.next_tok);  // step i + 1's rows
    g_t_late.push_back(now_s());
    e.written = std::max(e.written, o.pos0 + std::max(o.step_rows, o.rows + kD));
    if (i > 0) {
      ++e.vsteps, e.verified += o.step_rows - 1, e.accepted += o.rows - 1;
      e.probability(o, 0, o.step_rows - 1);
      ++e.rows_hist[std::min(o.step_rows - 1, kD)];
    }
    if (i > 4)  // the step before (done; past the graphs' captures): its GPU time by its rows
      e.gpu_ms[prev_rows] += m.gpu_ms(first + i - 1), ++e.gpu_n[prev_rows];
    prev_rows = o.step_rows;
    rate.add(c, o.rows, o.step_rows);
    e.spec_tok = rate.tok, e.spec_ms = rate.ms;
    // the tokens it emits, until a stop
    bool stop = false;
    int u = 0;
    for (int j = 0; j < o.rows && !stop; ++j) {
      const int32_t t = o.row[j].token;
      if (on) on(e.sampled, o, j);
      ++e.sampled, ++u;
      toks.push_back(t);
      if (o.ctl) stop = true, e.why = kEndCancel;
      for (int k = 0; !stop && k < c.n_stop; ++k)
        if (t == c.stop[k]) stop = true, e.why = kEndStop;
      if (!stop && e.sampled >= c.max_new) stop = true, e.why = kEndLength;
    }
    if (!stop && o.next_pos0 >= pos_end) stop = true, e.why = kEndContext;
    // positions o.pos0 .. F hold their final KV; the history now ends at H (F + 1: the token after them,
    // whose KV the next step's first row makes)
    const int F = o.pos0 + o.rows - 1, H = o.pos0 + u;
    bool next_ran = false;
    if (stop && i + 1 == launched && H > F && H <= feed) {  // the step that feeds the last token
      verify(H), ++launched;
      m.engram_late(o.next_pos0, o.next_rows, o.next_tok);
    }
    if (stop && i + 1 < launched) {  // the queued step still runs
      wd.started = now_s();
      const model::SampleOut& q = m.wait(first + i + 1);
      wd.started = 0;
      e.written = std::max(e.written, q.pos0 + std::max(q.step_rows, q.rows + kD));
      next_ran = true;
    }
    const int kv_end = H <= F || next_ran ? H + 1 : F + 1;  // positions [pos0, kv_end) hold the history's KV
    for (; next_snap <= kv_end; next_snap += c.snap_every)  // window snapshots (the ring holds s - 128 .. s - 1)
      if (c.handle >= 0 && next_snap > c.pos0 && next_snap >= 128) m.state_snap(c.handle, next_snap);
    if (stop) {
      e.fed = kv_end - c.pos0;
      e.last = toks[e.fed - 1];
      return e;
    }
    if (i + 1 == launched) {  // not queued ahead: the rows are known now
      verify(o.next_pos0 + o.next_rows - 1), ++launched;
      m.engram_late(o.next_pos0, o.next_rows, o.next_tok);
    }
    // the step after it starts at most at next_pos0 + next_rows (every draft accepted)
    if (c.ahead && launched == i + 2 && o.next_pos0 + o.next_rows <= feed)
      verify(o.next_pos0 + o.next_rows + kD), ++launched;
    g_t_launch.push_back(now_s());
  }
}

}  // namespace

DecodeEnd decode_loop(model::Model& m, const Cmd& c, Watchdog& wd, const OnSample& on) {
  g_t_wait.clear(), g_t_late.clear(), g_t_launch.clear();
  if (c.spec) return spec_loop(m, c, wd, on);
  auto may_run = [&](int i) { return i <= c.max_new && c.pos0 + i < c.limit; };
  // a step at position pos; the window snapshot at pos first (the ring then holds pos - 128 .. pos - 1)
  auto launch = [&](int pos, const int32_t* tok) {
    if (c.snap_every > 0 && c.handle >= 0 && pos > c.pos0 && pos >= 128 && pos % c.snap_every == 0)
      m.state_snap(c.handle, pos);
    return m.launch(1, pos, tok, c.sp);
  };
  DecodeEnd e;
  if (c.max_new < 1 || !may_run(0)) {
    e.why = c.max_new < 1 ? kEndLength : kEndContext;
    e.written = c.pos0;
    return e;
  }
  const int32_t from_device = -1;
  const uint64_t first = launch(c.pos0, c.tokens);
  int launched = 1;
  if (c.ahead && may_run(1)) launch(c.pos0 + 1, &from_device), ++launched;
  int32_t fed = c.tokens[0];
  for (int i = 0;; ++i) {
    wd.started = now_s();
    const model::SampleOut& o = m.wait(first + i);
    wd.started = 0;
    g_t_wait.push_back(now_s());
    const int32_t t = o.row[0].token;
    if (on) on(i, o, 0);
    e.sampled = i + 1;
    if (!c.ahead && i + 1 <= c.max_new && may_run(i + 1)) {  // diagnostics: launch only now
      bool stop = o.ctl != 0 || i + 1 >= c.max_new;
      for (int k = 0; k < c.n_stop; ++k) stop |= t == c.stop[k];
      if (!stop) launch(c.pos0 + i + 1, &t), ++launched;
    }
    if (c.ahead && i + 1 < launched) m.engram_late(c.pos0 + i + 1, 1, &t);  // step i + 1 feeds t
    g_t_late.push_back(now_s());
    bool stop = false;
    if (o.ctl) stop = true, e.why = kEndCancel;
    for (int k = 0; !stop && k < c.n_stop; ++k)
      if (t == c.stop[k]) stop = true, e.why = kEndStop;
    if (!stop && i + 1 >= c.max_new) stop = true, e.why = kEndLength;
    if (!stop && i + 1 >= launched) stop = true, e.why = kEndContext;
    if (stop) {
      if (i + 1 < launched) {  // the queued step: its KV counts, its sample does not
        wd.started = now_s();
        m.wait(first + i + 1);
        wd.started = 0;
        e.fed = i + 2, e.last = t;
      } else {
        e.fed = i + 1, e.last = fed;
      }
      e.written = c.pos0 + e.fed;
      return e;
    }
    fed = t;
    if (c.ahead && may_run(i + 2)) launch(c.pos0 + i + 2, &from_device), ++launched;
    g_t_launch.push_back(now_s());
  }
}

std::vector<BatchEnd> batch_loop(model::Model& m, const Cmd& c, const std::vector<BatchEntry>& e, Watchdog& wd,
                                 const OnBatch& on) {
  g_t_wait.clear(), g_t_late.clear(), g_t_launch.clear();
  const int n = (int)e.size(), slot0 = m.slot();
  std::vector<BatchEnd> end(n);
  // per entry: its steps launched, its row in the last step launched, sampling (live) or with its last
  // row still queued (ending)
  std::vector<int> launched(n, 0), last_row(n, -1);
  std::vector<char> live(n, 1), ending(n, 0);
  auto may_run = [&](int k, int j) { return j <= e[k].max_new && e[k].pos0 + j < e[k].limit; };
  std::vector<int> pin;
  for (const int32_t h : c.live)
    if (h > 0) pin.push_back(h);
  for (int k = 0; k < n; ++k) {
    if (e[k].handle >= 0) pin.push_back(e[k].handle);
    if (!may_run(k, 0)) live[k] = 0, end[k].why = e[k].max_new < 1 ? kEndLength : kEndContext;
  }
  m.state_pin(pin);  // the sessions' stored state, while snapshots make room
  struct Step {
    uint64_t id = 0;
    int R = 0;
    int ent[model::kMaxRows] = {};  // row r's entry
  };
  std::deque<Step> q;  // launched, not yet waited for
  // the next step of every session still decoding, a row each; its window snapshot first (the ring then
  // holds pos - 128 .. pos - 1)
  auto launch = [&] {
    model::Batch b;
    Step s;
    for (int k = 0; k < n; ++k) {
      const int j = launched[k];
      if (!live[k] || ending[k] || !may_run(k, j)) {
        last_row[k] = -1;
        continue;
      }
      const int r = b.R++, pos = e[k].pos0 + j;
      if (e[k].snap_every > 0 && e[k].handle >= 0 && j > 0 && pos >= 128 && pos % e[k].snap_every == 0) {
        m.use_slot(e[k].slot);
        m.state_snap(e[k].handle, pos);
      }
      b.tokens[r] = j == 0 ? e[k].token : -1 - last_row[k];  // the step before holds its row
      b.pos[r] = pos, b.slot[r] = e[k].slot, b.sess[r] = r, b.sp[r] = e[k].sp;
      s.ent[r] = k;
      last_row[k] = r;
      ++launched[k];
    }
    if (!b.R) return;
    b.nsess = s.R = b.R;
    s.id = m.launch_batch(b);
    q.push_back(s);
  };
  // entry k ends with why after its sample tok_{sampled}: its row in the step queued, if any, still runs
  auto stop = [&](int k, uint32_t why) {
    end[k].why = why;
    if (launched[k] > end[k].sampled) ending[k] = 1;
    else live[k] = 0, end[k].fed = launched[k];
  };
  launch();
  launch();  // queued behind the first, its tokens from the device
  while (!q.empty()) {
    const Step s = q.front();
    q.pop_front();
    wd.started = now_s();
    const model::SampleOut& o = m.wait(s.id);
    wd.started = 0;
    g_t_wait.push_back(now_s());
    std::vector<int> row_of(n, -1);  // entry -> its row in this step
    for (int r = 0; r < s.R; ++r) row_of[s.ent[r]] = r;
    if (!q.empty()) {  // the step queued behind this one feeds its samples: their Engram lookups
      const Step& nx = q.front();
      int32_t t[model::kMaxRows];
      for (int r = 0; r < nx.R; ++r) t[r] = o.row[row_of[nx.ent[r]]].token;
      m.engram_late_batch(t);
    }
    g_t_late.push_back(now_s());
    for (int r = 0; r < s.R; ++r) {
      const int k = s.ent[r];
      BatchEnd& x = end[k];
      if (ending[k]) {  // its row behind its end: the KV counts; the sample too on a pause, within max_new
        if (x.why == kEndPause && x.sampled < e[k].max_new) {
          if (on) on(k, o, r);
          ++x.sampled;
        }
        live[k] = 0, ending[k] = 0, x.fed = launched[k];
        continue;
      }
      const int32_t t = o.row[r].token;
      if (on) on(k, o, r);
      ++x.sampled;
      if (o.ctl & (1u << e[k].slot)) stop(k, kEndCancel);
      else if (std::find(e[k].stop, e[k].stop + e[k].n_stop, t) != e[k].stop + e[k].n_stop) stop(k, kEndStop);
      else if (x.sampled >= e[k].max_new) stop(k, kEndLength);
      else if (o.ctl & kCtlPause) stop(k, kEndPause);           // one that ends here ends, not pauses
      else if (launched[k] <= x.sampled) stop(k, kEndContext);  // no room for its next step
    }
    if ((c.flags & kBatchSolo) && n > 1) {  // one session left decoding: it pauses, for the drafts
      int left = 0, k1 = -1;
      for (int k = 0; k < n; ++k)
        if (live[k] && !ending[k]) ++left, k1 = k;
      if (left == 1) stop(k1, kEndPause);
    }
    launch();
    g_t_launch.push_back(now_s());
  }
  m.use_slot(slot0);
  return end;
}

void bcast_tf(comm::Ring& ring, const Cmd& c, std::vector<BatchEntry>& e, std::vector<std::vector<int32_t>>& tokens,
              int rank) {
  if (rank != 0) e.assign(c.n, BatchEntry{}), tokens.assign(c.n, {});
  if (c.n < 1 || (int)e.size() != c.n || (int)tokens.size() != c.n) die("batch_tf: %d entries", c.n);
  bcast_payload(ring, e.data(), e.size() * sizeof(BatchEntry), rank);
  size_t total = 0;
  for (int k = 0; k < c.n; ++k) total += (size_t)std::max(0, e[k].max_new);
  std::vector<int32_t> all;
  if (rank == 0)
    for (const std::vector<int32_t>& t : tokens) all.insert(all.end(), t.begin(), t.end());
  else
    all.resize(total);
  if (all.size() != total) die("batch_tf: %zu tokens for %zu", all.size(), total);
  if (total) bcast_payload(ring, all.data(), total * 4, rank);
  if (rank != 0)
    for (int k = 0, off = 0; k < c.n; off += e[k].max_new, ++k)
      tokens[k].assign(all.begin() + off, all.begin() + off + std::max(0, e[k].max_new));
}

std::vector<float> batch_tf_loop(model::Model& m, const Cmd& c, const std::vector<BatchEntry>& e,
                                 const std::vector<std::vector<int32_t>>& tokens, Watchdog& wd, const OnBatch& on,
                                 model::Sink* sink) {
  const int n = (int)e.size(), G = c.rows;
  if (n < 1 || n > model::ops::kMaxSess || G < 1 || G > model::kMaxSessRows || (int)tokens.size() != n)
    die("batch_tf: %d entries, %d rows of each per step (1..%d)", n, G, model::kMaxSessRows);
  for (int k = 0; k < n; ++k)
    if (e[k].max_new < 0 || (int)tokens[k].size() != e[k].max_new || e[k].pos0 < 0 ||
        e[k].pos0 + e[k].max_new > e[k].limit)
      die("batch_tf: entry %d: %d tokens at %d, limit %d", k, e[k].max_new, e[k].pos0, e[k].limit);
  std::vector<int> fed(n, 0);
  std::vector<float> ms;
  for (;;) {
    model::Batch b;
    int ent[model::kMaxRows] = {};  // row r's entry
    for (int k = 0; k < n; ++k) {
      const int rows = std::min(G, e[k].max_new - fed[k]);
      if (rows <= 0) continue;
      if (b.R + rows > model::kMaxRows) die("batch_tf: a step of more than %d rows", model::kMaxRows);
      const int s = b.nsess++;
      b.sp[s] = e[k].sp;
      for (int i = 0; i < rows; ++i) {
        const int r = b.R++;
        b.tokens[r] = tokens[k][fed[k] + i];
        b.pos[r] = e[k].pos0 + fed[k] + i;
        b.slot[r] = e[k].slot;
        b.sess[r] = s;
        ent[r] = k;
      }
      fed[k] += rows;
    }
    if (!b.R) break;
    b.flags = c.flags;
    b.sink = sink;
    wd.started = now_s();
    const model::SampleOut& o = m.wait(m.launch_batch(b));
    ms.push_back(m.last_gpu_ms());
    wd.started = 0;
    if (on)
      for (int r = 0; r < b.R; ++r) on(ent[r], o, r);
  }
  return ms;
}

// B3 (cmd.h): spec_loop's accounting per entry, over batch spec steps
std::vector<DecodeEnd> spec_batch_loop(model::Model& m, const Cmd& c, const std::vector<BatchEntry>& e, Watchdog& wd,
                                       const OnBatch& on, SpecBatchStats* st) {
  constexpr int kD = model::ops::kDrafts;
  g_t_wait.clear(), g_t_late.clear(), g_t_launch.clear();
  const int n = (int)e.size(), slot0 = m.slot();
  if (n < 1 || n > model::ops::kMaxSess || n > m.spec_batch_entries())
    die("spec batch: %d entries (the model takes %d)", n, m.spec_batch_entries());
  std::vector<DecodeEnd> end(n);
  SpecBatchStats stats;
  // per entry: the rows' bounds (spec_loop's pos_end and feed), live (not finished), ending (stopped, its
  // rows of one more step still to run: queued already (pend), or in the next step launched), the rows
  // of its next step as the last result chose them (nx), the drafts' rate, its history and snapshots
  std::vector<int> pos_end(n), feed(n), next_snap(n), F(n, 0), H(n, 0);
  std::vector<char> live(n, 1), ending(n, 0), pend(n, 0);
  std::vector<SpecRate> rate(n);
  std::vector<std::vector<int32_t>> toks(n);  // tok_0, tok_1, ..: positions pos0, pos0 + 1, ..
  struct Next {
    int rows = 1, pos0 = 0;
    int32_t tok[model::ops::kSessTok] = {};
  };
  std::vector<Next> nx(n);
  std::vector<int> pin;
  for (const int32_t h : c.live)
    if (h > 0) pin.push_back(h);
  for (int k = 0; k < n; ++k) {
    const BatchEntry& x = e[k];
    DecodeEnd& d = end[k];
    d.written = x.pos0;
    d.spec_tok = x.spec_tok, d.spec_ms = x.spec_ms;
    rate[k] = SpecRate{x.spec_tok, x.spec_ms};
    pos_end[k] = (int)std::min<int64_t>(x.limit, (int64_t)x.pos0 + x.max_new);
    feed[k] = std::min(pos_end[k], x.limit - 1);
    toks[k].push_back(x.token);
    nx[k].pos0 = x.pos0, nx[k].tok[0] = x.token;
    next_snap[k] = x.snap_every > 0 ? (x.pos0 / x.snap_every + 1) * x.snap_every : INT32_MAX;
    if (x.handle >= 0) pin.push_back(x.handle);
    if (x.max_new < 1 || x.pos0 >= x.limit) live[k] = 0, d.why = x.max_new < 1 ? kEndLength : kEndContext;
  }
  m.state_pin(pin);  // the sessions' stored state, while snapshots make room
  // entry k's window snapshots up to kv_end (positions below it hold its history's KV; the ring holds
  // s - 128 .. s - 1), in its slot
  auto snaps = [&](int k, int kv_end) {
    for (; next_snap[k] <= kv_end; next_snap[k] += e[k].snap_every)
      if (e[k].handle >= 0 && next_snap[k] > e[k].pos0 && next_snap[k] >= 128) {
        m.use_slot(e[k].slot);
        m.state_snap(e[k].handle, next_snap[k]);
      }
  };
  // entry k is done: positions [pos0, kv_end) hold its history's KV
  auto finish = [&](int k, int kv_end) {
    live[k] = 0, ending[k] = 0, pend[k] = 0;
    end[k].fed = kv_end - e[k].pos0;
    end[k].last = toks[k][end[k].fed - 1];
  };
  struct Step {
    uint64_t id = 0;
    bool late = false;  // a verify step whose Engram lookups wait for its rows (engram_late_rows)
    char on[model::ops::kMaxSess] = {};
  };
  std::deque<Step> q;  // launched, not yet waited for: the one running, then the one queued behind it
  auto on_now = [&](int k) { return live[k] && !(ending[k] && pend[k]); };
  // a verify step's rows, as the device lays them out (the entries on in order, each its nx), for its
  // Engram lookups: once the step before it has reported (nx), and before it is waited for
  auto late_rows = [&](Step& s) {
    int32_t t[model::kMaxRows], p[model::kMaxRows], sl[model::kMaxRows];
    int R = 0;
    for (int k = 0; k < n; ++k) {
      if (!s.on[k]) continue;
      if (nx[k].rows < 1 || nx[k].rows > model::ops::kSessTok || R + nx[k].rows > model::kMaxRows)
        die("spec batch: entry %d brings %d rows to a step of %d", k, nx[k].rows, R);
      for (int j = 0; j < nx[k].rows; ++j, ++R) t[R] = nx[k].tok[j], p[R] = nx[k].pos0 + j, sl[R] = e[k].slot;
    }
    m.engram_late_rows(R, t, p, sl);
    s.late = false;
  };
  // A launch: the entries on (live, or ending with their rows still to run), each with the drafts' cap
  // by their count; the first step given (one row per entry), the others verify steps (the rows the step
  // before chose), bound by their rows' last position at most: queued ahead, every draft of the step
  // before accepted and kD more rows after them.
  auto launch = [&](bool given, bool ahead) {
    model::SpecBatch b;
    Step s;
    int S = 0, bound = 0, lim = INT32_MAX;
    for (int k = 0; k < n; ++k)
      if (on_now(k)) ++S;
    const int cap = S ? std::min({kD, c.spec_drafts, (model::kMaxRows - S) / S}) : 0;
    b.n = n;
    b.cfg.max_drafts = std::max(0, std::min(kD, c.spec_drafts));
    b.cfg.gate = c.spec_gate;
    std::copy(c.spec_row_ms, c.spec_row_ms + kD + 1, b.cfg.step_ms);
    for (int k = 0; k < n; ++k) {
      b.on[k] = s.on[k] = on_now(k) ? 1 : 0;
      b.slot[k] = e[k].slot, b.sp[k] = e[k].sp, b.cap[k] = cap, b.pos_end[k] = pos_end[k];
      b.token[k] = nx[k].tok[0], b.pos0[k] = nx[k].pos0;
      b.cfg.pos_end = std::max(b.cfg.pos_end, pos_end[k]);
      if (!b.on[k]) continue;
      b.thresh[k] = rate[k].thresh(c);
      end[k].gate(b.thresh[k]);
      bound = std::max(bound, ahead ? nx[k].pos0 + nx[k].rows + kD : nx[k].pos0 + nx[k].rows - 1);
      lim = std::min(lim, e[k].limit);
      if (ending[k]) pend[k] = 1;
    }
    b.pos_bound = std::min(bound, lim - 1);
    s.id = given ? m.launch_spec_batch(b) : m.launch_verify_batch(b);
    s.late = !given;  // a given step's rows are the host's: launch_spec_batch looked them up
    q.push_back(s);
  };
  // the next step may be queued before the one running ends when every entry decoding fits it: its
  // rows start at most where the running step's end (every draft accepted), at or below feed
  auto fits = [&] {
    int any = 0;
    for (int k = 0; k < n; ++k) {
      if (!live[k] || ending[k]) continue;
      if (nx[k].pos0 + nx[k].rows > feed[k]) return false;
      ++any;
    }
    return any > 0;
  };
  bool any_live = false;
  for (int k = 0; k < n; ++k) any_live |= live[k] != 0;
  if (any_live) {
    launch(true, false);
    if (c.ahead && fits()) launch(false, true);
  }
  uint64_t prev_id = 0;
  int prev_R = 0;
  for (int i = 0; !q.empty(); ++i) {
    const Step s = q.front();
    q.pop_front();
    wd.started = now_s();
    const model::SampleOut& o = m.wait(s.id);
    wd.started = 0;
    g_t_wait.push_back(now_s());
    for (int k = 0; k < n; ++k) {  // each entry's next rows, the device's choice
      if (!s.on[k]) continue;
      nx[k].rows = o.next_srows[k], nx[k].pos0 = o.next_spos0[k];
      std::copy(o.next_stok[k], o.next_stok[k] + model::ops::kSessTok, nx[k].tok);
    }
    if (!q.empty() && q.front().late) late_rows(q.front());  // the step queued ahead: its rows are known now
    g_t_late.push_back(now_s());
    if (i > 4)  // the step before (done; past the graphs' captures): its GPU time by its rows
      stats.gpu_ms[std::min(prev_R, model::kMaxRows)] += m.gpu_ms(prev_id), ++stats.gpu_n[std::min(prev_R, model::kMaxRows)];
    prev_id = s.id, prev_R = o.step_rows;
    ++stats.steps, stats.rows += o.step_rows;
    for (int k = 0; k < n; ++k) {
      if (!s.on[k]) continue;
      DecodeEnd& d = end[k];
      const int p0 = o.sess_pos0[k], sr = o.sess_step_rows[k], er = o.sess_rows[k], r0 = o.sess_row0[k];
      if (sr < 1 || er < 1 || er > sr || r0 < 0 || r0 + sr > o.step_rows)
        die("spec batch: entry %d: rows %d (%d emitted) from row %d of %d", k, sr, er, r0, o.step_rows);
      d.written = std::max(d.written, p0 + std::max(sr, er + kD));
      if (ending[k]) {  // its rows behind its end: their KV counts, their tokens do not
        snaps(k, H[k] + 1);
        finish(k, H[k] + 1);
        continue;
      }
      if (i > 0) {
        ++d.vsteps, d.verified += sr - 1, d.accepted += er - 1;
        d.probability(o, r0, sr - 1);
        ++d.rows_hist[std::min(sr - 1, kD)];
      }
      rate[k].add(c, er, sr);
      d.spec_tok = rate[k].tok, d.spec_ms = rate[k].ms;
      // the tokens it emits, until a stop
      bool stop = false;
      int u = 0;
      for (int j = 0; j < er && !stop; ++j) {
        const int r = r0 + j;
        const int32_t t = o.row[r].token;
        if (on) on(k, o, r);
        ++d.sampled, ++u, ++stats.emitted;
        toks[k].push_back(t);
        if (o.ctl & (1u << e[k].slot)) stop = true, d.why = kEndCancel;
        for (int q2 = 0; !stop && q2 < e[k].n_stop; ++q2)
          if (t == e[k].stop[q2]) stop = true, d.why = kEndStop;
        if (!stop && d.sampled >= e[k].max_new) stop = true, d.why = kEndLength;
      }
      if (!stop && (o.ctl & kCtlPause)) stop = true, d.why = kEndPause;
      if (!stop && nx[k].pos0 >= pos_end[k]) stop = true, d.why = kEndContext;
      // positions p0 .. F hold their final KV; the history now ends at H
      F[k] = p0 + er - 1, H[k] = p0 + u;
      if (!stop) {
        snaps(k, H[k]);
        continue;
      }
      ending[k] = 1;
      if (!q.empty() && q.front().on[k]) {  // its rows in the step queued still run: the KV of its last token too
        pend[k] = 1;
      } else if (H[k] > F[k] && H[k] <= feed[k] && d.why != kEndPause) {  // the next step launched feeds its last token
        pend[k] = 0;
      } else {
        const int kv_end = H[k] <= F[k] ? H[k] + 1 : F[k] + 1;
        snaps(k, kv_end);
        finish(k, kv_end);
      }
    }
    if ((c.flags & kBatchSolo) && n > 1) {  // one entry left decoding: it pauses, to go on alone
      int left = 0, k1 = -1;
      for (int k = 0; k < n; ++k)
        if (live[k] && !ending[k]) ++left, k1 = k;
      if (left == 1) {
        end[k1].why = kEndPause;
        ending[k1] = 1;
        if (!q.empty() && q.front().on[k1]) {
          pend[k1] = 1;
        } else {
          const int kv_end = H[k1] <= F[k1] ? H[k1] + 1 : F[k1] + 1;
          snaps(k1, kv_end);
          finish(k1, kv_end);
        }
      }
    }
    bool any = false;
    for (int k = 0; k < n; ++k) any |= on_now(k);
    if (q.empty() && any) {  // not queued ahead: the rows are known now, its lookups go at once
      launch(false, false);
      late_rows(q.back());
    }
    if (c.ahead && q.size() == 1 && fits()) launch(false, true);
    g_t_launch.push_back(now_s());
  }
  m.use_slot(slot0);
  if (st) *st = stats;
  return end;
}

void follow(model::Model& m, comm::Ring& ring, Watchdog& wd, int rank) {
  bool prof = false;
  int32_t sampled[model::kMaxRows];
  for (;;) {
    const Cmd c = bcast(ring, Cmd{});
    if (c.op == kCmdStop) break;
    NVTX_RANGE("cmd %d pos0 %d rows %d n %d", c.op, c.pos0, c.rows, c.n);
    if ((c.flags & model::kFlagProfile) && !prof) {
      CUDA_CHECK(cudaProfilerStart());
      prof = true;
    }
    apply_ops(m, c);         // first, whatever the command
    (void)m.state_evicted();  // rank 0's server keeps track of the sessions
    if (c.op == kCmdStep) {
      wd.started = now_s();
      m.step(to_ctl(c), sampled, nullptr);
      wd.started = 0;
    } else if (c.op == kCmdDecode) {
      decode_loop(m, c, wd, nullptr);
    } else if (c.op == kCmdBatch || c.op == kCmdSpecBatch) {
      std::vector<BatchEntry> e(c.n);
      bcast_payload(ring, e.data(), e.size() * sizeof(BatchEntry), rank);
      if (c.op == kCmdSpecBatch) spec_batch_loop(m, c, e, wd, nullptr);
      else batch_loop(m, c, e, wd, nullptr);
    } else if (c.op == kCmdBatchTf) {
      std::vector<BatchEntry> e;
      std::vector<std::vector<int32_t>> t;
      bcast_tf(ring, c, e, t, rank);
      batch_tf_loop(m, c, e, t, wd, nullptr);
    } else if (c.op == kCmdTokens) {
      std::vector<int32_t> t(c.n);
      bcast_payload(ring, t.data(), t.size() * 4, rank);
      m.set_tokens(c.pos0, c.n, t.data());
    } else if (c.op == kCmdPrefill) {
      wd.started = now_s();
      const uint64_t n = run_prefill(m, c);
      if (n != ~0ull) m.wait(n);
      m.drain();
      wd.started = 0;
    } else if (c.op == kCmdRows) {
      wd.started = now_s();
      m.wait(run_rows(m, c));
      wd.started = 0;
    } else if (c.op != kCmdState) {
      die("rank %d: unknown command %d", rank, c.op);
    }
  }
  if (prof) CUDA_CHECK(cudaProfilerStop());
}

}  // namespace srv
