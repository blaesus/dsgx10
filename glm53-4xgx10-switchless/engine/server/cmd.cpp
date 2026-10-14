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

#include "common/check.h"

namespace srv {

SpecConf g_spec;

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

void apply_ops(model::Model& m, const Cmd& c) {
  for (int i = 0; i < c.n_ops; ++i) {
    const Op& o = c.ops[i];
    if (o.op == kOpCopy) m.cache_copy(o.a, o.b, o.n);
    else if (o.op == kOpReset) m.reset_slot(o.a);
    else die("cmd: unknown operation %d", o.op);
  }
}

void spill(model::Model& m, Disk& d, int slot, int positions, const std::vector<int32_t>& pages) {
  const int P = m.options().page;
  if ((int)pages.size() * P < positions) die("spill: %zu pages for %d positions", pages.size(), positions);
  for (size_t i = 0; i < pages.size(); ++i) {
    const int n = std::clamp(positions - (int)i * P, 0, P);
    std::memset(d.staging(), 0, d.page_bytes());  // the page's hash covers the unused tail too
    m.cache_get(slot + (int)i, 0, n, d.staging(), P);
    d.write_page(pages[i]);
  }
}

void unspill(model::Model& m, Disk& d, int slot, int positions, const std::vector<int32_t>& pages) {
  const int P = m.options().page;
  if ((int)pages.size() * P < positions) die("unspill: %zu pages for %d positions", pages.size(), positions);
  for (size_t i = 0; i < pages.size(); ++i) {
    const int n = std::clamp(positions - (int)i * P, 0, P);
    d.read_page(pages[i]);
    m.cache_put(slot + (int)i, 0, n, d.staging(), P);
  }
}

std::vector<BatchEnd> batch_loop(model::Model& m, const std::vector<BatchEntry>& e, Watchdog& wd, const OnBatch& on,
                                 std::vector<double>* step_t) {
  const int n = (int)e.size();
  std::vector<BatchEnd> end(n);
  // per entry: its steps launched, its row in the last step launched, sampling (live) or with its last row still
  // queued (ending)
  std::vector<int> launched(n, 0), last_row(n, -1);
  std::vector<char> live(n, 1), ending(n, 0);
  auto may_run = [&](int k, int j) { return j <= e[k].max_new && e[k].pos0 + j < e[k].limit; };
  for (int k = 0; k < n; ++k)
    if (!may_run(k, 0)) live[k] = 0, end[k].why = e[k].max_new < 1 ? kEndLength : kEndContext;
  struct Step {
    uint64_t id = 0;
    int R = 0;
    int ent[model::kMaxRows] = {};  // row r's entry
  };
  std::deque<Step> q;  // launched, not yet waited for
  auto launch = [&] {
    model::Batch b;
    Step s;
    for (int k = 0; k < n; ++k) {
      const int j = launched[k];
      if (!live[k] || ending[k] || !may_run(k, j)) {
        last_row[k] = -1;
        continue;
      }
      const int r = b.R++;
      b.tokens[r] = j == 0 ? e[k].token : -1 - last_row[k];  // the step before holds its row
      b.pos[r] = e[k].pos0 + j, b.slot[r] = e[k].slot, b.sess[r] = r, b.sp[r] = e[k].sp;
      s.ent[r] = k;
      last_row[k] = r;
      ++launched[k];
    }
    if (!b.R) return;
    b.nsess = s.R = b.R;
    s.id = m.launch(b);
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
    if (step_t) step_t->push_back(now_s());
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
      if (k < 31 && (o.ctl & (1u << k))) stop(k, kEndCancel);
      else if (std::find(e[k].stop, e[k].stop + e[k].n_stop, t) != e[k].stop + e[k].n_stop) stop(k, kEndStop);
      else if (x.sampled >= e[k].max_new) stop(k, kEndLength);
      else if (o.ctl & kCtlPause) stop(k, kEndPause);           // one that ends here ends, not pauses
      else if (launched[k] <= x.sampled) stop(k, kEndContext);  // no room for its next step
    }
    launch();
  }
  return end;
}

std::vector<BatchEnd> spec_loop(model::Model& m, const std::vector<BatchEntry>& e, Watchdog& wd, const OnBatch& on,
                                const SpecConf& sc, std::vector<double>* step_t) {
  const int kmax = sc.kmax;
  const model::SpecCost& cost = sc.cost;
  const int n = (int)e.size();
  if (kmax < 1 || kmax > m.options().drafts) die("spec_loop: kmax %d, the model drafts %d", kmax, m.options().drafts);
  std::vector<BatchEnd> end(n);
  std::vector<int> pos(n), fed(n, 0);
  std::vector<int32_t> x(n);
  std::vector<std::vector<int32_t>> D(n);
  std::vector<char> live(n, 1);
  std::vector<int> feed, fpos(n, 0);  // entries whose last token a final step feeds, at fpos
  std::vector<int32_t> last(n, 0);
  for (int k = 0; k < n; ++k) {
    pos[k] = e[k].pos0, x[k] = e[k].token;
    D[k].assign(e[k].draft, e[k].draft + std::clamp(e[k].n_draft, 0, model::kMaxDrafts));
    end[k].gate = e[k].gate;
    end[k].gate.kmax = kmax;
    if (e[k].max_new < 1) live[k] = 0, end[k].why = kEndLength;
    else if (pos[k] >= e[k].limit) live[k] = 0, end[k].why = kEndContext;
  }
  for (;;) {
    model::Batch b;
    b.spec = 1;
    int first[model::kMaxSess], kk[model::kMaxSess], ent[model::kMaxSess];
    int ns = 0, left = 0;
    for (int k = 0; k < n; ++k) left += live[k];
    for (int k = 0; k < n; ++k) {
      if (!live[k]) continue;
      --left;
      if (e[k].n_oracle > 0) {  // tests: the earlier request's next tokens, about one in four changed
        const int j0 = end[k].sampled;  // tok_{j0 + 1} follows x
        D[k].clear();
        for (int r = 1; r <= kmax && j0 + r - 1 < e[k].n_oracle; ++r) {
          int32_t t = e[k].oracle[j0 + r - 1];
          if (model::oracle_changed((uint32_t)(j0 + r))) t = (t + 1) % model::kVocabUsed;
          D[k].push_back(t);
        }
      }
      int kj = sc.force > 0 ? std::min(sc.force, (int)D[k].size()) : end[k].gate.choose(cost, (int)D[k].size());
      // up to max_new - sampled: the last row then feeds the request's last token when the drafts before it hold
      kj = std::min({kj, e[k].limit - 1 - pos[k], e[k].max_new - end[k].sampled, model::kMaxRows - b.R - 1 - left});
      kj = std::max(kj, 0);
      first[ns] = b.R, kk[ns] = kj, ent[ns] = k;
      for (int r = 0; r <= kj; ++r) {
        b.tokens[b.R] = r == 0 ? x[k] : D[k][r - 1];
        b.pos[b.R] = pos[k] + r, b.slot[b.R] = e[k].slot, b.sess[b.R] = ns;
        ++b.R;
      }
      b.sp[ns++] = e[k].sp;
    }
    if (!ns) break;
    b.nsess = ns;
    // the next step's drafts: as many as any gate may want, within every entry's pages
    int passes = 1;
    for (int s = 0; s < ns; ++s) passes = std::max(passes, sc.force > 0 ? sc.force : end[ent[s]].gate.choose(cost, kmax));
    for (int s = 0; s < ns; ++s) passes = std::min(passes, e[ent[s]].limit - pos[ent[s]] - kk[s]);
    b.passes = std::max(passes, 1);
    wd.started = now_s();
    const model::SampleOut& o = m.wait(m.launch(b));
    wd.started = 0;
    if (step_t) step_t->push_back(now_s());
    for (int s = 0; s < ns; ++s) {
      const int k = ent[s], c0 = first[s], a = o.accepted[s];
      BatchEnd& z = end[k];
      if (a < 0 || a > kk[s]) die("spec_loop: entry %d accepted %d of %d drafts", k, a, kk[s]);
      z.gate.update(kk[s], a, kk[s] == 0 && !D[k].empty() ? (int)(D[k][0] == o.row[c0].token) : -1);
      ++z.steps, z.verified += kk[s], z.accepted += a;
      const int p0 = pos[k];  // row 0's position; fed == sampled here
      fed[k] += a + 1, pos[k] += a + 1;
      bool ended = false;
      int i = 0;
      for (; i <= a && !ended; ++i) {
        const int32_t t = o.row[c0 + i].token;
        if (on) on(k, o, c0 + i);
        ++z.sampled;
        if (std::find(e[k].stop, e[k].stop + e[k].n_stop, t) != e[k].stop + e[k].n_stop) z.why = kEndStop, ended = true;
        else if (z.sampled >= e[k].max_new) z.why = kEndLength, ended = true;
      }
      if (ended) {  // batch_loop's queued row feeds the last token: here row i (when its draft held) or a step below
        if (i <= a) fed[k] = z.sampled + 1;
        else if (p0 + a + 1 < e[k].limit) feed.push_back(k), last[k] = o.row[c0 + a].token, fpos[k] = p0 + a + 1;
      } else {
        x[k] = o.row[c0 + a].token;
        D[k].assign(o.draft[s], o.draft[s] + o.drafts);
        if (k < 31 && (o.ctl & (1u << k))) {
          z.why = kEndCancel, ended = true;
          if (pos[k] < e[k].limit) feed.push_back(k), last[k] = x[k], fpos[k] = pos[k];
        }
        else if (o.ctl & kCtlPause) z.why = kEndPause, ended = true;
        else if (pos[k] >= e[k].limit) z.why = kEndContext, ended = true;
        if (ended) {
          z.n_draft = (int)D[k].size();
          std::copy(D[k].begin(), D[k].end(), z.draft);
        }
      }
      if (ended) live[k] = 0, z.fed = std::min(fed[k], z.sampled + 1);
    }
  }
  // the entries that ended (stop, length, cancel) on a token no row fed yet: one more step feeds it, as batch_loop's
  // queued row does (its sample unused), so that the session's caches and history are batch_loop's
  if (!feed.empty()) {
    model::Batch b;
    b.spec = 1, b.passes = 1;
    for (const int k : feed) {
      const int r = b.R++;
      b.tokens[r] = last[k], b.pos[r] = fpos[k], b.slot[r] = e[k].slot, b.sess[r] = r, b.sp[r] = e[k].sp;
    }
    b.nsess = b.R;
    wd.started = now_s();
    m.wait(m.launch(b));
    wd.started = 0;
    for (const int k : feed) end[k].fed = end[k].sampled + 1;
  }
  return end;
}

void follow(model::Model& m, comm::Ring& ring, Watchdog& wd, Disk* disk, uint64_t run_id) {
  (void)run_id;
  const int rank = ring.rank();
  for (;;) {
    const Cmd c = bcast(ring, Cmd{});
    if (c.op == kCmdStop) break;
    apply_ops(m, c);  // first, whatever the command
    if (c.op == kCmdPrefill) {
      std::vector<int32_t> t(c.n);
      bcast_payload(ring, t.data(), t.size() * 4, rank);
      wd.started = now_s();
      const uint64_t k = m.prefill(c.slot, c.pos0, c.rows, t.data(), (c.flags & kPfSample) ? &c.sp : nullptr);
      if (k != ~0ull) m.wait(k);
      m.drain();
      wd.started = 0;
    } else if (c.op == kCmdBatch) {
      std::vector<BatchEntry> e(c.n);
      bcast_payload(ring, e.data(), e.size() * sizeof(BatchEntry), rank);
      if (c.flags & kBatchSpec) spec_loop(m, e, wd, nullptr, g_spec);
      else batch_loop(m, e, wd, nullptr);
    } else if (c.op == kCmdSpill || c.op == kCmdUnspill) {
      std::vector<int32_t> p(c.n);
      bcast_payload(ring, p.data(), p.size() * 4, rank);
      if (!disk) die("rank %d: a spill without a disk tier", rank);
      if (c.op == kCmdSpill) spill(m, *disk, c.slot, c.rows, p);
      else unspill(m, *disk, c.slot, c.rows, p);
    } else if (c.op == kCmdKeep) {
      if (disk) disk->keep(c.gen);
    } else if (c.op != kCmdOps) {
      die("rank %d: unknown command %d", rank, c.op);
    }
  }
}

}  // namespace srv
