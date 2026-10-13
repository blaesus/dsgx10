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
      batch_loop(m, e, wd, nullptr);
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
