// Rank 0 as a server (serve.h, docs/serving.md, docs/protocol.md).
//
// Threads: the reader takes the front-end's frames (a new connection replaces the old one, whose requests then
// count as cancelled); the writer sends HELLO / TOKENS / DONE, so the engine never blocks on the socket; the engine
// (the caller's thread) runs the requests, every model call mirrored on the other ranks by a command (cmd.h).
//
// Sessions. A session is a token history and the caches of its positions on every rank: a run of pages of the
// paged cache pool (Options::page positions a page; position p of a session whose run starts at page s is pool
// position s * page + p), or pages of the disk tier when spilled, or nothing. GLM's state at position q is exactly
// the cache entries below q (no windows, no snapshots, no carries), so a request continues a session from any
// q <= the longest common prefix of the session's history and its prompt, bit for bit: q = min(LCP, n - 1) (the
// last prompt position runs again for its logits); entries from q on are rewritten as the prefill passes them,
// and no row reads an entry beyond its own position. A request's prompt positions always go through the prefill
// path (chunking changes no bit, a one-row chunk included), its generated ones through decode steps, so a request
// sent twice gives the same tokens. A fork (a new session with a base) copies the base's first q positions.
//
// The pool's pages are rank 0's to hand out, the same on every rank by command: a session gets a contiguous run
// (its needs plus `grow` positions); one that outgrows it extends in place, or moves (kOpCopy) to a free run, after
// compacting the pool if the free pages are scattered, after evicting the least recently used idle sessions (to
// the disk tier when there is one: kCmdSpill, else dropped). A session in flight is never evicted.
//
// Scheduling (the DeepSeek engine's B1 and B2): a request's prompt is prefilled at once, pausing the requests that
// decode (rank 0's control word, SampleOut::ctl, stops every rank after the same step), and its first token sent;
// then it decodes with the others, up to `slots` of them a row each per step; while more decode, they take turns of
// `slice` tokens. A request whose session is in flight waits for it; one that cannot get pages while others are in
// flight waits for one of them to end.
//
// Kept sessions: a planned stop moves the pool's sessions to the disk tier and every rank indexes its file
// (Disk::keep); this server writes their histories beside it (keep). The next start whose ranks all reused those
// files reads them back (load_kept), and the front-end's session ids find their sessions as before the stop.
#include "server/serve.h"

#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "common/check.h"
#include "server/keepfile.h"
#include "server/protocol.h"

namespace srv {
namespace {

constexpr uint32_t kMaxFrame = 64u << 20;
constexpr int kIoCpu = 12;

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

bool read_all(int fd, void* p, size_t n) {
  auto* b = static_cast<char*>(p);
  while (n > 0) {
    const ssize_t r = read(fd, b, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    b += r;
    n -= size_t(r);
  }
  return true;
}

bool write_all(int fd, const void* p, size_t n) {
  auto* b = static_cast<const char*>(p);
  while (n > 0) {
    const ssize_t r = write(fd, b, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    b += r;
    n -= size_t(r);
  }
  return true;
}

uint64_t us(double s) { return s > 0 ? uint64_t(s * 1e6) : 0; }
double unix_now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

// One front-end connection. Once it is closed or replaced its requests count as cancelled and frames for it are
// dropped; the descriptor closes with the last reference.
struct Conn {
  const int fd;
  std::atomic<bool> dead{false};
  std::mutex mu;
  std::set<uint64_t> cancel;  // CANCELs received (ids are per connection)
  explicit Conn(int f) : fd(f) {}
  ~Conn() { close(fd); }
  void kill() {
    if (!dead.exchange(true)) shutdown(fd, SHUT_RDWR);
  }
};
using ConnP = std::shared_ptr<Conn>;

struct Request {
  ConnP conn;
  fe::SubmitHeader h{};
  std::vector<int32_t> stop, prompt;
  double t_recv = 0;
};

struct Event {
  bool release = false;
  uint64_t session = 0;  // release
  Request rq;            // submit
};

struct Frame {
  std::vector<char> buf;
  explicit Frame(uint16_t type) : buf(sizeof(fe::FrameHeader)) {
    const fe::FrameHeader h{0, type, 0};
    std::memcpy(buf.data(), &h, sizeof h);
  }
  template <class T>
  Frame& add(const T* p, size_t n = 1) {
    const char* c = reinterpret_cast<const char*>(p);
    buf.insert(buf.end(), c, c + n * sizeof(T));
    return *this;
  }
  std::vector<char> take() {
    const uint32_t size = uint32_t(buf.size() - sizeof(fe::FrameHeader));
    std::memcpy(buf.data(), &size, sizeof size);
    return std::move(buf);
  }
};

// The socket's two threads (the DeepSeek engine's, unchanged but for the version).
class Io {
 public:
  Io(const std::string& path, std::vector<char> hello) : path_(path), hello_(std::move(hello)) {
    lfd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (lfd_ < 0 || path.size() >= sizeof addr.sun_path) die("serve: cannot make the socket %s", path.c_str());
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    unlink(path.c_str());
    const mode_t old = umask(0177);  // the socket is created 0600
    const int rc = bind(lfd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
    umask(old);
    if (rc != 0 || listen(lfd_, 4) != 0) die("serve: cannot listen on %s: %s", path.c_str(), std::strerror(errno));
    chmod(path.c_str(), 0600);
    rd_ = std::thread([this] {
      pin_self(kIoCpu);
      reader();
    });
    wr_ = std::thread([this] {
      pin_self(kIoCpu);
      writer();
    });
  }
  ~Io() {
    stop_ = true;
    wcv_.notify_all();
    rd_.join();
    wr_.join();
    close(lfd_);
    unlink(path_.c_str());
  }
  bool pop(Event& e, int timeout_ms) {
    std::unique_lock<std::mutex> l(mu_);
    if (!cv_.wait_for(l, std::chrono::milliseconds(timeout_ms), [&] { return !ev_.empty(); })) return false;
    e = std::move(ev_.front());
    ev_.pop_front();
    return true;
  }
  bool submit_waiting() {
    std::lock_guard<std::mutex> l(mu_);
    for (const Event& e : ev_)
      if (!e.release && !e.rq.conn->dead) return true;
    return false;
  }
  void send(const ConnP& c, std::vector<char> f) {
    if (c->dead) return;
    {
      std::lock_guard<std::mutex> l(wmu_);
      out_.emplace_back(c, std::move(f));
    }
    wcv_.notify_one();
  }
  static bool cancelled(const Request& r) {
    if (r.conn->dead || g_stop) return true;  // a stopping server ends the request in flight
    std::lock_guard<std::mutex> l(r.conn->mu);
    return r.conn->cancel.count(r.h.req_id) > 0;
  }
  static bool stopping(const Request& r) {  // the server's stop ends r, not its client or its front-end going away
    if (!g_stop || r.conn->dead) return false;
    std::lock_guard<std::mutex> l(r.conn->mu);
    return r.conn->cancel.count(r.h.req_id) == 0;
  }
  static void forget(const Request& r) {
    std::lock_guard<std::mutex> l(r.conn->mu);
    r.conn->cancel.erase(r.h.req_id);
  }

 private:
  void push(Event&& e) {
    {
      std::lock_guard<std::mutex> l(mu_);
      ev_.push_back(std::move(e));
    }
    cv_.notify_one();
  }
  bool frame(const ConnP& c) {
    fe::FrameHeader h;
    if (!read_all(c->fd, &h, sizeof h) || h.size > kMaxFrame) return false;
    std::vector<char> p(h.size);
    if (!p.empty() && !read_all(c->fd, p.data(), p.size())) return false;
    if (h.type == fe::kHello) {
      fe::HelloFe hi;
      if (p.size() < sizeof hi) return false;
      std::memcpy(&hi, p.data(), sizeof hi);
      if (hi.version != fe::kVersion) {
        std::fprintf(stderr, "serve: front-end speaks version %u, not %u\n", hi.version, fe::kVersion);
        return false;
      }
    } else if (h.type == fe::kSubmit) {
      Event e;
      Request& r = e.rq;
      if (p.size() < sizeof r.h) return false;
      std::memcpy(&r.h, p.data(), sizeof r.h);
      if (p.size() != sizeof r.h + 4 * (size_t(r.h.n_stop_ids) + r.h.n_prompt)) return false;
      r.stop.resize(r.h.n_stop_ids);
      r.prompt.resize(r.h.n_prompt);
      const char* a = p.data() + sizeof r.h;
      if (!r.stop.empty()) std::memcpy(r.stop.data(), a, r.stop.size() * 4);
      if (!r.prompt.empty()) std::memcpy(r.prompt.data(), a + r.stop.size() * 4, r.prompt.size() * 4);
      r.conn = c;
      r.t_recv = now_s();
      push(std::move(e));
    } else if (h.type == fe::kCancel) {
      fe::Cancel x;
      if (p.size() < sizeof x) return false;
      std::memcpy(&x, p.data(), sizeof x);
      std::lock_guard<std::mutex> l(c->mu);
      if (c->cancel.size() > 4096) c->cancel.clear();  // ids of requests long gone
      c->cancel.insert(x.req_id);
    } else if (h.type == fe::kRelease) {
      fe::Release x;
      if (p.size() < sizeof x) return false;
      std::memcpy(&x, p.data(), sizeof x);
      Event e;
      e.release = true;
      e.session = x.session_id;
      push(std::move(e));
    }  // other types are skipped
    return true;
  }
  void reader() {
    ConnP cur;
    while (!stop_) {
      pollfd fds[2] = {{lfd_, POLLIN, 0}, {cur ? cur->fd : -1, POLLIN, 0}};
      if (poll(fds, 2, 200) <= 0) continue;
      if (fds[0].revents & POLLIN) {
        const int fd = accept4(lfd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd >= 0) {
          if (cur) cur->kill();  // the new front-end replaces the old one
          cur = std::make_shared<Conn>(fd);
          send(cur, hello_);
          std::printf("serve: front-end connected\n");
          std::fflush(stdout);
        }
        continue;
      }
      if (cur && fds[1].revents && !frame(cur)) {
        cur->kill();
        cur.reset();
        std::printf("serve: front-end disconnected\n");
        std::fflush(stdout);
      }
    }
    if (cur) cur->kill();
  }
  void writer() {
    for (;;) {
      std::pair<ConnP, std::vector<char>> f;
      {
        std::unique_lock<std::mutex> l(wmu_);
        wcv_.wait(l, [&] { return stop_ || !out_.empty(); });
        if (stop_) return;
        f = std::move(out_.front());
        out_.pop_front();
      }
      if (!f.first->dead && !write_all(f.first->fd, f.second.data(), f.second.size())) f.first->kill();
    }
  }

  std::string path_;
  std::vector<char> hello_;
  int lfd_ = -1;
  std::atomic<bool> stop_{false};
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Event> ev_;
  std::mutex wmu_;
  std::condition_variable wcv_;
  std::deque<std::pair<ConnP, std::vector<char>>> out_;
  std::thread rd_, wr_;
};

struct Session {
  std::vector<int32_t> hist;  // the tokens whose caches the session holds: positions [0, size)
  int page = -1, npages = 0;  // its run in the pool (page < 0: none)
  std::vector<int32_t> disk;  // its pages in the disk tier when spilled
  uint64_t used = 0;          // LRU clock
  double t_used = 0;          // unix time of its last request (kept sessions age out by it)
};

// The server's index of a planned stop's kept sessions: a header, then each session: id, LRU clock and time,
// history, disk pages.
constexpr char kSrvMagic[8] = {'G', 'L', 'M', 'S', 'R', 'V', '0', '1'};
struct SrvHeader {
  char magic[8];
  uint64_t gen;
  int64_t written;
  int32_t n_sess, pad;
  uint64_t clock;
  uint64_t body_bytes, body_hash;
};

int lcp(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return (int)i;
}

// A request from its start to its DONE; between its runs it waits in the turns. Its next token to feed is
// out.back(), at position prompt + out - 1.
struct Flight {
  Request rq;
  model::Sampling sp;  // the seed fixed at the start: each run samples what one run would
  std::vector<int32_t> out;
  int q = 0;  // the reused positions (DONE's cached tokens)
  double t_start = 0, t_first = 0, dec_s = 0;
  int decoded = 0, runs = 0, batched = 0;
};

enum Go { kGoDecode, kGoDone, kGoWait };

class Server {
 public:
  Server(model::Model& m, comm::Ring& ring, Watchdog& wd, Disk* disk, const ServeOptions& o, Io& io)
      : m_(m), ring_(ring), wd_(wd), disk_(disk), o_(o), io_(io), P_(m.options().page), own_(m.options().slots, 0),
        down_(disk ? disk->pages() : 0, 0) {
    cap_ = std::min<int>(o.max_ctx - model::kMaxDrafts, (int)std::min<size_t>((size_t)P_ * own_.size(), 1u << 30));
    page_bytes_ = (size_t)P_ * m.cache_pos_bytes();
  }
  int cap() const { return cap_; }

  void loop() {
    while (!g_stop) {
      sweep();
      Event e;
      if (!ready_.empty()) {
        Request rq = std::move(ready_.front());
        ready_.pop_front();
        admit(std::move(rq));
      } else if (io_.pop(e, rot_.empty() ? 250 : 0)) {
        if (e.release) {
          if (flying_.count(e.session)) later_.insert(e.session);
          else release(e.session);
        } else if (!e.rq.conn->dead) {
          admit(std::move(e.rq));
        }
      } else if (!rot_.empty()) {
        run();
      }
    }
    sweep();
    for (std::deque<Request>* d : {&ready_, &held_, &wait_}) {
      for (const Request& rq : *d) done(rq, fe::kCancelled, 0, 0, now_s(), 0), Io::forget(rq);
      d->clear();
    }
    for (Flight& f : rot_) finish(f, fe::kCancelled);
    rot_.clear();
    keep();
    bcast(ring_, Cmd{});  // stop the other ranks
  }

  // The sessions an earlier run kept at its planned stop (keep), when every rank reused that stop's files
  // (o_.kept_gen): their histories from o_.keep_index, which goes once read.
  void load_kept() {
    if (o_.keep_index.empty()) return;
    std::vector<uint8_t> f;
    std::string why;
    const bool got = keepfile::read_private(o_.keep_index, 16ull << 30, &f, &why);
    unlink(o_.keep_index.c_str());
    unlink((o_.keep_index + ".new").c_str());
    if (!got || !disk_) {
      if (!why.empty()) std::printf("serve: kept sessions: none (%s)\n", why.c_str());
      return;
    }
    SrvHeader hd{};
    if (f.size() >= sizeof hd) std::memcpy(&hd, f.data(), sizeof hd);
    if (!o_.kept_gen) {
      std::printf("serve: kept sessions: none (the ranks did not reuse the stop's files)\n");
      return;
    }
    if (f.size() < sizeof hd || std::memcmp(hd.magic, kSrvMagic, sizeof kSrvMagic) != 0 || hd.gen != o_.kept_gen ||
        hd.body_bytes != f.size() - sizeof hd || keepfile::fnv(f.data() + sizeof hd, hd.body_bytes) != hd.body_hash) {
      std::printf("serve: kept sessions: none (the index is not of the stop whose files the ranks reused)\n");
      return;
    }
    keepfile::Reader r{f.data() + sizeof hd, (size_t)hd.body_bytes};
    const double wall = unix_now();
    size_t positions = 0;
    int n = 0, old = 0;
    for (int i = 0; i < hd.n_sess && r.ok; ++i) {
      const uint64_t id = r.get<uint64_t>();
      Session s;
      s.used = r.get<uint64_t>();
      s.t_used = r.get<double>();
      const int nh = r.get<int32_t>(), np = r.get<int32_t>();
      if (!r.ok || nh < 0 || nh > cap_ || np != pages_for(nh)) {
        r.ok = false;
        break;
      }
      s.hist.resize(nh);
      for (int32_t& t : s.hist) t = r.get<int32_t>();
      s.disk.resize(np);
      for (int32_t& p : s.disk) p = r.get<int32_t>();
      bool ok = r.ok && id && !sess_.count(id);
      for (const int32_t t : s.hist) ok = ok && t >= 0 && t < model::kVocabUsed;
      for (const int32_t p : s.disk) ok = ok && p >= 0 && p < (int)down_.size() && down_[p] == 0;
      if (!ok) continue;
      if (wall - s.t_used > o_.keep_age_s) {
        ++old;
        continue;
      }
      for (const int32_t p : s.disk) down_[p] = id;
      clock_ = std::max(clock_, s.used);
      positions += s.hist.size();
      ++n;
      sess_.emplace(id, std::move(s));
    }
    if (!r.ok || r.at != r.n) {
      for (auto& x : down_) x = 0;
      sess_.clear();
      std::printf("serve: kept sessions: none (the index is damaged)\n");
      return;
    }
    clock_ = std::max(clock_, hd.clock);
    std::printf("serve: kept sessions: %d back on disk (%zu positions), %d past their age dropped\n", n, positions, old);
  }

 private:
  int pages_for(int positions) const { return positions <= 0 ? 0 : (positions + P_ - 1) / P_; }
  int free_pages() const { return (int)std::count(own_.begin(), own_.end(), 0ull); }
  int find_run(int need) const {
    int run = 0;
    for (int p = 0; p < (int)own_.size(); ++p) {
      run = own_[p] ? 0 : run + 1;
      if (run == need) return p - need + 1;
    }
    return -1;
  }
  void mark(uint64_t id, int page, int n) {
    for (int p = page; p < page + n; ++p) own_[p] = id;
  }

  // Every rank: c's operations, then c (rank 0's part of c.op is the caller's).
  void issue(Cmd& c) {
    bcast(ring_, c);
    apply_ops(m_, c);
  }
  void ops(const std::vector<Op>& v) {
    for (size_t i = 0; i < v.size(); i += kMaxOps) {
      Cmd c{};
      c.op = kCmdOps;
      c.n_ops = (int)std::min<size_t>(kMaxOps, v.size() - i);
      std::copy(v.begin() + i, v.begin() + i + c.n_ops, c.ops);
      issue(c);
    }
  }

  // The pool's sessions moved down in page order, so that the free pages are one run at the end.
  void compact() {
    std::vector<std::pair<int, uint64_t>> v;
    for (auto& [id, s] : sess_)
      if (s.page >= 0) v.emplace_back(s.page, id);
    std::sort(v.begin(), v.end());
    std::vector<Op> o;
    int target = 0, moved = 0;
    size_t positions = 0;
    for (const auto& [page, id] : v) {
      Session& s = sess_.at(id);
      if (page != target) {
        o.push_back({kOpCopy, target, page, (int)s.hist.size()});
        o.push_back({kOpReset, target, 0, 0});
        s.page = target;
        ++moved, positions += s.hist.size();
      }
      target += s.npages;
    }
    std::fill(own_.begin(), own_.end(), 0);
    for (const auto& [page, id] : v) mark(id, sess_.at(id).page, sess_.at(id).npages);
    const double t0 = now_s();
    ops(o);
    m_.drain();
    std::printf("serve: pool compacted: %d sessions moved (%zu positions) in %.1f ms\n", moved, positions,
                1e3 * (now_s() - t0));
  }

  // Session `id` out of the pool: to the disk tier (making room there by dropping its least recently used
  // sessions), or dropped.
  void evict(uint64_t id) {
    Session& s = sess_.at(id);
    const int need = pages_for((int)s.hist.size());
    const double t0 = now_s();
    if (disk_ && need > 0 && need <= (int)down_.size()) {
      auto dfree = [&] { return (int)std::count(down_.begin(), down_.end(), 0ull); };
      while (dfree() < need) {  // the disk's least recently used sessions go
        auto victim = sess_.end();
        for (auto it = sess_.begin(); it != sess_.end(); ++it)
          if (!it->second.disk.empty() && (victim == sess_.end() || it->second.used < victim->second.used)) victim = it;
        if (victim == sess_.end()) break;
        std::printf("serve: session %016llx dropped from the disk tier to make room\n", (unsigned long long)victim->first);
        drop(victim->first);
      }
      if (dfree() >= need) {
        std::vector<int32_t> pages;
        for (int p = 0; p < (int)down_.size() && (int)pages.size() < need; ++p)
          if (!down_[p]) pages.push_back(p), down_[p] = id;
        Cmd c{};
        c.op = kCmdSpill, c.slot = s.page, c.rows = (int)s.hist.size(), c.n = need;
        issue(c);
        bcast_payload(ring_, pages.data(), pages.size() * 4, 0);
        spill(m_, *disk_, c.slot, c.rows, pages);
        mark(0, s.page, s.npages);
        s.page = -1, s.npages = 0, s.disk = pages;
        std::printf("serve: session %016llx spilled: %zu positions, %.1f MiB, in %.1f ms\n", (unsigned long long)id,
                    s.hist.size(), need * page_bytes_ / 1048576.0, 1e3 * (now_s() - t0));
        return;
      }
    }
    std::printf("serve: session %016llx evicted (%zu positions)\n", (unsigned long long)id, s.hist.size());
    drop(id);
  }
  void drop(uint64_t id) {
    const auto it = sess_.find(id);
    if (it == sess_.end()) return;
    if (it->second.page >= 0) mark(0, it->second.page, it->second.npages);
    for (const int32_t p : it->second.disk) down_[p] = 0;
    sess_.erase(it);
  }
  void release(uint64_t id) { drop(id); }

  // Session `id` given a run of the pool for `positions` positions (+ grow), its caches where they were (a spilled
  // session's read back): false if the pool cannot hold it without evicting a session in flight or in `pinned`.
  bool place(uint64_t id, int positions, const std::set<uint64_t>& pinned) {
    Session& s = sess_.at(id);
    const int need = std::max(1, pages_for(std::min(positions, cap_)));
    if (need > (int)own_.size()) return false;
    for (;;) {
      if (s.page >= 0 && s.npages >= need) return true;
      if (s.page >= 0) {  // extend in place
        bool ok = s.page + need <= (int)own_.size();
        for (int p = s.page + s.npages; ok && p < s.page + need; ++p) ok = own_[p] == 0;
        if (ok) {
          mark(id, s.page + s.npages, need - s.npages);
          s.npages = need;
          return true;
        }
      }
      const int at = find_run(need);
      if (at >= 0) {
        if (s.page >= 0) {  // move
          Cmd c{};
          c.op = kCmdOps;
          c.ops[c.n_ops++] = {kOpCopy, at, s.page, (int)s.hist.size()};
          c.ops[c.n_ops++] = {kOpReset, at, 0, 0};
          issue(c);
          mark(0, s.page, s.npages);
        }
        mark(id, at, need);
        const int was = s.page;
        s.page = at, s.npages = need;
        if (was < 0 && !s.disk.empty()) {  // read back from the disk tier
          const double t0 = now_s();
          Cmd c{};
          c.op = kCmdUnspill, c.slot = at, c.rows = (int)s.hist.size(), c.n = (int)s.disk.size();
          c.ops[c.n_ops++] = {kOpReset, at, 0, 0};
          issue(c);
          bcast_payload(ring_, s.disk.data(), s.disk.size() * 4, 0);
          unspill(m_, *disk_, at, c.rows, s.disk);
          for (const int32_t p : s.disk) down_[p] = 0;
          std::printf("serve: session %016llx back from disk: %zu positions, %.1f MiB, in %.1f ms\n",
                      (unsigned long long)id, s.hist.size(), s.disk.size() * page_bytes_ / 1048576.0,
                      1e3 * (now_s() - t0));
          s.disk.clear();
        }
        return true;
      }
      if (free_pages() >= need) {  // scattered: compact, then extend or find a run again
        compact();
        continue;
      }
      uint64_t victim = 0, used = UINT64_MAX;
      for (auto& [x, t] : sess_)
        if (t.page >= 0 && x != id && !flying_.count(x) && !pinned.count(x) && t.used < used) victim = x, used = t.used;
      if (!victim) return false;
      evict(victim);
    }
  }

  void tokens(const Request& rq, const model::SampleOut& o, int row) {
    const bool lp = rq.h.flags & fe::kFlagLogprobs;
    const int nt = lp ? std::min<int>({(int)rq.h.n_top_logprobs, model::kTopLogprobs, o.n_top}) : 0;
    const fe::TokensHeader t{rq.h.req_id, 1, (lp ? fe::kTokLogprobs : 0u) | (nt ? fe::kTokTop : 0u)};
    const uint32_t id = (uint32_t)o.row[row].token;
    Frame f(fe::kTokens);
    f.add(&t).add(&id);
    if (lp) f.add(&o.row[row].logprob);
    for (int k = 0; k < nt; ++k) {
      const fe::TopLogprob x{(uint32_t)o.top_id[row][k], o.top_lp[row][k]};
      f.add(&x);
    }
    io_.send(rq.conn, f.take());
  }

  void done(const Request& rq, uint32_t reason, uint32_t cached, uint32_t sent, double t_start, double t_first,
            std::string err = "") {
    if (reason == fe::kCancelled && Io::stopping(rq)) reason = fe::kError, err = "the server is stopping";
    const double t_end = now_s();
    if (t_first <= 0) t_first = t_end;
    fe::DoneHeader d{};
    d.req_id = rq.h.req_id;
    d.reason = reason;
    d.prompt_tokens = rq.h.n_prompt;
    d.cached_tokens = cached;
    d.completion_tokens = sent;
    d.queue_us = us(t_start - rq.t_recv);
    d.prefill_us = us(t_first - t_start);
    d.decode_us = us(t_end - t_first);
    d.error_len = (uint32_t)err.size();
    io_.send(rq.conn, Frame(fe::kDone).add(&d).add(err.data(), err.size()).take());
    if (!err.empty()) std::printf("serve: request %llu: %s\n", (unsigned long long)rq.h.req_id, err.c_str());
  }

  // A submit: held while its session is in flight (a session's requests run one at a time), else started.
  void admit(Request&& rq) {
    if (flying_.count(rq.h.session_id)) return held_.push_back(std::move(rq));
    start(std::move(rq));
  }

  void start(Request&& rq) {
    Flight f;
    f.rq = std::move(rq);
    const fe::SubmitHeader& h = f.rq.h;
    f.t_start = now_s();
    const int n = (int)f.rq.prompt.size();
    std::string err;
    if (n < 1) err = "empty prompt";
    else if (n >= cap_) err = "a prompt of " + std::to_string(n) + " tokens does not fit the context of " + std::to_string(cap_);
    else if (h.session_id == 0) err = "session id 0";
    else if (h.max_new_tokens < 1) err = "max_new_tokens 0";
    else if (f.rq.stop.size() > (size_t)kMaxStop) err = "more than " + std::to_string(kMaxStop) + " stop ids";
    else if (!(h.temperature >= 0.f) || !(h.top_p > 0.f && h.top_p <= 1.f)) err = "bad sampling parameters";
    for (const int32_t t : f.rq.prompt)
      if (err.empty() && (t < 0 || t >= model::kVocabUsed)) err = "token id out of range";
    for (const int32_t t : f.rq.stop)
      if (err.empty() && (t < 0 || t >= model::kVocabUsed)) err = "stop id out of range";
    if (!err.empty() || Io::cancelled(f.rq)) {
      done(f.rq, err.empty() ? fe::kCancelled : fe::kError, 0, 0, f.t_start, 0, err);
      return Io::forget(f.rq);
    }
    f.sp.temperature = h.temperature;
    f.sp.top_p = h.top_p;
    f.sp.top_k = (int32_t)std::min<uint32_t>(h.top_k, 1u << 30);
    f.sp.n_top = (h.flags & fe::kFlagLogprobs) ? (int32_t)std::min<uint32_t>(h.n_top_logprobs, model::kTopLogprobs) : 0;
    f.sp.seed = (h.flags & fe::kFlagSeed) ? h.seed : rng_();
    flying_.insert(h.session_id);
    const Go g = go(f);
    if (g == kGoWait) {
      flying_.erase(h.session_id);
      wait_.push_back(std::move(f.rq));
    } else if (g == kGoDecode) {
      rot_.push_front(std::move(f));  // it decodes in the next run, first
    }
  }

  // Flight f's prompt: its session's pages, the restart point, the prefill of [q, n), the first token sent.
  Go go(Flight& f) {
    const fe::SubmitHeader& h = f.rq.h;
    const std::vector<int32_t>& prompt = f.rq.prompt;
    const int n = (int)prompt.size();
    const uint64_t id = h.session_id;
    const bool existed = sess_.count(id) > 0;
    uint64_t src_id = existed ? id : (h.base_session_id && sess_.count(h.base_session_id) ? h.base_session_id : 0);
    const std::set<uint64_t> pinned = {id, h.base_session_id};
    const int grow = std::min<int>((int)h.max_new_tokens, o_.grow) + 1;
    // the source's caches into the pool first (a spilled session read back), for its common prefix
    if (src_id && sess_.at(src_id).page < 0) {
      Session& src = sess_.at(src_id);
      const int p = std::min(lcp(src.hist, prompt), n - 1);
      src.hist.resize(src_id == id ? p : src.hist.size());  // what follows the prefix is not read back
      if (src.hist.empty() || !place(src_id, src_id == id ? n + grow : (int)src.hist.size(), pinned)) {
        if (src_id == id) {
          drop(id);  // cold
        }
        src_id = 0;
      }
    }
    const int p = src_id ? std::min(lcp(sess_.at(src_id).hist, prompt), n - 1) : 0;
    if (!existed || !sess_.count(id)) sess_.emplace(id, Session{});
    Session& S = sess_.at(id);
    if (src_id == id) S.hist.resize(p);  // what follows q is rewritten
    if (!place(id, n + grow, pinned)) {
      if (src_id != id) drop(id);
      if (!flying_.empty() && flying_.size() > 1) return kGoWait;  // others hold the pool: after one of them ends
      flying_.erase(id);
      done(f.rq, fe::kError, 0, 0, f.t_start, 0,
           "a prompt of " + std::to_string(n) + " tokens does not fit the KV pool (" + std::to_string(own_.size() * P_) +
               " positions)");
      Io::forget(f.rq);
      finish_side(h.session_id, false);
      return kGoDone;
    }
    const int q = p;
    f.q = q;
    std::vector<Op> v;
    if (src_id && src_id != id) {  // a fork: the base's first q positions
      v.push_back({kOpCopy, S.page, sess_.at(src_id).page, q});
      S.hist.assign(prompt.begin(), prompt.begin() + q);
    }
    v.push_back({kOpReset, S.page, 0, 0});
    S.used = ++clock_;
    S.t_used = unix_now();
    std::printf("serve: req %llu session %016llx: %s; reused %d of %d\n", (unsigned long long)h.req_id,
                (unsigned long long)id,
                src_id == id ? "its own history" : src_id ? "a fork of its base" : h.base_session_id ? "a fork of a base this engine does not hold" : existed ? "its history was lost" : "a new session",
                q, n);
    // ---- prefill [q, n) in commands of o_.chunk rows; the last one sampled
    int pos = q;
    model::SampleOut first{};
    bool cancelled = false;
    while (pos < n) {
      if (Io::cancelled(f.rq)) {
        cancelled = true;
        break;
      }
      const int R = std::min(o_.chunk, n - pos);
      const bool last = pos + R == n;
      Cmd c{};
      c.op = kCmdPrefill;
      c.slot = S.page, c.pos0 = pos, c.rows = R, c.n = R;
      c.flags = last ? kPfSample : 0;
      c.sp = f.sp;
      if (pos == q) {
        c.n_ops = (int)v.size();
        std::copy(v.begin(), v.end(), c.ops);
      }
      issue(c);
      std::vector<int32_t> t(prompt.begin() + pos, prompt.begin() + pos + R);
      bcast_payload(ring_, t.data(), t.size() * 4, 0);
      wd_.started = now_s();
      const uint64_t k = m_.prefill(S.page, pos, R, t.data(), last ? &f.sp : nullptr);
      if (k != ~0ull) first = m_.wait(k);
      m_.drain();
      wd_.started = 0;
      S.hist.insert(S.hist.end(), prompt.begin() + pos, prompt.begin() + pos + R);
      pos += R;
    }
    f.t_first = now_s();
    if (cancelled || Io::cancelled(f.rq)) return ended(f, fe::kCancelled);
    const int32_t tok0 = first.row[0].token;
    tokens(f.rq, first, 0);
    f.out.push_back(tok0);
    if (stop_id(f, tok0)) return ended(f, fe::kStop);
    if ((int)f.out.size() >= (int)h.max_new_tokens) return ended(f, fe::kLength);
    return kGoDecode;
  }
  Go ended(Flight& f, uint32_t why) {
    finish(f, why);
    return kGoDone;
  }
  static bool stop_id(const Flight& f, int32_t t) { return std::count(f.rq.stop.begin(), f.rq.stop.end(), t) > 0; }

  static constexpr uint32_t kPaused = ~0u, kTurn = ~0u - 1, kGrow = ~0u - 2;  // a run's ends that are not the request's

  // A run: the requests at the front of the turns, up to o_.slots of them, decode together, a row of each per step
  // (batch_loop). Each runs to its end, or until a new request pauses them all (prefill first), or, while more
  // requests decode than slots, for a turn of o_.slice tokens, or until its session's pages are full (it is given
  // more before its next run). Then they wait again: the paused ones first, in their order, the others last.
  void run() {
    std::vector<Flight> fl;
    std::set<uint64_t> in_run;
    while ((int)fl.size() < o_.slots && !rot_.empty()) {
      Flight f = std::move(rot_.front());
      rot_.pop_front();
      Session& S = sess_.at(f.rq.h.session_id);
      const int pos = (int)(f.rq.prompt.size() + f.out.size()) - 1;
      if ((int)S.hist.size() < pos)
        die("serve: req %llu: its session holds %zu positions, its next token goes at %d",
            (unsigned long long)f.rq.h.req_id, S.hist.size(), pos);
      S.hist.resize(pos);
      const int left = (int)f.rq.h.max_new_tokens - (int)f.out.size();
      if (pos + 1 >= cap_) {
        finish(f, fe::kContext);
        continue;
      }
      if (!place(f.rq.h.session_id, pos + 2 + std::min(left, o_.grow), {})) {
        finish(f, fe::kContext, "the KV pool is full (sessions in flight hold it)");
        continue;
      }
      fl.push_back(std::move(f));
    }
    if (fl.empty()) return;
    const bool turns = o_.slice > 0 && !rot_.empty();
    const int n = (int)fl.size();
    std::vector<BatchEntry> e(n);
    for (int k = 0; k < n; ++k) {
      const Flight& f = fl[k];
      const Session& S = sess_.at(f.rq.h.session_id);
      BatchEntry& x = e[k];
      x.slot = S.page, x.pos0 = (int)S.hist.size(), x.token = f.out.back();
      const int left = (int)f.rq.h.max_new_tokens - (int)f.out.size();
      x.max_new = turns ? std::min(left, o_.slice) : left;
      x.limit = std::min(S.npages * P_, cap_);
      x.n_stop = (int)f.rq.stop.size();
      std::copy(f.rq.stop.begin(), f.rq.stop.end(), x.stop);
      x.sp = f.sp;
    }
    Cmd c{};
    c.op = kCmdBatch;
    c.n = n;
    const double t_dec = now_s();
    issue(c);
    bcast_payload(ring_, e.data(), e.size() * sizeof(BatchEntry), 0);
    uint32_t ctl = 0;
    std::vector<std::vector<int32_t>> samples(n);
    std::vector<char> cancel(n, 0);
    const OnBatch cb = [&](int k, const model::SampleOut& o, int row) {
      Flight& f = fl[k];
      samples[k].push_back(o.row[row].token);
      if (cancel[k]) return;
      if (Io::cancelled(f.rq)) {  // its bit: it ends after the step that reads it
        cancel[k] = 1;
        m_.set_ctl(ctl |= 1u << k);
        return;
      }
      const int32_t t = o.row[row].token;
      tokens(f.rq, o, row);
      f.out.push_back(t);
      const bool ends = stop_id(f, t) || (int)f.out.size() >= (int)f.rq.h.max_new_tokens;
      // a new request pauses the run; one waiting for pages does when this one ends
      if (!(ctl & kCtlPause) && (io_.submit_waiting() || (ends && !wait_.empty()))) m_.set_ctl(ctl |= kCtlPause);
    };
    std::vector<double> st;
    const std::vector<BatchEnd> end = batch_loop(m_, e, wd_, cb, &st);
    if (ctl) m_.set_ctl(0);
    const double dt = now_s() - t_dec;
    std::vector<uint32_t> why(n);
    for (int k = 0; k < n; ++k) {
      Flight& f = fl[k];
      const BatchEnd& x = end[k];
      Session& S = sess_.at(f.rq.h.session_id);
      f.dec_s += dt, f.decoded += x.sampled, ++f.runs;
      if (n > 1) f.batched += x.sampled;
      if (x.fed > 0) {  // the history: the token it started from, and the samples fed after it
        S.hist.push_back(e[k].token);
        S.hist.insert(S.hist.end(), samples[k].begin(), samples[k].begin() + std::max(0, x.fed - 1));
      }
      const int max_new = (int)f.rq.h.max_new_tokens;
      why[k] = cancel[k] || x.why == kEndCancel                ? fe::kCancelled
               : x.why == kEndStop || stop_id(f, f.out.back()) ? fe::kStop
               : (int)f.out.size() >= max_new                  ? fe::kLength
               : x.why == kEndContext                          ? (e[k].limit < cap_ ? kGrow : fe::kContext)
               : x.why == kEndPause                            ? kPaused
                                                               : kTurn;
    }
    std::vector<double> d;
    for (size_t i = 1; i < st.size(); ++i) d.push_back(1e3 * (st[i] - st[i - 1]));
    std::sort(d.begin(), d.end());
    std::printf("serve: run of %d: %zu steps in %.1f ms, %.2f ms a step (median)\n", n, st.size(), 1e3 * dt,
                d.empty() ? 0.0 : d[d.size() / 2]);
    std::vector<Flight> paused;
    for (int k = 0; k < n; ++k) {
      if (why[k] == kPaused || why[k] == kGrow) paused.push_back(std::move(fl[k]));
      else if (why[k] == kTurn) rot_.push_back(std::move(fl[k]));
      else finish(fl[k], why[k]);
    }
    for (auto it = paused.rbegin(); it != paused.rend(); ++it) rot_.push_front(std::move(*it));
  }

  // Requests that their client cancelled, or that the server's stop ends, end now, not at their turn.
  void sweep() {
    for (auto it = wait_.begin(); it != wait_.end();) {
      if (!Io::cancelled(*it)) {
        ++it;
        continue;
      }
      done(*it, fe::kCancelled, 0, 0, now_s(), 0);
      Io::forget(*it);
      it = wait_.erase(it);
    }
    for (size_t i = 0; i < rot_.size();) {
      if (!Io::cancelled(rot_[i].rq)) {
        ++i;
        continue;
      }
      Flight f = std::move(rot_[i]);
      rot_.erase(rot_.begin() + (std::ptrdiff_t)i);
      finish(f, fe::kCancelled);
    }
  }

  // After a request: the session kept or freed, the requests held for it go on, those waiting for pages retry.
  void finish_side(uint64_t sid, bool keep) {
    flying_.erase(sid);
    const bool released = later_.erase(sid) > 0;
    if (!keep || released) release(sid);
    for (auto it = held_.begin(); it != held_.end();)
      if (it->h.session_id == sid) ready_.push_back(std::move(*it)), it = held_.erase(it);
      else ++it;
    while (!wait_.empty()) ready_.push_back(std::move(wait_.front())), wait_.pop_front();
  }

  void finish(Flight& f, uint32_t why, const std::string& err = "") {
    const fe::SubmitHeader& h = f.rq.h;
    done(f.rq, err.empty() ? why : fe::kError, (uint32_t)f.q, (uint32_t)f.out.size(), f.t_start, f.t_first, err);
    Io::forget(f.rq);
    const auto it = sess_.find(h.session_id);
    const double pf = f.t_first > 0 ? f.t_first - f.t_start : 0;
    std::printf("serve: req %llu session %016llx: prompt %u, reused %d, prefill %.1f ms (%.0f tok/s), %zu tokens, "
                "decode %.1f tok/s, %d runs (%d tokens in batches), %s; session %zu positions; pool %d of %zu pages free\n",
                (unsigned long long)h.req_id, (unsigned long long)h.session_id, h.n_prompt, f.q, 1e3 * pf,
                (h.n_prompt - f.q) / std::max(1e-9, pf), f.out.size(), f.decoded / std::max(1e-9, f.dec_s), f.runs,
                f.batched,
                why == fe::kStop ? "stop" : why == fe::kLength ? "length" : why == fe::kCancelled ? "cancelled" : "context",
                it == sess_.end() ? (size_t)0 : it->second.hist.size(), free_pages(), own_.size());
    std::fflush(stdout);
    finish_side(h.session_id, (h.flags & fe::kFlagKeep) != 0);
  }

  // A planned stop (SIGTERM): the sessions kept for the next start (load_kept). Those in the pool are spilled, the
  // most recently used first, as far as keep_bytes goes (the rest dropped); sessions unused for keep_age_s go;
  // then every rank syncs its file and writes its index (kCmdKeep), and this server the sessions' histories.
  void keep() {
    if (!o_.keep_bytes || o_.keep_index.empty() || !disk_) return;
    const double t0 = now_s(), wall = unix_now();
    std::vector<std::pair<uint64_t, uint64_t>> ram;  // (used, id)
    int old = 0, freed = 0, spilled = 0;
    std::vector<uint64_t> gone;
    for (auto& [id, s] : sess_) {
      if (wall - s.t_used > o_.keep_age_s) gone.push_back(id), ++old;
      else if (s.page >= 0) ram.emplace_back(s.used, id);
    }
    for (const uint64_t id : gone) drop(id);
    std::sort(ram.rbegin(), ram.rend());
    size_t bytes = 0;
    for (const auto& [used, id] : ram) {
      if (!sess_.count(id)) continue;  // dropped from the disk to make room
      const size_t b = pages_for((int)sess_.at(id).hist.size()) * page_bytes_;
      if (bytes + b <= o_.keep_bytes && b > 0) {
        evict(id);
        if (sess_.count(id) && !sess_.at(id).disk.empty()) bytes += b, ++spilled;
      } else {
        drop(id), ++freed;
      }
    }
    uint64_t gen = rng_();
    gen += !gen;
    Cmd c{};
    c.op = kCmdKeep;
    c.gen = gen;
    issue(c);
    disk_->keep(gen);
    std::vector<uint8_t> f(sizeof(SrvHeader));
    using keepfile::put;
    int n = 0;
    for (const auto& [id, s] : sess_) {
      if (s.disk.empty()) continue;
      ++n;
      put<uint64_t>(f, id), put<uint64_t>(f, s.used), put<double>(f, s.t_used);
      put<int32_t>(f, (int32_t)s.hist.size()), put<int32_t>(f, (int32_t)s.disk.size());
      for (const int32_t t : s.hist) put<int32_t>(f, t);
      for (const int32_t p : s.disk) put<int32_t>(f, p);
    }
    SrvHeader hd{};
    std::memcpy(hd.magic, kSrvMagic, sizeof kSrvMagic);
    hd.gen = gen, hd.written = (int64_t)unix_now(), hd.n_sess = n, hd.clock = clock_;
    hd.body_bytes = f.size() - sizeof hd;
    hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
    std::memcpy(f.data(), &hd, sizeof hd);
    std::string why;
    const bool ok = keepfile::write_atomic(o_.keep_index, f.data(), f.size(), &why);
    std::printf("serve: kept sessions: %d for the next start (%d spilled now, %.2f GiB; %d dropped past %.1f GiB, %d "
                "unused for %.0f days) in %.2f s%s%s\n",
                ok ? n : 0, spilled, bytes / 1073741824.0, freed, o_.keep_bytes / 1073741824.0, old,
                o_.keep_age_s / 86400, now_s() - t0, ok ? "" : ": NOT WRITTEN, ", why.c_str());
  }

  model::Model& m_;
  comm::Ring& ring_;
  Watchdog& wd_;
  Disk* disk_;
  const ServeOptions& o_;
  Io& io_;
  const int P_;
  int cap_ = 0;
  size_t page_bytes_ = 0;
  std::vector<uint64_t> own_;   // the pool's pages: the session that holds each (0: free)
  std::vector<uint64_t> down_;  // the disk tier's pages
  std::map<uint64_t, Session> sess_;
  std::deque<Flight> rot_;                   // requests between their first token and their end, in turn
  std::deque<Request> held_, ready_, wait_;  // for a session in flight; let go when it ended; waiting for pages
  std::set<uint64_t> flying_, later_;        // the sessions in flight; RELEASEs of them, applied at their end
  uint64_t clock_ = 0;
  std::mt19937_64 rng_{std::random_device{}()};
};

}  // namespace

int serve(model::Model& m, comm::Ring& ring, Watchdog& wd, Disk* disk, const ServeOptions& o) {
  if (m.options().page < 1) die("serve: the model's caches must be paged (Options::page)");
  if (o.slots < 1 || o.slots > model::kMaxSess) die("serve: --slots %d (1 .. %d)", o.slots, model::kMaxSess);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  const int cap = std::min<int>(o.max_ctx - model::kMaxDrafts,
                                 (int)std::min<size_t>((size_t)m.options().page * m.options().slots, 1u << 30));
  fe::HelloEngine hi{};
  hi.version = fe::kVersion;
  hi.max_context = (uint32_t)cap;
  hi.vocab_size = model::kVocabUsed;
  hi.eos_id = (uint32_t)model::kStopIds[0];
  hi.max_sessions = (uint32_t)(m.options().slots + (disk ? disk->pages() : 0));
  hi.name_len = (uint32_t)o.name.size();
  Io io(o.socket, Frame(fe::kHelloEngine).add(&hi).add(o.name.data(), o.name.size()).take());
  std::printf("serve: listening on %s (%s, context %d; pool %d pages of %d positions, %.2f GiB a rank; disk tier %d "
              "pages; %d sessions a step, turns of %d tokens)\n",
              o.socket.c_str(), o.name.c_str(), cap, m.options().slots, m.options().page,
              (double)m.options().slots * m.options().page * m.cache_pos_bytes() / 1073741824.0, disk ? disk->pages() : 0,
              o.slots, o.slice);
  std::fflush(stdout);
  Server s(m, ring, wd, disk, o, io);
  s.load_kept();
  s.loop();
  std::printf("serve: stopped\n");
  return 0;
}

}  // namespace srv
