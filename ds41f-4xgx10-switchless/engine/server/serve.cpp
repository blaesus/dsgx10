// Rank 0 as a server (serve.h, docs/protocol.md).
//
// Threads: the reader takes the front-end's frames (a new connection replaces the old one, whose
// requests then count as cancelled); the writer sends HELLO / TOKENS / DONE, so the engine never
// blocks on the socket; the engine (the caller's thread) runs the requests, every model call mirrored on
// the other ranks by a command (cmd.h).
//
// Scheduling (docs/batching.md, B1 and B2). A request's first run prefills its prompt into a session slot
// and sends its first token; then it decodes with the others, as many as there are slots, a row each
// per step (batch_loop), or alone through the speculative loop (with drafts). A request that comes while
// others decode pauses them (prefill first), if it can have a slot: a free one, one whose session no
// request uses, or, with time-slicing (--slice), the slot of the paused request whose turn comes last.
// While more requests decode than there are slots, they take turns of --slice tokens. One slot is B1:
// one request decodes at a time, and a new one pauses it.
//
// Sessions. Each slot's working state (window rings, compressor carries, compressed caches) belongs to
// one session, live there. Every other kept session lives in the store (state.cu): its compressed
// caches by pages, an end snapshot (the whole ring and the carries at its last position) and window
// snapshots (the ring's last 128 positions) at the end of its last prompt and every snap_every positions,
// thinned further back (keep_snap). A request continues from the longest common prefix p of the session's history and its prompt
// (at most n - 1: the last prompt position runs again for its logits), restarting at the largest
// position q <= p from which the state can be rebuilt exactly:
//   - live: the working ring still holds q - 127 .. q - 1 when q >= E - 128 (E: the history's end),
//     and a ratio-2 entry completing at an odd q finds q - 1's projection in the carry when q - 1 >= E - 16;
//   - the end snapshot: the same about its position;
//   - a window snapshot at s: q = s (s even) or s - 1 (no carry is kept with it, so q is even);
//   - otherwise q = 0.
// The compressed caches are per position and append-only: entries below q are kept as they are (in the
// working buffers, or loaded from the store's pages); entries from q on are rewritten as the prefill
// passes them, and no row reads an entry beyond its own position.
// The ring's entries are exact for positions [max(lo, w - 256), end) and the carries' for [max(clo,
// w - 16), end): lo is where the state's lineage begins (a restored snapshot's first position), w past
// the last position anything wrote (a queued step, rejected drafts, the draft module's block keys).
// The draft module's windows (DSpark) follow the ring: every snapshot and restart carries them.
//
// CED (ServeOptions::ced, docs/ced.md): a turn of at least 128 new positions (n - q >= 128) runs the
// encoder over them and replays the decoder over [n - 128, n) from fresh rings, so it needs only the
// encoder's state at q; a shorter one runs every layer (an exact continuation) and needs the decoder
// layers' rings too. Those are not valid everywhere: window snapshots taken inside a CED prefill hold
// the encoder's rings only (Session::enc_snaps), and the live rings are valid from dec_lo on (from the
// replay's first position after a CED turn; nowhere after a cancelled one). The draft module's windows
// are the decoder's: the replay seeds them, and the encoder's snapshots hold none.
//
// Kept sessions (docs/engine.md): a planned stop moves the sessions to the disk tier and every rank
// indexes its file (state.cu); this server writes their histories beside it (keep). The next start
// whose store reused those files reads them back in (load_kept), and the front-end's session ids find
// their sessions as before the stop.
#include "server/serve.h"
#include "server/vision_prefix.h"
#include "server/vision_files.h"
#include "server/vision_window.h"

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

#include <cuda_profiler_api.h>

#include "common/check.h"
#include "model/nvtx.h"
#include "common/frontend_protocol.h"
#include "common/keepfile.h"

namespace srv {
namespace {

constexpr uint32_t kMaxFrame = 32u << 20;  // protocol 5 carries paths, never BF16 feature payloads
constexpr int kIoCpu = 12;

std::atomic<bool> g_stop{false};
bool g_prof = false;  // rank 0 opened nsys's capture range (--profile-from-req)
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

// One front-end connection. Once it is closed or replaced its requests count as cancelled and frames
// for it are dropped; the descriptor closes with the last reference.
struct Conn {
  const int fd;
  std::atomic<bool> dead{false};
  std::mutex mu;
  std::set<uint64_t> admitted; // protocol req IDs are single-use per connection
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
  std::vector<fe::ImageSpan> images;
  std::vector<VisionFile::Data> image_features;
  double t_recv = 0;
};

struct Event {
  bool release = false;
  uint64_t session = 0;  // release
  Request rq;            // submit
};

// A frame: header, fixed struct, then arrays.
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
  // a SUBMIT waits to be taken (time-slicing pauses the decode in flight for it)
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
  // the server's stop ends r, not its client or its front-end going away
  static bool stopping(const Request& r) {
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

  // One frame from c: false when the connection is to be closed.
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
      const size_t text_end = sizeof r.h + 4 * (size_t(r.h.n_stop_ids) + r.h.n_prompt);
      if (p.size() < text_end || r.h.n_images > r.h.n_prompt) return false;
      size_t offset = text_end;
      uint64_t end = 0;
      uint64_t feature_bytes = 0, path_bytes = 0;
      std::string image_error;
      if (r.h.n_images > fe::kMaxImageSpans) image_error = "image descriptor budget exceeded (maximum 4096)";
      struct Opened { fe::ImageSpan image; VisionFile::Data data; };
      std::map<std::string, Opened> opened;
      if (c->admitted.count(r.h.req_id)) return false;
      for (uint32_t i = 0; i < r.h.n_images; ++i) {
        if (p.size() - offset < sizeof(fe::ImageSpan)) return false;
        fe::ImageSpan image{};
        std::memcpy(&image, p.data() + offset, sizeof image);
        offset += sizeof image;
        if (!image.rows || image.rows > 1024 || image.start < end ||
            uint64_t(image.start) + image.rows > r.h.n_prompt) return false;
        end = uint64_t(image.start) + image.rows;
        const size_t elements = size_t(image.rows) * model::kDim;
        feature_bytes += elements * 2;
        if (p.size()-offset < 4) return false;
        uint32_t bytes; std::memcpy(&bytes,p.data()+offset,4); offset += 4;
        if (bytes == 0 || bytes > fe::kMaxFeaturePathBytes || p.size()-offset < bytes) return false;
        const std::string path(p.data()+offset, bytes);
        VisionFile::Data feature;
        if (image_error.empty()) {
          auto prior = opened.find(path);
          if (prior != opened.end()) {
            if (prior->second.image.rows != image.rows || prior->second.image.digest != image.digest)
              image_error = "one image feature path has inconsistent identity or shape";
            else feature = prior->second.data;
          } else {
            feature = VisionFile::open(image, path, image_error);
            if (feature) opened.emplace(path, Opened{image, feature});
          }
        }
        r.image_features.push_back(std::move(feature));
        offset += bytes; path_bytes += bytes;
        r.images.push_back(image);
      }
      if (offset != p.size()) return false;
      r.stop.resize(r.h.n_stop_ids);
      r.prompt.resize(r.h.n_prompt);
      const char* a = p.data() + sizeof r.h;
      if (!r.stop.empty()) std::memcpy(r.stop.data(), a, r.stop.size() * 4);
      if (!r.prompt.empty()) std::memcpy(r.prompt.data(), a + r.stop.size() * 4, r.prompt.size() * 4);
      if (!image_error.empty()) {
        // Only this request fails, before queue/GPU/session mutation. Never kill the connection.
        c->admitted.insert(r.h.req_id);
        fe::DoneHeader d{};
        d.req_id = r.h.req_id; d.reason = fe::kError; d.error_len = image_error.size();
        send(c,Frame(fe::kDone).add(&d).add(image_error.data(),image_error.size()).take());
        return true;
      }
      c->admitted.insert(r.h.req_id);
      std::printf("serve: req %llu vision wire: total %zu, inline bytes 0, logical feature bytes %llu, "
                  "path bytes %llu, unique open files %zu\n",
                  (unsigned long long)r.h.req_id,r.images.size(),(unsigned long long)feature_bytes,
                  (unsigned long long)path_bytes,opened.size());
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
        }
        continue;
      }
      if (cur && fds[1].revents && !frame(cur)) {
        cur->kill();
        cur.reset();
        std::printf("serve: front-end disconnected\n");
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
  int32_t prefill_policy = fe::kLegacyPrefix;
  std::vector<fe::ImageSpan> images;
  int h = 0;                   // the store's handle, the same on every rank
  int slot = -1;               // the slot whose working state is this session's (-1: none, the store's only)
  std::vector<int32_t> hist;   // the tokens whose state the session holds: positions [0, size)
  uint64_t used = 0;           // LRU clock
  int pe_snap = -1;            // the snapshot at its last prompt's end (when not a multiple of snap_every)
  int lo = 0, clo = 0, w = 0;  // the live state's exact ranges (the comment at the top)
  int end_lo = 0, end_clo = 0, end_w = 0;  // the end snapshot's
  // CED: the decoder layers' rings hold valid positions from dec_lo on (kNever: none), the end
  // snapshot's from end_dec_lo on; window snapshots holding the encoder layers' rings only
  int dec_lo = 0, end_dec_lo = 0;
  std::vector<int> enc_snaps;
  double t_used = 0;  // unix time of its last request (kept sessions age out by it)
};

// The server's index of a planned stop's kept sessions (keep(); keepfile.h: 0600 in a 0700 directory,
// whole or not at all): a header, then each session: id, handle, restart ranges, LRU clock and time,
// history, encoder-only snapshots.
constexpr char kSrvMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'T', '5'};
struct SrvHeader {
  char magic[8];
  uint64_t gen;  // the store's (state.cu): both indexes come from the same stop
  int64_t written;
  int32_t n_sess, next_h;
  uint64_t clock;
  uint64_t body_bytes, body_hash;
};

constexpr int kNever = 1 << 30;       // Session::dec_lo of rings that are valid nowhere

// The window snapshots a session keeps, by their distance d from its history's end E: every one (each
// `every` positions) while d < 8 * every; further back those at multiples of every * 2^k, k = 1, 2, ...
// as d passes 8, 16, 32, ... times every. That is four per doubling of d: a restart d positions before
// the end re-prefills at most d / 3 positions from the snapshot below it (at most a third of what the
// client dropped), and a session holds O(log E) snapshots (28 at 250k positions, not 245). As E grows
// the multiples coarsen, so a snapshot dropped once is never wanted again.
bool keep_snap(int s, int E, int every) {
  const int64_t d = (int64_t)E - s, dense = 8ll * every;
  if (d < dense) return true;
  int64_t g = 2ll * every;
  for (int64_t lim = 2 * dense; d >= lim; lim *= 2) g *= 2;
  return s % g == 0;
}

enum { kCold, kLive, kEnd, kSnap };
const char* const kHow[] = {"cold", "live", "end", "snapshot"};
struct Plan {
  int q = 0, how = kCold, snap = -1;
};

int lcp(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return (int)i;
}

// A run's time split for the log: host ms from its start to the prompt's tokens and state operations
// issued (and their GPU time), the chunks' GPU time (Model::prefill_split; rows: steps of run_rows), host
// ms from the first token to the decode command, and from it to the first sample.
struct Turn {
  double ops_ms = 0, ops_gpu_ms = 0, rows_ms = 0, tail_ms = 0, dec1_ms = -1;
  float pf[4] = {};
  int chunks = 0;
  bool pending = false;  // the last chunk's split, read once the first token is out
  void add_split(const model::Model& m) {
    float f[4];
    m.prefill_split(f);
    for (int i = 0; i < 4; ++i) pf[i] += f[i];
  }
};

// A request from its start to its DONE. Between its runs (decode with the others, or alone) it waits in
// the turns; its next token to feed is out.back(), at position prompt + out - 1 (fed again if its KV is
// there already: the same values). A request whose session lost its slot resumes from the store: the
// prompt and the tokens sent so far, one row.
struct Flight {
  Request rq;
  model::Sampling sp;        // the seed fixed at the start: each run samples what one run would
  std::vector<int32_t> out;  // the tokens sent
  int q = 0;                 // the first run's restart point (DONE's cached tokens), its plan and split
  Plan pl;
  Turn tu;
  double t_start = 0, t_first = 0, t_paused = 0;
  double paused_s = 0, resume_s = 0, dec_s = 0;  // waiting in turn, resuming (to each first token), decoding
  int slices = 0, decoded = 0, batched = 0;  // runs; tokens decoded, of them in batches
  DecodeEnd de;                      // the slices' drafts, summed
  double spec_tok = 1, spec_ms = 0;  // the drafts' rate (SpecRate) from slice to slice
  std::string error;  // a bounded feature-file failure ends this flight, not the engine
  uint64_t vision_disk_bytes = 0, vision_peak_chunk_bytes = 0, vision_file_reads = 0;
};

// b's drafts and steps added to a's (a request's decode slices)
void add(DecodeEnd& a, const DecodeEnd& b) {
  a.vsteps += b.vsteps, a.verified += b.verified, a.accepted += b.accepted;
  for (int r = 0; r <= model::ops::kDrafts; ++r) a.rows_hist[r] += b.rows_hist[r];
  for (int r = 0; r <= model::kMaxRows; ++r) a.gpu_ms[r] += b.gpu_ms[r], a.gpu_n[r] += b.gpu_n[r];
  if (b.gate_n) {
    a.gate_min = a.gate_n ? std::min(a.gate_min, b.gate_min) : b.gate_min;
    a.gate_max = a.gate_n ? std::max(a.gate_max, b.gate_max) : b.gate_max;
    a.gate_sum += b.gate_sum, a.gate_n += b.gate_n;
  }
}

class Server {
 public:
  Server(model::Model& m, comm::Ring& ring, Watchdog& wd, const ServeOptions& o, Io& io, size_t budget)
      : m_(m), ring_(ring), wd_(wd), o_(o), io_(io), budget_(budget), slot_(m.slots(), 0) {}

  // Requests let go after their session's last one ended first, then those that waited for a slot (when
  // one is there), then the front-end's frames, then a run of the requests decoding (run).
  void loop() {
    while (!g_stop) {
      sweep();
      Event e;
      if (!ready_.empty()) {
        Request rq = std::move(ready_.front());
        ready_.pop_front();
        admit(std::move(rq));
      } else if (!wait_.empty() && pick_slot({}) >= 0) {
        Request rq = std::move(wait_.front());
        wait_.pop_front();
        admit(std::move(rq));
      } else if (io_.pop(e, rot_.empty() ? 250 : 0)) {
        if (e.release) {
          if (flying_.count(e.session)) later_.insert(e.session);
          else release(e.session);
        } else if (!e.rq.conn->dead) {  // a gone front-end's requests need no answer
          admit(std::move(e.rq));
        }
      } else if (!rot_.empty()) {
        run();
      }
    }
    // the stop ends what is paused or held with an error (Io::stopping): "the server is stopping"
    sweep();
    for (std::deque<Request>* d : {&ready_, &held_, &wait_}) {
      for (const Request& rq : *d) done(rq, fe::kCancelled, 0, 0, now_s(), 0), Io::forget(rq);
      d->clear();
    }
    keep();
    bcast(ring_, Cmd{});  // stop the other ranks
  }

  // The sessions an earlier run kept at its planned stop (keep), when the store reused that stop's
  // files (Model::state_kept: the same generation): their histories and ranges from o_.keep_index, which
  // goes once read, so that a crash from now on starts clean. Sessions it names that the store does not
  // hold on disk, or unused for keep_age_s, are dropped, and so are the store's sessions it does not name.
  void load_kept() {
    if (o_.keep_index.empty()) return;
    const double t0 = now_s();
    std::vector<uint8_t> f;
    std::string why;
    const bool got = keepfile::read_private(o_.keep_index, 16ull << 30, &f, &why);
    unlink(o_.keep_index.c_str());
    unlink((o_.keep_index + ".new").c_str());
    const uint64_t gen = m_.state_kept();
    std::map<uint64_t, Session> kept;
    SrvHeader hd{};
    if (got && f.size() >= sizeof hd) std::memcpy(&hd, f.data(), sizeof hd);
    if (!got) {
      // why: "" when there is none
    } else if (!gen) {
      why = "the store did not reuse its files";
    } else if (f.size() < sizeof hd || std::memcmp(hd.magic, kSrvMagic, sizeof kSrvMagic) != 0 || hd.gen != gen ||
               hd.body_bytes != f.size() - sizeof hd ||
               keepfile::fnv(f.data() + sizeof hd, hd.body_bytes) != hd.body_hash || hd.n_sess < 0) {
      why = "the index is not of the stop whose files the store reused";
    } else {
      keepfile::Reader r{f.data() + sizeof hd, (size_t)hd.body_bytes};
      const double wall = unix_now();
      for (int i = 0; i < hd.n_sess && r.ok; ++i) {
        const uint64_t id = r.get<uint64_t>();
        Session s;
        s.prefill_policy = r.get<int32_t>();
        if (s.prefill_policy < fe::kLegacyPrefix || s.prefill_policy > fe::kTailPrefix) r.ok = false;
        const uint32_t ni = r.get<uint32_t>();
        if (ni > uint32_t(o_.max_ctx) || ni > (r.n - r.at) / sizeof(fe::ImageSpan)) { r.ok = false; break; }
        for (uint32_t k = 0; k < ni; ++k) s.images.push_back(r.get<fe::ImageSpan>());
        s.h = r.get<int32_t>(), s.pe_snap = r.get<int32_t>();
        s.end_lo = r.get<int32_t>(), s.end_clo = r.get<int32_t>(), s.end_w = r.get<int32_t>(), s.end_dec_lo = r.get<int32_t>();
        s.used = r.get<uint64_t>();
        s.t_used = r.get<double>();
        const int nh = r.get<int32_t>(), ne = r.get<int32_t>();
        if (!r.ok || nh < 0 || nh > o_.max_ctx || ne < 0 || ne > 4096) {
          r.ok = false;
          break;
        }
        s.hist.resize(nh);
        for (int32_t& t : s.hist) t = r.get<int32_t>();
        s.enc_snaps.resize(ne);
        for (int& e : s.enc_snaps) e = r.get<int32_t>();
        bool ok = r.ok && id && !kept.count(id) && s.h > 0 && wall - s.t_used <= o_.keep_age_s;
        for (const int32_t t : s.hist) ok = ok && t >= 0 && t < model::kVocab;
        const model::Model::StateInfo si = m_.state_info(s.h);
        ok = ok && si.on_disk && si.saved <= nh && si.end_pos <= nh;
        for (const auto& [k, x] : kept) ok = ok && x.h != s.h;
        s.lo = s.clo = s.w = s.dec_lo = 0;  // nothing is live
        if (ok) kept.emplace(id, std::move(s));
      }
      if (!r.ok || r.at != r.n) {
        kept.clear();
        why = "the index is damaged";
      }
    }
    // the store's sessions that no kept session names go
    std::vector<StateOp> v;
    std::set<int> named;
    for (const auto& [id, s] : kept) named.insert(s.h);
    int top = 0;
    for (const int h : m_.state_sessions()) {
      top = std::max(top, h);
      if (!named.count(h)) v.push_back({kStFree, h, 0, 0});
    }
    if (!v.empty()) state(v);
    next_h_ = std::max({next_h_, top + 1, kept.empty() ? 0 : hd.next_h});
    for (auto& [id, s] : kept) clock_ = std::max(clock_, s.used);
    size_t positions = 0;
    for (const auto& [id, s] : kept) positions += s.hist.size();
    if (!kept.empty()) clock_ = std::max<uint64_t>(clock_, hd.clock);
    sess_.insert(std::make_move_iterator(kept.begin()), std::make_move_iterator(kept.end()));
    if (!why.empty() || !sess_.empty() || !v.empty())
      std::printf("serve: kept sessions: %zu back (%zu positions, %.2f GiB on disk), %zu of the store's dropped, in %.1f ms%s%s\n",
                  sess_.size(), positions, m_.state_disk_bytes() / 1073741824.0, v.size(), 1e3 * (now_s() - t0),
                  why.empty() ? "" : "; ", why.c_str());
  }

 private:
  // Every rank: c's state operations, then c (rank 0's part of c.op is the caller's). c carries the
  // sessions live in the slots, which the store keeps (pinned).
  void issue(Cmd& c) {
    for (size_t s = 0; s < slot_.size(); ++s) {
      const auto it = slot_[s] ? sess_.find(slot_[s]) : sess_.end();
      c.live[s] = it == sess_.end() ? 0 : it->second.h;
    }
    bcast(ring_, c);
    apply_ops(m_, c);
  }
  // State operations on every rank, in commands of kMaxOps.
  void state(const std::vector<StateOp>& v) {
    for (size_t i = 0; i < v.size(); i += kMaxOps) {
      Cmd c{};
      c.op = kCmdState;
      c.n_ops = (int)std::min<size_t>(kMaxOps, v.size() - i);
      std::copy(v.begin() + i, v.begin() + i + c.n_ops, c.ops);
      issue(c);
    }
  }
  // A planned stop (SIGTERM): the sessions kept for the next start (load_kept; docs/engine.md). The slots'
  // sessions' states go to the store; sessions unused for keep_age_s are freed; those in device memory
  // are spilled to the disk tier, the most recently used first, as far as keep_bytes goes (the rest
  // freed); then every rank syncs its file and writes its index (kStKeep), and this server the kept
  // sessions' histories with the same generation. Spilling may drop the file's least recently used
  // sessions to make room (forget_evicted).
  void keep() {
    if (!o_.keep_bytes || o_.keep_index.empty()) return;
    const double t0 = now_s(), wall = unix_now();
    for (int s = 0; s < (int)slot_.size(); ++s) persist_slot(s, true);
    std::vector<StateOp> v;
    std::vector<std::pair<uint64_t, uint64_t>> ram;  // (used, id): the sessions in device memory
    int old = 0, before = 0;
    for (auto it = sess_.begin(); it != sess_.end();) {
      if (wall - it->second.t_used > o_.keep_age_s) {
        v.push_back({kStFree, it->second.h, 0, 0});
        it = sess_.erase(it);
        ++old;
        continue;
      }
      if (m_.state_info(it->second.h).on_disk) ++before;
      else ram.emplace_back(it->second.used, it->first);
      ++it;
    }
    std::sort(ram.rbegin(), ram.rend());
    size_t bytes = 0;
    int spilled = 0, freed = 0;
    for (const auto& [used, id] : ram) {
      const Session& s = sess_.at(id);
      const size_t b = m_.state_info(s.h).bytes;
      if (bytes + b <= o_.keep_bytes) {
        v.push_back({kStSpill, s.h, 0, 0});
        bytes += b, ++spilled;
      } else {
        v.push_back({kStFree, s.h, 0, 0});
        sess_.erase(id);
        ++freed;
      }
    }
    state(v);
    forget_evicted();  // dropped from the file to make room
    uint64_t gen = rng_();
    gen += !gen;
    state({{kStKeep, -1, (int32_t)(uint32_t)gen, (int32_t)(uint32_t)(gen >> 32)}});
    const double t1 = now_s();
    std::vector<uint8_t> f(sizeof(SrvHeader));
    using keepfile::put;
    int n = 0;
    for (const auto& [id, s] : sess_) {
      if (!m_.state_info(s.h).on_disk) continue;
      ++n;
      put<uint64_t>(f, id);
      put<int32_t>(f, s.prefill_policy);
      put<uint32_t>(f, (uint32_t)s.images.size());
      for (const auto& image : s.images) put<fe::ImageSpan>(f, image);
      put<int32_t>(f, s.h), put<int32_t>(f, s.pe_snap);
      put<int32_t>(f, s.end_lo), put<int32_t>(f, s.end_clo), put<int32_t>(f, s.end_w), put<int32_t>(f, s.end_dec_lo);
      put<uint64_t>(f, s.used), put<double>(f, s.t_used);
      put<int32_t>(f, (int32_t)s.hist.size()), put<int32_t>(f, (int32_t)s.enc_snaps.size());
      for (const int32_t t : s.hist) put<int32_t>(f, t);
      for (const int e : s.enc_snaps) put<int32_t>(f, e);
    }
    SrvHeader hd{};
    std::memcpy(hd.magic, kSrvMagic, sizeof kSrvMagic);
    hd.gen = gen, hd.written = (int64_t)unix_now(), hd.n_sess = n, hd.next_h = next_h_, hd.clock = clock_;
    hd.body_bytes = f.size() - sizeof hd;
    hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
    std::memcpy(f.data(), &hd, sizeof hd);
    std::string why;
    const bool ok = keepfile::write_atomic(o_.keep_index, f.data(), f.size(), &why);
    std::printf("serve: kept sessions: %d for the next start (%d on disk before, %d spilled now: %.2f GiB; %d freed past "
                "%.1f GiB, %d unused for %.0f days), the ranks' files synced and indexed in %.2f s; histories %.1f KiB "
                "in %.1f ms%s%s\n",
                ok ? n : 0, before, spilled, bytes / 1073741824.0, freed, o_.keep_bytes / 1073741824.0, old,
                o_.keep_age_s / 86400, t1 - t0, f.size() / 1024.0, 1e3 * (now_s() - t1), ok ? "" : ": NOT WRITTEN, ",
                why.c_str());
  }

  // Operations that move session x's working state (its slot's) into the store: its compressed caches as
  // far as its history goes, and its end snapshot. They act on its slot (the command's).
  void persist_ops(Session& x, std::vector<StateOp>& v) {
    const int E = (int)x.hist.size();
    const model::Model::StateInfo info = m_.state_info(x.h);
    if (info.saved < E) v.push_back({kStSave, x.h, info.saved, E});
    if (info.end_pos != E) {
      v.push_back({kStSnapEnd, x.h, E, 0});
      x.end_lo = x.lo, x.end_clo = x.clo, x.end_w = x.w, x.end_dec_lo = x.dec_lo;
    }
  }
  // Slot s's session's state into the store, in a command of its own acting on s; `free`: the slot is let
  // go (its session then lives in the store only).
  void persist_slot(int s, bool free) {
    Session* x = slot_[s] ? find(slot_[s]) : nullptr;
    if (free) {
      slot_[s] = 0;
      if (x) x->slot = -1;
    }
    if (!x) return;
    std::vector<StateOp> v;
    persist_ops(*x, v);
    if (v.empty()) return;
    Cmd c{};
    c.op = kCmdState;
    c.slot = s;
    c.n_ops = (int)v.size();
    std::copy(v.begin(), v.end(), c.ops);
    issue(c);
  }
  void release(uint64_t id) {
    const auto it = sess_.find(id);
    if (it == sess_.end()) return;
    state({{kStFree, it->second.h, 0, 0}});
    if (it->second.slot >= 0) slot_[it->second.slot] = 0;
    sess_.erase(it);
  }
  // Session s's window snapshots that keep_snap no longer keeps at its history's end, dropped (not the
  // one at its last prompt's end).
  void thin(Session& s) {
    if (!o_.snap_thin) return;
    const int E = (int)s.hist.size();
    auto gone = [&](int sn) { return sn != s.pe_snap && !keep_snap(sn, E, o_.snap_every); };
    std::vector<StateOp> v;
    for (const int sn : m_.state_info(s.h).snaps)
      if (gone(sn)) v.push_back({kStDropSnap, s.h, sn, 0});
    s.enc_snaps.erase(std::remove_if(s.enc_snaps.begin(), s.enc_snaps.end(), gone), s.enc_snaps.end());
    if (!v.empty()) state(v);
  }
  // Sessions the store evicted to make room (an operation found its pool full) are gone here too.
  void forget_evicted() {
    for (const int h : m_.state_evicted())
      for (auto it = sess_.begin(); it != sess_.end(); ++it)
        if (it->second.h == h) {
          std::printf("serve: session %016llx evicted to make room\n", (unsigned long long)it->first);
          if (it->second.slot >= 0) slot_[it->second.slot] = 0;
          sess_.erase(it);
          break;
        }
  }
  // Least recently used sessions out until the store fits the budget: spilled to the disk tier (one live
  // in a slot saved first, and the slot let go), or freed without one. Sessions in flight stay.
  void evict(uint64_t keep) {
    const bool disk = m_.state_disk_capacity() > 0;
    std::vector<StateOp> v;
    std::vector<uint64_t> out;
    size_t bytes = m_.state_bytes();
    while (bytes > budget_) {
      auto victim = sess_.end();
      for (auto it = sess_.begin(); it != sess_.end(); ++it)
        if (it->first != keep && !flying_.count(it->first) && std::find(out.begin(), out.end(), it->first) == out.end() &&
            (victim == sess_.end() || it->second.used < victim->second.used) &&
            !(disk && m_.state_info(it->second.h).on_disk))
          victim = it;
      if (victim == sess_.end()) break;
      const size_t b = m_.state_info(victim->second.h).bytes;
      std::printf("serve: %s session %016llx (%.1f MiB, %zu positions)\n", disk ? "spill" : "evict",
                  (unsigned long long)victim->first, b / 1048576.0, victim->second.hist.size());
      bytes -= std::min(bytes, b);
      if (disk) {
        if (victim->second.slot >= 0) persist_slot(victim->second.slot, true);  // what it holds live goes with it
        v.push_back({kStSpill, victim->second.h, 0, 0});
        out.push_back(victim->first);
        continue;
      }
      v.push_back({kStFree, victim->second.h, 0, 0});
      if (victim->second.slot >= 0) slot_[victim->second.slot] = 0;
      sess_.erase(victim);
    }
    if (!v.empty()) state(v);
    forget_evicted();  // dropped from the disk tier to make room
  }
  // A request's session (or a fork's base) that was spilled comes back to the store before its plan
  // sees it. The slots' sessions stay where they are (every command pins them).
  void unspill(const fe::SubmitHeader& h) {
    Session* x = find(h.session_id);
    uint64_t id = h.session_id;
    if (!x && h.base_session_id) x = find(h.base_session_id), id = h.base_session_id;
    if (!x) return;
    const model::Model::StateInfo si = m_.state_info(x->h);
    if (!si.on_disk) return;
    const double t0 = now_s();
    state({{kStUnspill, x->h, 0, 0}});
    m_.drain();
    std::printf("serve: session %016llx back from disk: %.1f MiB, %zu positions, in %.1f ms\n", (unsigned long long)id,
                si.disk_bytes / 1048576.0, x->hist.size(), 1e3 * (now_s() - t0));
    forget_evicted();
  }

  // The earliest position a restart of s could start from (its snapshots, its end), for the log; -1:
  // none
  int first_restart(const Session& s) const {
    const model::Model::StateInfo info = m_.state_info(s.h);
    int r = info.end_pos;
    for (const int sn : info.snaps) {
      const int q = sn & 1 ? sn - 1 : sn;
      if (r < 0 || q < r) r = q;
    }
    return r;
  }

  // The restart point for continuing session s (its state live or stored) after a common prefix of p,
  // for a prompt of n positions.
  Plan choose(const Session& s, bool live, int p, int n, bool allow_ced, int replay_rows) const {
    Plan best;
    const model::Model::StateInfo info = m_.state_info(s.h);
    p = live ? p : std::min(p, info.saved);  // a stored session's compressed caches go as far as its pages
    auto consider = [&](int q, int how, int snap) {
      if (q > best.q) best = Plan{q, how, snap};
    };
    // A restart leaving fewer than this policy's replay rows needs valid decoder rings at q.
    auto usable = [&](int q, bool dec_ok) { return (allow_ced && n - q >= replay_rows) || dec_ok; };
    // with the ring and the carries as they were at position `end` (exact for [max(lo, w - 256), end)
    // and [max(clo, w - 16), end)), at most at `cap`
    auto from_ring = [](int end, int cap, int lo, int clo, int w) {
      int q = std::min(cap, end);
      if ((q & 1) && q - 1 < std::max(clo, w - 16)) --q;
      return q >= 0 && std::max(0, q - 128) >= std::max(lo, w - 256) ? q : -1;
    };
    // CED: else the latest restart that replays the decoder
    auto ring = [&](int end, int lo, int clo, int w, int dec_lo) {
      const int q = from_ring(end, p, lo, clo, w);
      return q < 0 || usable(q, std::max(0, q - 127) >= dec_lo) ? q
                                                                 : allow_ced ? from_ring(end, std::min(p, n - replay_rows), lo, clo, w) : -1;
    };
    if (live) consider(ring((int)s.hist.size(), s.lo, s.clo, s.w, s.dec_lo), kLive, -1);
    if (info.end_pos >= 0) consider(ring(info.end_pos, s.end_lo, s.end_clo, s.end_w, s.end_dec_lo), kEnd, -1);
    for (const int sn : info.snaps) {
      const int q = sn & 1 ? sn - 1 : sn;
      if (q <= p && usable(q, std::find(s.enc_snaps.begin(), s.enc_snaps.end(), sn) == s.enc_snaps.end()))
        consider(q, kSnap, sn);
    }
    return best;
  }

  void tokens(const Request& rq, const model::SampleOut& o, int row) {
    const bool lp = rq.h.flags & fe::kFlagLogprobs;
    const int nt = lp ? std::min<int>({(int)rq.h.n_top_logprobs, model::ops::kTopLogprobs, o.n_top}) : 0;
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
    // cut short by the server's stop, an answer is an error to its client, not a finished one
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

  // A submit: held while its session is in flight (a session's requests run one at a time); waiting while
  // no slot can be had (every slot's session in flight, and no time-slicing); else started.
  void admit(Request&& rq) {
    if (flying_.count(rq.h.session_id)) return held_.push_back(std::move(rq));
    const auto it = sess_.find(rq.h.session_id);
    if ((it == sess_.end() || it->second.slot < 0) && pick_slot({}) < 0) return wait_.push_back(std::move(rq));
    start(std::move(rq));
  }

  // A request checked, then its flight's first run.
  void start(Request&& rq) {
    Flight f;
    f.rq = std::move(rq);
    const fe::SubmitHeader& h = f.rq.h;
    f.t_start = now_s();
    const int n = (int)f.rq.prompt.size();
    std::string err;
    if (n < 1) err = "empty prompt";
    else if (n >= o_.max_ctx) err = "a prompt of " + std::to_string(n) + " tokens does not fit the context of " + std::to_string(o_.max_ctx);
    else if (h.session_id == 0) err = "session id 0";
    else if (h.max_new_tokens < 1) err = "max_new_tokens 0";
    else if (f.rq.stop.size() > (size_t)kMaxStop) err = "more than " + std::to_string(kMaxStop) + " stop ids";
    else if (!(h.temperature >= 0.f) || !(h.top_p > 0.f && h.top_p <= 1.f)) err = "bad sampling parameters";
    if (!fe::valid_submit_flags(h.flags)) err = "unknown submit flags";
    if (!fe::valid_prefill_flags(h.flags)) err = "conflicting prefill policy flags";
    for (uint64_t id : {h.session_id, h.base_session_id})
      if (const Session* s = find(id); s && s->prefill_policy != fe::prefill_policy(h.flags))
        err = "cross-prefill-policy session reuse rejected";
    for (const auto& image : f.rq.images)
      for (uint32_t i = image.start; i < image.start + image.rows; ++i)
        if (f.rq.prompt[i] != 129264) err = "image span must contain image token IDs";
    for (const int32_t t : f.rq.prompt)
      if (err.empty() && (t < 0 || t >= model::kVocab)) err = "token id out of range";
    if (!err.empty() || Io::cancelled(f.rq)) {
      done(f.rq, err.empty() ? fe::kCancelled : fe::kError, 0, 0, f.t_start, 0, err);
      return Io::forget(f.rq);
    }
    f.sp.temperature = h.temperature;
    f.sp.top_p = h.top_p;
    f.sp.top_k = (int32_t)std::min<uint32_t>(h.top_k, 1u << 30);
    f.sp.n_top = (h.flags & fe::kFlagLogprobs) ? (int32_t)std::min<uint32_t>(h.n_top_logprobs, model::ops::kTopLogprobs) : 0;
    f.sp.seed = (h.flags & fe::kFlagSeed) ? h.seed : rng_();
    flying_.insert(h.session_id);
    if (go(f, false)) {  // it decodes in the next run, first
      f.t_paused = now_s();
      rot_.push_front(std::move(f));
    }
  }

  // Flight f's prompt run in a slot: the request's first (resume false), or the resume of a request whose
  // session lost its slot, which runs the prompt and the tokens sent so far (the session holds them: it
  // restarts at their last position, exact continuation, one row; `keep`: the sessions whose slots it may
  // not take). Either way it sends a token, the flight's next to feed, and leaves its session live in the
  // slot; false when that token (or a cancel) ended the request. The sampler's noise is a function of the
  // seed and the position, so a request run in pieces samples what a single run would.
  bool go(Flight& f, bool resume, const std::set<uint64_t>& keep = {}) {
    const fe::SubmitHeader& h = f.rq.h;
    const double t0 = now_s();
    std::vector<int32_t> cont;
    if (resume) {
      if (Io::cancelled(f.rq)) return ended(f, fe::kCancelled);
      cont.reserve(f.rq.prompt.size() + f.out.size());
      cont.assign(f.rq.prompt.begin(), f.rq.prompt.end());
      cont.insert(cont.end(), f.out.begin(), f.out.end());
      if ((int)cont.size() >= o_.max_ctx) return ended(f, fe::kContext);
    }
    const std::vector<int32_t>& prompt = resume ? cont : f.rq.prompt;
    const int n = (int)prompt.size();
    Turn resumed;
    Turn& tu = resume ? resumed : f.tu;  // the log splits a request's first run
    forget_evicted();
    unspill(h);

    // ---- the slot, where the state comes from, and from which position on it is computed
    Session* S = find(h.session_id);
    Session* src = S;
    uint64_t src_id = h.session_id;
    if (!S && h.base_session_id) src = find(h.base_session_id), src_id = h.base_session_id;
    // the session's own slot. A fork takes its base's when no request of the base is in flight: the base's
    // state goes to the store, and the fork goes on from it in place. A base in flight keeps its slot; its
    // state goes to the store, and the fork starts from there in a slot of its own.
    int s = S ? S->slot : -1;
    if (!S && src && src->slot >= 0) {
      if (!flying_.count(src_id)) s = src->slot;
      else persist_slot(src->slot, false);
    }
    if (s < 0) s = pick_slot(keep);
    if (s < 0) die("serve: no slot for request %llu", (unsigned long long)h.req_id);  // admit() saw one
    const bool src_live = src && src->slot == s;
    const auto policy = fe::prefill_policy(h.flags);
    const int replay_rows = policy == fe::kTailPrefix ? model::kCedTailRows : model::kCedReplayRows;
    // Only the new policy uses a dependency-complete decoder tail for images.
    // Legacy preserves its existing text-only, approximate 128-row policy.
    const bool allow_ced = o_.ced && policy != fe::kFullPrefix &&
                           (policy == fe::kTailPrefix || f.rq.images.empty());
    const int p = src ? std::min({lcp(src->hist, prompt), n - 1,
                                 vision_prefix_limit(src->images, f.rq.images, src->prefill_policy != fe::kLegacyPrefix)}) : 0;
    const Plan pl = src ? choose(*src, src_live, p, n, allow_ced, replay_rows) : Plan{};
    const int q = pl.q;
    NVTX_RANGE("req %llu new %d ctx %d %s", (unsigned long long)h.req_id, n - q, n, kHow[pl.how]);
    if (!resume) {
      f.q = q, f.pl = pl;
      // where the state came from, for the log (counts only), and why a start is cold: no base named
      // (the front-end found no session to continue), a session or base this engine does not hold (lost
      // at a restart, or evicted), or a common prefix short of every restart point of the one it holds.
      // A request's first run only: a resume continues from the session's own end.
      char from[192];
      if (!src) {
        if (h.base_session_id)
          std::snprintf(from, sizeof from, "a fork of %016llx, which this engine does not hold",
                        (unsigned long long)h.base_session_id);
        else
          std::snprintf(from, sizeof from, "no base named, and this engine does not hold the session");
      } else {
        const int k = S ? std::snprintf(from, sizeof from, "its own history: common prefix %d ids", p)
                        : std::snprintf(from, sizeof from, "a fork of %016llx: common prefix %d ids",
                                        (unsigned long long)h.base_session_id, p);
        if (pl.how == kCold && k > 0 && k < (int)sizeof from) {
          const int r = first_restart(*src);
          if (r < 0) std::snprintf(from + k, sizeof from - k, ", and it has no restart point");
          else if (p < r) std::snprintf(from + k, sizeof from - k, ", under its first restart point %d", r);
          else std::snprintf(from + k, sizeof from - k, ", and no restart point up to it is usable");
        }
      }
      std::printf("serve: req %llu restart: %s; reused %d (%s)\n", (unsigned long long)h.req_id, from, q, kHow[pl.how]);
    }
    std::vector<StateOp> v;
    if (slot_[s] && slot_[s] != h.session_id) {  // the slot's session (a fork's base too) keeps its state in the store
      if (Session* y = find(slot_[s])) {
        persist_ops(*y, v);
        y->slot = -1;
      }
      slot_[s] = 0;
    }
    if (!S) {
      Session ns;
      ns.prefill_policy = policy;
      ns.images = f.rq.images;
      ns.h = next_h_++;
      S = &sess_.emplace(h.session_id, std::move(ns)).first->second;
    }
    if (src) {
      if (!src_live && q > 0) v.push_back({kStLoad, src->h, q, 0});
      if (pl.how == kSnap) v.push_back({kStRestore, src->h, pl.snap, 0});
      if (pl.how == kEnd) v.push_back({kStRestoreEnd, src->h, 0, 0});
      if (src != S) v.push_back({kStCopySnaps, src->h, S->h, q});  // a fork: src's snapshots are S's too
      else v.push_back({kStDrop, S->h, q, 0});                     // S forgets what follows q
    }
    // the new live state's exact ranges
    if (pl.how == kCold) S->lo = S->clo = S->w = 0;
    else if (pl.how == kSnap) S->lo = pl.snap - 128, S->clo = q, S->w = q;
    else if (pl.how == kEnd) S->lo = src->end_lo, S->clo = src->end_clo, S->w = src->end_w;
    else S->lo = src->lo, S->clo = src->clo, S->w = src->w;
    slot_[s] = h.session_id;
    S->slot = s;
    S->used = ++clock_;
    S->t_used = unix_now();
    S->images = f.rq.images;
    S->hist.assign(prompt.begin(), prompt.begin() + q);
    if (S->pe_snap > q) S->pe_snap = -1;  // dropped
    // CED: S's encoder-only snapshots up to q (a fork's are src's), where its decoder rings are valid
    if (src && src != S) S->enc_snaps = src->enc_snaps;
    S->enc_snaps.erase(std::remove_if(S->enc_snaps.begin(), S->enc_snaps.end(), [q](int x) { return x > q; }),
                       S->enc_snaps.end());
    S->dec_lo = !src || pl.how == kCold ? 0
                : pl.how == kSnap       ? std::max(0, q - 127)
                : pl.how == kEnd        ? src->end_dec_lo
                                        : src->dec_lo;
    // Short extensions continue through all layers from the live decoder
    // state. A long extension reconstructs its own tail, wholly at/after q.
    // Complete image layout covers that replay even when its rows precede the final feature window.
    const bool ced = allow_ced && n - q >= replay_rows;
    if (ced) S->dec_lo = kNever;                  // until the replay has run

    // ---- the prompt's tokens (Engram hashes 3 positions back), with the state operations, in the slot
    {
      Cmd c{};
      c.op = kCmdTokens;
      c.slot = s;
      c.pos0 = std::max(0, q - 3);
      c.n = n - c.pos0;
      if (v.size() > (size_t)kMaxOps) die("serve: %zu state operations for one command", v.size());
      c.n_ops = (int)v.size();
      std::copy(v.begin(), v.end(), c.ops);
      tu = Turn{};
      if (!resume && o_.profile_from_req && h.req_id == o_.profile_from_req && !g_prof) {
        c.flags |= model::kFlagProfile;  // the other ranks open it too (cmd.cpp follow)
        CUDA_CHECK(cudaProfilerStart());
        g_prof = true;
      }
      m_.mark(0);
      issue(c);
      m_.mark(1);
      std::vector<int32_t> t(prompt.begin() + c.pos0, prompt.end());
      bcast_payload(ring_, t.data(), t.size() * 4, 0);
      m_.set_tokens(c.pos0, c.n, t.data());
      tu.ops_ms = 1e3 * (now_s() - (resume ? t0 : f.t_start));
    }
    // Complete layout is independent of the feature window: Engram reads three positions back,
    // and CED's final replay can begin before the last encoder chunk.
    {
      Cmd c{}; c.op = kCmdVisionLayout; c.slot = s;
      std::vector<model::ops::VisionSpan> spans;
      for (const auto& image : f.rq.images)
        spans.push_back({(int)image.start, (int)image.rows, 0});
      c.n = (int)spans.size();
      issue(c);
      bcast_payload(ring_, spans.data(), spans.size() * sizeof(spans[0]), 0);
      m_.set_vision_layout(spans);
    }
    // ---- prefill [q, n): chunks of prefill_rows; a short remainder as one step
    int pos = q;
    model::SampleOut first{};
    int first_row = 0;
    bool cancelled = false;
    while (pos < n) {
      if (Io::cancelled(f.rq)) {
        cancelled = true;
        break;
      }
      const int R = std::min(o_.prefill_rows, n - pos);
      const bool last = pos + R == n;
      if (!f.rq.images.empty()) {
        std::vector<size_t> indices;
        uint64_t bytes = 0;
        std::string error;
        if (!vision_window(f.rq.images, pos, R, model::kMaxVisionFeatureBytes, indices, bytes, error)) {
          f.error = error; m_.drain(); return ended(f, fe::kError);
        }
        std::vector<model::ops::VisionSpan> spans;
        std::vector<uint16_t> features(size_t(bytes / 2));
        f.vision_peak_chunk_bytes = std::max(f.vision_peak_chunk_bytes, bytes);
        size_t at = 0;
        for (size_t image_index : indices) {
          const auto& image = f.rq.images[image_index];
          spans.push_back({int(image.start), int(image.rows), int(at / model::kDim)});
          const auto& file = f.rq.image_features[image_index];
          ++f.vision_file_reads;
          if (!file || !file->read(features.data() + at, [&] { return Io::cancelled(f.rq); }, error, &f.vision_disk_bytes)) {
            m_.drain();
            if (Io::cancelled(f.rq)) return ended(f, fe::kCancelled);
            f.error = error.empty() ? "missing image feature file" : error;
            return ended(f, fe::kError);
          }
          at += size_t(image.rows) * model::kDim;
        }
        // No rank waits on a payload until every needed file is read and validated.
        if (Io::cancelled(f.rq)) { m_.drain(); return ended(f, fe::kCancelled); }
        Cmd v{}; v.op = kCmdVision; v.slot = s; v.n = spans.size(); v.rows = bytes / (model::kDim * 2);
        issue(v);
        bcast_payload(ring_, spans.data(), spans.size() * sizeof(spans[0]), 0);
        bcast_payload(ring_, features.data(), features.size() * 2, 0);
        m_.set_vision_chunk(spans, features.data(), v.rows);
        std::printf("serve: req %llu vision chunk: pos %d, rows %d, images %zu, feature bytes %llu\n",
                    (unsigned long long)h.req_id, pos, R, indices.size(), (unsigned long long)bytes);
      }
      Cmd c{};
      c.slot = s;
      c.pos0 = pos;
      c.rows = R;
      c.sp = f.sp;
      c.handle = S->h;
      for (int sn = (pos / o_.snap_every + 1) * o_.snap_every; sn <= pos + R; sn += o_.snap_every)
        if (sn >= 128 && c.n_snaps < kMaxSnaps && (!o_.snap_thin || keep_snap(sn, n, o_.snap_every)))
          c.snaps[c.n_snaps++] = sn;  // none that the prompt's end would thin at once
      wd_.started = now_s();
      if (R <= o_.small_rows && !ced && f.rq.images.empty()) {
        c.op = kCmdRows;
        std::copy(prompt.begin() + pos, prompt.begin() + pos + R, c.tokens);
        issue(c);
        const uint64_t k = run_rows(m_, c);
        first = m_.wait(k);
        tu.rows_ms += m_.last_gpu_ms();
        first_row = R - 1;
        S->w = std::max(S->w, pos + R + (m_.spec() ? model::ops::kDrafts : 0));  // its draft's block keys
      } else {
        c.op = kCmdPrefill;
        c.flags = (last ? kPfSample : 0) | (ced ? kPfCed : 0) |
                  (ced && policy == fe::kTailPrefix ? kPfTailExact : 0);
        const int rest = n - pos - R;
        c.next = rest > o_.small_rows || (ced && rest > 0) ? std::min(o_.prefill_rows, rest) : 0;
        for (int i = 0; i < c.n_snaps; ++i)
          if (ced && c.snaps[i] < n) S->enc_snaps.push_back(c.snaps[i]);  // the replay covers only one at n
        issue(c);
        const uint64_t k = run_prefill(m_, c);
        if (k != ~0ull) first = m_.wait(k), first_row = 0;
        if (!last) m_.drain();  // the last chunk's tail (the draft windows' seeding) drains after tok0 is out
        if (!last) tu.add_split(m_);
        else tu.pending = true;
        ++tu.chunks;
        S->w = std::max(S->w, pos + R);
      }
      wd_.started = 0;
      S->hist.insert(S->hist.end(), prompt.begin() + pos, prompt.begin() + pos + R);
      pos += R;
      // Completed at the existing wait/drain boundary: no additional GPU sync.
    }
    const double t_first = now_s();
    if (resume) f.resume_s += t_first - t0;
    else f.t_first = t_first;
    if (ced && !cancelled) S->dec_lo = model::ced_valid_from(n, replay_rows);
    auto drain = [&] {
      wd_.started = now_s();
      m_.drain();
      wd_.started = 0;
    };
    if (cancelled || Io::cancelled(f.rq)) {
      drain();
      return ended(f, fe::kCancelled);
    }

    // ---- the token, the next to feed; then the last chunk's tail drains
    const int32_t tok0 = first.row[first_row].token;
    tokens(f.rq, first, first_row);
    f.out.push_back(tok0);
    drain();
    tu.ops_gpu_ms = m_.mark_ms(0, 1);
    if (tu.pending) tu.add_split(m_);  // the last chunk's
    tu.tail_ms = 1e3 * (now_s() - t_first);
    // the window snapshot at the prompt's end (a first run's): a later request that keeps the prompt but
    // not what was generated after it (an agent dropping the reasoning, say) restarts there
    if (!resume && n >= 128 && n % o_.snap_every != 0 && n != S->pe_snap) {
      Cmd c{};
      c.op = kCmdState;
      c.slot = s;
      if (S->pe_snap >= 0) c.ops[c.n_ops++] = {kStDropSnap, S->h, S->pe_snap, 0};
      c.ops[c.n_ops++] = {kStSnap, S->h, n, 0};
      S->pe_snap = n;
      issue(c);
    }
    if (stop_id(f, tok0)) return ended(f, fe::kStop);
    if ((int)f.out.size() >= (int)h.max_new_tokens) return ended(f, fe::kLength);
    return true;
  }

  bool ended(Flight& f, uint32_t why) {
    finish(f, why);
    return false;
  }
  static bool stop_id(const Flight& f, int32_t t) { return std::count(f.rq.stop.begin(), f.rq.stop.end(), t) > 0; }

  // A slot for a session that holds none: a free one; else the least recently used one whose session no
  // request uses (its state goes to the store); else, with time-slicing, the slot of the paused request
  // whose turn comes last. Never the slot of a session in `keep`. -1: none.
  int pick_slot(const std::set<uint64_t>& keep) const {
    const int ns = (int)slot_.size();
    for (int s = 0; s < ns; ++s)
      if (!slot_[s]) return s;
    int best = -1;
    uint64_t used = UINT64_MAX;
    for (int s = 0; s < ns; ++s) {
      if (flying_.count(slot_[s]) || keep.count(slot_[s])) continue;
      const auto it = sess_.find(slot_[s]);
      const uint64_t u = it == sess_.end() ? 0 : it->second.used;
      if (u < used) best = s, used = u;
    }
    if (best >= 0 || o_.slice <= 0) return best;
    for (auto it = rot_.rbegin(); it != rot_.rend(); ++it) {
      const auto x = sess_.find(it->rq.h.session_id);
      if (x != sess_.end() && x->second.slot >= 0 && !keep.count(x->first)) return x->second.slot;
    }
    return -1;
  }
  // whether a submit that waits may pause the requests decoding: it can have a slot
  bool admissible() const { return o_.slice > 0 || pick_slot({}) >= 0; }

  static constexpr uint32_t kPaused = ~0u, kTurn = ~0u - 1;  // a run's ends that are not the request's

  // A run: the requests at the front of the turns, as many as there are slots, each with its session live
  // in its slot (one that lost its slot resumes first, in a slot taken from a request further back), decode
  // together (batch_loop, a row of each per step) or one alone (the speculative loop, with drafts). Each runs
  // to its end, or until a new request that can have a slot pauses them all (prefill first), or, while
  // requests wait for a slot, for a turn of o_.slice tokens. Then they wait again: the paused ones first, in
  // their order, those whose turn is over last.
  void run() {
    std::vector<Flight> fl;
    std::set<uint64_t> keep;
    while (fl.size() < slot_.size() && !rot_.empty()) {
      Flight f = std::move(rot_.front());
      rot_.pop_front();
      f.paused_s += now_s() - f.t_paused;
      const auto it = sess_.find(f.rq.h.session_id);
      if ((it == sess_.end() || it->second.slot < 0) && !go(f, true, keep)) continue;
      keep.insert(f.rq.h.session_id);
      fl.push_back(std::move(f));
    }
    if (fl.empty()) return;
    const bool turns = o_.slice > 0 && !rot_.empty();  // requests wait for a slot: a turn each
    std::vector<uint32_t> why(fl.size());
    if (fl.size() == 1) why[0] = decode_one(fl[0], turns);
    else decode_batch(fl, turns, why);
    std::vector<Flight> paused;
    for (size_t k = 0; k < fl.size(); ++k) {
      fl[k].t_paused = now_s();
      if (why[k] == kPaused) paused.push_back(std::move(fl[k]));
      else if (why[k] == kTurn) rot_.push_back(std::move(fl[k]));
      else finish(fl[k], why[k]);
    }
    for (auto it = paused.rbegin(); it != paused.rend(); ++it) rot_.push_front(std::move(*it));
  }

  // Session S's history for a run that feeds the flight's next token at pos: the positions before it (one
  // more, the token's own, when a run before fed it: it goes in again, with the same KV).
  void rejoin(Session& S, int pos, uint64_t req) {
    if ((int)S.hist.size() < pos)
      die("serve: req %llu: its session holds %zu positions, its next token goes at %d", (unsigned long long)req,
          S.hist.size(), pos);
    S.hist.resize(pos);
  }

  // Flight f's run alone, in its slot: the decode loop (speculative with drafts, unless --no-spec) from its
  // next token on. Its end (fe::k*), or kPaused / kTurn.
  uint32_t decode_one(Flight& f, bool turns) {
    const fe::SubmitHeader& h = f.rq.h;
    Session* S = find(h.session_id);
    const int pos = (int)(f.rq.prompt.size() + f.out.size()) - 1, max_new = (int)h.max_new_tokens;
    const int left = max_new - (int)f.out.size();
    rejoin(*S, pos, h.req_id);
    Cmd c{};
    c.op = kCmdDecode;
    c.slot = S->slot;
    c.pos0 = pos;
    c.tokens[0] = f.out.back();
    c.sp = f.sp;
    c.max_new = turns ? std::min(left, o_.slice) : left;  // a turn while others wait for a slot
    c.limit = o_.max_ctx;
    c.n_stop = (int)f.rq.stop.size();
    std::copy(f.rq.stop.begin(), f.rq.stop.end(), c.stop);
    c.ahead = 1;
    c.handle = S->h;
    c.snap_every = o_.snap_every;
    c.spec = o_.spec;
    c.spec_drafts = o_.spec_drafts;
    c.spec_thresh = o_.spec_thresh;
    std::copy(o_.spec_cost, o_.spec_cost + 3, c.spec_cost);
    c.spec_gate = o_.spec_gate;
    std::copy(o_.spec_row_ms, o_.spec_row_ms + model::ops::kDrafts + 1, c.spec_row_ms);
    c.spec_tok = f.spec_tok, c.spec_ms = f.spec_ms;  // the drafts' rate goes on from the last run
    const double t_dec = now_s();
    issue(c);
    std::vector<int32_t> samples;
    bool ctl = false, cancel = false, pause = false;
    const DecodeEnd e = decode_loop(m_, c, wd_, [&](int, const model::SampleOut& o, int row) {
      samples.push_back(o.row[row].token);
      if (f.tu.dec1_ms < 0) f.tu.dec1_ms = 1e3 * (now_s() - t_dec);
      if (cancel) return;
      if (Io::cancelled(f.rq)) {
        if (!ctl) m_.set_ctl(1);  // every rank stops after the step that reads it
        ctl = cancel = true;
        return;
      }
      tokens(f.rq, o, row);  // a paused request's last tokens too: the session holds them
      f.out.push_back(o.row[row].token);
      if (!ctl && io_.submit_waiting() && admissible()) {  // a new request: this one pauses for its prefill
        m_.set_ctl(1);
        ctl = pause = true;
      }
    });
    if (ctl) m_.set_ctl(0);
    f.dec_s += now_s() - t_dec;
    f.decoded += e.sampled;
    f.spec_tok = e.spec_tok, f.spec_ms = e.spec_ms;
    add(f.de, e);
    ++f.slices;
    S->w = std::max(S->w, e.written);
    if (e.fed > 0) {  // the history: the token it started from, and the samples fed after it
      S->hist.push_back(c.tokens[0]);
      S->hist.insert(S->hist.end(), samples.begin(), samples.begin() + std::max(0, e.fed - 1));
    }
    // a pause's last token may be a stop id (the control word ends the loop before it checks them)
    return cancel || (e.why == kEndCancel && !pause) ? fe::kCancelled
           : e.why == kEndStop || stop_id(f, f.out.back()) ? fe::kStop
           : e.why == kEndContext                          ? fe::kContext
           : (int)f.out.size() >= max_new                  ? fe::kLength
           : pause                                         ? kPaused
                                                           : kTurn;
  }

  // The flights' run together, each in its own slot: a row of each per step (batch_loop), or with drafts
  // (spec_batch_loop, B3) when the server has them and the batch is small enough for a draft each; with
  // drafts the last one left pauses, to go on alone. Each one's end (fe::k*), or kPaused / kTurn, in why.
  void decode_batch(std::vector<Flight>& fl, bool turns, std::vector<uint32_t>& why) {
    const int n = (int)fl.size();
    const bool drafts = o_.spec && m_.spec() && n <= model::kMaxRows / 2 && n <= m_.spec_batch_entries();
    std::vector<BatchEntry> e(n);
    std::vector<Session*> S(n);
    for (int k = 0; k < n; ++k) {
      const Flight& f = fl[k];
      S[k] = find(f.rq.h.session_id);
      const int pos = (int)(f.rq.prompt.size() + f.out.size()) - 1;
      rejoin(*S[k], pos, f.rq.h.req_id);
      BatchEntry& x = e[k];
      x.slot = S[k]->slot, x.handle = S[k]->h, x.pos0 = pos, x.token = f.out.back();
      const int left = (int)f.rq.h.max_new_tokens - (int)f.out.size();
      x.max_new = turns ? std::min(left, o_.slice) : left;  // a turn while others wait for a slot
      x.limit = o_.max_ctx, x.snap_every = o_.snap_every;
      x.n_stop = (int)f.rq.stop.size();
      std::copy(f.rq.stop.begin(), f.rq.stop.end(), x.stop);
      x.sp = f.sp;
      x.spec_tok = f.spec_tok, x.spec_ms = f.spec_ms;  // the drafts' rate goes on from the last run
    }
    Cmd c{};
    c.op = drafts ? kCmdSpecBatch : kCmdBatch;
    c.n = n;
    c.slot = e[0].slot;
    c.flags = o_.spec ? kBatchSolo : 0;
    if (drafts) {
      c.spec = 1;
      c.spec_drafts = o_.spec_drafts;
      c.spec_thresh = o_.spec_thresh;
      std::copy(o_.spec_cost, o_.spec_cost + 3, c.spec_cost);
      c.spec_gate = o_.spec_gate;
      std::copy(o_.spec_row_ms, o_.spec_row_ms + model::ops::kDrafts + 1, c.spec_row_ms);
    }
    const double t_dec = now_s();
    issue(c);
    bcast_payload(ring_, e.data(), e.size() * sizeof(BatchEntry), 0);
    uint32_t ctl = 0;
    std::vector<std::vector<int32_t>> samples(n);
    std::vector<char> cancel(n, 0);
    const OnBatch cb = [&](int k, const model::SampleOut& o, int row) {
      Flight& f = fl[k];
      samples[k].push_back(o.row[row].token);
      if (f.tu.dec1_ms < 0) f.tu.dec1_ms = 1e3 * (now_s() - t_dec);
      if (cancel[k]) return;
      if (Io::cancelled(f.rq)) {  // its slot's bit: it ends after the step that reads it
        cancel[k] = 1;
        m_.set_ctl(ctl |= 1u << e[k].slot);
        return;
      }
      const int32_t t = o.row[row].token;
      tokens(f.rq, o, row);  // a paused request's last tokens too: the session holds them
      f.out.push_back(t);
      // a new request pauses the batch when it can have a slot; one waiting for a slot does when this one
      // ends, which frees one (its session no longer in flight)
      const bool ends = stop_id(f, t) || (int)f.out.size() >= (int)f.rq.h.max_new_tokens;
      if (!(ctl & kCtlPause) && (io_.submit_waiting() || !wait_.empty()) && (ends || admissible()))
        m_.set_ctl(ctl |= kCtlPause);
    };
    std::vector<BatchEnd> end(n);
    std::vector<DecodeEnd> dend;
    SpecBatchStats st;
    if (drafts) {
      dend = spec_batch_loop(m_, c, e, wd_, cb, &st);
      for (int k = 0; k < n; ++k) end[k].sampled = dend[k].sampled, end[k].fed = dend[k].fed, end[k].why = dend[k].why;
    } else {
      end = batch_loop(m_, c, e, wd_, cb);
    }
    if (ctl) m_.set_ctl(0);
    const double dt = now_s() - t_dec;
    std::string ends;
    for (int k = 0; k < n; ++k) {
      Flight& f = fl[k];
      const BatchEnd& x = end[k];
      f.dec_s += dt;
      f.decoded += x.sampled, f.batched += x.sampled;
      ++f.slices;
      if (drafts) {
        f.spec_tok = dend[k].spec_tok, f.spec_ms = dend[k].spec_ms;
        add(f.de, dend[k]);
        S[k]->w = std::max(S[k]->w, dend[k].written);
      } else {
        S[k]->w = std::max(S[k]->w, e[k].pos0 + x.fed);
      }
      if (x.fed > 0) {  // the history: the token it started from, and the samples fed after it
        S[k]->hist.push_back(e[k].token);
        S[k]->hist.insert(S[k]->hist.end(), samples[k].begin(), samples[k].begin() + std::max(0, x.fed - 1));
      }
      const int max_new = (int)f.rq.h.max_new_tokens;
      why[k] = cancel[k] || x.why == kEndCancel                ? fe::kCancelled
               : x.why == kEndStop || stop_id(f, f.out.back()) ? fe::kStop
               : x.why == kEndContext                          ? fe::kContext
               : (int)f.out.size() >= max_new                  ? fe::kLength
               : x.why == kEndPause                            ? kPaused
                                                               : kTurn;
      char b[64];
      std::snprintf(b, sizeof b, "%s%llu %d %s", k ? ", " : "", (unsigned long long)f.rq.h.req_id, x.sampled,
                    why[k] == kPaused ? "paused" : why[k] == kTurn ? "turn" : why[k] == fe::kStop ? "stop"
                    : why[k] == fe::kLength ? "length" : why[k] == fe::kCancelled ? "cancelled" : "context");
      ends += b;
    }
    // the run for the log: its steps' median (g_t_wait: when each step's samples came); with drafts, the
    // tokens and rows per step, the drafts accepted of those verified, and the steps' GPU ms by rows
    std::vector<double> d;
    for (size_t i = 1; i < g_t_wait.size(); ++i) d.push_back(1e3 * (g_t_wait[i] - g_t_wait[i - 1]));
    std::sort(d.begin(), d.end());
    std::string dr;
    if (drafts) {
      int ver = 0, acc = 0;
      for (int k = 0; k < n; ++k) ver += dend[k].verified, acc += dend[k].accepted;
      char b[160];
      std::snprintf(b, sizeof b, "; drafts: %.2f tokens and %.2f rows per step, %d of %d drafts accepted; GPU ms by rows:",
                    st.steps ? (double)st.emitted / st.steps : 0.0, st.steps ? (double)st.rows / st.steps : 0.0, acc, ver);
      dr = b;
      for (int r = 1; r <= model::kMaxRows; ++r)
        if (st.gpu_n[r]) {
          std::snprintf(b, sizeof b, " %d: %.2f (%d)", r, st.gpu_ms[r] / st.gpu_n[r], st.gpu_n[r]);
          dr += b;
        }
    }
    std::printf("serve: batch of %d: %zu steps in %.1f ms, %.2f ms a step (median); req, tokens, end: %s%s\n", n,
                g_t_wait.size(), 1e3 * dt, d.empty() ? 0.0 : d[d.size() / 2], ends.c_str(), dr.c_str());
  }

  // Paused requests that their client cancelled, or that the server's stop ends, end now, not at their
  // turn; so do requests waiting for a slot.
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
      f.paused_s += now_s() - f.t_paused;
      finish(f, fe::kCancelled);
    }
  }

  // DONE, then the session kept (its snapshots thinned) or freed, and the store held to its budget; the
  // requests held for the session go on.
  void finish(Flight& f, uint32_t why) {
    const fe::SubmitHeader& h = f.rq.h;
    done(f.rq, why, (uint32_t)f.q, (uint32_t)f.out.size(), f.t_start, f.t_first, f.error);
    Io::forget(f.rq);
    const double t_end = now_s();
    flying_.erase(h.session_id);
    const bool released = later_.erase(h.session_id) > 0;  // the front-end let it go meanwhile
    forget_evicted();
    Session* S = why != fe::kError && (h.flags & fe::kFlagKeep) && !released ? find(h.session_id) : nullptr;
    if (S) thin(*S);
    const model::Model::StateInfo si = S ? m_.state_info(S->h) : model::Model::StateInfo{};
    const DecodeEnd* de = &f.de;
    if (de->vsteps) {
      std::printf("serve: req %llu drafts: %d verify steps, %.2f drafts verified and %.2f accepted per step, "
                  "steps verifying 0..5: %d %d %d %d %d %d; gate %.3f (%.3f-%.3f); GPU ms by rows:",
                  (unsigned long long)h.req_id, de->vsteps, (double)de->verified / de->vsteps,
                  (double)de->accepted / de->vsteps, de->rows_hist[0], de->rows_hist[1], de->rows_hist[2],
                  de->rows_hist[3], de->rows_hist[4], de->rows_hist[5], de->gate_sum / std::max(1, de->gate_n),
                  de->gate_min, de->gate_max);
      for (int r = 1; r <= model::ops::kDrafts + 1; ++r)
        if (de->gpu_n[r]) std::printf(" %d: %.2f (%d)", r, de->gpu_ms[r] / de->gpu_n[r], de->gpu_n[r]);
      std::printf("\n");
    }
    // the decode's time: from the first token to the end, less the time paused and resuming
    const double t_dec = f.t_first > 0 ? std::max(0.0, t_end - f.t_first - f.paused_s - f.resume_s) : 0;
    char sl[128] = "";  // (a request waits a moment in the turns between its first token and its decode)
    if (f.slices > 1 || f.paused_s >= 0.005 || f.batched)
      std::snprintf(sl, sizeof sl, "; %d slices, paused %.2f s, resumes %.1f ms, %d tokens in batches", f.slices,
                    f.paused_s, 1e3 * f.resume_s, f.batched);
    std::printf("serve: req %llu session %016llx: prompt %u, reused %d (%s), prefill %.1f ms (%.0f tok/s), %zu tokens "
                "in %.1f ms (%.1f tok/s), %s; store %.1f MiB; session %d positions, %.1f MiB, %zu window snapshots%s\n",
                (unsigned long long)h.req_id, (unsigned long long)h.session_id, h.n_prompt, f.q, kHow[f.pl.how],
                1e3 * (f.t_first > 0 ? f.t_first - f.t_start : 0.0),
                (h.n_prompt - f.q) / std::max(1e-9, f.t_first - f.t_start), f.out.size(), 1e3 * t_dec,
                f.decoded / std::max(1e-9, t_dec),
                why == fe::kStop ? "stop" : why == fe::kLength ? "length" : why == fe::kCancelled ? "cancelled" :
                why == fe::kError ? "error" : "context",
                m_.state_bytes() / 1048576.0, S ? (int)S->hist.size() : 0, si.bytes / 1048576.0, si.snaps.size(), sl);
    const Turn& u = f.tu;
    if (!f.rq.images.empty()) {
      uint64_t rows = 0;
      for (const auto& image : f.rq.images) rows += image.rows;
      std::printf("serve: req %llu vision audit: history_images=%zu history_rows=%llu logical_feature_bytes=%llu "
                  "disk_read_bytes=%llu max_chunk_feature_bytes=%llu file_reads=%llu inline_bytes=0\n",
                  (unsigned long long)h.req_id, f.rq.images.size(), (unsigned long long)rows,
                  (unsigned long long)(rows * model::kDim * 2), (unsigned long long)f.vision_disk_bytes,
                  (unsigned long long)f.vision_peak_chunk_bytes, (unsigned long long)f.vision_file_reads);
    }
    std::printf("serve: req %llu turn: state ops %.1f ms (GPU %.1f), %d chunks: layers 0-19 %.1f ms, replay %.1f ms, "
                "head %.1f ms, draft windows %.1f ms, rows %.1f ms; first token to decode %.1f ms, first decode step "
                "%.1f ms\n",
                (unsigned long long)h.req_id, u.ops_ms, u.ops_gpu_ms, u.chunks, u.pf[0], u.pf[1], u.pf[2], u.pf[3],
                u.rows_ms, u.tail_ms, u.dec1_ms);
    if (!S) release(h.session_id);
    for (auto it = held_.begin(); it != held_.end();)  // the session's held requests go on, in order
      if (it->h.session_id == h.session_id) ready_.push_back(std::move(*it)), it = held_.erase(it);
      else ++it;
    evict(h.session_id);
  }

  Session* find(uint64_t id) {
    const auto it = sess_.find(id);
    return it == sess_.end() ? nullptr : &it->second;
  }

  model::Model& m_;
  comm::Ring& ring_;
  Watchdog& wd_;
  const ServeOptions& o_;
  Io& io_;
  const size_t budget_;  // the store's bytes kept between requests (the rest: room for the next one)
  std::map<uint64_t, Session> sess_;
  std::vector<uint64_t> slot_;  // the session whose working state each slot holds (0: none)
  std::deque<Flight> rot_;      // requests between their first token and their end, in turn
  std::deque<Request> held_, ready_, wait_;  // submits for a session in flight; those let go when it ended;
                                             // those waiting for a slot
  std::set<uint64_t> flying_, later_;  // the sessions in flight; RELEASEs of them, applied at their end
  int next_h_ = 1;
  uint64_t clock_ = 0;
  std::mt19937_64 rng_{std::random_device{}()};
};

}  // namespace

int serve(model::Model& m, comm::Ring& ring, Watchdog& wd, const ServeOptions& o) {
  if (o.prefill_rows / o.snap_every > kMaxSnaps) die("serve: more than %d snapshots per prefill chunk", kMaxSnaps);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  fe::HelloEngine hi{};
  hi.version = fe::kVersion;
  hi.max_context = (uint32_t)o.max_ctx;
  hi.vocab_size = model::kVocab;
  hi.eos_id = 1;
  // between requests the store keeps at most its pool less room for one long request (a 64k-position
  // one holds ~0.25 GiB); a request that needs more evicts in the store (model.h state_reserve)
  const size_t cap = std::min(o.budget, m.state_capacity()), room = std::max<size_t>(256ull << 20, cap / 16);
  const size_t budget = cap > room ? cap - room : 0;
  hi.max_sessions = (uint32_t)std::max<size_t>(1, budget / (64ull << 20));
  hi.name_len = (uint32_t)o.name.size();
  Io io(o.socket, Frame(fe::kHelloEngine).add(&hi).add(o.name.data(), o.name.size()).take());
  std::printf("serve: listening on %s (context %d, session store %.2f GiB, kept between requests %.2f GiB)\n",
              o.socket.c_str(), o.max_ctx, m.state_capacity() / 1073741824.0, budget / 1073741824.0);
  Server s(m, ring, wd, o, io, budget);
  s.load_kept();
  s.loop();
  if (g_prof) CUDA_CHECK(cudaProfilerStop());
  std::printf("serve: stopped\n");
  return 0;
}

}  // namespace srv
