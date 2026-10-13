// A minimal rank-0 side of the front-end protocol (frontend_protocol.h, docs/protocol.md): accepts
// the front-end, exchanges HELLO, and answers every SUBMIT by streaming back the last few prompt
// ids in reverse order followed by EOS. It checks this header against the Rust side and shows the
// shape of the real server loop. Build and run:
//   g++ -std=c++20 -O2 -I engine engine/common/frontend_echo_engine.cpp -o echo_engine
//   ./echo_engine /tmp/dsgb10-engine.sock
#include "common/frontend_protocol.h"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMaxFrame = 64u << 20;
constexpr uint32_t kEos = 1;

bool read_all(int fd, void* p, size_t n) {
  auto* b = static_cast<char*>(p);
  while (n > 0) {
    ssize_t r = read(fd, b, n);
    if (r <= 0) return false;
    b += r;
    n -= size_t(r);
  }
  return true;
}

bool write_all(int fd, const void* p, size_t n) {
  auto* b = static_cast<const char*>(p);
  while (n > 0) {
    ssize_t r = write(fd, b, n);
    if (r <= 0) return false;
    b += r;
    n -= size_t(r);
  }
  return true;
}

// One outgoing frame: header, fixed struct, then arrays.
struct Out {
  std::vector<char> buf;
  explicit Out(uint16_t type) : buf(sizeof(fe::FrameHeader)) {
    fe::FrameHeader h{0, type, 0};
    std::memcpy(buf.data(), &h, sizeof h);
  }
  template <class T>
  Out& add(const T* p, size_t n = 1) {
    const char* c = reinterpret_cast<const char*>(p);
    buf.insert(buf.end(), c, c + n * sizeof(T));
    return *this;
  }
  bool send(int fd) {
    uint32_t size = uint32_t(buf.size() - sizeof(fe::FrameHeader));
    std::memcpy(buf.data(), &size, sizeof size);
    return write_all(fd, buf.data(), buf.size());
  }
};

bool done(int fd, uint64_t req_id, fe::DoneReason reason, uint32_t prompt, uint32_t sent,
          const std::string& error = "") {
  fe::DoneHeader d{};
  d.req_id = req_id;
  d.reason = reason;
  d.prompt_tokens = prompt;
  d.completion_tokens = sent;
  d.error_len = uint32_t(error.size());
  return Out(fe::kDone).add(&d).add(error.data(), error.size()).send(fd);
}

// Serves one front-end connection until it closes or sends a malformed frame.
void serve(int fd) {
  const std::string name = "echo";
  fe::HelloEngine hello{fe::kVersion, 1 << 20, 129280, kEos, 16, uint32_t(name.size())};
  if (!Out(fe::kHelloEngine).add(&hello).add(name.data(), name.size()).send(fd)) return;
  for (;;) {
    fe::FrameHeader h;
    if (!read_all(fd, &h, sizeof h) || h.size > kMaxFrame) return;
    std::vector<char> p(h.size);
    if (!read_all(fd, p.data(), p.size())) return;
    if (h.type == fe::kHello) {
      fe::HelloFe hi;
      if (p.size() < sizeof hi) return;
      std::memcpy(&hi, p.data(), sizeof hi);
      if (hi.version != fe::kVersion) return;
    } else if (h.type == fe::kSubmit) {
      fe::SubmitHeader s;
      if (p.size() < sizeof s) return;
      std::memcpy(&s, p.data(), sizeof s);
      if (p.size() != sizeof s + 4 * (size_t(s.n_stop_ids) + s.n_prompt)) return;
      std::vector<uint32_t> stop(s.n_stop_ids), prompt(s.n_prompt);
      const char* arrays = p.data() + sizeof s;
      if (!stop.empty()) std::memcpy(stop.data(), arrays, stop.size() * 4);
      if (!prompt.empty()) std::memcpy(prompt.data(), arrays + stop.size() * 4, prompt.size() * 4);
      if (prompt.empty() || s.session_id == 0 || s.max_new_tokens == 0) {
        if (!done(fd, s.req_id, fe::kError, s.n_prompt, 0, "bad request")) return;
        continue;
      }
      std::vector<uint32_t> out;
      for (size_t i = 0; i < 8 && i < prompt.size(); i++) out.push_back(prompt[prompt.size() - 1 - i]);
      out.push_back(kEos);
      uint32_t sent = 0;
      fe::DoneReason reason = fe::kLength;
      for (uint32_t id : out) {
        if (sent == s.max_new_tokens) break;
        fe::TokensHeader t{s.req_id, 1, 0};
        if (!Out(fe::kTokens).add(&t).add(&id).send(fd)) return;
        sent++;
        bool is_stop = false;
        for (uint32_t x : stop) is_stop |= x == id;
        if (is_stop) {
          reason = fe::kStop;
          break;
        }
      }
      if (!done(fd, s.req_id, reason, s.n_prompt, sent)) return;
    }  // CANCEL and RELEASE: nothing is in flight between frames and no session is kept
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : fe::kDefaultSocket;
  int ls = socket(AF_UNIX, SOCK_STREAM, 0);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof addr.sun_path, "%s", path);
  unlink(path);
  if (ls < 0 || bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || listen(ls, 1) != 0) {
    std::perror("bind/listen");
    return 1;
  }
  chmod(path, 0600);
  std::fprintf(stderr, "echo engine on %s\n", path);
  for (;;) {
    int fd = accept(ls, nullptr, nullptr);
    if (fd < 0) continue;
    serve(fd);
    close(fd);
  }
}
