// dsw_rank_test: the loader refuses another rank's weight file (dsw::rank_mismatch; WeightFile(path, rank,
// world), which the engine constructs with its own rank). Host only: WeightFile's constructor reads the
// header and TOC and makes no CUDA call, so no device is needed (CUDA_VISIBLE_DEVICES= makes sure).
//   dsw_rank_test [DIR]        rank_mismatch on headers; then a small synthetic file per rank in DIR
//                              (default .; read with O_DIRECT, so not on tmpfs): each opened as its own
//                              rank, refused as the three others with the file and both ranks in the
//                              message, and opened as any rank with no rank given (no check)
//   dsw_rank_test --file F R   a real rank file: opened as rank R of 4, refused as the three others
//   dsw_rank_test --make F R   writes a synthetic file that says rank R of 4 (for the engine's own refusal
//                              on the ring: dsgb10_rank --weights F)
// Exit 0 when every check passes.
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "weights/io.h"
#include "weights/loader.h"

using namespace dsw;

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++fails;
}

// A minimal valid rank file: the header, one TOC entry, one 4 KiB F32 tensor.
void make_file(const std::string& path, uint32_t rank, uint32_t world) {
  std::vector<uint8_t> f(3 * kAlign, 0);
  Header h{};
  std::memcpy(h.magic, kMagic, 8);
  h.version = kVersion;
  h.header_bytes = sizeof(Header);
  h.rank = rank;
  h.world = world;
  h.n_tensors = 1;
  h.toc_entry_bytes = sizeof(TocEntry);
  h.toc_offset = kAlign;
  h.data_offset = 2 * kAlign;
  h.data_bytes = kAlign;
  h.file_bytes = 3 * kAlign;
  h.alignment = kAlign;
  std::snprintf(h.model, sizeof h.model, "dsw_rank_test");
  TocEntry e{};
  std::snprintf(e.name, sizeof e.name, "test.weight");
  e.dtype = uint32_t(DType::F32);
  e.ndim = 1;
  e.shape[0] = kAlign / 4;
  e.offset = 2 * kAlign;
  e.bytes = kAlign;
  e.kind = uint32_t(SliceKind::Replicated);
  for (size_t i = 0; i < kAlign; i++) f[2 * kAlign + i] = uint8_t(i * 7 + rank);
  e.crc32c = crc32c(&f[2 * kAlign], kAlign);
  std::memcpy(&f[kAlign], &e, sizeof e);
  h.toc_crc32c = crc32c(&e, sizeof e);
  std::memcpy(&f[0], &h, sizeof h);
  FILE* out = std::fopen(path.c_str(), "wb");
  if (!out || std::fwrite(f.data(), 1, f.size(), out) != f.size() || std::fclose(out) != 0)
    die("%s: cannot write", path.c_str());
}

// `path` opened as rank `rank` of 4 (-1: no rank check) in a child process, which dies if refused
struct Outcome {
  bool opened = false, aborted = false;
  std::string err;  // the child's stderr
};
Outcome open_as(const std::string& path, int rank) {
  int p[2];
  if (pipe(p)) die("pipe");
  std::fflush(stdout);
  const pid_t c = fork();
  if (c < 0) die("fork");
  if (c == 0) {
    dup2(p[1], 2);
    close(p[0]);
    WeightFile wf(path, rank, 4);
    std::_Exit(0);
  }
  close(p[1]);
  Outcome o;
  char buf[512];
  for (ssize_t n; (n = read(p[0], buf, sizeof buf)) > 0;) o.err.append(buf, size_t(n));
  close(p[0]);
  int st = 0;
  waitpid(c, &st, 0);
  o.opened = WIFEXITED(st) && WEXITSTATUS(st) == 0;
  o.aborted = WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT;
  while (!o.err.empty() && o.err.back() == '\n') o.err.pop_back();
  return o;
}

// opened as its own rank; refused as each other rank, the message naming the file and both ranks
void check_file(const std::string& path, int own) {
  for (int r = 0; r < 4; r++) {
    const Outcome o = open_as(path, r);
    if (r == own) {
      expect(o.opened, path + " opened as rank " + std::to_string(r) + (o.err.empty() ? "" : ": " + o.err));
      continue;
    }
    const bool named = o.err.find(path) != std::string::npos &&
                       o.err.find("cut for rank " + std::to_string(own) + " of 4") != std::string::npos &&
                       o.err.find("this process is rank " + std::to_string(r) + " of 4") != std::string::npos;
    expect(o.aborted && named, "refused as rank " + std::to_string(r) + ": " + o.err);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "--make" && argc == 4) {
    make_file(argv[2], uint32_t(std::atoi(argv[3])), 4);
    std::printf("%s: a synthetic weight file of rank %s of 4\n", argv[2], argv[3]);
    return 0;
  }
  if (mode == "--file" && argc == 4) {
    check_file(argv[2], std::atoi(argv[3]));
  } else if (argc <= 2 && mode.rfind("--", 0) != 0) {
    Header h{};
    h.rank = 2, h.world = 4;
    expect(rank_mismatch(h, 2, 4).empty(), "a header of rank 2 of 4 is rank 2 of 4's");
    expect(rank_mismatch(h, 1, 4) == "the file was cut for rank 2 of 4, but this process is rank 1 of 4",
           "and not rank 1 of 4's: " + rank_mismatch(h, 1, 4));
    expect(!rank_mismatch(h, 2, 8).empty(), "nor rank 2 of 8's: " + rank_mismatch(h, 2, 8));
    const std::string dir = argc > 1 ? argv[1] : ".";
    for (int r = 0; r < 4; r++) {
      const std::string path = dir + "/dsw_rank_test." + std::to_string(r) + ".dsw";
      make_file(path, uint32_t(r), 4);
      check_file(path, r);
      const Outcome any = open_as(path, -1);
      expect(any.opened, path + " opened with no rank given (no check)");
      unlink(path.c_str());
    }
  } else {
    die("usage: dsw_rank_test [DIR] | --file FILE RANK | --make FILE RANK");
  }
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
