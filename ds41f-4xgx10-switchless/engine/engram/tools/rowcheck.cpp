// Checks a kit Engram shard against the checkpoint: the shard's records for random rows (and the
// first and last) must equal the checkpoint's weight row and scale row byte for byte.
//   engram_rowcheck <shard.bin> <checkpoint-dir> [samples=20000]
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "engram/engram.h"
#include "weights/safetensors.h"

using namespace engram;

int main(int argc, char** argv) {
  if (argc < 3) die("usage: %s <shard.bin> <checkpoint-dir> [samples]", argv[0]);
  const long samples = argc > 3 ? std::atol(argv[3]) : 20000;
  const int sfd = open(argv[1], O_RDONLY);
  if (sfd < 0) die("open %s", argv[1]);
  uint64_t h[6];
  if (pread(sfd, h, sizeof(h), 0) != sizeof(h) || h[0] != 0x31344E4531565344ull || h[5] != uint64_t(kRowBytes))
    die("%s: not a kit Engram shard", argv[1]);
  const uint64_t layer = h[1], lo = h[2], hi = h[3], rows = h[4];
  const dsw::Checkpoint ck = dsw::Checkpoint::open(argv[2]);
  const std::string pre = "layers." + std::to_string(layer) + ".engram.embed.";
  const dsw::SrcTensor& w = ck.at(pre + "weight");
  const dsw::SrcTensor& s = ck.at(pre + "scale");
  if (w.shape[0] != rows || w.shape[1] != uint64_t(kWeightBytes) || s.shape[1] != uint64_t(kRowBytes - kWeightBytes))
    die("checkpoint shapes do not match the shard");
  const int wfd = open(ck.shards[w.shard].c_str(), O_RDONLY), cfd = open(ck.shards[s.shard].c_str(), O_RDONLY);
  if (wfd < 0 || cfd < 0) die("open checkpoint shard");
  std::printf("%s: layer %llu rows [%llu, %llu) of %llu vs %s\n", argv[1], (unsigned long long)layer,
              (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)rows, ck.shards[w.shard].c_str());
  uint64_t x = 0x243F6A8885A308D3ull ^ layer ^ (lo << 1);
  long bad = 0;
  for (long i = 0; i < samples + 3; i++) {
    uint64_t row;
    if (i == 0) row = lo;
    else if (i == 1) row = lo + 1;
    else if (i == 2) row = hi - 1;
    else {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      row = lo + x % (hi - lo);
    }
    uint8_t rec[kRowBytes], ref[kRowBytes];
    if (pread(sfd, rec, kRowBytes, off_t(4096 + (row - lo) * kRowBytes)) != kRowBytes ||
        pread(wfd, ref, kWeightBytes, off_t(w.offset + row * kWeightBytes)) != kWeightBytes ||
        pread(cfd, ref + kWeightBytes, kRowBytes - kWeightBytes, off_t(s.offset + row * (kRowBytes - kWeightBytes))) !=
            kRowBytes - kWeightBytes)
      die("short read at row %llu", (unsigned long long)row);
    if (std::memcmp(rec, ref, kRowBytes) != 0 && bad++ < 5) std::printf("  row %llu differs\n", (unsigned long long)row);
  }
  std::printf("%ld rows compared: %s\n", samples + 3, bad ? "MISMATCH" : "identical");
  return bad ? 1 : 0;
}
