// Random 264-byte record reads from a kit Engram shard: O_DIRECT through io_uring, the smallest
// 512-byte-aligned window around each record, at a range of queue depths.
//   engram_iobench <shard.bin> [secs-per-depth=4] [depths=1,2,4,8,16,32,64,128] [wait]
//   engram_iobench <shard.bin> gap [gaps-us=0,100,300,1000,3000,10000,30000] [reads-per-gap=300] [batch=1]
// "wait" blocks in io_uring_enter for completions instead of spinning on the completion ring.
// "gap" issues single reads separated by an idle gap (busy-waited), as decode steps do; with batch > 1
// each read goes out with batch - 1 others and its own completion is timed.
#include <fcntl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "engram/engram.h"
#include "engram/uring.h"

using namespace engram;

static uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  if (argc < 2) die("usage: %s <shard.bin> [secs] [depths] [wait] | <shard.bin> gap [gaps-us] [reads] [batch]",
                     argv[0]);
  const bool gap_mode = argc > 2 && std::string(argv[2]) == "gap";
  const double secs = argc > 2 && !gap_mode ? std::atof(argv[2]) : 4.0;
  std::vector<int> depths;
  {
    std::string s = argc > 3 ? argv[3] : gap_mode ? "0,100,300,1000,3000,10000,30000" : "1,2,4,8,16,32,64,128";
    for (size_t i = 0; i < s.size();) {
      size_t j = s.find(',', i);
      if (j == std::string::npos) j = s.size();
      depths.push_back(std::atoi(s.substr(i, j - i).c_str()));
      i = j + 1;
    }
  }
  const bool wait = !gap_mode && argc > 4 && std::string(argv[4]) == "wait";
  const int gap_reads = gap_mode && argc > 4 ? std::atoi(argv[4]) : 300;
  const int gap_batch = gap_mode && argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;

  const int fd = open(argv[1], O_RDONLY | O_DIRECT);
  if (fd < 0) die("open %s: %s", argv[1], strerror(errno));
  auto* hdr = static_cast<uint8_t*>(aligned_alloc(4096, 4096));
  if (pread(fd, hdr, 4096, 0) != 4096) die("header read");
  uint64_t h[6];
  std::memcpy(h, hdr, sizeof(h));
  if (h[0] != 0x31344E4531565344ull || h[5] != uint64_t(kRowBytes)) die("%s: not a kit Engram shard", argv[1]);
  const uint64_t lo = h[2], hi = h[3], count = hi - lo;
  std::printf("%s: layer %llu, rows [%llu, %llu) of %llu, %s\n", argv[1], (unsigned long long)h[1],
              (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)h[4],
              wait ? "blocking for completions" : "spinning on the completion ring");

  const int max_qd = gap_mode ? gap_batch : *std::max_element(depths.begin(), depths.end());
  Uring ring(unsigned(max_qd < 8 ? 8 : max_qd));
  ring.register_files(&fd, 1);
  const size_t slot = 1024;
  auto* bufs = static_cast<uint8_t*>(aligned_alloc(4096, slot * max_qd));
  iovec iov{bufs, slot * size_t(max_qd)};
  ring.register_buffers(&iov, 1);

  uint64_t x = 0x9E3779B97F4A7C15ull;
  auto rnd = [&]() {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
  };
  std::vector<uint64_t> start(max_qd), rec_at(max_qd), win(max_qd);
  auto issue = [&](int i) {
    const uint64_t off = 4096 + (rnd() % count) * kRowBytes;
    const uint64_t a = off & ~uint64_t(511), e = (off + kRowBytes + 511) & ~uint64_t(511);
    io_uring_sqe* s = ring.sqe();
    if (!s) die("ring full");
    ring.prep_read_fixed(s, 0, bufs + size_t(i) * slot, unsigned(e - a), a, 0, uint64_t(i));
    rec_at[i] = off - a;
    win[i] = e - a;
    start[i] = now_ns();
  };

  if (gap_mode) {
    std::printf("%8s %8s %8s %8s %8s %8s %10s\n", "gap_us", "p10_us", "p50_us", "p90_us", "p99_us", "max_us",
                ">500us");
    for (int gap : depths) {
      std::vector<double> lat;
      uint64_t last = now_ns();
      for (int k = 0; k < gap_reads; k++) {
        while (now_ns() - last < uint64_t(gap) * 1000) {
        }
        for (int i = 0; i < gap_batch; i++) issue(i);
        ring.submit();
        for (int left = gap_batch; left > 0;) {
          const unsigned n = ring.cq_ready();
          if (!n) continue;
          const uint64_t t = now_ns();
          for (unsigned j = 0; j < n; j++) {
            const io_uring_cqe& c = ring.cqe(j);
            const int i = int(c.user_data);
            if (c.res < int(rec_at[i] + kRowBytes))
              die("read returned %d, wanted %llu", c.res, (unsigned long long)win[i]);
            if (i == 0) lat.push_back(double(t - start[0]) * 1e-3);
          }
          ring.cq_advance(n);
          left -= int(n);
        }
        last = now_ns();
      }
      std::sort(lat.begin(), lat.end());
      auto pct = [&](double q) { return lat[size_t(q * double(lat.size() - 1))]; };
      const size_t slow = size_t(lat.end() - std::upper_bound(lat.begin(), lat.end(), 500.0));
      std::printf("%8d %8.1f %8.1f %8.1f %8.1f %8.1f %9.1f%%\n", gap, pct(0.1), pct(0.5), pct(0.9), pct(0.99),
                  lat.back(), 100.0 * double(slow) / double(lat.size()));
    }
    return 0;
  }
  std::printf("%5s %10s %8s %8s %8s %8s %8s %9s\n", "QD", "IOPS", "MB/s", "mean_us", "p50_us", "p99_us", "p999_us",
              "bytes/rd");
  for (int qd : depths) {
    std::vector<uint32_t> lat;  // 10 ns units
    lat.reserve(size_t(secs * 600000));
    std::vector<int> freed;
    uint64_t bytes = 0, done = 0, scale_sum = 0;
    for (int i = 0; i < qd; i++) issue(i);
    ring.submit();
    const uint64_t t0 = now_ns(), t_end = t0 + uint64_t(secs * 1e9);
    uint64_t t = t0;
    while (t < t_end) {
      unsigned n = ring.cq_ready();
      if (!n) {
        if (wait) ring.submit(1);
        continue;
      }
      t = now_ns();
      freed.clear();
      for (unsigned k = 0; k < n; k++) {
        const io_uring_cqe& c = ring.cqe(k);
        const int i = int(c.user_data);
        if (c.res < int(rec_at[i] + kRowBytes))
          die("read returned %d, wanted %llu", c.res, (unsigned long long)win[i]);
        lat.push_back(uint32_t((t - start[i]) / 10));
        bytes += uint64_t(c.res);
        scale_sum += bufs[size_t(i) * slot + rec_at[i] + kWeightBytes];  // the record's first scale
        freed.push_back(i);
      }
      ring.cq_advance(n);
      done += n;
      for (int i : freed) issue(i);
      ring.submit();
    }
    const double el = double(now_ns() - t0) * 1e-9;
    for (int inflight = qd; inflight > 0;) {  // drain before the next depth
      unsigned n = ring.cq_ready();
      if (!n) {
        ring.submit(1);
        continue;
      }
      ring.cq_advance(n);
      inflight -= int(n);
    }
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double q) { return lat.empty() ? 0.0 : lat[size_t(q * double(lat.size() - 1))] / 100.0; };
    double mean = 0;
    for (uint32_t v : lat) mean += v;
    mean = lat.empty() ? 0 : mean / double(lat.size()) / 100.0;
    std::printf("%5d %10.0f %8.1f %8.1f %8.1f %8.1f %8.1f %9.0f\n", qd, double(done) / el, double(bytes) / el / 1e6,
                mean, pct(0.5), pct(0.99), pct(0.999), done ? double(bytes) / double(done) : 0.0);
    if (done && scale_sum == 0) std::printf("  warning: every scale byte read was 0\n");
  }
  return 0;
}
