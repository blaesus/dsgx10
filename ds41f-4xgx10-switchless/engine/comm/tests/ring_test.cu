// Ring collectives across the four boxes: correctness, payload integrity, latency.
// Every rank runs the same command (engine/comm/run4.sh does that); rank 0's output is the report.
// usage: ring_test [--fleet FILE] [--rank r] [--cap bytes] [--integrity N] [--graph-steps N] [tests...]
//        [--bulk-max bytes] [--bulk-blocks n] [--trace n] [--split-from bytes] [--relay-domain d] [--mode proxy|none]
//        [--rs-slow-pct p] [--race-us us]
//        tests: correct race integrity bulk-integrity latency bulk bulk2 graph wake loop
//        (default: correct race integrity latency graph; bulk-integrity runs --integrity / 30 bulk all-reduces;
//        race: the pipelined kernels with one block stalled --race-us (2000) in the race's window, needs the test
//        hooks of ring_test's comm library)
#include <cuda_bf16.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "comm/collectives.h"
#include "comm/ring.h"
#include "comm/ring_dev.cuh"
#include "common/check.h"

using comm::DevChannel;
using comm::Ring;

namespace {

__host__ __device__ inline uint32_t mix(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return (uint32_t)x;
}

// A value with a random sign, mantissa and an exponent spread over ~2^±12, so fixed-order fp32
// sums really depend on the order.
float test_value(int rank, size_t i, uint32_t salt) {
  uint32_t h = mix(((uint64_t)salt << 40) ^ ((uint64_t)rank << 32) ^ i);
  float m = 1.0f + (h & 0xffff) / 65536.0f;
  int e = (int)((h >> 16) % 25) - 12;
  return ((h >> 31) ? -1.0f : 1.0f) * std::ldexp(m, e);
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void sync_or_die(Ring& ring, cudaStream_t st, double timeout_s, const char* what) {
  double t0 = now_s();
  for (;;) {
    cudaError_t e = cudaStreamQuery(st);
    if (e == cudaSuccess) return;
    if (e != cudaErrorNotReady) die("%s: %s", what, cudaGetErrorString(e));
    if (now_s() - t0 > timeout_s) {
      std::fprintf(stderr, "rank %d: %s timed out\n%s", ring.rank(), what, ring.describe().c_str());
      std::fflush(stderr);
      _exit(2);
    }
    usleep(50);
  }
}

uint64_t fnv(const void* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
  return h;
}

bool same_on_all_ranks(Ring& ring, uint64_t h) {
  uint64_t all[4];
  ring.allgather_host(&h, all, sizeof h);
  return all[0] == all[1] && all[1] == all[2] && all[2] == all[3];
}

// ------------------------------------------------------------------ correctness

// The fusion pattern, as an engine uses it: one kernel (say a GEMV epilogue) writes its partial
// straight into the send buffer and publishes; the next kernel waits and reduces in its prologue.
__global__ void produce_kernel(DevChannel ch, const uint4* in, uint32_t nbytes) {
  const uint64_t s = comm::next_seq(ch);
  uint4* sb = (uint4*)comm::send_buf(ch, s);
  const uint32_t chunks = nbytes / 16;
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (uint32_t c = t0; c < chunks; c += stride) sb[c] = in[c];
  comm::post_grid(ch, s, nbytes);
}

__global__ void consume_kernel(DevChannel ch, uint4* out, uint32_t nbytes) {
  const uint64_t s = comm::next_seq(ch);  // the same s: nothing has called finish_grid since
  comm::wait_all(ch, s);
  const uint32_t chunks = nbytes / 16;
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (uint32_t c = t0; c < chunks; c += stride) {
    float f[8];
    comm::sum8_bf16(ch, s, (size_t)c * 16, f);
    uint32_t w[4];
    for (int j = 0; j < 4; j++) {
      __nv_bfloat162 b = __floats2bfloat162_rn(f[2 * j], f[2 * j + 1]);
      w[j] = *(uint32_t*)&b;
    }
    out[c] = make_uint4(w[0], w[1], w[2], w[3]);
  }
  comm::finish_grid(ch, s);
}

void split_allreduce_bf16(const DevChannel& ch, const void* in, void* out, size_t bytes, cudaStream_t st) {
  // different launch shapes on purpose: producer and consumer need not match
  produce_kernel<<<comm::grid_for(bytes), comm::block_for(bytes), 0, st>>>(ch, (const uint4*)in, (uint32_t)bytes);
  consume_kernel<<<1, 256, 0, st>>>(ch, (uint4*)out, (uint32_t)bytes);
}

int test_correct(Ring& ring, cudaStream_t st) {
  const DevChannel& ch = ring.dev();
  const int r = ring.rank();
  int failures = 0;
  enum Kind { kArBf16, kArF32, kGather, kRsBf16, kRsF32, kBulkBf16, kBulkF32, kSplitBf16, kRsBulkBf16, kRsBulkF32 };
  static const char* names[] = {"allreduce bf16", "allreduce f32", "allgather",     "redscatter bf16",
                                "redscatter f32", "bulk ar bf16",  "bulk ar f32",   "split ar bf16",
                                "bulk rs bf16",   "bulk rs f32"};
  const size_t cap = ch.cap;
  struct Case {
    int kind;
    size_t bytes;
  };
  std::vector<Case> cases;
  for (size_t b : {(size_t)16, (size_t)64, (size_t)128, (size_t)1024, (size_t)10240, (size_t)20480, (size_t)61440,
                   (size_t)262144, cap})
    for (int k : {kArBf16, kArF32, kGather, kRsBf16, kRsF32, kSplitBf16})
      if (b <= cap && (k < kRsBf16 || b % 64 == 0)) cases.push_back({k, b});
  for (size_t b : {(size_t)64, (size_t)10240, cap, 2 * cap + 4160, 6 * cap})
    for (int k : {kBulkBf16, kBulkF32}) cases.push_back({k, b});
  for (size_t b : {cap + 16, 3 * cap + 4112}) cases.push_back({kGather, b});  // chunked all-gathers
  for (size_t b : {(size_t)256, (size_t)10240, cap, 2 * cap + 4160, 6 * cap + 192})
    for (int k : {kRsBulkBf16, kRsBulkF32}) cases.push_back({k, b});
  size_t max_bytes = 4 * cap;
  for (const Case& c : cases) max_bytes = std::max(max_bytes, c.kind == kGather ? 4 * c.bytes : c.bytes);
  void *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, max_bytes));
  std::vector<uint8_t> h_in(max_bytes), h_out(max_bytes);
  uint32_t salt = 1;
  for (const Case& cs : cases) {
    const int kind = cs.kind;
    const size_t bytes = cs.bytes;
    const bool bf16 = kind == kArBf16 || kind == kRsBf16 || kind == kBulkBf16 || kind == kSplitBf16 || kind == kRsBulkBf16;
    const bool rs = kind == kRsBf16 || kind == kRsF32 || kind == kRsBulkBf16 || kind == kRsBulkF32;
    const size_t n = bf16 ? bytes / 2 : bytes / 4;
    salt++;
    if (bf16) {
      auto* p = (__nv_bfloat16*)h_in.data();
      for (size_t i = 0; i < n; i++) p[i] = __float2bfloat16(test_value(r, i, salt));
    } else if (kind != kGather) {
      auto* p = (float*)h_in.data();
      for (size_t i = 0; i < n; i++) p[i] = test_value(r, i, salt);
    } else {
      auto* p = (uint32_t*)h_in.data();
      for (size_t i = 0; i < n; i++) p[i] = mix(((uint64_t)salt << 40) ^ ((uint64_t)r << 32) ^ i);
    }
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice, st));
    auto* bi = (const __nv_bfloat16*)d_in;
    auto* bo = (__nv_bfloat16*)d_out;
    auto* fi = (const float*)d_in;
    auto* fo = (float*)d_out;
    switch (kind) {
      case kArBf16: comm::allreduce_bf16(ch, bi, bo, n, st); break;
      case kArF32: comm::allreduce_f32(ch, fi, fo, n, st); break;
      case kGather: comm::allgather(ch, d_in, d_out, bytes, st); break;
      case kRsBf16: comm::reduce_scatter_bf16(ch, bi, bo, n, st); break;
      case kRsF32: comm::reduce_scatter_f32(ch, fi, fo, n, st); break;
      case kBulkBf16: comm::allreduce_bulk_bf16(ch, bi, bo, n, st); break;
      case kBulkF32: comm::allreduce_bulk_f32(ch, fi, fo, n, st); break;
      case kSplitBf16: split_allreduce_bf16(ch, d_in, d_out, bytes, st); break;
      case kRsBulkBf16: comm::reduce_scatter_bulk_bf16(ch, bi, bo, n, st); break;
      case kRsBulkF32: comm::reduce_scatter_bulk_f32(ch, fi, fo, n, st); break;
    }
    const size_t out_n = kind == kGather ? 4 * n : rs ? n / 4 : n;
    const size_t first = rs ? r * (n / 4) : 0;  // element of the full vector that out[0] is
    const size_t out_bytes = out_n * (bf16 ? 2 : 4);
    CUDA_CHECK(cudaMemcpyAsync(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, st));
    sync_or_die(ring, st, 30, "correct");
    size_t bad = 0;
    for (size_t o = 0; o < out_n; o++) {
      const size_t i = first + o;
      if (bf16) {
        float acc = __bfloat162float(__float2bfloat16(test_value(0, i, salt)));
        for (int k = 1; k < 4; k++) acc += __bfloat162float(__float2bfloat16(test_value(k, i, salt)));
        __nv_bfloat16 want = __float2bfloat16_rn(acc);
        if (memcmp(&want, h_out.data() + 2 * o, 2)) bad++;
      } else if (kind != kGather) {
        float acc = test_value(0, i, salt);
        for (int k = 1; k < 4; k++) acc += test_value(k, i, salt);
        if (memcmp(&acc, h_out.data() + 4 * o, 4)) bad++;
      } else {
        size_t k = i / n, j = i % n;
        uint32_t want = mix(((uint64_t)salt << 40) ^ ((uint64_t)k << 32) ^ j);
        if (memcmp(&want, h_out.data() + 4 * o, 4)) bad++;
      }
    }
    // a reduce-scatter's output differs by rank, so only the reference check applies
    const uint64_t h = fnv(h_out.data(), out_bytes);
    bool same = true;
    if (rs) {
      uint64_t all[4];
      ring.allgather_host(&h, all, sizeof h);
    } else {
      same = same_on_all_ranks(ring, h);
    }
    if (bad || !same) {
      failures++;
      std::printf("rank %d FAIL %-15s %9zu B: %zu wrong elements, identical on all ranks: %s\n", r, names[kind], bytes,
                  bad, same ? "yes" : "NO");
    } else if (r == 0) {
      std::printf("  ok %-15s %9zu B (exact%s)\n", names[kind], bytes, rs ? "" : ", identical on all ranks");
    }
  }
  // reduce-scatters by rows of a column block: in [4 rows][ld], block [c0, c0 + cols); out [rows][ld_out]
  for (int f32 = 0; f32 < 2; f32++)
    for (int shape = 0; shape < 3; shape++) {
      const int dims[3][4] = {{8, 64, 128, 16}, {37, 1280, 5120, 2560}, {300, 2560, 5120, 1280}};  // rows cols ld c0
      const int rows = dims[shape][0], cols = dims[shape][1], ld = dims[shape][2], c0 = dims[shape][3];
      const int ld_out = cols + 64, es = f32 ? 4 : 2;
      const size_t nin = (size_t)4 * rows * ld, nout = (size_t)rows * ld_out;
      uint8_t *ri, *ro;
      CUDA_CHECK(cudaMalloc(&ri, nin * es));
      CUDA_CHECK(cudaMalloc(&ro, nout * es));
      CUDA_CHECK(cudaMemset(ro, 0, nout * es));
      std::vector<uint8_t> hin(nin * es), hout(nout * es);
      salt++;
      for (size_t i = 0; i < nin; i++) {
        const float v = test_value(r, i, salt);
        if (f32) ((float*)hin.data())[i] = v;
        else ((__nv_bfloat16*)hin.data())[i] = __float2bfloat16(v);
      }
      CUDA_CHECK(cudaMemcpyAsync(ri, hin.data(), nin * es, cudaMemcpyHostToDevice, st));
      if (f32)
        comm::reduce_scatter_rows_f32(ch, (const float*)ri + c0, ld, (float*)ro, ld_out, rows, cols, st);
      else
        comm::reduce_scatter_rows_bf16(ch, (const __nv_bfloat16*)ri + c0, ld, (__nv_bfloat16*)ro, ld_out, rows, cols, st);
      CUDA_CHECK(cudaMemcpyAsync(hout.data(), ro, nout * es, cudaMemcpyDeviceToHost, st));
      sync_or_die(ring, st, 30, "reduce_scatter_rows");
      size_t bad = 0;
      for (int rr = 0; rr < rows; rr++)
        for (int c = 0; c < ld_out; c++) {
          const size_t o = (size_t)rr * ld_out + c;
          if (c >= cols) {  // untouched
            const bool zero = f32 ? ((float*)hout.data())[o] == 0.f : __bfloat162float(((__nv_bfloat16*)hout.data())[o]) == 0.f;
            if (!zero) bad++;
            continue;
          }
          const size_t i = (size_t)(r * rows + rr) * ld + c0 + c;
          if (f32) {
            float acc = test_value(0, i, salt);
            for (int k = 1; k < 4; k++) acc += test_value(k, i, salt);
            if (memcmp(&acc, hout.data() + 4 * o, 4)) bad++;
          } else {
            float acc = __bfloat162float(__float2bfloat16(test_value(0, i, salt)));
            for (int k = 1; k < 4; k++) acc += __bfloat162float(__float2bfloat16(test_value(k, i, salt)));
            __nv_bfloat16 want = __float2bfloat16_rn(acc);
            if (memcmp(&want, hout.data() + 2 * o, 2)) bad++;
          }
        }
      uint64_t hh = fnv(hout.data(), nout * es), all[4];
      ring.allgather_host(&hh, all, sizeof hh);
      if (bad) {
        failures++;
        std::printf("rank %d FAIL rs rows %s %d x %d (ld %d, c0 %d): %zu wrong\n", r, f32 ? "f32" : "bf16", rows, cols, ld, c0, bad);
      } else if (r == 0) {
        std::printf("  ok rs rows %-4s %4d x %4d of ld %d from column %d (exact)\n", f32 ? "f32" : "bf16", rows, cols, ld, c0);
      }
      cudaFree(ri);
      cudaFree(ro);
    }
  // multi-segment all-gathers, separate and in place: out_g[k][j] = pattern(k, g, j)
  for (int inplace = 0; inplace < 2; inplace++)
    for (int shape = 0; shape < 3; shape++) {
      const size_t sz[3][3] = {{16, 48, 1024}, {cap / 3 & ~15ull, 160, cap + 4096}, {5 * cap + 16, 3 * cap / 4 & ~15ull, 48}};
      size_t tot_in = 0, tot_out = 0;
      for (int g = 0; g < 3; g++) tot_in += sz[shape][g], tot_out += 4 * sz[shape][g];
      uint8_t *mi, *mo;
      CUDA_CHECK(cudaMalloc(&mi, tot_in));
      CUDA_CHECK(cudaMalloc(&mo, tot_out));
      std::vector<uint32_t> hin(tot_in / 4), hout(tot_out / 4);
      comm::AgSeg segs[3];
      size_t oi = 0, oo = 0;
      salt++;
      for (int g = 0; g < 3; g++) {
        const size_t b = sz[shape][g];
        for (size_t j = 0; j < b / 4; j++) hin[oi / 4 + j] = mix(((uint64_t)salt << 40) ^ ((uint64_t)r << 32) ^ ((uint64_t)g << 28) ^ j);
        segs[g] = {inplace ? (const void*)(mo + oo + r * b) : (const void*)(mi + oi), mo + oo, b};
        oi += b, oo += 4 * b;
      }
      if (inplace) {  // each segment's own slot holds the input
        oi = 0, oo = 0;
        for (int g = 0; g < 3; g++) {
          const size_t b = sz[shape][g];
          CUDA_CHECK(cudaMemcpyAsync(mo + oo + r * b, hin.data() + oi / 4, b, cudaMemcpyHostToDevice, st));
          oi += b, oo += 4 * b;
        }
      } else {
        CUDA_CHECK(cudaMemcpyAsync(mi, hin.data(), tot_in, cudaMemcpyHostToDevice, st));
      }
      comm::allgather_multi(ch, segs, 3, st);
      CUDA_CHECK(cudaMemcpyAsync(hout.data(), mo, tot_out, cudaMemcpyDeviceToHost, st));
      sync_or_die(ring, st, 30, "allgather_multi");
      size_t bad = 0;
      oo = 0;
      for (int g = 0; g < 3; g++) {
        const size_t b = sz[shape][g];
        for (int k = 0; k < 4; k++)
          for (size_t j = 0; j < b / 4; j++)
            if (hout[(oo + k * b) / 4 + j] != mix(((uint64_t)salt << 40) ^ ((uint64_t)k << 32) ^ ((uint64_t)g << 28) ^ j)) bad++;
        oo += 4 * b;
      }
      const bool same = same_on_all_ranks(ring, fnv(hout.data(), tot_out));
      if (bad || !same) {
        failures++;
        std::printf("rank %d FAIL allgather_multi %zu+%zu+%zu B%s: %zu wrong words, identical: %s\n", r, sz[shape][0],
                    sz[shape][1], sz[shape][2], inplace ? " in place" : "", bad, same ? "yes" : "NO");
      } else if (r == 0) {
        std::printf("  ok allgather multi %zu+%zu+%zu B%s\n", sz[shape][0], sz[shape][1], sz[shape][2],
                    inplace ? " (in place)" : "");
      }
      cudaFree(mi);
      cudaFree(mo);
    }
  // argmax rows: ties across ranks must resolve to the lower index everywhere
  {
    const int rows = 8;
    comm::ArgMax in[rows], out[rows];
    for (int i = 0; i < rows; i++) in[i] = {float((i * 7 + r * 3) % 5), 1000 * r + i};
    comm::ArgMax *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, sizeof in));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof out));
    CUDA_CHECK(cudaMemcpyAsync(d_a, in, sizeof in, cudaMemcpyHostToDevice, st));
    comm::argmax_rows(ch, d_a, d_b, rows, st);
    CUDA_CHECK(cudaMemcpyAsync(out, d_b, sizeof out, cudaMemcpyDeviceToHost, st));
    sync_or_die(ring, st, 20, "argmax");
    int bad = 0;
    for (int i = 0; i < rows; i++) {
      comm::ArgMax best{-1e30f, 0x7fffffff};
      for (int k = 0; k < 4; k++) {
        comm::ArgMax a{float((i * 7 + k * 3) % 5), 1000 * k + i};
        if (a.v > best.v || (a.v == best.v && a.i < best.i)) best = a;
      }
      if (best.v != out[i].v || best.i != out[i].i) bad++;
    }
    bool same = same_on_all_ranks(ring, fnv(out, sizeof out));
    if (bad || !same) {
      failures++;
      std::printf("rank %d FAIL argmax: %d wrong rows, identical: %s\n", r, bad, same ? "yes" : "NO");
    } else if (r == 0) {
      std::printf("  ok argmax rows (ties to the lower index, identical on all ranks)\n");
    }
    cudaFree(d_a);
    cudaFree(d_b);
  }
  cudaFree(d_in);
  cudaFree(d_out);
  return failures;
}

// ------------------------------------------------------------------ integrity

__device__ __forceinline__ uint32_t word_pattern(uint64_t s, int k, uint32_t w) {
  return mix((s << 34) ^ ((uint64_t)k << 32) ^ w);
}

// Each rank publishes a pattern that depends on (seq, rank, word) and checks every word of every
// contribution it receives: stale, torn or misplaced data shows up as a mismatch.
__global__ void integrity_kernel(DevChannel ch, uint32_t nbytes, unsigned long long* err, unsigned long long* info) {
  const uint64_t s = comm::next_seq(ch);
  uint4* sb = (uint4*)comm::send_buf(ch, s);
  const uint32_t chunks = nbytes / 16;
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (uint32_t c = t0; c < chunks; c += stride)
    sb[c] = make_uint4(word_pattern(s, ch.rank, 4 * c), word_pattern(s, ch.rank, 4 * c + 1),
                       word_pattern(s, ch.rank, 4 * c + 2), word_pattern(s, ch.rank, 4 * c + 3));
  // every third collective (sizes a multiple of 64) is a reduce-scatter: check only this rank's quarter
  const bool rs = s % 3 == 1 && nbytes % 64 == 0;
  comm::post_grid(ch, s, nbytes, rs ? comm::kOpReduceScatter : comm::kOpAllExchange);
  comm::wait_all(ch, s);
  const uint32_t c0 = rs ? ch.rank * (chunks / 4) : 0, c1 = rs ? c0 + chunks / 4 : chunks;
  for (int k = 0; k < comm::kRanks; k++) {
    const uint8_t* p = comm::src(ch, s, k);
    for (uint32_t c = c0 + t0; c < c1; c += stride) {
      uint4 v = comm::ld_cv16(p + (size_t)c * 16);
      uint32_t got[4] = {v.x, v.y, v.z, v.w};
      for (int j = 0; j < 4; j++) {
        uint32_t want = word_pattern(s, k, 4 * c + j);
        if (got[j] != want && atomicAdd(err, 1ull) == 0) {
          info[0] = s;
          info[1] = k;
          info[2] = 4 * c + j;
          info[3] = got[j];
          info[4] = want;
          info[5] = nbytes;
        }
      }
    }
  }
  comm::finish_grid(ch, s);
}

int test_integrity(Ring& ring, cudaStream_t st, long total) {
  const DevChannel& ch = ring.dev();
  std::mt19937_64 rng(12345);  // same sizes on every rank
  const int nodes = 250;
  std::vector<uint32_t> sizes(nodes);
  double sum = 0;
  for (auto& s : sizes) {
    double lg = std::uniform_real_distribution<double>(4, std::log2((double)ch.cap))(rng);
    s = std::max<uint32_t>(16, ((uint32_t)std::pow(2.0, lg)) & ~15u);
    sum += s;
  }
  unsigned long long *d_err, *d_info;
  CUDA_CHECK(cudaMalloc(&d_err, 8));
  CUDA_CHECK(cudaMalloc(&d_info, 64));
  CUDA_CHECK(cudaMemset(d_err, 0, 8));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (uint32_t s : sizes) integrity_kernel<<<comm::grid_for(s), comm::block_for(s), 0, st>>>(ch, s, d_err, d_info);
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  const long reps = std::max(1L, total / nodes);
  ring.barrier();
  double t0 = now_s();
  for (long i = 0; i < reps; i++) {
    CUDA_CHECK(cudaGraphLaunch(ge, st));
    if (i % 64 == 63) sync_or_die(ring, st, 60, "integrity");
  }
  sync_or_die(ring, st, 60, "integrity");
  double dt = now_s() - t0;
  unsigned long long err = 0, info[6] = {0};
  CUDA_CHECK(cudaMemcpy(&err, d_err, 8, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(info, d_info, sizeof info, cudaMemcpyDeviceToHost));
  unsigned long long all_err[4];
  ring.allgather_host(&err, all_err, 8);
  if (ring.rank() == 0)
    std::printf("  integrity: %ld collectives (sizes 16 B to %u B, mean %.0f B), %.1f s, %.1f us each; "
                "mismatched words per rank: %llu %llu %llu %llu\n",
                reps * nodes, ch.cap, sum / nodes, dt, dt * 1e6 / (reps * nodes), all_err[0], all_err[1],
                all_err[2], all_err[3]);
  if (err)
    std::printf("rank %d first mismatch: seq %llu from rank %llu word %llu got %08llx want %08llx (size %llu)\n",
                ring.rank(), info[0], info[1], info[2], info[3], info[4], info[5]);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  cudaFree(d_err);
  cudaFree(d_info);
  return err ? 1 : 0;
}

// Bulk all-reduces of random sizes (several chunks each), checked on the GPU: every element is a
// small integer that depends on (round, rank, index), so sums are exact and stale or misplaced data
// shows. The round counter lives in device memory and advances after each check, so graph replays
// use new values.
__device__ __forceinline__ float bulk_value(uint32_t round, int rank, uint32_t j) {
  return (float)((int)(mix(((uint64_t)round << 34) ^ ((uint64_t)rank << 32) ^ j) % 17u) - 8);
}

__global__ void bulk_fill_kernel(const uint32_t* round, int rank, __nv_bfloat16* in, uint32_t n) {
  const uint32_t r = *round;
  for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x)
    in[j] = __float2bfloat16(bulk_value(r, rank, j));
}

__global__ void bulk_check_kernel(uint32_t* round, const __nv_bfloat16* out, uint32_t n, unsigned long long* err,
                                  unsigned long long* info, unsigned int* done) {
  const uint32_t r = *round;
  for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
    float want = 0;
    for (int k = 0; k < comm::kRanks; k++) want += bulk_value(r, k, j);
    const float got = __bfloat162float(out[j]);
    if (got != want && atomicAdd(err, 1ull) == 0) {
      info[0] = r;
      info[1] = j;
      info[2] = n;
      info[3] = (unsigned long long)(long long)got;
      info[4] = (unsigned long long)(long long)want;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(done, 1u) == gridDim.x - 1) {
      *done = 0;
      *round = r + 1;
    }
  }
}

int test_bulk_integrity(Ring& ring, cudaStream_t st, long total, size_t max_bytes, int blocks) {
  const DevChannel& ch = ring.dev();
  std::mt19937_64 rng(777);  // same sizes on every rank
  const int nodes = 50;
  std::vector<uint32_t> sizes(nodes);
  double sum = 0;
  for (auto& b : sizes) {
    double lg = std::uniform_real_distribution<double>(6, std::log2((double)max_bytes))(rng);
    b = std::max<uint32_t>(64, ((uint32_t)std::pow(2.0, lg)) & ~63u);
    sum += b;
  }
  __nv_bfloat16 *d_in, *d_out;
  uint32_t* d_round;
  unsigned long long *d_err, *d_info;
  unsigned int* d_done;
  CUDA_CHECK(cudaMalloc(&d_in, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_round, 4));
  CUDA_CHECK(cudaMalloc(&d_err, 8));
  CUDA_CHECK(cudaMalloc(&d_info, 64));
  CUDA_CHECK(cudaMalloc(&d_done, 4));
  CUDA_CHECK(cudaMemset(d_round, 0, 4));
  CUDA_CHECK(cudaMemset(d_err, 0, 8));
  CUDA_CHECK(cudaMemset(d_done, 0, 4));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (uint32_t b : sizes) {
    const uint32_t n = b / 2;
    bulk_fill_kernel<<<32, 256, 0, st>>>(d_round, ring.rank(), d_in, n);
    comm::allreduce_bulk_bf16(ch, d_in, d_out, n, st, blocks);
    bulk_check_kernel<<<32, 256, 0, st>>>(d_round, d_out, n, d_err, d_info, d_done);
  }
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  const long reps = std::max(1L, total / nodes);
  ring.barrier();
  double t0 = now_s();
  for (long i = 0; i < reps; i++) {
    CUDA_CHECK(cudaGraphLaunch(ge, st));
    if (i % 16 == 15) sync_or_die(ring, st, 60, "bulk integrity");
  }
  sync_or_die(ring, st, 60, "bulk integrity");
  double dt = now_s() - t0;
  unsigned long long err = 0, info[5] = {0};
  CUDA_CHECK(cudaMemcpy(&err, d_err, 8, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(info, d_info, sizeof info, cudaMemcpyDeviceToHost));
  unsigned long long all_err[4];
  ring.allgather_host(&err, all_err, 8);
  if (ring.rank() == 0)
    std::printf("  bulk integrity: %ld bulk all-reduces (64 B to %zu B, mean %.0f B, chunk %u B), %.1f s; "
                "wrong sums per rank: %llu %llu %llu %llu\n",
                reps * nodes, max_bytes, sum / nodes, ch.cap & ~63u, dt, all_err[0], all_err[1], all_err[2],
                all_err[3]);
  if (err)
    std::printf("rank %d first wrong sum: round %llu element %llu of %llu got %lld want %lld\n", ring.rank(), info[0],
                info[1], info[2], (long long)info[3], (long long)info[4]);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_round);
  cudaFree(d_err);
  cudaFree(d_info);
  cudaFree(d_done);
  return err ? 1 : 0;
}

// ------------------------------------------------------------------ latency

// Mean time per collective over `reps` replays of a graph of `per_graph` back-to-back collectives.
template <typename F>
double time_graph(Ring& ring, cudaStream_t st, int per_graph, int reps, F launch) {
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < per_graph; i++) launch();
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 3; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));  // warm up
  sync_or_die(ring, st, 30, "latency warmup");
  ring.barrier();
  double best = 1e30;
  for (int trial = 0; trial < 3; trial++) {
    double t0 = now_s();
    for (int i = 0; i < reps; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
    sync_or_die(ring, st, 60, "latency");
    best = std::min(best, (now_s() - t0) / ((double)reps * per_graph));
    ring.barrier();
  }
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  return best * 1e6;
}

void test_latency(Ring& ring, cudaStream_t st) {
  const DevChannel& ch = ring.dev();
  void *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, ch.cap));
  CUDA_CHECK(cudaMalloc(&d_out, 4 * (size_t)ch.cap));
  CUDA_CHECK(cudaMemset(d_in, 0, ch.cap));
  std::vector<size_t> sizes = {64, 256, 1024, 4096, 10240, 16384, 20480, 32768, 61440, 131072, 262144, 524288, 1048576, 2097152};
  if (ring.rank() == 0)
    std::printf("  %9s %12s %12s %12s %12s   (us per collective, back to back in a graph)\n", "bytes", "allreduce16",
                "allreduce32", "allgather", "redscat16");
  for (size_t b : sizes) {
    if (b > ch.cap) continue;
    int per = b <= 65536 ? 100 : 20, reps = b <= 65536 ? 50 : 20;
    double t16 = time_graph(ring, st, per, reps, [&] {
      comm::allreduce_bf16(ch, (const __nv_bfloat16*)d_in, (__nv_bfloat16*)d_out, b / 2, st);
    });
    double t32 = time_graph(ring, st, per, reps, [&] {
      comm::allreduce_f32(ch, (const float*)d_in, (float*)d_out, b / 4, st);
    });
    double tg = time_graph(ring, st, per, reps, [&] { comm::allgather(ch, d_in, d_out, b, st); });
    double trs = time_graph(ring, st, per, reps, [&] {
      comm::reduce_scatter_bf16(ch, (const __nv_bfloat16*)d_in, (__nv_bfloat16*)d_out, b / 2, st);
    });
    if (ring.rank() == 0) std::printf("  %9zu %12.2f %12.2f %12.2f %12.2f\n", b, t16, t32, tg, trs);
  }
  cudaFree(d_in);
  cudaFree(d_out);
}

// Large all-reduces: the bulk path (reduce-scatter + all-gather per chunk) against chained
// all-exchange all-reduces of cap-sized pieces. GB/s = payload bytes / time.
void test_bulk(Ring& ring, cudaStream_t st, size_t max_bytes, int blocks) {
  const DevChannel& ch = ring.dev();
  void *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, max_bytes));
  CUDA_CHECK(cudaMemset(d_in, 0, max_bytes));
  if (ring.rank() == 0)
    std::printf("  %10s %12s %8s %14s %8s %12s %8s %12s %8s %12s %8s   (bf16, %d blocks, chunk %u B; all-gathers of "
                "bytes/4 per rank; reduce-scatter of bytes per rank)\n",
                "bytes", "bulk us", "GB/s", "chained AR us", "GB/s", "allgather us", "GB/s", "redscat us", "GB/s",
                "ag multi us", "GB/s", blocks, ch.cap & ~63u);
  for (size_t b = 65536; b <= max_bytes; b *= 2) {
    const int reps = b <= (1 << 20) ? 20 : 5, per = b <= (1 << 20) ? 10 : 2;
    double tb = time_graph(ring, st, per, reps, [&] {
      comm::allreduce_bulk_bf16(ch, (const __nv_bfloat16*)d_in, (__nv_bfloat16*)d_out, b / 2, st, blocks);
    });
    double tc = time_graph(ring, st, per, reps, [&] {
      for (size_t off = 0; off < b; off += ch.cap) {
        const size_t len = std::min<size_t>(ch.cap, b - off);
        comm::allreduce_bf16(ch, (const __nv_bfloat16*)((const char*)d_in + off), (__nv_bfloat16*)((char*)d_out + off),
                             len / 2, st);
      }
    });
    double tg = time_graph(ring, st, per, reps, [&] { comm::allgather(ch, d_in, d_out, b / 4, st); });
    double tr = time_graph(ring, st, per, reps, [&] {
      comm::reduce_scatter_bulk_bf16(ch, (const __nv_bfloat16*)d_in, (__nv_bfloat16*)d_out, b / 2, st, blocks);
    });
    const comm::AgSeg seg{d_in, d_out, b / 4};
    double tm = time_graph(ring, st, per, reps, [&] { comm::allgather_multi(ch, &seg, 1, st, blocks); });
    if (ring.rank() == 0)
      std::printf("  %10zu %12.1f %8.2f %14.1f %8.2f %12.1f %8.2f %12.1f %8.2f %12.1f %8.2f\n", b, tb, b / tb / 1e3, tc,
                  b / tc / 1e3, tg, b / tg / 1e3, tr, b / tr / 1e3, tm, b / tm / 1e3);
  }
  cudaFree(d_in);
  cudaFree(d_out);
}

// Two channels at once: each half of the payload is a bulk all-reduce on its own Ring (own QPs,
// region and proxy core), on its own stream, so one channel's GPU copies overlap the other's
// network time.
void test_bulk2(Ring& a, Ring& b, cudaStream_t st, size_t max_bytes, int blocks) {
  cudaStream_t st2;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking));
  cudaEvent_t fork, join;
  CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
  __nv_bfloat16 *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, max_bytes));
  CUDA_CHECK(cudaMemset(d_in, 0, max_bytes));
  if (a.rank() == 0)
    std::printf("  %10s %12s %8s   (bf16, two channels of %d blocks, chunk %u B)\n", "bytes", "2-chan us", "GB/s", blocks,
                a.dev().cap & ~63u);
  for (size_t bytes = 1 << 20; bytes <= max_bytes; bytes *= 2) {
    const size_t half = bytes / 4;  // elements per channel
    const int reps = 5, per = 2;
    double t = time_graph(a, st, per, reps, [&] {
      CUDA_CHECK(cudaEventRecord(fork, st));
      CUDA_CHECK(cudaStreamWaitEvent(st2, fork, 0));
      comm::allreduce_bulk_bf16(a.dev(), d_in, d_out, half, st, blocks);
      comm::allreduce_bulk_bf16(b.dev(), d_in + half, d_out + half, half, st2, blocks);
      CUDA_CHECK(cudaEventRecord(join, st2));
      CUDA_CHECK(cudaStreamWaitEvent(st, join, 0));
    });
    if (a.rank() == 0) std::printf("  %10zu %12.1f %8.2f\n", bytes, t, bytes / t / 1e3);
  }
  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(fork);
  cudaEventDestroy(join);
  cudaStreamDestroy(st2);
}

// 80 all-reduces of one token's hidden state, back to back in one graph: a decode step's worth.
void test_graph(Ring& ring, cudaStream_t st, int steps) {
  const DevChannel& ch = ring.dev();
  const int n = 5120;
  __nv_bfloat16* d;
  float* f;
  CUDA_CHECK(cudaMalloc(&d, n * 2));
  CUDA_CHECK(cudaMalloc(&f, n * 4));
  CUDA_CHECK(cudaMemset(d, 0, n * 2));
  CUDA_CHECK(cudaMemset(f, 0, n * 4));
  comm::ArgMax* am;
  CUDA_CHECK(cudaMalloc(&am, 4 * sizeof(comm::ArgMax)));
  CUDA_CHECK(cudaMemset(am, 0, 4 * sizeof(comm::ArgMax)));
  double us16 = time_graph(ring, st, 80, steps, [&] { comm::allreduce_bf16(ch, d, d, n, st); });
  double us32 = time_graph(ring, st, 80, steps, [&] { comm::allreduce_f32(ch, f, f, n, st); });
  double split = time_graph(ring, st, 80, steps, [&] { split_allreduce_bf16(ch, d, d, n * 2, st); });
  double argmax = time_graph(ring, st, 80, steps, [&] { comm::argmax_rows(ch, am, am + 2, 2, st); });
  if (ring.rank() == 0) {
    std::printf("  decode step, 80 all-reduces of 5120 elements in one graph: bf16 %.2f us each (%.2f ms/step), "
                "fp32 %.2f us each (%.2f ms/step)\n",
                us16, us16 * 80 / 1000, us32, us32 * 80 / 1000);
    std::printf("  same, published by one kernel and reduced by the next (fusion pattern): bf16 %.2f us per pair\n", split);
    std::printf("  sampling all-gather of 2 (value, index) rows: %.2f us\n", argmax);
  }
  cudaFree(d);
  cudaFree(f);
  cudaFree(am);
}

// A decode step's worth of collectives (80 x 10 KB in one graph) right after the ring has been
// idle for `idle_s` (proxies backed off), against the same graph replayed right after another.
void test_wake(Ring& ring, cudaStream_t st, double idle_s) {
  const DevChannel& ch = ring.dev();
  const int n = 5120;
  __nv_bfloat16* d;
  CUDA_CHECK(cudaMalloc(&d, n * 2));
  CUDA_CHECK(cudaMemset(d, 0, n * 2));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < 80; i++) comm::allreduce_bf16(ch, d, d, n, st);
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  auto step = [&] {
    ring.barrier();
    double t0 = now_s();
    CUDA_CHECK(cudaGraphLaunch(ge, st));
    sync_or_die(ring, st, 30, "wake");
    return (now_s() - t0) * 1e6;
  };
  step();
  std::vector<double> idle, busy;
  for (int i = 0; i < 5; i++) {
    usleep((useconds_t)(idle_s * 1e6));
    idle.push_back(step());
    busy.push_back(step());
  }
  std::sort(idle.begin(), idle.end());
  std::sort(busy.begin(), busy.end());
  if (ring.rank() == 0)
    std::printf("  decode-step graph (80 x 10 KB): after %.1f s idle median %.0f us (max %.0f), right after another "
                "%.0f us (max %.0f)\n",
                idle_s, idle[2], idle[4], busy[2], busy[4]);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  cudaFree(d);
}

// kLoopNone diagnostics: one block publishes, fakes the four arrivals itself, then reduces.
__global__ void fake_allreduce_kernel(DevChannel ch, const uint4* in, uint4* out, uint32_t nbytes) {
  const uint64_t s = comm::next_seq(ch);
  uint4* sb = (uint4*)comm::send_buf(ch, s);
  const uint32_t chunks = nbytes / 16;
  for (uint32_t c = threadIdx.x; c < chunks; c += blockDim.x) sb[c] = in[c];
  comm::post_grid(ch, s, nbytes);
  if (threadIdx.x == 0)
    for (int k = 0; k < comm::kNumWait; k++) comm::st_release_sys(&ch.wait[s % comm::kSlots][k]->seq, s);
  comm::wait_all(ch, s);
  for (uint32_t c = threadIdx.x; c < chunks; c += blockDim.x) {
    float f[8];
    comm::sum8_bf16(ch, s, (size_t)c * 16, f);
    out[c] = make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]), __float_as_uint(f[3]));
  }
  comm::finish_grid(ch, s);
}

__global__ void empty_kernel() {}
// cost of individual memory-model operations, one thread, chained
__global__ void op_kernel(int op, uint64_t* host_word, uint64_t* dev_word, int iters, unsigned long long* out) {
  long long t0 = clock64();
  uint64_t acc = 0;
  for (int i = 0; i < iters; i++) {
    switch (op) {
      case 0: __threadfence_system(); break;
      case 1: comm::st_release_sys(host_word, i); break;
      case 2: acc += comm::ld_acquire_sys(host_word); break;
      case 3: acc += *(volatile uint64_t*)host_word; break;
      case 4: asm volatile("fence.acq_rel.sys;" ::: "memory"); break;
      case 5: atomicAdd((unsigned long long*)dev_word, 1ull); break;
      case 6: { unsigned long long v; asm volatile("atom.add.acq_rel.gpu.u64 %0, [%1], 1;" : "=l"(v) : "l"(dev_word) : "memory"); acc += v; } break;
      case 7: __threadfence(); break;
    }
  }
  *out = (clock64() - t0) / iters + (acc == 12345 ? 1 : 0);
}

// Single-box latency of the GPU side alone (mode none) or GPU → proxy → GPU (mode proxy).
void test_loop(Ring& ring, cudaStream_t st, comm::Options::Mode mode) {
  const DevChannel& ch = ring.dev();
  void *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, ch.cap));
  CUDA_CHECK(cudaMalloc(&d_out, ch.cap));
  CUDA_CHECK(cudaMemset(d_in, 0, ch.cap));
  {
    double us = time_graph(ring, st, 100, 50, [&] { empty_kernel<<<1, 32, 0, st>>>(); });
    std::printf("  empty kernel in a graph: %.2f us per node\n", us);
    unsigned long long* d_out;
    uint64_t* d_word;
    CUDA_CHECK(cudaMalloc(&d_out, 8));
    CUDA_CHECK(cudaMalloc(&d_word, 8));
    uint64_t* h_word = (uint64_t*)ch.post[1];  // a host-memory line nobody else touches here
    static const char* names[] = {"threadfence_system", "st.release.sys(host)", "ld.acquire.sys(host)",
                                  "ld.volatile(host)", "fence.acq_rel.sys", "atomicAdd(dev)", "atom.acq_rel.gpu(dev)",
                                  "threadfence(gpu)"};
    int clk = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    for (int op = 0; op < 8; op++) {
      op_kernel<<<1, 1, 0, st>>>(op, h_word, d_word, 2000, d_out);
      CUDA_CHECK(cudaStreamSynchronize(st));
      unsigned long long cyc;
      CUDA_CHECK(cudaMemcpy(&cyc, d_out, 8, cudaMemcpyDeviceToHost));
      std::printf("  %-24s %6llu cycles (%.2f us at %d MHz)\n", names[op], cyc, cyc / (clk / 1000.0), clk / 1000);
    }
    *h_word = 0;
  }
  for (size_t b : {64, 1024, 10240, 61440}) {
    double us = time_graph(ring, st, 100, 50, [&] {
      if (mode == comm::Options::kLoopNone)
        fake_allreduce_kernel<<<1, 256, 0, st>>>(ch, (const uint4*)d_in, (uint4*)d_out, (uint32_t)b);
      else
        comm::allreduce_bf16(ch, (const __nv_bfloat16*)d_in, (__nv_bfloat16*)d_out, b / 2, st);
    });
    std::printf("  loop(%s) %6zu B: %.2f us per collective\n", mode == comm::Options::kLoopNone ? "none" : "proxy", b, us);
  }
  cudaFree(d_in);
  cudaFree(d_out);
}

// Median proxy-side intervals over the traced collectives (after a warm-up of 20).
void print_trace(Ring& ring, int n) {
  const comm::TraceRec* t = ring.trace();
  if (!t) return;
  std::vector<double> a[8];
  for (int s = 21; s < n; s++) {
    const comm::TraceRec& r = t[s];
    bool all = r.post != 0;
    uint64_t last = 0;
    for (int k = 0; k < comm::kNumWait; k++) {
      all = all && r.arrive[k];
      last = std::max(last, r.arrive[k]);
    }
    if (!all) continue;
    auto d = [&](uint64_t x) { return x ? ((double)x - (double)r.post) / 1000.0 : NAN; };
    a[0].push_back(d(last));
    a[1].push_back(d(std::max(r.arrive[comm::kWaitPrevA], r.arrive[comm::kWaitPrevB])));
    a[2].push_back(d(std::max(r.arrive[comm::kWaitNextA], r.arrive[comm::kWaitNextB])));
    a[3].push_back(d(r.arrive[comm::kWaitOpp0]));
    a[4].push_back(d(r.arrive[comm::kWaitOpp1]));
    a[5].push_back(d(r.relay[0]));
    a[6].push_back(d(r.relay[1]));
    if (t[s + 1].post) a[7].push_back(((double)t[s + 1].post - (double)last) / 1000.0);
  }
  auto med = [](std::vector<double>& v) {
    if (v.empty()) return (double)NAN;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  std::printf("rank %d trace (median us after own post): all arrived %.2f | prev %.2f next %.2f opp0 %.2f opp1 %.2f | "
              "relayed to prev %.2f to next %.2f | last arrival -> next post %.2f\n",
              ring.rank(), med(a[0]), med(a[1]), med(a[2]), med(a[3]), med(a[4]), med(a[5]), med(a[6]), med(a[7]));
}

// ------------------------------------------------------------------ race

// The pipelined kernels' send-buffer race (docs/ring.md, the pipelined reduce-scatters' race). Block 1 of the
// kernel stalls (comm::test_stall, ring_test's build only) after posting piece i + 1 and before reducing piece i,
// at the last i whose send buffer the short last piece reuses. The other blocks run on; without wait_posted their
// copy of that piece lands on units block 1 has yet to read. The engine's shapes, scaled to --cap: the MoE's rows
// reduce-scatter (fp32, 1280 columns of 5120-wide rows from column 1280, 8 blocks of 1024 threads) with 4 pieces,
// the last short, as a 3,100-row prefill chunk's (776 rows at the engine's 4 MiB cap), and with 5; attention's
// all-to-all (rows of 2048 + 64 bytes, 8 blocks of 256 threads) with 4; and the multi-buffer all-gather (4
// chunks), which needs no wait and must stay so. Each runs without and with the stall and must be exact on every
// rank both times.
int test_race(Ring& ring, cudaStream_t st, uint32_t stall_us) {
#ifndef DSGB10_COMM_TEST_HOOKS
  if (ring.rank() == 0) std::printf("  FAIL race: this ring_test has no test hooks (DSGB10_COMM_TEST_HOOKS)\n");
  return 1;
#else
  const DevChannel& ch = ring.dev();
  const int r = ring.rank();
  const size_t piece = (ch.cap / 4) & ~63ull;
  uint32_t salt = 0x7ace;
  int failures = 0;
  // the fewest rows from `want` up whose quarter (rows * rb bytes) has n pieces, the last one short by a count of
  // 16-byte units at least a quarter of the grid's threads away from a multiple of them (so its quarters shift
  // well onto other blocks' units)
  auto pick = [&](size_t want, size_t rb, size_t n, size_t threads) {
    for (size_t rows = want;; ++rows) {
      const size_t q = rows * rb, np = (q + piece - 1) / piece, last = q - (np - 1) * piece;
      const size_t res = (last / 16) % threads;
      if (np > n) die("race: no %zu-piece shape of %zu-byte rows from %zu rows at cap %u", n, rb, want, ch.cap);
      if (np == n && last < piece && std::min(res, threads - res) >= threads / 4) return rows;
    }
  };
  // every rank's count of wrong elements; a failure on any rank fails the test on all of them
  auto report = [&](const std::string& what, size_t bad, size_t total) {
    uint64_t b = bad, all[4];
    ring.allgather_host(&b, all, sizeof b);
    if (all[0] + all[1] + all[2] + all[3]) {
      if (r == 0)
        std::printf("  FAIL race %s: wrong of %zu per rank: %lu %lu %lu %lu\n", what.c_str(), total, all[0], all[1],
                    all[2], all[3]);
      return 1;
    }
    if (r == 0) std::printf("  ok race %s (exact on every rank)\n", what.c_str());
    return 0;
  };
  auto stall_name = [&](int stall, size_t np) {
    char b[96];
    std::snprintf(b, sizeof b, stall ? "block 1 stalled %u us before piece %zu" : "no stall", stall_us, np - 4);
    return std::string(b);
  };
  // the MoE's reduce-scatter by rows: out[rr][c] = the rank-order sum of in[r rows + rr][c0 + c] over the ranks
  for (size_t np = 4; np <= 5; ++np) {
    const int cols = 1280, ld = 5120, c0 = 1280, blocks = 8;
    const int rows = (int)pick(piece * (np == 4 ? 379 : 441) / 100 / (cols * 4), cols * 4, np, blocks * 1024);
    const size_t nin = (size_t)4 * rows * ld, nout = (size_t)rows * cols, q = (size_t)rows * cols * 4;
    float *ri, *ro;
    CUDA_CHECK(cudaMalloc(&ri, nin * 4));
    CUDA_CHECK(cudaMalloc(&ro, nout * 4));
    std::vector<float> hin(nin, 0.f), hout(nout);
    ++salt;
    for (size_t row = 0; row < (size_t)4 * rows; ++row)
      for (int c = 0; c < cols; ++c) hin[row * ld + c0 + c] = test_value(r, row * ld + c0 + c, salt);
    CUDA_CHECK(cudaMemcpyAsync(ri, hin.data(), nin * 4, cudaMemcpyHostToDevice, st));
    for (int stall = 0; stall < 2; ++stall) {
      comm::test_stall(stall ? stall_us * 1000 : 0, 1, -1, st);
      CUDA_CHECK(cudaMemsetAsync(ro, 0, nout * 4, st));
      comm::reduce_scatter_rows_f32(ch, ri + c0, ld, ro, cols, rows, cols, st, blocks);
      CUDA_CHECK(cudaMemcpyAsync(hout.data(), ro, nout * 4, cudaMemcpyDeviceToHost, st));
      sync_or_die(ring, st, 30, "race: rs rows");
      size_t bad = 0;
      for (int rr = 0; rr < rows; ++rr)
        for (int c = 0; c < cols; ++c) {
          const size_t i = (size_t)(r * rows + rr) * ld + c0 + c;
          float acc = test_value(0, i, salt);
          for (int k = 1; k < 4; ++k) acc += test_value(k, i, salt);
          if (memcmp(&acc, &hout[(size_t)rr * cols + c], 4)) ++bad;
        }
      char w[160];
      std::snprintf(w, sizeof w, "rs rows f32 %d x %d (%zu pieces, the last %zu B), %s", rows, cols, np,
                    q - (np - 1) * piece, stall_name(stall, np).c_str());
      failures += report(w, bad, nout);
    }
    comm::test_stall(0, 1, -1, st);
    CUDA_CHECK(cudaFree(ri));
    CUDA_CHECK(cudaFree(ro));
  }
  // attention's all-to-all: rank r's out_g row (k rows + rr) is rank k's in_g row (r rows + rr)
  {
    const int blocks = 8, nseg = 2;
    const size_t rbs[2] = {2048, 64}, rb = rbs[0] + rbs[1], np = 4;
    const int rows = (int)pick(piece * 322 / 100 / rb, rb, np, blocks * 256);
    uint32_t* din[2];
    uint32_t* dout[2];
    std::vector<uint32_t> hin[2], hout[2];
    comm::A2aSeg segs[2];
    ++salt;
    for (int g = 0; g < nseg; ++g) {
      const size_t words = (size_t)4 * rows * rbs[g] / 4;
      CUDA_CHECK(cudaMalloc(&din[g], words * 4));
      CUDA_CHECK(cudaMalloc(&dout[g], words * 4));
      hin[g].resize(words), hout[g].resize(words);
      for (size_t w = 0; w < words; ++w) hin[g][w] = mix(((uint64_t)salt << 40) ^ ((uint64_t)r << 32) ^ ((uint64_t)g << 28) ^ w);
      CUDA_CHECK(cudaMemcpyAsync(din[g], hin[g].data(), words * 4, cudaMemcpyHostToDevice, st));
      segs[g] = {din[g], rbs[g], dout[g], rbs[g], rbs[g]};
    }
    for (int stall = 0; stall < 2; ++stall) {
      comm::test_stall(stall ? stall_us * 1000 : 0, 1, -1, st);
      for (int g = 0; g < nseg; ++g) CUDA_CHECK(cudaMemsetAsync(dout[g], 0, hout[g].size() * 4, st));
      comm::alltoall_rows(ch, segs, nseg, rows, st, blocks);
      for (int g = 0; g < nseg; ++g)
        CUDA_CHECK(cudaMemcpyAsync(hout[g].data(), dout[g], hout[g].size() * 4, cudaMemcpyDeviceToHost, st));
      sync_or_die(ring, st, 30, "race: all-to-all");
      size_t bad = 0, total = 0;
      for (int g = 0; g < nseg; ++g) {
        const size_t wpr = rbs[g] / 4;  // words a row
        for (int k = 0; k < 4; ++k)
          for (int rr = 0; rr < rows; ++rr)
            for (size_t j = 0; j < wpr; ++j) {
              const size_t w = ((size_t)r * rows + rr) * wpr + j;
              const uint32_t want = mix(((uint64_t)salt << 40) ^ ((uint64_t)k << 32) ^ ((uint64_t)g << 28) ^ w);
              if (hout[g][((size_t)k * rows + rr) * wpr + j] != want) ++bad;
              ++total;
            }
      }
      char w[160];
      std::snprintf(w, sizeof w, "all-to-all %d rows of %zu + %zu B (%zu pieces, the last %zu B), %s", rows, rbs[0],
                    rbs[1], np, rows * rb - (np - 1) * piece, stall_name(stall, np).c_str());
      failures += report(w, bad, total);
    }
    comm::test_stall(0, 1, -1, st);
    for (int g = 0; g < nseg; ++g) {
      CUDA_CHECK(cudaFree(din[g]));
      CUDA_CHECK(cudaFree(dout[g]));
    }
  }
  // the multi-buffer all-gather (prefill's input gathers: rows of 5120 + 160 B, 16 blocks of 1024 threads), 4
  // chunks of cap, the last short: it copies two chunks ahead with no wait, safe only because each unit of its
  // send buffer is written and read by one thread; this case keeps it so
  {
    const int nseg = 2;
    const size_t rbs[2] = {5120, 160}, rb = rbs[0] + rbs[1], chunk = ch.cap & ~63ull, threads = 16 * 1024;
    size_t rows = chunk * 36 / 10 / rb;
    for (;; ++rows) {
      const size_t t = rows * rb, nc = (t + chunk - 1) / chunk, last = t - (nc - 1) * chunk, res = (last / 16) % threads;
      if (nc > 4) die("race: no 4-chunk all-gather of %zu-byte rows at cap %u", rb, ch.cap);
      if (nc == 4 && last < chunk && std::min(res, threads - res) >= threads / 4) break;
    }
    uint32_t* din[2];
    uint32_t* dout[2];
    std::vector<uint32_t> hin[2], hout[2];
    comm::AgSeg segs[2];
    ++salt;
    for (int g = 0; g < nseg; ++g) {
      const size_t words = rows * rbs[g] / 4;
      CUDA_CHECK(cudaMalloc(&din[g], words * 4));
      CUDA_CHECK(cudaMalloc(&dout[g], 4 * words * 4));
      hin[g].resize(words), hout[g].resize(4 * words);
      for (size_t w = 0; w < words; ++w) hin[g][w] = mix(((uint64_t)salt << 40) ^ ((uint64_t)r << 32) ^ ((uint64_t)g << 28) ^ w);
      CUDA_CHECK(cudaMemcpyAsync(din[g], hin[g].data(), words * 4, cudaMemcpyHostToDevice, st));
      segs[g] = {din[g], dout[g], words * 4};
    }
    for (int stall = 0; stall < 2; ++stall) {
      comm::test_stall(stall ? stall_us * 1000 : 0, 1, -1, st);
      for (int g = 0; g < nseg; ++g) CUDA_CHECK(cudaMemsetAsync(dout[g], 0, hout[g].size() * 4, st));
      comm::allgather_multi(ch, segs, nseg, st);
      for (int g = 0; g < nseg; ++g)
        CUDA_CHECK(cudaMemcpyAsync(hout[g].data(), dout[g], hout[g].size() * 4, cudaMemcpyDeviceToHost, st));
      sync_or_die(ring, st, 30, "race: all-gather");
      size_t bad = 0, total = 0;
      for (int g = 0; g < nseg; ++g) {
        const size_t words = hin[g].size();
        for (int k = 0; k < 4; ++k)
          for (size_t w = 0; w < words; ++w, ++total)
            if (hout[g][k * words + w] != mix(((uint64_t)salt << 40) ^ ((uint64_t)k << 32) ^ ((uint64_t)g << 28) ^ w)) ++bad;
      }
      char w[160];
      std::snprintf(w, sizeof w, "allgather multi %zu rows of %zu + %zu B (4 chunks, the last %zu B), %s", rows, rbs[0],
                    rbs[1], rows * rb - 3 * chunk, stall_name(stall, 4).c_str());
      failures += report(w, bad, total);
    }
    comm::test_stall(0, 1, -1, st);
    for (int g = 0; g < nseg; ++g) {
      CUDA_CHECK(cudaFree(din[g]));
      CUDA_CHECK(cudaFree(dout[g]));
    }
  }
  return failures;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  comm::Options o;
  o.bootstrap_timeout_s = 120;  // run4.sh starts all ranks within seconds
  long integrity_n = 1000000;
  int graph_steps = 500, bulk_blocks = 16;
  uint32_t race_us = 2000;
  size_t bulk_max = 32 << 20;
  std::vector<std::string> tests;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--rank" && i + 1 < argc) o.rank = atoi(argv[++i]);
    else if (a == "--fleet" && i + 1 < argc) o.fleet = argv[++i];
    else if (a == "--cap" && i + 1 < argc) o.cap = atol(argv[++i]);
    else if (a == "--integrity" && i + 1 < argc) integrity_n = atol(argv[++i]);
    else if (a == "--graph-steps" && i + 1 < argc) graph_steps = atoi(argv[++i]);
    else if (a == "--verbose") o.verbose = true;
    else if (a == "--trace" && i + 1 < argc) o.trace = atoi(argv[++i]);
    else if (a == "--split-from" && i + 1 < argc) o.split_domains_from = atoi(argv[++i]);
    else if (a == "--relay-domain" && i + 1 < argc) o.relay_domain = atoi(argv[++i]);
    else if (a == "--rs-slow-pct" && i + 1 < argc) o.rs_slow_pct = atoi(argv[++i]);
    else if (a == "--bulk-blocks" && i + 1 < argc) bulk_blocks = atoi(argv[++i]);
    else if (a == "--bootstrap-timeout" && i + 1 < argc) o.bootstrap_timeout_s = atoi(argv[++i]);
    else if (a == "--idle-backoff-ms" && i + 1 < argc) o.idle_backoff_ms = atoi(argv[++i]);
    else if (a == "--proxy-cpu" && i + 1 < argc) o.proxy_cpu = atoi(argv[++i]);
    else if (a == "--bulk-max" && i + 1 < argc) bulk_max = atol(argv[++i]);
    else if (a == "--race-us" && i + 1 < argc) race_us = (uint32_t)atol(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) {
      std::string m = argv[++i];
      o.mode = m == "proxy" ? comm::Options::kLoopProxy : m == "none" ? comm::Options::kLoopNone : comm::Options::kRing;
    }
    else tests.push_back(a);
  }
  if (tests.empty()) tests = {"correct", "race", "integrity", "latency", "graph"};
  Ring ring(o);
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  int failures = 0;
  for (auto& t : tests) {
    if (ring.rank() == 0) std::printf("== %s\n", t.c_str());
    std::fflush(stdout);
    if (t == "correct") failures += test_correct(ring, st);
    else if (t == "race") failures += test_race(ring, st, race_us);
    else if (t == "integrity") failures += test_integrity(ring, st, integrity_n);
    else if (t == "latency") test_latency(ring, st);
    else if (t == "bulk") test_bulk(ring, st, bulk_max, bulk_blocks);
    else if (t == "bulk2") {
      comm::Options o2 = o;
      o2.port = o.port + 1;
      o2.proxy_cpu = o.proxy_cpu + 1;
      Ring ring2(o2);
      test_bulk2(ring, ring2, st, bulk_max, bulk_blocks);
    }
    else if (t == "bulk-integrity") failures += test_bulk_integrity(ring, st, integrity_n / 30, bulk_max, bulk_blocks);
    else if (t == "graph") test_graph(ring, st, graph_steps);
    else if (t == "wake") test_wake(ring, st, 2.0);
    else if (t == "loop") test_loop(ring, st, o.mode);
    else die("unknown test %s", t.c_str());
    std::fflush(stdout);
  }
  print_trace(ring, o.trace);
  comm::ProxyStats ps = ring.stats();
  std::printf("rank %d: %s; proxy posted %lu, relayed %lu + %lu, relays held %lu, queue-full retries %lu, "
              "%.2f us per ibv_post_send\n",
              ring.rank(), failures ? "FAILED" : "passed", ps.posted, ps.relayed[0], ps.relayed[1], ps.relay_waits,
              ps.post_retries, ps.post_calls ? ps.post_ns / 1000.0 / ps.post_calls : 0.0);
  return failures ? 1 : 0;
}
