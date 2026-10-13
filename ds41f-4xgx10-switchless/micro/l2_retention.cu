// L2 retention on GB10, timed inside the kernels (globaltimer span over all CTAs).
// build: nvcc -O3 -gencode arch=compute_121a,code=sm_121a -o l2_retention l2_retention.cu
// l2_retention [old|paced|window|lat|between|all]: old, prefetches and loads re-read 300 us later; paced, a paced
// side prefetch (TensorFold's issue pattern) and the GEMVs' read 0-300 us after it; window, that prefetch beside a
// spinning kernel (a collective's window) and the read after the spinner; lat, a chain of dependent loads (a
// latency-bound kernel's, as a collective's polls) beside each prefetcher; between, other bytes streamed between
// the prefetch and the read; evict, what a prefetch pushes out: lines written or read just before it, read
// again after it. Default: all.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint64_t gt() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint4 ld_nc(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ld_cg(const void* p) {
  uint4 r;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ld_na(const void* p) {
  uint4 r;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ld_ncd(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ld_last(const void* p, uint64_t pol) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
  return r;
}

// mode 0: ld.nc no_allocate, 1: ld.cg, 2: ld.nc + L2 evict_last policy, 3: ld.nc + evict_first policy,
// 10: bulk prefetch only, 11: prefetch.L2 lines only, 12: prefetch.L2::evict_last lines only
// after, gap_ns: thread 0 of each CTA first waits until *after + gap_ns (a time mark an earlier kernel of the stream
// wrote), so the gap from that kernel's end is exact and no host call sits in it.
// started: block 0 stores its start time there (lat's chain waits for it).
__global__ void k_read(const uint8_t* base, size_t slice, int mode, unsigned long long* span, uint32_t* sink,
                       const unsigned long long* after = nullptr, uint32_t gap_ns = 0,
                       unsigned long long* started = nullptr) {
  const uint8_t* p = base + (size_t)blockIdx.x * slice;
  __shared__ uint64_t t0s;
  if (threadIdx.x == 0) {
    if (after) {
      const uint64_t until = *(volatile const unsigned long long*)after + gap_ns;
      while (gt() < until) {
      }
    }
    t0s = gt();
    if (started && blockIdx.x == 0) *(volatile unsigned long long*)started = t0s;
  }
  __syncthreads();
  uint32_t acc = 0;
  uint64_t pol = 0;
  if (mode == 2) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  if (mode == 3) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  if (mode == 13) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol));
  if (mode == 14) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  if (mode == 10) {
    if (threadIdx.x == 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"((uint32_t)slice) : "memory");
  } else if (mode == 13 || mode == 14) {
    if (threadIdx.x == 0)
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(p), "r"((uint32_t)slice), "l"(pol) : "memory");
  } else if (mode == 11 || mode == 12 || mode == 15) {
    for (size_t o = threadIdx.x * 128; o < slice; o += blockDim.x * 128) {
      if (mode == 11) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + o));
      else if (mode == 12) asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p + o));
      else asm volatile("prefetch.global.L2::evict_normal [%0];" ::"l"(p + o));
    }
  } else {
    for (size_t o = threadIdx.x * 16; o < slice; o += blockDim.x * 64) {
      uint4 v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const size_t q = o + (size_t)u * blockDim.x * 16;
        if (q >= slice) { v[u] = make_uint4(0, 0, 0, 0); continue; }
        v[u] = mode == 0 ? ld_nc(p + q) : mode == 1 ? ld_cg(p + q) : mode == 4 ? ld_na(p + q) : mode == 5 ? ld_ncd(p + q) : ld_last(p + q, pol);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin(&span[0], (unsigned long long)t0s);
    atomicMax(&span[1], (unsigned long long)gt());
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

// The paced side prefetch: the issue pattern of TensorFold's l2_prefetch.cu (bertholomus/TensorFold) as
// sudoingX/dgx-spark-deepseek-v41 runs it (exl3_l2pf.cu, Apache-2.0); this code is ours. 128 threads a block issue
// one `chunk`-byte bulk L2 prefetch each in the buffer's order, a wave of blockDim chunks every wave_ns after the
// start (0: all at once), so DRAM's queue holds about a wave. kind 0: cp.async.bulk.prefetch.L2, 1: with an
// L2::evict_last policy. Nothing waits for the prefetches: span[1] is the last wave's issue. started: as k_read's.
__global__ void __launch_bounds__(128) k_paced(const uint8_t* p, size_t n, uint32_t chunk, uint32_t wave_ns, int kind,
                                               unsigned long long* span, unsigned long long* started = nullptr) {
  __shared__ uint64_t t0s;
  if (threadIdx.x == 0) {
    t0s = gt();
    if (started && blockIdx.x == 0) *(volatile unsigned long long*)started = t0s;
  }
  __syncthreads();
  const uint64_t t0 = t0s;
  uint64_t pol = 0;
  if (kind == 1) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  const size_t nc = (n + chunk - 1) / chunk;
  uint64_t wave = 0;
  for (size_t c0 = (size_t)blockIdx.x * blockDim.x; c0 < nc; c0 += (size_t)gridDim.x * blockDim.x, ++wave) {
    if (wave_ns) {
      if (threadIdx.x == 0)
        while (gt() - t0 < wave * wave_ns) __nanosleep(256);
      __syncthreads();
    }
    const size_t c = c0 + threadIdx.x;
    if (c < nc) {
      const size_t off = c * chunk;
      const uint32_t size = (uint32_t)(n - off < chunk ? n - off : chunk);
      if (kind == 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p + off), "r"(size) : "memory");
      else
        asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(p + off), "r"(size), "l"(pol)
                     : "memory");
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin(&span[0], (unsigned long long)t0);
    atomicMax(&span[1], (unsigned long long)gt());
  }
}

// A side prefetch by loads instead: one block of 1024 threads streams the bytes with ld.global.cg (which allocates in
// L2), 8 loads in flight a thread. span[1]: the last load's data in.
__global__ void __launch_bounds__(1024) k_ldpf(const uint8_t* p, size_t n, unsigned long long* span, uint32_t* sink,
                                              unsigned long long* started = nullptr) {
  __shared__ uint64_t t0s;
  if (threadIdx.x == 0) {
    t0s = gt();
    if (started) *(volatile unsigned long long*)started = t0s;
  }
  __syncthreads();
  uint32_t acc = 0;
  for (size_t o = threadIdx.x * 16; o < n; o += blockDim.x * 128) {
    uint4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const size_t q = o + (size_t)u * blockDim.x * 16;
      v[u] = q < n ? ld_cg(p + q) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin(&span[0], (unsigned long long)t0s);
    atomicMax(&span[1], (unsigned long long)gt());
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

__global__ void k_spin(uint32_t ns) {
  const uint64_t t0 = gt();
  while (gt() - t0 < ns) {
  }
}
// A latency-bound kernel: one thread follows n links of a random chain over 64 MiB (ld.global.cg: a line a link, out
// of L2 after a flush), from 2 us after *started (if given; up to 10 ms for it). out: the chain's start and end.
__global__ void k_chase(const uint32_t* chain, uint32_t start, int n, const unsigned long long* started,
                        unsigned long long* out, uint32_t* sink) {
  if (threadIdx.x) return;
  if (started) {
    const uint64_t tw = gt();
    uint64_t s0;
    while ((s0 = *(volatile const unsigned long long*)started) == 0 && gt() - tw < 10000000) {
    }
    while (s0 && gt() < s0 + 2000) {
    }
  }
  uint32_t i = start;
  const uint64_t t0 = gt();
  for (int k = 0; k < n; ++k) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(i) : "l"(chain + i) : "memory");
  const uint64_t t1 = gt();
  out[0] = t0;
  out[1] = t1;
  if (i == 0xFFFFFFFFu) sink[0] = i;
}
// a collective's window: CTAs that spin for ns and touch no memory but their span
__global__ void k_window(uint32_t ns, unsigned long long* span) {
  const uint64_t t0 = gt();
  while (gt() - t0 < ns) {
  }
  if (threadIdx.x == 0) {
    atomicMin(&span[0], (unsigned long long)t0);
    atomicMax(&span[1], (unsigned long long)gt());
  }
}
__global__ void k_marks(unsigned long long* m, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) m[i] = i % 2 ? 0ull : ~0ull;
}
__global__ void k_flush(const uint4* p, size_t n, uint32_t* sink) {
  uint32_t a = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a ^= p[i].x;
  if (a == 0x12345678u) sink[0] = a;
}
// resets an L2 line's priority to normal (after evict_last use)
__global__ void k_normal(const uint8_t* p, size_t n) {
  for (size_t o = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 128; o < n; o += (size_t)gridDim.x * blockDim.x * 128)
    asm volatile("applypriority.global.L2::evict_normal [%0], 128;" ::"l"(p + o));
}

constexpr size_t kFl = 96ull << 20, kBig = 64ull << 20;

struct Dev {
  uint8_t *buf, *fb;
  uint32_t* sink;
  unsigned long long* span;  // the old tests'
  unsigned long long* m;     // time marks: [0, 1] the prefetch's span, [2, 3] the read's, [4, 5] the window's, [7]
                             // the side kernel's start (lat), [8, 9] the chain's span
  uint32_t* chain;           // lat's: 64 MiB, a random cycle through its 128-byte lines
  cudaStream_t s0, s1;
  cudaEvent_t ev;
};

static double med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static void old_tests(const Dev& d) {
  uint8_t* buf = d.buf;
  uint32_t* sink = d.sink;
  unsigned long long* span = d.span;
  auto flush = [&]() { k_flush<<<480, 256>>>((const uint4*)d.fb, kFl / 16, sink); k_normal<<<480, 256>>>(buf, kBig); k_spin<<<1, 32>>>(20000); };
  auto run = [&](int mode, size_t slice, int grid) -> double {
    unsigned long long h[2] = {~0ull, 0ull};
    CK(cudaMemcpy(span, h, 16, cudaMemcpyHostToDevice));
    k_read<<<grid, 256>>>(buf, slice, mode, span, sink);
    CK(cudaMemcpy(h, span, 16, cudaMemcpyDeviceToHost));
    return (h[1] - h[0]) / 1e3;
  };
  for (size_t mb : {4, 16}) {
    const int grid = 48;
    const size_t slice = ((mb << 20) / grid) & ~(size_t)4095;
    printf("%zu MB (48 CTAs):\n", mb);
    const char* rn[] = {"ld.nc", "ld.cg", "ld.nc evict_last", "ld.nc evict_first", "ld.L1::no_allocate", "ld.nc (L1)"};
    for (int rm : {0, 1, 4, 5}) {
      std::vector<double> c1, c2, c3;
      for (int rep = 0; rep < 5; ++rep) {
        flush();
        c1.push_back(run(rm, slice, grid));        // cold
        k_spin<<<1, 32>>>(300000);
        c2.push_back(run(0, slice, grid));         // re-read 300 us later (plain ld.nc)
        c3.push_back(run(0, slice, grid));         // and right after
      }
      std::sort(c1.begin(), c1.end()), std::sort(c2.begin(), c2.end()), std::sort(c3.begin(), c3.end());
      printf("  read %-18s cold %7.1f us (%5.0f GB/s), re-read +300us %7.1f us (%5.0f GB/s), again %7.1f us (%5.0f GB/s)\n",
             rn[rm], c1[2], mb * 1.048576e3 / c1[2], c2[2], mb * 1.048576e3 / c2[2], c3[2], mb * 1.048576e3 / c3[2]);
    }
    const char* pn[] = {"bulk prefetch", "prefetch.L2", "prefetch.L2 evict_last", "bulk evict_normal", "bulk evict_last", "prefetch.L2 evict_normal"};
    for (int pm : {10, 11, 13, 14, 15}) {
      std::vector<double> c1, c2;
      for (int rep = 0; rep < 5; ++rep) {
        flush();
        c1.push_back(run(pm, slice, grid));
        k_spin<<<1, 32>>>(300000);
        c2.push_back(run(0, slice, grid));
      }
      std::sort(c1.begin(), c1.end()), std::sort(c2.begin(), c2.end());
      printf("  %-23s issue %7.1f us, read +300us %7.1f us (%5.0f GB/s)\n", pn[pm - 10], c1[2], c2[2], mb * 1.048576e3 / c2[2]);
    }
  }
  CK(cudaDeviceSynchronize());
}

// The new tests' flush: evict_last lines back to normal first, so the flush's 96 MB evict them too.
static void flush_s(const Dev& d) {
  k_normal<<<480, 256, 0, d.s0>>>(d.buf, kBig);
  k_flush<<<480, 256, 0, d.s0>>>((const uint4*)d.fb, kFl / 16, d.sink);
  k_spin<<<1, 32, 0, d.s0>>>(20000);
}

// a side prefetcher: kind 0 bulk prefetch, 1 bulk prefetch evict_last (k_paced: `blocks` of `threads` (128), a
// wave of `threads` chunks every wave_ns, 0: at once), 2 one block of ld.global.cg (k_ldpf), 3 the GEMVs' read
// (k_read: 48 CTAs of ld.nc.L1::no_allocate, a weight kernel streaming)
struct Pf {
  const char* name;
  int kind, blocks;
  uint32_t wave_ns;
  int threads;
};
static void prefetch(const Dev& d, cudaStream_t s, const Pf& pf, size_t n, unsigned long long* started = nullptr) {
  if (pf.kind == 3) k_read<<<48, 256, 0, s>>>(d.buf, (n / 48) & ~(size_t)4095, 0, d.m, d.sink, nullptr, 0, started);
  else if (pf.kind == 2) k_ldpf<<<1, 1024, 0, s>>>(d.buf, n, d.m, d.sink, started);
  else k_paced<<<pf.blocks, pf.threads ? pf.threads : 128, 0, s>>>(d.buf, n, 8192, pf.wave_ns, pf.kind, d.m, started);
}
static void marks(const Dev& d, unsigned long long* h) {
  CK(cudaMemcpyAsync(h, d.m, 10 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, d.s0));
  CK(cudaStreamSynchronize(d.s0));
}

static const uint32_t kWave = (uint32_t)(128 * 8192 / 230.0);  // ns: a wave of 128 8-KiB chunks at 230 GB/s
static uint32_t wave_ns(int chunks, double gbps) { return (uint32_t)(chunks * 8192 / gbps); }

// cold: flushed, then the GEMVs' read (ld.nc.L1::no_allocate, 48 CTAs); hot: right after an ld.cg read of the bytes
static void reference(const Dev& d, size_t slice, double& cold, double& hot) {
  unsigned long long h[10];
  std::vector<double> c, w;
  for (int rep = 0; rep < 5; ++rep) {
    k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
    flush_s(d);
    k_read<<<48, 256, 0, d.s0>>>(d.buf, slice, 0, d.m + 2, d.sink);
    marks(d, h);
    c.push_back((h[3] - h[2]) / 1e3);
    k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
    flush_s(d);
    k_read<<<48, 256, 0, d.s0>>>(d.buf, slice, 1, d.m + 4, d.sink);
    k_read<<<48, 256, 0, d.s0>>>(d.buf, slice, 0, d.m + 2, d.sink);
    marks(d, h);
    w.push_back((h[3] - h[2]) / 1e3);
  }
  cold = med(c), hot = med(w);
}

// The prefetch on the read's stream, then the read +gap after its last issue (k_ldpf: its last load in). "L2 %": the
// share of the read's time saved, (cold - t) / (cold - hot).
static void paced_tests(const Dev& d) {
  const Pf pfs[] = {{"bulk prefetch, paced 230", 0, 1, kWave}, {"bulk prefetch, at once x4", 0, 4, 0},
                    {"bulk evict_last, paced 230", 1, 1, kWave}, {"ld.cg, one block of 1024", 2, 1, 0}};
  const int gaps[] = {0, 10, 20, 30, 45, 60, 100, 200, 300};
  unsigned long long h[10];
  printf("paced: a side prefetch of the bytes, then the GEMVs' read of them (ld.nc.L1::no_allocate, 48 CTAs) +gap us "
         "after its last issue: read us (L2 %%)\n");
  for (size_t mb : {2, 4, 8, 16}) {
    const size_t slice = ((mb << 20) / 48) & ~(size_t)4095, n = slice * 48;
    double cold, hot;
    reference(d, slice, cold, hot);
    printf("%5.2f MiB: cold %.1f us (%.0f GB/s), from L2 %.1f us (%.0f GB/s)\n", n / 1048576.0, cold, n / 1e3 / cold,
           hot, n / 1e3 / hot);
    for (const Pf& pf : pfs) {
      std::vector<double> iss;
      std::string row;
      for (int g : gaps) {
        std::vector<double> rd;
        for (int rep = 0; rep < 5; ++rep) {
          k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
          flush_s(d);
          prefetch(d, d.s0, pf, n);
          k_read<<<48, 256, 0, d.s0>>>(d.buf, slice, 0, d.m + 2, d.sink, d.m + 1, g * 1000u);
          marks(d, h);
          iss.push_back((h[1] - h[0]) / 1e3);
          rd.push_back((h[3] - h[2]) / 1e3);
        }
        const double r = med(rd);
        char s[64];
        snprintf(s, sizeof s, "  +%d %.1f (%.0f)", g, r, 100 * (cold - r) / (cold - hot));
        row += s;
      }
      printf("  %-27s issued in %6.1f us (%4.0f GB/s);%s\n", pf.name, med(iss), n / 1e3 / med(iss), row.c_str());
    }
  }
}

// A window of W us (40 CTAs spinning, as a collective's do) on the read's stream, the prefetch of the read's bytes
// launched beside it on another stream from the window's start, then the read: its time, and the window's start to
// the read's end (no prefetch: the window + a cold read).
static void window_tests(const Dev& d) {
  const Pf pfs[] = {{"none", -1, 0, 0}, {"bulk paced 230", 0, 1, kWave}, {"ld.cg block", 2, 1, 0}};
  unsigned long long h[10];
  printf("window: W us of 40 spinning CTAs, a side prefetch of the read's bytes from its start, then the read: read "
         "us / window start to read end us (prefetch start - window start us)\n");
  for (size_t mb : {4, 8, 16}) {
    const size_t slice = ((mb << 20) / 48) & ~(size_t)4095, n = slice * 48;
    double cold, hot;
    reference(d, slice, cold, hot);
    printf("%5.2f MiB: cold %.1f us, from L2 %.1f us\n", n / 1048576.0, cold, hot);
    for (int W : {10, 20, 30, 45, 60}) {
      printf("  W %2d:", W);
      for (const Pf& pf : pfs) {
        std::vector<double> rd, e2e, st;
        for (int rep = 0; rep < 5; ++rep) {
          k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
          flush_s(d);
          CK(cudaEventRecord(d.ev, d.s0));
          CK(cudaStreamWaitEvent(d.s1, d.ev, 0));
          if (pf.kind >= 0) prefetch(d, d.s1, pf, n);
          k_window<<<40, 32, 0, d.s0>>>(W * 1000u, d.m + 4);
          k_read<<<48, 256, 0, d.s0>>>(d.buf, slice, 0, d.m + 2, d.sink);
          CK(cudaStreamSynchronize(d.s1));
          marks(d, h);
          rd.push_back((h[3] - h[2]) / 1e3);
          e2e.push_back(((double)h[3] - (double)h[4]) / 1e3);
          if (pf.kind >= 0) st.push_back(((double)h[0] - (double)h[4]) / 1e3);
        }
        if (pf.kind >= 0)
          printf("   %s %.1f / %.1f (%+.1f)", pf.name, med(rd), med(e2e), med(st));
        else
          printf("   %s %.1f / %.1f", pf.name, med(rd), med(e2e));
      }
      printf("\n");
    }
  }
}

// lat: the chain's latency a link (48 links from 2 us into the side kernel; 9 runs, the median) with nothing beside
// it and beside each prefetcher of 48 MiB (paced 230: ~210 us). "overlap": the share of the chain's span before the
// prefetch's last issue (the GEMV-like read: its end).
static void lat_tests(const Dev& d) {
  const Pf pfs[] = {{"none", -1, 0, 0, 0},
                    {"bulk paced 230, waves of 128", 0, 1, wave_ns(128, 230), 128},
                    {"bulk paced 230, waves of 32", 0, 1, wave_ns(32, 230), 32},
                    {"bulk paced 230, waves of 8", 0, 1, wave_ns(8, 230), 8},
                    {"bulk paced 150, waves of 8", 0, 1, wave_ns(8, 150), 8},
                    {"bulk paced 100, waves of 8", 0, 1, wave_ns(8, 100), 8},
                    {"bulk at once x4", 0, 4, 0, 128},
                    {"ld.cg, one block of 1024", 2, 1, 0, 0},
                    {"GEMV-like read, 48 CTAs", 3, 48, 0, 0}};
  const size_t n = 48ull << 20;
  const int links = 48;
  unsigned long long h[10];
  printf("lat: a chain of %d dependent loads (ld.global.cg, a 128-byte line each, flushed) beside a side kernel on 48 "
         "MiB: ns a link (side kernel's rate GB/s, overlap %%)\n", links);
  uint32_t start = 0;
  for (const Pf& pf : pfs) {
    std::vector<double> lat, rate, ov;
    for (int rep = 0; rep < 9; ++rep) {
      k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
      flush_s(d);
      CK(cudaEventRecord(d.ev, d.s0));
      CK(cudaStreamWaitEvent(d.s1, d.ev, 0));
      if (pf.kind >= 0) prefetch(d, d.s1, pf, n, d.m + 7);
      k_chase<<<1, 32, 0, d.s0>>>(d.chain, start, links, pf.kind >= 0 ? d.m + 7 : nullptr, d.m + 8, d.sink);
      start = (start + 32 * 4099) % (16u << 20) & ~31u;
      CK(cudaStreamSynchronize(d.s1));
      marks(d, h);
      lat.push_back((double)(h[9] - h[8]) / links);
      if (pf.kind >= 0) {
        rate.push_back(n / ((double)(h[1] - h[0])));
        const double end = (double)h[1], a = (double)h[8], b = (double)h[9];
        ov.push_back(100 * std::max(0.0, std::min(b, end) - a) / (b - a));
      }
    }
    if (pf.kind >= 0)
      printf("  %-30s %6.0f ns (%4.0f GB/s, overlap %3.0f %%)\n", pf.name, med(lat), med(rate), med(ov));
    else
      printf("  %-30s %6.0f ns\n", pf.name, med(lat));
  }
}

// between: X paced into L2 (230, waves of 128), then Y MiB of other bytes read by 48 CTAs (ld.nc.L1::no_allocate,
// the GEMVs'; or ld.cg), then the GEMVs' read of X: us (L2 %).
static void between_tests(const Dev& d) {
  unsigned long long h[10];
  printf("between: X paced into L2, Y MiB of other bytes read by 48 CTAs, then X read: us (L2 %%)\n");
  for (double xm : {4.0, 10.5}) {
    const size_t slice = ((size_t)(xm * 1048576) / 48) & ~(size_t)4095, n = slice * 48;
    double cold, hot;
    reference(d, slice, cold, hot);
    printf("X %5.2f MiB: cold %.1f us, from L2 %.1f us\n", n / 1048576.0, cold, hot);
    for (int ym : {0, 1}) {
      const char* yn[] = {"Y by ld.nc.L1::no_allocate", "Y by ld.cg"};
      printf("  %-27s", yn[ym]);
      for (double y : {0.0, 2.3, 8.0, 16.0, 32.0}) {
        const size_t ys = ((size_t)(y * 1048576) / 48) & ~(size_t)4095;
        std::vector<double> rd;
        for (int rep = 0; rep < 5; ++rep) {
          k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
          flush_s(d);
          k_paced<<<1, 128, 0, d.s0>>>(d.buf, n, 8192, kWave, 0, d.m);
          if (ys) k_read<<<48, 256, 0, d.s0>>>(d.buf + (16ull << 20), ys, ym == 0 ? 0 : 1, d.m + 4, d.sink);
          k_read<<<48, 256, 0, d.s0>>>(d.buf, slice, 0, d.m + 2, d.sink);
          marks(d, h);
          rd.push_back((h[3] - h[2]) / 1e3);
        }
        const double r = med(rd);
        printf("  Y %4.1f: %5.1f (%3.0f)", y, r, 100 * (cold - r) / (cold - hot));
      }
      printf("\n");
    }
  }
}

// k_write: 48 CTAs store D's bytes (as a kernel's outputs: activations, a KV write), so they sit in L2 dirty
__global__ void k_write(uint8_t* base, size_t slice) {
  uint8_t* p = base + (size_t)blockIdx.x * slice;
  for (size_t o = threadIdx.x * 16; o < slice; o += blockDim.x * 16)
    *(uint4*)(p + o) = make_uint4(o, o + 1, o + 2, blockIdx.x);
}

// evict: D (0.25-4 MiB, at 48 MiB into the buffer) written by 48 CTAs or read by ld.cg, then nothing or a paced
// prefetch (230, waves of 128) of X of other bytes, then D read again by ld.cg: us (the share of D's L2 benefit
// kept, against D cold after a flush and D read right after it was touched).
static void evict_tests(const Dev& d) {
  unsigned long long h[10];
  uint8_t* D = d.buf + (48ull << 20);
  printf("evict: D touched (written, or read by ld.cg), then a paced prefetch of X MiB of other bytes, then D read "
         "by ld.cg: us (%% of D's L2 benefit kept)\n");
  for (double dm : {0.25, 1.0, 4.0}) {
    const size_t ds = ((size_t)(dm * 1048576) / 48) & ~(size_t)255;
    for (int how : {0, 1}) {
      auto touch = [&]() {
        if (how == 0) k_write<<<48, 256, 0, d.s0>>>(D, ds);
        else k_read<<<48, 256, 0, d.s0>>>(D, ds, 1, d.m + 4, d.sink);
      };
      std::vector<double> c, w;
      for (int rep = 0; rep < 5; ++rep) {
        k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
        flush_s(d);
        k_read<<<48, 256, 0, d.s0>>>(D, ds, 1, d.m + 2, d.sink);
        marks(d, h);
        c.push_back((h[3] - h[2]) / 1e3);
        k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
        flush_s(d);
        touch();
        k_read<<<48, 256, 0, d.s0>>>(D, ds, 1, d.m + 2, d.sink);
        marks(d, h);
        w.push_back((h[3] - h[2]) / 1e3);
      }
      const double cold = med(c), hot = med(w);
      printf("  D %4.2f MiB %-8s cold %5.2f, hot %5.2f us:", ds * 48 / 1048576.0, how ? "read" : "written", cold, hot);
      for (double xm : {2.3, 4.0, 10.5, 16.0}) {
        const size_t xs = ((size_t)(xm * 1048576) / 48) & ~(size_t)4095, n = xs * 48;
        std::vector<double> rd;
        for (int rep = 0; rep < 5; ++rep) {
          k_marks<<<1, 32, 0, d.s0>>>(d.m, 10);
          flush_s(d);
          touch();
          k_paced<<<1, 128, 0, d.s0>>>(d.buf, n, 8192, kWave, 0, d.m);
          k_read<<<48, 256, 0, d.s0>>>(D, ds, 1, d.m + 2, d.sink, d.m + 1, 0);
          marks(d, h);
          rd.push_back((h[3] - h[2]) / 1e3);
        }
        const double r = med(rd);
        printf("  X %4.1f: %5.2f (%3.0f)", xm, r, 100 * (cold - r) / (cold - hot));
      }
      printf("\n");
    }
  }
}

int main(int argc, char** argv) {
  const std::string what = argc > 1 ? argv[1] : "all";
  Dev d;
  CK(cudaMalloc(&d.buf, kBig));
  CK(cudaMalloc(&d.fb, kFl));
  CK(cudaMalloc(&d.sink, 64));
  CK(cudaMalloc(&d.span, 16));
  CK(cudaMalloc(&d.m, 10 * sizeof(unsigned long long)));
  {  // lat's chain: a random cycle through the 128-byte lines of 64 MiB, as uint32 indices
    const size_t lines = (64ull << 20) / 128;
    std::vector<uint32_t> perm(lines), hc((64ull << 20) / 4, 0);
    for (size_t i = 0; i < lines; ++i) perm[i] = (uint32_t)i;
    uint64_t x = 88172645463325252ull;
    for (size_t i = lines - 1; i > 0; --i) {
      x ^= x << 13, x ^= x >> 7, x ^= x << 17;
      std::swap(perm[i], perm[x % (i + 1)]);
    }
    for (size_t i = 0; i < lines; ++i) hc[(size_t)perm[i] * 32] = perm[(i + 1) % lines] * 32;
    CK(cudaMalloc(&d.chain, 64ull << 20));
    CK(cudaMemcpy(d.chain, hc.data(), 64ull << 20, cudaMemcpyHostToDevice));
  }
  CK(cudaMemset(d.buf, 1, kBig));
  CK(cudaMemset(d.fb, 2, kFl));
  CK(cudaStreamCreateWithFlags(&d.s0, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&d.s1, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&d.ev, cudaEventDisableTiming));
  CK(cudaDeviceSynchronize());
  if (what == "old" || what == "all") old_tests(d);
  if (what == "paced" || what == "all") paced_tests(d);
  if (what == "window" || what == "all") window_tests(d);
  if (what == "lat" || what == "all") lat_tests(d);
  if (what == "between" || what == "all") between_tests(d);
  if (what == "evict" || what == "all") evict_tests(d);
  CK(cudaDeviceSynchronize());
  return 0;
}
