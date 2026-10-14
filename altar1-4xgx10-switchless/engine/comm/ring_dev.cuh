// Collectives for the four-box ring: device side (see ring.h).
//
// A collective has a producer half and a consumer half, which can live in different kernels so they
// fuse into neighbouring work (a GEMV epilogue publishes, the next kernel's prologue reduces):
//
//   uint64_t s = comm::next_seq(ch);                 // every block, at kernel start
//   ... write this rank's contribution to comm::send_buf(ch, s) ...
//   comm::post_grid(ch, s, nbytes);                  // every block; the last one rings the proxy
//                                                    // (reduce-scatter: post_grid(ch, s, nbytes, kOpReduceScatter))
//   --- (same or a later kernel; that kernel also reads s = next_seq(ch)) ---
//   comm::wait_all(ch, s);                           // every block
//   ... read comm::src(ch, s, k) for k = 0..3, e.g. comm::sum8_bf16 ...
//   comm::finish_grid(ch, s);                        // every block; the last one advances ch.seq
//
// Rules: collectives run in the same order on every rank; producer and consumer of one collective
// read the same next_seq (so nothing may call finish_grid in between); a rank consumes s - 1 before
// it posts s (or, at most, s - (kSlots - 1) after seeing s - 1 arrive, see ring.h); a block writes
// send_buf(ch, s) only once every block of its grid is done reading s - kSlots there (wait_posted);
// a kernel that waits must not have more blocks than can be resident at once (the last producer
// block must run).
#pragma once
#include <cuda_bf16.h>
#include <type_traits>

#include "comm/ring.h"

namespace comm {

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t ld_acquire_gpu(const unsigned long long* p) {
  uint64_t v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned long long* p, uint64_t v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Sequence number of the collective about to run. Every block of the kernel reads the same value.
__device__ __forceinline__ uint64_t next_seq(const DevChannel& ch) {
  return *(volatile unsigned long long*)ch.seq + 1;
}

__device__ __forceinline__ int slot(uint64_t s) { return (int)(s % kSlots); }

// Where this rank writes its contribution to collective s (host memory, GPU-mapped).
__device__ __forceinline__ uint8_t* send_buf(const DevChannel& ch, uint64_t s) {
  return ch.src[slot(s)][ch.rank];
}

// Where rank k's contribution to collective s is, once wait_all(ch, s) has returned.
__device__ __forceinline__ const uint8_t* src(const DevChannel& ch, uint64_t s, int k) {
  return ch.src[slot(s)][k];
}

// Producer: every block calls this after writing its part of send_buf. The last block to arrive
// publishes {s, nbytes, op} to the proxy. A single-block grid needs only the release store; with
// more blocks each one fences its stores at GPU scope before counting itself in, and the last one
// fences at system scope (cumulative over what it acquired) before publishing.
__device__ __forceinline__ void post_grid(const DevChannel& ch, uint64_t s, uint32_t nbytes, Op op = kOpAllExchange) {
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const unsigned total = gridDim.x * gridDim.y * gridDim.z;
    bool last = total == 1;
    if (!last) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      last = atomicAdd(ch.arrive, 1u) == total - 1;
      if (last) {
        *ch.arrive = 0;
        asm volatile("fence.acq_rel.sys;" ::: "memory");
      }
    }
    if (last) {
      Flag* f = ch.post[slot(s)];
      st_relaxed_sys_u32(&f->nbytes, nbytes);
      st_relaxed_sys_u32(&f->op, op);
      st_release_sys(&f->seq, s);
    }
  }
}

// Consumer: every block waits until the four contributions to collective s are complete. In block
// 0, lanes 0-5 of the first warp each poll one arrival flag and lane 6 this rank's own doorbell, so
// no block gets past wait_all(s) before every block of the grid has passed post_grid(s) (a grid
// barrier, which kernels that run several collectives in a row rely on). Block 0 then passes the
// news on through device memory, which the other blocks poll instead of host memory. Acquire loads
// order the data reads after.
// With before_release, every thread of block 0 runs it once the contributions are complete and before
// the other blocks are let go, so what it leaves in device memory is there when their wait_all returns
// (read it past L1, e.g. __ldcg): block 0 can read a piece every block needs from host memory once,
// instead of every block reading the same lines.
struct NoHook {
  __device__ void operator()() const {}
};
template <class F = NoHook>
__device__ __forceinline__ void wait_all(const DevChannel& ch, uint64_t s, F before_release = {}) {
  const bool first_block = blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
  const bool t0 = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
  if (first_block) {
    if (threadIdx.x < 32 && threadIdx.y == 0 && threadIdx.z == 0) {
      const int lane = threadIdx.x;
      const uint64_t* f = lane < kNumWait ? &ch.wait[slot(s)][lane]->seq : lane == kNumWait ? &ch.post[slot(s)]->seq : nullptr;
      bool ok = lane > kNumWait;
      while (!__all_sync(0xffffffffu, ok)) {
        if (!ok) ok = ld_acquire_sys(f) == s;
      }
    }
    __syncthreads();
    if constexpr (!std::is_same_v<F, NoHook>) {
      before_release();
      __syncthreads();
    }
    if (t0 && gridDim.x * gridDim.y * gridDim.z > 1) st_release_gpu(ch.ready, s);
  } else {
    if (t0)
      while (ld_acquire_gpu(ch.ready) < s) {
      }
    __syncthreads();
  }
}

// Every block of the grid has passed post_grid(ch, s): its last block publishes s only then, so
// whatever any block read before its post_grid(s) has been read. A pipelined kernel calls this (every
// thread of the block) before it rewrites a send buffer other blocks may still be reading.
__device__ __forceinline__ void wait_posted(const DevChannel& ch, uint64_t s) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    while (ld_acquire_sys(&ch.post[slot(s)]->seq) < s) {
    }
  __syncthreads();
}

// Consumer: every block calls this when it no longer reads the contributions; the last block to
// leave advances the device-side counter, which makes the next next_seq() return s + 1.
__device__ __forceinline__ void finish_grid(const DevChannel& ch, uint64_t s) {
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const unsigned total = gridDim.x * gridDim.y * gridDim.z;
    if (total == 1 || atomicAdd(ch.depart, 1u) == total - 1) {
      if (total > 1) *ch.depart = 0;
      *(volatile unsigned long long*)ch.seq = s;
    }
  }
}

// Received data is read with ld.global.cv: never served from a stale cached line of an earlier
// collective that used the same buffer set.
__device__ __forceinline__ uint4 ld_cv16(const uint8_t* p) { return __ldcv((const uint4*)p); }

__device__ __forceinline__ void bf16x8_to_f32(uint4 v, float f[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; j++) {
    f[2 * j] = __uint_as_float(w[j] << 16);
    f[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
  }
}

// Fixed-order sum of the 16-byte chunk at byte offset off: ((c0 + c1) + c2) + c3 in fp32.
__device__ __forceinline__ void sum8_bf16(const DevChannel& ch, uint64_t s, size_t off, float out[8]) {
  bf16x8_to_f32(ld_cv16(src(ch, s, 0) + off), out);
#pragma unroll
  for (int k = 1; k < kRanks; k++) {
    float f[8];
    bf16x8_to_f32(ld_cv16(src(ch, s, k) + off), f);
#pragma unroll
    for (int j = 0; j < 8; j++) out[j] += f[j];
  }
}
__device__ __forceinline__ float4 sum4_f32(const DevChannel& ch, uint64_t s, size_t off) {
  uint4 v = ld_cv16(src(ch, s, 0) + off);
  float4 a = make_float4(__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z), __uint_as_float(v.w));
#pragma unroll
  for (int k = 1; k < kRanks; k++) {
    v = ld_cv16(src(ch, s, k) + off);
    a.x += __uint_as_float(v.x);
    a.y += __uint_as_float(v.y);
    a.z += __uint_as_float(v.z);
    a.w += __uint_as_float(v.w);
  }
  return a;
}

}  // namespace comm
