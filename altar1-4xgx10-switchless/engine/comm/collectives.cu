// Standalone collectives on the ring (collectives.h), built from the ring_dev.cuh halves.
#include "comm/collectives.h"
#include "comm/ring_dev.cuh"
#include "common/check.h"

namespace comm {
namespace {

constexpr int kThreads = 1024;  // one 16-byte chunk per thread up to 16 KiB in a single block

#ifdef GLM_COMM_TEST_HOOKS
// Test only: compiled into ring_test's copy of this library (glm53_comm_test) and out of every other build. One
// block of a pipelined reduce-scatter, all-to-all or multi-buffer all-gather stalls for ns after posting piece
// i + 1 and before reading piece i, at i = `piece` (or npieces - 4 if negative: the last piece whose send buffer
// the short last piece reuses). That holds open the window wait_posted closes (docs/ring.md, the pipelined
// reduce-scatters' race); the all-gather needs no wait, as each unit of its send buffer has one thread.
struct TestStall {
  uint32_t ns;
  int block, piece;
};
__device__ TestStall g_test_stall;
__device__ __forceinline__ void stall_for_test(size_t i, size_t npieces) {
  const TestStall t = g_test_stall;
  if (!t.ns || (int)blockIdx.x != t.block || (long)i != (t.piece < 0 ? (long)npieces - 4 : (long)t.piece)) return;
  if (threadIdx.x == 0) {
    uint64_t t0, tn;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tn));
    while (tn - t0 < t.ns);
  }
  __syncthreads();
}
#define TEST_STALL(i, npieces) stall_for_test(i, npieces)
#else
#define TEST_STALL(i, npieces) ((void)0)
#endif

// The sum of the four contributions' 16 bytes at offset off, as 16 bytes of the same element type.
template <bool kBf16>
__device__ __forceinline__ uint4 reduce16(const DevChannel& ch, uint64_t s, size_t off) {
  if constexpr (kBf16) {
    float f[8];
    sum8_bf16(ch, s, off, f);
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      __nv_bfloat162 b = __floats2bfloat162_rn(f[2 * j], f[2 * j + 1]);
      w[j] = *(uint32_t*)&b;
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
  } else {
    float4 a = sum4_f32(ch, s, off);
    return make_uint4(__float_as_uint(a.x), __float_as_uint(a.y), __float_as_uint(a.z), __float_as_uint(a.w));
  }
}

template <bool kBf16>
__global__ void __launch_bounds__(kThreads) allreduce_kernel(DevChannel ch, const uint4* in, uint4* out, uint32_t nbytes) {
  const uint64_t s = next_seq(ch);
  uint4* sb = (uint4*)send_buf(ch, s);
  const uint32_t chunks = nbytes / 16;
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (uint32_t c = t0; c < chunks; c += stride) sb[c] = in[c];
  post_grid(ch, s, nbytes);
  wait_all(ch, s);
  for (uint32_t c = t0; c < chunks; c += stride) out[c] = reduce16<kBf16>(ch, s, (size_t)c * 16);
  finish_grid(ch, s);
}

template <bool kBf16>
__global__ void __launch_bounds__(kThreads) reduce_scatter_kernel(DevChannel ch, const uint4* in, uint4* out, uint32_t nbytes) {
  const uint64_t s = next_seq(ch);
  uint4* sb = (uint4*)send_buf(ch, s);
  const uint32_t chunks = nbytes / 16, qchunks = chunks / 4;
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (uint32_t c = t0; c < chunks; c += stride) sb[c] = in[c];
  post_grid(ch, s, nbytes, kOpReduceScatter);
  wait_all(ch, s);
  const size_t qoff = (size_t)ch.rank * qchunks * 16;
  for (uint32_t c = t0; c < qchunks; c += stride) out[c] = reduce16<kBf16>(ch, s, qoff + (size_t)c * 16);
  finish_grid(ch, s);
}

// Two collectives per chunk: reduce-scatter s (each rank sums its quarter, straight into its send
// buffer for s + 1) and all-gather s + 1. Pipelined one deep: chunk i + 1's reduce-scatter is
// posted as soon as chunk i's all-gather has arrived, and chunk i is copied out while it travels.
// That posts s + 2 before s + 1 is consumed, which the three buffer sets allow (ring.h). wait_all's
// grid barrier keeps blocks in step, so no block overwrites a buffer another block still reads.
template <bool kBf16>
__global__ void __launch_bounds__(kThreads) allreduce_bulk_kernel(DevChannel ch, const uint4* in, uint4* out, size_t nbytes,
                                                                  uint32_t chunk) {
  static_assert(kSlots >= 3, "the pipelined bulk all-reduce needs three buffer sets");
  const uint64_t s0 = next_seq(ch);
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const size_t nchunks = (nbytes + chunk - 1) / chunk;
  auto len = [&](size_t i) { return (uint32_t)(nbytes - i * chunk < chunk ? nbytes - i * chunk : chunk); };
  // copy chunk i in and post its reduce-scatter (seq s)
  auto scatter = [&](size_t i, uint64_t s) {
    const uint32_t chunks = len(i) / 16;
    const uint4* cin = in + i * (chunk / 16);
    uint4* sb = (uint4*)send_buf(ch, s);
    for (uint32_t j = t0; j < chunks; j += stride) sb[j] = cin[j];
    post_grid(ch, s, len(i), kOpReduceScatter);
  };
  // reduce this rank's quarter of chunk i's reduce-scatter s and post it as all-gather s + 1
  auto reduce = [&](size_t i, uint64_t s) {
    const uint32_t qchunks = len(i) / 64;
    wait_all(ch, s);
    uint4* red = (uint4*)send_buf(ch, s + 1);
    const size_t qoff = (size_t)ch.rank * qchunks * 16;
    for (uint32_t j = t0; j < qchunks; j += stride) red[j] = reduce16<kBf16>(ch, s, qoff + (size_t)j * 16);
    post_grid(ch, s + 1, len(i) / 4);
  };
  // copy chunk i's all-gather s + 1 out (it has arrived)
  auto gather = [&](size_t i, uint64_t s) {
    const uint32_t qchunks = len(i) / 64;
    uint4* cout = out + i * (chunk / 16);
    for (int k = 0; k < kRanks; k++) {
      const uint8_t* p = src(ch, s + 1, k);
      for (uint32_t j = t0; j < qchunks; j += stride) cout[(size_t)k * qchunks + j] = ld_cv16(p + (size_t)j * 16);
    }
  };
  scatter(0, s0);
  reduce(0, s0);
  for (size_t i = 0; i < nchunks; i++) {
    const uint64_t s = s0 + 2 * i;
    wait_all(ch, s + 1);
    if (i + 1 < nchunks) scatter(i + 1, s + 2);
    gather(i, s);
    if (i + 1 < nchunks) reduce(i + 1, s + 2);
  }
  finish_grid(ch, s0 + 2 * nchunks - 1);
}

__global__ void __launch_bounds__(kThreads) allgather_kernel(DevChannel ch, const uint4* in, uint4* out, uint32_t nbytes) {
  const uint64_t s = next_seq(ch);
  uint4* sb = (uint4*)send_buf(ch, s);
  const uint32_t chunks = nbytes / 16;
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (uint32_t c = t0; c < chunks; c += stride) sb[c] = in[c];
  post_grid(ch, s, nbytes);
  wait_all(ch, s);
  for (int k = 0; k < kRanks; k++) {
    const uint8_t* p = src(ch, s, k);
    for (uint32_t c = t0; c < chunks; c += stride) out[(size_t)k * chunks + c] = ld_cv16(p + (size_t)c * 16);
  }
  finish_grid(ch, s);
}

// All-gather of more than cap bytes per rank: chunks of up to cap bytes, pipelined one deep like
// the bulk all-reduce (chunk j + 1 is posted as soon as chunk j has arrived, and chunk j is copied
// out while j + 1 travels).
__global__ void __launch_bounds__(kThreads) allgather_bulk_kernel(DevChannel ch, const uint4* in, uint4* out, size_t nbytes,
                                                                  uint32_t chunk) {
  const uint64_t s0 = next_seq(ch);
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const size_t nchunks = (nbytes + chunk - 1) / chunk;
  auto len = [&](size_t j) { return (uint32_t)(nbytes - j * chunk < chunk ? nbytes - j * chunk : chunk); };
  auto send = [&](size_t j, uint64_t s) {
    const uint32_t chunks = len(j) / 16;
    const uint4* cin = in + j * (chunk / 16);
    uint4* sb = (uint4*)send_buf(ch, s);
    for (uint32_t i = t0; i < chunks; i += stride) sb[i] = cin[i];
    post_grid(ch, s, len(j));
  };
  send(0, s0);
  for (size_t j = 0; j < nchunks; j++) {
    const uint64_t s = s0 + j;
    wait_all(ch, s);
    if (j + 1 < nchunks) send(j + 1, s + 1);
    const uint32_t chunks = len(j) / 16;
    for (int k = 0; k < kRanks; k++) {
      const uint8_t* p = src(ch, s, k);
      uint4* o = out + ((size_t)k * nbytes + j * chunk) / 16;
      for (uint32_t i = t0; i < chunks; i += stride) o[i] = ld_cv16(p + (size_t)i * 16);
    }
  }
  finish_grid(ch, s0 + nchunks - 1);
}

// Reduce-scatter of more than cap bytes: post i carries piece i (`piece` bytes, the last one shorter)
// of every quarter, so quarter k of the send buffer is again rank k's. A quarter is `rows` rows of
// `rb` bytes (row r of quarter k at in + (k rows + r) ld_in bytes; its sum at out + r ld_out), taken
// as one stream of rows * rb bytes. Pipelined with the copies ahead:
// piece i + 1 is copied into its send buffer while i travels, so that it is posted the moment i has
// arrived; then i is reduced and i + 2 copied in while i + 1 travels. The copy of s + 2 reuses the
// send buffer of s - 1, which every neighbour had received before it posted s (ring.h's rule), so the
// NIC is done with it once s has arrived. This grid's own blocks read their parts of its quarter
// while they reduce s - 1, which a slow block may still do: quarter k is at k len(i), so with a
// shorter last piece one block's copy lands on another block's part. So the copy first waits until
// every block has posted s + 1, which each does only after reducing s - 1.
template <bool kBf16>
__global__ void __launch_bounds__(kThreads) reduce_scatter_bulk_kernel(DevChannel ch, const uint8_t* in, size_t ld_in,
                                                                       uint8_t* out, size_t ld_out, uint32_t rows,
                                                                       uint32_t rb, uint32_t piece,
                                                                       const uint8_t* in2 = nullptr) {
  const uint64_t s0 = next_seq(ch);
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const size_t qbytes = (size_t)rows * rb, npieces = (qbytes + piece - 1) / piece;
  const uint32_t upr = rb / 16;  // 16-byte units per row
  auto len = [&](size_t i) { return (uint32_t)(qbytes - i * piece < piece ? qbytes - i * piece : piece); };
  // byte offset of 16-byte unit u of a quarter's stream, relative to the quarter's (out's) first row
  auto at = [&](size_t u, size_t ld) { return (u / upr) * ld + (u % upr) * 16; };
  auto copy = [&](size_t i, uint64_t s) {
    const uint32_t pl = len(i) / 16;
    const size_t u0 = i * (piece / 16);
    uint4* sb = (uint4*)send_buf(ch, s);
    for (int k = 0; k < kRanks; k++) {
      const uint8_t* q = in + (size_t)k * rows * ld_in;
      if (!kBf16 && in2) {  // f32 only: this rank's contribution is in + in2, one __fadd_rn an element
        const uint8_t* q2 = in2 + (size_t)k * rows * ld_in;
        for (uint32_t j = t0; j < pl; j += stride) {
          const float4 x = *(const float4*)(q + at(u0 + j, ld_in)), y = *(const float4*)(q2 + at(u0 + j, ld_in));
          const float4 z = make_float4(__fadd_rn(x.x, y.x), __fadd_rn(x.y, y.y), __fadd_rn(x.z, y.z), __fadd_rn(x.w, y.w));
          sb[(size_t)k * pl + j] = make_uint4(__float_as_uint(z.x), __float_as_uint(z.y), __float_as_uint(z.z),
                                              __float_as_uint(z.w));
        }
      } else {
        for (uint32_t j = t0; j < pl; j += stride) sb[(size_t)k * pl + j] = *(const uint4*)(q + at(u0 + j, ld_in));
      }
    }
  };
  copy(0, s0);
  post_grid(ch, s0, 4 * len(0), kOpReduceScatter);
  if (npieces > 1) copy(1, s0 + 1);
  for (size_t i = 0; i < npieces; i++) {
    const uint64_t s = s0 + i;
    wait_all(ch, s);
    if (i + 1 < npieces) post_grid(ch, s + 1, 4 * len(i + 1), kOpReduceScatter);
    TEST_STALL(i, npieces);
    const uint32_t pl = len(i) / 16;
    const size_t u0 = i * (piece / 16), qoff = (size_t)ch.rank * pl * 16;
    for (uint32_t j = t0; j < pl; j += stride)
      *(uint4*)(out + at(u0 + j, ld_out)) = reduce16<kBf16>(ch, s, qoff + (size_t)j * 16);
    if (i + 2 < npieces) {
      if (i > 0) wait_posted(ch, s + 1);  // every block is done with s - 1's send buffer (see above)
      copy(i + 2, s + 2);
    }
  }
  finish_grid(ch, s0 + npieces - 1);
}

// All-gather of several buffers: per rank they are one stream (the segments one after another),
// sent in chunks of up to cap bytes, the copies ahead as in reduce_scatter_bulk_kernel.
struct AgArgs {
  const uint8_t* in[kMaxAgSegs];
  uint8_t* out[kMaxAgSegs];
  size_t bytes[kMaxAgSegs];
  size_t off[kMaxAgSegs + 1];  // stream offset of each segment; off[nseg] = total
  int nseg;
};

__global__ void __launch_bounds__(kThreads) allgather_multi_kernel(DevChannel ch, AgArgs a, uint32_t chunk) {
  const uint64_t s0 = next_seq(ch);
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const size_t total = a.off[a.nseg], nchunks = (total + chunk - 1) / chunk;
  auto len = [&](size_t j) { return (uint32_t)(total - j * chunk < chunk ? total - j * chunk : chunk); };
  auto seg_of = [&](size_t x) {
    int g = 0;
    while (g + 1 < a.nseg && x >= a.off[g + 1]) ++g;
    return g;
  };
  auto copy = [&](size_t j, uint64_t s) {
    const uint32_t units = len(j) / 16;
    uint4* sb = (uint4*)send_buf(ch, s);
    for (uint32_t i = t0; i < units; i += stride) {
      const size_t x = j * chunk + (size_t)i * 16;
      const int g = seg_of(x);
      sb[i] = *(const uint4*)(a.in[g] + (x - a.off[g]));
    }
  };
  copy(0, s0);
  post_grid(ch, s0, len(0));
  if (nchunks > 1) copy(1, s0 + 1);
  for (size_t j = 0; j < nchunks; j++) {
    const uint64_t s = s0 + j;
    wait_all(ch, s);
    if (j + 1 < nchunks) post_grid(ch, s + 1, len(j + 1));
    TEST_STALL(j, nchunks);
    const uint32_t units = len(j) / 16;
    for (int k = 0; k < kRanks; k++) {
      const uint8_t* p = src(ch, s, k);
      for (uint32_t i = t0; i < units; i += stride) {
        const size_t x = j * chunk + (size_t)i * 16;
        const int g = seg_of(x);
        *(uint4*)(a.out[g] + k * a.bytes[g] + (x - a.off[g])) = ld_cv16(p + (size_t)i * 16);
      }
    }
    if (j + 2 < nchunks) copy(j + 2, s + 2);
  }
  finish_grid(ch, s0 + nchunks - 1);
}

// All-to-all by rows (alltoall_rows): the bulk reduce-scatter's pieces and pipeline, with every
// contribution's piece of this rank's quarter copied out to its own place instead of summed.
struct A2aArgs {
  const uint8_t* in[kMaxA2aSegs];  // quarter k of segment g at in[g] + k rows ld_in[g]
  uint8_t* out[kMaxA2aSegs];       // rank k's quarter for this rank at out[g] + k rows ld_out[g]
  size_t ld_in[kMaxA2aSegs], ld_out[kMaxA2aSegs], rb[kMaxA2aSegs];
  size_t off[kMaxA2aSegs + 1];  // stream offset of each segment's rows within a quarter; off[nseg] = total
  int nseg, rows;
};

constexpr int kA2aThreads = 256;

__global__ void __launch_bounds__(kA2aThreads) alltoall_rows_kernel(DevChannel ch, A2aArgs a, uint32_t piece) {
  const uint64_t s0 = next_seq(ch);
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const size_t qbytes = a.off[a.nseg], npieces = (qbytes + piece - 1) / piece;
  auto len = [&](size_t i) { return (uint32_t)(qbytes - i * piece < piece ? qbytes - i * piece : piece); };
  // stream offset x (a multiple of 16) -> its segment g and the byte offsets of its row in quarter 0 of
  // in and of out; quarter k is k * rows rows further
  auto where = [&](size_t x, int& g, size_t& o_in, size_t& o_out) {
    g = 0;
    while (g + 1 < a.nseg && x >= a.off[g + 1]) ++g;
    const size_t l = x - a.off[g], row = l / a.rb[g], col = l - row * a.rb[g];
    o_in = row * a.ld_in[g] + col, o_out = row * a.ld_out[g] + col;
  };
  auto copy = [&](size_t i, uint64_t s) {
    const uint32_t pl = len(i) / 16;
    uint4* sb = (uint4*)send_buf(ch, s);
    for (uint32_t j = t0; j < pl; j += stride) {
      int g;
      size_t oi, oo;
      where(i * piece + (size_t)j * 16, g, oi, oo);
      const size_t qs = (size_t)a.rows * a.ld_in[g];
      uint4 v[kRanks];
#pragma unroll
      for (int k = 0; k < kRanks; k++) v[k] = *(const uint4*)(a.in[g] + k * qs + oi);
#pragma unroll
      for (int k = 0; k < kRanks; k++) sb[(size_t)k * pl + j] = v[k];
    }
  };
  copy(0, s0);
  post_grid(ch, s0, 4 * len(0), kOpReduceScatter);
  if (npieces > 1) copy(1, s0 + 1);
  for (size_t i = 0; i < npieces; i++) {
    const uint64_t s = s0 + i;
    wait_all(ch, s);
    if (i + 1 < npieces) post_grid(ch, s + 1, 4 * len(i + 1), kOpReduceScatter);
    TEST_STALL(i, npieces);
    const uint32_t pl = len(i) / 16;
    const size_t qoff = (size_t)ch.rank * pl * 16;
    for (uint32_t j = t0; j < pl; j += stride) {
      int g;
      size_t oi, oo;
      where(i * piece + (size_t)j * 16, g, oi, oo);
      const size_t qs = (size_t)a.rows * a.ld_out[g];
      uint4 v[kRanks];
#pragma unroll
      for (int k = 0; k < kRanks; k++) v[k] = ld_cv16(src(ch, s, k) + qoff + (size_t)j * 16);
#pragma unroll
      for (int k = 0; k < kRanks; k++) *(uint4*)(a.out[g] + k * qs + oo) = v[k];
    }
    if (i + 2 < npieces) {
      if (i > 0) wait_posted(ch, s + 1);  // every block is done with s - 1's send buffer (see above)
      copy(i + 2, s + 2);
    }
  }
  finish_grid(ch, s0 + npieces - 1);
}

__global__ void __launch_bounds__(256) argmax_kernel(DevChannel ch, const ArgMax* in, ArgMax* out, int rows) {
  const uint64_t s = next_seq(ch);
  ArgMax* sb = (ArgMax*)send_buf(ch, s);
  for (int r = threadIdx.x; r < rows; r += blockDim.x) sb[r] = in[r];
  post_grid(ch, s, rows * sizeof(ArgMax));
  wait_all(ch, s);
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    ArgMax best{-__int_as_float(0x7f800000), 0x7fffffff};
    for (int k = 0; k < kRanks; k++) {
      const uint2 w = __ldcv((const uint2*)(src(ch, s, k)) + r);
      const ArgMax a{__uint_as_float(w.x), (int)w.y};
      if (a.v > best.v || (a.v == best.v && a.i < best.i)) best = a;
    }
    out[r] = best;
  }
  finish_grid(ch, s);
}

void check_size(const DevChannel& ch, size_t bytes, size_t align = 16) {
  if (bytes % align || bytes > ch.cap || !bytes)
    die("comm: collective of %zu bytes (a multiple of %zu, at most %u)", bytes, align, ch.cap);
}

template <bool kBf16>
void bulk(const DevChannel& ch, const void* in, void* out, size_t bytes, cudaStream_t st, int blocks) {
  if (bytes % 64 || !bytes) die("comm: bulk all-reduce of %zu bytes (a multiple of 64)", bytes);
  const uint32_t chunk = ch.cap & ~63u;
  blocks = blocks < 1 ? 1 : blocks > 48 ? 48 : blocks;
  allreduce_bulk_kernel<kBf16><<<blocks, kThreads, 0, st>>>(ch, (const uint4*)in, (uint4*)out, bytes, chunk);
  CUDA_CHECK(cudaGetLastError());
}

// rows of rb bytes per quarter (a multiple of 16), row strides in bytes
template <bool kBf16>
void rs_bulk(const DevChannel& ch, const void* in, size_t ld_in, void* out, size_t ld_out, size_t rows, size_t rb,
             cudaStream_t st, int blocks, const void* in2 = nullptr) {
  if (rb % 16 || !rb || !rows || rows * rb > 0xffffffffull * 16 || rb > 0xffffffffu || rows > 0xffffffffu)
    die("comm: bulk reduce-scatter of %zu rows of %zu bytes per quarter (a multiple of 16)", rows, rb);
  const uint32_t piece = (ch.cap / 4) & ~63u;
  blocks = blocks < 1 ? 1 : blocks > 48 ? 48 : blocks;
  reduce_scatter_bulk_kernel<kBf16><<<blocks, kThreads, 0, st>>>(ch, (const uint8_t*)in, ld_in, (uint8_t*)out, ld_out,
                                                                  (uint32_t)rows, (uint32_t)rb, piece,
                                                                  (const uint8_t*)in2);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int grid_for(size_t bytes) {
  const size_t chunks = (bytes + 15) / 16;
  int g = (int)((chunks + kThreads - 1) / kThreads);
  return g < 1 ? 1 : g > 48 ? 48 : g;
}

int block_for(size_t bytes) {
  const size_t chunks = (bytes + 15) / 16;
  return chunks >= kThreads ? kThreads : (int)((chunks + 31) / 32 * 32);
}

void allreduce_bf16(const DevChannel& ch, const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t st) {
  const size_t bytes = n * 2;
  check_size(ch, bytes);
  allreduce_kernel<true><<<grid_for(bytes), block_for(bytes), 0, st>>>(ch, (const uint4*)in, (uint4*)out, (uint32_t)bytes);
  CUDA_CHECK(cudaGetLastError());
}

void allreduce_f32(const DevChannel& ch, const float* in, float* out, size_t n, cudaStream_t st) {
  const size_t bytes = n * 4;
  check_size(ch, bytes);
  allreduce_kernel<false><<<grid_for(bytes), block_for(bytes), 0, st>>>(ch, (const uint4*)in, (uint4*)out, (uint32_t)bytes);
  CUDA_CHECK(cudaGetLastError());
}

void allgather(const DevChannel& ch, const void* in, void* out, size_t nbytes, cudaStream_t st) {
  if (nbytes > ch.cap) {
    if (nbytes % 16) die("comm: all-gather of %zu bytes (a multiple of 16)", nbytes);
    allgather_bulk_kernel<<<16, kThreads, 0, st>>>(ch, (const uint4*)in, (uint4*)out, nbytes, ch.cap & ~63u);
  } else {
    check_size(ch, nbytes);
    allgather_kernel<<<grid_for(nbytes), block_for(nbytes), 0, st>>>(ch, (const uint4*)in, (uint4*)out, (uint32_t)nbytes);
  }
  CUDA_CHECK(cudaGetLastError());
}

void reduce_scatter_bf16(const DevChannel& ch, const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t st) {
  const size_t bytes = n * 2;
  check_size(ch, bytes, 64);
  reduce_scatter_kernel<true><<<grid_for(bytes), block_for(bytes), 0, st>>>(ch, (const uint4*)in, (uint4*)out, (uint32_t)bytes);
  CUDA_CHECK(cudaGetLastError());
}

void reduce_scatter_f32(const DevChannel& ch, const float* in, float* out, size_t n, cudaStream_t st) {
  const size_t bytes = n * 4;
  check_size(ch, bytes, 64);
  reduce_scatter_kernel<false><<<grid_for(bytes), block_for(bytes), 0, st>>>(ch, (const uint4*)in, (uint4*)out, (uint32_t)bytes);
  CUDA_CHECK(cudaGetLastError());
}

void allreduce_bulk_bf16(const DevChannel& ch, const __nv_bfloat16* in, __nv_bfloat16* out, size_t n, cudaStream_t st, int blocks) {
  bulk<true>(ch, in, out, n * 2, st, blocks);
}

void allreduce_bulk_f32(const DevChannel& ch, const float* in, float* out, size_t n, cudaStream_t st, int blocks) {
  bulk<false>(ch, in, out, n * 4, st, blocks);
}

void reduce_scatter_bulk_bf16(const DevChannel& ch, const __nv_bfloat16* in, __nv_bfloat16* out, size_t n,
                              cudaStream_t st, int blocks) {
  if (n * 2 % 64) die("comm: bulk reduce-scatter of %zu bytes (a multiple of 64)", n * 2);
  rs_bulk<true>(ch, in, n / 2, out, n / 2, 1, n / 2, st, blocks);
}

void reduce_scatter_bulk_f32(const DevChannel& ch, const float* in, float* out, size_t n, cudaStream_t st, int blocks) {
  if (n * 4 % 64) die("comm: bulk reduce-scatter of %zu bytes (a multiple of 64)", n * 4);
  rs_bulk<false>(ch, in, n, out, n, 1, n, st, blocks);
}

void reduce_scatter_rows_bf16(const DevChannel& ch, const __nv_bfloat16* in, size_t ld_in, __nv_bfloat16* out,
                              size_t ld_out, int rows, int cols, cudaStream_t st, int blocks) {
  rs_bulk<true>(ch, in, ld_in * 2, out, ld_out * 2, rows, (size_t)cols * 2, st, blocks);
}

void reduce_scatter_rows_f32(const DevChannel& ch, const float* in, size_t ld_in, float* out, size_t ld_out, int rows,
                             int cols, cudaStream_t st, int blocks) {
  rs_bulk<false>(ch, in, ld_in * 4, out, ld_out * 4, rows, (size_t)cols * 4, st, blocks);
}

void reduce_scatter_rows_f32_sum2(const DevChannel& ch, const float* in, const float* in2, size_t ld_in, float* out,
                                  size_t ld_out, int rows, int cols, cudaStream_t st, int blocks) {
  rs_bulk<false>(ch, in, ld_in * 4, out, ld_out * 4, rows, (size_t)cols * 4, st, blocks, in2);
}

void allgather_multi(const DevChannel& ch, const AgSeg* segs, int nseg, cudaStream_t st, int blocks) {
  if (nseg < 1 || nseg > kMaxAgSegs) die("comm: all-gather of %d segments (1 to %d)", nseg, kMaxAgSegs);
  AgArgs a{};
  a.nseg = nseg;
  for (int g = 0; g < nseg; ++g) {
    if (segs[g].bytes % 16 || !segs[g].bytes) die("comm: all-gather segment of %zu bytes (a multiple of 16)", segs[g].bytes);
    a.in[g] = (const uint8_t*)segs[g].in, a.out[g] = (uint8_t*)segs[g].out, a.bytes[g] = segs[g].bytes;
    a.off[g + 1] = a.off[g] + segs[g].bytes;
  }
  blocks = blocks < 1 ? 1 : blocks > 48 ? 48 : blocks;
  allgather_multi_kernel<<<blocks, kThreads, 0, st>>>(ch, a, ch.cap & ~63u);
  CUDA_CHECK(cudaGetLastError());
}

void alltoall_rows(const DevChannel& ch, const A2aSeg* segs, int nseg, int rows, cudaStream_t st, int blocks) {
  if (nseg < 1 || nseg > kMaxA2aSegs || rows < 1) die("comm: all-to-all of %d segments of %d rows", nseg, rows);
  A2aArgs a{};
  a.nseg = nseg, a.rows = rows;
  for (int g = 0; g < nseg; ++g) {
    const A2aSeg& e = segs[g];
    if (!e.rb || e.rb % 16 || e.ld_in % 16 || e.ld_out % 16 || e.rb > e.ld_in || e.rb > e.ld_out)
      die("comm: all-to-all segment of %zu-byte rows (strides %zu, %zu; multiples of 16)", e.rb, e.ld_in, e.ld_out);
    a.in[g] = (const uint8_t*)e.in, a.out[g] = (uint8_t*)e.out;
    a.ld_in[g] = e.ld_in, a.ld_out[g] = e.ld_out, a.rb[g] = e.rb;
    a.off[g + 1] = a.off[g] + (size_t)rows * e.rb;
  }
  blocks = blocks < 1 ? 1 : blocks > 48 ? 48 : blocks;
  alltoall_rows_kernel<<<blocks, kA2aThreads, 0, st>>>(ch, a, (ch.cap / 4) & ~63u);
  CUDA_CHECK(cudaGetLastError());
}

#ifdef GLM_COMM_TEST_HOOKS
void test_stall(uint32_t ns, int block, int piece, cudaStream_t st) {
  const TestStall t{ns, block, piece};
  CUDA_CHECK(cudaMemcpyToSymbolAsync(g_test_stall, &t, sizeof t, 0, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
}
#endif

void argmax_rows(const DevChannel& ch, const ArgMax* in, ArgMax* out, int rows, cudaStream_t st) {
  check_size(ch, rows * sizeof(ArgMax));
  argmax_kernel<<<1, 256, 0, st>>>(ch, in, out, rows);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace comm
