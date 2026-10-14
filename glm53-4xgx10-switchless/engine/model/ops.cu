// The runner's small kernels (ops.h).
#include <cfloat>
#include <climits>

#include "attn/launch.h"
#include "common/check.h"
#include "model/ops.h"

namespace glm::model::ops {
namespace {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_go() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float bf2f(uint32_t b) { return __uint_as_float((b & 0xFFFFu) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfround(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ void st_release_sys_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__global__ void k_step_in(const StepIn* ring, uint32_t* counter, const int32_t* fb, StepIn* dev) {
  pdl_wait();
  const uint32_t c = *counter;
  const StepIn* src = ring + c % kInSlots;
  const int words = sizeof(StepIn) / 4;
  for (int i = threadIdx.x; i < words; i += blockDim.x) {
    uint32_t v = reinterpret_cast<const volatile uint32_t*>(src)[i];
    if (i < kMaxRows && (int32_t)v < 0) v = (uint32_t)fb[-1 - (int32_t)v];  // tokens[] come first
    reinterpret_cast<uint32_t*>(dev)[i] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) *counter = c + 1;
  pdl_go();
}

// one block a row; 768 uint4 of a row's 6144 bf16
__global__ void k_embed_local(const uint16_t* table, const int32_t* tokens, int rank, uint16_t* ebuf) {
  pdl_wait();
  const int r = blockIdx.x;
  const int t = tokens[r] - rank * kVocabRank;
  uint4* o = reinterpret_cast<uint4*>(ebuf + ((size_t)r * kRanks + rank) * kDim);
  const bool mine = t >= 0 && t < kVocabRank;
  const uint4* in = reinterpret_cast<const uint4*>(table + (size_t)(mine ? t : 0) * kDim);
  for (int i = threadIdx.x; i < kDim / 8; i += blockDim.x) o[i] = mine ? in[i] : make_uint4(0, 0, 0, 0);
  pdl_go();
}

__global__ void k_embed_pick(const uint16_t* ebuf, const int32_t* tokens, const int32_t* pos, int zero_pos0, uint16_t* out) {
  pdl_wait();
  const int r = blockIdx.x;
  int k = tokens[r] / kVocabRank;
  k = k < 0 ? 0 : k > kRanks - 1 ? kRanks - 1 : k;
  const bool zero = zero_pos0 && pos[r] == 0;
  const uint4* in = reinterpret_cast<const uint4*>(ebuf + ((size_t)r * kRanks + k) * kDim);
  uint4* o = reinterpret_cast<uint4*>(out + (size_t)r * kDim);
  for (int i = threadIdx.x; i < kDim / 8; i += blockDim.x) o[i] = zero ? make_uint4(0, 0, 0, 0) : in[i];
  pdl_go();
}

__global__ void k_mtp_tokens(const StepIn* in, const int32_t* fb, int R, int32_t* tok, int32_t* pos, int32_t* sel,
                             int32_t* acc) {
  pdl_wait();
  if (threadIdx.x == 0) {  // at most 16 rows: one thread, session by session
    const bool spec = in->spec != 0;
    for (int b = 0; b < R;) {
      const int s = in->sess[b];
      int e = b + 1;
      while (e < R && in->sess[e] == s) ++e;
      int a = 0;
      if (spec)
        while (b + a + 1 < e && fb[b + a] == in->tokens[b + a + 1]) ++a;
      else
        a = e - b - 1;
      sel[s] = b + a, acc[s] = a;
      for (int r = b; r < e; ++r) tok[r] = r < b + a ? in->tokens[r + 1] : fb[r], pos[r] = in->pos[r];
      b = e;
    }
  }
  pdl_go();
}

// one block a session: its chain row's hidden, normed hidden and top-2048
__global__ void k_mtp_select(const StepIn* in, const int32_t* sel, const int32_t* pos, const uint16_t* h, const uint16_t* xn,
                             const int* topk, uint16_t* ch, uint16_t* cxn, int* ctopk, int32_t* cpos, int32_t* cslot) {
  pdl_wait();
  const int s = blockIdx.x, r = sel[s], p = pos[r];
  const uint4* hi = reinterpret_cast<const uint4*>(h + (size_t)r * kDim);
  const uint4* xi = reinterpret_cast<const uint4*>(xn + (size_t)r * kDim);
  uint4* ho = reinterpret_cast<uint4*>(ch + (size_t)s * kDim);
  uint4* xo = reinterpret_cast<uint4*>(cxn + (size_t)s * kDim);
  for (int i = threadIdx.x; i < kDim / 8; i += blockDim.x) ho[i] = hi[i], xo[i] = xi[i];
  for (int i = threadIdx.x; i < ::glm::kTopK; i += blockDim.x)
    ctopk[(size_t)s * ::glm::kTopK + i] = p >= ::glm::kTopK ? topk[(size_t)r * ::glm::kTopK + i] : i;
  if (threadIdx.x == 0) cpos[s] = p, cslot[s] = in->slot[r];
  pdl_go();
}

__global__ void k_mtp_next(const comm::glm::Cand* best, int R, int32_t* tok, int32_t* pos, int32_t* draft_k) {
  pdl_wait();
  const int r = threadIdx.x;
  if (r < R) {
    tok[r] = best[r].i;
    pos[r] = pos[r] + 1;
    if (draft_k) draft_k[r] = best[r].i;
  }
  pdl_go();
}

__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

__global__ void __launch_bounds__(1024) k_local_top1(const float* logits, int vocab0, comm::glm::Cand* out) {
  pdl_wait();
  __shared__ float sv[32];
  __shared__ int si[32];
  const int r = blockIdx.x;
  const float* l = logits + (size_t)r * kVocabRank;
  float m = -INFINITY;
  int mi = INT_MAX;
  for (int i = threadIdx.x; i < kVocabRank; i += blockDim.x) {
    const int id = vocab0 + i;
    const float v = id < kVocabUsed ? l[i] : -INFINITY;
    if (better(v, id, m, mi)) m = v, mi = id;
  }
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(~0u, m, o);
    const int oi = __shfl_xor_sync(~0u, mi, o);
    if (better(ov, oi, m, mi)) m = ov, mi = oi;
  }
  if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = m, si[threadIdx.x >> 5] = mi;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
      if (better(sv[k], si[k], m, mi)) m = sv[k], mi = si[k];
    out[r] = comm::glm::Cand{m, mi};
  }
  pdl_go();
}

__global__ void k_publish(SampleOut* outs, const StepIn* in, const int32_t* draft, const int32_t* acc, int ndraft,
                          int nsess) {
  pdl_wait();
  SampleOut& o = outs[in->step % kOutSlots];
  for (int i = threadIdx.x; i < kMaxSess * kMaxDrafts; i += blockDim.x) {
    const int s = i / kMaxDrafts, k = i % kMaxDrafts;
    o.draft[s][k] = s < nsess && k < ndraft ? draft[k * kMaxSess + s] : -1;
  }
  for (int s = threadIdx.x; s < kMaxSess; s += blockDim.x) o.accepted[s] = s < nsess ? acc[s] : -1;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    o.drafts = ndraft;
    __threadfence_system();
    st_release_sys_u32(&o.seq, in->step + 1);
  }
  pdl_go();
}

__global__ void k_add_rows(const uint16_t* res, const float* a, const float* b, size_t n, uint16_t* h) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float y = bfround(a[i]);
  if (b) y = bfround(__fadd_rn(y, bfround(b[i])));
  h[i] = f2bf(__fadd_rn(bf2f(res[i]), y));
}

__global__ void k_iota(int32_t* pos, int32_t* slot, int p0, int sl, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pos[i] = p0 + i, slot[i] = sl;
}

__global__ void k_copy_rows(const uint16_t* in, size_t ldi, uint16_t* out, size_t ldo, int cols) {
  const int r = blockIdx.x;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) out[(size_t)r * ldo + c] = in[(size_t)r * ldi + c];
}

}  // namespace

void step_in(const StepIn* ring, uint32_t* counter, const int32_t* fb, StepIn* dev, cudaStream_t s, bool pdl) {
  ::glm::launch(k_step_in, dim3(1), dim3(256), 0, s, pdl, ring, counter, fb, dev);
}
void embed_local(const uint16_t* table, const int32_t* tokens, int R, int rank, uint16_t* ebuf, cudaStream_t s, bool pdl) {
  ::glm::launch(k_embed_local, dim3(R), dim3(256), 0, s, pdl, table, tokens, rank, ebuf);
}
void embed_pick(const uint16_t* ebuf, const int32_t* tokens, const int32_t* pos, bool zero_pos0, int R, uint16_t* out,
                cudaStream_t s, bool pdl) {
  ::glm::launch(k_embed_pick, dim3(R), dim3(256), 0, s, pdl, ebuf, tokens, pos, zero_pos0 ? 1 : 0, out);
}
void mtp_tokens(const StepIn* in, const int32_t* fb, int R, int32_t* tok, int32_t* pos, int32_t* sel, int32_t* acc,
                cudaStream_t s, bool pdl) {
  ::glm::launch(k_mtp_tokens, dim3(1), dim3(32), 0, s, pdl, in, fb, R, tok, pos, sel, acc);
}
void mtp_select(const StepIn* in, const int32_t* sel, int nsess, const int32_t* pos, const uint16_t* h, const uint16_t* xn,
                const int* topk, uint16_t* ch, uint16_t* cxn, int* ctopk, int32_t* cpos, int32_t* cslot, cudaStream_t s,
                bool pdl) {
  ::glm::launch(k_mtp_select, dim3(nsess), dim3(256), 0, s, pdl, in, sel, pos, h, xn, topk, ch, cxn, ctopk, cpos, cslot);
}
void mtp_next(const comm::glm::Cand* best, int R, int32_t* tok, int32_t* pos, int32_t* draft_k, cudaStream_t s, bool pdl) {
  ::glm::launch(k_mtp_next, dim3(1), dim3(32), 0, s, pdl, best, R, tok, pos, draft_k);
}
void local_top1(const float* logits, int R, int vocab0, comm::glm::Cand* out, cudaStream_t s, bool pdl) {
  ::glm::launch(k_local_top1, dim3(R), dim3(1024), 0, s, pdl, logits, vocab0, out);
}
void publish(SampleOut* outs, const StepIn* in, const int32_t* draft, const int32_t* acc, int ndraft, int nsess,
             cudaStream_t s, bool pdl) {
  ::glm::launch(k_publish, dim3(1), dim3(128), 0, s, pdl, outs, in, draft, acc, ndraft, nsess);
}
void add_rows(const uint16_t* res, const float* a, const float* b, size_t n, uint16_t* h, cudaStream_t s) {
  if (!n) return;
  k_add_rows<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(res, a, b, n, h);
  CUDA_CHECK(cudaGetLastError());
}
void iota(int32_t* pos, int32_t* slot, int p0, int sl, int n, cudaStream_t s) {
  if (n <= 0) return;
  k_iota<<<(n + 255) / 256, 256, 0, s>>>(pos, slot, p0, sl, n);
  CUDA_CHECK(cudaGetLastError());
}
void copy_rows(const uint16_t* in, size_t ldi, uint16_t* out, size_t ldo, int n, int cols, cudaStream_t s) {
  if (n <= 0) return;
  k_copy_rows<<<n, 256, 0, s>>>(in, ldi, out, ldo, cols);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace glm::model::ops
