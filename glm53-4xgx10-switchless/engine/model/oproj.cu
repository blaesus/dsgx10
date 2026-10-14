// o_proj as the producer half of its all-reduce (Tune kTuneOproj; docs/engine.md, "Closing the decode gap"): attn's
// FP8 GEMV (attn/gemv.cu k_gemv, F32 out, split-K 1) with the same arithmetic bit for bit (per 64-k chunk four
// m16n8k16 MMAs from zero, acc = acc + s * c in k order), its outputs written straight into the send buffer of the
// ring's next collective and posted by the grid's last block, so the all-reduce kernel is the consumer half only
// (comm::glm::allreduce_add_norm_posted) and the network time starts as the GEMV's last block ends.
#include "attn/attn_common.cuh"
#include "comm/ring_dev.cuh"
#include "common/check.h"
#include "model/impl.h"

namespace glm::model::detail {
namespace {

constexpr int kN = kDim, kK = ::glm::kOProjK;  // [6144][4096] e4m3 tiles, f32 scales [48][32] (128 x 128 blocks)
constexpr int kJ = kK / 64;

template <bool HI, int P, int NWP>
__global__ void __launch_bounds__(NWP * 32) k_oproj_post(const uint16_t* __restrict__ x, int rows, const uint8_t* __restrict__ w,
                                                         const float* __restrict__ sc, comm::DevChannel ch) {
  using namespace ::glm;
  pdl_trigger();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int lt = blockIdx.x * NWP + warp, n0 = lt * 8;  // every tile valid: 768 tiles, NWP | 768
  const uint8_t* wp = w + ((size_t)lt * kJ) * 512 + lane * 16;
  const float* sp = sc + (size_t)(n0 >> 7) * (kK >> 7);
  const uint16_t* xb = x + 16 * t;
  const bool live0 = g < rows, live1 = HI && g + 8 < rows;
  const uint16_t* x0 = xb + (size_t)(live0 ? g : 0) * kK;
  const uint16_t* x1 = xb + (size_t)(live1 ? g + 8 : 0) * kK;
  uint4 buf[P];
#pragma unroll
  for (int u = 0; u < P; ++u) buf[u] = ldg_stream128(wp + (size_t)u * 512);
  pdl_wait();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  auto comp = [&](const uint4 b, int j) {
    uint32_t o[8];
    fp8x4_to_bf16(b.x, o[0], o[1]);
    fp8x4_to_bf16(b.y, o[2], o[3]);
    fp8x4_to_bf16(b.z, o[4], o[5]);
    fp8x4_to_bf16(b.w, o[6], o[7]);
    uint4 xa = make_uint4(0, 0, 0, 0), xc = xa, ya = xa, yc = xa;
    if (live0) {
      xa = __ldg(reinterpret_cast<const uint4*>(x0 + j * 64));
      xc = __ldg(reinterpret_cast<const uint4*>(x0 + j * 64 + 8));
    }
    if (live1) {
      ya = __ldg(reinterpret_cast<const uint4*>(x1 + j * 64));
      yc = __ldg(reinterpret_cast<const uint4*>(x1 + j * 64 + 8));
    }
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    mma_bf16(c, xa.x, ya.x, xa.y, ya.y, o[0], o[1]);
    mma_bf16(c, xa.z, ya.z, xa.w, ya.w, o[2], o[3]);
    mma_bf16(c, xc.x, yc.x, xc.y, yc.y, o[4], o[5]);
    mma_bf16(c, xc.z, yc.z, xc.w, yc.w, o[6], o[7]);
    const float s = __ldg(sp + ((j * 64) >> 7));
    acc[0] = __fadd_rn(acc[0], __fmul_rn(s, c[0]));
    acc[1] = __fadd_rn(acc[1], __fmul_rn(s, c[1]));
    acc[2] = __fadd_rn(acc[2], __fmul_rn(s, c[2]));
    acc[3] = __fadd_rn(acc[3], __fmul_rn(s, c[3]));
  };
  for (int j = 0; j < kJ; j += P) {
#pragma unroll
    for (int u = 0; u < P; ++u) {
      const int jj = j + u;
      const uint4 b = buf[u];
      if (jj + P < kJ) buf[u] = ldg_stream128(wp + (size_t)(jj + P) * 512);
      comp(b, jj);
    }
  }
  // the collective about to run: every earlier one has finished (this grid is past its wait), so its send buffer's
  // last reader, three collectives back, is done
  const uint64_t s = comm::next_seq(ch);
  float* y = reinterpret_cast<float*>(comm::send_buf(ch, s));
  const int n = n0 + 2 * t;
  if (live0) *reinterpret_cast<float2*>(y + (size_t)g * kN + n) = make_float2(acc[0], acc[1]);
  if (live1) *reinterpret_cast<float2*>(y + (size_t)(g + 8) * kN + n) = make_float2(acc[2], acc[3]);
  comm::post_grid(ch, s, (uint32_t)((size_t)rows * kN * 4));
}

template <bool HI>
void launch(const uint16_t* x, int rows, const Fp8W& w, const comm::DevChannel& ch, cudaStream_t st, bool pdl) {
  constexpr int P = 8, NWP = 8;
  static_assert(kJ % P == 0 && (kN / 8) % NWP == 0, "o_proj tiling");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kN / 8 / NWP);
  cfg.blockDim = dim3(NWP * 32);
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_oproj_post<HI, P, NWP>, x, rows, w.w, w.s, ch));
}

}  // namespace

void oproj_post(const uint16_t* x, int rows, const Fp8W& w, const comm::DevChannel& ch, cudaStream_t st, bool pdl) {
  if (rows < 1 || rows > kMaxRows || (size_t)rows * kN * 4 > ch.cap) die("oproj_post: %d rows", rows);
  if (rows > 8) launch<true>(x, rows, w, ch, st, pdl);
  else launch<false>(x, rows, w, ch, st, pdl);
}

}  // namespace glm::model::detail
