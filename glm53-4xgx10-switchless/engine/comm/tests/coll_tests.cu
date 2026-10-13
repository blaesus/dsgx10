// GLM-5.3's collectives (comm/glmcoll.h) on the four ranks: correctness against host references and against
// the separate kernels they replace, bit for bit and the same on every rank and run to run; then their
// latency at 1-16 rows, a whole decode step's collectives in one graph, and the prefill paths.
// Every rank runs the same command (engine/comm/run4.sh does that); rank 0 prints the report and, with --out,
// appends its measurements to a TSV file.
//   coll_tests [--fleet FILE] [--rank r] [--cap bytes] [--rounds n] [--out FILE] [--tag TEXT] test|bench|step|prefill ...
#include <cuda_bf16.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "comm/collectives.h"
#include "comm/glmcoll.h"
#include "comm/ring.h"
#include "common/check.h"

using comm::DevChannel;
using comm::Ring;
namespace G = comm::glm;

namespace {

constexpr int kDim = G::kDim;

uint32_t mix(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return (uint32_t)x;
}
uint32_t gen(int rank, uint32_t salt, size_t i) { return mix(((uint64_t)salt << 40) ^ ((uint64_t)rank << 32) ^ i); }
// a value with a random sign and mantissa and an exponent over 2^-12..2^12: f32 sums depend on their order
float test_value(int rank, size_t i, uint32_t salt) {
  const uint32_t h = gen(rank, salt, i);
  const float m = 1.0f + (h & 0xffff) / 65536.0f;
  const int e = (int)((h >> 16) % 25) - 12;
  return ((h >> 31) ? -1.0f : 1.0f) * std::ldexp(m, e);
}
float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t f2bf(float f) {  // round to nearest even (no NaNs here)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bfr(float f) { return bf2f(f2bf(f)); }

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void sync_or_die(Ring& ring, cudaStream_t st, double timeout_s, const char* what) {
  const double t0 = now_s();
  for (;;) {
    const cudaError_t e = cudaStreamQuery(st);
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

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
  return h;
}

// every rank's count of wrong values and output hash; a failure anywhere fails the case on every rank
struct Checker {
  Ring& ring;
  int failures = 0;
  void report(const std::string& what, size_t bad, size_t total, uint64_t hash) {
    uint64_t mine[2] = {bad, hash}, all[8];
    ring.allgather_host(mine, all, sizeof mine);
    const bool same = all[1] == all[3] && all[3] == all[5] && all[5] == all[7];
    const size_t b = all[0] + all[2] + all[4] + all[6];
    if (b || !same) {
      ++failures;
      if (ring.rank() == 0)
        std::printf("  FAIL %s: wrong of %zu per rank %lu %lu %lu %lu; identical on all ranks: %s\n", what.c_str(), total,
                    all[0], all[2], all[4], all[6], same ? "yes" : "NO");
    } else if (ring.rank() == 0) {
      std::printf("  ok %s (exact, identical on all ranks)\n", what.c_str());
    }
    std::fflush(stdout);
  }
};

template <class T>
T* dalloc(size_t n) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T) + 64));
  return p;
}

// ---- (a) the hidden all-reduce ------------------------------------------------------------------------------

__global__ void k_add_res(const float* sum, const uint16_t* res, uint16_t* h, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float y = __bfloat162float(__float2bfloat16_rn(sum[i]));
  const float r = __uint_as_float((uint32_t)res[i] << 16);
  h[i] = __bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(r, y)));
}

// host: rmsnorm of one row exactly as attn's kernel (256 threads of 24, xor-trees, warps in order)
void host_rmsnorm(const float* h, const uint16_t* w, float eps, uint16_t* y) {
  float t[256];
  for (int u = 0; u < 256; ++u) {
    float s = 0.f;
    for (int j = 0; j < 24; ++j) {
      const float x = h[24 * u + j];
      const float sq = x * x;
      s = s + sq;
    }
    t[u] = s;
  }
  float tot = 0.f;
  for (int wp = 0; wp < 8; ++wp) {
    float v[32];
    for (int l = 0; l < 32; ++l) v[l] = t[32 * wp + l];
    for (int o = 16; o > 0; o >>= 1) {
      float nv[32];
      for (int l = 0; l < 32; ++l) nv[l] = v[l] + v[l ^ o];
      std::memcpy(v, nv, sizeof v);
    }
    tot = tot + v[0];
  }
  const float var = tot / (float)kDim;
  const float rstd = 1.f / std::sqrt(var + eps);
  for (int c = 0; c < kDim; ++c) {
    const float a = bfr(h[c] * rstd);
    y[c] = f2bf(bf2f(w[c]) * a);
  }
}

void test_allreduce(Ring& ring, cudaStream_t st, Checker& ck) {
  const DevChannel& ch = ring.dev();
  const int me = ring.rank();
  const int rows_list[] = {1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 16};
  const size_t N = (size_t)G::kMaxRows * kDim;
  float* part = dalloc<float>(N);
  float* ref32 = dalloc<float>(N);
  float* out32 = dalloc<float>(N);
  uint16_t* out16 = dalloc<uint16_t>(N);
  uint16_t* res = dalloc<uint16_t>(N);
  uint16_t* res2 = dalloc<uint16_t>(N);
  uint16_t* hres = dalloc<uint16_t>(N);
  uint16_t* xn = dalloc<uint16_t>(N);
  uint16_t* xn_ref = dalloc<uint16_t>(N);
  uint16_t* w = dalloc<uint16_t>(kDim);
  std::vector<float> hp(N), hsum(N), g32(N);
  std::vector<uint16_t> hres_in(N), hw(kDim), g16(N), gres(N), gxn(N), gxr(N), exp_res(N), exp_xn(N);
  uint32_t salt = 0xa11;
  for (int c = 0; c < kDim; ++c) hw[c] = f2bf(0.5f + (gen(9, 0x77, c) & 0xffff) / 65536.0f * 1.5f);
  CUDA_CHECK(cudaMemcpy(w, hw.data(), kDim * 2, cudaMemcpyHostToDevice));
  for (int R : rows_list) {
    ++salt;
    const size_t n = (size_t)R * kDim;
    for (size_t i = 0; i < n; ++i) {
      float s = test_value(0, i, salt);
      for (int k = 1; k < 4; ++k) s = s + test_value(k, i, salt);
      hsum[i] = s;
      hp[i] = test_value(me, i, salt);
      hres_in[i] = f2bf(test_value(7, i, salt ^ 0x5555));
    }
    // expected residual and normed rows
    for (int r = 0; r < R; ++r) {
      std::vector<float> h(kDim);
      for (int c = 0; c < kDim; ++c) {
        const size_t i = (size_t)r * kDim + c;
        h[c] = bfr(bf2f(hres_in[i]) + bfr(hsum[i]));
        exp_res[i] = f2bf(h[c]);
      }
      host_rmsnorm(h.data(), hw.data(), G::kNormEps, &exp_xn[(size_t)r * kDim]);
    }
    CUDA_CHECK(cudaMemcpy(part, hp.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(res, hres_in.data(), n * 2, cudaMemcpyHostToDevice));
    // the generic all-reduce and the GPU reference chain: + residual, attn's rmsnorm
    comm::allreduce_f32(ch, part, ref32, n, st);
    k_add_res<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(ref32, res, hres, (int)n);
    ::glm::rmsnorm(hres, kDim, w, G::kNormEps, kDim, xn_ref, kDim, R, st);
    CUDA_CHECK(cudaMemcpyAsync(g32.data(), ref32, n * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(gxr.data(), xn_ref, n * 2, cudaMemcpyDeviceToHost, st));
    sync_or_die(ring, st, 30, "allreduce reference");
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) bad += std::memcmp(&g32[i], &hsum[i], 4) != 0;
    size_t badr = 0;
    for (size_t i = 0; i < n; ++i) badr += gxr[i] != exp_xn[i];
    char nm[160];
    std::snprintf(nm, sizeof nm, "reference chain %2d rows: comm::allreduce_f32 vs host, rmsnorm vs host", R);
    ck.report(nm, bad + badr, 2 * n, fnv(gxr.data(), n * 2));
    for (int S : {8, 16, 32}) {
      for (int pdl = 0; pdl < 2; ++pdl) {
        CUDA_CHECK(cudaMemsetAsync(out32, 0xff, n * 4, st));
        CUDA_CHECK(cudaMemsetAsync(out16, 0xff, n * 2, st));
        G::allreduce_hidden(ch, part, R, out32, out16, st, pdl, S);
        CUDA_CHECK(cudaMemcpyAsync(g32.data(), out32, n * 4, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(g16.data(), out16, n * 2, cudaMemcpyDeviceToHost, st));
        sync_or_die(ring, st, 30, "allreduce_hidden");
        bad = 0;
        for (size_t i = 0; i < n; ++i) bad += std::memcmp(&g32[i], &hsum[i], 4) != 0 || g16[i] != f2bf(hsum[i]);
        std::snprintf(nm, sizeof nm, "allreduce_hidden   %2d rows, %2d slices%s", R, S, pdl ? ", pdl" : "");
        ck.report(nm, bad, n, fnv(g16.data(), n * 2, fnv(g32.data(), n * 4)));
        // fused: separate residual out, then in place (twice: run to run)
        CUDA_CHECK(cudaMemsetAsync(res2, 0xff, n * 2, st));
        CUDA_CHECK(cudaMemsetAsync(xn, 0xff, n * 2, st));
        G::allreduce_add_norm(ch, part, R, res, res2, w, G::kNormEps, xn, st, pdl, S);
        CUDA_CHECK(cudaMemcpyAsync(gres.data(), res2, n * 2, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(gxn.data(), xn, n * 2, cudaMemcpyDeviceToHost, st));
        sync_or_die(ring, st, 30, "allreduce_add_norm");
        bad = 0;
        for (size_t i = 0; i < n; ++i) bad += gres[i] != exp_res[i] || gxn[i] != exp_xn[i] || gxn[i] != gxr[i];
        const uint64_t h1 = fnv(gxn.data(), n * 2, fnv(gres.data(), n * 2));
        for (int rep = 0; rep < 2; ++rep) {
          CUDA_CHECK(cudaMemcpyAsync(res2, res, n * 2, cudaMemcpyDeviceToDevice, st));
          CUDA_CHECK(cudaMemsetAsync(xn, 0xff, n * 2, st));
          G::allreduce_add_norm(ch, part, R, res2, res2, w, G::kNormEps, xn, st, pdl, S);
          CUDA_CHECK(cudaMemcpyAsync(gres.data(), res2, n * 2, cudaMemcpyDeviceToHost, st));
          CUDA_CHECK(cudaMemcpyAsync(gxn.data(), xn, n * 2, cudaMemcpyDeviceToHost, st));
          sync_or_die(ring, st, 30, "allreduce_add_norm in place");
          bad += fnv(gxn.data(), n * 2, fnv(gres.data(), n * 2)) != h1;
        }
        std::snprintf(nm, sizeof nm, "allreduce_add_norm %2d rows, %2d slices%s (vs host, vs rmsnorm, in place x2)", R, S,
                      pdl ? ", pdl" : "");
        ck.report(nm, bad, n, h1);
      }
    }
    // the MoE layer's two partials (allreduce2_add_norm): y = bf16(bf16(sum) + bf16(sum2)), then as above
    {
      std::vector<float> hp2(n), hsum2(n);
      std::vector<uint16_t> exp2_res(n), exp2_xn(n);
      for (size_t i = 0; i < n; ++i) {
        float s2 = test_value(0, i, salt ^ 0x99);
        for (int k = 1; k < 4; ++k) s2 = s2 + test_value(k, i, salt ^ 0x99);
        hsum2[i] = s2;
        hp2[i] = test_value(me, i, salt ^ 0x99);
      }
      for (int r = 0; r < R; ++r) {
        std::vector<float> h(kDim);
        for (int c = 0; c < kDim; ++c) {
          const size_t i = (size_t)r * kDim + c;
          h[c] = bfr(bf2f(hres_in[i]) + bfr(bfr(hsum[i]) + bfr(hsum2[i])));
          exp2_res[i] = f2bf(h[c]);
        }
        host_rmsnorm(h.data(), hw.data(), G::kNormEps, &exp2_xn[(size_t)r * kDim]);
      }
      CUDA_CHECK(cudaMemcpy(ref32, hp2.data(), n * 4, cudaMemcpyHostToDevice));
      for (int pdl = 0; pdl < 2; ++pdl) {
        CUDA_CHECK(cudaMemsetAsync(res2, 0xff, n * 2, st));
        CUDA_CHECK(cudaMemsetAsync(xn, 0xff, n * 2, st));
        G::allreduce2_add_norm(ch, part, ref32, R, res, res2, w, G::kNormEps, xn, st, pdl);
        CUDA_CHECK(cudaMemcpyAsync(gres.data(), res2, n * 2, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(gxn.data(), xn, n * 2, cudaMemcpyDeviceToHost, st));
        sync_or_die(ring, st, 30, "allreduce2_add_norm");
        size_t bad2 = 0;
        for (size_t i = 0; i < n; ++i) bad2 += gres[i] != exp2_res[i] || gxn[i] != exp2_xn[i];
        char nm2[160];
        std::snprintf(nm2, sizeof nm2, "allreduce2_add_norm %2d rows%s (two partials, vs host)", R, pdl ? ", pdl" : "");
        ck.report(nm2, bad2, n, fnv(gxn.data(), n * 2, fnv(gres.data(), n * 2)));
      }
    }
  }
  for (void* p : {(void*)part, (void*)ref32, (void*)out32, (void*)out16, (void*)res, (void*)res2, (void*)hres, (void*)xn,
                  (void*)xn_ref, (void*)w})
    cudaFree(p);
}

// ---- (b, c, d) row all-gathers ------------------------------------------------------------------------------
// Segments of rb bytes a row; in place: rank r's rows inside out ([R][4 rb], at r rb); else in [R][rb].
// call(in, out) launches the collective. Expected: out[row][k] = rank k's row, word j = gen(k, salt + g, ...).
struct GSeg {
  size_t rb;
  bool inplace;
};
void check_gather(Ring& ring, cudaStream_t st, Checker& ck, const std::string& name, const std::vector<GSeg>& segs, int R,
                  const std::function<void(std::vector<uint8_t*>&, std::vector<uint8_t*>&)>& call) {
  const int me = ring.rank();
  static uint32_t salt = 0x6a7;
  ++salt;
  std::vector<uint8_t*> din(segs.size()), dout(segs.size());
  std::vector<std::vector<uint32_t>> hout(segs.size());
  for (size_t g = 0; g < segs.size(); ++g) {
    const size_t rb = segs[g].rb, wpr = rb / 4, ow = (size_t)R * 4 * wpr;
    dout[g] = dalloc<uint8_t>(ow * 4);
    std::vector<uint32_t> o(ow, 0xa5a5a5a5u);
    if (segs[g].inplace) {
      for (int r = 0; r < R; ++r)
        for (size_t j = 0; j < wpr; ++j) o[((size_t)r * 4 + me) * wpr + j] = gen(me, salt + g, (size_t)r * wpr + j);
      din[g] = dout[g] + me * rb;
    } else {
      std::vector<uint32_t> in((size_t)R * wpr);
      for (size_t i = 0; i < in.size(); ++i) in[i] = gen(me, salt + g, i);
      din[g] = dalloc<uint8_t>(in.size() * 4);
      CUDA_CHECK(cudaMemcpy(din[g], in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dout[g], o.data(), ow * 4, cudaMemcpyHostToDevice));
    hout[g].resize(ow);
  }
  call(din, dout);
  for (size_t g = 0; g < segs.size(); ++g)
    CUDA_CHECK(cudaMemcpyAsync(hout[g].data(), dout[g], hout[g].size() * 4, cudaMemcpyDeviceToHost, st));
  sync_or_die(ring, st, 60, name.c_str());
  size_t bad = 0, total = 0;
  uint64_t h = 1469598103934665603ull;
  for (size_t g = 0; g < segs.size(); ++g) {
    const size_t wpr = segs[g].rb / 4;
    for (int r = 0; r < R; ++r)
      for (int k = 0; k < 4; ++k)
        for (size_t j = 0; j < wpr; ++j, ++total)
          bad += hout[g][((size_t)r * 4 + k) * wpr + j] != gen(k, salt + g, (size_t)r * wpr + j);
    h = fnv(hout[g].data(), hout[g].size() * 4, h);
    if (!segs[g].inplace) cudaFree(din[g]);
    cudaFree(dout[g]);
  }
  ck.report(name, bad, total, h);
}

void test_gathers(Ring& ring, cudaStream_t st, Checker& ck) {
  const DevChannel& ch = ring.dev();
  char nm[160];
  for (int R : {1, 2, 6, 16, 300, 2048}) {
    for (int full = 1; full >= 0; --full) {
      std::vector<GSeg> segs = {{(size_t)(full ? G::kXRows : G::kXRowsShared) * 2, true}};
      if (full) segs.push_back({(size_t)G::kIdxHeadsRank * 4, true});
      std::snprintf(nm, sizeof nm, "allgather_xproj %s %4d rows", full ? "full  " : "shared", R);
      check_gather(ring, st, ck, nm, segs, R, [&](auto& in, auto& out) {
        G::allgather_xproj(ch, (uint16_t*)out[0], full ? (float*)out[1] : nullptr, R, full, st);
      });
    }
  }
  // the pipelined path with a small chunk (several posts even at decode sizes) and an explicit grid
  for (int R : {16, 2048}) {
    std::snprintf(nm, sizeof nm, "allgather_rows xproj %4d rows, 64 KiB chunks, 4 blocks", R);
    check_gather(ring, st, ck, nm, {{(size_t)G::kXRows * 2, true}, {(size_t)G::kIdxHeadsRank * 4, true}}, R,
                 [&](auto& in, auto& out) {
                   G::RowSeg s[2] = {{in[0], 4 * G::kXRows * 2, out[0], 4 * G::kXRows * 2, (size_t)G::kXRows * 2},
                                     {in[1], 128, out[1], 128, 32}};
                   G::allgather_rows(ch, s, 2, R, st, 4, 65536);
                 });
  }
  for (int R : {1, 6, 16, 2048}) {
    std::snprintf(nm, sizeof nm, "allgather_idxq %4d rows", R);
    check_gather(ring, st, ck, nm, {{G::kIdxQRow, true}}, R,
                 [&](auto& in, auto& out) { G::allgather_idxq(ch, (uint16_t*)out[0], R, st); });
  }
  for (int R : {1, 6, 16}) {
    std::snprintf(nm, sizeof nm, "allgather_cand %4d rows", R);
    check_gather(ring, st, ck, nm, {{(size_t)G::kTopK * 4, false}, {(size_t)G::kTopK * 4, false}}, R,
                 [&](auto& in, auto& out) {
                   G::allgather_cand(ch, (const int*)in[0], (const uint32_t*)in[1], (int*)out[0], (uint32_t*)out[1], R, st);
                 });
  }
  for (int R : {1, 6, 16}) {
    for (int inplace = 0; inplace < 2; ++inplace) {
      std::snprintf(nm, sizeof nm, "allgather_logits f32  %2d rows%s", R, inplace ? ", in place" : "");
      check_gather(ring, st, ck, nm, {{G::kLogitsF32Row, (bool)inplace}}, R, [&](auto& in, auto& out) {
        G::allgather_logits_f32(ch, (const float*)in[0], inplace ? G::kVocab : G::kVocabRank, (float*)out[0], R, st);
      });
      std::snprintf(nm, sizeof nm, "allgather_logits bf16 %2d rows%s", R, inplace ? ", in place" : "");
      check_gather(ring, st, ck, nm, {{G::kLogitsBf16Row, (bool)inplace}}, R, [&](auto& in, auto& out) {
        G::allgather_logits_bf16(ch, (const uint16_t*)in[0], inplace ? G::kVocab : G::kVocabRank, (uint16_t*)out[0], R,
                                 st);
      });
    }
  }
}

void test_topk(Ring& ring, cudaStream_t st, Checker& ck) {
  const DevChannel& ch = ring.dev();
  const int me = ring.rank();
  uint32_t salt = 0x70c;
  for (int K : {1, 2, 7, 64})
    for (int R : {1, 6, 16}) {
      ++salt;
      // rank k's candidates: values on a coarse grid (ties across ranks), distinct indices in its vocabulary quarter
      auto cand = [&](int k, int r, int j) {
        const uint32_t h = gen(k, salt, (size_t)r * K + j);
        return G::Cand{(float)((int)(h % 13) - 6) * 0.25f, k * G::kVocabRank + (int)(h >> 8) % G::kVocabRank};
      };
      std::vector<G::Cand> mine((size_t)R * K), want((size_t)R * K), got((size_t)R * K);
      for (int r = 0; r < R; ++r) {
        std::vector<G::Cand> all;
        for (int k = 0; k < 4; ++k) {
          std::vector<int> used;
          for (int j = 0; j < K; ++j) {
            G::Cand c = cand(k, r, j);
            while (std::find(used.begin(), used.end(), c.i) != used.end()) c.i = k * G::kVocabRank + (c.i + 1) % G::kVocabRank;
            used.push_back(c.i);
            all.push_back(c);
            if (k == me) mine[(size_t)r * K + j] = c;
          }
        }
        std::sort(all.begin(), all.end(), [](const G::Cand& a, const G::Cand& b) { return a.v > b.v || (a.v == b.v && a.i < b.i); });
        std::copy(all.begin(), all.begin() + K, want.begin() + (size_t)r * K);
      }
      G::Cand *din = dalloc<G::Cand>(mine.size()), *dout = dalloc<G::Cand>(mine.size());
      CUDA_CHECK(cudaMemcpy(din, mine.data(), mine.size() * sizeof(G::Cand), cudaMemcpyHostToDevice));
      G::topk_exchange(ch, din, dout, R, K, st);
      CUDA_CHECK(cudaMemcpyAsync(got.data(), dout, got.size() * sizeof(G::Cand), cudaMemcpyDeviceToHost, st));
      sync_or_die(ring, st, 30, "topk");
      size_t bad = 0;
      for (size_t i = 0; i < got.size(); ++i) bad += got[i].v != want[i].v || got[i].i != want[i].i;
      char nm[96];
      std::snprintf(nm, sizeof nm, "topk_exchange K %2d, %2d rows", K, R);
      ck.report(nm, bad, got.size(), fnv(got.data(), got.size() * sizeof(G::Cand)));
      cudaFree(din);
      cudaFree(dout);
    }
}

// A run of GLM's collectives in one graph, with PDL, as a step issues them; the outputs of the last layer against
// the same collectives run one by one without PDL.
void test_sequence(Ring& ring, cudaStream_t st, Checker& ck) {
  const DevChannel& ch = ring.dev();
  const int me = ring.rank();
  for (int R : {1, 6, 16}) {
    const size_t n = (size_t)R * kDim;
    float* part = dalloc<float>(n);
    uint16_t *res = dalloc<uint16_t>(n), *resA = dalloc<uint16_t>(n), *xn = dalloc<uint16_t>(n), *w = dalloc<uint16_t>(kDim);
    uint16_t* xg = dalloc<uint16_t>((size_t)R * 4 * G::kXRows);
    float* iw = dalloc<float>((size_t)R * 32);
    uint16_t* iq = dalloc<uint16_t>((size_t)R * 4096);
    std::vector<float> hp(n);
    std::vector<uint16_t> hr(n), hw(kDim), hx((size_t)R * 4 * G::kXRows), hq((size_t)R * 4096);
    for (size_t i = 0; i < n; ++i) hp[i] = test_value(me, i, 0x5e9), hr[i] = f2bf(test_value(5, i, 0x5ea));
    for (int c = 0; c < kDim; ++c) hw[c] = f2bf(1.f + (c % 7) * 0.125f);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = (uint16_t)gen(me, 0x5eb, i);
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = (uint16_t)gen(me, 0x5ec, i);
    CUDA_CHECK(cudaMemcpy(part, hp.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w, hw.data(), kDim * 2, cudaMemcpyHostToDevice));
    auto layer = [&](bool pdl, bool full) {
      G::allgather_xproj(ch, xg, iw, R, full, st, pdl);
      if (full) G::allgather_idxq(ch, iq, R, st, pdl);
      G::allreduce_add_norm(ch, part, R, res, res, w, G::kNormEps, xn, st, pdl);
      G::allreduce_add_norm(ch, part, R, res, res, w, G::kNormEps, xn, st, pdl);
    };
    uint64_t hsh[2];
    for (int pass = 0; pass < 2; ++pass) {
      CUDA_CHECK(cudaMemcpy(res, hr.data(), n * 2, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(xg, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(iq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(iw, 0, (size_t)R * 32 * 4));
      if (pass == 0) {
        for (int l = 0; l < 8; ++l) layer(false, l % 4 == 0);
      } else {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < 8; ++l) layer(true, l % 4 == 0);
        CUDA_CHECK(cudaStreamEndCapture(st, &g));
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        CUDA_CHECK(cudaGraphLaunch(ge, st));
        sync_or_die(ring, st, 30, "sequence graph");
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
      }
      sync_or_die(ring, st, 30, "sequence");
      std::vector<uint16_t> a(n), b(n), c(hx.size()), d(hq.size());
      CUDA_CHECK(cudaMemcpy(a.data(), res, n * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(b.data(), xn, n * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(c.data(), xg, c.size() * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(d.data(), iq, d.size() * 2, cudaMemcpyDeviceToHost));
      hsh[pass] = fnv(d.data(), d.size() * 2, fnv(c.data(), c.size() * 2, fnv(b.data(), n * 2, fnv(a.data(), n * 2))));
    }
    char nm[128];
    std::snprintf(nm, sizeof nm, "sequence of 8 layers (%d rows): graph with PDL = one by one", R);
    ck.report(nm, hsh[0] != hsh[1], 1, hsh[1]);
    for (void* p : {(void*)part, (void*)res, (void*)resA, (void*)xn, (void*)w, (void*)xg, (void*)iw, (void*)iq}) cudaFree(p);
  }
}

// ---- timing --------------------------------------------------------------------------------------------------

int g_rounds = 3;
FILE* g_out = nullptr;
std::string g_tag;

// µs per collective: the best of g_rounds trials of `reps` replays of a graph of `per` back-to-back launches
template <typename F>
double time_graph(Ring& ring, cudaStream_t st, int per, int reps, F launch) {
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < per; i++) launch();
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 2; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
  sync_or_die(ring, st, 60, "timing warmup");
  ring.barrier();
  std::vector<double> t;
  for (int trial = 0; trial < g_rounds; trial++) {
    const double t0 = now_s();
    for (int i = 0; i < reps; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
    sync_or_die(ring, st, 120, "timing");
    t.push_back((now_s() - t0) / ((double)reps * per));
    ring.barrier();
  }
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  std::sort(t.begin(), t.end());
  return t[t.size() / 2] * 1e6;  // the median
}

void record(Ring& ring, const char* what, int rows, size_t bytes, double us) {
  if (ring.rank() != 0) return;
  std::printf("  %-34s %5d rows %10zu B/rank %9.2f us %8.2f GB/s\n", what, rows, bytes, us, bytes / us / 1e3);
  std::fflush(stdout);
  if (g_out) {
    std::fprintf(g_out, "%s\t%s\t%d\t%zu\t%.3f\n", g_tag.c_str(), what, rows, bytes, us);
    std::fflush(g_out);
  }
}

void bench(Ring& ring, cudaStream_t st) {
  const DevChannel& ch = ring.dev();
  const size_t N = (size_t)G::kMaxRows * kDim;
  float* part = dalloc<float>(N);
  float* f32 = dalloc<float>(N);
  uint16_t *bf = dalloc<uint16_t>(N), *res = dalloc<uint16_t>(N), *xn = dalloc<uint16_t>(N), *w = dalloc<uint16_t>(kDim);
  uint16_t* xg = dalloc<uint16_t>((size_t)G::kMaxRows * 4 * G::kXRows);
  float* iw = dalloc<float>((size_t)G::kMaxRows * 32);
  uint16_t* iq = dalloc<uint16_t>((size_t)G::kMaxRows * 4096);
  int* cp = dalloc<int>((size_t)G::kMaxRows * G::kTopK);
  uint32_t* ckey = dalloc<uint32_t>((size_t)G::kMaxRows * G::kTopK);
  int* cpa = dalloc<int>((size_t)G::kMaxRows * 4 * G::kTopK);
  uint32_t* ckeya = dalloc<uint32_t>((size_t)G::kMaxRows * 4 * G::kTopK);
  float* lg = dalloc<float>((size_t)G::kMaxRows * G::kVocab);
  G::Cand *tk = dalloc<G::Cand>(G::kMaxRows * 64), *tko = dalloc<G::Cand>(G::kMaxRows * 64);
  comm::ArgMax *am = dalloc<comm::ArgMax>(G::kMaxRows), *amo = dalloc<comm::ArgMax>(G::kMaxRows);
  // contents do not matter beyond being NaN-free where they are summed or compared
  CUDA_CHECK(cudaMemset(part, 0, N * 4));
  CUDA_CHECK(cudaMemset(res, 0, N * 2));
  CUDA_CHECK(cudaMemset(w, 0, kDim * 2));
  CUDA_CHECK(cudaMemset(tk, 0, G::kMaxRows * 64 * sizeof(G::Cand)));
  CUDA_CHECK(cudaMemset(am, 0, G::kMaxRows * sizeof(comm::ArgMax)));
  CUDA_CHECK(cudaMemset(lg, 0, (size_t)G::kMaxRows * G::kVocab * 4));
  const int per = 50, reps = 20;
  for (int R : {1, 2, 4, 6, 8, 12, 16}) {
    const size_t hb = (size_t)R * G::kHiddenF32Row;
    record(ring, "allreduce_f32 (comm, generic)", R, hb,
           time_graph(ring, st, per, reps, [&] { comm::allreduce_f32(ch, part, f32, (size_t)R * kDim, st); }));
    record(ring, "allreduce_bf16 (comm, bf16 partials)", R, hb / 2,
           time_graph(ring, st, per, reps, [&] {
             comm::allreduce_bf16(ch, (const __nv_bfloat16*)bf, (__nv_bfloat16*)bf, (size_t)R * kDim, st);
           }));
    for (int S : {8, 16, 32}) {
      char nm[64];
      std::snprintf(nm, sizeof nm, "allreduce_hidden S%d", S);
      record(ring, nm, R, hb, time_graph(ring, st, per, reps, [&] { G::allreduce_hidden(ch, part, R, f32, nullptr, st, true, S); }));
      std::snprintf(nm, sizeof nm, "allreduce_add_norm S%d", S);
      record(ring, nm, R, hb, time_graph(ring, st, per, reps, [&] {
               G::allreduce_add_norm(ch, part, R, res, res, w, G::kNormEps, xn, st, true, S);
             }));
    }
    record(ring, "allgather_xproj full", R, R * G::kXprojRow,
           time_graph(ring, st, per, reps, [&] { G::allgather_xproj(ch, xg, iw, R, true, st, true); }));
    record(ring, "allgather_xproj shared", R, R * G::kXprojRowShared,
           time_graph(ring, st, per, reps, [&] { G::allgather_xproj(ch, xg, iw, R, false, st, true); }));
    record(ring, "allgather_idxq", R, R * G::kIdxQRow,
           time_graph(ring, st, per, reps, [&] { G::allgather_idxq(ch, iq, R, st, true); }));
    record(ring, "allgather_cand", R, R * G::kCandRow,
           time_graph(ring, st, per, reps, [&] { G::allgather_cand(ch, cp, ckey, cpa, ckeya, R, st, true); }));
    record(ring, "allgather_logits_f32", R, R * G::kLogitsF32Row, time_graph(ring, st, 10, 10, [&] {
             G::allgather_logits_f32(ch, lg + (size_t)ch.rank * G::kVocabRank, G::kVocab, lg, R, st, true);
           }));
    record(ring, "allgather_logits_bf16", R, R * G::kLogitsBf16Row, time_graph(ring, st, 10, 10, [&] {
             G::allgather_logits_bf16(ch, (uint16_t*)lg + (size_t)ch.rank * G::kVocabRank, G::kVocab, (uint16_t*)lg, R, st,
                                      true);
           }));
    {
      const G::RowSeg s{lg + (size_t)ch.rank * G::kVocabRank, G::kVocab * 4, lg, G::kVocab * 4, G::kLogitsF32Row};
      record(ring, "allgather_logits_f32 256K chunks", R, R * G::kLogitsF32Row, time_graph(ring, st, 10, 10, [&] {
               G::allgather_rows(ch, &s, 1, R, st, 16, 262144, true);
             }));
    }
    const int ra = (R + 1) & ~1;  // comm::argmax_rows sends whole 16-byte units: an even row count
    record(ring, "argmax_rows (comm)", ra, ra * 8,
           time_graph(ring, st, per, reps, [&] { comm::argmax_rows(ch, am, amo, ra, st); }));
    for (int K : {1, 64}) {
      char nm[48];
      std::snprintf(nm, sizeof nm, "topk_exchange K%d", K);
      record(ring, nm, R, (size_t)R * ((K + 1) & ~1) * 8,
             time_graph(ring, st, per, reps, [&] { G::topk_exchange(ch, tk, tko, R, K, st, true); }));
    }
  }
  for (void* p : {(void*)part, (void*)f32, (void*)bf, (void*)res, (void*)xn, (void*)w, (void*)xg, (void*)iw, (void*)iq,
                  (void*)cp, (void*)ckey, (void*)cpa, (void*)ckeya, (void*)lg, (void*)tk, (void*)tko, (void*)am, (void*)amo})
    cudaFree(p);
}

// A decode step's collectives in one graph, nothing between them: per layer the x-projection's all-gather and
// the two fused all-reduces; on the 21 full-indexer layers also the indexer queries' all-gather, and with the
// position-split indexer its candidates'; then the head's exchange. µs a step.
void bench_step(Ring& ring, cudaStream_t st) {
  const DevChannel& ch = ring.dev();
  const size_t N = (size_t)G::kMaxRows * kDim;
  float* part = dalloc<float>(N);
  uint16_t *res = dalloc<uint16_t>(N), *xn = dalloc<uint16_t>(N), *w = dalloc<uint16_t>(kDim);
  uint16_t* xg = dalloc<uint16_t>((size_t)G::kMaxRows * 4 * G::kXRows);
  float* iw = dalloc<float>((size_t)G::kMaxRows * 32);
  uint16_t* iq = dalloc<uint16_t>((size_t)G::kMaxRows * 4096);
  int* cp = dalloc<int>((size_t)G::kMaxRows * G::kTopK);
  uint32_t* ckey = dalloc<uint32_t>((size_t)G::kMaxRows * G::kTopK);
  int* cpa = dalloc<int>((size_t)G::kMaxRows * 4 * G::kTopK);
  uint32_t* ckeya = dalloc<uint32_t>((size_t)G::kMaxRows * 4 * G::kTopK);
  G::Cand *tk = dalloc<G::Cand>(G::kMaxRows * 64), *tko = dalloc<G::Cand>(G::kMaxRows * 64);
  CUDA_CHECK(cudaMemset(part, 0, N * 4));
  CUDA_CHECK(cudaMemset(res, 0, N * 2));
  CUDA_CHECK(cudaMemset(w, 0, kDim * 2));
  CUDA_CHECK(cudaMemset(tk, 0, G::kMaxRows * 64 * sizeof(G::Cand)));
  for (int R : {1, 2, 4, 6, 8, 16})
    for (int split = 0; split < 2; ++split) {
      int count = 0;
      auto step = [&] {
        count = 0;
        for (int l = 0; l < ::glm::kLayers; ++l) {
          const bool full = ::glm::full_indexer(l);
          G::allgather_xproj(ch, xg, iw, R, full, st, true), ++count;
          if (full) G::allgather_idxq(ch, iq, R, st, true), ++count;
          if (full && split) G::allgather_cand(ch, cp, ckey, cpa, ckeya, R, st, true), ++count;
          G::allreduce_add_norm(ch, part, R, res, res, w, G::kNormEps, xn, st, true), ++count;
          G::allreduce_add_norm(ch, part, R, res, res, w, G::kNormEps, xn, st, true), ++count;
        }
        G::topk_exchange(ch, tk, tko, R, 64, st, true), ++count;
      };
      const double us = time_graph(ring, st, 1, 10, step);  // per = 1 step
      char nm[96];
      std::snprintf(nm, sizeof nm, "step %s (%d collectives)", split ? "position-split indexer" : "replicated indexer", count);
      record(ring, nm, R, 0, us);
      if (ring.rank() == 0) std::printf("      = %.2f us a collective\n", us / count);
    }
  for (void* p : {(void*)part, (void*)res, (void*)xn, (void*)w, (void*)xg, (void*)iw, (void*)iq, (void*)cp, (void*)ckey,
                  (void*)cpa, (void*)ckeya, (void*)tk, (void*)tko})
    cudaFree(p);
}

// Prefill chunks: the hidden all-reduce as a reduce-scatter by rows (f32 partials) and an all-gather of the
// reduced rows (bf16), or the bulk all-reduce; the x-projection's all-gather; for N rows a chunk.
void bench_prefill(Ring& ring, cudaStream_t st) {
  const DevChannel& ch = ring.dev();
  const int NMAX = 2048;
  float* in = dalloc<float>((size_t)NMAX * kDim);
  float* q = dalloc<float>((size_t)NMAX / 4 * kDim);
  uint16_t* h = dalloc<uint16_t>((size_t)NMAX * kDim);
  uint16_t* xg = dalloc<uint16_t>((size_t)NMAX * 4 * G::kXRows);
  float* iw = dalloc<float>((size_t)NMAX * 32);
  CUDA_CHECK(cudaMemset(in, 0, (size_t)NMAX * kDim * 4));
  CUDA_CHECK(cudaMemset(h, 0, (size_t)NMAX * kDim * 2));
  for (int N : {256, 1024, 2048}) {
    const int qr = N / 4;
    record(ring, "reduce_scatter_rows_f32 (16 blocks)", N, (size_t)N * kDim * 4, time_graph(ring, st, 2, 5, [&] {
             comm::reduce_scatter_rows_f32(ch, in, kDim, q, kDim, qr, kDim, st, 16);
           }));
    record(ring, "allgather bf16 rows (quarter a rank)", N, (size_t)qr * kDim * 2, time_graph(ring, st, 2, 5, [&] {
             comm::allgather(ch, h + (size_t)ch.rank * qr * kDim, h, (size_t)qr * kDim * 2, st);
           }));
    record(ring, "allreduce_bulk_f32 (16 blocks)", N, (size_t)N * kDim * 4, time_graph(ring, st, 2, 5, [&] {
             comm::allreduce_bulk_f32(ch, in, in, (size_t)N * kDim, st, 16);
           }));
    record(ring, "allgather_xproj full", N, (size_t)N * G::kXprojRow,
           time_graph(ring, st, 2, 5, [&] { G::allgather_xproj(ch, xg, iw, N, true, st, true); }));
  }
  cudaFree(in);
  cudaFree(q);
  cudaFree(h);
  cudaFree(xg);
  cudaFree(iw);
}

}  // namespace

int main(int argc, char** argv) {
  comm::Options o;
  o.bootstrap_timeout_s = 120;
  o.cap = 4 << 20;  // the DeepSeek engine's: a prefill chunk's pieces
  std::string out;
  std::vector<std::string> tests;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--rank" && i + 1 < argc) o.rank = atoi(argv[++i]);
    else if (a == "--fleet" && i + 1 < argc) o.fleet = argv[++i];
    else if (a == "--cap" && i + 1 < argc) o.cap = atol(argv[++i]);
    else if (a == "--rounds" && i + 1 < argc) g_rounds = atoi(argv[++i]);
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--tag" && i + 1 < argc) g_tag = argv[++i];
    else if (a == "--proxy-cpu" && i + 1 < argc) o.proxy_cpu = atoi(argv[++i]);
    else tests.push_back(a);
  }
  if (tests.empty()) tests = {"test", "bench", "step", "prefill"};
  Ring ring(o);
  if (ring.rank() == 0 && !out.empty() && !(g_out = std::fopen(out.c_str(), "a"))) die("cannot append to %s", out.c_str());
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Checker ck{ring};
  for (auto& t : tests) {
    if (ring.rank() == 0) std::printf("== %s\n", t.c_str());
    std::fflush(stdout);
    if (t == "test") {
      test_allreduce(ring, st, ck);
      test_gathers(ring, st, ck);
      test_topk(ring, st, ck);
      test_sequence(ring, st, ck);
    } else if (t == "bench") bench(ring, st);
    else if (t == "step") bench_step(ring, st);
    else if (t == "prefill") bench_prefill(ring, st);
    else die("unknown test %s", t.c_str());
    std::fflush(stdout);
  }
  const comm::ProxyStats ps = ring.stats();
  std::printf("rank %d: %s; proxy posted %lu, relayed %lu + %lu, queue-full retries %lu\n", ring.rank(),
              ck.failures ? "FAILED" : "passed", ps.posted, ps.relayed[0], ps.relayed[1], ps.post_retries);
  if (g_out) std::fclose(g_out);
  return ck.failures ? 1 : 0;
}
