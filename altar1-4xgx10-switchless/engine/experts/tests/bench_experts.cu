// Benchmarks of the routed-expert kernels on one GB10 (GPU: only through ring.sh).
//   bench_experts --bw                                   the window's streaming read
//   bench_experts --decode i4|b16|q4 [--rows 1,6,16] [--sweep | --ups 0,6,6,0] [--out f.tsv]
//                 (a layer of the format; --ups: those up variants in that order, e.g. the default against the
//                 prefill-order wk1, A B B A)
//   bench_experts --prefill i4|b16|q4 [--chunks 1024,4096,8192] [--down 0,1] [--yrkb 0] [--cq 4] [--up 0,1] [--diag]
//                 [--out f.tsv]   (down / up: PrefillCfg variants; yrkb: chained-down L2 budgets, KB; cq: chained-down
//                 passes in flight; diag: the gate/up kernel and the chained down with parts switched off)
//   bench_experts --shfused i4|b16 [--rows 1,6,16]   the BF16 shared expert fused into the decode kernels: bits against
//                 the GEMVs (every variant) and time A/B/B/A against GEMVs + experts (bench_shfused)
//   bench_experts --shared [--rows 1,6,16]   altar-1's BF16 shared expert through attn's BF16 GEMV (gate/up
//                 interleaved, SwiGLU epilogue; down to f32 partials), every split-K
// Decode: a pool of >= 2.2 GB of experts in the format (random bits, valid scales) is cycled so L2 holds no
// step's weights; each step routes its rows (logits that pick a chosen set) and runs the expert step; a CUDA
// graph of whole steps, the median of 7 replays. Bytes = the union's routed blocks (values, scales, zero points).
// No fused shared expert (altar-1's is BF16 and runs outside: --shared).
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "experts/formats.cuh"
#include "experts/tests/testdata.h"

using namespace gx;

static FILE* g_out = nullptr;
static std::vector<int> g_ups;  // --ups 0,6,6,0: decode up variants timed in this order
static void result(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  printf("%s\n", buf);
  fflush(stdout);
  if (g_out) fprintf(g_out, "%s\n", buf), fflush(g_out);
}

template <class F>
static double time_graph(int steps, F launch, int reps = 7) {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t gr;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int s = 0; s < steps; ++s) launch(s, st);
  CUDA_CHECK(cudaStreamEndCapture(st, &gr));
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
  CUDA_CHECK(cudaGraphLaunch(ge, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<double> v;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaEventRecord(e0, st));
    CUDA_CHECK(cudaGraphLaunch(ge, st));
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    v.push_back(ms * 1e3 / steps);
  }
  CUDA_CHECK(cudaGetLastError());
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(gr);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaStreamDestroy(st);
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// ---- data on the GPU ----------------------------------------------------------------------------------------
// `blocks` blocks of format fmt: random values, valid scales (testdata.h's generators)
__global__ void k_fill_mat(uint8_t* p, size_t blocks, uint32_t fmt, uint64_t seed) {
  const int bs = fmt == kNVFP4 ? kQ4Blk : fmt == kINT4 ? kI4Blk : kB16Blk;
  const size_t n = blocks * bs / 4;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint64_t h = mix64(seed ^ (i * 0x9E3779B97F4A7C15ull));
    const int o = (int)((i * 4) % bs);
    uint32_t v = (uint32_t)h;
    if (fmt == kBF16) v = rnd_b16(h) | (uint32_t)rnd_b16(h >> 21) << 16;
    else if (fmt == kINT4 && o >= 2048 && o < 2304) v = rnd_i4_scale(h) | (uint32_t)rnd_i4_scale(h >> 24) << 16;
    else if (fmt == kNVFP4 && o >= 2048) {
      v = 0;
      for (int b = 0; b < 4; ++b) v |= (uint32_t)rnd_scale(h >> (8 * b)) << (8 * b);
    }
    reinterpret_cast<uint32_t*>(p)[i] = v;
  }
}
__global__ void k_fill_b16(uint16_t* p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = rnd_b16(mix64(seed ^ (i * 0x9E3779B97F4A7C15ull)));
}
__global__ void k_read(const int4* __restrict__ p, size_t n, int* out) {
  int4 acc = make_int4(0, 0, 0, 0);
  size_t stride = (size_t)gridDim.x * blockDim.x, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    int4 a = __ldcs(p + i), b = __ldcs(p + i + stride), c = __ldcs(p + i + 2 * stride), d = __ldcs(p + i + 3 * stride);
    acc.x ^= a.x ^ b.x ^ c.x ^ d.x; acc.y ^= a.y ^ b.y ^ c.y ^ d.y; acc.z ^= a.z ^ b.z ^ c.z ^ d.z; acc.w ^= a.w ^ b.w ^ c.w ^ d.w;
  }
  for (; i < n; i += stride) { int4 v = p[i]; acc.x ^= v.x; }
  if (acc.x == 0x12345678 && acc.y == 1) out[0] = acc.z + acc.w;
}
static double g_bw = 0;
static double bench_bw() {
  const size_t bytes = (size_t)2 << 30;
  int4* p;
  int* o;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  CUDA_CHECK(cudaMalloc(&o, 4));
  CUDA_CHECK(cudaMemset(p, 1, bytes));
  double best = 0;
  for (int grid : {96, 192, 384, 768})
    for (int th : {256, 512}) {
      const double us = time_graph(4, [&](int, cudaStream_t st) { k_read<<<grid, th, 0, st>>>(p, bytes / 16, o); }, 5);
      best = std::max(best, bytes / us / 1e3);
    }
  cudaFree(p), cudaFree(o);
  result("BW\tstreaming read\t%.1f GB/s", best);
  g_bw = best;
  return best;
}

// ---- the decode bench ---------------------------------------------------------------------------------------------
struct Pool {
  uint32_t fmt = kINT4;
  int NP = 0;
  std::vector<uint64_t> off;  // [NP * 3]
  size_t per = 0;             // a stored expert's routed bytes
  uint8_t* d = nullptr;
  size_t total = 0;
};
static Pool make_pool(uint32_t fmt, double target) {
  Pool P;
  P.fmt = fmt;
  for (int j = 0; j < 3; ++j) P.per += mat_bytes(fmt, j < 2 ? kI : kH, j < 2 ? kH : kI);
  P.NP = (int)(target / P.per) + 1;
  for (int e = 0; e < P.NP; ++e)
    for (int j = 0; j < 3; ++j) {
      P.off.push_back(P.total);
      P.total += mat_bytes(fmt, j < 2 ? kI : kH, j < 2 ? kH : kI);
    }
  CUDA_CHECK(cudaMalloc(&P.d, P.total));
  k_fill_mat<<<384, 256>>>(P.d, P.total / blk_bytes(fmt), fmt, 12345 + fmt);
  CUDA_CHECK(cudaDeviceSynchronize());
  return P;
}
static MatDesc pool_desc(const Pool& P, int x, int j) {
  return make_desc(P.off[x * 3 + j], P.fmt == kNVFP4 ? (j < 2 ? kS2Up : kS2Down) : 1.f, P.fmt);
}

static std::vector<int> parse_rows(const std::string& s, std::vector<std::string>& names) {
  std::vector<int> r;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    const std::string t = s.substr(p, q - p);
    names.push_back(t);
    r.push_back(atoi(t.c_str()));
    p = q + 1;
  }
  return r;
}

static void bench_decode(const std::string& fname, const std::string& rows_s, bool sweep) {
  const uint32_t fmt = fmt_of(fname);
  Pool P = make_pool(fmt, 2.2e9);
  printf("== decode %s: %d experts in the pool (%.2f GB), %.3f MB an expert\n", fname.c_str(), P.NP, P.total / 1e9, P.per / 1e6);
  std::vector<std::string> names;
  const std::vector<int> rows = parse_rows(rows_s, names);
  constexpr int NS = 64;  // steps a graph
  MatDesc* ddesc;
  float *dl, *db, *dyr, *dys;
  int* di;
  float* dw;
  __nv_bfloat16* dx;
  void* ws;
  CUDA_CHECK(cudaMalloc(&ddesc, (size_t)NS * kE * 3 * sizeof(MatDesc)));
  CUDA_CHECK(cudaMalloc(&dl, (size_t)NS * kMaxRows * kE * 4));
  CUDA_CHECK(cudaMalloc(&db, kE * 4));
  CUDA_CHECK(cudaMemset(db, 0, kE * 4));
  CUDA_CHECK(cudaMalloc(&dyr, (size_t)kMaxRows * kH * 4));
  CUDA_CHECK(cudaMalloc(&dys, (size_t)kMaxRows * kH * 4));
  CUDA_CHECK(cudaMalloc(&di, (size_t)NS * kMaxRows * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dw, (size_t)NS * kMaxRows * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dx, (size_t)kMaxRows * kH * 2));
  CUDA_CHECK(cudaMalloc(&ws, decode_ws_bytes()));
  CUDA_CHECK(cudaMemset(ws, 0, decode_ws_bytes()));
  std::vector<__nv_bfloat16> hx;
  rnd_x(hx, (size_t)kMaxRows * kH, 5);
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  // per-step descriptors: id e of step s -> stored expert (s * kE + e) % NP, cycling the pool
  {
    std::vector<MatDesc> d((size_t)NS * kE * 3);
    for (int s = 0; s < NS; ++s)
      for (int e = 0; e < kE; ++e)
        for (int j = 0; j < 3; ++j) d[((size_t)s * kE + e) * 3 + j] = pool_desc(P, (int)(((size_t)s * kE + e) % P.NP), j);
    CUDA_CHECK(cudaMemcpy(ddesc, d.data(), d.size() * sizeof(MatDesc), cudaMemcpyHostToDevice));
  }
  for (size_t ri = 0; ri < rows.size(); ++ri) {
    const int M = rows[ri];
    // each step's picks (8 distinct uniformly random experts a row); logits that make route() pick exactly them
    std::vector<float> lg((size_t)NS * kMaxRows * kE, -8.f);
    double bytes = 0, uavg = 0;
    for (int s = 0; s < NS; ++s) {
      std::set<int> uni;
      for (int m = 0; m < M; ++m) {
        std::set<int> row;
        uint64_t h = mix64((uint64_t)s * 1000 + m * 17 + M);
        while ((int)row.size() < kTopK) row.insert((int)((h = mix64(h)) % kE));
        int r = 0;
        for (int e : row) lg[((size_t)s * kMaxRows + m) * kE + e] = 8.f + 0.01f * r++, uni.insert(e);
      }
      bytes += (double)P.per * uni.size();
      uavg += uni.size();
    }
    bytes /= NS;
    uavg /= NS;
    CUDA_CHECK(cudaMemcpy(dl, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
    auto layer = [&](int s) {
      LayerW L;
      L.pool = P.d, L.desc = ddesc + (size_t)s * kE * 3, L.fmt = fmt;
      return L;
    };
    auto rt = [&](int s, cudaStream_t st) {
      route(dl + (size_t)s * kMaxRows * kE, db, M, di + s * kMaxRows * kTopK, dw + s * kMaxRows * kTopK, st);
    };
    // every step's picks once, for the timings without the router
    for (int s = 0; s < NS; ++s) rt(s, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto line = [&](const char* what, const DecodeCfg& c, double t, double tr) {
      result("DECODE\t%s\trows %s\tU %.1f\t%s\tup %s%s\tdown %s%s\tMB %.2f\tus %.1f\t(route %.1f)\tGB/s %.1f\t%% of 235 %.1f\t%% of window %.1f\t73 layers ms %.2f",
             fname.c_str(), names[ri].c_str(), uavg, what, decode_up_name(c.up), c.dyn ? " dyn" : "", decode_down_name(c.down),
             c.pdl ? "" : " nopdl", bytes / 1e6, t, tr, bytes / t / 1e3, 100 * bytes / t / 1e3 / 235.0, 100 * bytes / t / 1e3 / g_bw,
             73 * t / 1e3);
    };
    auto time_cfg = [&](const DecodeCfg& c, bool with_route) {
      return time_graph(NS, [&](int s, cudaStream_t st) {
        if (with_route) rt(s, st);
        moe_decode(layer(s), dx, di + s * kMaxRows * kTopK, dw + s * kMaxRows * kTopK, M, ws, dyr, dys, st, c);
      });
    };
    const double t_route = time_graph(NS, rt);
    DecodeCfg best;
    double best_t = 1e30;
    if (sweep) {
      for (int u = 0; u < decode_up_variants(); ++u) {
        DecodeCfg c;
        c.up = u;
        const double t = time_cfg(c, true);
        line("step", c, t, t_route);
        if (t < best_t) best_t = t, best = c;
      }
      for (int d = 1; d < decode_down_variants(); ++d) {
        DecodeCfg c = best;
        c.down = d;
        const double t = time_cfg(c, true);
        line("step", c, t, t_route);
        if (t < best_t) best_t = t, best = c;
      }
    } else if (!g_ups.empty()) {  // --ups: these up variants in this order (A B B A), each with its parts
      for (int u : g_ups) {
        DecodeCfg c;
        c.up = u;
        line("step", c, time_cfg(c, true), t_route);
        line("no route", c, time_cfg(c, false), t_route);
        c.parts = 1;
        line("up only", c, time_cfg(c, false), t_route);
      }
      continue;
    } else {
      best_t = time_cfg(best, true);
      line("step", best, best_t, t_route);
    }
    // the best configuration's parts, and without the queue / without PDL
    {
      DecodeCfg c = best;
      line("no route", c, time_cfg(c, false), t_route);
      c.parts = 1;
      line("up only", c, time_cfg(c, false), t_route);
      c.parts = 2;
      line("down only", c, time_cfg(c, false), t_route);
      c = best;
      c.dyn = !c.dyn;
      line("step", c, time_cfg(c, true), t_route);
      c = best;
      c.pdl = false;
      line("step", c, time_cfg(c, true), t_route);
    }
  }
  cudaFree(ddesc), cudaFree(dl), cudaFree(db), cudaFree(dyr), cudaFree(dys), cudaFree(di), cudaFree(dw), cudaFree(dx);
  cudaFree(ws), cudaFree(P.d);
}

// ---- the BF16 shared expert fused into the decode kernels (SharedW::bgu, bd) against attn's two GEMVs ----------------
// A layer of the format (the decode bench's pool, picks and descriptors) plus 8 copies of a BF16 shared expert in
// kBF16T (gate/up interleaved [1024][6144], down [6144][512]), cycled with the steps. (1) bits: for every up and
// down variant, the fused step's ys and shared h against gemv(kSwiGLU, split-K 4) + gemv(kF32, split-K 1) on the
// same bytes, and its yr against the unfused step's; (2) time: A = route + the two GEMVs + the unfused experts
// (the runner's step until now), B = route + the fused experts, in the order A B B A; also the experts alone.
static int g_fail = 0;
static void bench_shfused(const std::string& fname, const std::string& rows_s) {
  const uint32_t fmt = fmt_of(fname);
  Pool P = make_pool(fmt, 2.2e9);
  constexpr int NS = 64, NC = 8;
  const size_t gu = (size_t)2 * kI * kH, dn = (size_t)kH * kI;  // elements
  uint16_t* shw;
  CUDA_CHECK(cudaMalloc(&shw, NC * (gu + dn) * 2));
  k_fill_b16<<<384, 256>>>(shw, NC * (gu + dn), 99);
  MatDesc* ddesc;
  float *dl, *db, *yr, *ys, *yr0, *ys0;
  int* di;
  float* dw;
  __nv_bfloat16 *dx, *hm;
  void* ws;
  CUDA_CHECK(cudaMalloc(&ddesc, (size_t)NS * kE * 3 * sizeof(MatDesc)));
  CUDA_CHECK(cudaMalloc(&dl, (size_t)NS * kMaxRows * kE * 4));
  CUDA_CHECK(cudaMalloc(&db, kE * 4));
  CUDA_CHECK(cudaMemset(db, 0, kE * 4));
  for (float** p : {&yr, &ys, &yr0, &ys0}) CUDA_CHECK(cudaMalloc(p, (size_t)kMaxRows * kH * 4));
  CUDA_CHECK(cudaMalloc(&di, (size_t)NS * kMaxRows * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dw, (size_t)NS * kMaxRows * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dx, (size_t)kMaxRows * kH * 2));
  CUDA_CHECK(cudaMalloc(&hm, (size_t)kMaxRows * kI * 2));
  CUDA_CHECK(cudaMalloc(&ws, decode_ws_bytes()));
  CUDA_CHECK(cudaMemset(ws, 0, decode_ws_bytes()));
  {
    std::vector<__nv_bfloat16> hx;
    rnd_x(hx, (size_t)kMaxRows * kH, 5);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    std::vector<MatDesc> d((size_t)NS * kE * 3);
    for (int s = 0; s < NS; ++s)
      for (int e = 0; e < kE; ++e)
        for (int j = 0; j < 3; ++j) d[((size_t)s * kE + e) * 3 + j] = pool_desc(P, (int)(((size_t)s * kE + e) % P.NP), j);
    CUDA_CHECK(cudaMemcpy(ddesc, d.data(), d.size() * sizeof(MatDesc), cudaMemcpyHostToDevice));
  }
  printf("== shfused %s: %d experts in the pool (%.2f GB), %d BF16 shared experts of %.2f MB\n", fname.c_str(), P.NP,
         P.total / 1e9, NC, (gu + dn) * 2 / 1e6);
  std::vector<std::string> names;
  const std::vector<int> rows = parse_rows(rows_s, names);
  for (size_t ri = 0; ri < rows.size(); ++ri) {
    const int M = rows[ri];
    std::vector<float> lg((size_t)NS * kMaxRows * kE, -8.f);
    double rbytes = 0;
    for (int s = 0; s < NS; ++s) {
      std::set<int> uni;
      for (int m = 0; m < M; ++m) {
        std::set<int> row;
        uint64_t h = mix64((uint64_t)s * 1000 + m * 17 + M);
        while ((int)row.size() < kTopK) row.insert((int)((h = mix64(h)) % kE));
        int r = 0;
        for (int e : row) lg[((size_t)s * kMaxRows + m) * kE + e] = 8.f + 0.01f * r++, uni.insert(e);
      }
      rbytes += (double)P.per * uni.size();
    }
    rbytes /= NS;
    const double bytes = rbytes + (gu + dn) * 2.0;
    CUDA_CHECK(cudaMemcpy(dl, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
    auto layer = [&](int s, bool fused) {
      LayerW L;
      L.pool = P.d, L.desc = ddesc + (size_t)s * kE * 3, L.fmt = fmt;
      if (fused) {
        const uint16_t* w = shw + (size_t)(s % NC) * (gu + dn);
        L.sh.bgu = reinterpret_cast<const uint8_t*>(w), L.sh.bd = reinterpret_cast<const uint8_t*>(w + gu);
      }
      return L;
    };
    auto rt = [&](int s, cudaStream_t st) {
      route(dl + (size_t)s * kMaxRows * kE, db, M, di + s * kMaxRows * kTopK, dw + s * kMaxRows * kTopK, st);
    };
    auto gemvs = [&](int s, cudaStream_t st, float* y) {  // the runner's GEMV path (model.cu), kBF16T
      const uint16_t* w = shw + (size_t)(s % NC) * (gu + dn);
      glm::GemvSeg g;
      g.fmt = glm::WFmt::kBF16T, g.w = w, g.n = 2 * kI, g.y = hm, g.ldy = kI;
      glm::gemv(reinterpret_cast<const uint16_t*>(dx), kH, M, kH, &g, 1, glm::GemvOut::kSwiGLU, kShParts, st, true);
      glm::GemvSeg d;
      d.fmt = glm::WFmt::kBF16T, d.w = w + gu, d.n = kH, d.y = y, d.ldy = kH;
      glm::gemv(reinterpret_cast<const uint16_t*>(hm), kI, M, kI, &d, 1, glm::GemvOut::kF32, 1, st, true);
    };
    auto dec = [&](int s, cudaStream_t st, bool fused, const DecodeCfg& c, float* r, float* sy) {
      moe_decode(layer(s, fused), dx, di + s * kMaxRows * kTopK, dw + s * kMaxRows * kTopK, M, ws, r, sy, st, c);
    };
    for (int s = 0; s < NS; ++s) rt(s, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    // (1) bits, steps 0 and 5 (two shared copies), every variant
    const size_t n = (size_t)M * kH;
    std::vector<uint32_t> a(n), b(n), c(n), d(n);
    std::vector<uint16_t> ha((size_t)M * kI), hb((size_t)M * kI);
    int bad = 0, cases = 0;
    for (int s : {0, 5})
      for (int u = 0; u < decode_up_variants(); ++u)
        for (int dv = 0; dv < decode_down_variants(); ++dv) {
          if (u && dv) continue;  // every up with the default down, every down with the default up
          DecodeCfg cf;
          cf.up = u, cf.down = dv;
          CUDA_CHECK(cudaMemset(ys0, 0xFF, n * 4));
          CUDA_CHECK(cudaMemset(ys, 0xFF, n * 4));
          gemvs(s, 0, ys0);
          dec(s, 0, false, cf, yr0, ys0);
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(hb.data(), hm, hb.size() * 2, cudaMemcpyDeviceToHost));
          dec(s, 0, true, cf, yr, ys);
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(ha.data(), static_cast<uint16_t*>(ws) + (size_t)kShSlot * M * kI, ha.size() * 2,
                                cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(a.data(), ys, n * 4, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(b.data(), ys0, n * 4, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(c.data(), yr, n * 4, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(d.data(), yr0, n * 4, cudaMemcpyDeviceToHost));
          size_t dh = 0, dys = 0, dyr = 0;
          for (size_t i = 0; i < ha.size(); ++i) dh += ha[i] != hb[i];
          for (size_t i = 0; i < n; ++i) dys += a[i] != b[i], dyr += c[i] != d[i];
          ++cases;
          if (dh || dys || dyr) {
            ++bad;
            result("SHFUSED\t%s\trows %d\tstep %d\tup %s\tdown %s\tFAIL: h %zu, ys %zu, yr %zu of %zu differ", fname.c_str(), M, s,
                   decode_up_name(u), decode_down_name(dv), dh, dys, dyr, n);
          }
        }
    g_fail += bad;
    result("SHFUSED\t%s\trows %d\tbits\t%s: fused h, ys and yr = the GEMVs' and the unfused step's in %d of %d cases (every up "
           "and down variant, two shared copies)", fname.c_str(), M, bad ? "FAIL" : "PASS", cases - bad, cases);
    // (2) time, default variants, A B B A
    const DecodeCfg cf;
    auto tA = [&] { return time_graph(NS, [&](int s, cudaStream_t st) { rt(s, st); gemvs(s, st, ys); dec(s, st, false, cf, yr, ys); }); };
    auto tB = [&] { return time_graph(NS, [&](int s, cudaStream_t st) { rt(s, st); dec(s, st, true, cf, yr, ys); }); };
    const double a1 = tA(), b1 = tB(), b2 = tB(), a2 = tA();
    const double r0 = time_graph(NS, [&](int s, cudaStream_t st) { rt(s, st); dec(s, st, false, cf, yr, ys); });
    const double ta = (a1 + a2) / 2, tb = (b1 + b2) / 2;
    result("SHFUSED\t%s\trows %s\ttime\tA gemvs+experts %.1f / %.1f us\tB fused %.1f / %.1f us\t(A B B A)\tB/A %.4f\tgain %.1f us "
           "a layer\texperts alone %.1f us\tshared adds: A %.1f, B %.1f us\tMB %.2f\tB %% of 235 %.1f", fname.c_str(),
           names[ri].c_str(), a1, a2, b1, b2, tb / ta, ta - tb, r0, ta - r0, tb - r0, bytes / 1e6,
           100 * bytes / tb / 1e3 / 235.0);
  }
  cudaFree(ddesc), cudaFree(dl), cudaFree(db), cudaFree(yr), cudaFree(ys), cudaFree(yr0), cudaFree(ys0), cudaFree(di);
  cudaFree(dw), cudaFree(dx), cudaFree(hm), cudaFree(ws), cudaFree(P.d), cudaFree(shw);
}

// ---- altar-1's BF16 shared expert through attn's BF16 GEMV ----------------------------------------------------------
// A rank's slice: gate and up [512][6144] interleaved row by row (2i gate, 2i + 1 up: GemvOut::kSwiGLU's order),
// down [6144][512]; 8 copies cycled so L2 (24 MB) holds none across steps. A step = the SwiGLU GEMV then the down
// GEMV (f32 partials, as the ring all-reduces them). Bytes = the three matrices, 18.9 MB.
static void bench_shared(const std::string& rows_s) {
  constexpr int NC = 8;
  const size_t gu = (size_t)2 * kI * kH, dn = (size_t)kH * kI;
  uint16_t* w;
  CUDA_CHECK(cudaMalloc(&w, NC * (gu + dn) * 2));
  k_fill_b16<<<384, 256>>>(w, NC * (gu + dn), 99);
  uint16_t* x;
  __nv_bfloat16* hx;
  float* y;
  CUDA_CHECK(cudaMalloc(&x, (size_t)kMaxRows * kH * 2));
  CUDA_CHECK(cudaMalloc(&hx, (size_t)kMaxRows * kI * 2));
  CUDA_CHECK(cudaMalloc(&y, (size_t)kMaxRows * kH * 4));
  {
    std::vector<__nv_bfloat16> v;
    rnd_x(v, (size_t)kMaxRows * kH, 7);
    CUDA_CHECK(cudaMemcpy(x, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const double bytes = (gu + dn) * 2.0;
  std::vector<std::string> names;
  for (int M : parse_rows(rows_s, names))
    for (int sku : {1, 2, 4})
      for (int skd : {1, 2, 4}) {
        auto step = [&](int s, cudaStream_t st, int parts) {
          const uint16_t* wg = w + (size_t)(s % NC) * (gu + dn);
          if (parts & 1) {
            glm::GemvSeg g;
            g.w = wg, g.n = 2 * kI, g.y = hx, g.ldy = kI;
            glm::gemv(false, x, kH, M, kH, &g, 1, glm::GemvOut::kSwiGLU, sku, st, true);
          }
          if (parts & 2) {
            glm::GemvSeg d;
            d.w = wg + gu, d.n = kH, d.y = y, d.ldy = kH;
            glm::gemv(false, reinterpret_cast<const uint16_t*>(hx), kI, M, kI, &d, 1, glm::GemvOut::kF32, skd, st, true);
          }
        };
        const double t = time_graph(64, [&](int s, cudaStream_t st) { step(s, st, 3); });
        const double tu = time_graph(64, [&](int s, cudaStream_t st) { step(s, st, 1); });
        const double td = time_graph(64, [&](int s, cudaStream_t st) { step(s, st, 2); });
        result("SHARED\tbf16 gemv\trows %d\tsplitk up %d down %d\tMB %.2f\tus %.1f\t(up %.1f, down %.1f)\tGB/s %.1f\t%% of 235 %.1f\t%% of window %.1f\t76 layers ms %.2f",
               M, sku, skd, bytes / 1e6, t, tu, td, bytes / t / 1e3, 100 * bytes / t / 1e3 / 235.0, 100 * bytes / t / 1e3 / g_bw, 76 * t / 1e3);
      }
  cudaFree(w), cudaFree(x), cudaFree(hx), cudaFree(y);
}

struct PfOpts {
  std::string down = "0,1", yrkb = "0", cq = "4", up = "0,1";
  bool diag = false;
};
void bench_prefill(const std::string& fname, const std::string& chunks, const PfOpts& o);  // bench_prefill.inc

int main(int argc, char** argv) {
  std::string mode, fmt, rows = "1,6,16", chunks = "1024,4096,8192";
  bool sweep = false;
  PfOpts po;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--bw") mode = "bw";
    else if (a == "--decode" && i + 1 < argc) mode = "decode", fmt = argv[++i];
    else if (a == "--prefill" && i + 1 < argc) mode = "prefill", fmt = argv[++i];
    else if (a == "--shared") mode = "shared";
    else if (a == "--shfused" && i + 1 < argc) mode = "shfused", fmt = argv[++i];
    else if (a == "--rows" && i + 1 < argc) rows = argv[++i];
    else if (a == "--chunks" && i + 1 < argc) chunks = argv[++i];
    else if (a == "--sweep") sweep = true;
    else if (a == "--ups" && i + 1 < argc) {
      std::vector<std::string> nm;
      g_ups = parse_rows(argv[++i], nm);
    }
    else if (a == "--down" && i + 1 < argc) po.down = argv[++i];
    else if (a == "--yrkb" && i + 1 < argc) po.yrkb = argv[++i];
    else if (a == "--cq" && i + 1 < argc) po.cq = argv[++i];
    else if (a == "--up" && i + 1 < argc) po.up = argv[++i];
    else if (a == "--diag") po.diag = true;
    else if (a == "--out" && i + 1 < argc) g_out = fopen(argv[++i], "a");
    else {
      fprintf(stderr, "usage: see the header of bench_experts.cu\n");
      return 2;
    }
  }
  bench_bw();
  if (mode == "decode") bench_decode(fmt, rows, sweep);
  else if (mode == "prefill") bench_prefill(fmt, chunks, po);
  else if (mode == "shared") bench_shared(rows);
  else if (mode == "shfused") bench_shfused(fmt, rows);
  return g_fail ? 1 : 0;
}

#include "experts/tests/bench_prefill.inc"
