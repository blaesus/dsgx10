// Benchmarks of the routed-expert kernels on one GB10 (GPU: only through ring.sh).
//   bench_experts --bw                                   the window's streaming read
//   bench_experts --decode q4|s3|mix [--rows 1,2,4,6,8,16,6c] [--sweep] [--out f.tsv]
//   bench_experts --prefill q4|s3|mix [--chunks 1024,4096,8192] [--fp4] [--down 0,1] [--yrkb 0] [--side 0] [--cq 4] [--up 0,1] [--sh 128] [--diag]
//                 [--out f.tsv]   (down: PrefillCfg::down variants; yrkb: chained-down L2 budgets, KB; side: the
//                 shared expert on a second stream; cq: chained-down passes in flight; up: PrefillCfg::up variants; diag: the gate/up kernel and the chained down with parts switched off; sh: the shared down's rows a CTA)
// Decode: a pool of >= 2.2 GB of experts in the format (random bits, valid scales) is cycled so L2 holds no
// step's weights; each step routes its rows (logits that pick a chosen set) and runs the expert step; a CUDA
// graph of whole steps, the median of 7 replays. Bytes = the union's weights and scales + the shared expert's.
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "experts/formats.cuh"
#include "experts/tests/testdata.h"

using namespace gx;

static FILE* g_out = nullptr;
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
// a matrix region of `blocks` blocks of size bs whose bytes from so on are scales: random bits, valid scales
__global__ void k_fill_mat(uint8_t* p, size_t blocks, int bs, int so, uint64_t seed, int sub2) {
  const size_t n = blocks * bs / 4;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint64_t h = mix64(seed ^ (i * 0x9E3779B97F4A7C15ull));
    uint32_t v = (uint32_t)h;
    if ((int)((i * 4) % bs) >= so) {
      v = 0;
      for (int b = 0; b < 4; ++b) v |= (uint32_t)(rnd_scale(h >> (8 * b)) | (sub2 && (h >> (32 + b)) & 1 ? 0x80 : 0)) << (8 * b);
    }
    reinterpret_cast<uint32_t*>(p)[i] = v;
  }
}
__global__ void k_fill_fp8(uint8_t* p, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n / 4; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t v = (uint32_t)mix64(seed ^ (i * 0x9E3779B97F4A7C15ull)) & 0xC7C7C7C7u;  // small exponents, no NaN
    reinterpret_cast<uint32_t*>(p)[i] = v | 0x30303030u;
  }
}
__global__ void k_fill_f32(float* p, size_t n, float v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v;
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
  int NP = 0;
  std::vector<uint32_t> fmt;  // [NP * 3]
  std::vector<uint64_t> off;
  std::vector<size_t> bytes;  // a stored expert's routed bytes
  uint8_t* d = nullptr;
  size_t total = 0;
};
static Pool make_pool(const Pattern& pat, double target) {
  Pool P;
  size_t per = 0;
  for (int j = 0; j < 3; ++j) per += mat_bytes(pat.f[j], j < 2 ? kI : kH, j < 2 ? kH : kI);
  P.NP = (int)(target / per) + 1;
  for (int e = 0; e < P.NP; ++e) {
    size_t b = 0;
    for (int j = 0; j < 3; ++j) {
      P.fmt.push_back(pat.f[j]);
      P.off.push_back(P.total);
      const size_t mb = mat_bytes(pat.f[j], j < 2 ? kI : kH, j < 2 ? kH : kI);
      P.total += mb;
      b += mb;
    }
    P.bytes.push_back(b);
  }
  CUDA_CHECK(cudaMalloc(&P.d, P.total));
  for (int e = 0; e < P.NP; ++e)
    for (int j = 0; j < 3; ++j) {
      const uint32_t f = pat.f[j];
      const int bs = f == kNVFP4 ? kQ4Blk : kS3Blk, so = f == kNVFP4 ? 2048 : 1536;
      const size_t blocks = mat_bytes(f, j < 2 ? kI : kH, j < 2 ? kH : kI) / bs;
      k_fill_mat<<<192, 256>>>(P.d + P.off[e * 3 + j], blocks, bs, so, e * 3 + j + 1, f == kS3 && pat.nsub[j] > 1);
    }
  CUDA_CHECK(cudaDeviceSynchronize());
  return P;
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
  const int pi = fname == "q4" ? 0 : fname == "s3" ? 2 : 3;
  const Pattern& pat = kPatterns[pi];
  Pool P = make_pool(pat, 2.2e9);
  // shared experts: 8 copies (each 9.4 MB, so L2 holds none across steps)
  constexpr int NSH = 8;
  const size_t shm = (size_t)kI * kH;
  uint8_t* shw;
  float* shs;
  CUDA_CHECK(cudaMalloc(&shw, NSH * 3 * shm));
  CUDA_CHECK(cudaMalloc(&shs, NSH * 3 * 4 * 48 * 4));
  k_fill_fp8<<<192, 256>>>(shw, NSH * 3 * shm, 77);
  k_fill_f32<<<16, 256>>>(shs, NSH * 3 * 4 * 48, 1.f / 512);
  const double sh_bytes = 3.0 * shm + 3 * 4 * 48 * 4;
  printf("== decode %s: %d experts in the pool (%.2f GB), %.3f MB an expert\n", fname.c_str(), P.NP, P.total / 1e9,
         P.bytes[0] / 1e6);
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
  // per-step descriptors: id e of step s -> stored expert (s * 211 + e * 7) % NP, cycling the pool
  {
    std::vector<MatDesc> d((size_t)NS * kE * 3);
    for (int s = 0; s < NS; ++s)
      for (int e = 0; e < kE; ++e) {
        const int x = (int)(((size_t)s * kE + e) % P.NP);
        for (int j = 0; j < 3; ++j) {
          const float mags[2][4] = {{kSubA[0], kSubA[1], kSubA[2], kSubA[3]}, {kSubB[0], kSubB[1], kSubB[2], kSubB[3]}};
          d[((size_t)s * kE + e) * 3 + j] = make_desc(P.off[x * 3 + j], j < 2 ? kS2Up : kS2Down, P.fmt[x * 3 + j], pat.nsub[j], mags);
        }
      }
    CUDA_CHECK(cudaMemcpy(ddesc, d.data(), d.size() * sizeof(MatDesc), cudaMemcpyHostToDevice));
  }
  for (size_t ri = 0; ri < rows.size(); ++ri) {
    const int M = rows[ri];
    const bool corr = names[ri].find('c') != std::string::npos;  // "6c": 6 rows over 28 experts (experiment 1's case)
    // each step's picks; logits that make route() pick exactly them
    std::vector<float> lg((size_t)NS * kMaxRows * kE, -8.f);
    double bytes = 0, uavg = 0;
    for (int s = 0; s < NS; ++s) {
      std::vector<int> pool;
      if (corr) {
        std::set<int> c;
        uint64_t h = mix64(s * 31 + 7);
        while ((int)c.size() < 28) c.insert((int)((h = mix64(h)) % kE));
        pool.assign(c.begin(), c.end());
      }
      std::set<int> uni;
      for (int m = 0; m < M; ++m) {
        std::set<int> row;
        uint64_t h = mix64((uint64_t)s * 1000 + m * 17 + M);
        int k = 0;
        while ((int)row.size() < kTopK) {
          h = mix64(h);
          int e = corr ? pool[(m * kTopK + k++) % 28] : (int)(h % kE);
          if (corr && row.count(e)) e = pool[h % 28];
          row.insert(e);
        }
        int r = 0;
        for (int e : row) lg[((size_t)s * kMaxRows + m) * kE + e] = 8.f + 0.01f * r++, uni.insert(e);
      }
      for (int e : uni) bytes += P.bytes[((size_t)s * kE + e) % P.NP];
      uavg += uni.size();
    }
    bytes = bytes / NS + sh_bytes;
    uavg /= NS;
    CUDA_CHECK(cudaMemcpy(dl, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
    auto layer = [&](int s) {
      const int c = s % NSH;
      return LayerW{P.d, ddesc + (size_t)s * kE * 3,
                    SharedW{shw + (3 * c) * shm, shw + (3 * c + 1) * shm, shw + (3 * c + 2) * shm, shs + (3 * c) * 192,
                            shs + (3 * c + 1) * 192, shs + (3 * c + 2) * 192}};
    };
    auto rt = [&](int s, cudaStream_t st) {
      route(dl + (size_t)s * kMaxRows * kE, db, M, di + s * kMaxRows * kTopK, dw + s * kMaxRows * kTopK, st);
    };
    // every step's picks once, for the timings without the router
    for (int s = 0; s < NS; ++s) rt(s, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto line = [&](const char* what, const DecodeCfg& c, double t, double tr) {
      result("DECODE\t%s\trows %s\tU %.1f\t%s\tup %s%s\tdown %s%s\tMB %.2f\tus %.1f\t(route %.1f)\tGB/s %.1f\t%% of 235 %.1f\t%% of window %.1f\t75 layers ms %.2f",
             fname.c_str(), names[ri].c_str(), uavg, what, decode_up_name(c.up), c.dyn ? " dyn" : "", decode_down_name(c.down),
             c.pdl ? "" : " nopdl", bytes / 1e6, t, tr, bytes / t / 1e3, 100 * bytes / t / 1e3 / 235.0, 100 * bytes / t / 1e3 / g_bw,
             75 * t / 1e3);
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
  cudaFree(ws), cudaFree(shw), cudaFree(shs), cudaFree(P.d);
}

struct PfOpts {
  std::string down = "0,1", yrkb = "0", side = "0", cq = "4", up = "0,1";
  std::string sh = "128";  // the shared down's rows a CTA (PrefillCfg::sh_rows)
  bool diag = false;
};
void bench_prefill(const std::string& fname, const std::string& chunks, bool fp4, const PfOpts& o);  // bench_prefill.inc

int main(int argc, char** argv) {
  std::string mode, fmt, rows = "1,2,4,6,8,16,6c", chunks = "1024,4096,8192";
  bool sweep = false, fp4 = false;
  PfOpts po;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--bw") mode = "bw";
    else if (a == "--decode" && i + 1 < argc) mode = "decode", fmt = argv[++i];
    else if (a == "--prefill" && i + 1 < argc) mode = "prefill", fmt = argv[++i];
    else if (a == "--rows" && i + 1 < argc) rows = argv[++i];
    else if (a == "--chunks" && i + 1 < argc) chunks = argv[++i];
    else if (a == "--sweep") sweep = true;
    else if (a == "--fp4") fp4 = true;
    else if (a == "--down" && i + 1 < argc) po.down = argv[++i];
    else if (a == "--sh" && i + 1 < argc) po.sh = argv[++i];
    else if (a == "--yrkb" && i + 1 < argc) po.yrkb = argv[++i];
    else if (a == "--side" && i + 1 < argc) po.side = argv[++i];
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
  else if (mode == "prefill") bench_prefill(fmt, chunks, fp4, po);
  return 0;
}

#include "experts/tests/bench_prefill.inc"
