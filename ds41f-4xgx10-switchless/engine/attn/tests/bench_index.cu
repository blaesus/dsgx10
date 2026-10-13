// The decode indexer's dense scans (every rank scans every position, one launch per index layer), timed
// per launch at the given contexts: rows 1 and 6, ratio 2 without the pool (layers 2, 8, 14) and ratio 1
// with it (layer 20), scan sized for the context's 4096-position bucket as a captured step is. index_fused
// (a CTA per 2048 positions and row) and index_dense (what the runner launches), and index_fused's parts as
// the separate kernels (q prep, scores, the selections). Random keys and queries (fill.cuh); every timed
// launch follows an L2 flush (a read of 4x its size). Checks each selection against a host top-k of its
// keys, and index_dense's outputs against index_fused's. --sessions: the rows split into that many sessions
// (batching: consecutive rows, each session's keys in a slot of its own, all at the context; slots in reverse
// order), through a slot table; the separate kernels take one session, so only fused and dense are timed.
// --table: a slot table for one session too (B2's steps pass one): the slotted kernels at S=1.
// A case that differs (a DIFFERENT) keeps its outputs for a diagnosis afterwards: the first row whose
// selections differ between the kernels or from a host top-k, its keys and selections from both kernels and the
// host top-k of each one's keys, with the case's fill seed (its data), in DIR/bench_index_<context>_<rows>_<sessions>_
// <kind>_seed<N>/ (.npy, meta.txt). --dump DIR (default .); --keep-all: every case's (a test of the dump);
// --seed0 N: the fills from seed N (one case alone on another run's data: its seed and the same arguments).
//   bench_index [--iters N] [--iqm] [--rows 1,6] [--sessions 1,4] [--table] [--dump DIR] [--seed0 N] ctx...
#include <algorithm>
#include <filesystem>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "fill.cuh"
#include "testutil.h"

using namespace attn_test;

namespace {

constexpr int kTopk = 512, kPool = 2048, kBucket = 4096;
cudaStream_t g_s;
int g_iters = 30, g_bad = 0;
bool g_iqm = false;  // --iqm: report the interquartile mean (the timer ticks every ~2 us)
bool g_table = false;  // --table: a slot table for one session too
std::string g_dump = ".";  // --dump: where a case that differs keeps its outputs
bool g_keep_all = false;   // --keep-all: every case keeps them
Dev<uint8_t> g_flush;
Dev<uint32_t> g_sink;

// reads the flush buffer: the L2 then holds clean lines only, as in a step, where weights stream through
// (a memset would leave it dirty, and the timed kernel would pay for the write-backs)
__global__ void flush_kernel(const uint4* p, size_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) *sink = acc;
}

// One launch's outputs (keys, block maxima, selections), copied when it is done
struct Out {
  std::vector<uint16_t> keys, bkeys, tkey, pkey;
  std::vector<int> top, pl;
};

struct Case {
  int rows, ratio, ctx, max_scan, kstride, sessions, cap;
  bool pool, blk;  // blk: the Reindex layers' scan of layer 20's pool (ratio 1)
  Dev<int> pos, top, pl, counters, blocks, slot;
  Dev<float2> tab;
  Dev<uint8_t> ik_k, ik_sc, q;
  Dev<uint16_t> iq, keys, bkeys, tkey, pkey;
  Dev<float> wp, qw;
  Dev<uint32_t> qsc, scratch;
  Case(int rows_, int ratio_, int ctx_, bool pool_, bool blk_ = false, int sessions_ = 1)
      : rows(rows_), ratio(ratio_), ctx(ctx_), sessions(sessions_), pool(pool_), blk(blk_) {
    const int bucket = (ctx + kBucket - 1) / kBucket * kBucket;  // the step's last position is ctx - 1
    max_scan = blk ? kPool * 8 : bucket / ratio;
    kstride = (max_scan + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
    const int rs = rows / sessions;  // rows per session
    std::vector<int> h(rows), hs(rows);
    for (int r = 0; r < rows; ++r) h[r] = ctx - rs + r % rs, hs[r] = sessions - 1 - r / rs;
    pos = Dev<int>(h.data(), h.size());
    slot = Dev<int>(hs.data(), hs.size());
    tab = Dev<float2>((size_t)(ctx + 1) * 32);
    attn::rope_table_build(tab.p, ctx + 1, true, g_s);
    cap = bucket / ratio + 64;
    ik_k = Dev<uint8_t>((size_t)cap * 64 * sessions);
    ik_sc = Dev<uint8_t>((size_t)cap * 4 * sessions);
    iq = Dev<uint16_t>((size_t)rows * attn::kIdxHeads * attn::kIdxDim);
    wp = Dev<float>((size_t)rows * attn::kIdxHeads);
    fill::u8(ik_k, 0xFF, 0);
    fill::u8(ik_sc, 7, 120);
    fill::bf16(iq, 2.f);
    fill::f32(wp, 1.f);
    keys = Dev<uint16_t>((size_t)rows * kstride);
    bkeys = Dev<uint16_t>((size_t)rows * kstride / 8);
    top = Dev<int>((size_t)rows * kTopk);
    tkey = Dev<uint16_t>((size_t)rows * kTopk);
    pl = Dev<int>((size_t)rows * kPool);
    pkey = Dev<uint16_t>((size_t)rows * kPool);
    counters = Dev<int>(std::max(rows, attn::kMaxDecodeRows));
    scratch = Dev<uint32_t>(attn::index_dense_scratch() / 4);
    q = Dev<uint8_t>((size_t)rows * attn::kIdxHeads * 64);
    qsc = Dev<uint32_t>((size_t)rows * attn::kIdxHeads);
    qw = Dev<float>((size_t)rows * attn::kIdxHeads);
    if (blk) {  // each row's pool: ascending distinct blocks it can see, -1 padded
      std::mt19937 rng(ctx + rows);
      std::vector<int> hb((size_t)rows * kPool, -1);
      for (int r = 0; r < rows; ++r) {
        const int nb = (h[r] + 1 + 7) / 8;
        std::vector<int> all(nb);
        for (int b = 0; b < nb; ++b) all[b] = b;
        std::shuffle(all.begin(), all.end(), rng);
        const int k = std::min(nb, kPool);
        std::sort(all.begin(), all.begin() + k);
        for (int i = 0; i < k; ++i) hb[(size_t)r * kPool + i] = all[i];
      }
      blocks = Dev<int>(hb.data(), hb.size());
    }
    CUDA_CHECK(cudaStreamSynchronize(g_s));
  }
  attn::IndexScan scan() const {
    attn::IndexScan s{};
    if (blk) s.blocks = blocks.p, s.nblocks = kPool;
    return s;
  }
  attn::IndexKeys ik() const {
    attn::IndexKeys k{ik_k.p, ik_sc.p, 0};
    if ((sessions > 1 || g_table) && rows <= attn::kMaxDecodeRows)
      k.slot = slot.p, k.k_stride = (size_t)cap * 64, k.sc_stride = (size_t)cap * 4;
    return k;
  }
  void fused() {
    attn::IndexSel t{kTopk, top.p, tkey.p, kTopk}, ps{};
    if (pool) ps = attn::IndexSel{kPool, pl.p, pkey.p, kPool};
    attn::index_fused(iq.p, wp.p, pos.p, tab.p, rows, ratio, ik(), scan(), max_scan, keys.p, kstride,
                      pool ? bkeys.p : nullptr, kstride / 8, t, ps, counters.p, false, g_s);
  }
  void dense() {
    attn::IndexSel t{kTopk, top.p, tkey.p, kTopk}, ps{};
    if (blk) {
      attn::index_dense_blocks(iq.p, wp.p, pos.p, tab.p, rows, ratio, ik(), blocks.p, kPool, keys.p, kstride, t,
                               scratch.p, false, g_s);
      return;
    }
    if (pool) ps = attn::IndexSel{kPool, pl.p, pkey.p, kPool};
    attn::index_dense(iq.p, wp.p, pos.p, tab.p, rows, ratio, ik(), max_scan, keys.p, kstride, bkeys.p,
                      kstride / 8, t, ps, scratch.p, false, g_s);
  }
  attn::IndexQ iqd() { return attn::IndexQ{q.p, qsc.p, qw.p}; }
  void q_prep() { attn::index_q_prep(iq.p, wp.p, pos.p, tab.p, iqd(), nullptr, rows, g_s); }
  void score() {
    attn::index_score(iqd(), pos.p, rows, ratio, ik(), scan(), max_scan, keys.p, kstride,
                      pool ? bkeys.p : nullptr, kstride / 8, g_s);
  }
  void topk() {
    attn::index_topk(keys.p, kstride, pos.p, rows, ratio, scan(), max_scan, false, kTopk, top.p, tkey.p,
                     kTopk, g_s);
  }
  void separate() {  // q prep, scores, selections: the path for many rows
    q_prep();
    score();
    topk();
    if (pool) pool_topk();
  }
  void pool_topk() {
    attn::index_topk(bkeys.p, kstride / 8, pos.p, rows, ratio, attn::IndexScan{}, max_scan, true, kPool, pl.p, pkey.p,
                     kPool, g_s);
  }
  Out out() const {
    CUDA_CHECK(cudaStreamSynchronize(g_s));
    return Out{keys.host(), bkeys.host(), tkey.host(), pkey.host(), top.host(), pl.host()};
  }
};

// torch.topk's choice over one row's keys (0 = no candidate): the kmax highest, ties to the lower scan index,
// in ascending scan order
std::vector<int> ref_topk(const uint16_t* keys, int n, int kmax) {
  std::vector<int> idx;
  for (int i = 0; i < n; ++i)
    if (keys[i]) idx.push_back(i);
  const int k = std::min<int>(kmax, (int)idx.size());
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return keys[a] != keys[b] ? keys[a] > keys[b] : a < b; });
  idx.resize(k);
  std::sort(idx.begin(), idx.end());
  return idx;
}

template <class T>
void npy_save(const std::string& path, const T* p, size_t n) {
  std::string h = std::string("{'descr': '") + (sizeof(T) == 2 ? "<u2" : "<i4") +
                  "', 'fortran_order': False, 'shape': (" + std::to_string(n) + ",), }";
  h += std::string((64 - (10 + h.size() + 1) % 64) % 64, ' ') + "\n";  // the data 64-byte aligned
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) die("bench_index: cannot write %s", path.c_str());
  const uint16_t hl = (uint16_t)h.size();
  std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(h.data(), 1, h.size(), f);
  std::fwrite(p, sizeof(T), n, f);
  std::fclose(f);
}

// A case's outputs from the fused (f) and the dense or separate (d) launches, for a diagnosis afterwards: the
// first row whose keys or selections differ between them, or whose selections are not a host top-k of the
// launch's own keys (as check() takes it; block scans: positions through the row's blocks). Its keys,
// selections (positions and keys) and the host top-k of each one's keys, the pool's alike, the row's blocks,
// and meta.txt (the arguments that make the case again: its seed, context, rows, sessions, kind; the row).
void keep(const Case& c, int kind, uint32_t seed, const Out& f, const Out& d) {
  const int n = c.blk ? kPool * 8 : c.ctx / c.ratio, nb = (n + 7) / 8, ks = c.kstride, bs = c.kstride / 8;
  const std::vector<int> bl = c.blk ? c.blocks.host() : std::vector<int>();
  auto host_top = [&](const std::vector<uint16_t>& keys, int stride, int r, int len, int kmax) {
    const std::vector<int> w = ref_topk(&keys[(size_t)r * stride], len, kmax);
    std::vector<int> p(kmax, -1);
    for (size_t i = 0; i < w.size(); ++i) {
      const int b = c.blk ? bl[(size_t)r * kPool + w[i] / 8] : 0;
      p[i] = !c.blk ? w[i] : b < 0 ? -1 : b * 8 + w[i] % 8;
    }
    return p;
  };
  auto part = [](const auto& v, size_t at, size_t len) {
    return std::vector<typename std::decay_t<decltype(v)>::value_type>(v.begin() + at, v.begin() + at + len);
  };
  auto row_bad = [&](int r) {
    bool bad = part(f.keys, (size_t)r * ks, n) != part(d.keys, (size_t)r * ks, n) ||
               part(f.top, (size_t)r * kTopk, kTopk) != part(d.top, (size_t)r * kTopk, kTopk) ||
               part(f.tkey, (size_t)r * kTopk, kTopk) != part(d.tkey, (size_t)r * kTopk, kTopk) ||
               host_top(f.keys, ks, r, n, kTopk) != part(f.top, (size_t)r * kTopk, kTopk) ||
               host_top(d.keys, ks, r, n, kTopk) != part(d.top, (size_t)r * kTopk, kTopk);
    if (c.pool)
      bad = bad || part(f.bkeys, (size_t)r * bs, nb) != part(d.bkeys, (size_t)r * bs, nb) ||
            part(f.pl, (size_t)r * kPool, kPool) != part(d.pl, (size_t)r * kPool, kPool) ||
            part(f.pkey, (size_t)r * kPool, kPool) != part(d.pkey, (size_t)r * kPool, kPool) ||
            host_top(f.bkeys, bs, r, nb, kPool) != part(f.pl, (size_t)r * kPool, kPool) ||
            host_top(d.bkeys, bs, r, nb, kPool) != part(d.pl, (size_t)r * kPool, kPool);
    return bad;
  };
  int r = 0;
  while (r + 1 < c.rows && !row_bad(r)) ++r;
  const std::string dir = g_dump + "/bench_index_" + std::to_string(c.ctx) + "_" + std::to_string(c.rows) + "_" +
                          std::to_string(c.sessions) + "_" + std::to_string(kind) + "_seed" + std::to_string(seed);
  std::filesystem::create_directories(dir);
  for (int w = 0; w < 2; ++w) {
    const Out& o = w ? d : f;
    const std::string pre = dir + (w ? "/dense_" : "/fused_");
    npy_save(pre + "keys.npy", &o.keys[(size_t)r * ks], n);
    npy_save(pre + "top.npy", &o.top[(size_t)r * kTopk], kTopk);
    npy_save(pre + "tkey.npy", &o.tkey[(size_t)r * kTopk], kTopk);
    const std::vector<int> h = host_top(o.keys, ks, r, n, kTopk);
    npy_save(pre + "host_top.npy", h.data(), h.size());
    if (c.pool) {
      npy_save(pre + "bkeys.npy", &o.bkeys[(size_t)r * bs], nb);
      npy_save(pre + "pool.npy", &o.pl[(size_t)r * kPool], kPool);
      npy_save(pre + "pkey.npy", &o.pkey[(size_t)r * kPool], kPool);
      const std::vector<int> hp = host_top(o.bkeys, bs, r, nb, kPool);
      npy_save(pre + "host_pool.npy", hp.data(), hp.size());
    }
  }
  if (c.blk) npy_save(dir + "/blocks.npy", &bl[(size_t)r * kPool], kPool);
  FILE* m = std::fopen((dir + "/meta.txt").c_str(), "w");
  if (!m) die("bench_index: cannot write %s/meta.txt", dir.c_str());
  std::fprintf(m, "--seed0 %u --rows %d --sessions %d --kinds %d%s %d\nrow %d (%s), scan keys %d; fused_*: index_fused, "
               "dense_*: %s\n", seed, c.rows, c.sessions, kind, g_table ? " --table" : "", c.ctx, r,
               row_bad(r) ? "differs" : "every row agrees", n,
               c.rows > attn::kMaxDecodeRows ? "the separate kernels" : c.blk ? "index_dense_blocks" : "index_dense");
  std::fclose(m);
  std::printf("    kept: row %d's outputs in %s\n", r, dir.c_str());
}

// the selections of the last launch against ref_topk of its own keys; returns the rows that differ
int check(const Case& c) {
  CUDA_CHECK(cudaStreamSynchronize(g_s));
  if (c.blk) return 0;  // fused_tests checks block scans
  const auto keys = c.keys.host(), bkeys = c.bkeys.host(), tkey = c.tkey.host(), pkey = c.pkey.host();
  const auto top = c.top.host(), pl = c.pl.host();
  const int n = c.ctx / c.ratio, nb = (n + 7) / 8;  // the last row sees the most
  int bad = 0;
  for (int r = 0; r < c.rows; ++r) {
    bool ok = true;
    auto sel = [&](const uint16_t* k, int len, int kmax, const int* out, const uint16_t* okey) {
      const std::vector<int> want = ref_topk(k, len, kmax);
      for (int i = 0; i < kmax; ++i) {
        const int p = i < (int)want.size() ? want[i] : -1;
        const uint16_t v = i < (int)want.size() ? k[p] : 0;
        if (out[i] != p || okey[i] != v) ok = false;
      }
    };
    sel(&keys[(size_t)r * c.kstride], n, kTopk, &top[(size_t)r * kTopk], &tkey[(size_t)r * kTopk]);
    if (c.pool) sel(&bkeys[(size_t)r * c.kstride / 8], nb, kPool, &pl[(size_t)r * kPool], &pkey[(size_t)r * kPool]);
    bad += !ok;
  }
  return bad;
}

// median time of f in microseconds, each launch after an L2 flush
template <class F>
double time_us(F&& f) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<double> t;
  for (int i = 0; i < g_iters + 2; ++i) {
    flush_kernel<<<192, 512, 0, g_s>>>(reinterpret_cast<const uint4*>(g_flush.p), g_flush.n / 16, g_sink.p);
    CUDA_CHECK(cudaEventRecord(e0, g_s));
    f();
    CUDA_CHECK(cudaEventRecord(e1, g_s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (i >= 2) t.push_back(1e3 * ms);
  }
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  std::sort(t.begin(), t.end());
  if (g_iqm) {
    double sum = 0;
    for (size_t i = t.size() / 4; i < t.size() - t.size() / 4; ++i) sum += t[i];
    return sum / (t.size() - 2 * (t.size() / 4));
  }
  return t[t.size() / 2];
}

std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* q = s; *q;) {
    v.push_back(std::atoi(q));
    while (*q && *q != ',') ++q;
    if (*q) ++q;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> ctxs, rows{1, 6}, kinds{0, 1}, sessions{1};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) g_iters = std::atoi(argv[++i]);
    else if (a == "--iqm") g_iqm = true;
    else if (a == "--table") g_table = true;
    else if (a == "--dump" && i + 1 < argc) g_dump = argv[++i];
    else if (a == "--keep-all") g_keep_all = true;
    else if (a == "--seed0" && i + 1 < argc) fill::g_seed = (uint32_t)std::atoll(argv[++i]);
    else if (a == "--rows" && i + 1 < argc) rows = parse_list(argv[++i]);
    else if (a == "--kinds" && i + 1 < argc) kinds = parse_list(argv[++i]);
    else if (a == "--sessions" && i + 1 < argc) sessions = parse_list(argv[++i]);
    else ctxs.push_back(std::atoi(argv[i]));
  }
  if (ctxs.empty()) ctxs = {1024, 16384, 32768, 65536, 131072, 262144};
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  g_flush = Dev<uint8_t>((size_t)std::max(prop.l2CacheSize, 1 << 20) * 4);
  g_sink = Dev<uint32_t>(1);
  std::printf("%s: %d SMs, L2 %d MB; %s of %d launches, each after an L2 flush (us)%s\n", prop.name,
              prop.multiProcessorCount, prop.l2CacheSize >> 20, g_iqm ? "interquartile mean" : "median", g_iters,
              g_table ? "; slot tables for one session too" : "");
  std::printf("%8s %4s %4s %-14s %9s %9s | %8s %8s %8s %8s | %s\n", "context", "rows", "sess", "layer", "fused",
              "dense", "q prep", "scores", "top-k", "pool", "selections vs a host top-k; dense vs fused");
  for (int ctx : ctxs)
    for (int R : rows)
      for (int S : sessions)
        for (int kind : kinds) {
          if (S < 1 || R % S || (S > 1 && R > attn::kMaxDecodeRows)) continue;
          const uint32_t seed = fill::g_seed;  // the case's data: fills from this seed on (--seed0)
          Case c(R, kind ? 1 : 2, ctx, kind == 1, kind == 2, S);
          c.fused();  // the kernels' one-time setup
          int bad = check(c);
          const Out fo = c.out();
          const bool many = R > attn::kMaxDecodeRows;  // prefill rows: the separate path instead of index_dense
          if (many) c.separate();
          else c.dense();
          const int dbad = check(c);
          const Out dout = c.out();
          // index_dense writes block maxima for every scan, index_fused for the pool
          const bool same = fo.keys == dout.keys && (!c.pool || fo.bkeys == dout.bkeys) && fo.top == dout.top &&
                            fo.pl == dout.pl;
          const double tf = time_us([&] { c.fused(); });
          const double td = many ? time_us([&] { c.separate(); }) : time_us([&] { c.dense(); });
          double tq = 0, ts = 0, tt = 0, tp = 0;  // the separate kernels: one session
          if (S == 1 && (!g_table || many)) {
            tq = time_us([&] { c.q_prep(); });
            c.q_prep();
            ts = time_us([&] { c.score(); });
            c.score();
            tt = time_us([&] { c.topk(); });
            tp = c.pool ? time_us([&] { c.pool_topk(); }) : 0.0;
          }
          std::printf("%8d %4d %4d %-14s %9.1f %9.1f | %8.1f %8.1f %8.1f %8.1f | %s; %s\n", ctx, R, S,
                      kind == 2 ? "24-36 (pool)" : kind ? "20 (+pool)" : "2, 8, 14", tf, td, tq, ts, tt, tp,
                      bad ? "DIFFERENT" : "same", dbad || !same ? "DIFFERENT" : "same");
          g_bad += bad + dbad + !same;
          if (bad || dbad || !same || g_keep_all) keep(c, kind, seed, fo, dout);
        }
  return g_bad ? 1 : 0;
}
