// One rank's decode step at TP=4 (DESIGN decision 1) as one CUDA graph: the fused attention side and mHC
// (engine/attn) with the decode GEMVs (engine/kernels) for all 40 layers and the head, every kernel a PDL
// launch, on synthetic weights and state. Collectives are local stand-ins (STAND-IN below): one PDL kernel per
// collective point that puts this rank's data where the gathered or reduced data would land. Variants are
// timed round-robin in one process (the GPU is shared) and reported as medians:
//   gemv             the GEMV chain alone: engine/kernels' step bench (STEP_FUSE=2: norms by dsk::rmsnorm and
//                    the gate's fused norm, the indexer's wq_b its own launch)
//   fused            the step: fused attention side + GEMVs + stand-ins; mHC projection split by rank (each
//                    sublayer's 25 mix partials all-reduced: a stand-in)
//   fused-hcrep      the same with the mHC projection replicated (no mix all-reduces)
//   fused-nostandin  fused without the stand-in kernels
//   attn             the fused attention side and mHC alone
//   attn-separate    the separate kernels (before fusion) alone
// --table: every state struct with a slot table of one session (batching's B2 passes one each step).
// --sessions S: the rows split into S sessions of consecutive rows, as even as can be (the first M % S one row
// more; at most 8 a session: attn.h), each with its state in a slot of its own and at the context, slots in
// reverse order, through a slot table (B4: 9-16 rows; the separate kernels take no table, so not
// attn-separate; past 8 rows the GEMVs need dsk's 16-row builds).
//   bench_step [--rows 1,6] [--rounds N] [--variants a,b,..] [--splits S] [--table] [--sessions S] [--counts]
//              [--attn-micro] ctx...
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"
#include "fill.cuh"
#include "kernels/dsk.h"
#include "testutil.h"

using namespace attn_test;
using attn::kDim;
using attn::kHcDim;

namespace {

constexpr int kL = 40, kParts = 4, kTopk = 512, kPool = 2048, kRows = attn::kMaxDecodeRows, kCopies = 4, kNE = 96;
constexpr int kPicks = kRows * 6;  // a layer's MoE picks: 6 a row
constexpr float kEps = 1e-20f;
int g_splits = 0;  // --splits: CTAs per row of the sparse attention (0: about one per SM in all)
const int* g_slot = nullptr;  // --table, --sessions: the rows' slots (configure)
bool g_table = false;         // --table
int g_sessions = 1;           // --sessions
int g_nslots = 1;             // the state's slots: the most sessions of the run
int ratio_of(int L) { return L < 2 ? 0 : L < 20 ? 2 : 1; }
bool kv_src(int L) { return L == 2 || L == 8 || L == 14 || L == 20; }
int kv_source(int L) { return L < 2 ? -1 : L < 8 ? 2 : L < 14 ? 8 : L < 20 ? 14 : 20; }
bool idx_src(int L) { return kv_src(L) || (L >= 24 && L % 4 == 0); }
int index_source(int L) { return L < 20 ? kv_source(L) : 20 + (L - 20) / 4 * 4; }

__nv_bfloat16* bf(uint16_t* p) { return reinterpret_cast<__nv_bfloat16*>(p); }

// ---- STAND-INs for the collectives ------------------------------------------------------------------

struct Slice {  // per row: src [bytes] -> dst [copies][bytes] (bytes a multiple of 16)
  const void* src;
  void* dst;
  int bytes, copies;
};
struct Slices {
  Slice t[4];
  int n;
};
// all-gather: this rank's slice into every slot of the gathered rows
__global__ void standin_gather(Slices a, int rows) {
  attn::pdl_trigger();
  attn::pdl_wait();
  for (int i = 0; i < a.n; ++i) {
    const Slice& sl = a.t[i];
    int words = sl.bytes / 16, per_row = words * sl.copies;
    for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < rows * per_row; w += gridDim.x * blockDim.x) {
      int r = w / per_row, k = w % per_row % words;
      reinterpret_cast<uint4*>(sl.dst)[w] = reinterpret_cast<const uint4*>(sl.src)[(size_t)r * words + k];
    }
  }
}
// all-reduce of the MoE's fp32 partial into the bf16 sublayer output
__global__ void standin_reduce(const float* src, uint16_t* dst, int n) {
  attn::pdl_trigger();
  attn::pdl_wait();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = attn::f2bf(src[i]);
}
// all-reduce of a sublayer's 25 mix partials per row: this rank's chunks summed (the ranks' sum would follow)
__global__ void standin_mix(const float* chunks, int nch, int rows, float* out) {
  attn::pdl_trigger();
  attn::pdl_wait();
  int i = threadIdx.x;
  if (i >= rows * 25) return;
  float s = 0.f;
  for (int c = 0; c < nch; ++c) s = __fadd_rn(s, chunks[(size_t)c * rows * 25 + i]);
  out[i] = s;
}

// ---- the bench ----------------------------------------------------------------------------------------

struct Variant {
  const char* name;
  bool gemv, attn, fused, standins, hc_split;
};
const Variant kVariants[] = {
    {"gemv", true, false, false, false, false},          {"fused", true, true, true, true, true},
    {"fused-hcrep", true, true, true, true, false},      {"fused-nostandin", true, true, true, false, true},
    {"attn", false, true, true, false, true},            {"attn-separate", false, true, false, false, true},
};

uint8_t* dev_bytes(size_t n, uint32_t mask, uint32_t add) {
  Dev<uint8_t> d(n);
  fill::u8(d, mask, add);
  uint8_t* p = d.p;
  d.p = nullptr;  // kept for the process
  return p;
}

struct DenseW {  // one copy of a layer's FP8/bf16 GEMV weights (engine/kernels per-rank shapes)
  uint8_t *wqa, *sqa, *wkv, *skv, *wqb, *sqb, *wix, *six, *woa, *soa, *wob, *sob, *gate;
  uint16_t *n_attn, *n_q, *n_ffn;
};

struct AttnLayer {
  int ratio = 0, cap = 0, pmax = 0, kstride = 0;
  Dev<uint8_t> ring_kv, ring_sc, ckv_kv, ckv_sc, ik_k, ik_sc;
  Dev<uint16_t> qnw, kvw, hnorm[2], wkv, wgate, cnorm, wk, knorm, wproj, keys, bkeys, cand_key, pcand_key, topkey,
      poolkey;
  Dev<float> sink, fn[2], hsc[2], hbase[2], carry_kv, carry_sc;
  Dev<int> cand, pcand, top, pool;
  attn::CompressorWeights cw() const { return {ratio, wkv.p, wgate.p, cnorm.p, wk.p, knorm.p}; }
  // the state holds g_nslots slots, each a 1/g_nslots of every buffer
  attn::CompressorCarry carry() const {
    attn::CompressorCarry c{carry_kv.p, carry_sc.p};
    if (g_slot) c.slot = g_slot, c.stride = carry_kv.n / g_nslots;
    return c;
  }
  attn::Compressed ckv() const {
    attn::Compressed c{ckv_kv.p, ckv_sc.p, cap};
    if (g_slot) c.slot = g_slot, c.kv_stride = ckv_kv.n / g_nslots, c.sc_stride = ckv_sc.n / g_nslots;
    return c;
  }
  attn::IndexKeys ik_one(int slot = 0) const {  // index_score: one session (slot), no table
    return {ik_k.p + slot * (ik_k.n / g_nslots), ik_sc.p + slot * (ik_sc.n / g_nslots), cap};
  }
  attn::IndexKeys ik() const {
    attn::IndexKeys k{ik_k.p, ik_sc.p, cap};
    if (g_slot) k.slot = g_slot, k.k_stride = ik_k.n / g_nslots, k.sc_stride = ik_sc.n / g_nslots;
    return k;
  }
  attn::Window win() const {
    attn::Window w{ring_kv.p, ring_sc.p};
    if (g_slot) w.slot = g_slot, w.kv_stride = ring_kv.n / g_nslots, w.sc_stride = ring_sc.n / g_nslots;
    return w;
  }
};

struct Bench {
  int ctx_max, ctx = 0, M = 0, nu = 0;
  cudaStream_t s;
  // GEMV weights
  std::vector<DenseW> dw;
  std::vector<dsk::MoeWeights> mw;
  uint8_t *head, *moe_ws;
  uint16_t* n_out;
  float* bias;
  Dev<int> idx, idx_dummy;
  Dev<float> wts;
  // attention side
  std::vector<AttnLayer> ly;
  Dev<float2> tab_plain, tab_yarn;
  Dev<float> fn_head, hsc_head, hbase_head;
  // activations
  Dev<uint16_t> stream[2], sub, y, xg, xn, qa_loc, kv_loc, qa, kvp, qr, qb, ix_loc, iq, o, ob;
  Dev<float> cproj_loc, cscore_loc, cproj, cscore, chunks, part, pre, post, comb, wp, lg, lg_all, ymoe, logits, po, pml;
  Dev<uint8_t> xq, xs, qq, qs, oq, os, iqq;
  Dev<uint32_t> iqsc;
  Dev<float> iqw;
  Dev<int> pos, cnt_idx, slot;
  std::vector<int> hslot;
  int cur = 0;  // the stream buffer the next mix reads

  explicit Bench(int ctx_max_) : ctx_max(ctx_max_) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    setup_gemv();
    setup_attn();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void setup_gemv() {
    const int kD = kDim;
    for (int c = 0; c < kCopies; ++c) {
      DenseW d;
      auto fp8 = [&](size_t n, size_t k, uint8_t*& w, uint8_t*& sc) {
        w = dev_bytes(n * k, 0x77, 0);
        sc = dev_bytes(n / 32 * (k / 32), 0x0F, 118);
      };
      fp8(320, kD, d.wqa, d.sqa);
      fp8(128, kD, d.wkv, d.skv);
      fp8(8192, 1280, d.wqb, d.sqb);
      fp8(1024, 1280, d.wix, d.six);
      fp8(2048, 4096, d.woa, d.soa);
      fp8(kD, 2048, d.wob, d.sob);
      d.gate = dev_bytes((size_t)96 * kD * 2, 1, 0x3c);
      d.n_attn = reinterpret_cast<uint16_t*>(dev_bytes(kD * 2, 0, 0x3f));
      d.n_q = reinterpret_cast<uint16_t*>(dev_bytes(1280 * 2, 0, 0x3f));
      d.n_ffn = reinterpret_cast<uint16_t*>(dev_bytes(kD * 2, 0, 0x3f));
      dw.push_back(d);
    }
    const size_t w13 = (size_t)576 * kD / 2, s13 = (size_t)576 * kD / 32, w2 = (size_t)kD * 576 / 2,
                 s2 = (size_t)kD * 576 / 32;
    dsk::MoeWeights base{};
    base.w1 = dev_bytes(kNE * w13, 0xFF, 0), base.s1 = dev_bytes(kNE * s13, 7, 120);
    base.w3 = dev_bytes(kNE * w13, 0xFF, 0), base.s3 = dev_bytes(kNE * s13, 7, 120);
    base.w2 = dev_bytes(kNE * w2, 0xFF, 0), base.s2 = dev_bytes(kNE * s2, 7, 120);
    for (int c = 0; c < kCopies; ++c) {
      dsk::MoeWeights m = base;
      m.sh_w1 = dev_bytes((size_t)576 * kD, 0x77, 0), m.sh_s1 = dev_bytes(18 * 160, 0x0F, 118);
      m.sh_w3 = dev_bytes((size_t)576 * kD, 0x77, 0), m.sh_s3 = dev_bytes(18 * 160, 0x0F, 118);
      m.sh_w2 = dev_bytes((size_t)kD * 576, 0x77, 0), m.sh_s2 = dev_bytes(160 * 18, 0x0F, 118);
      mw.push_back(m);
    }
    head = dev_bytes((size_t)32320 * kD * 2, 1, 0x3c);
    n_out = reinterpret_cast<uint16_t*>(dev_bytes(kD * 2, 0, 0x3f));
    bias = reinterpret_cast<float*>(dev_bytes(384 * 4, 0, 0));
    CUDA_CHECK(cudaMalloc(&moe_ws, dsk::moe_workspace_bytes()));
    CUDA_CHECK(cudaMemset(moe_ws, 0, dsk::moe_workspace_bytes()));
    idx = Dev<int>(kL * kPicks);
    idx_dummy = Dev<int>(kPicks);
    wts = Dev<float>(kL * kPicks);
    fill::f32_range(wts, 0.2f, 0.3f);
  }

  void setup_attn() {
    const int maxpos = ctx_max + 16;
    tab_plain = Dev<float2>((size_t)maxpos * 32);
    tab_yarn = Dev<float2>((size_t)maxpos * 32);
    attn::rope_table_build(tab_plain.p, maxpos, false, 0);
    attn::rope_table_build(tab_yarn.p, maxpos, true, 0);
    for (auto* d : {&stream[0], &stream[1]}) *d = Dev<uint16_t>((size_t)kRows * kHcDim);
    fill::bf16(stream[0], 1.f);
    fill::bf16(stream[1], 1.f);
    for (auto* d : {&sub, &y, &xn, &ob}) *d = Dev<uint16_t>((size_t)kRows * kDim);
    xg = Dev<uint16_t>((size_t)kRows * 8192);
    qa_loc = Dev<uint16_t>(kRows * 320);
    kv_loc = Dev<uint16_t>(kRows * 128);
    qa = Dev<uint16_t>(kRows * attn::kQLora);
    kvp = Dev<uint16_t>(kRows * attn::kHead);
    qr = Dev<uint16_t>(kRows * attn::kQLora);
    qb = Dev<uint16_t>((size_t)kRows * 8192);
    ix_loc = Dev<uint16_t>(kRows * 1024);
    iq = Dev<uint16_t>(kRows * 4096);
    o = Dev<uint16_t>((size_t)kRows * 8192);
    for (auto* d : {&sub, &xg, &qa, &kvp, &qb, &iq, &o}) fill::bf16(*d, 1.f);
    cproj_loc = Dev<float>(kRows * 128);
    cscore_loc = Dev<float>(kRows * 128);
    cproj = Dev<float>(kRows * attn::kHead);
    cscore = Dev<float>(kRows * attn::kHead);
    fill::f32(cproj, 1.f);
    fill::f32(cscore, 1.f);
    chunks = Dev<float>((size_t)attn::hc_chunks(0, kHcDim) * kRows * 25);
    part = Dev<float>(kRows * 25);
    pre = Dev<float>(kRows * 4);
    post = Dev<float>(kRows * 4);
    comb = Dev<float>(kRows * 16);
    fill::f32_range(post, 0.f, 2.f);
    fill::f32_range(comb, 0.f, 0.5f);
    wp = Dev<float>(kRows * 32);
    fill::f32(wp, 1.f);
    lg = Dev<float>(kRows * 96);
    lg_all = Dev<float>(kRows * 384);
    ymoe = Dev<float>((size_t)kRows * kDim);
    logits = Dev<float>((size_t)kRows * 32320);
    po = Dev<float>((size_t)kRows * 16 * attn::kHeads * attn::kHead);
    pml = Dev<float>((size_t)kRows * 16 * attn::kHeads * 2);
    xq = Dev<uint8_t>((size_t)kRows * kDim);
    xs = Dev<uint8_t>((size_t)kRows * kDim / 32);
    qq = Dev<uint8_t>(kRows * attn::kQLora);
    qs = Dev<uint8_t>(kRows * attn::kQLora / 32);
    oq = Dev<uint8_t>(kRows * 2048);
    os = Dev<uint8_t>(kRows * 2048 / 32);
    iqq = Dev<uint8_t>(kRows * 32 * 64);
    iqsc = Dev<uint32_t>(kRows * 32);
    iqw = Dev<float>(kRows * 32);
    cnt_idx = Dev<int>(attn::kMaxDecodeRows);
    fn_head = Dev<float>((size_t)attn::kMix * kHcDim);
    hsc_head = Dev<float>(3);
    hbase_head = Dev<float>(attn::kMix);
    fill::f32(fn_head, 0.01f);
    fill::f32_range(hsc_head, 0.5f, 1.5f);
    fill::f32(hbase_head, 1.f);
    ly.resize(kL);
    for (int L = 0; L < kL; ++L) {
      AttnLayer& l = ly[L];
      l.ratio = ratio_of(L);
      l.ring_kv = Dev<uint8_t>((size_t)g_nslots * attn::kRing * 512);
      l.ring_sc = Dev<uint8_t>((size_t)g_nslots * attn::kRing * 16);
      fill::u8(l.ring_kv, 0x77, 0);
      fill::u8(l.ring_sc, 7, 120);
      l.qnw = Dev<uint16_t>(attn::kQLora);
      l.kvw = Dev<uint16_t>(512);
      fill::bf16_range(l.qnw, 0.5f, 1.5f);
      fill::bf16_range(l.kvw, 0.5f, 1.5f);
      l.sink = Dev<float>(16);
      fill::f32(l.sink, 1.f);
      for (int h = 0; h < 2; ++h) {
        l.hnorm[h] = Dev<uint16_t>(kDim);
        l.fn[h] = Dev<float>((size_t)attn::kMix * kHcDim);
        l.hsc[h] = Dev<float>(3);
        l.hbase[h] = Dev<float>(attn::kMix);
        fill::bf16_range(l.hnorm[h], 0.5f, 1.5f);
        fill::f32(l.fn[h], 0.01f);
        fill::f32_range(l.hsc[h], 0.5f, 1.5f);
        fill::f32(l.hbase[h], 1.f);
      }
      if (kv_src(L)) {  // this rank's 128 output rows of the compressor projections
        l.cap = ctx_max / l.ratio + 64;
        l.wkv = Dev<uint16_t>((size_t)128 * kDim);
        fill::bf16(l.wkv, 0.02f);
        if (l.ratio > 1) {
          l.wgate = Dev<uint16_t>((size_t)128 * kDim);
          fill::bf16(l.wgate, 0.02f);
        }
        l.cnorm = Dev<uint16_t>(512);
        l.wk = Dev<uint16_t>((size_t)128 * 512);
        l.knorm = Dev<uint16_t>(128);
        fill::bf16_range(l.cnorm, 0.5f, 1.5f);
        fill::bf16(l.wk, 0.05f);
        fill::bf16_range(l.knorm, 0.5f, 1.5f);
        l.carry_kv = Dev<float>((size_t)g_nslots * 16 * 512);
        l.carry_sc = Dev<float>((size_t)g_nslots * 16 * 512);
        l.ckv_kv = Dev<uint8_t>((size_t)g_nslots * l.cap * 256);
        l.ckv_sc = Dev<uint8_t>((size_t)g_nslots * l.cap * 32);
        l.ik_k = Dev<uint8_t>((size_t)g_nslots * l.cap * 64);
        l.ik_sc = Dev<uint8_t>((size_t)g_nslots * l.cap * 4);
        fill::u8(l.ckv_kv, 0xFF, 0);
        fill::u8(l.ckv_sc, 0x1F, 0x28);
        fill::u8(l.ik_k, 0xFF, 0);
        fill::u8(l.ik_sc, 7, 120);
      }
      if (idx_src(L)) {
        l.wproj = Dev<uint16_t>((size_t)32 * kDim);
        fill::bf16(l.wproj, 0.02f);
        int n = ctx_max / l.ratio, pmax = L >= 24 ? kPool * 8 : ((n + kParts - 1) / kParts + 63) / 64 * 64;
        l.kstride = (pmax + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
        l.keys = Dev<uint16_t>((size_t)kRows * l.kstride);
        if (L == 20) l.bkeys = Dev<uint16_t>((size_t)kRows * l.kstride / 8);
        l.cand = Dev<int>((size_t)kRows * kParts * kTopk);
        l.cand_key = Dev<uint16_t>((size_t)kRows * kParts * kTopk);
        l.top = Dev<int>((size_t)kRows * kTopk);
        l.topkey = Dev<uint16_t>((size_t)kRows * kTopk);
        if (L == 20) {
          l.pcand = Dev<int>((size_t)kRows * kParts * kPool);
          l.pcand_key = Dev<uint16_t>((size_t)kRows * kParts * kPool);
          l.pool = Dev<int>((size_t)kRows * kPool);
          l.poolkey = Dev<uint16_t>((size_t)kRows * kPool);
        }
      }
    }
  }

  // positions, slots, MoE picks, scan lengths and the other ranks' index candidates for a context and row count
  void configure(int ctx_, int M_) {
    ctx = ctx_;
    M = M_;
    nu = M == 1 ? 6 : 27;
    const int S = g_sessions;  // session i: rows [first[i], first[i + 1]) (main checks the sizes)
    std::vector<int> ph(M), first(S + 1);
    for (int i = 0; i <= S; ++i) first[i] = i * (M / S) + std::min(i, M % S);
    hslot.assign(M, 0);
    for (int i = 0; i < S; ++i)
      for (int r = first[i]; r < first[i + 1]; ++r) ph[r] = ctx - (first[i + 1] - r), hslot[r] = S - 1 - i;
    pos = Dev<int>(ph.data(), M);
    slot = Dev<int>(hslot.data(), M);
    g_slot = g_table || S > 1 ? slot.p : nullptr;
    std::vector<int> h(kL * kPicks);
    for (int l = 0; l < kL; ++l)
      for (int m = 0; m < M; ++m)
        for (int k = 0; k < 6; ++k) h[l * kPicks + m * 6 + k] = (l * nu + (m * 6 + k) % nu) % kNE;
    CUDA_CHECK(cudaMemcpy(idx.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    for (int L = 0; L < kL; ++L)
      if (idx_src(L)) ly[L].pmax = L >= 24 ? kPool * 8 : ((ctx / ly[L].ratio + kParts - 1) / kParts + 63) / 64 * 64;
    for (int L : {2, 8, 14, 20}) {  // what the all-gather of the local candidates would deliver from ranks 1-3
      AttnLayer& l = ly[L];
      attn::IndexQ q{iqq.p, iqsc.p, iqw.p};
      attn::index_q_prep(iq.p, wp.p, pos.p, tab_yarn.p, q, nullptr, M, s);
      for (int part = 1; part < kParts; ++part) {
        attn::IndexScan sc;
        sc.part = part;
        sc.nparts = kParts;
        for (int i = 0; i < S; ++i) {  // a session at a time, on its slot's keys
          const int r0 = first[i], rs = first[i + 1] - r0;
          const attn::IndexQ qs{q.q + (size_t)r0 * 32 * 64, q.sc + r0 * 32, q.w + r0 * 32};
          uint16_t* keys = l.keys.p + (size_t)r0 * l.kstride;
          uint16_t* bkeys = L == 20 ? l.bkeys.p + (size_t)r0 * l.kstride / 8 : nullptr;
          attn::index_score(qs, pos.p + r0, rs, l.ratio, l.ik_one(hslot[r0]), sc, l.pmax, keys, l.kstride, bkeys,
                            l.kstride / 8, s);
          attn::index_topk(keys, l.kstride, pos.p + r0, rs, l.ratio, sc, l.pmax, false, kTopk,
                           l.cand.p + ((size_t)r0 * kParts + part) * kTopk,
                           l.cand_key.p + ((size_t)r0 * kParts + part) * kTopk, kParts * kTopk, s);
          if (L == 20)
            attn::index_topk(bkeys, l.kstride / 8, pos.p + r0, rs, l.ratio, sc, l.pmax, true, kPool,
                             l.pcand.p + ((size_t)r0 * kParts + part) * kPool,
                             l.pcand_key.p + ((size_t)r0 * kParts + part) * kPool, kParts * kPool, s);
        }
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
  }

  const float2* table(int L) const { return L < 2 ? tab_plain.p : tab_yarn.p; }

  void gather(std::initializer_list<Slice> t) {  // STAND-IN: all-gather
    Slices a{};
    for (const Slice& sl : t) a.t[a.n++] = sl;
    attn::launch(standin_gather, dim3(8), dim3(256), 0, s, true, a, M);
  }

  // mHC around a sublayer: hc_post of the previous sublayer's output into the stream (first: none; the layer
  // runner fuses it with the all-reduce before it), then hc pre: this sublayer's mixes, pre-mix, norm and
  // act_quant into y / xq / xs
  void mhc(const Variant& v, const float* fn, const float* sc, const float* base, const uint16_t* norm, bool first) {
    int k0 = 0, k1 = v.hc_split ? kDim : kHcDim, nch = attn::hc_chunks(k0, k1);
    if (!first) {
      attn::hc_post(sub.p, stream[cur].p, post.p, comb.p, M, stream[cur ^ 1].p, s, v.fused);
      cur ^= 1;
    }
    const uint16_t* x = stream[cur].p;
    if (v.fused) {
      attn::hc_project(x, M, fn, k0, k1, chunks.p, true, s);
      const float* mixes = chunks.p;
      if (v.hc_split && v.standins) {  // STAND-IN: all-reduce of the 25 mix partials
        attn::launch(standin_mix, dim3(1), dim3(256), 0, s, true, (const float*)chunks.p, nch, M, part.p);
        mixes = part.p;
        nch = 1;
      }
      attn::hc_mix(x, mixes, nch, M, sc, base, norm, pre.p, post.p, comb.p, y.p, xq.p, xs.p, true, s);
    } else {
      attn::hc_partial(x, M, fn, k0, k1, chunks.p, part.p, s);
      attn::hc_finish(part.p, M, sc, base, pre.p, post.p, comb.p, s);
      attn::hc_pre(x, pre.p, M, norm, y.p, nullptr, s);
      dsk::act_quant(bf(y.p), M, kDim, xq.p, xs.p, s);
    }
  }

  void prep(const Variant& v, int L) {
    AttnLayer& l = ly[L];
    if (v.fused) {
      attn::DecodePrep a;
      a.pos = pos.p;
      a.table = table(L);
      a.qa = qa.p;
      a.q_norm = l.qnw.p;
      a.qq = qq.p;
      a.qs = qs.p;
      a.kv_pre = kvp.p;
      a.kv_norm = l.kvw.p;
      a.win = l.win();
      if (kv_src(L)) {
        a.ckv_proj = cproj.p;
        a.cscore = cscore.p;
        a.cw = l.cw();
        a.carry = l.carry();
        a.ckv = l.ckv();
        a.ik = l.ik();
      }
      if (idx_src(L)) {
        a.x = y.p;
        a.wproj = l.wproj.p;
        a.wp = wp.p;
      }
      attn::decode_prep(a, M, true, s);
    } else {
      attn::q_norm(qa.p, l.qnw.p, qr.p, M, s);
      dsk::act_quant(bf(qr.p), M, attn::kQLora, qq.p, qs.p, s);
      attn::kv_write(kvp.p, l.kvw.p, pos.p, table(L), l.win(), nullptr, M, s);
      if (kv_src(L))
        attn::compressor_emit(cproj.p, cscore.p, pos.p, M, l.cw(), l.carry(), tab_yarn.p,
                              l.ckv(), l.ik(), nullptr, 0, s);
      if (idx_src(L)) attn::gemv_bf16(y.p, M, kDim, l.wproj.p, nullptr, 32, 32, wp.p, nullptr, true, s);
    }
  }

  void index(const Variant& v, int L) {
    AttnLayer& l = ly[L];
    bool pool = L == 20, merged = L < 24;
    attn::IndexScan sc;
    if (L >= 24) {
      sc.blocks = ly[20].pool.p;
      sc.nblocks = kPool;
    } else {
      sc.part = 0;
      sc.nparts = kParts;
    }
    const AttnLayer& src = ly[kv_source(L)];
    attn::IndexSel top = merged ? attn::IndexSel{kTopk, l.cand.p, l.cand_key.p, kParts * kTopk}
                                : attn::IndexSel{kTopk, l.top.p, l.topkey.p, kTopk};
    attn::IndexSel ps = pool ? attn::IndexSel{kPool, l.pcand.p, l.pcand_key.p, kParts * kPool} : attn::IndexSel{};
    if (v.fused) {
      attn::index_fused(iq.p, wp.p, pos.p, tab_yarn.p, M, l.ratio, src.ik(), sc, l.pmax, l.keys.p, l.kstride,
                        pool ? l.bkeys.p : nullptr, l.kstride / 8, top, ps, cnt_idx.p, true, s);
    } else {
      attn::IndexQ q{iqq.p, iqsc.p, iqw.p};
      attn::index_q_prep(iq.p, wp.p, pos.p, tab_yarn.p, q, nullptr, M, s);
      attn::index_score(q, pos.p, M, l.ratio, src.ik_one(), sc, l.pmax, l.keys.p, l.kstride, pool ? l.bkeys.p : nullptr,
                        l.kstride / 8, s);
      attn::index_topk(l.keys.p, l.kstride, pos.p, M, l.ratio, sc, l.pmax, false, kTopk, top.pos, top.key, top.stride,
                       s);
      if (pool)
        attn::index_topk(l.bkeys.p, l.kstride / 8, pos.p, M, l.ratio, sc, l.pmax, true, kPool, ps.pos, ps.key,
                         ps.stride, s);
    }
    if (!merged) return;
    if (v.standins) gather({});  // STAND-IN: all-gather of the local candidates (slot 0 is written in place)
    attn::IndexSel ptop{kTopk, l.top.p, l.topkey.p, kTopk}, ppool{};
    if (pool) ppool = attn::IndexSel{kPool, l.pool.p, l.poolkey.p, kPool};
    if (v.fused) {
      attn::index_merge2(M, l.cand.p, l.cand_key.p, kParts * kTopk, ptop, l.pcand.p, l.pcand_key.p, kParts * kPool,
                         ppool, true, s);
    } else {
      attn::index_merge(l.cand.p, l.cand_key.p, kParts * kTopk, M, kTopk, l.top.p, l.topkey.p, kTopk, s);
      if (pool) attn::index_merge(l.pcand.p, l.pcand_key.p, kParts * kPool, M, kPool, l.pool.p, l.poolkey.p, kPool, s);
    }
  }

  // A row's 64-key blocks (2 window + kmax/64) are split over CTAs, one block each while rows x blocks fits
  // the 48 SMs, else as many blocks per CTA as it takes (measured best for 1 and 6 rows).
  void attention(int L, bool pdl) {
    AttnLayer& l = ly[L];
    int kmax = L < 2 ? 0 : kTopk, nblocks = 2 + kmax / 64, per = (M * nblocks + 47) / 48;
    const int* topk = L < 2 ? nullptr : ly[index_source(L)].top.p;
    attn::Compressed ckv = L < 2 ? attn::Compressed{nullptr, nullptr, 0} : ly[kv_source(L)].ckv();
    attn::sparse_attn(qb.p, pos.p, {}, M, table(L), l.win(), ckv, topk, kmax, l.sink.p, o.p,
                      g_splits ? g_splits : (nblocks + per - 1) / per, attn::AttnScratch{po.p, pml.p, 16}, s, pdl);
  }

  // every layer's sparse attention back to back (--attn-micro)
  void attn_only(bool pdl) {
    for (int L = 0; L < kL; ++L) attention(L, pdl);
  }

  // layers [l0, l1), then the head when asked
  void step(const Variant& v, int l0, int l1, bool with_head) {
    const int kD = kDim;
    dsk::GemvCfg c8{1, 1, 16}, c8k{1, 1, 8}, cw{1, 4, 4}, cg{1, 1, 8}, ch{2, 1, 16};
    c8.tiled = c8k.tiled = true;
    dsk::MoeCfg mc{};
    mc.down = M == 1 ? 2 : 4;
    cur = 0;
    for (int L = l0; L < l1; ++L) {
      AttnLayer& l = ly[L];
      const DenseW& d = dw[L % kCopies];
      // ---- attention sublayer
      if (v.attn)
        mhc(v, l.fn[0].p, l.hsc[0].p, l.hbase[0].p, l.hnorm[0].p, L == 0);
      else
        dsk::rmsnorm(bf(xg.p), bf(d.n_attn), M, kD, kEps, bf(xn.p), xq.p, xs.p, s);
      if (v.gemv) {
        dsk::Fp8Seg seg[2] = {{d.wqa, d.sqa, bf(qa_loc.p), 320}, {d.wkv, d.skv, bf(kv_loc.p), 128}};
        dsk::gemv_fp8_multi(xq.p, xs.p, M, seg, 2, kD, c8k, s);
      }
      if (v.attn && kv_src(L))
        attn::compressor_project_rows(y.p, l.cw(), 0, 128, cproj_loc.p, cscore_loc.p, M, v.fused, s);
      if (v.standins) {  // STAND-IN: all-gather of wq_a, wkv and the compressor projections
        if (kv_src(L) && l.ratio > 1)
          gather({{qa_loc.p, qa.p, 640, 4}, {kv_loc.p, kvp.p, 256, 4}, {cproj_loc.p, cproj.p, 512, 4},
                  {cscore_loc.p, cscore.p, 512, 4}});
        else if (kv_src(L))
          gather({{qa_loc.p, qa.p, 640, 4}, {kv_loc.p, kvp.p, 256, 4}, {cproj_loc.p, cproj.p, 512, 4}});
        else
          gather({{qa_loc.p, qa.p, 640, 4}, {kv_loc.p, kvp.p, 256, 4}});
      }
      if (v.attn)
        prep(v, L);
      else
        dsk::rmsnorm(bf(qa.p), bf(d.n_q), M, attn::kQLora, kEps, bf(qr.p), qq.p, qs.p, s);
      if (v.gemv) {
        if (idx_src(L) && v.attn) {  // the indexer's wq_b as a second segment of wq_b's launch
          dsk::Fp8Seg seg[2] = {{d.wqb, d.sqb, bf(qb.p), 8192}, {d.wix, d.six, bf(ix_loc.p), 1024}};
          dsk::gemv_fp8_multi(qq.p, qs.p, M, seg, 2, attn::kQLora, c8k, s);
        } else {  // as engine/kernels' step
          dsk::gemv_fp8(qq.p, qs.p, M, d.wqb, d.sqb, 8192, attn::kQLora, bf(qb.p), c8k, s);
          if (idx_src(L)) dsk::gemv_fp8(qq.p, qs.p, M, d.wix, d.six, 1024, attn::kQLora, bf(ix_loc.p), c8k, s);
        }
      }
      if (v.attn) {
        if (idx_src(L)) {
          if (v.standins) gather({{ix_loc.p, iq.p, 2048, 4}});  // STAND-IN: all-gather of the indexer queries
          index(v, L);
        }
        attention(L, v.fused);
      }
      if (v.gemv) {
        dsk::gemv_woa(bf(o.p), M, d.woa, d.soa, 2048, 4096, 1024, nullptr, cw, false, s, oq.p, os.p);
        dsk::gemv_fp8(oq.p, os.p, M, d.wob, d.sob, kD, 2048, bf(ob.p), c8, s);
      }
      if (v.standins) gather({{ob.p, sub.p, kD * 2, 1}});  // STAND-IN: all-reduce of wo_b's partial
      // ---- FFN sublayer
      if (v.attn) mhc(v, l.fn[1].p, l.hsc[1].p, l.hbase[1].p, l.hnorm[1].p, false);
      if (v.gemv) {
        if (v.attn)
          dsk::gemv_bf16(bf(y.p), M, d.gate, 96, kD, lg.p, nullptr, cg, false, s);
        else
          dsk::gemv_bf16_norm(bf(xg.p), bf(d.n_ffn), kEps, M, d.gate, 96, kD, lg.p, nullptr, cg, s, xq.p, xs.p);
        if (v.standins) gather({{lg.p, lg_all.p, 96 * 4, 4}});  // STAND-IN: all-gather of the gate logits
        dsk::router_topk(lg_all.p, bias, M, 384, idx_dummy.p, wts.p + L * kPicks, 1.5f, s);
        dsk::moe(xq.p, xs.p, M, idx.p + L * kPicks, wts.p + L * kPicks, mw[L % kCopies], moe_ws, ymoe.p, mc, s);
      }
      if (v.standins)  // STAND-IN: all-reduce of the MoE's partial
        attn::launch(standin_reduce, dim3(40), dim3(256), 0, s, true, (const float*)ymoe.p, sub.p, M * kD);
    }
    if (!with_head) return;
    if (v.attn)  // the last FFN's hc_post and the output mix (as a sublayer's) with the final norm
      mhc(v, fn_head.p, hsc_head.p, hbase_head.p, n_out, false);
    else
      dsk::rmsnorm(bf(xg.p), bf(n_out), M, kD, kEps, bf(y.p), nullptr, nullptr, s);
    if (v.gemv) dsk::gemv_bf16(bf(y.p), M, head, 32320, kD, logits.p, nullptr, ch, false, s);
  }
};

int graph_kernels(Bench& b, const Variant& v, int l0, int l1, bool head) {
  cudaGraph_t g;
  CUDA_CHECK(cudaStreamBeginCapture(b.s, cudaStreamCaptureModeThreadLocal));
  b.step(v, l0, l1, head);
  CUDA_CHECK(cudaStreamEndCapture(b.s, &g));
  size_t n = 0;
  CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
  CUDA_CHECK(cudaGraphDestroy(g));
  return (int)n;
}

// replay times (us) of each function's graph, round-robin, sorted
std::vector<std::vector<double>> time_interleaved(cudaStream_t st, const std::vector<std::function<void()>>& fns,
                                                  int rounds) {
  std::vector<cudaGraphExec_t> execs;
  for (auto& fn : fns) {
    fn();  // warm-up, eagerly (kernel attributes)
    CUDA_CHECK(cudaStreamSynchronize(st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    fn();
    CUDA_CHECK(cudaStreamEndCapture(st, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    execs.push_back(exec);
  }
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<std::vector<double>> ts(fns.size());
  for (int r = 0; r < rounds + 1; ++r)
    for (size_t i = 0; i < execs.size(); ++i) {
      CUDA_CHECK(cudaEventRecord(a, st));
      CUDA_CHECK(cudaGraphLaunch(execs[i], st));
      CUDA_CHECK(cudaEventRecord(b, st));
      CUDA_CHECK(cudaEventSynchronize(b));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      if (r) ts[i].push_back(ms * 1000.0);  // round 0: first replays
    }
  for (auto& v : ts) std::sort(v.begin(), v.end());
  for (auto e : execs) cudaGraphExecDestroy(e);
  cudaEventDestroy(a), cudaEventDestroy(b);
  return ts;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i <= s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) out.push_back(s.substr(i, j - i));
    i = j + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> rows_list = {1, 6}, ctxs;
  std::vector<std::string> names = {"gemv", "fused", "fused-hcrep", "fused-nostandin", "attn", "attn-separate"};
  int rounds = 15;
  bool counts = false, micro = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--rows" && i + 1 < argc) {
      rows_list.clear();
      for (auto& t : split(argv[++i])) rows_list.push_back(std::atoi(t.c_str()));
    } else if (a == "--rounds" && i + 1 < argc) {
      rounds = std::atoi(argv[++i]);
    } else if (a == "--variants" && i + 1 < argc) {
      names = split(argv[++i]);
    } else if (a == "--counts") {
      counts = true;
    } else if (a == "--splits" && i + 1 < argc) {
      g_splits = std::atoi(argv[++i]);
    } else if (a == "--attn-micro") {
      micro = true;
    } else if (a == "--table") {
      g_table = true;
    } else if (a == "--sessions" && i + 1 < argc) {
      g_nslots = std::atoi(argv[++i]);
    } else {
      ctxs.push_back(std::atoi(a.c_str()));
    }
  }
  if (ctxs.empty()) ctxs = {1024, 16384, 131072};
  std::vector<const Variant*> vars;
  for (auto& n : names)
    for (auto& v : kVariants)
      if (n == v.name) vars.push_back(&v);
  g_sessions = g_nslots;
  const bool table = g_table || g_sessions > 1;
  const bool separate = std::find(names.begin(), names.end(), "attn-separate") != names.end();
  for (int M : rows_list)  // a session's rows: at most kDecodeGroup with a table (attn.h)
    if (g_sessions < 1 || M < g_sessions || M > kRows ||
        (table && (M + g_sessions - 1) / g_sessions > attn::kDecodeGroup) || (g_sessions > 1 && separate))
      die("bench_step: %d rows in %d sessions%s", M, g_sessions, g_table ? " (--table)" : "");
  Bench b(*std::max_element(ctxs.begin(), ctxs.end()));
  if (counts) {  // kernels per layer type, and per step
    b.configure(ctxs[0], 1);
    for (auto* v : vars) b.step(*v, 0, kL, true);  // eagerly first (kernel attributes)
    CUDA_CHECK(cudaStreamSynchronize(b.s));
    std::printf("kernels per layer (graph nodes): ");
    for (int L : {0, 3, 2, 20, 21, 24}) std::printf(" L%d", L);
    std::printf("   step(40 layers + head)\n");
    for (auto* v : vars) {
      std::printf("  %-16s", v->name);
      for (int L : {0, 3, 2, 20, 21, 24}) std::printf(" %3d", graph_kernels(b, *v, L, L + 1, false));
      std::printf("   %d\n", graph_kernels(b, *v, 0, kL, true));
    }
  }
  if (micro) {  // the 40 layers' sparse attention alone, PDL launches or not
    for (int M : rows_list)
      for (int ctx : ctxs) {
        b.configure(ctx, M);
        b.step(kVariants[1], 0, kL, true);  // the layers' top-k lists
        std::vector<std::function<void()>> fns;
        for (bool pdl : {true, false}) fns.push_back([&b, pdl] { b.attn_only(pdl); });
        auto ts = time_interleaved(b.s, fns, rounds);
        auto med = [&](int i) { return ts[i][ts[i].size() / 2]; };
        std::printf("sparse attention x40, rows %d ctx %6d: pdl %.0f  plain %.0f us\n", M, ctx, med(0), med(1));
      }
    return 0;
  }
  for (int M : rows_list)
    for (int ctx : ctxs) {
      b.configure(ctx, M);
      std::vector<std::function<void()>> fns;
      for (auto* v : vars) fns.push_back([&b, v] { b.step(*v, 0, kL, true); });
      auto ts = time_interleaved(b.s, fns, rounds);
      std::printf("rows %d ctx %6d:", M, ctx);
      for (size_t i = 0; i < vars.size(); ++i)
        std::printf("  %s %.0f (%.0f-%.0f)", vars[i]->name, ts[i][ts[i].size() / 2], ts[i][0], ts[i].back());
      std::printf("  us\n");
      std::fflush(stdout);
    }
  return 0;
}
