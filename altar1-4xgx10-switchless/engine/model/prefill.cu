// Prefill (docs/engine.md, "Prefill"): a prompt's positions in chunks of up to Options::prefill_rows rows, through
// the same layers as the decode step with the kernels' prefill paths: the FP8 projections as attn's tensor-core
// gemm() over the whole chunk (kTunePfGemm, on by default: each output equals the 16-row GEMV's bit for bit, attn's
// test P; off: the 16-row GEMV launches), the BF16 router as attn's gemm_bf16() (kTunePfRouter; the BF16 GEMV's
// bits), the head (only for kFlagLogits) as 16-row GEMVs, attn_prep, index_select (its prefill path: staged queries,
// the merged score-and-select launches) and sparse_attn over the whole chunk, the routed and shared experts' GEMMs
// (moe_prefill). A sublayer ends in a reduce-scatter by rows of the f32 partials
// (the all-reduce's sums, in rank order), the residual add and RMSNorm on this rank's quarter of the rows (attn's
// rmsnorm: the fused all-reduce's bits), and an all-gather of the quarter's residual and normed rows. So every
// value the decode step computes row by row is computed here with the same bits, except the routed experts'
// (prefill sums their K in another order, experts.md § 6). The MTP layer's cache is filled for the chunk's rows
// (eh_proj, its input norm, its x-projection and attn_prep: no attention, no MoE). With kTunePfOverlap (default) the
// layers run on two pieces of the chunk's rows, a piece's collectives on a second stream while the other piece
// computes (prefill_chunk below); the same bits.
#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>

#include "comm/collectives.h"
#include "model/impl.h"

namespace glm::model {

using namespace detail;
using ::glm::GemvOut;
using ::glm::GemvSeg;

struct Model::Impl::Pre {
  int cap = 0;  // rows a chunk (a multiple of 16)
  int32_t *tok = nullptr, *pos = nullptr, *slot = nullptr, *mpos = nullptr, *mslot = nullptr;
  uint16_t *emb, *eb, *h, *xn, *xg, *qr, *q, *iq, *qlat, *u, *o, *hm;
  float *iw, *part, *yr, *ys, *qa, *qb, *gl, *rw;
  int *ridx, *topk;
  float* scores = nullptr;
  uint32_t* work = nullptr;
  int idx_rows = 0;
  void* ws = nullptr;
  // kTunePfOverlap: the collectives' stream (the highest priority, so that a collective's waiting blocks get the next
  // free SMs) and a ring of events for the dependencies between it and st
  cudaStream_t cs = nullptr;
  static constexpr int kEv = 256;
  cudaEvent_t ev[kEv] = {};
  unsigned evi = 0;
};
// its buffers live as long as the process (one model a process)
void Model::Impl::PreFree::operator()(Pre* p) const { delete p; }

namespace {
constexpr int kEmbSub = 1024;  // rows of an embedding all-gather
constexpr int kMaxPieces = 8;
}

void Model::Impl::prefill_alloc() {
  if (pre) return;
  using namespace ::glm;
  pre.reset(new Pre());
  Pre& p = *pre;
  const size_t M = p.cap = opt.prefill_rows;
  p.tok = dalloc<int32_t>(M), p.pos = dalloc<int32_t>(M), p.slot = dalloc<int32_t>(M);
  p.mpos = dalloc<int32_t>(M), p.mslot = dalloc<int32_t>(M);
  p.emb = dalloc<uint16_t>(M * kDim);
  p.eb = dalloc<uint16_t>((size_t)kEmbSub * kRanks * kDim);
  p.h = dalloc<uint16_t>(M * kDim), p.xn = dalloc<uint16_t>(M * kDim);
  p.xg = dalloc<uint16_t>(M * kRanks * kXRows);
  p.iw = dalloc<float>(M * kIdxHeads);
  p.qr = dalloc<uint16_t>(M * kQLora);
  p.q = dalloc<uint16_t>(M * kOProjK);
  p.iq = dalloc<uint16_t>(M * kIdxHeads * kIdxDim);
  p.qlat = dalloc<uint16_t>(M * kHeads * kKvLora);  // the MTP fill's eh_proj output too
  p.u = dalloc<uint16_t>(M * kHeads * kKvLora);     // and its normed input
  p.o = dalloc<uint16_t>(M * kOProjK);
  p.hm = dalloc<uint16_t>(M * kDenseInterRank);
  p.part = dalloc<float>(M * kDim);  // the MTP fill's [M][12288] bf16 input too
  p.yr = dalloc<float>(M * kDim), p.ys = dalloc<float>(M * kDim);
  p.qa = dalloc<float>(M / kRanks * kDim), p.qb = dalloc<float>(M / kRanks * kDim);
  p.gl = dalloc<float>(M * gx::kE);
  p.rw = dalloc<float>(M * gx::kTopK);
  p.ridx = dalloc<int>(M * gx::kTopK);
  p.topk = dalloc<int>(M * kTopK);
  p.idx_rows = (int)std::clamp<size_t>(opt.index_scratch / (idx_ld * 4), 16, M);
  p.scores = dalloc<float>((size_t)p.idx_rows * idx_ld);
  p.work = dalloc<uint32_t>(index_scratch_words(p.idx_rows));
  p.ws = dalloc<uint8_t>(gx::prefill_ws_bytes((int)M));
  int lo = 0, hi = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&p.cs, cudaStreamNonBlocking, hi));
  for (auto& e : p.ev) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
}

namespace {
// y rows [0, M) of one projection, 16 rows a launch (the GEMVs serve 1-16 rows)
template <class F>
void rows16(int M, F f) {
  for (int r0 = 0; r0 < M; r0 += kMaxRows) f(r0, std::min(kMaxRows, M - r0));
}
}  // namespace

// impl.h
void detail::pf_proj(bool use_gemm, const uint16_t* x, int ldx, int M, int K, const GemvSeg* segs, int nseg, GemvOut out,
             int splitk, cudaStream_t st, bool pdl) {
  if (use_gemm && M > kMaxRows) {
    // glm53's binding is FP8 throughout; attn's gemm() now takes each segment's format (altar1: the runner passes
    // BF16/INT4 and a scratch for INT4, docs/attention.md § 7)
    GemvSeg s[::glm::kMaxSegs];
    for (int i = 0; i < nseg; ++i) s[i] = segs[i], s[i].fmt = s[i].fmt == ::glm::WFmt::kNone ? ::glm::WFmt::kFP8 : s[i].fmt;
    ::glm::gemm(x, ldx, M, K, s, nseg, out, splitk, st, pdl);
    return;
  }
  const size_t es = out == GemvOut::kF32 ? 4 : 2;
  for (int r0 = 0; r0 < M; r0 += kMaxRows) {
    GemvSeg s[::glm::kMaxSegs];
    for (int i = 0; i < nseg; ++i) s[i] = segs[i], s[i].y = static_cast<char*>(segs[i].y) + (size_t)r0 * segs[i].ldy * es;
    ::glm::gemv(true, x + (size_t)r0 * ldx, ldx, std::min(kMaxRows, M - r0), K, s, nseg, out, splitk, st, pdl);
  }
}

// A sublayer's end over rows [r0, r0 + m) of a chunk (r0 a multiple of 4, mp = m rounded up to 4): the partials a
// (and b) reduce-scattered by rows, the residual add and the next norm on this rank's quarter of the piece's rows,
// the quarter's h and xn rows all-gathered. A row's sums are the four ranks' partials added in rank order wherever
// the piece boundaries fall, and the add and the norm go row by row: any split of the chunk gives the same bits.
static void sublayer_end(const comm::DevChannel& ch, int rank, Model::Impl::Pre& p, int r0, int m, const float* a,
                         const float* b, const uint16_t* norm, cudaStream_t st, int blocks) {
  const int mp = (m + kRanks - 1) / kRanks * kRanks, rq = mp / kRanks;
  float* qa = p.qa + (size_t)(r0 / kRanks) * kDim;
  float* qb = p.qb + (size_t)(r0 / kRanks) * kDim;
  comm::reduce_scatter_rows_f32(ch, a + (size_t)r0 * kDim, kDim, qa, kDim, rq, kDim, st, blocks);
  if (b) comm::reduce_scatter_rows_f32(ch, b + (size_t)r0 * kDim, kDim, qb, kDim, rq, kDim, st, blocks);
  uint16_t* hq = p.h + ((size_t)r0 + (size_t)rank * rq) * kDim;
  uint16_t* xq = p.xn + ((size_t)r0 + (size_t)rank * rq) * kDim;
  ops::add_rows(hq, qa, b ? qb : nullptr, (size_t)rq * kDim, hq, st);
  ::glm::rmsnorm(hq, kDim, norm, ::glm::kNormEps, kDim, xq, kDim, rq, st);
  const size_t bytes = (size_t)rq * kEmbRow;
  const comm::AgSeg s[2] = {{hq, p.h + (size_t)r0 * kDim, bytes}, {xq, p.xn + (size_t)r0 * kDim, bytes}};
  comm::allgather_multi(ch, s, 2, st, blocks);
}

namespace {
struct Piece {
  int r0, m;
};
}  // namespace

// The layers of a chunk run piece by piece with kTunePfOverlap (docs/engine.md, "Prefill overlap"): each sublayer
// computes the pieces' rows in order on st while the sublayer ends of the pieces before run on p.cs, so the ring's
// time hides behind the compute. A row's arithmetic is the whole chunk's: the projections, the router, the indexer,
// the attention and the experts compute a row alike whatever rows share the launch (attn's test P, experts.h's
// moe_prefill), a piece's attention reads the cache entries its own and the earlier pieces' attn_prep wrote, and
// sublayer_end's bits do not depend on the split. One ring channel: every collective goes to p.cs, in the same order
// on every rank. A piece's x-projection and indexer-query all-gathers are short and the compute waits for them, so a
// sublayer end (long) goes to p.cs only after the next piece's short ones ("deferred"); its wait for the compute it
// reduces is an event on p.cs. Off (one piece, cs = st): the unpipelined chunk's launches in their order.
void Model::Impl::prefill_chunk(int sl, int a, int M, const int32_t* tokens, bool last, uint32_t flags, Sink* sink) {
  using namespace ::glm;
  Pre& p = *pre;
  CUDA_CHECK(cudaMemcpyAsync(p.tok, tokens, (size_t)M * 4, cudaMemcpyHostToDevice, st));
  ops::iota(p.pos, p.slot, a, sl, M, st);
  // the embedding (vocabulary-split: each rank its own rows, all-gathered by sub-chunks), kept for the MTP fill
  for (int s0 = 0; s0 < M; s0 += kEmbSub) {
    const int n = std::min(kEmbSub, M - s0);
    ops::embed_local(embed, p.tok + s0, n, rank, p.eb, st, false);
    const size_t rb = kEmbRow, ld = rb * kRanks;
    const comm::glm::RowSeg g{p.eb + (size_t)rank * kDim, ld, p.eb, ld, rb};
    comm::glm::allgather_rows(ch, &g, 1, n, st);
    ops::embed_pick(p.eb, p.tok + s0, p.pos + s0, false, n, p.emb + (size_t)s0 * kDim, st, false);
  }
  CUDA_CHECK(cudaMemcpyAsync(p.h, p.emb, (size_t)M * kEmbRow, cudaMemcpyDeviceToDevice, st));
  rmsnorm(p.h, kDim, lw[0].attn_norm, kNormEps, kDim, p.xn, kDim, M, st);

  // the pieces: a multiple of 64 rows each (the last the rest), at least Options::prefill_min_piece
  std::vector<Piece> pcs;
  const int minp = std::max(64, opt.prefill_min_piece);
  if (tune(kTunePfOverlap) && M >= 2 * minp) {
    const int n = std::clamp(opt.prefill_pieces, 1, kMaxPieces);
    const int rows = (std::max(minp, (M + n - 1) / n) + 63) / 64 * 64;
    for (int r0 = 0; r0 < M; r0 += rows) pcs.push_back({r0, std::min(rows, M - r0)});
  } else {
    pcs.push_back({0, M});
  }
  const int P = (int)pcs.size();
  const bool ov = P > 1;
  const cudaStream_t cs = ov ? p.cs : st;
  const int cb = ov ? std::clamp(opt.prefill_comm_blocks, 1, 16) : 16;
  auto rec = [&](cudaStream_t s) {  // a fresh event recorded on s (a ring of them, each waited on long before reuse)
    const cudaEvent_t e = p.ev[p.evi++ % Pre::kEv];
    CUDA_CHECK(cudaEventRecord(e, s));
    return e;
  };
  auto after = [](cudaStream_t s, cudaEvent_t e) {
    if (e) CUDA_CHECK(cudaStreamWaitEvent(s, e, 0));
  };
  std::vector<cudaEvent_t> done(P, nullptr);    // piece j's last sublayer end on cs (null: nothing to wait for)
  std::vector<std::function<void()>> deferred;  // sublayer ends to queue after the next piece's short collectives
  auto flush = [&] {
    for (auto& f : deferred) f();
    deferred.clear();
  };
  // piece j's sublayer end, to run on cs after the compute queued on st so far
  auto end_of = [&](int j, const float* x, const float* y, const uint16_t* norm) -> std::function<void()> {
    const cudaEvent_t e = ov ? rec(st) : nullptr;
    return [=, this, &p, &done, &pcs, &rec, &after] {
      after(cs, e);
      sublayer_end(ch, rank, p, pcs[j].r0, pcs[j].m, x, y, norm, cs, cb);
      done[j] = ov ? rec(cs) : nullptr;
    };
  };
  // a short collective on cs, the compute waiting for it
  auto short_coll = [&](const std::function<void(cudaStream_t)>& f) {
    if (ov) after(cs, rec(st));
    f(cs);
    if (ov) after(st, rec(cs));
  };

  const bool G = tune(kTunePfGemm);
  for (int l = 0; l < kLayers; ++l) {
    const detail::LayerW& w = lw[l];
    const bool full = full_indexer(l);
    const int xr = full ? kXRows : kXRowsShared;
    const uint16_t* next_norm = l + 1 < kLayers ? lw[l + 1].attn_norm : norm_w;
    for (int j = 0; j < P; ++j) {  // attention
      const int r0 = pcs[j].r0, m = pcs[j].m;
      after(st, done[j]);  // the piece's rows out of the layer before
      uint16_t* xg = p.xg + (size_t)r0 * kRanks * xr;
      float* iw = p.iw + (size_t)r0 * kIdxHeads;
      uint16_t* iq = p.iq + (size_t)r0 * kIdxHeads * kIdxDim;
      {
        GemvSeg s;
        s.w = w.xproj.w, s.s = w.xproj.s, s.n = xr, s.rs = 16, s.ks = 128;
        s.y = xg + (size_t)rank * xr, s.ldy = kRanks * xr;
        pf_proj(G, p.xn + (size_t)r0 * kDim, kDim, m, kDim, &s, 1, GemvOut::kBF16, 4, st, false);
      }
      if (full) index_weights(p.xn + (size_t)r0 * kDim, m, w.wproj, kIdxHeadsRank, iw + kIdxHeadsRank * rank, kIdxHeads, st);
      short_coll([&](cudaStream_t s) { comm::glm::allgather_xproj(ch, xg, iw, m, full, s); });
      PrepArgs pa;
      pa.xg = xg, pa.xrows = xr, pa.pos = p.pos + r0, pa.slot = p.slot + r0, pa.rope = rope, pa.q_norm = w.q_norm;
      pa.kv_norm = w.kv_norm;
      if (full) pa.k_norm_w = w.k_norm_w, pa.k_norm_b = w.k_norm_b, pa.ik = ik_of(l);
      pa.qr = p.qr + (size_t)r0 * kQLora, pa.kv = kv_of(l);
      attn_prep(pa, m, st);
      {
        GemvSeg s[2];
        s[0].w = w.q_b.w, s[0].s = w.q_b.s, s[0].n = kOProjK, s[0].y = p.q + (size_t)r0 * kOProjK, s[0].ldy = kOProjK;
        s[1].w = w.iq_b.w, s[1].s = w.iq_b.s, s[1].n = kIdxHeadsRank * kIdxDim, s[1].ldy = kIdxHeads * kIdxDim;
        s[1].y = iq + (size_t)rank * kIdxHeadsRank * kIdxDim;
        pf_proj(G, p.qr + (size_t)r0 * kQLora, kQLora, m, kQLora, s, full ? 2 : 1, GemvOut::kBF16, 1, st, false);
      }
      if (full) short_coll([&](cudaStream_t s) { comm::glm::allgather_idxq(ch, iq, m, s); });
      flush();  // the piece before's sublayer end (the layer before's last piece's), behind this piece's short ones
      if (full) {
        // rows at positions >= 2048 select (the rest attend densely), idx_rows at a time (the score scratch)
        for (int s0 = std::max(r0, kTopK - a); s0 < r0 + m; s0 += p.idx_rows) {
          const int n = std::min(p.idx_rows, r0 + m - s0);
          index_select(IdxQuery{p.iq + (size_t)s0 * kIdxHeads * kIdxDim, p.iw + (size_t)s0 * kIdxHeads}, p.pos + s0,
                       p.slot + s0, n, rope, ik_of(l), 0, 0, IdxScratch{p.scores, idx_ld, p.work},
                       p.topk + (size_t)s0 * kTopK, nullptr, st);
        }
      }
      uint16_t* qlat = p.qlat + (size_t)r0 * kHeads * kKvLora;
      uint16_t* u = p.u + (size_t)r0 * kHeads * kKvLora;
      {
        GemvSeg s;
        s.w = w.w_uk.w, s.s = w.w_uk.s, s.n = kHeads * kKvLora, s.rs = 128, s.ks = 64;
        s.y = qlat, s.ldy = kHeads * kKvLora, s.group_rows = kKvLora, s.x_group = kQkDim;
        pf_proj(G, p.q + (size_t)r0 * kOProjK, kOProjK, m, kNope, &s, 1, GemvOut::kBF16, 1, st, false);
      }
      if (!tune(kTunePfAttn)) sparse_attn_config(0);
      sparse_attn(qlat, p.q + (size_t)r0 * kOProjK, p.pos + r0, p.slot + r0, rope, kv_of(l), p.topk + (size_t)r0 * kTopK,
                  m, 1, AttnScratch{}, u, st);
      sparse_attn_config(-1);
      {
        GemvSeg s;
        s.w = w.w_uv.w, s.s = w.w_uv.s, s.n = kOProjK, s.rs = 64, s.ks = 128;
        s.y = p.o + (size_t)r0 * kOProjK, s.ldy = kOProjK, s.group_rows = kVDim, s.x_group = kKvLora;
        pf_proj(G, u, kHeads * kKvLora, m, kKvLora, &s, 1, GemvOut::kBF16, 1, st, false);
        GemvSeg d;
        d.w = w.o_proj.w, d.s = w.o_proj.s, d.n = kDim, d.y = p.part + (size_t)r0 * kDim, d.ldy = kDim;
        pf_proj(G, p.o + (size_t)r0 * kOProjK, kOProjK, m, kOProjK, &d, 1, GemvOut::kF32, 1, st, false);
      }
      deferred.push_back(end_of(j, p.part, nullptr, w.ffn_norm));
    }
    flush();
    for (int j = 0; j < P; ++j) {  // the MLP or the MoE
      const int r0 = pcs[j].r0, m = pcs[j].m;
      const uint16_t* x = p.xn + (size_t)r0 * kDim;
      after(st, done[j]);  // the piece's normed rows
      std::function<void()> e;
      if (dense_layer(l)) {
        GemvSeg s;
        s.w = w.gate_up.w, s.s = w.gate_up.s, s.n = 2 * kDenseInterRank;
        s.y = p.hm + (size_t)r0 * kDenseInterRank, s.ldy = kDenseInterRank;
        pf_proj(G, x, kDim, m, kDim, &s, 1, GemvOut::kSwiGLU, 1, st, false);
        GemvSeg d;
        d.w = w.down.w, d.s = w.down.s, d.n = kDim, d.y = p.part + (size_t)r0 * kDim, d.ldy = kDim;
        pf_proj(G, p.hm + (size_t)r0 * kDenseInterRank, kDenseInterRank, m, kDenseInterRank, &d, 1, GemvOut::kF32, 1,
                st, false);
        e = end_of(j, p.part, nullptr, next_norm);
      } else {
        float* gl = p.gl + (size_t)r0 * gx::kE;
        if (m > kMaxRows && tune(kTunePfRouter)) {
          GemvSeg s;
          s.w = w.router, s.n = gx::kE, s.y = gl, s.ldy = gx::kE;
          gemm_bf16(x, kDim, m, kDim, &s, 1, GemvOut::kF32, 4, st);
        } else {
          rows16(m, [&](int q0, int n) {
            GemvSeg s;
            s.w = w.router, s.n = gx::kE, s.y = gl + (size_t)q0 * gx::kE, s.ldy = gx::kE;
            gemv(false, x + (size_t)q0 * kDim, kDim, n, kDim, &s, 1, GemvOut::kF32, 4, st);
          });
        }
        int* ridx = p.ridx + (size_t)r0 * gx::kTopK;
        float* rw = p.rw + (size_t)r0 * gx::kTopK;
        gx::route(gl, w.router_bias, m, ridx, rw, st, false);
        gx::moe_prefill(w.moe, reinterpret_cast<const __nv_bfloat16*>(x), ridx, rw, m, p.ws, p.yr + (size_t)r0 * kDim,
                        p.ys + (size_t)r0 * kDim, st);
        e = end_of(j, p.yr, p.ys, next_norm);
      }
      if (ov && j + 1 == P) deferred.push_back(e);  // behind the next layer's first short collectives
      else e();
    }
    if (flags & kFlagDump) {
      flush();
      if (ov) after(st, rec(cs));
      dump_rows(sink, l, "hidden", p.pos, M, p.h, kEmbRow);
    }
  }
  flush();
  if (ov) after(st, rec(cs));  // the chunk's collectives done before anything else uses the ring
  // p.xn: the final normed rows
  if (flags & kFlagDump) dump_rows(sink, -1, "final", p.pos, M, p.xn, kEmbRow);
  if (flags & kFlagLogits)
    rows16(M, [&](int r0, int n) {
      GemvSeg s;
      s.w = head, s.n = kVocabRank, s.y = logits, s.ldy = kVocabRank;
      gemv(false, p.xn + (size_t)r0 * kDim, kDim, n, kDim, &s, 1, GemvOut::kF32, 1, st);
      gather_logits(sink, logits, p.pos + r0, n, false);
    });
  if (opt.drafts) mtp_fill(sl, a, M);
  (void)last;
}

// The MTP cache entries of the chunk's rows whose next token is known: t in [a - 1 (a pending row), a + M - 1).
// Row t: e = the embedding of token t + 1 (the chunk's row t + 1 - a), hid = the final hidden of row t (the pending
// row's, saved by the chunk before, or this chunk's row t - a). The last row waits for the token after it.
void Model::Impl::mtp_fill(int sl, int a, int M) {
  using namespace ::glm;
  Pre& p = *pre;
  const bool pend = pend_pos[sl] == a - 1 && a > 0;
  const int t0 = pend ? a - 1 : a, n = a + M - 1 - t0;
  if (n > 0) {
    const uint16_t* e = p.emb + (size_t)(t0 + 1 - a) * kDim;
    const uint16_t* hid = p.xn;
    if (pend) {  // the pending row then rows 0 .. M - 2, in p.h (free after the layers)
      CUDA_CHECK(cudaMemcpyAsync(p.h, m_pend + (size_t)sl * kDim, kEmbRow, cudaMemcpyDeviceToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(p.h + kDim, p.xn, (size_t)(M - 1) * kEmbRow, cudaMemcpyDeviceToDevice, st));
      hid = p.h;
    }
    uint16_t* x12 = reinterpret_cast<uint16_t*>(p.part);  // [n][12288] bf16 = [n][6144] f32's bytes
    uint16_t* mx = p.qlat;
    uint16_t* mxn = p.u;
    // position 0's embedding is zeroed (vLLM's rule; the decode pass's embed_pick does the same)
    if (t0 == 0) CUDA_CHECK(cudaMemsetAsync(p.emb + (size_t)(1 - a) * kDim, 0, kEmbRow, st));
    mtp_project(n, e, hid, x12, mx);
    ops::iota(p.mpos, p.mslot, t0, sl, n, st);
    const detail::LayerW& w = lw[kMtpLayer];
    rmsnorm(mx, kDim, w.attn_norm, kNormEps, kDim, mxn, kDim, n, st);
    {
      GemvSeg s;
      s.w = w.xproj.w, s.s = w.xproj.s, s.n = kXRows, s.rs = 16, s.ks = 128;
      s.y = p.xg + (size_t)rank * kXRows, s.ldy = kRanks * kXRows;
      pf_proj(tune(kTunePfGemm), mxn, kDim, n, kDim, &s, 1, GemvOut::kBF16, 4, st, false);
    }
    index_weights(mxn, n, w.wproj, kIdxHeadsRank, p.iw + kIdxHeadsRank * rank, kIdxHeads, st);
    comm::glm::allgather_xproj(ch, p.xg, p.iw, n, true, st);
    PrepArgs pa;
    pa.xg = p.xg, pa.xrows = kXRows, pa.pos = p.mpos, pa.slot = p.mslot, pa.rope = rope, pa.q_norm = w.q_norm;
    pa.kv_norm = w.kv_norm, pa.k_norm_w = w.k_norm_w, pa.k_norm_b = w.k_norm_b, pa.ik = ik_of(kMtpLayer);
    pa.qr = p.qr, pa.kv = kv_of(kMtpLayer);
    attn_prep(pa, n, st);
  }
  CUDA_CHECK(cudaMemcpyAsync(m_pend + (size_t)sl * kDim, p.xn + (size_t)(M - 1) * kDim, kEmbRow, cudaMemcpyDeviceToDevice, st));
  pend_pos[sl] = a + M - 1;
}

uint64_t Model::Impl::prefill(int sl, int a, int M, const int32_t* tokens, const Sampling* sp, uint32_t flags, Sink* sink) {
  if (!bound) die("model: prefill before bind");
  if (sl < 0 || sl >= opt.slots || a < 0 || M < 1 || a + M > opt.max_ctx - kMaxDrafts ||
      (size_t)sl * stride + a + M + opt.drafts > pool_positions())
    die("model: prefill of positions [%d, %d) in slot %d (max_ctx %d)", a, a + M, sl, opt.max_ctx);
  prefill_alloc();
  if (n_steps) wait(n_steps - 1);  // the steps queued before have finished with the shared buffers
  CUDA_CHECK(cudaEventRecord(ev_pf[0], st));
  for (int c0 = 0; c0 < M; c0 += pre->cap) {
    const int m = std::min(pre->cap, M - c0);
    prefill_chunk(sl, a + c0, m, tokens + c0, c0 + m == M, flags, sink);
  }
  uint64_t ret = ~0ull;
  if (sp) {  // the last row's sample, as a step of one row: the head, the sampler, the draft passes
    const uint64_t n = n_steps;
    if (n >= (uint64_t)ops::kInSlots) wait(n - ops::kInSlots);
    ops::StepIn& in = in_h[n % ops::kInSlots];
    for (int r = 0; r < kMaxRows; ++r) in.tokens[r] = tokens[M - 1], in.pos[r] = a + M - 1, in.slot[r] = sl, in.sess[r] = 0;
    for (auto& s : in.samp) s = ops::StepSamp{sp->seed, sp->temperature, sp->top_p, sp->top_k, sp->n_top};
    in.rows = 1, in.nsess = 1, in.step = (uint32_t)n;
    const int se = (int)(n % ops::kOutSlots);
    CUDA_CHECK(cudaEventRecord(ev_beg[se], st));
    ops::step_in(in_h, in_ctr, fb, in_d, st, false);
    CUDA_CHECK(cudaMemcpyAsync(xn, pre->xn + (size_t)((M - 1) % pre->cap) * kDim, kEmbRow, cudaMemcpyDeviceToDevice, st));
    ::glm::GemvSeg s;
    s.w = head, s.n = kVocabRank, s.y = logits, s.ldy = kVocabRank;
    ::glm::gemv(false, xn, kDim, 1, kDim, &s, 1, GemvOut::kF32, 1, st);
    ops::sample(ch, logits, rank * kVocabRank, in_d, out_h, fb, samp_ctr, st, opt.drafts == 0, 1, ctl_h);
    if (opt.drafts) {
      drafts(1, 0, nullptr);
      pend_pos[sl] = -1;
    }
    CUDA_CHECK(cudaEventRecord(ev_end[se], st));
    n_steps = n + 1;
    ret = n;
  }
  CUDA_CHECK(cudaEventRecord(ev_pf[1], st));
  return ret;
}

}  // namespace glm::model
