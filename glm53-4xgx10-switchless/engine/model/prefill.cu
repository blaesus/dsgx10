// Prefill (docs/engine.md, "Prefill"): a prompt's positions in chunks of up to Options::prefill_rows rows, through
// the same layers as the decode step with the kernels' prefill paths: the projections as 16-row GEMVs (the
// attention side has no tensor-core GEMM yet), attn_prep, index_select and sparse_attn over the whole chunk, the
// routed experts' grouped GEMMs (moe_prefill). A sublayer ends in a reduce-scatter by rows of the f32 partials
// (the all-reduce's sums, in rank order), the residual add and RMSNorm on this rank's quarter of the rows (attn's
// rmsnorm: the fused all-reduce's bits), and an all-gather of the quarter's residual and normed rows. So every
// value the decode step computes row by row is computed here with the same bits, except the routed experts'
// (prefill sums their K in another order, experts.md § 6). The MTP layer's cache is filled for the chunk's rows
// (eh_proj, its input norm, its x-projection and attn_prep: no attention, no MoE).
#include <algorithm>
#include <cstdio>

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
};
// its buffers live as long as the process (one model a process)
void Model::Impl::PreFree::operator()(Pre* p) const { delete p; }

namespace {
constexpr int kEmbSub = 1024;  // rows of an embedding all-gather
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
}

namespace {
// y rows [0, M) of one projection, 16 rows a launch (the GEMVs serve 1-16 rows)
template <class F>
void rows16(int M, F f) {
  for (int r0 = 0; r0 < M; r0 += kMaxRows) f(r0, std::min(kMaxRows, M - r0));
}
}  // namespace

// A sublayer's end over a chunk of M rows (Mp = M rounded up to 4): the partials a (and b) reduce-scattered by rows,
// the residual add and the next norm on this rank's quarter, the quarter's h and xn rows all-gathered.
static void sublayer_end(const comm::DevChannel& ch, int rank, Model::Impl::Pre& p, int M, const float* a, const float* b,
                         const uint16_t* norm, cudaStream_t st) {
  const int Mp = (M + kRanks - 1) / kRanks * kRanks, rq = Mp / kRanks;
  comm::reduce_scatter_rows_f32(ch, a, kDim, p.qa, kDim, rq, kDim, st);
  if (b) comm::reduce_scatter_rows_f32(ch, b, kDim, p.qb, kDim, rq, kDim, st);
  uint16_t* hq = p.h + (size_t)rank * rq * kDim;
  uint16_t* xq = p.xn + (size_t)rank * rq * kDim;
  ops::add_rows(hq, p.qa, b ? p.qb : nullptr, (size_t)rq * kDim, hq, st);
  ::glm::rmsnorm(hq, kDim, norm, ::glm::kNormEps, kDim, xq, kDim, rq, st);
  const size_t bytes = (size_t)rq * kEmbRow;
  const comm::AgSeg s[2] = {{hq, p.h, bytes}, {xq, p.xn, bytes}};
  comm::allgather_multi(ch, s, 2, st);
}

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
  for (int l = 0; l < kLayers; ++l) {
    const detail::LayerW& w = lw[l];
    const bool full = full_indexer(l);
    const int xr = full ? kXRows : kXRowsShared;
    const uint16_t* next_norm = l + 1 < kLayers ? lw[l + 1].attn_norm : norm_w;
    rows16(M, [&](int r0, int n) {
      GemvSeg s;
      s.w = w.xproj.w, s.s = w.xproj.s, s.n = xr, s.rs = 16, s.ks = 128;
      s.y = p.xg + (size_t)r0 * kRanks * xr + (size_t)rank * xr, s.ldy = kRanks * xr;
      gemv(true, p.xn + (size_t)r0 * kDim, kDim, n, kDim, &s, 1, GemvOut::kBF16, 4, st);
    });
    if (full) index_weights(p.xn, M, w.wproj, kIdxHeadsRank, p.iw + kIdxHeadsRank * rank, kIdxHeads, st);
    comm::glm::allgather_xproj(ch, p.xg, p.iw, M, full, st);
    PrepArgs pa;
    pa.xg = p.xg, pa.xrows = xr, pa.pos = p.pos, pa.slot = p.slot, pa.rope = rope, pa.q_norm = w.q_norm, pa.kv_norm = w.kv_norm;
    if (full) pa.k_norm_w = w.k_norm_w, pa.k_norm_b = w.k_norm_b, pa.ik = ik_of(l);
    pa.qr = p.qr, pa.kv = kv_of(l);
    attn_prep(pa, M, st);
    rows16(M, [&](int r0, int n) {
      GemvSeg s[2];
      s[0].w = w.q_b.w, s[0].s = w.q_b.s, s[0].n = kOProjK, s[0].y = p.q + (size_t)r0 * kOProjK, s[0].ldy = kOProjK;
      s[1].w = w.iq_b.w, s[1].s = w.iq_b.s, s[1].n = kIdxHeadsRank * kIdxDim, s[1].ldy = kIdxHeads * kIdxDim;
      s[1].y = p.iq + (size_t)r0 * kIdxHeads * kIdxDim + (size_t)rank * kIdxHeadsRank * kIdxDim;
      gemv(true, p.qr + (size_t)r0 * kQLora, kQLora, n, kQLora, s, full ? 2 : 1, GemvOut::kBF16, 1, st);
    });
    if (full) {
      comm::glm::allgather_idxq(ch, p.iq, M, st);
      // rows at positions >= 2048 select (the rest attend densely), idx_rows at a time (the score scratch)
      for (int s0 = std::max(0, kTopK - a); s0 < M; s0 += p.idx_rows) {
        const int n = std::min(p.idx_rows, M - s0);
        index_select(IdxQuery{p.iq + (size_t)s0 * kIdxHeads * kIdxDim, p.iw + (size_t)s0 * kIdxHeads}, p.pos + s0,
                     p.slot + s0, n, rope, ik_of(l), 0, 0, IdxScratch{p.scores, idx_ld, p.work},
                     p.topk + (size_t)s0 * kTopK, nullptr, st);
      }
    }
    rows16(M, [&](int r0, int n) {
      GemvSeg s;
      s.w = w.w_uk.w, s.s = w.w_uk.s, s.n = kHeads * kKvLora, s.rs = 128, s.ks = 64;
      s.y = p.qlat + (size_t)r0 * kHeads * kKvLora, s.ldy = kHeads * kKvLora, s.group_rows = kKvLora, s.x_group = kQkDim;
      gemv(true, p.q + (size_t)r0 * kOProjK, kOProjK, n, kNope, &s, 1, GemvOut::kBF16, 1, st);
    });
    sparse_attn(p.qlat, p.q, p.pos, p.slot, rope, kv_of(l), p.topk, M, 1, AttnScratch{}, p.u, st);
    rows16(M, [&](int r0, int n) {
      GemvSeg s;
      s.w = w.w_uv.w, s.s = w.w_uv.s, s.n = kOProjK, s.rs = 64, s.ks = 128;
      s.y = p.o + (size_t)r0 * kOProjK, s.ldy = kOProjK, s.group_rows = kVDim, s.x_group = kKvLora;
      gemv(true, p.u + (size_t)r0 * kHeads * kKvLora, kHeads * kKvLora, n, kKvLora, &s, 1, GemvOut::kBF16, 1, st);
      GemvSeg d;
      d.w = w.o_proj.w, d.s = w.o_proj.s, d.n = kDim, d.y = p.part + (size_t)r0 * kDim, d.ldy = kDim;
      gemv(true, p.o + (size_t)r0 * kOProjK, kOProjK, n, kOProjK, &d, 1, GemvOut::kF32, 1, st);
    });
    sublayer_end(ch, rank, p, M, p.part, nullptr, w.ffn_norm, st);
    if (dense_layer(l)) {
      rows16(M, [&](int r0, int n) {
        GemvSeg s;
        s.w = w.gate_up.w, s.s = w.gate_up.s, s.n = 2 * kDenseInterRank, s.y = p.hm + (size_t)r0 * kDenseInterRank,
        s.ldy = kDenseInterRank;
        gemv(true, p.xn + (size_t)r0 * kDim, kDim, n, kDim, &s, 1, GemvOut::kSwiGLU, 1, st);
        GemvSeg d;
        d.w = w.down.w, d.s = w.down.s, d.n = kDim, d.y = p.part + (size_t)r0 * kDim, d.ldy = kDim;
        gemv(true, p.hm + (size_t)r0 * kDenseInterRank, kDenseInterRank, n, kDenseInterRank, &d, 1, GemvOut::kF32, 1, st);
      });
      sublayer_end(ch, rank, p, M, p.part, nullptr, next_norm, st);
    } else {
      rows16(M, [&](int r0, int n) {
        GemvSeg s;
        s.w = w.router, s.n = gx::kE, s.y = p.gl + (size_t)r0 * gx::kE, s.ldy = gx::kE;
        gemv(false, p.xn + (size_t)r0 * kDim, kDim, n, kDim, &s, 1, GemvOut::kF32, 4, st);
      });
      gx::route(p.gl, w.router_bias, M, p.ridx, p.rw, st, false);
      gx::moe_prefill(w.moe, reinterpret_cast<const __nv_bfloat16*>(p.xn), p.ridx, p.rw, M, p.ws, p.yr, p.ys, st);
      sublayer_end(ch, rank, p, M, p.yr, p.ys, next_norm, st);
    }
    if (flags & kFlagDump) dump_rows(sink, l, "hidden", p.pos, M, p.h, kEmbRow);
  }
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
    rows16(n, [&](int r0, int k) {
      GemvSeg s;
      s.w = w.xproj.w, s.s = w.xproj.s, s.n = kXRows, s.rs = 16, s.ks = 128;
      s.y = p.xg + (size_t)r0 * kRanks * kXRows + (size_t)rank * kXRows, s.ldy = kRanks * kXRows;
      gemv(true, mxn + (size_t)r0 * kDim, kDim, k, kDim, &s, 1, GemvOut::kBF16, 4, st);
    });
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
