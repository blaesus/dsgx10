// The TP=4 decode step (model.h; docs/engine.md "The decode step"): the layer runner over attn/, experts/ and comm/,
// one CUDA graph per row count, PDL between kernels, the paced L2 prefetch on a side stream.
#include "model/model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "model/impl.h"

namespace glm::model {

using namespace detail;
using ::glm::GemvOut;
using ::glm::GemvSeg;

namespace {
constexpr int kMaxSplits = 32;
constexpr size_t kChannelCap = 4u << 20;  // the step's largest post: 16 rows of two f32 partials (786 KB)
// the decode indexer's score scratch: kMaxRows rows of the whole context
size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
}  // namespace

const char* kclass_name(int c) {
  static const char* n[kKClasses] = {"embed",   "x-projection", "xproj all-gather", "attn_prep", "q_b + wq_b",
                                     "idxq all-gather", "index_select", "W_UK", "sparse_attn", "W_UV",
                                     "o_proj", "all-reduce 1 (+norm)", "MLP / MoE", "all-reduce 2 (+norm)", "head",
                                     "sample", "MTP draft passes"};
  return c >= 0 && c < kKClasses ? n[c] : "?";
}

// ---- construction ---------------------------------------------------------------------------------------------

Model::Impl::Impl(int r, const Options& opts) : rank(r), opt(opts) {
  if (opt.max_ctx < 1 || opt.max_ctx > (1 << 20)) die("model: max_ctx %d (1 .. 1,048,576)", opt.max_ctx);
  if (opt.page < 0 || opt.page > opt.max_ctx) die("model: pages of %d positions (0 .. max_ctx %d)", opt.page, opt.max_ctx);
  if (opt.slots < 1 || opt.slots > (opt.page ? kMaxPages : kMaxSess))
    die("model: %d slots (1 .. %d)", opt.slots, opt.page ? kMaxPages : kMaxSess);
  if (opt.drafts < 0 || opt.drafts > kMaxDrafts) die("model: %d draft passes (0 .. %d)", opt.drafts, kMaxDrafts);
  if (opt.prefill_rows < 16 || opt.prefill_rows > kMaxChunk || opt.prefill_rows % 16)
    die("model: prefill chunks of %d rows (16 .. %d, a multiple of 16)", opt.prefill_rows, kMaxChunk);
  opt.max_ctx = (int)align_up(opt.max_ctx, 256);
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  int lo = 0, hi = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  if (opt.l2pf) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&sc, cudaStreamNonBlocking, hi));
    pfs = dalloc<PfState>(1);
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
  for (auto& e : ev_beg) CUDA_CHECK(cudaEventCreate(&e));
  for (auto& e : ev_end) CUDA_CHECK(cudaEventCreate(&e));
  for (auto& e : ev_pf) CUDA_CHECK(cudaEventCreate(&e));

  stride = opt.page ? (size_t)opt.page : (size_t)opt.max_ctx;
  alloc_caches();
  ctl_h = halloc<uint32_t>(1);

  const int R = kMaxRows;
  h = dalloc<uint16_t>((size_t)R * kDim);
  xn = dalloc<uint16_t>((size_t)R * kDim);
  hfin = dalloc<uint16_t>((size_t)R * kDim);
  xg = dalloc<uint16_t>((size_t)R * kRanks * ::glm::kXRows);
  iw = dalloc<float>((size_t)R * ::glm::kIdxHeads);
  qr = dalloc<uint16_t>((size_t)R * ::glm::kQLora);
  q = dalloc<uint16_t>((size_t)R * ::glm::kOProjK);
  iq = dalloc<uint16_t>((size_t)R * ::glm::kIdxHeads * ::glm::kIdxDim);
  qlat = dalloc<uint16_t>((size_t)R * ::glm::kHeads * ::glm::kKvLora);
  u = dalloc<uint16_t>((size_t)R * ::glm::kHeads * ::glm::kKvLora);
  o = dalloc<uint16_t>((size_t)R * ::glm::kOProjK);
  hm = dalloc<uint16_t>((size_t)R * ::glm::kDenseInterRank);
  ebuf = dalloc<uint16_t>((size_t)R * kRanks * kDim);
  part = dalloc<float>((size_t)R * kDim);
  yr = dalloc<float>((size_t)2 * R * kDim);
  gl = dalloc<float>((size_t)R * gx::kE);
  rw = dalloc<float>((size_t)R * gx::kTopK);
  ridx = dalloc<int>((size_t)R * gx::kTopK);
  logits = dalloc<float>((size_t)R * kVocabRank);
  topk = dalloc<int>((size_t)R * ::glm::kTopK);
  idx_ld = align_up(opt.max_ctx, 1024);
  idx_scores = dalloc<float>((size_t)R * idx_ld);
  idx_work = dalloc<uint32_t>(::glm::index_scratch_words(R));
  att_o = dalloc<float>((size_t)R * kMaxSplits * ::glm::kHeads * ::glm::kKvLora);
  att_ml = dalloc<float>((size_t)R * kMaxSplits * ::glm::kHeads * 2);
  moe_ws = dalloc<uint8_t>(gx::decode_ws_bytes());
  m_x12 = dalloc<uint16_t>((size_t)R * kEhK);
  m_xg = dalloc<uint16_t>((size_t)R * kDim);
  m_h = dalloc<uint16_t>((size_t)R * kDim);
  m_xn = dalloc<uint16_t>((size_t)R * kDim);
  m_tok = dalloc<int32_t>(R);
  m_pos = dalloc<int32_t>(R);
  m_topk = dalloc<int>((size_t)R * ::glm::kTopK);
  m_cand = dalloc<comm::glm::Cand>(R);
  m_best = dalloc<comm::glm::Cand>(R);
  m_draft = dalloc<int32_t>((size_t)kMaxDrafts * R);

  in_h = halloc<ops::StepIn>(ops::kInSlots);
  in_d = dalloc<ops::StepIn>(1);
  in_ctr = dalloc<uint32_t>(1);
  fb = dalloc<int32_t>(kMaxRows);
  out_h = halloc<SampleOut>(ops::kOutSlots);
  samp_ctr = dalloc<uint32_t>(1);
}

// caches: every slot holds `stride` positions of every layer (and the MTP layer); a paged session runs on across
// the slots after its own
void Model::Impl::alloc_caches() {
  if (st) CUDA_CHECK(cudaStreamSynchronize(st));
  for (auto& p : kv) {
    if (p) CUDA_CHECK(cudaFree(p));
    p = nullptr;
  }
  for (IdxCache& c : ik) {
    if (c.k) CUDA_CHECK(cudaFree(c.k));
    if (c.ks) CUDA_CHECK(cudaFree(c.ks));
    c = IdxCache{};
  }
  if (m_pend) CUDA_CHECK(cudaFree(m_pend));
  const size_t P = pool_positions();
  for (int l = 0; l < kCacheLayers; ++l) kv[l] = dalloc<uint16_t>(P * ::glm::kLat);
  for (int i = 0; i < kIdxCaches; ++i) {
    IdxCache& c = ik[i];
    c.fp8 = opt.fp8_index_keys;
    c.slot_stride = stride;
    c.k = dalloc<uint8_t>(P * (c.fp8 ? ::glm::kIdxDim : ::glm::kIdxEntryBf16));
    if (c.fp8) c.ks = dalloc<float>(P);
  }
  m_pend = dalloc<uint16_t>((size_t)opt.slots * kDim);
  pend_pos.assign(opt.slots, -1);
}

Model::Impl::~Impl() {
  if (st) cudaStreamSynchronize(st);
  for (auto& g : graphs) cudaGraphExecDestroy(g.second);
}

void Model::Impl::bind_common(comm::Ring& ring) {
  ch = ring.dev();
  if (ch.cap < kChannelCap) die("model: the ring's channel holds %u B a rank (at least %zu)", ch.cap, kChannelCap);
  for (int l = 0; l < kCacheLayers; ++l) {
    const LayerW& w = lw[l];
    if (!w.attn_norm || !w.xproj.w || !w.q_b.w || !w.w_uk.w || !w.w_uv.w || !w.o_proj.w || !w.ffn_norm)
      die("model: layer %d's attention weights are missing", l);
    if (full_layer(l) && (!w.iq_b.w || !w.wproj || !w.k_norm_w)) die("model: layer %d's indexer weights are missing", l);
    if (dense_layer(l) ? !w.gate_up.w || !w.down.w : !w.router || !w.moe.pool || !w.moe.desc || !w.moe.sh.g)
      die("model: layer %d's MLP weights are missing", l);
  }
  if (!embed || !head || !norm_w || !rope) die("model: the global weights are missing");
  if (opt.drafts && (!mtp.eh.w || !mtp.enorm || !mtp.hnorm || !mtp.head_norm)) die("model: the MTP weights are missing");
  ring.barrier();
  bound = true;
}

// ---- profiling marks ------------------------------------------------------------------------------------------

void Model::Impl::mark(int cls, int layer) {
  if (!prof) return;
  if (in_mtp) cls = kKMtp, layer = kMtpLayer;
  cudaEvent_t e;
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(e, st));
  pev.push_back(e);
  pmarks.emplace_back(cls, layer);
}

// ---- the paced L2 prefetch ------------------------------------------------------------------------------------

// Window w of layer l (detail::l2pf_window); the budgets: the collectives' measured times (docs/ring.md) at R rows,
// the attention's lower bound (docs/attention.md).
void Model::Impl::l2pf(int w, int l, int R) {
  if (!sc || !l2pf_on || prof || l >= kLayers) return;
  const LayerW& L = lw[l];
  auto fp8 = [](const Fp8W& x, int N, int K, int rs, int ks) -> std::pair<size_t, size_t> {
    return {x.w ? (size_t)N * K : 0, x.s ? (size_t)(N / rs) * (K / ks) * 4 : 0};
  };
  const bool full = full_layer(l);
  const int xr = full ? ::glm::kXRows : ::glm::kXRowsShared;
  PfTarget t[kPfTargets] = {};
  auto set = [&](int i, int j, const void* p, size_t n) { t[i].p[j] = p, t[i].n[j] = n; };
  auto a = fp8(L.xproj, xr, kDim, 16, 128);
  set(kPfXproj, 0, L.xproj.w, a.first), set(kPfXproj, 1, L.xproj.s, a.second);
  a = fp8(L.q_b, ::glm::kOProjK, ::glm::kQLora, 128, 128);
  set(kPfQb, 0, L.q_b.w, a.first), set(kPfQb, 1, L.q_b.s, a.second);
  if (full) {
    a = fp8(L.iq_b, 1024, ::glm::kQLora, 128, 128);
    set(kPfQb, 2, L.iq_b.w, a.first), set(kPfQb, 3, L.iq_b.s, a.second);
  }
  a = fp8(L.w_uk, ::glm::kHeads * ::glm::kKvLora, ::glm::kNope, 128, 64);
  set(kPfUk, 0, L.w_uk.w, a.first), set(kPfUk, 1, L.w_uk.s, a.second);
  a = fp8(L.w_uv, ::glm::kOProjK, ::glm::kKvLora, 64, 128);
  set(kPfUv, 0, L.w_uv.w, a.first), set(kPfUv, 1, L.w_uv.s, a.second);
  a = fp8(L.o_proj, kDim, ::glm::kOProjK, 128, 128);
  set(kPfOproj, 0, L.o_proj.w, a.first), set(kPfOproj, 1, L.o_proj.s, a.second);
  if (dense_layer(l)) {
    a = fp8(L.gate_up, 2 * ::glm::kDenseInterRank, kDim, 128, 128);
    set(kPfFfn0, 0, L.gate_up.w, a.first), set(kPfFfn0, 1, L.gate_up.s, a.second);
  } else {
    set(kPfFfn0, 0, L.router, (size_t)gx::kE * kDim * 2);
    set(kPfFfn1, 0, L.moe.sh.g, (size_t)gx::kI * kDim), set(kPfFfn1, 1, L.moe.sh.u, (size_t)gx::kI * kDim);
  }
  const int r = std::min(R, 8);
  const float us = w == 1   ? (dense_layer(l) || l == 0 ? 18.f : 24.f) + 5.f * (r - 1)  // the FFN all-reduce before layer l
                   : w == 2 ? 9.f                                                     // the x-projection all-gather
                   : w == 3 ? (full ? 30.f : 14.f)                                    // the attention
                            : 18.f + 5.f * (r - 1);                                   // the o_proj all-reduce
  fork(sc);
  const bool coll = w != 3;
  l2pf_window(t, l, w, us, coll ? ch.ready : nullptr, coll ? ch.seq : nullptr, pfs, sc);
}

// ---- one layer ------------------------------------------------------------------------------------------------

void Model::Impl::embed_rows(const int32_t* tok, const int32_t* pos, int R, bool zero_pos0, uint16_t* out) {
  const bool P = pdl();
  ops::embed_local(embed, tok, R, rank, ebuf, st, P);
  const size_t rb = (size_t)kEmbRow, ld = rb * kRanks;
  const comm::glm::RowSeg s{ebuf + (size_t)rank * kDim, ld, ebuf, ld, rb};
  comm::glm::allgather_rows(ch, &s, 1, R, st, 0, 0, P);
  ops::embed_pick(ebuf, tok, pos, zero_pos0, R, out, st, P);
}

void Model::Impl::layer(int l, int R, const int32_t* pos, const int32_t* slot, uint16_t* h_, uint16_t* xn_, int* topk_,
                        const uint16_t* next_norm, bool index_on, Sink* sink, uint32_t flags) {
  using namespace ::glm;
  const detail::LayerW& w = lw[l];
  const bool full = full_layer(l), P = pdl();
  const int xr = full ? kXRows : kXRowsShared;
  // the x-projection: this rank's rows into its block of xg [R][4][xr]; the indexer's head weights beside it
  {
    GemvSeg s;
    s.w = w.xproj.w, s.s = w.xproj.s, s.n = xr, s.rs = 16, s.ks = 128, s.y = xg + (size_t)rank * xr, s.ldy = kRanks * xr;
    gemv(true, xn_, kDim, R, kDim, &s, 1, GemvOut::kBF16, 4, st, P);
    if (full) index_weights(xn_, R, w.wproj, kIdxHeadsRank, iw + kIdxHeadsRank * rank, kIdxHeads, st, P);
  }
  mark(kKXproj, l);
  l2pf(2, l, R);
  comm::glm::allgather_xproj(ch, xg, iw, R, full, st, P);
  mark(kKAgX, l);
  {
    PrepArgs a;
    a.xg = xg, a.xrows = xr, a.pos = pos, a.slot = slot, a.rope = rope, a.q_norm = w.q_norm, a.kv_norm = w.kv_norm;
    if (full) a.k_norm_w = w.k_norm_w, a.k_norm_b = w.k_norm_b, a.ik = ik_of(l);
    a.qr = qr, a.kv = kv_of(l);
    attn_prep(a, R, st, P);
  }
  mark(kKPrep, l);
  const bool idx = full && index_on;
  {
    GemvSeg s[2];
    s[0].w = w.q_b.w, s[0].s = w.q_b.s, s[0].n = kOProjK, s[0].y = q, s[0].ldy = kOProjK;
    s[1].w = w.iq_b.w, s[1].s = w.iq_b.s, s[1].n = kIdxHeadsRank * kIdxDim, s[1].y = iq + (size_t)rank * kIdxHeadsRank * kIdxDim,
    s[1].ldy = kIdxHeads * kIdxDim;
    gemv(true, qr, kQLora, R, kQLora, s, idx ? 2 : 1, GemvOut::kBF16, 1, st, P);
  }
  mark(kKQb, l);
  if (idx) {
    comm::glm::allgather_idxq(ch, iq, R, st, P);
    mark(kKAgQ, l);
    index_select(IdxQuery{iq, iw}, pos, slot, R, rope, ik_of(l), 0, 0, IdxScratch{idx_scores, idx_ld, idx_work}, topk_,
                 nullptr, st, P);
    mark(kKIndex, l);
  }
  {  // absorbed W_UK: head h's q_nope (192) -> q_lat (512)
    GemvSeg s;
    s.w = w.w_uk.w, s.s = w.w_uk.s, s.n = kHeads * kKvLora, s.rs = 128, s.ks = 64, s.y = qlat, s.ldy = kHeads * kKvLora;
    s.group_rows = kKvLora, s.x_group = kQkDim;
    gemv(true, q, kOProjK, R, kNope, &s, 1, GemvOut::kBF16, 1, st, P);
  }
  mark(kKUk, l);
  l2pf(3, l, R);
  const int splits = opt.attn_splits ? opt.attn_splits : R <= 2 ? 16 : 8;
  sparse_attn(qlat, q, pos, slot, rope, kv_of(l), topk_, R, splits, AttnScratch{att_o, att_ml, kMaxSplits}, u, st, P);
  mark(kKAttn, l);
  {  // absorbed W_UV: head h's latent (512) -> v (256)
    GemvSeg s;
    s.w = w.w_uv.w, s.s = w.w_uv.s, s.n = kOProjK, s.rs = 64, s.ks = 128, s.y = o, s.ldy = kOProjK;
    s.group_rows = kVDim, s.x_group = kKvLora;
    gemv(true, u, kHeads * kKvLora, R, kKvLora, &s, 1, GemvOut::kBF16, 1, st, P);
  }
  mark(kKUv, l);
  {
    GemvSeg s;
    s.w = w.o_proj.w, s.s = w.o_proj.s, s.n = kDim, s.y = part, s.ldy = kDim;
    gemv(true, o, kOProjK, R, kOProjK, &s, 1, GemvOut::kF32, 1, st, P);
  }
  mark(kKOproj, l);
  l2pf(4, l, R);
  comm::glm::allreduce_add_norm(ch, part, R, h_, h_, w.ffn_norm, kNormEps, xn_, st, P);
  mark(kKAr1, l);
  if (dense_layer(l)) {
    GemvSeg s;
    s.w = w.gate_up.w, s.s = w.gate_up.s, s.n = 2 * kDenseInterRank, s.y = hm, s.ldy = kDenseInterRank;
    gemv(true, xn_, kDim, R, kDim, &s, 1, GemvOut::kSwiGLU, 1, st, P);
    GemvSeg d;
    d.w = w.down.w, d.s = w.down.s, d.n = kDim, d.y = part, d.ldy = kDim;
    gemv(true, hm, kDenseInterRank, R, kDenseInterRank, &d, 1, GemvOut::kF32, 1, st, P);
    mark(kKMlp, l);
    l2pf(1, l + 1, R);
    comm::glm::allreduce_add_norm(ch, part, R, h_, h_, next_norm, kNormEps, xn_, st, P);
  } else {
    GemvSeg s;  // the router's logits: fp32 products of bf16 values, f32 sums
    s.w = w.router, s.n = gx::kE, s.y = gl, s.ldy = gx::kE;
    gemv(false, xn_, kDim, R, kDim, &s, 1, GemvOut::kF32, 4, st, P);
    gx::route(gl, w.router_bias, R, ridx, rw, st, P);
    gx::DecodeCfg dc;
    dc.pdl = P;
    float* ys = yr + (size_t)R * kDim;
    gx::moe_decode(w.moe, reinterpret_cast<const __nv_bfloat16*>(xn_), ridx, rw, R, moe_ws, yr, ys, st, dc);
    mark(kKMlp, l);
    l2pf(1, l + 1, R);
    comm::glm::allreduce2_add_norm(ch, yr, ys, R, h_, h_, next_norm, kNormEps, xn_, st, P);
  }
  mark(kKAr2, l);
  if (flags & kFlagDump) dump_rows(sink, l, "hidden", pos, R, h_, kEmbRow);
}

void Model::Impl::dump_rows(Sink* sink, int L, const char* name, const int32_t* pos_dev, int R, const void* dev,
                            size_t row_bytes) {
  if (rank != 0 || !sink) return;
  CUDA_CHECK(cudaStreamSynchronize(st));
  std::vector<uint8_t> hb((size_t)R * row_bytes);
  std::vector<int32_t> hp(R);
  CUDA_CHECK(cudaMemcpy(hb.data(), dev, hb.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hp.data(), pos_dev, (size_t)R * 4, cudaMemcpyDeviceToHost));
  sink->tensor(L, name, hp.data(), R, hb.data(), row_bytes);
}

void Model::Impl::gather_logits(Sink* sink, const float* lg, const int32_t* pos_dev, int R, bool is_mtp) {
  static float* all = nullptr;
  if (!all) all = dalloc<float>((size_t)kMaxRows * kVocab);
  comm::glm::allgather_logits_f32(ch, lg, kVocabRank, all, R, st, false);
  CUDA_CHECK(cudaStreamSynchronize(st));
  if (rank != 0 || !sink) return;
  std::vector<float> hb((size_t)R * kVocab);
  std::vector<int32_t> hp(R);
  CUDA_CHECK(cudaMemcpy(hb.data(), all, hb.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hp.data(), pos_dev, (size_t)R * 4, cudaMemcpyDeviceToHost));
  sink->logits(hp.data(), R, hb.data(), is_mtp);
}

// ---- the step -------------------------------------------------------------------------------------------------

void Model::Impl::forward(int R, uint32_t flags, Sink* sink) {
  const bool P = pdl();
  const int32_t* tok = in_d->tokens;
  const int32_t* pos = in_d->pos;
  const int32_t* slot = in_d->slot;
  ops::step_in(in_h, in_ctr, fb, in_d, st, P);
  embed_rows(tok, pos, R, false, h);
  ::glm::rmsnorm(h, kDim, lw[0].attn_norm, ::glm::kNormEps, kDim, xn, kDim, R, st, P);
  mark(kKEmbed, -1);
  for (int l = 0; l < kLayers; ++l)
    layer(l, R, pos, slot, h, xn, topk, l + 1 < kLayers ? lw[l + 1].attn_norm : norm_w, true, sink, flags);
  if (forked) join(sc), forked = false;
  {  // the vocabulary-split head on the final normed rows: f32 logits (the reference's golden form)
    ::glm::GemvSeg s;
    s.w = head, s.n = kVocabRank, s.y = logits, s.ldy = kVocabRank;
    ::glm::gemv(false, xn, kDim, R, kDim, &s, 1, GemvOut::kF32, 1, st, P);
  }
  mark(kKHead, -1);
  if (flags & kFlagDump) dump_rows(sink, -1, "final", pos, R, xn, kEmbRow);
  if (flags & kFlagLogits) gather_logits(sink, logits, pos, R, false);
  ops::sample(ch, logits, rank * kVocabRank, in_d, out_h, fb, samp_ctr, st, opt.drafts == 0, R, ctl_h);
  mark(kKSample, -1);
  if (opt.drafts) drafts(R, flags, sink);
}

cudaGraphExec_t Model::Impl::graph(int R) {
  const auto it = graphs.find(R);
  if (it != graphs.end()) return it->second;
  cudaGraph_t g;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  forward(R, 0, nullptr);
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  cudaGraphExec_t e;
  CUDA_CHECK(cudaGraphInstantiate(&e, g, 0));
  CUDA_CHECK(cudaGraphDestroy(g));
  graphs.emplace(R, e);
  return e;
}

uint64_t Model::Impl::launch(const Batch& b, Sink* sink) {
  if (!bound) die("model: step before bind");
  const int R = b.R;
  if (R < 1 || R > kMaxRows || b.nsess < 1 || b.nsess > kMaxSess) die("model: a step of %d rows, %d sessions", R, b.nsess);
  for (int r = 0; r < R; ++r) {
    const bool same = r > 0 && b.sess[r] == b.sess[r - 1];
    bool ok = b.pos[r] >= 0 && b.pos[r] < opt.max_ctx && b.slot[r] >= 0 && b.slot[r] < opt.slots &&
              (size_t)b.slot[r] * stride + b.pos[r] + opt.drafts < pool_positions() && b.sess[r] >= 0 &&
              b.sess[r] < b.nsess;
    if (r > 0 && !same) ok &= b.sess[r] == b.sess[r - 1] + 1;
    if (same) ok &= b.slot[r] == b.slot[r - 1] && b.pos[r] == b.pos[r - 1] + 1;
    for (int q = 0; q < r; ++q) ok &= b.sess[q] == b.sess[r] || b.slot[q] != b.slot[r];
    if (opt.drafts > 1 && same) ok = false;  // chained passes would write over the next row's MTP entry (G7)
    if (opt.drafts && b.pos[r] + opt.drafts >= opt.max_ctx) ok = false;
    if (!ok) die("model: row %d: position %d, slot %d, session %d", r, b.pos[r], b.slot[r], b.sess[r]);
  }
  const uint64_t n = n_steps;
  if (n >= (uint64_t)ops::kInSlots) wait(n - ops::kInSlots);  // its StepIn slot has been read
  const auto h0 = std::chrono::steady_clock::now();
  ops::StepIn& in = in_h[n % ops::kInSlots];
  for (int r = 0; r < kMaxRows; ++r) {
    const int q = std::min(r, R - 1);
    in.tokens[r] = r < R ? b.tokens[r] : 0;
    in.pos[r] = b.pos[q], in.slot[r] = b.slot[q], in.sess[r] = b.sess[q];
  }
  for (int k = 0; k < kMaxSess; ++k) {
    const Sampling& s = b.sp[std::min(k, b.nsess - 1)];
    in.samp[k] = ops::StepSamp{s.seed, s.temperature, s.top_p, s.top_k, s.n_top};
  }
  in.rows = R, in.nsess = b.nsess, in.step = (uint32_t)n;
  for (int r = 0; r < R; ++r) pend_pos[b.slot[r]] = -1;  // a step's draft pass completes the MTP row before it
  const int slot_ev = (int)(n % ops::kOutSlots);
  CUDA_CHECK(cudaEventRecord(ev_beg[slot_ev], st));
  if (opt.graphs && !b.flags && !prof && warm[R]) {
    CUDA_CHECK(cudaGraphLaunch(graph(R), st));
  } else {
    forward(R, b.flags, sink);
    warm[R] = true;  // the kernels' one-time setup has run outside a capture
  }
  CUDA_CHECK(cudaEventRecord(ev_end[slot_ev], st));
  host_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0).count();
  n_steps = n + 1;
  return n;
}

const SampleOut& Model::Impl::wait(uint64_t n) {
  if (n >= n_steps) die("model: wait for step %llu, %llu launched", (unsigned long long)n, (unsigned long long)n_steps);
  const SampleOut& o = out_h[n % ops::kOutSlots];
  const uint32_t want = (uint32_t)n + 1;
  const auto t0 = std::chrono::steady_clock::now();
  for (long spin = 0;; ++spin) {
    const uint32_t s = __atomic_load_n(&o.seq, __ATOMIC_ACQUIRE);
    if (s == want) break;
    if ((int32_t)(s - want) > 0) die("model: the sample of step %llu was overwritten", (unsigned long long)n);
    if ((spin & 0xfffff) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(120))
      die("model: step %llu has not finished in 120 s (a rank is gone?)", (unsigned long long)n);
  }
  n_done = std::max(n_done, n + 1);
  return o;
}

// ---- Model ----------------------------------------------------------------------------------------------------

Model::Model(int rank, const Options& opt) : d_(std::make_unique<Impl>(rank, opt)) {}
Model::~Model() = default;
void Model::bind(glw::WeightFile& wf, comm::Ring& ring, const SynthOptions* fill_experts) {
  d_->bind_file(wf, fill_experts);
  d_->bind_common(ring);
}
void Model::bind_synthetic(const SynthOptions& so, comm::Ring& ring) {
  d_->bind_synth(so);
  d_->bind_common(ring);
}
uint64_t Model::launch(const Batch& b, Sink* sink) { return d_->launch(b, sink); }
uint64_t Model::launch(int R, int pos0, const int32_t* tokens, const Sampling& sp, int slot, uint32_t flags, Sink* sink) {
  Batch b;
  b.R = R, b.nsess = 1, b.flags = flags;
  for (int r = 0; r < R; ++r) b.tokens[r] = tokens[r], b.pos[r] = pos0 + r, b.slot[r] = slot, b.sess[r] = 0;
  b.sp[0] = sp;
  return d_->launch(b, sink);
}
const SampleOut& Model::wait(uint64_t n) { return d_->wait(n); }
uint64_t Model::prefill(int slot, int a, int M, const int32_t* tokens, const Sampling* sp, uint32_t flags, Sink* sink) {
  return d_->prefill(slot, a, M, tokens, sp, flags, sink);
}
void Model::reset_slot(int slot) { d_->pend_pos.at(slot) = -1; }
float Model::gpu_ms(uint64_t n) const {
  const int s = (int)(n % ops::kOutSlots);
  CUDA_CHECK(cudaEventSynchronize(d_->ev_end[s]));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, d_->ev_beg[s], d_->ev_end[s]));
  return ms;
}
float Model::prefill_ms() const {
  CUDA_CHECK(cudaEventSynchronize(d_->ev_pf[1]));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, d_->ev_pf[0], d_->ev_pf[1]));
  return ms;
}

Profile Model::profile(const Batch& b) {
  Impl& I = *d_;
  I.prof = true;
  I.pmarks.clear();
  for (cudaEvent_t e : I.pev) cudaEventDestroy(e);
  I.pev.clear();
  I.mark(-1, -1);
  const uint64_t n = I.launch(b, nullptr);
  I.wait(n);
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  I.prof = false;
  Profile p;
  for (size_t i = 1; i < I.pev.size(); ++i) {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, I.pev[i - 1], I.pev[i]));
    const int c = I.pmarks[i].first, l = I.pmarks[i].second;
    if (c >= 0 && c < kKClasses) p.ms[c] += ms;
    if (l >= 0 && l < kCacheLayers) p.layer_ms[l] += ms;
    p.total_ms += ms;
  }
  return p;
}

std::vector<uint64_t> Model::cache_hashes(int slot, int n) const {
  const Impl& I = *d_;
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  std::vector<uint64_t> out;
  auto fnv = [](const uint8_t* p, size_t n_, uint64_t h) {
    for (size_t i = 0; i < n_; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
  };
  std::vector<uint8_t> b;
  for (int l = 0; l < kCacheLayers; ++l) {
    const size_t bytes = (size_t)n * ::glm::kLat * 2;
    b.resize(bytes);
    CUDA_CHECK(cudaMemcpy(b.data(), I.kv[l] + (size_t)slot * I.stride * ::glm::kLat, bytes, cudaMemcpyDeviceToHost));
    uint64_t hsh = fnv(b.data(), bytes, 0xcbf29ce484222325ull);
    if (full_layer(l)) {
      const ::glm::IdxCache c = I.ik_of(l);
      const size_t eb = c.fp8 ? ::glm::kIdxDim : ::glm::kIdxEntryBf16;
      b.resize((size_t)n * eb);
      CUDA_CHECK(cudaMemcpy(b.data(), c.k + (size_t)slot * c.slot_stride * eb, b.size(), cudaMemcpyDeviceToHost));
      hsh = fnv(b.data(), b.size(), hsh);
    }
    out.push_back(hsh);
  }
  return out;
}

std::vector<uint16_t> Model::last_hidden(int R) const {
  CUDA_CHECK(cudaStreamSynchronize(d_->st));
  std::vector<uint16_t> v((size_t)R * kDim);
  CUDA_CHECK(cudaMemcpy(v.data(), d_->xn, v.size() * 2, cudaMemcpyDeviceToHost));
  return v;
}
// ---- serving ----------------------------------------------------------------------------------------------------

void Model::set_ctl(uint32_t v) { __atomic_store_n(d_->ctl_h, v, __ATOMIC_RELEASE); }
void Model::prefill_init() { d_->prefill_alloc(); }
void Model::resize_slots(int slots) {
  Impl& I = *d_;
  if (slots < 1 || slots > (I.opt.page ? kMaxPages : kMaxSess)) die("model: resize to %d slots", slots);
  drain();
  for (auto& g : I.graphs) CUDA_CHECK(cudaGraphExecDestroy(g.second));
  I.graphs.clear();
  I.opt.slots = slots;
  I.alloc_caches();
}
size_t Model::cache_pos_bytes() const {
  const Impl& I = *d_;
  size_t b = (size_t)kCacheLayers * ::glm::kLat * 2;
  for (const IdxCache& c : I.ik) b += c.fp8 ? ::glm::kIdxDim + 4 : ::glm::kIdxEntryBf16;
  return b;
}
namespace {
// n elements of `width` bytes from src to dst positions apart in one array (overlap allowed): pieces no longer than
// the distance, front to back when moving down, back to front when moving up
void move_entries(uint8_t* base, size_t dst, size_t src, size_t n, size_t width, cudaStream_t s) {
  if (dst == src || n == 0) return;
  const size_t d = dst > src ? dst - src : src - dst, piece = std::min(n, d);
  for (size_t k = 0; k < n; k += piece) {
    const size_t m = std::min(piece, n - k), off = dst < src ? k : n - k - m;
    CUDA_CHECK(cudaMemcpyAsync(base + (dst + off) * width, base + (src + off) * width, m * width,
                               cudaMemcpyDeviceToDevice, s));
  }
}
}  // namespace
void Model::cache_copy(int dst, int src, int n) {
  Impl& I = *d_;
  const size_t a = (size_t)dst * I.stride, b = (size_t)src * I.stride;
  if (dst < 0 || src < 0 || n < 0 || std::max(a, b) + n > I.pool_positions()) die("model: cache_copy(%d, %d, %d)", dst, src, n);
  for (int l = 0; l < kCacheLayers; ++l)
    move_entries(reinterpret_cast<uint8_t*>(I.kv[l]), a, b, n, ::glm::kLat * 2, I.st);
  for (IdxCache& c : I.ik) {
    move_entries(c.k, a, b, n, c.fp8 ? ::glm::kIdxDim : ::glm::kIdxEntryBf16, I.st);
    if (c.fp8) move_entries(reinterpret_cast<uint8_t*>(c.ks), a, b, n, 4, I.st);
  }
  if (dst != src) I.pend_pos.at(dst) = -1;
}
namespace {
template <class F>
void each_cache(const Model::Impl& I, F&& f) {  // f(device base, entry bytes) in cache_get's order
  for (int l = 0; l < kCacheLayers; ++l) f(reinterpret_cast<uint8_t*>(I.kv[l]), (size_t)::glm::kLat * 2);
  for (const IdxCache& c : I.ik) f(c.k, c.fp8 ? (size_t)::glm::kIdxDim : (size_t)::glm::kIdxEntryBf16);
  for (const IdxCache& c : I.ik)
    if (c.fp8) f(reinterpret_cast<uint8_t*>(c.ks), (size_t)4);
}
}  // namespace
void Model::cache_get(int slot, int a, int n, void* host, int block) const {
  const Impl& I = *d_;
  const size_t p0 = (size_t)slot * I.stride + a;
  if (slot < 0 || a < 0 || n < 0 || n > block || p0 + n > I.pool_positions()) die("model: cache_get(%d, %d, %d)", slot, a, n);
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  uint8_t* h = static_cast<uint8_t*>(host);
  each_cache(I, [&](uint8_t* base, size_t w) {
    if (n) CUDA_CHECK(cudaMemcpyAsync(h, base + p0 * w, (size_t)n * w, cudaMemcpyDeviceToHost, I.st));
    h += (size_t)block * w;
  });
  CUDA_CHECK(cudaStreamSynchronize(I.st));
}
void Model::cache_put(int slot, int a, int n, const void* host, int block) {
  Impl& I = *d_;
  const size_t p0 = (size_t)slot * I.stride + a;
  if (slot < 0 || a < 0 || n < 0 || n > block || p0 + n > I.pool_positions()) die("model: cache_put(%d, %d, %d)", slot, a, n);
  drain();
  const uint8_t* h = static_cast<const uint8_t*>(host);
  each_cache(I, [&](uint8_t* base, size_t w) {
    if (n) CUDA_CHECK(cudaMemcpyAsync(base + p0 * w, h, (size_t)n * w, cudaMemcpyHostToDevice, I.st));
    h += (size_t)block * w;
  });
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  I.pend_pos.at(slot) = -1;
}

void Model::drain() {
  if (d_->n_steps) d_->wait(d_->n_steps - 1);
  CUDA_CHECK(cudaStreamSynchronize(d_->st));
}
int Model::file_expert_layers() const { return d_->real_expert_layers; }
double Model::launch_us() const { return d_->host_us; }
int Model::graph_count() const { return (int)d_->graphs.size(); }
int Model::rank() const { return d_->rank; }
const Options& Model::options() const { return d_->opt; }
size_t Model::weight_bytes() const { return d_->wbytes; }
void Model::set_l2pf(bool on) {
  d_->l2pf_on = on;
  drain();
  for (auto& g : d_->graphs) CUDA_CHECK(cudaGraphExecDestroy(g.second));
  d_->graphs.clear();
}
double Model::step_bytes(double experts, bool with_mtp) const {
  using namespace ::glm;
  auto fp8 = [](int N, int K, int rs, int ks) { return (double)N * K + (double)(N / rs) * (K / ks) * 4; };
  double b = 0;
  for (int l = 0; l < (with_mtp ? kCacheLayers : kLayers); ++l) {
    const bool full = full_layer(l);
    b += fp8(full ? kXRows : kXRowsShared, kDim, 16, 128) + fp8(kOProjK, kQLora, 128, 128) +
         fp8(kHeads * kKvLora, kNope, 128, 64) + fp8(kOProjK, kKvLora, 64, 128) + fp8(kDim, kOProjK, 128, 128);
    if (full) b += fp8(kIdxHeadsRank * kIdxDim, kQLora, 128, 128) + (double)kIdxHeadsRank * kDim * 2;
    if (dense_layer(l)) b += fp8(2 * kDenseInterRank, kDim, 128, 128) + fp8(kDim, kDenseInterRank, 128, 128);
    else b += (double)gx::kE * kDim * 2 + 2 * fp8(gx::kI, gx::kH, 128, 128) + fp8(gx::kH, gx::kI, 128, 128) + experts * d_->exp_bytes;
  }
  b += (double)kVocabRank * kDim * 2 * (with_mtp ? 2 : 1);  // the head (and the draft's)
  if (with_mtp) b += fp8(kEhRows, kEhK, 128, 128);
  return b;
}

}  // namespace glm::model
