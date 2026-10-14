// The MTP layer (78) in the decode step: its draft passes (docs/engine.md, "Drafts"; model.md § 3).
// Pass 1, every row r of the step (position p, the token t after it, the main model's final hidden h_p after
// model.norm): x = eh_proj([enorm(embed(t)) | hnorm(h_p)]) (the embedding zeroed at position 0), then decoder layer
// 78 (its own cache, its own indexer, its MoE). It writes the MTP cache entry of position p, the one later passes
// attend over. t is the next row's input while the session's rows are given or accepted, else the row's sample
// (ops::mtp_tokens), so the entries of a verify step's accepted rows and of the row after them are exact.
// Each session's chain starts at its chain row c (its last row; a verify step's last accepted row): shared_head.norm
// and the head on row c give draft 1, the token after the bonus (row c's sample). Pass k > 1, a row a session: position
// pos[c] + k - 1, the previous pass's draft and the MTP's own output hidden (before its head norm; the DeepSeek-MTP
// convention, vLLM's), pass 1's top-2048 of row c shared (index_share_for_mtp_iteration). Its cache entry is
// provisional: it lies past the session's history, and the next step's pass 1 writes it again before any row reads it.
#include "model/impl.h"

namespace glm::model {

using namespace detail;

void Model::Impl::mtp_project(int n, const uint16_t* e, const uint16_t* hid, uint16_t* x12, uint16_t* out) {
  const bool P = pdl();
  ::glm::rmsnorm(e, kDim, mtp.enorm, ::glm::kNormEps, kDim, x12, kEhK, n, st, P);
  ::glm::rmsnorm(hid, kDim, mtp.hnorm, ::glm::kNormEps, kDim, x12 + kDim, kEhK, n, st, P);
  {  // this rank's 1536 rows of eh_proj into its block of out
    ::glm::GemvSeg s;
    s.w = mtp.eh.w, s.s = mtp.eh.s, s.n = kEhRows, s.y = out + (size_t)rank * kEhRows, s.ldy = kDim;
    detail::pf_proj(tune(kTunePfGemm), x12, kEhK, n, kEhK, &s, 1, ::glm::GemvOut::kBF16, 1, st, P);
  }
  const size_t ld = (size_t)kDim * 2, rb = (size_t)kEhRows * 2;
  const comm::glm::RowSeg g{out + (size_t)rank * kEhRows, ld, out, ld, rb};
  comm::glm::allgather_rows(ch, &g, 1, n, st, 0, 0, P);
}

void Model::Impl::mtp_top1(int n, const uint16_t* xn_rows) {
  const bool P = pdl();
  ::glm::GemvSeg s;
  s.w = head, s.n = kVocabRank, s.y = logits, s.ldy = kVocabRank;
  ::glm::gemv(false, xn_rows, kDim, n, kDim, &s, 1, ::glm::GemvOut::kF32, 1, st, P);
  ops::local_top1(logits, n, rank * kVocabRank, m_cand, st, P);
  comm::glm::topk_exchange(ch, m_cand, m_best, n, 1, st, P);
}

// prefill's one-row step (prefill.cu): one session, Options::drafts passes
void Model::Impl::drafts(int R, uint32_t flags, Sink* sink) { drafts(R, 1, opt.drafts, flags, sink, true); }

namespace {
constexpr uint32_t kFlagMtpAll = 1u << 16;  // mtp_forced: pass 1's greedy draft of every row into m_best
}

void Model::Impl::drafts(int R, int nsess, int passes, uint32_t flags, Sink* sink, bool publish) {
  const bool P = pdl();
  in_mtp = true;
  // pass 1, every row
  ops::mtp_tokens(in_d, fb, R, m_tok, m_pos, m_sel, m_acc, st, P);
  embed_rows(m_tok, m_pos, R, true, hfin);
  mtp_project(R, hfin, xn, m_x12, m_h);
  if (flags & kFlagDump) dump_rows(sink, kMtpLayer, "x", m_pos, R, m_h, kEmbRow);  // the layer's input
  ::glm::rmsnorm(m_h, kDim, lw[kMtpLayer].attn_norm, ::glm::kNormEps, kDim, m_xn, kDim, R, st, P);
  layer(kMtpLayer, R, m_pos, in_d->slot, m_h, m_xn, m_topk, mtp.head_norm, true, sink, flags);
  if (flags & (kFlagMtpLogits | kFlagMtpAll)) {  // eager: every row's head (the golden format's mtp/ rows)
    ::glm::GemvSeg s;
    s.w = head, s.n = kVocabRank, s.y = logits, s.ldy = kVocabRank;
    ::glm::gemv(false, m_xn, kDim, R, kDim, &s, 1, ::glm::GemvOut::kF32, 1, st, P);
    if (flags & kFlagMtpLogits) gather_logits(sink, logits, m_pos, R, true);
    if (flags & kFlagMtpAll) {
      ops::local_top1(logits, R, rank * kVocabRank, m_cand, st, P);
      comm::glm::topk_exchange(ch, m_cand, m_best, R, 1, st, P);
      CUDA_CHECK(cudaMemcpyAsync(m_all, m_best, (size_t)R * sizeof(comm::glm::Cand), cudaMemcpyDeviceToDevice, st));
    }
  }
  // the chain: a row a session, from its chain row
  ops::mtp_select(in_d, m_sel, nsess, m_pos, m_h, m_xn, m_topk, c_h, c_xn, c_topk, c_pos, c_slot, st, P);
  for (int k = 0; k < passes; ++k) {
    if (k > 0) {
      embed_rows(c_tok, c_pos, nsess, true, hfin);
      mtp_project(nsess, hfin, c_h, m_x12, c_h);
      ::glm::rmsnorm(c_h, kDim, lw[kMtpLayer].attn_norm, ::glm::kNormEps, kDim, c_xn, kDim, nsess, st, P);
      layer(kMtpLayer, nsess, c_pos, c_slot, c_h, c_xn, c_topk, mtp.head_norm, false, nullptr, 0);
    }
    mtp_top1(nsess, c_xn);
    ops::mtp_next(m_best, nsess, c_tok, c_pos, m_draft + (size_t)k * kMaxSess, st, P);
    mark(kKMtp, kMtpLayer);
  }
  if (publish) ops::publish(out_h, in_d, m_draft, m_acc, passes, nsess, st, P);
  in_mtp = false;
}

void Model::mtp_forced(int slot, int pos0, int R, const int32_t* next, const uint16_t* hid, int passes, int32_t* draft1,
                       int32_t* chain, Sink* sink) {
  Impl& I = *d_;
  if (!I.bound) die("model: mtp_forced before bind");
  if (R < 1 || R > kMaxRows || passes < 1 || passes > I.opt.drafts || slot < 0 || slot >= I.opt.slots || pos0 < 0 ||
      (size_t)slot * I.stride + pos0 + R + passes > I.pool_positions())
    die("model: mtp_forced(%d, %d, %d, passes %d)", slot, pos0, R, passes);
  drain();
  // a plain step's input: row r at pos0 + r; its next token the next row's input, the last row's from fb
  ops::StepIn in{};
  for (int r = 0; r < kMaxRows; ++r) {
    const int q = std::min(r, R - 1);
    in.tokens[r] = r >= 1 && r < R ? next[r - 1] : 0, in.pos[r] = pos0 + q, in.slot[r] = slot, in.sess[r] = 0;
  }
  in.rows = R, in.nsess = 1, in.spec = 0;
  int32_t fbh[kMaxRows] = {};
  fbh[R - 1] = next[R - 1];
  CUDA_CHECK(cudaMemcpyAsync(I.in_d, &in, sizeof in, cudaMemcpyHostToDevice, I.st));
  CUDA_CHECK(cudaMemcpyAsync(I.fb, fbh, sizeof fbh, cudaMemcpyHostToDevice, I.st));
  CUDA_CHECK(cudaMemcpyAsync(I.xn, hid, (size_t)R * kEmbRow, cudaMemcpyHostToDevice, I.st));
  I.drafts(R, 1, passes, kFlagMtpAll | (sink ? kFlagDump : 0u), sink, false);  // eager
  std::vector<comm::glm::Cand> c(R);
  std::vector<int32_t> dr((size_t)kMaxDrafts * kMaxSess);
  CUDA_CHECK(cudaMemcpyAsync(c.data(), I.m_all, (size_t)R * sizeof(comm::glm::Cand), cudaMemcpyDeviceToHost, I.st));
  CUDA_CHECK(cudaMemcpyAsync(dr.data(), I.m_draft, dr.size() * 4, cudaMemcpyDeviceToHost, I.st));
  CUDA_CHECK(cudaStreamSynchronize(I.st));
  for (int r = 0; r < R; ++r) draft1[r] = c[r].i;
  for (int k = 1; k < passes; ++k) chain[k - 1] = dr[(size_t)k * kMaxSess];
  I.pend_pos.at(slot) = -1;
}

}  // namespace glm::model
