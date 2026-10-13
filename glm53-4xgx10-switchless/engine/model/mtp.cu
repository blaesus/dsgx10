// The MTP layer (78) in the decode step: its draft passes (docs/engine.md, "The MTP layer"; model.md § 3).
// Pass 1 of row r (position p, the token t after it, the main model's final hidden h_p after model.norm):
//   x = eh_proj([enorm(embed(t)) | hnorm(h_p)]) (the embedding zeroed at position 0), then decoder layer 78 (its own
//   cache, its own indexer, its MoE), shared_head.norm, the head; the draft is the greedy token (ids >= 154,856
//   masked). It writes the MTP cache entry of position p, the one the next steps attend over.
// Pass k > 1 (chained): position p + k - 1, the previous pass's draft and the MTP's own output hidden (before its
// head norm; inferred from the DeepSeek-MTP convention, vLLM's), the top-2048 of pass 1 shared
// (index_share_for_mtp_iteration). Its cache entry is provisional: the next step's pass 1 writes it again.
#include "model/impl.h"

namespace glm::model {

using namespace detail;

void Model::Impl::mtp_project(int n, const uint16_t* e, const uint16_t* hid, uint16_t* x12, uint16_t* out) {
  const bool P = pdl();
  ::glm::rmsnorm(e, kDim, mtp.enorm, ::glm::kNormEps, kDim, x12, kEhK, n, st, P);
  ::glm::rmsnorm(hid, kDim, mtp.hnorm, ::glm::kNormEps, kDim, x12 + kDim, kEhK, n, st, P);
  for (int r0 = 0; r0 < n; r0 += kMaxRows) {  // this rank's 1536 rows of eh_proj into its block of out
    const int rows = std::min(kMaxRows, n - r0);
    ::glm::GemvSeg s;
    s.w = mtp.eh.w, s.s = mtp.eh.s, s.n = kEhRows, s.y = out + (size_t)r0 * kDim + (size_t)rank * kEhRows, s.ldy = kDim;
    ::glm::gemv(true, x12 + (size_t)r0 * kEhK, kEhK, rows, kEhK, &s, 1, ::glm::GemvOut::kBF16, 1, st, P);
  }
  const size_t ld = (size_t)kDim * 2, rb = (size_t)kEhRows * 2;
  const comm::glm::RowSeg g{out + (size_t)rank * kEhRows, ld, out, ld, rb};
  comm::glm::allgather_rows(ch, &g, 1, n, st, 0, 0, P);
}

void Model::Impl::drafts(int R, uint32_t flags, Sink* sink) {
  const bool P = pdl();
  in_mtp = true;
  ops::mtp_tokens(in_d, fb, R, m_tok, m_pos, st, P);
  const uint16_t* hid = xn;  // the main model's final normed rows
  for (int k = 0; k < opt.drafts; ++k) {
    embed_rows(m_tok, m_pos, R, true, hfin);
    mtp_project(R, hfin, hid, m_x12, m_h);
    ::glm::rmsnorm(m_h, kDim, lw[kMtpLayer].attn_norm, ::glm::kNormEps, kDim, m_xn, kDim, R, st, P);
    layer(kMtpLayer, R, m_pos, in_d->slot, m_h, m_xn, m_topk, mtp.head_norm, k == 0, sink, k == 0 ? flags : 0);
    ::glm::GemvSeg s;
    s.w = head, s.n = kVocabRank, s.y = logits, s.ldy = kVocabRank;
    ::glm::gemv(false, m_xn, kDim, R, kDim, &s, 1, ::glm::GemvOut::kF32, 1, st, P);
    if (k == 0 && (flags & kFlagMtpLogits)) gather_logits(sink, logits, m_pos, R, true);
    ops::local_top1(logits, R, rank * kVocabRank, m_cand, st, P);
    comm::glm::topk_exchange(ch, m_cand, m_best, R, 1, st, P);
    ops::mtp_next(m_best, R, m_tok, m_pos, m_draft + (size_t)k * kMaxRows, st, P);
    hid = m_h;  // the next pass chains on the MTP's own output
    mark(kKMtp, kMtpLayer);
  }
  ops::publish(out_h, in_d, m_draft, opt.drafts, samp_ctr, st, P);
  in_mtp = false;
}

}  // namespace glm::model
