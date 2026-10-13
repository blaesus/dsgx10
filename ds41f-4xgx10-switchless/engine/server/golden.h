// The golden comparison on rank 0 (docs/engine.md "Golden check"): the engine's per-layer tensors and
// logits against the records of golden/run_ref.sh (golden/README.md), position by position.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model/model.h"
#include "server/npy.h"

namespace golden {

inline float bf16_to_f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f_to_bf16_exact(float f) {  // golden bf16 values stored as fp32: the low half is zero
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);
}

// Error statistics of one tensor kind over positions.
struct Acc {
  int n = 0;
  double sum_rel = 0, max_rel = 0;  // relative L2 per position
  double max_abs = 0, max_ref = 0;  // largest |diff| and |ref| seen
  int worst_pos = -1;
  // integer kinds: agreement
  long agree = 0, total = 0;
  void add_rel(double rel, double mabs, double mref, int pos) {
    ++n;
    sum_rel += rel;
    if (rel > max_rel) max_rel = rel, worst_pos = pos;
    max_abs = std::max(max_abs, mabs);
    max_ref = std::max(max_ref, mref);
  }
  void merge(const Acc& o) {
    n += o.n;
    sum_rel += o.sum_rel;
    if (o.max_rel > max_rel) max_rel = o.max_rel, worst_pos = o.worst_pos;
    max_abs = std::max(max_abs, o.max_abs);
    max_ref = std::max(max_ref, o.max_ref);
    agree += o.agree;
    total += o.total;
  }
  double mean() const { return n ? sum_rel / n : 0; }
};

struct LogitRec {
  int pos, ours, gold;
  float our_margin, gold_margin;  // top-1 minus top-2 (fp32 logits)
  float our_at_gold, gold_at_ours;
  int top5_overlap;
  float max_abs_top32;  // max |ours - golden| over golden's top-32 ids
  bool has_full;        // golden full logits at this position: the measures below
  double d5[5], kl, lp_margin_gold, lp_margin_ours;
  int ga;               // golden argmax from the full logits
};

class Golden : public model::Sink {
 public:
  explicit Golden(const std::string& dir) : dir_(dir) {
    tokens_ = std::make_unique<Npy>(dir + "/tokens.npy");
    Npy pos(dir + "/positions.npy");
    for (size_t i = 0; i < pos.count(); ++i) index_[(int)pos.ival(i)] = (int)i;
    Npy rec(dir + "/record_positions.npy");
    for (size_t i = 0; i < rec.count(); ++i) rec_index_[(int)rec.ival(i)] = (int)i;
  }
  int length() const { return (int)tokens_->count(); }
  int token(int p) const { return (int)tokens_->ival(p); }
  bool saved(int p) const { return index_.count(p) != 0; }
  // Positions >= p also go to a second set of statistics: report(f, true) prints only those.
  void set_split(int p) { split_ = p; }
  // Keep each saved position's error of every block's output stream and of the final hidden state:
  // report_positions prints how the difference grows along the sequence and through the layers.
  void set_per_position(bool on) { per_pos_ = on; }

  // ---- Sink ----
  void fill(int L, const char* name, int pos0, int R, void* host, size_t row_bytes) override {
    std::string n = name;
    int gl = L;
    if (n == "x") n = "out", gl = L - 1;
    if (n == "pre_mix") gl = L - 1;
    const Npy& g = arr(gl, n);
    const bool bf = n != "pre_mix";
    const size_t elems = row_bytes / (bf ? 2 : 4);
    if (g.row_elems() != elems) die("golden: %s has %zu per row, engine %zu", key(gl, n).c_str(), g.row_elems(), elems);
    for (int r = 0; r < R; ++r) {
      auto it = index_.find(pos0 + r);
      if (it == index_.end()) die("golden: position %d not saved, cannot inject", pos0 + r);
      const float* src = g.f32(it->second);
      uint8_t* dst = static_cast<uint8_t*>(host) + (size_t)r * row_bytes;
      if (bf)
        for (size_t i = 0; i < elems; ++i) reinterpret_cast<uint16_t*>(dst)[i] = f_to_bf16_exact(src[i]);
      else
        std::memcpy(dst, src, row_bytes);
    }
  }

  void tensor(int L, const char* name, int pos0, int R, const void* data, size_t row_bytes) override {
    const std::string n = name;
    if (n == "index_topk") return topk(L, pos0, R, static_cast<const int32_t*>(data));
    if (n == "route_idx") return route(L, pos0, R, static_cast<const int32_t*>(data));
    if (n == "pool") return pool(L, pos0, R, static_cast<const int32_t*>(data));
    const Npy* g = find(L, n);
    if (!g) return;
    const bool bf = n.rfind("hc_", 0) != 0 && n != "pre_mix" && n != "route_w";
    const size_t elems = bf ? row_bytes / 2 : row_bytes / 4;
    if (g->row_elems() != elems) die("golden: %s has %zu per row, engine %zu", key(L, n).c_str(), g->row_elems(), elems);
    for (int r = 0; r < R; ++r) {
      auto it = index_.find(pos0 + r);
      if (it == index_.end()) continue;
      const float* ref = g->f32(it->second);
      double d2 = 0, r2 = 0, mabs = 0, mref = 0;
      for (size_t i = 0; i < elems; ++i) {
        const float v = bf ? bf16_to_f(static_cast<const uint16_t*>(data)[(size_t)r * elems + i])
                           : static_cast<const float*>(data)[(size_t)r * elems + i];
        const double d = (double)v - ref[i];
        d2 += d * d;
        r2 += (double)ref[i] * ref[i];
        mabs = std::max(mabs, std::fabs(d));
        mref = std::max(mref, (double)std::fabs(ref[i]));
      }
      Acc t;
      t.add_rel(r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2), mabs, mref, pos0 + r);
      add(L, n, pos0 + r, t);
      if (per_pos_ && (n == "out" || n == "final_hidden")) {
        std::vector<double>& e = pos_err_[pos0 + r];
        e.resize(model::kLayers + 1, -1);
        e[L < 0 ? model::kLayers : L] = t.max_rel;
      }
    }
  }

  void logits(int pos0, int R, const float* full) override {
    const Npy& tv = top("logits_topv");
    const Npy& ti = top("logits_topi");
    for (int r = 0; r < R; ++r) {
      const int p = pos0 + r;
      if ((size_t)p >= tv.shape[0]) continue;
      const float* l = full + (size_t)r * model::kVocab;
      int b1 = 0, b2 = -1;
      for (int i = 1; i < model::kVocab; ++i)
        if (l[i] > l[b1]) b2 = b1, b1 = i;
        else if (b2 < 0 || l[i] > l[b2]) b2 = i;
      LogitRec rec{};
      rec.pos = p;
      rec.ours = b1;
      rec.gold = (int)ti.ival((size_t)p * 32);
      rec.our_margin = l[b1] - l[b2];
      rec.gold_margin = tv.f32(p)[0] - tv.f32(p)[1];
      rec.our_at_gold = l[rec.gold];
      rec.gold_at_ours = -INFINITY;
      float m = 0;
      for (int k = 0; k < 32; ++k) {
        const int id = (int)ti.ival((size_t)p * 32 + k);
        m = std::max(m, std::fabs(l[id] - tv.f32(p)[k]));
        if (id == b1) rec.gold_at_ours = tv.f32(p)[k];
      }
      rec.max_abs_top32 = m;
      // our top 5
      std::vector<int> ours5;
      {
        std::vector<std::pair<float, int>> best;
        for (int i = 0; i < model::kVocab; ++i) {
          if (best.size() < 5 || l[i] > best.back().first) {
            best.emplace_back(l[i], i);
            std::sort(best.begin(), best.end(), [](auto& a, auto& b) { return a.first > b.first; });
            if (best.size() > 5) best.pop_back();
          }
        }
        for (auto& b : best) ours5.push_back(b.second);
      }
      for (int k = 0; k < 5; ++k)
        for (int o : ours5) rec.top5_overlap += o == ti.ival((size_t)p * 32 + k);
      // golden/cmp_runs.py's measures where the golden full logits exist: |d logprob| over golden's
      // top 5, KL(golden || ours), and whether a top-1 flip is clear (both margins > 1 in logprob)
      rec.has_full = false;
      if (const Npy* gf = full_logits(); gf && p >= length() - (int)gf->shape[0]) {
        const float* gl = gf->f32((size_t)(p - (length() - (int)gf->shape[0])));
        std::vector<double> a(model::kVocab), b(model::kVocab);
        auto lsm = [](const float* x, std::vector<double>& o) {
          double m = -INFINITY, z = 0;
          for (int i = 0; i < model::kVocab; ++i) m = std::max(m, (double)x[i]);
          for (int i = 0; i < model::kVocab; ++i) z += std::exp((double)x[i] - m);
          const double lz = m + std::log(z);
          for (int i = 0; i < model::kVocab; ++i) o[i] = x[i] - lz;
        };
        lsm(gl, a);
        lsm(l, b);
        std::vector<int> oa(model::kVocab);
        for (int i = 0; i < model::kVocab; ++i) oa[i] = i;
        std::partial_sort(oa.begin(), oa.begin() + 5, oa.end(), [&](int x, int y) { return a[x] > a[y]; });
        for (int k = 0; k < 5; ++k) rec.d5[k] = std::fabs(b[oa[k]] - a[oa[k]]);
        double kl = 0;
        for (int i = 0; i < model::kVocab; ++i) kl += std::exp(a[i]) * (a[i] - b[i]);
        rec.kl = kl;
        rec.ga = oa[0];
        rec.lp_margin_gold = a[oa[0]] - a[b1];
        rec.lp_margin_ours = b[b1] - b[oa[0]];
        rec.has_full = true;
      }
      logit_recs_.push_back(rec);
    }
  }

  // ---- report ----
  // Per layer, mean / max relative L2 error of each tensor kind; then index and routing agreement.
  void report(FILE* f, bool late = false) const {
    static const char* kinds[] = {"engram_out", "attn_in", "attn_out", "mid", "ffn_in", "ffn_out", "out"};
    const auto& acc = late ? late_ : acc_;
    if (late) std::fprintf(f, "\npositions >= %d only:", split_);
    std::fprintf(f, "\nrelative L2 error per position, mean / max (block tensors vs golden):\n");
    std::fprintf(f, "  L ");
    for (const char* k : kinds) std::fprintf(f, " %17s", k);
    std::fprintf(f, " %13s %13s %9s\n", "hc mixes", "route agree", "topk ovl");
    for (int L = 0; L < model::kLayers; ++L) {
      std::fprintf(f, " %2d ", L);
      for (const char* k : kinds) {
        auto it = acc.find({L, k});
        if (it == acc.end() || !it->second.n) std::fprintf(f, " %17s", "-");
        else std::fprintf(f, "  %7.1e/%7.1e", it->second.mean(), it->second.max_rel);
      }
      double hc = 0;
      for (const char* k : {"hc_attn_pre", "hc_attn_post", "hc_attn_comb", "hc_ffn_pre", "hc_ffn_post", "hc_ffn_comb"}) {
        auto it = acc.find({L, k});
        if (it != acc.end()) hc = std::max(hc, it->second.max_rel);
      }
      std::fprintf(f, " %13.1e", hc);
      auto rt = acc.find({L, "route_idx"});
      if (rt != acc.end() && rt->second.total)
        std::fprintf(f, " %6ld/%-6ld", rt->second.agree, rt->second.total);
      else
        std::fprintf(f, " %13s", "-");
      auto tk = acc.find({L, "index_topk"});
      if (tk != acc.end() && tk->second.total)
        std::fprintf(f, " %8.5f", (double)tk->second.agree / tk->second.total);
      std::fprintf(f, "\n");
    }
    if (auto it = acc.find({20, "pool"}); it != acc.end() && it->second.total)
      std::fprintf(f, "layer 20 pool: %ld of %ld golden-kept blocks kept at %d record positions (we keep %.0f on average)\n",
                   it->second.agree, it->second.total, it->second.n, it->second.sum_rel / it->second.n);
    for (const char* k : {"final_pre_norm", "final_hidden"}) {
      auto it = acc.find({-1, k});
      if (it != acc.end()) std::fprintf(f, "%s: %.2e / %.2e\n", k, it->second.mean(), it->second.max_rel);
    }
  }
  // Top-1 agreement; every disagreement with both margins.
  int report_logits(FILE* f, int from = 0) const {
    int n = 0, agree = 0, near = 0;
    double sum_top32 = 0, max_top32 = 0, sum_t5 = 0;
    for (const LogitRec& r : logit_recs_) {
      if (r.pos < from) continue;
      ++n;
      agree += r.ours == r.gold;
      sum_top32 += r.max_abs_top32;
      max_top32 = std::max(max_top32, (double)r.max_abs_top32);
      sum_t5 += r.top5_overlap;
    }
    std::fprintf(f, "\nlogits (positions >= %d): top-1 %d/%d agree, top-5 overlap %.2f/5, max |diff| over golden top-32: mean %.3f max %.3f\n",
                 from, agree, n, n ? sum_t5 / n : 0, n ? sum_top32 / n : 0, max_top32);
    // against the sequence's own next token (prose: SGLang's greedy completion)
    int nt = 0, ours_ok = 0, gold_ok = 0;
    for (const LogitRec& r : logit_recs_) {
      if (r.pos < from || r.pos + 1 >= length()) continue;
      ++nt;
      ours_ok += r.ours == token(r.pos + 1);
      gold_ok += r.gold == token(r.pos + 1);
    }
    std::fprintf(f, "argmax == the next token of the sequence: ours %d/%d, golden (model.py) %d/%d\n", ours_ok, nt, gold_ok, nt);
    int shown = 0;
    for (const LogitRec& r : logit_recs_) {
      if (r.pos < from || r.pos + 1 >= length() || r.ours == token(r.pos + 1) || ++shown > 30) continue;
      std::fprintf(f, "  pos %5d: next %6d, ours %6d (margin %.3f), golden %6d%s\n", r.pos, token(r.pos + 1), r.ours,
                   r.our_margin, r.gold, r.gold == token(r.pos + 1) ? "" : " (golden misses too)");
    }
    int listed = 0;
    for (const LogitRec& r : logit_recs_) {
      if (r.pos < from || r.ours == r.gold) continue;
      const bool tie = r.gold_margin < 0.25f;
      near += tie;
      if (++listed > 40) continue;
      std::fprintf(f, "  pos %5d: ours %6d (margin %.3f, golden gives it %.3f) golden %6d (margin %.3f, we give it %.3f)%s\n",
                   r.pos, r.ours, r.our_margin, r.gold_at_ours, r.gold, r.gold_margin, r.our_at_gold,
                   tie ? "  near-tie" : "");
    }
    return agree;
  }
  // Per window of positions [edges[i], edges[i+1]): top-1 agreement with golden's own argmax, clear
  // flips (both margins > 1), and each side's accuracy against the sequence's next token (a quality
  // change of the model itself shows in both columns alike).
  void report_windows(FILE* f, const std::vector<int>& edges) const {
    std::fprintf(f, "\nper window: top-1 vs golden's argmax (clear flips), next-token accuracy ours / golden\n");
    for (size_t i = 0; i + 1 < edges.size(); ++i) {
      int n = 0, agree = 0, flips = 0, nt = 0, ours_ok = 0, gold_ok = 0;
      for (const LogitRec& r : logit_recs_) {
        if (r.pos < edges[i] || r.pos >= edges[i + 1]) continue;
        ++n;
        agree += r.ours == r.gold;
        flips += r.ours != r.gold && r.our_margin > 1.f && r.gold_margin > 1.f;
        if (r.pos + 1 < length()) {
          ++nt;
          ours_ok += r.ours == token(r.pos + 1);
          gold_ok += r.gold == token(r.pos + 1);
        }
      }
      if (n)
        std::fprintf(f, "  [%6d, %6d): top-1 %5d/%5d = %.4f (%d clear flips), next token %.3f / %.3f\n", edges[i],
                     edges[i + 1], agree, n, (double)agree / n, flips, nt ? (double)ours_ok / nt : 0.,
                     nt ? (double)gold_ok / nt : 0.);
    }
  }
  // Per saved position (runs of consecutive ones every 32nd): the relative L2 error of the output
  // stream of every 5th block and of the final hidden state, and whether top-1 agrees there.
  void report_positions(FILE* f) const {
    std::fprintf(f, "\nper position: relative L2 error of block outputs (out) and the final hidden state\n   pos ");
    for (int L = 4; L < model::kLayers; L += 5) std::fprintf(f, "    L%02d", L);
    std::fprintf(f, "   final top-1\n");
    std::map<int, const LogitRec*> lr;
    for (const LogitRec& r : logit_recs_) lr[r.pos] = &r;
    int prev = -2, run = 0;
    for (const auto& [p, e] : pos_err_) {
      run = p == prev + 1 ? run + 1 : 0;
      prev = p;
      if (run % 32) continue;
      std::fprintf(f, " %5d ", p);
      for (int L = 4; L < model::kLayers; L += 5) std::fprintf(f, " %6.3f", e[L]);
      auto it = lr.find(p);
      std::fprintf(f, "  %6.3f %s\n", e[model::kLayers], it == lr.end() ? "-" : it->second->ours == it->second->gold ? "=" : "x");
    }
  }
  const std::vector<LogitRec>& logit_recs() const { return logit_recs_; }
  // golden/cmp_runs.py's summary over positions [from, to) that have full golden logits
  void report_cmp(FILE* f, int from, int to) const {
    int n = 0, agree = 0, near = 0;
    std::vector<double> d5;
    double kl_sum = 0, kl_max = 0;
    std::vector<const LogitRec*> clear;
    for (const LogitRec& r : logit_recs_) {
      if (!r.has_full || r.pos < from || r.pos >= to) continue;
      ++n;
      for (double d : r.d5) d5.push_back(d);
      kl_sum += r.kl;
      kl_max = std::max(kl_max, r.kl);
      if (r.ga == r.ours) ++agree;
      else if (r.lp_margin_gold > 1.0 && r.lp_margin_ours > 1.0) clear.push_back(&r);
      else ++near;
    }
    if (!n) return;
    std::sort(d5.begin(), d5.end());
    double mean = 0;
    for (double d : d5) mean += d;
    mean /= d5.size();
    std::fprintf(f, "vs golden (cmp_runs.py measures), positions %d..%d: %d rows, argmax agree %d/%d (near-tie flips %d, clear flips %zu); "
                 "|d logprob| over golden top-5: mean %.4f median %.4f p95 %.4f max %.4f; KL mean %.5f max %.4f\n",
                 from, to - 1, n, agree, n, near, clear.size(), mean, d5[d5.size() / 2], d5[d5.size() * 95 / 100], d5.back(),
                 kl_sum / n, kl_max);
    for (const LogitRec* r : clear)
      std::fprintf(f, "  clear flip: pos %d golden %d ours %d margins %.3f / %.3f\n", r->pos, r->ga, r->ours,
                   r->lp_margin_gold, r->lp_margin_ours);
  }

 private:
  static std::string key(int L, const std::string& n) {
    if (L < 0) return n;
    char b[16];
    std::snprintf(b, sizeof b, "L%02d.", L);
    return b + n;
  }
  const Npy* find(int L, const std::string& n) {
    const std::string k = key(L, n);
    auto it = arrays_.find(k);
    if (it != arrays_.end()) return it->second.get();
    const std::string path = dir_ + "/" + k + ".npy";
    std::unique_ptr<Npy> a;
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
      std::fclose(f);
      a = std::make_unique<Npy>(path);
    }
    return (arrays_[k] = std::move(a)).get();
  }
  const Npy& arr(int L, const std::string& n) {
    const Npy* a = find(L, n);
    if (!a) die("golden: no %s", key(L, n).c_str());
    return *a;
  }
  const Npy& top(const std::string& n) { return arr(-1, n); }
  const Npy* full_logits() { return find(-1, "logits_full"); }
  void add(int L, const std::string& n, int pos, const Acc& t) {
    acc_[{L, n}].merge(t);
    if (split_ >= 0 && pos >= split_) late_[{L, n}].merge(t);
  }
  void route(int L, int pos0, int R, const int32_t* idx) {
    const Npy* g = find(L, "route_idx");
    if (!g) return;
    for (int r = 0; r < R; ++r) {
      auto it = index_.find(pos0 + r);
      if (it == index_.end()) continue;
      const size_t base = (size_t)it->second * model::kTopK;
      Acc t;
      for (int k = 0; k < model::kTopK; ++k)
        for (int j = 0; j < model::kTopK; ++j) t.agree += idx[r * model::kTopK + k] == g->ival(base + j);
      t.total = model::kTopK;
      add(L, "route_idx", pos0 + r, t);
    }
  }
  void pool(int L, int pos0, int R, const int32_t* blocks) {  // layer 20's kept blocks vs pool_blocks
    const Npy* g = find(L, "pool_blocks");
    if (!g) return;
    for (int r = 0; r < R; ++r) {
      auto it = rec_index_.find(pos0 + r);
      if (it == rec_index_.end()) continue;
      const uint8_t* keep = g->data + (size_t)it->second * g->row_elems();
      std::vector<int32_t> ours(blocks + r * model::kPoolBlocks, blocks + (r + 1) * model::kPoolBlocks);
      std::sort(ours.begin(), ours.end());
      Acc t;
      for (size_t b = 0; b < g->row_elems(); ++b) {
        if (!keep[b]) continue;
        ++t.total;
        t.agree += std::binary_search(ours.begin(), ours.end(), (int32_t)b);
      }
      long ours_n = 0;
      for (int32_t v : ours) ours_n += v >= 0;
      t.n = 1;
      t.sum_rel = (double)ours_n;  // blocks we kept, for the report
      add(L, "pool", pos0 + r, t);
    }
  }
  void topk(int L, int pos0, int R, const int32_t* idx) {
    const Npy* g = find(L, "index_topk");
    if (!g) return;
    for (int r = 0; r < R; ++r) {
      auto it = rec_index_.find(pos0 + r);
      if (it == rec_index_.end()) continue;
      const size_t w = g->row_elems(), base = (size_t)it->second * w;  // min(512, entries) columns
      std::vector<int32_t> ours(idx + r * model::kIdxTopk, idx + (r + 1) * model::kIdxTopk);
      std::sort(ours.begin(), ours.end());
      Acc t;
      for (size_t k = 0; k < w; ++k) {
        const int64_t ref = g->ival(base + k);
        if (ref < 0) continue;
        ++t.total;
        t.agree += std::binary_search(ours.begin(), ours.end(), (int32_t)ref);
      }
      add(L, "index_topk", pos0 + r, t);
    }
  }

  std::string dir_;
  std::unique_ptr<Npy> tokens_;
  std::map<int, int> index_, rec_index_;
  std::map<std::string, std::unique_ptr<Npy>> arrays_;
  std::map<std::pair<int, std::string>, Acc> acc_, late_;
  int split_ = -1;
  bool per_pos_ = false;
  std::map<int, std::vector<double>> pos_err_;
  std::vector<LogitRec> logit_recs_;
};

}  // namespace golden
