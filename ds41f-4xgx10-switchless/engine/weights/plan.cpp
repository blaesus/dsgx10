#include "weights/plan.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <unordered_map>

#include "weights/io.h"
#include "weights/layout.h"

namespace dsw {

namespace {

enum class Rule { Skip, Rep, Rows, Cols, RowsSeg5 };

struct Entry {
  const char* name;
  Rule rule;
  Layout layout = Layout::Natural;  // byte layout in the file (dsw.h); scales of L1/L2 tensors stay natural
};

constexpr Layout L1 = Layout::L1, L2 = Layout::L2, L3 = Layout::L3;

// The partition and the layouts. Keys are normalised names (every numeric component is N). Every
// tensor in the checkpoint must match exactly one key, so nothing is dropped or duplicated by
// accident. The tiled layouts are what engine/kernels' GEMVs read (engine/kernels/README.md).
const Entry kGlobal[] = {
    {"embed.weight", Rule::Rep},  // replicated: a lookup reads one 10 KiB row, so no collective is needed
    {"head.weight", Rule::Rows, L2},  // vocab split, 32,320 rows per rank
    {"norm.weight", Rule::Rep},
    {"image_start", Rule::Skip},  // vision is not planned
    {"image_end", Rule::Skip},
    {"image_newline", Rule::Skip},
};

// Every main layer (layers.N.) and every DSpark block (mtp.N.).
const Entry kBlock[] = {
    {"attn_norm.weight", Rule::Rep},
    {"ffn_norm.weight", Rule::Rep},
    {"attn.attn_sink", Rule::Rows},  // heads 16r .. 16r+15
    {"attn.wq_a.weight", Rule::Rows, L1},  // split by output, all-gathered with wkv
    {"attn.wq_a.scale", Rule::Rows},
    {"attn.q_norm.weight", Rule::Rep},
    {"attn.wkv.weight", Rule::Rows, L1},
    {"attn.wkv.scale", Rule::Rows},
    {"attn.kv_norm.weight", Rule::Rep},
    {"attn.wq_b.weight", Rule::Rows, L1},  // heads 16r .. 16r+15
    {"attn.wq_b.scale", Rule::Rows},
    {"attn.wo_a.weight", Rule::Rows, L1},  // groups 2r, 2r+1 (the same heads); scales checked <= 134
    {"attn.wo_a.scale", Rule::Rows},
    {"attn.wo_b.weight", Rule::Cols, L1},  // the input columns of those two groups; partial sums
    {"attn.wo_b.scale", Rule::Cols},
    // all 24 mixes over copy r of the 4 x 5120 stream: 24 partial dot products and a partial sum of
    // squares per row, all-reduced (25 floats)
    {"hc_attn_fn", Rule::Cols},
    {"hc_ffn_fn", Rule::Cols},
    {"hc_attn_base", Rule::Rep},
    {"hc_ffn_base", Rule::Rep},
    {"hc_attn_scale", Rule::Rep},
    {"hc_ffn_scale", Rule::Rep},
    {"ffn.gate.weight", Rule::Rows, L2},  // a quarter of the expert scores, all-gathered
    {"ffn.gate.bias", Rule::Rep},
    {"ffn.gate.bias_vl", Rule::Rep},
    {"ffn.shared_experts.w1.weight", Rule::Rows, L1},  // 576 of 2304 intermediate columns
    {"ffn.shared_experts.w1.scale", Rule::Rows},
    {"ffn.shared_experts.w3.weight", Rule::Rows, L1},
    {"ffn.shared_experts.w3.scale", Rule::Rows},
    {"ffn.shared_experts.w2.weight", Rule::Cols, L1},
    {"ffn.shared_experts.w2.scale", Rule::Cols},
    // tensor-split like the shared expert, stacked per layer by expert id; values and scales tiled
    {"ffn.experts.N.w1.weight", Rule::Rows, L3},
    {"ffn.experts.N.w1.scale", Rule::Rows, L3},
    {"ffn.experts.N.w3.weight", Rule::Rows, L3},
    {"ffn.experts.N.w3.scale", Rule::Rows, L3},
    {"ffn.experts.N.w2.weight", Rule::Cols, L3},
    {"ffn.experts.N.w2.scale", Rule::Cols, L3},
};

const Entry kLayerOnly[] = {
    // compressor (layers 2, 8, 14: wkv and wgate; 20: wkv): rows [128r, +128) of 512, a quarter of
    // the projections (the attention line's split); its norm stays whole
    {"attn.compressor.wkv.weight", Rule::Rows},
    {"attn.compressor.wgate.weight", Rule::Rows},
    {"attn.compressor.norm.weight", Rule::Rep},
    // indexer: replicated (decode splits the scanned positions across ranks, not the heads)
    {"attn.indexer.wq_b.weight", Rule::Rep, L1},
    {"attn.indexer.wq_b.scale", Rule::Rep},
    {"attn.indexer.weights_proj.weight", Rule::Rep},
    {"attn.indexer.wk.weight", Rule::Rep},
    {"attn.indexer.k_norm.weight", Rule::Rep},
    // 4 keys + 1 value, each 5120 wide: every rank takes the same quarter of each
    {"engram.wkv.weight", Rule::RowsSeg5},
    {"engram.wkv.scale", Rule::RowsSeg5},
    {"engram.q_weight", Rule::Rep},
    {"engram.k_weight", Rule::Rep},
    {"engram.embed.weight", Rule::Skip},  // the tables stay in the kit's per-rank NVMe shards
    {"engram.embed.scale", Rule::Skip},
};

const Entry kMtpOnly[] = {
    {"main_proj.weight", Rule::Rows, L1},  // mtp.0 only: 15360 -> 5120, split by output
    {"main_proj.scale", Rule::Rows},
    {"main_norm.weight", Rule::Rep},
    {"norm.weight", Rule::Rep},  // mtp.2 only
    {"markov_head.embed.weight", Rule::Rep},  // lookups, like embed
    {"markov_head.head.weight", Rule::Rows, L2},  // same vocab split as head, so its bias adds locally
    {"confidence_head.proj.weight", Rule::Rep},  // [1][5376]: one row does not fit L2's 8-row tiles
};

constexpr uint32_t kSegs = 5;  // RowsSeg5

const std::unordered_map<std::string, Entry>& rules() {
  static const auto table = [] {
    std::unordered_map<std::string, Entry> m;
    auto add = [&](const std::string& k, const Entry& e) {
      if (!m.emplace(k, e).second) die("plan table lists %s twice", k.c_str());
    };
    for (auto& e : kGlobal) add(e.name, e);
    for (auto& e : kBlock) add(std::string("layers.N.") + e.name, e), add(std::string("mtp.N.") + e.name, e);
    for (auto& e : kLayerOnly) add(std::string("layers.N.") + e.name, e);
    for (auto& e : kMtpOnly) add(std::string("mtp.N.") + e.name, e);
    return m;
  }();
  return table;
}

std::string component_of(const std::string& n) {
  bool mtp = n.rfind("mtp.", 0) == 0;
  const char* c;
  auto has = [&](const char* s) { return n.find(s) != std::string::npos; };
  if (n == "embed.weight") c = "embed";
  else if (n == "head.weight") c = "head";
  else if (has(".experts.N.")) c = "routed experts";
  else if (has("shared_experts")) c = "shared expert";
  else if (has("compressor")) c = "compressor";
  else if (has("indexer")) c = "indexer";
  else if (has("engram")) c = "engram wkv/gates";
  else if (has("hc_")) c = "mHC";
  else if (has("ffn.gate.")) c = "router";
  else if (has("markov_head") || has("confidence_head") || has("main_proj")) c = "draft heads";
  else if (has("norm")) c = "norms";
  else if (has("attn.")) c = "attention";
  else c = "other";
  return mtp ? std::string("dspark ") + c : std::string(c);
}

// Sort key: globals, then layers in order, then DSpark blocks.
int group_of(const std::string& name) {
  unsigned v = 0;
  if (std::sscanf(name.c_str(), "layers.%u.", &v) == 1) return 1 + int(v);
  if (std::sscanf(name.c_str(), "mtp.%u.", &v) == 1) return 1000 + int(v);
  return 0;
}

DType dtype_of(const SrcTensor& t) {
  if (t.dtype == "F32") return DType::F32;
  if (t.dtype == "BF16") return DType::BF16;
  if (t.dtype == "F8_E4M3") return DType::F8_E4M3;
  if (t.dtype == "F8_E8M0") return DType::F8_E8M0;
  if (t.dtype == "I8") {
    if (t.name.find(".experts.") == std::string::npos || t.name.find("shared_experts") != std::string::npos)
      die("%s is I8 but is not a routed expert weight", t.name.c_str());
    return DType::E2M1X2;
  }
  die("%s: dtype %s not expected in the text model", t.name.c_str(), t.dtype.c_str());
}

struct View {
  uint64_t rows, row_bytes;
};
View view_of(const SrcTensor& t) {
  uint64_t rb = st_dtype_bytes(t.dtype);
  for (size_t i = 1; i < t.shape.size(); i++) rb *= t.shape[i];
  return {t.shape.empty() ? 1 : t.shape[0], rb};
}

std::string shape_str(const std::vector<uint64_t>& s) {
  std::string o = "[";
  for (size_t i = 0; i < s.size(); i++) o += (i ? "," : "") + std::to_string(s[i]);
  return o + "]";
}

struct Cut {
  std::vector<Piece> pieces;
  std::vector<uint64_t> shape;
  uint64_t bytes = 0;
  SliceKind kind = SliceKind::Replicated;
  uint32_t nseg = 0;
  std::string desc;
};

// One source tensor's slice for rank r, with destination offsets starting at `base`.
Cut cut(const SrcTensor& t, Rule rule, int r, int w, uint64_t base) {
  View v = view_of(t);
  Cut c;
  c.shape = t.shape;
  char buf[96];
  switch (rule) {
    case Rule::Rep:
      c.pieces.push_back({&t, 0, v.rows, 0, v.row_bytes, base});
      c.kind = SliceKind::Replicated;
      c.desc = "replicated";
      break;
    case Rule::Rows: {
      if (t.shape.empty() || v.rows % w) die("%s: %llu rows do not split %d ways", t.name.c_str(), (unsigned long long)v.rows, w);
      uint64_t n = v.rows / w;
      c.pieces.push_back({&t, n * r, n, 0, v.row_bytes, base});
      c.shape[0] = n;
      c.kind = SliceKind::Rows;
      std::snprintf(buf, sizeof buf, "rows [%llu*r, +%llu) of %llu", (unsigned long long)n, (unsigned long long)n,
                    (unsigned long long)v.rows);
      c.desc = buf;
      break;
    }
    case Rule::Cols: {
      if (t.shape.size() != 2 || t.shape[1] % w) die("%s: columns do not split %d ways", t.name.c_str(), w);
      uint64_t nc = t.shape[1] / w, eb = st_dtype_bytes(t.dtype);
      c.pieces.push_back({&t, 0, v.rows, nc * eb * r, nc * eb, base});
      c.shape[1] = nc;
      c.kind = SliceKind::Cols;
      std::snprintf(buf, sizeof buf, "cols [%llu*r, +%llu) of %llu%s", (unsigned long long)nc, (unsigned long long)nc,
                    (unsigned long long)t.shape[1], t.dtype == "I8" ? " (packed bytes)" : "");
      c.desc = buf;
      break;
    }
    case Rule::RowsSeg5: {
      if (t.shape.empty() || v.rows % (kSegs * w)) die("%s: rows do not split into %u x %d", t.name.c_str(), kSegs, w);
      uint64_t seg = v.rows / kSegs, n = seg / w;
      for (uint32_t s = 0; s < kSegs; s++) c.pieces.push_back({&t, s * seg + n * r, n, 0, v.row_bytes, base + s * n * v.row_bytes});
      c.shape[0] = n * kSegs;
      c.kind = SliceKind::RowsSeg;
      c.nseg = kSegs;
      std::snprintf(buf, sizeof buf, "%u segments of %llu rows, each rows [%llu*r, +%llu)", kSegs,
                    (unsigned long long)seg, (unsigned long long)n, (unsigned long long)n);
      c.desc = buf;
      break;
    }
    case Rule::Skip: die("cut() on a skipped tensor");
  }
  for (auto& p : c.pieces) c.bytes += p.nrows * p.ncols;
  return c;
}

// Block geometry of a quantised weight, in logical elements: FP8 weights carry one ue8m0 per 32x32
// block, MXFP4 experts one per row and 32 elements along K.
void block_of(const SrcTensor& wt, uint64_t& br, uint64_t& bc, uint64_t& elems_per_byte) {
  if (wt.dtype == "F8_E4M3") br = 32, bc = 32, elems_per_byte = 1;
  else if (wt.dtype == "I8") br = 1, bc = 32, elems_per_byte = 2;
  else die("%s has a scale but dtype %s", wt.name.c_str(), wt.dtype.c_str());
}

// A scale must be cut exactly along the weight's blocks: same pieces, divided by the block size.
void check_scale_pair(const OutTensor& w, const OutTensor& s) {
  if (w.pieces.size() != s.pieces.size()) die("%s and its scale have different piece counts", w.name.c_str());
  for (size_t i = 0; i < w.pieces.size(); i++) {
    const Piece &pw = w.pieces[i], &ps = s.pieces[i];
    const SrcTensor &wt = *pw.src, &st = *ps.src;
    if (st.dtype != "F8_E8M0") die("%s: scale dtype %s", st.name.c_str(), st.dtype.c_str());
    uint64_t br, bc, epb;
    block_of(wt, br, bc, epb);
    uint64_t R = wt.shape[0], C = (wt.shape.size() == 2 ? wt.shape[1] : 1) * epb;
    if (st.shape.size() != 2 || st.shape[0] != (R + br - 1) / br || st.shape[1] != (C + bc - 1) / bc)
      die("%s: scale shape %s does not match weight %s", st.name.c_str(), shape_str(st.shape).c_str(),
          shape_str(wt.shape).c_str());
    uint64_t c0 = pw.col0 * epb, nc = pw.ncols * epb;  // weight columns in elements (1-byte elements)
    bool ok = pw.row0 % br == 0 && pw.nrows % br == 0 && c0 % bc == 0 && nc % bc == 0 &&
              ps.row0 == pw.row0 / br && ps.nrows == pw.nrows / br && ps.col0 == c0 / bc && ps.ncols == nc / bc;
    if (!ok)
      die("%s: slice rows [%llu,+%llu) cols [%llu,+%llu) cuts through its %llux%llu scale blocks", w.name.c_str(),
          (unsigned long long)pw.row0, (unsigned long long)pw.nrows, (unsigned long long)c0, (unsigned long long)nc,
          (unsigned long long)br, (unsigned long long)bc);
  }
}

void copy_str(char* dst, size_t cap, const std::string& s) {
  if (s.size() >= cap) die("string too long for the TOC: %s", s.c_str());
  std::memset(dst, 0, cap);
  std::memcpy(dst, s.data(), s.size());
}

}  // namespace

std::string normalize(const std::string& name) {
  std::string out;
  size_t i = 0;
  while (i <= name.size()) {
    size_t j = name.find('.', i);
    if (j == std::string::npos) j = name.size();
    std::string comp = name.substr(i, j - i);
    bool digits = !comp.empty() && std::all_of(comp.begin(), comp.end(), [](char c) { return c >= '0' && c <= '9'; });
    out += digits ? "N" : comp;
    if (j < name.size()) out += '.';
    i = j + 1;
  }
  return out;
}

Plan make_plan(const Checkpoint& ck, int rank, int world) {
  if (world < 1 || rank < 0 || rank >= world) die("bad rank %d / world %d", rank, world);
  Plan plan;
  plan.rank = rank;
  plan.world = world;

  // group sources by output: stacked experts share one output, ordered by expert index
  struct Group {
    Rule rule;
    Layout layout;
    std::string norm;
    std::vector<std::pair<int, const SrcTensor*>> srcs;  // (expert index or -1, tensor)
  };
  std::map<std::string, Group> groups;
  std::vector<std::string> unknown;
  for (auto& [name, t] : ck.tensors) {
    if (name.rfind("vision.", 0) == 0 || name.rfind("aligner.", 0) == 0) {
      plan.skipped.push_back(name);
      continue;
    }
    std::string norm = normalize(name);
    auto it = rules().find(norm);
    if (it == rules().end()) {
      unknown.push_back(name);
      continue;
    }
    if (it->second.rule == Rule::Skip) {
      plan.skipped.push_back(name);
      continue;
    }
    std::string out = name;
    int expert = -1;
    size_t pos = name.find(".ffn.experts.");
    if (pos != std::string::npos) {
      size_t a = pos + 13, b = name.find('.', a);
      expert = std::stoi(name.substr(a, b - a));
      out = name.substr(0, a) + name.substr(b + 1);
    }
    auto& g = groups[out];
    g.rule = it->second.rule;
    g.layout = it->second.layout;
    g.norm = norm;
    g.srcs.emplace_back(expert, &t);
  }
  if (!unknown.empty()) {
    std::sort(unknown.begin(), unknown.end());
    for (auto& n : unknown) std::fprintf(stderr, "no partition rule for %s (%s)\n", n.c_str(), normalize(n).c_str());
    die("%zu checkpoint tensors have no rule in plan.cpp", unknown.size());
  }
  std::sort(plan.skipped.begin(), plan.skipped.end());

  for (auto& [out, g] : groups) {
    std::sort(g.srcs.begin(), g.srcs.end());
    OutTensor o;
    o.name = out;
    o.component = component_of(g.norm);
    const SrcTensor& first = *g.srcs[0].second;
    o.dtype = dtype_of(first);
    o.src_shape = first.shape;
    bool stacked = g.srcs[0].first >= 0;
    if (stacked) {
      for (size_t e = 0; e < g.srcs.size(); e++) {
        if (g.srcs[e].first != (int)e) die("%s: expert %zu missing", out.c_str(), e);
        const SrcTensor& t = *g.srcs[e].second;
        if (t.shape != first.shape || t.dtype != first.dtype) die("%s: experts differ in shape or dtype", out.c_str());
      }
      o.nstack = (uint32_t)g.srcs.size();
      size_t a = out.find(".ffn.experts.") + 13;
      o.source = out.substr(0, a) + "{e}." + out.substr(a);
    } else {
      if (g.srcs.size() != 1) die("%s: %zu sources", out.c_str(), g.srcs.size());
      o.source = first.name;
    }
    uint64_t per = 0;
    for (size_t e = 0; e < g.srcs.size(); e++) {
      Cut c = cut(*g.srcs[e].second, g.rule, rank, world, per * e);
      if (e == 0) {
        per = c.bytes;
        o.shape = c.shape;
        o.kind = c.kind;
        o.nseg = c.nseg;
        o.slice = (stacked ? std::to_string(o.nstack) + " experts, " : std::string()) + c.desc;
        // the first expert's pieces were placed at base 0, which is also per * 0
      }
      o.pieces.insert(o.pieces.end(), c.pieces.begin(), c.pieces.end());
      o.bytes += c.bytes;
    }
    if (stacked) o.shape.insert(o.shape.begin(), o.nstack);
    uint64_t elems = 1;
    for (auto d : o.shape) elems *= d;
    if (elems * dtype_bytes(o.dtype) != o.bytes) die("%s: shape %s does not match %llu bytes", out.c_str(),
                                                      shape_str(o.shape).c_str(), (unsigned long long)o.bytes);
    if (o.shape.size() > 4) die("%s: more than 4 dims", out.c_str());
    o.layout = g.layout;
    TileGeom geom;
    std::string why;
    if (o.layout != Layout::Natural && !tile_geom(o.layout, o.dtype, uint32_t(o.shape.size()), o.shape.data(), geom, &why))
      die("%s: shape %s does not fit layout %s: %s", out.c_str(), shape_str(o.shape).c_str(), layout_name(o.layout),
          why.c_str());
    plan.tensors.push_back(std::move(o));
  }

  // every scale is cut along its weight's blocks
  std::unordered_map<std::string, const OutTensor*> by_name;
  for (auto& o : plan.tensors) by_name[o.name] = &o;
  for (auto& o : plan.tensors) {
    if (o.name.size() < 6 || o.name.compare(o.name.size() - 6, 6, ".scale") != 0) continue;
    auto it = by_name.find(o.name.substr(0, o.name.size() - 6) + ".weight");
    if (it == by_name.end()) die("%s has no weight", o.name.c_str());
    check_scale_pair(*it->second, o);
  }

  std::stable_sort(plan.tensors.begin(), plan.tensors.end(), [](const OutTensor& a, const OutTensor& b) {
    int ga = group_of(a.name), gb = group_of(b.name);
    return ga != gb ? ga < gb : a.name < b.name;
  });

  plan.toc_offset = sizeof(Header);
  plan.data_offset = align_up(plan.toc_offset + plan.tensors.size() * sizeof(TocEntry));
  uint64_t off = plan.data_offset;
  for (auto& o : plan.tensors) {
    o.offset = off;
    off += align_up(o.bytes);
  }
  plan.file_bytes = off;
  return plan;
}

void check_coverage(const Checkpoint& ck, int world) {
  struct Rect {
    int rank;
    uint64_t r0, r1, c0, c1;
  };
  std::unordered_map<const SrcTensor*, std::vector<Rect>> cover;
  std::unordered_map<const SrcTensor*, SliceKind> kind;
  size_t skipped = 0;
  for (int r = 0; r < world; r++) {
    Plan p = make_plan(ck, r, world);
    skipped = p.skipped.size();
    for (auto& o : p.tensors)
      for (auto& pc : o.pieces) {
        cover[pc.src].push_back({r, pc.row0, pc.row0 + pc.nrows, pc.col0, pc.col0 + pc.ncols});
        kind[pc.src] = o.kind;
      }
  }
  if (cover.size() + skipped != ck.tensors.size())
    die("coverage: %zu tensors placed + %zu skipped != %zu in the checkpoint", cover.size(), skipped, ck.tensors.size());
  for (auto& [src, rects] : cover) {
    View v = view_of(*src);
    if (kind[src] == SliceKind::Replicated) {
      if ((int)rects.size() != world) die("coverage: %s replicated %zu times", src->name.c_str(), rects.size());
      for (auto& x : rects)
        if (x.r0 || x.c0 || x.r1 != v.rows || x.c1 != v.row_bytes) die("coverage: %s not whole", src->name.c_str());
      continue;
    }
    uint64_t area = 0;
    for (size_t i = 0; i < rects.size(); i++) {
      auto& a = rects[i];
      if (a.r1 > v.rows || a.c1 > v.row_bytes) die("coverage: %s piece out of bounds", src->name.c_str());
      area += (a.r1 - a.r0) * (a.c1 - a.c0);
      for (size_t j = i + 1; j < rects.size(); j++) {
        auto& b = rects[j];
        if (a.r0 < b.r1 && b.r0 < a.r1 && a.c0 < b.c1 && b.c0 < a.c1)
          die("coverage: %s pieces of ranks %d and %d overlap", src->name.c_str(), a.rank, b.rank);
      }
    }
    if (area != v.rows * v.row_bytes) die("coverage: %s covered %llu of %llu bytes", src->name.c_str(),
                                          (unsigned long long)area, (unsigned long long)(v.rows * v.row_bytes));
  }
}

// Fills a TOC entry from a planned tensor (crc32c filled by the caller).
void fill_entry(const OutTensor& o, TocEntry& e) {
  std::memset(&e, 0, sizeof e);
  copy_str(e.name, sizeof e.name, o.name);
  copy_str(e.source, sizeof e.source, o.source);
  copy_str(e.slice, sizeof e.slice, o.slice);
  e.dtype = uint32_t(o.dtype);
  e.ndim = uint32_t(o.shape.size());
  for (size_t i = 0; i < o.shape.size(); i++) e.shape[i] = o.shape[i];
  e.offset = o.offset;
  e.bytes = o.bytes;
  e.kind = uint32_t(o.kind);
  e.nstack = o.nstack;
  e.nseg = o.nseg;
  e.src_ndim = uint32_t(o.src_shape.size());
  for (size_t i = 0; i < o.src_shape.size(); i++) e.src_shape[i] = o.src_shape[i];
  e.layout = uint32_t(o.layout);
}

}  // namespace dsw
