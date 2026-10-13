// The fused decode path (attn.h) against the separate kernels, bit for bit, on random state:
//   - mHC: hc_post launched with PDL, hc_project (the whole stream, one rank's share) and hc_mix (chunks
//     or pre-summed partials);
//   - the compressor projection by rows (four ranks' shares);
//   - decode_prep: q_norm + act_quant, kv_write, compressor emit + carry (ratio 2 at both position
//     parities, ratio 1), the indexer's head weights;
//   - index_fused (a dense quarter, with the pool, a block scan) and index_merge2;
//   - sparse_attn launched with PDL (main kernel and split-K combine);
//   - two whole layers chained with PDL as a decode step runs them (a ratio-2 Full layer; layer 20's
//     Full layer with the pool, then a Reindex layer on that pool).
// Rows 1, 6 and 8, and 9, 12 and 16 (B4): there the instances that take the rows in groups of up to 8 are also
// checked against calls of at most 8 rows (compressor_project_rows; index_dense and index_dense_blocks: each
// row's selections). decode_prep and the chained layers stay at 8 rows here: past 8 rows a step has several
// sessions (attn.h), which slot_tests checks. Prints one line per group; exits non-zero on any difference.
//   fused_tests
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "fill.cuh"
#include "kernels/dsk.h"
#include "testutil.h"

using namespace attn_test;
using attn::kDim;
using attn::kHcDim;
using attn::kHead;

namespace {

cudaStream_t g_s;
int g_checks = 0, g_fail = 0;
std::string g_group;
int g_group_checks = 0, g_group_fail = 0;

void begin(const std::string& name) {
  g_group = name;
  g_group_checks = g_checks;
  g_group_fail = g_fail;
}
void end() {
  std::printf("%-58s %3d checks  %s\n", g_group.c_str(), g_checks - g_group_checks,
              g_fail == g_group_fail ? "identical" : "DIFFERENT");
}

template <class T>
void same(const char* what, const Dev<T>& a, const Dev<T>& b) {
  CUDA_CHECK(cudaStreamSynchronize(g_s));
  auto ha = a.host(), hb = b.host();
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < ha.size(); ++i)
    if (std::memcmp(&ha[i], &hb[i], sizeof(T))) {
      if (!bad) first = i;
      ++bad;
    }
  ++g_checks;
  if (bad) {
    ++g_fail;
    std::printf("  %s: %s differs in %zu of %zu (first at %zu)\n", g_group.c_str(), what, bad, ha.size(), first);
  }
}

template <class T>
Dev<T> clone(const Dev<T>& a) {
  Dev<T> b(a.n);
  CUDA_CHECK(cudaMemcpyAsync(b.p, a.p, a.n * sizeof(T), cudaMemcpyDeviceToDevice, g_s));
  return b;
}

const __nv_bfloat16* bf(const uint16_t* p) { return reinterpret_cast<const __nv_bfloat16*>(p); }

Dev<int> positions(int rows, int last) {
  std::vector<int> h(rows);
  for (int r = 0; r < rows; ++r) h[r] = last - (rows - 1) + r;
  return Dev<int>(h.data(), h.size());
}
Dev<float2> table(int maxpos, bool compressed) {
  Dev<float2> t((size_t)maxpos * 32);
  attn::rope_table_build(t.p, maxpos, compressed, g_s);
  return t;
}
void sync() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ---- mHC ------------------------------------------------------------------------------------------

void test_mhc(int rows) {
  begin("mhc rows=" + std::to_string(rows));
  const int nch = attn::hc_chunks(0, kHcDim);
  Dev<uint16_t> x((size_t)rows * kHcDim), sub((size_t)rows * kDim), nw(kDim);
  Dev<float> post(rows * 4), comb(rows * 16), fn((size_t)attn::kMix * kHcDim), scale(3), base(attn::kMix);
  fill::bf16(x, 1.f);
  fill::bf16(sub, 1.f);
  fill::bf16_range(nw, 0.5f, 1.5f);
  fill::f32_range(post, 0.f, 2.f);
  fill::f32_range(comb, 0.f, 1.f);
  fill::f32(fn, 0.01f);
  fill::f32_range(scale, 0.5f, 1.5f);
  fill::f32(base, 1.f);
  sync();
  // separate kernels
  Dev<uint16_t> y((size_t)rows * kHcDim), yn((size_t)rows * kDim);
  Dev<float> chunks((size_t)nch * rows * 25), part(rows * 25), pre(rows * 4), po(rows * 4), cb(rows * 16);
  Dev<uint8_t> q((size_t)rows * kDim), qs((size_t)rows * kDim / 32);
  auto separate = [&](const uint16_t* stream) {
    attn::hc_partial(stream, rows, fn.p, 0, kHcDim, chunks.p, part.p, g_s);
    attn::hc_finish(part.p, rows, scale.p, base.p, pre.p, po.p, cb.p, g_s);
    attn::hc_pre(stream, pre.p, rows, nw.p, yn.p, nullptr, g_s);
    dsk::act_quant(bf(yn.p), rows, kDim, q.p, qs.p, g_s);
  };
  attn::hc_post(sub.p, x.p, post.p, comb.p, rows, y.p, g_s);
  separate(y.p);
  // fused
  Dev<uint16_t> y2((size_t)rows * kHcDim), yn2((size_t)rows * kDim);
  Dev<float> chunks2((size_t)nch * rows * 25), pre2(rows * 4), po2(rows * 4), cb2(rows * 16);
  Dev<uint8_t> q2((size_t)rows * kDim), qs2((size_t)rows * kDim / 32);
  auto check_mix = [&](const char* tag) {
    std::string t(tag);
    same((t + " pre").c_str(), pre, pre2);
    same((t + " post").c_str(), po, po2);
    same((t + " comb").c_str(), cb, cb2);
    same((t + " y").c_str(), yn, yn2);
    same((t + " q").c_str(), q, q2);
    same((t + " qs").c_str(), qs, qs2);
  };
  attn::hc_post(sub.p, x.p, post.p, comb.p, rows, y2.p, g_s, true);
  attn::hc_project(y2.p, rows, fn.p, 0, kHcDim, chunks2.p, true, g_s);
  attn::hc_mix(y2.p, chunks2.p, nch, rows, scale.p, base.p, nw.p, pre2.p, po2.p, cb2.p, yn2.p, q2.p, qs2.p, true, g_s);
  same("stream", y, y2);
  same("chunks", chunks, chunks2);
  check_mix("mix");
  // the partials summed already (all-reduced)
  CUDA_CHECK(cudaMemsetAsync(yn2.p, 0, yn2.n * 2, g_s));
  attn::hc_mix(y2.p, part.p, 1, rows, scale.p, base.p, nw.p, pre2.p, po2.p, cb2.p, yn2.p, q2.p, qs2.p, true, g_s);
  check_mix("summed");
  // one rank's share of the projection (copy 1)
  {
    int n4 = attn::hc_chunks(kDim, 2 * kDim);
    Dev<float> c4((size_t)n4 * rows * 25), o4(rows * 25), c4f((size_t)n4 * rows * 25);
    attn::hc_partial(y.p, rows, fn.p, kDim, 2 * kDim, c4.p, o4.p, g_s);
    attn::hc_project(y.p, rows, fn.p, kDim, 2 * kDim, c4f.p, true, g_s);
    same("rank-1 chunks", c4, c4f);
  }
  // another stream (layer 0's: no hc_post)
  separate(x.p);
  attn::hc_project(x.p, rows, fn.p, 0, kHcDim, chunks2.p, true, g_s);
  attn::hc_mix(x.p, chunks2.p, nch, rows, scale.p, base.p, nw.p, pre2.p, po2.p, cb2.p, yn2.p, q2.p, qs2.p, true, g_s);
  same("layer-0 chunks", chunks, chunks2);
  check_mix("layer-0");
  end();
}

// ---- compressor projection rows -------------------------------------------------------------------

void test_comp_proj(int rows) {
  begin("compressor_project_rows rows=" + std::to_string(rows));
  Dev<uint16_t> x((size_t)rows * kDim), wkv((size_t)kHead * kDim), wg((size_t)kHead * kDim);
  fill::bf16(x, 1.f);
  fill::bf16(wkv, 0.02f);
  fill::bf16(wg, 0.02f);
  sync();
  for (int ratio : {2, 1}) {
    attn::CompressorWeights cw{ratio, wkv.p, ratio > 1 ? wg.p : nullptr, nullptr, nullptr, nullptr};
    Dev<float> kv((size_t)rows * kHead), sc((size_t)rows * kHead), kv4((size_t)rows * kHead), sc4((size_t)rows * kHead);
    attn::compressor_project(x.p, cw, kv.p, sc.p, rows, g_s);
    for (int part = 0; part < 4; ++part) {
      Dev<float> kp((size_t)rows * 128), sp((size_t)rows * 128);
      attn::compressor_project_rows(x.p, cw, 128 * part, 128, kp.p, sp.p, rows, true, g_s);
      CUDA_CHECK(cudaMemcpy2DAsync(kv4.p + 128 * part, kHead * 4, kp.p, 128 * 4, 128 * 4, rows, cudaMemcpyDeviceToDevice, g_s));
      CUDA_CHECK(cudaMemcpy2DAsync(sc4.p + 128 * part, kHead * 4, sp.p, 128 * 4, 128 * 4, rows, cudaMemcpyDeviceToDevice, g_s));
      if (rows > attn::kDecodeGroup) {  // rows [0, 8) and [8, rows) in calls of their own
        Dev<float> kg((size_t)rows * 128), sg((size_t)rows * 128);
        for (int r0 = 0; r0 < rows; r0 += attn::kDecodeGroup)
          attn::compressor_project_rows(x.p + (size_t)r0 * kDim, cw, 128 * part, 128, kg.p + (size_t)r0 * 128,
                                        sg.p + (size_t)r0 * 128, std::min(attn::kDecodeGroup, rows - r0), true, g_s);
        same("kv as calls of 8 rows", kp, kg);
        if (ratio > 1) same("score as calls of 8 rows", sp, sg);
      }
      CUDA_CHECK(cudaStreamSynchronize(g_s));
    }
    same(ratio > 1 ? "kv (ratio 2)" : "kv (ratio 1)", kv, kv4);
    if (ratio > 1) same("score", sc, sc4);
  }
  end();
}

// ---- decode_prep ----------------------------------------------------------------------------------

struct CompState {  // a kv-source layer's compressor weights and caches
  int ratio = 0, cap = 0;
  Dev<uint16_t> norm, wk, knorm;
  Dev<float> carry_kv, carry_sc;
  Dev<uint8_t> ckv_kv, ckv_sc, ik_k, ik_sc;
  CompState() = default;
  CompState(int ratio_, int maxpos) : ratio(ratio_), cap(maxpos / ratio_ + 16) {
    norm = Dev<uint16_t>(kHead);
    wk = Dev<uint16_t>((size_t)attn::kIdxDim * kHead);
    knorm = Dev<uint16_t>(attn::kIdxDim);
    carry_kv = Dev<float>(16 * kHead);
    carry_sc = Dev<float>(16 * kHead);
    ckv_kv = Dev<uint8_t>((size_t)cap * kHead / 2);
    ckv_sc = Dev<uint8_t>((size_t)cap * 32);
    ik_k = Dev<uint8_t>((size_t)cap * 64);
    ik_sc = Dev<uint8_t>((size_t)cap * 4);
    fill::bf16_range(norm, 0.5f, 1.5f);
    fill::bf16(wk, 0.05f);
    fill::bf16_range(knorm, 0.5f, 1.5f);
    fill::f32(carry_kv, 1.f);
    fill::f32(carry_sc, 1.f);
    fill::u8(ckv_kv, 0xFF, 0);
    fill::u8(ckv_sc, 0x1F, 0x28);
    fill::u8(ik_k, 0xFF, 0);
    fill::u8(ik_sc, 7, 120);
  }
  CompState copy() const {
    CompState c;
    c.ratio = ratio;
    c.cap = cap;
    c.norm = clone(norm);
    c.wk = clone(wk);
    c.knorm = clone(knorm);
    c.carry_kv = clone(carry_kv);
    c.carry_sc = clone(carry_sc);
    c.ckv_kv = clone(ckv_kv);
    c.ckv_sc = clone(ckv_sc);
    c.ik_k = clone(ik_k);
    c.ik_sc = clone(ik_sc);
    return c;
  }
  attn::CompressorWeights cw(const uint16_t* wkv, const uint16_t* wgate) const {
    return {ratio, wkv, wgate, norm.p, wk.p, knorm.p};
  }
  attn::CompressorCarry carry() const { return {carry_kv.p, carry_sc.p}; }
  attn::Compressed ckv() const { return {ckv_kv.p, ckv_sc.p, cap}; }
  attn::IndexKeys ik() const { return {ik_k.p, ik_sc.p, cap}; }
  void check(const CompState& o, const char* tag) const {
    std::string t(tag);
    same((t + " ckv").c_str(), ckv_kv, o.ckv_kv);
    same((t + " ckv scales").c_str(), ckv_sc, o.ckv_sc);
    same((t + " index keys").c_str(), ik_k, o.ik_k);
    same((t + " index key scales").c_str(), ik_sc, o.ik_sc);
    same((t + " carry kv").c_str(), carry_kv, o.carry_kv);
    same((t + " carry score").c_str(), carry_sc, o.carry_sc);
  }
};

struct Ring {
  Dev<uint8_t> kv, sc;
  Ring() : kv((size_t)attn::kRing * kHead), sc((size_t)attn::kRing * 16) {
    fill::u8(kv, 0x77, 0);
    fill::u8(sc, 7, 120);
  }
  Ring(const Ring& o) : kv(clone(o.kv)), sc(clone(o.sc)) {}
  attn::Window win() const { return attn::Window{kv.p, sc.p}; }
};

// ---- DSpark's uses: a rank's fn slice, the lagged pre-mix, hc_pre as a PDL launch, kv_write_sets ----------

void test_dspark(int rows) {
  begin("dspark pieces rows=" + std::to_string(rows));
  const int nch = attn::hc_chunks(0, kHcDim);
  Dev<uint16_t> x((size_t)rows * kHcDim), nw(kDim);
  Dev<float> fn((size_t)attn::kMix * kHcDim), scale(3), base(attn::kMix), pin(rows * 4);
  fill::bf16(x, 1.f);
  fill::bf16_range(nw, 0.5f, 1.5f);
  fill::f32(fn, 0.01f);
  fill::f32_range(scale, 0.5f, 1.5f);
  fill::f32(base, 1.f);
  fill::f32_range(pin, 0.f, 1.f);
  sync();
  // each rank's chunks from its own [24][5120] slice of fn, in rank order, are the whole projection's
  Dev<float> chunks((size_t)nch * rows * 25), gathered((size_t)nch * rows * 25);
  attn::hc_project(x.p, rows, fn.p, 0, kHcDim, chunks.p, true, g_s);
  auto hfn = fn.host();
  for (int r = 0; r < 4; ++r) {
    std::vector<float> slice((size_t)attn::kMix * kDim);
    for (int i = 0; i < attn::kMix; ++i)
      std::memcpy(&slice[(size_t)i * kDim], &hfn[(size_t)i * kHcDim + (size_t)r * kDim], kDim * sizeof(float));
    Dev<float> fr(slice.data(), slice.size());
    attn::hc_project(x.p, rows, fr.p, r * kDim, (r + 1) * kDim, gathered.p + (size_t)r * (nch / 4) * rows * 25, true,
                     g_s, kDim);
    sync();
  }
  same("rank slices gathered", chunks, gathered);
  // pre_in: y collapses the stream with it; pre / post / comb stay the sublayer's own
  Dev<float> pre(rows * 4), po(rows * 4), cb(rows * 16), pre2(rows * 4), po2(rows * 4), cb2(rows * 16);
  Dev<uint16_t> yn((size_t)rows * kDim), yn2((size_t)rows * kDim), raw((size_t)rows * kDim), raw2((size_t)rows * kDim);
  Dev<uint8_t> q((size_t)rows * kDim), qs((size_t)rows * kDim / 32), q2((size_t)rows * kDim), qs2((size_t)rows * kDim / 32);
  attn::hc_mix(x.p, chunks.p, nch, rows, scale.p, base.p, nw.p, pre.p, po.p, cb.p, yn.p, q.p, qs.p, true, g_s);
  attn::hc_pre(x.p, pin.p, rows, nw.p, yn.p, raw.p, g_s);
  dsk::act_quant(bf(yn.p), rows, kDim, q.p, qs.p, g_s);
  attn::hc_mix(x.p, chunks.p, nch, rows, scale.p, base.p, nw.p, pre2.p, po2.p, cb2.p, yn2.p, q2.p, qs2.p, true, g_s,
               pin.p);
  auto check = [&](const char* tag) {
    std::string t(tag);
    same((t + " pre").c_str(), pre, pre2);
    same((t + " post").c_str(), po, po2);
    same((t + " comb").c_str(), cb, cb2);
    same((t + " y").c_str(), yn, yn2);
    same((t + " q").c_str(), q, q2);
    same((t + " qs").c_str(), qs, qs2);
  };
  check("pre_in");
  // pre_in aliasing pre (DSpark keeps one pre buffer)
  CUDA_CHECK(cudaMemcpyAsync(pre2.p, pin.p, rows * 4 * sizeof(float), cudaMemcpyDeviceToDevice, g_s));
  CUDA_CHECK(cudaMemsetAsync(yn2.p, 0, yn2.n * 2, g_s));
  attn::hc_mix(x.p, chunks.p, nch, rows, scale.p, base.p, nw.p, pre2.p, po2.p, cb2.p, yn2.p, q2.p, qs2.p, true, g_s,
               pre2.p);
  check("pre_in = pre");
  // hc_pre as a PDL launch
  attn::hc_pre(x.p, pin.p, rows, nw.p, yn2.p, raw2.p, g_s, true);
  same("hc_pre pdl y", yn, yn2);
  same("hc_pre pdl raw", raw, raw2);
  // kv_write_sets: rows [0, count) of 3 sets at pos0 + r, as kv_write of each
  const int count = rows > 1 ? rows - 1 : 1, pos0 = 300, kMax = attn::kMaxDecodeRows;
  auto tab = table(pos0 + kMax, false);
  Dev<uint16_t> kvp((size_t)3 * kMax * kHead), kvw((size_t)3 * kHead);
  fill::bf16(kvp, 2.f);
  fill::bf16_range(kvw, 0.5f, 1.5f);
  Ring ru[3];
  sync();
  Ring rf[3] = {Ring(ru[0]), Ring(ru[1]), Ring(ru[2])};
  std::vector<int> hp(count);
  for (int r = 0; r < count; ++r) hp[r] = pos0 + r;
  Dev<int> pos(hp.data(), count), dcount(&count, 1), dpos0(&pos0, 1);
  attn::KvSets sets;
  sets.n = 3;
  for (int b = 0; b < 3; ++b) {
    attn::kv_write(kvp.p + (size_t)b * kMax * kHead, kvw.p + b * kHead, pos.p, tab.p, ru[b].win(), nullptr, count, g_s);
    sets.s[b] = {kvp.p + (size_t)b * kMax * kHead, kvw.p + b * kHead, rf[b].win()};
  }
  attn::kv_write_sets(sets, kMax, dcount.p, dpos0.p, tab.p, true, g_s);
  for (int b = 0; b < 3; ++b) {
    same("kv_write_sets ring", ru[b].kv, rf[b].kv);
    same("kv_write_sets scales", ru[b].sc, rf[b].sc);
  }
  end();
}

// ratio 0: a window-only layer; index: also the head weights
void test_prep(int rows, int ratio, bool index, int last) {
  begin("decode_prep rows=" + std::to_string(rows) + " ratio=" + std::to_string(ratio) + (index ? " +wproj" : "") +
        " last=" + std::to_string(last));
  int maxpos = last + 2;
  auto tab = table(maxpos, ratio > 0);
  auto pos = positions(rows, last);
  Dev<uint16_t> qa((size_t)rows * attn::kQLora), qnw(attn::kQLora), kvp((size_t)rows * kHead), kvw(kHead);
  Dev<uint16_t> x((size_t)rows * kDim), wproj((size_t)attn::kIdxHeads * kDim);
  Dev<float> proj((size_t)rows * kHead), score((size_t)rows * kHead);
  fill::bf16(qa, 2.f);
  fill::bf16_range(qnw, 0.5f, 1.5f);
  fill::bf16(kvp, 2.f);
  fill::bf16_range(kvw, 0.5f, 1.5f);
  fill::bf16(x, 1.f);
  fill::bf16(wproj, 0.02f);
  fill::f32(proj, 1.f);
  fill::f32(score, 1.f);
  Ring ring_u;
  CompState cu = ratio ? CompState(ratio, maxpos) : CompState();
  sync();
  Ring ring_f(ring_u);
  CompState cf = cu.copy();
  // separate
  Dev<uint16_t> qr((size_t)rows * attn::kQLora);
  Dev<uint8_t> qq((size_t)rows * attn::kQLora), qs((size_t)rows * attn::kQLora / 32);
  Dev<float> wp(rows * attn::kIdxHeads);
  attn::q_norm(qa.p, qnw.p, qr.p, rows, g_s);
  dsk::act_quant(bf(qr.p), rows, attn::kQLora, qq.p, qs.p, g_s);
  attn::kv_write(kvp.p, kvw.p, pos.p, tab.p, ring_u.win(), nullptr, rows, g_s);
  if (ratio)
    attn::compressor_emit(proj.p, score.p, pos.p, rows, cu.cw(nullptr, nullptr), cu.carry(), tab.p, cu.ckv(), cu.ik(),
                          nullptr, 0, g_s);
  if (index) attn::gemv_bf16(x.p, rows, kDim, wproj.p, nullptr, 32, 32, wp.p, nullptr, true, g_s);
  // fused
  Dev<uint16_t> qr2((size_t)rows * attn::kQLora);
  Dev<uint8_t> qq2((size_t)rows * attn::kQLora), qs2((size_t)rows * attn::kQLora / 32);
  Dev<float> wp2(rows * attn::kIdxHeads);
  attn::DecodePrep a;
  a.pos = pos.p;
  a.table = tab.p;
  a.qa = qa.p;
  a.q_norm = qnw.p;
  a.qq = qq2.p;
  a.qs = qs2.p;
  a.qr = qr2.p;
  a.kv_pre = kvp.p;
  a.kv_norm = kvw.p;
  a.win = ring_f.win();
  if (ratio) {
    a.ckv_proj = proj.p;
    a.cscore = score.p;
    a.cw = cf.cw(nullptr, nullptr);
    a.carry = cf.carry();
    a.ckv = cf.ckv();
    a.ik = cf.ik();
  }
  if (index) {
    a.x = x.p;
    a.wproj = wproj.p;
    a.wp = wp2.p;
  }
  attn::decode_prep(a, rows, true, g_s);
  same("qr", qr, qr2);
  same("q", qq, qq2);
  same("q scales", qs, qs2);
  same("ring", ring_u.kv, ring_f.kv);
  same("ring scales", ring_u.sc, ring_f.sc);
  if (ratio) cu.check(cf, "compressor");
  if (index) same("head weights", wp, wp2);
  end();
}

// ---- indexer --------------------------------------------------------------------------------------

constexpr int kTopk = 512, kPool = 2048, kParts = 4;

struct IdxBufs {  // one path's scan keys and selections
  int rows, kstride;
  Dev<uint16_t> keys, bkeys, tkey, pkey;
  Dev<int> top, pool;
  IdxBufs(int rows_, int kstride_, int top_stride, int pool_stride)
      : rows(rows_), kstride(kstride_), keys((size_t)rows_ * kstride_), bkeys((size_t)rows_ * kstride_ / 8),
        tkey((size_t)rows_ * top_stride), pkey((size_t)rows_ * pool_stride), top((size_t)rows_ * top_stride),
        pool((size_t)rows_ * pool_stride) {}
};

// the scan of a dense part (or a block list) for ratio-`ratio` keys of a context of `ctx` positions
struct IdxCase {
  int rows, ratio, ctx, max_scan;
  attn::IndexScan scan;
  Dev<int> pos, blocks;
  Dev<float2> tab;
  Dev<uint8_t> ik_k, ik_sc;
  Dev<uint16_t> iq;
  Dev<float> wp;
  IdxCase(int rows_, int ratio_, int ctx_, int part, bool use_blocks) : rows(rows_), ratio(ratio_), ctx(ctx_) {
    pos = positions(rows, ctx - 1);
    tab = table(ctx + 1, true);
    int cap = ctx / ratio + 64;
    ik_k = Dev<uint8_t>((size_t)cap * 64);
    ik_sc = Dev<uint8_t>((size_t)cap * 4);
    iq = Dev<uint16_t>((size_t)rows * attn::kIdxHeads * attn::kIdxDim);
    wp = Dev<float>(rows * attn::kIdxHeads);
    fill::u8(ik_k, 0xFF, 0);
    fill::u8(ik_sc, 7, 120);
    fill::bf16(iq, 2.f);
    fill::f32(wp, 1.f);
    if (use_blocks) {  // each row's pool: ascending distinct blocks it can see, -1 padded
      std::mt19937 rng(ctx + rows);
      std::vector<int> h((size_t)rows * kPool, -1);
      for (int r = 0; r < rows; ++r) {
        int nb = ((ctx - rows + r + 1) / ratio + 7) / 8;
        std::vector<int> all(nb);
        for (int b = 0; b < nb; ++b) all[b] = b;
        std::shuffle(all.begin(), all.end(), rng);
        int k = std::min(nb, kPool - 3 * r);
        std::sort(all.begin(), all.begin() + k);
        for (int i = 0; i < k; ++i) h[(size_t)r * kPool + i] = all[i];
      }
      blocks = Dev<int>(h.data(), h.size());
      scan.blocks = blocks.p;
      scan.nblocks = kPool;
      max_scan = kPool * 8;
    } else if (part < 0) {  // the whole scan, sized for the context's 4096-position bucket as the runner's
      max_scan = (ctx + 4095) / 4096 * 4096 / ratio;
    } else {
      scan.part = part;
      scan.nparts = kParts;
      int n = ctx / ratio;
      max_scan = ((n + kParts - 1) / kParts + 63) / 64 * 64;
    }
  }
  int kstride() const { return (max_scan + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign; }
  attn::IndexKeys ik() const { return {ik_k.p, ik_sc.p, 0}; }
};

// separate: q prep, scores, top-k (+ pool) into slot `slot` of the outputs (stride top_stride / pool_stride)
void index_separate(const IdxCase& c, IdxBufs& b, bool pool, int slot, int top_stride, int pool_stride) {
  Dev<uint8_t> q((size_t)c.rows * attn::kIdxHeads * 64);
  Dev<uint32_t> qsc(c.rows * attn::kIdxHeads);
  Dev<float> qw(c.rows * attn::kIdxHeads);
  attn::IndexQ iq{q.p, qsc.p, qw.p};
  attn::index_q_prep(c.iq.p, c.wp.p, c.pos.p, c.tab.p, iq, nullptr, c.rows, g_s);
  attn::index_score(iq, c.pos.p, c.rows, c.ratio, c.ik(), c.scan, c.max_scan, b.keys.p, b.kstride,
                    pool ? b.bkeys.p : nullptr, b.kstride / 8, g_s);
  attn::index_topk(b.keys.p, b.kstride, c.pos.p, c.rows, c.ratio, c.scan, c.max_scan, false, kTopk,
                   b.top.p + slot * kTopk, b.tkey.p + slot * kTopk, top_stride, g_s);
  if (pool)
    attn::index_topk(b.bkeys.p, b.kstride / 8, c.pos.p, c.rows, c.ratio, c.scan, c.max_scan, true, kPool,
                     b.pool.p + slot * kPool, b.pkey.p + slot * kPool, pool_stride, g_s);
  CUDA_CHECK(cudaStreamSynchronize(g_s));
}
void index_fused(const IdxCase& c, IdxBufs& b, bool pool, int slot, int top_stride, int pool_stride, int* counters) {
  attn::IndexSel top{kTopk, b.top.p + slot * kTopk, b.tkey.p + slot * kTopk, top_stride}, ps{};
  if (pool) ps = attn::IndexSel{kPool, b.pool.p + slot * kPool, b.pkey.p + slot * kPool, pool_stride};
  attn::index_fused(c.iq.p, c.wp.p, c.pos.p, c.tab.p, c.rows, c.ratio, c.ik(), c.scan, c.max_scan, b.keys.p,
                    b.kstride, pool ? b.bkeys.p : nullptr, b.kstride / 8, top, ps, counters, true, g_s);
}

void test_index(int rows, int ratio, int ctx, int part, bool pool, bool blocks) {
  begin("index_fused rows=" + std::to_string(rows) + " ratio=" + std::to_string(ratio) + " ctx=" + std::to_string(ctx) +
        (blocks ? " blocks" : " part=" + std::to_string(part)) + (pool ? " +pool" : ""));
  IdxCase c(rows, ratio, ctx, part, blocks);
  sync();
  IdxBufs u(rows, c.kstride(), kTopk, kPool), f(rows, c.kstride(), kTopk, kPool);
  Dev<int> counters(attn::kMaxDecodeRows);
  index_separate(c, u, pool, 0, kTopk, kPool);
  for (int rep = 0; rep < 2; ++rep) index_fused(c, f, pool, 0, kTopk, kPool, counters.p);  // counters reset
  same("keys", u.keys, f.keys);
  same("top", u.top, f.top);
  same("top keys", u.tkey, f.tkey);
  if (pool) {
    same("block maxima", u.bkeys, f.bkeys);
    same("pool", u.pool, f.pool);
    same("pool keys", u.pkey, f.pkey);
  }
  same("counters left zero", counters, Dev<int>(attn::kMaxDecodeRows));
  end();
}

// index_dense (the whole scan, all rows in one grid) against the separate kernels; cand: its candidate room
void test_dense(int rows, int ratio, int ctx, bool pool, int cand) {
  begin("index_dense rows=" + std::to_string(rows) + " ratio=" + std::to_string(ratio) + " ctx=" + std::to_string(ctx) +
        " cand=" + std::to_string(cand) + (pool ? " +pool" : ""));
  IdxCase c(rows, ratio, ctx, -1, false);
  sync();
  IdxBufs u(rows, c.kstride(), kTopk, kPool), f(rows, c.kstride(), kTopk, kPool);
  Dev<uint32_t> scratch(attn::index_dense_scratch() / 4);
  index_separate(c, u, pool, 0, kTopk, kPool);
  attn::IndexSel top{kTopk, f.top.p, f.tkey.p, kTopk}, ps{};
  if (pool) ps = attn::IndexSel{kPool, f.pool.p, f.pkey.p, kPool};
  for (int rep = 0; rep < 2; ++rep)  // the scratch is left zero
    attn::index_dense(c.iq.p, c.wp.p, c.pos.p, c.tab.p, rows, ratio, c.ik(), c.max_scan, f.keys.p, f.kstride,
                      f.bkeys.p, f.kstride / 8, top, ps, scratch.p, true, g_s, cand);
  same("keys", u.keys, f.keys);
  same("top", u.top, f.top);
  same("top keys", u.tkey, f.tkey);
  if (pool) {
    same("block maxima", u.bkeys, f.bkeys);
    same("pool", u.pool, f.pool);
    same("pool keys", u.pkey, f.pkey);
  }
  if (rows > attn::kDecodeGroup) {  // rows [0, 8) and [8, rows) in calls of their own
    IdxBufs g(rows, c.kstride(), kTopk, kPool);
    for (int r0 = 0; r0 < rows; r0 += attn::kDecodeGroup) {
      attn::IndexSel gt{kTopk, g.top.p + (size_t)r0 * kTopk, g.tkey.p + (size_t)r0 * kTopk, kTopk}, gp{};
      if (pool) gp = attn::IndexSel{kPool, g.pool.p + (size_t)r0 * kPool, g.pkey.p + (size_t)r0 * kPool, kPool};
      attn::index_dense(c.iq.p + (size_t)r0 * attn::kIdxHeads * attn::kIdxDim, c.wp.p + (size_t)r0 * attn::kIdxHeads,
                        c.pos.p + r0, c.tab.p, std::min(attn::kDecodeGroup, rows - r0), ratio, c.ik(), c.max_scan,
                        g.keys.p + (size_t)r0 * g.kstride, g.kstride, g.bkeys.p + (size_t)r0 * (g.kstride / 8),
                        g.kstride / 8, gt, gp, scratch.p, true, g_s, cand);
    }
    same("keys as calls of 8 rows", f.keys, g.keys);
    same("top as calls of 8 rows", f.top, g.top);
    same("top keys as calls of 8 rows", f.tkey, g.tkey);
    if (pool) {
      same("block maxima as calls of 8 rows", f.bkeys, g.bkeys);
      same("pool as calls of 8 rows", f.pool, g.pool);
      same("pool keys as calls of 8 rows", f.pkey, g.pkey);
    }
  }
  same("scratch left zero", scratch, Dev<uint32_t>(scratch.n));
  end();
}

// index_dense_blocks (a pool scan on index_dense's grid) against the separate kernels; cand: its candidate room
void test_dense_blocks(int rows, int ctx, int cand) {
  begin("index_dense_blocks rows=" + std::to_string(rows) + " ctx=" + std::to_string(ctx) + " cand=" + std::to_string(cand));
  IdxCase c(rows, 1, ctx, -1, true);
  sync();
  IdxBufs u(rows, c.kstride(), kTopk, kPool), f(rows, c.kstride(), kTopk, kPool);
  Dev<uint32_t> scratch(attn::index_dense_scratch() / 4);
  index_separate(c, u, false, 0, kTopk, kPool);
  attn::IndexSel top{kTopk, f.top.p, f.tkey.p, kTopk};
  for (int rep = 0; rep < 2; ++rep)  // the scratch is left zero
    attn::index_dense_blocks(c.iq.p, c.wp.p, c.pos.p, c.tab.p, rows, 1, c.ik(), c.scan.blocks, c.scan.nblocks, f.keys.p,
                             f.kstride, top, scratch.p, true, g_s, cand);
  same("keys", u.keys, f.keys);
  same("top", u.top, f.top);
  same("top keys", u.tkey, f.tkey);
  if (rows > attn::kDecodeGroup) {  // rows [0, 8) and [8, rows) in calls of their own
    IdxBufs g(rows, c.kstride(), kTopk, kPool);
    for (int r0 = 0; r0 < rows; r0 += attn::kDecodeGroup) {
      attn::IndexSel gt{kTopk, g.top.p + (size_t)r0 * kTopk, g.tkey.p + (size_t)r0 * kTopk, kTopk};
      attn::index_dense_blocks(c.iq.p + (size_t)r0 * attn::kIdxHeads * attn::kIdxDim,
                               c.wp.p + (size_t)r0 * attn::kIdxHeads, c.pos.p + r0, c.tab.p,
                               std::min(attn::kDecodeGroup, rows - r0), 1, c.ik(),
                               c.scan.blocks + (size_t)r0 * c.scan.nblocks, c.scan.nblocks,
                               g.keys.p + (size_t)r0 * g.kstride, g.kstride, gt, scratch.p, true, g_s, cand);
    }
    same("keys as calls of 8 rows", f.keys, g.keys);
    same("top as calls of 8 rows", f.top, g.top);
    same("top keys as calls of 8 rows", f.tkey, g.tkey);
  }
  same("scratch left zero", scratch, Dev<uint32_t>(scratch.n));
  end();
}

void test_merge(int rows) {
  begin("index_merge2 rows=" + std::to_string(rows));
  std::mt19937 rng(rows);
  auto cands = [&](int per, int span, std::vector<int>& p, std::vector<uint16_t>& k) {
    p.assign((size_t)rows * kParts * per, -1);
    k.assign(p.size(), 0);
    for (int r = 0; r < rows; ++r)
      for (int part = 0; part < kParts; ++part) {
        int n = per - (int)(rng() % 64) * (part == 3);  // the last part may run short
        std::vector<int> all(span);
        for (int i = 0; i < span; ++i) all[i] = part * span + i;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + n);
        for (int i = 0; i < n; ++i) {
          size_t o = ((size_t)r * kParts + part) * per + i;
          p[o] = all[i];
          k[o] = (uint16_t)(0x8000 + rng() % 2048);  // plenty of ties
        }
      }
  };
  std::vector<int> hp, hpp;
  std::vector<uint16_t> hk, hpk;
  cands(kTopk, 3000, hp, hk);
  cands(kPool, 5000, hpp, hpk);
  Dev<int> cp(hp.data(), hp.size()), pcp(hpp.data(), hpp.size());
  Dev<uint16_t> ck(hk.data(), hk.size()), pck(hpk.data(), hpk.size());
  Dev<int> tu(rows * kTopk), tf(rows * kTopk), pu(rows * kPool), pf(rows * kPool);
  Dev<uint16_t> tku(rows * kTopk), tkf(rows * kTopk), pku(rows * kPool), pkf(rows * kPool);
  attn::index_merge(cp.p, ck.p, kParts * kTopk, rows, kTopk, tu.p, tku.p, kTopk, g_s);
  attn::index_merge(pcp.p, pck.p, kParts * kPool, rows, kPool, pu.p, pku.p, kPool, g_s);
  attn::index_merge2(rows, cp.p, ck.p, kParts * kTopk, {kTopk, tf.p, tkf.p, kTopk}, pcp.p, pck.p, kParts * kPool,
                     {kPool, pf.p, pkf.p, kPool}, true, g_s);
  same("top", tu, tf);
  same("top keys", tku, tkf);
  same("pool", pu, pf);
  same("pool keys", pku, pkf);
  end();
}

// ---- attention ------------------------------------------------------------------------------------

void test_attn(int rows, int kmax, int splits) {
  begin("sparse_attn PDL rows=" + std::to_string(rows) + " kmax=" + std::to_string(kmax) +
        " splits=" + std::to_string(splits));
  const int ctx = 5000, cap = ctx / 2 + 64;
  auto pos = positions(rows, ctx - 1);
  auto tab = table(ctx + 1, kmax > 0);
  Ring ring;
  Dev<uint8_t> ckv_kv((size_t)cap * kHead / 2), ckv_sc((size_t)cap * 32);
  Dev<uint16_t> q((size_t)rows * attn::kHeads * kHead);
  Dev<float> sink(attn::kHeads);
  fill::u8(ckv_kv, 0xFF, 0);
  fill::u8(ckv_sc, 0x1F, 0x28);
  fill::bf16(q, 1.f);
  fill::f32(sink, 1.f);
  std::mt19937 rng(rows * 7 + kmax);
  std::vector<int> ht((size_t)rows * std::max(kmax, 1), -1);
  for (int r = 0; r < rows; ++r) {
    int n = (ctx - rows + r + 1) / 2, k = std::min(kmax - 37 * r, n);
    std::vector<int> all(n);
    for (int i = 0; i < n; ++i) all[i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    std::sort(all.begin(), all.begin() + std::max(k, 0));
    for (int i = 0; i < k; ++i) ht[(size_t)r * kmax + i] = all[i];
  }
  Dev<int> topk(ht.data(), ht.size());
  sync();
  const int max_splits = 16;
  Dev<float> po((size_t)rows * max_splits * attn::kHeads * kHead), pml((size_t)rows * max_splits * attn::kHeads * 2);
  attn::AttnScratch scratch{po.p, pml.p, max_splits};
  attn::Compressed ckv{ckv_kv.p, ckv_sc.p, cap};
  Dev<uint16_t> ou((size_t)rows * attn::kHeads * kHead), of((size_t)rows * attn::kHeads * kHead);
  attn::sparse_attn(q.p, pos.p, {}, rows, tab.p, ring.win(), ckv, topk.p, kmax, sink.p, ou.p, splits, scratch, g_s);
  attn::sparse_attn(q.p, pos.p, {}, rows, tab.p, ring.win(), ckv, topk.p, kmax, sink.p, of.p, splits, scratch, g_s,
                    true);
  same("out", ou, of);
  end();
}

// ---- whole layers, chained with PDL ---------------------------------------------------------------

// A kv-source Full layer of a decode step (layer 2: ratio 2; layer 20: ratio 1 with the pool, then a Reindex
// layer on the merged pool) from the previous sublayer's output to the attention output, every fused kernel
// launched with PDL right after the previous one; the separate kernels on copies of the same state.
void test_chain(int rows, int ratio, int ctx) {
  begin("chain rows=" + std::to_string(rows) + " ratio=" + std::to_string(ratio) + " ctx=" + std::to_string(ctx));
  const bool pool = ratio == 1;
  const int nch = attn::hc_chunks(0, kHcDim), last = ctx - 1;
  // inputs: the stream and the previous sublayer, mHC weights, the gathered projections (from the GEMVs
  // and the other ranks), the other ranks' candidates
  Dev<uint16_t> x((size_t)rows * kHcDim), sub((size_t)rows * kDim), nw(kDim), qnw(attn::kQLora), kvw(kHead);
  Dev<float> post(rows * 4), comb(rows * 16), fn((size_t)attn::kMix * kHcDim), scale(3), base(attn::kMix);
  Dev<uint16_t> qa((size_t)rows * attn::kQLora), kvp((size_t)rows * kHead), wproj((size_t)attn::kIdxHeads * kDim);
  Dev<float> proj((size_t)rows * kHead), score((size_t)rows * kHead), sink(attn::kHeads), wsink(attn::kHeads);
  Dev<uint16_t> qpre((size_t)rows * attn::kHeads * kHead);
  for (auto* d : {&x, &sub, &qa, &kvp, &qpre}) fill::bf16(*d, 1.f);
  for (auto* d : {&nw, &qnw, &kvw}) fill::bf16_range(*d, 0.5f, 1.5f);
  fill::bf16(wproj, 0.02f);
  fill::f32_range(post, 0.f, 2.f);
  fill::f32_range(comb, 0.f, 1.f);
  fill::f32(fn, 0.01f);
  fill::f32_range(scale, 0.5f, 1.5f);
  fill::f32(base, 1.f);
  fill::f32(proj, 1.f);
  fill::f32(score, 1.f);
  fill::f32(sink, 1.f);
  auto pos = positions(rows, last);
  auto tab = table(ctx + 1, true);
  Ring ring_u;
  CompState cu(ratio, ctx + 1);
  IdxCase full(rows, ratio, ctx, 0, false);  // its keys/iq/wp are replaced by the chain's
  sync();
  Ring ring_f(ring_u);
  CompState cf = cu.copy();
  const int ts = kParts * kTopk, pls = kParts * kPool;
  IdxBufs iu(rows, full.kstride(), ts, pls), ifu(rows, full.kstride(), ts, pls);
  Dev<int> top_u(rows * kTopk), top_f(rows * kTopk), pool_u(rows * kPool), pool_f(rows * kPool);
  Dev<uint16_t> tk_u(rows * kTopk), tk_f(rows * kTopk), pk_u(rows * kPool), pk_f(rows * kPool);
  Dev<int> counters(attn::kMaxDecodeRows);
  const int max_splits = 16;
  Dev<float> po((size_t)rows * max_splits * attn::kHeads * kHead), pml((size_t)rows * max_splits * attn::kHeads * 2);
  attn::AttnScratch scratch{po.p, pml.p, max_splits};
  // the index source's keys: this layer's own cache (with the entry the emit of this step writes)
  IdxCase& c = full;
  attn::IndexKeys ik_u = cu.ik(), ik_f = cf.ik();
  // ---- separate kernels
  Dev<uint16_t> y((size_t)rows * kHcDim), yn((size_t)rows * kDim), qr((size_t)rows * attn::kQLora);
  Dev<float> chunks((size_t)nch * rows * 25), part(rows * 25), pre(rows * 4), po_(rows * 4), cb(rows * 16);
  Dev<uint8_t> xq((size_t)rows * kDim), xs((size_t)rows * kDim / 32), qq((size_t)rows * attn::kQLora),
      qs((size_t)rows * attn::kQLora / 32);
  Dev<float> wp(rows * attn::kIdxHeads);
  Dev<uint16_t> out_u((size_t)rows * attn::kHeads * kHead), out_r_u((size_t)rows * attn::kHeads * kHead);
  attn::hc_post(sub.p, x.p, post.p, comb.p, rows, y.p, g_s);
  attn::hc_partial(y.p, rows, fn.p, 0, kHcDim, chunks.p, part.p, g_s);
  attn::hc_finish(part.p, rows, scale.p, base.p, pre.p, po_.p, cb.p, g_s);
  attn::hc_pre(y.p, pre.p, rows, nw.p, yn.p, nullptr, g_s);
  dsk::act_quant(bf(yn.p), rows, kDim, xq.p, xs.p, g_s);
  attn::q_norm(qa.p, qnw.p, qr.p, rows, g_s);
  dsk::act_quant(bf(qr.p), rows, attn::kQLora, qq.p, qs.p, g_s);
  attn::kv_write(kvp.p, kvw.p, pos.p, tab.p, ring_u.win(), nullptr, rows, g_s);
  attn::compressor_emit(proj.p, score.p, pos.p, rows, cu.cw(nullptr, nullptr), cu.carry(), tab.p, cu.ckv(), ik_u,
                        nullptr, 0, g_s);
  attn::gemv_bf16(yn.p, rows, kDim, wproj.p, nullptr, 32, 32, wp.p, nullptr, true, g_s);
  CUDA_CHECK(cudaMemcpyAsync(c.wp.p, wp.p, wp.n * 4, cudaMemcpyDeviceToDevice, g_s));
  c.ik_k = clone(cu.ik_k);  // with this step's entry
  c.ik_sc = clone(cu.ik_sc);
  for (int prt = kParts - 1; prt >= 0; --prt) {  // the other ranks' parts first: inputs of the merge
    c.scan.part = prt;
    index_separate(c, iu, pool, prt, ts, pls);
  }
  c.scan.part = 0;
  attn::index_merge(iu.top.p, iu.tkey.p, ts, rows, kTopk, top_u.p, tk_u.p, kTopk, g_s);
  if (pool) attn::index_merge(iu.pool.p, iu.pkey.p, pls, rows, kPool, pool_u.p, pk_u.p, kPool, g_s);
  attn::sparse_attn(qpre.p, pos.p, {}, rows, tab.p, ring_u.win(), cu.ckv(), top_u.p, kTopk, sink.p, out_u.p, 10,
                    scratch, g_s);
  // Reindex (layer 20's pool): scores of each row's pool blocks, then attention on that top-k
  IdxCase re(rows, ratio, ctx, 0, false);
  IdxBufs ru(rows, (kPool * 8 + 255) / 256 * 256, kTopk, kPool), rf(rows, (kPool * 8 + 255) / 256 * 256, kTopk, kPool);
  if (pool) {
    re.ik_k = clone(cu.ik_k);
    re.ik_sc = clone(cu.ik_sc);
    re.scan = attn::IndexScan{};
    re.scan.blocks = pool_u.p;
    re.scan.nblocks = kPool;
    re.max_scan = kPool * 8;
    index_separate(re, ru, false, 0, kTopk, kPool);
    attn::sparse_attn(qpre.p, pos.p, {}, rows, tab.p, ring_u.win(), cu.ckv(), ru.top.p, kTopk, wsink.p, out_r_u.p, 10,
                      scratch, g_s);
  }
  // ---- fused, PDL back to back (the other ranks' candidates copied in first)
  CUDA_CHECK(cudaMemcpyAsync(ifu.top.p, iu.top.p, iu.top.n * 4, cudaMemcpyDeviceToDevice, g_s));
  CUDA_CHECK(cudaMemcpyAsync(ifu.tkey.p, iu.tkey.p, iu.tkey.n * 2, cudaMemcpyDeviceToDevice, g_s));
  CUDA_CHECK(cudaMemcpyAsync(ifu.pool.p, iu.pool.p, iu.pool.n * 4, cudaMemcpyDeviceToDevice, g_s));
  CUDA_CHECK(cudaMemcpyAsync(ifu.pkey.p, iu.pkey.p, iu.pkey.n * 2, cudaMemcpyDeviceToDevice, g_s));
  CUDA_CHECK(cudaMemset2DAsync(ifu.top.p, (size_t)ts * 4, 0x7f, (size_t)kTopk * 4, rows, g_s));  // slot 0: rewritten
  Dev<uint16_t> y2((size_t)rows * kHcDim), yn2((size_t)rows * kDim), qr2((size_t)rows * attn::kQLora);
  Dev<float> chunks2((size_t)nch * rows * 25), pre2(rows * 4), po2(rows * 4), cb2(rows * 16), wp2(rows * attn::kIdxHeads);
  Dev<uint8_t> xq2((size_t)rows * kDim), xs2((size_t)rows * kDim / 32), qq2((size_t)rows * attn::kQLora),
      qs2((size_t)rows * attn::kQLora / 32);
  Dev<uint16_t> out_f((size_t)rows * attn::kHeads * kHead), out_r_f((size_t)rows * attn::kHeads * kHead);
  attn::hc_post(sub.p, x.p, post.p, comb.p, rows, y2.p, g_s, true);
  attn::hc_project(y2.p, rows, fn.p, 0, kHcDim, chunks2.p, true, g_s);
  attn::hc_mix(y2.p, chunks2.p, nch, rows, scale.p, base.p, nw.p, pre2.p, po2.p, cb2.p, yn2.p, xq2.p, xs2.p, true, g_s);
  attn::DecodePrep a;
  a.pos = pos.p;
  a.table = tab.p;
  a.qa = qa.p;
  a.q_norm = qnw.p;
  a.qq = qq2.p;
  a.qs = qs2.p;
  a.qr = qr2.p;
  a.kv_pre = kvp.p;
  a.kv_norm = kvw.p;
  a.win = ring_f.win();
  a.ckv_proj = proj.p;
  a.cscore = score.p;
  a.cw = cf.cw(nullptr, nullptr);
  a.carry = cf.carry();
  a.ckv = cf.ckv();
  a.ik = ik_f;
  a.x = yn2.p;
  a.wproj = wproj.p;
  a.wp = wp2.p;
  attn::decode_prep(a, rows, true, g_s);
  attn::IndexSel top{kTopk, ifu.top.p, ifu.tkey.p, ts}, psel{};
  if (pool) psel = attn::IndexSel{kPool, ifu.pool.p, ifu.pkey.p, pls};
  attn::IndexScan scan0;
  scan0.part = 0;
  scan0.nparts = kParts;
  attn::index_fused(c.iq.p, wp2.p, pos.p, tab.p, rows, ratio, ik_f, scan0, c.max_scan, ifu.keys.p, ifu.kstride,
                    pool ? ifu.bkeys.p : nullptr, ifu.kstride / 8, top, psel, counters.p, true, g_s);
  attn::index_merge2(rows, ifu.top.p, ifu.tkey.p, ts, {kTopk, top_f.p, tk_f.p, kTopk}, ifu.pool.p, ifu.pkey.p, pls,
                     pool ? attn::IndexSel{kPool, pool_f.p, pk_f.p, kPool} : attn::IndexSel{}, true, g_s);
  attn::sparse_attn(qpre.p, pos.p, {}, rows, tab.p, ring_f.win(), cf.ckv(), top_f.p, kTopk, sink.p, out_f.p, 10,
                    scratch, g_s, true);
  if (pool) {
    attn::IndexScan rs;
    rs.blocks = pool_f.p;
    rs.nblocks = kPool;
    attn::index_fused(re.iq.p, re.wp.p, pos.p, tab.p, rows, ratio, ik_f, rs, kPool * 8, rf.keys.p, rf.kstride, nullptr,
                      0, {kTopk, rf.top.p, rf.tkey.p, kTopk}, {}, counters.p, true, g_s);
    attn::sparse_attn(qpre.p, pos.p, {}, rows, tab.p, ring_f.win(), cf.ckv(), rf.top.p, kTopk, wsink.p, out_r_f.p, 10,
                      scratch, g_s, true);
  }
  same("stream", y, y2);
  same("mix y", yn, yn2);
  same("mix q", xq, xq2);
  same("mix qs", xs, xs2);
  same("pre", pre, pre2);
  same("post", po_, po2);
  same("comb", cb, cb2);
  same("qr", qr, qr2);
  same("q", qq, qq2);
  same("q scales", qs, qs2);
  same("head weights", wp, wp2);
  same("ring", ring_u.kv, ring_f.kv);
  same("ring scales", ring_u.sc, ring_f.sc);
  cu.check(cf, "compressor");
  same("candidates", iu.top, ifu.top);
  same("candidate keys", iu.tkey, ifu.tkey);
  same("top", top_u, top_f);
  same("top keys", tk_u, tk_f);
  same("attention", out_u, out_f);
  if (pool) {
    same("pool candidates", iu.pool, ifu.pool);
    same("pool", pool_u, pool_f);
    same("pool keys", pk_u, pk_f);
    same("reindex top", ru.top, rf.top);
    same("reindex attention", out_r_u, out_r_f);
  }
  end();
}

}  // namespace

int main() {
  CUDA_CHECK(cudaStreamCreate(&g_s));
  for (int rows : {1, 6, 8, 9, 12, 16}) {
    const bool one_session = rows <= attn::kDecodeGroup;  // decode_prep: at most 8 rows a session
    test_mhc(rows);
    test_dspark(rows);
    test_comp_proj(rows);
    if (one_session) {
      for (int last : {1001, 1002}) {
        test_prep(rows, 2, true, last);
        test_prep(rows, 1, true, last);
      }
      test_prep(rows, 0, false, 777);
    }
    test_index(rows, 2, 20000, 0, false, false);
    test_index(rows, 2, 20000, 3, false, false);
    test_index(rows, 1, 3000, 1, true, false);
    test_index(rows, 1, 70000, 0, true, false);
    test_index(rows, 1, 70000, 0, false, true);
    test_dense(rows, 2, 1000, false, -1);
    test_dense(rows, 2, 20000, false, -1);
    test_dense(rows, 1, 3000, true, -1);
    test_dense(rows, 1, 70000, true, -1);
    test_dense(rows, 1, 70000, true, 0);
    test_dense(rows, 1, 70000, true, 1500);
    test_dense(rows, 2, 140000, false, -1);
    test_dense(rows, 1, 140000, true, -1);
    test_dense_blocks(rows, 3000, -1);
    test_dense_blocks(rows, 70000, -1);
    test_dense_blocks(rows, 70000, 0);
    test_dense_blocks(rows, 140000, 1500);
    test_merge(rows);
    for (int splits : {10, 4}) test_attn(rows, 512, splits);
    test_attn(rows, 0, 2);
    if (one_session) {
      test_chain(rows, 2, 9000);
      test_chain(rows, 1, 41000);
    }
  }
  std::printf("%d checks, %d differ: %s\n", g_checks, g_fail, g_fail ? "FAILED" : "PASSED");
  return g_fail ? 1 : 0;
}
