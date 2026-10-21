// Prefill in chunks of many rows (model.h prefill): every layer, every row, with the prefill GEMMs
// (engine/kernels prefill.cu) and the attention kernels' multi-row forms. The arithmetic follows the
// decode step (model.cu): the same weights, sums in rank order, bf16 roundings at the same points, so
// a position's state does not depend on which path computed it beyond summation order.
//
// Sequence parallel around the residual stream (prefill.h): rank r owns rows [r q, r q + n) of the
// chunk. Its stream, the mixes and each sublayer boundary (dsk::hc_boundary_p: hc_post, the next
// sublayer's mixes, hc_pre, its norm and act_quant) cover those rows only, and a sublayer's partial
// sums arrive by a reduce-scatter by rows instead of an all-reduce. Per layer:
// - attention: wq_a, wkv (and the compressor's wkv / wgate) for this rank's rows with every rank's
//   weights (PLayer), one all-gather of the results; then on every row, as the decode step splits it:
//   the window through a linear buffer (the 127 positions before the chunk, from the ring, then the
//   chunk; the last 256 back into the ring), q_norm, wq_b (this rank's heads), the index layers' top-k
//   (the indexer for this rank's rows, all-gathered), sparse attention, wo_a; an all-to-all of wo_a's
//   outputs, so that wo_b runs on this rank's rows with every rank's slice of it (PLayer), the four
//   partial sums added in rank order as a reduce-scatter adds them; the boundary into the FFN;
// - FFN: the router for this rank's rows; one all-gather of the FFN input's act_quant and the routes;
//   the MoE over every row (this rank's 576 columns of every expert); a reduce-scatter in fp32
//   (model.py's), with the experts' combine inside it unless the MoE fuses it into its down kernel;
//   the boundary into the next layer;
// - Engram (layers 1 and 14): this rank's rows with every rank's wkv (PLayer).
//
// CED (docs/ced.md "What the engine has to do"): an encoder chunk runs layers 0-19, then layer 20's
// compressor for its rows (the decoder's global KV and index keys, the exact path's), and keeps its
// last 128 positions' layer-20 input (ced_keep). The chunk that samples then replays layers 20-39 over
// the selected bounded suffix as one chunk whose window starts at that suffix (ced_replay).
#include "model/prefill.h"
#include "model/vision_layout.h"

#include <algorithm>

#include "kernels/prefill_attn.h"
#include "model/nvtx.h"
#include "model/state.h"

namespace model {

using namespace detail;

void Model::Impl::prefill_alloc(comm::Ring* ring) {
  if (pf) return;
  size_t free0 = 0, free1 = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  pf = std::make_unique<Prefill>();
  Prefill& p = *pf;
  const int M = std::max(opt.prefill_rows, kReplay), Q = q_rows(M), M4 = 4 * Q;  // CED replays 128 rows
  p.cap = M, p.Q = Q;
  int mf = M;  // the largest chunk whose MoE fuses its combine: only those use ymoe and ysum
  while (mf > 1 && !dsk::moe_prefill_fuses(mf)) --mf;
  p.x = dalloc<bf16>((size_t)Q * kHcDim);
  p.mid = dalloc<bf16>((size_t)Q * kHcDim);
  p.y = dalloc<bf16>((size_t)Q * kDim);
  p.yraw = dalloc<bf16>((size_t)Q * kDim);
  p.sub = dalloc<bf16>((size_t)Q * kDim);
  p.a_pre = dalloc<float>((size_t)Q * kHc);
  p.a_post = dalloc<float>((size_t)Q * kHc);
  p.a_comb = dalloc<float>((size_t)Q * kHc * kHc);
  p.pm = dalloc<float>((size_t)Q * kHc);
  p.f_post = dalloc<float>((size_t)Q * kHc);
  p.f_comb = dalloc<float>((size_t)Q * kHc * kHc);
  p.ysum = dalloc<float>((size_t)q_rows(mf) * kDim);
  p.gl = dalloc<float>((size_t)Q * kExperts);
  p.wpf = dalloc<float>((size_t)Q * kIdxHeads);
  p.iq = dalloc<bf16>((size_t)Q * kIdxQ);
  p.part = dalloc<bf16>((size_t)M4 * kDim);
  p.qa = dalloc<bf16>((size_t)M4 * kQLora);
  p.kvx = dalloc<bf16>((size_t)M4 * kHead);
  p.qr = dalloc<bf16>((size_t)M * kQLora);
  p.q = dalloc<bf16>((size_t)M * kQ);
  p.o = dalloc<bf16>((size_t)M * kQ);
  p.ckvf = dalloc<float>((size_t)M4 * kHead);
  p.cscf = dalloc<float>((size_t)M4 * kHead);
  p.rwts = dalloc<float>((size_t)M4 * kTopK);
  p.ymoe = dalloc<float>((size_t)4 * q_rows(mf) * kDim);
  p.yq = dalloc<uint8_t>((size_t)M4 * kDim);
  p.ys = dalloc<uint8_t>((size_t)M4 * kDim / 32);
  p.qq = dalloc<uint8_t>((size_t)M * kQLora);
  p.qs = dalloc<uint8_t>((size_t)M * kQLora / 32);
  p.oq = dalloc<uint8_t>((size_t)M * kOPart);
  p.os = dalloc<uint8_t>((size_t)M * kOPart / 32);
  p.oqa = dalloc<uint8_t>((size_t)M4 * kOPart);
  p.osa = dalloc<uint8_t>((size_t)M4 * kOPart / 32);
  p.gath = dalloc<uint8_t>((size_t)kRanks * kDim * 2);
  p.lin_kv = dalloc<uint8_t>((size_t)(M + 127) * kHead);
  p.lin_sc = dalloc<uint8_t>((size_t)(M + 127) * kHead / 32);
  p.tok = dalloc<int>(M);
  p.pos = dalloc<int>(M);
  p.image_mask = dalloc<int>(p.Q);
  p.ridx = dalloc<int>((size_t)M4 * kTopK);
  p.counters = dalloc<int>(Q);
  p.topk_mine = dalloc<int>((size_t)Q * kIdxTopk);
  for (auto& t : p.topk) t = dalloc<int>((size_t)M4 * kIdxTopk);
  p.pool = dalloc<int>((size_t)Q * kPoolBlocks);
  p.keys = dalloc<uint16_t>(std::max((size_t)kIdxBatch * kstride, (size_t)Q * kPoolBlocks * 8));
  p.bkeys = dalloc<uint16_t>((size_t)kIdxBatch * kstride / 8);
  p.iqf = dalloc<uint8_t>((size_t)Q * kIdxHeads * 64);
  p.iqsc = dalloc<uint32_t>((size_t)Q * kIdxHeads);
  p.iqw = dalloc<float>((size_t)Q * kIdxHeads);
  p.topkey = dalloc<uint16_t>((size_t)Q * kIdxTopk);
  p.poolkey = dalloc<uint16_t>((size_t)Q * kPoolBlocks);
  p.moe_ws = dalloc<uint8_t>(dsk::moe_prefill_workspace_bytes(M, kExperts));
  {
    int lo = 0, hi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&p.cs, cudaStreamNonBlocking, hi));
    for (auto& e : p.ev_grp) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&p.ev_cs, cudaEventDisableTiming));
  }
  p.tok_h = halloc<int32_t>(M);
  p.ced_rows = dalloc<uint8_t>((size_t)kReplay * kCedRow);
  p.ced_gath = dalloc<uint8_t>((size_t)kRanks * kReplay * kCedRow);
  std::fill(p.ced_pos, p.ced_pos + kCedTailRows, -1);
  const size_t seg = (size_t)M * kEngramCols * kEngramRow;  // one rank may own every cell
  for (int set = 0; set < 2; ++set)
    for (int e = 0; e < 2; ++e) {
      p.e_send[set][e] = halloc<uint8_t>(seg);
      p.e_off[set][e] = halloc<uint32_t>((size_t)M * kEngramCols);
    }
  p.e_done = halloc<uint64_t>(4);
  p.e_wait = halloc<uint64_t>(4);
  p.e_gath = dalloc<uint8_t>(kRanks * seg);
  p.e_xq = dalloc<uint8_t>((size_t)Q * kEngramK);
  p.e_xs = dalloc<uint8_t>((size_t)Q * kEngramK / 32);
  p.e_kv = dalloc<bf16>((size_t)kRanks * Q * kEngramN);
  p.e_part = dalloc<uint8_t>((size_t)kRanks * Q * kEngramPartBytes);
  // bf16 weights the prefill GEMMs read in the GEMV's tiled layout (tile_fp8 with 2 bytes per element),
  // and every rank's slices of the weights this rank's rows need whole (PLayer)
  uint8_t* tmp = dalloc<uint8_t>((size_t)kHead * kDim * 2);  // a gathered compressor matrix before tiling
  const size_t ewb = (size_t)kEngramN * kEngramK, esb = (size_t)kEngramN / 32 * (kEngramK / 32);
  for (int L = 0; L < kLayers; ++L) {
    LayerW& w = lw[L];
    PLayer& g = p.pl[L];
    if (w.ix_wproj) {
      w.p_wproj = dalloc<uint8_t>((size_t)kIdxHeads * kDim * 2);
      dsk::tile_fp8(reinterpret_cast<const uint8_t*>(w.ix_wproj), w.p_wproj, kIdxHeads, kDim * 2, st);
    }
    for (int h = 0; h < 2; ++h) {
      w.p_hcfn[h] = reinterpret_cast<uint32_t*>(dalloc<uint8_t>(dsk::hc_fn_parts_bytes()));
      dsk::hc_fn_parts(w.hc_fn[h], w.p_hcfn[h], st);
    }
    g.wqa = dalloc<uint8_t>((size_t)kQLora * kDim);
    g.sqa = dalloc<uint8_t>((size_t)kQLora / 32 * (kDim / 32));
    g.wkv = dalloc<uint8_t>((size_t)kHead * kDim);
    g.skv = dalloc<uint8_t>((size_t)kHead / 32 * (kDim / 32));
    g.wob = dalloc<uint8_t>((size_t)kRanks * kDim * kOPart);
    g.sob = dalloc<uint8_t>((size_t)kRanks * (kDim / 32) * (kOPart / 32));
    if (kv_src(L)) {
      g.ckv = dalloc<uint8_t>((size_t)kHead * kDim * 2);
      if (w.c_wgate) g.cgate = dalloc<uint8_t>((size_t)kHead * kDim * 2);
    }
    if (engram_layer(L) && store) g.ewkv = dalloc<uint8_t>(kRanks * ewb), g.eskv = dalloc<uint8_t>(kRanks * esb);
  }
  if (ring) ring->barrier();
  for (int L = 0; L < kLayers; ++L) {
    const LayerW& w = lw[L];
    PLayer& g = p.pl[L];
    // L1 tiles of 8 rows and scale rows of 32 concatenate, so the gathered slices are the whole matrices
    const comm::AgSeg ga[6] = {{w.wqa, g.wqa, (size_t)kQaPart * kDim},
                               {w.sqa, g.sqa, (size_t)kQaPart / 32 * (kDim / 32)},
                               {w.wkv, g.wkv, (size_t)kKvPart * kDim},
                               {w.skv, g.skv, (size_t)kKvPart / 32 * (kDim / 32)},
                               {w.wob, g.wob, (size_t)kDim * kOPart},
                               {w.sob, g.sob, (size_t)(kDim / 32) * (kOPart / 32)}};
    comm::allgather_multi(ch, ga, 6, st);
    if (kv_src(L)) {  // this rank's 128 rows of each compressor matrix, gathered, then tiled
      comm::allgather(ch, w.c_wkv, tmp, (size_t)kCompPart * kDim * 2, st);
      dsk::tile_fp8(tmp, g.ckv, kHead, kDim * 2, st);
      if (w.c_wgate) {
        comm::allgather(ch, w.c_wgate, tmp, (size_t)kCompPart * kDim * 2, st);
        dsk::tile_fp8(tmp, g.cgate, kHead, kDim * 2, st);
      }
    }
    if (engram_layer(L) && store) {
      const comm::AgSeg ge[2] = {{w.e_wkv, g.ewkv, ewb}, {w.e_skv, g.eskv, esb}};
      comm::allgather_multi(ch, ge, 2, st);
    }
  }
  sync();
  CUDA_CHECK(cudaFree(tmp));
  CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  std::printf("rank %d: prefill buffers and gathered weights for %d-row chunks: %.2f GiB (%.2f GiB free)\n", rank,
              M, (double)(free0 - free1) / 1073741824.0, free1 / 1073741824.0);
}

#ifdef DSGB10_NVTX
// Profiling builds: DSGB10_ROUTE_DUMP=<dir> has rank 0 write each pass's routes (every row, after the
// FFN's all-gather) to <dir>/a<pos>.m<M>.L<LL>.i32, int32 [M][6] (bench_kernels moereal's layout). It
// synchronizes the stream once a layer: for counting the experts a turn touches, never for timing.
static const char* route_dump_dir() {
  static const char* d = std::getenv("DSGB10_ROUTE_DUMP");
  return d && *d ? d : nullptr;
}
static int g_dump_a = -1;  // the pass's first position (the replay's: P - 128)
static void dump_routes(const Prefill& p, int L, int M, cudaStream_t st) {
  std::vector<int> h((size_t)kRanks * p.rq * kTopK), rows;
  CUDA_CHECK(cudaMemcpyAsync(h.data(), p.ridx, h.size() * 4, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  for (int r = 0; r < kRanks; ++r) {
    const int n = std::max(0, std::min(p.rq, M - r * p.rq));
    rows.insert(rows.end(), h.begin() + (size_t)r * p.rq * kTopK, h.begin() + ((size_t)r * p.rq + n) * kTopK);
  }
  char path[512];
  std::snprintf(path, sizeof path, "%s/a%d.m%d.L%02d.i32", route_dump_dir(), g_dump_a, M, L);
  FILE* f = std::fopen(path, "wb");
  if (!f) die("route dump: cannot write %s", path);
  std::fwrite(rows.data(), 4, rows.size(), f);
  std::fclose(f);
}
#endif

// ---- Engram ---------------------------------------------------------------------------------------

void Model::Impl::prefill_engram(int a, int M, int set) {
  Prefill& p = *pf;
  p.e_pos0[set] = a, p.e_n[set] = M;
  if (!store) return;
  // a lookup no chunk used (a cancelled request's) may still be writing the set's buffers
  for (int i = 2 * set; i < 2 * set + 2; ++i)
    while (__atomic_load_n(&p.e_done[i], __ATOMIC_ACQUIRE) < p.e_wait[i]) {
    }
  if ((int)hist.size() < a + M) die("prefill: Engram lookups for [%d, %d) with %zu tokens", a, a + M, hist.size());
  p.ids[set].resize((size_t)M * 2 * kEngramCols);
  engram::hash_positions(*emap, hist.data(), live_of(cur_slot), a, M, p.ids[set].data());
  for (int e = 0; e < 2; ++e) {
    engram::CellMap cm;
    cm.build(store->split(), e, p.ids[set].data(), M);
    cm.offsets(p.e_off[set][e]);
    p.e_seg[set][e] = cm.seg_bytes();
    engram::Request rq;
    rq.layer = e;
    rq.ids = p.ids[set].data();
    rq.n_pos = M;
    rq.out = p.e_send[set][e];
    rq.done = reinterpret_cast<std::atomic<uint64_t>*>(&p.e_done[2 * set + e]);
    rq.seq = p.e_wait[2 * set + e] = ++e_seq;
    store->submit(rq);
  }
}

// This rank's rows: the lookups (every rank's cells, all-gathered), then every rank's quarter of the
// keys and the value with every rank's weights (one GEMM of four segments), the partial dots in the
// decode step's per-rank pieces, and the combine summing them in rank order.
void Model::Impl::engram_apply_p(int L, int M, int set) {
  NVTX_RANGE("engram L%02d", L);
  Prefill& p = *pf;
  const int e = engram_index(L);
  const LayerW& w = lw[L];
  const PLayer& g = p.pl[L];
  ops::wait_seq(&p.e_done[2 * set + e], &p.e_wait[2 * set + e], st);
  comm::allgather(ch, p.e_send[set][e], p.e_gath, p.e_seg[set][e], st);
  const int n = p.n;
  if (n == 0) return;
  ops::engram_pack(p.e_gath, p.e_off[set][e] + (size_t)p.r0 * kEngramCols, n, p.e_xq, p.e_xs, st);
  const size_t wb = (size_t)kEngramN * kEngramK, sb = (size_t)kEngramN / 32 * (kEngramK / 32);
  dsk::Fp8Seg seg[kRanks];
  for (int r = 0; r < kRanks; ++r) seg[r] = {g.ewkv + r * wb, g.eskv + r * sb, p.e_kv + (size_t)r * p.Q * kEngramN, kEngramN};
  dsk::gemm_fp8_p(p.e_xq, p.e_xs, n, seg, kRanks, kEngramK, st);
  for (int r = 0; r < kRanks; ++r)
    ops::engram_partial(p.x, p.e_kv + (size_t)r * p.Q * kEngramN, w.e_w, r, n, p.e_part + (size_t)r * n * kEngramPartBytes,
                        st);
  ops::engram_combine(p.x, p.e_part, n, p.x, st, vision_count ? p.image_mask : nullptr);
  (void)M;
}

// ---- mHC -------------------------------------------------------------------------------------------

// A sublayer boundary on this rank's M rows (model.py's single-pass mHC; every pointer at the rank's
// first row): the stream out = hc_post(sub, res, post, comb) (no sub: res itself, not copied), then for
// the next sublayer (nL, nh) its input y = norm(hc_pre(out, pre)) with yq / ys (at the rank's rows of the
// all-row arrays), and its mixes from out (pre_out: the pre of the sublayer after it). nL < 0: nothing to
// prepare (the stream changes before the next sublayer: an Engram layer; or the head's collapse, which
// takes yraw = hc_pre(out, pre) without a norm).
void Model::Impl::boundary_p(int M, const bf16* sub, const bf16* res, const float* post, const float* comb,
                             const float* pre, bf16* out, int nL, int nh, float* pre_out, float* post_out,
                             float* comb_out) {
  Prefill& p = *pf;
  dsk::HcBoundary b;
  b.sub = U(sub), b.res = U(res), b.post = post, b.comb = comb, b.pre = pre, b.y = U(out);
  b.x_out = U(p.yraw);
  if (nL >= 0) {
    const LayerW& w = lw[nL];
    b.fn_parts = w.p_hcfn[nh], b.scale = w.hc_scale[nh], b.base = w.hc_base[nh];
    b.norm_w = U(nh ? w.ffn_norm : w.attn_norm);
    b.x_out = U(p.y), b.xq = p.yq + (size_t)p.r0 * kDim, b.xs = p.ys + (size_t)p.r0 * (kDim / 32);
    b.pre_out = pre_out, b.post_out = post_out, b.comb_out = comb_out;
  }
  dsk::hc_boundary_p(b, M, st);
}

// ---- row-split pieces ------------------------------------------------------------------------------

// An index layer's top-k (layer 20: also the candidate pool) for this rank's rows, all-gathered into
// the layer's slot. On entry qq/qs hold every row's queries' act_quant and y this rank's normed input.
void Model::Impl::index_p(int L, int a, int M) {
  NVTX_RANGE("index L%02d", L);
  Prefill& p = *pf;
  const LayerW& w = lw[L];
  const int Q = p.rq, r0 = p.r0, n = p.n;
  const int ratio = ratio_of(L), slot = idx_slot(L);
  if (n > 0) {
    const dsk::Fp8Seg qs{w.ix_wqb, w.ix_sqb, p.iq, kIdxQ};
    dsk::gemm_fp8_p(p.qq + (size_t)r0 * kQLora, p.qs + (size_t)r0 * kQLora / 32, n, &qs, 1, kQLora, st);
    dsk::gemm_bf16_p(p.y, n, w.p_wproj, kIdxHeads, kDim, p.wpf, st);
    ops::round_bf16(p.wpf, (size_t)n * kIdxHeads, st);  // a bf16 F.linear, as the decode step's
    const attn::IndexKeys ik = kvs[kv_slot(kv_source(L))].keys();
    if (reindex(L)) {  // layer 20's pool: a CTA per (2048 positions, row)
      attn::IndexScan sc;
      sc.blocks = p.pool;
      sc.nblocks = kPoolBlocks;
      const attn::IndexSel top{kIdxTopk, p.topk_mine, p.topkey, kIdxTopk};
      attn::index_fused(U(p.iq), p.wpf, p.pos + r0, rope_yarn, n, ratio, ik, sc, kPoolBlocks * 8, p.keys,
                        kPoolBlocks * 8, nullptr, 0, top, attn::IndexSel{}, p.counters, false, st);
    } else {  // every position so far: the prefill scan, a CTA per 256 positions and rows, in batches of rows
      const int max_scan =
          std::min(kstride, ((a + M) / ratio + attn::kScoreAlign) / attn::kScoreAlign * attn::kScoreAlign);
      const attn::IndexQ q{p.iqf, p.iqsc, p.iqw};
      attn::index_q_prep(U(p.iq), p.wpf, p.pos + r0, rope_yarn, q, nullptr, n, st);
      dsk::index_topk_p(q, p.pos + r0, n, ratio, ik, max_scan, kIdxBatch, p.keys, kstride, kIdxTopk, p.topk_mine,
                        p.topkey, L == 20 ? p.bkeys : nullptr, kPoolBlocks, p.pool, p.poolkey, st);
    }
  }
  comm::allgather(ch, p.topk_mine, p.topk[slot], (size_t)Q * kIdxTopk * 4, st);
}

// ---- the window ------------------------------------------------------------------------------------

void Model::Impl::ring_copy(int L, int p0, int p1, int base, bool to_lin) {
  Prefill& p = *pf;
  constexpr size_t kv = kHead, sc = kHead / 32;
  for (int q = p0; q < p1;) {
    const int slot = q & (attn::kRing - 1), n = std::min(p1 - q, attn::kRing - slot);
    uint8_t* rk = ring_kv[L] + slot * kv;
    uint8_t* rs = ring_sc[L] + slot * sc;
    uint8_t* lk = p.lin_kv + (size_t)(q - base) * kv;
    uint8_t* ls = p.lin_sc + (size_t)(q - base) * sc;
    CUDA_CHECK(cudaMemcpyAsync(to_lin ? lk : rk, to_lin ? rk : lk, n * kv, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(to_lin ? ls : rs, to_lin ? rs : ls, n * sc, cudaMemcpyDeviceToDevice, st));
    q += n;
  }
}

// ---- column groups -----------------------------------------------------------------------------------

// The comm stream continues after the main stream's work so far (column group c is complete): the
// collective issued next on it overlaps what the main stream does next. The comm stream runs the
// collectives in order and the main stream issues none until rs_join, so every rank still sees one
// sequence of collectives.
static void cs_after(Prefill& p, cudaStream_t st, int c) {
  CUDA_CHECK(cudaEventRecord(p.ev_grp[c], st));
  CUDA_CHECK(cudaStreamWaitEvent(p.cs, p.ev_grp[c], 0));
}

// The main stream waits for the column groups' reduce-scatters.
static void rs_join(Prefill& p, cudaStream_t st) {
  CUDA_CHECK(cudaEventRecord(p.ev_cs, p.cs));
  CUDA_CHECK(cudaStreamWaitEvent(st, p.ev_cs, 0));
}

// ---- attention ---------------------------------------------------------------------------------------

// On entry y / yq / ys hold this rank's rows' normed input and a_pre / a_post / a_comb their mixes.
void Model::Impl::attention_p(int L, int a, int M, const std::vector<std::pair<int, uint8_t*>>& snaps, int win_lo) {
  NVTX_RANGE("attn L%02d", L);
  Prefill& p = *pf;
  const LayerW& w = lw[L];
  const PLayer& g = p.pl[L];
  const int Q = p.rq, r0 = p.r0, n = p.n;
  // this rank's rows of wq_a, wkv and the compressor (all of each), then every row by one all-gather
  const int ncomp = kv_src(L) && win_lo < 0 ? (ratio_of(L) > 1 ? 2 : 1) : 0;
  if (n > 0) {
    const dsk::Fp8Seg seg[2] = {{g.wqa, g.sqa, p.qa + (size_t)r0 * kQLora, kQLora},
                                {g.wkv, g.skv, p.kvx + (size_t)r0 * kHead, kHead}};
    dsk::gemm_fp8_p(p.yq + (size_t)r0 * kDim, p.ys + (size_t)r0 * (kDim / 32), n, seg, 2, kDim, st);
    if (ncomp) {
      dsk::gemm_bf16_p(p.y, n, g.ckv, kHead, kDim, p.ckvf + (size_t)r0 * kHead, st);
      if (ncomp > 1) dsk::gemm_bf16_p(p.y, n, g.cgate, kHead, kDim, p.cscf + (size_t)r0 * kHead, st);
      else ops::round_bf16(p.ckvf + (size_t)r0 * kHead, (size_t)n * kHead, st);  // ratio 1: model.py projects in bf16
    }
  }
  const comm::AgSeg ag[4] = {{p.qa + (size_t)r0 * kQLora, p.qa, (size_t)Q * kQLora * 2},
                             {p.kvx + (size_t)r0 * kHead, p.kvx, (size_t)Q * kHead * 2},
                             {p.ckvf + (size_t)r0 * kHead, p.ckvf, (size_t)Q * kHead * 4},
                             {p.cscf + (size_t)r0 * kHead, p.cscf, (size_t)Q * kHead * 4}};
  comm::allgather_multi(ch, ag, 2 + ncomp, st);
  // the window: the 127 positions before the chunk (none in the replay), then the chunk's own keys
  const int base = win_lo >= 0 ? win_lo : std::max(0, a - 127);
  const float2* tab = L < 2 ? rope_plain : rope_yarn;
  const attn::Window win{p.lin_kv, p.lin_sc, -1, base};
  ring_copy(L, base, a, base, true);
  attn::kv_write(U(p.kvx), U(w.kv_norm), p.pos, tab, win, nullptr, M, st);
  if (ncomp) {
    const KvSrc& k = kvs[kv_slot(L)];
    const attn::CompressorWeights cw{ratio_of(L), nullptr, nullptr, U(w.c_norm), U(w.c_wk), U(w.c_knorm)};
    attn::compressor_emit(p.ckvf, p.cscf, p.pos, M, cw, attn::CompressorCarry{k.carry_kv, k.carry_sc}, rope_yarn,
                          k.compressed(), k.keys(), nullptr, 0, st);
  }
  // queries: q_norm, act_quant, wq_b; the index layers' top-k
  attn::q_norm(U(p.qa), U(w.q_norm), U(p.qr), M, st);
  dsk::act_quant(p.qr, M, kQLora, p.qq, p.qs, st);
  const dsk::Fp8Seg qseg{w.wqb, w.sqb, p.q, kQ};
  dsk::gemm_fp8_p(p.qq, p.qs, M, &qseg, 1, kQLora, st);
  if (idx_src(L)) index_p(L, a, M);
  // attention over the window and the selected compressed entries
  const int kmax = L < 2 ? 0 : kIdxTopk;
  const int* tk = L < 2 ? nullptr : p.topk[idx_slot(index_source(L))];
  const attn::Compressed ckv = L < 2 ? attn::Compressed{nullptr, nullptr, 0} : kvs[kv_slot(kv_source(L))].compressed();
  dsk::sparse_attn_p(U(p.q), p.pos, M, tab, win, ckv, tk, kmax, w.sink, U(p.o), st, std::max(win_lo, 0));
  // the window's last 256 positions back into the ring; snapshots of it inside the chunk
  ring_copy(L, std::max(base, a + M - attn::kRing), a + M, base, false);
  for (const auto& sn : snaps) {
    const size_t off = (size_t)(sn.first - kSnapRows - base);
    CUDA_CHECK(cudaMemcpyAsync(snap_kv(sn.second, L), p.lin_kv + off * kHead, kSnapKv, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(snap_sc(sn.second, L), p.lin_sc + off * (kHead / 32), kSnapSc, cudaMemcpyDeviceToDevice, st));
  }
  // output projections: wo_a on every row (this rank's heads); then wo_b on this rank's rows with every
  // rank's slice of it, whose inputs (every rank's wo_a outputs for these rows) an all-to-all brings on
  // the comm stream while this rank's own slice computes. The bf16 partials, added in rank order, are
  // the sums a reduce-scatter of every row's partials gives, for a fifth of its traffic. Then the
  // boundary on this rank's rows: mid and the FFN's input
  dsk::gemm_woa_p(p.o, M, w.woa, w.soa, kOPart, kWoaK, kOGroupRows, nullptr, p.oq, p.os, st);
  cs_after(p, st, 0);
  const comm::A2aSeg oa[2] = {{p.oq, kOPart, p.oqa, kOPart, kOPart},
                              {p.os, kOPart / 32, p.osa, kOPart / 32, kOPart / 32}};
  comm::alltoall_rows(ch, oa, 2, Q, p.cs, kRsBlocks);
  auto wob = [&](int k, const uint8_t* xq, const uint8_t* xs) {
    const dsk::Fp8Seg seg{g.wob + (size_t)k * kDim * kOPart, g.sob + (size_t)k * (kDim / 32) * (kOPart / 32),
                          p.part + (size_t)k * Q * kDim, kDim};
    dsk::gemm_fp8_p(xq, xs, n, &seg, 1, kOPart, st);
  };
  if (n > 0) wob(rank, p.oq + (size_t)r0 * kOPart, p.os + (size_t)r0 * (kOPart / 32));
  rs_join(p, st);
  if (n > 0) {
    for (int k = 0; k < kRanks; ++k)
      if (k != rank) wob(k, p.oqa + (size_t)k * Q * kOPart, p.osa + (size_t)k * Q * (kOPart / 32));
    dsk::sum4_bf16_p(p.part, (size_t)Q * kDim, (size_t)n * kDim, p.sub, st);
    boundary_p(n, p.sub, p.x, p.a_post, p.a_comb, p.a_pre, p.mid, L, 1, p.pm, p.f_post, p.f_comb);
  }
}

// ---- FFN ---------------------------------------------------------------------------------------------

// On entry y / yq / ys hold this rank's rows' normed input and f_post / f_comb their mixes.
void Model::Impl::ffn_p(int L, int M) {
  NVTX_RANGE("ffn L%02d", L);
  Prefill& p = *pf;
  const LayerW& w = lw[L];
  const int Q = p.rq, r0 = p.r0, n = p.n;
  if (n > 0) {
    dsk::gemm_bf16_p(p.y, n, w.gate, kExperts, kDim, p.gl, st);
    dsk::router_topk(p.gl, w.gate_bias, n, kExperts, p.ridx + (size_t)r0 * kTopK, p.rwts + (size_t)r0 * kTopK,
                     kRouteScale, st, w.gate_bias_vl, vision_count ? p.image_mask : nullptr);
  }
  // every row's act_quant and routes for the MoE
  const comm::AgSeg ag[4] = {{p.yq + (size_t)r0 * kDim, p.yq, (size_t)Q * kDim},
                             {p.ys + (size_t)r0 * (kDim / 32), p.ys, (size_t)Q * (kDim / 32)},
                             {p.ridx + (size_t)r0 * kTopK, p.ridx, (size_t)Q * kTopK * 4},
                             {p.rwts + (size_t)r0 * kTopK, p.rwts, (size_t)Q * kTopK * 4}};
  comm::allgather_multi(ch, ag, 4, st);
#ifdef DSGB10_NVTX
  if (route_dump_dir() && rank == 0) dump_routes(p, L, M, st);
#endif
  // the MoE (bucketing, shared expert, up), then down in column groups. Each group's combine and its
  // reduce-scatter by rows (fp32 sums in rank order, rounded to bf16) run as one kernel on the comm
  // stream under the next group's down; for chunks whose combine runs inside the down kernel
  // (moe_prefill_fuses), each group's fp32 outputs are reduce-scattered instead
  dsk::MoePCfg mc;
  mc.phases = 1 | 2 | 4;
  dsk::moe_prefill(p.yq, p.ys, M, p.ridx, p.rwts, kExperts, w.moe, p.moe_ws, p.ymoe, st, mc);
  const bool fuses = dsk::moe_prefill_fuses(M);
  for (int c = 0; c < kColGroups; ++c) {
    mc.phases = fuses ? 8 | 16 : 8, mc.col0 = c * kGroupCols, mc.cols = kGroupCols;
    dsk::moe_prefill(p.yq, p.ys, M, p.ridx, p.rwts, kExperts, w.moe, p.moe_ws, p.ymoe, st, mc);
    cs_after(p, st, c);
    if (fuses)
      comm::reduce_scatter_rows_f32(ch, p.ymoe + mc.col0, kDim, p.ysum + mc.col0, kDim, Q, kGroupCols, p.cs,
                                    kRsBlocks);
    else
      dsk::moe_combine_rs(ch, p.moe_ws, M, kExperts, Q, mc.col0, kGroupCols, p.sub, kDim, p.cs, kCombRsBlocks);
  }
  rs_join(p, st);
  if (n == 0) return;
  if (fuses) ops::f32_to_bf16(p.ysum, p.sub, n * kDim, st);
  // the boundary: x and the next layer's attention input, unless Engram changes x first; after the last
  // layer only the head's collapse
  const int nL = L + 1 < kLayers && !(engram_layer(L + 1) && store) ? L + 1 : -1;
  boundary_p(n, p.sub, p.mid, p.f_post, p.f_comb, p.pm, L + 1 < kLayers ? p.x : nullptr, nL, 0, p.a_pre, p.a_post,
             p.a_comb);
}

// ---- CED -------------------------------------------------------------------------------------------

// Rows [i, i + cnt) of this rank's rows to (or from) consecutive records starting at rec.
static void ced_copy(Prefill& p, uint8_t* rec, int i, int cnt, bool to_rec, cudaStream_t st) {
  const struct {
    uint8_t* buf;
    size_t off, row;
  } f[7] = {{reinterpret_cast<uint8_t*>(p.x + (size_t)i * kHcDim), kCedX, (size_t)kHcDim * 2},
            {reinterpret_cast<uint8_t*>(p.y + (size_t)i * kDim), kCedY, (size_t)kDim * 2},
            {p.yq + (size_t)(p.r0 + i) * kDim, kCedYq, (size_t)kDim},
            {p.ys + (size_t)(p.r0 + i) * (kDim / 32), kCedYs, (size_t)kDim / 32},
            {reinterpret_cast<uint8_t*>(p.a_pre + (size_t)i * kHc), kCedPre, (size_t)kHc * 4},
            {reinterpret_cast<uint8_t*>(p.a_post + (size_t)i * kHc), kCedPost, (size_t)kHc * 4},
            {reinterpret_cast<uint8_t*>(p.a_comb + (size_t)i * kHc * kHc), kCedComb, (size_t)kHc * kHc * 4}};
  for (const auto& e : f) {
    if (to_rec)
      CUDA_CHECK(cudaMemcpy2DAsync(rec + e.off, kCedRow, e.buf, e.row, e.row, cnt, cudaMemcpyDeviceToDevice, st));
    else
      CUDA_CHECK(cudaMemcpy2DAsync(e.buf, e.row, rec + e.off, kCedRow, e.row, cnt, cudaMemcpyDeviceToDevice, st));
  }
}

// Layer 20's compressor for every row of an encoder chunk: on entry y holds this rank's rows' layer-20
// input (the boundary after layer 19), as the exact path's layer 20 has it.
void Model::Impl::ced_project(int M) {
  NVTX_RANGE("ced_project");
  Prefill& p = *pf;
  constexpr int L = 20;
  const LayerW& w = lw[L];
  if (p.n > 0) {
    dsk::gemm_bf16_p(p.y, p.n, p.pl[L].ckv, kHead, kDim, p.ckvf + (size_t)p.r0 * kHead, st);
    ops::round_bf16(p.ckvf + (size_t)p.r0 * kHead, (size_t)p.n * kHead, st);  // ratio 1: model.py projects in bf16
  }
  const comm::AgSeg ag{p.ckvf + (size_t)p.r0 * kHead, p.ckvf, (size_t)p.rq * kHead * 4};
  comm::allgather_multi(ch, &ag, 1, st);
  const KvSrc& k = kvs[kv_slot(L)];
  const attn::CompressorWeights cw{ratio_of(L), nullptr, nullptr, U(w.c_norm), U(w.c_wk), U(w.c_knorm)};
  attn::compressor_emit(p.ckvf, p.cscf, p.pos, M, cw, attn::CompressorCarry{k.carry_kv, k.carry_sc}, rope_yarn,
                        k.compressed(), k.keys(), nullptr, 0, st);
}

// After an encoder chunk [a, a + M): the layer-20 input of its last replay_rows positions (this rank's
// rows of them) into their slots. The caller must supply at least that many new encoder rows over
// the complete request, so a saved/forked session never depends on this transient scratch buffer.
void Model::Impl::ced_keep(int a, int M) {
  NVTX_RANGE("ced_keep");
  Prefill& p = *pf;
  const int R = p.ced_replay_rows;
  const int lo = std::max(a, a + M - R);
  for (int q = lo; q < a + M; ++q) p.ced_pos[q % R] = q, p.ced_src[q % R] = (q - a) / p.rq;
  for (int i = std::max(0, lo - a - p.r0); i < p.n;) {
    const int slot = (a + p.r0 + i) % R, cnt = std::min(p.n - i, R - slot);
    ced_copy(p, p.ced_rows + (size_t)slot * kCedRow, i, cnt, true, st);
    i += cnt;
  }
}

// Decoder replay over the request's bounded suffix. The legacy policy is 128; the tail policy is
// 2560. For the latter only the suffix beginning at ced_valid_from(P, R) is safe for later restarts.
void Model::Impl::ced_replay(int P, const std::vector<std::pair<int, uint8_t*>>& snaps) {
  NVTX_RANGE("replay P=%d", P);
  Prefill& p = *pf;
  const int R = p.ced_replay_rows, s = P - R, M = R;
#ifdef DSGB10_NVTX
  g_dump_a = s;
#endif
  for (int q = s; q < P; ++q)
    if (p.ced_pos[q % R] != q) die("model: CED replay of [%d, %d) without the encoder's position %d", s, P, q);
  p.rq = q_rows(M), p.r0 = rank * p.rq, p.n = std::max(0, std::min(p.rq, M - p.r0));
  comm::allgather(ch, p.ced_rows, p.ced_gath, (size_t)R * kCedRow, st);
  for (int i = 0; i < p.n;) {  // runs of consecutive slots from one rank
    const int slot = (s + p.r0 + i) % R, src = p.ced_src[slot];
    int cnt = 1;
    while (i + cnt < p.n && slot + cnt < R && p.ced_src[slot + cnt] == src) ++cnt;
    ced_copy(p, p.ced_gath + ((size_t)src * R + slot) * kCedRow, i, cnt, false, st);
    i += cnt;
  }
  ops::iota(p.pos, s, M, st);
  if (vision_count) ops::vision_rows(p.pos + p.r0, p.n, vision_spans, vision_count,
                                    vision_features, nullptr, p.image_mask, st);
  for (int L = 20; L < kLayers; ++L) {
    NVTX_RANGE("rep L%02d", L);
    // DSpark gathers the last 256 rows using this replay's actual sequence-parallel partition.
    if (spec) spec_prefill_hidden(L, s, M, {});  // DSpark: main_hidden of the replay's rows
    attention_p(L, s, M, snaps, s);
    ffn_p(L, M);
  }
}

// ---- a chunk ---------------------------------------------------------------------------------------

uint64_t Model::Impl::prefill(int a, int M, const Sampling* sp, int handle, const std::vector<int>& snap_pos,
                              bool ced, int replay_rows) {
  NVTX_RANGE("chunk a=%d M=%d%s%s", a, M, ced ? " ced" : "", sp ? " sample" : "");
  if (!bound) die("model: prefill before bind");
  prefill_alloc();
  Prefill& p = *pf;
  if (M < 1 || M > p.cap || a < 0 || a + M > opt.max_ctx) die("model: bad prefill [%d, +%d)", a, M);
  if ((int)hist.size() < a + M) die("model: prefill [%d, %d) with %zu tokens", a, a + M, hist.size());
  if (opt.first_layer != 0 || opt.n_layers != kLayers) die("model: prefill runs the whole model");
  if (!ced_replay_supported(replay_rows)) die("model: unsupported CED replay length %d", replay_rows);
  if (ced && replay_rows > p.cap)
    die("model: CED replay needs %d rows, but prefill capacity is %d", replay_rows, p.cap);
  if (ced && sp && a + M < replay_rows)
    die("model: CED prefill ending at %d (the replay needs %d positions)", a + M, replay_rows);
  drain();  // the decode buffers and the Engram sets are free; earlier steps' state is in place
  if (ced && replay_rows > p.ced_capacity) {
    // One bounded growth, outside capture and after draining. Do not grow ordinary prefill scratch
    // or retain a second old/new record pool. The production chunk capacity already exceeds 2560.
    sync();
    CUDA_CHECK(cudaFree(p.ced_rows));
    CUDA_CHECK(cudaFree(p.ced_gath));
    p.ced_rows = dalloc<uint8_t>((size_t)replay_rows * kCedRow);
    p.ced_gath = dalloc<uint8_t>((size_t)kRanks * replay_rows * kCedRow);
    p.ced_capacity = replay_rows;
  }
  // this chunk's Engram lookups, unless prefill_lookup() started them
  int set = -1;
  for (int k = 0; k < 2; ++k)
    if (p.e_pos0[k] == a && p.e_n[k] == M) set = k;
  if (set < 0) {  // after the drain either set is free on the device
    set = 1 - p.e_last;
    prefill_engram(a, M, set);
  }
  p.e_last = set;
  std::vector<std::pair<int, uint8_t*>> snaps;
  for (int s : snap_pos) {
    if (s <= a || s > a + M || s < kSnapRows) die("model: snapshot at %d outside (%d, %d]", s, a, a + M);
    if (uint8_t* b = state_snap_buf(handle, s)) snaps.emplace_back(s, b);  // none: the store had no room
  }
  p.rq = q_rows(M), p.r0 = rank * p.rq, p.n = std::max(0, std::min(p.rq, M - p.r0));
  const int slot = (int)(n_steps % ops::kOutSlots);
  CUDA_CHECK(cudaEventRecord(ev_beg[slot], st));
  CUDA_CHECK(cudaEventRecord(ev_pf[0], st));
  std::memcpy(p.tok_h, hist.data() + a, (size_t)M * 4);
  CUDA_CHECK(cudaMemcpyAsync(p.tok, p.tok_h, (size_t)M * 4, cudaMemcpyHostToDevice, st));
  ops::iota(p.pos, a, M, st);
  if (p.n > 0) ops::embed(embed, p.tok + p.r0, p.n, p.x, p.pm, st);
  // Always initialize the current chunk's mask when the request has images: an all-text chunk
  // after an image chunk must not inherit its old router/Engram mask.
  if (vision_count) ops::vision_rows(p.pos + p.r0, p.n, vision_feature_spans, vision_feature_count,
                                    vision_features, p.x, p.image_mask, st);
  if (ced && (a != p.ced_end || p.ced_replay_rows != replay_rows))
    std::fill(p.ced_pos, p.ced_pos + kCedTailRows, -1);  // a new run or policy
  p.ced_replay_rows = replay_rows;
  p.ced_end = ced ? a + M : -1;
#ifdef DSGB10_NVTX
  g_dump_a = a;
#endif
  for (int L = 0; L < (ced ? 20 : kLayers); ++L) {
    NVTX_RANGE("%s L%02d", ced ? "enc" : "ex", L);
    bool fresh = L == 0;
    if (engram_layer(L) && store) {
      engram_apply_p(L, M, set);
      fresh = true;
    }
    if (fresh && p.n > 0)
      boundary_p(p.n, nullptr, p.x, nullptr, nullptr, p.pm, nullptr, L, 0, p.a_pre, p.a_post, p.a_comb);
    if (spec) spec_prefill_hidden(L, a, M, snap_pos);  // DSpark: main_hidden of the rows its windows take
    attention_p(L, a, M, snaps);
    if ((opt.wave_probe >> L & 1) && a == opt.wave_probe_at) wave_probe(L, a, M);
    ffn_p(L, M);
  }
  p.e_pos0[set] = -1;  // used
  CUDA_CHECK(cudaEventRecord(ev_pf[1], st));
  if (ced) {
    ced_project(M);
    ced_keep(a, M);
    if (sp) {  // the replay; of the snapshots only one at its end covers the decoder layers
      std::vector<std::pair<int, uint8_t*>> at_end;
      for (const auto& sn : snaps)
        if (sn.first == a + M) at_end.push_back(sn);
      ced_replay(a + M, at_end);
    }
  }
  CUDA_CHECK(cudaEventRecord(ev_pf[2], st));
  const int Mh = ced ? replay_rows : M;  // the rows of the pass the head reads
  uint64_t n = ~0ull;
  if (sp) {  // the last row through the head and the sampling, as a step of one row
    NVTX_RANGE("head");
    // the last row's collapsed stream, from the rank that owns it
    comm::allgather(ch, p.yraw + (size_t)std::max(p.n - 1, 0) * kDim, p.gath, (size_t)kDim * 2, st);
    const bf16* last = reinterpret_cast<const bf16*>(p.gath) + (size_t)((Mh - 1) / p.rq) * kDim;
    n = n_steps;
    if (n >= (uint64_t)ops::kInSlots) wait_done(n - ops::kInSlots);
    const int32_t t = hist[a + M - 1];
    next_in(n, 1, a + M - 1, &t, *sp);
    ops::step_in(in_h, in_ctr, fb, in_d, st);
    if (head_coded) dsk::gemv_bf16x_norm(last, norm_w, kNormEps, 1, head, kVocabPart, kDim, logits, nullptr, chead, st);
    else dsk::gemv_bf16_norm(last, norm_w, kNormEps, 1, head, kVocabPart, kDim, logits, nullptr, chead, st);
    ops::sample(ch, logits, kVocabPart * rank, in_d, ctl_h, out_h, fb, samp_ctr, st);
    n_steps = n + 1;
  }
  CUDA_CHECK(cudaEventRecord(ev_pf[3], st));
  if (spec) spec_prefill_accept(a, M, handle, snap_pos, ced, sp);  // into the draft module's windows
  CUDA_CHECK(cudaEventRecord(ev_end[slot], st));
  CUDA_CHECK(cudaEventRecord(ev_pf[4], st));
  return n;
}

// ---- Model -----------------------------------------------------------------------------------------

void Model::set_vision_layout(const std::vector<ops::VisionSpan>& spans) {
  auto& d = *d_;
  d.drain();  // no prior request/graph/lookup reads the buffers or host mask after this boundary
  if (!detail::valid_vision_layout(spans, d.opt.max_ctx)) die("model: bad vision layout");
  auto& live = d.vision_live[d.cur_slot];
  detail::vision_live_layout(live, spans, d.opt.max_ctx);
  if (!spans.empty()) {
    if (spans.size() > size_t(d.vision_span_capacity)) {
      if (d.vision_spans) CUDA_CHECK(cudaFree(d.vision_spans));
      d.vision_spans = dalloc<ops::VisionSpan>(spans.size());
      d.vision_span_capacity = (int)spans.size();
    }
    CUDA_CHECK(cudaMemcpyAsync(d.vision_spans, spans.data(), spans.size() * sizeof(spans[0]), cudaMemcpyHostToDevice, d.st));
    CUDA_CHECK(cudaStreamSynchronize(d.st));
  }
  d.vision_count = (int)spans.size();
  d.vision_feature_count = 0;  // a new request must explicitly install each chunk's features
  // A cancelled request may have prefetched the same position in the alternate Engram set.
  if (d.pf) for (int k = 0; k < 2; ++k) d.pf->e_pos0[k] = -1, d.pf->e_n[k] = 0;
}

void Model::set_vision_chunk(const std::vector<ops::VisionSpan>& spans, const uint16_t* features, int rows) {
  auto& d = *d_;
  d.drain();  // old chunk's device feature reads have finished; host lookup mask remains unchanged
  if (!detail::valid_vision_chunk(spans, rows, d.opt.max_ctx)) die("model: bad vision chunk");
  if (!spans.empty()) {
    if (spans.size() > size_t(d.vision_feature_span_capacity)) {
      if (d.vision_feature_spans) CUDA_CHECK(cudaFree(d.vision_feature_spans));
      d.vision_feature_spans = dalloc<ops::VisionSpan>(spans.size());
      d.vision_feature_span_capacity = (int)spans.size();
    }
    if (rows > d.vision_capacity) {
      if (d.vision_features) CUDA_CHECK(cudaFree(d.vision_features));
      d.vision_features = dalloc<bf16>((size_t)rows * kDim);
      d.vision_capacity = rows;
    }
    CUDA_CHECK(cudaMemcpyAsync(d.vision_feature_spans, spans.data(), spans.size() * sizeof(spans[0]), cudaMemcpyHostToDevice, d.st));
    CUDA_CHECK(cudaMemcpyAsync(d.vision_features, features, (size_t)rows * kDim * 2, cudaMemcpyHostToDevice, d.st));
    CUDA_CHECK(cudaStreamSynchronize(d.st));
  }
  d.vision_feature_count = (int)spans.size();
  // In particular: do not clear e_pos0/e_n. run_prefill already queued the next lookup using
  // the complete immutable vision_live layout, not this feature window.
}

void Model::set_tokens(int pos0, int n, const int32_t* tokens) {
  auto& h = d_->hist;
  if (pos0 < 0 || n < 0) die("model: set_tokens(%d, %d)", pos0, n);
  h.resize(pos0 + n);
  std::copy(tokens, tokens + n, h.begin() + pos0);
}

size_t Model::prefill_init(comm::Ring& ring) {
  d_->prefill_alloc(&ring);
  size_t free = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  return free;
}

void Model::prefill_lookup(int a, int M) {
  d_->prefill_alloc();
  Prefill& p = *d_->pf;
  // not the set of the chunk in flight; the other one's chunk has finished (prefill() drains first)
  d_->prefill_engram(a, M, 1 - p.e_last);
}

uint64_t Model::prefill(int a, int M, const Sampling* sp, int handle, const std::vector<int>& snaps, bool ced,
                        int replay_rows) {
  return d_->prefill(a, M, sp, handle, snaps, ced, replay_rows);
}

}  // namespace model
