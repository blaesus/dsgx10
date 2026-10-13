// Model::Impl, shared by the model's translation units: model.cu (weights, the decode/verify step,
// launch and wait), prefill.cu (chunks of many rows), state.cu (sessions' saved state) and spec.cu
// (DSpark: the draft module and the speculative steps).
#pragma once
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attn/attn.h"
#include "comm/collectives.h"
#include "common/check.h"
#include "kernels/dsk.h"
#include "model/model.h"
#include "model/ops.h"

namespace model {
namespace detail {

using bf16 = __nv_bfloat16;
inline uint16_t* U(bf16* p) { return reinterpret_cast<uint16_t*>(p); }
inline const uint16_t* U(const bf16* p) { return reinterpret_cast<const uint16_t*>(p); }

// Zeroed before it returns: cudaMemset runs on the legacy stream, which the model's non-blocking
// streams do not wait for, so a kernel writing the buffer right after could be overtaken by the zeroing.
template <class T>
T* dalloc(size_t n) {
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
  CUDA_CHECK(cudaStreamSynchronize(0));
  return static_cast<T*>(p);
}
template <class T>
T* halloc(size_t n) {  // pinned, mapped host memory (the GPU reads it in place)
  void* p = nullptr;
  CUDA_CHECK(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped));
  std::memset(p, 0, n * sizeof(T));
  return static_cast<T*>(p);
}

constexpr int kMaxSplits = 2 + kIdxTopk / 64;  // sparse_attn split-K: 2 window blocks + 8 selected blocks
constexpr size_t kEngramSegMax = (size_t)kMaxRows * kEngramCols * kEngramRow;  // one rank owning every cell

struct LayerW {
  const uint8_t *wqa, *sqa, *wkv, *skv, *wqb, *sqb, *woa, *soa, *wob, *sob;
  const bf16 *q_norm, *kv_norm, *attn_norm, *ffn_norm;
  const float* sink;
  // compressor (kv sources): this rank's 128 rows of wkv / wgate, the norm, the index key projection
  const bf16 *c_wkv = nullptr, *c_wgate = nullptr, *c_norm = nullptr, *c_wk = nullptr, *c_knorm = nullptr;
  // indexer (index sources; replicated)
  const uint8_t *ix_wqb = nullptr, *ix_sqb = nullptr;
  const bf16* ix_wproj = nullptr;
  // FFN
  uint8_t* gate = nullptr;  // bf16 [384][5120], tile_fp8 layout: every rank's 96 rows, all-gathered at bind
  const float* gate_bias = nullptr;
  dsk::MoeWeights moe{};
  // mHC: the full projections [24][20480] (all-gathered at bind), base [24], scale [3]
  float* hc_fn[2] = {};
  const float *hc_base[2] = {}, *hc_scale[2] = {};
  // Engram
  const uint8_t *e_wkv = nullptr, *e_skv = nullptr;
  float* e_w = nullptr;  // q_weight * k_weight [4][5120]
  // prefill only (prefill.cu): bf16 weights in the GEMMs' tiled layout, made on the first prefill
  uint8_t *p_ckv = nullptr, *p_cgate = nullptr, *p_wproj = nullptr;
  uint32_t* p_hcfn[2] = {};  // hc_fn as dsk::hc_fn_parts
};

struct KvSrc {  // a kv-source layer's compressed KV, index keys and compressor carry (replicated)
  uint8_t *ckv, *csc, *ik, *iksc;
  int cap;
  float *carry_kv, *carry_sc;
  attn::Compressed compressed() const { return attn::Compressed{ckv, csc, cap}; }
  attn::IndexKeys keys() const { return attn::IndexKeys{ik, iksc, cap}; }
  // session slots: slot s's buffers are slot 0's + s strides (whole caches of cap entries; the carries' 16
  // rows of kHead floats)
  size_t ckv_stride() const { return (size_t)cap * 256; }
  size_t csc_stride() const { return (size_t)cap * 32; }
  size_t ik_stride() const { return (size_t)cap * 64; }
  size_t iksc_stride() const { return (size_t)cap * 4; }
  static constexpr size_t kCarryStride = 16 * kHead;  // floats
  KvSrc at(int s) const {
    KvSrc k = *this;
    k.ckv += s * ckv_stride(), k.csc += s * csc_stride(), k.ik += s * ik_stride(), k.iksc += s * iksc_stride();
    k.carry_kv += s * kCarryStride, k.carry_sc += s * kCarryStride;
    return k;
  }
};
constexpr size_t kRingKvBytes = (size_t)attn::kRing * kHead, kRingScBytes = (size_t)attn::kRing * kHead / 32;

struct Prefill;  // prefill.cu's buffers
struct Store;    // state.cu's sessions
struct Spec;     // spec.cu's draft module and spec steps

}  // namespace detail

struct Model::Impl {
  using bf16 = detail::bf16;
  int rank;
  Options opt;
  cudaStream_t st = nullptr;
  comm::DevChannel ch{};
  comm::DevChannel ch2{};  // --wave-probe
  bool has_ch2 = false;
  bool bound = false;

  // weights
  detail::LayerW lw[kLayers];
  const bf16 *embed = nullptr, *norm_w = nullptr;
  const uint8_t* head = nullptr;
  bool head_coded = false;  // head holds the coded head (code_head): the head GEMVs read it with gemv_bf16x*

  // state: Options::slots session slots side by side, slot s at slot 0's pointers + s strides. ring_kv,
  // ring_sc and kvs are the current slot's (use_slot), which the eager paths act on; a step reaches each
  // row's slot through slot 0's (ring_kv0, ring_sc0, kvs0) and the step's slot table (step_win and the others)
  uint8_t *ring_kv[kLayers], *ring_sc[kLayers];
  detail::KvSrc kvs[kKvSlots];
  uint8_t *ring_kv0[kLayers], *ring_sc0[kLayers];
  detail::KvSrc kvs0[kKvSlots];
  int nslots = 1, cur_slot = 0;
  float2 *rope_plain, *rope_yarn;
  int kstride;

  // step buffers
  bf16 *x, *mid, *y, *yraw, *sub, *part, *qa, *kvx, *q, *iqpre, *o, *hfin;
  float *pm[2], *a_pre, *a_post, *a_comb, *f_post, *f_comb, *hc_chunks, *amix;
  float *ckvf, *cscf, *wproj, *gl, *rwts, *ymoe, *logits, *lgath, *po, *pml;
  uint8_t *yq, *ys, *oq, *os, *pack;
  uint32_t* dense_scratch;  // index_dense's counters and histograms (zero between launches)
  uint16_t *keys, *bkeys, *topkey[kIdxSlots], *poolkey;
  int *topk[kIdxSlots], *pool, *ridx;
  void* moe_ws;

  // step inputs and results
  ops::StepIn *in_h, *in_d;  // pinned ring [kInSlots] (the host writes), device copy (kernels read)
  const int *tok_d, *pos_d;  // in_d->tokens, in_d->pos
  uint32_t* in_ctr;          // device: steps started (step_in's slot)
  int32_t* fb;               // device [kMaxRows]: the last step's sampled tokens
  ops::SampleOut* out_h;     // pinned ring [kOutSlots]
  uint32_t* ctl_h;           // pinned: rank 0's control word
  uint32_t* samp_ctr;        // device: the sampler's completion count
  uint64_t n_steps = 0;      // launched
  int pos_stride = 1;        // Model::set_pos_stride (bench)
  int pos_group = 1;         // its runs of consecutive rows
  int route_shift = 0;       // Model::set_route_shift (bench)
  int route_group = 1;       // its rows per rotation
  uint64_t n_done = 0;       // known finished
  uint64_t last_n = 0;       // the last step waited for
  cudaEvent_t ev_beg[ops::kOutSlots], ev_end[ops::kOutSlots];
  cudaEvent_t ev_pf[5];    // the last prefill chunk: begin, layers 0-19 done, replay done, head done, end
  cudaEvent_t ev_mark[4];  // Model::mark

  // captured steps: one graph per (rows, context bucket); a row count runs eagerly once first
  std::map<std::pair<int, int>, cudaGraphExec_t> graphs;
  bool warm[kMaxRows + 1] = {};
  // side streams: the mHC mixes (sa); the window/compressed KV writes and the indexer's head weights (sb)
  cudaStream_t sa = nullptr, sb = nullptr;
  cudaEvent_t ev_fork, ev_join[2];
  // Options::l2pf (kernels/l2pf.cu): the L2 prefetch's side stream and its progress
  cudaStream_t sc = nullptr;
  dsk::L2pfState* l2s = nullptr;

  // Engram
  const engram::HashMap* emap = nullptr;
  engram::Store* store = nullptr;
  std::vector<int32_t> hist;  // token ids of the sequence so far (hashing reads 3 back): the current slot's
  std::vector<int32_t> hists[kMaxSlots];  // the other slots' (use_slot swaps them in and out)
  std::vector<int32_t>& hist_of(int s) { return s == cur_slot ? hist : hists[s]; }
  std::vector<uint32_t> ids;  // [R][2][24]
  uint8_t* e_send[2];         // pinned: this rank's packed records per Engram layer
  uint32_t* e_off[2];         // pinned: [R][24] offsets into the gathered segments
  uint64_t* e_done;           // pinned: the store's completion flags [2]
  uint64_t e_seq = 0;
  uint8_t *e_gath, *e_xq, *e_xs, *e_part, *e_pgath;
  bf16* e_kv;
  struct Late {  // a launched step's Engram lookups, waiting for its tokens (engram_late)
    bool on = false;
    bool any = false;    // a verify step: its positions and rows are the device's choice
    bool batch = false;  // a batch step (engram_late_batch): its rows' positions and slots
    int pos0 = -1, R = 0;
    uint64_t seq[2] = {};
    int32_t pos[kMaxRows] = {}, slot[kMaxRows] = {};
  } late;

  // re-laying weights the file holds in their natural layout (freed after bind)
  uint8_t* scratch = nullptr;
  size_t scratch_bytes = 0;

  // golden injection / dumps
  uint8_t *inj_h, *inj_g;  // staging [R][20480] bf16 + [R][4] f32 (pinned), gathered (device)
  std::vector<uint8_t> dump_buf;

  dsk::GemvCfg c8k{1, 1, 8}, c8{1, 1, 16}, cwoa{1, 4, 4}, cgate{1, 1, 8}, chead{2, 1, 16};

  std::unique_ptr<detail::Prefill> pf;
  std::unique_ptr<detail::Store> store_s;
  std::unique_ptr<detail::Spec> spec;
  bool mh_on = false;  // capturing a spec step: main_hidden at the inputs of layers 37-39

  Impl(int r, const Options& op);
  ~Impl();

  // ---- weights ------------------------------------------------------------------------------
  void bind(dsw::WeightFile& wf, comm::Ring& ring);
  size_t code_head(int* raw_tiles);

  // ---- session slots ------------------------------------------------------------------------
  void use_slot(int s);
  // A step's view of the state: with one slot, its pointers (the kernels' one-session code, no table); with
  // more, slot 0's and the step's slot table (in_d->slot), so that one captured step serves rows in any slots.
  attn::Window step_win(int L) const;
  attn::Compressed step_ckv(int k) const;
  attn::IndexKeys step_keys(int k) const;
  attn::CompressorCarry step_carry(int k) const;

  // ---- one step -----------------------------------------------------------------------------
  void sync() { CUDA_CHECK(cudaStreamSynchronize(st)); }
  void dump(Sink* sink, int L, const char* name, int pos0, int R, const void* dev, size_t row_bytes);
  void inject(Sink* sink, int L, const char* name, int pos0, int R, void* dev, size_t row_bytes);
  // Engram's all-gather moves a fixed segment per row count (one rank may own every cell), so a
  // captured step does not depend on where the step's rows live.
  static size_t engram_seg(int R) { return (size_t)R * kEngramCols * kEngramRow; }
  // The positions the dense index scans are checked against: the whole cache. index_dense takes the scan's
  // length from the step's positions, so its launch, and the graph of a row count, is the same for every
  // context (the graph maps keep this key).
  int scan_ctx(int) const { return kstride; }
  // side-stream work: fork(s) makes s wait for the main stream so far; join(s) the main stream for s
  void fork(cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(ev_fork, st));
    CUDA_CHECK(cudaStreamWaitEvent(s, ev_fork, 0));
  }
  void join(cudaStream_t s) {
    cudaEvent_t e = ev_join[s == sb];
    CUDA_CHECK(cudaEventRecord(e, s));
    CUDA_CHECK(cudaStreamWaitEvent(st, e, 0));
  }
  void engram_prepare(int pos0, int R, const uint64_t seq[2]);
  void engram_submit(int R, const uint64_t seq[2]);  // the store's requests for the rows' ids
  // a batch step's rows: each one's token into its slot's history, then its session's hashes
  void engram_rows(int R, const int32_t* tok, const int32_t* pos, const int32_t* slot, const uint64_t seq[2]);
  void engram_apply(int L, int R);
  void project(const bf16* stream, int L, int h, int R);
  void attention(int L, int R, int ctx, bool dumps, bool sub_in, Sink* sink, int pos0);
  void indexer(int L, int R, int ctx);
  void ffn(int L, int R, bool dumps, Sink* sink, int pos0);
  void l2pf(int w, int L, int R);  // Options::l2pf: window w's prefetch of layer L's weights, on sc
  // The step's device work, from the StepIn ring onwards: eager (flags: dumps, injection, logits) or captured.
  void forward(int R, int ctx, uint32_t flags, Sink* sink, int pos0);
  // its rows through the model: the embedding to this rank's logits (and kFlagLogits' gather)
  void forward_rows(int R, int ctx, uint32_t flags, Sink* sink, int pos0);
  cudaGraphExec_t graph(int R, int ctx);  // captured on first use
  uint64_t launch(int R, int pos0, const int32_t* tokens, const Sampling& sp, uint32_t flags, Sink* sink);
  uint64_t launch_batch(const Batch& b);
  void engram_late_batch(const int32_t* tokens);
  // a launch's device work, once its StepIn slot n is filled: the graph of R rows, or eagerly
  uint64_t run_step(uint64_t n, int R, int ctx, uint32_t flags, Sink* sink, int pos0);
  // fill the next StepIn slot (the step's number n) and record its Engram seqs
  ops::StepIn& next_in(uint64_t n, int R, int pos0, const int32_t* tokens, const Sampling& sp);
  void engram_late(int pos0, int R, const int32_t* tokens);
  const ops::SampleOut& wait(uint64_t n);
  void wait_done(uint64_t n) {  // step n has finished (its StepIn slot and Engram buffers are free)
    if (n < n_done) return;
    wait(n);
  }
  void drain() {  // every launched step has finished
    if (n_steps) wait_done(n_steps - 1);
    sync();
  }

  // ---- chunks of many rows (prefill.cu) -----------------------------------------------------
  void prefill_alloc(comm::Ring* ring = nullptr);
  void prefill_engram(int a, int M, int set);  // the Engram lookups of positions [a, a + M) (hist's tokens)
  void engram_apply_p(int L, int M, int set);
  void boundary_p(int M, const bf16* sub, const bf16* res, const float* post, const float* comb, const float* pre,
                  bf16* out, int nL, int nh, float* pre_out, float* post_out, float* comb_out);
  void index_p(int L, int a, int M);
  // win_lo >= 0: CED's decoder replay (the window starts at win_lo with nothing before it; layer 20's
  // compressor ran in the encoder chunks)
  void attention_p(int L, int a, int M, const std::vector<std::pair<int, uint8_t*>>& snaps, int win_lo = -1);
  void ffn_p(int L, int M);
  void wave_probe(int L, int a, int M);
  // positions [p0, p1) between layer L's ring and the linear window buffer that starts at position base
  void ring_copy(int L, int p0, int p1, int base, bool to_lin);
  uint64_t prefill(int a, int M, const Sampling* sp, int handle, const std::vector<int>& snaps, bool ced);
  // CED (docs/ced.md): layer 20's compressor for an encoder chunk's rows; keeping the chunk's last
  // positions' layer-20 input; the decoder replay of [P - 128, P)
  void ced_project(int M);
  void ced_keep(int a, int M);
  void ced_replay(int P, const std::vector<std::pair<int, uint8_t*>>& snaps);

  // ---- sessions' saved state (state.cu) -----------------------------------------------------
  detail::Store& store_of();
  void state_reserve(size_t bytes);
  uint8_t* store_alloc(int h, size_t bytes);  // a block of the pool for session h (evicting), or nullptr
  void store_release(uint8_t* p, size_t bytes);
  uint64_t state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes, std::string* why);
  void state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id);
  void state_keep(uint64_t gen);
  size_t disk_take(int h, size_t bytes);  // room in the disk tier's file for session h (dropping), or kNone
  void state_spill(int h);
  void state_unspill(int h);
  void state_save(int h, int from, int to);
  void state_load(int h, int to);
  uint8_t* state_snap_buf(int h, int pos);  // a new snapshot's block (the caller fills it), nullptr: no room
  void state_snap(int h, int pos);
  void state_snap_end(int h, int pos);
  void state_restore(int h, int pos);
  void state_restore_end(int h);
  void state_drop(int h, int from);
  void state_free(int h);
  void state_copy_snaps(int src, int dst, int upto);
  void state_drop_snap(int h, int pos);
  void state_snap_spec(int h, int pos);  // the draft module's windows at positions [pos - 128, pos)

  // ---- DSpark (spec.cu) -----------------------------------------------------------------------
  void bind_spec(dsw::WeightFile& wf);
  void spec_head_coded();  // code_head: the draft module's bf16 head GEMV on the coded head
  void spec_use_slot(int s);  // the draft module's eager paths act on slot s's windows
  void spec_hidden(int L, int R);  // main_hidden of the step's rows at the input of layer L
  // prefill: main_hidden of the rows that enter the draft's window (the chunk's last 256, the 128 before
  // each snapshot), then their accept()s and the snapshots' copies of the windows. Under CED only the
  // replay (of the chunk that samples) runs layers 37-39: its 128 rows, and a snapshot at its end.
  void spec_prefill_hidden(int L, int a, int M, const std::vector<int>& snaps);
  void spec_prefill_accept(int a, int M, int handle, const std::vector<int>& snaps, bool ced, bool sampled);
  cudaGraphExec_t spec_graph(int ctx);
  void spec_graphs();
  void spec_fill(ops::StepIn& in, int mode, const SpecCfg& c);
  uint64_t launch_spec(int R, int pos0, const int32_t* tokens, const Sampling& sp, const SpecCfg& c);
  uint64_t launch_verify(const Sampling& sp, const SpecCfg& c, int pos_bound);
};

}  // namespace model
