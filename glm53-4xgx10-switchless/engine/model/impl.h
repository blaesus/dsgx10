// Model::Impl, shared by the model's translation units: model.cu (binding, the decode step, launch and wait),
// prefill.cu (chunks), mtp.cu (the MTP layer's fill and draft passes), synth.cu (synthetic weights).
#pragma once
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attn/attn.h"
#include "comm/glmcoll.h"
#include "common/check.h"
#include "experts/experts.h"
#include "model/model.h"
#include "model/ops.h"

namespace glm::model {
namespace detail {

template <class T>
T* dalloc(size_t n) {  // zeroed before it returns (cudaMemset runs on the legacy stream)
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T) + 256));
  CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T) + 256));
  CUDA_CHECK(cudaDeviceSynchronize());
  return static_cast<T*>(p);
}
template <class T>
T* halloc(size_t n) {  // pinned, mapped host memory
  void* p = nullptr;
  CUDA_CHECK(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped));
  std::memset(p, 0, n * sizeof(T));
  return static_cast<T*>(p);
}

struct Fp8W {  // an FP8 weight in the attention GEMV's tiles and its f32 scales
  const uint8_t* w = nullptr;
  const float* s = nullptr;
};
struct LayerW {
  const uint16_t *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *kv_norm = nullptr;
  const uint16_t *k_norm_w = nullptr, *k_norm_b = nullptr;  // full layers
  Fp8W xproj, q_b, w_uk, w_uv, o_proj, iq_b;                 // iq_b: the indexer's wq_b (full layers)
  const uint16_t* wproj = nullptr;                            // the indexer's weights_proj rows [8][6144]
  Fp8W gate_up, down;                                         // dense layers 0-2
  const uint16_t* router = nullptr;                           // MoE: BF16 [256][6144]
  const float* router_bias = nullptr;
  gx::LayerW moe{};                                           // MoE: the routed pool, descriptors, shared expert
};
struct MtpW {
  const uint16_t *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
  Fp8W eh;  // [1536][12288] rows of this rank
};

// The next weights the step reads, for the paced L2 prefetch (l2pf.cu): a layer's tensors in read order.
enum { kPfXproj, kPfQb, kPfUk, kPfUv, kPfOproj, kPfFfn0, kPfFfn1, kPfTargets };
struct PfTarget {
  const void* p[4];
  size_t n[4];
};
struct PfState {  // device: how far a layer's windows got
  int layer;
  uint32_t chunk;
};
// Window w of layer `layer` (1: the FFN all-reduce before it, 2: its x-projection's all-gather, 3: its attention,
// 4: its o_proj all-reduce): prefetch from the window's first unread target on, budget_us at the pace; ends early
// once the window's collective has arrived (arrived, finished: the channel's counters; null for the attention).
void l2pf_window(const PfTarget (&t)[kPfTargets], int layer, int window, float budget_us,
                 const unsigned long long* arrived, const unsigned long long* finished, PfState* state, cudaStream_t s);

// Synthetic weights (bind.cu): allocations kept here so the model can free them.
struct Synth;
struct SynthFree {
  void operator()(Synth* s) const;  // bind.cu
};

}  // namespace detail

struct Model::Impl {
  using bf16 = __nv_bfloat16;
  int rank;
  Options opt;
  cudaStream_t st = nullptr;
  comm::DevChannel ch{};
  bool bound = false;
  size_t wbytes = 0;
  double exp_bytes = 0;  // one routed expert's bytes a rank (gate, up, down of one MoE layer)
  bool l2pf_on = true;
  std::unique_ptr<detail::Synth, detail::SynthFree> synth;

  // weights
  detail::LayerW lw[kCacheLayers];
  detail::MtpW mtp;
  const uint16_t *embed = nullptr, *head = nullptr, *norm_w = nullptr;
  const uint32_t* rope = nullptr;
  int rope_positions = 0;

  // caches: per layer [slots][max_ctx][576] bf16; per full layer the index keys
  uint16_t* kv[kCacheLayers] = {};
  ::glm::IdxCache ik[kIdxCaches] = {};
  ::glm::KvCache kv_of(int l) const { return ::glm::KvCache{kv[l], (size_t)opt.max_ctx}; }
  ::glm::IdxCache ik_of(int l) const { return full_layer(l) ? ik[l == kMtpLayer ? kIdxCaches - 1 : idx_slot(l)] : ::glm::IdxCache{}; }

  // decode step buffers (kMaxRows rows)
  uint16_t *h, *xn, *xg, *qr, *q, *iq, *qlat, *u, *o, *hm, *ebuf, *hfin;
  float *iw, *part, *yr, *gl, *rw, *logits;  // yr: [2][kMaxRows][6144], ys = yr + R rows (contiguous)
  int *ridx, *topk;
  uint32_t* idx_work;
  float* idx_scores;
  size_t idx_ld = 0;
  float *att_o, *att_ml;
  void* moe_ws = nullptr;
  // the MTP's draft passes (decode)
  uint16_t *m_x12, *m_xg, *m_h, *m_xn;  // [R][12288], eh_proj gathered [R][4][1536], residual, normed
  int32_t *m_tok, *m_pos;               // the pass's tokens and positions
  int* m_topk;                          // the MTP layer's top-2048 (shared by the chained passes)
  comm::glm::Cand *m_cand, *m_best;     // local top-1 and the exchanged draft, per row
  int32_t* m_draft;                     // [kMaxDrafts][kMaxRows]

  // step inputs and results
  ops::StepIn *in_h, *in_d;
  uint32_t* in_ctr;
  int32_t* fb;  // the last step's sampled tokens
  SampleOut* out_h;
  uint32_t* samp_ctr;
  uint64_t n_steps = 0, n_done = 0;
  double host_us = 0;  // the last launch's host time, without waiting for a free input slot
  cudaEvent_t ev_beg[ops::kOutSlots], ev_end[ops::kOutSlots], ev_pf[2];
  std::map<int, cudaGraphExec_t> graphs;
  bool warm[kMaxRows + 1] = {};

  // side streams: l2pf (sc); events for fork/join
  cudaStream_t sc = nullptr;
  cudaEvent_t ev_fork, ev_join;
  detail::PfState* pfs = nullptr;
  bool forked = false;  // the step being built has side-stream work to join
  void fork(cudaStream_t s) {
    forked = true;
    CUDA_CHECK(cudaEventRecord(ev_fork, st));
    CUDA_CHECK(cudaStreamWaitEvent(s, ev_fork, 0));
  }
  void join(cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(ev_join, s));
    CUDA_CHECK(cudaStreamWaitEvent(st, ev_join, 0));
  }

  // per-kernel-class profiling (Model::profile): events between classes on the main stream
  bool prof = false;
  bool in_mtp = false;  // marks inside the draft passes count as kKMtp
  std::vector<cudaEvent_t> pev;
  std::vector<std::pair<int, int>> pmarks;  // (class, layer) of the work since the previous mark
  void mark(int cls, int layer);

  // the launch flags of the step being built: PDL unless profiling
  bool pdl() const { return opt.pdl && !prof; }

  // MTP pending rows: per slot, the last prefilled row's final hidden waits for the token after it
  uint16_t* m_pend = nullptr;  // [slots][6144]
  std::vector<int> pend_pos;   // per slot: that row's position, -1: none

  // prefill
  struct Pre;
  struct PreFree {
    void operator()(Pre* p) const;  // prefill.cu
  };
  std::unique_ptr<Pre, PreFree> pre;

  Impl(int r, const Options& opts);
  ~Impl();
  void bind_common(comm::Ring& ring);
  void bind_file(glw::WeightFile& wf, const SynthOptions* fill);
  void synth_pools(const SynthOptions& so, std::vector<const uint8_t*>& pools, std::vector<const gx::MatDesc*>& descs);
  bool synth_experts = false;
  int real_expert_layers = 0;  // MoE layers bound to the file's routed experts  // a file's routed experts replaced by synthetic pools
  void bind_synth(const SynthOptions& so);

  // ---- the decode step (model.cu) ----
  // one layer for R rows of the step: on entry xn = input_layernorm(h) of the rows; on exit h is the layer's
  // output and xn the next layer's normed input (or the final norm's / the MTP head norm's output)
  void layer(int l, int R, const int32_t* pos, const int32_t* slot, uint16_t* h_, uint16_t* xn_, int* topk_,
             const uint16_t* next_norm, bool index_on, Sink* sink, uint32_t flags);
  void forward(int R, uint32_t flags, Sink* sink);
  cudaGraphExec_t graph(int R);
  void l2pf(int window, int l, int R);
  void embed_rows(const int32_t* tok, const int32_t* pos, int R, bool zero_pos0, uint16_t* out);
  void dump_rows(Sink* sink, int L, const char* name, const int32_t* pos_dev, int R, const void* dev, size_t row_bytes);
  void gather_logits(Sink* sink, const float* lg, const int32_t* pos_dev, int R, bool mtp);
  uint64_t launch(const Batch& b, Sink* sink);
  const SampleOut& wait(uint64_t n);

  // ---- the MTP layer (mtp.cu) ----
  // the draft passes of a step's rows after its sample (decode): pass 1 at each row's position with the token
  // after it, the next passes chained
  void drafts(int R, uint32_t flags, Sink* sink);
  // the MTP layer's input for n rows: x = eh_proj([enorm(e) | hnorm(hid)]) all-gathered into out [n][6144] (x12
  // [n][12288] scratch); e, hid [n][6144] bf16. Decode (n <= 16) or prefill (16-row GEMVs).
  void mtp_project(int n, const uint16_t* e, const uint16_t* hid, uint16_t* x12, uint16_t* out);
  // prefill: the MTP cache entries of rows t in [t0, t1) of a chunk at a (embedding rows of tokens t + 1 in emb at
  // row t + 1 - a; hidden rows of the main model's final output at row t - a, or the slot's pending row)
  void mtp_fill(int slot, int a, int M);

  // ---- prefill (prefill.cu) ----
  void prefill_alloc();
  uint64_t prefill(int slot, int a, int M, const int32_t* tokens, const Sampling* sp, uint32_t flags, Sink* sink);
  void prefill_chunk(int slot, int a, int M, const int32_t* tokens, bool last, uint32_t flags, Sink* sink);
};

}  // namespace glm::model
