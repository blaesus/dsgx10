// DSpark in the decode loop (spec.h, model.h, DESIGN.md decision 7, engine/dspark/README.md).
//
// A spec step is one graph: spec_in (the step's rows: the host's, or the ones the step before chose), a
// switch over the row count whose body R - 1 runs R rows through the model (main_hidden at the inputs
// of layers 37-39 on the way), the sampling (a verify step also judges each draft by the sample of the
// row before it), the draft module's accept of the step's final rows and its draft from the token after
// them, and spec_next, which picks the next step's drafts by the confidence head's survival and publishes
// the step. The device decides the next step's rows, tokens and positions, so the host queues it before
// this one ends; every rank decides the same from the same bits.
#include "model/spec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>

#include "model/nvtx.h"
#include "model/prefill.h"

namespace model {

using namespace detail;

namespace detail {

namespace {

constexpr int kSegs = dspark::kMaxGatherSegs;
struct GatherArgs {
  dspark::GatherSeg s[kSegs];
  uint32_t off[kSegs];  // the segment's bytes in a rank's block
  uint32_t total;       // a rank's block, rounded up to 16 bytes
  int n;
};

// this rank's rows of each segment into its block (4-byte words)
__global__ void k_gather_pack(GatherArgs a, uint32_t* send) {
  for (int i = 0; i < a.n; ++i) {
    const dspark::GatherSeg& g = a.s[i];
    const int words = (int)(g.width / 4);
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < g.rows * words; t += gridDim.x * blockDim.x) {
      const int r = t / words, k = t % words;
      send[(a.off[i] + r * g.width) / 4 + k] =
          reinterpret_cast<const uint32_t*>(static_cast<const char*>(g.src) + r * g.src_ld)[k];
    }
  }
}
// every rank's block into place: dst + r * dst_ld + q * width <- rank q's row r
__global__ void k_gather_unpack(GatherArgs a, const uint32_t* recv) {
  for (int i = 0; i < a.n; ++i) {
    const dspark::GatherSeg& g = a.s[i];
    const int words = (int)(g.width / 4), per = g.rows * words;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < kRanks * per; t += gridDim.x * blockDim.x) {
      const int q = t / per, r = t % per / words, k = t % words;
      reinterpret_cast<uint32_t*>(static_cast<char*>(g.dst) + r * g.dst_ld + q * g.width)[k] =
          recv[(q * a.total + a.off[i] + r * g.width) / 4 + k];
    }
  }
}

__device__ __forceinline__ void st_release_sys_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// The step's input (as ops::step_in), a verify step's rows, tokens and positions from the spec state;
// then the switch's value, the row count - 1.
__global__ void k_spec_in(const ops::StepIn* ring, uint32_t* counter, const ops::SpecState* ss, ops::StepIn* d,
                          cudaGraphConditionalHandle h) {
  constexpr int n = sizeof(ops::StepIn) / 4;
  static_assert(sizeof(ops::StepIn) % 4 == 0, "StepIn");
  const uint32_t slot = *counter % ops::kInSlots;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    reinterpret_cast<uint32_t*>(d)[i] = reinterpret_cast<const volatile uint32_t*>(ring + slot)[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    if (d->spec == ops::kSpecVerify) {
      const int R = ss->next_rows;
      d->rows = R;
      for (int r = 0; r < kMaxRows; ++r) {
        d->tokens[r] = r < R ? ss->next_tok[r] : 0;
        d->pos[r] = ss->next_pos0 + min(r, R - 1);
      }
    }
    *counter += 1;
    cudaGraphSetConditional(h, (unsigned)(d->rows - 1));
  }
}

// After the draft pass: the drafts the next step verifies, by the gate (ops.h; at most max_drafts, its
// rows below pos_end); then the step is published.
__global__ void k_spec_next(const ops::StepIn* in, ops::SpecState* ss, const int* drafts, const float* surv,
                            ops::SampleOut* outs) {
  if (threadIdx.x != 0) return;
  ops::SampleOut& o = outs[in->step % ops::kOutSlots];
  const int P = ss->pos0 + ss->count;
  const int maxk = min(min(ops::kDrafts, in->max_drafts), in->pos_end - 1 - P);
  int k = 0;
  if (in->gate == ops::kGateRatio) {
    // a step emits 1 + its accepted drafts, and d_1 .. d_n are all accepted with probability survival[n - 1]
    float e = 1.f, best = 1.f / in->step_ms[0];
    for (int n = 1; n <= maxk; ++n) {
      e += surv[n - 1];
      const float r = e / in->step_ms[n];
      if (r > best) best = r, k = n;
    }
  } else if (in->gate == ops::kGateMarginal) {
    while (k < maxk && surv[k] >= in->thresh * (in->step_ms[k + 1] - in->step_ms[k])) ++k;
  } else {
    while (k < maxk && surv[k] >= in->thresh) ++k;
  }
  ss->next_rows = 1 + k;
  ss->next_pos0 = P;
  ss->next_tok[0] = ss->bonus;
  for (int j = 1; j < kMaxRows; ++j) ss->next_tok[j] = j <= k ? drafts[j - 1] : 0;
  o.next_rows = 1 + k;
  o.next_pos0 = P;
  for (int j = 0; j < kMaxRows; ++j) o.next_tok[j] = ss->next_tok[j];
  for (int j = 0; j < ops::kDrafts; ++j) o.drafts[j] = drafts[j], o.survival[j] = surv[j];
  __threadfence_system();
  st_release_sys_u32(&o.seq, in->step + 1);
}

// EXPERIMENT (Options::tree_probe): the drafter's probe into the step's SampleOut, before k_spec_next
// publishes it.
__global__ void k_probe_out(const ops::StepIn* in, const dspark::ProbeOut* p, ops::SampleOut* outs) {
  static_assert(sizeof(dspark::ProbeOut) == sizeof(ops::SampleOut::probe), "ProbeOut");
  ops::SampleOut& o = outs[in->step % ops::kOutSlots];
  for (int i = threadIdx.x; i < ops::kProbeWords; i += blockDim.x) o.probe[i] = reinterpret_cast<const uint32_t*>(p)[i];
  __threadfence_system();
}

// The positions of a pass over [a, a + M) that enter the draft module's windows, in ranges in this order:
// - the 128 before each snapshot inside the pass (from its first row);
// - its last 256: every slot of the windows, what the main model's ring holds after the chunk. A
//   snapshot at a + M is taken after them.
// Snapshots outside (a, a + M] are not the pass's (CED: the encoder's, which hold no decoder state).
// They go to accept() in groups of up to 8 rows. A group stays within one rank's rows of the pass
// (prefill.h: rank r owns rows [r rq, r rq + rq)), so it is a run of rows in that rank's block.
struct PfGroup {
  int pos, rows, owner, row;  // row: the group's first in its owner's block
  int snap;                   // >= 0: the last group before this snapshot
};
std::vector<PfGroup> pf_groups(int a, int M, int rq, std::vector<int> snaps, int& per_rank) {
  std::sort(snaps.begin(), snaps.end());
  std::vector<std::array<int, 3>> ranges;  // [lo, hi), the snapshot after it (-1: none)
  for (int s : snaps)
    if (s > a && s < a + M) ranges.push_back({std::max(a, s - kSnapRows), s, s});
  const bool at_end = std::count(snaps.begin(), snaps.end(), a + M) > 0;
  ranges.push_back({std::max(a, a + M - kSpecPfRows), a + M, at_end ? a + M : -1});
  std::vector<PfGroup> g;
  int used[kRanks] = {};
  for (const auto& rg : ranges) {
    for (int p = rg[0]; p < rg[1];) {
      const int r = (p - a) / rq;
      const int n = std::min({dspark::kMaxAcc, rg[1] - p, a + (r + 1) * rq - p});
      g.push_back({p, n, r, used[r], -1});
      used[r] += n, p += n;
    }
    if (rg[1] > rg[0]) g.back().snap = rg[2];
  }
  per_rank = *std::max_element(used, used + kRanks);
  return g;
}

// ---- EXPERIMENT, measurement only (the draft pass study): host references of dspark::sim_fp4 and sim_fp8_l2 on
// one 32x32 block, so a run says whether the rounded weights are what the format would hold
float e4m3_host(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  const float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
  return (b >> 7) ? -v : v;
}
float round_pow2_host(float amax, float inv) {  // attn::fast_round_scale
  const float p = amax * inv;
  uint32_t b;
  std::memcpy(&b, &p, 4);
  return std::ldexp(1.0f, (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0));
}
float e2m1_host(float y) {  // attn::f_to_e2m1's thresholds, |y| <= 6
  const float a = std::fabs(y);
  float q = 0.f;
  if (a > 0.25f) q = 0.5f;
  if (a >= 0.75f) q = 1.f;
  if (a > 1.25f) q = 1.5f;
  if (a >= 1.75f) q = 2.f;
  if (a > 2.5f) q = 3.f;
  if (a >= 3.5f) q = 4.f;
  if (a > 5.0f) q = 6.f;
  return y < 0 ? -q : q;
}
float e4m3_round_host(float y) {  // round to nearest even e4m3, saturating
  const float a = std::fabs(y);
  if (a == 0.f) return 0.f;
  int e;
  std::frexp(a, &e);  // a in [2^(e-1), 2^e)
  const float step = a < 0x1p-6f ? 0x1p-9f : std::ldexp(1.0f, e - 4);
  const float r = std::min(448.f, std::nearbyint(a / step) * step);
  return y < 0 ? -r : r;
}
// block (rows 0-31, columns 0-31) of an e4m3 matrix [N][K] (L1 or natural), as host bytes
std::vector<uint8_t> e4m3_block0(const uint8_t* w, int K, bool l1) {
  std::vector<uint8_t> h(32 * 32);
  for (int n = 0; n < 32; ++n)
    CUDA_CHECK(cudaMemcpy(h.data() + n * 32, w + (l1 ? (size_t)(n / 8) * (K / 64) * 512 + (n % 8) * 64 : (size_t)n * K), 32,
                          cudaMemcpyDeviceToHost));
  return h;
}
// sim_fp4 on block 0: of 1024 values, how many hold MXFP4's (the e2m1 value under the group's scale) halved, as e4m3
// stores it (exact but in rows ~2^9 below the block's amax, whose halves fall below e4m3's range)
int check_fp4(const std::vector<uint8_t>& before, const std::vector<uint8_t>& after) {
  int ok = 0;
  for (int n = 0; n < 32; ++n) {
    float amax = 0.f;
    for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(e4m3_host(before[n * 32 + k])));
    const float g = round_pow2_host(std::max(amax, 6.0f * 0x1p-126f), 1.0f / 6.0f);
    for (int k = 0; k < 32; ++k) {
      const float v = e4m3_host(before[n * 32 + k]);
      const float q = e2m1_host(std::min(6.f, std::max(-6.f, v / g))) * g;
      ok += e4m3_host(after[n * 32 + k]) == e4m3_round_host(q * 0.5f);
    }
  }
  return ok;
}
// bf16 L2 block 0 of [N][K]
std::vector<float> bf16_l2_block0(const uint8_t* w, int K) {
  std::vector<uint16_t> h(32 * 32);
  for (int n = 0; n < 32; ++n)
    CUDA_CHECK(cudaMemcpy(h.data() + n * 32, w + (size_t)(n / 8) * (2 * K / 64) * 512 + (n % 8) * 64, 64,
                          cudaMemcpyDeviceToHost));
  std::vector<float> f(32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    const uint32_t u = (uint32_t)h[i] << 16;
    std::memcpy(&f[i], &u, 4);
  }
  return f;
}
int check_fp8(const std::vector<float>& before, const std::vector<float>& after) {
  float amax = 0.f;
  for (float v : before) amax = std::max(amax, std::fabs(v));
  const float sc = round_pow2_host(std::max(amax, 1e-4f), 1.0f / 448.0f);
  int ok = 0;
  for (int i = 0; i < 32 * 32; ++i) ok += after[i] == e4m3_round_host(before[i] / sc) * sc;
  return ok;
}

}  // namespace

// The draft module's collectives over the ring (dspark::Comm): a gather is one all-gather of a packed
// block per rank, the all-reduces engine/comm's (fp32 sums in rank order), argmax argmax_rows.
class RingComm final : public dspark::Comm {
 public:
  explicit RingComm(const comm::DevChannel& ch) : ch_(ch) {
    send_ = dalloc<uint32_t>(kBlock / 4);
    recv_ = dalloc<uint32_t>(kRanks * kBlock / 4);
    f32_ = dalloc<float>((size_t)dspark::kDraft * dspark::kDim);
  }
  ~RingComm() override {
    cudaFree(send_);
    cudaFree(recv_);
    cudaFree(f32_);
  }
  void all_gather(const dspark::GatherSeg* segs, int n, cudaStream_t s) override {
    if (n < 1 || n > kSegs) die("spec: all_gather of %d segments", n);
    GatherArgs a{};
    uint32_t off = 0;
    int words = 0;
    for (int i = 0; i < n; ++i) {
      const dspark::GatherSeg& g = segs[i];
      if (g.width % 4 || g.src_ld % 4 || g.dst_ld % 4) die("spec: all_gather of width %zu", g.width);
      a.s[i] = g;
      a.off[i] = off;
      off += (uint32_t)(g.width * g.rows);
      words = std::max(words, (int)(g.rows * g.width / 4));
    }
    a.total = (off + 15) / 16 * 16;
    a.n = n;
    if (a.total > kBlock) die("spec: all_gather of %u bytes", a.total);
    const int blocks = std::max(1, std::min(16, (words + 255) / 256));
    k_gather_pack<<<blocks, 256, 0, s>>>(a, send_);
    comm::allgather(ch_, send_, recv_, a.total, s);
    k_gather_unpack<<<std::min(64, 4 * blocks), 256, 0, s>>>(a, recv_);
  }
  void all_reduce(const void* src, bool f32, uint16_t* dst, size_t n, cudaStream_t s) override {
    if (n > (size_t)dspark::kDraft * dspark::kDim) die("spec: all_reduce of %zu", n);
    if (!f32) {
      comm::allreduce_bf16(ch_, static_cast<const bf16*>(src), reinterpret_cast<bf16*>(dst), n, s);
    } else {
      comm::allreduce_f32(ch_, static_cast<const float*>(src), f32_, n, s);
      ops::f32_to_bf16(f32_, reinterpret_cast<bf16*>(dst), (int)n, s);
    }
  }
  // argmax_rows takes rows in pairs: the second is the candidate after src, whose result lands in dst[1]
  // (the draft's next argmax overwrites it, or it is past the five)
  void argmax(const dspark::Cand* src, dspark::Cand* dst, cudaStream_t s) override {
    static_assert(sizeof(dspark::Cand) == sizeof(comm::ArgMax), "Cand");
    comm::argmax_rows(ch_, reinterpret_cast<const comm::ArgMax*>(src), reinterpret_cast<comm::ArgMax*>(dst), 2, s);
  }

 private:
  static constexpr uint32_t kBlock = 32768;
  comm::DevChannel ch_;
  uint32_t *send_, *recv_;
  float* f32_;
};

Spec::~Spec() {
  for (auto& g : graphs) cudaGraphExecDestroy(g.second);
  for (void* p : whole) cudaFree(p);
  cudaFree(rope);
  cudaFree(mh);
  cudaFree(ss);
  cudaFree(pf_mh);
  cudaFree(pf_acc);
  cudaFreeHost(pf_acc_h);
}

}  // namespace detail

// ---- setup ------------------------------------------------------------------------------------------

void Model::Impl::bind_spec(dsw::WeightFile& wf) {
  if (!bound) die("model: bind_spec before bind");
  if (head_coded) die("model: bind_spec after code_head (the FP8 head is made from the raw head)");
  if (opt.first_layer != 0 || opt.n_layers != kLayers) die("model: DSpark needs the whole model");
  spec = std::make_unique<Spec>();
  Spec& sp = *spec;
  sp.comm = std::make_unique<RingComm>(ch);
  sp.rope = dalloc<float2>((size_t)(opt.max_ctx + 8) * 32);
  attn::rope_table_build(sp.rope, opt.max_ctx + 8, false, st);
  // the drafter's mHC projections and router rows whole on every rank (dspark.h BlockW): 9 collectives fewer
  // in each draft, the same bits
  dspark::Weights dw = dspark::weights_from({&wf});
  float* hc_tmp = dalloc<float>((size_t)kMix * kHcDim);
  for (dspark::BlockW& b : dw.b) {
    for (int h = 0; h < 2; ++h) {
      float* all = dalloc<float>((size_t)kMix * kHcDim);
      comm::allgather(ch, h ? b.ffn_fn : b.attn_fn, hc_tmp, (size_t)kMix * kDim * 4, st);  // rank r: [24][5120r, +5120)
      for (int r = 0; r < kRanks; ++r)
        CUDA_CHECK(cudaMemcpy2DAsync(all + (size_t)kDim * r, (size_t)kHcDim * 4, hc_tmp + (size_t)kMix * kDim * r,
                                     (size_t)kDim * 4, (size_t)kDim * 4, kMix, cudaMemcpyDeviceToDevice, st));
      (h ? b.ffn_fn_all : b.attn_fn_all) = all;
      sp.whole.push_back(all);
    }
    uint8_t* g = dalloc<uint8_t>((size_t)dspark::kExperts * kDim * 2);  // the L2 tiles of 8 rows concatenate
    comm::allgather(ch, b.gate, g, (size_t)dspark::kExperts / kRanks * kDim * 2, st);
    b.gate_all = g;
    sp.whole.push_back(g);
  }
  if (opt.draft_fp8 || opt.draft_head4) {
    uint8_t* q = dalloc<uint8_t>((size_t)dspark::kVocabRank * kDim);
    uint8_t* s = dalloc<uint8_t>((size_t)dspark::kVocabRank * kDim / 1024);
    dspark::head_fp8(dw.head, q, s, st);
    dw.head8 = q, dw.head8_s = s;
    if (opt.draft_head4) {  // the MXFP4 head from it, then the FP8 copy goes
      uint8_t* q4 = dalloc<uint8_t>((size_t)dspark::kVocabRank * kDim / 2);
      uint8_t* s4 = dalloc<uint8_t>((size_t)dspark::kVocabRank * kDim / 32);
      const long moved = dspark::head_fp4(q, s, q4, s4, st);
      if (rank == 0)
        std::printf("draft head as MXFP4: %ld of %d groups under a scale above sim_fp4's (its e4m3 rounding)\n",
                    moved, dspark::kVocabRank * kDim / 32);
      dw.head4 = q4, dw.head4_s = s4;
      dw.head8 = dw.head8_s = nullptr;
      sync();
      CUDA_CHECK(cudaFree(q));
      CUDA_CHECK(cudaFree(s));
      sp.whole.push_back(q4), sp.whole.push_back(s4);
    } else {
      sp.whole.push_back(q), sp.whole.push_back(s);
    }
  }
  if (opt.draft_markov8) {
    uint8_t* q = dalloc<uint8_t>((size_t)dspark::kVocabRank * dspark::kMarkov);
    uint8_t* s = dalloc<uint8_t>((size_t)dspark::kVocabRank * dspark::kMarkov / 1024);
    dspark::markov_fp8(dw.markov_head, q, s, st);
    dw.markov8 = q, dw.markov8_s = s;
    sp.whole.push_back(q), sp.whole.push_back(s);
  }
  // EXPERIMENT, measurement only (the draft pass study): DSPARK_SIM, a comma list of fp4blocks (the blocks' e4m3
  // matrices and main_proj), fp4head (the FP8 head copy) and fp8markov (the Markov head), rounds those weights of
  // the drafter in place to the cheaper format's values (dspark::sim_fp4, sim_fp8_l2). Drafts only: the tokens
  // cannot change.
  if (const char* sim = getenv("DSPARK_SIM")) {
    const std::string s = std::string(",") + sim + ",";
    auto has = [&](const char* k) { return s.find(std::string(",") + k + ",") != std::string::npos; };
    int n = 0;
    if (has("fp4blocks")) {
      std::vector<std::string> names = {"mtp.0.main_proj"};
      for (int i = 0; i < dspark::kBlocks; ++i) {
        const std::string p = "mtp." + std::to_string(i) + ".";
        for (const char* m : {"attn.wq_a", "attn.wkv", "attn.wq_b", "attn.wo_a", "attn.wo_b", "ffn.shared_experts.w1",
                              "ffn.shared_experts.w3", "ffn.shared_experts.w2"})
          names.push_back(p + m);
      }
      for (const std::string& nm : names) {
        const dsw::DeviceTensor *t = wf.find(nm + ".weight"), *ts = wf.find(nm + ".scale");
        if (!t || !ts || !t->data || !ts->data || t->dtype != dsw::DType::F8_E4M3 || t->ndim != 2 ||
            ts->layout != dsw::Layout::Natural ||
            (t->layout != dsw::Layout::L1 && t->layout != dsw::Layout::Natural) ||
            ts->bytes != (uint64_t)(t->shape[0] / 32) * (t->shape[1] / 32))
          die("DSPARK_SIM: %s is not an e4m3 matrix with ue8m0 32x32 scales", nm.c_str());
        const bool l1 = t->layout == dsw::Layout::L1;
        uint8_t* wp = static_cast<uint8_t*>(t->data);
        sync();
        const std::vector<uint8_t> before = e4m3_block0(wp, (int)t->shape[1], l1);
        dspark::sim_fp4(wp, static_cast<uint8_t*>(ts->data), (int)t->shape[0], (int)t->shape[1], l1, st);
        sync();
        const int ok = check_fp4(before, e4m3_block0(wp, (int)t->shape[1], l1));
        if (rank == 0 && n == 0) std::printf("DSPARK_SIM: %s block 0: %d of 1024 values MXFP4's\n", nm.c_str(), ok);
        if (ok < 1024) die("DSPARK_SIM: %s block 0: only %d of 1024 values MXFP4's", nm.c_str(), ok);
        ++n;
      }
    }
    if (has("fp4head")) {
      if (!dw.head8) die("DSPARK_SIM=fp4head needs --draft-fp8 (and not --draft-head4)");
      uint8_t* hp = const_cast<uint8_t*>(dw.head8);
      sync();
      const std::vector<uint8_t> before = e4m3_block0(hp, kDim, true);
      dspark::sim_fp4(hp, const_cast<uint8_t*>(dw.head8_s), dspark::kVocabRank, kDim, true, st);
      sync();
      const int ok = check_fp4(before, e4m3_block0(hp, kDim, true));
      if (rank == 0) std::printf("DSPARK_SIM: the FP8 head copy, block 0: %d of 1024 values MXFP4's\n", ok);
      if (ok < 1024) die("DSPARK_SIM: the head copy's block 0: only %d of 1024 values MXFP4's", ok);
      ++n;
    }
    if (has("fp8markov")) {
      if (dw.markov8) die("DSPARK_SIM=fp8markov with --draft-markov8: the pass reads the FP8 copy");
      uint8_t* mp = const_cast<uint8_t*>(dw.markov_head);
      sync();
      const std::vector<float> before = bf16_l2_block0(mp, dspark::kMarkov);
      dspark::sim_fp8_l2(mp, dspark::kVocabRank, dspark::kMarkov, st);
      sync();
      const int ok = check_fp8(before, bf16_l2_block0(mp, dspark::kMarkov));
      if (rank == 0) std::printf("DSPARK_SIM: the Markov head, block 0: %d of 1024 values e4m3's\n", ok);
      if (ok < 1024) die("DSPARK_SIM: the Markov head's block 0: only %d of 1024 values e4m3's", ok);
      ++n;
    }
    if (rank == 0) std::printf("DSPARK_SIM=%s: %d tensors rounded in place\n", sim, n);
  }
  sync();
  cudaFree(hc_tmp);
  sp.draft = std::make_unique<dspark::Draft>(dw, rank, sp.comm.get(), sp.rope, nslots);
  sp.draft->use_slot(cur_slot);
  if (opt.tree_probe) sp.draft->enable_probe();
  sp.mh = dalloc<uint16_t>((size_t)kMaxRows * dspark::kMainK);
  sp.ss = dalloc<ops::SpecState>(1);
  // the prefill's seed rows, sized for a 4096-row chunk with snapshots every 1024 (768 rows in 96
  // groups, 384 on the last rank), so that a prefill rarely has to grow them (which synchronises)
  sp.pf_rows = kSpecPfRows + 2 * kSnapRows, sp.pf_groups = 128;
  sp.pf_mh = dalloc<uint16_t>((size_t)kRanks * sp.pf_rows * dspark::kMainK);
  sp.pf_acc = dalloc<int>(2 * (size_t)sp.pf_groups);
  sp.pf_acc_h = halloc<int>(2 * (size_t)sp.pf_groups);
  sync();
}

void Model::Impl::spec_head_coded() {
  if (!spec->graphs.empty()) die("model: code_head after a spec step was captured");
  if (!opt.draft_fp8 && !opt.draft_head4) spec->draft->set_head_coded(head);  // dspark's Weights::head is this slot too
}

void Model::Impl::spec_hidden(int L, int R) {
  dspark::main_hidden(U(x), R, L - kSpecLayer0, spec->mh, true, st);
}

// ---- prefill: the draft module's windows -------------------------------------------------------------

void Model::Impl::spec_prefill_hidden(int L, int a, int M, const std::vector<int>& snaps) {
  if (L < kSpecLayer0) return;
  NVTX_RANGE("spec_hidden L%02d", L);
  Spec& sp = *spec;
  const Prefill& p = *pf;
  int C;
  const auto gs = pf_groups(a, M, p.rq, snaps, C);
  if (C > sp.pf_rows || (int)gs.size() > sp.pf_groups) {  // more rows than room for
    sync();
    cudaFree(sp.pf_mh);
    cudaFree(sp.pf_acc);
    cudaFreeHost(sp.pf_acc_h);
    sp.pf_rows = std::max(sp.pf_rows, C), sp.pf_groups = std::max(sp.pf_groups, (int)gs.size());
    sp.pf_mh = dalloc<uint16_t>((size_t)kRanks * sp.pf_rows * dspark::kMainK);
    sp.pf_acc = dalloc<int>(2 * (size_t)sp.pf_groups);
    sp.pf_acc_h = halloc<int>(2 * (size_t)sp.pf_groups);
  }
  // this rank's rows (p.x row 0 is the chunk's row p.r0) into its block
  for (const PfGroup& g : gs)
    if (g.owner == rank)
      dspark::main_hidden(U(p.x) + (size_t)(g.pos - a - p.r0) * kHcDim, g.rows, L - kSpecLayer0,
                          sp.pf_mh + ((size_t)rank * C + g.row) * dspark::kMainK, false, st);
}

void Model::Impl::spec_prefill_accept(int a, int M, int handle, const std::vector<int>& snaps, bool ced,
                                      bool sampled) {
  NVTX_RANGE("spec_accept");
  if (ced) {  // the replay's rows (ced_replay: spec_prefill_hidden(L, P - 128, 128, {})), then a snapshot at P
    if (!sampled) return;
    const int P = a + M;
    std::vector<int> at_end;
    if (std::count(snaps.begin(), snaps.end(), P)) at_end.push_back(P);
    return spec_prefill_accept(P - kReplay, kReplay, handle, at_end, false, true);
  }
  Spec& sp = *spec;
  int C;
  const auto gs = pf_groups(a, M, pf->rq, snaps, C);
  const comm::AgSeg seg{sp.pf_mh + (size_t)rank * C * dspark::kMainK, sp.pf_mh, (size_t)C * dspark::kMainK * 2};
  comm::allgather_multi(ch, &seg, 1, st);
  for (size_t i = 0; i < gs.size(); ++i) sp.pf_acc_h[2 * i] = gs[i].rows, sp.pf_acc_h[2 * i + 1] = gs[i].pos;
  // pinned staging: the next chunk's prefill() drains before it is written again
  CUDA_CHECK(cudaMemcpyAsync(sp.pf_acc, sp.pf_acc_h, gs.size() * 8, cudaMemcpyHostToDevice, st));
  for (size_t i = 0; i < gs.size(); ++i) {
    const PfGroup& g = gs[i];
    sp.draft->accept(sp.pf_mh + ((size_t)g.owner * C + g.row) * dspark::kMainK, sp.pf_acc + 2 * i,
                     sp.pf_acc + 2 * i + 1, st);
    if (g.snap >= 0) state_snap_spec(handle, g.snap);  // a snapshot's windows
  }
}

// ---- the spec step -------------------------------------------------------------------------------------

cudaGraphExec_t Model::Impl::spec_graph(int ctx) {
  Spec& sp = *spec;
  const auto it = sp.graphs.find(ctx);
  if (it != sp.graphs.end()) return it->second;
  const clock_t t0 = clock();
  size_t free0 = 0, free1 = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  const auto mode = cudaStreamCaptureModeThreadLocal;
  cudaGraph_t g, cap;
  CUDA_CHECK(cudaGraphCreate(&g, 0));
  cudaGraphConditionalHandle h;
  CUDA_CHECK(cudaGraphConditionalHandleCreate(&h, g, 0, 0));
  // the step's rows, and the switch's value
  CUDA_CHECK(cudaStreamBeginCaptureToGraph(st, g, nullptr, nullptr, 0, mode));
  k_spec_in<<<1, 128, 0, st>>>(in_h, in_ctr, sp.ss, in_d, h);
  CUDA_CHECK(cudaGetLastError());
  cudaStreamCaptureStatus cs;
  const cudaGraphNode_t* deps = nullptr;
  size_t nd = 0;
  CUDA_CHECK(cudaStreamGetCaptureInfo(st, &cs, nullptr, nullptr, &deps, nullptr, &nd));
  const std::vector<cudaGraphNode_t> head(deps, deps + nd);
  CUDA_CHECK(cudaStreamEndCapture(st, &cap));
  // the switch: body R - 1 runs R rows
  cudaGraphNodeParams cp{};
  cp.type = cudaGraphNodeTypeConditional;
  cp.conditional.handle = h;
  cp.conditional.type = cudaGraphCondTypeSwitch;
  cp.conditional.size = kMaxRows;
  cudaGraphNode_t sw;
  CUDA_CHECK(cudaGraphAddNode(&sw, g, head.data(), nullptr, head.size(), &cp));
  mh_on = true;
  for (int R = 1; R <= kMaxRows; ++R) {
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(st, cp.conditional.phGraph_out[R - 1], nullptr, nullptr, 0, mode));
    forward_rows(R, ctx, 0, nullptr, 0);
    CUDA_CHECK(cudaStreamEndCapture(st, &cap));
  }
  mh_on = false;
  size_t nodes = 0;
  for (int R = 1; R <= kMaxRows; ++R) {
    size_t k = 0;
    CUDA_CHECK(cudaGraphGetNodes(cp.conditional.phGraph_out[R - 1], nullptr, &k));
    nodes += k;
  }
  sp.nodes = nodes;
  // the verdicts, the draft module, the next step's rows, the publication
  CUDA_CHECK(cudaStreamBeginCaptureToGraph(st, g, &sw, nullptr, 1, mode));
  ops::sample(ch, logits, kVocabPart * rank, in_d, ctl_h, out_h, fb, samp_ctr, st, sp.ss);
  // the step's session's windows (every entry of its slot table holds its slot)
  const int* slot = nslots > 1 ? in_d->slot : nullptr;
  sp.draft->accept(sp.mh, &sp.ss->count, &sp.ss->pos0, st, slot);
  sp.draft->draft(&sp.ss->bonus, st, slot);
  if (opt.tree_probe) {  // EXPERIMENT (docs/engine.md, "Draft trees")
    sp.draft->probe(in_d->probe, st);
    k_probe_out<<<1, 128, 0, st>>>(in_d, sp.draft->probe_out(), out_h);
  }
  k_spec_next<<<1, 32, 0, st>>>(in_d, sp.ss, sp.draft->tokens(), sp.draft->survival(), out_h);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamEndCapture(st, &cap));
  cudaGraphExec_t e;
  CUDA_CHECK(cudaGraphInstantiate(&e, g, cudaGraphInstantiateFlagUseNodePriority));
  CUDA_CHECK(cudaGraphDestroy(g));
  sp.graphs.emplace(ctx, e);
  CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  if (rank == 0 && !sp.quiet)
    std::printf("spec: the step graph of context %d (%zu nodes in its bodies) in %.2f s (CPU), %.1f MiB of device "
                "memory\n", ctx, nodes, (double)(clock() - t0) / CLOCKS_PER_SEC, ((double)free0 - (double)free1) / 1048576.0);
  return e;
}

// The step graphs of every context now (serve's startup): the device memory a graph takes (its conditional
// bodies live on the device) would otherwise have to be free in the middle of a session. One graph serves
// every context (scan_ctx), so the sweep captures one.
void Model::Impl::spec_graphs() {
  if (!spec) die("model: spec_graphs without bind_spec");
  const clock_t t0 = clock();
  size_t free0 = 0, free1 = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  spec->quiet = true;
  for (int p = 0; p < opt.max_ctx; p += kCtxBucket) spec_graph(scan_ctx(p));
  spec_graph(scan_ctx(opt.max_ctx - 1));
  spec->quiet = false;
  CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  if (rank == 0)
    std::printf("spec: %zu step graph%s for contexts up to %d (%zu nodes in its bodies) in %.1f s (CPU), %.2f GiB of "
                "device memory\n", spec->graphs.size(), spec->graphs.size() == 1 ? "" : "s", opt.max_ctx, spec->nodes,
                (double)(clock() - t0) / CLOCKS_PER_SEC, ((double)free0 - (double)free1) / 1073741824.0);
}

void Model::Impl::spec_use_slot(int s) {
  if (spec && spec->draft) spec->draft->use_slot(s);
}

void Model::Impl::spec_fill(ops::StepIn& in, int mode, const SpecCfg& c) {
  in.spec = mode;
  in.thresh = c.thresh;
  in.max_drafts = std::max(0, std::min(ops::kDrafts, c.max_drafts));
  in.pos_end = std::min(c.pos_end, opt.max_ctx);
  in.force = c.force;
  std::copy(c.probe, c.probe + ops::kDrafts, in.probe);
  in.gate = c.gate;
  std::copy(c.step_ms, c.step_ms + ops::kDrafts + 1, in.step_ms);
}

uint64_t Model::Impl::launch_spec(int R, int pos0, const int32_t* tokens, const Sampling& sp, const SpecCfg& c) {
  if (!spec) die("model: launch_spec without bind_spec");
  const int pos_last = pos0 + R - 1;
  if (R < 1 || R > kMaxRows || pos0 < 0 || pos_last >= opt.max_ctx) die("model: bad spec step R=%d pos0=%d", R, pos0);
  for (int r = 0; r < R; ++r)
    if (tokens[r] < 0) die("model: launch_spec takes the step's tokens");
  const uint64_t n = n_steps;
  if (n >= (uint64_t)ops::kInSlots) wait_done(n - ops::kInSlots);
  ops::StepIn& in = next_in(n, R, pos0, tokens, sp);
  spec_fill(in, ops::kSpecGiven, c);
  if (store) {
    if (late.on) die("model: a step waits for its tokens");
    const uint64_t seq[2] = {e_seq + 1, e_seq + 2};
    e_seq += 2;
    in.e_wait[0] = seq[0], in.e_wait[1] = seq[1];
    if (n) wait_done(n - 1);  // the previous step has read the Engram buffers
    hist.resize(pos0);
    for (int r = 0; r < R; ++r) hist.push_back(tokens[r]);
    engram_prepare(pos0, R, seq);
  }
  const cudaGraphExec_t e = spec_graph(scan_ctx(pos_last));
  const int slot = (int)(n % ops::kOutSlots);
  CUDA_CHECK(cudaEventRecord(ev_beg[slot], st));
  CUDA_CHECK(cudaGraphLaunch(e, st));
  CUDA_CHECK(cudaEventRecord(ev_end[slot], st));
  n_steps = n + 1;
  return n;
}

uint64_t Model::Impl::launch_verify(const Sampling& sp, const SpecCfg& c, int pos_bound) {
  if (!spec) die("model: launch_verify without bind_spec");
  if (pos_bound < 0) die("model: launch_verify(%d)", pos_bound);
  const uint64_t n = n_steps;
  if (n >= (uint64_t)ops::kInSlots) wait_done(n - ops::kInSlots);
  const int32_t none = 0;
  ops::StepIn& in = next_in(n, 1, 0, &none, sp);
  spec_fill(in, ops::kSpecVerify, c);
  if (store) {
    if (late.on) die("model: two steps wait for their tokens");
    const uint64_t seq[2] = {e_seq + 1, e_seq + 2};
    e_seq += 2;
    in.e_wait[0] = seq[0], in.e_wait[1] = seq[1];
    late.on = true, late.any = true, late.batch = false;
    late.pos0 = -1, late.R = 0, late.seq[0] = seq[0], late.seq[1] = seq[1];
  }
  const cudaGraphExec_t e = spec_graph(scan_ctx(std::min(pos_bound, opt.max_ctx - 1)));
  const int slot = (int)(n % ops::kOutSlots);
  CUDA_CHECK(cudaEventRecord(ev_beg[slot], st));
  CUDA_CHECK(cudaGraphLaunch(e, st));
  CUDA_CHECK(cudaEventRecord(ev_end[slot], st));
  n_steps = n + 1;
  return n;
}

// ---- Model ------------------------------------------------------------------------------------------

void Model::bind_spec(dsw::WeightFile& wf) { d_->bind_spec(wf); }
void Model::spec_graphs() { d_->spec_graphs(); }
float Model::gpu_ms(uint64_t n) const {
  if (n >= d_->n_steps || n + ops::kOutSlots < d_->n_steps) die("model: gpu_ms of step %llu", (unsigned long long)n);
  const int slot = (int)(n % ops::kOutSlots);
  CUDA_CHECK(cudaEventSynchronize(d_->ev_end[slot]));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, d_->ev_beg[slot], d_->ev_end[slot]));
  return ms;
}
bool Model::spec() const { return d_->spec != nullptr; }
uint64_t Model::launch_spec(int R, int pos0, const int32_t* tokens, const Sampling& sp, const SpecCfg& c) {
  return d_->launch_spec(R, pos0, tokens, sp, c);
}
uint64_t Model::launch_verify(const Sampling& sp, const SpecCfg& c, int pos_bound) {
  return d_->launch_verify(sp, c, pos_bound);
}

}  // namespace model
