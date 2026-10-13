// DSpark in the decode loop (spec.cu, DESIGN.md decision 7): the draft module, its collectives over the
// ring, and the spec step graphs, whose rows the device chooses.
#pragma once
#include <map>
#include <memory>

#include "dspark/dspark.h"
#include "model/impl.h"

namespace model::detail {

constexpr int kSpecLayer0 = 37;   // main_hidden: the inputs of layers 37, 38, 39
constexpr int kSpecPfRows = 256;  // a prefill chunk's last rows that enter the draft's windows (every slot)
// the draft module's windows as sessions keep them (state.cu): 3 rings of 256 slots of KV and scales
constexpr size_t kSpecSlot = (size_t)kHead + kHead / 32;
constexpr size_t kSpecSnapBytes = (size_t)dspark::kBlocks * 128 * kSpecSlot;     // positions [pos - 128, pos)
constexpr size_t kSpecEndBytes = (size_t)dspark::kBlocks * attn::kRing * kSpecSlot;  // the whole rings

class RingComm;

struct Spec {
  std::unique_ptr<RingComm> comm;
  std::unique_ptr<dspark::Draft> draft;
  std::vector<void*> whole;      // the drafter's mHC projections and router rows (gathered at bind), FP8 head
  float2* rope = nullptr;        // RoPE without YaRN, max_ctx + 8 positions (drafts run 5 past the anchor)
  uint16_t* mh = nullptr;        // main_hidden [kMaxRows][15360] of a spec step's rows
  ops::SpecState* ss = nullptr;  // device
  // a spec step per context bucket: spec_in, a switch over the row count (body R - 1: R rows), then the
  // sampling with the verdicts, accept, draft and spec_next
  std::map<int, cudaGraphExec_t> graphs;
  bool quiet = false;  // spec_graphs: one line for them all
  size_t nodes = 0;    // in the last graph's bodies
  // prefill: main_hidden of the chunk's rows that enter the windows. Each rank computes the rows it owns
  // into its block of pf_mh ([4][C][15360] for C rows a rank at most), one all-gather shares them, then
  // accept() takes them kMaxAcc at a time.
  int pf_rows = 0, pf_groups = 0;  // capacities: 4 C rows, accept() calls
  uint16_t* pf_mh = nullptr;
  int* pf_acc = nullptr;    // device [pf_groups][2]: count, pos0
  int* pf_acc_h = nullptr;  // pinned staging
  ~Spec();
};

}  // namespace model::detail
