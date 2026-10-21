// The many-row path (prefill.cu): buffers for chunks of up to `cap` rows, made on the first prefill.
#pragma once
#include "model/impl.h"

namespace model::detail {

constexpr bool ced_tail_architecture_supported() {
  if (kLayers != 2 * kCedDecoderLayers) return false;
  for (int L = kCedDecoderLayers; L < kLayers; ++L)
    if (kv_source(L) != kCedDecoderLayers || ratio_of(L) != 1 || engram_layer(L)) return false;
  return true;
}
static_assert(ced_tail_architecture_supported(), "tail replay requires decoder KV projected from encoder output");

// A window snapshot: the ring slots of positions [pos - 128, pos) of every layer (state.cu keeps them).
constexpr int kSnapRows = 128;
constexpr size_t kSnapKv = (size_t)kSnapRows * kHead;       // bytes per layer
constexpr size_t kSnapSc = (size_t)kSnapRows * kHead / 32;
constexpr size_t kSnapBytes = kLayers * (kSnapKv + kSnapSc);
inline uint8_t* snap_kv(uint8_t* b, int L) { return b + L * kSnapKv; }
inline uint8_t* snap_sc(uint8_t* b, int L) { return b + kLayers * kSnapKv + L * kSnapSc; }

// Sequence parallel: of a chunk of M rows, rank r owns rows [r q, r q + n) with q = q_rows(M) (n = q
// but for the last ranks): their residual stream, mixes, sublayer boundaries and the row-split work.
// Arrays of every row have 4q rows, rank r's at row r q, so that all-gathers fill them in place; the
// rows past M are padding.
inline int q_rows(int M) { return ((M + 3) / 4 + 7) & ~7; }

// CED (docs/ced.md): the decoder replays the request's 128 or 2560 positions. Its input is layer 20's
// input as the encoder left it, kept per position (slot = position mod replay_rows) in records: the
// stream x, the normed input y with its act_quant (yq, ys) and layer 20's attention mixes.
constexpr int kReplay = kCedReplayRows;  // the legacy policy; a tail-policy request uses kCedTailRows
// The dense index scans' rows per batch (dsk::index_topk_p: a batch's keys are selected while in L2).
constexpr int kIdxBatch = 128;
constexpr size_t kCedX = 0, kCedY = kCedX + (size_t)kHcDim * 2, kCedYq = kCedY + (size_t)kDim * 2,
                 kCedYs = kCedYq + kDim, kCedPre = kCedYs + kDim / 32, kCedPost = kCedPre + kHc * 4,
                 kCedComb = kCedPost + kHc * 4, kCedRow = kCedComb + kHc * kHc * 4;

// The MoE's down and combine run in column groups, and each group's reduce-scatter goes out on the
// comm stream while the next group computes.
constexpr int kColGroups = 4, kGroupCols = kDim / kColGroups;
constexpr int kRsBlocks = 8;  // blocks of a column group's reduce-scatter (the rest of the GPU keeps computing)
constexpr int kCombRsBlocks = 32;  // blocks of 256 threads of the MoE's combine fused into its reduce-scatter

// Weights the sequence-parallel parts need whole: every rank's slices, gathered on the first prefill.
struct PLayer {
  uint8_t *wqa = nullptr, *sqa = nullptr, *wkv = nullptr, *skv = nullptr;  // wq_a [1280], wkv [512] (L1), scales
  uint8_t *ckv = nullptr, *cgate = nullptr;  // compressor wkv / wgate [512] (kv sources), bf16 tiled (L2)
  uint8_t *ewkv = nullptr, *eskv = nullptr;  // Engram wkv: [rank][6400] (L1), scales [rank][200][192]
  uint8_t *wob = nullptr, *sob = nullptr;    // wo_b: [rank][5120][2048] (L1, each rank's K slice), scales
};

struct Prefill {
  int cap = 0, Q = 0;        // Q = q_rows(cap)
  int rq = 0, r0 = 0, n = 0;  // the chunk in progress: rows per rank, this rank's first row and count
  // this rank's rows (row 0 = the chunk's row r0)
  bf16 *x, *mid, *y, *yraw, *sub;
  // the mixes: the attention's own (post, comb) and the FFN's pre (from x), the FFN's own and the next
  // attention's pre (from mid)
  float *a_pre, *a_post, *a_comb, *pm, *f_post, *f_comb;
  float *ysum, *gl, *wpf;
  bf16 *iq, *e_kv;
  // every row
  bf16 *part, *qa, *kvx, *qr, *q, *o;
  float *ckvf, *cscf, *rwts, *ymoe;
  uint8_t *yq, *ys, *qq, *qs, *oq, *os, *gath;
  uint8_t *oqa, *osa;  // every rank's wo_a output (act_quant) for this rank's rows: [rank][Q][2048], scales
  uint8_t *lin_kv, *lin_sc;  // one layer's window: up to 127 positions before the chunk, then the chunk
  int* image_mask;
  int *tok, *pos, *ridx, *counters, *topk_mine, *topk[kIdxSlots], *pool;
  // keys: a batch of the dense scans' rows (kIdxBatch x kstride), or the Reindex scans' rows (Q x the pool)
  uint16_t *keys, *bkeys, *topkey, *poolkey;
  uint8_t* iqf;  // the dense scans' queries (attn::IndexQ): codes, scales, head weights of this rank's rows
  uint32_t* iqsc;
  float* iqw;
  void* moe_ws;
  cudaStream_t cs = nullptr;  // the reduce-scatters of the column groups (high priority)
  cudaEvent_t ev_grp[kColGroups], ev_cs;
  int32_t* tok_h;  // pinned staging of the chunk's tokens
  // CED: this rank's records of the positions kept (its own rows of them), every rank's (the replay's
  // all-gather); per slot the position (-1: none) and the rank whose row it was; the end of the last
  // encoder chunk (a chunk that does not continue it starts a new run of positions)
  uint8_t *ced_rows, *ced_gath;
  int ced_pos[kCedTailRows], ced_src[kCedTailRows];
  int ced_capacity = kReplay, ced_replay_rows = kReplay;
  int ced_end = -1;
  // Engram: two sets of lookups, so the next chunk's run while this one computes
  uint8_t *e_send[2][2], *e_gath, *e_xq, *e_xs, *e_part;
  uint32_t* e_off[2][2];
  uint64_t *e_done, *e_wait;  // pinned [set][layer]: the store's completion flags, the seqs to wait for
  size_t e_seg[2][2] = {};
  int e_pos0[2] = {-1, -1}, e_n[2] = {0, 0};  // the positions of each set's pending lookups (-1: none)
  int e_last = 1;  // the set of the chunk enqueued last: the next chunk's lookups go to the other
  std::vector<uint32_t> ids[2];
  PLayer pl[kLayers];
};

}  // namespace model::detail
