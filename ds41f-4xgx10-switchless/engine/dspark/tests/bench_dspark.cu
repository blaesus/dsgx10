// One rank's DSpark pass with STAND-IN collectives (dspark::standin_comm()), as CUDA graphs: draft(), and accept()
// of 1 or 6 positions + draft(), timed over graph replays (accept alone would replay from L2: 22 MB of weights). Inputs from the golden run (the layer 37-39 input
// streams through main_hidden(), the tokens); weights from a rank file; the windows hold positions 0..anchor.
//
//   bench_dspark [--rank0 FILE] [--golden DIR] [--iters N] [--fp8 1 [--vrows N,...]] [--head4 1 [--cfg4 MT,WN]]
//                [--markov8 1]
// --fp8 1: the FP8 head copy (serve's default); --vrows: the pass with its head GEMV and Markov chain over the
// first N rows of the rank's vocabulary (DSPARK_VROWS, the draft pass study: a subset's time, not its drafts);
// --head4 1: the head as MXFP4 (made from the FP8 copy, as bind_spec does with --draft-head4), --cfg4: its
// GEMV's mt and wn for the head-alone timing (wk stays the pass's 8); --markov8 1: the Markov head as FP8
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "common/check.h"
#include "dspark/dspark.h"
#include "weights/loader.h"

using namespace dspark;

namespace {

// the data of a little-endian float32 / int32 .npy (mapped)
const char* npy_data(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) die("cannot open %s", path.c_str());
  struct stat st;
  fstat(fd, &st);
  void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (m == MAP_FAILED) die("cannot map %s", path.c_str());
  const char* p = static_cast<const char*>(m);
  uint16_t hl;
  std::memcpy(&hl, p + 8, 2);
  if (p[6] != 1) die("%s: npy version %d", path.c_str(), p[6]);
  return p + 10 + hl;
}

uint16_t bf16_of(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

struct Graph {
  cudaGraphExec_t exec = nullptr;
  size_t nodes = 0;
};
template <class F>
Graph capture(cudaStream_t s, F&& f) {
  cudaGraph_t g;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  f();
  CUDA_CHECK(cudaStreamEndCapture(s, &g));
  Graph r;
  CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &r.nodes));
  CUDA_CHECK(cudaGraphInstantiate(&r.exec, g, 0));
  CUDA_CHECK(cudaGraphDestroy(g));
  return r;
}
// ms per replay: the median of 5 runs of `iters` back-to-back replays
double time_graph(const Graph& g, cudaStream_t s, int iters) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaGraphLaunch(g.exec, s));
  std::vector<double> t;
  for (int rep = 0; rep < 5; ++rep) {
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(g.exec, s));
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    t.push_back(ms / iters);
  }
  std::sort(t.begin(), t.end());
  return t[2];
}

}  // namespace

int main(int argc, char** argv) {
  std::string rank0 = "/data/dsgb10/weights/rank0.dsw", golden = "/data/dsgb10/golden/prose";
  int iters = 50;
  bool fp8 = false, head4 = false, markov8 = false;
  int mt4 = 2, wn4 = 1;
  std::vector<int> vrows;  // EXPERIMENT (the draft pass study): DSPARK_VROWS values to time, with --fp8 1
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--rank0") rank0 = v;
    else if (k == "--golden") golden = v;
    else if (k == "--iters") iters = std::stoi(v);
    else if (k == "--fp8") fp8 = v == "1";
    else if (k == "--head4") head4 = v == "1";
    else if (k == "--markov8") markov8 = v == "1";
    else if (k == "--cfg4") {
      if (std::sscanf(v.c_str(), "%d,%d", &mt4, &wn4) != 2) die("--cfg4 MT,WN");
    }
    else if (k == "--vrows")
      for (size_t a = 0; a < v.size();) {
        size_t b = v.find(',', a);
        if (b == std::string::npos) b = v.size();
        vrows.push_back(std::stoi(v.substr(a, b - a)));
        a = b + 1;
      }
    else die("unknown argument %s", k.c_str());
  }
  dsw::WeightFile f(rank0);
  dsw::LoadOptions o;
  o.select = [](const dsw::DeviceTensor& t) {
    return t.name.rfind("mtp.", 0) == 0 || t.name == "head.weight" || t.name == "embed.weight";
  };
  f.load(o);
  Weights w = weights_from({&f});
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  if (fp8 || head4) {  // the FP8 head copy (serve's default), as the runner makes it at bind
    uint8_t *q, *qs;
    CUDA_CHECK(cudaMalloc(&q, (size_t)kVocabRank * kDim));
    CUDA_CHECK(cudaMalloc(&qs, (size_t)kVocabRank * kDim / 1024));
    head_fp8(w.head, q, qs, s);
    w.head8 = q, w.head8_s = qs;
  }
  if (head4) {  // the MXFP4 head from it (--draft-head4)
    uint8_t *q, *qs;
    CUDA_CHECK(cudaMalloc(&q, (size_t)kVocabRank * kDim / 2));
    CUDA_CHECK(cudaMalloc(&qs, (size_t)kVocabRank * kDim / 32));
    const long moved = head_fp4(w.head8, w.head8_s, q, qs, s);
    std::printf("the head as MXFP4: %ld groups under a scale above sim_fp4's\n", moved);
    CUDA_CHECK(cudaFree(const_cast<uint8_t*>(w.head8)));
    CUDA_CHECK(cudaFree(const_cast<uint8_t*>(w.head8_s)));
    w.head8 = w.head8_s = nullptr;
    w.head4 = q, w.head4_s = qs;
    fp8 = false;
  }
  if (markov8) {  // --draft-markov8
    uint8_t *q, *qs;
    CUDA_CHECK(cudaMalloc(&q, (size_t)kVocabRank * kMarkov));
    CUDA_CHECK(cudaMalloc(&qs, (size_t)kVocabRank * kMarkov / 1024));
    markov_fp8(w.markov_head, q, qs, s);
    w.markov8 = q, w.markov8_s = qs;
  }
  float2* table;
  CUDA_CHECK(cudaMalloc(&table, 256 * 32 * sizeof(float2)));
  attn::rope_table_build(table, 256, false, s);
  if (vrows.empty()) vrows.push_back(0);
  for (const int vr : vrows) {
  if (vr) setenv("DSPARK_VROWS", std::to_string(vr).c_str(), 1);
  Draft d(w, 0, standin_comm(), table);

  // inputs: the golden streams of positions 0..100 (anchor 100), the token at 101
  const int anchor = 100;
  const float* L[3];
  for (int l = 0; l < 3; ++l) L[l] = reinterpret_cast<const float*>(npy_data(golden + "/L" + std::to_string(36 + l) + ".out.npy"));
  const int32_t* tokens = reinterpret_cast<const int32_t*>(npy_data(golden + "/tokens.npy"));
  uint16_t *x, *mh;
  int *count, *pos0, *token;
  CUDA_CHECK(cudaMalloc(&x, (size_t)kMaxAcc * kHc * kDim * 2));
  CUDA_CHECK(cudaMalloc(&mh, (size_t)kMaxAcc * kMainK * 2));
  CUDA_CHECK(cudaMalloc(&count, 4));
  CUDA_CHECK(cudaMalloc(&pos0, 4));
  CUDA_CHECK(cudaMalloc(&token, 4));
  auto hidden = [&](int p, int n) {
    std::vector<uint16_t> h((size_t)n * kHc * kDim);
    for (int l = 0; l < 3; ++l) {
      for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_of(L[l][(size_t)p * kHc * kDim + i]);
      CUDA_CHECK(cudaMemcpy(x, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
      main_hidden(x, n, l, mh, false, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
    }
    CUDA_CHECK(cudaMemcpy(count, &n, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pos0, &p, 4, cudaMemcpyHostToDevice));
  };
  for (int p = 0; p <= anchor; p += kMaxAcc) {
    hidden(p, std::min(kMaxAcc, anchor + 1 - p));
    d.accept(mh, count, pos0, s);
  }
  const int tok = tokens[anchor + 1];
  CUDA_CHECK(cudaMemcpy(token, &tok, 4, cudaMemcpyHostToDevice));

  // distinct experts per block in this draft (what the MoE reads)
  int* ridx[kBlocks];
  for (int b = 0; b < kBlocks; ++b) {
    CUDA_CHECK(cudaMalloc(&ridx[b], kDraft * 6 * 4));
    d.taps.route_idx[b] = ridx[b];
  }
  d.draft(token, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::printf("distinct experts per block (5 rows x top-3; the stand-in gathers repeat rank 0's router logits):");
  for (int b = 0; b < kBlocks; ++b) {
    std::vector<int> h(kDraft * 6);
    CUDA_CHECK(cudaMemcpy(h.data(), ridx[b], h.size() * 4, cudaMemcpyDeviceToHost));
    std::printf(" %zu", std::set<int>(h.begin(), h.end()).size());
    d.taps.route_idx[b] = nullptr;
  }
  std::printf("\n");

  hidden(anchor - 5, 6);  // accept 6 positions ending at the anchor (count 6; the graphs read it each replay)
  Graph ga = capture(s, [&] { d.accept(mh, count, pos0, s); });
  Graph gd = capture(s, [&] { d.draft(token, s); });
  Graph gb = capture(s, [&] {
    d.accept(mh, count, pos0, s);
    d.draft(token, s);
  });
  const uint8_t* head = w.head;
  float* lg;
  uint16_t* hx;
  CUDA_CHECK(cudaMalloc(&lg, (size_t)kDraft * kVocabRank * 4));
  CUDA_CHECK(cudaMalloc(&hx, (size_t)kDraft * kDim * 2));
  CUDA_CHECK(cudaMemset(hx, 0, (size_t)kDraft * kDim * 2));
  dsk::GemvCfg ch{2, 1, 16}, ch8{2, 1, 8}, ch4{mt4, wn4, 8};  // draft()'s (ch4: draft() takes ch8)
  const int hrows = vr ? vr : kVocabRank;
  Graph gh = capture(s, [&] {
    if (head4)  // the MXFP4 head over the rows the pass covers
      dsk::gemv_fp4w_f32(reinterpret_cast<const __nv_bfloat16*>(hx), kDraft, w.head4, w.head4_s, hrows, kDim, lg, ch4, s);
    else if (fp8)  // the FP8 head copy over the rows the pass covers
      dsk::gemv_fp8w_f32(reinterpret_cast<const __nv_bfloat16*>(hx), kDraft, w.head8, w.head8_s, hrows, kDim, lg, ch8, s);
    else
      dsk::gemv_bf16(reinterpret_cast<const __nv_bfloat16*>(hx), kDraft, head, kVocabRank, kDim, lg, nullptr, ch, false,
                     s);
  });

  const double t_b6 = time_graph(gb, s, iters);
  hidden(anchor, 1);
  const double t_b1 = time_graph(gb, s, iters);
  const double t_d = time_graph(gd, s, iters);
  const double t_h = time_graph(gh, s, iters);
  std::printf("one rank, STAND-IN collectives (%d replays x 5, median)%s%s, vocabulary rows %d:\n", iters,
              head4 ? ", the MXFP4 head" : fp8 ? ", the FP8 head" : "", markov8 ? ", the FP8 Markov head" : "",
              vr ? vr : kVocabRank);
  std::printf("  draft (3 blocks, head, chain) %7.3f ms  (%zu nodes)\n", t_d, gd.nodes);
  std::printf("  accept 1 + draft              %7.3f ms  (%zu nodes; accept %zu)\n", t_b1, gb.nodes, ga.nodes);
  std::printf("  accept 6 + draft              %7.3f ms\n", t_b6);
  if (head4)
    std::printf("  of which the head GEMV alone  %7.3f ms  (e2m1 [%d][5120] and its scales, %.0f MB; mt %d wn %d)\n", t_h,
                hrows, hrows * 5120.0 * 17 / 32 / 1e6, mt4, wn4);
  else if (fp8)
    std::printf("  of which the head GEMV alone  %7.3f ms  (e4m3 [%d][5120], %.0f MB)\n", t_h, hrows, hrows * 5120.0 / 1e6);
  else
    std::printf("  of which the head GEMV alone  %7.3f ms  (bf16 [32320][5120], 331 MB)\n", t_h);
  CUDA_CHECK(cudaFree(lg));
  CUDA_CHECK(cudaFree(hx));
  }
  return 0;
}
