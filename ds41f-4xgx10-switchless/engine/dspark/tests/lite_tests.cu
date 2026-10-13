// The draft pass's cheaper formats (docs/engine.md, "The draft pass's approximations": bundle B) against the
// study's simulations of them, which they must reproduce bit for bit:
//   gemv:   dsk::gemv_fp4w_f32 on head_fp4()'s copy of a random FP8 matrix of the head's shape (some rows with
//           tiny values, so that groups' scales move) against dsk::gemv_fp8w_f32 (ch8) on sim_fp4's rounding of
//           it, at 1, 5 and 8 rows, for every mt and wn the kernel takes (wk 8, as the pass);
//   module: the pass on --draft-head4 and --draft-markov8's weights (head_fp4(), markov_fp8()) against the pass
//           on DSPARK_SIM=fp4head,fp8markov's (sim_fp4 on the FP8 copy, sim_fp8_l2 on the Markov head): one rank,
//           stand-in collectives, the golden run's positions; logits, Markov bias, biased logits, drafts and
//           confidences, every bit.
//
//   lite_tests [--rank0 FILE] [--golden DIR] [--steps N] [--module 0]
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "common/check.h"
#include "dspark/dspark.h"
#include "weights/loader.h"

using namespace dspark;

namespace {

int failures = 0;

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

template <class T>
T* dev(size_t n) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// a and b bit for bit; prints the first difference
template <class T>
void same(const char* what, const std::vector<T>& a, const std::vector<T>& b) {
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(T)) && bad++ == 0) first = i;
  if (bad) {
    ++failures;
    double va, vb;
    if constexpr (std::is_floating_point_v<T>) va = a[first], vb = b[first];
    else va = (double)a[first], vb = (double)b[first];
    std::printf("FAIL %s: %zu of %zu differ; first at %zu: %.9g against %.9g\n", what, bad, a.size(), first, va, vb);
  }
}

void test_gemv() {
  constexpr int N = kVocabRank, K = kDim;
  std::mt19937 rng(20261012);
  std::vector<uint8_t> w((size_t)N * K), sc((size_t)N / 32 * (K / 32));
  // rows in L1 tiles: most random e4m3 (no NaN), every 50th of subnormals only, every 50th from 25 of the
  // subnormals 0..3 (x 2^-9) with a 3 leading each group of 32: sim_fp4 puts such a group under 2^-10, where its
  // 3 becomes 6 x 2^-10; halved into e4m3 (1.5 x 2^-9) that rounds up to 2^-8, so the simulated value is 2^-7,
  // 8 x 2^-10, past e2m1's 6: head_fp4 moves the group's scale up (the head itself has no such group)
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      uint32_t b = rng() & 0xFF;
      if (n % 50 == 0) b = (b & 0x80) | (b & 0x07);
      else if (n % 50 == 25) b = (b & 0x80) | (k % 32 == 0 ? 3 : b % 4);
      else if ((b & 0x7F) == 0x7F) b ^= 1;
      w[((size_t)(n / 8) * (K / 64) + k / 64) * 512 + (n % 8) * 64 + k % 64] = (uint8_t)b;
    }
  for (uint8_t& s : sc) s = (uint8_t)(104 + rng() % 18);  // group scales up to 2, gemv_fp4w_f32's range
  std::vector<uint16_t> x(8 * (size_t)K);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (uint16_t& v : x) v = bf16_of(u(rng) * std::exp2(std::floor(u(rng) * 6)));

  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  uint8_t *w8 = dev<uint8_t>(w.size()), *s8 = dev<uint8_t>(sc.size()), *ws = dev<uint8_t>(w.size()),
          *ss = dev<uint8_t>(sc.size());
  uint8_t *w4 = dev<uint8_t>(w.size() / 2), *s4 = dev<uint8_t>(w.size() / 32);
  uint16_t* xd = dev<uint16_t>(x.size());
  float *ya = dev<float>(8 * (size_t)N), *yb = dev<float>(8 * (size_t)N);
  CUDA_CHECK(cudaMemcpy(w8, w.data(), w.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ws, w.data(), w.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s8, sc.data(), sc.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ss, sc.data(), sc.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(xd, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  sim_fp4(ws, ss, N, K, true, st);  // the study's simulation
  const long moved = head_fp4(w8, s8, w4, s4, st);
  std::printf("gemv: head_fp4 of a random FP8 head: %ld of %d groups under a scale above sim_fp4's\n", moved, N * K / 32);
  const long tiny = (long)((N - 1 - 25) / 50 + 1) * (K / 32);  // every group of the rows n % 50 == 25
  if (moved != tiny) {
    ++failures;
    std::printf("FAIL gemv: %ld groups under a moved scale, not the tiny rows' %ld\n", moved, tiny);
  }
  const auto* xb = reinterpret_cast<const __nv_bfloat16*>(xd);
  for (int M : {1, 5, 8}) {
    CUDA_CHECK(cudaMemset(ya, 0xFF, 8 * (size_t)N * 4));
    dsk::gemv_fp8w_f32(xb, M, ws, ss, N, K, ya, dsk::GemvCfg{2, 1, 8}, st);
    const std::vector<float> ha = host(ya, (size_t)M * N);
    size_t nz = 0;
    for (float v : ha) nz += v != 0.f;
    const std::pair<int, int> cfgs[] = {{1, 1}, {2, 1}, {4, 1}, {1, 2}, {2, 2}};
    for (const auto& c : cfgs) {
      CUDA_CHECK(cudaMemset(yb, 0xFF, 8 * (size_t)N * 4));
      dsk::gemv_fp4w_f32(xb, M, w4, s4, N, K, yb, dsk::GemvCfg{c.first, c.second, 8}, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      const std::string what = "gemv M=" + std::to_string(M) + " mt=" + std::to_string(c.first) + " wn=" +
                               std::to_string(c.second);
      same(what.c_str(), ha, host(yb, (size_t)M * N));
    }
    std::printf("gemv M=%d: %zu of %zu outputs nonzero, 5 configurations compared\n", M, nz, ha.size());
  }
  for (void* p : {(void*)w8, (void*)s8, (void*)ws, (void*)ss, (void*)w4, (void*)s4, (void*)xd, (void*)ya, (void*)yb})
    CUDA_CHECK(cudaFree(p));
  CUDA_CHECK(cudaStreamDestroy(st));
}

// One pass's outputs per step: the head's logits, the Markov bias, the biased logits, the chain's tokens and the
// confidences.
struct Out {
  std::vector<float> logits, bias, dl, conf;
  std::vector<int> ids;
};

std::vector<Out> run_module(const Weights& w, const std::string& golden, int steps, cudaStream_t s) {
  float2* table;
  CUDA_CHECK(cudaMalloc(&table, 256 * 32 * sizeof(float2)));
  attn::rope_table_build(table, 256, false, s);
  std::vector<Out> outs;
  {
    Draft d(w, 0, standin_comm(), table);
    const float* L[3];
    for (int l = 0; l < 3; ++l) L[l] = reinterpret_cast<const float*>(npy_data(golden + "/L" + std::to_string(36 + l) + ".out.npy"));
    const int32_t* tokens = reinterpret_cast<const int32_t*>(npy_data(golden + "/tokens.npy"));
    uint16_t *x = dev<uint16_t>((size_t)kMaxAcc * kHc * kDim), *mh = dev<uint16_t>((size_t)kMaxAcc * kMainK);
    int *count = dev<int>(1), *pos0 = dev<int>(1), *token = dev<int>(1);
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
    float *lg = dev<float>((size_t)kDraft * kVocabRank), *bias = dev<float>((size_t)kDraft * kVocabRank);
    d.taps.logits = lg, d.taps.bias = bias;
    const int anchor0 = 100;
    for (int p = 0; p <= anchor0; p += kMaxAcc) {
      hidden(p, std::min(kMaxAcc, anchor0 + 1 - p));
      d.accept(mh, count, pos0, s);
    }
    for (int i = 0; i < steps; ++i) {
      const int anchor = anchor0 + i;
      if (i) {
        hidden(anchor, 1);
        d.accept(mh, count, pos0, s);
      }
      const int tok = tokens[anchor + 1];
      CUDA_CHECK(cudaMemcpy(token, &tok, 4, cudaMemcpyHostToDevice));
      d.draft(token, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      const size_t n = (size_t)kDraft * kVocabRank;
      outs.push_back({host(lg, n), host(bias, n), host(d.logits(), n), host(d.confidence(), 16), host(d.ids(), 6)});
    }
    for (void* p : {(void*)x, (void*)mh, (void*)count, (void*)pos0, (void*)token, (void*)lg, (void*)bias})
      CUDA_CHECK(cudaFree(p));
  }
  CUDA_CHECK(cudaFree(table));
  return outs;
}

void test_module(const std::string& rank0, const std::string& golden, int steps) {
  dsw::WeightFile f(rank0);
  dsw::LoadOptions o;
  o.select = [](const dsw::DeviceTensor& t) {
    return t.name.rfind("mtp.", 0) == 0 || t.name == "head.weight" || t.name == "embed.weight";
  };
  f.load(o);
  Weights w = weights_from({&f});
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  uint8_t *q8 = dev<uint8_t>((size_t)kVocabRank * kDim), *s8 = dev<uint8_t>((size_t)kVocabRank * kDim / 1024);
  head_fp8(w.head, q8, s8, s);
  // --draft-head4 --draft-markov8's weights, made from the weights as loaded
  uint8_t *q4 = dev<uint8_t>((size_t)kVocabRank * kDim / 2), *s4 = dev<uint8_t>((size_t)kVocabRank * kDim / 32);
  uint8_t *m8 = dev<uint8_t>((size_t)kVocabRank * kMarkov), *m8s = dev<uint8_t>((size_t)kVocabRank * kMarkov / 1024);
  const long moved = head_fp4(q8, s8, q4, s4, s);
  markov_fp8(w.markov_head, m8, m8s, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::printf("module: the real head as MXFP4: %ld of %d groups under a scale above sim_fp4's\n", moved,
              kVocabRank * kDim / 32);
  Weights wn = w;
  wn.head4 = q4, wn.head4_s = s4, wn.markov8 = m8, wn.markov8_s = m8s;
  const std::vector<Out> a = run_module(wn, golden, steps, s);
  // DSPARK_SIM=fp4head,fp8markov's: the FP8 copy and the Markov head rounded in place
  sim_fp4(q8, s8, kVocabRank, kDim, true, s);
  sim_fp8_l2(const_cast<uint8_t*>(w.markov_head), kVocabRank, kMarkov, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  Weights ws = w;
  ws.head8 = q8, ws.head8_s = s8;
  const std::vector<Out> b = run_module(ws, golden, steps, s);
  int differ = 0;
  for (int i = 0; i < steps; ++i) {
    const int f0 = failures;
    same(("module step " + std::to_string(i) + " logits").c_str(), a[i].logits, b[i].logits);
    same(("module step " + std::to_string(i) + " bias").c_str(), a[i].bias, b[i].bias);
    same(("module step " + std::to_string(i) + " biased logits").c_str(), a[i].dl, b[i].dl);
    same(("module step " + std::to_string(i) + " drafts").c_str(), a[i].ids, b[i].ids);
    same(("module step " + std::to_string(i) + " confidences").c_str(), a[i].conf, b[i].conf);
    differ += failures > f0;
  }
  std::printf("module: %d steps compared, %d with a difference; drafts of the last: %d %d %d %d %d\n", steps, differ,
              a.back().ids[1], a.back().ids[2], a.back().ids[3], a.back().ids[4], a.back().ids[5]);
  for (void* p : {(void*)q8, (void*)s8, (void*)q4, (void*)s4, (void*)m8, (void*)m8s}) CUDA_CHECK(cudaFree(p));
  CUDA_CHECK(cudaStreamDestroy(s));
}

}  // namespace

int main(int argc, char** argv) {
  std::string rank0 = "/data/dsgb10/weights/rank0.dsw", golden = "/data/dsgb10/golden/prose";
  int steps = 16;
  bool module = true;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--rank0") rank0 = v;
    else if (k == "--golden") golden = v;
    else if (k == "--steps") steps = std::stoi(v);
    else if (k == "--module") module = v == "1";
    else die("unknown argument %s", k.c_str());
  }
  test_gemv();
  if (module) test_module(rank0, golden, steps);
  std::printf(failures ? "lite_tests: %d FAILED\n" : "lite_tests: all passed\n", failures);
  return failures ? 1 : 0;
}
