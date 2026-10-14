// Binding the weights (model.h): a rank's .glw file (engine/weights; the plan in weights/glw.py), or synthetic
// weights in the same formats and layouts (benches before the files exist, and the state checks). altar-1's formats
// a layer (docs/engine.md, "What the runner binds"; DESIGN decision 7): q_b, W_UV, o_proj and the routed experts INT4
// on layers 4-76, BF16 on 0-3, 77 and the MTP layer 78; every other projection BF16 (row-major or kBF16T); W_UK^T BF16
// everywhere. A tensor in a format the plan does not give its layer is refused, never reinterpreted.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "experts/layout.h"
#include "model/impl.h"

namespace glm::model {

using namespace detail;
using ::glm::WFmt;

namespace {

// the rank files' layouts beyond glw.h's first four (weights/glw.py: INT4_TILE 4, BF16_TILE 5), by value so that
// this file builds against the container header before and after the converter's names for them land
constexpr glw::Layout kLayInt4Tile = glw::Layout(4);  // an INT4 projection: U8 [N/8][K/512][2368]
constexpr glw::Layout kLayBf16Tile = glw::Layout(5);  // a BF16 projection in kBF16T order, shape [N][K]

// the plan's format class of q_b, W_UV, o_proj and the routed experts of layer l
constexpr bool int4_layer(int l) { return l >= 4 && l <= 76; }

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
__device__ __forceinline__ float unit(uint64_t v) { return (float)(v >> 40) * (1.f / 16777216.f); }
__device__ __forceinline__ uint64_t hsh(uint64_t seed, size_t i) { return mix64(seed ^ (i * 0x632BE59BD9B4E019ull)); }
__device__ __forceinline__ uint16_t bf16_of(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// e4m3 codes of magnitude 0.25 .. 3.75 (exponents 5..8), either sign (fill_cache's FP8 index keys)
__global__ void k_fill_fp8(uint8_t* p, size_t n, uint64_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint64_t v = hsh(seed, i);
    p[i] = (uint8_t)((v & 0x80) | ((5 + (v >> 8) % 4) << 3) | ((v >> 16) & 7));
  }
}
__global__ void k_fill_f32(float* p, size_t n, uint64_t seed, float center, float half) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = center + half * (2.f * unit(hsh(seed, i)) - 1.f);
}
// uniform in center +- half; with two regions [0, n1) and [n1, n) of their own half widths (an expert's gate+up, down)
__global__ void k_fill_bf16(uint16_t* p, size_t n, uint64_t seed, float center, float half, size_t per = 0,
                            size_t n1 = 0, float half2 = 0.f) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float h = per && i % per >= n1 ? half2 : half;
    p[i] = bf16_of(center + h * (2.f * unit(hsh(seed, i)) - 1.f));
  }
}
// INT4 tiles (attn.h kINT4, experts.h kI4Blk: the same 2,368-byte block of 8 rows x 512 k): as u16 words, 1,184 a
// block, words 0-1023 the nibbles (uniform), 1024-1151 the bf16 scales (s in c * [0.75, 1.25)), 1152-1183 the
// zero-point nibbles (uniform). Weights (u - u_z) s of rms ~ 6.5 c. per, n1, c2: regions of an expert as above
// (bytes: every matrix is a whole number of blocks, so a word's block offset is its offset in the expert mod 1,184).
__global__ void k_fill_i4(uint16_t* p, size_t n, uint64_t seed, float c, size_t per = 0, size_t n1 = 0, float c2 = 0.f) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint64_t v = hsh(seed, i);
    const size_t o = per ? i % per : i;
    const int w = (int)(o % (gx::kI4Blk / 2));
    const float s = per && o >= n1 ? c2 : c;
    p[i] = w >= 1024 && w < 1152 ? bf16_of(s * (0.75f + 0.5f * unit(v))) : (uint16_t)v;
  }
}
// kBF16T [N][K] -> row-major (attn.h WFmt::kBF16T: element (n, k) at ((n/8 * K/64 + k/64) * 2 + k%16/8) * 256 +
// (4 (n%8) + k%64/16) * 8 + k%8); pack.py bf16_untile
__global__ void k_untile_bf16t(const uint16_t* t, int N, int K, uint16_t* out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * K) return;
  const int n = (int)(i / K), k = (int)(i % K);
  const size_t src = (((size_t)(n / 8) * (K / 64) + k / 64) * 2 + (k % 16) / 8) * 256 + (4 * (n % 8) + (k % 64) / 16) * 8 + k % 8;
  out[i] = t[src];
}
// RoPE table [P][32]: angle p * inv_freq[i] in f32, cos and sin rounded to bf16 (not torch's table bit for bit)
__global__ void k_rope(uint32_t* t, int P) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)P * 32) return;
  const int p = (int)(i / 32), j = (int)(i % 32);
  const float inv = 1.f / powf(8000000.f, (float)(2 * j) / 64.f);
  const float ang = (float)p * inv;
  const uint32_t c = bf16_of(cosf(ang));
  const uint32_t s = bf16_of(sinf(ang));
  t[i] = c | s << 16;
}

int grid_of(size_t n) { return (int)std::min<size_t>((n + 255) / 256, 4096); }

constexpr float kI4Rms = 6.52f;  // rms of u - u_z for independent uniform nibbles: sqrt(2 * 255 / 12)

}  // namespace

struct detail::Synth {
  std::vector<void*> ptrs;
  uint64_t seed = 1, k = 0;
  size_t bytes = 0;
  bool bf16t = true;
  ~Synth() {
    for (void* p : ptrs) cudaFree(p);
  }
  template <class T>
  T* alloc(size_t n) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T) + 256));
    ptrs.push_back(p);
    bytes += n * sizeof(T);
    return static_cast<T*>(p);
  }
  uint64_t next() { return seed * 0x9E3779B97F4A7C15ull + (++k) * 0xD1B54A32D192ED03ull; }
  uint16_t* bf16(size_t n, float center, float half) {
    uint16_t* p = alloc<uint16_t>(n);
    k_fill_bf16<<<grid_of(n), 256>>>(p, n, next(), center, half);
    return p;
  }
  float* f32(size_t n, float center, float half) {
    float* p = alloc<float>(n);
    k_fill_f32<<<grid_of(n), 256>>>(p, n, next(), center, half);
    return p;
  }
  // a projection [N][K] with outputs of rms ~ gain x the input's: BF16 (kBF16T unless row-major is asked) or INT4
  Proj proj(WFmt f, int N, int K, float gain = 1.f) {
    Proj w;
    w.fmt = f == WFmt::kBF16 && bf16t ? WFmt::kBF16T : f, w.n = N, w.k = K;
    const float rms = gain / std::sqrt((float)K);
    if (f == WFmt::kINT4) {
      const size_t words = ::glm::weight_bytes(WFmt::kINT4, N, K) / 2;
      uint16_t* p = alloc<uint16_t>(words);
      k_fill_i4<<<grid_of(words), 256>>>(p, words, next(), rms / kI4Rms);
      w.w = p;
    } else {
      w.w = bf16((size_t)N * K, 0.f, std::sqrt(3.f) * rms);
    }
    return w;
  }
};

void detail::SynthFree::operator()(Synth* s) const { delete s; }

// Synthetic routed-expert pools of all 168 experts in the kernels' layout (bind_synth; bind_file for a file whose
// experts are not filled yet): gate/up weights of rms 1/sqrt(6144), down of 0.5/sqrt(512) (outputs of rms ~ the
// input's, glm53's synthetic scale), descriptors with s2 = 1.
Model::Impl::SynthPools Model::Impl::synth_pools(const SynthOptions& so, uint32_t fmt) {
  if (!synth) synth.reset(new Synth()), synth->seed = so.seed;
  Synth& S = *synth;
  if (fmt != gx::kINT4 && fmt != gx::kBF16) die("model: synthetic routed experts in format %s", gx::fmt_name(fmt));
  const int npool = std::max(1, fmt == gx::kINT4 ? so.expert_pools : so.bf16_pools);
  const size_t gu = gx::mat_bytes(fmt, gx::kI, gx::kH), dn = gx::mat_bytes(fmt, gx::kH, gx::kI);
  const size_t per = 2 * gu + dn, pool_bytes = per * gx::kE;
  const float rgu = 1.f / std::sqrt((float)gx::kH), rdn = 0.5f / std::sqrt((float)gx::kI);
  SynthPools P;
  P.exp_bytes = (double)per;
  for (int i = 0; i < npool; ++i) {
    uint16_t* p = S.alloc<uint16_t>(pool_bytes / 2);
    if (fmt == gx::kINT4)
      k_fill_i4<<<grid_of(pool_bytes / 2), 256>>>(p, pool_bytes / 2, S.next(), rgu / kI4Rms, per / 2, gu, rdn / kI4Rms);
    else
      k_fill_bf16<<<grid_of(pool_bytes / 2), 256>>>(p, pool_bytes / 2, S.next(), 0.f, std::sqrt(3.f) * rgu, per / 2,
                                                    gu, std::sqrt(3.f) * rdn);
    std::vector<gx::MatDesc> d((size_t)gx::kE * 3);
    for (int e = 0; e < gx::kE; ++e)
      for (int j = 0; j < 3; ++j) d[e * 3 + j] = gx::make_desc(per * e + gu * j, 1.f, fmt);
    gx::MatDesc* dd = S.alloc<gx::MatDesc>(d.size());
    CUDA_CHECK(cudaMemcpy(dd, d.data(), d.size() * sizeof(gx::MatDesc), cudaMemcpyHostToDevice));
    P.pool.push_back(reinterpret_cast<const uint8_t*>(p)), P.desc.push_back(dd);
  }
  return P;
}

void Model::Impl::bind_synth(const SynthOptions& so) {
  using namespace ::glm;
  if (!synth) synth.reset(new Synth());
  Synth& S = *synth;
  S.seed = so.seed;
  S.bf16t = !so.bf16_rowmajor;
  const float wb = std::sqrt(3.f / kDim);  // bf16 matrices over the hidden: uniform, output rms ~ the input's
  embed = S.bf16((size_t)kVocabRank * kDim, 0.f, 1.f);
  head = S.proj(WFmt::kBF16, kVocabRank, kDim);
  norm_w = S.bf16(kDim, 1.f, 0.1f);
  rope_positions = opt.max_ctx;
  uint32_t* rt = S.alloc<uint32_t>((size_t)rope_positions * 32);
  k_rope<<<(unsigned)(((size_t)rope_positions * 32 + 255) / 256), 256>>>(rt, rope_positions);
  rope = rt;
  const SynthPools p4 = synth_pools(so, gx::kINT4), p16 = synth_pools(so, gx::kBF16);
  int n4 = 0, n16 = 0;  // MoE layers bound so far, by format
  for (int l = 0; l < kCacheLayers; ++l) {
    detail::LayerW& w = lw[l];
    const bool full = full_layer(l);
    const WFmt fa = int4_layer(l) ? WFmt::kINT4 : WFmt::kBF16;
    w.attn_norm = S.bf16(kDim, 1.f, 0.1f);
    w.ffn_norm = S.bf16(kDim, 1.f, 0.1f);
    w.q_norm = S.bf16(kQLora, 1.f, 0.1f);
    w.kv_norm = S.bf16(kKvLora, 1.f, 0.1f);
    w.xproj = S.proj(WFmt::kBF16, full ? kXRows : kXRowsShared, kDim);
    w.q_b = S.proj(fa, kOProjK, kQLora);
    w.w_uk = S.proj(WFmt::kBF16, kHeads * kKvLora, kNope);
    w.w_uv = S.proj(fa, kOProjK, kKvLora);
    w.o_proj = S.proj(fa, kDim, kOProjK, 0.5f);  // the four ranks' partials add
    if (full) {
      w.k_norm_w = S.bf16(kIdxDim, 1.f, 0.1f);
      w.k_norm_b = S.bf16(kIdxDim, 0.f, 0.1f);
      w.iq_b = S.proj(WFmt::kBF16, kIdxHeadsRank * kIdxDim, kQLora);
      w.wproj = S.bf16((size_t)kIdxHeadsRank * kDim, 0.f, wb);
    }
    if (dense_layer(l)) {
      w.gate_up = S.proj(WFmt::kBF16, 2 * kDenseInterRank, kDim);
      w.down = S.proj(WFmt::kBF16, kDim, kDenseInterRank, 0.5f);
      continue;
    }
    w.router = S.bf16((size_t)gx::kE * kDim, 0.f, wb);
    w.router_bias = S.f32(gx::kE, 0.f, 0.5f);
    w.sh_gate_up = S.proj(WFmt::kBF16, 2 * kSharedInterRank, kDim);
    w.sh_down = S.proj(WFmt::kBF16, kDim, kSharedInterRank, 0.5f);
    const SynthPools& P = int4_layer(l) ? p4 : p16;
    const int i = int4_layer(l) ? n4++ : n16++;
    w.moe.pool = P.pool[i % P.pool.size()], w.moe.desc = P.desc[i % P.desc.size()];
    w.moe.fmt = int4_layer(l) ? gx::kINT4 : gx::kBF16;
    w.exp_bytes = P.exp_bytes;
  }
  mtp.enorm = S.bf16(kDim, 1.f, 0.1f);
  mtp.hnorm = S.bf16(kDim, 1.f, 0.1f);
  mtp.head_norm = S.bf16(kDim, 1.f, 0.1f);
  mtp.eh = S.proj(WFmt::kBF16, kEhRows, kEhK);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  wbytes = S.bytes;
  synth_experts = true;
}

// ---- a rank file ----------------------------------------------------------------------------------------------

void Model::Impl::bind_file(glw::WeightFile& wf, const SynthOptions* fill) {
  using namespace ::glm;
  const char* path = wf.path().c_str();
  auto find = [&](const std::string& n) -> const glw::DeviceTensor& {
    const glw::DeviceTensor* t = wf.find(n);
    if (!t) die("model: %s: no tensor %s", path, n.c_str());
    if (t->reserved() || !t->data)
      die("model: %s: %s is not loaded (routed experts reserved and not yet filled?)", path, n.c_str());
    return *t;
  };
  auto get = [&](const std::string& n, size_t bytes, glw::Layout lay) -> const void* {
    const glw::DeviceTensor& t = find(n);
    if (bytes && t.bytes != bytes)
      die("model: %s: %s has %llu bytes, the kernels want %zu", path, n.c_str(), (unsigned long long)t.bytes, bytes);
    if (t.layout != lay)
      die("model: %s: %s is laid out %s (%u), the kernels want %s (%u)", path, n.c_str(), glw::layout_name(t.layout),
          (unsigned)t.layout, glw::layout_name(lay), (unsigned)lay);
    wbytes += t.bytes;
    return t.data;
  };
  auto bf = [&](const std::string& n, size_t elems) {
    return static_cast<const uint16_t*>(get(n, elems * 2, glw::Layout::Natural));
  };
  // a projection [N][K]: BF16 row-major or kBF16T, or (int4_ok) an INT4 tile; its format from its layout, which
  // must be the plan's (want_int4: the layer's q_b, W_UV, o_proj on 4-76)
  auto proj = [&](const std::string& n, int N, int K, bool want_int4) {
    const glw::DeviceTensor& t = find(n);
    Proj w;
    w.n = N, w.k = K;
    if (t.layout == kLayInt4Tile) {
      w.fmt = WFmt::kINT4;
      if (t.dtype != glw::DType::U8 || t.ndim != 3 || t.shape[0] != N / 8 || t.shape[1] != K / kInt4BlkK ||
          t.shape[2] != kInt4Blk)
        die("model: %s: %s is not an INT4 tile [%d][%d][%d]", path, n.c_str(), N / 8, K / kInt4BlkK, kInt4Blk);
    } else if (t.layout == glw::Layout::Natural || t.layout == kLayBf16Tile) {
      w.fmt = t.layout == kLayBf16Tile ? WFmt::kBF16T : WFmt::kBF16;
      if (t.dtype != glw::DType::BF16 || t.ndim != 2 || t.shape[0] != N || t.shape[1] != K)
        die("model: %s: %s is not BF16 [%d][%d]", path, n.c_str(), N, K);
    } else {
      die("model: %s: %s is laid out %s (%u): not a BF16 or INT4 projection", path, n.c_str(),
          glw::layout_name(t.layout), (unsigned)t.layout);
    }
    if ((w.fmt == WFmt::kINT4) != want_int4)
      die("model: %s: %s is %s; layer's plan: %s (docs/engine.md, \"What the runner binds\")", path, n.c_str(),
          w.fmt == WFmt::kINT4 ? "INT4" : "BF16", want_int4 ? "INT4" : "BF16");
    w.w = get(n, ::glm::weight_bytes(w.fmt, N, K), t.layout);
    return w;
  };
  // a BF16 [N][K] the kernels read row-major (the index weights): a kBF16T entry is un-tiled into a copy of its own
  auto rowmajor = [&](const std::string& n, int N, int K) -> const uint16_t* {
    const glw::DeviceTensor& t = find(n);
    if (t.layout != kLayBf16Tile) return bf(n, (size_t)N * K);
    const uint16_t* src = static_cast<const uint16_t*>(get(n, (size_t)N * K * 2, kLayBf16Tile));
    if (!synth) synth.reset(new Synth());
    uint16_t* out = synth->alloc<uint16_t>((size_t)N * K);
    k_untile_bf16t<<<(unsigned)(((size_t)N * K + 255) / 256), 256>>>(src, N, K, out);
    CUDA_CHECK(cudaGetLastError());
    wbytes += (size_t)N * K * 2;
    return out;
  };
  embed = bf("embed.weight", (size_t)kVocabRank * kDim);
  head = proj("head.weight", kVocabRank, kDim, false);
  norm_w = bf("norm.weight", kDim);
  const glw::DeviceTensor& rt = wf["rope.table"];
  rope_positions = (int)rt.shape[0];
  if (rope_positions < opt.max_ctx) die("model: the RoPE table covers %d positions, max_ctx is %d", rope_positions, opt.max_ctx);
  rope = static_cast<const uint32_t*>(get("rope.table", (size_t)rope_positions * 32 * 4, glw::Layout::Natural));
  SynthPools spool[2];  // synthetic routed experts (fill): INT4, BF16
  int nsyn[2] = {0, 0};
  for (int l = 0; l < kCacheLayers; ++l) {
    detail::LayerW& w = lw[l];
    const std::string p = "layers." + std::to_string(l) + ".";
    const bool full = full_layer(l), i4 = int4_layer(l);
    w.attn_norm = bf(p + "attn_norm.weight", kDim);
    w.ffn_norm = bf(p + "ffn_norm.weight", kDim);
    w.q_norm = bf(p + "attn.q_norm.weight", kQLora);
    w.kv_norm = bf(p + "attn.kv_norm.weight", kKvLora);
    w.xproj = proj(p + "attn.xproj.weight", full ? kXRows : kXRowsShared, kDim, false);
    w.q_b = proj(p + "attn.q_b.weight", kOProjK, kQLora, i4);
    w.w_uk = proj(p + "attn.w_uk.weight", kHeads * kKvLora, kNope, false);
    w.w_uv = proj(p + "attn.w_uv.weight", kOProjK, kKvLora, i4);
    w.o_proj = proj(p + "attn.o_proj.weight", kDim, kOProjK, i4);
    if (full) {
      w.k_norm_w = bf(p + "indexer.k_norm.weight", kIdxDim);
      w.k_norm_b = bf(p + "indexer.k_norm.bias", kIdxDim);
      w.iq_b = proj(p + "indexer.wq_b.weight", kIdxHeadsRank * kIdxDim, kQLora, false);
      w.wproj = rowmajor(p + "indexer.weights_proj.weight", kIdxHeadsRank, kDim);
    }
    if (dense_layer(l)) {
      w.gate_up = proj(p + "mlp.gate_up.weight", 2 * kDenseInterRank, kDim, false);
      w.down = proj(p + "mlp.down.weight", kDim, kDenseInterRank, false);
      continue;
    }
    w.router = bf(p + "router.weight", (size_t)gx::kE * kDim);
    w.router_bias = static_cast<const float*>(get(p + "router.bias", gx::kE * 4, glw::Layout::Natural));
    w.sh_gate_up = proj(p + "shared.gate_up.weight", 2 * kSharedInterRank, kDim, false);
    w.sh_down = proj(p + "shared.down.weight", kDim, kSharedInterRank, false);
    // the routed pool: experts 0..167, one entry each, back to back from expert 0's; the layer's one format
    const glw::DeviceTensor* e0 = wf.find(p + "experts.0");
    if (!e0) die("model: %s: no tensor %sexperts.0", path, p.c_str());
    const uint32_t tag = e0->fmt, f = tag & 15;
    // the plan: INT4 on 4-76, BF16 on 3 and 77, BF16 or NVFP4 on the MTP layer (DESIGN decision 5's option)
    const bool plan_ok = i4 ? f == gx::kINT4 : f == gx::kBF16 || (l == kMtpLayer && f == gx::kNVFP4);
    if (tag != f * 0x111u || !plan_ok)
      die("model: %s: layer %d's routed experts have format tag 0x%x; the plan wants %s", path, l, tag,
          i4 ? "INT4 (0x222)" : l == kMtpLayer ? "BF16 (0x333) or NVFP4 (0)" : "BF16 (0x333)");
    w.moe.fmt = f;
    if (fill && (e0->reserved() || !e0->data)) {  // not filled yet: synthetic routed experts of the layer's format
      if (f == gx::kNVFP4) die("model: synthetic NVFP4 experts are not built in this engine");
      SynthPools& sp = spool[f == gx::kINT4 ? 0 : 1];
      if (sp.pool.empty()) {
        const size_t before = synth ? synth->bytes : 0;
        sp = synth_pools(*fill, f);
        wbytes += synth->bytes - before;
      }
      const int i = nsyn[f == gx::kINT4 ? 0 : 1]++;
      w.moe.pool = sp.pool[i % sp.pool.size()], w.moe.desc = sp.desc[i % sp.desc.size()];
      w.exp_bytes = sp.exp_bytes;
      synth_experts = true;
      continue;
    }
    const size_t eb = gx::mat_bytes(f, gx::kI, gx::kH) * 2 + gx::mat_bytes(f, gx::kH, gx::kI);
    w.moe.pool = static_cast<const uint8_t*>(get(p + "experts.0", eb, glw::Layout::ExpertPool));
    for (int e = 1; e < gx::kE; ++e) {
      const std::string n = p + "experts." + std::to_string(e);
      const uint8_t* d = static_cast<const uint8_t*>(get(n, eb, glw::Layout::ExpertPool));
      if (d != w.moe.pool + (size_t)e * eb || wf[n].fmt != tag)
        die("model: %s: %s is not where the pool puts it, or not of the layer's format (entries back to back, %zu B "
            "each, tag 0x%x)", path, n.c_str(), eb, tag);
    }
    real_expert_layers++;
    w.exp_bytes = (double)eb;
    const glw::DeviceTensor& dt = find(p + "experts.desc");
    w.moe.desc = static_cast<const gx::MatDesc*>(get(p + "experts.desc", (size_t)gx::kE * 3 * sizeof(gx::MatDesc),
                                                     glw::Layout::Natural));
    // the descriptors: the layer's format, s2, alignment, inside the pool (the kernels trap on another format)
    std::vector<gx::MatDesc> hd((size_t)gx::kE * 3);
    CUDA_CHECK(cudaMemcpy(hd.data(), dt.data, hd.size() * sizeof(gx::MatDesc), cudaMemcpyDeviceToHost));
    for (int e = 0; e < gx::kE; ++e)
      for (int j = 0; j < 3; ++j) {
        const gx::MatDesc& d = hd[(size_t)e * 3 + j];
        const size_t mb = j < 2 ? gx::mat_bytes(f, gx::kI, gx::kH) : gx::mat_bytes(f, gx::kH, gx::kI);
        if (!gx::desc_ok(d, f) || d.off + mb > (size_t)gx::kE * eb)
          die("model: %s: layer %d expert %d matrix %d: a bad descriptor (format %u, offset %llu, s2 %g)", path, l, e, j,
              d.fmt, (unsigned long long)d.off, d.s2);
      }
  }
  const std::string m = "layers.78.mtp.";
  mtp.enorm = bf(m + "enorm.weight", kDim);
  mtp.hnorm = bf(m + "hnorm.weight", kDim);
  mtp.head_norm = bf(m + "head_norm.weight", kDim);
  mtp.eh = proj(m + "eh_proj.weight", kEhRows, kEhK, false);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void Model::fill_cache(int slot, int n, uint64_t seed) {
  Impl& I = *d_;
  if (slot < 0 || slot >= I.opt.slots || n < 0 || n > I.opt.max_ctx || (size_t)slot * I.stride + n > I.pool_positions())
    die("model: fill_cache(%d, %d)", slot, n);
  if (n == 0) return;
  for (int l = 0; l < kCacheLayers; ++l) {
    const size_t e = (size_t)n * ::glm::kLat;
    k_fill_bf16<<<grid_of(e), 256, 0, I.st>>>(I.kv[l] + (size_t)slot * I.stride * ::glm::kLat, e, seed * 131 + l, 0.f, 1.f);
    if (!full_layer(l)) continue;
    const ::glm::IdxCache c = I.ik_of(l);
    if (c.fp8) {
      k_fill_fp8<<<grid_of((size_t)n * ::glm::kIdxDim), 256, 0, I.st>>>(c.k + (size_t)slot * c.slot_stride * ::glm::kIdxDim,
                                                                       (size_t)n * ::glm::kIdxDim, seed * 977 + l);
      k_fill_f32<<<grid_of(n), 256, 0, I.st>>>(c.ks + (size_t)slot * c.slot_stride, n, seed * 37 + l, 0.01f, 0.005f);
    } else {
      k_fill_bf16<<<grid_of((size_t)n * ::glm::kIdxDim), 256, 0, I.st>>>(
          reinterpret_cast<uint16_t*>(c.k) + (size_t)slot * c.slot_stride * ::glm::kIdxDim, (size_t)n * ::glm::kIdxDim,
          seed * 977 + l, 0.f, 1.f);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(I.st));
}

}  // namespace glm::model
