// Binding the weights (model.h): a rank's .glw file (engine/weights; the plan in weights/glw.py), or synthetic
// weights in the same layouts (benches before the files exist, and the state checks).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "experts/layout.h"
#include "model/impl.h"

namespace glm::model {

using namespace detail;

// ---- synthetic contents ---------------------------------------------------------------------------------------

namespace {

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
__device__ __forceinline__ float unit(uint64_t v) { return (float)(v >> 40) * (1.f / 16777216.f); }

// e4m3 codes of magnitude 0.25 .. 3.75 (exponents 5..8), either sign: an rms of ~1.9
__global__ void k_fill_fp8(uint8_t* p, size_t n, uint64_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint64_t v = mix64(seed ^ (i * 0x632BE59BD9B4E019ull));
    p[i] = (uint8_t)((v & 0x80) | ((5 + (v >> 8) % 4) << 3) | ((v >> 16) & 7));
  }
}
__global__ void k_fill_f32(float* p, size_t n, uint64_t seed, float center, float half) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = center + half * (2.f * unit(mix64(seed ^ (i * 0x632BE59BD9B4E019ull))) - 1.f);
}
__global__ void k_fill_bf16(uint16_t* p, size_t n, uint64_t seed, float center, float half) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = __bfloat16_as_ushort(
        __float2bfloat16_rn(center + half * (2.f * unit(mix64(seed ^ (i * 0x632BE59BD9B4E019ull))) - 1.f)));
}
// an expert pool: blocks of blk bytes, random codes below sc_off, e4m3 scales of 1 .. 3.75 from it (subset 0)
__global__ void k_fill_pool(uint8_t* p, size_t n, uint64_t seed, uint32_t blk, uint32_t sc_off) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint64_t v = mix64(seed ^ (i * 0x632BE59BD9B4E019ull));
    p[i] = i % blk < sc_off ? (uint8_t)v : (uint8_t)(((7 + (v >> 8) % 2) << 3) | ((v >> 16) & 7));
  }
}
// RoPE table [P][32]: angle p * inv_freq[i] in f32, cos and sin rounded to bf16 (not torch's table bit for bit)
__global__ void k_rope(uint32_t* t, int P) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)P * 32) return;
  const int p = (int)(i / 32), j = (int)(i % 32);
  const float inv = 1.f / powf(8000000.f, (float)(2 * j) / 64.f);
  const float ang = (float)p * inv;
  const uint32_t c = __bfloat16_as_ushort(__float2bfloat16_rn(cosf(ang)));
  const uint32_t s = __bfloat16_as_ushort(__float2bfloat16_rn(sinf(ang)));
  t[i] = c | s << 16;
}

int grid_of(size_t n) { return (int)std::min<size_t>((n + 255) / 256, 4096); }

}  // namespace

struct detail::Synth {
  std::vector<void*> ptrs;
  uint64_t seed = 1, k = 0;
  size_t bytes = 0;
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
  // FP8 [N][K] with f32 scales (N / rs) x (K / ks): outputs of rms ~ the input's
  Fp8W fp8(int N, int K, int rs, int ks) {
    Fp8W w;
    uint8_t* p = alloc<uint8_t>((size_t)N * K);
    k_fill_fp8<<<grid_of((size_t)N * K), 256>>>(p, (size_t)N * K, next());
    const size_t ns = (size_t)(N / rs) * (K / ks);
    float* s = alloc<float>(ns);
    const float c = 0.5f / std::sqrt((float)K);
    k_fill_f32<<<grid_of(ns), 256>>>(s, ns, next(), c, 0.25f * c);
    w.w = p, w.s = s;
    return w;
  }
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
};

void detail::SynthFree::operator()(Synth* s) const { delete s; }

// Synthetic routed-expert pools in the kernels' layouts (bind_synth; bind_file for a file whose experts are not
// filled yet): so.expert_pools pools of all 256 experts, layer l using pool l % n.
void Model::Impl::synth_pools(const SynthOptions& so, std::vector<const uint8_t*>& pools,
                              std::vector<const gx::MatDesc*>& descs) {
  if (!synth) synth.reset(new Synth()), synth->seed = so.seed;
  Synth& S = *synth;
  const int npool = std::max(1, so.expert_pools);
  if (so.s3) die("model: synthetic S3 experts: S3 is GLM-5.3's format, retired in this engine");
  const uint32_t fmt = gx::kNVFP4;
  const size_t gu = gx::mat_bytes(fmt, gx::kI, gx::kH), dn = gx::mat_bytes(fmt, gx::kH, gx::kI);
  const size_t per = 2 * gu + dn, pool_bytes = per * gx::kE;
  exp_bytes = (double)per;
  pools.assign(npool, nullptr), descs.assign(npool, nullptr);
  for (int i = 0; i < npool; ++i) {
    uint8_t* p = S.alloc<uint8_t>(pool_bytes);
    const uint32_t blk = gx::kQ4Blk, sc = 2048;
    k_fill_pool<<<grid_of(pool_bytes), 256>>>(p, pool_bytes, S.next(), blk, sc);
    // weight rms ~ 2.9 (e2m1) x ~1.4 (scales) x s2: gate/up and down outputs of rms ~ their inputs'
    const float s2gu = 1.f / (4.2f * std::sqrt((float)gx::kH)), s2dn = 1.f / (4.2f * std::sqrt(4.f * gx::kI));
    std::vector<gx::MatDesc> d((size_t)gx::kE * 3);
    for (int e = 0; e < gx::kE; ++e)
      for (int j = 0; j < 3; ++j)
        d[e * 3 + j] = gx::make_desc(per * e + gu * j, j < 2 ? s2gu : s2dn, fmt);
    gx::MatDesc* dd = S.alloc<gx::MatDesc>(d.size());
    CUDA_CHECK(cudaMemcpy(dd, d.data(), d.size() * sizeof(gx::MatDesc), cudaMemcpyHostToDevice));
    pools[i] = p, descs[i] = dd;
  }
}

void Model::Impl::bind_synth(const SynthOptions& so) {
  using namespace ::glm;
  if (!synth) synth.reset(new Synth());
  Synth& S = *synth;
  S.seed = so.seed;
  const float wb = std::sqrt(3.f / kDim);  // bf16 matrices over the hidden: uniform, output rms ~ the input's
  embed = S.bf16((size_t)kVocabRank * kDim, 0.f, 1.f);
  head = S.bf16((size_t)kVocabRank * kDim, 0.f, wb);
  norm_w = S.bf16(kDim, 1.f, 0.1f);
  rope_positions = opt.max_ctx;
  uint32_t* rt = S.alloc<uint32_t>((size_t)rope_positions * 32);
  k_rope<<<(unsigned)(((size_t)rope_positions * 32 + 255) / 256), 256>>>(rt, rope_positions);
  rope = rt;
  // the routed-expert pools, shared by layers l % n
  std::vector<const uint8_t*> pools;
  std::vector<const gx::MatDesc*> descs;
  synth_pools(so, pools, descs);
  const int npool = (int)pools.size();
  for (int l = 0; l < kCacheLayers; ++l) {
    detail::LayerW& w = lw[l];
    const bool full = full_layer(l);
    w.attn_norm = S.bf16(kDim, 1.f, 0.1f);
    w.ffn_norm = S.bf16(kDim, 1.f, 0.1f);
    w.q_norm = S.bf16(kQLora, 1.f, 0.1f);
    w.kv_norm = S.bf16(kKvLora, 1.f, 0.1f);
    w.xproj = S.fp8(full ? kXRows : kXRowsShared, kDim, 16, 128);
    w.q_b = S.fp8(kOProjK, kQLora, 128, 128);
    w.w_uk = S.fp8(kHeads * kKvLora, kNope, 128, 64);
    w.w_uv = S.fp8(kOProjK, kKvLora, 64, 128);
    w.o_proj = S.fp8(kDim, kOProjK, 128, 128);
    if (full) {
      w.k_norm_w = S.bf16(kIdxDim, 1.f, 0.1f);
      w.k_norm_b = S.bf16(kIdxDim, 0.f, 0.1f);
      w.iq_b = S.fp8(kIdxHeadsRank * kIdxDim, kQLora, 128, 128);
      w.wproj = S.bf16((size_t)kIdxHeadsRank * kDim, 0.f, wb);
    }
    if (dense_layer(l)) {
      w.gate_up = S.fp8(2 * kDenseInterRank, kDim, 128, 128);
      w.down = S.fp8(kDim, kDenseInterRank, 128, 128);
    } else {
      w.router = S.bf16((size_t)gx::kE * kDim, 0.f, wb);
      w.router_bias = S.f32(gx::kE, 0.f, 0.5f);
      const Fp8W g = S.fp8(gx::kI, gx::kH, 128, 128), u = S.fp8(gx::kI, gx::kH, 128, 128), d = S.fp8(gx::kH, gx::kI, 128, 128);
      w.moe.sh = gx::SharedW{g.w, u.w, d.w, g.s, u.s, d.s};
      w.moe.pool = pools[l % npool];
      w.moe.desc = descs[l % npool];
    }
  }
  mtp.enorm = S.bf16(kDim, 1.f, 0.1f);
  mtp.hnorm = S.bf16(kDim, 1.f, 0.1f);
  mtp.head_norm = S.bf16(kDim, 1.f, 0.1f);
  mtp.eh = S.fp8(kEhRows, kEhK, 128, 128);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  wbytes = S.bytes;
}

// ---- a rank file ----------------------------------------------------------------------------------------------

void Model::Impl::bind_file(glw::WeightFile& wf, const SynthOptions* fill) {
  using namespace ::glm;
  std::vector<const uint8_t*> spools;
  std::vector<const gx::MatDesc*> sdescs;
  auto get = [&](const std::string& n, size_t bytes, glw::Layout lay) -> const void* {
    const glw::DeviceTensor* t = wf.find(n);
    if (!t) die("model: %s: no tensor %s", wf.path().c_str(), n.c_str());
    if (t->reserved() || !t->data)
      die("model: %s: %s is not loaded (routed experts reserved and not yet filled?)", wf.path().c_str(), n.c_str());
    if (bytes && t->bytes != bytes)
      die("model: %s: %s has %llu bytes, the kernels want %zu", wf.path().c_str(), n.c_str(), (unsigned long long)t->bytes,
          bytes);
    if (t->layout != lay) die("model: %s: %s is laid out %s, the kernels want %s", wf.path().c_str(), n.c_str(),
                              glw::layout_name(t->layout), glw::layout_name(lay));
    wbytes += t->bytes;
    return t->data;
  };
  auto bf = [&](const std::string& n, size_t elems) { return static_cast<const uint16_t*>(get(n, elems * 2, glw::Layout::Natural)); };
  auto f8 = [&](const std::string& n, int N, int K, int rs, int ks, glw::Layout lay = glw::Layout::T64) {
    Fp8W w;
    w.w = static_cast<const uint8_t*>(get(n + ".weight", (size_t)N * K, lay));
    w.s = static_cast<const float*>(get(n + ".scale", (size_t)(N / rs) * (K / ks) * 4, glw::Layout::Natural));
    return w;
  };
  embed = bf("embed.weight", (size_t)kVocabRank * kDim);
  head = bf("head.weight", (size_t)kVocabRank * kDim);
  norm_w = bf("norm.weight", kDim);
  const glw::DeviceTensor& rt = wf["rope.table"];
  rope_positions = (int)rt.shape[0];
  if (rope_positions < opt.max_ctx) die("model: the RoPE table covers %d positions, max_ctx is %d", rope_positions, opt.max_ctx);
  rope = static_cast<const uint32_t*>(get("rope.table", (size_t)rope_positions * 32 * 4, glw::Layout::Natural));
  for (int l = 0; l < kCacheLayers; ++l) {
    detail::LayerW& w = lw[l];
    const std::string p = "layers." + std::to_string(l) + ".";
    const bool full = full_layer(l);
    w.attn_norm = bf(p + "attn_norm.weight", kDim);
    w.ffn_norm = bf(p + "ffn_norm.weight", kDim);
    w.q_norm = bf(p + "attn.q_norm.weight", kQLora);
    w.kv_norm = bf(p + "attn.kv_norm.weight", kKvLora);
    w.xproj = f8(p + "attn.xproj", full ? kXRows : kXRowsShared, kDim, 16, 128);
    w.q_b = f8(p + "attn.q_b", kOProjK, kQLora, 128, 128);
    w.w_uk = f8(p + "attn.w_uk", kHeads * kKvLora, kNope, 128, 64);
    w.w_uv = f8(p + "attn.w_uv", kOProjK, kKvLora, 64, 128);
    w.o_proj = f8(p + "attn.o_proj", kDim, kOProjK, 128, 128);
    if (full) {
      w.k_norm_w = bf(p + "indexer.k_norm.weight", kIdxDim);
      w.k_norm_b = bf(p + "indexer.k_norm.bias", kIdxDim);
      w.iq_b = f8(p + "indexer.wq_b", kIdxHeadsRank * kIdxDim, kQLora, 128, 128);
      w.wproj = bf(p + "indexer.weights_proj.weight", (size_t)kIdxHeadsRank * kDim);
    }
    if (dense_layer(l)) {
      w.gate_up = f8(p + "mlp.gate_up", 2 * kDenseInterRank, kDim, 128, 128);
      w.down = f8(p + "mlp.down", kDim, kDenseInterRank, 128, 128);
      continue;
    }
    w.router = bf(p + "router.weight", (size_t)gx::kE * kDim);
    w.router_bias = static_cast<const float*>(get(p + "router.bias", gx::kE * 4, glw::Layout::Natural));
    if (!wf.find(p + "shared.gate.weight"))
      die("model: %s: layer %d's shared expert is not in the experts' T128 tiles (convert with the shared layout t128: "
          "the decode step fuses it into the expert kernels)", wf.path().c_str(), l);
    const Fp8W g = f8(p + "shared.gate", gx::kI, gx::kH, 128, 128, glw::Layout::T128);
    const Fp8W uu = f8(p + "shared.up", gx::kI, gx::kH, 128, 128, glw::Layout::T128);
    const Fp8W d = f8(p + "shared.down", gx::kH, gx::kI, 128, 128, glw::Layout::T128);
    w.moe.sh = gx::SharedW{g.w, uu.w, d.w, g.s, uu.s, d.s};
    // the pool: experts 0..255, one entry each, back to back from expert 0's (docs/layout.md § 6)
    const glw::DeviceTensor* e0 = wf.find(p + "experts.0");
    if (!e0) die("model: %s: no tensor %sexperts.0", wf.path().c_str(), p.c_str());
    if (fill && (e0->reserved() || !e0->data)) {  // not filled yet: synthetic routed experts
      if (spools.empty()) {
        const size_t before = synth ? synth->bytes : 0;
        synth_pools(*fill, spools, sdescs);
        wbytes += synth->bytes - before;
      }
      w.moe.pool = spools[l % spools.size()], w.moe.desc = sdescs[l % sdescs.size()];
      synth_experts = true;
      continue;
    }
    w.moe.pool = static_cast<const uint8_t*>(get(p + "experts.0", 0, glw::Layout::ExpertPool));
    for (int e = 1; e < gx::kE; ++e) {
      const std::string n = p + "experts." + std::to_string(e);
      const uint8_t* d = static_cast<const uint8_t*>(get(n, 0, glw::Layout::ExpertPool));
      if (d != w.moe.pool + (size_t)e * e0->bytes || wf[n].bytes != e0->bytes)
        die("model: %s: %s is not where the pool puts it (expert entries back to back, %llu B each)", wf.path().c_str(),
            n.c_str(), (unsigned long long)e0->bytes);
    }
    real_expert_layers++;
    exp_bytes = (double)e0->bytes;
    w.moe.desc = static_cast<const gx::MatDesc*>(get(p + "experts.desc", (size_t)gx::kE * 3 * sizeof(gx::MatDesc), glw::Layout::Natural));
  }
  const std::string m = "layers.78.mtp.";
  mtp.enorm = bf(m + "enorm.weight", kDim);
  mtp.hnorm = bf(m + "hnorm.weight", kDim);
  mtp.head_norm = bf(m + "head_norm.weight", kDim);
  mtp.eh = f8(m + "eh_proj", kEhRows, kEhK, 128, 128);
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
