// The prefill bench (included at the end of bench_experts.cu): one layer of 256 experts in the format, chunks of
// T rows with 8 distinct uniformly random experts a row (balanced buckets), the whole call and each phase.
// TFLOPS count the useful MACs: T x 8 x 3 x 512 x 6144 routed, T x 3 x 512 x 6144 shared, 2 FLOPs each.
void bench_prefill(const std::string& fname, const std::string& chunks_s, bool fp4) {
  const int pi = fname == "q4" ? 0 : fname == "s3" ? 2 : 3;
  const Pattern& pat = kPatterns[pi];
  size_t per = 0;  // 256 distinct experts: a real layer's bytes
  for (int j = 0; j < 3; ++j) per += mat_bytes(pat.f[j], j < 2 ? kI : kH, j < 2 ? kH : kI);
  Pool P = make_pool(pat, (double)per * (kE - 1) + 1);
  const size_t shm = (size_t)kI * kH;
  uint8_t* shw;
  float* shs;
  CUDA_CHECK(cudaMalloc(&shw, 3 * shm));
  CUDA_CHECK(cudaMalloc(&shs, 3 * 4 * 48 * 4));
  k_fill_fp8<<<192, 256>>>(shw, 3 * shm, 77);
  k_fill_f32<<<16, 256>>>(shs, 3 * 4 * 48, 1.f / 512);
  MatDesc* ddesc;
  CUDA_CHECK(cudaMalloc(&ddesc, kE * 3 * sizeof(MatDesc)));
  {
    std::vector<MatDesc> d(kE * 3);
    const float mags[2][4] = {{kSubA[0], kSubA[1], kSubA[2], kSubA[3]}, {kSubB[0], kSubB[1], kSubB[2], kSubB[3]}};
    for (int e = 0; e < kE; ++e)
      for (int j = 0; j < 3; ++j)
        d[e * 3 + j] = make_desc(P.off[(e % P.NP) * 3 + j], j < 2 ? kS2Up : kS2Down, P.fmt[(e % P.NP) * 3 + j], pat.nsub[j], mags);
    CUDA_CHECK(cudaMemcpy(ddesc, d.data(), d.size() * sizeof(MatDesc), cudaMemcpyHostToDevice));
  }
  const LayerW L{P.d, ddesc, SharedW{shw, shw + shm, shw + 2 * shm, shs, shs + 192, shs + 384}};
  printf("== prefill %s%s: %d experts (%.2f GB)\n", fname.c_str(), fp4 ? " fp4" : "", P.NP, P.total / 1e9);
  std::vector<std::string> names;
  const std::vector<int> chunks = parse_rows(chunks_s, names);
  const int TM = *std::max_element(chunks.begin(), chunks.end());
  float *dyr, *dys, *dw;
  int* di;
  __nv_bfloat16* dx;
  void* ws;
  CUDA_CHECK(cudaMalloc(&dyr, (size_t)TM * kH * 4));
  CUDA_CHECK(cudaMalloc(&dys, (size_t)TM * kH * 4));
  CUDA_CHECK(cudaMalloc(&di, (size_t)TM * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dw, (size_t)TM * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dx, (size_t)TM * kH * 2));
  CUDA_CHECK(cudaMalloc(&ws, prefill_ws_bytes(TM)));
  CUDA_CHECK(cudaMemset(ws, 0, prefill_ws_bytes(TM)));
  {
    std::vector<__nv_bfloat16> hx;
    rnd_x(hx, (size_t)TM * kH, 5);
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    std::vector<int> idx((size_t)TM * kTopK);
    std::vector<float> w((size_t)TM * kTopK, 2.5f / kTopK);
    for (int t = 0; t < TM; ++t) {
      std::set<int> s;
      uint64_t h = mix64(t + 12345);
      while ((int)s.size() < kTopK) s.insert((int)((h = mix64(h)) % kE));
      int k = 0;
      for (int e : s) idx[(size_t)t * kTopK + k++] = e;
    }
    CUDA_CHECK(cudaMemcpy(di, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  }
  PrefillCfg cfg;
  cfg.fp4 = fp4;
  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    const int T = chunks[ci];
    const double mac = (double)T * kI * kH * 3, fl_r = 2 * mac * kTopK, fl_s = 2 * mac;
    auto run = [&](int ph) {
      return time_graph(3, [&](int, cudaStream_t st) { moe_prefill_phases(L, dx, di, dw, T, ws, dyr, dys, st, cfg, ph); }, 5);
    };
    const double all = run(kPhAll);
    const double bucket = run(kPhBucket);
    // the GEMM phases read the bucketing left in the workspace by the call before
    const double up = run(kPhUp), down = run(kPhDown), comb = run(kPhCombine), sh = run(kPhShared);
    result("PREFILL\t%s%s\tT %d\tall us %.0f\tTFLOPS %.1f\t%% of 107 %.1f\t75 layers ms %.1f\tms/1k rows %.2f", fname.c_str(),
           fp4 ? " fp4" : "", T, all, (fl_r + fl_s) / all / 1e6, 100 * (fl_r + fl_s) / all / 1e6 / 107, 75 * all / 1e3,
           all / 1e3 * 1024 / T);
    result("PREFILL\t%s%s\tT %d\tphases us: bucket %.0f, up %.0f (%.1f TFLOPS), down %.0f (%.1f TFLOPS), combine %.0f, shared %.0f (%.1f TFLOPS)",
           fname.c_str(), fp4 ? " fp4" : "", T, bucket, up, fl_r * 2 / 3 / up / 1e6, down, fl_r / 3 / down / 1e6, comb, sh,
           fl_s / sh / 1e6);
  }
  // where the gate/up kernel's time goes (bf16 path, the largest chunk): parts switched off
  if (!fp4) {
    const int T = TM;
    for (int dg : {0, 1, 2, 4, 3, 5, 6}) {
      PrefillCfg c = cfg;
      c.diag = dg;
      const double us = time_graph(3, [&](int, cudaStream_t st) { moe_prefill_phases(L, dx, di, dw, T, ws, dyr, dys, st, c, kPhUp); }, 5);
      result("PREFILL\t%s\tT %d\tup kernel diag %d (%s%s%s)\tus %.0f", fname.c_str(), T, dg, dg & 1 ? "no weights " : "",
             dg & 2 ? "no MMAs " : "", dg & 4 ? "no A loads" : "", us);
    }
  }
  cudaFree(dyr), cudaFree(dys), cudaFree(di), cudaFree(dw), cudaFree(dx), cudaFree(ws), cudaFree(shw), cudaFree(shs),
      cudaFree(ddesc), cudaFree(P.d);
}
