// The prefill bench (included at the end of bench_experts.cu): one layer of 168 distinct experts in the format,
// chunks of T rows with 8 distinct uniformly random experts a row (balanced buckets), the whole call and each phase.
// TFLOPS count the useful MACs: T x 8 x 3 x 512 x 6144 routed, 2 FLOPs each (no fused shared expert).
static std::vector<int> int_list(const std::string& s) {
  std::vector<int> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    v.push_back(atoi(s.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return v;
}
void bench_prefill(const std::string& fname, const std::string& chunks_s, const PfOpts& o) {
  const uint32_t fmt = fmt_of(fname);
  size_t per = 0;
  for (int j = 0; j < 3; ++j) per += mat_bytes(fmt, j < 2 ? kI : kH, j < 2 ? kH : kI);
  Pool P = make_pool(fmt, (double)per * (kE - 1) + 1);  // kE distinct experts: a real layer's bytes
  MatDesc* ddesc;
  CUDA_CHECK(cudaMalloc(&ddesc, kE * 3 * sizeof(MatDesc)));
  {
    std::vector<MatDesc> d(kE * 3);
    for (int e = 0; e < kE; ++e)
      for (int j = 0; j < 3; ++j) d[e * 3 + j] = pool_desc(P, e % P.NP, j);
    CUDA_CHECK(cudaMemcpy(ddesc, d.data(), d.size() * sizeof(MatDesc), cudaMemcpyHostToDevice));
  }
  LayerW L;
  L.pool = P.d, L.desc = ddesc, L.fmt = fmt;
  printf("== prefill %s: %d experts (%.2f GB)\n", fname.c_str(), P.NP, P.total / 1e9);
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
  const std::vector<int> downs = int_list(o.down), yrkbs = int_list(o.yrkb), cqs = int_list(o.cq), ups = int_list(o.up);
  for (size_t ci = 0; ci < chunks.size(); ++ci)
    for (int uv : ups)
      for (int dn : downs)
        for (int yk : yrkbs)
          for (int q : cqs) {
            if (dn == 0 && (yk != yrkbs[0] || q != cqs[0])) continue;  // budget and cq apply to the chained down only
            const int T = chunks[ci];
            cfg.up = uv, cfg.down = dn, cfg.yr_kb = yk, cfg.chain_q = q;
            char tag[128];
            snprintf(tag, sizeof tag, "%s\tT %d\tup %d down %d yrkb %d cq %d", fname.c_str(), T, uv, dn, yk, q);
            const double fl_r = 2.0 * T * kI * kH * 3 * kTopK;
            auto run = [&](int ph) {
              return time_graph(3, [&](int, cudaStream_t st) { moe_prefill_phases(L, dx, di, dw, T, ws, dyr, dys, st, cfg, ph); }, 5);
            };
            const double all = run(kPhAll);
            const double bucket = run(kPhBucket);
            // the GEMM phases read the bucketing left in the workspace by the call before
            const double up = run(kPhUp), down = run(kPhDown), comb = dn ? 0 : run(kPhCombine);
            result("PREFILL\t%s\tall us %.0f\tTFLOPS %.1f\t%% of 107 %.1f\t73 layers ms %.1f\tms/1k rows %.2f", tag, all, fl_r / all / 1e6,
                   100 * fl_r / all / 1e6 / 107, 73 * all / 1e3, all / 1e3 * 1024 / T);
            result("PREFILL\t%s\tphases us: bucket %.0f, up %.0f (%.1f TFLOPS), down %.0f (%.1f TFLOPS), combine %.0f", tag, bucket, up,
                   fl_r * 2 / 3 / up / 1e6, down, fl_r / 3 / down / 1e6, comb);
          }
  // where the gate/up kernel's time goes (INT4, the largest chunk): parts switched off
  if (o.diag && fmt == kINT4) {
    const int T = TM;
    for (int dg : {0, 1, 2, 4}) {
      PrefillCfg c = cfg;
      c.diag = dg;
      c.up = 1;
      const double us = time_graph(3, [&](int, cudaStream_t st) { moe_prefill_phases(L, dx, di, dw, T, ws, dyr, dys, st, c, kPhUp); }, 5);
      result("PREFILL\t%s\tT %d\tup 1 kernel diag %d (%s%s%s)\tus %.0f", fname.c_str(), T, dg, dg & 1 ? "no weights " : "",
             dg & 2 ? "no MMAs " : "", dg & 4 ? "no A loads" : "", us);
    }
    for (int dg : {0, 8, 16}) {
      PrefillCfg c = cfg;
      c.ddiag = dg;
      c.down = 1;
      moe_prefill_phases(L, dx, di, dw, T, ws, dyr, dys, 0, c, kPhBucket | kPhUp);
      CUDA_CHECK(cudaDeviceSynchronize());
      const double us = time_graph(3, [&](int, cudaStream_t st) { moe_prefill_phases(L, dx, di, dw, T, ws, dyr, dys, st, c, kPhDown); }, 5);
      result("PREFILL\t%s\tT %d\tchained down diag %d (%s%s)\tus %.0f", fname.c_str(), T, dg, dg & 8 ? "no adds " : "",
             dg & 16 ? "no waits" : "", us);
    }
  }
  cudaFree(dyr), cudaFree(dys), cudaFree(di), cudaFree(dw), cudaFree(dx), cudaFree(ws), cudaFree(ddesc), cudaFree(P.d);
}
