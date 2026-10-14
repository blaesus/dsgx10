// Prefill tests (included at the end of test_experts.cu): sampled rows against a double reference, every row
// against the decode step (the same arithmetic up to the order of the K sums) and against the decode step in
// prefill's K order (bit for bit), the kernel variants against each other, determinism, and chunk invariance (a
// row's bits do not depend on the other rows of its chunk). Then test_equiv: an INT4 layer against the BF16 layer
// of its dequantised values, decode and prefill, every h and output bit for bit.

// picks of T rows: 8 distinct experts a row, a few hot experts (so some have many tiles), random weights
static void prefill_picks(int T, uint64_t seed, std::vector<int>& idx, std::vector<float>& w) {
  idx.resize((size_t)T * kTopK), w.resize((size_t)T * kTopK);
  for (int t = 0; t < T; ++t) {
    std::set<int> s;
    uint64_t r = mix64(seed * 7919 + t);
    while ((int)s.size() < kTopK) {
      r = mix64(r);
      const int e = (r % 3 == 0) ? (int)((r >> 8) % 6) : (int)((r >> 8) % kE);  // experts 0-5 hot
      s.insert(e);
    }
    int k = 0;
    float sum = 0;
    for (int e : s) {
      idx[(size_t)t * kTopK + k] = e;
      w[(size_t)t * kTopK + k] = 0.05f + (float)((mix64(r + k) >> 11) % 1000) / 1000.f;
      sum += w[(size_t)t * kTopK + k];
      ++k;
    }
    for (k = 0; k < kTopK; ++k) w[(size_t)t * kTopK + k] *= 2.5f / sum;
    // the order of a row's picks is the router's (by score), not by id: shuffle
    for (k = kTopK - 1; k > 0; --k) {
      const int j = (int)(mix64(r + 100 + k) % (k + 1));
      std::swap(idx[(size_t)t * kTopK + k], idx[(size_t)t * kTopK + j]);
      std::swap(w[(size_t)t * kTopK + k], w[(size_t)t * kTopK + j]);
    }
  }
}

// one row's h for expert e (-1: the fused shared expert) with its range (ref_h's rule)
static void ref_h_one(Ref& R, const float* x, int e, HRef& out) {
  const bool sh = e < 0;
  const std::vector<float>& G = sh ? R.sg : R.mat(e, 0);
  const std::vector<float>& U = sh ? R.su : R.mat(e, 1);
  const double kg = sh ? 1 : R.s2(e, 0), ku = sh ? 1 : R.s2(e, 1);
  out.h.assign(kI, 0), out.lo.assign(kI, 0), out.hi.assign(kI, 0);
#pragma omp parallel for
  for (int c = 0; c < kI; ++c) {
    double g = 0, u = 0, gm = 0, um = 0;
    for (int j = 0; j < kH; ++j) {
      const double xv = x[j], a = xv * G[(size_t)c * kH + j], b = xv * U[(size_t)c * kH + j];
      g += a, u += b, gm += fabs(a), um += fabs(b);
    }
    out.h[c] = silu_mul_ref(bf16r((float)(g * kg)), bf16r((float)(u * ku)));
    float lo = out.h[c], hi = out.h[c];
    const double g0 = (g - kSumTol * gm) * kg, g1 = (g + kSumTol * gm) * kg;
    const double u0 = (u - kSumTol * um) * ku, u1 = (u + kSumTol * um) * ku;
    const double gs[3] = {g0, g1, std::clamp(-1.2785, g0, g1)};
    for (double gg : gs)
      for (double uu : {u0, u1}) {
        const float vv = silu_mul_ref(bf16r((float)gg), bf16r((float)uu));
        lo = std::min(lo, vv), hi = std::max(hi, vv);
      }
    out.lo[c] = lo, out.hi[c] = hi;
  }
}

struct PrefillOut {
  std::vector<float> yr, ys;
  std::vector<std::vector<uint16_t>> h;  // per row and pick in expert-id order: h bits [T * 8][kI] (+ fused shared [T][kI])
};
static PrefillOut run_prefill(const TestLayer& L, const __nv_bfloat16* dx, const int* di, const float* dw, int T, void* ws,
                              float* dyr, float* dys, const PrefillCfg& cfg = PrefillCfg(), bool want_h = false) {
  CUDA_CHECK(cudaMemset(dyr, 0xFF, (size_t)T * kH * 4));
  CUDA_CHECK(cudaMemset(dys, 0xFF, (size_t)T * kH * 4));
  moe_prefill(L.layer(), dx, di, dw, T, ws, dyr, dys, 0, cfg);
  CUDA_CHECK(cudaDeviceSynchronize());
  PrefillOut o{d2h(dyr, (size_t)T * kH), d2h(dys, (size_t)T * kH), {}};
  if (want_h) {  // h by row and pick (bucket positions come from atomics and differ between calls)
    const PrefillView v = prefill_view(ws, T);
    const auto hall = d2h(v.h, (size_t)T * kTopK * kI);
    const auto rpos = d2h(v.rpos, (size_t)T * kTopK);
    o.h.resize(2);
    o.h[0].resize((size_t)T * kTopK * kI);
    for (size_t i = 0; i < rpos.size(); ++i)
      for (int c = 0; c < kI; ++c) o.h[0][i * kI + c] = bits(hall[(size_t)rpos[i] * kI + c]);
    if (L.fused) {
      const auto hsh = d2h(v.hsh, (size_t)T * kI);
      o.h[1].resize(hsh.size());
      for (size_t i = 0; i < hsh.size(); ++i) o.h[1][i] = bits(hsh[i]);
    }
  }
  return o;
}
static size_t out_diff(const PrefillOut& a, const PrefillOut& b) {
  size_t d = 0;
  for (size_t i = 0; i < a.yr.size(); ++i) d += fbits(a.yr[i]) != fbits(b.yr[i]) || fbits(a.ys[i]) != fbits(b.ys[i]);
  return d;
}
static size_t h_diff(const PrefillOut& a, const PrefillOut& b) {
  size_t d = 0;
  for (int j = 0; j < 2; ++j)
    for (size_t i = 0; i < a.h[j].size(); ++i) d += a.h[j][i] != b.h[j][i];
  return d;
}

void test_prefill(TestLayer& L, bool light) {  // light: T <= 300 only
  Ref R(L);
  const int TB = 2500;  // two down passes
  void *ws, *dws;
  CUDA_CHECK(cudaMalloc(&ws, prefill_ws_bytes(TB)));
  CUDA_CHECK(cudaMemset(ws, 0, prefill_ws_bytes(TB)));
  CUDA_CHECK(cudaMalloc(&dws, decode_ws_bytes()));
  CUDA_CHECK(cudaMemset(dws, 0, decode_ws_bytes()));
  float *dyr, *dys, *dyr2, *dys2, *dw;
  int* di;
  __nv_bfloat16* dx;
  CUDA_CHECK(cudaMalloc(&dyr, (size_t)TB * kH * 4));
  CUDA_CHECK(cudaMalloc(&dys, (size_t)TB * kH * 4));
  CUDA_CHECK(cudaMalloc(&dyr2, (size_t)kMaxRows * kH * 4));
  CUDA_CHECK(cudaMalloc(&dys2, (size_t)kMaxRows * kH * 4));
  CUDA_CHECK(cudaMalloc(&di, (size_t)TB * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dw, (size_t)TB * kTopK * 4));
  CUDA_CHECK(cudaMalloc(&dx, (size_t)TB * kH * 2));
  std::vector<__nv_bfloat16> x;
  rnd_x(x, (size_t)TB * kH, 31);
  std::vector<int> idx;
  std::vector<float> w;
  prefill_picks(TB, 32, idx, w);
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(di, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  cudaStream_t side;
  CUDA_CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));

  for (int T : light ? std::vector<int>{1, 37, 300} : std::vector<int>{1, 37, 300, TB}) {
    const PrefillOut p = run_prefill(L, dx, di, dw, T, ws, dyr, dys);
    char name[128];
    // sampled rows: each pick's h (from the workspace) within the range of f32 sums of the double gate/up, and
    // the outputs against double from the GPU's h (as the decode test)
    {
      const PrefillView v = prefill_view(ws, T);
      const auto rpos = d2h(v.rpos, (size_t)T * kTopK), re = d2h(v.re, (size_t)T * kTopK);
      const auto hall = d2h(v.h, (size_t)T * kTopK * kI);
      const auto hsh = L.fused ? d2h(v.hsh, (size_t)T * kI) : std::vector<__nv_bfloat16>();
      ErrStat er, es;
      size_t hfar = 0, hexact = 0, hn = 0;
      const int ns = std::min(T, 8);
      for (int si = 0; si < ns; ++si) {
        const int t = (int)((size_t)si * (T - 1) / std::max(1, ns - 1));
        std::vector<double> yr(kH, 0), yrm(kH, 0), ys(kH, 0), ysm(kH, 0);
        for (int k = 0; k < kTopK + (L.fused ? 1 : 0); ++k) {
          const bool sh = k == kTopK;
          const int e = sh ? -1 : re[(size_t)t * kTopK + k];
          float wk = 0;
          for (int j = 0; j < kTopK; ++j)
            if (!sh && idx[(size_t)t * kTopK + j] == e) wk = w[(size_t)t * kTopK + j];
          const __nv_bfloat16* hg = sh ? &hsh[(size_t)t * kI] : &hall[(size_t)rpos[(size_t)t * kTopK + k] * kI];
          HRef hr;
          std::vector<float> xf(kH);
          for (int j = 0; j < kH; ++j) xf[j] = bf(x[(size_t)t * kH + j]);
          ref_h_one(R, xf.data(), e, hr);
          for (int c = 0; c < kI; ++c) {
            const float gv = bf(hg[c]);
            hexact += fbits(gv) == fbits(hr.h[c]) || (gv == 0 && hr.h[c] == 0);
            hfar += !hr.ok(c, gv);
            ++hn;
          }
          const std::vector<float>& D = sh ? R.sd : R.mat(e, 2);
          const double kd = sh ? 1 : R.s2(e, 2) * (double)wk;
#pragma omp parallel for
          for (int n = 0; n < kH; ++n) {
            double s = 0, sm = 0;
            for (int c = 0; c < kI; ++c) {
              const double tt = (double)bf(hg[c]) * D[(size_t)n * kI + c];
              s += tt, sm += fabs(tt);
            }
            (sh ? ys : yr)[n] += s * kd, (sh ? ysm : yrm)[n] += sm * fabs(kd);
          }
        }
        for (int n = 0; n < kH; ++n) {
          er.add(p.yr[(size_t)t * kH + n], yr[n], yrm[n], 1e-4);
          if (L.fused) es.add(p.ys[(size_t)t * kH + n], ys[n], ysm[n], 1e-4);
        }
      }
      snprintf(name, sizeof name, "prefill T=%d h (%d rows, 8 picks%s)", T, ns, L.fused ? " + shared" : "");
      // The contract is the range (0 outside); the bit-exact share against double is a smoke statistic. Prefill sums
      // each strip's 6144 k in one f32 chain (decode default: six chains), so more of its sums land one bf16 ulp off
      // the double's rounding: 99.07-99.51 % on random INT4 and BF16 layers (2026-10-14, A3e) against glm53's 99.5 % bar.
      check(name, hfar == 0 && hexact >= hn * 0.985, "%zu of %zu bit-exact, %zu outside the range of f32 sums", hexact, hn, hfar);
      snprintf(name, sizeof name, "prefill T=%d routed out vs double (from GPU h)", T);
      check(name, !er.bad, "max |err|/sum|terms| %.2e over %zu", er.worst, er.n);
      if (L.fused) {
        snprintf(name, sizeof name, "prefill T=%d shared out vs double (from GPU h)", T);
        check(name, !es.bad, "max |err|/sum|terms| %.2e over %zu", es.worst, es.n);
      } else {
        const size_t t = ys_touched(p.ys);
        snprintf(name, sizeof name, "prefill T=%d ys untouched (no fused shared expert)", T);
        check(name, !t, "%zu of %zu written", t, p.ys.size());
      }
    }
    // the default kernels (warp-specialised gate/up, chained down) against glm53's kernels of 2026-10-13 (k_pup,
    // the per-pick buffer + combine): h and every output bit-identical; also with narrow passes (more tickets),
    // passes claimed round-robin, and a fused shared expert on a side stream
    {
      PrefillCfg c0;
      c0.down = 0;
      c0.up = 0;
      c0.sh_rows = 64;  // and the fused shared down at 64-row tiles
      const PrefillOut r = run_prefill(L, dx, di, dw, T, ws, dyr, dys, c0, true);
      const PrefillOut d = run_prefill(L, dx, di, dw, T, ws, dyr, dys, PrefillCfg(), true);
      PrefillCfg c1;
      c1.yr_kb = 1;  // 256-column passes, three claimed round-robin
      c1.chain_q = 3;
      c1.side = side;
      const PrefillOut q = run_prefill(L, dx, di, dw, T, ws, dyr, dys, c1, true);
      snprintf(name, sizeof name, "prefill T=%d h: warp-specialised gate/up vs k_pup", T);
      check(name, !h_diff(d, r) && !h_diff(q, r), "%zu, %zu of %zu differ", h_diff(d, r), h_diff(q, r), r.h[0].size() + r.h[1].size());
      snprintf(name, sizeof name, "prefill T=%d default kernels vs those of 2026-10-13", T);
      check(name, !out_diff(p, r), "%zu of %zu outputs differ", out_diff(p, r), 2 * r.yr.size());
      snprintf(name, sizeof name, "prefill T=%d 256-column passes, 3 round-robin, side stream", T);
      check(name, !out_diff(q, r), "%zu of %zu outputs differ", out_diff(q, r), 2 * r.yr.size());
    }
    // every row against the decode step: identical but for the order of the K sums
    if (T <= 300) {
      size_t same = 0, n = 0;
      double worst = 0;
      for (int t0 = 0; t0 < T; t0 += kMaxRows) {
        const int M = std::min(kMaxRows, T - t0);
        CUDA_CHECK(cudaMemset(dys2, 0, (size_t)M * kH * 4));
        moe_decode(L.layer(), dx + (size_t)t0 * kH, di + t0 * kTopK, dw + t0 * kTopK, M, dws, dyr2, dys2, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto dr = d2h(dyr2, (size_t)M * kH), ds = d2h(dys2, (size_t)M * kH);
        for (int m = 0; m < M; ++m) {
          double mr = 0, ms = 0;  // the row's scale: largest output
          for (int c = 0; c < kH; ++c) mr = std::max(mr, (double)fabs(dr[m * kH + c])), ms = std::max(ms, (double)fabs(ds[m * kH + c]));
          for (int c = 0; c < kH; ++c) {
            const size_t i = (size_t)(t0 + m) * kH + c;
            same += fbits(p.yr[i]) == fbits(dr[m * kH + c]);
            worst = std::max(worst, fabs(p.yr[i] - dr[m * kH + c]) / mr);
            ++n;
            if (L.fused) {
              same += fbits(p.ys[i]) == fbits(ds[m * kH + c]);
              worst = std::max(worst, fabs(p.ys[i] - ds[m * kH + c]) / ms);
              ++n;
            }
          }
        }
      }
      snprintf(name, sizeof name, "prefill T=%d vs decode, every row", T);
      check(name, worst <= 2e-3, "%zu of %zu outputs bit-identical, max diff %.2e of the row's largest output", same, n, worst);
    }
    // decode in prefill's K order (DecodeCfg::up = kUpPrefillOrder and the other wk1 variants): every h (each
    // pick's and a fused shared expert's) and every output bit for bit equal to the prefill's, 16-row steps
    if (T <= 300) {
      const PrefillView v = prefill_view(ws, T);
      run_prefill(L, dx, di, dw, T, ws, dyr, dys);  // the default's workspace (the variants above changed it)
      const auto rpos = d2h(v.rpos, (size_t)T * kTopK), re = d2h(v.re, (size_t)T * kTopK);
      const auto hall = d2h(v.h, (size_t)T * kTopK * kI);
      const auto hsh = L.fused ? d2h(v.hsh, (size_t)T * kI) : std::vector<__nv_bfloat16>();
      for (int uv = 0; uv < decode_up_variants(); ++uv) {
        if (!strstr(decode_up_name(uv), "wk1")) continue;
        size_t yd = 0, hd = 0, hn = 0;
        for (int t0 = 0; t0 < T; t0 += kMaxRows) {
          const int M = std::min(kMaxRows, T - t0);
          DecodeCfg c;
          c.up = uv;
          CUDA_CHECK(cudaMemset(dys2, 0xFF, (size_t)M * kH * 4));
          moe_decode(L.layer(), dx + (size_t)t0 * kH, di + t0 * kTopK, dw + t0 * kTopK, M, dws, dyr2, dys2, 0, c);
          CUDA_CHECK(cudaDeviceSynchronize());
          const auto dr = d2h(dyr2, (size_t)M * kH), ds = d2h(dys2, (size_t)M * kH);
          const auto dh = d2h(reinterpret_cast<const __nv_bfloat16*>(dws), (size_t)(kMaxUnion + 1) * kMaxRows * kI);
          for (int m = 0; m < M; ++m)
            for (int cc = 0; cc < kH; ++cc) {
              const size_t i = (size_t)(t0 + m) * kH + cc;
              yd += fbits(p.yr[i]) != fbits(dr[m * kH + cc]);
              yd += fbits(p.ys[i]) != fbits(ds[m * kH + cc]);
            }
          std::set<int> us(idx.begin() + (size_t)t0 * kTopK, idx.begin() + (size_t)(t0 + M) * kTopK);
          const std::vector<int> uni(us.begin(), us.end());
          for (int m = 0; m < M; ++m) {
            const int t = t0 + m;
            for (int k = 0; k < kTopK + (L.fused ? 1 : 0); ++k) {
              const bool sh = k == kTopK;
              const int slot = sh ? kShSlot : (int)(std::lower_bound(uni.begin(), uni.end(), re[(size_t)t * kTopK + k]) - uni.begin());
              const __nv_bfloat16* hp = sh ? &hsh[(size_t)t * kI] : &hall[(size_t)rpos[(size_t)t * kTopK + k] * kI];
              for (int cc = 0; cc < kI; ++cc) hd += bits(hp[cc]) != bits(dh[((size_t)slot * M + m) * kI + cc]), ++hn;
            }
          }
        }
        snprintf(name, sizeof name, "prefill T=%d = decode up %s, bit for bit", T, decode_up_name(uv));
        check(name, yd == 0 && hd == 0, "h: %zu of %zu differ; outputs: %zu of %zu differ", hd, hn, yd, (size_t)2 * T * kH);
      }
    }
  }
  // determinism and chunk invariance: the first 300 rows inside the 2500-row chunk, run twice
  if (!light) {
    const PrefillOut a = run_prefill(L, dx, di, dw, 300, ws, dyr, dys);
    const PrefillOut b = run_prefill(L, dx, di, dw, TB, ws, dyr, dys);
    const PrefillOut c = run_prefill(L, dx, di, dw, TB, ws, dyr, dys);
    size_t bad = 0, bad2 = 0;
    for (size_t i = 0; i < b.yr.size(); ++i) bad += fbits(b.yr[i]) != fbits(c.yr[i]) || fbits(b.ys[i]) != fbits(c.ys[i]);
    for (size_t i = 0; i < a.yr.size(); ++i) bad2 += fbits(a.yr[i]) != fbits(b.yr[i]) || fbits(a.ys[i]) != fbits(b.ys[i]);
    check("prefill deterministic (T=2500 twice)", !bad, "%zu differ", bad);
    check("prefill chunk invariance (300 rows alone vs in 2500)", !bad2, "%zu differ", bad2);
  }
  cudaStreamDestroy(side);
  cudaFree(ws), cudaFree(dws), cudaFree(dyr), cudaFree(dys), cudaFree(dyr2), cudaFree(dys2), cudaFree(di), cudaFree(dw), cudaFree(dx);
}

// An INT4 layer A against B, the BF16 layer of A's contract values (the same experts, the same map): the INT4
// kernels' only difference is the decode, so every h and output must be B's bit for bit, at every decode row count
// and variant and every prefill chunk and variant.
void test_equiv(TestLayer& A, bool light) {
  if (A.fmt != kINT4) die("test_equiv: an INT4 layer");
  TestLayer B = A.dequantised();
  B.upload();
  char name[128];
  {  // decode
    void* ws;
    CUDA_CHECK(cudaMalloc(&ws, decode_ws_bytes()));
    CUDA_CHECK(cudaMemset(ws, 0, decode_ws_bytes()));
    float *dyr, *dys;
    CUDA_CHECK(cudaMalloc(&dyr, (size_t)kMaxRows * kH * 4));
    CUDA_CHECK(cudaMalloc(&dys, (size_t)kMaxRows * kH * 4));
    Rows rw;
    rw.make(kMaxRows, 41);
    for (int M : {1, 2, 6, 9, 16}) {
      size_t hd = 0, yd = 0, hn = 0, n = 0;
      for (int uv = 0; uv < decode_up_variants(); ++uv)
        for (int dv = 0; dv < decode_down_variants(); ++dv) {
          if (uv && dv) continue;
          DecodeCfg c;
          c.up = uv, c.down = dv;
          const DecodeRun a = run_decode(A, rw.dx, rw.di, rw.dw, M, rw.idx, ws, dyr, dys, c);
          const DecodeRun b = run_decode(B, rw.dx, rw.di, rw.dw, M, rw.idx, ws, dyr, dys, c);
          for (size_t i = 0; i < a.h.size(); ++i) hd += bits(a.h[i]) != bits(b.h[i]);
          for (size_t i = 0; i < a.yr.size(); ++i) yd += fbits(a.yr[i]) != fbits(b.yr[i]) || fbits(a.ys[i]) != fbits(b.ys[i]);
          hn += a.h.size(), n += a.yr.size();
        }
      snprintf(name, sizeof name, "equiv decode M=%d, %d variants: INT4 = BF16(dequantised)", M,
               decode_up_variants() + decode_down_variants() - 1);
      check(name, !hd && !yd, "h %zu of %zu, outputs %zu of %zu differ", hd, hn, yd, n);
    }
    cudaFree(ws), cudaFree(dyr), cudaFree(dys);
    rw.free_dev();
  }
  {  // prefill
    const int TB = light ? 300 : 2500;
    void* ws;
    CUDA_CHECK(cudaMalloc(&ws, prefill_ws_bytes(TB)));
    CUDA_CHECK(cudaMemset(ws, 0, prefill_ws_bytes(TB)));
    float *dyr, *dys, *dw;
    int* di;
    __nv_bfloat16* dx;
    CUDA_CHECK(cudaMalloc(&dyr, (size_t)TB * kH * 4));
    CUDA_CHECK(cudaMalloc(&dys, (size_t)TB * kH * 4));
    std::vector<__nv_bfloat16> x;
    rnd_x(x, (size_t)TB * kH, 43);
    std::vector<int> idx;
    std::vector<float> w;
    prefill_picks(TB, 44, idx, w);
    dx = h2d(x), di = h2d(idx), dw = h2d(w);
    for (int T : light ? std::vector<int>{1, 37, 300} : std::vector<int>{1, 37, 300, TB})
      for (int v = 0; v < 2; ++v) {
        PrefillCfg c;
        if (v) c.up = 0, c.down = 0;
        const PrefillOut a = run_prefill(A, dx, di, dw, T, ws, dyr, dys, c, true);
        const PrefillOut b = run_prefill(B, dx, di, dw, T, ws, dyr, dys, c, true);
        snprintf(name, sizeof name, "equiv prefill T=%d %s: INT4 = BF16(dequantised)", T, v ? "up 0 down 0" : "default");
        check(name, !h_diff(a, b) && !out_diff(a, b), "h %zu of %zu, outputs %zu of %zu differ", h_diff(a, b), a.h[0].size(),
              out_diff(a, b), a.yr.size());
      }
    cudaFree(ws), cudaFree(dyr), cudaFree(dys), cudaFree(dw), cudaFree(di), cudaFree(dx);
  }
  B.free_dev();
}
