// Prefill tests (included at the end of test_experts.cu): sampled rows against a double reference, every row
// against the decode step (the same arithmetic up to the order of the K sums), determinism, and chunk
// invariance (a row's bits do not depend on the other rows of its chunk).

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

// one row's h for expert e (-1: the shared expert) with its range (ref_h's rule)
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
};
static PrefillOut run_prefill(const TestLayer& L, const __nv_bfloat16* dx, const int* di, const float* dw, int T, void* ws,
                              float* dyr, float* dys, const PrefillCfg& cfg = PrefillCfg()) {
  CUDA_CHECK(cudaMemset(dyr, 0xFF, (size_t)T * kH * 4));
  CUDA_CHECK(cudaMemset(dys, 0xFF, (size_t)T * kH * 4));
  moe_prefill(L.layer(), dx, di, dw, T, ws, dyr, dys, 0, cfg);
  CUDA_CHECK(cudaDeviceSynchronize());
  return PrefillOut{d2h(dyr, (size_t)T * kH), d2h(dys, (size_t)T * kH)};
}

void test_prefill() {
  TestLayer L;
  L.build(10, 3);
  L.upload();
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

  for (int T : {1, 37, 300, TB}) {
    const PrefillOut p = run_prefill(L, dx, di, dw, T, ws, dyr, dys);
    char name[96];
    // sampled rows: each pick's h (from the workspace) within the range of f32 sums of the double gate/up, and
    // the outputs against double from the GPU's h (as the decode test)
    {
      const PrefillView v = prefill_view(ws, T);
      const auto rpos = d2h(v.rpos, (size_t)T * kTopK), re = d2h(v.re, (size_t)T * kTopK);
      const auto hall = d2h(v.h, (size_t)T * kTopK * kI);
      const auto hsh = d2h(v.hsh, (size_t)T * kI);
      ErrStat er, es;
      size_t hfar = 0, hexact = 0, hn = 0;
      const int ns = std::min(T, 8);
      for (int si = 0; si < ns; ++si) {
        const int t = (int)((size_t)si * (T - 1) / std::max(1, ns - 1));
        std::vector<double> yr(kH, 0), yrm(kH, 0), ys(kH, 0), ysm(kH, 0);
        for (int k = 0; k <= kTopK; ++k) {
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
        for (int n = 0; n < kH; ++n)
          er.add(p.yr[(size_t)t * kH + n], yr[n], yrm[n], 1e-4), es.add(p.ys[(size_t)t * kH + n], ys[n], ysm[n], 1e-4);
      }
      snprintf(name, sizeof name, "prefill T=%d h (%d rows, 8 picks + shared)", T, ns);
      check(name, hfar == 0 && hexact >= hn * 0.995, "%zu of %zu bit-exact, %zu outside the range of f32 sums", hexact, hn, hfar);
      snprintf(name, sizeof name, "prefill T=%d routed out vs double (from GPU h)", T);
      check(name, !er.bad, "max |err|/sum|terms| %.2e over %zu", er.worst, er.n);
      snprintf(name, sizeof name, "prefill T=%d shared out vs double (from GPU h)", T);
      check(name, !es.bad, "max |err|/sum|terms| %.2e over %zu", es.worst, es.n);
    }
    // the default kernels (warp-specialised gate/up, chained down) against those of 2026-10-13 (k_pup, the
    // per-pick buffer + combine): h and every output bit-identical; also with narrow passes (more tickets),
    // passes claimed round-robin, and the shared expert on a side stream
    {
      PrefillCfg c0;
      c0.down = 0;
      c0.up = 0;
      const PrefillOut r = run_prefill(L, dx, di, dw, T, ws, dyr, dys, c0);
      // h by row and pick (bucket positions come from atomics and differ between calls)
      const auto h0 = d2h(prefill_view(ws, T).h, (size_t)T * kTopK * kI);
      const auto p0 = d2h(prefill_view(ws, T).rpos, (size_t)T * kTopK);
      PrefillCfg c1;
      c1.yr_kb = 1;  // 256-column passes, three claimed round-robin
      c1.chain_q = 3;
      c1.side = side;
      c1.up = 2;  // and the gate/up kernel with 64 x 64 warp tiles
      c1.down = 2;  // and the warp-specialised chained down
      const PrefillOut q = run_prefill(L, dx, di, dw, T, ws, dyr, dys, c1);
      const auto h1 = d2h(prefill_view(ws, T).h, (size_t)T * kTopK * kI);
      const auto p1 = d2h(prefill_view(ws, T).rpos, (size_t)T * kTopK);
      size_t hd = 0;
      for (size_t i = 0; i < p0.size(); ++i)
        for (int c = 0; c < kI; ++c) hd += bits(h0[(size_t)p0[i] * kI + c]) != bits(h1[(size_t)p1[i] * kI + c]);
      snprintf(name, sizeof name, "prefill T=%d h: warp-specialised gate/up (64 x 64 warps) vs k_pup", T);
      check(name, !hd, "%zu of %zu differ", hd, h0.size());
      size_t bad = 0, bad1 = 0;
      for (size_t i = 0; i < r.yr.size(); ++i) {
        bad += fbits(p.yr[i]) != fbits(r.yr[i]) || fbits(p.ys[i]) != fbits(r.ys[i]);
        bad1 += fbits(q.yr[i]) != fbits(r.yr[i]) || fbits(q.ys[i]) != fbits(r.ys[i]);
      }
      snprintf(name, sizeof name, "prefill T=%d default kernels vs those of 2026-10-13", T);
      check(name, !bad, "%zu of %zu outputs differ", bad, 2 * r.yr.size());
      snprintf(name, sizeof name, "prefill T=%d up 2, down 2 in 256-column passes, 3 round-robin, shared on a side stream", T);
      check(name, !bad1, "%zu of %zu outputs differ", bad1, 2 * r.yr.size());
    }
    // every row against the decode step: identical but for the order of the K sums
    if (T <= 300) {
      size_t same = 0, n = 0;
      double worst = 0;
      for (int t0 = 0; t0 < T; t0 += kMaxRows) {
        const int M = std::min(kMaxRows, T - t0);
        moe_decode(L.layer(), dx + (size_t)t0 * kH, di + t0 * kTopK, dw + t0 * kTopK, M, dws, dyr2, dys2, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto dr = d2h(dyr2, (size_t)M * kH), ds = d2h(dys2, (size_t)M * kH);
        for (int m = 0; m < M; ++m) {
          double mr = 0, ms = 0;  // the row's scale: largest output
          for (int c = 0; c < kH; ++c) mr = std::max(mr, (double)fabs(dr[m * kH + c])), ms = std::max(ms, (double)fabs(ds[m * kH + c]));
          for (int c = 0; c < kH; ++c) {
            const size_t i = (size_t)(t0 + m) * kH + c;
            same += fbits(p.yr[i]) == fbits(dr[m * kH + c]);
            same += fbits(p.ys[i]) == fbits(ds[m * kH + c]);
            worst = std::max(worst, fabs(p.yr[i] - dr[m * kH + c]) / mr);
            worst = std::max(worst, fabs(p.ys[i] - ds[m * kH + c]) / ms);
            n += 2;
          }
        }
      }
      snprintf(name, sizeof name, "prefill T=%d vs decode, every row", T);
      check(name, worst <= 2e-3, "%zu of %zu outputs bit-identical, max diff %.2e of the row's largest output", same, n, worst);
    }
  }
  // determinism and chunk invariance: the first 300 rows inside the 2500-row chunk, run twice
  {
    const PrefillOut a = run_prefill(L, dx, di, dw, 300, ws, dyr, dys);
    const PrefillOut b = run_prefill(L, dx, di, dw, TB, ws, dyr, dys);
    const PrefillOut c = run_prefill(L, dx, di, dw, TB, ws, dyr, dys);
    size_t bad = 0, bad2 = 0;
    for (size_t i = 0; i < b.yr.size(); ++i) bad += fbits(b.yr[i]) != fbits(c.yr[i]) || fbits(b.ys[i]) != fbits(c.ys[i]);
    for (size_t i = 0; i < a.yr.size(); ++i) bad2 += fbits(a.yr[i]) != fbits(b.yr[i]) || fbits(a.ys[i]) != fbits(b.ys[i]);
    check("prefill deterministic (T=2500 twice)", !bad, "%zu differ", bad);
    check("prefill chunk invariance (300 rows alone vs in 2500)", !bad2, "%zu differ", bad2);
  }
  // the FP4-activation path, 300 rows: the activations' NVFP4 bits against the host quantiser, h within the
  // range of f32 sums from the dequantised x, the outputs against double from the GPU's quantised h, the shared
  // expert unchanged, deterministic; and how far its outputs move from the bf16 path's (informational)
  {
    const int T = 300;
    const PrefillOut base = run_prefill(L, dx, di, dw, T, ws, dyr, dys);
    float amx = 0, amh = 0;
    for (size_t i = 0; i < (size_t)T * kH; ++i) amx = std::max(amx, fabsf(bf(x[i])));
    {
      const PrefillView v = prefill_view(ws, T);
      const auto hb = d2h(v.h, (size_t)T * kTopK * kI);
      for (auto hv : hb) amh = std::max(amh, fabsf(bf(hv)));
    }
    PrefillCfg cfg;
    cfg.fp4 = true;
    cfg.in_scale_gu = amx / (6.f * 448.f);
    cfg.in_scale_dn = amh / (6.f * 448.f);
    auto run4 = [&]() {
      CUDA_CHECK(cudaMemset(dyr, 0xFF, (size_t)T * kH * 4));
      CUDA_CHECK(cudaMemset(dys, 0xFF, (size_t)T * kH * 4));
      moe_prefill(L.layer(), dx, di, dw, T, ws, dyr, dys, 0, cfg);
      CUDA_CHECK(cudaDeviceSynchronize());
      return PrefillOut{d2h(dyr, (size_t)T * kH), d2h(dys, (size_t)T * kH)};
    };
    const PrefillOut f = run4();
    const PrefillView v = prefill_view(ws, T);
    const auto xq = d2h(v.xq, (size_t)T * kH / 2), xs = d2h(v.xs, (size_t)T * kH / 16);
    const auto hq = d2h(v.hq, (size_t)T * kTopK * kI / 2), hs = d2h(v.hs, (size_t)T * kTopK * kI / 16);
    const auto hall = d2h(v.h, (size_t)T * kTopK * kI);
    const auto rpos = d2h(v.rpos, (size_t)T * kTopK), re = d2h(v.re, (size_t)T * kTopK);
    auto quant_check = [&](const char* nm, const std::vector<__nv_bfloat16>& src, size_t groups, const std::vector<uint8_t>& q,
                           const std::vector<uint8_t>& sf, float S) {
      size_t bad = 0;
      for (size_t gi = 0; gi < groups; ++gi) {
        float xv[16];
        for (int i = 0; i < 16; ++i) xv[i] = bf(src[gi * 16 + i]);
        uint8_t q8[8], sb;
        nvfp4_quant16_host(xv, S, q8, sb);
        bad += sb != sf[gi] || memcmp(q8, &q[gi * 8], 8) != 0;
      }
      check(nm, !bad, "%zu of %zu groups differ (values or scale)", bad, groups);
    };
    quant_check("prefill fp4: x quantised as NVFP4 (host bits)", x, (size_t)T * kH / 16, xq, xs, cfg.in_scale_gu);
    quant_check("prefill fp4: h quantised as NVFP4 (host bits)", hall, (size_t)T * kTopK * kI / 16, hq, hs, cfg.in_scale_dn);
    auto deq = [](const uint8_t* q, const uint8_t* sf, float S, int K, std::vector<float>& o) {
      static const float g[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
      o.resize(K);
      for (int k = 0; k < K; ++k) o[k] = g[(q[k / 2] >> (4 * (k & 1))) & 15] * (e4m3_to_f(sf[k / 16]) * S);
    };
    ErrStat er;
    size_t hfar = 0, hexact = 0, hn = 0;
    for (int si = 0; si < 6; ++si) {
      const int t = si * (T - 1) / 5;
      std::vector<float> xf;
      deq(&xq[(size_t)t * kH / 2], &xs[(size_t)t * kH / 16], cfg.in_scale_gu, kH, xf);
      std::vector<double> yr(kH, 0), yrm(kH, 0);
      for (int k = 0; k < kTopK; ++k) {
        const int e = re[(size_t)t * kTopK + k], pos = rpos[(size_t)t * kTopK + k];
        float wk = 0;
        for (int j = 0; j < kTopK; ++j)
          if (idx[(size_t)t * kTopK + j] == e) wk = w[(size_t)t * kTopK + j];
        HRef hr;
        ref_h_one(R, xf.data(), e, hr);
        for (int c = 0; c < kI; ++c) {
          const float gv = bf(hall[(size_t)pos * kI + c]);
          hexact += fbits(gv) == fbits(hr.h[c]) || (gv == 0 && hr.h[c] == 0);
          hfar += !hr.ok(c, gv);
          ++hn;
        }
        std::vector<float> hf;
        deq(&hq[(size_t)pos * kI / 2], &hs[(size_t)pos * kI / 16], cfg.in_scale_dn, kI, hf);
        const std::vector<float>& D = R.mat(e, 2);
        const double kd = R.s2(e, 2) * (double)wk;
#pragma omp parallel for
        for (int n = 0; n < kH; ++n) {
          double s = 0, sm = 0;
          for (int c = 0; c < kI; ++c) s += (double)hf[c] * D[(size_t)n * kI + c], sm += fabs((double)hf[c] * D[(size_t)n * kI + c]);
          yr[n] += s * kd, yrm[n] += sm * fabs(kd);
        }
      }
      for (int n = 0; n < kH; ++n) er.add(f.yr[(size_t)t * kH + n], yr[n], yrm[n], 1e-4);
    }
    check("prefill fp4: h (6 rows) from the quantised x", hfar == 0 && hexact >= hn * 0.995, "%zu of %zu bit-exact, %zu outside the range of f32 sums",
          hexact, hn, hfar);
    check("prefill fp4: routed out vs double (from the GPU's quantised h)", !er.bad, "max |err|/sum|terms| %.2e over %zu", er.worst, er.n);
    size_t shd = 0;
    for (size_t i = 0; i < f.ys.size(); ++i) shd += fbits(f.ys[i]) != fbits(base.ys[i]);
    check("prefill fp4: shared expert as the bf16 path", !shd, "%zu differ", shd);
    const PrefillOut f2 = run4();
    size_t nd = 0;
    for (size_t i = 0; i < f.yr.size(); ++i) nd += fbits(f.yr[i]) != fbits(f2.yr[i]);
    check("prefill fp4: deterministic", !nd, "%zu differ", nd);
    double se = 0, sr = 0;
    for (size_t i = 0; i < f.yr.size(); ++i) se += (f.yr[i] - base.yr[i]) * (double)(f.yr[i] - base.yr[i]), sr += (double)base.yr[i] * base.yr[i];
    char msg[160];
    snprintf(msg, sizeof msg, "prefill fp4: routed out vs the bf16 path (info)");
    check(msg, true, "relative RMS difference %.3e (input scales %.3g, %.3g)", sqrt(se / sr), cfg.in_scale_gu, cfg.in_scale_dn);
  }
  cudaStreamDestroy(side);
  cudaFree(ws), cudaFree(dws), cudaFree(dyr), cudaFree(dys), cudaFree(dyr2), cudaFree(dys2), cudaFree(di), cudaFree(dw), cudaFree(dx);
  L.free_dev();
}
