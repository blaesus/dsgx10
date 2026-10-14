"""Golden data for engine/attn's tests: one GLM-5.3 attention layer (a full-indexer layer) at the real dimensions
with random weights, run by a torch reference written from docs/model.md § 2 (CPU, bf16 activations, f32
accumulation, the FP8 weights dequantised exactly: W = e4m3 * scale), plus the engine's per-rank weight tiles
(tools/pack.py) for all four ranks. Rows are positions 0..S-1 of one sequence (a prefill of S rows); the last
--decode rows also serve as decode rows. With S > 2048 the top-2048 selects.

Also: an f64 "exact" attention (the absorbed formula in double precision over the same bf16 inputs and the
reference's top-k), the reference's own error against it sets the tolerance; the dense MLP, the shared expert
and rank 0's head on the last --decode rows.

usage: python3 -I gen_ref.py --out DIR [--S 4600] [--decode 16] [--seed 0] [--mlp] [--hf]
Tensors go to DIR/<name>.npy (bf16 as uint16 bit patterns), listed in DIR/manifest.json."""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import pack  # noqa: E402
import rope_table  # noqa: E402

D, H, QL, KL, R, NOPE, VD, IH, ID, TOPK, TP = 6144, 64, 2048, 512, 64, 192, 256, 32, 128, 2048, 4
QK = NOPE + R
bf = torch.bfloat16


class Rec:
    def __init__(self, out):
        self.out, self.manifest = out, {}
        os.makedirs(out, exist_ok=True)

    def __call__(self, name, a):
        if isinstance(a, torch.Tensor):
            if a.dtype == bf:
                a = a.contiguous().view(torch.int16).numpy().view(np.uint16)
            else:
                a = a.contiguous().numpy()
        a = np.ascontiguousarray(a)
        np.save(os.path.join(self.out, name + ".npy"), a)
        self.manifest[name] = {"dtype": str(a.dtype), "shape": list(a.shape)}

    def save(self, extra):
        with open(os.path.join(self.out, "manifest.json"), "w") as f:
            json.dump({"tensors": self.manifest, **extra}, f, indent=1)


def rmsnorm(x, w, eps):  # model.md 2.2: two roundings
    h = x.float()
    v = h.pow(2).mean(-1, keepdim=True)
    return w * (h * torch.rsqrt(v + eps)).to(bf)


def rope(x, cos, sin):  # model.md 2.4.1: bf16 arithmetic, de-interleaved output; x [..., 64], cos/sin [S, 32]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def lin(x, wdq):  # bf16 x, f32 dequantised weight: f32 accumulation, bf16 out
    return (x.float() @ wdq.T).to(bf)


def master(n, k, g):
    # random BF16 master with per-128-block magnitudes spread over 2^-3..2^3, so the block scales differ
    w = torch.randn(n, k, generator=g) / math.sqrt(k)
    nb, kb = (n + 127) // 128, k // 128
    f = torch.exp2(torch.randint(-3, 4, (nb, kb), generator=g).float())
    w = w * f.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return w.to(bf)


def u16(t):
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def int4_random(n, k, g):
    """a random canonical compressed-tensors INT4 tensor [n][k] (packed uint32 [n][k/8], scale bf16 bits [n][k/32],
    zero points uint32 [n/8][k/32]): every nibble value for weights and zero points, positive bf16 scales spread over
    2^-3..2^3 of ~1/(7.5 sqrt(k)) by 128-row block (so the dequantised rows have unit-ish norm, the products round)"""
    nib = torch.randint(0, 16, (n, k), generator=g).numpy().astype(np.uint8)
    znib = torch.randint(0, 16, (n, k // 32), generator=g).numpy().astype(np.uint8)
    nb = (n + 127) // 128
    f = torch.exp2(torch.randint(-3, 4, (nb, 1), generator=g).float()).repeat_interleave(128, 0)[:n]
    s = (torch.rand(n, k // 32, generator=g) + 0.5) * f / (7.5 * math.sqrt(k))
    return pack.int4_pack_nibbles(nib), u16(s.to(bf)), pack.int4_pack_zp(znib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--S", type=int, default=4600)
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fmt", choices=("int4", "bf16", "fp8"), default="int4",
                    help="the layer's weight formats: an altar-1 INT4 layer (4-76), a BF16 layer (0-3, 77, 78), or glm53's FP8")
    ap.add_argument("--mlp", action="store_true", help="also the dense MLP, the shared expert and rank 0's head")
    ap.add_argument("--hf", action="store_true", help="check the reference against transformers' GlmMoeDsaAttention")
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count()))
    g = torch.Generator().manual_seed(a.seed)
    rec = Rec(a.out)
    S, t0 = a.S, time.time()

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    # ---- weights: each tensor in its format (--fmt), then the engine's per-rank tiles (4 ranks) ----
    # int4: q_b, kv_b, o_proj in compressed-tensors INT4 (an altar-1 layer 4-76), the rest BF16; bf16: all BF16
    # (layers 0-3, 77, 78); fp8: glm53's zai FP8 from random masters
    shapes = {"q_a": (QL, D), "kv_a": (KL + R, D), "q_b": (H * QK, QL), "kv_b": (H * (NOPE + VD), KL),
              "o_proj": (D, H * VD), "wq_b": (IH * ID, QL), "wk": (ID, D)}
    fmt_of = {n: a.fmt if a.fmt == "fp8" else ("int4" if a.fmt == "int4" and n in ("q_b", "kv_b", "o_proj") else "bf16")
              for n in shapes}
    src, dq = {}, {}
    for name, (n, k) in shapes.items():
        f = fmt_of[name]
        if f == "fp8":
            src[name] = pack.fp8_from_master(master(n, k, g))
            dq[name] = torch.from_numpy(pack.dequant(*src[name]))
            rec("zai_" + name + "_scale", src[name][1])
        elif f == "bf16":
            w = master(n, k, g)
            src[name] = u16(w)
            dq[name] = w.float()
        else:
            src[name] = int4_random(n, k, g)
            dq[name] = torch.from_numpy(pack.bf16_to_f32(pack.int4_dequant(*src[name])).copy())
            for i, part in enumerate(("packed", "scale", "zp")):
                if name != "kv_b":
                    rec(f"ct_{name}_{part}", src[name][i])
    if a.fmt == "fp8":
        for name in ("q_a", "kv_b"):  # the global codes of two tensors, for the tile decode check
            rec("zai_" + name + "_codes", src[name][0])
    else:  # the reference's weights of two tensors (bf16 bits), for the tile decode checks
        rec("dq_q_a", pack.f32_to_bf16(dq["q_a"].numpy()))
        rec("dq_kv_b", pack.f32_to_bf16(dq["kv_b"].numpy()))
    log(f"weights made ({a.fmt}: {fmt_of})")
    norm = lambda n, lo=0.5, hi=1.5: (torch.rand(n, generator=g) * (hi - lo) + lo).to(bf)
    w_in, w_qn, w_kvn = norm(D), norm(QL), norm(KL)
    kn_w, kn_b = norm(ID), (torch.randn(ID, generator=g) * 0.1).to(bf)
    wp = (torch.randn(IH, D, generator=g) / math.sqrt(D)).to(bf)  # weights_proj, BF16
    for n, t in (("w_in", w_in), ("w_qn", w_qn), ("w_kvn", w_kvn), ("kn_w", kn_w), ("kn_b", kn_b), ("wp", wp)):
        rec(n, t)
    for r in range(TP):
        if a.fmt == "fp8":
            tiles = {
                "xproj": pack.xproj(src["q_a"], src["kv_a"], src["wk"], r),
                "q_b": pack.rows_block(src["q_b"], r, 4096),
                "wq_b": pack.rows_block(src["wq_b"], r, 1024),
                "w_uk": pack.w_uk(src["kv_b"], r),
                "w_uv": pack.w_uv(src["kv_b"], r),
                "o_proj": pack.cols_block(src["o_proj"], r, 4096),
            }
        else:
            i4 = a.fmt == "int4"
            tiles = {
                "xproj": (pack.bf16_xproj(src["q_a"], src["kv_a"], src["wk"], r),),
                "q_b": (pack.int4_q_b(src["q_b"], r) if i4 else pack.bf16_rows(src["q_b"], r, 4096),),
                "wq_b": (pack.bf16_rows(src["wq_b"], r, 1024),),
                "w_uk": (pack.int4_w_uk(src["kv_b"], r) if i4 else pack.bf16_w_uk(src["kv_b"], r),),
                "w_uv": (pack.int4_w_uv(src["kv_b"], r) if i4 else pack.bf16_w_uv(src["kv_b"], r),),
                "o_proj": (pack.int4_o_proj(src["o_proj"], r) if i4 else pack.bf16_cols(src["o_proj"], r, 4096),),
            }
            if i4 and r in (0, 3):  # the dequantised tiles (pack.py's own reader), for the INT4 = BF16 checks
                rec(f"r{r}_q_b_deq", pack.int4_dequant(*pack.int4_rows(src["q_b"], 4096 * r, 4096)))
                rec(f"r{r}_w_uv_deq", pack.int4_dequant(*pack.int4_untile(tiles["w_uv"][0], 4096, KL)))
                rec(f"r{r}_o_proj_deq", pack.int4_dequant(*pack.int4_cols(src["o_proj"], 4096 * r, 4096)))
        for name, t in tiles.items():
            rec(f"r{r}_{name}_w", t[0])
            if len(t) > 1:
                rec(f"r{r}_{name}_s", t[1])
    tile_fmt = {"xproj": "q_a", "q_b": "q_b", "wq_b": "wq_b", "w_uk": "kv_b", "w_uv": "kv_b", "o_proj": "o_proj"}
    fmts = {t: ("bf16" if t == "w_uk" and fmt_of[s] == "int4" else fmt_of[s]) for t, s in tile_fmt.items()}
    log(f"tiles packed {fmts}")

    # ---- inputs ----
    x_raw = (torch.randn(S, D, generator=g) * (0.5 + torch.rand(S, 1, generator=g))).to(bf)
    rec("x_raw", x_raw)
    tab = rope_table.table(S + 64)
    rec("rope", tab)
    cos = torch.from_numpy((tab[:S] & 0xFFFF).astype(np.int16)).view(bf)
    sin = torch.from_numpy((tab[:S] >> 16).astype(np.uint16).astype(np.int16)).view(bf)

    # ---- the layer (model.md 2.3-2.6) ----
    x = rmsnorm(x_raw, w_in, 1e-5)
    rec("x", x)
    qa, ckv, kpre = lin(x, dq["q_a"]), lin(x, dq["kv_a"]), lin(x, dq["wk"])
    rec("qa", qa); rec("ckv", ckv); rec("kpre", kpre)
    qr = rmsnorm(qa, w_qn, 1e-6)
    c = rmsnorm(ckv[:, :KL], w_kvn, 1e-6)
    kr = rope(ckv[:, KL:], cos, sin)
    rec("qr", qr); rec("c", c); rec("kr", kr)
    q = lin(qr, dq["q_b"])  # [S, 64*256]
    rec("q", q)
    log("projections")
    # indexer
    ln = torch.nn.LayerNorm(ID, eps=1e-6).to(bf)
    with torch.no_grad():
        ln.weight.copy_(kn_w)
        ln.bias.copy_(kn_b)
        kn = ln(kpre)
    kidx = torch.cat([rope(kn[:, :R], cos, sin), kn[:, R:]], dim=-1)
    rec("kidx", kidx)
    iq = lin(qr, dq["wq_b"]).view(S, IH, ID)
    rec("iq", iq)
    iqr = torch.cat([rope(iq[..., :R], cos[:, None, :], sin[:, None, :]), iq[..., R:]], dim=-1)
    iw = (x.float() @ wp.float().T) * (IH ** -0.5)
    rec("iw", iw)
    rows_sel = list(range(TOPK, S))
    isc = torch.empty(S, S)
    topk = torch.full((max(0, S - TOPK), TOPK), -1, dtype=torch.int32)
    kf = kidx.float()
    for r0 in range(0, S, 256):
        r1 = min(S, r0 + 256)
        sc = torch.relu(torch.matmul(iqr[r0:r1].float(), kf.T.unsqueeze(0)) * (ID ** -0.5))  # [rows, 32, S]
        I = torch.matmul(iw[r0:r1].unsqueeze(-2), sc).squeeze(-2)
        causal = torch.arange(S)[None, :] > torch.arange(r0, r1)[:, None]
        I = I.masked_fill(causal, float("-inf"))
        isc[r0:r1] = I
        for i in range(max(r0, TOPK), r1):
            topk[i - TOPK] = I[i - r0].topk(TOPK).indices.sort().values.to(torch.int32)
    rec("iscores", isc[TOPK:] if S > TOPK else isc[:0])
    rec("topk", topk)
    log("indexer")
    # attention: the reference's eager form (expanded K/V, bf16 scores, f32 softmax, bf16 P and PV)
    wkvb = dq["kv_b"].view(H, NOPE + VD, KL)
    kv = (c.float() @ dq["kv_b"].T).to(bf).view(S, H, NOPE + VD)
    k_nope, v = kv[..., :NOPE], kv[..., NOPE:]
    qh = q.view(S, H, QK)
    qrope = rope(qh[..., NOPE:], cos[:, None, :], sin[:, None, :])
    mask = torch.zeros(S, S, dtype=torch.bool)  # True = masked
    mask |= torch.arange(S)[None, :] > torch.arange(S)[:, None]
    for i in rows_sel:
        m = torch.ones(S, dtype=torch.bool)
        m[topk[i - TOPK].long()] = False
        mask[i] |= m
    o_ref = torch.empty(S, H, VD, dtype=bf)
    o_ex = torch.empty(S, H, VD)
    for h0 in range(0, H, 4):
        hs = slice(h0, h0 + 4)
        Q = torch.cat([qh[:, hs, :NOPE], qrope[:, hs]], dim=-1).transpose(0, 1)  # [4, S, 256]
        K = torch.cat([k_nope[:, hs], kr[:, None, :].expand(S, 4, R)], dim=-1).transpose(0, 1)
        Vh = v[:, hs].transpose(0, 1)
        sc = (Q.float() @ K.float().transpose(1, 2)).to(bf) * 0.0625
        sc = sc.masked_fill(mask[None], torch.finfo(bf).min)
        P = torch.softmax(sc, dim=-1, dtype=torch.float32).to(bf)
        o_ref[:, hs] = (P.float() @ Vh.float()).to(bf).transpose(0, 1)
        # exact: the absorbed function in f64 over the same bf16 q, c, k_rope and the dequantised kv_b
        for h in range(h0, h0 + 4):
            wuk, wuv = wkvb[h, :NOPE].double(), wkvb[h, NOPE:].double()
            ql = qh[:, h, :NOPE].double() @ wuk  # [S, 512]
            s64 = (ql @ c.double().T + qrope[:, h].double() @ kr.double().T) * 0.0625
            s64 = s64.masked_fill(mask, float("-inf"))
            P64 = torch.softmax(s64, dim=-1)
            o_ex[:, h] = ((P64 @ c.double()) @ wuv.T).float()
    rec("o_ref", o_ref.view(S, H * VD))
    rec("o_exact", o_ex.view(S, H * VD))
    out_ref = lin(o_ref.view(S, H * VD), dq["o_proj"])
    rec("out_ref", out_ref)
    log("attention")

    extra = {"S": S, "decode": a.decode, "seed": a.seed, "fmt": a.fmt, "fmts": fmts}
    if a.mlp:
        nd = a.decode
        xd = x[S - nd:]
        mf = "fp8" if a.fmt == "fp8" else "bf16"  # altar-1's dense MLPs and shared experts are BF16
        extra["fmts"]["mlp"] = mf
        for nm, inter, rk in (("dense", 12288, 3072), ("shared", 2048, 512)):
            ms = (master(inter, D, g), master(inter, D, g), master(D, inter, g))
            if mf == "fp8":
                wg, wu, wd = (pack.fp8_from_master(m) for m in ms)
                fg, fu, fd = (torch.from_numpy(pack.dequant(*w)) for w in (wg, wu, wd))
            else:
                wg, wu, wd = (u16(m) for m in ms)
                fg, fu, fd = (m.float() for m in ms)
            gt, up = lin(xd, fg), lin(xd, fu)
            h = (torch.nn.functional.silu(gt) * up)
            rec(f"{nm}_h", h)
            rec(f"{nm}_y", lin(h, fd))
            for r in range(TP):
                if mf == "fp8":
                    t, s, _, _ = pack.gate_up(wg, wu, r, rk)
                    rec(f"r{r}_{nm}_gu_w", t); rec(f"r{r}_{nm}_gu_s", s)
                    t, s, _, _ = pack.cols_block(wd, r, rk)
                    rec(f"r{r}_{nm}_down_w", t); rec(f"r{r}_{nm}_down_s", s)
                else:
                    rec(f"r{r}_{nm}_gu_w", pack.bf16_gate_up(wg, wu, r, rk))
                    rec(f"r{r}_{nm}_down_w", pack.bf16_cols(wd, r, rk))
            log(nm)
        head0 = (torch.randn(154880 // TP, D, generator=g) / math.sqrt(D)).to(bf)
        rec("head_r0", head0)
        rec("logits_r0", (xd.float() @ head0.float().T).to(bf))
        log("head")
    if a.hf:
        extra["hf"] = hf_check(a, x, cos, sin, dq, w_qn, w_kvn, kn_w, kn_b, wp, o_ref, out_ref, topk)
        log(f"transformers check: {extra['hf']}")
    rec.save(extra)
    log(f"wrote {len(rec.manifest)} tensors to {a.out}")


def hf_check(a, x, cos, sin, dq, w_qn, w_kvn, kn_w, kn_b, wp, o_ref, out_ref, topk):
    """Run transformers' GlmMoeDsaAttention (eager, the arbiter) with the dequantised weights rounded to bf16 and
    compare with this reference run on the same rounded weights (on the first rows only: it is slow)."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
    S = min(a.S, 2600)
    cfg = GlmMoeDsaConfig(hidden_size=D, num_attention_heads=H, num_key_value_heads=H, q_lora_rank=QL,
                          kv_lora_rank=KL, qk_nope_head_dim=NOPE, qk_rope_head_dim=R, v_head_dim=VD,
                          index_n_heads=IH, index_head_dim=ID, index_topk=TOPK, num_hidden_layers=1,
                          rope_parameters={"rope_type": "default", "rope_theta": 8000000.0},
                          indexer_types=["full"], mlp_layer_types=["dense"])
    cfg._attn_implementation = "eager"
    att = M.GlmMoeDsaAttention(cfg, 0).to(bf).eval()
    wb = {k: v.to(bf) for k, v in dq.items()}
    with torch.no_grad():
        att.q_a_proj.weight.copy_(wb["q_a"]); att.q_a_layernorm.weight.copy_(w_qn)
        att.q_b_proj.weight.copy_(wb["q_b"]); att.kv_a_proj_with_mqa.weight.copy_(wb["kv_a"])
        att.kv_a_layernorm.weight.copy_(w_kvn); att.kv_b_proj.weight.copy_(wb["kv_b"])
        att.o_proj.weight.copy_(wb["o_proj"]); att.indexer.wq_b.weight.copy_(wb["wq_b"])
        att.indexer.wk.weight.copy_(wb["wk"]); att.indexer.k_norm.weight.copy_(kn_w)
        att.indexer.k_norm.bias.copy_(kn_b); att.indexer.weights_proj.weight.copy_(wp)
        att.indexer.weights_proj.float()
        pos = torch.arange(S)[None]
        cs = (torch.cat([cos[:S], cos[:S]], -1)[None], torch.cat([sin[:S], sin[:S]], -1)[None])
        # This CPU's bf16 GEMM (nvpl/oneDNN on aarch64) does not accumulate in f32 (1.6x the error of an f32
        # accumulation, 41 % of outputs differ from the correctly rounded product); the contract is bf16 operands
        # with f32 accumulation (what torch's CUDA GEMM does), so bf16 linears and matmuls run that way here.
        F, lin0, mm0 = torch.nn.functional, torch.nn.functional.linear, torch.matmul
        def lin32(i, w, b=None):
            if i.dtype == bf and w.dtype == bf:
                return lin0(i.float(), w.float(), None if b is None else b.float()).to(bf)
            return lin0(i, w, b)
        def mm32(a1, a2, **k):
            if a1.dtype == bf and a2.dtype == bf:
                return mm0(a1.float(), a2.float(), **k).to(bf)
            return mm0(a1, a2, **k)
        F.linear, torch.matmul = lin32, mm32
        try:
            out, _, tk = att(x[None, :S], cs, None, None, pos)
        finally:
            F.linear, torch.matmul = lin0, mm0
    # this reference with the same bf16-rounded weights
    mine = run_mine(x[:S], cos[:S], sin[:S], wb, w_qn, w_kvn, kn_w, kn_b, wp)
    d = (out[0].float() - mine[0].float()).abs()
    tk_eq = 0
    if S > TOPK:
        tk_eq = sum(int(torch.equal(tk[0, i].sort().values.to(torch.int32), mine[1][i - TOPK])) for i in range(TOPK, S))
    return {"rows": S, "out_max_abs_diff": float(d.max()), "out_equal_frac": float((d == 0).float().mean()),
            "topk_rows_equal": tk_eq, "topk_rows": max(0, S - TOPK)}


def run_mine(x, cos, sin, w, w_qn, w_kvn, kn_w, kn_b, wp):
    """This file's reference on bf16 weights (a bf16 F.linear: f32 accumulation of bf16 operands)."""
    S = x.shape[0]
    L = lambda t, W: (t.float() @ W.float().T).to(bf)
    qr = rmsnorm(L(x, w["q_a"]), w_qn, 1e-6)
    ckv = L(x, w["kv_a"])
    c = rmsnorm(ckv[:, :KL], w_kvn, 1e-6)
    kr = rope(ckv[:, KL:], cos, sin)
    q = L(qr, w["q_b"]).view(S, H, QK)
    ln = torch.nn.LayerNorm(ID, eps=1e-6).to(bf)
    with torch.no_grad():
        ln.weight.copy_(kn_w); ln.bias.copy_(kn_b)
        kn = ln(L(x, w["wk"]))
    kidx = torch.cat([rope(kn[:, :R], cos, sin), kn[:, R:]], dim=-1)
    iq = L(qr, w["wq_b"]).view(S, IH, ID)
    iqr = torch.cat([rope(iq[..., :R], cos[:, None, :], sin[:, None, :]), iq[..., R:]], dim=-1)
    iw = (x.float() @ wp.float().T) * (IH ** -0.5)
    sc = torch.relu(torch.matmul(iqr.float(), kidx.float().T.unsqueeze(0)) * (ID ** -0.5))
    I = torch.matmul(iw.unsqueeze(-2), sc).squeeze(-2).masked_fill(torch.arange(S)[None] > torch.arange(S)[:, None], float("-inf"))
    tk = [None] * S
    mask = torch.arange(S)[None, :] > torch.arange(S)[:, None]
    for i in range(TOPK, S):
        tk[i - TOPK] = I[i].topk(TOPK).indices.sort().values.to(torch.int32)
        m = torch.ones(S, dtype=torch.bool); m[tk[i - TOPK].long()] = False
        mask[i] |= m
    kv = L(c, w["kv_b"]).view(S, H, NOPE + VD)
    qrope = rope(q[..., NOPE:], cos[:, None, :], sin[:, None, :])
    Q = torch.cat([q[..., :NOPE], qrope], -1).transpose(0, 1)
    K = torch.cat([kv[..., :NOPE], kr[:, None].expand(S, H, R)], -1).transpose(0, 1)
    sc = (Q.float() @ K.float().transpose(1, 2)).to(bf) * 0.0625
    sc = sc.masked_fill(mask[None], torch.finfo(bf).min)
    P = torch.softmax(sc, -1, dtype=torch.float32).to(bf)
    o = (P.float() @ kv[..., NOPE:].transpose(0, 1).float()).to(bf).transpose(0, 1).reshape(S, H * VD)
    return L(o, w["o_proj"]), tk


if __name__ == "__main__":
    main()
