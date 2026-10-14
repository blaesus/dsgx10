"""Layer 0 op by op: the engine's prefill dump (glm_run --dump-ops, every rank's raw files) against glmref.py's
--dump-ops (the expanded form, and the --attn absorbed prototype), then the attention's roundings one at a time
(docs/engine.md § 11, experiment A9).

  python -I cmp_ops.py --engine DIR --ref DIR [--abs DIR] [--golden RUN] [--ckpt CKPT --verified LOG] [--json OUT]

--engine: r<rank>.<name>.bin for ranks 0-3 (engine/model/prefill.cu, --dump-ops); --ref / --abs: glmref.py's
<seq>-L00 directories; --golden: a glmref run (a1a-f32) whose layers/L00/hidden.npy holds the sequence's rows first;
--ckpt: the checkpoint, for the attention variants (kv_b_proj, o_proj and the MLP of layer 0 are read).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

BF16, F32 = torch.bfloat16, torch.float32
D, NH, RQ, RKV, DN, DR, DQK, DV = 6144, 64, 2048, 512, 192, 64, 256, 256
R, HR = 4, 16              # ranks, heads a rank
XR = 512 + 128 + 16 + 32   # a full layer's x-projection rows a rank (q_a, kv latent, k_rope, wk)
DI = 12288


def bf16_from_raw(path, shape):
    a = np.fromfile(path, dtype=np.int16)
    return torch.from_numpy(a).view(BF16).reshape(shape)


def f32_from_raw(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(shape)


def load_ref(d):
    with open(os.path.join(d, "ops.json")) as f:
        meta = json.load(f)
    out = {}
    for k, shp in meta["shapes"].items():
        a = np.load(os.path.join(d, k + ".npy"))
        t = torch.from_numpy(a)
        out[k] = t.view(BF16) if k in meta["bf16_as_int16"] else t
    return out


def load_engine(d, M):
    e = {}
    p = lambda r, n: os.path.join(d, f"r{r}.{n}.bin")
    for n in ("h_in", "xn_in", "h_attn", "xn_mlp", "h_out"):
        e[n] = bf16_from_raw(p(0, n), (M, D))
        for r in range(1, R):  # replicated: every rank the same
            e[n + "_ranks_equal"] = e.get(n + "_ranks_equal", True) and torch.equal(
                bf16_from_raw(p(r, n), (M, D)).view(torch.int16), e[n].view(torch.int16))
    xg = bf16_from_raw(p(0, "xg"), (M, R, XR))
    e["qa"] = torch.cat([xg[:, r, 0:512] for r in range(R)], 1)
    e["ckv"] = torch.cat([torch.cat([xg[:, r, 512:640] for r in range(R)], 1),
                          torch.cat([xg[:, r, 640:656] for r in range(R)], 1)], 1)
    e["cq"] = bf16_from_raw(p(0, "qr"), (M, RQ))
    kv = bf16_from_raw(p(0, "kv"), (M, RKV + DR))
    e["c"], e["k_rot"] = kv[:, :RKV], kv[:, RKV:]
    e["q_raw"] = torch.cat([bf16_from_raw(p(r, "q"), (M, HR, DQK)) for r in range(R)], 1)
    e["qlat"] = torch.cat([bf16_from_raw(p(r, "qlat"), (M, HR, RKV)) for r in range(R)], 1)
    e["u"] = torch.cat([bf16_from_raw(p(r, "u"), (M, HR, RKV)) for r in range(R)], 1)
    e["o"] = torch.cat([bf16_from_raw(p(r, "o"), (M, HR * DV)) for r in range(R)], 1)
    parts = [f32_from_raw(p(r, "attn_part"), (M, D)) for r in range(R)]
    e["attn_out"] = ((parts[0] + parts[1]) + parts[2]) + parts[3]   # the reduce-scatter's rank order
    e["mlp_h"] = torch.cat([bf16_from_raw(p(r, "hm"), (M, DI // R)) for r in range(R)], 1)
    parts = [f32_from_raw(p(r, "mlp_part"), (M, D)) for r in range(R)]
    e["mlp_out"] = ((parts[0] + parts[1]) + parts[2]) + parts[3]
    if os.path.exists(p(0, "topk")) and M > 2048:  # the index sets of the rows at positions >= 2048
        e["topk"] = torch.from_numpy(np.fromfile(p(0, "topk"), dtype=np.int32).reshape(M, 2048)[2048:].copy())
    return e


def cmp(a, b):
    """a: the engine (or the variant), b: the reference. Bit-equal share, max |diff|, rows' relative error."""
    a32, b32 = a.to(F32).reshape(a.shape[0], -1), b.to(F32).reshape(b.shape[0], -1)
    if a.dtype == b.dtype:
        eq = (a.reshape(a.shape[0], -1).view(torch.int16 if a.dtype == BF16 else torch.int32) ==
              b.reshape(b.shape[0], -1).view(torch.int16 if b.dtype == BF16 else torch.int32))
    else:
        eq = a32 == b32
    d = (a32 - b32)
    rel = d.norm(dim=1) / b32.norm(dim=1).clamp(min=1e-30)
    return {"equal": float(eq.float().mean()), "max_abs": float(d.abs().max()),
            "rel_median": float(rel.median()), "rel_max": float(rel.max()), "n": int(eq.numel())}


def fmt(name, c):
    return (f"{name:<10} bit-equal {c['equal'] * 100:8.4f} %  max|d| {c['max_abs']:.3e}  rel median "
            f"{c['rel_median'] * 100:.4f} %  max {c['rel_max'] * 100:.4f} %")


# ---------------------------------------------------------------------------------------------- attention variants

def attn_variant(q_raw, c, k_rot, Wkvb, cos, sin, absorbed, score_bf16, online, norm_p):
    """o [S, 64 * 256] bf16 from q (q_b's output, before RoPE), the cache's c and k_rope, by one arithmetic:
    absorbed (q_lat, u) or expanded (K, V); scores rounded to bf16 or kept f32; the softmax online in blocks of 64
    with P = bf16(exp(s - m)) unnormalised and u = bf16(O / l), or whole with P = bf16(softmax) (norm_p)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import glmref
    S = q_raw.shape[0]
    q = torch.cat([q_raw[..., :DN], glmref.rope(q_raw[..., DN:], cos[:, None], sin[:, None])], -1)  # [S, 64, 256]
    Wkv = Wkvb.view(NH, DN + DV, RKV)
    Wuk, Wuv = Wkv[:, :DN], Wkv[:, DN:]
    mask = torch.arange(S)[None, :] > torch.arange(S)[:, None]                       # [S(q), S(k)] blocked
    qh = q.transpose(0, 1)                                                           # [64, S, 256]
    if absorbed:
        qlat = torch.matmul(qh[..., :DN], Wuk)                                       # [64, S, 512] bf16
        Q = torch.cat([qlat, qh[..., DN:]], -1)                                      # [64, S, 576]
        Kk = torch.cat([c, k_rot], -1)                                               # [S, 576]
        Vv = c                                                                       # [S, 512]
    else:
        kv = torch.matmul(c, Wkvb.T).view(S, NH, DN + DV)                            # bf16
        Q = qh
        Kk = torch.cat([kv[..., :DN], k_rot[:, None].expand(S, NH, DR)], -1).transpose(0, 1)   # [64, S, 256]
        Vv = kv[..., DN:].transpose(0, 1)                                            # [64, S, 256]
    if score_bf16:
        s = torch.matmul(Q, Kk.transpose(-1, -2)) * glmref.ATT_SCALE                 # bf16 output, x 1/16 exact
        s = s.to(F32)
    else:
        s = torch.matmul(Q.to(F32), Kk.to(F32).transpose(-1, -2)) * glmref.ATT_SCALE
    s = s.masked_fill(mask, float("-inf"))
    if not online:
        p = torch.softmax(s, dim=-1)
        P = p.to(BF16)
        x = torch.matmul(P, Vv if Vv.dim() == 3 else Vv.expand(NH, S, Vv.shape[-1]))   # bf16 (f32 accumulation)
        if not norm_p:
            raise ValueError("online=False needs norm_p")
    else:
        Vf = Vv.to(F32)
        m = torch.full((NH, S, 1), -1e30)
        l = torch.zeros(NH, S, 1)
        O = torch.zeros(NH, S, Vv.shape[-1])
        for k0 in range(0, S, 64):
            sb = s[..., k0:k0 + 64]
            mn = torch.maximum(m, sb.amax(-1, keepdim=True))
            p = torch.exp(sb - mn)
            al = torch.exp(m - mn)
            l = l * al + p.sum(-1, keepdim=True)
            Pb = p.to(BF16).to(F32)
            O = O * al + torch.matmul(Pb, Vf[..., k0:k0 + 64, :] if Vf.dim() == 3 else Vf[k0:k0 + 64])
            m = mn
        x = (O / l).to(BF16)
    if absorbed:
        o = torch.matmul(x, Wuv.transpose(1, 2))                                     # [64, S, 256] bf16
        return o.transpose(0, 1).reshape(S, NH * DV), x.transpose(0, 1)
    return x.transpose(0, 1).reshape(S, NH * DV), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--abs", default=None)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--engine-tf", default=None, help="the engine's tf dump of the same job (layers/L00/hidden.npy)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--verified", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-variants", action="store_true", help="skip the attention variants (dense masks: short only)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    ref = load_ref(a.ref)
    M = ref["h_in"].shape[0]
    eng = load_engine(a.engine, M)
    res = {"rows": M, "replicated_equal_on_ranks": {k: v for k, v in eng.items() if k.endswith("_ranks_equal")}}
    print("replicated tensors equal on the four ranks:", res["replicated_equal_on_ranks"])
    order = ["h_in", "xn_in", "qa", "ckv", "cq", "c", "k_rot", "q_raw", "o", "attn_out", "h_attn", "xn_mlp",
             "mlp_h", "mlp_out", "h_out"]
    res["engine_vs_ref"] = {}
    print("\n== the engine against the reference (expanded form, --combine f32)")
    for k in order:
        b = ref[k]
        if k == "q_raw":
            b = b.reshape(M, NH, DQK)
        if k == "ckv":
            b = b.reshape(M, -1)
        c = cmp(eng[k], b)
        res["engine_vs_ref"][k] = c
        print(fmt(k, c))
    if a.abs:
        ab = load_ref(a.abs)
        res["engine_vs_absorbed"], res["absorbed_vs_ref"] = {}, {}
        print("\n== the engine against the reference with --attn absorbed (the engine's attention arithmetic)")
        for k in ["q_raw", "qlat", "u", "o", "attn_out", "h_attn", "xn_mlp", "mlp_h", "mlp_out", "h_out"]:
            b = ab[k]
            if k == "q_raw":
                b = b.reshape(M, NH, DQK)
            c = cmp(eng[k], b)
            res["engine_vs_absorbed"][k] = c
            print(fmt(k, c))
        print("\n== --attn absorbed against the expanded reference")
        for k in ["o", "attn_out", "h_attn", "h_out"]:
            c = cmp(ab[k], ref[k])
            res["absorbed_vs_ref"][k] = c
            print(fmt(k, c))
    if a.golden:
        # the golden run's saved rows of this sequence (sample_rows.npy: global rows; the manifest: offsets)
        seq = json.load(open(os.path.join(a.ref, "ops.json")))["sequence"]
        man = json.load(open(os.path.join(a.golden, "manifest.json")))
        off = next(q["off"] for q in man["sequences"] if q["name"] == seq)
        srows = np.load(os.path.join(a.golden, "sample_rows.npy")).astype(np.int64)
        sel = (srows >= off) & (srows < off + M)
        ps = torch.from_numpy(srows[sel] - off)
        g = torch.from_numpy(np.load(os.path.join(a.golden, "layers", "L00", "hidden.npy"))[sel])
        res["golden_rows"] = int(len(ps))
        res["golden"] = {"ref_h_out": cmp(ref["h_out"][ps].to(F32), g), "engine_h_out": cmp(eng["h_out"][ps].to(F32), g)}
        if a.abs:
            res["golden"]["absorbed_h_out"] = cmp(ab["h_out"][ps].to(F32), g)
        if a.engine_tf:
            t = torch.from_numpy(np.load(os.path.join(a.engine_tf, "layers", "L00", "hidden.npy"))[:M])
            res["golden"]["engine_tf_h_out"] = cmp(t, g)
            res["golden"]["engine_tf_vs_dump"] = cmp(t, eng["h_out"].to(F32))
        print("\n== layer 0's output against the golden set (", a.golden, ")")
        for k, v in res["golden"].items():
            print(fmt(k, v))
        if "topk" in eng:
            seq = json.load(open(os.path.join(a.ref, "ops.json")))["sequence"]
            gt = torch.from_numpy(np.load(os.path.join(a.golden, "layers", "L00", f"topk_{seq}.npy")).astype(np.int64))
            et = eng["topk"].long()
            only = [len(set(et[i].tolist()) - set(gt[i].tolist())) for i in range(gt.shape[0])]
            mg = np.load(os.path.join(a.golden, "layers", "L00", f"topk_margin_{seq}.npy"))
            res["topk"] = {"rows": int(gt.shape[0]), "sets_equal": sum(1 for n in only if n == 0),
                           "positions_swapped": int(sum(only)), "max_swapped_in_a_row": int(max(only)),
                           "golden_margin_of_differing_rows": [float(mg[i]) for i, n in enumerate(only) if n],
                           "golden_margin_median": float(np.median(mg))}
            print("index sets of rows >= 2048 against the golden set:", res["topk"])
    if a.ckpt:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import glmref
        ck = glmref.Ckpt(a.ckpt, None, a.verified)
        Wkvb = ck.get("model.layers.0.self_attn.kv_b_proj.weight")
        cos, sin = glmref.rope_tables(M)
        # from the engine's own q_b output and cache entries, so only the attention's arithmetic varies
        q_raw, c, k_rot = eng["q_raw"], eng["c"], eng["k_rot"]
        variants = {} if a.no_variants else {
            "expanded (reference)": dict(absorbed=False, score_bf16=True, online=False, norm_p=True),
            "expanded, f32 scores": dict(absorbed=False, score_bf16=False, online=False, norm_p=True),
            "expanded, f32 scores, online P": dict(absorbed=False, score_bf16=False, online=True, norm_p=False),
            "absorbed, bf16 scores, softmax P": dict(absorbed=True, score_bf16=True, online=False, norm_p=True),
            "absorbed, f32 scores, softmax P": dict(absorbed=True, score_bf16=False, online=False, norm_p=True),
            "absorbed, f32 scores, online P (engine)": dict(absorbed=True, score_bf16=False, online=True, norm_p=False),
        }
        res["variants_vs_engine_o"] = {}
        res["variants_vs_reference_o"] = {}
        print("\n== attention variants from the engine's own q and cache entries: o against the engine's o, the reference's o")
        outs = {}
        for name, kw in variants.items():
            o, u = attn_variant(q_raw, c, k_rot, Wkvb, cos, sin, **kw)
            outs[name] = o
            ce, cr = cmp(eng["o"], o), cmp(o, ref["o"])
            res["variants_vs_engine_o"][name], res["variants_vs_reference_o"][name] = ce, cr
            print(f"  {name}")
            print("    " + fmt("vs engine", ce))
            print("    " + fmt("vs ref", cr))
            if u is not None:
                cu = cmp(eng["u"], u)
                res["variants_vs_engine_o"][name + " (u)"] = cu
                print("    " + fmt("u vs eng", cu))
        # each variant's o carried through o_proj, the residual, the norm and the dense MLP in the contract's order
        Wo = ck.get("model.layers.0.self_attn.o_proj.weight").to(F32)
        wn = ck.get("model.layers.0.post_attention_layernorm.weight")
        g, u_, dw = (ck.get(f"model.layers.0.mlp.{m}_proj.weight") for m in ("gate", "up", "down"))
        dw = dw.to(F32)
        h0 = ref["h_in"]
        res["variants_h_out_vs_golden"] = {}
        print("\n== layer 0's output with each variant's attention (the rest the reference's) against the reference's h_out")
        for name, o in outs.items():
            h1 = (h0.to(F32) + torch.nn.functional.linear(o.to(F32), Wo)).to(BF16)
            x = glmref.rms_norm(h1, wn, 1e-5)
            y = torch.nn.functional.linear(glmref.silu_mul(glmref.linear(x, g), glmref.linear(x, u_)).to(F32), dw)
            h2 = (h1.to(F32) + y).to(BF16)
            cg = cmp(h2, ref["h_out"])
            res["variants_h_out_vs_golden"][name] = cg
            print(fmt(name[:10], cg), "|", name)
        # one rounding point at a time: pairs of variants that differ in one choice only
        pairs = {} if not outs else {
            "form (expanded K, V vs absorbed q_lat, u), f32 scores, online P":
                ("expanded, f32 scores, online P", "absorbed, f32 scores, online P (engine)"),
            "scores (bf16 vs f32), absorbed, softmax P":
                ("absorbed, bf16 scores, softmax P", "absorbed, f32 scores, softmax P"),
            "P (bf16(softmax) vs online bf16(exp(s - m)) / l), absorbed, f32 scores":
                ("absorbed, f32 scores, softmax P", "absorbed, f32 scores, online P (engine)"),
            "scores (bf16 vs f32), expanded, softmax P": ("expanded (reference)", "expanded, f32 scores"),
            "P, expanded, f32 scores": ("expanded, f32 scores", "expanded, f32 scores, online P"),
        }
        res["one_choice"] = {}
        print("\n== one choice at a time: o of the two variants against each other")
        for k, (x, y) in pairs.items():
            cc = cmp(outs[x], outs[y])
            res["one_choice"][k] = cc
            print(fmt("o", cc), "|", k)

        # every op of layer 0 on the engine's own inputs, by the reference's arithmetic (glmref.py's functions; the
        # attention by the --attn absorbed prototype): which op's arithmetic differs, apart from the f32 sum order
        L = "model.layers.0."
        Wd = lambda n: ck.get(L + n)
        F = torch.nn.functional
        e = eng
        iso = {}
        iso["qa"] = (glmref.linear(e["xn_in"], Wd("self_attn.q_a_proj.weight")), e["qa"])
        iso["ckv"] = (glmref.linear(e["xn_in"], Wd("self_attn.kv_a_proj_with_mqa.weight")), e["ckv"])
        iso["cq"] = (glmref.rms_norm(e["qa"], Wd("self_attn.q_a_layernorm.weight"), 1e-6), e["cq"])
        iso["c"] = (glmref.rms_norm(e["ckv"][:, :RKV], Wd("self_attn.kv_a_layernorm.weight"), 1e-6), e["c"])
        iso["k_rot"] = (glmref.rope(e["ckv"][:, RKV:], cos, sin), e["k_rot"])
        iso["q_raw"] = (glmref.linear(e["cq"], Wd("self_attn.q_b_proj.weight")).view(M, NH, DQK), e["q_raw"])
        Wkv = Wkvb.view(NH, DN + DV, RKV)
        iso["qlat"] = (torch.matmul(e["q_raw"][..., :DN].transpose(0, 1), Wkv[:, :DN]).transpose(0, 1), e["qlat"])
        qrope = glmref.rope(e["q_raw"][..., DN:], cos[:, None], sin[:, None]).transpose(0, 1)       # [64, S, 64]
        keys = glmref.absorbed_keys(0, M, e.get("topk"))
        Kc = torch.cat([e["c"], e["k_rot"]], -1)
        iso["u"] = (glmref.absorbed_attn(e["qlat"].transpose(0, 1), qrope, Kc, keys).transpose(0, 1), e["u"])
        iso["o"] = (torch.matmul(e["u"].transpose(0, 1), Wkv[:, DN:].transpose(1, 2)).transpose(0, 1).reshape(M, -1),
                    e["o"])
        iso["attn_out"] = (F.linear(e["o"].to(F32), Wo), e["attn_out"])
        iso["h_attn"] = ((e["h_in"].to(F32) + e["attn_out"]).to(BF16), e["h_attn"])
        iso["xn_mlp"] = (glmref.rms_norm(e["h_attn"], wn, 1e-5), e["xn_mlp"])
        iso["mlp_h"] = (glmref.silu_mul(glmref.linear(e["xn_mlp"], g), glmref.linear(e["xn_mlp"], u_)), e["mlp_h"])
        iso["mlp_out"] = (F.linear(e["mlp_h"].to(F32), dw), e["mlp_out"])
        iso["h_out"] = ((e["h_attn"].to(F32) + e["mlp_out"]).to(BF16), e["h_out"])
        res["isolated"] = {}
        print("\n== each op on the engine's own inputs, the reference's arithmetic, against the engine's output")
        for k, (x, y) in iso.items():
            cc = cmp(y, x)
            res["isolated"][k] = cc
            print(fmt(k, cc))
        # the SwiGLU's rounding points from the engine's own gate and up are not dumped; mlp_h above takes the
        # gate/up GEMMs with it. The f32 outputs (attn_out, mlp_out) cannot be bit-equal across sum orders.

        # a worked example: the element of o where the engine's and the reference's (expanded, from the engine's q
        # and cache) differ most against their spread, with the scores, P and the f32 sums behind both
        if not outs:
            if a.json:
                with open(a.json, "w") as f:
                    json.dump(res, f, indent=1)
            return
        oe, orf = e["o"].to(F32), outs["expanded (reference)"].to(F32)
        dif = (oe - orf).abs()
        i, j = divmod(int(dif.argmax()), oe.shape[1])
        h, dcol = divmod(j, DV)
        qh = torch.cat([e["q_raw"][i, h, :DN], glmref.rope(e["q_raw"][i:i + 1, h, DN:], cos[i:i + 1], sin[i:i + 1])[0]])
        kvx = torch.matmul(e["c"][:i + 1], Wkvb.T).view(i + 1, NH, DN + DV)[:, h]                  # bf16 K_nope, V
        Kh = torch.cat([kvx[:, :DN], e["k_rot"][:i + 1]], -1)
        s_ref = (torch.matmul(Kh, qh) * glmref.ATT_SCALE).to(F32)                                   # bf16 scores
        s_ref32 = torch.matmul(Kh.to(F32), qh.to(F32)) * glmref.ATT_SCALE
        P_ref = torch.softmax(s_ref, -1).to(BF16)
        o_ref32 = float((P_ref.to(F32) * kvx[:, DN + dcol].to(F32)).sum())
        ql = e["qlat"][i, h]
        qe = torch.cat([ql, qh[DN:]]).to(F32)
        s_eng = torch.matmul(Kc[:i + 1].to(F32), qe) * glmref.ATT_SCALE
        mx = float(s_eng.max())
        p_un = torch.exp(s_eng - s_eng[:64].max()) if i < 64 else None
        uh = e["u"][i, h].to(F32)
        o_eng32 = float((Wkv[h, DN + dcol].to(F32) * uh).sum())
        top = torch.argsort(P_ref.to(F32), descending=True)[:3]
        ex = {"row": i, "head": h, "col": dcol, "flat_col": j, "o_engine": float(oe[i, j]), "o_reference": float(orf[i, j]),
              "o_engine_f32_before_rounding": o_eng32, "o_reference_f32_before_rounding": o_ref32,
              "keys": [{"pos": int(k), "s_ref_bf16": float(s_ref[k]), "s_ref_f32_unrounded": float(s_ref32[k]),
                        "s_engine_f32": float(s_eng[k]), "P_ref_bf16": float(P_ref[k]),
                        "P_engine_eff": float(torch.exp(s_eng[k] - s_eng.max()) / torch.exp(s_eng - s_eng.max()).sum())}
                       for k in top]}
        res["example"] = ex
        print("\n== example:", json.dumps(ex, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
