"""One real altar-1 decoder layer: glmref.py against transformers' GlmMoeDsa modules with compressed-tensors'
decompression, the independent stack (A1a).

  python3 -I check_layer.py --ckpt CKPT --run RUN --layer L [--seqs prose2400 short] [--group 24] [--threads 8]

RUN is a glmref.py run in transformers' order (--combine bf16) that saved every position (passes.sh's
a1a-bf16): its hidden state after layer L-1 (the embedding for L = 0) is the input, its hidden state after layer
L, index sets and routes are glmref's answer. The other side:
- transformers (the venv's version) GlmMoeDsaAttention, GlmMoeDsaRMSNorm, GlmMoeDsaTopkRouter, GlmMoeDsaMLP
  and GlmMoeDsaExperts, built from the checkpoint's config (eager attention, grouped_mm experts) and loaded
  with load_state_dict(strict=True) from the checkpoint's tensors. No released transformers loads altar-1 with
  from_pretrained correctly (altar-1 model.md § 4.4: the experts' zero points are dropped), so the modules are
  built and loaded here.
- Every INT4 tensor dequantised by compressed_tensors' PackedQuantizationCompressor.decompress with the
  checkpoint's scheme (not by glmref's code).
- The layer composed as GlmMoeDsaDecoderLayer.forward and GlmMoeDsaMoE.forward. The routed experts run through
  transformers' grouped_mm experts forward in groups of --group experts (a whole layer is 12.7 GB in bf16);
  pairs routed outside the group go to its sentinel id, which the forward zeroes; each group's f32 sum over the
  8 slots is kept f32 (the hidden input given as f32 holding the bf16 values, so the final .to() does not
  round), the groups are summed in f32 and rounded once to bf16. That is transformers' arithmetic except the
  association of the f32 sum over the 8 slots.
Shared layers take the previous full layer's sets from RUN (positions < 2048 see every causal key, as
transformers' own indexer gives them).

Prints bit-equality, max |diff| and rel-l2 of the attention sublayer output (against glmref.attention on the same
input, in the bf16 order), the index sets, the routes and the layer output.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmref as G  # noqa: E402


def ulps(a, b):
    """Distance in bf16 units in the last place (a, b bf16): the gap between their positions on the ordered bf16 line."""
    def key(x):
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)       # monotone in the value (sign-magnitude to two's complement)
    return (key(a) - key(b)).abs()


def cmp(tag, a, b):
    u = ulps(a.to(torch.bfloat16), b.to(torch.bfloat16))
    a, b = a.float(), b.float()
    eq = (a == b).float().mean().item()
    d = (a - b).abs()
    rel = (d.norm() / b.norm()).item()
    r = {"bit_equal": eq, "max_abs": d.max().item(), "rel_l2": rel, "max_ulps": int(u.max()),
         "share_1ulp": (u == 1).float().mean().item(), "share_gt1ulp": (u > 1).float().mean().item()}
    print(f"  {tag:30s} bit-equal {eq:7.4f}  max |diff| {d.max().item():.3e}  rel-l2 {rel:.3e}  ulps: max "
          f"{r['max_ulps']}, =1 {r['share_1ulp']:.4f}, >1 {r['share_gt1ulp']:.5f}", flush=True)
    return r


def bf16_rows(path, rows):
    return torch.from_numpy(np.ascontiguousarray(np.load(path, mmap_mode="r")[rows])).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--verified", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seqs", nargs="+", default=["prose2400", "short"])
    ap.add_argument("--group", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    import transformers
    from transformers import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    L = a.layer
    with open(os.path.join(a.ckpt, "config.json")) as f:
        raw = json.load(f)
    grp = raw["quantization_config"]["config_groups"]["group_0"]
    scheme = QuantizationScheme(targets=grp["targets"], weights=QuantizationArgs(**grp["weights"]))
    raw.pop("quantization_config")
    cfg = GlmMoeDsaConfig(**raw)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "grouped_mm"
    print(f"transformers {transformers.__version__}, torch {torch.__version__}; layer {L} "
          f"({cfg.mlp_layer_types[L]}, indexer {raw['indexer_types'][L]})")
    ck = G.Ckpt(a.ckpt, verified=a.verified)
    pre = f"model.layers.{L}."
    names = [n for n in ck.map if n.startswith(pre)]

    def load(module, sub, skip=()):
        """module's state dict from the checkpoint under pre + sub; INT4 modules via compressed-tensors."""
        sd = {}
        for n in names:
            if not n.startswith(pre + sub) or any(n.startswith(pre + s) for s in skip):
                continue
            k = n[len(pre + sub):]
            if k.endswith(".weight_packed"):
                m = n[:-len(".weight_packed")]
                packed, scale, zp, shape = ck.int4_raw(m)
                w = PackedQuantizationCompressor.decompress(
                    {"weight_packed": packed, "weight_scale": scale, "weight_zero_point": zp, "weight_shape": shape},
                    scheme)["weight"]
                sd[k[:-len("_packed")]] = w
                fmt[m] = "int4"
            elif k.endswith((".weight_scale", ".weight_zero_point", ".weight_shape")):
                continue
            else:
                sd[k] = ck.get(n)
                fmt[n] = str(sd[k].dtype)
        module.load_state_dict(sd, strict=True)
        return module

    fmt = {}
    t0 = time.time()
    norm_in = load(M.GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(torch.bfloat16), "input_layernorm.")
    norm_post = load(M.GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(torch.bfloat16), "post_attention_layernorm.")
    attn = M.GlmMoeDsaAttention(cfg, L).to(torch.bfloat16)
    load(attn, "self_attn.")
    if attn.indexer is not None:
        attn.indexer.weights_proj.float()          # _keep_in_fp32_modules
    sparse = cfg.mlp_layer_types[L] == "sparse"
    if sparse:
        router = M.GlmMoeDsaTopkRouter(cfg).to(torch.bfloat16)
        router.e_score_correction_bias = router.e_score_correction_bias.float()
        load(router, "mlp.gate.")
        shared = load(M.GlmMoeDsaMLP(cfg, cfg.moe_intermediate_size * cfg.n_shared_experts).to(torch.bfloat16),
                      "mlp.shared_experts.")
    else:
        mlp = load(M.GlmMoeDsaMLP(cfg).to(torch.bfloat16), "mlp.")
    rot = M.GlmMoeDsaRotaryEmbedding(cfg)
    print(f"  non-expert modules loaded in {time.time() - t0:.0f} s; formats: "
          f"{sorted(set((k.split(pre)[-1].rsplit('.', 1)[0] if 'experts' not in k else 'experts', v) for k, v in fmt.items()))}")

    def experts_out(x, idx, w):
        """Routed experts' f32 sum (see the docstring), x bf16 [S, D]."""
        E, Gs = cfg.n_routed_experts, a.group
        acc = torch.zeros(x.shape[0], cfg.hidden_size, dtype=torch.float32)
        for e0 in range(0, E, Gs):
            e1 = min(E, e0 + Gs)
            gcfg = GlmMoeDsaConfig(**{**raw, "n_routed_experts": e1 - e0})
            gcfg._experts_implementation = "grouped_mm"
            ex = M.GlmMoeDsaExperts(gcfg)
            gu = torch.empty(e1 - e0, 2 * cfg.moe_intermediate_size, cfg.hidden_size, dtype=torch.bfloat16)
            dn = torch.empty(e1 - e0, cfg.hidden_size, cfg.moe_intermediate_size, dtype=torch.bfloat16)
            for e in range(e0, e1):
                ws = {}
                for m in ("gate", "up", "down"):
                    p = f"{pre}mlp.experts.{e}.{m}_proj"
                    if ck.is_int4(p):
                        packed, scale, zp, shape = ck.int4_raw(p)
                        ws[m] = PackedQuantizationCompressor.decompress(
                            {"weight_packed": packed, "weight_scale": scale, "weight_zero_point": zp,
                             "weight_shape": shape}, scheme)["weight"]
                        fmt["experts"] = "int4"
                    else:
                        ws[m] = ck.get(p + ".weight")
                        fmt["experts"] = str(ws[m].dtype)
                gu[e - e0, :cfg.moe_intermediate_size] = ws["gate"]
                gu[e - e0, cfg.moe_intermediate_size:] = ws["up"]
                dn[e - e0] = ws["down"]
            ex.load_state_dict({"gate_up_proj": gu, "down_proj": dn}, strict=True)
            del gu, dn
            gi = torch.where((idx >= e0) & (idx < e1), idx - e0, torch.full_like(idx, e1 - e0))  # sentinel
            acc += ex(x.float(), gi, w)
            del ex
        return acc

    # glmref's attention on the same input, transformers' rounding order
    G.COMBINE = "bf16"
    W = G.LayerW(ck, L)
    gargs = argparse.Namespace(attn_block=256, head_group=16, index_block=128)
    man = json.load(open(os.path.join(a.run, "manifest.json")))
    assert man["combine"] == "bf16", "the run must be in transformers' order (--combine bf16)"
    sr = np.load(os.path.join(a.run, "sample_rows.npy"))
    pos_of = {int(r): j for j, r in enumerate(sr)}
    full = [t == "full" for t in raw["indexer_types"]]
    res = {"layer": L, "transformers": transformers.__version__, "seqs": {}}
    for sname in a.seqs:
        s = [q for q in man["sequences"] if q["name"] == sname][0]
        rows = [pos_of[r] for r in range(s["off"], s["off"] + s["n"])]       # every position must be saved
        S = s["n"]
        src = os.path.join(a.run, "embed_sample.npy") if L == 0 else os.path.join(a.run, "layers", f"L{L - 1:02d}", "hidden.npy")
        h = bf16_rows(src, rows)
        mine_out = bf16_rows(os.path.join(a.run, "layers", f"L{L:02d}", "hidden.npy"), rows)
        print(f"{sname}: {S} rows")
        t0 = time.time()
        pos = torch.arange(S)[None]
        cos, sin = rot(h[None], pos)
        prev = None
        if not full[L]:
            lf = max(i for i in range(L + 1) if full[i])
            T = min(G.TOPK, S)
            prev = torch.arange(T, dtype=torch.int32)[None, None].expand(1, S, T).clone()
            if S > G.TOPK:
                prev[0, G.TOPK:] = torch.from_numpy(np.load(os.path.join(a.run, "layers", f"L{lf:02d}", f"topk_{sname}.npy")).astype(np.int32))
        x = norm_in(h[None])
        a_ref, _, tk_ref = attn(x, (cos, sin), None, position_ids=pos, prev_topk_indices=prev)
        gcos, gsin = G.rope_tables(S)
        a_mine, tk, mg = G.attention(G.rms_norm(h, W("input_layernorm.weight"), 1e-5), W, full[L], None if full[L] else
                                     (prev[0, G.TOPK:] if S > G.TOPK else None), gcos, gsin, gargs)
        r = {"rows": S, "attention_s": time.time() - t0}
        if full[L] and S > G.TOPK:
            sets_ref = tk_ref[0, G.TOPK:].long().sort(dim=1).values
            same = (sets_ref == tk.long()).all(dim=1)
            r["index_sets_equal"] = f"{int(same.sum())}/{len(same)}"
            r["index_boundary_ties"] = int((mg == 0).sum())
            print(f"  index sets equal on {r['index_sets_equal']} sparse rows; boundary ties {r['index_boundary_ties']}")
        r["attention"] = cmp("attention output", a_mine, a_ref[0])
        h1 = h[None] + a_ref
        x2 = norm_post(h1)
        if sparse:
            _, w_ref, i_ref = router(x2)
            ri = torch.from_numpy(np.load(os.path.join(a.run, "layers", f"L{L:02d}", "route_idx.npy"))[s["off"]:s["off"] + S]).long()
            rw = torch.from_numpy(np.load(os.path.join(a.run, "layers", f"L{L:02d}", "route_w.npy"))[s["off"]:s["off"] + S])
            i_s, o1 = i_ref.sort(dim=1)
            m_s, o2 = ri.sort(dim=1)
            r["route_sets_equal"] = (i_s == m_s).all(1).float().mean().item()
            r["route_w_max_abs"] = (w_ref.gather(1, o1) - rw.gather(1, o2)).abs().max().item()
            print(f"  routing (transformers' router on its own post-attention state vs glmref's): sets equal on "
                  f"{r['route_sets_equal']:.4f} of rows; weights max |diff| {r['route_w_max_abs']:.3e}")
            t1 = time.time()
            m = experts_out(x2[0], i_ref, w_ref).to(torch.bfloat16) + shared(x2[0])
            r["experts_s"] = time.time() - t1
        else:
            m = mlp(x2[0])
        out = h1[0] + m
        r["layer"] = cmp("layer output", mine_out, out)
        res["seqs"][sname] = r
    res["formats"] = sorted(set((k.split(pre)[-1].rsplit(".", 1)[0] if "experts" not in k else "experts", v)
                                for k, v in fmt.items()))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
