"""Reference data for the mHC (hyper-connection) kernels.

model.py's Block.hc_mixes / hc_pre / hc_post (torchref's hc_split_sinkhorn underneath) on a layer's
real hc_attn_* / hc_ffn_* weights, a seeded random residual stream of 4 copies with per-copy scales
and a few outlier channels, and a random sublayer output.

Inside the canary-roce image (see gen.sh):
  python3 gen_mhc.py --layers 2,30 --T 64 --out /out/mhc
Names: L<layer>_<attn|ffn>_<what>; manifest.json as gen_attn.py.
"""
import argparse, json, os, sys, types
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ref", "torchref"))
sys.path.insert(1, os.path.join(ROOT, "ref", "inference"))
import kernel as K  # noqa: E402,F401  (torchref stand-in, what model.py imports)
import model as M  # noqa: E402
from ref_run import CkptIndex  # noqa: E402
from gen_attn import Rec  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def make_stream(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = torch.tensor([0.5, 1.0, 2.0, 4.0], device="cuda").view(1, 4, 1)
    h = torch.randn(T, 4, 5120, generator=g, device="cuda") * scale
    out = torch.randint(0, 5120, (8,), generator=g, device="cuda")
    h[:, :, out] *= 20
    return h.to(torch.bfloat16).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/models")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(ROOT, "ref", "inference", "config.json")))
    args = M.ModelArgs(**cfg)
    blk = types.SimpleNamespace(norm_eps=args.norm_eps, hc_mult=args.hc_mult,
                                hc_sinkhorn_iters=args.hc_sinkhorn_iters, hc_eps=args.hc_eps)
    ck = CkptIndex(a.ckpt)
    rec = Rec(a.out)
    layers = [int(v) for v in a.layers.split(",")]
    for L in layers:
        for which in ("attn", "ffn"):
            p = f"L{L}_{which}_"
            fn = ck.get(f"layers.{L}.hc_{which}_fn").float()
            scale = ck.get(f"layers.{L}.hc_{which}_scale").float()
            base = ck.get(f"layers.{L}.hc_{which}_base").float()
            x = make_stream(a.T, 100 * L + (which == "ffn"))
            g = torch.Generator(device="cuda").manual_seed(7 + L)
            sub = (torch.randn(1, a.T, 5120, generator=g, device="cuda") * 0.7).to(torch.bfloat16)
            pre, post, comb = M.Block.hc_mixes(blk, x, fn, scale, base)
            # the same steps unrolled, for the intermediates (asserted identical below)
            xf = x.flatten(2).float()
            rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + args.norm_eps)
            raw = F.linear(xf, fn)
            mixes = raw * rs
            pre2, post2, comb2 = K.hc_split_sinkhorn(mixes, scale, base, args.hc_mult, args.hc_sinkhorn_iters,
                                                     args.hc_eps)
            assert torch.equal(pre, pre2) and torch.equal(post, post2) and torch.equal(comb, comb2)
            y_pre = M.Block.hc_pre(blk, x, pre)
            y_post = M.Block.hc_post(blk, sub, x, post, comb)
            for name, t in [("fn", fn), ("scale", scale), ("base", base), ("x", x[0]), ("sub", sub[0]),
                            ("rsqrt", rs[0, :, 0]), ("raw", raw[0]), ("mixes", mixes[0]), ("pre", pre[0]),
                            ("post", post[0]), ("comb", comb[0]), ("y_pre", y_pre[0]), ("y_post", y_post[0])]:
                rec(p + name, t)
        print("layer", L, "done", flush=True)
    rec.save({"layers": layers, "T": a.T})
    print("wrote", len(rec.manifest), "tensors to", a.out, flush=True)


if __name__ == "__main__":
    main()
