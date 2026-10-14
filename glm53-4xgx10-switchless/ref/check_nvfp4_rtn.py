"""Check that quant/formats.py's nvfp4_rtn, applied to the BF16 master's experts, reproduces NVIDIA's NVFP4
checkpoint byte for byte (packed codes, E4M3 block scales, weight_scale_2), gate and up sharing one amax.

  python3 check_nvfp4_rtn.py --ckpt CKPT --master MASTER_DIR --layers 3 40 [--experts 0-255]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmref as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 40])
    ap.add_argument("--experts", default="0-255")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fm = G.quant_formats()
    ck = G.Ckpt(a.ckpt)
    lo, hi = (int(x) for x in a.experts.split("-"))
    for l in a.layers:
        m = G.MasterExperts(os.path.expanduser(a.master), l)
        tot = {"weight": [0, 0], "weight_scale": [0, 0], "weight_scale_2": [0, 0]}
        bad = []
        for e in range(lo, hi + 1):
            g, u, d = [t.to(torch.float32) for t in m.load(e)]
            amax = torch.maximum(g.abs().max(), u.abs().max())
            for name, w, am in (("gate", g, amax), ("up", u, amax), ("down", d, None)):
                pk = fm.nvfp4_rtn(w, am)
                p = f"model.layers.{l}.mlp.experts.{e}.{name}_proj"
                ref = {"weight": ck.get(p + ".weight").numpy(),
                       "weight_scale": ck.get(p + ".weight_scale").view(torch.uint8).numpy(),
                       "weight_scale_2": ck.get(p + ".weight_scale_2").numpy()}
                for k in tot:
                    eq = (pk[k].reshape(ref[k].shape) == ref[k])
                    tot[k][0] += int(eq.sum())
                    tot[k][1] += eq.size
                    if not eq.all() and len(bad) < 5:
                        bad.append((e, name, k, int((~eq).sum())))
        print(f"layer {l} experts {lo}-{hi}: " + ", ".join(f"{k} {v[0]}/{v[1]} equal" for k, v in tot.items()))
        for b in bad:
            print("  differs:", b)


if __name__ == "__main__":
    main()
