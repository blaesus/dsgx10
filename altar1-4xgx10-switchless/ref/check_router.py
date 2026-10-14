"""DESIGN decision 10: altar-1's routers are the parent's at the kept experts.

  python3 -I check_router.py extract --parent PARENT_CKPT --layers 3 4 40 77 78 --out parent_routers.npz
  python3 -I check_router.py compare --ckpt CKPT --parent-npz parent_routers.npz [--verified LIST]

extract: the parent GLM-5.3's mlp.gate.weight [256, 6144] and e_score_correction_bias [256] of the layers, from
any checkpoint of it that stores them unquantised (NVIDIA's GLM-5.3-NVFP4 keeps the routers in BF16 / F32).
compare: for each layer, altar-1's 168 rows and bias entries against the parent's at the ids of
reap_prune_plan.json's layers[L] (altar-1 expert j = parent expert plan[L][j]): the bias bit for bit, and the
router weight bit for bit; where the weight differs (layers 4-76, where AWQ's smoothing factors are folded into
the router's input columns, altar-1 model.md § 5), the per-column ratio altar/parent: its spread over the 168 rows
(a pure column scaling gives one ratio a column up to bf16 rounding) and how often bf16(parent * the column's
median ratio) reproduces altar-1's value.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmref as G  # noqa: E402


def extract(a):
    ck = G.Ckpt(a.parent)
    out = {}
    for L in a.layers:
        w = ck.get(f"model.layers.{L}.mlp.gate.weight")
        b = ck.get(f"model.layers.{L}.mlp.gate.e_score_correction_bias")
        print(f"layer {L}: weight {w.dtype} {list(w.shape)}, bias {b.dtype} {list(b.shape)}")
        out[f"w{L}"] = w.view(torch.int16).numpy()
        out[f"wdtype{L}"] = np.array(str(w.dtype))
        out[f"b{L}"] = b.float().numpy()
        out[f"bdtype{L}"] = np.array(str(b.dtype))
    np.savez(a.out, **out)


def compare(a):
    ck = G.Ckpt(a.ckpt, verified=a.verified)
    plan = json.load(open(os.path.join(a.ckpt, "reap_prune_plan.json")))
    P = np.load(a.parent_npz)
    res = {}
    for key in P.files:
        if not key.startswith("w") or key.startswith("wdtype"):
            continue
        L = int(key[1:])
        ids = np.array(plan["layers"][str(L)], dtype=np.int64)
        assert len(ids) == G.NE and np.all(np.diff(ids) > 0), "plan: 168 sorted ids"
        assert str(P[f"wdtype{L}"]) == "torch.bfloat16"
        pw = torch.from_numpy(P[key]).view(torch.bfloat16)[ids]
        pb = torch.from_numpy(P[f"b{L}"])[ids]
        aw = ck.get(f"model.layers.{L}.mlp.gate.weight")
        ab = ck.get(f"model.layers.{L}.mlp.gate.e_score_correction_bias")
        r = {"parent_bias_dtype": str(P[f"bdtype{L}"]), "bias_dtype": str(ab.dtype),
             "bias_equal": int((ab.float() == pb).sum()), "weight_equal": int((aw.view(torch.int16) == pw.view(torch.int16)).sum()),
             "weight_values": aw.numel(), "bias_max_abs": float((ab.float() - pb).abs().max())}
        if r["weight_equal"] != aw.numel():
            ratio = aw.float() / pw.float()
            ok = pw.float() != 0
            med = torch.nanmedian(torch.where(ok, ratio, torch.nan), dim=0).values           # per column [6144]
            dev = ((ratio - med) / med).abs()
            r["col_ratio_range"] = [float(med.min()), float(med.max())]
            r["ratio_rel_dev_median"] = float(torch.nanmedian(torch.where(ok, dev, torch.nan)))
            r["ratio_rel_dev_p99"] = float(np.nanpercentile(torch.where(ok, dev, torch.nan).numpy(), 99))
            r["bf16_parent_x_colfactor_equal"] = float(((pw.float() * med).to(torch.bfloat16) == aw).float().mean())
            # the kept rows against the removed ones: a router row of altar-1 matched to the wrong parent row
            # would show as a ratio spread of order 1
        res[L] = r
        print(f"layer {L}: bias {r['bias_equal']}/168 equal (parent {r['parent_bias_dtype']}, max |diff| "
              f"{r['bias_max_abs']:.3g}); weight bit-equal {r['weight_equal']}/{r['weight_values']}"
              + ("" if "col_ratio_range" not in r else
                 f"; column factor {r['col_ratio_range'][0]:.3f}..{r['col_ratio_range'][1]:.3f}, |ratio/column median - 1| "
                 f"median {r['ratio_rel_dev_median']:.2e} p99 {r['ratio_rel_dev_p99']:.2e}, "
                 f"bf16(parent x column factor) == altar-1 on {r['bf16_parent_x_colfactor_equal']:.4f}"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--parent", required=True)
    e.add_argument("--layers", type=int, nargs="+", default=[3, 4, 40, 77, 78])
    e.add_argument("--out", required=True)
    c = sp.add_parser("compare")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--parent-npz", required=True)
    c.add_argument("--verified", default=None)
    c.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    extract(a) if a.cmd == "extract" else compare(a)


if __name__ == "__main__":
    main()
