#!/usr/bin/env python3
"""Empirical entropy of NVFP4 expert tensors (E2M1 nibbles and E4M3 group-16 scales) in a ModelOpt
checkpoint, reading only the sampled tensors by their header offsets.

    python -I nvfp4_entropy.py CKPT [--layers 3,10,30,50,77] [--experts 0,85,170,255]

Per projection (gate/up/down), over the sample: order-0 entropy of the nibbles, of the nibbles given
their group's scale byte, of the scale bytes, of a scale given the previous scale in its row, and the
bits a fixed-length code needs for a scale as an offset from its row's (or its 8-row by row tile's)
smallest scale. Needs numpy only.
"""
import argparse, json, os, struct
from collections import defaultdict
import numpy as np


def H(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def Hcond(joint):
    """H(Y|X) from a joint count table [x, y]."""
    tot = joint.sum()
    h = 0.0
    for row in joint:
        s = row.sum()
        if s:
            h += s / tot * H(row)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--layers", default="3,10,30,50,77")
    ap.add_argument("--experts", default="0,85,170,255")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    hdr = {}

    def read(name):
        f = os.path.join(a.ckpt, idx[name])
        if f not in hdr:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr[f] = (n, json.loads(fh.read(n)))
        n, h = hdr[f]
        v = h[name]
        s, e = v["data_offsets"]
        with open(f, "rb") as fh:
            fh.seek(8 + n + s)
            return np.frombuffer(fh.read(e - s), dtype=np.uint8).reshape(v["shape"])

    acc = defaultdict(lambda: {"nib": np.zeros(16, np.int64), "nib_sc": np.zeros((256, 16), np.int64),
                               "sc": np.zeros(256, np.int64), "sc_prev": np.zeros((256, 256), np.int64),
                               "row_bits": [], "tile_bits": [], "n_sc": 0})
    for L in [int(x) for x in a.layers.split(",")]:
        for E in [int(x) for x in a.experts.split(",")]:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"model.layers.{L}.mlp.experts.{E}.{proj}.weight"
                w = read(base)                      # [R, C/2] packed
                sc = read(base + "_scale")          # [R, C/16] e4m3 bytes
                R = w.shape[0]
                nib = np.stack([w & 15, w >> 4], -1).reshape(R, -1)        # [R, C]
                d = acc[proj]
                d["nib"] += np.bincount(nib.ravel(), minlength=16)
                grp = np.repeat(sc, 16, axis=1)                            # each nibble's scale byte
                d["nib_sc"] += np.bincount(grp.ravel().astype(np.int64) * 16 + nib.ravel(),
                                           minlength=256 * 16).reshape(256, 16)
                d["sc"] += np.bincount(sc.ravel(), minlength=256)
                d["sc_prev"] += np.bincount(sc[:, :-1].ravel().astype(np.int64) * 256 + sc[:, 1:].ravel(),
                                            minlength=65536).reshape(256, 256)
                rng = sc.max(1).astype(int) - sc.min(1).astype(int)
                d["row_bits"].append(np.ceil(np.log2(rng + 1)))
                t = sc[: R - R % 8].reshape(R // 8, 8, -1, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
                trng = t.max(1).astype(int) - t.min(1).astype(int)
                d["tile_bits"].append(np.ceil(np.log2(trng + 1)))
                d["n_sc"] += sc.size
    print(f"layers {a.layers}, experts {a.experts}")
    print(f"{'proj':10s} {'H(nib)':>7s} {'H(nib|sc)':>9s} {'H(sc)':>6s} {'H(sc|prev)':>10s} "
          f"{'fixed bits: row':>15s} {'8x8 tile':>8s} {'distinct sc':>11s} {'zero nib %':>10s}")
    for proj, d in acc.items():
        rb = np.concatenate(d["row_bits"]).mean()
        tb = np.concatenate(d["tile_bits"]).mean()
        print(f"{proj:10s} {H(d['nib']):7.4f} {Hcond(d['nib_sc']):9.4f} {H(d['sc']):6.3f} "
              f"{Hcond(d['sc_prev']):10.3f} {rb:15.2f} {tb:8.2f} {(d['sc'] > 0).sum():11d} "
              f"{100 * (d['nib'][0] + d['nib'][8]) / d['nib'].sum():10.2f}")
    for proj, d in acc.items():
        tb = np.concatenate(d["tile_bits"])
        rb = np.concatenate(d["row_bits"])
        print(f"{proj}: 8x8 tiles needing <=3/4/5/6 bits: " + " / ".join(
            f"{100 * (tb <= k).mean():.2f} %" for k in (3, 4, 5, 6)) + "; rows needing <=4/5/6 bits: " +
            " / ".join(f"{100 * (rb <= k).mean():.2f} %" for k in (4, 5, 6)))
    allnib = sum(d["nib"] for d in acc.values())
    allsc = sum(d["sc"] for d in acc.values())
    print("all: H(nib) %.4f, H(nib|sc) %.4f, H(sc) %.3f, H(sc|prev) %.3f" % (
        H(allnib), Hcond(sum(d["nib_sc"] for d in acc.values())), H(allsc),
        Hcond(sum(d["sc_prev"] for d in acc.values()))))
    print("nibble histogram (codes 0-15):", (allnib / allnib.sum()).round(4).tolist())
    top = np.argsort(-allsc)[:12]
    print("most common scale bytes:", [(int(b), round(float(allsc[b] / allsc.sum()), 4)) for b in top])


if __name__ == "__main__":
    main()
