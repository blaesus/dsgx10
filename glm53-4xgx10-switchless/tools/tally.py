#!/usr/bin/env python3
"""Tally a GLM-5.3 safetensors checkpoint by tensor class, from the shard headers only.

Reads the 8-byte header length and the JSON header of every *.safetensors shard in CKPT (no tensor
data). Prints, per class: tensor count, parameters, the stored dtypes and bytes, and the bytes per
rank at TP=4 for each precision option (classes marked "rep" are replicated on every rank).

    python3 -I tally.py CKPT [--json OUT]

Run with -I (isolated mode): the checkpoint directory is untrusted data.
"""
import argparse, glob, json, os, re, struct, sys
from collections import defaultdict

GIB = 2 ** 30
DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1}

# bytes per parameter of each precision option, scales included
OPTS = {
    "bf16": 2.0,
    "fp8 b128": 1 + 4 / (128 * 128),        # e4m3, one f32 scale per 128x128 block (the official format)
    "nvfp4": 0.5 + 1 / 16,                  # e2m1, one e4m3 scale per 16 (+ f32 per tensor, ignored)
    "mxfp4": 0.5 + 1 / 32,                  # e2m1, one ue8m0 per 32
    "int3 g64 f16": 3 / 8 + 2 / 64,         # 3.25 bpw
    "trellis 3.0": 3 / 8,                   # EXL3/QTIP-style at 3.0 bpw, scales negligible (per-channel)
}


def classify(name):
    """Return (class, replicated?) for a weight name; scales map to the class of their weight."""
    m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if not m:
        if name.startswith("lm_head"):
            return "head (lm_head)", False
        if name.startswith("model.embed_tokens"):
            return "embed", True
        return "norms", True
    layer, rest = int(m.group(1)), m.group(2)
    mtp = layer == 78
    pre = "mtp " if mtp else ""
    if "layernorm" in rest or rest.endswith("norm.weight") and "indexer" not in rest:
        return "norms", True
    if rest.startswith("mlp.experts."):
        part = rest.split(".", 3)[3]          # e.g. gate_proj.weight_scale
        if mtp:
            return "mtp experts", False
        return "routed experts (layers 3-77)", False
    if rest.startswith("mlp.shared_experts"):
        return pre + "shared expert", False
    if rest.startswith("mlp.gate.") or rest == "mlp.gate.weight":
        return pre + "router", True
    if rest.startswith("mlp."):
        return "dense MLP (layers 0-2)", False
    if rest.startswith("self_attn.indexer"):
        return pre + "indexer", True
    if rest.startswith("self_attn."):
        proj = rest.split(".")[1]
        if mtp:
            return "mtp attention", False
        return "attn " + proj, False
    if rest.startswith(("eh_proj", "enorm", "hnorm", "shared_head")):
        return "mtp eh_proj/norms", rest.startswith(("enorm", "hnorm", "shared_head"))
    return "other: " + rest, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--json")
    a = ap.parse_args()
    cls = defaultdict(lambda: {"n": 0, "params": 0, "bytes": 0, "dtypes": defaultdict(int), "rep": False})
    total = 0
    for f in sorted(glob.glob(os.path.join(a.ckpt, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            c, rep = classify(k)
            e = cls[c]
            nel = 1
            for s in v["shape"]:
                nel *= s
            b = v["data_offsets"][1] - v["data_offsets"][0]
            assert b == nel * DT[v["dtype"]], k
            e["n"] += 1
            e["bytes"] += b
            e["dtypes"][v["dtype"]] += b
            e["rep"] = rep
            # parameters: the logical weight count (packed e2m1 holds two per byte); scales count as 0
            if k.endswith((".weight", ".bias")):
                e["params"] += nel * (2 if v["dtype"] == "U8" else 1)
            total += b
    print(f"total stored {total / GIB:.2f} GiB")
    hdr = f"{'class':30s} {'tensors':>8s} {'params (M)':>11s} {'stored GiB':>10s} {'dtypes':24s} {'stored/rk':>9s} " + \
          " ".join(f"{o:>12s}" for o in OPTS)
    print(hdr)
    print(" " * 97 + "  per rank at TP=4, GiB (rep = whole on every rank)")
    out = {}
    tot = defaultdict(float)
    for c in sorted(cls):
        e = cls[c]
        div = 1 if e["rep"] else 4
        per = {o: e["params"] * bpp / div / GIB for o, bpp in OPTS.items()}
        dts = ",".join(f"{d}:{b / GIB:.2f}" for d, b in sorted(e["dtypes"].items()))
        print(f"{c + (' (rep)' if e['rep'] else ''):30s} {e['n']:8d} {e['params'] / 1e6:11.1f} "
              f"{e['bytes'] / GIB:10.3f} {dts:24s} {e['bytes'] / div / GIB:9.3f} " + " ".join(f"{per[o]:12.3f}" for o in OPTS))
        for o in OPTS:
            tot[o] += per[o]
        tot["stored"] += e["bytes"] / div / GIB
        out[c] = {"tensors": e["n"], "params": e["params"], "stored_bytes": e["bytes"],
                  "dtypes": dict(e["dtypes"]), "replicated": e["rep"],
                  "per_rank_gib": per, "stored_per_rank_gib": e["bytes"] / div / GIB}
    print(f"{'all, per rank':30s} {'':8s} {'':11s} {'':10s} {'':24s} {tot['stored']:9.3f} "
          + " ".join(f"{tot[o]:12.3f}" for o in OPTS))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
