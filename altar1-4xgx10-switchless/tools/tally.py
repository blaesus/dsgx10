#!/usr/bin/env python3
"""Tally AikidoSec/altar-1 by tensor class from its safetensors headers: bytes a rank at TP=4 and decode bytes a token.

Reads only the 8-byte header length and the JSON header of every shard in DIR: either the checkpoint's
`*.safetensors` files or header prefixes fetched by HTTP range request and saved as `*.safetensors.hdr`
(the first 8 + n bytes of each shard). No tensor data is read.

    python3 -I tally.py DIR [--json OUT] [--gbps 235]

Adapted from glm53-4xgx10-switchless/tools/tally.py. The split per class follows the glm53 engine's rank files
(glm53 docs/layout.md § 3): everything is cut four ways except the norms, the routers (weight and bias) and the
indexer's k_norm, which are whole on every rank; the embedding and head are split by vocabulary. The decode read
a token is one row at one rank: every weight of layers 0-77 once, the routed experts at 8 of 168 a layer, the head
whole, the embedding one row (counted as 0), the MTP layer (78) not at all (it runs only in a draft pass, given
separately). Run with -I: the headers are untrusted data.
"""
import argparse, glob, json, os, re, struct, sys
from collections import defaultdict

GIB = 2 ** 30
DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1, "I32": 4, "I64": 8}
N_EXP, TOPK, TP = 168, 8, 4


def classify(name):
    """(class, part, replicated) for a tensor name. part: packed / scale / zp / shape / bf16 / f32."""
    part = "bf16"
    for suf, p in ((".weight_packed", "packed"), (".weight_scale", "scale"), (".weight_zero_point", "zp"),
                   (".weight_shape", "shape"), (".e_score_correction_bias", "f32")):
        if name.endswith(suf):
            part = p
    m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if not m:
        if name.startswith("lm_head"):
            return "head", part, False
        if name.startswith("model.embed_tokens"):
            return "embed", part, False                 # vocab-split as glm53's rank files
        return "norms", part, True
    L, rest = int(m.group(1)), m.group(2)
    mtp = L == 78
    pre = "mtp " if mtp else ""
    if rest.endswith("layernorm.weight") or rest.startswith(("enorm", "hnorm", "shared_head.norm")):
        return pre + "norms", part, True
    if rest.startswith("mlp.experts."):
        if mtp:
            return "mtp routed experts BF16", part, False
        if part == "bf16":
            return "routed experts BF16 (layers 3, 77)", part, False
        return "routed experts INT4 (layers 4-76)", part, False
    if rest.startswith("mlp.shared_experts"):
        return pre + "shared expert", part, False
    if rest.startswith("mlp.gate."):
        return pre + "router + bias", part, True
    if rest.startswith("mlp."):
        return "dense MLP (layers 0-2)", part, False
    if rest.startswith("self_attn.indexer"):
        return pre + "indexer", part, "k_norm" in rest
    if rest.startswith("self_attn."):
        proj = rest.split(".")[1]
        if mtp:
            return "mtp attention BF16", part, False
        if part != "bf16":
            return f"attn {proj} INT4", part, False
        if proj in ("q_a_proj", "kv_a_proj_with_mqa"):
            return f"attn {proj} BF16", part, False
        return "attn q_b/kv_b/o_proj BF16 (layers 0-3, 77)", part, False
    if rest.startswith("eh_proj"):
        return "mtp eh_proj", part, False
    return "other: " + rest, part, True


def read_headers(d):
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")) + glob.glob(os.path.join(d, "*.safetensors.hdr")))
    if not files:
        sys.exit(f"no *.safetensors or *.safetensors.hdr in {d}")
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        yield os.path.basename(f), h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json")
    ap.add_argument("--gbps", type=float, default=235.0)
    a = ap.parse_args()
    cls = defaultdict(lambda: {"n": 0, "params": 0, "bytes": 0, "parts": defaultdict(int), "rep": False})
    total, nshards, ntensors = 0, 0, 0
    for _, h in read_headers(a.dir):
        nshards += 1
        for k, v in h.items():
            c, part, rep = classify(k)
            e = cls[c]
            nel = 1
            for s in v["shape"]:
                nel *= s
            b = v["data_offsets"][1] - v["data_offsets"][0]
            assert b == nel * DT[v["dtype"]], k
            e["n"] += 1
            e["bytes"] += b
            e["parts"][part] += b
            e["rep"] = rep if e["n"] == 1 else (e["rep"] and rep)
            if part == "packed":
                e["params"] += nel * 8                  # eight int4 a 32-bit word
            elif part in ("bf16", "f32") and k.endswith((".weight", ".bias")):
                e["params"] += nel
            total += b
            ntensors += 1
    out, rows = {}, []
    t_rank = t_read = t_mtp_read = 0.0
    for c in sorted(cls):
        e = cls[c]
        # bytes a rank: split classes / 4, whole classes x 1 (a mixed class, the indexer with its whole k_norm, is
        # counted split: 0.5 KB a layer)
        div = 1 if e["rep"] else TP
        stored_rank = sum(b for p, b in e["parts"].items() if p != "shape") / div
        if c.startswith("routed experts"):
            frac = TOPK / N_EXP
        elif c == "embed":
            frac = 0.0
        else:
            frac = 1.0
        read = stored_rank * frac
        if c.startswith("mtp "):
            mtp_frac = TOPK / N_EXP if "experts" in c and "routed" in c else 1.0
            t_mtp_read += stored_rank * mtp_frac
            read_main = 0.0
        else:
            read_main = read
        t_rank += stored_rank
        t_read += read_main
        parts = {p: b for p, b in sorted(e["parts"].items())}
        out[c] = {"tensors": e["n"], "params": e["params"], "stored_bytes": e["bytes"], "parts": parts,
                  "replicated": e["rep"], "bytes_a_rank": stored_rank, "decode_read_a_rank": read_main}
        rows.append((c, e, parts, stored_rank, read_main))
    t_mtp_read += out.get("head", {}).get("bytes_a_rank", 0)   # a draft pass reads the head too
    print(f"{nshards} shards, {ntensors} tensors, {total / GIB:.2f} GiB stored ({total / 1e9:.1f} GB)")
    print(f"{'class':44s} {'tensors':>7s} {'params M':>10s} {'stored GiB':>10s}  {'parts GiB':40s} {'GiB/rank':>9s} {'read GB/tok/rank':>16s}")
    for c, e, parts, sr, rd in rows:
        ps = " ".join(f"{p}:{b / GIB:.3f}" for p, b in parts.items())
        print(f"{c + (' (whole)' if e['rep'] else ''):44s} {e['n']:7d} {e['params'] / 1e6:10.1f} {e['bytes'] / GIB:10.3f}  "
              f"{ps:40s} {sr / GIB:9.3f} {rd / 1e9:16.4f}")
    print(f"{'all, a rank (weight_shape metadata excluded)':44s} {'':7s} {'':10s} {'':10s}  {'':40s} {t_rank / GIB:9.3f} {t_read / 1e9:16.4f}")
    print(f"decode floor at {a.gbps:.0f} GB/s, one row, weights only: {t_read / 1e9:.3f} GB -> {t_read / 1e9 / a.gbps * 1e3:.2f} ms")
    print(f"one MTP draft pass (layer 78 at 8 of 168 experts + the head): {t_mtp_read / 1e9:.3f} GB -> "
          f"{t_mtp_read / 1e9 / a.gbps * 1e3:.2f} ms")
    if a.json:
        json.dump({"classes": out, "total_stored": total, "bytes_a_rank": t_rank, "decode_read_a_rank": t_read,
                   "mtp_pass_read_a_rank": t_mtp_read}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
