"""Repack the BF16 master's routed experts into the quantiser's formats for the tracks (experiment 3), once,
into a cache that glmref.py's tracks read; check every packed tensor's SHA-256 against the quantiser's results.

  python3 qprep.py --master MASTER_DIR --out CACHE --layers 3 40 --formats e2m1sub3x2_g16 \
      [--experts all|0,51,...] [--ref-results RESULTS.jsonl ...] [--threads 2]

The packing is quant/run.py's own (make_formats, phash), so the bytes are the quantiser's. Stored per tensor:
  e2m1sub3x2_g16: CACHE/L{l}/{fmt}/{e}.{proj}.npz with v2 (int8 [R, C], twice the signed E2M1 level),
                  sc8 (uint8 [R, C/16], the e4m3 group scale without the subset bit) and s2 (f32). The track
                  computes W' = (v2 / 2) * e4m3(sc8), exact in f32 (a 3-bit E2M1 value times an e4m3 scale), and
                  applies s2 after the GEMM, as the reference does NVFP4's weight_scale_2 (the kernel's order). The
                  quantiser's levelset3_dequant rounds e4m3·s2 first; the two agree within one f32 rounding
                  (asserted).
  trellis_k3/k4:  CACHE/L{l}/{fmt}/{e}.{proj}.npy, the f32 dequantised weight (trellis.dequantise), scale 1.
A tensor already in the cache is skipped (resumable). Writes CACHE/provenance.jsonl: one line per tensor with
the packed SHA-256, the master's SHA-256 and whether it equals the quantiser's recorded hash.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

QDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quant")
sys.path.insert(0, QDIR)
import formats as FM  # noqa: E402
import run as RQ  # noqa: E402


def factor_levelset(p):
    """levelset3 packed bytes -> (v2 int8, sc8 uint8, s2 f32)."""
    g = int(p["g"])
    assert g == 16
    code = torch.from_numpy(FM.unpack3(p["codes"])).long()
    R = code.shape[0]
    code = code.reshape(R, -1, g)
    raw = torch.from_numpy(p["scale"])
    sel = (raw >> 7).long()
    mags = torch.from_numpy(p["subsets"])
    j, s = code & 3, code >> 2
    val = mags[sel.unsqueeze(-1).expand_as(j), j] * (1 - 2 * s).float()
    v2 = (val * 2).round().to(torch.int8).reshape(R, -1)
    assert torch.equal(v2.float() / 2, val.reshape(R, -1))
    return v2, (raw & 127), float(p["scale_2"])


def factored_weight(v2, sc8):
    R = v2.shape[0]
    sc = sc8.view(torch.float8_e4m3fn).float()
    return ((v2.float() / 2).reshape(R, -1, 16) * sc.unsqueeze(-1)).reshape(R, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 40])
    ap.add_argument("--formats", default="e2m1sub3x2_g16")
    ap.add_argument("--experts", default="all")
    ap.add_argument("--ref-results", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    RQ._THREADS[0] = a.threads
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    M = RQ.Master(a.master)
    fmts = RQ.make_formats(a.formats.split(","))
    ref = {}
    for path in a.ref_results:
        for line in open(os.path.expanduser(path)):
            r = json.loads(line)
            if "proj" in r:
                for p, v in r["proj"].items():
                    ref.setdefault((r["layer"], r["expert"], r["format"], p), set()).add(v["sha256"])
    prov = open(os.path.join(out, "provenance.jsonl"), "a")
    experts = range(256) if a.experts == "all" else [int(x) for x in a.experts.split(",")]
    for L in a.layers:
        for fname, (qf, dqf) in fmts.items():
            d = os.path.join(out, f"L{L}", fname)
            os.makedirs(d, exist_ok=True)
            for e in experts:
                for p in RQ.PROJ:
                    stem = os.path.join(d, f"{e}.{p}")
                    fn = stem + (".npz" if fname.startswith("e2m1") else ".npy")
                    if os.path.exists(fn):
                        continue
                    t0 = time.time()
                    w, src = M.get(RQ.nkey(L, e, p))
                    packed = qf(w)
                    h = RQ.phash(packed)
                    if fname.startswith("e2m1"):
                        v2, sc8, s2 = factor_levelset(packed)
                        wq = dqf(packed).float()
                        fw = factored_weight(v2, sc8) * torch.tensor(s2, dtype=torch.float32)
                        assert ((fw - wq).abs() <= wq.abs() * 2.0 ** -23).all()   # (v·e4m3)·s2 vs v·(e4m3·s2)
                        np.savez(stem + ".tmp.npz", v2=v2.numpy(), sc8=sc8.numpy(), s2=np.float32(s2))
                        os.replace(stem + ".tmp.npz", fn)
                    else:
                        np.save(stem + ".tmp.npy", dqf(packed).float().numpy())
                        os.replace(stem + ".tmp.npy", fn)
                    known = ref.get((L, e, fname, p))
                    rec = {"layer": L, "expert": e, "format": fname, "proj": p, "sha256": h, "src_sha256": src,
                           "quantiser_sha256_equal": None if known is None else (h in known),
                           "wall_s": round(time.time() - t0, 2)}
                    prov.write(json.dumps(rec) + "\n")
                    prov.flush()
                    print(L, e, fname, p, rec["quantiser_sha256_equal"], rec["wall_s"], flush=True)


if __name__ == "__main__":
    main()
