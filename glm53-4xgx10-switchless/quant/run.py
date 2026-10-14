#!/usr/bin/env python3
"""Quantise the fetched BF16 experts into every candidate format and measure them.

    python -I run.py MASTER NV OUT --layers 3 40 [--experts all|0,1,2] [--formats nvfp4,...]
                     [--trellis-experts 0,37,...] [--threads 8] [--cov DIR]

MASTER: fetch_master.py's directory (tensors/*.bin + manifest.jsonl); NV: the NVFP4 checkpoint snapshot.
Writes OUT/results.jsonl, one line per (layer, expert, format): per projection the bits a weight, the relative
RMS weight error, the relative output error on Gaussian inputs, the SHA-256 of the packed bytes and the CPU
seconds; per expert the error of the whole expert's output. For nvfp4 it also compares the packed bytes,
the scales and scale_2 with the checkpoint's. Every master tensor's bytes are re-hashed against the manifest.

Inputs for the output errors (fixed seed): x = g * z, z ~ N(0, 1) of shape [512, 6144], g the layer's
post_attention_layernorm.weight (the RMSNorm output of a unit-RMS hidden state); gate/up see x, down sees
h = silu(x Wg^T) * (x Wu^T) computed with the master weights; the expert error is that of
down(silu(gate x) * up x) with all three projections quantised. With --cov DIR, DIR/layer{L}_expert{E}.pt
holding {'gate_up': Sigma_x, 'down': Sigma_h} adds sqrt(tr(E S E^T) / tr(W S W^T)) per projection.
"""
import argparse, hashlib, json, os, struct, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formats as F  # noqa: E402
import trellis as T  # noqa: E402

PROJ = ("gate", "up", "down")
# mixtures scored at the expert level from formats run in the same pass: name -> (gate/up format, down format)
MIXES = {"mix_k3gu_k4d": ("trellis_k3", "trellis_k4"), "mix_sub3x2gu_nvfp4d": ("e2m1sub3x2_g16", "nvfp4")}
SUBSET_A = (0.5, 2, 4, 6)          # the best single subset (quantisation.md § 3.3; subset_study.py)
SUBSET_B = (0.5, 1.5, 3, 6)        # the best partner for A (subset_study.py, layer 3, 2026-10-13)


def nkey(layer, e, p):
    return f"model.layers.{layer}.mlp.experts.{e}.{p}_proj.weight"


class Master:
    def __init__(self, d):
        self.d = os.path.expanduser(d)
        self.man = {}
        for l in open(os.path.join(self.d, "manifest.jsonl")):
            if l.strip():
                r = json.loads(l)
                self.man[r["name"]] = r

    def get(self, name):
        r = self.man[name]
        b = open(os.path.join(self.d, "tensors", name + ".bin"), "rb").read()
        assert hashlib.sha256(b).hexdigest() == r["sha256"], name
        return torch.frombuffer(bytearray(b), dtype=torch.bfloat16).reshape(r["shape"]).float(), r["sha256"]


class NV:
    DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}

    def __init__(self, d):
        self.d = os.path.expanduser(d)
        self.idx = json.load(open(os.path.join(self.d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}

    def raw(self, name):
        f = os.path.join(self.d, self.idx[name])
        if f not in self.hdr:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.hdr[f] = (n, json.loads(fh.read(n)))
        n, h = self.hdr[f]
        v = h[name]
        a, b = v["data_offsets"]
        with open(f, "rb") as fh:
            fh.seek(8 + n + a)
            data = fh.read(b - a)
        return data, v

    def get(self, name):
        data, v = self.raw(name)
        t = torch.frombuffer(bytearray(data), dtype=self.DT[v["dtype"]])
        return t.reshape(v["shape"]) if v["shape"] else t[0]


def phash(packed):
    h = hashlib.sha256()
    for k in sorted(packed):
        v = np.ascontiguousarray(packed[k])
        h.update(k.encode())
        h.update(str(v.dtype).encode() + str(v.shape).encode())
        h.update(v.tobytes())
    return h.hexdigest()


def make_formats(names):
    """name -> (quantise(w float32 torch) -> packed, dequantise(packed) -> float32 torch)."""
    allf = {
        "nvfp4": (F.nvfp4_rtn, F.nvfp4_dequant),     # called with the gate/up shared amax below
        "e2m1sub3_g16": (lambda w: F.levelset3(w, [SUBSET_A]), F.levelset3_dequant),
        "e2m1sub3x2_g16": (lambda w: F.levelset3(w, [SUBSET_A, SUBSET_B]), F.levelset3_dequant),
        "int3_g16_e4m3": (lambda w: F.int3(w, 16, "e4m3"), F.int3_dequant),
        "int3_g32_e4m3": (lambda w: F.int3(w, 32, "e4m3"), F.int3_dequant),
        "int3_g32_bf16": (lambda w: F.int3(w, 32, "bf16"), F.int3_dequant),
        "int3_g16_bf16": (lambda w: F.int3(w, 16, "bf16"), F.int3_dequant),
    }
    for K in (3, 4):
        allf[f"trellis_k{K}"] = (
            (lambda K: lambda w, seed=0: _tq(w, K))(K),
            lambda p: torch.from_numpy(T.dequantise(_unbox(p)).astype(np.float32)))
    return {n: allf[n] for n in names}


_CB = {}
_THREADS = [8]


def _tq(w, K):
    if "mul1" not in _CB:
        _CB["mul1"] = T.codebook("mul1")
    q = T.quantise(w.numpy(), K, "mul1", seed=0, threads=_THREADS[0], cb=_CB["mul1"])
    return {"trellis": q["trellis"], "suh": q["suh"], "svh": q["svh"], "K": np.array(K, np.int32),
            "g": np.array(q["g"], np.float64)}


def _unbox(p):
    return {"trellis": p["trellis"], "suh": p["suh"], "svh": p["svh"], "K": int(p["K"]), "codebook": "mul1"}


def bpw_of(name, packed, numel):
    if name.startswith("trellis"):
        return 8 * (packed["trellis"].nbytes + packed["suh"].nbytes + packed["svh"].nbytes) / numel
    return F.bpw(packed, numel)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master")
    ap.add_argument("nv")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--experts", default="all")
    ap.add_argument("--formats", default="nvfp4,e2m1sub3_g16,e2m1sub3x2_g16,int3_g16_e4m3,int3_g32_e4m3,int3_g32_bf16,int3_g16_bf16")
    ap.add_argument("--trellis-experts", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cov", default=None)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    _THREADS[0] = a.threads
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    M, N = Master(a.master), NV(a.nv)
    fmts = make_formats([f for f in a.formats.split(",") if f])
    texp = {int(x) for x in a.trellis_experts.split(",") if x}
    resf = open(os.path.join(out, f"results-{a.tag}.jsonl"), "a")
    for L in a.layers:
        g = N.get(f"model.layers.{L}.post_attention_layernorm.weight").float()
        gen = torch.Generator().manual_seed(1000 + L)
        X = torch.randn(512, g.shape[0], generator=gen) * g
        experts = range(256) if a.experts == "all" else [int(x) for x in a.experts.split(",")]
        for e in experts:
            W, src = {}, {}
            for p in PROJ:
                W[p], src[p] = M.get(nkey(L, e, p))
            Yg, Yu = X @ W["gate"].T, X @ W["up"].T
            Hh = torch.nn.functional.silu(Yg) * Yu
            Y = Hh @ W["down"].T
            ins = {"gate": X, "up": X, "down": Hh}
            cov = None
            if a.cov:
                cp = os.path.join(os.path.expanduser(a.cov), f"layer{L}_expert{e}.pt")
                if os.path.exists(cp):
                    cov = torch.load(cp)
            deq = {}
            for fname, (qf, dqf) in fmts.items():
                if fname.startswith("trellis") and e not in texp:
                    continue
                rec = {"layer": L, "expert": e, "format": fname, "proj": {}, "date": time.strftime("%Y-%m-%d")}
                Wq = {}
                for p in PROJ:
                    t0 = time.process_time(), time.time()
                    if fname == "nvfp4":
                        am = torch.maximum(W["gate"].abs().max(), W["up"].abs().max()) if p != "down" else None
                        packed = qf(W[p], am)
                    else:
                        packed = qf(W[p])
                    t1 = time.process_time(), time.time()
                    Wq[p] = dqf(packed).float()
                    r = {"bpw": bpw_of(fname, packed, W[p].numel()), "werr": rel(Wq[p], W[p]),
                         "oerr": rel(ins[p] @ Wq[p].T, ins[p] @ W[p].T), "sha256": phash(packed),
                         "src_sha256": src[p], "cpu_s": t1[0] - t0[0], "wall_s": t1[1] - t0[1]}
                    if cov is not None:
                        S = cov["gate_up" if p != "down" else "down"].float()
                        E_ = Wq[p] - W[p]
                        r["cerr"] = float(((E_ @ S * E_).sum() / ((W[p] @ S) * W[p]).sum()).sqrt())
                    if fname == "nvfp4":
                        nm = nkey(L, e, p)
                        r["nv_weight_eq"] = packed["weight"].tobytes() == N.raw(nm)[0]
                        r["nv_scale_eq"] = packed["weight_scale"].tobytes() == N.raw(nm + "_scale")[0]
                        r["nv_scale2_eq"] = packed["weight_scale_2"].tobytes() == N.raw(nm + "_scale_2")[0]
                        if not r["nv_weight_eq"]:
                            nvb = np.frombuffer(N.raw(nm)[0], np.uint8)
                            r["nv_weight_byte_match"] = float((nvb == packed["weight"].reshape(-1)).mean())
                    if fname.startswith("trellis"):
                        r["g"] = float(packed["g"])
                    rec["proj"][p] = r
                Yq = (torch.nn.functional.silu(X @ Wq["gate"].T) * (X @ Wq["up"].T)) @ Wq["down"].T
                rec["expert_oerr"] = rel(Yq, Y)
                deq[fname] = Wq
                resf.write(json.dumps(rec) + "\n")
                resf.flush()
                print(L, e, fname, " ".join(f"{p}:{rec['proj'][p]['werr']:.4f}/{rec['proj'][p]['oerr']:.4f}"
                                             for p in PROJ), f"exp:{rec['expert_oerr']:.4f}",
                      f"{sum(rec['proj'][p]['wall_s'] for p in PROJ):.1f}s", flush=True)
            for mname, (fgu, fd) in MIXES.items():
                if fgu in deq and fd in deq:
                    Wg, Wu, Wd = deq[fgu]["gate"], deq[fgu]["up"], deq[fd]["down"]
                    Yq = (torch.nn.functional.silu(X @ Wg.T) * (X @ Wu.T)) @ Wd.T
                    rec = {"layer": L, "expert": e, "format": mname, "mix": [fgu, fd], "expert_oerr": rel(Yq, Y),
                           "date": time.strftime("%Y-%m-%d")}
                    resf.write(json.dumps(rec) + "\n")
                    resf.flush()
                    print(L, e, mname, f"exp:{rec['expert_oerr']:.4f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
