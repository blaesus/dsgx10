#!/usr/bin/env python3
"""Where the NVFP4 checkpoint's numbers come from, checked on sample tensors.

Fetches single tensors from the Hugging Face repos with HTTP range requests (header, then the tensor's
bytes; nothing else is downloaded) into OUT, then checks against the local NVFP4 checkpoint NV:
  1. BF16 tensors of NV (attention, shared experts, indexer, MTP experts): rounding them to e4m3 with
     the official FP8 release's 128x128 block scales gives the release's FP8 codes exactly, i.e. NV's
     BF16 is the master the FP8 release was cut from;
  2. NVFP4 experts of NV: round-to-nearest-even from the BF16 mirror, with
     scale_2 = amax/(6*448), block scale = e4m3(amax16/6/scale_2), code = E2M1_rne(w/(scale*scale_2)),
     reproduces NV's codes and scales bit for bit.
Optionally (--formats) prints the relative RMS weight error of candidate 3-bit codes on the fetched
BF16 experts (512 random rows each).

    python -I provenance.py NV OUT [--formats]
Needs torch (CPU). OUT must be an empty or earlier OUT directory; fetched bytes are data only.
"""
import argparse, itertools, json, os, struct, sys, urllib.request
import torch

FP8_REPO, FP8_REV = "zai-org/GLM-5.3", "c5087a76d148760811ffbdedfe6ff9f194e46e38"
BF16_REPO, BF16_REV = "zai-org/GLM-5.3-BF16", "9d2398f478cab2de883137db3a36ad2c96205e24"
BF16_CHECK = ["model.layers.10.self_attn.o_proj.weight", "model.layers.10.mlp.shared_experts.gate_proj.weight",
              "model.layers.10.self_attn.indexer.wk.weight", "model.layers.78.mlp.experts.0.down_proj.weight"]
EXPERTS = ["model.layers.10.mlp.experts.0.gate_proj.weight", "model.layers.10.mlp.experts.0.down_proj.weight"]
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def fetch(repo, rev, name, out):
    """Fetch one tensor (by the repo's index) into out/<repo>/<name>; return it."""
    d = os.path.join(out, repo.replace("/", "_"))
    os.makedirs(d, exist_ok=True)
    ip = os.path.join(d, "index.json")
    base = f"https://huggingface.co/{repo}/resolve/{rev}/"
    if not os.path.exists(ip):
        urllib.request.urlretrieve(base + "model.safetensors.index.json", ip)
    shard = json.load(open(ip))["weight_map"][name]

    def rng(a, b):
        r = urllib.request.Request(base + shard, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(r, timeout=300) as f:
            return f.read()
    hp = os.path.join(d, shard + ".header.json")
    if not os.path.exists(hp):
        n = struct.unpack("<Q", rng(0, 7))[0]
        json.dump({"n": n, "h": json.loads(rng(8, 8 + n - 1))}, open(hp, "w"))
    hh = json.load(open(hp))
    v = hh["h"][name]
    tp = os.path.join(d, name)
    if not os.path.exists(tp):
        a, b = v["data_offsets"]
        data = rng(8 + hh["n"] + a, 8 + hh["n"] + b - 1)
        assert len(data) == b - a
        open(tp, "wb").write(data)
    return torch.frombuffer(bytearray(open(tp, "rb").read()), dtype=DT[v["dtype"]]).reshape(v["shape"])


def local(nv, name, _c={}):
    idx = _c.setdefault("idx", json.load(open(os.path.join(nv, "model.safetensors.index.json")))["weight_map"])
    f = os.path.join(nv, idx[name])
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        v = json.loads(fh.read(n))[name]
        a, b = v["data_offsets"]
        fh.seek(8 + n + a)
        t = torch.frombuffer(bytearray(fh.read(b - a)), dtype=DT[v["dtype"]])
    return t.reshape(v["shape"]) if v["shape"] else t[0]


def blockfull(s, R, C):
    return s.repeat_interleave(128, 0)[:R].repeat_interleave(128, 1)[:, :C]


def rne_e2m1(x):
    """Nearest E2M1 magnitude, ties to the even code; returns signed values."""
    mag = E2M1[:8]
    d = (x.abs().clamp(max=6).unsqueeze(-1) - mag).abs()
    cand = d == d.min(-1, keepdim=True).values
    ar = torch.arange(8).expand_as(d)
    even = torch.where(cand & (ar % 2 == 0), ar, 99).min(-1).values
    low = torch.where(cand, ar, 99).min(-1).values
    q = torch.where(even == 99, low, even)
    return mag[q] * torch.sign(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nv")
    ap.add_argument("out")
    ap.add_argument("--formats", action="store_true")
    a = ap.parse_args()
    for nm in BF16_CHECK:
        w8 = fetch(FP8_REPO, FP8_REV, nm, a.out).float()
        s = fetch(FP8_REPO, FP8_REV, nm + "_scale_inv", a.out)
        b = local(a.nv, nm).float()
        R, C = b.shape
        q = (b / blockfull(s, R, C)).to(torch.float8_e4m3fn).float()
        print(f"{nm}: e4m3(NVFP4-ckpt bf16 / official scale_inv) == official FP8 codes: "
              f"{100 * (q == w8).float().mean().item():.4f} % of {b.numel()}")
    for nm in EXPERTS:
        w = fetch(BF16_REPO, BF16_REV, nm, a.out).float()
        p = local(a.nv, nm)
        sc = local(a.nv, nm + "_scale").float()
        s2 = local(a.nv, nm + "_scale_2").item()
        R = w.shape[0]
        codes = torch.stack([(p & 15).long(), (p >> 4).long()], -1).reshape(R, -1)
        my_s2 = (torch.tensor(w.abs().max().item()) / torch.tensor(6 * 448.0)).item()
        s2t = torch.tensor(s2)
        blk = w.reshape(R, -1, 16)
        my_sc = (blk.abs().amax(-1) / 6 / s2t).to(torch.float8_e4m3fn).float()
        v = rne_e2m1(torch.nan_to_num(w / (sc * s2t).repeat_interleave(16, 1), 0.0))
        print(f"{nm}: scale_2 == amax/2688: {my_s2 == s2}; block scales equal "
              f"{100 * (my_sc == sc).float().mean().item():.4f} %; values equal "
              f"{100 * (v == E2M1[codes]).float().mean().item():.4f} %; NVFP4 rel RMS error vs master "
              f"{((E2M1[codes] * sc.repeat_interleave(16, 1) * s2 - w).norm() / w.norm()).item():.4f}")
    if a.formats:
        formats(a)


def formats(a):
    torch.manual_seed(0)
    E = [0, .5, 1, 1.5, 2, 3, 4, 6]
    full = E + [-v for v in E[1:]]
    for nm in EXPERTS:
        w = fetch(BF16_REPO, BF16_REV, nm, a.out).float()
        w = w[torch.randperm(w.shape[0])[:512]]

        def quant(levels, g=16, sc_fmt="e4m3", search=False):
            lv = torch.tensor(sorted(levels))
            blk = w.reshape(w.shape[0], -1, g)
            amax = blk.abs().amax(-1, keepdim=True)
            top = lv.abs().max()
            s2 = w.abs().max() / (top * 448)
            best = best_err = None
            for m in ([1.0, .95, .9, .85, .8, .75, .7] if search else [1.0]):
                raw = amax / top * m / s2
                sc = raw.to(torch.float8_e4m3fn).float() * s2 if sc_fmt == "e4m3" else (raw * s2).half().float()
                sc = torch.where(sc == 0, torch.ones_like(sc), sc)
                deq = lv[(blk / sc).unsqueeze(-1).sub(lv).abs().argmin(-1)] * sc
                err = ((deq - blk) ** 2).sum(-1, keepdim=True)
                if best is None:
                    best, best_err = deq, err
                else:
                    t = err < best_err
                    best, best_err = torch.where(t, deq, best), torch.where(t, err, best_err)
            return ((best - blk).norm() / blk.norm()).item()
        r = {"NVFP4, amax scales (4.5 bpw)": quant(full),
             "NVFP4, best of 7 scales per block (4.5)": quant(full, search=True),
             "int3 {-4..3}, g16 e4m3 (3.5)": quant([-4, -3, -2, -1, 0, 1, 2, 3], search=True),
             "int3 {-3..3}, g32 e4m3 (3.25)": quant([-3, -2, -1, 0, 1, 2, 3], g=32, search=True),
             "int3 {-4..3}, g32 f16 (3.5)": quant([-4, -3, -2, -1, 0, 1, 2, 3], g=32, sc_fmt="f16", search=True)}
        sub = []
        for mags in itertools.combinations(E[1:], 4):
            sub.append((quant(list(mags) + [-m for m in mags], search=True), "±" + ",".join(map(str, mags))))
        for mags in itertools.combinations(E[1:], 3):
            sub.append((quant([0] + list(mags) + [-m for m in mags], search=True), "0,±" + ",".join(map(str, mags))))
        sub.sort()
        for e, n in sub[:3]:
            r[f"E2M1 subset {{{n}}}, g16 e4m3 (3.5)"] = e
        print(nm.split("experts.0.")[1])
        for k, v in r.items():
            print(f"   {k:45s} rel RMS error {v:.4f}")


if __name__ == "__main__":
    sys.exit(main())
