"""Third, independent check of a .dsw file: torch + safetensors (the reference's own stack) slice the
checkpoint tensors from each TOC entry's declarative description, and the bytes must match. Tiled
tensors (version 2) are un-tiled with torch reshapes and permutes, a third implementation of the
layouts after the kernels' tile functions (which wrote them) and layout.h (which dsw_verify uses).
It also confirms the cuts agree with the reference's tensor-parallel split (ref/inference/convert.py
`mapping`) wherever ours follows it, and lists where it deliberately does not.

Run in the canary-roce image (it has torch and safetensors):
  docker run --rm -v CKPT:/models:ro -v DSWDIR:/dsw:ro -v REPO:/work dsv41-4x-spark:canary-roce \
      /opt/sglang/bin/python3 /work/engine/weights/check_dsw.py /models /dsw/rank0.dsw [--sample N]
"""

import json
import os
import random
import struct
import sys

import torch
from safetensors import safe_open

HDR = struct.Struct("<8sIIIIII QQQQ II 64s256s32s")
ENT = struct.Struct("<128s128s96s II 4Q QQ IIII II 4Q")
KIND = {1: "replicated", 2: "rows", 3: "cols", 4: "rows-seg"}
E2M1X2 = 5
# convert.py's split dims for the reference stack (wq_b/weights_proj there are the attention's and
# the indexer's alike; embed is split there too)
REF_DIM = {"wq_b": 0, "wo_a": 0, "wo_b": 1, "attn_sink": 0, "head": 0, "embed": 0, "weights_proj": 0}


def cstr(b):
    return b.split(b"\0", 1)[0].decode()


def untile(buf, layout, dtype, shape):
    """Natural bytes of a tiled tensor (dsw.h Layout), as a flat uint8 tensor."""
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    if layout in (1, 2):  # [N/8][K/64][8 rows][64 B] -> [N][K] bytes (L2: K = 2 x the bf16 columns)
        n = shape[0]
        k = len(buf) // n
        return t.view(n // 8, k // 64, 8, 64).permute(0, 2, 1, 3).reshape(-1)
    assert layout == 3, layout
    rows, last = shape[-2], shape[-1]
    mats = len(buf) // (rows * last)
    if dtype == E2M1X2:  # values: (mat, n16, p, g, t, h, l, byte) -> rows (n16, h, g), bytes (p, l, t, byte)
        p = last * 2 // 64
        return t.view(mats, rows // 16, p, 8, 4, 2, 2, 4).permute(0, 1, 5, 3, 2, 6, 4, 7).reshape(-1)
    p = last * 32 // 64  # scales: (mat, n16, p, g, h, l) -> rows (n16, h, g), columns (p, l)
    return t.view(mats, rows // 16, p, 8, 2, 2).permute(0, 1, 4, 3, 2, 5).reshape(-1)


def main():
    ckpt, path = sys.argv[1], sys.argv[2]
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 0
    weight_map = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        f = weight_map[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(ckpt, f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    with open(path, "rb") as fh:
        h = HDR.unpack(fh.read(HDR.size))
        magic, version, _, rank, world, n, entry_bytes, toc_off, data_off = h[:9]
        assert magic == b"DSGB10W1" and version in (1, 2) and entry_bytes == 512, h[:7]
        fh.seek(toc_off)
        toc = [ENT.unpack(fh.read(512)[: ENT.size]) for _ in range(n)]
        picks = range(n) if not sample else sorted(random.Random(0).sample(range(n), min(sample, n)))
        bad = tiled = 0
        deliberate = set()
        for i in picks:
            e = toc[i]
            name, source, kind, nstack, nseg = cstr(e[0]), cstr(e[1]), e[11], e[12], e[13]
            offset, nbytes, dtype, layout = e[9], e[10], e[3], e[16]
            fh.seek(offset)
            got = fh.read(nbytes)
            if layout:
                got = untile(got, layout, dtype, list(e[5 : 5 + e[4]])).numpy().tobytes()
                tiled += 1
            parts = []
            for k in range(max(1, nstack)):
                t = get(source.replace("{e}", str(k)))
                # bytes, keeping the logical row structure (I8 experts are already packed bytes)
                b = t.view(torch.uint8) if t.element_size() == 1 else t.contiguous().view(torch.uint8)
                b = b.reshape(t.shape[0], -1) if t.dim() > 1 else b.reshape(t.shape[0], -1)
                esz = t.element_size()
                if kind == 1:
                    s = b
                elif kind == 2:
                    m = t.shape[0] // world
                    s = b[m * rank : m * (rank + 1)]
                elif kind == 3:
                    c = t.shape[1] // world * esz
                    s = b[:, c * rank : c * (rank + 1)]
                elif kind == 4:
                    seg = t.shape[0] // nseg
                    m = seg // world
                    s = torch.cat([b[j * seg + m * rank : j * seg + m * (rank + 1)] for j in range(nseg)])
                parts.append(s.contiguous().reshape(-1))
            want = torch.cat(parts).numpy().tobytes()
            if want != got:
                bad += 1
                print("MISMATCH", name)
            # agreement with the reference's own TP split where ours follows it
            key = name.split(".")[-1] if name.endswith("attn_sink") else name.split(".")[-2]
            if key in REF_DIM:
                ours = {2: 0, 3: 1}.get(kind)
                if ours != REF_DIM[key]:
                    where = "indexer." if "indexer" in name else "markov_head." if "markov" in name else ""
                    deliberate.add(f"{where}{key}: reference splits dim {REF_DIM[key]}, here {KIND[kind]}")
        print(f"{path}: version {version}, rank {rank}/{world}, checked {len(picks)} of {n} tensors with torch "
              f"slicing ({tiled} un-tiled with torch), {bad} bad")
        for d in sorted(deliberate):
            print("  deliberate difference from convert.py:", d)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
