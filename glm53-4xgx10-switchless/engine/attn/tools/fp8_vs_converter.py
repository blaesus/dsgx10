#!/usr/bin/env python3
"""pack.py's FP8 quantiser against the converter's (engine/weights/quant.py fp8_rtn, zai's rule) on real
GLM-5.3 tensors: codes and scales must be byte-identical, and (with the fetch cache) equal to zai's FP8 release.
The BF16 masters are read as engine/weights/fp8check.py reads them: NVIDIA's checkpoint where it holds BF16, else
zai's BF16 mirror from the fetch cache.

    python3 -I fp8_vs_converter.py --ckpt NV --cache CACHE [NAME ...]

Exit 0 when every tensor is identical."""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "weights"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pack  # noqa: E402
import quant  # noqa: E402
import sources  # noqa: E402

# small tensors of three classes, one with a partial 128-row block (kv_a: 576 rows)
DEFAULT = [
    "model.layers.42.self_attn.indexer.wk.weight",
    "model.layers.5.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.q_a_proj.weight",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    nv = sources.Checkpoint(a.ckpt)
    fp8 = sources.Fetched(a.cache, "fp8")
    bf16 = sources.Fetched(a.cache, "bf16")
    bad = 0
    for name in a.names or DEFAULT:
        t0 = time.time()
        if name in nv.meta and nv.meta[name][1] == "BF16":
            w, _ = nv.get(name)
            src = "nvidia"
        else:
            w, _ = bf16.get(name)
            src = "zai-bf16"
        pc, ps = pack.fp8_from_master(w)
        qc, qs = quant.fp8_rtn(w)
        same = pc.shape == qc.shape and ps.shape == qs.shape and pc.tobytes() == qc.tobytes() and \
            ps.view(np.uint32).tobytes() == qs.view(np.uint32).tobytes()
        zc, _ = fp8.get(name)
        zs, _ = fp8.get(name + "_scale_inv")
        zc = zc.view(torch.uint8).numpy()
        zs = zs.float().numpy()
        rel = pc.tobytes() == zc.tobytes() and ps.view(np.uint32).tobytes() == zs.view(np.uint32).tobytes()
        bad += not (same and rel)
        print(f"{'PASS' if same and rel else 'FAIL'} {name} [{src}] {list(w.shape)}: pack.py = converter "
              f"{'byte for byte' if same else 'DIFFERS'}; = zai's release {'byte for byte' if rel else 'DIFFERS'} "
              f"({pc.size} codes, {ps.size} scales, {time.time() - t0:.1f} s)", flush=True)
    print(f"{len(a.names or DEFAULT) - bad} of {len(a.names or DEFAULT)} identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
