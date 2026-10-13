#!/usr/bin/env python3
"""Does the converter's BF16 -> FP8 rule reproduce zai's FP8 release byte for byte?

For each named linear weight: the BF16 master (NVIDIA's checkpoint where it holds BF16, else fetched from
zai-org/GLM-5.3-BF16), and zai's FP8 codes and weight_scale_inv fetched from zai-org/GLM-5.3 (both pinned,
fetch.py). The converter's rule (quant.py fp8_rtn: per 128x128 block scale = f32(amax) * f32(1/448) (a multiply by the
rounded reciprocal, not a division), codes = e4m3(w / scale) in f32, round to nearest even) is applied to the master
and the codes and scales are compared bit for bit with the release. --rule pack applies engine/attn/tools/pack.py's
fp8_from_master instead (scale = amax / 448, a division), which differs from the release by one ulp in about
half the scales and so in ~0.1 % of the codes.

    python -I fp8check.py --ckpt NV --cache DIR [--json OUT] [NAME ...]
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "attn", "tools"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pack  # noqa: E402
import quant  # noqa: E402
import sources  # noqa: E402

# Across layers and every FP8 class the converter makes: attention (q_a, kv_a with its partial 128-row block,
# q_b, kv_b, o_proj), the indexer (wq_b, wk), shared experts, the MTP layer's, and the dense MLPs of layers 0-2
# (their master is not in NVIDIA's checkpoint, which requantised them per tensor: fetched from the BF16 mirror).
DEFAULT = [
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.5.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.20.self_attn.q_b_proj.weight",
    "model.layers.33.self_attn.kv_b_proj.weight",
    "model.layers.77.self_attn.o_proj.weight",
    "model.layers.42.self_attn.indexer.wq_b.weight",
    "model.layers.42.self_attn.indexer.wk.weight",
    "model.layers.3.mlp.shared_experts.up_proj.weight",
    "model.layers.60.mlp.shared_experts.down_proj.weight",
    "model.layers.78.self_attn.q_b_proj.weight",
    "model.layers.78.mlp.shared_experts.gate_proj.weight",
] + [f"model.layers.{L}.mlp.{p}_proj.weight" for L in range(3) for p in ("gate", "up", "down")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rule", choices=["converter", "pack"], default="converter")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    nv = sources.Checkpoint(a.ckpt)
    fp8 = sources.Fetched(a.cache, "fp8")
    bf16 = sources.Fetched(a.cache, "bf16")
    rows, tot = [], {"codes": 0, "codes_eq": 0, "scales": 0, "scales_eq": 0, "tensors": 0, "tensors_eq": 0}
    for name in a.names or DEFAULT:
        t0 = time.time()
        if name in nv.meta and nv.meta[name][1] == "BF16":
            w, wsha = nv.get(name)
            msrc = "nvidia"
        else:
            w, wsha = bf16.get(name)
            msrc = "zai-bf16"
        zc, _ = fp8.get(name)
        zs, _ = fp8.get(name + "_scale_inv")
        codes, scales = quant.fp8_rtn(w) if a.rule == "converter" else pack.fp8_from_master(w)
        zc = zc.view(torch.uint8).numpy()
        zs = zs.float().numpy()
        ceq = int((codes == zc).sum())
        seq = int((scales.view(np.uint32) == zs.view(np.uint32)).sum())
        ok = ceq == codes.size and seq == scales.size
        r = {"name": name, "master": msrc, "master_sha256": wsha, "shape": list(w.shape), "codes": codes.size,
             "codes_equal": ceq, "scales": scales.size, "scales_equal": seq, "identical": ok}
        if not ok:  # diagnose: codes under zai's own scales, and how far the scales are apart
            q = (w.float() / torch.from_numpy(zs).repeat_interleave(128, 0)[:w.shape[0]].repeat_interleave(128, 1))
            r["codes_equal_with_zai_scales"] = int((q.to(torch.float8_e4m3fn).view(torch.uint8).numpy() == zc).sum())
            r["scale_max_rel_diff"] = float(np.abs(scales / zs - 1).max())
        rows.append(r)
        tot["codes"] += codes.size
        tot["codes_eq"] += ceq
        tot["scales"] += scales.size
        tot["scales_eq"] += seq
        tot["tensors"] += 1
        tot["tensors_eq"] += ok
        print(f"{name} [{msrc}] {list(w.shape)}: codes {ceq}/{codes.size}, scales {seq}/{scales.size} "
              f"{'IDENTICAL' if ok else 'DIFFERENT ' + json.dumps({k: r[k] for k in r if k.startswith(('codes_eq', 'scale_max'))})}"
              f" ({time.time() - t0:.1f} s)", flush=True)
    print(f"total: {tot['tensors_eq']}/{tot['tensors']} tensors identical; codes {tot['codes_eq']}/{tot['codes']}, "
          f"scales {tot['scales_eq']}/{tot['scales']}")
    if a.json:
        json.dump({"rule": a.rule, "tensors": rows, "total": tot, "fp8": fetch_rev("fp8"), "bf16": fetch_rev("bf16")},
                  open(a.json, "w"), indent=1)
    return 0 if tot["tensors_eq"] == tot["tensors"] else 1


def fetch_rev(k):
    import fetch
    return "%s@%s" % fetch.REPOS[k]


if __name__ == "__main__":
    sys.exit(main())
