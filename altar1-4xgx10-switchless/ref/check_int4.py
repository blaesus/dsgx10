"""glmref.py's INT4 dequantisation against compressed-tensors' own (PackedQuantizationCompressor.decompress with
the checkpoint's quantization_config scheme) on real tensors, bit for bit.

  python3 -I check_int4.py --ckpt CKPT [--verified LIST] [--tensors NAME ...] [--all-experts L ...] [--threads 8]

NAME is a module prefix (e.g. model.layers.4.self_attn.o_proj); --all-experts L checks gate, up and down of all
168 experts of layer L. Prints, per tensor, the values compared and the bit-equal count; exits non-zero on any
difference. Also prints the rounding statistics: how many weights are not exact in BF16 before the rounding
(|q - z| * s needs more than 8 significant bits), i.e. where a dequantiser in another order could differ.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmref as G  # noqa: E402

DEFAULT = ["model.layers.4.mlp.experts.0.gate_proj", "model.layers.4.mlp.experts.0.up_proj",
           "model.layers.4.mlp.experts.0.down_proj", "model.layers.4.self_attn.q_b_proj",
           "model.layers.4.self_attn.kv_b_proj", "model.layers.4.self_attn.o_proj",
           "model.layers.40.mlp.experts.100.down_proj", "model.layers.76.mlp.experts.167.up_proj",
           "model.layers.76.self_attn.o_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--verified", default=None)
    ap.add_argument("--tensors", nargs="*", default=DEFAULT)
    ap.add_argument("--all-experts", type=int, nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    import compressed_tensors
    with open(os.path.join(a.ckpt, "config.json")) as f:
        q = json.load(f)["quantization_config"]
    grp = q["config_groups"]["group_0"]
    scheme = QuantizationScheme(targets=grp["targets"], weights=QuantizationArgs(**grp["weights"]))
    print(f"compressed-tensors {compressed_tensors.__version__}, torch {torch.__version__}; scheme weights: "
          f"{scheme.weights.num_bits} bits, group {scheme.weights.group_size}, symmetric {scheme.weights.symmetric}, "
          f"strategy {scheme.weights.strategy}")
    ck = G.Ckpt(a.ckpt, verified=a.verified)
    names = list(a.tensors)
    for L in a.all_experts:
        names += [f"model.layers.{L}.mlp.experts.{e}.{m}_proj" for e in range(G.NE) for m in ("gate", "up", "down")]
    tot = eq = inexact = 0
    bad = []
    t0 = time.time()
    for i, p in enumerate(names):
        packed, scale, zp, shape = ck.int4_raw(p)
        mine = G.int4_dequant(packed, scale, zp)
        lib = PackedQuantizationCompressor.decompress(
            {"weight_packed": packed, "weight_scale": scale, "weight_zero_point": zp, "weight_shape": shape},
            scheme)["weight"]
        assert lib.dtype == torch.bfloat16 and lib.shape == mine.shape, (lib.dtype, lib.shape, mine.shape)
        n = mine.numel()
        e = int((mine.view(torch.int16) == lib.view(torch.int16)).sum())
        # where (q - z) * s is not representable in bf16: the f32 product differs from its bf16 rounding
        N, K = mine.shape
        xq = (G.unpack_nibbles(packed).reshape(N, -1, G.GROUP) - 8).float()
        z = (G.unpack_nibbles(zp).permute(0, 2, 1).reshape(N, -1) - 8).float().unsqueeze(-1)
        exact = (xq - z) * scale.float().unsqueeze(-1)
        ni = int((exact.reshape(N, K) != mine.float()).sum())
        tot, eq, inexact = tot + n, eq + e, inexact + ni
        if e != n:
            bad.append(p)
        if i < len(a.tensors) or e != n:
            print(f"{p:48s} {list(mine.shape)} bit-equal {e}/{n}  rounded (not exact in bf16) {ni / n:.4f}")
        elif (i - len(a.tensors)) % 84 == 83:
            print(f"  ... {i + 1 - len(a.tensors)} expert matrices, {time.time() - t0:.0f} s", flush=True)
    print(f"TOTAL {len(names)} tensors, {tot} values: bit-equal {eq} ({eq / tot:.6%}); "
          f"values whose product needed the bf16 rounding: {inexact / tot:.4%}; {time.time() - t0:.0f} s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
