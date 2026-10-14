#!/usr/bin/env python3
"""The INT4 contract checked on real altar-1 tensors (milestone A0's CPU half for the attention projections):

1. compressed-tensors' own `PackedQuantizationCompressor.decompress` (the library's dequantisation, as transformers
   applies it to q_b/kv_b/o_proj) against pack.py's `int4_dequant`, bf16_rne((nib - znib) * s): every value's bits;
2. what other rounding points would give (f32 weight, q*s - z*s, fma), to show the check has teeth;
3. pack.py's engine tile round trip (int4_tiles -> int4_untile = the checkpoint's packed, scale and zero-point
   words, byte for byte) and the tiles' dequantisation = 1;
4. exhaustively: torch's bf16 (q - z) * s for every bf16 scale pattern and every q - z in [-15, 15] = the f32
   product rounded to nearest even (the host emulation the GPU tests compare __hmul2 against).

    python3 -I int4_check.py --ckpt DIR [--index JSON] [--layers 4,10] [--projs q_b_proj,kv_b_proj,o_proj]

DIR holds the safetensors shards and model.safetensors.index.json (read-only; shards should be verified against the
manifest's sha256 first). Exit 0 when everything is bit-equal."""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pack  # noqa: E402


def load(ckpt, index, name):
    from safetensors import safe_open
    with safe_open(os.path.join(ckpt, index[name]), framework="pt") as f:
        return f.get_tensor(name)


def u16(t):
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def exhaustive():
    s = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    sf = s.float()
    ok = torch.isfinite(sf)
    s, sf = s[ok], sf[ok]
    bad = 0
    for d in range(-15, 16):
        lib = (torch.tensor(float(d)).to(torch.bfloat16) * s)  # bf16 x bf16 in torch, as _dequantize
        ref = pack.f32_to_bf16(np.float32(d) * sf.numpy())
        bad += int((u16(lib) != ref).sum())
    return int(ok.sum()) * 31, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", help="model.safetensors.index.json (default: DIR/model.safetensors.index.json)")
    ap.add_argument("--layers", default="4,10")
    ap.add_argument("--projs", default="q_b_proj,kv_b_proj,o_proj")
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count()))
    import compressed_tensors
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))["quantization_config"]["config_groups"]["group_0"]
    wargs = {k: v for k, v in cfg["weights"].items() if k in ("num_bits", "type", "symmetric", "strategy", "group_size")}
    scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(**wargs))
    print(f"compressed-tensors {compressed_tensors.__version__}, torch {torch.__version__}, scheme weights {wargs}")
    index = json.load(open(a.index or os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    fails = 0
    t0 = time.time()
    for l in [int(x) for x in a.layers.split(",")]:
        for p in a.projs.split(","):
            base = f"model.layers.{l}.self_attn.{p}."
            sd = {k: load(a.ckpt, index, base + k) for k in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")}
            N, K = [int(v) for v in sd["weight_shape"]]
            lib = PackedQuantizationCompressor.decompress(dict(sd), scheme)["weight"]
            assert lib.dtype == torch.bfloat16 and tuple(lib.shape) == (N, K), (lib.dtype, lib.shape)
            libb = u16(lib)
            packed, scale, zp = sd["weight_packed"].numpy(), u16(sd["weight_scale"]), sd["weight_zero_point"].numpy()
            mine = pack.int4_dequant(packed, scale, zp)
            diff = int((mine != libb).sum())
            fails += diff != 0
            # other rounding points: the f32 weight, q*s - z*s in bf16, fma(q, s, -z*s) in f32 then bf16
            q = pack.int4_nibbles(packed).astype(np.float32).reshape(N, K // 32, 32)
            z = pack.int4_zp_nibbles(zp)[:N].astype(np.float32)[:, :, None]
            s = pack.bf16_to_f32(scale)[:, :, None]
            exact = ((q - z) * s).reshape(N, K)
            inexact = int((pack.bf16_to_f32(libb) != exact).sum())
            alt1 = pack.f32_to_bf16(pack.bf16_to_f32(pack.f32_to_bf16(q * s)) - pack.bf16_to_f32(pack.f32_to_bf16(z * s))).reshape(N, K)
            alt2 = pack.f32_to_bf16(np.float32(q * s - pack.bf16_to_f32(pack.f32_to_bf16(z * s)))).reshape(N, K)
            d1, d2 = int((alt1 != libb).sum()), int((alt2 != libb).sum())
            # the engine tile round trip and its dequantisation
            t = pack.int4_tiles(packed, scale, zp)
            pp, ss, zz = pack.int4_untile(t, N, K)
            rt = np.array_equal(pp, packed.view(np.uint32)) and np.array_equal(ss, scale) and np.array_equal(zz, zp.view(np.uint32))
            fails += not rt
            fails += t.size != N // 8 * K // 512 * pack.INT4_BLK
            nibs = np.bincount(pack.int4_nibbles(packed).ravel(), minlength=16)
            zn = pack.int4_zp_nibbles(zp).ravel()
            print(f"L{l} {p} [{N}, {K}]: library vs bf16_rne((nib - znib) * s): {diff} of {N * K} values differ; "
                  f"product inexact in bf16 (rounded) for {inexact} ({inexact / (N * K):.1%}); bf16(q s) - bf16(z s) "
                  f"would differ in {d1}, f32 q s - bf16(z s) rounded once in {d2}; tile round trip "
                  f"{'byte-equal' if rt else 'DIFFERS'} ({t.size} B); value nibbles min {int(np.nonzero(nibs)[0].min())} "
                  f"max {int(np.nonzero(nibs)[0].max())}, zero-point nibbles {int(zn.min())}..{int(zn.max())} mean "
                  f"{zn.mean():.2f}, scales {float(s.min()):.3g}..{float(s.max()):.3g} [{time.time() - t0:.0f}s]", flush=True)
    n, bad = exhaustive()
    fails += bad != 0
    print(f"exhaustive: torch bf16 (q - z) * s = f32 product rounded to nearest even: {bad} of {n} differ")
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
