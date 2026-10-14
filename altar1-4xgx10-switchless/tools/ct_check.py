#!/usr/bin/env python3
"""A0: the engine's INT4 unpacking and dequantisation against compressed-tensors itself, on real tensors.

For each INT4 module (prefix ending in _proj) the checkpoint's weight_packed, weight_scale, weight_zero_point and
weight_shape go through the library's own PackedQuantizationCompressor.decompress (unpack_from_int32 of values and
zero points, then forward.py dequantize with the checkpoint's QuantizationArgs) and through engine/weights/
int4_pack.py (canon_int4, value_bits, rank_slice, pack, untile); every element must agree bit for bit:
  - the unpacked values and zero points (the library's signed q, z against our stored nibbles minus 8);
  - the dequantised BF16 weight (the library's against value_bits, which is bf16((u - u_z) * s) rounded once);
  - each rank's slice through the kernels' blocks (pack, then untile and read_packed) against the library's rows or
    columns of the same weight.
It also counts how often the rounding matters (the exact product not representable in BF16) and how far two other
rounding rules would be from the library (f32 kept, and bf16(u s) - bf16(u_z s)), so the rounding point is settled
by the data, not assumed. A BF16 module (prefix with .weight) is checked through pack / untile / read_packed.

  python -I tools/ct_check.py --ct-path SITE --ckpt DIR --index model.safetensors.index.json --config config.json \
      model.layers.4.mlp.experts.0.gate_proj ... [--out results.tsv]
torch from the running interpreter; compressed_tensors from SITE (a site-packages holding it and loguru).
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "engine", "weights"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--samples", type=int, default=20000, help="per-element reads of each rank's packed slice")
    ap.add_argument("modules", nargs="+")
    a = ap.parse_args()
    sys.path.append(a.ct_path)
    import numpy as np
    import torch
    import compressed_tensors
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    import int4_pack as P

    torch.set_num_threads(8)
    cfg = json.load(open(a.config))["quantization_config"]
    g0 = cfg["config_groups"]["group_0"]
    scheme = QuantizationScheme(targets=g0["targets"], weights=QuantizationArgs(**g0["weights"]))
    S = P.Shards(a.ckpt, a.index)
    out = open(a.out, "a") if a.out else None

    def emit(*f):
        line = "\t".join(str(x) for x in f)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()

    emit("RUN", time.strftime("%F %T %z"), f"compressed-tensors {compressed_tensors.__version__}", f"torch {torch.__version__}",
         f"checkpoint written by {cfg.get('version')}")
    rng = np.random.default_rng(1)
    bad_all = 0
    for mod in a.modules:
        t0 = time.time()
        if S.has(mod + ".weight_packed"):
            wp = S.get(mod + ".weight_packed")
            sc = S.get(mod + ".weight_scale")
            zp = S.get(mod + ".weight_zero_point")
            shp = tuple(int(x) for x in S.get(mod + ".weight_shape"))
            sd = {"weight_packed": torch.from_numpy(wp.copy()), "weight_scale": torch.from_numpy(sc.copy()).view(torch.bfloat16),
                  "weight_zero_point": torch.from_numpy(zp.copy()), "weight_shape": torch.tensor(shp)}
            lib = PackedQuantizationCompressor.decompress(sd, scheme)
            W = lib["weight"]
            assert W.dtype == torch.bfloat16 and tuple(W.shape) == shp, (W.dtype, W.shape)
            lib_bits = W.view(torch.int16).numpy().view(np.uint16)
            q_lib = unpack_from_int32(sd["weight_packed"], 4, torch.Size(shp)).numpy()
            z_lib = lib["weight_zero_point"].numpy()  # the library's unpacked zero points [N][K/32]
            c = P.canon_int4(wp, sc, zp)
            N, K = c["N"], c["K"]
            u = P.nibbles(c["wp"]).astype(np.int16)
            uz = P.zp_nibbles(c["zp"], N).astype(np.int16)
            bq = int(np.count_nonzero(u - 8 != q_lib))
            bz = int(np.count_nonzero(uz - 8 != z_lib))
            ours = P.value_bits(c)
            bw = int(np.count_nonzero(ours != lib_bits))
            # how much the rounding point matters
            d = (u - np.repeat(uz, 32, axis=1)).astype(np.float32)
            sf = np.repeat(P.bits_to_f32(c["s"]), 32, axis=1)
            exact = d * sf
            inexact = int(np.count_nonzero(P.bits_to_f32(ours) != exact))
            alt = P.bf16_round_bits(P.bits_to_f32(P.bf16_round_bits(u.astype(np.float32) * sf)) -
                                    P.bits_to_f32(P.bf16_round_bits(np.repeat(uz, 32, axis=1).astype(np.float32) * sf)))
            balt = int(np.count_nonzero(alt != lib_bits))
            sfin = bool(np.all(np.isfinite(P.bits_to_f32(c["s"]))) and np.all(P.bits_to_f32(c["s"]) > 0))
            # each rank's slice through the kernels' blocks
            m = 2 if mod.endswith("down_proj") else 0
            bslice = 0
            nread = 0
            if (N, K) in ((P.INTER, P.KH), (P.KH, P.INTER)):
                for r in range(P.TP):
                    sl = P.rank_slice(c, m, r)
                    pk = P.pack(sl)
                    w2, s2, z2 = P.untile(pk, P.FMT_INT4, sl["N"], sl["K"])
                    bslice += int(np.count_nonzero(w2 != sl["wp"]) + np.count_nonzero(s2 != sl["s"]) + np.count_nonzero(z2 != sl["zp"]))
                    ref = lib_bits[512 * r:512 * (r + 1)] if m < 2 else lib_bits[:, 512 * r:512 * (r + 1)]
                    bslice += int(np.count_nonzero(P.value_bits(sl) != ref))
                    for _ in range(a.samples // P.TP):
                        n, k = int(rng.integers(sl["N"])), int(rng.integers(sl["K"]))
                        bslice += P.read_packed(pk, P.FMT_INT4, sl["K"], n, k) != int(ref[n, k])
                        nread += 1
            bad = bq + bz + bw + bslice
            bad_all += bad
            emit("INT4", mod, f"{N}x{K}", f"values {N * K}", f"q mismatches {bq}", f"z mismatches {bz} of {N * K // 32}",
                 f"weight mismatches {bw}", f"inexact in bf16 {inexact} ({100.0 * inexact / (N * K):.2f} %)",
                 f"bf16(u s) - bf16(uz s) mismatches {balt}", f"scales finite > 0 {sfin}",
                 f"zp nibble range {int(uz.min())}-{int(uz.max())}", f"rank slices: packed/untiled/value/read mismatches {bslice} ({nread} reads)",
                 f"{time.time() - t0:.1f} s", "OK" if bad == 0 else "FAIL")
        else:
            w = S.get(mod + ".weight")
            c = P.canon_bf16(w)
            N, K = c["N"], c["K"]
            m = 2 if mod.endswith("down_proj") else 0
            bad = 0
            nread = 0
            for r in range(P.TP):
                sl = P.rank_slice(c, m, r)
                pk = P.pack(sl)
                bad += int(np.count_nonzero(P.untile(pk, P.FMT_BF16, sl["N"], sl["K"]) != sl["w"]))
                for _ in range(a.samples // P.TP):
                    n, k = int(rng.integers(sl["N"])), int(rng.integers(sl["K"]))
                    bad += P.read_packed(pk, P.FMT_BF16, sl["K"], n, k) != int(sl["w"][n, k])
                    nread += 1
            bad_all += bad
            emit("BF16", mod, f"{N}x{K}", f"values {N * K}", f"rank slices: untiled/read mismatches {bad} ({nread} reads)",
                 f"{time.time() - t0:.1f} s", "OK" if bad == 0 else "FAIL")
    emit("END", "all OK" if bad_all == 0 else f"{bad_all} mismatches")
    sys.exit(1 if bad_all else 0)


if __name__ == "__main__":
    main()
