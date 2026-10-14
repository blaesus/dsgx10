#!/usr/bin/env python3
"""ctcheck: int4.py against compressed-tensors itself (the library that defines altar-1's weights).

    python -I ctcheck.py --ckpt DIR [--index J] [--manifest M --hashes H] --config config.json
                         [--tensors BASE ...] [--all-kv-b] [--exhaustive] [--file rankR.glw ...]

For each INT4 tensor BASE (e.g. model.layers.4.self_attn.kv_b_proj): the library's
PackedQuantizationCompressor.decompress, with the scheme built from the checkpoint's own quantization_config, gives
the BF16 weight; int4.dequant_bf16 must give the same bits on every element, and int4's value and zero-point
nibbles minus 8 must equal the library's unpack_from_int32. --exhaustive: the rounding of (q - z) * s for every
integer q - z in [-15, 15] against every finite bf16 scale, the library's _dequantize against int4's.
--all-kv-b: every INT4 layer's kv_b_proj. --file: also every INT4 layer's W_UK^T entry in the rank files (BF16,
dequantised by the converter) against the library's weight, transposed and cut as the plan says.
Needs torch and compressed-tensors (a venv of its own); the data are untrusted and only parsed."""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import int4  # noqa: E402
import sources  # noqa: E402


VERBOSE = [True]


def scheme_from(config):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    g = json.load(open(config))["quantization_config"]["config_groups"]["group_0"]
    return QuantizationScheme(targets=g["targets"], weights=QuantizationArgs(**g["weights"]))


def library(t, scheme):
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    N, K = int4.shape(t)
    sd = {"weight_packed": torch.from_numpy(t["packed"].copy()),
          "weight_scale": torch.from_numpy(t["scale"].copy()).view(torch.bfloat16),
          "weight_zero_point": torch.from_numpy(t["zp"].copy()),
          "weight_shape": torch.tensor([N, K], dtype=torch.int64)}
    out = PackedQuantizationCompressor.decompress(sd, scheme)
    w = out["weight"]
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (N, K):
        raise SystemExit(f"library weight {w.dtype} {tuple(w.shape)}")
    q = unpack_from_int32(sd["weight_packed"], 4, torch.Size([N, K]))
    z = unpack_from_int32(sd["weight_zero_point"], 4, torch.Size([N, K // 32]), packed_dim=0)
    return w.view(torch.int16).numpy().view(np.uint16), q.numpy(), z.numpy()


def check_tensor(t, scheme, name):
    lw, lq, lz = library(t, scheme)
    ours = int4.dequant_bf16(t)
    q_ok = np.array_equal(int4.nibbles(t["packed"]).astype(np.int16) - 8, lq.astype(np.int16))
    z_ok = np.array_equal(int4.zp_nibbles(t["zp"]).astype(np.int16) - 8, lz.astype(np.int16))
    nd = int((ours != lw).sum())
    inexact = int((int4.bf16_to_f32(ours).astype(np.float64) !=
                   (lq.astype(np.float64) - np.repeat(lz, 32, axis=1)) *
                   np.repeat(int4.bf16_to_f32(t["scale"]), 32, axis=1)).sum())
    if nd or not (q_ok and z_ok) or VERBOSE[0]:
        print(f"  {name}: {ours.size:,} weights, {nd} differ from the library; values {'=' if q_ok else 'DIFFER'}, "
          f"zero points {'=' if z_ok else 'DIFFER'}; {inexact:,} products rounded ({inexact / ours.size:.2%})",
              flush=True)
    return (nd == 0 and q_ok and z_ok), ours.size, lw


def exhaustive():
    from compressed_tensors.quantization.lifecycle.forward_helpers import _dequantize
    s = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    f = int4.bf16_to_f32(s)
    s = s[np.isfinite(f)]
    d = np.arange(-15, 16, dtype=np.int8)
    xq = torch.from_numpy(np.repeat(d, s.size).reshape(d.size, s.size))
    st = torch.from_numpy(np.tile(s, (d.size, 1))).view(torch.bfloat16)
    lib = _dequantize(xq, st, torch.zeros_like(xq)).view(torch.int16).numpy().view(np.uint16)
    ours = int4.f32_to_bf16_rne(d[:, None].astype(np.float32) * int4.bf16_to_f32(np.tile(s, (d.size, 1))))
    nd = int((lib != ours).sum())
    print(f"exhaustive: {lib.size:,} (q - z, s) pairs (every finite bf16 scale), {nd} differ")
    return nd == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index")
    ap.add_argument("--manifest")
    ap.add_argument("--hashes")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tensors", nargs="*", default=[])
    ap.add_argument("--all-kv-b", action="store_true")
    ap.add_argument("--exhaustive", action="store_true")
    ap.add_argument("--file", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import compressed_tensors
    print(f"compressed-tensors {compressed_tensors.__version__}, torch {torch.__version__}", flush=True)
    t0 = time.time()
    ok = True
    if a.exhaustive:
        ok &= exhaustive()
    ck = sources.Checkpoint(a.ckpt, a.index, a.manifest, a.hashes)
    scheme = scheme_from(a.config)
    names = list(a.tensors)
    if a.all_kv_b or a.file:
        VERBOSE[0] = False
        names += [f"model.layers.{L}.self_attn.kv_b_proj" for L in range(4, 77)]
    files = []
    if a.file:
        import glw
        for p in a.file:
            hd, ents, _ = glw.read_toc(p)
            files.append((p, hd["rank"], {e["name"]: e for e in ents}))
    n_tensors = n_weights = n_uk = 0
    for base in dict.fromkeys(names):
        t, _ = int4.parts(ck, base)
        good, n, lw = check_tensor(t, scheme, base)
        ok &= good
        n_tensors += 1
        n_weights += n
        L = int(base.split(".")[2])
        for p, r, ed in files:
            e = ed.get(f"layers.{L}.attn.w_uk.weight")
            if e is None:
                continue
            fd = os.open(p, os.O_RDONLY)
            got = np.frombuffer(os.pread(fd, e["bytes"], e["offset"]), np.uint16).reshape(e["shape"][:2])
            if e["layout"] == glw.BF16_TILE:  # kBF16T: [N/8][K/64][2][8 g][4 t][8] -> row-major
                N, K = e["shape"][:2]
                got = np.ascontiguousarray(got.reshape(N // 8, K // 64, 2, 8, 4, 8).transpose(0, 3, 1, 4, 2, 5)
                                           ).reshape(N, K)
            os.close(fd)
            want = np.concatenate([lw[448 * H:448 * H + 192].T for H in range(16 * r, 16 * r + 16)])
            same = np.array_equal(got, want)
            ok &= same
            n_uk += got.size
            if not same:
                print(f"  MISMATCH {p} layers.{L}.attn.w_uk.weight: {int((got != want).sum())} elements", flush=True)
    if files:
        print(f"W_UK^T entries: {n_uk:,} BF16 elements over {len(files)} files equal to the library's dequantisation"
              if ok else "W_UK^T entries: MISMATCH")
    print(f"{n_tensors} tensors, {n_weights:,} weights: {'all equal to compressed-tensors' if ok else 'FAIL'} "
          f"({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
