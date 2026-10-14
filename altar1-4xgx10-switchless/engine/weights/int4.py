"""compressed-tensors `pack-quantized` INT4 (4 bits, group 32 along K, asymmetric, BF16 scale) in the checkpoint's
own form, numpy only (docs/model.md § 4):

    weight_packed      I32  [N][K/8]    nibble i of word j is k = 8j + i (bits 4i), stored q + 8
    weight_scale       BF16 [N][K/32]   (uint16 bit patterns here)
    weight_zero_point  I32  [N/8][K/32] nibble i of word m is row n = 8m + i, stored z + 8
    weight_shape       I64  [2]         [N, K]: metadata, checked, not stored in the rank files

An INT4 tensor here is a dict {"packed", "scale", "zp"} of those three arrays, whole or a rank's group-aligned cut
(rows in multiples of 8, columns in multiples of 32): the rank files carry these bytes re-tiled, and the verifier
un-tiles them back to exactly these arrays. `dequant_bf16` is the numerics contract's weight,
W = bf16_rne((q - z) * s), compressed-tensors' `dequantize` in the scale's dtype; checked against the library on
real tensors by `ctcheck.py`."""
import numpy as np

G = 32
SH8 = (4 * np.arange(8, dtype=np.uint32))


def parts(ck, base):
    """The three parts of checkpoint tensor `base` (e.g. model.layers.4.self_attn.o_proj) and their SHA-256s,
    with weight_shape checked against the parts' shapes."""
    p, sp = ck.get(base + ".weight_packed")
    s, ss = ck.get(base + ".weight_scale")
    z, zs = ck.get(base + ".weight_zero_point")
    shp, hs = ck.get(base + ".weight_shape")
    check(dict(packed=p, scale=s, zp=z), base, tuple(int(x) for x in shp))
    return dict(packed=p, scale=s, zp=z), {base + ".weight_packed": sp, base + ".weight_scale": ss,
                                          base + ".weight_zero_point": zs, base + ".weight_shape": hs}


def check(t, name, shape=None):
    N, K8 = t["packed"].shape
    K = 8 * K8
    if shape is not None and shape != (N, K):
        raise SystemExit(f"{name}: weight_shape {shape}, packed implies {(N, K)}")
    if t["packed"].dtype != np.int32 or t["zp"].dtype != np.int32 or t["scale"].dtype != np.uint16:
        raise SystemExit(f"{name}: dtypes {t['packed'].dtype} {t['scale'].dtype} {t['zp'].dtype}")
    if t["scale"].shape != (N, K // G) or t["zp"].shape != (N // 8, K // G) or N % 8 or K % G:
        raise SystemExit(f"{name}: shapes {t['packed'].shape} {t['scale'].shape} {t['zp'].shape}")
    return N, K


def shape(t):
    return t["packed"].shape[0], 8 * t["packed"].shape[1]


def rows(t, a, b):
    """Output rows [a, b) (multiples of 8): packed and scale rows, zero-point rows [a/8, b/8)."""
    assert a % 8 == 0 and b % 8 == 0, (a, b)
    return dict(packed=np.ascontiguousarray(t["packed"][a:b]), scale=np.ascontiguousarray(t["scale"][a:b]),
                zp=np.ascontiguousarray(t["zp"][a // 8:b // 8]))


def cols(t, a, b):
    """Input columns [a, b) (multiples of 32): packed words [a/8, b/8), scale and zero-point groups [a/32, b/32)."""
    assert a % G == 0 and b % G == 0, (a, b)
    return dict(packed=np.ascontiguousarray(t["packed"][:, a // 8:b // 8]),
                scale=np.ascontiguousarray(t["scale"][:, a // G:b // G]),
                zp=np.ascontiguousarray(t["zp"][:, a // G:b // G]))


def cat_rows(ts):
    return dict(packed=np.concatenate([t["packed"] for t in ts]), scale=np.concatenate([t["scale"] for t in ts]),
                zp=np.concatenate([t["zp"] for t in ts]))


def equal(a, b):
    return all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k])
               for k in ("packed", "scale", "zp"))


def nibbles(packed):
    """Stored value nibbles (q + 8) [N][K] uint8."""
    N, K8 = packed.shape
    u = packed.view(np.uint32)
    return ((u[:, :, None] >> SH8) & 0xF).astype(np.uint8).reshape(N, 8 * K8)


def zp_nibbles(zp):
    """Stored zero-point nibbles (z + 8) [N][K/32] uint8."""
    M, Gn = zp.shape
    u = zp.view(np.uint32)
    return ((u[:, None, :] >> SH8[None, :, None]) & 0xF).astype(np.uint8).reshape(8 * M, Gn)


def pack_nibbles(nib):
    """Inverse of `nibbles`: [N][K] values 0..15 -> I32 [N][K/8]."""
    N, K = nib.shape
    w = (nib.reshape(N, K // 8, 8).astype(np.uint32) << SH8).sum(axis=2, dtype=np.uint32)
    return w.view(np.int32)


def pack_zp_nibbles(znib):
    """Inverse of `zp_nibbles`: [N][G] values 0..15 -> I32 [N/8][G]."""
    N, Gn = znib.shape
    w = (znib.reshape(N // 8, 8, Gn).astype(np.uint32) << SH8[None, :, None]).sum(axis=1, dtype=np.uint32)
    return w.view(np.int32)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(x):
    """Round f32 to bf16, nearest-even (as torch's .to(bfloat16) and its bf16 arithmetic); no NaN expected."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    if np.isnan(x).any():
        raise SystemExit("NaN in a dequantisation")
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def dequant_bf16(t):
    """The contract's weight: bf16_rne((nib - znib) * s) [N][K] as uint16 bit patterns. (nib - znib) is an exact
    integer in [-15, 15] (the +8 offsets cancel) and its product with a bf16 scale has at most 12 significant bits,
    so the f32 product is exact and the one rounding is the final one, as in compressed-tensors' bf16 multiply."""
    N, K = shape(t)
    d = nibbles(t["packed"]).astype(np.int16) - np.repeat(zp_nibbles(t["zp"]).astype(np.int16), G, axis=1)
    s = np.repeat(bf16_to_f32(t["scale"]), G, axis=1)
    return f32_to_bf16_rne(d.astype(np.float32) * s)


def selftest(seed=1):
    rng = np.random.default_rng(seed)
    N, K = 64, 256
    nib = rng.integers(0, 16, (N, K), dtype=np.uint8)
    zn = rng.integers(0, 16, (N, K // G), dtype=np.uint8)
    t = dict(packed=pack_nibbles(nib), scale=rng.integers(0x3B00, 0x3C80, (N, K // G), dtype=np.uint16),
             zp=pack_zp_nibbles(zn))
    assert np.array_equal(nibbles(t["packed"]), nib) and np.array_equal(zp_nibbles(t["zp"]), zn)
    # word j bits 4i <-> k = 8j + i
    assert ((t["packed"][3, 5].view(np.uint32) >> 8) & 15) == nib[3, 42]
    assert ((t["zp"][2, 7].view(np.uint32) >> 12) & 15) == zn[19, 7]
    w = bf16_to_f32(dequant_bf16(t))
    ref = (nib.astype(np.float64) - np.repeat(zn, G, axis=1)) * np.repeat(bf16_to_f32(t["scale"]), G, axis=1)
    # bf16 of the exact product: within half an ulp, ties to even
    assert np.all(np.abs(w - ref) <= np.abs(ref) * 2.0 ** -8 + 1e-30)
    assert equal(rows(t, 8, 24), dict(packed=t["packed"][8:24], scale=t["scale"][8:24], zp=t["zp"][1:3]))
    c = cols(t, 64, 128)
    assert np.array_equal(nibbles(c["packed"]), nib[:, 64:128]) and np.array_equal(zp_nibbles(c["zp"]), zn[:, 2:4])
    x = np.array([1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, 1.0 + 2 ** -9], np.float32)  # ties to even, below half
    assert list(f32_to_bf16_rne(x)) == [0x3F80, 0x3F82, 0x3F80], f32_to_bf16_rne(x)
    print("int4 selftest OK")


if __name__ == "__main__":
    selftest()
