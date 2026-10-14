"""Grid formats for the routed experts: NVFP4 as NVIDIA cut it, an E2M1-subset 3-bit code, int3 with group
scales. Every function takes a torch-layout weight (out, in) as float32 and groups along `in` (the GEMM's K).
Each quantiser returns (packed: dict of numpy arrays, the stored bytes) and each dequantiser goes from those
bytes back to float32, so every metric is measured through the real encoding. All ops are elementwise or
fixed-shape reductions on the CPU; the same input gives the same bytes.

nvfp4_rtn (4.5 bpw + 32 bits a tensor), NVIDIA's recipe (docs/quantisation.md § 3.0):
    s2 = f32(amax) * f32(1/2688)                          (weight_scale_2; a multiply by the f32 reciprocal, which
                                                           differs from f32 division in the last bit on ~1/3 of
                                                           tensors; amax over gate and up together for those two)
    sc = e4m3_rne(amax16 / f32(6 * s2))                     (weight_scale, one per 16 along in; amax16 / 6 / s2
                                                           rounds differently on ~1.6 % of tensors)
    v  = W / (f32(sc) * s2), 0/0 -> 0;  code = E2M1 round-to-nearest, ties to the even code, |v| > 6 -> 6
    nibble = sign << 3 | magnitude index into {0, .5, 1, 1.5, 2, 3, 4, 6}, sign = (v < 0); byte = lo | hi << 4,
    lo nibble = even column.

levelset3 (the E2M1-subset 3-bit code; 3.5 bpw at group 16 with an e4m3 scale):
    a subset is 4 magnitudes m0 < m1 < m2 < m3 of the E2M1 grid (m0 may be 0); a 3-bit code c = s << 2 | j
    means (-1)^s * m_j. One or two subsets per tensor; with two, the e4m3 scale's sign bit selects the subset of
    each group (the scale itself is |scale|), so the second subset costs no bits.
    Scales: s2 = amax(|W|) / (t * 448), t the smallest top magnitude among the subsets (6 for every subset
    used here: the NVFP4 rule); for every group, subset and multiplier
    m in MULTS the candidate scale is e4m3_rne(amax_g / top_subset * m / s2) * s2, values go to the nearest
    level (ties to the lower level), and the (subset, m) pair with the smallest squared error wins, earliest in
    (subset, MULTS) order on ties.
    Packing: per group of 16, 48 bits little-endian, value j at bits 3j..3j+2 (6 bytes), then the scales as a
    separate e4m3 array (one byte a group), as NVFP4 keeps them.
    Tensor-core path: c -> the E2M1 nibble of (-1)^s * m_j is a per-subset 8-entry table, so the codes feed
    `mxf4nvf4` as NVFP4 with the same scale.

int3 (signed {-4..3} with group scales; round to nearest even, clamp):
    e4m3 scales: s2 = amax(|W|) / (4 * 448); candidate scale e4m3_rne(amax_g / t * m / s2) * s2 for t in {4, 3}
    and m in MULTS; bf16 scales: bf16(amax_g / t * m). Best of the 14 by squared error. Code q + 4, packed
    as levelset3. (-4..3 are E2M1 values, so the tensor-core path is native too.)
"""
import numpy as np
import torch

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
MULTS = (1.0, .95, .9, .85, .8, .75, .7)


def e4m3(x):
    return x.to(torch.float8_e4m3fn)


# ---- NVFP4 ------------------------------------------------------------------------------------------

def e2m1_rne_index(a):
    """Magnitude index of round-to-nearest-even onto {0,.5,1,1.5,2,3,4,6} (a >= 0, saturating)."""
    q = torch.zeros_like(a, dtype=torch.uint8)
    for t in (.25, 1.25, 2.5, 5.0):
        q += (a > t).to(torch.uint8)
    for t in (.75, 1.75, 3.5):
        q += (a >= t).to(torch.uint8)
    return q


def nvfp4_rtn(w, amax=None):
    """amax: the tensor maximum to use (gate and up share max(amax_gate, amax_up) in NVIDIA's checkpoint)."""
    R, C = w.shape
    if amax is None:
        amax = w.abs().max()
    s2 = amax * torch.tensor(1 / 2688.0, dtype=torch.float32)
    blk = w.reshape(R, C // 16, 16)
    sc8 = e4m3(blk.abs().amax(-1) / (6 * s2))
    v = torch.nan_to_num(blk / (sc8.float() * s2).unsqueeze(-1), nan=0.0)
    nib = e2m1_rne_index(v.abs()) | ((v < 0).to(torch.uint8) << 3)
    nib = nib.reshape(R, C // 2, 2)
    packed = nib[..., 0] | (nib[..., 1] << 4)
    return {"weight": packed.numpy(), "weight_scale": sc8.view(torch.uint8).numpy(),
            "weight_scale_2": s2.reshape(()).numpy()}


def nvfp4_dequant(p):
    b = torch.from_numpy(p["weight"])
    nib = torch.stack([b & 15, b >> 4], -1).reshape(b.shape[0], -1).long()
    val = torch.from_numpy(E2M1)[nib & 7] * (1 - 2 * (nib >> 3)).float()
    sc = torch.from_numpy(p["weight_scale"]).view(torch.float8_e4m3fn).float()
    return val * sc.repeat_interleave(16, 1) * float(p["weight_scale_2"])


# ---- 3-bit level sets ---------------------------------------------------------------------------------

def _nearest(x, levels):
    """Index into ascending `levels` of the nearest level, ties to the lower one."""
    mids = (levels[1:] + levels[:-1]) / 2
    return torch.bucketize(x, mids, right=False)


def _search(blk, cands):
    """cands: list of (scale tensor [G,1] f32, levels tensor asc, tag). Returns best (idx, scale, tag) per group."""
    best_err = best_idx = best_sc = best_tag = None
    for sc, lv, tag in cands:
        safe = torch.where(sc == 0, torch.ones_like(sc), sc)
        idx = _nearest(blk / safe, lv)
        deq = lv[idx] * sc
        err = ((deq - blk) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_idx, best_sc = err, idx, sc
            best_tag = torch.full_like(err, tag, dtype=torch.int8)
        else:
            t = err < best_err
            best_err = torch.where(t, err, best_err)
            best_idx = torch.where(t, idx, best_idx)
            best_sc = torch.where(t, sc, best_sc)
            best_tag = torch.where(t, torch.full_like(best_tag, tag), best_tag)
    return best_idx, best_sc, best_tag


def pack3(codes):
    """codes (R, n) uint8 in 0..7 -> (R, n/16, 6) uint8: each run of 16 values is a little-endian 48-bit word,
    value j at bits 3j..3j+2."""
    c = codes.reshape(codes.shape[0], -1, 16).astype(np.uint64)
    word = np.zeros(c.shape[:-1], np.uint64)
    for j in range(16):
        word |= c[..., j] << np.uint64(3 * j)
    return np.stack([((word >> np.uint64(8 * i)) & np.uint64(255)).astype(np.uint8) for i in range(6)], -1)


def unpack3(b):
    word = np.zeros(b.shape[:-1], np.uint64)
    for i in range(6):
        word |= b[..., i].astype(np.uint64) << np.uint64(8 * i)
    return np.stack([((word >> np.uint64(3 * j)) & np.uint64(7)).astype(np.uint8) for j in range(16)], -1)


def levelset3(w, subsets, g=16):
    """subsets: list of 1 or 2 tuples of 4 magnitudes. Returns packed dict."""
    assert len(subsets) in (1, 2)
    R, C = w.shape
    blk = w.reshape(R, C // g, g)
    amax = blk.abs().amax(-1, keepdim=True)
    top = min(max(m) for m in subsets)
    s2 = w.abs().max() / torch.tensor(top * 448.0, dtype=torch.float32)
    cands = []
    for si, mags in enumerate(subsets):
        m = torch.tensor(sorted(mags), dtype=torch.float32)
        lv = torch.cat([-m.flip(0), m])            # ascending, 8 entries (a 0 magnitude appears twice)
        for mult in MULTS:
            sc = e4m3(amax / m.max() * mult / s2).float() * s2
            cands.append((sc, lv, si))
    idx, sc, tag = _search(blk, cands)
    # idx into [-m3,-m2,-m1,-m0, m0,m1,m2,m3] -> code s<<2|j
    idx = idx.to(torch.uint8)
    code = torch.where(idx >= 4, idx - 4, (3 - idx) | 4).to(torch.uint8)
    sc8 = e4m3(sc / s2).view(torch.uint8) | (tag.to(torch.uint8) << 7)
    return {"codes": pack3(code.reshape(R, C).numpy()), "scale": sc8.reshape(R, C // g).numpy(), "scale_2": s2.reshape(()).numpy(),
            "subsets": np.array([sorted(s) for s in subsets], np.float32), "g": np.array(g)}


def levelset3_dequant(p):
    g = int(p["g"])
    code = torch.from_numpy(unpack3(p["codes"])).long()
    code = code.reshape(code.shape[0], -1, g)
    raw = torch.from_numpy(p["scale"])
    sel = (raw >> 7).long()
    sc = (raw & 127).view(torch.float8_e4m3fn).float() * float(p["scale_2"])
    mags = torch.from_numpy(p["subsets"])                     # [n_subsets, 4]
    j, s = code & 3, code >> 2
    val = mags[sel.unsqueeze(-1).expand_as(j), j] * (1 - 2 * s).float()
    R = val.shape[0]
    return (val * sc.unsqueeze(-1)).reshape(R, -1)


def int3(w, g=16, scale="e4m3"):
    R, C = w.shape
    blk = w.reshape(R, C // g, g)
    amax = blk.abs().amax(-1, keepdim=True)
    lv = torch.arange(-4, 4, dtype=torch.float32)
    s2 = w.abs().max() / torch.tensor(4 * 448.0, dtype=torch.float32)
    cands = []
    for t in (4.0, 3.0):
        for mult in MULTS:
            if scale == "e4m3":
                sc = e4m3(amax / t * mult / s2).float() * s2
            else:
                sc = (amax / t * mult).to(torch.bfloat16).float()
            cands.append((sc, lv, 0))
    idx, sc, _ = _search(blk, cands)
    out = {"codes": pack3(idx.to(torch.uint8).reshape(R, C).numpy()), "g": np.array(g)}
    if scale == "e4m3":
        out["scale"] = e4m3(sc / s2).view(torch.uint8).reshape(R, C // g).numpy()
        out["scale_2"] = s2.reshape(()).numpy()
    else:
        out["scale_bf16"] = sc.to(torch.bfloat16).view(torch.int16).reshape(R, C // g).numpy()
    return out


def int3_dequant(p):
    g = int(p["g"])
    q = torch.from_numpy(unpack3(p["codes"])).float() - 4
    q = q.reshape(q.shape[0], -1, g)
    if "scale" in p:
        sc = torch.from_numpy(p["scale"]).view(torch.float8_e4m3fn).float() * float(p["scale_2"])
    else:
        sc = torch.from_numpy(p["scale_bf16"]).view(torch.bfloat16).float()
    return (q * sc.unsqueeze(-1)).reshape(q.shape[0], -1)


def bpw(packed, numel):
    """Bits a weight of everything stored except the format's constant tables."""
    skip = {"subsets", "g"}
    return 8 * sum(v.nbytes for k, v in packed.items() if k not in skip) / numel
