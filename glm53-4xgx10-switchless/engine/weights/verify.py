#!/usr/bin/env python3
"""glw verify: check per-rank .glw files, three ways.

    python -I verify.py FILE [FILE ...] --crc                         structure, TOC CRC, every tensor's CRC, padding,
                                                                       and the manifest's per-tensor and file SHA-256
    python -I verify.py FILE [...] --ckpt NV --cache DIR --fresh      every byte against a fresh conversion (the
                                                                       converter's own code, run again)
    python -I verify.py FILE [...] --ckpt NV --cache DIR --indep      every byte against the master by an independent
                                                                       description (below); the strongest check

--indep does not use pack.py, quant.py or the plan. From the cut each tensor's name describes (docs/attention.md
§ 1-2, written out again here as source row/column pieces), it builds the natural matrix of master values each
FP8 tensor should hold, the scale the FP8 rule gives each source block (f32(amax) * f32(1/448), in numpy), and
checks (1) the file's scale table, read at [n / rs][k / ks] (gate/up pairs: [n / 2 / rs][n % 2][k / ks]) with
the rs, ks the TOC states, equals the block scale of every element's source block, and (2) every code un-tiled
from the file with the layout formula of glw.h equals e4m3_rne(master / scale). BF16/F32 tensors must equal their
source rows byte for byte; the RoPE table must equal rope_table.py's. It also checks that the files hold exactly
the entries the description expects, that replicated tensors agree across the files given, and that split ones
are cut into disjoint, complete pieces (the cross-rank check, when all four ranks are given).
"""
import argparse, hashlib, json, os, re, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import glw  # noqa: E402

GIB = 1 << 30


def read_entry(fd, e, pad=False):
    n = glw.align_up(e["bytes"]) if pad else e["bytes"]
    b = os.pread(fd, n, e["offset"])
    if len(b) != n:
        raise SystemExit(f"{e['name']}: short read")
    return b


def check_crc(path):
    hd, ents, raw = glw.read_toc(path)
    st = os.stat(path)
    bad = []
    if hd["version"] != glw.VERSION or hd["header_bytes"] != glw.HDR_BYTES or hd["entry_bytes"] != glw.ENT_BYTES or \
            hd["alignment"] != glw.ALIGN or hd["file_bytes"] != st.st_size or \
            hd["data_offset"] + hd["data_bytes"] != st.st_size:
        bad.append("header geometry")
    if glw.crc32c(raw) != hd["toc_crc"]:
        bad.append("TOC CRC")
    man_p = path + ".manifest.json"
    man = json.load(open(man_p)) if os.path.exists(man_p) else None
    msha = {t["name"]: t["sha256"] for t in man["tensors"]} if man else {}
    fd = os.open(path, os.O_RDONLY)
    expect = hd["data_offset"]
    names = set()
    try:
        for e in ents:
            if e["name"] in names:
                bad.append(f"{e['name']} twice")
            names.add(e["name"])
            if e["kind"] == glw.RESERVED:
                if e["bytes"] or e["offset"] or not e["flags"] & glw.RESERVED_FLAG:
                    bad.append(f"{e['name']}: a reserved entry with data")
                continue
            if e["offset"] != expect:
                bad.append(f"{e['name']}: offset {e['offset']}, expected {expect}")
            if int(np.prod(e["shape"])) * glw.DTYPE_BYTES[e["dtype"]] != e["bytes"]:
                bad.append(f"{e['name']}: shape and size disagree")
            b = read_entry(fd, e, pad=True)
            os.posix_fadvise(fd, e["offset"], len(b), os.POSIX_FADV_DONTNEED)
            if glw.crc32c(b[:e["bytes"]]) != e["crc"]:
                bad.append(f"{e['name']}: CRC")
            if any(b[e["bytes"]:]):
                bad.append(f"{e['name']}: padding not zero")
            if msha and hashlib.sha256(b[:e["bytes"]]).hexdigest() != msha.get(e["name"]):
                bad.append(f"{e['name']}: SHA-256 differs from the manifest")
            expect += glw.align_up(e["bytes"])
    finally:
        os.close(fd)
    if expect != st.st_size:
        bad.append(f"tensors end at {expect}, the file at {st.st_size}")
    fsha = None
    if man:
        import convert
        fsha = convert.file_sha256(path)
        if fsha != man["file_sha256"]:
            bad.append("file SHA-256 differs from the manifest")
    print(f"{path}: rank {hd['rank']}/{hd['world']}, {len(ents)} entries, {st.st_size / GIB:.3f} GiB, created "
          f"{hd['created']}, tool {hd['tool']}; CRCs, padding{', manifest SHA-256s' if man else ''}: "
          f"{'OK' if not bad else str(len(bad)) + ' BAD'}" + (f"; file sha256 {fsha}" if fsha else ""))
    for b in bad[:20]:
        print("  BAD", b)
    return hd, ents, len(bad)


# ---- the independent description --------------------------------------------------------------------------------
TP, D = 4, 6144


def describe(name, r, full):
    """For an FP8 weight entry: (pieces, interleaved, master names), where the natural matrix is the pieces stacked
    by rows; a piece is (source, rows (a, b), cols (a, b) or None, transposed). Written from docs/attention.md."""
    m = re.match(r"layers\.(\d+)\.(.+)\.weight$", name)
    L, k = int(m.group(1)), m.group(2)
    p, a = f"model.layers.{L}.", f"model.layers.{L}.self_attn."
    if k == "attn.xproj":
        pcs = [(a + "q_a_proj.weight", (512 * r, 512 * r + 512), None, False),
               (a + "kv_a_proj_with_mqa.weight", (128 * r, 128 * r + 128), None, False),
               (a + "kv_a_proj_with_mqa.weight", (512 + 16 * r, 512 + 16 * r + 16), None, False)]
        if full:
            pcs.append((a + "indexer.wk.weight", (32 * r, 32 * r + 32), None, False))
        return pcs, False
    if k == "attn.q_b":
        return [(a + "q_b_proj.weight", (4096 * r, 4096 * r + 4096), None, False)], False
    if k == "indexer.wq_b":
        return [(a + "indexer.wq_b.weight", (1024 * r, 1024 * r + 1024), None, False)], False
    if k == "attn.w_uk":  # head h: kv_b rows 448H .. 448H+191 transposed to [512][192]
        return [(a + "kv_b_proj.weight", (448 * H, 448 * H + 192), None, True) for H in range(16 * r, 16 * r + 16)], False
    if k == "attn.w_uv":  # head h: kv_b rows 448H+192 .. 448H+447, [256][512]
        return [(a + "kv_b_proj.weight", (448 * H + 192, 448 * H + 448), None, False) for H in range(16 * r, 16 * r + 16)], False
    if k == "attn.o_proj":
        return [(a + "o_proj.weight", (0, D), (4096 * r, 4096 * r + 4096), False)], False
    if k == "mtp.eh_proj":
        return [(p + "eh_proj.weight", (1536 * r, 1536 * r + 1536), None, False)], False
    mm = re.match(r"(mlp|shared)\.(gate_up|gate|up|down)$", k)
    if mm:
        base = p + "mlp." + ("shared_experts." if mm.group(1) == "shared" else "")
        i = 3072 if mm.group(1) == "mlp" else 512
        if mm.group(2) == "gate_up":
            return [(base + "gate_proj.weight", (i * r, i * r + i), None, False),
                    (base + "up_proj.weight", (i * r, i * r + i), None, False)], True
        if mm.group(2) in ("gate", "up"):
            return [(base + mm.group(2) + "_proj.weight", (i * r, i * r + i), None, False)], False
        return [(base + "down_proj.weight", (0, D), (i * r, i * r + i), False)], False
    raise SystemExit(f"{name}: no independent description")


def describe_plain(name, r):
    """BF16/F32 entries: (source, rows (a, b) or None for all)."""
    v = 154880 // TP
    fixed = {"embed.weight": ("model.embed_tokens.weight", (v * r, v * r + v)),
             "head.weight": ("lm_head.weight", (v * r, v * r + v)), "norm.weight": ("model.norm.weight", None)}
    if name in fixed:
        return fixed[name]
    m = re.match(r"layers\.(\d+)\.(.+)$", name)
    L, k = int(m.group(1)), m.group(2)
    p, a = f"model.layers.{L}.", f"model.layers.{L}.self_attn."
    table = {"attn_norm.weight": p + "input_layernorm.weight", "ffn_norm.weight": p + "post_attention_layernorm.weight",
             "attn.q_norm.weight": a + "q_a_layernorm.weight", "attn.kv_norm.weight": a + "kv_a_layernorm.weight",
             "indexer.k_norm.weight": a + "indexer.k_norm.weight", "indexer.k_norm.bias": a + "indexer.k_norm.bias",
             "router.weight": p + "mlp.gate.weight", "router.bias": p + "mlp.gate.e_score_correction_bias",
             "mtp.enorm.weight": p + "enorm.weight", "mtp.hnorm.weight": p + "hnorm.weight",
             "mtp.head_norm.weight": p + "shared_head.norm.weight"}
    if k == "indexer.weights_proj.weight":
        return a + "indexer.weights_proj.weight", (8 * r, 8 * r + 8)
    if k in table:
        return table[k], None
    raise SystemExit(f"{name}: no independent description")


def untile(b, layout, N, K):
    """Natural [N][K] codes from tiled bytes, gathering by glw.h's offset formula."""
    t = np.frombuffer(b, np.uint8)
    n = np.arange(N, dtype=np.int64)[:, None]
    k = np.arange(K, dtype=np.int64)[None, :]
    if layout == glw.T64:
        off = ((n // 8) * (K // 64) + k // 64) * 512 + (4 * (n % 8) + (k % 64) // 16) * 16 + k % 16
    elif layout == glw.T128:
        off = (((n // 8) * (K // 128) + k // 128) * 32 + 4 * (n % 8) + (k % 64) // 16) * 32 + 16 * ((k % 128) // 64) + k % 16
    else:
        raise SystemExit(f"layout {layout}")
    return t[off]


INV448 = np.float32(1.0 / 448.0)


def block_scales(w):
    """f32 [ceil(N/128)][K/128]: f32(amax of the block) * f32(1/448), numpy only."""
    N, K = w.shape
    nb = (N + 127) // 128
    pad = np.zeros((nb * 128, K), np.float32)
    pad[:N] = np.abs(w)
    amax = pad.reshape(nb, 128, K // 128, 128).max(axis=(1, 3))
    return (amax.astype(np.float32) * INV448).astype(np.float32)


class Masters:
    def __init__(self, ck, mirror):
        self.ck, self.mirror, self.cache = ck, mirror, {}

    def get(self, name):
        """(f32 master [N][K] numpy, block scales) for a weight; cached per unit."""
        if name not in self.cache:
            dense = re.match(r"model\.layers\.[012]\.mlp\.(gate|up|down)_proj\.weight$", name)
            t, _ = (self.mirror if dense else self.ck).get(name)
            w = t.float().numpy()
            self.cache[name] = (w, block_scales(w))
        return self.cache[name]


def e4m3_rne(x):
    import torch
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def check_fp8(fd, e, se, r, full, M):
    pcs, inter = describe(e["name"], r, full)
    N, K = e["shape"]
    rs, ks = e["blk"]
    if se["blk"] != e["blk"]:
        return "weight and scale tables disagree on the block"
    wvals, svals = [], []
    for src, (a0, a1), cols, tr in pcs:
        w, bs = M.get(src)
        c0, c1 = cols or (0, w.shape[1])
        piece = w[a0:a1, c0:c1]
        rb = np.repeat(np.arange(a0, a1) // 128, 1)[:, None]
        cb = (np.arange(c0, c1) // 128)[None, :]
        sc = bs[rb, cb]
        if tr:
            piece, sc = piece.T, sc.T
        wvals.append(piece)
        svals.append(sc)
    if inter:  # rows 2i gate, 2i+1 up
        g, u = wvals
        gs, us = svals
        W = np.empty((2 * g.shape[0], g.shape[1]), np.float32)
        W[0::2], W[1::2] = g, u
        Sx = np.empty_like(W)
        Sx[0::2], Sx[1::2] = gs, us
    else:
        W, Sx = np.concatenate(wvals), np.concatenate(svals)
    if W.shape != (N, K):
        return f"described {W.shape}, TOC {(N, K)}"
    st = np.frombuffer(read_entry(fd, se), np.float32).reshape(se["shape"])
    n = np.arange(N)[:, None]
    k = np.arange(K)[None, :]
    fs = st[(n // 2) // rs, n % 2, k // ks] if inter else st[n // rs, k // ks]
    if not np.array_equal(fs.view(np.uint32), Sx.view(np.uint32)):
        return f"scales: {int((fs != Sx).sum())} of {fs.size} elements read a scale other than their source block's"
    codes = untile(read_entry(fd, e), e["layout"], N, K)
    want = e4m3_rne(W / Sx)
    if not np.array_equal(codes, want):
        return f"codes: {int((codes != want).sum())} of {codes.size} differ from e4m3_rne(master / scale)"
    return None


EXPERT_SAMPLE = (0, 1, 17, 128, 255)


def nvfp4_untile(b, N, K):
    """A rank's NVFP4 expert matrix from its [N/8][K/512][2304 B] blocks, by docs/layout.md's byte formula
    (written out per index here, not by pack_nvfp4's reshape): codes [N][K/2], scales [N][K/16]."""
    B = K // 512
    n, j = np.indices((N, K // 2))
    bb, r = j // 256, j % 256
    jj, q, t, c = r // 64, (r // 32) % 2, (r // 8) % 4, r % 8
    w = b[(n // 8 * B + bb) * glw.Q4_BLK + 512 * jj + 16 * (4 * (n % 8) + t) + 8 * q + c]
    n, k = np.indices((N, K // 16))
    bb, r = k // 32, k % 32
    jj, q, t = r // 8, (r // 4) % 2, r % 4
    s = b[(n // 8 * B + bb) * glw.Q4_BLK + 2048 + 8 * (4 * (n % 8) + t) + 2 * jj + q]
    return w, s


def check_expert(fd, e, ed, r, ck, cache):
    """An expert entry: the desc's offsets, scale_2 and format tag; a sampled expert's codes and scales against
    NVIDIA's (layers 3-77) or nvfp4_rtn of the BF16 (the MTP layer), cut to the rank's rows (gate, up) or input
    columns (down)."""
    import struct
    import torch
    L = e["layer"]
    p = f"model.layers.{L}.mlp.experts."
    if e["name"].endswith(".desc"):
        d = read_entry(fd, e)
        for x in range(256):
            for m in range(3):
                off, s2, fmt = struct.unpack_from("<QfI", d, (3 * x + m) * 64)
                n = f"{p}{x}.{('gate', 'up', 'down')[m]}_proj.weight"
                want = (np.float32(ck.get(n + "_scale_2")[0].reshape(-1)[0].item()) if L != glw.MTP else
                        rtn_s2(ck, p, x, m, cache))
                if off != (3 * x + m) * 1769472 or fmt != 0 or np.float32(s2) != want:
                    return f"descriptor {x}.{m}: offset {off}, scale_2 {s2}, fmt {fmt}"
        return None
    x = int(e["name"].split(".")[-1])
    b = np.frombuffer(read_entry(fd, e), np.uint8)
    for m, proj in enumerate(("gate", "up", "down")):
        N, K = (512, 6144) if m < 2 else (6144, 512)
        w, s = nvfp4_untile(b[m * 1769472:(m + 1) * 1769472], N, K)
        n = f"{p}{x}.{proj}_proj.weight"
        if L != glw.MTP:
            cw = ck.get(n)[0].numpy()
            cs = ck.get(n + "_scale")[0].view(torch.uint8).numpy()
        else:
            q = rtn(ck, p, x, cache)[m]
            cw, cs = q["weight"], q["weight_scale"]
        if m < 2:
            cw, cs = cw[512 * r:512 * r + 512], cs[512 * r:512 * r + 512]
        else:
            cw, cs = cw[:, 256 * r:256 * r + 256], cs[:, 32 * r:32 * r + 32]
        if not np.array_equal(w, cw) or not np.array_equal(s, cs):
            return f"{proj}: {int((w != cw).sum())} code bytes, {int((s != cs).sum())} scales differ"
    return None


def rtn(ck, p, x, cache):
    if x not in cache:
        sys.path.insert(0, os.path.join(ROOT, "..", "..", "quant"))
        import formats
        W = [ck.get(f"{p}{x}.{proj}_proj.weight")[0].float() for proj in ("gate", "up", "down")]
        gu = max(W[0].abs().max(), W[1].abs().max())
        cache.clear()
        cache[x] = [formats.nvfp4_rtn(W[0], gu), formats.nvfp4_rtn(W[1], gu), formats.nvfp4_rtn(W[2])]
    return cache[x]


def rtn_s2(ck, p, x, m, cache):
    """scale_2 = amax * f32(1/2688), amax shared by gate and up; from the BF16 maxima alone."""
    import torch
    W = [ck.get(f"{p}{x}.{proj}_proj.weight")[0] for proj in ("gate", "up", "down")]
    a = torch.maximum(W[0].abs().max(), W[1].abs().max()) if m < 2 else W[2].abs().max()
    return np.float32(a.float().item()) * np.float32(1 / 2688.0)


def indep(files, ck_dir, cache):
    import rope_table
    import sources
    ck = sources.Checkpoint(ck_dir)
    M = Masters(ck, sources.Fetched(cache, "bf16"))
    per = []
    for path in files:
        hd, ents, _ = glw.read_toc(path)
        per.append((path, hd, {e["name"]: e for e in ents}, os.open(path, os.O_RDONLY)))
    # every rank's file must hold the same names, and those the checkpoint implies
    names = sorted(per[0][2])
    for path, hd, ed, _ in per[1:]:
        if sorted(ed) != names:
            raise SystemExit(f"{path}: entries differ from {per[0][0]}'s")
    groups = {}
    for n in names:
        m = re.match(r"layers\.(\d+)\.", n)
        groups.setdefault(int(m.group(1)) if m else -1, []).append(n)
    bad = checked = 0
    t0 = time.time()
    rope = None
    rtn_cache = {}
    for L in sorted(groups):
        full = L >= 0 and f"model.layers.{L}.self_attn.indexer.wk.weight" in ck
        for n in groups[L]:
            for path, hd, ed, fd in per:
                e, r, why = ed[n], hd["rank"], None
                if e["kind"] == glw.RESERVED:
                    continue
                if re.match(r"layers\.\d+\.experts\.", n):
                    if n.endswith(".desc") or int(n.split(".")[-1]) in EXPERT_SAMPLE:
                        why = check_expert(fd, e, ed, r, ck, rtn_cache)
                        checked += 1
                        if why:
                            bad += 1
                            print(f"MISMATCH {path} {n}: {why}", flush=True)
                    continue
                if n.endswith(".scale"):
                    continue  # checked with its weight
                if e["dtype"] == glw.F8:
                    why = check_fp8(fd, e, ed[n[:-len("weight")] + "scale"], r, full, M)
                elif n == "rope.table":
                    if rope is None:
                        rope = rope_table.table(1 << 20).tobytes()
                    why = None if read_entry(fd, e) == rope else "differs from rope_table.py"
                else:
                    src, rows = describe_plain(n, r)
                    t, _ = ck.get(src)
                    import torch
                    b = (t.view(torch.uint8) if t.dtype != torch.uint8 else t).numpy()
                    if rows:
                        b = b.reshape(t.shape[0], -1)[rows[0]:rows[1]]
                    why = None if read_entry(fd, e) == b.tobytes() else f"differs from {src} {rows}"
                checked += 1
                if why:
                    bad += 1
                    print(f"MISMATCH {path} {n}: {why}", flush=True)
        M.cache.clear()
        rtn_cache.clear()
        if L % 10 == 0 or L == 78:
            print(f"  through layer {L}: {checked} tensor checks, {bad} bad, {time.time() - t0:.0f} s", flush=True)
    # cross-rank: replicated entries identical, split ones distinct pieces (by CRC; their content was checked above)
    ranks = sorted(hd["rank"] for _, hd, _, _ in per)
    xbad = 0
    for n in names:
        es = [ed[n] for _, _, ed, _ in per]
        if es[0]["kind"] == glw.RESERVED:
            continue
        crcs = [x["crc"] for x in es]
        if es[0]["kind"] in (glw.REPLICATED, glw.GENERATED) and len(set(crcs)) != 1:
            xbad += 1
            print(f"CROSS-RANK {n}: replicated but CRCs differ")
    for _, _, _, fd in per:
        os.close(fd)
    print(f"independent check: {checked} tensor checks over {len(files)} files (ranks {ranks}), {bad} bad; "
          f"cross-rank: {xbad} bad ({time.time() - t0:.0f} s)")
    return bad + xbad


def fresh(files, ck_dir, cache, shared_layout):
    import convert
    import sources
    ck = sources.Checkpoint(ck_dir)
    mirror = sources.Fetched(cache, "bf16")
    units = glw.make_plan(ck, shared_layout)
    per = []
    for path in files:
        hd, ents, _ = glw.read_toc(path)
        per.append((path, hd["rank"], {e["name"]: e for e in ents}, os.open(path, os.O_RDONLY)))
    bad = checked = 0
    t0 = time.time()
    for u in units:
        if not u.srcs or not any(o.name in ed for _, r, ed, _ in per for o in u.outs(r)):
            continue
        S = convert.load_sources(u, ck, mirror, {})
        for path, r, ed, fd in per:
            for n, arr in u.build(S, r).items():
                if n not in ed:
                    continue
                checked += 1
                if read_entry(fd, ed[n]) != np.ascontiguousarray(arr).tobytes():
                    bad += 1
                    print(f"MISMATCH {path} {n}: differs from a fresh conversion", flush=True)
        del S
    print(f"fresh conversion: {checked} tensors compared over {len(files)} files, {bad} bad ({time.time() - t0:.0f} s)")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--crc", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--indep", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache")
    ap.add_argument("--shared-layout", choices=["t64", "t128"], default="t128")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    glw._check_crc()
    bad = 0
    if a.crc or not (a.fresh or a.indep):
        for f in a.files:
            bad += check_crc(f)[2]
    if a.fresh:
        bad += fresh(a.files, a.ckpt, a.cache, a.shared_layout)
    if a.indep:
        bad += indep(a.files, a.ckpt, a.cache)
    print("PASS" if not bad else f"FAIL: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
