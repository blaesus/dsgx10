#!/usr/bin/env python3
"""glw verify: check altar-1's per-rank .glw files.

    python -I verify.py FILE [FILE ...] --crc                      structure, TOC CRC, every tensor's CRC, padding,
                                                                    the manifest's per-tensor and file SHA-256
    python -I verify.py FILE [...] --ckpt DIR --manifest M --hashes H --indep
                                                                    every byte un-tiled back to the checkpoint's
    python -I verify.py --cross TOC0 TOC1 TOC2 TOC3                 cross-rank: four files' (or their saved TOCs')
                                                                    entries, replicated CRCs equal, split ones distinct
    python -I verify.py FILE --save-toc OUT                         the header and TOC alone (for --cross later)

--indep does not use the plan or the packers. From each entry's name it describes, written out again here from
docs/layout.md, which bytes of which checkpoint tensors the entry must hold (rows, columns, per-head pieces,
transposes, the gate/up interleave), un-tiles the entry with this file's own description of the kernels' layouts
(UNTILE_EXPERT, UNTILE_ATTN: the byte formulas of engine/experts/layout.h and attn/tools/pack.py, written out
here) and compares: every INT4 packed word, scale and zero point, every BF16 element, the F32 router biases, the RoPE
table (rope_table.py), and the descriptor tables. W_UK^T of the INT4 layers is compared with the contract's
dequantisation (int4.dequant_bf16; ctcheck.py --file compares it with compressed-tensors itself). It also checks
that the files hold exactly the entries the description expects and no reserved one. It counts what it compared.
Several files are checked in one pass over the checkpoint (each source tensor read once).
"""
import argparse, collections, hashlib, json, os, re, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import glw  # noqa: E402
import int4  # noqa: E402

GIB = 1 << 30
TP, D, V = 4, 6144, 154880


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
          f"{'OK' if not bad else str(len(bad)) + ' BAD'}" + (f"; file sha256 {fsha}" if fsha else ""), flush=True)
    for b in bad[:20]:
        print("  BAD", b)
    return hd, ents, len(bad)


# ---- the independent description ------------------------------------------------------------------------------------
# A piece: (source, rows (a, b) or None, cols (a, b) or None, transposed). An entry's natural matrix is its pieces
# stacked by rows (or interleaved row by row: gate/up). INT4 sources are cut the same way on their three parts.

def describe(name, r):
    """-> (kind, pieces, interleaved) for every non-expert entry; kind: "plain" (BF16/F32 bytes of the pieces),
    "int4" (the pieces' packed / scale / zero-point parts, through UNTILE_ATTN), "dequant" (bf16((q-z)s) of the
    pieces). Written from docs/layout.md § 3, not from glw.py."""
    vr = V // TP
    fixed = {"embed.weight": "model.embed_tokens.weight", "head.weight": "lm_head.weight"}
    if name in fixed:
        return "plain", [(fixed[name], (vr * r, vr * r + vr), None, False)], False
    if name == "norm.weight":
        return "plain", [("model.norm.weight", None, None, False)], False
    m = re.match(r"layers\.(\d+)\.(.+)$", name)
    L, k = int(m.group(1)), m.group(2)
    p, a = f"model.layers.{L}.", f"model.layers.{L}.self_attn."
    i4 = 4 <= L <= 76
    whole = {"attn_norm.weight": p + "input_layernorm.weight", "ffn_norm.weight": p + "post_attention_layernorm.weight",
             "attn.q_norm.weight": a + "q_a_layernorm.weight", "attn.kv_norm.weight": a + "kv_a_layernorm.weight",
             "indexer.k_norm.weight": a + "indexer.k_norm.weight", "indexer.k_norm.bias": a + "indexer.k_norm.bias",
             "router.weight": p + "mlp.gate.weight", "router.bias": p + "mlp.gate.e_score_correction_bias",
             "mtp.enorm.weight": p + "enorm.weight", "mtp.hnorm.weight": p + "hnorm.weight",
             "mtp.head_norm.weight": p + "shared_head.norm.weight"}
    if k in whole:
        return "plain", [(whole[k], None, None, False)], False
    heads = range(16 * r, 16 * r + 16)
    if k == "attn.xproj.weight":
        pcs = [(a + "q_a_proj.weight", (512 * r, 512 * r + 512), None, False),
               (a + "kv_a_proj_with_mqa.weight", (128 * r, 128 * r + 128), None, False),
               (a + "kv_a_proj_with_mqa.weight", (512 + 16 * r, 512 + 16 * r + 16), None, False)]
        if L in FULL:
            pcs.append((a + "indexer.wk.weight", (32 * r, 32 * r + 32), None, False))
        return "plain", pcs, False
    if k == "indexer.wq_b.weight":
        return "plain", [(a + "indexer.wq_b.weight", (1024 * r, 1024 * r + 1024), None, False)], False
    if k == "indexer.weights_proj.weight":
        return "plain", [(a + "indexer.weights_proj.weight", (8 * r, 8 * r + 8), None, False)], False
    if k == "mtp.eh_proj.weight":
        return "plain", [(p + "eh_proj.weight", (1536 * r, 1536 * r + 1536), None, False)], False
    mm = re.match(r"(mlp|shared)\.(gate_up|down)\.weight$", k)
    if mm:
        base = p + "mlp." + ("shared_experts." if mm.group(1) == "shared" else "")
        i = 3072 if mm.group(1) == "mlp" else 512
        if mm.group(2) == "gate_up":
            return "plain", [(base + "gate_proj.weight", (i * r, i * r + i), None, False),
                             (base + "up_proj.weight", (i * r, i * r + i), None, False)], True
        return "plain", [(base + "down_proj.weight", None, (i * r, i * r + i), False)], False
    ma = re.match(r"attn\.(q_b|w_uk|w_uv|o_proj)(\..+)$", k)
    if ma:
        src = {"q_b": "q_b_proj", "w_uk": "kv_b_proj", "w_uv": "kv_b_proj", "o_proj": "o_proj"}[ma.group(1)]
        src = a + src + ("" if i4 else ".weight")
        pcs = {"q_b": [(src, (4096 * r, 4096 * r + 4096), None, False)],
               "w_uk": [(src, (448 * H, 448 * H + 192), None, True) for H in heads],
               "w_uv": [(src, (448 * H + 192, 448 * H + 448), None, False) for H in heads],
               "o_proj": [(src, None, (4096 * r, 4096 * r + 4096), False)]}[ma.group(1)]
        if not i4:
            if ma.group(2) != ".weight":
                raise SystemExit(f"{name}: no such entry on a BF16 layer")
            return "plain", pcs, False
        if ma.group(1) == "w_uk" and ma.group(2) == ".weight" and FLAGS.get(name, 0) & glw.DEQUANT:
            return "dequant", pcs, False
        return "int4", pcs, False
    raise SystemExit(f"{name}: no independent description")


FULL = set()   # layers with an indexer (from the checkpoint)
FLAGS = {}     # entry flags (from the TOC): DEQUANT marks the BF16 W_UK^T of an INT4 layer


def cut2(t, rows, cols):
    if rows:
        t = t[rows[0]:rows[1]]
    if cols:
        t = t[:, cols[0]:cols[1]]
    return t


def plain_bytes(C, pcs, inter):
    arrs = []
    for src, rows, cols, tr in pcs:
        t = C.get(src)
        t = cut2(t.reshape(t.shape[0], -1) if t.ndim > 1 else t, rows, cols)
        arrs.append(np.ascontiguousarray(t.T if tr else t))
    if inter:
        g, u = arrs
        out = np.empty((2 * g.shape[0], g.shape[1]), g.dtype)
        out[0::2], out[1::2] = g, u
        return out
    return np.concatenate(arrs) if len(arrs) > 1 else arrs[0]


def int4_pieces(C, pcs):
    """The INT4 parts of the pieces (rows in multiples of 8, columns of 32), stacked by rows."""
    outs = []
    for src, rows, cols, tr in pcs:
        t = C.int4(src)
        if rows:
            a, b = rows
            assert a % 8 == 0 and b % 8 == 0
            t = dict(packed=t["packed"][a:b], scale=t["scale"][a:b], zp=t["zp"][a // 8:b // 8])
        if cols:
            a, b = cols
            assert a % 32 == 0 and b % 32 == 0
            t = dict(packed=t["packed"][:, a // 8:b // 8], scale=t["scale"][:, a // 32:b // 32],
                     zp=t["zp"][:, a // 32:b // 32])
        outs.append((t, tr))
    if any(tr for _, tr in outs):
        return outs
    return dict(packed=np.concatenate([t["packed"] for t, _ in outs]), scale=np.concatenate([t["scale"] for t, _ in outs]),
                zp=np.concatenate([t["zp"] for t, _ in outs]))


class Cache:
    """Checkpoint tensors for one layer, read once for every file checked."""

    def __init__(self, ck):
        self.ck, self.c = ck, {}

    def get(self, n):
        if n not in self.c:
            self.c[n] = self.ck.get(n)[0]
        return self.c[n]

    def int4(self, base):
        if base not in self.c:
            self.c[base] = int4.parts(self.ck, base)[0]
        return self.c[base]

    def clear(self):
        self.c.clear()


# ---- the kernels' layouts, written out again: un-tilers keyed by the expert format / the attention INT4 layout ------
def _untile_test_int4(b, n, k):
    a, s = n * k // 2, n * (k // 32) * 2
    return dict(packed=b[:a].view(np.int32).reshape(n, k // 8), scale=b[a:a + s].view(np.uint16).reshape(n, k // 32),
                zp=b[a + s:a + s + (n // 8) * (k // 32) * 4].view(np.int32).reshape(n // 8, k // 32))


def _untile_test_bf16(b, n, k):
    return b[:n * k * 2].view(np.uint16).reshape(n, k)


# (format tag nibble, packer name) -> un-tiler(bytes uint8, n, k) -> int4 dict or uint16 [n][k]
UNTILE_EXPERT = {(glw.FMT_INT4, "int4 (TEST stand-in)"): _untile_test_int4,
                 (glw.FMT_BF16, "bf16 (TEST stand-in)"): _untile_test_bf16}
# packer name -> un-tiler({suffix: bytes}, N, K, grouped) -> int4 dict
UNTILE_ATTN = {"TEST": lambda d, N, K, g: _untile_test_int4(d[".weight"], N, K)}
DESC_EXPECT = {}  # packer name -> fn(off, fmt) -> the 64 desc bytes the kernels expect (layout.h make_desc)


def _test_desc(off, fmt):
    import struct
    return struct.pack("<QfI", off, 1.0, fmt) + bytes(glw.MATDESC_BYTES - 16)


DESC_EXPECT["int4 (TEST stand-in)"] = DESC_EXPECT["bf16 (TEST stand-in)"] = _test_desc


def count_int4(t):
    n, k = int4.shape(t)
    return {"int4 values": n * k, "int4 scales": t["scale"].size, "int4 zero points": t["zp"].size * 8}


def indep(files, ck):
    per = []
    for path in files:
        hd, ents, _ = glw.read_toc(path)
        man_p = path + ".manifest.json"
        man = json.load(open(man_p)) if os.path.exists(man_p) else {}
        per.append((path, hd, {e["name"]: e for e in ents}, os.open(path, os.O_RDONLY), man))
    names = sorted(per[0][2])
    for path, hd, ed, _, _ in per[1:]:
        if sorted(ed) != names:
            raise SystemExit(f"{path}: entries differ from {per[0][0]}'s")
    for n in ck.names():
        m = re.match(r"model\.layers\.(\d+)\.self_attn\.indexer\.wk\.weight$", n)
        if m:
            FULL.add(int(m.group(1)))
    for _, _, ed, _, _ in per:
        for n, e in ed.items():
            FLAGS[n] = e["flags"]
    groups = collections.OrderedDict()
    for n in names:
        m = re.match(r"layers\.(\d+)\.", n)
        groups.setdefault(int(m.group(1)) if m else -1, []).append(n)
    attn_packer = (per[0][4].get("packers") or {}).get("attention int4")
    expert_packers = {}
    for path, _, _, _, man in per:
        for L, rec in (man.get("expert_layers") or {}).items():
            expert_packers[int(L)] = rec["format"]
    C = Cache(ck)
    cnt = collections.Counter()
    bad = checked = 0
    t0 = time.time()
    rope = None
    for L, ns in groups.items():
        for n in ns:
            for path, hd, ed, fd, man in per:
                e, r, why = ed[n], hd["rank"], None
                if e["kind"] == glw.RESERVED:
                    bad += 1
                    print(f"MISMATCH {path} {n}: reserved (not filled)", flush=True)
                    continue
                if re.match(r"layers\.\d+\.experts\.", n):
                    why = check_expert(fd, e, ed, r, C, expert_packers.get(L), cnt)
                elif n == "rope.table":
                    if rope is None:
                        import rope_table
                        rope = rope_table.table(1 << 20).tobytes()
                    why = None if read_entry(fd, e) == rope else "differs from rope_table.py"
                    cnt["rope u32"] += len(rope) // 4
                else:
                    kind, pcs, inter = describe(n, r)
                    b = read_entry(fd, e)
                    if kind == "plain":
                        want = plain_bytes(C, pcs, inter)
                        if tuple(want.shape) != tuple(e["shape"]):
                            why = f"described {want.shape}, TOC {e['shape']}"
                        elif b != want.tobytes():
                            why = "differs from the checkpoint's bytes"
                        cnt["f32" if want.dtype == np.float32 else "bf16"] += want.size
                    elif kind == "dequant":
                        t = C.int4(pcs[0][0])
                        key = ("dq", pcs[0][0])
                        if key not in C.c:
                            C.c[key] = int4.dequant_bf16(t)
                        dq = C.c[key]
                        want = np.concatenate([cut2(dq, rows, cols).T if tr else cut2(dq, rows, cols)
                                               for _, rows, cols, tr in pcs])
                        why = None if b == np.ascontiguousarray(want).tobytes() else "differs from bf16((q-z)s)"
                        cnt["bf16 dequantised"] += want.size
                    else:
                        why = check_attn_int4(fd, n, e, ed, pcs, C, attn_packer, cnt)
                        if why == "sibling":
                            continue
                checked += 1
                if why:
                    bad += 1
                    print(f"MISMATCH {path} {n}: {why}", flush=True)
            mx = re.match(r"layers\.\d+\.experts\.(\d+)$", n)
            if mx:  # every file has checked this expert: drop its sources (a BF16 expert is 75 MB)
                tag = f".mlp.experts.{mx.group(1)}."
                for k in [k for k in C.c if tag in k]:
                    del C.c[k]
        C.clear()
        if L % 10 == 0 or L == 78:
            print(f"  through layer {L}: {checked} tensor checks, {bad} bad, {time.time() - t0:.0f} s", flush=True)
    for _, _, _, fd, _ in per:
        os.close(fd)
    print("compared: " + ", ".join(f"{k} {v:,}" for k, v in sorted(cnt.items())))
    print(f"independent check: {checked} tensor checks over {len(files)} files (ranks "
          f"{sorted(hd['rank'] for _, hd, _, _, _ in per)}), {bad} bad ({time.time() - t0:.0f} s)", flush=True)
    return bad, cnt


def check_attn_int4(fd, n, e, ed, pcs, C, packer, cnt):
    """An INT4 attention tensor: its entries (the weight and any siblings the layout has) un-tiled together."""
    m = re.match(r"(layers\.\d+\.attn\.\w+)(\..+)$", n)
    stem, suf = m.group(1), m.group(2)
    sib = sorted(x for x in ed if x.startswith(stem + "."))
    if suf != ".weight" and stem + ".weight" in ed:
        return "sibling"  # checked with .weight
    un = UNTILE_ATTN.get("TEST" if packer == "TEST" else packer)
    if un is None:
        return f"no independent un-tiler for the attention INT4 layout ({packer})"
    d = {x[len(stem):]: np.frombuffer(read_entry(fd, ed[x]), np.uint8) for x in sib}
    want = int4_pieces(C, pcs)
    if isinstance(want, list):
        return "a transposed INT4 tile: no independent description"
    N, K = int4.shape(want)
    got = un(d, N, K, "w_uv" in stem)
    if not int4.equal(got, want):
        return "un-tiled INT4 parts differ from the checkpoint's"
    cnt.update(count_int4(want))
    return None


def check_expert(fd, e, ed, r, C, fmt_name, cnt):
    """An expert entry: every matrix un-tiled and compared with the checkpoint's rank slice; the desc table: every
    descriptor equal to the kernels' expectation for its matrix offset and format."""
    L = e["layer"]
    tag = e["fmt"]
    f = tag & 15
    if tag != f | f << 4 | f << 8 or f != (glw.FMT_INT4 if 4 <= L <= 76 else glw.FMT_BF16):
        return f"format tag {tag:x} on layer {L}"
    shapes = [(512, 6144), (512, 6144), (6144, 512)]
    un = UNTILE_EXPERT.get((f, fmt_name))
    if un is None:
        return f"no independent un-tiler for {fmt_name}"
    eb = ed[f"layers.{L}.experts.0"]["bytes"]
    if e["name"].endswith(".desc"):
        de = DESC_EXPECT.get(fmt_name)
        if de is None:
            return f"no descriptor description for {fmt_name}"
        d = read_entry(fd, e)
        mb = MAT_BYTES[fmt_name]
        moff = [0, mb(*shapes[0]), mb(*shapes[0]) + mb(*shapes[1])]
        for x in range(glw.N_EXPERTS):
            for m in range(3):
                if d[(3 * x + m) * 64:(3 * x + m + 1) * 64] != de(x * eb + moff[m], f):
                    return f"descriptor {x}.{m}"
        cnt["descriptors"] += 3 * glw.N_EXPERTS
        return None
    x = int(e["name"].split(".")[-1])
    b = np.frombuffer(read_entry(fd, e), np.uint8)
    off = 0
    for m, proj in enumerate(("gate", "up", "down")):
        mb = MAT_BYTES[fmt_name](*shapes[m])
        got = un(b[off:off + mb], *shapes[m])
        off += mb
        base = f"model.layers.{L}.mlp.experts.{x}.{proj}_proj"
        if f == glw.FMT_INT4:
            t = C.int4(base)
            if m < 2:
                t = dict(packed=t["packed"][512 * r:512 * r + 512], scale=t["scale"][512 * r:512 * r + 512],
                         zp=t["zp"][64 * r:64 * r + 64])
            else:
                t = dict(packed=t["packed"][:, 64 * r:64 * r + 64], scale=t["scale"][:, 16 * r:16 * r + 16],
                         zp=t["zp"][:, 16 * r:16 * r + 16])
            if not int4.equal(got, t):
                return f"{proj}: un-tiled INT4 parts differ from the checkpoint's"
            cnt.update(count_int4(t))
        else:
            w = C.get(base + ".weight")
            w = w[512 * r:512 * r + 512] if m < 2 else w[:, 512 * r:512 * r + 512]
            if not np.array_equal(got, w):
                return f"{proj}: {int((got != w).sum())} BF16 elements differ"
            cnt["bf16 experts"] += w.size
    if off != e["bytes"]:
        return f"entry {e['bytes']} B, matrices {off} B"
    cnt["expert entries"] += 1
    return None


MAT_BYTES = {"int4 (TEST stand-in)": lambda n, k: glw.align_up(n * k // 2 + n * (k // 32) * 2 + (n // 8) * (k // 32) * 4),
             "bf16 (TEST stand-in)": lambda n, k: n * k * 2}


# ---- cross-rank ---------------------------------------------------------------------------------------------------
def cross(tocs):
    per = [glw.read_toc(p)[:2] for p in tocs]
    ranks = sorted(hd["rank"] for hd, _ in per)
    if ranks != [0, 1, 2, 3]:
        raise SystemExit(f"--cross needs ranks 0-3, got {ranks}")
    ents = [{e["name"]: e for e in es} for _, es in per]
    names = sorted(ents[0])
    bad = 0
    if any(sorted(x) != names for x in ents[1:]):
        print("CROSS-RANK: the files hold different entry names")
        bad += 1
    rep = split = 0
    for n in names:
        es = [x[n] for x in ents]
        if any(e["kind"] == glw.RESERVED for e in es):
            bad += 1
            print(f"CROSS-RANK {n}: reserved")
            continue
        same_meta = len({(e["dtype"], e["shape"], e["layout"], e["kind"], e["fmt"], e["flags"], e["bytes"]) for e in es}) == 1
        crcs = [e["crc"] for e in es]
        if not same_meta:
            bad += 1
            print(f"CROSS-RANK {n}: dtype, shape, layout, kind, format or size differ between ranks")
        if es[0]["kind"] in (glw.REPLICATED, glw.GENERATED) or n.endswith(".experts.desc"):
            rep += 1  # a descriptor table holds offsets, the same on every rank
            if len(set(crcs)) != 1:
                bad += 1
                print(f"CROSS-RANK {n}: replicated, CRCs differ {crcs}")
        else:
            split += 1
            if len(set(crcs)) != 4:
                bad += 1
                print(f"CROSS-RANK {n}: split, but two ranks' CRCs are equal {crcs}")
    print(f"cross-rank: {len(names)} entries over ranks 0-3: {rep} replicated (and descriptor tables) with equal CRCs, {split} split with "
          f"four distinct CRCs; {bad} bad")
    return bad


def save_toc(path, out):
    hd, _, _ = glw.read_toc(path)
    with open(path, "rb") as f:
        b = f.read(hd["data_offset"])
    open(out, "wb").write(b)
    print(f"{out}: {len(b)} B (header and TOC of {path})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--crc", action="store_true")
    ap.add_argument("--indep", action="store_true")
    ap.add_argument("--cross", nargs=4)
    ap.add_argument("--save-toc")
    ap.add_argument("--ckpt")
    ap.add_argument("--index")
    ap.add_argument("--manifest")
    ap.add_argument("--hashes")
    ap.add_argument("--unverified-ok", action="store_true")
    ap.add_argument("--test-packers", action="store_true")
    ap.add_argument("--expert-packer", help="the expert kernels' int4_pack.py (default: engine/weights/int4_pack.py)")
    ap.add_argument("--attn-packer", help="the attention side's pack.py with int4_tiles (default: attn/tools/pack.py)")
    ap.add_argument("--selftest", action="store_true", help="the un-tilers against the formulas and the packers")
    ap.add_argument("--json", help="write the counts and verdicts here")
    a = ap.parse_args()
    glw._check_crc()
    import packers
    packers.register(test=a.test_packers, experts=a.expert_packer, attn=a.attn_packer)
    import verify_layouts  # the kernels' layouts written out again
    verify_layouts.register(sys.modules[__name__])
    if a.selftest:
        return 1 if verify_layouts.selftest(sys.modules[__name__]) else 0
    bad = 0
    res = {}
    if a.save_toc:
        save_toc(a.files[0], a.save_toc)
        return 0
    if a.cross:
        bad += cross(a.cross)
    if a.crc:
        for f in a.files:
            bad += check_crc(f)[2]
    if a.indep:
        import sources
        if not (a.manifest and a.hashes) and not a.unverified_ok:
            raise SystemExit("--manifest and --hashes, or --unverified-ok")
        ck = sources.Checkpoint(a.ckpt, a.index, a.manifest, a.hashes)
        b, cnt = indep(a.files, ck)
        bad += b
        res["counts"] = dict(cnt)
    res["bad"] = bad
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print("PASS" if not bad else f"FAIL: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
