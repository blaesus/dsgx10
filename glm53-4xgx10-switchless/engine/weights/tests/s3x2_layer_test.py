#!/usr/bin/env python3
"""The S3x2 packing end to end on a file: rewrite a COPY of a one-layer test file's routed experts (convert.py
--only layers.L. then experts_fill.py --layers L) in S3x2 from the BF16 master, through s3x2_pack.py, so that
glw_kcheck (--partial --layers L --blocks) and glw_expert_fill_test can check what the converter will write.
Not the converter: experts_fill.py owns the fill; this only shows how it calls s3x2_pack and checks the result.

    python -I s3x2_layer_test.py --file COPY.glw --master MASTER --layer 3 --recipe mix --dump DIR [--experts-dump 0 17 255]

The copy's layer-L expert entries are re-declared with the recipe's format tag and size (shape, reserve_bytes),
the data from the desc entry on is rewritten (desc, then the 256 entries back to back), the TOC and header
rewritten; no manifest is touched (the copy has none). --dump writes the canonical rank slices of a few experts as
L<layer>.r<rank>.e<E>.<proj>.{w,s} for glw_expert_fill_test.
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import glw  # noqa: E402
import s3x2_pack as S  # noqa: E402
from experts_fill import Target  # noqa: E402  (the TOC rewrite only)

E = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--recipe", default="mix", choices=sorted(S.RECIPES))
    ap.add_argument("--dump")
    ap.add_argument("--experts-dump", type=int, nargs="*", default=[0, 17, 255])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t = Target(a.file)
    if t.man:
        raise SystemExit("refusing: the file has a manifest (use a copy)")
    L, r = a.layer, t.r
    names = [f"layers.{L}.experts.desc"] + [f"layers.{L}.experts.{e}" for e in range(E)]
    ids = [t.idx[n] for n in names]
    fm = S.RECIPES[a.recipe]
    tag = glw.expert_fmt(*fm)
    ebytes = S.mat_bytes(fm[0], S.KI, S.KH) + S.mat_bytes(fm[1], S.KI, S.KH) + S.mat_bytes(fm[2], S.KH, S.KI)
    start = t.ents[ids[0]]["offset"] if t.ents[ids[0]]["kind"] != glw.RESERVED else t.end
    for i in ids[1:]:
        t.ents[i].update(shape=(ebytes,), fmt=tag, reserve_bytes=ebytes, kind=glw.RESERVED, offset=0, bytes=0)
    t.end = start
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
    base = start + E * 3 * glw.MATDESC_BYTES
    desc = bytearray()
    t0 = time.time()
    for e in range(E):
        W = S.load_master(a.master, L, e)
        q = S.quantise_expert(*W, a.recipe)
        sl = [S.rank_slice(f, c, m, r) for m, (f, c) in enumerate(q)]
        ent, etag, _ = S.expert_entry(sl)
        assert etag == tag and ent.size == ebytes
        desc += S.expert_descs(sl, e * ebytes)
        rel = max(float((S.value_nos2(s) * float(s["s2"]) - (w[512 * r:512 * r + 512] if m < 2 else w[:, 512 * r:512 * r + 512])).norm() /
                        (w[512 * r:512 * r + 512] if m < 2 else w[:, 512 * r:512 * r + 512]).norm()) for m, (s, w) in enumerate(zip(sl, W)))
        t.put(ids[e + 1], ent, base + e * ebytes, rel, True)
        if a.dump and e in a.experts_dump:
            for m, s in enumerate(sl):
                s["w"].tofile(os.path.join(a.dump, f"L{L}.r{r}.e{e}.{S.PROJ[m]}.w"))
                s["s"].tofile(os.path.join(a.dump, f"L{L}.r{r}.e{e}.{S.PROJ[m]}.s"))
        if e % 32 == 31:
            print(f"  expert {e} ({time.time() - t0:.0f} s)", flush=True)
    t.put(ids[0], np.frombuffer(bytes(desc), np.uint8), start)
    t.end = base + E * ebytes
    t.finish({})
    print(f"layer {L} rank {r}: {a.recipe}, {E} experts of {ebytes} B, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
