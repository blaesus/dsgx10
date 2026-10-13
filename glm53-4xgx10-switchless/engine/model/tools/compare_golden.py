#!/usr/bin/env python3
"""The engine's teacher-forced dumps (glm_run tf --dir) against the reference's golden files (docs/reference.md § 7).

  compare_golden.py rows GOLDEN SEQ OUT.npy     the sequence's sample rows as positions in it (glm_run --sample-rows)
  compare_golden.py cmp GOLDEN SEQ ENGINE_DIR   the comparison: top-1, top-20 overlap, logsumexp and next-token
                                                logprob differences, KL at the sample rows, hidden-state cosines

GOLDEN is a golden run directory (manifest.json); SEQ a sequence's name in it. The reference attends in BF16
(NVIDIA's dequantised checkpoint) and this engine in FP8 (docs/attention.md), so the comparison is an envelope, not
bit equality: what it prints is measured, not passed or failed.
"""
import json
import os
import sys

import numpy as np


def seq_of(golden, name):
    with open(os.path.join(golden, "manifest.json")) as f:
        m = json.load(f)
    for s in m["sequences"]:
        if s["name"] == name:
            return s["off"], s["n"]
    sys.exit(f"no sequence {name} in {golden}: {[s['name'] for s in m['sequences']]}")


def rows(golden, name, out):
    off, n = seq_of(golden, name)
    sr = np.load(os.path.join(golden, "sample_rows.npy"))
    p = sr[(sr >= off) & (sr < off + n)] - off
    np.save(out, p.astype(np.int32))
    print(f"{len(p)} sample rows of {name} ({n} tokens) -> {out}")


def logsoftmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def cmp(golden, name, eng):
    off, n = seq_of(golden, name)
    G = lambda f: np.load(os.path.join(golden, f))
    E = lambda f: np.load(os.path.join(eng, f))
    tok = G("tokens.npy")[off:off + n]
    etok = E("tokens.npy")
    if len(etok) != n or not (etok == tok).all():
        sys.exit("the engine's tokens are not the golden sequence's")
    ga, ea = G("final/argmax.npy")[off:off + n], E("final/argmax.npy")
    print(f"{name}: {n} positions")
    print(f"  top-1 agreement            {np.mean(ga == ea):.4f}")
    gi, ei = G("final/top20_ids.npy")[off:off + n], E("final/top20_ids.npy")
    ov = np.mean([len(set(gi[i]) & set(ei[i])) / 20 for i in range(n)])
    print(f"  top-20 overlap             {ov:.4f}")
    gl, el = G("final/lse.npy")[off:off + n], E("final/lse.npy")
    d = np.abs(gl - el)
    print(f"  |lse difference|           mean {d.mean():.4f}  p99 {np.percentile(d, 99):.4f}  max {d.max():.4f}")
    gn, en = G("final/next_logprob.npy")[off:off + n - 1], E("final/next_logprob.npy")[:n - 1]
    d = en - gn
    print(f"  next-token logprob         golden mean {gn.mean():.4f}  engine mean {en.mean():.4f}  "
          f"|diff| mean {np.abs(d).mean():.4f}  p99 {np.percentile(np.abs(d), 99):.4f}")
    # full logits at the rows both kept
    gr, er = G("final/logits_rows.npy"), E("final/logits_rows.npy")
    gpos = {int(r) - off: i for i, r in enumerate(gr) if off <= r < off + n}
    both = [(gpos[int(p)], j) for j, p in enumerate(er) if int(p) in gpos]
    if both:
        glg = np.load(os.path.join(golden, "final/logits.npy"), mmap_mode="r")
        elg = np.load(os.path.join(eng, "final/logits.npy"), mmap_mode="r")
        kl = []
        for gi_, ej in both:
            lp = logsoftmax(np.asarray(glg[gi_][:154856], np.float64))
            lq = logsoftmax(np.asarray(elg[ej][:154856], np.float64))
            kl.append(float((np.exp(lp) * (lp - lq)).sum()))
        kl = np.array(kl)
        print(f"  KL(golden || engine)       {len(kl)} rows: mean {kl.mean():.5f}  p99 {np.percentile(kl, 99):.5f}  max {kl.max():.5f}")
    # hidden states at the sample rows
    sr = G("sample_rows.npy")
    sel = [(i, int(r) - off) for i, r in enumerate(sr) if off <= r < off + n]
    for L in sorted(os.listdir(os.path.join(eng, "layers"))) if os.path.isdir(os.path.join(eng, "layers")) else []:
        gp = os.path.join(golden, "layers", L, "hidden.npy")
        if not os.path.exists(gp):
            continue
        gh = np.load(gp, mmap_mode="r")
        eh = np.load(os.path.join(eng, "layers", L, "hidden.npy"))
        epos = {int(p): j for j, p in enumerate(np.load(os.path.join(eng, "layers", L, "hidden_rows.npy")))}
        cs, rel = [], []
        for i, p in sel:
            if p in epos:
                a, b = np.asarray(gh[i], np.float64), eh[epos[p]].astype(np.float64)
                cs.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
                rel.append(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
        if cs:
            print(f"  {L} hidden                 {len(cs)} rows: cosine min {min(cs):.5f} mean {np.mean(cs):.5f}  "
                  f"relative error mean {np.mean(rel):.4f}")
    if os.path.exists(os.path.join(eng, "mtp/argmax.npy")) and os.path.exists(os.path.join(golden, "mtp/argmax.npy")):
        rm = G("mtp/rows_main.npy")
        mrows = [i for i, r in enumerate(rm) if off <= r < off + n]
        gm = G("mtp/argmax.npy")[mrows]
        em = E("mtp/argmax.npy")[:len(mrows)]
        seen = em >= 0
        print(f"  MTP top-1 agreement        {np.mean(gm[seen] == em[seen]):.4f} over {seen.sum()} rows")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "rows":
        rows(*sys.argv[2:])
    elif len(sys.argv) == 5 and sys.argv[1] == "cmp":
        cmp(*sys.argv[2:])
    else:
        sys.exit(__doc__)
