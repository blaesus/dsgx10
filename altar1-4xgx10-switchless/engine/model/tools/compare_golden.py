#!/usr/bin/env python3
"""The engine's teacher-forced dumps (glm_run tf --dir) against the reference's golden files (docs/reference.md § 7).

  compare_golden.py rows GOLDEN SEQ OUT.npy     the sequence's sample rows as positions in it (glm_run --sample-rows)
  compare_golden.py cmp GOLDEN SEQ ENGINE_DIR   the comparison: top-1, top-20 overlap, logsumexp and next-token
                                                logprob differences, KL and max |Δ logit| at the sample rows,
                                                hidden-state cosines, the MTP's argmax
  compare_golden.py g8 ROOT ENGINE_ROOT [--runs all,all-f32] [--seqs ...] [--mtp-runs mtp-nvfp4,all-f32] [--json F]
                                                G8 (docs/engine.md, "End to end on real weights"): the engine's dumps
                                                ENGINE_ROOT/<seq>/ against each golden run ROOT/<run>, and the first two
                                                runs against each other over the same positions (the reference's own
                                                order variance, the scale of DESIGN decision 3's 2x criterion)

GOLDEN is a golden run directory (manifest.json); SEQ a sequence's name in it. An engine dump may hold a prefix of the
sequence (glm_run --tf-max): the comparison is then over its positions. For altar-1 both sides hold the
checkpoint's bytes, but the f32 sum orders differ and routing flips amplify them (docs/engine.md § 10), so the comparison is an envelope, not bit
equality: what it prints is measured, not passed or failed.
"""
import json
import os
import sys

import numpy as np

VOCAB_USED = 154856


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


def lsm(x):
    x = np.asarray(x, np.float64)[:VOCAB_USED]
    return x - np.logaddexp.reduce(x)


class Run:
    """A dump in the golden format: a golden run (a sequence at an offset) or an engine dump (offset 0)."""

    def __init__(self, path, name=None, golden=True):
        self.path = path
        if golden:
            self.off, self.n = seq_of(path, name)
        else:
            self.off, self.n = 0, len(np.load(os.path.join(path, "tokens.npy")))
        self.golden = golden

    def a(self, f, n):  # an [N]-indexed array, the sequence's first n positions
        return np.load(os.path.join(self.path, f), mmap_mode="r")[self.off:self.off + n]

    def exists(self, f):
        return os.path.exists(os.path.join(self.path, f))

    def logit_rows(self):  # position -> row of final/logits.npy
        r = np.load(os.path.join(self.path, "final/logits_rows.npy"))
        return {int(p) - self.off: i for i, p in enumerate(r) if self.off <= p < self.off + self.n}

    def hidden_rows(self, L):  # position -> row of layers/L/hidden.npy
        if self.golden:
            sr = np.load(os.path.join(self.path, "sample_rows.npy"))
        else:
            sr = np.load(os.path.join(self.path, "layers", L, "hidden_rows.npy"))
        return {int(p) - self.off: i for i, p in enumerate(sr) if self.off <= p < self.off + self.n}

    def mtp_argmax(self, n):  # by main position t (hidden t, token t + 1 -> token t + 2); -1 where none
        out = np.full(n, -1, np.int64)
        if not self.exists("mtp/argmax.npy"):
            return None
        am = np.load(os.path.join(self.path, "mtp/argmax.npy"))
        if self.golden:
            rm = np.load(os.path.join(self.path, "mtp/rows_main.npy"))
            for i, r in enumerate(rm):
                if self.off <= r < self.off + n:
                    out[int(r) - self.off] = am[i]
        else:
            out[:] = am[:n]
        return out


def compare(A, B, n, layers=True):
    """A against B (A the reference side of the KL) over positions [0, n)."""
    r = {"n": int(n)}
    aa, ba = A.a("final/argmax.npy", n), B.a("final/argmax.npy", n)
    r["argmax_equal"] = float(np.mean(aa == ba))
    tok = A.a("tokens.npy", n)
    r["top1_text_A"] = float(np.mean(aa[:-1] == tok[1:]))
    r["top1_text_B"] = float(np.mean(ba[:-1] == tok[1:]))
    ai, bi = A.a("final/top20_ids.npy", n), B.a("final/top20_ids.npy", n)
    r["top20_overlap"] = float(np.mean([len(set(ai[i]) & set(bi[i])) / 20 for i in range(n)]))
    d = np.abs(A.a("final/lse.npy", n) - B.a("final/lse.npy", n))
    r["lse_absdiff_mean"], r["lse_absdiff_max"] = float(d.mean()), float(d.max())
    ga, gb = A.a("final/next_logprob.npy", n)[:n - 1], B.a("final/next_logprob.npy", n)[:n - 1]
    d = np.abs(ga - gb)
    r["nextlp_A"], r["nextlp_B"], r["nextlp_absdiff_mean"] = float(ga.mean()), float(gb.mean()), float(d.mean())
    ra, rb = A.logit_rows(), B.logit_rows()
    common = sorted(p for p in set(ra) & set(rb) if p < n)
    if common:
        LA = np.load(os.path.join(A.path, "final/logits.npy"), mmap_mode="r")
        LB = np.load(os.path.join(B.path, "final/logits.npy"), mmap_mode="r")
        kl, mx = [], []
        for p in common:
            la, lb = LA[ra[p]], LB[rb[p]]
            pa, pb = lsm(la), lsm(lb)
            kl.append(float(np.sum(np.exp(pa) * (pa - pb))))
            mx.append(float(np.abs(np.asarray(la[:VOCAB_USED], np.float64) - np.asarray(lb[:VOCAB_USED], np.float64)).max()))
        kl = np.array(kl)
        r["kl_rows"], r["kl_mean"], r["kl_p99"], r["kl_max"] = len(kl), float(kl.mean()), float(np.percentile(kl, 99)), float(kl.max())
        r["max_abs_dlogit"] = float(max(mx))
        r["median_max_abs_dlogit"] = float(np.median(mx))
    if layers:
        hid = {}
        Ldirs = sorted(os.listdir(os.path.join(B.path, "layers"))) if os.path.isdir(os.path.join(B.path, "layers")) else []
        for L in Ldirs:
            if L == "L78" or not os.path.exists(os.path.join(A.path, "layers", L, "hidden.npy")) \
                    or not os.path.exists(os.path.join(B.path, "layers", L, "hidden.npy")):
                continue
            ha, hb = A.hidden_rows(L), B.hidden_rows(L)
            com = sorted(p for p in set(ha) & set(hb) if p < n)
            if not com:
                continue
            HA = np.load(os.path.join(A.path, "layers", L, "hidden.npy"), mmap_mode="r")
            HB = np.load(os.path.join(B.path, "layers", L, "hidden.npy"), mmap_mode="r")
            X = np.asarray(HA[[ha[p] for p in com]], np.float64)
            Y = np.asarray(HB[[hb[p] for p in com]], np.float64)
            cs = (X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-30)
            rel = np.linalg.norm(X - Y, axis=1) / (np.linalg.norm(X, axis=1) + 1e-30)
            hid[L] = {"rows": len(com), "cos_min": float(cs.min()), "cos_mean": float(cs.mean()),
                      "rel_median": float(np.median(rel)), "rel_mean": float(rel.mean())}
        r["hidden"] = hid
    ma, mb = A.mtp_argmax(n), B.mtp_argmax(n)
    if ma is not None and mb is not None:
        both = (ma >= 0) & (mb >= 0)
        if both.any():
            r["mtp_rows"], r["mtp_equal"] = int(both.sum()), float(np.mean(ma[both] == mb[both]))
            # the one-draft greedy acceptance on teacher-forced text: draft t (token t + 2) against the main argmax at t + 1
            for nm, m, am in (("A", ma, aa), ("B", mb, ba)):
                ok = (m[:n - 1] >= 0)
                r["mtp_accept_" + nm] = float(np.mean(m[:n - 1][ok] == am[1:][ok]))
    return r


def cmp(golden, name, eng):
    G = Run(golden, name)
    E = Run(eng, golden=False)
    n = E.n
    if not (E.a("tokens.npy", n) == G.a("tokens.npy", n)).all():
        sys.exit("the engine's tokens are not the golden sequence's")
    r = compare(G, E, n)
    print(json.dumps(r, indent=1))


def g8(root, eng_root, runs, seqs, mtp_runs, out):
    res = {}
    for s in seqs:
        ed = os.path.join(eng_root, s)
        if not os.path.exists(os.path.join(ed, "tokens.npy")):
            continue
        E = Run(ed, golden=False)
        n = E.n
        rs = {"n": n}
        golds = {g: Run(os.path.join(root, g), s) for g in runs}
        for g, G in golds.items():
            if not (E.a("tokens.npy", n) == G.a("tokens.npy", n)).all():
                sys.exit(f"{s}: the engine's tokens are not {g}'s")
            rs["engine_vs_" + g] = compare(G, E, n)
            print(s, g, json.dumps({k: v for k, v in rs["engine_vs_" + g].items() if k != "hidden"}), flush=True)
        if len(runs) >= 2:
            rs["order_variance"] = compare(golds[runs[0]], golds[runs[1]], n)
            print(s, "order variance", json.dumps({k: v for k, v in rs["order_variance"].items() if k != "hidden"}), flush=True)
        for g in mtp_runs:
            G = Run(os.path.join(root, g), s)
            gm, em = G.mtp_argmax(n), E.mtp_argmax(n)
            if gm is None or em is None:
                continue
            both = (gm >= 0) & (em >= 0)
            if both.any():
                rs["mtp_vs_" + g] = {"rows": int(both.sum()), "equal": float(np.mean(gm[both] == em[both]))}
                print(s, "mtp vs", g, rs["mtp_vs_" + g], flush=True)
        res[s] = rs
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == "rows":
        rows(*a[1:])
    elif len(a) == 4 and a[0] == "cmp":
        cmp(*a[1:])
    elif len(a) >= 3 and a[0] == "g8":
        opt = {"--runs": "all,all-f32", "--seqs": "short,python,prose,agentic", "--mtp-runs": "mtp-nvfp4,all-f32,all",
               "--json": ""}
        for i in range(3, len(a), 2):
            opt[a[i]] = a[i + 1]
        g8(a[1], a[2], opt["--runs"].split(","), opt["--seqs"].split(","),
           [x for x in opt["--mtp-runs"].split(",") if x], opt["--json"])
    else:
        sys.exit(__doc__)
