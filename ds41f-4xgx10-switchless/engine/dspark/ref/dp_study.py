#!/usr/bin/env python3
# The draft pass's approximations, a study (docs/engine.md, "The draft pass's approximations: a study"). The drafts
# never change the tokens (the verify judges them by the target's own samples), so the drafter may be approximated
# for speed; what it costs is acceptance. The ring job (dpjob) measured both sides:
#   - spectf over the same sequences with the drafter's weights rounded in place to a cheaper format's values
#     (DSPARK_SIM: none, fp4blocks, fp4head, fp8markov, all three), so each variant's own drafts and survival are
#     recorded along the target's tokens (tf_V.log), and for none and all the drafter's top 8 along them (tp_V.jsonl);
#   - bench_dspark's pass by the vocabulary rows its head GEMV and Markov chain cover (b_vrows.log): half the rows
#     reads the bytes of an FP4 head, and of an FP8 Markov head.
# This replays today's chain (pld_study.py's walk, gate and step costs, per-trace offsets) on each variant's records
# with the pass shortened by the variant's saving (a bootstrap over sequences gives the spread), and scores
# vocabulary subsets on top of none and all from the probe records (vocab_study.py's sub_rec: exact while the drafts
# are right): static, with each session's context, or adaptive (the subset only while the context's last W tokens
# are all in it, the whole vocabulary otherwise). Bundles add the modelled levers; those without the FP4 blocks use
# fp4head's records.
#   dp_study.py JOBDIR [--counts COUNTS.json] [--latin LATIN.json] [--fp4blocks-ms MS] [--accept-ms MS]
#               [--woa-ms MS] [--window W] [--gate DIR] [--pld DIR]
# JOBDIR: seqs.json (seq -> name, group, dir), sl_all.txt, tf_V.log, tp_V.jsonl, b_vrows.log. No numpy.
import collections, json, math, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pld_study as P  # noqa: E402
import vocab_study as VS  # noqa: E402

V = VS.V
GATE, PLD = "/data/dsgb10/bench/gate", "/data/dsgb10/kernels/pld"
SIMS = [("none", "today's drafter"), ("fp4blocks", "the blocks' FP8 matrices and main_proj as MXFP4"),
        ("fp4head", "the FP8 head copy as MXFP4"), ("fp8markov", "the Markov head as FP8"),
        ("all", "all three")]
# Modelled (no kernel yet), ms a pass: the blocks' dense FP8 (40.5 MB a block a rank: wq_a, wkv, wq_b, wo_a, wo_b,
# the shared expert) and main_proj (19.66 MB) as MXFP4, 17 bytes a group of 32 in place of 32 (the e4m3 copies'
# 32x32 scales aside), at the ~220 GB/s the blocks' GEMVs stream; accept beside the verify's tail (main_proj's
# DRAM read and the window writes behind the sampling's latency); wo_a's act_quant no longer fixing 64 CTAs
# (175 -> ~235 GB/s, on FP4's bytes; twice that on FP8's)
FP4BLOCKS_MS = (3 * 40.5e6 + 19.66e6) * (1 - 17 / 32) / 220e9 * 1e3
ACCEPT_MS = 0.05
WOA_MS = 0.02
ORDER = ["edit-r2", "bench-prose", "bench-code", "bench-agentic", "bench-json", "cat-prose", "cat-structural",
         "cat-code", "nonlatin-zh", "nonlatin-ja", "nonlatin-ru"]


def opt(a, k, dflt):
    return a[a.index(k) + 1] if k in a else dflt


def timings(path):
    """b_vrows.log -> {rows: {draft, a1, a6, head}} (ms, one rank, stand-in collectives)."""
    out, cur = {}, None
    pats = [("draft", r"draft \(3 blocks, head, chain\)\s+([\d.]+) ms"), ("a1", r"accept 1 \+ draft\s+([\d.]+) ms"),
            ("a6", r"accept 6 \+ draft\s+([\d.]+) ms"), ("head", r"head GEMV alone\s+([\d.]+) ms")]
    for line in open(path):
        m = re.search(r"vocabulary rows (\d+):", line)
        if m:
            cur = out.setdefault(int(m.group(1)), {})
            continue
        for k, p in pats:
            m = re.search(p, line)
            if m and cur is not None:
                cur[k] = float(m.group(1))
    return out


def group_of_seq(s):
    g = s["group"]
    if g == "nonlatin":
        g = "nonlatin-" + s["name"].split("_")[1]
    return g


def load(job, gate, pld):
    seqs = json.load(open(os.path.join(job, "seqs.json")))
    prompt = [int(l.split()[1]) for l in open(os.path.join(job, "sl_all.txt")) if l.strip()]
    gtrs, _ = P.load_all(gate)
    ptrs, pmeta = P.load_all(pld)
    alltr = gtrs + ptrs
    bts = P.betas(alltr)
    offs = P.offsets(alltr, pmeta, bts)
    byname = {tr.name: tr for tr in alltr}
    trs = {}
    for s in seqs:
        k = s["seq"]
        if s["name"] in byname:
            trs[k] = byname[s["name"]]
        else:
            trs[k] = P.Trace(s["name"], P.npy_i4(os.path.join(s["dir"], "tokens.npy")), prompt[k])
        assert trs[k].P == prompt[k], (s["name"], trs[k].P, prompt[k])
    return seqs, trs, bts, offs


def replay(seqs, trs, bts, offs, recs, save, per=None):
    """{group: [tok, ms, steps]} of today's chain on recs ({seq: {i: (drafts, survival)}}), the pass `save` ms
    shorter; per (a dict): each sequence's (tok, ms) too."""
    res = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for s in seqs:
        k = s["seq"]
        tr, r = trs[k], recs.get(k)
        if not r:
            continue
        saved = tr.recs
        tr.recs = r
        cost = dict(P.COST)
        cost["c0"] += offs.get(tr.name, 0.0)
        cost["draft"] -= save
        w = P.walk(tr, P.P_chain(), cost, bts[P.group_of(tr.name)])
        tr.recs = saved
        g = res[group_of_seq(s)]
        g[0] += w["tok"]
        g[1] += w["ms"]
        g[2] += w["steps"]
        if per is not None:
            per[k] = (w["tok"], w["ms"])
    return res


def spread(seqs, p0, p1, g, n=400):
    """The gain's standard deviation over sequences resampled with replacement within group g (4 or more)."""
    ks = [s["seq"] for s in seqs if group_of_seq(s) == g and s["seq"] in p0 and s["seq"] in p1]
    if len(ks) < 4:
        return None
    rnd = random.Random(7)
    v = []
    for _ in range(n):
        b = [rnd.choice(ks) for _ in ks]
        t0, m0 = sum(p0[k][0] for k in b), sum(p0[k][1] for k in b)
        t1, m1 = sum(p1[k][0] for k in b), sum(p1[k][1] for k in b)
        v.append(100.0 * ((t1 / m1) / (t0 / m0) - 1))
    mu = sum(v) / n
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))


def gain(r, r0, g):
    return 100.0 * ((r[g][0] / r[g][1]) / (r0[g][0] / r0[g][1]) - 1) if r[g][2] and r0[g][2] else float("nan")


def acc(r, g):
    return (r[g][0] - r[g][2]) / r[g][2] if r[g][2] else float("nan")


def subset_recs(seqs, trs, recs, probe, ins, ctx):
    """The subset drafter's records ({seq: {i: ...}}) from a variant's records and probe ({(seq, i): record}); ctx:
    each anchor's context (the prompt and the output through it) joins the subset. Also the context's tokens outside
    the static subset per sequence."""
    out, extra = {}, {}
    for s in seqs:
        k = s["seq"]
        tr, r = trs[k], recs.get(k, {})
        m = bytearray(ins) if ctx else ins
        seen = 0
        nr = {}
        for i in sorted(r):
            if ctx:
                while seen <= min(i + 1, tr.L - 1):
                    m[tr.toks[seen]] = 1
                    seen += 1
            pr = probe.get((k, i))
            nr[i] = VS.sub_rec(pr, m) if pr is not None else VS.lower_rec(r[i], m)
        out[k] = nr
        extra[k] = sum(1 for t in set(tr.toks) if not ins[t])
    return out, extra


def adaptive_recs(seqs, trs, recs, probe, ins, W):
    """The subset drafter's records where the context's last W tokens (through the anchor) are all in the subset,
    today's records elsewhere; and which steps use the subset ({seq: {i: bool}})."""
    out, use = {}, {}
    for s in seqs:
        k = s["seq"]
        tr, r = trs[k], recs.get(k, {})
        ps = [0]
        for t in tr.toks:
            ps.append(ps[-1] + (0 if ins[t] else 1))
        nr, u = {}, {}
        for i in sorted(r):
            hi = min(i + 2, tr.L)  # tokens 0 .. i + 1
            if ps[hi] - ps[max(0, hi - W)] == 0:
                pr = probe.get((k, i))
                nr[i] = VS.sub_rec(pr, ins) if pr is not None else VS.lower_rec(r[i], ins)
                u[i] = True
            else:
                nr[i] = r[i]
                u[i] = False
        out[k], use[k] = nr, u
    return out, use


def main():
    a = sys.argv
    if len(a) < 2 or a[1].startswith("-"):
        sys.exit("dp_study.py JOBDIR [--counts COUNTS.json] [--latin LATIN.json] [--fp4blocks-ms MS] [--accept-ms MS] "
                 "[--woa-ms MS] [--window W] [--gate DIR] [--pld DIR]")
    job = a[1]
    fp4b = float(opt(a, "--fp4blocks-ms", FP4BLOCKS_MS))
    acc_ms = float(opt(a, "--accept-ms", ACCEPT_MS))
    woa_ms = float(opt(a, "--woa-ms", WOA_MS))
    seqs, trs, bts, offs = load(job, opt(a, "--gate", GATE), opt(a, "--pld", PLD))
    groups = [g for g in ORDER if any(group_of_seq(s) == g for s in seqs)]
    recs = {v: P.load_spectf(os.path.join(job, "tf_%s.log" % v)) for v, _ in SIMS
            if os.path.exists(os.path.join(job, "tf_%s.log" % v))}
    probes = {v: VS.load_probe(os.path.join(job, "tp_%s.jsonl" % v)) for v in recs}

    # ---- the records
    print("### The records: spectf steps per variant; drafts differing from today's (any of the five)")
    print("| variant | | steps | " + " | ".join(groups) + " |")
    print("|---|---|---|" + "---|" * len(groups))
    base = recs.get("none", {})
    for v, desc in SIMS:
        if v not in recs:
            continue
        n = sum(len(x) for x in recs[v].values())
        cells = []
        for g in groups:
            tot = dif = 0
            for s in seqs:
                if group_of_seq(s) != g:
                    continue
                for i, rr in recs[v].get(s["seq"], {}).items():
                    b = base.get(s["seq"], {}).get(i)
                    if b is not None:
                        tot += 1
                        dif += list(rr[0]) != list(b[0])
            cells.append("%.1f %%" % (100.0 * dif / tot) if tot else "-")
        print("| %s | %s | %d | " % (v, desc, n) + " | ".join(cells) + " |")
    # today's records against the gate study's (the same code, so the same bits)
    same = tot = 0
    for s in seqs:
        tr = trs[s["seq"]]
        for i, rr in base.get(s["seq"], {}).items():
            if i in tr.recs:
                tot += 1
                same += list(tr.recs[i][0]) == list(rr[0]) and all(
                    abs(x - y) < 1e-4 for x, y in zip(tr.recs[i][1], rr[1]))
    print("\ntoday's records against the earlier runs' on the same sequences: %d of %d the same" % (same, tot))

    # ---- the pass's time
    T = timings(os.path.join(job, "b_vrows.log"))
    print("\n### The pass by vocabulary rows (bench_dspark --fp8 1 --vrows; one rank, stand-in collectives), ms")
    print("| rows | draft (blocks, head, chain) | accept 1 + draft | accept 6 + draft | the head GEMV alone |")
    print("|---|---|---|---|---|")
    for n in sorted(T, reverse=True):
        t = T[n]
        print("| %d | %.3f | %.3f | %.3f | %.3f |" % (n, t.get("draft", 0), t.get("a1", 0), t.get("a6", 0),
                                                       t.get("head", 0)))
    full, half = 32320, 16192

    def at(n, k="draft"):  # T[n][k], linear between the measured row counts
        ns = sorted(T)
        if n in T:
            return T[n][k]
        lo = max(x for x in ns if x < n)
        hi = min(x for x in ns if x > n)
        return T[lo][k] + (T[hi][k] - T[lo][k]) * (n - lo) / (hi - lo)

    chain = lambda n: T[n]["draft"] - T[n]["head"]  # the blocks, the chain, hc_pre and the rest
    s_head4 = T[full]["head"] - T[half]["head"]
    s_markov8 = chain(full) - chain(half)
    s_latin = T[full]["draft"] - T[21376]["draft"]
    s_latin4 = T[half]["draft"] - T[10688]["draft"]  # on an FP4 head and an FP8 Markov head
    per_row = (T[full]["draft"] - T[4096]["draft"]) / (full - 4096)
    sav = {"none": 0.0, "fp4blocks": fp4b, "fp4head": s_head4, "fp8markov": s_markov8,
           "all": fp4b + s_head4 + s_markov8}
    print("\nsavings, ms a pass: FP4 head %.3f (the head at %d rows against %d); FP8 Markov head %.3f (the rest of the"
          " pass at half the rows); FP4 blocks %.3f (modelled); the Latin subset %.3f (%d rows a rank at most), %.3f on"
          " the FP4 head and FP8 Markov head; a row of head and chain %.2f us; accept beside the verify %.3f and"
          " wo_a %.3f (modelled)" % (s_head4, half, full, s_markov8, fp4b, s_latin, 21376, s_latin4, 1e3 * per_row,
                                     acc_ms, woa_ms))

    # ---- the replay per variant
    p0 = {}
    r0 = replay(seqs, trs, bts, offs, recs["none"], 0.0, p0)
    print("\n### Today's chain replayed per variant: accepted drafts a step; the gain in tok/s with the variant's "
          "saving")
    print("| variant (saving) | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * len(groups))
    print("| none: acc/step, tok/s | " + " | ".join(
        "%.2f, %.1f" % (acc(r0, g), 1e3 * r0[g][0] / r0[g][1] if r0[g][2] else float("nan")) for g in groups) + " |")
    res = {"none": r0}
    for v, _ in SIMS[1:]:
        if v not in recs:
            continue
        r = replay(seqs, trs, bts, offs, recs[v], sav[v])
        pq = {}
        rq = replay(seqs, trs, bts, offs, recs[v], 0.0, pq)  # its acceptance alone
        sd = {g: spread(seqs, p0, pq, g) for g in groups}
        res[v] = r
        print("| %s (-%.2f ms) | " % (v, sav[v]) + " | ".join(
            "%.2f, %+.2f %% (%+.2f %%%s at no saving)" % (acc(r, g), gain(r, r0, g), gain(rq, r0, g),
                                                          " +- %.2f" % sd[g] if sd[g] is not None else "")
            for g in groups) + " |")

    # ---- vocabulary subsets from the probe records: on today's drafter, and in bundles on all three's records
    subsets = []  # (name, ids mask, saving on today's pass, saving on the FP4 head and FP8 Markov head)
    lat = opt(a, "--latin", "")
    if lat:
        ins = bytearray(V)
        for t in json.load(open(lat))["ids"]:
            ins[t] = 1
        subsets.append(("the Latin subset (%d ids)" % sum(ins), ins, s_latin, s_latin4))
    cnt = opt(a, "--counts", "")
    if cnt:
        order, _ = VS.ranking(cnt, ["repo", "bench", "catalogs", "golden", "pubcode", "pubprose"])
        for n, rows4, rows_half in ((32768, 8192, 4096), (65536, 16384, 8192)):
            ins = bytearray(V)
            for t in order[:n]:
                ins[t] = 1
            subsets.append(("the %dk most frequent, interleaved" % (n // 1024), ins,
                            at(full) - at(rows4), at(half) - at(rows_half)))


    def run_row(rr, s, pen=None, use=None, s_full=0.0):
        """{group: [tok, ms, steps]} on records rr, the pass s ms shorter (less pen[seq], the context's rows);
        with use ({seq: {i: bool}}), a step that does not use the subset saves s_full instead."""
        r = collections.defaultdict(lambda: [0.0, 0.0, 0])
        for sq in seqs:
            k = sq["seq"]
            tr, rk = trs[k], rr.get(k)
            if not rk:
                continue
            saved = tr.recs
            tr.recs = rk
            cost = dict(P.COST)
            cost["c0"] += offs.get(tr.name, 0.0)
            bd, sk = cost["draft"], s - (pen or {}).get(k, 0.0)
            if use is None:
                cost["draft"] = bd - sk
                pol = P.P_chain()
            else:
                cost["draft"] = bd - s_full

                def pol(tr_, i, lam, st, uk=use[k], bd=bd, sk=sk):  # the walk's cost dict is st["cost"]
                    st["cost"]["draft"] = bd - (sk if uk.get(i) else s_full)
                    st["src"] = "c" * P.NDEP
                    return P.chain_rows(tr_, i, lam), True
            w = P.walk(tr, pol, cost, bts[P.group_of(tr.name)])
            tr.recs = saved
            g = r[group_of_seq(sq)]
            g[0] += w["tok"]
            g[1] += w["ms"]
            g[2] += w["steps"]
        return r

    def table(title, rows):
        print("\n### " + title)
        print("| (saving a pass) | " + " | ".join(groups) + " |")
        print("|---|" + "---|" * len(groups))
        for nm, kw in rows:
            r = run_row(**kw)
            print("| %s (-%.2f ms) | " % (nm, kw["s"]) + " | ".join("%+.2f %%" % gain(r, r0, g) for g in groups) + " |")

    W = int(opt(a, "--window", 128))

    def subset_rows(v, prefix, s0, half_bytes):
        rows, shares = [], []
        for nm, ins, s_plain, s4 in subsets:
            sv = s0 + (s4 if half_bytes else s_plain)
            for ctx in (False, True):
                sr, extra = subset_recs(seqs, trs, recs[v], probes.get(v, {}), ins, ctx)
                # the context's rows outside the subset cost their share of the largest rank's head and chain
                # (~1.2x a quarter), at half a row's bytes on the FP4 head and FP8 Markov head
                pen = {k: 1.2 * e / 4 * per_row * (0.5 if half_bytes else 1.0) for k, e in extra.items()} if ctx else {}
                rows.append(("%s%s%s" % (prefix, nm, ", with the context" if ctx else ""),
                             dict(rr=sr, s=sv, pen=pen)))
            # adaptive: the subset while the context's last W tokens are all in it, else the whole vocabulary
            sr, use = adaptive_recs(seqs, trs, recs[v], probes.get(v, {}), ins, W)
            rows.append(("%s%s, only while the last %d tokens are in it" % (prefix, nm, W),
                         dict(rr=sr, s=sv, use=use, s_full=s0)))
            shares.append((nm, use))
        return rows, shares

    def print_shares(shares):
        print("\nsteps on the subset (adaptive), by group: " + "; ".join(
            "%s: %s" % (nm, ", ".join("%s %.0f %%" % (g, 100.0 * share(use, g)) for g in groups))
            for nm, use in shares))

    def share(use, g):
        n = u = 0
        for sq in seqs:
            if group_of_seq(sq) == g:
                for x in use.get(sq["seq"], {}).values():
                    n += 1
                    u += x
        return u / n if n else float("nan")

    if subsets and "none" in probes:
        rows, shares = subset_rows("none", "", 0.0, False)
        table("Vocabulary subsets on today's drafter (none's probe records): gain in tok/s", rows)
        print_shares(shares)
    if "all" in recs:
        extra_ms = acc_ms + woa_ms
        rows = [("all three", dict(rr=recs["all"], s=sav["all"])),
                ("all three, accept beside the verify, wo_a", dict(rr=recs["all"], s=sav["all"] + extra_ms))]
        if subsets and "all" in probes:
            more, _ = subset_rows("all", "all three, accept, wo_a + ", sav["all"] + extra_ms, True)
            rows += more
        table("Bundles on all three's records, with the modelled levers and vocabulary subsets on top: gain in tok/s",
              rows)
    if "fp4head" in recs:
        # without the FP4 blocks: fp4head's records (fp8markov's acceptance is today's within its spread), wo_a at
        # FP8's bytes (twice the FP4 figure); the subsets' steps from today's probe records (an approximation: the
        # FP4 head moves 0.2 % of acceptance)
        s_nb = s_head4 + s_markov8 + acc_ms + 2 * woa_ms
        rows = [("the FP4 head, the FP8 Markov head, accept, wo_a", dict(rr=recs["fp4head"], s=s_nb))]
        for nm, ins, s_plain, s4 in subsets:
            if "none" in probes:
                sr, use = adaptive_recs(seqs, trs, recs["fp4head"], probes["none"], ins, W)
                rows.append(("the same + %s, only while the last %d tokens are in it" % (nm, W),
                             dict(rr=sr, s=s_nb + s4, use=use, s_full=s_nb)))
        table("Bundles without the FP4 blocks (fp4head's records): gain in tok/s", rows)

if __name__ == "__main__":
    main()
