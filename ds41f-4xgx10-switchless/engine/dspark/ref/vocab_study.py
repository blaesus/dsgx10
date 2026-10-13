#!/usr/bin/env python3
# The draft pass's vocabulary subset, a study (docs/engine.md, "The draft pass's approximations: a study"). The
# drafter's head GEMV and Markov bias would cover only the N most frequent tokens of public text (vocab_freq.py's
# counts; ties and unseen tokens in BPE merge order, i.e. by id) instead of all 129,280; its argmax runs over them.
#
# The drafter is first order: depth j's logits are the block's row j through the head plus the Markov bias of the
# previous id, so its draft at depth j depends only on that id, and the confidence at depth j only on it too. The
# probe records (spectf --tree-probe: the drafter's top 8 at each depth along the target's own tokens, and the
# confidence there) therefore give the subset drafter's choice exactly while its drafts are right: the first of the
# top 8 that is in the subset. Off the path a draft outside the subset counts as rejected, and a confidence that
# is not recorded (a rejected row's) is the chain's. Traces without probe records (the golden turns) take the
# lower bound: a chain draft outside the subset is rejected.
#
# Today's chain is replayed with pld_study.py's walk, gate and cost model (its betas and per-trace offsets), the
# draft pass shortened by the subset's saving (--save N:MS,...: measured; otherwise a bandwidth model).
# --context: each anchor's own context (the prompt and the output so far) joins the subset, as a per-session
# extension would; --subset FILE: a fixed subset (vocab_script.py's) in place of the frequency ranks.
#   vocab_study.py COUNTS.json [--sources a,b] [--sizes N,...] [--save N:MS,...] [--context] [--subset FILE]
#                  [--gate DIR] [--pld DIR] [--tree DIR]
import collections, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pld_study as P  # noqa: E402

V = 129280
NDEP = 5
SIZES = [8192, 16384, 24576, 32768, 49152, 65536]
GATE, PLD, TREE = "/data/dsgb10/bench/gate", "/data/dsgb10/kernels/pld", "/data/dsgb10/kernels/tree"
# the draft pass (engine.md, "The draft pass", FP8 head, 6 rows): the head GEMV over a rank's 32320 rows reads
# 165 MB in 0.66 ms; the Markov chain 0.21 ms: markov[0] ~90 us (16.5 MB from DRAM), four more ~19 us from L2,
# five argmaxes 4-10 us
HEAD_MS, MARKOV0_MS, MARKOVN_MS, ARGMAX_MS = 0.66, 0.090, 0.019, 0.007
FIXED_MS = 0.008  # a launch's floor


def model_save(n):
    """ms the pass saves with a subset of n tokens, by bytes: the head and the Markov head scale with n; markov[0]
    from DRAM, the other four from L2 (a subset's Markov head stays in L2 too)."""
    f = n / V
    head = max(FIXED_MS, HEAD_MS * f)
    markov = max(FIXED_MS, MARKOV0_MS * f) + 4 * max(FIXED_MS / 2, MARKOVN_MS * f)
    argm = 5 * max(0.003, ARGMAX_MS * f)
    full = HEAD_MS + MARKOV0_MS + 4 * MARKOVN_MS + 5 * ARGMAX_MS
    return full - head - markov - argm


def ranking(path, sources):
    d = json.load(open(path))
    score = [0] * V
    for s in sources:
        for t, c in enumerate(d["sources"][s]["counts"]):
            score[t] += c
    order = sorted(range(V), key=lambda t: (-score[t], t))
    return order, score


def load_probe(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if "i" in r:  # the tree study's files also hold sibling records
                out[(r.get("seq", 0), r["i"])] = r
    return out


def sub_rec(r, ins):
    """The subset drafter's drafts and survival at a probe record: exact along the target's tokens, the chain's
    where its previous id is the chain's, -1 (rejected) where unknown. Also whether the record changed."""
    d, s, path, ids, confp = r["d"], r["s"], r["path"], r["id"], r["conf"]
    qc = [s[0]] + [s[j] / s[j - 1] if s[j - 1] > 0 else 0.0 for j in range(1, NDEP)]  # the chain's sigmoid(conf)
    dn, sv, surv = [], [], 1.0
    for j in range(NDEP):
        prev = None if j == 0 else dn[j - 1]
        if j == 0 or prev == d[j - 1]:
            q = qc[j]
        elif prev == path[j - 1]:
            q = 1.0 / (1.0 + math.exp(-confp[j]))
        else:
            q = qc[j]
        surv *= q
        sv.append(surv)
        if j == 0 or (j - 1 < len(path) and prev == path[j - 1]):
            c = next((t for t in ids[j] if ins[t]), -1) if j < len(ids) else -1
        elif prev == d[j - 1]:
            c = d[j] if ins[d[j]] else -1
        else:
            c = -1
        dn.append(c)
    return dn, sv


def lower_rec(rec, ins):
    d, s = rec
    return [t if ins[t] else -1 for t in d], list(s)


def load_traces(gate, pld, tree):
    """(group, trace, probe dict by spectf step or None): the bench types and the golden turns from the PLD
    study's records, the edit bench from the gate study's (r1 greedy: g1; r2 sampled: s0..s2)."""
    out = []
    trs, meta = P.load_all(pld)
    for tr in trs:
        if tr.name.startswith("bench-"):
            tp = load_probe(os.path.join(tree, "tp_%s.jsonl" % tr.name[6:]))
            out.append(("bench " + tr.name[6:], tr, {i: v for (sq, i), v in tp.items()}))
        elif tr.name.startswith("turn"):
            out.append(("golden turns", tr, None))
    gm = json.load(open(os.path.join(gate, "meta.json")))
    seq_of = {m["name"]: (run, m["seq"]) for run, ms in gm["runs"].items() for m in ms}
    tps = {run: load_probe(os.path.join(gate, "tp_%s.jsonl" % run)) for run in gm["runs"]}
    gtrs, _ = P.load_all(gate)
    for tr in gtrs:
        run, sq = seq_of[tr.name]
        grp = "edit r1 greedy" if run == "g1" else "edit r2 sampled"
        out.append((grp, tr, {i: v for (s2, i), v in tps[run].items() if s2 == sq}))
    return out, trs + gtrs, meta


def main():
    a = sys.argv
    if len(a) < 2 or a[1].startswith("-"):
        sys.exit("vocab_study.py COUNTS.json [--sources a,b] [--sizes N,...] [--save N:MS,...] [--context] "
                 "[--subset FILE] [--gate DIR] [--pld DIR] [--tree DIR]")
    opt = lambda k, dflt: a[a.index(k) + 1] if k in a else dflt
    sources = opt("--sources", "repo,bench,catalogs,golden").split(",")
    sizes = [int(x) for x in opt("--sizes", ",".join(map(str, SIZES))).split(",")]
    saves = {int(k): float(v) for k, v in (x.split(":") for x in opt("--save", "").split(",") if x)}
    order, score = ranking(a[1], sources)
    print("sources %s: %d distinct tokens seen, %d tokens" % (",".join(sources), sum(1 for c in score if c),
                                                               sum(score)))
    if "--subset" in a:  # a fixed subset (vocab_script.py's): its ids first, by frequency
        ids = set(json.load(open(opt("--subset", "")))["ids"])
        order = [t for t in order if t in ids] + [t for t in order if t not in ids]
        sizes = [len(ids)]
        print("the subset %s: %d ids" % (opt("--subset", ""), len(ids)))
    traces, alltr, meta = load_traces(opt("--gate", GATE), opt("--pld", PLD), opt("--tree", TREE))
    bts = P.betas(alltr)
    offs = P.offsets(alltr, meta, bts)
    groups = collections.OrderedDict()
    for g, tr, tp in traces:
        groups.setdefault(g, []).append((tr, tp))
    print("traces: " + ", ".join("%s %d" % (g, len(v)) for g, v in groups.items()))

    # coverage of the outputs
    print("\n### Output tokens outside the subset (share of the targets' own outputs)")
    print("| group | outputs | " + " | ".join("N = %d" % n for n in sizes) + " |")
    print("|---|---|" + "---|" * len(sizes))
    masks = {}
    for n in sizes:
        m = bytearray(V)
        for t in order[:n]:
            m[t] = 1
        masks[n] = m
    for g, lst in groups.items():
        outs = [t for tr, _ in lst for t in tr.toks[tr.P:]]
        print("| %s | %d | " % (g, len(outs)) + " | ".join(
            "%.3f %%" % (100.0 * sum(1 for t in outs if not masks[n][t]) / max(1, len(outs))) for n in sizes) + " |")

    # the replay
    print("\n### Today's chain replayed (pld_study.py's walk and costs): accepted drafts a step, tok/s by the model")
    hdr = "| group | base: acc/step, tok/s | " + " | ".join(
        "N = %d (-%.2f ms): acc/step, tok/s, gain" % (n, saves.get(n, model_save(n))) for n in sizes) + " |"
    print(hdr)
    print("|---|---|" + "---|" * len(sizes))
    tot = {n: [0.0, 0.0] for n in [0] + sizes}
    ctx = "--context" in a
    extra = {}
    for g, lst in groups.items():
        cells = []
        res = {}
        for n in [0] + sizes:
            tok = ms = steps = acc = 0.0
            changed = rec_n = 0
            for tr, tp in lst:
                cost = dict(P.COST)
                cost["c0"] += offs.get(tr.name, 0.0)
                recs0 = tr.recs
                if n:
                    cost["draft"] -= saves.get(n, model_save(n))
                    nr = {}
                    m = bytearray(masks[n]) if ctx else masks[n]
                    seen = 0  # the context's tokens added to m: toks[:seen]
                    for i in sorted(recs0):
                        rec = recs0[i]
                        if ctx:  # the anchor's context: the prompt and the output through it (i + 1)
                            while seen <= min(i + 1, tr.L - 1):
                                m[tr.toks[seen]] = 1
                                seen += 1
                        if tp is not None and i in tp:
                            nr[i] = sub_rec(tp[i], m)
                        else:
                            nr[i] = lower_rec(rec, m)
                        rec_n += 1
                        changed += list(nr[i][0]) != list(rec[0])
                    if ctx:
                        extra.setdefault(n, []).append(sum(1 for t in set(tr.toks) if not masks[n][t]))
                    tr.recs = nr
                w = P.walk(tr, P.P_chain(), cost, bts[P.group_of(tr.name)])
                tr.recs = recs0
                tok += w["tok"]
                ms += w["ms"]
                steps += w["steps"]
            res[n] = (tok, ms, steps)
            tot[n][0] += tok
            tot[n][1] += ms
        t0, m0, s0 = res[0]
        cells.append("%.2f, %.1f" % ((t0 - s0) / s0, 1e3 * t0 / m0))
        for n in sizes:
            t, m, s = res[n]
            cells.append("%.2f, %.1f, %+.2f %%" % ((t - s) / s, 1e3 * t / m, 100.0 * ((t / m) / (t0 / m0) - 1)))
        print("| %s | " % g + " | ".join(cells) + " |")
        if ctx:
            print("|   (context tokens outside the static subset, mean / max a trace) | | " + " | ".join(
                "%.0f / %d" % (sum(extra[n]) / len(extra[n]), max(extra[n])) for n in sizes) + " |")
            extra = {}

    # the subset's rows by rank: the head's own slices (id // 32320) against ids interleaved (id % 4)
    print("\n### The subset's rows by rank: the head's slices (id // 32320), or interleaved (id % 4)")
    for n in sizes:
        sl = collections.Counter(t // 32320 for t in order[:n])
        il = collections.Counter(t % 4 for t in order[:n])
        print("N = %d: slices %s (the largest %.2f of N / 4); interleaved %s" % (
            n, [sl[r] for r in range(4)], max(sl.values()) / (n / 4), [il[r] for r in range(4)]))


if __name__ == "__main__":
    main()
