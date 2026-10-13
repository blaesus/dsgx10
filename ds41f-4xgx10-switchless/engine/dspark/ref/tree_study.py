#!/usr/bin/env python3
# Draft trees, a feasibility study (docs/engine.md, "Draft trees: a feasibility study"): the offline analysis of
# the drafter's probe records (spectf --tree-probe FILE --tree-sib N on the target's own greedy generation of
# each bench prompt: DIR/tp_<type>.jsonl). It checks the records, then gives the drafter's ranks of the true
# tokens by depth; the routed experts of siblings against chain rows; today's chain under its gate, replayed;
# the chain and trees within 6, 8, 12 and 16 verify rows under per-node gates, and fixed: accepted drafts and
# tokens per step, rows, and tok/s by a cost model (tree_costs.py fits its constants). No numpy.
#   tree_study.py DIR [--costs FILE.json] [--types prose,code,...] [--sib-new measured|chain|zero] [--sweep]
#   tree_study.py DIR --tables [--costs FILE.json]   (the tables in docs/engine.md)
import json, math, os, sys, heapq, collections

USAGE = "tree_study.py DIR [--costs FILE.json] [--types a,b] [--sib-new measured|chain|zero] [--sweep] [--tables]"
D = "."  # the records' directory: the first argument
TYPES = ["prose", "code", "agentic", "json"]
PROMPT = {"prose": 39, "code": 36, "agentic": 55, "json": 51}  # the bench prompts' tokens (bench/bench.py)
K = 8          # dspark::kProbeK
NDEP = 5       # depths (drafts per pass)
GEN = 512      # tokens generated after the prompt
BUDGETS = [6, 8, 12, 16]  # verify rows: the bonus row and the draft nodes
C1, DC, CDRAFT = 19.4, 3.6, 3.3  # the gate's default costs (cmd.h spec_cost)
LAYERS = 40

# ---- cost model (ms of a spec step, its draft pass included) --------------------------------------------------
# Today's chain on the bench prompts, by rows (docs/engine.md, "Gates by a cost per row count"): rows 1..6.
T_CHAIN = [None, 21.1, 24.5, 28.0, 31.2, 34.5, 37.3]
COST = dict(  # tree_costs.py's fit (2026-10-12); --costs replaces them
    b_row=1.1,      # ms per row besides its experts (the engine's steps; bench_chain's kernels alone: 0.36)
    c_exp=0.713,    # ms per distinct routed expert per layer
    o16=3.09,       # the 9..16-row kernels' extra ms over the <= 8-row ones (bench_chain, B4's kernels)
    sib_new=[3.0],  # distinct experts per layer a further node of a depth adds (the k-th: sib_new[k - 1], the
                    # last one repeating); set from the sibling records unless --sib-new chain / zero
    tree_draft=0.15,  # ms the tree's drafting adds to the draft pass (per-depth Markov bias and top-k for every
                      # parent, one gather per depth instead of the argmax): a rough estimate
)


def sib_ms(cost, k):  # the k-th further node of a depth (k >= 1)
    if cost.get("sib_flat") is not None:  # the sensitivity sweep: every further node at a flat cost
        return cost["sib_flat"]
    s = cost["sib_new"]
    return cost["b_row"] + cost["c_exp"] * s[min(k - 1, len(s) - 1)]


def tree_cost(counts, cost):
    """counts[j]: nodes at depth j + 1. The first node of each depth costs as a chain row (the measured chain
    step times), each further node of a depth as a sibling: b_row + c_exp * the experts it adds."""
    spine = sum(1 for n in counts if n > 0)
    t = T_CHAIN[1 + spine]
    for n in counts:
        for k in range(1, n):
            t += sib_ms(cost, k)
    if 1 + sum(counts) > 8:
        t += cost["o16"]
    if any(n > 1 for n in counts):
        t += cost["tree_draft"]
    return t


# ---- data ----------------------------------------------------------------------------------------------------
def load(t):
    recs, sibs = {}, []
    path = f"{D}/tp_{t}.jsonl"
    if not os.path.exists(path):
        return None, None
    for line in open(path):
        r = json.loads(line)
        if "sib_at" in r:
            sibs.append(r)
        else:
            recs[r["i"]] = r
    return recs, sibs


def lead(d, path):
    m = 0
    while m < NDEP and path[m] >= 0 and d[m] == path[m]:
        m += 1
    return m


def check(t, recs):
    n = bad_d1 = bad_conf = bad_chain = bad_lead = bad_p = 0
    agree = 0
    for i, r in recs.items():
        n += 1
        if r["d"][0] != r["id"][0][0]:
            bad_d1 += 1
        s0 = 1.0 / (1.0 + math.exp(-r["conf"][0]))
        if abs(s0 - r["s"][0]) > 1e-5:
            bad_conf += 1
        for j in range(1, NDEP):
            if r["path"][j - 1] >= 0 and all(r["d"][q] == r["path"][q] for q in range(j)) and r["d"][j] != r["id"][j][0]:
                bad_chain += 1
        m = lead(r["d"], r["path"])
        m2 = 0
        while m2 < NDEP and r["rank"][m2] == 0:
            m2 += 1
        if m != m2:
            bad_lead += 1
        for j in range(NDEP):
            ps = r["p"][j]
            if sum(ps) > 1.0001 or any(ps[q] < ps[q + 1] - 1e-7 for q in range(K - 1)):
                bad_p += 1
        agree += r["sample"] == r["bonus"]
    print(f"{t}: {n} records; d1 != the probe's top 1: {bad_d1}; conf[0] != the draft's: {bad_conf}; the chain's "
          f"d_j+1 != the probe's top 1 after a matching prefix: {bad_chain}; leading matches != leading rank 0: "
          f"{bad_lead}; p rows unsorted or > 1: {bad_p}; the step's own sample == the generation's token: {agree}/{n}")


def rank_stats(t, recs, anchors):
    """P(rank r at depth j | the path's ranks above are < K), and the drafter's calibration at depth 1."""
    hist = [collections.Counter() for _ in range(NDEP)]
    for A in anchors:
        rho = recs[A]["rank"]
        for j in range(NDEP):
            if rho[j] < 0:
                break
            hist[j][min(rho[j], K)] += 1
            if rho[j] >= K:
                break
    print(f"  {t}: rank of the true token by depth (given every rank above < {K}; columns 0..{K - 1}, then >= {K}):")
    for j in range(NDEP):
        tot = sum(hist[j].values())
        if tot:
            print(f"    depth {j + 1} ({tot}): " + " ".join(f"{hist[j][r] / tot:.3f}" for r in range(K + 1)))
    # calibration of the top-1 probability at depth 1 (exact: the bonus's children)
    bins = [[0, 0, 0.0] for _ in range(5)]
    for A in anchors:
        r = recs[A]
        p0 = r["p"][0][0]
        b = min(int(p0 * 5), 4)
        bins[b][0] += 1
        bins[b][1] += r["rank"][0] == 0
        bins[b][2] += p0
    print("    depth 1 top-1 calibration (p bin: n, mean p, accepted): " +
          "; ".join(f"{q / 5:.1f}-{(q + 1) / 5:.1f}: {b[0]}, {b[2] / max(b[0], 1):.2f}, {b[1] / max(b[0], 1):.2f}"
                    for q, b in enumerate(bins) if b[0]))


def union_size(sets):
    u = set()
    for s in sets:
        u |= s
    return len(u)


def sib_stats(t, sibs):
    """Distinct experts per layer: n chain rows (golden, depths 1..n), n siblings (the probe's top n at depth 1),
    and the experts each further depth-1 sibling (rank order, the golden token skipped) adds to the 4 chain rows."""
    if not sibs:
        return None
    uc = [0.0] * 4
    us = [0.0] * 4
    new = [0.0] * 3
    nnew = [0] * 3
    for s in sibs:
        gold1 = s["golden"][0]
        extras = [k for k in range(4) if s["tokens"][k] != gold1][:3]
        for L in range(LAYERS):
            chain = [set(s["chain"][L][6 * q:6 * q + 6]) for q in range(4)]
            sib = [set(s["sib"][k][L]) for k in range(4)]
            for n in range(4):
                uc[n] += union_size(chain[:n + 1])
                us[n] += union_size(sib[:n + 1])
            base = union_size(chain)
            for q, k in enumerate(extras):
                u = union_size(chain + [sib[x] for x in extras[:q + 1]])
                new[q] += u - base
                nnew[q] += 1
                base = u
    m = len(sibs) * LAYERS
    uc = [x / m for x in uc]
    us = [x / m for x in us]
    new = [new[q] / nnew[q] if nnew[q] else float("nan") for q in range(3)]
    print(f"  {t}: {len(sibs)} sibling samples; distinct experts per layer, 1..4 chain rows: " +
          " ".join(f"{x:.1f}" for x in uc) + "; 1..4 siblings: " + " ".join(f"{x:.1f}" for x in us) +
          "; added to 4 chain rows by the 1st..3rd further sibling: " + " ".join(f"{x:.2f}" for x in new))
    return dict(uc=uc, us=us, new=new)


# ---- policies: (record, lam) -> (accepted drafts, rows, step ms) ---------------------------------------------
def chain_policy(thresh=None):
    def f(r, lam):
        th = thresh if thresh is not None else lam * DC
        k = 0
        while k < NDEP and r["s"][k] >= th:
            k += 1
        a = min(k, lead(r["d"], r["path"]))
        return a, 1 + k, T_CHAIN[1 + k]
    return f


def chain_fixed(k):  # k drafts every step, no gate
    def f(r, lam):
        return min(k, lead(r["d"], r["path"])), 1 + k, T_CHAIN[1 + k]
    return f


def accepted(rho, nodes):
    a, v = 0, ()
    for j in range(NDEP):
        if rho[j] < 0 or rho[j] >= K:
            break
        v = v + (rho[j],)
        if v not in nodes:
            break
        a = j + 1
    return a


def depth_counts(nodes):
    c = [0] * NDEP
    for v in nodes:
        c[len(v) - 1] += 1
    return c


def static_policy(nodes, cost):
    nodes = frozenset(nodes)
    t = tree_cost(depth_counts(nodes), cost)
    def f(r, lam):
        return accepted(r["rank"], nodes), 1 + len(nodes), t
    return f


def best_nodes(r, budget, score):
    """The `budget` nodes of the highest score, best first; score must not grow along a path, so the set is
    prefix-closed. Children of v: v + (k,), k < K."""
    out, h = [], []
    for k in range(K):
        heapq.heappush(h, (-score(r, (k,)), (k,)))
    while h and len(out) < budget:
        s, v = heapq.heappop(h)
        out.append((-s, v))
        if len(v) < NDEP:
            for k in range(K):
                c = v + (k,)
                heapq.heappush(h, (-score(r, c), c))
    return out


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


# Node scores. The probe has the drafter's distributions along the true path: exact for the nodes whose parent
# is on it (every depth-1 node, the true node's children, ...); a node below an off-path node gets its depth's
# values from the path (an approximation, which only decides how many off-path nodes compete for the rows).
def p_score(r, v):  # the drafter's probability of the node: the product of its ranks' probabilities
    s = 1.0
    for j, k in enumerate(v):
        s *= r["p"][j][k]
    return s


def conf_score(r, v):  # the confidence head's P(top 1 accepted) at each depth, times p(rank) / p(top 1)
    s = 1.0
    for j, k in enumerate(v):
        s *= sig(r["conf"][j]) * r["p"][j][k] / max(r["p"][j][0], 1e-12)
    return s


def p_chain(r, v):  # the chain alone (every rank 0) under p_score
    return 0.0 if any(v) else p_score(r, v)


def conf_chain(r, v):  # the chain alone under conf_score: today's survival
    return 0.0 if any(v) else conf_score(r, v)


_BN = {}  # best_nodes per (record, budget, score): the same for every cost and alpha


def dyn_policy(budget, cost, score, alpha=None):
    """At most budget - 1 nodes by score (EAGLE-2-like); with alpha, only nodes whose score >= alpha * lam *
    their row's ms (today's gate, per node)."""
    def f(r, lam):
        key = (id(r), budget, score.__name__)
        if key not in _BN:
            _BN[key] = best_nodes(r, budget - 1, score)
        sel = _BN[key]
        if alpha is not None:
            keep, kept, counts = [], set(), [0] * NDEP
            for s, v in sel:
                d = len(v) - 1
                row_ms = (T_CHAIN[2 + d] - T_CHAIN[1 + d]) if counts[d] == 0 else sib_ms(cost, counts[d])
                if s >= alpha * lam * row_ms and (len(v) == 1 or v[:-1] in kept):
                    keep.append((s, v))
                    kept.add(v)
                    counts[d] += 1
            sel = keep
        nodes = frozenset(v for _, v in sel)
        return accepted(r["rank"], nodes), 1 + len(nodes), tree_cost(depth_counts(nodes), cost)
    return f


# ---- the anchor walk ------------------------------------------------------------------------------------------
class Rate:  # cmd.cpp SpecRate (the thresh gate)
    def __init__(s):
        s.tok, s.ms = 1.0, 0.0

    def add(s, emitted, rows):
        a = 1.0 / 16
        if s.ms == 0:
            s.ms = C1 + CDRAFT
        s.tok += a * (emitted - s.tok)
        s.ms += a * (C1 + (rows - 1) * DC + CDRAFT - s.ms)

    def lam(s):
        return s.tok / (s.ms if s.ms > 0 else C1 + CDRAFT)


def walk(recs, A0, A_end, policy):
    """Step 0 is the first spec step (one row at the prompt's end; it emits one token); step n's draft pass at
    anchor A_n decides step n + 1's rows with the rate after steps <= n - 2 (cmd.cpp spec_loop queues a step
    ahead). Today's lambda (the gate's costs) for every policy."""
    rate = Rate()
    lams = [rate.lam()]
    A, rows, e, ms = A0, 1, 1, T_CHAIN[1]
    tok = tms = 0.0
    n = steps = rows_sum = acc_sum = 0
    hist = collections.Counter()  # the verify steps (every step after step 0) by drafts verified
    while True:
        tok += e
        tms += ms
        steps += 1
        rows_sum += rows
        if steps > 1:
            hist[rows - 1] += 1
        acc_sum += e - 1
        rate.add(e, rows)
        lams.append(rate.lam())
        if A not in recs or A > A_end:
            break
        a, rows, ms = policy(recs[A], lams[max(0, n - 1)])
        n += 1
        e = a + 1
        A = A + 1 + a
    return dict(tok=tok, ms=tms, steps=steps, rows=rows_sum / steps, acc=acc_sum / steps, tps=1e3 * tok / tms,
                per_step=tok / steps, ms_step=tms / steps, hist=[hist[k] for k in range(max(list(hist) + [NDEP]) + 1)])


def prefix_counts(recs, anchors):
    cnt = collections.Counter()
    for A in anchors:
        rho = recs[A]["rank"]
        v = ()
        for j in range(NDEP):
            if rho[j] < 0 or rho[j] >= K:
                break
            v = v + (rho[j],)
            cnt[v] += 1
    return cnt


def static_tree(cnt, nodes_budget):
    """The nodes_budget most frequent rank prefixes (prefix-closed: a prefix is at least as frequent)."""
    order = sorted(cnt.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    nodes = set()
    for v, c in order:
        if len(nodes) >= nodes_budget:
            break
        if len(v) == 1 or v[:-1] in nodes:
            nodes.add(v)
    return nodes


def fmt_tree(nodes):
    return " ".join("".join(str(k) for k in v) for v in sorted(nodes, key=lambda v: (len(v), v)))


HONEST = ("staticX", "dynp", "dync", "gatep", "gatec")  # trees chosen without the type's own outcomes
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # ms per further node of a depth


def gated(recs, A0, A_end, B, c, sc, alph, name):
    """The per-node gate at every alpha (kept in alph[name]); returns the best one."""
    alph[name] = {}
    for alpha in ALPHAS:
        w = walk(recs, A0, A_end, dyn_policy(B, c, sc, alpha))
        w.update(alpha=alpha, B=B)
        alph[name][alpha] = w
    return max(alph[name].values(), key=lambda w: w["tps"])


def evaluate(recs, anchors, A0, A_end, cnt, other, c, alph=None):
    alph = {} if alph is None else alph
    res = {"chain": walk(recs, A0, A_end, chain_policy())}
    for name, sc in (("cgatep", p_chain), ("cgatec", conf_chain)):
        res[name] = gated(recs, A0, A_end, 1 + NDEP, c, sc, alph, name)
    for k in range(1, NDEP + 1):
        w = walk(recs, A0, A_end, chain_fixed(k))
        w["Ea"] = sum(min(k, lead(recs[A]["d"], recs[A]["path"])) for A in anchors) / len(anchors)
        w["B"] = 1 + k
        res[f"chain{k}"] = w
    for B in BUDGETS:
        for name, cn in (("static", cnt), ("staticX", other)):
            if not cn:
                continue
            nodes = static_tree(cn, B - 1)
            w = walk(recs, A0, A_end, static_policy(nodes, c))
            w.update(tree=fmt_tree(nodes), B=B,
                     Ea=sum(accepted(recs[A]["rank"], nodes) for A in anchors) / len(anchors))
            res[f"{name}{B}"] = w
        for name, sc in (("dynp", p_score), ("dync", conf_score)):
            pol = dyn_policy(B, c, sc)
            w = walk(recs, A0, A_end, pol)
            w["B"] = B
            w["Ea"] = sum(pol(recs[A], 0.0)[0] for A in anchors) / len(anchors)
            res[f"{name}{B}"] = w
        for name, sc in (("gatep", p_score), ("gatec", conf_score)):
            res[f"{name}{B}"] = gated(recs, A0, A_end, B, c, sc, alph, f"{name}{B}")
    return res


def best_key(res, B=None):
    ks = [k for k, w in res.items() if k.rstrip("0123456789") in HONEST and (B is None or w["B"] == B)]
    return max(ks, key=lambda k: res[k]["tps"]) if ks else None


def type_cost(cost, sib_mode, sstat):
    c = dict(cost)
    if sib_mode == "measured" and sstat:
        c["sib_new"] = [x for x in sstat["new"] if x == x] or c["sib_new"]
    elif sib_mode == "chain":
        c["sib_new"] = [(3.3 - c["b_row"]) / c["c_exp"]]  # a further node as dear as a chain row
    elif sib_mode == "zero":
        c["sib_new"] = [0.0]
    return c


BENCH = {"prose": (1.81, 1.43, 69.5), "code": (3.17, 3.03, 102.1), "agentic": (3.09, 2.91, 99.9),
         "json": (4.90, 4.43, 136.3)}  # tok/step, drafts verified per step, tok/s (docs/engine.md, the bench)
BENCH_HIST = {"prose": "29, 131, 97, 21, 4, 0", "code": "2, 25, 42, 29, 27, 36", "agentic": "1, 32, 46, 27, 17, 42",
              "json": "0, 8, 3, 5, 8, 80"}  # the engine's report: steps verifying 0..5 drafts
NAME = {"prose": "prose", "code": "code", "agentic": "agentic", "json": "JSON"}


def pct(x, base):
    return f"{100 * (x / base - 1):+.1f} %"


def tables(cost):
    """The markdown tables of docs/engine.md's section (every type, the measured sibling costs)."""
    data, sib, win = {}, {}, {}
    for t in TYPES:
        recs, sibs = load(t)
        data[t], sib[t] = recs, sibs
        P = PROMPT[t]
        A0, A_end = P, P + GEN - 2 - NDEP
        win[t] = (A0, A_end, [a for a in sorted(recs) if A0 <= a <= A_end])
    sstat = {t: sib_stats(t, sib[t]) for t in data}
    cnts = {t: prefix_counts(data[t], win[t][2]) for t in data}
    res, alph, cs = {}, {}, {}
    for t in data:
        other = collections.Counter()
        for u in data:
            if u != t:
                other.update(cnts[u])
        c = type_cost(cost, "measured", sstat[t])
        cs[t] = c
        alph[t] = {}
        res[t] = evaluate(data[t], win[t][2], win[t][0], win[t][1], cnts[t], other, c, alph[t])

    print("\n## validation: today's gate replayed against the bench\n")
    print("| | tok/step | drafts verified per step | steps verifying 0, 1, 2, 3, 4, 5 drafts | tok/s, the cost model |")
    print("|---|---|---|---|---|")
    for t in data:
        w, b = res[t]["chain"], BENCH[t]
        print(f"| {NAME[t]} | {w['per_step']:.2f} ({b[0]:.2f}) | {w['rows'] - 1:.2f} ({b[1]:.2f}) | "
              f"{', '.join(str(x) for x in w['hist'])} ({BENCH_HIST[t]}) | {w['tps']:.1f} ({b[2]:.1f}) |")

    print("\n## ranks: the true token's rank at depths 1 and 2 (given the path above it in the top 8)\n")
    print("| | depth | 0 | 1 | 2 | 3 | 4–7 | ≥ 8 |")
    print("|---|---|---|---|---|---|---|---|")
    for t in data:
        for j in range(2):
            h = collections.Counter()
            for a in win[t][2]:
                rho = data[t][a]["rank"]
                if any(rho[q] < 0 or rho[q] >= K for q in range(j)) or rho[j] < 0:
                    continue
                h[min(rho[j], K)] += 1
            n = sum(h.values())
            f = [h[r] / n for r in range(K + 1)]
            print(f"| {NAME[t] if j == 0 else ''} | {j + 1} | {f[0]:.3f} | {f[1]:.3f} | {f[2]:.3f} | {f[3]:.3f} | "
                  f"{sum(f[4:8]):.3f} | {f[8]:.3f} |")

    print("\n## siblings' routed experts (distinct per layer, mean over 40 layers and 32 samples)\n")
    print("| | 1–4 chain rows | 1–4 depth-1 siblings | added to 4 chain rows by the 1st, 2nd, 3rd sibling | "
          "a sibling row, ms |")
    print("|---|---|---|---|---|")
    for t in data:
        s = sstat[t]
        print(f"| {NAME[t]} | {' '.join(f'{x:.1f}' for x in s['uc'])} | {' '.join(f'{x:.1f}' for x in s['us'])} | "
              f"{' '.join(f'{x:.2f}' for x in s['new'])} | "
              f"{', '.join(f'{sib_ms(cs[t], k):.2f}' for k in range(1, 4))} |")

    print("\n## E[a]: accepted drafts per step over every anchor, by shape, the gate off\n")
    print("| | chain, 5 drafts (6 rows) | tree ≤ 6 rows | ≤ 8 | ≤ 12 | ≤ 16 | today's gated chain (rows) |")
    print("|---|---|---|---|---|---|---|")
    for t in data:
        r = res[t]
        row = f"| {NAME[t]} | {r['chain5']['Ea']:.2f} |"
        for B in BUDGETS:
            k = max((f"{n}{B}" for n in ("staticX", "dynp", "dync")), key=lambda k: r[k]["Ea"])
            row += f" {r[k]['Ea']:.2f} |"
        print(row + f" {r['chain']['acc']:.2f} ({r['chain']['rows']:.2f}) |")

    print("\n## tok/s by the cost model: each column one policy (score and alpha) for every type, the best sum of "
          "log gains over the four\n")
    def pick(cands):  # [(name, alpha)] -> the one with the best sum of log gains over the types
        return max(cands, key=lambda na: sum(math.log(alph[t][na[0]][na[1]]["tps"] / res[t]["chain"]["tps"])
                                             for t in data))
    cols = [("chain, per-node gate", pick([(n, al) for n in ("cgatep", "cgatec") for al in ALPHAS]))]
    for B in BUDGETS:
        cols.append((f"tree ≤ {B} rows", pick([(f"gate{s}{B}", al) for s in "pc" for al in ALPHAS])))
    print("| | today's chain | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for t in data:
        base = res[t]["chain"]["tps"]
        row = f"| {NAME[t]} | {base:.1f} ({res[t]['chain']['per_step']:.2f}, {res[t]['chain']['rows']:.2f}) |"
        for _, (n, al) in cols:
            w = alph[t][n][al]
            row += f" {w['tps']:.1f}, {pct(w['tps'], base)} ({w['per_step']:.2f}, {w['rows']:.2f}) |"
        print(row)
    print("\npolicies: " + ", ".join(f"{c}: {n} alpha {al}" for c, (n, al) in cols))
    print("\nper-type alpha (in-sample): " + "; ".join(
        f"{NAME[t]}: chain gate {pct(max(res[t]['cgatep']['tps'], res[t]['cgatec']['tps']), res[t]['chain']['tps'])}, "
        f"tree {pct(max(res[t][k]['tps'] for k in res[t] if k.rstrip('0123456789') in HONEST), res[t]['chain']['tps'])}"
        for t in data))
    print("\nfixed trees (no gate), tok/s: " + "; ".join(
        f"{NAME[t]}: " + ", ".join(
            f"{B} rows {max(res[t][f'{n}{B}']['tps'] for n in ('staticX', 'dynp', 'dync')):.1f}" for B in BUDGETS)
        for t in data))
    print("\nchain of k drafts, no gate, tok/s: " + "; ".join(
        f"{NAME[t]}: " + " ".join(f"{res[t][f'chain{k}']['tps']:.1f}" for k in range(1, 6)) for t in data))

    print("\n## sweep: every further node of a depth at a flat X ms; the best tree (per-type alpha) against today's "
          "chain, and against the best chain gate in brackets\n")
    print("| X, ms | " + " | ".join(NAME[t] for t in data) + " |")
    print("|---|" + "---|" * len(data))
    for X in SWEEP + ["free"]:
        row = f"| {X} |"
        for t in data:
            other = collections.Counter()
            for u in data:
                if u != t:
                    other.update(cnts[u])
            c = dict(cs[t], sib_flat=0.0 if X == "free" else X)
            if X == "free":
                c["o16"], c["tree_draft"] = 0.0, 0.0
            r = evaluate(data[t], win[t][2], win[t][0], win[t][1], cnts[t], other, c)
            k = best_key(r)
            cg = max(r["cgatep"]["tps"], r["cgatec"]["tps"])
            row += f" {pct(r[k]['tps'], r['chain']['tps'])} ({pct(r[k]['tps'], cg)}) |"
        print(row)


def main():
    global D
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        sys.exit(USAGE)
    D = args.pop(0)
    cost = dict(COST)
    types = TYPES
    sib_mode = "measured"
    sweep = only_tables = False
    while args:
        a = args.pop(0)
        if a == "--costs":
            cost.update(json.load(open(args.pop(0))))
        elif a == "--types":
            types = args.pop(0).split(",")
        elif a == "--sib-new":
            sib_mode = args.pop(0)
        elif a == "--sweep":
            sweep = True
        elif a == "--tables":
            only_tables = True
        else:
            sys.exit(USAGE)
    if only_tables:
        tables(cost)
        return
    print("costs:", {k: v for k, v in cost.items() if k != "sib_new"})
    data, sib = {}, {}
    for t in types:
        recs, sibs = load(t)
        if recs is None:
            print(f"{t}: no records")
            continue
        check(t, recs)
        data[t] = recs
        sib[t] = sibs
    print()
    win = {}
    for t, recs in data.items():
        P = PROMPT[t]
        A0, A_end = P, P + GEN - 2 - NDEP  # every path token of the last anchor exists
        win[t] = (A0, A_end, [A for A in sorted(recs) if A0 <= A <= A_end])
        rank_stats(t, recs, win[t][2])
    print()
    sstat = {t: sib_stats(t, sib[t]) for t in data}
    print()
    cnts = {t: prefix_counts(data[t], win[t][2]) for t in data}
    others = {}
    for t in data:
        o = collections.Counter()
        for u in data:
            if u != t:
                o.update(cnts[u])
        others[t] = o
    out = {}
    for t, recs in data.items():
        A0, A_end, anchors = win[t]
        c = type_cost(cost, sib_mode, sstat.get(t))
        alph = {}
        res = evaluate(recs, anchors, A0, A_end, cnts[t], others[t], c, alph)
        out[t] = dict(res=res, sib_new=c["sib_new"], alph=alph)
        base = res["chain"]["tps"]
        print(f"== {t}: {len(anchors)} anchors; a further node of a depth costs " +
              ", ".join(f"{sib_ms(c, k):.2f}" for k in range(1, 4)) + " ms (1st, 2nd, 3rd)")
        for k, w in res.items():
            extra = f"  tree {w['tree']}" if "tree" in w else ""
            extra += f"  (E[a] over every anchor {w['Ea']:.2f})" if "Ea" in w else ""
            extra += f"  alpha {w['alpha']}" if "alpha" in w else ""
            print(f"  {k:9s} {w['per_step']:.2f} tok/step, {w['acc']:.2f} accepted, {w['rows']:.2f} rows, "
                  f"{w['ms_step']:.2f} ms/step, {w['tps']:.1f} tok/s ({100 * (w['tps'] / base - 1):+.1f} %){extra}")
    print("\nsummary (" + sib_mode + "): today's chain, then the best tree not fitted to the type's own outcomes "
          "(staticX, dynp, dync, gatep, gatec) within 6, 8, 12, 16 rows: tok/step / rows, tok/s")
    for t in out:
        res = out[t]["res"]
        ch = res["chain"]
        row = f"  {t:8s} chain {ch['per_step']:.2f} tok/step, {ch['rows']:.2f} rows, {ch['tps']:.1f} tok/s"
        for k in ("cgatep", "cgatec"):
            w = res[k]
            row += (f" | {k} {w['per_step']:.2f}/{w['rows']:.1f} {w['tps']:.1f} "
                    f"({100 * (w['tps'] / ch['tps'] - 1):+.1f} %, alpha {w['alpha']})")
        for B in BUDGETS:
            k = best_key(res, B)
            w = res[k]
            row += (f" | {B}: {k} {w['per_step']:.2f}/{w['rows']:.1f} {w['tps']:.1f} "
                    f"({100 * (w['tps'] / ch['tps'] - 1):+.1f} %)")
        print(row)
    print("\naccepted drafts per step over every anchor, E[a], by shape (the gate off): the chain of 5 drafts (6 rows); "
          "then the best fixed tree within 6, 8, 12, 16 rows not fitted to the type (staticX, dynp, dync), "
          "and the type's own static tree in brackets")
    for t in out:
        res = out[t]["res"]
        row = f"  {t:8s} chain5 {res['chain5']['Ea']:.2f}"
        for B in BUDGETS:
            k = max((f"{n}{B}" for n in ("staticX", "dynp", "dync")), key=lambda k: res[k]["Ea"])
            row += f" | {B}: {res[k]['Ea']:.2f} {k} [{res[f'static{B}']['Ea']:.2f}]"
        print(row + f" | gated chain today: {res['chain']['acc']:.2f} per step at {res['chain']['rows']:.2f} rows")
    names = list(out[next(iter(out))]["alph"]) if out else []
    print("\none alpha for every type (the best sum of log gains over the types): the gain against today's chain")
    for name in names:
        best = max(ALPHAS, key=lambda al: sum(math.log(out[t]["alph"][name][al]["tps"] / out[t]["res"]["chain"]["tps"])
                                             for t in out))
        print(f"  {name:8s} alpha {best:4}: " + ", ".join(
            f"{t} {100 * (out[t]['alph'][name][best]['tps'] / out[t]['res']['chain']['tps'] - 1):+.1f} % "
            f"({out[t]['alph'][name][best]['per_step']:.2f}/{out[t]['alph'][name][best]['rows']:.1f})" for t in out))
        for t in out:
            out[t].setdefault("alpha1", {})[name] = dict(alpha=best, **{k: out[t]["alph"][name][best][k]
                                                                       for k in ("tps", "per_step", "rows", "ms_step")})
    for t in out:
        out[t]["alph"] = {n: {str(al): w["tps"] for al, w in d.items()} for n, d in out[t]["alph"].items()}
    if sweep:
        print("\nsweep: the best tree's tok/s against today's chain when every further node of a depth costs a flat "
              "X ms (o16 and the tree's drafting as above; 'free': X = 0, no o16, no drafting cost)")
        print("  X ms    " + "".join(f"{t:>24s}" for t in data))
        for X in SWEEP + ["free"]:
            row = f"  {X!s:7s} "
            for t, recs in data.items():
                A0, A_end, anchors = win[t]
                c = type_cost(cost, sib_mode, sstat.get(t))
                c["sib_flat"] = 0.0 if X == "free" else X
                if X == "free":
                    c["o16"], c["tree_draft"] = 0.0, 0.0
                res = evaluate(recs, anchors, A0, A_end, cnts[t], others[t], c)
                k = best_key(res)
                w = res[k]
                cg = max(res["cgatep"]["tps"], res["cgatec"]["tps"])
                row += (f"{k:>9s} {w['tps']:6.1f} {100 * (w['tps'] / res['chain']['tps'] - 1):+5.1f}%"
                        f" ({100 * (w['tps'] / cg - 1):+5.1f}%)")
                out[t].setdefault("sweep", {})[str(X)] = dict(best=k, tps=w["tps"], chain=res["chain"]["tps"],
                                                             rows=w["rows"], per_step=w["per_step"])
            print(row)
    json.dump(out, open(f"{D}/tree_{sib_mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
