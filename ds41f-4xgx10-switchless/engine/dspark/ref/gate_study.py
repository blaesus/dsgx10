#!/usr/bin/env python3
# The draft gate under sampling (docs/engine.md, "The draft gate under sampling"): an offline replay of the verify-
# length gate on spectf's records along served outputs, in DIR as gate_inputs.py and the ring passes leave it:
#   tf_RUN.log    the drafts d1..d5 and their survival at every anchor (spectf, one 1-row step per position);
#   tp_RUN.jsonl  the drafter's probe along the output (its top 8 per depth with probabilities) and the target's own
#                 top 20 logprobs per position (the recording build, runner-gate-rec);
#   rt_RUN.bin    each position's routed experts (--route-file, 8 rows at every 8th anchor).
# A gate only picks how many drafts a step verifies, so the outputs never change, and acceptance along the served
# output is exact for any gate: a sampled verify accepts d_j when the plain sampler's draw at its position is d_j,
# and that draw is the served token. The walk and the step cost are pld_study.py's: lambda lags two steps, and
# T = c0 + 1.10 (R - 1) + 0.713 (U - 6) + 2.55 (the draft pass), U the distinct experts of the step's own rows plus
# beta per rejected row, c0 offset per run set so that today's gate replays the served decode rate.
#   gate_study.py DIR [--check] [--cal] [--replay] [--sweep] [--tables] [--ss] [--sets name=RUN+RUN,...] [--only a,b]
# No numpy.
import collections, json, math, os, sys
import pld_study as S

USAGE = "gate_study.py DIR [--check] [--cal] [--replay] [--sweep] [--tables] [--ss] [--sets name=RUN+RUN,...] [--only a,b]"
NDEP = 5
TOP_P = 0.95                                     # dsfront's default sampling: T 1.0, top-p 0.95
DC = S.DC                                        # today's gate: survival[j - 1] >= lambda * dC (cmd.h spec_cost)
ROW_MS = [21.10, 25.38, 29.26, 32.76, 35.61, 38.70]  # --spec-row-ms's default: a step verifying 0..5 drafts
T_CHAIN = [None, 21.1, 24.5, 28.0, 31.2, 34.5, 37.3]  # the tree study's chain steps by rows (its probability gate)
BETA = {"edit": -0.002, "turns": 0.051, "bench": -0.020}  # pld_study.py --cal: experts a rejected draft row adds
EDGES = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]    # confidence buckets (the learned table, the calibration tables)
SETS = {"r2": ["s0", "s1", "s2"], "r2-train": ["s0"], "r2-test": ["s1", "s2"], "r1": ["g1"]}
ONLY = []  # --only a,b: only the candidate families whose names hold one of these


def sig(x):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def bucket(c):
    b = 0
    while b < len(EDGES) and c >= EDGES[b]:
        b += 1
    return b


# ---- loading -----------------------------------------------------------------------------------------------------
def load_tp(path):
    """{seq: {i: probe record}} (records of --tree-probe; sequence 0 without --seq-list)."""
    out = collections.defaultdict(dict)
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if "sib_at" not in r:
                out[r.get("seq", 0)][r["i"]] = r
    return out


def nucleus(ids, ps, top_p=TOP_P):
    """{id: P(top-p sampling at T 1 takes it)} over the top ids (their probabilities sorted down): the kept set is
    the shortest head reaching top_p; a head that does not reach it within the list stands for one that does."""
    kept, cum = {}, 0.0
    for t, p in zip(ids, ps):
        kept[t] = p
        cum += p
        if cum >= top_p:
            break
    z = max(cum, top_p)
    return {t: p / z for t, p in kept.items()}


def prepare(tr, greedy):
    """Per record i (anchor i + 1): a (drafts accepted along the output), c (the confidence head's conditional
    P(d_j accepted | d_1..d_j-1), from survival), p1 (the drafter's probability of d_j, the probe's top 1 along
    the output: exact through d_a+1), pn (that under top-p 0.95 on the drafter's top 8), q (the served sampler's
    P(d_j) at d_j's position given the output before it, from record i + j: known through d_a+1), tb (the target
    at the bonus's position, which the step before sampled: P(the bonus) and its most likely token's P, T = 1)."""
    tr.greedy = greedy
    tr.a, tr.c, tr.p1, tr.pn, tr.q = {}, {}, {}, {}, {}
    tr.nuc, tr.tb = {}, {}
    for i, r in tr.tp.items():
        if greedy:
            tr.nuc[i] = {r["tid"][0]: 1.0} if r.get("tid") else {}
        else:
            tr.nuc[i] = nucleus(r.get("tid", []), [math.exp(x) for x in r.get("tlp", [])])
        if r.get("tlp"):  # record i's distribution is for position i + 1, the bonus's
            lp = dict(zip(r["tid"], r["tlp"]))
            tr.tb[i] = (math.exp(lp.get(tr.toks[i + 1], r["tlp"][-1])), math.exp(r["tlp"][0]))
    for i, (d, s) in tr.recs.items():
        r = tr.tp.get(i)
        if r is not None:
            s = r["s"]  # 6 digits, where the log's line has 4
        A = i + 1
        a = S.run_len(d, tr.toks, A, tr.L)
        tr.a[i] = a
        c, prev = [], 1.0
        for j in range(NDEP):
            c.append(min(1.0, s[j] / prev) if prev > 0 else 0.0)
            prev = s[j]
        tr.c[i] = c
        if r is not None:
            tr.p1[i] = [r["p"][j][0] for j in range(NDEP)]
            pn = []
            for j in range(NDEP):
                ps = r["p"][j]
                cum, z = 0.0, None
                for x in ps:
                    cum += x
                    if cum >= TOP_P:
                        z = cum
                        break
                pn.append(min(1.0, ps[0] / (z if z else TOP_P)))
            tr.pn[i] = pn
        q = []
        for j in range(1, min(a + 1, NDEP) + 1):  # d_j for j <= a + 1: the output before it is d_1..d_j-1
            if A + j >= tr.L or (i + j) not in tr.nuc:
                break
            q.append(tr.nuc[i + j].get(d[j - 1], 0.0))
        tr.q[i] = q


def load(D, runs):
    meta = json.load(open(os.path.join(D, "meta.json")))
    trs = []
    for run in runs:
        seqs = meta["runs"].get(run)
        log = os.path.join(D, "tf_%s.log" % run)
        if not seqs or not os.path.exists(log):
            continue
        recs, tps = S.load_spectf(log), load_tp(os.path.join(D, "tp_%s.jsonl" % run))
        grp = []
        for m in seqs:
            tr = S.Trace(m["name"], S.npy_i4(os.path.join(m["golden"], "tokens.npy")), m["prompt"])
            tr.recs, tr.tp, tr.m, tr.run = recs.get(m["seq"], {}), tps.get(m["seq"], {}), m, run
            tr.kind, tr.tps, tr.hist = m.get("kind"), m.get("tps"), m.get("hist")
            tr.session = tuple(str(m.get(k)) for k in ("thinking", "rep", "task"))
            tr.session = (run,) + tr.session
            grp.append(tr)
        rt = os.path.join(D, "rt_%s.bin" % run)
        if os.path.exists(rt):
            S.attach_routes(grp, S.load_routes(rt))
        for tr in grp:
            prepare(tr, tr.m.get("sampling") == "greedy")
        trs += grp
    trs.sort(key=lambda t: (t.session, t.m.get("turn") or 0))  # a session's turns in order (the learners)
    return trs, meta


# ---- gates -------------------------------------------------------------------------------------------------------
class RowRate:  # cmd.cpp SpecRate for --spec-gate ratio / marginal: tokens per ms by --spec-row-ms
    def __init__(s):
        s.tok, s.ms = 1.0, 0.0

    def add(s, emitted, rows):
        a = 1.0 / 16
        if s.ms == 0:
            s.ms = ROW_MS[0]
        s.tok += a * (emitted - s.tok)
        s.ms += a * (ROW_MS[min(rows - 1, NDEP)] - s.ms)

    def lam(s):
        return s.tok / (s.ms if s.ms > 0 else ROW_MS[0])


class Sess:
    """What a gate learns from a session's own steps (nothing, for the static gates)."""

    def est(s, tr, i, base):
        return base

    def learn(s, tr, i, k, a):
        pass


class Offsets(Sess):
    """Per depth, a logit offset on the confidence head, learned online from the session's verified drafts
    (stochastic gradient on the log loss): P = sig(logit(c_j) + b_j)."""

    def __init__(s, eta):
        s.eta, s.b = eta, [0.0] * NDEP

    def est(s, tr, i, base):
        return [sig(logit(x) + s.b[j]) for j, x in enumerate(base)]

    def learn(s, tr, i, k, a):
        c = tr.c[i]
        for j in range(min(k, a + 1)):
            y = 1.0 if j < a else 0.0
            s.b[j] += s.eta * (y - sig(logit(c[j]) + s.b[j]))


class Table(Sess):
    """Per depth and confidence bucket, the session's acceptance so far (an EMA, from the head's own value
    until a bucket has seen a draft)."""

    def __init__(s, eta):
        s.eta, s.v = eta, {}

    def est(s, tr, i, base):
        return [s.v.get((j, bucket(x)), x) for j, x in enumerate(base)]

    def learn(s, tr, i, k, a):
        c = tr.c[i]
        for j in range(min(k, a + 1)):
            key = (j, bucket(c[j]))
            v = s.v.get(key, c[j])
            s.v[key] = v + s.eta * ((1.0 if j < a else 0.0) - v)


class Gate:
    """est: the drafts' conditional acceptance, from conf (the confidence head: today's), prob (the drafter's
    probability), probn (that under top-p), cal (a fitted map of conf and prob), oracle (the served sampler's own
    probabilities, known through the first rejected draft; the head's beyond), or hind (verify exactly the drafts
    accepted: a perfect gate). rule: thresh (today's: cumulative >= th * lambda * dC), tree (the tree study's
    per-node gate: >= th * lambda * the chain row's ms), marg (>= th * lambda' * the row's ms by --spec-row-ms,
    lambda' by those ms), value (the k with the most sum - th * lambda' * ms), ratio (the most tokens per ms by
    --spec-row-ms). sess: what the gate learns per session (a factory: Sess, Offsets, Table)."""

    def __init__(s, name, est="conf", rule="thresh", th=1.0, sess=Sess, cal=None, greedy_today=False):
        s.name, s.est, s.rule, s.th, s.sess, s.cal, s.greedy_today = name, est, rule, th, sess, cal, greedy_today

    def base(s, tr, i):
        if s.est == "conf":
            return tr.c[i]
        if s.est == "prob":
            return tr.p1.get(i, tr.c[i])
        if s.est == "probn":
            return tr.pn.get(i, tr.c[i])
        if s.est == "cal":
            return s.cal.apply(tr, i)
        if s.est == "oracle":
            q = tr.q[i]
            return q + tr.c[i][len(q):]
        raise ValueError(s.est)

    def k(s, tr, i, lam_g, lam_r, sess):
        if s.est == "hind":
            return tr.a[i]
        if s.greedy_today and tr.greedy:  # a sampling-aware gate leaves greedy requests to today's
            return pick(tr.c[i], "thresh", 1.0, lam_g, lam_r)
        return pick(sess.est(tr, i, s.base(tr, i)), s.rule, s.th, lam_g, lam_r)


def pick(est, rule, th, lam_g, lam_r):
    cum, p = [], 1.0
    for x in est:
        p *= x
        cum.append(p)
    k = 0
    if rule == "thresh":
        while k < NDEP and cum[k] >= th * lam_g * DC:
            k += 1
    elif rule == "tree":
        while k < NDEP and cum[k] >= th * lam_g * (T_CHAIN[k + 2] - T_CHAIN[k + 1]):
            k += 1
    elif rule == "marg":
        while k < NDEP and cum[k] >= th * lam_r * (ROW_MS[k + 1] - ROW_MS[k]):
            k += 1
    elif rule == "value":
        best, acc = 0.0, 0.0
        for j in range(NDEP):
            acc += cum[j]
            v = acc - th * lam_r * (ROW_MS[j + 1] - ROW_MS[0])
            if v > best:
                best, k = v, j + 1
    elif rule == "ratio":
        best, acc = 1.0 / ROW_MS[0], 1.0
        for j in range(NDEP):
            acc += cum[j]
            v = acc / (ROW_MS[j + 1] * th)
            if v > best:
                best, k = v, j + 1
    else:
        raise ValueError(rule)
    return k


# ---- the walk ----------------------------------------------------------------------------------------------------
def gwalk(tr, g, cost, beta, sess):
    """pld_study.walk's anchors and costs under a Gate: step 0 is the prompt's last position; a step at record i
    verifies k drafts, emits the a accepted and one more; lambda, and what the gate learns, lag two steps."""
    rate, rrate = S.Rate(), RowRate()
    lams, lrs = [rate.lam()], [rrate.lam()]
    i, end = tr.first, tr.L - 1
    tok, ms, steps, n = 1, S.step_ms(cost, 1, 6, True), 1, 0
    rate.add(1, 1)
    rrate.add(1, 1)
    lams.append(rate.lam())
    lrs.append(rrate.lam())
    hist = collections.Counter()
    acc = ver = 0
    pend = collections.deque()
    while i in tr.recs and i + 1 < end:
        while len(pend) > 1:
            sess.learn(*pend.popleft())
        A = i + 1
        k = g.k(tr, i, lams[max(0, n - 1)], lrs[max(0, n - 1)], sess)
        n += 1
        a = min(tr.a[i], k)
        e = min(a + 1, tr.L - 1 - A)
        t = S.step_ms(cost, 1 + k, tr.U(A, k) + beta * (k - a), True)
        ms += t
        tok += e
        steps += 1
        hist[k] += 1
        acc += a
        ver += k
        pend.append((tr, i, k, a))
        rate.add(e, 1 + k)
        rrate.add(e, 1 + k)
        lams.append(rate.lam())
        lrs.append(rrate.lam())
        i = i + 1 + a
    while pend:
        sess.learn(*pend.popleft())
    return dict(tok=tok, ms=ms, steps=steps, hist=hist, acc=acc, ver=ver)


def run_gate(trs, g, offs):
    """Every trace under g, a session's turns in order with one learner; per trace the walk's totals."""
    out, sess, cur = {}, None, None
    for tr in trs:
        if tr.session != cur:
            cur, sess = tr.session, (g.sess() if g.sess is Sess else g.sess(g.sess_eta))
        cost = dict(S.COST, c0=S.COST["c0"] + offs.get(tr.name, 0.0))
        out[tr.name] = gwalk(tr, g, cost, BETA[S.group_of(tr.name)], sess)
    return out


def agg(trs, res, sel=lambda tr: True):
    t = [res[tr.name] for tr in trs if sel(tr) and tr.name in res]
    tok, ms = sum(x["tok"] for x in t), sum(x["ms"] for x in t)
    steps = sum(x["steps"] for x in t)
    return dict(tps=1e3 * tok / ms if ms else 0.0, tok=tok, ms=ms, steps=steps,
                per_step=tok / steps if steps else 0.0, ver=sum(x["ver"] for x in t) / max(1, steps - len(t)),
                acc=sum(x["acc"] for x in t) / max(1, sum(x["ver"] for x in t)),
                hist=[sum(x["hist"].get(k, 0) for x in t) for k in range(NDEP + 1)])


def offsets(trs):
    """c0's offset per run set (gate_inputs' runs that share a served rate): today's gate replays the served decode
    (pld_study.offsets' rule for traces with a decode rate of their own)."""
    offs = {}
    for name, runs in SETS.items():
        sel = [tr for tr in trs if tr.run in runs and tr.tps and tr.name not in offs]
        if not sel:
            continue
        res = run_gate(sel, TODAY, {})
        meas = sum(1e3 * (tr.L - tr.P) / tr.tps for tr in sel)
        mod = sum(res[tr.name]["ms"] * (tr.L - tr.P) / res[tr.name]["tok"] for tr in sel)
        o = (meas - mod) / sum(res[tr.name]["steps"] for tr in sel)
        for tr in sel:
            offs[tr.name] = o
    return offs


TODAY = Gate("today", "conf", "thresh", 1.0)


# ---- the calibration -----------------------------------------------------------------------------------------------
def solve(m, v):
    n = len(v)
    a = [row[:] + [v[r]] for r, row in enumerate(m)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        for r in range(n):
            if r != i and a[i][i]:
                f = a[r][i] / a[i][i]
                for c in range(n + 1):
                    a[r][c] -= f * a[i][c]
    return [a[i][n] / a[i][i] if a[i][i] else 0.0 for i in range(n)]


def fit_logistic(xs, ys, ridge=1e-3, iters=30):
    """Log-loss logistic regression with soft labels (Newton steps): w over [1, x...]."""
    d = len(xs[0]) + 1
    w = [0.0] * d
    for _ in range(iters):
        g, h = [0.0] * d, [[ridge if r == c else 0.0 for c in range(d)] for r in range(d)]
        for x, y in zip(xs, ys):
            v = [1.0] + x
            p = sig(sum(wi * vi for wi, vi in zip(w, v)))
            for r in range(d):
                g[r] += (p - y) * v[r]
                for c in range(d):
                    h[r][c] += p * (1 - p) * v[r] * v[c]
        step = solve(h, g)
        w = [wi - si for wi, si in zip(w, step)]
        if max(abs(x) for x in step) < 1e-7:
            break
    return w


class Cal:
    """A fitted map per depth: P(d_j accepted | d_1..d_j-1) = sig(w0 + w1 logit(c_j) [+ w2 logit(p1_j)] [+ w3
    logit(P(the bonus)) + w4 logit(its top P)]), fitted on the training traces to the served sampler's
    probabilities (q) where they are known (the drafts through the first rejected one)."""

    def __init__(s, trs, use_p=False, use_t=False):
        s.use_p, s.use_t, s.w = use_p, use_t, []
        for j in range(NDEP):
            xs, ys = [], []
            for tr in trs:
                for i, q in tr.q.items():
                    if j < len(q) and s.has(tr, i):
                        xs.append(s.feat(tr, i, j))
                        ys.append(q[j])
            s.w.append(fit_logistic(xs, ys) if len(xs) > 20 else [0.0, 1.0] + [0.0] * (use_p + 2 * use_t))

    def has(s, tr, i):
        return (not s.use_p or i in tr.p1) and (not s.use_t or i in tr.tb)

    def feat(s, tr, i, j):
        f = [logit(tr.c[i][j])]
        if s.use_p:
            f.append(logit(tr.p1[i][j]))
        if s.use_t:
            f += [logit(tr.tb[i][0]), logit(tr.tb[i][1])]
        return f

    def apply(s, tr, i):
        out = []
        for j in range(NDEP):
            if not s.has(tr, i):
                out.append(tr.c[i][j])
                continue
            v = [1.0] + s.feat(tr, i, j)
            out.append(sig(sum(wi * vi for wi, vi in zip(s.w[j], v))))
        return out


# ---- reports -----------------------------------------------------------------------------------------------------
KINDS = ["edit", "write", "read", "prose"]


def sel_of(set_name):
    runs = SETS[set_name]
    return lambda tr: tr.run in runs


def check(trs):
    for run in sorted({tr.run for tr in trs}):
        rs = [tr for tr in trs if tr.run == run]
        n = sum(tr.L - 1 - tr.first for tr in rs)
        recs = sum(len(tr.recs) for tr in rs)
        tps = sum(len(tr.tp) for tr in rs)
        routed = sum(sum(1 for p in range(tr.P, tr.L) if p in tr.masks) for tr in rs)
        bad_d = bad_s = bad_top = top_in = top_n = 0
        for tr in rs:
            for i, (d, s) in tr.recs.items():
                r = tr.tp.get(i)
                if r is None:
                    continue
                bad_d += r["d"] != d
                bad_s += any(abs(x - y) > 6e-5 for x, y in zip(r["s"], s))
                for j in range(min(tr.a[i] + 1, NDEP)):  # through the first rejected draft the probe is the chain's
                    bad_top += r["id"][j][0] != d[j]
                tn = tr.nuc.get(i + 1)  # the output's own token among the sampler's kept ones
                if tn is not None and i + 2 < tr.L:
                    top_n += 1
                    top_in += tr.toks[i + 2] in tn
        print("%s: %d traces, %d positions: %d spectf records, %d probe records, %d routed; probe vs log: drafts "
              "differ at %d, survival at %d; probe's top 1 != the draft through the first rejected one: %d; the "
              "output's token in the sampler's kept set: %d of %d" % (run, len(rs), n, recs, tps, routed, bad_d,
                                                                       bad_s, bad_top, top_in, top_n))


def calib(trs, set_name):
    sel = sel_of(set_name)
    print("\n### %s: the drafts' acceptance against the estimates, by depth and the head's confidence" % set_name)
    print("| depth | confidence | drafts | head's c | accepted | sampler's P (q) | drafter's p | p under top-p |")
    print("|---|---|---|---|---|---|---|---|")
    for j in range(3):
        cells = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for tr in trs:
            if not sel(tr):
                continue
            for i, q in tr.q.items():
                if j < len(q):
                    b = bucket(tr.c[i][j])
                    v = cells[b]
                    v[0] += 1
                    v[1] += tr.c[i][j]
                    v[2] += 1.0 if j < tr.a[i] else 0.0
                    v[3] += q[j]
                    v[4] += tr.p1[i][j] if i in tr.p1 else tr.c[i][j]
                    v[5] += tr.pn[i][j] if i in tr.pn else tr.c[i][j]
        lo = [0.0] + EDGES
        hi = EDGES + [1.0]
        tot = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for b in sorted(cells):
            v = cells[b]
            tot = [x + y for x, y in zip(tot, v)]
            print("| %d | %.2f-%.2f | %d | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                j + 1, lo[b], hi[b], v[0], v[1] / v[0], v[2] / v[0], v[3] / v[0], v[4] / v[0], v[5] / v[0]))
        if tot[0]:
            print("| %d | all | %d | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                j + 1, tot[0], tot[1] / tot[0], tot[2] / tot[0], tot[3] / tot[0], tot[4] / tot[0], tot[5] / tot[0]))


def replay(trs, offs):
    res = run_gate(trs, TODAY, offs)
    for set_name in SETS:
        sel = sel_of(set_name)
        rs = [tr for tr in trs if sel(tr) and tr.tps]
        if not rs:
            continue
        print("\n### %s: today's gate replayed against the served decode (offset %+.2f ms a step)" % (
            set_name, offs.get(rs[0].name, 0.0)))
        print("| kind | traces | tokens | served tok/s | replay tok/s | served steps verifying 0..5 | replayed |")
        print("|---|---|---|---|---|---|---|")
        for k in KINDS + ["all"]:
            ks = [tr for tr in rs if k == "all" or tr.kind == k]
            if not ks:
                continue
            meas = sum(tr.L - tr.P for tr in ks) / sum((tr.L - tr.P) / tr.tps for tr in ks)
            a = agg(ks, res)
            hs = [sum(tr.hist[x] for tr in ks if tr.hist) for x in range(NDEP + 1)]
            print("| %s | %d | %d | %.1f | %.1f | %s | %s |" % (k, len(ks), sum(tr.L - tr.P for tr in ks), meas,
                                                                a["tps"], " ".join(map(str, hs)),
                                                                " ".join(map(str, a["hist"][:]))))


def candidates(trs_train):
    """The gates compared, with their parameter grids (th: the threshold's multiplier, eta: a learner's rate)."""
    c = [("today's gate", [Gate("today", "conf", "thresh", 1.0)]),
         ("thresh, scaled", [Gate("thresh x%.2f" % t, "conf", "thresh", t) for t in (0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)]),
         ("ratio", [Gate("ratio", "conf", "ratio", 1.0)]),
         ("marginal", [Gate("marg x%.2f" % t, "conf", "marg", t) for t in (0.7, 0.85, 1.0, 1.2, 1.4)]),
         ("1: probability, tree rule", [Gate("prob tree a%.2f" % t, "prob", "tree", t) for t in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)]),
         ("1: probability, thresh", [Gate("prob x%.2f" % t, "prob", "thresh", t) for t in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5)]),
         ("1: probability under top-p, thresh", [Gate("probn x%.2f" % t, "probn", "thresh", t) for t in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5)]),
         ("1: probability, value", [Gate("prob value x%.2f" % t, "prob", "value", t) for t in (0.6, 0.8, 1.0, 1.25, 1.5)]),
         ]
    for eta in (0.02, 0.05, 0.1, 0.2):
        for rule, ths in [("thresh", (0.8, 1.0, 1.2)), ("value", (0.8, 1.0, 1.25)), ("ratio", (1.0,))]:
            gs = []
            for t in ths:
                g = Gate("2: offsets eta %.2f, %s x%.2f" % (eta, rule, t), "conf", rule, t, Offsets)
                g.sess_eta = eta
                gs.append(g)
                g = Gate("2: table eta %.2f, %s x%.2f" % (eta, rule, t), "conf", rule, t, Table)
                g.sess_eta = eta
                gs.append(g)
            c.append(("2: learned per session, eta %.2f, %s" % (eta, rule), gs))
    cal1, cal2, cal3 = Cal(trs_train), Cal(trs_train, use_p=True), Cal(trs_train, use_p=True, use_t=True)
    for nm, cal in [("3: calibrated head", cal1), ("3: calibrated head and drafter", cal2),
                    ("3: calibrated head, drafter and target", cal3)]:
        for rule, ths in [("thresh", (0.7, 0.85, 1.0, 1.2)), ("value", (0.8, 1.0, 1.25)), ("ratio", (1.0,))]:
            c.append(("%s, %s" % (nm, rule), [Gate("%s %s x%.2f" % (nm, rule, t), "cal", rule, t, cal=cal,
                                                   greedy_today=True) for t in ths]))
    c.append(("bound: the sampler's probabilities, value", [Gate("oracle value x%.2f" % t, "oracle", "value", t) for t in (0.8, 1.0, 1.25)]))
    c.append(("bound: the sampler's probabilities, thresh", [Gate("oracle x%.2f" % t, "oracle", "thresh", t) for t in (0.7, 0.85, 1.0, 1.2)]))
    c.append(("bound: a perfect gate", [Gate("hindsight", "hind")]))
    if ONLY:
        c = [x for x in c if x[0] == "today's gate" or any(o in x[0] for o in ONLY)]
    return c, (cal1, cal2, cal3)


def sweep(trs, offs, train="r2-train", tests=("r2-test", "r2", "r1")):
    tests = tuple(t for t in tests if any(sel_of(t)(tr) for tr in trs))
    base = run_gate(trs, TODAY, offs)
    b = {s: agg(trs, base, sel_of(s)) for s in (train,) + tests}
    trs_train = [tr for tr in trs if sel_of(train)(tr)]
    cands, cals = candidates(trs_train)
    for j, w in enumerate(cals[0].w):
        print("calibrated head, depth %d: sig(%.3f + %.3f logit(c))" % (j + 1, w[0], w[1]))
    for j, w in enumerate(cals[1].w):
        print("calibrated head and drafter, depth %d: sig(%.3f + %.3f logit(c) + %.3f logit(p))" % (j + 1, *w))
    for j, w in enumerate(cals[2].w):
        print("calibrated head, drafter and target, depth %d: sig(%.3f + %.3f logit(c) + %.3f logit(p) + %.3f "
              "logit(P(bonus)) + %.3f logit(its top P))" % (j + 1, *w))
    print("\n| gate | %s (train) | %s |" % (train, " | ".join(tests)))
    print("|---|---|%s" % ("---|" * len(tests)))
    chosen = []
    for fam, gs in cands:
        rows = []
        for g in gs:
            res = run_gate(trs, g, offs)
            v = {s: agg(trs, res, sel_of(s)) for s in (train,) + tests}
            rows.append((g, v, res))
        best = max(rows, key=lambda x: x[1][train]["tps"])
        for g, v, res in rows:
            mark = " (best on %s)" % train if g is best[0] and len(rows) > 1 else ""
            print("| %s%s | %s |" % (g.name, mark, " | ".join(
                "%.1f, %+.1f %%" % (v[s]["tps"], 100 * (v[s]["tps"] / b[s]["tps"] - 1)) for s in (train,) + tests)))
        chosen.append((fam, best[0], best[2]))
    return chosen, base


def tables(trs, chosen, base, sets=("r2-test", "r2", "r1")):
    for s in sets:
        sel = sel_of(s)
        if not any(sel(tr) for tr in trs):
            continue
        print("\n### %s: tok/s by kind (each family at its best setting on the training set), against today's gate" % s)
        cols = [(k, lambda tr, k=k: tr.kind == k) for k in KINDS if any(sel(tr) and tr.kind == k for tr in trs)]
        ths = sorted({str(tr.m.get("thinking")) for tr in trs if sel(tr)})
        if len(ths) > 1:
            cols += [("thinking " + t, lambda tr, t=t: str(tr.m.get("thinking")) == t) for t in ths]
        cols.append(("all", lambda tr: True))
        print("| gate | %s | drafts verified/step | accepted |" % " | ".join(k for k, _ in cols))
        print("|---|%s---|---|" % ("---|" * len(cols)))
        rows = [("today's gate", base)] + [("%s (%s)" % (fam, g.name), res) for fam, g, res in chosen if fam != "today's gate"]
        for nm, res in rows:
            cells = []
            for _, c in cols:
                f = (lambda tr, c=c: sel(tr) and c(tr))
                v, b0 = agg(trs, res, f), agg(trs, base, f)
                cells.append("%.1f" % v["tps"] if res is base else "%+.1f %%" % (100 * (v["tps"] / b0["tps"] - 1)))
            v = agg(trs, res, sel)
            print("| %s | %s | %.2f | %.0f %% |" % (nm, " | ".join(cells), v["ver"], 100 * v["acc"]))


TAUS = [0.5, 0.7, 1.0, 1.5, 2.0]  # --ss: the drafter's temperature for a sampled draft


def ssamp(trs, set_name):
    """An aside outside the gate's bar (it changes which tokens a seed gives, not their distribution): what a
    distribution-exact speculative sampling would accept. A draft sampled from the drafter's p (its top 8 at
    temperature tau) and kept with probability min(1, q / p) is accepted with probability sum_x min(p(x), q(x));
    today's draft (the drafter's top 1, kept when the sampler draws it) with q(d_j). Both along the served output
    (the probe's parents and the target's distributions are the output's: past the first rejected draft a proxy for
    the chain's own): the mean by depth, and a 5-draft step's expected drafts accepted, sum_k prod_j<=k; the
    realized mean of today's for scale."""
    sel = sel_of(set_name)
    cells = collections.defaultdict(lambda: dict(n=0, a=0.0, ed=0.0, es=[0.0] * len(TAUS), d=[0.0] * NDEP,
                                                 s=[[0.0] * NDEP for _ in TAUS]))
    for tr in trs:
        if not sel(tr) or tr.greedy:
            continue
        for i, r in tr.tp.items():
            if i not in tr.a or any((i + j) not in tr.nuc for j in range(1, NDEP + 1)):
                continue
            det, ss = [], [[] for _ in TAUS]
            for j in range(NDEP):
                q, ids, ps = tr.nuc[i + j + 1], r["id"][j], r["p"][j]
                det.append(q.get(ids[0], 0.0))
                for t, tau in enumerate(TAUS):
                    w = [p ** (1.0 / tau) for p in ps]
                    z = sum(w)
                    ss[t].append(sum(min(x / z, q.get(y, 0.0)) for x, y in zip(w, ids)))
            expect = lambda b: sum(math.prod(b[:k + 1]) for k in range(NDEP))
            for key in (tr.kind, "thinking " + str(tr.m.get("thinking")), "all"):
                v = cells[key]
                v["n"] += 1
                v["a"] += tr.a[i]
                v["ed"] += expect(det)
                for t in range(len(TAUS)):
                    v["es"][t] += expect(ss[t])
                    for j in range(NDEP):
                        v["s"][t][j] += ss[t][j]
                for j in range(NDEP):
                    v["d"][j] += det[j]
    if not cells:
        return
    print("\n### %s: speculative sampling (an aside): a 5-draft step's expected drafts accepted" % set_name)
    print("| | anchors | accepted today (realized) | today's draft (expected) | %s | tokens a step, best tau |" %
          " | ".join("sampled draft, tau %.1f" % t for t in TAUS))
    print("|---|---|---|---|%s---|" % ("---|" * len(TAUS)))
    for key in [k for k in KINDS if k in cells] + sorted(k for k in cells if k.startswith("thinking")) + ["all"]:
        v = cells[key]
        n = v["n"]
        best = max(v["es"])
        print("| %s | %d | %.2f | %.2f | %s | %+.1f %% |" % (key, n, v["a"] / n, v["ed"] / n,
                                                         " | ".join("%.2f" % (x / n) for x in v["es"]),
                                                         100 * ((1 + best / n) / (1 + v["ed"] / n) - 1)))
    v = cells["all"]
    print("\n| depth | today's draft | %s |" % " | ".join("sampled, tau %.1f" % t for t in TAUS))
    print("|---|---|%s" % ("---|" * len(TAUS)))
    for j in range(NDEP):
        print("| %d | %.3f | %s |" % (j + 1, v["d"][j] / v["n"], " | ".join("%.3f" % (v["s"][t][j] / v["n"])
                                                                         for t in range(len(TAUS)))))


def main():
    a = sys.argv[1:]
    if not a or a[0].startswith("--"):
        sys.exit(USAGE)
    D = a[0]
    if "--only" in a:
        ONLY[:] = a[a.index("--only") + 1].split(",")
    if "--sets" in a:
        for x in a[a.index("--sets") + 1].split(","):
            k, v = x.split("=")
            SETS[k] = v.split("+")
    runs = sorted({r for v in SETS.values() for r in v})
    trs, meta = load(D, runs)
    print("%d traces from runs %s" % (len(trs), ", ".join(sorted({tr.run for tr in trs}))))
    if "--check" in a:
        check(trs)
    offs = offsets(trs)
    if "--cal" in a:
        for s in SETS:
            if any(sel_of(s)(tr) for tr in trs):
                calib(trs, s)
    if "--replay" in a:
        replay(trs, offs)
    if "--ss" in a:
        for s in SETS:
            ssamp(trs, s)
    if "--sweep" in a or "--tables" in a:
        chosen, base = sweep(trs, offs)
        if "--tables" in a:
            tables(trs, chosen, base)


if __name__ == "__main__":
    main()
