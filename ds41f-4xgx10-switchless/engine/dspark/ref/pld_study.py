#!/usr/bin/env python3
# Prompt-lookup drafts, a feasibility study (docs/engine.md, "Prompt-lookup drafts: a feasibility study"): the
# offline analysis of spectf's records along the target's own outputs (main.cpp --seq-list, with --route-file's
# routed experts), in DIR as pld_inputs.py and the two ring passes leave it. PLD proposes what followed the most
# recent occurrence of the longest suffix (n = 8 .. 2) of the history before the anchor. The analysis gives how
# much of each output PLD would copy (--copy), checks the records (--check), fits the extra experts of rejected
# rows (--cal), replays today's chain against pass 1's decode (--replay), and walks today's chain, PLD and hybrids
# at row caps 8 and 16 under a cost model (--policies; --tables prints docs/engine.md's), and bounds a gate on the
# match's length by lengths chosen in hindsight on the edit traces (--hindsight), with the walks and costs behind
# the shortfall (--why). No numpy.
#   pld_study.py DIR [--check [--tree TREEDIR]] [--copy] [--cal] [--replay] [--policies [--only=a,b]] [--tables]
#                    [--hindsight] [--why]
# DIR/meta.json: {"runs": {RUN: [{"seq", "name", "golden", "prompt"}]}, "pass1": {NAME: {"tps", "report", ...}},
#                 "bench_tps": {NAME: tok/s}}; DIR/tf_RUN.log (spectf's output), DIR/rt_RUN.bin (--route-file).
import collections, json, os, re, struct, sys

USAGE = ("pld_study.py DIR [--check [--tree TREEDIR]] [--copy] [--cal] [--replay] [--policies [--only=a,b]] [--tables] "
         "[--hindsight] [--why]")
NDEP = 5             # drafts per pass
NMAX = 8             # the longest suffix PLD matches
AGREE = 256          # Index.agreement's cap
LAYERS, TOPK = 40, 6
C1, DC, CDRAFT = 19.4, 3.6, 3.3  # the gate's costs (cmd.h spec_cost defaults)
O16 = 3.09           # the cost fit's extra ms of a step past 8 rows, as a gate sees it
THINK_END = 128822   # </think>
DSML = 128825        # a tool call's first token: a turn with it is a tool-call turn
STR_FALSE = [3418, 1281, 19836, 3320]  # ' string', '="', 'false', '">': a DSML parameter whose value is JSON
CLOSE = [1718, DSML]                   # '</', '｜DSML｜': a parameter's (or a call's) end


def esc_modes(toks):
    """Per position: inside a JSON-valued DSML parameter (its strings JSON-escaped) after the token there."""
    out, esc = [], False
    for p, t in enumerate(toks):
        if toks[max(0, p - 3):p + 1] == STR_FALSE:
            esc = True
        elif toks[max(0, p - 1):p + 1] == CLOSE:
            esc = False
        out.append(esc)
    return out


class Matcher:
    """PLD over a history; with copies (the escaped matcher) the anchors inside a JSON-valued parameter search the
    history and the copies, the others the history alone."""

    def __init__(s, toks, virt=()):
        s.raw = Index(toks)
        s.esc = Index(toks, virt) if virt else None
        s.mode = esc_modes(toks) if virt else None

    def at(s, A):  # the index for anchor A, advanced to it
        ix = s.esc if s.esc is not None and s.mode[A] else s.raw
        ix.advance(A)
        return ix


def npy_i4(path):
    b = open(path, "rb").read()
    hl = struct.unpack("<H", b[8:10])[0]
    n = (len(b) - 10 - hl) // 4
    return list(struct.unpack("<%di" % n, b[10 + hl:10 + hl + 4 * n]))


# ---- PLD ---------------------------------------------------------------------------------------------------------
class Index:
    """n-grams (n = 1..NMAX) of toks ending before an anchor, each mapped to its latest end. virt: [(p, ids)], copies
    the matcher also searches, each as if it followed position p (the escaped matcher: a tool result JSON-escaped,
    so that a quote of it inside a tool call's JSON arguments matches); a -1 on each side keeps matches inside."""

    def __init__(s, toks, virt=()):
        s.C, s.cmap = [], []
        vi = sorted((p, list(v)) for p, v in virt)
        k = 0
        for p, t in enumerate(toks):
            s.C.append(t)
            s.cmap.append(len(s.C) - 1)
            while k < len(vi) and vi[k][0] == p:
                s.C += [-1] + vi[k][1] + [-1]
                k += 1
        s.d = [None] + [dict() for _ in range(NMAX)]
        s.upto = -1

    def advance(s, A):  # hold the n-grams ending before A (and the copies placed before it)
        t, d = s.C, s.d
        while s.upto < s.cmap[A] - 1:
            s.upto += 1
            e = s.upto
            for n in range(1, NMAX + 1):
                if e - n + 1 < 0:
                    break
                d[n][tuple(t[e - n + 1:e + 1])] = e

    def find(s, suffix, A, nmin=2, kmax=15):
        """The longest n (NMAX .. nmin) whose last n tokens of suffix occurred (ending before A): (n, where that
        latest occurrence ends in C, the tokens after it, up to kmax, within toks[.. A] or the copy it is in)."""
        for n in range(min(NMAX, len(suffix)), nmin - 1, -1):
            j = s.d[n].get(tuple(suffix[-n:]))
            if j is not None:
                cont = s.C[j + 1:min(j + 1 + kmax, s.cmap[A] + 1)]
                if -1 in cont:
                    cont = cont[:cont.index(-1)]
                if cont:
                    return n, j, cont
        return 0, -1, []

    def lookup(s, suffix, A, nmin=2, kmax=15):
        n, j, cont = s.find(suffix, A, nmin, kmax)
        return n, cont

    def agreement(s, n, j, seq):
        """How far the match goes back: the tokens before seq's end equal to those before the occurrence's end (n at
        least, AGREE at most), the match's uncapped length."""
        L = n
        while L < AGREE and j - L >= 0 and L < len(seq) and s.C[j - L] != -1 and s.C[j - L] == seq[-1 - L]:
            L += 1
        return L


def run_len(prop, toks, A, end):
    """The leading tokens of prop equal to the sequence's after the anchor A."""
    r = 0
    while r < len(prop) and A + 1 + r < end and prop[r] == toks[A + 1 + r]:
        r += 1
    return r


# ---- traces ------------------------------------------------------------------------------------------------------
class Trace:
    def __init__(s, name, toks, P):
        s.name, s.toks, s.P = name, toks, P
        s.L = len(toks)
        s.first = P - 1
        s.recs = {}      # spectf step i -> (drafts d1..d5, survival s1..s5): anchor i + 1
        s.masks = {}     # position -> [LAYERS] expert bitmasks of the sequence's own token there
        s.exact = []     # (kind, anchor, rows' tokens, [per row: [LAYERS] masks]): the drafts' (1), PLD's (2)
        s.props = {}     # anchor -> (n, proposal of up to 15)
        s.vprops = {}    # anchor -> [(n, proposal) after the anchor's drafts d1..dk, k = 1..5]
        s.ag = {}        # anchor -> the match's uncapped length (Index.agreement), and after d1..dk
        s.vag = {}
        s.up = {}        # (anchor, k) -> distinct experts per layer of the own rows anchor .. anchor + k
        s.p1 = None      # pass 1's decode of this output (turns)
        s.tool = DSML in toks[P:]
        s.virt = []      # the escaped matcher's copies (Index)
        s.kind = None    # an edit trace's call: edit, write, read, prose
        s.tps = None     # a measured decode rate (editbench.py)
        s.hist = None    # its steps verifying 0..5 drafts, as served

    def build_props(s, nmin=2, kmax=15):
        mt = Matcher(s.toks, s.virt)
        for A in range(s.P, s.L):
            ix = mt.at(A)
            seq = s.toks[max(0, A - AGREE - NMAX):A + 1]
            n, j, cont = ix.find(seq[-NMAX:], A, nmin, kmax)
            s.props[A] = (n, cont)
            s.ag[A] = ix.agreement(n, j, seq) if n else 0
            r = s.recs.get(A - 1)
            if r is not None:
                s.vprops[A], s.vag[A] = [], []
                for k in range(1, NDEP + 1):
                    sk = seq + list(r[0][:k])
                    n, j, cont = ix.find(sk[-NMAX:], A, nmin, kmax)
                    s.vprops[A].append((n, cont))
                    s.vag[A].append(ix.agreement(n, j, sk) if n else 0)

    def U(s, A, k):
        """Distinct routed experts per layer over the sequence's own rows A .. A + k (clipped at the end)."""
        key = (A, k)
        v = s.up.get(key)
        if v is None:
            rows = [s.masks[p] for p in range(A, min(A + k + 1, s.L)) if p in s.masks]
            v = union_U(rows) + 3.2 * (k + 1 - len(rows)) if rows else 6.0 + 3.2 * k
            s.up[key] = v
        return v


def union_U(rows):
    tot = nl = 0
    for l in range(LAYERS):
        acc = 0
        for m in rows:
            acc |= m[l]
        if acc:
            tot += bin(acc).count("1")
            nl += 1
    return tot / nl if nl else 0.0


def load_spectf(log):
    """{seq: {i: (drafts, survival)}} from spectf's lines ('seq K spectf i: drafts ..., survival ..., n equal')."""
    pat = re.compile(r"^(?:seq (\d+) )?spectf (\d+): drafts ([\d ]+), survival ([\d. ]+), (\d+) equal")
    out = collections.defaultdict(dict)
    for line in open(log, errors="replace"):
        m = pat.match(line)
        if m:
            sq = int(m.group(1)) if m.group(1) else 0
            out[sq][int(m.group(2))] = ([int(x) for x in m.group(3).split()], [float(x) for x in m.group(4).split()])
    return out


def load_tp(path):
    """{i: (drafts, survival)} from the tree study's probe records."""
    recs = {}
    for line in open(path):
        r = json.loads(line)
        if "sib_at" not in r:
            recs[r["i"]] = (r["d"], r["s"])
    return recs


def load_routes(path):
    """--route-file's records: int32 {seq, kind, anchor, pos0, nr}, int32 tokens[nr], int16 experts [LAYERS][nr][TOPK]
    (-1: none) -> (seq, kind, anchor, pos0, tokens, [per row: [LAYERS] bitmasks])."""
    b = open(path, "rb").read()
    o = 0
    out = []
    while o < len(b):
        sq, kind, anchor, pos0, nr = struct.unpack_from("<5i", b, o)
        o += 20
        t = list(struct.unpack_from("<%di" % nr, b, o))
        o += 4 * nr
        e = struct.unpack_from("<%dh" % (LAYERS * nr * TOPK), b, o)
        o += 2 * LAYERS * nr * TOPK
        ms = []
        for r in range(nr):
            row = []
            for l in range(LAYERS):
                m = 0
                for x in e[(l * nr + r) * TOPK:(l * nr + r + 1) * TOPK]:
                    if x >= 0:
                        m |= 1 << x
                row.append(m)
            ms.append(row)
        out.append((sq, kind, anchor, pos0, t, ms))
    return out


def attach_routes(traces, recs):
    """Kind 0 records give each position's own experts. Kinds 1 and 2 (the drafts' or a proposal's rows behind the
    anchor's own token) are joined across their pieces of <= 8 rows into exact records."""
    pend = {}
    for sq, kind, anchor, pos0, t, ms in recs:
        tr = traces[sq]
        if kind == 0:
            for r, m in enumerate(ms):
                tr.masks[pos0 + r] = m
            continue
        key = (sq, kind, anchor)
        if pos0 == anchor:  # a record's first piece (another at the same anchor closes the one before)
            if key in pend:
                v = pend.pop(key)
                traces[v[0]].exact.append(tuple(v[1:]))
            pend[key] = [sq, kind, anchor, list(t), list(ms)]
        else:
            pend[key][3] += t
            pend[key][4] += ms
    for v in pend.values():
        traces[v[0]].exact.append(tuple(v[1:]))


def load_all(D, escaped=True):
    """The traces of meta.json's runs with spectf's records and the routes; with DIR/esc.json (pld_inputs.py escaped)
    and escaped, PLD is the escaped matcher."""
    meta = json.load(open(os.path.join(D, "meta.json")))
    ep = os.path.join(D, "esc.json")
    esc = json.load(open(ep)) if escaped and os.path.exists(ep) else {}
    out = []
    for run, seqs in meta["runs"].items():
        log, rt = os.path.join(D, "tf_%s.log" % run), os.path.join(D, "rt_%s.bin" % run)
        if not os.path.exists(log):
            continue
        recs = load_spectf(log)
        trs = []
        for m in seqs:
            tr = Trace(m["name"], npy_i4(os.path.join(D, m["golden"], "tokens.npy")), m["prompt"])
            tr.recs = recs.get(m["seq"], {})
            tr.p1 = meta.get("pass1", {}).get(m["name"])
            tr.kind = m.get("kind")
            tr.tps = m.get("tps")
            tr.hist = m.get("hist")
            tr.virt = esc.get(m["golden"], [])
            trs.append(tr)
        if os.path.exists(rt):
            attach_routes(trs, load_routes(rt))
        out += trs
    for tr in out:
        tr.build_props()
    return out, meta


def group_of(name):
    return "bench" if name.startswith("bench") else "edit" if name.startswith("edit") else "turns"


# ---- the extra experts of rejected rows --------------------------------------------------------------------------
def calibrate(traces, kinds):
    """The exact unions of the drafts' (1) or proposals' (2) rows against the model's (the own rows at the same
    positions), for every prefix of each record: the mean difference by (rows, rejected rows), and beta, the fit
    through 0 per rejected row."""
    sxy = sxx = 0.0
    by = collections.defaultdict(lambda: [0, 0.0])
    for tr in traces:
        for kind, A, t, ms in tr.exact:
            if kind not in kinds or t[0] != tr.toks[A]:
                continue
            a = run_len(t[1:], tr.toks, A, tr.L)
            for k in range(1, len(t)):
                if A + k >= tr.L:
                    break
                off = k - min(a, k)
                dlt = union_U(ms[:k + 1]) - tr.U(A, k)
                sxy += off * dlt
                sxx += off * off
                b = by[(k, off)]
                b[0] += 1
                b[1] += dlt
    return (sxy / sxx if sxx else 0.0), {key: (v[0], v[1] / v[0]) for key, v in sorted(by.items())}


def betas(trs, verbose=False):
    """Per group (bench, turns, edit) and row source (c: the drafts, p: PLD)."""
    out = {}
    for g in ["bench", "turns", "edit"]:
        sel = [tr for tr in trs if group_of(tr.name) == g]
        if not sel:
            continue
        out[g] = {"c": calibrate(sel, (1,))[0], "p": calibrate(sel, (2,))[0]}
        if verbose:
            print("beta, %s: drafts %+.3f, PLD %+.3f experts per rejected row" % (g, out[g]["c"], out[g]["p"]))
    return out


# ---- costs -------------------------------------------------------------------------------------------------------
# the tree study's fit (tree_costs.py): T = c0 + b_row (R - 1) + c_exp (U - 6), o16 above 8 rows, and the draft
# pass (FP8 head); a step that skips the pass still runs accept (the drafter's windows)
COST = dict(c0=18.64, b_row=1.10, c_exp=0.713, o16=3.09, draft=2.55, accept=0.15)


def step_ms(cost, R, U, draft):
    t = cost["c0"] + cost["b_row"] * (R - 1) + cost["c_exp"] * (U - 6)
    if R > 8:
        t += cost["o16"]
    if draft == "acc":
        t += cost["accept"]
    elif draft:
        t += cost["draft"]
    return t


class Rate:  # cmd.cpp SpecRate (the thresh gate's lambda), as the engine computes it
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


# ---- policies: (trace, i, lam, state) -> (rows after the bonus, the draft pass: True, False or "acc") ------------
# state["src"] marks each row's source (c: a draft, p: PLD) for the rejected rows' experts; state["last"] the PLD
# rows' gate to update
def chain_rows(tr, i, lam):
    d, sv = tr.recs[i]
    th = lam * DC
    k = 0
    while k < NDEP and sv[k] >= th:
        k += 1
    return list(d[:k])


def P_chain():
    def f(tr, i, lam, st):
        st["src"] = "c" * NDEP
        return chain_rows(tr, i, lam), True
    f.name = "chain"
    return f


class PldGate:
    """P(a PLD token is right): q by the match (the first token), c for each further one (per key with per_key_c);
    EMAs of what earlier steps saw (causal)."""

    def __init__(s, a=1.0 / 8, per_key_c=False):
        s.q = collections.defaultdict(lambda: 0.5)
        s.c = 0.9
        s.ck = collections.defaultdict(lambda: 0.9) if per_key_c else None
        s.a = a

    def cof(s, key):
        return s.ck[key] if s.ck is not None else s.c

    def k(s, key, plen, lam, kmax):  # verify while P(all so far right) >= lam * dC
        th = lam * DC
        p = s.q[key]
        k = 0
        while k < min(plen, kmax) and p >= th:
            k += 1
            p *= s.c
        return k

    def learn(s, key, k, acc):
        if k < 1:
            return
        s.q[key] += s.a * ((1.0 if acc >= 1 else 0.0) - s.q[key])
        for j in range(2, k + 1):
            if acc < j - 1:
                break
            if s.ck is not None:
                s.ck[key] += s.a / 4 * ((1.0 if acc >= j else 0.0) - s.ck[key])
            else:
                s.c += s.a / 4 * ((1.0 if acc >= j else 0.0) - s.c)


def P_pld(m, K, gate="fixed", fallback=None, name=None, skip_draft=True):
    """PLD when the longest match has n >= m (a proposal of up to K - 1 tokens); else the fallback (None: one row and
    no drafter). gate: 'fixed' (the whole proposal), 'ema' (PldGate), 'oracle' (exactly the right run)."""
    def f(tr, i, lam, st):
        A = i + 1
        n, prop = tr.props.get(A, (0, []))
        if n >= m and prop:
            if gate == "fixed":
                k = min(len(prop), K - 1)
            elif gate == "oracle":
                k = min(run_len(prop, tr.toks, A, tr.L), K - 1)
            else:
                g = st.setdefault("gate", PldGate())
                k = g.k(min(n, NMAX), len(prop), lam, K - 1)
            if k > 0:
                st["last"] = ("pld", min(n, NMAX), k, 0)
                st["src"] = "p" * k
                return prop[:k], (True if not skip_draft else "acc" if fallback is not None else False)
        st["last"] = None
        st["src"] = "c" * NDEP
        if fallback is None:
            return [], False
        return fallback(tr, i, lam, st)
    f.name = name or "pld m%d K%d %s" % (m, K, gate)
    return f


def P_h1(m, K, gate="ema"):
    """PLD when its match has n >= m, else today's chain; a PLD step skips the draft pass (not accept)."""
    return P_pld(m, K, gate, fallback=lambda tr, i, lam, st: (chain_rows(tr, i, lam), True),
                 name="h1 pld m%d K%d %s, else chain" % (m, K, gate))


def P_h2(m, K, gate="ema"):
    """PLD (as h1, the draft pass every step), its proposal extended by the chain's drafts beyond it where they
    agree with it, up to K - 1 rows."""
    base = P_pld(m, K, gate, fallback=lambda tr, i, lam, st: (chain_rows(tr, i, lam), True), skip_draft=False)

    def f(tr, i, lam, st):
        rows, dr = base(tr, i, lam, st)
        last = st.get("last")
        if last and last[0] == "pld":
            d, sv = tr.recs[i]
            k = len(rows)
            if k < NDEP and list(d[:k]) == list(rows):
                th = lam * DC
                j = k
                while j < NDEP and len(rows) < K - 1 and sv[j] >= th:
                    rows = rows + [d[j]]
                    j += 1
            st["src"] = "p" * k + "c" * (len(rows) - k)
        return rows, True
    f.name = "h2 pld m%d K%d %s + chain ext" % (m, K, gate)
    return f


def P_h4(K, m=2):
    """The draft pass every step; then PLD's proposal (PldGate) or today's gated chain, whichever expects more
    tokens per modelled ms (the gate's row costs)."""
    def f(tr, i, lam, st):
        A = i + 1
        d, sv = tr.recs[i]
        ch = chain_rows(tr, i, lam)
        e_ch = sum(sv[:len(ch)])
        n, prop = tr.props.get(A, (0, []))
        st["last"] = None
        if n >= m and prop:
            g = st.setdefault("gate", PldGate())
            key = min(n, NMAX)
            k = g.k(key, len(prop), lam, K - 1)
            if k > 0:
                p, e_p = g.q[key], 0.0
                for j in range(k):
                    e_p += p
                    p *= g.c
                if (1 + e_p) / (C1 + k * DC) > (1 + e_ch) / (C1 + len(ch) * DC):
                    st["last"] = ("pld", key, k, 0)
                    st["src"] = "p" * k
                    return prop[:k], True
        st["src"] = "c" * NDEP
        return ch, True
    f.name = "h4 best of pld/chain K%d m%d" % (K, m)
    return f


def ext_after(tr, i, k):
    """PLD's proposal after the anchor and the chain's first k drafts: (n, tokens)."""
    if k == 0:
        return tr.props.get(i + 1, (0, []))
    v = tr.vprops.get(i + 1)
    return v[k - 1] if v else (0, [])


def P_h3(K, m=2):
    """Today's chain, then PLD's continuation after its drafts (history + d1..dk), ungated, up to K - 1 rows."""
    def f(tr, i, lam, st):
        rows = chain_rows(tr, i, lam)
        k = len(rows)
        if k and k < K - 1:
            n, ext = ext_after(tr, i, k)
            if n >= m and ext:
                rows = rows + ext[:K - 1 - k]
        st["src"] = "c" * k + "p" * (len(rows) - k)
        return rows, True
    f.name = "h3 chain+pld-ext K%d m%d" % (K, m)
    return f


def P_h3g(K, m=2):
    """Today's chain, then PLD's continuation after its k drafts, verified while the estimated survival stays
    >= lam * dC: the drafts' survival[k-1] (1 for k = 0) times q(n) for the first token and c for each further one
    (PldGate, learning from the steps whose drafts were all accepted)."""
    def f(tr, i, lam, st):
        d, sv = tr.recs[i]
        rows = chain_rows(tr, i, lam)
        k = len(rows)
        st["last"] = None
        if k < K - 1:
            n, ext = ext_after(tr, i, k)
            if n >= m and ext:
                g = st.setdefault("gate", PldGate())
                key = min(n, NMAX)
                p = (sv[k - 1] if k else 1.0) * g.q[key]
                j = 0
                while j < min(len(ext), K - 1 - k) and p >= lam * DC:
                    j += 1
                    p *= g.c
                if j:
                    rows += ext[:j]
                    st["last"] = ("ext", key, j, k)
        st["src"] = "c" * k + "p" * (len(rows) - k)
        return rows, True
    f.name = "h3g chain+gated pld-ext K%d m%d" % (K, m)
    return f


def P_h3v(K, m=2, agree=False):
    """As h3g, but the extension's length maximises its expected tokens minus lam times its ms (dC a row, and O16
    once the step passes 8 rows). agree: q by (n, whether the drafts equal PLD's own proposal from the anchor)."""
    def f(tr, i, lam, st):
        A = i + 1
        d, sv = tr.recs[i]
        rows = chain_rows(tr, i, lam)
        k = len(rows)
        st["last"] = None
        if k < K - 1:
            n, ext = ext_after(tr, i, k)
            if n >= m and ext:
                g = st.setdefault("gate", PldGate())
                key = min(n, NMAX)
                if agree:
                    p0 = tr.props.get(A, (0, []))[1]
                    key = (key, k > 0 and list(p0[:k]) == list(d[:k]))
                p = (sv[k - 1] if k else 1.0) * g.q[key]
                best_j, best_v, v = 0, 0.0, 0.0
                for j in range(1, min(len(ext), K - 1 - k) + 1):
                    v += p - lam * DC - (lam * O16 if 1 + k + j == 9 else 0.0)
                    if v > best_v:
                        best_j, best_v = j, v
                    p *= g.c
                if best_j:
                    rows += ext[:best_j]
                    st["last"] = ("ext", key, best_j, k)
        st["src"] = "c" * k + "p" * (len(rows) - k)
        return rows, True
    f.name = "h3v chain+value-gated pld-ext K%d m%d%s" % (K, m, " agree" if agree else "")
    return f


def full_key(n, L):
    """n below NMAX; else the match's uncapped length L by bucket (8, 16, 32, 64, 128)."""
    if n < NMAX:
        return n
    for b in [128, 64, 32, 16]:
        if L >= b:
            return b
    return NMAX


def P_h3s(K, m=2):
    """As h3v, with q and c both keyed by the match's uncapped length (full_key): inside a long copy, is the next
    token of it likelier, and the run longer?"""
    def f(tr, i, lam, st):
        A = i + 1
        d, sv = tr.recs[i]
        rows = chain_rows(tr, i, lam)
        k = len(rows)
        st["last"] = None
        if k < K - 1:
            n, ext = ext_after(tr, i, k)
            if n >= m and ext:
                g = st.setdefault("gate", PldGate(per_key_c=True))
                key = full_key(n, tr.ag.get(A, 0) if k == 0 else tr.vag[A][k - 1])
                p, c = (sv[k - 1] if k else 1.0) * g.q[key], g.cof(key)
                best_j, best_v, v = 0, 0.0, 0.0
                for j in range(1, min(len(ext), K - 1 - k) + 1):
                    v += p - lam * DC - (lam * O16 if 1 + k + j == 9 else 0.0)
                    if v > best_v:
                        best_j, best_v = j, v
                    p *= c
                if best_j:
                    rows += ext[:best_j]
                    st["last"] = ("ext", key, best_j, k)
        st["src"] = "c" * k + "p" * (len(rows) - k)
        return rows, True
    f.name = "h3s chain+full-key pld-ext K%d m%d" % (K, m)
    return f


def P_table(K, tab, m=2):
    """Today's chain, then a fixed length of PLD's continuation per key (full_key, whether the chain verified all 5
    drafts): the gate the hindsight search tunes."""
    def f(tr, i, lam, st):
        A = i + 1
        rows = chain_rows(tr, i, lam)
        k = len(rows)
        if k < K - 1:
            n, ext = ext_after(tr, i, k)
            if n >= m and ext:
                key = (full_key(n, tr.ag.get(A, 0) if k == 0 else tr.vag[A][k - 1]), k == NDEP)
                rows += ext[:min(tab.get(key, 0), len(ext), K - 1 - k)]
        st["src"] = "c" * k + "p" * (len(rows) - k)
        return rows, True
    f.name = "table K%d" % K
    return f


def hindsight(trs, costs, bts, sel, Ks=(8, 16)):
    """A bound for gates on these features: P_table's lengths chosen on sel's traces themselves (in-sample), by
    coordinate descent from all zero."""
    names = [tr.name for tr in sel]
    base = rate_of({tr.name: walk(tr, P_chain(), costs[tr.name], bts[group_of(tr.name)]) for tr in sel}, names)
    keys = [(x, f) for x in [2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128] for f in [False, True]]
    out = {}
    for K in Ks:
        js = [0, 1, 2, 3, 4, 5, 6, 7] if K <= 8 else [0, 1, 2, 3, 4, 6, 8, 10, 12, 14]
        tab = {key: 0 for key in keys}

        def gain(t):
            w = {tr.name: walk(tr, P_table(K, t), costs[tr.name], bts[group_of(tr.name)]) for tr in sel}
            return 100 * (rate_of(w, names) / base - 1)
        cur = gain(tab)
        for rnd in range(4):
            moved = False
            for key in keys:
                best = (cur, tab[key])
                for j in js:
                    if j != tab[key] and j <= K - 1 - (NDEP if key[1] else 0):
                        v = gain({**tab, key: j})
                        if v > best[0] + 1e-9:
                            best = (v, j)
                if best[1] != tab[key]:
                    cur, tab[key], moved = best[0], best[1], True
            if not moved:
                break
        out[K] = (cur, {key: j for key, j in tab.items() if j})
        print("hindsight, %d rows at most: %+.1f %% over today's chain; lengths by (key, all 5 drafts): %s" % (
            K, cur, ", ".join("%s%s %d" % (key[0], "*" if key[1] else "", j) for key, j in sorted(out[K][1].items()))))
    return out


def why(trs, costs, bts, sel):
    """docs/engine.md's "Why the gates fall short", on sel's traces: the match length against PLD's right run, walks
    at 16 rows by row source, the experts and ms of a step by its rows, and the gain with cheaper rows."""
    global DC, O16
    names = [tr.name for tr in sel]
    st = collections.defaultdict(lambda: [0, 0, 0, 0])
    for tr in sel:
        for A in range(tr.P, tr.L - 1):
            n, prop = tr.props[A]
            if n >= 2 and prop:
                r = run_len(prop, tr.toks, A, tr.L)
                x = st[full_key(n, tr.ag[A])]
                x[0] += 1
                x[1] += r >= 1
                x[2] += r
                x[3] += r == len(prop)
    tot = sum(x[0] for x in st.values())
    print("\n| match | anchors | P(first right) | mean right run (of up to 15) | whole proposal right |")
    print("|---|---|---|---|---|")
    for key in sorted(st):
        x = st[key]
        print("| %s | %.0f %% | %.2f | %.1f | %.2f |" % (key, 100.0 * x[0] / tot, x[1] / x[0], x[2] / x[0], x[3] / x[0]))
    print("\n| | tok/step | rows/step | steps verifying all 5 drafts | steps extended by PLD | PLD rows/step: accepted, "
          "rejected | steps past 8 rows | tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    base = None
    for label, p in [("today's chain", P_chain()), ("gated (n ≥ 8)", P_h3g(16, 8)), ("bound", P_h3o(16))]:
        g = collections.Counter()
        for tr in sel:
            walk(tr, p, costs[tr.name], bts[group_of(tr.name)], steps_out=g)
        tps = 1e3 * g["tok"] / g["ms"]
        base = base or tps
        s = g["steps"]
        print("| %s | %.2f | %.2f | %.0f %% | %s | %s | %s | %s |" % (
            label, g["tok"] / s, g["rows"] / s, 100.0 * g["c5"] / s, "%.0f %%" % (100.0 * g["p_steps"] / s) if g["p_rows"]
            else "", "%.2f, %.2f" % (g["p_acc"] / s, g["p_rej"] / s) if g["p_rows"] else "",
            "%.0f %%" % (100.0 * g["over8"] / s) if g["over8"] else "",
            "%.1f" % tps if p.name == "chain" else ("%+.1f %%" % (100 * (tps / base - 1))).replace("-", "−")))
    us = collections.defaultdict(list)
    for tr in sel:
        for A in range(tr.P, tr.L - 16, 7):
            for k in range(16):
                us[k].append(tr.U(A, k))
    m = [sum(us[k]) / len(us[k]) for k in range(16)]
    print("\ndistinct experts per layer of 1..16 consecutive own rows:", " ".join("%.1f" % x for x in m))
    print("added by rows 2..16:", " ".join("%.2f" % (m[k] - m[k - 1]) for k in range(1, 16)))
    c = dict(COST)
    c["c0"] += sum(costs[n]["c0"] - COST["c0"] for n in names) / len(names)
    t6 = step_ms(c, 6, m[5], True)
    for R in [6, 8, 9, 12, 16]:
        t = step_ms(c, R, m[R - 1], True)
        print("a step of %2d rows, all accepted: %.1f ms, %.0f tok/s; to match 6 rows it must emit %.1f" % (
            R, t, 1e3 * R / t, 6 * t / t6))
    print("\n| | the best gate | the bound |\n|---|---|---|")
    dc0, o0 = DC, O16
    for label, mod, dc in [("as measured", {}, dc0), ("without the 16-row overhead", {"o16": 0.0}, dc0),
                           ("without it, rows at half cost", {"o16": 0.0, "b_row": COST["b_row"] / 2,
                                                              "c_exp": COST["c_exp"] / 2}, dc0 / 2),
                           ("without it, a row's experts free", {"o16": 0.0, "c_exp": 0.0}, COST["b_row"])]:
        DC, O16 = dc, mod.get("o16", o0)  # the gate's and the rate's own row costs, set to match
        cs = {n: dict(costs[n], **mod) for n in names}
        b = rate_of({tr.name: walk(tr, P_chain(), cs[tr.name], bts[group_of(tr.name)]) for tr in sel}, names)
        v = [rate_of({tr.name: walk(tr, p, cs[tr.name], bts[group_of(tr.name)]) for tr in sel}, names)
             for p in [P_h3v(16, 8), P_h3o(16)]]
        print("| %s | %+.1f %% | %+.1f %% |" % (label, 100 * (v[0] / b - 1), 100 * (v[1] / b - 1)))
    DC, O16 = dc0, o0


def P_chain_oracle():
    """Bound: verify exactly the drafts that are right."""
    def f(tr, i, lam, st):
        d, sv = tr.recs[i]
        r = run_len(d, tr.toks, i + 1, tr.L)
        st["src"] = "c" * r
        return list(d[:r]), True
    f.name = "chain oracle"
    return f


def P_h3o(K):
    """Bound: today's gated chain; when all its drafts are right, exactly the right part of PLD's continuation after
    them, up to K - 1 rows."""
    def f(tr, i, lam, st):
        A = i + 1
        rows = chain_rows(tr, i, lam)
        k = len(rows)
        if run_len(rows, tr.toks, A, tr.L) == k and k < K - 1:
            n, ext = ext_after(tr, i, k)
            if n and ext:
                ext = ext[:K - 1 - k]
                rows += ext[:run_len(ext, tr.toks, A + k, tr.L)]
        st["src"] = "c" * k + "p" * (len(rows) - k)
        return rows, True
    f.name = "h3o chain+pld-ext oracle K%d" % K
    return f


def P_full_oracle(K):
    """Bound with a perfect gate on both sources: per step the best (tokens per modelled ms) of the drafts that are
    right, those drafts and PLD's right continuation after them (up to K - 1 rows), and PLD's right run alone
    (without the draft pass)."""
    def f(tr, i, lam, st):
        A = i + 1
        d, sv = tr.recs[i]
        r = run_len(d, tr.toks, A, tr.L)
        rows_a = list(d[:r])
        rows_b = list(rows_a)
        n, ext = ext_after(tr, i, r)
        if n and ext and r < K - 1:
            ext = ext[:K - 1 - r]
            rows_b += ext[:run_len(ext, tr.toks, A + r, tr.L)]
        prop = tr.props.get(A, (0, []))[1][:K - 1]
        rows_c = prop[:run_len(prop, tr.toks, A, tr.L)]
        best = None
        for rows, src, dr in [(rows_a, "c" * r, True), (rows_b, "c" * r + "p" * (len(rows_b) - r), True),
                              (rows_c, "p" * len(rows_c), "acc")]:
            v = (1 + len(rows)) / step_ms(st["cost"], 1 + len(rows), tr.U(A, len(rows)), dr)
            if best is None or v > best[0]:
                best = (v, rows, src, dr)
        st["src"] = best[2]
        return best[1], best[3]
    f.name = "full oracle K%d" % K
    return f


# ---- the walk ----------------------------------------------------------------------------------------------------
def walk(tr, policy, cost, beta, log=None, steps_out=None):
    """Step 0: one row at the prompt's last position. Then each step at record i (anchor i + 1) verifies the
    policy's rows and emits the accepted ones and one more; lambda lags two steps (spec_loop queues a step ahead).
    A step's U: the own rows' distinct experts, plus beta[source] for each rejected row. steps_out (a Counter)
    adds up the steps by row source (why())."""
    rate = Rate()
    lams = [rate.lam()]
    st = {"cost": cost}
    i = tr.first
    end = tr.L - 1
    tok, ms = 1, step_ms(cost, 1, 6, True)
    steps, n = 1, 0
    ext_steps = ext_tok = 0
    hist = collections.Counter()
    rate.add(1, 1)
    lams.append(rate.lam())
    if steps_out is not None:  # step 0
        for key, v in [("tok", 1), ("ms", ms), ("steps", 1), ("rows", 1)]:
            steps_out[key] += v
    while i in tr.recs and i + 1 < end:
        A = i + 1
        rows, dr = policy(tr, i, lams[max(0, n - 1)], st)
        n += 1
        a = run_len(rows, tr.toks, A, tr.L)
        e = min(a + 1, tr.L - 1 - A)
        k = len(rows)
        src = st.get("src") or "c" * k
        U = tr.U(A, k) + sum(beta[src[r]] for r in range(a, k))
        t = step_ms(cost, 1 + k, U, dr)
        ms += t
        if log is not None:
            log.append((1 + k, t))
        tok += e
        steps += 1
        hist[k] += 1
        if steps_out is not None:
            nc, npl = src[:k].count("c"), src[:k].count("p")
            acc_p = max(0, a - nc) if npl else 0
            for key, v in [("tok", e), ("ms", t), ("steps", 1), ("rows", 1 + k), ("c5", nc == NDEP),
                           ("p_steps", npl > 0), ("p_rows", npl), ("p_acc", acc_p), ("p_rej", npl - acc_p),
                           ("over8", 1 + k > 8)]:
                steps_out[key] += v
        last = st.get("last")
        if last:
            kind, key, j, k0 = last
            if a >= k0:  # the rows before PLD's were right: its outcome is seen
                ext_steps += 1
                ext_tok += a - k0
                st["gate"].learn(key, j, a - k0) if "gate" in st else None
        rate.add(e, 1 + k)
        lams.append(rate.lam())
        i = i + 1 + a
    return dict(tok=tok, ms=ms, tps=1e3 * tok / ms, steps=steps, per_step=tok / steps, pld_steps=ext_steps,
                pld_tok=ext_tok, hist=dict(hist))


# ---- calibration against measured speed -------------------------------------------------------------------------
def gpu_by_rows(rep):
    """pass 1's 'spec step GPU ms (rows, draft pass) by rows: R: ms (n) ...' -> {R: (ms, n)}"""
    s = rep[rep.find("by rows:") + 8:]
    return {int(m.group(1)): (float(m.group(2)), int(m.group(3))) for m in re.finditer(r"(\d+): ([\d.]+) \((\d+)\)", s)}


def offsets(trs, meta, bts, verbose=False):
    """c0's offset per trace: for a turn, pass 1's measured GPU ms by row count against the replayed chain's model;
    for a bench type, the offset that brings the replayed chain to bench.py's decode (meta's bench_tps)."""
    offs = {}
    for tr in trs:
        b = bts[group_of(tr.name)]
        if tr.p1:
            log = []
            w = walk(tr, P_chain(), COST, b, log=log)
            rep = " ".join(tr.p1["report"])
            meas = gpu_by_rows(rep)
            by = collections.defaultdict(list)
            for R, t in log:
                by[R].append(t)
            num = den = 0.0
            cells = []
            for R in sorted(meas):
                if by.get(R):
                    mod = sum(by[R]) / len(by[R])
                    cells.append("%d: %.2f/%.2f" % (R, meas[R][0], mod))
                    num += meas[R][1] * (meas[R][0] - mod)
                    den += meas[R][1]
            offs[tr.name] = num / den if den else 0.0
            if verbose:
                h = rep[rep.find("steps verifying 0..5 drafts:") + 28:].split("verify")[0].split()
                print("%-8s replay: steps by drafts %s, engine %s | GPU ms measured/model by rows %s | offset %+.2f" % (
                    tr.name, [w["hist"].get(k, 0) for k in range(6)], [int(x) for x in h[:6]], " ".join(cells),
                    offs[tr.name]))
        elif tr.name in meta.get("bench_tps", {}):
            t = meta["bench_tps"][tr.name]
            w = walk(tr, P_chain(), COST, b)
            offs[tr.name] = (1e3 * w["tok"] / t - w["ms"]) / w["steps"]
            if verbose:
                print("%-14s chain model %.1f tok/s, bench.py %.1f: offset %+.2f ms per step" % (
                    tr.name, w["tps"], t, offs[tr.name]))
    # traces with a measured decode rate of their own (editbench.py's decode_tps): one offset for them all, which
    # brings the replayed chain's total time to the measured one (short turns' rates are noisy)
    sel = [tr for tr in trs if tr.tps and tr.name not in offs]
    if sel:
        ws = {tr.name: walk(tr, P_chain(), COST, bts[group_of(tr.name)]) for tr in sel}
        meas = sum(1e3 * (tr.L - tr.P) / tr.tps for tr in sel)
        mod = sum(w["ms"] * (tr.L - tr.P) / w["tok"] for tr, w in zip(sel, ws.values()))
        o = (meas - mod) / sum(w["steps"] for w in ws.values())
        for tr in sel:
            offs[tr.name] = o
        if verbose:
            print("%d traces with their own decode rates: chain model %.1f tok/s, measured %.1f: offset %+.2f ms per step"
                  % (len(sel), 1e3 * sum(tr.L - tr.P for tr in sel) / mod, 1e3 * sum(tr.L - tr.P for tr in sel) / meas, o))
            hr = [sum(ws[tr.name]["hist"].get(k, 0) for tr in sel) for k in range(6)]
            hm = [sum(tr.hist[k] for tr in sel if tr.hist) for k in range(6)]
            print("  their steps verifying 0..5 drafts: replay %s, served %s" % (hr, hm))
    return offs


# ---- Q1: how much is copy ----------------------------------------------------------------------------------------
KS = [1, 2, 4, 8, 15]
CAPS = [7, 15, 63]
BK = ["0", "1", "2-3", "4-7", "8-15"]


def bucket(r):
    return "0" if r == 0 else "1" if r == 1 else "2-3" if r < 4 else "4-7" if r < 8 else "8-15"


def pld_walk(toks, pr, cap, lo, hi):
    """PLD alone: at anchor a verify min(cap, len) proposal tokens; emit the right run and one more."""
    steps = copied = 0
    h = collections.Counter()
    a = lo
    while a < hi - 1:
        r = run_len(pr.get(a, (0, []))[1][:cap], toks, a, hi)
        e = min(r + 1, hi - 1 - a)
        steps += 1
        copied += min(r, e)
        h[bucket(r)] += 1
        a += e
    return steps, copied, h


class Copy:
    def __init__(s):
        s.out = s.anchors = 0
        s.pk, s.cop, s.steps, s.h = (collections.Counter() for _ in range(4))
        s.think = [0, 0, 0, 0]  # thinking tokens, copied; the rest, copied (cap 15)
        s.runs = [0, 0]         # output tokens in right runs of >= 8, >= 16 (proposals followed across anchors)
        s.byn = collections.defaultdict(lambda: [0, 0, 0])  # n -> anchors, first token right, sum of runs (cap 15)

    def add(s, o):
        s.out += o.out
        s.anchors += o.anchors
        for x, y in [(s.pk, o.pk), (s.cop, o.cop), (s.steps, o.steps), (s.h, o.h)]:
            x.update(y)
        for j in range(4):
            s.think[j] += o.think[j]
        for j in range(2):
            s.runs[j] += o.runs[j]
        for n, v in o.byn.items():
            for j in range(3):
                s.byn[n][j] += v[j]


def measure_copy(toks, P, nmin=2, virt=()):
    o = Copy()
    L = len(toks)
    o.out = L - P
    mt = Matcher(toks, virt)
    pr = {}
    for a in range(P, L):
        pr[a] = mt.at(a).lookup(toks[max(0, a - NMAX + 1):a + 1], a, nmin, 63)
    for a in range(P, L - 1):
        n, p = pr[a]
        r = run_len(p, toks, a, L)
        o.anchors += 1
        for k in KS:
            o.pk[k] += r >= k
        b = o.byn[n]
        b[0] += 1
        b[1] += r >= 1
        b[2] += min(r, 15)
    for cap in CAPS:
        s, c, h = pld_walk(toks, pr, cap, P, L)
        o.cop[cap], o.steps[cap] = c, s
        if cap == 15:
            o.h = h
    a = P
    while a < L - 1:  # right runs, followed across anchors while each proposal is right to its end
        n, p = pr[a]
        r = run = run_len(p, toks, a, L)
        b = a + r
        while r == len(p) and r > 0 and b < L - 1:
            n, p = pr[b]
            r = run_len(p, toks, b, L)
            run += r
            b += r
        o.runs[0] += run if run >= 8 else 0
        o.runs[1] += run if run >= 16 else 0
        a = b + 1
    te = next((p + 1 for p in range(P, L) if toks[p] == THINK_END), None)
    if te is not None and te < L - 1:
        o.think = [te - P, pld_walk(toks, pr, 15, P, te)[1], L - te, pld_walk(toks, pr, 15, te, L)[1]]
    else:
        o.think = [0, 0, L - P, o.cop[15]]
    return o


def copy_line(name, o):
    pc = lambda x, y: "%5.1f" % (100.0 * x / max(1, y))
    return "%-17s %5d | %s | %s | %4.2f | %s | %s %s | %5d %s | %5d %s" % (
        name, o.out, " ".join(pc(o.pk[k], o.anchors) for k in KS), " ".join(pc(o.cop[c], o.out) for c in CAPS),
        o.out / max(1, o.steps[15]), " ".join("%4d" % o.h[b] for b in BK), pc(o.runs[0], o.out), pc(o.runs[1], o.out),
        o.think[0], pc(o.think[1], o.think[0]), o.think[2], pc(o.think[3], o.think[2]))


def copy_tables(trs):
    print("PLD: the longest match, n = 8 .. 2, its most recent occurrence")
    print("%-17s %5s | %s | copied %% at caps %s | tok/step | steps by right run (cap 15) %s | %% in runs >= 8, 16 | "
          "thinking: tokens, copied %% | the rest" % ("trace", "out", " ".join("r>=%-2d" % k for k in KS),
                                                     "/".join(map(str, CAPS)), " ".join(BK)))
    G = collections.OrderedDict()
    for tr in trs:
        o = measure_copy(tr.toks, tr.P, virt=tr.virt)
        print(copy_line(tr.name[-17:], o))
        for g, f in GROUPS:
            if f(tr):
                G.setdefault(g, Copy()).add(o)
    print()
    for g, o in G.items():
        print(copy_line(g, o))
    print("\nby the match's length n (cap 15): the anchors' share, P(first token right), mean right run")
    for g, o in G.items():
        cells = []
        for n in range(0, NMAX + 1):
            b = o.byn.get(n)
            if b and b[0]:
                cells.append("n%d %4.1f%% %3.0f%% %4.1f" % (n, 100 * b[0] / o.anchors, 100 * b[1] / b[0], b[2] / b[0]))
        print("%-17s %s" % (g, " | ".join(cells)))


# ---- Q2: hybrids against today's chain ---------------------------------------------------------------------------
def policies(K):
    ps = [P_chain(), P_pld(2, K, "fixed"), P_pld(2, K, "ema"), P_pld(2, K, "oracle")]
    ps += [P_h1(m, K) for m in [2, 4, 8]]
    ps += [P_h2(2, K), P_h4(K), P_h3(K)]
    ps += [P_h3g(K, m) for m in [2, 4, 8]]
    ps += [P_h3v(K, m) for m in [2, 8]] + [P_h3v(K, 8, True)] + [P_h3s(K, m) for m in [2, 8]]
    ps += [P_chain_oracle(), P_h3o(K), P_full_oracle(K)]
    return ps


GROUPS = [("bench", lambda tr: tr.name.startswith("bench")),
          ("turns", lambda tr: tr.name.startswith("turn")),
          ("tool-call turns", lambda tr: tr.name.startswith("turn") and tr.tool),
          ("other turns", lambda tr: tr.name.startswith("turn") and not tr.tool),
          ("edit traces", lambda tr: tr.name.startswith("edit")),
          ("edit, write", lambda tr: tr.name.startswith("edit") and tr.kind in ("edit", "write")),
          ("read, prose", lambda tr: tr.name.startswith("edit") and tr.kind in ("read", "prose")),
          ("thinking off", lambda tr: tr.name.startswith("edit-off")),
          ("thinking max", lambda tr: tr.name.startswith("edit-max"))]


def run_policies(trs, costs, bts, Ks=(8, 16), only=None):
    res = collections.OrderedDict()
    for K in Ks:
        for p in policies(K):
            if only and p.name != "chain" and not any(x in p.name for x in only):
                continue
            res[(K, p.name)] = {tr.name: walk(tr, p, costs[tr.name], bts[group_of(tr.name)]) for tr in trs}
    return res


def rate_of(w, names):
    return 1e3 * sum(w[n]["tok"] for n in names) / sum(w[n]["ms"] for n in names)


def policy_table(trs, res):
    base = res[(8, "chain")]
    cols = [tr.name for tr in trs]
    print("%-46s %s | %s" % ("policy (tok/s; % over today's chain)", " ".join("%8s" % c[-8:] for c in cols),
                             " ".join("%15s" % g for g, _ in GROUPS)))
    for (K, nm), w in res.items():
        chain = nm == "chain"
        cells = ["%8s" % ("%.1f" % w[c]["tps"] if chain else "%+.1f%%" % (100 * (w[c]["tps"] / base[c]["tps"] - 1)))
                 for c in cols]
        gs = []
        for g, f in GROUPS:
            sel = [tr.name for tr in trs if f(tr)]
            if not sel:
                gs.append("%15s" % "-")
                continue
            v, b = rate_of(w, sel), rate_of(base, sel)
            gs.append("%15s" % ("%.1f" % v if chain else "%+.1f%%" % (100 * (v / b - 1))))
        print("%-46s %s | %s" % (("K%d " % K if not chain else "") + nm, " ".join(cells), " ".join(gs)))


DOC_ROWS = [
    ("today's chain (tok/s)", "chain"),
    ("PLD alone, the whole proposal", "pld m2 K%d fixed"),
    ("PLD alone, gated", "pld m2 K%d ema"),
    ("PLD alone, exactly its right run (bound)", "pld m2 K%d oracle"),
    ("PLD when n ≥ 4, else the chain; no draft pass on PLD steps", "h1 pld m4 K%d ema, else chain"),
    ("the same with n ≥ 8", "h1 pld m8 K%d ema, else chain"),
    ("PLD, then the chain's drafts where they agree", "h2 pld m2 K%d ema + chain ext"),
    ("per step, PLD or the chain by expected tokens per ms", "h4 best of pld/chain K%d m2"),
    ("the chain, then PLD's continuation after its drafts, ungated", "h3 chain+pld-ext K%d m2"),
    ("the chain, then PLD's continuation, gated (n ≥ 2)", "h3g chain+gated pld-ext K%d m2"),
    ("the same, n ≥ 8", "h3g chain+gated pld-ext K%d m8"),
    ("the same, value-gated (n = 8)", "h3v chain+value-gated pld-ext K%d m8"),
    ("the same, q also by agreement", "h3v chain+value-gated pld-ext K%d m8 agree"),
    ("the same, q and c by the match's uncapped length (n ≥ 2)", "h3s chain+full-key pld-ext K%d m2"),
    ("the same, n ≥ 8", "h3s chain+full-key pld-ext K%d m8"),
    ("bound: the chain, then exactly PLD's right continuation", "h3o chain+pld-ext oracle K%d"),
    ("bound: a perfect gate on the drafts alone", "chain oracle"),
    ("bound: a perfect gate on the drafts and PLD", "full oracle K%d"),
]


def doc_copy_tables(trs):
    """Q1's tables: per group the anchors' P(the proposal's first k tokens right), the PLD-only walk's copied share
    and its steps by right run, and the output in right runs of >= 8 and >= 16."""
    groups = [("prose", lambda tr: tr.name == "bench-prose"), ("code", lambda tr: tr.name == "bench-code"),
              ("agentic", lambda tr: tr.name == "bench-agentic"), ("JSON", lambda tr: tr.name == "bench-json"),
              ("the transcript", GROUPS[1][1]), ("its tool-call turns", GROUPS[2][1]), ("its other turns", GROUPS[3][1]),
              ("edit traces", GROUPS[4][1]),
              ("their edit calls", lambda tr: tr.name.startswith("edit") and tr.kind == "edit"),
              ("their write calls", lambda tr: tr.name.startswith("edit") and tr.kind == "write"),
              ("their read calls", lambda tr: tr.name.startswith("edit") and tr.kind == "read"),
              ("their prose answers", lambda tr: tr.name.startswith("edit") and tr.kind == "prose"),
              ("thinking off", GROUPS[7][1]), ("thinking max", GROUPS[8][1])]
    G = collections.OrderedDict((g, Copy()) for g, _ in groups)
    R = collections.OrderedDict((g, Copy()) for g, _ in groups)
    for tr in trs:
        o = measure_copy(tr.toks, tr.P, virt=tr.virt)
        o_raw = measure_copy(tr.toks, tr.P) if tr.virt else o
        for g, f in groups:
            if f(tr):
                G[g].add(o)
                R[g].add(o_raw)
    pc = lambda x, y: "%.1f" % (100.0 * x / max(1, y))
    print("\n| | output | P(next 1, 2, 4, 8, 15 right), % | copied at caps 7, 15, 63, % | in runs ≥ 8, ≥ 16, % "
          "| tok/step, PLD alone | raw matcher: copied, in runs ≥ 16, % |")
    print("|---|---|---|---|---|---|---|")
    for g, o in G.items():
        if not o.out:
            continue
        r = R[g]
        print("| %s | %d | %s | %s | %s | %.2f | %s, %s |" % (
            g, o.out, ", ".join(pc(o.pk[k], o.anchors) for k in KS), ", ".join(pc(o.cop[c], o.out) for c in CAPS),
            ", ".join(pc(x, o.out) for x in o.runs), o.out / max(1, o.steps[15]), pc(r.cop[15], r.out),
            pc(r.runs[1], r.out)))
    print("\n| | steps by right run (cap 15): 0 | 1 | 2–3 | 4–7 | 8–15 |")
    print("|---|---|---|---|---|---|")
    for g, o in G.items():
        if o.out:
            print("| %s | %s |" % (g, " | ".join("%d" % o.h[b] for b in BK)))


def doc_tables(trs, res):
    cols = [("prose", ["bench-prose"]), ("code", ["bench-code"]), ("agentic", ["bench-agentic"]),
            ("JSON", ["bench-json"]), ("transcript", [t.name for t in trs if t.name.startswith("turn")]),
            ("its tool-call turns", [t.name for t in trs if t.name.startswith("turn") and t.tool]),
            ("its other turns", [t.name for t in trs if t.name.startswith("turn") and not t.tool])]
    ecols = [(g, [t.name for t in trs if f(t)]) for g, f in GROUPS[4:]]
    base = res[(8, "chain")]
    for cs, what in [(cols, "the bench and the transcript"), (ecols, "the edit traces")]:
        cs = [(c, n) for c, n in cs if n]
        if not cs:
            continue
        for K in [8, 16]:
            print("\n**%s, %d rows at most** (%% over today's chain)\n" % (what, K))
            print("| | " + " | ".join(c for c, _ in cs) + " |")
            print("|---|" + "---|" * len(cs))
            for label, key in DOC_ROWS:
                k = (K, key % K if "%d" in key else key)
                if k not in res:
                    continue
                cells = []
                for c, names in cs:
                    v, b = rate_of(res[k], names), rate_of(base, names)
                    cells.append("%.1f" % v if key == "chain" else ("%+.1f %%" % (100 * (v / b - 1))).replace("-", "−"))
                print("| %s | %s |" % (label, " | ".join(cells)))
    print("\n| turn | context | output | tool call | copied (cap 15) | in runs ≥ 16 | today's chain, tok/s "
          "(pass 1) | gated ext (n ≥ 2), 8 rows | gated ext (n ≥ 8), 16 rows | ext bound, 16 rows |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tr in trs:
        if not tr.name.startswith("turn"):
            continue
        o = measure_copy(tr.toks, tr.P, virt=tr.virt)
        b = base[tr.name]["tps"]
        g = lambda k: ("%+.1f %%" % (100 * (res[k][tr.name]["tps"] / b - 1))).replace("-", "−")
        print("| %s | %d | %d | %s | %.0f %% | %.0f %% | %.1f (%.1f) | %s | %s | %s |" % (
            tr.name[5:], tr.P, tr.L - tr.P, "yes" if tr.tool else "", 100 * o.cop[15] / o.out, 100 * o.runs[1] / o.out,
            b, tr.p1["tps"] if tr.p1 else 0, g((8, "h3g chain+gated pld-ext K8 m2")),
            g((16, "h3g chain+gated pld-ext K16 m8")), g((16, "h3o chain+pld-ext oracle K16"))))
    tasks = collections.OrderedDict()
    for tr in trs:
        if tr.name.startswith("edit"):
            tasks.setdefault(tr.name.rsplit("-", 1)[0], []).append(tr)
    if not tasks:
        return
    print("\n| thinking, task | turns | output | copied (cap 15) | in runs ≥ 16 | today's chain, tok/s (served) "
          "| gated ext (n ≥ 2), 8 rows | gated ext (n ≥ 8), 16 rows | ext bound, 16 rows |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t, sel in tasks.items():
        o = Copy()
        for tr in sel:
            o.add(measure_copy(tr.toks, tr.P, virt=tr.virt))
        names = [tr.name for tr in sel]
        b = rate_of(base, names)
        meas = sum(tr.L - tr.P for tr in sel) / sum((tr.L - tr.P) / tr.tps for tr in sel) if all(tr.tps for tr in sel) else 0
        g = lambda k: ("%+.1f %%" % (100 * (rate_of(res[k], names) / b - 1))).replace("-", "−")
        print("| %s | %d | %d | %.0f %% | %.0f %% | %.1f (%.1f) | %s | %s | %s |" % (
            t[5:].replace("-", ", ", 1), len(sel), o.out, 100 * o.cop[15] / o.out, 100 * o.runs[1] / o.out, b, meas,
            g((8, "h3g chain+gated pld-ext K8 m2")), g((16, "h3g chain+gated pld-ext K16 m8")),
            g((16, "h3o chain+pld-ext oracle K16"))))


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(USAGE)
    D = sys.argv[1]
    trs, meta = load_all(D)
    if "--check" in sys.argv:
        tree = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else None
        for tr in trs:
            miss_r = sum(1 for i in range(tr.first, tr.L - 1) if i not in tr.recs)
            miss_m = sum(1 for p in range(tr.P, tr.L) if p not in tr.masks)
            msg = "%-14s records %d (missing %d), routed positions %d (missing %d), exact records %d" % (
                tr.name, len(tr.recs), miss_r, len(tr.masks), miss_m, len(tr.exact))
            if tree and tr.name.startswith("bench"):
                tp = load_tp(os.path.join(tree, "tp_%s.jsonl" % tr.name[6:]))
                same = sum(1 for i, v in tr.recs.items() if i in tp and list(v[0]) == list(tp[i][0])
                           and all(abs(x - y) < 1e-4 for x, y in zip(v[1], tp[i][1])))
                msg += "; equal to the tree study's records: %d of %d" % (same, len(tp))
            print(msg)
    if "--copy" in sys.argv:
        copy_tables(trs)
    if "--cal" in sys.argv:
        for g in ["bench", "turns", "edit"]:
            sel = [tr for tr in trs if group_of(tr.name) == g]
            if not sel:
                continue
            for kinds, nm in [((1,), "drafts"), ((2,), "PLD proposals")]:
                beta, by = calibrate(sel, kinds)
                print("%s, %s: beta %+.3f experts per rejected row" % (g, nm, beta))
                for (k, off), (n, m) in by.items():
                    if n >= 20 and off:
                        print("   rows %2d, rejected %2d: n %4d, exact - model %+.2f" % (k + 1, off, n, m))
    if any(x in sys.argv for x in ["--replay", "--policies", "--tables", "--hindsight", "--why"]):
        bts = betas(trs, verbose="--replay" in sys.argv)
        offs = offsets(trs, meta, bts, verbose="--replay" in sys.argv)
        costs = {}
        for tr in trs:
            costs[tr.name] = dict(COST)
            costs[tr.name]["c0"] += offs.get(tr.name, 0.0)
        if "--policies" in sys.argv or "--tables" in sys.argv:
            only = None
            for x in sys.argv:
                if x.startswith("--only="):
                    only = x[7:].split(",")
            res = run_policies(trs, costs, bts, only=only)
            if "--policies" in sys.argv:
                policy_table(trs, res)
            if "--tables" in sys.argv:
                doc_copy_tables(trs)
                doc_tables(trs, res)
        if "--hindsight" in sys.argv:
            sel = [tr for tr in trs if tr.name.startswith("edit")] or trs
            hindsight(trs, costs, bts, sel)
        if "--why" in sys.argv:
            why(trs, costs, bts, [tr for tr in trs if tr.name.startswith("edit")] or trs)


if __name__ == "__main__":
    main()
