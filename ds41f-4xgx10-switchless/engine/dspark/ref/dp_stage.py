#!/usr/bin/env python3
# A build stage of bundle B (docs/engine.md, "The draft pass's approximations: a study") against the study's model.
# The stage's spectf replay (its kernels in place of DSPARK_SIM's rounded weights, over the study's sequences:
# tf_VARIANT.log) goes through today's chain (dp_study.py's replay) with the stage's measured saving a pass, beside
# what the study modelled for it: the records it stood on (--model-recs, a DSPARK_SIM variant) at the modelled saving.
#   dp_stage.py STUDYJOB STAGEJOB VARIANT --saving MS --model MS [--model-recs V] [--gate DIR] [--pld DIR]
# STUDYJOB: dp_study.py's job (seqs.json, sl_all.txt, tf_none.log, tf_V.log); STAGEJOB: tf_VARIANT.log. No numpy.
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp_study as D  # noqa: E402
import pld_study as P  # noqa: E402


def equal_counts(log):
    """{(seq, i): drafts equal to the target's tokens, from the first} (spectf's 'n equal')."""
    pat = re.compile(r"^(?:seq (\d+) )?spectf (\d+): drafts [\d ]+, survival [\d. ]+, (\d+) equal")
    out = {}
    for line in open(log, errors="replace"):
        m = pat.match(line)
        if m:
            out[(int(m.group(1) or 0), int(m.group(2)))] = int(m.group(3))
    return out


def differ(seqs, groups, recs, ref):
    """{group: (records compared, drafts differing)}, and the records of recs missing from ref or the reverse."""
    out = {g: [0, 0] for g in groups}
    miss = 0
    for s in seqs:
        r, b = recs.get(s["seq"], {}), ref.get(s["seq"], {})
        miss += len(set(r) ^ set(b))
        c = out[D.group_of_seq(s)]
        for i, rr in r.items():
            if i in b:
                c[0] += 1
                c[1] += list(rr[0]) != list(b[i][0])
    return out, miss


def main():
    a = sys.argv
    if len(a) < 4 or a[1].startswith("-"):
        sys.exit("dp_stage.py STUDYJOB STAGEJOB VARIANT --saving MS --model MS [--model-recs V] [--gate DIR] "
                 "[--pld DIR]")
    study, stage, var = a[1:4]
    s_meas = float(D.opt(a, "--saving", 0.0))
    s_model = float(D.opt(a, "--model", 0.0))
    mv = D.opt(a, "--model-recs", "fp4head")
    seqs, trs, bts, offs = D.load(study, D.opt(a, "--gate", D.GATE), D.opt(a, "--pld", D.PLD))
    groups = [g for g in D.ORDER if any(D.group_of_seq(s) == g for s in seqs)]
    base = P.load_spectf(os.path.join(study, "tf_none.log"))
    model = P.load_spectf(os.path.join(study, "tf_%s.log" % mv))
    recs = P.load_spectf(os.path.join(stage, "tf_%s.log" % var))

    print("### The records: drafts differing (any of the five) from today's, and from %s's" % mv)
    print("| against | steps | missing either side | " + " | ".join(groups) + " |")
    print("|---|---|---|" + "---|" * len(groups))
    for nm, ref in (("today's (none)", base), (mv, model)):
        d, miss = differ(seqs, groups, recs, ref)
        print("| %s | %d | %d | " % (nm, sum(len(x) for x in recs.values()), miss) + " | ".join(
            "%.1f %%" % (100.0 * d[g][1] / d[g][0]) if d[g][0] else "-" for g in groups) + " |")

    # teacher-forced: the drafts right from the first, a position (the walk's noise aside)
    eqs = [("today's", equal_counts(os.path.join(study, "tf_none.log"))),
           (mv, equal_counts(os.path.join(study, "tf_%s.log" % mv))),
           (var, equal_counts(os.path.join(stage, "tf_%s.log" % var)))]
    grp = {s["seq"]: D.group_of_seq(s) for s in seqs}
    print("\n### Teacher-forced: drafts equal to the target's tokens from the first, a position")
    print("| | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * len(groups))
    for nm, e in eqs:
        cells = []
        for g in groups:
            ks = [k for k in eqs[0][1] if grp.get(k[0]) == g and k in e]
            cells.append("%.3f" % (sum(e[k] for k in ks) / len(ks)) if ks else "-")
        print("| %s | " % nm + " | ".join(cells) + " |")
    e1, e2 = eqs[1][1], eqs[2][1]
    print("| %s against %s: positions worse, better | " % (var, mv) + " | ".join(
        "%d, %d" % (sum(1 for k in e2 if grp.get(k[0]) == g and k in e1 and e2[k] < e1[k]),
                    sum(1 for k in e2 if grp.get(k[0]) == g and k in e1 and e2[k] > e1[k])) for g in groups) + " |")

    p0, pq, pm = {}, {}, {}
    r0 = D.replay(seqs, trs, bts, offs, base, 0.0, p0)
    rq = D.replay(seqs, trs, bts, offs, recs, 0.0, pq)
    rs = D.replay(seqs, trs, bts, offs, recs, s_meas)
    rx = D.replay(seqs, trs, bts, offs, recs, s_model)
    rm = D.replay(seqs, trs, bts, offs, model, s_model, pm)
    sd = {g: D.spread(seqs, p0, pq, g) for g in groups}
    print("\n### Today's chain replayed: accepted drafts a step; the gain in tok/s over today's drafter")
    print("| | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * len(groups))
    print("| today's: acc/step, tok/s | " + " | ".join(
        "%.2f, %.1f" % (D.acc(r0, g), 1e3 * r0[g][0] / r0[g][1] if r0[g][2] else float("nan")) for g in groups) + " |")
    print("| %s at no saving (its acceptance alone) | " % var + " | ".join(
        "%.2f, %+.2f %%%s" % (D.acc(rq, g), D.gain(rq, r0, g), " +- %.2f" % sd[g] if sd[g] is not None else "")
        for g in groups) + " |")
    for nm, r in (("%s, the measured saving (-%.3f ms)" % (var, s_meas), rs),
                  ("%s, the modelled saving (-%.3f ms)" % (var, s_model), rx),
                  ("the study's model: %s's records, -%.3f ms" % (mv, s_model), rm)):
        print("| %s | " % nm + " | ".join("%.2f, %+.2f %%" % (D.acc(r, g), D.gain(r, r0, g)) for g in groups) + " |")


if __name__ == "__main__":
    main()
