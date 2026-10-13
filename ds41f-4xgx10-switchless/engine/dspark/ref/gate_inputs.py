#!/usr/bin/env python3
# The draft gate under sampling (docs/engine.md, "The draft gate under sampling"): the inputs of the ring passes
# and of gate_study.py, in one directory DIR, laid out as pld_study.py reads it (meta.json's "runs", spectf's
# sequence lists sl_RUN.txt). Each trace is a golden-style directory (tokens.npy: the prompt, then the output that
# was served) with its served decode rate and steps; spectf teacher-forces the output, so its records give the
# acceptance of the decoding that made it (sampled or greedy). No numpy.
#   gate_inputs.py edit DIR RUN INDEX.jsonl [--sel off:0,max:0] [--tag r2] [--sampling greedy|default]
#       one run of an editbench.py index (--traces): its turns of those thinking settings and repeats (default all)
#   gate_inputs.py served DIR RUN INDEX.jsonl [--tag sd] [--sampling greedy|default]
#       one run of serve_traces.py's index (prompts sent through dsfront, the same fields)
# --sampling: how the outputs were decoded, greedy or dsfront's defaults (T 1.0, top-p 0.95; the default)
import json, os, sys

USAGE = ("gate_inputs.py edit|served DIR RUN INDEX.jsonl [--sel THINKING:REP,...] [--tag TAG] [--sampling greedy|default]")


def meta_of(D):
    p = os.path.join(D, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def opt(args, k, d=None):
    return args[args.index(k) + 1] if k in args else d


def add_run(D, run, rows, name, sampling):
    meta = meta_of(D)
    seqs = []
    with open(os.path.join(D, "sl_%s.txt" % run), "w") as fl:
        for r in rows:
            g = r["golden"]
            fl.write("%s %d 0\n" % (g, r["n_prompt"]))
            seqs.append(dict(seq=len(seqs), name=name(r), golden=g, prompt=r["n_prompt"], kind=r.get("kind"),
                             tps=r.get("decode_tps"), hist=r.get("hist"), thinking=r.get("thinking"), rep=r.get("rep"),
                             task=r.get("task"), turn=r.get("turn"), n_out=r.get("n_out"), gate=r.get("gate"),
                             vsteps=r.get("vsteps"), sampling=sampling))
    meta.setdefault("runs", {})[run] = seqs
    json.dump(meta, open(os.path.join(D, "meta.json"), "w"), indent=1)
    print("%s: %d sequences, %d prompt tokens, %d output tokens" % (
        run, len(seqs), sum(r["n_prompt"] for r in rows), sum(r.get("n_out") or 0 for r in rows)))


def main():
    a = sys.argv[1:]
    if len(a) < 4 or a[0] not in ("edit", "served"):
        sys.exit(USAGE)
    cmd, D, run, index = a[:4]
    os.makedirs(D, exist_ok=True)
    base = os.path.dirname(os.path.abspath(index))
    rows = [json.loads(l) for l in open(index)]
    sel, tag, sampling = opt(a, "--sel"), opt(a, "--tag", run), opt(a, "--sampling", "default")
    if sel:
        keep = {(t, int(k)) for t, k in (x.split(":") for x in sel.split(","))}
        rows = [r for r in rows if (r.get("thinking"), r.get("rep", 0)) in keep]
    for r in rows:
        r["golden"] = os.path.join(base, r["dir"])
    if cmd == "edit":  # pld_study.py's group_of(): "edit..." names are the edit traces
        add_run(D, run, rows, lambda r: "edit-%s-%s-rep%d-%s-t%02d" % (tag, r["thinking"], r.get("rep", 0), r["task"],
                                                                       r["turn"]), sampling)
    else:
        add_run(D, run, rows, lambda r: "%s-%s-%s" % (tag, r["set"], r["name"]), sampling)


if __name__ == "__main__":
    main()
