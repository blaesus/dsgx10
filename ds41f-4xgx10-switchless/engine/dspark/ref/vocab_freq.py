#!/usr/bin/env python3
# The draft pass's vocabulary subset (docs/engine.md, "The draft pass's approximations: a study"): token counts
# from public text only, for vocab_study.py. Each source is a list of text files, tokenised together through
# dsfront's tokenizer (`dsfront tokenize`: the HF tokenizer's ids), or token-id files counted as they are
# (golden/*.tokens.json; engram's tokenize_corpus.py streams, *.i32). No user data: the repository, the golden
# transcripts, the bench and catalog prompts, and public corpora (the Python stdlib, /usr/share/doc).
#   vocab_freq.py OUT.json DSFRONT TOKENIZER NAME=text:LISTFILE|NAME=ids:FILE[,FILE...] ...
# LISTFILE: one path per line. OUT.json: {"vocab": 129280, "sources": {NAME: {"files", "chars", "tokens",
# "counts": [V]}}}.
import json, os, re, struct, subprocess, sys, tempfile

V = 129280  # the head's rows: 128000 BPE tokens and 1280 added ones


def tokenize(dsfront, tok, paths):
    """The files' text, each headed by a blank line, through dsfront: (ids, files read, chars)."""
    parts, n = [], 0
    for p in paths:
        try:
            s = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        parts.append(s)
        n += 1
    text = "\n\n".join(parts)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8",
                                     dir=os.path.dirname(os.path.abspath(sys.argv[1]))) as f:
        f.write(text)
        tmp = f.name
    try:
        out = subprocess.run([dsfront, "tokenize", tmp, "--tokenizer", tok], check=True, capture_output=True,
                             text=True).stdout
    finally:
        os.unlink(tmp)
    return [int(x) for x in re.findall(r"\d+", out)], n, len(text)


def main():
    if len(sys.argv) < 5:
        sys.exit("vocab_freq.py OUT.json DSFRONT TOKENIZER NAME=text:LISTFILE|NAME=ids:FILE[,FILE...] ...")
    out, dsfront, tok = sys.argv[1:4]
    res = {"vocab": V, "sources": {}}
    for spec in sys.argv[4:]:
        name, rest = spec.split("=", 1)
        kind, arg = rest.split(":", 1)
        if kind == "text":
            paths = [l.strip() for l in open(arg) if l.strip()]
            ids, nf, nc = tokenize(dsfront, tok, paths)
        else:
            ids, nf, nc = [], 0, 0
            for p in arg.split(","):
                if p.endswith(".i32"):  # tokenize_corpus.py's streams: little-endian int32
                    b = open(p, "rb").read()
                    ids += list(struct.unpack("<%di" % (len(b) // 4), b))
                else:
                    d = json.load(open(p))
                    ids += d["tokens"] if isinstance(d, dict) else d
                nf += 1
        counts = [0] * V
        for t in ids:
            counts[t] += 1
        res["sources"][name] = {"files": nf, "chars": nc, "tokens": len(ids), "counts": counts}
        print("%-10s %4d files, %9d chars, %8d tokens, %6d distinct" % (
            name, nf, nc, len(ids), sum(1 for c in counts if c)))
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
