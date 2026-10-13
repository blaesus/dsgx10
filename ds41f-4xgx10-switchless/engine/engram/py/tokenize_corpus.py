"""Token streams for Engram cache measurements: documents tokenised with the model's tokenizer and
concatenated (EOS between documents) into little-endian int32 files.

Run in the canary-roce image:
    python3 tokenize_corpus.py /models /out
with the corpora mounted at /corpus/py (Python stdlib), /corpus/doc (/usr/share/doc),
/corpus/lic (/usr/share/common-licenses) and /corpus/techreport.txt.
"""
import glob
import os
import random
import sys

import numpy as np


def files(patterns):
    out = []
    for p in patterns:
        out += [f for f in glob.glob(p, recursive=True) if os.path.isfile(f)]
    return sorted(set(out))


def build(tok, name, paths, limit, out):
    random.Random(name).shuffle(paths)
    ids, docs = [], 0
    for path in paths:
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if len(text) < 200:
            continue
        ids += tok.encode(text, add_special_tokens=False) + [1]
        docs += 1
        if len(ids) >= limit:
            break
    arr = np.array(ids[:limit], dtype=np.int32)
    arr.tofile(os.path.join(out, name + ".i32"))
    os.chown(os.path.join(out, name + ".i32"), 1000, 1000)
    print(f"{name}: {docs} documents, {len(arr)} tokens")


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    build(tok, "code", files(["/corpus/py/**/*.py"]), 4_000_000, out)
    prose = files(["/corpus/techreport.txt", "/corpus/lic/*", "/corpus/doc/**/README*", "/corpus/doc/**/*.md",
                   "/corpus/doc/**/*.txt"])
    build(tok, "prose", prose, 2_000_000, out)


if __name__ == "__main__":
    main()
