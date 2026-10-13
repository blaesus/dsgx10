#!/usr/bin/env python3
"""The CED check's inputs ("cedcheck"): golden-format directories (engine/server/golden.h) of the long
transcript for the engine's CED runs. Standard library only.

  python3 golden/make_cedcheck.py --out DIR [--src SRC] [--repo REPO] [--tokenizer TOKENIZER] [--check DIR]
  python3 golden/make_cedcheck.py --out DIR --tokens-only [--repo REPO] [--tokenizer TOKENIZER] [--check DIR]

From SRC, three directories:
  long8k-ced128   the transcript's first 8,936 tokens with the top-32 logits of SRC/long8k's stream ced128
                  (CED over a prompt of 8,680 tokens, then 256 teacher-forced), for
                  `--mode tf --big --golden DIR/long8k-ced128 --prompt 8680 --limit 8936 --prefill-rows 4096 --ced`
  long8k-exact    the same tokens with SRC/long8k's exact stream (the exact forward), for the same run
                  without --ced (the engine's exact mode against exact model.py)
  long28k-ced128  the whole transcript, 28,489 tokens, with SRC/long28k's stream ced128 (P = 28,489): the
                  CED reference hash's run, `--mode tf --big --golden DIR/long28k-ced128 --prompt 28489
                  --limit 28489 --prefill-rows 8192 --ced`, which teacher-forces nothing and reads only the tokens
Each holds tokens.npy (int32), empty positions.npy and record_positions.npy (no saved activations), and
logits_topv.npy / logits_topi.npy, [L, 32]: the stream's top-32 logits and ids over [s, L) (s = P - 128 for
a CED stream), zeros before s, where the engine compares nothing.

SRC (default /data/dsgb10/golden/ced) is ref/torchref/ced_run.py's output, made on the checkpoint under test
in the reference image through golden/run_ref.sh (CKPT=<checkpoint>, OUT=<dir>: SRC is then <dir>/golden/ced).
The runs rank 0's /data/dsgb10/kernels/cedcheck was made from (2026-10-11, dealignai's variant; 1,054 s and
1,496 s on a GB10; SRC/<run>/manifest.json records each one):
  OUT=<dir> golden/run_ref.sh py ced_run.py --ckpt /models --out-dir /out/golden/ced/long8k \\
      --tokens /work/golden/long.tokens.json --length 8936 --stream ced128:8680:128 --stream ced256:8680:256 \\
      --stream ced512:8680:512 --stream ced128at4728:4728:128 --iso ced128
  OUT=<dir> golden/run_ref.sh py ced_run.py --ckpt /models --out-dir /out/golden/ced/long28k \\
      --tokens /work/golden/long.tokens.json --stream ced128:28489:128 --stream ced128at12453:12453:128 --iso ced128
Only their ced128 and exact streams are read here.

--tokens-only needs no SRC: one directory, long-tokens, with the whole transcript's tokens from
REPO/golden/long.tokens.json and zero logits. That is all the reference hashes read (golden/refhash.sh): the
decode runs and the CED run read only the tokens, and the teacher-forced run opens the logits files to compare
with, whatever they hold (its comparison lines then mean nothing). So it serves as both G and C of
golden/refhash.sh, made from this repository alone, for any checkpoint with this tokenizer.

Checks: every directory's tokens equal the first L of REPO/golden/long.tokens.json, and every logits id is below
the vocabulary's 129,280; with --tokenizer (the checkpoint's tokenizer.json) the ids spell the transcript's text
(long.tokens.json's "text") under that tokenizer's byte-level vocabulary, and its sha256 is printed
(deepseek-ai's DeepSeek-V4.1-Flash and dealignai's variant ship the same file, c90dfa01...); with --check DIR,
every file written equals DIR's, byte for byte. Exit 1 if a check fails.
"""
import argparse
import ast
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = 129280  # model::kVocab
DIRS = (  # output directory, ced_run.py's run, its stream, tokens
    ("long8k-ced128", "long8k", "ced128", 8936),
    ("long8k-exact", "long8k", "exact", 8936),
    ("long28k-ced128", "long28k", "ced128", 28489),
)
TOKENS_ONLY = "long-tokens"


def parse_npy(b, what):
    """(header dict, data bytes) of a C-order .npy file's bytes"""
    if b[:6] != b"\x93NUMPY":
        sys.exit(f"{what}: not an npy file")
    hl, off = (struct.unpack("<H", b[8:10])[0], 10) if b[6] == 1 else (struct.unpack("<I", b[8:12])[0], 12)
    h = ast.literal_eval(b[off:off + hl].decode("latin1"))
    if h["fortran_order"]:
        sys.exit(f"{what}: Fortran order")
    return h, b[off + hl:]


def read_npy(path):
    return parse_npy(open(path, "rb").read(), path)


def npy_bytes(descr, shape, data):
    """an .npy file as numpy writes it (format 1.0, the header padded to a multiple of 64 bytes)"""
    d = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (descr, repr(tuple(shape)))
    hdr = d + " " * (64 - (10 + len(d) + 1) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode() + data


def ints(h, data):
    fmt = {"<i4": "i", "<i8": "q"}.get(h["descr"])
    if fmt is None:
        sys.exit(f"ids of type {h['descr']}: expected <i4 or <i8")
    return struct.unpack("<%d%s" % (len(data) // struct.calcsize(fmt), fmt), data)


def bytes_to_unicode():
    """GPT-2's byte-level alphabet: byte -> printable character (as ref/torchref/make_tokens.py)"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def spelling(tokenizer_json):
    """id -> the bytes it stands for, from tokenizer.json (byte-level BPE; added tokens literally)"""
    tj = json.load(open(tokenizer_json, encoding="utf-8"))
    u2b = {u: b for b, u in bytes_to_unicode().items()}
    sp = {}
    for s, i in tj["model"]["vocab"].items():
        sp[i] = bytes(u2b[c] for c in s) if all(c in u2b for c in s) else s.encode()
    for a in tj["added_tokens"]:
        sp[a["id"]] = a["content"].encode()
    return sp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="where the directories go (created)")
    ap.add_argument("--src", default="/data/dsgb10/golden/ced", help="ced_run.py's output (its long8k and long28k)")
    ap.add_argument("--tokens-only", action="store_true", help="no SRC: long-tokens, the tokens and zero logits")
    ap.add_argument("--repo", default=os.path.dirname(HERE), help="the configuration directory (golden/long.tokens.json)")
    ap.add_argument("--tokenizer", help="the checkpoint's tokenizer.json: the ids must spell the transcript")
    ap.add_argument("--check", help="a directory made before (e.g. /data/dsgb10/kernels/cedcheck): the same bytes")
    a = ap.parse_args()

    lt = json.load(open(os.path.join(a.repo, "golden", "long.tokens.json"), encoding="utf-8"))
    repo_tokens, text = lt["tokens"], lt["text"]
    bad = 0
    if a.tokenizer:
        sha = hashlib.sha256(open(a.tokenizer, "rb").read()).hexdigest()
        sp = spelling(a.tokenizer)
        spelt = b"".join(sp.get(t, b"\xff") for t in repo_tokens)
        ok = spelt == text.encode("utf-8")
        bad += not ok
        print(f"tokenizer {a.tokenizer} (sha256 {sha[:16]}): golden/long.tokens.json's {len(repo_tokens)} ids "
              f"{'spell' if ok else 'DO NOT spell'} its text ({len(text)} characters)")

    jobs = [(TOKENS_ONLY, None, None, len(repo_tokens))] if a.tokens_only else DIRS
    written = 0
    for name, run, stream, L in jobs:
        if run is not None and not os.path.isdir(os.path.join(a.src, run)):
            print(f"{name}: no {os.path.join(a.src, run)}, skipped")
            continue
        written += 1
        out = os.path.join(a.out, name)
        os.makedirs(out, exist_ok=True)
        files = {}
        if run is None:
            files["tokens.npy"] = npy_bytes("<i4", (L,), struct.pack("<%di" % L, *repo_tokens[:L]))
        else:
            files["tokens.npy"] = open(os.path.join(a.src, run, "tokens.npy"), "rb").read()
        toks = ints(*parse_npy(files["tokens.npy"], name + "/tokens.npy"))
        if len(toks) != L or list(toks) != repo_tokens[:L]:
            print(f"{name}: its {len(toks)} tokens are NOT golden/long.tokens.json's first {L}")
            bad += 1
        for f in ("positions.npy", "record_positions.npy"):
            files[f] = npy_bytes("<i4", (0,), b"")
        for k, descr, es in (("logits_topv", "<f4", 4), ("logits_topi", "<i8", 8)):
            if run is None:
                files[k + ".npy"] = npy_bytes(descr, (L, 32), bytes(L * 32 * es))
                continue
            h, b = read_npy(os.path.join(a.src, run, "%s.%s.npy" % (stream, k)))
            n, w = h["shape"]
            if w != 32 or len(b) != n * w * es or h["descr"] != descr or n > L:
                sys.exit(f"{run}/{stream}.{k}: {h}, {len(b)} bytes")
            lo = L - n  # the stream covers [lo, L)
            if k == "logits_topi":
                ids = ints(h, b)
                if min(ids) < 0 or max(ids) >= VOCAB:
                    print(f"{name}: ids outside [0, {VOCAB}) in {run}/{stream}.{k}")
                    bad += 1
            files[k + ".npy"] = npy_bytes(descr, (L, 32), bytes(lo * 32 * es) + b)
        for f, data in files.items():
            with open(os.path.join(out, f), "wb") as fh:
                fh.write(data)
        what = "tokens only, zero logits" if run is None else f"{run}'s stream {stream} over [{lo}, {L})"
        line = f"{name}: {L} positions, {what}"
        if a.check:
            same = []
            for f, data in files.items():
                p = os.path.join(a.check, name, f)
                same.append(os.path.exists(p) and open(p, "rb").read() == data)
            line += f"; equal to {os.path.join(a.check, name)}: {sum(same)} of {len(same)} files"
            bad += not all(same)
        print(line)
    bad += not written
    print(f"wrote {written} director{'y' if written == 1 else 'ies'} in {a.out}" + (f"; {bad} check(s) FAILED" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
