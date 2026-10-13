#!/usr/bin/env python3
# Prompt-lookup drafts, a feasibility study (docs/engine.md): the inputs of the ring passes and of pld_study.py, in
# one directory DIR. No numpy.
#   pld_inputs.py turns DIR TRANSCRIPT.tokens.json      pass 1's lists: each assistant turn of the golden transcript
#                                                       (its context through "<｜Assistant｜><think>") as a prompt
#   pld_inputs.py pass1 DIR LOG...                      pass 1's logs -> DIR/turn_KK (golden dirs of each context and
#                                                       the target's own turn) and meta.json's "pass1"
#   pld_inputs.py runs DIR [--dsfront BIN] [--bench NAME=GOLDEN:PROMPT ...] [--turns RUN:K-K ...]
#                      [--edit INDEX.jsonl ...]         meta.json's "runs", spectf's sequence lists (sl_RUN.txt) and
#                                                       the PLD proposals whose rows are routed (props_RUN.txt); with
#                                                       --dsfront, esc.json first (as escaped), so that the proposals
#                                                       are the escaped matcher's
#   pld_inputs.py escaped DIR [--dsfront BIN]           esc.json: each tool result of every sequence's prompt,
#                                                       JSON-escaped and tokenized (dsfront tokenize), for the
#                                                       escaped matcher
import json, os, re, struct, subprocess, sys
import pld_study as S

USAGE = __doc__ if __doc__ else "pld_inputs.py turns|pass1|runs|escaped DIR ..."
TOKJ = "/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json"
ASSIST, THINK = 128804, 128821  # <｜Assistant｜>, <think>
GEN = 2048                      # pass 1: at most this many tokens a turn (EOS ends it)


def write_npy_i4(path, vals):
    hdr = "{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }" % len(vals)
    hdr += " " * (64 - (10 + len(hdr) + 1) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode())
        f.write(struct.pack("<%di" % len(vals), *vals))


def golden_dir(path, toks):  # what main.cpp's --golden reads
    os.makedirs(path, exist_ok=True)
    write_npy_i4(path + "/tokens.npy", toks)
    write_npy_i4(path + "/positions.npy", [])
    write_npy_i4(path + "/record_positions.npy", [])


def meta_of(D):
    p = os.path.join(D, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def decoder():
    """id -> bytes from the byte-level BPE's vocabulary (added tokens by their text)."""
    t = json.load(open(TOKJ))
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = dict(zip(map(chr, cs), bs))
    d = {i: b"".join(bytes([u2b[c]]) if c in u2b else c.encode() for c in s) for s, i in t["model"]["vocab"].items()}
    for x in t["added_tokens"]:
        d[x["id"]] = x["content"].encode()
    return d


def cmd_turns(D, transcript):
    toks = json.load(open(transcript))["tokens"]
    golden_dir(os.path.join(D, "long"), toks)
    turns = [s + 2 for s, x in enumerate(toks) if x == ASSIST and toks[s + 1] == THINK]
    groups = {"a": turns[0:6], "b": turns[6:11], "c": turns[11:]}  # three ring steps of similar prefill
    for k, ps in groups.items():
        with open(os.path.join(D, "gen_%s.txt" % k), "w") as f:
            for p in ps:
                f.write("%s %d %d\n" % (os.path.join(D, "long"), p, GEN))
    print(len(turns), "turns; prompts:", turns)


def cmd_pass1(D, logs):
    hdr = re.compile(r"^seq (\d+) \((.*)\): prompt (\d+) in ([\d.]+) s; (\d+) tokens in ([\d.]+) s, ([\d.]+) tok/s after "
                     r"the first (\d+), end (\d+); tokens hash (\w+)")
    long_toks = S.npy_i4(os.path.join(D, "long", "tokens.npy"))
    meta = meta_of(D)
    meta["pass1"] = {}
    k = 0
    for log in logs:
        cur = None
        for line in open(log, errors="replace"):
            m = hdr.match(line)
            if m:
                cur = dict(prompt=int(m.group(3)), prefill_s=float(m.group(4)), n=int(m.group(5)), s=float(m.group(6)),
                           tps=float(m.group(7)), end=int(m.group(9)), hash=m.group(10), report=[])
                continue
            if cur is not None and line.startswith("  "):
                cur["report"].append(line.rstrip())
            m2 = re.match(r"^seq (\d+) generated: (.*)$", line)
            if m2 and cur is not None:
                gen = [int(x) for x in m2.group(2).split()]
                if long_toks[cur["prompt"] - 2:cur["prompt"]] != [ASSIST, THINK]:
                    print("skipping a sequence with prompt %d: not a turn of the transcript" % cur["prompt"])
                    cur = None
                    continue
                name = "turn-%02d" % k
                golden_dir(os.path.join(D, "turn_%02d" % k), long_toks[:cur["prompt"]] + gen)
                meta["pass1"][name] = cur
                print("%s: prompt %5d, %4d tokens, %.1f tok/s" % (name, cur["prompt"], len(gen), cur["tps"]))
                k += 1
                cur = None
    json.dump(meta, open(os.path.join(D, "meta.json"), "w"), indent=1)


def props_lines(toks, P, sq, virt):
    """PLD proposals (up to 14 tokens) with a rejected token: every third such anchor."""
    mt = S.Matcher(toks, virt)
    out, elig = [], 0
    for a in range(P, len(toks) - 1):
        n, prop = mt.at(a).lookup(toks[max(0, a - S.NMAX + 1):a + 1], a, 2, 14)
        if prop and S.run_len(prop, toks, a, len(toks)) < len(prop):
            if elig % 3 == 0:
                out.append("%d %d %s\n" % (sq, a, " ".join(map(str, prop))))
            elig += 1
    return out


def cmd_runs(D, args):
    meta = meta_of(D)
    runs = meta.setdefault("runs", {})
    dsfront = None
    i = 0
    while i < len(args):
        if args[i] == "--dsfront":
            i += 1
            dsfront = args[i]
        elif args[i] == "--bench":  # NAME=GOLDEN:PROMPT, one run of them all
            seqs = []
            while i + 1 < len(args) and not args[i + 1].startswith("--"):
                i += 1
                name, rest = args[i].split("=")
                g, P = rest.rsplit(":", 1)
                seqs.append(dict(name=name, golden=g, prompt=int(P)))
            runs["bench"] = seqs
        elif args[i] == "--turns":  # RUN:K-K, the turns of pass 1
            i += 1
            run, rng = args[i].split(":")
            lo, hi = map(int, rng.split("-"))
            runs[run] = [dict(name="turn-%02d" % k, golden="turn_%02d" % k, prompt=meta["pass1"]["turn-%02d" % k]["prompt"])
                         for k in range(lo, hi + 1)]
        elif args[i] == "--edit":  # editbench.py's index: five runs of about equal output
            i += 1
            base = os.path.dirname(os.path.abspath(args[i]))
            rows = [json.loads(l) for l in open(args[i])]
            total, acc, k = sum(r["n_out"] for r in rows), 0, 1
            for r in rows:
                runs.setdefault("e%d" % k, []).append(dict(name="edit-%s-%s-t%02d" % (r["thinking"], r["task"], r["turn"]),
                                                            golden=os.path.join(base, r["dir"]), prompt=r["n_prompt"],
                                                            kind=r["kind"]))
                acc += r["n_out"]
                if acc >= total * k / 5 and k < 5:
                    k += 1
        i += 1
    if dsfront:
        json.dump(meta, open(os.path.join(D, "meta.json"), "w"), indent=1)
        cmd_escaped(D, dsfront)
    esc = json.load(open(os.path.join(D, "esc.json"))) if os.path.exists(os.path.join(D, "esc.json")) else {}
    for run, seqs in runs.items():
        with open(os.path.join(D, "sl_%s.txt" % run), "w") as fl, open(os.path.join(D, "props_%s.txt" % run), "w") as fp:
            for sq, m in enumerate(seqs):
                m["seq"] = sq
                g = m["golden"] if os.path.isabs(m["golden"]) else os.path.join(D, m["golden"])
                fl.write("%s %d 0\n" % (g, m["prompt"]))
                fp.writelines(props_lines(S.npy_i4(g + "/tokens.npy"), m["prompt"], sq, esc.get(m["golden"], [])))
        print("%s: %d sequences" % (run, len(seqs)))
    json.dump(meta, open(os.path.join(D, "meta.json"), "w"), indent=1)


def cmd_escaped(D, dsfront):
    meta = meta_of(D)
    dec = decoder()
    tmp = os.path.join(D, ".esc_tmp.txt")
    cache, out = {}, {}

    def tokenize(text):
        if text not in cache:
            open(tmp, "w", encoding="utf-8").write(text)
            o = subprocess.run([dsfront, "tokenize", tmp], capture_output=True, text=True, check=True).stdout
            cache[text] = json.loads(o[o.index("["):o.rindex("]") + 1])
        return cache[text]

    for seqs in meta.get("runs", {}).values():
        for m in seqs:
            g = m["golden"] if os.path.isabs(m["golden"]) else os.path.join(D, m["golden"])
            toks = S.npy_i4(g + "/tokens.npy")
            offs, b = [], bytearray()
            for t in toks[:m["prompt"]]:
                offs.append(len(b))
                b += dec.get(t, b"")
            offs.append(len(b))
            cps, o = [], 0
            while True:
                s = b.find(b"<tool_result>", o)
                e = b.find(b"</tool_result>", s) if s >= 0 else -1
                if e < 0:
                    break
                raw = bytes(b[s + 13:e]).decode("utf-8", "replace")
                esc = json.dumps(raw, ensure_ascii=False)[1:-1]
                end = e + 14
                if esc != raw:  # the copy goes after the token that ends the closing tag
                    cps.append([next(i for i in range(len(offs) - 1) if offs[i + 1] >= end), tokenize(esc)])
                o = end
            out[m["golden"]] = cps
    if os.path.exists(tmp):
        os.remove(tmp)
    json.dump(out, open(os.path.join(D, "esc.json"), "w"))
    print("%d sequences, %d escaped copies" % (len(out), sum(len(v) for v in out.values())))


def main():
    if len(sys.argv) < 3:
        sys.exit(USAGE)
    cmd, D = sys.argv[1], sys.argv[2]
    os.makedirs(D, exist_ok=True)
    if cmd == "turns":
        cmd_turns(D, sys.argv[3])
    elif cmd == "pass1":
        cmd_pass1(D, sys.argv[3:])
    elif cmd == "runs":
        cmd_runs(D, sys.argv[3:])
    elif cmd == "escaped":
        cmd_escaped(D, sys.argv[sys.argv.index("--dsfront") + 1] if "--dsfront" in sys.argv else "dsfront")
    else:
        sys.exit(USAGE)


if __name__ == "__main__":
    main()
