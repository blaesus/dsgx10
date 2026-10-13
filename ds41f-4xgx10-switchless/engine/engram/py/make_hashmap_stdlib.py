"""Write the Engram hash map, engram_hashmap.bin, from a checkpoint, with Python's standard library alone.

The file holds the model constants the C++ hasher (engine/engram/hash.cpp) needs: the token map (token
id -> compressed id) and the hash multipliers, with the primes and offsets it recomputes and checks.
make_hashmap.py writes the same file (and the hashing test cases) by importing the publisher's reference
in the SGLang image; this script needs neither: it re-implements, from the checkpoint's own
tokenizer.json and inference/config.json, what DeepSeek's inference/engram.py computes:

- build_compressed_token_map: each token decoded as the byte-level BPE decoder does, then normalised as
  its `tokenizers` normalizer sequence does (NFKC, NFD, strip marks, lowercase, collapse whitespace, a
  lone space kept through Strip by a sentinel), and tokens with the same key share an id;
- EngramLayout's primes: for each layer and n-gram size, the next primes above engram_vocab_size - 1
  not yet handed out, and the offsets, their running sums;
- compute_hash_multipliers: numpy.random.default_rng(10007 * layer).integers(0, bound, 4) * 2 + 1,
  with numpy's SeedSequence, PCG64 and bounded-integer draw re-implemented bit for bit.

    python3 make_hashmap_stdlib.py CHECKPOINT_DIR OUT_FILE      (under a second)

It prints the file's sha256 and fails unless it is the map the engine was validated with for
DeepSeek-V4.1-Flash (both the original and dealignai's variant: their tokenizer.json and config.json are
the same files), 518,048 bytes. Python's Unicode database differs a little between versions; Python 3.12
(Unicode 15.0) makes this file.

The algorithms are DeepSeek's (inference/engram.py, MIT, Copyright (c) 2023 DeepSeek) and numpy's
(SeedSequence, PCG64 and Generator.integers, BSD-3-Clause, Copyright (c) NumPy Developers;
PCG64 after M. E. O'Neill's PCG): see the repository's NOTICE.
"""
import hashlib
import json
import os
import re
import struct
import sys
import unicodedata

MAGIC = b"DSGBENG1"
SHA256 = "2405ae1d345ec2f76f637e2de71918060c2160f24bc7ed5a0e0ac4f9e123b02e"  # DeepSeek-V4.1-Flash's map

M32, M64, M128 = (1 << 32) - 1, (1 << 64) - 1, (1 << 128) - 1


# ---- the token map ------------------------------------------------------------------------------

def byte_alphabet():
    """The byte-level BPE alphabet: the character each byte is written as -> the byte."""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


# What Rust's char::is_whitespace strips (Unicode White_Space): the normalizer's Strip uses it, and
# Python's str.strip() would also strip U+001C-U+001F.
WHITE_SPACE = ("\t\n\x0b\x0c\r \x85\xa0 " + "".join(chr(c) for c in range(0x2000, 0x200B))
               + "    　")
SENTINEL = ""  # engram.py's: a token that is exactly one space survives Strip()


def normalize(text):
    s = unicodedata.normalize("NFKC", text)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.category(c).startswith("M"))  # StripAccents: marks
    s = s.lower()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    if s == " ":
        s = SENTINEL
    return s.strip(WHITE_SPACE).replace(SENTINEL, " ")


def token_map(tok):
    if (tok.get("decoder") or {}).get("type") != "ByteLevel":
        sys.exit("tokenizer.json: the decoder is not ByteLevel; this script knows only that one")
    alphabet = byte_alphabet()
    id_to_tok = {i: t for t, i in tok["model"]["vocab"].items()}
    for a in tok["added_tokens"]:
        id_to_tok[a["id"]] = a["content"]
    n = max(id_to_tok) + 1
    if sorted(id_to_tok) != list(range(n)):
        sys.exit("tokenizer.json: the token ids are not 0..n-1")
    keys, lookup = {}, []
    for i in range(n):
        t = id_to_tok[i]
        try:  # the ByteLevel decoder: the token's bytes, or its UTF-8 if it is not in the alphabet
            raw = bytes(alphabet[c] for c in t)
        except KeyError:
            raw = t.encode("utf-8")
        text = raw.decode("utf-8", errors="replace")
        if "�" in text:  # a partial UTF-8 byte token: keyed by its raw form
            key = t
        else:
            key = normalize(text) or text
        lookup.append(keys.setdefault(key, len(keys)))
    return lookup, len(keys)


# ---- numpy's default_rng(seed).integers(0, high, n, dtype=int64) ---------------------------------

def seed_sequence(seed, words64=4):
    """numpy.random.SeedSequence(seed).generate_state(words64, uint64)."""
    ent = []
    while True:
        ent.append(seed & M32)
        seed >>= 32
        if not seed:
            break
    hash_const = [0x43B0D7E5]

    def hashmix(v):
        v = (v ^ hash_const[0]) & M32
        hash_const[0] = (hash_const[0] * 0x931E8875) & M32
        v = (v * hash_const[0]) & M32
        return v ^ (v >> 16)

    def mix(x, y):
        r = (0xCA01F9DD * x - 0x4973F715 * y) & M32
        return r ^ (r >> 16)

    pool = [hashmix(ent[i] if i < len(ent) else 0) for i in range(4)]
    for s in range(4):
        for d in range(4):
            if s != d:
                pool[d] = mix(pool[d], hashmix(pool[s]))
    for s in range(4, len(ent)):
        for d in range(4):
            pool[d] = mix(pool[d], hashmix(ent[s]))
    hb, words = 0x8B51F9DD, []
    for i in range(2 * words64):
        v = pool[i % 4] ^ hb
        hb = (hb * 0x58F38DED) & M32
        v = (v * hb) & M32
        words.append(v ^ (v >> 16))
    return [words[2 * i] | (words[2 * i + 1] << 32) for i in range(words64)]


class PCG64:
    """numpy's PCG64 (XSL-RR 128/64) seeded as numpy seeds it."""
    MULT = (2549297995355413924 << 64) + 4865540595714422341

    def __init__(self, seed):
        v = seed_sequence(seed)
        initstate, initseq = (v[0] << 64) | v[1], (v[2] << 64) | v[3]
        self.inc = ((initseq << 1) | 1) & M128
        self.state = 0
        self.step()
        self.state = (self.state + initstate) & M128
        self.step()

    def step(self):
        self.state = (self.state * self.MULT + self.inc) & M128

    def next64(self):
        self.step()
        s = self.state
        x, rot = ((s >> 64) ^ s) & M64, s >> 122
        return ((x >> rot) | (x << ((-rot) & 63))) & M64


def integers(gen, high, n):
    """Generator.integers(0, high, n, dtype=int64) for high - 1 above 2^32: Lemire's 64-bit draw."""
    rng = high - 1
    excl = rng + 1
    out = []
    for _ in range(n):
        m = gen.next64() * excl
        if m & M64 < excl:
            threshold = (M64 - rng) % excl
            while m & M64 < threshold:
                m = gen.next64() * excl
        out.append(m >> 64)
    return out


def is_prime(n):
    if n < 2 or n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


# ---- the file ------------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python3 make_hashmap_stdlib.py CHECKPOINT_DIR OUT_FILE")
    ckpt, out = sys.argv[1], sys.argv[2]
    with open(os.path.join(ckpt, "inference", "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(ckpt, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    lookup, vocab = token_map(tok)
    if vocab != cfg["engram_compressed_vocab_size"]:
        sys.exit(f"the compressed vocab has {vocab} ids, the config says {cfg['engram_compressed_vocab_size']}")
    layer_ids, n_emb = cfg["engram_layer_ids"], cfg["engram_num_embeddings"]
    max_ng, nh = cfg["engram_max_ngram_size"], cfg["engram_n_heads"]
    nl, ng = len(layer_ids), max_ng - 1

    primes, seen = [], set()  # [layer][n-gram size * head], drawn in order, never reused
    for _ in layer_ids:
        per = []
        for _ in range(ng):
            cur = cfg["engram_vocab_size"] - 1
            for _ in range(nh):
                cur += 1
                while cur in seen or not is_prime(cur):
                    cur += 1
                seen.add(cur)
                per.append(cur)
        primes.append(per)
    offsets = []
    for li, per in enumerate(primes):
        o = [sum(per[:k]) for k in range(len(per))]
        if o[-1] + per[-1] != n_emb[li]:
            sys.exit(f"layer {layer_ids[li]}: the buckets add up to {o[-1] + per[-1]} rows, the config says {n_emb[li]}")
        offsets.append(o)
    bound = max(1, ((2**63 - 1) // vocab) // 2)
    mults = [[v * 2 + 1 for v in integers(PCG64(10007 * lid), bound, max_ng)] for lid in layer_ids]
    pad = lookup[cfg["engram_pad_id"]]
    print(f"{len(lookup)} tokens, {vocab} compressed ids, pad {pad}; layers {layer_ids}, rows {n_emb}")
    print("multipliers", mults)

    # header: magic, version, n_tokens, vocab, pad, n_layers, max_ngram, n_heads, layer_ids[nl],
    # num_embeddings[nl], multipliers[nl][max_ngram], primes[nl][ng * nh], offsets[nl][ng * nh] (int64,
    # little-endian), then the token map (int32): make_hashmap.py's layout, which hash.cpp reads
    data = MAGIC + struct.pack("<7q", 1, len(lookup), vocab, pad, nl, max_ng, nh)
    data += struct.pack(f"<{nl}q", *layer_ids) + struct.pack(f"<{nl}q", *n_emb)
    data += struct.pack(f"<{nl * max_ng}q", *[m for row in mults for m in row])
    data += struct.pack(f"<{nl * ng * nh}q", *[p for row in primes for p in row])
    data += struct.pack(f"<{nl * ng * nh}q", *[o for row in offsets for o in row])
    data += struct.pack(f"<{len(lookup)}i", *lookup)
    with open(out + ".part", "wb") as f:
        f.write(data)
    os.replace(out + ".part", out)
    digest = hashlib.sha256(data).hexdigest()
    print(f"{out}: {len(data)} bytes, sha256 {digest}")
    if digest != SHA256:
        sys.exit(f"this is not DeepSeek-V4.1-Flash's map (sha256 {SHA256}): another checkpoint, or another "
                 f"Python's Unicode database (this is {sys.version.split()[0]}, Unicode {unicodedata.unidata_version})")
    print("the map the engine was validated with")


if __name__ == "__main__":
    main()
