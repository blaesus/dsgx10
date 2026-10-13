"""Write the Engram hash map and hashing test cases from the publisher's reference.

The C++ hasher needs two model constants it cannot cheaply derive itself: the token map
(token id -> compressed id, built with Unicode normalisers in engram.py) and the hash multipliers
(drawn from numpy's PCG64). Both go into one small file, `engram_hashmap.bin`, together with the
primes and offsets (which the C++ recomputes and checks).

Run in the canary-roce image with the checkpoint at /models and the repo at /work:
    python3 make_hashmap.py /models /out     -> /out/engram_hashmap.bin, /out/hash_cases.bin
"""
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/ref/inference")
import model as M  # noqa: E402
from engram import EngramLayout, NgramHashState  # noqa: E402

MAGIC = b"DSGBENG1"


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    torch.set_default_device("cpu")
    cfg = json.load(open("/work/ref/inference/config.json"))
    args = M.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = 1 << 16
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    layout = EngramLayout.from_args(args)
    hs = NgramHashState(args, layout, tok)
    token_map = hs.token_map.numpy().astype(np.int32)
    n_tokens = len(token_map)
    vocab = int(token_map.max()) + 1
    assert vocab == args.engram_compressed_vocab_size, vocab
    nl, ng, nh = len(layout.layer_ids), layout.max_ngram_size - 1, layout.n_heads
    primes = hs.primes.numpy().astype(np.int64).reshape(nl, ng, nh)
    offsets = hs.offsets.numpy().astype(np.int64).reshape(nl, ng * nh)
    mults = hs.multipliers.numpy().astype(np.int64).reshape(nl, layout.max_ngram_size)
    print("layers", layout.layer_ids, "n_tokens", n_tokens, "vocab", vocab, "pad", hs.pad_id)
    print("multipliers", mults.tolist())
    print("primes[0][0]", primes[0][0].tolist(), "offsets[1][-1]", int(offsets[1][-1]))
    for li in range(nl):
        assert int(offsets[li][-1] + primes[li].reshape(-1)[-1]) == layout.num_embeddings[li]

    # header: magic, version, n_tokens, vocab, pad, n_layers, max_ngram, n_heads,
    # layer_ids[nl], num_embeddings[nl], multipliers[nl][4], primes[nl][3][8], offsets[nl][24]
    hdr = MAGIC + struct.pack("<7q", 1, n_tokens, vocab, int(hs.pad_id), nl, layout.max_ngram_size, nh)
    hdr += struct.pack(f"<{nl}q", *layout.layer_ids) + struct.pack(f"<{nl}q", *layout.num_embeddings)
    hdr += mults.tobytes() + primes.tobytes() + offsets.tobytes()
    with open(os.path.join(out, "engram_hashmap.bin"), "wb") as f:
        f.write(hdr)
        f.write(token_map.tobytes())

    # test cases: ['ENGC', L, has_mask][tokens i32 L][mask u8 L][hashes i64 L*nl*24]
    rng = np.random.default_rng(1234)
    cases = []
    for L in (1, 2, 3, 4, 5, 9, 17, 300, 4000):
        cases.append((rng.integers(0, n_tokens, L).astype(np.int64), None))
    # a low-entropy stream: a few tokens repeated, special tokens 0..2 included
    cases.append((rng.choice(np.array([0, 1, 2, 3, 100, 5000, 128000]), 500).astype(np.int64), None))
    # real text
    for path in ("/work/ref/inference/engram.py", "/work/DESIGN.md"):
        ids = tok.encode(open(path).read(), add_special_tokens=False)
        cases.append((np.array(ids[:6000], dtype=np.int64), None))
    # a masked (DEAD) span in the middle, as image spans are
    ids = rng.integers(0, n_tokens, 64).astype(np.int64)
    mask = np.ones(64, dtype=bool)
    mask[20:27] = False
    cases.append((ids, mask))

    with open(os.path.join(out, "hash_cases.bin"), "wb") as f:
        for ids, mask in cases:
            L = len(ids)
            t = torch.tensor(ids)[None]
            m = None if mask is None else torch.tensor(mask)[None]
            h = hs(t, 0, m)[0].numpy().astype(np.int64)  # [L, nl, 24]
            # the reference must agree with itself when fed in chunks (the prefill/decode split)
            if L >= 9:
                parts, pos = [], 0
                for n in (1, 2, 5, 1, 1, L):
                    n = min(n, L - pos)
                    if n <= 0:
                        break
                    mm = None if m is None else m[:, pos:pos + n]
                    parts.append(hs(t[:, pos:pos + n], pos, mm)[0].numpy())
                    pos += n
                assert np.array_equal(np.concatenate(parts), h), "reference not chunk-consistent"
            f.write(b"ENGC" + struct.pack("<qq", L, 0 if mask is None else 1))
            f.write(ids.astype(np.int32).tobytes())
            f.write((np.ones(L, np.uint8) if mask is None else mask.astype(np.uint8)).tobytes())
            f.write(h.tobytes())
    print("cases", len(cases), "written to", out)
    for name in ("engram_hashmap.bin", "hash_cases.bin"):  # the image runs as root
        os.chown(os.path.join(out, name), 1000, 1000)


if __name__ == "__main__":
    main()
