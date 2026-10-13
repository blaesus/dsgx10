#!/usr/bin/env python3
# The draft pass's vocabulary subset by script (docs/engine.md, "The draft pass's approximations: a study"): the
# tokens whose text is Latin script, digits, punctuation, symbols or emoji, and every added token; byte pieces of a
# character (an incomplete UTF-8 sequence) count by the character's block when their lead byte is there, and are
# kept otherwise. Writes the ids as JSON for vocab_study.py --subset.
#   vocab_script.py TOKENIZER.json OUT.json
import json, sys
import unicodedata

V = 129280
OK_RANGES = [
    (0x0000, 0x024F),    # ASCII, Latin-1, Latin Extended-A/B
    (0x0250, 0x02FF),    # IPA, spacing modifiers
    (0x0300, 0x036F),    # combining diacritics
    (0x1E00, 0x1EFF),    # Latin Extended Additional
    (0x2000, 0x2BFF),    # punctuation, super/subscripts, currency, letterlike, arrows, math, technical, boxes,
                         # shapes, symbols, dingbats, braille, misc arrows
    (0x2C60, 0x2C7F), (0xA720, 0xA7FF),  # Latin Extended-C, D
    (0x3000, 0x3003), (0xFE00, 0xFE0F),  # ideographic space and punctuation; variation selectors
    (0xFF01, 0xFF5E),    # fullwidth ASCII
    (0x1F000, 0x1FAFF),  # emoji and pictographs
]


def ok(cp):
    return any(a <= cp <= b for a, b in OK_RANGES)


def byte_decoder():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def lead_cp(bs):
    """The code point a UTF-8 lead byte starts, as far as the bytes go (the rest zero), or None."""
    b = bs[0]
    if b < 0x80:
        return b
    n, mask = (2, 0x1F) if 0xC0 <= b < 0xE0 else (3, 0x0F) if 0xE0 <= b < 0xF0 else (4, 0x07) if 0xF0 <= b < 0xF8 \
        else (0, 0)
    if not n:
        return None
    cp = b & mask
    for k in range(1, n):
        cp = (cp << 6) | ((bs[k] & 0x3F) if k < len(bs) else 0)
    return cp


def main():
    tj = json.load(open(sys.argv[1]))
    dec = byte_decoder()
    keep = set(a["id"] for a in tj["added_tokens"])
    kinds = {"text": 0, "pieces": 0, "out": 0}
    for s, i in tj["model"]["vocab"].items():
        if any(ch not in dec for ch in s):  # the specials kept in the BPE vocabulary (ids 0-2)
            keep.add(i)
            continue
        bs = bytes(dec[ch] for ch in s)
        try:
            txt = bs.decode("utf-8")
            good = all(ok(ord(c)) for c in txt)
            kinds["text" if good else "out"] += 1
        except UnicodeDecodeError:
            # a character's pieces: judge by the first complete or leading character
            j = 0
            while j < len(bs) and 0x80 <= bs[j] < 0xC0:
                j += 1  # continuation bytes of a character begun in an earlier token: judged there
            cp = lead_cp(bs[j:]) if j < len(bs) else None
            good = cp is None or ok(cp)
            kinds["pieces"] += good
            kinds["out"] += not good
        if good:
            keep.add(i)
    ids = sorted(keep)
    json.dump({"ids": ids, "about": "Latin script, digits, punctuation, symbols, emoji; all added tokens"},
              open(sys.argv[2], "w"))
    print("kept %d of %d (%s); unicodedata %s" % (len(ids), V, kinds, unicodedata.unidata_version))


if __name__ == "__main__":
    main()
