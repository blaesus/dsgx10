#!/usr/bin/env python3
"""Reference token ids for the front-end tokeniser tests, from the Hugging Face tokenizer.

Runs inside the canary-roce image (tokenizers 0.22, transformers 5.12), e.g.

    docker run --rm -v /data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8:/models:ro \\
        -v $PWD/frontend:/work:ro -v ~/dsgb10-data/frontend:/out \\
        -v $PWD/ref/techreport.txt:/techreport.txt:ro \\
        --entrypoint /opt/sglang/bin/python3 dsv41-4x-spark:canary-roce /work/tools/gen_tokenizer_cases.py

Writes /out/tok_<set>.jsonl, one {"text", "ids"} per line:
  corpus  Python sources, the model README and tech report, multilingual prose, special tokens
  fuzz    random strings over a mixed alphabet (whitespace, digits of many scripts, CJK, marks, emoji, specials)
  sweep   every assigned code point (and every 97th unassigned one) in five probe contexts
"""
import glob
import json
import os
import random
import sys
import unicodedata

from tokenizers import Tokenizer
from transformers import AutoTokenizer

OUT = "/out"
tk = Tokenizer.from_file("/models/tokenizer.json")
hf = AutoTokenizer.from_pretrained("/models")

SPECIALS = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<｜System｜>",
            "<think>", "</think>", "｜DSML｜", "<｜DSML｜ calls>", "</｜DSML｜ invoke>", "<dsml:", "</dsml:",
            "<｜latest_reminder｜>", "<｜deepseek_image｜>", "<｜▁pad▁｜>", "<|place_holder_mm_span_0436|>",
            "<｜tool▁calls▁begin｜>", "<|EOT|>", "<｜fim▁hole｜>"]

MULTILINGUAL = [
    "The quick brown fox jumps over the lazy dog. It's 3.14159 o'clock — isn't it?",
    "人工智能正在改变世界。我们需要考虑它的影响，以及如何负责任地使用它。2026年10月11日。",
    "日本語のテキスト：ひらがな、カタカナ・漢字が混ざっています。ｶﾀｶﾅ（半角）もあります。",
    "한국어 문장입니다. 오늘 날씨가 정말 좋네요! 123개의 사과.",
    "Привет, мир! Это тест токенизатора — с цифрами 42 и знаками препинания…",
    "Ελληνικά: Γεια σου κόσμε! Αυτό είναι ένα τεστ.",
    "العربية: مرحبا بالعالم! هذا اختبار ١٢٣٤٥ للأرقام.",
    "हिन्दी: नमस्ते दुनिया! यह एक परीक्षण है। १२३४",
    "ภาษาไทย: สวัสดีชาวโลก นี่คือการทดสอบ",
    "Tiếng Việt: Xin chào thế giới! Đây là bài kiểm tra.",
    "Emoji: 👍🏽 👨‍👩‍👧‍👦 🇬🇧 ❤️ 🔥🔥🔥 ✅ ½ ⅞ ² ③ Ⅻ",
    "Math: ∀x∈ℝ, x² ≥ 0; ∑_{i=1}^{n} i = n(n+1)/2 ≈ 𝔼[X] ± σ",
    "Combining: e\u0301 a\u0308 n\u0303 \u0301\u0301x Z\u0335\u0321a\u0337l\u0336g\u0334o",
    "Mixed\u3000full-width\u3000spaces and\u00a0nbsp and\u2009thin\u2028line\u2029para\u0085nel",
    "Tabs\tand\r\nCRLF\r\n\r\n  indented\n\t\tmore\r\r\n",
    "URL https://example.com/path?q=1&x=%20#frag and email a.b@c.co.uk, path C:\\Users\\x\\file.txt",
    "JSON {\"key\": [1, 2.5e-3, true, null], \"nested\": {\"a\": \"b\\n\"}}",
    "HTML <div class=\"x\">&nbsp;<b>bold</b></div><!-- comment --> <script>var a = 1 < 2;</script>",
]


def corpus():
    texts = []
    files = sorted(glob.glob("/usr/lib/python3.12/*.py")) or sorted(glob.glob("/opt/sglang/lib/python3.12/site-packages/*/*.py"))
    random.Random(1).shuffle(files)
    total = 0
    for f in files:
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        texts.append(t)
        total += len(t)
        if total > 6_000_000:
            break
    for f in ["/models/README.md", "/models/encoding/README.md", "/techreport.txt"]:
        if os.path.exists(f):
            texts.append(open(f, encoding="utf-8").read())
    texts += MULTILINGUAL
    # special tokens embedded in text, adjacent, partial, nested
    for s in SPECIALS:
        texts += [s, "x" + s + "y", " " + s + " ", s + s, s[:-1], "<" + s, s + "\n\n", "a" + s[:4]]
    texts.append("".join(SPECIALS))
    # long whitespace runs
    for n in [1, 2, 3, 7, 16, 100, 1000]:
        for ws in [" ", "\t", "\n", "\r\n", " \n", "\n ", "\u3000", "\u00a0"]:
            texts += [ws * n, "a" + ws * n + "b", ws * n + "x", "x" + ws * n]
    # long words, digits
    texts += ["a" * 5000, "ab" * 3000, "1" * 1000, "x" * 100 + "1" * 7 + "y", "=" * 3000, "-_" * 2000,
              "0123456789" * 50, "ｱ" * 500, "中" * 2000]
    return texts


ALPHABET = (
    [" "] * 12 + ["\t", "\n", "\n", "\r", "\r\n", "\u3000", "\u00a0", "\u2028", "\u200b", "\u200d", "\ufeff"]
    + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") * 2
    + list("0123456789") * 2 + list("٠١٢٣٤٥६७८९１２３½²③Ⅻ")
    + list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") * 2 + list("«»“”‘’—–…·・、。，！？：；（）【】《》¥€£©®™°±×÷")
    + list("中文字符測試漢語日本ひらがなカタカナーヽヾ゛゜゠ヿ㐀𠀀가나다한국") + list("абвгдежзαβγδ")
    + list("éèñüçøåßıİ") + ["e\u0301", "\u0301", "\u0308", "\u20dd", "\u0903"]
    + list("👍🏽👨‍👩‍👧🇬🇧❤️🔥✅🤖") + ["\U0001d4b3", "\U0010fffd", "\ue000", "\x00", "\x1f", "\x7f", "\x85"]
)


def fuzz(n=60000, seed=7):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.choice([1, 2, 3, 5, 8, 13, 30, 80, 200])
        parts = []
        for _ in range(k):
            if rng.random() < 0.02:
                parts.append(rng.choice(SPECIALS))
            else:
                parts.append(rng.choice(ALPHABET))
        out.append("".join(parts))
    return out


def sweep():
    out = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if unicodedata.category(c) == "Cn" and cp % 97 != 0:
            continue  # unassigned: a sample is enough
        out += ["a" + c + "a", "a" + c + c + "!", c + c + c + c, " " + c + "\n" + c + "x", "1" + c + " " + c + "中"]
    return out


def write(name, texts):
    encs = tk.encode_batch(texts, add_special_tokens=False)
    path = os.path.join(OUT, f"tok_{name}.jsonl")
    bad_rt = 0
    with open(path, "w") as fh:
        for t, e in zip(texts, encs):
            if tk.decode(e.ids, skip_special_tokens=False) != t:
                bad_rt += 1
            fh.write(json.dumps({"text": t, "ids": e.ids}, ensure_ascii=False) + "\n")
    print(f"{name}: {len(texts)} cases -> {path} (reference decode != text in {bad_rt})", flush=True)


def check_transformers(texts):
    # The transformers wrapper must agree with the tokenizers backend it wraps.
    diff = 0
    for t in texts:
        if hf.encode(t, add_special_tokens=False) != tk.encode(t, add_special_tokens=False).ids:
            diff += 1
    print(f"transformers vs tokenizers on {len(texts)} texts: {diff} differ; "
          f"add_special_tokens=True adds bos? {hf.encode('hi', add_special_tokens=True)}", flush=True)


if __name__ == "__main__":
    sets = sys.argv[1:] or ["corpus", "fuzz", "sweep"]
    if "corpus" in sets:
        c = corpus()
        check_transformers(c[:400] + fuzz(2000, seed=3))
        write("corpus", c)
    if "fuzz" in sets:
        write("fuzz", fuzz())
    if "sweep" in sets:
        write("sweep", sweep())
