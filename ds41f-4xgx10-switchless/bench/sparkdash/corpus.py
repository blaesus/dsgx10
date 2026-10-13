#!/usr/bin/env python3
"""Real text for the cold-prefill runs: the repository's documentation and source code at a pinned commit
(git archive), text files only, in a seeded shuffle so that every slice mixes prose and code. Each file is
headed by its path. Writes corpus.txt (UTF-8) and prints its size.
  corpus.py SRC.tar OUT.txt"""
import random
import sys
import tarfile

EXT = ('.md', '.py', '.rs', '.cu', '.cuh', '.cpp', '.h', '.sh', '.toml', '.txt', '.cmake')
src, out = sys.argv[1], sys.argv[2]
files = []
with tarfile.open(src) as t:
    for m in t.getmembers():
        if not m.isfile() or not m.name.endswith(EXT) or m.size > 300_000 or m.size < 200:
            continue
        b = t.extractfile(m).read()
        try:
            s = b.decode('utf-8')
        except UnicodeDecodeError:
            continue
        files.append((m.name, s))
files.sort()
random.Random(4110).shuffle(files)
with open(out, 'w', encoding='utf-8') as f:
    for name, s in files:
        f.write('=== %s ===\n%s\n' % (name, s))
n = sum(len(s) for _, s in files)
print('%d files, %d chars, %d .md' % (len(files), n, sum(1 for x, _ in files if x.endswith('.md'))))
