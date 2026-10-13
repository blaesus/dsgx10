#!/usr/bin/env python3
"""editbench's read/edit/write against pi's own tool code (edit_pi_check.mjs, under node) on the same cases:
the result texts, error or not, and the files after. Run it when pi changes (edit_bench.json's pi_version).

  bench/edit_pi_check.py        # PI_DIR: pi's package (default: npm's global one)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import editbench  # noqa: E402

PI_DIR = os.environ.get("PI_DIR") or os.path.join(
    subprocess.run(["npm", "root", "-g"], capture_output=True, check=True, text=True).stdout.strip(),
    "@earendil-works", "pi-coding-agent")
W = tempfile.mkdtemp(prefix="edit_pi_check.")
spec = json.load(open(os.path.join(S, "edit_bench.json")))
schemas = {x["function"]["name"]: x["function"]["parameters"] for x in spec["tools"]}

PY = '"""Doc."""\n\n\ndef f(x):\n    return x + 1  \n\n\ndef g(y):\n    return y * 2\n'
big = "".join(f"line {i}\n" for i in range(2500))
wide = "".join(("x" * 99) + "\n" for i in range(800))
C = [
    ("read", {"path": "a.py"}, {"a.py": PY}),
    ("read", {"path": "a.py", "offset": 4}, {"a.py": PY}),
    ("read", {"path": "a.py", "offset": 2, "limit": 3}, {"a.py": PY}),
    ("read", {"path": "a.py", "offset": 99}, {"a.py": PY}),
    ("read", {"path": "a.py", "limit": 100}, {"a.py": PY}),
    ("read", {"path": "missing.py"}, {"a.py": PY}),
    ("read", {"path": "big.txt"}, {"big.txt": big}),
    ("read", {"path": "big.txt", "offset": 2001}, {"big.txt": big}),
    ("read", {"path": "wide.txt"}, {"wide.txt": wide}),
    ("read", {"path": "a.py", "offset": "2", "limit": "2"}, {"a.py": PY}),
    ("read", {"path": "@a.py"}, {"a.py": PY}),
    ("read", {"path": "sub/../a.py"}, {"a.py": PY}),
    ("read", {}, {"a.py": PY}),
    ("read", {"path": "a.py", "offset": None}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "def f(x):", "newText": "def f(x, k=1):"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "def f(x):", "newText": "def f(z):"},
                                        {"oldText": "return y * 2", "newText": "return y * 3"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "nope", "newText": "x"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "def f", "newText": "x"}, {"oldText": "nope", "newText": "x"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "return", "newText": "yield"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "def f", "newText": "x"}, {"oldText": "return", "newText": "x"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "def f(x):\n    return", "newText": "A"},
                                        {"oldText": "return x + 1", "newText": "B"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "", "newText": "x"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "def g", "newText": "x"}, {"oldText": "", "newText": "x"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "def g(y):", "newText": "def g(y):"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "return x + 1\n", "newText": "return x + 2\n"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "    return x + 1\n\n\ndef g", "newText": "    return x - 1\n\n\ndef h"}]},
     {"a.py": PY}),
    ("edit", {"path": "q.py", "edits": [{"oldText": "print('it’s')", "newText": "print('ok')"}]},
     {"q.py": "a = 1   \nprint('it's')\nb = 2\n"}),
    ("edit", {"path": "q.py", "edits": [{"oldText": "print(\"it's\")", "newText": "print('ok')"}]},
     {"q.py": "a = 1   \nprint(“it’s”)\nb = 2  \n"}),
    ("edit", {"path": "c.txt", "edits": [{"oldText": "two\nthree", "newText": "2\n3"}]}, {"c.txt": "one\r\ntwo\r\nthree\r\n"}),
    ("edit", {"path": "b.txt", "edits": [{"oldText": "two", "newText": "2"}]}, {"b.txt": "﻿one\ntwo\n"}),
    ("edit", {"path": "a.py", "oldText": "def g(y):", "newText": "def g(w):"}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": json.dumps([{"oldText": "def g(y):", "newText": "def g(w):"}])}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": {"oldText": "def g(y):", "newText": "def g(w):"}}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": []}, {"a.py": PY}),
    ("edit", {"path": "nope.py", "edits": [{"oldText": "a", "newText": "b"}]}, {"a.py": PY}),
    ("edit", {"path": "a.py"}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "def f(x):"}]}, {"a.py": PY}),
    ("write", {"path": "n.py", "content": "print('héllo \U0001F600')\n"}, {}),
    ("write", {"path": "deep/dir/n.py", "content": "x = 1\n"}, {}),
    ("write", {"path": "a.py", "content": "x = 2\n"}, {"a.py": PY}),
    ("write", {"path": "a.py"}, {"a.py": PY}),
    ("edit", {"path": "a.py", "edits": [{"oldText": "    return x + 1", "newText": "    return x + 5"}]}, {"a.py": PY}),
    ("edit", {"path": "t.py", "edits": [{"oldText": "a—b", "newText": "c"}, {"oldText": "z = 1", "newText": "z = 2"}]},
     {"t.py": "s = 'a-b'  \nz = 1\nw = 3   \n"}),
]
cases = [dict(tool=t, args=a, files=f) for t, a, f in C]
json.dump(cases, open(os.path.join(W, "cases.json"), "w"))
pi = json.loads(subprocess.run(["node", os.path.join(S, "edit_pi_check.mjs"), os.path.join(W, "cases.json"), os.path.join(W, "pi")],
                               capture_output=True, check=True, text=True, env=dict(os.environ, PI_DIR=PI_DIR)).stdout)
bad = 0
for i, c in enumerate(cases):
    d = os.path.join(W, "py", f"c{i}")
    os.makedirs(d)
    for p, t in c["files"].items():
        os.makedirs(os.path.dirname(os.path.join(d, p)), exist_ok=True)
        with open(os.path.join(d, p), "w", encoding="utf-8", newline="") as f:
            f.write(t)
    pd = os.path.join(W, "pi", f"c{i}")
    ws = editbench.Workspace(pd + ".py", pd)  # the same cwd text as pi's, files under a sibling directory
    shutil.copytree(d, pd + ".py")
    text, err = editbench.call_tool(ws, schemas, c["tool"], json.loads(json.dumps(c["args"])))
    files = {}
    for root, _, fs in os.walk(pd + ".py"):
        for f in fs:
            p = os.path.join(root, f)
            files[os.path.relpath(p, pd + ".py")] = open(p, encoding="utf-8", newline="").read()
    ok = text == pi[i]["text"] and err == pi[i]["err"] and files == pi[i]["files"]
    if not ok:
        bad += 1
        print(f"case {i} {c['tool']} {json.dumps(c['args'])[:120]}")
        if text != pi[i]["text"] or err != pi[i]["err"]:
            print(f"   pi: {pi[i]['err']} {pi[i]['text'][:400]!r}\n   py: {err} {text[:400]!r}")
        if files != pi[i]["files"]:
            print(f"   pi files: {json.dumps(pi[i]['files'])[:300]}\n   py files: {json.dumps(files)[:300]}")
shutil.rmtree(W)
print(f"{len(cases)} cases, {bad} differ from pi's")
sys.exit(1 if bad else 0)
