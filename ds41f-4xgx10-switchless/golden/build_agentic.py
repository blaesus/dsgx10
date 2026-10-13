"""Compose the long golden sequence: a coding-agent transcript with tool calls whose results are
real files (Python stdlib sources, this repo's DESIGN.md) and real command output.

  python3 golden/build_agentic.py golden/long.messages.json

Run on the host (stdlib only). The output is committed, so the token sequence does not depend on
the host's Python version afterwards; ref/torchref/make_tokens.py messages encodes it.
"""
import hashlib
import json
import os
import subprocess
import sys

STDLIB = "/usr/lib/python3.12"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOOLS = [
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List the entries of a directory with their sizes.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute directory path"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file and return its full contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute file path"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command in the workspace and return its combined output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command line"},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Search files under a path for a regular expression; returns matching lines.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "File or directory to search"}},
            "required": ["pattern", "path"]}}},
]

SOURCES = {}


def read(path):
    data = open(path, encoding="utf-8").read()
    SOURCES[path] = hashlib.sha256(data.encode()).hexdigest()
    return data


def sh(cmd):
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=STDLIB)
    return (out.stdout + out.stderr).rstrip("\n")


def listing(path):
    rows = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        kind = "dir " if os.path.isdir(full) else "file"
        size = "-" if os.path.isdir(full) else str(os.path.getsize(full))
        rows.append(f"{kind} {size:>7}  {name}")
    return "\n".join(rows)


class Conv:
    def __init__(self):
        self.messages = []
        self.n = 0

    def say(self, role, content):
        self.messages.append({"role": role, "content": content})

    def call(self, reasoning, calls):
        """One assistant turn with tool calls, then one tool message per call."""
        tcs = []
        results = []
        for name, args, result in calls:
            cid = f"call_{self.n}"
            self.n += 1
            tcs.append({"id": cid, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}})
            results.append({"role": "tool", "tool_call_id": cid, "content": result})
        self.messages.append({"role": "assistant", "reasoning_content": reasoning, "content": "",
                              "tool_calls": tcs})
        self.messages.extend(results)

    def answer(self, reasoning, content):
        self.messages.append({"role": "assistant", "reasoning_content": reasoning, "content": content})


def main():
    out = sys.argv[1]
    j = f"{STDLIB}/json"
    c = Conv()
    c.say("system", "You are a careful software engineering agent working in a Linux workspace. Use the "
                    "tools to inspect code and run commands before answering. Quote file paths and line "
                    "numbers when you refer to code. Keep answers precise and grounded in what you read.")
    c.say("user", "A service of ours round-trips configuration through Python's json module and we see "
                  "two oddities: very large floats come back as `Infinity`, and some keys change order "
                  f"between versions. The interpreter's standard library is under {STDLIB}. Please read the "
                  "json package and explain exactly how floats are parsed and serialised, which code path "
                  "(C accelerator or pure Python) is used, and where key ordering is decided.")
    c.call("Start by listing the package to see which modules exist.",
           [("list_directory", {"path": j}, listing(j))])
    c.call("The package entry point wires dumps/loads to the encoder and decoder; read it first.",
           [("read_file", {"path": f"{j}/__init__.py"}, read(f"{j}/__init__.py"))])
    c.call("Now the decoder and the scanner, which together implement parsing.",
           [("read_file", {"path": f"{j}/decoder.py"}, read(f"{j}/decoder.py")),
            ("read_file", {"path": f"{j}/scanner.py"}, read(f"{j}/scanner.py"))])
    c.call("Serialisation lives in encoder.py; float formatting is in floatstr.",
           [("read_file", {"path": f"{j}/encoder.py"}, read(f"{j}/encoder.py"))])
    cmd = ("python3 -c \"import json, json.decoder as d, json.scanner as s, json.encoder as e; "
           "print(json.dumps({'big': 1e400, 'neg': -0.0, 'z': 1, 'a': 2})); "
           "print(json.loads('[1e400, -1e400, 1.5e308]')); "
           "print(d.scanstring is d.py_scanstring, s.make_scanner is s.py_make_scanner, "
           "e.c_make_encoder is None); "
           "print(json.dumps({'b': 1, 'a': 2}, sort_keys=True))\"")
    c.call("Check behaviour and which implementation is active with a quick experiment.",
           [("run_command", {"command": cmd, "timeout": 30}, sh(cmd))])
    c.answer("Summarise the float and ordering behaviour from the code read so far.",
             "Floats: `json.loads` hands numbers to `parse_float` (default `float`), so `1e400` overflows to "
             "`inf` in `float()` itself and the decoder never sees an error. On the way out, "
             "`encoder.floatstr` maps `inf` to `Infinity` unless `allow_nan=False`, in which case it raises "
             "`ValueError`. The C accelerator (`c_make_encoder`, `c_scanstring`, `c_make_scanner`) is used "
             "when `_json` imports, and the pure-Python versions are fallbacks with the same semantics. "
             "Key order is the dict's insertion order unless `sort_keys=True`; nothing in the decoder "
             "reorders keys, so differing order between versions comes from the producer, not from json.")
    c.say("user", "Thanks. Next, our report generator wraps text with textwrap and keeps a priority queue "
                  "with heapq. Long words sometimes get split mid-token and ties in the queue come out in an "
                  "order we did not expect. Find where those behaviours are implemented and explain them, "
                  "with line references.")
    cmd = "grep -n 'def \\|break_long_words\\|break_on_hyphens' textwrap.py | head -60"
    c.call("Search textwrap for the wrapping entry points and the long-word options.",
           [("search_code", {"pattern": "def |break_long_words|break_on_hyphens",
                             "path": f"{STDLIB}/textwrap.py"}, sh(cmd))])
    c.call("Read the whole module to follow _handle_long_word and _wrap_chunks.",
           [("read_file", {"path": f"{STDLIB}/textwrap.py"}, read(f"{STDLIB}/textwrap.py"))])
    c.call("Now heapq: tie behaviour depends on tuple comparison and the sift functions.",
           [("read_file", {"path": f"{STDLIB}/heapq.py"}, read(f"{STDLIB}/heapq.py"))])
    cmd = ("python3 -c \"import heapq, textwrap; h=[]; "
           "[heapq.heappush(h, (1, n)) for n in 'dcba']; print([heapq.heappop(h) for _ in range(4)]); "
           "print(textwrap.wrap('supercalifragilisticexpialidocious and more words here', width=10)); "
           "print(textwrap.wrap('supercalifragilisticexpialidocious and more', width=10, "
           "break_long_words=False))\"")
    c.call("Confirm both behaviours with a small run.",
           [("run_command", {"command": cmd, "timeout": 30}, sh(cmd))])
    c.answer("Explain both behaviours.",
             "textwrap: `TextWrapper._handle_long_word` splits a chunk that is longer than the remaining "
             "width whenever `break_long_words` is true (the default), so a long identifier is cut at the "
             "column limit; with `break_long_words=False` it is placed on its own line and may overflow. "
             "heapq: the heap invariant only orders by comparison, so equal priorities fall back to "
             "comparing the next tuple element; when that element is not comparable or not meaningful, add "
             "a monotonically increasing counter as the second field to make ties first-in, first-out.")
    c.say("user", "Last, a different project: read the design document of the inference engine in "
                  f"{REPO}/DESIGN.md and tell me which of its decisions would be affected if the cluster "
                  "had a switch instead of a ring.")
    c.call("Read the design document.",
           [("read_file", {"path": f"{REPO}/DESIGN.md"}, read(f"{REPO}/DESIGN.md"))])
    c.answer("Identify ring-specific decisions.",
             "Decision 2 is the ring-specific one: the two-hop all-reduce with a proxy relay exists because "
             "opposite boxes have no cable. With a switch every rank reaches every other directly, so a "
             "one-shot all-reduce (each rank writes its vector to all three peers) would replace the relay "
             "and remove a hop. Decisions 1 and 3 to 10 do not depend on the topology.")
    c.say("user", "Good. Now write a short engineering note that combines the three investigations: what we "
                  "learned about json floats and key order, the textwrap and heapq behaviours with the fixes "
                  "you recommend, and the design-document point. Keep it under 400 words.")
    case = {"thinking_mode": "thinking", "reasoning_effort": None, "tools": TOOLS, "messages": c.messages,
            "sources_sha256": SOURCES}
    json.dump(case, open(out, "w"), indent=1, ensure_ascii=False)
    chars = sum(len(m.get("content") or "") for m in c.messages)
    print(f"{len(c.messages)} messages, {chars} content chars -> {out}")


if __name__ == "__main__":
    main()
