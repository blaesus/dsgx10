#!/usr/bin/env python3
"""The agentic-edit bench: pi-style multi-turn coding tasks (edit_bench.json) through an OpenAI endpoint,
the model's read/edit/write calls executed against a scratch copy of repo files (stdlib only).

  bench.py --modes edit --model M --key-file F --edit-out DIR [--edit-thinking off,max] [--edit-tasks a,b]
  editbench.py report DIR --dump-dir D --log RANK0_LOG [--traces T]

Requests are pi 0.84.3's (pi -p --tools read,edit,write): its system prompt and tools as captured, the
user's turn as a text block, the assistant's tool calls with compact JSON arguments, tool results as pi's
own texts (read's truncation notes, edit's matching rules and errors, write's byte count). Greedy unless
--edit-temperature says otherwise; pi itself sends no temperature. Nothing the model writes is executed:
there is no bash tool, and the checks at the end are regexes and, for Python, a compile() of the source.
Each request carries `user: edit:RUN:THINKING:TASK:TURN`, which the report joins to dsfront's --dump-dir
records (prompt and output ids) and to rank 0's per-request lines (decode rate, drafts per step).

pi's prompt, tool definitions and tool texts are pi's (MIT, Copyright (c) 2025 Mario Zechner): THIRD_PARTY.md.
"""
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# --- pi's tools (dist/core/tools: read.js, truncate.js, edit.js, edit-diff.js, write.js) ---

MAX_LINES, MAX_BYTES = 2000, 50 * 1024


class ToolError(Exception):
    pass


def js_len(s):  # a JS string's length: UTF-16 code units
    return len(s.encode("utf-16-le")) // 2


def js_num(x):
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)


def fmt_size(n):
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def truncate_head(content):
    lines = content.split("\n") if content else []
    if content.endswith("\n"):
        lines.pop()
    if len(lines) <= MAX_LINES and len(content.encode()) <= MAX_BYTES:
        return dict(content=content, truncated=False)
    if len(lines[0].encode()) > MAX_BYTES:
        return dict(content="", truncated=True, by="bytes", lines=0, first_too_long=True)
    out, n, by = [], 0, "lines"
    for i, line in enumerate(lines[:MAX_LINES]):
        b = len(line.encode()) + (1 if i else 0)
        if n + b > MAX_BYTES:
            by = "bytes"
            break
        out.append(line)
        n += b
    if len(out) >= MAX_LINES and n <= MAX_BYTES:
        by = "lines"
    return dict(content="\n".join(out), truncated=True, by=by, lines=len(out), first_too_long=False)


class Workspace:
    """Files under `root` stand in for `cwd`; paths outside it do not exist (nothing outside is read or written)."""

    def __init__(self, root, cwd):
        self.root, self.cwd = root, cwd.rstrip("/")

    def resolve(self, path):  # pi's resolveToCwd: '@' stripped, '~' expanded, relative to cwd
        p = re.sub(r"[  -   　]", " ", path)
        p = p[1:] if p.startswith("@") else p
        if p == "~" or p.startswith("~/"):
            p = os.path.expanduser(p)
        v = os.path.normpath(os.path.join(self.cwd, p))
        if v == self.cwd or v.startswith(self.cwd + "/"):
            return v, os.path.join(self.root, os.path.relpath(v, self.cwd))
        return v, None


def tool_read(ws, a):
    path, offset, limit = a["path"], a.get("offset"), a.get("limit")
    v, real = ws.resolve(path)
    if real is None or not os.path.exists(real):
        raise ToolError(f"ENOENT: no such file or directory, access '{v}'")
    if os.path.isdir(real):
        raise ToolError("EISDIR: illegal operation on a directory, read")
    lines = open(real, "rb").read().decode("utf-8", "replace").split("\n")
    start = max(0, int(offset) - 1) if offset else 0
    if start >= len(lines):
        raise ToolError(f"Offset {js_num(offset)} is beyond end of file ({len(lines)} lines total)")
    user_lines = None
    if limit is not None:
        end = min(start + int(limit), len(lines))
        sel, user_lines = "\n".join(lines[start:end]), end - start
    else:
        sel = "\n".join(lines[start:])
    tr = truncate_head(sel)
    if tr["truncated"] and tr["first_too_long"]:
        size = fmt_size(len(lines[start].encode()))
        return (f"[Line {start + 1} is {size}, exceeds {fmt_size(MAX_BYTES)} limit. Use bash: sed -n '{start + 1}p' "
                f"{path} | head -c {MAX_BYTES}]")
    if tr["truncated"]:
        last = start + tr["lines"]
        lim = "" if tr["by"] == "lines" else f" ({fmt_size(MAX_BYTES)} limit)"
        return tr["content"] + f"\n\n[Showing lines {start + 1}-{last} of {len(lines)}{lim}. Use offset={last + 1} to continue.]"
    if user_lines is not None and start + user_lines < len(lines):
        rest = len(lines) - (start + user_lines)
        return tr["content"] + f"\n\n[{rest} more lines in file. Use offset={start + user_lines + 1} to continue.]"
    return tr["content"]


def fuzzy_norm(t):
    t = unicodedata.normalize("NFKC", t)
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub("[‘’‚‛]", "'", t)
    t = re.sub("[“”„‟]", '"', t)
    t = re.sub("[‐-―−]", "-", t)
    return re.sub("[  -   　]", " ", t)


def fuzzy_find(content, old):  # (found, index, length, fuzzy)
    i = content.find(old)
    if i >= 0:
        return True, i, len(old), False
    fc, fo = fuzzy_norm(content), fuzzy_norm(old)
    i = fc.find(fo)
    return (True, i, len(fo), True) if i >= 0 else (False, -1, 0, False)


def lines_with_endings(s):
    return re.findall(r"[^\n]*\n|[^\n]+", s)


def apply_reps(content, reps, offset=0):
    for at, n, new, _ in reversed(reps):
        at -= offset
        content = content[:at] + new + content[at + n:]
    return content


def apply_reps_keeping_lines(original, base, reps):  # applyReplacementsPreservingUnchangedLines
    orig = lines_with_endings(original)
    spans, off = [], 0
    for line in lines_with_endings(base):
        spans.append((off, off + len(line)))
        off += len(line)
    if len(orig) != len(spans):
        raise ToolError("Cannot preserve unchanged lines because the base content has a different line count.")
    groups = []
    for r in sorted(reps, key=lambda r: r[0]):
        lo = next((i for i, (s, e) in enumerate(spans) if s <= r[0] < e), -1)
        if lo < 0:
            raise ToolError("Replacement range is outside the base content.")
        hi = lo
        while hi < len(spans) and spans[hi][1] < r[0] + r[1]:
            hi += 1
        if hi >= len(spans):
            raise ToolError("Replacement range is outside the base content.")
        if groups and lo < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], hi + 1)
            groups[-1][2].append(r)
        else:
            groups.append([lo, hi + 1, [r]])
    out, at = "", 0
    for lo, hi, rs in groups:
        out += "".join(orig[at:lo])
        s, e = spans[lo][0], spans[hi - 1][1]
        out += apply_reps(base[s:e], rs, s)
        at = hi
    return out + "".join(orig[at:])


def apply_edits(content, edits, path):  # applyEditsToNormalizedContent
    lf = lambda s: s.replace("\r\n", "\n").replace("\r", "\n")
    ed = [(lf(e["oldText"]), lf(e["newText"])) for e in edits]
    n = len(ed)
    for i, (old, _) in enumerate(ed):
        if not old:
            raise ToolError(f"oldText must not be empty in {path}." if n == 1 else f"edits[{i}].oldText must not be empty in {path}.")
    fuzzy = any(fuzzy_find(content, old)[3] for old, _ in ed)
    base = fuzzy_norm(content) if fuzzy else content
    reps = []
    for i, (old, new) in enumerate(ed):
        found, at, ln, _ = fuzzy_find(base, old)
        if not found:
            raise ToolError(f"Could not find the exact text in {path}. The old text must match exactly including all whitespace "
                            f"and newlines." if n == 1 else f"Could not find edits[{i}] in {path}. The oldText must match exactly "
                            f"including all whitespace and newlines.")
        k = fuzzy_norm(base).count(fuzzy_norm(old))
        if k > 1:
            raise ToolError(f"Found {k} occurrences of the text in {path}. The text must be unique. Please provide more context "
                            f"to make it unique." if n == 1 else f"Found {k} occurrences of edits[{i}] in {path}. Each oldText "
                            f"must be unique. Please provide more context to make it unique.")
        reps.append((at, ln, new, i))
    reps.sort(key=lambda r: r[0])
    for p, c in zip(reps, reps[1:]):
        if p[0] + p[1] > c[0]:
            raise ToolError(f"edits[{p[3]}] and edits[{c[3]}] overlap in {path}. Merge them into one edit or target disjoint regions.")
    out = apply_reps_keeping_lines(content, base, reps) if fuzzy else apply_reps(base, reps)
    if out == content:
        raise ToolError(f"No changes made to {path}. The replacement produced identical content. This might indicate an issue "
                        f"with special characters or the text not existing as expected." if n == 1 else
                        f"No changes made to {path}. The replacements produced identical content.")
    return out


def tool_edit(ws, a):
    path, edits = a["path"], a["edits"]
    if not isinstance(edits, list) or not edits:
        raise ToolError("Edit tool input is invalid. edits must contain at least one replacement.")
    _, real = ws.resolve(path)
    if real is None or not os.path.exists(real):
        raise ToolError(f"Could not edit file: {path}. Error code: ENOENT.")
    if os.path.isdir(real):
        raise ToolError("EISDIR: illegal operation on a directory, read")
    raw = open(real, "rb").read().decode("utf-8", "replace")
    bom = "﻿" if raw.startswith("﻿") else ""
    text = raw[len(bom):]
    cr, nl = text.find("\r\n"), text.find("\n")
    crlf = nl >= 0 and 0 <= cr < nl
    out = apply_edits(text.replace("\r\n", "\n").replace("\r", "\n"), edits, path)
    with open(real, "w", encoding="utf-8", newline="") as f:
        f.write(bom + (out.replace("\n", "\r\n") if crlf else out))
    return f"Successfully replaced {len(edits)} block(s) in {path}."


def tool_write(ws, a):
    v, real = ws.resolve(a["path"])
    if real is None:
        raise ToolError(f"EACCES: permission denied, open '{v}'")
    os.makedirs(os.path.dirname(real), exist_ok=True)
    with open(real, "w", encoding="utf-8", newline="") as f:
        f.write(a["content"])
    return f"Successfully wrote {js_len(a['content'])} bytes to {a['path']}"


def is_single_edit(v):
    return isinstance(v, dict) and isinstance(v.get("oldText"), str) and isinstance(v.get("newText"), str)


def prepare_edit(a):  # prepareEditArguments: edits as a JSON string or one object, or a legacy oldText/newText
    if not isinstance(a, dict):
        return a
    e = a.get("edits")
    if isinstance(e, str):
        try:
            p = json.loads(e)
            if isinstance(p, list):
                a["edits"] = p
            elif is_single_edit(p):
                a["edits"] = [p]
        except ValueError:
            pass
    elif is_single_edit(e):
        a["edits"] = [e]
    if not (isinstance(a.get("oldText"), str) and isinstance(a.get("newText"), str)):
        return a
    edits = list(a["edits"]) if isinstance(a.get("edits"), list) else []
    edits.append({"oldText": a["oldText"], "newText": a["newText"]})
    rest = {k: v for k, v in a.items() if k not in ("oldText", "newText")}
    rest["edits"] = edits
    return rest


def check_schema(schema, v, at=""):  # the parts of typebox's check that these tools' schemas use
    t = schema.get("type")
    ok = {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str),
          "number": isinstance(v, (int, float)) and not isinstance(v, bool)}.get(t, True)
    if not ok:
        return [f"  - {at or 'root'}: must be {t}"]
    errs = []
    if t == "object":
        for k in schema.get("required", []):
            if k not in v:
                errs.append(f"  - {at + '.' if at else ''}{k}: must have required properties {k}")
        for k, s in schema.get("properties", {}).items():
            if k in v:
                errs += check_schema(s, v[k], f"{at}.{k}" if at else k)
    elif t == "array" and "items" in schema:
        for i, x in enumerate(v):
            errs += check_schema(schema["items"], x, f"{at}.{i}")
    return errs


def coerce(schema, a):  # Value.Convert for numbers given as strings; optional nulls dropped
    for k, s in schema.get("properties", {}).items():
        if k in a and a[k] is None and k not in schema.get("required", []):
            del a[k]
        elif k in a and s.get("type") == "number" and isinstance(a[k], str):
            try:
                a[k] = float(a[k]) if "." in a[k] else int(a[k])
            except ValueError:
                pass
    return a


def js_json(v):  # JSON.stringify: compact, integral floats as integers
    def fix(x):
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, dict):
            return {k: fix(y) for k, y in x.items()}
        if isinstance(x, list):
            return [fix(y) for y in x]
        return x
    return json.dumps(fix(v), separators=(",", ":"), ensure_ascii=False)


TOOLS = {"read": tool_read, "edit": tool_edit, "write": tool_write}


def call_tool(ws, schemas, name, args):
    """pi's tool step: prepareArguments, schema validation, execute; (text, error?)"""
    if name not in TOOLS or name not in schemas:
        return f"Tool {name} not found", True
    if name == "edit":
        args = prepare_edit(args)
    if isinstance(args, dict):
        args = coerce(schemas[name], args)
    errs = check_schema(schemas[name], args)
    if errs:
        return (f'Validation failed for tool "{name}":\n' + "\n".join(errs) + "\n\nReceived arguments:\n"
                + json.dumps(args, indent=2, ensure_ascii=False)), True
    try:
        return TOOLS[name](ws, args), False
    except ToolError as e:
        return str(e), True
    except OSError as e:
        return f"{e.strerror}: {e.filename}", True


# --- the run ---

def chat(url, key, body, timeout=3600):
    """One streamed request: content, reasoning, tool calls (dsfront sends each call whole), usage, times
    (seconds after the request: first event of any kind, last reasoning, first answer, end)."""
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
    t0 = time.perf_counter()
    content, reasoning, calls = [], [], {}
    r = dict(finish=None, usage=None, error=None, t_first=None, t_reason_last=None, t_answer_first=None)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("error"):
                r["error"] = ev["error"]
            if ev.get("usage"):
                r["usage"] = ev["usage"]
            now = time.perf_counter() - t0
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                    r["t_reason_last"] = now
                if d.get("content") or d.get("tool_calls"):
                    if r["t_answer_first"] is None:
                        r["t_answer_first"] = now
                if d.get("content"):
                    content.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    c["id"] = tc.get("id") or c["id"]
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                if (d.get("reasoning_content") or d.get("content") or d.get("tool_calls")) and r["t_first"] is None:
                    r["t_first"] = now
                if ch.get("finish_reason"):
                    r["finish"] = ch["finish_reason"]
    r.update(t_end=time.perf_counter() - t0, content="".join(content), reasoning="".join(reasoning),
             calls=[calls[i] for i in sorted(calls)])
    return r


def task_files(spec, t, repo):
    """The task's files at the pinned commit (sha256-checked), with its setup applied."""
    import hashlib
    files = {}
    for p in t["files"]:
        b = subprocess.run(["git", "-C", repo, "show", f"{spec['commit']}:{spec['prefix']}{p}"], capture_output=True,
                           check=True).stdout
        if hashlib.sha256(b).hexdigest() != spec["sha256"][p]:
            sys.exit(f"edit: {p} at {spec['commit']} is not the file the tasks were written for")
        files[p] = b.decode()
    for s in t["setup"]:
        assert files[s["path"]].count(s["old"]) == 1, s
        files[s["path"]] = files[s["path"]].replace(s["old"], s["new"])
    return files


def finish_task(ws_root, orig, t):
    """The checks, and whether each changed Python file still compiles (compile() only: nothing is run)."""
    checks, compiled = [], {}
    for c in t["checks"]:
        try:
            text = open(os.path.join(ws_root, c["path"]), encoding="utf-8", errors="replace").read()
        except OSError:
            text = ""
        checks.append(bool(re.search(c["re"], text)) == c["want"])
    changed = []
    for p, before in orig.items():
        try:
            now = open(os.path.join(ws_root, p), encoding="utf-8", errors="replace").read()
        except OSError:
            now = None
        if now != before:
            changed.append(p)
            if p.endswith(".py") and now is not None:
                try:
                    compile(now, p, "exec")
                    compiled[p] = True
                except (SyntaxError, ValueError):
                    compiled[p] = False
    return dict(checks=checks, passed=all(checks) and all(compiled.values()), changed=changed, compiles=compiled)


def run_task(a, base, spec, t, thinking, run, out, f_turns, rep=0):
    ws_root = os.path.join(out, "ws", thinking, f"rep{rep}", t["id"])
    shutil.rmtree(ws_root, ignore_errors=True)
    orig = task_files(spec, t, a.edit_repo)
    for p, text in orig.items():
        os.makedirs(os.path.dirname(os.path.join(ws_root, p)), exist_ok=True)
        with open(os.path.join(ws_root, p), "w", encoding="utf-8", newline="") as f:
            f.write(text)
    ws = Workspace(ws_root, spec["cwd"])
    schemas = {x["function"]["name"]: x["function"]["parameters"] for x in spec["tools"]}
    msgs = [{"role": "system", "content": spec["system"].replace("{cwd}", spec["cwd"]).replace("{pi_dir}", spec["pi_dir"])},
            {"role": "user", "content": [{"type": "text", "text": t["prompt"]}]}]
    kwargs = {"thinking": False} if thinking == "off" else {"thinking": True, "reasoning_effort": thinking}
    t0, kinds = time.time(), []
    name = f"{thinking} {t['id']}" + (f" rep {rep}" if a.edit_repeats > 1 else "")
    for turn in range(a.edit_max_turns):
        tag = f"edit:{run}:{thinking}:{rep}:{t['id']}:{turn}"
        body = {"model": spec["model"], "messages": msgs, "stream": True, "stream_options": {"include_usage": True},
                "store": False, "max_completion_tokens": spec["max_completion_tokens"], "tools": spec["tools"],
                "chat_template_kwargs": kwargs, "user": tag}
        if a.edit_temperature != "none":
            body["temperature"] = float(a.edit_temperature)
        try:
            r = chat(base, a.key, body)
        except Exception as e:  # noqa: BLE001 (recorded, and the task ends)
            r = dict(error=repr(e), calls=[], content="", reasoning="", finish=None, usage=None, t_end=None)
        calls = []
        for c in r["calls"]:
            try:
                args = json.loads(c["arguments"]) if c["arguments"].strip() else {}
            except ValueError:
                args = {}
            calls.append(dict(id=c["id"], name=c["name"], args=args, raw=c["arguments"]))
        names = [c["name"] for c in calls]
        kind = "prose" if not names else names[0] if len(set(names)) == 1 else "mixed"
        if r.get("error"):
            kind = "error"
        kinds.append(kind)
        msg = {"role": "assistant", "content": r["content"] or None}
        if thinking != "off" and r["reasoning"]:
            msg["reasoning_content"] = r["reasoning"]
        if calls:
            msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": js_json(c["args"])}}
                                 for c in calls]
        results = []
        for c in calls:
            if r["finish"] == "length":
                text, err = (f'Tool call "{c["name"]}" was not executed: the response hit the output token limit, so its '
                             f'arguments may be truncated. Re-issue the tool call with complete arguments.'), True
            else:
                text, err = call_tool(ws, schemas, c["name"], json.loads(json.dumps(c["args"])))
            results.append({"role": "tool", "content": text, "tool_call_id": c["id"]})
            c.update(error=text if err else None, result_chars=len(text))
            if c["name"] == "edit" and isinstance(c["args"].get("edits"), list):
                c["edits"] = len(c["args"]["edits"])
        u = r.get("usage") or {}
        rec = dict(run=run, thinking=thinking, rep=rep, task=t["id"], turn=turn, tag=tag, kind=kind, finish=r["finish"],
                   error=r.get("error"), temperature=body.get("temperature"),
                   prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                   cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   t_first=r.get("t_first"), t_reason_last=r.get("t_reason_last"), t_answer_first=r.get("t_answer_first"),
                   t_end=r.get("t_end"), content=r["content"], reasoning=r["reasoning"],
                   calls=[{k: c.get(k) for k in ("name", "raw", "edits", "error", "result_chars")} for c in calls])
        f_turns.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_turns.flush()
        errs = sum(1 for c in calls if c.get("error"))
        print(f"edit {name} turn {turn}: {kind}, {u.get('completion_tokens')} tokens"
              + (f" ({rec['reasoning_tokens']} reasoning)" if rec["reasoning_tokens"] else "")
              + f", {r['t_end'] or 0:.1f} s" + (f", {errs} tool errors" if errs else "")
              + (f", error {r['error']}" if r.get("error") else ""), flush=True)
        if not calls or r.get("error"):
            break
        msgs += [msg] + results
    res = finish_task(ws_root, orig, t)
    res.update(run=run, thinking=thinking, rep=rep, task=t["id"], turns=len(kinds), kinds=kinds,
               elapsed_s=round(time.time() - t0, 1))
    print(f"edit {name}: {len(kinds)} turns ({' '.join(kinds)}), checks {sum(res['checks'])}/{len(res['checks'])}"
          f"{'' if all(res['compiles'].values()) else ', does not compile'}, {res['elapsed_s']} s", flush=True)
    return res


def run(a, base):
    spec = json.load(open(a.edit_spec))
    by_id = {t["id"]: t for t in spec["tasks"]}
    tasks = spec["tasks"] if a.edit_tasks == "all" else [by_id[x] for x in a.edit_tasks.split(",")]
    out = a.edit_out or f"edit-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(out, exist_ok=True)
    run_tag = a.edit_tag or os.path.basename(os.path.abspath(out))
    t0, res = time.time(), []
    with open(os.path.join(out, "turns.jsonl"), "a") as f_turns, open(os.path.join(out, "tasks.jsonl"), "a") as f_tasks:
        for rep in range(a.edit_repeats):  # outermost: a deadline cuts the last repeat, not every repeat's last tasks
            for thinking in a.edit_thinking.split(","):
                for t in tasks:
                    if a.edit_deadline_s and time.time() - t0 > a.edit_deadline_s:
                        print(f"edit {thinking} {t['id']} rep {rep}: not started (past --edit-deadline-s "
                              f"{a.edit_deadline_s})", flush=True)
                        continue
                    r = run_task(a, base, spec, t, thinking, run_tag, out, f_turns, rep)
                    f_tasks.write(json.dumps(r) + "\n")
                    f_tasks.flush()
                    res.append(r)
    return res


def add_args(ap):
    ap.add_argument("--edit-spec", default=os.path.join(HERE, "edit_bench.json"))
    ap.add_argument("--edit-tasks", default="all", help="all, or task ids (comma-separated, run in that order)")
    ap.add_argument("--edit-thinking", default="off", help="off, or a reasoning effort (max, high); comma-separated runs")
    ap.add_argument("--edit-max-turns", type=int, default=12)
    ap.add_argument("--edit-repeats", type=int, default=1, help="run everything this many times (sampled decoding: "
                    "without a seed the engine draws one per request)")
    ap.add_argument("--edit-deadline-s", type=float, default=0, help="start no task after this many seconds")
    ap.add_argument("--edit-temperature", default="0", help="sent as temperature; none: not sent (as pi does)")
    ap.add_argument("--edit-out", default="", help="the run's directory: turns.jsonl, tasks.jsonl, ws/")
    ap.add_argument("--edit-tag", default="", help="the run's name in each request's user field (default: the out dir's)")
    ap.add_argument("--edit-repo", default=os.path.dirname(HERE), help="the git checkout the task files come from")


# --- the report: the run joined to dsfront's dump and rank 0's log ---

RE_DRAFTS = re.compile(r"serve: req (\d+) drafts: (\d+) verify steps, ([\d.]+) drafts verified and ([\d.]+) accepted per step, "
                       r"steps verifying 0\.\.5: (\d+) (\d+) (\d+) (\d+) (\d+) (\d+); gate ([\d.]+) \(([\d.]+)-([\d.]+)\); "
                       r"GPU ms by rows:(.*)")
RE_REQ = re.compile(r"serve: req (\d+) session [0-9a-f]+: prompt (\d+), reused (\d+) \((\w+)\), prefill ([\d.]+) ms \(\d+ tok/s\), "
                    r"(\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\), (\w+);")


def parse_log(path):
    eng = {}
    for line in open(path, errors="replace"):
        m = RE_DRAFTS.search(line)
        if m:
            g = m.groups()
            eng.setdefault(int(g[0]), {}).update(
                vsteps=int(g[1]), verified=float(g[2]), accepted=float(g[3]), hist=[int(x) for x in g[4:10]],
                gate=float(g[10]), gpu_by_rows={int(r): (float(ms), int(n)) for r, ms, n in
                                                re.findall(r"(\d+): ([\d.]+) \((\d+)\)", g[13])})
            continue
        m = RE_REQ.search(line)
        if m:
            g = m.groups()
            eng.setdefault(int(g[0]), {}).update(
                n_prompt=int(g[1]), reused=int(g[2]), how=g[3], prefill_ms=float(g[4]), out_tokens=int(g[5]),
                decode_ms=float(g[6]), decode_tps=float(g[7]), end=g[8])
    return eng


def write_npy(path, ids):
    import array
    hdr = "{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }" % len(ids)
    hdr += " " * (63 - (10 + len(hdr)) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + len(hdr).to_bytes(2, "little") + hdr.encode())
        f.write(array.array("i", ids).tobytes())


def copied(prompt, out, n=8):
    """Output tokens inside verbatim runs of at least n tokens found in the prompt (greedy, longest run first):
    what a prompt-lookup drafter could at best have proposed."""
    at = {}
    for i in range(len(prompt) - n + 1):
        at.setdefault(tuple(prompt[i:i + n]), []).append(i)
    i = got = 0
    while i + n <= len(out):
        best = 0
        for j in at.get(tuple(out[i:i + n]), [])[:64]:
            k = n
            while i + k < len(out) and j + k < len(prompt) and out[i + k] == prompt[j + k]:
                k += 1
            best = max(best, k)
        if best:
            got += best
            i += best
        else:
            i += 1
    return got


def agg(rows):
    """Token-weighted: decode rate = tokens / decode time; tokens per step = (steps + accepted) / steps."""
    e = [r for r in rows if r.get("decode_ms")]
    dec = sum(r["decode_tps"] * r["decode_ms"] / 1e3 for r in e)
    ms = sum(r["decode_ms"] for r in e)
    v = [r for r in e if r.get("vsteps")]
    steps = sum(r["vsteps"] for r in v)
    hist = [sum(r["hist"][i] for r in v) for i in range(6)]
    pf = [r for r in rows if r.get("prefill_ms") is not None and r.get("n_prompt") is not None]
    c = [r for r in rows if r.get("copied") is not None and r.get("out_ids")]
    ver = sum(r["vsteps"] * r["verified"] for r in v)
    return dict(turns=len(rows), tokens=sum(r.get("out_tokens") or 0 for r in rows),
                copied=sum(r["copied"] for r in c) / sum(len(r["out_ids"]) for r in c) if c else None,
                decode_tps=dec / ms * 1e3 if ms else None,
                tok_per_step=sum(r["vsteps"] * (1 + r["accepted"]) for r in v) / steps if steps else None,
                verified_per_step=ver / steps if steps else None,
                accept=sum(r["vsteps"] * r["accepted"] for r in v) / ver if ver else None,
                gate=sum(r["vsteps"] * r["gate"] for r in v) / steps if steps else None,
                hist=[h / sum(hist) for h in hist] if sum(hist) else None,
                ttft_ms=statistics.median(r["ttft_s"] * 1e3 for r in rows if r.get("ttft_s") is not None) if any(
                    r.get("ttft_s") is not None for r in rows) else None,
                prefilled=sum(r["n_prompt"] - r["reused"] for r in pf),
                prefill_tps=sum(r["n_prompt"] - r["reused"] for r in pf) / sum(r["prefill_ms"] for r in pf) * 1e3
                if pf and sum(r["prefill_ms"] for r in pf) else None)


def fmt(x, s="{:.1f}"):
    return "-" if x is None else s.format(x)


def fmt_table(groups):
    lines = ["| | turns | output tokens | copied | decode tok/s | tokens/step | drafts verified/step (accepted) | "
             "steps verifying 0..5 | gate | TTFT median | prefilled | prefill tok/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, g in groups:
        lines.append(f"| {name} | {g['turns']} | {g['tokens']} | {fmt(g['copied'] and 100 * g['copied'], '{:.0f} %')} | "
                     f"{fmt(g['decode_tps'])} | {fmt(g['tok_per_step'], '{:.2f}')} | "
                     f"{fmt(g['verified_per_step'], '{:.2f}')} ({fmt(g['accept'] and 100 * g['accept'], '{:.0f} %')}) | "
                     f"{' '.join(f'{100 * h:.0f}' for h in g['hist']) + ' %' if g['hist'] else '-'} | "
                     f"{fmt(g['gate'], '{:.3f}')} | {fmt(g['ttft_ms'], '{:.0f} ms')} | {g['prefilled']} | "
                     f"{fmt(g['prefill_tps'], '{:.0f}')} |")
    return "\n".join(lines)


def fmt_compare(base, run, kinds):
    """This run against a baseline's joined.jsonl rows, by kind: baseline -> this run."""
    lines = ["| kind | turns | decode tok/s | tokens/step | drafts accepted | drafts verified/step | steps verifying 5 | "
             "gate |", "|---|---|---|---|---|---|---|---|"]
    for k in kinds:
        b, r = agg([x for x in base if x["kind"] == k] if k != "all" else base), agg(
            [x for x in run if x["kind"] == k] if k != "all" else run)
        pair = lambda key, s="{:.1f}", m=1: f"{fmt(b[key] and m * b[key], s)} → {fmt(r[key] and m * r[key], s)}"
        lines.append(f"| {k} | {b['turns']} → {r['turns']} | {pair('decode_tps')} | {pair('tok_per_step', '{:.2f}')} | "
                     f"{pair('accept', '{:.0f} %', 100)} | {pair('verified_per_step', '{:.2f}')} | "
                     f"{fmt(b['hist'] and 100 * b['hist'][5], '{:.0f} %')} → {fmt(r['hist'] and 100 * r['hist'][5], '{:.0f} %')} | "
                     f"{pair('gate', '{:.3f}')} |")
    return "\n".join(lines)


def report(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="editbench.py report")
    ap.add_argument("dir", nargs="+", help="run directories (turns.jsonl, tasks.jsonl)")
    ap.add_argument("--dump-dir", action="append", default=[], help="dsfront --dump-dir (one per run dir, or one)")
    ap.add_argument("--log", action="append", default=[], help="rank 0's log (one per run dir, or one)")
    ap.add_argument("--traces", default="", help="write each turn's ids here (tokens.npy: prompt then output)")
    ap.add_argument("--out", default="", help="summary.md and joined.jsonl go here (default: the first run directory)")
    ap.add_argument("--baseline", default="", help="an earlier report's joined.jsonl: by kind, it -> this run")
    a = ap.parse_args(argv)
    rows, tasks = [], []
    for i, d in enumerate(a.dir):
        dump = a.dump_dir[min(i, len(a.dump_dir) - 1)]
        eng = parse_log(a.log[min(i, len(a.log) - 1)])
        req_of = {}
        for fn in os.listdir(dump):
            if re.fullmatch(r"\d+\.json", fn):
                rec = json.load(open(os.path.join(dump, fn)))
                req_of[(rec.get("request") or {}).get("user")] = rec
        for line in open(os.path.join(d, "turns.jsonl")):
            t = json.loads(line)
            rec = req_of.get(t["tag"])
            if rec:
                t["req_id"] = rec["req_id"]
                t.update(eng.get(rec["req_id"], {}))
                t["prompt_ids"] = rec["ids"]
                try:
                    o = json.load(open(os.path.join(dump, f"{rec['req_id']:06}.out.json")))
                    t.update(out_ids=o["ids"][:o["kept"]], ids_after_stop=len(o["ids"]) - o["kept"], ttft_s=o["ttft_s"],
                             total_s=o["total_s"])
                    t["copied"] = copied(t["prompt_ids"], t["out_ids"])
                except OSError:
                    pass
            rows.append(t)
        tasks += [json.loads(x) for x in open(os.path.join(d, "tasks.jsonl"))]
    missing = sum(1 for r in rows if "req_id" not in r)
    if a.traces:
        os.makedirs(a.traces, exist_ok=True)
        with open(os.path.join(a.traces, "index.jsonl"), "w") as idx:
            for r in rows:
                if "prompt_ids" not in r or "out_ids" not in r:
                    continue
                sub = os.path.join(r["run"], r["thinking"], f"rep{r.get('rep', 0)}", r["task"], f"t{r['turn']:02d}")
                os.makedirs(os.path.join(a.traces, sub), exist_ok=True)
                write_npy(os.path.join(a.traces, sub, "tokens.npy"), r["prompt_ids"] + r["out_ids"])
                write_npy(os.path.join(a.traces, sub, "positions.npy"), [])
                write_npy(os.path.join(a.traces, sub, "record_positions.npy"), [])
                meta = {k: r.get(k) for k in ("run", "thinking", "rep", "temperature", "task", "turn", "kind", "req_id",
                                              "reasoning_tokens", "reused", "how", "vsteps", "accepted", "verified", "hist",
                                              "gate", "decode_tps", "finish")}
                meta.update(dir=sub, n_prompt=len(r["prompt_ids"]), n_out=len(r["out_ids"]))
                json.dump(meta, open(os.path.join(a.traces, sub, "meta.json"), "w"))
                idx.write(json.dumps(meta) + "\n")
    out = []
    base = [json.loads(x) for x in open(a.baseline)] if a.baseline else []
    order = ["edit", "write", "read", "prose", "mixed", "error"]
    for th in sorted({r["thinking"] for r in rows}, key=lambda x: x != "off"):
        rs = [r for r in rows if r["thinking"] == th]
        kinds = sorted({r["kind"] for r in rs}, key=lambda k: order.index(k) if k in order else 9)
        temps = {r.get("temperature", 0.0) for r in rs}  # runs before the field: greedy
        sampling = "greedy" if temps == {0.0} else "dsfront's sampling defaults" if temps == {None} else f"temperatures {temps}"
        reps = len({r.get("rep", 0) for r in rs})
        out.append(f"### thinking {th}, {sampling}{f', {reps} repeats' if reps > 1 else ''}: by kind of call\n")
        out.append(fmt_table([(k, agg([r for r in rs if r["kind"] == k])) for k in kinds] + [("all", agg(rs))]))
        bs = [r for r in base if r["thinking"] == th]
        if bs:
            out.append(f"\n### thinking {th}: against the baseline ({a.baseline}; baseline → this run)\n")
            out.append(fmt_compare(bs, rs, [k for k in kinds if any(r["kind"] == k for r in bs)] + ["all"]))
        out.append(f"\n### thinking {th}: by task\n")
        out.append(fmt_table([(t, agg([r for r in rs if r["task"] == t])) for t in dict.fromkeys(r["task"] for r in rs)]))
        if th != "off":
            out.append(f"\n### thinking {th}: reasoning and answer (client clock)\n")
            out.append("| kind | turns | reasoning tokens | reasoning tok/s | answer tokens | answer tok/s |\n|---|---|---|---|---|---|")
            for k in kinds:
                ks = [r for r in rs if r["kind"] == k and r.get("t_first") is not None and r.get("t_end")]
                rt = sum(r.get("reasoning_tokens") or 0 for r in ks)
                rsec = sum((r["t_reason_last"] - r["t_first"]) for r in ks if r.get("t_reason_last") and r.get("reasoning_tokens"))
                at = sum((r.get("completion_tokens") or 0) - (r.get("reasoning_tokens") or 0) for r in ks)
                asec = sum(r["t_end"] - (r.get("t_reason_last") or r["t_first"]) for r in ks)
                out.append(f"| {k} | {len(ks)} | {rt} | {rt / rsec if rsec else 0:.1f} | {at} | {at / asec if asec else 0:.1f} |")
        ts = [t for t in tasks if t["thinking"] == th]
        by_task = {}
        for t in ts:
            by_task.setdefault(t["task"], []).append(t)
        out.append(f"\ntasks run: {len(ts)}, checks passed {sum(1 for t in ts if t['passed'])}/{len(ts)}: " + ", ".join(
            f"{k} {sum(1 for t in v if t['passed'])}/{len(v)} ({'/'.join(str(t['turns']) for t in v)} turns)"
            for k, v in by_task.items()) + "\n")
    if missing:
        out.append(f"{missing} turns had no dump record (not joined)")
    text = "\n".join(out)
    print(text)
    out_dir = a.out or a.dir[0]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "joined.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("prompt_ids", "out_ids")}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(sys.argv[2:])
    else:
        sys.exit(__doc__)
