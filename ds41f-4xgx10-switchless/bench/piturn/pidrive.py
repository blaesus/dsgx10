#!/usr/bin/env python3
"""The edit bench's tasks through pi itself: one pi process per task, read/edit/write only (nothing is
run), in a scratch workspace made as editbench makes it (the task's files at the pinned commit, its setup
applied), then editbench's checks. pi is isolated: PI_CODING_AGENT_DIR (models.json pointing at dsfront,
the key from $DSGB10_DEV_KEY), --no-session, no context files, extensions, skills, templates or themes,
--offline, --no-approve.
  --mode text: pi -p (print mode: the final answer on stdout)
  --mode json: pi -p --mode json, each event line stamped on arrival (time_ns) in OUT/events/
  --mode tui:  interactive pi on a pseudo-terminal (120x40), the task as its initial message; ended once
               dsfront's dump shows the task's final answer (an output record whose finish is not
               tool_calls) and 1.5 s have passed
Each run's wall-clock window (time_ns, the dump files' clock) goes to OUT/runs.jsonl.
  pidrive.py OUT --dump DUMP_DIR [--mode text|json|tui] [--repeats N] [--tasks a,b] [--thinking off]
part.sh runs it on a dev engine (docs/engine.md, "pi's turnaround")."""
import argparse
import fcntl
import glob
import json
import os
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bench/
WT = os.environ.get('EDIT_REPO') or os.path.dirname(os.path.dirname(B))  # a checkout that has edit_bench.json's commit
sys.path.insert(0, B)
import editbench  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--dump', required=True)
ap.add_argument('--mode', default='text', choices=['text', 'json', 'tui'])
ap.add_argument('--repeats', type=int, default=1)
ap.add_argument('--tasks', default='')
ap.add_argument('--thinking', default='off')
ap.add_argument('--prompt', default='', help='instead of each task\'s prompt (a mock engine\'s test)')
ap.add_argument('--timeout', type=float, default=240)
ap.add_argument('--deadline-s', type=float, default=400, help='no task starts after this many seconds')
a = ap.parse_args()
spec = json.load(open(os.path.join(B, 'edit_bench.json')))
tasks = [t for t in spec['tasks'] if not a.tasks or t['id'] in a.tasks.split(',')]
PI = shutil.which('pi')
assert PI and os.environ.get('PI_CODING_AGENT_DIR') and os.environ.get('DSGB10_DEV_KEY')
os.makedirs(os.path.join(a.out, 'events'), exist_ok=True)
runs = open(os.path.join(a.out, 'runs.jsonl'), 'a')
FLAGS = ['--no-session', '--no-context-files', '--no-extensions', '--no-skills', '--no-prompt-templates',
         '--no-themes', '--offline', '--no-approve', '--tools', 'read,edit,write', '--provider', 'dsgb10',
         '--model', 'deepseek-v4.1-flash-uncensored', '--thinking', a.thinking]


def outs():
    return set(glob.glob(os.path.join(a.dump, '*.out.json')))


def final_seen(before):
    for p in sorted(outs() - before):
        try:
            if json.load(open(p)).get('finish') != 'tool_calls':
                return True
        except (OSError, ValueError):
            pass
    return False


def run_print(cmd, ws, env, tag):
    ev = open(os.path.join(a.out, 'events', tag + '.jsonl'), 'w') if a.mode == 'json' else None
    p = subprocess.Popen(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    text, deadline = [], time.time() + a.timeout
    for line in p.stdout:  # each line stamped on arrival
        t = time.time_ns()
        if ev:
            try:
                e = json.loads(line)
                ev.write(json.dumps({'t': t, 'type': e.get('type'),
                                     'sub': (e.get('assistantMessageEvent') or {}).get('type'),
                                     'tool': e.get('toolName'), 'role': (e.get('message') or {}).get('role')}) + '\n')
            except ValueError:
                pass
        else:
            text.append(line.decode(errors='replace'))
        if time.time() > deadline:
            p.kill()
            break
    err = p.stderr.read().decode(errors='replace')
    rc = p.wait()
    return rc, ''.join(text)[-400:], err[-400:]


def run_tui(cmd, ws, env, tag):
    before = outs()
    m, s = pty.openpty()
    fcntl.ioctl(s, termios.TIOCSWINSZ, struct.pack('HHHH', 40, 120, 0, 0))
    p = subprocess.Popen(cmd, cwd=ws, env=dict(env, TERM='xterm-256color'), stdin=s, stdout=s, stderr=s,
                         start_new_session=True)
    os.close(s)
    n, deadline, done_at = 0, time.time() + a.timeout, None
    while p.poll() is None and time.time() < deadline:
        r, _, _ = select.select([m], [], [], 0.05)
        if r:
            try:
                n += len(os.read(m, 65536))  # drained, never answered
            except OSError:
                break
        if done_at is None and final_seen(before):
            done_at = time.time()
        if done_at is not None and time.time() - done_at > 1.5:
            break
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    os.close(m)
    return p.returncode, f'{n} bytes drawn; final answer {"seen" if done_at else "not seen"}', ''


t_begin = time.time()
for rep in range(a.repeats):
    for t in tasks:
        if time.time() - t_begin > a.deadline_s:
            print(f'pidrive: past --deadline-s, {t["id"]} rep {rep} not started', flush=True)
            continue
        ws = os.path.join(a.out, 'ws', f'rep{rep}', t['id'])
        shutil.rmtree(ws, ignore_errors=True)
        orig = editbench.task_files(spec, t, WT)
        for p_, text in orig.items():
            os.makedirs(os.path.dirname(os.path.join(ws, p_)), exist_ok=True)
            with open(os.path.join(ws, p_), 'w', encoding='utf-8', newline='') as f:
                f.write(text)
        cmd = [PI] + (['-p'] if a.mode != 'tui' else []) + (['--mode', 'json'] if a.mode == 'json' else []) + FLAGS
        cmd += ['--', a.prompt or t['prompt']]
        tag = f'{a.mode}-rep{rep}-{t["id"]}'
        t0 = time.time_ns()
        rc, tail, err = (run_tui if a.mode == 'tui' else run_print)(cmd, ws, dict(os.environ), tag)
        t1 = time.time_ns()
        res = editbench.finish_task(ws, orig, t)
        rec = dict(mode=a.mode, rep=rep, task=t['id'], t0=t0, t1=t1, rc=rc, passed=res['passed'], checks=res['checks'],
                   changed=res['changed'], tail=tail, err=err)
        runs.write(json.dumps(rec) + '\n')
        runs.flush()
        print(f'pidrive {tag}: exit {rc}, {(t1 - t0) / 1e9:.1f} s, checks {sum(res["checks"])}/{len(res["checks"])}'
              + ('' if res['passed'] else ' FAILED') + (f'; stderr: {err.strip()[:200]}' if err.strip() else ''), flush=True)
