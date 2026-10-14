#!/usr/bin/env python3
"""Fetch the routed experts of chosen MoE layers from the BF16 master by HTTP range requests.

    python -I fetch_master.py OUT --layers 3 40 [--threads 4]

For every tensor model.layers.{L}.mlp.experts.*.{gate,up,down}_proj.weight it reads the shard's
safetensors header (cached in OUT/headers/), fetches the tensor's bytes into OUT/tensors/<name>.bin
(written to .part and renamed when complete, so a rerun resumes), and records name, shard, dtype,
shape, byte count and the SHA-256 of the bytes in OUT/manifest.jsonl (one line per tensor, appended
as each completes; a rerun rehashes nothing it already recorded). HF_TOKEN is used when set.
The bytes are data only: nothing fetched is executed.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, struct, sys, threading, time, urllib.request

REPO, REV = "zai-org/GLM-5.3-BF16", "9d2398f478cab2de883137db3a36ad2c96205e24"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"


def req(path, a=None, b=None, tries=8):
    h = {}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    if a is not None:
        h["Range"] = f"bytes={a}-{b}"
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + path, headers=h), timeout=600) as f:
                data = f.read()
            if a is not None and len(data) != b - a + 1:
                raise IOError(f"short read {len(data)} of {b - a + 1}")
            return data
        except Exception as e:  # network: retry with backoff
            if t == tries - 1:
                raise
            print(f"retry {path} {a}: {e}", flush=True)
            time.sleep(min(60, 2 ** t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    for d in ("headers", "tensors"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    ip = os.path.join(out, "model.safetensors.index.json")
    if not os.path.exists(ip):
        open(ip, "wb").write(req("model.safetensors.index.json"))
    wm = json.load(open(ip))["weight_map"]
    pat = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
    names = sorted((n for n in wm if (m := pat.match(n)) and int(m.group(1)) in a.layers),
                   key=lambda n: tuple(int(x) if x.isdigit() else x for x in pat.match(n).groups()))
    headers = {}
    for shard in sorted({wm[n] for n in names}):
        hp = os.path.join(out, "headers", shard + ".json")
        if not os.path.exists(hp):
            n = struct.unpack("<Q", req(shard, 0, 7))[0]
            json.dump({"n": n, "h": json.loads(req(shard, 8, 8 + n - 1))}, open(hp + ".part", "w"))
            os.replace(hp + ".part", hp)
        headers[shard] = json.load(open(hp))
    mp = os.path.join(out, "manifest.jsonl")
    done = set()
    if os.path.exists(mp):
        done = {json.loads(l)["name"] for l in open(mp) if l.strip()}
    todo = [n for n in names if n not in done]
    print(f"{len(names)} tensors in {len({wm[n] for n in names})} shards; {len(todo)} to fetch", flush=True)
    lock = threading.Lock()
    t0, nbytes = time.time(), [0]

    def one(name):
        shard = wm[name]
        hh = headers[shard]
        v = hh["h"][name]
        s, e = v["data_offsets"]
        tp = os.path.join(out, "tensors", name + ".bin")
        if not (os.path.exists(tp) and os.path.getsize(tp) == e - s):
            data = req(shard, 8 + hh["n"] + s, 8 + hh["n"] + e - 1)
            with open(tp + ".part", "wb") as f:
                f.write(data)
            os.replace(tp + ".part", tp)
        else:
            data = open(tp, "rb").read()
        rec = {"name": name, "shard": shard, "dtype": v["dtype"], "shape": v["shape"], "bytes": e - s,
               "sha256": hashlib.sha256(data).hexdigest()}
        with lock:
            with open(mp, "a") as f:
                f.write(json.dumps(rec) + "\n")
            nbytes[0] += e - s
            dt = time.time() - t0
            print(f"{name} {nbytes[0] / 1e9:.2f} GB {nbytes[0] / 1e6 / dt:.1f} MB/s", flush=True)

    with cf.ThreadPoolExecutor(a.threads) as ex:
        for f in [ex.submit(one, n) for n in todo]:
            f.result()
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
