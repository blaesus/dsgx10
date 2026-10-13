import json, struct, sys, os, glob, re, collections
d = sys.argv[1]
out = []
for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        out.append((k, v["dtype"], v["shape"], os.path.basename(f), v["data_offsets"][1]-v["data_offsets"][0]))
json.dump(out, open(sys.argv[2], "w"))
# summarize: collapse layer and expert indices
agg = collections.OrderedDict()
for k, dt, sh, f, nb in out:
    g = re.sub(r"(^|\.)layers\.\d+\.", "layers.N.", k)
    g = re.sub(r"\.experts\.\d+\.", ".experts.E.", g)
    key = (g, dt, tuple(sh))
    a = agg.setdefault(key, [0, 0])
    a[0] += 1; a[1] += nb
tot = 0
for (g, dt, sh), (c, nb) in agg.items():
    tot += nb
    print(f"{c:6d} {dt:8s} {str(list(sh)):28s} {nb/2**30:8.2f}GiB {g}")
print("total GiB", tot/2**30)
