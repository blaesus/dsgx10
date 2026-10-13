# sass_loops.py SASS [function-substring]: per kernel in a `cuobjdump -sass` listing, its loops (backward
# branches) with their static instruction counts and opcode mix, to read a kernel's instructions a step.
# Lossless weight compression study (docs/engine.md): today's GEMVs and MoE kernels, bench_wcomp's decoders,
# fixdec.cu's.
import re, sys, collections
path = sys.argv[1]; want = sys.argv[2] if len(sys.argv) > 2 else ''
funcs = collections.OrderedDict(); cur = None
for line in open(path):
    m = re.search(r'Function : (\S+)', line)
    if m: cur = m.group(1); funcs[cur] = []; continue
    m = re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(.*?);', line)
    if m and cur is not None:
        funcs[cur].append((int(m.group(1), 16), m.group(2).strip()))
for f, ins in funcs.items():
    if want and want not in f: continue
    print('==', f[:150], len(ins), 'instructions')
    addr = {a: i for i, (a, _) in enumerate(ins)}
    for i, (a, t) in enumerate(ins):
        m = re.search(r'\bBRA\b.*?(0x[0-9a-f]+)', t)
        if not m: continue
        tgt = int(m.group(1), 16)
        if tgt <= a and tgt in addr:
            body = ins[addr[tgt]:i + 1]
            ops = collections.Counter()
            for _, s in body:
                s2 = re.sub(r'^@!?U?P\w+\s+', '', s)
                ops[s2.split()[0].split('.')[0]] += 1
            n = len(body)
            mma = sum(v for k, v in ops.items() if 'MMA' in k)
            print('  loop %05x..%05x: %d instr, %d MMA; ' % (tgt, a, n, mma) +
                  ' '.join('%s:%d' % kv for kv in ops.most_common(14)))
