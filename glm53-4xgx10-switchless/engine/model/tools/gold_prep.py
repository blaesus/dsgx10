"""A golden run's data as raw files for glm_run's mtpgold and spec modes (docs/engine.md, "Drafts"). Run where the
golden run is (with the reference's torch), then copy PREFIX.* to every box's ~/dsgb10-data/<tree>/gold/:

  python -I gold_prep.py RUN PREFIX [--l78]

  PREFIX.hfinal.bf16     state/hfinal.pt: the final hidden after model.norm, every row, raw bf16 [N][6144]
                         (checked bit-equal to final_hidden_sample.npy at the sample rows)
  PREFIX.tokens.i32      tokens.npy
  PREFIX.argmax.i32      final/argmax.npy: the main model's argmax per row
  PREFIX.mtp_argmax.i32  mtp/argmax.npy by main row (-1 where the row has no MTP row)
  PREFIX.seqs.txt        the sequences: name, offset, length
  PREFIX.l78.bf16        (--l78) layers/L78/hidden.npy, the MTP layer's output at the sample rows, by main row
                         (zeros elsewhere): mtpgold's per-row comparison on rank 0

It prints the reference's own teacher-forced agreement (its MTP's argmax against its main argmax at t + 1 and
against the token at t + 2), the numbers mtpgold's are read against.
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('prefix')
    ap.add_argument('--l78', action='store_true')
    a = ap.parse_args()
    run, out = a.run, a.prefix
    # the final hidden
    h = torch.load(os.path.join(run, 'state', 'hfinal.pt'), map_location='cpu')
    if isinstance(h, dict):
        h = h[list(h.keys())[0]]
    h = h.reshape(-1, 6144)
    if h.dtype != torch.bfloat16:
        print('hfinal is %s: rounded to bf16' % h.dtype)
        h = h.to(torch.bfloat16)
    raw = h.contiguous().view(torch.int16).numpy()
    rows = np.load(os.path.join(run, 'sample_rows.npy'))
    ref = np.load(os.path.join(run, 'final_hidden_sample.npy'), mmap_mode='r')
    mine = ((raw[rows].astype(np.uint32) & 0xFFFF) << 16).view(np.float32)
    print('hfinal: %d rows; bit-equal to final_hidden_sample at %d sample rows: %.6f' % (
        raw.shape[0], len(rows), np.mean(mine == np.asarray(ref))))
    raw.tofile(out + '.hfinal.bf16')
    # the ids
    tok = np.load(os.path.join(run, 'tokens.npy')).astype(np.int32)
    am = np.load(os.path.join(run, 'final', 'argmax.npy')).astype(np.int32)
    mam = np.load(os.path.join(run, 'mtp', 'argmax.npy')).astype(np.int32)
    rm = np.load(os.path.join(run, 'mtp', 'rows_main.npy')).astype(np.int64)
    by = np.full(tok.shape[0], -1, np.int32)
    by[rm] = mam
    seqs = json.load(open(os.path.join(run, 'manifest.json')))['sequences']
    tok.tofile(out + '.tokens.i32')
    am.tofile(out + '.argmax.i32')
    by.tofile(out + '.mtp_argmax.i32')
    with open(out + '.seqs.txt', 'w') as f:
        for s in seqs:
            f.write('%s %d %d\n' % (s['name'], s['off'], s['n']))
    for s in seqs:
        o, n = s['off'], s['n']
        r = [t for t in range(o, o + n - 2) if by[t] >= 0]
        print('%-8s mtp rows %5d  reference: MTP argmax = main argmax at t+1 %.4f, = token t+2 %.4f' % (
            s['name'], len(r), np.mean([by[t] == am[t + 1] for t in r]), np.mean([by[t] == tok[t + 2] for t in r])))
    if a.l78:
        l78 = np.load(os.path.join(run, 'layers', 'L78', 'hidden.npy'), mmap_mode='r')
        sr = set(int(x) for x in rows)
        main_rows = [int(x) for x in rm] if l78.shape[0] == rm.shape[0] else [int(x) for x in rm if int(x) in sr]
        assert len(main_rows) == l78.shape[0], (len(main_rows), l78.shape)
        o = np.zeros((tok.shape[0], 6144), np.uint16)
        for i, x in enumerate(main_rows):
            o[x] = (np.asarray(l78[i], np.float32).view(np.uint32) >> 16).astype(np.uint16)
        o.tofile(out + '.l78.bf16')
        print('l78: %d rows' % len(main_rows))


if __name__ == '__main__':
    main()
