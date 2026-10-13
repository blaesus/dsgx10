"""Do the rows our hash picks look trained? For each Engram sub-table (n-gram order x head), the rows hit
most often in the sequence against a random sample of the same sub-table: if the hash matches training,
rows of frequent n-grams were updated far more than a random row, and their sizes should stand apart.

  golden/run_ref.sh py /work/golden/engram_rowfreq.py /out/probe5120.npz
"""
import sys
from collections import Counter

import numpy as np

d = np.load(sys.argv[1])
for li, layer in enumerate((1, 14)):
    k = f"L{layer:02d}."
    rows, norms, samp = d["hashes"][:, li], d[k + "row_deq_norm"], d[k + "sample_deq_norm"]
    gate = d[k + "gate"].max(1)
    print(f"== layer {layer}: per sub-table, row-norm percentile within a random sample of 4096 rows "
          f"(50 = typical), for the 20 most frequent rows here, all rows hit here, and gate>0.1 positions")
    for o in range(3):
        cells = []
        for hd in range(8):
            j = o * 8 + hd
            s = np.sort(samp[j])
            pct = lambda v: 100.0 * np.searchsorted(s, v) / len(s)  # noqa: E731
            top = [r for r, _ in Counter(rows[:, j].tolist()).most_common(20)]
            first = {r: norms[np.argmax(rows[:, j] == r), j] for r in top}
            cells.append(f"{np.median([pct(first[r]) for r in top]):3.0f}/{np.median(pct(norms[:, j])):3.0f}"
                         f"/{np.median(pct(norms[gate > 0.1, j])) if (gate > 0.1).any() else float('nan'):3.0f}")
        print(f"   {o + 2}-gram heads: " + "  ".join(cells))
    print(f"   distinct rows per sub-table: " + " ".join(str(len(set(rows[:, j].tolist()))) for j in range(24)))
