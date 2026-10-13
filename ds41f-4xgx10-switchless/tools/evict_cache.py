"""Drop the clean page-cache pages of the given files/directories (posix_fadvise DONTNEED, no root):
on GB10 the page cache competes with CUDA allocations."""
import os
import sys

n = 0
for root in sys.argv[1:]:
    paths = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        n += 1
print("fadvised", n, "files")
