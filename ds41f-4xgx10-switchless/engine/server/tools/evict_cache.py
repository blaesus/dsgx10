"""Drop this box's page cache of the model's big files (checkpoint shards, Engram shards, per-rank
weights) before the engine starts. On GB10 a CUDA allocation sees only free pages, not reclaimable
cache (docs/findings.md), and a box can hold ~30 GB of cache from the checkpoint alone. The engine
reads its files with O_DIRECT, so their cache is never of use to it. posix_fadvise needs no root and
drops only clean cached pages; the files are untouched. Prints this box's MemFree before and after.

  python3 evict_cache.py [FILE_OR_GLOB...]   # this project's paths below, and these (run4.sh and
                                             # service.sh pass the fleet file's for the box's rank)
A leading ~/ is this box's home directory."""
import glob
import os
import sys

GLOBS = [
    "~/DeepSeek-V4.1-Flash*/*.safetensors", "/data/models/DeepSeek-V4.1-Flash*/*.safetensors",
    "~/dsv41-4x-ring/engram/*.bin", "/data/models/dsv41-engram/*.bin", "~/dsgb10/engram/*.bin",
    "/data/dsgb10/engram/*.bin", "~/dsgb10/weights/*.dsw", "/data/dsgb10/weights/*.dsw",
] + sys.argv[1:]


def mem_free_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemFree:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


before = mem_free_gb()
for g in GLOBS:
    for p in glob.glob(os.path.expanduser(g)):
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass
print(f"{os.uname().nodename}: MemFree {before:.1f} -> {mem_free_gb():.1f} GiB after dropping the model files' cache")
