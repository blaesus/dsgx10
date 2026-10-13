"""Print a .dsw file's header and table of contents (standard library only).

  python3 toc.py FILE                 one line per tensor: name dtype shape kind layout crc32c slice
  python3 toc.py FILE --replicated    only replicated tensors, for comparing CRCs across ranks
"""

import struct
import sys

HDR = struct.Struct("<8sIIIIII QQQQ II 64s256s32s")
ENT = struct.Struct("<128s128s96s II 4Q QQ IIII II 4Q")
DTYPE = {1: "F32", 2: "BF16", 3: "F8_E4M3", 4: "F8_E8M0", 5: "E2M1X2"}
KIND = {1: "replicated", 2: "rows", 3: "cols", 4: "rows-seg"}
LAYOUT = {0: "natural", 1: "L1", 2: "L2", 3: "L3"}  # dsw.h Layout; version 1 files are all natural


def cstr(b):
    return b.split(b"\0", 1)[0].decode()


def main():
    path = sys.argv[1]
    only_rep = "--replicated" in sys.argv
    with open(path, "rb") as f:
        h = HDR.unpack(f.read(HDR.size))
        version, rank, world, n, toc_off, data_off, data_bytes = h[1], h[3], h[4], h[5], h[7], h[8], h[9]
        print(f"# {path}: version {version}, rank {rank}/{world}, {n} tensors, {data_bytes / 2**30:.2f} GiB, "
              f"created {cstr(h[15])}, from {cstr(h[14])}")
        f.seek(toc_off)
        for _ in range(n):
            e = ENT.unpack(f.read(512)[: ENT.size])
            if only_rep and e[11] != 1:
                continue
            shape = "x".join(str(d) for d in e[5 : 5 + e[4]])
            print(f"{cstr(e[0])}\t{DTYPE[e[3]]}\t{shape}\t{KIND[e[11]]}\t{LAYOUT[e[16]]}\t{e[14]:08x}\t{cstr(e[2])}")


if __name__ == "__main__":
    main()
