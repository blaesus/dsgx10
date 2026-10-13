"""Turns a golden case's token ids and Engram hashes (tokens.npy [L] int32, hashes.npy [L, 2, 24]
int64, from the golden harness) into a hash case for engram_hashcheck and a token stream for
engram_cachesim / engram_e2e. Plain Python, no numpy:
    python3 golden_case.py <golden-case-dir> <out-prefix>
writes <out-prefix>.cases (one 'ENGC' case) and <out-prefix>.i32.
"""
import ast
import struct
import sys


def load_npy(path):
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise SystemExit(f"{path}: not an .npy file")
        major = f.read(2)[0]
        hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        hdr = ast.literal_eval(f.read(hlen).decode("latin1"))
        if hdr["fortran_order"]:
            raise SystemExit(f"{path}: Fortran order")
        return hdr["descr"], hdr["shape"], f.read()


def main():
    case, out = sys.argv[1], sys.argv[2]
    tdescr, tshape, tok = load_npy(f"{case}/tokens.npy")
    hdescr, hshape, hashes = load_npy(f"{case}/hashes.npy")
    if tdescr != "<i4" or hdescr != "<i8" or len(tshape) != 1 or hshape[0] != tshape[0] or hshape[2] != 24:
        raise SystemExit(f"unexpected arrays: tokens {tdescr} {tshape}, hashes {hdescr} {hshape}")
    L = tshape[0]
    with open(f"{out}.cases", "wb") as f:
        f.write(b"ENGC" + struct.pack("<qq", L, 0))
        f.write(tok)
        f.write(b"\x01" * L)
        f.write(hashes)
    with open(f"{out}.i32", "wb") as f:
        f.write(tok)
    print(f"{L} positions -> {out}.cases, {out}.i32")


if __name__ == "__main__":
    main()
