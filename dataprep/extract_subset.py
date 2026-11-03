#!/usr/bin/env python3
"""
Copy a random (or leading) subset of rows out of a binary vector dataset
(<prefix>.meta.txt / .ids.bin / .vecs.bin, see convert_to_binary.cpp) into a
new dataset with the same layout.

Rows are chosen uniformly without replacement, then the source is scanned once
in sequential chunks and only the chosen rows are written, so the read pattern
is sequential even though the sample is random. Output keeps source row order.

  python extract_subset.py SRC_PREFIX DST_PREFIX --n 10000000
"""
import argparse
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src_prefix", help="e.g. .../zinc20_split_database.bin")
    p.add_argument("dst_prefix", help="e.g. .../zinc20_10M.bin")
    p.add_argument("--n", type=int, default=10_000_000)
    p.add_argument("--first", action="store_true",
                   help="take the first n rows instead of a random sample "
                        "(fast, but the file order may be biased)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=int, default=1_000_000, help="rows per read")
    a = p.parse_args()

    N, D = map(int, open(a.src_prefix + ".meta.txt").read().split())
    if a.n > N:
        raise SystemExit(f"--n {a.n:,} is larger than the source ({N:,} rows)")
    vecs = np.memmap(a.src_prefix + ".vecs.bin", dtype=np.float32, mode="r", shape=(N, D))
    ids = np.memmap(a.src_prefix + ".ids.bin", dtype=np.int32, mode="r", shape=(N,))

    if a.first:
        pick = np.arange(a.n, dtype=np.int64)
    else:
        pick = np.sort(np.random.default_rng(a.seed).choice(N, size=a.n, replace=False))

    t0 = time.time()
    last = int(pick[-1]) + 1
    with open(a.dst_prefix + ".vecs.bin", "wb") as vf, open(a.dst_prefix + ".ids.bin", "wb") as idf:
        lo_p = 0
        for lo in range(0, last, a.chunk):
            hi = min(lo + a.chunk, N)
            hi_p = np.searchsorted(pick, hi)            # picks inside [lo, hi)
            if hi_p > lo_p:
                local = pick[lo_p:hi_p] - lo
                np.asarray(vecs[lo:hi])[local].tofile(vf)
                np.asarray(ids[lo:hi])[local].tofile(idf)
                lo_p = hi_p
            if (lo // a.chunk) % 50 == 0:
                print(f"  scanned {hi:,}/{N:,} rows ({time.time() - t0:.0f}s)", flush=True)
    with open(a.dst_prefix + ".meta.txt", "w") as fh:
        fh.write(f"{a.n} {D}\n")
    print(f"wrote {a.n:,} x {D} -> {a.dst_prefix}.* ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()


