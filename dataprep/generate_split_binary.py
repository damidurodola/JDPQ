#!/usr/bin/env python3
"""
Generate Morgan count vectors from ZINC .smi file(s) and split them into three
disjoint sets with no duplicate vectors and no duplicate ids:

  <out>_train.bin.{meta.txt,ids.bin,vecs.bin}     --train-size vectors (binary)
  <out>_database.bin.{meta.txt,ids.bin,vecs.bin}  everything else       (binary)
  <out>_query.txt                                 --query-size vectors  (text)

Binary layout matches convert_to_binary.cpp / readData.h:
  .meta.txt -> "N D"          .ids.bin  -> N x int32
  .vecs.bin -> N x D x float32, row-major

Ids are the numeric ZINC number (ZINC000000141911_2 -> 141911). Only the first
compound seen per number is kept; numbers >= 2^31 are skipped (int32 ids).

Stages (the first is the long one):
  1. fingerprint the whole input in bounded batches, dropping repeated ids, and
     append (vector, id, 64-bit vector hash) to <out>.tmp_* files
  2. drop repeated vectors (stable sort of the hashes; first occurrence wins),
     then pick query and train rows uniformly at random from the unique rows
  3. stream the temp vectors once and write the three outputs
Temp files are about the size of the database (~1 TB at 1B vectors); they are
deleted at the end unless --keep-temp. --reuse-temp reruns stages 2-3 only.
"""
import argparse
import glob
import os
import time
import numpy as np

# --dim must reach surechembl before it is imported (it builds its fingerprint
# generator at import time), so it is read from argv here and passed via FP_DIM.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--dim", type=int, default=256)
os.environ["FP_DIM"] = str(_pre.parse_known_args()[0].dim)

from surechembl import iter_smiles_zinco, _fp_worker

DIM = int(os.environ["FP_DIM"])
ID_LIMIT = 2 ** 31          # ids are stored as int32


def _batches(path_glob, batch, standardize):
    tasks = []
    for smi, cid in iter_smiles_zinco(path_glob):
        tasks.append((smi, cid, standardize))
        if len(tasks) >= batch:
            yield tasks
            tasks = []
    if tasks:
        yield tasks


def stage1_fingerprint(path_glob, tmp, standardize, n_workers, batch, limit):
    """Fingerprint input -> tmp.vecs / tmp.ids / tmp.hash. Returns row count."""
    import multiprocessing as mp
    rng = np.random.default_rng(12345)
    mult = rng.integers(1, 2 ** 63, size=DIM, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    seen_id = np.zeros(ID_LIMIT, dtype=bool)          # 2 GB bitmap over int32 ids
    n_rows = n_bad = n_big = n_dup_id = n_input = 0
    t0 = time.time()

    def process(results, vf, idf, hf):
        nonlocal n_rows, n_bad, n_big, n_dup_id
        ids, fps = [], []
        for cid, fp in results:
            if fp is None:
                n_bad += 1
                continue
            try:
                num = int(cid[4:].split("_")[0]) if cid.startswith("ZINC") else int(cid)
            except ValueError:
                n_bad += 1
                continue
            if num >= ID_LIMIT:
                n_big += 1
                continue
            ids.append(num)
            fps.append(fp)
        if not ids:
            return
        ids = np.asarray(ids, dtype=np.int64)
        fps = np.stack(fps).astype(np.float32, copy=False)
        fresh = ~seen_id[ids]
        n_dup_id += int((~fresh).sum())
        # within the batch keep only the first row of each id
        idx = np.flatnonzero(fresh)
        _, first = np.unique(ids[idx], return_index=True)
        n_dup_id += len(idx) - len(first)
        idx = idx[np.sort(first)]
        if limit is not None:
            idx = idx[: max(0, limit - n_rows)]
        if len(idx) == 0:
            return
        ids, fps = ids[idx], fps[idx]
        seen_id[ids] = True
        h = (fps.view(np.uint32).astype(np.uint64) * mult).sum(axis=1, dtype=np.uint64)
        fps.tofile(vf)
        ids.astype(np.int32).tofile(idf)
        h.tofile(hf)
        n_rows += len(ids)

    pool = mp.Pool(n_workers)
    with open(tmp + ".vecs", "wb") as vf, open(tmp + ".ids", "wb") as idf, \
            open(tmp + ".hash", "wb") as hf:
        gen = _batches(path_glob, batch, standardize)
        cur = next(gen, None)
        cur_async = pool.map_async(_fp_worker, cur, chunksize=256) if cur else None
        while cur is not None:
            nxt = next(gen, None)       # parse the next lines while workers run
            nxt_async = pool.map_async(_fp_worker, nxt, chunksize=256) if nxt else None
            process(cur_async.get(), vf, idf, hf)
            n_input += len(cur)
            print(f"  input {n_input:,}  kept {n_rows:,}  "
                  f"({n_input / (time.time() - t0):,.0f}/s)", flush=True)
            if limit is not None and n_rows >= limit:
                if nxt_async is not None:
                    nxt_async.get()       # never leave work in flight at shutdown
                break
            cur, cur_async = nxt, nxt_async
    pool.close()                          # orderly shutdown; terminate() can hang
    pool.join()
    print(f"  stage 1: {n_rows:,} rows kept; skipped {n_bad:,} unparseable, "
          f"{n_dup_id:,} repeated ids, {n_big:,} ids >= 2^31 ({time.time() - t0:.0f}s)")
    return n_rows


def _write_meta(prefix, n):
    with open(prefix + ".meta.txt", "w") as fh:
        fh.write(f"{n} {DIM}\n")


def stage2_select(tmp, n_rows, train_size, query_size, seed):
    """Return uint8 role per row: 0 dropped (dup vector), 1 db, 2 train, 3 query."""
    h = np.fromfile(tmp + ".hash", dtype=np.uint64)
    assert len(h) == n_rows
    order = np.argsort(h, kind="stable")        # ties stay in file order
    hs = h[order]
    first = np.ones(n_rows, dtype=bool)
    first[1:] = hs[1:] != hs[:-1]
    keep = order[first]
    del order, hs, first, h
    n_unique = len(keep)
    print(f"  stage 2: {n_unique:,} unique vectors of {n_rows:,} "
          f"({n_rows - n_unique:,} repeated vectors dropped)")
    need = train_size + query_size
    if n_unique <= need:
        raise SystemExit(f"only {n_unique:,} unique vectors, need more than {need:,}")
    role = np.zeros(n_rows, dtype=np.uint8)
    role[keep] = 1
    rng = np.random.default_rng(seed)
    pick = np.sort(keep)[rng.choice(n_unique, size=need, replace=False)]
    rng.shuffle(pick)
    role[pick[:query_size]] = 3
    role[pick[query_size:]] = 2
    return role


def stage3_write(tmp, out_prefix, role, chunk):
    n_rows = len(role)
    vecs = np.memmap(tmp + ".vecs", dtype=np.float32, mode="r", shape=(n_rows, DIM))
    ids = np.fromfile(tmp + ".ids", dtype=np.int32)
    outs = {1: out_prefix + "_database.bin", 2: out_prefix + "_train.bin"}
    counts = {1: 0, 2: 0}
    q_ids, q_vecs = [], []
    files = {k: (open(p + ".vecs.bin", "wb"), open(p + ".ids.bin", "wb"))
             for k, p in outs.items()}
    t0 = time.time()
    for lo in range(0, n_rows, chunk):
        hi = min(lo + chunk, n_rows)
        r = role[lo:hi]
        v = np.asarray(vecs[lo:hi])
        for k in (1, 2):
            m = r == k
            if m.any():
                v[m].tofile(files[k][0])
                ids[lo:hi][m].tofile(files[k][1])
                counts[k] += int(m.sum())
        m = r == 3
        if m.any():
            q_vecs.append(v[m].copy())
            q_ids.append(ids[lo:hi][m].copy())
        if (lo // chunk) % 50 == 0:
            print(f"  stage 3: {hi:,}/{n_rows:,} ({time.time() - t0:.0f}s)", flush=True)
    for a, b in files.values():
        a.close()
        b.close()
    for k, p in outs.items():
        _write_meta(p, counts[k])

    q_ids = np.concatenate(q_ids)
    q_vecs = np.vstack(q_vecs)
    with open(out_prefix + "_query.txt", "w") as fh:
        fh.write(f"{len(q_ids)} {DIM}\n")
        for i, row in zip(q_ids, np.char.mod("%.6g", q_vecs)):
            fh.write(f"{i} {' '.join(row)}\n")
    print(f"  database: {counts[1]:,}  train: {counts[2]:,}  query: {len(q_ids):,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("smi_glob", help="ZINC .smi file or glob")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--train-size", type=int, default=5_000_000)
    p.add_argument("--query-size", type=int, default=5_000)
    p.add_argument("--dim", type=int, default=256, help="fingerprint dimension")
    p.add_argument("--standardize", action="store_true",
                   help="salt-strip + charge-normalize before fingerprinting")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch", type=int, default=50_000,
                   help="molecules in flight per batch (bounds memory)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many kept rows (for testing)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=int, default=1_000_000,
                   help="rows per read in the final write stage")
    p.add_argument("--reuse-temp", action="store_true",
                   help="skip fingerprinting; reuse <out>.tmp_* from an earlier run")
    p.add_argument("--keep-temp", action="store_true")
    a = p.parse_args()

    t0 = time.time()
    tmp = a.out_prefix + ".tmp"
    if a.reuse_temp:
        n_rows = os.path.getsize(tmp + ".hash") // 8
    else:
        n_rows = stage1_fingerprint(a.smi_glob, tmp, a.standardize, a.workers,
                                    a.batch, a.limit)
    role = stage2_select(tmp, n_rows, a.train_size, a.query_size, a.seed)
    stage3_write(tmp, a.out_prefix, role, a.chunk)
    if not a.keep_temp:
        for ext in (".vecs", ".ids", ".hash"):
            os.remove(tmp + ext)
    print(f"done in {time.time() - t0:.0f}s")
