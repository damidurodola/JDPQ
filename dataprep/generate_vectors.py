#!/usr/bin/env python3
"""
Generate Morgan fingerprint vectors from ZINC-style .smi tranche file(s)
and split them into:

  - query vectors    : a random sample of --query-size compounds
  - database vectors  : everything else

Streams the input instead of loading it into memory: fingerprints are
computed one at a time (optionally across a multiprocessing pool) and
written straight to disk. The query set is chosen with reservoir sampling
(exact uniform sample of size --query-size, single pass, O(query_size)
memory), so this scales to corpora far larger than RAM.

Reuses fingerprint generation from surechembl.py so both files stay
consistent (same fingerprint config / normalization).

Output: two whitespace-delimited .txt files, each with a header line
'num_vectors dim' followed by one 'id v1 v2 ... vdim' row per compound
(id is the ZINC id from the input .smi file).
"""
import argparse
import hashlib
import os
import shutil
import time
import numpy as np

# --dim must reach surechembl before it is imported (it builds its fingerprint
# generator at import time), so it is read from argv here and passed via FP_DIM.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--dim", type=int, default=256)
os.environ["FP_DIM"] = str(_pre.parse_known_args()[0].dim)

from surechembl import stream_fingerprints_zinco, iter_smiles_zinco, _fp_worker

DIM = int(os.environ["FP_DIM"])


def _write_rows(fh, items):
    """Append (zinc_id, vector) pairs as 'zinc_id v1 ... vdim' rows."""
    if not items:
        return
    vecs = np.vstack([fp for _cid, fp in items])
    for (cid, _fp), row in zip(items, np.char.mod("%.6g", vecs)):
        fh.write(f"{cid} {' '.join(row)}\n")


def _read_ids(path):
    with open(path) as fh:
        next(fh)
        return [line.split(" ", 1)[0] for line in fh]


def save_for_ids(path, out_prefix, ids_from, standardize=True, n_workers=1,
                 write_chunk=5000, batch=20000):
    """Fingerprint exactly the compounds listed in <ids_from>_database.txt /
    <ids_from>_query.txt (numeric ids), splitting them the same way. The first
    parseable compound seen for each ZINC number is used, matching --numeric-ids.

    Only wanted ids are fingerprinted, and work is submitted in bounded
    batches so memory stays flat (Pool.imap over a generator would queue the
    whole input)."""
    q_ids = _read_ids(f"{ids_from}_query.txt")
    db_ids = _read_ids(f"{ids_from}_database.txt")
    q_set, db_set = set(q_ids), set(db_ids)
    remaining = len(q_set) + len(db_set)
    found_q, taken, buffer, db_count = {}, set(), [], 0

    db_path = f"{out_prefix}_database.txt"
    pool = None
    if n_workers and n_workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(n_workers)

    def run(tasks):
        if pool is not None:
            return pool.map(_fp_worker, tasks, chunksize=max(1, len(tasks) // (4 * n_workers)))
        return [_fp_worker(t) for t in tasks]

    def pending():
        """Yield batches of (smi, cid, standardize) for wanted, not-yet-taken numbers."""
        tasks, queued = [], set()
        for smi, cid in iter_smiles_zinco(path):
            num = str(int(cid[4:].split("_")[0]))
            if num in taken or (num not in q_set and num not in db_set):
                continue
            tasks.append((smi, cid, standardize))
            queued.add(num)
            if len(queued) == len(q_set) + len(db_set):
                break         # every wanted number seen: stop scanning the input
            if len(tasks) >= batch:
                yield tasks
                tasks = []
        if tasks:
            yield tasks

    t0, n_done = time.time(), 0
    with open(db_path + ".tmp", "w") as db_fh:
        for tasks in pending():
            for cid, fp in run(tasks):
                if fp is None:
                    continue
                num = str(int(cid[4:].split("_")[0]))
                if num in taken:
                    continue
                taken.add(num)
                remaining -= 1
                if num in q_set:
                    found_q[num] = (num, fp)
                else:
                    buffer.append((num, fp))
                if len(buffer) >= write_chunk:
                    _write_rows(db_fh, buffer)
                    db_count += len(buffer)
                    buffer.clear()
            n_done += len(tasks)
            print(f"  fingerprinted {n_done:,} ({time.time()-t0:.0f}s), "
                  f"{remaining:,} ids still missing", flush=True)
            if remaining == 0:
                break
        _write_rows(db_fh, buffer)
        db_count += len(buffer)
    if pool is not None:
        pool.terminate()

    with open(db_path, "w") as out_fh:
        out_fh.write(f"{db_count} {DIM}\n")
        with open(db_path + ".tmp") as in_fh:
            shutil.copyfileobj(in_fh, out_fh)
    os.remove(db_path + ".tmp")

    q_rows = [found_q[i] for i in q_ids if i in found_q]
    with open(f"{out_prefix}_query.txt", "w") as q_fh:
        q_fh.write(f"{len(q_rows)} {DIM}\n")
        _write_rows(q_fh, q_rows)
    print(f"  database: {db_count:,}/{len(db_ids):,}  query: {len(q_rows):,}/{len(q_ids):,}"
          f"  ({time.time()-t0:.0f}s)")
    if db_count != len(db_ids) or len(q_rows) != len(q_ids):
        print("  WARNING: some requested ids were not found / failed to fingerprint")


def split_and_save(path, out_prefix, query_size=3000, limit=None,
                    standardize=True, seed=42, n_workers=1, write_chunk=5000,
                    dedup=False, numeric_ids=False):
    rng = np.random.default_rng(seed)
    reservoir = []   # up to query_size (zinc_id, fingerprint) pairs (exact random sample)
    n_seen = 0

    db_path = f"{out_prefix}_database.txt"
    db_tmp_path = db_path + ".tmp"
    db_count = 0
    buffer = []
    seen_ids = set()      # numeric_ids: ZINC numbers already kept
    seen_hashes = set()   # dedup: 16-byte digests of vectors already kept

    t0 = time.time()
    with open(db_tmp_path, "w") as db_fh:
        # with dedup, --limit counts unique vectors, so it is applied here
        for cid, fp in stream_fingerprints_zinco(path,
                                                    limit=None if (dedup or numeric_ids) else limit,
                                                    standardize=standardize,
                                                    n_workers=n_workers):
            if numeric_ids:
                # ZINC000000141911_2 -> 141911; later variants of a number are dropped
                cid = int(cid[4:].split("_")[0])
                if cid in seen_ids:
                    continue
            if dedup:
                h = hashlib.blake2b(fp.tobytes(), digest_size=16).digest()
                if h in seen_hashes:
                    continue
                if limit is not None and n_seen >= limit:
                    break
                seen_hashes.add(h)
            elif numeric_ids and limit is not None and n_seen >= limit:
                break
            if numeric_ids:
                seen_ids.add(cid)
            n_seen += 1
            if len(reservoir) < query_size:
                reservoir.append((cid, fp))
                continue

            j = rng.integers(0, n_seen)
            if j < query_size:
                buffer.append(reservoir[j])
                reservoir[j] = (cid, fp)
            else:
                buffer.append((cid, fp))

            if len(buffer) >= write_chunk:
                _write_rows(db_fh, buffer)
                db_count += len(buffer)
                buffer.clear()

            if n_seen % 1_000_000 == 0:
                rate = n_seen / (time.time() - t0)
                print(f"  ...{n_seen:,} seen ({rate:,.0f}/s)")

        if buffer:
            _write_rows(db_fh, buffer)
            db_count += len(buffer)

    with open(db_path, "w") as out_fh:
        out_fh.write(f"{db_count} {DIM}\n")
        with open(db_tmp_path) as in_fh:
            shutil.copyfileobj(in_fh, out_fh)
    os.remove(db_tmp_path)

    if len(reservoir) < query_size:
        print(f"  WARNING: only {len(reservoir):,} valid compounds found, "
              f"fewer than --query-size {query_size:,}; database is empty")

    query_path = f"{out_prefix}_query.txt"
    with open(query_path, "w") as q_fh:
        q_fh.write(f"{len(reservoir)} {DIM}\n")
        _write_rows(q_fh, reservoir)

    print(f"  total seen : {n_seen:,} valid fingerprints")
    print(f"  query      : {len(reservoir):,} vectors -> {query_path}")
    print(f"  database   : {db_count:,} vectors -> {db_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("smi_glob", help="ZINC .smi file or glob, e.g. 'dir/*.smi'")
    p.add_argument("--out-prefix", default="vectors")
    p.add_argument("--query-size", type=int, default=3000)
    p.add_argument("--limit", type=int, default=None,
                    help="cap the number of valid fingerprints processed")
    p.add_argument("--standardize", action="store_true",
                    help="salt-strip + charge-normalize before fingerprinting")
    p.add_argument("--dedup", action="store_true",
                    help="keep only the first compound for each identical vector, "
                         "so database is unique and shares no vector with the "
                         "query set (--limit then counts unique vectors; keeps "
                         "a hash set in memory, ~16B+ per vector)")
    p.add_argument("--ids-from", metavar="PREFIX",
                    help="use exactly the ids in PREFIX_database.txt / PREFIX_query.txt "
                         "(numeric ids) instead of sampling; ignores --limit, "
                         "--query-size, --dedup, --numeric-ids")
    p.add_argument("--dim", type=int, default=256,
                    help="fingerprint dimension (Morgan fpSize)")
    p.add_argument("--numeric-ids", action="store_true",
                    help="write ids as plain ints (ZINC000000141911_1 -> 141911); "
                         "only the first compound per number is kept so ids are "
                         "unique (--limit then counts kept vectors)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1,
                    help="fingerprint in parallel across this many processes "
                         "(1 = serial; this is a shared machine, so raise "
                         "cautiously)")
    a = p.parse_args()

    t0 = time.time()
    if a.ids_from:
        save_for_ids(a.smi_glob, a.out_prefix, a.ids_from, a.standardize, a.workers)
        print(f"done in {time.time()-t0:.2f}s")
        raise SystemExit
    split_and_save(a.smi_glob, a.out_prefix, a.query_size, a.limit,
                   a.standardize, a.seed, a.workers, dedup=a.dedup,
                   numeric_ids=a.numeric_ids)
    print(f"done in {time.time()-t0:.2f}s")
