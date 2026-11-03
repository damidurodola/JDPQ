#!/usr/bin/env python3
"""Convert a (query file, database file) pair of dense text vectors into the
canonical on-disk format used by every other part of this baseline.

Input format (both files):
    line 1:      "<N> <d>"
    lines 2..:   "<id><tab or space><v_1> <v_2> ... <v_d>"  (dense, nonnegative floats)

Output (in <out>/<name>/):
    db.npz, query.npz, train.npz   SciPy CSR, uncompressed (scipy.sparse.save_npz);
                                   data float64, indices int32, indptr int64,
                                   column indices sorted, explicit zeros dropped.
    {db,query,train}_ids.txt       original string id of every row, in row order
    manifest.json                  provenance, cleaning counts, seed, statistics

--fast replaces the per-row Python parser and hash loop with a parallel, vectorized path
(`--workers` processes parse byte ranges of the file; duplicate detection hashes all rows at
once and verifies every hash match exactly).  It produces byte-identical output; the only
difference is that `--no-sha256` can skip hashing the (possibly tens of GB) source files.

Binary database (--db-bin): instead of --db a raw float32 file (N x D row-major, no header, as written
by the zinc split script) can be read, with its `.meta.txt` ("N D", --db-meta, or give --dim) and
optional `.ids.bin` (N x int32, --db-ids-bin; without it ids are the row numbers).  The file is
memory-mapped and parsed in chunks by --workers processes; zeros are dropped to make the sparse rows.
The ids are only recorded in db_ids.txt: nothing downstream uses them (rows are addressed by index).

Cleaning, in this order (decided with the user, see README):
    1. zero vectors (L1 norm 0) are dropped from every file and counted;
    2. exact-duplicate queries are dropped (first occurrence kept); then, with --n-queries N, N of the
       remaining queries are drawn with default_rng([seed, 1]) (database rows are then compared only
       against the queries actually used);
    3. exact-duplicate database vectors are dropped (first occurrence kept);
    4. database vectors identical to any query are dropped;
    5. with `--train-bin`, the externally supplied train split (raw float32, row-major,
       d columns; optional uint32 ids in `--train-ids-bin`; only its first `--train-n` rows if given) is used as the train split and
       database rows equal to a train row are dropped (counted).  Otherwise `--heldout` rows are drawn uniformly (numpy default_rng(seed)) from the
       cleaned database, removed from it, and written as the "train" split.
       The train split is only used for tuning and for the Step 0 checks.
"Identical" means bitwise-equal float64 values after parsing (the same string
always parses to the same double).  Values are never rescaled or normalized.
"""
import argparse, hashlib, json, multiprocessing as mp, os, sys, time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp


def file_sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_dense_text(path):
    """Stream the file; return (ids, indptr, indices, data, d, n_header)."""
    ids, indptr, idx_chunks, val_chunks = [], [0], [], []
    with open(path, 'r') as f:
        header = f.readline().split()
        if len(header) != 2:
            sys.exit(f'{path}: bad header {header!r}')
        n_header, d = int(header[0]), int(header[1])
        for lineno, line in enumerate(f, start=2):
            parts = line.split(None, 1)  # id and values may be separated by a tab or a space
            if len(parts) != 2:
                sys.exit(f'{path}:{lineno}: no values after id')
            head, rest = parts
            v = np.fromstring(rest, sep=' ')
            if v.size != d:
                sys.exit(f'{path}:{lineno}: {v.size} values, expected {d}')
            if not np.all(np.isfinite(v)):
                sys.exit(f'{path}:{lineno}: non-finite value')
            if np.any(v < 0):
                sys.exit(f'{path}:{lineno}: negative value (J_w needs nonnegative vectors)')
            nz = np.flatnonzero(v)
            ids.append(head.strip())
            idx_chunks.append(nz.astype(np.int32))
            val_chunks.append(v[nz])
            indptr.append(indptr[-1] + nz.size)
    if len(ids) != n_header:
        sys.exit(f'{path}: header says N={n_header} but found {len(ids)} rows')
    indices = np.concatenate(idx_chunks) if idx_chunks else np.zeros(0, np.int32)
    data = np.concatenate(val_chunks) if val_chunks else np.zeros(0)
    return ids, np.asarray(indptr, np.int64), indices, data, d, n_header


# ----------------------------------------------------------------- fast path (--fast)
def _parse_slow(lines, d, where):
    """Per-line parser (same checks as read_dense_text) used if a chunk is not plain numeric."""
    ids, idx, val, counts = [], [], [], []
    for n, line in enumerate(lines):
        parts = line.split(None, 1)
        if len(parts) != 2:
            sys.exit(f'{where}: line {n}: no values after id')
        v = np.fromstring(parts[1], sep=' ')
        if v.size != d:
            sys.exit(f'{where}: line {n}: {v.size} values, expected {d}')
        ids.append(parts[0].decode())
        nz = np.flatnonzero(v)
        idx.append(nz.astype(np.int32)); val.append(v[nz]); counts.append(nz.size)
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            sys.exit(f'{where}: line {n}: non-finite or negative value')
    return (ids, np.concatenate(idx) if idx else np.zeros(0, np.int32),
            np.concatenate(val) if val else np.zeros(0), np.asarray(counts, np.int64))


def _parse_chunk(args):
    path, lo, hi, d = args
    with open(path, 'rb') as f:
        f.seek(lo)
        raw = f.read(hi - lo)
    lines = raw.split(b'\n')
    if lines and lines[-1] == b'':
        lines.pop()
    where = f'{path}[{lo}:{hi}]'
    if not lines:
        return [], np.zeros(0, np.int32), np.zeros(0), np.zeros(0, np.int64)
    try:
        arr = np.fromstring(raw, sep=' ')  # tabs/newlines count as separators
        if arr.size != len(lines) * (d + 1):
            raise ValueError('shape')
        arr = arr.reshape(len(lines), d + 1)
        ids = [ln.split(None, 1)[0].decode() for ln in lines]
        if not np.array_equal(arr[:, 0], np.array([float(i) for i in ids])):
            raise ValueError('id misaligned')  # a row had the wrong number of values
        vals = arr[:, 1:]
        if not np.all(np.isfinite(vals)) or np.any(vals < 0):
            sys.exit(f'{where}: non-finite or negative value')
        rows, cols = np.nonzero(vals)
        return (ids, cols.astype(np.int32), np.ascontiguousarray(vals[rows, cols]),
                np.bincount(rows, minlength=len(lines)).astype(np.int64))
    except ValueError:
        return _parse_slow(lines, d, where)  # non-numeric ids or malformed rows: exact per-line checks


def read_dense_text_fast(path, workers):
    with open(path, 'rb') as f:
        header = f.readline().split()
        if len(header) != 2:
            sys.exit(f'{path}: bad header {header!r}')
        n_header, d = int(header[0]), int(header[1])
        start = f.tell()
        size = os.path.getsize(path)
        n_chunks = max(workers * 4, (size - start) // (64 << 20) + 1)
        pts = [start + (size - start) * i // n_chunks for i in range(n_chunks + 1)]
        for i in range(1, n_chunks):
            f.seek(pts[i] - 1)
            f.readline()  # advance to the next line start (stays put if already at one)
            pts[i] = f.tell()
    jobs = [(path, pts[i], pts[i + 1], d) for i in range(n_chunks) if pts[i + 1] > pts[i]]
    ids, idx_chunks, val_chunks, counts = [], [], [], []
    with mp.Pool(workers) as pool:
        for i_ids, i_idx, i_val, i_cnt in pool.imap(_parse_chunk, jobs):  # ordered: keeps file order
            ids += i_ids; idx_chunks.append(i_idx); val_chunks.append(i_val); counts.append(i_cnt)
    if len(ids) != n_header:
        sys.exit(f'{path}: header says N={n_header} but found {len(ids)} rows')
    indptr = np.zeros(len(ids) + 1, np.int64)
    np.cumsum(np.concatenate(counts), out=indptr[1:])
    return ids, indptr, np.concatenate(idx_chunks), np.concatenate(val_chunks), d, n_header


def _parse_bin_chunk(args):
    path, lo, hi, d = args
    mm = np.memmap(path, dtype=np.float32, mode='r', shape=(hi - lo, d), offset=lo * d * 4)
    vals = np.asarray(mm, dtype=np.float64)  # float32 -> float64 is exact
    if not np.all(np.isfinite(vals)) or np.any(vals < 0):
        sys.exit(f'{path}[rows {lo}:{hi}]: non-finite or negative value')
    rows, cols = np.nonzero(vals)
    return (cols.astype(np.int32), np.ascontiguousarray(vals[rows, cols]),
            np.bincount(rows, minlength=hi - lo).astype(np.int64))


def read_bin_vectors(path, meta, dim, ids_path, workers, chunk=250_000):
    """Raw float32 N x D file (+ optional int32 ids) -> same tuple as read_dense_text."""
    if meta:
        n_meta, d = map(int, open(meta).read().split())
        if dim and dim != d:
            sys.exit(f'{meta} says D={d} but --dim {dim}')
    elif dim:
        n_meta, d = None, dim
    else:
        sys.exit('--db-bin needs --db-meta or --dim')
    size = os.path.getsize(path)
    if size % (4 * d):
        sys.exit(f'{path}: size {size} is not a multiple of 4*D = {4 * d}')
    n = size // (4 * d)
    if n_meta is not None and n != n_meta:
        sys.exit(f'{path}: holds {n} rows but {meta} says N={n_meta}')
    if ids_path:
        raw_ids = np.fromfile(ids_path, dtype=np.int32)
        if raw_ids.size != n:
            sys.exit(f'{ids_path}: {raw_ids.size} ids for {n} rows')
        ids = [str(i) for i in raw_ids]
    else:
        ids = [str(i) for i in range(n)]
    jobs = [(path, lo, min(lo + chunk, n), d) for lo in range(0, n, chunk)]
    idx_chunks, val_chunks, counts = [], [], []
    with mp.Pool(workers) as pool:
        for i_idx, i_val, i_cnt in pool.imap(_parse_bin_chunk, jobs):  # ordered
            idx_chunks.append(i_idx); val_chunks.append(i_val); counts.append(i_cnt)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.concatenate(counts), out=indptr[1:])
    return ids, indptr, np.concatenate(idx_chunks), np.concatenate(val_chunks), d, n


def row_hashes(X):
    """64-bit hash of every row (all rows must be non-empty); only a filter, matches are verified exactly."""
    with np.errstate(over='ignore'):
        z = (X.indices.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ np.ascontiguousarray(X.data).view(np.uint64)
        z ^= z >> np.uint64(30); z *= np.uint64(0xBF58476D1CE4E5B9)
        z ^= z >> np.uint64(27); z *= np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
        h = np.add.reduceat(z, X.indptr[:-1])
        return h ^ (np.diff(X.indptr).astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93))


def dedup_fast(X, against=None):
    """Same result as dedup(): rows equal to a row of `against` are dropped (n_overlap), then rows equal
    to an earlier kept row of X (n_dup)."""
    keep = np.ones(X.shape[0], bool)
    n_dup = n_overlap = 0
    hX = row_hashes(X)
    if against is not None and against.shape[0]:
        hA = row_hashes(against)
        cand = np.flatnonzero(np.isin(hX, hA))
        amap = defaultdict(list)
        for j in np.flatnonzero(np.isin(hA, hX[cand])):
            amap[hA[j]].append(j)
        for i in cand:
            if any(rows_equal(X, i, against, j) for j in amap.get(hX[i], ())):
                keep[i] = False
                n_overlap += 1
    live = np.flatnonzero(keep)
    order = live[np.argsort(hX[live], kind='stable')]  # equal hashes stay in row order
    hs = hX[order]
    starts = np.r_[0, np.flatnonzero(hs[1:] != hs[:-1]) + 1]
    ends = np.r_[starts[1:], len(hs)]
    for s0, e0 in zip(starts[ends - starts > 1], ends[ends - starts > 1]):
        distinct = []
        for i in order[s0:e0]:
            if any(rows_equal(X, i, X, j) for j in distinct):
                keep[i] = False
                n_dup += 1
            else:
                distinct.append(i)
    return keep, n_dup, n_overlap


def row_key(indices, data, i, indptr):
    a, b = indptr[i], indptr[i + 1]
    return hashlib.blake2b(indices[a:b].tobytes() + data[a:b].tobytes(), digest_size=16).digest()


def rows_equal(X, i, Y, j):
    a, b = X.indptr[i], X.indptr[i + 1]
    c, e = Y.indptr[j], Y.indptr[j + 1]
    return (b - a == e - c and np.array_equal(X.indices[a:b], Y.indices[c:e])
            and np.array_equal(X.data[a:b], Y.data[c:e]))


def dedup(X, against=None):
    """Return mask of rows of X to keep: drop exact duplicates within X (keep
    first) and, if `against` is given, rows equal to any row of `against`."""
    keep = np.ones(X.shape[0], bool)
    seen = {}
    n_dup = n_overlap = 0
    if against is not None:
        ref = {}
        for j in range(against.shape[0]):
            ref.setdefault(row_key(against.indices, against.data, j, against.indptr), []).append(j)
    for i in range(X.shape[0]):
        key = row_key(X.indices, X.data, i, X.indptr)
        if against is not None and any(rows_equal(X, i, against, j) for j in ref.get(key, ())):
            keep[i] = False
            n_overlap += 1
            continue
        prev = seen.get(key)
        if prev is not None and any(rows_equal(X, i, X, p) for p in prev):
            keep[i] = False
            n_dup += 1
            continue
        seen.setdefault(key, []).append(i)
    return keep, n_dup, n_overlap


def stats(X):
    nnz = np.diff(X.indptr)
    l1 = np.asarray(X.sum(axis=1)).ravel()
    return {
        'N': int(X.shape[0]), 'd': int(X.shape[1]),
        'nnz_mean': float(nnz.mean()), 'nnz_median': float(np.median(nnz)), 'nnz_max': int(nnz.max()),
        'l1_min': float(l1.min()), 'l1_median': float(np.median(l1)), 'l1_max': float(l1.max()),
        'min_nonzero_weight': float(X.data.min()), 'max_weight': float(X.data.max()),
    }


def save(X, ids, out_dir, split):
    X = sp.csr_matrix((X.data.astype(np.float64), X.indices.astype(np.int32), X.indptr.astype(np.int64)),
                      shape=X.shape)
    assert X.has_sorted_indices and X.nnz == np.count_nonzero(X.data)
    sp.save_npz(os.path.join(out_dir, f'{split}.npz'), X, compressed=False)
    with open(os.path.join(out_dir, f'{split}_ids.txt'), 'w') as f:
        f.write('\n'.join(ids) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='dataset name (output subdirectory)')
    ap.add_argument('--query', required=True)
    ap.add_argument('--db', help='text database file (or use --db-bin)')
    ap.add_argument('--db-bin', help='raw float32 N x D database file instead of --db')
    ap.add_argument('--db-meta', help='.meta.txt ( "N D" ) of --db-bin')
    ap.add_argument('--db-ids-bin', help='raw int32 ids of --db-bin rows (optional)')
    ap.add_argument('--dim', type=int, default=0, help='D for --db-bin when there is no --db-meta')
    ap.add_argument('--out', required=True, help='root data directory')
    ap.add_argument('--heldout', type=int, default=500)
    ap.add_argument('--train-bin', help='raw float32 N x d train split (already removed from the db file)')
    ap.add_argument('--train-n', type=int, default=1000,
                    help='with --train-bin: keep only the first N train vectors (0 = all); limits the cost of train ground truth')
    ap.add_argument('--train-ids-bin', help='raw uint32 ids for --train-bin rows')
    ap.add_argument('--fast', action='store_true', help='parallel vectorized parsing and duplicate detection')
    ap.add_argument('--workers', type=int, default=os.cpu_count(), help='processes for --fast')
    ap.add_argument('--no-sha256', action='store_true', help='do not hash the source files (slow for huge files)')
    ap.add_argument('--n-queries', type=int, default=3000,
                    help='use only this many queries: a seeded random subset of the cleaned queries, kept in file order')
    ap.add_argument('--seed', type=int, default=20261014)
    args = ap.parse_args()

    if bool(args.db) == bool(args.db_bin):
        sys.exit('give exactly one of --db and --db-bin')
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    clean = {}
    mats = {}
    for split, path in (('query', args.query), ('db', args.db or args.db_bin)):
        if split == 'db' and args.db_bin:
            ids, indptr, indices, data, d, n_header = read_bin_vectors(
                path, args.db_meta, args.dim, args.db_ids_bin, args.workers)
        else:
            ids, indptr, indices, data, d, n_header = (read_dense_text_fast(path, args.workers) if args.fast
                                                       else read_dense_text(path))
        X = sp.csr_matrix((data, indices, indptr), shape=(len(ids), d))
        nonempty = np.diff(X.indptr) > 0
        clean[f'{split}_rows_in_file'] = n_header
        clean[f'{split}_zero_vectors_dropped'] = int((~nonempty).sum())
        X = X[nonempty]
        ids = [s for s, k in zip(ids, nonempty) if k]
        mats[split] = (X, ids, d)
        print(f'[{time.time() - t0:7.1f}s] read {path}: {n_header} rows, d={d}', flush=True)
    if mats['query'][2] != mats['db'][2]:
        sys.exit('query and db dimensions differ')

    Q, qids, _ = mats['query']
    dedup_fn = dedup_fast if args.fast else dedup
    keep, n_dup, _ = dedup_fn(Q)
    clean['query_duplicates_dropped'] = n_dup
    Q, qids = Q[keep], [s for s, k in zip(qids, keep) if k]
    clean['queries_available_after_cleaning'] = int(Q.shape[0])
    if args.n_queries:
        if Q.shape[0] < args.n_queries:
            print(f'WARNING: only {Q.shape[0]} queries available, --n-queries {args.n_queries} ignored', flush=True)
        else:
            pick = np.sort(np.random.default_rng([args.seed, 1]).choice(Q.shape[0], size=args.n_queries, replace=False))
            Q, qids = Q[pick], [qids[i] for i in pick]
    clean['queries_used'] = int(Q.shape[0])

    X, xids, _ = mats['db']
    keep, n_dup, n_overlap = dedup_fn(X, against=Q)
    clean['db_duplicates_dropped'] = n_dup
    clean['db_rows_equal_to_a_query_dropped'] = n_overlap
    X, xids = X[keep], [s for s, k in zip(xids, keep) if k]
    print(f'[{time.time() - t0:7.1f}s] dedup done: {clean}', flush=True)

    if args.train_bin:
        d = X.shape[1]
        n_file = os.path.getsize(args.train_bin) // (4 * d)
        n_keep = min(args.train_n, n_file) if args.train_n else n_file
        clean['train_rows_in_file'] = int(n_file)
        TV = np.fromfile(args.train_bin, dtype=np.float32, count=n_keep * d).reshape(-1, d).astype(np.float64)
        tids = ([str(i) for i in np.fromfile(args.train_ids_bin, dtype=np.uint32, count=n_keep)] if args.train_ids_bin
                else [f'train{i}' for i in range(TV.shape[0])])
        T = sp.csr_matrix(TV)
        T.sort_indices()
        nonempty = np.diff(T.indptr) > 0
        clean['train_zero_vectors_dropped'] = int((~nonempty).sum())
        T, tids = T[nonempty], [s for s, k in zip(tids, nonempty) if k]
        keep, _, n_overlap = dedup_fn(X, against=T)
        clean['db_rows_equal_to_a_train_row_dropped'] = n_overlap
        X, xids = X[keep], [s for s, k in zip(xids, keep) if k]
    else:
        rng = np.random.default_rng(args.seed)
        held = np.sort(rng.choice(X.shape[0], size=args.heldout, replace=False))
        mask = np.ones(X.shape[0], bool)
        mask[held] = False
        T, tids = X[held], [xids[i] for i in held]
        X, xids = X[mask], [s for s, k in zip(xids, mask) if k]

    # Duplicate string ids that survive (distinct vectors sharing an id) are
    # reported but kept: rows are addressed by index everywhere.
    clean['db_duplicate_string_ids_remaining'] = len(xids) - len(set(xids))
    clean['query_ids_also_in_db'] = len(set(qids) & set(xids))

    for split, M, ids in (('db', X, xids), ('query', Q, qids), ('train', T, tids)):
        save(M, ids, out_dir, split)

    manifest = {
        'name': args.name,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'converter_args': vars(args),
        'sources': {s: {'path': os.path.abspath(p), 'bytes': os.path.getsize(p), 'sha256': None if args.no_sha256 else file_sha256(p)}
                    for s, p in [('query', args.query), ('db', args.db or args.db_bin)] + ([('train', args.train_bin)] if args.train_bin else [])},
        'cleaning': clean,
        'stats': {'db': stats(X), 'query': stats(Q), 'train': stats(T)},
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({'cleaning': clean, 'stats': manifest['stats']}, indent=2))
    print(f'[{time.time() - t0:7.1f}s] wrote {out_dir}')


if __name__ == '__main__':
    main()
