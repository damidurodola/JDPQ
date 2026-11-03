#!/usr/bin/env python3
"""
Jaccard (== Tanimoto) molecular similarity over SureChEMBL 2.0 bulk data.

For BINARY fingerprints, the Jaccard index and the Tanimoto coefficient are the
same number:  |A & B| / |A | B|.  RDKit computes it directly.

Two modes:
  query    one query SMILES  ->  ranked most-similar compounds in the file
  pairwise all-vs-all within the loaded subset  ->  pairs above a threshold

Pass --standardize to salt-strip + charge-normalize every molecule (query and
dataset) before fingerprinting, so equivalent depictions collapse to one form.

Input: any parquet with an `id` column and a `smiles` column
(that's the schema of SureChEMBL's compounds.parquet).
"""
import os
import argparse
import time
import pyarrow.parquet as pq
import numpy as np
import glob
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse spam

# ---- fingerprint config (ECFP4-style Morgan; radius 2, FP_DIM bits, default 256) ----
# FP_DIM env var overrides the dimension (set before import; worker processes inherit it)
FP_DIM = int(os.environ.get("FP_DIM", 256))
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_DIM, includeChirality=True)

# ---- standardization objects (built once, reused for every molecule) --------
_METAL = rdMolStandardize.MetalDisconnector()
_NORMALIZER = rdMolStandardize.Normalizer()
_LARGEST = rdMolStandardize.LargestFragmentChooser()  # drops salts / solvents
_UNCHARGER = rdMolStandardize.Uncharger()             # neutralizes charges


def standardize_mol(mol):
    """Salt strip + normalize functional groups + neutralize charges.

    Order: disconnect metals -> normalize -> keep largest organic fragment
    (this is the salt/solvent strip) -> uncharge -> sanitize.
    Returns a cleaned mol, or None if anything fails.
    """
    try:
        mol = _METAL.Disconnect(mol)
        mol = _NORMALIZER.normalize(mol)
        mol = _LARGEST.choose(mol)      # <- salt stripping
        mol = _UNCHARGER.uncharge(mol)  # <- charge normalization
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def fp_from_smiles(smi, standardize=True, normalize=True):
    """SMILES -> Morgan bit-vector fingerprint, or None if unparseable."""
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if standardize:
        mol = standardize_mol(mol)
        if mol is None:
            return None

    cfp = _GEN.GetCountFingerprint(mol)          # counts, not bits
    arr = np.zeros(FP_DIM, dtype=np.float32)
    for idx, cnt in cfp.GetNonzeroElements().items():
        arr[idx] = cnt
    if normalize and arr.max() > 0:
        arr /= arr.max()                          # -> floats in [0,1]
    return arr


def load_fingerprints(path, limit=None, standardize=True,
                      smiles_col="smiles", id_col="id"):
    """Read parquet, keep only rows whose SMILES parse. Returns (ids, fps)."""
    table = pq.read_table(path, columns=[id_col, smiles_col])
    if limit is not None:
        table = table.slice(0, limit)
    ids = table.column(id_col).to_pylist()
    smis = table.column(smiles_col).to_pylist()

    good_ids, fps, bad = [], [], 0
    for cid, smi in zip(ids, smis):
        fp = fp_from_smiles(smi, standardize=standardize)
        if fp is None:
            bad += 1
            continue
        good_ids.append(cid)
        fps.append(fp)
    tag = " (standardized)" if standardize else ""
    print(f"  loaded {len(fps):,} usable fingerprints{tag} "
          f"({bad:,} SMILES failed)")
    return good_ids, fps

def load_fingerprints_zinco(path, limit=None, standardize=True):
    """Read ZINC .smi tranche file(s) -> (ids, fingerprint_matrix).
    path_glob can be one file or a wildcard like 'ZINC20_2D/**/*.smi'."""
    good_ids, fps, bad = [], [], 0
    for path in sorted(glob.glob(path, recursive=True)):
        with open(path) as fh:
            for n, line in enumerate(fh):
                parts = line.split()
                if len(parts) < 2:
                    continue
                if n == 0 and parts[0].lower() == "smiles":
                    continue                      # skip ZINC header
                smi, cid = parts[0], parts[1]
                fp = fp_from_smiles(smi, standardize=standardize)  # unchanged
                if fp is None:
                    bad += 1
                    continue
                good_ids.append(cid)
                fps.append(fp)
                if limit and len(fps) >= limit:
                    tag = " (standardized)" if standardize else ""
                    print(f"  loaded {len(fps):,}{tag}, {bad:,} failed")
                    return good_ids, np.vstack(fps)
    print(f"  loaded {len(good_ids):,}, {bad:,} SMILES failed")
    return good_ids, np.vstack(fps)

def iter_smiles_zinco(path_glob):
    """Stream (smi, cid) tuples from ZINC .smi tranche file(s) matching
    path_glob, without loading any file fully into memory."""
    for path in sorted(glob.glob(path_glob, recursive=True)):
        with open(path) as fh:
            for n, line in enumerate(fh):
                parts = line.split()
                if len(parts) < 2:
                    continue
                if n == 0 and parts[0].lower() == "smiles":
                    continue                      # skip ZINC header
                yield parts[0], parts[1]


def _fp_worker(args):
    smi, cid, standardize = args
    return cid, fp_from_smiles(smi, standardize=standardize)


def stream_fingerprints_zinco(path_glob, limit=None, standardize=True,
                               n_workers=1, chunksize=256):
    """Yield (cid, fingerprint) for each parseable molecule matched by
    path_glob, one at a time, so the caller never has to hold the whole
    dataset in memory. n_workers=1 fingerprints serially in this process;
    n_workers>1 fans the work out to a multiprocessing pool (results are
    still yielded in file order, via Pool.imap)."""
    tasks = ((smi, cid, standardize) for smi, cid in iter_smiles_zinco(path_glob))
    n_good = n_bad = 0

    if n_workers and n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            for cid, fp in pool.imap(_fp_worker, tasks, chunksize=chunksize):
                if fp is None:
                    n_bad += 1
                    continue
                n_good += 1
                yield cid, fp
                if limit and n_good >= limit:
                    break
    else:
        for smi, cid, std in tasks:
            fp = fp_from_smiles(smi, standardize=std)
            if fp is None:
                n_bad += 1
                continue
            n_good += 1
            yield cid, fp
            if limit and n_good >= limit:
                break

    tag = " (standardized)" if standardize else ""
    print(f"  streamed {n_good:,} usable fingerprints{tag}, {n_bad:,} SMILES failed")


def ruzicka(a, mat):
    """Weighted Jaccard of vector a against each row of 2-D mat -> [0,1]."""
    mn = np.minimum(mat, a).sum(axis=1)
    mx = np.maximum(mat, a).sum(axis=1)
    return np.divide(mn, mx, out=np.zeros_like(mn), where=mx > 0)

def jprod(a, mat):
    """Jprod similarity of vector a against each row of 2-D mat."""
    num = np.dot(mat, a)
    den = np.sum(mat, axis=1) + np.sum(a) - num
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

def tanimoto(a, mat):
    """Tanimoto similarity of vector a against each row of 2-D mat."""
    num = np.dot(mat, a)
    den = np.sum(mat**2, axis=1) + np.sum(a**2, axis=0) - num
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

def query(path, query_smiles, top_n=20, threshold=0.0, limit=None,
          standardize=True):
    q = fp_from_smiles(query_smiles, standardize=standardize)
    if q is None:
        raise SystemExit(f"Query SMILES could not be parsed: {query_smiles!r}")
    ids, fps = load_fingerprints(path, limit=limit, standardize=standardize)
    # sims = ruzicka(q, fps)  # == Jaccard for bit vecs
    sims = jprod(q, fps)  # == Jaccard for bit vecs
    # sims = tanimoto(q, fps)  # == Tanimoto for bit vecs
    ranked = sorted(zip(ids, sims), key=lambda x: x[1], reverse=True)
    ranked = [(i, s) for i, s in ranked if s >= threshold][:top_n]
    print(f"\nTop {len(ranked)} matches for {query_smiles!r}:")
    print(f"{'compound_id':>14}  {'jprod':>8}")
    for cid, s in ranked:
        print(f"{cid:>14}  {s:8.4f}")
    return ranked


def pairwise(path, threshold=0.4, limit=None, out=None, standardize=True):
    ids, fps = load_fingerprints(path, limit=limit, standardize=standardize)
    n = len(fps)
    if n > 50_000:
        print(f"  WARNING: {n:,} compounds -> ~{n*(n-1)//2:,} pairs. "
              "This is O(n^2); consider a smaller subset or FPSim2.")
    pairs = []
    for i in range(n - 1):
        # sims = ruzicka(fps[i], fps[i + 1:])
        sims = jprod(fps[i], fps[i + 1:])
        # sims = tanimoto(fps[i], fps[i + 1:])

        for off, s in enumerate(sims):
            if s >= threshold:
                pairs.append((ids[i], ids[i + 1 + off], s))
    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n{len(pairs):,} pairs with Jprod >= {threshold}")
    if out:
        import csv
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["compound_id_a", "compound_id_b", "jprod"])
            w.writerows(pairs)
        print(f"  written -> {out}")
    else:
        for a, b, s in pairs[:20]:
            print(f"  {a} - {b} : {s:.4f}")
    return pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    pq_ = sub.add_parser("query", help="one SMILES vs the file")
    pq_.add_argument("parquet")
    pq_.add_argument("smiles")
    pq_.add_argument("--top-n", type=int, default=20)
    pq_.add_argument("--threshold", type=float, default=0.0)
    pq_.add_argument("--limit", type=int, default=None)
    pq_.add_argument("--standardize", action="store_true",
                     help="salt-strip + charge-normalize before fingerprinting")

    pp_ = sub.add_parser("pairwise", help="all-vs-all within the subset")
    pp_.add_argument("parquet")
    pp_.add_argument("--threshold", type=float, default=0.7)
    pp_.add_argument("--limit", type=int, default=None)
    pp_.add_argument("--out", default=None, help="CSV output path")
    pp_.add_argument("--standardize", action="store_true",
                     help="salt-strip + charge-normalize before fingerprinting")

    a = p.parse_args()
    t0 = time.time()
    if a.mode == "query":
        query(a.parquet, a.smiles, a.top_n, a.threshold, a.limit, a.standardize)
    else:
        pairwise(a.parquet, a.threshold, a.limit, a.out, a.standardize)
    print(f"\ndone in {time.time()-t0:.2f}s")
