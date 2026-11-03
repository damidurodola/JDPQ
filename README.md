# Benchmark harness for Jaccard-distance product quantization (JD-PQ)

Benchmark scripts, evaluation harness, dataset-preparation code and baseline
wrappers accompanying the paper **Jaccard-Aware Product Quantization for Memory-Efficient Approximate Nearest Neighbor Search**.

> **Release status.** The JD-PQ implementation itself (k-medoids codebook
> learning under Jaccard distance and the asymmetric-distance scan) is **not
> included yet**. It will be released here upon acceptance. Everything else
> needed to prepare the data, compute exact ground truth, run the baselines and
> score any method is included and builds without it. See
> [What can be reproduced now](#what-can-be-reproduced-now).

## Contents

| Path | What |
|---|---|
| `encoder/` | Quadtree occupancy encoder: OpenStreetMap polygons -> 256-d vectors (Cemetery, Parks) |
| `dataprep/` | Molecular datasets: SMILES -> standardized Morgan count fingerprints -> disjoint train/database/query splits (ZINC20, SureChEMBL); random subsets |
| `convert_to_binary.cpp`, `extract_train.cpp`, `export_train.cpp`, `remove_train.cpp` | Text -> binary conversion; duplicate-ID removal; carving a training set out of a database |
| `src/` | Exact Jaccard distance, exact brute-force search (ground truth), data readers, recall metrics, polygon ground truth |
| `src/AnnIndex.h`, `src/benchmark.cpp` | Method-agnostic benchmark driver and the interface a method implements (see [Plugging in a method](#plugging-in-a-method)) |
| `faiss_pq_ablation.cpp` | Reference FAISS (Euclidean) PQ / IVFPQ baseline, scored against L2 and Jaccard ground truth |
| `python/hnsw.py` | HNSW baseline |
| `baselines/lsh/` | Weighted MinHash baseline (DartMinHash, FastICWS, BagMinHash2): sketching, retrieval, exact ground truth, job scripts. Has its own README; run `baselines/lsh/fetch_third_party.sh` first |
| `test_rerank.cpp` | Re-ranks shortlists from external baselines (HNSW, weighted MinHash) with exact distances |


## Datasets

All vectors are 256-dimensional, non-negative and sparse; distance is weighted Jaccard,
`1 - sum_i min(x_i,y_i) / sum_i max(x_i,y_i)`. Training, database and query sets are disjoint.

| Dataset | Source | Database | Training | Queries |
|---|---|---|---:|---:|
| Cemetery | OpenStreetMap polygons | 169,421 | 20,000 | 3,000 |
| Parks | OpenStreetMap polygons | 9,125,627 | 500,000 | 3,000 |
| SureChEMBL | patent compounds | 26,408,870 | 1,500,000 | 3,000 |
| ZINC20 | purchasable compounds | 959,039,988 (+ 10M, 100M, 500M random subsets) | 5,000,000 | 5,000 (first 3,000 used) |

**Encodings.**
*Polygons:* quadtree occupancy vector (node capacity 200 / 2,400 and maximum depth 6 / 8 for
Cemetery / Parks); about 63% of entries are zero.
*Molecules:* standardization (metal disconnection, functional-group normalization, largest
fragment, charge neutralization), then a Morgan count fingerprint (radius 2, chirality
included) folded to 256 dimensions; each count is divided by the molecule's largest count, so
values lie in [0, 1]. Unparseable molecules, duplicate identifiers and duplicate vectors are
removed; queries and training vectors are sampled uniformly at random from the unique vectors.

**Ground truth.** Exact brute-force Jaccard, top 100 neighbours. Recall@R (R = 1, 10, 50, 100)
is the fraction of the true top-R found in a method's top-R, averaged over queries, using a
shortlist of 1,000 candidates that is re-ranked with exact distances.

Raw sources: [ZINC20](https://zinc20.docking.org/), [SureChEMBL](https://www.surechembl.org/),
[OpenStreetMap](https://www.openstreetmap.org/) (ODbL). Check each source's licence before
redistributing derived data.

## Setup

```bash
conda env create -f environment.yml && conda activate jdpq-bench
make            # builds the data tools, test_rerank, test_rtree
make faiss_pq_ablation FAISS_PREFIX=$CONDA_PREFIX
```
The Makefile uses `-march=native`; build on the machine that runs the job (`NATIVE=0 make`
for a portable binary). C++17 with OpenMP is required.

## Preparing the data

Data file layout (text): first line `N D`, then one `id v1 ... vD` row per vector. Binary layout
(`<prefix>.meta.txt` = `N D`, `<prefix>.ids.bin` = N int32, `<prefix>.vecs.bin` = N x D float32,
row-major) is produced by `./convert_to_binary` and is memory-mapped by the harness, so datasets
larger than RAM work.

**Polygons (Cemetery, Parks).**
```bash
g++ -O3 -std=c++17 encoder/main.cpp -o encode $(pkg-config --cflags --libs geos)
./encode <polygons.csv> <vectors.txt>                       # quadtree encoder
python python/extract_polygons.py                            # polygon WKT for the exact polygon ground truth (edit paths at the top)
```
**Molecules (ZINC20, SureChEMBL).**
```bash
python dataprep/generate_split_binary.py 'ZINC20_2D/**/*.smi' --out-prefix zinc20_split \
    --standardize --train-size 5000000 --query-size 5000 --workers 16     # train/database/query, binary
python dataprep/extract_subset.py zinc20_split_database.bin zinc20_100M.bin --n 100000000   # random subset
```
SureChEMBL uses the same fingerprinting on `compounds.parquet` / the `.smi` export.

**Training set from a database** (used for Cemetery and Parks, where one file is split):
```bash
./extract_train db.txt train.ids 20000          # drops duplicate IDs in place, samples training IDs
./export_train  db.txt train.ids train_prefix    # saves those vectors (binary)
./remove_train  db.txt train.ids train_prefix   # removes them from the database (verified first)
```
`extract_train` and `remove_train` rewrite the database in place and are not crash-safe: keep a backup.

## What can be reproduced now

* Dataset construction and all dataset statistics above.
* Exact ground truth and the evaluation protocol (`src/benchmark.cpp`).
* The FAISS (Euclidean) PQ / IVFPQ baseline: `FAISS_DB=... FAISS_QUERY=... FAISS_TRAIN_PREFIX=... ./faiss_pq_ablation`.
* The weighted-MinHash baseline end to end (`baselines/lsh/`), and HNSW / MinHash shortlists re-ranked with exact distances through `test_rerank`.

**Not reproducible until release:** the JD-PQ and JD-IVFPQ rows (their raw per-configuration result
files are provided under `results/` **[to be added]** so tables and figures can be regenerated from them).

## Plugging in a method

`src/benchmark.cpp` knows nothing about how an index works. It loads the data, computes exact
ground truth, builds each configuration from a grid, searches, re-ranks, scores recall and writes
one CSV row per configuration. A method implements `AnnIndex` (`src/AnnIndex.h`: `train`, `add`,
`search`, `memory_usage_bytes`; `add` numbers vectors 0, 1, 2, ... and `search` returns those
positions) and is created by `make_jdpq_index(spec)` for kinds the harness does not know. Two
method slots feed the two column groups of the CSV ("IndexPQ ..." and "IVFPQ ..."):
`ANN_METHOD_A` (default `pq`) and `ANN_METHOD_B` (default `ivfpq`); an empty value disables a slot.
The exact reference method `flat` is always available, so the harness can be tried without any
plugin:

```bash
make benchmark
FAISS_DB=db.txt FAISS_QUERY=query.txt FAISS_TRAIN_PREFIX=train_prefix \
  ANN_METHOD_A=flat ANN_METHOD_B= PQ_GRID="4:8:256:16" ./benchmark   # Recall@10/50/100 ~ 1.0
```
(Recall@1 can fall a fraction of a percent below 1 where several vectors are exactly equidistant
from a query.) The released JD-PQ code is a second implementation of the same interface:
`src/jdpq_index.cpp` replaces `src/jdpq_stub.cpp` in the build.


