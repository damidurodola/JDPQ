# Weighted-MinHash ANN baseline (weighted Jaccard)

Baseline for ANN search under weighted Jaccard,
`J_w(x, y) = Σ min(x_i, y_i) / Σ max(x_i, y_i)`, built on the upstream
[dartminhash](https://github.com/tobc/dartminhash) sketchers (DartMinHash,
FastICWS, BagMinHash2). Everything here is sketching, retrieval, exact ground
truth and measurement; the PQ methods it is compared against live elsewhere.

All numbers in tables and figures are generated from the CSVs by scripts; none
are typed by hand. See `results/<dataset>/tables.md` and `results/summary.md`.

## Layout

```
third_party/dartminhash/   upstream, unmodified (git hash recorded in every CSV row)
third_party/xxhash/        xxHash v0.8.2, needed by BagMinHash2 (outside the upstream checkout)
wmh/                       all code written for this baseline
  scripts/convert.py       text -> canonical CSR .npz (+ cleaning, held-out split, statistics)
  scripts/dataset_table.py dataset statistics table
  scripts/step0_report.py  Step 0 estimator-accuracy summary
  scripts/sweep.py         full pipeline over every dataset in data/
  scripts/report.py        tables + figures from the result CSVs
  src/sketch.cpp           the ONLY translation unit that includes upstream headers
  src/sanity.cpp           Step 0 checks
  src/groundtruth.cpp      Step 1 exact ground truth
  src/search.cpp           Mode A / Mode B / rerank / exact reference, timing, memory
  include/*.hpp            npz reader, CSR + exact J_w, ground truth, sketcher API, index structures
  tests/                   unit tests (matching, truncation, band hashing, top-C, recall, GT, npz)
data/<dataset>/            canonical data, ground-truth caches
results/<dataset>/         CSVs, tables, figures, logs
```

## Build and reproduce

Requirements: GCC 14 with OpenMP (`g++-14`), Python 3
with numpy, scipy, pandas, matplotlib.

```sh
git clone https://github.com/tobc/dartminhash third_party/dartminhash
git clone --depth 1 --branch v0.8.2 https://github.com/Cyan4973/xxHash.git third_party/xxhash
make -C third_party/xxhash lib CC=gcc-14 CFLAGS="-O3 -march=native"
make -C wmh            # -std=c++17 -O3 -march=native -fopenmp
make -C wmh test       # unit tests

# 1. convert each dataset (one call per dataset; names are not hard-coded anywhere)
python3 -P wmh/scripts/convert.py --name cemetery --query <queryfile> --db <dbfile> --out data
# large inputs: add --fast [--workers N] [--no-sha256] (parallel parse + vectorized duplicate
# detection, identical output; ~9x faster parsing on 16 workers for cemetery)
python3 -P wmh/scripts/dataset_table.py data results

# 2. Step 0 sanity checks
wmh/bin/sanity data/<ds> results/<ds> 5 40 && python3 -P wmh/scripts/step0_report.py results/<ds>

# 3-6. ground truth, exact reference, all retrieval configurations, then tables and figures
python3 -P wmh/scripts/sweep.py
python3 -P wmh/scripts/report.py results
```

Re-running reproduces the cleaned data, the sketches (`sketch_checksum` column)
and every non-timing CSV column. `sweep.py` skips run files that already exist;
delete `results/<ds>/runs/` to start over.

## Design decisions

### Data
- **Input format.** `N d` header, then `id<TAB>v_1 ... v_d` dense nonnegative
  floats. The converter fails on negative/non-finite values, wrong row lengths, or
  a header N that disagrees with the row count.
- **Canonical format.** `scipy.sparse.save_npz(..., compressed=False)`: float64
  data (values exactly as parsed), int32 sorted column indices, int64 indptr.
  The C++ side reads these files directly (`include/npz.hpp`).
- **No transformation** of vectors anywhere outside a sketcher. Ground truth and
  rerank use the stored values.
- **Cleaning (agreed with the user).** Zero vectors dropped (none occurred);
  exact-duplicate queries dropped, first kept (sports: 9, leaving 2,991 queries);
  exact-duplicate database vectors dropped, first kept; database vectors identical
  to a query dropped. Counts are in `data/<ds>/manifest.json` and
  `results/dataset_stats.md`. Distinct vectors that share a string id are kept
  (rows are addressed by index).
- **Held-out ("train") split.** 500 vectors drawn uniformly from the cleaned
  database (`numpy.random.default_rng(20261014)`) and removed from it. Used only
  for Step 0 and for pruning Mode B settings, never for evaluation.

### Ground truth
- Exact top-100 per query by an inverted index: accumulate Σ min over the
  query's postings, then `J_w = s / (|q|_1 + |x|_1 − s)`. Contributions are added
  in ascending coordinate order, the same order as the merge-based brute force,
  so both give bit-identical J values.
- Cross-checked against brute force on 200 random queries per split; ids and J
  values must agree exactly (`data/<ds>/gt_*.json`).
- **Ties**: J descending, then smaller database row index. Ties across rank 100
  are counted in `gt_*.json`; a tie-aware Recall@100 column is also reported.
- Caches store content fingerprints of the query and database matrices and are
  refused if the data change.

### Sketchers
- One sketcher object per (algorithm, k, seed), constructed from
  `mt19937_64(seed)` with seed **20261014**, sketches both database and queries.
- **Upstream headers define non-inline functions** (not only
  `bagminhash_wrappers.hpp`: also `hashing.hpp` and `similarity.hpp`), so all of
  them are included only from `wmh/src/sketch.cpp`.
- **FastICWS weight scaling.** Upstream FastICWS computes
  `T2((uint64_t)t_k)`, where `t_k = floor(log S_k / r_k + β_k)` is negative for
  weights below 1. Converting a negative double to `uint64_t` is undefined
  behaviour; on this machine (Apple M1, GCC 14) it saturates to 0, so different
  `t_k` collapse to one hash and matches are inflated. Every weight here is ≤ 1.
  Inside the FastICWS wrapper weights are therefore multiplied by 2^40 (J_w is
  scale-invariant; weights down to 9e-13 keep `t_k ≥ 0`, the smallest weight in
  the data is 1e-10; a counter reports any weight that would still fall below 1).
  Step 0 shows the unscaled upstream version (`ficws_raw`) is biased upward on
  cemetery and the scaled one is not. Only the scaled version is used for retrieval.
- Upstream `ICWS` (non-fast) returns `(k*, y*)` but upstream's collision count
  compares only `k*`; it is not used.
- **FastICWS_t must not be copied or moved** (its FastICWS members hold
  references to its own tables), so sketchers live behind `unique_ptr`.
- **BagMinHash2** uses fixed internal hash seeds, so the `seed` column reads
  `upstream-fixed` and Step 0 has a single replicate for it. It discretizes weights
  to float32 internally. It is ~30x slower to sketch than DartMinHash here, so
  (being optional) it is swept only up to k = 256.
- **Empty vectors** are rejected by the sketcher (DartMinHash would divide by a
  zero L1 norm, FastICWS would take log 0). None exist after cleaning.

### Slots and estimator
- A slot stores the low b bits of `splitmix64(id)` (bijective mix first, then
  truncation; b = 64 stores the id unchanged). b ∈ {8, 16, 64}.
- Matching is slot by slot: slot j of the query against slot j of the database
  vector. Estimate = fraction of matching slots. The false-match correction
  `(p − 2^−b)/(1 − 2^−b)` is monotone and does not change rankings; Step 0
  tables report both uncorrected (default) and corrected values, labelled.

### Retrieval
- **Mode A (exhaustive scan).** Row-major b-bit sketch matrix; rows are scanned in
  256 KB chunks and every chunk is compared against a batch of 64 queries, so each
  row is read from memory once per batch. The inner loop over k slots is
  auto-vectorized. The top C is kept by buffered selection (append above the
  cut-off, trim to C with `nth_element` when the buffer reaches 2C).
- **Mode B (banded LSH).** k = L·r; one table per band keyed by a 64-bit hash of
  the band's r full 64-bit ids (independent of b). Tables are sorted (key, row)
  arrays with an open-addressing index. Candidates = union over bands (per-thread
  epoch stamps), then ranked by matches over all k stored slots.
  r ∈ {2, 4, 8, 16, 32, 64} with r | k.
- **Pruning of (L, r), held-out queries only.** Before building tables, the
  average union size for 50 held-out queries is computed directly from the
  sketches; settings whose estimate exceeds N/2 are skipped (Mode A then scans
  less), as are settings whose worst-case table size would exceed the 12 GB
  memory budget. Skipped settings are rows with a `skipped_*` status.
- **Ranking ties** (equal match counts) are broken by smaller row index, as in the
  ground truth.
- **Rerank.** Switchable: S = 0 returns the top-100 by match count; S ∈ {100, 500,
  1000, 2000} shortlists top-S by match count and returns the top-100 by exact
  J_w on the original vectors (ties: smaller row index).
- **Shortlist passes and timing per S.** A shortlist pass (query sketching +
  filter) keeps the top C(S) = max(S, 100) rows; S = 0 and S = 100 share the
  C = 100 pass, every other S has a pass of its own capacity (the scan cost
  grows with C: ~2x between C = 100 and C = 2000 on cemetery, so a shared
  C = 2000 pass would overstate small-S costs). Each S then has its own timed
  rerank pass; end-to-end time = shortlist pass + rerank pass, both measured.
  Results are identical to running each S separately (verified).
- **Exact reference.** The inverted-index exact search, timed the same way.

### Measurement
- **Recall@R** = |returned top-R ∩ true top-R| / R for R ∈ {1, 10, 100}, averaged
  over all evaluation queries. `recall_at_100_tie_aware` counts a returned item
  whose exact J_w equals the 100th true value as a hit.
- **Candidate and shortlist recall** (separate untimed pass, filter stage only):
  `candidate_recall_at_100` = fraction of the true top-100 in the candidate set
  (1 for Mode A; the LSH loss for Mode B); `shortlist_recall_at_100` = fraction in
  the top-C by match count (C = S, or 100 without rerank), i.e. the ceiling a
  rerank can reach. Checks: without rerank it must equal Recall@100
  (`diag_check`), and Recall@100 <= shortlist <= candidate recall always.
- **QPS.** `qps_mt`: all queries, all cores (`threads_mt`), dynamic schedule over
  batches of 64, one warm-up batch, query sketching and rerank included.
  `qps_mt_filter_only`: same, with precomputed query sketches and no rerank. Both MT numbers are the best of 2 timed passes.
  `qps_st`: one thread, the first 200 queries, query sketching and rerank
  included; per-phase times in `query_sketch_us_st`, `filter_us_st`,
  `rerank_us_st`. `timing_check` flags any row where end-to-end beats
  filter-only; `st_mt_identical` checks single- and multi-thread results agree.
- **Measurement conditions.** Timings use `CLOCK_UPTIME_RAW`, which stops while
  macOS sleeps; any timed pass during which the system slept is repeated
  (`sleep_retries` counts repeats; a value >= 100 would mean the guard gave up).
  `sweep.py` refuses to run on battery power or with Low Power Mode on, and the
  sweep is run under `caffeinate -i -s`. A first sweep (archived in
  `archive/sleep_contaminated_2026-10-17/`) ran while the laptop idle-slept
  overnight on battery in Low Power Mode; its timings are invalid and were
  discarded (its recall and memory columns are deterministic; they are compared with the rerun).
- **Sketching time**: `sketch_db_wall_s` (all cores) and
  `sketch_us_per_vector_st` (one thread, 500-vector sample).
  **Index build**: truncation (+ table build for Mode B).
- **Memory** (bytes, exact array sizes): sketches `N·k·b/8`, LSH tables, ids
  `8N` (int64 per vector, as faiss), posting ids inside the LSH tables also counted as 8 bytes (stored as int32), raw vectors counted as dense float32 `N·d·4` (same as the PQ side; the CSR actually held in RAM is larger). `mem_excl_raw_mb`
  and `mem_incl_raw_mb` are both reported; `mem_total_mb` (used in the memory
  plot) includes the raw vectors exactly when the configuration reranks.
- **k = 1024** is run only where its 64-bit database sketches fit in memory
  (cemetery);
