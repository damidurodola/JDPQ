#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <map>
#include <memory>
#include <tuple>
#include <cmath>
#include <limits>
#include <random>
#include <cstring>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sstream>
#include <iomanip>
#include "AnnIndex.h"
#include "IndexFlat.h"
#include "readData.h"
#include "distances.h"
#include "recall_metrics.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

// [[maybe_unused]]
void print_results(const std::string &index_name,
                                    int nq,
                                    int k,
                                    const std::vector<faiss_mimic::idx_t> &result_ids,
                                    const std::vector<float> &distances,
                                    const std::vector<int> &query_ids,
                                    std::ostream &out)
{
    out << "\n--- Results for " << index_name << " ---" << std::endl;
    int queries_to_print = std::min(nq, 2);
    int results_to_print = std::min(k, 5);

    for (int i = 0; i < queries_to_print; ++i)
    {
        int query_id = (i >= 0 && (size_t)i < query_ids.size()) ? query_ids[i] : -1;
        out << "Query " << i << " (ID: " << query_id << "):" << std::endl;
        for (int j = 0; j < results_to_print; ++j)
        {
            faiss_mimic::idx_t result_id = result_ids[(size_t)i * k + j];
            out << "  " << j + 1 << ": ID=" << result_id << ", Dist=" << distances[(size_t)i * k + j] << std::endl;
        }
    }
}

bool file_is_empty(const std::string &path)
{
    std::ifstream in(path);
    if (!in.good())
    {
        return true;
    }
    return in.peek() == std::ifstream::traits_type::eof();
}

float compute_metric_distance(const float *a,
                              const float *b,
                              int dim,
                              faiss_mimic::DistanceMetric metric)
{
    return (metric == faiss_mimic::DistanceMetric::Jaccard)
                         ? faiss_mimic::fvec_jaccard(a, b, dim)
                         : (metric == faiss_mimic::DistanceMetric::ShapeJaccard)
                                     ? faiss_mimic::fvec_shape_jaccard(a, b, dim)
                                     : faiss_mimic::fvec_L2sqr(a, b, dim);
}

struct GroundTruth
{
  std::vector<faiss_mimic::idx_t> labels;
  std::vector<float> distances;
  std::vector<faiss_mimic::idx_t> original_ids;
};

// Environment override for a compile-time default, so a batch job (see slurm/)
// can point the same binary at its own data and result files.
static std::string env_or(const char *name, const std::string &fallback)
{
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

int main(int argc, char** argv)
{
    // Auto-flush stdout after every write. Without this, output piped
    // through `tee`/a log file is fully buffered, so a crash (bad_alloc,
    // segfault) loses everything printed since the last flush -- which is
    // exactly what happened diagnosing the 1B-scale OOM below.
    std::cout << std::unitbuf;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    const bool is_root = (mpi_rank == 0);
#else
    [[maybe_unused]] const bool is_root = true;
#endif

    // Dataset file paths (declared up-front so the --gt-only fast path below
    // can use them without loading the full float vector files).
    // Each path below can be overridden by an environment variable (FAISS_DB,
    // FAISS_QUERY, FAISS_DB_BINARY_PREFIX, FAISS_TRAIN_PREFIX, FAISS_LOG,
    // FAISS_CSV); the literals are the defaults. PQ_GRID replaces pq_param_grid.
    std::string db_filename = env_or("FAISS_DB", "/path/to/data/"); // add dataset (training rows removed)
    std::string query_filename = env_or("FAISS_QUERY", "/path/to/data/");


    // const bool use_binary_db = !env_or("FAISS_DB_BINARY_PREFIX", "").empty();
    // // const bool use_binary_db = true;
    // const std::string db_binary_prefix = env_or("FAISS_DB_BINARY_PREFIX", "/path/to/data/");

    // Training set: saved separately from the database (the training rows
    // are removed from the database file, so the database is the add set in
    // its entirety). Binary prefix <prefix>.meta.txt / .ids.bin / .vecs.bin,

    const std::string train_prefix = env_or("FAISS_TRAIN_PREFIX", "/path/to/data/");




    std::ofstream log_file(env_or("FAISS_LOG", "/path/to/data/"), std::ios::app); // Log file for results

    if (!log_file.is_open())
    {
        std::cerr << "Failed to open pq_results-ivfpq-kmediods_parks.txt for writing" << std::endl;
        return 1;
    }
    auto log = [&](const std::string &msg)
    {
        std::cout << msg << std::flush;
        log_file << msg << std::flush;
    };

    const std::string csv_path = env_or("FAISS_CSV", "/path/to/data/");

    const std::vector<std::string> csv_headers = {
        "Subspaces(M)",
        "Clusters(k)",
        "Recall@1",
        "Recall@10",
        "Recall@50",
        "Recall@100",
        "IndexPQ Train-time",
        "IndexPQ Add-time",
        "IndexPQ Search-time",
        "IndexPQ QPS",
        "IndexPQ Rerank-time",
        "IndexPQ Size (MB)",
        "Brute-time",
        "Clustering niter",
        "IVFPQ Avg Candidates Scanned",
        // "IndexPQ Jaccard distortion",
        // "IVFPQ Coarse distortion",
        "IVFPQ Coarse_list",
        "IVFPQ nprobe",
        "IVFPQ Recall@1",
        "IVFPQ Recall@10",
        "IVFPQ Recall@50",
        "IVFPQ Recall@100",

        "IVFPQ Train-time",
        "IVFPQ Add-time",
        "IVFPQ Search-time",
        "IVFPQ QPS",
        // "IVFPQ End-to-end QPS",
        "IVFPQ Rerank-time",
        "IVFPQ Size (MB)",
        "PQ+PolyR Recall@1",
        "PQ+PolyR Recall@10",
        "PQ+PolyR Recall@50",
        "PQ+PolyR Recall@100",

        "IVFPQ+PolyR Recall@1",
        "IVFPQ+PolyR Recall@10",
        "IVFPQ+PolyR Recall@50",
        "IVFPQ+PolyR Recall@100",

        // Appended last so the columns above keep their positions.
        "Train size",
        "Add size",
        "Query size",
        };

    const bool need_csv_header = file_is_empty(csv_path);
    std::ofstream csv_file(csv_path, std::ios::app);
    if (!csv_file.is_open())
    {
        std::cerr << "Failed to open " << csv_path << " for writing" << std::endl;
        return 1;
    }

    // [[maybe_unused]]
    auto write_csv_line = [&](const std::vector<std::string> &fields)
    {
        for (size_t i = 0; i < fields.size(); ++i)
        {
            if (i > 0)
            {
                csv_file << ',';
            }
            csv_file << fields[i];
        }
        csv_file << '\n' << std::flush; // each row survives a kill between rows, like log()
    };

    if (need_csv_header)
    {
        write_csv_line(csv_headers);
    }


    auto format_double = [](double value)
    {
        if (!std::isfinite(value))
        {
            return std::string("nan");
        }
        std::ostringstream oss;
        oss << std::setprecision(10) << value;
        return oss.str();
    };

    log("Starting FAISS_MIMIC test\n");
#ifdef _OPENMP
    log("OpenMP threads: " + std::to_string(omp_get_max_threads()) + "\n");
#endif

    // Load Data
    log("Loading data...\n");

    // Training set: mmap'd, kept alive for all of main().
    // train_prefix may also name one of the three files; strip the suffix.
    std::string train_path = train_prefix;
    for (const std::string suffix : {".vecs.bin", ".ids.bin", ".meta.txt"})
    {
        if (train_path.size() > suffix.size() &&
            train_path.compare(train_path.size() - suffix.size(), suffix.size(), suffix) == 0)
        {
            train_path.resize(train_path.size() - suffix.size());
            break;
        }
    }
    faiss_mimic::MappedDataset train_mapped;
    if (!faiss_mimic::read_data_binary(train_path, train_mapped))
    {
        std::cerr << "Failed to load training set at prefix " << train_path
                  << " (run export_train first)." << std::endl;
        return 1;
    }
    // FAISS_NTRAIN trains on only the first N rows of the training file (the
    // rest of the mapping is never touched). The file's rows come from random
    // contiguous blocks, so a prefix is a union of whole blocks. The unused
    // rows were already removed from the database, so they stay out of the add
    // set. Unset = all rows.
    int ntrain_use = train_mapped.num_vectors;
    if (const std::string n_env = env_or("FAISS_NTRAIN", ""); !n_env.empty())
    {
        ntrain_use = std::atoi(n_env.c_str());
        if (ntrain_use <= 0 || ntrain_use > train_mapped.num_vectors)
        {
            std::cerr << "FAISS_NTRAIN=" << n_env << " must be between 1 and the training file's "
                      << train_mapped.num_vectors << " rows." << std::endl;
            return 1;
        }
    }
    const int ntrain = ntrain_use;
    const float *training_set = train_mapped.data;
    log("Loaded training set with " + std::to_string(ntrain) + " points of dimension " +
        std::to_string(train_mapped.dim) + " (first " + std::to_string(ntrain) + " of " +
        std::to_string(train_mapped.num_vectors) + " rows in the file)\n");

    int db_N = 0, Dim = 0;
    // ASCII path (small datasets): fully materialized, as before.
    std::vector<float> db_data;
    std::vector<int> db_ids;
    // Binary path (zinc20-scale): mmap'd, never copied into a vector. Kept
    // alive for the rest of main() since db_data_ptr/db_ids_ptr below point
    // into it.
    faiss_mimic::MappedDataset db_mapped;

    const float* db_data_ptr = nullptr;
    const int* db_ids_ptr = nullptr;

    if (use_binary_db)
    {
        if (!faiss_mimic::read_data_binary(db_binary_prefix, db_mapped))
        {
            std::cerr << "Failed to mmap binary database at prefix " << db_binary_prefix
                      << " (run convert_to_binary first)." << std::endl;
            return 1;
        }
        db_N = db_mapped.num_vectors;
        Dim = db_mapped.dim;
        db_data_ptr = db_mapped.data;
        db_ids_ptr = db_mapped.ids;
    }
    else
    {
        // db_filename / query_filename declared earlier (used by the
        // --gt-only fast path above too).
        faiss_mimic::read_data(db_filename, db_N, Dim, db_data, db_ids); // read database file
        db_data_ptr = db_data.data();
        db_ids_ptr = db_ids.data();
    }

    if (db_N == 0 || Dim == 0)
    {
        std::cerr << "Failed to load database file or data is empty." << std::endl;
        return 1;
    }
    log("Loaded database file with " + std::to_string(db_N) + " points of dimension " + std::to_string(Dim) + "\n");

    int q_N = 0, q_Dim = 0;
    std::vector<float> query_data;
    std::vector<int> query_ids;

    faiss_mimic::read_data(query_filename, q_N, q_Dim, query_data, query_ids); // read query file

    if (q_N == 0 || q_Dim == 0)
    {
        std::cerr << "Failed to load query file or data is empty." << std::endl;
        return 1;

    }
    log("Loaded query file with " + std::to_string(q_N) + " points of dimension " + std::to_string(q_Dim) + "\n");

    if (Dim != q_Dim || Dim != train_mapped.dim)
    {
        std::cerr << "Training (" << train_mapped.dim << "), database (" << Dim
                  << ") and query (" << q_Dim << ") dimensions do not match!" << std::endl;
        return 1;
    }

    // The database file is the add set in its entirety; queries are used as
    // loaded. Nothing is sampled or copied.
    int ndb = db_N;
    // FAISS_NQUERY uses only the first N queries of the file (the rest are never
    // read by the search or the ground truth). Unset = all of them.
    int nq = q_N;
    if (const std::string n_env = env_or("FAISS_NQUERY", ""); !n_env.empty())
    {
        nq = std::atoi(n_env.c_str());
        if (nq <= 0 || nq > q_N)
        {
            std::cerr << "FAISS_NQUERY=" << n_env << " must be between 1 and the query file's "
                      << q_N << " queries." << std::endl;
            return 1;
        }
    }
    const float *add_set = db_data_ptr;
    const int *add_set_ids = db_ids_ptr; // IDs corresponding to add_set vectors
    const float *queries = query_data.data();



    log("\nDatasets loaded:\n");
    log("  Training set size: " + std::to_string(ntrain) +
        (ntrain < train_mapped.num_vectors
             ? " (SUBSET: first " + std::to_string(ntrain) + " of " + std::to_string(train_mapped.num_vectors) +
                   " rows in the training file)"
             : " (all rows in the training file)") +
        "\n");
    log("  Database (add) set size: " + std::to_string(ndb) + "\n");
    log("  Query set size: " + std::to_string(nq) +
        (nq < q_N ? " (SUBSET: first " + std::to_string(nq) + " of " + std::to_string(q_N) +
                        " queries in the query file)"
                  : " (all queries in the query file)") +
        "\n");



    int k = std::min(100, ndb); // Number of ground truth neighbors to retrieve

    int R_max = std::min(1000, ndb); // Shortlist size: approximate neighbors retrieved per query before exact re-ranking.

    const std::vector<int> R_values = {1, 10, 50, 100};

    if (k <= 0 || R_max <= 0)
    {
        std::cerr << "k/R_max resolved to zero for add subset size " << ndb << "." << std::endl;
        return 1;
    }

    // std::vector<int> R_values = {1, 2, 5};
    // R_values declared above (used to size the IVFPQ shortlist).

    auto recall_lookup = [](const std::map<int, float> &table, int R) -> double
    {
        auto it = table.find(R);
        return (it != table.end()) ? static_cast<double>(it->second) : 0.0;
    };

    // Vector ground-truth
    auto build_ground_truth = [&](faiss_mimic::DistanceMetric metric)
    {
        GroundTruth gt;
        faiss_mimic::IndexFlat gt_index(Dim, metric);
        // add_no_copy, not add(): add_set already points into the mmap'd
        // binary dataset (or into a std::vector that outlives gt_index
        // either way), and ndb ~1B * Dim floats is ~1TB -- add()'s internal
        // copy would need that much RAM again on top of it.
        gt_index.add_no_copy(ndb, add_set);
        gt.labels.resize((size_t)nq * k);
        gt.distances.resize((size_t)nq * k);
        gt_index.search(nq, queries, k, gt.distances.data(), gt.labels.data());
        gt.original_ids.resize((size_t)nq * k);
        for (size_t i = 0; i < gt.labels.size(); ++i)
        {
            const auto label = gt.labels[i];
            gt.original_ids[i] =
                (label >= 0 && label < (faiss_mimic::idx_t)ndb) ? add_set_ids[label] : -1;
        }
        return gt;
    };

    // Rerank a PQ shortlist by exact vector distance. Mirrors
    // polygon_rerank_shortlist, but rescores against the original float
    // vectors instead of the WKT polygons, so it pairs with the
    // vector-Jaccard ground truth above. Takes the raw PQ labels (positions
    // into add_set) so no id -> position lookup is needed; entries that are
    // padding (-1) are dropped and the tail is left at +inf / -1.
    auto vector_rerank_shortlist = [&](const std::vector<faiss_mimic::idx_t> &shortlist_labels,
                                       int width,
                                       faiss_mimic::DistanceMetric metric,
                                       std::vector<faiss_mimic::idx_t> &out_ids,
                                       std::vector<float> &out_dists)
    {
        out_ids.assign((size_t)nq * width, -1);
        out_dists.assign((size_t)nq * width, std::numeric_limits<float>::infinity());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nq; ++i)
        {
            const float *qvec = queries + (size_t)i * Dim;
            std::vector<std::pair<float, faiss_mimic::idx_t>> scored;
            scored.reserve(width);
            for (int j = 0; j < width; ++j)
            {
                const faiss_mimic::idx_t pos = shortlist_labels[(size_t)i * width + j];
                if (pos < 0 || pos >= (faiss_mimic::idx_t)ndb)
                {
                    continue;
                }
                scored.emplace_back(
                    compute_metric_distance(qvec, add_set + (size_t)pos * Dim, Dim, metric),
                    pos);
            }
            std::sort(scored.begin(), scored.end(),
                      [](const std::pair<float, faiss_mimic::idx_t> &a,
                         const std::pair<float, faiss_mimic::idx_t> &b)
                      { return a.first < b.first; });
            for (size_t j = 0; j < scored.size(); ++j)
            {
                out_dists[(size_t)i * width + j] = scored[j].first;
                out_ids[(size_t)i * width + j] = add_set_ids[scored[j].second];
            }
        }
    };

    struct PQBenchmarkConfig
    {
        std::string label;
        faiss_mimic::DistanceMetric metric;
        bool show_debug;
    };

    struct PQParamSetting
    {
        int M;
        int nbits;
        int nlist_ivfpq;
        int nprobe_ivfpq;
    };

    std::vector<PQBenchmarkConfig> pq_configs = {
        {"Jaccard", faiss_mimic::DistanceMetric::Jaccard, false},
        // {"L2", faiss_mimic::DistanceMetric::L2, false}
    };

    // Cemetery grid (d=256, ntrain=20,000, ndb=169,421). Each entry is
    // {M, nbits, nlist, nprobe}; every row runs slot A with
    // (M, nbits) and slot B with (M, nbits, nlist, nprobe).
    //   * M in {1,...,128}: every divisor of d=256 up to 128 (bytes/vector =
    //     M*nbits/8), so PQ gets the full M x nbits ablation.
    //   * nbits in {2,4,6,8,10}: ksub up to 1024 (~19.5 training points per
    //     codeword at ntrain=20,000, the low end).
    //   * nlist in {128, 256, 512, 1024}: ntrain/nlist = 156, 78, 39 and 19.5
    //     points per centroid. 128-512 sit inside the 30-256 window the clustering
    //     needs; 1024 is below it and is the stress point. The 4-16*sqrt(ndb)
    //     rule of thumb ([1646, 6586]) is out of reach at this ntrain. The old
    //     zinc-scale nlist of 8192/16384 would give ~1-2 points per centroid.
    //   * nprobe: 1 up to nlist/4 (25%) in powers of two.
    // Per M: an nbits sweep at nlist=256/nprobe=16, then the nprobe sweeps at
    // nbits=8 for each nlist. Rows sharing (M, nbits) reuse one slot-A result
    // and rows sharing (M, nbits, nlist) reuse one built slot-B index, so only
    // the distinct builds are paid for (5 PQ and 8 IVFPQ builds per M) -- but
    // only the most recent IVFPQ index is kept, so rows with the same
    // (M, nbits, nlist) must stay adjacent. Set PQ_GRID="M:nbits:nlist:nprobe,..."
    // to replace the whole grid at run time (e.g. a small zinc-scale grid).
    std::vector<PQParamSetting> pq_param_grid = {
        // M=1 (dsub=256, 1..2 bytes/vector)
        {1, 2, 256, 16}, {1, 4, 256, 16}, {1, 6, 256, 16}, {1, 10, 256, 16},   // nbits sweep (nbits=8 is in the nlist=256 sweep below)
        {1, 8, 128, 1}, {1, 8, 128, 2}, {1, 8, 128, 4}, {1, 8, 128, 8}, {1, 8, 128, 16}, {1, 8, 128, 32},   // nbits=8, nlist=128 nprobe sweep
        {1, 8, 256, 1}, {1, 8, 256, 2}, {1, 8, 256, 4}, {1, 8, 256, 8}, {1, 8, 256, 16}, {1, 8, 256, 32}, {1, 8, 256, 64},   // nbits=8, nlist=256 nprobe sweep
        {1, 8, 512, 1}, {1, 8, 512, 2}, {1, 8, 512, 4}, {1, 8, 512, 8}, {1, 8, 512, 16}, {1, 8, 512, 32}, {1, 8, 512, 64}, {1, 8, 512, 128},   // nbits=8, nlist=512 nprobe sweep
        {1, 8, 1024, 1}, {1, 8, 1024, 2}, {1, 8, 1024, 4}, {1, 8, 1024, 8}, {1, 8, 1024, 16}, {1, 8, 1024, 32}, {1, 8, 1024, 64}, {1, 8, 1024, 128}, {1, 8, 1024, 256},   // nbits=8, nlist=1024 nprobe sweep
        // M=2 (dsub=128, 1..3 bytes/vector)
        {2, 2, 256, 16}, {2, 4, 256, 16}, {2, 6, 256, 16}, {2, 10, 256, 16},   // nbits sweep (nbits=8 is in the nlist=256 sweep below)
        {2, 8, 128, 1}, {2, 8, 128, 2}, {2, 8, 128, 4}, {2, 8, 128, 8}, {2, 8, 128, 16}, {2, 8, 128, 32},   // nbits=8, nlist=128 nprobe sweep
        {2, 8, 256, 1}, {2, 8, 256, 2}, {2, 8, 256, 4}, {2, 8, 256, 8}, {2, 8, 256, 16}, {2, 8, 256, 32}, {2, 8, 256, 64},   // nbits=8, nlist=256 nprobe sweep
        {2, 8, 512, 1}, {2, 8, 512, 2}, {2, 8, 512, 4}, {2, 8, 512, 8}, {2, 8, 512, 16}, {2, 8, 512, 32}, {2, 8, 512, 64}, {2, 8, 512, 128},   // nbits=8, nlist=512 nprobe sweep
        {2, 8, 1024, 1}, {2, 8, 1024, 2}, {2, 8, 1024, 4}, {2, 8, 1024, 8}, {2, 8, 1024, 16}, {2, 8, 1024, 32}, {2, 8, 1024, 64}, {2, 8, 1024, 128}, {2, 8, 1024, 256},   // nbits=8, nlist=1024 nprobe sweep
        // M=4 (dsub=64, 1..5 bytes/vector)
        {4, 2, 256, 16}, {4, 4, 256, 16}, {4, 6, 256, 16}, {4, 10, 256, 16},   // nbits sweep (nbits=8 is in the nlist=256 sweep below)
        {4, 8, 128, 1}, {4, 8, 128, 2}, {4, 8, 128, 4}, {4, 8, 128, 8}, {4, 8, 128, 16}, {4, 8, 128, 32},   // nbits=8, nlist=128 nprobe sweep
        {4, 8, 256, 1}, {4, 8, 256, 2}, {4, 8, 256, 4}, {4, 8, 256, 8}, {4, 8, 256, 16}, {4, 8, 256, 32}, {4, 8, 256, 64},   // nbits=8, nlist=256 nprobe sweep
        {4, 8, 512, 1}, {4, 8, 512, 2}, {4, 8, 512, 4}, {4, 8, 512, 8}, {4, 8, 512, 16}, {4, 8, 512, 32}, {4, 8, 512, 64}, {4, 8, 512, 128},   // nbits=8, nlist=512 nprobe sweep
        {4, 8, 1024, 1}, {4, 8, 1024, 2}, {4, 8, 1024, 4}, {4, 8, 1024, 8}, {4, 8, 1024, 16}, {4, 8, 1024, 32}, {4, 8, 1024, 64}, {4, 8, 1024, 128}, {4, 8, 1024, 256},   // nbits=8, nlist=1024 nprobe sweep
        // M=8 (dsub=32, 2..10 bytes/vector)
        {8, 2, 256, 16}, {8, 4, 256, 16}, {8, 6, 256, 16}, {8, 10, 256, 16},   // nbits sweep (nbits=8 is in the nlist=256 sweep below)
        {8, 8, 128, 1}, {8, 8, 128, 2}, {8, 8, 128, 4}, {8, 8, 128, 8}, {8, 8, 128, 16}, {8, 8, 128, 32},   // nbits=8, nlist=128 nprobe sweep
        {8, 8, 256, 1}, {8, 8, 256, 2}, {8, 8, 256, 4}, {8, 8, 256, 8}, {8, 8, 256, 16}, {8, 8, 256, 32}, {8, 8, 256, 64},   // nbits=8, nlist=256 nprobe sweep
        {8, 8, 512, 1}, {8, 8, 512, 2}, {8, 8, 512, 4}, {8, 8, 512, 8}, {8, 8, 512, 16}, {8, 8, 512, 32}, {8, 8, 512, 64}, {8, 8, 512, 128},   // nbits=8, nlist=512 nprobe sweep
        {8, 8, 1024, 1}, {8, 8, 1024, 2}, {8, 8, 1024, 4}, {8, 8, 1024, 8}, {8, 8, 1024, 16}, {8, 8, 1024, 32}, {8, 8, 1024, 64}, {8, 8, 1024, 128}, {8, 8, 1024, 256},   // nbits=8, nlist=1024 nprobe sweep
        // M=16 (dsub=16, 4..20 bytes/vector)
        {16, 2, 256, 16}, {16, 4, 256, 16}, {16, 6, 256, 16}, {16, 10, 256, 16},   // nbits sweep (nbits=8 is in the nlist=256 sweep below)
        {16, 8, 128, 1}, {16, 8, 128, 2}, {16, 8, 128, 4}, {16, 8, 128, 8}, {16, 8, 128, 16}, {16, 8, 128, 32},   // nbits=8, nlist=128 nprobe sweep
        {16, 8, 256, 1}, {16, 8, 256, 2}, {16, 8, 256, 4}, {16, 8, 256, 8}, {16, 8, 256, 16}, {16, 8, 256, 32}, {16, 8, 256, 64},   // nbits=8, nlist=256 nprobe sweep
        {16, 8, 512, 1}, {16, 8, 512, 2}, {16, 8, 512, 4}, {16, 8, 512, 8}, {16, 8, 512, 16}, {16, 8, 512, 32}, {16, 8, 512, 64}, {16, 8, 512, 128},   // nbits=8, nlist=512 nprobe sweep
        {16, 8, 1024, 1}, {16, 8, 1024, 2}, {16, 8, 1024, 4}, {16, 8, 1024, 8}, {16, 8, 1024, 16}, {16, 8, 1024, 32}, {16, 8, 1024, 64}, {16, 8, 1024, 128}, {16, 8, 1024, 256},   // nbits=8, nlist=1024 nprobe sweep
        // M=32 (dsub=8, 8..40 bytes/vector)
        {32, 2, 256, 16}, {32, 4, 256, 16}, {32, 6, 256, 16}, {32, 10, 256, 16},   // nbits sweep (nbits=8 is in the nlist=256 sweep below)
        {32, 8, 128, 1}, {32, 8, 128, 2}, {32, 8, 128, 4}, {32, 8, 128, 8}, {32, 8, 128, 16}, {32, 8, 128, 32},   // nbits=8, nlist=128 nprobe sweep
        {32, 8, 256, 1}, {32, 8, 256, 2}, {32, 8, 256, 4}, {32, 8, 256, 8}, {32, 8, 256, 16}, {32, 8, 256, 32}, {32, 8, 256, 64},   // nbits=8, nlist=256 nprobe sweep
        {32, 8, 512, 1}, {32, 8, 512, 2}, {32, 8, 512, 4}, {32, 8, 512, 8}, {32, 8, 512, 16}, {32, 8, 512, 32}, {32, 8, 512, 64}, {32, 8, 512, 128},   // nbits=8, nlist=512 nprobe sweep
        {32, 8, 1024, 1}, {32, 8, 1024, 2}, {32, 8, 1024, 4}, {32, 8, 1024, 8}, {32, 8, 1024, 16}, {32, 8, 1024, 32}, {32, 8, 1024, 64}, {32, 8, 1024, 128}, {32, 8, 1024, 256},   // nbits=8, nlist=1024 nprobe sweep
        // M=64 (dsub=4, 16..80 bytes/vector)
        {64, 2, 256, 16}, {64, 4, 256, 16}, {64, 6, 256, 16}, {64, 10, 256, 16},   // nbits sweep (nbits=8 is in the nlist=256 sweep below)
        {64, 8, 128, 1}, {64, 8, 128, 2}, {64, 8, 128, 4}, {64, 8, 128, 8}, {64, 8, 128, 16}, {64, 8, 128, 32},   // nbits=8, nlist=128 nprobe sweep
        {64, 8, 256, 1}, {64, 8, 256, 2}, {64, 8, 256, 4}, {64, 8, 256, 8}, {64, 8, 256, 16}, {64, 8, 256, 32}, {64, 8, 256, 64},   // nbits=8, nlist=256 nprobe sweep
        {64, 8, 512, 1}, {64, 8, 512, 2}, {64, 8, 512, 4}, {64, 8, 512, 8}, {64, 8, 512, 16}, {64, 8, 512, 32}, {64, 8, 512, 64}, {64, 8, 512, 128},   // nbits=8, nlist=512 nprobe sweep
        {64, 8, 1024, 1}, {64, 8, 1024, 2}, {64, 8, 1024, 4}, {64, 8, 1024, 8}, {64, 8, 1024, 16}, {64, 8, 1024, 32}, {64, 8, 1024, 64}, {64, 8, 1024, 128}, {64, 8, 1024, 256},   // nbits=8, nlist=1024 nprobe sweep
        // M=128 (dsub=2, 32..160 bytes/vector)
        {128, 2, 256, 16}, {128, 4, 256, 16}, {128, 6, 256, 16}, {128, 10, 256, 16},   // nbits sweep (nbits=8 is in the nlist=256 sweep below)
        {128, 8, 128, 1}, {128, 8, 128, 2}, {128, 8, 128, 4}, {128, 8, 128, 8}, {128, 8, 128, 16}, {128, 8, 128, 32},   // nbits=8, nlist=128 nprobe sweep
        {128, 8, 256, 1}, {128, 8, 256, 2}, {128, 8, 256, 4}, {128, 8, 256, 8}, {128, 8, 256, 16}, {128, 8, 256, 32}, {128, 8, 256, 64},   // nbits=8, nlist=256 nprobe sweep
        {128, 8, 512, 1}, {128, 8, 512, 2}, {128, 8, 512, 4}, {128, 8, 512, 8}, {128, 8, 512, 16}, {128, 8, 512, 32}, {128, 8, 512, 64}, {128, 8, 512, 128},   // nbits=8, nlist=512 nprobe sweep
        {128, 8, 1024, 1}, {128, 8, 1024, 2}, {128, 8, 1024, 4}, {128, 8, 1024, 8}, {128, 8, 1024, 16}, {128, 8, 1024, 32}, {128, 8, 1024, 64}, {128, 8, 1024, 128}, {128, 8, 1024, 256},   // nbits=8, nlist=1024 nprobe sweep

    };

    if (const std::string grid_env = env_or("PQ_GRID", ""); !grid_env.empty())
    {
        std::vector<PQParamSetting> parsed;
        std::stringstream rows(grid_env);
        std::string row;
        while (std::getline(rows, row, ','))
        {
            if (row.empty()) continue;
            std::replace(row.begin(), row.end(), ':', ' ');
            std::stringstream fields(row);
            PQParamSetting setting{};
            if (!(fields >> setting.M >> setting.nbits >> setting.nlist_ivfpq >> setting.nprobe_ivfpq) ||
                setting.M <= 0 || setting.nbits <= 0 || setting.nlist_ivfpq <= 0 || setting.nprobe_ivfpq <= 0)
            {
                std::cerr << "Bad PQ_GRID entry '" << row << "' (expected M:nbits:nlist:nprobe)." << std::endl;
                return 1;
            }
            parsed.push_back(setting);
        }
        pq_param_grid = parsed;
        log("PQ_GRID override: " + std::to_string(pq_param_grid.size()) + " rows\n");
    }


#ifdef USE_MPI
        // Non-root ranks only participate in GT computation; exit now.
        if (!is_root) {
            MPI_Finalize();
            return 0;
        }
#endif


        // ---- Methods under test --------------------------------------------
        // Two method slots feed the two column groups of the CSV: slot A fills the
        // non-inverted-file group ("IndexPQ ..."), slot B the inverted-file group
        // ("IVFPQ ..."). Each is a name understood by make_index() (see
        // AnnIndex.h); an empty name disables the slot. "flat" (exact search) is
        // always available; "pq" / "ivfpq" need the JD-PQ plugin.
        // getenv directly: an empty value must disable the slot, not select the default.
        const char *env_a = std::getenv("ANN_METHOD_A");
        const char *env_b = std::getenv("ANN_METHOD_B");
        const std::string kind_a = env_a ? env_a : "pq";
        const std::string kind_b = env_b ? env_b : "ivfpq";
        log("Methods: slot A = " + (kind_a.empty() ? std::string("<off>") : kind_a) +
            ", slot B = " + (kind_b.empty() ? std::string("<off>") : kind_b) + "\n");

        struct SearchScores
        {
            std::map<int, float> recalls;        // overlap with the exact top-k, from the raw shortlist
            std::map<int, float> rerank_recalls; // after exact re-ranking of the shortlist
            double search_ms_per_query = 0.0;
            double qps = 0.0;
            double rerank_seconds = 0.0;
            double avg_candidates = 0.0;
        };
        struct BuiltIndex
        {
            std::unique_ptr<faiss_mimic::AnnIndex> index;
            double train_seconds = 0.0;
            double add_seconds = 0.0;
            double size_mb = 0.0; // MiB
        };

        // Train and populate one index, logging each stage.
        auto build_index = [&](const faiss_mimic::IndexSpec &spec) -> BuiltIndex
        {
            BuiltIndex built;
            built.index = faiss_mimic::make_index(spec);
            if (!built.index)
            {
                std::cerr << "Method '" << spec.kind << "' is not available in this build "
                          << "(the JD-PQ plugin is released upon acceptance; use ANN_METHOD_A=flat "
                          << "ANN_METHOD_B= for the exact reference)." << std::endl;
                std::exit(1);
            }
            const std::string tag = "[" + spec.kind + "] ";
            auto t0 = std::chrono::high_resolution_clock::now();
            built.index->train(ntrain, training_set);
            auto t1 = std::chrono::high_resolution_clock::now();
            built.train_seconds = std::chrono::duration<double>(t1 - t0).count();
            log(tag + "train time: " + std::to_string(built.train_seconds) + " s\n");
            built.index->add(ndb, add_set);
            auto t2 = std::chrono::high_resolution_clock::now();
            built.add_seconds = std::chrono::duration<double>(t2 - t1).count();
            log(tag + "add time: " + std::to_string(built.add_seconds) + " s\n");
            built.size_mb = static_cast<double>(built.index->memory_usage_bytes()) / (1024.0 * 1024.0);
            log(tag + "memory size: " + std::to_string(built.size_mb) + " MiB\n");
            const std::string extra = built.index->build_log();
            if (!extra.empty()) log(tag + extra);
            return built;
        };

        // Search with the given parameters and score it against the exact ground truth.
        auto search_and_score = [&](faiss_mimic::AnnIndex &index, const std::string &kind,
                                    const faiss_mimic::SearchParams &sp) -> SearchScores
        {
            SearchScores s;
            const std::string tag = "[" + kind + "] ";
            std::vector<float> dists((size_t)nq * R_max);
            std::vector<faiss_mimic::idx_t> pos((size_t)nq * R_max);
            auto t0 = std::chrono::high_resolution_clock::now();
            index.search(nq, queries, R_max, sp, pos.data(), dists.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            const double secs = std::chrono::duration<double>(t1 - t0).count();
            s.search_ms_per_query = (nq > 0) ? (secs / static_cast<double>(nq)) * 1000.0 : 0.0;
            s.qps = (secs > 0.0) ? static_cast<double>(nq) / secs : 0.0;
            s.avg_candidates = index.avg_candidates_scanned();
            log(tag + "search time (for R=" + std::to_string(R_max) + "): " + std::to_string(secs) + " s\n");
            log(tag + "QPS: " + std::to_string(s.qps) + "\n");
            if (s.avg_candidates > 0.0)
                log(tag + "avg candidates scanned per query (nprobe=" + std::to_string(sp.nprobe) +
                    "): " + std::to_string(s.avg_candidates) + "\n");

            // positions -> dataset ids
            std::vector<faiss_mimic::idx_t> ids((size_t)nq * R_max);
            for (size_t i = 0; i < pos.size(); ++i)
                ids[i] = (pos[i] >= 0 && pos[i] < (faiss_mimic::idx_t)ndb) ? add_set_ids[pos[i]] : -1;
            print_results(kind, nq, R_max, ids, dists, query_ids, log_file);
            s.recalls = compute_recall_map(nq, k, R_max, R_values, gt_ptr->original_ids, ids);

            // Exact re-ranking of the shortlist (vector Jaccard, matching the vector ground truth).
            auto r0 = std::chrono::high_resolution_clock::now();
            std::vector<faiss_mimic::idx_t> rr_ids;
            std::vector<float> rr_dists;
            vector_rerank_shortlist(pos, R_max, config.metric, rr_ids, rr_dists);
            auto r1 = std::chrono::high_resolution_clock::now();
            s.rerank_seconds = std::chrono::duration<double>(r1 - r0).count();
            log(tag + "rerank time (shortlist width=" + std::to_string(R_max) + "): " +
                std::to_string(s.rerank_seconds) + " s\n");
            s.rerank_recalls = compute_recall_map(nq, k, R_max, R_values, gt_ptr->original_ids, rr_ids);
            for (int R : R_values)
                log(config.label + " " + kind + " Recall@" + std::to_string(R) +
                    " [overlap=" + std::to_string(recall_lookup(s.recalls, R)) +
                    ", rerank=" + std::to_string(recall_lookup(s.rerank_recalls, R)) + "]\n");
            return s;
        };

        // Rows that differ only in nlist/nprobe (or only in nprobe) share all the
        // expensive work, so it is done once and reused:
        //   * slot A depends only on (M, nbits): built, searched and scored once; the
        //     stored result fills every later row with the same (M, nbits).
        //   * slot B depends on (M, nbits, nlist): the populated index is built once
        //     and each nprobe row only re-runs the search. Only the most recent index
        //     is kept, so memory stays at one index even at 1B vectors; rows sharing
        //     (M, nbits, nlist) must therefore be adjacent in the grid.
        struct SlotAResult
        {
            SearchScores scores;
            double train_seconds = 0.0, add_seconds = 0.0, size_mb = 0.0;
            int niter = 0;
        };
        std::map<std::pair<int, int>, SlotAResult> slot_a_cache;
        BuiltIndex slot_b_current;
        std::tuple<int, int, size_t> slot_b_key{-1, -1, 0};

        for (const auto &pq_params : pq_param_grid)
        {
            const int M_pq = pq_params.M;
            const int nbits_pq = pq_params.nbits;
            const size_t nlist_ivfpq = pq_params.nlist_ivfpq;
            const int nprobe_ivfpq = pq_params.nprobe_ivfpq;

            std::cout << "\n=== Setting: metric=" << config.label << ", M=" << M_pq << ", nbits=" << nbits_pq
                      << ", nlist=" << nlist_ivfpq << ", nprobe=" << nprobe_ivfpq << " ===" << std::endl;
            log("\n=== Setting: metric=" + config.label + ", M=" + std::to_string(M_pq) +
                ", nbits=" + std::to_string(nbits_pq) + ", nlist=" + std::to_string(nlist_ivfpq) +
                ", nprobe=" + std::to_string(nprobe_ivfpq) + " ===\n");

            faiss_mimic::IndexSpec spec;
            spec.dim = Dim;
            spec.M = M_pq;
            spec.nbits = nbits_pq;
            spec.nlist = nlist_ivfpq;
            spec.metric = config.metric;

            // --- slot A ---
            SlotAResult a;
            if (!kind_a.empty())
            {
                const auto key_a = std::make_pair(M_pq, nbits_pq);
                auto cached = slot_a_cache.find(key_a);
                if (cached == slot_a_cache.end())
                {
                    spec.kind = kind_a;
                    BuiltIndex built = build_index(spec);
                    SlotAResult r;
                    r.scores = search_and_score(*built.index, kind_a, faiss_mimic::SearchParams{});
                    r.train_seconds = built.train_seconds;
                    r.add_seconds = built.add_seconds;
                    r.size_mb = built.size_mb;
                    r.niter = built.index->clustering_niter();
                    cached = slot_a_cache.emplace(key_a, std::move(r)).first;
                }
                else
                {
                    log("[" + kind_a + "] (M=" + std::to_string(M_pq) + ", nbits=" + std::to_string(nbits_pq) +
                        ") already benchmarked: reusing its result for this row\n");
                }
                a = cached->second;
            }

            // --- slot B ---
            BuiltIndex *b = nullptr;
            SearchScores b_scores;
            if (!kind_b.empty())
            {
                const auto key_b = std::make_tuple(M_pq, nbits_pq, nlist_ivfpq);
                if (!slot_b_current.index || key_b != slot_b_key)
                {
                    slot_b_current = BuiltIndex(); // free the previous index before building the next
                    spec.kind = kind_b;
                    slot_b_current = build_index(spec);
                    slot_b_key = key_b;
                }
                else
                {
                    log("[" + kind_b + "] index (M=" + std::to_string(M_pq) + ", nbits=" + std::to_string(nbits_pq) +
                        ", nlist=" + std::to_string(nlist_ivfpq) + ") already built: reusing it, only nprobe changes\n");
                }
                b = &slot_b_current;
                faiss_mimic::SearchParams sp;
                sp.nprobe = nprobe_ivfpq;
                b_scores = search_and_score(*b->index, kind_b, sp);
            }

            const int row_niter = a.niter ? a.niter : (b ? b->index->clustering_niter() : 0);
            {
                const std::vector<std::string> csv_row = {
                    std::to_string(M_pq),
                    std::to_string(1 << nbits_pq),
                    format_double(recall_lookup(a.scores.recalls, 1)),
                    format_double(recall_lookup(a.scores.recalls, 10)),
                    format_double(recall_lookup(a.scores.recalls, 50)),
                    format_double(recall_lookup(a.scores.recalls, 100)),
                    format_double(a.train_seconds),
                    format_double(a.add_seconds),
                    format_double(a.scores.search_ms_per_query),
                    format_double(a.scores.qps),
                    format_double(a.scores.rerank_seconds),
                    format_double(a.size_mb),
                    format_double(brute_seconds),
                    std::to_string(row_niter),
                    format_double(b_scores.avg_candidates),
                    std::to_string(b ? nlist_ivfpq : 0),
                    std::to_string(b ? nprobe_ivfpq : 0),
                    format_double(recall_lookup(b_scores.recalls, 1)),
                    format_double(recall_lookup(b_scores.recalls, 10)),
                    format_double(recall_lookup(b_scores.recalls, 50)),
                    format_double(recall_lookup(b_scores.recalls, 100)),
                    format_double(b ? b->train_seconds : 0.0),
                    format_double(b ? b->add_seconds : 0.0),
                    format_double(b_scores.search_ms_per_query),
                    format_double(b_scores.qps),
                    format_double(b_scores.rerank_seconds),
                    format_double(b ? b->size_mb : 0.0),

                    // slot A, re-ranked
                    format_double(recall_lookup(a.scores.rerank_recalls, 1)),
                    format_double(recall_lookup(a.scores.rerank_recalls, 10)),
                    format_double(recall_lookup(a.scores.rerank_recalls, 50)),
                    format_double(recall_lookup(a.scores.rerank_recalls, 100)),

                    // slot B, re-ranked
                    format_double(recall_lookup(b_scores.rerank_recalls, 1)),
                    format_double(recall_lookup(b_scores.rerank_recalls, 10)),
                    format_double(recall_lookup(b_scores.rerank_recalls, 50)),
                    format_double(recall_lookup(b_scores.rerank_recalls, 100)),

                    std::to_string(ntrain), // training vectors actually used (FAISS_NTRAIN may be a subset of the file)
                    std::to_string(ndb),
                    std::to_string(nq),
                };
                write_csv_line(csv_row);
            }
        }
    }


    return 0;
}



