// PQ / IVFPQ ablation with the reference FAISS library (Euclidean distance) on
// the cemetery dataset, for comparison with test.cpp's Jaccard-based indexes.
//
// Same data and protocol as test.cpp: the training set is the saved 20,000-row
// binary subset, the database is the add set (training rows already removed),
// the ground truth is an exact top-100 scan -- reported against BOTH exact L2
// (IndexFlatL2) and exact weighted Jaccard distance (fvec_jaccard, the metric
// test.cpp's indexes are scored on) -- and
// Recall@R = |A_R ∩ G_R| / R averaged over queries, R in {1, 10, 50, 100}.
// The (M, nbits, nlist, nprobe) grid mirrors the cemetery grid in test.cpp.
//
// Links against the real libfaiss, not src/*: build with `make faiss_pq_ablation`
// (FAISS_PREFIX points at the install, default ~/faiss-cpp-env).
//
// Environment overrides: FAISS_DB, FAISS_QUERY, FAISS_TRAIN_PREFIX, FAISS_CSV,
// PQ_GRID ("M:nbits:nlist:nprobe,..."), FAISS_BY_RESIDUAL (default 1),
// FAISS_NQ (limit queries).

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>

#include "src/distances.h"
#include "src/readData.h"

namespace {

struct Setting
{
    int M, nbits, nlist, nprobe;
};

std::string env_or(const char *name, const std::string &fallback)
{
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// IVFPQ is only ablated for these M and nbits=8; IndexPQ covers every M and nbits.
bool runs_ivfpq(const Setting &s)
{
    return s.nbits == 8 && (s.M == 8 || s.M == 16 || s.M == 32 || s.M == 64);
}

// Cemetery grid as in test.cpp, trimmed: per M an IndexPQ-only nbits sweep, then
// for the IVFPQ Ms nprobe sweeps at nbits=8 for each nlist (M=128 gets a single
// IndexPQ nbits=8 row). Rows sharing (M, nbits, nlist) stay adjacent so one
// trained IVFPQ index serves all of them.
std::vector<Setting> default_grid()
{
    const std::vector<int> Ms = {8, 16, 32, 64, 128};
    const std::vector<int> nbits_sweep = {2, 4, 6, 10};
    const std::vector<std::pair<int, int>> nlist_max_nprobe = {{128, 32}, {256, 64}, {512, 128}, {1024, 256}};
    std::vector<Setting> grid;
    for (int M : Ms)
    {
        for (int nb : nbits_sweep)
            grid.push_back({M, nb, 256, 16});
        if (!runs_ivfpq({M, 8, 0, 0}))
            grid.push_back({M, 8, 256, 16});
        else
            for (const auto &[nlist, max_nprobe] : nlist_max_nprobe)
                for (int np = 1; np <= max_nprobe; np *= 2)
                    grid.push_back({M, 8, nlist, np});
    }
    return grid;
}

std::vector<Setting> parse_grid(const std::string &s)
{
    std::vector<Setting> grid;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        Setting g;
        if (std::sscanf(item.c_str(), "%d:%d:%d:%d", &g.M, &g.nbits, &g.nlist, &g.nprobe) != 4)
        {
            std::cerr << "Bad PQ_GRID entry: " << item << "\n";
            std::exit(1);
        }
        grid.push_back(g);
    }
    return grid;
}

// Recall@R = |A_R ∩ G_R| / R, averaged over queries. `res` has stride `res_k`,
// `gt` has stride `gt_k`; both hold database positions.
double recall_at(int nq, int R, int gt_k, const std::vector<faiss::idx_t> &gt,
                 int res_k, const std::vector<faiss::idx_t> &res)
{
    long long found = 0;
    for (int i = 0; i < nq; ++i)
    {
        std::unordered_set<faiss::idx_t> truth(gt.begin() + (size_t)i * gt_k,
                                               gt.begin() + (size_t)i * gt_k + R);
        for (int j = 0; j < R; ++j)
            found += truth.count(res[(size_t)i * res_k + j]);
    }
    return (double)found / ((double)nq * R);
}

bool file_is_empty(const std::string &path)
{
    std::ifstream in(path);
    return !in.good() || in.peek() == std::ifstream::traits_type::eof();
}

} // namespace

int main()
{
    std::cout << std::unitbuf;

    const std::string db_file = env_or("FAISS_DB", "/path/to/data/ICDE_DATA/testfinal_cemetery_vectors2.txt");
    const std::string query_file = env_or("FAISS_QUERY", "/path/to/data/ICDE_DATA/queryfinal_cemetery_vectors2.txt");
    const std::string train_prefix = env_or("FAISS_TRAIN_PREFIX", "/path/to/data/ICDE_DATA/testfinal_cemetery_vectors2_train20000");
    const std::string csv_path = env_or("FAISS_CSV", "faiss_pq_ablation_cemetery.csv");
    const bool by_residual = env_or("FAISS_BY_RESIDUAL", "1") != "0";

    std::vector<Setting> grid = default_grid();
    if (const std::string g = env_or("PQ_GRID", ""); !g.empty())
        grid = parse_grid(g);

    // --- load data ---
    int ndb = 0, dim = 0, nq = 0, qdim = 0;
    std::vector<float> db, queries;
    std::vector<int> db_ids, q_ids;
    faiss_mimic::read_data(db_file, ndb, dim, db, db_ids);
    faiss_mimic::read_data(query_file, nq, qdim, queries, q_ids);
    if (ndb == 0 || nq == 0 || dim != qdim)
    {
        std::cerr << "Failed to load database/query (ndb=" << ndb << ", nq=" << nq
                  << ", dim=" << dim << ", qdim=" << qdim << ")\n";
        return 1;
    }
    if (const int lim = std::atoi(env_or("FAISS_NQ", "0").c_str()); lim > 0 && lim < nq)
        nq = lim;

    faiss_mimic::MappedDataset train;
    if (!faiss_mimic::read_data_binary(train_prefix, train) || train.dim != dim)
    {
        std::cerr << "Failed to load training set " << train_prefix << "\n";
        return 1;
    }
    const faiss::idx_t nt = train.num_vectors;
    std::cout << "Loaded db=" << ndb << " queries=" << nq << " train=" << nt
              << " dim=" << dim << " by_residual=" << by_residual << "\n";

    // --- exact L2 ground truth ---
    const int gt_k = std::min(100, ndb);
    const int R_max = std::min(1000, ndb);
    const std::vector<int> R_values = {1, 10, 50, 100};
    std::vector<faiss::idx_t> gt((size_t)nq * gt_k);
    std::vector<faiss::idx_t> gt_jd((size_t)nq * gt_k);
    {
        std::vector<float> gt_dist((size_t)nq * gt_k);
        faiss::IndexFlatL2 flat(dim);
        flat.add(ndb, db.data());
        auto t0 = std::chrono::steady_clock::now();
        flat.search(nq, queries.data(), gt_k, gt_dist.data(), gt.data());
        std::cout << "L2 ground truth: " << seconds_since(t0) << " s\n";

        // Exact weighted-Jaccard top-gt_k, brute force (ties broken by position).
        t0 = std::chrono::steady_clock::now();
#pragma omp parallel
        {
            std::vector<std::pair<float, faiss::idx_t>> d(ndb);
#pragma omp for schedule(dynamic)
            for (int i = 0; i < nq; ++i)
            {
                const float *q = queries.data() + (size_t)i * dim;
                for (faiss::idx_t j = 0; j < ndb; ++j)
                    d[j] = {faiss_mimic::fvec_jaccard(q, db.data() + (size_t)j * dim, dim), j};
                std::partial_sort(d.begin(), d.begin() + gt_k, d.end());
                for (int r = 0; r < gt_k; ++r)
                    gt_jd[(size_t)i * gt_k + r] = d[r].second;
            }
        }
        std::cout << "Weighted-JD ground truth: " << seconds_since(t0) << " s\n";
    }

    const bool need_header = file_is_empty(csv_path);
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv)
    {
        std::cerr << "Cannot open " << csv_path << "\n";
        return 1;
    }
    if (need_header)
        csv << "Subspaces(M),nbits,Clusters(k),"
               "IndexPQ L2-GT Recall@1,IndexPQ L2-GT Recall@10,IndexPQ L2-GT Recall@50,IndexPQ L2-GT Recall@100,"
               "IndexPQ JD-GT Recall@1,IndexPQ JD-GT Recall@10,IndexPQ JD-GT Recall@50,IndexPQ JD-GT Recall@100,"
               "IndexPQ Train-time,IndexPQ Add-time,IndexPQ Search-time,IndexPQ QPS,IndexPQ Size (MB),"
               "IVFPQ Coarse_list,IVFPQ nprobe,"
               "IVFPQ L2-GT Recall@1,IVFPQ L2-GT Recall@10,IVFPQ L2-GT Recall@50,IVFPQ L2-GT Recall@100,"
               "IVFPQ JD-GT Recall@1,IVFPQ JD-GT Recall@10,IVFPQ JD-GT Recall@50,IVFPQ JD-GT Recall@100,"
               "IVFPQ Train-time,IVFPQ Add-time,IVFPQ Search-time,IVFPQ QPS,IVFPQ Size (MB),"
               "By residual,Train size,Add size,Query size\n";

    auto search_recalls = [&](faiss::Index &index, double &search_s, std::vector<double> &recalls)
    {
        std::vector<faiss::idx_t> labels((size_t)nq * R_max);
        std::vector<float> dists((size_t)nq * R_max);
        auto t0 = std::chrono::steady_clock::now();
        index.search(nq, queries.data(), R_max, dists.data(), labels.data());
        search_s = seconds_since(t0);
        recalls.clear();
        for (const auto *truth : {&gt, &gt_jd})
            for (int R : R_values)
                recalls.push_back(recall_at(nq, std::min(R, gt_k), gt_k, *truth, R_max, labels));
    };

    // IndexPQ is cached per (M, nbits); IVFPQ per (M, nbits, nlist).
    std::tuple<int, int> pq_key{-1, -1};
    std::tuple<int, int, int> ivf_key{-1, -1, -1};
    struct PQResult
    {
        std::vector<double> recalls;
        double train_s = 0, add_s = 0, search_s = 0, size_mb = 0;
    } pq_res;
    faiss::IndexFlatL2 *quantizer = nullptr;
    faiss::IndexIVFPQ *ivfpq = nullptr;
    double ivf_train_s = 0, ivf_add_s = 0, ivf_size_mb = 0;

    for (const Setting &s : grid)
    {
        if (dim % s.M != 0)
        {
            std::cerr << "Skipping M=" << s.M << " (does not divide dim=" << dim << ")\n";
            continue;
        }
        std::cout << "M=" << s.M << " nbits=" << s.nbits << " nlist=" << s.nlist
                  << " nprobe=" << s.nprobe << "\n";

        if (pq_key != std::make_tuple(s.M, s.nbits))
        {
            faiss::IndexPQ pq(dim, s.M, s.nbits, faiss::METRIC_L2);
            auto t0 = std::chrono::steady_clock::now();
            pq.train(nt, train.data);
            pq_res.train_s = seconds_since(t0);
            t0 = std::chrono::steady_clock::now();
            pq.add(ndb, db.data());
            pq_res.add_s = seconds_since(t0);
            search_recalls(pq, pq_res.search_s, pq_res.recalls);
            pq_res.size_mb = (double)pq.codes.size() / (1024.0 * 1024.0);
            pq_key = {s.M, s.nbits};
        }

        const bool do_ivf = runs_ivfpq(s);
        if (do_ivf && ivf_key != std::make_tuple(s.M, s.nbits, s.nlist))
        {
            delete ivfpq;
            delete quantizer;
            quantizer = new faiss::IndexFlatL2(dim);
            ivfpq = new faiss::IndexIVFPQ(quantizer, dim, s.nlist, s.M, s.nbits, faiss::METRIC_L2);
            ivfpq->by_residual = by_residual;
            auto t0 = std::chrono::steady_clock::now();
            ivfpq->train(nt, train.data);
            ivf_train_s = seconds_since(t0);
            t0 = std::chrono::steady_clock::now();
            ivfpq->add(ndb, db.data());
            ivf_add_s = seconds_since(t0);
            ivf_size_mb = (double)ndb * ivfpq->code_size / (1024.0 * 1024.0);
            ivf_key = {s.M, s.nbits, s.nlist};
        }
        double ivf_search_s = 0;
        std::vector<double> ivf_recalls;
        if (do_ivf)
        {
            ivfpq->nprobe = s.nprobe;
            search_recalls(*ivfpq, ivf_search_s, ivf_recalls);
        }

        csv << s.M << ',' << s.nbits << ',' << (1 << s.nbits);
        for (double r : pq_res.recalls)
            csv << ',' << r;
        csv << ',' << pq_res.train_s << ',' << pq_res.add_s << ',' << pq_res.search_s
            << ',' << nq / pq_res.search_s << ',' << pq_res.size_mb;
        if (do_ivf)
        {
            csv << ',' << s.nlist << ',' << s.nprobe;
            for (double r : ivf_recalls)
                csv << ',' << r;
            csv << ',' << ivf_train_s << ',' << ivf_add_s << ',' << ivf_search_s
                << ',' << nq / ivf_search_s << ',' << ivf_size_mb << ',' << by_residual;
        }
        else
            csv << std::string(16, ','); // blank IVFPQ columns, through "By residual"
        csv << ',' << nt << ',' << ndb << ',' << nq << '\n';
        csv.flush();
        std::cout << "  PQ JD-GT R@10=" << pq_res.recalls[5];
        if (do_ivf)
            std::cout << "  IVFPQ JD-GT R@10=" << ivf_recalls[5] << "  IVFPQ QPS=" << nq / ivf_search_s;
        std::cout << "\n";
    }

    delete ivfpq;
    delete quantizer;
    return 0;
}
