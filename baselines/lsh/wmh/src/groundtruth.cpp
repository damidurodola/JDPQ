// Step 1: exact weighted-Jaccard top-K ground truth.
//   groundtruth <dataset_dir> [K=100] [n_check=200] [seed=20261014]
// For split in {query, train}: computes exact top-K over db.npz with the
// inverted index, cross-checks n_check random queries against brute force
// (ids and J values must agree exactly), and caches <dataset_dir>/gt_<split>.bin
// plus gt_<split>.json (timings, tie statistics, cross-check result).
// Ties are broken by smaller database row index.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "csr.hpp"
#include "gt.hpp"

using namespace wmh;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) { std::fprintf(stderr, "usage: groundtruth <dataset_dir> [K] [n_check] [seed]\n"); return 1; }
    std::string dir = argv[1];
    int K = argc > 2 ? std::atoi(argv[2]) : 100;
    int n_check = argc > 3 ? std::atoi(argv[3]) : 200;
    uint64_t seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 20261014ull;
    Csr X = load_csr(dir + "/db.npz");
    uint64_t fpx = fingerprint(X);
    InvertedIndex inv(X);
    int fails = 0;
    for (std::string split : {"query", "train"}) {
        Csr Q = load_csr(dir + "/" + split + ".npz");
        uint64_t fpq = fingerprint(Q);
        GroundTruth g;
        g.nq = Q.n; g.K = K;
        g.ids.resize(Q.n * K); g.sims.resize(Q.n * K);
        std::vector<int64_t> tied_beyond(Q.n, 0);
        double t0 = now();
#pragma omp parallel
        {
            std::vector<double> acc(X.n), sc;
#pragma omp for schedule(dynamic, 4)
            for (int64_t q = 0; q < Q.n; q++) {
                inv.scores(Q, q, X, acc, sc);
                auto top = topk_of(sc, K);
                for (int j = 0; j < K; j++) { g.ids[q * K + j] = top[j].id; g.sims[q * K + j] = top[j].s; }
                // rows outside the returned top-K whose J equals the K-th value (boundary ties)
                int64_t at_k = 0;
                for (double s : sc) at_k += s == top[K - 1].s;
                int64_t in_top = 0;
                for (int j = 0; j < K; j++) in_top += top[j].s == top[K - 1].s;
                tied_beyond[q] = at_k - in_top;
            }
        }
        double t_gt = now() - t0;

        // brute-force cross-check on random queries
        std::mt19937_64 rng(seed);
        std::vector<int64_t> check;
        for (int64_t q = 0; q < Q.n; q++) check.push_back(q);
        std::shuffle(check.begin(), check.end(), rng);
        check.resize(std::min<int64_t>(n_check, Q.n));
        int64_t mism = 0;
        t0 = now();
#pragma omp parallel for reduction(+ : mism) schedule(dynamic, 1)
        for (size_t c = 0; c < check.size(); c++) {
            int64_t q = check[c];
            auto bt = brute_topk(Q, q, X, K);
            for (int j = 0; j < K; j++) mism += bt[j].id != g.ids[q * K + j] || bt[j].s != g.sims[q * K + j];
        }
        double t_brute = now() - t0;

        int64_t q_ties = 0, n_ties = 0, exact_dups = 0;
        double j1 = 0, j10 = 0, j100 = 0;
        for (int64_t q = 0; q < Q.n; q++) {
            q_ties += tied_beyond[q] > 0;
            n_ties += tied_beyond[q];
            exact_dups += g.sims[q * K] == 1.0;
            j1 += g.sims[q * K]; j10 += g.sims[q * K + std::min(9, K - 1)]; j100 += g.sims[q * K + K - 1];
        }
        save_gt(dir + "/gt_" + split + ".bin", g, fpq, fpx);
        FILE* f = std::fopen((dir + "/gt_" + split + ".json").c_str(), "w");
        std::fprintf(f,
                     "{\n  \"split\": \"%s\", \"n_queries\": %lld, \"n_db\": %lld, \"K\": %d, \"threads\": %d,\n"
                     "  \"gt_seconds\": %.3f, \"tie_break\": \"J desc, then db row index asc\",\n"
                     "  \"bruteforce_checked_queries\": %zu, \"bruteforce_mismatching_entries\": %lld, \"bruteforce_seconds\": %.3f,\n"
                     "  \"queries_with_ties_across_rank_K\": %lld, \"db_rows_tied_with_rank_K_but_excluded\": %lld,\n"
                     "  \"queries_with_exact_copy_in_db\": %lld,\n"
                     "  \"mean_J_rank1\": %.6f, \"mean_J_rank10\": %.6f, \"mean_J_rankK\": %.6f\n}\n",
                     split.c_str(), (long long)Q.n, (long long)X.n, K, omp_get_max_threads(), t_gt, check.size(),
                     (long long)mism, t_brute, (long long)q_ties, (long long)n_ties, (long long)exact_dups, j1 / Q.n,
                     j10 / Q.n, j100 / Q.n);
        std::fclose(f);
        std::printf("%s/%s: nq=%lld gt %.1fs | brute-force check %zu queries: %lld mismatching entries (%.1fs) | "
                    "queries with rank-%d ties: %lld (%lld excluded tied rows) | exact copies: %lld | mean J@1=%.4f J@10=%.4f J@%d=%.4f\n",
                    dir.c_str(), split.c_str(), (long long)Q.n, t_gt, check.size(), (long long)mism, t_brute, K,
                    (long long)q_ties, (long long)n_ties, (long long)exact_dups, j1 / Q.n, j10 / Q.n, K, j100 / Q.n);
        if (mism || exact_dups) fails++;
    }
    return fails ? 2 : 0;
}
