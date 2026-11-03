// Step 0: sanity checks of the sketchers on a dataset's held-out ("train") queries.
//   sanity <dataset_dir> <out_dir> [n_seeds=5] [n_bias_seeds=40] [base_seed=20261014]
//          [algos=dart,ficws,ficws_raw,bmh2] [file_tag=""]
// Writes <out_dir>/step0_unit.csv, step0_estimator_raw.csv (all k, scales, bit
// widths; seeds base..base+n_seeds-1) and step0_bias_raw.csv (k in {64,256},
// scale 1, seeds base+1000.. ; more replicates to test for bias).
// Evaluation queries are never touched here.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "csr.hpp"
#include "gt.hpp"
#include "sketch.hpp"

using namespace wmh;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Pair {
    int32_t q, r;
    double J;
    int type;  // 0 = nn (exact top-100), 1 = rank-stratified, 2 = random, 3 = disjoint copy (J = 0)
};
static const char* kPairType[] = {"nn", "strat", "random", "disjoint"};
constexpr int kNumPairTypes = 4;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: sanity <dataset_dir> <out_dir> [n_seeds] [n_bias_seeds] [base_seed]\n"); return 1; }
    std::string dir = argv[1], out = argv[2];
    int n_seeds = argc > 3 ? std::atoi(argv[3]) : 5;
    int n_bias_seeds = argc > 4 ? std::atoi(argv[4]) : 40;
    uint64_t base_seed = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 20261014ull;
    std::string algo_list = argc > 6 ? argv[6] : "dart,ficws,ficws_raw,bmh2";
    std::string tag = argc > 7 ? argv[7] : "";
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // progress visible when redirected to a log
    double t0 = now();
    Csr X = load_csr(dir + "/db.npz");
    Csr T = load_csr(dir + "/train.npz");
    std::printf("[%.1fs] db n=%lld d=%lld nnz=%zu | train n=%lld | threads=%d\n", now() - t0, (long long)X.n,
                (long long)X.d, X.indices.size(), (long long)T.n, omp_get_max_threads());

    std::vector<Algo> algos;
    for (size_t p = 0, e; p < algo_list.size(); p = e + 1) {
        e = algo_list.find(',', p);
        if (e == std::string::npos) e = algo_list.size();
        algos.push_back(parse_algo(algo_list.substr(p, e - p)));
    }
    auto seeded = [](Algo, int) { return true; };
    const std::vector<int> bits_list = {8, 16, 32, 64};

    // ---------------------------------------------------------------- unit checks
    FILE* fu = std::fopen((out + "/step0_unit" + tag + ".csv").c_str(), "w");
    std::fprintf(fu, "algo,k,check,n,mean,min,max,expected\n");
    std::mt19937_64 prng(base_seed);
    std::vector<int64_t> sample;
    for (int i = 0; i < 2000; i++) sample.push_back((int64_t)(prng() % X.n));
    for (Algo a : algos)
        for (int k : {16, 64, 256}) {
            Sketcher S(a, k, base_seed), S2(a, k, base_seed);
            std::vector<uint64_t> h1(k), h2(k), h3(k);
            auto emit = [&](const char* check, const std::vector<double>& v, double expected) {
                double mn = 1e300, mx = -1e300, s = 0;
                for (double x : v) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
                std::fprintf(fu, "%s,%d,%s,%zu,%.6f,%.6f,%.6f,%.6f\n", algo_name(a).c_str(), k, check, v.size(),
                             s / v.size(), mn, mx, expected);
                std::printf("  %-9s k=%-4d %-24s mean=%9.4f min=%9.4f max=%9.4f expected=%.4f\n", algo_name(a).c_str(), k,
                            check, s / v.size(), mn, mx, expected);
            };
            std::vector<double> same, fresh, disj64, disj8, half;
            for (int64_t r : sample) {
                std::vector<int32_t> idx(X.idx(r), X.idx(r) + X.nnz(r)), shifted(idx);
                std::vector<double> val(X.val(r), X.val(r) + X.nnz(r)), twice(val);
                for (auto& c : shifted) c += (int32_t)X.d;
                for (auto& w : twice) w *= 2;
                S.sketch(idx.data(), val.data(), idx.size(), h1.data());
                S.sketch(idx.data(), val.data(), idx.size(), h2.data());
                S2.sketch(idx.data(), val.data(), idx.size(), h3.data());
                same.push_back(count_matches(h1.data(), h2.data(), k));
                fresh.push_back(count_matches(h1.data(), h3.data(), k));
                S.sketch(shifted.data(), val.data(), idx.size(), h2.data());
                disj64.push_back(count_matches(h1.data(), h2.data(), k));
                int m8 = 0;
                for (int j = 0; j < k; j++) m8 += truncate_id(h1[j], 8) == truncate_id(h2[j], 8);
                disj8.push_back(m8);
                S.sketch(idx.data(), twice.data(), idx.size(), h2.data());
                half.push_back((double)count_matches(h1.data(), h2.data(), k) / k);
            }
            emit("identical_same_object", same, k);
            emit("identical_fresh_object", fresh, k);
            emit("disjoint_matches_64bit", disj64, 0);
            emit("disjoint_matches_8bit", disj8, k / 256.0);
            emit("est_x_vs_2x(J=0.5)", half, 0.5);
            // thread determinism + single-thread sketch time
            omp_set_num_threads(1);
            double ts = now();
            auto a1 = sketch_rows(S, X, &sample);
            double per_vec_us = (now() - ts) / sample.size() * 1e6;
            omp_set_num_threads(omp_get_num_procs());
            auto aN = sketch_rows(S, X, &sample);
            double diff = 0;
            for (size_t i = 0; i < a1.size(); i++) diff += a1[i] != aN[i];
            emit("slots_differing_1_vs_N_threads", {diff}, 0);
            emit("sketch_us_per_vector_1thread", {per_vec_us}, NAN);
        }
    std::fclose(fu);
    std::printf("[%.1fs] unit checks done\n", now() - t0);

    // --------------------------------------------------- exact scores for train queries
    const int K = 100;
    const std::vector<int64_t> strat_ranks = {200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000};
    const int n_random = 20;
    std::vector<std::vector<Pair>> per_q(T.n);
    InvertedIndex inv(X);
#pragma omp parallel
    {
        std::vector<double> acc(X.n), sc;
#pragma omp for schedule(dynamic, 1)
        for (int64_t q = 0; q < T.n; q++) {
            inv.scores(T, q, X, acc, sc);
            std::vector<Scored> all(X.n);
            for (int64_t r = 0; r < X.n; r++) all[r] = {sc[r], (int32_t)r};
            std::sort(all.begin(), all.end(), better);
            for (int j = 0; j < K; j++) per_q[q].push_back({(int32_t)q, all[j].id, all[j].s, 0});
            for (int64_t rk : strat_ranks)
                if (rk <= X.n) per_q[q].push_back({(int32_t)q, all[rk - 1].id, all[rk - 1].s, 1});
            std::mt19937_64 g(base_seed * 7919 + q);
            for (int j = 0; j < n_random; j++) {
                int32_t r = (int32_t)(g() % X.n);
                per_q[q].push_back({(int32_t)q, r, sc[r], 2});
            }
        }
    }
    std::vector<Pair> pairs;
    for (auto& v : per_q) pairs.insert(pairs.end(), v.begin(), v.end());
    std::printf("[%.1fs] exact scores done, %zu pairs\n", now() - t0, pairs.size());

    // cross-check inverted-index J against merge-based J and brute-force top-K on 100 queries
    double max_abs = 0;
    int topk_mismatch = 0;
#pragma omp parallel for reduction(max : max_abs) reduction(+ : topk_mismatch) schedule(dynamic, 1)
    for (int64_t q = 0; q < std::min<int64_t>(100, T.n); q++) {
        auto bt = brute_topk(T, q, X, K);
        for (int j = 0; j < K; j++) {
            const Pair& p = per_q[q][j];
            topk_mismatch += (bt[j].id != p.r) || (bt[j].s != p.J);
            max_abs = std::max(max_abs, std::fabs(jw_direct(T, q, X, p.r) - p.J));
        }
    }
    std::printf("  brute-force cross-check (100 queries): top-%d mismatches=%d, max|J_formula - J_direct|=%.3g\n", K,
                topk_mismatch, max_abs);

    // ------------------------------------------------------------- estimator accuracy
    std::vector<int64_t> rows;
    std::vector<int32_t> slot_of(X.n, -1);
    for (auto& p : pairs)
        if (slot_of[p.r] < 0) { slot_of[p.r] = (int32_t)rows.size(); rows.push_back(p.r); }
    std::vector<int64_t> trows(T.n);
    for (int64_t q = 0; q < T.n; q++) trows[q] = q;
    // disjoint-support copies of the train queries (coordinates shifted by d), J_w = 0 exactly
    Csr Tshift = T;
    Tshift.d = 2 * T.d;
    for (auto& c : Tshift.indices) c += (int32_t)T.d;
    std::vector<Pair> disjoint;
    for (int64_t q = 0; q < T.n; q++) disjoint.push_back({(int32_t)q, (int32_t)q, 0.0, 3});

    const char* header = "scale,seed,algo,k,bits,corrected,pair_type,bucket,n,sum_J,sum_est,sum_err,sum_abs_err,sum_sq_err,sum_var_unit\n";
    FILE* fe = std::fopen((out + "/step0_estimator_raw" + tag + ".csv").c_str(), "w");
    std::fprintf(fe, "%s", header);
    const int NB = 10;
    struct Acc { double n = 0, J = 0, est = 0, err = 0, abs = 0, sq = 0, var = 0; };
    auto run = [&](FILE* fo, const std::vector<int>& bl, double scale, uint64_t seed, Algo a, int k) {
        Sketcher S(a, k, seed);
        auto hx = sketch_rows(S, X, &rows, scale);
        auto hq = sketch_rows(S, T, &trows, scale);
        auto hs = sketch_rows(S, Tshift, &trows, scale);
        for (int bits : bl)
            for (int corr = 0; corr <= (bits < 64 ? 1 : 0); corr++) {
                std::vector<Acc> acc(kNumPairTypes * NB);
                std::vector<uint64_t> tq(k), tx(k);
                for (int set = 0; set < 2; set++)
                for (auto& p : set == 0 ? pairs : disjoint) {
                    const uint64_t* a1 = &hq[(size_t)p.q * k];
                    const uint64_t* b1 = p.type == 3 ? &hs[(size_t)p.r * k] : &hx[(size_t)slot_of[p.r] * k];
                    for (int j = 0; j < k; j++) { tq[j] = truncate_id(a1[j], bits); tx[j] = truncate_id(b1[j], bits); }
                    double e = estimate_jw(count_matches(tq.data(), tx.data(), k), k, bits, corr);
                    int b = std::min(NB - 1, (int)(p.J * NB));
                    Acc& A = acc[p.type * NB + b];
                    A.n++; A.J += p.J; A.est += e; A.err += e - p.J; A.abs += std::fabs(e - p.J);
                    A.sq += (e - p.J) * (e - p.J); A.var += p.J * (1 - p.J);
                }
                for (int t = 0; t < kNumPairTypes; t++)
                    for (int b = 0; b < NB; b++) {
                        const Acc& A = acc[t * NB + b];
                        if (A.n == 0) continue;
                        std::fprintf(fo, "%g,%llu,%s,%d,%d,%d,%s,%d,%.0f,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", scale,
                                     (unsigned long long)seed, algo_name(a).c_str(), k, bits, corr, kPairType[t], b, A.n,
                                     A.J, A.est, A.err, A.abs, A.sq, A.var);
                    }
            }
        if (a == Algo::FastICWS && S.small_weight_count())
            std::printf("  WARNING ficws scale=%g: %llu weights below 1 after internal scaling\n", scale,
                        (unsigned long long)S.small_weight_count());
    };
    for (int s = 0; s < n_seeds; s++) {
        uint64_t seed = base_seed + s;
        for (Algo a : algos)
            if (seeded(a, s))
                for (int k : {16, 32, 64, 128, 256, 512, 1024}) run(fe, bits_list, 1.0, seed, a, k);
        for (double scale : {1e-6, 1e-3, 1e3, 1e6})
            for (Algo a : algos)
                if (seeded(a, s))
                    for (int k : {64, 256}) run(fe, bits_list, scale, seed, a, k);
        std::printf("[%.1fs] seed %llu done\n", now() - t0, (unsigned long long)seed);
    }
    std::fclose(fe);

    // more independent replicates (fresh seeds) at scale 1 to test for bias
    FILE* fb = std::fopen((out + "/step0_bias_raw" + tag + ".csv").c_str(), "w");
    std::fprintf(fb, "%s", header);
    for (int s = 0; s < n_bias_seeds; s++) {
        for (Algo a : algos)
            for (int k : {64, 256}) run(fb, {8, 64}, 1.0, base_seed + 1000 + s, a, k);
        if ((s + 1) % 10 == 0) std::printf("[%.1fs] bias seeds %d/%d\n", now() - t0, s + 1, n_bias_seeds);
    }
    std::fclose(fb);
    return 0;
}
