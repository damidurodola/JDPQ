// Exact weighted-Jaccard top-K via a sparse inverted index, plus brute force.
// Ties are broken by smaller database row index (total order: J desc, row asc),
// so both methods return identical lists.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "csr.hpp"

namespace wmh {

struct Scored {
    double s;
    int32_t id;
};
// Strict total order: higher score first, then smaller row index.
inline bool better(const Scored& a, const Scored& b) { return a.s > b.s || (a.s == b.s && a.id < b.id); }

// Exact top-K of an array of N scores (K > N allowed: returns all, sorted).
inline std::vector<Scored> topk_of(const std::vector<double>& scores, int K) {
    std::vector<Scored> all(scores.size());
    for (size_t r = 0; r < scores.size(); r++) all[r] = {scores[r], (int32_t)r};
    size_t k = std::min<size_t>(K, all.size());
    std::nth_element(all.begin(), all.begin() + k - 1, all.end(), better);
    all.resize(k);
    std::sort(all.begin(), all.end(), better);
    return all;
}

struct InvertedIndex {
    int64_t n = 0;
    std::vector<int64_t> ptr;   // d+1 offsets into rows/vals
    std::vector<int32_t> rows;  // ascending within each posting list
    std::vector<double> vals;

    explicit InvertedIndex(const Csr& X) : n(X.n), ptr(X.d + 1, 0) {
        for (int32_t c : X.indices) ptr[c + 1]++;
        for (int64_t c = 0; c < X.d; c++) ptr[c + 1] += ptr[c];
        rows.resize(X.indices.size());
        vals.resize(X.indices.size());
        std::vector<int64_t> fill(ptr.begin(), ptr.end() - 1);
        for (int64_t r = 0; r < X.n; r++)
            for (int64_t p = X.indptr[r]; p < X.indptr[r + 1]; p++) {
                int64_t at = fill[X.indices[p]]++;
                rows[at] = (int32_t)r;
                vals[at] = X.data[p];
            }
    }

    // J_w(q, x_r) for every database row r. `acc` is caller-provided scratch of size n.
    void scores(const Csr& Q, int64_t q, const Csr& X, std::vector<double>& acc, std::vector<double>& out) const {
        std::fill(acc.begin(), acc.end(), 0.0);
        const int32_t* qi = Q.idx(q);
        const double* qv = Q.val(q);
        for (int64_t a = 0; a < Q.nnz(q); a++) {  // ascending coordinate order
            int32_t c = qi[a];
            if (c >= (int32_t)ptr.size() - 1) continue;
            double w = qv[a];
            for (int64_t p = ptr[c]; p < ptr[c + 1]; p++) acc[rows[p]] += w < vals[p] ? w : vals[p];
        }
        out.resize(n);
        for (int64_t r = 0; r < n; r++) out[r] = jw_from_parts(acc[r], Q.l1[q], X.l1[r]);
    }
};

struct GroundTruth {
    int64_t nq = 0;
    int K = 0;
    std::vector<int32_t> ids;  // nq*K
    std::vector<double> sims;  // nq*K
};

inline GroundTruth gt_inverted(const Csr& Q, const Csr& X, int K) {
    InvertedIndex inv(X);
    GroundTruth g;
    g.nq = Q.n;
    g.K = K;
    g.ids.resize(Q.n * K);
    g.sims.resize(Q.n * K);
#pragma omp parallel
    {
        std::vector<double> acc(X.n), sc;
#pragma omp for schedule(dynamic, 4)
        for (int64_t q = 0; q < Q.n; q++) {
            inv.scores(Q, q, X, acc, sc);
            auto top = topk_of(sc, K);
            for (int j = 0; j < K; j++) { g.ids[q * K + j] = top[j].id; g.sims[q * K + j] = top[j].s; }
        }
    }
    return g;
}

inline std::vector<Scored> brute_topk(const Csr& Q, int64_t q, const Csr& X, int K) {
    std::vector<double> sc(X.n);
    for (int64_t r = 0; r < X.n; r++) sc[r] = jw(Q, q, X, r);
    return topk_of(sc, K);
}

// Cheap content fingerprint of a matrix (FNV-1a over the CSR arrays), stored in GT caches.
inline uint64_t fingerprint(const Csr& M) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t len) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < len; i++) { h ^= b[i]; h *= 1099511628211ull; }
    };
    mix(&M.n, 8); mix(&M.d, 8);
    mix(M.indptr.data(), M.indptr.size() * 8);
    mix(M.indices.data(), M.indices.size() * 4);
    mix(M.data.data(), M.data.size() * 8);
    return h;
}

inline void save_gt(const std::string& path, const GroundTruth& g, uint64_t fp_q, uint64_t fp_x) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    const char magic[8] = {'W', 'M', 'H', 'G', 'T', '1', 0, 0};
    int64_t K = g.K;
    std::fwrite(magic, 1, 8, f);
    std::fwrite(&g.nq, 8, 1, f); std::fwrite(&K, 8, 1, f);
    std::fwrite(&fp_q, 8, 1, f); std::fwrite(&fp_x, 8, 1, f);
    std::fwrite(g.ids.data(), 4, g.ids.size(), f);
    std::fwrite(g.sims.data(), 8, g.sims.size(), f);
    std::fclose(f);
}

// Returns false if missing or if the fingerprints do not match the given data.
inline bool load_gt(const std::string& path, GroundTruth& g, uint64_t fp_q, uint64_t fp_x) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    int64_t nq, K;
    uint64_t a, b;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::string(magic, 6) == "WMHGT1" && std::fread(&nq, 8, 1, f) == 1 &&
              std::fread(&K, 8, 1, f) == 1 && std::fread(&a, 8, 1, f) == 1 && std::fread(&b, 8, 1, f) == 1 &&
              a == fp_q && b == fp_x;
    if (ok) {
        g.nq = nq; g.K = (int)K;
        g.ids.resize(nq * K); g.sims.resize(nq * K);
        ok = std::fread(g.ids.data(), 4, g.ids.size(), f) == g.ids.size() &&
             std::fread(g.sims.data(), 8, g.sims.size(), f) == g.sims.size();
    }
    std::fclose(f);
    return ok;
}

}  // namespace wmh
