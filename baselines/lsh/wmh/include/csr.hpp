// CSR sparse matrix (loaded from the canonical .npz) and exact weighted Jaccard.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "npz.hpp"

namespace wmh {

struct Csr {
    int64_t n = 0, d = 0;
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;  // sorted ascending within each row
    std::vector<double> data;      // strictly positive
    std::vector<double> l1;        // per-row L1 norm, summed in index order

    int64_t nnz(int64_t i) const { return indptr[i + 1] - indptr[i]; }
    const int32_t* idx(int64_t i) const { return indices.data() + indptr[i]; }
    const double* val(int64_t i) const { return data.data() + indptr[i]; }

    void finalize() {
        l1.assign(n, 0.0);
        for (int64_t i = 0; i < n; i++) {
            double s = 0;
            for (int64_t p = indptr[i]; p < indptr[i + 1]; p++) {
                if (!(data[p] > 0)) throw std::runtime_error("csr: nonpositive stored value");
                if (p > indptr[i] && indices[p] <= indices[p - 1]) throw std::runtime_error("csr: unsorted indices");
                s += data[p];
            }
            l1[i] = s;
        }
    }
};

inline Csr load_csr(const std::string& path) {
    auto a = load_npz(path);
    Csr m;
    auto shape = a.at("shape").as<int64_t>();
    m.n = shape.at(0);
    m.d = shape.at(1);
    m.indptr = a.at("indptr").as<int64_t>();
    m.indices = a.at("indices").as<int32_t>();
    m.data = a.at("data").as<double>();
    if ((int64_t)m.indptr.size() != m.n + 1 || (int64_t)m.indices.size() != m.indptr.back() ||
        m.data.size() != m.indices.size())
        throw std::runtime_error(path + ": inconsistent CSR arrays");
    m.finalize();
    return m;
}

// sum_i min(x_i, y_i) by merging the two sorted supports; contributions are
// added in ascending coordinate order (the inverted-index ground truth adds
// them in the same order, so both give bit-identical results).
inline double sum_min(const int32_t* xi, const double* xv, int64_t nx, const int32_t* yi, const double* yv, int64_t ny) {
    double s = 0;
    int64_t a = 0, b = 0;
    while (a < nx && b < ny) {
        if (xi[a] == yi[b]) { s += xv[a] < yv[b] ? xv[a] : yv[b]; a++; b++; }
        else if (xi[a] < yi[b]) a++;
        else b++;
    }
    return s;
}

// J_w = sum min / sum max, with sum max = |x|_1 + |y|_1 - sum min.
inline double jw_from_parts(double smin, double l1x, double l1y) {
    double smax = l1x + l1y - smin;
    return smax > 0 ? smin / smax : 0.0;
}

inline double jw(const Csr& X, int64_t i, const Csr& Y, int64_t j) {
    double s = sum_min(X.idx(i), X.val(i), X.nnz(i), Y.idx(j), Y.val(j), Y.nnz(j));
    return jw_from_parts(s, X.l1[i], Y.l1[j]);
}

// Direct definition (independent code path, used only in tests).
inline double jw_direct(const Csr& X, int64_t i, const Csr& Y, int64_t j) {
    std::vector<double> x(X.d, 0.0), y(Y.d, 0.0);
    for (int64_t p = 0; p < X.nnz(i); p++) x[X.idx(i)[p]] = X.val(i)[p];
    for (int64_t p = 0; p < Y.nnz(j); p++) y[Y.idx(j)[p]] = Y.val(j)[p];
    double mn = 0, mx = 0;
    for (int64_t c = 0; c < X.d; c++) { mn += x[c] < y[c] ? x[c] : y[c]; mx += x[c] > y[c] ? x[c] : y[c]; }
    return mx > 0 ? mn / mx : 0.0;
}

}  // namespace wmh
