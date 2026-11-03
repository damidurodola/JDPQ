// The only translation unit that includes upstream dartminhash headers
// (they define non-inline functions and must not be included twice).
#include "sketch.hpp"

#include <atomic>
#include <random>
#include <stdexcept>
#include <utility>

#include "dartminhash.hpp"
#include "icws.hpp"
#ifdef WMH_WITH_BAGMINHASH
#include "bagminhash_wrappers.hpp"
#endif

namespace wmh {

std::string algo_name(Algo a) {
    switch (a) {
        case Algo::Dart: return "dart";
        case Algo::FastICWS: return "ficws";
        case Algo::FastICWSRaw: return "ficws_raw";
        case Algo::BagMinHash2: return "bmh2";
    }
    return "?";
}

Algo parse_algo(const std::string& s) {
    if (s == "dart") return Algo::Dart;
    if (s == "ficws") return Algo::FastICWS;
    if (s == "ficws_raw") return Algo::FastICWSRaw;
    if (s == "bmh2") return Algo::BagMinHash2;
    throw std::runtime_error("unknown algorithm " + s);
}

struct Sketcher::Impl {
    // Held by unique_ptr and never moved: FastICWS keeps references into
    // FastICWS_t's own gamma tables, so the FastICWS_t object must not move.
    std::unique_ptr<DartMinHash> dart;
    std::unique_ptr<FastICWS_t> ficws;
#ifdef WMH_WITH_BAGMINHASH
    std::unique_ptr<BagMinHash2> bmh2;
#endif
    mutable std::atomic<uint64_t> small_weights{0};
};

Sketcher::Sketcher(Algo algo, int k, uint64_t seed) : impl_(new Impl), algo_(algo), k_(k), seed_(seed) {
    if (k < 1) throw std::runtime_error("k must be positive");
    std::mt19937_64 rng(seed);
    switch (algo) {
        case Algo::Dart: impl_->dart.reset(new DartMinHash(rng, (uint64_t)k)); break;
        case Algo::FastICWS:
        case Algo::FastICWSRaw: impl_->ficws.reset(new FastICWS_t(rng, (uint64_t)k)); break;
        case Algo::BagMinHash2:
#ifdef WMH_WITH_BAGMINHASH
            // Upstream hash seeds are fixed; the seed enters through coordinate relabeling in sketch().
            impl_->bmh2.reset(new BagMinHash2((uint64_t)k));
            break;
#else
            throw std::runtime_error("built without WMH_WITH_BAGMINHASH");
#endif
    }
}

Sketcher::~Sketcher() = default;

uint64_t Sketcher::small_weight_count() const { return impl_->small_weights.load(); }

void Sketcher::sketch(const int32_t* idx, const double* val, int64_t nnz, uint64_t* out, double scale) const {
    if (nnz <= 0) throw std::runtime_error("cannot sketch an empty vector");
    double s = scale;
    if (algo_ == Algo::FastICWS) s *= kFastICWSScale;
    // BagMinHash2: seed-dependent bijective relabeling of coordinate ids, giving
    // independent hash functions per seed without touching upstream (J_w unchanged).
    const bool relabel = algo_ == Algo::BagMinHash2;
    const uint64_t salt = mix64(seed_ ^ 0x5bd1e9955bd1e995ull);
    std::vector<std::pair<uint64_t, double>> x(nnz);
    uint64_t small = 0;
    for (int64_t p = 0; p < nnz; p++) {
        uint64_t id = (uint64_t)idx[p];
        x[p] = {relabel ? mix64(id ^ salt) : id, val[p] * s};
        small += x[p].second < 1.0;
    }
    if (algo_ == Algo::FastICWS && small) impl_->small_weights += small;

    // The upstream call operators are non-const but only read their tables.
    std::vector<std::pair<uint64_t, double>> h;
    switch (algo_) {
        case Algo::Dart: h = (*impl_->dart)(x); break;
        case Algo::FastICWS:
        case Algo::FastICWSRaw: h = (*impl_->ficws)(x); break;
        case Algo::BagMinHash2:
#ifdef WMH_WITH_BAGMINHASH
            h = (*impl_->bmh2)(x);
#endif
            break;
    }
    if ((int)h.size() != k_) throw std::runtime_error("upstream returned wrong sketch length");
    for (int j = 0; j < k_; j++) out[j] = h[j].first;
}

std::vector<uint64_t> sketch_rows(const Sketcher& s, const Csr& M, const std::vector<int64_t>* rows, double scale) {
    int64_t n = rows ? (int64_t)rows->size() : M.n;
    int k = s.k();
    std::vector<uint64_t> out((size_t)n * k);
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t t = 0; t < n; t++) {
        int64_t r = rows ? (*rows)[t] : t;
        s.sketch(M.idx(r), M.val(r), M.nnz(r), out.data() + (size_t)t * k, scale);
    }
    return out;
}

}  // namespace wmh
