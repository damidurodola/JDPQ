// Weighted MinHash sketchers wrapping the upstream dartminhash code.
// Upstream headers define non-inline functions, so they are included only in
// src/sketch.cpp; everything else goes through this interface.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "csr.hpp"

namespace wmh {

enum class Algo {
    Dart,         // upstream DartMinHash
    FastICWS,     // upstream FastICWS_t, weights multiplied by 2^40 inside the method (see README)
    FastICWSRaw,  // upstream FastICWS_t on the raw weights (diagnostic only: negative t_k is UB)
    BagMinHash2,  // upstream BagMinHash2 wrapper (only if built with WMH_WITH_BAGMINHASH)
};
std::string algo_name(Algo a);
Algo parse_algo(const std::string& s);

// Scale applied inside FastICWS so that every stored weight w satisfies
// w * scale >= 1, which keeps upstream's t_k >= 0 (it casts t_k to uint64_t).
constexpr double kFastICWSScale = 1099511627776.0;  // 2^40

class Sketcher {
   public:
    // One object per (algo, k, seed); the same object must sketch queries and database.
    Sketcher(Algo algo, int k, uint64_t seed);
    ~Sketcher();
    Sketcher(const Sketcher&) = delete;
    Sketcher& operator=(const Sketcher&) = delete;

    Algo algo() const { return algo_; }
    int k() const { return k_; }
    uint64_t seed() const { return seed_; }

    // Writes k 64-bit hash ids for the sparse vector (idx, val). Throws on an
    // empty vector. `scale` multiplies every weight first (used only by the
    // Step 0 scale-invariance check). Safe to call concurrently.
    void sketch(const int32_t* idx, const double* val, int64_t nnz, uint64_t* out, double scale = 1.0) const;

    // Number of weights seen so far (FastICWS only) with w * scale * 2^40 < 1,
    // i.e. where a negative t_k is still possible. Expected to stay 0.
    uint64_t small_weight_count() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Algo algo_;
    int k_;
    uint64_t seed_;
};

// Sketch rows of M (all rows, or the listed ones) in parallel; row-major n x k.
std::vector<uint64_t> sketch_rows(const Sketcher& s, const Csr& M, const std::vector<int64_t>* rows = nullptr,
                                  double scale = 1.0);

// Bijective 64-bit mixer (splitmix64 finalizer), applied before truncation.
inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
// Stored slot value for a b-bit slot: low b bits of mix64(id); b = 64 keeps the id as is.
inline uint64_t truncate_id(uint64_t id, int bits) {
    return bits >= 64 ? id : (mix64(id) & ((1ull << bits) - 1));
}

// Number of slots j with a[j] == b[j] (slot-by-slot comparison).
inline int count_matches(const uint64_t* a, const uint64_t* b, int k) {
    int m = 0;
    for (int j = 0; j < k; j++) m += a[j] == b[j];
    return m;
}

// Fraction of matching slots, optionally corrected for b-bit false matches:
// E[p_hat] = J + (1 - J) 2^-b  =>  J_hat = (p_hat - 2^-b) / (1 - 2^-b).
inline double estimate_jw(int matches, int k, int bits, bool corrected) {
    double p = (double)matches / k;
    if (!corrected || bits >= 64) return p;
    double f = 1.0 / (double)(1ull << bits);
    return (p - f) / (1 - f);
}

}  // namespace wmh
