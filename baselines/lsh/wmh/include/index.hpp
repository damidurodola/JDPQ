// Retrieval building blocks: stored b-bit sketches, bounded top-C selection,
// banded LSH tables, and recall.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include "sketch.hpp"

namespace wmh {

// ----------------------------------------------------------------- stored sketches
// Row-major N x k matrix of b-bit slot values (T = uint8_t / uint16_t / uint64_t).
// One row is k contiguous values, so a row-vs-query comparison is a vectorizable
// loop, and a row read from memory serves a whole batch of queries.
template <class T>
struct StoredSketches {
    int64_t n = 0;
    int k = 0;
    int bits = 0;
    std::vector<T> own;      // storage (empty when aliasing the 64-bit ids)
    const T* data = nullptr;

    const T* row(int64_t i) const { return data + (size_t)i * k; }
    // Nominal size n*k*bits/8; slots narrower than 8 bits are held in whole bytes in RAM but counted at their width.
    size_t bytes() const { return ((size_t)n * k * bits + 7) / 8; }
};

template <class T>
StoredSketches<T> make_stored(const std::vector<uint64_t>& ids64, int64_t n, int k, int bits) {
    StoredSketches<T> s;
    s.n = n; s.k = k; s.bits = bits;
    if (sizeof(T) == 8 && bits == 64) {
        s.data = reinterpret_cast<const T*>(ids64.data());
        return s;
    }
    s.own.resize((size_t)n * k);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)((size_t)n * k); i++) s.own[i] = (T)truncate_id(ids64[i], bits);
    s.data = s.own.data();
    return s;
}

template <class T>
inline void truncate_into(const uint64_t* in, int k, int bits, T* out) {
    for (int j = 0; j < k; j++) out[j] = (T)truncate_id(in[j], bits);
}

template <class T>
inline int count_eq(const T* __restrict a, const T* __restrict b, int k) {
    int m = 0;
    for (int j = 0; j < k; j++) m += a[j] == b[j];
    return m;
}

// ----------------------------------------------------------------- top-C selection
struct Cand {
    int32_t score;  // number of matching slots
    int32_t id;     // database row
};
// Total order: more matches first, then smaller row id (deterministic ties).
inline bool cand_better(const Cand& a, const Cand& b) { return a.score > b.score || (a.score == b.score && a.id < b.id); }

// Keeps the C best candidates under cand_better. Buffered selection: candidates
// better than the current cut-off are appended (O(1)); when the buffer reaches
// 2C it is trimmed back to the best C with nth_element and the cut-off rises.
// This makes the cost depend little on C (a heap would pay log C per insert).
struct TopC {
    int C = 0;
    std::vector<Cand> buf;
    bool have_cut = false;  // true once a trim has established the C-th best so far
    Cand cut{-1, 0};        // worst of the kept C after the last trim
    explicit TopC(int c = 0) { reset(c); }
    void reset(int c) { C = c; buf.clear(); buf.reserve(2 * (size_t)std::max(c, 1)); have_cut = false; }
    // Score a new candidate must exceed when ids arrive in increasing order
    // (an equal score with a larger id is never better than the cut-off).
    int32_t threshold() const { return have_cut ? cut.score : -1; }
    void trim() {
        std::nth_element(buf.begin(), buf.begin() + (C - 1), buf.end(), cand_better);
        buf.resize(C);
        cut = *std::max_element(buf.begin(), buf.end(), cand_better);  // worst kept = max under "better"
        have_cut = true;
    }
    void push(Cand c) {
        if (C <= 0) return;
        if (have_cut && !cand_better(c, cut)) return;
        buf.push_back(c);
        if ((int)buf.size() >= 2 * C) trim();
    }
    std::vector<Cand> sorted() const {
        std::vector<Cand> v(buf);
        size_t m = std::min(v.size(), (size_t)C);
        std::partial_sort(v.begin(), v.begin() + m, v.end(), cand_better);
        v.resize(m);
        return v;
    }
};

// ----------------------------------------------------------------- banded LSH
// Bucket key of band l: hash of its r slot ids. Callers pass the ids at the stored slot
// width (truncate_all), so a b-bit index buckets on b-bit ids exactly like its sketches.
// Order-sensitive; any key collision only adds
// candidates, which are then ranked by match count.
inline std::vector<uint64_t> truncate_all(const std::vector<uint64_t>& ids, int bits) {
    std::vector<uint64_t> t(ids.size());
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)ids.size(); i++) t[i] = truncate_id(ids[i], bits);
    return t;
}

inline uint64_t band_key(const uint64_t* ids, int r, int band) {
    uint64_t h = mix64(0x9e3779b97f4a7c15ull * (uint64_t)(band + 1));
    for (int j = 0; j < r; j++) h = mix64(h ^ (ids[j] + 0x632be59bd9b4e019ull * (uint64_t)(j + 1)));
    return h;
}

struct BandTable {
    std::vector<uint64_t> keys;   // distinct bucket keys
    std::vector<uint32_t> offs;   // keys.size()+1 offsets into ids
    std::vector<int32_t> ids;     // rows, ascending within a bucket
    std::vector<uint32_t> slots;  // open addressing: bucket index + 1, 0 = empty
    uint64_t mask = 0;

    // Accounting convention: posting ids are counted as 8 bytes (int64, as in faiss
    // inverted lists) although they are stored as int32 here.
    size_t bytes() const {
        return keys.size() * 8 + offs.size() * 4 + ids.size() * 8 + slots.size() * 4;
    }
    // Returns [begin, end) of the rows in the bucket with this key (empty if none).
    std::pair<const int32_t*, const int32_t*> lookup(uint64_t key) const {
        for (uint64_t p = key & mask;; p = (p + 1) & mask) {
            uint32_t s = slots[p];
            if (s == 0) return {nullptr, nullptr};
            if (keys[s - 1] == key) return {ids.data() + offs[s - 1], ids.data() + offs[s]};
        }
    }
};

struct BandIndex {
    int L = 0, r = 0, k = 0;
    std::vector<BandTable> tables;
    size_t bytes() const { size_t b = 0; for (auto& t : tables) b += t.bytes(); return b; }
};

inline BandIndex build_band_index(const std::vector<uint64_t>& ids64, int64_t n, int k, int L, int r) {
    BandIndex B;
    B.L = L; B.r = r; B.k = k;
    B.tables.resize(L);
#pragma omp parallel for schedule(dynamic, 1)
    for (int l = 0; l < L; l++) {
        std::vector<std::pair<uint64_t, int32_t>> kv(n);
        for (int64_t i = 0; i < n; i++) kv[i] = {band_key(ids64.data() + (size_t)i * k + (size_t)l * r, r, l), (int32_t)i};
        std::sort(kv.begin(), kv.end());
        BandTable& t = B.tables[l];
        t.ids.resize(n);
        for (int64_t i = 0; i < n; i++) {
            if (i == 0 || kv[i].first != kv[i - 1].first) { t.keys.push_back(kv[i].first); t.offs.push_back((uint32_t)i); }
            t.ids[i] = kv[i].second;
        }
        t.offs.push_back((uint32_t)n);
        t.keys.shrink_to_fit(); t.offs.shrink_to_fit();
        size_t cap = 1;
        while (cap < 2 * t.keys.size()) cap <<= 1;
        t.slots.assign(cap, 0);
        t.mask = cap - 1;
        for (size_t b = 0; b < t.keys.size(); b++) {
            uint64_t p = t.keys[b] & t.mask;
            while (t.slots[p]) p = (p + 1) & t.mask;
            t.slots[p] = (uint32_t)(b + 1);
        }
    }
    return B;
}

// Union of the query's buckets over all bands. `stamp` is per-thread scratch of
// size n (initialized to 0) and `epoch` must be distinct for every call.
inline void band_candidates(const BandIndex& B, const uint64_t* q64, std::vector<uint32_t>& stamp, uint32_t epoch,
                            std::vector<int32_t>& out) {
    out.clear();
    for (int l = 0; l < B.L; l++) {
        auto rg = B.tables[l].lookup(band_key(q64 + (size_t)l * B.r, B.r, l));
        for (const int32_t* p = rg.first; p != rg.second; p++)
            if (stamp[*p] != epoch) { stamp[*p] = epoch; out.push_back(*p); }
    }
}

// ----------------------------------------------------------------- recall
// Recall@R = |returned[0:R] ∩ truth[0:R]| / R  (truth = exact top-K, K >= R).
inline double recall_at(const int32_t* returned, int n_returned, const int32_t* truth, int R) {
    std::vector<int32_t> t(truth, truth + R);
    std::sort(t.begin(), t.end());
    int hit = 0;
    for (int i = 0; i < std::min(R, n_returned); i++) hit += std::binary_search(t.begin(), t.end(), returned[i]);
    return (double)hit / R;
}

// Tie-aware Recall@R: a returned item counts if its exact J_w is >= the R-th
// true J_w (so an item tied with the R-th neighbour is not a miss).
inline double recall_at_tie_aware(const double* returned_J, int n_returned, const double* truth_J, int R) {
    int hit = 0;
    for (int i = 0; i < std::min(R, n_returned); i++) hit += returned_J[i] >= truth_J[R - 1];
    return (double)std::min(hit, R) / R;
}

}  // namespace wmh
