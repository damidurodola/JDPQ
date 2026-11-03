#pragma once

#include "index.h"
#include "distances.h"
#include <vector>

namespace faiss_mimic {

class IndexFlat : public Index {
protected:
    std::vector<float> xb; // database vectors (owned copy, from add())
    const float* xb_ext = nullptr; // non-owning database pointer, from add_no_copy()
    std::vector<float> xb_sums; // per-vector sum(x[i]), Jaccard metric only
    DistanceMetric metric;

    const float* data_ptr() const { return xb_ext ? xb_ext : xb.data(); }

public:
    IndexFlat(int d, DistanceMetric metric = DistanceMetric::Jaccard);
    ~IndexFlat() override;

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    // Registers n vectors backed by external, caller-owned storage (which
    // must outlive this index and every search() call against it) instead
    // of copying them into xb. For brute-force search over datasets too
    // large to duplicate in RAM -- e.g. billion-vector ground truth, where
    // add()'s xb.insert() copy would need ~1TB it doesn't have. Do not mix
    // with add() on the same index.
    void add_no_copy(idx_t n, const float* x);
    void search(idx_t nq, const float* xq, int k, float* distances, idx_t* labels) override;
    void reset() override;
    const float* get_xb() const override;
    size_t memory_usage_bytes() const override;
    DistanceMetric get_metric() const { return metric; }
};

} // namespace faiss_mimic
