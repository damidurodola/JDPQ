#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace faiss_mimic {

using idx_t = int64_t;

class Index {
public:
    int d; // dimension of vectors
    idx_t ntotal; // number of vectors
    bool is_trained;

    Index(int d = 0) : d(d), ntotal(0), is_trained(true) {}
    virtual ~Index() {}

    virtual void train(idx_t n, const float* x) = 0;
    // Add vectors to the index with optional IDs.
    virtual void add_with_ids(idx_t n, const float* x, const idx_t* xids);
    // Compatibility add method
    virtual void add(idx_t n, const float* x);
    virtual void search(idx_t nq, const float* xq, int k, float* distances, idx_t* labels) = 0;
    virtual void reset() = 0;

    // Get a pointer to the indexed vectors.
    virtual const float* get_xb() const { return nullptr; }

    // Rough estimate of persistent memory usage in bytes.
    virtual size_t memory_usage_bytes() const;
};

} // namespace faiss_mimic
