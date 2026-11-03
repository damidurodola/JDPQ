#include "IndexFlat.h"
#include "distances.h"
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <limits>
#include <queue>
#include <utility>
#include <iostream>

namespace faiss_mimic {

static const char *metric_name(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::Jaccard:
        case DistanceMetric::ShapeJaccard:
            return "Jaccard";
        case DistanceMetric::L2:
            return "L2";
        default:
            return "Unknown";
    }
}

IndexFlat::IndexFlat(int d, DistanceMetric metric) : Index(d), metric(metric) {}

IndexFlat::~IndexFlat() {}

void IndexFlat::train(idx_t n, const float* x) {
    // Flat index does not need training
    is_trained = true;
}

void IndexFlat::add(idx_t n, const float* x) {
    xb.insert(xb.end(), x, x + (size_t)n * d);
    if (metric == DistanceMetric::Jaccard) {
        xb_sums.reserve(xb_sums.size() + (size_t)n);
        for (idx_t i = 0; i < n; ++i) {
            xb_sums.push_back(fvec_l1sum(x + (size_t)i * d, d));
        }
    }
    ntotal += n;
}

void IndexFlat::add_no_copy(idx_t n, const float* x) {
    xb_ext = x;
    if (metric == DistanceMetric::Jaccard) {
        xb_sums.reserve(xb_sums.size() + (size_t)n);
        for (idx_t i = 0; i < n; ++i) {
            xb_sums.push_back(fvec_l1sum(x + (size_t)i * d, d));
        }
    }
    ntotal += n;
}

void IndexFlat::search(idx_t nq, const float* xq, int k, float* distances, idx_t* labels) {
    if (ntotal == 0) {
        for (idx_t i = 0; i < nq; ++i) {
            float* dists = distances ? distances + (size_t)i * k : nullptr;
            idx_t* ids = labels + (size_t)i * k;
            for (int j = 0; j < k; ++j) {
                if (dists) {
                    dists[j] = std::numeric_limits<float>::infinity();
                }
                ids[j] = -1;
            }
        }
        return;
    }
// std::cout << "IndexFlat metric: " << metric_name(metric) << std::endl;

    const size_t top = std::min((size_t)k, (size_t)ntotal);
    using Pair = std::pair<float, idx_t>;
    auto cmp = [](const Pair& a, const Pair& b) { return a.first < b.first; };
    using Heap = std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)>;

    // One heap per query, persisted across database blocks below. Each
    // heap's underlying vector is pre-reserved to `top` -- its hard max
    // size, since heap.size() never exceeds top by construction below --
    // so it never reallocates once the parallel region starts. ThreadSanitizer
    // flagged a race inside std::vector's own growth/iterator machinery
    // here (vector::end()/reallocation on the heap's backing store) under
    // heavy repeated invocation (PQ training calls this hundreds of times);
    // pre-reserving removes every reallocation during concurrent access,
    // which is the operation that was racing, independent of whether the
    // underlying cause was an application bug or a libgomp/TSan interaction.
    std::vector<Heap> heaps;
    heaps.reserve(nq);
    for (idx_t i = 0; i < nq; ++i) {
        std::vector<Pair> backing;
        backing.reserve(top);
        heaps.emplace_back(cmp, std::move(backing));
    }

    std::vector<float> query_sums;
    if (metric == DistanceMetric::Jaccard) {
        query_sums.resize(nq);
        for (idx_t i = 0; i < nq; ++i) {
            query_sums[i] = fvec_l1sum(xq + (size_t)i * d, d);
        }
    }

    const float* xb_base = data_ptr();

    // Database-outer, query-inner, in blocks: every db vector is read
    // exactly once across the whole search regardless of nq. The original
    // query-outer loop had each query independently rescan all of xb, which
    // is fine when xb fits in RAM but means nq full passes over it when it
    // doesn't -- e.g. add_no_copy()'s mmap'd billion-vector ground truth
    // case, where xb is ~1TB against ~500GB RAM: nq=3000 full rescans would
    // have meant petabytes of disk I/O instead of one ~1TB pass.
    constexpr idx_t kBlock = 5000000; // ~5GB/block at d=256
    for (idx_t j0 = 0; j0 < ntotal; j0 += kBlock) {
        const idx_t j1 = std::min(j0 + kBlock, ntotal);
#pragma omp parallel for schedule(dynamic)
        for (idx_t i = 0; i < nq; ++i) {
            const float* query = xq + (size_t)i * d;
            const float query_sum = (metric == DistanceMetric::Jaccard) ? query_sums[i] : 0.0f;
            Heap& heap = heaps[i];
            for (idx_t j = j0; j < j1; ++j) {
                const float* base = xb_base + (size_t)j * d;
                float dist = (metric == DistanceMetric::Jaccard)
                                 ? fvec_jaccard_presummed(query, base, d, query_sum, xb_sums[j])
                                 : (metric == DistanceMetric::ShapeJaccard)
                                       ? fvec_shape_jaccard(query, base, d)
                                       : fvec_L2sqr(query, base, d);
                if (heap.size() < top) {
                    heap.emplace(dist, j);
                } else if (dist < heap.top().first) {
                    heap.pop();
                    heap.emplace(dist, j);
                }
            }
        }
    }

    for (idx_t i = 0; i < nq; ++i) {
        float* dists = distances ? distances + (size_t)i * k : nullptr;
        idx_t* ids = labels + (size_t)i * k;
        Heap& heap = heaps[i];

        std::vector<Pair> top_results(top);
        for (size_t idx = top; idx-- > 0;) {
            top_results[idx] = heap.top();
            heap.pop();
        }

        size_t j = 0;
        for (; j < top; ++j) {
            if (dists) {
                dists[j] = top_results[j].first;
            }
            ids[j] = top_results[j].second;
        }
        for (; j < (size_t)k; ++j) {
            if (dists) {
                dists[j] = std::numeric_limits<float>::infinity();
            }
            ids[j] = -1;
        }
    }
}

void IndexFlat::reset() {
    xb.clear();
    xb_ext = nullptr;
    xb_sums.clear();
    ntotal = 0;
}

const float* IndexFlat::get_xb() const {
    return data_ptr();
}

size_t IndexFlat::memory_usage_bytes() const {
    return (xb.size() + xb_sums.size()) * sizeof(float);
}

} // namespace faiss_mimic
