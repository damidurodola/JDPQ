#pragma once

// Method-agnostic interface between the benchmark driver (benchmark.cpp) and an
// approximate-nearest-neighbour method. The driver owns data loading, exact
// ground truth, re-ranking, recall and the CSV; a method only has to build an
// index and answer searches.


#include "distances.h"
#include "index.h" // idx_t

#include <cstddef>
#include <memory>
#include <string>

namespace faiss_mimic {

struct IndexSpec
{
    std::string kind;   // "flat" (public reference), or a method name such as "pq" / "ivfpq"
    int dim = 0;
    int M = 0;          // subspaces (quantization methods)
    int nbits = 0;      // bits per subspace
    size_t nlist = 0;   // inverted lists (IVF methods)
    DistanceMetric metric = DistanceMetric::Jaccard;
};

struct SearchParams
{
    int nprobe = 1; // inverted lists visited (IVF methods); ignored otherwise
};

class AnnIndex
{
public:
    virtual ~AnnIndex() = default;

    virtual void train(idx_t n, const float *x) = 0;
    virtual void add(idx_t n, const float *x) = 0;
    // Returns the R best positions per query in out_pos (nq*R, -1 padded) and their
    // method-internal distances in out_dist (nq*R); both row-major by query.
    virtual void search(idx_t nq, const float *queries, int R, const SearchParams &params,
                        idx_t *out_pos, float *out_dist) = 0;

    // Bytes the index would occupy persisted (codes, ids if the method stores them,
    // codebooks, coarse centroids)
    virtual size_t memory_usage_bytes() const = 0;

    // Optional diagnostics.
    virtual double avg_candidates_scanned() const { return 0.0; } // per query, last search
    virtual int clustering_niter() const { return 0; }            // 0 = not applicable
    virtual std::string build_log() const { return ""; }          // extra lines for the run log
};

// Returns nullptr if `spec.kind` is not available in this build.
std::unique_ptr<AnnIndex> make_index(const IndexSpec &spec);

// Provided by the method plugin (the JD-PQ implementation), or by a stub that
// returns nullptr when the plugin is not part of the build.
std::unique_ptr<AnnIndex> make_jdpq_index(const IndexSpec &spec);

} // namespace faiss_mimic
