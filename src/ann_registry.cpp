// Public part of the method registry: the exact brute-force reference method
// ("flat") and the dispatch to the method plugin.

#include "AnnIndex.h"
#include "IndexFlat.h"

namespace faiss_mimic {

namespace {

// Exact search under the spec's metric. Recall is ~1.0 against the driver's own
// ground truth (a fraction of a percent can be lost where several vectors are
// exactly equidistant from a query and the two searches keep different ones), which
// makes it a sanity check that the harness is wired correctly, and a template for
// adding a method.
class FlatAnn : public AnnIndex
{
public:
    explicit FlatAnn(const IndexSpec &spec) : index_(spec.dim, spec.metric) {}

    void train(idx_t, const float *) override {}
    // Keeps the pointer (no copy): x must outlive this object.
    void add(idx_t n, const float *x) override { index_.add_no_copy(n, x); }
    void search(idx_t nq, const float *queries, int R, const SearchParams &,
                idx_t *out_pos, float *out_dist) override
    {
        index_.search(nq, queries, R, out_dist, out_pos);
    }
    size_t memory_usage_bytes() const override { return index_.memory_usage_bytes(); }

private:
    IndexFlat index_;
};

} // namespace

std::unique_ptr<AnnIndex> make_index(const IndexSpec &spec)
{
    if (spec.kind == "flat")
        return std::make_unique<FlatAnn>(spec);
    return make_jdpq_index(spec);
}

} // namespace faiss_mimic
