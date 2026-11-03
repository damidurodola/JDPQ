// Stand-in for the JD-PQ implementation, which is released upon acceptance.
// Replace this file with the released jdpq_index.cpp (plus the core sources it
// needs) to enable the "pq" and "ivfpq" methods.

#include "AnnIndex.h"

namespace faiss_mimic {

std::unique_ptr<AnnIndex> make_jdpq_index(const IndexSpec &)
{
    return nullptr;
}

} // namespace faiss_mimic
