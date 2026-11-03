#include "index.h"
#include <stdexcept>

namespace faiss_mimic {

void Index::add(idx_t n, const float* x) {
    // Default implementation for add without IDs
    add_with_ids(n, x, nullptr);
}

void Index::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    // This is a virtual method. If an index does not support add_with_ids,
    // it should not be called. We throw an error.
    throw std::runtime_error("add_with_ids is not implemented for this index type.");
}

size_t Index::memory_usage_bytes() const {
    return 0;
}

} // namespace faiss_mimic
