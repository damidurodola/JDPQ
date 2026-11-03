#pragma once

#include <vector>
#include <string>

namespace faiss_mimic {

void read_data(const std::string& filename, int& num_vectors, int& dim, std::vector<float>& data, std::vector<int>& ids);

// Parses only per-row IDs and skips the float values. Memory cost is O(num_vectors) ints
// instead of O(num_vectors * dim) floats — use this for ID-only paths like polygon GT.
void read_ids_only(const std::string& filename, int& num_vectors, int& dim, std::vector<int>& ids);

// Read-only view over a dataset produced by convert_to_binary.cpp: a fixed
// record layout mmap'd straight from disk instead of copied into a
// std::vector. Residency is handled by the OS page cache, so datasets far
// larger than physical RAM (e.g. 1B x 256 floats ~= 1TB) can be trained/added
// from directly, as long as no caller copies the whole thing into a vector.
struct MappedDataset {
    int num_vectors = 0;
    int dim = 0;
    const float* data = nullptr; // num_vectors x dim, row-major
    const int* ids = nullptr;    // num_vectors

    MappedDataset() = default;
    MappedDataset(const MappedDataset&) = delete;
    MappedDataset& operator=(const MappedDataset&) = delete;
    MappedDataset(MappedDataset&& other) noexcept;
    MappedDataset& operator=(MappedDataset&& other) noexcept;
    ~MappedDataset();

    void* vecs_map_ = nullptr;
    size_t vecs_map_size_ = 0;
    void* ids_map_ = nullptr;
    size_t ids_map_size_ = 0;
};

// Loads "<prefix>.meta.txt" / "<prefix>.vecs.bin" / "<prefix>.ids.bin" as
// written by convert_to_binary.cpp. Returns false (logging to stderr) on any
// I/O error or size mismatch.
bool read_data_binary(const std::string& prefix, MappedDataset& out);

} // namespace faiss_mimic
