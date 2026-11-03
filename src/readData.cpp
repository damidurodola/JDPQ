#include "readData.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <charconv>
#include <utility>

namespace faiss_mimic {

void read_data(const std::string& filename, int& num_vectors, int& dim, std::vector<float>& data, std::vector<int>& ids) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("Error opening file");
        return;
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error reading filesize");
        close(fd);
        return;
    }

    size_t filesize = sb.st_size;

    // Map file into memory
    char* addr = static_cast<char*>(
        mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0));
    if (addr == MAP_FAILED) {
        perror("error mapping file");
        close(fd);
        return;
    }

    close(fd); // fd no longer needed

    char* ptr = addr;
    char* end = addr + filesize;

    auto nextInt = [&](int& out) {
        while (ptr < end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t'))
            ptr++;
        char* start = ptr;
        while (ptr < end && *ptr > ' ')
            ptr++;
        std::from_chars(start, ptr, out);
    };

    auto nextFloat = [&](float& out) {
        while (ptr < end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t'))
            ptr++;
        char* start = ptr;
        while (ptr < end && *ptr > ' ')
            ptr++;
        std::string token(start, ptr);
        out = std::stof(token); // Use stof for float
    };

    // Read header: num_vectors dim
    nextInt(num_vectors);
    nextInt(dim);

    data.resize((size_t)num_vectors * dim, 0.0);
    ids.reserve(num_vectors);

    for (int i = 0; i < num_vectors; i++) {
        int id;
        nextInt(id);
        ids.push_back(id);

        for (int j = 0; j < dim; j++) {
            float val;
            nextFloat(val);
            // Zero out components below the noise floor once, here, so every
            // downstream distance call sees an exact 0 rather than a tiny
            // nonzero value. This makes the "effectively zero" decision a
            // per-vector fact instead of a per-pair one (fvec_jaccard's mask
            // used to test xi+yi against this same threshold jointly), which
            // is what makes precomputed per-vector sums exactly cacheable.
            if (std::fabs(val) < 1e-6f) {
                val = 0.0f;
            }
            data[size_t(i) * dim + j] = val;
        }
    }

    // Unmap file
    munmap(addr, filesize);
}

void read_ids_only(const std::string& filename, int& num_vectors, int& dim, std::vector<int>& ids) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("Error opening file");
        return;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error reading filesize");
        close(fd);
        return;
    }

    size_t filesize = sb.st_size;

    char* addr = static_cast<char*>(
        mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0));
    if (addr == MAP_FAILED) {
        perror("error mapping file");
        close(fd);
        return;
    }

    close(fd);

    char* ptr = addr;
    char* end = addr + filesize;

    auto nextInt = [&](int& out) {
        while (ptr < end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t'))
            ptr++;
        char* start = ptr;
        while (ptr < end && *ptr > ' ')
            ptr++;
        std::from_chars(start, ptr, out);
    };

    auto skipToken = [&]() {
        while (ptr < end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t'))
            ptr++;
        while (ptr < end && *ptr > ' ')
            ptr++;
    };

    nextInt(num_vectors);
    nextInt(dim);

    ids.clear();
    ids.reserve(num_vectors);

    for (int i = 0; i < num_vectors; i++) {
        int id;
        nextInt(id);
        ids.push_back(id);
        for (int j = 0; j < dim; j++) skipToken();
    }

    munmap(addr, filesize);
}

MappedDataset::MappedDataset(MappedDataset&& other) noexcept {
    *this = std::move(other);
}

MappedDataset& MappedDataset::operator=(MappedDataset&& other) noexcept {
    if (this == &other) return *this;
    if (vecs_map_) munmap(vecs_map_, vecs_map_size_);
    if (ids_map_) munmap(ids_map_, ids_map_size_);

    num_vectors = other.num_vectors;
    dim = other.dim;
    data = other.data;
    ids = other.ids;
    vecs_map_ = other.vecs_map_;
    vecs_map_size_ = other.vecs_map_size_;
    ids_map_ = other.ids_map_;
    ids_map_size_ = other.ids_map_size_;

    other.num_vectors = 0;
    other.dim = 0;
    other.data = nullptr;
    other.ids = nullptr;
    other.vecs_map_ = nullptr;
    other.vecs_map_size_ = 0;
    other.ids_map_ = nullptr;
    other.ids_map_size_ = 0;
    return *this;
}

MappedDataset::~MappedDataset() {
    if (vecs_map_) munmap(vecs_map_, vecs_map_size_);
    if (ids_map_) munmap(ids_map_, ids_map_size_);
}

static bool mmap_readonly(const std::string& path, size_t expected_size, void*& out_addr) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        perror(("Error opening " + path).c_str());
        return false;
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror(("Error stat'ing " + path).c_str());
        close(fd);
        return false;
    }
    if ((size_t)sb.st_size != expected_size) {
        std::cerr << path << ": expected " << expected_size << " bytes, found "
                  << sb.st_size << "\n";
        close(fd);
        return false;
    }
    void* addr = mmap(nullptr, expected_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // mapping stays valid after fd is closed
    if (addr == MAP_FAILED) {
        perror(("Error mapping " + path).c_str());
        return false;
    }
    out_addr = addr;
    return true;
}

bool read_data_binary(const std::string& prefix, MappedDataset& out) {
    std::ifstream meta(prefix + ".meta.txt");
    if (!meta.is_open()) {
        std::cerr << "Failed to open " << (prefix + ".meta.txt") << "\n";
        return false;
    }
    int num_vectors = 0, dim = 0;
    meta >> num_vectors >> dim;
    if (num_vectors <= 0 || dim <= 0) {
        std::cerr << "Invalid metadata in " << (prefix + ".meta.txt") << "\n";
        return false;
    }

    const size_t vecs_bytes = (size_t)num_vectors * (size_t)dim * sizeof(float);
    const size_t ids_bytes = (size_t)num_vectors * sizeof(int);

    void* vecs_addr = nullptr;
    if (!mmap_readonly(prefix + ".vecs.bin", vecs_bytes, vecs_addr)) return false;

    void* ids_addr = nullptr;
    if (!mmap_readonly(prefix + ".ids.bin", ids_bytes, ids_addr)) {
        munmap(vecs_addr, vecs_bytes);
        return false;
    }

    out.num_vectors = num_vectors;
    out.dim = dim;
    out.data = static_cast<const float*>(vecs_addr);
    out.ids = static_cast<const int*>(ids_addr);
    out.vecs_map_ = vecs_addr;
    out.vecs_map_size_ = vecs_bytes;
    out.ids_map_ = ids_addr;
    out.ids_map_size_ = ids_bytes;
    return true;
}

} // namespace faiss_mimic
