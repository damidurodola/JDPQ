#pragma once

// In-place compaction of a database file (dropping flagged rows), shared by
// extract_train.cpp (duplicate-ID removal) and remove_train.cpp (training-row
// removal). Works on the text layout ("N D\n id v0 ... v(D-1)\n ...") and the
// convert_to_binary layout (<prefix>.meta.txt / .ids.bin / .vecs.bin).
//
// NOT crash-safe: an interrupted run leaves the file corrupt. Callers create
// a "<db>.dedup_in_progress" marker around the call and refuse to run while it
// exists.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace db_compact {

struct Bitmap {
    std::vector<uint64_t> words;
    explicit Bitmap(size_t nbits) : words((nbits + 63) / 64, 0) {}
    bool test(size_t i) const { return (words[i >> 6] >> (i & 63)) & 1ULL; }
    void set(size_t i) { words[i >> 6] |= 1ULL << (i & 63); }
};

inline bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Opens `path` read-write and maps it shared. Returns nullptr on failure.
inline char* map_rw(const std::string& path, size_t& size, int& fd) {
    fd = open(path.c_str(), O_RDWR);
    if (fd < 0) {
        perror(("Error opening " + path).c_str());
        return nullptr;
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror(("Error stat'ing " + path).c_str());
        close(fd);
        return nullptr;
    }
    size = (size_t)sb.st_size;
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        perror(("Error mapping " + path).c_str());
        close(fd);
        return nullptr;
    }
    madvise(addr, size, MADV_SEQUENTIAL);
    return static_cast<char*>(addr);
}

inline bool finish_rw(char* base, size_t map_size, int fd, size_t new_size, const std::string& path) {
    bool ok = munmap(base, map_size) == 0;
    ok = ok && ftruncate(fd, (off_t)new_size) == 0;
    ok = ok && fsync(fd) == 0;
    close(fd);
    if (!ok) perror(("Error finalizing " + path).c_str());
    return ok;
}

// Compacts a file of `n` fixed-size records in place, dropping the records
// flagged in `dropped`. Kept records only ever move toward the front, so
// memmove over contiguous kept runs is safe.
inline bool compact_fixed(const std::string& path, size_t rec_bytes, size_t n,
                   const Bitmap& dropped, size_t new_n) {
    size_t size = 0;
    int fd = -1;
    char* base = map_rw(path, size, fd);
    if (!base) return false;
    if (size != n * rec_bytes) {
        std::cerr << path << ": expected " << n * rec_bytes << " bytes, found " << size << "\n";
        munmap(base, size);
        close(fd);
        return false;
    }
    size_t w = 0, run_start = 0;
    bool in_run = false;
    for (size_t i = 0; i <= n; ++i) {
        const bool keep = (i < n) && !dropped.test(i);
        if (keep && !in_run) {
            run_start = i;
            in_run = true;
        } else if (!keep && in_run) {
            const size_t len = i - run_start;
            if (w != run_start) std::memmove(base + w * rec_bytes, base + run_start * rec_bytes, len * rec_bytes);
            w += len;
            in_run = false;
        }
    }
    if (w != new_n) {
        std::cerr << path << ": internal error, kept " << w << " rows, expected " << new_n << "\n";
        munmap(base, size);
        close(fd);
        return false;
    }
    return finish_rw(base, size, fd, new_n * rec_bytes, path);
}

inline bool compact_binary(const std::string& prefix, size_t n, size_t dim,
                    const Bitmap& dropped, size_t new_n) {
    if (!compact_fixed(prefix + ".vecs.bin", dim * sizeof(float), n, dropped, new_n)) return false;
    if (!compact_fixed(prefix + ".ids.bin", sizeof(int), n, dropped, new_n)) return false;
    // meta last: until now a size mismatch against the old meta flags a partial run.
    std::ofstream meta(prefix + ".meta.txt");
    meta << new_n << " " << dim << "\n";
    return meta.good();
}

// Text variant. Row i's "chunk" is its bytes from its first token up to the
// first token of row i+1 (i.e. including trailing whitespace); kept chunks are
// copied verbatim, so float text is never re-formatted.
inline bool compact_text(const std::string& path, size_t n, size_t dim,
                  const Bitmap& dropped, size_t new_n) {
    size_t size = 0;
    int fd = -1;
    char* base = map_rw(path, size, fd);
    if (!base) return false;
    const char* end = base + size;
    char* p = base;
    auto skip_ws = [&]() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p; };
    auto skip_token = [&]() { skip_ws(); while (p < end && *p > ' ') ++p; };

    skip_token(); skip_token(); skip_ws(); // header "N D" -> p at first row
    const std::string new_header = std::to_string(new_n) + " " + std::to_string(dim) + "\n";
    if (new_header.size() > (size_t)(p - base)) {
        std::cerr << "New header does not fit in the old one; leaving " << path << " untouched.\n";
        munmap(base, size);
        close(fd);
        return false;
    }
    // Everything below modifies the file.
    std::memcpy(base, new_header.data(), new_header.size());
    size_t w = new_header.size();
    for (size_t i = 0; i < n; ++i) {
        char* row_start = p;
        if (row_start >= end) {
            std::cerr << path << ": truncated at row " << i << "\n";
            munmap(base, size);
            close(fd);
            return false;
        }
        for (size_t t = 0; t <= dim; ++t) skip_token(); // id + dim components
        skip_ws();
        if (!dropped.test(i)) {
            const size_t len = (size_t)(p - row_start);
            std::memmove(base + w, row_start, len);
            w += len;
        }
    }
    return finish_rw(base, size, fd, w, path);
}


} // namespace db_compact
