// Streams a large "N D\n id v0 v1 ... v(D-1)\n ..." ASCII dataset file into
// a fixed-record binary layout that readData.h's read_data_binary() can
// mmap directly:
//   <prefix>.meta.txt  -> "N D" (text)
//   <prefix>.ids.bin   -> N x int32, raw
//   <prefix>.vecs.bin  -> N x D x float32, raw, row-major
//
// This exists so that datasets far larger than physical RAM (e.g. the
// zinc20 1B-vector file, ~1TB as raw float32) never have to be materialized
// in a single in-process buffer: the converter itself only ever holds one
// bounded flush buffer, and downstream consumers mmap the binary output
// instead of loading it into a std::vector.
//
// Usage: ./convert_to_binary <input.txt> <output_prefix>

#include <charconv>
#include <chrono>
#include <cmath>
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

namespace {

// Number of rows buffered in memory before being flushed to disk. Bounds
// this tool's own memory use regardless of dataset size.
constexpr size_t kFlushRows = 500000;

struct Cursor {
    const char* ptr;
    const char* end;

    void skip_ws() {
        while (ptr < end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t' || *ptr == '\r'))
            ++ptr;
    }

    bool next_int(int& out) {
        skip_ws();
        const char* start = ptr;
        while (ptr < end && *ptr > ' ') ++ptr;
        if (start == ptr) return false;
        auto res = std::from_chars(start, ptr, out);
        return res.ec == std::errc();
    }

    bool next_float(float& out) {
        skip_ws();
        const char* start = ptr;
        while (ptr < end && *ptr > ' ') ++ptr;
        if (start == ptr) return false;
        auto res = std::from_chars(start, ptr, out);
        return res.ec == std::errc();
    }
};

bool write_all(FILE* f, const void* data, size_t bytes) {
    return std::fwrite(data, 1, bytes, f) == bytes;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <output_prefix>\n";
        return 1;
    }
    const std::string input_path = argv[1];
    const std::string prefix = argv[2];

    int fd = open(input_path.c_str(), O_RDONLY);
    if (fd < 0) {
        perror(("Error opening " + input_path).c_str());
        return 1;
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error stat'ing input file");
        close(fd);
        return 1;
    }
    const size_t filesize = (size_t)sb.st_size;
    void* addr = mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        perror("Error mmap'ing input file");
        return 1;
    }
    // Sequential single pass over the whole file; hint the kernel so it
    // read-aheads and drops pages behind us instead of trying to cache the
    // entire 740GB file.
    madvise(addr, filesize, MADV_SEQUENTIAL);

    Cursor cur{static_cast<const char*>(addr), static_cast<const char*>(addr) + filesize};

    int num_vectors = 0, dim = 0;
    if (!cur.next_int(num_vectors) || !cur.next_int(dim) || num_vectors <= 0 || dim <= 0) {
        std::cerr << "Failed to parse header (N, D) from " << input_path << "\n";
        munmap(addr, filesize);
        return 1;
    }
    std::cout << "Converting " << num_vectors << " vectors of dim " << dim << "\n";

    // Metadata written up front — vecs/ids sizes are already known.
    {
        std::ofstream meta(prefix + ".meta.txt");
        if (!meta.is_open()) {
            std::cerr << "Failed to open " << (prefix + ".meta.txt") << " for writing\n";
            munmap(addr, filesize);
            return 1;
        }
        meta << num_vectors << " " << dim << "\n";
    }

    FILE* vecs_out = std::fopen((prefix + ".vecs.bin").c_str(), "wb");
    FILE* ids_out = std::fopen((prefix + ".ids.bin").c_str(), "wb");
    if (!vecs_out || !ids_out) {
        perror("Failed to open output binary files");
        if (vecs_out) std::fclose(vecs_out);
        if (ids_out) std::fclose(ids_out);
        munmap(addr, filesize);
        return 1;
    }

    std::vector<int> id_buf(kFlushRows);
    std::vector<float> vec_buf((size_t)kFlushRows * dim);
    size_t buf_rows = 0;

    const auto t0 = std::chrono::steady_clock::now();
    auto flush = [&]() -> bool {
        if (buf_rows == 0) return true;
        bool ok = write_all(ids_out, id_buf.data(), buf_rows * sizeof(int));
        ok = ok && write_all(vecs_out, vec_buf.data(), buf_rows * (size_t)dim * sizeof(float));
        buf_rows = 0;
        return ok;
    };

    bool ok = true;
    for (int i = 0; i < num_vectors && ok; ++i) {
        int id;
        if (!cur.next_int(id)) {
            std::cerr << "Parse error reading id for row " << i << "\n";
            ok = false;
            break;
        }
        id_buf[buf_rows] = id;
        float* row = vec_buf.data() + buf_rows * dim;
        for (int j = 0; j < dim; ++j) {
            float val;
            if (!cur.next_float(val)) {
                std::cerr << "Parse error reading component " << j << " of row " << i << "\n";
                ok = false;
                break;
            }
            // Match read_data()'s noise-floor snap so binary and ASCII
            // loaders produce byte-identical vectors.
            if (std::fabs(val) < 1e-6f) val = 0.0f;
            row[j] = val;
        }
        if (!ok) break;
        ++buf_rows;

        if (buf_rows == kFlushRows) {
            if (!flush()) { ok = false; break; }
        }

        if ((i + 1) % 20000000 == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double secs = std::chrono::duration<double>(now - t0).count();
            const double rate = (i + 1) / std::max(secs, 1e-9);
            const double eta_s = (num_vectors - (i + 1)) / std::max(rate, 1e-9);
            std::cout << "  " << (i + 1) << "/" << num_vectors
                      << " (" << (100.0 * (i + 1) / num_vectors) << "%), "
                      << rate << " rows/s, ETA " << (eta_s / 60.0) << " min\n";
        }
    }

    if (ok) ok = flush();

    std::fclose(vecs_out);
    std::fclose(ids_out);
    munmap(addr, filesize);

    if (!ok) {
        std::cerr << "Conversion failed.\n";
        return 1;
    }

    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "Done in " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    std::cout << "Wrote " << prefix << ".meta.txt / .vecs.bin / .ids.bin\n";
    return 0;
}
