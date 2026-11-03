// Picks a training set out of a database file and saves only its IDs, after
// removing duplicate-ID rows from the database IN PLACE.
//
// Input is either an ASCII "N D\n id v0 ... v(D-1)\n ..." file, or a binary
// prefix written by convert_to_binary (detected by "<input>.meta.txt"
// existing: <prefix>.meta.txt / .ids.bin / .vecs.bin). The vectors are never
// read or copied, so this works on the ~1B-row zinc20 file.
//
// Step 1 -- duplicate-ID removal: every row whose ID already appeared earlier
// in the file is dropped (first occurrence wins). Seen IDs are tracked in a
// bitmap over [min_id, max_id] and dropped rows in a bitmap over row indices.
// If any duplicates exist, the database is compacted in place (kept rows
// shift forward, the file is truncated, N in the header/meta is updated).
// With no duplicates the database is not touched at all.
//
// Step 2 -- sampling: same block-wise scheme as test.cpp ("Random subset
// sampling"): random non-overlapping contiguous blocks of up to
// kSampleBlockRows rows, fixed seed. Rows dropped in step 1 are skipped and
// the block extends past them, so exactly ntrain unique IDs are selected.
// Contiguous blocks keep test.cpp's later gather of these rows sequential.
//
// Output: <train_ids_file> in the format of src/train_ids.h (IDs plus the
// post-dedup row count and dim of the database, which test.cpp checks).
//
// The in-place rewrite is NOT crash-safe: if it is interrupted the database
// is corrupt. A "<input>.dedup_in_progress" marker is left behind in that
// case and this tool refuses to run until it is removed; restore the database
// from a backup first. Use --dry-run to only report duplicates.
//
// Usage: ./extract_train [--dry-run] <input.txt | input_prefix> <train_ids_file> <ntrain> [seed]

#include "src/readData.h"
#include "src/db_compact.h"
#include "src/train_ids.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace db_compact;

namespace {

constexpr int kSampleBlockRows = 20000; // matches test.cpp
constexpr uint64_t kDefaultSeed = 12345; // matches test.cpp

} // namespace

int main(int argc, char** argv) {
    bool dry_run = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dry-run") dry_run = true;
        else args.push_back(argv[i]);
    }
    if (args.size() < 3 || args.size() > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " [--dry-run] <input.txt | input_prefix> <train_ids_file> <ntrain> [seed]\n";
        return 1;
    }
    const std::string input = args[0];
    const std::string train_ids_path = args[1];
    const long long ntrain_arg = std::stoll(args[2]);
    const uint64_t seed = (args.size() == 4) ? std::stoull(args[3]) : kDefaultSeed;
    if (ntrain_arg <= 0) {
        std::cerr << "ntrain must be positive\n";
        return 1;
    }
    const int ntrain = (int)ntrain_arg;

    const std::string marker = input + ".dedup_in_progress";
    if (file_exists(marker)) {
        std::cerr << marker << " exists: a previous in-place dedup was interrupted and "
                  << input << " may be corrupt. Restore it from a backup and delete the marker.\n";
        return 1;
    }

    // Load IDs only
    const bool is_binary = file_exists(input + ".meta.txt");
    int N = 0, D = 0;
    const int* ids = nullptr;
    std::vector<int> text_ids;
    faiss_mimic::MappedDataset mapped;
    if (is_binary) {
        std::cout << "Reading binary dataset " << input << "\n";
        if (!faiss_mimic::read_data_binary(input, mapped)) return 1;
        N = mapped.num_vectors;
        D = mapped.dim;
        ids = mapped.ids;
    } else {
        std::cout << "Reading text dataset " << input << "\n";
        faiss_mimic::read_ids_only(input, N, D, text_ids);
        N = (int)text_ids.size();
        ids = text_ids.data();
    }
    if (N == 0 || D == 0) {
        std::cerr << "Failed to load " << input << " or it is empty.\n";
        return 1;
    }
    std::cout << "Loaded " << N << " vectors of dim " << D << "\n";

    // Step 1: duplicate-ID detection (first occurrence wins).
    int min_id = ids[0], max_id = ids[0];
    for (int i = 1; i < N; ++i) {
        min_id = std::min(min_id, ids[i]);
        max_id = std::max(max_id, ids[i]);
    }
    const size_t id_range = (size_t)((int64_t)max_id - (int64_t)min_id) + 1;
    std::cout << "ID range [" << min_id << ", " << max_id << "] ("
              << (id_range / 8 / (1024 * 1024)) << " MB seen-ID bitmap)\n";

    Bitmap seen(id_range);
    Bitmap dropped((size_t)N);
    size_t num_dup_rows = 0;
    std::vector<int> example_dups;
    for (int i = 0; i < N; ++i) {
        const size_t key = (size_t)((int64_t)ids[i] - (int64_t)min_id);
        if (seen.test(key)) {
            dropped.set((size_t)i);
            ++num_dup_rows;
            if (example_dups.size() < 10) example_dups.push_back(ids[i]);
        } else {
            seen.set(key);
        }
    }
    const size_t num_unique = (size_t)N - num_dup_rows;
    std::cout << "Duplicate-ID rows: " << num_dup_rows << " (unique rows: " << num_unique << ")\n";
    if (!example_dups.empty()) {
        std::cout << "  e.g. duplicated IDs:";
        for (int id : example_dups) std::cout << " " << id;
        std::cout << "\n";
    }
    if ((size_t)ntrain > num_unique) {
        std::cerr << "ntrain=" << ntrain << " exceeds the " << num_unique
                  << " unique rows available.\n";
        return 1;
    }
    if (N < kSampleBlockRows) {
        std::cerr << "Dataset has fewer than " << kSampleBlockRows
                  << " rows; block sampling needs at least one full block.\n";
        return 1;
    }

    // Step 2: block-wise sampling over the deduplicated rows (IDs only).
    std::vector<int> train_ids(ntrain);
    {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> dist(0, N - kSampleBlockRows);
        std::vector<std::pair<int, int>> used_ranges; // [start, end) of accepted blocks
        auto overlaps_used = [&](int s, int e) {
            for (const auto& r : used_ranges)
                if (s < r.second && r.first < e) return true;
            return false;
        };

        // Starts are drawn from test.cpp's range first; if that keeps landing
        // on used ranges (ntrain a large share of N, so the free rows are
        // near the end of the file), widen to every start that fits the
        // block. Gives up only if even that fails.
        constexpr int kWidenAfter = 10000;
        constexpr int kMaxAttempts = 1000000;
        int filled = 0;
        while (filled < ntrain) {
            const int block_rows = std::min(kSampleBlockRows, ntrain - filled);
            std::uniform_int_distribution<int> wide_dist(0, N - block_rows);
            int start, attempts = 0;
            do {
                start = (attempts < kWidenAfter) ? dist(rng) : wide_dist(rng);
                if (++attempts > kMaxAttempts) {
                    std::cerr << "Could not find a free block after " << kMaxAttempts
                              << " attempts (filled " << filled << "/" << ntrain << ").\n";
                    return 1;
                }
            } while (overlaps_used(start, start + block_rows));

            // Walk forward from start, skipping dropped rows, until the block
            // has block_rows unique rows, end of file, or another block.
            int row = start, taken = 0;
            while (taken < block_rows && row < N) {
                if (row > start && overlaps_used(row, row + 1)) break;
                if (!dropped.test((size_t)row)) train_ids[filled + taken++] = ids[row];
                ++row;
            }
            used_ranges.emplace_back(start, row);
            filled += taken;
        }
        std::cout << "Sampled " << ntrain << " IDs from " << used_ranges.size()
                  << " block(s), seed " << seed << "\n";
    }

    if (dry_run) {
        std::cout << "--dry-run: database and " << train_ids_path << " left untouched.\n";
        return 0;
    }

    // Step 3: drop the duplicates from the database in place.
    if (num_dup_rows > 0) {
        mapped = faiss_mimic::MappedDataset(); // release the read-only mapping
        std::cout << "Removing " << num_dup_rows << " duplicate rows from " << input << " in place...\n";
        { std::ofstream(marker) << "in-place dedup of " << input << " in progress\n"; }
        const bool ok = is_binary ? compact_binary(input, (size_t)N, (size_t)D, dropped, num_unique)
                                  : compact_text(input, (size_t)N, (size_t)D, dropped, num_unique);
        if (!ok) {
            std::cerr << "In-place dedup FAILED; " << input << " may be corrupt (marker "
                      << marker << " left in place).\n";
            return 1;
        }
        std::remove(marker.c_str());
        std::cout << "Database now has " << num_unique << " rows.\n";
    }

    if (!faiss_mimic::write_train_ids(train_ids_path, train_ids, num_unique, (uint64_t)D)) return 1;
    std::cout << "Wrote " << ntrain << " train IDs to " << train_ids_path << "\n";
    return 0;
}
