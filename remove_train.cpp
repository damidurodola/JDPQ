// Removes the training rows named by an extract_train IDs file from the
// database IN PLACE, once their vectors have been saved with export_train.
//
// Before touching the database it checks that every training ID is present
// and that its row is bit-identical to the vector saved in the export (and
// that the export lists the same IDs in the same order as the IDs file), so a
// row is only ever deleted if a verified copy exists elsewhere. Then the
// database is compacted in place (text file, or convert_to_binary prefix; N
// in the header/meta is updated). Any other rows sharing a training ID are
// removed too.
//
// After this the IDs file no longer matches the database (it records the
// pre-removal row count), so test.cpp must load the training vectors from the
// export (train_prefix) rather than by ID. The in-place rewrite is NOT
// crash-safe -- see db_compact.h; back the database up first.
//
// Usage: ./remove_train [--dry-run] <db.txt | db_prefix> <train_ids_file> <train_vectors_prefix>

#include "src/db_compact.h"
#include "src/readData.h"
#include "src/train_ids.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace db_compact;

int main(int argc, char** argv) {
    bool dry_run = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dry-run") dry_run = true;
        else args.push_back(argv[i]);
    }
    if (args.size() != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " [--dry-run] <db.txt | db_prefix> <train_ids_file> <train_vectors_prefix>\n";
        return 1;
    }
    const std::string db_path = args[0], ids_path = args[1], train_prefix = args[2];

    const std::string marker = db_path + ".dedup_in_progress";
    if (file_exists(marker)) {
        std::cerr << marker << " exists: a previous in-place rewrite was interrupted and "
                  << db_path << " may be corrupt. Restore it from a backup and delete the marker.\n";
        return 1;
    }

    std::vector<int> train_ids;
    uint64_t ids_db_rows = 0, ids_dim = 0;
    if (!faiss_mimic::read_train_ids(ids_path, ids_db_rows, ids_dim, train_ids)) return 1;
    const size_t ntrain = train_ids.size();

    // The saved copy of the training vectors.
    faiss_mimic::MappedDataset train_mapped;
    if (!faiss_mimic::read_data_binary(train_prefix, train_mapped)) return 1;
    if ((size_t)train_mapped.num_vectors != ntrain || (uint64_t)train_mapped.dim != ids_dim ||
        std::memcmp(train_mapped.ids, train_ids.data(), ntrain * sizeof(int)) != 0) {
        std::cerr << "Export at " << train_prefix << " does not match " << ids_path
                  << " (count, dim or ID order differ). Rerun export_train.\n";
        return 1;
    }

    // The database.
    const bool is_binary = file_exists(db_path + ".meta.txt");
    int N = 0, D = 0;
    const float* data = nullptr;
    const int* ids = nullptr;
    std::vector<float> text_data;
    std::vector<int> text_ids;
    faiss_mimic::MappedDataset db_mapped;
    if (is_binary) {
        if (!faiss_mimic::read_data_binary(db_path, db_mapped)) return 1;
        N = db_mapped.num_vectors;
        D = db_mapped.dim;
        data = db_mapped.data;
        ids = db_mapped.ids;
    } else {
        faiss_mimic::read_data(db_path, N, D, text_data, text_ids);
        N = (int)text_ids.size();
        data = text_data.data();
        ids = text_ids.data();
    }
    if (N == 0 || D == 0) {
        std::cerr << "Failed to load " << db_path << " or it is empty.\n";
        return 1;
    }
    if (ids_db_rows != (uint64_t)N || ids_dim != (uint64_t)D) {
        std::cerr << ids_path << " was made for a " << ids_db_rows << " x " << ids_dim << " db; "
                  << db_path << " is " << N << " x " << D
                  << " (already removed, or a different db?). Nothing changed.\n";
        return 1;
    }
    std::cout << "Loaded " << N << " vectors of dim " << D << "; " << ntrain << " training rows to remove\n";

    // Verify, then flag.
    std::unordered_map<int, size_t> id_to_slot;
    id_to_slot.reserve(ntrain);
    for (size_t i = 0; i < ntrain; ++i) id_to_slot.emplace(train_ids[i], i);
    if (id_to_slot.size() != ntrain) {
        std::cerr << ids_path << " contains duplicate IDs.\n";
        return 1;
    }
    Bitmap dropped((size_t)N);
    std::vector<char> verified(ntrain, 0);
    size_t num_dropped = 0, mismatched = 0;
    for (int row = 0; row < N; ++row) {
        auto it = id_to_slot.find(ids[row]);
        if (it == id_to_slot.end()) continue;
        dropped.set((size_t)row);
        ++num_dropped;
        const size_t slot = it->second;
        if (!verified[slot]) {
            if (std::memcmp(data + (size_t)row * D, train_mapped.data + slot * D, (size_t)D * sizeof(float)) == 0)
                verified[slot] = 1;
            else
                ++mismatched;
        }
    }
    size_t num_verified = 0;
    for (char v : verified) num_verified += v;
    if (num_verified != ntrain) {
        std::cerr << "Only " << num_verified << " of " << ntrain
                  << " training rows are present in the db and identical to the saved copy ("
                  << mismatched << " differ). Nothing changed.\n";
        return 1;
    }
    std::cout << "All " << ntrain << " training rows verified against " << train_prefix << " ("
              << num_dropped << " db rows to drop)\n";

    if (dry_run) {
        std::cout << "--dry-run: " << db_path << " left untouched.\n";
        return 0;
    }

    const size_t new_n = (size_t)N - num_dropped;
    db_mapped = faiss_mimic::MappedDataset(); // release the read-only mapping
    { std::ofstream(marker) << "in-place removal of training rows from " << db_path << " in progress\n"; }
    std::cout << "Removing " << num_dropped << " rows from " << db_path << " in place...\n";
    const bool ok = is_binary ? compact_binary(db_path, (size_t)N, (size_t)D, dropped, new_n)
                              : compact_text(db_path, (size_t)N, (size_t)D, dropped, new_n);
    if (!ok) {
        std::cerr << "In-place removal FAILED; " << db_path << " may be corrupt (marker "
                  << marker << " left in place).\n";
        return 1;
    }
    std::remove(marker.c_str());
    std::cout << "Database now has " << new_n << " rows.\n";
    return 0;
}
