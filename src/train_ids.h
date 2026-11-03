#pragma once

// File format shared by extract_train.cpp (writer) and test.cpp (reader): the
// IDs of the training rows, in training order, plus enough about the
// (deduplicated) database they were drawn from to detect a mismatched db.
//   char     magic[8]  = "TRAINID1"
//   uint64_t count       number of training IDs
//   uint64_t db_rows     rows in the db AFTER duplicate-ID removal
//   uint64_t dim         vector dimension of the db
//   int32_t  ids[count]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace faiss_mimic {

inline bool write_train_ids(const std::string& path, const std::vector<int>& ids,
                            uint64_t db_rows, uint64_t dim) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        perror(("Error opening " + path).c_str());
        return false;
    }
    const uint64_t count = ids.size();
    bool ok = std::fwrite("TRAINID1", 1, 8, f) == 8;
    ok = ok && std::fwrite(&count, sizeof(count), 1, f) == 1;
    ok = ok && std::fwrite(&db_rows, sizeof(db_rows), 1, f) == 1;
    ok = ok && std::fwrite(&dim, sizeof(dim), 1, f) == 1;
    ok = ok && std::fwrite(ids.data(), sizeof(int), ids.size(), f) == ids.size();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) std::cerr << "Failed writing " << path << "\n";
    return ok;
}

inline bool read_train_ids(const std::string& path, uint64_t& db_rows, uint64_t& dim,
                           std::vector<int>& ids) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        perror(("Error opening " + path).c_str());
        return false;
    }
    char magic[8];
    uint64_t count = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "TRAINID1", 8) == 0;
    ok = ok && std::fread(&count, sizeof(count), 1, f) == 1;
    ok = ok && std::fread(&db_rows, sizeof(db_rows), 1, f) == 1;
    ok = ok && std::fread(&dim, sizeof(dim), 1, f) == 1;
    if (ok) {
        ids.resize(count);
        ok = std::fread(ids.data(), sizeof(int), count, f) == count;
    }
    std::fclose(f);
    if (!ok) std::cerr << path << " is not a valid train-IDs file (run extract_train)\n";
    return ok;
}

} // namespace faiss_mimic
