// test_rerank.cpp
//
// Baseline reranking test for approximate-NN methods (HNSW / LSH / etc).
//
// Input: a CSV of per-query shortlists with columns
//   query_id,rank,neighbor_id,distance
//
// For each query the program collects its shortlist (sorted by `rank`),
// re-scores every candidate with exact polygon-Jaccard distance (GEOS),
// re-sorts by that distance, and reports:
//   * total rerank wall time + QPS
//   * Recall@R before and after rerank, against the brute polygon-Jaccard GT
//
// The reranked shortlists are written back out as a CSV with the same schema.
//
// Build (from project root):
//   make test_rerank

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/PolygonGT.h"
#include "src/index.h"
#include "src/readData.h"

// ----------------------------------------------------------------------------
// Hardcoded paths — edit per run. Currently set to the cemetery dataset.
// ----------------------------------------------------------------------------

// CSV of approximate neighbors produced by HNSW / LSH / etc.
//   query_id,rank,neighbor_id,distance
// static const std::string INPUT_CSV =
//     "/path/to/data/ChumHash_Results/Parks_L128_b32_r4_k500_neighbors.csv";
static const std::string INPUT_CSV =
    "/path/to/data/results/Parks_wmh_neighbors.csv";
    
// Where to write the reranked shortlists.
static const std::string OUTPUT_CSV =
    "/path/to/home/FAISS_MIMIC/data/chumHash_parks_neighbors_polyrr.csv";

// Vector files: used only to reconstruct the same (query_ids, add_ids) ordering
// that test.cpp uses, so the polygon-GT cache key matches and we get a cache hit.
static const std::string DB_VECTOR_FILE =
    "/path/to/data/testfinal_parks_vectors.txt";
static const std::string QUERY_VECTOR_FILE =
     "/path/to/data/queryfinal_parks_vectors.txt";
    
// static const std::string DB_VECTOR_FILE =
//     "/path/to/data/testfinal_cemetery_vectors.txt";
// static const std::string QUERY_VECTOR_FILE =
//     "/path/to/data/queryfinal_cemetery_vectors.txt";

// Polygon TSVs and brute polygon-Jaccard GT cache (must match test.cpp).
static const std::string DB_POLYGON_TSV =
    "/path/to/home/FAISS_MIMIC/data/db_parks_polygons.tsv";
static const std::string QUERY_POLYGON_TSV =
    "/path/to/home/FAISS_MIMIC/data/query_parks_polygons.tsv";
static const std::string POLYGON_GT_CACHE =
    "/path/to/home/FAISS_MIMIC/data/polygon_gt_parks_cache.bin";
    
// static const std::string DB_POLYGON_TSV =
//     "/path/to/home/FAISS_MIMIC/data/db_cemetery_polygons.tsv";
// static const std::string QUERY_POLYGON_TSV =
//     "/path/to/home/FAISS_MIMIC/data/query_cemetery_polygons.tsv";
// static const std::string POLYGON_GT_CACHE =
//     "/path/to/home/FAISS_MIMIC/data/polygon_gt_cemetery_cache.bin";

// Training-set size to skip when carving add_ids out of db_ids. Must match
// test.cpp so the polygon-GT cache hash key matches.
static const int NTRAIN = 100000;
// static const int NTRAIN = 20000;

// Top-k used when computing / loading polygon GT. Must match test.cpp (= 100).
static const int K_GT = 100;

// R values to report Recall@R at.
static const std::vector<int> R_VALUES = {1, 10, 50, 100};

// ----------------------------------------------------------------------------

struct ShortlistEntry {
    int rank;
    long long neighbor_id;
    float distance;
};

// Parse the input CSV. Header line is optional (auto-detected by checking
// whether the first non-empty field is non-numeric).
// Returns query_ids in first-seen order and a map query_id -> sorted entries.
static bool read_shortlist_csv(
    const std::string& path,
    std::vector<long long>& out_query_ids,           // first-seen order
    std::unordered_map<long long, std::vector<ShortlistEntry>>& out_by_query)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        std::fprintf(stderr, "Cannot open input CSV: %s\n", path.c_str());
        return false;
    }
    std::string line;
    bool first = true;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) continue;
        // Detect & skip header on first non-empty line.
        if (first) {
            first = false;
            if (!line.empty() && !(std::isdigit((unsigned char)line[0]) || line[0] == '-')) {
                continue;
            }
        }
        std::stringstream ss(line);
        std::string tok;
        long long qid = 0, nid = 0;
        int rank = 0;
        float dist = 0.f;
        try {
            if (!std::getline(ss, tok, ',')) continue;
            qid = std::stoll(tok);
            if (!std::getline(ss, tok, ',')) continue;
            rank = std::stoi(tok);
            if (!std::getline(ss, tok, ',')) continue;
            nid = std::stoll(tok);
            if (!std::getline(ss, tok, ',')) continue;
            dist = std::stof(tok);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[csv] skipping line %zu: %s\n",
                         line_no, e.what());
            continue;
        }
        auto it = out_by_query.find(qid);
        if (it == out_by_query.end()) {
            out_query_ids.push_back(qid);
            out_by_query.emplace(qid, std::vector<ShortlistEntry>{
                                          {rank, nid, dist}});
        } else {
            it->second.push_back({rank, nid, dist});
        }
    }

    // Sort each query's entries by rank ascending.
    for (auto& kv : out_by_query) {
        std::sort(kv.second.begin(), kv.second.end(),
                  [](const ShortlistEntry& a, const ShortlistEntry& b) {
                      return a.rank < b.rank;
                  });
    }
    return true;
}

// Recall@R (ANN-benchmarks definition): |A_R intersect G_R| / R, averaged.
// `gt_stride` and `res_stride` are the per-query row widths of the GT and
// result arrays. R must be <= both.
static float recall_at(int nq, int R,
                       int gt_stride,
                       const std::vector<faiss_mimic::idx_t>& gt,
                       int res_stride,
                       const std::vector<faiss_mimic::idx_t>& res)
{
    if (R <= 0 || nq <= 0) return 0.f;
    if (R > gt_stride) R = gt_stride;
    if (R > res_stride) R = res_stride;
    long long hits = 0;
    for (int q = 0; q < nq; ++q) {
        std::unordered_set<faiss_mimic::idx_t> gset;
        gset.reserve(R);
        for (int j = 0; j < R; ++j) {
            faiss_mimic::idx_t gid = gt[(size_t)q * gt_stride + j];
            if (gid >= 0) gset.insert(gid);
        }
        for (int j = 0; j < R; ++j) {
            if (gset.count(res[(size_t)q * res_stride + j])) ++hits;
        }
    }
    return (float)((double)hits / ((double)nq * (double)R));
}

// FAISS / PQ-paper 1-Recall@R (Jégou et al. 2011): fraction of queries whose
// true 1-NN (GT rank 0) appears anywhere in the first R returned candidates.
static float one_recall_at(int nq, int R,
                           int gt_stride,
                           const std::vector<faiss_mimic::idx_t>& gt,
                           int res_stride,
                           const std::vector<faiss_mimic::idx_t>& res)
{
    if (R <= 0 || nq <= 0 || gt_stride <= 0) return 0.f;
    if (R > res_stride) R = res_stride;
    long long hits = 0;
    for (int q = 0; q < nq; ++q) {
        faiss_mimic::idx_t truth = gt[(size_t)q * gt_stride + 0];
        if (truth < 0) continue;
        for (int j = 0; j < R; ++j) {
            if (res[(size_t)q * res_stride + j] == truth) { ++hits; break; }
        }
    }
    return (float)((double)hits / (double)nq);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    // ------------------------------------------------------------------
    // 1. Read the shortlist CSV.
    // ------------------------------------------------------------------
    std::printf("[1] Reading shortlist CSV: %s\n", INPUT_CSV.c_str());
    std::vector<long long> csv_query_ids;
    std::unordered_map<long long, std::vector<ShortlistEntry>> by_query;
    if (!read_shortlist_csv(INPUT_CSV, csv_query_ids, by_query)) return 1;

    const int nq = (int)csv_query_ids.size();
    if (nq == 0) {
        std::fprintf(stderr, "No queries parsed from CSV.\n");
        return 1;
    }
    // stride = max shortlist length across queries (pad shorter ones with -1).
    int stride = 0;
    for (const auto& qid : csv_query_ids)
        stride = std::max(stride, (int)by_query[qid].size());
    std::printf("    queries: %d   stride (max shortlist): %d\n", nq, stride);

    // ------------------------------------------------------------------
    // 2. Build the shortlist matrix the rerank API expects.
    // ------------------------------------------------------------------
    std::vector<int> query_ids_int(nq);
    for (int i = 0; i < nq; ++i) query_ids_int[i] = (int)csv_query_ids[i];

    std::vector<faiss_mimic::idx_t> shortlist_ids((size_t)nq * stride, -1);
    for (int q = 0; q < nq; ++q) {
        const auto& entries = by_query[csv_query_ids[q]];
        for (size_t r = 0; r < entries.size() && (int)r < stride; ++r) {
            shortlist_ids[(size_t)q * stride + r] =
                (faiss_mimic::idx_t)entries[r].neighbor_id;
        }
    }

    // ------------------------------------------------------------------
    // 3. Rerank with exact polygon Jaccard (timed).
    // ------------------------------------------------------------------
    std::printf("[2] Reranking with polygon Jaccard (GEOS)...\n");
    std::vector<faiss_mimic::idx_t> rr_ids;
    std::vector<float> rr_dists;
    auto t0 = std::chrono::high_resolution_clock::now();
    faiss_mimic::polygon_rerank_shortlist(
        query_ids_int, shortlist_ids, stride,
        DB_POLYGON_TSV, QUERY_POLYGON_TSV,
        rr_ids, rr_dists);
    double rerank_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    double qps = (rerank_s > 0.0) ? nq / rerank_s : 0.0;
    std::printf("    rerank wall time: %.3f s   (%.2f QPS)\n", rerank_s, qps);

    // ------------------------------------------------------------------
    // 4. Build / load the brute polygon-Jaccard GT (for recall scoring).
    //     Uses the same add_ids / query_ids ordering as test.cpp so the
    //     existing cache file is reused.
    // ------------------------------------------------------------------
    std::printf("[3] Loading polygon GT (cache=%s)\n", POLYGON_GT_CACHE.c_str());
    int db_n = 0, db_dim = 0;
    std::vector<int> db_ids_full;
    faiss_mimic::read_ids_only(DB_VECTOR_FILE, db_n, db_dim, db_ids_full);
    if (db_n == 0) {
        std::fprintf(stderr, "Could not read db ids from %s\n",
                     DB_VECTOR_FILE.c_str());
        return 1;
    }
    if (db_n <= NTRAIN) {
        std::fprintf(stderr, "DB has %d vectors but NTRAIN=%d\n", db_n, NTRAIN);
        return 1;
    }
    std::vector<int> add_ids(db_ids_full.begin() + NTRAIN, db_ids_full.end());

    int q_n = 0, q_dim = 0;
    std::vector<int> all_query_ids;
    faiss_mimic::read_ids_only(QUERY_VECTOR_FILE, q_n, q_dim, all_query_ids);
    if (q_n == 0) {
        std::fprintf(stderr, "Could not read query ids from %s\n",
                     QUERY_VECTOR_FILE.c_str());
        return 1;
    }

    std::vector<faiss_mimic::idx_t> gt_original_ids;
    std::vector<float>              gt_distances;
    bool gt_ok = faiss_mimic::build_polygon_ground_truth(
        all_query_ids, add_ids, DB_POLYGON_TSV, QUERY_POLYGON_TSV,
        K_GT, gt_original_ids, gt_distances, POLYGON_GT_CACHE);
    if (!gt_ok) {
        std::fprintf(stderr,
                     "[warn] polygon GT had missing polygons; some slots are -1\n");
    }

    // Per-query GT row lookup: query_id -> row index in gt_original_ids.
    std::unordered_map<int, int> qid_to_row;
    qid_to_row.reserve(all_query_ids.size());
    for (int i = 0; i < (int)all_query_ids.size(); ++i)
        qid_to_row[all_query_ids[i]] = i;

    // Build a CSV-aligned GT view (nq rows of width K_GT).
    std::vector<faiss_mimic::idx_t> gt_aligned((size_t)nq * K_GT, -1);
    int n_missing_gt = 0;
    for (int q = 0; q < nq; ++q) {
        auto it = qid_to_row.find(query_ids_int[q]);
        if (it == qid_to_row.end()) { ++n_missing_gt; continue; }
        const size_t src = (size_t)it->second * K_GT;
        const size_t dst = (size_t)q * K_GT;
        for (int j = 0; j < K_GT; ++j)
            gt_aligned[dst + j] = gt_original_ids[src + j];
    }
    if (n_missing_gt)
        std::fprintf(stderr,
                     "[warn] %d/%d CSV query ids missing from GT (recall undercounts)\n",
                     n_missing_gt, nq);

    // ------------------------------------------------------------------
    // 5. Recall@R (ANN-benchmarks) and 1-Recall@R (FAISS / PQ-paper),
    //     before and after rerank.
    // ------------------------------------------------------------------
    std::printf("\n[4] Recall@R   (ANN = raw shortlist, +polyrr = after polygon rerank)\n");
    std::printf("    %-6s  %-12s  %-12s  %-14s  %-14s\n",
                "R", "Recall ANN", "Recall +rr", "1-Recall ANN", "1-Recall +rr");
    for (int R : R_VALUES) {
        if (R > stride) continue;
        float r_before  = recall_at    (nq, R, K_GT, gt_aligned, stride, shortlist_ids);
        float r_after   = recall_at    (nq, R, K_GT, gt_aligned, stride, rr_ids);
        float o1_before = one_recall_at(nq, R, K_GT, gt_aligned, stride, shortlist_ids);
        float o1_after  = one_recall_at(nq, R, K_GT, gt_aligned, stride, rr_ids);
        std::printf("    %-6d  %-12.4f  %-12.4f  %-14.4f  %-14.4f\n",
                    R, r_before, r_after, o1_before, o1_after);
    }

    // ------------------------------------------------------------------
    // 6. Write reranked CSV.
    // ------------------------------------------------------------------
    std::printf("\n[5] Writing reranked CSV: %s\n", OUTPUT_CSV.c_str());
    std::ofstream out(OUTPUT_CSV);
    if (!out.is_open()) {
        std::fprintf(stderr, "Cannot open output: %s\n", OUTPUT_CSV.c_str());
        return 1;
    }
    out << "query_id,rank,neighbor_id,distance\n";
    out << std::fixed;
    out.precision(8);
    for (int q = 0; q < nq; ++q) {
        const int qid = query_ids_int[q];
        const int row_n = (int)by_query[csv_query_ids[q]].size();  // skip padding
        for (int r = 0; r < row_n; ++r) {
            faiss_mimic::idx_t nid = rr_ids[(size_t)q * stride + r];
            if (nid < 0) continue;
            out << qid << "," << (r + 1) << "," << nid << ","
                << rr_dists[(size_t)q * stride + r] << "\n";
        }
    }
    out.close();

    std::printf("\nDone. rerank=%.3fs (%.2f QPS) over %d queries.\n",
                rerank_s, qps, nq);
    return 0;
}
