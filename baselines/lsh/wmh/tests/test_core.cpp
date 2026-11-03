// Unit tests: npz reader, exact J_w, ground truth (inverted index vs brute
// force, ties), slot matching, truncation and estimator correction.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "csr.hpp"
#include "gt.hpp"
#include "index.hpp"
#include "sketch.hpp"

using namespace wmh;
static int fails = 0, checks = 0;
#define CHECK(c) do { checks++; if (!(c)) { fails++; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static Csr random_csr(std::mt19937_64& g, int n, int d, double density, int dup_every) {
    Csr m; m.n = n; m.d = d; m.indptr.push_back(0);
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < n; i++) {
        if (dup_every && i % dup_every == 1) {  // exact copy of previous row -> ties in J
            int64_t a = m.indptr[i - 1], b = m.indptr[i];
            for (int64_t p = a; p < b; p++) { m.indices.push_back(m.indices[p]); m.data.push_back(m.data[p]); }
        } else {
            for (int c = 0; c < d; c++) if (u(g) < density) { m.indices.push_back(c); m.data.push_back(0.01 + u(g)); }
            if ((int64_t)m.indices.size() == m.indptr.back()) { m.indices.push_back(i % d); m.data.push_back(1.0); }
        }
        m.indptr.push_back(m.indices.size());
    }
    m.finalize();
    return m;
}

int main(int argc, char** argv) {
    // npz reader
    Csr F = load_csr(argc > 1 ? argv[1] : "tests/fixture.npz");
    CHECK(F.n == 3 && F.d == 4 && F.indices.size() == 7);
    CHECK(F.data[1] == 1e-10 && F.indices[1] == 2 && F.data[2] == 2.25 && F.indices[2] == 3);
    CHECK(F.l1[2] == 10.0);

    // exact J_w
    CHECK(jw(F, 2, F, 2) == 1.0);
    CHECK(jw(F, 0, F, 1) == 0.0);  // disjoint support
    std::mt19937_64 g(1);
    Csr A = random_csr(g, 300, 40, 0.3, 7), B = random_csr(g, 20, 40, 0.3, 0);
    double maxdiff = 0;
    for (int i = 0; i < B.n; i++) for (int j = 0; j < A.n; j++) maxdiff = std::max(maxdiff, std::fabs(jw(B, i, A, j) - jw_direct(B, i, A, j)));
    CHECK(maxdiff < 1e-12);
    {   // x vs 2x has J_w = 1/2 exactly
        Csr C = F; for (auto& v : C.data) v *= 2; C.finalize();
        CHECK(std::fabs(jw(F, 2, C, 2) - 0.5) < 1e-15);
    }

    // ground truth: inverted index == brute force, including ties (duplicate rows)
    GroundTruth gt = gt_inverted(B, A, 25);
    int mism = 0;
    for (int q = 0; q < B.n; q++) {
        auto bt = brute_topk(B, q, A, 25);
        for (int j = 0; j < 25; j++) mism += bt[j].id != gt.ids[q * 25 + j] || bt[j].s != gt.sims[q * 25 + j];
        for (int j = 1; j < 25; j++) {  // order: J desc, then id asc
            double s0 = gt.sims[q * 25 + j - 1], s1 = gt.sims[q * 25 + j];
            CHECK(s0 > s1 || (s0 == s1 && gt.ids[q * 25 + j - 1] < gt.ids[q * 25 + j]));
        }
    }
    CHECK(mism == 0);
    {   // tie-break on an explicit tie
        std::vector<double> sc = {0.5, 0.9, 0.5, 0.9, 0.1};
        auto t = topk_of(sc, 4);
        CHECK(t[0].id == 1 && t[1].id == 3 && t[2].id == 0 && t[3].id == 2);
        auto all = topk_of(sc, 10);  // K > N
        CHECK(all.size() == 5);
    }

    // slot matching is position-wise
    uint64_t a[4] = {1, 2, 3, 4}, b[4] = {4, 3, 2, 1}, c[4] = {1, 9, 3, 9};
    CHECK(count_matches(a, a, 4) == 4);
    CHECK(count_matches(a, b, 4) == 0);  // same multiset, different slots
    CHECK(count_matches(a, c, 4) == 2);

    // truncation: b=64 is identity; b-bit values fit; false match rate ~ 2^-b
    CHECK(truncate_id(0x123456789abcdefull, 64) == 0x123456789abcdefull);
    CHECK(truncate_id(~0ull, 8) < 256 && truncate_id(~0ull, 16) < 65536);
    int fm = 0, trials = 1 << 20;
    for (int i = 0; i < trials; i++) fm += truncate_id(g(), 8) == truncate_id(g(), 8);
    CHECK(std::fabs((double)fm / trials - 1.0 / 256) < 0.0005);
    // estimator correction inverts E[p] = J + (1-J) 2^-b
    double J = 0.3, f = 1.0 / 256, p = J + (1 - J) * f;
    CHECK(std::fabs(estimate_jw((int)std::lround(p * 1e6), 1000000, 8, true) - J) < 1e-5);
    CHECK(estimate_jw(5, 10, 8, false) == 0.5 && estimate_jw(5, 10, 64, true) == 0.5);

    // sketchers: determinism and slot-wise agreement for identical input
    for (Algo al : {Algo::Dart, Algo::FastICWS, Algo::BagMinHash2}) {
        Sketcher s1(al, 32, 42), s2(al, 32, 42), s3(al, 32, 43);
        uint64_t h1[32], h2[32], h3[32];
        s1.sketch(A.idx(5), A.val(5), A.nnz(5), h1);
        s2.sketch(A.idx(5), A.val(5), A.nnz(5), h2);
        s3.sketch(A.idx(5), A.val(5), A.nnz(5), h3);
        CHECK(count_matches(h1, h2, 32) == 32);
        CHECK(count_matches(h1, h3, 32) < 32);  // different seed, different hash functions
        bool threw = false;
        try { s1.sketch(nullptr, nullptr, 0, h1); } catch (const std::exception&) { threw = true; }
        CHECK(threw);  // empty vectors are rejected
    }

    // top-C selection equals a full sort under (score desc, id asc), in any insertion order
    for (int trial = 0; trial < 50; trial++) {
        int n = 1 + (int)(g() % 3000), C = 1 + (int)(g() % 200);
        std::vector<Cand> all(n);
        for (int i = 0; i < n; i++) all[i] = {(int32_t)(g() % 20), i};  // many ties
        TopC inorder(C), shuffled(C);
        for (auto& c : all) if (c.score > inorder.threshold()) inorder.push(c);  // fast path used by the scan
        std::vector<Cand> sh(all);
        std::shuffle(sh.begin(), sh.end(), g);
        for (auto& c : sh) shuffled.push(c);
        std::sort(all.begin(), all.end(), cand_better);
        all.resize(std::min(n, C));
        auto a = inorder.sorted(), b = shuffled.sorted();
        bool same = a.size() == all.size() && b.size() == all.size();
        for (size_t i = 0; same && i < all.size(); i++) same = a[i].id == all[i].id && b[i].id == all[i].id;
        CHECK(same);
    }

    // band keys: deterministic, order- and band-sensitive
    uint64_t t1[4] = {1, 2, 3, 4}, t2[4] = {2, 1, 3, 4};
    CHECK(band_key(t1, 4, 0) == band_key(t1, 4, 0));
    CHECK(band_key(t1, 4, 0) != band_key(t2, 4, 0));
    CHECK(band_key(t1, 4, 0) != band_key(t1, 4, 1));
    // band index returns exactly the rows that agree with the query on all r slots of some band
    {
        int n = 3000, k = 12, L = 3, r = 4;
        std::vector<uint64_t> ids((size_t)n * k);
        for (auto& v : ids) v = g() % 3;  // tiny alphabet -> many collisions
        BandIndex B = build_band_index(ids, n, k, L, r);
        std::vector<uint32_t> stamp(n, 0);
        std::vector<int32_t> cand;
        int bad = 0;
        for (int q = 0; q < 200; q++) {
            std::vector<uint64_t> qs(k);
            for (auto& v : qs) v = g() % 3;
            band_candidates(B, qs.data(), stamp, (uint32_t)(q + 1), cand);
            std::set<int32_t> got(cand.begin(), cand.end()), want;
            CHECK(got.size() == cand.size());  // no duplicates
            for (int i = 0; i < n; i++)
                for (int l = 0; l < L; l++) {
                    bool eq = true;
                    for (int j = 0; j < r; j++) eq &= ids[(size_t)i * k + l * r + j] == qs[l * r + j];
                    if (eq) { want.insert(i); break; }
                }
            bad += got != want;
        }
        CHECK(bad == 0);
        std::vector<uint64_t> none(k, 7);  // value never stored
        band_candidates(B, none.data(), stamp, 9999, cand);
        CHECK(cand.empty());
    }

    // b-bit band index: buckets are formed from truncated ids (1-bit slots collide far more than 64-bit ones)
    {
        int n = 2000, k = 8, L = 2, r = 4;
        std::vector<uint64_t> ids((size_t)n * k);
        for (auto& v : ids) v = g();
        std::vector<uint64_t> t1b = truncate_all(ids, 1);
        for (uint64_t v : t1b) CHECK(v <= 1);
        BandIndex B1 = build_band_index(t1b, n, k, L, r), B64 = build_band_index(ids, n, k, L, r);
        std::vector<uint32_t> stamp(n, 0);
        std::vector<int32_t> c1, c64;
        band_candidates(B1, t1b.data(), stamp, 1, c1);
        band_candidates(B64, ids.data(), stamp, 2, c64);
        CHECK(c1.size() > 10 * c64.size());  // 16 buckets per band vs ~unique keys
        CHECK(std::find(c1.begin(), c1.end(), 0) != c1.end());  // row 0 is its own candidate
    }

    // recall
    int32_t truth[5] = {10, 11, 12, 13, 14}, ret[5] = {11, 99, 10, 13, 98};
    CHECK(recall_at(ret, 5, truth, 1) == 0.0);   // top-1 returned (11) is not the true top-1 (10)
    CHECK(recall_at(ret, 5, truth, 5) == 0.6);   // {10, 11, 13}
    CHECK(recall_at(ret, 2, truth, 5) == 0.2);   // fewer returned than R
    CHECK(std::fabs(recall_at(truth, 5, truth, 5) - 1.0) < 1e-15);
    double tJ[3] = {0.9, 0.8, 0.8}, rJ[3] = {0.9, 0.8, 0.7};
    CHECK(std::fabs(recall_at_tie_aware(rJ, 3, tJ, 3) - 2.0 / 3) < 1e-15);
    double rJ2[3] = {0.8, 0.8, 0.9};             // tied item instead of a listed neighbour: still a hit
    CHECK(recall_at_tie_aware(rJ2, 3, tJ, 3) == 1.0);

    std::printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
