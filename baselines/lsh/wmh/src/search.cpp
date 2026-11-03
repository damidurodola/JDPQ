// Retrieval over weighted-MinHash sketches: Mode A (exhaustive sketch scan),
// Mode B (banded LSH), optional exact-J_w rerank; plus an exact reference.
//
//   search --data <dataset_dir> --out <csv> --algo dart|ficws|bmh2 --k K
//          [--bits 1,2,4,8,16,64] [--threads T] [--modes A,B] [--r 2,4,8,16,32,64] [--S 0,100,500,1000,2000]
//          [--seed 20261014] [--st-queries 200] [--batch 64] [--prune-frac 0.5]
//          [--prune-queries 50] [--mem-budget-gb 12] [--upstream-git H] [--xxhash-git H]
//   search --data <dataset_dir> --out <csv> --exact [--st-queries 200]
//
// One CSV row per configuration (appended). S = 0 means no rerank (rank by
// match count, return top-100); S > 0 means shortlist the top-S by match count
// and return the top-100 of the shortlist by exact J_w on the original vectors.
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "csr.hpp"
#include "gt.hpp"
#include "index.hpp"
#include "sketch.hpp"

using namespace wmh;

// All timings use CLOCK_UPTIME_RAW, which stops while the machine sleeps; a timed
// pass during which the system slept (uptime and monotonic clocks disagree) is
// repeated, because caches and clock frequencies are disturbed after a wake.
// On Linux CLOCK_MONOTONIC does not count suspend and CLOCK_BOOTTIME does.
#ifdef __APPLE__
static double now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }
static double now_incl_sleep() { return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) * 1e-9; }
#else
static double clk(clockid_t c) { timespec t; clock_gettime(c, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double now() { return clk(CLOCK_MONOTONIC); }
static double now_incl_sleep() { return clk(CLOCK_BOOTTIME); }
#endif
static bool slept_between(double up0, double mono0) {
    double up = now() - up0, mono = now_incl_sleep() - mono0;
    return mono - up > 0.05 + 0.01 * up;
}
static const int R = 100;  // returned list length / recall depth

// ------------------------------------------------------------------ arguments
struct Args {
    std::string data, out, algo = "dart", upstream_git = "unknown", xxhash_git = "unknown";
    int threads = 0;  // 0 = all cores
    int k = 64, st_queries = 3000, batch = 64, prune_queries = 50;
    double prune_frac = 0.5, mem_budget_gb = 400.0;
    uint64_t seed = 20261014ull;
    bool exact = false;
    std::vector<int> bits = {8, 16, 64}, rs = {2, 4, 8, 16, 32, 64}, S = {0, 100, 200, 500,1000};
    std::vector<std::string> modes = {"B"};
    // std::vector<std::string> modes = {"A", "B"};
};

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
    return v;
}
static std::vector<int> split_int(const std::string& s) {
    std::vector<int> v;
    for (auto& t : split(s)) v.push_back(std::stoi(t));
    return v;
}

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string f = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + f); return argv[++i]; };
        if (f == "--data") a.data = val();
        else if (f == "--out") a.out = val();
        else if (f == "--algo") a.algo = val();
        else if (f == "--k") a.k = std::stoi(val());
        else if (f == "--threads") a.threads = std::stoi(val());
        else if (f == "--bits") a.bits = split_int(val());
        else if (f == "--modes") a.modes = split(val());
        else if (f == "--r") a.rs = split_int(val());
        else if (f == "--S") a.S = split_int(val());
        else if (f == "--seed") a.seed = std::stoull(val());
        else if (f == "--st-queries") a.st_queries = std::stoi(val());
        else if (f == "--batch") a.batch = std::stoi(val());
        else if (f == "--prune-frac") a.prune_frac = std::stod(val());
        else if (f == "--prune-queries") a.prune_queries = std::stoi(val());
        else if (f == "--mem-budget-gb") a.mem_budget_gb = std::stod(val());
        else if (f == "--upstream-git") a.upstream_git = val();
        else if (f == "--xxhash-git") a.xxhash_git = val();
        else if (f == "--exact") a.exact = true;
        else throw std::runtime_error("unknown flag " + f);
    }
    if (a.data.empty() || a.out.empty()) throw std::runtime_error("--data and --out are required");
    return a;
}

// ------------------------------------------------------------------ CSV output
static const char* kColumns[] = {
    "dataset", "status", "algorithm", "mode", "k", "L", "r", "slot_bits", "rerank", "S", "seed", "n_db", "n_queries",
    "threads_mt", "batch_size", "warmup", "query_sketch_in_time", "n_queries_st",
    "recall_at_1", "recall_at_10", "recall_at_100", "recall_at_100_tie_aware", "returned_per_query",
    "candidate_recall_at_100", "shortlist_recall_at_100", "diag_check", "shortlist_capacity",
    "candidates_per_query", "heldout_candidates_est",
    "sketch_db_wall_s", "sketch_db_threads", "sketch_us_per_vector_st", "index_build_s",
    "query_us_st", "query_sketch_us_st", "filter_us_st", "rerank_us_st", "qps_st", "qps_mt", "qps_mt_filter_only",
    "timing_check", "st_mt_identical", "sleep_retries",
    "mem_sketch_bytes", "mem_tables_bytes", "mem_ids_bytes", "mem_raw_bytes", "mem_excl_raw_mb", "mem_incl_raw_mb",
    "mem_total_mb", "mem_total_includes_raw", "sketch_checksum", "upstream_git", "xxhash_git", "date", "host"};

struct Row {
    std::map<std::string, std::string> v;
    template <class T> void set(const std::string& c, T x) { std::ostringstream o; o.precision(10); o << x; v[c] = o.str(); }
};

static void write_row(const std::string& path, const Row& r) {
    bool fresh = access(path.c_str(), F_OK) != 0;
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) throw std::runtime_error("cannot open " + path);
    size_t nc = sizeof(kColumns) / sizeof(kColumns[0]);
    if (fresh)
        for (size_t i = 0; i < nc; i++) std::fprintf(f, "%s%s", kColumns[i], i + 1 < nc ? "," : "\n");
    for (size_t i = 0; i < nc; i++) {
        auto it = r.v.find(kColumns[i]);
        std::string val = it == r.v.end() ? "" : it->second;
        if (val.find_first_of(",\"\n") != std::string::npos) {  // RFC 4180 quoting
            std::string q = "\"";
            for (char ch : val) q += ch == '"' ? std::string("\"\"") : std::string(1, ch);
            val = q + "\"";
        }
        std::fprintf(f, "%s%s", val.c_str(), i + 1 < nc ? "," : "\n");
    }
    std::fclose(f);
}

static std::string today() {
    char b[32];
    std::time_t t = std::time(nullptr);
    std::strftime(b, sizeof b, "%Y-%m-%d", std::localtime(&t));
    return b;
}
static std::string hostname() { char b[256] = {0}; gethostname(b, sizeof b - 1); return b; }
static std::string basename_of(std::string p) {
    while (!p.empty() && p.back() == '/') p.pop_back();
    auto s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}
static size_t raw_bytes(const Csr& X) { return X.indptr.size() * 8 + X.indices.size() * 4 + X.data.size() * 8 + X.l1.size() * 8; }
// Stored id per database vector, counted as 8 bytes (int64, as faiss does) so memory is comparable with IVF-PQ.
static const size_t kIdBytes = 8;
// Raw vectors are counted as dense float32 (n * d * 4), the same as the PQ side, not as the CSR actually held.
static size_t raw_bytes_acct(const Csr& X) { return (size_t)X.n * X.d * 4; }

// ------------------------------------------------------------------ query execution
// Two timed phases. (1) Shortlist pass: sketch each query (unless precomputed)
// and filter (Mode A scan, or Mode B candidates) keeping the top Cmax rows by
// matching slots. (2) Rerank pass for one setting S: rerank the first S
// shortlist entries by exact J_w and return the top 100 (S = 0: return the first
// 100 as is). Under the total order (matches desc, row asc) the first S entries
// of the top-Cmax list are exactly the top-S list, so one shortlist pass serves
// every S; the end-to-end time of setting S is shortlist pass + its rerank pass.
struct Shortlists {
    int C = 0;
    std::vector<Cand> c;  // nq x C, best first
    std::vector<int> n;   // valid entries per query
};

struct PassOut {
    double wall = 0, t_sketch = 0, t_filter = 0, t_rerank = 0;  // phase times are thread-seconds
    double cand = 0, cand_hits = 0;                             // cand_hits: diagnostic pass only
    bool slept = false;
    void add(const PassOut& o) {
        t_sketch += o.t_sketch; t_filter += o.t_filter; t_rerank += o.t_rerank;
        cand += o.cand; cand_hits += o.cand_hits;
    }
};

struct Engine {
    const Csr& X;
    const Csr& Q;
    const Sketcher& S;
    int k;
};

// Shortlist phase for queries ql[b0:b1). Mode A if B == nullptr.
template <class T>
static void filter_batch(const Engine& E, const StoredSketches<T>& st, const BandIndex* B,
                         const std::vector<uint64_t>* pre64, const std::vector<int64_t>& ql, size_t b0, size_t b1,
                         Shortlists& out, PassOut& acc, std::vector<uint32_t>& stamp, uint32_t& epoch,
                         const GroundTruth* dg) {
    const int k = E.k;
    const size_t nb = b1 - b0;
    std::vector<uint64_t> q64(nb * k);
    std::vector<T> qT(nb * k);
    double t0 = now();
    for (size_t i = 0; i < nb; i++) {
        int64_t q = ql[b0 + i];
        if (pre64) std::memcpy(&q64[i * k], &(*pre64)[(size_t)q * k], k * 8);
        else E.S.sketch(E.Q.idx(q), E.Q.val(q), E.Q.nnz(q), &q64[i * k]);
        truncate_into(&q64[i * k], k, st.bits, &qT[i * k]);
    }
    double t1 = now();
    std::vector<TopC> heaps(nb, TopC(out.C));
    if (!B) {
        // Mode A: rows in cache-sized chunks; each chunk is scanned for every query in the batch.
        const int64_t chunk = std::max<int64_t>(16, (256 << 10) / ((int64_t)k * sizeof(T)));
        for (int64_t c0 = 0; c0 < st.n; c0 += chunk) {
            int64_t c1 = std::min(st.n, c0 + chunk);
            for (size_t i = 0; i < nb; i++) {
                TopC& h = heaps[i];
                const T* qv = &qT[i * k];
                int32_t thr = h.threshold();
                for (int64_t row = c0; row < c1; row++) {
                    int m = count_eq(qv, st.row(row), k);
                    if (m > thr) { h.push({m, (int32_t)row}); thr = h.threshold(); }
                }
            }
        }
        acc.cand += (double)st.n * nb;
        if (dg) acc.cand_hits += (double)R * nb;  // every row is a candidate
    } else {
        std::vector<int32_t> cl;
        std::vector<uint64_t> qb;
        for (size_t i = 0; i < nb; i++) {
            // bucket keys come from the query's ids at the stored slot width (as the tables were built)
            const uint64_t* qk = &q64[i * k];
            if (st.bits < 64) {
                qb.resize(k);
                for (int j = 0; j < k; j++) qb[j] = truncate_id(qk[j], st.bits);
                qk = qb.data();
            }
            band_candidates(*B, qk, stamp, ++epoch, cl);
            if (dg) {  // diagnostic pass: how many of the true top-100 are among the candidates
                const int32_t* tru = &dg->ids[(size_t)ql[b0 + i] * dg->K];
                for (int j = 0; j < R; j++) acc.cand_hits += stamp[tru[j]] == epoch;
            }
            const T* qv = &qT[i * k];
            for (int32_t row : cl) heaps[i].push({count_eq(qv, st.row(row), k), row});
            acc.cand += cl.size();
        }
    }
    for (size_t i = 0; i < nb; i++) {
        int64_t q = ql[b0 + i];
        auto L = heaps[i].sorted();
        std::copy(L.begin(), L.end(), out.c.begin() + (size_t)q * out.C);
        out.n[q] = (int)L.size();
    }
    double t2 = now();
    acc.t_sketch += t1 - t0;
    acc.t_filter += t2 - t1;
}

// Rerank phase for setting S on queries ql[b0:b1).
static void rerank_batch(const Engine& E, const Shortlists& sl, int S, const std::vector<int64_t>& ql, size_t b0,
                         size_t b1, std::vector<int32_t>& ids, std::vector<int>& nret, PassOut& acc) {
    double t0 = now();
    for (size_t i = b0; i < b1; i++) {
        int64_t q = ql[i];
        const Cand* L = &sl.c[(size_t)q * sl.C];
        if (S > 0) {
            int m = std::min(S, sl.n[q]);
            std::vector<Scored> sc(m);
            for (int j = 0; j < m; j++) sc[j] = {jw(E.Q, q, E.X, L[j].id), L[j].id};
            std::sort(sc.begin(), sc.end(), better);
            int n = std::min<int>(R, m);
            for (int j = 0; j < n; j++) ids[(size_t)q * R + j] = sc[j].id;
            nret[q] = n;
        } else {
            int n = std::min(R, sl.n[q]);
            for (int j = 0; j < n; j++) ids[(size_t)q * R + j] = L[j].id;
            nret[q] = n;
        }
    }
    acc.t_rerank += now() - t0;
}

// Runs `batch(b0, b1, acc, scratch...)` over ql in batches of QB on `threads`
// threads, after one untimed warm-up batch. Returns wall time and phase sums.
template <class F>
static PassOut timed_batches(size_t nq, int threads, int QB, bool need_stamp, int64_t n_db, F batch) {
    size_t nbatch = (nq + QB - 1) / QB;
    {
        PassOut w;
        std::vector<uint32_t> stamp(need_stamp ? n_db : 0, 0);
        uint32_t ep = 0;
        batch((size_t)0, std::min(nq, (size_t)QB), w, stamp, ep);
    }
    PassOut o;
    double w0 = now(), m0 = now_incl_sleep();
#pragma omp parallel num_threads(threads)
    {
        PassOut local;
        std::vector<uint32_t> stamp(need_stamp ? n_db : 0, 0);
        uint32_t ep = 0;
#pragma omp for schedule(dynamic, 1)
        for (size_t b = 0; b < nbatch; b++) batch(b * QB, std::min(nq, (b + 1) * QB), local, stamp, ep);
#pragma omp critical
        o.add(local);
    }
    o.wall = now() - w0;
    o.slept = slept_between(w0, m0);
    return o;
}

template <class T>
static PassOut filter_pass(const Engine& E, const StoredSketches<T>& st, const BandIndex* B,
                           const std::vector<uint64_t>* pre64, const std::vector<int64_t>& ql, int threads, int QB,
                           Shortlists& out, const GroundTruth* dg = nullptr) {
    out.c.resize((size_t)E.Q.n * out.C);
    out.n.assign(E.Q.n, 0);
    return timed_batches(ql.size(), threads, QB, B != nullptr, E.X.n,
                         [&](size_t b0, size_t b1, PassOut& acc, std::vector<uint32_t>& stamp, uint32_t& ep) {
                             filter_batch(E, st, B, pre64, ql, b0, b1, out, acc, stamp, ep, dg);
                         });
}

static PassOut rerank_pass(const Engine& E, const Shortlists& sl, int S, const std::vector<int64_t>& ql, int threads,
                           int QB, std::vector<int32_t>& ids, std::vector<int>& nret) {
    ids.assign((size_t)E.Q.n * R, -1);
    nret.assign(E.Q.n, 0);
    return timed_batches(ql.size(), threads, QB, false, E.X.n,
                         [&](size_t b0, size_t b1, PassOut& acc, std::vector<uint32_t>&, uint32_t&) {
                             rerank_batch(E, sl, S, ql, b0, b1, ids, nret, acc);
                         });
}

struct Recalls { double r1 = 0, r10 = 0, r100 = 0, r100t = 0, ret = 0; };

static Recalls eval_recall(const std::vector<int32_t>& ids, const std::vector<int>& nret, const Csr& Q, const Csr& X,
                           const GroundTruth& g) {
    Recalls rc;
    for (int64_t q = 0; q < Q.n; q++) {
        const int32_t* ret = &ids[(size_t)q * R];
        const int32_t* tru = &g.ids[(size_t)q * g.K];
        int n = nret[q];
        rc.r1 += recall_at(ret, n, tru, 1);
        rc.r10 += recall_at(ret, n, tru, 10);
        rc.r100 += recall_at(ret, n, tru, 100);
        std::vector<double> J(n);
        for (int j = 0; j < n; j++) J[j] = jw(Q, q, X, ret[j]);
        rc.r100t += recall_at_tie_aware(J.data(), n, &g.sims[(size_t)q * g.K], 100);
        rc.ret += n;
    }
    rc.r1 /= Q.n; rc.r10 /= Q.n; rc.r100 /= Q.n; rc.r100t /= Q.n; rc.ret /= Q.n;
    return rc;
}

// Fraction of the true top-100 among the first C entries of each shortlist.
static double shortlist_recall(const Shortlists& sl, int C, const GroundTruth& g) {
    double hits = 0;
    for (int64_t q = 0; q < g.nq; q++) {
        std::vector<int32_t> t(&g.ids[(size_t)q * g.K], &g.ids[(size_t)q * g.K] + R);
        std::sort(t.begin(), t.end());
        int m = std::min(C, sl.n[q]);
        for (int j = 0; j < m; j++) hits += std::binary_search(t.begin(), t.end(), sl.c[(size_t)q * sl.C + j].id);
    }
    return hits / ((double)g.nq * R);
}

// ------------------------------------------------------------------ one configuration group
struct Common {
    Args a;
    std::string dataset;
    const Csr* X;
    const Csr* Q;
    const GroundTruth* g;
    int threads;
    std::vector<int64_t> all_q, st_q;
    double sketch_wall = 0, sketch_us_st = 0;
    uint64_t checksum = 0;
    const std::vector<uint64_t>* q64_all = nullptr;  // precomputed query sketches (filter-only runs)
};

static Row base_row(const Common& c) {
    Row r;
    r.set("dataset", c.dataset);
    r.set("status", "ok");
    r.set("algorithm", c.a.algo);
    r.set("k", c.a.k);
    r.set("seed", c.a.seed);
    r.set("n_db", c.X->n);
    r.set("n_queries", c.Q->n);
    r.set("threads_mt", c.threads);
    r.set("batch_size", c.a.batch);
    r.set("warmup", "1 batch per pass; MT = best of 2 passes");
    r.set("query_sketch_in_time", "yes for qps_st and qps_mt; no for qps_mt_filter_only");
    r.set("n_queries_st", c.st_q.size());
    r.set("sketch_db_wall_s", c.sketch_wall);
    r.set("sketch_db_threads", c.threads);
    r.set("sketch_us_per_vector_st", c.sketch_us_st);
    char cs[32];
    std::snprintf(cs, sizeof cs, "%016llx", (unsigned long long)c.checksum);
    r.set("sketch_checksum", std::string(cs));
    r.set("upstream_git", c.a.upstream_git);
    r.set("xxhash_git", c.a.algo == "bmh2" ? c.a.xxhash_git : std::string(""));
    r.set("date", today());
    r.set("host", hostname());
    r.set("mem_ids_bytes", (size_t)c.X->n * kIdBytes);
    r.set("mem_raw_bytes", raw_bytes_acct(*c.X));
    return r;
}

// All rerank settings S for one (mode, slot width, tables). Shortlist capacity
// C(S) = max(S, 100): S = 0 and S = 100 share a C = 100 pass, every other S gets
// a pass of its own capacity, because the scan cost grows with C (measured ~2x
// between C = 100 and C = 2000 on cemetery), so a shared C = 2000 pass would
// overstate the cost of small S. Each S then gets its own timed rerank pass.
template <class T>
static void run_group(const Common& c, const Engine& E, const StoredSketches<T>& st, const BandIndex* B, Row r,
                      double build_s, size_t tables_bytes) {
    const int QB = c.a.batch;
    int Cmax = R;
    for (int S : c.a.S) Cmax = std::max(Cmax, S);
    r.set("slot_bits", st.bits);
    r.set("index_build_s", build_s);
    // Each timed pass is repeated (up to 5 times) if the machine slept during it.
    int retries = 0;
    auto timed = [&](auto pass) {
        for (int attempt = 0;; attempt++) {
            PassOut o = pass();
            if (!o.slept || attempt == 4) { if (o.slept) retries += 100; return o; }  // +100 marks a failed guard
            retries++;
        }
    };
    auto best2 = [&](auto pass) {  // multi-thread timings: best (minimum wall) of 2 timed passes
        PassOut a = timed(pass);
        a.wall = std::min(a.wall, timed(pass).wall);
        return a;
    };
    // Untimed diagnostic pass: candidate recall (independent of the capacity).
    Shortlists sl_dg;
    sl_dg.C = R;
    PassOut dg = filter_pass(E, st, B, c.q64_all, c.all_q, c.threads, QB, sl_dg, c.g);
    // --st-queries 0 gives a multi-thread-only run: no single-thread pass is timed and the *_st columns stay empty.
    const bool has_st = !c.st_q.empty();
    double nst = c.st_q.size(), nq = c.all_q.size();
    double cand_rec = dg.cand_hits / (nq * R);

    int last_cap = -1;
    Shortlists sl_mt, sl_fo, sl_st;
    PassOut sf, fo, sst;
    std::vector<int> Ss(c.a.S);
    std::sort(Ss.begin(), Ss.end());
    for (int S : Ss) {
        int retries0 = retries;
        int cap = std::max(S, R);
        if (cap != last_cap) {  // timed shortlist passes for this capacity
            sl_mt.C = sl_fo.C = sl_st.C = cap;
            sf = best2([&] { return filter_pass(E, st, B, nullptr, c.all_q, c.threads, QB, sl_mt); });
            fo = best2([&] { return filter_pass(E, st, B, c.q64_all, c.all_q, c.threads, QB, sl_fo); });
            if (has_st) sst = timed([&] { return filter_pass(E, st, B, nullptr, c.st_q, 1, QB, sl_st); });
            last_cap = cap;
        }
        double qps_fo = nq / fo.wall;
        std::vector<int32_t> ids, ids_st;
        std::vector<int> nret, nret_st;
        PassOut rr = best2([&] { return rerank_pass(E, sl_mt, S, c.all_q, c.threads, QB, ids, nret); });
        PassOut rst;
        if (has_st) rst = timed([&] { return rerank_pass(E, sl_st, S, c.st_q, 1, QB, ids_st, nret_st); });
        Recalls rc = eval_recall(ids, nret, *c.Q, *c.X, *c.g);
        bool same = true;
        for (int64_t q : c.st_q)
            for (int j = 0; j < R; j++) same &= ids_st[(size_t)q * R + j] == ids[(size_t)q * R + j];
        double short_rec = shortlist_recall(sl_mt, S > 0 ? S : R, *c.g);
        double qps_mt = nq / (sf.wall + rr.wall), qps_st = has_st ? nst / (sst.wall + rst.wall) : std::nan("");
        Row row = r;
        row.set("rerank", S > 0 ? 1 : 0);
        row.set("S", S);
        row.set("shortlist_capacity", cap);
        row.set("recall_at_1", rc.r1);
        row.set("recall_at_10", rc.r10);
        row.set("recall_at_100", rc.r100);
        row.set("recall_at_100_tie_aware", rc.r100t);
        row.set("returned_per_query", rc.ret);
        row.set("candidate_recall_at_100", cand_rec);
        row.set("shortlist_recall_at_100", short_rec);
        // Without rerank the shortlist is the returned list, so the two recalls must agree.
        row.set("diag_check", S > 0 || std::fabs(short_rec - rc.r100) < 1e-12 ? "ok" : "MISMATCH");
        row.set("candidates_per_query", sf.cand / nq);
        if (has_st) {
            row.set("query_us_st", (sst.wall + rst.wall) / nst * 1e6);
            row.set("query_sketch_us_st", sst.t_sketch / nst * 1e6);
            row.set("filter_us_st", sst.t_filter / nst * 1e6);
            row.set("rerank_us_st", rst.t_rerank / nst * 1e6);
            row.set("qps_st", qps_st);
        }
        row.set("qps_mt", qps_mt);
        row.set("qps_mt_filter_only", qps_fo);
        // End-to-end includes everything filter-only does plus query sketching (and rerank):
        // it cannot legitimately be faster (5% slack for timer noise).
        row.set("timing_check", qps_mt <= 1.05 * qps_fo ? "ok" : "SUSPICIOUS");
        row.set("st_mt_identical", !has_st ? "n/a" : same ? "yes" : "NO");
        row.set("sleep_retries", retries - retries0);
        size_t sk = st.bytes();
        size_t excl = sk + tables_bytes + (size_t)c.X->n * kIdBytes;
        size_t incl = excl + raw_bytes_acct(*c.X);
        row.set("mem_sketch_bytes", sk);
        row.set("mem_tables_bytes", tables_bytes);
        row.set("mem_excl_raw_mb", excl / 1048576.0);
        row.set("mem_incl_raw_mb", incl / 1048576.0);
        row.set("mem_total_mb", (S > 0 ? incl : excl) / 1048576.0);
        row.set("mem_total_includes_raw", S > 0 ? "yes" : "no");
        write_row(c.a.out, row);
        std::printf("  %s k=%d L=%s r=%s b=%d S=%d | R@1=%.3f R@10=%.3f R@100=%.3f | candR=%.3f shortR=%.3f %s | "
                    "cand/q=%.0f | qps st=%.1f mt=%.1f filter=%.1f | mem=%.1fMB | %s %s\n",
                    row.v["mode"].c_str(), c.a.k, row.v["L"].c_str(), row.v["r"].c_str(), st.bits, S, rc.r1, rc.r10,
                    rc.r100, cand_rec, short_rec, row.v["diag_check"].c_str(), sf.cand / nq, qps_st, qps_mt, qps_fo,
                    (S > 0 ? incl : excl) / 1048576.0, row.v["timing_check"].c_str(), same ? "" : "ST/MT DIFFER");
    }
}

// Average union size over held-out queries, computed directly from the sketches
// (no tables needed), used only to prune (L, r) settings.
static double heldout_candidates(const std::vector<uint64_t>& ids64, int64_t n, int k, int L, int r,
                                 const std::vector<uint64_t>& t64, int nt) {
    double total = 0;
#pragma omp parallel for reduction(+ : total) schedule(static)
    for (int64_t i = 0; i < n; i++) {
        const uint64_t* x = &ids64[(size_t)i * k];
        for (int q = 0; q < nt; q++) {
            const uint64_t* y = &t64[(size_t)q * k];
            for (int l = 0; l < L; l++) {
                int j = 0;
                while (j < r && x[l * r + j] == y[l * r + j]) j++;
                if (j == r) { total += 1; break; }
            }
        }
    }
    return total / nt;
}

template <class T>
static void sweep_bits(const Common& c, const Engine& E, const std::vector<uint64_t>& ids64, int bits,
                       const BandIndex* B, Row r, double tables_build_s, size_t tables_bytes) {
    double t0 = now();
    StoredSketches<T> st = make_stored<T>(ids64, c.X->n, c.a.k, bits);
    double trunc_s = now() - t0;
    run_group(c, E, st, B, r, tables_build_s + trunc_s, tables_bytes);
}

static void sweep_one_bit(const Common& c, const Engine& E, const std::vector<uint64_t>& ids64, int b, const BandIndex* B,
                          const Row& r, double build_s, size_t tb) {
    {
        if (b == 1 || b == 2 || b == 4 || b == 8) sweep_bits<uint8_t>(c, E, ids64, b, B, r, build_s, tb);  // b < 8 packed notionally
        else if (b == 16) sweep_bits<uint16_t>(c, E, ids64, 16, B, r, build_s, tb);
        else if (b == 32) sweep_bits<uint32_t>(c, E, ids64, 32, B, r, build_s, tb);
        else if (b == 64) sweep_bits<uint64_t>(c, E, ids64, 64, B, r, build_s, tb);
        else throw std::runtime_error("unsupported slot width");
    }
}

static void sweep_all_bits(const Common& c, const Engine& E, const std::vector<uint64_t>& ids64, const BandIndex* B,
                           const Row& r, double build_s, size_t tb) {
    for (int b : c.a.bits) sweep_one_bit(c, E, ids64, b, B, r, build_s, tb);
}

// ------------------------------------------------------------------ exact reference
static void run_exact(const Args& a, const Csr& X, const Csr& Q, const GroundTruth& g, int threads,
                      const std::vector<int64_t>& st_q) {
    double t0 = now();
    InvertedIndex inv(X);
    double build = now() - t0;
    auto run_once = [&](const std::vector<int64_t>& ql, int nth, std::vector<int32_t>& ids, bool& slept) {
        double w0 = now(), m0 = now_incl_sleep();
#pragma omp parallel num_threads(nth)
        {
            std::vector<double> acc(X.n), sc;
#pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < ql.size(); i++) {
                inv.scores(Q, ql[i], X, acc, sc);
                auto top = topk_of(sc, R);
                for (int j = 0; j < R; j++) ids[(size_t)ql[i] * R + j] = top[j].id;
            }
        }
        double w = now() - w0;
        slept = slept_between(w0, m0);
        return w;
    };
    auto run = [&](const std::vector<int64_t>& ql, int nth, std::vector<int32_t>& ids) {
        bool slept = true;
        double w = 0;
        for (int attempt = 0; slept && attempt < 5; attempt++) w = run_once(ql, nth, ids, slept);
        if (slept) throw std::runtime_error("system kept sleeping during exact timing");
        return w;
    };
    std::vector<int64_t> all(Q.n);
    for (int64_t q = 0; q < Q.n; q++) all[q] = q;
    std::vector<int32_t> ids((size_t)Q.n * R, -1), ids_st((size_t)Q.n * R, -1);
    std::vector<int64_t> warm(all.begin(), all.begin() + std::min<int64_t>(8, Q.n));
    run(warm, threads, ids);
    double wmt = run(all, threads, ids);
    double wst = st_q.empty() ? std::nan("") : run(st_q, 1, ids_st);
    int64_t mism = 0;
    for (int64_t q = 0; q < Q.n; q++)
        for (int j = 0; j < R; j++) mism += ids[(size_t)q * R + j] != g.ids[(size_t)q * g.K + j];
    Row r;
    r.set("dataset", basename_of(a.data));
    r.set("status", mism ? "MISMATCH_WITH_GT" : "ok");
    r.set("algorithm", "exact_inverted_index");
    r.set("mode", "exact");
    r.set("rerank", 0); r.set("S", 0);
    r.set("n_db", X.n); r.set("n_queries", Q.n);
    r.set("threads_mt", threads); r.set("batch_size", 1); r.set("warmup", "8 queries");
    r.set("query_sketch_in_time", "n/a"); r.set("n_queries_st", st_q.size());
    double rec = 1.0 - (double)mism / ((double)Q.n * R);
    r.set("recall_at_1", rec); r.set("recall_at_10", rec); r.set("recall_at_100", rec); r.set("recall_at_100_tie_aware", rec);
    r.set("returned_per_query", R);
    r.set("candidates_per_query", X.n);
    r.set("index_build_s", build);
    if (!st_q.empty()) {
        r.set("query_us_st", wst / st_q.size() * 1e6);
        r.set("filter_us_st", wst / st_q.size() * 1e6);
        r.set("qps_st", st_q.size() / wst);
    }
    r.set("qps_mt", Q.n / wmt);
    r.set("qps_mt_filter_only", Q.n / wmt);
    r.set("timing_check", "ok");
    size_t inv_bytes = inv.ptr.size() * 8 + inv.rows.size() * 8 + inv.vals.size() * 8;
    size_t raw = raw_bytes_acct(X);
    r.set("mem_sketch_bytes", 0); r.set("mem_tables_bytes", inv_bytes); r.set("mem_ids_bytes", (size_t)X.n * kIdBytes);
    r.set("mem_raw_bytes", raw);
    r.set("mem_excl_raw_mb", (inv_bytes + X.n * kIdBytes) / 1048576.0);
    r.set("mem_incl_raw_mb", (inv_bytes + X.n * kIdBytes + raw) / 1048576.0);
    r.set("mem_total_mb", (inv_bytes + X.n * kIdBytes + raw) / 1048576.0);
    r.set("mem_total_includes_raw", "yes");
    r.set("upstream_git", a.upstream_git); r.set("date", today()); r.set("host", hostname());
    write_row(a.out, r);
    std::printf("exact: qps st=%.1f mt=%.1f, mismatches vs GT=%lld, mem=%.1fMB\n", st_q.size() / wst, Q.n / wmt,
                (long long)mism, (inv_bytes + X.n * kIdBytes + raw) / 1048576.0);
}

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Args a = parse(argc, argv);
    int threads = a.threads > 0 ? a.threads : omp_get_num_procs();
    omp_set_num_threads(threads);
    Csr X = load_csr(a.data + "/db.npz");
    Csr Q = load_csr(a.data + "/query.npz");
    Csr T = load_csr(a.data + "/train.npz");
    GroundTruth g;
    if (!load_gt(a.data + "/gt_query.bin", g, fingerprint(Q), fingerprint(X)) || g.K < R)
        throw std::runtime_error("missing or stale ground truth; run groundtruth first");
    for (int64_t i = 0; i < X.n; i++) if (X.nnz(i) == 0) throw std::runtime_error("empty db vector");

    Common c;
    c.a = a;
    c.dataset = basename_of(a.data);
    c.X = &X; c.Q = &Q; c.g = &g;
    c.threads = threads;
    for (int64_t q = 0; q < Q.n; q++) c.all_q.push_back(q);
    c.st_q.assign(c.all_q.begin(), c.all_q.begin() + std::min<int64_t>(a.st_queries, Q.n));

    if (a.exact) { run_exact(a, X, Q, g, threads, c.st_q); return 0; }

    Sketcher S(parse_algo(a.algo), a.k, a.seed);
    // single-thread sketch cost on a fixed sample (us per vector)
    {
        std::vector<int64_t> sample;
        for (int64_t i = 0; i < std::min<int64_t>(500, X.n); i++) sample.push_back(i * (X.n / std::min<int64_t>(500, X.n)));
        omp_set_num_threads(1);
        double t0 = now();
        sketch_rows(S, X, &sample);
        c.sketch_us_st = (now() - t0) / sample.size() * 1e6;
        omp_set_num_threads(threads);
    }
    double t0 = now();
    std::vector<uint64_t> ids64 = sketch_rows(S, X);
    c.sketch_wall = now() - t0;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t v : ids64) { h ^= v; h *= 1099511628211ull; }
    c.checksum = h;
    std::vector<uint64_t> q64 = sketch_rows(S, Q);
    c.q64_all = &q64;
    std::printf("%s %s k=%d: sketched db in %.1fs (%d threads), %.1f us/vector single-thread, checksum %016llx\n",
                c.dataset.c_str(), a.algo.c_str(), a.k, c.sketch_wall, threads, c.sketch_us_st, (unsigned long long)h);

    Engine E{X, Q, S, a.k};
    for (const std::string& mode : a.modes) {
        if (mode == "A") {
            Row r = base_row(c);
            r.set("mode", "A");
            sweep_all_bits(c, E, ids64, nullptr, r, 0.0, 0);
        } else if (mode == "B") {
            int nt = std::min<int64_t>(a.prune_queries, T.n);
            std::vector<int64_t> trows(nt);
            for (int i = 0; i < nt; i++) trows[i] = i;
            std::vector<uint64_t> t64 = sketch_rows(S, T, &trows);
            // One index per (slot width, L, r): bucket keys are built from the b-bit ids, so b changes
            // the candidate set as well as the ranking.
            for (int b : a.bits) {
                std::vector<uint64_t> idsb_own, tb_own;
                if (b < 64) { idsb_own = truncate_all(ids64, b); tb_own = truncate_all(t64, b); }
                const std::vector<uint64_t>& idsb = b < 64 ? idsb_own : ids64;
                const std::vector<uint64_t>& tb = b < 64 ? tb_own : t64;
                for (int rr : a.rs) {
                    if (rr > a.k || a.k % rr) continue;
                    int L = a.k / rr;
                    Row r = base_row(c);
                    r.set("mode", "B"); r.set("L", L); r.set("r", rr); r.set("slot_bits", b);
                    double est = heldout_candidates(idsb, X.n, a.k, L, rr, tb, nt);
                    r.set("heldout_candidates_est", est);
                    // worst-case table bytes: ids + keys + offsets + 2x hash slots per row per band
                    double worst_gb = (double)L * X.n * (4 + 8 + 4 + 8) / 1e9;
                    double resident_gb = (ids64.size() * 8.0 + q64.size() * 8.0 + raw_bytes(X) + (size_t)X.n * a.k * 2) / 1e9;
                    std::string skip;
                    if (est > a.prune_frac * X.n) skip = "skipped_heldout_candidates_gt_" + std::to_string(a.prune_frac) + "N";
                    else if (worst_gb + resident_gb > a.mem_budget_gb) skip = "skipped_memory";
                    if (!skip.empty()) {
                        r.set("status", skip);
                        write_row(a.out, r);
                        std::printf("  B b=%d L=%d r=%d: %s (held-out candidates/query %.0f of %lld)\n", b, L, rr, skip.c_str(), est,
                                    (long long)X.n);
                        continue;
                    }
                    double tb0 = now();
                    BandIndex B = build_band_index(idsb, X.n, a.k, L, rr);
                    double build_s = now() - tb0;
                    sweep_one_bit(c, E, ids64, b, &B, r, build_s, B.bytes());
                }
            }
        } else {
            throw std::runtime_error("unknown mode " + mode);
        }
    }
    if (S.small_weight_count())
        std::printf("WARNING: %llu ficws weights fell below 1 after internal scaling\n",
                    (unsigned long long)S.small_weight_count());
    return 0;
}
