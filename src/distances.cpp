#include "distances.h"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__clang__) || defined(__GNUC__)
#define FAISS_MIMIC_RESTRICT __restrict__
#else
#define FAISS_MIMIC_RESTRICT
#endif

// Explicit vector kernels for the pairwise reductions below. These are the
// hot path for k-medoids training and PQ code assignment, which together
// dominate benchmark runtime, so they are worth hand-vectorizing rather than
// leaving to the auto-vectorizer. Guarded on the ISA macros so non-x86 (Apple
// arm64) and conservatively-targeted builds fall back to the `omp simd` loops.
#if defined(__AVX512F__)
#include <immintrin.h>
#define FAISS_MIMIC_SIMD_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define FAISS_MIMIC_SIMD_AVX2 1
#endif

#if defined(FAISS_MIMIC_SIMD_AVX2)
namespace {
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
} // namespace
#endif

namespace faiss_mimic {

namespace {

inline bool should_parallelize_ny(size_t ny, size_t d, size_t tile) {
#ifdef _OPENMP
    const size_t total_work = ny * d;
    return !omp_in_parallel() && ny >= tile * 2 && total_work >= 4096;
#else
    (void)ny;
    (void)d;
    (void)tile;
    return false;
#endif
}

} // namespace

// Compute and store the sum of x, sum of y once,
// then sum of min(x,y)
// fvec_jaccard() is the same as sum of min(x,y) / sum of max(x,y) = sum of min(x,y) / (sum of x + sum of y - sum of min(x,y))



float fvec_jaccard(const float* FAISS_MIMIC_RESTRICT x, const float* FAISS_MIMIC_RESTRICT y, size_t d) {
    float intersection = 0.0f;
    float union_val = 0.0f;
#if defined(FAISS_MIMIC_SIMD_AVX512)
    __m512 acc_i = _mm512_setzero_ps();
    __m512 acc_u = _mm512_setzero_ps();
    const __m512 thr = _mm512_set1_ps(1e-6f);
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m512 xv = _mm512_loadu_ps(x + i);
        const __m512 yv = _mm512_loadu_ps(y + i);
        // Same "both effectively zero contribute nothing" rule as the scalar
        // path, expressed as a mask so the loop stays branch-free.
        const __mmask16 k = _mm512_cmp_ps_mask(_mm512_add_ps(xv, yv), thr, _CMP_GE_OQ);
        acc_i = _mm512_mask_add_ps(acc_i, k, acc_i, _mm512_min_ps(xv, yv));
        acc_u = _mm512_mask_add_ps(acc_u, k, acc_u, _mm512_max_ps(xv, yv));
    }
    if (i < d) {
        const __mmask16 tail = (__mmask16)((1u << (d - i)) - 1);
        const __m512 xv = _mm512_maskz_loadu_ps(tail, x + i);
        const __m512 yv = _mm512_maskz_loadu_ps(tail, y + i);
        const __mmask16 k =
            _mm512_cmp_ps_mask(_mm512_add_ps(xv, yv), thr, _CMP_GE_OQ) & tail;
        acc_i = _mm512_mask_add_ps(acc_i, k, acc_i, _mm512_min_ps(xv, yv));
        acc_u = _mm512_mask_add_ps(acc_u, k, acc_u, _mm512_max_ps(xv, yv));
    }
    intersection = _mm512_reduce_add_ps(acc_i);
    union_val = _mm512_reduce_add_ps(acc_u);
#elif defined(FAISS_MIMIC_SIMD_AVX2)
    __m256 acc_i = _mm256_setzero_ps();
    __m256 acc_u = _mm256_setzero_ps();
    const __m256 thr = _mm256_set1_ps(1e-6f);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 yv = _mm256_loadu_ps(y + i);
        const __m256 k = _mm256_cmp_ps(_mm256_add_ps(xv, yv), thr, _CMP_GE_OQ);
        acc_i = _mm256_add_ps(acc_i, _mm256_and_ps(k, _mm256_min_ps(xv, yv)));
        acc_u = _mm256_add_ps(acc_u, _mm256_and_ps(k, _mm256_max_ps(xv, yv)));
    }
    intersection = hsum256(acc_i);
    union_val = hsum256(acc_u);
    for (; i < d; ++i) {
        const float xi = x[i], yi = y[i];
        const float mask = (xi + yi >= 1e-6f) ? 1.0f : 0.0f;
        intersection += mask * std::min(xi, yi);
        union_val += mask * std::max(xi, yi);
    }
#else
    // Branch-free form of "skip dims where both are effectively zero": the
    // `continue` version cannot be vectorized and runs ~6x slower.
#pragma omp simd reduction(+:intersection, union_val)
    for (size_t i = 0; i < d; i++) {
        const float xi = x[i];
        const float yi = y[i];
        const float mask = (xi + yi >= 1e-6f) ? 1.0f : 0.0f;
        intersection += mask * std::min(xi, yi);
        union_val += mask * std::max(xi, yi);
    }
#endif
    if (union_val <= 0.0f) {
        return 0.0f;
    }
    return 1.0f - (intersection / union_val);
}

float fvec_shape_jaccard(const float* FAISS_MIMIC_RESTRICT x, const float* FAISS_MIMIC_RESTRICT y, size_t d) {
    float intersection = 0.0f;
#pragma omp simd reduction(+:intersection)
    for (size_t i = 0; i < d; i++) {
        intersection += std::min(x[i], y[i]);
    }
    const float denom = 2.0f - intersection;
    return (denom <= 0.0f) ? 0.0f : (1.0f - (intersection / denom));
}

void fvec_jaccard_ny(float* FAISS_MIMIC_RESTRICT dis, const float* FAISS_MIMIC_RESTRICT x,
                     const float* FAISS_MIMIC_RESTRICT y, size_t d, size_t ny) {
    constexpr size_t kTile = 8; // tweak if wider tiles show better cache reuse
    const bool use_parallel = should_parallelize_ny(ny, d, kTile);
#pragma omp parallel for if(use_parallel) schedule(static)
    for (size_t i = 0; i < ny; i += kTile) {
        size_t current = std::min(kTile, ny - i);
        float intersection[kTile] = {0};
        float union_val[kTile] = {0};
        const float* y_ptr = y + i * d;
        for (size_t dim = 0; dim < d; ++dim) {
            float xi = x[dim];
            for (size_t j = 0; j < current; ++j) {
                float yi = y_ptr[j * d + dim];
                const float sum = xi + yi;
                if (sum < 1e-6f) {
                    continue;
                }
                intersection[j] += std::min(xi, yi);
                union_val[j] += std::max(xi, yi);
            }
        }
        for (size_t j = 0; j < current; ++j) {
            float uni = union_val[j];
            dis[i + j] = (uni <= 0.0f) ? 0.0f : (1.0f - (intersection[j] / uni));
        }
    }
}

void fvec_shape_jaccard_ny(float* FAISS_MIMIC_RESTRICT dis, const float* FAISS_MIMIC_RESTRICT x,
                           const float* FAISS_MIMIC_RESTRICT y, size_t d, size_t ny) {
    constexpr size_t kTile = 8;
    const bool use_parallel = should_parallelize_ny(ny, d, kTile);
#pragma omp parallel for if(use_parallel) schedule(static)
    for (size_t i = 0; i < ny; i += kTile) {
        size_t current = std::min(kTile, ny - i);
        float intersection[kTile] = {0};
        const float* y_ptr = y + i * d;
        for (size_t dim = 0; dim < d; ++dim) {
            float xi = x[dim];
            for (size_t j = 0; j < current; ++j) {
                float yi = y_ptr[j * d + dim];
                intersection[j] += std::min(xi, yi);
            }
        }
        for (size_t j = 0; j < current; ++j) {
            const float denom = 2.0f - intersection[j];
            dis[i + j] = (denom <= 0.0f) ? 0.0f : (1.0f - (intersection[j] / denom));
        }
    }
}

float fvec_jaccard_min(const float* FAISS_MIMIC_RESTRICT x, const float* FAISS_MIMIC_RESTRICT y, size_t d) {
    // Hand-vectorized like fvec_jaccard above: left to "#pragma omp simd"
    // alone, gcc -O3 -march=native only emitted 256-bit AVX2 with a single
    // accumulator (verified via generated assembly), even on an AVX-512
    // host -- leaving both vector width and instruction-level parallelism on
    // the table. This is now the hottest kernel in the codebase (every
    // fvec_jaccard_presummed call bottoms out here), so both are worth
    // claiming explicitly. Multiple independent accumulators break the
    // serial add-latency dependency chain a single accumulator creates.
#if defined(FAISS_MIMIC_SIMD_AVX512)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 64 <= d; i += 64) {
        acc0 = _mm512_add_ps(acc0, _mm512_min_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
        acc1 = _mm512_add_ps(acc1, _mm512_min_ps(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(y + i + 16)));
        acc2 = _mm512_add_ps(acc2, _mm512_min_ps(_mm512_loadu_ps(x + i + 32), _mm512_loadu_ps(y + i + 32)));
        acc3 = _mm512_add_ps(acc3, _mm512_min_ps(_mm512_loadu_ps(x + i + 48), _mm512_loadu_ps(y + i + 48)));
    }
    for (; i + 16 <= d; i += 16) {
        acc0 = _mm512_add_ps(acc0, _mm512_min_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
    if (i < d) {
        const __mmask16 tail = (__mmask16)((1u << (d - i)) - 1);
        const __m512 xv = _mm512_maskz_loadu_ps(tail, x + i);
        const __m512 yv = _mm512_maskz_loadu_ps(tail, y + i);
        acc0 = _mm512_mask_add_ps(acc0, tail, acc0, _mm512_min_ps(xv, yv));
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    return _mm512_reduce_add_ps(acc0);
#elif defined(FAISS_MIMIC_SIMD_AVX2)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        acc0 = _mm256_add_ps(acc0, _mm256_min_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
        acc1 = _mm256_add_ps(acc1, _mm256_min_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8)));
    }
    for (; i + 8 <= d; i += 8) {
        acc0 = _mm256_add_ps(acc0, _mm256_min_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
    float intersection = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < d; ++i) {
        intersection += std::min(x[i], y[i]);
    }
    return intersection;
#else
    float intersection = 0.0f;
    float intersection2 = 0.0f;
#pragma omp simd reduction(+:intersection, intersection2)
    for (size_t i = 0; i + 1 < d; i += 2) {
        intersection += std::min(x[i], y[i]);
        intersection2 += std::min(x[i + 1], y[i + 1]);
    }
    if (d & 1) {
        intersection += std::min(x[d - 1], y[d - 1]);
    }
    return intersection + intersection2;
#endif
}

float fvec_l1sum(const float* FAISS_MIMIC_RESTRICT x, size_t d) {
    float sum = 0.0f;
#pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < d; i++) {
        sum += x[i];
    }
    return sum;
}

float fvec_jaccard_presummed(const float* FAISS_MIMIC_RESTRICT x, const float* FAISS_MIMIC_RESTRICT y,
                             size_t d, float sum_x, float sum_y) {
    const float intersection = fvec_jaccard_min(x, y, d);
    const float union_val = sum_x + sum_y - intersection;
    if (union_val <= 0.0f) {
        return 0.0f;
    }
    return 1.0f - (intersection / union_val);
}


void fvec_jaccard_min_ny(float* FAISS_MIMIC_RESTRICT dis, const float* FAISS_MIMIC_RESTRICT x,
                         const float* FAISS_MIMIC_RESTRICT y, size_t d, size_t ny) {
    constexpr size_t kTile = 8;
    const bool use_parallel = should_parallelize_ny(ny, d, kTile);
#pragma omp parallel for if(use_parallel) schedule(static)
    for (size_t i = 0; i < ny; i += kTile) {
        size_t current = std::min(kTile, ny - i);
        float accum[kTile] = {0};
        const float* y_ptr = y + i * d;
        for (size_t dim = 0; dim < d; ++dim) {
            float xi = x[dim];
            for (size_t j = 0; j < current; ++j) {
                float yi = y_ptr[j * d + dim];
                accum[j] += std::min(xi, yi);
            }
        }
        for (size_t j = 0; j < current; ++j) {
            dis[i + j] = accum[j];
        }
    }
}

float fvec_jaccard_max(const float* FAISS_MIMIC_RESTRICT x, const float* FAISS_MIMIC_RESTRICT y, size_t d) {
    float union_jd = 0.0f;
#pragma omp simd reduction(+:union_jd)
    for (size_t i = 0; i < d; i++) {
        union_jd += std::max(x[i], y[i]);
    }
    return union_jd;
}

void fvec_jaccard_max_ny(float* FAISS_MIMIC_RESTRICT dis, const float* FAISS_MIMIC_RESTRICT x,
                         const float* FAISS_MIMIC_RESTRICT y, size_t d, size_t ny) {
    constexpr size_t kTile = 8;
    const bool use_parallel = should_parallelize_ny(ny, d, kTile);
#pragma omp parallel for if(use_parallel) schedule(static)
    for (size_t i = 0; i < ny; i += kTile) {
        size_t current = std::min(kTile, ny - i);
        float accum[kTile] = {0};
        const float* y_ptr = y + i * d;
        for (size_t dim = 0; dim < d; ++dim) {
            float xi = x[dim];
            for (size_t j = 0; j < current; ++j) {
                float yi = y_ptr[j * d + dim];
                accum[j] += std::max(xi, yi);
            }
        }
        for (size_t j = 0; j < current; ++j) {
            dis[i + j] = accum[j];
        }
    }
}

float fvec_L2sqr(const float* FAISS_MIMIC_RESTRICT x, const float* FAISS_MIMIC_RESTRICT y, size_t d) {
    float res = 0;
#pragma omp simd reduction(+:res)
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

void fvec_L2sqr_ny(float* FAISS_MIMIC_RESTRICT dis, const float* FAISS_MIMIC_RESTRICT x,
                   const float* FAISS_MIMIC_RESTRICT y, size_t d, size_t ny) {
    const bool use_parallel = should_parallelize_ny(ny, d, 1);
#pragma omp parallel for if(use_parallel) schedule(static)
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr(x, y + i * d, d);
    }
}

} // namespace faiss_mimic

#undef FAISS_MIMIC_RESTRICT
