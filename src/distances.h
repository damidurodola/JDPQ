#pragma once

#include <cstddef>

namespace faiss_mimic {

enum class DistanceMetric {
	Jaccard,
	ShapeJaccard,
	L2
};

// float fvec_L2sqr(const float* x, const float* y, size_t d);
// void fvec_L2sqr_ny(float* dis, const float* x, const float* y, size_t d, size_t ny);
float fvec_jaccard(const float* x, const float* y, size_t d);
void fvec_jaccard_ny(float* dis, const float* x, const float* y, size_t d, size_t ny);

// sum(x[i]) over d components. Callers precompute this once per vector (it
// does not depend on which other vector it will be paired against) and reuse
// it across every fvec_jaccard_presummed call involving that vector.
float fvec_l1sum(const float* x, size_t d);

// Equivalent to fvec_jaccard(x, y, d) given sum_x = fvec_l1sum(x, d) and
// sum_y = fvec_l1sum(y, d), but skips recomputing sum(max(xi,yi)) directly:
// union = sum_x + sum_y - intersection. Requires inputs to have already had
// components below the 1e-6 noise floor clamped to exactly 0 at ingestion
// (see read_data) -- that clamping is what makes sum_x/sum_y valid regardless
// of which y/x they end up paired with.
float fvec_jaccard_presummed(const float* x, const float* y, size_t d, float sum_x, float sum_y);

float fvec_shape_jaccard(const float* x, const float* y, size_t d);
void fvec_shape_jaccard_ny(float* dis, const float* x, const float* y, size_t d, size_t ny);

float fvec_jaccard_min(const float* x, const float* y, size_t d);
float fvec_jaccard_max(const float* x, const float* y, size_t d);
void fvec_jaccard_min_ny(float* dis, const float* x, const float* y, size_t d, size_t ny);
void fvec_jaccard_max_ny(float* dis, const float* x, const float* y, size_t d, size_t ny);

float fvec_L2sqr(const float* x, const float* y, size_t d);

void fvec_L2sqr_ny(float* dis, const float* x, const float* y, size_t d, size_t ny);

} // namespace faiss_mimic
