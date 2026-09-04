#ifndef FIG_MATH_SIMD_MATH_H
#define FIG_MATH_SIMD_MATH_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define FIG_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define FIG_RESTRICT __restrict
#else
#define FIG_RESTRICT
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace fig::math {

class SpatialOptimization {
 public:
  SpatialOptimization()                                      = delete;
  SpatialOptimization(const SpatialOptimization&)            = delete;
  SpatialOptimization& operator=(const SpatialOptimization&) = delete;
  ~SpatialOptimization()                                     = delete;

  [[nodiscard]] static float SquaredEuclideanDistance(
      const std::vector<float>& lhs,
      const std::vector<float>& rhs) noexcept {
    const std::size_t n = lhs.size();
    assert(n == rhs.size() &&
           "SquaredEuclideanDistance: operand dimension mismatch");
    if (n != rhs.size()) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return AccumulateSquaredDifference(lhs.data(), rhs.data(), n);
  }

  [[nodiscard]] static float EuclideanDistance(
      const std::vector<float>& lhs,
      const std::vector<float>& rhs) noexcept {
    const std::size_t n = lhs.size();
    assert(n == rhs.size() &&
           "EuclideanDistance: operand dimension mismatch");
    if (n != rhs.size()) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return std::sqrt(AccumulateSquaredDifference(lhs.data(), rhs.data(), n));
  }

 private:
  [[nodiscard]] static float AccumulateSquaredDifference(
      const float* FIG_RESTRICT a,
      const float* FIG_RESTRICT b,
      std::size_t count) noexcept {
#if defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
      const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),
                                      _mm256_loadu_ps(b + i));
      const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8),
                                      _mm256_loadu_ps(b + i + 8));
#if defined(__FMA__)
      acc0 = _mm256_fmadd_ps(d0, d0, acc0);
      acc1 = _mm256_fmadd_ps(d1, d1, acc1);
#else
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
#endif
    }

    for (; i + 8 <= count; i += 8) {
      const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i),
                                     _mm256_loadu_ps(b + i));
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d, d));
    }

    const __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);

    for (; i < count; ++i) {
      const float d = a[i] - b[i];
      sum += d * d;
    }
    return sum;
#else
    constexpr std::size_t kUnroll = 4;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    const std::size_t tail  = count % kUnroll;
    const std::size_t limit = count - tail;

    for (std::size_t i = 0; i < limit; i += kUnroll) {
      const float d0 = a[i + 0] - b[i + 0];
      const float d1 = a[i + 1] - b[i + 1];
      const float d2 = a[i + 2] - b[i + 2];
      const float d3 = a[i + 3] - b[i + 3];
      acc0 += d0 * d0;
      acc1 += d1 * d1;
      acc2 += d2 * d2;
      acc3 += d3 * d3;
    }

    float tail_acc = 0.0f;
    for (std::size_t i = limit; i < count; ++i) {
      const float d = a[i] - b[i];
      tail_acc += d * d;
    }

    const float sum = (acc0 + acc1) + (acc2 + acc3) + tail_acc;
    return static_cast<float>(sum);
#endif
  }
};

}  // namespace fig::math

#undef FIG_RESTRICT

#endif  // FIG_MATH_SIMD_MATH_H