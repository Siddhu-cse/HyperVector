// ==============================
// include/simd_math.h
// ==============================
//
// High-performance facial identity graph engine — spatial acceleration kernels.
//
// SpatialOptimization provides dependency-free distance kernels for
// latency-sensitive similarity search. The implementation is fully manual:
// no Eigen, no OpenCV, no Boost, no external math frameworks.
//
// Optimization strategy:
//   - Manual unroll factor of 4 over FOUR independent accumulators. A single
//     accumulator would serialize each multiply-add behind the previous one
//     (a long latency-bound dependency chain). Four parallel chains expose
//     instruction-level parallelism so the CPU's out-of-order engine and the
//     auto-vectorizer can keep the FP units saturated; the partial sums are
//     folded once at the end (tree reduction) to minimize the critical path.
//   - The remainder (size % 4) is handled by a short scalar epilogue.
//   - FIG_RESTRICT promises non-aliasing operands, unlocking vectorization
//     without forcing a non-standard keyword onto unknown compilers.
//
// Complexity: O(N) time, O(1) additional space, where N == vector dimension.

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

// Pull in AVX2 / FMA intrinsics only when the target ISA actually provides
// them. Builds for older CPUs transparently fall back to the portable,
// auto-vectorizable scalar kernel below — no source change required.
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace fig::math {

/// @brief Stateless collection of spatial distance kernels.
/// Non-instantiable: all functionality is exposed through static methods.
class SpatialOptimization {
 public:
  SpatialOptimization()                                      = delete;
  SpatialOptimization(const SpatialOptimization&)            = delete;
  SpatialOptimization& operator=(const SpatialOptimization&) = delete;
  ~SpatialOptimization()                                     = delete;

  /// @brief Squared Euclidean distance between two equal-length vectors.
  ///
  /// Preferred for ranking / nearest-neighbour comparisons because it avoids
  /// the std::sqrt latency while preserving ordering (monotonic in distance).
  ///
  /// @returns sum((lhs[i] - rhs[i])^2), or quiet NaN on a size mismatch.
  /// @complexity O(N) time, O(1) space. noexcept — never throws.
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

  /// @brief Euclidean (L2) distance between two equal-length vectors.
  ///
  /// @returns sqrt(sum((lhs[i] - rhs[i])^2)), or quiet NaN on a size mismatch.
  /// @complexity O(N) time, O(1) space. noexcept — never throws.
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
  /// @brief Core accumulation kernel: sum of squared differences.
  ///
  /// Unrolled x4 across four independent accumulators to break the
  /// floating-point dependency chain and maximize ILP. Operands are assumed
  /// non-overlapping (FIG_RESTRICT). `count` is the element count of both
  /// arrays; callers guarantee both pointers reference at least `count`
  /// valid floats.
  [[nodiscard]] static float AccumulateSquaredDifference(
      const float* FIG_RESTRICT a,
      const float* FIG_RESTRICT b,
      std::size_t count) noexcept {
#if defined(__AVX2__)
    // --- AVX2 hardware-intrinsic path -------------------------------------
    // Eight single-precision lanes per 256-bit YMM register. We keep two
    // independent accumulators so two FMA/mul-add chains run in parallel,
    // hiding instruction latency exactly as the scalar x4 unroll does — but
    // at 16 floats per loop iteration instead of 4.
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
    // 8-wide remainder.
    for (; i + 8 <= count; i += 8) {
      const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i),
                                     _mm256_loadu_ps(b + i));
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d, d));
    }
    // Fold the two YMM accumulators, then horizontally reduce 8 -> 1.
    const __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    // Scalar epilogue for the (0..7) trailing elements.
    for (; i < count; ++i) {
      const float d = a[i] - b[i];
      sum += d * d;
    }
    return sum;
#else
    // --- Portable scalar fallback (manual x4 unroll, ILP-friendly) --------
    constexpr std::size_t kUnroll = 4;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    const std::size_t tail  = count % kUnroll;
    const std::size_t limit = count - tail;

    // Main vectorizable body: four parallel reduction lanes, no branches.
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

    // Scalar epilogue for the (0..3) leftover elements.
    float tail_acc = 0.0f;
    for (std::size_t i = limit; i < count; ++i) {
      const float d = a[i] - b[i];
      tail_acc += d * d;
    }

    // Tree reduction of the partial sums to keep the critical path short.
    const float sum = (acc0 + acc1) + (acc2 + acc3) + tail_acc;
    return static_cast<float>(sum);
#endif  // __AVX2__
  }
};

}  // namespace fig::math

#undef FIG_RESTRICT

#endif  // FIG_MATH_SIMD_MATH_H