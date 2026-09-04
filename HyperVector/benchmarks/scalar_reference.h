// ==============================
// benchmarks/scalar_reference.h
// ==============================
//
// A deliberately un-vectorized squared-Euclidean kernel, used ONLY as the
// baseline in the SIMD microbenchmark. Compiling the comparison fairly is
// tricky: if the baseline lived in the same translation unit as the AVX2 build
// flags, the auto-vectorizer would turn it into SIMD too and the "speedup"
// would be meaningless. This function is defined in its own .cpp and pinned to
// scalar codegen, so the benchmark measures a true scalar-vs-AVX2 delta.

#ifndef FIG_BENCH_SCALAR_REFERENCE_H
#define FIG_BENCH_SCALAR_REFERENCE_H

#include <cstddef>

namespace fig::bench {

/// Plain scalar sum of squared differences (no SIMD, no unrolling).
[[nodiscard]] float ScalarSquaredDistance(const float* a, const float* b,
                                          std::size_t n) noexcept;

}  // namespace fig::bench

#endif  // FIG_BENCH_SCALAR_REFERENCE_H
