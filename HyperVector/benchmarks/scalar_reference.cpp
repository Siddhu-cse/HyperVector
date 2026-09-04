// ==============================
// benchmarks/scalar_reference.cpp
// ==============================

#include "scalar_reference.h"

namespace fig::bench {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
float
ScalarSquaredDistance(const float* a, const float* b, std::size_t n) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

}  // namespace fig::bench
