// ==============================
// include/embedding_store.h
// ==============================
//
// High-performance facial identity graph engine — contiguous embedding store.
//
// EmbeddingStore is the Structure-of-Arrays (SoA) layout that identity_node.h
// explicitly recommends for the latency-critical scan path: all feature vectors
// packed back-to-back in one arena-owned region, indexed by a dense row id.
//
// It complements (does not replace) the owning IdentityNode model:
//   - IdentityNode remains the authoritative, owning representation of a profile
//     (id, metadata, adjacency, features).
//   - EmbeddingStore is a cold-data-free, cache-streaming mirror of just the
//     float features, used for exact brute-force baselines and any operation
//     that benefits from pure sequential float access.
//
// All rows share a fixed dimension. Storage is bump-allocated from an
// ArenaAllocator, so N inserts cost ~N pointer advances instead of N mallocs,
// and consecutive rows are contiguous in memory.
//
// Ownership: the store owns its arena. Row pointers are stable for the store's
// lifetime (the arena never relocates handed-out storage). Move-only.

#ifndef FIG_CORE_EMBEDDING_STORE_H
#define FIG_CORE_EMBEDDING_STORE_H

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "arena_allocator.h"

namespace fig::core {

/// @brief Fixed-dimension, arena-backed SoA store of float embeddings.
class EmbeddingStore {
 public:
  /// @param dimension   Number of floats per row (must be > 0).
  /// @param block_bytes Backing arena block size.
  explicit EmbeddingStore(
      std::size_t dimension,
      std::size_t block_bytes = ArenaAllocator::kDefaultBlockBytes)
      : dimension_{dimension}, arena_{block_bytes} {
    if (dimension_ == 0) {
      throw std::invalid_argument("EmbeddingStore: dimension must be > 0");
    }
  }

  EmbeddingStore(const EmbeddingStore&)            = delete;
  EmbeddingStore& operator=(const EmbeddingStore&) = delete;
  EmbeddingStore(EmbeddingStore&&) noexcept            = default;
  EmbeddingStore& operator=(EmbeddingStore&&) noexcept = default;
  ~EmbeddingStore()                                    = default;

  /// @brief Copy one embedding into the arena.
  /// @return Dense row index of the stored vector.
  /// @throws std::invalid_argument on a dimension mismatch.
  std::size_t Add(std::span<const float> embedding) {
    if (embedding.size() != dimension_) {
      throw std::invalid_argument("EmbeddingStore::Add: dimension mismatch");
    }
    float* dst = arena_.AllocateArray<float>(dimension_);
    for (std::size_t i = 0; i < dimension_; ++i) {
      dst[i] = embedding[i];
    }
    rows_.push_back(dst);
    return rows_.size() - 1;
  }

  /// @brief Read-only view of row `index` (no bounds check on the hot path;
  /// callers iterate over [0, size())).
  [[nodiscard]] std::span<const float> Row(std::size_t index) const noexcept {
    return {rows_[index], dimension_};
  }

  [[nodiscard]] std::size_t size() const noexcept { return rows_.size(); }
  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] const ArenaAllocator& arena() const noexcept { return arena_; }

 private:
  std::size_t dimension_;
  ArenaAllocator arena_;
  std::vector<float*> rows_{};  // row index -> pointer into the arena
};

}  // namespace fig::core

#endif  // FIG_CORE_EMBEDDING_STORE_H
