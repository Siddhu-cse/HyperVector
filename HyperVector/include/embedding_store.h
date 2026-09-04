#ifndef FIG_CORE_EMBEDDING_STORE_H
#define FIG_CORE_EMBEDDING_STORE_H

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "arena_allocator.h"

namespace fig::core {

class EmbeddingStore {
 public:
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

  [[nodiscard]] std::span<const float> Row(std::size_t index) const noexcept {
    return {rows_[index], dimension_};
  }

  [[nodiscard]] std::size_t size() const noexcept { return rows_.size(); }
  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] const ArenaAllocator& arena() const noexcept { return arena_; }

 private:
  std::size_t dimension_;
  ArenaAllocator arena_;
  std::vector<float*> rows_{};
};

}  // namespace fig::core

#endif  // FIG_CORE_EMBEDDING_STORE_H
