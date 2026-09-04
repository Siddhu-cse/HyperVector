#ifndef FIG_CORE_ARENA_ALLOCATOR_H
#define FIG_CORE_ARENA_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace fig::core {

class ArenaAllocator {
 public:
  static constexpr std::size_t kDefaultBlockBytes = std::size_t{1} << 20;

  explicit ArenaAllocator(std::size_t block_bytes = kDefaultBlockBytes)
      : block_bytes_{block_bytes} {
    if (block_bytes_ == 0) {
      throw std::invalid_argument("ArenaAllocator: block_bytes must be > 0");
    }
  }

  ArenaAllocator(const ArenaAllocator&)            = delete;
  ArenaAllocator& operator=(const ArenaAllocator&) = delete;
  ArenaAllocator(ArenaAllocator&&) noexcept            = default;
  ArenaAllocator& operator=(ArenaAllocator&&) noexcept = default;
  ~ArenaAllocator()                                    = default;

  [[nodiscard]] void* Allocate(
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) {
    if (bytes == 0) {
      return nullptr;
    }

    if (current_ != nullptr) {
      const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(current_);
      const std::uintptr_t aligned = AlignUp(base + offset_, alignment);
      const std::size_t new_offset = (aligned - base) + bytes;
      if (new_offset <= blocks_.back().size) {
        offset_ = new_offset;
        bytes_used_ += bytes;
        return reinterpret_cast<void*>(aligned);
      }
    }

    AcquireBlock(block_bytes_ > bytes + alignment ? block_bytes_
                                                   : bytes + alignment);
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(current_);
    const std::uintptr_t aligned = AlignUp(base, alignment);
    offset_ = (aligned - base) + bytes;
    bytes_used_ += bytes;
    return reinterpret_cast<void*>(aligned);
  }

  template <typename T>
  [[nodiscard]] T* AllocateArray(std::size_t count) {
    return static_cast<T*>(Allocate(count * sizeof(T), alignof(T)));
  }

  void Reset() noexcept {
    blocks_.clear();
    current_ = nullptr;
    offset_ = 0;
    bytes_used_ = 0;
  }

  [[nodiscard]] std::size_t bytes_used() const noexcept { return bytes_used_; }
  [[nodiscard]] std::size_t block_count() const noexcept {
    return blocks_.size();
  }
  [[nodiscard]] std::size_t block_bytes() const noexcept {
    return block_bytes_;
  }

 private:
  struct Block {
    std::unique_ptr<std::byte[]> data;
    std::size_t size;
  };

  [[nodiscard]] static constexpr std::uintptr_t AlignUp(
      std::uintptr_t n, std::size_t a) noexcept {
    const std::uintptr_t mask = static_cast<std::uintptr_t>(a) - 1;
    return (n + mask) & ~mask;
  }

  void AcquireBlock(std::size_t size) {
    Block block{std::make_unique<std::byte[]>(size), size};
    current_ = block.data.get();
    offset_ = 0;
    blocks_.push_back(std::move(block));
  }

  std::size_t block_bytes_;
  std::vector<Block> blocks_{};
  std::byte* current_{nullptr};
  std::size_t offset_{0};
  std::size_t bytes_used_{0};
};

}  // namespace fig::core

#endif  // FIG_CORE_ARENA_ALLOCATOR_H
