// ==============================
// include/arena_allocator.h
// ==============================
//
// High-performance facial identity graph engine — contiguous embedding arena.
//
// ArenaAllocator is a monotonic ("bump pointer") region allocator. It carves
// aligned sub-regions out of large, pre-reserved blocks and releases all of
// them together when the arena is destroyed. There is no per-object free: an
// allocation is a single pointer advance.
//
// Why an arena belongs in this engine (see identity_node.h, "Data-oriented
// note"): the node model deliberately keeps the owning, "fat" representation
// (string + adjacency + features) for correctness, but recommends storing the
// raw feature vectors in a *separate contiguous arena* (Structure-of-Arrays)
// for the latency-critical scan path. This type is exactly that arena:
//
//   - Embeddings are inserted once and live for the index lifetime — the
//     canonical "allocate-many, free-together" lifetime an arena is built for.
//   - Bump allocation keeps consecutively inserted vectors adjacent in memory,
//     so a sequential similarity scan streams cleanly through cache instead of
//     chasing N independent heap allocations.
//   - It removes N malloc/free calls and the fragmentation they cause.
//
// Ownership: the arena exclusively owns its backing blocks via unique_ptr. It
// is move-only (copying would alias the blocks). Pointers it returns remain
// valid until the arena is destroyed or Reset() is called.
//
// Thread-safety: not internally synchronized. The owning structure is expected
// to serialize writes (the index already inserts under a single-writer model),
// so all Allocate() calls happen single-threaded; reads of returned storage are
// then safe to share across threads.
//
// Complexity: Allocate() is amortized O(1) (occasional O(block) on growth).

#ifndef FIG_CORE_ARENA_ALLOCATOR_H
#define FIG_CORE_ARENA_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace fig::core {

/// @brief Monotonic region allocator for same-lifetime, contiguous storage.
class ArenaAllocator {
 public:
  /// Default backing-block size: 1 MiB. Large enough to amortize OS allocation
  /// while small enough to avoid wasteful over-reservation for tiny indexes.
  static constexpr std::size_t kDefaultBlockBytes = std::size_t{1} << 20;

  /// @param block_bytes Size of each backing block. A single request larger
  ///        than this is still honoured by allocating an oversized block.
  /// @throws std::invalid_argument if block_bytes == 0.
  explicit ArenaAllocator(std::size_t block_bytes = kDefaultBlockBytes)
      : block_bytes_{block_bytes} {
    if (block_bytes_ == 0) {
      throw std::invalid_argument("ArenaAllocator: block_bytes must be > 0");
    }
  }

  // Move-only: copying would alias backing storage.
  ArenaAllocator(const ArenaAllocator&)            = delete;
  ArenaAllocator& operator=(const ArenaAllocator&) = delete;
  ArenaAllocator(ArenaAllocator&&) noexcept            = default;
  ArenaAllocator& operator=(ArenaAllocator&&) noexcept = default;
  ~ArenaAllocator()                                    = default;

  /// @brief Allocate `bytes` of uninitialized storage aligned to `alignment`.
  /// @return Pointer into the arena, or nullptr for a zero-byte request.
  /// @throws std::bad_alloc if a backing block cannot be obtained.
  [[nodiscard]] void* Allocate(
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t)) {
    if (bytes == 0) {
      return nullptr;
    }

    // Fast path: satisfy from the current block if the aligned request fits.
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

    // Slow path: reserve a fresh block sized for at least this request.
    AcquireBlock(block_bytes_ > bytes + alignment ? block_bytes_
                                                   : bytes + alignment);
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(current_);
    const std::uintptr_t aligned = AlignUp(base, alignment);
    offset_ = (aligned - base) + bytes;
    bytes_used_ += bytes;
    return reinterpret_cast<void*>(aligned);
  }

  /// @brief Typed convenience wrapper. Storage is uninitialized; the caller is
  /// responsible for constructing trivially-copyable payloads (e.g. floats).
  template <typename T>
  [[nodiscard]] T* AllocateArray(std::size_t count) {
    return static_cast<T*>(Allocate(count * sizeof(T), alignof(T)));
  }

  /// @brief Release all blocks and reset the cursor. Invalidates every pointer
  /// previously handed out. O(blocks) — frees the backing memory.
  void Reset() noexcept {
    blocks_.clear();
    current_ = nullptr;
    offset_ = 0;
    bytes_used_ = 0;
  }

  // ---- Introspection (snake_case accessors, per project convention) --------

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

  /// Round `n` up to the next multiple of power-of-two `a`.
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

  std::size_t block_bytes_;          // configured size of each new block
  std::vector<Block> blocks_{};      // owned backing storage
  std::byte* current_{nullptr};      // base of the active block
  std::size_t offset_{0};            // bytes consumed in the active block
  std::size_t bytes_used_{0};        // payload bytes handed out (excl. padding)
};

}  // namespace fig::core

#endif  // FIG_CORE_ARENA_ALLOCATOR_H
