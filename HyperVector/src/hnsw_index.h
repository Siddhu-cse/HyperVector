#ifndef FIG_CORE_HNSW_INDEX_H
#define FIG_CORE_HNSW_INDEX_H

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "identity_node.h"

namespace fig::core {

class HNSWIndex {
 public:
  explicit HNSWIndex(std::size_t M = kDefaultM,
                     std::uint64_t seed = kDefaultSeed);

  HNSWIndex(const HNSWIndex&)                = default;
  HNSWIndex& operator=(const HNSWIndex&)     = default;
  HNSWIndex(HNSWIndex&&) noexcept            = default;
  HNSWIndex& operator=(HNSWIndex&&) noexcept = default;
  ~HNSWIndex()                               = default;

  void InsertProfile(std::uint64_t id,
                     const std::string& metadata,
                     const std::vector<float>& features);

  [[nodiscard]] std::vector<std::uint64_t> SearchKNN(
      const std::vector<float>& query_vector,
      std::size_t k) const noexcept;

  [[nodiscard]] std::size_t Size() const noexcept { return nodes_.size(); }
  [[nodiscard]] bool Empty() const noexcept { return nodes_.empty(); }
  [[nodiscard]] int MaxLevel() const noexcept { return max_level_; }
  [[nodiscard]] std::uint64_t EntryNode() const noexcept {
    return enter_node_id_;
  }

 private:
  struct Candidate {
    float distance;
    std::uint64_t id;
  };

  static constexpr std::size_t kDefaultM        = 16;
  static constexpr std::uint64_t kDefaultSeed   = 0x9E3779B97F4A7C15ULL;
  static constexpr std::size_t kEfConstruction  = 200;
  static constexpr std::size_t kEfSearch        = 64;

  [[nodiscard]] int RandomLevel() noexcept;

  [[nodiscard]] float NodeDistance(const std::vector<float>& query,
                                   std::uint64_t id) const noexcept;

  [[nodiscard]] std::uint64_t GreedyClosest(const std::vector<float>& query,
                                            std::uint64_t entry,
                                            int layer) const noexcept;

  [[nodiscard]] std::vector<Candidate> SearchLayer(
      const std::vector<float>& query,
      const std::vector<std::uint64_t>& entry_points,
      std::size_t ef,
      int layer) const;

  [[nodiscard]] std::vector<std::uint64_t> SelectNeighbors(
      std::vector<Candidate> candidates,
      std::size_t max_neighbors) const;

  void ConnectBidirectional(std::uint64_t id,
                            const std::vector<std::uint64_t>& neighbors,
                            int layer);

  void PruneNeighbors(std::uint64_t id, int layer);

  std::unordered_map<std::uint64_t, IdentityNode> nodes_;
  std::size_t M_;
  int max_level_;
  std::uint64_t enter_node_id_;
  std::mt19937 rng_;
};

}  // namespace fig::core

#endif  // FIG_CORE_HNSW_INDEX_H