// ==============================
// src/hnsw_index.h
// ==============================

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

/// @brief Hierarchical Navigable Small World (HNSW) approximate
/// nearest-neighbour index over IdentityNode feature vectors.
///
/// Ownership: the index exclusively owns every node (and its multilayer
/// adjacency) stored in `nodes_`. Distances are squared-Euclidean throughout;
/// since squaring is monotonic, ordering and k-NN results are identical to
/// true Euclidean distance while avoiding the per-comparison sqrt.
class HNSWIndex {
 public:
  /// @param M     Target maximum out-degree per node per layer.
  /// @param seed  Deterministic seed for the layer-assignment PRNG.
  explicit HNSWIndex(std::size_t M = kDefaultM,
                     std::uint64_t seed = kDefaultSeed);

  HNSWIndex(const HNSWIndex&)                = default;
  HNSWIndex& operator=(const HNSWIndex&)     = default;
  HNSWIndex(HNSWIndex&&) noexcept            = default;
  HNSWIndex& operator=(HNSWIndex&&) noexcept = default;
  ~HNSWIndex()                               = default;

  /// @brief Insert a profile and wire it into the navigable graph.
  /// Idempotent on duplicate ids; ignores empty feature vectors. Provides the
  /// basic exception-safety guarantee: a throw during wiring leaves the index
  /// in a valid, queryable state (the node may simply be under-connected).
  void InsertProfile(std::uint64_t id,
                     const std::string& metadata,
                     const std::vector<float>& features);

  /// @brief Return up to `k` node ids ordered by increasing distance to the
  /// query. Never throws: on any internal failure an empty result is returned.
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
  /// Scored (distance, id) pair used by the priority-queue traversals.
  struct Candidate {
    float distance;
    std::uint64_t id;
  };

  static constexpr std::size_t kDefaultM        = 16;
  static constexpr std::uint64_t kDefaultSeed   = 0x9E3779B97F4A7C15ULL;
  static constexpr std::size_t kEfConstruction  = 200;
  static constexpr std::size_t kEfSearch        = 64;

  /// Draw a node level from the exponential-decay distribution
  /// floor(-ln(U) / ln(M)), so layer occupancy shrinks geometrically.
  [[nodiscard]] int RandomLevel() noexcept;

  /// Squared-Euclidean distance from `query` to node `id` (infinity if absent).
  [[nodiscard]] float NodeDistance(const std::vector<float>& query,
                                   std::uint64_t id) const noexcept;

  /// Greedy single-best descent within one layer; returns the closest reachable
  /// node to `query` starting from `entry`.
  [[nodiscard]] std::uint64_t GreedyClosest(const std::vector<float>& query,
                                            std::uint64_t entry,
                                            int layer) const noexcept;

  /// Best-first beam search within `layer`, returning up to `ef` nearest nodes.
  [[nodiscard]] std::vector<Candidate> SearchLayer(
      const std::vector<float>& query,
      const std::vector<std::uint64_t>& entry_points,
      std::size_t ef,
      int layer) const;

  /// Select the `max_neighbors` closest candidates (ascending distance).
  [[nodiscard]] std::vector<std::uint64_t> SelectNeighbors(
      std::vector<Candidate> candidates,
      std::size_t max_neighbors) const;

  /// Create bidirectional edges between `id` and each neighbour at `layer`,
  /// then enforce the degree bound on every touched node.
  void ConnectBidirectional(std::uint64_t id,
                            const std::vector<std::uint64_t>& neighbors,
                            int layer);

  /// Retain only the `M_` closest edges of node `id` at `layer`.
  void PruneNeighbors(std::uint64_t id, int layer);

  std::unordered_map<std::uint64_t, IdentityNode> nodes_;
  std::size_t M_;
  int max_level_;
  std::uint64_t enter_node_id_;
  std::mt19937 rng_;
};

}  // namespace fig::core

#endif  // FIG_CORE_HNSW_INDEX_H