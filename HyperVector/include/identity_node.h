#ifndef FIG_CORE_IDENTITY_NODE_H
#define FIG_CORE_IDENTITY_NODE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fig::core {

inline constexpr std::size_t kCacheLineBytes = 64;
class alignas(kCacheLineBytes) IdentityNode {
 public:
  using ProfileId      = std::uint64_t;
  using Feature        = float;
  using FeatureVector  = std::vector<Feature>;
  using NeighborId     = std::uint32_t;
  using AdjacencyLayer = std::vector<NeighborId>;
  using AdjacencyGraph = std::vector<AdjacencyLayer>;

  IdentityNode() noexcept = default;
  explicit IdentityNode(ProfileId profile_id) noexcept
      : profile_id_{profile_id} {}
  IdentityNode(ProfileId profile_id,
               std::string metadata,
               FeatureVector features,
               AdjacencyGraph adjacency) noexcept
      : profile_id_{profile_id},
        metadata_{std::move(metadata)},
        features_{std::move(features)},
        adjacency_{std::move(adjacency)} {}

  IdentityNode(const IdentityNode&)                = default;
  IdentityNode& operator=(const IdentityNode&)     = default;
  IdentityNode(IdentityNode&&) noexcept            = default;
  IdentityNode& operator=(IdentityNode&&) noexcept = default;
  ~IdentityNode()                                  = default;

  [[nodiscard]] ProfileId profile_id() const noexcept { return profile_id_; }
  void set_profile_id(ProfileId id) noexcept { profile_id_ = id; }

  [[nodiscard]] const std::string& metadata() const noexcept {
    return metadata_;
  }

  void set_metadata(std::string value) noexcept {
    metadata_ = std::move(value);
  }

  [[nodiscard]] const FeatureVector& features() const noexcept {
    return features_;
  }
  [[nodiscard]] FeatureVector& features() noexcept { return features_; }
  void set_features(FeatureVector value) noexcept {
    features_ = std::move(value);
  }

  [[nodiscard]] std::size_t feature_dimension() const noexcept {
    return features_.size();
  }

  [[nodiscard]] const AdjacencyGraph& adjacency() const noexcept {
    return adjacency_;
  }
  [[nodiscard]] AdjacencyGraph& adjacency() noexcept { return adjacency_; }
  void set_adjacency(AdjacencyGraph value) noexcept {
    adjacency_ = std::move(value);
  }

  [[nodiscard]] std::size_t layer_count() const noexcept {
    return adjacency_.size();
  }

  [[nodiscard]] const AdjacencyLayer& neighbors(std::size_t layer) const
      noexcept {
    return adjacency_[layer];
  }

  void swap(IdentityNode& other) noexcept {
    using std::swap;
    swap(profile_id_, other.profile_id_);
    swap(metadata_, other.metadata_);
    swap(features_, other.features_);
    swap(adjacency_, other.adjacency_);
  }

 private:
  ProfileId      profile_id_{0};
  std::string    metadata_{};
  FeatureVector  features_{};
  AdjacencyGraph adjacency_{};
};

inline void swap(IdentityNode& lhs, IdentityNode& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace fig::core

#endif  // FIG_CORE_IDENTITY_NODE_H