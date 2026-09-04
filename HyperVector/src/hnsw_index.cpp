#include "hnsw_index.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>

#include "simd_math.h"

namespace fig::core {
namespace {

[[nodiscard]] bool ContainsId(const std::vector<std::uint32_t>& list,
                              std::uint32_t value) noexcept {
  for (const std::uint32_t entry : list) {
    if (entry == value) {
      return true;
    }
  }
  return false;
}

}

HNSWIndex::HNSWIndex(std::size_t M, std::uint64_t seed)
    : nodes_{},
      M_{M == 0 ? kDefaultM : M},
      max_level_{-1},
      enter_node_id_{0},
      rng_{static_cast<std::mt19937::result_type>(seed)} {}

int HNSWIndex::RandomLevel() noexcept {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r = dist(rng_);
  if (r < std::numeric_limits<double>::min()) {
    r = std::numeric_limits<double>::min();
  }
  const double denom = std::log(static_cast<double>(M_ < 2 ? 2 : M_));
  const double level_multiplier = 1.0 / denom;
  const int level = static_cast<int>(-std::log(r) * level_multiplier);
  return level < 0 ? 0 : level;
}

float HNSWIndex::NodeDistance(const std::vector<float>& query,
                                 std::uint64_t id) const noexcept {
  const auto it = nodes_.find(id);
  if (it == nodes_.end()) {
    return std::numeric_limits<float>::infinity();
  }
  return fig::math::SpatialOptimization::SquaredEuclideanDistance(
      query, it->second.features());
}

std::uint64_t HNSWIndex::GreedyClosest(const std::vector<float>& query,
                                       std::uint64_t entry,
                                       int layer) const noexcept {
  std::uint64_t current = entry;
  float current_distance = NodeDistance(query, current);

  bool improved = true;
  while (improved) {
    improved = false;
    const auto it = nodes_.find(current);
    if (it == nodes_.end()) {
      break;
    }
    const IdentityNode& node = it->second;
    if (layer < 0 || static_cast<std::size_t>(layer) >= node.layer_count()) {
      break;
    }
    for (const std::uint32_t neighbor : node.neighbors(
             static_cast<std::size_t>(layer))) {
      const std::uint64_t neighbor_id = static_cast<std::uint64_t>(neighbor);
      const float distance = NodeDistance(query, neighbor_id);
      if (distance < current_distance) {
        current_distance = distance;
        current = neighbor_id;
        improved = true;
      }
    }
  }
  return current;
}

std::vector<HNSWIndex::Candidate> HNSWIndex::SearchLayer(
    const std::vector<float>& query,
    const std::vector<std::uint64_t>& entry_points,
    std::size_t ef,
    int layer) const {
  const auto closer_first = [](const Candidate& a, const Candidate& b) noexcept {
    return a.distance > b.distance;
  };
  const auto farther_first =
      [](const Candidate& a, const Candidate& b) noexcept {
        return a.distance < b.distance;
      };

  std::priority_queue<Candidate, std::vector<Candidate>, decltype(closer_first)>
      frontier(closer_first);
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(farther_first)>
      results(farther_first);

  std::unordered_set<std::uint64_t> visited;
  visited.reserve(ef * 4);

  for (const std::uint64_t ep : entry_points) {
    if (nodes_.find(ep) == nodes_.end()) {
      continue;
    }
    if (!visited.insert(ep).second) {
      continue;
    }
    const Candidate seed{NodeDistance(query, ep), ep};
    frontier.push(seed);
    results.push(seed);
  }
  while (results.size() > ef) {
    results.pop();
  }

  while (!frontier.empty()) {
    const Candidate nearest = frontier.top();
    frontier.pop();

    if (!results.empty() && nearest.distance > results.top().distance) {
      break;
    }

    const auto it = nodes_.find(nearest.id);
    if (it == nodes_.end()) {
      continue;
    }
    const IdentityNode& node = it->second;
    if (layer < 0 || static_cast<std::size_t>(layer) >= node.layer_count()) {
      continue;
    }

    for (const std::uint32_t neighbor : node.neighbors(
             static_cast<std::size_t>(layer))) {
      const std::uint64_t neighbor_id = static_cast<std::uint64_t>(neighbor);
      if (!visited.insert(neighbor_id).second) {
        continue;
      }
      const float distance = NodeDistance(query, neighbor_id);
      if (results.size() < ef || distance < results.top().distance) {
        const Candidate candidate{distance, neighbor_id};
        frontier.push(candidate);
        results.push(candidate);
        if (results.size() > ef) {
          results.pop();
        }
      }
    }
  }

  std::vector<Candidate> out;
  out.reserve(results.size());
  while (!results.empty()) {
    out.push_back(results.top());
    results.pop();
  }
  return out;
}

std::vector<std::uint64_t> HNSWIndex::SelectNeighbors(
    std::vector<Candidate> candidates,
    std::size_t max_neighbors) const {
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) noexcept {
              return a.distance < b.distance;
            });

  std::vector<std::uint64_t> selected;
  selected.reserve(std::min(max_neighbors, candidates.size()));
  for (const Candidate& candidate : candidates) {
    if (selected.size() >= max_neighbors) {
      break;
    }
    selected.push_back(candidate.id);
  }
  return selected;
}

void HNSWIndex::ConnectBidirectional(
    std::uint64_t id,
    const std::vector<std::uint64_t>& neighbors,
    int layer) {
  const auto self_it = nodes_.find(id);
  if (self_it == nodes_.end()) {
    return;
  }
  IdentityNode& self = self_it->second;
  if (layer < 0 ||
      static_cast<std::size_t>(layer) >= self.adjacency().size()) {
    return;
  }
  std::vector<std::uint32_t>& self_layer =
      self.adjacency()[static_cast<std::size_t>(layer)];

  const std::uint32_t self_key = static_cast<std::uint32_t>(id);

  for (const std::uint64_t neighbor_id : neighbors) {
    if (neighbor_id == id) {
      continue;
    }
    const auto other_it = nodes_.find(neighbor_id);
    if (other_it == nodes_.end()) {
      continue;
    }
    IdentityNode& other = other_it->second;
    if (static_cast<std::size_t>(layer) >= other.adjacency().size()) {
      continue;
    }

    const std::uint32_t neighbor_key =
        static_cast<std::uint32_t>(neighbor_id);
    if (!ContainsId(self_layer, neighbor_key)) {
      self_layer.push_back(neighbor_key);
    }
    std::vector<std::uint32_t>& other_layer =
        other.adjacency()[static_cast<std::size_t>(layer)];
    if (!ContainsId(other_layer, self_key)) {
      other_layer.push_back(self_key);
    }
    PruneNeighbors(neighbor_id, layer);
  }
  PruneNeighbors(id, layer);
}

void HNSWIndex::PruneNeighbors(std::uint64_t id, int layer) {
  const auto it = nodes_.find(id);
  if (it == nodes_.end()) {
    return;
  }
  IdentityNode& node = it->second;
  if (layer < 0 ||
      static_cast<std::size_t>(layer) >= node.adjacency().size()) {
    return;
  }
  std::vector<std::uint32_t>& list =
      node.adjacency()[static_cast<std::size_t>(layer)];
  if (list.size() <= M_) {
    return;
  }

  const std::vector<float>& features = node.features();
  std::vector<Candidate> scored;
  scored.reserve(list.size());
  for (const std::uint32_t neighbor : list) {
    const auto neighbor_it =
        nodes_.find(static_cast<std::uint64_t>(neighbor));
    if (neighbor_it == nodes_.end()) {
      continue;
    }
    const float distance =
        fig::math::SpatialOptimization::SquaredEuclideanDistance(
            features, neighbor_it->second.features());
    scored.push_back({distance, static_cast<std::uint64_t>(neighbor)});
  }

  std::sort(scored.begin(), scored.end(),
            [](const Candidate& a, const Candidate& b) noexcept {
              return a.distance < b.distance;
            });

  list.clear();
  const std::size_t keep = std::min(M_, scored.size());
  list.reserve(keep);
  for (std::size_t i = 0; i < keep; ++i) {
    list.push_back(static_cast<std::uint32_t>(scored[i].id));
  }
}

void HNSWIndex::InsertProfile(std::uint64_t id,
                              const std::string& metadata,
                              const std::vector<float>& features) {
  if (features.empty()) {
    return;
  }
  if (nodes_.find(id) != nodes_.end()) {
    return;
  }

  const int node_level = RandomLevel();

  IdentityNode::AdjacencyGraph adjacency(
      static_cast<std::size_t>(node_level) + 1);
  IdentityNode node(id, metadata, features, std::move(adjacency));

  const auto [it, inserted] = nodes_.emplace(id, std::move(node));
  if (!inserted) {
    return;
  }

  if (nodes_.size() == 1) {
    enter_node_id_ = id;
    max_level_ = node_level;
    return;
  }

  const int top = max_level_;
  std::uint64_t entry = enter_node_id_;

  for (int layer = top; layer > node_level; --layer) {
    entry = GreedyClosest(features, entry, layer);
  }

  const int start_layer = std::min(node_level, top);
  std::vector<std::uint64_t> entry_points{entry};

  for (int layer = start_layer; layer >= 0; --layer) {
    std::vector<Candidate> found =
        SearchLayer(features, entry_points, kEfConstruction, layer);
    const std::vector<std::uint64_t> selected = SelectNeighbors(found, M_);
    ConnectBidirectional(id, selected, layer);

    entry_points.clear();
    if (!found.empty()) {
      entry_points.reserve(found.size());
      for (const Candidate& candidate : found) {
        entry_points.push_back(candidate.id);
      }
    } else {
      entry_points.push_back(entry);
    }
  }

  if (node_level > max_level_) {
    max_level_ = node_level;
    enter_node_id_ = id;
  }
}

std::vector<std::uint64_t> HNSWIndex::SearchKNN(
    const std::vector<float>& query_vector,
    std::size_t k) const noexcept {
  std::vector<std::uint64_t> result;
  if (k == 0 || query_vector.empty() || nodes_.empty()) {
    return result;
  }

  try {
    std::uint64_t entry = enter_node_id_;

    for (int layer = max_level_; layer > 0; --layer) {
      entry = GreedyClosest(query_vector, entry, layer);
    }

    const std::size_t ef = std::max(k, kEfSearch);
    std::vector<Candidate> found = SearchLayer(
        query_vector, std::vector<std::uint64_t>{entry}, ef, 0);

    std::sort(found.begin(), found.end(),
              [](const Candidate& a, const Candidate& b) noexcept {
                return a.distance < b.distance;
              });

    result.reserve(std::min(k, found.size()));
    for (const Candidate& candidate : found) {
      if (result.size() >= k) {
        break;
      }
      result.push_back(candidate.id);
    }
  } catch (...) {
    result.clear();
  }
  return result;
}

}