// ==============================
// include/synthetic_dataset.h
// ==============================
//
// High-performance facial identity graph engine — synthetic data generator.
//
// Produces L2-normalized, *clustered* embeddings that imitate a real facial
// recognition gallery: each "identity" is a random center direction on the unit
// hypersphere, and each enrolled image is that center plus small Gaussian jitter
// (intra-identity variation), renormalized to unit length. This gives the index
// genuine neighbourhood structure — the true nearest neighbours of a probe are
// the other shots of the same identity — so recall is a meaningful quality
// signal rather than noise.
//
// Why unit vectors: the engine ranks by squared-Euclidean distance, and on
// unit-normalized vectors squared-L2 is a strictly monotonic function of cosine
// distance (||a-b||^2 = 2 - 2*cos). Ranking is therefore identical to cosine
// similarity — the metric face-embedding models (ArcFace, FaceNet) are trained
// for — while keeping the engine's sqrt-free hot path.
//
// Each identity's center is stored explicitly so a fresh, never-enrolled probe
// of a known identity can be generated deterministically without fragile RNG
// stream replay.

#ifndef FIG_DATA_SYNTHETIC_DATASET_H
#define FIG_DATA_SYNTHETIC_DATASET_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace fig::data {

/// One generated gallery: vectors, their identities, and the cluster centers.
struct Dataset {
  std::size_t dimension{0};
  std::size_t identity_count{0};
  std::vector<std::vector<float>> vectors;            // enrolled embeddings
  std::vector<std::uint32_t> identity_of_vector;      // parallel to `vectors`
  std::vector<std::vector<float>> identity_centers;   // size == identity_count
};

/// @brief Configuration for synthetic gallery generation.
struct DatasetConfig {
  std::size_t dimension = 128;        // embedding width (face models ~128-512)
  std::size_t identity_count = 1000;  // distinct people
  std::size_t shots_per_identity = 5; // enrolled images per person
  float intra_identity_noise = 0.35f; // jitter magnitude (0 = identical shots)
  std::uint64_t seed = 0xC0FFEEULL;
};

namespace detail {

inline void Normalize(std::vector<float>& v) noexcept {
  float norm = 0.0f;
  for (const float x : v) norm += x * x;
  norm = std::sqrt(norm);
  if (norm <= 1e-30f) return;
  const float inv = 1.0f / norm;
  for (float& x : v) x *= inv;
}

}  // namespace detail

/// @brief Generate a clustered, unit-normalized embedding gallery.
[[nodiscard]] inline Dataset GenerateGallery(const DatasetConfig& cfg) {
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);

  Dataset out;
  out.dimension = cfg.dimension;
  out.identity_count = cfg.identity_count;
  const std::size_t total = cfg.identity_count * cfg.shots_per_identity;
  out.vectors.reserve(total);
  out.identity_of_vector.reserve(total);
  out.identity_centers.reserve(cfg.identity_count);

  for (std::size_t identity = 0; identity < cfg.identity_count; ++identity) {
    std::vector<float> center(cfg.dimension);
    for (float& c : center) c = gauss(rng);
    detail::Normalize(center);

    // Scale per-dimension jitter by 1/sqrt(dim) so the *total* noise magnitude
    // is ~intra_identity_noise regardless of dimension. Without this, noise
    // grows like sqrt(dim) and drowns the unit-length center in high dimensions.
    const float jitter =
        cfg.intra_identity_noise / std::sqrt(static_cast<float>(cfg.dimension));

    for (std::size_t shot = 0; shot < cfg.shots_per_identity; ++shot) {
      std::vector<float> sample(cfg.dimension);
      for (std::size_t d = 0; d < cfg.dimension; ++d) {
        sample[d] = center[d] + jitter * gauss(rng);
      }
      detail::Normalize(sample);
      out.vectors.push_back(std::move(sample));
      out.identity_of_vector.push_back(static_cast<std::uint32_t>(identity));
    }
    out.identity_centers.push_back(std::move(center));
  }
  return out;
}

/// @brief Produce a fresh probe (a held-out shot) from a stored identity center.
/// @param center Identity center, e.g. Dataset::identity_centers[id].
/// @param noise  Jitter magnitude (typically DatasetConfig::intra_identity_noise).
/// @param seed   Probe-specific seed for reproducibility.
[[nodiscard]] inline std::vector<float> GenerateProbe(std::span<const float> center,
                                                      float noise,
                                                      std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  const float jitter = noise / std::sqrt(static_cast<float>(center.size()));
  std::vector<float> probe(center.size());
  for (std::size_t d = 0; d < center.size(); ++d) {
    probe[d] = center[d] + jitter * gauss(rng);
  }
  detail::Normalize(probe);
  return probe;
}

}  // namespace fig::data

#endif  // FIG_DATA_SYNTHETIC_DATASET_H
