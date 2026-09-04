// ==============================
// examples/demo.cpp
// ==============================
//
// Live demonstration of the HyperVector facial identity search engine.
//
// Story: we enroll a gallery of distinct identities (several "photos" each),
// then present a fresh probe image of one known identity and ask the engine to
// retrieve the closest matches. The engine returns the gallery entries nearest
// to the probe in embedding space; the metadata reveals which person they are.

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "hnsw_index.h"
#include "synthetic_dataset.h"

int main() {
  // --- Configure a small, human-readable gallery ---------------------------
  fig::data::DatasetConfig cfg;
  cfg.dimension = 128;
  cfg.identity_count = 50;
  cfg.shots_per_identity = 4;
  cfg.intra_identity_noise = 0.30f;

  const fig::data::Dataset gallery = fig::data::GenerateGallery(cfg);

  std::cout << "HyperVector — facial identity search demo\n";
  std::cout << "Enrolling " << gallery.vectors.size() << " face embeddings ("
            << cfg.identity_count << " identities x " << cfg.shots_per_identity
            << " shots, dim=" << cfg.dimension << ")\n\n";

  // --- Build the index. Metadata records the human identity per node. ------
  fig::core::HNSWIndex index(/*M=*/16);
  for (std::size_t i = 0; i < gallery.vectors.size(); ++i) {
    const std::uint32_t person = gallery.identity_of_vector[i];
    index.InsertProfile(static_cast<std::uint64_t>(i),
                        "person_" + std::to_string(person),
                        gallery.vectors[i]);
  }
  std::cout << "Index built. nodes=" << index.Size()
            << "  top_layer=" << index.MaxLevel()
            << "  entry_node=" << index.EntryNode() << "\n\n";

  // --- Present a fresh probe of a known identity ---------------------------
  const std::uint32_t target_identity = 7;
  const std::vector<float> probe = fig::data::GenerateProbe(
      gallery.identity_centers[target_identity], cfg.intra_identity_noise,
      /*seed=*/0xFACEuLL);

  std::cout << "Query: a new photo of person_" << target_identity
            << " (never enrolled)\n";
  std::cout << "Top-5 matches returned by the engine:\n";
  std::cout << std::string(48, '-') << "\n";

  const std::vector<std::uint64_t> matches = index.SearchKNN(probe, 5);
  for (std::size_t rank = 0; rank < matches.size(); ++rank) {
    const std::uint64_t node_id = matches[rank];
    const std::uint32_t person = gallery.identity_of_vector[node_id];
    std::cout << "  #" << (rank + 1) << "  node " << std::setw(5) << node_id
              << "  ->  person_" << person
              << (person == target_identity ? "   [correct identity]" : "")
              << "\n";
  }

  std::cout << std::string(48, '-') << "\n";
  const bool top1_correct =
      !matches.empty() &&
      gallery.identity_of_vector[matches.front()] == target_identity;
  std::cout << "Top-1 identity match: " << (top1_correct ? "CORRECT" : "miss")
            << "\n";
  return 0;
}
