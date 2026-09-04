// ==============================
// tests/unit_tests.cpp
// ==============================
//
// Self-contained test suite (no external framework). Returns 0 on success and
// the number of failed checks otherwise, so it plugs straight into CTest /
// CI. Covers the arena allocator, the SoA store, the SIMD distance kernel, and
// an end-to-end HNSW recall gate.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "arena_allocator.h"
#include "embedding_store.h"
#include "hnsw_index.h"
#include "simd_math.h"
#include "synthetic_dataset.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& name) {
  if (condition) {
    std::cout << "  [pass] " << name << "\n";
  } else {
    std::cout << "  [FAIL] " << name << "\n";
    ++g_failures;
  }
}

// ---- Arena ----------------------------------------------------------------

void TestArena() {
  std::cout << "Arena allocator:\n";
  fig::core::ArenaAllocator arena(/*block_bytes=*/256);

  // Alignment is honoured.
  void* p16 = arena.Allocate(8, 16);
  void* p64 = arena.Allocate(8, 64);
  Check(reinterpret_cast<std::uintptr_t>(p16) % 16 == 0, "16-byte alignment");
  Check(reinterpret_cast<std::uintptr_t>(p64) % 64 == 0, "64-byte alignment");

  // Distinct allocations do not overlap and are writable.
  float* a = arena.AllocateArray<float>(4);
  float* b = arena.AllocateArray<float>(4);
  for (int i = 0; i < 4; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(-i);
  }
  bool independent = true;
  for (int i = 0; i < 4; ++i)
    if (a[i] != static_cast<float>(i)) independent = false;
  Check(independent && a != b, "non-overlapping writable regions");

  // A request larger than the block size still succeeds (oversized block).
  void* big = arena.Allocate(1024, 32);
  Check(big != nullptr, "oversized request honoured");
  Check(arena.bytes_used() > 0, "bytes_used tracks payload");

  // Reset clears state.
  arena.Reset();
  Check(arena.block_count() == 0 && arena.bytes_used() == 0, "reset clears");
}

// ---- EmbeddingStore -------------------------------------------------------

void TestStore() {
  std::cout << "Embedding store:\n";
  fig::core::EmbeddingStore store(/*dimension=*/3);
  const std::vector<float> v0{1.0f, 2.0f, 3.0f};
  const std::vector<float> v1{4.0f, 5.0f, 6.0f};
  const std::size_t i0 = store.Add(v0);
  const std::size_t i1 = store.Add(v1);
  Check(i0 == 0 && i1 == 1, "dense increasing indices");

  const auto row = store.Row(1);
  Check(row[0] == 4.0f && row[2] == 6.0f, "round-trip read");
  Check(store.size() == 2 && store.dimension() == 3, "size/dimension");

  bool threw = false;
  try {
    store.Add(std::vector<float>{1.0f});  // wrong dimension
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "dimension mismatch throws");
}

// ---- SIMD kernel ----------------------------------------------------------

float NaiveSquared(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

void TestSimd() {
  std::cout << "SIMD distance kernel:\n";
  using K = fig::math::SpatialOptimization;

  std::vector<float> a, b;
  for (int i = 0; i < 130; ++i) {  // 130 exercises the 16/8/scalar tails
    a.push_back(static_cast<float>(i) * 0.5f);
    b.push_back(static_cast<float>(i) * 0.25f - 1.0f);
  }
  const float got = K::SquaredEuclideanDistance(a, b);
  const float want = NaiveSquared(a, b);
  Check(std::fabs(got - want) < 1e-2f, "matches naive reference");

  Check(K::SquaredEuclideanDistance(a, a) == 0.0f, "self-distance is zero");
  Check(std::fabs(K::SquaredEuclideanDistance(a, b) -
                  K::SquaredEuclideanDistance(b, a)) < 1e-3f,
        "symmetric");
  Check(std::fabs(K::EuclideanDistance(a, b) - std::sqrt(want)) < 1e-2f,
        "euclidean == sqrt(squared)");
}

// ---- HNSW recall gate -----------------------------------------------------

void TestHnswRecall() {
  std::cout << "HNSW recall gate:\n";
  fig::data::DatasetConfig cfg;
  cfg.identity_count = 400;
  cfg.shots_per_identity = 5;
  const fig::data::Dataset gallery = fig::data::GenerateGallery(cfg);
  const std::size_t n = gallery.vectors.size();
  const std::size_t k = 10;

  fig::core::HNSWIndex index(16);
  for (std::size_t i = 0; i < n; ++i)
    index.InsertProfile(static_cast<std::uint64_t>(i), "", gallery.vectors[i]);

  Check(index.Size() == n, "all nodes inserted");
  Check(index.SearchKNN(std::vector<float>{}, k).empty(), "empty query -> empty");

  // Recall vs exact brute force over a sample of probes.
  std::size_t hits = 0, total = 0, returned_k_ok = 0;
  const std::size_t probes = 100;
  for (std::size_t q = 0; q < probes; ++q) {
    const std::vector<float>& query = gallery.vectors[q * (n / probes)];
    std::vector<std::pair<float, std::uint64_t>> gt;
    gt.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
      gt.push_back({NaiveSquared(query, gallery.vectors[i]), i});
    std::partial_sort(gt.begin(), gt.begin() + k, gt.end());
    std::unordered_set<std::uint64_t> truth;
    for (std::size_t i = 0; i < k; ++i) truth.insert(gt[i].second);

    const auto got = index.SearchKNN(query, k);
    if (got.size() == k) ++returned_k_ok;
    for (const std::uint64_t id : got)
      if (truth.count(id)) ++hits;
    total += k;
  }
  const double recall = static_cast<double>(hits) / static_cast<double>(total);
  std::cout << "    measured recall@" << k << " = " << recall << "\n";
  Check(returned_k_ok == probes, "always returns k results");
  Check(recall >= 0.85, "recall@10 >= 0.85");
}

}  // namespace

int main() {
  TestArena();
  TestStore();
  TestSimd();
  TestHnswRecall();

  std::cout << "\n"
            << (g_failures == 0 ? "ALL TESTS PASSED"
                                : std::to_string(g_failures) + " CHECK(S) FAILED")
            << "\n";
  return g_failures;
}
