// ==============================
// benchmarks/benchmark.cpp
// ==============================
//
// Reproducible performance + quality benchmark for the HyperVector engine.
//
// Sections:
//   1. SIMD kernel microbenchmark  — AVX2 intrinsic path vs a pinned scalar
//      baseline, reporting nanoseconds/op and speedup.
//   2. End-to-end ANN benchmark    — build throughput, query latency
//      distribution (mean / p50 / p95 / p99), QPS, recall@k against exact
//      brute force, and the HNSW-vs-brute-force speedup.
//
// Usage: ./hv_benchmark [identities] [shots_per_identity] [queries]

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

#include "embedding_store.h"
#include "hnsw_index.h"
#include "scalar_reference.h"
#include "simd_math.h"
#include "synthetic_dataset.h"

namespace {

using Clock = std::chrono::steady_clock;

double MillisSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void PrintRule() { std::cout << std::string(64, '-') << "\n"; }

// ---- 1. SIMD microbenchmark ----------------------------------------------

void RunSimdMicrobench(std::size_t dim) {
  std::cout << "\n[1] SIMD distance kernel microbenchmark (dim=" << dim << ")\n";
  PrintRule();

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  constexpr std::size_t kPairs = 4096;
  std::vector<std::vector<float>> a(kPairs), b(kPairs);
  for (std::size_t i = 0; i < kPairs; ++i) {
    a[i].resize(dim);
    b[i].resize(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      a[i][d] = nd(rng);
      b[i][d] = nd(rng);
    }
  }

  constexpr int kReps = 2000;
  volatile float sink = 0.0f;  // defeat dead-code elimination

  auto t0 = Clock::now();
  for (int r = 0; r < kReps; ++r)
    for (std::size_t i = 0; i < kPairs; ++i)
      sink += fig::math::SpatialOptimization::SquaredEuclideanDistance(a[i],
                                                                       b[i]);
  const double avx_ms = MillisSince(t0);

  t0 = Clock::now();
  for (int r = 0; r < kReps; ++r)
    for (std::size_t i = 0; i < kPairs; ++i)
      sink += fig::bench::ScalarSquaredDistance(a[i].data(), b[i].data(), dim);
  const double scalar_ms = MillisSince(t0);

  const double ops = static_cast<double>(kReps) * kPairs;
  const double avx_ns = avx_ms * 1e6 / ops;
  const double scalar_ns = scalar_ms * 1e6 / ops;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  scalar baseline : " << scalar_ns << " ns/op\n";
  std::cout << "  AVX2 intrinsic  : " << avx_ns << " ns/op\n";
  std::cout << "  speedup         : " << (scalar_ns / avx_ns) << "x\n";
  std::cout << "  (checksum " << static_cast<float>(sink) << ")\n";
}

// ---- 2. End-to-end ANN benchmark -----------------------------------------

struct ExactNeighbors {
  std::vector<std::uint64_t> ids;  // top-k by exact squared-L2
};

ExactNeighbors BruteForceTopK(const fig::core::EmbeddingStore& store,
                              std::span<const float> query, std::size_t k) {
  std::vector<std::pair<float, std::uint64_t>> scored;
  scored.reserve(store.size());
  for (std::size_t i = 0; i < store.size(); ++i) {
    const auto row = store.Row(i);
    float s = 0.0f;
    for (std::size_t d = 0; d < query.size(); ++d) {
      const float diff = query[d] - row[d];
      s += diff * diff;
    }
    scored.push_back({s, static_cast<std::uint64_t>(i)});
  }
  const std::size_t kk = std::min(k, scored.size());
  std::partial_sort(scored.begin(), scored.begin() + kk, scored.end());
  ExactNeighbors out;
  out.ids.reserve(kk);
  for (std::size_t i = 0; i < kk; ++i) out.ids.push_back(scored[i].second);
  return out;
}

void RunAnnBenchmark(std::size_t identities, std::size_t shots,
                     std::size_t queries) {
  fig::data::DatasetConfig cfg;
  cfg.identity_count = identities;
  cfg.shots_per_identity = shots;
  const std::size_t dim = cfg.dimension;
  const std::size_t k = 10;

  const fig::data::Dataset gallery = fig::data::GenerateGallery(cfg);
  const std::size_t n = gallery.vectors.size();

  std::cout << "\n[2] Approximate nearest-neighbour benchmark\n";
  PrintRule();
  std::cout << "  vectors=" << n << "  dim=" << dim
            << "  identities=" << identities << "  k=" << k << "\n";

  // Build the HNSW index (timed).
  fig::core::HNSWIndex index(/*M=*/16);
  auto t0 = Clock::now();
  for (std::size_t i = 0; i < n; ++i) {
    index.InsertProfile(static_cast<std::uint64_t>(i), "",
                        gallery.vectors[i]);
  }
  const double build_ms = MillisSince(t0);

  // Build the arena-backed SoA store for exact ground truth (timed).
  fig::core::EmbeddingStore store(dim);
  t0 = Clock::now();
  for (std::size_t i = 0; i < n; ++i) store.Add(gallery.vectors[i]);
  const double store_ms = MillisSince(t0);

  // Run probes: measure latency, recall vs exact, and brute-force time.
  std::mt19937_64 pick(123);
  std::vector<double> latencies_us;
  latencies_us.reserve(queries);
  std::size_t hits = 0, total = 0;
  double hnsw_us_sum = 0.0, brute_us_sum = 0.0;

  for (std::size_t q = 0; q < queries; ++q) {
    const std::uint32_t identity =
        static_cast<std::uint32_t>(pick() % identities);
    const std::vector<float> probe = fig::data::GenerateProbe(
        gallery.identity_centers[identity], cfg.intra_identity_noise,
        0xABCDuLL + q);

    auto s0 = Clock::now();
    const auto approx = index.SearchKNN(probe, k);
    const double hnsw_us =
        std::chrono::duration<double, std::micro>(Clock::now() - s0).count();
    latencies_us.push_back(hnsw_us);
    hnsw_us_sum += hnsw_us;

    s0 = Clock::now();
    const ExactNeighbors exact = BruteForceTopK(store, probe, k);
    brute_us_sum +=
        std::chrono::duration<double, std::micro>(Clock::now() - s0).count();

    std::unordered_set<std::uint64_t> truth(exact.ids.begin(),
                                            exact.ids.end());
    for (const std::uint64_t id : approx)
      if (truth.count(id)) ++hits;
    total += exact.ids.size();
  }

  std::sort(latencies_us.begin(), latencies_us.end());
  const auto pct = [&](double p) {
    const std::size_t idx = static_cast<std::size_t>(
        p * (static_cast<double>(latencies_us.size()) - 1.0));
    return latencies_us[idx];
  };

  const double recall = static_cast<double>(hits) / static_cast<double>(total);
  const double mean_us = hnsw_us_sum / static_cast<double>(queries);
  const double qps = 1e6 / mean_us;
  const double speedup = brute_us_sum / hnsw_us_sum;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  HNSW build      : " << build_ms << " ms  ("
            << (static_cast<double>(n) / (build_ms / 1000.0)) << " vec/s)\n";
  std::cout << "  Arena SoA build : " << store_ms << " ms  (arena blocks="
            << store.arena().block_count() << ", "
            << (store.arena().bytes_used() / (1024.0 * 1024.0)) << " MiB)\n";
  std::cout << "  recall@" << k << "       : " << recall << "\n";
  std::cout << "  query latency   : mean=" << mean_us << "us  p50=" << pct(0.50)
            << "us  p95=" << pct(0.95) << "us  p99=" << pct(0.99) << "us\n";
  std::cout << "  throughput      : " << qps << " QPS (single thread)\n";
  std::cout << "  vs brute force  : " << speedup << "x faster\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t identities = 2000;
  std::size_t shots = 5;
  std::size_t queries = 500;
  if (argc > 1) identities = std::stoul(argv[1]);
  if (argc > 2) shots = std::stoul(argv[2]);
  if (argc > 3) queries = std::stoul(argv[3]);

  std::cout << "HyperVector benchmark\n";
#if defined(__AVX2__)
  std::cout << "build: AVX2 intrinsics ENABLED\n";
#else
  std::cout << "build: scalar fallback (AVX2 not compiled in)\n";
#endif

  RunSimdMicrobench(128);
  RunAnnBenchmark(identities, shots, queries);
  std::cout << "\nDone.\n";
  return 0;
}
