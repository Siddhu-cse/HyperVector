# HyperVector — Native C++ Vector Search Engine

A high-performance, dependency-free C++20 engine for approximate nearest-neighbour
(ANN) search over high-dimensional embeddings, built around a **multi-layer HNSW
graph** (Hierarchical Navigable Small World). It targets the facial-identity
retrieval problem: given a probe face embedding, find the most similar enrolled
identities in **O(log N)** expected time instead of the O(N) of a linear scan.

The design follows the HNSW paper (Malkov & Yashunin, 2016) for the index, and
the Structure-of-Arrays storage philosophy used by production vector databases
(Pinecone, Milvus) for the cache-friendly data layout.

---

## Highlights

- **Multi-layer HNSW graph** with greedy descent through upper layers and a
  best-first beam search on layer 0.
- **AVX2 + FMA SIMD distance kernel** (hand-written intrinsics) with an automatic
  portable scalar fallback — measured **~2.5× faster** than scalar on this
  machine.
- **Custom monotonic arena allocator** providing contiguous, cache-streaming
  Structure-of-Arrays embedding storage (the layout recommended inside
  `identity_node.h`).
- **Cache-line-aligned graph nodes** (`IdentityNode`) to avoid false sharing.
- **Zero external dependencies** — only the C++20 standard library.
- **Tested**: arena, store, SIMD-correctness, and an end-to-end recall gate run
  under CTest.

## Measured results

Synthetic clustered gallery, 128-dimensional unit embeddings, `M = 16`,
`efConstruction = 200`, `efSearch = 64`, single thread. Numbers are from one
reference machine — **re-run `hv_benchmark` on your own hardware for your report**.

| Metric (10,000 vectors)        | Value                |
|--------------------------------|----------------------|
| SIMD kernel (scalar → AVX2)    | 86.3 → 35.0 ns/op (**2.46×**) |
| Build throughput               | ~3,260 vectors/s     |
| Recall@10 vs exact brute force | **0.947**            |
| Query latency (mean / p99)     | 221 µs / 356 µs      |
| Throughput                     | ~4,520 QPS           |
| Speedup vs brute force         | 3.6×                 |

Scaling (the O(log N) advantage widening with N):

| Vectors | Recall@10 | Mean latency | Speedup vs brute force |
|---------|-----------|--------------|------------------------|
| 10,000  | 0.947     | 221 µs       | 3.6×                   |
| 40,000  | 0.832     | 390 µs       | **8.1×**               |

Recall falls gently as N grows at a *fixed* search width; it is recovered by
raising `efSearch` (the standard recall/latency dial — see `docs/REPORT.md`).

---

## Architecture

```
                         ┌─────────────────────────────┐
   probe embedding  ──▶  │          HNSWIndex          │
                         │  (multi-layer graph search) │
                         └──────────────┬──────────────┘
                                        │ ranks by distance
                                        ▼
   ┌──────────────────┐   uses   ┌─────────────────────┐
   │ SpatialOptimiz.  │ ◀─────── │  GreedyClosest /     │
   │ (AVX2 SIMD dist) │          │  SearchLayer (beam)  │
   └──────────────────┘          └──────────┬──────────┘
                                            │ owns
                                            ▼
                                ┌───────────────────────┐
                                │  IdentityNode (×N)     │
                                │  cache-aligned, owns   │
                                │  features + adjacency  │
                                └───────────────────────┘

   ArenaAllocator ──▶ EmbeddingStore (SoA)   # contiguous mirror for exact scans
```

| File | Namespace | Responsibility |
|------|-----------|----------------|
| `include/identity_node.h`     | `fig::core` | Owning, cache-aligned graph node. |
| `include/simd_math.h`         | `fig::math` | Squared-/Euclidean distance; AVX2 path + scalar fallback. |
| `include/arena_allocator.h`   | `fig::core` | Monotonic arena (bump) allocator. |
| `include/embedding_store.h`   | `fig::core` | Arena-backed SoA embedding store. |
| `include/synthetic_dataset.h` | `fig::data` | Clustered unit-vector gallery + probe generator. |
| `src/hnsw_index.{h,cpp}`      | `fig::core` | HNSW build + k-NN search. |
| `examples/demo.cpp`           | —           | Enroll-and-match CLI demo. |
| `benchmarks/benchmark.cpp`    | —           | SIMD microbench + ANN recall/latency/QPS. |
| `tests/unit_tests.cpp`        | —           | Unit + recall gate (CTest). |

---

## Build & run

Requires CMake ≥ 3.16 and a C++20 compiler (GCC ≥ 10, Clang ≥ 12, or MSVC 2022).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/hv_demo                 # enroll-and-match demonstration
./build/hv_tests                # unit + recall tests
ctest --test-dir build          # same, via CTest
./build/hv_benchmark            # full benchmark (args: identities shots queries)
./build/hv_benchmark 4000 5 500 # 20,000-vector run
```

**No AVX2 on your CPU?** Configure with `-DFIG_ENABLE_AVX2=OFF`; the engine then
uses the portable scalar kernel automatically. (Building with AVX2 and running on
a CPU without it can crash with an illegal-instruction fault.)

---

## Notes & limitations

- Node ids are stored internally as 32-bit, so the index supports up to 2³² nodes.
- Index build is single-writer; searches over a built index are read-only and
  parallelizable across threads.
- Distance is squared-Euclidean on unit-normalized vectors, which ranks
  identically to cosine similarity (the face-embedding metric) without a sqrt.
- Tuning knobs and the published recall-boosting neighbour heuristic are
  discussed in `docs/REPORT.md` → *Tuning & known improvements*.
