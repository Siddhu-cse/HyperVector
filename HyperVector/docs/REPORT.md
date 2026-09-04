# HyperVector: A Native C++ Vector Search Engine for Facial Identity Retrieval

> **Project report scaffold.** This document contains the technical substance —
> theory, architecture, complexity analysis, methodology, and the results
> measured on the reference machine. Adapt it to your institution's report
> template, **re-run the benchmark on your own hardware and replace the numbers
> with yours**, write the final prose in your own words, and check your course's
> policy on permitted assistance before submission.

---

## Abstract

HyperVector is a dependency-free C++20 engine that performs approximate
nearest-neighbour (ANN) search over high-dimensional facial embeddings. It builds
a multi-layer Hierarchical Navigable Small World (HNSW) graph to answer
similarity queries in expected **O(log N)** time, versus the **O(N)** of a linear
scan. The engine combines three systems-level optimizations: a hand-written
**AVX2/FMA SIMD** distance kernel, a **custom monotonic arena allocator** for
contiguous Structure-of-Arrays embedding storage, and **cache-line-aligned**
graph nodes. On a 10,000-vector, 128-dimensional gallery it achieves a recall@10
of 0.947 at ~4,500 queries per second on a single thread, and its speedup over
exact brute-force search grows from 3.6× at 10,000 vectors to 8.1× at 40,000 —
empirically demonstrating the logarithmic scaling that motivates the design.

---

## 1. Introduction

Modern facial recognition systems represent each face as a dense **embedding**: a
fixed-length vector (typically 128–512 floats) produced by a neural network such
that two images of the same person map to nearby vectors and different people map
to distant ones. Identifying a probe face against a gallery of *N* enrolled
identities therefore reduces to a **nearest-neighbour search** in vector space.

The naive approach compares the probe against all *N* vectors — exact, but
**O(N)** per query, which is untenable for galleries of millions. HyperVector
instead builds an index that trades a small, controllable amount of accuracy for
a large, asymptotic reduction in query cost.

**Contributions of this project.**
1. A correct, tested HNSW index over an owning, cache-aligned node model.
2. An AVX2 SIMD distance kernel with a portable scalar fallback.
3. A custom arena allocator realizing the Structure-of-Arrays layout that
   production vector databases rely on.
4. A reproducible benchmark quantifying recall, latency, throughput, and the
   speedup-vs-N scaling curve.

---

## 2. Background

### 2.1 The curse of dimensionality
Classic spatial indexes (k-d trees, ball trees) degrade to linear scans in high
dimensions because the data becomes nearly equidistant. Graph-based ANN methods
sidestep this by navigating a proximity graph rather than partitioning space.

### 2.2 Approximate nearest neighbour
An **exact** k-NN search returns the true *k* closest vectors. An **approximate**
search returns *k* vectors that are *usually* the closest. Quality is measured by
**recall@k** = (number of true top-*k* returned) / *k*, averaged over queries. The
engineering goal is high recall at low latency.

### 2.3 Distance metric
Face-embedding models are trained so that **cosine similarity** separates
identities. For unit-normalized vectors,

```
||a - b||² = ||a||² + ||b||² - 2·(a·b) = 2 - 2·cos(a, b),
```

so squared-Euclidean distance is a strictly decreasing function of cosine
similarity. Ranking by squared-L2 is therefore identical to ranking by cosine,
and avoids a per-comparison `sqrt`. HyperVector normalizes embeddings and ranks
by squared-L2 throughout.

---

## 3. HNSW: theory of operation

HNSW generalizes the **skip list** to a navigable proximity graph.

### 3.1 Layered structure
Every inserted node is assigned a maximum **level** drawn from an exponentially
decaying distribution:

```
level = floor( -ln(U) · mL ),     U ~ Uniform(0,1],     mL = 1 / ln(M)
```

A node present at level *ℓ* also appears at every level below it. The result is a
pyramid: layer 0 contains all *N* nodes; each higher layer holds a geometrically
smaller, sparser sample. The top layers are "express lanes" for covering large
distances in few hops; layer 0 is dense for fine-grained search.

### 3.2 Search algorithm
A query proceeds in two phases:
1. **Greedy descent (upper layers).** Starting at the single global entry point on
   the top layer, repeatedly move to the neighbour closest to the query until no
   neighbour improves; drop down one layer and repeat. This routes the search into
   the right neighbourhood in O(log N) hops.
2. **Beam search (layer 0).** From the entry found above, run a best-first search
   that maintains a candidate frontier (min-heap by distance) and a result set of
   the `ef` best nodes seen (max-heap). Expansion stops when the nearest unexpanded
   candidate is farther than the current worst result. The `ef ≥ k` parameter is
   the **recall/latency dial**: larger `ef` explores more of the graph, raising
   recall at the cost of more distance computations.

### 3.3 Insertion
Insertion runs the same descent to the new node's level, then for each layer from
that level down to 0 performs a beam search (with `efConstruction`), selects up to
`M` neighbours, and creates **bidirectional** edges. When a node exceeds its degree
bound it is pruned back to its closest `M` neighbours, keeping the graph sparse and
bounded-degree. If the new node's level exceeds the current maximum, it becomes the
global entry point.

### 3.4 Why O(log N)
Each layer is navigated in roughly constant expected hops (small-world property),
and there are O(log N) layers, giving expected **O(log N)** search. Memory is
**O(N·M)** for the graph plus **O(N·d)** for the vectors.

---

## 4. System architecture

| Module | Namespace | Responsibility | Key design choice |
|--------|-----------|----------------|-------------------|
| `IdentityNode` | `fig::core` | Owning graph node: id, metadata, features, multilayer adjacency. | `alignas(64)` to prevent false sharing; `noexcept` move for fast vector reallocation. |
| `SpatialOptimization` | `fig::math` | Distance kernels. | AVX2/FMA intrinsics with scalar fallback; squared-L2 to skip `sqrt`. |
| `ArenaAllocator` | `fig::core` | Monotonic bump allocator. | Allocate-many/free-together lifetime; O(1) amortized alloc. |
| `EmbeddingStore` | `fig::core` | SoA embedding store on the arena. | Contiguous floats for cache-streaming exact scans. |
| `HNSWIndex` | `fig::core` | Build + k-NN search. | Greedy descent + `ef` beam search; bounded-degree pruning. |

The owning `IdentityNode` model guarantees correctness and clear ownership; the
arena-backed `EmbeddingStore` provides the cold-metadata-free, contiguous layout
recommended in `identity_node.h` for the scan path.

---

## 5. Implementation details

### 5.1 SIMD distance kernel
The squared-difference kernel processes **16 floats per loop iteration** across two
independent 256-bit YMM accumulators (`_mm256_sub_ps`, `_mm256_fmadd_ps`), folds
them, and horizontally reduces to a scalar, with an 8-wide and a scalar epilogue
for the tail. Two accumulators expose instruction-level parallelism so independent
FMA chains overlap in the CPU pipeline. The entire path is guarded by `__AVX2__`;
on other targets the portable x4-unrolled scalar kernel compiles instead, so the
same source runs everywhere.

### 5.2 Arena allocator
`ArenaAllocator` reserves 1 MiB blocks and hands out aligned sub-regions by
advancing a cursor (alignment computed on the absolute address, so 32-byte SIMD
alignment is possible). Requests larger than a block trigger an oversized block.
There is no per-object free; the destructor releases all blocks. This removes *N*
heap allocations for embeddings and keeps consecutively inserted vectors
contiguous.

### 5.3 Exception & thread safety
All ownership is via `std::vector`/`std::unique_ptr` (RAII); no raw owning
pointers. `InsertProfile` provides the basic guarantee (a throw mid-wiring leaves
a valid, queryable index). `SearchKNN` is `noexcept` and returns an empty result
on any internal failure. Build is single-writer; a fully built index is immutable
and safe for concurrent readers.

---

## 6. Complexity analysis

| Operation | Time | Space |
|-----------|------|-------|
| Distance (dimension *d*) | O(d) | O(1) |
| Insert one node | O(log N · efConstruction · d) expected | O(M) edges added |
| Search k-NN | **O(log N · ef · d)** expected | O(ef) |
| Build (N nodes) | O(N · log N · efConstruction · d) | O(N·M + N·d) |
| Brute-force search (baseline) | O(N · d) | O(1) |

The search bound is independent of *N* except through the log factor, which is the
source of the widening speedup over brute force as *N* grows.

---

## 7. Experimental methodology

- **Data.** Synthetic clustered gallery (`synthetic_dataset.h`): each identity is a
  random unit-vector center; each enrolled shot is the center plus Gaussian jitter
  scaled by `1/√d` (so total noise is dimension-independent), renormalized. Probes
  are fresh, never-enrolled shots of a known identity.
- **Ground truth.** Exact top-*k* by brute force over the arena SoA store.
- **Parameters.** `d = 128`, `M = 16`, `efConstruction = 200`, `efSearch = 64`,
  `k = 10`, single thread.
- **Measured.** SIMD ns/op (with a checksum to defeat dead-code elimination and a
  pinned non-vectorized baseline), build throughput, recall@10, latency
  percentiles, QPS, and HNSW-vs-brute-force speedup.

---

## 8. Results

**Reference machine (replace with yours):** GCC 13, `-O3 -mavx2 -mfma`, x86-64.

**SIMD microbenchmark (128-d):** scalar 86.3 ns/op → AVX2 35.0 ns/op = **2.46×**.

**10,000 vectors:** build 3.07 s (≈3,260 vec/s); recall@10 = **0.947**; latency
mean 221 µs, p50 214 µs, p95 295 µs, p99 356 µs; **≈4,520 QPS**; **3.6×** vs brute
force. Arena: 4.88 MiB across 5 blocks.

**Scaling:**

| Vectors | Recall@10 | Mean latency | Speedup vs brute force |
|---------|-----------|--------------|------------------------|
| 10,000  | 0.947     | 221 µs       | 3.6×                   |
| 40,000  | 0.832     | 390 µs       | 8.1×                   |

### Discussion
The speedup over brute force more than doubles (3.6×→8.1×) for a 4× increase in
gallery size, consistent with O(N)-vs-O(log N) scaling: brute-force cost grows
linearly while HNSW query cost grows only logarithmically. Recall declines slightly
at the larger size because `efSearch` is held fixed; widening the beam restores it
(§9). The AVX2 kernel directly shrinks the dominant cost of both build and search —
the inner-product/distance computation.

---

## 9. Tuning & known improvements

- **`efSearch` (recall/latency dial).** Currently fixed at 64. Raising it to 128–256
  lifts recall at scale for a modest latency cost; exposing it as a `SearchKNN`
  parameter is a one-line change.
- **Diversity neighbour heuristic (Malkov & Yashunin, Algorithm 4).** The index uses
  the *simple* selection (closest `M`). The paper's heuristic — keep a candidate only
  if it is closer to the query than to any already-selected neighbour — yields a more
  navigable graph and higher recall at large *N*. It is the standard upgrade path.
- **Layer-0 degree `M₀ = 2·M`.** Allowing twice the degree on the densest layer is the
  paper's recommended setting and improves layer-0 recall.

---

## 10. Limitations
- Up to 2³² nodes (32-bit internal ids).
- Single-writer build (no concurrent insertion).
- Synthetic data; real face embeddings (e.g. LFW + ArcFace) would strengthen the
  evaluation.
- No on-disk persistence/serialization of the index.

## 11. Future work
Concurrent insertion with fine-grained locking; index serialization; product
quantization to compress vectors; the Algorithm-4 heuristic and `M₀ = 2M`;
runtime CPU-feature dispatch (AVX-512 where available); a multi-threaded query
server.

---

## References
1. Yu. A. Malkov, D. A. Yashunin. *Efficient and robust approximate nearest
   neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI,
   2020 (arXiv:1603.09320).
2. W. Pugh. *Skip Lists: A Probabilistic Alternative to Balanced Trees.* CACM, 1990.
3. J. Johnson, M. Douze, H. Jégou. *Billion-scale similarity search with GPUs
   (FAISS).* IEEE Big Data, 2019.
4. Pinecone / Milvus production vector-database architecture documentation.
5. Intel 64 and IA-32 Architectures Software Developer's Manual (AVX2 intrinsics).
