# HyperVector — Presentation Notes

Everything here is to help you **present and defend** the project. Read it until
you can explain each piece in your own words — examiners reward understanding far
more than a big codebase.

---

## 1. The 90-second pitch

> "Facial recognition turns each face into a list of numbers — an *embedding* —
> where similar faces are close together. To identify a probe face you find its
> nearest neighbours in a gallery. Checking every entry is O(N) and doesn't scale.
> HyperVector builds an **HNSW graph** — think of a multi-level skip list over a
> proximity graph — so search is **O(log N)**. I wrote it in modern C++20 with no
> external libraries, added a hand-written **AVX2 SIMD** distance kernel that's
> about 2.5× faster than scalar, and a **custom arena allocator** for cache-
> friendly storage. On 10,000 faces it hits **0.947 recall@10** at ~4,500 queries
> per second, and its speedup over brute force grows with the dataset — which is
> the whole point of the logarithmic design."

## 2. Explain HNSW in plain English (the diagram to draw)

Draw a pyramid of dots:
- **Top layers** = few nodes, long-range "express" links. You start here and take
  big jumps toward the query.
- **You drop down** layer by layer, each time getting closer.
- **Bottom layer (layer 0)** = every node, dense local links. Here you do a careful
  "beam search" — keep the best `ef` candidates, expand the closest, stop when you
  can't improve.

Analogy: a skip list lets you skip ahead in a sorted list in O(log N); HNSW is the
same idea but in many dimensions, over a graph of "who is near whom."

## 3. The three optimizations, one line each
- **AVX2 SIMD:** the distance between two 128-float vectors is computed 8–16 floats
  at a time in one instruction instead of one float at a time → ~2.5× faster.
- **Arena allocator:** instead of N separate `new` calls for N vectors, grab big
  blocks and bump a pointer; vectors end up contiguous, so scanning streams through
  cache.
- **Cache-line alignment:** each node sits on its own 64-byte cache line so threads
  editing neighbouring nodes don't fight over the same line ("false sharing").

## 4. Demo run order (rehearse this)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/hv_demo        # shows it retrieving the right person from a fresh photo
./build/hv_tests       # all green, including the recall gate
./build/hv_benchmark   # the headline numbers, live
```
Talk track while it runs: "It enrolls 200 faces, I hand it a *new* photo of person 7
that was never enrolled, and the top matches are all person 7 — found by walking the
graph, not by scanning everyone."

## 5. Likely examiner questions — and solid answers

**Q: What is HNSW and why O(log N)?**
A multi-layer proximity graph. Upper layers are sparse for long jumps, layer 0 is
dense for precision. There are ~log N layers and each takes roughly constant hops,
so search is expected O(log N). (See report §3.)

**Q: Approximate — so it can be wrong. How do you measure quality?**
Recall@k against exact brute force. I measured 0.947 at 10k. It's tunable: the
`ef` search-width trades recall for latency.

**Q: Why squared-Euclidean, not cosine, for face embeddings?**
On unit-normalized vectors `||a-b||² = 2 - 2·cos`, so squared-L2 ranks identically
to cosine but skips the `sqrt`. I normalize all vectors, so it *is* cosine ranking.

**Q: Show me the AVX2 speedup is real and not a compiler trick.**
The baseline is pinned to non-vectorized codegen in its own file
(`scalar_reference.cpp`, `optimize("no-tree-vectorize")`), and the loop has a
checksum so the optimizer can't delete it. Measured 86 ns → 35 ns/op.

**Q: Where does the arena actually help?**
N embeddings become ~N pointer bumps instead of N mallocs, and rows are contiguous,
which is why the exact brute-force scan is fast. It's the SoA layout production
databases use.

**Q: Recall dropped from 0.947 to 0.832 at 40k — why?**
Expected: at a fixed `ef`, recall declines slowly as N grows because the beam covers
a smaller fraction of the graph. Raising `ef` restores it — the standard dial.
There's also a known graph-quality upgrade, the paper's diversity heuristic
(Algorithm 4), which I describe in the report's improvements section.

**Q: How would you scale to millions / production?**
Expose `ef`; add the Algorithm-4 heuristic and `M₀=2M`; product-quantize vectors to
cut memory; concurrent inserts with fine-grained locks; index serialization; AVX-512
with runtime dispatch; shard across a query server. (Report §11.)

**Q: Thread safety?**
Build is single-writer; a built index is immutable and safe for concurrent readers.

**Q: Biggest weakness?**
Synthetic data and the simple neighbour selection. I'd validate on a real benchmark
(e.g. LFW embeddings) and enable the diversity heuristic next.

## 6. What to be honest about
- The data is **synthetic but structured** to mimic real face galleries.
- The numbers are from **your** machine — run the benchmark and quote your results.
- The HNSW uses the **simple** neighbour selection; the diversity heuristic is the
  documented next step. Saying this *demonstrates* understanding — it doesn't weaken
  the project.

## 7. One-sentence summary to memorize
> "HyperVector is a from-scratch C++20 HNSW vector search engine: O(log N) graph
> search, AVX2 SIMD distance, and a custom arena allocator — 0.947 recall@10 with
> a speedup over brute force that grows with the dataset."
