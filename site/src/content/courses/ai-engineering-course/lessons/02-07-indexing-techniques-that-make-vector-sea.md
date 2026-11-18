---
id: "02.07"
course: "ai-engineering-course"
module: "02"
slug: "indexing-techniques-that-make-vector-search-scale"
title: "Indexing techniques that make vector search scale"
keyIdeas: ["IVF (inverted file): k-means splits vectors into nlist clusters and a query scans only the nprobe closest; memory-efficient and pairs with PQ, but misses neighbours across cluster borders.", "HNSW builds a layered graph and searches it greedily from sparse long-range layers down to a dense bottom layer; M, efConstruction and efSearch trade recall for memory and speed.", "DiskANN keeps a Vamana graph and full vectors on SSD and compressed vectors in RAM, serving billion-scale collections from one machine.", "Strict filters break graph search: the filtered subgraph falls apart. Engines switch to brute force on the filtered set, add filter-aware edges, or keep scanning until enough results pass.", "Read a recall vs QPS curve at your target recall: build parameters move the curve, the query knob moves you along it."]
question: "Explain how HNSW finds a nearest neighbour, name the parameters that trade recall for speed and memory, and say what goes wrong when a query filter matches only 0.5% of the vectors."
answer: "HNSW stores vectors in a multi-layer proximity graph. A query enters at the sparse top layer, greedily hops to whichever neighbour is closest to the query until none is closer, drops to the next layer and repeats; at layer 0 it keeps a candidate list of size efSearch and returns the best k. M sets links per node (recall and robustness against memory and build time), efConstruction sets build quality, and efSearch sets query-time recall against latency. With a filter matching 0.5% of vectors, post-filtering the usual candidate list returns almost nothing, and restricting traversal to matching nodes breaks the graph into disconnected islands, so the search gets stuck and recall collapses. The fix is to detect the selective filter and run exact search over the matching vectors (cheap, because there are few of them), or use an engine with filter-aware graph edges or iterative scanning."
rubric: ["Describes the layered greedy descent and the efSearch candidate list at layer 0.", "Names M, efConstruction and efSearch with the right direction of each trade-off.", "Explains that post-filtering returns too few results for a selective filter.", "Explains that traversing only matching nodes disconnects the graph.", "Proposes brute force over the filtered set, filter-aware edges, or iterative scans."]
sources: [{"title": "Malkov & Yashunin: HNSW (2016)", "url": "https://arxiv.org/abs/1603.09320", "note": "Hierarchical navigable small world graphs."}, {"title": "Subramanya et al.: DiskANN (NeurIPS 2019)", "url": "https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html", "note": "The Vamana graph and billion-point search from SSD on one machine."}, {"title": "Pinecone: HNSW explained", "url": "https://www.pinecone.io/learn/series/faiss/hnsw/", "note": "Layers, M, efConstruction and efSearch with diagrams."}, {"title": "Faiss wiki: Guidelines to choose an index", "url": "https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index", "note": "Which index for which size and memory budget."}]
practice: ["ann-recall-at-k", "kmeans-step", "embedding-similarity", "precision-recall-at-k"]
tools: ["HNSW", "IVF", "DiskANN", "Faiss", "Qdrant"]
---

## IVF: search a few clusters

1. **Train**: run k-means on a sample to get `nlist` centroids (for example 4,096).
2. **Assign**: put every vector in the list of its nearest centroid.
3. **Query**: find the `nprobe` centroids closest to the query (say 16) and scan only those lists.

~~~dg-flow IVF scans a small fraction of the data.
Query -(nearest centroids)-> *nprobe of nlist clusters -(scan)-> Candidates -(sort)-> Top k
~~~

With `nlist` = 4,096 and `nprobe` = 16 you scan about 16 / 4,096 = 0.4% of the vectors. Raising `nprobe` increases recall and cost linearly. IVF combines naturally with product quantization of residuals (IVF-PQ, [Vector compression](lesson:aieng:02.04)), which is how billion-scale indexes fit in RAM. In the figure, put the query near the border between two clusters: its true nearest neighbour may sit just across the border, in a list you didn't scan. That is the recall IVF gives up.

~~~dg-raw
%%widget ivf IVF search: the query scans only the nprobe nearest clusters.
~~~

## HNSW: navigate a graph

Malkov and Yashunin's HNSW links each vector to nearby vectors and organises the links in layers like a skip list. The top layers are sparse with long-range links; layer 0 holds every vector.

~~~dg-layers Search drops through the layers, getting more local each time.
Layer 2 | a handful of nodes, long jumps across the space
Layer 1 | more nodes, medium jumps
*Layer 0 | every vector, short links to close neighbours
~~~

Search starts at an entry point on the top layer, greedily moves to whichever neighbour is closest to the query, and when no neighbour is closer it drops a layer. At layer 0 it keeps a candidate list of size `efSearch` and returns the best k. The graph adds roughly M × 2 × 4 bytes per vector on top of the vectors, all in RAM, and deletes are awkward: most implementations mark deleted nodes and rebuild or repair later.

~~~dg-raw
%%widget hnsw HNSW search: greedy hops through the layers toward the query.
~~~

## Tuning knobs

| Index | Knob | Effect of raising it |
| --- | --- | --- |
| IVF | nlist (clusters) | Smaller lists, faster scans, but needs higher nprobe for the same recall |
| IVF | nprobe | Higher recall, linearly more work per query |
| HNSW | M (links per node) | Better recall and robustness, more memory, slower build |
| HNSW | efConstruction | Better graph quality, slower build |
| HNSW | efSearch | Higher recall, slower queries; can be set per query |

Starting points that work for many workloads: IVF with `nlist` about the square root of the vector count and `nprobe` 1 to 5% of `nlist`; HNSW with M = 16, efConstruction = 200, then raise efSearch until recall hits your target. The Faiss guidelines reduce the choice to size and memory: flat or HNSW under about a million vectors, HNSW for millions when RAM allows, IVF with PQ for hundreds of millions on tight memory, and a disk-based graph for billions.

## DiskANN: a graph on SSD

HNSW needs every vector and the whole graph in RAM. DiskANN (Subramanya et al., 2019) keeps them on SSD instead. Its graph, called Vamana, has a single layer built with a pruning rule that deliberately keeps some long edges, so any point is reachable in few hops. Each node's full vector and neighbour list are stored together in one SSD block, and a PQ-compressed copy of every vector stays in RAM to guide the search. A query runs a beam search: compressed distances decide which nodes to expand, each expansion is one SSD read that also returns the full vector, and the final candidates are re-ranked with those exact vectors. The paper reports a billion SIFT vectors served from one 64 GB machine with an SSD at over 5,000 QPS, under 3 ms mean latency and 95%+ recall@1. Milvus and several other databases offer DiskANN-style indexes for collections too large for RAM.

## Filters break graphs

Most real queries carry a filter: this tenant, this language, in stock. How the index handles it decides whether you get results at all.

- **Post-filter**: search normally, then drop non-matching results. With a selective filter, few or none of the efSearch candidates match.
- **Filtered traversal**: only walk through matching nodes. With a strict filter, the matching nodes no longer form a connected graph; the walk gets stuck in a small island and recall collapses.
- **What engines do**: Qdrant adds extra graph edges between vectors that share indexed payload values and falls back to a full scan when the planner estimates few matches; pgvector 0.8 keeps scanning until enough rows pass ([Matching storage to your workload](lesson:aieng:02.03)); other engines use filter-aware graph builds such as ACORN or Filtered-DiskANN.

<div class="ex">
<div class="lab">Worked example</div>

**20 M vectors, 768 dims: index choice and a selective filter**

*Index choice.* HNSW at M = 16 gives sub-10 ms queries at 0.98 recall but needs the 61 GB of vectors plus 20M × 128 bytes = 2.6 GB of graph in RAM. IVF-PQ with `nlist` = 4,096 and 96-byte codes needs about 2 GB and rebuilds quickly, but needs rescoring to get above about 0.9 recall. With RAM to spare and slowly changing data, choose HNSW; with tight memory or frequent rebuilds, IVF-PQ with rescoring.

*A filter matching 0.5%.* That is 100,000 vectors. Exact search over just those reads 100,000 × 768 × 4 bytes = 307 MB, about 1.5 ms at 200 GB/s, with perfect recall. Graph search would be slower and less accurate here. The rule: estimate the filter's match count first, brute-force small sets, and use the graph only when the filter keeps a large share of the data.

</div>

## Reading a recall vs QPS curve

Benchmarks such as ann-benchmarks plot recall on the x-axis against queries per second (log scale) on the y-axis, one curve per index configuration. Up and to the right is better. Read it in three steps:

1. **Draw a vertical line at your target recall** (say 0.95). Compare indexes only where they cross it; the fastest at 0.80 may be the slowest at 0.95.
2. **Find the knee.** Curves are flat, then fall steeply as recall approaches 1. The last points before the knee cost little; the last 1% of recall can halve throughput.
3. **Know what moves what.** Query knobs (efSearch, nprobe) move you along one curve. Build parameters (M, efConstruction, nlist, quantization) move the whole curve, and also change memory and build time, which the plot does not show.

Benchmark curves use mean latency on one dataset with no filters and no concurrent writes. Re-plot on your own vectors, with p99 latency, before you choose.

## Where this breaks

- **Stale IVF centroids.** Clusters trained on last year's data become unbalanced as new content arrives; some lists grow huge and slow. Retrain periodically.
- **Deletes degrade HNSW.** Many marked-deleted nodes waste search effort and can cut paths through the graph; compaction or rebuilds restore recall ([Inside a vector database](lesson:aieng:02.09)).
- **Filters you didn't benchmark.** Measure recall on filtered queries at the selectivities you actually serve.

<div class="take">
<div class="lab">Key takeaway</div>

IVF narrows the search to a few clusters, HNSW walks a layered graph from coarse to fine, and DiskANN moves the graph to SSD. Each has a query-time knob that trades recall for speed; check where filters make the graph fall apart and brute-force small filtered sets.

</div>
