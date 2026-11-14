---
id: "02.07"
course: "ai-engineering-course"
module: "02"
slug: "indexing-techniques-that-make-vector-search-scale"
title: "Indexing techniques that make vector search scale"
keyIdeas: ["IVF (inverted file): k-means splits vectors into nlist clusters. A query scans only the nprobe closest clusters.", "IVF is memory-efficient and pairs well with PQ (IVF-PQ), but needs training and recall drops at cluster borders.", "HNSW builds a layered graph: few nodes and long links on top, all nodes and short links at the bottom.", "Search starts at the top layer, greedily moves to the closest neighbour, drops down a layer, and repeats.", "HNSW knobs: M (links per node: recall and memory), efConstruction (build quality), efSearch (query-time recall vs speed).", "HNSW is the common default for in-memory search; disk-based graphs like DiskANN handle billion-scale on SSDs."]
question: "Explain how HNSW finds a nearest neighbour, and name the parameters that trade recall for speed and memory."
answer: "HNSW stores vectors in a multi-layer proximity graph. A query enters at the sparse top layer, greedily hops to the neighbour closest to the query until no improvement, drops to the next layer and repeats; at layer 0 it explores a candidate list of size efSearch and returns the top k. M controls links per node (recall vs memory), efConstruction controls build quality, and efSearch controls query-time recall vs latency."
sources: [{"title": "Malkov & Yashunin: HNSW (2016)", "url": "https://arxiv.org/abs/1603.09320", "note": "Hierarchical navigable small world graphs."}, {"title": "Pinecone: HNSW explained", "url": "https://www.pinecone.io/learn/series/faiss/hnsw/", "note": "Layers, M, efConstruction and efSearch with diagrams."}, {"title": "Faiss wiki: Guidelines to choose an index", "url": "https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index", "note": "Which index for which size and memory budget."}]
tools: ["HNSW", "IVF"]
---

## Two families dominate

Most production vector search uses one of two index types, or a combination with quantization.

## IVF: search a few clusters

1. **Train**: run k-means on a sample to get nlist centroids (for example 4,096).
2. **Assign**: put every vector in the list of its nearest centroid.
3. **Query**: find the nprobe centroids closest to the query (say 16) and scan only those lists.

~~~dg-flow IVF scans a small fraction of the data.
Query -(nearest centroids)-> *nprobe of nlist clusters -(scan)-> Candidates -(sort)-> Top k
~~~

With nlist = 4,096 and nprobe = 16 you scan about 0.4% of vectors. Raising nprobe increases recall and cost. The weak spot: a true neighbour sitting just across a cluster boundary is missed unless you probe more clusters. IVF combines naturally with PQ (IVF-PQ in Faiss), which is how billion-scale indexes fit in RAM.

## HNSW: navigate a graph

Malkov and Yashunin's HNSW builds a graph where each vector links to its nearby vectors, organised in layers like a skip list. The top layers are sparse with long-range links; layer 0 has every vector.

~~~dg-layers Search drops through the layers, getting more local each time.
Layer 2 | a handful of nodes, long jumps across the space
Layer 1 | more nodes, medium jumps
*Layer 0 | every vector, short links to close neighbours
~~~

Search starts at an entry point on the top layer, greedily moves to whichever neighbour is closest to the query, and when no neighbour is closer it drops a layer. At layer 0 it keeps a candidate list of size efSearch and returns the best k.

| Parameter | Higher value means |
| --- | --- |
| M | More links per node: better recall, more memory, slower build |
| efConstruction | Better-quality graph, slower build |
| efSearch | Better recall, slower queries (tunable per query) |

Pinecone's HNSW guide shows the trade-off curves. HNSW gives excellent recall and latency but stores the graph in memory, adding roughly M x 2 x 4 bytes per vector on top of the vectors themselves, and deletes are awkward.

## Choosing

The Faiss wiki's guidelines boil down to:

- Under ~1M vectors: flat or HNSW.
- Millions, memory available: HNSW.
- Hundreds of millions, memory tight: IVF with PQ.
- Billions on SSD: disk-based graph indexes (DiskANN).

## IVF: click to move the query

In the IVF figure, points are grouped into clusters around centroids (the squares). A query only scans the clusters whose centroids are closest. Raise nprobe to scan more clusters: recall goes up and so does work.

~~~dg-raw
%%widget ivf IVF search: the query scans only the nprobe nearest clusters.
~~~

The failure case is easy to see: put the query near the border between two clusters. Its true nearest neighbor may sit just across the border, in a cluster you didn't scan. That is the recall IVF gives up.

## HNSW: watch the greedy walk

HNSW stores a graph where each point links to its near neighbors, with sparse upper layers for long jumps. Search starts at the top, greedily moves to whichever neighbor is closer to the query, drops a layer when no neighbor is closer, and repeats until the bottom layer.

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

Starting points that work for many workloads: IVF with nlist about the square root of the vector count and nprobe 1 to 5% of nlist; HNSW with M = 16, efConstruction = 200, then raise efSearch until recall hits your target.

<div class="ex">
<div class="lab">Worked example</div>

**picking between them for 20 M vectors**

HNSW at M = 16 gives sub-10 ms queries at 0.98 recall but needs all vectors plus about 2.5 GB of graph in RAM, and deletes are awkward.

IVF-PQ with nlist = 4,096 fits in a fraction of the memory and rebuilds quickly, but needs rescoring to get above about 0.9 recall.

If RAM is available and the data changes slowly, HNSW. If memory is tight or you rebuild often, IVF-PQ with rescoring.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

IVF narrows the search to a few clusters; HNSW walks a layered graph from coarse to fine. Both have one query-time knob (nprobe, efSearch) that trades recall for speed.

</div>
