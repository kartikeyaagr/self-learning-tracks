---
id: "02.06"
course: "ai-engineering-course"
module: "02"
slug: "vector-search"
title: "Vector search"
keyIdeas: ["Vector search returns the k stored vectors nearest to a query vector.", "Exact (flat, brute-force) search compares the query with every vector: perfect recall, cost linear in N × d, and bounded by memory bandwidth.", "Approximate nearest neighbour (ANN) indexes skip most vectors, trading a little recall for orders of magnitude less work.", "Judge an index by recall@k against exact search, together with latency (p50 and p99), QPS, memory and build time; a latency number without its recall means nothing.", "Every ANN index has a query-time knob (efSearch, nprobe) that picks a point on its recall/speed curve."]
question: "Why can't you just do exact k-NN over 50 million 768-dimension vectors per query, and how would you judge an approximate index to replace it?"
answer: "Exact k-NN computes a distance from the query to all 50 million vectors: 50M × 768 is about 38 billion multiply-adds, and it has to stream about 154 GB of float32 data from memory. Even at 200 GB/s of memory bandwidth that is around 0.8 seconds for one query on one machine, so a handful of queries per second at best. An ANN index avoids looking at most vectors. To judge one, build ground truth with exact search on a few hundred real queries, then measure recall@k: the fraction of the true top k the index returns. Report it together with p50 and p99 latency, queries per second at that recall, memory per vector and build or update time. Sweep the query-time knob (efSearch or nprobe) to get a curve and pick the cheapest point that meets your recall target, typically 0.95 or more for RAG."
rubric: ["Estimates the work: tens of billions of operations and over 100 GB read per query.", "Explains that ANN indexes skip most vectors to cut that work.", "Defines recall@k against exact-search ground truth on real queries.", "Pairs recall with latency (including tail), QPS and memory rather than reporting one number.", "Mentions sweeping efSearch or nprobe to choose a point on the recall/speed curve."]
sources: [{"title": "Pinecone: k-NN vs ANN", "url": "https://www.pinecone.io/learn/series/faiss/vector-indexes/", "note": "Flat, LSH, HNSW and IVF indexes compared on speed and recall."}, {"title": "ann-benchmarks", "url": "https://ann-benchmarks.com/", "note": "Recall vs queries-per-second for real ANN libraries."}, {"title": "Elastic: Reciprocal rank fusion", "url": "https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html", "note": "Merging keyword and vector results by rank."}]
practice: ["ann-recall-at-k", "embedding-similarity", "precision-recall-at-k", "document-ranking"]
tools: ["Faiss"]
---

## The problem

Given a query vector $q$ and $N$ stored vectors, return the $k$ closest under the model's metric (usually cosine or dot product on unit vectors, [Vector embeddings and semantic space](lesson:aieng:02.02)). That's it. The difficulty is doing it in milliseconds when $N$ is in the millions or billions.

~~~dg-flow Exact search touches every vector; an ANN index touches a small fraction.
[Exact] Query -(N distances)-> All N vectors -(keep best)-> True top k
[ANN] Query -(index)-> *A few % of vectors -(keep best)-> Approximate top k
~~~

## Exact search

A flat index computes the distance from $q$ to every vector and keeps the best $k$ in a heap. Recall is perfect and there is nothing to tune. Cost is $N \times d$ multiply-adds per query, and because each stored number is used once, the real limit is how fast the machine can stream the vectors from memory, not how fast it can multiply.

<div class="ex">
<div class="lab">Worked example</div>

**50 million vectors, 768 dims**

Work per query: 50M × 768 = 38.4 billion multiply-adds.

Data per query: 50M × 768 × 4 bytes = 154 GB of float32.

At 200 GB/s of memory bandwidth, streaming 154 GB takes about 0.77 s, so one machine serves one or two queries per second. Batching queries helps (each pass over memory serves the whole batch), and a GPU's higher bandwidth helps, but neither changes the linear growth.

At 500,000 vectors the same query reads 1.5 GB, about 8 ms: flat search is often the right answer at that size, and it is always the ground truth for testing an ANN index.

</div>

The figure estimates one exact query as the collection grows.

~~~dg-raw
%%widget flatcost Time for one brute-force query as the collection grows.
~~~

Up to about a million vectors, brute force on a modern CPU or GPU is often fast enough, and it never misses. Past that, approximate indexes win.

## Approximate search

ANN indexes avoid looking at most vectors. Pinecone's index guide covers the main families:

- **Clustering (IVF)**: group vectors into clusters, scan only the few clusters nearest the query.
- **Graphs (HNSW, DiskANN)**: hop through a graph of neighbours toward the query.
- **Hashing (LSH)**: hash so that nearby vectors tend to collide; mostly superseded by graphs for dense embeddings.
- **Quantization (PQ, int8, binary)**: combined with the above to cut memory ([Vector compression](lesson:aieng:02.04)).

[Indexing techniques](lesson:aieng:02.07) goes inside IVF, HNSW and DiskANN.

## Judging an index

Recall@k compares the index's answer $A$ with the exact answer $E$ for the same query:

$$
\text{recall@}k = \frac{\lvert A \cap E \rvert}{k}
$$

averaged over a few hundred real queries. Recall on its own is half a number, so report all of these together:

| Metric | Meaning |
| --- | --- |
| recall@k | Of the true top k, how many the index returned; 0.95 or more suits most RAG |
| Latency (p50 / p99) | Time per query; the tail matters because one slow shard slows every query that touches it |
| QPS | Queries per second per machine at that recall |
| Memory | Bytes per vector including index overhead |
| Build and update time | How long indexing takes, and whether inserts and deletes are cheap |

ann-benchmarks.com plots recall against QPS for many libraries. Every index traces a curve, and the query-time knob (HNSW's efSearch, IVF's nprobe) picks a point on it. An index tuned for 0.99 recall may serve half the traffic of one tuned for 0.95, so choose the cheapest point that meets your target.

## Combining with keyword search

Vector search misses exact terms such as product codes and names, so production systems often run a keyword (BM25) search beside it and merge the two ranked lists. Reciprocal rank fusion scores each document by $\sum 1/(k + \text{rank})$ over the lists it appears in, rewarding agreement without needing comparable scores. The figure shows the merge; tuning fusion, and when weighted score fusion beats RRF, is in [Hybrid search and score fusion](lesson:aieng:11.02).

~~~dg-raw
%%widget rrf Merging a keyword ranking and a vector ranking with Reciprocal Rank Fusion.
~~~

## Where this breaks

- **Ground truth with the wrong metric.** Recall measured against exact L2 search on an index built for cosine, or on unnormalized vectors, measures the wrong thing.
- **Unfiltered benchmarks.** An index that hits 0.98 recall on open queries can return almost nothing on selective filters ([Search execution](lesson:aieng:02.08)).
- **Synthetic queries.** Random vectors or documents-as-queries don't share the distribution of real user queries; recall on them is usually optimistic.
- **Recall is not relevance.** Recall@k measures how well the index reproduces exact search, not whether the results answer the question. A perfect index over a weak embedding model still returns the wrong chunks; that is measured with labelled relevance ([Evaluating RAG](lesson:aieng:10.04)).

<div class="take">
<div class="lab">Key takeaway</div>

Exact search is linear and memory-bound, fine up to about a million vectors and the ground truth beyond that. ANN indexes trade a few points of recall for huge speedups; judge them by recall@k together with tail latency, QPS and memory, on your own queries.

</div>
