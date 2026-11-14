---
id: "02.06"
course: "ai-engineering-course"
module: "02"
slug: "vector-search"
title: "Vector search"
keyIdeas: ["Vector search returns the k items nearest to a query vector.", "Exact (brute-force, flat) search compares the query with every vector: perfect recall, cost grows linearly with N.", "Approximate nearest neighbour (ANN) search skips most vectors with an index, trading a little recall for orders of magnitude less work.", "Measure with recall@k (share of true top-k returned) against latency or QPS. ann-benchmarks plots exactly this.", "Most ANN indexes have a knob (efSearch, nprobe) that moves along the recall/speed curve at query time.", "Hybrid search merges vector results with keyword (BM25) results, often with reciprocal rank fusion."]
question: "Why can't you just do exact k-NN over 50 million vectors per query, and what metric do you use to judge an approximate index?"
answer: "Exact k-NN must compute distances to all 50M vectors per query (tens of billions of operations and reading ~150 GB of floats), giving high latency and very low QPS. ANN indexes skip most vectors. They're judged by recall@k (fraction of the true top-k found) versus latency/QPS and memory, measured against exact search as ground truth."
sources: [{"title": "Pinecone: k-NN vs ANN", "url": "https://www.pinecone.io/learn/series/faiss/vector-indexes/", "note": "Flat, LSH, HNSW and IVF indexes compared on speed and recall."}, {"title": "ann-benchmarks", "url": "https://ann-benchmarks.com/", "note": "Recall vs queries-per-second for real ANN libraries."}, {"title": "Elastic: Hybrid search with reciprocal rank fusion", "url": "https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html", "note": "Merging keyword and vector results."}]
---

## The problem

Given a query vector q and a set of N stored vectors, return the k closest. That's it. The difficulty is doing it in milliseconds when N is in the millions or billions.

## Exact search

A flat index computes the distance from q to every vector and keeps the best k. Recall is perfect. Cost is N x d multiply-adds per query.

<div class="ex">
<div class="lab">Worked example</div>

**50 million vectors, 768 dims**

One query = 50M x 768, about 38 billion multiply-adds, and 150 GB of float32 data to read.

Even at high memory bandwidth that is a large fraction of a second per query on one machine, so a few QPS at best.

Exact search is fine for a few hundred thousand vectors, or as a ground truth to test an ANN index.

</div>

## Approximate search

ANN indexes avoid looking at most vectors. Pinecone's index guide covers the main families:

- **Clustering (IVF)**: group vectors into clusters, only scan the few clusters nearest the query.
- **Graphs (HNSW)**: hop through a graph of neighbours toward the query.
- **Hashing (LSH)**: hash so that nearby vectors collide.
- **Quantization (PQ)**: often combined with the above to cut memory.

The next lesson goes inside IVF and HNSW.

## Judging an index

| Metric | Meaning |
| --- | --- |
| recall@k | Of the true top k, how many did the index return |
| Latency (p50/p99) | Time per query |
| QPS | Queries per second per machine |
| Memory | Bytes per vector including index overhead |
| Build time | How long indexing takes, and whether updates are cheap |

ann-benchmarks.com plots recall against QPS for many libraries. Every index has a curve, and the query-time knob (HNSW's efSearch, IVF's nprobe) picks a point on it.

## Hybrid search

~~~dg-flow Combining keyword and vector results.
[Keyword] Query -(BM25)-> Ranked list A
[Vector] Query -(ANN)-> Ranked list B
[Fusion] List A + List B -(RRF)-> *Final ranking
~~~

Reciprocal rank fusion (RRF), as in Elasticsearch, scores each document as the sum of 1 / (k + rank) over the lists it appears in. It needs no score calibration and handles the case where vector search misses exact terms.

## Why exact search stops scaling

Exact search computes the distance from the query to every stored vector. It is perfectly accurate and simple, and its cost grows linearly. The figure estimates how long one exact query takes, limited by how fast the machine can stream vectors from memory.

~~~dg-raw
%%widget flatcost Time for one brute-force query as the collection grows.
~~~

Up to about a million vectors, brute force on a modern CPU or GPU is often fast enough, and it never misses. Past that, approximate indexes trade a little recall for orders of magnitude less work.

## Fusing keyword and vector results

Hybrid search runs a keyword search (BM25) and a vector search, then merges the two ranked lists. The scores are on different scales, so you can't just add them. Reciprocal Rank Fusion (RRF) ignores scores and uses ranks:

```
score(doc) = sum over lists of 1 / (k + rank_in_that_list)      with k = 60 by default
```

A document ranked 1st by keywords and 4th by vectors scores 1/61 + 1/64 = 0.0320. One ranked 2nd in only one list scores 1/62 = 0.0161. Agreement between the two lists beats a single high rank. Change k in the figure to see how a smaller k rewards top ranks more strongly.

~~~dg-raw
%%widget rrf Merging a keyword ranking and a vector ranking with Reciprocal Rank Fusion.
~~~

## Measuring an index

- **Recall@k**: of the true top k neighbors (from exact search), how many did the index return? Aim for 0.95 or higher for most RAG use.
- **Latency at p50 and p99**: the tail matters, because one slow shard slows every query that touches it.
- **Queries per second at that recall**: an index tuned for 0.99 recall may serve half the traffic of one tuned for 0.95.

Always report the three together. A latency number without the recall it was measured at means nothing.

<div class="take">
<div class="lab">Key takeaway</div>

exact search is linear and too slow at scale. ANN indexes trade a few points of recall for huge speedups; always judge them by recall@k against latency, on your own data.

</div>
