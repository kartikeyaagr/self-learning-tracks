---
id: "03.03"
course: "ml-system-design"
module: "03"
slug: "embeddings-and-similarity-search"
title: "Embeddings and similarity search"
keyIdeas: ["An **embedding** maps an entity (user, item, query, image, text) to a dense vector so that similarity in the vector space reflects similarity for the task.", "**Two-tower** models learn query and item embeddings separately, so item vectors can be precomputed and searched with approximate nearest neighbour (ANN) indexes.", "ANN indexes such as HNSW and IVF-PQ trade a little recall for orders of magnitude less latency and memory. Measure recall@k against exact search.", "Embeddings go stale and versions are incompatible. Re-embed and re-index when the model changes, and never compare vectors from different model versions."]
question: "You need to retrieve 500 candidate products from a catalogue of 50 million for each search query within 20 ms. Design the embedding and indexing approach, estimate the memory needed, and say how you would keep it up to date."
answer: "Train a two-tower model: a query tower (text encoder plus user and context features) and a product tower (title, category, image and attribute features), trained on query-product engagement with in-batch negatives and logQ correction for popularity. Use 128-dimensional normalised vectors and dot-product similarity. Raw float32 vectors take 50M × 128 × 4 bytes = 25.6 GB, which is large but fits in memory on a few machines; with product quantisation to 32 bytes per vector it is about 1.6 GB. Use an IVF-PQ or HNSW index sharded across machines, tuned for recall@500 of about 95% against exact search at under 15 ms. Keep it fresh by re-embedding new or changed products incrementally (append to the index every few minutes) and rebuilding the full index nightly. When the model is retrained, re-embed the whole catalogue with the new product tower and switch query and index versions together."
sources: [{"title": "Yi et al., Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (2019)", "url": "https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/", "note": "Two-tower retrieval with in-batch negatives and logQ correction."}, {"title": "Malkov and Yashunin, Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs (2016)", "url": "https://arxiv.org/abs/1603.09320", "note": "The HNSW index."}, {"title": "Johnson, Douze and Jégou, Billion-scale Similarity Search with GPUs (2017)", "url": "https://arxiv.org/abs/1702.08734", "note": "IVF and product quantisation at scale (FAISS)."}]
practice: ["cosine-similarity-matrix", "in-batch-softmax-logq", "uniform-negative-sampling", "clip-cosine-retrieval", "mmr-diversity-rerank"]
tags: ["embeddings", "approximate nearest neighbour"]
tools: ["NumPy"]
---

## Why vectors

Many systems need to find "things like this" among millions of options: products for a query, songs for a user, images similar to a photo, documents relevant to a question. Comparing raw features pairwise is too slow and does not capture meaning. An embedding model maps each entity to a vector, typically 64 to 1,024 dimensions, trained so that a similarity function (dot product or cosine) is high for pairs that should match.

The AI Engineering course's embeddings module (aieng 02) covers vector databases and index internals in depth; here the focus is design choices.

## Where embeddings come from

| Source | Example | Pros | Cons
| --- | --- | --- | --- |
| Pretrained encoders | text or image models | no task data needed | not tuned for your notion of similar
| Co-occurrence | items bought together, word2vec on sessions | cheap, captures behaviour | cold start for new items
| Two-tower models | query tower and item tower trained on engagement | task-specific, precomputable | needs lots of interaction data
| Fine-tuned encoders | pretrained model fine-tuned on your pairs | strong with modest data | more training cost

## Two-tower retrieval

A two-tower model has one network for the query side (user, context, query text) and one for the item side. Their output vectors meet only in a dot product: $s(q, i) = u_q^\top v_i$. Because the item tower does not see the query, all item vectors can be computed offline and indexed. At request time you compute one query vector and search.

Training uses positives from engagement (clicks, purchases) and negatives sampled from other items. **In-batch negatives** reuse the other items in a minibatch, which is cheap but over-samples popular items; a **logQ correction** subtracts the log of each item's sampling probability from its logit to remove that bias.

~~~dg-flow Two-tower retrieval. Item vectors are built offline; only the query is embedded per request.
Catalogue -(item tower, offline)-> Item vectors -(build)-> *ANN index
Request -(query tower, online)-> Query vector -(top-k search)-> ANN index -(500 candidates)-> Ranker
~~~

## Approximate nearest neighbour search

Exact search over 50M vectors of 128 dimensions is 6.4 billion multiply-adds per query: too slow at high QPS. ANN indexes find most of the true neighbours much faster:

- **HNSW** builds a layered proximity graph and walks it greedily. It gives high recall and low latency but uses a lot of memory.
- **IVF** clusters vectors and searches only the nearest clusters.
- **Product quantisation (PQ)** compresses each vector to a few bytes, cutting memory 10 to 100 times at some cost in accuracy.

Tune on **recall@k versus latency**: for a random sample of queries, compare the ANN results with exact search. A recall of 95% at 10 ms is a typical target.

## Memory estimates

Vectors: $N \times d \times 4$ bytes in float32. 100M items at 256 dimensions is about 100 GB, which calls for sharding or compression. HNSW adds graph links, often 1.5 to 2 times the vector memory. PQ at 64 bytes per vector cuts 100M items to about 6.4 GB.

<div class="ex">
<div class="lab">Worked example</div>

**version mismatch**

A team retrained the item tower and rolled it out by re-indexing items gradually over a day. The query tower was switched immediately. For several hours, queries from the new model were compared against item vectors from the old one, and relevance collapsed for items not yet re-indexed. Versioning the index and switching query and item towers atomically fixed it.

</div>

## Keeping embeddings fresh

New items need vectors before anyone can find them. Embed them on creation and add them to the index incrementally; rebuild the full index periodically for balance. When the model changes, re-embed everything and switch index versions in one step. Monitor index size, build time, recall on a fixed query set, and the share of results that are new items.

## Beyond retrieval

Embeddings also make good features for downstream rankers, for clustering and deduplication, and for diversity re-ranking (choose items that are relevant but not too similar to each other). They are compact, but treat them as a versioned artifact: a vector means nothing without the model that produced it.

<div class="take">
<div class="lab">Key takeaway</div>

train embeddings so that dot products reflect task similarity, use two towers so item vectors can be precomputed, search them with an ANN index tuned on recall versus latency, and version models and indexes together so vectors are never mixed.

</div>
