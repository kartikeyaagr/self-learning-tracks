---
id: "09.13"
course: "system-design-101"
module: "09"
slug: "vector-search-and-hybrid-retrieval"
title: "Vector search and hybrid retrieval"
keyIdeas: ["Vector search represents queries and documents as embeddings and finds the nearest vectors, so it matches meaning, not just words.", "Exact nearest-neighbour search is too slow at scale, so engines use approximate methods: **HNSW** (layered proximity graphs), IVF (clustering), and quantization to compress vectors.", "Keyword search (BM25) is good at exact terms, names, IDs and rare words; vector search is good at paraphrases and synonyms. Each fails where the other works.", "**Hybrid retrieval** runs both and fuses results, e.g. with reciprocal rank fusion (score = sum of 1/(k + rank))."]
question: "Why combine BM25 with vector search instead of using only one?"
answer: "Because they fail in different places. BM25 is precise for exact tokens such as product codes, names, error messages and rare terms, but misses documents that express the same idea in different words. Vector search over embeddings catches synonyms, paraphrases and natural-language questions, but blurs exact identifiers and can rank near-misses highly. Running both in parallel and merging with reciprocal rank fusion, which only uses each document's rank in each list (summing 1/(60 + rank)), rewards documents both methods find and gets the strengths of each without having to make their incompatible scores comparable. Evaluations consistently show the hybrid beats either alone."
sources: [{"title": "Pinecone: HNSW", "url": "https://www.pinecone.io/learn/series/faiss/hnsw/", "note": "How approximate nearest neighbour search works."}, {"title": "Elastic: Reciprocal rank fusion", "url": "https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion", "note": "Combining keyword and vector results."}, {"title": "Hello Interview: Vector databases", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/vector-databases", "note": "Interview view."}]
---

## Why vectors

Keyword search matches words. A query for "cheap flights to the coast" won't find a page that only says "budget airfare to the seaside", because no analysed terms overlap. Synonym lists (lesson 09.07) patch some of this by hand, but language has endless ways to say the same thing, and users increasingly type whole questions ("what shoes are good for flat feet") rather than keywords. This is the **vocabulary mismatch** problem.

**Vector search** (also called semantic or dense retrieval) addresses it by representing both queries and documents as **embeddings**: arrays of numbers produced by a trained model so that texts with similar meanings land close together in a high-dimensional space. "Budget airfare" and "cheap flights" end up near each other even though they share no letters. Typical embeddings have 128 to 1,536 dimensions; some models produce 3,072. Search becomes: **find the stored vectors nearest to the query vector**.

### Where embeddings come from

An embedding model (usually a transformer encoder) is trained on pairs of texts that should match (a question and its answer, a query and the product bought) and pairs that shouldn't, adjusting its weights so matching pairs get nearby vectors. At indexing time, you run every document (or every chunk of a long document) through the model and store the vector. At query time, you embed the query with the same model, which takes a few milliseconds on a GPU or tens of milliseconds on a CPU. Changing the model means re-embedding the whole collection, because vectors from different models aren't comparable.

### Distance measures

- **Euclidean (L2)** distance: straight-line distance; considers direction and magnitude.
- **Cosine similarity**: the cosine of the angle between vectors, ignoring length; 1 means same direction, 0 unrelated. The usual choice for text embeddings.
- **Dot product**: cosine without normalising; slightly cheaper. If vectors are normalised to length 1, dot product equals cosine, so many systems normalise once at indexing time and use dot product.
- **Hamming** distance: the count of differing bits between binary vectors; extremely fast, used with binary-quantised embeddings.

## Approximate nearest neighbours

Comparing the query with every stored vector (**exact** or brute-force search) is simple and perfectly accurate, but slow at scale.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of brute force**

100 million documents with 768-dimension float32 embeddings take 100,000,000 × 768 × 4 bytes ≈ 307 GB. A brute-force query computes 100 million dot products of 768 multiply-adds each, about 77 billion operations, and must stream all 307 GB out of RAM. Even at 50 GB/s memory bandwidth, that's about 6 seconds per query on one machine. At 1,000 queries a second, impossible. Under about 100,000 vectors, though, brute force takes a few milliseconds and is a perfectly good choice.

</div>

So systems use **approximate nearest neighbour** (ANN) indexes that examine a small fraction of the vectors and *usually* find the true nearest ones. How often they do is measured as **recall@k**: the share of the true top k that the index returned. A recall of 95% means on average 9.5 of the true 10 nearest are found. Note this is recall of the *index* compared with exact search, not relevance recall (lesson 09.09).

| Index | Idea | Trade-off |
| --- | --- | --- |
| HNSW | Multi-layer graph; sparse top layers act as express lanes | 95%+ recall at low latency; memory heavy (vectors plus links); slow to build; inserts costly |
| IVF | k-means clusters; search only the nprobe nearest clusters | Faster to build, lighter, handles inserts better; lower recall at the same latency |
| LSH | Hash functions that make similar vectors collide | Simple, fast to build, good for streams; usually lower recall than HNSW |
| Annoy | Forest of random-projection trees, memory-mapped | Loads instantly, shareable between processes; immutable, must rebuild to change |

**Quantisation** is often combined with these to save memory: **product quantisation** (PQ) splits each vector into sub-vectors and replaces each with the ID of its nearest centroid, compressing a 3 KB vector to perhaps 64 bytes. **Scalar quantisation** stores each dimension as int8 instead of float32 (4 times smaller). Search runs on compressed vectors, then the top candidates are **re-scored** with full-precision vectors to recover accuracy.

## How HNSW works

**HNSW** (Hierarchical Navigable Small World) is the most widely used ANN index: it's in Elasticsearch, OpenSearch, pgvector, Lucene, Weaviate, Qdrant and many others. It combines two ideas.

- A **skip list** is a sorted linked list with extra, sparser layers on top. A search starts in the sparse top layer, takes long jumps, then drops down a layer for shorter jumps, finishing in the full bottom list. Search takes O(log n) steps.
- A **navigable small world** (NSW) graph links each vector to some of its near neighbours plus a few longer links. **Greedy routing** (always step to the neighbour closest to the target) reaches the target region in few hops, like the "six degrees of separation" effect.

HNSW stacks NSW graphs into layers like a skip list.

~~~dg-tree HNSW layers for 1 million vectors with M = 16: each layer up holds about 1/16 of the one below, so the top layers are tiny and every vector is in layer 0. Search enters at the top and descends.
Layer 4 | ~15 nodes, longest jumps
  Layer 3 | ~240 nodes
    Layer 2 | ~3,900 nodes
      Layer 1 | ~62,000 nodes, medium jumps
        *Layer 0 | all 1M vectors, short local links
~~~

### Building

- Each new vector is assigned a random **top layer**. The probability of reaching higher layers falls off exponentially (controlled by a level multiplier mL = 1/ln(M)), so most vectors live only in layer 0, and very few reach the top.
- To insert, search from the top down to find the vector's nearest existing neighbours at each of its layers, and link it to the best **M** of them (M is the number of neighbours per node, typically 16 to 64; layer 0 often allows 2M).
- **efConstruction** controls how many candidates are explored when finding those neighbours: higher means a better graph and a slower build.

### Searching

1. Start at the entry point in the top layer.
2. Greedily hop to whichever neighbour is closest to the query until no neighbour is closer.
3. Drop to the next layer, starting from that node, and repeat.
4. In layer 0, run a wider search that keeps a candidate list of size **efSearch**, and return the best k.

~~~dg-flow HNSW search: start at the sparse top layer, greedily move toward the query, then drop a layer and repeat.
Entry point, top layer -(greedy hops)-> Closest in top layer -(descend)-> Layer below -(greedy hops)-> Layer 0 -(efSearch candidates)-> *Nearest neighbours
~~~

### The knobs and their costs

- **M**: more links means higher recall and more memory. On a standard 1-million-vector benchmark (SIFT1M, 128 dimensions), index memory rises from about 0.5 GB at M = 2 to about 5 GB at M = 512.
- **efSearch**: more candidates means higher recall and higher latency. Typical curves run from around 80% recall at about 1 ms per query to close to 100% at tens of milliseconds.
- Updates are a delete plus an insert, and deletes are soft (marked, skipped during search) until a rebuild or merge. Building HNSW over hundreds of millions of vectors can take hours. Teams often keep a small **hot index** for recent inserts beside a large **cold index**, search both, and rebuild the cold one in the background and swap.
- The whole graph needs to be in memory for good latency. That's the main cost driver.

~~~dg-bars Typical recall of an HNSW index against exact search as efSearch grows. Each step up costs latency.
efSearch 16 | 82 | ~82% recall, ~1 ms
efSearch 64 | 94 | ~94% recall, ~2 ms
efSearch 256 | 98.5 | ~98.5% recall, ~6 ms
*efSearch 1024 | 99.7 | ~99.7% recall, ~20 ms
~~~

## Filtering

Real queries add structured filters: "trail shoes under \$100 in stock, size 9". The options:

- **Post-filter**: get the top N vectors (N much bigger than K), then drop those failing the filter. Risk: a strict filter (only 1% of products match) leaves fewer than K results, or none.
- **Pre-filter**: apply the filter first, then search only the matching vectors. With brute force over the filtered set, this is exact and fine when the filtered set is small. But an ANN graph built over all vectors can't be used efficiently on an arbitrary subset.
- **Filter during traversal**: walk the HNSW graph as usual but only accept results that pass the filter, still using non-matching nodes as stepping stones. Elasticsearch's kNN `filter`, and most dedicated vector databases, do this. When the filter is very selective, engines automatically switch to brute force over the filtered set.

## Hybrid retrieval and reciprocal rank fusion

Neither method wins everywhere.

- **BM25 is precise for exact terms**: product codes ("WH-1000XM5"), names, rare jargon, error messages ("ERR_SSL_PROTOCOL_ERROR"), quoted phrases. Embeddings tend to blur these: the vector for one model number lands near other model numbers, so the exact match may not rank first.
- **Vector search handles meaning**: paraphrases, synonyms, natural-language questions, cross-lingual matches, where keyword search finds nothing.

Evaluations across many datasets consistently find that combining them beats either alone. **Hybrid search** runs both retrievals in parallel and merges the results.

~~~dg-compare Keyword retrieval versus vector retrieval. Each catches what the other misses, which is why hybrid search combines them.
[BM25 keyword]
Query -(analysed terms)-> Inverted index -> Exact matches
+ Exact codes, names, rare terms
+ Cheap, explainable, no model
- Misses paraphrases and synonyms
[Vector kNN]
Query -(embedding model)-> HNSW index -> Similar meaning
+ Paraphrases, questions, synonyms
- Blurs exact identifiers
- Memory heavy, needs a model
[*Hybrid with RRF]
Both in parallel -(fuse by rank)-> One list
+ Best of both
- Two retrievals per query
~~~

### Why not just add the scores

BM25 scores are unbounded and depend on the query (a long query gives larger scores), while cosine similarity lies between −1 and 1 and clusters in a narrow band (often 0.7 to 0.9). Adding them, or even a weighted sum, needs normalisation that is fragile and query-dependent. **Reciprocal rank fusion** (RRF) sidesteps this by using only **ranks**:

`RRF score(d) = Σ over result lists of 1 / (k + rank of d in that list)`

A document missing from a list contributes nothing from it. Elasticsearch's default `rank_constant` k is **60**, and `rank_window_size` (how many results from each list are considered) defaults to the requested size. The constant k dampens the advantage of the very top ranks, so being 1st versus 2nd matters little, but appearing in both lists matters a lot. RRF needs no training and no score normalisation, which is why it's the default.

~~~dg-flow Hybrid search with RRF: run keyword and vector retrieval in parallel, then merge by rank alone.
[Keyword] Query -(BM25)-> Ranked list A -(1 / (60 + rank))-> *RRF sum per document
[Vector] Query -(embed, kNN)-> Ranked list B -(1 / (60 + rank))-> *RRF sum per document
[Merge] *RRF sum per document -(sort by sum)-> Final ranking
~~~

<div class="ex">
<div class="lab">Worked example</div>

**fusing two lists with k = 60**

Doc A is 1st in BM25 and 5th in kNN: 1/61 + 1/65 = 0.0164 + 0.0154 = 0.0318. Doc B is 2nd in BM25 and absent from kNN: 1/62 = 0.0161. Doc C is 1st in kNN only: 1/61 = 0.0164. Doc D is 10th in both: 2 × 1/70 = 0.0286.

Order: A (0.0318), D (0.0286), C (0.0164), B (0.0161). A document both methods like beats one that only one method ranks first, and even D, which neither ranks highly, beats the single-list winners. If you need more control, use weighted RRF or a learned re-ranker (lesson 09.08) over the fused list.

</div>

After fusion, many systems apply a **cross-encoder** re-ranker to the top 50 to 100: a model that reads the query and each document together and scores relevance directly. It's far more accurate than comparing separate embeddings but too slow to run on more than a short list.

## Common mistakes

- Replacing BM25 entirely with vectors, then finding exact product codes and names no longer work.
- Embedding long documents as one vector; the meaning blurs. Split into passages (chunks) of a few hundred words.
- Forgetting the memory bill: HNSW wants vectors and graph in RAM.
- Post-filtering with a strict filter and returning empty pages.
- Adding BM25 and cosine scores directly.
- Changing the embedding model without re-embedding everything.

## In an interview

Say: "BM25 for exact terms, an HNSW kNN index on embeddings for meaning, run both in parallel, fuse with RRF, and optionally re-rank the top 50 with a cross-encoder." Mention ANN recall versus latency (efSearch), memory sizing, filtering during traversal, and that below about 100,000 vectors brute force is fine.

<div class="take">
<div class="lab">Key takeaway</div>

vector search finds meaning through nearest neighbours in embedding space, using ANN indexes such as HNSW (high recall, memory-hungry) or IVF (lighter, lower recall). BM25 catches exact terms that embeddings blur. Hybrid search runs both and merges with reciprocal rank fusion, which uses ranks (1/(60 + rank)) so incompatible scores don't need normalising.

</div>
