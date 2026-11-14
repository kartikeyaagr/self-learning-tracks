---
id: "02.02"
course: "ai-engineering-course"
module: "02"
slug: "vector-embeddings-and-semantic-space"
title: "Vector embeddings and semantic space"
keyIdeas: ["An embedding space is a geometry where distance stands for difference in meaning.", "Directions can carry meaning (gender, tense, topic) but individual dimensions are usually not interpretable.", "Three common metrics: Euclidean (L2) distance, cosine similarity (angle only) and dot product (angle and length).", "If all vectors are normalized to length 1, cosine and dot product give identical rankings, and L2 gives the same order too.", "Use the metric the model was trained with. Most text embedding models expect cosine or normalized dot product.", "Semantic search finds paraphrases that keyword search misses, but can miss exact terms like product codes. That is why hybrid search exists."]
question: "When is cosine similarity equivalent to dot product, and why do most embedding pipelines normalize vectors?"
answer: "Cosine similarity equals dot product when both vectors have unit length, since cosine is the dot product divided by the product of norms. Pipelines normalize so the cheapest operation (dot product) gives cosine rankings, L2 ordering also matches, and vector length no longer biases results."
sources: [{"title": "Pinecone: What are vector embeddings", "url": "https://www.pinecone.io/learn/vector-embeddings/", "note": "Intro to embeddings and the geometry of meaning."}, {"title": "Google ML Crash Course: Embeddings", "url": "https://developers.google.com/machine-learning/crash-course/embeddings", "note": "Embedding spaces, distance and visualisation."}, {"title": "Pinecone: Vector similarity explained", "url": "https://www.pinecone.io/learn/vector-similarity/", "note": "Euclidean, cosine and dot product compared."}]
---

## Meaning as geometry

In an embedding space, every item is a point. Items with similar meaning sit near each other, and some directions line up with concepts: moving from "walk" to "walked" is roughly the same step as from "swim" to "swam". Google's crash course shows this with projections of word vectors. The individual dimensions usually have no human meaning; only distances and directions do.

## Measuring closeness

| Metric | What it measures | Range |
| --- | --- | --- |
| Euclidean (L2) | Straight-line distance | 0 to infinity, smaller is closer |
| Cosine similarity | Angle between vectors, ignores length | -1 to 1, larger is closer |
| Dot product | Angle and length together | Unbounded, larger is closer |

Pinecone's guide gives the rule of thumb: use the metric the embedding model was trained with.

## Why normalize

If every vector has length 1, then:

- cosine(a, b) = a . b, so cosine and dot product are identical;
- ||a - b||^2 = 2 - 2 (a . b), so L2 ranking is the same too.
Normalizing once at write time lets the index use the fastest operation (a dot product) with no change in results. Leaving vectors unnormalized under dot product means longer vectors win more often, which can bias results toward certain documents.

## What semantic space is good and bad at

~~~dg-flow Same query, two kinds of search.
[Keyword] "reset my password" -(BM25)-> Docs containing "reset" + "password"
[Semantic] "reset my password" -(embed)-> *Nearby vectors -> "Recover account access" too
~~~

Semantic search wins on paraphrases and synonyms. It loses on exact identifiers (error code E1234, SKU numbers, names), where a keyword index is sharper. Production systems often run both and merge results (hybrid search), covered later in the module.

## Drag a query through the space

The figure is a 2D slice of a semantic space with a handful of documents. Drag the query point and watch the ranking change. Switch the metric between cosine, dot product and Euclidean distance, and toggle normalization.

~~~dg-raw
%%widget simspace Ranking documents by cosine, dot product or Euclidean distance.
~~~

Two things to notice:

- With **normalized vectors**, all three metrics give the same order. Cosine similarity, dot product and Euclidean distance are then just different scales of the same angle.
- Without normalization, **dot product favors long vectors**. A document whose vector happens to be long can outrank a closer-in-meaning document. Most embedding APIs return unit-length vectors for exactly this reason; check yours before choosing a metric.

## How close is close?

Cosine scores are not probabilities, and their range depends on the model. With many modern models, unrelated texts still score 0.6 to 0.7, and near-duplicates score above 0.9. That makes a fixed cut-off like "only use results above 0.8" fragile:

| Pair (illustrative, one model) | Cosine |
| --- | --- |
| "reset my password" vs "I forgot my password" | 0.91 |
| "reset my password" vs "change account email" | 0.78 |
| "reset my password" vs "pasta recipes" | 0.62 |

The numbers shift when you switch models, so calibrate thresholds on your own labelled pairs and re-check them after any model change.

## Where geometry fails

Embeddings capture topical similarity well and logical structure poorly. "Flights from Delhi to Paris" and "flights from Paris to Delhi" are almost identical vectors. So are "the drug is safe in pregnancy" and "the drug is not safe in pregnancy". Exact identifiers (error code E-4012, SKU 88-1203) also embed badly, because the model has never learned what they mean. That is why production search combines vectors with keyword search and filters, covered in 02.06.

<div class="take">
<div class="lab">Key takeaway</div>

embedding spaces turn meaning into distance. Normalize vectors and use cosine or dot product as the model expects, and remember that semantic similarity is not exact match.

</div>
