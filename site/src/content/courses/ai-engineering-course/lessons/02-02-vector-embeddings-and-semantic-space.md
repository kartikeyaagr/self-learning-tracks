---
id: "02.02"
course: "ai-engineering-course"
module: "02"
slug: "vector-embeddings-and-semantic-space"
title: "Vector embeddings and semantic space"
keyIdeas: ["An embedding space is a geometry where distance stands for difference in meaning; directions can carry meaning, single dimensions usually do not.", "Euclidean distance, cosine similarity and dot product give the same ranking once vectors are unit length, so pipelines normalize once and use the cheapest operation.", "Retrieval is asymmetric: short queries and long documents are embedded differently, through prefixes, input types or separate encoders.", "Embedding spaces are anisotropic (unrelated texts still score high) and suffer from hubness (a few points are near everything), so raw cosine thresholds do not transfer between models.", "Embeddings capture topic well and logic, negation and exact identifiers poorly, which is why lexical and hybrid search exist."]
question: "When is cosine similarity equivalent to dot product, why do pipelines normalize vectors, and why does a fixed 'cosine above 0.8' threshold break when you switch embedding models?"
answer: "Cosine similarity is the dot product divided by the product of the two lengths, so it equals the dot product when both vectors have unit length. Then squared Euclidean distance is 2 minus twice the dot product, so all three metrics rank identically. Pipelines normalize once at write time so the index can use the cheapest operation, a dot product, and so long vectors do not win just because they are long. A fixed threshold breaks because cosine scores are not calibrated probabilities. Embedding spaces are anisotropic: vectors crowd into a narrow cone, so with one model unrelated texts may score 0.6 and paraphrases 0.9, while a more spread-out model gives 0.1 and 0.75 for the same pairs. A cut-off tuned on one model silently drops or admits results on another. Calibrate thresholds on labelled pairs for each model, or rank and take the top k instead."
rubric: ["States cosine equals dot product for unit-length vectors, and that L2 ranking then matches too.", "Gives a reason to normalize: cheapest operation with identical rankings, and no bias toward long vectors.", "Explains that cosine scores are model-specific and not probabilities, mentioning anisotropy or the narrow cone.", "Recommends calibrating thresholds per model on labelled pairs, or using top-k instead of a fixed cut-off."]
sources: [{"title": "Google ML Crash Course: Embeddings", "url": "https://developers.google.com/machine-learning/crash-course/embeddings", "note": "Embedding spaces, distance and visualisation."}, {"title": "Pinecone: Vector similarity explained", "url": "https://www.pinecone.io/learn/vector-similarity/", "note": "Euclidean, cosine and dot product compared."}, {"title": "Ethayarajh: How Contextual are Contextualized Word Representations? (2019)", "url": "https://arxiv.org/abs/1909.00512", "note": "Shows contextual embeddings occupy a narrow cone: the anisotropy behind high baseline cosines."}, {"title": "Wang et al.: Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5, 2022)", "url": "https://arxiv.org/abs/2212.03533", "note": "An asymmetric model trained with query: and passage: prefixes."}]
practice: ["cosine-similarity-matrix", "l2-vector-norm", "embedding-similarity", "clip-cosine-retrieval"]
---

## Meaning as geometry

In an embedding space every item is a point. Items with similar meaning sit near each other, and some directions line up with concepts: moving from "walk" to "walked" is roughly the same step as from "swim" to "swam". Google's crash course shows this with projections of word vectors. The individual dimensions usually have no human meaning; only distances and directions do.

## Measuring closeness

| Metric | What it measures | Range |
| --- | --- | --- |
| Euclidean (L2) | Straight-line distance | 0 to infinity, smaller is closer |
| Cosine similarity | Angle between vectors, ignores length | −1 to 1, larger is closer |
| Dot product | Angle and length together | Unbounded, larger is closer |

Cosine similarity is the dot product divided by the lengths:

$$
\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}
$$

If every vector has length 1, then $\cos(a, b) = a \cdot b$, and $\lVert a - b \rVert^2 = 2 - 2\, a \cdot b$, so all three metrics produce the same ranking. Normalizing once at write time lets the index use the fastest operation (a dot product) with no change in results. Without normalization, dot product favours long vectors: a document whose vector happens to be long can outrank one that is closer in meaning. Most embedding APIs return unit-length vectors for this reason; check yours, and use the metric the model was trained with.

The figure is a 2D slice of a semantic space with a handful of documents. Drag the query point, switch between cosine, dot product and Euclidean distance, and toggle normalization to watch the rankings converge.

~~~dg-raw
%%widget simspace Ranking documents by cosine, dot product or Euclidean distance.
~~~

## Queries are not documents

Search is asymmetric. A query is five words; the document that answers it is three hundred. "Reset password" and a help article about account recovery are not paraphrases, yet the search should match them. Models handle this in one of three ways:

- **Prefixes**: E5 was trained with "query: " on questions and "passage: " on documents, and many open models since use a prefix or a one-line task instruction on the query side.
- **Input types**: API models such as Cohere's and Voyage's take a parameter that says whether the text is a query or a document.
- **Separate encoders**: some systems train one tower for queries and another for documents, the classic two-tower design.

For symmetric tasks (deduplication, clustering, finding similar tickets) both sides use the same form. Using the document form for queries, or no prefix where one is expected, costs recall silently: nothing errors, the right documents just rank lower.

## Anisotropy and hubness

Two quirks of real embedding spaces explain most surprising scores.

**Anisotropy.** Ethayarajh (2019) showed that contextual embeddings do not spread evenly over the sphere; they crowd into a narrow cone. Every vector shares a large common component, so even unrelated texts have high cosine similarity. With many models, unrelated texts score 0.6 to 0.7 and near-duplicates above 0.9. Contrastive training reduces this, but by different amounts in different models, so the score scale is model-specific.

**Hubness.** In high-dimensional spaces a few points, called hubs, end up close to a disproportionate share of all others. In a search index a hub document appears in the top 10 for hundreds of unrelated queries: a generic "Contact us" page, or a chunk that is mostly boilerplate. A quick diagnostic is the k-occurrence count: for a sample of queries, count how often each document appears in a top-k list. With 1,000 queries, top 10 each and 10,000 documents, the average document appears once; a document appearing 300 times is a hub. Fixes include removing boilerplate before embedding, subtracting the corpus mean vector before normalizing (which also reduces anisotropy), and re-ranking.

<div class="ex">
<div class="lab">Worked example</div>

**A threshold that broke on a model upgrade (scores illustrative)**

A support bot only answers when the best match scores above 0.80. With the old model:

| Pair | Old model | New model |
| --- | --- | --- |
| "reset my password" vs "I forgot my password" | 0.91 | 0.74 |
| "reset my password" vs "change account email" | 0.78 | 0.41 |
| "reset my password" vs "pasta recipes" | 0.62 | 0.08 |

Under the old model, 0.80 separates the paraphrase from the near miss. The new model is more spread out and ranks the pairs in the same order with a wider gap, so it is the better model, but its paraphrase now scores 0.74 and the bot stops answering. Re-calibrating on 200 labelled pairs moves the threshold to about 0.60, which again separates the first row from the second.

</div>

## Where geometry fails

~~~dg-flow Same query, two kinds of search.
[Keyword] "reset my password" -(BM25)-> Docs containing "reset" + "password"
[Semantic] "reset my password" -(embed)-> *Nearby vectors -> "Recover account access" too
~~~

Semantic search wins on paraphrases and synonyms, but embeddings capture topical similarity far better than logical structure. "Flights from Delhi to Paris" and "flights from Paris to Delhi" are almost identical vectors. So are "the drug is safe in pregnancy" and "the drug is not safe in pregnancy". Exact identifiers (error code E-4012, SKU 88-1203) embed badly because the model never learned what they mean. That is why production search adds keyword search and filters: see [Lexical search and BM25](lesson:aieng:11.01) and [Hybrid search and score fusion](lesson:aieng:11.02).

<div class="take">
<div class="lab">Key takeaway</div>

Embedding spaces turn meaning into distance. Normalize vectors and use the metric the model expects, embed queries the way the model was trained to see them, and treat cosine scores as model-specific numbers to calibrate, not probabilities.

</div>
