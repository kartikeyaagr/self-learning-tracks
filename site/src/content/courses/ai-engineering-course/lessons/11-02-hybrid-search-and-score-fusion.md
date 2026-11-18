---
id: "11.02"
course: "ai-engineering-course"
module: "11"
slug: "hybrid-search-and-score-fusion"
title: "Hybrid search and score fusion"
keyIdeas: ["Lexical and dense retrievers fail on different queries, so running both and merging the lists beats either alone; the hard part is that their scores are on unrelated scales.", "Reciprocal rank fusion uses only ranks: each list adds w / (k + rank). It needs no tuning to work, but k changes the answer: large k rewards agreement between lists, small k rewards being first in one.", "Convex combination normalises each list (min-max, z-score) and adds alpha times dense plus (1 - alpha) times lexical. Tuned on a few dozen labelled queries it usually beats RRF.", "Normalisation is computed on the returned top-k, so the worst returned document always gets 0 under min-max and a missing document needs an explicit rule.", "Learned sparse models such as SPLADE put expansion terms with learned weights into an inverted index, giving some of dense retrieval's recall at lexical serving cost."]
question: "You run BM25 and a dense retriever in parallel for a support-docs RAG system. Should you fuse them with RRF or a weighted score combination, and how would you choose and validate the weight?"
answer: "Start with RRF (k = 60) because it works with no labels and no normalisation, and it is robust when one retriever's score distribution shifts. Once you have a labelled set of a few hundred queries with relevant chunks, try convex combination: min-max (or z-score) normalise each list over its returned top-k, give missing documents the list minimum, fuse alpha x dense + (1 - alpha) x lexical, and grid-search alpha from 0 to 1 in steps of 0.1 on recall@k or nDCG@10. Hold out part of the set to confirm the gain, check slices (identifier queries want low alpha, natural-language questions high), and keep RRF if the tuned combination does not beat it on held-out queries. Re-tune whenever the embedding model, analyzer or chunking changes, because each shifts the score distributions."
rubric: ["Explains why raw scores cannot be added", "Describes RRF correctly including the role of k and missing documents", "Describes normalisation and convex combination with alpha", "Tunes alpha on labelled queries with a held-out check", "Considers query slices or query-dependent weighting", "Notes re-tuning when models or chunking change"]
sources: [{"title": "Cormack, Clarke and Buettcher: Reciprocal Rank Fusion (SIGIR 2009)", "url": "https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf", "note": "The original RRF paper and the k = 60 constant."}, {"title": "Bruch, Gai and Ingber: An Analysis of Fusion Functions for Hybrid Retrieval (2022)", "url": "https://arxiv.org/abs/2210.11934", "note": "Convex combination beats RRF in and out of domain; RRF is sensitive to k; few labels needed to tune alpha."}, {"title": "Formal et al.: SPLADE v2 (2021)", "url": "https://arxiv.org/abs/2109.10086", "note": "Learned sparse retrieval with expansion and a FLOPS sparsity regulariser."}, {"title": "Qdrant: Hybrid and multi-stage queries", "url": "https://qdrant.tech/documentation/concepts/hybrid-queries/", "note": "RRF, weighted RRF and distribution-based score fusion in one engine."}]
practice: ["weighted-score-fusion", "document-ranking", "ndcg-at-k", "precision-recall-at-k"]
tools: ["Elasticsearch", "OpenSearch", "Qdrant", "Weaviate", "Vespa", "SPLADE", "ELSER"]
---

## Two retrievers, two kinds of miss

BM25 ([Lexical search and BM25](lesson:aieng:11.01)) misses paraphrases; dense retrieval ([Vector search](lesson:aieng:02.06)) blurs identifiers and rare names. Their errors are only weakly correlated, so the union of their top lists has much higher recall than either. Anthropic's contextual retrieval experiments are a clean public example: on their benchmark, adding BM25 to contextual embeddings cut the top-20 retrieval failure rate from 3.7% to 2.9%. System Design covers the intuition and a first RRF example in [Vector search and hybrid retrieval](lesson:sdsd:09.13); this lesson is about the fusion maths, how to tune it and how to deploy it.

~~~dg-flow Hybrid retrieval: two lists in parallel, one fused list out, then a reranker.
[Lexical] Query -(analyzer, BM25)-> Top 100 by BM25
[Dense] Query -(embed, ANN)-> Top 100 by cosine
[Fuse] Both lists -(RRF or normalise + alpha)-> *Fused top 100 -(rerank)-> Top 5 to the prompt
~~~

The obstacle is scale. A BM25 score is unbounded and grows with query length (12 for one query, 40 for another). Cosine similarity from a modern embedding model sits in a narrow band, often 0.6 to 0.9. Adding them means the BM25 side decides everything. Fusion is the set of ways to put them on common ground.

## Rank fusion: RRF

Reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009) throws scores away and keeps positions:

$$
\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{w_r}{k + \text{rank}_r(d)}
$$

with $w_r = 1$ in the original and $k = 60$. A document absent from a list contributes nothing from it. The constant $k$ controls how fast credit decays. The ratio between rank 1 and rank 10 is $\frac{k+10}{k+1}$: 1.15 at $k = 60$ and 4 at $k = 2$. So large $k$ says "appearing in both lists matters most" and small $k$ says "being first anywhere matters most". Defaults differ: Elasticsearch's `rank_constant` is 60, Qdrant's RRF defaulted to $k = 2$ (configurable since v1.16), and the same two lists can fuse differently in each.

RRF's virtues are real: no labels, no normalisation, and it cannot be hijacked by one retriever's score outliers. Its cost is that it ignores how confident each retriever was. A BM25 top hit with a huge margin and one that barely beat the second get the same credit.

## Score fusion: normalise, then combine

Convex combination keeps the scores but first maps each list onto a common range, then mixes with one weight:

$$
s(d) = \alpha\,\tilde s_{\text{dense}}(d) + (1-\alpha)\,\tilde s_{\text{lex}}(d)
$$

Common normalisers, each computed over the top-k list a retriever returned for this query:

| Normaliser | Formula | Behaviour |
| --- | --- | --- |
| Min-max | $\frac{s - \min}{\max - \min}$ | Best returned gets 1, worst returned gets 0, whatever their absolute quality |
| z-score | $\frac{s - \mu}{\sigma}$ | Centres on the list mean; outliers stay large |
| Distribution-based (Qdrant DBSF) | maps $\mu \pm 3\sigma$ to 0 and 1 | Like z-score, rescaled to roughly 0 to 1 |
| Theoretical min-max | uses the score's known bounds | Works for cosine in $[-1, 1]$; BM25 has no upper bound |

Bruch, Gai and Ingber compared these carefully and found that a tuned convex combination beat RRF both in domain and out of domain, that RRF's quality depends on $k$ more than its reputation suggests, that the choice of normaliser mattered little once $\alpha$ was tuned, and that a small labelled sample was enough to tune it. Engines expose this directly: Elasticsearch's `linear` retriever (weights plus `minmax` or `l2_norm`), OpenSearch's normalization processor, Weaviate's relative score fusion with `alpha` (1 is pure vector), Qdrant's DBSF.

<div class="ex">
<div class="lab">Worked example</div>

**one query, two lists, three answers**

BM25 returned A 12.0, B 9.0, C 4.0, D 3.0. Dense returned C 0.84, E 0.83, B 0.80, D 0.70.

**RRF, k = 60.** C: $\frac{1}{63} + \frac{1}{61} = 0.0323$; B: $\frac{1}{62} + \frac{1}{63} = 0.0320$; D: $\frac{2}{64} = 0.03125$; A: $\frac{1}{61} = 0.0164$; E: $\frac{1}{62} = 0.0161$. Order **C, B, D, A, E**: D, fourth in both lists, beats A, the clear BM25 winner.

**RRF, k = 2.** C 0.533, B 0.450, A and D tie at 0.333 ($\frac{1}{3}$ versus $\frac{1}{6} + \frac{1}{6}$), E 0.250. A has caught up with D.

**Min-max, then $\alpha$.** Lexical becomes A 1, B 0.667, C 0.111, D 0; dense becomes C 1, E 0.929, B 0.714, D 0; missing documents get 0.

| $\alpha$ | Fused order (scores) |
| --- | --- |
| 0.3 | A 0.700, B 0.681, C 0.378, E 0.279, D 0 |
| 0.5 | B 0.690, C 0.556, A 0.500, E 0.464, D 0 |
| 0.7 | C 0.733, B 0.700, E 0.650, A 0.300, D 0 |

Two lessons. The top result depends on the method and its one parameter, so pick both on labelled data, not by taste. And min-max sends D to exactly 0 because it was the last item of both lists, even though both retrievers returned it: normalising over a short top-k makes "worst of the returned" look like "irrelevant".

</div>

## Tuning alpha on an eval set

1. Collect a few hundred real queries with labelled relevant chunks ([Evaluating RAG](lesson:aieng:10.04) covers building it). Include slices: identifier lookups, short keyword queries, full questions.
2. Retrieve the top 100 from each retriever once and cache both lists; fusion is then pure arithmetic and a grid search over $\alpha \in \{0, 0.1, \dots, 1\}$ takes seconds.
3. Score each $\alpha$ by the metric that feeds the next stage: recall@k at the reranker's depth if you rerank, nDCG@10 if the fused list goes straight to the prompt ([retrieval metrics](lesson:mlsd:05.02)).
4. Choose on one split, confirm on another, and compare against RRF on the same held-out queries.
5. Read the per-slice curves. Identifier queries usually peak at low $\alpha$ and questions at high $\alpha$. If the peaks are far apart, a cheap query classifier (does the query contain a code-like token?) that switches $\alpha$ beats any single value.

Re-tune when the embedding model, analyzer, chunking or corpus changes: each one moves a score distribution.

## One engine or two

| | One engine (both indexes in Elasticsearch, OpenSearch, Vespa, Qdrant, Weaviate, Milvus, or Postgres with pgvector plus a BM25 extension) | Two engines (a search engine plus a vector database) |
| --- | --- | --- |
| Filters and ACLs | Defined once, applied to both sides | Must be kept identical in two query languages |
| Freshness | One ingestion pipeline | Two pipelines that can drift ([Keeping a RAG index correct](lesson:aieng:11.06)) |
| Fusion | Built-in retrievers, done server side | In your service; you control every detail |
| Quality ceiling | Bounded by what the engine supports | Best-in-class on each side |
| Latency | One round trip | Parallel calls, so the slower one plus fusion |

Most teams should start with one engine; the second engine is justified only when one side is clearly too weak or too slow in the first.

## Learned sparse retrieval

SPLADE (Formal et al.) sits between the two families. A BERT-style encoder reads the text, and its masked-language-model head produces a weight for every vocabulary term (30,522 wordpieces for BERT). SPLADE v2 pools over tokens with $w_j = \max_i \log\big(1 + \text{ReLU}(z_{ij})\big)$, and a FLOPS regulariser during training pushes most weights to zero, leaving a sparse vector of typically tens to a few hundred terms per passage. Crucially, the vector includes **expansion terms** the text never used: a passage about "car insurance" gets weight on "vehicle" and "policy". The result is served from an ordinary inverted index, so it keeps lexical search's exact matching and explainability while recovering part of dense retrieval's synonym recall. Elastic's ELSER is a production learned-sparse model of this kind, and BGE-M3 emits dense, sparse and multi-vector outputs from one model.

The costs: documents must be encoded by a model (a reindex when it changes), query encoding adds a model call unless you use a document-only variant, posting lists are longer than BM25's so queries are slower, and out-of-domain vocabulary still needs plain BM25. A common strong setup is BM25 + dense, with SPLADE replacing BM25 only when eval shows it helps.

## Where this breaks

- **Window size.** Fusing the top 10 from each side means a document ranked 11th by one retriever counts as absent. Fuse over a deeper window (50 to 100) than you return.
- **Missing documents.** Decide what score a document gets from a list it is not in: 0 under RRF, the list minimum (or 0) under min-max. Leaving it at the z-score mean silently promotes it.
- **An empty list.** A query with no lexical matches gives min-max nothing to normalise; fall back to the other list rather than dividing by zero.
- **Chunk versus document ids.** If one side returns chunks and the other documents, fuse on the same unit or the same document can appear twice.
- **Filters on one side only.** A tenant filter applied to the vector query but not the BM25 query leaks documents through fusion.

## What interviewers probe

- "Why not just add the scores?" Different, query-dependent scales.
- "What does k do in RRF?" Show the rank-1 versus rank-10 ratio.
- "RRF or weighted?" RRF with no labels; tuned convex combination once you have them, validated on held-out queries.
- "How is SPLADE different from BM25 and from dense?" Learned weights plus expansion, served from an inverted index.

<div class="take">
<div class="lab">Key takeaway</div>

Run lexical and dense retrieval together and fuse. RRF is the safe default with no labels; a normalised convex combination tuned on a few hundred labelled queries usually beats it. Either way, the fusion method and its single parameter change the top results, so choose them on held-out data and re-tune when models or chunking change.

</div>
