---
id: "06.02"
course: "ml-system-design"
module: "06"
slug: "retrieval-and-ranking-pipelines"
title: "Retrieval and ranking pipelines"
keyIdeas: ["Large-scale recommendation and search use **multi-stage pipelines**: candidate generation from millions of items, a light ranker, a heavy ranker, then re-ranking for business rules and diversity.", "**Candidate generation** combines several sources (embedding retrieval, co-occurrence, popularity, follows, recent interactions) to get high recall cheaply.", "Each stage has a compute budget per request. Stage sizes (for example 2,000 → 500 → 100 → 20) follow from the latency and cost estimates.", "Re-ranking applies constraints the model does not learn well: diversity, freshness, deduplication, fairness of exposure and policy filters."]
question: "Design the serving pipeline for a short-video feed with 500M videos and a 150 ms budget for the whole request. Give the stages, how many items each handles, what model each uses, and where business rules apply."
answer: "Stage 1, candidate generation (about 40 ms, parallel sources): two-tower embedding retrieval over an ANN index of recent, eligible videos (1,000 candidates); videos from followed creators (300); co-engagement neighbours of the user's last 20 watched videos (300); trending in the user's region and language (200); a small exploration pool of new videos (100). Merge and deduplicate to about 1,500. Filter policy-violating, already-watched and region-blocked videos. Stage 2, light ranker (about 15 ms): a small model or gradient-boosted trees on cached features scores 1,500 to keep 300. Stage 3, heavy ranker (about 50 ms on GPU or optimised CPU): a multi-task neural model predicting watch completion, like, share, follow and report, combined into a value score, keeping 50. Stage 4, re-ranking (about 5 ms): diversity across creators and topics, freshness boost, limit on consecutive similar videos, insertion of ads or promoted content at allowed positions; return 10 to 20 for the next page. Remaining budget covers feature fetching and network."
sources: [{"title": "Covington, Adams and Sargin, Deep Neural Networks for YouTube Recommendations (2016)", "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/", "note": "Two-stage candidate generation and ranking."}, {"title": "Eksombatchai et al., Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time (2018)", "url": "https://arxiv.org/abs/1711.07601", "note": "Graph-based candidate generation."}, {"title": "Carbonell and Goldstein, The Use of MMR, Diversity-Based Reranking (1998)", "url": "https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf", "note": "Maximal marginal relevance."}]
practice: ["in-batch-softmax-logq", "ndcg-at-k", "mmr-diversity-rerank", "item-item-cf-scores", "cosine-similarity-matrix"]
tags: ["multi-stage ranking", "candidate generation"]
---

## Why stages

Scoring every item with the best model for every request is impossible at scale: 500M videos × 10,000 requests per second is far beyond any budget. Instead, systems narrow the set in stages, spending little compute per item early and more later.

~~~dg-flow A typical multi-stage pipeline. Counts and budgets are illustrative.
Corpus: 100M+ items -(candidate sources, ~40 ms)-> ~2,000 candidates -(filters)-> ~1,500 -(light ranker, ~15 ms)-> ~300 -(heavy ranker, ~50 ms)-> *~50 -(re-rank, ~5 ms)-> 10-20 shown
~~~

## Candidate generation

The first stage needs high recall, low cost and variety. Most systems merge several sources:

| Source | What it captures | Cost
| --- | --- | --- |
| Embedding retrieval (two-tower + ANN) | personalised similarity | one query embedding plus ANN search
| Item co-occurrence ("viewed together") | short-term intent from recent items | batch-precomputed neighbour lists
| Social graph | follows, friends' activity | graph lookups
| Popularity and trending | what is hot in a region or category | cheap, cached
| Recent interactions | items the user engaged with or saved | lookup
| Exploration pool | new or under-exposed items | small, sampled

Merging sources protects against any one failing and gives the ranker a diverse set to choose from. Track recall per source: if the ranker rarely picks items from a source, it may be wasting budget.

## Filtering

Before ranking, remove what must never be shown: policy violations, items the user blocked or already purchased, out-of-stock or region-restricted items. Filters are rules, not models, and they are cheap. Applying them early saves ranking compute.

## Ranking stages

The **light ranker** uses cheap features and a small model to cut thousands to hundreds; it is often distilled from the heavy ranker. The **heavy ranker** uses rich features (user history sequences, cross features, content embeddings) and multi-task heads (04.02). Its compute per item is the main cost in the pipeline, so its input size is set by the latency and cost budget.

Train each ranker on the distribution it will see. A heavy ranker trained only on items that were shown to users never learns to score the kind of candidates retrieval produces but the old ranker rejected. Logging a sample of non-shown candidates (02.01) helps.

## Re-ranking

The final step applies things that are hard to learn or that encode policy:

- **diversity**: avoid ten near-identical items; maximal marginal relevance picks items that are relevant but dissimilar to those already chosen;
- **freshness and exploration**: boost new items to meet an exposure target (03.04);
- **business rules**: ad slots, sponsored placements, legal requirements;
- **fairness of exposure**: limits on concentration per creator or seller.

MMR picks the next item $i$ to maximise $\lambda \cdot \text{rel}(i) - (1 - \lambda) \max_{j \in S} \text{sim}(i, j)$ over items not yet selected, where $S$ is the set already chosen.

<div class="ex">
<div class="lab">Worked example</div>

**a ranker blamed for a retrieval problem**

A music app's new heavy ranker gave no online gain despite strong offline results. Analysis showed that 85% of the songs users later saved were never in the candidate set: retrieval drew only from the user's top genres. Adding a co-listening source and an embedding source raised candidate recall from 15% to 45%, and the ranker's improvement then showed up online.

</div>

## Evaluating the pipeline

Measure each stage: recall of candidate generation (05.02), the light ranker's recall of the heavy ranker's top items, the heavy ranker's NDCG, and the effect of re-ranking on diversity and engagement. Then measure the whole pipeline online. A change in any stage can shift what later stages see.

## Budgets and parallelism

Candidate sources run in parallel, so their cost is the slowest source, not the sum. Feature fetching for the rankers is batched. The pipeline should degrade gracefully: if a source times out, continue with the rest; if the heavy ranker is slow, return the light ranker's order (06.04).

## Caching across stages

Parts of the pipeline can be cached. Candidate lists for a user change slowly and can be reused for a few minutes across page loads; item features and embeddings can be cached in the ranking service's memory; only the final ranking needs per-request computation. Caching candidates cuts the cost of the most expensive fan-out step and smooths latency, at the price of slightly staler results.

<div class="take">
<div class="lab">Key takeaway</div>

serve large catalogues through stages that spend more compute per item as the set shrinks: multi-source candidate generation for recall, light and heavy rankers for precision, and rule-based re-ranking for diversity, freshness and policy, each measured on its own.

</div>
