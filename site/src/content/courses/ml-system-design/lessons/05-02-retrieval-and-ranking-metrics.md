---
id: "05.02"
course: "ml-system-design"
module: "05"
slug: "retrieval-and-ranking-metrics"
title: "Retrieval and ranking metrics"
keyIdeas: ["Retrieval stages are judged by **recall@k**: does the candidate set contain the items the final ranker or the user would choose?", "Ranking stages are judged by position-aware metrics such as **NDCG@k**, **MRR** and **MAP**, which reward putting relevant items near the top.", "Logged engagement is biased by what was shown and where. Offline metrics on logs favour the current model; correct with position-bias models, randomised data or counterfactual estimators.", "Track **beyond-accuracy** metrics too: coverage, diversity, novelty and freshness. A ranker can win NDCG while narrowing what users see."]
question: "A new video ranker improves offline NDCG@10 on last week's logged sessions by 6%, but an A/B test shows no change in watch time. Give three reasons the offline gain might not transfer and how you would change the offline evaluation."
answer: "(1) Exposure bias: logs only contain engagement on videos the old ranker showed, so the new ranker is rewarded for re-ranking the old model's choices and is never credited or penalised for videos it would surface that were never shown. (2) Position bias: clicks at the top of the old list partly reflect position, so learning to copy the old ordering looks good offline. (3) Metric mismatch: NDCG on clicks is not watch time; the new ranker may promote clickable but short videos. Also possible: the retrieval stage limits what the ranker can change, or the offline set over-represents heavy users. Fixes: evaluate with position-debiased labels (inverse propensity weighting from a small randomised slot experiment), use watch-time-weighted gains in NDCG, collect a small amount of randomised exposure data for counterfactual evaluation, check recall of the retrieval stage separately, and report results by user segment."
sources: [{"title": "Wikipedia, Discounted cumulative gain", "url": "https://en.wikipedia.org/wiki/Discounted_cumulative_gain", "note": "NDCG and its variants, citing Järvelin and Kekäläinen (2002)."}, {"title": "Joachims, Swaminathan and Schnabel, Unbiased Learning-to-Rank with Biased Feedback (2016)", "url": "https://arxiv.org/abs/1608.04468", "note": "Correcting position bias with propensity weighting."}, {"title": "Wikipedia, Recommender system (evaluation and beyond-accuracy measures)", "url": "https://en.wikipedia.org/wiki/Recommender_system", "note": "Diversity, novelty, serendipity and coverage, with references."}]
practice: ["ndcg-at-k", "hit-rate-and-mrr", "precision-recall-at-k", "catalog-coverage-and-novelty", "mmr-diversity-rerank"]
tags: ["ranking metrics", "offline evaluation"]
tools: ["NumPy"]
---

## Two stages, two kinds of metric

Large ranking systems retrieve a few hundred candidates from millions, then rank them (04.02). Each stage needs its own metric.

**Retrieval** is judged by recall: of the items the user eventually engaged with (or that the full ranker would put on top), how many are in the candidate set?
$$\text{recall@}k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}$$
Retrieval with low recall caps the whole system; no ranker can recover an item that was never retrieved.

**Ranking** is judged by how high relevant items appear:

| Metric | Definition | Good for
| --- | --- | --- |
| Precision@k | share of the top k that is relevant | fixed-size lists
| Hit rate@k | did any relevant item appear in the top k? | next-item prediction
| MRR | average of 1/rank of the first relevant item | one right answer (search, autocomplete)
| MAP | mean of precision at each relevant position | several relevant items
| NDCG@k | discounted gain, normalised by the ideal ordering | graded relevance (purchase > click)

NDCG uses $\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$, divided by the best possible DCG for that list. It handles graded relevance and rewards the top positions most.

~~~dg-flow Evaluate each stage with its own metric, then the system end to end.
Retrieval -(recall@500)-> Ranking -(NDCG@10)-> Re-ranking -(diversity, coverage)-> *Online metric: engagement, retention
~~~

## Logged data is biased

Offline ranking evaluation usually replays logged sessions: the items shown, and what the user clicked. Two biases distort it:

- **Exposure bias**: you only see feedback on items the old system showed. A new ranker that would surface different items gets no credit for them.
- **Position bias**: users click top positions more regardless of relevance, so logs over-reward whatever the old ranker put first.

Corrections:

- estimate how likely each position is to be examined (from a small experiment that randomises order among top results) and weight clicks by the inverse of that propensity;
- reserve a little traffic for randomised exposure, giving unbiased data for evaluation;
- use counterfactual estimators (inverse propensity scoring) when you log the probability each item was shown.

## Human relevance judgements

For search, a fixed set of queries with graded human labels gives a stable benchmark unaffected by the current ranker. It is expensive but valuable as a regression test: every new model must not lose NDCG on it.

## Beyond accuracy

| Metric | What it catches
| --- | --- |
| Catalogue coverage | share of items ever recommended; a narrowing system shows the same few items
| Diversity | dissimilarity within a list; lists of near-duplicates feel repetitive
| Novelty | how unpopular the recommended items are; too low means only hits
| Freshness | share of new content; low values starve new creators
| Fairness of exposure | exposure by provider group compared with their relevance

These are often guardrails in A/B tests: the new ranker must improve engagement without dropping coverage or new-creator exposure below a threshold.

<div class="ex">
<div class="lab">Worked example</div>

**a better ranker that shrank the catalogue**

An online shop's new ranker raised offline NDCG by 5% and online conversion by 1%. Three months later, sellers complained that sales had concentrated in fewer listings. Coverage of the top 10 lists had dropped from 30% to 9% of the catalogue. The team added a diversity re-ranking step (maximal marginal relevance) and a coverage guardrail, recovering most coverage at a 0.2% conversion cost.

</div>

## Choosing k

Pick $k$ to match the product: the number of results above the fold, the slots on a home screen, or the length of an autocomplete list. A metric at $k = 100$ for a product that shows five items measures the wrong thing.

## Offline-online agreement

Keep a log of past launches with offline metric changes and the online result. If offline NDCG gains of 2% have reliably produced engagement gains, trust the offline metric for screening. If not, fix the offline setup before relying on it.

## Metrics per query type

Search and recommendation traffic mixes very different requests: navigational queries with one right answer, broad exploratory ones, and long-tail queries seen once. Report ranking metrics for each type. A model that helps head queries but hurts the long tail can look neutral overall while frustrating the users who search hardest.

<div class="take">
<div class="lab">Key takeaway</div>

judge retrieval by recall@k and ranking by position-aware metrics such as NDCG and MRR, correct logged data for exposure and position bias, keep a human-labelled regression set, and guard coverage, diversity and freshness alongside accuracy.

</div>
