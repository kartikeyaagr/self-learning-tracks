---
id: "08.05"
course: "machine-learning"
module: "08"
slug: "ranking-metrics-and-re-ranking"
title: "Ranking metrics and re-ranking"
keyIdeas: ["Ranking metrics reward relevant items **near the top**: $\\text{DCG@}k = \\sum_{p=1}^{k} \\frac{2^{rel_p}-1}{\\log_2(p+1)}$, and NDCG divides by the DCG of the ideal order so that 1.0 means perfect.", "**MRR** averages $1/\\text{rank}$ of the first relevant item and suits single-answer tasks; **hit rate@k** only asks whether the held-out item is in the top $k$; **recall@k** is the right lens for retrieval stages.", "Accuracy metrics ignore what the list looks like as a whole. **Coverage** (share of the catalogue ever recommended) and **novelty** (mean $-\\log_2$ popularity) detect a system that only recommends hits.", "**MMR** re-ranks greedily: pick $\\arg\\max_{d} \\lambda\\,\\text{rel}(d) - (1-\\lambda)\\max_{d' \\in S}\\text{sim}(d, d')$, trading a little relevance for less redundancy.", "Ranking against a small sample of negatives (rank among 100 random items) can reverse model comparisons; rank against the full catalogue when you can."]
question: "A new ranker improves offline NDCG@10 by 4% over production but catalogue coverage of the top-10 lists drops from 22% to 6%. Your manager asks whether to ship. What would you check, and what would you do?"
answer: "The coverage collapse suggests the gain comes from recommending popular items more often, which offline labels reward because they were collected under a system that already favoured them. First check whether the NDCG gain survives a full-catalogue evaluation on a time split and whether it is concentrated in heavy users or popular items, by slicing NDCG by item popularity bucket. Check novelty and per-user intra-list diversity too. Then add a re-ranking stage (MMR or a popularity-aware penalty) and tune it so coverage recovers while most of the NDCG gain remains. Finally, decide with an online A/B test whose primary metric reflects long-term value (sessions per user, retention), with coverage and creator-side exposure as guardrails, rather than shipping on offline NDCG alone."
sources: [{"title": "Carbonell and Goldstein, The Use of MMR, Diversity-Based Reranking (1998)", "url": "https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf", "note": "The original MMR formulation; two pages of method."}, {"title": "Krichene and Rendle, On Sampled Metrics for Item Recommendation (2020)", "url": "https://research.google/pubs/on-sampled-metrics-for-item-recommendation/", "note": "Why ranking against sampled negatives can flip model comparisons, and corrected estimators."}, {"title": "Discounted cumulative gain (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Discounted_cumulative_gain", "note": "Compact definitions of DCG, IDCG and NDCG with a worked example and the two gain variants."}]
practice: ["ndcg-at-k", "hit-rate-and-mrr", "mmr-diversity-rerank", "catalog-coverage-and-novelty", "precision-recall-at-k"]
tags: ["NDCG", "diversity"]
tools: ["NumPy"]
---

## What a ranking metric must capture

A user sees a list, mostly the first few slots. A good metric for that list has three properties: it rewards relevant items, it rewards them more the higher they appear, and it is normalised so that values are comparable across users with different numbers of relevant items. System Design's "Search evaluation metrics" lesson introduces precision@k, MAP and NDCG for search; this lesson focuses on how they behave in recommendation and what else you need.

## The core metrics

Let $rel_p$ be the relevance of the item at position $p$ (binary, or graded such as 0 = not watched, 1 = watched, 2 = finished).

$$\text{DCG@}k = \sum_{p=1}^{k} \frac{2^{rel_p}-1}{\log_2(p+1)}, \qquad \text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

IDCG@k is the DCG of the best possible ordering of that user's relevant items. The log discount means position 1 has weight 1.0, position 2 has 0.63, position 10 has 0.29. The $2^{rel}-1$ gain makes a highly relevant item much more valuable than a marginal one; with binary relevance it reduces to 1.

| Metric | Question it answers | Good for | Blind spot
| --- | --- | --- | --- |
| Hit rate@k | is the held-out item in the top k? | leave-one-out evaluation | position within the top k
| Recall@k | what share of relevant items made the top k? | retrieval stages, large k | ordering
| MRR | how high is the first relevant item? | one right answer (next song) | everything after the first hit
| NDCG@k | how close to the ideal order is the list? | ranking stages, graded labels | absolute quality: 1.0 can still be a poor list if labels are poor
| Coverage | what share of the catalogue is ever shown? | detecting popularity collapse | relevance
| Novelty | how unpopular (surprising) are the shown items? | balancing discovery | relevance

```python
import numpy as np
def ndcg_at_k(ranked_rels, all_rels, k=10):
    rel = np.asarray(ranked_rels[:k], float)
    disc = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = ((2 ** rel - 1) * disc).sum()
    ideal = np.sort(np.asarray(all_rels, float))[::-1][:k]
    idcg = ((2 ** ideal - 1) * disc[:ideal.size]).sum()
    return dcg / idcg if idcg > 0 else 0.0
```

Users with no relevant items in the test window have IDCG = 0. Decide explicitly whether they are dropped or counted as 0, and report it; that choice alone can move NDCG by several points.

<div class="ex">
<div class="lab">Worked example</div>

**same hits, different scores**

A user has two relevant items. List A puts them at positions 1 and 5; list B at positions 4 and 5. Hit rate@5 = 1 for both, recall@5 = 1 for both. MRR: A = 1, B = 0.25. NDCG@5 with binary relevance: IDCG = 1 + 0.631 = 1.631; DCG(A) = 1 + 0.387 = 1.387, NDCG 0.85; DCG(B) = 0.431 + 0.387 = 0.818, NDCG 0.50. Only the position-aware metrics see that B buries the best item.

</div>

## Sampled metrics can mislead

Ranking each test item against the full catalogue costs one score per item per test user, so many papers rank the held-out item among 100 random unobserved items instead. Krichene and Rendle (2020) showed that sampled hit rate and NDCG are not consistent with their full-catalogue versions: a model that is better on the full ranking can look worse on the sampled one, because beating 100 random (mostly obscure) items is easy and the metric saturates. If you must sample, apply their correction, or at least validate the ranking of models on a full-catalogue run.

## Beyond accuracy: what the list looks like

A system can score well on NDCG by recommending the same 500 popular items to everyone, because popular items are, on average, what people interact with next. Two cheap aggregate metrics catch this:

- **Catalogue coverage**: $|\bigcup_u \text{top}_k(u)| / |\text{catalogue}|$. If it falls from 30% to 4% after a model change, the new model is a popularity list with extra steps.
- **Novelty**: average of $-\log_2 p(i)$ over recommended items, where $p(i)$ is the item's share of interactions. Higher means less obvious recommendations.
- **Intra-list diversity**: mean pairwise dissimilarity within one user's list, using embedding or category similarity.

## Re-ranking with MMR

The ranker scores items independently, so near-duplicates (five episodes of the same show, six colours of the same shirt) can fill the top slots. Maximal marginal relevance, from Carbonell and Goldstein's 1998 work on search summarisation, builds the final list greedily:

~~~dg-flow MMR re-ranking: each step picks the candidate with the best balance of its own relevance against similarity to what is already chosen.
Ranked candidates -(score: relevance minus similarity penalty)-> Pick best -(append to list)-> *Selected list -(update max-similarity)-> Pick best
~~~

$$d^* = \arg\max_{d \in C \setminus S} \Big[\lambda\,\text{rel}(d) - (1-\lambda)\max_{d' \in S}\text{sim}(d, d')\Big]$$

With $\lambda = 1$ it returns the ranker's order; with $\lambda = 0.7$ a candidate that is 0.95 similar to something already chosen needs a relevance about 0.4 higher to win the slot. Relevance and similarity must be on comparable scales (normalise ranker scores to $[0,1]$ per request). Cost is $O(k \cdot |C|)$ similarity lookups, fine for 500 candidates and $k = 20$. Determinantal point processes are a more principled alternative used at YouTube; MMR remains the default because it is easy to tune and explain.

Re-ranking always costs some offline NDCG, because offline labels reward predicting what users did under the old, less diverse system. Whether diversity helps is an online question (09.04): measure long-term engagement and return rate, not just next-click.

<div class="take">
<div class="lab">Key takeaway</div>

use recall@k for retrieval, NDCG or MRR for ranking, always alongside coverage and novelty, and treat diversity re-ranking as a product decision validated online rather than by offline accuracy.

</div>
