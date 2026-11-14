---
id: "08.05"
course: "machine-learning"
module: "08"
slug: "ranking-metrics-and-re-ranking"
title: "Ranking metrics and re-ranking"
keyIdeas: ["Ranking metrics reward relevant items **near the top**: $\\text{DCG@}k = \\sum_{p=1}^{k} \\frac{2^{rel_p}-1}{\\log_2(p+1)}$, and NDCG divides by the DCG of the ideal order so that 1.0 means perfect. State whether you use exponential or linear gain.", "**MRR** averages $1/\\text{rank}$ of the first relevant item and suits single-answer tasks; **hit rate@k** only asks whether the held-out item is in the top $k$; **recall@k** with large $k$ is the right lens for retrieval stages.", "Accuracy metrics ignore what the lists look like across users. **Coverage** (share of the catalogue ever recommended) and **novelty** (mean $-\\log_2$ popularity) detect a system that only recommends hits.", "**MMR** re-ranks greedily: pick $\\arg\\max_{d} \\lambda\\,\\text{rel}(d) - (1-\\lambda)\\max_{d' \\in S}\\text{sim}(d, d')$, trading a little relevance for less redundancy. Tune $\\lambda$ on the relevance-diversity curve and confirm online.", "Ranking against a small sample of negatives (the held-out item among 100 random items) can reverse model comparisons; rank against the full catalogue when you can."]
question: "A new ranker improves offline NDCG@10 by 4% over production, but catalogue coverage of the top-10 lists drops from 22% to 6% and mean novelty falls from 9.1 to 7.4 bits. The offline numbers were computed by ranking each held-out item against 100 sampled negatives. Your manager asks whether to ship. What would you check, and what would you do?"
answer: "Start by doubting the 4%. It was measured with sampled negatives, which saturate and can reverse model comparisons, so recompute NDCG@10 (and recall@100 for the retrieval side) against the full catalogue on a time split, and bootstrap over users to get an interval for the difference. Check that IDCG uses all relevant items and that empty users are handled the same way for both models.\n\nThen explain the coverage and novelty collapse. Together they say the new model recommends popular items much more often. Offline labels reward that, because they were collected under a system that already favoured popular items and popular items are what people interact with next on average. Slice NDCG by item popularity bucket and by user activity: if the gain is concentrated on head items and heavy users, it is likely a popularity effect rather than better personalisation. Check intra-list diversity per user as well.\n\nIf a real gain survives, add a re-ranking stage (MMR on item embeddings, or a popularity-aware penalty) and sweep $\\lambda$ to trace the NDCG-versus-coverage curve; choose a point near the bend where coverage recovers most of the way while most of the NDCG gain remains. Finally decide with an online A/B test (09.04) whose primary metric reflects long-term value (sessions per user, retention, return rate), with coverage, novelty and creator-side exposure as guardrails, rather than shipping on offline NDCG alone."
sources: [{"title": "Carbonell and Goldstein, The Use of MMR, Diversity-Based Reranking (1998)", "url": "https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf", "note": "The original MMR formulation; two pages of method."}, {"title": "Krichene and Rendle, On Sampled Metrics for Item Recommendation (2020)", "url": "https://research.google/pubs/on-sampled-metrics-for-item-recommendation/", "note": "Why ranking against sampled negatives can flip model comparisons, and corrected estimators."}, {"title": "Discounted cumulative gain (Wikipedia)", "url": "https://en.wikipedia.org/wiki/Discounted_cumulative_gain", "note": "Compact definitions of DCG, IDCG and NDCG with a worked example and the two gain variants."}]
practice: ["ndcg-at-k", "hit-rate-and-mrr", "mmr-diversity-rerank", "catalog-coverage-and-novelty", "precision-recall-at-k"]
tags: ["NDCG", "diversity"]
tools: ["NumPy"]
---

## What you will learn

- Why a recommender needs **ranking metrics** rather than accuracy or squared error, and what a good one must reward.
- How to compute hit rate, precision, recall, MRR and NDCG at a cutoff $k$ by hand, and which question each one answers.
- Why ranking the held-out item against 100 random items instead of the whole catalogue can reverse which model looks better.
- How coverage, novelty and intra-list diversity catch a system that only recommends hits, which accuracy metrics cannot see.
- How **maximal marginal relevance (MMR)** re-ranks a list to remove near-duplicates, and how to choose its trade-off knob $\lambda$.

*Before you start:* you should know the retrieval-then-ranking funnel and implicit feedback (08.01), and what a retrieval model such as item-item CF (08.02), matrix factorisation (08.03) or a two-tower model (08.04) outputs: a scored list of candidate items per user. Logarithms base 2 appear throughout; $\log_2 4 = 2$ and $\log_2 8 = 3$ is all you need.

The previous lessons built models that *produce* ranked lists. This last lesson of the module is about *judging* those lists, and about adjusting the final list so it works as a whole. System Design's "Search evaluation metrics" lesson covers the search side; here the focus is recommendation data.

## The idea in one picture

A user sees a list, and mostly looks at the first few slots. Suppose the user later engaged with two items, and two models produced these top-5 lists:

| Position | 1 | 2 | 3 | 4 | 5
| --- | --- | --- | --- | --- | --- |
| List A | hit | - | - | - | hit
| List B | - | - | - | hit | hit

A metric that only counts hits calls them equal, yet List A is clearly better: the user meets something good immediately.

~~~dg-fig 08.05-lists.svg
Same two hits, very different experience. Hit rate and recall cannot tell the lists apart; MRR and NDCG, which care about position, score List A much higher.
~~~

A **ranking metric** turns one list into one number. A good one has three properties:

- It rewards relevant items. An item is **relevant** if the user engaged with it in the held-out test data.
- It rewards them *more the higher they appear*, because attention falls off quickly down a list.
- It is **normalised**, so a user with 2 relevant items and a user with 20 produce scores on the same 0-to-1 scale and can be averaged.

Almost every metric is computed "at $k$" (@$k$): only the top $k$ positions count, because that is what the screen shows.

## Step 1: decide what counts as relevant

1. **Split by time.** Train on interactions before a cutoff date and test on the ones after it. A random split leaks the future (08.01).
2. **Choose the relevance label** $rel_p$ for the item shown at position $p$. It can be **binary** (1 = the user interacted, 0 = not) or **graded**, for example 0 = not watched, 1 = started, 2 = finished, 3 = finished and rated highly.
3. **Remove what the user already had.** Items seen in training are usually filtered out of the list, otherwise a model can score well by re-recommending the past.

A common shortcut is **leave-one-out**: hold out each user's last interaction, rank items for that user, and ask where the single held-out item landed.

## Step 2: count the hits in the top k

The simplest metrics ignore order inside the top $k$ and just count.

**Hit rate@k**: is the held-out item anywhere in the top $k$? It is 1 or 0 per user, then averaged over users. It is the natural metric for leave-one-out.

**Precision@k** and **recall@k** handle several relevant items. Let $\text{hits}$ be the number of relevant items in the top $k$, and $|R_u|$ the number of relevant items user $u$ has in the test window:

$$\text{precision@}k = \frac{\text{hits}}{k}, \qquad \text{recall@}k = \frac{\text{hits}}{|R_u|}.$$

For List A above, hits = 2, $k = 5$ and $|R_u| = 2$, so precision@5 = 0.4 and recall@5 = 1.0. List B gets exactly the same values.

Recall@k with a large $k$ (100 to 1000) is the right lens for a **retrieval stage**: its only job is to get the good items into the candidate pool, and the ranker reorders them later.

## Step 3: reward the first hit for being early (MRR)

**Reciprocal rank** looks only at the position of the *first* relevant item and scores it $1/\text{rank}$. A first hit at position 1 scores 1, at position 2 scores 0.5, at position 4 scores 0.25, and no hit in the list scores 0. **Mean reciprocal rank (MRR)** averages it over the test users $U$:

$$\text{MRR} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\text{rank}_u}.$$

List A has its first hit at 1 (score 1.0); List B at 4 (score 0.25).

MRR suits tasks with one right answer, such as "the next song" or "the product they bought". Its blind spot is everything after the first hit: a list with hits at positions 1, 2 and 3 scores the same as one with a single hit at position 1.

## Step 4: weigh every hit by its position (DCG and NDCG)

We want every relevant item to count, with a weight that falls as it moves down. **Discounted cumulative gain** adds up a *gain* for each position, divided by a *discount* that grows with the position:

$$\text{DCG@}k = \sum_{p=1}^{k} \frac{2^{rel_p}-1}{\log_2(p+1)}.$$

- $p$ is the position (1 = top) and $rel_p$ the relevance grade of the item shown there.
- The numerator $2^{rel_p}-1$ is the **gain**. It is 0 for an irrelevant item, 1 for grade 1, 3 for grade 2, 7 for grade 3: a highly relevant item is worth much more than a marginal one. With binary labels it is just 1 per hit.
- The denominator $\log_2(p+1)$ is the **discount**. Position 1 has weight $1/\log_2 2 = 1.0$, position 2 has 0.63, position 3 has 0.50, position 10 has 0.29.

~~~dg-fig 08.05-discount.svg
Left: the NDCG weight of a hit drops fast over the first few positions and then flattens, so rank 15 versus 20 barely matters; MRR's 1/p falls much more steeply. Right: exponential gain makes grade 3 worth seven times grade 1, linear gain only three times.
~~~

DCG on its own is not comparable across users: a user with five relevant items can reach a much higher DCG than a user with one. So divide by the best DCG that user could possibly get. **IDCG@k** (ideal DCG) is the DCG of the perfect ordering: the user's relevant items sorted from highest grade to lowest, placed at the top. Then

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k},$$

which is 1.0 for a perfect list and 0 for a list with no hits. Check on the opening example with binary labels: IDCG = $1 + 0.631 = 1.631$. List A has hits at 1 and 5, DCG = $1 + 0.387 = 1.387$, NDCG = 0.85. List B has hits at 4 and 5, DCG = $0.431 + 0.387 = 0.818$, NDCG = 0.50. Only the position-aware metrics see that B buries the best item.

<div class="ex">
<div class="lab">Worked example</div>

**NDCG@5 with graded relevance**

A user has three relevant items with grades 3, 2 and 1. The model's top 5 shows items with grades $(2, 0, 3, 0, 1)$.

Discounts for positions 1 to 5: $1/\log_2(p+1) = (1.000, 0.631, 0.500, 0.431, 0.387)$.

Gains $2^{rel}-1$: $(3, 0, 7, 0, 1)$. Gain times discount: $(3.000, 0, 3.500, 0, 0.387)$. Sum: DCG@5 $= 6.887$.

Ideal order puts grades $(3, 2, 1)$ at the top: gains $(7, 3, 1)$. IDCG@5 $= 7 \times 1 + 3 \times 0.631 + 1 \times 0.5 = 7 + 1.893 + 0.5 = 9.393$.

NDCG@5 $= 6.887 / 9.393 = 0.733$. The grade-3 item sitting at position 3 instead of 1 costs most of the gap: it loses half of its gain of 7.

With linear gain ($rel$ instead of $2^{rel}-1$) the same list scores 0.816. Both are "NDCG"; always say which gain you used.

</div>

## Step 5: average over users, and decide about empty users

Every metric above is computed per user and then averaged, so each user counts equally however active they are.

Users with no relevant items in the test window have IDCG = 0, so NDCG is 0/0. Decide explicitly whether they are dropped or counted as 0, and report it; that choice alone can move NDCG by several points.

## Ranking metrics in code

The functions are short enough to write yourself, which guarantees you know the conventions.

```python
import numpy as np
from sklearn.metrics import ndcg_score

def dcg(rels, k):
    rels = np.asarray(rels, float)[:k]
    disc = 1.0 / np.log2(np.arange(2, rels.size + 2))       # 1, 0.631, 0.5, ...
    return ((2 ** rels - 1) * disc).sum()

def ndcg_at_k(ranked_rels, all_rels, k=10):
    ideal = dcg(sorted(all_rels, reverse=True), k)
    return dcg(ranked_rels, k) / ideal if ideal > 0 else 0.0

def reciprocal_rank(ranked_items, relevant):
    for p, item in enumerate(ranked_items, start=1):
        if item in relevant:
            return 1.0 / p
    return 0.0

def precision_recall_at_k(ranked_items, relevant, k=10):
    hits = len(set(ranked_items[:k]) & set(relevant))
    return hits / k, hits / len(relevant)

print(round(ndcg_at_k([2, 0, 3, 0, 1], [3, 2, 1], k=5), 3))   # 0.733 (the worked example)
ranked = ["a", "x", "y", "z", "b"]
print(reciprocal_rank(ranked, {"a", "b"}))                  # 1.0
print(precision_recall_at_k(ranked, {"a", "b"}, k=5))       # (0.4, 1.0)

# scikit-learn uses LINEAR gain, so the same list gives a different number
y_true  = np.array([[2, 0, 3, 0, 1]])      # true grade of each candidate
y_score = np.array([[5, 4, 3, 2, 1]])      # model scores (they define the order)
print(round(ndcg_score(y_true, y_score, k=5), 3))           # 0.816
```

`ndcg_at_k` needs *all* of the user's relevant grades, not just those that made the list, otherwise IDCG is too small and every model looks near-perfect. `ndcg_score` takes scores, not a ranked list, and averages over ties.

## Sampled metrics can mislead

Scoring every item in the catalogue for every test user costs one model call per item per user, so many papers rank the held-out item among 100 randomly sampled unobserved items instead, and report hit rate or NDCG on that 101-item list. Krichene and Rendle (2020) showed that these sampled metrics are not consistent with their full-catalogue versions: the model that is better on the full ranking can look worse on the sampled one.

A simulation shows why. Model A puts the held-out item in the true top 10 for 35% of users and loses it somewhere random for the rest. Model B never gets it into the top 10, but always places it around rank 250 out of 20,000.

~~~dg-fig 08.05-sampled.svg
Full catalogue: model A wins, 0.35 against 0.00. Against 100 sampled negatives, model B scores 1.00 and looks far better, because rank 250 of 20,000 beats almost every random item.
~~~

For B, only $250/20{,}000 = 1.25\%$ of items rank above the held-out one, so among 100 random items about 1.25 are expected to beat it: it lands near position 2 almost every time. Random items are mostly obscure and easy to beat, so the sampled metric saturates. If you must sample, use their corrected estimators, or at least confirm the ranking of your candidate models with one full-catalogue run.

## Beyond accuracy: what the list looks like as a whole

A system can score well on NDCG by recommending the same few hundred popular items to everyone, because popular items are, on average, what people interact with next. Accuracy metrics never ask what the system does across users; three cheap aggregate metrics do.

- **Catalogue coverage**: the share of the catalogue that appears in at least one user's top $k$, $|\bigcup_u \text{top}_k(u)| \,/\, |\text{catalogue}|$. If it falls from 30% to 4% after a model change, the new model is a popularity list with extra steps.
- **Novelty**: the average of $-\log_2 p(i)$ over recommended items, where $p(i)$ is item $i$'s share of all interactions. An item with 2% of interactions has novelty $-\log_2 0.02 = 5.6$ bits; one with 0.01% has 13.3 bits. Higher means less obvious recommendations.
- **Intra-list diversity**: the mean pairwise dissimilarity ($1 - \text{similarity}$) between the items in one user's list, using embedding or category similarity, then averaged over users.

~~~dg-fig 08.05-coverage.svg
Left: in a long-tail catalogue the top 50 of 10,000 items take 46% of interactions. Right: the same popularity list for everyone covers 0.1% of the catalogue; a popularity-leaning model about 14%; a personalised model about 45%, with novelty rising in the same order.
~~~

None is a goal on its own (a random recommender maximises all three); they are guardrails that tell you *how* an accuracy gain was achieved.

## Re-ranking with MMR

The ranker scores each candidate independently, so near-duplicates (five episodes of the same show, six colours of the same shirt) can fill the top slots. **Re-ranking** is a final cheap stage that reorders the top few hundred candidates with a list-level objective.

**Maximal marginal relevance (MMR)**, from Carbonell and Goldstein's 1998 work on search and summarisation, builds the final list greedily, one slot at a time:

~~~dg-flow MMR re-ranking: each step picks the candidate with the best balance of its own relevance against similarity to what is already chosen.
Ranked candidates -(score: relevance minus similarity penalty)-> Pick best -(append to list)-> *Selected list -(update max-similarity)-> Pick best
~~~

At each step it picks

$$d^* = \arg\max_{d \in C \setminus S} \Big[\lambda\,\text{rel}(d) - (1-\lambda)\max_{d' \in S}\text{sim}(d, d')\Big].$$

- $C$ is the candidate set and $S$ the items already selected; $C \setminus S$ is "candidates not yet picked".
- $\text{rel}(d)$ is the ranker's score for candidate $d$, normalised to $[0, 1]$ per request.
- $\text{sim}(d, d')$ is a similarity in $[0, 1]$, often the cosine of item embeddings or 1 for same-series items. The $\max$ finds the already-chosen item that $d$ most resembles: its redundancy.
- $\lambda \in [0, 1]$ is the knob: 1 keeps the ranker's order, 0 ignores relevance.

With $\lambda = 0.7$, a candidate that is 0.95 similar to something already chosen takes a penalty of $0.3 \times 0.95 = 0.285$, which needs a relevance about $0.285 / 0.7 \approx 0.4$ higher than a completely different candidate to win the slot.

<div class="ex">
<div class="lab">Worked example</div>

**MMR by hand, three slots, $\lambda = 0.7$**

Candidates with relevance: A1 0.95, A2 0.90, A3 0.85 (three episodes of show A), B1 0.70, C1 0.60. Similarities: A-episodes to each other 0.90 to 0.95 (A1-A2 0.95, A1-A3 0.90, A2-A3 0.92); any A to B1 0.20; any A to C1 0.10; B1-C1 0.30.

Slot 1: nothing is chosen yet, so the penalty is 0 and the score is $0.7 \times \text{rel}$. A1 wins with $0.7 \times 0.95 = 0.665$.

Slot 2: A2: $0.7 \times 0.90 - 0.3 \times 0.95 = 0.630 - 0.285 = 0.345$. A3: $0.595 - 0.3 \times 0.90 = 0.325$. B1: $0.490 - 0.3 \times 0.20 = 0.430$. C1: $0.420 - 0.3 \times 0.10 = 0.390$. B1 wins.

Slot 3: A2's closest chosen item is still A1 (0.95), so it stays at 0.345; A3 stays at 0.325; C1's closest is now B1 (0.30): $0.420 - 0.090 = 0.330$. A2 wins narrowly.

Final list: A1, B1, A2 instead of the ranker's A1, A2, A3. MMR did not ban the second episode; it delayed it by one slot to make room for something different.

</div>

On a larger simulated candidate set, the ranker's top 10 all come from one dense cluster; MMR with $\lambda = 0.6$ picks 7 from it and one from each other cluster, keeping 91% of the summed relevance.

~~~dg-fig 08.05-mmr.svg
Left: circles mark the ranker's top 10, all in one cluster; crosses mark MMR's picks, spread across all four. Right: lowering λ raises diversity; relevance falls slowly at first and then steeply, so the useful λ sits near the bend.
~~~

Tune on the right panel's curve. Going from $\lambda = 1$ to 0.6 lifts diversity from 0.17 to 0.62 while giving up 9% of relevance; going on to $\lambda = 0$ buys a little more diversity for another 20 points of relevance. MMR costs $O(k \cdot |C|)$ similarity lookups, trivial for 500 candidates and $k = 20$.

## Reading the results

A model comparison you can trust has these pieces:

- **The right metric for the stage.** Recall@100 to @1000 for retrieval, NDCG@10 or MRR for the ranker, all on a time split against the full catalogue.
- **A baseline.** Always report the popularity recommender. If your model beats it by less than the noise, nothing has been learned yet.
- **Uncertainty.** Bootstrap over users (resample users, recompute) to get an interval for the *difference* between two models; a 0.3-point NDCG gain on 2,000 users is often noise.
- **Slices.** Break NDCG down by user activity and item popularity; a gain concentrated on head items is a warning sign.
- **Guardrails.** Coverage, novelty and diversity next to every accuracy number, so you see how a gain was made.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Sampled negatives (rank among 100 random items) | model ordering differs from full-catalogue results; metrics near 1 | rank against the full catalogue, or use corrected estimators
| IDCG built only from items that made the list | NDCG close to 1 for every model | compute IDCG from all of the user's relevant items
| Mixed gain conventions | your NDCG differs from a paper's or library's | state linear or exponential gain; `ndcg_score` is linear
| Silent handling of users with no relevant items | NDCG shifts by points between runs | decide drop vs count-as-0 and report it
| Random instead of time split | offline gains that vanish online | train before a cutoff, test after it
| Shipping on offline NDCG alone after re-ranking | re-ranker "loses" offline yet users engage more | judge diversity changes online (09.04)

Re-ranking almost always costs some offline NDCG, because offline labels were collected under the old, less diverse system. Whether diversity helps is an online question: measure long-term engagement and return rate, not just the next click.

## Going deeper

**Why sampled metrics break.** If the held-out item has full rank $r$ in a catalogue of $N$, the number of sampled negatives above it is Binomial$(m, (r-1)/(N-1))$ for $m$ samples. Hit@10 on the sample is then a smooth, nearly flat function of $r$ until $r$ reaches a few percent of the catalogue, so with $N = 20{,}000$ it cannot distinguish rank 1 from rank 200. Krichene and Rendle's correction re-weights the sampled rank to estimate the full-catalogue metric; it reduces bias at the cost of variance.

**Position bias in the labels.** Logged clicks favour the old ranking's top slots, so offline metrics partly reward copying it. Inverse propensity weighting divides each click by the estimated chance its position was examined, removing that bias at the cost of variance.

**Beyond MMR.** Determinantal point processes score a whole set at once: the probability of a set is proportional to the determinant of a kernel matrix built from relevance and similarity, which shrinks when items are alike. YouTube reported using them for feed diversification. **Calibration** (Steck, 2018) re-ranks so that the genre mix of a list matches the mix of the user's history, which handles "70% documentaries, 30% comedy" users better than generic diversity. MMR remains the default because it is easy to tune and explain.

**Metric choice is a product decision.** The discount $1/\log_2(p+1)$ is a convention; a carousel showing 4 items may justify NDCG@4. Pick the cutoff from how your interface is used.

## Key terms

| Term | Meaning
| --- | --- |
| @k | Only the top $k$ positions of the list are scored
| Hit rate@k | Share of users whose held-out item appears in the top $k$
| Precision@k / recall@k | Hits divided by $k$ / hits divided by the user's number of relevant items
| MRR | Average of $1/\text{rank}$ of each user's first relevant item
| DCG / IDCG | Sum of discounted gains of the list / the same for the perfect order
| NDCG@k | DCG divided by IDCG; 1.0 is a perfect ordering
| Catalogue coverage | Share of all items that appear in at least one user's list
| Novelty | Mean $-\log_2$ popularity share of recommended items
| Intra-list diversity | Mean pairwise dissimilarity inside one list
| MMR | Greedy re-ranking that trades relevance against similarity to already-picked items

<div class="take">
<div class="lab">Key takeaway</div>

use recall@k for retrieval and NDCG or MRR for ranking, computed against the full catalogue on a time split, always alongside coverage and novelty; treat diversity re-ranking such as MMR as a product decision tuned on the relevance-diversity curve and validated online rather than by offline accuracy.

</div>
