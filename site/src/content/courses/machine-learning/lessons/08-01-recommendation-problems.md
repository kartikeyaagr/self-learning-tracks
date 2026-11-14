---
id: "08.01"
course: "machine-learning"
module: "08"
slug: "recommendation-problems"
title: "Recommendation problems"
keyIdeas: ["A recommender estimates, for a user $u$ and item $i$ it has not seen, how much $u$ would value $i$, then shows the top few. The output is a **ranked list**, not a single prediction, so errors at the top cost far more than errors at position 500.", "**Explicit** feedback (stars, thumbs) is sparse but signed; **implicit** feedback (clicks, plays, purchases) is plentiful but only positive, and a missing entry means *unknown*, not *disliked*.", "Production systems are a **funnel**: cheap retrieval narrows millions of items to a few thousand, an expensive ranker orders a few hundred, and a re-ranker applies business rules and diversity to the final dozen.", "Evaluate offline by hiding each user's latest interactions (leave-last-out or a time split) and asking whether the model ranks them highly; random splits leak the future.", "A popularity baseline is embarrassingly strong. Every model in this module should be compared against it first."]
question: "A music service trains a next-song model on a random 80/20 split of play events and reports hit rate@10 of 0.31. After a switch to leave-last-out evaluation the number drops to 0.12. Explain the drop and say which number should drive a launch decision."
answer: "The random split leaks the future. Many held-out plays are repeats or near-repeats of songs the user also played *later* in the training portion, and popular-in-a-session songs appear on both sides of the split, so the model is rewarded for memorising history it would not have at serving time. Leave-last-out asks the deployment question: given everything up to now, predict the next interaction. The 0.12 figure is the honest one and should drive decisions, ideally checked again with a global time split that also includes users who appear only after the cutoff. It should also be compared with a popularity baseline under the same protocol, because the relevant question is the lift over the cheapest alternative, not the absolute number."
sources: [{"title": "Google ML: Recommendation systems course", "url": "https://developers.google.com/machine-learning/recommendation", "note": "Short course: candidate generation, scoring and re-ranking stages, with exercises."}, {"title": "Covington et al., Deep Neural Networks for YouTube Recommendations (2016)", "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/", "note": "The classic two-stage funnel paper; read the sections on candidate generation and the 'example age' feature."}, {"title": "Eugene Yan, System design for recommendations and search", "url": "https://eugeneyan.com/writing/system-design-for-discovery/", "note": "Practitioner overview of offline/online, retrieval/ranking splits across many companies."}]
practice: ["popularity-baseline-leave-last-out", "hit-rate-and-mrr", "precision-recall-at-k", "top-n-per-group"]
tags: ["implicit feedback", "retrieval vs ranking"]
tools: ["pandas"]
---

## What is being predicted

A recommender works on a giant, mostly empty matrix $R$ with one row per user and one column per item. A cell $r_{ui}$ holds whatever signal you have: a 4-star rating, a play count, a purchase. A catalogue of 1 million items and 10 million users has $10^{13}$ cells; a busy service might observe $10^{9}$ of them, a density of 0.01%. The task is to fill in the blanks well enough that, for each user, the items you place at the top of a list are ones they would choose.

Two consequences follow. First, the target is a **ranking**: what matters is the order of the top 10 or 20 items, so later lessons use ranking losses and ranking metrics rather than squared error. Second, almost all the matrix is unobserved, and *why* a cell is missing carries information. Users rate what they chose to watch, and they chose it because they expected to like it, so observed ratings skew high. The data is missing not at random.

## Explicit vs implicit feedback

| Property | Explicit (ratings) | Implicit (clicks, plays, buys)
| --- | --- | --- |
| Volume | tiny: most users rate almost nothing | huge: every session produces events
| Sign | positive and negative | positive only; absence is ambiguous
| Noise | rating scales drift between users | an accidental click looks like intent
| Natural loss | regression on observed cells | ranking or weighted classification over all cells

Implicit data dominates in practice because it comes for free. Its awkward feature is that a zero means "not seen or not chosen", which mixes dislike with ignorance. Methods for implicit feedback therefore either treat zeros as weak negatives with low confidence (weighted matrix factorisation, lesson 08.03) or only assert that observed items should outrank unobserved ones (BPR, also 08.03). Strength of the signal is usually encoded as *confidence*: watching 90% of a film is stronger evidence than a 5-second preview.

## Retrieval, ranking, re-ranking

No model can score 10 million items for every request inside a 100 ms budget with a rich neural network. Systems split the problem into stages that trade recall for cost.

~~~dg-flow The recommendation funnel: each stage sees fewer items and can afford a more expensive model per item.
Catalogue (10M) -(retrieval: ANN, co-visitation)-> Candidates (1-5k) -(ranker: GBDT or deep net)-> Scored (500) -(re-rank: diversity, rules)-> *Shown (20)
~~~

- **Retrieval (candidate generation)** must be fast and high-recall. Typical sources: item-item co-occurrence ("people who watched X watched Y"), embedding nearest neighbours from a two-tower model (08.04), recent and trending items. Several sources are unioned.
- **Ranking** scores each candidate with a model that can use rich cross features (user history crossed with item attributes, context, time of day). It predicts a calibrated probability of the action you care about, such as click, purchase or watch time.
- **Re-ranking** applies constraints the ranker ignores: remove already-purchased items, enforce diversity (08.05), freshness boosts, policy filters.

A retrieval stage that misses an item can never be rescued by a perfect ranker, so retrieval is judged by recall@k on a few thousand slots, the ranker by NDCG on the top few. System Design 101's "A recommendation system" lesson covers the serving architecture around this funnel; this module focuses on the models.

## Offline evaluation without leaking the future

Recommendation data is a time series of events. A random 80/20 split of interactions lets the model learn from a user's March purchase to predict their February one, which inflates every metric. Two honest protocols:

1. **Leave-last-out**: for each user, hold out the most recent interaction (or last $n$) as the test item and train on the rest.
2. **Global time split**: train on everything before a cutoff date, test on the week after. This mirrors deployment best, and also exposes cold-start users who appear only after the cutoff.

Then ask: for each test user, where does the held-out item land in the model's ranking over all items they have not interacted with? Hit rate@10 (did it appear in the top 10?) and MRR are the simplest summaries.

```python
# Popularity baseline under leave-last-out (pandas)
df = df.sort_values("ts")
test = df.groupby("user").tail(1)
train = df.drop(test.index)
top = train["item"].value_counts().index[:50]
seen = train.groupby("user")["item"].agg(set)
def recs(u):
    return [i for i in top if i not in seen.get(u, set())][:10]
hit10 = test.apply(lambda r: r.item in recs(r.user), axis=1).mean()
```

<div class="ex">
<div class="lab">Worked example</div>

**why the baseline matters**

A streaming catalogue has 20k titles. The 100 most popular titles account for 30% of plays. A popularity list scores hit rate@10 of about 0.08 under leave-last-out. A new model reports 0.11. That 38% relative lift is real, but if the model had been compared against a random baseline (hit rate@10 = 10/20000 = 0.0005) it would have looked 200 times better than it is.

</div>

## Things that go wrong

- **Feedback loops**: the system recommends popular items, they get clicked because they were shown, and the next model learns they are even more popular. Logging what was *shown* (impressions) and not just what was clicked is what lets you correct for this later.
- **Cold start**: new users and new items have no interactions. Content features (text, category, price) and popularity fallbacks cover the gap; pure collaborative methods cannot.
- **Position bias**: items at the top get clicked more regardless of quality, so clicks are a biased label for the ranker.

<div class="take">
<div class="lab">Key takeaway</div>

a recommender is a ranking problem on a sparse, biased, implicit matrix, solved as a funnel of increasingly expensive stages, and it should always be evaluated on future interactions against a popularity baseline.

</div>
