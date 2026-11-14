---
id: "08.01"
course: "machine-learning"
module: "08"
slug: "recommendation-problems"
title: "Recommendation problems"
keyIdeas: ["A recommender estimates, for a user $u$ and item $i$ it has not seen, how much $u$ would value $i$, then shows the top few. The output is a **ranked list**, not a single prediction, so errors at the top cost far more than errors at position 500.", "**Explicit** feedback (stars, thumbs) is sparse but signed; **implicit** feedback (clicks, plays, purchases) is plentiful but only positive, and a missing entry means *unknown*, not *disliked*.", "Production systems are a **funnel**: cheap retrieval narrows millions of items to a few thousand, an expensive ranker orders a few hundred, and a re-ranker applies business rules and diversity to the final dozen.", "Evaluate offline by hiding each user's latest interactions (leave-last-out or a global time split) and checking where the model ranks them, with hit rate@k and MRR. Random splits leak the future and can even pick the wrong model.", "A popularity baseline is embarrassingly strong. Every model in this module should be compared against it under the same split, and reported as a lift over it."]
question: "A music service trains a next-song model on a random 80/20 split of play events and reports hit rate@10 of 0.31. After a switch to leave-last-out evaluation the number drops to 0.12. Explain the drop and say which number should drive a launch decision, and what else you would check before launching."
answer: "The random split leaks the future. Many held-out plays are repeats or near-repeats of songs the user also played *later* in the training portion, and songs that were popular in a session or a given week appear on both sides of the split, so the model is rewarded for memorising history it would not have at serving time. Leave-last-out asks the deployment question: given everything up to now, predict the next interaction. The 0.12 figure is the honest one and should drive decisions. Before launching, three checks. First, compare against a popularity baseline (ideally both all-time and recent popularity) under the same leave-last-out protocol: if recent popularity already scores around 0.12, the model adds nothing, and the relevant number is the lift over the cheapest alternative, not the absolute value. Second, confirm with a global time split, which also removes cross-user leakage of later trends and includes users who appear only after the cutoff (cold start). Third, slice hit rate by heavy versus light users and by head versus tail songs, and look at the whole hit-rate@k curve, since a model can win at k = 50 and lose at k = 5. If it passes, the final decision belongs to an online A/B test on listening time or retention, because offline metrics only measure how well the model predicts behaviour under the old system."
sources: [{"title": "Google ML: Recommendation systems course", "url": "https://developers.google.com/machine-learning/recommendation", "note": "Short course: candidate generation, scoring and re-ranking stages, with exercises."}, {"title": "Covington et al., Deep Neural Networks for YouTube Recommendations (2016)", "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/", "note": "The classic two-stage funnel paper; read the sections on candidate generation and the 'example age' feature."}, {"title": "Eugene Yan, System design for recommendations and search", "url": "https://eugeneyan.com/writing/system-design-for-discovery/", "note": "Practitioner overview of offline/online, retrieval/ranking splits across many companies."}]
practice: ["popularity-baseline-leave-last-out", "hit-rate-and-mrr", "precision-recall-at-k", "top-n-per-group"]
tags: ["implicit feedback", "retrieval vs ranking"]
tools: ["pandas"]
---

## What you will learn

- What a recommender actually predicts: a short ranked list per user, built from a huge and almost empty user-by-item table.
- The difference between **explicit** feedback (stars) and **implicit** feedback (plays, clicks, purchases), and why a missing entry means "unknown", not "disliked".
- Why real systems work as a **funnel**: cheap retrieval, then an expensive ranker, then a rule-based re-ranker.
- How to evaluate offline without leaking the future (leave-last-out and time splits), and how to compute hit rate@k and MRR by hand and in pandas.
- Why a popularity baseline is the first thing to build, and the classic traps: feedback loops, cold start and position bias.

*Before you start:* you should be comfortable with train/validation/test splits and why leakage inflates scores (module 06), and with basic pandas `groupby`. Nothing from earlier model lessons is required; matrix notation is used lightly.

## Where this module starts

Earlier modules predicted one label per example. Here, for each user you must pick a handful of items out of thousands or millions, only the order of the top few matters, and the data is a log of what people did, mostly missing. This lesson sets up the problem, the vocabulary and an honest way to measure progress. The rest of the module builds models on top of it: neighbourhood methods that say "people who watched this also watched that" (08.02), matrix factorisation that learns hidden taste dimensions (08.03), two-tower neural retrieval for huge catalogues (08.04), and ranking metrics plus re-ranking for diversity (08.05). Each of them must beat the baseline you build here.

## The idea in one picture

A small streaming service has five users and six films. For every film a user started, the log records what percentage of it they watched:

| User | A | B | C | D | E | F
| --- | --- | --- | --- | --- | --- | --- |
| Ana | 100 | 95 | 40 | 100 | ? | ?
| Ben | 80 | 100 | 100 | ? | ? | ?
| Cai | 100 | 60 | ? | ? | 100 | ?
| Dee | 90 | 100 | 30 | ? | ? | 100
| Eli | 100 | ? | 100 | 70 | ? | ?

~~~dg-fig 08.01-matrix.svg
Left: the toy table, where every "?" is a film the user never started. Right: a corner of a realistic simulated log (2,000 users, 800 titles); with only about 2% of cells filled, the picture is mostly empty space.
~~~

This is the **interaction matrix** $R$: one row per user $u$, one column per item $i$, and a cell $r_{ui}$ holding whatever signal you have. The job of a **recommender** is to estimate the "?" cells well enough that, when Ana opens the app, the films placed at the top of her screen are ones she will actually watch.

Three things are already visible:

- **It is mostly blank in real systems.** The toy is 60% filled. A catalogue of 1 million items and 10 million users has $10^{13}$ cells; a busy service might observe $10^{9}$ of them, a density of 0.01%. The public MovieLens 25M dataset is about 0.25% filled.
- **The output is a list, not a number.** Ana has two unseen films (E and F). What matters is which one goes first. In a real app she sees maybe 20 titles out of hundreds of thousands, so a mistake at position 1 costs far more than one at position 500, which is why this module uses ranking metrics, not squared error.
- **Blanks are not random.** Ana never started E. Maybe she has never heard of it; maybe she saw the poster and decided against it. People choose what to watch, and they choose things they expect to like, so the cells you do observe are a biased sample. Statisticians call this **missing not at random**.

## Step 1: decide what signal you have

The cells can hold two very different kinds of evidence.

**Explicit feedback** is a deliberate judgement: 1 to 5 stars, thumbs up or down. **Implicit feedback** is behaviour you log anyway: plays, watch percentage, clicks, add-to-cart, purchases.

| Property | Explicit (ratings) | Implicit (clicks, plays, buys)
| --- | --- | --- |
| Volume | tiny: most users rate almost nothing | huge: every session produces events
| Sign | positive and negative | positive only; absence is ambiguous
| Noise | rating scales drift between users | an accidental click looks like intent
| Natural loss | regression on observed cells | ranking or weighted classification over all cells

Implicit data dominates because it comes for free, but an empty cell mixes dislike with ignorance. Methods for implicit feedback therefore either treat empty cells as weak negatives with low confidence (weighted matrix factorisation, lesson 08.03) or only assert that observed items should outrank unobserved ones (BPR, also 08.03).

The strength of an implicit signal is encoded as **confidence**, not as a rating: Dee's 30% of film C is weak evidence (she may have abandoned it), her 100% of F is strong evidence.

## Step 2: expect a long tail

Interaction data is extremely uneven. A few titles get most of the attention, and a long tail of titles is touched by a handful of people each.

~~~dg-fig 08.01-longtail.svg
In the simulated log, the top 10% of titles collect 42% of all plays and just 14% of titles collect half of them. A real catalogue is usually even more skewed.
~~~

So recommending what is popular is right surprisingly often (Step 5), while tail items have so few interactions that a model learning from the matrix alone knows almost nothing about them; content features (genre, text, price) earn their keep there.

## Step 3: split the work into a funnel

Suppose a rich neural network takes 0.1 ms to score one user-item pair. Scoring 10 million items would take 1,000 seconds per request, against a budget of about 100 ms. No single model can be both rich and exhaustive, so systems split the work into stages that trade recall for cost.

~~~dg-flow The recommendation funnel: each stage sees fewer items and can afford a more expensive model per item.
Catalogue (10M) -(retrieval: ANN, co-visitation)-> Candidates (1-5k) -(ranker: GBDT or deep net)-> Scored (500) -(re-rank: diversity, rules)-> *Shown (20)
~~~

- **Retrieval** (also called **candidate generation**) must be fast and high-recall. Typical sources: item-item co-occurrence ("people who watched X watched Y", 08.02), embedding nearest neighbours from a two-tower model (08.04), and recent or trending items. Several sources are unioned.
- **Ranking** scores each candidate with a model that can use rich cross features: the user's history crossed with item attributes, context, time of day. It predicts a calibrated probability of the action you care about, such as click, purchase or watch time.
- **Re-ranking** applies constraints the ranker ignores: remove already-purchased items, enforce diversity (08.05), boost fresh items, apply policy filters.

The stages are judged differently. A retrieval stage that misses an item can never be rescued by a perfect ranker, so retrieval is judged by **recall@k** over a few thousand slots: of the items the user really went on to choose, what fraction made it into the candidate set? The ranker is judged on the order of the top few (NDCG, 08.05). System Design 101's "A recommendation system" lesson covers the serving architecture around this funnel; this module focuses on the models.

## Step 4: hold out the future, not a random sample

Recommendation data is a time series of events. That changes how you must split it.

A **random split** of events feels natural and is wrong: it lets the model learn from a user's March purchase to predict their February one, so knowledge of later behaviour (including what became popular later) leaks into the score.

Two honest protocols:

1. **Leave-last-out**: for each user, hold out the most recent interaction (or last $n$) as the test item and train on everything before it.
2. **Global time split**: pick a cutoff date, train on everything before it, test on the days after. This mirrors deployment best, and it also exposes cold-start users who appear only after the cutoff.

~~~dg-fig 08.01-splits.svg
The same five users split three ways. In the random split, Ana's test film B (day 3) has two later films, C and D, in training. Leave-last-out always tests the final event. The global cutoff at day 7 tests only what happened afterwards, so Cai, whose last film was on day 7, has nothing to test.
~~~

Then, for every test user, rank all items they have not interacted with in training and see where the held-out item lands. Two summaries:

$$\text{HR@}k = \frac{1}{|U|}\sum_{u \in U} \mathbb{1}[\text{rank}_u \le k], \qquad \text{MRR} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\text{rank}_u}.$$

Here $U$ is the set of test users, $\text{rank}_u$ is the position (1 = top) of user $u$'s held-out item in the ranked list, and $\mathbb{1}[\cdot]$ is 1 when the condition holds and 0 otherwise. **Hit rate@k** asks "did the item make the top $k$?" **Mean reciprocal rank** (MRR) gives 1 point for rank 1, half a point for rank 2, a third for rank 3, so it rewards getting the item near the very top.

## Step 5: build the popularity baseline first

The **popularity baseline** gives every user the same list: the most interacted-with training items, minus anything that user already has. It needs no learning and is hard to beat; report every model as a lift over it.

<div class="ex">
<div class="lab">Worked example</div>

**leave-last-out and a popularity list, by hand**

Use the toy log, each user's films in viewing order: Ana A, B, C, D; Ben A, C, B; Cai B, A, E; Dee C, A, B, F; Eli A, D, C.

Leave-last-out holds out each user's final film: Ana D, Ben B, Cai E, Dee F, Eli C. Training keeps the rest: Ana {A, B, C}, Ben {A, C}, Cai {B, A}, Dee {C, A, B}, Eli {A, D}.

Count training plays per film: A appears for all 5 users, so 5. B: Ana, Cai, Dee, so 3. C: Ana, Ben, Dee, so 3. D: Eli only, so 1. E and F: 0. Break ties alphabetically, so the global list is A, B, C, D, E, F.

Remove each user's training films and find the held-out film. Ana: list D, E, F; D is at rank 1. Ben: B, D, E, F; B is at rank 1. Cai: C, D, E, F; E is at rank 3. Dee: D, E, F; F is at rank 3. Eli: B, C, E, F; C is at rank 2.

Hit rate@1 = 2/5 = 0.4. Hit rate@2 = (Ana, Ben, Eli)/5 = 3/5 = 0.6. Hit rate@3 = 5/5 = 1.0.

MRR = (1 + 1 + 1/3 + 1/3 + 1/2)/5 = 3.167/5 = 0.633.

Now compare with a random order. With $m$ unseen films, a random list puts the target in the top 2 with probability $2/m$. Ana and Dee have $m = 3$, the others $m = 4$, so expected hit rate@2 = (2/3 + 2/4 + 2/4 + 2/3 + 2/4)/5 = 2.833/5 = 0.567. Popularity's 0.6 is barely better. On a six-film catalogue almost any list looks good; the number only means something next to a baseline.

</div>

Notice also that Dee's F landed at rank 3 only because the E-F tie was broken alphabetically. Ties are common for count-based scores; fix a deterministic tie-break, or your metric will wobble between runs.

At scale the point is sharper. On a 20k-title catalogue where popularity scores hit rate@10 of 0.08, a model scoring 0.11 is a real 38% lift; measured against a random list (10/20,000 = 0.0005) it would look 220 times better than it is.

## Recommendation evaluation in code

The code simulates a play log (titles that trend after release and then fade, users with favourite genres) and evaluates two baselines, all-time and last-14-days popularity, under both splits.

```python
import numpy as np, pandas as pd

# 1. Simulated play log: 2,000 users, 800 titles, 100 days; titles fade after release
rng = np.random.default_rng(0)
n_users, n_items = 2000, 800
genre = rng.integers(0, 8, n_items)
appeal = rng.lognormal(0, 1.2, n_items)
release = np.where(rng.random(n_items) < 0.4, 0, rng.uniform(0, 95, n_items))
rows = []
for u in range(n_users):
    fav = rng.choice(8, 2, replace=False)                   # two favourite genres
    seen = set()
    for t in np.sort(rng.uniform(0, 100, rng.integers(5, 30))):
        w = appeal * np.exp(-np.clip(t - release, 0, None) / 25) * (release <= t)
        w = w * np.where(np.isin(genre, fav), 6.0, 1.0)
        w[list(seen)] = 0                                    # no repeat plays
        i = rng.choice(n_items, p=w / w.sum()); seen.add(i)
        rows.append((u, i, t))
df = pd.DataFrame(rows, columns=["user", "item", "ts"]).sort_values("ts")
print(len(df), f"density = {len(df) / (n_users * n_items):.4f}")
# 34891 density = 0.0218

# 2. Two ways to hold out one event per user.
def split(df, how):
    test = df.groupby("user").tail(1) if how == "last" else \
           df.groupby("user").sample(1, random_state=1)
    return df.drop(test.index), test

# 3. Score every title, hide what the user already played, find the held-out title's rank.
def evaluate(train, test, scores_for):
    seen = train.groupby("user")["item"].agg(list)
    ranks = []
    for u, i in zip(test.user, test.item):
        s = scores_for(u).astype(float)
        s[seen.get(u, [])] = -np.inf                         # never re-recommend history
        ranks.append(1 + (s > s[i]).sum())
    ranks = np.array(ranks)
    return (ranks <= 10).mean(), (1 / ranks).mean()          # hit rate@10, MRR

for how in ["random", "last"]:
    train, test = split(df, how)
    tie = np.arange(n_items) * 1e-9                          # deterministic tie-break
    alltime = np.bincount(train.item, minlength=n_items) + tie
    recent = np.bincount(train[train.ts > train.ts.max() - 14].item,
                         minlength=n_items) + 1e-4 * alltime
    for name, sc in [("all-time popular", alltime), ("popular last 14 days", recent)]:
        hr, mrr = evaluate(train, test, lambda u: sc)
        print(f"{how:6s} {name:21s} HR@10={hr:.3f}  MRR={mrr:.3f}")
# random all-time popular      HR@10=0.137  MRR=0.066
# random popular last 14 days  HR@10=0.067  MRR=0.036
# last   all-time popular      HR@10=0.126  MRR=0.049
# last   popular last 14 days  HR@10=0.231  MRR=0.129
```

Under the random split, all-time popularity wins by a factor of two and you would discard the recency idea. Under leave-last-out the verdict flips: recent popularity scores 0.231 against 0.126. The random split tests many users on old plays, where counts that include *later* plays happen to fit; the honest split asks "what will they watch next?", and people mostly watch what is new.

## Reading the result

~~~dg-fig 08.01-baselines.svg
Left: hit rate@k under leave-last-out for four scorers; random order is near zero, and recent popularity leads at every list length. Right: the same two baselines under the two splits. The protocol, not the model, decided which one looked best.
~~~

A few checks before you trust any number:

- **Compare with random order.** Random hit rate@10 here is about 10/780 = 0.013. If a model is close to that, something is broken (often a user or item id mismatch between train and test).
- **Compare with popularity under the same split.** The left panel also includes a simple "co-watched with your history" scorer (the idea behind 08.02). It reaches 0.128 under leave-last-out, no better than all-time popularity and far behind recent popularity. Without the baseline you might have shipped it.
- **Look at the whole curve.** Winning at $k = 50$ but losing at $k = 5$ suggests a retrieval source, not a final ranker.
- **Slice the result** by heavy versus light users and head versus tail items; averages hide that most models only help users with long histories.

## What goes wrong

Recommenders learn from logs they helped create, which causes failure modes other models do not have.

~~~dg-fig 08.01-feedback.svg
A small simulation of a feedback loop. Showing only the 10 most-clicked titles freezes the list at about 84% of the best achievable quality, because titles that are never shown never collect clicks. Spending 2 of 10 slots on exploration and ranking by estimated click rate climbs to about 96%.
~~~

| Problem | Symptom | Fix
| --- | --- | --- |
| Random split | offline scores high, online launch disappoints | leave-last-out or a global time split
| Feedback loop | same items everywhere; new items never take off | log impressions, rank by click *rate* not count, reserve exploration slots
| Cold start | new users get junk; new items never shown | popularity and content features as fallbacks; ask new users a few questions
| Position bias | top slots get clicks regardless of quality | randomise positions on a small slice of traffic, or model position as a feature
| Recommending history | "you bought a fridge, here are fridges" | filter already-consumed items in re-ranking (unless repeats are normal, as in music)
| No baseline | a "great" model is just popularity in disguise | report every metric as a lift over popularity

The feedback loop deserves the most attention: the system recommends popular items, they get clicked because they were shown, and the next model learns they are even more popular. Logging what was *shown* (impressions), not just what was clicked, is what lets you correct for this later.

## Going deeper

**Leave-last-out still leaks a little.** It respects each user's own timeline, but not the global one: one user's training data may contain events dated after another user's test event. Global trends (a hit released late in the period) leak across users. On data with strong trends, confirm the final choice with a global time split, and keep the evaluation window short enough to resemble how often you retrain.

**Sampled metrics can reorder models.** Some papers rank the test item against 100 random negatives instead of the whole catalogue. Krichene and Rendle (2020) showed this can flip which of two models looks better; rank against the full catalogue when you can (more in 08.05).

**Offline is a proxy for online.** Offline metrics measure whether you can predict what users did under the *old* system; a new recommender changes what they see and therefore what they do. The final judge is an online A/B test on the business metric (watch time, purchases, retention); offline evaluation decides which models deserve one. Correcting offline estimates for the old policy (inverse propensity weighting) needs logged impressions and their probabilities.

**Recall bounds the funnel.** If retrieval has recall@2000 of 0.6, then 40% of the items users will choose are invisible to the ranker, however good it is.

## Key terms

| Term | Meaning
| --- | --- |
| Interaction matrix | Users by items table of observed signals; almost all cells are empty
| Explicit feedback | Deliberate ratings such as stars; scarce but can be negative
| Implicit feedback | Logged behaviour such as plays or clicks; plentiful but positive only
| Missing not at random | Blank cells are not a random sample: people choose what they interact with
| Retrieval / candidate generation | Fast, high-recall stage that narrows the catalogue to a few thousand items
| Ranking / re-ranking | Expensive scoring of candidates, then rule and diversity adjustments
| Leave-last-out | Hold out each user's latest interaction as the test item
| Hit rate@k | Share of test users whose held-out item appears in the top $k$
| MRR | Average of 1/rank of the held-out item; rewards the very top
| Popularity baseline | Same most-popular list for everyone, minus items already seen

<div class="take">
<div class="lab">Key takeaway</div>

a recommender is a ranking problem on a sparse, biased, implicit matrix, solved as a funnel of increasingly expensive stages, and it should always be evaluated on future interactions against a popularity baseline.

</div>
