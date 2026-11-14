---
id: "16.05"
course: "system-design-101"
module: "16"
slug: "a-recommendation-system"
title: "A recommendation system"
keyIdeas: ["A recommender has offline and online parts.", "**Offline**: train models on interaction logs, compute item and user embeddings, build nearest-neighbour indexes, precompute candidates for users.", "**Online**: on request, gather candidates (precomputed, ANN search, popular, recently viewed), fetch features from a feature store, rank with a model, re-rank for diversity and business rules.", "**Nearline**: stream recent events to update user features within minutes.", "Cold start: new users get popular or onboarding-based picks; new items use content features.", "Measure with A/B tests on engagement, not just offline accuracy."]
question: "What's computed offline vs online in a recommender?"
answer: "Offline, in batch: training the embedding (for example two-tower) and ranking models, computing embeddings for every item, building ANN indexes and item-to-item similarity tables, and computing user and item features into a feature store (sometimes whole per-user lists for the most active users). Online, per request: building the context including the current session, running the user tower to embed the user, retrieving candidates via ANN and other generators such as similar-to-last-viewed and popular-in-region, filtering out seen or unavailable items, fetching features including real-time streaming ones, scoring the few hundred candidates with the ranking model, and re-ranking for diversity, freshness and business rules. The split exists because ranking every item per request would cost thousands of CPU-seconds."
sources: [{"title": "Eugene Yan: System design for recommendations and search", "url": "https://eugeneyan.com/writing/system-design-for-discovery/", "note": "Offline/online split."}, {"title": "Google: Recommendation systems course", "url": "https://developers.google.com/machine-learning/recommendation", "note": "Fundamentals."}]
tags: ["Design"]
---

## What we are building

A recommender answers "what should this user see next?": the next video, the products under "you may also like", the posts in an explore tab. It has to pick a handful of items from a catalogue of millions, personalised to one person, in tens of milliseconds, thousands of times per second.

That combination is impossible to do naively. A good scoring model might cost a millisecond of compute per item. Scoring 10 million items for one request would take 10,000 CPU-seconds. So every production recommender splits the work along two axes (lesson 12.08 introduces them):

- **Offline versus online**: what is computed in batch ahead of time, and what is computed per request.
- **Retrieval versus ranking**: cheaply narrowing millions of items to hundreds, then carefully ordering those hundreds.

### Requirements

- Given a user (and optionally a context such as the item being viewed), return about 10 to 50 recommended items.
- p99 latency under about 100 ms for the whole call, inside a page that has its own budget.
- Catalogue of 10 million items, 100 million users, 20,000 recommendation requests per second at peak.
- Recommendations should reflect what the user did in the last few minutes, not just yesterday.
- New items should get some exposure even before anyone has interacted with them (the cold-start problem).

| | Retrieval (millions to hundreds) | Ranking (hundreds to the final list) |
| --- | --- | --- |
| **Offline** | train embedding models; embed all items; build ANN indexes; item-to-item similarity tables | train the ranking model; compute user and item features into a feature store |
| **Online** | embed the user or query; nearest-neighbour lookup; filters | fetch features for each candidate; score with the model; apply business rules |

## Key ideas first

A few terms before the design:

- An **embedding** is a vector of numbers (often 64 to 256 of them) that represents an item or a user, trained so that similar things have nearby vectors. "Nearby" is usually measured by the dot product or cosine of the angle between vectors.
- A **two-tower model** has one neural network (tower) that turns user features into a user embedding and another that turns item features into an item embedding. It is trained so that the dot product is high for user and item pairs that led to clicks or purchases. The towers are separate, so item embeddings can be computed offline for the whole catalogue and only the user tower runs per request.
- **Approximate nearest neighbour (ANN)** search finds the vectors closest to a query vector without comparing against all of them, using structures such as HNSW graphs or quantised inverted-file indexes (lesson 09.13). It might return 95% of the true top 500 in 5 ms instead of 100% in several seconds.
- A **feature store** is a database of precomputed facts (features) about users and items, readable by key in a millisecond or two, used both to train models and to serve them.

## What is computed offline

Batch jobs, running hourly or daily over logs of views, clicks, purchases, ratings and dwell time:

- **Item embeddings**, for example from a two-tower model, or from word2vec-style training on user sessions, where items viewed together in a session are treated like words that appear together in a sentence. Alibaba builds a graph of items from session co-occurrence, generates random walks over it, and trains embeddings on those walks.
- **The ANN index** over those embeddings. Facebook's search, for example, quantises embeddings into a Faiss index so hundreds of millions fit in memory. Alternatively, a precomputed **"similar items" table**: for every item, its top 100 neighbours, stored in a key-value store.
- **The ranking model**, trained on past impressions labelled with what happened (clicked, bought, watched 80%, skipped).
- **Features**: per user (favourite categories, 30-day purchase count, average price paid), per item (click-through rate, price, age, average rating), and per user and item pair (has this user bought from this seller before). These go into the feature store.
- Sometimes **whole recommendation lists** per user, written to a key-value store. Very cheap to serve, but stale, blind to the current session, and wasted on the majority of users who will not visit today.

<div class="ex">
<div class="lab">Worked example</div>

**offline job sizes**

10 million items with 128-dimensional float32 embeddings: 10,000,000 x 128 x 4 bytes = about 5 GB. That fits in the memory of a single large server, so the ANN index can be replicated to every retrieval node.

With product quantisation down to 32 bytes per item, the same index is 320 MB, small enough to keep many versions (one per region or experiment) side by side.

100 million users x 50 features x 8 bytes = 40 GB of user features, which is a modest sharded key-value cluster.

A daily training run over 30 days of logs (say 20 billion impressions) might take a few hours on a GPU cluster. That is why it is offline: no request could wait for it.

</div>

## What is computed online

Per request, within a budget of tens of milliseconds:

1. **Build the request context**: user ID, time of day, device, location, and **recent actions** in the current session. The three products the user just viewed are the strongest signal you have, and they were not known when last night's batch ran.
2. **Embed the request**: run the user tower on those features, which takes about a millisecond.
3. **Retrieve candidates** from several generators in parallel: ANN neighbours of the user vector, items similar to the last item viewed, items popular in the user's region today, new uploads from creators the user follows, and a slot for brand-new items. Each returns 100 to 500 IDs; merge and deduplicate to about 1,000.
4. **Filter**: already seen or bought, out of stock, not allowed in this country, blocked seller. Bloom filters or exact sets of seen IDs (lesson 12.07) make this cheap.
5. **Rank**: fetch features for each candidate from the feature store in one batch call (plus **real-time features** such as "clicked this category 2 minutes ago"), and score with the ranking model, often a gradient-boosted tree or a deeper neural network that predicts click or purchase probability.
6. **Re-rank** with business rules: diversity (not 10 items from one seller), freshness boosts, promotions, fairness to new creators, and removing near-duplicates.

~~~dg-flow The recommender: offline jobs build what the online path looks up.
[Offline] Interaction logs -(daily training)-> Embedding model -(embed all items)-> ANN index and features
[Online] Request -(embed user + session)-> ANN retrieval -(1,000 candidates)-> *Ranking model -(top 50)-> Re-rank to 10
~~~

~~~dg-seq One request against a 60 ms budget. Retrieval generators run in parallel, ranking runs once on the merged set.
App > Recs API : recommendations for user 42
Recs API > Feature store : user features + session
Recs API > Retrieval : ANN, similar-to-last, popular
Retrieval -> Recs API : 1,000 candidate IDs (8 ms)
Recs API > Recs API : filter seen and out of stock
Recs API > Feature store : features for 800 items (5 ms)
Recs API > Ranker : score 800 candidates
Ranker -> Recs API : scores (15 ms)
Recs API -> App : 10 items after re-rank
~~~

## Why the split works

The funnel is the core idea. Each stage is cheaper per item than the next, and each sees far fewer items.

~~~dg-bars Items considered at each stage of the funnel for one request.
Catalogue | 10000000 | 10,000,000
Retrieved | 1000 | 1,000
After filters | 800 | 800
*Ranked top | 50 | 50
Shown | 10 | 10
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the cost of skipping retrieval**

Ranking with a model that costs 1 ms of CPU per item over 10 million items is 10,000 CPU-seconds per request. At 20,000 requests per second you would need 200 million CPU cores. Clearly impossible.

With retrieval: ANN over 10 million vectors takes about 5 ms on one core and returns 1,000 candidates. Ranking 800 survivors in a batch on a GPU or vectorised CPU costs perhaps 15 ms of wall time and a fraction of a core-second. At 20,000 requests per second, that is a few thousand cores. Retrieval is fast but coarse (it discards 99.99% of items); ranking is slow but precise.

</div>

Real systems often add more stages. LinkedIn's talent search is a concrete multi-stage example: a first, lightweight model retrieves and scores candidates, the top **1,000** get richer features and a second, heavier model, and **125** are shown. YouTube's design is the same shape: candidate generation from millions to hundreds, then a ranker over hundreds. The general pipeline is often named **candidate generation, scoring, re-ranking**.

## Freshness and consistency

### Train-serve skew

If a feature is computed one way for training (a Spark job computing "purchases in the last 30 days" over the data warehouse) and another way at serving time (a service counting purchases in a cache), small differences in definitions, time zones or missing data mean the model sees different inputs live than it learned on, and quality drops silently. Using **one feature store** with one feature definition for both training and serving, and logging the exact features used at serving time for the next training run, are the standard defences.

### Cold start

New items have no interactions, so a collaborative embedding cannot place them. Options: **content-based embeddings** computed from text, images or category, so a new red dress lands near other red dresses on day one; a dedicated **"new items" generator** that reserves a few slots per page for exploration; and bandit-style exploration that tests new items on a small share of traffic. New users get popular-in-region items, or a short onboarding quiz, until their history builds up.

### Streaming features

Daily batches leave a gap of up to 24 hours. A stream processor (Flink or Kafka Streams) reading click events updates short-window features such as "items viewed in the last hour" or "category click counts the current session" in the feature store within seconds. The online path reads them like any other feature.

~~~dg-compare What the same user sees with batch-only features versus batch plus streaming features.
[Batch only]
Yesterday's features -> Ranker
- Ignores the hiking boots just viewed
+ Simple, cheap pipeline
[*Batch plus streaming]
Yesterday + last 5 minutes -> Ranker
+ Reacts within seconds
- Stream jobs to run and monitor
~~~

## Failure handling and degradation

A recommender is rarely a hard dependency: a page should still render if it is slow. Build in fallbacks:

- If ranking times out, return candidates in retrieval order.
- If retrieval fails, return a cached list of popular items for the region.
- If the feature store is slow, use default feature values rather than waiting.
- Precompute a small fallback list per user for the top 1% of most active users, so the most valuable traffic never sees a blank shelf.
Set a strict timeout per stage and measure how often each fallback fires.

## Measuring it

Offline metrics (recall at 500 for retrieval, AUC or NDCG for ranking, lesson 09.09) tell you whether a new model is plausible. Only an online **A/B test** tells you whether it is better: click-through rate, purchases per session, long-term retention. Offline and online results disagree surprisingly often, usually because of feedback loops: the model is trained on what the old model chose to show.

## Common mistakes

- Trying to score the whole catalogue per request.
- Precomputing everything, so the recommender ignores what the user did a minute ago.
- Different feature code for training and serving.
- No plan for new items or new users.
- Making recommendations a hard dependency of the page.
- Judging a model only by offline metrics.

## In an interview

Draw the two-by-two table (offline and online, retrieval and ranking) early; it organises everything. Give the funnel numbers to justify retrieval. Then pick one deep dive: the two-tower model and ANN, the feature store and skew, or real-time session features. Ask: what is the objective (clicks, purchases, watch time, long-term satisfaction), how large is the catalogue, and how fast do new items arrive?

<div class="take">
<div class="lab">Key takeaway</div>

offline, train models, embed every item, build ANN indexes and compute features. Online, embed the user and current session, retrieve about a thousand candidates from several generators, filter, fetch fresh features, rank with the model and re-rank for business rules. Retrieval is cheap and coarse, ranking is expensive and precise, streaming features close the freshness gap, and a shared feature store keeps training and serving consistent.

</div>
