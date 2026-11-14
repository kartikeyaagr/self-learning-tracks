---
id: "12.08"
course: "system-design-101"
module: "12"
slug: "matching-and-recommendation-algorithms"
title: "Matching and recommendation algorithms"
keyIdeas: ["Recommendation and matching systems usually run in stages.", "**Candidate generation**: quickly narrow millions of items to hundreds, using collaborative filtering, embeddings with nearest-neighbour search, co-visitation, or simple rules (same city, popular).", "**Ranking**: a heavier model scores each candidate using many features (user history, item features, context) to predict engagement.", "**Re-ranking**: apply business rules: diversity, freshness, fairness, removing seen items.", "Splitting stages keeps latency low: the expensive model only runs on a small set.", "Two-sided matching (dating, ride-hailing) must consider both sides' preferences and constraints."]
question: "Why split recommendations into candidate generation and ranking?"
answer: "The precise ranking model, with hundreds of user, item and context features, costs around a millisecond per item, so running it on millions of items per request would take hours of compute. Candidate generation uses cheap methods prepared offline, such as ANN search over two-tower embeddings, co-occurrence maps or follow graphs, to cut the catalogue to a few hundred items in milliseconds with high recall; the expensive ranker can then order that shortlist precisely within the latency budget using fresh request-time features. The split also lets each stage optimise its own goal (recall versus top-of-list precision), lets several retrieval sources feed one ranker, and leaves a re-ranking step for diversity and business rules."
sources: [{"title": "Google: Recommendation systems course", "url": "https://developers.google.com/machine-learning/recommendation", "note": "Candidate generation, scoring, re-ranking."}, {"title": "Eugene Yan: System design for recommendations and search", "url": "https://eugeneyan.com/writing/system-design-for-discovery/", "note": "Real industry architectures."}]
---

## The problem

You have a **catalogue** of millions or billions of items (videos, products, people, jobs, restaurants), one **user** with a history and a context (time, place, device), and a few tens of milliseconds to decide which 10 items to show. The best models for this are deep networks that look at hundreds of **features** (measurable properties) of the user, the item and the context, and predict something like "probability this user watches this video for more than 30 seconds".

Those models are accurate and expensive. Running one on every item in the catalogue, for every request, is impossible. This lesson explains the architecture every large recommender uses to get around that, the **multi-stage funnel**, why it is split the way it is, and how the same ideas apply to two-sided **matching**: dating, jobs and assigning drivers to riders.

## The multi-stage funnel

A modern recommender works in stages, each cheaper per item than the next and each looking at far fewer items.

1. **Candidate generation** (also called **retrieval**): narrow the huge corpus to a manageable set. On a video platform, from **billions** of videos to **hundreds or thousands**. This must be very fast per item, so it uses simple signals and precomputed structures. Several **generators** usually run in parallel, each nominating a different kind of candidate: "similar to what you watched", "popular near you", "from creators you follow", "trending now".
2. **Scoring** (or **ranking**): a precise model scores each candidate with rich features and orders them. It can afford to be expensive because it sees only hundreds of items.
3. **Re-ranking**: apply rules and constraints the model does not handle: remove items the user disliked or has already seen (lesson 12.07), boost fresh content, enforce **diversity** (not ten videos from one channel), apply **fairness** or business rules, insert ads.

~~~dg-bars Items considered at each stage of a typical video recommender for one request. Each stage cuts the set by one or more orders of magnitude.
Catalogue | 1000000000 | 1 billion videos
Candidate generation | 1000 | about 1,000
After filtering | 500 | about 500
Ranked by model | 500 | 500 scored
Re-ranked | 50 | top 50
*Shown | 10 | 10
~~~

## Why split the stages: the cost arithmetic

Suppose the ranking model costs **1 ms of compute per item** (a deep network with a few hundred features; on a GPU much of this is batched, but the total compute is what matters).

- **Score the whole catalogue**: 10 million items × 1 ms = 10,000 CPU-seconds **per request**. At 10,000 requests per second, you would need 100 million cores. Impossible.
- **Retrieve, then rank**: an **approximate nearest neighbour** (ANN) search over 10 million item embeddings (lesson 09.13 covers vector search and HNSW) takes **a few milliseconds**. Then rank 500 candidates: 500 ms of compute, batched on a GPU in about 10 to 20 ms of wall-clock time. Total: about 30 ms.

The split puts compute where it pays off. It also lets each stage optimise a **different goal**:

- **Retrieval optimises recall.** Recall is the fraction of the truly good items that make it into the candidate set. If a great item is not retrieved, no ranker can show it. Precision (how many candidates are good) barely matters here; that is the ranker's job.
- **Ranking optimises precision at the top.** It must put the best 10 of the 500 in the right order. Metrics are things like **NDCG** (a score that rewards relevant items near the top) or click-through rate.
- **Different data.** Retrieval uses **embeddings** and indexes computed offline, perhaps daily. Ranking uses **fresh, request-time features**: time of day, device, what the user did 30 seconds ago, how many times this item was shown to them today.

A useful framing is that retrieval is **fast but coarse**, cutting millions of items to hundreds (a 99.99% reduction) and trading precision for efficiency; ranking is **slower but precise**, adding item, user and context features and bigger models because it sees so few items.

~~~dg-compare One big model versus the funnel. The funnel spends cheap compute on the many and expensive compute on the few.
[One stage]
Every item -(deep model)-> Sorted list
- 10,000 CPU seconds per request
- Cannot meet any latency budget
- One model must do everything
[*Retrieve then rank]
Every item -(ANN, few ms)-> 500 candidates -(deep model)-> Top 10
+ About 30 ms in total
+ Recall and precision tuned separately
+ Many generators feed one ranker
~~~

## Offline versus online

The same architecture can be laid out on two axes: **offline versus online** (precomputed in batch jobs versus computed per request) and **retrieval versus ranking**.

~~~dg-matrix What happens where. Heavy learning and indexing run offline; requests only embed, look up and score.
Stage | Offline (batch jobs) | Online (per request)
Retrieval | Learn item embeddings, build ANN indexes, item-to-item maps, graphs | Embed the user and context, fetch nearest neighbours and graph neighbours
Ranking | Train ranking models, load features into a feature store | Join candidates with stored features, score with the model, return the order
~~~

A **feature store** is a database of precomputed features keyed by user or item (for example "user's watch time per category over 7 days"), with a batch side used for training and a low-latency side used at serving. Using **the same feature definitions** for training and serving prevents **train-serve skew**: the model learning on one version of a feature and being served a subtly different one, which silently degrades results.

## How candidate generation works

### Embeddings and the two-tower model

An **embedding** is a vector of, say, 64 to 256 numbers that represents an item or user so that similar things have nearby vectors. The most common retrieval model is the **two-tower** model:

- An **item tower** (a neural network) turns item features into an item embedding.
- A **user tower** turns user history and context into a user embedding in the **same space**.
- Training pushes the user embedding close to the embeddings of items the user engaged with, and away from others. The score is just the **dot product** of the two vectors.

Because the towers are separate, you can embed every item **offline** and put the vectors into an ANN index. At request time you run only the user tower (one forward pass, about a millisecond) and ask the index for the nearest few hundred items.

~~~dg-flow A two-tower retrieval model: items are embedded offline into an ANN index; at request time only the user is embedded.
[Offline] Item features -(item tower)-> Item vectors -(build)-> *ANN index
[Online] User history and context -(user tower)-> User vector -(top 300 by dot product)-> *ANN index -> Candidates
~~~

### Other generators

- **Item-to-item co-occurrence**: "people who watched A also watched B", computed offline from sessions and stored as a map from each item to its top 100 neighbours. Look up the user's last few items.
- **Graph walks**: build a weighted item graph from session behaviour, generate random walks, learn item embeddings from them (word2vec-style), and store nearest neighbours in an item-to-item map. Alibaba uses this approach before ranking with a deep network.
- **Search-style retrieval**: an inverted index plus a **knowledge graph** for query expansion. DoorDash expands a search for "KFC" to fried chicken and wings, so Popeyes and Bonchon are retrieved too.
- **Rules**: subscriptions, trending in the user's region, new releases.

Real examples of the funnel:

- **Facebook search** uses a two-tower model whose document embeddings are quantised into a **Faiss** ANN index; queries are embedded online, candidates retrieved by ANN plus Boolean filters, then ranked.
- **LinkedIn talent search** uses a first-level XGBoost model to retrieve and score; the top **1,000** candidates get extra features and a second-level model, and **125** results are shown.

## Ranking and re-ranking

The ranker takes each candidate, fetches its features (item statistics, user-item interaction counts, context), and predicts one or more outcomes: click, watch time, like, share, hide. Large systems predict several and combine them in a weighted **value function**, for example `0.3 × p(click) + 0.6 × expected watch minutes + 2.0 × p(share) − 5.0 × p(hide)`. The weights encode product priorities.

Re-ranking then applies list-level logic the per-item model cannot see:

- **Diversity**: at most 3 items per channel or seller, mix categories.
- **Freshness**: guarantee a slot for new uploads so they get a chance to collect data (exploration).
- **Deduplication** across generators, and the **seen filter** (lesson 12.07).
- **Policy**: remove borderline content, enforce age restrictions.

~~~dg-timeline The latency budget of one home-page request, about 60 ms end to end.
0 ms | Request arrives, load user profile
5 ms | Run user tower, query ANN and two other generators in parallel
15 ms | Merge 600 candidates, dedupe to 520, drop watched to 480
20 ms | Fetch 200 features per candidate from the feature store
25 ms | *Ranking model scores 480 items on a GPU
40 ms | Re-rank: diversity caps, insert one fresh upload
45 ms | Hydrate titles and thumbnails from cache
60 ms | Response sent with 10 videos
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a video home page end to end**

Offline, nightly: train the two-tower model on the last 30 days of watches, embed 50 million videos (64 floats each, 256 bytes, so 12.8 GB) into an HNSW index, and refresh the feature store.

Online, one request: three generators return 300 (two-tower ANN), 200 (subscriptions) and 100 (trending) candidates. Merge and deduplicate to 520; remove videos already watched, leaving 480. The ranking model scores 480 items with 200 features each in about 15 ms on a GPU. Re-ranking caps three per channel and inserts one fresh upload. Ten videos are returned in about 60 ms.

If the ranker alone had to score all 50 million videos at 1 ms each, one request would take about 14 hours of compute.

</div>

## Matching is two-sided

Recommendation shows items to a user. **Matching** pairs up two sides who both have preferences or constraints: riders and drivers, job seekers and employers, people on a dating app. Two things change.

### Both sides must want it

On a dating app, a good recommendation for A is someone A likes **and** who likes A. Candidate generation filters by **hard constraints** on both sides (location, each person's age range, the seen filter), and ranking predicts **mutual interest**, for example `p(A likes B) × p(B likes A)`. Re-ranking also **spreads attention**: if the most popular 5% of profiles were shown to everyone, they would be flooded and everyone else ignored, which is bad for both. Systems cap how often a profile is shown per day.

### Assignment, not just ranking

For ride-hailing, each driver can take only one rider. Ranking drivers for each rider independently and giving each rider their best one (**greedy matching**) is simple but can be globally poor. The alternative is **batched matching**: collect requests for a short window (a few seconds), then solve an **assignment problem** over all open riders and available drivers in an area, minimising the total pickup time. The classic exact method is the **Hungarian algorithm**, O(n³) for n riders and drivers, fine for the few hundred in one area per batch.

<div class="ex">
<div class="lab">Worked example</div>

**greedy versus batched assignment**

Two riders and two drivers. Pickup times: R1 to D1 is 2 minutes, R1 to D2 is 3 minutes, R2 to D1 is 3 minutes, R2 to D2 is 9 minutes.

**Greedy**: R1 requests first and gets the nearest driver, D1 (2 min). R2 gets what is left, D2 (9 min). Total waiting: **11 minutes**, and R2 probably cancels.

**Batched**: wait 2 seconds, then consider both riders together. Assign R1 to D2 (3 min) and R2 to D1 (3 min). Total: **6 minutes**, and nobody waits 9. R1 waits one minute longer than greedy would have given them, which is the price of a better global result.

</div>

~~~dg-seq Batched ride matching over a 2-second window. The matcher solves one assignment per area instead of one greedy choice per request.
Rider app > Matcher : request R1
Rider app > Matcher : request R2
Matcher > Location index : available drivers near R1 and R2
Location index -> Matcher : D1, D2 with positions
Matcher > ETA service : pickup times for all pairs
ETA service -> Matcher : 2 by 2 cost matrix
Matcher > Matcher : solve assignment, total 6 min
Matcher > Driver app : offer R1 to D2, R2 to D1
~~~

The candidate generation for ride matching is the geospatial search from lesson 12.04 (drivers within a few kilometres), and the ranking is an **ETA model** that understands roads and traffic rather than straight-line distance. So the same retrieve-then-rank structure appears again, followed by a global assignment step.

## Common mistakes

- **One model for everything.** Skipping retrieval and scoring the catalogue does not scale; skipping ranking and showing ANN neighbours directly gives mediocre ordering.
- **Optimising the wrong stage metric.** Measuring retrieval by precision encourages tiny, safe candidate sets that starve the ranker. Measure retrieval by recall.
- **Train-serve skew.** Features computed one way in a batch job and another way at serving time. Share definitions through a feature store.
- **No exploration.** A system that only shows what it already knows works never learns about new items. Reserve slots for exploration.
- **Greedy matching in a two-sided market.** Locally best choices can be globally bad, as the ride example shows.

## In an interview

Draw the funnel first, with rough numbers at each stage, and justify it with the cost arithmetic. Name what runs offline (embeddings, ANN index, model training, feature computation) and online (user embedding, ANN lookup, feature fetch, scoring, re-ranking). Say that retrieval optimises recall and ranking optimises top-of-list precision. For matching problems, add the two-sided twist (mutual interest, attention spreading) and, for rides or deliveries, batched assignment over a short window.

<div class="take">
<div class="lab">Key takeaway</div>

recommendation is a funnel. Candidate generation (fast, recall-oriented, often ANN over two-tower embeddings plus co-occurrence and rule-based generators) cuts millions of items to hundreds; ranking (slow, precise, rich fresh features) orders them; re-ranking enforces diversity, freshness and rules. The split exists because the precise model is far too expensive to run on the whole catalogue. Two-sided matching adds mutual-interest scoring and, where each side can take only one partner, a global assignment step.

</div>
