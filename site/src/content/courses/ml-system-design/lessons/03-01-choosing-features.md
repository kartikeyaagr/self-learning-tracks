---
id: "03.01"
course: "ml-system-design"
module: "03"
slug: "choosing-features"
title: "Choosing features"
keyIdeas: ["Group candidate features by entity (user, item, context, user-item interaction) and by freshness (static, batch-updated, real-time). The grid makes gaps obvious.", "Interaction and aggregate features, such as a user's click rate on a category over 7 days, usually carry most of the signal in ranking and risk models.", "Every feature has a cost: computing it, storing it, fetching it within the latency budget, keeping it consistent, and the privacy risk. Add features whose gain pays for that.", "High-cardinality IDs can be encoded by hashing, target statistics computed out of fold, or learned embeddings. Each has its own failure mode."]
question: "You are building a model to predict whether a user will book a hotel they view. List features in four groups (user, item, context, user-item), say which must be fresh within minutes, and name one feature you would leave out and why."
answer: "User: account age, past bookings in the last year, average nightly price booked, typical trip length, loyalty tier, device. Item (hotel): star rating, review score and count, price for the dates, distance to city centre, cancellation policy, booking rate over the last 30 days. Context: search destination and dates, days until check-in, number of guests, time of day, rank position, filters applied. User-item: has the user viewed or booked this hotel before, price relative to the user's usual price, distance from the user's previously booked hotels. Must be fresh: price and availability for the dates (change minute to minute), the user's activity in the current session, and recent views. Leave out the user's precise home address: privacy risk, little extra signal over coarse region and trip distance, and it is hard to justify. Also avoid any feature only known after booking, such as \"booking confirmation email opened\"."
sources: [{"title": "Google, Rules of Machine Learning, Rules 16-22 (feature engineering)", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Practical advice on adding and pruning features."}, {"title": "Google, Machine Learning Crash Course: categorical data", "url": "https://developers.google.com/machine-learning/crash-course/categorical-data", "note": "Encoding, hashing and embedding categorical features."}, {"title": "scikit-learn, Preprocessing data", "url": "https://scikit-learn.org/stable/modules/preprocessing.html", "note": "Scaling, encoding and imputation in code."}]
practice: ["categorical-encoding", "feature-scaling", "missing-value-imputation", "permutation-feature-importance", "sql-running-total"]
tags: ["features", "feature engineering"]
tools: ["pandas", "SQL"]
---

## Where signal comes from

Most of a model's quality comes from its features. A plain logistic regression with well-chosen features often beats a deep model with poor ones. The design question is which features to compute, how fresh they need to be, and whether each is worth its cost. The Machine Learning course's "Feature engineering" module (mlcore 07) covers transformations such as scaling, encoding and imputation; this lesson is about choosing features at the system level.

## A grid of candidates

Organise ideas along two axes: which entity the feature describes, and how often it changes.

| | Static | Batch (hours to days) | Real-time (seconds to minutes)
| --- | --- | --- | --- |
| User | country, signup date | purchases last 30 days, preferred categories | items viewed this session
| Item | category, brand, price tier | rating, sales last 7 days | current stock, live price
| Context | | | time of day, device, query, page position
| User-item | | user's past rating of the brand | clicked this item earlier in the session

Filling the grid for a new problem gives a long list quickly and shows gaps. Real-time features are the most expensive to build, so ask whether a batch version would capture most of the signal.

## Aggregates and interactions

The strongest features in ranking and risk models are usually aggregates over windows:

- counts and rates: user's click-through rate on this category over 7 and 30 days;
- recency: time since the user last bought from this shop;
- velocity: number of transactions on this card in the last hour;
- ratios: this price divided by the user's median price.

Use several windows (1 hour, 1 day, 30 days) so the model can separate habits from bursts. Compute them as of the prediction time (01.03) and smooth small counts, for example with a prior, so a user with one view and one click does not get a 100% click rate.

```sql
-- 7-day category click-through per user, as of each impression time
SELECT i.impression_id,
       SUM(c.clicked) * 1.0 / NULLIF(COUNT(c.event_id), 0) AS user_cat_ctr_7d
FROM impressions i
LEFT JOIN events c
  ON c.user_id = i.user_id AND c.category = i.category
 AND c.ts >= i.ts - INTERVAL '7 days' AND c.ts < i.ts
GROUP BY i.impression_id;
```

## High-cardinality categories

User IDs, item IDs, merchants and query strings have millions of values.

- **One-hot** is impossible at that scale.
- **Hashing** maps values into a fixed number of buckets. It is cheap and handles new values, but collisions merge unrelated values.
- **Target statistics** (mean label per value) are powerful but leak unless computed out of fold or from earlier time windows.
- **Learned embeddings** let a neural model learn a dense vector per value. They need enough data per value and a strategy for unseen values (03.04).

## The cost of a feature

| Cost | Question
| --- | --- |
| Compute | how much work to produce it, and how often?
| Serving latency | can it be fetched within budget at p99?
| Consistency | is it computed the same way offline and online?
| Maintenance | which upstream team owns the data, and will it change?
| Privacy and fairness | is it allowed, and could it act as a proxy for a protected attribute?
| Coverage | what fraction of requests have it, and what happens when it is missing?

A feature that adds 0.2% AUC but needs a new streaming pipeline and a synchronous call to another team's service may not be worth it. Measure each feature's marginal value with ablations or permutation importance, and remove features that do not pay their way. Unused features still cost money and still break.

~~~dg-flow A feature's life. Most of the work is after the first experiment shows it helps.
Idea -(offline ablation)-> Proven gain -(build pipeline + online version)-> *Production feature -(monitor coverage + drift)-> Keep or retire
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a feature that hurt in production**

A delivery-time model gained 3% accuracy offline from "restaurant's current queue length". Online, the feature came from a service with 150 ms p99 latency and was missing for 8% of requests, where the model fell back to a default and produced worse estimates than before. A batch feature, "average queue length at this hour of the week", gave most of the gain with no latency and full coverage.

</div>

## Missing values

Missingness is common and often informative: a new user has no purchase history. Keep an explicit "missing" indicator rather than silently imputing a mean, and make sure the serving path produces the same default as training. A feature that is missing at a different rate online than offline is a skew bug.

## Pruning

Feature lists grow with every experiment. Review them on a schedule: drop features with near-zero importance, features whose upstream data is being deprecated, and near-duplicates. Fewer features mean faster serving, simpler debugging and fewer ways for a pipeline change to break the model.

<div class="take">
<div class="lab">Key takeaway</div>

generate features from an entity-by-freshness grid, lean on windowed aggregates and interactions computed as of prediction time, handle high-cardinality IDs deliberately, and keep only features whose gain pays for their compute, latency, consistency and privacy costs.

</div>
