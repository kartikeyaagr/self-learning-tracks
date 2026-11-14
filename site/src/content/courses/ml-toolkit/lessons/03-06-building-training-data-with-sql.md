---
id: "03.06"
course: "ml-toolkit"
module: "03"
slug: "building-training-data-with-sql"
title: "Building training data with SQL"
keyIdeas: ["A training set is a **spine** of (entity, prediction time) rows. Each row gets features computed only from data before its cutoff and a label computed from a window after it.", "Point-in-time correctness means joining each spine row to the latest feature value whose timestamp is at or before the cutoff, using the time the data became **available**, not the time the event happened.", "Leakage usually enters through innocent-looking joins: a current-state table (`users.plan` today), an aggregate over all history, a row that was later overwritten, or a feature derived from the outcome itself.", "Split by time, and keep each entity on one side of the split, so evaluation looks like deployment: the model always predicts a future it has not seen."]
question: "A fraud model is trained on `transactions` joined to `accounts` for features like `account_age_days`, `num_chargebacks` and `is_frozen`. Offline AUC is 0.97; in production it is 0.71. Identify at least two likely leaks in this setup and say how to rebuild the features point-in-time correctly."
answer: "First, `accounts` is a current-state table. `is_frozen` is very likely set *because* fraud was detected, so it encodes the label; it must be dropped or taken as of the transaction time, which would usually be false. Second, `num_chargebacks` counted over all history includes chargebacks filed after the transaction, often for that very transaction. It must count only chargebacks known before the transaction time, using the date the chargeback was recorded, not the date of the original purchase. Third, `account_age_days` computed from today's date rather than the transaction date gives every account a later age, distorting the feature. The rebuild: make the spine one row per transaction with its timestamp as the cutoff, compute each feature from history tables filtered to records available before that cutoff, and evaluate on a later time period than training."
sources: [{"title": "Feast docs: Point-in-time joins", "url": "https://docs.feast.dev/getting-started/concepts/point-in-time-joins", "note": "How a feature store joins feature history to an entity table at each event time."}, {"title": "DuckDB blog: ASOF joins, fuzzy temporal lookups", "url": "https://duckdb.org/2023/09/15/asof-joins-fuzzy-temporal-lookups.html", "note": "The as-of join explained with examples, and how it is executed efficiently."}, {"title": "scikit-learn: Common pitfalls (data leakage)", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Leakage from preprocessing and splitting, from the modelling side."}, {"title": "Sculley et al.: Hidden technical debt in machine learning systems", "url": "https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems", "note": "Data dependencies and feedback loops that make training data drift from serving."}]
practice: ["point-in-time-feature-join", "lag-and-lead", "date-functions", "null-handling"]
tags: ["point-in-time", "leakage", "training data"]
tools: ["SQL", "SQLite"]
---

## The shape of a training set

Event tables (clicks, orders, logins, model requests) are not training data. A supervised dataset needs one row per **prediction**: an entity and the moment at which the model would have been asked. Call this table the **spine**. For a churn model scored every Monday:

~~~dg-flow Every spine row splits time at its cutoff: features look back, the label looks forward.
Feature window, 30 days before -(cutoff t)-> *Prediction time -(label window)-> Did the user return within 30 days?
~~~

- **Features** use only data with timestamps before the cutoff `t`: purchases in the last 30 days, days since last login, plan at time `t`.
- **Label** uses only data in the label window after `t`: no login in the 30 days after `t`.
- An entity appears once per cutoff, so 100,000 users scored weekly for a year gives 5.2 million rows.

Building the dataset this way, rather than as one row per user with "all their data", is what makes it resemble production, where the model sees the past and predicts the future.

## Labels from event tables

The spine and labels are ordinary SQL. With dates as text in ISO format (`'2025-03-03'`), SQLite's `date()` function does the arithmetic; PostgreSQL would use `interval '30 days'`.

```sql
WITH spine AS (
  SELECT u.user_id, c.cutoff
  FROM users u CROSS JOIN cutoffs c            -- one row per Monday
  WHERE u.signup_date < c.cutoff),             -- only users who existed then
labels AS (
  SELECT s.user_id, s.cutoff,
         CASE WHEN EXISTS (SELECT 1 FROM logins l
                           WHERE l.user_id = s.user_id
                             AND l.ts >= s.cutoff
                             AND l.ts <  date(s.cutoff, '+30 days'))
              THEN 0 ELSE 1 END AS churned
  FROM spine s)
SELECT * FROM labels;
```

Two details matter. The spine includes only users who existed at the cutoff; including later sign-ups creates rows with impossible features. And cutoffs too close to the present have incomplete label windows: a cutoff 10 days ago cannot yet say whether the user returned within 30. Drop those rows, or the newest data will look full of churners.

## Point-in-time feature joins

Feature tables usually store history as rows `(user_id, valid_from, value)`. For each spine row you need the latest value **as of** the cutoff:

```sql
SELECT s.user_id, s.cutoff, f.value AS credit_score
FROM spine s
LEFT JOIN features f
  ON f.user_id = s.user_id
 AND f.valid_from = (SELECT MAX(f2.valid_from) FROM features f2
                     WHERE f2.user_id = s.user_id
                       AND f2.valid_from <= s.cutoff);
```

A `ROW_NUMBER() OVER (PARTITION BY s.user_id, s.cutoff ORDER BY f.valid_from DESC)` over the join filtered to `valid_from <= cutoff` does the same thing. Warehouses such as DuckDB and Snowflake offer an `ASOF JOIN` that expresses it directly, and feature stores such as Feast implement exactly this join for you.

The subtle part is which timestamp you compare. An event that happened at 09:00 but landed in the warehouse at 23:00 (batch ingestion, late-arriving data, a nightly ETL job) was not available to a model scoring at noon. If the table records it, join on the ingestion or availability time, or add the pipeline's known delay to the event time. Otherwise offline features are fresher than online ones, and the model looks better in evaluation than in production. This mismatch is one form of training/serving skew.

## Where leakage comes from

| source | how it looks in SQL | fix
| --- | --- | --- |
| current-state tables | `JOIN users` for `plan` or `country` as they are today | use a history table or snapshot as of the cutoff
| all-history aggregates | `COUNT(*)` of orders per user over the whole table | restrict to `ts < cutoff` inside the aggregate
| overwritten rows | `status` updated in place after the outcome | read from change logs or append-only history
| outcome proxies | `refund_issued` used to predict fraud | drop features caused by the label
| encodings fit on everything | target mean per category over all rows | compute from training rows before the cutoff only
| random splits | same user in train and test, future in train | split by time, group by entity

The last row matters as much as the others. A random row split puts a user's week-10 row in training and their week-11 row in test; the model learns the user, not the pattern. Choose a time boundary: train on cutoffs before it, validate on cutoffs after it. Keep a gap at least as long as the label window, so no training label peeks into the validation period. If entities repeat, also assign each entity to a single side, for example by a stable hash of its id. The Machine Learning course's lessons on splits and pipelines treat leakage from the modelling side, and the pandas module covers lag features without leakage.

<div class="ex">
<div class="lab">Worked example</div>

**a churn table, sized and checked**

100,000 users, 52 weekly cutoffs, about 70% of users existing at a typical cutoff: about 3.6 million spine rows. With a 30-day label window, the cutoffs less than 30 days old (the last 4 or 5 weeks) are dropped as incomplete.

Sanity checks worth running in SQL: the churn rate per cutoff should be roughly stable (a jump at the end means incomplete label windows); no joined feature is newer than its row's cutoff (a query for rows with `f.valid_from > s.cutoff` must return nothing); and the count of spine rows per user never exceeds the number of cutoffs.

A feature with suspiciously high importance, such as "days until next login", is almost always a leak.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

build a spine of (entity, cutoff) rows, join features as of each cutoff using availability time, derive labels strictly after it, and split by time and entity so evaluation repeats deployment.

</div>
