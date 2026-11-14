---
id: "03.06"
course: "ml-toolkit"
module: "03"
slug: "building-training-data-with-sql"
title: "Building training data with SQL"
keyIdeas: ["A training set is a **spine** of (entity, cutoff) rows, one per prediction the model would make. Each row gets features computed only from data before its cutoff and a label computed from a window after it.", "Drop cutoffs whose label window has not closed yet; otherwise everyone who has not come back *so far* is labelled a churner and the newest rows look full of churn.", "Point-in-time correctness means joining each spine row to the latest value whose timestamp is at or before the cutoff (`ROW_NUMBER()` or `ASOF JOIN`), using the time the data became **available**, not the time the event happened.", "Leakage usually enters through innocent-looking joins: a current-state table (`users.plan` today), an aggregate over all history, a row that was later overwritten, or a feature derived from the outcome itself. It makes offline scores too good to be true.", "Split by time with a gap at least as long as the label window, and keep each entity on one side when that matters, so evaluation looks like deployment: the model always predicts a future it has not seen."]
question: "A fraud model is trained on `transactions` joined to `accounts` for features like `account_age_days`, `num_chargebacks` and `is_frozen`. Offline AUC is 0.97; in production it is 0.71. Identify at least two likely leaks in this setup and say how to rebuild the features point-in-time correctly, including how you would split the data for evaluation."
answer: "First, `accounts` is a current-state table. `is_frozen` is very likely set *because* fraud was detected, so it encodes the label; it must be dropped, or taken as of the transaction time from a history of status changes, where it would almost always be false. Second, `num_chargebacks` counted over all history includes chargebacks filed after the transaction, often for that very transaction. It must count only chargebacks known before the transaction time, using the date the chargeback was recorded (its availability time), not the date of the original purchase. Third, `account_age_days` computed from today's date rather than the transaction date gives every account a later age, distorting the feature. The rebuild: make the spine one row per transaction with its timestamp as the cutoff; compute each feature from history or append-only tables filtered to records available before that cutoff (an as-of join with `ROW_NUMBER()` or `ASOF JOIN`, comparing load time rather than event time where ingestion is delayed); and derive the label from chargebacks in a window after the transaction, dropping recent transactions whose window has not closed. For evaluation, train on transactions before a date and validate on a later period, leaving a gap at least as long as the chargeback window so training labels do not see the validation period. Before shipping, check that no joined feature is newer than its cutoff, that offline and online feature values agree on a recent sample, and that no feature has suspiciously high importance."
sources: [{"title": "Feast docs: Point-in-time joins", "url": "https://docs.feast.dev/getting-started/concepts/point-in-time-joins", "note": "How a feature store joins feature history to an entity table at each event time."}, {"title": "DuckDB blog: ASOF joins, fuzzy temporal lookups", "url": "https://duckdb.org/2023/09/15/asof-joins-fuzzy-temporal-lookups.html", "note": "The as-of join explained with examples, and how it is executed efficiently."}, {"title": "scikit-learn: Common pitfalls (data leakage)", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Leakage from preprocessing and splitting, from the modelling side."}, {"title": "Sculley et al.: Hidden technical debt in machine learning systems", "url": "https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems", "note": "Data dependencies and feedback loops that make training data drift from serving."}]
practice: ["point-in-time-feature-join", "lag-and-lead", "date-functions", "null-handling"]
tags: ["point-in-time", "leakage", "training data"]
tools: ["SQL", "SQLite"]
---

## What you will learn

- How a **spine** of (entity, cutoff) rows turns event tables into one row per prediction.
- How to compute labels from a window *after* each cutoff and features from a window *before* it, in plain SQL.
- How to write a **point-in-time** (as-of) join, and why the timestamp you compare must be the time the data became available.
- Where leakage hides in innocent-looking SQL, and checks that catch it.
- How to split the finished table by time (with a gap) and by entity, so evaluation repeats deployment.

*Before you start:* joins (03.02), `GROUP BY` (03.03), window functions such as `ROW_NUMBER()` (03.04), and CTEs and correlated subqueries (03.05). All queries run in Python's built-in `sqlite3` (SQLite 3.53).

Lesson 03.05 stacked CTEs into readable pipelines. Here that skill does the job SQL most often does in ML work: turning logs into a training set that tells the truth about production.

## The idea in one picture

A churn model scores each user every Monday: how likely are they to stop logging in? Three tiny tables; user 3 signed up on 5 March, and data stops on 20 March.

```text
logins                     plan_history (valid_from, plan)    cutoffs
user 1: 26 Feb, 1 Mar,     user 1: 10 Jan free                3 Mar
        4 Mar, 8 Mar,      user 1:  6 Mar pro                 10 Mar
        12 Mar             user 2: 20 Feb free                17 Mar
user 2: 24 Feb, 28 Feb     user 3:  5 Mar free
user 3: 6 Mar, 7 Mar, 15 Mar
```

A training set is not "one row per user" but one row per moment the model would have been asked. That moment is the **cutoff** (prediction time, as-of time). The table of all (user, cutoff) pairs is the **spine**: it fixes the row count before any feature exists.

~~~dg-fig 03.06-timeline.svg
The toy data as a timeline. Around the 10 March cutoff, the shaded band on the left is the feature window (only the past) and the band on the right is the label window (only the future). Each spine row gets its own label. The 17 March cutoff is dropped because its label window would run past the end of the data.
~~~

Read it row by row. At 10 March, user 1 logged in twice in the week before (features) and again on 12 March (`churned = 0`). User 2's last login was 28 February and nothing follows (`churned = 1`). User 3 signed up on 5 March, so is on the spine at 10 March but not at 3 March. Five rows, and every feature is something the model could have known that Monday morning.

Three new words:

- **Feature window**: the slice of history before the cutoff that features may use, for example the last 30 days.
- **Label window**: the slice of time after the cutoff that defines the answer, for example "no login in the next 30 days".
- **Leakage**: information from at or after the cutoff that sneaks into a feature, making offline scores look better than production.

## Step 1: build the spine

The spine is a cross product of entities and cutoffs, filtered to pairs that make sense: a user must exist at the cutoff, and the label window must be complete.

```sql
WITH spine AS (
  SELECT u.user_id, c.cutoff
  FROM users u CROSS JOIN cutoffs c                 -- one row per user per Monday
  WHERE u.signup_date < c.cutoff                    -- only users who existed then
    AND date(c.cutoff, '+30 days') <= :data_end)    -- label window is complete
SELECT * FROM spine;
```

Dates are ISO text (`'2025-03-10'`), so string order is date order and SQLite's `date()` does the arithmetic (PostgreSQL: `c.cutoff + interval '30 days'`). A cutoff `'2025-03-10'` means midnight, so a login at `'2025-03-10 07:45'` is already in its future. Use `<` for "before" and `>=` for "at or after" everywhere.

Many teams also keep only users *active* in the 30 days before the cutoff. Whatever filter production applies, the spine must apply too.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a weekly churn spine**

Suppose 100,000 users, 52 weekly cutoffs in a year, and about 70% of the users existing at a typical cutoff.

Spine rows $\approx 100{,}000 \times 52 \times 0.70 = 3{,}640{,}000$, so about 3.6 million.

With a 30-day label window, the last 4 or 5 Mondays have incomplete windows and are dropped.

Each user appears at most 52 times, so `SELECT user_id FROM spine GROUP BY user_id HAVING COUNT(*) > 52` must return nothing; otherwise a join duplicated rows.

</div>

## Step 2: labels from the future window

The label is a fact about the label window, and `EXISTS` (from 03.05) expresses "did anything happen in that window" directly:

$$\text{churned}(u, t) = \begin{cases} 0 & \text{if user } u \text{ has a login with } t \le ts < t + 30\text{ days} \\ 1 & \text{otherwise} \end{cases}$$

Here $u$ is the user, $t$ is the cutoff and $ts$ is a login timestamp.

In SQL it is a `CASE WHEN EXISTS (SELECT 1 FROM logins l WHERE l.user_id = s.user_id AND l.ts >= s.cutoff AND l.ts < date(s.cutoff, '+30 days')) THEN 0 ELSE 1 END` per spine row; the full query appears in the code section.

The completeness filter from step 1 matters. A cutoff 10 days ago cannot yet say whether the user returns within 30 days, so everyone who has not come back *so far* is labelled a churner, and the newest rows look full of churn.

~~~dg-fig 03.06-incomplete.svg
Churn rate per weekly cutoff, measured on a synthetic 3,000-user warehouse: flat near 10% until the last cutoffs, whose 30-day label windows run past the end of the data, where it jumps to about 30%. Drop those rows.
~~~

This is also a cheap diagnostic: compute the label rate per cutoff (`GROUP BY cutoff`) on every rebuild. A cliff at either end usually means an incomplete window or a spine filter that changed.

## Step 3: features from the past window

Features are aggregates over the feature window, which is the `GROUP BY` from 03.03 with the time filter inside the join condition:

```sql
SELECT s.user_id, s.cutoff,
       COUNT(l.ts) AS logins_30d                       -- 0 when there are none
FROM spine s
LEFT JOIN logins l
  ON l.user_id = s.user_id
 AND l.ts >= date(s.cutoff, '-30 days')
 AND l.ts <  s.cutoff                                   -- strictly before the cutoff
GROUP BY s.user_id, s.cutoff;
```

Use `LEFT JOIN` with `COUNT(l.ts)`, not `COUNT(*)`, so a user with no logins gets 0 instead of vanishing or getting 1. Keep the time filter in `ON`: in `WHERE` it would discard the left join's NULL rows, making it an inner join.

"Days since last login" is a `MAX(ts)` over rows strictly before the cutoff; window-function frames (03.04) must also end before it.

## Step 4: the point-in-time join

Slowly changing attributes (plan, country, credit limit) live in history tables with one row per change, `(user_id, valid_from, value)`. Each spine row needs the value **as of** its cutoff: the row with the largest `valid_from` at or before the cutoff.

~~~dg-fig 03.06-asof.svg
One user's plan history (solid steps) and the value an as-of join returns at each cutoff (dots). The dashed line is what a plain join to a current-state users table returns: today's value, copied into every past row.
~~~

In SQL, join every history row that is not in the future, then keep the newest one per spine row with `ROW_NUMBER()`:

```sql
SELECT user_id, cutoff, plan_asof FROM (
  SELECT s.user_id, s.cutoff, p.plan AS plan_asof,
         ROW_NUMBER() OVER (PARTITION BY s.user_id, s.cutoff
                            ORDER BY p.valid_from DESC) AS rn
  FROM spine s
  LEFT JOIN plan_history p
    ON p.user_id = s.user_id AND p.valid_from <= s.cutoff)
WHERE rn = 1;
```

A correlated subquery, `f.valid_from = (SELECT MAX(f2.valid_from) ... WHERE f2.valid_from <= s.cutoff)`, says the same thing. DuckDB and Snowflake offer an `ASOF JOIN` that states the intent directly; in DuckDB, `FROM spine s ASOF LEFT JOIN plan_history p ON s.user_id = p.user_id AND s.cutoff >= p.valid_from`. Feature stores such as Feast run exactly this join when you request historical features for an entity table with timestamps.

<div class="ex">
<div class="lab">Worked example</div>

**the toy table by hand**

Spine: users signed up before each cutoff, with complete 7-day label windows (data ends 20 March, so 17 March is dropped): (1, 3 Mar), (2, 3 Mar), (1, 10 Mar), (2, 10 Mar), (3, 10 Mar), 5 rows.

Labels, window $[t, t + 7\text{ days})$. Row (1, 3 Mar): logins on 4 and 8 March fall in $[3, 10)$ March, so churned $= 0$. Row (2, 3 Mar): no login after 28 February, churned $= 1$. Likewise (1, 10 Mar) $= 0$ (12 March), (2, 10 Mar) $= 1$, (3, 10 Mar) $= 0$ (15 March). Churn rate $2/5 = 0.4$.

Features, window $[t - 7\text{ days}, t)$. Row (1, 3 Mar): 24 Feb to 2 Mar holds logins on 26 Feb and 1 Mar, so `logins_7d` $= 2$; last login 1 March, so `days_since_last` $= 2$. Row (2, 10 Mar): no logins from 3 to 9 March, so 0; last login 28 February, $1 + 9 = 10$ days (to 1 March, then to 10 March).

As-of plan. Row (1, 3 Mar): the only history row with `valid_from` $\le$ 3 March is (10 Jan, free), so free. Row (1, 10 Mar): (6 Mar, pro) is the newest, so pro.

The leak: `users.plan` today says pro for user 1, so a plain join gives row (1, 3 Mar) pro, three days before the upgrade. One wrong value in five here; in real data, thousands, correlated with the outcome.

</div>

## Step 5: compare the time the data became available

The subtle part is *which* timestamp you compare. An event at 09:00 that landed in the warehouse at 02:00 the next night (nightly ETL, late-arriving data) was not available to a model scoring at noon. **Event time** is when something happened; **availability time** (or load time) is when your systems could first see it.

~~~dg-fig 03.06-avail.svg
A nightly batch load. Six logins happened before the noon scoring time, but only the three from the previous day had been loaded. A feature computed on event time counts six; the production model saw three.
~~~

If the table records a load timestamp, join on it; if not, add the pipeline's known delay (`date(ts, '+1 day')` for a nightly load). Otherwise offline features are fresher than online ones and the model looks better in evaluation than in production, one form of **training/serving skew**.

In the toy data, user 1's upgrade has `valid_from = '2025-03-06'` but `loaded_at = '2025-03-07 02:00'`: a model scoring at 20:00 on 6 March saw free, not pro.

## Building it in code

The full toy pipeline in `sqlite3`, with real output. `pd.read_sql` returns DataFrames ready for scikit-learn.

```python
import sqlite3, pandas as pd
con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE users(user_id INT, signup_date TEXT, plan TEXT);
INSERT INTO users VALUES (1,'2025-01-10','pro'),(2,'2025-02-20','free'),(3,'2025-03-05','free');
CREATE TABLE logins(user_id INT, ts TEXT);
INSERT INTO logins VALUES
 (1,'2025-02-26 08:10'),(1,'2025-03-01 19:30'),(1,'2025-03-04 07:45'),(1,'2025-03-08 12:00'),
 (1,'2025-03-12 21:05'),(2,'2025-02-24 09:00'),(2,'2025-02-28 18:20'),
 (3,'2025-03-06 10:00'),(3,'2025-03-07 22:40'),(3,'2025-03-15 08:30');
CREATE TABLE plan_history(user_id INT, valid_from TEXT, plan TEXT, loaded_at TEXT);
INSERT INTO plan_history VALUES (1,'2025-01-10','free','2025-01-11 02:00'),
 (1,'2025-03-06','pro','2025-03-07 02:00'),(2,'2025-02-20','free','2025-02-21 02:00'),
 (3,'2025-03-05','free','2025-03-06 02:00');
CREATE TABLE cutoffs(cutoff TEXT);
INSERT INTO cutoffs VALUES ('2025-03-03'),('2025-03-10'),('2025-03-17');
""")
SPINE = """WITH spine AS (
  SELECT u.user_id, c.cutoff FROM users u CROSS JOIN cutoffs c
  WHERE u.signup_date < c.cutoff AND date(c.cutoff, '+7 days') <= '2025-03-20')"""
```

Labels and window features in one query:

```python
train = pd.read_sql(SPINE + """
SELECT s.user_id, s.cutoff,
  (SELECT COUNT(*) FROM logins l WHERE l.user_id = s.user_id
     AND l.ts >= date(s.cutoff, '-7 days') AND l.ts < s.cutoff)            AS logins_7d,
  julianday(s.cutoff) - julianday((SELECT date(MAX(ts)) FROM logins l
     WHERE l.user_id = s.user_id AND l.ts < s.cutoff))                     AS days_since_last,
  CASE WHEN EXISTS (SELECT 1 FROM logins l WHERE l.user_id = s.user_id
     AND l.ts >= s.cutoff AND l.ts < date(s.cutoff, '+7 days')) THEN 0 ELSE 1 END AS churned
FROM spine s ORDER BY s.cutoff, s.user_id""", con)
print(train)
#    user_id      cutoff  logins_7d  days_since_last  churned
# 0        1  2025-03-03          2              2.0        0
# 1        2  2025-03-03          2              3.0        1
# 2        1  2025-03-10          2              2.0        0
# 3        2  2025-03-10          0             10.0        1
# 4        3  2025-03-10          2              3.0        0
```

The as-of join next to the leaky current-state join, plus a check query:

```python
asof = SPINE + """, asof AS (
  SELECT s.user_id, s.cutoff, p.valid_from, p.plan AS plan_asof, u.plan AS plan_now,
         ROW_NUMBER() OVER (PARTITION BY s.user_id, s.cutoff
                            ORDER BY p.valid_from DESC) AS rn
  FROM spine s JOIN users u ON u.user_id = s.user_id
  LEFT JOIN plan_history p ON p.user_id = s.user_id AND p.valid_from <= s.cutoff)"""
print(pd.read_sql(asof + """ SELECT user_id, cutoff, plan_asof, plan_now FROM asof
                         WHERE rn = 1 ORDER BY cutoff, user_id""", con))
#    user_id      cutoff plan_asof plan_now
# 0        1  2025-03-03      free      pro      <- today's value leaks backwards
# 1        2  2025-03-03      free     free
# 2        1  2025-03-10       pro      pro
# ...
print(pd.read_sql(asof + """ SELECT SUM(valid_from > cutoff) AS future_rows,
       SUM(plan_asof <> plan_now) AS today_differs, COUNT(*) AS spine_rows
FROM asof WHERE rn = 1""", con))
#    future_rows  today_differs  spine_rows
# 0            0              1           5
```

`future_rows = 0` proves the join never looked ahead; `today_differs` counts rows a current-state join would get wrong.

## Step 6: split by time and by entity

A random row split puts a user's 10 March row in training and their 17 March row in test (the model has seen that user), and puts later cutoffs in training than in test (it has seen the future). Choose a **time boundary** instead: train on cutoffs before it, validate on cutoffs after it.

~~~dg-fig 03.06-split.svg
Weekly cutoffs as rows, each with its feature window behind it (grey) and its 30-day label window ahead of it. Training rows end at a boundary; the next five cutoffs are dropped as a gap, so no training label looks into the validation period.
~~~

The gap must be at least as long as the label window; otherwise the last training labels contain the very events the first validation rows try to predict.

```sql
SELECT *, CASE WHEN cutoff <  '2025-08-04' THEN 'train'
               WHEN cutoff >= '2025-09-08' THEN 'valid'     -- 35-day gap > 30-day label window
               ELSE 'gap' END AS split,
          (user_id * 2654435761) % 100 AS bucket             -- stable per-user hash, 0..99
FROM training_rows;
```

To measure performance on *new* users, also put each user on one side by a stable hash of the id (`bucket < 80` for training; DuckDB has a built-in `hash()`). On the synthetic warehouse behind the figures this gives 34,871 train, 7,749 gap and 19,868 validation rows.

## Reading the result: does leakage really matter?

The synthetic warehouse (spine: users active in the 30 days before each cutoff) has three leaky columns next to honest ones: `users.plan` today, `users.status` (overwritten to closed after 45 silent days), and a login count over *all* history. One gradient-boosted model was trained with them, one without.

~~~dg-fig 03.06-auc.svg
Measured AUC on a synthetic churn table (62,488 rows). Leaky features score 0.98 offline under either split; fed the values production would really have, the same model falls to 0.55. Point-in-time features give an honest 0.92 to 0.93.
~~~

Leakage produces a score that is *too good*, so it survives review. A time split alone does not fix it, because the leak is in the columns, not in which rows go where. And the leaky model collapses because it relied on `closed_now`, which is always "not closed" at scoring time. The honest model's two splits agree here; on real data with per-user patterns or drift, the random split is usually optimistic.

On every rebuild, also check that no feature has suspiciously high importance ("days until next login" is always a leak) and that offline and online values agree on recent rows.

## What goes wrong

| Source | How it looks in SQL | Symptom | Fix
| --- | --- | --- | --- |
| Current-state tables | `JOIN users` for `plan` or `country` as they are today | great offline, poor online | use a history table or snapshot as of the cutoff
| All-history aggregates | `COUNT(*)` of orders per user over the whole table | feature counts future events | restrict to `ts < cutoff` inside the aggregate
| Overwritten rows | `status` updated in place after the outcome | one feature dominates importance | read change logs or append-only history
| Outcome proxies | `refund_issued` used to predict fraud | near-perfect AUC | drop features caused by the label
| Encodings fit on everything | target mean per category over all rows | validation better than it should be | compute from training rows before the cutoff only
| Event time instead of load time | `valid_from <= cutoff` on late-arriving data | online features staler than offline | join on `loaded_at` or add the known delay
| Incomplete label windows | cutoffs within 30 days of today | churn rate spikes at the end | drop cutoffs whose window is not closed
| Random splits | same user in train and test, future in train | test score drops after launch | split by time with a gap, group by entity

The Machine Learning course's lessons on splits and pipelines treat leakage from the modelling side, and the pandas module covers lag features without leakage.

## Going deeper

**Why the gap equals the label window.** A training row at cutoff $t$ uses events up to $t + W$, where $W$ is the label window. A validation row at cutoff $v$ predicts events in $[v, v + W)$. They overlap unless $t + W \le v$, so the first validation cutoff must be at least $t_{\max} + W$. Validation *features* may overlap training labels; that also happens in production.

**Cost of the as-of join.** A correlated subquery costs up to (spine rows) times (history rows per entity). An index on `(user_id, valid_from)` makes each lookup a short B-tree seek (03.07). Native as-of joins sort both inputs and merge them in one pass.

**Slowly changing dimensions.** Warehouses often store history with `valid_from` and `valid_to` (a "type 2" dimension); the as-of join becomes `p.valid_from <= s.cutoff AND (p.valid_to > s.cutoff OR p.valid_to IS NULL)`. A table that keeps only the current row cannot recover the past, so start snapshotting it today.

## Key terms

| Term | Meaning
| --- | --- |
| Spine | The table of (entity, cutoff) pairs; one row per prediction the model would make
| Cutoff | The moment of prediction; features use only data before it, labels only data after it
| Feature window | The slice of history before the cutoff that a feature aggregates over
| Label window | The slice of time after the cutoff that defines the outcome
| Point-in-time (as-of) join | Joining each spine row to the newest history row at or before its cutoff
| Availability time | When data could first be seen by your systems, as opposed to when the event happened
| Leakage | Information from at or after the cutoff that reaches a feature
| Training/serving skew | Features computed differently offline than online

<div class="take">
<div class="lab">Key takeaway</div>

build a spine of (entity, cutoff) rows, join features as of each cutoff using availability time, derive labels strictly after it from complete windows, and split by time with a gap (and by entity) so evaluation repeats deployment.

</div>
