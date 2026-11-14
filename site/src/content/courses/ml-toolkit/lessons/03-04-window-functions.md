---
id: "03.04"
course: "ml-toolkit"
module: "03"
slug: "window-functions"
title: "Window functions"
keyIdeas: ["A **window function** (`f(...) OVER (PARTITION BY ... ORDER BY ... frame)`) computes a value for every row from a set of related rows, without collapsing them the way GROUP BY does.", "Windows are evaluated after WHERE, GROUP BY and HAVING, so you cannot filter on them directly: compute the window in a subquery or CTE and filter in the outer query.", "`ROW_NUMBER` gives unique positions, `RANK` leaves gaps after ties (1, 1, 3) and `DENSE_RANK` does not (1, 1, 2). With ties, ROW_NUMBER's choice is arbitrary unless you add a tie-breaker to the ORDER BY.", "With ORDER BY and no explicit frame, the default is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which lumps tied rows together. Say `ROWS` when you mean physical rows, and use a RANGE frame or a filled calendar when you mean calendar days.", "`LAG` and `LEAD` read the previous or next row in the partition. A flag on large gaps plus a running SUM of that flag turns a click stream into sessions.", "For ML features, end the frame at `1 PRECEDING` so each row only sees its past; anything that reaches the current or later rows (LEAD, whole-partition aggregates) belongs in labels or reports, not features."]
question: "`SELECT day, SUM(gpu_hours) OVER (ORDER BY day) FROM usage` is meant to give cumulative GPU hours, but `usage` has one row per (team, day). What does the query return for a day with three rows, how does adding `ROWS UNBOUNDED PRECEDING` change it, and what is the cleanest correct query? If you then wanted a per-team feature 'GPU hours used before today' for a model that predicts each team's usage today, how would you write it?"
answer: "With ORDER BY and no frame, the default is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, and in RANGE mode the current row includes all its peers (rows with the same `day`). So each of the three rows for that day gets the same value: the cumulative total through the end of that day, including all three teams. With `ROWS UNBOUNDED PRECEDING` the three rows get three different partial totals (for example 6, 11 and 14 when the day's rows hold 2, 5 and 3 after a previous total of 4), in an arbitrary order among the ties, so the intermediate values depend on which team happens to come first and can change between runs. Neither is a clean per-day cumulative series. The clean query aggregates to one row per day first and then runs the window over the daily rows: `SELECT day, SUM(SUM(gpu_hours)) OVER (ORDER BY day ROWS UNBOUNDED PRECEDING) AS cumulative FROM usage GROUP BY day`, or equivalently a CTE with daily sums and a window over it. Because there is now exactly one row per day, ROWS and RANGE agree. For the per-team feature, partition by team and stop the frame before the current row so today's usage (the label) is not part of its own feature: `SELECT team, day, gpu_hours, COALESCE(SUM(gpu_hours) OVER (PARTITION BY team ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS hours_before_today FROM usage`. This assumes one row per (team, day); if a team can have several rows on a day, aggregate to (team, day) first, otherwise rows from the same day would leak into each other. If days can be missing, the ROWS frame still gives 'all earlier usage', which is correct for a cumulative feature; a fixed-length window such as 'last 7 days' would instead need `RANGE BETWEEN 7 PRECEDING AND 1 PRECEDING` over a numeric day."
sources: [{"title": "PostgreSQL tutorial: Window functions", "url": "https://www.postgresql.org/docs/current/tutorial-window.html", "note": "The clearest short introduction, including frames and the WINDOW clause."}, {"title": "PostgreSQL docs: Window functions", "url": "https://www.postgresql.org/docs/current/functions-window.html", "note": "Reference for ROW_NUMBER, RANK, LAG, LEAD, FIRST_VALUE and the rest."}, {"title": "SQLite: Window functions", "url": "https://www.sqlite.org/windowfunctions.html", "note": "SQLite's implementation, frame types, defaults and the EXCLUDE clause."}]
practice: ["row-number", "rank-and-dense-rank", "running-totals", "lag-and-lead", "sql-running-total"]
tags: ["window functions", "ranking", "frames"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## What you will learn

- What a window function is and how it differs from GROUP BY: a value for every row, computed from related rows.
- The three parts of a window (`PARTITION BY`, `ORDER BY`, the frame) and what each controls.
- Ranking with `ROW_NUMBER`, `RANK`, `DENSE_RANK`, and the "latest record per key" dedup pattern.
- Running totals, moving averages, and `ROWS` versus `RANGE` frames (including the default frame trap).
- `LAG`/`LEAD`, sessionisation, and leak-free ML features from frames that stop before the current row.

*Before you start:* SELECT and WHERE (03.01), joins (03.02) and GROUP BY (03.03). Some queries use a CTE (`WITH ...`), covered in 03.05; here you only need to read one.

Lesson 03.03 showed how GROUP BY squeezes many rows into one summary row per group. That is right for a report and wrong for a training set, where every example (an order, a click) must stay its own row while also carrying context: the customer's spend so far, the time since the previous click. Window functions add that context without losing the rows.

## The idea in one picture

Take a tiny `orders` table: two users, six orders, with `day` as a simple day number.

```sql
user_id | day | amount
u1      |  1  |  20
u1      |  3  |  35
u1      |  4  |  10
u1      |  8  |  15
u2      |  2  |  50
u2      |  5  |  25
```

GROUP BY on `user_id` returns two rows: u1 80, u2 75. A window function keeps all six rows and attaches extra columns.

~~~dg-fig 03.04-collapse.svg
The same six orders three ways. GROUP BY (middle) collapses each user to one row. Window functions (right) keep every row and add values computed from that row's group: a running total and the user's mean order size.
~~~

The query behind the right-hand table:

```sql
SELECT user_id, day, amount,
       SUM(amount) OVER (PARTITION BY user_id ORDER BY day
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_spend,
       AVG(amount) OVER (PARTITION BY user_id) AS user_mean
FROM orders;
```

In plain words: for u1's order on day 4, `running_spend` is 65 (20 + 35 + 10 so far) and `user_mean` is 80 / 4 = 20. For u2 the running total restarts at 50, because u2 is a different group.

- A **window function** is a function followed by `OVER (...)`. It returns one value per input row, computed from the rows related to it (its **window**).
- A **partition** is a group of rows sharing the `PARTITION BY` values. A window never reaches across partitions.
- The **frame** is the slice of the ordered partition that feeds the function for the current row (for a running total: "from the first row up to me").

## Step 1: the anatomy of OVER

Every window call has the same shape:

```sql
function(args) OVER (PARTITION BY ...  ORDER BY ...  frame_clause)
```

All three parts are optional, and each one answers a different question.

- **PARTITION BY**: "which rows belong together?" Independent groups, like GROUP BY but not collapsed. Without it the whole result is one partition.
- **ORDER BY**: "in what sequence?" Needed for ranking, `LAG`/`LEAD` and running totals; not for a plain per-group mean.
- **The frame** (`ROWS` or `RANGE BETWEEN start AND end`): "which ordered rows around me count?" Bounds are `UNBOUNDED PRECEDING`, `n PRECEDING`, `CURRENT ROW`, `n FOLLOWING` and `UNBOUNDED FOLLOWING`.

Any aggregate (`SUM`, `AVG`, `COUNT`, `MIN`, `MAX`) can go in front of `OVER`, plus dedicated window functions: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`. SQLite has supported them since version 3.25; PostgreSQL and the major warehouses share the same core syntax.

When several columns use the same window, name it once with a `WINDOW` clause and refer to it by name:

```sql
SELECT user_id, day, amount,
       LAG(amount) OVER w AS prev_amount,
       SUM(amount) OVER w AS running_spend
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY day);
```

## Step 2: where windows run, and why you cannot filter on them

Window functions are evaluated late: after FROM, WHERE, GROUP BY and HAVING, and just before the final ORDER BY and LIMIT.

~~~dg-flow Window functions run late, after grouping; to filter on one, wrap the query.
FROM + WHERE -> GROUP BY + HAVING -> *window functions -> ORDER BY + LIMIT -(outer query)-> filter on rank
~~~

So WHERE cannot see a window result: it ran before the window existed. `WHERE ROW_NUMBER() OVER (...) = 1` is an error (SQLite says `misuse of window function ROW_NUMBER()`). Compute the window in a subquery or CTE and filter in the outer query. The same order means a window can run *on top of* a GROUP BY, which Step 4 uses.

Some warehouses (BigQuery, Snowflake, DuckDB) add a `QUALIFY` clause that filters on window results directly; SQLite and PostgreSQL do not, so the CTE pattern is the portable one.

## Step 3: ranking rows

Ranking functions number the rows of each partition in `ORDER BY` order. They differ only in how they treat **ties** (rows with equal ordering values). For scores 95, 95, 90 sorted descending:

| function | values | typical use
| --- | --- | --- |
| ROW_NUMBER() | 1, 2, 3 | pick exactly one row per group (dedup, latest record)
| RANK() | 1, 1, 3 | competition ranking: ties share, then a gap
| DENSE_RANK() | 1, 1, 2 | "top 3 distinct scores", no gaps
| NTILE(4) | bucket 1 to 4 | quartiles, deciles, score buckets for calibration plots

`RANK` is 1 + the number of rows strictly ahead; `DENSE_RANK` is 1 + the number of distinct values ahead.

The most common ranking job in pipelines is **deduplicating to the latest version** of each record:

```sql
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY record_id
                               ORDER BY updated_at DESC, ingest_id DESC) AS rn
  FROM records)
SELECT * FROM ranked WHERE rn = 1;
```

The second ORDER BY column matters. If two versions share `updated_at`, ROW_NUMBER picks one arbitrarily, and two runs of the same pipeline can keep different rows, so your training set changes for no visible reason. Always make the ordering unique with a tie-breaker.

<div class="ex">
<div class="lab">Worked example</div>

**a leaderboard with ties**

Scores per model on one benchmark: A 91.2, B 91.2, C 90.4, D 88.0.

`RANK()`: A and B have nobody ahead, so 1. C has two models ahead: 2 + 1 = 3. D has three ahead: 4. Result 1, 1, 3, 4.

`DENSE_RANK()`: distinct scores ahead. A, B: none, so 1. C: one (91.2), so 2. D: two, so 3. Result 1, 1, 2, 3.

`ROW_NUMBER()` gives 1, 2, 3, 4, but whether A or B is first is unspecified. Ordering by `score DESC, model` makes it reproducible: A 1, B 2.

Top two per benchmark: compute `RANK() OVER (PARTITION BY benchmark ORDER BY score DESC)` in a CTE and keep `rnk <= 2` outside. Here that returns A and B (C is rank 3). On a benchmark with E 80.0, F 75.5, G 75.5 it returns three rows (ranks 1, 2, 2), because the tie sits on the boundary. With `ROW_NUMBER` you get exactly two, chosen by your tie-breaker.

</div>

## Step 4: running totals and frames

For an aggregate with `ORDER BY`, the frame decides which rows feed each row's value. Picture one partition sorted by `ts`, and look at the fifth row:

~~~dg-fig 03.04-frames.svg
Eight ordered rows with the current row fifth (ts = 7). Each line shades the rows that one frame clause sends to the aggregate. ROWS frames count physical rows. RANGE frames compare ts values, so they pull in the tied row after the current one, and the default frame with ORDER BY is a RANGE frame.
~~~

The two frame modes:

- **ROWS** counts physical rows: `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` is "me and the two rows before me".
- **RANGE** works on the ORDER BY value. `CURRENT ROW` means "every row tied with me" (its **peers**), and `RANGE BETWEEN 2 PRECEDING AND CURRENT ROW` means "rows whose ts is within 2 below mine". SQLite supports numeric offsets; PostgreSQL 11+ also accepts date and interval offsets such as `INTERVAL '6 days' PRECEDING`.

**The default frame trap.** With `ORDER BY` and no frame, the default is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. Because RANGE includes peers, tied rows all receive the same total, which already includes all of them. Here is a GPU usage table with one row per (team, day):

~~~dg-fig 03.04-rows-vs-range.svg
Nine rows of GPU usage with three rows on day 2 and three on day 5. The default RANGE frame (squares) gives all three day-2 rows the end-of-day total of 14. A ROWS frame (line) gives partial totals 6, 11, 14 whose order depends on how the tie is broken.
~~~

Neither is a clean daily cumulative series. Aggregate to one row per day first, then run the window:

```sql
SELECT day,
       SUM(gpu_hours) AS daily,
       SUM(SUM(gpu_hours)) OVER (ORDER BY day ROWS UNBOUNDED PRECEDING) AS cumulative
FROM usage GROUP BY day;
-- day 1: 4, 4   day 2: 10, 14   day 3: 6, 20   day 4: 1, 21   day 5: 9, 30
```

The inner `SUM` is the GROUP BY aggregate (one value per day); the outer `SUM ... OVER` runs across those daily rows. `ROWS UNBOUNDED PRECEDING` is shorthand for `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.

The same default bites `LAST_VALUE(x) OVER (ORDER BY ts)`: the frame ends at the current row, so it returns the current row's own value. Use `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`, or `FIRST_VALUE` with the order reversed.

## Step 5: moving averages, and gaps in the calendar

A moving average is a sliding frame: `AVG(x) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` averages the current row and the six before it. That is a 7-*day* average only if every day has exactly one row, and days with no events simply have no row.

~~~dg-fig 03.04-moving-avg.svg
Daily signups with six missing days (crosses). The 7-row average (dashed) quietly stretches over more than a week after a gap: on day 16 its seven rows reach back to day 7. The RANGE frame over the day number (solid) always covers exactly days d-6 to d.
~~~

Two honest ways to get a true 7-day window:

- A RANGE frame over a numeric day: `RANGE BETWEEN 6 PRECEDING AND CURRENT ROW` averages the rows that exist in those 7 days. If a missing day should count as zero, use `SUM(x) OVER (...) / 7.0` instead of `AVG`.
- Fill the calendar first (a calendar table, or a recursive CTE as in 03.05) so every day has a row. Then ROWS and RANGE agree.

<div class="ex">
<div class="lab">Worked example</div>

**running total, moving average and LAG by hand**

User u1 has orders on days 1, 3, 4, 8 with amounts 20, 35, 10, 15.

Running total: day 1: 20. Day 3: 20 + 35 = 55. Day 4: 55 + 10 = 65. Day 8: 65 + 15 = 80, the GROUP BY total.

3-row moving average, `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`: day 1 has only itself, 20 / 1 = 20. Day 3: (20 + 35) / 2 = 27.5. Day 4: (20 + 35 + 10) / 3 = 65 / 3 = 21.67. Day 8: (35 + 10 + 15) / 3 = 60 / 3 = 20.

3-day average, `RANGE BETWEEN 2 PRECEDING AND CURRENT ROW` over `day`: day 4 covers days 2 to 4, holding 35 and 10, so (35 + 10) / 2 = 22.5. Day 8 covers days 6 to 8, holding only 15, so 15. The ROWS version said 20 because it reached back to day 3.

`amount - LAG(amount)`: day 1 has no previous row, so NULL. Then 35 - 20 = 15, 10 - 35 = -25, 15 - 10 = 5. `day - LAG(day)` gives NULL, 2, 1, 4: the gaps between orders.

SQLite returns exactly these values.

</div>

## Step 6: LAG, LEAD and sessions

`LAG(x, k, default)` returns `x` from k rows earlier in the partition (k defaults to 1), or `default` (NULL if omitted) when there is none. `LEAD` looks forward. No self-join needed.

**Sessionisation** combines LAG with a running sum. A new session starts whenever the gap since the user's previous event is more than 30 minutes (or there is no previous event). Mark those rows with a 1, then number sessions by summing the marks:

```sql
WITH g AS (
  SELECT user_id, ts,
         CASE WHEN LAG(ts) OVER w IS NULL
                OR ts - LAG(ts) OVER w > 1800 THEN 1 ELSE 0 END AS new_session
  FROM clicks
  WINDOW w AS (PARTITION BY user_id ORDER BY ts))
SELECT user_id, ts,
       SUM(new_session) OVER (PARTITION BY user_id ORDER BY ts
                              ROWS UNBOUNDED PRECEDING) AS session_id
FROM g;
```

Here `ts` is Unix seconds, so 1800 is 30 minutes. The running sum stays flat inside a session and steps up by one at every start flag.

~~~dg-fig 03.04-sessions.svg
Two users' clicks. Gaps over 30 minutes (41, 65 and 39 minutes for u1) start new sessions (diamonds); the running sum of those flags labels sessions s1 to s4.
~~~

A GROUP BY on `(user_id, session_id)` then gives session length and clicks per session, standard features for engagement and churn models.

## Window functions in code

This runs in Python's built-in `sqlite3` (SQLite 3.53 here). It builds per-order features; the frame stops at `1 PRECEDING`, so each feature sees only earlier orders:

```python
import sqlite3
con = sqlite3.connect(':memory:')
con.executescript('''
CREATE TABLE orders(user_id TEXT, day INT, amount REAL);
INSERT INTO orders VALUES ('u1',1,20),('u1',3,35),('u1',4,10),('u1',8,15),
                          ('u2',2,50),('u2',5,25);''')
rows = con.execute('''
SELECT user_id, day, amount,
  SUM(amount) OVER w_past AS past_spend,
  COUNT(*)    OVER w_past AS past_orders,
  LEAD(day) OVER (PARTITION BY user_id ORDER BY day) - day AS days_to_next,
  ROUND(amount / SUM(amount) OVER (PARTITION BY user_id), 3) AS share
FROM orders
WINDOW w_past AS (PARTITION BY user_id ORDER BY day
                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY user_id, day''').fetchall()
for r in rows: print(r)
# ('u1', 1, 20.0, None, 0, 2, 0.25)
# ('u1', 3, 35.0, 20.0, 1, 1, 0.438)
# ('u1', 4, 10.0, 55.0, 2, 4, 0.125)
# ('u1', 8, 15.0, 65.0, 3, None, 0.188)
# ('u2', 2, 50.0, None, 0, 3, 0.667)
# ('u2', 5, 25.0, 50.0, 1, None, 0.333)
```

Three details. `SUM` over an empty frame is NULL while `COUNT` gives 0; wrap the sum in `COALESCE(..., 0)` if your model needs a number. `days_to_next` looks into the future: fine as a *label* ("will the user order again within 3 days?"), never as a *feature*. And `share` uses the whole partition, so it also peeks at future orders; it is a reporting column, not a feature.

## Checking the result

Window queries fail quietly: they return a number for every row, right or wrong. Quick checks:

- **Row count.** The output must have as many rows as the input. Fewer means an accidental GROUP BY; more means an upstream join duplicated rows.
- **End of the running total.** The last running total per partition must equal the GROUP BY total (u1 80, u2 75).
- **Ties.** `SELECT key, ts, COUNT(*) FROM t GROUP BY key, ts HAVING COUNT(*) > 1`. Any hits mean ROWS frames and ROW_NUMBER depend on tie order.
- **Frame width.** Add `COUNT(*)` and `MIN(ts)` over the same window as debug columns; they show how far back each frame reaches (that is how the day-16 problem above shows up).
- **Leakage.** For each feature, ask whether its frame includes the current row or anything after it.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Filtering in WHERE on a window | error such as "misuse of window function" | compute in a CTE or subquery, filter outside
| No frame with ORDER BY, tied keys | several rows share one running total; total jumps by many rows | aggregate to one row per key first, or use ROWS with a unique ORDER BY
| Non-unique ORDER BY in ROW_NUMBER | dedup keeps different rows on different runs | add a tie-breaker such as an ingest id
| ROWS frame over data with missing days | "7-day" average covers 10 or more days after gaps | RANGE frame over a day number, or fill the calendar first
| Missing PARTITION BY | running totals and LAG values cross from one user into the next | partition by the entity the feature describes
| Frame includes current or future rows | offline metrics look great, production is much worse | end feature frames at 1 PRECEDING; keep LEAD for labels only
| NULLs in the ORDER BY column | NULL rows sort first in SQLite and last in PostgreSQL (ascending), so ranks differ by engine | filter or COALESCE NULLs, or say NULLS FIRST / NULLS LAST

## Going deeper

**Why windows beat self-joins.** Without window functions, a running total is a correlated subquery that re-sums every earlier row for each row, so the work grows with the square of the row count. A window function sorts each partition once and slides along it, adding the entering row and (for bounded frames) removing the leaving one.

~~~dg-fig 03.04-timing.svg
Running total over n rows in SQLite, measured here on a shared CPU (timings are machine-dependent). The correlated subquery grows roughly with n squared (about 52 ms at 1,000 rows, 3.2 s at 8,000); the window version stays in the low milliseconds.
~~~

At 8,000 rows the measured gap was about 270 times. The window query is dominated by its sort, roughly $O(n \log n)$, and columns sharing one `PARTITION BY` / `ORDER BY` can share that sort.

**GROUPS frames and EXCLUDE.** A third frame mode, `GROUPS`, counts peer groups instead of rows or values. Frames can also drop rows with `EXCLUDE CURRENT ROW`, `EXCLUDE GROUP` or `EXCLUDE TIES` (SQLite 3.28+, PostgreSQL 11+). `EXCLUDE CURRENT ROW` gives "mean of the other members of my group", a classic target-encoding feature.

**Filters change window results.** Because windows run after WHERE, adding `WHERE day >= 3` changes every running total: days 1 and 2 are no longer in any frame. To keep full-history totals but show recent rows, compute the window in a CTE and filter outside; forgetting this makes backfills and daily jobs disagree.

**Point-in-time correctness.** For training data, "past" must mean "known at prediction time". If events arrive late, a frame over event time can include rows production had not yet seen; framing by ingestion time, or an as-of join, avoids this. Lesson 03.06 builds training tables with these rules.

## Key terms

| Term | Meaning
| --- | --- |
| Window function | A function with `OVER (...)` that returns one value per row, computed from related rows
| Partition | The group of rows sharing the `PARTITION BY` values; windows never cross it
| Frame | The slice of the ordered partition that feeds the function for the current row
| ROWS frame | Frame bounds counted in physical rows
| RANGE frame | Frame bounds measured on the ORDER BY value; includes tied peers
| Peers | Rows with the same ORDER BY value as the current row
| ROW_NUMBER / RANK / DENSE_RANK | Unique positions / ties share then gap / ties share, no gap
| LAG / LEAD | The value from an earlier / later row in the partition

<div class="take">
<div class="lab">Key takeaway</div>

partition to define the group, order to define the sequence, and always state the frame and a unique ordering; then filter window results in an outer query, and end feature frames before the current row so the model never sees the future.

</div>
