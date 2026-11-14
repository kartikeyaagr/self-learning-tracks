---
id: "03.04"
course: "ml-toolkit"
module: "03"
slug: "window-functions"
title: "Window functions"
keyIdeas: ["A window function computes a value for each row from a set of related rows, without collapsing them: `f(...) OVER (PARTITION BY ... ORDER BY ... frame)`.", "Windows are evaluated after WHERE, GROUP BY and HAVING, so you cannot filter on them directly; wrap the query in a subquery or CTE and filter outside.", "`ROW_NUMBER` gives unique positions, `RANK` leaves gaps after ties (1, 1, 3), `DENSE_RANK` does not (1, 1, 2). With ties, ROW_NUMBER's choice is arbitrary unless the ORDER BY is unique.", "With ORDER BY and no explicit frame, the default frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which includes tied rows. Say `ROWS` when you mean physical rows.", "`LAG` and `LEAD` read the previous or next row in the partition; a flag on large gaps plus a running SUM of that flag turns a click stream into sessions."]
question: "`SELECT day, SUM(gpu_hours) OVER (ORDER BY day) FROM usage` is meant to give cumulative GPU hours, but `usage` has one row per (team, day). What does the query return for a day with three rows, how does adding `ROWS UNBOUNDED PRECEDING` change it, and what is the cleanest correct query?"
answer: "With ORDER BY and no frame, the default is RANGE from the start to the current row, and in RANGE mode the current row includes all its peers. Each of the three rows for that day gets the same value: the cumulative total through the end of that day, including all three teams. With `ROWS UNBOUNDED PRECEDING` the three rows get three different partial totals, in an arbitrary order among the ties, so the intermediate values depend on which team happens to come first. Neither is a clean per-day cumulative series. The clean query aggregates to one row per day first and then runs the window: `SELECT day, SUM(SUM(gpu_hours)) OVER (ORDER BY day ROWS UNBOUNDED PRECEDING) FROM usage GROUP BY day`, or equivalently a CTE with daily sums and a window over it. Per-team cumulative totals would add `PARTITION BY team` instead."
sources: [{"title": "PostgreSQL tutorial: Window functions", "url": "https://www.postgresql.org/docs/current/tutorial-window.html", "note": "The clearest short introduction, including frames and the WINDOW clause."}, {"title": "PostgreSQL docs: Window functions", "url": "https://www.postgresql.org/docs/current/functions-window.html", "note": "Reference for ROW_NUMBER, RANK, LAG, LEAD, FIRST_VALUE and the rest."}, {"title": "SQLite: Window functions", "url": "https://www.sqlite.org/windowfunctions.html", "note": "SQLite's implementation, frame types, defaults and the EXCLUDE clause."}]
practice: ["row-number", "rank-and-dense-rank", "running-totals", "lag-and-lead", "sql-running-total"]
tags: ["window functions", "ranking", "frames"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## Aggregates that keep the rows

GROUP BY returns one row per group. A window function returns one value **per input row**, computed over a window of rows related to it. The syntax:

```sql
SELECT user_id, ts, amount,
       SUM(amount) OVER (PARTITION BY user_id ORDER BY ts
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_spend,
       AVG(amount) OVER (PARTITION BY user_id) AS user_mean
FROM orders;
```

- **PARTITION BY** splits rows into independent groups; the window never crosses a partition boundary. Without it, the whole result is one partition.
- **ORDER BY** orders rows inside each partition. It is required for ranking, LAG and LEAD, and running totals.
- **The frame** (`ROWS` or `RANGE BETWEEN ... AND ...`) selects which ordered rows around the current row feed the aggregate.

Any aggregate (`SUM`, `AVG`, `COUNT`, `MIN`, `MAX`) can be used as a window function, plus dedicated ones: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`. SQLite has supported them since version 3.25.

~~~dg-flow Window functions run late, after grouping; to filter on one, wrap the query.
FROM + WHERE -> GROUP BY + HAVING -> *window functions -> ORDER BY + LIMIT -(outer query)-> filter on rank
~~~

Because windows are computed after WHERE, `WHERE ROW_NUMBER() OVER (...) = 1` is an error. The standard pattern is a CTE or subquery with the window, then a WHERE outside.

## Ranking functions

For scores 95, 95, 90 in descending order:

| function | values | use
| --- | --- | --- |
| ROW_NUMBER() | 1, 2, 3 | pick exactly one row per group (dedup, latest record)
| RANK() | 1, 1, 3 | competition ranking: ties share, then a gap
| DENSE_RANK() | 1, 1, 2 | "top 3 distinct scores", no gaps
| NTILE(4) | bucket 1 to 4 | quartiles, deciles

Deduplicating to the latest version of each record is the most common use in data pipelines:

```sql
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY record_id
                               ORDER BY updated_at DESC, ingest_id DESC) AS rn
  FROM records)
SELECT * FROM ranked WHERE rn = 1;
```

The second ORDER BY column matters. If two versions share `updated_at`, ROW_NUMBER picks one arbitrarily, and two runs of the same pipeline can keep different rows. Make the ordering unique.

## Running totals and frames

A frame is defined relative to the current row: `UNBOUNDED PRECEDING`, `n PRECEDING`, `CURRENT ROW`, `n FOLLOWING`, `UNBOUNDED FOLLOWING`.

- **ROWS** counts physical rows. `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` is the last 7 rows, whatever dates they carry.
- **RANGE** works on the ORDER BY value. `CURRENT ROW` in RANGE mode means "all rows tied with this one" (its peers). With a numeric or date offset, `RANGE BETWEEN 6 PRECEDING AND CURRENT ROW` means values within 6 of the current one (PostgreSQL 11+ supports date intervals; SQLite supports numeric offsets).

The trap: an ORDER BY without a frame defaults to `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. If two rows share a timestamp, both get the running total *including both*, so the total jumps by two rows at once, and the per-row values no longer reflect the intended cumulative sequence. Specify `ROWS` and a unique ordering when you mean a row-by-row cumulative sum.

The same default bites `LAST_VALUE(x) OVER (ORDER BY ts)`: the frame ends at the current row, so it returns the current row's value. Use `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`, or `FIRST_VALUE` with the order reversed.

A 7-row moving average over daily data is only a 7-day average if every day has a row. Fill missing days first (a calendar table or recursive CTE, next lesson), or use a RANGE frame over a day number.

## LAG and LEAD

`LAG(x, k, default)` returns `x` from k rows earlier in the partition (default k = 1), or `default` (NULL if omitted) when there is no such row. `LEAD` looks forward. They express "change since the previous event" or "time until the next one" without a self-join.

Sessionisation combines them with a running sum: start a new session whenever the gap since the user's previous event exceeds 30 minutes, then number sessions by summing the start flags.

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

Here `ts` is Unix seconds. The `WINDOW` clause names a window definition for reuse.

<div class="ex">
<div class="lab">Worked example</div>

**a leaderboard with ties**

Scores per model on a benchmark: A 91.2, B 91.2, C 90.4, D 88.0.

`RANK() OVER (ORDER BY score DESC)` gives 1, 1, 3, 4: C is third because two models beat it. `DENSE_RANK` gives 1, 1, 2, 3, which answers "which score level is this".

`ROW_NUMBER` gives 1, 2, 3, 4, but whether A or B is first is unspecified; add `model` to the ORDER BY to make the result reproducible.

To keep the top two per benchmark: compute `RANK() OVER (PARTITION BY benchmark ORDER BY score DESC)` in a CTE and filter `rnk <= 2` outside. Ties at the boundary can return more than two rows; with ROW_NUMBER you get exactly two, chosen by your tie-breaker.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

partition to define the group, order to define the sequence, and always state the frame and a unique ordering; then filter window results in an outer query.

</div>
