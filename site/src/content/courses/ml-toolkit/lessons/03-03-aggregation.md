---
id: "03.03"
course: "ml-toolkit"
module: "03"
slug: "aggregation"
title: "Aggregation"
keyIdeas: ["`GROUP BY` produces one output row per distinct combination of the grouping columns; every other selected column must be inside an aggregate. NULL keys form one group of their own.", "`COUNT(*)` counts rows, `COUNT(col)` counts non-NULL values, `COUNT(DISTINCT col)` counts distinct non-NULL values. `AVG` and `SUM` skip NULLs, and `SUM` over nothing is NULL, not 0. With integer columns, `SUM(a) / COUNT(*)` truncates; use `AVG` or multiply by 1.0.", "WHERE filters rows before grouping; HAVING filters groups after aggregation. Push every condition that does not need an aggregate into WHERE.", "Conditional aggregation (`SUM(CASE WHEN ... THEN 1 ELSE 0 END)` or `COUNT(*) FILTER (WHERE ...)`) computes several breakdowns in one pass, such as a whole confusion matrix per model, and keeps groups with zero matches.", "Distinct counts and averages are not additive: daily active users do not sum to weekly ones, and an average of per-group averages is not the overall average. Store counts and sums, and aggregate per entity before aggregating across entities."]
question: "A dashboard computes weekly active users as the sum of seven daily `COUNT(DISTINCT user_id)` values, and the average session length per country as the average of per-city averages. Explain what each gets wrong, and write the shape of the correct query for each."
answer: "Summing daily distinct counts counts a user once for every day they were active, so a user active all week contributes 7; distinct counts are not additive (they are 'holistic': no fixed-size daily summary lets you combine them exactly). Weekly actives must be one distinct count over the whole week's raw rows: `SELECT week, COUNT(DISTINCT user_id) FROM events GROUP BY week`, with week derived from the timestamp. In a simulation of 1,000 users, the seven daily counts summed to 2,032 while only 671 distinct users were active, so the error is large, not a rounding issue. If raw events are too expensive to rescan, keep a mergeable sketch such as HyperLogLog per day and accept a small approximation error. The average of per-city averages gives each city equal weight regardless of how many sessions it had, so a small city with unusual sessions distorts the country figure (10 sessions at 30 minutes, 200 at 12 and 2,000 at 8 give 16.67 unweighted but 8.46 overall). The correct figure aggregates raw sessions directly, `SELECT country, AVG(session_length) FROM sessions GROUP BY country`, or, if only per-city summaries exist, weights by count: `SELECT country, SUM(city_avg * city_n) / SUM(city_n) FROM city_stats GROUP BY country`. Both mistakes come from treating non-additive statistics as if they could be combined from group results; the general fix is to store counts and sums (which are additive) in summary tables and derive means and rates at the last step, and to show `COUNT(*)` next to every average so the weights are visible."
sources: [{"title": "PostgreSQL tutorial: Aggregate functions", "url": "https://www.postgresql.org/docs/current/tutorial-agg.html", "note": "GROUP BY, HAVING and the WHERE vs HAVING distinction, with examples."}, {"title": "PostgreSQL docs: Aggregate functions", "url": "https://www.postgresql.org/docs/current/functions-aggregate.html", "note": "Every aggregate, NULL behaviour, and FILTER and ordered-set aggregates."}, {"title": "SQLite: Built-in aggregate functions", "url": "https://www.sqlite.org/lang_aggfunc.html", "note": "SQLite's aggregates, including sum() vs total() on empty input."}, {"title": "Modern SQL: The FILTER clause", "url": "https://modern-sql.com/feature/filter", "note": "Standard conditional aggregation and the CASE fallback for engines without it."}]
practice: ["conditional-aggregation", "group-by", "having-clause", "count-sum-avg", "nested-aggregations"]
tags: ["GROUP BY", "conditional aggregation"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## What you will learn

- What `GROUP BY` does (split, apply, combine) and which columns you may select afterwards.
- Exactly which rows each aggregate counts: `COUNT(*)` versus `COUNT(col)` versus `COUNT(DISTINCT col)`, and what NULLs do to `SUM` and `AVG`.
- WHERE (filters rows) versus HAVING (filters groups), and the order a grouped query runs in.
- Conditional aggregation: a whole confusion matrix per model in a single scan.
- Why distinct counts and averages cannot be added up across groups, and the two-level query that fixes "average per user".

*Before you start:* lesson 03.01 (SELECT, WHERE, ORDER BY, NULL comparisons) and 03.02 (joins, especially fan-out). The code runs with Python's built-in `sqlite3`; no database server is needed.

## From joined rows to one row per thing

Lesson 03.02 produced detailed rows: one per prediction, request or order. Most ML questions are about *summaries* of them: accuracy per model, latency per region, spend per user. Aggregation turns many rows into one row per thing you care about. It is also where most silently wrong dashboard numbers come from, so the traps get as much space as the syntax.

## The idea in one picture

Here is a tiny evaluation log. Each row is one evaluation run of a model on a data split; one run of model B crashed and has no score (NULL).

| run | model | split | score
| --- | --- | --- | --- |
| 1 | A | test | 0.82
| 2 | A | test | 0.78
| 3 | A | val | 0.90
| 4 | B | test | 0.71
| 5 | B | test | NULL
| 6 | B | val | 0.75
| 7 | C | test | 0.88
| 8 | C | val | 0.64
| 9 | C | test | 0.85

We ask: "per model, how many runs, how many have a score, and what is the mean score?"

```sql
SELECT model, COUNT(*) AS n, COUNT(score) AS n_scored, AVG(score) AS mean_score
FROM evals
GROUP BY model;
-- A | 3 | 3 | 0.833
-- B | 3 | 2 | 0.73
-- C | 3 | 3 | 0.79
```

~~~dg-fig 03.03-groupby.svg
GROUP BY in three moves: the nine rows are split into one bucket per model, each bucket is reduced by the aggregates, and the results are stacked into a three-row table. Note how B's NULL is counted by COUNT(*) but ignored by COUNT(score) and AVG.
~~~

In plain words: **GROUP BY** sorts rows into buckets, one per distinct value of the grouping column. An **aggregate function** (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) reduces a bucket to one value, so the output has one row per bucket. B's mean is $(0.71 + 0.75)/2 = 0.73$: the crashed run is not a zero, it is left out. Whether you want that is a decision, and this lesson is largely about making it consciously.

## Step 1: GROUP BY splits, aggregates reduce

`GROUP BY model, region` partitions the rows that survived WHERE into groups whose `(model, region)` values are equal, and emits one row per group. This is **split-apply-combine**, the same idea as pandas `groupby`.

Each output row stands for a whole group, so SELECT may contain only grouping columns (constant within a group) and aggregates or expressions built from them, such as `SUM(tp) * 1.0 / COUNT(*)`.

Any other column is ambiguous: group A has three runs, so which `run` should appear? PostgreSQL rejects the query (unless you grouped by the table's primary key, which determines every other column). SQLite accepts it and returns the value from an arbitrary row of the group; with a single `MIN` or `MAX` in the query it takes the row holding that extreme, so `SELECT model, run, MAX(score) ... GROUP BY model` gives run 3 for A. Otherwise, treat a bare column as a bug.

Two more grouping facts:

- **NULL keys form one group.** Grouping treats NULLs as equal, unlike `WHERE key = NULL`, which matches nothing (lesson 03.01).
- **No GROUP BY means one group**: the whole table, one output row.

## Step 2: know exactly which rows each aggregate sees

Every aggregate is a numerator, a denominator, or both, and NULLs quietly change them:

| expression | what it computes
| --- | --- |
| `COUNT(*)` | rows in the group, NULLs included
| `COUNT(latency)` | rows where latency is not NULL
| `COUNT(DISTINCT user_id)` | distinct non-NULL user ids
| `SUM(tokens)` | sum of non-NULL values; NULL if there are none
| `AVG(latency)` | `SUM(latency) / COUNT(latency)`: mean of non-NULL values only
| `MIN(x)`, `MAX(x)` | smallest / largest non-NULL value

In symbols, for a group $G$ and column $x$, write $G^{+} = \{\, i \in G : x_i \text{ is not NULL} \,\}$ for the rows where $x$ is present. Then

$$\texttt{AVG}(x) = \frac{\sum_{i \in G^{+}} x_i}{|G^{+}|}, \qquad \frac{\texttt{SUM}(x)}{\texttt{COUNT(*)}} = \frac{\sum_{i \in G^{+}} x_i}{|G|}.$$

$|G|$ is the number of rows in the group, $|G^{+}|$ the number with a value. Same numerator, different denominators; they agree only without NULLs.

~~~dg-fig 03.03-nulls.svg
Ten requests, two of which timed out (NULL latency). AVG divides 935 ms by the 8 completed requests; SUM / COUNT(*) divides by all 10. Neither is "the" mean latency: one describes completed requests, the other treats timeouts as zero, which is wrong in a different way.
~~~

Consequences that show up in real reports:

- Whenever x has NULLs, decide which denominator you mean and write it explicitly.
- A group with only NULLs (or no rows) sums to NULL, not 0. Use `COALESCE(SUM(x), 0)`, or SQLite's `TOTAL(x)`, which returns `0.0`.
- **Integer division truncates.** With integer columns, `SUM(correct) / COUNT(*)` is integer division in SQLite and PostgreSQL, so 6/8 prints as 0. Use `AVG(correct)` (fractional in both) or multiply by `1.0` first.

## Step 3: WHERE filters rows, HAVING filters groups

A grouped query runs in a fixed logical order, regardless of the order in which you write the clauses:

~~~dg-flow Rows are filtered, grouped, aggregated, and then the groups themselves are filtered; SELECT and ORDER BY come last.
FROM rows -(WHERE)-> kept rows -(GROUP BY)-> groups -(aggregates)-> one row per group -(HAVING)-> kept groups -(SELECT, ORDER BY)-> *result
~~~

**WHERE** sees individual rows and runs before grouping, so it cannot use an aggregate (`WHERE COUNT(*) > 30` is an error: no groups exist yet). **HAVING** sees groups after aggregation, so it can: `HAVING COUNT(*) >= 30 AND MIN(score) > 0.8`.

```sql
SELECT model, COUNT(*) AS n, AVG(score) AS mean_score
FROM evals
WHERE split = 'test'              -- row filter: before grouping
GROUP BY model
HAVING COUNT(score) >= 2          -- group filter: after aggregation
ORDER BY mean_score DESC, model;
-- C | 2 | 0.865
-- A | 2 | 0.80      (B is dropped: its test runs have only one score)
```

~~~dg-fig 03.03-pipeline.svg
On a synthetic eval log with 60 models, WHERE cuts 6,063 rows to 1,875 test rows, GROUP BY collapses them to 59 model rows (one model had no test runs), and HAVING keeps the 33 models with at least 30 test runs. Note the log scale.
~~~

A condition without an aggregate, such as `split = 'test'`, could go in either clause. Put it in WHERE: it shrinks the input before the expensive grouping step (1,875 rows instead of 6,063 in the figure) and reads as what it is, a row filter. Many planners move it for you; do not rely on it.

Portability: SQLite accepts a SELECT alias in HAVING (`HAVING n >= 30`); PostgreSQL does not, since HAVING logically runs before SELECT. Repeating the aggregate works everywhere.

## Step 4: conditional aggregation

Often you want several breakdowns of each group side by side: correct and incorrect predictions, clicks per variant. Rather than one query per breakdown, put a condition *inside* the aggregate. The group still contains all its rows; the CASE expression decides what each row contributes:

$$\texttt{SUM(CASE WHEN } c \texttt{ THEN 1 ELSE 0 END)} = \#\{\, i \in G : c_i \text{ is TRUE} \,\}.$$

Each row adds 1 if the condition $c$ holds and 0 otherwise, so the sum is a count of matching rows. This turns long-format data (one row per prediction) into a wide summary (one row per model, one column per cell of the confusion matrix) in one scan of the table. Four such SUMs, one per cell (`label = 1 AND pred = 1` for TP, `label = 0 AND pred = 1` for FP, and so on), plus `AVG(CASE WHEN label = pred THEN 1.0 ELSE 0.0 END)` for accuracy, give a confusion matrix per model; the full query runs in the code section below.

~~~dg-fig 03.03-confusion.svg
One GROUP BY query with four conditional SUMs produces a full confusion matrix for each of three models (400 synthetic predictions each). Accuracy alone ranks them; the matrices show that model_c's errors are mostly false positives.
~~~

PostgreSQL and SQLite (3.30+) also support the standard **FILTER** clause, `COUNT(*) FILTER (WHERE label = 1 AND pred = 1)`, which says it more directly. Two details:

- For SUM, `ELSE 0` matters: without it a group with no matching rows sums to NULL, not 0.
- For AVG, the ELSE branch *is* the denominator. With `ELSE 0.0` you get the fraction of rows where cond holds; `AVG(CASE WHEN cond THEN 1.0 END)` averages only matching rows, so it is always 1 (or NULL when nothing matches).

<div class="ex">
<div class="lab">Worked example</div>

**two models, same accuracy, different errors**

The `predictions` table has 8 rows per model. Model A: labels $(1,1,1,0,0,0,0,0)$, predictions $(1,1,0,1,0,0,0,0)$. Model B: same labels, predictions $(1,1,1,1,1,0,0,0)$.

Model A, row by row: rows 1-2 (label 1, pred 1) give TP = 2; row 3 (1, 0) gives FN = 1; row 4 (0, 1) gives FP = 1; rows 5-8 (0, 0) give TN = 4. Check: $2 + 1 + 1 + 4 = 8$.

Model B. Rows 1-3: TP = 3. Rows 4-5: FP = 2. Rows 6-8: TN = 3. FN = 0. Check: $3 + 2 + 0 + 3 = 8$.

Accuracy is (TP + TN) / rows: A gives $(2 + 4)/8 = 0.75$, B gives $(3 + 3)/8 = 0.75$. Identical.

Precision is TP / (TP + FP): A gives $2/3 \approx 0.667$, B gives $3/5 = 0.6$. Recall is TP / (TP + FN): A gives $2/3$, B gives $3/3 = 1.0$. B catches every positive but raises more false alarms.

The integer trap: `SUM(label = pred) / COUNT(*)` is $6 / 8$ in integer arithmetic, which is 0 for both models. `AVG(label = pred)` returns 0.75. The SQLite run below confirms all of this.

</div>

## Step 5: non-additive metrics and two-level aggregation

Counts and sums are **additive**: weekly rows are the sum of daily rows. Two everyday metrics are not.

**Distinct counts are not additive.** Summing daily active users over a week counts a user once per active day. In a simulation of 1,000 users, daily distinct counts were 276 to 302; they sum to 2,032, but only 671 different people were active that week, a threefold overstatement. Weekly actives must be one `COUNT(DISTINCT user_id)` over the week's raw rows.

~~~dg-fig 03.03-dau.svg
Simulated activity of 1,000 users over one week. Each daily distinct count is correct, but adding them counts a regular user up to seven times: 2,032 versus the true 671 weekly distinct users.
~~~

**An average of averages is weighted wrongly.** A dashboard averages per-city mean session lengths to get a country figure. Smallville has 10 sessions averaging 30 minutes, Midtown 200 averaging 12, Metro 2,000 averaging 8. The average of the three means is $(30 + 12 + 8)/3 = 16.67$ minutes. The true mean over all sessions weights each city by its count:

$$\bar x = \frac{\sum_g n_g \bar x_g}{\sum_g n_g} = \frac{10 \cdot 30 + 200 \cdot 12 + 2000 \cdot 8}{2210} = \frac{18700}{2210} = 8.46.$$

Here $n_g$ is the number of sessions in city $g$ and $\bar x_g$ its mean. Unweighted, ten small-town sessions count as much as two thousand city ones, nearly doubling the answer.

~~~dg-fig 03.03-avgavg.svg
Averaging three city means gives each city one vote (16.67 min); averaging all 2,210 sessions gives each session one vote (8.46 min). Store counts next to means so you can always re-weight.
~~~

**Per-entity, then across entities.** "Average spend per user, by country" needs two levels: total each user, then average the totals. A single `AVG(amount) GROUP BY country` gives the average *order*: UK orders of 10, 10, 10 (user 1) and 30 (user 2) average $60/4 = 15$, yet each user spent 30. The pattern is a subquery (or CTE, lesson 03.05) in FROM:

```sql
SELECT country, AVG(user_total) AS avg_spend_per_user
FROM (SELECT country, user_id, SUM(amount) AS user_total
      FROM orders GROUP BY country, user_id) AS per_user
GROUP BY country;
```

## Aggregation in code

The whole lesson in one runnable script, using Python's standard library. The printed values are from SQLite 3.53.

```python
import sqlite3
con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE predictions(model TEXT, label INTEGER, pred INTEGER);
INSERT INTO predictions VALUES
 ('A',1,1),('A',1,1),('A',1,0),('A',0,1),('A',0,0),('A',0,0),('A',0,0),('A',0,0),
 ('B',1,1),('B',1,1),('B',1,1),('B',0,1),('B',0,1),('B',0,0),('B',0,0),('B',0,0);
""")
print(con.execute("""
SELECT model,
  SUM(CASE WHEN label = 1 AND pred = 1 THEN 1 ELSE 0 END) AS tp,
  SUM(CASE WHEN label = 0 AND pred = 1 THEN 1 ELSE 0 END) AS fp,
  SUM(CASE WHEN label = 1 AND pred = 0 THEN 1 ELSE 0 END) AS fn,
  SUM(CASE WHEN label = 0 AND pred = 0 THEN 1 ELSE 0 END) AS tn,
  AVG(label = pred)                    AS accuracy,
  SUM(label = pred) / COUNT(*)         AS acc_intdiv,
  ROUND(1.0 * COUNT(*) FILTER (WHERE label = 1 AND pred = 1)
            / COUNT(*) FILTER (WHERE pred = 1), 3) AS precision_
FROM predictions GROUP BY model ORDER BY model""").fetchall())
# [('A', 2, 1, 1, 4, 0.75, 0, 0.667), ('B', 3, 2, 0, 3, 0.75, 0, 0.6)]

# SUM without ELSE 0 gives NULL for a group with no matches
print(con.execute("""SELECT model, SUM(CASE WHEN label = 1 AND pred = 0 THEN 1 END)
FROM predictions GROUP BY model ORDER BY model""").fetchall())
# [('A', 1), ('B', None)]
```

(In SQLite a comparison such as `label = pred` evaluates to the integer 1 or 0, which is why `AVG(label = pred)` works. In PostgreSQL it is a boolean; write `AVG((label = pred)::int)` or the CASE form.)

To plot a result, `pd.read_sql_query(sql, con)` returns it as a pandas DataFrame. Aggregate in SQL and pull only the small summary across.

## Checking the result

Aggregates fail silently: they always return a number. A few habits catch most mistakes.

- **Always show the count.** Put `COUNT(*)` next to every mean or rate: 100% accuracy on 3 rows is noise.
- **Reconcile totals.** Per-group counts and sums must add up to the filtered table's; if they add up to more, a join fanned out (lesson 03.02).
- **Show the NULLs.** Report `COUNT(*) - COUNT(x)` next to `AVG(x)`.
- **Check the grain.** Say in words what one output row is ("one per model per day") and check that the GROUP BY columns match exactly.
- **Recompute one group by hand** from its raw rows, as we did for model B.

<div class="ex">
<div class="lab">Worked example</div>

**latency report per model and region**

`requests(model, region, latency_ms)` has 2 million rows; 1% have NULL latency (timeouts). Per model and region select `COUNT(*) AS n`, `COUNT(latency_ms) AS n_ok`, `AVG(latency_ms) AS mean_ok` and `1.0 - 1.0 * COUNT(latency_ms) / COUNT(*) AS timeout_rate`, with `HAVING COUNT(*) >= 100`.

`mean_ok` covers completed requests only, so it understates what users experienced; `timeout_rate` beside it exposes the excluded rows. Overall it is 1%, but one region can be far higher, and only the per-group column shows it.

</div>

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Bare column in SELECT (SQLite) | value from an arbitrary row; changes between runs or versions | group by it or aggregate it (`MIN`, `MAX`, a subquery)
| `SUM(a) / COUNT(*)` on integers | rates of exactly 0 or 1 | `AVG(...)`, or multiply by `1.0` first
| AVG over a column with NULLs | mean looks too good, hides failures | report `COUNT(*) - COUNT(x)` beside it; decide the denominator
| `SUM` over no rows | NULL in a report or a crash downstream | `COALESCE(SUM(x), 0)` or `TOTAL(x)`
| Filtering rows with WHERE instead of counting them | groups with zero matches vanish from the output | conditional aggregation keeps every group, with 0
| Summing daily distinct counts | weekly users 2-7x too high | one `COUNT(DISTINCT ...)` over the full period
| Averaging group averages | small groups dominate | aggregate raw rows, or weight by counts
| Aggregating after a one-to-many join | totals inflated by the fan-out factor | aggregate before joining, or check row counts

On the fifth row: `WHERE score > 0.8 GROUP BY model` returns only A and C, because no row of B survives WHERE; `SUM(CASE WHEN score > 0.8 THEN 1 ELSE 0 END)` keeps B with a 0.

## Going deeper

**Empty input.** Without GROUP BY, an aggregate query returns one row even on an empty table (`COUNT(*)` 0, `SUM` NULL). With GROUP BY it returns zero rows. Code expecting one row breaks on an empty day.

**Which aggregates can be merged?** Parallel and incremental engines aggregate parts of the data and merge the partial results. The data-cube literature names three kinds:

- **Distributive**: the partial results combine with the same function. `COUNT`, `SUM`, `MIN`, `MAX`.
- **Algebraic**: they combine through a fixed-size summary. `AVG` keeps `(sum, count)`; variance keeps `(count, sum, sum of squares)`. So store counts and sums, not means, in summary tables: means can always be rebuilt and re-weighted.
- **Holistic**: no fixed-size summary is enough. Exact `COUNT(DISTINCT)` needs the set of ids, and the median needs the whole distribution. Hence daily distinct counts cannot be combined. Sketches such as HyperLogLog give approximate distinct counts that *can* be merged.

**How the database does it.** Engines use **hash aggregation** (a hash table keyed by group holding running state: one pass, memory proportional to the number of groups) or **sort aggregation** (sort by the keys, then reduce each run of equal keys; cheap when an index already gives that order). `COUNT(DISTINCT x)` costs more because each group must remember the values seen. Lesson 03.07 reads these choices in query plans.

**Several groupings at once.** PostgreSQL supports `GROUPING SETS`, `ROLLUP` and `CUBE` (per-(model, region) rows, per-model subtotals and a grand total in one query); in SQLite you combine GROUP BY queries with `UNION ALL`. A group value *on every row* is a window function (lesson 03.04).

## Key terms

| Term | Meaning
| --- | --- |
| GROUP BY | Splits rows into groups with equal key values; output has one row per group
| Aggregate function | Reduces a group to one value: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
| `COUNT(*)` vs `COUNT(col)` | All rows versus rows where `col` is not NULL
| HAVING | Filter on groups, applied after aggregation; may use aggregates
| Conditional aggregation | An aggregate over a CASE (or with FILTER) that counts or sums only matching rows
| Additive metric | Can be summed across groups to get the total (counts, sums); distinct counts and means are not
| Grain | What one output row represents, set by the GROUP BY columns
| Two-level aggregation | Aggregate per entity in a subquery, then aggregate those results per group

<div class="take">
<div class="lab">Key takeaway</div>

GROUP BY turns rows into one row per group, and every aggregate is only as correct as its denominator: know which rows `COUNT`, `SUM` and `AVG` actually see when NULLs are present, filter rows in WHERE and groups in HAVING, use conditional aggregation for side-by-side breakdowns, and compute distinct counts and averages from raw rows rather than by combining group results.

</div>
