---
id: "03.03"
course: "ml-toolkit"
module: "03"
slug: "aggregation"
title: "Aggregation"
keyIdeas: ["GROUP BY produces one output row per distinct combination of the grouping columns; every other selected column must be inside an aggregate. NULL keys form one group of their own.", "`COUNT(*)` counts rows, `COUNT(col)` counts non-NULL values, `COUNT(DISTINCT col)` counts distinct non-NULL values. `AVG` and `SUM` skip NULLs, and `SUM` over nothing is NULL, not 0.", "WHERE filters rows before grouping; HAVING filters groups after aggregation. Push every condition that does not need an aggregate into WHERE.", "Conditional aggregation (`SUM(CASE WHEN ... THEN 1 ELSE 0 END)` or `COUNT(*) FILTER (WHERE ...)`) computes several breakdowns in one pass, such as a whole confusion matrix per model.", "Distinct counts and averages are not additive: daily active users do not sum to weekly ones, and an average of per-group averages is not the overall average."]
question: "A dashboard computes weekly active users as the sum of seven daily `COUNT(DISTINCT user_id)` values, and the average session length per country as the average of per-city averages. Explain what each gets wrong, and write the shape of the correct query for each."
answer: "Summing daily distinct counts counts a user once for every day they were active, so a user active all week contributes 7. Weekly actives must be one distinct count over the whole week: `SELECT week, COUNT(DISTINCT user_id) FROM events GROUP BY week`, with week derived from the timestamp. The average of per-city averages gives each city equal weight regardless of how many sessions it had, so a small city with unusual sessions distorts the country figure. The correct figure aggregates raw sessions directly, `SELECT country, AVG(session_length) FROM sessions GROUP BY country`, or, if only per-city summaries exist, weights by count: `SUM(city_avg * city_n) / SUM(city_n)`. Both mistakes come from treating non-additive statistics as if they could be combined from group results."
sources: [{"title": "PostgreSQL tutorial: Aggregate functions", "url": "https://www.postgresql.org/docs/current/tutorial-agg.html", "note": "GROUP BY, HAVING and the WHERE vs HAVING distinction, with examples."}, {"title": "PostgreSQL docs: Aggregate functions", "url": "https://www.postgresql.org/docs/current/functions-aggregate.html", "note": "Every aggregate, NULL behaviour, and FILTER and ordered-set aggregates."}, {"title": "SQLite: Built-in aggregate functions", "url": "https://www.sqlite.org/lang_aggfunc.html", "note": "SQLite's aggregates, including sum() vs total() on empty input."}, {"title": "Modern SQL: The FILTER clause", "url": "https://modern-sql.com/feature/filter", "note": "Standard conditional aggregation and the CASE fallback for engines without it."}]
practice: ["conditional-aggregation", "group-by", "having-clause", "count-sum-avg", "nested-aggregations"]
tags: ["GROUP BY", "conditional aggregation"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## GROUP BY collapses rows

`GROUP BY model, region` partitions the rows that survive WHERE into groups with equal `(model, region)` values and emits one row per group. In the SELECT list you may use the grouping columns and aggregates over the group (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, plus engine-specific ones such as `percentile_cont` in PostgreSQL or `group_concat` in SQLite).

Selecting a column that is neither grouped nor aggregated is an error in PostgreSQL. SQLite accepts it and returns the value from an arbitrary row of the group (or, with a single `MIN` or `MAX`, from the row holding that extreme). Treat that as a bug unless you meant exactly that. For grouping purposes NULLs compare equal, so all rows with a NULL key land in one group.

~~~dg-flow Rows are filtered, grouped, aggregated, and then the groups themselves are filtered.
Rows -(WHERE)-> kept rows -(GROUP BY)-> groups -(aggregates)-> one row per group -(HAVING)-> *result
~~~

## Aggregates and NULL

| expression | what it counts or computes
| --- | --- |
| `COUNT(*)` | rows in the group, NULLs included
| `COUNT(latency)` | rows where latency is not NULL
| `COUNT(DISTINCT user_id)` | distinct non-NULL user ids
| `AVG(latency)` | mean of the non-NULL values only
| `SUM(tokens)` | sum of non-NULL values; NULL if there are none

The consequences:

- `AVG(x)` and `SUM(x) / COUNT(*)` differ whenever x has NULLs. Decide which denominator you mean.
- A group where every value is NULL sums to NULL. Wrap with `COALESCE(SUM(x), 0)` when a missing total should read as zero. SQLite also has `TOTAL(x)`, which returns 0.0.
- **Integer division**: with integer columns, `SUM(correct) / COUNT(*)` truncates to 0 or 1 in both SQLite and PostgreSQL. Use `AVG(correct)`, which returns a fractional result in both, or multiply by 1.0 before dividing.

## WHERE versus HAVING

WHERE sees individual rows and runs before grouping. HAVING sees groups and can use aggregates: `HAVING COUNT(*) >= 30 AND MIN(score) > 0.8`. A condition that does not involve an aggregate (`split = 'test'`) can go in either place, but in WHERE it shrinks the input before the expensive grouping step, and it reads as what it is: a row filter. Many planners move such conditions into WHERE for you; some do not.

```sql
SELECT model, COUNT(*) AS n, AVG(score) AS mean_score
FROM evals
WHERE split = 'test'
GROUP BY model
HAVING COUNT(*) >= 30 AND MIN(score) >= 0.7
ORDER BY mean_score DESC, model;
```

## Conditional aggregation

An aggregate over a CASE expression counts or sums only the rows that meet a condition, while the group still contains all rows. This turns long-format data into a wide summary in one scan:

```sql
SELECT model,
       SUM(CASE WHEN label = 1 AND pred = 1 THEN 1 ELSE 0 END) AS tp,
       SUM(CASE WHEN label = 0 AND pred = 1 THEN 1 ELSE 0 END) AS fp,
       SUM(CASE WHEN label = 1 AND pred = 0 THEN 1 ELSE 0 END) AS fn,
       SUM(CASE WHEN label = 0 AND pred = 0 THEN 1 ELSE 0 END) AS tn,
       AVG(CASE WHEN label = pred THEN 1.0 ELSE 0.0 END)      AS accuracy
FROM predictions
GROUP BY model;
```

PostgreSQL and SQLite (3.30 and later) also support the standard `FILTER` clause, `COUNT(*) FILTER (WHERE label = 1 AND pred = 1)`, which says the same thing more directly. Two details: `ELSE 0` matters for SUM (without it, a group with no matching rows gives NULL rather than 0), and for AVG the ELSE branch defines the denominator. `AVG(CASE WHEN cond THEN 1.0 END)` averages only matching rows and is always 1.

## Distinct counts and nested aggregation

Some metrics do not combine across groups:

- **Distinct counts are not additive.** Summing daily active users over a week counts a user once per active day. Weekly actives must be `COUNT(DISTINCT user_id)` over the week's rows.
- **An average of averages is weighted wrongly.** If one country has 10 users and another 10,000, averaging the two per-country means gives the small country as much weight as the large one. Compute the overall mean from the raw rows, or weight by the counts.
- **Per-entity then across entities.** "Average spend per user, by country" needs two levels: first `SUM(amount)` per user, then `AVG` of those totals per country. A single `AVG(amount) GROUP BY country` is the average *transaction*, not the average *user*.

The two-level pattern is a subquery or CTE in FROM:

```sql
SELECT country, AVG(user_total) AS avg_spend_per_user
FROM (SELECT country, user_id, SUM(amount) AS user_total
      FROM orders GROUP BY country, user_id) AS per_user
GROUP BY country;
```

<div class="ex">
<div class="lab">Worked example</div>

**latency report per model and region**

`requests(model, region, latency_ms)` with 2 million rows, of which 1% have NULL latency (timeouts).

Per model and region, select `COUNT(latency_ms) AS n_ok` and `AVG(latency_ms) AS mean_ok` next to the row count as `n`, and keep groups with at least 100 requests in HAVING.

The `timeout_rate` is `1.0 - 1.0 * n_ok / n`, computed in an outer query or by repeating the aggregates. Overall it is 1%, but a single region can be far higher.

`mean_ok` is the mean over completed requests only, so it understates user-facing latency; `timeout_rate` makes the excluded rows visible. The HAVING clause hides groups too small for the numbers to mean much.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

for every aggregate, ask which rows are in the denominator and what NULLs do to it; filter rows with WHERE, groups with HAVING, and compute non-additive metrics from the raw rows.

</div>
