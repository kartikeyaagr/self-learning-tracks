---
id: "03.01"
course: "ml-toolkit"
module: "03"
slug: "select-filtering-and-sorting"
title: "SELECT, filtering and sorting"
keyIdeas: ["A query is evaluated in a logical order that differs from how it is written: FROM, WHERE, GROUP BY, HAVING, SELECT, DISTINCT, ORDER BY, LIMIT. That order explains which names are visible where.", "`AND` binds tighter than `OR`. Parenthesise every mixed condition, or `split = 'val' OR split = 'test' AND score > 0.9` will keep every validation row.", "SQL uses three-valued logic: any comparison with NULL is UNKNOWN, and WHERE keeps only TRUE rows. `x <> 'a'` silently drops the rows where x is NULL.", "Without ORDER BY, row order is undefined; with ties, it is still undefined. Add a unique tie-breaker, and remember databases disagree on where NULLs sort."]
question: "A table `requests(id, model, status, latency_ms)` has NULL latency for timed-out requests. `SELECT COUNT(*) FROM requests WHERE latency_ms < 500 OR latency_ms >= 500` returns a smaller number than the table's row count. Explain why, and say what this means for a query reporting the share of requests that were 'fast' (under 500 ms)."
answer: "For a row with NULL latency both comparisons are UNKNOWN, and UNKNOWN OR UNKNOWN is UNKNOWN, so WHERE drops the row. The two conditions look complementary, but under three-valued logic they cover only rows with a known latency; the missing count is exactly the timed-out requests. For the 'fast share', the denominator matters. If you compute it over rows with `latency_ms IS NOT NULL`, timeouts disappear and the service looks faster than it is. Timeouts are the slowest requests, so they should normally count as not fast: compute the numerator with `latency_ms < 500` (which is never TRUE for NULL) and the denominator with `COUNT(*)` over all requests, and report the timeout count separately so the choice is visible."
sources: [{"title": "PostgreSQL docs: SELECT", "url": "https://www.postgresql.org/docs/current/sql-select.html", "note": "The reference for clause order, ORDER BY with NULLS FIRST/LAST, LIMIT and DISTINCT."}, {"title": "SQLite: NULL handling in SQLite versus other engines", "url": "https://www.sqlite.org/nulls.html", "note": "A table of how different databases treat NULL in DISTINCT, UNIQUE and sorting."}, {"title": "Modern SQL: Three-valued logic", "url": "https://modern-sql.com/concept/three-valued-logic", "note": "Truth tables for UNKNOWN and how they surface in WHERE, CHECK and NOT IN."}, {"title": "PostgreSQL docs: Comparison functions and operators", "url": "https://www.postgresql.org/docs/current/functions-comparison.html", "note": "BETWEEN, IS DISTINCT FROM and the NULL predicates."}]
practice: ["where-clauses", "order-by", "basic-select", "distinct-values", "limit-and-offset"]
tags: ["NULL", "three-valued logic"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## Why SQL for ML

Most training data starts life in a database or warehouse: event logs, labels, model registries, evaluation results. Pulling a whole table into pandas and filtering there is slow at scale and moves data you do not need. This module teaches enough SQL to build datasets and analyses where the data lives. The examples are standard SQL that runs in SQLite (the dialect of the practice problems) and PostgreSQL; differences are called out where they matter.

## The anatomy of a query

A `SELECT` is written in one order and evaluated, logically, in another:

~~~dg-flow Logical evaluation order of a SELECT; each step only sees what earlier steps produced.
FROM / JOIN -> WHERE -> GROUP BY -> HAVING -> *SELECT -> ORDER BY -> LIMIT
~~~

```sql
SELECT run_id, model, steps * batch_size AS examples_seen
FROM training_runs
WHERE status = 'finished' AND (lr >= 1e-4 OR warmup_steps > 0)
ORDER BY examples_seen DESC, run_id
LIMIT 10;
```

Two consequences of that order trip people up:

- A column alias defined in SELECT is **not** visible in WHERE in standard SQL, because WHERE runs first. PostgreSQL rejects `WHERE examples_seen > 1e6`; SQLite accepts it as an extension. Repeat the expression or move it into a subquery.
- The alias **is** visible in ORDER BY, which runs after SELECT.

The database is free to execute the query in any physical order it likes (using an index first, say). The logical order defines only what the answer must be.

## WHERE logic

`WHERE` keeps the rows for which the condition evaluates to TRUE. The operators: comparisons (`=`, `<>`, `<`, `>=`), `BETWEEN a AND b` (inclusive at both ends), `IN (...)`, `LIKE 'gpt-%'` (`%` matches any run of characters, `_` exactly one), `IS NULL`, combined with `AND`, `OR`, `NOT`.

Precedence is `NOT`, then `AND`, then `OR`. So

```sql
WHERE split = 'val' OR split = 'test' AND confidence > 0.9
```

means `split = 'val' OR (split = 'test' AND confidence > 0.9)`: every validation row comes through regardless of confidence. Parentheses cost nothing; use them every time AND and OR mix.

With timestamps stored as text (`'2025-03-01 14:00:00'`), prefer half-open ranges: `ts >= '2025-03-01' AND ts < '2025-04-01'`. `BETWEEN '2025-03-01' AND '2025-03-31'` misses everything after midnight on the 31st, because `'2025-03-31 09:00:00'` sorts after `'2025-03-31'`.

## NULL and three-valued logic

NULL means "unknown", and any comparison with an unknown is itself UNKNOWN, not TRUE or FALSE. WHERE discards both FALSE and UNKNOWN rows.

| expression | value when x is NULL | row kept by WHERE?
| --- | --- | --- |
| `x = 1` | UNKNOWN | no
| `x <> 1` | UNKNOWN | no
| `x = NULL` | UNKNOWN, always | no, even for NULL rows
| `x IS NULL` | TRUE | yes
| `NOT (x = 1)` | UNKNOWN | no
| `x <> 1 OR x IS NULL` | TRUE | yes

The practical rules:

- Test for missing values with `IS NULL` / `IS NOT NULL`, never `= NULL`.
- A filter like `label <> 'spam'` drops unlabelled rows as well. If you meant "not known to be spam", write `label IS NOT 'spam'` in SQLite or `label IS DISTINCT FROM 'spam'` in PostgreSQL (and in SQLite since version 3.39).
- `COALESCE(x, default)` substitutes a value for NULL. It is a modelling decision: filling missing latency with 0 makes slow requests look fast.
- Arithmetic with NULL gives NULL: `price * qty` is NULL if either is.

`NOT IN` with a NULL inside the list is the worst version of this trap; it gets its own section in the subqueries lesson.

## ORDER BY, LIMIT and DISTINCT

A table has no inherent order. Without ORDER BY the database may return rows in any order, and that order can change between runs, after an index is added, or when the data grows. With ORDER BY on a column that has ties, the order among tied rows is still arbitrary, so `ORDER BY score DESC LIMIT 10` can return a different tenth row each time. Add a unique column as a final tie-breaker: `ORDER BY score DESC, model_id`.

NULLs sort differently by database: SQLite and MySQL put NULLs **first** in ascending order, PostgreSQL puts them **last**. State it explicitly with `NULLS LAST` (PostgreSQL, and SQLite since 3.30).

`LIMIT n OFFSET k` skips k rows and returns the next n. It is fine for small pages but the database still produces and discards the k skipped rows; the performance lesson shows the keyset alternative. `SELECT DISTINCT a, b` removes duplicate rows over the listed columns, and NULLs count as equal to each other for this purpose.

<div class="ex">
<div class="lab">Worked example</div>

**confident mistakes on the validation split**

Table `predictions(id, split, label, pred, confidence)`. You want the 20 most confident wrong predictions on validation data, ignoring rows with no label.

`SELECT id, label, pred, confidence FROM predictions WHERE split = 'val' AND label IS NOT NULL AND pred <> label ORDER BY confidence DESC, id LIMIT 20;`

`label IS NOT NULL` is technically redundant, because `pred <> label` is UNKNOWN when `label` is NULL and the row is dropped anyway. Writing it states the intent and survives someone later rewriting the condition with COALESCE.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

read a query in its logical order, parenthesise mixed AND/OR, treat every comparison as possibly UNKNOWN, and make every ORDER BY total with a unique tie-breaker.

</div>
