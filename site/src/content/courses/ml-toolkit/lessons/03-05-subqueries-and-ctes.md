---
id: "03.05"
course: "ml-toolkit"
module: "03"
slug: "subqueries-and-ctes"
title: "Subqueries and CTEs"
keyIdeas: ["A subquery is a query used as a value: a scalar (one row, one column), a set for `IN` or `EXISTS`, or a derived table in FROM. A **correlated** subquery refers to the outer row and is logically re-run for each one.", "`x NOT IN (subquery)` returns no rows at all if the subquery yields a single NULL. Use `NOT EXISTS` for anti-joins.", "`WITH name AS (...)` names intermediate results so a long query reads as a pipeline of steps. In PostgreSQL 12+ and SQLite, a CTE is usually inlined into the main query rather than computed separately.", "A recursive CTE has an anchor query `UNION ALL` a recursive query that references the CTE itself; it repeats until a step adds no rows. Use it for hierarchies, lineage and generated sequences, with a depth guard against cycles."]
question: "A query lists users who never bought anything: `SELECT user_id FROM users WHERE user_id NOT IN (SELECT user_id FROM orders)`. It worked for months, then started returning zero rows after guest checkout was launched. Explain the cause and give two correct rewrites."
answer: "Guest checkout probably writes orders with a NULL `user_id`. Once the subquery returns a NULL, `user_id NOT IN (...)` expands to a chain of `<>` comparisons joined by AND, one of which is `user_id <> NULL`, which is UNKNOWN. The condition can then be FALSE or UNKNOWN but never TRUE, so every row is filtered out. Rewrite 1: `SELECT u.user_id FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.user_id)`, where NULL order rows simply never match. Rewrite 2: `SELECT u.user_id FROM users u LEFT JOIN orders o ON o.user_id = u.user_id WHERE o.user_id IS NULL`. Adding `WHERE user_id IS NOT NULL` inside the original subquery also works, but it leaves the trap ready for the next nullable column."
sources: [{"title": "PostgreSQL docs: WITH queries (CTEs)", "url": "https://www.postgresql.org/docs/current/queries-with.html", "note": "Recursive CTEs, cycle detection, and the MATERIALIZED / NOT MATERIALIZED options."}, {"title": "PostgreSQL docs: Subquery expressions", "url": "https://www.postgresql.org/docs/current/functions-subquery.html", "note": "EXISTS, IN, NOT IN, ANY and ALL, including their exact NULL semantics."}, {"title": "SQLite: The WITH clause", "url": "https://www.sqlite.org/lang_with.html", "note": "SQLite's recursive CTE rules, with tree-walking and sequence examples."}]
practice: ["subqueries", "common-table-expressions", "nested-aggregations"]
tags: ["CTE", "recursive CTE", "NOT IN"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## Kinds of subqueries

A subquery is a SELECT nested in another statement. Where it sits decides what it must return:

- **Scalar subquery** in SELECT or WHERE: exactly one column and at most one row. `WHERE score > (SELECT AVG(score) FROM evals)`. If it returns no row, the value is NULL; if it returns two, PostgreSQL raises an error and SQLite silently takes the first.
- **Set subquery** with `IN`, `NOT IN`, `EXISTS`, `NOT EXISTS`, `ANY`, `ALL`.
- **Derived table** in FROM: `FROM (SELECT ...) AS t`, used like any table. This is how you aggregate twice, or filter on a window function.

A subquery is **correlated** when it refers to a column of the outer query:

```sql
SELECT e.*
FROM evals e
WHERE e.score = (SELECT MAX(e2.score) FROM evals e2
                 WHERE e2.benchmark = e.benchmark);   -- best run per benchmark
```

Logically it runs once per outer row. Good planners rewrite many correlated subqueries into joins (decorrelation); when they cannot, and the inner query has no supporting index, the cost becomes outer rows times inner rows. An index on `evals(benchmark, score)` makes each inner lookup a quick seek. The same question can also be answered with a window function, `RANK() OVER (PARTITION BY benchmark ORDER BY score DESC)`, which scans the table once.

## The NOT IN trap

`x NOT IN (a, b, NULL)` expands to `x <> a AND x <> b AND x <> NULL`. The last comparison is UNKNOWN, so the whole condition is never TRUE, and the query returns nothing.

```sql
-- models without a safety eval: returns ZERO rows if any evals.model_id is NULL
SELECT * FROM models WHERE model_id NOT IN (SELECT model_id FROM evals WHERE suite = 'safety');

-- correct regardless of NULLs
SELECT * FROM models m
WHERE NOT EXISTS (SELECT 1 FROM evals e WHERE e.model_id = m.model_id AND e.suite = 'safety');
```

`IN` does not have the problem (a NULL in the list just fails to match), which is why the bug appears only when someone adds the `NOT`. Make `NOT EXISTS` your default for "has no match"; it is also what the LEFT JOIN plus `IS NULL` pattern from the joins lesson expresses.

## Common table expressions

A CTE gives a subquery a name, defined before the main query:

~~~dg-flow A query written as named steps: each CTE reads from the ones before it.
events -(group per user, day)-> daily -(n_events >= 3)-> active -(final SELECT)-> *engaged users per day
~~~

```sql
WITH daily AS (
  SELECT user_id, date(ts) AS d, COUNT(*) AS n_events
  FROM events GROUP BY user_id, date(ts)),
active AS (
  SELECT user_id, d FROM daily WHERE n_events >= 3)
SELECT d, COUNT(DISTINCT user_id) AS engaged_users
FROM active GROUP BY d ORDER BY d;
```

CTEs change readability, not semantics: the same query can be written with nested derived tables. Performance is engine-specific. Before PostgreSQL 12, every CTE was computed once and stored, which blocked the planner from pushing filters into it (an "optimisation fence"). Since version 12, a non-recursive CTE referenced once is inlined; `AS MATERIALIZED` forces the old behaviour when you want a result computed once and reused. SQLite also inlines or materialises at its discretion and accepts the same hints since 3.35.

## Recursive CTEs

`WITH RECURSIVE` lets a CTE refer to itself. It has two parts joined by `UNION ALL`:

1. The **anchor** query produces the starting rows.
2. The **recursive** query references the CTE and produces new rows from the rows added in the previous step.
3. The engine repeats step 2 until it produces no new rows, and the CTE is the union of everything produced.

Model lineage is a natural fit: each fine-tuned model points at its parent.

```sql
WITH RECURSIVE lineage(model_id, parent_id, depth) AS (
  SELECT model_id, parent_id, 0 FROM models WHERE model_id = 'chat-v7'
  UNION ALL
  SELECT m.model_id, m.parent_id, l.depth + 1
  FROM models m JOIN lineage l ON m.model_id = l.parent_id
  WHERE l.depth < 50)                       -- guard against cycles
SELECT * FROM lineage ORDER BY depth;
```

This walks from `chat-v7` up to the base model, one ancestor per step. Without the depth guard, a data error that makes a model its own ancestor loops forever (or until a recursion limit). `UNION` instead of `UNION ALL` deduplicates rows and so stops on exact repeats, but with a growing `depth` column no two rows repeat, so the explicit guard is the reliable choice. Recursive CTEs also generate sequences: a calendar of every date in a range, which you LEFT JOIN against sparse daily data to make missing days explicit.

## Choosing a form

Several forms often answer the same question: "latest eval per model" can be a correlated subquery on `MAX(ts)`, a join to a grouped derived table, or `ROW_NUMBER` in a CTE. Prefer the one that states the intent most directly and scans the data the fewest times. The window version usually wins on both counts, and unlike the `MAX` version it returns exactly one row per model even when two evals share the latest timestamp, provided the ordering is unique.

<div class="ex">
<div class="lab">Worked example</div>

**depth of every model in a registry**

`models(model_id, parent_id)` with 2,000 models; base models have `parent_id` NULL.

Anchor: all base models with depth 0. Recursive step: children of rows found so far, `JOIN models c ON c.parent_id = l.model_id`, with depth + 1.

The result has one row per model reachable from a base model. Any model missing from it sits on a cycle or points at a nonexistent parent, which makes this query a cheap integrity check as well.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

use derived tables and CTEs to build queries as readable steps, `NOT EXISTS` instead of `NOT IN`, and recursive CTEs, always with a depth guard, for hierarchies and generated ranges.

</div>
