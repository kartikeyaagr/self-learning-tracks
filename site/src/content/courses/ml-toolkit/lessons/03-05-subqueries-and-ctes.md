---
id: "03.05"
course: "ml-toolkit"
module: "03"
slug: "subqueries-and-ctes"
title: "Subqueries and CTEs"
keyIdeas: ["A **subquery** is a query used as a value: a scalar (one row, one column) in SELECT or WHERE, a set for `IN` or `EXISTS`, or a derived table in FROM. A **correlated** subquery refers to the outer row and is logically re-run for each one, so without an index it costs outer rows times inner rows.", "`x NOT IN (subquery)` returns no rows at all if the subquery yields a single NULL, because `x <> NULL` is UNKNOWN and an AND with UNKNOWN is never TRUE. Use `NOT EXISTS` (or LEFT JOIN plus `IS NULL`) for anti-joins.", "A scalar subquery that returns no row gives NULL; one that returns several rows is an error in PostgreSQL but silently takes the first row in SQLite, so make it an aggregate or a unique lookup.", "`WITH name AS (...)` names intermediate results so a long query reads as a pipeline of steps. CTEs change readability, not meaning; PostgreSQL 12+ inlines a CTE referenced once, and SQLite inlines or materialises case by case, with `MATERIALIZED` hints in both.", "A recursive CTE has an anchor query `UNION ALL` a recursive query that references the CTE; each step sees only the rows added by the previous step, and it stops when a step adds no rows. Use it for hierarchies, lineage and generated sequences, always with a depth guard against cycles."]
question: "Your model registry table `models(model_id, parent_id)` has 3,000 rows. A nightly job (1) lists models that have never had a safety eval with `WHERE model_id NOT IN (SELECT model_id FROM safety_evals)`, and (2) computes each model's depth from its base model with a recursive CTE that has no depth guard. One night, job (1) reports zero unsafe models and job (2) never finishes. A teammate says both are unrelated infrastructure flukes. What are the likely causes in the data, how would you confirm each, and how would you rewrite both queries so they are robust?"
answer: "Neither is a fluke; both are classic data-triggered failures. (1) Zero rows from `NOT IN` almost always means a NULL reached `safety_evals.model_id` (for example an eval logged before the model was registered, or a failed upstream join). `model_id NOT IN (a, b, NULL)` expands to `model_id <> a AND model_id <> b AND model_id <> NULL`; the last term is UNKNOWN, so the condition is never TRUE and every row is filtered out. Confirm with `SELECT COUNT(*) FROM safety_evals WHERE model_id IS NULL`. Rewrite as an anti-join: `SELECT m.model_id FROM models m WHERE NOT EXISTS (SELECT 1 FROM safety_evals s WHERE s.model_id = m.model_id)`, or the equivalent LEFT JOIN with `WHERE s.model_id IS NULL`; both ignore NULL rows. Separately, investigate why NULL model ids are being written. (2) A recursive CTE stops only when a step adds no rows. If someone set a model's parent to one of its own descendants (or to itself), the walk goes round the cycle forever. Confirm by running the same CTE with a guard such as `WHERE t.depth < 100` and looking for rows at the maximum depth, or by finding models unreachable from any root: run the downward walk from `parent_id IS NULL` with a guard, then list `models` rows with no match in the result (via NOT EXISTS); cycle members and orphans whose parent does not exist both show up there. Rewrite (2) with an explicit guard in the recursive step (`WHERE t.depth < 50`, well above any real lineage length), add a check that alerts if any row reaches the limit, and index `parent_id` so each step is a seek. In PostgreSQL 14+ the `CYCLE` clause can mark cycles directly; in SQLite the depth guard (or `UNION` over key-only rows, or a `LIMIT`) is the tool. Finally, add a constraint or validation at write time so a model cannot be registered as its own ancestor."
sources: [{"title": "PostgreSQL docs: WITH queries (CTEs)", "url": "https://www.postgresql.org/docs/current/queries-with.html", "note": "Recursive CTEs, cycle detection, and the MATERIALIZED / NOT MATERIALIZED options."}, {"title": "PostgreSQL docs: Subquery expressions", "url": "https://www.postgresql.org/docs/current/functions-subquery.html", "note": "EXISTS, IN, NOT IN, ANY and ALL, including their exact NULL semantics."}, {"title": "SQLite: The WITH clause", "url": "https://www.sqlite.org/lang_with.html", "note": "SQLite's recursive CTE rules, with tree-walking and sequence examples."}]
practice: ["subqueries", "common-table-expressions", "nested-aggregations"]
tags: ["CTE", "recursive CTE", "NOT IN"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## What you will learn

- What a subquery is, and how its position decides the shape it must return.
- What makes a subquery **correlated**, and when re-running it per row gets slow.
- Why `NOT IN` silently returns zero rows when the subquery contains a NULL, and why `NOT EXISTS` does not.
- How `WITH` (CTEs) turns a long query into named steps, and what engines do with them.
- How a recursive CTE walks model lineage, and how to stop it looping forever.

*Before you start:* SELECT and WHERE (03.01), joins and the LEFT JOIN plus `IS NULL` pattern (03.02), GROUP BY (03.03) and window functions (03.04). All examples run in Python's built-in `sqlite3` (SQLite 3.53).

## From windows to nested queries

Lesson 03.04 filtered on a window function by wrapping the query and filtering outside. That wrapper was a subquery. When a question needs a value computed from the data itself ("above average", "has no match"), you nest a query or name an intermediate result: **subqueries** and **common table expressions (CTEs)**.

## The idea in one picture

An evaluation table holds eight runs of four models on two benchmarks:

```sql
-- evals(run_id, model_id, benchmark, score)
-- run  model    benchmark  score       run  model    benchmark  score
--  1   base-7b  math        49          5   base-7b  code        36
--  2   chat-v1  math        62          6   chat-v1  code        40
--  3   chat-v2  math        71          7   chat-v2  code        55
--  4   code-v1  math        58          8   code-v1  code        77
```

Which runs are "above average"? `WHERE score > AVG(score)` is not allowed: WHERE sees one row at a time. You need the average as a value first. A **subquery** is exactly that: a SELECT placed inside another statement, whose result is used like a value or a table.

```sql
-- above the overall average (56)
SELECT run_id FROM evals
WHERE score > (SELECT AVG(score) FROM evals);
-- runs 2, 3, 4, 8

-- above the average of *its own* benchmark (math 60, code 52)
SELECT run_id FROM evals e
WHERE score > (SELECT AVG(e2.score) FROM evals e2 WHERE e2.benchmark = e.benchmark);
-- runs 2, 3, 7, 8
```

~~~dg-fig 03.05-avg.svg
The two queries keep different runs. code-v1's 58 on math beats the overall average but not the math average; chat-v2's 55 on code misses the overall average but beats the (harder) code benchmark's 52.
~~~

The first inner query computes one number, once. The second mentions `e.benchmark`, a column of the *outer* row, so its answer changes per row: a **correlated subquery**. For comparing models the second is usually the honest one, because a single global number mixes benchmark difficulty with model quality.

## Step 1: three places, three shapes

Where a subquery sits decides what it must return, like a type check.

| Position | Shape required | Typical use
| --- | --- | --- |
| `WHERE x > (SELECT ...)` or in the SELECT list | **scalar**: one column, at most one row | compare with an average, a max, a threshold
| `WHERE x IN (SELECT ...)`, `EXISTS (SELECT ...)` | **set**: one column (any rows) for IN; anything for EXISTS | "has a match", "has no match"
| `FROM (SELECT ...) AS t` | **derived table**: any rows and columns | aggregate twice, filter on a window result

A **scalar subquery** behaves like a single value. If it returns no row, its value is NULL, so comparisons with it are not TRUE and outer rows drop out. If it returns several rows, PostgreSQL raises an error ("more than one row returned by a subquery used as an expression"), while SQLite silently uses the first row:

```sql
SELECT (SELECT score FROM evals WHERE benchmark = 'math');    -- SQLite: 49 (first row found)
SELECT (SELECT score FROM evals WHERE benchmark = 'vision');  -- NULL (no rows)
```

The SQLite behaviour is the dangerous one: a query that "works" may be picking an arbitrary row. Make such subqueries an aggregate (`MAX`, `AVG`) or a unique lookup.

A **derived table** is a subquery in FROM, treated like any table; give it an alias (PostgreSQL before 16 requires one). Its main use is two levels of aggregation:

```sql
-- average over models of each model's best score
SELECT AVG(best) FROM (
  SELECT model_id, MAX(score) AS best FROM evals GROUP BY model_id
) AS t;
-- bests: base-7b 49, chat-v1 62, chat-v2 71, code-v1 77  ->  64.75
```

## Step 2: correlated subqueries, row by row

Logically, a correlated subquery is a loop: for each outer row, substitute its values into the inner query, run it, and use the result for that row. This describes *meaning*, not how the engine must execute it.

<div class="ex">
<div class="lab">Worked example</div>

**evaluate the correlated filter by hand**

Outer row 4 is (code-v1, math, 58). Substitute `e.benchmark = 'math'`: the inner query averages the math scores $(49 + 62 + 71 + 58)/4 = 240/4 = 60$. Is $58 > 60$? No, so run 4 is dropped.

Outer row 7 is (chat-v2, code, 55). The inner query now averages the code scores $(36 + 40 + 55 + 77)/4 = 208/4 = 52$. Is $55 > 52$? Yes, so run 7 is kept.

Doing this for all eight rows: math runs are compared with 60 (keep 62 and 71), code runs with 52 (keep 55 and 77). Result: runs 2, 3, 7, 8.

The uncorrelated version computes $(240 + 208)/8 = 448/8 = 56$ once and keeps 62, 71, 58 and 77: runs 2, 3, 4, 8.

</div>

The classic correlated pattern is "the row holding the maximum of its group":

```sql
SELECT e.model_id, e.benchmark, e.score
FROM evals e
WHERE e.score = (SELECT MAX(e2.score) FROM evals e2
                 WHERE e2.benchmark = e.benchmark);   -- best run per benchmark
-- ('chat-v2', 'math', 71), ('code-v1', 'code', 77)
```

If the inner query scans the whole table for every outer row, the cost is outer rows times inner rows: quadratic. Good planners **decorrelate** many such queries into a join to a grouped result. SQLite does not here (its plan shows `CORRELATED SCALAR SUBQUERY` over a full `SCAN`); an index on `evals(benchmark, score)` rescues it by making each inner lookup a short seek.

~~~dg-fig 03.05-timing.svg
Measured in SQLite on this machine (timings are machine-dependent; best of three runs). The unindexed correlated subquery grows about four times per doubling of the table, reaching about 14 s at 16,000 rows. With an index, or rewritten as a join or a window function, the same answer takes milliseconds.
~~~

Two rewrites scan the table once: a join to a grouped derived table, and `RANK()` in a CTE (both in the code section). Prefer them, or the index, on big tables. The window version also states intent most directly, and with `ROW_NUMBER()` and a unique ordering it returns exactly one row per group even when scores tie.

## Step 3: set subqueries, EXISTS and the NOT IN trap

Set subqueries ask membership questions, such as which models do or do not appear in a `safety` table of safety-eval runs.

- `x IN (SELECT ...)` is TRUE if `x` equals some value in the list.
- `EXISTS (SELECT ...)` is TRUE if the inner query returns at least one row. What it selects does not matter (people write `SELECT 1`). Keeping outer rows with a match is a **semi-join**; keeping those with none (`NOT EXISTS`) is an **anti-join**.

The trap is `NOT IN`. SQL uses three-valued logic: a comparison with NULL is neither TRUE nor FALSE but UNKNOWN, and WHERE keeps only rows whose condition is TRUE. `x NOT IN (a, b, c)` is shorthand for `x <> a AND x <> b AND x <> c`. If one of those values is NULL, that term is UNKNOWN, and an AND containing an UNKNOWN can be FALSE or UNKNOWN, but never TRUE.

~~~dg-fig 03.05-notin.svg
The safety table holds chat-v2, code-v1, chat-v4 and one NULL. For base-7b, three comparisons are TRUE and the NULL one is UNKNOWN, so NOT IN is UNKNOWN and the row is dropped. NOT EXISTS only asks whether a matching row exists, so it gives the expected TRUE.
~~~

```sql
-- models with no safety run
SELECT model_id FROM models WHERE model_id NOT IN (SELECT model_id FROM safety);
-- 9 models before; after one safety row with model_id NULL is inserted: ZERO rows

SELECT model_id FROM models m
WHERE NOT EXISTS (SELECT 1 FROM safety s WHERE s.model_id = m.model_id);
-- 9 models, NULL or not
```

`IN` does not have the problem: a NULL in the list simply never matches. So the bug appears only when someone adds `NOT`, and only once a NULL reaches the column, often months later. Make `NOT EXISTS` your default for "has no match"; it means the same as the LEFT JOIN plus `IS NULL` pattern from 03.02.

`ANY` and `ALL` (`score > ALL (SELECT ...)`) exist in PostgreSQL but not SQLite, and share the NULL subtleties of `NOT IN`; `MAX` or `MIN` subqueries are more portable.

## Step 4: CTEs, naming the steps

A **common table expression** is a subquery named with `WITH` before the main query. Each CTE can read the ones before it, so the query reads top to bottom like a pipeline.

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

Written with nested derived tables you would read the same query from the middle out. **CTEs change readability, not meaning**: any non-recursive CTE can be rewritten as nested derived tables. They also let you test a pipeline step by step (`SELECT * FROM daily LIMIT 20`) and reference one result twice without copying it.

**What the engine does with a CTE** is engine-specific. Before PostgreSQL 12 every CTE was computed once and stored, blocking outer filters from being pushed into it (an "optimisation fence"). Since 12, a non-recursive, side-effect-free CTE referenced once is inlined; `AS MATERIALIZED` and `AS NOT MATERIALIZED` override that. SQLite decides case by case and has accepted the same two hints since 3.35. In SQLite's `EXPLAIN QUERY PLAN`, a simple filtering CTE disappears into the main query, a grouped CTE used once shows `CO-ROUTINE` (streamed), and the same grouped CTE used twice shows `MATERIALIZE` (stored once).

## Step 5: recursive CTEs

`WITH RECURSIVE` lets a CTE refer to itself, so you can walk a tree or graph of unknown depth. It has two parts joined by `UNION ALL`:

1. The **anchor** query produces the starting rows.
2. The **recursive** query references the CTE and builds new rows from the rows added by the previous step only.
3. The engine repeats step 2 until a step adds no rows. The CTE's result is everything produced, from all steps.

Model lineage is a natural fit: each fine-tuned model points at its parent in `models(model_id, parent_id)`, and base models have `parent_id` NULL.

```sql
WITH RECURSIVE lineage(model_id, parent_id, depth) AS (
  SELECT model_id, parent_id, 0 FROM models WHERE model_id = 'chat-v4'   -- anchor
  UNION ALL
  SELECT m.model_id, m.parent_id, l.depth + 1                           -- recursive step
  FROM models m JOIN lineage l ON m.model_id = l.parent_id
  WHERE l.depth < 50)                                                   -- guard against cycles
SELECT * FROM lineage ORDER BY depth;
-- chat-v4 0, chat-v3 1, chat-v2 2, chat-v1 3, base-7b 4
```

Each step finds the parent of the previous row. At base-7b `parent_id` is NULL, the join matches nothing, and recursion stops. Walking down instead (anchor: all base models; step: `JOIN models c ON c.parent_id = t.model_id`) gives every model's depth.

~~~dg-fig 03.05-recursive.svg
Walking the registry downward. The anchor finds the two base models; each step adds the children of the rows found by the previous step (3, 2, 1, 1 rows), and step 5 adds none, so recursion stops. Three models are never reached.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**depth of every model, and a free integrity check**

The registry has 12 models. Anchor (depth 0): rows with `parent_id IS NULL`, which are base-7b and base-13b. 2 rows.

Step 1 joins children of those: chat-v1, code-v1 (children of base-7b) and med-v1 (child of base-13b). 3 rows, depth 1.

Step 2: chat-v2 and code-v2, depth 2. Step 3: chat-v3. Step 4: chat-v4. Step 5: chat-v4 has no children, so 0 rows and the engine stops.

The result has $2 + 3 + 2 + 1 + 1 = 9$ rows out of 12. The 3 never reached are exp-x (its parent `deleted-model` does not exist) and loop-a and loop-b, which are each other's parents.

So listing `models` rows with no match in the result (with `NOT EXISTS`) is a cheap integrity check for broken lineage.

</div>

**Why the depth guard matters.** Walk upward from loop-a and the step alternates loop-a, loop-b, loop-a, ... forever, so the stopping rule never fires. With `WHERE l.depth < 5` SQLite returns six rows (depths 0 to 5: the guard tests the *previous* row) and stops. `UNION` instead of `UNION ALL` drops repeated rows, which stops key-only recursion, but with a growing `depth` column no two rows repeat, so the explicit guard is the reliable choice. SQLite also accepts a `LIMIT` on the recursive CTE as a hard cap.

Recursive CTEs also generate sequences. LEFT JOIN sparse daily counts onto a calendar so that missing days appear as explicit zeros instead of vanishing:

```sql
WITH RECURSIVE cal(d) AS (
  SELECT '2025-03-01'
  UNION ALL
  SELECT date(d, '+1 day') FROM cal WHERE d < '2025-03-05')
SELECT d FROM cal;      -- 2025-03-01 ... 2025-03-05, five rows
```

## Subqueries and CTEs in code

The opening example end to end, with the plan check:

```python
import sqlite3
con = sqlite3.connect(':memory:')
con.executescript("""
CREATE TABLE evals(run_id INTEGER PRIMARY KEY, model_id TEXT, benchmark TEXT, score REAL);
INSERT INTO evals VALUES
 (1,'base-7b','math',49),(2,'chat-v1','math',62),(3,'chat-v2','math',71),(4,'code-v1','math',58),
 (5,'base-7b','code',36),(6,'chat-v1','code',40),(7,'chat-v2','code',55),(8,'code-v1','code',77);
""")
q = lambda sql: con.execute(sql).fetchall()

print(q("""SELECT run_id FROM evals e
           WHERE score > (SELECT AVG(e2.score) FROM evals e2
                          WHERE e2.benchmark = e.benchmark)"""))
# [(2,), (3,), (7,), (8,)]

print(q("""EXPLAIN QUERY PLAN SELECT run_id FROM evals e
           WHERE score > (SELECT AVG(e2.score) FROM evals e2
                          WHERE e2.benchmark = e.benchmark)"""))
# [(2, 0, 216, 'SCAN e'), (8, 0, 0, 'CORRELATED SCALAR SUBQUERY 1'), (13, 8, 216, 'SCAN e2')]
```

The inner `SCAN e2` sits under the outer `SCAN e`: one full inner scan per outer row. The two one-scan rewrites of "best run per benchmark" agree:

```python
join_version = q("""
  SELECT e.model_id, e.benchmark, e.score FROM evals e
  JOIN (SELECT benchmark, MAX(score) AS best FROM evals GROUP BY benchmark) b
    ON b.benchmark = e.benchmark AND e.score = b.best
  ORDER BY e.benchmark""")
window_version = q("""
  WITH r AS (SELECT model_id, benchmark, score,
                    RANK() OVER (PARTITION BY benchmark ORDER BY score DESC) AS rk
             FROM evals)
  SELECT model_id, benchmark, score FROM r WHERE rk = 1 ORDER BY benchmark""")
print(join_version == window_version, window_version)
# True [('code-v1', 'code', 77.0), ('chat-v2', 'math', 71.0)]
```

If you work in pandas, the correlated "above own group's mean" filter is `df[df.score > df.groupby('benchmark').score.transform('mean')]`: `transform` broadcasts each group's value back to its rows, which is exactly what the correlated subquery computes.

## Checking the result

Nested queries fail quietly, returning plausible but wrong rows. Habits that catch most bugs:

- **Run the inner query alone**, with a literal in place of the outer column (`WHERE e2.benchmark = 'math'`), and check it by hand.
- **Count rows at every CTE step** with `SELECT COUNT(*) FROM step_n`.
- **Test with a NULL** in the key column. If the result collapses to zero rows, look for `NOT IN`.
- **Look for ties**: `= (SELECT MAX ...)` and `RANK()` return two rows when two runs share the top score.
- **Read the plan** (`EXPLAIN QUERY PLAN` in SQLite, `EXPLAIN` in PostgreSQL): a correlated subquery over a `SCAN` warns of quadratic cost.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| `NOT IN (subquery)` on a nullable column | zero rows, suddenly, after new data arrives | `NOT EXISTS`, or LEFT JOIN plus `IS NULL`
| Scalar subquery returns several rows | PostgreSQL error; SQLite silently takes one row | aggregate (`MAX`) or a unique filter
| Scalar subquery returns no row | value is NULL, outer rows vanish | `COALESCE(...)`, or check that the key exists
| Unindexed correlated subquery, big table | time grows ×4 per doubling | index the inner lookup, or use a join or window
| Recursive CTE on data with a cycle | query never finishes | depth guard in the recursive step, or `UNION` on key-only rows
| Assuming a CTE is computed once | slow repeated work, or a filter not pushed down | check the plan; use `MATERIALIZED` / `NOT MATERIALIZED` hints
| Max-per-group with ties | duplicate rows per group | `ROW_NUMBER()` with a unique ORDER BY

## Going deeper

**Decorrelation.** `x > (SELECT AVG(y) FROM t2 WHERE t2.k = t1.k)` equals a join of `t1` with `SELECT k, AVG(y) FROM t2 GROUP BY k`, except for keys with no inner rows. For `COUNT` the subquery returns 0 there, while the join drops the row, so a correct rewrite needs a LEFT JOIN plus `COALESCE(n, 0)`. This classic "count bug" is why optimisers decorrelate cautiously.

**Semi-joins.** `EXISTS` and `IN` never duplicate outer rows when several inner rows match, unlike a join (the fan-out of 03.02).

**How recursion executes.** PostgreSQL keeps a "working table" of the rows added in the last step and runs the recursive query against only those; SQLite uses a queue, one row at a time. That is why aggregates are forbidden in the recursive step: they would see a partial, moving set. PostgreSQL 14 added `SEARCH DEPTH FIRST / BREADTH FIRST` and a `CYCLE` clause that marks and stops cycles for you; SQLite has neither, so the depth guard is the portable choice. Index `parent_id` (walking down) so each step is a seek.

**LATERAL.** PostgreSQL allows `FROM t, LATERAL (SELECT ... WHERE x = t.x LIMIT 3)`, a derived table correlated with earlier tables, for "top 3 per group". SQLite has no LATERAL; use a window function.

**Scope rules.** An unqualified name in a subquery that is not found in the inner table silently resolves to the outer table. If `runs` has no `model_id` column, `WHERE model_id IN (SELECT model_id FROM runs)` is not an error: it compares the outer `model_id` with itself and keeps every non-NULL row. Qualify columns with aliases inside subqueries.

## Key terms

| Term | Meaning
| --- | --- |
| Subquery | A SELECT nested in another statement, used as a value, set or table
| Scalar subquery | A subquery returning one column and at most one row; NULL if it returns none
| Correlated subquery | Refers to the outer row, so it is logically re-run per outer row
| Derived table | A subquery in FROM, used like a table; must usually be aliased
| Semi-join / anti-join | Keep outer rows that have / have no match (`EXISTS` / `NOT EXISTS`)
| CTE | A named subquery defined with `WITH` before the main query
| Recursive CTE | Anchor `UNION ALL` recursive step, repeated until a step adds no rows
| Depth guard | A condition such as `depth < 50` that stops recursion on cyclic data

<div class="take">
<div class="lab">Key takeaway</div>

a subquery lets a query use a value or set it computes itself, and a correlated one is logically re-run per row, so index it or rewrite it as a join or window function; use `NOT EXISTS` rather than `NOT IN`, CTEs to write queries as readable named steps, and recursive CTEs, always with a depth guard, for hierarchies and generated ranges.

</div>
