---
id: "03.02"
course: "ml-toolkit"
module: "03"
slug: "joins"
title: "Joins"
keyIdeas: ["A join pairs rows from two tables that satisfy the ON condition. INNER keeps only matched pairs; LEFT also keeps unmatched left rows, padding the right side with NULLs; FULL keeps unmatched rows from both sides.", "Row counts multiply: a left row that matches three right rows appears three times (**fan-out**). Check key uniqueness before joining, and aggregate the many side to one row per key first, or sums and averages are silently inflated.", "Semi-join (`EXISTS`) asks 'has at least one match' without duplicating rows; anti-join (`NOT EXISTS`, or LEFT JOIN plus `IS NULL`) asks 'has no match'. Avoid `NOT IN` over a subquery that may contain NULL.", "For an outer join, a condition on the right table belongs in ON. Put it in WHERE, or follow the LEFT JOIN with an INNER JOIN on its columns, and the NULL-padded rows are dropped, quietly turning the LEFT JOIN into an INNER JOIN.", "Keys must match exactly: NULL never equals NULL, text and integer ids may or may not match depending on declared types, and composite keys need every column. Check row counts and match rates after every join."]
question: "`runs(run_id, model_id, gpu_hours)` has one row per run and `evals(run_id, benchmark, score)` has 0 to 5 rows per run. An analyst writes `SELECT r.model_id, SUM(r.gpu_hours), AVG(e.score) FROM runs r LEFT JOIN evals e ON e.run_id = r.run_id GROUP BY r.model_id`. Which numbers are wrong, why, and how would you fix the query? How would you have caught the problem with a quick check?"
answer: "The join fans out: a run with k evals appears k times, and once if it has none, thanks to the LEFT JOIN. `SUM(r.gpu_hours)` therefore counts each run's hours once per eval, so models with many evals are overcharged; each model's total is inflated by roughly its average of max(k, 1) per run (for example, runs with hours 2, 3, 10, 6 and 2, 0, 3, 1 evals give 43 instead of 21). `AVG(e.score)` is correct as an average over eval rows (NULLs from runs without evals are ignored), but runs with more benchmarks get more weight, which may or may not be intended. Fix by aggregating each side to one row per key before joining: compute GPU hours per model from `runs` alone, compute the score average per model (or per run first, then per model, if every run should count equally) from `evals` joined to `runs`, then LEFT JOIN the two per-model results on `model_id` so that models without evals keep their hours. Each side is then one-to-one at the join. To catch it early, compare `COUNT(*)` of the joined result with the row count of `runs` (7 vs 4 in the small example means fan-out), or check `SELECT run_id, COUNT(*) FROM evals GROUP BY run_id HAVING COUNT(*) > 1`; and as a sanity check, `SUM(gpu_hours)` over the final result should equal `SUM(gpu_hours)` over `runs`."
sources: [{"title": "PostgreSQL docs: Table expressions (joined tables)", "url": "https://www.postgresql.org/docs/current/queries-table-expressions.html", "note": "Precise definitions of every join type and of ON vs USING vs NATURAL."}, {"title": "PostgreSQL tutorial: Joins between tables", "url": "https://www.postgresql.org/docs/current/tutorial-join.html", "note": "A short, example-driven introduction including outer and self joins."}, {"title": "Use The Index, Luke: The join operation", "url": "https://use-the-index-luke.com/sql/join", "note": "How databases execute joins (nested loops, hash, merge) and what that means for indexing."}, {"title": "SQLite: SELECT", "url": "https://www.sqlite.org/lang_select.html", "note": "SQLite's join syntax and its processing rules."}]
practice: ["left-join", "inner-join", "multiple-joins", "self-join", "cross-join"]
tags: ["joins", "anti-join", "fan-out"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## What you will learn

- What a join computes, row by row, and how INNER, LEFT, RIGHT, FULL and CROSS joins differ in what they keep.
- How to predict the number of output rows before you run a join, and why one-to-many joins silently inflate sums.
- How to ask "has a match?" or "has no match?" with semi-joins and anti-joins, without duplicating rows.
- Why a filter on the right table belongs in ON, not WHERE, when the join is an outer join.
- How keys go wrong (NULLs, types, composite keys) and how databases execute a join.

*Before you start:* SELECT, WHERE, NULL and three-valued logic from lesson 03.01. GROUP BY (lesson 03.03) appears briefly; reading `SUM(x) ... GROUP BY model_id` as "one total per model" is enough.

Lesson 03.01 worked with one table at a time. Real ML data is spread over many: runs, the model registry, evaluation scores, labels. A **join** puts those pieces side by side, and it is where most silent bugs in training sets come from.

## The idea in one picture

Take two tiny tables from an experiment tracker. `runs` has one row per training run; `models` is the model registry.

```sql
-- runs                              -- models
-- run_id | model_id | gpu_hours     -- model_id | name
--   101  |    1     |   2.0         --     1    | tiny-cnn
--   102  |    1     |   3.0         --     2    | resnet18
--   103  |    2     |  10.0         --     3    | vit-small
--   104  |    5     |   6.0
```

Two things are off, as in real data: run 104 points to model 5, which is not in the registry, and vit-small has never been trained.

~~~dg-fig 03.02-join-types.svg
The same two tables joined three ways on model_id. INNER keeps only the three matched pairs; LEFT also keeps orphan run 104 with a NULL name; FULL additionally keeps the untrained vit-small with NULL run columns.
~~~

Read the figure from left to right:

- The **INNER JOIN** returns 3 rows. Run 104 and vit-small have no partner, so both disappear without any warning.
- The **LEFT JOIN** returns 4 rows: every row of the left table (`runs`) survives. Run 104 gets **NULL padding**, meaning the columns that would have come from `models` are filled with NULL.
- The **FULL JOIN** returns 5 rows: every row from both sides, padded on whichever side is missing.

"Left" and "right" just mean written before and after JOIN. Putting a table on the left is a decision that its rows must never be lost.

## Step 1: what a join means

`A JOIN B ON cond` conceptually forms every pair of one row from A with one row from B, then keeps the pairs for which `cond` is TRUE. With 4 runs and 3 models there are $4 \times 3 = 12$ candidate pairs; the condition `m.model_id = r.model_id` is TRUE for 3 of them, so the inner join has 3 rows.

The outer joins add one rule on top: rows that found no partner are added back, once each, with NULLs on the missing side.

| join | keeps | unmatched rows
| --- | --- | --- |
| INNER JOIN | pairs where ON is TRUE | dropped from both sides
| LEFT JOIN | all left rows | left row kept once, right columns NULL
| RIGHT JOIN | all right rows | mirror of LEFT (rarely needed; swap the tables)
| FULL OUTER JOIN | all rows from both | padded with NULLs on the missing side
| CROSS JOIN | every pair | no condition at all

SQLite supports RIGHT and FULL joins since version 3.39; older versions only have INNER, LEFT and CROSS. Plain `JOIN` means INNER JOIN, and `LEFT JOIN` is short for `LEFT OUTER JOIN`.

`JOIN models USING (model_id)` is shorthand for equal `model_id`, output once. `NATURAL JOIN` matches on *every* shared column name, so adding `created_at` to both tables later silently changes it. Prefer explicit ON.

## Step 2: predict the row count (cardinality)

The number of output rows of an inner join is the number of matching pairs. For each left row, count its matches $k_i$; the inner join has

$$\text{rows} = \sum_i k_i,$$

and a LEFT JOIN has $\sum_i \max(k_i, 1)$, because a row with zero matches still appears once. Here $i$ runs over the left rows and $k_i$ is how many right rows satisfy the ON condition for row $i$.

This is why you should always name the **cardinality** of a join before writing it:

- **Many-to-one**: many runs, one model each. Every $k_i \le 1$, so the join never adds rows. This is the safe "lookup" join.
- **One-to-many**: one run, many eval rows. $k_i$ can be 0, 1, 5, so rows multiply. This is called **fan-out**.
- **Many-to-many**: both sides repeat the key. Rows multiply on both sides, and the result is rarely what you want.

~~~dg-flow Joining a one-to-many table repeats the left rows; aggregates over the left columns then overcount.
1,000 runs -(JOIN evals, 3 per run)-> *3,000 rows -(SUM gpu_hours)-> 3x the true total
~~~

Fan-out is fine if you want one row per (run, eval). It becomes a bug when you aggregate a column from the "one" side, now repeated $k_i$ times.

<div class="ex">
<div class="lab">Worked example</div>

**count the rows and the GPU hours by hand**

Add an `evals` table with run 101 evaluated twice (val, test), run 102 never, run 103 three times (val, test, safety) and run 104 once. So $k = (2, 0, 3, 1)$ for runs 101 to 104.

The inner join `runs JOIN evals` has $2 + 0 + 3 + 1 = 6$ rows. The LEFT JOIN has $2 + 1 + 3 + 1 = 7$ rows, because run 102 survives once with NULL eval columns.

The true total GPU time is $2 + 3 + 10 + 6 = 21$ hours.

After the LEFT JOIN, each run's hours appear once per row: $2 \times 2 + 3 \times 1 + 10 \times 3 + 6 \times 1 = 4 + 3 + 30 + 6 = 43$ hours. The query reports more than double the truth.

Grouped by model, model 1 shows $2 \times 2 + 3 = 7$ hours instead of $5$, model 2 shows $30$ instead of $10$, and model 5 shows $6$, correct only by luck (its run has exactly one eval).

SQLite agrees: `SELECT SUM(r.gpu_hours), COUNT(*) FROM runs r LEFT JOIN evals e ON e.run_id = r.run_id` returns `(43.0, 7)`.

</div>

~~~dg-fig 03.02-fanout.svg
Measured on 1,000 synthetic runs: summing GPU hours after joining evals inflates each model's total by roughly its evals per run (never below x1, because a LEFT JOIN keeps runs with no evals once). Nothing errors; the numbers are just wrong.
~~~

## Step 3: verify keys, then pre-aggregate

Two habits prevent almost every fan-out bug. First, **check that the key is unique on the side you believe is "one"**; if this returns any row, every join through the table multiplies:

```sql
SELECT model_id, COUNT(*) FROM models GROUP BY model_id HAVING COUNT(*) > 1;
```

A `PRIMARY KEY` or `UNIQUE` constraint makes the database enforce it. In pandas, `merge(..., validate="many_to_one")` raises a `MergeError` when the right side repeats a key.

Second, **aggregate the many side down to one row per key before joining**. For GPU hours and mean score per model, compute each from its own table (a subquery or CTE, 03.05), then join two results with one row per `model_id` each. Nothing can be repeated.

## Step 4: semi-joins and anti-joins

A **semi-join** keeps left rows that have *at least one* match. An **anti-join** keeps left rows that have *no* match.

```sql
-- semi-join: runs that have at least one eval; one row per run
SELECT r.* FROM runs r
WHERE EXISTS (SELECT 1 FROM evals e WHERE e.run_id = r.run_id);
-- 101, 103, 104

-- anti-join: runs with no safety eval
SELECT r.* FROM runs r
LEFT JOIN evals e ON e.run_id = r.run_id AND e.benchmark = 'safety'
WHERE e.run_id IS NULL;
-- 101, 102, 104
```

The semi-join cannot duplicate rows. `runs JOIN evals` would return run 101 twice and run 103 three times, and the DISTINCT people then add hides the symptom and costs a sort.

The anti-join relies on NULL padding: an unmatched run keeps one row with NULLs on the right, and `e.run_id IS NULL` selects those. Test a column that cannot be NULL in a real match, such as the join key. `NOT EXISTS (SELECT 1 FROM evals e WHERE ...)` expresses the same thing and is usually optimised identically. Avoid `NOT IN (SELECT run_id FROM evals)`: if that subquery returns a single NULL, `x NOT IN (...)` can never be TRUE and the result is empty (the three-valued logic from 03.01).

## Step 5: ON versus WHERE in outer joins

For an inner join it makes no difference whether a condition sits in ON or in WHERE: both just remove pairs. For an outer join the placement changes the result, because the two clauses run at different times:

1. ON decides which right rows count as a match. Unmatched left rows are then padded with NULLs and kept.
2. WHERE runs afterwards, on the padded result, and discards every row where its condition is not TRUE.

So a WHERE condition on a right-table column is UNKNOWN on every padded row and throws it away. On our tables, `runs LEFT JOIN evals ON e.run_id = r.run_id AND e.benchmark = 'safety'` returns 4 rows: run 103 with score 0.95 and three runs with NULL. Move the benchmark test into WHERE and only run 103 is left. In the anti-join above it is worse: `e.benchmark = 'safety'` and `e.run_id IS NULL` can never both hold, so it returns nothing.

The rule to memorise: in a LEFT JOIN, conditions on the **right** table go in ON; conditions on the **left** table go in WHERE. Any WHERE condition on a right-table column that is not an `IS NULL` test removes the padded rows.

<div class="ex">
<div class="lab">Worked example</div>

**datasets and their approved licences**

`datasets(id, name)` has 500 rows; `licenses(dataset_id, license, status)` has zero to three rows per dataset, with `status` either 'approved' or 'pending'.

Wanted: every dataset, with its approved licence if one exists. Correct: `SELECT d.name, l.license FROM datasets d LEFT JOIN licenses l ON l.dataset_id = d.id AND l.status = 'approved';` returns at least 500 rows (more if a dataset has two approved licences).

Wrong: moving `l.status = 'approved'` into WHERE drops every dataset without an approved licence, so the report misses exactly the datasets that need attention.

</div>

~~~dg-fig 03.02-on-vs-where.svg
Measured on 500 synthetic datasets. With the status filter in ON, all 500 datasets survive; in WHERE, the result shrinks to the 215 datasets that have an approved licence, exactly what an INNER JOIN returns.
~~~

## Step 6: chains, self-joins and cross joins

With three or more tables, each join works on the result so far. One trap is specific to chains: an inner join *after* a left join undoes it. Suppose models also have a `team_id`:

```sql
FROM runs r
LEFT JOIN models m ON m.model_id = r.model_id
JOIN teams t ON t.team_id = m.team_id      -- run 104 has m.team_id NULL: dropped
```

Run 104 survived the LEFT JOIN with NULLs, but `t.team_id = NULL` is never TRUE. Every join hanging off an optional table must also be a LEFT JOIN.

A **self-join** joins a table to itself under two aliases. For example, to pair runs with identical hyperparameters but different seeds: `FROM runs a JOIN runs b ON a.config_hash = b.config_hash AND a.run_id < b.run_id`. The `<` keeps each pair once and never pairs a run with itself; on `model_id` our table gives the single pair (101, 102).

A **cross join** forms all combinations, useful for grids (every model times every benchmark) that you compare against actual results to find gaps; on our data it shows tiny-cnn lacks a safety eval. Watch the size: 10,000 users times 10,000 items is 100 million rows.

## Joins in code

Every query in this lesson runs as is in Python's built-in `sqlite3`:

```python
import sqlite3
con = sqlite3.connect(":memory:")
con.executescript('''
CREATE TABLE models(model_id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE runs(run_id INTEGER PRIMARY KEY, model_id INTEGER, gpu_hours REAL);
CREATE TABLE evals(run_id INTEGER, benchmark TEXT, score REAL);
INSERT INTO models VALUES (1,'tiny-cnn'),(2,'resnet18'),(3,'vit-small');
INSERT INTO runs VALUES (101,1,2.0),(102,1,3.0),(103,2,10.0),(104,5,6.0);
INSERT INTO evals VALUES (101,'val',0.81),(101,'test',0.79),(103,'val',0.88),
                         (103,'test',0.86),(103,'safety',0.95),(104,'val',0.74);
''')

q = '''SELECT r.run_id, r.gpu_hours, m.name
       FROM runs r LEFT JOIN models m ON m.model_id = r.model_id'''
print(con.execute(q).fetchall())
# [(101, 2.0, 'tiny-cnn'), (102, 3.0, 'tiny-cnn'), (103, 10.0, 'resnet18'), (104, 6.0, None)]
```

The fan-out-safe version of "GPU hours and mean score per model" aggregates each side first, then joins one-row-per-model results:

```python
q = '''
WITH hrs AS (SELECT model_id, SUM(gpu_hours) AS gpu_hours
             FROM runs GROUP BY model_id),
     sc  AS (SELECT r.model_id, ROUND(AVG(e.score), 3) AS avg_score
             FROM evals e JOIN runs r ON r.run_id = e.run_id
             GROUP BY r.model_id)
SELECT h.model_id, h.gpu_hours, s.avg_score
FROM hrs h LEFT JOIN sc s ON s.model_id = h.model_id'''
print(con.execute(q).fetchall())
# [(1, 5.0, 0.8), (2, 10.0, 0.897), (5, 6.0, 0.74)]
```

The hours are now the true 5, 10 and 6. In pandas, `merge(..., indicator=True)` adds a `_merge` column: 'both' for runs 101 to 103, 'left_only' for run 104.

## Reading and checking the result

After any join in a data pipeline, run three cheap checks before trusting it:

- **Row count.** Compare `COUNT(*)` before and after. A many-to-one LEFT JOIN must return exactly as many rows as the left table. More means duplicate keys on the right; fewer means you wrote INNER or a WHERE clause undid the LEFT.
- **Match rate.** Count padded rows: `SUM(m.model_id IS NULL)` after a LEFT JOIN tells you how many left rows found no partner. A sudden jump usually means a key format changed upstream.
- **Key uniqueness.** `COUNT(*)` versus `COUNT(DISTINCT key)` on the side you expect to be unique.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Totals larger than expected, no error | fan-out from a one-to-many join | aggregate the many side first, or use EXISTS
| Rows vanish after a LEFT JOIN | right-table condition in WHERE, or an INNER JOIN later in the chain | move it into ON; make downstream joins LEFT
| Rows with NULL keys never match | `NULL = NULL` is UNKNOWN | fill or exclude NULL keys deliberately
| Some ids match, others do not | key stored as text on one side, number on the other | cast both sides to one type
| Output explodes to millions of rows | composite key joined on only part of it | join on every key column
| Anti-join returns nothing | `NOT IN` over a subquery containing NULL | use NOT EXISTS or LEFT JOIN + IS NULL
| Join takes minutes on modest tables | no usable index; join on an expression | index the key; join on plain columns

**NULL keys never match**, not even each other, because `NULL = NULL` is UNKNOWN. If you really want NULLs to pair up, SQLite's `IS` operator (`ON a.k IS b.k`) treats two NULLs as equal; PostgreSQL spells it `IS NOT DISTINCT FROM`.

**Types must agree.** Text ids on one side and integers on the other fail silently. PostgreSQL rejects `text = integer` with an error, which is at least loud. SQLite applies **type affinity** conversions whose outcome depends on how each column was declared. Comparing an `INTEGER` column with a `TEXT` column, SQLite converts the text to a number first, so `'42'` and even `' 7'` match 42 and 7. But if both columns were created without a declared type (`CREATE TABLE t(id)`, common in quick imports), the integer 42 and the text `'42'` do not match at all. Cast both sides to one type explicitly and do not rely on the declared types.

**Composite keys** need every part: joining hourly features on `user_id` alone, when the key is `(user_id, hour)`, turns 24 rows per user-day into 576. Joining on expressions (`ON lower(a.email) = lower(b.email)`) works but usually prevents index use.

## Going deeper: how a database executes a join

"All pairs, then filter" costs $n \times m$ comparisons. Engines use three strategies instead:

- **Nested loop join**: for each left row, find its matches; $O(nm)$ by scanning, about $O(n \log m)$ with a B-tree index on the right key.
- **Hash join**: build a hash table on the smaller input's key and stream the other through it; roughly $O(n + m)$, equality conditions only.
- **Merge join**: sort both inputs on the key, then walk them in step; linear if already sorted.

PostgreSQL chooses among all three based on table statistics. SQLite implements every join as a nested loop, but its query planner builds a temporary **automatic index** on the inner table's join key when no suitable index exists, and recent versions add a Bloom filter in front of it. You can see the choice with `EXPLAIN QUERY PLAN`; on our tables it reports `SEARCH e USING AUTOMATIC COVERING INDEX (run_id=?)`.

~~~dg-fig 03.02-timing.svg
Join time measured in SQLite on this machine (timings are machine-dependent; shapes are what matter). With automatic indexing switched off the nested loop grows quadratically; the automatic index and a permanent index on evals(run_id) grow close to linearly.
~~~

In the measured run, at 4,000 runs and 12,000 eval rows the plain nested loop took about 4.8 seconds, the automatic index about 14 ms (it rebuilds the index on every query) and a permanent index under 1 ms. An index on the join key of the large table is the most effective fix (more in 03.07).

The optimiser may reorder inner joins freely (they are associative and commutative); outer joins restrict this. A semi-join can stop after the first match per left row, which is why it can beat a join plus DISTINCT.

## Key terms

| Term | Meaning
| --- | --- |
| Join | Combining rows of two tables into pairs that satisfy the ON condition
| INNER / LEFT / FULL join | Keep only matched pairs / also unmatched left rows / also unmatched rows from both sides
| NULL padding | NULLs filled into the missing side's columns for an unmatched row of an outer join
| Cardinality | The expected relationship between keys: one-to-one, many-to-one, one-to-many, many-to-many
| Fan-out | Row multiplication when a left row matches several right rows
| Semi-join | Keep left rows with at least one match, without duplicating them (`EXISTS`)
| Anti-join | Keep left rows with no match (`NOT EXISTS`, or LEFT JOIN plus `IS NULL`)
| Self-join / cross join | A table joined to itself under two aliases / every pair with no condition
| Hash, merge, nested loop join | The physical algorithms a database uses to execute a join

<div class="take">
<div class="lab">Key takeaway</div>

before each join, write down the expected cardinality and check it; aggregate the many side before joining when you need sums; filter right-table rows in ON for outer joins; use EXISTS and NOT EXISTS when you need a yes-or-no test rather than columns.

</div>
