---
id: "03.02"
course: "ml-toolkit"
module: "03"
slug: "joins"
title: "Joins"
keyIdeas: ["A join pairs rows from two tables that satisfy the ON condition. INNER keeps only matched pairs; LEFT also keeps unmatched left rows, padding the right side with NULLs; FULL keeps unmatched rows from both sides.", "Row counts multiply: a left row that matches three right rows appears three times. Check key uniqueness before joining, or sums and averages are silently inflated.", "Semi-join (`EXISTS`) asks 'has at least one match' without duplicating rows; anti-join (`NOT EXISTS`, or LEFT JOIN plus `IS NULL`) asks 'has no match'.", "For an outer join, a condition on the right table belongs in ON. Put it in WHERE and the NULL-padded rows fail it, quietly turning the LEFT JOIN into an INNER JOIN."]
question: "`runs(run_id, model_id, gpu_hours)` has one row per run and `evals(run_id, benchmark, score)` has 0 to 5 rows per run. An analyst writes `SELECT r.model_id, SUM(r.gpu_hours), AVG(e.score) FROM runs r LEFT JOIN evals e ON e.run_id = r.run_id GROUP BY r.model_id`. Which numbers are wrong, why, and how would you fix the query?"
answer: "The join fans out: a run with k evals appears k times (and once if it has none, thanks to the LEFT JOIN). `SUM(r.gpu_hours)` therefore counts each run's hours once per eval, so models with many evals are overcharged; the total is inflated by the eval count per run. `AVG(e.score)` is correct as an average over eval rows (NULLs from runs without evals are ignored), but runs with more benchmarks get more weight, which may or may not be intended. Fix by aggregating each side to one row per key before joining: compute gpu hours per model from `runs` alone, compute the score average per model (or per run first, then per model) from `evals` joined to `runs`, then join the two per-model results on `model_id`. Each side is then one-to-one at the join."
sources: [{"title": "PostgreSQL docs: Table expressions (joined tables)", "url": "https://www.postgresql.org/docs/current/queries-table-expressions.html", "note": "Precise definitions of every join type and of ON vs USING vs NATURAL."}, {"title": "PostgreSQL tutorial: Joins between tables", "url": "https://www.postgresql.org/docs/current/tutorial-join.html", "note": "A short, example-driven introduction including outer and self joins."}, {"title": "Use The Index, Luke: The join operation", "url": "https://use-the-index-luke.com/sql/join", "note": "How databases execute joins (nested loops, hash, merge) and what that means for indexing."}, {"title": "SQLite: SELECT", "url": "https://www.sqlite.org/lang_select.html", "note": "SQLite's join syntax and its processing rules."}]
practice: ["left-join", "inner-join", "multiple-joins", "self-join", "cross-join"]
tags: ["joins", "anti-join", "fan-out"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## What a join computes

Conceptually, `A JOIN B ON cond` forms every pair of a row from A with a row from B and keeps the pairs where `cond` is TRUE. Databases never actually build that full product when they can avoid it (they use hash tables, sorted merges or index lookups), but the definition tells you what the result must contain.

| join | keeps | unmatched rows
| --- | --- | --- |
| INNER JOIN | pairs where ON is TRUE | dropped from both sides
| LEFT JOIN | all left rows | left row kept once, right columns NULL
| RIGHT JOIN | all right rows | mirror of LEFT (rarely needed; swap the tables)
| FULL OUTER JOIN | all rows from both | padded with NULLs on the missing side
| CROSS JOIN | every pair | no condition at all

SQLite supports RIGHT and FULL joins since version 3.39; older versions only have INNER, LEFT and CROSS.

## Cardinality and fan-out

The number of output rows is the number of matching pairs. That is harmless for a many-to-one join (each training run joined to its one model) and dangerous for one-to-many (each run joined to its many evaluation results).

~~~dg-flow Joining a one-to-many table repeats the left rows; aggregates over the left columns then overcount.
1,000 runs -(JOIN evals, 3 per run)-> *3,000 rows -(SUM gpu_hours)-> 3x the true total
~~~

Before writing a join, state the expected relationship (one-to-one, many-to-one, one-to-many) and verify it:

```sql
SELECT model_id, COUNT(*) FROM models GROUP BY model_id HAVING COUNT(*) > 1;
```

If this returns rows, the "lookup" table has duplicates and every join through it multiplies. When you need aggregates from both sides of a one-to-many relationship, aggregate the many side first (in a subquery or CTE) and join the one-row-per-key result.

## Join keys

- **NULL keys never match**, not even each other, because `NULL = NULL` is UNKNOWN. Rows with a NULL key vanish from an inner join and get NULL padding in a left join.
- **Types must agree.** An id stored as text in one table and as an integer in the other is a classic silent failure. PostgreSQL rejects `text = integer` with an error, which is at least loud. SQLite applies type-affinity conversions whose outcome depends on how each column was declared, so the join can match some rows and miss others (`' 42'` with a leading space never matches). Cast both sides to one type explicitly.
- **Composite keys** need every part: joining hourly features on `user_id` alone, when the key is `(user_id, hour)`, pairs every hour with every other hour.
- Joining on expressions (`ON lower(a.email) = lower(b.email)`) works but usually prevents index use.

## Semi-joins and anti-joins

Often you want to filter by the existence of a match, not to fetch columns from the other table.

```sql
-- semi-join: models that have at least one eval; one row per model
SELECT m.* FROM models m
WHERE EXISTS (SELECT 1 FROM evals e WHERE e.model_id = m.model_id);

-- anti-join: models with no safety eval
SELECT m.* FROM models m
LEFT JOIN evals e ON e.model_id = m.model_id AND e.suite = 'safety'
WHERE e.model_id IS NULL;
```

The semi-join cannot duplicate rows, however many evals match, which an INNER JOIN would. The anti-join pattern relies on NULL padding: a model with no match keeps one row with NULLs on the right, and `e.model_id IS NULL` selects exactly those. Test a right-side column that cannot be NULL in a real match, such as the join key. `NOT EXISTS` expresses the same thing and is usually optimised identically.

## ON versus WHERE in outer joins

For an inner join it makes no difference whether a condition is in ON or WHERE. For an outer join it changes the result. In the anti-join above, `e.suite = 'safety'` sits in ON: it restricts which evals count as a match, and every model still survives the LEFT JOIN. Move it to WHERE and the query asks for rows where `e.suite = 'safety'` and `e.model_id IS NULL` simultaneously, which is impossible, so the result is empty.

The general rule: in a LEFT JOIN, conditions on the **right** table go in ON; conditions on the **left** table go in WHERE. Any WHERE condition on a right-table column that is not an `IS NULL` test removes the NULL-padded rows and makes the query an inner join in disguise.

## Self-joins and cross joins

A **self-join** joins a table to itself under two aliases, for example to pair runs with identical hyperparameters but different seeds: `FROM runs a JOIN runs b ON a.config_hash = b.config_hash AND a.run_id < b.run_id`. The `<` keeps each unordered pair once and excludes pairing a run with itself. A **cross join** deliberately forms all combinations, which is useful for building grids (every model x every benchmark) that you then LEFT JOIN against actual results to find what is missing.

<div class="ex">
<div class="lab">Worked example</div>

**datasets and their approved licences**

`datasets(id, name)` has 500 rows; `licenses(dataset_id, license, status)` has several rows per dataset, with `status` either 'approved' or 'pending'.

Wanted: every dataset, with its approved licence if one exists. Correct: `SELECT d.name, l.license FROM datasets d LEFT JOIN licenses l ON l.dataset_id = d.id AND l.status = 'approved';` returns at least 500 rows (more if a dataset has two approved licences).

Wrong: moving `l.status = 'approved'` into WHERE drops every dataset without an approved licence, so the report misses exactly the datasets that need attention.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

before each join, write down the expected cardinality and check it; filter right-table rows in ON for outer joins; use EXISTS and NOT EXISTS when you need a yes-or-no test rather than columns.

</div>
