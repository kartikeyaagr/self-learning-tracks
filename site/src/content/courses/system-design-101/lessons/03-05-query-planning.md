---
id: "03.05"
course: "system-design-101"
module: "03"
slug: "query-planning"
title: "Query planning"
keyIdeas: ["The query planner turns SQL into an execution plan: which indexes to use, join order, join algorithm (nested loop, hash, merge), and whether to sort.", "It picks the plan with the lowest estimated cost, using table statistics (row counts, value distributions).", "`EXPLAIN` shows the plan; `EXPLAIN ANALYZE` runs it and shows actual rows and times. Compare estimated vs actual rows to spot bad statistics.", "A sequential scan can beat an index when the query returns a large fraction of the table, because random index lookups cost more than reading pages in order.", "Common fixes: add the right index, run ANALYZE, rewrite the query, avoid functions on indexed columns."]
question: "When does the planner choose a sequential scan even though an index exists?"
answer: "When the planner estimates that a sequential scan is cheaper. That happens when the condition matches a large fraction of the table (roughly more than 5 to 20% of rows), because fetching each row through the index is a random page read priced at about 4 times a sequential one, so one sequential pass beats thousands of random reads. It also happens on tiny tables (a few pages), when the index cannot serve the condition as written (filtering only on a non-leading column of a composite index, a function on the column, a leading wildcard, a type mismatch), and when stale or misleading statistics make it overestimate how many rows will match."
sources: [{"title": "PostgreSQL docs: Using EXPLAIN", "url": "https://www.postgresql.org/docs/current/using-explain.html", "note": "Reading plans, costs, rows."}, {"title": "Use The Index, Luke: Execution plans", "url": "https://use-the-index-luke.com/sql/explain-plan", "note": "Plans across databases."}]
tools: ["PostgreSQL"]
---

## The planner decides how

SQL is **declarative**: it says what result you want, not how to get it. `SELECT * FROM orders WHERE customer_id = 7` does not say "use the index" or "read the whole table". A component called the **query planner** (or optimiser) makes that choice. It enumerates possible plans (which index to use, which order to join tables in, which join algorithm), estimates the cost of each one using statistics about the data, and runs the cheapest.

This matters because the difference between a good and a bad plan for the same query can be a factor of 10,000: 2 ms against 20 seconds. When a query is slow, the **execution plan** is the first thing to look at, before adding hardware, caches or indexes.

~~~dg-flow The path from SQL text to rows. Statistics feed the cost estimates that decide which plan wins.
[Planning] SQL text -(parse)-> Query tree -(rewrite views)-> Candidate plans -(cost with stats)-> *Cheapest plan
[Execution] *Cheapest plan -(run)-> Rows to client
~~~

## How the planner estimates cost

The planner needs two kinds of knowledge.

**Statistics about the data.** `ANALYZE` (run automatically by Postgres's autovacuum, or by hand) samples each table and records, per column: the number of distinct values, the fraction of nulls, a list of the **most common values** with their frequencies, and a **histogram** that splits the remaining values into buckets of equal population. From these the planner estimates **selectivity**: the fraction of rows a condition will match. If `status = 'shipped'` appears in the most-common list with frequency 0.85, the planner expects 85% of rows to match.

**A cost model.** Postgres prices everything relative to reading one page **sequentially**, which costs 1.0. Its default parameters are:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `seq_page_cost` | 1.0 | Read one page as part of a sequential scan |
| `random_page_cost` | 4.0 | Read one page at a random location |
| `cpu_tuple_cost` | 0.01 | Process one row |
| `cpu_index_tuple_cost` | 0.005 | Process one index entry |
| `cpu_operator_cost` | 0.0025 | Evaluate one operator, such as a comparison |

Random reads are priced at 4 times sequential reads because on spinning disks they really were far more expensive. On SSDs the real gap is much smaller, and many teams lower `random_page_cost` to about 1.1, which makes the planner more willing to use indexes.

### A worked cost

Take a table `tenk1` with 10,000 rows in 345 pages. A full sequential scan costs 345 pages × 1.0 plus 10,000 rows × 0.01 = **445**. Adding a filter `WHERE unique1 < 7000` adds 10,000 × 0.0025 for the comparison, giving about 470.

## Reading EXPLAIN

`EXPLAIN` prints the chosen plan as a tree of nodes. Each node takes rows from its children, does one job (scan, filter, join, sort, aggregate) and passes rows up. Read it from the most indented line outwards. Each line shows:

- `cost=X..Y`: the estimated **start-up cost** (before the first row is produced) and the **total cost**, in the units above.
- `rows`: the estimated number of rows this node outputs.
- `width`: the estimated average row size in bytes.

```
Hash Join  (cost=8.6..412.0 rows=100 width=44)
  Hash Cond: (o.customer_id = c.id)
  ->  Seq Scan on orders o  (cost=0..360.0 rows=10000 width=36)
  ->  Hash  (cost=8.5..8.5 rows=10 width=12)
        ->  Index Scan using customers_city on customers c  (rows=10)
              Index Cond: (city = 'Pune')
```

~~~dg-tree The plan above as a tree: the hash join consumes a hash table built from Pune customers and probes it with every order.
Hash Join | orders.customer_id = customers.id, 100 rows
  Seq Scan on orders | 10,000 rows, probe side
  *Hash | build side, 10 rows
    Index Scan on customers_city | city = 'Pune'
~~~

## The same table, different plans

Here is how the plan for a single-table query on `tenk1` changes as the condition matches more rows:

| Query on tenk1 | Rows matched | Plan | Estimated cost |
| --- | --- | --- | --- |
| `unique1 = 42` | 1 | Index Scan | 8.30 |
| `unique1 < 100` | 100 (1%) | Bitmap Index Scan + Bitmap Heap Scan | 225 |
| `unique1 < 7000` | 7,000 (70%) | Seq Scan with filter | 470 |

The scan types, from fewest rows to most:

- **Index scan**: descend the index, then fetch each matching row in index order. Each fetch may be a random page read. Best for a handful of rows.
- **Index-only scan**: answer entirely from the index when it contains every column the query needs. In Postgres it also needs the table's visibility map to be up to date (lesson 03.07).
- **Bitmap scan**: first collect the locations of every matching row from the index into a bitmap, sort them by physical page, then read each needed page once, in order. This turns random reads into nearly sequential ones and reads each page only once even if it holds many matches. Good for hundreds to tens of thousands of rows. Two bitmaps from two indexes can be combined with `BitmapAnd` or `BitmapOr`.
- **Sequential scan**: read the whole table in physical order and filter. Best when a large fraction of rows match, because sequential reads are cheap and there are no random jumps.

~~~dg-bars Estimated cost for each plan type on tenk1 as the query matches more rows. The index plan's cost climbs with every matching row; the seq scan's is flat.
Index scan, 1 row | 8.3 | 8.3
Bitmap scan, 100 rows | 225 | 225
*Seq scan, 7,000 rows | 470 | 470
Index scan forced, 7,000 rows | 7000 | about 7,000+
~~~

## When the planner ignores your index

Usually the planner is **right** to ignore it:

- **The query matches a large share of the table.** As a rough guide, once a condition matches more than about 5 to 20% of rows (it depends on row width and caching), thousands of random page reads cost more than one sequential pass.
- **The table is tiny.** A table of 3 pages is cheaper to scan than to probe through an index. This is why you should never judge plans on a small development database: the plan at 1,000 rows tells you nothing about the plan at 100 million.
- **The index cannot serve the condition.** The query filters only on the second column of a composite index (lesson 03.03), wraps the column in a function (`lower(email) = ?` with an index on `email`), uses a leading wildcard (`LIKE '%son'`), or compares mismatched types (a text column against a number).
- **`ORDER BY ... LIMIT` changes the trade-off.** With `LIMIT 10` the planner may walk an index in sort order and stop early, even if that index does not match the filter well. If matching rows are rare, this can backfire badly.

Sometimes the planner is **wrong**, almost always because its row estimates are wrong:

- **Stale statistics** after a bulk load or delete. Running `ANALYZE` fixes it.
- **Correlated columns.** The planner assumes conditions are independent. For `city = 'Paris' AND country = 'France'` it multiplies the two selectivities and badly underestimates the rows. Postgres's `CREATE STATISTICS` can record such dependencies.
- **Skewed data and parameters.** A prepared statement planned for a typical customer may be reused for a customer with 1,000 times more rows.

## EXPLAIN ANALYZE

`EXPLAIN ANALYZE` actually **runs** the query and shows, next to each estimate, the actual time, actual rows and `loops` (how many times the node ran; times and rows are per loop, so multiply). The single most useful check is whether estimated rows are close to actual rows at each node. When they differ by 100 times or more, the planner chose based on a false picture, and a bad plan is likely. Adding `BUFFERS` shows how many pages came from the cache (`shared hit`) and how many from disk (`read`), which explains why the same query is fast the second time.

Careful: because it really executes, `EXPLAIN ANALYZE` on an `UPDATE` or `DELETE` really changes data. Wrap it in `BEGIN; ... ROLLBACK;`.

## Joins in plans

For each join, the planner also picks an algorithm:

- **Nested loop**: for each row of the outer input, look up matching rows in the inner input, ideally through an index. Cost is roughly outer rows × lookup cost. Best when the outer side is small.
- **Hash join**: build an in-memory hash table on the smaller input, then stream the larger input and probe it. Cost is roughly linear in both inputs. Good for large, unsorted inputs, as long as the hash table fits in `work_mem`; otherwise it spills to disk in batches.
- **Merge join**: walk two inputs that are already sorted on the join key, like zipping two sorted lists. Good when both sides come out of indexes in order, or when sorting is cheap.

~~~dg-matrix Join algorithms and the situations each one suits.
Join | Small outer side | Two large inputs | Needs index on inner | Needs sorted input | Memory use
Nested loop | yes | no | yes, ideally | no | low
Hash join | partial | yes | no | no | high, build side
Merge join | partial | yes | no | yes | low
~~~

The join **order** matters as much as the algorithm. For three tables there are 12 possible orders; for 10 tables, millions. Postgres searches all orders exhaustively up to about 12 tables, then switches to a genetic search. A bad row estimate early in the plan multiplies through every join above it, which is why big queries are the ones most often planned badly.

<div class="ex">
<div class="lab">Worked example</div>

**why the index was skipped**

`SELECT * FROM orders WHERE status = 'shipped'` on 10 million rows in 400,000 pages, where 85% are shipped. The planner estimates 8.5 million rows.

Seq scan: 400,000 × 1.0 plus 10 million × 0.0125 = about 525,000.

Index scan: 8.5 million entries, and since shipped rows are on almost every page, nearly every one of the 400,000 pages is fetched, many of them more than once at random cost 4.0. The estimate is well over 2 million. The seq scan wins by a factor of 4.

For `status = 'refunded'` (0.2%, 20,000 rows) the index or a bitmap scan wins easily. A partial index `ON orders (created_at) WHERE status = 'refunded'` would be tiny and even faster.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**a 100× misestimate and how to spot it**

A report joins `events` to `users` and takes 40 seconds. `EXPLAIN ANALYZE` shows a nested loop whose outer node estimated `rows=12` but actually produced `rows=180000`. The planner believed it would do 12 index lookups; it did 180,000.

The cause: 50 million events were loaded an hour earlier and autovacuum had not yet analysed the table, so the statistics described yesterday's data.

Running `ANALYZE events` corrects the estimate to about 175,000 rows. The planner switches to a hash join and the query takes 1.8 seconds. No index was added; the plan just had to see reality.

</div>

## Common mistakes

- **Testing plans on tiny tables.** The planner rightly scans them; production will behave differently.
- **Forcing plans.** Disabling seq scans globally to "make it use the index" fixes one query and breaks others.
- **Ignoring estimate versus actual.** That gap is the single best clue to a bad plan.
- **Reading cost as milliseconds.** Cost units are relative; compare them only within one plan.
- **Running EXPLAIN ANALYZE on a write without a rollback.**

## In an interview

If asked why a query ignores an index, list the reasons in order: the condition matches too many rows; the table is small; the index cannot serve the condition as written (column order, function, type); or statistics are stale. Then say how you would check: `EXPLAIN ANALYZE`, compare estimated and actual rows, look at buffers.

<div class="take">
<div class="lab">Key takeaway</div>

the planner picks the plan with the lowest estimated cost, using statistics about the data and a model where random page reads cost more than sequential ones. Sequential scans win when many rows match. When a plan is bad, compare estimated and actual rows in EXPLAIN ANALYZE.

</div>
