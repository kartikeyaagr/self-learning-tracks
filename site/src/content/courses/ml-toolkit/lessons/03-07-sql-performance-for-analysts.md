---
id: "03.07"
course: "ml-toolkit"
module: "03"
slug: "sql-performance-for-analysts"
title: "SQL performance for analysts"
keyIdeas: ["Query time is driven by the rows and pages the database must **read**, not the rows it returns. A full scan of 100 million rows costs the same whether one row or a million match.", "A B-tree index on `(a, b)` serves conditions on `a`, or on `a` and `b`, but not on `b` alone; wrapping the column in a function (`date(ts) = ...`) usually stops the index from being used.", "Read the plan before guessing: `EXPLAIN QUERY PLAN` in SQLite shows SCAN vs SEARCH, and `EXPLAIN ANALYZE` in PostgreSQL shows estimated vs actual rows for each step.", "Most slow analyst queries come from a handful of patterns: non-sargable filters, join fan-out followed by DISTINCT, per-row correlated subqueries without an index, deep OFFSET pagination and needless wide scans.", "In columnar warehouses (BigQuery, Snowflake, DuckDB) the levers are different: select fewer columns and filter on partition or cluster keys, because cost follows bytes scanned."]
question: "A dashboard query, `SELECT * FROM predictions WHERE model_id = 'm7' AND CAST(created_at AS DATE) = '2025-05-01' ORDER BY created_at`, is slow despite an index on `(created_at, model_id)`. Explain why the index does not help and propose an index and a query rewrite that would."
answer: "Two problems. The filter casts `created_at`, so the condition is not a bare-column comparison and the planner cannot use the index's ordering on `created_at`. Even without the cast, the index puts `created_at` first, so it would still read that day's predictions for every model and discard the ones that are not `m7`. Rewrite the filter as a half-open range, `created_at >= '2025-05-01' AND created_at < '2025-05-02'`, and create an index on `(model_id, created_at)`: equality column first, range column second. The planner then seeks to model `m7`, reads one day's contiguous range, and returns rows already in `created_at` order, so no sort is needed. `SELECT *` still requires a table lookup per row; listing the needed columns and adding them to the index would make it covering."
sources: [{"title": "Use The Index, Luke", "url": "https://use-the-index-luke.com/", "note": "A free book on SQL indexing for developers: leftmost prefixes, functions in WHERE, joins and sorting."}, {"title": "SQLite: The query planner", "url": "https://www.sqlite.org/queryplanner.html", "note": "How SQLite uses indexes, covering indexes and sorting, with diagrams."}, {"title": "SQLite: EXPLAIN QUERY PLAN", "url": "https://www.sqlite.org/eqp.html", "note": "How to read SCAN, SEARCH and temp B-tree lines."}, {"title": "PostgreSQL docs: Using EXPLAIN", "url": "https://www.postgresql.org/docs/current/using-explain.html", "note": "Reading plan nodes, costs, and estimated vs actual rows with EXPLAIN ANALYZE."}]
practice: ["limit-and-offset", "subqueries", "multiple-joins"]
tags: ["indexes", "query plans", "performance"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## Where the time goes

A row-store database keeps each table as pages (8 KB in PostgreSQL, 4 KB by default in SQLite) on disk, with recently used pages cached in memory. A query's cost is dominated by how many pages it reads and how much work it does per row: comparisons, hashing for joins and GROUP BY, sorting for ORDER BY and window functions.

Without a usable index, finding the requests of one user in a 100-million-row table means reading all of it, a **full scan**. With an index on `user_id`, the database descends a B-tree (three or four levels even for a huge table) to the matching entries and reads only the pages that hold those rows. That is the difference between seconds and milliseconds. The System Design course's lessons on database indexing, B-trees and query planning go deeper into the data structures; this lesson is about using them from the query side.

~~~dg-layers What an index lookup touches for WHERE user_id = 42, versus a full scan.
Root page | one page, always cached
Inner pages | 1 to 2 more levels, mostly cached
*Leaf entries | (user_id, row pointer) for every row with user_id 42
Table pages | only the pages holding those rows; a full scan reads every page
~~~

## How indexes are used

A B-tree index is a sorted copy of one or more columns with pointers back to the rows. It helps with equality and range conditions, joins on the indexed column, and ORDER BY on it (the rows come out already sorted). The rules that decide whether it can be used:

- **Leftmost prefix.** An index on `(user_id, ts)` serves `user_id = 42`, and `user_id = 42 AND ts >= '2025-03-01'`, and `ORDER BY ts` within one user. It cannot serve `ts >= '2025-03-01'` alone, because entries are sorted by `user_id` first. Put equality columns first, then the range column.
- **Sargable predicates.** The condition must compare the bare column with a value. `WHERE date(ts) = '2025-03-01'` or `WHERE lower(email) = ...` hides the column inside a function; rewrite as a range, `ts >= '2025-03-01' AND ts < '2025-03-02'`, or create an expression index on `lower(email)`. Leading wildcards (`LIKE '%gpt'`) and implicit type casts on the column defeat indexes the same way.
- **Selectivity.** If a condition matches a large fraction of the table (`split = 'train'` for 80% of rows), jumping between index and table pages is slower than scanning, and the planner rightly ignores the index.
- **Covering indexes.** If the index contains every column the query needs, the table is never touched. `CREATE INDEX ON evals (model_id, score)` lets `SELECT MAX(score) ... WHERE model_id = ?` run from the index alone.

Indexes are not free: each one slows inserts and updates and takes disk space. Build them for the queries you actually run repeatedly.

## Reading a query plan

Every engine will tell you what it intends to do. In SQLite:

```sql
EXPLAIN QUERY PLAN
SELECT COUNT(*) FROM requests WHERE user_id = 42 AND ts >= '2025-03-01';
-- before:  SCAN requests
-- after CREATE INDEX idx_req_user_ts ON requests(user_id, ts):
--          SEARCH requests USING COVERING INDEX idx_req_user_ts (user_id=? AND ts>?)
```

Words to look for: `SCAN` (full pass over a table), `SEARCH ... USING INDEX` (index lookup), `USE TEMP B-TREE FOR ORDER BY` or `GROUP BY` (an explicit sort, often avoidable with an index in the right order), and `CORRELATED SCALAR SUBQUERY` (something run per outer row).

In PostgreSQL, `EXPLAIN ANALYZE` actually runs the query and prints each node (Seq Scan, Index Scan, Hash Join, Nested Loop, Sort) with its estimated and actual row counts and time. The most useful single check is whether estimated and actual rows differ by orders of magnitude. A planner that expects 10 rows and gets 10 million will pick a nested loop that runs forever. Out-of-date statistics are a common cause; run `ANALYZE`.

## Why some queries are slow

1. **Non-sargable filters**: functions or casts on an indexed column. Rewrite as ranges on the bare column.
2. **Fan-out then DISTINCT**: a one-to-many join multiplies rows, and `SELECT DISTINCT` hides it at the cost of a huge sort. Use `EXISTS`, or aggregate before joining.
3. **Correlated subqueries without support**: per-row execution over an unindexed inner table is outer times inner work. Index the correlation column or rewrite as a join or window function.
4. **Deep OFFSET pagination**: `LIMIT 50 OFFSET 1000000` reads and discards a million rows. Keyset pagination remembers the last key, `WHERE id > :last_id ORDER BY id LIMIT 50`, and seeks straight there.
5. **Sorting for windows**: `OVER (PARTITION BY user_id ORDER BY ts)` sorts the whole input unless an index on `(user_id, ts)` supplies the order.
6. **Wide reads**: `SELECT *` on wide tables drags every column through memory and the network.

## Columnar warehouses

Analytical engines store each column separately and compress it. There are usually no B-tree indexes to tune, and a full scan of a few columns is fast. What you pay for, in time and often in money, is bytes read. The levers become:

- select only the columns you need (`SELECT *` on a 300-column event table reads all 300);
- filter on the **partition** column (usually a date) so whole partitions are skipped, and on clustering keys so blocks can be pruned by their min/max statistics;
- avoid wrapping the partition column in a function in the filter, for the same sargability reason as above.

<div class="ex">
<div class="lab">Worked example</div>

**a slow feature query, diagnosed**

`SELECT user_id, COUNT(*) FROM events WHERE date(ts) >= '2025-01-01' GROUP BY user_id` takes 40 s on 200 million rows. The plan shows `SCAN events` and `USE TEMP B-TREE FOR GROUP BY`.

Rewriting the filter as `ts >= '2025-01-01'` makes it sargable, and an index on `(ts, user_id)` lets the range be read from the index alone. The GROUP BY still sorts, because the index is ordered by `ts`, not `user_id`.

If the query mostly runs for one user at a time, an index on `(user_id, ts)` is better: each lookup becomes a seek plus a short range read.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

ask the planner what it will do, make filters sargable, index for the access paths you actually use (equality columns first, range last), and in columnar warehouses read fewer columns and fewer partitions.

</div>
