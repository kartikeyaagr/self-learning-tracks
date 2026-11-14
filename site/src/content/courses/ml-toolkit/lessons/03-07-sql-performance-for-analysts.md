---
id: "03.07"
course: "ml-toolkit"
module: "03"
slug: "sql-performance-for-analysts"
title: "SQL performance for analysts"
keyIdeas: ["Query time is driven by the rows and pages the database must **read**, not the rows it returns. A full scan of a million rows costs the same whether 86 rows or none match; an index seek on the same table touched about 4 pages instead of 11,106.", "A B-tree index on `(a, b)` serves conditions on `a`, or on `a` and `b`, but not on `b` alone; wrapping the column in a function (`date(ts) = ...`) usually stops the index from being used, so rewrite it as a half-open range on the bare column.", "Indexes help only selective filters: when a condition matches more than a few per cent of rows, jumping between index and table costs more than a scan, and the planner rightly ignores the index.", "Read the plan before guessing: `EXPLAIN QUERY PLAN` in SQLite shows SCAN vs SEARCH, covering indexes and temp B-tree sorts; `EXPLAIN ANALYZE` in PostgreSQL shows estimated vs actual rows for each step.", "Most slow analyst queries come from a handful of patterns: non-sargable filters, join fan-out followed by DISTINCT, per-row correlated subqueries without an index, deep OFFSET pagination, unsupported sorts and needless wide scans.", "In columnar warehouses (BigQuery, Snowflake, DuckDB) the levers are different: select fewer columns and filter on partition or cluster keys, because cost follows bytes scanned."]
question: "A dashboard query, `SELECT * FROM predictions WHERE model_id = 'm7' AND CAST(created_at AS DATE) = '2025-05-01' ORDER BY created_at`, is slow despite an index on `(created_at, model_id)`. Explain why the index does not help and propose an index and a query rewrite that would."
answer: "Two problems. The filter casts `created_at`, so the condition is not a bare-column comparison (it is not sargable) and the planner cannot use the index's ordering on `created_at`; it falls back to scanning. Even without the cast, the index puts `created_at` first, so it would read that day's predictions for every model and discard the ones that are not `m7`: the leftmost-prefix rule says equality columns should come before the range column. Rewrite the filter as a half-open range, `created_at >= '2025-05-01' AND created_at < '2025-05-02'`, and create an index on `(model_id, created_at)`. The planner then seeks to model `m7`, reads one day's contiguous run of entries, and returns them already in `created_at` order, so no sort is needed (no `USE TEMP B-TREE FOR ORDER BY` in SQLite, no Sort node in PostgreSQL). Confirm with `EXPLAIN QUERY PLAN` or `EXPLAIN ANALYZE` that the plan shows an index search with both conditions. `SELECT *` still requires a table lookup per matching row; if the dashboard needs only a few columns, list them and append them to the index (or use PostgreSQL's `INCLUDE`) so it becomes covering and the table is never touched."
sources: [{"title": "Use The Index, Luke", "url": "https://use-the-index-luke.com/", "note": "A free book on SQL indexing for developers: leftmost prefixes, functions in WHERE, joins and sorting."}, {"title": "SQLite: The query planner", "url": "https://www.sqlite.org/queryplanner.html", "note": "How SQLite uses indexes, covering indexes and sorting, with diagrams."}, {"title": "SQLite: EXPLAIN QUERY PLAN", "url": "https://www.sqlite.org/eqp.html", "note": "How to read SCAN, SEARCH and temp B-tree lines."}, {"title": "PostgreSQL docs: Using EXPLAIN", "url": "https://www.postgresql.org/docs/current/using-explain.html", "note": "Reading plan nodes, costs, and estimated vs actual rows with EXPLAIN ANALYZE."}]
practice: ["limit-and-offset", "subqueries", "multiple-joins"]
tags: ["indexes", "query plans", "performance"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## What you will learn

- Where a query actually spends its time: pages read and rows examined, not rows returned.
- How a B-tree index turns a full scan into a short seek, and the four rules that decide whether it can be used (leftmost prefix, sargable filters, selectivity, covering).
- How to ask the database for its plan with `EXPLAIN QUERY PLAN` (SQLite) and `EXPLAIN ANALYZE` (PostgreSQL), and which words in the output matter.
- The six patterns behind most slow analyst queries, each with a measured before/after and a rewrite.
- Why columnar warehouses (BigQuery, Snowflake, DuckDB) need different habits: fewer columns, pruned partitions.

*Before you start:* you should be comfortable with filtering (03.01), joins (03.02), GROUP BY (03.03), window functions (03.04) and correlated subqueries (03.05). No knowledge of database internals is assumed.

In 03.06 you built training sets with SQL: a spine, features from event tables, labels from the future. Those queries hit your largest tables, again and again while you iterate. This lesson makes them fast without guessing.

## The idea in one picture

Take an `events` table with 1,000,000 rows: one row per user action, with a user id (10,000 users), a timestamp covering 2025, a kind and a value.

```sql
CREATE TABLE events(id INTEGER PRIMARY KEY, user_id INT, ts TEXT, kind TEXT, value REAL);
SELECT COUNT(*) FROM events WHERE user_id = 42 AND ts >= '2025-03-01';   -- 86 rows match
```

Without help, SQLite must look at all million rows to find the 86 that match: a **full scan**. After `CREATE INDEX idx_user_ts ON events(user_id, ts)` it jumps straight to user 42's entries: an **index seek**. The figure shows how the two behave as the table grows.

~~~dg-fig 03.07-scan-vs-seek.svg
Same query, same answer, two access paths. The scan's time grows in step with the table (both axes are logarithmic); the seek barely moves from 10 thousand to 1 million rows. Timings come from SQLite on this course's build machine and depend on hardware.
~~~

On a million rows the scan took about 71 ms and the seek about 0.006 ms. Wall-clock times are noisy, so here is a steadier measure: SQLite runs each query as a program of virtual-machine (VM) instructions, which Python's `set_progress_handler` can count. The scan executed 3,000,306; the seek 271. That is about 11,000 times less work for the same 86 rows.

The point of the whole lesson: **the cost of a query follows what the engine has to read, not what it returns.** A query that returns one row can be the slowest thing you run all day.

## Step 1: where the time goes

A row-store database (SQLite, PostgreSQL, MySQL) keeps each table as fixed-size **pages** (4 KB by default in SQLite, 8 KB in PostgreSQL), caching recent ones in memory. A query's cost is mostly:

- how many pages it has to touch (from disk or from the cache);
- how much work it does per row: comparing, hashing for joins and GROUP BY, sorting for ORDER BY and window functions.

Our million-row table fills 11,106 pages, about 90 rows per page. A full scan touches all of them. An **index** is a separate structure, a **B-tree**: a sorted copy of one or more columns, plus a pointer to each row (in SQLite the pointer is the rowid). Its pages form a shallow tree. The root page points to inner pages, the inner pages point to leaf pages, and the leaf pages hold the sorted entries.

~~~dg-layers What an index lookup touches for WHERE user_id = 42, versus a full scan.
Root page | one page, always cached
Inner pages | 1 to 2 more levels, mostly cached
*Leaf entries | (user_id, ts, row pointer) for every row with user_id 42, side by side
Table pages | only the pages holding those rows; a full scan reads every page
~~~

Each inner page holds over a hundred child pointers, so three levels cover a million rows and four cover a hundred million. A seek reads one page per level, then walks along the leaves: that is why its time hardly changed in the first figure.

<div class="ex">
<div class="lab">Worked example</div>

**count the pages by hand**

Table: 1,000,000 rows at about 90 rows per 4 KB page gives $1{,}000{,}000 / 90 \approx 11{,}100$ pages. SQLite's `dbstat` view reports 11,106 (11,077 leaves plus 29 inner pages). A full scan touches all of them.

Index on `(user_id, ts)`: each leaf holds about 132 entries on average, so $1{,}000{,}000 / 132 \approx 7{,}580$ leaves (measured: 7,540). Each inner page points to about 114 children, so $7{,}540 / 114 \approx 66$ inner pages, few enough for a single root above them. Depth: root, inner, leaf = 3 levels.

Lookup for user 42: 3 page reads reach the first matching leaf. User 42 has 104 entries, which fit in one leaf or spill into a second, so the covering `COUNT(*)` reads 3 to 4 pages. Compared with the scan: $11{,}106 / 4 \approx 2{,}800$ times fewer pages.

If the query also needs a column that is not in the index (say `value`), each of the 104 rows means one jump to a table page. Users are spread at random, so that is about 104 different pages: $4 + 104 = 108$ pages, still $11{,}106 / 108 \approx 103$ times fewer than a scan.

Break-even: if every matching row costs one random page, the index stops paying off once the number of matches approaches the 11,106 pages a scan would read. That is about $11{,}106 / 1{,}000{,}000 \approx 1.1\%$ of the table. Keep that number in mind for Step 3.

</div>

## Step 2: the leftmost-prefix rule

A composite index on `(user_id, ts)` is sorted by `user_id` first. Only among rows with the same `user_id` is it sorted by `ts`. Like a phone book sorted by surname then first name: every "Smith" is easy to find, every "Anna" is not.

~~~dg-fig 03.07-leftmost.svg
Toy index entries (u = user, m = month of ts) in index order. Top: an equality on user_id plus a range on ts is one contiguous run, found with one seek. Bottom: a filter on ts alone matches entries spread across every user, so the engine reads them all.
~~~

What this means for `(user_id, ts)`:

- `WHERE user_id = 42` can use it: one contiguous run.
- `WHERE user_id = 42 AND ts >= '2025-03-01'` can use it: seek to the start of the run and read to its end.
- `WHERE user_id = 42 ORDER BY ts` can use it, and needs no sort, because the run is already in `ts` order.
- `WHERE ts >= '2025-12-01'` cannot seek. SQLite's plan says `SCAN events USING COVERING INDEX idx_user_ts`: it reads the whole index, because the index is smaller than the table, but it still visits every entry.

The design rule: **equality columns first, then the one range or sort column.** Columns after a range can only be checked entry by entry, not used for seeking.

## Step 3: sargable filters and selectivity

**Sargable** (from "search argument able") means the condition compares the bare column with a value, so the engine can find the range in the sorted index. Wrap the column in a function and the sorted order is useless, because the index stores `ts`, not `date(ts)`.

~~~dg-fig 03.07-sargable.svg
Four ways to ask for the same day, with an index on ts present every time. The three filters that transform or pattern-match the column all scan; the half-open range seeks. Machine-dependent timings from SQLite.
~~~

All four return the same 2,743 rows. The three that transform the column ran about 4 million VM instructions each (74 to 126 ms); the range ran 16,472 (0.13 ms).

Rewrites you will use constantly:

- `date(ts) = '2025-06-01'` becomes `ts >= '2025-06-01' AND ts < '2025-06-02'` (a **half-open** range: start included, end excluded, so no midnight edge cases).
- `strftime('%Y', ts) = '2025'` becomes `ts >= '2025-01-01' AND ts < '2026-01-01'`.
- `lower(email) = ?` needs an **expression index**, `CREATE INDEX ON users(lower(email))`. SQLite and PostgreSQL both support these.

`LIKE 'abc%'` uses an index only under particular collation settings (not in SQLite's default setup: third bar); `LIKE '%abc'` never can.

**Selectivity** is the share of rows a condition keeps. An index pays off only when it is small: each match costs a jump to a random table page, while a scan reads pages in order.

~~~dg-fig 03.07-selectivity.svg
Forcing SQLite to use an index on user_id (INDEXED BY) or to scan (NOT INDEXED) for filters that match more and more of the table. The index wins easily for small matches, and loses from a few per cent of rows; at 100% it was about 20 times slower than the scan.
~~~

The measured crossover sits between 1% and 3%, close to the 1.1% from the hand count. So a planner with good statistics rightly ignores an index for `split = 'train'` when that matches 80% of rows. Low-cardinality columns (booleans, a few categories) are poor indexes on their own, though they can lead a composite index followed by something selective.

## Step 4: covering indexes and sorts

If the index holds every column the query reads, the table is never touched. That is a **covering index**, and SQLite says so in the plan (`USING COVERING INDEX`). `SELECT MAX(ts) FROM events WHERE user_id = 42` is covered by `(user_id, ts)`. `SELECT value ...` is not, and pays the extra table jump per row, as in the hand example. Adding a column to the end of an index to cover a hot query is a common, cheap win. PostgreSQL also has `INCLUDE (...)` for columns that should be stored in the index but not sorted on.

Sorts are the other big cost: `ORDER BY`, `GROUP BY`, `DISTINCT` and ordered windows all need rows in order. If an index supplies that order, the sort disappears; if not, SQLite reports `USE TEMP B-TREE FOR ORDER BY` (or `GROUP BY`, or `DISTINCT`), and PostgreSQL shows a `Sort` node. A window like `OVER (PARTITION BY user_id ORDER BY ts)` over the whole table is exactly the order of `(user_id, ts)`, so that index removes the sort.

Indexes are not free: every write must update every index, and each takes space (our two-column index is 7,606 pages, about 70% of the table). Build them for queries you run repeatedly.

## Reading a query plan in code

Every engine will tell you its plan before you run the query. Here it is from Python's built-in `sqlite3`, on a 500,000-row copy of the events table.

```python
import sqlite3, random, time, datetime

random.seed(0)
con = sqlite3.connect(":memory:")
con.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, user_id INT, ts TEXT, kind TEXT, value REAL)")
start = datetime.datetime(2025, 1, 1)
con.executemany("INSERT INTO events VALUES (?,?,?,?,?)",
    [(i + 1, random.randrange(10_000),
      (start + datetime.timedelta(seconds=63 * i)).strftime("%Y-%m-%d %H:%M:%S"),
      random.choice(["click", "view", "buy"]), random.random()) for i in range(500_000)])

def plan(sql):                       # the 4th column of EXPLAIN QUERY PLAN is the readable step
    return [row[3] for row in con.execute("EXPLAIN QUERY PLAN " + sql)]

def ms(sql, reps=5):                 # best of several runs, in milliseconds
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); con.execute(sql).fetchall()
        best = min(best, time.perf_counter() - t)
    return round(1000 * best, 3)

q = "SELECT COUNT(*) FROM events WHERE user_id = 42 AND ts >= '2025-03-01'"
print(plan(q), ms(q))   # ['SCAN events'] 23.052
con.execute("CREATE INDEX idx_user_ts ON events(user_id, ts)")
print(plan(q), ms(q))   # ['SEARCH events USING COVERING INDEX idx_user_ts (user_id=? AND ts>?)'] 0.002
print(plan("SELECT COUNT(*) FROM events WHERE ts >= '2025-12-01'"))
# ['SCAN events USING COVERING INDEX idx_user_ts']        <- no leading user_id: no seek
print(plan("SELECT * FROM events WHERE user_id = 42 ORDER BY ts"))
# ['SEARCH events USING INDEX idx_user_ts (user_id=?)']   <- index supplies the order
print(plan("SELECT * FROM events WHERE user_id = 42 ORDER BY value"))
# ['SEARCH events USING INDEX idx_user_ts (user_id=?)', 'USE TEMP B-TREE FOR ORDER BY']
```

The words to look for in SQLite:

| Plan text | Meaning | Usually fine?
| --- | --- | --- |
| `SCAN t` | reads every row of t | only for small tables or most-of-table queries
| `SEARCH t USING INDEX i (a=?)` | seeks in index i, then fetches rows | yes
| `... COVERING INDEX ...` | answered from the index alone | best case
| `USE TEMP B-TREE FOR ORDER BY / GROUP BY / DISTINCT` | an explicit sort | check whether an index could supply the order
| `CORRELATED SCALAR SUBQUERY` | inner query runs once per outer row | fine only if the inner query seeks

In PostgreSQL, `EXPLAIN` prints a tree of nodes (`Seq Scan`, `Index Scan`, `Index Only Scan`, `Hash Join`, `Nested Loop`, `Sort`). `EXPLAIN ANALYZE` runs the query and adds real row counts and times to each node. `EXPLAIN (ANALYZE, BUFFERS)` also shows how many pages came from cache versus disk. Because `ANALYZE` executes the statement, wrap data-changing statements in a transaction you roll back.

The most useful single check in PostgreSQL is **estimated versus actual rows** on each node. If the planner expected 10 rows and got 10 million, it may have chosen a nested loop that does 10 million index lookups. Stale statistics are the usual cause; running `ANALYZE` on the table refreshes them.

## The six slow patterns, measured

Correlated subqueries and join fan-out cost the most when the inner table has no usable index. Continuing the session above:

```python
con.execute("CREATE TABLE users(user_id INTEGER PRIMARY KEY, country TEXT)")
con.executemany("INSERT INTO users VALUES (?,?)",
                [(u, random.choice(["DE", "FR", "US"])) for u in range(10_000)])
con.execute("CREATE TABLE ev_raw AS SELECT * FROM events")   # same rows, no index

last_seen = """SELECT u.user_id,
  (SELECT MAX(e.ts) FROM {t} e WHERE e.user_id = u.user_id) AS last_seen
FROM users u WHERE u.user_id < 200"""
print(ms(last_seen.format(t="ev_raw"), reps=1))   # 4905.605  (200 users x full scan each)
print(ms(last_seen.format(t="events")))           # 0.087     (200 seeks into idx_user_ts)

con.execute("CREATE INDEX idx_raw_user_kind ON ev_raw(user_id, kind)")
fan = """SELECT DISTINCT u.user_id, u.country FROM users u
JOIN ev_raw e ON e.user_id = u.user_id WHERE e.kind = 'buy'"""
ex = """SELECT u.user_id, u.country FROM users u WHERE EXISTS
  (SELECT 1 FROM ev_raw e WHERE e.user_id = u.user_id AND e.kind = 'buy')"""
print(ms(fan), plan(fan)[-1])   # 42.232 USE TEMP B-TREE FOR DISTINCT
print(ms(ex), plan(ex)[-1])     # 6.977 SEARCH e EXISTS USING COVERING INDEX idx_raw_user_kind (user_id=? AND kind=?)
```

The join made 166,916 rows for `DISTINCT` to collapse back to 10,000; `EXISTS` stops at each user's first purchase. The full list:

| Pattern | Symptom in the plan | Fix
| --- | --- | --- |
| Non-sargable filter, e.g. `date(ts) = ...` | `SCAN` despite an index on the column | half-open range on the bare column, or an expression index
| Fan-out then `DISTINCT` | big join, then `TEMP B-TREE FOR DISTINCT` | `EXISTS`, or aggregate before joining
| Correlated subquery, no index | `CORRELATED SCALAR SUBQUERY` over a `SCAN` | index the correlation column, or rewrite as a join or window
| Deep `OFFSET` pagination | time grows with page number | keyset pagination (below)
| Window or ORDER BY on a big table | `TEMP B-TREE FOR ORDER BY` | index in `(partition cols, order col)` order
| `SELECT *` on wide tables | nothing visible in the plan; slow transfer | list only the columns you need

**Keyset pagination.** `LIMIT 50 OFFSET 900000` produces and discards 900,000 rows to return 50. Instead, remember the last key returned and seek past it: `WHERE id > :last_id ORDER BY id LIMIT 50`. For a non-unique sort column, compare a pair: `WHERE (ts, id) > (:last_ts, :last_id) ORDER BY ts, id`. Both SQLite and PostgreSQL accept that row-value syntax.

~~~dg-fig 03.07-pagination.svg
Time to fetch one 50-row page on the 1-million-row table. OFFSET grows linearly with how deep the page is (50 ms at 900,000 rows here); the keyset version seeks and stays flat. Machine-dependent timings.
~~~

## Columnar warehouses

Analytical engines such as BigQuery, Snowflake, Redshift and DuckDB store each column separately and compressed. There are usually no B-tree indexes to tune, and scanning a few columns of a billion rows is routine. What you pay for, in time and often in money (BigQuery's on-demand pricing bills bytes scanned), is **bytes read**. The levers change:

- **Select only the columns you need.** `SELECT *` on a 300-column event table reads all 300; three columns of similar width read about 1% as much.
- **Filter on the partition column** (usually a date), so whole partitions are skipped. This is called **partition pruning**.
- **Filter on clustering or sort keys**, so blocks whose stored min/max values cannot match are skipped.
- **Keep partition filters sargable.** Wrapping the partition column in a function can stop pruning, for the same reason as in Step 3.

~~~dg-fig 03.07-columnar.svg
A schematic, not a measurement: 12 columns by 12 monthly partitions. A query for 3 columns in 1 month reads all 144 blocks in an unpartitioned row layout, and 3 blocks in a partitioned columnar layout.
~~~

## Going deeper

**Join algorithms.** A **nested loop** runs the inner side once per outer row: excellent when the inner side is an index seek, terrible otherwise (the 4.9-second query above). A **hash join** hashes the smaller input and streams the larger past it: linear time, no index, but memory-hungry. A **merge join** walks two inputs sorted on the key. SQLite mostly uses nested loops, adding temporary automatic indexes and Bloom filters; PostgreSQL and the warehouses pick among all three using row estimates.

**Cost-based planning.** The planner estimates rows per step from statistics (distinct counts, histograms) and picks the cheapest plan it finds. Errors multiply through joins: 10x at the bottom can be 1,000x at the top. Correlated columns (city and postcode) are a classic source, because the planner assumes independence.

**Write cost.** Every extra index adds a B-tree insert to each row written. Indexes that no query uses are pure cost; PostgreSQL's `pg_stat_user_indexes` shows how often each one is scanned.

**Caching and measuring honestly.** Later runs of a query hit the cache, so time several runs and report the best or median. Our timings varied twofold between runs on a shared machine; instruction and page counts did not.

## Key terms

| Term | Meaning
| --- | --- |
| Full scan | Reading every row (or page) of a table to answer a query
| B-tree index | Sorted copy of chosen columns with row pointers, organised as a shallow tree for fast seeks
| Leftmost prefix | A composite index serves conditions on its first column(s) only, in order
| Sargable | A filter that compares the bare column with a value, so an index range can be used
| Selectivity | The share of rows a condition keeps; indexes pay off when it is small
| Covering index | An index that contains every column a query needs, so the table is never read
| Query plan | The engine's chosen steps for a query, shown by EXPLAIN QUERY PLAN or EXPLAIN ANALYZE
| Keyset pagination | Paging by remembering the last key and seeking past it, instead of OFFSET
| Partition pruning | Skipping whole partitions of a table because the filter rules them out

<div class="take">
<div class="lab">Key takeaway</div>

query cost follows what the engine reads, not what it returns. Ask for the plan before guessing, keep filters sargable, index for the access paths you actually use (equality columns first, range or sort column last, extra columns to cover), and in columnar warehouses read fewer columns and fewer partitions.

</div>
