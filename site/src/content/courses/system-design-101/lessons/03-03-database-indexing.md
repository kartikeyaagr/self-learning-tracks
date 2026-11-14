---
id: "03.03"
course: "system-design-101"
module: "03"
slug: "database-indexing"
title: "Database indexing"
keyIdeas: ["An index is a separate sorted structure (usually a B-tree) that maps column values to row locations, so lookups avoid scanning the whole table.", "Indexes speed up reads (WHERE, JOIN, ORDER BY) but slow down writes and use space, because every insert/update must update each index.", "Composite index on (a, b) is sorted by a, then b. It helps `WHERE a = ?` and `WHERE a = ? AND b = ?`, but not `WHERE b = ?` alone (leftmost prefix rule).", "Covering index: includes all columns a query needs, so the database never touches the table.", "Low-selectivity columns (boolean flags) rarely benefit from a plain index. Partial indexes can help."]
question: "Why does an index on (a, b) help WHERE a = ? but not WHERE b = ?"
answer: "An index on (a, b) is sorted by a first and by b only within each value of a, like a phone book sorted by surname then first name. For WHERE a = ? the database descends the tree to the one contiguous block of entries with that a and reads only those. Rows with a given b are scattered: there is one small run of them inside every distinct value of a, so there is no single place in the sort order to seek to. The database would have to read the whole index (or the whole table), so the planner usually does a full scan instead. Some engines can skip-scan across values of a, but that only helps when a has very few distinct values."
sources: [{"title": "Use The Index, Luke", "url": "https://use-the-index-luke.com/", "note": "Free book on SQL indexing. Read chapter 1 and the WHERE clause chapter."}, {"title": "PostgreSQL docs: Indexes", "url": "https://www.postgresql.org/docs/current/indexes.html", "note": "Index types and multicolumn indexes."}, {"title": "Hello Interview: Database Indexing", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/db-indexing", "note": "Interview view."}]
tools: ["PostgreSQL"]
---

## Without an index, every query reads everything

A table's rows are stored in fixed-size **pages** on disk (8 KB in Postgres). In Postgres the table itself is a **heap**: rows sit wherever there was free space when they were inserted, in no useful order. To find `WHERE email = 'a@b.com'` without an index, the database must read every page of the table and test every row. This is a **sequential scan** (or full table scan). It is like looking for one book in a library by pulling every book off every shelf.

Put numbers on it. A users table with 10 million rows of about 200 bytes each occupies roughly 2 GB, or about 250,000 pages. Reading 2 GB from a fast SSD at 2 GB/s takes about a second, and that is if nothing else is running. If the query runs 100 times a second, the database is doing nothing else.

An **index** is a separate data structure, kept sorted by the indexed column or columns, where each entry points at a row. Because it is sorted, the database can jump to the right place, the way you find a name in a phone book without reading every page. With a B-tree index (lesson 03.04 explains its structure) the same email lookup reads 3 or 4 index pages and 1 table page.

~~~dg-compare The same lookup with and without an index on a 10-million-row table.
[Sequential scan]
WHERE email = x -(read every page)-> 250,000 pages -> 1 match
- About 1 s of I/O per query
- Cost grows with table size
[*B-tree index]
WHERE email = x -(descend tree)-> 4 index pages -> 1 table page
+ Under 1 ms when cached
+ Cost barely grows with size
- Every write updates the index
~~~

## What the pointer points at

The index entry stores the indexed value plus a way to find the row. How that works differs between engines, and it matters for performance.

- **Postgres**: the entry holds a **TID** (tuple ID), the physical address of the row as (page number, slot). Finding the row is one jump to that page.
- **InnoDB (MySQL)**: the table itself is stored as a B-tree sorted by primary key, called a **clustered index**. A secondary index entry holds the **primary key value**, not a physical address. Finding the row means a second B-tree descent through the primary key. That is why InnoDB rewards short primary keys and why rows with nearby primary keys sit physically together.

## What indexes cost

Indexes are not free, and every one you add is paid for on every write.

- **Write cost**: every insert, and every update or delete that touches indexed columns, must also update every index on the table. A table with five indexes does up to six page modifications per insert, often on six different pages that may need reading from disk first.
- **Space**: an index on a wide column can be nearly as large as the table. On a 2 GB table, five indexes can add 2 to 4 GB.
- **Memory**: indexes compete with table data for the buffer pool (the in-memory page cache). An index that is never used still gets updated and still pushes useful pages out of memory.
- **Not always used**: for small tables, or queries that match a large fraction of rows, scanning is cheaper and the planner will ignore the index (lesson 03.05).

So index the queries you actually run, not every column. Write-heavy tables like event logs should have few indexes. In Postgres, `pg_stat_user_indexes` shows how often each index is used, and an `idx_scan` of zero after weeks of traffic is a strong hint to drop it.

## Index types

Postgres offers several index types. The B-tree is the default and covers most needs.

| Type | Good for | Example query |
| --- | --- | --- |
| B-tree | Equality, ranges, sorting, prefix `LIKE 'abc%'` | `WHERE created_at > now() - interval '1 day'` |
| Hash | Equality only | `WHERE session_id = ?` |
| GIN | Values containing many keys: arrays, JSONB, full-text (an inverted index) | `WHERE tags @> '{sql}'` |
| GiST / SP-GiST | Geometry, ranges, nearest neighbour | `ORDER BY location <-> point(...) LIMIT 10` |
| BRIN | Huge tables whose values follow physical order, such as timestamps in an append-only log | `WHERE ts BETWEEN ...` on 1 TB of logs |

A **BRIN** (block range index) stores only the minimum and maximum value for each range of, say, 128 pages. It is tiny (megabytes for a terabyte table) but approximate: the database reads every block range whose min/max could contain the value. It only works when the physical order of rows follows the column, as with insert timestamps.

Other index families appear later: LSM trees, used by write-heavy stores such as Cassandra and RocksDB (Module 7), and geospatial indexes such as geohashes, quadtrees and R-trees (Module 12).

~~~dg-matrix Which Postgres index type can serve which kind of query.
Index | Equality | Range | Sorting | Contains / full-text | Nearest neighbour
B-tree | yes | yes | yes | no | no
Hash | yes | no | no | no | no
GIN | yes, inside values | no | no | yes | no
GiST | partial | yes | no | partial | yes
BRIN | partial, coarse | yes, coarse | no | no | no
~~~

## Composite indexes and column order

A **composite index** covers several columns. An index on `(a, b)` is sorted by `a`, and by `b` only **within** each value of `a`. It is exactly like a phone book sorted by surname, then first name. Finding every "Smith" is easy: all the Smiths are together. Finding every "John" means reading the whole book, because Johns are scattered across every surname.

So a composite index helps queries that use its **leftmost columns** (the "leftmost prefix" rule):

- `(a, b, c)` helps `WHERE a = ?`, `WHERE a = ? AND b = ?`, and `WHERE a = ? AND b = ? AND c = ?`.
- It helps `WHERE a = ? AND b > ?` (equality on `a`, range on `b`).
- It does **not** help `WHERE b = ?` or `WHERE c = ?` alone.
- For `WHERE a = ? AND c = ?` it can seek on `a` but must then check `c` on every entry for that `a`.

Some databases can do a **skip scan**, jumping from one value of `a` to the next and seeking `b` inside each. Oracle and MySQL 8 have it and Postgres 18 added it. It only pays off when `a` has very few distinct values, so do not design around it.

~~~dg-tree An index on (country, city) is sorted by country first. Cities only form a contiguous run inside one country, so WHERE city = 'Pune' alone has nowhere to seek.
Index (country, city)
  DE
    Berlin
    Munich
    Pune | a Pune in DE? sorted here if so
  IN
    Delhi
    Mumbai
    *Pune | WHERE country='IN' AND city='Pune' seeks here
  US
    Austin
    Pune | another possible Pune, far away
~~~

### A real case

After a company merger, an employees table's primary key became `(employee_id, subsidiary_id)` so that IDs from the two companies could not clash. Queries for "all employees of subsidiary 3" suddenly did a full table scan, because `subsidiary_id` was the second column. Reversing the key to `(subsidiary_id, employee_id)` kept uniqueness exactly the same and turned the subsidiary query into a range scan. One well-ordered composite index beats two single-column indexes: it uses less space and costs less on every write.

### How to choose the order

1. Put columns compared with **equality** first.
2. Put the column used for a **range** or for **sorting** last. Once the index hits a range condition, the columns after it can no longer be used to seek.
3. Among equality columns, lead with the one that **most queries** filter on, so one index serves many queries.

The often-repeated rule "put the most selective column first" is mostly a myth for B-trees. For a query with equality on both columns, the order barely matters to the lookup cost. What matters is which queries can use the leftmost prefix.

~~~dg-matrix Which queries an index on (a, b, c) can use to seek, and how far.
Query | Seek on a | Seek on b | Seek on c
WHERE a = 1 | yes | no | no
WHERE a = 1 AND b = 2 | yes | yes | no
WHERE a = 1 AND b = 2 AND c > 5 | yes | yes | yes, range
WHERE a = 1 AND b > 2 AND c = 5 | yes | yes, range | no, filter only
WHERE b = 2 | no | no | no
WHERE a = 1 AND c = 5 | yes | no | no, filter only
~~~

## More tools in the box

- **Unique indexes** enforce uniqueness as well as speeding lookups (lesson 03.01).
- **Expression indexes** index the result of a function: `CREATE INDEX ON users (lower(email))` serves `WHERE lower(email) = ?`. A plain index on `email` cannot, because the stored values are not lowercased.
- **Partial indexes** index only some rows: `CREATE INDEX ON orders (created_at) WHERE billed IS NOT TRUE` keeps an "unbilled orders" index tiny when 99.9% of orders are billed.
- **Covering indexes**: if the index holds every column a query needs, the database answers from the index alone (an **index-only scan**) and skips the table. Postgres lets you add non-key payload columns with `INCLUDE`, for example `CREATE INDEX ON posts (user_id, created_at) INCLUDE (title)`.
- **Sorted output**: a B-tree on `(user_id, created_at)` returns a user's posts already ordered by time, so `ORDER BY created_at DESC LIMIT 20` reads 20 entries and stops, with no sort step at all.

<div class="ex">
<div class="lab">Worked example</div>

**the feed query, two index orders**

Query: `SELECT id, body FROM posts WHERE user_id = 42 ORDER BY created_at DESC LIMIT 20`. The table has 500 million posts from 10 million users, so about 50 per user on average.

With an index on `(user_id, created_at)`: descend 4 levels to the end of user 42's block, read 20 entries backwards (all on one or two leaf pages), then fetch 20 rows. About 26 page reads, under 2 ms even with some disk reads.

With an index on `(created_at, user_id)`: the index is sorted by time, so user 42's posts are spread across the whole index. The database must walk backwards through every post from every user until it finds 20 by user 42. If user 42 last posted a month ago, that is most of a month's posts, millions of entries. The planner will probably choose a different plan entirely, and it will still be slow.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the cost of an extra index**

An orders table takes 2,000 inserts per second and has 4 indexes. Each insert touches the heap plus 4 index leaf pages: 5 page writes, 10,000 per second.

Someone adds 3 more "just in case" indexes on columns no query uses. Now each insert touches 8 pages: 16,000 page writes per second, 60% more write I/O and roughly 60% more write-ahead log. Buffer pool hit rates fall because the new indexes take memory. Dropping them is often the cheapest performance win available.

</div>

## Common mistakes

- **Indexing every column separately.** Single-column indexes on `a` and `b` are rarely as good as one composite index on `(a, b)`.
- **Wrong column order.** A range column first stops the rest of the index being used for seeks.
- **Wrapping the column in a function.** `WHERE date(created_at) = '2026-01-01'` cannot use an index on `created_at`. Rewrite as a range: `created_at >= '2026-01-01' AND created_at < '2026-01-02'`.
- **Type mismatches.** Comparing a text column to a number can force a conversion on every row and disable the index.
- **Leading wildcards.** `LIKE '%son'` cannot seek in a B-tree. Use a trigram (GIN) index or full-text search.
- **Never checking usage.** Unused indexes cost writes and memory forever.

## In an interview

When you name a table, name its indexes in the same breath and tie each one to a query: "posts gets an index on `(user_id, created_at)` for the profile page." If the workload is write-heavy, say that you are keeping indexes to a minimum and why. If asked why a query is slow, think aloud: which index, in which column order, and does the query's `WHERE` clause match its leftmost prefix?

<div class="take">
<div class="lab">Key takeaway</div>

an index is a sorted copy of some columns that lets the database seek instead of scan, paid for on every write and in memory. A composite index works left to right, so order its columns as equality first, range or sort last, and match them to the queries you actually run.

</div>
