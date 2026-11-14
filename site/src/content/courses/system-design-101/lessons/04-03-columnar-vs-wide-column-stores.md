---
id: "04.03"
course: "system-design-101"
module: "04"
slug: "columnar-vs-wide-column-stores"
title: "Columnar vs wide-column stores"
keyIdeas: ["**Columnar databases** (ClickHouse, BigQuery, Redshift) store each column separately. Analytics queries that read a few columns across billions of rows are fast, and columns compress well. Single-row writes and updates are slow.", "**Wide-column stores** (Cassandra, HBase, Bigtable) store rows grouped by partition key, sorted by clustering key within a partition, with flexible columns per row. Built for high write throughput and key-range reads.", "Despite the similar names: columnar = analytics (OLAP); wide-column = operational scale (OLTP-ish, key-based).", "Cassandra data modeling is query-first: one table per query pattern, duplicate data freely."]
question: "Cassandra and ClickHouse are both \"column\" databases. How are they different?"
answer: "Cassandra is a wide-column store: rows are grouped into partitions by a hashed partition key spread across nodes and sorted inside each partition by clustering columns, and writes go through an LSM tree (commit log plus memtable), so it is built for huge write volumes and reading a known partition, an OLTP-style workload serving the application. ClickHouse is a columnar store: each column is stored separately and heavily compressed on disk, and queries are executed in vectorised batches, so analytical queries read only the few columns they need across billions of rows, an OLAP workload. Cassandra is bad at scans and aggregations; ClickHouse is bad at frequent single-row writes and updates. Cassandra serves the app; ClickHouse serves analytics."
sources: [{"title": "Cassandra docs: Data modeling", "url": "https://cassandra.apache.org/doc/latest/cassandra/developing/data-modeling/index.html", "note": "Query-first wide-column design."}, {"title": "ClickHouse: What is a columnar database?", "url": "https://clickhouse.com/docs/intro", "note": "Column storage for analytics."}]
---

## Two very different "column" databases

The word "column" appears in the names of two unrelated families of database, and mixing them up is a common interview mistake.

- **Wide-column stores** (Cassandra, ScyllaDB, HBase, Google Bigtable) are distributed, **row-oriented** key-value stores built for huge write volumes and fast lookups by key. "Wide" means a row can have many columns, and different rows can have different columns. They serve applications: **OLTP** (online transaction processing), meaning many small reads and writes.
- **Columnar stores** (ClickHouse, BigQuery, Redshift, Snowflake, DuckDB, and the Parquet file format) store **each column's values together on disk**. They are built for analytical queries that scan and aggregate billions of rows. They serve analysts and dashboards: **OLAP** (online analytical processing).

The names come from different ideas. In a wide-column store, "column" is about the **data model** (flexible columns inside a row). In a columnar store, it is about the **physical layout** on disk. Cassandra stores a row's columns together; ClickHouse stores a column's rows together.

## Wide-column stores: Cassandra

Cassandra was built at Facebook for inbox search and is now used for messaging, time-series and event data at companies such as Apple, Netflix and Discord (which later moved to the Cassandra-compatible ScyllaDB). Its design goals are: accept writes at very high rates, scale linearly by adding nodes, and stay available when nodes or whole data centres fail.

### Design from the queries

Cassandra's data modelling rule is to **design from the queries, not from the entities**. There are no joins and only limited secondary indexes, so everything a query needs must be in one table, readable with one key lookup. The usual pattern is **one table per query**. The same data is written to several tables, each organised for a different query. That duplication is cheap because Cassandra writes are very fast.

### The primary key has two parts

- The **partition key** decides which nodes hold the data. Cassandra hashes it onto a ring of nodes using consistent hashing (lesson 04.07), and each partition is stored on a few replica nodes (typically 3). All rows with the same partition key live together on the same nodes.
- The **clustering columns** sort the rows **inside** a partition on disk. Reading a range of neighbouring rows ("messages after 10:00") is therefore one sequential read.

For a chat app: `PRIMARY KEY ((conversation_id), message_ts)`. The double parentheses mark the partition key. All of a conversation's messages are stored together, sorted by time, so "latest 50 messages in this conversation" reads one partition, one contiguous slice. To list magazines by publisher, you would make `publisher` the partition key and `id` a clustering column.

~~~dg-tree A Cassandra messages table: the partition key picks the replica nodes, and clustering columns keep rows sorted within each partition.
messages table
  partition (conv 77, 2026-03) | stored on nodes 2, 5, 9
    msg_ts 10:00:01 | from Ana, hi
    msg_ts 10:00:09 | from Raj, hello
    *msg_ts 10:01:30 | read slice starts here
  partition (conv 77, 2026-04) | stored on nodes 4, 6, 1
    msg_ts 09:12:44
  partition (conv 81, 2026-03) | stored on nodes 7, 3, 8
~~~

### Two goals for the partition key

1. **Spread data evenly** across the cluster. A partition key with few values (country, status) piles data onto a few nodes.
2. **Read as few partitions as possible** per query, ideally exactly one.

The two goals pull against each other, and good modelling balances them. Keep partitions **bounded**: a common guideline is under about 100,000 rows and 100 MB per partition. A chat conversation that runs for years would grow without limit, so it is split into **buckets**, for example `PRIMARY KEY ((conversation_id, month), message_ts)`. Reading the latest messages then reads the current month's partition, and older history means stepping back one bucket at a time.

### Why writes are so fast

Cassandra uses a **log-structured merge tree** (LSM tree), not a B-tree. A write is appended to a **commit log** on disk (sequential, for durability) and inserted into an in-memory sorted table called a **memtable**. That is all; the client gets its acknowledgement. When the memtable fills, it is flushed to disk as an immutable sorted file called an **SSTable**. Background **compaction** merges SSTables to discard overwritten and deleted values. No write ever reads or rewrites a page in place, which is why a single node can absorb tens of thousands of writes per second. Reads pay instead: they may have to check the memtable and several SSTables, using Bloom filters (lesson 04.13) to skip files that cannot contain the key. Module 7 covers LSM trees in depth.

~~~dg-seq A Cassandra write on one replica: append to the commit log, insert into the memtable, acknowledge. Disk files are written later.
Client > Coordinator : INSERT message, consistency QUORUM
Coordinator > Replica : write
Replica > Replica : append to commit log
Replica > Replica : insert into memtable
Replica -> Coordinator : ack
Coordinator -> Client : success after 2 of 3 acks
Replica > Replica : later, flush memtable to SSTable
~~~

Wide-column stores suit massive write volumes where data is read back by a known key and often by time range: messaging, time-series, telemetry, activity feeds, event logs.

## Columnar stores: ClickHouse

ClickHouse is a column-oriented database for OLAP: queries that aggregate over very large datasets, often needing answers in under a second.

### Row layout versus column layout

- **Row storage** keeps each row's fields together on disk. That is ideal for "fetch order 123 with all its fields" or "update this one row": one page read or write.
- **Column storage** keeps all the values of one column together, in a separate file per column. A query that touches 3 of a table's 100 columns reads only those 3 columns from disk and ignores the other 97.

~~~dg-compare The same orders table on disk: a row store keeps each order together; a column store keeps each field together.
[Row store]
Page: order1 all fields -> order2 all fields -> order3 all fields
+ Fetch one whole order: one read
+ Cheap single-row update
- SUM(amount) reads every field
[*Column store]
File amount: 19.9, 5.0, 42.0 ... -> File country: IN, IN, DE ...
+ SUM(amount) reads one file
+ Similar values compress 5 to 10 times
- Fetching one whole row touches every file
- Updates are expensive
~~~

### Why columnar is so fast for analytics

Three effects multiply together.

1. **Read only the columns you need.** If the query uses 3 columns out of 100, you read about 3% of the data.
2. **Compression.** Values of one column are similar (country codes, timestamps that increase, prices in a narrow range), so they compress extremely well, often 5 to 10 times and sometimes far more. Run-length encoding turns "IN, IN, IN, IN" into "IN × 4"; delta encoding stores increasing timestamps as small differences. Less data on disk means less I/O.
3. **Vectorised execution.** The engine processes columns in batches of thousands of values with tight loops that suit modern CPUs (SIMD instructions), instead of interpreting one row at a time.

ClickHouse also keeps each table sorted by a chosen key and stores a **sparse index** (one entry per 8,192 rows) plus min/max values per block, so a filter on a date range skips most blocks without reading them.

For a feel of the speed: a query over 100 million rows of web analytics, filtering by region, date and phone model and grouping by model, ran in 92 ms, roughly a billion rows per second on one server.

~~~dg-bars Data read from disk for SUM(amount) WHERE country = 'IN' over 1 billion orders with 50 columns, about 200 bytes per row.
Row store, full scan | 200 | about 200 GB
Column store, 2 columns uncompressed | 12 | about 12 GB
*Column store, 2 columns compressed | 2 | about 2 GB
~~~

### What columnar stores are bad at

Columnar stores are poor at frequent single-row updates and deletes (a change touches many column files, and ClickHouse handles them as heavy background "mutations"), at point lookups of whole rows, and at many tiny inserts (they prefer batches of thousands of rows). That is why they sit **next to** an OLTP database rather than replacing it. Data usually flows from the application database or event stream into the columnar store in batches or through a pipeline (Module 8).

## Side by side

~~~dg-matrix Wide-column and columnar stores compared on the properties that decide which one to use.
Property | Wide-column, Cassandra | Columnar, ClickHouse
Workload | OLTP, many small reads and writes | OLAP, big scans and aggregations
Layout on disk | rows grouped by partition key | each column in its own file
Typical query | messages in conv X after time T | avg order value by country last quarter
High write rate of single rows | yes | partial, batch inserts
Point lookup by key | yes | partial
Scan billions of rows | no | yes
Ad hoc aggregations | no | yes
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one product, two stores**

A messaging app sends 50,000 messages per second. They are stored in Cassandra with `PRIMARY KEY ((conversation_id, bucket), message_ts)`. Opening a chat reads the newest 50 rows of one partition: one replica set, one sorted slice, about 5 ms.

Every message event is also streamed, in batches of 100,000 rows, into ClickHouse. Analysts run "messages per country per day for the last year": 1.6 trillion rows in total, but the query reads just the `country` and `date` columns, compressed to a few bytes per row, and the sort key lets it skip irrelevant blocks. It returns in seconds on a cluster.

Running the analytics query on Cassandra would mean reading every partition in the cluster. Running "open this chat" on ClickHouse would work, but each lookup would touch many column files and the tiny per-message inserts would overwhelm it. Different jobs, different stores.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing a Cassandra partition**

An IoT platform stores readings with `PRIMARY KEY ((sensor_id), ts)`. Each sensor reports every second: 86,400 rows a day, about 31 million a year. At 50 bytes per row, a year is 1.5 GB in one partition, far beyond the guideline of about 100 MB.

Bucketing by day, `PRIMARY KEY ((sensor_id, day), ts)`, gives 86,400 rows and about 4.3 MB per partition. "Last hour for sensor 7" reads one partition; "last week" reads 7 partitions, which is still cheap and can be done in parallel.

</div>

## Common mistakes

- **Saying "Cassandra is columnar".** Its data model has flexible columns, but it is row-oriented on disk and terrible at analytical scans.
- **Modelling Cassandra like SQL**, with normalised tables and an expectation of joins.
- **Unbounded partitions**, such as all messages of a conversation forever in one partition.
- **Low-cardinality partition keys** that concentrate data on a few nodes.
- **Using a columnar store as the application database** and doing many single-row updates.

## In an interview

When asked to store high-volume events, separate the two needs out loud: an OLTP store keyed by the access pattern (wide-column, or sharded SQL) for serving users, and a columnar store fed by a stream for analytics. Name the partition key and clustering order for the first, and the sort key for the second.

<div class="take">
<div class="lab">Key takeaway</div>

wide-column stores (Cassandra) are distributed, query-first, row-oriented stores for heavy writes and reads by partition key, with rows sorted inside each partition. Columnar stores (ClickHouse) store each column separately and compressed, so analytical scans read only the columns they need. Different jobs, often used side by side.

</div>
