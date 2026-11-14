---
id: "04.01"
course: "system-design-101"
module: "04"
slug: "nosql-decision-boundaries"
title: "NoSQL decision boundaries"
keyIdeas: ["\"NoSQL\" covers key-value, document, wide-column and graph databases. They trade some relational features (joins, multi-row transactions, flexible queries) for scale, availability or a better data shape.", "NoSQL wins for: huge write throughput with simple access patterns (time series, logs, events), key lookups at massive scale, flexible schemas, and graph traversals.", "Relational wins for: complex queries and reporting, strong consistency across rows, many relationships, and when the access pattern isn't known yet.", "Many NoSQL systems require designing tables around queries up front. If queries change, you may need to rewrite data.", "Default to a relational database unless a specific requirement points elsewhere."]
question: "Name 2 workloads where NoSQL is a clear win and 2 where it isn't."
answer: "NoSQL is a clear win for (1) very high-volume, append-heavy writes read back by key and time range, such as event logs, metrics or chat history in a wide-column store like Cassandra, where write rates exceed one machine and the database partitions for you; and (2) simple, latency-critical key lookups on ephemeral data, such as sessions, carts, rate-limit counters or live locations in Redis or DynamoDB. It is a poor fit for (1) payments, ledgers, inventory or bookings, which need multi-row ACID transactions and constraints, and (2) reporting or admin tools with ad hoc queries and joins across many entities, because NoSQL tables are designed around fixed queries."
sources: [{"title": "System Design Primer: SQL or NoSQL", "url": "https://github.com/donnemartin/system-design-primer#sql-or-nosql", "note": "Reasons for each."}, {"title": "Hello Interview: Data Modeling", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/data-modeling", "note": "Choosing a database in interviews."}]
---

## Start from SQL, move only for a reason

The default choice for a new system's main data store should be a relational database, usually Postgres or MySQL. Relational databases give you joins, constraints, ACID transactions (lesson 03.01), a mature query planner, and decades of tooling for backups, monitoring and migrations. The claim that they "do not scale" is overstated. Read replicas, caching, connection pooling, partitioning and finally sharding (lesson 03.10) take them a very long way: Shopify, GitHub, Stack Overflow and Notion run enormous businesses on MySQL or Postgres.

Picking an unusual database in an interview, or at work, without a concrete reason tends to hurt rather than help. The interviewer will ask "why?", and "it scales" is not an answer. A good answer names a specific access pattern or requirement that the relational database handles badly, and a specific store that handles it well.

## What "NoSQL" actually means

**NoSQL** is not one thing. It is a loose family of databases that each give up some relational features to gain something specific. The main members:

- **Key-value stores** (Redis, DynamoDB, Memcached): look up an opaque value by a key. Extremely fast and simple (lesson 04.02).
- **Document stores** (MongoDB, Couchbase, Firestore): store JSON-like documents and can index fields inside them (lesson 04.02).
- **Wide-column stores** (Cassandra, ScyllaDB, HBase, Bigtable): rows grouped into partitions, sorted inside each partition, built for huge write rates (lesson 04.03).
- **Graph databases** (Neo4j, Neptune): nodes and edges, built for multi-hop traversal (lesson 04.04).
- **Search engines** (Elasticsearch, OpenSearch): inverted indexes for full-text search and filtering (Module 9).
- **Time-series and columnar analytics stores** (InfluxDB, ClickHouse) are often discussed alongside them, though ClickHouse speaks SQL.

The common trade most of them make: they **partition data automatically** across many machines, they encourage **denormalising** data (storing copies to avoid joins), they do joins in application code if at all, they offer **limited transactions** (often single-key or single-partition only), and many default to **eventual consistency**, where replicas may briefly disagree. That style is sometimes summarised as **BASE**: basically available, soft state, eventually consistent, in contrast to ACID.

~~~dg-compare What you typically gain and lose moving from a relational database to a partitioned NoSQL store.
[*Relational, e.g. Postgres]
Normalised tables -(joins at read time)-> Any query
+ Ad hoc queries and joins
+ Multi-row ACID transactions
+ Constraints enforce rules
- Scaling writes past one node is your job
[Partitioned NoSQL, e.g. Cassandra]
One table per query -(lookup by partition key)-> Precomputed answer
+ Scales writes by adding nodes
+ Predictable latency at huge size
- Queries must be planned up front
- Joins and constraints move to your code
~~~

## When SQL is the better fit

- **Structured data with a clear schema** that changes occasionally.
- **Relational data that needs joins**: orders with customers with addresses with products.
- **Transactions across several rows or tables**: payments, inventory, bookings, anything where "half done" is a bug.
- **Many different query patterns**, including ones nobody has thought of yet (reporting, support tools, analytics), served by indexes on many columns.
- **Moderate scale**: up to tens of thousands of writes per second and a few terabytes on one primary, which covers most products.

## When NoSQL is the better fit

- **Data accessed by a known key**, without joins, at very high rates: sessions, carts, feature flags, rate-limit counters.
- **Write rates or data sizes beyond one machine** where you want the database to partition and rebalance for you: terabytes to petabytes, hundreds of thousands of writes per second.
- **Semi-structured data** whose shape varies a lot between records or changes often.
- **Very low, predictable latency** for simple operations, such as an in-memory store answering in under a millisecond.
- **Append-heavy, time-ordered data** read back by entity and time range: event logs, metrics, chat history, IoT readings.

## Choosing by access pattern

The most important input is the **access pattern**: for each API endpoint, what does it read and write, by which key, and how often? Then data volume, growth rate, and consistency needs. A useful exercise is to write the list of queries before naming any database.

~~~dg-matrix Workloads and the store family that usually fits them best, with the reason.
Workload | Relational | Key-value | Document | Wide-column | Columnar
Orders, payments, inventory | yes, transactions | no | partial | no | no
Sessions, flags, rate limits | partial | yes, by key | partial | no | no
Catalog with varied attributes | partial, JSONB | no | yes | no | no
Chat messages, events at huge scale | partial, until sharded | no | partial | yes | no
Analytics over billions of rows | partial | no | no | no | yes
~~~

Postgres covers more of this table than people expect: JSONB columns with GIN indexes for flexible attributes, partial indexes, full-text search, table partitioning for time-series, and extensions for geospatial data and vectors. "SQL for the core, a key-value cache in front" is the most common real architecture. A second or third store is added when a particular access pattern outgrows it.

~~~dg-tree A decision path for one dataset. Most branches end at the relational database.
Does it need multi-row transactions or joins?
  Yes | Relational
    *Fits on one primary? | Postgres or MySQL
    Too big? | Shard it, or NewSQL like Spanner, CockroachDB
  No
    Fetched whole by one key? | Key-value, Redis or DynamoDB
    Query fields inside nested records? | Document store, or JSONB
    Huge append rate, read by key and time? | Wide-column, Cassandra
    Scans and aggregates over billions? | Columnar, ClickHouse
    Multi-hop relationships? | Graph, rarely needed
~~~

A newer option blurs the line: **NewSQL** or distributed SQL databases (Google Spanner, CockroachDB, YugabyteDB, TiDB) partition data automatically like NoSQL stores but keep SQL and multi-row transactions. They cost more latency per write (consensus between replicas, Module 6) and more money, but they remove the choice between "SQL" and "scale" for many workloads.

## The real costs of NoSQL

The benefits are easy to list. The costs are what an interviewer wants to hear you acknowledge.

- **Queries are designed up front.** In Cassandra or DynamoDB, you design tables around the queries you will run. A new query pattern next year may need a new table and a backfill of all historical data.
- **No joins** means denormalised copies of the same data in several tables, and your code must keep them in sync. When a user changes their name, every copy must change.
- **Weaker transactions** push correctness work into your application: idempotency keys, conditional writes, compensating actions.
- **Eventual consistency** means users can see stale data, such as a message that appears on one device a second before another.
- **Operational cost**: a second database is a second set of backups, upgrades, dashboards, failure modes and on-call knowledge.

## Signals in an interview

Phrases that should make you consider a NoSQL store: "billions of events a day", "write-heavy", "only ever looked up by ID", "time-ordered history per user", "global, always writable". Phrases that should keep you on SQL: "money", "inventory", "booking", "must never double-sell", "reporting", "admin can search by anything". When in doubt, say "Postgres, and here is what would make me move off it".

<div class="ex">
<div class="lab">Worked example</div>

**choosing for a ride-hailing app**

Trips, payments and driver payouts go in Postgres. They need transactions (a trip completing must create a charge and a payout atomically) and constraints. Volume: a city of 2 million rides a day is about 23 trips per second on average, perhaps 200 at peak, trivially within one primary.

Live driver locations go in Redis. 500,000 active drivers sending a location every 4 seconds is 125,000 writes per second, only the latest value matters, and lookups are by driver ID or by area (Redis has geospatial commands). Losing a few seconds of locations in a crash is harmless; they are refreshed in 4 seconds.

Raw GPS traces for analytics (the same 125,000 points per second, kept for a year: about 4 trillion points) go to a columnar store, where queries like "average speed by road segment by hour" scan a few columns over billions of rows.

Three stores, each chosen for one access pattern, and each with a clear reason.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**two workloads where NoSQL is a clear win, and two where it is not**

Win 1: an ad platform logs 2 million impression events per second, each looked up later by campaign and time. That is 170 billion rows a day. Cassandra spreads writes across 100 nodes and reads one campaign-hour partition at a time. A single relational primary cannot absorb that write rate.

Win 2: a game stores session state for 10 million concurrent players, each read and written by player ID 10 times a second, lost harmlessly on expiry. Redis answers in under 1 ms from RAM; a relational database would spend most of its effort on durability nobody needs.

Not a win 1: a bank ledger, where a transfer must debit and credit atomically and never go negative. You want multi-row transactions and check constraints.

Not a win 2: an internal reporting tool where analysts ask new questions every week, joining customers, orders and support tickets. A store designed around fixed queries would need a new table for every question.

</div>

## Common mistakes

- **Choosing NoSQL "for scale"** when the numbers fit on one Postgres primary.
- **Choosing a document store to avoid migrations**, then discovering the schema still exists, now scattered through application code.
- **Using a key-value store for data that needs secondary queries**, then scanning everything.
- **Ignoring the operational cost** of every additional database.
- **Forgetting that NoSQL data models are query-first**, and being surprised when a new feature needs a new table.

<div class="take">
<div class="lab">Key takeaway</div>

use a relational database by default. Pick a NoSQL store when a specific access pattern, scale or latency need is something it handles clearly better, and accept the loss of joins, ad hoc queries and multi-row transactions knowingly. Many real systems use SQL for the core plus one or two specialised stores.

</div>
