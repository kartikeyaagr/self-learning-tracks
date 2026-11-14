---
id: "03.02"
course: "system-design-101"
module: "03"
slug: "a-key-value-store-on-sql"
title: "A key-value store on SQL"
keyIdeas: ["You can build a key-value store on a relational database: a table with `key` primary key and `value` blob, plus a version or timestamp.", "Why bother: SQL databases are well understood, durable, have mature backup and replication tooling, and your team already runs them.", "Uber's Schemaless and FriendFeed stored JSON blobs in MySQL, sharded by key, with separate index tables for secondary lookups.", "Append-only cells (never update, always insert a new version) simplify replication and give history.", "Limits: no rich queries over values, and you build sharding and secondary indexes yourself."]
question: "Why would a company build a KV store on top of SQL instead of adopting a NoSQL product?"
answer: "Because the storage engine is proven and the team already knows how to run it: crash recovery, replication, backups, failover and 3 a.m. debugging are solved problems. Storing values as blobs means new fields need no migrations, and building the layer yourself lets you choose the trade-offs, such as eventually consistent secondary indexes, buffered writes during failover, and append-only cells that double as a change log for triggers. A new NoSQL product brings unfamiliar failure modes, and at the time few products met requirements like accepting writes during failover or built-in change notification. The cost is that you own sharding, routing and index repair, and lose joins and cross-shard transactions."
sources: [{"title": "Uber: Schemaless, part one", "url": "https://www.uber.com/blog/schemaless-part-one-mysql-datastore/", "note": "A key-value store built on MySQL."}, {"title": "How FriendFeed uses MySQL", "url": "https://backchannel.org/blog/friendfeed-schemaless-mysql", "note": "The same idea, earlier."}]
tags: ["Design"]
tools: ["PostgreSQL"]
---

## The surprising pattern

When a relational database starts to hurt at scale, the obvious move is to adopt a NoSQL product built for scale. Yet several large companies did something that looks backwards: they kept MySQL and built a key-value or document store **on top of it**. Two well-known cases are FriendFeed in 2009 and Uber's Schemaless in 2016. Facebook's TAO, which serves the social graph, also sits on sharded MySQL, and Pinterest and Dropbox both built similar layers.

A **key-value store** is the simplest kind of database: you `put(key, value)` and `get(key)`, and the store does not understand the value. A **document store** is a key-value store whose values are structured documents (usually JSON) that it can partly index. The question this lesson answers is why a company would build one of these itself, on a relational engine, rather than download one.

## FriendFeed: schema changes had become impossible

FriendFeed was a social feed aggregator. It stored over 250 million entries in sharded MySQL, with read replicas and memcache in front. Traffic was not the problem. The problem was **change**. In the MySQL of that era, adding an index or a column to a table beyond 10 to 20 million rows rewrote the table and locked it for hours. Product teams started avoiding features that needed a schema change, which is a terrible place for a young company to be.

### The design

- **One entities table** holds every object as a compressed blob (a serialised Python dictionary) keyed by a 16-byte UUID. There is no column per property. Adding a property is just writing it into the blob. No migration, no lock.
- **Each index is its own table.** To query entries by user, there is a table `index_user_id(user_id, entity_id)`, sharded by `user_id`. Adding a new index means creating a new, empty table, then letting a background process called the **Cleaner** populate it from the entities while the site runs. Index rollouts went from about a week to about a day.
- **Joins happen in the application.** A query reads the index table to get entity IDs, then fetches those entities by primary key from their own shards.

~~~dg-tree FriendFeed's layout: one table of opaque blobs, plus one small index table per query pattern, each sharded independently.
Sharded MySQL
  entities | source of truth
    id | 16-byte UUID, primary key
    updated | timestamp
    body | compressed blob of all properties
  index_user_id | sharded by user_id
    user_id + entity_id
  index_link | sharded by link
    link + entity_id
  *index_new_feature | created later, filled by the Cleaner
~~~

### Living without cross-shard transactions

Because an entity and its index rows live on different shards, there is no transaction that covers both. If the server crashes between writing the entity and writing the index, they disagree. FriendFeed made that safe with four rules:

1. The entities table is the **source of truth**. Index tables are only hints.
2. Writes go to the entity first, then to the indexes.
3. Reads use an index only to find **candidate** IDs, then fetch the real entities and re-check the filter against them. A stale index row that points at an entity that no longer matches is simply discarded.
4. The Cleaner continuously scans entities and repairs missing or wrong index rows, usually within a couple of seconds.

The effect is that an index can be briefly incomplete (a new entry might not show up in a user's list for a second or two) but it can never return a wrong result, because every result is verified against the source of truth.

~~~dg-seq A FriendFeed write and a later read. The read re-checks each candidate against the entity, so a stale index row is harmless.
App > Entities shard : write entity 9f3 (user 42, link L)
App > Index user shard : insert (42, 9f3)
App > Index link shard : insert (L, 9f3)
App > Index user shard : find entities for user 42
Index user shard -> App : 9f3, 7a1
App > Entities shard : fetch 9f3, 7a1
Entities shard -> App : 7a1 now belongs to user 50
App > App : drop 7a1, return 9f3
~~~

## Uber Schemaless: running out of Postgres

In early 2014, Uber projected that trip growth would exhaust the storage of its single Postgres database before the end of the year. The requirements for a replacement were concrete:

- Add capacity by adding servers (horizontal scaling).
- Keep **accepting writes** even while a database is failing over, because a trip that cannot be written is lost revenue.
- **Notify downstream systems** (billing, analytics, receipts) when data changes.
- Support **secondary indexes**, such as "trips by driver".
- Be something the on-call team could debug at 3 a.m.

The team evaluated Cassandra, Riak and MongoDB. All three could scale out, but few accepted writes during failover in the way Uber needed, none had a built-in change notification mechanism, and the team lacked operational experience with them. The decisive factor was operational confidence: they knew MySQL deeply, including how it fails.

### The data model

Schemaless borrows from Bigtable's model. Data is a sparse three-dimensional map.

- A **cell** is an immutable JSON blob addressed by a **row key** (a UUID, for example the trip ID), a **column name** (a string such as `BASE`) and a **ref key** (an integer version).
- Updating means writing a new cell with a higher ref key. Nothing is ever overwritten in place.
- A trip stores `BASE`, `STATUS`, `NOTES` and `FARE_ADJUSTMENT` as separate columns. Different services write different columns at different times, so they never conflict on the same cell.

~~~dg-tree How a Schemaless trip is addressed: row key, then column, then version. Every version is kept.
trip 7c1e... | row key, a UUID
  BASE | written by dispatch
    ref 1 | pickup, dropoff, fare
  STATUS | written by billing
    ref 1 | charge pending
    *ref 2 | charge succeeded
  FARE_ADJUSTMENT | written by support
    ref 1 | refund 4.50
~~~

### Triggers: the store is also a change log

Because the data is append-only, each shard is effectively an ordered log of every change. Schemaless exposes this as **triggers**: a service subscribes to a column and is called for each new cell. When a trip's `BASE` cell is written at the end of a ride, the billing service's trigger fires, charges the card and writes a `STATUS` cell. A failed charge is retried because the trigger tracks how far it has read, much like a consumer's position in a message queue (Module 8).

~~~dg-timeline One trip's life in Schemaless: separate services write separate columns, and triggers chain the work.
0 s | Ride ends, dispatch writes BASE ref 1
0.1 s | *Billing trigger fires on new BASE cell
2 s | Billing writes STATUS ref 1, charge pending
3 s | Card charged, billing writes STATUS ref 2
30 s | Receipt service trigger fires on STATUS ref 2
2 days | Support writes FARE_ADJUSTMENT ref 1
~~~

### Secondary indexes

Indexes are sharded by a chosen field. Driver UUID is a good choice because it spreads evenly. City ID is a bad choice because a few large cities would overload a few shards (lesson 04.06 covers this "hot partition" problem). Indexes are **eventually consistent**, typically lagging well under 20 ms. Making them strictly consistent would require two-phase commit (a protocol that coordinates a commit across several databases) between the cell's shard and the index's shard, which Uber judged too slow and too fragile.

### Writes during failover

Each Schemaless cluster had a MySQL primary and replicas. If the primary for a shard was down, a write would be **buffered**: written to a different, healthy shard's buffer table, then copied to the right place once the primary returned. Because cells are immutable and versioned, replaying a buffered write later is safe. This is a good example of a feature that is easy to build in your own thin layer and hard to get from someone else's product.

## Why build on SQL?

The reasons generalise beyond these two companies:

- **A proven storage engine.** InnoDB has decades of hardening: crash recovery through a write-ahead log (lesson 03.08), replication, consistent backups, checksums, monitoring tools. Writing a new storage engine that is that reliable takes years.
- **Operational knowledge.** The team already knows how to tune it, back it up, restore it and debug it. In an outage, that knowledge is worth more than any feature list.
- **Use only the safe parts.** Each shard is used as a durable, sorted key-value store with single-row atomic writes. You skip the parts that do not scale across shards (joins, cross-shard transactions, global constraints).
- **You choose the trade-offs.** Eventually consistent indexes, buffered writes, triggers, retention rules: these are your decisions, not a vendor's defaults.

~~~dg-compare Adopting a NoSQL product versus building a thin key-value layer on sharded MySQL, as seen by a team that already runs MySQL.
[Adopt NoSQL product]
Cluster handles sharding and replication
+ Scaling is built in
+ Less code to own
- New failure modes to learn
- Trade-offs fixed by the product
- New backup and monitoring stack
[*KV layer on MySQL]
Your service routes keys to MySQL shards
+ Proven engine and tooling
+ Team already knows it
+ Custom features like triggers
- You own routing and resharding
- You own index repair
~~~

### What it costs

You now own a database layer. Sharding, request routing, resharding when shards fill up, index maintenance and consistency repair are your code, and they need tests, on-call and documentation. You also give up ad hoc SQL: an analyst cannot write a join across blobs. Most companies that did this were large enough to staff a team for it. For a ten-person start-up, the trade usually goes the other way: use a managed database and spend the engineering time on the product.

The modern picture has also moved. MySQL 8 and Postgres can add many columns and indexes without long locks (lessons 03.11 and 03.12), and both have good JSON column types with indexes. So the original FriendFeed pain is smaller today, and "JSON column in a normal table" is often enough.

<div class="ex">
<div class="lab">Worked example</div>

**a user settings store**

A team needs a flexible per-user settings store for 500 million users. Each user has about 30 settings, and new setting types appear every week.

Instead of a new database, they create `settings(user_id BIGINT, key VARCHAR(64), value_json JSON, updated_at TIMESTAMP, PRIMARY KEY (user_id, key))` on 64 MySQL shards, routed by `user_id % 64` through a small library (lesson 04.07 explains a better routing scheme).

Size: 500M × 30 = 15 billion rows. At about 120 bytes per row with index overhead, that is 1.8 TB, or about 28 GB per shard. That fits comfortably on each shard's SSD and mostly in memory.

Reads are "all settings for user X", which is one range scan on one shard: about 1 ms. New setting types need no migration. Backups, failover and dashboards are the same ones the DBAs already run.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the crash between entity and index**

A FriendFeed-style write stores entity E on shard 3, then the server crashes before the index row (user 42, E) is written on shard 7.

For the next few seconds, a query for user 42's entries does not return E. Nothing incorrect is shown; E is just missing.

The Cleaner's scan reaches E, sees that its user is 42, finds no index row on shard 7, and inserts it. From then on E appears. The window of inconsistency is the Cleaner's scan interval, about 2 seconds, and the failure mode is "missing for a moment", never "wrong".

</div>

## Common mistakes

- **Treating the index as truth.** If reads trust index rows without re-checking the entity, a crash leaves permanently wrong answers.
- **Sharding an index by a skewed field.** A city or country index puts New York on one overloaded shard.
- **Building it too early.** The layer is a product in itself. Without a team to run it, a managed database wins.
- **Forgetting that blobs hide the schema.** The schema still exists, now in application code. Version your blob formats and handle old versions when reading.

## In an interview

If asked "why not just use Cassandra?", answer with requirements, not fashion: operational familiarity, write availability during failover, change notification, and the cost of owning a new failure mode. Then name the price: application-level joins, eventually consistent indexes and a team that owns routing and resharding.

<div class="take">
<div class="lab">Key takeaway</div>

FriendFeed and Uber used MySQL as a reliable, well-understood storage engine and built a thin schemaless layer on top. They traded SQL features (joins, cross-shard transactions, strict indexes) for painless schema change, custom features like triggers, and operational confidence.

</div>
