---
id: "09.10"
course: "system-design-101"
module: "09"
slug: "search-index-synchronization"
title: "Search index synchronization"
keyIdeas: ["The search index is usually a copy of data whose source of truth is a database. It must stay in sync.", "Dual writes (app writes DB, then index) drift: if one write fails or they race, the index is wrong forever.", "Safer: **change data capture**. Read the DB's replication log (Debezium on Postgres/MySQL WAL), publish changes to Kafka, and an indexer updates the search index. Order and completeness come from the log.", "Reindexing: build a new index from a snapshot plus the change stream, then switch an alias atomically.", "Accept some lag (usually seconds) and design the UI for it."]
question: "Why is dual-writing to the DB and the search index risky? What's the safer pattern?"
answer: "Dual writes are not atomic: if the database write succeeds and the index write fails or the app crashes in between, the index stays wrong indefinitely, and concurrent updates can reach the index in a different order from the one in which they committed, leaving stale values; any script or code path that forgets the second write also causes drift. The safer pattern is log-based change data capture: the app writes only to the database, a tool such as Debezium reads the write-ahead log or binlog and publishes ordered change events (including deletes) to Kafka keyed by ID, and an idempotent indexer upserts each document using the log position as an external version, resuming from its offset after failures. For complex documents, the same can be done through a transactional outbox table."
sources: [{"title": "Debezium: Features", "url": "https://debezium.io/documentation/reference/stable/features.html", "note": "CDC from DB to search index."}, {"title": "Hello Interview: Elasticsearch deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/elasticsearch", "note": "Keeping ES in sync."}, {"title": "Hello Interview: Change Data Capture", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/change-data-capture", "note": "CDC explained."}]
---

## The search index is a copy

A search engine such as Elasticsearch should not be your primary store. The **source of truth** is a database such as Postgres, MySQL or DynamoDB, with transactions, constraints and backups. The search index is a **derived copy** of some of that data, reshaped for search: denormalised (a product document includes its brand name, category path and stock status from other tables), analysed into terms, and missing anything search doesn't need.

Keeping a derived copy in sync is the whole problem of this lesson. Every change in the database (a new product, a price change, a deletion, a stock update) must reach the index, **exactly as it happened, in the right order, eventually, without anything slipping through**. If it doesn't, users see products that no longer exist, miss products that do, or see a price that's wrong. Search drift is a quiet failure: nothing errors, the results are just wrong.

There are three broad ways to sync: write to both from the application (dual writes), poll the database for changes, or stream the database's own change log (change data capture). Only the last is safe and efficient at scale.

## Why dual writes go wrong

The tempting approach is for the application to write to both: update the database, then update the index.

```
db.execute("UPDATE products SET price = 80 WHERE id = 123")
es.update(index="products", id=123, doc={"price": 80})
```

This **dual-write** pattern looks fine in testing, but the two writes are **not atomic**, and nothing coordinates them.

~~~dg-seq A dual write that fails halfway: the database has the new price, the index still shows the old one, and nothing will fix it.
App > Database : UPDATE price = 80
Database -> App : committed
App > Search index : update price = 80
Search index -> App : timeout (write lost)
App > App : crash before retry
~~~

### The failure cases

- **Partial failure.** One write succeeds and the other fails: a timeout, a crash, a deploy, a full disk on the search cluster. The index is now permanently wrong until something happens to rewrite that document, which may be never. Retrying in-process helps only if the process survives.
- **Race conditions.** Two requests update the same row. They commit to the database in order A then B, but their index writes arrive in order B then A (different threads, different network paths, a GC pause). The database ends with B's value; the index ends with A's stale value. No error is raised anywhere.
- **Every writer must remember.** Any code path, batch job, migration script or admin tool that changes the database without also updating the index silently creates drift. In a large codebase, someone always forgets.
- **No shared transaction.** You can't wrap both writes in one transaction, because the search engine doesn't take part in the database's transactions. Two-phase commit isn't available.
- **Writing the index first** is worse: if the database transaction then rolls back, the index shows a change that never happened.

<div class="ex">
<div class="lab">Worked example</div>

**the race, with timestamps**

At 10:00:00.000 a merchant sets price 90 (request A). At 10:00:00.010 a bulk discount job sets price 80 (request B). Postgres commits A then B; the row ends at 80. A's index update is delayed 40 ms by a slow connection; B's goes straight through. The index receives 80 at 10:00:00.015, then 90 at 10:00:00.045. The index shows 90; the database says 80. Customers searching "under \$85" don't find the product, and it stays that way until the next edit.

</div>

## Polling

A step better: write only to the database, and run a job that periodically queries for changes: `SELECT * FROM products WHERE updated_at > :last_seen ORDER BY updated_at`. It's simple, and every writer is covered automatically. Its problems:

- **Deletes are invisible**: a deleted row isn't returned by any query. You need soft deletes (a `deleted_at` column) or a separate tombstone table.
- **Needs an `updated_at` column** that every writer maintains correctly, ideally by trigger.
- **Clock and commit-order issues**: a long transaction that started earlier can commit with an older `updated_at` after the poller has moved past it, so the change is skipped.
- **Latency versus load**: polling every second adds steady query load; polling every minute makes the index a minute stale.

Polling is acceptable for small systems with relaxed freshness. For anything serious, read the log.

## Change data capture

The safer pattern is to write **only** to the database, and let a separate process read the **database's own change log** and apply each change to the index. This is **change data capture** (CDC).

Every serious database already keeps an ordered log of committed changes for crash recovery and replication: the **write-ahead log** (WAL) in Postgres, the **binlog** in MySQL, **change streams** in MongoDB, **DynamoDB Streams**. A CDC connector reads that log like a replica would and turns each committed change into an event. **Debezium** is the common open-source tool, usually running on Kafka Connect and publishing to Kafka topics.

What log-based CDC gives you:

- **All changes are captured, including deletes**, because they're all in the log. No polling queries, no `updated_at` column, no soft deletes.
- **Low delay** (typically milliseconds to a second) and **low overhead** on the database compared with polling queries.
- **Commit order**: events come out in the order transactions committed, so the race above disappears.
- **No change to the data model or to application code**: every writer, including scripts and admin tools, is covered automatically.
- **Rich events**: each event includes the row's **old state** and **new state**, the operation (create, update, delete), and metadata such as the transaction ID and log position.
- **Initial snapshots**: to start, the connector reads a consistent snapshot of existing rows, then switches to streaming from the log position the snapshot corresponds to. Incremental snapshots let you re-snapshot tables without pausing the stream.
- **Filtering and transformation**: include only some tables or columns, mask sensitive fields, reshape events with single message transformations.

~~~dg-flow CDC: the application writes once; the database's log becomes the ordered stream that keeps the index in sync.
[Write] App -(single write)-> Database -(WAL or binlog)-> *Debezium
[Sync] Debezium -(change events, keyed by id)-> Kafka -(indexer consumer)-> Search index
~~~

Because the change events are published to Kafka **keyed by primary key**, all changes to one row go to one partition and stay in order (lesson 08.02). If the indexer crashes, it restarts from its last committed offset, so nothing is lost. A brand-new index can be built by running a fresh snapshot plus the stream. And other consumers (a cache invalidator, an analytics pipeline) can read the same change stream for free.

~~~dg-compare Three ways to keep a search index in sync with the primary database.
[Dual write]
App -> Database
App -> Search index
- Partial failures leave drift
- Races reorder updates
- Every writer must remember
[Polling]
Database -(updated_at query)-> Poller -> Index
+ Single write path
- Misses deletes and late commits
- Load versus staleness trade
[*Log-based CDC]
Database -(WAL)-> Debezium -> Kafka -> Indexer
+ All changes incl. deletes, in order
+ Replayable, sub-second lag
- More infrastructure to run
~~~

### The outbox variant

CDC on raw tables makes your table schema the event schema. If the search document needs data from several tables, or you want deliberate, stable event shapes, use the **transactional outbox** (lesson 08.03): write a `ProductChanged` row to an outbox table in the same transaction, and stream only the outbox with CDC.

## What you still have to handle

### Lag and read-your-writes

The index is **eventually consistent**: behind the database by the CDC delay plus the indexer's processing plus the search engine's refresh interval (1 second by default in Elasticsearch). Typically one to a few seconds. A merchant who saves a product and immediately searches for it may not see it. Common fixes:

- On the item's own page, read from the database, not the index.
- After a user's edit, show their change directly in the UI (optimistic display) or overlay their recent edits on search results.
- For the rare request that truly needs it, ask the engine to refresh (`refresh=wait_for`) and accept the latency cost.

### Idempotent, versioned writes

Replays and redeliveries can apply an event twice, and after a re-snapshot old events can arrive after new ones. The indexer should **upsert by document ID** and use **external versioning**: pass the log position (Postgres LSN) or a row version as the document's version, and have the engine reject writes with a lower version than the stored one. Elasticsearch supports this with `version_type=external`. Then applying an event twice, or an old event late, is harmless.

### Denormalisation

A product search document joins several tables: product, brand, category, inventory, reviews summary. A change to the brand's name must update every product document of that brand. The indexer must react to changes in **any** source table, look up the affected documents, and rebuild them. Keep the fan-out bounded: a brand rename touching 2 million products is a bulk reindex job, not a per-event action.

### Reindexing

When the mapping changes (a new analyser, a new field), you can't change existing documents in place. Build a **new index** (`products_v8`) from a snapshot while the stream continues, let it catch up, then atomically switch an **alias** (`products`) from v7 to v8. Queries always use the alias, so the switch is invisible to users.

~~~dg-timeline A zero-downtime reindex: build the new index beside the old one, catch it up from the stream, then flip the alias.
[v7] 09:00 | Alias products serves v7, CDC keeps it current
[v8] 09:00 | Create products_v8 with new mapping
[v8] 09:05 | Snapshot copy starts at log position 88,100
[v8] 11:40 | Snapshot done; replay stream from 88,100
[v8] 11:52 | *Caught up: lag under 1 s
[Alias] 11:53 | Flip alias products to v8
[v7] 12:30 | Delete v7 after checks
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a price change, end to end**

A merchant sets a shoe's price to 80. The app commits one `UPDATE` to Postgres. Debezium reads the WAL entry and publishes `{op: "u", before: {price: 95}, after: {price: 80}, source: {lsn: 4411}}` to the Kafka topic `db.products`, keyed by product 123. The indexer consumes it, rebuilds document 123 and writes it with `version = 4411, version_type = external`. About 1.2 seconds after the commit, the next refresh makes it searchable.

The indexer crashes after writing but before committing its Kafka offset. On restart it re-reads the event and writes version 4411 again; with external versioning a write must carry a strictly higher version, so Elasticsearch answers with a version conflict, which the indexer treats as "already applied" and moves on. Later, a re-snapshot replays an older event with LSN 4300 for the same product; Elasticsearch rejects it as an older version. The index ends correct.

</div>

## Monitoring

- **End-to-end lag**: time from database commit to searchable. Alert if it exceeds, say, 30 seconds.
- **Connector health and replication slot size**: in Postgres, a stuck CDC connector keeps its replication slot open, and the database must retain WAL until it catches up, which can fill the disk. Alert on retained WAL size.
- **Periodic reconciliation**: sample documents and compare index against database, or compare counts per category, to catch drift from bugs.

## Common mistakes

- Dual writes "with retries" in application code.
- Polling without handling deletes.
- Indexer writes without versioning, so replays or re-snapshots overwrite newer data.
- Forgetting the Postgres replication slot, which can fill the primary's disk when the connector stops.
- Changing mappings in place instead of reindexing behind an alias.

## In an interview

Say: "The database is the source of truth. I won't dual-write: it's not atomic and updates can arrive out of order. I'll use log-based CDC (Debezium on the WAL) into Kafka keyed by ID, and an indexer that upserts with external versions so replays are harmless. The index lags by a second or two, so item pages read from the database. Mapping changes go through a new index and an alias flip."

<div class="take">
<div class="lab">Key takeaway</div>

dual writes to a database and a search index drift apart on partial failures and races, and polling misses deletes. Write only to the database and let log-based change data capture stream ordered changes, including deletes, to an idempotent, versioned indexer, and reindex behind an alias.

</div>
