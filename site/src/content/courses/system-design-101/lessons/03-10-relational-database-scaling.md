---
id: "03.10"
course: "system-design-101"
module: "03"
slug: "relational-database-scaling"
title: "Relational database scaling"
keyIdeas: ["Scale a relational database step by step: optimize queries and indexes, scale up the machine, add a cache, add read replicas, then partition.", "**Read replicas**: copy the WAL to followers that serve reads. Reads may be slightly stale (replication lag).", "**Federation**: split by function (users DB, orders DB).", "**Sharding**: split one table's rows across databases by a key. Scales writes, but cross-shard joins and transactions get hard.", "**Denormalization**: precompute joins to avoid expensive queries.", "Notion sharded Postgres into hundreds of logical shards on a few physical hosts, keyed by workspace, so a workspace's data stays together."]
question: "In what order would you try read replicas, caching, vertical scaling and sharding?"
answer: "First measure which resource is exhausted. Then: 1) tune queries and indexes (cheapest, often fixes it); 2) cache hot reads in Redis or Memcached; 3) scale vertically to a bigger machine with more RAM and faster disks; 4) add read replicas for remaining read load, handling replication lag with read-your-writes. Caching, replicas and vertical scaling can be interleaved depending on the workload: read-heavy loads favour cache and replicas, while write-heavy loads get nothing from them and go to vertical scaling and then sharding sooner. Shard last, only when write volume or data size outgrows one primary, because it brings cross-shard queries and transactions, hot shards and rebalancing work. When sharding, choose a key that keeps related data together and create many logical shards."
sources: [{"title": "System Design Primer: RDBMS scaling", "url": "https://github.com/donnemartin/system-design-primer#relational-database-management-system-rdbms", "note": "Replication, federation, sharding, denormalization."}, {"title": "Notion: Sharding Postgres", "url": "https://www.notion.com/blog/sharding-postgres-at-notion", "note": "A real sharding migration."}]
---

## Diagnose before you scale

A relational database that is struggling rarely needs sharding first. Sharding is the most powerful option and also the most expensive to build and to live with. Before choosing any fix, find out **what** is running out. The four usual bottlenecks are:

- **CPU**: too many queries, or a few expensive ones (sorts, joins, scans).
- **Memory**: the **working set** (the data and indexes touched regularly) no longer fits in the buffer pool, so reads hit disk and latency jumps from microseconds to milliseconds.
- **Disk I/O**: too many random reads, or too much write and log volume.
- **Connections and locks**: too many open connections, or transactions waiting on each other.

A further category is **operational**: the database is big enough that backups take 12 hours, an index build takes a day, or vacuum cannot keep up (lesson 03.07). Each bottleneck has different fixes, so measure first: the slowest queries by total time (`pg_stat_statements` in Postgres), cache hit rate, CPU, I/O wait and replication lag.

## The order to try things

Cheapest and least disruptive first.

**1. Tune queries and indexes.** Find the queries that consume the most total time, read their plans (lesson 03.05) and fix them: add the right composite index, remove a function from an indexed column, stop selecting columns you do not need, replace `OFFSET` pagination (lesson 04.12). Fix the schema too: sensible types (`DECIMAL` for money, integers for IDs), and move rarely used large columns to a side table so hot rows are small. Often a single missing index is the whole problem. This step costs hours, not months.

**2. Add a connection pooler.** Each Postgres connection is a separate process using several megabytes. Hundreds of app servers each holding 20 connections can exhaust the database before any query is slow. A pooler such as PgBouncer lets thousands of client connections share a few dozen real ones.

**3. Cache hot reads.** Put Redis or Memcached in front of the database for data that is read far more often than written: profiles, product pages, configuration (Module 5). A cache hit costs well under a millisecond and no database work at all, so a 90% hit rate removes 90% of those reads. The price is staleness and invalidation logic.

**4. Scale vertically.** Move to a bigger machine: more RAM so the working set fits in memory, faster NVMe disks, more cores. Cloud instances reach hundreds of cores and several terabytes of RAM. It is boring and it works, with no code changes. But it has a ceiling, the price grows faster than linearly near the top, and some limits are not hardware at all: at Notion, vacuum degraded and maintenance became risky well before the hardware ran out. Vertical scaling is a bridge, not a destination.

**5. Add read replicas.** Replicas receive the primary's write-ahead log (lesson 03.08) and replay it, then serve read-only queries. They multiply read capacity and give you a standby for failover. Caveats:

- **Replication lag**: replicas are milliseconds to seconds behind. A user who saves a profile and is immediately routed to a replica may see the old version.
- **Writes do not scale**: every write still goes to the one primary, and every replica replays every write.
- **Failover** can lose writes not yet copied (with asynchronous replication), and promoting a replica needs tooling.

**6. Partition by function (federation).** Split into separate databases per domain: users, orders, payments, messaging. Each is smaller, and writes to different domains run on different machines in parallel. The cost: joins across domains move into the application, and a transaction that touches two domains needs extra care (lesson 08.07 covers sagas).

**7. Shard.** Split one table's rows across many databases by a **shard key**, such as `user_id`. Each shard holds a slice and takes a slice of the writes, so write capacity and storage scale out. Costs: queries that span shards are slow or impossible, joins and transactions across shards are hard, unique constraints only hold within a shard, an uneven key creates hot shards, and moving data between shards is complex (Module 4).

~~~dg-bars Rough engineering effort for each scaling step, in engineer-weeks. Sharding dwarfs the rest.
Tune queries and indexes | 1 | days
Connection pooler | 1 | days
Cache hot reads | 3 | weeks
Bigger machine | 1 | a maintenance window
Read replicas | 3 | weeks
Split by domain | 12 | months
*Shard | 40 | many months
~~~

### The order is a guide, not a law

Caching and replicas usually come before vertical scaling only when the load is read-heavy and cheap to cache. In practice teams interleave steps 3 to 5, and a write-heavy load skips caching and replicas because they do not help writes at all. **Denormalisation** (storing a copy, such as a `like_count` column, to avoid an expensive join or count) also helps reads, at the cost of keeping copies in sync.

~~~dg-matrix What each step helps with. Only splitting and sharding increase write capacity and total data size.
Step | Read load | Write load | Data size | Code change
Tune queries | yes | yes | no | small
Cache | yes | no | no | moderate
Bigger machine | yes | yes | partial, to a limit | none
Read replicas | yes | no | no | small, routing
Split by domain | yes | yes | yes | large
Shard | yes | yes | yes | very large
~~~

## Living with replicas: read-your-writes

The most common replica bug: a user updates their display name, the write goes to the primary, the next page load reads from a replica that is 300 ms behind, and the user sees the old name. They assume the save failed and try again.

The fixes, from simplest:

- **Read your own data from the primary** for a short window after writing, for example 5 seconds, tracked in the session.
- **Track the write position**: remember the WAL position (LSN) of the user's last write and only read from a replica that has replayed past it.
- **Route by data**: always read a user's own profile from the primary; read other users' profiles from replicas.

~~~dg-seq Read-your-writes on replicas: the app remembers the user's last write and avoids replicas that have not caught up.
User > App : save name "Ana"
App > Primary : UPDATE users, commit at LSN 900
App -> User : saved, remember LSN 900 in session
User > App : load profile
App > Replica : replayed past LSN 900?
Replica -> App : no, at LSN 870
App > Primary : read profile
Primary -> App : name "Ana"
~~~

## Case study: Notion shards Postgres

Notion ran on a single Postgres database for five years, through four orders of magnitude of growth. By mid-2020 CPU spikes were paging on-call engineers, schema changes had become unsafe, vacuum was stalling and transaction ID wraparound loomed (lesson 03.07).

### Decisions

- **Application-level sharding**, rather than an extension such as Citus or a proxy such as Vitess, to keep full control over data placement and behaviour.
- **Shard everything reachable from a block by foreign key.** A block is Notion's unit of content. Comments, discussions and other tables that hang off blocks go to the same shard, so related data stays together and transactions within it still work.
- **Partition key: workspace ID.** Every block belongs to exactly one workspace, and users work within one workspace at a time, so nearly every query touches one shard.
- **480 logical shards on 32 physical databases**, 15 per host. Each logical shard is a Postgres schema. Many small logical shards mean you can later move whole shards to new hosts without splitting any of them. 480 was chosen because it has many factors: it divides evenly by 32, 40, 48, 60, 80 and more, so the fleet can grow 32 to 40 to 48 hosts with an even spread. 512 would only divide evenly by powers of two.

~~~dg-tree Notion's layout: workspace ID picks one of 480 logical shards, and each physical host holds 15 of them.
Workspace ID
  hash to logical shard 0 to 479
    Host 1 | logical shards 0 to 14
      schema0001 | blocks, comments, spaces
      schema0002
    Host 2 | logical shards 15 to 29
    *Host 32 | logical shards 465 to 479
~~~

### Migration

1. **Double-write**: new writes went to both the old database and the shards. Writing directly to both from the application proved flaky, and logical replication could not keep up, so writes were recorded in an audit log and a catch-up script applied them to the shards.
2. **Backfill** existing data, which took about three days on a 96-vCPU machine. Records were skipped if the shard already held a newer version.
3. **Verify** by sampling records and by "dark reads": serving each read from the old database but also reading from the shards and comparing the results. Different engineers wrote the migration code and the verification code, so the same bug was unlikely to appear in both.
4. **Switch over** in about a five-minute maintenance window.

The main lesson Notion drew: **shard earlier**, before the old database is so strained that it struggles to absorb the extra load of the migration itself.

~~~dg-timeline Notion's move from one Postgres database to 480 logical shards.
First 5 years | One Postgres, four orders of magnitude of growth
Mid 2020 | *CPU spikes, unsafe migrations, vacuum stalls
Planning | Workspace ID key, 480 shards on 32 hosts
Step 1 | Double-write via audit log
Step 2 | Backfill, about 3 days on 96 vCPUs
Step 3 | Dark reads compare old and new
Cut-over | *About 5 minutes of maintenance
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a social app at 50,000 queries per second**

95% of traffic is reads of profiles and posts: 47,500 reads and 2,500 writes per second. The primary is at 85% CPU and p99 latency is 400 ms.

Step 1: `pg_stat_statements` shows one query, "recent posts by user", takes 40% of total database time. `EXPLAIN` reveals a missing index on `posts(user_id, created_at)`. After adding it (concurrently, lesson 03.11), CPU falls to 55% and p99 halves.

Step 2: cache profiles and post bodies in Redis with a 60-second TTL. The hit rate is 85%, removing about 40,000 reads per second from the database.

Step 3: add two read replicas for the remaining 7,500 reads per second, with read-your-writes handled by reading the user's own profile from the primary.

Writes at 2,500 per second fit easily on one large primary. Sharding waits until write volume or data size demands it, but the plan (shard by `user_id`, many logical shards) is written down now so new tables are designed with it in mind.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**when replicas are the wrong tool**

An IoT platform ingests 60,000 sensor readings per second and the primary's disk is saturated by writes and WAL. The team adds three replicas and nothing improves: each replica must replay all 60,000 writes per second too, and reads were never the problem.

The right steps are write-side: batch inserts (one insert of 500 rows instead of 500 inserts), drop unused indexes on the readings table, use time-based partitions so old data is dropped as whole partitions instead of deleted row by row, and if that is not enough, shard by device ID or move the readings to a store built for this write pattern (Module 7).

</div>

## Common mistakes

- **Sharding first.** Months of work and permanent complexity for a problem an index would have fixed.
- **Adding replicas for a write problem.** Every replica replays every write.
- **Ignoring replication lag** until users report "my change disappeared".
- **Waiting too long to shard.** The migration itself needs headroom on the old system.
- **Too few logical shards.** If you start with exactly as many shards as hosts, growing means splitting shards, which is far harder than moving them.

## In an interview

When the interviewer says "the database is the bottleneck", ask which resource: reads, writes, data size or connections. Then walk the ladder out loud: tune and index, pool connections, cache, scale up, replicas for reads, and shard only for writes or size. When you do shard, name the key, say which queries it keeps on one shard and which become scatter-gather, and mention many logical shards for future moves.

<div class="take">
<div class="lab">Key takeaway</div>

find the bottleneck first, then tune, cache, scale up and add replicas before sharding. Replicas and caches only help reads; splitting and sharding are the tools for write volume and data size. When you shard, pick a key that keeps related data and common queries on one shard, and use many logical shards so you can move them later.

</div>
