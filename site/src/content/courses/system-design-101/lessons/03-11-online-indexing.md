---
id: "03.11"
course: "system-design-101"
module: "03"
slug: "online-indexing"
title: "Online indexing"
keyIdeas: ["A normal `CREATE INDEX` locks the table against writes for the whole build, which on a large busy table can mean minutes of failed writes.", "`CREATE INDEX CONCURRENTLY` (Postgres) builds the index in multiple passes while allowing writes. Slower and can fail, leaving an invalid index you must drop and retry.", "MySQL's online DDL and tools like gh-ost or pt-online-schema-change do similar things by copying the table in the background.", "Always check lock behaviour of any schema change before running it in production, and set a lock timeout."]
question: "Why can a normal CREATE INDEX cause an outage on a busy table?"
answer: "A normal CREATE INDEX holds a lock that blocks every INSERT, UPDATE and DELETE on the table until the build finishes, and on a large table that is minutes to hours. On a busy table, writes pile up behind the lock, the application's connection pool fills with waiting requests, new requests time out and retries add load, so the application goes down even though the database is healthy. Worse, the build must first wait for any in-progress write transaction on the table, and new writes queue behind it while it waits (for ALTER TABLE, which needs an exclusive lock, even reads queue). CREATE INDEX CONCURRENTLY avoids blocking writes by building in two scans and waiting for old transactions, at the cost of a slower build, extra load, and the risk of leaving an INVALID index to clean up if it fails. Setting lock_timeout on migrations prevents the lock queue pile-up."
sources: [{"title": "PostgreSQL docs: CREATE INDEX CONCURRENTLY", "url": "https://www.postgresql.org/docs/current/sql-createindex.html#SQL-CREATEINDEX-CONCURRENTLY", "note": "Building indexes without blocking writes."}]
tools: ["PostgreSQL"]
---

## Why CREATE INDEX can take a site down

You have found the missing index that will fix your slowest query (lesson 03.03). You run `CREATE INDEX orders_customer_idx ON orders (customer_id)` on production. Ten seconds later, alerts fire: checkout is timing out.

A plain `CREATE INDEX` in Postgres takes a `SHARE` lock on the table for the whole build. That lock allows reads but **blocks every `INSERT`, `UPDATE` and `DELETE`**. The lock is held while the database scans the entire table, sorts all the values and writes the new B-tree. On a table of hundreds of millions of rows that takes minutes to hours.

On a busy table even a short block is a disaster. Picture a table taking 3,000 writes per second. Every write that arrives during the build waits. The application's connection pool fills with waiting requests within a second or two. New requests cannot get a connection and time out. Upstream services retry, adding load. The database is healthy, CPU is fine, yet the application is effectively down.

## The lock queue trap

There is a second trap, and it applies to any statement that needs a strong lock, including quick ones like `ALTER TABLE ... ADD COLUMN`.

Locks in Postgres are granted in queue order. A request for a strong lock must wait until every transaction currently holding a conflicting lock finishes. **While it waits, every new query that needs a conflicting lock queues behind it**, even plain `SELECT`s if the waiting lock is `ACCESS EXCLUSIVE`. So:

1. A reporting query has been reading `orders` for 4 minutes and will run for 6 more.
2. A migration runs `ALTER TABLE orders ADD COLUMN note text`, which needs `ACCESS EXCLUSIVE`. It would take 2 ms, but it must wait for the report.
3. Every new query on `orders`, reads included, now queues behind the migration.
4. For 6 minutes, nothing can touch `orders`.

A statement that would have taken milliseconds causes a 6-minute outage. The defence is to set a short **`lock_timeout`** on migrations (for example `SET lock_timeout = '2s'`). If the lock is not granted in 2 seconds, the migration fails, the queue drains instantly, and you retry later.

~~~dg-seq The lock queue trap: a 2 ms ALTER waits behind a long report, and every new query waits behind the ALTER.
Report > orders : SELECT, runs 10 minutes
Migration > orders : ALTER TABLE, wants exclusive lock
orders > Migration : wait for report
App query 1 > orders : SELECT
orders > App query 1 : wait behind migration
App query 2 > orders : INSERT
orders > App query 2 : wait behind migration
Report -> orders : finishes after 6 more minutes
orders -> Migration : lock granted, 2 ms
orders -> App query 1 : finally runs
~~~

## CREATE INDEX CONCURRENTLY

`CREATE INDEX CONCURRENTLY` builds the index without blocking writes. It takes a weaker lock (`SHARE UPDATE EXCLUSIVE`) that lets inserts, updates and deletes continue, and solves the problem of rows changing during the build by working in phases:

1. **Register the index** in the system catalog, marked as **not ready** and **invalid**. Then wait for every transaction that could have the table open with the old definition to finish, so that from now on every new write knows the index exists.
2. **First scan.** Take a snapshot, scan the whole table and build the index from the rows visible in it. Meanwhile, ongoing writes start adding their own entries to the index (it is now marked **ready** for inserts, but still invalid for queries).
3. **Wait again**, then do a **second scan** to add any rows that were inserted or updated between the first snapshot and the moment writes began maintaining the index.
4. **Wait for older snapshots** that might not see the new entries to finish, then mark the index **valid**. From now on the planner can use it.

~~~dg-state The states an index passes through during CREATE INDEX CONCURRENTLY, including the failure path.
[*] -> Registered : catalog entry, invalid, not ready
Registered -> Ready : writers now maintain it
Ready -> Built : first scan done
Built -> Caught up : second scan done
Caught up -> *Valid : old snapshots finished
Ready -> Invalid leftover : error or cancel
Built -> Invalid leftover : duplicate key or deadlock
Invalid leftover -> [*] : DROP INDEX CONCURRENTLY
~~~

### The costs

- **More work and more time.** Two full table scans plus the waits. Expect it to take two to three times as long as a plain build, while adding CPU and I/O load to a live system.
- **It waits for existing transactions**, several times. A single long-running transaction, or an idle-in-transaction session, stalls the build until it ends.
- **It cannot run inside a transaction block.** Migration tools that wrap every migration in `BEGIN ... COMMIT` must be told to skip that for this step.
- **One at a time.** Only one concurrent index build can run on a table at once.

~~~dg-compare A plain index build versus a concurrent one on a busy 400-million-row table.
[Plain CREATE INDEX]
Lock table for writes -> Scan and sort once -> Index valid
+ Fastest build, about 40 minutes
+ Atomic: works or leaves nothing
- Writes blocked the whole time
- Outage on a busy table
[*CREATE INDEX CONCURRENTLY]
Register -> Scan 1 -> Scan 2 -> Wait -> Valid
+ Writes keep flowing
- About 2 to 3 times slower
- Extra load on a live system
- Can fail, leaving an INVALID index
~~~

## When it fails

A concurrent build can fail partway: a deadlock, a cancelled statement, a timeout, or duplicate values when building a **unique** index. Unlike a plain build, it does not clean up after itself. It leaves an **INVALID** index behind. Queries ignore it, but every write still pays to maintain it, so it is pure cost. You can find such leftovers with `SELECT indexrelid::regclass FROM pg_index WHERE NOT indisvalid`. Fix them with `DROP INDEX CONCURRENTLY` and a retry, or with `REINDEX INDEX CONCURRENTLY`.

Unique indexes have an extra twist. Uniqueness is enforced from the moment the second scan starts, so other transactions can get unique-violation errors **before** the index is usable, and even if the build then fails. If the existing data contains duplicates, the build will fail after doing most of its work, so check for duplicates with a `GROUP BY ... HAVING count(*) > 1` query first.

Partitioned tables add one more wrinkle: `CREATE INDEX CONCURRENTLY` is not supported directly on the parent. You build the index concurrently on each partition, create the parent index with `ON ONLY` (a metadata-only step), and attach each partition's index to it.

## Online schema changes in general

Every schema change on a large, busy table needs a plan. The first question is always: **which lock does this statement take, and for how long?**

~~~dg-matrix Common Postgres schema changes, the lock they take and whether they rewrite or scan the table.
Change | Blocks reads | Blocks writes | Rewrites or scans table
Add nullable column, no default | yes, briefly | yes, briefly | no
Add column with constant default (11+) | yes, briefly | yes, briefly | no
Change column type | yes | yes | yes, rewrite
Add NOT NULL | yes | yes | yes, full scan
Add foreign key NOT VALID, then VALIDATE | no | partial, briefly | yes, scan in validate step
CREATE INDEX | no | yes | yes, build
CREATE INDEX CONCURRENTLY | no | no | yes, two scans
~~~

Some patterns that turn dangerous changes into safe ones:

- **Constraints in two steps.** Add a foreign key or check constraint with `NOT VALID` (instant, applies only to new rows), then run `VALIDATE CONSTRAINT`, which scans the table under a weak lock that allows writes.
- **NOT NULL via a check constraint.** Add `CHECK (col IS NOT NULL) NOT VALID`, validate it, and then `SET NOT NULL` can use the validated constraint instead of scanning.
- **MySQL online DDL.** InnoDB supports many changes with `ALGORITHM=INPLACE, LOCK=NONE` or even `ALGORITHM=INSTANT`. For the rest, tools such as **gh-ost** and **pt-online-schema-change** create a new copy of the table with the new schema, copy rows across in small batches, keep the copy in sync with ongoing changes (gh-ost by reading the binary log, pt-osc with triggers), and finally swap the two tables in an instant rename.
- **Always** set `lock_timeout`, run large changes in low-traffic hours, watch replication lag (index builds generate a lot of WAL) and have a way to cancel.

<div class="ex">
<div class="lab">Worked example</div>

**adding an index to orders**

The orders table has 400 million rows and takes 3,000 writes per second. A plain `CREATE INDEX` would block writes for roughly 40 minutes, queueing about 7 million writes. That is not an option.

The team runs `CREATE INDEX CONCURRENTLY orders_customer_idx ON orders (customer_id);` outside a transaction, with `statement_timeout` disabled for that session and `lock_timeout` at 5 seconds so it cannot block traffic while waiting.

It takes 90 minutes. Writes flow normally the whole time; disk I/O rises by about 30% and replicas fall a few seconds behind during the scans.

Afterwards they check `pg_index.indisvalid` for the new index. It is true, so they run `EXPLAIN` on the target query and confirm it now uses an index scan. Had the build failed, they would `DROP INDEX CONCURRENTLY orders_customer_idx` and retry in a quieter hour.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the migration that took checkout down for 7 minutes**

At 14:00 an analytics job starts a 9-minute query on `payments`. At 14:02 a deploy runs `ALTER TABLE payments ADD COLUMN risk_score int`. That needs an `ACCESS EXCLUSIVE` lock and waits for the analytics query.

From 14:02 every checkout request reads `payments`, queues behind the `ALTER`, and holds a pooled connection while it waits. Within 3 seconds all 200 pooled connections are waiting. Checkout returns errors until 14:09, when the analytics query ends, the `ALTER` runs in 3 ms, and the queue drains.

With `SET lock_timeout = '3s'` in the migration, the `ALTER` would have failed at 14:02:03, freeing the queue at once. The deploy would have retried successfully at 14:09, and no customer would have noticed.

</div>

## Common mistakes

- **Running a plain CREATE INDEX on a live, busy table.**
- **No lock_timeout on migrations**, letting a quick change queue behind a long query.
- **Wrapping CONCURRENTLY in a transaction**, which makes it fail immediately.
- **Not checking for INVALID indexes** after a failed build; they slow every write.
- **Building a unique index on dirty data** without checking for duplicates first.
- **Ignoring replicas.** The index build's WAL must be replayed, and lagging replicas serve stale reads.

## In an interview

When your design adds an index or changes a schema on an existing large table, say how you would roll it out: concurrent or online variant, a lock timeout, low-traffic window, watching replication lag, and checking the result. It signals that you have operated real databases, not just designed them.

<div class="take">
<div class="lab">Key takeaway</div>

a plain CREATE INDEX blocks writes for the whole build, and any strong lock request can queue all traffic behind a long transaction. Use CREATE INDEX CONCURRENTLY and online DDL tools on live tables, always set a lock timeout, accept slower builds, and check for and clean up invalid indexes when a build fails.

</div>
