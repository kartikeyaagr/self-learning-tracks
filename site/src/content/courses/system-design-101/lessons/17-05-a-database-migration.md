---
id: "17.05"
course: "system-design-101"
module: "17"
slug: "a-database-migration"
title: "A database migration"
keyIdeas: ["Zero-downtime migration of a big table or database:", "1) Set up the new store. 2) **Dual-write**: app writes to old and new. 3) **Backfill**: copy existing data in batches, without overwriting newer dual-written rows. 4) **Verify**: compare data and shadow-read from the new store. 5) **Switch reads** gradually. 6) Stop writing to the old store, then decommission it.", "Every step must be reversible until the final one.", "Alternatives to app dual-writes: CDC replication from old to new, or tools like gh-ost for same-database schema changes.", "Stripe's online migrations post describes this four-phase pattern."]
question: "Move a 2 TB table to a new database with zero downtime. List the phases."
answer: "1. Create the new table and put read and write targets behind feature flags. 2. Start capturing all changes to the new database, by dual writes or by change data capture from the old database's log. 3. Backfill existing rows in throttled, resumable batches by key range, using guarded upserts so a stale copy never overwrites newer data that arrived through step 2. 4. Verify with counts and checksums per key range and re-copy any range that differs. 5. Shadow-read both databases on live traffic and compare until mismatches are zero or explained. 6. Switch reads to the new database, keeping writes on both. 7. Switch writes, with a brief pause until CDC lag is zero or an atomic cut-over, optionally keeping reverse replication for rollback. 8. Keep the old table as a fallback for days or weeks, then remove it and the migration code."
sources: [{"title": "Stripe: Online migrations at scale", "url": "https://stripe.com/blog/online-migrations", "note": "Dual writing pattern."}, {"title": "GitHub: gh-ost", "url": "https://github.com/github/gh-ost", "note": "Triggerless online migration."}]
tags: ["Design"]
---

## The goal

Move a 2 TB table (say `orders`) from the old database to a new one while the application keeps running: no maintenance window, no lost writes, no wrong data. The new home might be a new cluster with bigger machines, a different engine (MySQL to PostgreSQL, or to a distributed SQL database), or a sharded layout. The techniques are the same in each case.

This lesson combines three ideas covered earlier: the four-step dual-write migration (lesson 14.05), shadow reads that compare old and new results on live traffic (lesson 14.06), and, for schema changes inside one MySQL server, gh-ost's copy, replay and swap approach. The question asks for the phases, so the lesson is organised around them, with the failure modes of each.

## Why not just copy it?

The naive plan is "stop the app, dump the table, load it into the new database, point the app at it, start the app". It fails on time alone.

<div class="ex">
<div class="lab">Worked example</div>

**how long does the copy take?**

2 TB at 100 MB/s is 2,000,000 MB / 100 = 20,000 seconds, roughly 5.5 hours. Throttled to 50 MB/s to protect production, about 11 hours. Building indexes on the target can add hours more.

A 6 to 12 hour outage for an orders table is not acceptable for most businesses. And while a live copy runs for those hours, the table keeps changing: at 2,000 writes per second, 11 hours is about 80 million changed rows that a one-time copy would miss.

So the plan must copy in the background and also capture every change made during and after the copy.

</div>

## The core idea: change capture plus backfill

Any zero-downtime migration has the same skeleton:

1. Make sure every **new change** reaches the new database from some moment T.
2. Copy the **existing rows** (the backfill) as they were around T, without overwriting newer changes.
3. Prove the two are the same.
4. Move reads, then writes, one reversible step at a time.

~~~dg-timeline The phases of a zero-downtime table migration. Every phase before the last can be rolled back.
Day 0 | Prepare: new table, flags
Day 1 | Start change capture
Day 1 to 2 | Backfill 2 TB in batches
Day 3 | Verify with checksums
Day 3 to 10 | Shadow reads, fix mismatches
Day 10 | Switch reads to new DB
Day 12 | *Switch writes (cut-over)
Day 12 to 26 | Bake, old table as fallback
Day 30 | Remove old table and code
~~~

## The phases in detail

### 0. Prepare

Create the table in the new database, with the schema and indexes it needs. Make sure the application can connect to both. Put each switch (where reads go, where writes go, whether to shadow-read) behind **feature flags**, so each step is a configuration change that takes seconds and can be reversed in seconds, not a deploy.

### 1. Start capturing changes

From this moment, every insert, update and delete on `orders` must also reach the new database. Two ways:

- **Dual writes in the application**: every write goes to both databases. Simple to understand, but if the second write fails you must retry or record it somewhere, it adds latency, and the two writes are not atomic, so the databases can diverge after a crash between them.
- **Change data capture (CDC)**: read the old database's replication log (MySQL binlog, PostgreSQL logical replication) with a tool such as Debezium and apply each change to the new database. It needs no application changes and catches writes from every source: batch jobs, admin tools, other services, manual fixes. gh-ost uses this approach inside MySQL.

Stripe ramped its dual writes gradually while watching database load, rather than switching them on for all traffic at once.

~~~dg-compare Two ways to capture changes during a migration.
[Dual writes]
App -(write)-> Old DB + New DB
+ No extra infrastructure
- Misses writes from other tools
- Not atomic across two databases
[*Change data capture]
Old DB -(binlog)-> CDC -> New DB
+ Catches every writer
+ No app code changes
- Replication lag to monitor
~~~

### 2. Backfill

Copy the existing rows in batches by primary key range, for example 10,000 rows at a time: `SELECT ... WHERE id > :last AND id <= :last + 10000`. Three properties are essential:

- **Throttled**: pause when the source's replica lag or CPU rises, so production does not suffer. gh-ost throttles itself on replica lag; your backfill should too.
- **Resumable**: store the last completed key range as a checkpoint, so a crash at hour 7 does not restart from zero.
- **Never overwrite newer data**: a row copied by the backfill may have been changed a second ago through step 1. Use upserts that skip rows that are already newer in the target (compare an `updated_at` or version column), or start CDC from a log position recorded just before the copy, so replay applies later changes on top.

Stripe avoided scanning production at all for one migration: it found the rows to migrate from a database **snapshot** loaded into Hadoop, then wrote them through the normal application path.

<div class="ex">
<div class="lab">Worked example</div>

**the backfill race**

At 10:00:00.000 the backfill reads order 501 with status `paid` (version 7).

At 10:00:00.020 the customer cancels. The old database updates order 501 to `cancelled` (version 8), and CDC writes version 8 to the new database at 10:00:00.150.

At 10:00:00.300 the backfill's batch insert reaches order 501 with its stale copy, version 7. A plain `INSERT ... ON DUPLICATE KEY UPDATE` would overwrite `cancelled` with `paid`: a cancelled order would ship.

With a guarded upsert (`... UPDATE status = IF(VALUES(version) > version, VALUES(status), status)`), the stale version 7 is ignored and the new database keeps version 8. This single rule is the difference between a safe and an unsafe backfill.

</div>

### 3. Verify

Compare **row counts and checksums** per primary key range between the two databases (for example a hash of all rows with IDs 1 to 100,000), and fix differences by re-copying those ranges. Repeat until clean. Stripe re-ran its migration job afterwards to confirm nothing was missing. gh-ost's test-on-replica mode leaves both tables in place so you can compare them before any cut-over.

### 4. Shadow reads

Checksums prove the data matches; shadow reads prove the **application** gets the same answers. With a library like GitHub's Scientist (lesson 14.06), each read runs against both databases, the old result is returned to the user, and any mismatch is logged with both results. Start at 1% of traffic and ramp up. Mismatches reveal subtle issues checksums miss: different collation (sorting of accented characters), different time zone handling, different default ordering when a query has no `ORDER BY`. Continue until mismatches are zero or every one is explained.

### 5. Switch reads

Flip the read flag. The new database now serves reads, while writes still go to both (or to the old one plus CDC). If anything looks wrong, flip it back: the old database is still complete and current.

### 6. Switch writes: the cut-over

Make the new database the one that is written to. This is the riskiest moment, because after it the old database starts to fall behind. Options:

- **With dual writes**: make the new database the primary write (written first, its failure fails the request), then stop writing the old one.
- **With CDC**: briefly pause writes (a few seconds, using the flag), wait for CDC lag to reach zero, flip the write target, and resume. gh-ost does the equivalent with an atomic table rename at **cut-over**, which it can postpone until a quiet moment.
- Optionally run **reverse replication** (new to old) for a while, so a rollback to the old database does not lose the writes made since cut-over.

~~~dg-seq Write cut-over with CDC: a few seconds of paused writes, no lost data.
Ops > App : pause writes to orders (flag)
CDC > NewDB : apply remaining binlog events
CDC -> Ops : lag = 0
Ops > App : write target = NewDB, resume
App > NewDB : new writes
NewDB > OldDB : reverse replication (fallback)
~~~

Stripe found the write switch the hardest step, because writes were spread across many code paths. It changed them a few hundred lines at a time and made the old data accessor **raise an error** when called, to flush out forgotten callers in tests and staging before they could cause silent divergence in production.

### 7. Bake

Keep the old table intact (and, with reverse replication, current) for days or weeks. Some problems only appear at month-end, during a sale, or when a rare report runs.

### 8. Remove

Drop the old table, the dual-write or CDC pipeline, the shadow-read code and the flags. Only this step is irreversible, which is why it comes last and after a long bake.

~~~dg-state Where reads and writes point as the flags change. Each state can return to the previous one until the old table is removed.
[*] -> Old only
Old only -> Capture on : start CDC
Capture on -> Shadow reads : backfill verified
Shadow reads -> Reads on new : no mismatches
Reads on new -> *Writes on new : cut-over
Writes on new -> Old removed : bake complete
Reads on new -> Shadow reads : roll back
~~~

## What to watch for

- **Long transactions and locks**: never `SELECT * FROM orders` in one transaction. Batch by key ranges and keep each batch's transaction short.
- **Replication lag** on the old database caused by the backfill's reads and CDC's load.
- **Auto-increment IDs**: the new database must generate IDs above the old maximum, or switch to application-generated IDs (lesson 04.08) before the migration, so the two never hand out the same ID.
- **Foreign keys and joins**: queries that join `orders` with tables still in the old database must be rewritten or served from one side during the transition.
- **Other writers**: batch jobs, admin tools and other services. CDC catches them; application dual writes do not.
- **Capacity**: the new database must take full production load on day 10, not just the backfill. Load test it with replayed traffic.

## Common mistakes

- Backfilling before starting change capture, so writes during the copy are lost.
- Plain upserts in the backfill that overwrite newer rows.
- Skipping verification because "the copy finished without errors".
- Switching reads and writes in the same step.
- Deleting the old table the day after cut-over.
- Deploys instead of flags, so each rollback takes 30 minutes.

## In an interview

List the phases in order, then pick two to go deep: the backfill race (and the guarded upsert) and the cut-over (pause, drain lag, flip, reverse replication). Mention dual writes versus CDC. Ask: who else writes to this table, how much write downtime (if any) is acceptable at cut-over, and are we changing the schema at the same time? (Prefer not to: one change at a time.)

<div class="take">
<div class="lab">Key takeaway</div>

to move a 2 TB table with no downtime, start capturing changes first (dual writes or CDC), backfill in throttled, resumable batches that never overwrite newer data, verify with checksums, shadow-read with comparison, switch reads, then switch writes with a brief pause or atomic cut-over, keep the old table as a fallback, and only then delete it. Every step before deletion can be rolled back with a flag.

</div>
