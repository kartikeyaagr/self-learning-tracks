---
id: "14.05"
course: "system-design-101"
module: "14"
slug: "database-migration-safety"
title: "Database migration safety"
keyIdeas: ["Database migrations must not lock tables for long or break running code.", "Expand/contract: add new structures (nullable column, new table), deploy code that writes both, backfill in small batches, switch reads, then remove the old.", "Adding a NOT NULL column with a default can rewrite the table on some databases. Safer: add it nullable, backfill, then add the constraint (validate separately where supported).", "Use online schema change tools (gh-ost, pt-online-schema-change) for big MySQL tables.", "Set lock timeouts so a migration fails fast instead of blocking traffic."]
question: "Why add a column as nullable first, backfill, then add NOT NULL?"
answer: "Adding a NOT NULL column directly can force the database to rewrite or scan the whole table under a heavy lock to fill or check the value, blocking traffic on a large table. It also breaks any old code still running during the deploy that inserts rows without the new column. Adding it as nullable is an instant metadata change that every version of the code tolerates. Code then starts writing the column, a batched backfill fills old rows without long locks or replication lag, and only when every row has a value do you add the constraint (in PostgreSQL as a NOT VALID check, then validated separately without blocking writes). Every step is small, safe while old and new code coexist, and reversible."
sources: [{"title": "Stripe: Online migrations at scale", "url": "https://stripe.com/blog/online-migrations", "note": "The 4-step pattern."}, {"title": "GitHub: gh-ost", "url": "https://github.com/github/gh-ost", "note": "Online schema changes for MySQL."}]
---

## The constraint: no downtime, no wrong data

Code is easy to roll back; data is not. A deploy that goes wrong can be reverted in minutes, but a migration that rewrote 200 million rows, dropped a column or locked a table for an hour has already done its damage. Database migrations are the most common one-way doors (lesson 14.04) in software, so they need their own discipline.

Two things make them hard. First, **the application keeps running**: there is no maintenance window in which nothing reads or writes. Old and new versions of the code run side by side during every deploy, so at every moment both must work against whatever schema exists. Second, **scale**: operations that are instant on a 10,000-row table take hours on a 2-billion-row table, and some of them hold locks that block every query while they run.

Stripe is a good case study because it has both constraints at their extreme. It has no maintenance windows, every service must keep getting correct data during a migration, and the data runs to hundreds of millions of objects. At one second per object, migrating 100 million objects one by one would take over three years, so even the backfill has to be designed.

## Expand and contract

The case: each Customer record used to hold its subscriptions **inline** (first a single field, later an array). Every subscription change rewrote the whole customer, and finding subscriptions meant scanning customers. The goal was a separate `subscriptions` table.

The pattern is called **expand and contract** (also parallel change or dual writing). First you expand: add the new structure alongside the old, so both exist. Then you migrate traffic onto it step by step. Finally you contract: remove the old structure. In four steps:

1. **Dual write.** Write every change to both the old place and the new table. Then **backfill** the existing data into the new table.
2. **Switch reads.** Move every read to the new table, after verifying it returns the same answers.
3. **Switch writes.** Make the new table the only place written to.
4. **Remove the old data.** Delete the old field and its contents.

~~~dg-timeline Expand and contract over several weeks: at every point one store is complete and correct, and each step can be undone.
Week 1 | Create subscriptions table, empty
Week 1 | Dual writes ramped 1% to 100%, watching DB load
Week 2 | Backfill copies historic subscriptions; verify job confirms none missing
Week 3 | *Shadow reads compare old and new; reads switch to new table
Week 5 | Writes moved to new table one code path at a time
Week 7 | Old accessor raises errors; stragglers fixed
Week 8 | Old arrays emptied lazily, final sweep deletes the rest
~~~

Each step can be rolled back on its own, and each runs for a while before the next. That is the whole point: you never flip everything at once, and you never reach a state where neither store is trustworthy.

**Step 1 in detail.** Dual writes were ramped up gradually while watching database metrics, because writing twice adds load. Existing subscriptions were copied lazily whenever they were updated. The rest needed a **backfill**. Instead of scanning the production database (which would compete with live traffic), Stripe loaded a database snapshot into Hadoop. A MapReduce job (written in Scalding) listed the subscription IDs to copy, and a fleet of workers copied them in parallel. Afterwards the job ran again to confirm nothing was missing.

**Step 2** used GitHub's Scientist library (lesson 14.06) to read from both stores and compare results in production. Mismatches raised alerts. Reads moved to the new table only once results matched.

**Step 3** was the hardest. Subscription logic (updates, proration, renewals) spanned thousands of lines across services. Stripe changed one small code path at a time, never more than a few hundred lines at once, with more Scientist experiments along the way. To find forgotten callers, they made the old accessor **raise an error** if anything still used it.

**Step 4.** Old arrays were emptied lazily as customers were loaded. A final job swept up whatever was left.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a backfill**

You must copy 400 million rows. A single worker doing one row per round trip at 2 ms manages 500 rows per second: 400,000,000 ÷ 500 = 800,000 s, about 9 days.

Batch 1,000 rows per statement at about 200 ms per batch: 5,000 rows per second per worker. Twenty workers give 100,000 rows per second: 4,000 s, just over an hour of pure work.

But the primary also serves live traffic and replicas must keep up. If replication lag climbs above 5 s, pause. In practice you might throttle to 20,000 rows per second, so plan for about 6 hours, run it in the quiet period, and make it resumable (track the last ID copied) so a crash does not restart from zero.

</div>

## Schema changes without locking the table

The same idea applies within one table. Some `ALTER TABLE` operations rewrite the whole table or take a lock that blocks reads and writes, which on a big table blocks traffic for minutes or hours. Which operations do this depends on the database and its version, so always check before running one in production.

There is a subtle extra hazard in PostgreSQL. Even a "fast" `ALTER TABLE` needs a brief exclusive lock. If a long-running query holds a conflicting lock, the `ALTER` waits in the queue, and **every new query on that table queues behind the ALTER**. A 1 ms change can cause a 2-minute outage. Always set `lock_timeout` (say 2 s) so the migration gives up and retries instead of blocking everyone.

### The safe sequence for a new required column

1. **Add the column as nullable**, with no default that forces a rewrite. On modern databases this changes only metadata and is instant. (PostgreSQL 11 and later can also add a column with a constant default instantly, but a volatile default such as `now()` still rewrites.)
2. **Deploy code that writes it** for all new and updated rows. Old code that does not know the column still works, because the column accepts NULL.
3. **Backfill** old rows in small batches (say 1,000 rows each, with short pauses), so you avoid huge transactions, long lock waits and replication lag.
4. **Add the NOT NULL constraint** once every row has a value. In PostgreSQL, add `CHECK (col IS NOT NULL) NOT VALID` first (instant, enforced for new writes), then `VALIDATE CONSTRAINT` separately; validation scans the table without blocking writes. From PostgreSQL 12, a validated check constraint lets `SET NOT NULL` skip its own full scan.
5. Only then deploy code that **relies** on the column always being present.

~~~dg-state A new required column moves through states; at each one, both the old and the new code still work.
[*] -> Absent
Absent -> Nullable : ADD COLUMN, metadata only
Nullable -> Written : deploy code that fills it
Written -> Backfilled : batched UPDATE of old rows
Backfilled -> Checked : CHECK NOT VALID, then VALIDATE
Checked -> *Required : SET NOT NULL, code relies on it
~~~

If you add NOT NULL first, one of two bad things happens: the database rewrites or scans the table under a heavy lock to fill or check the value, or old code still running during the deploy fails every insert because it does not supply the new column.

### Other common changes, done safely

| Change | Unsafe way | Safe way |
| --- | --- | --- |
| Rename a column | `RENAME COLUMN` while old code runs | add new column, dual write, backfill, move reads, drop old |
| Drop a column | drop, then deploy code | deploy code that ignores it, wait, then drop |
| Add an index | `CREATE INDEX` (blocks writes) | `CREATE INDEX CONCURRENTLY` in PostgreSQL, online DDL in MySQL |
| Change a column type | `ALTER TYPE` with a table rewrite | new column, dual write, backfill, switch |
| Add a foreign key | add and validate in one step | add `NOT VALID`, then `VALIDATE` separately |

Notice the order for dropping: **code first, schema second**. For adding: **schema first, code second**. In both cases the schema must always be compatible with every version of code that might be running.

## gh-ost: online schema change for MySQL

For changes MySQL cannot do in place, GitHub's **gh-ost** copies the table instead of altering it:

1. Create an empty **ghost table** with the new schema.
2. **Copy existing rows** into it in small batches, in primary key order.
3. At the same time, **read the MySQL binary log** (the replication stream of every row change) and apply every insert, update and delete on the original table to the ghost table.
4. When the ghost table has caught up, **cut over**: atomically rename the tables so the ghost becomes the real table and the original is kept under another name.

~~~dg-seq gh-ost: copy rows, replay changes from the binary log, then swap the tables atomically.
gh-ost > MySQL : CREATE TABLE _orders_gho (new schema)
gh-ost > MySQL : copy rows 1 to 1,000 into ghost
App > MySQL : UPDATE orders row 57
MySQL -> gh-ost : binlog event: row 57 updated
gh-ost > MySQL : apply same update to ghost
gh-ost > MySQL : copy rows 1,001 to 2,000 (repeat)
gh-ost > gh-ost : ghost caught up, wait for quiet period
gh-ost > MySQL : RENAME orders to _orders_del, _orders_gho to orders
~~~

Older tools (such as pt-online-schema-change) did step 3 with **triggers** on the original table. gh-ost avoids triggers, because they add work to every write inside the application's own transaction, compete for locks, and cannot be paused: once the trigger exists, every write pays for it.

~~~dg-compare Trigger-based online schema change versus binlog-based: same idea, different way of capturing ongoing changes.
[Triggers]
app write -(same transaction)-> trigger -> ghost table
- extra work inside every app transaction
- cannot pause without dropping triggers
- lock contention on busy tables
[*Binlog (gh-ost)]
app write -> binlog -(async)-> gh-ost -> ghost table
+ app transactions unchanged
+ true pause when replicas lag
+ cut-over can wait for a quiet time
~~~

Reading the binlog asynchronously gives gh-ost control:

- **True pause.** When throttled (for example because replicas are lagging), it stops all its own writes, both row copying and event applying, so the primary returns to normal load.
- **Interactive control.** You can change settings such as batch size while it runs, through a socket.
- **Postponed cut-over.** You can delay the swap until a quiet period or until a human approves.
- **Test on a replica.** It can run the whole migration on a replica and leave both tables for comparison. GitHub runs this continuously against production replicas.

<div class="ex">
<div class="lab">Worked example</div>

**the lock queue outage**

At 10:00 an analyst starts a 5-minute report query on `orders`. At 10:01 a migration runs `ALTER TABLE orders ADD COLUMN note text`, which needs a short exclusive lock but must wait for the report.

From 10:01, every new `SELECT` and `INSERT` on `orders` waits behind the ALTER. The connection pool fills in seconds, and checkout returns errors for 4 minutes, until the report finishes.

With `SET lock_timeout = '2s'`, the ALTER gives up at 10:01:02, nothing queues behind it, and the migration tool retries a minute later. Same change, no outage.

</div>

## Common mistakes

- **Changing schema and code in one deploy** so that old code running during the rollout breaks against the new schema.
- **One giant UPDATE** to backfill: a single transaction that holds locks for an hour and floods replication.
- **No lock timeout**, so a quick DDL queues behind a slow query and blocks the table.
- **Dropping the old column too early**, before every reader has moved and before you are sure you will not roll back.
- **No verification** after the backfill: assuming the copy worked instead of counting and comparing.

## In an interview

If your design changes a data model, say how you would migrate it live: expand, backfill in throttled batches, verify with shadow reads, switch reads, switch writes, contract. Mention that schema changes go first and code relying on them second, and that you would set lock timeouts and use online tools for large tables.

<div class="take">
<div class="lab">Key takeaway</div>

change data at scale with expand and contract: dual write, backfill, move reads (verified), move writes, then delete the old data, so each step can be undone. For schema changes, add columns as nullable, backfill in throttled batches, add constraints last, set lock timeouts, and use tools like gh-ost that copy into a shadow table and swap instead of locking the live one.

</div>
