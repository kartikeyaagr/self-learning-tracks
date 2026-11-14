---
id: "03.12"
course: "system-design-101"
module: "03"
slug: "schema-evolution"
title: "Schema evolution"
keyIdeas: ["Schema changes on a live system must keep old and new code working at the same time, because deploys aren't instant.", "Expand/contract pattern: add the new thing, write to both, backfill, switch reads, stop writing the old, remove the old.", "Renaming a column safely: add new column, dual-write, backfill old rows, switch reads, stop writing old column, drop it later.", "Never do a destructive change in the same deploy as the code that stops using the old schema.", "For serialized data (events, Protobuf), keep changes backward and forward compatible."]
question: "List the steps to rename a column with zero downtime."
answer: "1) Add the new column as nullable with no default (a fast change, with a lock timeout). 2) Deploy code that writes both the old and new columns on every insert and update but still reads the old one (or add a trigger that copies writes). 3) Backfill existing rows into the new column in small, throttled, resumable batches. 4) Deploy code that reads the new column while still writing both, and verify by comparing the two. 5) Deploy code that stops writing the old column. 6) After a waiting period with no remaining readers, drop the old column. Each step works with the code versions running before and after it, and every step except the final drop can be rolled back."
sources: [{"title": "Stripe: Online migrations at scale", "url": "https://stripe.com/blog/online-migrations", "note": "Dual writes, backfill, switch reads, clean up."}, {"title": "Kleppmann: Schema evolution in Avro, Protobuf and Thrift", "url": "https://martin.kleppmann.com/2012/12/05/schema-evolution-in-avro-protocol-buffers-thrift.html", "note": "Compatibility rules."}]
---

## Code and schema cannot change at the same instant

On your laptop, changing a schema is easy: edit the table, edit the code, restart. In production it is not, because **there is no single instant at which everything changes**.

- A deploy rolls out over minutes. For that whole time, old and new versions of the application run side by side against the same database.
- If something goes wrong, you roll back the code, but not the data.
- Background workers, cron jobs, mobile apps that users never update, and other teams' services all read the same tables or messages on their own schedules.
- Queues, caches and replicas may still hold data written in the old shape.

So the rule is: **every individual change must be compatible with every version of code that might still be running.** A breaking change (rename, type change, split a column, move data to a new table) is never done in one step. It is broken into a sequence of compatible steps. This approach is usually called **expand and contract** (or parallel change): first expand the schema so it supports both the old and new shape, migrate code and data across, then contract by removing the old shape.

## Renaming a column with zero downtime

A rename is the classic case. `ALTER TABLE users RENAME COLUMN fullname TO display_name` is instant inside the database. But at that instant, every running app server still sends `SELECT fullname ...` and starts failing. Deploying the code first fails the other way: the new code asks for `display_name`, which does not exist yet. There is no ordering of "rename" and "deploy" that works.

Instead:

1. **Add the new column.** `ALTER TABLE users ADD COLUMN display_name text;` Nullable and with no default, it is a fast catalog-only change (with a `lock_timeout`, lesson 03.11). Old code ignores it.
2. **Deploy code that writes both columns** on every insert and update, and still **reads the old one**. A database trigger that copies `fullname` into `display_name` on write is an alternative that also covers writers you do not control.
3. **Backfill.** Copy existing values: `UPDATE users SET display_name = fullname WHERE id BETWEEN ? AND ? AND display_name IS NULL`, in batches of a few thousand rows, with a pause between batches so replicas keep up and locks are short.
4. **Deploy code that reads the new column** (still writing both). Verify: compare the two columns, watch error rates. If anything is wrong, roll back to step 2's code; the data is still intact in both places.
5. **Deploy code that stops writing the old column.** Nothing reads it any more.
6. **Drop the old column**, after a waiting period long enough that you are sure no straggling process, report or other team uses it.

~~~dg-timeline Renaming fullname to display_name: schema changes and code deploys alternate, so every running version always works.
[Schema] Day 1 | Add display_name, nullable
[Code] Day 2 | v2 writes both, reads fullname
[Schema] Day 2 | Backfill display_name in batches
[Code] Day 4 | *v3 reads display_name, writes both
[Code] Day 6 | v4 writes display_name only
[Schema] Day 20 | *Drop fullname
~~~

Notice that each step is safe for the code running before **and** after it. That is the property to check for any migration plan.

~~~dg-matrix At each step of the rename, which code versions work correctly against the schema and data. Every step overlaps with the one before.
Step | v1 old code | v2 writes both | v3 reads new | v4 new only
After add column | yes | yes | no, data missing | no
After backfill | yes | yes | yes | no, v1 still reads old
v3 deployed | partial, its writes miss new | yes | yes | yes
After drop old | no | no | no | yes
~~~

## Stripe's four-phase pattern

The same idea scales to much bigger changes. Stripe moved hundreds of millions of subscription records out of a field embedded in each customer record and into their own table, with no downtime. The scale makes the point: at one second per object, a naive serial migration of 100 million records would take over three years.

1. **Dual write.** Change the code to write every subscription change to both the old and new places. Then **backfill** existing records. To find which records needed copying without loading production databases, Stripe ran jobs on Hadoop over database snapshots, then copied records with a fleet of parallel workers.
2. **Change reads.** Switch reads to the new table. To check it was right, Stripe used GitHub's Scientist library: each read runs against both the old and new path, returns the old result to the user, and records any mismatch. Users are unaffected; engineers get a list of differences to fix.
3. **Change writes.** Make the new table the source of truth, writing to it first. This was the hardest phase: subscription logic spanned thousands of lines and several services. They made the old access paths raise errors in testing so stragglers surfaced.
4. **Remove old data and code.**

Each step was a small change (a few hundred lines) that could be rolled back on its own.

~~~dg-seq Comparing old and new read paths before switching: the user always gets the old result, and mismatches are logged.
App > Old path : read subscription
App > New table : read subscription
Old path -> App : result A
New table -> App : result B
App > Mismatch log : record if A differs from B
App -> App : return result A to user
~~~

## Making backfills safe

Backfills are where many migrations hurt production. A single `UPDATE users SET display_name = fullname` on 200 million rows would hold row locks on every row until it commits, generate hundreds of gigabytes of WAL in one go, bloat the table with 200 million dead tuples (lesson 03.07) and push replicas minutes behind. Make backfills:

- **Batched**: a few thousand rows per transaction, walking the primary key (`WHERE id > last_id ORDER BY id LIMIT 5000`), not `OFFSET` (lesson 04.12).
- **Throttled**: sleep between batches, and slow down automatically if replication lag or database CPU rises.
- **Resumable**: record the last processed key so a crash or deploy does not restart from zero.
- **Idempotent**: running a batch twice gives the same result (`WHERE display_name IS NULL`).

## Evolving serialised data

The same thinking applies to data outside the database: messages in Kafka, values in caches, files in object storage, and requests between services. A consumer running last week's code may read a message written by this week's producer, and the other way round. Two kinds of compatibility matter:

- **Backward compatible**: new code can read data written by old code.
- **Forward compatible**: old code can read data written by new code (and ignore what it does not understand).

Binary encoding formats are built around these rules. Consider one small record encoded three ways: about 82 bytes as JSON, 33 bytes in Protocol Buffers, and 32 bytes in Avro.

- **Protocol Buffers and Thrift** store a numeric **tag** with each field, not its name. A reader that meets an unknown tag skips it, which gives forward compatibility. You can add fields with new tag numbers, remove optional fields, and rename fields freely, because names never appear in the data. You must never **reuse** a removed field's tag for something else, and never change a tag's type in incompatible ways. Marking a field `required` makes evolution risky, because removing it later breaks old readers; newer protobuf versions dropped `required` for this reason.
- **Avro** stores no tags at all, which makes it the most compact. The reader resolves each record using **both** the writer's schema and its own, matching fields by name. New fields need default values so old data can be read; renames go through aliases. The catch is that the reader must know the exact writer's schema, so it travels in a file header or is fetched from a **schema registry** by an ID stored with each message.

~~~dg-compare How Protocol Buffers and Avro make schema evolution safe.
[Protocol Buffers]
Each field: tag number + value
+ Unknown tags are skipped
+ Rename fields freely
- Never reuse a removed tag
- About 33 bytes for the sample
[Avro]
No tags, fields in schema order
+ Smallest, about 32 bytes
+ Reader matches fields by name
- Needs the writer's schema
- New fields need defaults
~~~

These are the same rules as in lessons 01.11 and 02.08 for APIs and events: additive, optional changes are safe; changing the meaning of a field or reusing an identifier is not.

## Practical guardrails

- **Separate schema migrations from code deploys**, and ship them in the safe order: expand first, contract last.
- **Make every step reversible** until the final cleanup. Dropping the old column is the only one-way door, so do it last and late.
- **Compare before you switch**: dual reads, row-count checks, checksums on samples.
- **Test against production-sized data**, because a migration that takes 2 seconds on staging can take 2 hours on production.
- **Lint migrations** in code review for dangerous operations: type changes, `NOT NULL` without a check constraint, non-concurrent index builds, missing lock timeouts.

<div class="ex">
<div class="lab">Worked example</div>

**splitting name into first_name and last_name**

The `customers` table has 80 million rows with a `name` column. The product needs `first_name` and `last_name`.

Day 1: add both columns, nullable. Deploy code that writes all three columns on every change and still reads `name`.

Days 2 to 3: backfill by splitting existing names in batches of 5,000 rows, with a 50 ms pause. That is 16,000 batches; at about 150 ms each including the pause, about 40 minutes. Names that do not split cleanly ("Madonna", "Mary Ann de la Cruz") are logged for review rather than guessed.

Days 4 to 10: deploy code that reads the new columns but compares them with `name` on each read, logging mismatches. Fix the 0.3% of oddly split names.

Day 11: switch all reads to the new columns and stop writing `name`. Day 25: drop `name`. At no point was any running version of the code broken.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**what goes wrong if you skip a step**

A team renames `price` to `price_cents` in one migration that runs just before the deploy. The rename takes 3 ms. The deploy takes 6 minutes to roll across 40 servers.

For those 6 minutes, the servers still on old code send queries naming `price` and get "column does not exist". At 2,000 requests per second and an average of 20 old servers, roughly 360,000 requests fail. Rolling back the code does not help, because the column is now `price_cents`; only another emergency migration does. The six-step version would have produced zero errors.

</div>

## Common mistakes

- **Renaming or changing a type in one step.** There is no deploy order that makes it safe.
- **One giant backfill transaction.** Locks, WAL floods, bloat and replica lag.
- **Dropping the old column too soon**, before every reader has moved.
- **Reusing a protobuf tag number** for a different field; old data will be misread silently.
- **Adding a required field** to a message format; old producers cannot supply it.

## In an interview

If asked to change a schema in a running system, say "expand and contract" and walk through the steps: add, dual write, backfill in batches, switch reads with verification, stop old writes, drop. Then mention the risk in each step (lock time, backfill load, stragglers) and how you limit it.

<div class="take">
<div class="lab">Key takeaway</div>

never make a breaking change in one step. Expand (add the new shape and dual write), migrate (backfill in batches and switch reads with verification), then contract (stop old writes and drop the old shape), keeping every step compatible with the code before and after it.

</div>
