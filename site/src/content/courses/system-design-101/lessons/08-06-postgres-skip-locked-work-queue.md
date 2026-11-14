---
id: "08.06"
course: "system-design-101"
module: "08"
slug: "postgres-skip-locked-work-queue"
title: "Postgres SKIP LOCKED work queue"
keyIdeas: ["Postgres can act as a job queue: a `jobs` table with status, and workers claim work with `SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1`.", "`FOR UPDATE` locks the row; `SKIP LOCKED` makes other workers skip rows already locked instead of waiting, so many workers pull in parallel without contention.", "The worker processes the job and marks it done in the same transaction, or sets a lease and finishes later.", "Benefits: no extra infrastructure, and jobs can be enqueued in the same transaction as the business data.", "Limits: polling load and table bloat at very high volumes. Fine up to thousands of jobs/s with care."]
question: "Why does SKIP LOCKED let many workers pull from one table without blocking each other?"
answer: "Each worker claims rows with SELECT ... FOR UPDATE, which writes a row lock owned by its transaction. Plain FOR UPDATE makes a second worker wait for that lock; SKIP LOCKED tells it to skip any row it can't lock immediately and keep scanning, so each worker grabs a different set of free rows at once with no blocking and no duplicates. The locks last until the worker's transaction commits (deleting or completing the rows) or rolls back after a failure or crash (releasing them for another worker), which gives acknowledgement and redelivery from ordinary transaction rules."
sources: [{"title": "PostgreSQL docs: SELECT ... SKIP LOCKED", "url": "https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE", "note": "The locking clause."}, {"title": "Crunchy Data: Message queuing using native PostgreSQL", "url": "https://www.crunchydata.com/blog/message-queuing-using-native-postgresql", "note": "A worked example."}]
tools: ["PostgreSQL"]
---

## A queue in a table

You don't always need a message broker. If you already run Postgres and need a job queue for thousands of jobs a second rather than millions, a plain table can do the job, with one special clause: `FOR UPDATE SKIP LOCKED`. Libraries such as Que and GoodJob (Ruby), Oban (Elixir), graphile-worker and pg-boss (Node), and River (Go) are all built on it.

Start with the table:

```
CREATE TABLE jobs (
  id          bigint generated always as identity primary key,
  queue       text not null default 'default',
  run_at      timestamptz not null default now(),
  priority    int not null default 0,
  payload     jsonb not null,
  attempts    int not null default 0
);
CREATE INDEX jobs_ready ON jobs (queue, priority, run_at);
```

Enqueuing is an `INSERT`. Because it's a normal (logged) table, a committed job survives a crash. And because it's in the same database as your business data, you can enqueue in the **same transaction** as the business write. That one property is often the main reason to choose this design.

## Why the obvious queries fail

### Plain SELECT: two workers take the same job

```
SELECT * FROM jobs ORDER BY id LIMIT 10;
```

Two workers run this at the same moment and both get jobs 1 to 10. Both process them. Every job runs twice.

### FOR UPDATE: workers queue up behind each other

Adding a row lock fixes duplicates:

```
SELECT * FROM jobs ORDER BY id LIMIT 10 FOR UPDATE;
```

`FOR UPDATE` takes a **row-level lock** on each row it returns, held until the transaction ends. A second transaction that tries to lock the same rows **waits**. So worker 2 runs the same query, reaches row 1, finds it locked by worker 1, and blocks until worker 1 commits. Then it re-checks the row, finds it deleted, and moves on. In effect, workers run one at a time, and adding workers adds nothing but waiting. Under load you get a convoy: dozens of workers all blocked on the same first few rows.

~~~dg-compare Three ways to claim jobs from one table, and what a second worker experiences with each.
[Plain SELECT]
Worker 2 -(reads rows 1-10)-> Same jobs as worker 1
- Duplicates every job
[FOR UPDATE]
Worker 2 -(row 1 locked)-> Waits for worker 1
- Workers run one at a time
[*FOR UPDATE SKIP LOCKED]
Worker 2 -(rows 1-10 locked, skip)-> Rows 11-20
+ No duplicates, no waiting
~~~

## SKIP LOCKED

Postgres's `SELECT` has two options for what to do when a row is already locked:

- `NOWAIT`: raise an error immediately instead of waiting.
- `SKIP LOCKED`: silently **skip** rows that can't be locked immediately, and keep looking for rows that can.

With `SKIP LOCKED`, worker 1 locks rows 1 to 10. Worker 2's scan meets row 1, can't lock it, skips it without waiting, and so on until it finds row 11, which is free. It locks rows 11 to 20. Worker 3 gets 21 to 30. No duplicates, no blocking. MySQL 8.0 and Oracle support the same clause; SQL Server has a similar hint, `READPAST`.

`SKIP LOCKED` gives each query an **inconsistent view** of the table: it pretends locked rows aren't there. That would be wrong for a report or a balance check, so don't use it for general queries. For a queue, it's exactly what you want: "give me some work nobody else is doing right now".

~~~dg-flow Two workers on one table: SKIP LOCKED makes the second take the next free rows instead of waiting.
[Worker 1] Worker 1 -(FOR UPDATE SKIP LOCKED)-> *Rows 1-10 locked
[Worker 2] Worker 2 -(rows 1-10 locked, skip)-> *Rows 11-20 locked
[Worker 3] Worker 3 -(rows 1-20 locked, skip)-> *Rows 21-30 locked
~~~

### How the lock works underneath

Postgres doesn't keep row locks in a big lock table in memory. A row lock is recorded **on the row itself**: the locking transaction's ID is written into the row's header (the `xmax` field) with a flag saying it's a lock rather than a delete. When another transaction meets the row, it reads that header, sees the lock belongs to a transaction that is still running, and either waits for it (normal `FOR UPDATE`), errors (`NOWAIT`) or skips it (`SKIP LOCKED`). This is why row locks are cheap even for millions of rows, and why the lock vanishes automatically when the owning transaction commits or aborts.

## The dequeue pattern

A common form claims and deletes in one statement:

```
BEGIN;
DELETE FROM jobs
USING (
  SELECT id FROM jobs
  WHERE queue = 'email' AND run_at <= now()
  ORDER BY priority DESC, run_at
  LIMIT 10
  FOR UPDATE SKIP LOCKED
) q
WHERE q.id = jobs.id
RETURNING jobs.*;
-- process the returned rows
COMMIT;
```

Rows stay locked until `COMMIT`, so other workers skip them while this worker processes. If processing fails or the worker crashes, the transaction rolls back (Postgres aborts it when the connection drops) and the rows reappear as if they were never touched. This gives you **acknowledgement and redelivery for free**, from the database's normal transaction rules.

~~~dg-seq A worker's batch stays locked until commit; a crash rolls it back so another worker can take it.
Worker > Postgres : BEGIN
Worker > Postgres : DELETE ... SKIP LOCKED RETURNING
Postgres -> Worker : 10 rows, now locked
Worker > Worker : process rows
Worker > Worker : crash, connection drops
Postgres > Postgres : abort, rows unlocked
Worker 2 > Postgres : DELETE ... SKIP LOCKED RETURNING
Postgres -> Worker 2 : the same 10 rows
~~~

Notice what that means: if the worker did something external (sent an email) and then crashed before commit, the job runs again. Delivery is at least once, just like any queue. Jobs must be idempotent.

### Long jobs: claim, commit, then work

Holding a transaction open for the whole job is fine for jobs that take milliseconds. It's a problem for jobs that take minutes:

- Each in-flight job holds a database **connection**. 200 concurrent jobs need 200 connections, and Postgres connections are expensive (each is a process using several MB).
- A long-running transaction stops **vacuum** from cleaning up dead rows anywhere in the database that are newer than its snapshot, so tables bloat.

For long jobs, use a status-and-lease design: claim quickly, commit, then work.

```
UPDATE jobs SET status = 'running', locked_by = 'w7',
                lease_until = now() + interval '5 minutes', attempts = attempts + 1
WHERE id IN (
  SELECT id FROM jobs
  WHERE status = 'ready' AND run_at <= now()
  ORDER BY run_at LIMIT 10
  FOR UPDATE SKIP LOCKED)
RETURNING *;
```

The claim transaction lasts a few milliseconds. The worker then processes without a transaction open, heartbeats by extending `lease_until`, and finally marks the job done or deletes it. A sweeper resets jobs whose `lease_until` has passed back to `ready`. This is the same lease model as SQS visibility timeouts (lesson 08.01).

~~~dg-state A job row under the status-and-lease design. Only the claim step uses SKIP LOCKED; the lease handles crashes.
[*] -> Ready
Ready -> Running : claimed with SKIP LOCKED
Running -> Done : worker commits result
Running -> Ready : lease expired, sweeper resets
Running -> Ready : error, run_at pushed back
Running -> *Dead : attempts over limit
~~~

## Limits and how to live with them

### Table bloat

Postgres uses **MVCC** (multi-version concurrency control): an update or delete doesn't overwrite a row, it marks the old version dead and (for updates) writes a new one. A queue table with 2,000 inserts and 2,000 deletes a second creates 2,000 dead rows a second. **Autovacuum** must reclaim them, or the table and its index grow full of dead entries, and every `SELECT ... LIMIT 10` has to step over thousands of dead index entries to find live ones. Mitigations:

- Tune autovacuum to run more aggressively on the jobs table (lower `autovacuum_vacuum_scale_factor` for that table).
- Keep the table small: delete finished jobs promptly, move history to a separate table.
- Avoid long-running transactions anywhere in the database.
- For very high churn, partition the table and drop or truncate old partitions instead of deleting rows.

### Polling

Workers have to ask "any work?" repeatedly. Polling every second from 50 workers is 50 queries a second, which is cheap, but it adds up to a second of latency. Postgres's `LISTEN` / `NOTIFY` lets the enqueuing transaction send a notification on commit, waking idle workers immediately; workers still poll occasionally as a safety net, because notifications aren't stored if no one is listening.

### Throughput

A single well-tuned Postgres can handle a few thousand to around ten thousand job claims a second, depending on hardware, batch size and how much else the database is doing. Claiming in batches (10 to 100 rows per query) multiplies throughput. Beyond that, or if you need fan-out to many consumer groups or replay, use a broker.

~~~dg-bars Rough, illustrative throughput of a Postgres queue on one primary, with a partitioned Kafka topic for scale. Batching claims matters more than worker count.
One job per claim | 1500 | ~1.5k jobs/s
Batches of 10 | 6000 | ~6k jobs/s
Batches of 100 | 12000 | ~12k jobs/s
*Kafka partitioned topic | 500000 | 500k+ msgs/s
~~~

## When to use it

Use a Postgres queue when:

- You already run Postgres and don't want to operate a broker.
- Volume is moderate: thousands of jobs a second at most.
- You want **enqueue in the same transaction as the business write**, which solves the dual-write problem from lesson 08.03 with no outbox or relay. If the order insert rolls back, the job disappears with it.
- You want to query the queue with SQL: "how many jobs failed for customer 42 today?"

Use a broker when you need very high throughput, many independent consumers of the same message, replay, or cross-team event streams.

<div class="ex">
<div class="lab">Worked example</div>

**sending receipts**

The order service inserts the order and a `send_receipt` job in one transaction, so a receipt job exists if and only if the order exists. Eight workers loop on the SKIP LOCKED query in batches of 10. Peak load is 300 orders a second. Each receipt takes about 40 ms to render and send, so a batch of 10 takes 400 ms, and 8 workers handle 8 × 10 / 0.4 s = 200 receipts a second. That's below peak, so the team runs 16 workers: about 400 a second, with headroom.

Worker 3 crashes mid-batch after sending 4 of its 10 receipts. Its connection drops, Postgres aborts the transaction, and the 10 rows unlock. Another worker takes them and sends all 10. Four customers would get two receipts, so each job records `receipt_sent_at` on the order row inside its own small transaction and skips if it's already set. No message broker was needed.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**why FOR UPDATE alone stalls**

Without SKIP LOCKED, the same 16 workers all run `SELECT ... LIMIT 10 FOR UPDATE`. Worker 1 locks rows 1 to 10 and takes 400 ms. Workers 2 to 16 all wait on row 1. When worker 1 commits, they wake up, find rows 1 to 10 gone, and race for rows 11 to 20; one wins, 14 wait again. Throughput is about one batch per 400 ms, 25 receipts a second, no matter how many workers you add.

</div>

## Common mistakes

- Forgetting `ORDER BY` and expecting FIFO. Without it, Postgres returns whatever rows it finds first.
- No index matching the `WHERE` and `ORDER BY`, so every claim scans the table.
- Holding the claim transaction open while calling slow external APIs.
- Letting finished jobs accumulate, so the table and index bloat.
- Using `SKIP LOCKED` in ordinary business queries, where silently skipping rows gives wrong answers.

## In an interview

If the design already has Postgres and the job volume is modest, proposing a table-backed queue with `SKIP LOCKED` shows judgement: fewer moving parts, and the job is enqueued atomically with the business write. Say where it stops scaling (vacuum, connections, around ten thousand jobs a second) and what you'd move to.

<div class="take">
<div class="lab">Key takeaway</div>

`FOR UPDATE SKIP LOCKED` lets many workers each lock a different set of rows from the same table without waiting, turning Postgres into a reliable, transactional job queue for moderate volumes. Keep claim transactions short, delete finished jobs and watch for bloat.

</div>
