---
id: "03.06"
course: "system-design-101"
module: "03"
slug: "database-locking-and-isolation"
title: "Database locking and isolation"
keyIdeas: ["Isolation levels define what concurrent transactions can see. Weaker levels allow anomalies in exchange for less blocking.", "Read committed: no dirty reads, but non-repeatable reads and lost updates can happen. Postgres default.", "Repeatable read / snapshot isolation: each transaction sees a stable snapshot. Write skew is still possible.", "Serializable: result is as if transactions ran one at a time. Safest, but some transactions abort and must retry.", "Lost update example: two transactions read balance 100, both add 10, both write 110. Fix with `SELECT ... FOR UPDATE`, an atomic `UPDATE ... SET balance = balance + 10`, or optimistic version checks.", "Locks can deadlock; databases detect this and abort one transaction."]
question: "Give an example of a lost update and the isolation level or lock that prevents it."
answer: "A lost update: a wallet holds 100. Transaction 1 reads 100 to deduct 30 and transaction 2 reads 100 to deduct 50. Transaction 1 writes 70 and commits, then transaction 2 writes 50 and commits, so the first deduction is silently lost (it should be 20). Read committed, the default in Postgres, does not prevent this because the computation happens in the application. Prevent it with SELECT ... FOR UPDATE so the second transaction waits for the row lock and then reads 70; or an atomic UPDATE wallets SET balance = balance - 30 WHERE id = 1 AND balance >= 30; or optimistic locking with a version column (UPDATE ... WHERE version = 7); or repeatable read or serializable isolation, where the second transaction's update fails with a serialisation error and is retried."
sources: [{"title": "PostgreSQL docs: Transaction Isolation", "url": "https://www.postgresql.org/docs/current/transaction-iso.html", "note": "Read committed, repeatable read, serializable."}, {"title": "PostgreSQL docs: Explicit Locking", "url": "https://www.postgresql.org/docs/current/explicit-locking.html", "note": "Row and table locks, deadlocks."}, {"title": "Jepsen: Consistency models", "url": "https://jepsen.io/consistency", "note": "Map of isolation levels and anomalies."}]
tools: ["PostgreSQL"]
---

## Concurrency causes anomalies

A busy database runs hundreds of transactions at the same moment. Their reads and writes **interleave**: transaction A reads a row, then B writes it, then A writes it. The ideal is that the final result is the same as if the transactions had run one at a time in some order. That property is called **serializability**. Without care, interleaving produces results that no one-at-a-time order could ever produce. These wrong results are called **anomalies**.

The classic anomalies, from mildest to subtlest:

- **Dirty read**: you read another transaction's change before it commits. If that transaction then rolls back, you acted on data that never existed.
- **Non-repeatable read**: you read the same row twice in one transaction and get different values, because another transaction committed a change in between.
- **Phantom read**: you run the same query with a condition twice (`WHERE status = 'open'`) and get a different set of rows, because someone inserted or deleted a matching row in between.
- **Lost update**: two transactions read the same value, both compute a new value from it, and both write. The second write silently overwrites the first. It is not in the SQL standard's list, but it is the most common concurrency bug in real applications.
- **Write skew**: two transactions read overlapping data, each makes a decision based on it, and each writes a **different** row. Neither overwrites the other, yet together they break a rule. A well-known case is two doctors on call: the rule is that at least one must stay on call, both check that two are on call, and both remove themselves.
- **Serialisation anomaly**: the general term for any set of committed transactions whose combined result matches no serial order. Write skew is the most common example.

## The lost update in detail

~~~dg-seq A lost update on a wallet: both transactions read 100, and the second write erases the first deduction.
Tx 1 > Wallet row : SELECT balance
Wallet row -> Tx 1 : 100
Tx 2 > Wallet row : SELECT balance
Wallet row -> Tx 2 : 100
Tx 1 > Wallet row : UPDATE balance = 70
Tx 1 > Wallet row : COMMIT
Tx 2 > Wallet row : UPDATE balance = 50
Tx 2 > Wallet row : COMMIT, final 50 not 20
~~~

The pattern is **read, compute in the application, write back**. It hides in many places: incrementing a counter by reading it first, editing a JSON document by loading it, changing it and saving it, decrementing stock in an online shop. Every one of them is safe in testing, where requests arrive one at a time, and broken in production, where two requests can land in the same 10 ms.

## Isolation levels

**Isolation** is the "I" in ACID: how much concurrent transactions are shielded from each other. The SQL standard defines four levels, each allowing fewer anomalies, and databases interpret them differently. In Postgres:

| Level | What a transaction sees | What it prevents |
| --- | --- | --- |
| Read uncommitted | Same as read committed in Postgres | Dirty reads |
| Read committed (default) | Each **statement** sees data committed before that statement began | Dirty reads |
| Repeatable read | The whole **transaction** sees one snapshot taken at its first statement (snapshot isolation) | Also non-repeatable reads, phantoms and lost updates on the same row |
| Serializable | One snapshot, plus tracking of read/write dependencies | Everything: the result always equals some serial order |

~~~dg-matrix Which anomalies each Postgres isolation level allows. "no" means the anomaly cannot happen at that level.
Level | Dirty read | Non-repeatable read | Phantom | Lost update | Write skew
Read committed | no | yes, possible | yes, possible | yes, possible | yes, possible
Repeatable read | no | no | no | no, aborts one | yes, possible
Serializable | no | no | no | no | no
~~~

MySQL's InnoDB differs: its default is repeatable read, and it uses locks on index ranges (gap locks) for locking reads rather than Postgres's snapshot rules. The names match across databases but the exact behaviour does not, so check before relying on a level.

### Read committed and its subtlety

At read committed, an `UPDATE` that finds a row changed and committed by another transaction **waits** for that transaction, then re-reads the latest committed version, re-checks its `WHERE` clause against it and applies the change to that version. So a single statement such as `UPDATE counters SET n = n + 1 WHERE id = 1` is safe: the increment is computed inside the database on the newest value.

But read-then-write in application code is **not** safe: `SELECT n` returns 5, the application computes 6, `UPDATE counters SET n = 6` overwrites another session's increment. The database cannot know that your 6 came from a stale 5.

### Repeatable read fails instead of waiting

At repeatable read, your transaction works on a snapshot. If it tries to update a row that another transaction changed and committed after your snapshot was taken, Postgres does not quietly apply your change on top. It raises `could not serialize access due to concurrent update`. The lost update is prevented by turning it into an error. Your application must **retry the whole transaction**, which will then read the new value.

### Serializable catches write skew

Repeatable read still allows write skew, because the two transactions write different rows and so never collide on a row. Serializable (in Postgres, **Serializable Snapshot Isolation**, SSI) also records what each transaction **read**, and looks for dangerous patterns of read/write dependencies between concurrent transactions. When it finds one, it aborts one of them.

An example: transaction A sums the amounts in class 1 and inserts the total as a new row in class 2, while transaction B sums class 2 and inserts its total into class 1. Under repeatable read both commit, and the result matches neither order (in any serial order, one of them would have seen the other's insert). Under serializable, one of them fails with SQLSTATE `40001` (serialization failure) and must be retried.

Serializable has costs: more aborts under contention, and some memory for tracking reads. Keep serializable transactions short, retry them automatically, and mark read-only ones `READ ONLY`, which lets Postgres optimise them.

## Explicit locks

Isolation levels are one tool. Locks are the other.

- **Row locks**: `SELECT ... FOR UPDATE` reads rows and locks them, so no other transaction can update, delete or lock them until you commit. It is the classic fix for read-modify-write. Plain `SELECT`s are not blocked, thanks to MVCC (lesson 03.07). `FOR UPDATE SKIP LOCKED` skips rows other sessions hold, which turns a table into a simple job queue for multiple workers. `FOR UPDATE NOWAIT` errors instead of waiting.
- **Table locks** are taken automatically by every statement. Most are harmless and compatible with each other. The dangerous one is `ACCESS EXCLUSIVE`, taken by many `ALTER TABLE` forms, `DROP`, `TRUNCATE` and `VACUUM FULL`: it blocks even reads. Plain `CREATE INDEX` takes a lock that blocks writes (lesson 03.11).
- **Advisory locks**: locks on an application-chosen number, for example `pg_advisory_lock(42)`. The database does not attach them to any row; they are useful for "only one worker runs this nightly job at a time".

~~~dg-seq SELECT FOR UPDATE fixes the lost update: the second transaction waits for the lock, then reads the new balance.
Tx 1 > Wallet row : SELECT balance FOR UPDATE
Wallet row -> Tx 1 : 100, row locked
Tx 2 > Wallet row : SELECT balance FOR UPDATE
Wallet row > Wallet row : Tx 2 waits for lock
Tx 1 > Wallet row : UPDATE balance = 70, COMMIT
Wallet row -> Tx 2 : 70, row locked
Tx 2 > Wallet row : UPDATE balance = 20, COMMIT
~~~

### Optimistic locking

Instead of locking, you can **detect** conflicts. Add a `version` column. Read the row and remember its version, say 7. Write with `UPDATE accounts SET balance = 70, version = 8 WHERE id = 1 AND version = 7`. If another transaction got there first, the version is now 8, the `UPDATE` matches zero rows, and the application knows to reload and retry. This is called **optimistic concurrency control** because it assumes conflicts are rare. It holds no locks between the read and the write, so it works across HTTP requests (a user editing a form for five minutes) where holding a database lock would be unthinkable.

~~~dg-compare Pessimistic locking waits for conflicts it expects; optimistic locking detects conflicts after the fact and retries.
[Pessimistic: FOR UPDATE]
Lock row -> Compute -> Write -> Commit releases
+ No retries needed
+ Good under heavy contention
- Others wait while you hold it
- Cannot span user think time
[*Optimistic: version check]
Read version 7 -> Compute -> Write WHERE version = 7
+ No locks held
+ Works across HTTP requests
- Retries under heavy contention
~~~

## Deadlocks

A **deadlock** happens when two transactions each hold a lock that the other is waiting for. Transaction A updates account 1 then account 2; transaction B updates account 2 then account 1. A holds 1 and waits for 2; B holds 2 and waits for 1. Neither can proceed. Postgres checks for cycles after a short wait (`deadlock_timeout`, 1 second by default), picks one transaction and aborts it with a deadlock error.

To avoid deadlocks: always lock rows in a **consistent order** (for example, by ascending ID, so a transfer from 2 to 1 still locks 1 first), keep transactions short, and avoid user interaction inside a transaction. Treat deadlock errors like serialisation failures: retry.

## The bigger map

All of this is one corner of a wider subject: **consistency models**, which are rules about which histories of operations a system may produce. They are often defined by which dependency cycles between transactions are forbidden. Serializable is near the top. Snapshot isolation and read committed are lower down. Distributed systems add models such as **linearizability** (every operation appears to take effect at one instant between its start and end) and **causal consistency** (Module 6).

<div class="ex">
<div class="lab">Worked example</div>

**lost update on a wallet, and four fixes**

The balance is 100. T1 reads 100 to deduct 30. T2 reads 100 to deduct 50. T1 writes 70 and commits. T2 writes 50 and commits. The final balance is 50; it should be 20. The customer just got 30 for free.

Fix 1, atomic statement: `UPDATE wallets SET balance = balance - 30 WHERE id = 1 AND balance >= 30`. The arithmetic happens in the database on the latest value, and the `balance >= 30` check prevents overdrafts. Check the affected row count: zero means insufficient funds.

Fix 2, pessimistic lock: `SELECT balance FROM wallets WHERE id = 1 FOR UPDATE`, compute, update, commit. T2 waits about 2 ms for T1, then reads 70.

Fix 3, optimistic: `UPDATE ... SET balance = 70, version = 8 WHERE id = 1 AND version = 7`. T2's update matches zero rows; it re-reads 70 and version 8 and retries.

Fix 4, isolation: run at repeatable read. T2's update raises a serialisation error; the retry reads 70 and writes 20.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**write skew on a meeting room**

Rule: a room may not be double-booked. Two users book room 4 for 10:00 at the same moment. Each transaction runs `SELECT count(*) FROM bookings WHERE room = 4 AND slot = '10:00'`, sees 0 and inserts a booking.

The two inserts are different rows, so `FOR UPDATE` on the select locks nothing (there was no row to lock), and repeatable read lets both commit. Now the room has two bookings.

Fixes: a unique constraint on `(room, slot)`, which turns the race into a constraint error (the best fix); or serializable isolation, which detects the read/write dependency and aborts one; or an exclusion constraint for overlapping time ranges. This is why constraints (lesson 03.01) are so valuable: they catch races that isolation levels below serializable miss.

</div>

## Common mistakes

- **Assuming the default level is serializable.** Postgres, Oracle and SQL Server default to read committed; MySQL to repeatable read.
- **Read-modify-write in application code** without a lock, version check or atomic statement.
- **Using serializable without retries.** Serialisation failures are expected, not bugs.
- **Locking in inconsistent orders**, which invites deadlocks.
- **Holding transactions open across network calls** or user think time, which holds locks and snapshots for seconds.

## In an interview

When two writers can touch the same data (stock counts, seat bookings, balances), say how you will stop lost updates before you are asked: an atomic conditional update for counters, `FOR UPDATE` or a version column for read-modify-write, and a unique constraint for "only one" rules. Mention that the default isolation level does not protect application read-modify-write cycles.

<div class="take">
<div class="lab">Key takeaway</div>

the default isolation level (read committed) does not stop read-modify-write races. Prevent lost updates with atomic statements, SELECT FOR UPDATE, optimistic version checks, or repeatable read and serializable with automatic retries, and back business rules with constraints.

</div>
