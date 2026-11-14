---
id: "03.08"
course: "system-design-101"
module: "03"
slug: "database-wal-and-recovery"
title: "Database WAL and recovery"
keyIdeas: ["Write-ahead logging: before changing data pages, append a description of the change to a sequential log and flush it to disk. A commit is durable once its log record is on disk.", "Data pages can be written lazily later, since sequential log writes are much cheaper than random page writes.", "On crash, recovery replays the log from the last checkpoint to redo committed changes (and undo uncommitted ones in ARIES-style systems).", "Checkpoints flush dirty pages and mark a point so recovery doesn't replay the whole log.", "The WAL also drives replication (stream it to replicas) and point-in-time recovery (base backup + archived WAL)."]
question: "The server crashes after COMMIT returns but before data pages are flushed. How is the data recovered?"
answer: "Before COMMIT returned, the transaction's change was recorded in the write-ahead log and the log, including its commit record, was fsynced to durable storage; the data page itself was only changed in memory. When the server restarts, the database finds the last checkpoint and replays the WAL forward from it. For each record it compares the record's LSN with the LSN stored on the target page, and reapplies any change the page is missing (full-page images in the log repair any torn pages). Because the commit record is in the log, the transaction counts as committed, so the paid invoice is fully restored. Uncommitted transactions are replayed too but stay invisible under MVCC (or are undone in engines like InnoDB)."
sources: [{"title": "PostgreSQL docs: Write-Ahead Logging", "url": "https://www.postgresql.org/docs/current/wal-intro.html", "note": "Why logging before writing gives durability."}, {"title": "CMU 15-445: Logging and Recovery lectures", "url": "https://15445.courses.cs.cmu.edu/", "note": "ARIES-style recovery."}]
tags: ["Code"]
tools: ["PostgreSQL"]
---

## The durability problem

When `COMMIT` returns, the application tells the user "payment received". From that moment the change must survive anything: a power cut, a kernel panic, the database process being killed. That is the **D** in ACID, **durability**.

The naive way to provide it is to write every changed data page to its place on disk before returning from `COMMIT`. That has two problems.

- **It is slow.** A transaction that updates an order, a payment and three index entries dirties five pages scattered across different files. Writing them means five random writes, each followed by a wait for the disk to confirm.
- **It is not even safe.** If the machine loses power after two of the five pages are written, the files are left half-updated: the payment exists but the order does not. A single 8 KB page can also be **torn**, with the first 4 KB new and the last 4 KB old, because disks only guarantee atomic writes of smaller sectors.

The opposite naive approach, writing pages lazily and returning immediately, loses committed data on every crash. The database needs a way to make commits durable without writing data pages at commit time.

~~~dg-layers Where a committed change can be when the power goes out. Only what has reached durable storage survives.
Application | told "committed"
Database buffer pool | changed pages in RAM, lost on crash
WAL buffer | log records in RAM, lost on crash
OS page cache | written but not fsynced, lost on power cut
*WAL on disk | fsynced log, survives the crash
Data files on disk | updated later at checkpoint
~~~

## Write-ahead logging

The solution is the **write-ahead log (WAL)**, also called a redo log or journal. The rule is one sentence: **a change to a data page may only be written to disk after the log record describing that change has been written to durable storage.** Log first, data later.

A commit then works like this:

1. The transaction changes pages in memory, in the **buffer pool** (Postgres calls it shared buffers). The pages are now **dirty**: newer in memory than on disk.
2. For every change, a **log record** is appended to the WAL buffer in memory. A record says, in effect, "on page 4711 of table orders, at slot 3, write these bytes". Each record gets a **log sequence number (LSN)**, its byte position in the log, which only ever increases. Each data page remembers the LSN of the last record that changed it.
3. At `COMMIT`, a commit record is appended and the WAL is flushed to disk with **fsync** (a system call that waits until the device confirms the data is durable). Only then is the client told "committed".
4. The dirty data pages are written to their real locations **later**, in the background, by a writer process or at a checkpoint. Before writing a page, the database checks that the WAL is flushed at least up to that page's LSN. That check is the write-ahead rule in code.

~~~dg-seq A commit with WAL: the log is flushed before COMMIT returns; the data page reaches disk much later.
Tx > Buffer pool : modify page 4711 in memory
Tx > WAL buffer : append record, LSN 0/5A10
Tx > WAL buffer : append commit record
WAL buffer > WAL on disk : write and fsync
WAL on disk -> Tx : durable
Tx -> Tx : return COMMIT to client
Buffer pool > Data files : write page 4711 at checkpoint
~~~

### Why this is fast

- The WAL is written **sequentially**, always appending at the end. Sequential writes are the fastest thing a disk can do.
- A commit needs **one** fsync, of the log, no matter how many pages it changed.
- **Group commit**: if 50 transactions commit within the same millisecond, one fsync can make all of their commit records durable at once. Throughput rises with concurrency instead of being capped at one transaction per fsync.
- Data pages are written lazily, so a hot page changed 1,000 times between checkpoints is written to disk once, not 1,000 times.

The cost of an fsync depends heavily on the hardware. A datacentre SSD with power-loss protection can confirm in 20 to 100 microseconds. A consumer SSD without it may take several milliseconds. A spinning disk takes about one rotation, 4 to 8 ms. Cloud network block storage typically sits around 0.5 to 2 ms.

~~~dg-bars Typical time for one durable fsync of the log, which sets the floor for a single commit's latency.
Enterprise SSD with power-loss protection | 0.05 | about 50 µs
Cloud network block storage | 1 | about 1 ms
Consumer SSD | 3 | 1 to 5 ms
*Spinning disk | 6 | 4 to 8 ms
~~~

## Recovery after a crash

On restart after a crash, the data files may be stale (pages never written) or contain a mix of old and new pages. The WAL contains everything that happened. Recovery works as follows:

1. Find the **last checkpoint** recorded in the WAL (its location is stored in a small control file).
2. Read the log forward from there. For each record, look at the target page. If the page's LSN is older than the record's LSN, the change never reached disk, so **redo** it. If the page's LSN is already newer, skip the record. This makes replay **idempotent**: replaying a record twice has no extra effect.
3. Continue to the end of the valid log. The database is now exactly as it was at the moment of the crash.
4. Accept connections.

This is **roll-forward**, or **REDO**, recovery. In Postgres there is no undo phase. Changes from transactions that never committed are redone too, but their tuples carry an `xmin` of a transaction that has no commit record, so MVCC (lesson 03.07) treats them as invisible, and vacuum later removes them.

Engines that write **uncommitted** changes into data pages in place, such as InnoDB, need one more step. The general recovery design, called **ARIES**, has three phases: **analysis** (work out which transactions were active and which pages were dirty at the crash), **redo** (replay history, including uncommitted changes) and **undo** (roll back the changes of transactions that never committed, using undo records).

### Torn pages and full-page writes

Redo assumes the page on disk is a valid old version. A torn page breaks that. Postgres therefore writes a **full copy of each page** into the WAL the first time that page is modified after a checkpoint (`full_page_writes`). Recovery restores the whole page from the log first, then applies the later records. This makes WAL noticeably larger just after each checkpoint. InnoDB solves the same problem with a separate doublewrite buffer.

## Checkpoints bound recovery time

Without checkpoints, recovery would replay the WAL from the very beginning. A **checkpoint** periodically flushes every dirty page to the data files and writes a checkpoint record into the WAL. After that, all changes before the checkpoint are safely in the data files, so recovery starts from there, and older WAL segments can be recycled.

The trade-off: frequent checkpoints mean short recovery but more write I/O (pages flushed more often, and more full-page images in the WAL). Infrequent checkpoints mean less I/O but longer recovery. Postgres's defaults are a checkpoint every 5 minutes (`checkpoint_timeout`) or after about 1 GB of WAL (`max_wal_size`), whichever comes first, with writes spread over most of the interval to avoid I/O spikes.

~~~dg-timeline A crash and recovery. Only the WAL after the last checkpoint needs replaying.
10:00:00 | Checkpoint: all dirty pages flushed
10:01:12 | Tx 501 commits payment, WAL fsynced
10:02:40 | Tx 502 starts, changes 3 pages, no commit
10:03:05 | *Power lost: data page for Tx 501 never written
10:03:50 | Restart: read WAL from 10:00:00 checkpoint
10:03:58 | Redo Tx 501 and Tx 502 records into pages
10:04:00 | *Open: Tx 501 visible, Tx 502 invisible
~~~

## WAL does much more than crash recovery

Once every change is in one ordered log, the log becomes the backbone of several other features:

- **Point-in-time recovery (PITR)**: take a base backup (a copy of the data files) and archive every WAL segment after it. To recover from a mistake, restore the backup and replay WAL up to any chosen moment, such as 14:01:59, just before someone ran `DROP TABLE`. The base backup does not even need to be an instant, consistent snapshot, because replaying the log from its start repairs any inconsistency.
- **Replication**: replicas receive the WAL stream and replay it, so they stay a few milliseconds behind the primary (Module 6). This is called physical or streaming replication.
- **Logical decoding and change data capture (CDC)**: the WAL can be decoded into row-level change events ("order 7 updated: status shipped") and published to other systems. Tools such as Debezium do this to feed Kafka (Module 8).

## Trade-offs you can tune

- `synchronous_commit = off` returns success before the WAL flush. The commit is still written within a few hundred milliseconds (Postgres flushes about every 200 ms by default), and a crash loses at most that window of recent commits. Crucially, the database stays **consistent**: you lose the newest transactions, never half of one. It suits metrics or session data, not payments, and can be set per transaction.
- **Hardware that lies.** A disk with a volatile write cache and no battery or capacitor may report "done" for an fsync while the data is still in its RAM. A power cut then loses committed transactions, and no database setting can fix that. Use drives with power-loss protection.
- **`fsync = off`** disables flushing altogether. A crash can then corrupt the whole database, not just lose recent commits. It is only for throwaway test data.

~~~dg-compare Default synchronous commit versus asynchronous commit for a high-rate event table.
[*synchronous_commit = on]
Change -> WAL fsync -> Reply to client
+ Committed means durable
- Each commit waits for fsync
[synchronous_commit = off]
Change -> Reply to client -> WAL fsync within 200 ms
+ Much lower commit latency
+ Database stays consistent
- Crash loses last fraction of a second
~~~

<div class="ex">
<div class="lab">Worked example</div>

**crash after commit**

A user pays an invoice. The `UPDATE invoices SET status = 'paid'` changes page 812 in memory and appends a WAL record at LSN 3/A1000. `COMMIT` appends a commit record, fsyncs the WAL (0.1 ms on the server's SSD) and returns success. The payment page shows "Thank you".

One second later, before any checkpoint, the server loses power. Page 812 on disk still says "unpaid".

On restart, Postgres finds the last checkpoint at LSN 3/9F000, replays forward and reaches the record at 3/A1000. Page 812 on disk has an LSN older than that, so the change is reapplied. The commit record is present, so the transaction counts as committed. When the database opens, the invoice shows as paid. Total recovery time: a few seconds.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**what group commit buys you**

A service commits small transactions from 200 concurrent connections. The log disk's fsync takes 1 ms.

Without group commit, commits would be serialised at one per fsync: at most 1,000 commits per second, however fast the CPU is.

With group commit, each fsync carries the commit records of every transaction waiting at that moment. If 40 are waiting on average, the same disk sustains around 40,000 commits per second, while each commit still takes only about 1 to 2 ms. Throughput scales with concurrency; single-commit latency does not drop below the fsync time.

</div>

## Common mistakes

- **Believing data is safe once written to a file.** Without fsync it may still be in the OS cache.
- **Turning off fsync for speed.** You trade a small speed gain for possible total corruption.
- **Ignoring WAL volume.** Bulk loads and frequent checkpoints can generate gigabytes of WAL per minute, filling disks and lagging replicas.
- **Forgetting archived WAL.** Point-in-time recovery only works if every segment since the base backup was kept.
- **Treating a replica as a backup.** A replica replays a `DROP TABLE` within milliseconds; only PITR undoes it.

## In an interview

If asked how a database survives a crash, explain the write-ahead rule, the single sequential fsync at commit, and redo from the last checkpoint. Then connect it to replication and change data capture: they are all the same log. Mentioning group commit and the cost of fsync on different hardware shows real depth.

<div class="take">
<div class="lab">Key takeaway</div>

log first, data later. A commit only needs a sequential fsync of the write-ahead log; data pages are written lazily. After a crash, the database replays the log from the last checkpoint and rebuilds every committed change. The same log powers point-in-time recovery, replication and change data capture.

</div>
