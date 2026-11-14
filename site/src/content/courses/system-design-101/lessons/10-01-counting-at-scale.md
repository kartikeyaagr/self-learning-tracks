---
id: "10.01"
course: "system-design-101"
module: "10"
slug: "counting-at-scale"
title: "Counting at scale"
keyIdeas: ["Counting things (likes, views, clicks) at high volume is hard because one counter row becomes a write hotspot with lock contention.", "Fixes: shard the counter into N sub-counters and sum on read; buffer increments in memory or Redis and flush to the DB in batches; stream events through Kafka and aggregate.", "Decide on exact vs approximate. Exact counts need dedup and careful retries; approximate counts can use sketches.", "Show slightly stale counts: users rarely notice if a like count lags a few seconds."]
question: "Why does one counter row in a DB fall over at 50k increments per second?"
answer: "Every increment must take an exclusive lock on the same row and hold it until its commit is flushed to the log, so all 50,000 writers queue on one lock and one sequence of commits. With around 1 ms per commit that caps the row at roughly a thousand to a few thousand updates per second; the waiting transactions exhaust the connection pool and the whole database stalls. Extra CPUs or replicas don't help, MVCC creates a dead row version per update, and a single row cannot be sharded. The fixes are to aggregate in memory (Redis INCR, or a stream consumer partitioned by key that sums counts and flushes one write per few seconds) and to split a hot counter into several sub-counters that are summed on read."
sources: [{"title": "Hello Interview: Top-K breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/top-k", "note": "Counting at scale end to end."}, {"title": "Redis docs: INCR", "url": "https://redis.io/docs/latest/commands/incr/", "note": "Atomic counters and their limits."}]
---

## Why counting gets hard

Counting looks like the easiest thing a computer can do: `UPDATE videos SET views = views + 1 WHERE id = 42`. At small scale it is. A blog with a few hundred readers a day can run that statement on every page view for years and never notice. The trouble starts when one number is incremented by a huge number of people at the same moment.

Consider a video platform with 70 billion views a day. That is about **700,000 increments per second** (70 billion divided by roughly 100,000 seconds in a day; a day is really 86,400 seconds, so rounding up to 100,000 makes the arithmetic easy but gives a slightly low answer: the true average is about 810,000, and peaks are higher still). A well-tuned relational database node handles on the order of 10,000 durable writes per second. A design where every view becomes its own database write misses by almost two orders of magnitude.

But this lesson's question is narrower and more surprising: why does even 50,000 increments per second on **one row** fall over, when the same database might happily take 50,000 writes per second spread across many rows? The answer is that a single row is a single point of serialisation, and nothing you add to the machine makes a single point faster.

## What happens inside one UPDATE

Walk through what a relational database such as PostgreSQL or MySQL (InnoDB) does for one `views = views + 1`:

1. **Find the row.** Look up id 42 in the primary-key B-tree. This is fast: the page is almost certainly in memory because it is so hot.
2. **Take an exclusive row lock.** No other transaction may modify this row until this one commits or aborts. This is what stops two increments from both reading 1,000 and both writing 1,001.
3. **Write the new version.** PostgreSQL writes a whole new copy of the row (MVCC, multi-version concurrency control: old readers keep seeing the old version). InnoDB updates in place but writes an undo record so readers can reconstruct the old version.
4. **Append a log record.** The change is recorded in the write-ahead log (WAL), the append-only file that makes the change durable.
5. **Commit.** Wait until the WAL record is flushed to disk with `fsync`, so a power cut cannot lose an acknowledged write.
6. **Release the lock.** Only now can the next transaction waiting on row 42 proceed.

The lock is held from step 2 until step 6. That window includes a disk flush, which on a good SSD takes perhaps 0.1 to 1 ms, and on network-attached cloud storage often 1 to 3 ms.

~~~dg-seq Two writers on the same row: the second one waits for the first one's whole commit, including the disk flush.
Writer A > Database : UPDATE views + 1 (row 42)
Database > Database : lock row 42 for A
Writer B > Database : UPDATE views + 1 (row 42)
Database > Database : B waits for lock
Database > Database : A: write WAL, fsync (1 ms)
Database -> Writer A : commit ok, release lock
Database > Database : lock row 42 for B
Database > Database : B: write WAL, fsync (1 ms)
Database -> Writer B : commit ok
~~~

## The arithmetic of a hot row

If every increment holds the lock for 1 ms, the row can complete at most **1,000 increments per second**, regardless of how many CPU cores the server has. Cores don't help because the work cannot run in parallel: increment number 2 must see the result of increment number 1.

**Group commit** improves this. When many transactions are waiting to flush the WAL, the database flushes all their log records in one `fsync`. That amortises the disk cost across many commits, but it does not remove the lock: transactions on the same row still take turns, so each one must wait for the previous holder to commit before it can even write its log record. In practice a single hot row in a well-tuned database tops out somewhere between a few hundred and a few thousand updates per second. Fifty thousand is out of reach by a factor of 10 to 100.

The failure is not graceful either. At 50,000 arriving per second and 2,000 completing per second, the queue of waiting transactions grows by 48,000 every second. Each waiting transaction holds a connection from the pool. Within a second or two, the pool is exhausted, unrelated queries cannot get a connection, timeouts fire, clients retry (adding even more load), and the whole database looks down even though only one row is hot.

### The other costs stacked on top

- **MVCC garbage.** In PostgreSQL every update leaves a dead row version behind. 50,000 updates per second on one row means 50,000 dead tuples per second for VACUUM to clean. Until it does, the table and its indexes bloat, and every reader has to skip past dead versions.
- **Replication.** Each update produces a log record that every replica must apply in order. A single-threaded replay on the replica becomes another bottleneck, and replicas fall behind.
- **No way to shard one row.** Partitioning (sharding) spreads different keys across machines. A single key always lands on one machine, so adding machines does nothing for one hot counter.
- **Deadlocks and retries.** If the same transaction also touches other rows (a user row, a daily stats row), lock ordering problems appear, and aborted transactions retry, which adds load at the worst moment.

~~~dg-bars Rough ceilings on increments per second for one logical counter, by technique. Exact figures vary with hardware; the orders of magnitude are what matter.
One SQL row, fsync each commit | 1000 | about 1k/s
One SQL row, group commit | 3000 | a few k/s
Redis INCR, one key | 100000 | about 100k/s
16 sharded SQL rows | 30000 | about 30k/s
*Batched in a stream consumer | 10000000 | millions/s of events
~~~

## Fix 1: count in memory with Redis INCR

`INCR key` increments the integer stored at a key by one and returns the new value. If the key doesn't exist it starts at 0. It is **O(1)**, works on 64-bit signed integers, and because Redis executes commands one at a time on a single thread, concurrent increments never lose updates: there is no read-modify-write race to protect against, so no lock is needed. With no disk wait per increment, one Redis node can absorb on the order of **100,000 simple commands per second per core**, and more with pipelining (sending many commands without waiting for each reply). `INCRBY key n` adds n in one step, which matters for batching.

The catch is durability. Redis is memory-first. If the counter must survive a crash exactly, you need the append-only file (AOF) with `appendfsync always`, which brings back a disk flush per write and much of the cost you were escaping. The common middle ground is `appendfsync everysec` (lose up to a second of increments on a crash) or a periodic job that copies counters to a database. A single hot key also still lives on one Redis node, so the 100,000 per second ceiling is real for one key.

## Fix 2: split the counter

A **sharded counter** stores N sub-counters for one logical counter, for example `views:42:0` to `views:42:15`. Each write picks a random sub-counter and increments only that one. A read sums all N.

~~~dg-flow A sharded counter: writes pick a random slot, so 16 writers rarely collide; reads add up all 16 slots.
[Write] View event -(random slot 0..15)-> *views:42:7 += 1
[Read] Read count -(sum 16 slots)-> views:42 = total
~~~

With 16 sub-counters, the lock contention on each row drops by 16 times, so a database that managed 2,000 increments per second on one row can now manage roughly 30,000 on one logical counter. The sub-counters can even live on different shards or Redis nodes. Reads cost N lookups, which is fine when reads are rarer than writes or are served from a short-lived cache ("refresh the displayed total every 5 seconds").

Choosing N is a trade-off. Too few and the hottest counter still contends; too many and every read gets slow and every counter costs N times the storage, even for the millions of cold counters that see one view a day. A common refinement is **adaptive sharding**: start every counter with one slot and split it only when it gets hot.

## Fix 3: batch before writing

The biggest win is to stop doing one write per event. Instead of telling the database about each view, tell it about the total for the last few seconds. Put view events onto **Kafka, partitioned by video ID**, so all events for video 42 go to one partition and therefore one consumer. That consumer (or a stream processor such as Flink) keeps a hash map of `video_id → pending count` in memory and every few seconds flushes one write per video, such as `views = views + 3,812`.

~~~dg-flow Batching: Kafka partitions by video, a consumer sums in memory, and the database sees one write per video per flush.
View events -(partition by video id)-> Kafka -> *Consumer sums per video for 5 s -(views += 3,812)-> Counter DB, sharded by video
~~~

Batching reduces write volume by roughly **2 to 100 times** in general, and far more for hot keys: a viral video's 40,000 views per second become one write every 5 seconds, a reduction of 200,000 times. This is why batching is also the answer to hot keys. Sharding the counter database by video ID (about 70 shards to bring 700,000 writes per second down to about 10,000 per instance) handles the remaining volume of cold and warm videos.

The trade-off is freshness: counts lag by the flush interval. Crash safety comes from Kafka. The consumer commits its Kafka offset only after the batch has been written. If it crashes before flushing, it restarts from the last committed offset and recounts those events. If it crashes after writing but before committing the offset, it will count those events twice when it restarts. Lesson 10.03 shows how to make that exactly right where it matters (ads); for a view count, a tiny overcount after a rare crash is usually acceptable.

<div class="ex">
<div class="lab">Worked example</div>

**a viral video, three designs**

Video 42 gets 40,000 views per second at peak.

Direct writes: 40,000 lock acquisitions per second on one row. With about 1 ms per commit, the row completes about 1,000 per second, the backlog grows by 39,000 per second, the connection pool of 200 is exhausted in well under a second, and other pages that need the database start timing out.

Sharded counter with 16 slots: about 2,500 increments per second per slot. Each slot still needs a commit per increment, so this is at the limit of what the database can do, and every page view of the count now costs 16 reads.

Batching: the Kafka partition for video 42 feeds one consumer, which adds to an in-memory counter and every 5 seconds runs `UPDATE videos SET views = views + 200000 WHERE id = 42`. That is one write per 5 seconds. The displayed count is up to 5 seconds stale, which nobody notices.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing the whole platform**

700,000 view events per second across all videos, so 3.5 million events in each 5-second window. Views follow a power law: the top few thousand videos absorb a large share of them, and most other videos get a single view. Suppose those 3.5 million events land on 1.5 million distinct videos. With a 5-second flush, the consumers emit 1.5 million writes per 5 seconds, about 300,000 per second: only a 2 to 3 times reduction overall, but a 100,000 times reduction on the hottest rows, which were the ones that broke. Spread over 70 database shards, that is about 4,000 writes per second each, which is comfortable. Lengthening the flush to 30 seconds for cold videos cuts it further.

</div>

## Picking a technique

| Situation | Use |
| --- | --- |
| Low rate, must be exact and transactional with other data | Plain SQL update |
| Medium rate, must be readable instantly | Redis INCR, with periodic copy to a database |
| One key gets very hot, reads are rare | Sharded counter |
| Very high total rate, seconds of staleness fine | Stream batching into sharded storage |
| Huge scale, distinct users rather than raw events | HyperLogLog (lessons 10.02 and 10.04) |

Real systems combine them: Kafka batching for durability and volume, Redis for the "live" number shown on screen, and a periodic reconciliation job that recounts from raw events (lesson 10.07).

## Common mistakes

- **Reaching for a bigger database.** More cores, more RAM and read replicas do nothing for one row's lock. The fix is structural, not hardware.
- **SELECT then UPDATE in application code.** Reading the count, adding one in your code and writing it back loses updates under concurrency (two readers both see 1,000 and both write 1,001). Always increment atomically in the store (`SET views = views + 1`, `INCR`).
- **Retrying blindly.** When the hot row times out, client retries multiply the load. Retries need backoff, and counters usually need to drop or batch rather than retry.
- **Forgetting the read side.** A sharded counter with 64 slots read on every page view moves the problem from writes to reads. Cache the summed value.
- **Assuming the batch is exactly-once.** A crash between "write batch" and "commit offset" double-counts. Decide whether that matters before you choose the design.

## In an interview

When a design includes "count likes", "count views" or "count clicks", say early that you will not write to a database per event. State the rate ("at peak 50,000 per second on a single viral item"), name the bottleneck ("a single row serialises on its lock and its commit"), and propose batching through a partitioned log, with sharded counters or Redis for the hottest keys. Then ask the question that decides the rest: "How exact and how fresh does this number need to be?" A like count can be 5 seconds stale and slightly off; an account balance cannot.

<div class="take">
<div class="lab">Key takeaway</div>

one counter row serialises every increment on a single lock and a durable commit, so it tops out at a few thousand per second however big the server is. Count in memory (Redis INCR), split hot counters into sub-counters, and above all batch increments in a stream consumer so the database sees one write per key per interval.

</div>
