---
id: "15.07"
course: "system-design-101"
module: "15"
slug: "a-kafka-style-log"
title: "A Kafka-style log"
keyIdeas: ["A Kafka-style log: topics split into partitions; each partition is an ordered, append-only log on disk, replicated across brokers.", "Producers append to a partition (chosen by key hash, so the same key always goes to the same partition). Consumers read sequentially and track their own offset.", "Order is guaranteed **within a partition only**. Need per-user order? Key by user ID.", "Replication: each partition has a leader and followers; the in-sync replica set (ISR) must have a write before it's committed (with `acks=all`).", "Speed comes from sequential disk I/O, the OS page cache, batching and zero-copy transfer.", "Consumer groups split partitions among consumers, so parallelism is capped at the partition count."]
question: "How does Kafka guarantee order, and over what scope?"
answer: "Kafka guarantees order only within a single partition: each partition is an append-only log, every message gets the next offset, the idempotent producer keeps retried batches in sequence, and consumers read a partition in offset order, with exactly one consumer per partition in a group. There is no order across partitions of a topic. Producers choose the partition by hashing the message key, so all messages with the same key (say the same order ID) go to the same partition and stay in order relative to each other. To keep related events ordered, give them the same key and do not change the partition count; keyless messages are spread across partitions and lose relative order."
sources: [{"title": "Jay Kreps: The Log", "url": "https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying", "note": "Why logs."}, {"title": "Kafka docs: Design", "url": "https://kafka.apache.org/documentation/#design", "note": "Partitions, replication, ISR."}, {"title": "Hello Interview: Kafka deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/kafka", "note": "Interview view."}]
tags: ["Design"]
tools: ["Kafka"]
---

## The log

Start from the simplest storage structure there is: an **append-only sequence of records, ordered by time**. You only ever add at the end, and every record gets the next sequential number, its **offset**. Record 0, record 1, record 2. That number acts as a timestamp that does not depend on any clock, so there is never a question of which of two records came first: the one with the lower offset.

Databases already depend on logs. Every change goes to the write-ahead log before the tables (lesson 03.08), and replicas stay in sync by copying the log. The principle underneath is **state machine replication**: if deterministic processes read the same inputs in the same order, they end up in the same state. So a replica's state is described by one number, the last offset it has applied, and the **order** is everything: "+1 then ×2" applied to 3 gives 8, "×2 then +1" gives 7.

Tables and logs are two views of the same thing. Applying a log of changes in order gives you the table; recording a table's changes gives you the log. Bank transactions and balances are the everyday example: the balance is a table, the statement is a log, and you can always rebuild the balance from the statement.

## The log as a company's data backbone

At LinkedIn the problem was data integration. With N systems (databases, search, Hadoop, monitoring, caches, recommendation engines), connecting each pair directly means up to N² custom pipelines, each with its own format, failure modes and backlog. Instead, every source publishes its changes to a log and every consumer reads it **at its own pace**, tracking its own position. Adding a consumer changes nothing for the producer; a slow consumer does not slow anyone else. LinkedIn built Kafka for this, and it was handling over 60 billion message writes a day early in its life; large deployments today handle trillions.

~~~dg-flow N producers and M consumers meet at one log instead of N × M point-to-point pipelines.
[Producers] Profile DB changes, page views, metrics -(append)-> *Kafka topics
[Consumers] Kafka topics -(each reads at its own offset)-> Search index, Hadoop, caches, recommendations
~~~

## Partitions: how a log scales

A single log cannot hold a company's data on one machine or take all its writes, so a **topic** (a named stream, such as `orders`) is split into **partitions**. Each partition is its own ordered log, stored on one broker (a Kafka server) and replicated to others. On disk a partition is a directory of **segment files** (1 GB each by default) plus an index from offset to file position, so finding offset 1,234,567 is a quick lookup, and old data is deleted a whole segment at a time.

~~~dg-tree How a Kafka topic is laid out: partitions are the unit of order and parallelism, segments the unit of storage and deletion.
Topic orders | 6 partitions, replication factor 3
  Partition 0 | leader on broker 1, offsets 0 to 9,812,004
    *Active segment | appends go here
    Older segments | 1 GB each, deleted by age or size
    Offset index | offset to file position
  Partition 1 | leader on broker 2, its own offsets
  Partitions 2 to 5 | spread over brokers 1 to 6
~~~

The trade: **each partition is totally ordered; there is no global order across partitions**. Offset 5 in partition 0 and offset 5 in partition 1 have no relationship at all. In return, appends to different partitions need no coordination, so throughput grows with the number of partitions and brokers.

### Choosing the partition

How producers choose a partition:

- With a **key**, the key is hashed (murmur2 by default) and taken modulo the number of partitions. The same key always lands in the same partition, so **all messages for one key are in order**.
- Without a key, messages are spread across partitions for balance (newer clients fill a batch for one partition, then move on), and related messages have no order between them.

So Kafka's ordering guarantee is: **messages are ordered within a partition, by offset**. Choose the key so that everything that must stay in order shares it: `user_id` for a user's actions, `order_id` for an order's state changes, `account_id` for ledger entries. Messages from one producer to one partition are stored in the order sent.

<div class="ex">
<div class="lab">Worked example</div>

**why the key matters for an order's events**

Order 42 emits `created`, `paid`, `shipped`. With key `order_id = 42`, all three hash to partition 1 and get offsets 7,001, 7,002, 7,003. A consumer always sees them in that order.

Without a key, `created` lands in partition 4, `paid` in partition 0 and `shipped` in partition 2. Three different consumers may process them in parallel, and the warehouse may receive `shipped` before `paid`, then reject it.

Order 77's events, with key 77, land in partition 3. There is no order between order 42 and order 77, and none is needed.

</div>

### Pitfalls with ordering

- **Changing the partition count** changes `hash(key) mod N`, so a key's new messages go to a different partition while its old ones are still being consumed from the old one. Order breaks across the change. Pick enough partitions up front (for example, enough for 2 to 3 years of growth).
- **Retries can reorder.** If a producer has several batches in flight to one partition and the first fails and is retried, the second may land first. The **idempotent producer** (on by default in recent versions) uses sequence numbers so the broker rejects out-of-order batches, preserving order with up to 5 in-flight requests.
- **Hot partitions**: if one key gets huge traffic (a viral ad ID, say), its partition becomes a bottleneck, because one partition is consumed by one consumer. Options: drop the key (lose order), add a random suffix (spreads load but complicates aggregating), or use a **compound key** such as ad ID plus region, if order only matters within a region.

## Why it is fast

- **Sequential disk I/O.** On spinning disks, linear writes can run around 600 MB/s against roughly 100 KB/s for small random writes: over 6,000× faster. Kafka only appends and reads in order, so operations stay O(1) however much data is stored.
- **The operating system's page cache** instead of an in-process cache: no Java object overhead or garbage collection pauses, the cache survives broker restarts, and the OS handles read-ahead. Consumers that keep up are served from RAM, not disk.
- **Batching** everywhere: producers gather messages (up to, say, 64 KB or 10 ms), brokers append whole batches, consumers fetch large chunks. A few large I/Os replace millions of small ones.
- **Zero-copy**: producers, brokers and consumers share one binary format, so a broker sends file data straight from the page cache to the network socket with `sendfile`, skipping copies through user space.
- **Batch compression** (gzip, Snappy, LZ4, zstd): batches are stored and sent compressed and only decompressed by the consumer.

~~~dg-bars Disk throughput by access pattern on a spinning disk: why an append-only log is so fast.
*Sequential writes | 600000 | about 600 MB/s
Random small writes | 100 | about 100 KB/s
~~~

<div class="ex">
<div class="lab">Worked example</div>

**sizing a cluster**

Producers write 500 MB/s at peak (about 1 million messages per second of 500 bytes, after compression). With replication factor 3, the cluster writes 1.5 GB/s to disk and moves 1 GB/s between brokers for replication.

Retention is 7 days: 500 MB/s × 604,800 s ≈ 302 TB, × 3 replicas ≈ 907 TB of disk. With brokers holding 40 TB usable at 70% fill, that is about 32 brokers for storage alone.

Network: each broker takes 1.5 GB/s ÷ 32 ≈ 47 MB/s in, plus consumer reads; if 4 consumer groups each read everything, that is another 2 GB/s ÷ 32 ≈ 63 MB/s out per broker. Comfortable on 10 or 25 Gbit/s networking.

Partitions: if one consumer can process 10 MB/s, the busiest topic at 200 MB/s needs at least 20 partitions. Choose 64: room for three times the growth before you would ever need to change the count, and an even spread of 2 partitions per broker.

</div>

## Consumers

Consumers **pull**: they ask for data after their position, so a slow consumer is never overwhelmed and can catch up later. Pulling also allows big batched reads. In a **consumer group**, each partition is read by exactly one consumer in the group at a time, so the group splits the work while each partition is still processed in order. With 12 partitions, up to 12 consumers in a group can work in parallel; a 13th sits idle. This is why the partition count caps consumer parallelism.

A consumer's state is just **one offset per partition**, stored in an internal Kafka topic. That is cheap and lets a consumer **rewind** and reprocess (after fixing a bug, for instance) by resetting its offsets. If a consumer dies or a new one joins, the group **rebalances**: partitions are reassigned, and the new owner resumes from the last committed offset. Data is kept by time or size (7 days by default), independent of whether anyone has read it.

### Delivery guarantees

What happens on a crash depends on when the consumer commits its offset relative to processing.

~~~dg-matrix Delivery guarantees depend on the order of processing and committing, and on what the producer does.
Approach | Lost on crash? | Duplicated on crash? | Extra requirement
Commit, then process | yes, at most once | no | none
Process, then commit | no | yes, at least once (default) | idempotent processing
Transactions, read-process-write | no | no, exactly once within Kafka | idempotent producer, transactional offsets
~~~

- **Commit, then process**: crash in between and those messages are never processed (**at most once**).
- **Process, then commit**: crash in between and the messages are processed again by whoever takes over (**at least once**, the default). Make processing idempotent: an upsert keyed by message ID, or a "processed IDs" table checked in the same database transaction.
- **Exactly once** within Kafka: an **idempotent producer** (the broker drops duplicate retries using a producer ID and sequence numbers) plus **transactions** that write the output messages and the consumer's new offsets atomically. If the side effect is outside Kafka (sending an email, charging a card), you are back to at-least-once plus idempotency on that external system.

~~~dg-seq At least once: processing before committing means a crash causes a repeat, never a loss.
Consumer A > Broker : fetch partition 3 from offset 100
Broker -> Consumer A : messages 100 to 149
Consumer A > Database : process 100 to 149 (upsert by message ID)
Consumer A > Consumer A : crash before committing offset 150
Broker > Consumer B : rebalance: partition 3 is yours
Consumer B > Broker : fetch from last committed offset 100
Broker -> Consumer B : messages 100 to 149 again
Consumer B > Database : upserts are no-ops for 100 to 149
Consumer B > Broker : commit offset 150
~~~

## Replication

Each partition has a **leader** and followers on other brokers (replication factor 3 is common). Producers write to the leader; followers fetch from it, just like consumers. The **in-sync replicas (ISR)** are the replicas that are keeping up (within `replica.lag.time.max.ms`, 30 s by default). A message is **committed** once every ISR member has it, and only committed messages are given to consumers, so a consumer never sees a message that could later disappear in a failover.

With `acks=all` the producer waits for commitment; with `min.insync.replicas = 2` the partition refuses writes if fewer than 2 replicas are in sync, so an acknowledged write is never on just one machine. With f + 1 in-sync replicas, Kafka tolerates f failures without losing committed data. If the leader fails, the controller picks a new leader from the ISR. If every in-sync replica dies, Kafka by default waits for one to come back rather than electing an out-of-date replica; enabling **unclean leader election** trades possible data loss for uptime.

**Log compaction** is the alternative to deleting by age: Kafka keeps at least the latest value for each key and removes older ones, and a message with a null value (a **tombstone**) deletes the key. A compacted topic is then a complete, replayable snapshot of current state, which is the table and log duality in practice, and how Kafka stores consumer offsets themselves.

## Common mistakes

- **Assuming global order** across a topic and building logic that breaks when events for one entity land in different partitions.
- **Too few partitions**, capping consumer parallelism, then changing the count and breaking key order.
- **`acks=1` for important data**, so a leader crash loses acknowledged writes.
- **Non-idempotent consumers** with at-least-once delivery, causing double charges or double emails.
- **Treating Kafka as a queue with per-message acknowledgement**: a slow message blocks its whole partition behind it.

## In an interview

State the guarantee precisely: "Kafka orders messages within a partition; I will key by order ID so each order's events stay in order." Then cover partitions for scale, consumer groups for parallelism, replication with `acks=all` and `min.insync.replicas=2`, and at-least-once delivery with idempotent consumers. Size partitions from throughput per consumer.

<div class="take">
<div class="lab">Key takeaway</div>

a Kafka topic is a set of partitioned, replicated, append-only logs. Order is guaranteed only within a partition, so choose the key so that messages that must stay ordered share it, and fix the partition count early. Consumers pull and track one offset per partition, giving at-least-once delivery by default and exactly-once within Kafka with idempotent producers and transactions.

</div>
