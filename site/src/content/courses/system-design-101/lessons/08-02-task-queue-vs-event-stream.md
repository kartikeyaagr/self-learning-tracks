---
id: "08.02"
course: "system-design-101"
module: "08"
slug: "task-queue-vs-event-stream"
title: "Task queue vs event stream"
keyIdeas: ["**Task queue** (SQS, RabbitMQ, Celery): each message is a unit of work handed to one worker and deleted once acknowledged. Good for jobs.", "**Event stream / log** (Kafka, Kinesis): messages are appended to a durable, ordered, partitioned log and kept for a retention period. Many consumer groups read independently, each tracking its offset.", "Keeping messages allows replay (rebuild a view, fix a bug and reprocess), multiple independent consumers, and ordering within a partition.", "Queues offer per-message acks, delays, and easy parallelism without partition limits."]
question: "A task queue deletes messages when consumed; a stream keeps them. What does keeping them let you do?"
answer: "Keeping messages lets you replay history (reset a consumer group's offset to reprocess after a bug fix or rebuild a derived store such as a search index), add new consumers months later that read past events from the start of retention, and let several independent consumer groups read the same events at their own pace. Each consumer only tracks its own offset, so one slow or broken consumer does not affect the others, and the log doubles as an ordered audit trail for debugging."
sources: [{"title": "Hello Interview: Kafka deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/kafka", "note": "Log-based streams."}, {"title": "RabbitMQ tutorials", "url": "https://www.rabbitmq.com/tutorials", "note": "Classic task queues."}]
---

## Two shapes of messaging

"Use a queue" hides a choice between two quite different tools. Both move messages from producers to consumers, but they disagree on one basic question: what happens to a message after someone reads it?

- A **task queue** (RabbitMQ, SQS, Celery on Redis) treats a message as a **unit of work**. A worker takes it, processes it and **acknowledges** it (tells the broker "done"), and then the broker deletes it. Each message goes to exactly one worker. The queue is a to-do list: once an item is done, it's crossed off.
- An **event stream** or **log** (Kafka, Kinesis, Pulsar, Redpanda) treats a message as a **fact** appended to an ordered log. Consumers read it at their own pace and track their own position. Messages stay until a retention limit is reached, regardless of who has read them. The stream is a ledger: entries are never crossed off, only aged out.

Everything else (scaling, ordering, retries, replay) follows from that one difference.

~~~dg-compare A queue hands each message to one worker and deletes it on ack; a stream keeps the log and each consumer group tracks its own offset.
[Task queue]
Producer -> Queue -(one worker)-> Worker -(ack)-> Deleted
+ Per-message ack, retry, dead-letter
+ Add workers freely to go faster
- A message is gone once processed
- Hard to add a second reader later
[Event stream]
Producer -(append)-> Log -(own offset)-> Group A
+ Many independent readers
+ Replay history after a bug fix
+ Ordered within a partition
- Parallelism capped by partitions
- Retries and dead letters are your job
~~~

## How a task queue works

RabbitMQ is a good model to learn because it shows each building block explicitly.

### Work queues and competing consumers

Several workers subscribe to one queue. The broker hands messages out **round-robin**, so the workers compete for work and share the load (the **competing-consumers** pattern). To go faster, you add workers; nothing else changes.

Three settings make this safe:

- **Acknowledgements.** A worker sends an ack only when it has finished. If the worker's connection drops before it acks, the broker puts the message back and gives it to someone else. Acking on receipt (auto-ack) is faster but loses the message if the worker crashes mid-task.
- **Prefetch.** Without a limit, the broker pushes as many messages as it can to each worker, and one worker can hoard 10,000 while others sit idle. A prefetch of, say, 10 means each worker holds at most 10 unacked messages at a time.
- **Durability.** Queues declared durable and messages marked persistent are written to disk, so they survive a broker restart. Non-durable messages live only in memory.

SQS works the same way under different names. Receiving a message hides it for a **visibility timeout** (30 seconds by default). Deleting it is the ack. If you don't delete it in time, it reappears for another consumer. After a configured number of receives (`maxReceiveCount`), a **redrive policy** moves it to a **dead-letter queue** (DLQ), a separate queue for messages that keep failing.

### Exchanges and routing

In RabbitMQ, producers don't send to a queue directly. They send to an **exchange**, which decides which queues get a copy:

- A **fanout** exchange copies every message to every bound queue: publish/subscribe.
- A **direct** exchange routes by an exact **routing key**, such as `pdf` or `email`.
- A **topic** exchange matches patterns: a queue bound to `orders.*.eu` gets `orders.created.eu` but not `orders.created.us`. `*` matches one word, `#` matches zero or more.

So a task-queue broker can do publish/subscribe too, but each subscriber needs its own queue, created in advance. A subscriber added next month only sees messages published after its queue existed.

~~~dg-flow RabbitMQ publish/subscribe: producers send to an exchange, which copies or routes messages into one queue per consumer.
[Email] Producer -> *Exchange -(orders.#)-> Email queue -> Email worker
[Fraud] Producer -> Exchange -(orders.*.eu)-> Fraud queue -> Fraud worker
~~~

RabbitMQ also supports request/reply over queues (the reply goes to a queue named in the message, matched by a correlation ID) and **publisher confirms**, where the broker tells the producer it has safely stored a message. Recent versions add **streams**, a log-style structure, which shows the line between the two models has blurred. The models still matter more than the product names.

## How an event stream works

Kafka is the reference design for logs.

- **Brokers** are the servers. A **topic** is a named stream, such as `orders`.
- Each topic is split into **partitions**. A partition is an append-only, ordered log stored on disk and replicated to other brokers (a replication factor of 3 is typical).
- Each message has an **offset**, its position in the partition: 0, 1, 2, and so on. Offsets never change.
- A producer chooses a partition from the message's **key**: `hash(key) % partitions`. All messages with the same key (say, `order_id = 9001`) land in the same partition, so they stay in order relative to each other.
- **Order is guaranteed only within a partition.** Across partitions, there is no global order.
- Consumers belong to a **consumer group**. Inside a group, each partition is assigned to exactly one consumer, so the group shares the work. Different groups each get every message independently.
- A consumer **commits** its offset (stores "I've processed up to 4,812 in partition 3") to an internal topic. After a restart, it resumes from the committed offset.

~~~dg-layers Kafka's pieces, from cluster down to a single message.
Cluster | several brokers, one controller
Broker | a server; leads some partitions, follows others
Topic | a named stream, e.g. orders
*Partition | ordered append-only log; order guaranteed only here
Offset | a message's fixed position in its partition
Consumer group | each partition read by exactly one member
~~~

### Retention and compaction

Messages are kept for a **retention period**: 7 days by default, set with `retention.ms` or capped by size with `retention.bytes`. Old **segments** (the files a partition is split into, typically 1 GB each) are deleted whole when they age out. Reading does not delete anything.

**Log compaction** is the other option: Kafka keeps at least the latest message for each key and discards older ones. A compacted `user-profiles` topic works like a table you can rebuild from: replay it and you get every user's latest profile.

### Why logs are fast

A log is written sequentially, and disks and SSDs are very fast at sequential writes. Consumers usually read recent data that is still in the operating system's page cache. Kafka also batches messages and can send file data straight from the page cache to the network socket without copying through the application (`sendfile`, "zero copy"). As a ballpark, a single broker can handle hundreds of thousands to about a million small messages a second and hold terabytes of data. Keep individual messages small, under 1 MB (the default limit); put large blobs in object storage and send a reference.

### The partition cap on parallelism

Because a partition is read by only one consumer in a group, **a group can't usefully have more consumers than partitions**. With 12 partitions, a 13th consumer sits idle. You can add partitions later, but that changes `hash(key) % partitions` for existing keys, so new messages for key K may land in a different partition from old ones and per-key ordering breaks during the change. So choose partition counts with headroom: if one consumer handles 2,000 messages a second and you expect 30,000 at peak, you need at least 15 partitions; 24 or 32 gives room to grow.

<div class="ex">
<div class="lab">Worked example</div>

**sizing an orders topic**

Peak load is 20,000 order events a second, averaging 1 KB each: 20 MB/s of writes. With a replication factor of 3, the cluster writes 60 MB/s to disk in total. Retaining 7 days: 20 MB/s × 86,400 s × 7 ≈ 12 TB of data, or 36 TB with replicas. Spread over 6 brokers, that's 6 TB each, comfortably within a broker's disks. Each consumer of the fraud service handles 1,500 events a second, so it needs about 14 consumers; 24 partitions gives room to grow without re-partitioning.

</div>

## What retention buys you

Keeping messages after they're read sounds wasteful. It's actually the main reason to choose a stream.

- **Replay.** You can reset a consumer group's offset to an earlier point and reprocess history, for example after fixing a bug that corrupted a derived store.
- **New consumers later.** A service added next month can read from the start of retained data and build its own view, without the producer doing anything.
- **Many independent readers.** Search indexing, analytics, notifications and fraud each read the same events. The producer doesn't know they exist, and one slow reader doesn't slow the others.
- **Auditing and debugging.** You can see exactly what happened, in order, and reproduce an incident by replaying it into a test environment.

~~~dg-timeline Three consumer groups reading the same partition at their own pace; nothing is deleted when a group reads.
[Log] 09:00 | offsets 0 to 9,000 written
[Billing] 09:00 | committed 9,000, caught up
[Search] 09:00 | *committed 7,200, slow indexer
[Analytics] 09:00 | committed 3,000, hourly batch
[Search] 09:20 | committed 9,000, caught up
[Analytics] 10:00 | committed 9,400, batch run
~~~

The cost: consumers must track offsets themselves, and Kafka has no built-in per-message retry, delay or dead-letter queue. If message 4,812 fails, the consumer can't skip it and come back later without help; it must either retry in place (blocking the partition) or write the message to a separate retry topic and move on. You build those patterns yourself, or use a framework that does.

## Choosing between them

~~~dg-matrix Typical behaviour of each model. Exact features vary by product and version.
Property | Task queue | Event stream
Message after read | deleted on ack | kept until retention
Readers per message | one worker | every consumer group
Add a reader later | partial, sees only new | yes, from retained history
Replay | no | yes, reset offset
Per-message retry and DLQ | yes, built in | no, build retry topics
Ordering | partial, FIFO queues only | yes, per partition
Scale consumers | yes, add workers | partial, up to partitions
~~~

Pick a **task queue** when each message is a job to be done once: send this email, resize this image, generate this PDF. You want per-message retries, delays, priorities and a DLQ, and you never want to read the job again after it's done.

Pick a **stream** when each message is a fact other systems care about: order placed, payment captured, user updated. Several teams need it, ordering per entity matters, or you may want to rebuild a derived store from history.

Many systems use both: an `OrderPlaced` event goes on a stream, and one of its consumers turns it into a task on a queue ("send confirmation email to customer 42").

<div class="ex">
<div class="lab">Worked example</div>

**rebuilding a search index**

An `orders` topic retains 30 days. The team ships a search indexer with a bug that drops the currency field, and it runs for 3 days before anyone notices. After fixing it, they reset the indexer's consumer group to the offsets of 3 days ago and let it reprocess. Three days at 5,000 events a second is about 1.3 billion events. The indexer can process 40,000 a second while live traffic keeps adding 5,000, so it gains 35,000 a second and catches up in about 10 hours. The billing and email consumers are untouched because they track their own offsets. With a task queue, those messages would be gone, and the team would need to re-export from the database instead.

</div>

## Common mistakes

- **Using a stream as a job queue** with very slow, variable jobs. One stuck message blocks its whole partition (head-of-line blocking), and you can't scale beyond the partition count.
- **Using a queue for events several teams need.** Each new team needs a new queue wired in by the producer's owners, and nobody can replay.
- **Assuming global order in Kafka.** Order holds per partition only. Key by the entity whose order matters.
- **Too few partitions.** Easy to add later in theory, painful in practice because it reshuffles keys.
- **Committing offsets before processing.** If the consumer crashes after committing but before finishing, those messages are skipped. Commit after processing for at-least-once.

## In an interview

Say which model you're using and why in one sentence: "Order events go on a Kafka topic keyed by order ID, because three services consume them and we want replay; email sending goes through SQS with a DLQ, because each email is a job." Then mention partition count, key choice and retention, which are the follow-ups interviewers ask about.

<div class="take">
<div class="lab">Key takeaway</div>

a task queue hands each message to one worker and deletes it on ack; a stream keeps an ordered, partitioned log that many consumer groups read at their own pace. Use a queue for jobs with per-message retries, and a stream for events that several systems need or that you may want to replay.

</div>
