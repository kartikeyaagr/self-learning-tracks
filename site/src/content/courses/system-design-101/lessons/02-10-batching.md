---
id: "02.10"
course: "system-design-101"
module: "02"
slug: "batching"
title: "Batching"
keyIdeas: ["Batching groups many small operations into one larger one: multi-row inserts, bulk API calls, Kafka producer batches.", "It cuts per-operation overhead (network round trips, syscalls, disk flushes), so throughput goes up a lot.", "Cost: added latency while waiting to fill a batch, bigger blast radius on failure, and partial failure handling.", "Tuning knobs are size and time: flush when the batch reaches N items or after T milliseconds, whichever comes first (Kafka's `batch.size` and `linger.ms`).", "DataLoader batches the many individual lookups in one request tick into a single query, fixing the N+1 problem."]
question: "Batching raises throughput. What does it cost you, and which knob controls that?"
answer: "Batching costs latency: items that arrive early wait for the batch to fill or for the timer to fire, so per-item latency rises and becomes more uneven. It also adds memory use, a larger blast radius when a batch fails (or harder per-item error handling), and possible loss of an unsent batch on a crash. The knob is the maximum wait time (such as Kafka's linger.ms or DataLoader's batch schedule), usually paired with a maximum batch size so a full batch is sent immediately."
sources: [{"title": "Kafka docs: Design, efficiency", "url": "https://kafka.apache.org/documentation/#design", "note": "Why batching makes Kafka fast."}, {"title": "GraphQL DataLoader", "url": "https://github.com/graphql/dataloader", "note": "Request batching to fix N+1 calls."}]
---

## Why batching helps

Almost every operation has a **fixed cost** on top of the work itself: a network round trip, a system call, a disk seek or flush, a transaction commit, a request's headers and authentication, a lock acquisition. If you do 1,000 operations one at a time, you pay that overhead 1,000 times. Group them into one batch and you pay it once.

Write it as a formula. If each call has a fixed overhead `F` and each item costs `c`, then the time per item with batch size `b` is `F / b + c`. As `b` grows, the overhead per item shrinks towards zero and throughput approaches the limit set by the per-item work alone. Batching is one of the most effective throughput tools there is, and it shows up everywhere: database bulk inserts, Kafka producers, TCP itself (Nagle's algorithm), GPU inference, log shippers, and the `WHERE id IN (...)` query that replaces a hundred single-row lookups.

~~~dg-compare The same 1,000 messages sent one at a time and in batches of 100. The work per message is the same; the overhead is paid ten times instead of a thousand.
[One at a time]
1,000 messages -(1,000 round trips)-> Broker -> 1,000 x overhead
- ~1 s of round trips on one connection
- 1,000 small disk writes
[*Batched]
1,000 messages -(10 batches of 100)-> Broker -> 10 x overhead
+ ~20 ms of round trips
+ 10 large sequential writes
- Each message waits for its batch
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the arithmetic of amortisation**

A client sends records to a service 1 ms away (a 1 ms round trip). Processing each record takes 10 µs on the server. The client sends on one connection, one request at a time.

Batch size 1: each record costs 1 ms + 0.01 ms ≈ 1.01 ms, so about 990 records per second.

Batch size 100: each batch costs 1 ms + 100 × 0.01 ms = 2 ms, so 100 / 0.002 = 50,000 records per second, fifty times more.

Batch size 1,000: 1 ms + 10 ms = 11 ms per batch, about 91,000 records per second. The gain is flattening, because the per-item work now dominates. Past this point bigger batches buy little throughput and cost more latency and memory.

</div>

~~~dg-bars Records per second on one connection as batch size grows, with a 1 ms round trip and 10 µs of work per record. Gains are huge at first, then flatten.
Batch of 1 | 990 | ~1,000/s
Batch of 10 | 9100 | ~9,100/s
Batch of 100 | 50000 | 50,000/s
*Batch of 1,000 | 91000 | ~91,000/s
Batch of 10,000 | 99000 | ~99,000/s
~~~

## Kafka: batching all the way down

Kafka is a good case study because its design is built around two inefficiencies: **too many small I/O operations**, and **excessive byte copying**. Its fixes:

**Message sets.** The protocol is built around a "message set" abstraction that groups messages. The producer sends a batch in one network request, amortising the round trip over many messages. The broker appends the whole batch to its log in one write. Consumers fetch large, linear chunks at a time. Network packets get bigger and disk writes become large **sequential appends**, which are fast even on spinning disks (around 200 MB/s sequential, against about 2 MB/s for random 8 KB writes).

**Producer knobs.** Two settings control batching on the producer:

- `batch.size`: the maximum bytes in one batch per partition (16 KB by default). When a batch is full, it is sent immediately.
- `linger.ms`: how long the producer waits for more messages before sending a batch that is not full. Historically the default was 0 (send as soon as possible); recent versions default to a few milliseconds. A larger linger gives fuller batches and higher throughput, but each message waits longer.

**One binary format everywhere.** Producer, broker and consumer share the same message format, so batches pass from the producer's memory, through the broker's disk, to the consumer without being decoded and re-encoded.

**Zero-copy.** The naive way for a program to send a file to a socket takes four copies and two system calls: disk to the kernel's page cache, page cache to a buffer in the application, application buffer back into a kernel socket buffer, socket buffer to the network card. Kafka uses the `sendfile` system call, which tells the kernel to send data from the page cache straight to the network card. The application never touches the bytes, so serving many consumers costs little CPU.

~~~dg-compare Sending log data to a consumer. The naive path copies the bytes through the application; sendfile hands them from the page cache straight to the network card.
[Naive read and write]
Disk -> Page cache -> App buffer -> Socket buffer -> NIC
- 4 copies, 2 system calls
- CPU and memory bandwidth spent on copying
[*sendfile (zero-copy)]
Disk -> Page cache -(sendfile)-> NIC
+ No copies through the application
+ Data already cached serves many consumers
~~~

**The page cache as the cache.** Rather than keeping its own in-process cache, Kafka relies on the operating system's page cache. On a machine with 32 GB of RAM, that gives roughly 28 to 30 GB of cache with no garbage-collection pauses, and the cache survives a broker process restart.

**Batch compression.** The producer compresses a whole batch at once (gzip, Snappy, LZ4 or zstd). A batch of similar JSON events compresses far better than each event alone, because the field names and common values repeat across messages: ratios of 5 to 10 times are common for a batch, against very little for a single small message. The broker decompresses the batch only to validate it (for example, that the record count matches the header), writes it to disk still compressed, and sends it to consumers compressed; consumers decompress. Compression thus saves network, disk and page cache all at once.

## DataLoader: batching inside one request

The same idea works at the application layer. GraphQL resolvers naturally ask for data one item at a time: load a user, then each of their 10 friends, then each friend's best friend. That is the **N+1 problem**: one query for the list, then one query per item. A single GraphQL query can easily make 13 or 130 database calls.

**DataLoader** (a small library from the GraphQL ecosystem, with ports in most languages) fixes it. Every call to `loader.load(key)` returns a promise, and the loader collects all keys requested during one tick of the JavaScript event loop. At the end of the tick, it calls your **batch function once** with all the keys, so 13 queries become a few `WHERE id IN (...)` queries.

~~~dg-seq DataLoader collects the keys requested in one tick of the event loop and makes one batched call, then hands each caller its own result.
Resolver A > DataLoader : load(user 7)
Resolver B > DataLoader : load(user 9)
Resolver C > DataLoader : load(user 7)
DataLoader > DataLoader : end of tick: keys [7, 9], 7 cached once
DataLoader > Database : SELECT * FROM users WHERE id IN (7, 9)
Database -> DataLoader : 2 rows
DataLoader -> Resolver A : user 7
DataLoader -> Resolver B : user 9
DataLoader -> Resolver C : user 7 (same object)
~~~

Its rules:

- **The batch function must return results in the same order and the same length as the keys**, filling gaps with null or an error. A database returns rows in any order, so you re-order them by key.
- **It also caches by key** within its lifetime, so loading the same user twice returns the same result without a second query.
- **Create a new loader per request.** A loader shared across requests would leak one user's cached data into another user's response (a security bug), and its cache would grow forever.
- **`maxBatchSize`** caps the batch (useful when the database limits the size of `IN` lists), and **`batchScheduleFn`** can wait longer than one tick (for example a few milliseconds) to collect more keys, which adds latency.

## The cost: latency and blast radius

Batching makes the **first item wait for the last**. An item that arrives just after a batch was sent waits for the whole next window. Every batching system has a knob that trades latency for throughput: Kafka's `linger.ms`, DataLoader's schedule function, a database driver's flush interval, a log shipper's flush period. Most also have a **size cap**, so a batch is sent as soon as it is full, without waiting out the timer. "Send when 100 items **or** 10 ms, whichever comes first" is the standard shape.

~~~dg-timeline A batch with a 10 ms linger and a size cap of 5. Messages that arrive early wait longest; a full batch leaves without waiting for the timer.
0 ms | message 1 arrives, timer starts
3 ms | messages 2 and 3 arrive
10 ms | *timer fires: batch of 3 sent, message 1 waited 10 ms
11 ms | message 4 arrives, new timer starts
12 ms | messages 5 to 8 arrive: batch is full
12 ms | batch of 5 sent at once, no wait for timer
~~~

Other costs:

- **Partial failure.** If one item in a batch fails (a constraint violation in a bulk insert, one invalid record), you must decide: fail the whole batch (simple, but one bad record blocks 999 good ones), or return per-item results (more complex for both sides). Most good batch APIs return per-item status.
- **Memory.** Items wait in memory before being sent, and a slow downstream means batches pile up. Buffers must be bounded (lesson 02.11).
- **Loss on crash.** An unsent batch lives only in memory. If the process crashes, it is gone, unless you write it to disk first or the producer only acknowledges its own callers after the batch is sent.
- **Uneven latency.** Items arriving just after a batch leaves wait the longest, so batching widens the latency distribution.
- **Bigger blast radius.** A batch that hits a timeout or a poison record affects every item in it.

| Knob | Turn up | Turn down |
| --- | --- | --- |
| Max wait (linger) | Fuller batches, higher throughput | Lower per-item latency |
| Max size | Fewer round trips, better compression | Smaller failures, less memory, less per-batch latency |

<div class="ex">
<div class="lab">Worked example</div>

**analytics events from a mobile SDK**

The SDK used to send one HTTP request per tap, about 50 requests per minute per active user. With 2 million concurrently active users, that is about 1.7 million requests per second at the ingestion endpoint, each paying for TLS records, headers and an authentication check, and draining phone batteries by waking the radio constantly.

Batching every 10 seconds or every 100 events, whichever comes first, cuts that to about 6 requests per minute per user: about 200,000 requests per second, an eightfold reduction, and each gzip-compressed batch is about a fifth of its raw size.

The cost: events show up on dashboards up to 10 seconds later, and an app crash could lose the unsent batch, so the SDK appends events to a small file on disk and deletes them only after the server acknowledges the batch. Duplicate sends after a crash are removed on the server using each event's ID.

</div>

## Common mistakes

- **No size cap**, so a burst creates one enormous batch that times out.
- **No time cap**, so a quiet period leaves items waiting indefinitely.
- **All-or-nothing batches** where one bad record blocks the rest.
- **Sharing a DataLoader between requests.**
- **Batching on a latency-critical path** (a user waiting on a click) with a long linger.

## In an interview

Mention batching whenever you see many small writes: event ingestion, logging, metrics, notification fan-out, bulk database inserts. Name both knobs ("flush every 50 ms or 500 records"), say what latency that adds, and say how you handle a failed item and a crash with unsent data.

<div class="take">
<div class="lab">Key takeaway</div>

batching amortises fixed per-call costs (round trips, syscalls, commits, headers) over many items, often for tenfold or bigger throughput gains, and enables better compression and sequential I/O. You pay in latency, memory and blast radius, controlled by a maximum wait time paired with a maximum batch size.

</div>
