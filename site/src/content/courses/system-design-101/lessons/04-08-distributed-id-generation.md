---
id: "04.08"
course: "system-design-101"
module: "04"
slug: "distributed-id-generation"
title: "Distributed ID generation"
keyIdeas: ["Single-database auto-increment IDs break when you shard: each shard would generate duplicates, and a central sequence becomes a bottleneck.", "Requirements to weigh: uniqueness, size (64-bit vs 128-bit), rough time ordering (good for B-tree inserts and sorting), no coordination, not guessable.", "Options: UUIDs (no coordination, 128-bit), ticket servers (central DB sequences), Snowflake-style (timestamp + machine ID + sequence, 64-bit, sortable), ID range allocation per server.", "Instagram generated IDs inside Postgres: 41 bits of milliseconds, 13 bits shard ID, 10 bits per-shard sequence."]
question: "Why are auto-increment IDs a problem once you shard?"
answer: "Each shard has its own auto-increment counter starting at 1, so different shards hand out the same numbers, and IDs collide as soon as rows are referenced globally, moved between shards during rebalancing or merged in analytics. Fixing that with a single central counter puts a network round trip on every insert and makes one database a throughput bottleneck and single point of failure. Per-shard counters also lose any global creation order (ID 5,000 on one shard may be newer than ID 7,000 on another) and an ID no longer tells you which shard holds the row. The usual fix is Snowflake-style 64-bit IDs generated locally from a timestamp, a machine or shard number, and a sequence."
sources: [{"title": "ByteByteGo: Design a Unique ID Generator", "url": "https://bytebytego.com/courses/system-design-interview/design-a-unique-id-generator-in-distributed-systems", "note": "UUID, ticket server, Snowflake compared."}, {"title": "Instagram: Sharding and IDs at Instagram", "url": "https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c", "note": "IDs generated inside Postgres."}]
---

## The problem

On a single database, an auto-increment primary key is close to perfect. It is **unique** (the database's counter never repeats), **compact** (an 8-byte integer), **ordered by creation** (row 1,001 was created after row 1,000), and **free** (no extra service, no extra round trip).

Once data is **sharded** across many databases (lesson 04.05), each shard has its own counter, and they all start at 1. Shard 3 creates tweet 5,000 and so does shard 9. Several things break at once:

- **Uniqueness.** An ID no longer identifies one row. URLs like `/tweet/5000`, cache keys, foreign references from other tables and messages in queues all become ambiguous.
- **Moving data.** Rebalancing a user from shard 3 to shard 12 copies rows whose IDs may already exist on shard 12.
- **Merging data.** An analytics warehouse that combines all shards sees duplicate keys.
- **Ordering.** There is no global creation order: shard 9's ID 5,000 may be hours newer than shard 3's ID 7,000.
- **Location.** An ID tells you nothing about where the row lives, so you need a separate lookup to find it.

Twitter hit exactly this in 2010. Moving tweets from a single MySQL setup to Cassandra and sharded MySQL meant losing automatic IDs. The service it built, **Snowflake**, became the template most large systems still follow.

## What we want from IDs

Twitter's requirements for the new IDs are a good checklist for any system:

- **Unique** across all machines, forever, including across data centres.
- **Fits in 64 bits**, so it works as a compact integer key everywhere. A larger ID would have meant changing every system, client and database column that stored tweet IDs as 64-bit integers.
- **Roughly time-ordered**, so that sorting by ID approximates sorting by creation time. Timelines are sorted newest first, and if IDs follow time, "tweets newer than ID X" is a simple range query.
- **High throughput**: at least 10,000 IDs per second per process, with headroom for spikes.
- **High availability and low latency**: ID generation is on the critical path of every write, so it must never block or be a single point of failure.

## The options

~~~dg-matrix The main ways to generate IDs across shards, against the requirements above.
Option | Unique | 64 bits | Time-ordered | No per-ID network hop | No single point of failure
Central auto-increment | yes | yes | yes | no | no
Multi-master with step | yes | yes | no | no | yes
UUID v4 | yes | no | no | yes | yes
Coordination service | yes | yes | yes | no | partial, a quorum
*Snowflake style | yes | yes | yes, roughly | yes | yes
~~~

**Central auto-increment.** One database hands out every ID. It is simple and strictly ordered, but every insert anywhere needs a round trip to it, it caps throughput at what one machine can do, and when it is down, nothing can be written anywhere.

**Multi-master auto-increment with a step.** N databases each increment by N with different offsets: one issues 1, 3, 5..., the other 2, 4, 6.... Flickr's ticket servers (lesson 03.09) are this idea with two dedicated servers, and block allocation makes it fast. But IDs from different servers are not time-ordered (one server can race ahead of the other), and changing N later is painful. Twitter rejected it mainly for the lack of ordering.

**UUIDs.** A 128-bit identifier generated anywhere with no coordination; the common version 4 is 122 random bits. Collisions are so unlikely they can be ignored. But it is twice the size of a 64-bit integer, it is not time-ordered, and random values scatter inserts across B-tree indexes (lesson 04.09). Twitter rejected UUIDs as too big.

**A coordination service.** ZooKeeper or etcd can hand out sequential numbers through a consensus protocol. The IDs are strictly ordered, but every ID costs a network round trip to a quorum of servers (milliseconds), throughput is limited, and the service is a dependency on every write. Twitter rejected it for performance and availability.

~~~dg-seq Asking a coordination service for every ID puts a consensus round trip on every write; a Snowflake-style generator answers locally.
App > Coordinator : next ID please
Coordinator > Quorum : agree on sequence number
Quorum -> Coordinator : agreed
Coordinator -> App : 1,000,042, about 2 ms later
App > Local generator : next ID please
Local generator -> App : time + machine + sequence, about 1 µs
~~~

**Snowflake style.** Each generator builds IDs locally from three parts: the **current time** in milliseconds, its own **machine number**, and a **sequence** counter that resets every millisecond. Two generators can never produce the same ID because their machine numbers differ; one generator never repeats because the time or the sequence always moves forward. There is no network call per ID, the result fits in 64 bits, and because time is in the most significant bits, IDs sort roughly by creation time. This is what Twitter built. Lesson 04.10 breaks down the bits, and lesson 04.11 covers the hard part, which is clocks.

The price of Snowflake-style IDs is that each generator needs a **unique machine number**, assigned and kept unique (usually through configuration or a coordination service at startup, not per ID), and its **clock must not go backwards**.

## Instagram's variation

Instagram needed IDs for photos sharded across thousands of logical Postgres shards, and did not want to run a separate ID service. It generated Snowflake-like 64-bit IDs **inside Postgres itself**, with a stored function used as each table's default value:

- **41 bits**: milliseconds since a custom epoch (enough for about 69 years).
- **13 bits**: the **logical shard ID** (up to 8,192 shards).
- **10 bits**: a per-shard sequence, taken modulo 1,024, so each shard can create 1,024 IDs per millisecond.

~~~dg-tree The bits of an Instagram-style ID: time first, so IDs sort by creation, then the shard, then a per-shard sequence.
64-bit photo ID
  *41 bits | milliseconds since custom epoch
  13 bits | logical shard ID, 0 to 8,191
  10 bits | sequence mod 1,024, per shard per ms
~~~

Embedding the shard ID is the clever part: **any ID tells you which shard holds the row**. Given a photo ID, the service shifts out the shard bits and queries that shard directly, with no directory lookup. Because generation happens in the same database transaction as the insert, there is also no extra round trip at all.

## Why "roughly" ordered is enough

Strict global ordering requires a single point that decides the order, which is exactly the bottleneck we are trying to remove. Snowflake IDs are instead **k-sorted**: IDs generated within a small window of each other (about a second, given clock differences between machines) may be out of order relative to each other, but IDs generated further apart always sort correctly.

~~~dg-timeline Two generators with clocks 3 ms apart. IDs created within a few ms of each other can sort out of order; IDs further apart never do.
[Machine A, clock fast by 3 ms] real 10:00:00.000 | issues ID with time .003
[Machine B, accurate] real 10:00:00.001 | *issues ID with time .001, sorts before A's
[Machine B, accurate] real 10:00:00.010 | issues ID with time .010
[Machine A, clock fast by 3 ms] real 10:00:01.000 | issues ID with time 1.003, sorts after all above
~~~

That is good enough for nearly every purpose:

- **Timelines**: a few milliseconds of disorder between tweets from different people is invisible to users.
- **Index locality**: new IDs are always near the "end" of the B-tree, so inserts touch a small, hot, cached set of pages (lesson 04.09).
- **Pagination**: "IDs less than X" pages through history by time (lesson 04.12).

What it is not good enough for: anything that needs a guaranteed order of events, such as "this payment happened before that refund". Use a sequence within one shard or a log for that.

<div class="ex">
<div class="lab">Worked example</div>

**putting the shard in the ID**

A photo app has 2,048 logical shards. Each photo ID is laid out as 41 bits of milliseconds, 11 bits of shard, and 12 bits of sequence. With 12 sequence bits, each shard can create 4,096 IDs per millisecond.

To find a photo, the service computes `shard = (id >> 12) & 0x7FF`. For an ID whose middle 11 bits hold 731, the request goes straight to logical shard 731, which the shard map places on host 23. No directory service, no cache lookup.

IDs still sort roughly by upload time, so "newest photos by this user" is an index range scan on `(user_id, id DESC)`, and the ID itself doubles as a creation timestamp: `(id >> 23) + epoch` gives the upload time to the millisecond.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**why the central counter fails at scale**

A social app creates 30,000 objects per second at peak (posts, comments, likes, messages) across 64 shards.

With a central auto-increment database, each insert first makes a 1 ms round trip to it. That adds 1 ms to every write, and the central database must sustain 30,000 durable commits per second on its own, which needs group commit and a fast log disk. If it fails over (30 seconds), every write in the whole system fails for 30 seconds.

With Snowflake-style generators running in each app server's process, each ID takes about a microsecond, the 64 shards are independent, and losing any one machine affects only that machine's writes.

</div>

## Common mistakes

- **Keeping auto-increment after sharding** and discovering collisions during the first data move.
- **Choosing UUIDv4 by default** without considering index performance and size (lesson 04.09).
- **Assuming time-ordered IDs are strictly ordered** across machines.
- **Assigning machine IDs by hand** and accidentally running two generators with the same one, which produces duplicates.
- **Exposing sequential IDs publicly** when they leak business volume (competitors can count your orders); consider an opaque public ID as well.

## In an interview

When you shard, say how IDs are generated and why: "64-bit Snowflake-style IDs, time in the top bits for ordering and index locality, a worker ID per generator, a per-millisecond sequence; no network call per ID". Optionally put the shard number in the ID so routing needs no lookup, and mention that clocks must not run backwards.

<div class="take">
<div class="lab">Key takeaway</div>

sharding breaks auto-increment because each shard counts independently, so IDs collide, lose their global order and say nothing about where a row lives. The common answer is 64-bit, roughly time-ordered IDs built locally from the time, a machine or shard number, and a per-millisecond sequence.

</div>
