---
id: "18.03"
course: "system-design-101"
module: "18"
slug: "how-discord-stores-trillions-of-messages"
title: "How Discord stores trillions of messages"
keyIdeas: ["Discord stores messages in a wide-column database partitioned by (channel_id, bucket), where bucket is a fixed time window. Messages in a partition are sorted by Snowflake message ID.", "Bucketing by time keeps partitions from growing without limit in busy channels and makes \"latest messages in channel\" a read from one partition.", "On Cassandra, hot partitions (huge servers) and GC pauses caused latency problems.", "They moved to ScyllaDB (C++, no GC) and added Rust \"data services\" that coalesce identical concurrent requests for the same channel, so one hot channel doesn't flood the database.", "Result: trillions of messages, lower and more stable latency, fewer nodes."]
question: "Why partition messages by (channel_id, time bucket)?"
answer: "Every query reads messages from one channel in time order, so channel_id in the partition key keeps a channel's messages together on one replica set and contiguous on disk, making \"latest messages in this channel\" a single sequential read, with message_id (a time-sortable Snowflake) as the clustering key giving newest-first order. But a busy channel's partition would grow without bound, and Cassandra partitions over about 100 MB hurt compaction and garbage collection and cannot be spread across nodes. Adding a time bucket (about 10 days, computed from the Snowflake timestamp, so no lookup is needed) caps each partition's size while keeping messages in order; older history is read bucket by bucket. The trade-off is that live reads for hot channels concentrate on the newest bucket."
sources: [{"title": "Discord: How Discord stores trillions of messages", "url": "https://discord.com/blog/how-discord-stores-trillions-of-messages", "note": "Cassandra to ScyllaDB, data services."}, {"title": "Discord: How Discord stores billions of messages", "url": "https://discord.com/blog/how-discord-stores-billions-of-messages", "note": "The earlier Cassandra design."}, {"title": "Hello Interview: Discord messages on ScyllaDB", "url": "https://www.hellointerview.com/learn/system-design/in-the-wild/discord-messages-scylladb", "note": "Annotated walkthrough."}]
tags: ["Case"]
tools: ["Discord", "Cassandra"]
---

## Why this case is worth studying

Discord is a chat platform where messages are kept forever. A user can scroll back years in a channel, search old messages, or jump to a message linked from elsewhere. Storing billions, then trillions, of messages and serving them with low, predictable latency took Discord through three databases in seven years: MongoDB, Cassandra and ScyllaDB. Each move teaches something general about data modelling for wide-column stores, partition sizing, tombstones, hot partitions and migrations.

The central modelling decision, partitioning messages by `(channel_id, time bucket)`, is what this lesson's question asks about. It only makes sense once you see the access pattern and how wide-column databases lay out data.

~~~dg-timeline Discord's message storage over seven years.
2015 | MongoDB replica set, one index
Nov 2015 | About 100M messages, index outgrows RAM
2016 to 2017 | Move to Cassandra, (channel, bucket) key
2017 | 12 nodes, about 1 TB each, 120M messages a day
Early 2022 | 177 Cassandra nodes, hot partitions, GC pauses
May 2022 | *Migration to 72 ScyllaDB nodes
Dec 2022 | World Cup final spikes handled smoothly
~~~

## From MongoDB to Cassandra

Messages first lived in a single MongoDB replica set with a compound index on `(channel_id, created_at)`. Around November 2015, at about **100 million** stored messages, the data and index no longer fitted in RAM and latency became unpredictable. Every read that missed memory went to disk.

The access pattern made this worse:

- Reads and writes were roughly **50/50**, unusual for a consumer app.
- Reads were very **random**. Small private servers (a group of friends) post rarely, and when someone opens the channel they read history that has long left any cache. Big public servers have heavy traffic but read mostly the last hour or so.
- Everything is **"messages in this channel, newest first"**. There are almost no queries across channels.

Discord's requirements for a replacement: linear scaling without manual resharding, automatic failover, low maintenance, predictable latency (alerting at a p95 of 80 ms) **without** relying on a cache, and open source. They chose **Cassandra**.

## How a wide-column store lays out data

Cassandra (and ScyllaDB, which is compatible with it) stores rows grouped by a **partition key**. The partition key is hashed to decide which nodes hold the data (with replication factor 3, three nodes), and all rows with the same partition key are stored together, contiguously, in each on-disk file. Within a partition, rows are kept sorted by a **clustering key**.

So the cost of a query depends heavily on whether it reads **one partition** (one hash, one replica set, one sequential range on disk) or many. A table design starts from the queries.

~~~dg-tree How one channel's messages are laid out: partitions by channel and bucket, rows sorted by message ID inside each.
messages table
  Partition (42, bucket 1630) | newest 10 days
    msg 1129...901
    msg 1129...877
  Partition (42, bucket 1629) | 10 days earlier
    msg 1121...350
  *Partition (77, bucket 1630) | another channel
~~~

## Why (channel_id, bucket)

### Step 1: partition by channel

Every query is "messages in this channel". So one channel's messages should live together and be read in one go from one set of replicas. `channel_id` is the natural partition key.

### Step 2: cluster by message ID

Within the partition, messages are sorted by `message_id`, a **Snowflake** ID (lesson 04.10): 64 bits made of a millisecond timestamp, a worker ID and a sequence. It is unique and sorts by time, so "the latest 50 messages" is the first 50 rows of the partition in descending order, and "messages before X" is a range scan from X. `created_at` could not be the clustering key, because two messages can share a timestamp and the second would overwrite the first.

### Step 3: the partition size problem

When Discord imported its existing data, Cassandra logged warnings about partitions over **100 MB**. Cassandra allows partitions up to about 2 GB, but large partitions cause real trouble:

- **Compaction and repair** process a partition as a unit, so a huge one causes long garbage-collection pauses on the Java heap.
- A partition lives on exactly one replica set, so it **cannot be spread** across the cluster no matter how many nodes you add.
- A busy channel's partition would **grow forever**, so a design that is fine today fails in two years.

### Step 4: add a time bucket

So the primary key became `((channel_id, bucket), message_id)`. The double parentheses mark the composite partition key. Each bucket covers about **10 days**, a size chosen so that even the busiest channels stay under 100 MB per partition. The bucket number is computed from the Snowflake ID's timestamp (`bucket = timestamp_ms / (10 x 86,400,000)`), so no lookup table is needed: given a message ID or a time, you know its partition.

~~~dg-compare Three possible partition keys for messages, and what each costs.
[channel_id only]
Channel -> One growing partition
+ One read for any history
- Busy channels grow without bound
[*channel_id + 10-day bucket]
Channel + bucket -> Bounded partitions
+ Partitions stay under 100 MB
+ Recent history still one read
- Hot channel hits one bucket
[message_id only]
Each message -> Own partition
+ Perfect spread
- Reading history = thousands of lookups
~~~

<div class="ex">
<div class="lab">Worked example</div>

**sizing the bucket**

A busy channel averages 1 message per second around the clock: 86,400 messages a day. At about 100 bytes stored per message, that is 8.6 MB a day.

With no bucket, its partition passes 100 MB in under two weeks and reaches about 3 GB in a year, past Cassandra's practical limit. With a 10-day bucket, each partition holds about 86 MB, just under the 100 MB warning line. That is the logic behind the bucket size: pick it from the busiest real channels so they stay under the limit.

A channel ten times busier would produce about 860 MB per bucket, which is why extreme outliers later needed more care.

A typical small server's channel with 50 messages a day produces 50 KB per 10-day bucket: tiny partitions, read in one go.

</div>

### Reading history across buckets

To load recent messages, the service generates bucket numbers from the current time backwards to the channel's creation, and queries them newest first until it has enough messages:

~~~dg-seq Loading the latest 50 messages: busy channels are served by one bucket, quiet ones may need a few.
App > Messages API : latest 50 in channel 42
Messages API > Cassandra : partition (42, 1630) LIMIT 50
Cassandra -> Messages API : 12 messages
Messages API > Cassandra : partition (42, 1629) LIMIT 38
Cassandra -> Messages API : 38 messages
Messages API -> App : 50 messages, newest first
~~~

Busy channels fill the page from the first bucket. Quiet channels may need several queries, which is fine because their partitions are tiny. Later, Discord tracked empty buckets per channel so it could skip them.

### The trade-off

`channel_id` keeps a channel together for efficient reads; the bucket **caps partition size**; `message_id` gives newest-first order without sorting. The cost is that all live traffic for a popular channel hits the **newest bucket**, which lives on the same three replicas. That concentrated load came back to bite them years later.

## Lessons from running Cassandra

- **Upserts and missing columns.** Cassandra resolves conflicts per column with last write wins. When one user edited a message at the same moment another deleted it, the result was a row containing only the primary key and the edited text, with no author or channel. Fix: treat rows missing a required column as corrupt and delete them on read.
- **Nulls are tombstones.** Writing `null` to a column creates a **tombstone** (a deletion marker), just like a delete. Messages had 16 columns but typically set 4, so each insert wrote about 12 needless tombstones. Fix: write only non-null fields.
- **The tombstone surprise.** Six months in, Cassandra started having 10-second garbage-collection pauses. A public channel had had millions of messages deleted through the API, leaving one visible message. Loading it meant scanning millions of tombstones to find that one row. Fixes: keep tombstones for 2 days instead of the default 10 (safe because repairs ran nightly, so no replica would miss a delete for longer), and skip empty buckets.

At that point the cluster had 12 nodes, replication factor 3, nearly 1 TB per node, and took over 120 million messages a day.

## Trillions: hot partitions and the move to ScyllaDB

By early 2022 the cluster had grown to **177 nodes** storing trillions of messages, and it was painful:

- **Hot partitions.** A huge server's busy channel sends heavy concurrent reads to one `(channel, bucket)` partition. Because reads use quorum consistency, an overloaded node also slows unrelated queries that touch it, so one popular announcement could degrade latency across the cluster.
- **Reads cost more than writes** in Cassandra's LSM design (memtable plus several SSTables per read, lesson 07.11), and **compaction backlogs** made reads slower still.
- **Java garbage-collection pauses** caused latency spikes and frequent manual restarts.
- p99 latency for fetching history was **40 to 125 ms**, and inserts 5 to 70 ms.

### Data services: fixing the traffic problem

Discord put **data services written in Rust** between the API and the database:

- **Request coalescing**: if many users ask for the same rows at the same time, only one database query runs and every waiting request receives its result. This is not a cache: it never serves stale data, it only deduplicates concurrent identical requests.
- **Consistent hash routing on channel ID**: all requests for a channel reach the same data service instance, which is what makes coalescing effective across a fleet of instances.

~~~dg-flow Coalescing: a thousand identical reads become one database query.
[Route] 1,000 clients load channel 42 -(hash on channel_id)-> *Same data service
[Coalesce] Same data service -(one in-flight query)-> Partition (42, newest) -(one result)-> All 1,000 requests
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what coalescing saves**

A large server posts an announcement and 50,000 online members open the channel within 2 seconds. Without coalescing, that is 50,000 reads of the same partition, about 25,000 per second, against three replicas.

A single read takes about 5 ms. With coalescing, each data service instance runs at most one query for that partition at a time, so roughly 200 queries per second (1 / 5 ms) reach the database instead of 25,000, with every caller receiving the same fresh rows.

</div>

### ScyllaDB: fixing the storage engine problem

**ScyllaDB** is a Cassandra-compatible database written in C++, with no garbage collector and a shard-per-core architecture. Discord replaced Cassandra with it. It wrote its own migrator in Rust, estimated to take **9 days** instead of the 3 months a generic tool predicted, peaking at **3.2 million messages per second**. It compared reads from both databases before switching (shadow reads, lesson 14.06), and cut over in May 2022.

| | Cassandra | ScyllaDB |
| --- | --- | --- |
| Nodes | 177 | 72 (9 TB each) |
| p99 history read | 40 to 125 ms | about 15 ms |
| p99 insert | 5 to 70 ms | about 5 ms |

During the 2022 World Cup final, message traffic spiked at every goal and the system handled it without trouble.

## The general lessons

- **Design partitions from queries.** One query, one partition.
- **Bound every partition.** Anything keyed only by an entity that grows forever eventually breaks.
- **Separate storage-engine problems from traffic problems.** Garbage collection and compaction were fixed by ScyllaDB; many identical concurrent reads were fixed by coalescing. A faster database alone does not stop a hot partition.
- **Know your database's write semantics**: per-column last write wins and null-as-tombstone surprised an experienced team.

## In an interview

For any chat or timeline store, propose `((conversation_id, time_bucket), message_id)` with a Snowflake ID, explain the bucket as a partition size cap, and mention the hot newest bucket and request coalescing before you are asked.

<div class="take">
<div class="lab">Key takeaway</div>

Discord partitions messages by (channel_id, time bucket) so a channel's recent messages sit together for fast in-order reads, while the bucket caps partition size; Snowflake message IDs order rows within it and give the bucket without a lookup. The cost is that hot channels concentrate reads on the newest bucket, which Discord handled with Rust data services that route by channel and coalesce identical requests, plus a move from 177 Cassandra nodes to 72 ScyllaDB nodes.

</div>
