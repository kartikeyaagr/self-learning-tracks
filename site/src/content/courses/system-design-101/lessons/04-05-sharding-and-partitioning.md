---
id: "04.05"
course: "system-design-101"
module: "04"
slug: "sharding-and-partitioning"
title: "Sharding and partitioning"
keyIdeas: ["Sharding splits data across multiple databases by a shard key, so each holds a subset and handles a fraction of the load.", "**Range sharding**: key ranges map to shards. Supports range scans; risks hotspots (all new timestamps hit one shard).", "**Hash sharding**: hash the key. Even spread; range queries must hit all shards.", "**Directory-based**: a lookup table maps keys to shards. Flexible but the directory must be fast and available.", "Choose a key that spreads load evenly and keeps data you query together on one shard. For chat messages, `conversation_id` keeps a conversation together but makes \"all messages by user X\" a scatter-gather query.", "Resharding is painful. Use many logical shards mapped onto fewer physical machines so you can move them later."]
question: "Pick a shard key for a chat app's messages table. What query does it make hard?"
answer: "Shard the messages table by conversation_id (hashed, with many logical shards). One conversation's messages then live together on one shard, ordered by time, so loading or paginating a chat, the most common query, touches a single shard, and with hundreds of millions of conversations the load spreads evenly (very large group chats can be bucketed by month). It makes queries across conversations hard: \"all messages sent by user X\", account export, moderation by sender and global search must fan out to every shard. Serve those from secondary stores keyed differently, such as a messages_by_user table sharded by sender_id and a search index, filled asynchronously from message events."
sources: [{"title": "Hello Interview: Sharding", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/sharding", "note": "Shard keys, range vs hash."}, {"title": "System Design Primer: Sharding", "url": "https://github.com/donnemartin/system-design-primer#sharding", "note": "Pros and cons."}]
---

## Partitioning versus sharding

The two words are often used loosely, but they mean different things.

- **Partitioning** splits one big table into pieces **within one database**. Postgres can store an `orders` table as one partition per month. A query for March only reads the March partition (this is called **partition pruning**), and dropping a year of old data is dropping 12 partitions instead of deleting millions of rows. It makes a large table more manageable, but it does not add any hardware.
- **Sharding** splits data across **separate machines**, each holding a subset of the rows and taking a subset of the load. Each piece is a **shard**. This is the one that adds capacity: more disks, more memory, more CPUs, more write throughput.

In distributed databases such as Cassandra and DynamoDB, the pieces are called partitions even though they are on different machines, which is where the confusion comes from. In this lesson, "sharding" means spreading data across machines.

## When to shard

Shard when one machine cannot keep up with **storage**, **write throughput** or **read throughput**, even after query tuning, caching, a bigger machine and read replicas (lesson 03.10). Name the bottleneck before you shard; sharding to fix a missing index is a very expensive mistake.

Some reference numbers. 500 million users at 5 KB each is about 2.5 TB, which one well-provisioned Postgres server handles. Ten times that growth, or a sustained 50,000 writes per second at peak, probably does not. Other signs: backups and restores taking most of a day, index builds taking many hours, or vacuum unable to keep up (lesson 03.07).

## How a request finds its shard

Every sharded system needs a **routing** step: given a key, which shard holds it? The routing can live in a client library, in a proxy in front of the databases (Vitess's VTGate, a MongoDB `mongos` router), or inside the database itself (Cassandra and DynamoDB nodes route for you).

~~~dg-flow A request carries the shard key, and the router maps it to the one database that owns it.
[Route] Request user_id 9123 -> Router -(hash to logical shard 211)-> *Shard map: host 14
[Answer] *Host 14 -(query one shard)-> Result
~~~

## Choosing a shard key

The **shard key** (also called the partition key) is the column whose value decides which shard a row lives on. It is the most important decision in a sharded design, because changing it later means moving every row. A good key has three properties:

- **High cardinality**: many distinct values. A boolean key gives you at most two shards; a country key gives you about 200, very unevenly sized.
- **Even distribution** of both data and traffic. Sharding by country fails if 40% of users are in one country. Sharding by creation time sends **all** new writes to the newest shard, so one machine does all the work while the rest sit idle.
- **Alignment with queries**: the most common queries should include the shard key, so they hit exactly one shard. Shard by `user_id`, and "get user profile", "get user's orders" and "get user's settings" each touch one shard.

A useful trick is to **co-locate** related tables by sharding them on the same key. If `users`, `orders` and `addresses` are all sharded by `user_id`, a user's orders join with their addresses inside one shard, and a transaction that touches all three stays on one machine. Notion did exactly this with workspace ID (lesson 03.10).

## Strategies

There are three ways to map key values to shards.

**Range-based**: each shard owns a contiguous range of key values, such as user IDs 1 to 1 million on shard A and 1 million to 2 million on shard B. Range scans over the key are efficient because neighbouring keys live together. The danger is that new data clusters on one shard if keys increase over time. It works well for multi-tenant systems where each tenant is a range, and it is how HBase, Bigtable and MongoDB's default ranged sharding work.

**Hash-based** (the usual default): compute `hash(key)` and use it to pick the shard. A good hash spreads keys evenly, so load is even even if the keys themselves are sequential. The cost: neighbouring keys end up on different shards, so range queries over the key must ask every shard. The naive `hash(key) % N` moves almost every key when N changes, so real systems use **consistent hashing** (lesson 04.07) or a large, fixed number of **logical shards** mapped to fewer physical machines.

**Directory-based**: a lookup service stores, for each key or group of keys, which shard holds it. It is the most flexible, because you can move one heavy customer to its own dedicated shard, but every request needs the lookup (usually cached), and the directory becomes critical infrastructure that must never be wrong or down.

~~~dg-compare Three ways to map keys to shards, for the same user_id key.
[Range]
user_id 1 to 1M -> Shard A
user_id 1M to 2M -> Shard B
+ Range scans on one shard
- New users all land on the newest shard
[*Hash]
hash(user_id) -> Logical shard -> Host
+ Even spread, even for sequential IDs
- Range queries hit every shard
[Directory]
Lookup table -> Shard per key or tenant
+ Move any tenant anywhere
- Extra lookup on every request
- Directory must never fail
~~~

### Logical shards: plan for growth

Splitting a shard later is hard; moving a whole shard is easy (copy it, catch up, switch). So most systems create many more **logical shards** than they have machines, and place several logical shards on each machine. With 1,024 logical shards on 16 hosts, each host holds 64. Growing to 32 hosts means moving half the logical shards from each host to a new one, with no change to which logical shard any key maps to.

~~~dg-tree Logical shards decouple the key mapping from the hardware: keys map to one of 1,024 logical shards, and logical shards map to hosts.
hash(user_id) mod 1024
  Host 1 | logical shards 0 to 63
    shard 0
    shard 1
    shard 63
  Host 2 | logical shards 64 to 127
  *Host 16 | logical shards 960 to 1023
    move shards 992 to 1023 to new Host 17 when growing
~~~

## What sharding costs

- **Cross-shard queries.** Anything not keyed by the shard key must ask every shard and merge the answers, a pattern called **scatter-gather**. A global "top 10 posts today" across 64 shards is 64 queries plus a merge, and its latency is set by the slowest shard. Handle these with caching, precomputation, a denormalised copy keyed differently (a secondary index table or a search index), or by accepting that rare admin queries are slow.
- **Cross-shard transactions.** A single database transaction cannot span shards. Two-phase commit is slow and blocks if the coordinator fails. Prefer data models where each transaction stays within one shard; otherwise use sagas with compensating steps (lesson 08.07).
- **Global constraints.** A unique constraint only holds within one shard. "Usernames must be unique" across shards needs a separate table sharded by username, or a lookup service.
- **Hot shards.** One celebrity can generate a thousand times normal traffic on one key (lesson 04.06).
- **Resharding.** Moving data while serving traffic needs double writes, backfills and verification (lesson 03.10), and application logic gets more complex everywhere.

~~~dg-seq A scatter-gather query: a request without the shard key asks every shard and waits for the slowest.
App > Shard 1 : top posts today, LIMIT 10
App > Shard 2 : top posts today, LIMIT 10
App > Shard 64 : top posts today, LIMIT 10
Shard 1 -> App : 10 rows, 8 ms
Shard 2 -> App : 10 rows, 9 ms
Shard 64 -> App : 10 rows, 140 ms, busy
App > App : merge 640 rows, keep top 10
~~~

## Where the databases stand

- **Cassandra** hashes partition keys onto a ring with virtual nodes (lesson 04.07).
- **DynamoDB** hashes partition keys and splits partitions automatically as they grow in size or traffic.
- **MongoDB** shards by key ranges (optionally on a hashed key), with a balancer that moves chunks between shards.
- **Vitess** (MySQL) and **Citus** (Postgres) add sharding to relational databases, with a router that sends each query to the right shard.
- **Spanner and CockroachDB** split data into ranges automatically and move them around, while keeping SQL and transactions.

<div class="ex">
<div class="lab">Worked example</div>

**choosing a key for a chat app's messages table**

Candidates: `message_id`, `sender_id`, `conversation_id`, `created_at`.

`created_at` sends every new message to the shard holding "now": one hot shard. `message_id` (hashed) spreads writes evenly, but loading one conversation's history must ask every shard. `sender_id` keeps a user's sent messages together, but a conversation's messages come from several senders, so loading a chat still touches several shards.

`conversation_id` wins. Loading or paging a conversation, by far the most common query, hits one shard, and messages are ordered by time within it (in Cassandra, as a clustering column; lesson 04.03). With 500 million conversations, distribution is even except for a few huge group chats, which are bucketed by month.

What it makes hard: "all messages sent by user X across all conversations" (for account export or moderation) and global search. Both are served from secondary stores keyed differently: a `messages_by_user` table sharded by `sender_id`, and a search index, both filled asynchronously from message events.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**when does this app need to shard?**

200 million messages a day at about 300 bytes each, plus indexes, is about 90 GB a day, or 33 TB a year. Writes average 2,300 per second, peaking near 10,000.

The write rate alone would fit on one strong primary, but storage would not: a year of data far exceeds what one server should hold and back up. With 1,024 logical shards on 32 hosts, each host stores about 1 TB per year and takes about 300 writes per second at peak. Growing to 64 hosts later moves 16 logical shards per host, with no key remapping.

</div>

## Common mistakes

- **Sharding by time**, creating a single hot shard for all new writes.
- **A low-cardinality key** such as country or status.
- **Ignoring which queries become scatter-gather** until production.
- **As many shards as machines**, so growth means splitting shards instead of moving them.
- **Assuming unique constraints and transactions still work globally.**

## In an interview

Propose a shard key from the main access pattern, choose hash-based distribution with consistent hashing or logical shards, then say which queries the key makes hard and how you will serve them (secondary index tables, search, precomputation). Mention how you will grow, for example "start with 1,024 logical shards on 16 hosts".

<div class="take">
<div class="lab">Key takeaway</div>

partitioning splits a table inside one database; sharding spreads data across machines. Pick a shard key with many values, even load, and alignment with your most common query, and use many logical shards so growth means moving shards, not splitting them. Every key makes some queries cross-shard: name them and plan how to serve them.

</div>
