---
id: "04.06"
course: "system-design-101"
module: "04"
slug: "hot-partitions"
title: "Hot partitions"
keyIdeas: ["A hot partition is a shard receiving far more traffic than others: a celebrity's account, a viral post, today's date as a key.", "One hot partition caps the throughput of the whole system, no matter how many shards you add.", "Fixes: add a random suffix to hot keys (`post_123#0..9`) and spread writes, then aggregate on read; cache hot reads; split hot partitions; choose a higher-cardinality key.", "For counters, write sharded sub-counters and sum them, or buffer increments and flush in batches.", "Detect them by monitoring per-partition throughput, not just totals."]
question: "A celebrity's post gets 1M likes a minute on one key. How do you spread it?"
answer: "One key cannot exceed one partition's limit (for example about 1,000 writes per second per DynamoDB partition), and 1 million likes a minute is about 16,700 per second. Split the counter into N shards (for example likes:post123:0 to likes:post123:63), send each like to a random shard, and sum the shards on read (cached for a couple of seconds) or roll them up periodically. Better still, buffer and batch: publish like events to a stream partitioned by post and random suffix, aggregate them for a second, and apply one increment per counter shard, so one write carries thousands of likes. Store who liked what separately under an evenly distributed key like (user_id, post_id), and show an approximate, slightly stale total."
sources: [{"title": "AWS DynamoDB: Designing partition keys", "url": "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-partition-key-uniform-load.html", "note": "Spreading load evenly."}, {"title": "AWS DynamoDB: Write sharding", "url": "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-partition-key-sharding.html", "note": "Suffixing hot keys."}]
---

## What a hot partition is

A partitioned database (lesson 04.05) splits data by key and gives each partition a **share** of the system's capacity: a slice of one machine's CPU, disk and network. The design assumes traffic spreads roughly evenly across keys. If traffic concentrates on one key instead, the partition holding that key hits its limit while the others sit idle. Requests to it are **throttled** (rejected with "slow down" errors) or queue up and become slow, even though the system as a whole has plenty of spare capacity.

This is called a **hot partition**, **hot shard** or **hot key**. It is one of the most common ways real distributed databases fail, because the total load graphs look healthy right up to the moment one key melts.

Hot keys come from the real world, which is never uniform:

- **Celebrities**: a pop star posts and gets a million likes in a minute; their profile is read by millions of fans.
- **Events**: a flash sale on one product, a breaking news article, a ticket release for one concert.
- **Time**: a key based on today's date receives every one of today's writes.
- **Bots and bugs**: one misbehaving client or a retry loop hammering a single ID.

Traffic on social platforms is famously skewed: a few accounts can receive a thousand times the traffic of a typical one.

~~~dg-bars Requests per second on eight partitions during a viral moment. The table is at about 30% of total capacity, yet partition 7 is throttling.
Partition 1 | 250 | 250
Partition 2 | 300 | 300
Partition 3 | 220 | 220
Partition 4 | 280 | 280
Partition 5 | 260 | 260
Partition 6 | 240 | 240
*Partition 7 | 16700 | 16,700, limit 1,000
Partition 8 | 310 | 310
~~~

## Uniformity matters, not the number of keys

What matters is how **uniformly** traffic spreads across partition key values, not how many key values exist. A table with a billion keys can still have one hot one.

| Partition key | Uniform? | Why |
| --- | --- | --- |
| User ID, many active users | Good | Each user is a small share of traffic |
| Status code, a few values | Bad | "active" holds most rows and traffic |
| Creation date rounded to the day | Bad | All of today's writes hit one key |
| Device ID, similar traffic per device | Good | Even load |
| Device ID, one device far busier than the rest | Bad | That device's partition runs hot |

### The limits are per partition

In DynamoDB, each physical partition supports roughly **3,000 read units and 1,000 write units per second** (a write unit is one write of up to 1 KB), and stores up to 10 GB. So a single key is limited to about 1,000 small writes per second, **no matter how much capacity the table has**. DynamoDB softens short spikes with **burst capacity** (unused capacity saved from the last few minutes) and **adaptive capacity** (shifting unused throughput towards busy partitions, and eventually splitting a hot partition so its keys spread over more machines). These help with uneven-but-spread traffic. They cannot help a single item that needs 16,000 writes per second, because one key cannot be split.

Other systems have the same shape of limit: one Cassandra partition lives on its few replica nodes, and one Redis key lives on one shard's single thread.

## Write sharding

The standard fix for write-heavy hot keys is **write sharding**: spread one logical key across many physical keys, so the load lands on many partitions.

**Random suffix.** Instead of writing every order for 9 July to the key `2014-07-09`, write to `2014-07-09.1` through `2014-07-09.200`, choosing the suffix at random for each write. Writes spread evenly over 200 keys, and so over many partitions. The cost: reading all of that day's items means querying all 200 keys and merging the results, and you cannot look up a single item directly, because you do not know which suffix it got.

**Calculated suffix.** Derive the suffix from another attribute instead, such as `hash(order_id) % 200`. Writes still spread evenly, and now you **can** find one item: recompute its suffix from its order ID and read one key. Reading the whole day still fans out to all 200.

~~~dg-compare Two ways to spread one hot key across 200 physical keys.
[Random suffix]
Write -(random 1 to 200)-> key 2014-07-09.137
+ Perfectly even writes
- Single-item lookup must try all 200
- Full read fans out to 200 keys
[*Calculated suffix]
Write -(hash order_id mod 200)-> key 2014-07-09.42
+ Even writes
+ Single-item lookup recomputes the key
- Full read still fans out to 200 keys
~~~

How many suffixes? Enough to keep each one comfortably under one partition's limit, and no more than readers can afford to query. For 16,000 writes per second against a limit of 1,000, you need at least 16; 32 or 64 gives headroom for growth and uneven randomness.

## Other tools for hot keys

### Sharded counters

A like or view counter is the classic hot key. Store the count as N separate counter keys: `likes:post123:0` to `likes:post123:63`. Each like increments a random one. Reading the total means summing 64 small values, which is cheap and can be cached. A background job can periodically roll the shards into a single total if reads are frequent.

### Buffer and batch

Instead of applying 18,000 increments of `+1`, collect them for a short window (in memory, or in a stream such as Kafka) and apply one update of `+18,000`. This turns a write rate problem into a tiny write rate. The cost is a delay of about the window length before the count is updated, and you must make the aggregation durable or accept a small loss on a crash.

~~~dg-seq Buffer and batch: thousands of like events are aggregated for one second and written as one increment per counter shard.
Clients > Stream : 16,700 like events per second
Stream > Aggregator : events for post 9, partitioned by post
Aggregator > Aggregator : sum for 1 second
Aggregator > Counter shard 5 : add 520
Aggregator > Counter shard 6 : add 515
Reader > Cache : like count for post 9
Cache -> Reader : 1,000,421, up to 2 s old
~~~

### Cache hot reads

A key that is **read** heavily, such as a celebrity profile, is best served from a cache. A single cache node can itself become hot, so very hot items are replicated across several cache nodes (append a random suffix to the cache key, or keep a small in-process cache on each app server for the top few hundred keys). Static content goes to a CDN.

### Isolate

Move known-hot keys (verified celebrities, the flash-sale product) to dedicated capacity: their own shard, their own cache cluster, or a separate code path. A directory-based shard map (lesson 04.05) makes this easy.

### Approximate

For display counts, a slightly stale or approximate number is fine. Nobody can tell 1,000,421 from 1,000,388, and "1M" is what the button shows anyway. Exactness matters for money and inventory, not for likes.

## Detecting hot partitions

Watch **per-partition or per-key** metrics, not just totals: throttled requests, latency per shard, and top keys by request rate. DynamoDB's CloudWatch Contributor Insights and Redis's `--hotkeys` option show the busiest keys. A table at 30% of provisioned capacity can still be throttling one key, and an average latency of 5 ms can hide one shard at 2 seconds. Alerts should fire on the worst partition, not the mean.

<div class="ex">
<div class="lab">Worked example</div>

**a viral like counter**

A post gets 1 million likes per minute: about 16,700 writes per second, far above one partition's limit of about 1,000.

Design: each like event is written to a stream partitioned by `(post_id, random 0 to 31)`, so the stream's load is spread too. Aggregators read the stream, sum likes per post per second, and apply one increment per second to each of 32 counter shards: 32 writes per second instead of 16,700.

The displayed count is the sum of the 32 shards, cached for 2 seconds. Whether a given user has liked this post (to show a filled heart, and to stop double likes) is stored separately under `(user_id, post_id)`, which is evenly distributed because each user likes each post at most once.

Result: no partition sees more than a few hundred writes per second, and the count lags reality by at most 3 seconds.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the date key that melted on Black Friday**

An orders table in DynamoDB uses `order_date` as its partition key so the operations team can list a day's orders. On a normal day there are 200,000 orders, about 2 per second on average and 50 at peak: no problem.

On Black Friday, peak traffic is 3,000 orders per second, all for the same date key. The partition is limited to about 1,000 writes per second, so two thirds of orders are throttled, and the checkout service's retries make it worse.

Fix: partition key `order_date#(hash(order_id) mod 50)`. Each suffix receives about 60 writes per second at peak. Listing a day's orders becomes 50 parallel queries, which the operations dashboard can afford; looking up one order recomputes its suffix.

</div>

## Common mistakes

- **Judging health by total capacity** rather than per-partition load.
- **Time-based or low-cardinality partition keys** on write-heavy tables.
- **Exact, synchronous counters** for things users only see as "1.2M".
- **Too many suffixes**, making every read fan out to hundreds of keys for no benefit.
- **Caching a hot key on one cache node** and moving the hot spot instead of removing it.

## In an interview

When your design has counters, popular items or celebrities, raise hot keys before the interviewer does. Say which key gets hot, give the per-partition limit you are designing against, then pick the tool: sharded counters and batching for writes, replicated caching for reads, isolation for known-hot entities, and approximation where users cannot tell.

<div class="take">
<div class="lab">Key takeaway</div>

partitions fail on uneven load, not total load. Spread hot keys with random or calculated suffixes and sharded counters, buffer and batch writes, replicate hot reads in caches, isolate known-hot entities, and accept approximate counts where users cannot tell.

</div>
