---
id: "05.04"
course: "system-design-101"
module: "05"
slug: "ttl-expiration-and-cache-reapers"
title: "TTL expiration and cache reapers"
keyIdeas: ["A TTL (time to live) makes a key expire after a set time. It bounds staleness and frees memory for unused data.", "Redis expires keys two ways: **passively** (checked when a key is accessed) and **actively** (a background job samples keys with TTLs and deletes expired ones, repeating if many were expired).", "Deleting every key exactly at expiry would need a timer per key, too expensive at millions of keys, so expired keys may linger briefly in memory.", "Add jitter to TTLs so many keys set at the same time don't expire together and cause a load spike."]
question: "Why can't Redis delete every expired key the instant it expires?"
answer: "Deleting each key exactly at its expiry would need a timer per key or a continuously maintained sorted index over millions of keys, costing memory per key, work on every write and bursts of deletion work that would stall Redis's single command thread. Instead Redis deletes expired keys lazily when a command touches them, so clients never see them, and runs an active cycle about 10 times a second that samples around 20 volatile keys, deletes the expired ones and repeats while many are expired, within a CPU budget. Memory is reclaimed slightly late, but the work stays small and bounded."
sources: [{"title": "Redis docs: EXPIRE", "url": "https://redis.io/docs/latest/commands/expire/", "note": "Passive and active expiry, explained at the bottom."}]
tags: ["Code"]
---

## What a TTL is for

A **TTL** (time to live) gives a key an expiry time. After that moment the key is treated as gone. In Redis, `EXPIRE key 60` sets a 60 second TTL, `SET key value EX 60` sets the value and TTL in one atomic command, and `PEXPIRE` / `PX` do the same in milliseconds. A key with a TTL is called **volatile**; a key without one is **persistent**.

TTLs do two different jobs:

- **Bound staleness.** If invalidation fails (lesson 05.01), the TTL is the upper limit on how long a wrong value can survive. A product cached for 10 minutes is at most 10 minutes out of date, whatever else goes wrong.
- **Clean up temporary data.** Sessions, rate-limit windows, one-time codes, idempotency keys and distributed lock leases (06.08) should all disappear on their own. A TTL means you never need a cleanup job for them.

### The rules in Redis

TTL behaviour has sharp edges that cause real bugs:

- `EXPIRE` takes options: `NX` (only set if the key has no expiry yet), `XX` (only if it has one), `GT` and `LT` (only if the new expiry is later, or earlier, than the current one).
- **Overwriting** a key with `SET` (without `KEEPTTL`) clears the TTL. A cache refresh that does a plain `SET` makes the key permanent by accident.
- **Modifying in place** (`INCR`, `LPUSH`, `HSET`, `APPEND`) keeps the existing TTL.
- `PERSIST` removes a TTL. `RENAME` carries the TTL over to the new name.
- An expiry of zero or in the past deletes the key immediately.
- `TTL key` returns the seconds left, -1 if the key exists with no expiry, and -2 if the key does not exist.
- Expiry times are stored as **absolute Unix timestamps** in milliseconds, so they keep counting while Redis is stopped. A key with 1 hour left that is loaded from a snapshot 2 hours later is already expired.

~~~dg-state The life of a cache key with a TTL. Expired keys are invisible at once but deleted a little later.
[*] -> Persistent
[*] -> Volatile : SET k v EX 60
Persistent -> Volatile : EXPIRE k 60
Volatile -> Persistent : PERSIST or plain SET
Volatile -> Volatile : INCR keeps TTL
Volatile -> *Logically expired : time passes 60 s
Logically expired -> Deleted : read finds it or sampler hits it
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the rate limiter that never resets**

A rate limiter allows 100 requests per minute per user. The code runs `INCR rl:user42` and then, if the result is 1, `EXPIRE rl:user42 60`.

The app server crashes (or the network drops) between the two commands. The counter now exists with no TTL. It keeps climbing and never resets, so after 100 more requests user 42 is blocked forever.

Fixes: run both commands in one Lua script or `MULTI/EXEC` transaction so they are atomic; or create the key first with `SET rl:user42 0 EX 60 NX` and then `INCR` it. Either way, the key can never exist without its TTL.

</div>

## Why expired keys are not deleted instantly

Here is the obvious design: when a key's TTL reaches zero, delete it right then. Redis deliberately does not do this, and the reason is cost.

To delete every key at its exact expiry moment you would need either a **timer per key** or a **sorted index** of all expiry times that you watch continuously. Consider the numbers. A large Redis node can hold 100 million volatile keys, and under a workload of sessions and cache entries, tens of thousands might expire every second. A timer per key means 100 million timer objects in memory. A sorted structure (a heap or skip list ordered by expiry time) costs about O(log n) work on every write that sets a TTL, plus pointers per key, and still needs something to wake up at exactly the right moments.

Worse, Redis runs commands on **one thread**. Every microsecond that thread spends firing timers is a microsecond it is not serving requests. If 50,000 keys happened to share an expiry second (because they were all written together after a deploy), an exact-deletion design would have to delete all 50,000 at once, stalling every client.

So Redis splits the problem in two: make expired keys **invisible** immediately (cheap and exact), and **reclaim their memory** a little later (cheap but approximate). Two mechanisms do this.

### Passive (lazy) expiry

Whenever a command accesses a key, Redis first checks its expiry timestamp. If it has passed, Redis deletes the key there and then and behaves as if it never existed. A `GET` returns nil, an `EXISTS` returns 0. The check is a single comparison of two integers, so it costs almost nothing.

This guarantees **correctness**: no client ever sees an expired value. But it does nothing for keys nobody reads again. A million expired sessions from users who never come back would sit in memory forever.

### Active expiry (the sampler)

To catch the keys nobody reads, Redis runs a background **expire cycle** as part of its periodic housekeeping, which runs `hz` times a second (10 by default). Each cycle works roughly like this:

1. Pick a small batch of keys from the set of keys that have TTLs (about 20 per database per loop).
2. Delete every one of them that has expired.
3. If a large share of the sample was expired (historically more than 25%; newer versions tune this with an `active-expire-effort` setting), assume there are many more and repeat immediately.
4. Stop when the expired share drops, or when the cycle's time budget (a fraction of CPU, around 25% at most) is used up, and resume next tick.

Redis also runs a short "fast" cycle (around 1 ms) before it goes back to waiting for network events, so expired keys are reclaimed promptly between bursts of requests.

The sampling is a statistical argument. If 20 random volatile keys contain only 2 expired ones, then roughly 10% of all volatile keys are expired, and it is not worth spending more CPU right now. If they contain 15, then most volatile keys are probably expired and Redis should keep going.

~~~dg-flow Redis removes expired keys two ways: lazily when a command touches the key, and by a background sampler that adapts to how many it finds.
[Passive] Client reads key -(expiry passed)-> *Delete now, return nil
[Active] Every 100 ms -(sample ~20 keys with TTL)-> Delete expired ones -(many expired?)-> *Repeat, within CPU budget
~~~

### What you get

Expired keys are invisible immediately, but their memory may be reclaimed a little later. In steady state the share of memory held by expired-but-not-yet-deleted keys stays small (roughly the threshold the sampler aims for). Under a sudden mass expiry, it can briefly be higher.

<div class="ex">
<div class="lab">Worked example</div>

**watching the sampler work**

A node has 10 million volatile keys. At 12:00:00, 4 million of them expire together (they were all written with the same TTL an hour earlier).

The first cycle samples 20 keys and finds about 8 expired (40%), above the threshold, so it repeats. It keeps finding around 40% and keeps looping until it hits its CPU budget for that tick.

Each tick can delete on the order of tens of thousands of keys without stalling clients for long. Over the next tens of seconds, the expired share falls, the samples start coming back mostly unexpired, and the cycle relaxes to its normal light load.

Meanwhile any client reading one of those 4 million keys gets nil immediately, thanks to passive expiry. Nobody ever sees stale data; the only cost is memory being freed over seconds rather than instantly. This is also a good argument for TTL jitter: spreading those 4 million expiries over 10 minutes avoids the burst entirely.

</div>

~~~dg-timeline One volatile key from creation to reclaimed memory. Correctness is instant; cleanup is lazy.
11:00:00 | SET session:9 EX 3600, expiry stored as absolute time
11:59:59 | GET returns the value as normal
12:00:00 | *Key is logically expired; any read now returns nil
12:00:03 | Active sampler happens to pick it, deletes it, frees memory
12:00:03 | Primary sends DEL session:9 to replicas and AOF
~~~

## Expiry and replication

Replicas do not expire keys on their own timers. When the primary expires a key (passively or actively), it generates an explicit `DEL` and sends it down the replication stream and into the append-only file. That way every copy agrees about exactly when a key disappeared, even if clocks differ slightly between machines.

Before that `DEL` arrives, a replica still holds the key, but it checks the expiry on reads and hides logically expired keys from clients. So reads from replicas also never return expired values (in modern versions).

## Cache reapers in general

The same design shows up everywhere: rather than precise timers, systems **mark** entries with an expiry, **check on read**, and run a periodic **background sweeper**, sometimes called a reaper.

- **Memcached** reclaims expired items lazily when it encounters them, and runs an LRU crawler thread in the background that walks item lists and frees expired ones.
- **Facebook's memcache** found that short-lived keys were wasting memory while waiting to be lazily reclaimed. They added a **transient item cache**: a linked list of keys indexed by expiry second, swept every second. It cut one key family's share of memory from 6% to 0.3%.
- **Databases** do the same with TTL columns. **Cassandra** writes a tombstone when a TTL expires and removes the data during compaction, after a grace period. **DynamoDB's TTL** deletes expired items in the background, typically within a few days, not instantly, so queries must still filter on the expiry attribute.
- **Operating systems and runtimes** use **timer wheels**: an array of buckets, one per tick, where each timer is dropped into the bucket for its expiry tick. Adding a timer is O(1), and each tick only looks at one bucket. This is how Kafka, Netty and the Linux kernel handle huge numbers of timeouts cheaply.

~~~dg-compare Two ways to expire data. Exact timers are precise but costly at scale; lazy checks plus sampling are cheap and still never show expired data.
[Exact timers]
Write key -(insert into sorted index)-> Timer fires -> Delete at exact time
+ Memory freed immediately
- Per-key memory and O(log n) work per write
- Bursts of expiries stall the server
[*Lazy plus sampling]
Write key -(store timestamp)-> Read checks it -> Sampler sweeps the rest
+ Almost free per key
+ Work per cycle is bounded
- Memory freed a little late
~~~

## Choosing TTLs

- **Match the TTL to how stale the data may be** and how often it changes. Prices or stock levels: seconds to a minute. A user's profile: minutes. A country list: a day.
- **Add jitter.** Use, for example, 600 seconds plus or minus 10% so keys written together do not expire together and cause a stampede (05.05).
- **Use soft and hard TTLs** (05.01) when serving slightly old data beats failing: refresh after the soft TTL, give up only after the hard TTL.
- **Do not rely on exact expiry timing** for business logic. A key's memory may linger, and in databases like DynamoDB the item may be readable for days. If something must stop at an exact moment, store the deadline and check it.

## Common mistakes

- **Plain `SET` on refresh** wipes the TTL and makes the key permanent. Use `SET ... EX` or `KEEPTTL`.
- **Separate `INCR` and `EXPIRE`** calls that can be split by a crash.
- **Identical TTLs** for keys written in a batch, causing synchronised expiry.
- **Expecting memory to drop the instant keys expire** and alerting on it.

## In an interview

If asked about expiry, say "Redis expires lazily on access plus by random sampling, so expired keys are never returned but memory is reclaimed a little later". Then connect TTLs to staleness bounds and stampede prevention.

<div class="take">
<div class="lab">Key takeaway</div>

TTLs bound staleness and clean up temporary data. Redis makes expired keys invisible immediately by checking on every access, and reclaims their memory with a background sampler that adapts to how many expired keys it finds, which keeps the work cheap and bounded.

</div>
