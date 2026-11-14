---
id: "12.04"
course: "system-design-101"
module: "12"
slug: "redis-geo-hot-path"
title: "Redis GEO hot path"
keyIdeas: ["Redis GEO stores points in a sorted set, using a 52-bit geohash as the score.", "`GEOADD` adds or updates a point; `GEOSEARCH` finds members within a radius or box, sorted by distance.", "Updates are cheap in-memory writes, so it handles frequent location updates from moving objects (drivers sending location every few seconds).", "A relational database would struggle with that write rate and index churn, and the data is short-lived anyway.", "Shard by region or city when one sorted set gets too big. Persist trip history elsewhere."]
question: "Drivers update location every 4 s. Why keep that in Redis rather than Postgres?"
answer: "Location updates are hundreds of thousands of overwrites per second of data that is useless after about 30 seconds and refills itself within 4 seconds if lost. Postgres would pay for each update with a new MVCC row version, a spatial index insertion, WAL writes and fsync, replication to replicas and later vacuum, all to make worthless data durable. Redis keeps the positions in memory as 52-bit geohash scores in a sorted set, so GEOADD is an O(log N) in-memory update and GEOSEARCH answers radius queries with a few range scans in under a millisecond; durability is unnecessary because the next update rebuilds the state. Trips and sampled history go to a durable database on a separate, asynchronous path."
sources: [{"title": "Redis docs: Geospatial", "url": "https://redis.io/docs/latest/develop/data-types/geospatial/", "note": "GEOADD, GEOSEARCH."}]
tools: ["Redis"]
---

## The workload

A ride-hailing app tracks its drivers continuously. Each driver's phone sends its position **every 4 seconds** while the app is online. With **1 million active drivers** at peak, that is **250,000 location writes per second**. At the same time, riders open the app and ask "which drivers are within 3 km?", and the matching service asks the same question every time it needs to dispatch a ride: thousands of queries per second, each needing an answer in a few milliseconds.

Before choosing a store, look closely at what this data is like. It has four unusual properties, and the design follows from them.

- **Write-heavy and overwrite-only.** Each update replaces the previous position for that driver. You never read an old position on the hot path; the history of a trip, if you want it, belongs somewhere else.
- **Short-lived.** A position older than about 30 seconds is useless for matching. The driver has moved 300 m or more.
- **Loss-tolerant.** If the whole store vanished, it would refill itself within one update interval, about 4 seconds, as every online driver sends a fresh position.
- **Small.** Each record is a driver ID, two coordinates and maybe a status: well under 100 bytes. A million drivers is about 100 MB.

Those properties describe a **hot, ephemeral working set**: small enough to fit in memory, changing too fast to be worth making durable. That points to an in-memory store.

## Redis GEO: what it is

Redis has a built-in geospatial index. It is not a separate data structure: it is a **sorted set** (Redis's ordered collection of unique members, each with a numeric score) with helper commands that convert coordinates to and from scores.

### Writing

`GEOADD key longitude latitude member` adds a member, or **updates** its position if it already exists. Underneath:

1. Redis normalises the coordinates. Latitude is limited to about ±85.05 degrees (the limit of the Web Mercator projection), longitude to ±180.
2. It interleaves 26 bits of longitude with 26 bits of latitude into a **52-bit geohash integer** (lesson 12.02), which gives a resolution of well under a metre.
3. It stores that integer as the member's **score** in the sorted set. A 52-bit integer fits exactly in the 53-bit mantissa of the double-precision float that sorted sets use for scores, which is why 52 was chosen.

A sorted set is implemented as a **skip list** (ordered by score, for range queries) plus a **hash table** (member to score, for direct lookups). Updating a member means finding it in the hash table, removing it from the skip list and reinserting it at its new score: **O(log N)**, a few microseconds.

### Querying

`GEOSEARCH key FROMLONLAT lng lat BYRADIUS 3 km ASC COUNT 20 WITHDIST` returns members inside a circle (or a box with `BYBOX`), sorted nearest first, with their distances. (`GEORADIUS` is the older form of the same thing.) Underneath:

1. Redis picks a geohash precision whose cells are large enough for the radius.
2. It computes the query point's cell and its **8 neighbours**, exactly the 3 × 3 trick from lesson 12.02.
3. For each of the 9 cells it does a **score range scan** on the skip list: all members whose score falls inside that cell's range.
4. It computes the exact distance to each candidate, discards those outside the circle, sorts and applies `COUNT`.

The cost is **O(N + log M)**, where M is the size of the set and N is the number of members inside the 9 cells. Everything happens in memory, so a typical query over a city takes well under a millisecond.

```
GEOADD drivers:nyc -73.9857 40.7484 driver:812          # every 4 s per driver
GEOSEARCH drivers:nyc FROMLONLAT -73.99 40.75 BYRADIUS 3 km ASC COUNT 20 WITHDIST
GEOPOS drivers:nyc driver:812                           # read one driver's position
ZREM drivers:nyc driver:812                             # remove: it is just a sorted set
```

~~~dg-seq The hot path: drivers overwrite their position every 4 seconds; the matcher reads nearby drivers from the same in-memory index.
Driver app > Location service : position (lat, lng), every 4 s
Location service > Redis : GEOADD drivers:nyc lng lat d812
Location service > Redis : ZADD seen:nyc now d812
Rider app > Matching service : request ride at pickup point
Matching service > Redis : GEOSEARCH 3 km ASC COUNT 20
Redis -> Matching service : 20 drivers with distances
Matching service > Matching service : drop busy and stale, rank by ETA
Matching service > Driver app : offer ride to best driver
~~~

## Why not Postgres for this?

PostgreSQL with PostGIS can index points in a GiST (R-tree style) index and answer radius queries well. For the static business listings of lesson 12.01, it is a strong choice. The problem here is not reading; it is the **write pattern**. Follow one `UPDATE drivers SET location = ... WHERE id = 812` through Postgres:

1. **MVCC creates a new row version.** Postgres never updates a row in place; it writes a new tuple and marks the old one dead, so concurrent readers can still see the old version (lesson 03.07 covers MVCC).
2. **The spatial index gets a new entry.** Because the indexed column changed, the update cannot use Postgres's cheap "heap-only tuple" path; the GiST index gets an insertion pointing at the new tuple, which may split nodes.
3. **The write-ahead log records it.** Each update writes WAL records for the heap and the index change, and the first change to each page after a checkpoint writes a full 8 KB page image.
4. **The commit waits for fsync** (unless you relax it), so the change is durable on disk.
5. **Replication ships the WAL** to every replica, which replays the same work.
6. **Vacuum cleans up later.** Every update leaves a dead tuple and a dead index entry that autovacuum must scan for and remove.

Multiply by 250,000 per second. Every one of those costs buys durability and history for data that is worthless 30 seconds later and refills itself in 4 seconds if lost. The table also stays tiny (a million rows) while generating gigabytes of dead versions and WAL per minute, which is the worst case for vacuum.

Redis does none of this on the hot path. An update is a hash lookup and a skip-list move in RAM. Persistence can be turned off for this dataset, or relaxed to periodic snapshots, because losing it is cheap.

~~~dg-bars Rough sustained update capacity per node for this workload. Postgres pays MVCC, WAL, index and vacuum costs on each write; Redis updates memory.
Postgres with PostGIS | 15000 | about 10,000 to 20,000 per s
Redis GEO, no pipelining | 80000 | about 80,000 per s
*Redis GEO, pipelined | 300000 | 200,000 to 400,000 per s
~~~

These are order-of-magnitude figures, and both systems can be tuned, but the gap of about 10 to 30 times per node is real, and it comes from doing far less work per write.

<div class="ex">
<div class="lab">Worked example</div>

**the numbers**

1 million drivers × 1 update per 4 s = 250,000 GEOADDs per second. Spread over 20 Redis shards by city or region, that is 12,500 per shard on average, and perhaps 40,000 on the shard holding a big city at rush hour. A Redis instance handles that comfortably, especially if the location service **pipelines** updates (sends a batch of commands in one round trip).

Memory: a sorted set entry with a member name like `driver:1234567` costs roughly 100 bytes including the skip list and hash table overhead. One million drivers is about 100 MB in total, or 5 MB per shard. A "last seen" set doubles it. That is trivial.

The same load on Postgres would be 250,000 row updates per second, each with WAL, a GiST insertion and a dead tuple. At about 300 bytes of WAL per update that is 75 MB per second of WAL, about 6.5 TB a day, replicated to every replica, for a table whose live size is 100 MB.

</div>

## Separate the hot path from the durable path

The design that comes out of this is two paths with different guarantees.

~~~dg-compare The same location stream, split by what each consumer needs. The hot path optimises for speed; the durable path for history.
[*Hot path: Redis GEO]
Every update -> GEOADD overwrite -> Matching reads
+ Latest position only, in memory
+ Sub-millisecond reads and writes
+ Rebuilt by the next round of updates
- No history, no durability
[Durable path: database or lake]
Sampled updates -(batched, async)-> Kafka -> Trips, analytics
+ Full trip traces, billing, disputes
+ Durable and replayable
- Seconds of delay, not used for matching
~~~

The location service writes every update to Redis and also publishes it to a log such as Kafka (lesson 08.02 covers logs and queues). Downstream consumers sample or batch it into a durable store: per-trip GPS traces for fare calculation and disputes, positions every 30 seconds for analytics, and aggregates by H3 cell (lesson 12.03) for pricing. None of that sits on the latency-critical path.

## Operating it

### Shard by region

A single sorted set for the whole world would be one key on one Redis node: a hot spot that cannot scale out. Use **one key per city or per coarse cell** (for example a geohash-4 or an S2 level-6 cell), and spread the keys across a Redis Cluster. Queries stay local: a rider in Manhattan only touches the New York key. Drivers near a region boundary need care: either write them to both neighbouring keys, or have queries near a boundary search both keys and merge.

### Stale drivers

Sorted-set members **cannot expire individually**; `EXPIRE` works only on whole keys. If a driver's phone dies, their last position stays in the set forever and they keep being offered rides. Three common fixes:

- **A last-seen set.** Alongside each GEOADD, write `ZADD seen:nyc <timestamp> driver:812`. A background job runs every few seconds: `ZRANGEBYSCORE seen:nyc -inf <now − 30 s>` finds stale drivers, and it removes them from both sets.
- **Filter at query time.** Fetch a few more results than needed and drop any whose last-seen time is too old.
- **Time-bucketed keys.** Write to `drivers:nyc:<minute>` and set a 2-minute TTL on each key. Queries search the current and previous minute and keep the newest position per driver. Old buckets disappear on their own.

### Availability state

A driver on a trip, or one who has just been offered a ride, must not be offered another. Either keep separate keys for available drivers (`ZREM` from `available:nyc` when they accept a trip, `GEOADD` back when they finish), or store status in a hash and filter after the search. Keeping the available set separate makes queries return useful results directly.

~~~dg-state A driver's lifecycle as the location store sees it. Only Available drivers are in the set that matching searches.
[*] -> Offline
Offline -> *Available : app online, first GEOADD
Available -> Offered : matcher reserves driver
Offered -> Available : declined or 15 s timeout
Offered -> On trip : accepts, removed from available set
On trip -> Available : trip ends, GEOADD again
Available -> Offline : no update for 30 s, reaper removes
~~~

### Reserving a driver safely

Two matching servers may find the same nearby driver at the same moment. Use an atomic reservation, for example `SET reserve:driver:812 ride:991 NX EX 15` (set only if absent, expire in 15 seconds). Whichever matcher succeeds offers the ride; the other moves to its next candidate. The TTL releases the driver automatically if the offer is never answered.

### Failure

If a Redis shard dies, its replica can take over within seconds, but even without replicas the damage is bounded: the next round of updates repopulates the set within about 4 seconds. During that window, riders in that region may see "no drivers nearby" or slightly fewer options. That is why giving up durability here is an acceptable, deliberate choice.

<div class="ex">
<div class="lab">Worked example</div>

**a shard failure walked through**

18:02:00: the Redis node holding `drivers:chicago` crashes. Its replica is promoted 3 seconds later, but asynchronous replication lost the last few hundred milliseconds of updates, so some drivers' positions are up to a second older. Nobody notices.

Now suppose there was no replica. The location service gets connection errors, a new empty node is assigned the key, and from 18:02:05 updates start arriving again. 25,000 Chicago drivers each update every 4 seconds, so after 1 second about a quarter are back, after 4 seconds almost all. Ride requests in those few seconds see thin supply and are retried by the matcher. No trip data is lost, because trips live in the durable path.

</div>

## Pitfalls

- **Treating Redis as the system of record.** Trip state, fares and driver accounts belong in a durable database. Redis holds only what can be rebuilt.
- **One giant key.** A worldwide `drivers` key concentrates all traffic on one node and makes every search scan a huge set. Shard by geography.
- **Forgetting staleness.** Without a reaper or time buckets, offline drivers haunt the map.
- **Too large a radius.** A 50 km `GEOSEARCH` in a dense city returns tens of thousands of members before `COUNT` trims them. Start small and widen only if needed.
- **Assuming it is exact distance by road.** GEOSEARCH gives straight-line distance. Rank the top 20 by an ETA service that knows roads and traffic.

## In an interview

State the properties of the data first (overwrite-only, short-lived, self-healing, small), because they justify everything else. Then explain how Redis GEO works underneath (52-bit geohash scores in a sorted set, 9-cell range scans), give the throughput arithmetic, and show the split between the hot path and the durable path. Mention sharding by region, stale-driver cleanup and atomic reservation. If asked "what if Redis goes down?", the honest answer is that the data rebuilds itself in seconds, which is the point.

<div class="take">
<div class="lab">Key takeaway</div>

frequent driver location updates are write-heavy, overwrite-only, short-lived and self-healing. Redis GEO stores them in memory as 52-bit geohash scores in a sorted set (GEOADD is O(log N), GEOSEARCH scans the 9 surrounding cells), avoiding the per-write MVCC, WAL, index, replication and vacuum costs Postgres would pay. Shard keys by region, reap stale drivers, reserve drivers atomically, and keep trips and history in a durable store on a separate path.

</div>
