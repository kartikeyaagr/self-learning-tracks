---
id: "05.06"
course: "system-design-101"
module: "05"
slug: "cache-availability-and-database-fallback"
title: "Cache availability and database fallback"
keyIdeas: ["Design for the cache failing. If the database can only handle 10% of traffic and the cache dies, the database dies next.", "Falling back to the database \"just in case\" is a mode that rarely runs, so it's poorly tested and fails exactly when needed.", "Options: replicate the cache, shed or rate-limit traffic to the database during cache outages, serve stale data from a local cache, and warm new cache nodes gradually.", "Test cache failures in staging or with game days.", "Avoid hard dependencies on the cache for correctness. It should make things faster, not be the only copy."]
question: "Your cache cluster dies. Why might sending all traffic to the database take the database down too?"
answer: "The database was sized only for the small fraction of reads that miss the cache, so with a 95% hit rate, losing the cache sends 20 times its normal read load to it all at once. Arrivals exceed capacity, queues and latency grow, requests time out while the database keeps doing the abandoned work, retries multiply the load further, connections run out and application threads block, so the database collapses and takes the rest of the site with it, and a cold cache then slows recovery. Instead of falling back blindly, cap and shed database load, serve stale or degraded data, and keep the cache itself highly available."
sources: [{"title": "AWS Builders' Library: Avoiding fallback in distributed systems", "url": "https://builder.aws.com/content/3EuS9Sakq7L3VLQIF3qzfMfke1Y/avoiding-fallback-in-distributed-systems", "note": "Why falling back to the DB can make outages worse."}, {"title": "AWS Builders' Library: Caching challenges", "url": "https://builder.aws.com/content/3K9TeXp6strMiJ4vhakpRD7BQWO/caching-challenges-and-strategies", "note": "Cold caches and downstream protection."}]
---

## The trap: falling back to the database

A cache usually exists because the database cannot handle the full read load on its own. So a tempting design follows: "if the cache is down, read from the database instead". It looks like resilience. The code is two lines in a `catch` block. In practice it is often the opposite of resilience, and it has caused some of the largest outages on record.

The core of the problem is arithmetic. A database behind a cache is sized for the **misses**, not for the total. At a 95% hit rate, the database sees 5% of reads. Losing the cache multiplies that by 20. At 99% it multiplies by 100.

| Hit rate | Database share of reads | Load multiplier if the cache dies |
| --- | --- | --- |
| 80% | 20% | 5 × |
| 90% | 10% | 10 × |
| 95% | 5% | 20 × |
| 99% | 1% | 100 × |
| 99.9% | 0.1% | 1,000 × |

Notice the irony: the better your cache works, the worse its failure is for the database. A very effective cache lets you run a very small database, which is exactly the database that cannot survive without it.

~~~dg-bars Database read load at 100,000 reads per second with a 95% hit rate, before and after the cache fails, against what the database can actually handle.
Normal (5% misses) | 5000 | 5,000 reads/s
Database capacity | 15000 | 15,000 reads/s
*Cache down | 100000 | 100,000 reads/s
~~~

## A real outage, step by step

Around 2001, amazon.com showed shipping speed estimates on product pages by querying a supply-chain database. That database could not handle the page traffic, so each web server got a local cache in front of it, with a fallback: if the cache failed, query the database directly. It worked for months.

Then the caches failed at roughly the same time (they shared a cause, so their failures were correlated). What happened next is the general pattern:

~~~dg-timeline How a cache fallback turned a missing shipping estimate into a site-wide outage (timings are illustrative).
T+0 | Caches on many web servers fail together
T+1 s | Every web server falls back to the database at once
T+5 s | *Database saturates, query latency rises to seconds
T+10 s | Web server threads block waiting on database calls
T+30 s | No free threads; product pages stop rendering
T+1 min | Site down; warehouses on the same database stop too
~~~

The fallback turned a small problem (no shipping estimates on product pages) into a total outage, and because the same database also ran fulfilment, warehouses stopped processing orders as well. The sharpest lesson: **if the database could reliably serve all the traffic directly, you would not have needed the cache**. A fallback to it is a plan that relies on something you already know is false.

## Why it gets worse than "just slower"

You might expect the database to simply serve requests more slowly. In practice overload causes **collapse**, for several connected reasons:

- **Queues grow without bound.** Once arrivals exceed capacity, every request waits behind all the earlier ones. Latency goes from 5 ms to seconds.
- **Timeouts become wasted work.** The caller gives up after 1 second, but the database keeps executing the query. Its capacity is spent on answers nobody receives, so the useful throughput falls towards zero.
- **Retries multiply load.** Each timed-out request is retried, perhaps 3 times, so 100,000 reads per second can become 300,000 attempts.
- **Resources exhaust.** Connection pools fill, the database runs out of connections or memory, and lock contention rises. Some databases start thrashing their buffer pool because the working set of a cache-less workload no longer fits.
- **The failure spreads upstream.** Application threads blocked on the database cannot serve anything else, including requests that never needed the database. That is how a shipping estimate takes down checkout.

And when the cache comes back, recovery is slow: it is **cold**, so its hit rate starts near zero and the database stays overloaded while it refills.

~~~dg-flow The fallback trap: when the cache fails, all its traffic lands on a database sized for misses only, and the failure spreads upward.
Cache cluster down -(100% of reads)-> *Database sized for 5% -(timeouts, retries)-> App threads blocked -> Whole site down
~~~

## Why fallbacks fail in general

This is a specific case of a wider rule about **fallback paths**: code that runs only when something else has failed.

- **They rarely run**, so they are untested at real load and may hide bugs written years ago by someone who has left.
- **They activate under stress**, exactly when the system has the least spare capacity.
- **They shift load to the weakest component.** The cache was there because the database was weak.
- **A single-node test does not reveal the problem.** Failing one cache node in a test sends a little extra traffic to the database and looks fine. The danger is many caches failing together, which tests rarely simulate.

This is sometimes called **modal behaviour** (or bimodal behaviour): the system has a normal mode and a failure mode that behaves completely differently, and the failure mode is the one you know least about. The safer goal is a system that behaves the same way whether things are going well or badly.

## Better options

### Make the primary path more reliable

Instead of adding a rarely used second path, make the cache itself hard to lose. Replicate each shard (Redis primary plus replicas with automatic failover), spread nodes across availability zones, and size the cluster with headroom so losing one node or one zone does not overload the rest. A cache cluster of 12 nodes across 3 zones, sized so 8 nodes can carry peak load, survives a whole zone failing.

### Cap and shed instead of passing everything through

When the cache is unhealthy, limit how much load reaches the database. Put a **concurrency limit** or **rate limiter** in front of database reads (per instance or global) set below the database's real capacity. Requests over the limit get a degraded answer or a fast error rather than queueing. The database then stays healthy enough to serve the requests it can, at normal latency. Serving 15% of requests well beats serving 0% badly.

### Serve stale data

Keep a local, longer-lived copy: a small in-process cache with a **hard TTL** much longer than the normal TTL (soft and hard TTLs from 05.01), or the last known good value. When the shared cache is down, serve that. For most read paths (product descriptions, profiles, recommendations), minutes-old data is far better than an error.

### Degrade the feature

Decide in advance which features can disappear. Hide the shipping estimate, show "recommendations unavailable", drop the like count. The page still renders and checkout still works. This is a product decision as much as a technical one, so make it before the outage.

### Turn fallback into failover

If you truly need two paths, **exercise both all the time**. For example, send a small share of real traffic down the second path constantly, so it is proven at load every day. A path that runs every day is a failover; a path that runs once a year is a gamble.

### Push data ahead of time

Change the design so readers do not need the source at request time. AWS, for example, pushes IAM credentials to instances in advance, valid for hours, so a brief outage of the credential service does not stop instances authenticating. The same idea works for configuration, feature flags and small reference data: ship a full copy to every server.

### Load test with the cache off

Find out what really happens. Many teams run a "cache off" game day in a staging environment at production load. If the answer is "the database dies", you know your limits must be in place before the real outage.

~~~dg-compare Two responses to a dead cache. Blind fallback passes the full load to a weak database; capping and degrading keeps the database alive.
[Blind fallback]
Cache fails -(all reads)-> Database -> Overload, total outage
- 20 × normal database load at 95% hits
- Timeouts and retries add more
- Slow recovery, cache refills cold
[*Cap, shed, serve stale]
Cache fails -(limited reads)-> Database -> Healthy at capacity
+ Over-limit requests get stale or degraded data
+ Core features keep working
+ Fast recovery when the cache returns
~~~

## Facebook's gutter pool

Facebook's memcache handles a failed cache **server** (rather than a whole cluster) with a small **gutter pool**: about 1% of cache servers kept idle. When a client gets no response from a memcache server, it does not go to the database. It retries the request on the gutter pool. On a gutter miss, it queries the database and fills the gutter entry with a **short TTL**, so that stale data does not linger there (the gutter does not receive invalidations the same way).

This absorbs the failed server's load. Client-visible failures dropped by 99%, and gutter hit rates passed 35% within 4 minutes of a failure, and often reached 50%. Facebook chose this over **rehashing** the failed server's keys onto the remaining servers, because the failed server might hold a hot key that would then overload its new owner and cause a cascade.

~~~dg-seq Gutter pool: when a cache server does not answer, the client tries a small idle pool before the database.
Client > Cache node 7 : get k
Cache node 7 -> Client : no response (timeout)
Client > Gutter pool : get k
Gutter pool -> Client : miss
Client > Database : read k
Client > Gutter pool : set k, short TTL
~~~

## Worked examples

<div class="ex">
<div class="lab">Worked example</div>

**cache outage plan for a profile service**

Normal: 200,000 reads per second, 97% hit rate, so the database serves 6,000 per second. Load testing shows it handles about 15,000 per second at acceptable latency.

One Redis node of 20 fails: its share (10,000 reads/s) moves to a gutter node, not the database. The gutter warms within minutes.

The whole cluster fails: a global limiter allows at most 10,000 database reads per second (two thirds of capacity, leaving headroom). That serves 5% of requests fully. The other 95% get a degraded profile (name and avatar from a small in-process cache with a 1 hour hard TTL) or a fast 503 with `Retry-After`. The database stays up, latency stays normal for the requests it serves, and when Redis returns, the warm-up is quick because the database was never overloaded.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**how retries turn 20 × into 60 ×**

A cache with a 95% hit rate in front of 50,000 reads per second fails. The database now receives 50,000 reads per second instead of 2,500: 20 times normal.

Its capacity is 8,000 per second, so latency spikes and most requests hit the client timeout of 500 ms. Each client retries twice. Now the database receives up to 150,000 attempts per second, 60 times normal, and its useful throughput drops below 8,000 because it wastes work on abandoned queries.

With a retry budget (retries capped at 10% of requests, lesson 02.09) and a limiter that admits 6,000 database reads per second, the database serves 6,000 useful reads per second and rejects the rest quickly.

</div>

## Common mistakes

- **`catch: return db.get(key)`** as the whole cache error handling.
- **Testing by failing one node** and concluding the fallback is safe.
- **Sizing the database for the cache-filtered load** with no plan for the cache-less load.
- **Retrying without a budget** during the outage.
- **Forgetting the cold restart**: the outage is not over when the cache comes back.

## In an interview

When you put a cache in front of a database, say what happens when the cache fails. A strong answer: "The database is sized for the misses, so I won't fall back blindly. I'll replicate the cache across zones, cap database reads with a limiter, serve stale or degraded data for the rest, and use a small gutter pool for single-node failures." Ask what the database can handle without the cache, because that number decides the plan.

<div class="take">
<div class="lab">Key takeaway</div>

a database sized for cache-filtered traffic cannot absorb the full load when the cache dies, and overload makes it collapse rather than just slow down. Do not blindly fall back. Make the cache highly available, cap and shed database load, serve stale or degraded data, and use small backup capacity such as a gutter pool.

</div>
