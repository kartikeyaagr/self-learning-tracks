---
id: "05.01"
course: "system-design-101"
module: "05"
slug: "caching-layers"
title: "Caching layers"
keyIdeas: ["Caches sit at many layers: browser, CDN, load balancer/reverse proxy, application memory, distributed cache (Redis/Memcached), database buffer pool.", "**Cache-aside**: app checks cache, on miss reads DB and fills cache. Simple and common; data can be stale until TTL expires or you invalidate.", "**Write-through**: app writes to cache and DB together. Cache stays fresh, writes are slower.", "**Write-behind**: write to cache, flush to DB later. Fast writes, risk of losing data if the cache fails.", "Invalidation is the hard part: delete the cache key on write (not update it) to avoid race conditions, and use TTLs as a safety net.", "Measure hit rate. A cache with a 30% hit rate may not be worth its complexity."]
question: "Compare cache-aside and write-through. Which one can serve stale data and when?"
answer: "Cache-aside loads data into the cache only on a read miss and relies on TTLs or deletes to remove old values, so it can serve stale data after a database update if the delete fails or a slow reader writes an old value back after the delete (the read-write race); the staleness lasts until the TTL expires. Write-through updates the cache on every write, so reads of recent writes are fresh, but writes are slower and the cache fills with data that may never be read. It can still serve stale data if one of its two writes fails or concurrent writes land in different orders, so both patterns need a TTL as a safety net."
sources: [{"title": "System Design Primer: Cache", "url": "https://github.com/donnemartin/system-design-primer#cache", "note": "Client, CDN, web server, DB and app caches."}, {"title": "Hello Interview: Caching", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/caching", "note": "Cache-aside, write-through, write-behind."}, {"title": "AWS Builders' Library: Caching challenges and strategies", "url": "https://builder.aws.com/content/3K9TeXp6strMiJ4vhakpRD7BQWO/caching-challenges-and-strategies", "note": "What goes wrong in production."}]
tools: ["MySQL"]
---

## Why cache at all

A **cache** keeps a copy of data somewhere faster or closer than the place the data really lives (the **source of truth**, usually a database). Reading the copy is cheaper than reading the original, so you serve most requests from the copy and only go to the source when you have to.

Two things make caching worth the trouble. The first is **latency**. Rough numbers: a Postgres query that has to touch disk might take around 50 ms, while a Redis read over the local network takes about 1 ms. A CDN edge server near the user answers in 20 to 40 ms, compared with 250 to 300 ms from an origin server on another continent. An in-process cache (a hash map inside your own application) answers in well under a microsecond because there is no network at all.

The second is **load**. A cache absorbs traffic that would otherwise hit the database. At 10,000 requests per second with a 95% **hit rate** (the share of reads the cache can answer), the database sees only 500 requests per second. Push the hit rate to 99% and it sees 100. That is why caching is often the cheapest way to scale reads: one Redis node handling 100,000 operations per second costs far less than the database replicas you would need to serve the same load.

~~~dg-bars Typical read latency at each caching layer. Each step outward saves roughly an order of magnitude.
In-process map | 0.0001 | ~0.1 µs
Redis on the LAN | 1 | ~1 ms
CDN edge | 30 | 20 to 40 ms
*Database from disk | 50 | ~50 ms
Distant origin | 275 | 250 to 300 ms
~~~

The price you pay is **staleness**. The moment you have two copies of a piece of data, they can disagree. Every caching decision is really a decision about how much disagreement you can tolerate, for how long, and what you will do to limit it. The rest of this lesson is about that trade.

## Where caches live

Caches appear at every layer between the user and the data. Each layer has a different trade-off between speed and how much control you have over staleness.

- **Client**: the browser HTTP cache, mobile app storage. Fastest of all because no request leaves the device, but you control staleness least. Once a browser has a response marked `Cache-Control: max-age=86400`, you cannot call it back.
- **CDN**: edge servers around the world cache static media and sometimes public API responses (lesson 13.02 covers this). Great for content that is the same for everyone.
- **In-process**: memory inside each application server, such as a Guava or Caffeine cache in Java, or a plain dictionary. No network hop. But every server holds its own copy, so with 200 servers you have 200 copies that can disagree, and invalidating them all is hard. Good for small, rarely changing data such as configuration or feature flags, and as a shield for extremely hot keys.
- **External cache**: a shared Redis or Memcached fleet that all application servers talk to. One copy (per key) that everyone sees, so invalidation is a single delete. This is the default answer in interviews.
- **Database caches**: the database's own **buffer pool** keeps hot pages in memory without you doing anything. A database whose working set fits in the buffer pool is already fast; caching in front of it helps mostly with expensive queries and with load.

~~~dg-layers Where caches live, from closest to the user to closest to the data.
Browser / client | HTTP cache, app memory
CDN | edge copies of static and public content
In-process | per-server memory, no network hop
*Shared application cache | Redis / Memcached beside the services
Database | buffer pool keeps hot pages in RAM
~~~

You will often combine several. A product page might be cached as HTML at the CDN for 60 seconds, the product record in Redis for 10 minutes, and the currency exchange rates in-process for 5 minutes.

## How the cache is kept filled: the five patterns

The interesting question is who puts data into the cache and who keeps it up to date. There are five standard patterns.

| Pattern | Reads | Writes | Watch out for |
| --- | --- | --- | --- |
| Cache-aside (lazy loading) | App reads cache; on a miss reads DB and stores the result | App writes DB, then deletes the cache key | First read is slow; stale until expiry if delete fails or races |
| Read-through | Cache itself loads from DB on a miss | Usually combined with another pattern | Needs a cache layer that knows how to load (CDNs work this way) |
| Write-through | Served from cache | Write goes to cache, which writes DB synchronously | Slower writes; cache fills with data nobody reads |
| Write-behind (write-back) | Served from cache | Write goes to cache; flushed to DB later in batches | Data loss if the cache dies before flushing |
| Refresh-ahead | Served from cache | Cache refreshes popular entries before they expire | Wasted work if it predicts wrong |

### Cache-aside step by step

**Cache-aside** is the usual default because it is simple and the cache stays optional. The application owns all the logic; the cache is just a key-value store.

1. The app asks the cache for `user:42`.
2. On a **hit**, it returns the value. Done.
3. On a **miss**, it queries the database.
4. It writes the result into the cache with a TTL (time to live, an expiry time), then returns it.
5. When the user is updated, the app writes the database first, then **deletes** `user:42` from the cache. The next read misses and reloads the fresh row.

Why delete rather than update the cache on a write? Because a delete is **idempotent** (safe to repeat): sending it twice has the same effect as once. An update is not: if two writers race, their cache updates can land in the opposite order from their database writes, leaving the older value cached. Deleting also avoids computing an expensive cache value for data nobody may read again. Facebook's memcache (05.02) follows exactly this "update the database, then delete the key" rule.

~~~dg-seq Cache-aside: the app checks the cache, falls back to the database on a miss, and fills the cache.
App > Cache : GET user:42
Cache -> App : miss
App > Database : SELECT user 42
Database -> App : row
App > Cache : SET user:42, TTL 300 s
App > App : return row to caller
~~~

Cache-aside has a useful property: if the cache is unavailable, the app can still work (slowly) by going to the database. (Lesson 05.06 explains why that fallback is more dangerous than it looks.)

### Write-through step by step

With **write-through**, every write goes to the cache and the database together, and the write is acknowledged only once both have it. Usually the application writes the database and then sets the cache with the new value, or a cache layer does both. Reads of recently written data are always fresh, because the write itself put the new value into the cache.

The costs: every write pays two round trips. The cache fills with everything that was written, including data that is never read (think of an audit log entry). And a brand new cache node starts empty, because write-through only fills keys when they are written. For that reason write-through is almost always combined with cache-aside: write-through keeps hot data fresh, cache-aside fills in anything missing.

~~~dg-compare Cache-aside versus write-through on the same update. Cache-aside removes the old value and reloads later; write-through puts the new value in immediately.
[*Cache-aside]
Write DB -(then)-> Delete key -(next read)-> Miss, reload
+ Cache only holds data that was read
+ Cache outage does not block writes
- First read after a write is slow
- Stale if the delete fails or races
[Write-through]
Write DB -(then)-> Set key to new value -> Hit
+ Recently written data reads fresh
- Every write pays two round trips
- Cache fills with unread data
- Two writes can still partly fail
~~~

### Write-behind, read-through, refresh-ahead

**Write-behind** acknowledges a write as soon as the cache has it, then flushes to the database later, often in batches. It gives the fastest writes and can collapse many updates to one row into a single database write (useful for counters such as view counts). But if the cache node dies before flushing, those writes are gone. Use it only for data you can afford to lose or rebuild.

**Read-through** is cache-aside with the loading logic moved into the cache layer. The app only ever talks to the cache; on a miss, the cache calls a loader function or the origin. CDNs are read-through caches: the edge fetches from your origin on a miss. Libraries such as Caffeine also support it with a loader callback.

**Refresh-ahead** reloads popular entries shortly before they expire, so readers never see a miss. It works well for a small set of known hot keys. If it guesses wrong, it wastes database work refreshing things nobody reads. Lesson 05.05 covers a clever probabilistic version (XFetch).

## Staleness: where each pattern goes wrong

With cache-aside the cached value only changes when it expires or the app deletes it. So stale data appears whenever the delete does not happen or happens at the wrong moment.

### Failure 1: the lost delete

The app writes the database, then the delete call to Redis times out, or the process crashes between the two steps. The old value now lives in the cache until its TTL runs out. This is why every cache entry should have a TTL even if you invalidate explicitly: the TTL is your upper bound on staleness when invalidation fails.

<div class="ex">
<div class="lab">Worked example</div>

**bounding a lost delete**

A product record is cached with a 10 minute TTL. At 14:00:00 the price changes from 20 to 25. The database write succeeds, but the delete to Redis times out because a network blip lasts 2 seconds.

The cached entry was written at 13:58:30, so it expires at 14:08:30. Until then every product page shows 20. That is 8.5 minutes of staleness.

If the TTL were 60 seconds, the worst case would be 60 seconds. If there were no TTL at all, the wrong price would stay until the next price change or a cache restart, possibly days. Shorter TTLs mean less staleness but more misses: with 50,000 reads per second spread over 100,000 products, a 60 s TTL costs roughly 100,000 / 60 ≈ 1,700 extra database reads per second for keys that are read at least once a minute.

</div>

### Failure 2: the read-write race

This one is subtle and happens even when every call succeeds.

1. Reader A misses on `user:42` and reads version 1 from the database.
2. Before A writes to the cache, writer B updates the row to version 2.
3. B deletes `user:42` from the cache (nothing to delete yet).
4. A finally writes version 1 into the cache.

The cache now holds version 1, the database holds version 2, and no further delete is coming. The cache stays wrong until the TTL. This race needs A to be slow (a garbage collection pause, a slow network), so it is rare per request but certain at scale. Facebook's **leases** (05.05) exist to stop exactly this: the cache refuses A's write because the key was deleted after A's lease was issued.

~~~dg-seq The classic stale-cache race: a slow reader writes an old value back after the writer invalidated it.
Reader > Database : read v1 (cache miss path)
Writer > Database : update to v2
Writer > Cache : delete key
Reader > Cache : SET v1 (stale until TTL)
~~~

Other mitigations: **delayed double delete** (delete, then delete again a second or so later, which catches most slow readers), or writing with a version number and having the cache reject older versions.

### Failure 3: write-through partial failure

Write-through avoids most read staleness, but it is still two writes to two systems with no transaction across them. If the database write succeeds and the cache write fails, the cache holds the old value. If the cache write succeeds and the database write fails, the cache holds a value that was never committed, which is worse. Two concurrent write-through writers can also land in opposite orders in the two systems. So even write-through needs a TTL as a safety net.

## Operational lessons from running caches at scale

Teams that run caches across large fleets keep relearning the same lessons.

- **Caches create two modes.** Systems quietly come to depend on the hit rate. If it drops (a cache flush, a deploy that changes key formats, a traffic shift to a new region), the source gets far more load than it was sized for. Test with the cache disabled so you know what happens. Lesson 05.06 is entirely about this.
- **Local caches** have coherence problems (servers disagree with each other), load on the source that grows with fleet size (each of 500 servers misses separately), and cold starts after every deploy. **External caches** fix those, but are another fleet to run and can fail.
- **Soft and hard TTLs**: after the soft TTL, try to refresh; if the source is down, keep serving the old value until the hard TTL. This turns a source outage into slightly stale data instead of errors.
- **Negative caching**: cache "not found" and errors briefly too (say 30 seconds). Otherwise a request for a missing item hits the database every time, and an attacker can exploit this by requesting random IDs.
- **Monitor** hit rate, evictions, cache CPU and memory, and the downstream request volume. A falling hit rate is often the first sign of trouble.
- **Version your cached format** like stored data. If a deploy changes the serialised shape, old and new servers must both read it, or you must change the key prefix (`v2:user:42`) gradually so the whole cache is not invalidated at once.

<div class="ex">
<div class="lab">Worked example</div>

**product page design**

Product details are read 50,000 times per second and change a few times a day. Use cache-aside in Redis with key `product:{id}` and a 10 minute TTL with ±10% jitter so keys written together do not expire together.

On a price update, the service writes the database, then deletes `product:{id}`. If the delete fails, the page can show the old price for up to about 11 minutes. That is acceptable for display because checkout always reads the price from the database, so customers are never charged a stale price.

Sizing: 2 million products, average 2 KB each, but only the 200,000 most popular are hot. Hot set is 200,000 × 2 KB = 400 MB, which fits on one Redis node with plenty of room. With a 98% hit rate the database sees 1,000 reads per second instead of 50,000.

</div>

## Common mistakes

- **Updating the cache instead of deleting** on writes, which invites ordering races between writers.
- **No TTL** on entries "because we invalidate explicitly". Invalidation always fails sometimes.
- **Using write-behind for data that matters**, then losing orders when a cache node restarts.
- **Reading critical values from the cache**: prices at checkout, account balances, permissions after a revoke.

## In an interview

Do not open with caching. Name a bottleneck first (read volume, an expensive query, a latency target). Then say what you cache, the key design, the pattern (usually cache-aside with delete on write), eviction (LRU plus TTL), and one or two failure modes with how you bound them. A good question to ask: "How stale can this data be?" The answer decides the TTL and whether caching is allowed at all.

<div class="take">
<div class="lab">Key takeaway</div>

cache-aside is the default: read the cache, fall back to the database on a miss, fill the cache, and delete the key on write. Both cache-aside and write-through can serve stale data when invalidation fails or races, so always set a TTL to bound how long.

</div>
