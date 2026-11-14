---
id: "15.03"
course: "system-design-101"
module: "15"
slug: "a-url-shortener-at-scale"
title: "A URL shortener at scale"
keyIdeas: ["At scale (billions of URLs, 100k+ redirects/s), add caching and distributed ID generation.", "Cache code → URL in Redis or in-memory LRU on app servers; hot links have a very high hit rate. Put a CDN in front for redirects if analytics allow.", "ID generation without a single DB sequence: a counter service handing out ranges (e.g. 1,000 IDs at a time) to each server, or Snowflake-style IDs.", "Store mappings in a sharded key-value store or sharded Postgres keyed by code.", "Click analytics go async: log redirect events to a queue and aggregate separately."]
question: "Redirects are 100x writes. Where do you cache, and 301 or 302?"
answer: "Cache at every layer of the read path: a CDN or edge cache that answers popular links near users, an in-memory cache (Redis, plus a small in-process cache for hot keys) of code to long URL in front of the database, and indexed lookups on database replicas for the remaining misses. Links do not change, so the caches rarely need invalidating; just keep TTLs within any expiry and seed the cache on creation so new links work at once. With a 90% CDN hit rate and 99% Redis hit rate, a 600,000 per second peak leaves only about 600 per second for the database. Use 302 if you want click analytics, the ability to change, expire or block links, or to avoid browsers caching the redirect forever; the caching layers absorb the extra load, and clicks are logged asynchronously. Use 301 only if you never need to see or change repeat clicks."
sources: [{"title": "Hello Interview: Bitly breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/bitly", "note": "Scaling reads with cache, unique IDs at scale."}]
tags: ["Design"]
tools: ["PostgreSQL", "Redis"]
---

## Why scale changes the design

Lesson 15.01 built a URL shortener on one machine and settled how to make codes. Now take a Bitly-scale version: **1 billion URLs**, **100 million daily users**, redirects under **100 ms**, 99.99% availability, and a preference for availability over consistency (a redirect that works is better than a perfectly fresh one). Traffic is extremely read-heavy: each new URL is clicked about 1,000 times on average over its life, and clicks are very unevenly spread.

The estimates:

- **Storage**: about 500 bytes a row (generous: code, long URL, owner, timestamps, expiry), × 1 billion rows ≈ **500 GB**. That fits on one modern Postgres instance. Storage is not the problem.
- **Writes**: about 100,000 new URLs a day ≈ **1 per second**. Also not the problem.
- **Reads**: 100 million users × 5 redirects ≈ 500 million a day ≈ **5,800 per second** on average. Allow a 100× spike when a link goes viral: about **600,000 per second** at peak.

So almost the whole design is about **serving redirects fast and reliably**. It is worth saying this out loud in an interview: the estimate tells you where not to spend effort (sharding a 500 GB table, distributed writes) and where to spend it (the read path).

## Splitting reads from writes

Because reads outnumber writes by about 5,000 to 1 on average (500 million redirects against 100,000 new links a day), and far more during a spike, split them into separate services that scale independently:

- **Write service**: a few stateless instances handling `POST /shorten`: validate the URL, get an ID, encode, insert.
- **Read service**: many stateless instances handling `GET /{code}`: look up and redirect. They autoscale on request rate.
- **Shared state**: the database (primary plus replicas), a Redis cache, and a counter for IDs.

Separate services also isolate failures: a bug or overload in the shorten path cannot slow redirects.

## Generating codes across many servers

On one machine an auto-increment column was enough. With several write servers (in several regions), you need IDs that are unique across all of them. The options:

- **Use a prefix of the long URL**: rejected; URLs sharing a prefix collide.
- **Hash + Base62** (canonicalise the URL, SHA-256, Base62, take the first N characters): the same URL always gets the same code. Collisions are caught by a UNIQUE constraint and retried with a salt, as in lesson 15.01.
- **Global counter + Base62**: a Redis `INCR` gives each new URL a unique number. No collisions, no checks. Codes are predictable, so scramble them if that matters.
- **Counter batching**: each write server reserves a **block of 1,000** numbers from Redis at a time (`INCRBY counter 1000`) and hands them out locally. That is one Redis call per 1,000 URLs. Different regions can get separate ranges. If a server crashes, its unused numbers are lost, which is fine: codes only need to be unique, not contiguous.
- **Snowflake-style IDs** (lesson 04.10): each server builds IDs from a timestamp, a machine number and a sequence, with no coordination at all. They are 64-bit, so they encode to 11 Base62 characters, longer than the alternatives.

~~~dg-seq Counter batching: one Redis call gives a write server 1,000 IDs, which it then hands out from memory.
Write server 1 > Redis : INCRBY url_counter 1000
Redis -> Write server 1 : 6000 (range 5,001 to 6,000 is yours)
Client > Write server 1 : POST /shorten long URL
Write server 1 > Write server 1 : next local id 5,001, base62 = 1iF
Write server 1 > Database : INSERT (5001, 1iF, long URL)
Write server 1 -> Client : https://sho.rt/1iF
Write server 1 > Write server 1 : ids 5,002 to 6,000 served locally
Write server 1 > Redis : INCRBY url_counter 1000 when the block runs out
~~~

What if the Redis counter is lost? If it restarts from an old snapshot, it could hand out ranges already used, causing duplicate codes. Protect against it: persist the counter with append-only logging, keep the database's unique constraint as the final guard, and on restart set the counter to `max(id)` from the database plus a safety gap.

Length: 62^6 ≈ 56 billion is plenty for 1 billion URLs, so **6 characters** suffice here.

## Making redirects fast

The read path has layers, each catching most of what the previous one missed. From the database outward:

1. **Index the code.** A primary key or B-tree index makes each lookup O(log n) instead of a table scan: about 4 levels for a billion rows, often all but the leaf already in memory.
2. **Database replicas** absorb cache misses and give availability. Since links never change once created, replica lag only matters for brand-new links (see below).
3. **In-memory cache** (Redis or Memcached) of `code → long URL`, read-through: check the cache; on a miss, read the database and fill the cache. Memory is roughly 1,000× faster than SSD. Links are immutable, so caching is easy: there is nothing to invalidate unless a link expires or is deleted. Keep cache TTLs no longer than the link's expiry, and evict with LRU, since a small fraction of links gets most clicks.
4. **CDN / edge.** Serve popular redirects from points of presence near users, either by letting the CDN cache the redirect response for a short time or with edge functions doing the lookup in an edge key-value store. Fastest for users, but adds cost and complexity.

~~~dg-layers The redirect read path, outermost first: each layer absorbs most of the traffic before it reaches the next.
*CDN edge | caches redirect responses for seconds to minutes, near users
Read service | stateless, autoscaled, in-process LRU for the hottest codes
Redis cache | code to long URL, read-through, LRU eviction
DB replicas | indexed lookup for cache misses
Primary DB | writes only, plus fallback reads for brand-new links
~~~

<div class="ex">
<div class="lab">Worked example</div>

**how much traffic reaches each layer at peak**

Peak is 600,000 redirects per second. Click traffic is heavily skewed: suppose the CDN can cache the hottest links and answers 90% at the edge. That leaves 60,000 per second for the read service.

Redis holds the 20 million most-clicked codes (20M × about 200 bytes ≈ 4 GB, fits in one shard with a replica; use 3 shards for throughput). It hits 99% of what reaches it: 59,400 per second, which 3 shards serve easily.

The remaining 1%, about 600 per second, go to the database replicas: trivial for two replicas doing indexed lookups.

Without the CDN, Redis would need to serve 600,000 per second, about 6 to 10 shards. Without Redis, the database would face 60,000 per second, needing many replicas. Each layer turns an expensive problem into a cheap one.

</div>

~~~dg-bars Requests per second reaching each layer at a 600,000 per second peak (log scale).
CDN edge | 600000 | 600,000/s
Read service | 60000 | 60,000/s
Redis | 60000 | 60,000/s, 99% hits
*Database replicas | 600 | 600/s
~~~

### The brand-new link problem

A user shortens a link and immediately clicks it. The write went to the primary; the read goes to a replica that may lag by a second, so it returns 404, and the 404 might even be cached. Fixes: write the new code into Redis as part of the shorten request (so the first read is a cache hit), fall back to the primary when a replica misses on a code younger than a few seconds, and never cache 404s for long (a few seconds at most).

### Hot keys

A viral link sends all its traffic to one Redis shard. At 200,000 clicks per second for one key, that shard is saturated. The CDN layer usually absorbs this. Without one, keep a small in-process LRU cache (a few thousand entries, a few seconds' TTL) on each read server, so each server asks Redis for the hot code only once every few seconds.

## 301 or 302 at scale

A **301** is cached by the browser (often indefinitely), so repeat clicks from the same browser never reach you. That sounds like free scaling, but you lose control: you cannot count clicks, change the destination, take down a link that turns out to be phishing, or expire the link, because browsers keep following the cached redirect. A **302** sends every click through you.

~~~dg-compare 301 versus 302 for a large shortener: 301 saves traffic, 302 keeps control.
[301 Moved Permanently]
click -(browser cache hit)-> long URL directly
+ repeat clicks cost you nothing
- no click analytics for repeats
- cannot change, expire or block the link
[*302 Found]
click -> CDN or read service -> long URL
+ every click counted
+ links can expire, be edited or blocked
- every click is served by you
- needs the cache and CDN layers
~~~

Pick **302** for a product like Bitly, whose value is analytics and control, and handle the extra load with the cache and CDN layers above. If you never need analytics or expiry, 301 is a valid way to cut traffic. **307** is like 302 but preserves the HTTP method; for plain link clicks (GET) the difference does not matter.

**Analytics without slowing redirects.** If clicks are counted, do not write to the database on the redirect path. Emit a click event (code, timestamp, country, referrer) to a log or stream (Kafka) asynchronously and aggregate it elsewhere (lesson 10.02). The redirect stays a pure read.

**Expiry.** Expired links return **410 Gone**, and a background job deletes expired rows. Store the expiry with the cached value so the read service can enforce it without a database call.

**Custom aliases** (`sho.rt/my-launch`) need their own namespace or a distinguishing rule (for example, aliases must be at least 8 characters or contain a hyphen) so a user cannot claim a string the generator will produce later.

## Availability: 99.99%

Four nines is about 4 minutes of downtime a month (lesson 14.02), too little for humans to react. So:

- run read services in at least two regions, behind health-checked DNS or anycast;
- replicate the database to each region (asynchronous is fine: links are immutable);
- if Redis fails, the read service falls back to replicas (with a circuit breaker so a dead Redis does not add a timeout to every request);
- if the primary fails, redirects keep working from replicas and cache; only shortening stops until failover.

<div class="ex">
<div class="lab">Worked example</div>

**a viral link, second by second**

A celebrity posts `sho.rt/xK3pQ9`. Within a minute it is getting 200,000 clicks per second worldwide.

The first request at each of 300 CDN edges misses and goes to a read service, which finds the code in Redis (or loads it from a replica and caches it). That is about 300 requests to the origin in total.

Every later click at each edge is answered by the CDN in a few milliseconds, until the edge's cached copy expires (say after 60 seconds), at which point the next request refreshes it. The origin sees about 5 requests per second for this code; the database sees almost none. The write path (1 per second) is unaffected.

Click events stream to Kafka at 200,000 per second and are counted by an aggregation job, so the dashboard shows the spike within a minute.

</div>

## Common mistakes

- **Over-engineering the write path** (sharding, distributed transactions) when it handles 1 write per second.
- **Writing click counts synchronously** on the redirect path.
- **Choosing 301 by default** and later discovering you cannot take down a malicious link.
- **Caching 404s for a long time**, breaking freshly created links.
- **One Redis shard for a viral key** with no in-process or CDN layer in front.

## In an interview

Show the arithmetic first (500 GB, 1 write per second, 600,000 reads per second at peak), then say: "the problem is the read path". Present code generation with batched counters, then the cache layers with hit-rate arithmetic, then the 301 versus 302 choice with a reason. Finish with analytics off the hot path and multi-region reads.

<div class="take">
<div class="lab">Key takeaway</div>

at 1 billion URLs, storage and writes are small; redirects (up to about 600,000 per second at peak) are everything. Generate codes with a counter handed out in batches, then put an index, replicas, an in-memory cache and a CDN in front of the database, so each layer absorbs most of what reaches it. Choose 302 if you need analytics, expiry or the ability to block links, record clicks asynchronously, and absorb the extra traffic with caching.

</div>
