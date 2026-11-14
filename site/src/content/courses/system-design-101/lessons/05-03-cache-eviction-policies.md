---
id: "05.03"
course: "system-design-101"
module: "05"
slug: "cache-eviction-policies"
title: "Cache eviction policies"
keyIdeas: ["When the cache is full, the eviction policy decides what to remove.", "**LRU** (least recently used): evict what hasn't been touched longest. Good default.", "**LFU** (least frequently used): evict what's used least often. Better when popularity is stable; must age counts so old popular items don't stick forever.", "**FIFO**, **random**, **TTL-based**: simpler, sometimes good enough.", "Exact LRU needs a linked list update on every access. Redis samples a few keys and evicts the oldest among them, which approximates LRU with less memory and CPU.", "Scan-resistant policies (ARC, W-TinyLFU) avoid flushing the cache when one big scan touches many items once."]
question: "Why does Redis use approximate LRU instead of exact LRU?"
answer: "Exact LRU needs every key linked into one ordered list that is updated on every access, which costs about 16 bytes of pointers per key (gigabytes across hundreds of millions of keys) and extra work on every read on Redis's single command thread. Redis instead keeps a small last-access clock in each key's existing header and, when it must evict, samples a few random keys (5 by default), keeps the most idle ones in a 16-entry candidate pool, and evicts the oldest. For real, skewed workloads this is almost as good as true LRU at a fraction of the memory and CPU cost."
sources: [{"title": "Redis docs: Key eviction", "url": "https://redis.io/docs/latest/develop/reference/eviction/", "note": "LRU, LFU, approximations."}, {"title": "Wikipedia: Cache replacement policies", "url": "https://en.wikipedia.org/wiki/Cache_replacement_policies", "note": "LRU, LFU, ARC and more."}]
tags: ["Code"]
---

## When memory is full

A cache has a fixed amount of memory. Once it is full and a new item arrives, something has to go. The **eviction policy** decides what. A good policy keeps the items most likely to be requested again soon, which maximises the hit rate. A bad one throws away exactly the items you are about to need, so the cache stays full but rarely helps.

Eviction is different from expiry. **Expiry** (lesson 05.04) removes an item because its TTL ran out, regardless of memory. **Eviction** removes a perfectly valid item because there is no room. A healthy cache does some of both; a cache that evicts constantly is too small for its working set.

### The benchmark nobody can implement

The theoretical best policy is **Bélády's algorithm** (also called OPT or MIN): evict the item that will next be needed furthest in the future. It is provably optimal, but it needs to know the future, so it exists only as a benchmark. Researchers replay a recorded trace of requests, compute what OPT would have achieved, and compare real policies against it. If LRU gets 85% hits and OPT gets 90% on the same trace and size, there is at most 5 percentage points left to gain.

### Why access patterns matter

Real workloads are **skewed**. A small fraction of keys gets most of the requests, often following a power-law (Zipf) distribution: the most popular key gets twice the requests of the second, three times the third, and so on. That skew is why caching works at all. If 20% of products get 80% of views, caching that 20% captures most of the benefit. Eviction policies are different bets on how to recognise that popular minority.

## The common policies

### LRU (least recently used)

Evict the item that has not been touched for the longest time. The bet is **recency**: something used a moment ago is likely to be used again soon. The classic implementation is a hash map plus a doubly linked list. The hash map finds the item in O(1); the list keeps items in order of last use. On every access, you unlink the item and move it to the head. To evict, you remove the tail. All operations are O(1).

LRU's weakness is a **scan**: a one-time read of many items (a nightly report reading every product, a crawler walking old pages) pushes all of them to the head of the list and evicts everything that was genuinely popular. After the scan, the cache is full of items nobody will read again.

### LFU (least frequently used)

Evict the item used least often. The bet is **frequency**: keys that have been popular will stay popular. LFU resists scans, because a single read does not make an item look popular. Its weakness is **history**: an item that was hugely popular last week (yesterday's news story) keeps a big count and stays forever unless counts **decay** over time. LFU also needs a counter per item and a way to find the minimum quickly.

### FIFO and random

**FIFO** evicts the oldest inserted item, ignoring use. It is cheap (just a queue), but on its own it often evicts popular items simply because they were inserted early. **Random** evicts any item, with no bookkeeping at all. It is surprisingly decent when access is close to uniform, and it never suffers a pathological pattern, but it ignores skew.

### Scan-resistant designs

Several designs combine recency and frequency so that one-off reads cannot flush the cache:

- **SLRU (segmented LRU)**: new items enter a small **probation** segment. Only a second hit promotes them into the **protected** segment. A scan only churns the probation segment.
- **2Q** and **ARC** (adaptive replacement cache): keep separate lists for items seen once and items seen more than once, and in ARC's case adjust the size of each list automatically based on which one is producing hits.
- **TinyLFU** (used in Java's Caffeine library): a compact frequency sketch decides whether a new item is worth admitting at all, by comparing its estimated frequency with that of the eviction victim.
- **S3-FIFO** (2023): a small FIFO queue (about 10% of space) catches new items; items hit while there move to a main FIFO, the rest are dropped quickly. It exploits the fact that most items in real caches are **one-hit wonders**, requested once and never again.
- **CLOCK**: approximates LRU cheaply. Items sit in a circular buffer with a "referenced" bit. A pointer sweeps round; items with the bit set get a second chance (the bit is cleared), items without it are evicted. Operating systems use CLOCK variants for page replacement.

~~~dg-matrix Eviction policies compared. More bookkeeping usually buys better resistance to awkward access patterns.
Policy | Cost per access | Scan resistant | Adapts to change | Typical use
LRU | partial, list move | no | yes | general default
LFU | partial, counter | yes | no, unless decayed | stable popularity
FIFO | yes, almost free | no | partial | simple buffers
Random | yes, free | partial | yes | uniform access
SLRU / 2Q | partial | yes | yes | databases, CDNs
ARC / TinyLFU | partial | yes | yes | Caffeine, ZFS
CLOCK | yes, set one bit | no | yes | OS page cache
~~~

The trade-off is always the same: tracking more about usage improves the hit rate but costs memory and CPU on every access. On a cache handling a million reads per second, an extra 100 nanoseconds per access is a tenth of a CPU core.

~~~dg-widget lru A tiny cache. Request keys, switch policy, and try a one-off scan to see which policy keeps the popular keys.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**LRU versus a scan, traced**

A cache holds 4 items. The workload repeatedly reads A, B, A, C, A, B (so A and B are hot). Under LRU the cache holds A, B, C plus one slot, and hits almost every time.

Now a batch job reads W, X, Y, Z once each. Under LRU each new item goes to the head and the tail is evicted: after the scan the cache holds W, X, Y, Z. The next reads of A and B both miss, and so do the next reads of C.

Under SLRU with a probation segment of 2 and protected segment of 2, A and B were hit twice, so they sit in the protected segment. The scan items only cycle through probation (W and X are pushed out by Y and Z). After the scan, A and B still hit.

</div>

## Redis's options

Redis evicts only when memory use passes the `maxmemory` setting. What it evicts depends on `maxmemory-policy`:

| Policy | Evicts |
| --- | --- |
| noeviction | Nothing; writes that need memory fail with an OOM error |
| allkeys-lru | Least recently used among all keys |
| allkeys-lfu | Least frequently used among all keys |
| allkeys-random | Random keys |
| volatile-lru / volatile-lfu / volatile-random | Same, but only among keys that have a TTL |
| volatile-ttl | Keys with the shortest remaining TTL |
| allkeys-lrm / volatile-lrm | Least recently modified, counting writes only (Redis 8.6 and later) |

`allkeys-lru` is a good default when some keys are much more popular than others, which is nearly always. `allkeys-lfu` suits workloads with stable popularity and periodic scans. `allkeys-random` suits uniform access. The `volatile-*` policies are for an instance that mixes cache keys (with TTLs) and keys that must never be evicted (without TTLs). That works, but two separate instances are safer: one misconfigured TTL and your "permanent" data is gone.

Two traps. First, the default policy is **noeviction**, so a Redis used as a cache with no policy set will start rejecting writes when full. Second, replication output buffers and AOF buffers are not counted against `maxmemory`, so set it to perhaps 70 to 80% of the machine's RAM to leave headroom, or the operating system's out-of-memory killer may kill Redis instead.

### Reading the signs

`INFO stats` reports `keyspace_hits`, `keyspace_misses`, `evicted_keys` and `expired_keys`. Low hit rate plus many evictions means the cache is too small or the wrong keys are being evicted. Low hit rate plus many expirations means TTLs are too short. A high hit rate with a big eviction count can be fine: you are evicting the cold tail, as intended.

## Why Redis approximates LRU

Exact LRU needs every key in one ordered list, updated on every access. In Redis that would mean two extra 8-byte pointers per key (16 bytes) plus list manipulation on every read. With 200 million keys, that is about 3.2 GB of pointers alone, and every `GET` would have to touch shared list structure, adding CPU work and cache-line misses on Redis's single command thread.

Instead, Redis stores a 24-bit **last access clock** inside each key's object header (space it already had), and approximates:

1. When memory is needed, it picks a few random keys (`maxmemory-samples`, 5 by default).
2. It computes each one's **idle time** (now minus last access).
3. It adds good candidates to a small **eviction pool** of 16 entries that persists across rounds, sorted by idle time.
4. It evicts the key in the pool with the largest idle time.

The pool (added in Redis 3.0) is what makes this work well: over several rounds, the pool accumulates genuinely old keys, so each eviction picks from the best candidates seen recently rather than just the best of 5.

~~~dg-flow Redis's approximate LRU: sample a few keys, keep the best candidates in a pool, evict the most idle.
Memory over limit -(sample 5 random keys)-> Compute idle times -(merge)-> Pool of 16 candidates -> *Evict most idle
~~~

How good is it? With 10 samples the result is very close to true LRU. For the power-law access patterns of real workloads, where the cold keys are obviously cold, the difference in hit rate is little or none. The approximation only looks worse on uniform or adversarial patterns, where no LRU policy does well anyway. You can raise `maxmemory-samples` to 10 for slightly better accuracy at slightly higher CPU cost.

<div class="ex">
<div class="lab">Worked example</div>

**what the sampling costs and saves**

A Redis node holds 100 million keys and evicts 20,000 keys per second under memory pressure.

Exact LRU: 100 million × 16 bytes = 1.6 GB of list pointers, plus a list update on every one of perhaps 300,000 reads per second.

Sampling with 5 keys: zero extra memory per key (the clock fits in existing bits), and 20,000 × 5 = 100,000 random lookups per second only while evicting, with no work at all on reads beyond writing a timestamp. The hit rate difference on a skewed workload is typically under one percentage point.

</div>

### LFU in Redis

Redis's LFU uses the same sampling and pool but replaces the 24-bit clock with two fields: an 8-bit **logarithmic counter** (a Morris counter, range 0 to 255) and a 16-bit last-decrement time. The counter increments **probabilistically**: the higher it already is, the less likely an access is to bump it. With the default `lfu-log-factor` of 10, a counter of 255 represents around a million hits. So 8 bits can tell apart keys with 100, 100,000 and 1,000,000 accesses.

The counter also **decays**: by default (`lfu-decay-time 1`) it drops by one for every minute the key goes untouched. That solves LFU's "popular last week" problem. New keys start at 5, not 0, so they are not evicted before they have had a chance to be read twice.

~~~dg-bars Rough number of accesses needed to reach each LFU counter value with the default log factor of 10. The counter grows logarithmically, so 8 bits cover millions of hits.
Counter 10 | 100 | ~100 hits
Counter 18 | 1000 | ~1K hits
Counter 142 | 100000 | ~100K hits
*Counter 255 | 1000000 | ~1M hits
~~~

## Picking a policy

<div class="ex">
<div class="lab">Worked example</div>

**three caches, three policies**

A session store must never lose a session before it expires. Use a separate Redis instance with `noeviction`, enough memory for peak sessions (say 5 million sessions × 1 KB = 5 GB, provisioned at 8 GB) and an alert at 70% memory.

A product-page cache where a few thousand products get most traffic: `allkeys-lru`, sized for the hot set plus margin.

A cache that a nightly batch job sweeps by reading every product once: `allkeys-lfu`, so the scan's one-off reads have low counts and are evicted first, leaving the popular products in place.

</div>

## Common mistakes

- **Leaving `noeviction`** on a cache and getting OOM errors on writes at peak.
- **Setting `maxmemory` to all of RAM**, forgetting buffers and fragmentation.
- **Mixing permanent and cache data** in one instance with a `volatile-*` policy.
- **Blaming eviction for a low hit rate** that is really caused by short TTLs or poor key design.
- **Scans through the cache.** Batch jobs should read the database directly, or the cache should use a scan-resistant policy.

## In an interview

Say "LRU plus a TTL" as the default, then show depth: mention scan resistance (LFU or a segmented policy) if there are batch reads, and explain that Redis approximates LRU by sampling because exact LRU costs memory and CPU per access. Ask how skewed the access pattern is, because skew decides how small the cache can be.

<div class="take">
<div class="lab">Key takeaway</div>

LRU is the default, LFU protects consistently popular items, and scan-resistant policies stop one-time reads flushing the cache. Redis approximates LRU by sampling a few keys and keeping a pool of candidates, because exact LRU would cost extra memory per key and work on every access for almost no gain in hit rate.

</div>
