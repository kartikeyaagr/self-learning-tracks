---
id: "05.05"
course: "system-design-101"
module: "05"
slug: "cache-concurrency-control"
title: "Cache concurrency control"
keyIdeas: ["**Cache stampede (thundering herd)**: a hot key expires and thousands of requests miss at once, all hitting the database.", "Fixes: a lock or lease so only one request recomputes while others wait or get the stale value; serve stale while revalidating in the background; refresh early with a probability that rises near expiry (XFetch); never expire hot keys and update them proactively.", "**Stale sets**: request A reads old data from DB, request B updates DB and deletes the cache, then A writes its old value into the cache. Memcache leases fix this by rejecting sets whose lease was invalidated.", "Delete-on-write is safer than update-on-write for concurrent writers."]
question: "A hot key expires and 10,000 requests miss at once. Name 2 fixes."
answer: "(1) Make only one request rebuild the value: a lease issued by the cache (Facebook's memcache leases, which cut one hot key's database load from 17,000 to 1,300 queries per second), a distributed lock such as SET rebuild:key token NX EX 5, or request coalescing in each process, while the other requests wait briefly and re-read or get a stale copy. (2) Refresh the key before it expires, with a background job, a soft TTL that serves the old value while one request refreshes, or probabilistic early expiration (XFetch), so the key never goes missing under load. Adding jitter to TTLs also stops many keys expiring at once, and a database concurrency cap is the last line of defence."
sources: [{"title": "Facebook memcache paper, section 3.2.1 leases", "url": "https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf", "note": "Fixing stale sets and thundering herds."}, {"title": "Optimal Probabilistic Cache Stampede Prevention (XFetch)", "url": "https://cseweb.ucsd.edu/~avattani/papers/cache_stampede.pdf", "note": "Refresh early with probability."}]
tools: ["Redis"]
---

## The stampede

A **cache stampede** (also called a **thundering herd** or **dog-piling**) happens when a popular key expires or is invalidated, and every request for it misses at the same moment. Each of those requests then does the same expensive thing: query the database and recompute the value.

The numbers explain why this is dangerous. Suppose a key is read 10,000 times a second and rebuilding it takes 200 ms. In the 200 ms between the first miss and the first successful refill, about 10,000 × 0.2 = **2,000 requests** miss and all go to the database for the same value. And it gets worse, because the load is self-reinforcing:

1. 2,000 identical queries make the database slower.
2. A slower database makes each rebuild take longer, say 2 seconds instead of 200 ms.
3. A longer rebuild window lets more requests through: now 20,000.
4. Those time out, clients retry, and the database falls over.

The problem needs keys that are both **hot** (many readers) and **expensive** to rebuild, and it is worst for keys that are **frequently invalidated**, because each invalidation opens a new window. Facebook's memcache deployment hit exactly this with keys that were both very popular and updated often.

~~~dg-timeline A stampede on one hot key: in the gap between expiry and refill, thousands of identical rebuilds pile onto the database.
0 ms | Hot key expires (read 10,000 times a second)
1 ms | First request misses, starts a 200 ms rebuild
50 ms | 500 more requests have missed, all querying the database
200 ms | *2,000 identical queries in flight; database latency climbs
600 ms | Rebuilds now take 2 s; misses keep arriving
2 s | Requests time out and retry; database overloaded
~~~

There are two families of fixes, and you normally use one from each. **Make only one request rebuild** the value while the others wait or get a stale copy. Or **never let the key go missing** under load, by refreshing it before it expires. On top of both, avoid synchronised expiry and protect the database anyway.

## Fix 1: leases (one rebuilder, enforced by the cache)

Facebook's solution is the **lease**, built into its memcache servers:

- On a miss, memcache gives the client a **lease token**, a 64-bit number bound to that key.
- Only a client holding the current lease may write the value back. A `set` without a valid token is rejected.
- Memcache hands out a lease for a given key **at most once every 10 seconds**. Other clients that miss during that window get a special reply telling them to wait briefly and retry. A few milliseconds later the value is usually back, and they get a hit.

The effect at Facebook: peak database queries for one hot, frequently invalidated key dropped from **17,000 per second to 1,300**.

~~~dg-seq Memcache leases: only the lease holder rebuilds; other missing clients are told to wait and retry.
App A > Memcache : get k
Memcache -> App A : miss + lease token 7
App B > Memcache : get k
Memcache -> App B : miss, wait and retry
App A > Database : recompute k
App A > Memcache : set k with token 7
App B > Memcache : get k
Memcache -> App B : hit
~~~

### Leases also fix stale sets

Leases solve a second problem at no extra cost. Recall the race in lesson 05.01: a slow reader reads an old value from the database, a writer updates the database and deletes the key, then the slow reader writes its old value into the cache, where it stays until the TTL.

With leases, the writer's delete **invalidates the outstanding lease**. When the slow reader finally tries to set the old value with its token, memcache rejects it because the token is no longer valid. The stale value never enters the cache.

### Accepting slightly stale values

Facebook added one more refinement. When a key is deleted, memcache keeps the old value briefly in a separate area. A client that misses and is told to wait can instead choose to take that **slightly stale** value immediately, if its use case tolerates it (a like count, say, but not a privacy setting). This turns waiting into a fast, slightly old answer.

## Fix 2: request coalescing (one rebuilder per process)

The same "only one rebuild" idea works inside your own service without any cache support. When many requests in one process miss on the same key at the same time, only the first actually fetches from the database; the others **wait on its result** and share it. This is often called **single flight** (Go's `singleflight` package is the standard example) or **request collapsing**. Proxies such as NGINX (`proxy_cache_lock`) and Varnish do it for HTTP.

```
inflight = {}   # key -> future
def get(key):
    v = cache.get(key)
    if v is not None: return v
    if key in inflight: return inflight[key].wait()
    f = inflight[key] = Future()
    try:
        v = db.load(key); cache.set(key, v, ttl=jitter(600)); f.resolve(v)
    finally:
        del inflight[key]
    return v
```

Coalescing does not coordinate across servers, but it shrinks the herd from "every request" to "one per server". With 100 app servers, that is at most 100 concurrent rebuilds instead of 2,000.

### Extending it across servers with a lock

A distributed lock in Redis extends this to the whole fleet. On a miss, a server runs `SET rebuild:k <random token> NX EX 5`. The one server that succeeds rebuilds the value, writes it, then releases the lock (checking the token first, as lesson 06.08 explains). Servers that fail to get the lock either sleep 50 ms and re-read the cache, or serve a stale copy kept under a separate key. The `EX 5` matters: if the rebuilder crashes, the lock frees itself after 5 seconds.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the herd with and without coalescing**

A key is read 10,000 times a second across 50 app servers (200 per server per second). Rebuild takes 200 ms.

No protection: about 2,000 database queries for one value, then the slowdown spiral.

Per-process coalescing: each of the 50 servers sends one query; 50 concurrent queries. The database barely notices.

Distributed lock: one query in total. The other 49 servers wait or serve stale for 200 ms. That saves 49 queries but adds a Redis round trip to every miss and a dependency on the lock, so for this load, coalescing alone is probably enough.

</div>

## Fix 3: refresh before expiry

The other family prevents the miss in the first place.

### Probabilistic early expiration (XFetch)

Each reader, on a cache hit, rolls a die to decide whether to refresh the value **a little early**. The chance rises as the expiry approaches, and is higher for values that are slow to compute. The rule:

`refresh if now - delta × beta × ln(rand()) >= expiry`

Here `delta` is how long the last recompute took, `beta` (default 1) tunes eagerness, and `rand()` is uniform in (0, 1]. Because `ln(rand())` is negative, `-delta × beta × ln(rand())` is a positive random "head start", usually small but occasionally large. Far from expiry, almost no head start is large enough. Close to expiry, a few readers' head starts push them over, and they refresh.

With many readers, typically **one** refreshes a bit before expiry and everyone else keeps hitting the cache. No locks, no coordination. The algorithm needs `delta` and `expiry` stored alongside the value.

~~~dg-flow XFetch: on each hit, a reader may refresh early; the chance rises as expiry nears and with longer recompute times.
Cache hit -(now - delta x beta x ln(rand) >= expiry?)-> *Yes: recompute early and reset TTL
Cache hit -(no)-> Serve cached value
~~~

<div class="ex">
<div class="lab">Worked example</div>

**XFetch in numbers**

A value takes delta = 0.5 s to compute and expires at t = 60 s. beta = 1. A reader at t = 59 s refreshes if -0.5 × ln(rand()) >= 1, which means ln(rand()) <= -2, so rand() <= e^-2 ≈ 0.135. So about 13.5% of readers at t = 59 s would refresh.

At t = 55 s the reader needs -0.5 × ln(rand()) >= 5, so rand() <= e^-10 ≈ 0.000045. With 1,000 reads a second, that is about one early refresh every 20 seconds. In practice the first reader to win the roll, somewhere around 56 to 57 s, refreshes, and the key never expires. (Add a small lock or coalescing so two simultaneous winners do not both recompute.)

</div>

### Background refresh

For a small, known set of hot keys (the homepage, top-100 lists, configuration), a scheduled job recomputes them every N seconds and writes them with a TTL longer than N. They never expire under normal operation. The job is the only rebuilder, so there is no herd.

### Soft TTL (stale-while-revalidate)

Store the value with a **soft** expiry inside it and a longer **hard** TTL on the key. A reader that finds the soft expiry passed returns the existing value immediately and triggers one asynchronous refresh (guarded by coalescing or a lock). HTTP has the same idea built in: `Cache-Control: max-age=60, stale-while-revalidate=30`.

~~~dg-compare The two families of stampede fixes. In practice you combine one of each, plus jitter.
[One rebuilder]
Miss -(lease, lock or single flight)-> One query -> Others wait or take stale
+ Works for any key, including invalidations
+ Leases also block stale sets
- Waiters see extra latency
- Locks add a dependency
[*Refresh before expiry]
Hit -(XFetch, soft TTL, background job)-> Refresh early -> Key never missing
+ Readers never wait
+ No coordination for XFetch
- Does not help after an explicit delete
- Wasted work on keys that go cold
~~~

## Fix 4: avoid synchronised expiry

Keys created together expire together. After a deploy, a cache flush or a bulk import, millions of keys can get the same TTL, and an hour later they all miss in the same second: a stampede across many keys at once. Add **jitter**: instead of exactly 3,600 seconds, use 3,600 plus a random amount up to ±10%. Expiries spread over 12 minutes instead of one instant.

## Fix 5: protect the database anyway

Even with all of the above, a cold start or a full cache flush can cause a herd across thousands of different keys, and per-key fixes do not help. So cap concurrency at the database: a semaphore of, say, 50 concurrent queries per app server, or a per-key-prefix limit. Requests over the cap wait briefly, get stale data, or are shed with a fast error (load shedding is covered in 02.13). A database at 100% of its capacity is fine; a database asked for 300% falls over and serves nobody.

## Common mistakes

- **Fixing only expiry.** Refresh-before-expiry does nothing after an explicit delete, so keep a single-rebuilder fix (lease, lock or coalescing) for invalidations too.
- **Locks without TTLs**, so a crashed rebuilder blocks the key forever.
- **Waiters that spin** by hammering the cache in a tight loop. Sleep with a little jitter between retries.
- **Identical TTLs** for a whole batch of keys.
- **Negative results not cached**, so a missing key stampedes the database every time.

<div class="ex">
<div class="lab">Worked example</div>

**homepage trending list**

The trending list is read 10,000 times a second and costs 300 ms to compute. With a plain 60 s TTL, every minute a burst of about 3,000 requests hits the database.

Fix: a background job recomputes it every 30 s and writes it with a 120 s TTL, so it never expires under normal operation. The job also writes a copy to `trending:stale` with a 1 day TTL.

If the job fails for more than 2 minutes, readers start missing. Each server coalesces its misses into one; that one tries `SET rebuild:trending NX EX 10`. The winner rebuilds; everyone else serves `trending:stale`. The database sees one query, users see a list that is a few minutes old.

</div>

## In an interview

When you add a cache for hot data, mention the stampede before the interviewer does. Name one "single rebuilder" fix (lease, lock, single flight) and one "refresh early" fix (background refresh, soft TTL, XFetch), plus jitter. Ask how expensive a rebuild is and how often the key is invalidated: those decide which fix matters most.

<div class="take">
<div class="lab">Key takeaway</div>

stop stampedes by letting only one request rebuild a key (leases, locks, request coalescing), by refreshing before expiry (background refresh, soft TTLs, XFetch), and by jittering TTLs. Leases also stop slow readers writing stale values back, and a concurrency cap keeps the database alive when everything else fails.

</div>
