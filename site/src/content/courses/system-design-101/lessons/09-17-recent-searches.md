---
id: "09.17"
course: "system-design-101"
module: "09"
slug: "recent-searches"
title: "Recent searches"
keyIdeas: ["Recent searches show a user their last few queries.", "Store per user a capped list: Redis `LPUSH` then `LTRIM 0 9`, or a small table with (user_id, query, timestamp). Dedupe so a repeated query moves to the top.", "Sync across devices by storing server-side; store locally if it only needs to work on one device.", "Privacy: let users delete entries and clear history, expire old entries, and don't record searches in private modes.", "Reads happen on every search-box focus, so serve them from a fast store."]
question: "Store the last 10 searches per user. Where, with what eviction, and what about privacy?"
answer: "Use a Redis list per user (key recent:{user_id}), sharded by user across a Redis cluster; at under 1 KB per user, 10 million users need about 10 GB. On each search, asynchronously and in one MULTI/EXEC or Lua script, LREM any older copy of the query, LPUSH it to the head, LTRIM the list to indexes 0 to 9 and refresh an EXPIRE such as 90 days; read it with LRANGE 0 9. Eviction is by recency: the oldest entry falls off when an eleventh distinct search is pushed. Add a durable key-value copy only if history must sync across devices or persist, and use browser local storage for logged-out users. For privacy, let users delete single entries, clear all or turn history off, limit retention, delete it (and any durable copy) with the account or on a legal request, and never feed individual histories into suggestions shown to others."
sources: [{"title": "Redis docs: Lists", "url": "https://redis.io/docs/latest/develop/data-types/lists/", "note": "LPUSH + LTRIM for a capped recent list."}]
tags: ["Design"]
---

## What the feature needs

"Recent searches" shows a user their last few queries when they tap or click into the search box, before they type anything. It's small, but it's one of the most used parts of search: people repeat searches a lot (checking an order, re-finding a product, a colleague's name in a chat app), and a tap on a recent search saves typing and lands on a query that already worked.

Write the requirements down precisely, because each one drives a design choice:

- **Per user**: every user has their own list.
- **Ordered newest first.**
- **A fixed maximum**, say 10 entries. Older ones fall off.
- **No duplicates**: searching "running shoes" again moves it to the top rather than adding a second copy.
- **Fast reads**: the list appears the instant the search box gets focus, so the read must take a millisecond or two on the server.
- **Cheap writes**: one write per search, which can be a very high rate across all users.
- **User control**: delete one entry, or clear all.
- **Privacy**: search history is personal data, with retention limits and legal obligations.

The question "where, with what eviction, and what about privacy?" maps directly onto these: a store suited to small per-user ordered lists, eviction by recency with a fixed cap, and a set of privacy controls.

## Redis lists fit exactly

A **Redis list** is an ordered sequence of strings stored under one key. Internally it's a **quicklist**: a doubly linked list of compact, contiguous blocks (listpacks). That gives:

- **O(1)** push and pop at either end (head or tail).
- O(N) access by index in general, which doesn't matter at length 10.
- Up to 2³² − 1 elements per list, far more than needed.

Redis lists support a well-known pattern called the **capped list**: push the newest item to the head, then trim the list to a fixed length so the oldest item falls off the tail. That is exactly "keep the 10 most recent".

```
LREM   recent:u42 0 "running shoes"   # remove any older copy
LPUSH  recent:u42 "running shoes"     # newest at the head, O(1)
LTRIM  recent:u42 0 9                 # keep only the 10 newest
EXPIRE recent:u42 7776000             # whole list expires after 90 idle days
LRANGE recent:u42 0 9                 # read newest to oldest
```

What each command does and costs:

- `LREM key 0 value` removes **all** copies of a value. It's O(N) in the list's length, trivial at 10. It provides de-duplication, and it also implements "delete this one entry" when the user taps the x.
- `LPUSH` adds to the head in O(1).
- `LTRIM 0 9` keeps indexes 0 to 9 and drops the rest. Its cost is O(N) in the number of elements **removed**, which here is at most one per push.
- `EXPIRE` sets a time-to-live on the whole key. Each new search resets it, so the list only disappears after 90 days of no searching.
- `LRANGE 0 9` reads the list; its cost is proportional to the offset plus the number of elements returned, tiny for 10.

~~~dg-flow Recording a search: remove the older copy, push to the head, trim to 10, and refresh the expiry.
New search -(LREM duplicate)-> LPUSH to head -(LTRIM 0 9)-> *10 newest kept -(EXPIRE)-> Idle lists vanish
~~~

### Atomicity

Two searches from the same user can arrive at once (two tabs, a retry). If their commands interleave (`LREM` from one, `LREM` from the other, then both `LPUSH`es), you can end up with a duplicate. Wrap the sequence in `MULTI` / `EXEC` (a transaction that runs the commands back to back with nothing interleaved) or in a short **Lua script**, which Redis runs atomically. Pipelining the commands also cuts the round trips from four to one.

### Eviction by recency

**Eviction** is the rule for what to throw away when the list is full. Here it's **recency**: the oldest entry is removed when an eleventh is pushed. That matches what users expect from "recent". Frequency-based eviction (keep the most-searched) would be a different feature, "frequent searches", and would need counts, which a sorted set provides instead.

~~~dg-timeline One user's list over a day. Repeats move to the top; the oldest entry falls off when an eleventh distinct search arrives.
09:00 | Search "running shoes": list has 1 entry
09:05 | Search "trail shoes": 2 entries, newest first
12:30 | Search "running shoes" again: *moved to top, still 2
14:00 | Eight more searches: list is full at 10
14:10 | Eleventh distinct search: oldest entry trimmed
18:00 | User taps x on "running shoes": LREM removes it
~~~

### Alternatives you could use

- A **sorted set** keyed by query with the timestamp as score: `ZADD` naturally de-duplicates (updating the score moves it to the top), and `ZREMRANGEBYRANK key 0 -11` trims to the newest 10. Equally good; slightly more memory.
- A **database table** `(user_id, query, searched_at)` with an index on `(user_id, searched_at desc)`: works, but every search becomes an insert plus a delete of old rows, and reads hit the database on every search-box focus. Fine at small scale, wasteful at large.
- **Inside the main search logs**: no, those are analytical, batch-oriented and shouldn't be read per user in real time.

## Where it lives

~~~dg-compare Three places to keep recent searches. Most products use Redis, adding a durable copy only when history must sync or persist.
[Browser local storage]
Search -(write locally)-> Device only
+ Never leaves the device
+ Works logged out
- Doesn't sync across devices
- Lost when storage is cleared
[*Redis per user]
Search -(LPUSH, LTRIM)-> Redis cluster
+ Sub-millisecond reads
+ Syncs across devices
- Lost if cache data is lost
[Redis + durable store]
Search -(write both)-> Redis + key-value DB
+ Survives cache loss
- More writes, more to delete
~~~

### Sizing

Memory is small. 10 entries of about 30 bytes each, plus Redis's per-key and list overhead (roughly 100 to 200 bytes), is under 1 KB per user.

<div class="ex">
<div class="lab">Worked example</div>

**sizing for 10 million active users**

10,000,000 users × about 800 bytes ≈ 8 GB, call it 10 GB with overhead and headroom. A Redis cluster of 3 primaries with 4 to 8 GB each, plus replicas, holds it comfortably, sharded by the user key (`recent:{user_id}` hashes to a slot). If those users run 50 million searches a day with peaks of 2,000 searches a second, that's 2,000 pipelined write transactions a second, and maybe 5,000 reads a second from search-box focus events. A single Redis primary handles over 100,000 simple operations a second, so the load is trivial; the cluster is sized for memory and availability, not throughput.

</div>

### Durability choices

- If losing the history on a cache failure is acceptable (it usually is: it's a convenience, and it rebuilds itself as the user searches), **Redis alone** is enough. Enable persistence (AOF or snapshots) and replicas so routine restarts and failovers don't lose it.
- If the history must **sync across devices** reliably or be kept as part of the user's account (some products show it on a "your activity" page), write it to a **durable key-value store** too, for example DynamoDB keyed by user, and treat Redis as a cache in front. Write asynchronously so search latency isn't affected.
- **Logged-out users** can keep recent searches in the browser's **local storage**, which never leaves the device. Optionally merge it into the account history when they sign in.

~~~dg-seq A search writes the history asynchronously; focusing the search box later reads it from Redis in about a millisecond.
User > Search API : search "running shoes"
Search API > Search engine : run query
Search API > Redis : MULTI LREM, LPUSH, LTRIM, EXPIRE (async)
Search engine -> Search API : results
Search API -> User : results page
User > Search API : focus search box later
Search API > Redis : LRANGE recent:u42 0 9
Redis -> Search API : 10 recent queries
Search API -> User : show recent searches
~~~

Note the write is **off the critical path**: the search results don't wait for the history write. If Redis is briefly unavailable, the search still works and that one history entry is simply missed.

## Privacy

Search history is **personal data**, and often sensitive: it can reveal health conditions, finances, sexuality, religion, politics, job hunting or relationship trouble. Treat it accordingly.

- **User control.** A visible "Clear history" (delete the key, and any durable copy) and a per-item remove (`LREM`). A setting to **turn history off**, after which nothing is recorded. Honour these immediately, not in tomorrow's batch.
- **Retention limits.** Keep only what the feature needs. The `EXPIRE` above deletes idle history automatically; you might also cap the age of individual entries. Data you don't keep can't leak.
- **Separation from analytics.** Aggregate query logs used for autocomplete, related searches and ranking (lessons 09.14, 09.16, 09.08) should be pseudonymised, aggregated and **thresholded** (a query must come from many distinct users before it can be suggested), and must never surface one user's searches to another.
- **Account deletion and legal requests.** Deleting an account must delete this key and any durable copy. Under laws such as the GDPR, users can ask for a copy of their data or its erasure, so you must be able to find and remove every copy, including backups within their retention window.
- **Shared and private contexts.** Show history only to the signed-in user. Don't record searches made in private or incognito modes. On shared devices, consider not showing history at all until the user signs in.
- **Security.** Restrict who can read the Redis cluster; don't log full history in application logs; encrypt durable copies at rest.

<div class="ex">
<div class="lab">Worked example</div>

**one user's day, with deletion**

u42 searches "running shoes", then "trail shoes", then "running shoes" again. On the third search the transaction runs LREM (removing the old "running shoes"), LPUSH (adding it at the head), LTRIM (no-op, only 2 entries) and EXPIRE (90 days from now). The list reads ["running shoes", "trail shoes"].

That evening, u42 searches for a medical symptom, then decides they don't want it in their history and taps the x: `LREM recent:u42 0 "<that query>"`. It's gone from Redis at once and from the durable copy within seconds. Because autocomplete only suggests queries seen from at least 50 distinct users, the query was never eligible to appear for anyone else. If u42 doesn't search for 90 days, the whole key expires and the history disappears without anyone running a cleanup job.

</div>

## Common mistakes

- Appending without de-duplication, so the list shows "running shoes" five times.
- Doing the history write synchronously in the search request path.
- Storing history forever with no expiry or user controls.
- Feeding raw individual histories into shared suggestions, leaking one user's searches to others.
- Forgetting the durable copy and backups when handling deletion requests.

## In an interview

Answer the three parts directly. Where: a Redis list per user (key `recent:{user_id}`), optionally backed by a durable key-value store if history must sync or persist, and local storage for logged-out users. Eviction: LREM the duplicate, LPUSH, LTRIM to 10 in one transaction, so the oldest falls off (recency), plus an EXPIRE for idle users. Privacy: per-item delete and clear-all, a history-off setting, retention limits, deletion with the account and on legal request, and strict separation from aggregated, thresholded analytics. Then do the memory arithmetic.

<div class="take">
<div class="lab">Key takeaway</div>

store recent searches in a per-user Redis list: LREM the duplicate, LPUSH the new query, LTRIM to the newest 10 and set an expiry, all in one transaction and off the request's critical path. Eviction is by recency and every operation is effectively constant time at this size. Treat the history as personal data: user-controlled deletion, retention limits, and no leakage into shared suggestions.

</div>
