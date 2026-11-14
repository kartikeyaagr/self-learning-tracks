---
id: "02.07"
course: "system-design-101"
module: "02"
slug: "consistent-hashing-load-balancing"
title: "Consistent-hashing load balancing"
keyIdeas: ["Hash a request key (user ID, cache key) to choose a backend, so the same key always lands on the same server. This keeps local caches warm and enables stateful routing.", "Plain `hash % N` reshuffles almost every key when N changes. Consistent hashing moves only about 1/N of keys.", "Virtual nodes (many ring positions per server) smooth out uneven load.", "Hot keys still overload one server. \"Bounded loads\" caps each server at, say, 1.25x the average and spills extra keys to the next server on the ring.", "Used in caches, sharded databases, and L7 balancers (Envoy ring hash, Maglev)."]
question: "Why route requests for the same key to the same backend? What breaks if one backend gets too hot?"
answer: "Routing the same key to the same backend keeps that server's cache warm for it (a high hit rate without every server caching everything), keeps per-key state such as a session or chat room in one place, and matches requests to the shard that owns the data. If one backend gets too hot, for example because a viral key lives on it, its queue grows and latency and errors rise for every key it owns, even cold ones, and it may fall over and push its load onto a neighbour. Fixes are replicating or salting the hot key, a local cache in front, or consistent hashing with bounded loads so overflow spills to the next server on the ring."
sources: [{"title": "Google Research: Consistent hashing with bounded loads", "url": "https://research.google/blog/consistent-hashing-with-bounded-loads/", "note": "How Vimeo and Google keep backends evenly loaded."}, {"title": "Hello Interview: Consistent Hashing", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/consistent-hashing", "note": "The ring and virtual nodes."}]
---

## Why route by key at all

Most load balancing (lesson 02.06) treats backends as interchangeable: any server can handle any request. Sometimes that is wasteful or impossible, and you want **the same key to go to the same server every time**:

- **Cache locality**: if all requests for user 42 hit the same cache server, that server holds user 42's data and the hit rate is high. Spread randomly, every server must cache everything, so each needs the full working set in memory, and a new item is missed once on every server before it is cached everywhere.
- **Stateful connections or sessions**: a chat room, a multiplayer game session or a collaborative document lives on one server, and every participant must reach that server.
- **Partitioned data**: a database or cache shard owns part of the key space, and requests for a key must go to its owner.

The routing function is a **hash** of the key: a function that turns any key into a number that looks random, so keys spread evenly. The question is how to map that number to a server, especially when servers are added or removed.

## The problem with modulo

The obvious approach is `server = hash(key) % N`. It spreads keys evenly and is trivial to compute. It works until N changes.

Going from 3 servers to 4 changes the result of the modulo for most keys. A key keeps its server only if `hash % 3` happens to equal `hash % 4`, which is true for just a quarter of keys. So **75% of keys move**. In general, going from N to N + 1 servers moves N / (N + 1) of the keys: 90% when going from 9 to 10.

For a cache tier, that is a disaster. Almost every lookup misses at once, and the database behind the cache is suddenly hit with nearly the full read load it was protected from. The same happens when a server crashes: N drops by one and almost everything moves.

<div class="ex">
<div class="lab">Worked example</div>

**a cache stampede from modulo hashing**

A cache tier of 9 servers absorbs 95% of 200,000 reads per second, so the database sees 10,000 per second, which is comfortable.

One cache server dies. With `hash % 8` instead of `hash % 9`, about 8 / 9 ≈ 89% of keys now map to a different server, where they are not cached. The hit rate drops from 95% to roughly 10% until caches refill.

The database briefly receives about 180,000 reads per second, eighteen times its normal load. It slows down, requests time out, and the cache can't refill because the database can't answer. One server failure has become an outage.

</div>

## The hash ring

**Consistent hashing** places both servers and keys on a circle of hash values (for example 0 to 2³² − 1, wrapping around). Each server is hashed (by name or address) to a position on the ring. A key belongs to the **first server found by walking clockwise** from the key's position.

- **Adding a server** only takes the keys between it and the previous server counter-clockwise. On a ring of 0 to 100 with servers at 15, 50 and 75, adding a server at 90 takes only the keys from 75 to 90 (previously owned by the server at 15, which wrapped round): about 15% of the key space, instead of 75%.
- **Removing a server** hands only its keys to the next server clockwise. Every other key stays where it was.

On average, only about **1/N of keys move** when one server joins or leaves, which is the minimum possible: the new server has to get its fair share from somewhere.

~~~dg-compare Adding a fourth server. Modulo hashing reshuffles most keys; a hash ring moves only the slice the new server takes over.
[Modulo: hash % N]
3 servers -(add 1)-> 4 servers -> 75% of keys move
- Cache hit rate collapses
- Same on any server failure
[*Hash ring]
3 servers -(add 1 at 90)-> 4 servers -> ~25% move, all to the new server
+ Other servers keep their keys
+ Failure moves only the dead server's keys
~~~

~~~dg-bars Share of keys that move when a fifth server joins a pool of four. Modulo moves almost everything; a ring moves only the new server's fair share.
hash % N | 80 | 80%
*Hash ring | 20 | 20%, the minimum possible
~~~

## Virtual nodes

With only a few servers, their positions on the ring are random, so the gaps between them are uneven. One server may own 40% of the ring and another 10%. And when a server leaves, its entire slice lands on one neighbour, doubling that neighbour's load.

The fix is to place each physical server at **many points** on the ring, called **virtual nodes** (vnodes): for example 100 to 200 points per server, each at the hash of "server-A#1", "server-A#2", and so on. Each server now owns many small slices scattered around the ring:

- **Even ownership**: the sizes of many small slices average out. With 100 or more vnodes per server, each server's share is typically within about 10% of the mean.
- **Spread-out failover**: when a server leaves, its many small slices go to many different neighbours, so the extra load is shared across the whole pool.
- **Weighting**: a server with twice the memory gets twice as many vnodes and so twice the keys.

The cost is a bigger lookup table (servers × vnodes positions, kept sorted so the lookup is a binary search), which is still tiny: 50 servers × 200 vnodes is 10,000 entries.

### Variations in real systems

- **Redis Cluster** splits the key space into **16,384 fixed hash slots** (`CRC16(key) % 16384`) and assigns ranges of slots to nodes. Rebalancing means moving whole slots between nodes. The client keeps a slot-to-node map and is redirected (`MOVED`) if it is out of date. Fixed slots give the same benefit as a ring with explicit control over what moves.
- **Cassandra** and **DynamoDB** use rings with virtual nodes to place data partitions on storage nodes.
- **Load balancers** such as NGINX (`hash $key consistent`), HAProxy and Envoy (ring hash and Maglev) offer consistent hashing for routing requests.

~~~dg-tree A Redis Cluster slot map. Keys hash to one of 16,384 slots, and each node owns ranges of slots that can be moved one at a time.
Key space: 16,384 slots
  Node A | slots 0 to 5460
  Node B | slots 5461 to 10922
    *Slot 7000 moving to Node D
  Node C | slots 10923 to 16383
  Node D | new, receiving slots
~~~

## The hot-key problem

Consistent hashing balances **keys**, not **traffic**. If one key is enormously popular (a celebrity's profile, a viral video, the front page during a big news event), its server is overloaded while others are idle. Every other key that happens to live on that server suffers too: their requests queue behind the hot key's, so latency rises for cold keys that did nothing wrong.

Ways to handle hot keys:

- **Replicate hot keys** to several servers and spread reads across the copies. Writes must update all copies, so this suits read-heavy hot keys.
- **Salt the key**: store it under `video123#1` to `video123#8`, so it maps to eight positions on the ring; readers pick a suffix at random. Writers update all eight.
- **A local cache in front**: each application server keeps the hottest few hundred keys in its own memory for a second or two, so most reads for the hot key never reach the cache tier at all.
- **Monitor and rebalance adaptively**: detect keys whose request rate crosses a threshold and apply the above automatically.

<div class="ex">
<div class="lab">Worked example</div>

**salting a viral key**

A cache server can handle about 20,000 reads per second. A video goes viral and its metadata key receives 60,000 reads per second, all to one server.

Salting into 8 copies spreads that to 7,500 reads per second per copy, on (most likely) 8 different servers, each comfortably under capacity alongside its normal traffic.

The cost: an update to the video's metadata now writes 8 keys, and a reader may briefly see an old copy while the writes land. For view counts and titles, that's fine.

</div>

## Consistent hashing with bounded loads

Replication and salting need you to know which keys are hot. A refinement developed at Google gives a hard guarantee instead: **cap each server's load**. Compute the average load per server and set a capacity of `(1 + ε)` times that average, for example ε = 0.25 means 125% of average. A request goes to its key's usual server unless that server is at its cap, in which case it walks clockwise to the next server with room.

~~~dg-flow Bounded loads: if the key's own server is already at its cap, the request walks on round the ring to the next server with room.
Request for key k -(hash ring)-> Server B at 125% cap -(full, skip)-> Server C at 90% -(under cap)-> *Served by C
~~~

This guarantees that no server exceeds the cap, while keeping most keys on their usual server: overflow only happens to requests that would have overloaded someone. The parameter ε trades the two goals: a small ε keeps load very even but moves more requests away from their home server (more cache misses); a large ε keeps more keys at home but allows more imbalance. When servers are added or removed, the expected number of moved keys grows as ε shrinks, roughly in proportion to 1/ε².

Google uses it in Cloud Pub/Sub. Vimeo added it to HAProxy for routing video requests to cache servers, and cut the bandwidth of their busiest cache servers by almost 8 times, because popular videos no longer overloaded single servers.

<div class="ex">
<div class="lab">Worked example</div>

**an image cache tier**

10 cache servers sit in front of object storage, chosen by consistent hash of the image ID with 150 virtual nodes each. Each server holds about 10% of images.

An 11th server is added. About 1/11 ≈ 9% of images move to it; the other 91% keep hitting warm caches. With modulo hashing, about 91% would have moved.

A viral image would overload one server. With bounded loads at ε = 0.25, once that server reaches 1.25 times the average load, extra requests spill to the next server on the ring, which fetches the image once and then serves it from its own cache. The hot image ends up warm on two or three servers, and no server goes over 125%.

</div>

## Common mistakes

- **`hash % N` for anything stateful**, which reshuffles everything on every change.
- **Too few virtual nodes**, leaving one server with twice its share.
- **Assuming even keys means even load**: hot keys break that.
- **A poor hash function** (or hashing sequential IDs with something weak), which clusters keys.
- **Clients with different ring views**, for example one app server that hasn't learned about a new node, so the same key goes to two places. Distribute membership through one source of truth.

## In an interview

Whenever a design has a cache tier, sharded storage or sticky stateful servers, say "consistent hashing with virtual nodes, so adding or losing a node moves only about 1/N of keys". Then raise hot keys yourself, and offer replication, salting, a local cache or bounded loads as the fix.

<div class="take">
<div class="lab">Key takeaway</div>

route a key to the same server to keep its cache warm or its state in one place. Use consistent hashing so membership changes move only about 1/N of keys, virtual nodes to even out ownership and spread failover, and replication, salting or bounded loads for keys that are hot.

</div>
