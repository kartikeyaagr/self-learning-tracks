---
id: "05.02"
course: "system-design-101"
module: "05"
slug: "distributed-cache-design"
title: "Distributed cache design"
keyIdeas: ["A distributed cache spreads keys across many nodes so capacity and throughput scale beyond one machine.", "Clients (or a proxy) pick the node by consistent hashing on the key.", "Replicate for availability, or accept that a node loss just causes misses that fall through to the database.", "Facebook's memcache paper: leases to prevent stale sets and thundering herds, regional pools for different key popularity, invalidations driven from the database commit log (mcsqueal).", "Redis adds data structures (lists, sorted sets, hashes) and optional persistence; Memcached is simpler and multithreaded."]
question: "How does a client decide which cache node holds a key?"
answer: "The client hashes the key and maps the hash to a node using a scheme every client shares. With consistent hashing it places the key on a hash ring and picks the first node clockwise, so adding a node moves only about 1/N of keys. With Redis Cluster it computes CRC16(key) mod 16,384 to get a hash slot, looks up which node owns that slot in its cached slot map, sends the request there directly, and refreshes the map when a node replies MOVED. Alternatively a proxy such as mcrouter or Twemproxy does this hashing on the client's behalf, at the cost of an extra hop."
sources: [{"title": "Scaling Memcache at Facebook (paper)", "url": "https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf", "note": "Leases, regional pools, invalidation."}, {"title": "Hello Interview: Redis deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/redis", "note": "Redis as a distributed cache."}]
tools: ["Redis"]
---

## Spreading a cache across machines

One cache server eventually runs out of something. A large Redis or Memcached box might have 64 to 256 GB of RAM, a 10 to 25 Gbit/s network card and enough CPU for a few hundred thousand operations per second. A cache for a big product can need terabytes of memory and millions of operations per second. So large caches are **partitioned** (split) across many nodes. Each key lives on exactly one node (plus replicas, sometimes), and each node holds only its share.

That raises the core question of this lesson: **routing**. Given a key, which node has it? The answer has to be computed quickly (it happens on every request), agreed by every client (otherwise one client writes to node 3 and another reads from node 7), and stable when nodes are added or removed (otherwise every topology change empties the cache).

There are three places the routing decision can live: in the client, in the cache cluster itself, or in a proxy in between.

~~~dg-tree Three places the "which node has this key?" decision can live, with a real system for each.
Who routes a key?
  The client library
    *Consistent hash ring | Memcached at Facebook
  The cache cluster
    Hash slots plus MOVED replies | Redis Cluster
  A proxy in between
    Proxy hashes for the client | mcrouter, Twemproxy, Envoy
~~~

## Why plain modulo hashing fails

The naive approach is `node = hash(key) mod N`, where N is the number of nodes. It spreads keys evenly and needs no shared state. The problem appears when N changes. Going from 10 to 11 nodes changes the result of `mod N` for almost every key: only about 1 in 11 keys stays on the same node. So adding one node turns roughly 91% of your cache into misses at the same moment, and the database takes the full load.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of adding a node with modulo hashing**

A cache tier serves 500,000 reads per second at a 95% hit rate, so the database sees 25,000 reads per second.

You add an eleventh node using `hash mod N`. About 10/11 of keys now map to a different node, so the effective hit rate drops to roughly 95% × 1/11 ≈ 9% until the cache refills.

The database suddenly sees about 455,000 reads per second, eighteen times its normal load. It will not survive. With consistent hashing, only about 1/11 of keys move, so the hit rate dips to roughly 86% and the database sees about 70,000 reads per second for a short while: painful, but survivable with headroom.

</div>

## Option 1: clients hash the key (consistent hashing)

**Consistent hashing** (lesson 04.07 covers it in full) fixes the remapping problem. Imagine the hash space as a ring from 0 to 2^32. Each node is placed at several points on the ring (often 100 to 200 **virtual nodes** per server, to even out the spread). To find a key, hash it onto the ring and walk clockwise to the first node point. Adding a node only takes over the arcs just before its points, so only about 1/N of keys move.

Facebook's memcache deployment is the classic example of putting all the routing intelligence in the client. Its Memcached servers are independent, single-machine hash tables that never talk to each other. They do not know a cluster exists. The client library does everything:

- Hashes the key with consistent hashing to pick a server among hundreds.
- Serialises and compresses values, batches requests, and handles errors.
- Sends gets over **UDP** for speed (about 20% lower latency than TCP; a dropped or late packet is simply treated as a cache miss). Sets and deletes go over TCP through a local proxy called **mcrouter**, which pools connections so each web server does not hold thousands of TCP connections.

The scale is large: billions of requests per second over trillions of items. A popular page fetches hundreds of items (521 distinct items on average, 1,740 at the 95th percentile). Clients build a dependency graph of the keys a page needs and fetch independent keys in parallel batches (24 keys per request on average). They also use a **sliding window** to limit how many requests are outstanding, avoiding **incast congestion**: when a client asks 100 servers at once, all the replies arrive in the same instant and overflow the switch buffer, causing packet loss and retries.

~~~dg-flow Client-side routing: the client library hashes the key onto the ring and talks to the owning node directly, with no extra hop.
[Read] App -(hash key on ring)-> Client library -(UDP get)-> *Memcached node 7
[Write] App -(TCP set or delete)-> mcrouter -(pooled TCP)-> *Memcached node 7
~~~

### Facebook's other structures

Routing is only the first layer. Facebook's design adds several more ideas that are worth knowing.

- **Pools** separate workloads so they do not evict each other. Keys that change often and are cheap to recompute go in one pool; keys that are expensive to miss go in another. Without pools, a burst of cheap high-churn keys would push out the expensive ones.
- **Replication within a pool** for small, very hot key sets. If one server holding 100 keys is receiving 500,000 requests per second, splitting the keys across two servers does not help, because clients batch all 100 keys into one request either way. Replicating the whole set to two servers and letting each client pick one halves the per-server load.
- **Clusters and regions.** Several frontend clusters (each with its own web servers and memcache) share one database tier. When the database commits a write, a daemon called **mcsqueal** tails the commit log and sends deletes to every cluster. Because invalidations are driven by the log, they survive crashes and can be replayed, unlike deletes sent by web servers that might die mid-request.
- **Cold cluster warmup.** A new cluster with an empty cache would hammer the database. Instead, its clients fetch misses from a warm cluster's cache and fill their own. This cut warmup from days to a few hours.

## Option 2: the cache cluster routes for you (Redis Cluster)

Redis Cluster takes a different approach: the nodes know about each other and the cluster owns the mapping. The key space is split into **16,384 hash slots**. Every key maps to one slot with `CRC16(key) mod 16384`, and each primary node owns a set of slots. A three-node cluster might give slots 0 to 5,460 to node A, 5,461 to 10,922 to node B and 10,923 to 16,383 to node C.

How a request finds its node:

1. On startup, the client asks any node for the slot map (`CLUSTER SHARDS` or the older `CLUSTER SLOTS`) and caches it.
2. For each command, the client computes the slot and sends the request straight to the owning node. No extra hop.
3. If the slot has moved permanently, the node replies `MOVED 9301 10.0.0.12:6379`. The client retries there and refreshes its map.
4. If the slot is being migrated right now, the node may reply `ASK`, meaning "try the other node for this one request, but do not update your map yet".

Nodes do **not** forward requests on the client's behalf; redirecting keeps the nodes simple and fast. Nodes share the slot map and health information with each other over a **gossip** bus (lesson 06.11), and replicas take over automatically when a primary fails.

~~~dg-seq Redis Cluster redirect: a client with an outdated slot map is told where the slot now lives.
Client > Node C : GET user:42:profile (slot 9301)
Node C -> Client : MOVED 9301 to Node D
Client > Client : update slot map
Client > Node D : GET user:42:profile
Node D -> Client : value
~~~

### Hash tags and multi-key commands

Commands that touch several keys (`MGET`, transactions, Lua scripts) only work when every key is in the same slot, because a single node must execute them atomically. **Hash tags** let you force related keys together: if a key contains `{...}`, only the part inside the braces is hashed. So `user:{42}:profile` and `user:{42}:settings` both hash `42` and land in the same slot. Be careful: a hash tag that is too coarse (say `{global}`) puts everything on one node and recreates the single-server problem.

### Resharding

Adding a node means moving whole slots to it. Redis moves a slot key by key with `MIGRATE`, answering `ASK` for keys already moved. Because the unit of movement is a slot (on average 1/16,384 of the data), resharding is gradual and the cache stays warm for everything not in flight.

## Option 3: a proxy routes

A **proxy** such as mcrouter, Twemproxy or Envoy sits between the applications and the cache nodes. Applications send every request to the proxy (often running on the same host as a sidecar), and the proxy does the hashing, connection pooling, retries and failover.

~~~dg-compare Client-side routing versus a proxy. The proxy costs a hop but puts topology changes in one place.
[Smart client]
App -(hash in library)-> Cache node
+ No extra hop, lowest latency
+ No extra component to run
- Every language needs the library
- Topology changes must reach all clients
[*Proxy or sidecar]
App -> Proxy -(hash, pool, retry)-> Cache node
+ Clients stay simple, any language
+ Pools connections, fewer sockets per node
- Extra hop, about 0.1 to 0.5 ms
- Proxy fleet must scale and stay up
~~~

A proxy really earns its place when you have many services in many languages, or thousands of application processes that would otherwise each open a connection to every cache node. 5,000 processes × 100 cache nodes is 500,000 connections; with a local proxy per host (say 500 hosts), it is 50,000.

## Replication and durability

A cache node can fail. Two questions follow: is the data replicated, and is it persisted?

Redis replicates **asynchronously**: the primary acknowledges a write before replicas have it. If the primary crashes, a replica is promoted, and any writes it had not received yet are lost. Persistence is optional. **RDB snapshots** write the whole dataset to disk periodically and lose everything since the last snapshot on a crash. The **AOF** (append-only file) logs every write; with the default `appendfsync everysec` it can lose about a second of writes. So Redis is not a system of record by default. For a cache that is fine, because the database has the truth and a lost key is just a miss.

Redis executes commands on a **single thread**, one at a time. That makes each individual command atomic without locks (an `INCR` can never be interleaved with another `INCR` on the same key) and lets one node handle on the order of 100,000 operations per second, more with pipelining. Network I/O can use extra threads in recent versions, but command execution stays single-threaded, so one slow command (`KEYS *` on 50 million keys) blocks everyone.

## Hot keys in a distributed cache

Partitioning spreads keys, not load. One key can overload its node no matter how many nodes you have. A celebrity's profile or a flash-sale product might get 1 million reads per second; a node handling 100,000 cannot serve it.

<div class="ex">
<div class="lab">Worked example</div>

**spreading a hot key**

A key `product:123` receives 600,000 reads per second during a sale. One Redis node manages about 100,000.

Option A, an in-process cache on each of 300 app servers with a 1 second TTL. Each server sends at most one read per second to Redis for that key, so Redis sees about 300 reads per second. Staleness is at most 1 second.

Option B, ten copies `product:123:0` to `product:123:9`, which hash to different nodes. Readers pick a copy at random, so each node sees about 60,000 reads per second. Writers must update or delete all ten copies.

Option C, read from replicas. With one primary and five replicas taking reads, each sees about 100,000 per second, right at the limit, so this alone is not enough here.

</div>

You also need to **detect** hot keys: Redis has `redis-cli --hotkeys` (with an LFU policy), and many teams sample client requests and count top keys over a sliding window.

## Common mistakes

- **Modulo hashing** in the client, so every scaling event wipes the cache.
- **Clients with different node lists** during a deploy, so the same key is read from one node and written to another.
- **Multi-key operations across slots** in Redis Cluster, which fail with `CROSSSLOT` errors.
- **Treating Redis as durable** without understanding that asynchronous replication can lose acknowledged writes.
- **Assuming more nodes fixes a hot key.** It does not; only copying the key or caching it closer does.

## In an interview

Say how routing works in one sentence ("the client hashes the key onto a consistent hash ring" or "Redis Cluster's 16,384 slots"), then mention what happens when a node is added or fails, and how you handle hot keys. Asking "what is the read rate on the hottest key?" shows you know partitioning alone is not the whole story.

<div class="ex">
<div class="lab">Worked example</div>

**routing a lookup end to end**

The app needs `user:42:profile`. With Redis Cluster, the client computes CRC16("user:42:profile") mod 16384, which gives (say) slot 9,301. Its cached map says slots 8,192 to 12,287 belong to node C, so it sends GET to node C over an existing connection: about 0.5 ms round trip.

During a rebalance, node C answers MOVED 9301 to node D. The client refreshes its map, retries on D and gets the value. One request paid an extra round trip; every later request for slot 9,301 goes straight to D.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

distributed caches route each key to one node by hashing it, either in the client (a consistent hash ring), in the cluster (Redis's 16,384 hash slots with MOVED redirects) or in a proxy. Use schemes that move only a small share of keys when nodes change, invalidate from the database's commit log, and plan separately for hot keys.

</div>
