---
id: "04.07"
course: "system-design-101"
module: "04"
slug: "consistent-hashing"
title: "Consistent hashing"
keyIdeas: ["Consistent hashing maps both servers and keys onto a ring (hash space). A key belongs to the first server clockwise from it.", "Adding or removing a server only moves the keys in its section of the ring, about K/N keys, instead of nearly all keys with `hash % N`.", "Example: with modulo, going from 10 to 11 servers moves about 90% of keys. With consistent hashing, about 9%.", "Virtual nodes: each server takes many points on the ring, which evens out load and spreads a failed server's keys across many others.", "Used in Dynamo, Cassandra, Memcached clients and CDNs."]
question: "With modulo hashing, adding 1 server to 10 moves how many keys? With consistent hashing?"
answer: "With modulo hashing, going from 10 to 11 servers changes hash % N for about 10/11 of keys, so roughly 91% of keys move to a different server, and a cache's hit rate collapses. With consistent hashing, only the keys in the arcs taken over by the new server move, about 1/11, roughly 9% of keys, while every other key stays where it was; with virtual nodes those 9% come in small slices evenly from all 10 existing servers instead of from one neighbour."
sources: [{"title": "ByteByteGo: Design Consistent Hashing", "url": "https://bytebytego.com/courses/system-design-interview/design-consistent-hashing", "note": "Ring, virtual nodes, rebalancing."}, {"title": "Hello Interview: Consistent Hashing", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/consistent-hashing", "note": "Interview version."}]
tags: ["Code"]
---

## The rehashing problem

You have N cache servers (or database shards) and need every client to agree which server holds each key, without asking anyone. The simple formula is:

```
server = hash(key) % N
```

A good hash spreads keys evenly, every client computes the same answer, and lookups cost nothing. It works perfectly until **N changes**.

Take 4 servers and 8 keys. Removing one server changes the formula to `% 3`, and for most keys the remainder changes: in a typical example, 6 of the 8 keys now map to a different server. Adding a server has the same effect. For a cache, that is a disaster: most lookups suddenly go to a server that does not have the key, the cache **hit rate collapses**, and the full read load falls on the database behind it, often at the worst moment (you were adding a server because load was high). For a sharded database it is worse: most rows would have to be physically moved.

### How many keys move?

With modulo hashing, a key stays put only if `hash % N` and `hash % (N+1)` happen to agree, which is true for about 1 in N+1 keys. So going from N to N+1 servers moves about **N/(N+1)** of all keys. From 10 to 11 servers, about 10/11, or **91%**, of keys move.

The ideal is that only the new server's fair share moves: it should end up with 1/11 of the keys, so about **9%** should move to it, all other keys should stay where they are, and the 9% should come evenly from all existing servers.

~~~dg-bars Share of keys that change server when growing from 10 to 11 servers.
Modulo hashing | 91 | about 91% move
*Consistent hashing | 9 | about 9% move
~~~

## The hash ring

**Consistent hashing** gets close to that ideal. The construction:

1. Take the output range of a hash function (for a 32-bit hash, 0 to 2^32 − 1; for SHA-1, 0 to 2^160 − 1) and imagine its two ends joined, forming a **ring**.
2. Hash each **server** (its name or IP address) to a position on the ring.
3. Hash each **key** to a position on the same ring. The key belongs to the **first server clockwise** from its position.

Each server therefore owns the **arc** of the ring between the previous server (counter-clockwise) and itself.

**Adding a server**: it lands somewhere on the ring and takes over only the keys between it and the previous server counter-clockwise. Those keys used to belong to the next server clockwise. Every other key stays where it was.

**Removing a server**: only its keys move, all to the next server clockwise. Everything else stays put.

To find which keys are affected by a change, walk counter-clockwise from the changed server to the previous one; only keys in that arc move. Try it below: add and remove servers and watch which keys change owner.

~~~dg-widget ring Each key belongs to the first server clockwise. Add or remove a server and only the keys in one arc move; more virtual nodes per server even out the load.
~~~

### How a lookup works in code

In practice the ring is a **sorted array** of server positions. To find a key's server, hash the key and **binary search** for the first position greater than or equal to it, wrapping around to the first entry if you fall off the end. With 1,000 positions, that is about 10 comparisons: well under a microsecond. Every client holds the same array, built from the same server list, so they all agree without any coordination.

## Two problems with the basic ring

1. **Uneven arcs.** Servers are placed at pseudo-random positions, so some arcs are much longer than others. With few servers, one may own 3 times the share of another. Worse, when a server is removed, its whole arc goes to **one** neighbour, which can suddenly double that neighbour's load.
2. **Uneven key distribution.** Even with equal arcs, keys can cluster by chance on one arc.

### The fix: virtual nodes

Give each physical server many positions on the ring, called **virtual nodes** (vnodes), for example by hashing `"cache-07#0"`, `"cache-07#1"` up to `"cache-07#199"`. Each server now owns many small arcs scattered around the ring, and its total share is the sum of those arcs, which averages out close to 1/N.

~~~dg-tree Virtual nodes: each physical server appears at many scattered points on the ring, so its share is the sum of many small arcs.
Ring positions, sorted
  cache-01
    vnode at 0.03 | owns arc 0.01 to 0.03
    vnode at 0.41
    vnode at 0.77 | ... 150 vnodes in all
  cache-02
    vnode at 0.12
    vnode at 0.56
  *cache-11, new
    vnode at 0.09 | takes keys from cache-02's arc
    vnode at 0.63 | takes keys from cache-05's arc
~~~

The effects:

- **Balance improves with more vnodes.** With 100 virtual nodes per server, the standard deviation of load is about 10% of the mean; with 200, about 5%.
- **Moves are spread out.** When a server leaves, its many small arcs go to many different neighbours, so the load spreads across the whole cluster instead of doubling one machine. When a server joins, it takes small slices from every existing server.
- **Heterogeneous hardware.** A machine with twice the memory can be given twice as many vnodes and so receives about twice the keys.

The cost is metadata: 100 servers × 200 vnodes is a sorted array of 20,000 entries, which is trivial for memory and lookup speed but means more ranges to track when data is physically moved (as in Cassandra, where each vnode's range is streamed separately).

## Replication on a ring

Distributed databases use the same ring to place **replicas**. Dynamo-style systems such as Cassandra store each key on the first server clockwise and then on the next **N − 1 distinct physical servers** clockwise (skipping further vnodes of servers already chosen). With a replication factor of 3, each key lives on three different machines, and losing one machine leaves two copies. Cassandra can also skip servers in the same rack or data centre so that replicas survive a rack failure.

## Where it is used

Consistent hashing is used by Amazon's Dynamo and by Cassandra and ScyllaDB for data partitioning, by Discord to route users to chat servers, by Akamai's CDN to pick cache servers, by Memcached client libraries (the "ketama" scheme), and in load balancers such as Google's Maglev, which uses a variant designed for fast lookup tables. Lesson 02.07 covers **bounded loads**, a variant that caps how much more than average any single server can receive, and how to handle hot keys on a ring by replicating or salting them (lesson 04.06).

## Alternatives worth knowing

~~~dg-matrix Ways to map keys to servers, compared on how many keys move when a server is added and what each costs.
Scheme | Keys moved on add | Even load | Lookup cost | Notes
hash mod N | about N/(N+1) | yes | O(1) | breaks on resize
Ring, no vnodes | about 1/N | no | O(log N) | arcs uneven
*Ring with vnodes | about 1/N | yes | O(log V) | widely used
Rendezvous hashing | about 1/N | yes | O(N) | no ring needed
Jump hash | about 1/N | yes | O(log N) | servers numbered 0 to N-1 only
Fixed slots, e.g. 16,384 | slots moved explicitly | yes | O(1) | moves are planned
~~~

- **Fixed logical shards (slots)**: create far more shards than machines and move whole shards between machines when the cluster changes. Redis Cluster uses 16,384 hash slots; Notion uses 480 logical shards (lesson 03.10). It is simple to reason about, a slot-to-server table is tiny, and moves are explicit and controlled.
- **Rendezvous hashing** (also called highest random weight): for each key, compute a score `hash(key, server)` for every server and pick the highest. Removing a server only affects keys for which it scored highest, so about 1/N of keys move, with no ring at all. The cost is scoring every server on every lookup, which is fine for tens of servers.
- **Jump consistent hash**: a tiny algorithm that maps a key to one of N numbered buckets with minimal movement and no memory, but it only supports adding or removing the **highest-numbered** bucket, so it suits storage shards more than caches with arbitrary failures.

<div class="ex">
<div class="lab">Worked example</div>

**growing from 10 to 11 cache servers**

A cache holds 100 million keys across 10 servers and serves 200,000 reads per second at a 95% hit rate, so the database sees 10,000 reads per second.

With modulo hashing, about 91 million keys change server. For the next several minutes the hit rate drops towards 9%, and the database suddenly receives about 180,000 reads per second, 18 times its normal load. It falls over, and the cache cannot refill because the database is down.

With a ring and 150 virtual nodes per server, the new server takes about 9 million keys, roughly 900,000 from each existing server. The other 91 million stay put. The hit rate dips from 95% to about 86% and recovers as the new server warms up; the database sees about 28,000 reads per second for a few minutes. Annoying, not fatal.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**what happens when one server dies**

A ring of 4 cache servers with no virtual nodes has arcs of 40%, 25%, 20% and 15% by chance. The 40% server dies. All of its keys move to its clockwise neighbour, the 25% server, which now owns 65% of keys and receives 65% of traffic. It overloads and dies too, and its 65% moves to the next server. This is a **cascading failure**.

With 200 virtual nodes per server, each server owns about 25% ± 1.5%. When one dies, its 200 small arcs are spread across the other three, each going from about 25% to about 33%. A 33% increase is something to provision for; it does not cascade.

</div>

## Common mistakes

- **Using modulo hashing for anything that will be resized**, especially caches in front of a database.
- **A ring without virtual nodes**, which gives uneven load and cascading failure on node loss.
- **Clients with different server lists**, which then disagree about key ownership. The membership list must be distributed consistently (a configuration service, or gossip).
- **Forgetting hot keys.** Consistent hashing balances the number of keys, not the traffic per key.

## In an interview

When you introduce a cache cluster or shards, say how keys map to nodes: "consistent hashing with virtual nodes, so adding a node moves only about 1/N of keys, spread evenly", or "a fixed number of slots moved explicitly". Give the numbers for one change, and mention that replicas go to the next distinct nodes on the ring.

<div class="take">
<div class="lab">Key takeaway</div>

modulo hashing remaps almost every key when the server count changes, which can collapse a cache and overload the database behind it. A hash ring remaps only about 1/N of keys, those in one arc, and virtual nodes make both the steady-state split and the redistribution after a change even.

</div>
