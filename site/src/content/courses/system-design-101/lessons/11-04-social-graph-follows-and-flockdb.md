---
id: "11.04"
course: "system-design-101"
module: "11"
slug: "social-graph-follows-and-flockdb"
title: "Social graph: follows and FlockDB"
keyIdeas: ["FlockDB was Twitter's store for the social graph: who follows whom, blocks, and similar adjacency lists.", "It stored each edge twice, once in the forward direction (A follows B) and once backward (B is followed by A), so \"who do I follow\" and \"who follows me\" were both a single indexed range scan on one shard.", "Built on sharded MySQL with an edge state (normal, removed, archived) and supported fast set operations (intersection, e.g. mutual follows).", "It only supported one-hop queries, which was enough for Twitter's needs. No multi-hop graph traversal."]
question: "Why did Twitter store each edge twice (forward and backward)?"
answer: "Because the graph was split across many shards by each row's first ID, a lookup is only cheap if all the rows it needs sit together on one shard. Storing the forward edge with the source and the backward edge with the destination makes both \"whom does X follow\" and \"who follows X\" single-shard, sorted, pageable reads. With a single copy, one of those two common queries, including the follower lookup needed to fan out every tweet, would have to scatter to every shard and wait for the slowest. The cost is double storage and two writes per change, which is cheap because follows are written far less often than they are read, and the copies are kept consistent with timestamped, idempotent writes that are retried until both succeed."
sources: [{"title": "twitter-archive/flockdb", "url": "https://github.com/twitter-archive/flockdb", "note": "Twitter's graph store; README explains the design."}]
---

## What FlockDB was

FlockDB is the distributed graph database Twitter built to store its social graph: who follows whom, who blocks whom, and similar relationships. It (now archived) stored **adjacency lists** for high-throughput, low-latency online use, on top of ordinary MySQL databases, with a sharding framework called **Gizzard** in front. As of April 2010, Twitter's cluster held **over 13 billion edges** and sustained peaks of **20,000 writes per second** and **100,000 reads per second**.

It is worth studying not because you will run FlockDB, but because its design is the standard answer to "how do I store a huge social graph", and every choice in it is a deliberate trade-off you can reuse.

## Goals and non-goals

It was explicit about scope:

- **Goals**: a high rate of add, update and remove operations; potentially complex **set arithmetic** queries; **paging** through result sets with millions of entries; archiving edges and restoring them later; horizontal scaling including replication; online data migration.
- **Non-goals**: **multi-hop graph traversal** ("friends of friends of friends") and automatic shard migration.

It described itself as much simpler than general graph databases such as Neo4j, because it solves fewer problems. Twitter's online graph queries are almost all one hop: "who follows X?", "does A follow B?", "whom do A and B both follow?". A multi-hop traversal on a graph of billions of edges touches an unpredictable number of shards and can take seconds; a one-hop query touches one sorted list. Choosing not to support multi-hop queries is what let the system be fast, predictable and easy to shard.

## The edge model

Each edge has:

- a 64-bit **source ID** and 64-bit **destination ID**;
- a **state**: normal, removed, or archived;
- a 32-bit **position** used for sorting. For a follow ("Nick follows Robey"), the position is the time the follow happened, so follower lists come back newest first.

The (source, destination) pair is unique; state and position can change. Note what is missing: there are no properties, labels or weights beyond position. An edge is a fact that A relates to B, with a state and a sort key. Different relationship types (follows, blocks, mutes) are separate graphs.

### Edge states

Deleting a row is a poor fit for a distributed system with retries and replication: a delayed "add" that arrives after a "delete" would resurrect the edge, and a deleted row leaves no trace for auditing. So FlockDB never hard-deletes in normal operation. It changes state, and every change carries a timestamp, so a later operation always wins regardless of arrival order.

~~~dg-state The lifecycle of one edge: unfollowing marks it removed, suspending an account archives its edges, and reinstatement restores them.
[*] -> *Normal
Normal -> Removed : unfollow or unblock
Removed -> Normal : follow again
Normal -> Archived : account suspended or deactivated
Archived -> Normal : account restored
~~~

**Archived** edges go into "cold sleep". When an account is suspended, its thousands of edges are archived rather than deleted, so they vanish from every list and count. If the account is reinstated, they are restored in one operation, and the person gets their followers back.

## Stored twice, forward and backward

Edges are stored **in both directions**, so they can be queried by source or by destination. For example, node 134 follows node 90. That produces a forward row (134 → 90) and a backward row (90 ← 134).

~~~dg-flow FlockDB writes each edge twice so both directions are a single-shard lookup.
"134 follows 90" -(forward row, stored with 134)-> *134: 90, ... -(backward row, stored with 90)-> *90: 134, ...
~~~

Why twice? Because of sharding. At billions of edges, the graph is split across many MySQL databases. Gizzard decides which shard a row lives on from its **first ID**, through a forwarding table that maps ranges of IDs to shards:

- The forward copy of (134 → 90) has first ID 134, so it sits on 134's shard.
- The backward copy (90 ← 134) has first ID 90, so it sits on 90's shard.
- "Whom does 134 follow?" reads only 134's forward rows, on one shard, already sorted.
- "Who follows 90?" reads only 90's backward rows, on one shard, already sorted.

~~~dg-tree Where the two copies of "134 follows 90" live: each copy is placed by its first ID, so each direction of lookup reads one shard.
Gizzard forwarding table | ID range to shard
  Shard 7, IDs 100 to 199
    *forward 134 → 90 | whom 134 follows
    forward 134 → 412
  Shard 3, IDs 0 to 99
    *backward 90 ← 134 | who follows 90
    backward 90 ← 57
~~~

With a single copy, one of those queries would have to visit **every** shard. For someone with 50 million followers, scanning every shard for each tweet's fan-out would be impossible at 100,000 reads per second: each tweet would become hundreds of shard queries, most finding a handful of rows, and the slowest shard would set the latency of every delivery.

The cost is double the storage and two writes per change, which must be kept consistent. Since follows change far less often than they are read (20,000 writes per second at peak against 100,000 reads, and the reads are much heavier, often returning thousands of rows), the trade is clearly worth it.

## How writes stay consistent

Two copies on two shards, written by a distributed system with failures, will sometimes disagree. Gizzard's approach is simple and worth copying: **every write must be idempotent and commutative**.

- **Idempotent**: applying the same operation twice gives the same result. "Set edge (134, 90) to normal at time T" can be retried safely.
- **Commutative**: operations can be applied in any order and still converge. Because each operation carries a timestamp and the newer state wins, "remove at T2" and "add at T1" give "removed" whichever arrives first.

With those two properties, the system doesn't need distributed transactions. Writes go to both shards (and their replicas); any that fail go to an **error queue** and are retried later, and the copies converge. A short window where the forward and backward copies disagree is acceptable for a follow graph.

~~~dg-seq An unfollow whose backward write fails at first: the retry queue applies it later, and timestamps make the late write safe.
Graph service > Shard of 134 : set (134 → 90) removed at T2
Shard of 134 -> Graph service : ok
Graph service > Shard of 90 : set (90 ← 134) removed at T2
Shard of 90 -> Graph service : timeout
Graph service > Error queue : retry backward write, T2
Error queue > Shard of 90 : set (90 ← 134) removed at T2
Shard of 90 -> Error queue : ok, T2 newer than T1, applied
~~~

## Set operations and paging

Because each node's adjacency list is stored sorted, one-hop set queries are merges of sorted lists, the same trick as intersecting posting lists in search (lesson 09.03):

- **Intersection**: "people A follows who also follow B" (for "followed by people you know").
- **Difference**: "people who follow me whom I don't follow back".
- **Union**: "everyone either of these two accounts follows".
- **Paging**: return 100 edges at a time ordered by position, with a cursor (the last position seen), so a 50-million-follower list is never loaded at once.

<div class="ex">
<div class="lab">Worked example</div>

**intersecting two sorted lists**

A follows (sorted by ID): 3, 9, 14, 22. B follows: 9, 17, 22. Walk both with two pointers:

3 vs 9: 3 is smaller, advance A. 9 vs 9: match, output 9, advance both. 14 vs 17: advance A. 22 vs 17: advance B. 22 vs 22: match, output 22. A is exhausted: done.

Result: both follow 9 and 22. The cost is proportional to the lengths of the two lists, with no random lookups. For large lists on different shards, each shard streams its sorted list and the merge happens in the query service.

</div>

~~~dg-compare FlockDB's narrow design against a general graph database for a social network's online queries.
[*FlockDB-style adjacency store]
Sorted adjacency lists, each edge twice
+ One-hop queries hit one shard
+ Set arithmetic by merging sorted lists
+ Simple sharding on MySQL
- No multi-hop traversal
- Double storage and writes
[General graph database]
Nodes, edges, properties, traversals
+ Friends of friends, paths, patterns
+ Rich queries in one language
- Traversals cross shards unpredictably
- Hard to shard at billions of edges
~~~

<div class="ex">
<div class="lab">Worked example</div>

**blocking**

A blocks B: write edge (A → B) in the "blocks" graph, forward copy on A's shard and backward copy on B's shard. When B loads A's profile, the service checks the backward rows for B ("who blocks B?") or the forward rows for A, each a one-shard point lookup.

When A unblocks, the edge's state becomes removed rather than being deleted, so the change can propagate safely even if an earlier "block" write is still sitting in a retry queue (the newer timestamp wins), and the history is auditable.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing a FlockDB-style cluster**

13 billion edges stored twice is 26 billion rows. At roughly 30 bytes of row data (two 8-byte IDs, a 4-byte position, state, timestamps) plus MySQL index overhead, budget 60 bytes per row: about 1.6 TB.

Spread over, say, 64 shards with replicas, that's 25 GB per shard, mostly in the buffer pool. 100,000 reads per second is about 1,600 per second per shard; 20,000 edge writes per second is 40,000 row writes (both copies), about 600 per second per shard. Ordinary MySQL numbers, which was the point.

</div>

## Lessons to reuse

- **Restrict the query model** to what the product needs (one hop), and the storage design becomes simple.
- **Store data the way you query it**, even if that means duplicating it.
- **Make writes idempotent and commutative** with timestamps, so retries and reordering are harmless and you can avoid distributed transactions.
- **Soft-delete with states**, so late writes can't resurrect deleted data and accounts can be restored.
- **Paginate everything** that can be large.

## In an interview

If asked to store a social graph at scale, describe the FlockDB layout: edges with (source, destination, state, position), stored forward on the source's shard and backward on the destination's shard, sorted by position so lists are pageable newest first. Explain the trade (double storage and writes, cheap because writes are rare) and how consistency works (timestamped, idempotent writes with a retry queue). Say explicitly that multi-hop queries are out of scope online and computed offline.

<div class="take">
<div class="lab">Key takeaway</div>

FlockDB stored a huge, one-hop social graph as sorted adjacency lists, each edge written twice (forward under the source, backward under the destination) so "who does X follow" and "who follows X" each hit one shard. Timestamped, idempotent, commutative writes with soft-delete states kept the two copies consistent without distributed transactions, and it deliberately skipped multi-hop traversal in exchange for very high throughput.

</div>
