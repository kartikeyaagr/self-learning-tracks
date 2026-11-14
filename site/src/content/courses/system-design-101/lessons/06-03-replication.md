---
id: "06.03"
course: "system-design-101"
module: "06"
slug: "replication"
title: "Replication"
keyIdeas: ["Replication keeps copies on several nodes for availability, durability and read scaling.", "**Single-leader**: all writes go to the leader, which streams changes to followers. Simple. Failover risks losing recent writes if replication is async.", "**Multi-leader**: several nodes accept writes (e.g. one per region). Concurrent writes can conflict and need resolution.", "**Leaderless** (Dynamo-style): clients write to W of N replicas and read from R. If W + R > N, read and write sets overlap, so a read sees at least one copy of the latest write.", "Sync replication = durable but slower and blocked by slow replicas. Async = fast but can lose data on failover."]
question: "With N=3 replicas, why do W=2 and R=2 guarantee you read the latest write?"
answer: "With N = 3 replicas, a successful write was acknowledged by some 2 of them and a read contacts some 2 of them. Two groups of 2 drawn from only 3 replicas cannot be disjoint, because that would need 4 different replicas (2 + 2 = 4 > 3), so by the pigeonhole principle at least one replica is in both groups. That replica holds the latest acknowledged write, and the reader picks the newest version among the answers by its version number or timestamp, so it returns the latest write. The same setting tolerates one replica being down for both reads and writes. Edge cases such as sloppy quorums and concurrent writes mean this is not full linearizability."
sources: [{"title": "Kleppmann notes: Replication", "url": "https://www.cl.cam.ac.uk/teaching/2122/ConcDisSys/dist-sys-notes.pdf", "note": "Section 5: quorums and replication."}, {"title": "System Design Primer: Replication", "url": "https://github.com/donnemartin/system-design-primer#replication", "note": "Master-slave and master-master."}]
---

## Why replicate

**Replication** keeps copies of the same data on several machines, called **replicas**. There are three reasons to do it:

- **Durability and availability**: if one machine dies, the data survives and the service keeps running on the others.
- **Read throughput**: more copies can serve more reads in parallel.
- **Latency**: a copy near the user (in their region) answers faster than one across an ocean.

If data never changed, replication would be trivial: copy it once. The hard part is keeping the copies in agreement while data changes, when messages can be delayed and machines can fail at any moment. There are three main designs, distinguished by **who is allowed to accept writes**: one leader, several leaders, or no leader at all.

~~~dg-compare The three replication designs, by who accepts writes. Each step to the right gains write availability and pays in conflicts.
[*Single leader]
Writes -> Leader -(log)-> Followers
+ No write conflicts
+ Simple to reason about
- Leader is a bottleneck
- Failover needed on leader loss
[Multi-leader]
Writes -> Leader per region -(async)-> Other leaders
+ Local, fast writes everywhere
+ Regions work when cut off
- Concurrent writes conflict
[Leaderless]
Writes -(to all N)-> Replicas -(W acks)-> Done
+ No failover, any node takes writes
- Conflicts and repair needed
- Tunable but subtle guarantees
~~~

## Single-leader replication

One node, the **leader** (or primary), accepts all writes. It records each change in a **replication log** and streams it to the other nodes, the **followers** (replicas, secondaries), which apply the changes in the same order. Reads can go to the leader or to any follower. This is how PostgreSQL, MySQL, MongoDB replica sets and most relational databases work.

### Synchronous or asynchronous

The key choice is when the leader tells the client "your write succeeded".

- **Synchronous** replication waits until followers confirm they have the write. No acknowledged write is lost if the leader dies. But every write pays the round trip to the slowest synchronous follower, and if a synchronous follower is down or slow, writes stall.
- **Asynchronous** replication acknowledges as soon as the leader has written locally. Writes are fast. But if the leader crashes, writes it acknowledged and had not yet shipped are **lost**, and followers serve **stale reads** while they lag (replication lag).
- **Semi-synchronous** is the common middle ground: one follower is synchronous and the rest are asynchronous. Every acknowledged write exists on at least two machines. If the synchronous follower fails, another is promoted to synchronous.

~~~dg-timeline Replication lag in an asynchronous setup. The client is told "done" before followers have the write, so a read from a follower can miss it.
[Leader] 0 ms | Write x = 5 committed locally
[Leader] 1 ms | *Ack sent to client
[Follower 2] 3 ms | Still has x = 4, serves a stale read
[Follower 1] 4 ms | Applies x = 5
[Follower 2] 250 ms | Applies x = 5 after a slow network spell
~~~

### Replication lag in numbers

In a healthy datacenter, lag is usually under a few milliseconds. Under load, during a large batch update, or across regions, it can grow to seconds or minutes. Lag causes the anomalies from lesson 06.02: a user who writes and then reads from a lagging follower does not see their own change (missing read-your-writes); a user whose successive reads hit different followers sees data go backwards (missing monotonic reads).

### Failover

When the leader dies, a follower must be promoted. Done automatically, that means: detect the failure (with a timeout, which can be wrong, lesson 06.01), choose the most up-to-date follower, reconfigure clients to send writes to it, and stop the old leader from accepting writes if it comes back. Each step has pitfalls. With asynchronous replication, the new leader may lack the last writes of the old one; those writes are usually discarded. A false failure detection can produce **two leaders** (split brain, lesson 06.07). GitHub once had an incident where a promoted MySQL follower was behind, and auto-increment IDs it reissued collided with IDs already stored in Redis, exposing some private data to the wrong users.

## Multi-leader replication

Several nodes accept writes, typically one leader per datacenter. Each leader replicates its changes asynchronously to the others. Writes stay local and fast (a few milliseconds instead of a 100 ms round trip to a distant leader), and each datacenter keeps working if the link to the others is cut.

The price is **write conflicts**: two leaders accept changes to the same record at the same time, and neither knows about the other until replication catches up. You need a resolution rule:

- **Last writer wins (LWW)**: keep the write with the highest timestamp. Simple, but it silently discards the other write, and clock skew (06.05) can make the "last" write actually the earlier one.
- **Merge logic**: application code combines the versions (union the items in two shopping carts).
- **CRDTs** (conflict-free replicated data types): counters, sets and text types designed so that concurrent updates always merge to the same result, whatever the order.

Offline-capable apps (calendars, note-taking apps) are multi-leader systems in disguise: each device is a leader while offline.

## Leaderless replication and quorums

In **Dynamo-style** systems (Cassandra, Riak, ScyllaDB, and Amazon's original Dynamo, the ancestor of DynamoDB), there is no leader. The client, or a coordinator node acting for it, sends each write to all **N** replicas responsible for the key, and considers the write successful once **W** of them acknowledge. A read asks **R** replicas and takes the newest value among their answers (by version number or timestamp).

### Why W + R > N works

The rule: if **W + R > N**, every set of W replicas and every set of R replicas must share at least one replica. This is the pigeonhole principle. There are only N replicas; if the write set and read set together name more than N, at least one replica is in both. That overlapping replica has the latest acknowledged write, and since the reader picks the newest version among its R answers, it returns that write.

Take N = 3, W = 2, R = 2. A write succeeded on some 2 of the 3 replicas. A read asks some 2 of the 3. Two sets of 2 drawn from 3 cannot be disjoint, since that would need 4 distinct replicas. So at least one replica in the read set holds the new value.

~~~dg-widget quorum When W + R > N, every read set shares at least one replica with the latest write set.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**tracing a quorum read and write**

N = 3 replicas A, B, C hold `cart:7` at version 3. A client writes version 4 with W = 2. A and B acknowledge; C is slow and has not received it yet. The write is reported as successful.

Another client reads with R = 2. Whatever pair it reaches: {A, B} both return v4; {A, C} returns v4 and v3, so the client takes v4; {B, C} the same. Every possible read set sees v4.

With R = 1 instead (W + R = 3, not greater than N), a read that happens to reach only C gets v3: a stale read.

The client that saw v4 from A and v3 from C can also send v4 back to C, which is **read repair**.

</div>

| N | W | R | W + R > N? | Effect |
| --- | --- | --- | --- | --- |
| 3 | 2 | 2 | Yes | Overlapping quorums; tolerates one node down for reads and writes |
| 3 | 3 | 1 | Yes | Fast reads, but any one node down blocks writes |
| 3 | 1 | 3 | Yes | Fast writes, but reads need every node |
| 3 | 1 | 1 | No | Fastest, eventually consistent; reads can miss writes |
| 5 | 3 | 3 | Yes | Tolerates two nodes down for both |

Lower W means faster, more available writes. Lower R means faster reads. W + R > N is the knob that buys overlap. Latency is set by the slowest replica you wait for: with N = 3 and W = 2, a write waits for the second-fastest replica, which hides one slow node.

### Why quorums are not a full guarantee

Even with W + R > N, edge cases weaken the guarantee:

- **Sloppy quorums**: if some of the key's home replicas are down, some systems write to other nodes temporarily ("hinted handoff") to stay available. Now the W nodes that acknowledged may not overlap the R nodes a reader asks.
- **Concurrent writes**: two writes to the same key at the same time may land in different orders on different replicas. With last-writer-wins, one is silently lost.
- **Partial failure**: a write that reached only 1 of the needed 2 replicas is reported as failed but is not rolled back. A later read may or may not see it.
- **Read and write racing**: a read concurrent with a write may see the new value on one replica and the old on another, and a second read may see the old again.

So quorums by themselves do not guarantee linearizability (06.02). They give a strong "probably fresh" property that is good enough for many workloads.

### How stale replicas catch up

- **Read repair**: when a read notices one replica returned an older version, the client or coordinator writes the newer version back to it. Good for frequently read keys.
- **Anti-entropy**: a background process compares replicas and copies missing data. Comparing every key would be slow, so replicas build **Merkle trees**: hash trees where each leaf is a hash of a range of keys and each parent hashes its children. Two replicas compare root hashes; if they differ, they descend only into the branches that differ, finding a handful of mismatched keys among billions with a few dozen hash comparisons.
- **Hinted handoff**: if a replica is down, another node stores its writes as "hints" and hands them over when it returns.

~~~dg-tree A Merkle tree over a replica's key ranges. Two replicas compare top-down and only descend where hashes differ.
Root hash | differs between replicas
  Hash of keys 0 to 499 | matches, skip
  *Hash of keys 500 to 999 | differs, descend
    Keys 500 to 749 | matches, skip
    *Keys 750 to 999 | differs, sync these keys
~~~

## State machine replication

The strongest form of replication: every replica starts in the same state and applies the same sequence of **commands** in the same order, so they all end in the same state. Deterministic commands plus identical order equals identical replicas. The hard part is getting every replica to agree on that order, even when nodes fail. That is **consensus** (lesson 06.06), which Raft, Paxos and ZooKeeper's Zab provide. etcd, ZooKeeper, Consul and the metadata layers of most distributed databases work this way.

## Availability arithmetic

Replication raises availability only if failures are independent. The two basic formulas:

- Components **in sequence** (all must work) multiply: two 99.9% components give 0.999 × 0.999 ≈ 99.8%.
- Components **in parallel** (any one is enough) give 1 - (1 - A)^n: two 99.9% replicas give 1 - 0.001² = 99.9999%.

<div class="ex">
<div class="lab">Worked example</div>

**what replication buys, and what it does not**

A database server is up 99.9% of the time, about 8.8 hours down a year. Two replicas with instant, perfect failover: 99.9999%, about 32 seconds a year.

Reality is worse. Failover takes 30 s and happens about 4 times a year (2 minutes). Correlated failures (same rack power, same bad deploy, same software bug on both) happen too. A realistic result might be 99.99%, about 53 minutes a year: still a big improvement, but not six nines. Put replicas in different availability zones, stagger deploys, and measure failover time.

</div>

## Common mistakes

- **Reading from async followers** for data the user just wrote.
- **Assuming W + R > N gives linearizability.** It narrows the window but does not close it.
- **Last-writer-wins** on data where losing a write matters.
- **Untested failover**, which finds its bugs during a real outage.
- **Replicas in the same rack or zone**, so one failure takes all of them.

## In an interview

Name the scheme and the acknowledgement rule: "single leader, one synchronous follower in another zone, async read replicas; reads that must be fresh go to the leader". For a leaderless store, give N, W and R and say what they tolerate. Ask what happens to writes acknowledged just before a leader crash.

<div class="take">
<div class="lab">Key takeaway</div>

single-leader replication is simple but has lag and needs failover; multi-leader and leaderless designs gain write availability and pay with conflicts. In a quorum system, W + R > N makes every read set overlap every write set, so a read reaches at least one replica with the latest acknowledged write.

</div>
