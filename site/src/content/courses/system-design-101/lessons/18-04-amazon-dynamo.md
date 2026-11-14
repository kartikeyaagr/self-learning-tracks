---
id: "18.04"
course: "system-design-101"
module: "18"
slug: "amazon-dynamo"
title: "Amazon Dynamo"
keyIdeas: ["Dynamo (2007) is Amazon's highly available key-value store for things like shopping carts, where accepting writes always beats strict consistency.", "Consistent hashing with virtual nodes for partitioning; each key replicated to N nodes.", "Quorums with tunable N, R, W.", "**Sloppy quorum and hinted handoff**: if a target replica is down, the write goes to another healthy node with a hint; when the original recovers, the hint is delivered. Writes keep succeeding during failures.", "Vector clocks detect concurrent versions; the application merges them (cart union).", "Anti-entropy with Merkle trees repairs replicas; gossip spreads membership.", "It inspired Cassandra, Riak and Voldemort."]
question: "What is hinted handoff, and what availability problem does it solve?"
answer: "Hinted handoff is how Dynamo keeps a write's quorum when some of the key's home nodes are down or unreachable. With a strict quorum, a W = 2 write fails if two of the three preference-list nodes cannot be reached; with a sloppy quorum the coordinator instead sends the replica to the next healthy node on the ring, tagged with a hint naming the intended owner. That stand-in stores it in a separate local store, counts towards W, and periodically checks the owner; once the owner recovers it delivers the replica and deletes its own copy. It solves availability during temporary failures, partitions and even data-centre outages, so the shopping cart stays always writeable and the data keeps N copies. It does not guarantee reads see the write, and if the stand-in dies the hint is lost, which is why Merkle-tree anti-entropy repairs longer-lasting divergence."
sources: [{"title": "Amazon Dynamo paper", "url": "https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf", "note": "Consistent hashing, vector clocks, sloppy quorum, hinted handoff."}, {"title": "Werner Vogels: Amazon's Dynamo", "url": "https://www.allthingsdistributed.com/2007/10/amazons_dynamo.html", "note": "Intro post."}]
tags: ["Case"]
tools: ["Amazon"]
---

## Why Amazon built Dynamo

Dynamo, described publicly in 2007, is the key-value store that sat behind Amazon's shopping cart, customer sessions, best-seller lists, preferences and parts of the product catalogue. It is one of the most influential designs in distributed systems: Cassandra, Riak and Voldemort copied its techniques, and almost every "leaderless replication" discussion (lesson 06.03) uses its vocabulary.

Amazon ran tens of thousands of servers across several data centres for tens of millions of customers. At that size, **something is always failing**: a disk, a server, a switch, a network link between data centres. Dynamo was designed on the assumption that failure is the normal state, not an exception. Two business requirements shaped everything:

- **Always writeable.** A customer must be able to add an item to their cart even while servers, or a whole data centre, are failing. Rejecting a write is lost revenue and a worse experience than almost any inconsistency.
- **Latency at the 99.9th percentile.** The service level agreement was 99.9% of requests answered within **300 ms** at peak load. Measuring the average would hide the customers having a bad time, and those are often the most valuable customers (with the longest histories and biggest carts).

Most of these services only needed `get(key)` and `put(key, value)` on small objects (under 1 MB). A relational database would have brought joins, transactions and a single primary that they did not need, and the single primary was exactly the availability risk they wanted to remove. So Dynamo deliberately trades strong consistency for availability: in CAP terms (lesson 06.04), it chooses availability during partitions, and it returns conflicting versions to the application rather than refusing writes.

## The techniques at a glance

~~~dg-matrix Dynamo's problems and the technique it used for each, with what each technique gives and costs.
Problem | Technique | Gives | Costs
Partitioning | consistent hashing, virtual nodes | yes, add nodes incrementally | uneven load without many tokens
Always writeable | vector clocks, reconcile on read | yes, no rejected writes | app must merge versions
Temporary failures | sloppy quorum, hinted handoff | yes, writes during outages | hint lost if stand-in dies
Permanent failures | Merkle tree anti-entropy | yes, cheap background repair | trees rebuilt when ranges change
Membership | gossip, failure detection | yes, no central registry | limits cluster size
~~~

The rest of this lesson walks through placement and quorums briefly, then spends most of its time on the question: what is hinted handoff, and what availability problem does it solve?

## Placement: the ring and preference lists

Keys are hashed with MD5 into a 128-bit space arranged as a **ring** (consistent hashing, lesson 04.07). Each physical node owns several positions on the ring, called **virtual nodes** or tokens. Virtual nodes spread each machine's load across many small ranges, so when a node joins or leaves, its load is taken from or given to many others instead of one neighbour; a bigger machine simply gets more tokens.

A key is stored on **N** nodes: walk clockwise from the key's hash, and take the first N **distinct physical** nodes you meet (skipping virtual nodes of a machine already chosen). That ordered list is the key's **preference list**. Any node can act as **coordinator** for a request, but typically it is the first healthy node in the preference list.

## Quorums

A read waits for **R** replies, a write for **W** acknowledgements, out of the N replicas. If R + W > N, every read set overlaps every write set in at least one node, so a read sees the latest acknowledged write (as long as the sets are drawn from the same N nodes, which, as we will see, sloppy quorums relax). The common setting is **(N, R, W) = (3, 2, 2)**. Services tuned this per use: a read-heavy catalogue might use R = 1, W = 3 for very fast reads; a session store might use W = 1 for very fast writes and accept more risk.

<div class="ex">
<div class="lab">Worked example</div>

**what a strict quorum does during a failure**

Key `cart:alice` hashes to a ring position whose preference list is A, B, C. N = 3, W = 2.

Node A's rack loses power and node B is cut off from the coordinator by a network problem. Only C is reachable.

A strict quorum needs 2 acknowledgements from {A, B, C}. Only C can answer. The write fails, and Alice sees "Sorry, we couldn't add that item". Repeat that for every key whose preference list includes A and B, and a single rack failure plus one network glitch has stopped sales for a slice of customers.

</div>

## The availability problem

That example is exactly the outage Dynamo exists to prevent. With **strict quorums**, a request can only count acknowledgements from the key's N home nodes. If enough of them are down, or merely unreachable from this coordinator (which happens during network partitions even when the nodes are healthy), the write fails. Making N larger does not help much: more replicas also means more nodes that can be unreachable. The problem is the rule that only the home nodes count.

## Sloppy quorums and hinted handoff

Dynamo relaxes the rule. With a **sloppy quorum**, a write goes to the first **N healthy** nodes walking the ring from the key's position, not necessarily the N home nodes. If a home node is down, the next healthy node beyond the preference list stands in for it.

Here is the full mechanism, with N = 3. Key K normally lives on A, B and C. Node A is down.

1. The coordinator sends the write to B and C as usual, and to **D**, the next healthy node on the ring after C.
2. The copy sent to D carries a **hint** in its metadata: "this replica really belongs to A".
3. D stores hinted replicas in a **separate local database**, apart from the data it owns, and counts its acknowledgement towards W. The write succeeds with W = 2 easily, and the data still exists on N = 3 nodes.
4. D periodically checks whether A is reachable again (failure detection comes from gossip).
5. When A is back, D **delivers the hinted replica to A**. Once A confirms, D **deletes** its own copy.

That is **hinted handoff**: a healthy node temporarily holds a replica on behalf of an unavailable one, then hands it back.

~~~dg-seq Hinted handoff with N = 3: A is down, so D holds A's copy with a hint and hands it back when A recovers.
Coordinator > B : put K (home replica)
Coordinator > C : put K (home replica)
Coordinator > D : put K, hint: belongs to A
B -> Coordinator : ack
D -> Coordinator : ack, W = 2 met
D > D : store in hinted store, check A
D > A : A is back: deliver K
A -> D : stored
D > D : delete hinted copy
~~~

~~~dg-timeline The same key during a two-hour outage of node A.
09:00 | A fails; writes for K go to B, C and D (hint)
09:05 | *Cart write succeeds; data on 3 nodes
10:30 | Reads of K answered by B, C (and D)
11:00 | A recovers, gossip spreads the news
11:02 | D hands hinted replicas to A, deletes its copies
11:10 | Merkle sync confirms A matches B and C
~~~

### What it solves

- **Availability during temporary failures and partitions.** Writes and reads keep succeeding when some home replicas are down or unreachable, because healthy stand-ins count towards the quorum.
- **Durability during the outage.** The data still has N copies, just not all on their home nodes.
- **Data-centre failure.** Dynamo builds preference lists that span multiple data centres, so the stand-ins can be in another data centre and a whole site can fail without stopping writes.

### What it does not solve

- **Read-your-writes is no longer guaranteed.** With a sloppy quorum, the W nodes that took a write and the R nodes that answer a later read may not overlap (D held the write; the read went to A, which just came back and has not received the handoff yet). R + W > N no longer implies overlap. This is why sloppy quorums are a durability and availability tool, not a consistency guarantee.
- **The stand-in can die.** If D fails permanently before handing the data back, the hinted copy is lost; K now has only two up-to-date replicas. Long-term divergence needs a separate repair mechanism.
- **Concurrent versions.** Two coordinators on different sides of a partition can both accept writes for K. Someone must reconcile them later.

~~~dg-compare Strict quorum versus sloppy quorum with hinted handoff when two of three home replicas are unreachable.
[Strict quorum]
Write K -(only A, B, C count)-> 1 of 3 reachable
- Write rejected
+ Reads always overlap writes
[*Sloppy quorum + hints]
Write K -(first 3 healthy)-> C, D, E with hints
+ Write accepted, 3 copies kept
+ Handed back on recovery
- Reads may miss it briefly
~~~

## Repairing permanent divergence: Merkle trees

For failures that outlast hinted handoff, replicas run **anti-entropy**: they compare their data and fix differences in the background. Comparing every key would mean sending all the data. Instead each node keeps a **Merkle tree** per key range: the leaves are hashes of individual values, and each parent is the hash of its children. Two replicas compare root hashes. If equal, the entire range matches and nothing more is sent. If not, they compare the children and walk down only the branches that differ, finding the out-of-sync keys while exchanging very little data.

<div class="ex">
<div class="lab">Worked example</div>

**what Merkle trees save**

A key range holds 1 million keys, and 10 differ between two replicas. Sending all keys and hashes would be about 1 million x 40 bytes = 40 MB.

A binary Merkle tree over the range has about 20 levels. Each differing key adds one path of 20 hash comparisons; 10 differing keys need at most about 200 node hashes exchanged (plus siblings), a few kilobytes, then the 10 values themselves.

The cost is that trees must be recomputed when a node's key ranges change (a node joins or leaves), which is expensive. Background repair is throttled so it does not hurt foreground latency.

</div>

## Conflicts: vector clocks and the shopping cart

Sloppy quorums and partitions mean two nodes can accept different writes for the same key. Dynamo does not pick a winner by timestamp. Each version carries a **vector clock**: a list of (node, counter) pairs recording which coordinators have updated it (lesson 06.05). If one version's clock is greater than or equal to the other's in every entry, it descends from it and supersedes it. If neither dominates, they are **concurrent** and both are kept.

1. A write through node Sx creates D1 [(Sx,1)]; another through Sx creates D2 [(Sx,2)], which supersedes D1.
2. A write through Sy creates D3 [(Sx,2),(Sy,1)]; separately, during a partition, a write through Sz creates D4 [(Sx,2),(Sz,1)].
3. Neither D3 nor D4 descends from the other, so both are kept and returned together on the next read.
4. The client merges them and writes the result through Sx, producing D5 [(Sx,3),(Sy,1),(Sz,1)], which supersedes both.

~~~dg-flow Vector clocks: D3 and D4 conflict, and the client merges them into D5.
[Branch] D2 (Sx,2) -(update via Sy)-> D3 (Sx,2)(Sy,1)
[Branch] D2 (Sx,2) -(update via Sz)-> D4 (Sx,2)(Sz,1)
[Merge] D3 + D4 -(client merges via Sx)-> *D5 (Sx,3)(Sy,1)(Sz,1)
~~~

For the shopping cart, merging means taking the **union** of items: an "add to cart" is never lost, though an item the customer deleted can occasionally reappear. Amazon judged that far better than losing an addition. Other services chose simpler rules: last write wins for sessions, or R = 1 and W = N for read-heavy catalogue data that rarely conflicts.

## Results

- Over 24 hours of shopping cart traffic, **99.94%** of requests saw exactly one version. Divergence was rare, and mostly caused by busy automated clients writing concurrently, not by failures.
- 99.9th percentile latency was about **200 ms**, around 10 times the average. Moving request coordination into the client library (so the client contacts replicas directly instead of going through a random node) cut 99.9th percentile read latency from about 69 ms to 30 ms in one measurement.
- **99.9995%** of requests succeeded, with no data loss reported.
- Lessons: let each service tune N, R and W; measure at the 99.9th percentile; throttle background work so it does not hurt foreground requests; full-membership gossip works for hundreds of nodes but would not scale to tens of thousands.

Dynamo's ideas live on in Cassandra, Riak and Voldemort (Cassandra has its own hinted handoff, with a configurable window, usually 3 hours, after which repair must take over). Its name lives on in **DynamoDB**, Amazon's managed service, which is a different design with leader-based replication per partition.

## In an interview

If asked about hinted handoff, state the problem first (strict quorums fail writes when home replicas are unreachable), then the mechanism (next healthy node, hint, separate store, hand back, delete), then the limits (no read-your-writes guarantee, stand-in can die, so Merkle anti-entropy backs it up). Mentioning the shopping cart's union merge shows you understand why availability was worth the conflicts.

<div class="take">
<div class="lab">Key takeaway</div>

hinted handoff lets Dynamo accept reads and writes when some of a key's home replicas are down or unreachable: the request goes to the next healthy node on the ring with a hint naming the intended owner, which keeps it separately and hands it back when the owner recovers. Combined with sloppy quorums, vector clocks for conflicts and Merkle-tree repair, Dynamo stays always writeable at the cost of occasional conflicting versions and weaker read guarantees.

</div>
