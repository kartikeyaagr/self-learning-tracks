---
id: "06.13"
course: "system-design-101"
module: "06"
slug: "distributed-hash-tables"
title: "Distributed hash tables"
keyIdeas: ["A distributed hash table spreads key-value pairs across many nodes with no central directory, and lets any node find the owner of a key.", "**Chord**: nodes and keys sit on a hash ring. Each node keeps a \"finger table\" of nodes at exponentially growing distances, so a lookup takes O(log N) hops.", "**Kademlia**: distance is the XOR of IDs. Nodes keep buckets of contacts at each distance range and query closer nodes in parallel. Used by BitTorrent and IPFS.", "DHTs handle nodes joining and leaving by moving only nearby keys.", "In datacenters, most systems use a central metadata service or gossip instead, since membership is smaller and more stable."]
question: "How does a node find a key in a DHT without knowing every other node?"
answer: "Each node keeps a small routing table whose entries are spaced exponentially around the ID space: many contacts near itself and few far away (Chord's finger table points to the nodes 1, 2, 4, 8 and so on positions ahead; Kademlia's k-buckets hold up to k contacts for each range of XOR distance). To find a key, a node sends the lookup to the known node closest to the key without passing it (or, in Kademlia, asks the few closest known nodes for even closer ones), and that node does the same with its own table. Each hop at least halves the remaining distance, so the owner is found in about O(log N) hops (about 10 for a million nodes) while each node stores only O(log N) contacts."
sources: [{"title": "Chord paper", "url": "https://pdos.csail.mit.edu/papers/chord:sigcomm01/chord_sigcomm.pdf", "note": "A classic DHT with O(log n) lookups."}, {"title": "Wikipedia: Kademlia", "url": "https://en.wikipedia.org/wiki/Kademlia", "note": "XOR-distance DHT used by BitTorrent."}]
---

## The problem DHTs solve

A **distributed hash table (DHT)** stores key-value pairs across many nodes, and lets any node find the value for any key, with **no central directory** and **no node knowing all the others**. It offers the same two operations as an ordinary hash table, `put(key, value)` and `get(key)`, but the table is spread over thousands or millions of machines.

DHTs were designed for **peer-to-peer** networks: file sharing, decentralised storage, blockchain node discovery. Three properties of those environments shape the design:

- **Huge membership.** Millions of nodes. No single node can hold a list of all of them, let alone keep it current.
- **Churn.** Nodes join and leave constantly; a typical peer might stay online for minutes or hours.
- **No trust and no centre.** There is no company running a directory server, and any central server would be a single point of failure (and of legal or political attack).

Consistent hashing (lesson 04.07) already answers **who owns a key**: hash nodes and keys onto the same ring, and a key belongs to the next node along. What a DHT adds is **routing**: how do you reach that owner when you only know a few dozen other nodes? The answer in every major DHT is the same trick: keep a small routing table that is **dense near yourself and sparse far away**, so that every hop at least halves the remaining distance, and a lookup takes **O(log N)** hops.

## Chord

Chord is the simplest DHT to understand. It supports exactly one operation: given a key, return the node responsible for it. Storing values on top of that is easy once you can find the right node.

### The ring

- Every node and every key gets an **m-bit identifier** by hashing (for example SHA-1 of the node's IP address, or of the key). With m = 160, there are 2^160 positions, arranged in a circle from 0 to 2^m - 1.
- A key belongs to its **successor**: the first node whose ID is equal to or follows the key's ID going clockwise.

### Why a successor pointer is not enough

If each node only knew its successor, a lookup could follow successors around the ring until it reached the owner. That is correct, but with N nodes it takes up to N hops, about N/2 on average. With a million nodes that is half a million network round trips: useless.

### The finger table

Each node n keeps a **finger table** of up to m entries. The i-th finger points to the successor of the position **n + 2^(i-1)**. So node n knows the nodes roughly 1, 2, 4, 8, 16, 32 and so on positions ahead of it, up to half the ring away. It knows its close neighbourhood in detail and the far side of the ring only coarsely, much like you know your own street well and only the major cities of other countries.

To look up key k, node n finds the finger that gets **closest to k without passing it**, and forwards the query there. That node does the same with its own fingers. Because the fingers double in distance, each hop at least halves the remaining distance to the target. Halving a distance of up to 2^m takes at most m steps, but with N nodes spread over the ring, only about log2 N hops matter. So a lookup takes **O(log N)** hops, and each node stores only **O(log N)** distinct routing entries.

~~~dg-tree Node 8's finger table on a 64-position Chord ring. Fingers double in distance, so the node knows its neighbourhood densely and the far ring sparsely.
Node 8 finger table
  +1: successor of 9 | node 10
  +2: successor of 10 | node 10
  +4: successor of 12 | node 14
  +8: successor of 16 | node 21
  +16: successor of 24 | node 32
  *+32: successor of 40 | node 42
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a Chord lookup traced**

A Chord ring with 6-bit IDs (positions 0 to 63) and about 10 nodes. Node 8 wants key 54.

Node 8's fingers point to the successors of 9, 10, 12, 16, 24 and 40. The finger closest to 54 without passing it is the successor of 40, node 42. Node 8 forwards the lookup there. Remaining distance: from 54 - 8 = 46 positions to 54 - 42 = 12.

Node 42's fingers point to the successors of 43, 44, 46, 50, 58 and 10. The closest one before 54 is the successor of 50, say node 51. Remaining distance: 3.

Node 51's successor is node 56, which is at or after 54, so node 56 owns key 54. Three hops instead of walking the ring node by node.

</div>

~~~dg-seq The same Chord lookup as messages: each hop forwards to the finger closest to the key without passing it.
Node 8 > Node 42 : find successor of 54 (finger +32)
Node 42 > Node 51 : find successor of 54 (finger +8)
Node 51 > Node 51 : my successor 56 covers 54
Node 51 -> Node 8 : owner of key 54 is node 56
Node 8 > Node 56 : get(54)
Node 56 -> Node 8 : value
~~~

### How many hops in practice

On average a lookup takes about **½ log2 N** hops, because the first hop typically covers half the distance and the remaining distance is random. For a million nodes, log2 of a million is about 20, so a typical lookup is about **10 hops**. If each hop is a 50 ms internet round trip, that is roughly half a second, which is why DHTs suit peer-to-peer systems better than latency-sensitive datacenter services.

### Churn

When a node joins, it takes over some keys from its successor; when it leaves, its keys go to its successor. Either way only about **1/N of keys** move. Fixing everyone's routing tables after a join or leave takes **O(log² N)** messages. Nodes run a periodic **stabilisation** protocol: each node asks its successor "who is your predecessor?", fixes its successor pointer if a new node has slipped in between, and notifies the successor about itself. Fingers are refreshed periodically too. Correctness only needs successor pointers to be right; fingers just make it fast, so stale fingers slow lookups down but do not break them.

### Failures

Each node keeps a **successor list** of several (say r = log N) successors, not just one. If the first is dead, it uses the next. With such lists, lookups still succeed with high probability even if half the nodes fail at once. Values are also **replicated** on the first few successors of their key, so data survives the owner's failure.

## Kademlia

Kademlia is the most widely deployed DHT. It uses the same "halve the distance each hop" idea with a different, cleverer distance function.

### XOR distance

The distance between two IDs is their **bitwise XOR**, read as an integer. For example d(1011, 1001) = 0010 = 2. XOR distance has useful properties:

- It is **symmetric**: d(a, b) = d(b, a). (Chord's clockwise distance is not, so Chord nodes cannot learn useful routing information from queries they receive. Kademlia nodes can.)
- d(a, a) = 0, and for any point and any distance there is exactly one other point at that distance.
- It respects shared prefixes: two IDs that share a long common prefix are close. The **position of the first differing bit** decides the rough distance.

### k-buckets

Each node keeps one **bucket** per bit position of the ID: bucket i holds contacts whose distance from the node is between 2^i and 2^(i+1), which means they share exactly the first (m - i - 1) bits with the node. Each bucket holds up to **k** contacts (typically k = 20). Half of all IDs fall in the farthest bucket, a quarter in the next, and so on, but each bucket holds at most k. So a node knows many nodes near itself and a few far away: the same dense-near, sparse-far shape as Chord's fingers.

~~~dg-tree Kademlia k-buckets for a node with ID 0000 (4-bit IDs). Each bucket covers a range of XOR distance twice as wide as the last and keeps at most k contacts.
Node 0000
  Bucket 3: IDs 1xxx | half the space, keep 20
  Bucket 2: IDs 01xx | a quarter, keep 20
  Bucket 1: IDs 001x | an eighth
  *Bucket 0: ID 0001 | nearest neighbour
~~~

### Lookup

Lookups are **iterative**: the node doing the lookup stays in control rather than forwarding the query.

1. Pick the **α** closest contacts to the target from your own buckets (α is typically 3).
2. Send them **FIND_NODE** (or FIND_VALUE) in parallel. Each replies with the k closest contacts **it** knows.
3. Add those to your candidate list and query the α closest ones you have not asked yet.
4. Stop when a round produces no contacts closer than the closest you already have. The k closest nodes found are where the key lives.

Each round roughly halves the distance, giving O(log N) rounds. Querying α nodes in parallel means one slow or dead node does not stall the lookup.

### Preferring long-lived nodes

When a node learns of a new contact for a bucket that is already full, it **pings the oldest contact** in that bucket. If the oldest answers, it is kept and the newcomer is dropped. Measurements of peer-to-peer networks show that the longer a node has been online, the more likely it is to stay online for another hour, so old contacts are the better bet. This policy also resists some attacks: flooding the network with new fake nodes cannot push out established contacts.

### The protocol

Kademlia has just four RPCs: **PING** (are you alive?), **STORE** (keep this key-value pair), **FIND_NODE** (give me the k closest nodes to this ID), and **FIND_VALUE** (like FIND_NODE, but return the value if you have it). Values are stored on the k nodes closest to the key and republished periodically so they survive churn.

Kademlia is used by **BitTorrent's Mainline DHT** (which lets torrents work without a central tracker, with millions of concurrent nodes), **IPFS**, and **Ethereum's** node discovery protocol.

~~~dg-bars Typical lookup hops at different network sizes when each hop halves the distance (about half of log2 N). Ten times more nodes costs fewer than two extra hops.
1,000 nodes | 5 | ~5 hops
100,000 nodes | 8.5 | ~8 to 9 hops
*1 million nodes | 10 | ~10 hops
100 million nodes | 13 | ~13 hops
~~~

## The general idea

Both designs give each node a routing table that is **dense near itself and sparse far away**, with entries spaced exponentially. Every hop jumps to a node at least about twice as close to the target. So **log N hops suffice without anyone knowing the full membership**, each node stores only O(log N) contacts, and the failure of any node affects only its own small neighbourhood. Other DHTs (Pastry, Tapestry, CAN) differ in geometry but follow the same principle.

## DHTs in datacenter design

Inside a datacenter, the trade-offs flip. A cluster has hundreds or thousands of nodes, not millions; nodes are trusted and long-lived; and every millisecond of latency matters. Every node (or client) can simply hold the **whole ring**: a few thousand entries is a tiny amount of memory. So datacenter systems use **one-hop** designs: Dynamo and Cassandra gossip the full ring membership to all nodes (lesson 06.11), and a client or coordinator routes a request directly to a replica of the key in a single hop. Redis Cluster clients cache the full slot map for the same reason (05.02).

~~~dg-compare Multi-hop DHT routing versus one-hop routing with a full membership view. Scale and churn decide which fits.
[Multi-hop DHT]
Lookup -(log N hops)-> Closer nodes -> Owner
+ Each node knows O(log N) peers
+ Survives millions of nodes and churn
- About 10 round trips per lookup
[*One-hop, full view]
Lookup -(read local ring)-> Owner
+ One network hop, lowest latency
+ Simple to reason about
- Every node stores the full ring
- Needs gossip or a registry to stay current
~~~

<div class="ex">
<div class="lab">Worked example</div>

**when to choose which**

A storage service with 300 nodes in two datacenters, latency budget 10 ms: the full ring is 300 entries, a few kilobytes. One-hop routing via gossiped membership costs nothing and avoids 4 to 5 extra hops that a DHT would need (about 1 ms each inside a datacenter, too much for the budget).

A peer-to-peer file sharing network with 5 million nodes, half of which change every few hours: keeping a full ring on every node would mean gossiping millions of changes per hour to millions of nodes. A Kademlia DHT keeps about 20 × 23 ≈ 460 contacts per node and finds any key in roughly 11 or 12 hops (half of log2 of 5 million, about 22). Half a second per lookup is fine for starting a download.

</div>

## Common mistakes

- **Proposing a multi-hop DHT inside a datacenter**, where every node can know the full ring.
- **Assuming O(log N) means fast.** Ten internet round trips is hundreds of milliseconds.
- **Forgetting replication.** A key stored on one node disappears when that peer goes offline.
- **Confusing consistent hashing with a DHT.** Consistent hashing decides ownership; the DHT's routing tables decide how to reach the owner.

## In an interview

If asked how a node finds a key without a directory, describe the finger table or k-buckets in one sentence, then the halving argument for O(log N). Then add the practical note: in a datacenter-sized cluster you would just gossip the full ring and route in one hop.

<div class="take">
<div class="lab">Key takeaway</div>

a DHT finds the owner of a key without a central directory by giving each node routing pointers spaced exponentially around the ID space (Chord's fingers, Kademlia's k-buckets). Each hop at least halves the distance to the key, so lookups take O(log N) hops while each node knows only O(log N) others. Inside a datacenter, a full membership view and one-hop routing are usually better.

</div>
