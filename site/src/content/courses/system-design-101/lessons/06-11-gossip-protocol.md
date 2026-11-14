---
id: "06.11"
course: "system-design-101"
module: "06"
slug: "gossip-protocol"
title: "Gossip protocol"
keyIdeas: ["In a gossip protocol, each node periodically picks a few random peers and exchanges state (membership, health, metadata).", "Information spreads like an epidemic: it reaches all N nodes in about O(log N) rounds, with no central coordinator.", "Robust to node failures and scales to thousands of nodes, but convergence is eventual and messages are redundant.", "SWIM-style failure detection: ping a random member; if no reply, ask others to ping it indirectly before marking it suspect, then dead.", "Used in Cassandra, Consul/Serf, and Redis Cluster."]
question: "How many rounds does gossip need to reach N nodes, roughly?"
answer: "About O(log N) rounds. Each round every node that has the information passes it to a few random peers, so the number of informed nodes roughly multiplies each round (for example roughly threefold with a fanout of 2) until nearly everyone knows, then a few extra rounds catch the stragglers. For 1,000 nodes that is on the order of 10 rounds and for a million about 20, while each node sends only a constant number of messages per round regardless of cluster size."
sources: [{"title": "HashiCorp Serf: Gossip protocol", "url": "https://developer.hashicorp.com/serf/docs/internals/gossip", "note": "SWIM-based membership in practice."}, {"title": "SWIM paper", "url": "https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf", "note": "Scalable failure detection."}]
tags: ["Code"]
---

## Why gossip

Large clusters need every node to know two kinds of things: **who else is alive** (membership and failure detection), and **shared state** that changes over time (which node owns which data, configuration, schema versions, load). The question is how to spread that knowledge to hundreds or thousands of nodes cheaply and reliably.

The obvious approaches do not scale:

- **Everyone heartbeats everyone.** With N nodes that is N × (N - 1) messages per interval. For 1,000 nodes sending a heartbeat every second, that is about a million messages a second, just to say "I'm alive". Cost grows as O(N²).
- **A central server** collects heartbeats and broadcasts changes. Now that server is a hot spot and a single point of failure, and its view of who is alive is only as good as its own network links.
- **Broadcast trees** (a node tells its children, who tell theirs) are efficient but fragile: one failed interior node cuts off a whole subtree.

**Gossip protocols** (also called **epidemic protocols**) spread information the way rumours or diseases spread through a population. Each node periodically picks a few other nodes **at random** and exchanges what it knows with them. Nobody is in charge, no node is special, and no single failure stops the spread.

## How gossip spreads information

The basic loop on every node, once per **round** (a fixed interval, for example every 1 second or every 200 ms):

1. Pick **k** random peers (the **fanout**, typically 1 to 4).
2. Send them your recent updates (or a summary of your state).
3. Merge whatever you receive, keeping the newest version of each item.

There are three styles of exchange:

- **Push**: informed nodes send updates to random peers. Fast at the start, when few nodes know and almost every push reaches someone new. Slow at the end, because most pushes land on nodes that already know.
- **Pull**: every node asks random peers "what's new?". Slow at the start, fast at the end: the last few uninformed nodes are very likely to ask someone who knows.
- **Push-pull**: both at once. Combines the fast start and the fast finish, and is what most systems use.

Delivery to every node is not strictly guaranteed. But the probability that some node is missed shrinks very quickly with each extra round, so with suitable settings, full coverage is extremely likely. And the protocol tolerates lost messages and crashed nodes naturally: if one exchange fails, the information arrives by another path a round later.

### How fast: O(log N) rounds

In each round, every informed node tells a few more. So the number of informed nodes roughly **multiplies** each round, until most of the cluster knows. Starting from 1 node with a fanout of 2 (each informed node tells 2 others), the informed count goes roughly 1, 3, 9, 27, 81, and so on: multiplying by 3 each round. Reaching N nodes takes about log base 3 of N rounds, plus a few extra rounds to catch the stragglers as pushes start landing on nodes that already know.

In general the information reaches all N nodes in about **O(log N)** rounds. With 1,000 nodes, that is on the order of 10 rounds; with 1,000,000 nodes, around 20. Meanwhile each node sends only a **constant** number of messages per round (k), however big the cluster is. So total traffic grows as O(N) and time grows as O(log N). That combination is why gossip scales.

~~~dg-timeline A rumour spreading through 1,000 nodes with fanout 2. Growth is roughly threefold per round until most nodes know, then a short tail catches the rest.
Round 0 | 1 node knows
Round 1 | about 3 know
Round 2 | about 9 know
Round 4 | about 80 know
Round 6 | *about 550 know; many pushes now hit informed nodes
Round 8 | about 990 know
Round 10 | all 1,000 know, with very high probability
~~~

<div class="ex">
<div class="lab">Worked example</div>

**rounds and time for a real cluster**

A 2,000-node cluster gossips every 1 second with fanout 3 using push-pull.

The growth phase multiplies the informed count by up to 4 each round (each informed node plus 3 new ones): log base 4 of 2,000 is about 5.5 rounds. Add a few rounds for the tail, where pull does most of the work, and full coverage takes roughly 8 to 10 rounds: about 10 seconds.

Load: each node sends 3 messages per second and receives about 3, whatever the cluster size. Doubling to 4,000 nodes adds only about half a round (log base 4 of 2 = 0.5), not double the time and not double the per-node traffic.

</div>

~~~dg-bars Messages each node sends per second for liveness: gossip with fanout 3 stays constant at any size, while all-to-all heartbeats grow with the cluster.
*Gossip, any cluster size | 3 | 3 per node
All-to-all, 100 nodes | 99 | 99 per node
All-to-all, 1,000 nodes | 999 | 999 per node
All-to-all, 10,000 nodes | 9999 | 9,999 per node
~~~

## SWIM: gossip-based membership

**SWIM** (Scalable Weakly-consistent Infection-style process group Membership) is the most influential gossip-based membership protocol. HashiCorp's Serf and Consul use it with modifications, as do many other systems. Its key idea is to **separate two jobs** that naive designs mix together.

### Job 1: failure detection

Each **protocol period** (length T, say 1 second), every node does this:

1. Pick one peer and send it a **ping**.
2. If an **ack** comes back in time, the peer is alive. Done.
3. If not, ask **k** other random nodes (say 3) to ping the target on your behalf: a **ping-req**. They ping it and forward any ack.
4. If no ack arrives by the end of the period, by any route, the target is **suspected**.

The indirect ping is the clever part. A missing ack might mean the target is dead, or just that the link between **you and it** is bad. If 3 other nodes, on different network paths, also cannot reach it, the target is much more likely to be dead. This cuts false alarms caused by one bad link.

~~~dg-seq SWIM failure detection: a direct ping, then indirect pings through other members before suspecting the target.
Node A > Node C : ping
Node A > Node A : no ack within timeout
Node A > Node B : ping-req C
Node B > Node C : ping
Node C -> Node B : ack
Node B -> Node A : ack (C is alive)
~~~

### Job 2: dissemination

Membership updates ("D joined", "E left", "F suspected", "F confirmed dead") need to reach everyone. Instead of sending separate messages, SWIM **piggybacks** them on the ping, ping-req and ack messages that are flowing anyway. Each update is attached to a node's next several messages, then dropped. Because every node pings a random peer each period, updates spread like gossip, in a number of periods that grows logarithmically with group size. No extra messages are needed.

### Refinements that matter

- **Suspicion before death.** A node that fails its probes is first marked **suspect**, and that is gossiped. If the node is actually alive, it hears the rumour about itself and **refutes** it by gossiping an "alive" message with a higher **incarnation number** (a counter only that node may increase). Only if no refutation arrives within a suspicion timeout is it declared **dead**. This sharply cuts false positives from nodes that were merely slow for a moment.
- **Round-robin probing.** Instead of choosing a fully random peer each time, each node probes members in a shuffled order and reshuffles each pass. Every member is then probed within a bounded time, rather than possibly being skipped by bad luck.

~~~dg-state How a SWIM member sees a peer. Suspicion gives a slow node time to refute before it is declared dead.
[*] -> Alive
Alive -> Suspect : direct and indirect pings fail
Suspect -> Alive : refutes with higher incarnation
Suspect -> *Dead : suspicion timeout passes
Dead -> Alive : rejoins with new incarnation
~~~

The results: each member sends and receives about **2 messages per period** regardless of group size (a constant load), and the expected time to first detect a failure does **not** grow with group size. In tests with 10% packet loss, the suspicion mechanism kept far more members correctly in the group than the basic version without it.

## Where gossip is used

- **Cassandra and ScyllaDB**: every second, each node gossips with up to three others about cluster state: which nodes are up, their token ranges (which data they own), schema versions and load. Each piece of state carries a version so newer information always wins.
- **Consul and Serf**: membership and failure detection on SWIM, with practical changes. They add a periodic **full state sync over TCP** between random pairs (faster convergence, and recovery after a partition heals); a **dedicated gossip layer** besides piggybacking, so gossip can run often (every 200 ms) while failure probes run slower (every second); and they keep dead nodes' state for a while so full syncs include it.
- **Lifeguard** extensions (from HashiCorp) tackle false positives caused by the **detecting** node being unhealthy. If a node is overloaded, it sends and processes messages late and wrongly suspects others. Lifeguard adds **nack** messages to indirect probes, so a node that keeps missing nacks realises **it** may be the degraded one and slows its own failure detector. It also starts the suspicion timeout long and shortens it as other nodes independently confirm the suspicion.
- **Redis Cluster**: nodes gossip slot ownership and failure reports over a cluster bus; a primary is marked failed when a majority of primaries report it.
- **Anti-entropy** between replicas (lesson 06.03) is gossip-like too: random pairs compare Merkle trees and repair differences.

<div class="ex">
<div class="lab">Worked example</div>

**a 1,000-node cluster learns of a failure**

Node 417 crashes at t = 0. Probing order means some node probes 417 within about one protocol period: say node 52 at t = 0.6 s. No ack. Node 52 sends ping-req to 3 random members; none get an ack. At t = 1 s, node 52 marks 417 **suspect** and piggybacks that on its next messages.

Each period, every node that knows tells the nodes it pings. After about log2(1,000) ≈ 10 periods of 1 s, essentially the whole cluster has heard "417 suspect".

Node 417 does not refute within the suspicion timeout (say 5 s, though implementations scale it with log N), so at about t = 6 s it is declared dead, and that is gossiped too. Its token ranges are taken over by their replicas. Had 417 merely been paused for 3 s, it would have woken, heard the rumour and refuted it with incarnation + 1, staying in the cluster.

</div>

## Trade-offs

- **Eventually consistent.** Different nodes have slightly different views for a few rounds. One node may think 417 is alive while another already marked it suspect.
- **Not for decisions that need agreement.** Who is the leader, which write committed, who holds a lock: these need consensus (06.06). Gossip is for spreading information where a few seconds of disagreement is harmless.
- **Message size.** If each round sends the whole state, messages grow with cluster size. Systems gossip **digests** (version summaries) and pull the full details only for items that differ.
- **Tuning.** Faster rounds mean faster spread and more bandwidth. Shorter suspicion timeouts mean faster detection and more false positives.

## Common mistakes

- **Using gossip for leader election** or anything that must have one agreed answer.
- **Suspicion timeouts that ignore cluster size**, causing false deaths in large clusters.
- **Ignoring the detecting node's own health**, so one overloaded node accuses everyone.
- **Gossiping full state** every round in large clusters.

## In an interview

If your design has hundreds of nodes that must know about each other, say "membership and failure detection via a SWIM-style gossip protocol, so per-node load is constant and news reaches everyone in about log N rounds", and keep the authoritative decisions in a small consensus group.

<div class="take">
<div class="lab">Key takeaway</div>

gossip spreads information by having each node exchange state with a few random peers each round, reaching all N nodes in about O(log N) rounds with constant work per node and no single point of failure. SWIM uses it for scalable failure detection, with indirect pings and suspicion to keep false positives low.

</div>
