---
id: "06.04"
course: "system-design-101"
module: "06"
slug: "cap-and-pacelc"
title: "CAP and PACELC"
keyIdeas: ["**CAP**: during a network partition, a system must choose between consistency (refuse some requests) and availability (answer with possibly stale data). Without a partition you can have both.", "CAP's definitions are narrow, and many real systems fit neither \"CP\" nor \"AP\" cleanly, which is why the labels mislead.", "**PACELC**: if Partition, choose Availability or Consistency; Else, choose Latency or Consistency.", "PACELC is more useful because partitions are rare, but the latency vs consistency trade-off applies to every request (waiting for replicas to agree takes time).", "Examples: DynamoDB and Cassandra default to PA/EL; Spanner is PC/EC."]
question: "Explain PACELC in one sentence. Why is it more useful than CAP?"
answer: "PACELC says that if there is a network partition a replicated system must choose between availability and consistency, and else, in normal operation, it must choose between lower latency and consistency. It is more useful than CAP because partitions are rare, while the latency cost of keeping replicas consistent (waiting for a leader, a quorum or another region) applies to every single request, so PACELC describes the trade-off users actually feel, explains why systems such as Cassandra weaken consistency even when nothing has failed, and lets you choose per operation."
sources: [{"title": "Kleppmann: Please stop calling databases CP or AP", "url": "https://martin.kleppmann.com/2015/05/11/please-stop-calling-databases-cp-or-ap.html", "note": "What CAP really says."}, {"title": "Daniel Abadi: PACELC paper", "url": "https://www.cs.umd.edu/~abadi/papers/abadi-pacelc.pdf", "note": "Latency vs consistency when there's no partition."}]
---

## What CAP actually says

A **network partition** is a failure where some nodes cannot communicate with others, even though every node is still running. A cable is cut, a switch fails, a misconfigured firewall drops traffic between two availability zones. Each side can still talk to clients, but not to the other side.

The **CAP theorem** says that during a partition, a replicated data system must choose between two properties:

- **Consistency (C)**: every read returns the most recent write, as if there were one copy.
- **Availability (A)**: every request that reaches a non-failed node gets a non-error response.

It cannot have both. Imagine replicas A and B are cut apart, and a client writes x = 2 to A. Another client now asks B for x. B has two options. It can **answer** with what it has (x = 1), which keeps it available but returns a stale value. Or it can **refuse** (error or wait) until it can confirm with A, which keeps it consistent but unavailable. There is no third option: B cannot know about a write it never received.

The "P" (partition tolerance) is not really a choice. Networks do fail, so any system that spans more than one machine must handle partitions. The real choice is **between C and A, during a partition**. That is why the popular slogan "pick two of three" is misleading: you cannot pick CA for a distributed system, because you do not get to opt out of partitions.

~~~dg-seq During a partition a replica that has not seen the latest write must choose: answer with stale data (A) or refuse (C).
Client 1 > Replica A : write x = 2
Replica A -> Client 1 : ok
Replica A > Replica B : replicate x = 2 (lost, partition)
Client 2 > Replica B : read x
Replica B -> Client 2 : choose A: x = 1 (stale) or choose C: error
~~~

### CAP's definitions are narrower than you think

CAP is a precise theorem, and its definitions are much stricter than everyday usage:

- **Consistency in CAP means linearizability** (lesson 06.02). It is unrelated to the C in ACID, which means "the database keeps your invariants".
- **Availability means every request to a non-failed node** eventually gets a non-error response. Not "most requests", not "99.99% uptime", and not "fast". A system that answers after two minutes counts as available.
- It models a **single register** (one value), and considers **only network partitions**. Crashes, disk failures, bugs and overload are not part of it.
- It says nothing about **latency**, which is what users experience every day.

So a system can fail to be "CAP-available" and still have excellent real-world availability (a majority side keeps serving and the minority is a tiny share of clients), and a system can be "CAP-consistent" for some operations and not others.

## Why the CP and AP labels mislead

Many databases are casually labelled "CP" or "AP". Look closely and most fit neither label cleanly:

- A typical **single-leader database** with async followers is neither. A client cut off from the leader cannot write (not available), and reads from followers can be stale (not linearizable).
- **ZooKeeper** is usually called CP, but its default reads are served by whichever server you are connected to and may be stale, so they are not linearizable. You need to call `sync` before a read to make it so.
- **Dynamo-style stores** (Cassandra, Riak) are usually called AP. With quorum settings (W + R > N), a minority side of a partition cannot reach a quorum, so it is not fully available, and quorums still do not guarantee linearizability (06.03).
- **One system can offer different guarantees per operation.** DynamoDB offers eventually consistent and strongly consistent reads on the same table. Cassandra lets you pick a consistency level per query. MongoDB's behaviour depends on read and write concern settings.

The more useful habit: talk about **specific consistency models** and **per-operation trade-offs**, not labels for whole systems.

~~~dg-matrix What happens to each kind of request on the minority side of a partition, for common setups. Real behaviour depends on configuration, not on a label.
Setup | Writes on minority side | Reads on minority side | Normal-case read latency
Raft or ZooKeeper (linearizable reads) | no, refused | no, refused | partial, leader round trip
ZooKeeper default reads | no, refused | partial, only in read-only mode | yes, local server, maybe stale
Single leader, async followers | no, refused | yes, maybe stale | yes, any replica
Cassandra at QUORUM | no, if no quorum | no, if no quorum | partial, waits for 2 of 3
Cassandra at ONE | yes, accepted | yes, maybe stale | yes, fastest replica
~~~

## PACELC: the trade-off you live with every day

CAP only applies **during** partitions, and partitions are rare: perhaps a few minutes a year in a good datacenter. But every replicated system faces a trade-off **all the time**, even when the network is perfect. To make a write consistent, it must reach other replicas, possibly in another region, before it is acknowledged. That costs **latency**. To make a read consistent, you must ask the leader or a quorum, not just the nearest replica. That also costs latency.

**PACELC** extends CAP to capture this: if there is a **P**artition, choose **A**vailability or **C**onsistency; **E**lse (normal operation), choose **L**atency or **C**onsistency.

Systems get a two-part label: PA/EL, PC/EC, PA/EC or PC/EL.

| System | During partition | Normal operation | Label |
| --- | --- | --- | --- |
| Dynamo, Cassandra, Riak (default settings) | Availability | Latency | PA/EL |
| BigTable, HBase | Consistency | Consistency | PC/EC |
| VoltDB / H-Store, Megastore | Consistency | Consistency | PC/EC |
| MongoDB (classic defaults) | Availability | Consistency | PA/EC |
| PNUTS (Yahoo) | Consistency | Latency | PC/EL |
| Spanner | Consistency | Consistency | PC/EC |

~~~dg-tree PACELC as a decision tree. The first question rarely applies; the second applies to every request.
Is the network partitioned?
  Yes, rare | choose one
    Availability | answer, maybe stale
    Consistency | refuse or wait
  *No, almost always | choose one
    Latency | answer from nearest replica
    Consistency | wait for leader or quorum
~~~

### Why PACELC is more useful than CAP

- It explains why systems like Cassandra **weaken consistency when nothing is broken**. They are not preparing for partitions; they are cutting latency on every request.
- It forces you to state the **normal-case cost** of your choice, which is what users feel every day. A design that is "CP" may add 100 ms to every write because of a cross-region round trip.
- It makes the "else" choice **tunable per operation**, matching how real systems work.

### Where the latency comes from

The trade-off comes directly from how replication works:

- **Send updates to all replicas at once** (leaderless): different replicas might receive concurrent updates in different orders. To agree on an order you need a coordination round, which costs latency. Skipping it costs consistency.
- **Send updates through a leader**: the leader orders updates. But then either the leader waits for replicas before acknowledging (latency) or it does not (a crash loses writes, and reads from replicas can be stale).
- **Read from a replica**: either check with the leader that you are up to date (latency) or accept that you might be behind (consistency).
- **Quorums**: waiting for more replicas reduces staleness and adds the latency of the slowest one you wait for.

<div class="ex">
<div class="lab">Worked example</div>

**the price of EC across regions**

Replicas live in Virginia, Oregon and Frankfurt. Round trips: Virginia to Oregon about 70 ms, Virginia to Frankfurt about 90 ms.

EC (consensus with a majority): a write from Virginia must reach one other region before it is acknowledged, so it takes at least 70 ms plus processing, maybe 80 ms in total. A linearizable read must confirm with a majority too.

EL (acknowledge locally, replicate asynchronously): the write takes about 2 ms. Other regions see it 70 to 90 ms later, and a crash of the Virginia node in that window loses it.

For a "like" button, 2 ms with a tiny risk of losing a like is the right trade. For a bank transfer, 80 ms is cheap insurance. Same infrastructure, different choice per operation.

</div>

~~~dg-bars Write latency for one write originating in Virginia under different "else" choices. Consistency costs at least one cross-region round trip.
Local ack, async replication (EL) | 2 | ~2 ms
Quorum in one region (3 zones) | 5 | ~5 ms
*Majority across 3 regions (EC) | 80 | ~80 ms
Wait for all 3 regions | 100 | ~100 ms
~~~

## How to use this in interviews

Do not say "we pick AP" for a whole system. Say it per operation, and cover both halves:

- "Likes and view counts tolerate staleness, so they are acknowledged locally and replicated asynchronously, and reads go to the nearest replica (PA/EL)."
- "Payments and username claims need one strongly consistent source, so they go through the leader with synchronous replication, accept the extra latency, and fail rather than diverge if the leader is unreachable (PC/EC)."

Good questions to ask: "If two regions lose contact, should users in the smaller one still be able to write?" and "How many milliseconds of write latency is acceptable?" The answers place each operation in the PACELC grid.

<div class="ex">
<div class="lab">Worked example</div>

**global shopping cart**

A store runs in two regions. During a partition between them, the cart stays writable in both (PA): a user adding items in either region succeeds, and carts are merged afterwards by taking the union of items (a removed item may reappear, which the business accepts).

In normal operation, cart writes are acknowledged locally and replicated asynchronously (EL), so adding to the cart takes 20 ms instead of 120 ms.

Checkout, by contrast, goes to the region that owns the order and uses a strongly consistent inventory decrement (PC/EC). If that region is unreachable, checkout shows "try again shortly" rather than risk selling the last item twice.

</div>

## Common mistakes

- **"We chose CA."** Not possible for a distributed system; partitions happen.
- **Treating CAP consistency as ACID consistency.**
- **Labelling a whole database CP or AP** when its behaviour depends on per-query settings.
- **Ignoring the normal case.** The latency you pay every day matters more than the rare partition.

<div class="take">
<div class="lab">Key takeaway</div>

CAP only covers what happens during a partition and uses very strict definitions, so whole-system "CP" or "AP" labels mislead. PACELC adds that even without partitions, replicated systems trade latency against consistency on every request, which is the choice users feel every day; make it per operation.

</div>
