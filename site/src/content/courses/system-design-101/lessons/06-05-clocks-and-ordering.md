---
id: "06.05"
course: "system-design-101"
module: "06"
slug: "clocks-and-ordering"
title: "Clocks and ordering"
keyIdeas: ["Physical clocks drift, so timestamps from different machines can't reliably order events.", "**Lamport clocks**: each node keeps a counter, increments on each event, and on receiving a message sets counter = max(local, received) + 1. If A happened before B, A's timestamp is smaller. The reverse isn't true.", "**Vector clocks**: each node keeps a counter per node. Comparing vectors tells you if one event happened before another or if they're concurrent.", "Concurrent writes detected by vector clocks can be kept as siblings for the application to merge (Dynamo did this for shopping carts).", "Hybrid logical clocks combine physical time with a logical counter."]
question: "What can vector clocks detect that Lamport clocks can't?"
answer: "Vector clocks can detect that two events are concurrent: compare the vectors entry by entry, and if neither vector is less than or equal to the other in every entry, neither event could have caused the other, so they are concurrent (for example two conflicting updates made on different replicas during a partition). Lamport clocks only guarantee that a cause has a smaller timestamp than its effect; a smaller Lamport timestamp does not tell you whether the events were causally related or independent, because one integer cannot record what each process knew of the others."
sources: [{"title": "Leslie Lamport: Time, Clocks, and the Ordering of Events", "url": "https://lamport.azurewebsites.net/pubs/time-clocks.pdf", "note": "The original paper on logical clocks."}, {"title": "Kleppmann notes: Time, clocks and ordering", "url": "https://www.cl.cam.ac.uk/teaching/2122/ConcDisSys/dist-sys-notes.pdf", "note": "Section 3-4: Lamport and vector clocks."}]
---

## Why physical time is not enough

On one machine, ordering events is easy: read the clock, and the event with the smaller timestamp happened first. Across machines that breaks. Each machine has its own quartz clock, and quartz clocks **drift**, typically by 10 to 100 parts per million, which is up to about 8 seconds a day. NTP (lesson 04.11) pulls them back towards the true time, but only to within a few milliseconds on a good network, and much worse after a network hiccup. Clocks can even jump **backwards** when NTP corrects them.

So if machine A logs an event at 10:00:00.005 and machine B logs one at 10:00:00.003, you **cannot** conclude that B's happened first. A's clock might be 10 ms fast. Systems that order writes by wall-clock time (last-writer-wins) silently lose data this way: a write that genuinely came later, from a machine with a slow clock, is discarded as "older".

<div class="ex">
<div class="lab">Worked example</div>

**last-writer-wins loses a write**

Node A's clock is 15 ms fast; node B's is accurate. At true time 12:00:00.000 a client writes x = "red" via A, stamped 12:00:00.015. At true time 12:00:00.010 another client writes x = "blue" via B, stamped 12:00:00.010.

"blue" really happened 10 ms later, but its timestamp is smaller. When replicas compare timestamps, "red" wins everywhere and "blue" vanishes with no error. Any clock skew larger than the gap between two writes can reorder them.

</div>

The way out is to stop asking "what time was it?" and instead ask "**could this event have influenced that one?**" That idea, causality, is the foundation of logical clocks.

## Happens-before

Event a **happened before** event b, written a → b, if one of these holds:

- a and b are on the same process, and a came first in that process's order.
- a is the sending of a message and b is the receipt of that same message.
- There is some event c with a → c and c → b (transitivity: a chain of the first two rules).

If neither a → b nor b → a, the events are **concurrent**, written a ∥ b. Concurrent does not mean "at the same instant". It means neither could have influenced the other: no chain of messages connects them. The system genuinely cannot say which came first, and for many purposes it does not matter.

Happens-before is a **partial order**: some pairs of events are ordered, others are not. It captures exactly the causal structure of the system and nothing more.

~~~dg-seq Happens-before: a send happens before its receipt, so a1 causally precedes b1. Node C's event is connected to neither, so it is concurrent with both.
Node A > Node A : event a1
Node A > Node B : message m
Node B > Node B : event b1 (after receiving m)
Node C > Node C : event c1 (concurrent with a1 and b1)
~~~

## Lamport clocks

A **Lamport clock** is a single integer counter per process, designed so that causality always shows up in the numbers. The rules:

1. Before each local event (including a send), increment the counter.
2. Attach the counter value to every message you send.
3. When you receive a message carrying value m, set your counter to max(your counter, m) + 1, and use that as the receive event's timestamp.

This guarantees the **clock condition**: if a → b, then L(a) < L(b). Causes always get smaller numbers than their effects.

To get a **total order** (every pair of events ordered), break ties by process ID: compare (L, process ID) pairs. This total order is consistent with causality, which makes it useful. For example, if every replica applies commands in Lamport-timestamp order, they all apply them in the same order, which is the core of a replicated state machine. The same idea gives a mutual exclusion algorithm without a central coordinator: requests are granted in timestamp order.

~~~dg-seq Lamport clocks: on receipt, take max(local, received) + 1, so a message's receipt always gets a bigger number than its send.
Node A > Node A : local event, L = 1
Node A > Node B : send with L = 2
Node B > Node B : L was 0, now max(0, 2) + 1 = 3
Node B > Node C : send with L = 4
Node C > Node C : L was 7, now max(7, 4) + 1 = 8
~~~

### What Lamport clocks cannot tell you

The clock condition only works one way. L(a) < L(b) does **not** imply a → b. Two concurrent events still get different numbers, and from the numbers alone you cannot tell whether the smaller one caused the larger one or whether they were completely independent.

In the trace above, Node C's counter was already 7 from its own unrelated activity. If C had an unrelated event at L = 5, and A an event at L = 6, the numbers say 5 < 6 but there is no causal link. Lamport clocks lose this information because they squash all of a process's knowledge into one number.

There is also an anomaly with events outside the system. If a user does action A in one app, then phones a friend who does action B, B really happened after A, but no message inside the system connects them, so the system may order B first. Physical clocks with a **known, bounded** error can fix this: Google Spanner's TrueTime exposes clock uncertainty as an interval (typically a few milliseconds) and makes a transaction wait out that uncertainty before committing, so real-time order is respected even between unconnected clients.

## Vector clocks

A **vector clock** keeps one counter **per process**, so a process tracks how much it knows of every other process's history. For three processes A, B, C, a vector looks like [A:2, B:0, C:1].

1. On a local event or a send, increment **your own** entry.
2. Attach the whole vector to every message.
3. On receipt, take the **element-wise maximum** of your vector and the message's vector, then increment your own entry.

To compare two vectors V and W:

- If every entry of V is less than or equal to the matching entry of W, and they are not equal, then V < W: the event with V **happened before** the event with W.
- If V < W fails and W < V fails, the events are **concurrent**. For example [A:2, B:2, C:0] and [A:0, B:0, C:1]: the first has more of A and B, the second more of C.

This is exactly the converse that Lamport clocks lack: **V(a) < V(b) if and only if a → b**. So vector clocks can **detect concurrency**. That is their whole purpose.

~~~dg-timeline Version vectors (vector clocks for one data item) across two replicas of a cart. After a partition, neither vector dominates the other, so the store knows the edits were concurrent.
[Replica X] t1 | Cart created, [X:1, Y:0]
[Replica Y] t2 | Receives a copy (no edit), [X:1, Y:0]
[Replica X] t3 | Partition. Add book, [X:2, Y:0]
[Replica Y] t4 | Add pen, [X:1, Y:1]
[Replica X] t5 | *Sync: [X:2, Y:0] vs [X:1, Y:1], neither dominates, conflict
[Replica X] t6 | App merges to book + pen, [X:3, Y:1]
~~~

<div class="ex">
<div class="lab">Worked example</div>

**comparing vectors step by step**

Three events have vectors e1 = [A:1, B:0, C:0], e2 = [A:2, B:1, C:0] and e3 = [A:1, B:0, C:1].

e1 vs e2: 1 ≤ 2, 0 ≤ 1, 0 ≤ 0, not equal, so e1 → e2.

e1 vs e3: 1 ≤ 1, 0 ≤ 0, 0 ≤ 1, not equal, so e1 → e3.

e2 vs e3: A is 2 > 1 (so e2 is not before e3), but C is 0 < 1 (so e3 is not before e2). They are concurrent.

Lamport clocks on the same history might give e1 = 1, e2 = 3, e3 = 2. Reading those numbers you would conclude e3 came before e2, and you could not tell that e2 and e3 were independent edits that might conflict.

</div>

### The cost of vector clocks

A vector has one entry per process, so it grows with the number of writers. With 5 replicas that is trivial. With millions of client devices, it is not. Real systems use variations:

- **Per-replica entries** rather than per-client, so the vector is as long as the replica count.
- **Version vectors**: the same idea applied to replicas of a data item rather than to every event.
- **Dotted version vectors** (used by Riak) fix a subtle problem where several clients writing through the same replica produce false conflicts or lost updates.
- **Pruning**: drop the oldest entries when a vector grows past a limit, accepting occasional false conflicts.

## Hybrid logical clocks

A **hybrid logical clock (HLC)** combines a physical timestamp with a small logical counter. It stays close to wall-clock time (so timestamps are meaningful to humans and usable for "read as of 10:00"), but it always respects causality like a Lamport clock: a receipt never gets a timestamp below its send, even if the receiver's physical clock is behind. CockroachDB and YugabyteDB use HLCs to order transactions without special clock hardware.

~~~dg-matrix What each kind of clock gives you. Choose by what you need to know about two events.
Clock | Size | Respects causality | Detects concurrency | Close to real time
Wall clock | one timestamp | no, skew reorders | no | yes
Lamport clock | one integer | yes | no | no
Vector clock | one integer per node | yes | yes | no
Hybrid logical clock | timestamp plus counter | yes | no | yes
TrueTime interval | two timestamps | yes, with commit wait | no | yes, bounded error
~~~

## Where these appear

- **Conflict detection** in multi-leader and leaderless stores. Amazon's original Dynamo attached vector clocks to shopping cart versions, so two concurrent updates were both kept and merged by the application instead of one silently overwriting the other. Riak does the same with dotted version vectors.
- **Causal broadcast**: deliver messages so that causes always arrive before effects. A replica holds back a message until it has delivered everything the message's vector says came before it. Compare with FIFO broadcast (per-sender order) and total order broadcast (one global order, which needs consensus).
- **Last-writer-wins** with Lamport timestamps or HLCs gives a cheap total order, but silently discards one of two concurrent writes. Fine for a "last seen" timestamp, wrong for a shopping cart.
- **Distributed debugging and tracing**: ordering log lines from many services by causality rather than by skewed timestamps.

| | Lamport clock | Vector clock |
| --- | --- | --- |
| Size | One integer | One integer per node |
| If a → b | L(a) < L(b) | V(a) < V(b) |
| If L(a) < L(b) or V(a) < V(b) | Nothing certain | a → b |
| Detects concurrency | No | Yes |
| Gives total order | Yes, with tie-break | No (partial order) |

## Common mistakes

- **Ordering writes by wall-clock time** across machines and losing updates to skew.
- **Treating a Lamport order as causal order.** It is consistent with causality, but it invents an order between concurrent events.
- **Using vector clocks keyed by client** and letting them grow without bound.
- **Assuming concurrent means simultaneous.** It means "not causally connected", which can be seconds apart.

## In an interview

If your design merges replicas or accepts writes in several places, say how you detect conflicts. "Each version carries a version vector; if neither dominates, the writes were concurrent and we merge them" is a strong answer. If last-writer-wins is acceptable, say so explicitly and say what is lost.

<div class="take">
<div class="lab">Key takeaway</div>

happens-before is the true order of events in a distributed system, and wall clocks cannot be trusted to reflect it. Lamport clocks respect causality and give a cheap total order; vector clocks capture causality exactly, so they can tell concurrent events apart, at the cost of one counter per node.

</div>
