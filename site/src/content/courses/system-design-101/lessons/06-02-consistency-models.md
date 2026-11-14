---
id: "06.02"
course: "system-design-101"
module: "06"
slug: "consistency-models"
title: "Consistency models"
keyIdeas: ["A consistency model is a promise about what reads can return given concurrent writes.", "**Linearizable** (strong): every operation appears to happen at a single instant between its start and end; everyone sees the latest write.", "**Sequential**: all nodes see operations in the same order, not necessarily real-time order.", "**Causal**: operations that are causally related are seen in order; concurrent ones may differ.", "**Eventual**: replicas converge if writes stop; reads may be stale meanwhile.", "Session guarantees: read-your-writes, monotonic reads.", "Linearizability is about single objects and real time. Serializability is about transactions over many objects and says nothing about real time. Strict serializability is both."]
question: "What's the difference between linearizability and serializability?"
answer: "Linearizability is a guarantee about single operations on one object: each operation appears to take effect at one instant between its start and end, consistent with real time, so a read always sees any write that completed before the read began and the system behaves as if there were one copy. Serializability is a guarantee about transactions over many objects: the outcome equals some serial order of the transactions, which prevents anomalies like write skew but need not match real time, so a serializable system may legally return stale data. Strict serializability requires both properties together."
sources: [{"title": "Jepsen: Consistency models", "url": "https://jepsen.io/consistency", "note": "Linearizable, sequential, causal, eventual."}, {"title": "Werner Vogels: Eventually Consistent", "url": "https://www.allthingsdistributed.com/2008/12/eventually_consistent.html", "note": "Read-your-writes and session guarantees."}]
---

## What a consistency model is

When several clients read and write shared data at the same time, and that data may be copied across several machines, you need rules for what a read is allowed to return. A **consistency model** is exactly that: a contract stating which **histories** (sequences of operations with their start times, end times and results) the system is allowed to produce.

A stronger model forbids more strange outcomes (called **anomalies**), which makes programs easier to write because you can reason as if there were one copy and one user at a time. Stronger models usually cost latency (replicas must coordinate before answering) or availability (some nodes must refuse requests when they cannot coordinate). Weaker models are faster and more available but push the reasoning onto you.

The models form a hierarchy. At the top sits **strict serializability**. Below it there are two branches. One branch is about transactions: serializable, snapshot isolation, repeatable read, read committed. The other is about single objects and replicas: linearizable, sequential, causal, then the session guarantees and plain eventual consistency.

## Two families that people mix up

The words sound alike, but there are two separate questions:

- **Isolation** (transactions over **multiple objects**): what can concurrent transactions see of each other's partial work? Serializable, snapshot isolation and read committed live here (lesson 03.06 covers them).
- **Consistency of replicated data** (operations on **single objects**, in **real time**): when one client's write completes, will every later reader see it, no matter which replica they ask? Linearizable, sequential, causal and eventual live here.

A database can be strong on one axis and weak on the other. A single-node PostgreSQL in serializable mode is serializable and, because there is one copy, also linearizable. Add asynchronous read replicas and reads from those replicas are no longer linearizable, even though transactions on the primary are still serializable.

~~~dg-tree The consistency hierarchy. Each model implies the ones below it on its branch; strict serializability sits on top of both.
Strict serializable
  Serializable | transactions, any serial order
    Snapshot isolation
      Read committed
  Linearizable | one object, real-time order
    Sequential | one global order, not real time
      *Causal | causes before effects
        Read-your-writes, monotonic reads
          Eventual
~~~

## Linearizability

**Linearizability** (also called atomic consistency or strong consistency) is about operations on a **single object**, such as a key or a register. The rule: each operation appears to take effect atomically at a single point in time somewhere between when it started and when it finished, and that order respects **real time**. If write A finishes before read B starts, B must see A's value or something newer.

The intuition is that the system behaves **as if there were exactly one copy of the data**, even if there are five replicas. Once any client has seen the new value, no client may later see the old one.

Why it matters: linearizability is what you need for anything that relies on "the latest value" being unique. A lock (two clients must not both see "unlocked"), a unique username (two sign-ups must not both see "available"), a leader lease, an account balance check before a withdrawal.

Its cost: it cannot stay available during a network partition. A replica cut off from the others cannot know whether a newer write happened elsewhere, so it must refuse to answer (lesson 06.04). It also costs latency in normal operation, because a write must reach a quorum or the leader must confirm it is still the leader before a read.

~~~dg-seq A linearizability violation: a read that starts after a write completed still returns the old value from a lagging replica.
Client A > Replica 1 : write x = 2
Replica 1 -> Client A : ok (write complete)
Client B > Replica 2 : read x (replica lagging)
Replica 2 -> Client B : x = 1, not linearizable
~~~

<div class="ex">
<div class="lab">Worked example</div>

**is this history linearizable?**

Register x starts at 0. Client A runs write(x = 1) from t = 0 to t = 10 ms. Client B runs read(x) from t = 2 to t = 4 ms and gets 1. Client C runs read(x) from t = 6 to t = 8 ms and gets 0.

B's read overlaps A's write, so B may see 0 or 1; seeing 1 means the write's "instant" was before t = 4 ms. C's read starts at t = 6 ms, after B's read finished with 1. In real-time order, C must therefore see 1 or newer. It saw 0. This history is **not** linearizable, even though every read overlapped the write. Once anybody has seen the new value, nobody may see the old one.

</div>

## Serializability

**Serializability** is about **transactions**, which may touch many objects. The rule: the result of running transactions concurrently must equal the result of running them one at a time in **some** serial order. Crucially, that order does not have to match real time. A serializable database is allowed to place a transaction that started later before one that finished earlier, as long as the outcome matches some serial schedule.

So serializability protects you from anomalies **between** transactions, such as **write skew**: two doctors each check "is at least one other doctor on call?", both see yes, both go off call, and now nobody is on call. Under serializability one of the two transactions would have run first, and the second would have seen only one doctor left.

But serializability alone allows **stale reads** in real time. A serializable system could legally answer your read-only transaction from a snapshot taken a minute ago, ordering it "before" the latest writes. Some distributed databases do exactly this for read-only transactions to make them faster.

### Strict serializability

**Strict serializability** combines both: transactions are serializable, and the serial order respects real time. If transaction T1 commits before T2 starts, T2 sees T1's effects. Google Spanner offers this, using tightly bounded clock uncertainty (lesson 06.05) to order transactions across the globe. FoundationDB and CockroachDB (for single-key operations) aim for similar guarantees.

| | Linearizable | Serializable | Strict serializable |
| --- | --- | --- | --- |
| Unit | One operation on one object | Transactions over many objects | Transactions over many objects |
| Real-time order? | Yes | No | Yes |
| Prevents write skew? | No (no transactions) | Yes | Yes |
| Example violation | Read misses a write that already completed | Two transactions break an invariant together | Either of those |

~~~dg-matrix Which anomalies each model prevents. Linearizable and serializable guard against different problems; strict serializable guards against both.
Model | Stale read after write completes | Write skew | Lost update | Reading uncommitted data
Strict serializable | yes, prevented | yes, prevented | yes, prevented | yes, prevented
Serializable | no, allowed | yes, prevented | yes, prevented | yes, prevented
Linearizable (single key) | yes, prevented | no, not covered | partial, with compare-and-set | yes, prevented
Snapshot isolation | no, allowed | no, allowed | yes, prevented | yes, prevented
Eventual | no, allowed | no, allowed | no, allowed | partial
~~~

### Sequential consistency

Between linearizable and causal sits **sequential consistency**: all clients see all operations in one single order, and each client's own operations appear in the order it issued them, but that global order need not match real time. Your write might "take effect" after a read that began later. It is weaker than linearizability and rarely offered by name in databases, but you meet it in CPU memory models.

## Eventual consistency and its stronger cousins

At the weak end, describe consistency from the client's point of view:

- **Strong**: after a write completes, every later read sees it.
- **Weak**: reads may not see it. The period during which they might not is the **inconsistency window**.
- **Eventual**: if no new writes happen, all reads eventually return the last written value. DNS is the classic example: change a record, and resolvers around the world pick it up over minutes to hours as caches expire.

Eventual consistency on its own promises very little: no bound on how long "eventually" is, and no promise about what you see in the meantime. You might see a new value, then an old one, then the new one again. So there are useful in-between guarantees, which can be combined:

- **Read-your-writes**: you always see your own updates. Other people may not yet.
- **Session consistency**: read-your-writes within a session (a login, a connection).
- **Monotonic reads**: once you have seen a value, you never see an older one. Time does not go backwards for you.
- **Monotonic writes**: one client's writes are applied everywhere in the order it made them.
- **Writes follow reads**: if you read a value and then write, your write is ordered after the value you read.
- **Causal consistency**: if one operation could have influenced another (a reply to a comment), everyone sees them in that order. Unrelated operations may be seen in different orders by different people. Causal consistency is the strongest model that can stay available during a partition.

Monotonic reads and session consistency are the most practical for developers. Most "eventually consistent" product bugs are really missing session guarantees: a user posts a comment, refreshes, and it is gone because the read hit a lagging replica.

### How systems provide session guarantees

- **Read-your-writes**: route a user's reads to the primary for a short time after they write, or send the version (log position) of their last write with each read and only read from a replica that has caught up to at least that version.
- **Monotonic reads**: pin each user to one replica (by hashing the user ID), so they never jump from a fresh replica to a staler one. If that replica fails, move them to one that is at least as current.
- **Causal**: carry a dependency marker (a vector clock or version, lesson 06.05) with each write, and do not make a write visible on a replica until the writes it depends on are visible too.

<div class="ex">
<div class="lab">Worked example</div>

**the comment that disappears and reappears**

A user posts a comment at t = 0. The write goes to the primary. Replica R1 applies it at t = 0.2 s, replica R2 at t = 3 s.

At t = 1 s the user refreshes. The load balancer picks R1: the comment appears. At t = 1.5 s they refresh again and hit R2: the comment is gone. At t = 4 s it is back. Every step is "eventually consistent", but the user sees time go backwards.

Fix 1, monotonic reads: pin this user's reads to R1. Fix 2, read-your-writes: the write returns log position 5,041; the client sends it with reads, and R2 (at position 5,030) either waits briefly or forwards the read to the primary. Neither fix needs linearizability across all users.

</div>

## Choosing

Pick per operation, not per system:

- **Linearizable or serializable**: money movement, unique usernames, inventory decrements, seat booking, leader election, locks. Anything where two clients acting on stale data would break an invariant.
- **Causal**: comments and replies, chat messages, anything where effects must not be seen before causes.
- **Eventual plus session guarantees**: social feeds, like counts, view counts, recommendations, profile pictures. Strangers seeing a change a second late is fine; the writer not seeing their own change is a bug.

## Common mistakes

- **Saying "strongly consistent"** without saying whether you mean linearizable, serializable or both.
- **Assuming serializable implies fresh reads.** It does not; it allows reading from the past.
- **Assuming the C in ACID is the C in CAP.** ACID's C means "invariants hold", which is the application's job.
- **Paying for linearizability everywhere** when only a few operations need it.

## In an interview

When you choose a database or replication scheme, say which operations need which guarantee. "Username claims need a linearizable compare-and-set; the feed is eventually consistent with read-your-writes for the person who posted." That one sentence shows you understand the hierarchy and its costs.

<div class="take">
<div class="lab">Key takeaway</div>

linearizability is about single objects in real time: once anyone sees a new value, nobody sees the old one. Serializability is about transactions over many objects in some serial order, not necessarily real-time order. Strict serializability is both. Most applications need strong guarantees for a few operations and session guarantees such as read-your-writes for the rest.

</div>
