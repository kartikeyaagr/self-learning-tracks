---
id: "06.08"
course: "system-design-101"
module: "06"
slug: "distributed-locks-and-leases"
title: "Distributed locks and leases"
keyIdeas: ["A distributed lock lets only one process act on a resource at a time across machines.", "Locks need a lease (expiry), or a crashed holder would block everyone forever.", "Leases create a new problem: the holder might pause past expiry, another process takes the lock, and both now act.", "So for correctness, combine the lock with fencing: each lock grant comes with an increasing token, and the resource rejects requests with a token lower than one it has seen.", "Use locks for efficiency (avoid duplicate work) when occasional double execution is fine; for correctness, use consensus-backed locks plus fencing.", "Google's Chubby provided coarse-grained locks and small files for this."]
question: "Why must a lock have a lease (expiry)? What new problem does the lease create?"
answer: "Without a lease, a client that crashes, hangs or is partitioned while holding the lock would block every other client forever, because the lock service can never be sure it is dead, so the lock must expire unless renewed. The new problem is that expiry is decided by the lock service's sense of time, and a client paused by garbage collection, VM migration or a slow network may not notice its lease has gone: it can keep acting, even after checking the lease, while another client holds the lock, so two holders exist at once and can overwrite each other's work. Fencing tokens, increasing numbers checked by the resource, make that safe."
sources: [{"title": "Kleppmann: How to do distributed locking", "url": "https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html", "note": "Leases, GC pauses, fencing."}, {"title": "Google: The Chubby lock service (paper)", "url": "https://research.google/pubs/the-chubby-lock-service-for-loosely-coupled-distributed-systems/", "note": "Coarse-grained locks at Google."}]
---

## What a distributed lock is for

A **lock** guarantees **mutual exclusion**: only one holder at a time may do a particular thing. On one machine a mutex does this cheaply and reliably, because the operating system sees every thread and knows instantly when one dies. A **distributed lock** extends the idea across many machines: only one process anywhere in the fleet may process job 77, modify file X, run tonight's billing, or apply a schema migration.

Before reaching for one, ask the question that decides everything else: **why do you want the lock?**

- **Efficiency**: avoid doing the same work twice. If the lock occasionally fails and two processes run, you send a duplicate email, recompute a report or waste some CPU. Annoying, but not dangerous.
- **Correctness**: prevent concurrent changes that would corrupt data. If the lock fails, you lose an update, double-charge a customer or corrupt a file. Dangerous.

For efficiency, a simple lock that is right 99.99% of the time is perfect. For correctness, "usually right" is not good enough, and as this lesson shows, **no lock alone** can be fully right; the protected resource must take part.

~~~dg-compare Two reasons to lock, with very different requirements. Decide which one you have before choosing a mechanism.
[Efficiency lock]
Two holders -> Duplicate work -> Mild cost
+ A single Redis key is enough
+ Simple and fast
- Rare double runs, accept them
[*Correctness lock]
Two holders -> Corrupted data -> Real damage
+ Consensus-backed lock service
+ Fencing tokens checked by the resource
- More moving parts and latency
~~~

## Why locks need leases

Imagine a lock with no expiry. A client takes it and then crashes, hangs forever on a bug, or gets cut off by a network partition. The lock service has no way to know the client will never release it, so the lock is held forever and every other client waits forever. A single crashed machine deadlocks the whole system.

So distributed locks are **leases**: a lock granted for a limited time. The holder must either finish and release, or **renew** (extend) the lease before it expires. If it does neither, the lock service assumes it has failed and frees the lock for someone else. A lease is the distributed answer to "how do I know the holder is dead?": you do not know, so you decide in advance how long you are willing to wait.

Every serious lock service works this way:

- **Chubby**, Google's lock service built on Paxos and used by GFS and Bigtable to elect masters, provides coarse-grained, advisory locks held for hours or days, with sessions kept alive by periodic keep-alive messages. It prioritises availability and reliability over raw performance.
- **ZooKeeper** ties locks to **sessions**. A lock is an ephemeral node that disappears if the session's heartbeats stop for longer than the session timeout.
- **etcd** has explicit **leases** with a TTL; keys attached to a lease are deleted when it expires.
- **Redis** locks are keys with a TTL (`SET lock:job77 <token> NX PX 30000`).
- **Databases** can act as lock services: a row with an owner and an expiry time, taken with a conditional update.

### Choosing the lease length

The lease length is a trade-off:

- **Too short**, and healthy holders lose the lock during normal hiccups (a 2 second garbage collection pause, a slow disk write), causing needless handovers and more double-holder windows.
- **Too long**, and when a holder really crashes, everyone waits a long time before the lock is freed.

A common pattern: a lease several times longer than the expected worst pause (say 30 s), renewed at a third of that (every 10 s), with renewal failures causing the holder to stop work immediately.

~~~dg-state A lease-based lock as seen by the lock service. Expiry frees the lock whether or not the holder knows.
[*] -> Free
Free -> Held : acquire, token 33
Held -> Held : renew before expiry
Held -> Free : holder releases
Held -> *Expired : no renewal within TTL
Expired -> Held : another client acquires, token 34
~~~

## The problem leases create

Expiry solves the deadlock problem, but it creates a new one. The lease expires by the **lock service's** sense of time. The holder finds out later, if at all. Nothing stops the holder from continuing to act after its lease has gone. Here is the failure scenario, step by step:

1. Client 1 gets the lock with a 10 s lease, reads a file, and starts modifying it.
2. Client 1 hits a long **stop-the-world** garbage collection pause. Every thread freezes. The lease expires while it is frozen.
3. Client 2 gets the lock, reads the file, writes its changes, and releases.
4. Client 1 wakes up. From its point of view, no time has passed. It still believes it holds the lock, and writes its version of the file, **overwriting client 2's work**.

~~~dg-seq The lease problem: client 1 pauses past its lease, client 2 gets the lock and writes, then client 1 writes anyway.
Client 1 > Lock service : acquire (lease 10 s)
Client 1 > Client 1 : GC pause 15 s
Client 2 > Lock service : acquire (lease expired)
Client 2 > Storage : write version B
Client 1 > Storage : write version A, overwrites B
~~~

### Why checking does not help

The obvious fix is "check the lease is still valid just before writing". It does not work, because the pause can happen **between** the check and the write. Client 1 checks at t = 9.9 s ("still valid until 10 s"), and then pauses before the write leaves the process. There is always a gap, however small, between checking and acting, and any pause can land in it.

### Pauses are more common than you think

A process can stop for seconds at a time without crashing:

- **Garbage collection**: stop-the-world pauses in JVMs and other runtimes, occasionally many seconds on large heaps.
- **Virtual machine suspension or live migration** by the hypervisor.
- **Page faults and swapping** when memory is under pressure.
- **CPU starvation** on an overloaded host, or a noisy neighbour.
- **Network delays**: a write that leaves the client in time can sit in a queue or be retransmitted. GitHub once saw network packets delayed by about 90 seconds; a write sent while the lease was valid arrived long after it expired.
- **Clock jumps**: if expiry is computed from a wall clock and NTP steps that clock forward, a lease can expire early.

So a lease **guarantees no deadlock** (a crashed holder cannot block everyone forever) but **reintroduces two holders at the same time** whenever a holder is paused or slow. It swaps a liveness problem for a safety problem.

<div class="ex">
<div class="lab">Worked example</div>

**the double holder, in numbers**

A worker fleet uses a 30 s lease. The JVM's worst observed garbage collection pause is 4 s, and the host occasionally suffers a 40 s VM migration pause, about once a month across the fleet.

With 2,000 lock acquisitions an hour, most pauses are harmless (4 s is well inside 30 s). But the monthly 40 s pause, if it hits a lock holder mid-task, guarantees a double-holder window of about 10 s: the lease expires at 30 s, another worker takes over, and the first resumes at 40 s still writing.

A longer lease (60 s) would cover that pause but delays recovery from real crashes by a minute. No lease length removes the risk; it only changes how rare it is. That is why correctness needs a fence.

</div>

## Fencing tokens

The fix is to let the **resource** do the check, because only the resource sees the writes actually arriving, in the order they arrive.

1. Each time the lock service grants the lock, it returns a **fencing token**: a number that strictly increases with each grant (33, then 34, then 35).
2. The client includes the token with every write to the protected resource.
3. The resource remembers the **highest token it has processed** and **rejects** any write carrying a lower one.

Replay the scenario: client 1 got token 33 and paused. Client 2 got 34 and wrote; the storage recorded 34. When client 1 wakes and writes with 33, the storage sees 33 < 34 and refuses. Client 1 gets an error and learns it lost the lock.

~~~dg-seq Fencing tokens: the storage rejects a write carrying a token older than one it has already accepted.
Client 1 > Lock service : acquire, gets token 33
Client 1 > Client 1 : pause, lease expires
Client 2 > Lock service : acquire, gets token 34
Client 2 > Storage : write with 34, accepted
Client 1 > Storage : write with 33
Storage -> Client 1 : rejected, 33 < 34
~~~

Where tokens come from: anything the lock service guarantees to increase. ZooKeeper's **zxid** (transaction ID) or a znode's version; an etcd **revision** (every change to etcd increments a global revision); a database **sequence**; a Raft **term**. Chubby has the same idea under the name **sequencers**: a lock holder gets an opaque sequencer string that it passes to servers, which check it with Chubby (or compare it with the last one seen) before acting.

The resource must cooperate. That is easy for a database (`UPDATE ... WHERE lock_token < 34`) and for storage you build yourself; it is harder for systems you do not control, like a third-party API. For those, idempotency keys or making the operation naturally idempotent is the fallback. Lesson 06.09 looks at fencing again in the context of Redis's Redlock debate.

## Choosing a lock

- **Efficiency only**: a single Redis instance is fine. Acquire with `SET key <random token> NX PX ttl`; release with a short Lua script that deletes the key only if it still holds your token. Document that it can occasionally fail and make sure that failure is harmless.
- **Correctness**: a consensus-backed lock service (ZooKeeper, etcd, Consul) **plus** fencing tokens checked by the resource. The consensus backing ensures the lock service itself does not hand the same lock to two clients when one of its own nodes fails; the fencing handles paused clients.
- **Often simplest of all: skip the external lock.** If the data lives in a database, use the database's own guarantees. `SELECT ... FOR UPDATE` row locks, a unique constraint (only one row per job can be inserted), or a conditional update on a version column (optimistic concurrency). These are already enforced at the resource, so there is no gap between lock and write.

<div class="ex">
<div class="lab">Worked example</div>

**exclusive file processor**

Workers process uploaded files and write results to a results table. Worker A takes an etcd lease on `file/77` and receives revision 5012 as its token. It stalls for 40 s on a stuck disk; the lease (30 s TTL) expires.

Worker B acquires the lock and receives revision 5020. Both eventually finish and write results.

The results table stores the highest token per file and only accepts a write if its token is greater: `UPDATE results SET data = ?, token = 5020 WHERE file_id = 77 AND token < 5020`. B's result is accepted. A's late write, with 5012, matches no rows, so A sees "0 rows updated", logs that it lost the lock, and discards its work.

</div>

## Common mistakes

- **Locks without a TTL**, so a crash deadlocks everyone.
- **Releasing without checking ownership.** A client whose lease expired deletes the new holder's lock. Always compare the token before deleting.
- **Trusting "I checked the lease"** as protection against pauses.
- **Using an efficiency lock for correctness**, then losing data during a failover.
- **Lease shorter than common pauses**, causing constant handovers.

## In an interview

When a design needs mutual exclusion, say which kind ("this is a correctness lock"), name the service (etcd or ZooKeeper), give the lease length and renewal interval, and then volunteer the pause problem and the fencing token fix. Better still, ask whether the database's own row locks or conditional writes can do the job without a separate lock.

<div class="take">
<div class="lab">Key takeaway</div>

a lock must have a lease, or a crashed or partitioned holder blocks everyone forever. But the lease expires on the lock service's clock, so a paused or delayed holder can keep acting after someone else has the lock. Fencing tokens checked by the protected resource close that gap.

</div>
