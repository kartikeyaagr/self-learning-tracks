---
id: "06.07"
course: "system-design-101"
module: "06"
slug: "leader-election"
title: "Leader election"
keyIdeas: ["Leader election picks one node to do work that must not be done twice: run a scheduler, own a partition, coordinate writes.", "Usually built on a consensus store (ZooKeeper, etcd) or a lease in a strongly consistent database: the leader holds a lease and renews it before it expires.", "The danger is two leaders at once, e.g. an old leader paused by GC wakes up still believing it's the leader.", "Defences: leases shorter than the time others wait before taking over, plus **fencing tokens** (an increasing number checked by the storage layer, which rejects writes from older leaders).", "Keep leader work small and make it safe to redo."]
question: "Two nodes both think they're the leader. What mechanism stops them corrupting data?"
answer: "A fencing token or epoch checked by the resource: each new leadership grant comes with a strictly higher number from the lock service or consensus system (a Raft term, an epoch, a ZooKeeper zxid), the leader includes it in every write, and the storage remembers the highest number it has seen and rejects any write carrying a lower one. So when the old leader resumes after a pause or partition and writes with its stale token, the write fails and it steps down. Conditional writes on an epoch or version column in the database achieve the same thing; checking one's own lease alone is not enough, because a pause can occur between the check and the write."
sources: [{"title": "AWS Builders' Library: Leader election", "url": "https://builder.aws.com/content/3Ev0vH0hfkcUizISUWYTvHibtcp/leader-election-in-distributed-systems", "note": "Leases and how AWS uses leaders."}]
tags: ["Code"]
---

## Why have a leader

**Leader election** is the process by which a group of identical processes picks one of them to be in charge of something: accepting writes for a shard, running scheduled jobs, assigning work, coordinating a migration. The others stand by, ready to take over.

Having one process in charge has real benefits:

- **One place to reason about.** Concurrency, logs and metrics all live in one process. Debugging "who did this?" has one answer.
- **Efficiency.** The leader can tell the others what happened, instead of the group reaching agreement on every single change. One consensus round to elect a leader can cover millions of decisions.
- **Easier consistency.** One node sees and orders every change, so ordering conflicts cannot arise between peers.
- **Simpler code.** Most programmers find "if I am the leader, do X" much easier to write correctly than a peer-to-peer protocol.

And real drawbacks:

- **A single point of failure.** When the leader dies, the work stops until a new one is chosen.
- **A single point of scaling.** All the work goes through one machine.
- **A single point of trust.** A buggy leader can do a lot of damage quickly, and followers will faithfully copy its mistakes.
- **Harder partial deployments.** You cannot easily test new leader code on 1% of traffic when there is only one leader.

The usual fix for the scaling problem is **sharding**: split the data into many shards and give each shard its own leader, so leadership is spread across the fleet. DynamoDB, Amazon EBS and EFS all work this way, as do Kafka (a leader per partition) and CockroachDB (a leader per range). It is also worth asking whether you need a leader at all. Workflows, idempotent APIs and optimistic locking (conditional writes) often solve the coordination problem without one.

## How elections usually work: leases

The most common practical approach is a **lease** stored in a strongly consistent store:

1. There is one record, say `leader/scheduler`, with fields: holder, lease duration, version.
2. A candidate tries a **conditional write**: "set holder = me, if the record is empty or its lease has expired". The store's strong consistency ensures only one candidate's write succeeds.
3. The winner is leader. It **heartbeats**: renews the lease well before it expires (for example every 3 s for a 10 s lease).
4. If the leader stops renewing (crash, partition, pause), the lease expires and another candidate's conditional write succeeds.

~~~dg-seq Leader election by lease: whoever holds the lease is leader and must renew it before it expires; when renewals stop, another node takes over.
Node A > Lock service : acquire lease, 10 s
Lock service -> Node A : granted, epoch 41
Node A > Lock service : renew at 3 s, 6 s, 9 s
Node A > Node A : crashes, renewals stop
Node B > Lock service : acquire (lease expired)
Lock service -> Node B : granted, epoch 42
~~~

### Use proven tools

Writing your own election code is a classic source of bugs. Use something tested: the DynamoDB lock client, ZooKeeper's leader election recipe, etcd's election API, Consul sessions, or Kubernetes' `Lease` objects (which is how Kubernetes controllers pick an active instance). Consensus systems such as Raft (06.06) elect leaders internally using terms. ZooKeeper offers a recipe using **ephemeral sequential nodes**: each candidate creates a node like `/election/n_0000000017`, which disappears automatically if its session dies, and the one with the lowest number is leader (lesson 06.12).

### Two implementation details that matter

- **Measure the lease with local elapsed time** (a monotonic timer that only counts forward), not by comparing wall-clock timestamps across machines. Wall clocks can jump (06.05). The lease holder should consider its lease over **earlier** than the server does, by a safety margin, to allow for clock drift and message delay.
- **Do not heartbeat from a separate background thread** that is unaware of the real work. If the work thread hangs, the heartbeat thread keeps a useless leader in office forever. If the heartbeat thread dies, the lease lapses while the work thread carries on as if nothing happened. Tie renewal to the health of the actual work.

## Two leaders at once

Here is the uncomfortable truth: **a distributed system cannot guarantee exactly one leader at every moment**. During failures there may briefly be zero leaders (between the old one dying and the new one being chosen) or two (the old one does not know it has been replaced). The election itself can be correct, with only one node granted the lease, and still two nodes behave as leader. How that happens:

- **A long pause.** The old leader is stopped by a garbage collection pause, a VM migration, a swap storm or a slow disk for longer than its lease. Its lease expires, a new leader is elected, and the old one resumes, still believing it is in charge. From its point of view no time passed.
- **A partition.** The old leader is cut off from the lock store but can still reach the database or file storage it writes to. It cannot renew, so its lease lapses, but its writes still land.
- **Check-then-act gaps.** Checking "do I still hold the lease?" before each write does not fix it on its own: the pause can happen **between** the check and the write.

~~~dg-timeline Two leaders at once: the election was correct, but the old leader's pause makes its view of the world stale.
0 s | A holds lease (epoch 41), renews every 3 s
4 s | A checks: lease valid until 13 s, starts a write
4.1 s | *A enters a 15 s garbage collection pause
13 s | A's lease expires on the lock service
13.5 s | B acquires the lease, epoch 42, starts writing
19.1 s | *A wakes, finishes its write as epoch 41
~~~

This is the same situation as lesson 06.01's "crashed or slow?" problem. The cluster decided A was gone because it was silent. A was merely slow, and now it acts on stale beliefs.

## What stops them corrupting data

Since you cannot prevent two nodes **believing** they lead, the protection must sit at the **resource being changed** (the database, the file, the queue), which sees every write and can reject the stale ones.

### Fencing tokens and epochs

Every time leadership is granted, the lock service hands out a number that **only goes up**: a term (Raft), an epoch, ZooKeeper's zxid, an etcd revision, a database sequence. The leader sends this number with every write. The storage remembers the highest number it has seen and **rejects** writes with lower numbers.

When the old leader wakes up and writes with token 41, storage has already seen 42 from the new leader and refuses. The old leader gets an error, realises it has been replaced, and steps down. Lessons 06.08 and 06.09 cover fencing tokens in detail.

~~~dg-seq Fencing with epochs: the storage remembers the highest epoch it has seen and rejects the paused old leader's write.
New leader B > Storage : write job 991, epoch 42
Storage -> New leader B : ok, highest epoch now 42
Old leader A > Storage : write job 991, epoch 41
Storage -> Old leader A : rejected, 41 < 42
Old leader A > Old leader A : step down
~~~

### Conditional writes

If the resource is a database, it can do the check itself: store the current leader's epoch in a row and make every write conditional on it. `UPDATE jobs SET status = 'dispatched' WHERE id = 991 AND leader_epoch <= 42` combined with bumping the epoch on takeover, or a DynamoDB `ConditionExpression`. The database's atomic conditional update is the fence.

### Other defences

- **Idempotent work**: if the leader's actions are idempotent and recorded durably before they are announced, two leaders mostly waste effort rather than corrupt state. Sending the same "dispatch job 991" twice is harmless if the worker de-duplicates on job ID.
- **Check the lease often**, especially just before side effects, and stop work immediately when renewal fails. This narrows the window but does not close it.
- **Keep an audit trail** of who led when, with epochs, so you can reconstruct incidents.
- **Model the algorithm.** Teams at AWS and elsewhere use formal specification languages such as TLA+ to check election and failover logic for these edge cases before writing code.

~~~dg-compare Protecting data from a stale leader. Checking your own lease narrows the window; only a check at the resource closes it.
[Leader checks its own lease]
Check lease -(pause here)-> Write -> Stale write lands
- Pause between check and write
- Partition hides lease loss
+ Simple, no storage changes
[*Resource checks the epoch]
Write with epoch -> Storage compares -> Stale write rejected
+ Works whatever the leader believes
+ Old leader learns and steps down
- Storage must support the check
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a scheduler leader**

A cron-like scheduler runs as three instances. The leader holds a lease in DynamoDB: 10 s, renewed every 3 s. Each takeover increments an epoch counter.

Leader A stalls in a 15 s garbage collection pause. Its lease expires at 10 s. B takes over with epoch 42 and starts dispatching jobs.

A wakes up and tries to mark job 991 as dispatched with epoch 41. The jobs table write uses a condition expression requiring the item's epoch to be at most the writer's epoch, and B already wrote epoch 42 on that item. So A's write fails with a conditional check error. A treats this error as "I am no longer leader", stops its loops and rejoins as a candidate. No job is dispatched twice.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing the lease**

You want failover within about 15 s. Lease 10 s, renew every 3 s: a leader survives up to two lost renewals (a brief network blip) without losing leadership.

On a crash, the worst case is: crash just after a renewal, lease expires 10 s later, candidates poll every 2 s, so a new leader is in place within about 12 s.

Shorter leases give faster failover but more false takeovers during garbage collection pauses or network blips. If pauses up to 5 s are common, a 3 s lease would cause frequent leader changes, each with a burst of rejected stale writes.

</div>

## Common mistakes

- **Believing the election guarantees one leader.** It guarantees one lease holder; it cannot stop an old holder acting.
- **No fence at the resource.** The single most common bug.
- **Comparing wall clocks** across machines to decide whether a lease expired.
- **Heartbeat threads that outlive the work**, or vice versa.
- **Leases shorter than common pause times.**

## In an interview

When your design has a leader (a scheduler, a shard owner, a primary), say how it is elected (lease in etcd, ZooKeeper or DynamoDB, or Raft internally), how fast failover is, and then volunteer the split-brain problem and the fix: "every write carries the leader's epoch, and the storage rejects older epochs". Interviewers often ask "what if the old leader comes back?", and this is the answer.

<div class="take">
<div class="lab">Key takeaway</div>

leader election usually uses a lease with heartbeats in a strongly consistent store, but no scheme can stop an old, paused or partitioned leader from believing it still leads. Protect the data at the resource, with fencing tokens, epochs or conditional writes that reject the stale leader.

</div>
