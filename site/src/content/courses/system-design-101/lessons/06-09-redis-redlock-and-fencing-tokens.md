---
id: "06.09"
course: "system-design-101"
module: "06"
slug: "redis-redlock-and-fencing-tokens"
title: "Redis Redlock and fencing tokens"
keyIdeas: ["Redlock acquires a lock on a majority of N independent Redis nodes with the same random value and TTL. If it gets a majority within a time limit, it holds the lock.", "The main critique: Redlock depends on timing assumptions (bounded pauses, clock drift), and it doesn't produce fencing tokens, so a paused client can still write after its lock expires.", "antirez's reply argues the timing assumptions are reasonable in practice.", "**Fencing token**: a number that increases on every lock grant. The storage service records the highest token it has seen and rejects writes carrying a lower one, so a stale holder can't corrupt data.", "Takeaway: a single Redis lock is fine for efficiency; use something with fencing for correctness."]
question: "How does a fencing token protect storage from a client whose lock expired?"
answer: "Each lock grant comes with a strictly increasing number (for example an etcd revision or ZooKeeper zxid). The client sends that token with every write, and the storage atomically remembers the highest token it has accepted for the resource and rejects any write carrying a lower one. So if a client's lock expired while it was paused and a newer client has already written with a higher token, the stale client's write is refused (for example a conditional UPDATE ... WHERE lock_token < 34 matches no rows), regardless of what the stale client believes about holding the lock."
sources: [{"title": "Redis docs: Distributed locks (Redlock)", "url": "https://redis.io/docs/latest/develop/use/patterns/distributed-locks/", "note": "The algorithm."}, {"title": "Kleppmann: How to do distributed locking", "url": "https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html", "note": "The critique and fencing tokens."}, {"title": "antirez: Is Redlock safe?", "url": "http://antirez.com/news/101", "note": "The reply."}]
tools: ["Redis"]
---

## Locks on a single Redis

Redis is the most common place teams build a distributed lock, because it is already there, it is fast (a lock round trip is well under a millisecond) and it has the two primitives a lease needs: set-if-absent and expiry. The basic recipe has two halves.

**Acquire**: `SET resource my_random_value NX PX 30000`. `NX` sets the key only if it does not exist, so only one client can succeed. `PX 30000` attaches a 30 second expiry, which is the lease (lesson 06.08). The value is a **random token** unique to this client and this acquisition, for example 20 random bytes. It identifies the owner.

**Release**: delete the key **only if it still holds my token**, atomically, with a short Lua script:

```
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
```

Why the check? Suppose client A's lease expired while it was slow, and client B acquired the lock. If A then ran a plain `DEL`, it would delete **B's** lock, and a third client could acquire it while B is still working. The token comparison prevents a client from releasing a lock it no longer owns. The comparison and the delete must be in one script because Redis runs scripts atomically; a separate `GET` then `DEL` could be split by another client's `SET`.

### The replication weakness

A single Redis node is a single point of failure, so production Redis usually has a primary and a replica with automatic failover. That breaks the lock in a specific way, because Redis replication is **asynchronous**:

1. Client A acquires the lock on the primary.
2. The primary crashes before the key is copied to the replica.
3. The replica is promoted to primary. It has no lock key.
4. Client B acquires the same lock on the new primary.

Now A and B both hold the lock. The window is small (milliseconds of replication lag), but at scale, with failovers happening, it will occur.

## Redlock

**Redlock** is an algorithm designed to avoid that single point of failure without relying on replication. It uses **N independent Redis primaries** (typically 5), with no replication between them, ideally on separate machines or zones. To acquire a lock:

1. Read the current time in milliseconds.
2. Try to set the key on each of the N nodes in turn, using the same key and the same random value, with `NX PX ttl`. Use a short per-node timeout (for example 5 to 50 ms for a 10 s TTL), so a dead node does not stall the whole attempt.
3. Compute how long acquisition took. The lock is acquired **only if** a majority of nodes (at least 3 of 5) succeeded **and** the total time spent is less than the TTL.
4. The lock's **validity time** is the TTL minus the time spent acquiring, minus an allowance for clock drift between nodes.
5. If acquisition failed (fewer than a majority, or validity time not positive), release the key on **all** nodes, including those that seemed to fail, then retry after a random delay so competing clients do not keep colliding.

The majority requirement means that if one node crashes and loses its keys, the lock is still held on enough other nodes to stop a second client getting a majority. But a node that restarts **empty** could tip the balance: if A held the lock on nodes 1, 2 and 3, and node 3 restarts without the key, B could get 3, 4 and 5. So crashed nodes should stay down for at least the maximum TTL before rejoining (**delayed restart**), or persist with `fsync` on every write, so a restarted node cannot forget a lock it granted.

~~~dg-seq Redlock: acquire the same key with the same token on a majority of 5 independent Redis nodes, quickly enough that time remains on the lease.
Client > Client : t0 = now, ttl = 10 s
Client > Redis 1 : SET k token NX PX 10000 (ok)
Client > Redis 2 : SET k token NX PX 10000 (ok)
Client > Redis 3 : SET k token NX PX 10000 (timeout)
Client > Redis 4 : SET k token NX PX 10000 (ok)
Client > Redis 5 : SET k token NX PX 10000 (taken)
Client > Client : 3 of 5 in 40 ms, validity about 9.9 s
~~~

<div class="ex">
<div class="lab">Worked example</div>

**computing Redlock's validity time**

TTL is 10,000 ms. The client tries 5 nodes with a 50 ms timeout each. Nodes 1, 2 and 4 succeed in 3 ms each; node 3 times out after 50 ms; node 5 refuses because another client holds the key there. Total elapsed: 3 + 3 + 50 + 3 + 3 = 62 ms.

3 of 5 is a majority. Validity = 10,000 - 62 - drift allowance. With an allowance of 1% of TTL plus 2 ms (102 ms), validity is about 9,836 ms. The client must finish its work, or extend the lock, within that time.

If node 3 had taken 9,950 ms to time out instead, validity would be negative and the client would release everything and retry. That is why per-node timeouts must be much smaller than the TTL.

</div>

## Why Redlock is unsafe for correctness

Redlock is controversial. The main argument against it, made by distributed systems researchers, goes like this.

### No fencing tokens

Redlock's token is a **random** value, not an increasing number. The protected storage cannot compare two random values and decide which holder is newer, so it cannot reject a stale holder. Lesson 06.08 showed that with any lease-based lock, a paused client can wake up after its lease expired and keep writing. The standard defence is a fencing token checked by the resource, and Redlock does not produce one. Generating a safely increasing token across 5 independent nodes would itself need consensus, which Redlock was designed to avoid.

### Timing assumptions

Redlock's safety depends on **timing**: bounded network delay, bounded process pauses and clocks that run at roughly the right rate. Real systems violate all three occasionally (06.01). The sharpest example is a clock jump:

~~~dg-timeline A clock jump on one Redis node lets two clients each believe they hold a majority.
t = 0 s | *Client A acquires on nodes 1, 2, 3 (TTL 10 s)
t = 1 s | Network cuts A off from nodes 4 and 5
t = 2 s | Node 3's clock jumps forward 10 s (NTP step)
t = 2 s | Node 3 thinks A's key expired, deletes it
t = 3 s | *Client B acquires on nodes 3, 4, 5: also a majority
t = 3 s | A and B both believe they hold the lock
~~~

Redis stores key expiry as an absolute wall-clock time, so a node whose clock jumped forward could expire a lock early, letting a second client win a majority. A process pause has the same effect without any clock problem: a client gets the majority, pauses in garbage collection past the TTL, and resumes acting while another client holds the lock. Redlock does measure time spent **during** acquisition, but it cannot detect a pause that happens **after** acquisition, between deciding "I hold the lock" and performing the write.

### The conclusion of that argument

Redlock is **too heavy for efficiency locks** (where a single Redis node is enough, and the occasional double run is acceptable) and **not safe enough for correctness locks** (where you need a consensus system such as ZooKeeper or etcd, which also provides increasing tokens, plus fencing at the resource).

## The counter-argument

Redlock's defenders make several points worth knowing:

- If your storage can check fencing tokens, it can already perform atomic compare-and-set, so it is effectively linearizable for that resource and you barely need the external lock. A random unique token, stored and compared with compare-and-set, gives similar protection: write only if the resource's current owner value matches mine.
- A client pausing after acquiring a lock is a problem for **every** auto-expiring lock system, including ZooKeeper and etcd, not just Redlock. Fencing is the answer in all cases.
- Redis should use a **monotonic clock** for expiry (time that only moves forward, unaffected by NTP steps), and operators should configure NTP to **slew** (adjust gradually) rather than step, and avoid manual clock changes.
- Redlock measures elapsed time during acquisition, so delays and pauses **during** acquisition are detected.

Opinions on Redlock still differ. The practical consensus: **Redlock, or a single Redis lock, is fine for avoiding duplicate work. For anything where two holders would corrupt data, put a check at the resource**, and ideally get the lock and its token from a consensus-backed service.

~~~dg-matrix Lock options compared. Only consensus plus a check at the resource is safe against paused clients.
Option | Survives a node crash | Increasing token | Safe if client pauses | Good for
Single Redis key | no | no | no | efficiency
Redis primary plus replica | partial, failover can lose it | no | no | efficiency
Redlock on 5 nodes | yes | no | no | efficiency
etcd or ZooKeeper lease | yes | yes | no, alone | correctness with fencing
Any lock plus fenced writes | yes | yes | yes | correctness
~~~

## How a fencing token protects storage

The mechanism is simple once you see it:

1. The lock service issues token 33 to client A.
2. A pauses; its lease expires. The service issues token 34 to client B.
3. B writes to storage with token 34. Storage accepts and records 34 as the highest token seen for this resource.
4. A resumes and writes with token 33. Storage sees 33 < 34 and rejects the write.

The storage does the enforcement, so it does not matter that A wrongly believes it still holds the lock. A's beliefs can be as stale as they like; its writes cannot land. In a database, the check is a conditional update:

`UPDATE accounts SET balance = ?, lock_token = 34 WHERE id = 1 AND lock_token < 34`

If A later runs the same statement with 33, the `WHERE` clause matches no row, so "0 rows updated" tells A it lost the lock.

~~~dg-seq A fencing token in action: storage keeps the highest token seen and refuses anything older.
Lock service -> Client A : token 33
Client A > Client A : pause, lease expires
Lock service -> Client B : token 34
Client B > Storage : write with 34
Storage -> Client B : ok, highest now 34
Client A > Storage : write with 33
Storage -> Client A : rejected, 33 < 34
~~~

Two requirements make it work. Tokens must be **strictly increasing** across grants, which is why they come from a consensus system (an etcd revision, a ZooKeeper zxid). And the check-and-write at the resource must be **atomic**; a separate "read highest token, then write" would reopen the race.

<div class="ex">
<div class="lab">Worked example</div>

**Redis lock for a report job versus a balance update**

Generating a daily report twice is wasteful but harmless. A single Redis lock is enough: `SET report:2026-11-09 host-7-<random> NX PX 600000`, released with the token-checking script. If a failover lets two hosts run it, two identical reports are written and the second overwrites the first. Nobody is hurt.

Updating an account balance is not harmless. There the service uses the database row itself: each account has a `version` column, the update is `UPDATE accounts SET balance = 150, version = 8 WHERE id = 1 AND version = 7`, and a writer that sees "0 rows updated" re-reads and retries. No Redis lock is involved at all; the database's atomic conditional update is both lock and fence.

</div>

## Common mistakes

- **Plain `DEL` to release**, deleting another client's lock after your lease expired.
- **Using Redlock or Redis for a correctness lock** with no check at the resource.
- **Per-node timeouts close to the TTL**, leaving no validity time.
- **Restarting a Redlock node immediately** with no persistence.
- **Separate read and write** for the fencing check.

## In an interview

If someone proposes "a Redis lock", ask whether it is for efficiency or correctness. For efficiency, agree and mention the token-checked release. For correctness, explain the pause problem and say: "the resource must reject stale holders, so I'd use an etcd lease's revision as a fencing token, or skip the lock and use a conditional write on a version column".

<div class="take">
<div class="lab">Key takeaway</div>

a single Redis lock, or Redlock across several nodes, is fine for avoiding duplicate work. For correctness a lock alone is never enough, because a paused client can outlive its lease: the protected resource must reject stale holders using increasing fencing tokens or conditional writes.

</div>
