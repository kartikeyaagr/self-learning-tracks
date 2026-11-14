---
id: "06.06"
course: "system-design-101"
module: "06"
slug: "consensus"
title: "Consensus"
keyIdeas: ["Consensus gets a group of nodes to agree on a value (or a log of values) even if some crash, as long as a majority is up.", "**Raft**: one elected leader accepts commands, appends them to its log and replicates to followers. An entry is committed once a majority has it.", "Leader election: followers that stop hearing heartbeats become candidates and request votes; a candidate with a majority becomes leader for a new term.", "Majority quorums guarantee any two majorities overlap, so a new leader always has all committed entries.", "With 5 nodes you tolerate 2 failures. With 3 of 5 down, the cluster stops accepting writes rather than risk inconsistency.", "Used in etcd, Consul, ZooKeeper (ZAB, similar), CockroachDB, Kafka KRaft."]
question: "Why does Raft need a majority to commit, and what happens with 2 of 5 nodes down?"
answer: "Raft needs a majority because any two majorities of the same cluster overlap in at least one server: so two candidates cannot both win the same term (the shared server votes only once), and a committed entry, being on a majority, is held by at least one server in every future election majority, which refuses to vote for a candidate missing it, so every new leader has all committed entries. A minority cannot commit, because two minorities could then decide different values. With 2 of 5 nodes down, the 3 survivors are still a majority, so they elect a leader and keep committing writes, though every write now needs all three and there is no margin left; if a third node fails, nothing can commit until one returns, but nothing incorrect is ever returned."
sources: [{"title": "The Raft site", "url": "https://raft.github.io/", "note": "Paper, visualisation, talks."}, {"title": "The Secret Lives of Data: Raft", "url": "https://thesecretlivesofdata.com/raft/", "note": "Step-by-step animation."}, {"title": "MIT 6.5840 Distributed Systems", "url": "https://pdos.csail.mit.edu/6.824/", "note": "Free lectures and labs; implement Raft."}]
---

## The problem consensus solves

**Consensus** means several servers agreeing on a value, such that once a value is decided, the decision never changes, even if some servers crash and restart. It sounds modest. It turns out to be the foundation for almost every strongly consistent distributed system.

The main use is a **replicated state machine**. Each server keeps a **log** of commands (`set x = 5`, `delete y`) and a **state machine** that applies them (for example a key-value table). If every server applies the same commands in the same order, every server ends in the same state, because the commands are deterministic. Consensus is how servers agree on that order: for each position (index) in the log, which command goes there.

With that, a cluster can keep a single consistent view of things that must never diverge: who the leader is, what the configuration is, which writes are committed, who holds a lock. It does so while tolerating failures of a minority of servers.

A correct consensus algorithm must guarantee:

- **Safety**: servers never decide different values for the same log index, and a decided value is never undone. This must hold no matter how messages are delayed or reordered.
- **Liveness**: the cluster eventually makes progress, as long as a majority of servers is running and can talk to each other with reasonable delays.

Recall from lesson 06.01 that in a fully asynchronous system no algorithm can guarantee both. Real algorithms keep safety unconditionally and rely on timing (timeouts) only for liveness. If the network misbehaves, they stall; they never give wrong answers.

**Paxos** solved consensus first and is famously hard to understand and implement. **Raft** was designed to give equivalent fault tolerance and performance while being easier to understand, by splitting the problem into three parts: leader election, log replication and safety.

## Raft's building blocks

### Terms

Raft divides time into numbered **terms**: 1, 2, 3 and so on. Each term begins with an election, and has **at most one leader**. If an election fails (nobody wins), the term ends with no leader and a new term starts. Terms act as a logical clock: every message carries the sender's term, and a server that sees a higher term than its own immediately updates its term and, if it was a leader or candidate, steps down to follower. Messages with a stale (lower) term are rejected.

### Server roles

Every server is in exactly one of three states:

- **Follower**: passive. Responds to requests from leaders and candidates. Every server starts here.
- **Candidate**: trying to become leader.
- **Leader**: handles all client requests and replicates log entries to followers.

~~~dg-state A Raft server's roles. Timeouts start elections; a majority of votes makes a leader; any higher term sends a server back to follower.
[*] -> Follower
Follower -> Candidate : election timeout, no heartbeat
Candidate -> Candidate : split vote, timeout, new term
Candidate -> *Leader : votes from a majority
Candidate -> Follower : sees current leader or higher term
Leader -> Follower : sees higher term
~~~

## Leader election step by step

1. The leader sends regular **heartbeats** (empty AppendEntries messages), for example every 50 ms.
2. Each follower has an **election timeout**, randomised for each server, for example between 150 and 300 ms. If it hears nothing from a leader in that time, it assumes the leader has failed.
3. It becomes a **candidate**: increments its term, votes for itself, and sends **RequestVote** to every other server.
4. Each server grants **at most one vote per term**, first come first served, and only to a candidate whose log is **at least as up to date** as its own. Raft defines up to date by the last log entry: a later term wins; with equal last terms, the longer log wins.
5. A candidate that collects votes from a **majority** (including itself) becomes leader and immediately sends heartbeats to assert authority.
6. If two candidates split the vote and nobody gets a majority, both time out (at different random moments) and try again in a new term. The randomisation makes repeated split votes rare: usually one server times out well before the others and wins before anyone else starts.

The "up to date" rule in step 4 is what makes elections safe for data. A committed entry is stored on a majority. A candidate needs votes from a majority. The two majorities overlap in at least one server, which holds the entry and will refuse to vote for a candidate missing it. So **a new leader always has every committed entry**.

~~~dg-timeline A leader crash and re-election in a 5-node Raft cluster with 150 to 300 ms election timeouts.
0 ms | Leader A (term 4) sends its last heartbeat, then crashes
150 ms | Followers start missing heartbeats
212 ms | *D's random timeout fires first: term 5, votes for itself
214 ms | C and E grant votes; D now has 3 of 5, a majority
215 ms | D becomes leader for term 5, sends heartbeats
240 ms | B's timeout would have fired, but it heard D first
~~~

## Log replication step by step

1. A client sends a command to the leader. (A follower that receives one redirects the client to the leader.)
2. The leader appends the command to its own log as a new entry tagged with the current term and the next index.
3. It sends **AppendEntries** to every follower in parallel. Each call includes the **index and term of the entry just before** the new ones.
4. A follower checks that its own log has an entry at that previous index with that term. If yes, it appends the new entries and replies success. If no, it rejects, and the leader steps back one entry (or more, with optimisations) and retries, until it finds the point where the logs agree. Then the follower deletes any conflicting entries after that point and takes the leader's.
5. Once the entry is stored on a **majority** of servers (the leader counts itself), it is **committed**. The leader applies it to its state machine and replies to the client.
6. The leader includes its latest commit index in later AppendEntries (and heartbeats), so followers learn which entries are committed and apply them too.

The consistency check in step 4 enforces the **Log Matching Property**: if two logs contain an entry with the same index and term, the logs are identical in all entries up to that index. Inductively, every successful AppendEntries proves the follower's log matches the leader's up to that point.

~~~dg-seq A Raft write in a 3-node cluster: the leader commits once a majority (itself plus one follower) has the entry.
Client > Leader : SET x = 5
Leader > Follower 1 : AppendEntries (term 3, index 7, prev 6/3)
Leader > Follower 2 : AppendEntries (term 3, index 7, prev 6/3)
Follower 1 -> Leader : ok, now 2 of 3 have it
Leader > Leader : commit index 7, apply x = 5
Leader -> Client : ok
Follower 2 -> Leader : ok (late, already committed)
~~~

### The subtle commit rule

A leader only counts replicas towards commitment for entries from its **own current term**. Entries from earlier terms become committed indirectly, when a later entry from the current term is committed after them. Why? Because an old entry that is on a majority could still, in a specific sequence of crashes, be overwritten by a leader from a different term that never saw it. Committing a current-term entry on a majority rules that out, because any future leader must then have that entry and everything before it. Practical systems have a new leader append a no-op entry at the start of its term to commit earlier entries promptly.

## What Raft guarantees

- **Election Safety**: at most one leader per term.
- **Leader Append-Only**: a leader never overwrites or deletes its own entries; it only appends.
- **Log Matching**: same index and term means identical logs up to there.
- **Leader Completeness**: a committed entry appears in the log of every later leader.
- **State Machine Safety**: no two servers ever apply different commands at the same index.

Raft needs one timing condition for liveness: **broadcastTime ≪ electionTimeout ≪ MTBF**. A heartbeat round trip (perhaps 0.5 to 20 ms) must be much shorter than the election timeout (150 to 300 ms locally, more across regions), which must be much shorter than the average time between server failures (months). Break the first inequality and the cluster holds elections constantly.

Two practical extensions: **membership changes** use **joint consensus**, where during the switch decisions need majorities of both the old and new configurations, so there is never a moment when two disjoint majorities could each elect a leader. And logs are compacted with **snapshots** that record the state machine plus the last included index and term, so the log does not grow forever.

## Why a majority

The whole algorithm rests on one fact: **any two majorities of the same cluster overlap in at least one server**. In a 5-node cluster, any 3 and any other 3 must share at least one node, because 3 + 3 = 6 > 5. From that:

- Two leaders cannot both win in the same term: each needs a majority of votes, and the overlapping server votes only once per term.
- A committed entry is on a majority, and any future leader needs votes from a majority, which includes at least one server holding that entry. With the up-to-date voting rule, committed data survives every leader change.

A minority cannot do either. Two servers out of five could disagree with the other three forever; if they could commit on their own, the two sides could commit different values at the same index.

| Cluster size | Majority | Failures tolerated |
| --- | --- | --- |
| 1 | 1 | 0 |
| 3 | 2 | 1 |
| 4 | 3 | 1 |
| 5 | 3 | 2 |
| 7 | 4 | 3 |

A cluster of 2f + 1 servers tolerates f failures. A fourth server does not help (4 servers still tolerate only 1, and you now need 3 to agree instead of 2), which is why clusters have odd sizes. Larger clusters tolerate more failures but every write waits for more acknowledgements, so 3 or 5 is typical; 7 is rare.

~~~dg-matrix What a 5-node Raft cluster can do as servers fail. It stops rather than risk inconsistent answers.
Servers up | Elect a leader | Commit writes | Linearizable reads | Margin left
5 of 5 | yes | yes | yes | 2 more failures
4 of 5 | yes | yes | yes | 1 more failure
3 of 5 | yes | yes | yes | none
2 of 5 | no | no | no | stalled until one returns
~~~

## When too many fail

A 5-server cluster keeps working with 2 failures; with more, it stops making progress but never returns an incorrect result.

<div class="ex">
<div class="lab">Worked example</div>

**2 of 5 down**

Nodes A to E, A is leader in term 7. A rack loss takes out A and B. C, D and E stop receiving heartbeats. D's timeout fires first; it starts term 8 and gets votes from C, E and itself: 3 of 5, a majority.

D becomes leader and keeps committing writes, each needing 3 acknowledgements (D plus C and E). Every write now needs **all** remaining servers, so one slow server slows every write, and there is no margin left.

Now C also fails. D and E are 2 of 5. D can append entries but can never reach 3 acknowledgements, so nothing commits. Clients time out. Linearizable reads also stop, because D cannot confirm it is still the leader. E's election attempts fail too. The moment A, B or C returns, a majority exists again and the cluster resumes, with no committed data lost.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**a partition with the old leader on the minority side**

In a 5-node cluster, leader A and node B are partitioned from C, D and E. A still thinks it is leader (term 7) and accepts a client write, appending it at index 50. It sends AppendEntries, but only B answers: 2 of 5, so index 50 is never committed and A never acknowledges the client.

On the majority side, C wins term 8 and commits different entries at index 50 and beyond. When the partition heals, A sees term 8, steps down, and C's AppendEntries overwrites A's uncommitted entry 50. The client that wrote via A got a timeout, not a success, so no acknowledged write was lost.

</div>

## Where it is used

etcd (the data store behind Kubernetes), Consul, CockroachDB, TiKV, YugabyteDB, Kafka's KRaft controller and many others use Raft. ZooKeeper uses a similar protocol called ZAB; Google's Chubby and Spanner use Paxos variants. Implementing Raft from scratch is a classic exercise in distributed systems courses, and is the best way to learn its corner cases.

## Common mistakes

- **Even-sized clusters**, which add cost without adding fault tolerance.
- **Spreading 3 nodes across 2 zones**, so losing the zone with 2 nodes stops the cluster. Use 3 zones.
- **Election timeouts close to network latency**, causing constant elections across regions.
- **Reading from followers** and expecting linearizable results.

## In an interview

Say "the metadata lives in a 3- or 5-node Raft group, so it tolerates 1 or 2 failures and stops rather than diverges beyond that". If asked why, give the overlap argument in one sentence: any two majorities share a server.

<div class="take">
<div class="lab">Key takeaway</div>

Raft elects one leader per term by majority vote and commits a log entry once a majority stores it. Because any two majorities overlap, there is never more than one leader per term and committed entries survive every leader change. A cluster of 2f + 1 servers tolerates f failures and stops, rather than giving wrong answers, beyond that.

</div>
