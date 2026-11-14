---
id: "07.16"
course: "system-design-101"
module: "07"
slug: "partition-manager-and-map-table"
title: "Partition manager and map table"
keyIdeas: ["The partition manager owns the mapping of key ranges to partition servers, stored in a partition map table.", "Clients or front ends cache the map and route requests to the right server; on a miss or stale entry they refresh it.", "The manager monitors load and splits a hot range into two, moves partitions off overloaded servers, and merges cold ones.", "It also reassigns partitions when a server dies. Since data lives in a shared stream layer, moving a partition only means changing ownership and loading its index, not copying data.", "The manager must be highly available itself (replicated, with a lock-elected leader)."]
question: "What does a partition manager do when one range gets too hot?"
answer: "It first decides whether moving is enough: if the range's server is overloaded but the range itself could be handled by one server, it moves the range to a lightly loaded partition server (checkpoint, release lease, new owner loads the streams and replays a short commit log). If the range itself is too hot or too big, it orders a split: the owning server profiles the traffic, picks a split key that divides the load (not just the data), checkpoints, and creates two new partitions whose streams point at the same immutable extents, so no data is copied; the manager then updates the partition map table, moves one half to another server, and front ends route to the new owners. It repeats on later passes if a half is still hot, and merges cold neighbouring ranges to keep the partition count under control."
sources: [{"title": "Windows Azure Storage paper: Partition Layer", "url": "https://www.microsoft.com/en-us/research/publication/windows-azure-storage-a-highly-available-cloud-storage-service-with-strong-consistency/", "note": "Partition Manager and Partition Map Table."}]
---

## Why ranges need a manager

Lesson 07.15 showed that range partitioning keeps keys sorted, which makes prefix listing cheap, but leaves load uneven: some ranges grow huge, some get hot, some go cold. Someone has to watch every range and keep the assignment of ranges to servers healthy. That someone is the **partition manager**, and the table it maintains (which range lives on which server) is the **partition map**.

This lesson uses Windows Azure Storage (WAS) as the worked example, because its design is described in unusual detail and because the same pattern appears in Bigtable (its master and tablets), HBase (HMaster and regions), CockroachDB and Spanner (ranges and splits), and inside S3's automatic prefix scaling.

## The setting: a storage stamp

WAS groups hardware into **storage stamps**. A stamp is a cluster of 10 to 20 racks with 18 disk-heavy storage nodes per rack: about 2 PB of raw capacity in the first generation and up to 30 PB in the next. Stamps are kept around **70% utilised**, never above 80%, so there is room to absorb a rack failure and to give compaction and garbage collection space to work. A separate **Location Service** assigns each customer account to a stamp and updates DNS so the account's hostname points at it.

Inside a stamp there are three layers:

~~~dg-layers The three layers inside one WAS storage stamp. The partition layer, the subject of this lesson, sits between stateless front ends and an append-only stream layer.
Front-end servers | stateless: authenticate, route using cached partition map
*Partition layer | objects, RangePartitions, transactions, consistency
Stream layer | append-only extents, replicated three ways (lesson 07.18)
~~~

The front ends receive requests, authenticate them, look up which partition server owns the requested key in a cached copy of the partition map, and forward the request. The stream layer stores bytes. The partition layer turns bytes into objects (blobs, table rows, queue messages), orders transactions, and guarantees strong consistency for each object.

## The partition layer's parts

- **Partition Map Table**: records every RangePartition of every Object Table and which partition server is serving it. Front ends cache it to route requests; if they route to the wrong server (because the map changed), that server rejects the request and the front end refreshes.
- **Partition Manager (PM)**: splits Object Tables into RangePartitions, assigns each to **exactly one** partition server with no overlaps, and keeps load balanced. Several PM instances run for availability; the one holding the **leader lock** in the Lock Service is active, and the others wait to take over.
- **Partition Servers (PS)**: serve all requests for their assigned RangePartitions, about ten each on average. A PS may serve a partition only while it holds a **lease** for it from the Lock Service, which guarantees one owner at a time and therefore strong consistency: two servers can never both accept writes for the same key.
- **Lock Service**: a Paxos-based service (Paxos being a consensus algorithm that lets a group of machines agree despite failures) used for PM leader election and for PS leases.

### How a RangePartition is stored

Each RangePartition is stored **LSM-style** (lessons 07.09 to 07.11) on top of the stream layer, as a handful of streams:

~~~dg-tree One RangePartition's persistent state. Everything is an append-only stream in the shared stream layer, so any partition server can load it.
RangePartition B | keys from acme/m to globex/
  Metadata stream | the root: names the other streams, last checkpoint
  *Commit log stream | recent changes since last checkpoint
  Row data stream | checkpoints of rows plus their index
  Blob data stream | blob bytes, for blob tables only
  In memory on the PS
    Memory table | recent updates, like a memtable
    Index cache, row data cache
    Bloom filters | one per checkpoint
~~~

Writes go to the commit log stream and an in-memory **memory table**; periodically the memory table is checkpointed into the row data stream, and checkpoints are merged in the background. Reads check the memory table, then checkpoints, using Bloom filters to skip checkpoints that can't hold the key. It is the LSM tree from earlier lessons, built on a distributed append-only log instead of local files.

The crucial property: **all of a partition's state lives in the shared stream layer, not on the partition server's local disk**. A PS is just the current owner and cache. That is what makes everything in the next section cheap.

## How the PM reacts to load

The PM continuously collects load statistics for every RangePartition: transactions per second, average pending transactions, throttling rate, CPU usage, network usage, request latency and data size. Every **15 seconds** it sorts partitions by these measures and acts on a few of the worst. It has three tools.

### 1. Load balance: move a range

If a partition server is overloaded but no single range on it is too hot to handle on its own, the PM moves one or more busy RangePartitions to a lightly loaded PS. The sequence:

1. The PM tells the old PS to unload the partition. The PS writes a checkpoint (flushing its memory table), so the commit log to replay is short, and stops serving.
2. The PM assigns the partition to the new PS and updates the Partition Map Table.
3. The new PS acquires the lease, reads the metadata stream, replays the short commit log, and starts serving.

No data is copied. The new owner reads the same streams the old one wrote. The pause is typically well under a second for the moving partition.

### 2. Split: one hot range becomes two

If a single RangePartition is too big or too busy for any one server, it must be **split**. The PM decides **that** a split should happen; the PS that owns the partition decides **where**, because it has the detailed traffic data:

- A **size split** picks a key that roughly halves the data.
- A **load split** uses **Adaptive Range Profiling**, which tracks the distribution of requests across the key range, to pick the key that splits the **traffic** roughly in half. That can be very different from the middle of the data, if a small slice of keys is hot.

Then, for partition B splitting into C and D:

1. The PS checkpoints B and briefly stops serving it.
2. It uses a stream-layer operation called **MultiModify** to create new streams for C and D that **point at the same extents, in the same order, as B's streams**. Extents are immutable (lesson 07.18), so sharing them is safe. Each new partition simply ignores keys outside its range until compaction rewrites its data.
3. It starts serving C and D, and tells the PM.
4. The PM updates the Partition Map Table and typically moves one of the halves to another PS.

~~~dg-seq A split is metadata work only: both halves keep pointing at the same immutable extents, so no blob data is copied.
PM > PS-7 : split B
PS-7 > PS-7 : pick key from traffic
PS-7 > PS-7 : checkpoint B, pause
PS-7 > Stream layer : MultiModify C, D
PS-7 -> PM : serving C and D
PM > Map table : C: PS-7, D: PS-12
PM > PS-7 : unload D
PM > PS-12 : load and serve D
~~~

### 3. Merge: cold neighbours become one

Splits alone would leave the system with ever more, ever smaller partitions, each with its own memory and per-partition overhead. So the PM also **merges** adjacent cold RangePartitions: it moves them onto the same PS, which then combines them into one, again with a metadata operation over existing streams. The goal is to keep the total number of RangePartitions between low and high watermarks, typically around **ten times the number of partition servers** in the stamp.

~~~dg-state The life of a RangePartition under the partition manager. Most transitions change only metadata, because the data stays in the shared stream layer.
[*] -> *Served
Served -> Moving : server overloaded
Moving -> Served : loaded
Served -> Splitting : too hot or big
Splitting -> Served : split done
Served -> Merging : cold neighbours
Merging -> Served : merged
~~~

## What happens when a partition server fails

If a PS crashes, its leases expire. The PM notices (through the Lock Service and heartbeats), and reassigns each of the dead server's RangePartitions to live partition servers, spreading them out. Each new owner reads the partition's metadata stream, replays the commit log since the last checkpoint and starts serving. Again no data is copied; recovery time is dominated by commit log replay, which checkpointing keeps short. The lease protects correctness: even if the "dead" server was only partitioned from the network and is still running, its lease has expired, so it must stop serving before anyone else starts.

<div class="ex">
<div class="lab">Worked example</div>

**a viral container**

One customer's container suddenly receives 20,000 requests a second, all inside RangePartition B on PS-7, which can handle about 8,000. Requests start queueing and some are throttled. On the next 15-second pass, the PM sees B at the top of its list by pending transactions and throttling rate.

PS-7 profiles B's traffic and finds most requests fall in a narrow slice of keys. It picks a split key that divides the traffic, not the data, creating C and D with about 10,000 requests a second each. The PM moves D to idle PS-12. If C is still too hot, the next pass splits it again. Within seconds to a minute the load is spread over several servers, and not one byte of blob data was copied.

</div>

~~~dg-timeline The viral container, second by second, as the partition manager splits and moves the hot range.
0 s | Traffic on range B jumps to 20,000 per second, PS-7 saturated
12 s | *PM pass: B tops the list, split ordered
13 s | PS-7 checkpoints B, splits into C and D via shared extents
14 s | PM moves D to PS-12, map table updated
15 s | Front ends refresh map, D's traffic goes to PS-12
27 s | Next PM pass: C still hot, split again into C1 and C2
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why moving is cheap here and expensive elsewhere**

RangePartition B holds 400 GB of data. In a design where each server stores its partitions on local disk, moving B to another server means copying 400 GB over the network: at 1 GB/s that's nearly 7 minutes, during which the source is busier than ever. In WAS, B's 400 GB lives in the stream layer, so the move costs a checkpoint, a lease handover and a commit log replay of perhaps tens of megabytes: about a second. This separation of compute (partition servers) from storage (streams) is the same idea behind many modern cloud databases.

</div>

## Pitfalls

- **Splitting on data size only.** A range can be small and still hot; split on load too.
- **Splitting sequential hot spots.** If keys are increasing timestamps, the new last range is hot again immediately. Splitting buys time; the real fix is a key design that spreads writes.
- **No merging.** Endless splits leave millions of tiny partitions, bloating the map and the per-partition overhead.
- **Stale routing.** Front ends must cope with a moved partition by retrying and refreshing the map, not by failing the request.
- **Two owners.** Without leases from a consensus service, a slow or partitioned server can keep accepting writes after its partition moved, breaking consistency.

## In an interview

If your design range-partitions anything (object metadata, a sorted key-value store), mention the manager: "A partition manager tracks per-range load and size, splits hot ranges at a key chosen from the traffic, moves ranges off overloaded servers and merges cold ones; servers hold leases so each range has exactly one owner, and front ends cache the partition map." If storage is separated from compute, add that splits and moves are metadata-only, which is why they can react in seconds.

<div class="take">
<div class="lab">Key takeaway</div>

a partition manager owns the key-range-to-server map and keeps it balanced by moving, splitting and merging ranges, using leases so each range has exactly one owner. Keeping the data in a shared append-only stream layer makes all three operations cheap, metadata-only changes that take seconds instead of hours of copying.

</div>
