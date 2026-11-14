---
id: "07.18"
course: "system-design-101"
module: "07"
slug: "append-only-stream-layer"
title: "Append-only stream layer"
keyIdeas: ["At the bottom of Azure Storage is a stream layer: files that are append-only, made of extents. An extent is appended to until it's \"sealed\", then becomes immutable.", "Append-only storage makes replication simple: replicas just need to agree on the order and length of appends.", "Sealed extents can be erasure-coded in the background to save space, and moved or repaired without coordination with writers.", "Higher layers (tables, blobs, queues) are built as indexes over these logs, with checkpoints so recovery doesn't replay everything."]
question: "Why make the lowest storage layer append-only?"
answer: "With append-only extents, the only thing replicas can disagree about after a failure is how long the extent is, so the system can reconcile them simply: seal the extent at the smallest committed length (anything beyond it was never acknowledged, since acknowledgement requires all replicas) and force every replica to exactly that length, making them bitwise identical, then continue appending in a new extent. Sealed data never changes, so it can be erasure-coded, geo-replicated, cached, scrubbed and shared between streams without locks, which is what makes partition splits and merges metadata-only and recovery a matter of replaying logs. Appends are also sequential and disk-friendly. The cost is that deleted and overwritten data remains until higher layers compact live data and the stream manager garbage-collects unreferenced extents."
sources: [{"title": "Windows Azure Storage paper: Stream Layer", "url": "https://www.microsoft.com/en-us/research/publication/windows-azure-storage-a-highly-available-cloud-storage-service-with-strong-consistency/", "note": "Append-only extents and sealing."}]
---

## Why put a log at the very bottom?

Lessons 07.16 and 07.17 described the partition layer of Windows Azure Storage (WAS): the part that knows about objects, keys, ranges and transactions. Underneath it sits a layer that knows nothing about any of that. It stores **streams of bytes that can only be appended to**. Every blob, table row and queue message in the system, and every commit log and checkpoint of the partition layer, ends up as appends to these streams.

This is a deliberate and widespread design. Google's GFS and its successor Colossus, HDFS, Kafka's partitions, and the storage layer under Amazon Aurora all make the lowest layer append-only. This lesson walks through how the WAS stream layer works and then answers the real question: why is append-only the right contract for the bottom of a storage system?

## The building blocks

~~~dg-tree The stream layer's data model. A stream is a list of extents; an extent is a sequence of blocks; only the last extent of a stream accepts appends.
Stream //acct/partition-B/commitlog | named, ordered list of extent pointers
  Extent 17 | sealed, immutable, about 1 GB
    Block | up to 4 MB, own checksum
    Block
  Extent 18 | sealed, immutable
  *Extent 19 | unsealed: appends go here
    Block
    Block | newest append
~~~

- A **block** is the unit of reading and writing: up to 4 MB, appended atomically, with its own **checksum** that is verified on every read. Clients read whole blocks, so every byte read is checked.
- An **extent** is a sequence of appended blocks and is the **unit of replication**. Each extent is stored as a file on several extent nodes. The partition layer targets extents of about **1 GB**.
- A **stream** is a named, ordered list of pointers to extents. It looks to its user like one big append-only file, readable at any offset. Crucially, the **same extent can appear in more than one stream**, which is what lets the partition layer split a RangePartition by creating two streams that point at the same extents (lesson 07.16).
- Only the **last extent** of a stream can be appended to. When it reaches its target size, it is **sealed** at a block boundary and becomes **immutable forever**; a new extent is added to the stream for further appends.

Two kinds of server run the layer:

- The **Stream Manager (SM)** is a small Paxos cluster (Paxos being a consensus algorithm that keeps a group of replicas in agreement despite failures). It tracks the stream namespace, which extents make up each stream, and where every extent's replicas live. It is **off the critical path** of appends: clients ask it where to write only when a new extent starts. In the background it re-replicates extents that lost a replica, garbage-collects extents no stream references, and schedules erasure coding of sealed extents. Its entire state fits in about **32 GB of memory** for up to 50 million extents and 100,000 streams per stamp.
- **Extent Nodes (ENs)** store extent replicas as files on their local disks: the blocks, their checksums, and an index from offset to block. An EN knows nothing about streams, only about the extents it holds.

## How an append works

When a new extent is created, the SM assigns it **three replicas** on extent nodes in different **fault domains** (racks with independent power and network) and **upgrade domains** (groups of servers that are rebooted together during software rollouts), so no single rack failure or upgrade takes out two copies. One replica is the **primary**; the other two are **secondaries**. This assignment doesn't change while the extent is unsealed, so writers can talk to the replicas directly, without consulting the SM per append.

The append path:

1. The client (a partition server) sends the block to the extent's **primary** EN.
2. The primary chooses the **offset** at which the block goes, which orders concurrent appends from different clients.
3. It forwards the block and offset to both secondaries.
4. Each replica writes the append durably.
5. The primary acknowledges the client **only after all three replicas** have the append on disk.

~~~dg-seq An append from a partition server (PS) is acknowledged only when all three replicas have it durably on disk, at the offset chosen by the primary.
PS > Primary : append block
Primary > Primary : pick offset X, journal
Primary > Secondary 1 : append at X
Primary > Secondary 2 : append at X
Secondary 1 -> Primary : durable at X
Secondary 2 -> Primary : durable at X
Primary -> PS : success, offset X
~~~

Because each EN has many extents on many disks, appends from many streams interleave on the same disk, which would turn sequential writes into random ones. WAS solves that with a dedicated **journal drive** on each EN: every append is first written sequentially to the journal (and acknowledged as soon as it's there), then lazily copied to the extent's data disk. This cut average append latency from about **30 ms to 6 ms**. It is the same trick as an LSM's write-ahead log: make the durable write sequential, organise the data later.

## Sealing: how replicas agree after a failure

What if a replica fails or is unreachable during an append? The writer doesn't try to patch things up in place. It asks the SM to **seal** the extent:

1. The SM asks the reachable replicas for their current **commit length**: how many bytes they have durably.
2. It seals the extent at the **smallest** commit length among them. Any append beyond that point was never acknowledged to a client (acknowledgement requires all three), so dropping it loses nothing that was promised.
3. If a replica that was unreachable comes back later, the SM forces it to synchronise to **exactly** the sealed length, truncating or copying as needed.

After sealing, every replica of the extent is **bitwise identical**. The writer simply continues in a new extent with a fresh set of replicas, and the SM re-replicates the sealed extent to a healthy node in the background.

~~~dg-timeline A replica fails during appends. Sealing at the smallest commit length makes every copy identical without any complex repair protocol.
t0 | Appends to extent 9 on EN-A (primary), EN-B, EN-C
t1 | EN-C stops responding mid-append
t1 + 50 ms | Writer asks SM to seal extent 9
t1 + 80 ms | *SM reads lengths: A = 812 MB, B = 812 MB, seals at 812 MB
t1 + 90 ms | Writer opens extent 10 on EN-A, EN-D, EN-E and carries on
Later | EN-C returns, is synced to exactly 812 MB; SM adds a third copy elsewhere
~~~

The partition layer is designed to cope with this simply: it only ever reads data at offsets for which it received an acknowledgement, so it can never read a "phantom" append that only some replicas had.

~~~dg-state The life of an extent. Data is only ever appended while unsealed; after sealing it never changes, which enables everything that follows.
[*] -> *Unsealed
Unsealed -> Sealed : reached 1 GB, or a replica failed
Sealed -> ErasureCoded : background, to cut storage cost
Sealed -> Unreferenced : no stream points at it
ErasureCoded -> Unreferenced : no stream points at it
Unreferenced -> GarbageCollected : SM deletes replicas
~~~

## Reads

Reads can go to any replica of a sealed extent, since they're all identical. WAS sends reads with a **deadline**: if an extent node can't serve the read in time (because its disk is busy), it says so immediately, and the client tries another replica rather than waiting. For erasure-coded extents (lesson 07.19), the client can read more fragments than it needs and use the first ones to arrive. These tricks cut tail latency without any coordination.

## Why make the lowest layer append-only?

- **Replication becomes simple.** For a mutable block, replicas can disagree about content at any offset after a failure, and reconciling that needs a heavy protocol. For an append-only extent, there is exactly **one** thing replicas can disagree about: its **length**. Sealing at an agreed length settles it completely. No conflict resolution, no merging, no version vectors.
- **Immutability for everything above.** Sealed extents never change, so they can be erasure-coded, geo-replicated, cached, scrubbed, and **shared between streams** safely and without locks. That sharing is what makes partition splits and merges metadata-only operations that take seconds (lesson 07.16).
- **Sequential I/O.** Appends are disk-friendly, especially with the journal drive. Higher layers keep their own LSM-style indexes to find rows, so they never need in-place updates.
- **Simple crash recovery for the layer above.** A partition server that crashes leaves behind streams whose contents are exactly the acknowledged appends. Its replacement reads the metadata stream, replays the commit log from the last checkpoint and serves.
- **Clean separation of concerns.** The stream layer worries only about durability and placement; the partition layer worries only about objects and consistency. Each can be reasoned about and tested on its own.

~~~dg-compare Replicating mutable blocks versus append-only extents. With appends, the only possible disagreement after a failure is how long the extent is.
[Mutable blocks]
Write offset X -(replicate)-> 3 copies, any byte can change
- Replicas can differ anywhere after a crash
- Repair needs per-block versions or quorum reads
- Data can't be safely shared or cached
[*Append-only extents]
Append at end -(replicate)-> 3 copies, then seal
+ Only the length can differ
+ Seal at the smallest acked length
+ Sealed data shared, cached, erasure-coded
~~~

## What it costs

The usual bill for logs. Deleted and overwritten data stays in sealed extents until the partition layer compacts: it reads the live data out of old extents, writes it to new ones, updates its streams, and the SM then garbage-collects extents that no stream references. That costs I/O and requires spare capacity, one reason stamps are kept below 80% full.

<div class="ex">
<div class="lab">Worked example</div>

**a partition server crash**

PS-3 dies while appending to extent 9 of RangePartition B's commit log stream. The SM asks the extent nodes holding extent 9 for their lengths: 812.0 MB, 812.0 MB and 811.9 MB (the last never received the final append). It seals at 811.9 MB. The final append was never acknowledged to anyone, because acknowledgement needed all three replicas, so nothing promised is lost.

The PM assigns B to PS-8. PS-8 reads B's metadata stream to find the last checkpoint and the commit log position after it, replays about 30 MB of commit log into its memory table, and starts serving B, typically within a couple of seconds. No data was copied and no replica is in an ambiguous state.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the cost of garbage**

A RangePartition's blob stream holds 2 TB across 2,000 sealed 1 GB extents. Over a month, customers delete or overwrite 30% of the blobs. Those bytes stay in their extents, so 600 GB is dead. The partition layer's garbage collection reads the 1.4 TB of live data from extents that are mostly dead, rewrites it to new extents, and updates its index; the SM then deletes about 600 old extents. That's 1.4 TB read and 1.4 TB written (more with replication) to reclaim 600 GB, the log-structured trade-off from lesson 07.07, now at datacentre scale.

</div>

## In an interview

If you're designing a storage system from the bottom up, propose an append-only replicated log as the lowest layer: "Data is appended to extents replicated three ways across fault domains, acknowledged when all replicas have it; on any failure, the extent is sealed at the smallest committed length and writing continues in a new extent; sealed extents are immutable, so we can erasure-code, share and garbage-collect them freely." Then name the cost: compaction and garbage collection of dead data.

<div class="take">
<div class="lab">Key takeaway</div>

an append-only lowest layer means replicas can only disagree about length, so sealing at the shortest acknowledged length keeps every copy identical without a complex repair protocol. Sealed data is immutable, so everything above it can move, share, erasure-code and recover it cheaply; the price is garbage collection of dead data.

</div>
