---
id: "07.19"
course: "system-design-101"
module: "07"
slug: "durability-and-replication"
title: "Durability and replication"
keyIdeas: ["Durability comes from storing redundant data across independent failure domains (disks, racks, zones).", "**Replication**: keep 3 full copies. Simple, fast repair and reads, but 200% storage overhead.", "**Erasure coding**: split data into k data chunks plus m parity chunks; any k of the k+m rebuild the data. 10+4 survives 4 losses with only 40% overhead, but repair reads from 10 nodes and costs more CPU and network.", "Common pattern: replicate fresh data for speed, erasure-code it once sealed.", "Async replication to another region protects against a whole-region disaster, with some lag."]
question: "3x replication vs 10+4 erasure coding: compare storage overhead and repair cost."
answer: "3x replication stores 3 bytes per byte (200% overhead) and tolerates 2 failures; reads use any one copy, and repairing a lost copy just reads 1 unit from a surviving replica. 10 + 4 erasure coding stores 1.4 bytes per byte (40% overhead) and tolerates any 4 failures, so it is far cheaper for the same or better durability, but rebuilding one lost shard requires reading 10 shards from 10 other machines, so repair traffic after a disk failure is about 10 times higher, and healthy reads must gather 10 shards. That is why systems replicate fresh data on the write path and erasure-code large sealed data in the background."
sources: [{"title": "Windows Azure Storage paper: replication", "url": "https://www.microsoft.com/en-us/research/publication/windows-azure-storage-a-highly-available-cloud-storage-service-with-strong-consistency/", "note": "Intra-stamp sync, inter-stamp async replication."}, {"title": "Backblaze: Reed-Solomon erasure coding", "url": "https://www.backblaze.com/blog/reed-solomon/", "note": "Durability with less overhead than 3 copies."}]
---

## The problem: disks die, constantly

**Durability** is the probability that data you stored is still there, intact, when you come back for it. It's different from **availability**, which is whether you can reach the data right now. An outage hurts availability; losing the bytes hurts durability, and is usually far worse.

At scale, hardware failure is not an event, it's a rate. Hard drives in large fleets fail at an **annual failure rate** (AFR) of roughly 1 to 2%. With 100,000 drives that's 1,000 to 2,000 dead drives a year, three to six every day. Add whole servers, power supplies, racks, network switches, and occasionally an entire data centre, and a storage system has to treat losing a disk as routine.

The only defence is **redundancy**: store more than one copy's worth of information, spread so that no single failure removes too much of it, and **repair** quickly after each failure so redundancy is restored before the next one. There are two ways to add redundancy: **replication** and **erasure coding**. Large systems use both.

## Replication

**Replication** keeps N full copies of the data on different disks, ideally in different **failure domains** (different servers, racks, power feeds, Availability Zones). Three copies (3x) is the classic choice:

- **Tolerates 2 failures**: the data survives as long as any one copy remains.
- **Storage overhead 3x**: 1 TB of data needs 3 TB of disk, which is 200% extra.
- **Reads are cheap**: read any single copy, so the busiest copy can be avoided.
- **Writes are simple**: send the same bytes to three places.
- **Repair is cheap**: when a disk dies, copy each lost piece from one surviving replica. Lose 1 TB, read 1 TB.

## Erasure coding

**Erasure coding** splits data into **k data shards** and computes **m parity shards** from them, for k + m shards in total, stored on k + m different disks. The code is designed so that **any k of the k + m shards** are enough to rebuild the original. The usual family is **Reed-Solomon** codes.

### The simplest case: one XOR parity

Take two data shards, A and B, and compute parity P = A XOR B (bitwise exclusive-or). Store A, B and P on three disks. Lose A? Compute A = P XOR B. Lose B? B = P XOR A. Any 2 of the 3 rebuild the data, at 1.5x storage, tolerating 1 failure. This is RAID 5.

Reed-Solomon generalises it to any k and m. The maths is linear algebra over bytes (a finite field with 256 elements): arrange the data as a matrix, multiply by a coding matrix to produce the parity rows. To recover, take the rows of the coding matrix for the shards that survive, invert that square matrix, and multiply it by the surviving shards. The data comes back exactly.

~~~dg-compare The same 1 TB stored with 3x replication and with 10 + 4 erasure coding. Erasure coding stores less and tolerates more failures.
[3x replication]
1 TB -(copy x3)-> 3 disks, full copies
+ Survives any 2 failures
+ Read and repair from one copy
- 3 TB stored, 200% overhead
[*10 + 4 erasure coding]
1 TB -(split 10, add 4 parity)-> 14 disks
+ Survives any 4 failures
+ 1.4 TB stored, 40% overhead
- Repair reads 10 shards per lost shard
~~~

### Real-world parameters

- **Backblaze** splits each file into **17 data shards plus 3 parity shards**, 20 pieces on 20 different storage servers, and can rebuild from any 17. Overhead is 20/17, about **1.18x**, tolerating 3 failures. Backblaze open-sourced its Java Reed-Solomon library under the MIT licence; it handles up to 256 shards in total and encoded about 149 MB/s on a single CPU thread of a storage server.
- **Facebook's warm photo storage (f4)** used Reed-Solomon with 10 data and 4 parity shards, 1.4x within a region, with an XOR of two regions' data held in a third for disaster recovery, for an effective 2.1x across regions, down from about 3.6x in the hot photo store it replaced (three replicas, each on RAID-6).
- **Windows Azure Storage** uses Reed-Solomon in production and later a refinement called **Local Reconstruction Codes** (LRC), which adds small local parity groups so the most common repair (one lost fragment) reads only a handful of fragments rather than all k. Its configurations land at about **1.3x to 1.5x**.
- **S3** uses erasure coding across Availability Zones for its standard storage classes, which is how it reaches eleven nines without storing several full copies in each zone.

~~~dg-bars Raw storage needed per byte of data under common redundancy schemes.
3x replication | 3 | 3.0x, 2 failures
10 + 4 Reed-Solomon | 1.4 | 1.4x, 4 failures
*17 + 3 Reed-Solomon | 1.18 | 1.18x, 3 failures
12 + 2 + 2 LRC | 1.33 | 1.33x
~~~

## The hidden cost: repair

Erasure coding looks strictly better until a disk fails. With 3x replication, rebuilding one lost piece means **reading one copy** of it. With k + m erasure coding, rebuilding one lost shard means **reading k other shards** (from k different machines) and recomputing. For 10 + 4, a lost shard costs reading 10 shards' worth of data across the network.

So a dead disk generates roughly **k times more repair traffic** than it would with replication. In a large fleet with several disk deaths a day, that repair traffic is a significant, permanent load on disks and network. Repair speed also matters for durability: the longer a stripe runs with missing shards, the more exposed it is to further failures.

Erasure coding has other costs too:

- **Small reads touch many disks.** Reading a whole object means reading k shards. For small objects this multiplies I/O operations, so systems erasure-code large extents (often a gigabyte) rather than individual small objects.
- **Writes need encoding** and a write to k + m disks. Updating part of a stripe means recomputing parity, which is why erasure coding pairs naturally with **immutable, sealed** data.
- **CPU** for encoding and decoding, though modern libraries use SIMD instructions and run at gigabytes per second.

~~~dg-matrix Replication versus erasure coding on the properties that decide which to use where.
Property | 3x replication | 10 + 4 erasure coding
Storage overhead | no, 3.0x | yes, 1.4x
Failures tolerated | partial, 2 | yes, 4
Healthy read | yes, any one copy | partial, read 10 shards
Repair one lost unit | yes, read 1 unit | no, read 10 units
Small or changing data | yes | no
Large sealed data | partial, costly | yes
~~~

## Replicate first, erasure-code later

Because each scheme suits a different stage of data's life, large systems use **both**. Windows Azure Storage is a clear example:

- **Intra-stamp replication** happens in the stream layer, synchronously, on the write path. Every append goes to three replicas in different fault domains before it is acknowledged (lesson 07.18). An acknowledged write therefore survives disk, node and rack failures, and the contract is at least three durable copies, even through a cluster-wide power failure.
- **Erasure coding** happens in the background on **sealed** extents. Each sealed blob extent is split at block boundaries into N fragments plus M Reed-Solomon parity fragments, spread across fault domains, and then the extra full replicas are dropped. Storage cost falls from 3x to about 1.3x to 1.5x.
- **Inter-stamp (geo) replication** happens in the partition layer, asynchronously, copying objects to a second stamp in another region for disaster recovery. It is committed on the secondary within about 30 seconds on average.

~~~dg-timeline The life of a blob extent's redundancy in a replicate-then-erasure-code design.
0 s | Appended: three full replicas across fault domains, acknowledged
About 30 s | Async geo-replication copies the objects to another region
Minutes to hours | Extent reaches 1 GB and is sealed, immutable
Later, background | *Erasure-coded into N + M fragments; extra replicas deleted, 3x becomes about 1.3x
On disk failure | Lost fragments rebuilt from surviving ones elsewhere
~~~

The reasoning: while data is being written, replication is simpler, faster and allows reads from any copy; recently written data is also the most likely to be read or deleted soon. Once it's sealed and cooling, the storage cost dominates, and erasure coding's repair cost is paid rarely.

Erasure coding even helps reads in one way. WAS sends reads with a deadline; for erasure-coded data on a busy disk, it can read all N + M fragments and use the first N to arrive, so one slow disk never sets the latency.

## Thinking about durability numbers

Durability figures like "eleven nines" come from models of failure and repair rates. A rough version shows what matters.

<div class="ex">
<div class="lab">Worked example</div>

**why repair speed matters as much as copies**

Suppose drives have a 2% AFR, so the chance a given drive fails on a given day is about 0.02 / 365 ≈ 0.000055. With 3x replication and a 1-day repair window, after one copy is lost, data is lost only if both other copies' drives fail before repair: about 0.000055 × 0.000055 ≈ 3 × 10^-9 per incident.

Halve the repair time to 12 hours and that drops about fourfold, because both remaining failures must fall in a shorter window. Lengthen it to a week (a slow rebuild of a 20 TB disk) and the risk rises about 50-fold. Erasure coding with 4 parity shards needs 5 failures in the window, which pushes the risk down by many more orders of magnitude, provided repairs keep up. Real models also include correlated failures (a rack losing power, a bad batch of drives), which is why shards must be spread across failure domains: independence is the assumption every one of these numbers depends on.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**1 PB of backups**

Stored with 3x replication, 1 PB of data needs 3 PB of raw disk. With 10 + 4 erasure coding it needs 1.4 PB, saving 1.6 PB, about 80 drives of 20 TB, and survives four simultaneous failures instead of two.

When a 20 TB disk dies, though, rebuilding its shards reads about 200 TB from other disks (10 shards read per shard rebuilt) instead of 20 TB. Spread over 500 other disks in the cluster, that's 400 GB each, which the cluster can absorb in a few hours of background bandwidth, but it's a real, recurring cost that has to be budgeted, and it's why codes like LRC exist.

</div>

## Common mistakes

- **Counting copies but ignoring placement.** Three replicas in one rack survive a disk failure but not a rack failure.
- **Erasure-coding small, hot or mutable data.** Every read touches k disks and every update recomputes parity.
- **Ignoring repair bandwidth.** A design that tolerates 4 failures but takes a week to repair one can be less durable than expected.
- **Confusing durability with backups.** Redundancy protects against hardware failure, not against a bug or a user deleting data; that needs versioning and backups.

## In an interview

Quote the numbers: "3x replication costs 200% extra and tolerates 2 failures with cheap repair; 10 + 4 Reed-Solomon costs 40% extra and tolerates 4, but repairing one shard reads 10. So I'd replicate on the write path and erasure-code sealed, colder data in the background, with shards spread across racks or zones." Mention that repair speed and independent failure domains drive real durability.

<div class="take">
<div class="lab">Key takeaway</div>

replication is simple, cheap to read and cheap to repair but costs 3x; erasure coding gives equal or better durability at about 1.2x to 1.5x but makes repairs about k times more expensive and small reads touch many disks. Large systems replicate data while it is written and erasure-code it once sealed, spreading every copy or shard across failure domains.

</div>
