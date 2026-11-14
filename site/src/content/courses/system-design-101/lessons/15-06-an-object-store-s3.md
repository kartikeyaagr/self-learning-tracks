---
id: "15.06"
course: "system-design-101"
module: "15"
slug: "an-object-store-s3"
title: "An object store (S3)"
keyIdeas: ["Object store design: API layer (PUT/GET/DELETE/LIST, auth), metadata service (bucket/key → object ID, size, checksum, location), data service (stores bytes on many nodes).", "Write path: stream data to data nodes, replicate or erasure-code across failure domains, verify checksum, then commit metadata. Ack only after both.", "Read path: metadata lookup, then fetch bytes (parallel from chunks).", "Small objects can be packed into larger files to reduce per-file overhead.", "Bottleneck is usually the metadata layer: shard it (range partitioning for LIST support) and cache it.", "Background processes: garbage collection of deleted data, repair after disk loss, scrubbing for corruption."]
question: "Draw an object store: API, metadata, data nodes, replication. Where is the bottleneck?"
answer: "API: PUT, GET (with ranges), DELETE and LIST by prefix on bucket/key over HTTP, plus multipart upload for large objects. A stateless front-end fleet authenticates and routes. A metadata service, partitioned by sorted key range, maps each key to the locations of its pieces and commits a key only after the data is durable, which gives read-after-write consistency. Storage nodes hold the data, either replicated (often 3 copies, 3× storage) or erasure coded (k data + m parity shards such as 10 + 4, any k rebuild it, 1.4× storage), spread across drives, racks and zones, with background repair and scrubbing. The main bottleneck is disk random I/O, about 120 IOPS per drive however large it gets, so hot data overloads disks unless every customer's data is spread thinly across very many drives; the second is hot key ranges in the metadata index, handled by splitting and moving ranges automatically."
sources: [{"title": "AWS: Building and operating S3", "url": "https://www.allthingsdistributed.com/2023/07/building-and-operating-a-pretty-big-storage-system.html", "note": "Real S3 internals."}, {"title": "Windows Azure Storage paper", "url": "https://www.microsoft.com/en-us/research/publication/windows-azure-storage-a-highly-available-cloud-storage-service-with-strong-consistency/", "note": "Full object store design."}]
tags: ["Design"]
tools: ["S3"]
---

## What an object store is

An object store like Amazon S3 stores **objects** (a blob of bytes plus metadata such as content type and custom headers) under a **key** inside a **bucket**, accessed over an HTTP API. There is no directory tree and no in-place editing: you write a whole object and read it back, or replace it. Keys like `photos/2024/cat.jpg` look like paths, but the slash is just a character; "folders" are an illusion created by listing keys with a common prefix.

That simple model is exactly what lets it scale. Because objects are immutable once written, there is no need for byte-range locking, partial-update consistency or a file system's metadata operations (rename a directory with a million files). The design must provide three things at once:

- **Enormous scale**: trillions of objects, exabytes of data, millions of requests per second.
- **Very high durability**: S3 is designed for 99.999999999% (eleven nines) annual durability, meaning that if you store 10 million objects you would expect to lose one about every 10,000 years.
- **Throughput for any one customer's burst**: a single job may suddenly read petabytes in parallel.

### The API

- `PUT /bucket/key` with the body: create or replace an object.
- `GET /bucket/key` (optionally with a `Range` header for part of it).
- `DELETE /bucket/key`.
- `LIST /bucket?prefix=photos/2024/&continuation-token=…`: keys in sorted order, 1,000 per page.
- **Multipart upload** for large objects: initiate, upload parts (5 MB to 5 GB each, in parallel, retryable individually), then complete, which stitches the parts into one object. Objects can reach 5 TB.

## The four parts

1. **Front end**: a stateless fleet that speaks the HTTP API, authenticates and authorises requests, and routes them.
2. **Namespace (metadata index) service**: maps `bucket + key` to where the object's pieces are stored, answers `LIST`, and makes new objects visible. It is a huge, partitioned, sorted key-value store (lesson 07.17).
3. **Storage fleet**: millions of hard drives holding the data. In S3 each disk is managed by **ShardStore**, a per-disk storage layer written in Rust.
4. **Background services**: replication, repair, moving data between storage classes, scrubbing (re-reading data to detect silent corruption), and garbage collection of deleted objects.

### The write path

~~~dg-seq A PUT: data is made durable on storage nodes first, and only then is the key committed in the index.
Client > Front end : PUT /photos/cat.jpg (4 MB)
Front end > Front end : authenticate, authorise, checksum body
Front end > Placement : choose 14 drives in 3 zones
Front end > Storage nodes : write 10 data + 4 parity shards
Storage nodes -> Front end : 14 acks, each checksum verified
Front end > Metadata index : commit key -> shard locations, version
Metadata index -> Front end : committed
Front end -> Client : 200 OK, ETag
~~~

The front end splits the object into pieces, writes them to storage nodes on different drives, racks and zones, and only after enough pieces are durable does it **commit the key** in the index. Committing last means a reader never sees a key whose data is not there yet. If the front end crashes halfway, the written shards are orphans with no key pointing at them, and garbage collection removes them later. Checksums travel with the data end to end (lesson 07.20), so corruption anywhere along the path is detected rather than stored.

Since late 2020, S3 gives **strong read-after-write consistency**: once a PUT returns, every subsequent GET and LIST sees it. That is a property of the metadata index: the commit is the single point where the new version becomes visible.

### The read path

A GET looks up the key in the index, gets the shard locations, reads enough shards (from the least busy drives), reassembles and streams the object. For a 4 MB object erasure coded 10 + 4, that is 10 reads of 400 KB, in parallel.

## Durability: replication and erasure coding

- **Replication** keeps full copies, often 3. Simple, and a read can go to any copy, which helps steer load away from busy disks. But it costs 3× the storage.
- **Erasure coding** (Reed-Solomon style) splits an object into **k data shards** plus **m parity shards**; **any k of the k + m** are enough to rebuild it. For example, 10 + 4 survives losing any 4 shards for 1.4× the storage instead of 3×. Reads touch k drives, and rebuilding a lost shard needs k reads, so it suits larger and colder data.

~~~dg-compare Three-way replication versus 10 + 4 erasure coding for the same object.
[3x replication]
object -(copy)-> drive A, drive B, drive C
+ read from any one drive
+ cheap repair: copy one replica
- 3x storage
- survives only 2 lost copies
[*Erasure coding 10 + 4]
object -(split + parity)-> 14 shards on 14 drives
+ 1.4x storage
+ survives any 4 lost shards
- a read touches 10 drives
- repair reads 10 shards to rebuild 1
~~~

Shards go on different drives, racks and Availability Zones, so one failure takes out at most one shard (or a few, in the case of a zone). A background repair process constantly finds lost shards (dead disks, failed scrubs) and rebuilds them elsewhere. Durability is a race: **repair must be faster than failures accumulate**.

<div class="ex">
<div class="lab">Worked example</div>

**why repair speed sets durability**

A fleet of 1 million drives with a 1% annual failure rate loses about 10,000 drives a year, roughly 27 a day. Each 20 TB drive holds shards of millions of objects.

With 10 + 4 erasure coding, an object is lost only if 5 of its 14 drives fail before repair completes. If repair takes 1 hour, the chance that one particular drive fails within the hour is about 0.01 ÷ 8,760 ≈ 1.1 × 10^-6. Losing 4 more of the other 13 drives in the same hour is astronomically unlikely.

If repair takes a week because the repair system is short of bandwidth, the per-drive probability rises to about 2 × 10^-4, and correlated failures (a batch of bad drives, a rack power fault) start to matter. Spreading each object's shards across many drives also means a dead drive's data is rebuilt by thousands of drives in parallel, so a 20 TB drive is rebuilt in minutes, not the 30-plus hours a single drive would need to copy itself.

</div>

## The real bottleneck: disk I/O, not capacity

The key point is physics. Hard drive capacity has grown about 7.2 million times since IBM's first disk in 1956, but a drive still manages only about **120 random I/Os per second**: the arm must move and the platter must rotate, and that mechanical time barely changes. Drives keep getting bigger (26 TB shipping, with 200 TB on roadmaps), so **I/O per terabyte keeps falling**.

~~~dg-bars Random I/O available per terabyte as drives get bigger, at a fixed 120 IOPS per drive.
1 TB drive | 120 | 120 IOPS per TB
4 TB drive | 30 | 30 IOPS per TB
26 TB drive | 4.6 | 4.6 IOPS per TB
*200 TB drive | 0.6 | 0.6 IOPS per TB
~~~

So the problem is **heat**: the request rate hitting each disk. If a popular dataset sits on a few disks, those disks queue up, and their tail latency spreads to every request that needs them. With erasure coding a read needs 10 drives, so it is as slow as the slowest of the 10.

S3's answer is to make scale an advantage:

- **Spread every customer across many disks.** Objects are placed on different sets of drives, so any one customer uses only a tiny slice of any disk. Tens of thousands of customers' buckets span millions of drives, and each drive holds data from thousands of customers.
- **Aggregate demand is smooth.** Any single workload is bursty and mostly idle, but millions of them added together are steady and predictable. No one workload moves the total much.
- **Bursts get huge parallelism.** A genomics job using thousands of Lambda functions can read from over a million disks at once, far more than it could ever buy for itself.
- **Read the least busy shards.** With 14 shards and only 10 needed, a reader can pick the 10 fastest, or request 11 and use the first 10 to arrive, cutting tail latency (lesson 02.12).

<div class="ex">
<div class="lab">Worked example</div>

**can one customer melt a disk?**

A customer stores 100 TB and reads all of it once a day: about 1.2 GB/s average, or 300 random 4 MB reads per second.

If that 100 TB sat on five 20 TB drives, each would need 60 large reads per second plus seeks, near saturation, and every other customer on those drives would suffer.

Spread across 1 million drives, each drive holds about 100 MB of this customer's data. The 300 reads per second become about 3,000 shard reads per second spread over a million drives, so each drive sees one of them every five minutes or so. The customer is invisible in any one drive's load, and even a sudden 100× burst is only one request per drive every few seconds.

</div>

## The metadata index: the other bottleneck

Every request touches the index, and its load is not spread by hashing: keys are kept **sorted** so that LIST by prefix works, which means a burst of writes to keys with the same prefix (`logs/2024-11-12/…`) all land in one key range. The index is therefore partitioned by key range, and it **splits hot ranges automatically** when their request rate grows, moving the halves to different servers. S3 documents a baseline of at least 3,500 writes and 5,500 reads per second per prefix, with automatic scaling beyond that. Clients that need far more spread their keys across many prefixes (for example, a short hash at the start of the key).

~~~dg-layers Where an object store's limits sit, and how each is handled.
Front end | stateless; scale horizontally behind load balancers
Metadata index | sorted key ranges; split and move hot ranges automatically
*Storage nodes | about 120 random IOPS per disk; spread data so no disk runs hot
Repair and background | rebuild lost shards faster than disks fail
~~~

## The Windows Azure Storage design

Windows Azure Storage is a comparable system, and its layering makes the same split very explicit.

~~~dg-tree Windows Azure Storage: accounts map to stamps, and each stamp has three layers.
Location service | account to stamp
  Storage stamp | 10 to 20 racks
    Front-end layer | stateless
    *Partition layer | key ranges
    Stream layer | 3 replicas
  Second stamp | async geo copy
~~~

- **Storage stamps** are clusters of roughly 10 to 20 racks of storage nodes, each a self-contained unit. A **location service** maps accounts to stamps and updates DNS, so it can move an account between stamps.
- The **partition layer** understands blobs, tables and queues; it splits the namespace into key **ranges** served by partition servers, and moves or splits ranges to balance load.
- The **stream layer** stores data as append-only **extents** replicated 3 ways synchronously within the stamp. Full extents are sealed and can then be erasure coded to save space.
- **Replication within a stamp is synchronous** (strong consistency for writes), and **between stamps it is asynchronous** (for geo-disaster recovery).

## Reliability engineering

Durability is protected as a process, not just a design. At S3 every change gets a **durability review**, like a security threat model, listing what could go wrong and how it is handled. ShardStore is checked against an executable model about 1% of its size, with property-based tests on every commit. Background **scrubbing** reads every shard periodically and verifies its checksum, because a disk can return wrong data without reporting an error.

## Common mistakes

- **Putting metadata and data in one system**: the bytes and the index have completely different access patterns and scaling limits.
- **Committing the key before the data is durable**, exposing objects that cannot be read.
- **Sizing by capacity only**, then finding that the drives cannot serve the I/O.
- **Hashing keys in the index**, which breaks prefix listing; or sorting without splitting hot ranges.
- **Replication only**, paying 3× storage for cold data that erasure coding would protect better at 1.4×.

## In an interview

Draw the four parts, walk the PUT path (shards first, commit last) and the GET path, explain replication versus erasure coding with the storage overhead numbers, then name the bottlenecks: random I/O per disk (spread data thinly, read the fastest k shards) and hot key ranges in the index (range partitioning with automatic splits). Mention repair speed as the core of durability.

<div class="take">
<div class="lab">Key takeaway</div>

an object store has a stateless HTTP front end, a range-partitioned metadata index mapping keys to shard locations, and a storage fleet that replicates or erasure codes data across drives, racks and zones, committing the key only after the data is durable. The bottlenecks are disk random I/O (about 120 IOPS per drive while capacity keeps growing) and hot index ranges, so spread every customer's data thinly across huge numbers of disks, split hot key ranges, and repair faster than drives fail.

</div>
