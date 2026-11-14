---
id: "07.14"
course: "system-design-101"
module: "07"
slug: "s3-object-storage-architecture"
title: "S3 object storage architecture"
keyIdeas: ["An S3-style object store has a front-end/API layer (auth, routing), a metadata/index layer (bucket + key → where the bytes live), and a storage layer of many disks holding the bytes.", "A PUT: the front end authenticates, data is split and written to storage nodes (replicated or erasure-coded across failure domains), then the metadata entry is committed. Only then is the PUT acknowledged.", "A GET looks up metadata, then reads the bytes (often in parallel from several nodes).", "S3 has been strongly consistent for reads after writes since 2020, using a replicated cache-coherence layer for metadata.", "At S3's scale, hard drive failures are constant, so repair is a continuous background process."]
question: "Name the main components behind an S3 PUT: front end, metadata, storage nodes. What does each do?"
answer: "The front end is a stateless fleet that terminates HTTP and TLS, authenticates and authorises the request, and routes it, choosing where the new data will go. The storage nodes (a fleet of hard disks, each managed by the ShardStore engine) durably store the object's erasure-coded shards, spread across disks, racks and Availability Zones as separate failure domains, and verify checksums. The metadata (namespace) service records the bucket and key, version, size, checksum and the locations of the shards, and also serves LIST; the metadata commit happens only after the shards are durable, and once it commits the object becomes visible to reads and the client gets 200 OK."
sources: [{"title": "AWS: Building and operating S3", "url": "https://www.allthingsdistributed.com/2023/07/building-and-operating-a-pretty-big-storage-system.html", "note": "Andy Warfield's overview."}, {"title": "AWS: S3 strong consistency announcement", "url": "https://www.allthingsdistributed.com/2021/04/s3-strong-consistency.html", "note": "How S3 became strongly consistent."}]
tools: ["S3"]
---

## What has to happen behind a PUT

From the outside, Amazon S3 looks almost trivial: `PUT bucket/key` with some bytes, `GET bucket/key` to read them back, `LIST` to see what's there. Behind that small API is one of the largest storage systems ever built: hundreds of microservices, a fleet of millions of hard disks, and a design target of **eleven nines of durability** (99.999999999%, meaning that if you store 10 million objects you'd expect to lose one about every 10,000 years).

You don't need the internals of any one provider to design an object store in an interview, but you do need the shape, because every large object store (S3, Google Cloud Storage, Azure Blob Storage, Ceph, MinIO) has the same three concerns separated into different subsystems:

1. **Speaking the protocol**: HTTP, authentication, routing.
2. **Knowing where things are**: the mapping from bucket and key to the location of the bytes, plus versions and listing.
3. **Storing bytes durably**: disks, redundancy, repair.

## The parts of S3

At a high level, S3 is organised like this:

~~~dg-layers The main layers behind S3. Each is owned by separate teams and runs its own fleet behind an API contract.
Front-end fleet | REST API, authentication, request routing
*Namespace (index) service | bucket + key to version and data location
Storage fleet | millions of hard disks holding the bytes
ShardStore on each disk | per-disk storage engine, written in Rust
Background services | replication, repair, tiering, lifecycle, audits
~~~

- The **front-end fleet** terminates HTTP and TLS, authenticates the request (checks the signature against IAM credentials and bucket policies), applies rate limits, and routes the request to the right internal services. Front-end servers are **stateless**, so there can be many of them behind load balancers.
- The **namespace service** (often called the index or metadata service) maps `(bucket, key)` to the object's current version, size, checksum and the location of its data. It also serves `LIST`, which is a range scan over keys in sorted order (lesson 07.15 explains why that needs range partitioning).
- The **storage fleet** is the huge population of hard disks that hold the bytes. On each disk, a storage engine called **ShardStore**, rewritten in Rust, manages how data is laid out. It was developed alongside lightweight formal verification (executable reference models and automated checks of crash consistency), because a bug at this layer could corrupt data on millions of disks.
- **Background services** do the work no request waits for: repairing lost redundancy after disk failures, moving objects between storage classes, enforcing lifecycle rules, scrubbing data to detect silent corruption, and replicating across regions.

Each of these is owned by its own teams and runs as a fleet behind an API contract, which is how an organisation of that size changes one part without breaking the others.

## A PUT, step by step

Here is the path of a single-part PUT, as large object stores are generally built:

1. **Front end**: the request arrives, TLS is terminated, the signature is checked, and the bucket's policy allows the write. If the client supplied a checksum header, the front end prepares to verify it (lesson 07.20).
2. **Placement**: the system chooses where the bytes will go. For durability, the object is **erasure-coded**: split into k data shards plus m parity shards (lesson 07.19), so any k of the k + m shards can rebuild it. The shards are placed on disks in different **failure domains**: different machines, racks and, for standard storage classes, different Availability Zones (separate data centres in one region).
3. **Write the bytes**: the shards are streamed to their storage nodes, which write them to disk. The checksum of the received data is computed and compared with the client's.
4. **Wait for durability**: once enough shards are durably stored to meet the durability target, the bytes are safe.
5. **Commit the metadata**: only now is a record written in the namespace service: key, version ID, size, checksum, and the locations of the shards.
6. **Acknowledge**: the front end returns `200 OK`. From this moment a GET or LIST can find the object.

~~~dg-seq A PUT writes the bytes first and makes them visible only when the metadata commits. Before that commit, nobody can find the new data.
Client > Front end : PUT key + checksum
Front end > Front end : auth, authorise
Front end > Storage nodes : write k + m shards
Storage nodes -> Front end : durable, CRC ok
Front end > Namespace : commit shard map
Namespace -> Front end : committed
Front end -> Client : 200 OK
~~~

The ordering is the important idea, and it's the same rule as lesson 07.13: **bytes first, metadata last**. Until the metadata commit, the new bytes are invisible, so a crash halfway leaves unreferenced shards (cleaned up later), never a key pointing to missing data. And because an overwrite writes new shards and then swaps the metadata, a reader sees either the complete old object or the complete new one.

### Multipart uploads

Large objects are uploaded in **parts**: the client starts a multipart upload, uploads parts (5 MB to 5 GB each, up to 10,000 parts) in parallel and in any order, retrying only failed ones, then sends a "complete" request listing the parts. Each part is stored like a small object; "complete" is a single metadata commit that stitches them into one object. This is how clients reach gigabytes per second for one upload, and why a 1 TB upload can survive a flaky network.

### A GET

A GET is the reverse: front end authenticates, the namespace returns the current version's shard locations, the front end reads k shards (it can read from whichever disks respond fastest, and with spare shards it doesn't have to wait for a slow one), reassembles the bytes, and streams them back with the stored checksum available for the client to verify.

## Heat: the real scaling problem

The central operational idea in a fleet of hard disks is **heat**: the request load on a given disk at a given time. Lesson 07.01 explained the constraint: a hard drive does only about 120 random I/Os per second, and that number hasn't improved for decades, while capacity keeps rising (drives of 26 TB at the time this design was discussed, with 200 TB forecast). So the I/O available **per terabyte** keeps falling.

~~~dg-bars Random I/Os per second available per terabyte as drives grow, assuming about 120 IOPS per drive regardless of size.
4 TB drive | 30 | 30 IOPS per TB
12 TB drive | 10 | 10 IOPS per TB
*26 TB drive | 4.6 | about 4.6 IOPS per TB
200 TB drive | 0.6 | about 0.6 IOPS per TB
~~~

If one customer's data sat on a few disks, a burst of reads from that customer would overwhelm those disks while thousands of others sat idle. S3 attacks this in three ways:

- **Spread every customer's objects over many disks.** Placement deliberately scatters shards so that one customer's working set lives on a very large number of drives. A single customer's burst can then be served by over a million disks in parallel.
- **Aggregate many workloads.** Individual workloads are extremely bursty, but millions of uncorrelated bursty workloads, added together, produce a smooth total. The fleet is sized for the aggregate, not for each customer's peak.
- **Use redundancy for reads.** With erasure coding, there are more shards than needed to read an object. If one disk is hot, the read can use the others and skip it.

<div class="ex">
<div class="lab">Worked example</div>

**why spreading matters**

A customer stores 50 TB of images and occasionally triggers a batch job that reads 2 million of them in 10 minutes: about 3,300 reads a second. If those 50 TB sat on two 26 TB drives, the drives could deliver about 240 random reads a second: the job would take over two hours and other customers on those drives would starve.

Spread across shards on, say, 100,000 drives, the same burst is about 0.03 extra reads a second per drive, invisible in the noise. And because each object can be rebuilt from any k of its k + m shards, the few reads that land on a temporarily busy disk can simply use a different shard.

</div>

## Strong consistency and the witness

For its first 14 years, S3 was **eventually consistent** for some operations: after an overwrite or delete, a GET might briefly return the old data, and a LIST might not yet show a new key. In December 2020 S3 became **strongly consistent** for all operations, including LIST: once a PUT returns 200, every subsequent read sees it.

The old behaviour came from the **metadata cache**. To keep reads fast and available, metadata lookups are served from a caching layer in front of the persistent metadata store. A write could update the store through one path while a read was answered by a cache node still holding the previous entry.

The fix added a component called a **witness**. Every change to an object's metadata notifies the witness, which keeps a small amount of in-memory state about recent writes. Before a cache serves a metadata entry, it checks with the witness: "is my entry current?" If the witness says it's stale, the cache reads through to the persistent tier. The witness acts as a **read barrier**: a cheap, fast check that makes the cache safe without making every read go to the persistent store. Because its state is small and in memory, it is fast and easy to replace if a node fails.

~~~dg-seq The witness acts as a read barrier: the cache serves a value only if the witness confirms it is current; otherwise it reads through to the persistent metadata store.
Writer > Store : PUT k (v2)
Store > Witness : k changed
Reader > Cache : GET k
Cache > Witness : is v1 current?
Witness -> Cache : no, stale
Cache > Store : read k
Store -> Cache : v2
Cache -> Reader : v2
~~~

The team verified the new protocol with formal proofs and model checking, which took more effort than writing the code, because a subtle ordering bug would break the guarantee silently. The change shipped at no extra cost to customers and with no announced performance or availability trade-off.

<div class="ex">
<div class="lab">Worked example</div>

**upload then list**

Before December 2020, a data pipeline could PUT `reports/2020-11.csv` and immediately LIST `reports/` without seeing it, because LIST read a stale cache entry. Pipelines worked around it by sleeping, retrying, or writing a separate manifest file listing what they'd uploaded (or by using a consistency layer such as a DynamoDB table). With the witness, the LIST is guaranteed to include the new key, and those workarounds became unnecessary.

</div>

## Pitfalls when using an object store like S3

- **Treating it like a filesystem.** There are no directories, only keys with slashes; renaming a "folder" means copying and deleting every object under it.
- **Hot prefixes.** Request rates scale per key range: S3 supports at least 3,500 writes and 5,500 reads per second per partitioned prefix, and splits busy ranges automatically, but a sudden burst on one narrow prefix can be throttled (HTTP 503 Slow Down) until it does. Lesson 07.16 explains the splitting.
- **Small objects.** Each request has fixed overhead and cost; billions of 1 KB objects are expensive. Pack them (lesson 07.05).
- **Ignoring checksums.** Supply one on upload and verify on download (lesson 07.20).

## In an interview

If asked to design an object store, separate the three concerns and name the commit point: "Stateless front ends for HTTP and auth; a partitioned metadata service mapping bucket and key to version and shard locations, which also serves LIST; a storage layer that erasure-codes objects across failure domains. A PUT writes the shards first and commits metadata last; the object exists from that commit." Then mention heat (spread data so bursts hit many disks) and repair (background services restore lost shards).

<div class="take">
<div class="lab">Key takeaway</div>

S3 separates the HTTP front end, the namespace metadata and a disk fleet running a per-disk engine, with background services for repair and tiering. A PUT writes erasure-coded shards across failure domains first and commits metadata last, which is the moment the object exists; spreading data over huge numbers of disks tames heat, and the witness keeps cached metadata from serving stale answers.

</div>
