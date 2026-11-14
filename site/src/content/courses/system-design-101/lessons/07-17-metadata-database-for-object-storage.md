---
id: "07.17"
course: "system-design-101"
module: "07"
slug: "metadata-database-for-object-storage"
title: "Metadata database for object storage"
keyIdeas: ["Object stores keep metadata (name, size, checksum, owner, timestamps, location of the bytes) separate from the object bytes.", "Metadata is small and needs fast lookups, listings and updates; bytes are large and need cheap, durable bulk storage. Different needs, different systems.", "Metadata often lives in a sharded, replicated database or range-partitioned table; bytes live on storage nodes or in a log/stream layer.", "Separation lets you list a bucket without touching data disks, and move or repair data without changing names.", "The metadata layer is usually the scaling and consistency bottleneck, so it's sharded carefully."]
question: "Why keep object metadata separate from object bytes?"
answer: "Metadata is small, changes often (versions, tags, ACLs) and must be looked up by key and listed in sorted order, while bytes are huge, immutable and need cheap capacity, so each wants different hardware (SSD or RAM versus dense disks) and different structures (a sorted, partitioned index versus erasure-coded blobs). Separating them lets each scale on its own terms (metadata by object count, bytes by volume), lets LIST, HEAD and existence checks run without touching data, makes the metadata commit the single atomic moment an object becomes visible (bytes can be written first, safely invisible), lets repair and tiering move bytes by just updating a pointer, and makes garbage collection a matter of finding bytes no metadata row references. Haystack showed the cost of mixing them: three or more disk operations per photo on NFS versus one once metadata moved into memory."
sources: [{"title": "Facebook Haystack paper", "url": "https://www.usenix.org/legacy/event/osdi10/tech/full_papers/Beaver.pdf", "note": "Directory service for object metadata."}, {"title": "AWS: Building and operating S3", "url": "https://www.allthingsdistributed.com/2023/07/building-and-operating-a-pretty-big-storage-system.html", "note": "Index vs storage separation."}]
---

## Bytes and metadata are different workloads

Every object in an object store has two parts:

- The **bytes**: the photo, the video, the backup. Large (kilobytes to terabytes), immutable once written, read sequentially or by byte range, and the overwhelming majority of total storage.
- The **metadata**: the bucket and key, version ID, size, content type, checksum, owner, access control, creation time, storage class, user-defined tags, and, crucially, **where the bytes live** (which disks or extents hold its shards). Tiny (a few hundred bytes to a few kilobytes), changed often (tags, ACLs, versions, lifecycle state), and accessed by key lookup and sorted listing.

These two workloads want opposite things from hardware and from data structures. Putting them in one structure makes each worse. Every large object store therefore keeps object metadata in a separate, indexed, partitioned **metadata store**, and the bytes in a capacity-optimised **data store**. This lesson explains why, and what the metadata store looks like.

~~~dg-matrix The two halves of an object store have almost nothing in common as workloads.
Property | Object bytes | Object metadata
Typical size | KB to TB | 200 bytes to a few KB
Share of total storage | about 99.99% | tiny
Mutable after write | no | yes, tags, ACLs, versions
Access pattern | sequential, ranges | point lookup, sorted scan
Needs sorted index | no | yes, for LIST
Best hardware | dense hard disks | SSD or RAM
~~~

## Evidence from real systems

### Haystack: metadata was the bottleneck

Facebook's photo store (lesson 07.13) is the clearest demonstration. On NFS filers, every photo was a file, and the filesystem mixed metadata (directory entries, inodes) with data on the same disks. Reading one photo took **3 or more disk operations** just to resolve its location before reading any bytes, and the metadata for billions of files couldn't be cached in RAM.

Haystack separated them. The metadata needed to locate a photo was cut to about **10 bytes** and kept in memory on each storage machine, while the bytes sat in 100 GB volume files. One disk operation per read. Separately, the **Directory** held the mapping from logical volumes to machines, a metadata store that never touched photo bytes.

### S3 and Windows Azure Storage: separate services

S3 runs a **namespace service** (bucket and key to version and data location) as a separate system from its **storage fleet**, each built and operated by different teams (lesson 07.14). Windows Azure Storage does the same: its **partition layer** holds objects and their index, partitioned by key range, on top of a **stream layer** that only knows about bytes in extents (lessons 07.16 and 07.18).

~~~dg-flow A GET goes through metadata first to find the bytes; a LIST is answered entirely from metadata and never touches the bytes.
[GET] Client -(bucket + key)-> *Metadata store -(shard locations)-> Data store -(range reads)-> Client
[LIST] Client -(prefix, page token)-> *Metadata store -(sorted keys, sizes)-> Client
~~~

## What the metadata store holds

A simplified schema for an object store with versioning:

| Table | Key | Main fields |
| --- | --- | --- |
| buckets | bucket name | owner, region, versioning on/off, policies, created |
| objects | bucket, object key, version (newest first) | size, checksum, content type, ACL, tags, storage class, is-delete-marker, data locations |
| uploads | bucket, key, upload ID | parts uploaded so far, for multipart uploads in progress |

The primary key of the objects table is the important design decision: `(bucket, key, version descending)`. Sorting by bucket then key means all of a bucket's objects are adjacent and in key order, so listing a prefix is a range scan. Sorting versions newest first means "get the current version" is "read the first row for this key".

### Listing and fake folders

Object stores have no real directories. A "folder" view comes from LIST with a **delimiter**. `LIST prefix=photos/2024/ delimiter=/` asks for keys starting with `photos/2024/`, but groups any key that has another `/` after the prefix into a **common prefix** (`photos/2024/06/`) instead of listing each one. The metadata store implements this by scanning in key order and, on finding a common prefix, **seeking past** all keys that share it. That's only efficient because keys are sorted; it's one reason metadata is range-partitioned (lesson 07.15).

~~~dg-tree How a flat, sorted key space looks as folders when listed with delimiter "/". The folders exist only as common prefixes of keys.
bucket: acme-media
  photos/2024/05/ | common prefix, 12,000 keys under it
  photos/2024/06/ | common prefix, 9,500 keys
  *photos/2024/cover.jpg | an actual object
  photos/2024/index.json | an actual object
~~~

### Versions and delete markers

With versioning on, an overwrite adds a new version row instead of replacing the old one. A DELETE without a version ID adds a **delete marker**: a version row that says "the current state is deleted". GET without a version returns 404 because the newest version is a delete marker, but older versions remain and can be restored. Lifecycle rules later expire non-current versions. All of this is metadata-only: no bytes move until garbage collection deletes data no version references.

## Why the separation pays off

- **Each half scales on its own terms.** Metadata needs fast random lookups and range scans, so it lives in a partitioned, indexed store on SSD or in memory, sized by object **count**. Bytes need cheap capacity, so they live on dense hard disks with erasure coding, sized by **volume**. A workload of billions of tiny objects stresses the first; a workload of a few huge video files stresses the second.
- **Listing and querying are cheap.** LIST, HEAD (metadata only), checking existence, and filtering by tags or dates never read the data store at all.
- **Atomic visibility.** An object becomes visible at the instant its metadata row commits. The bytes can be written first, at leisure, across many disks, because nobody can find them until the commit. Crashes before the commit leave unreferenced bytes, never half-visible objects.
- **Moving data is invisible to users.** Repair after a disk failure, moving to a colder storage class, re-erasure-coding or compaction can relocate bytes and then update the location pointer in the metadata. The user's key never changes.
- **Garbage collection becomes a join.** Bytes not referenced by any metadata row are garbage, found by comparing the data store's inventory against the metadata.

<div class="ex">
<div class="lab">Worked example</div>

**overwrite with versioning**

A user uploads a new `logo.png` to a versioned bucket. The data store writes the new bytes to fresh disks: they're invisible. Then the metadata store, in one transaction on the row range for `(acme, logo.png)`, adds version 7 pointing at them, which becomes the newest. A GET that started a moment earlier already read version 6's locations and streams version 6 to the end, because its bytes are untouched. A GET that starts after the commit gets version 7. No reader ever sees a mix.

</div>

~~~dg-seq Overwriting an object in a versioned bucket. The bytes are written first and stay invisible until the metadata commit makes version 7 current.
Client > Front end : PUT logo.png
Front end > Data store : write new shards
Data store -> Front end : durable at L7
Front end > Metadata store : add v7 -> L7
Metadata store -> Front end : committed
Front end -> Client : 200 OK, v7
Reader > Metadata store : GET logo.png
Metadata store -> Reader : v7 -> L7
~~~

## Sizing the metadata store

The metadata is small per object but there are a lot of objects, so it needs its own capacity plan.

<div class="ex">
<div class="lab">Worked example</div>

**how big is the metadata?**

A region stores 100 billion objects. Each metadata row (key averaging 80 bytes, version ID, size, checksum, timestamps, ACL reference, tags, and a list of shard locations for 14 erasure-coded shards) is about 600 bytes. Total: 100 billion × 600 bytes = **60 TB of metadata**, before replication.

That's far too much for one machine but very manageable as a partitioned store: at 1 TB of SSD per partition server, 60 to 200 servers with replication, each serving a range of buckets and keys. Meanwhile the bytes, at an average object size of 1 MB, total 100 PB. The metadata is 0.06% of the data by size but drives the request rate: every GET, PUT, HEAD and LIST hits it, often hundreds of thousands to millions of times a second across the region.

</div>

~~~dg-bars Storage for 100 billion objects of 1 MB average: the metadata is tiny by volume but is touched by every request.
Object bytes | 100000 | about 100 PB
*Object metadata | 60 | about 60 TB
~~~

What metadata stores are built on varies: range-partitioned key-value stores with consensus (like WAS's partition layer), sharded relational databases keyed by bucket, or transactional key-value stores such as FoundationDB, which several large systems use for metadata. Ceph's object gateway keeps a per-bucket index in its object store, sharded because a single huge bucket index becomes a hot spot. Some simpler self-hosted object stores, such as MinIO, skip a central metadata database and store each object's metadata in a small file beside its data; that's simpler to operate but makes large prefix listings slower, since they have to walk many per-object files instead of scanning a sorted index.

## Pitfalls

- **Hot buckets.** Billions of objects in one bucket under one prefix concentrate metadata load on a few ranges; the store must split ranges by load (lesson 07.16).
- **Sequential keys.** Timestamp-first keys send every new PUT to the last range of the bucket.
- **Losing metadata loses data.** Bytes with no metadata row are unreachable garbage. The metadata store needs at least the durability of the data store: synchronous replication, consensus, backups.
- **Unbounded listing.** LIST must be paginated with continuation tokens (the last key returned), so one request never scans millions of rows.
- **Forgetting GC.** Without a sweep, failed uploads and expired versions leave bytes on disk forever.

## In an interview

When you design file or media storage, draw two boxes: "a metadata service, range-partitioned by bucket and key, holding size, checksum, version and shard locations, which serves GET lookups, HEAD and LIST; and a data service that stores immutable erasure-coded shards on dense disks." Say the metadata commit is the visibility point, size the metadata by object count, and mention garbage collection of unreferenced bytes. That shows you understand why the split exists, not just that it does.

<div class="take">
<div class="lab">Key takeaway</div>

keep small, mutable, queryable metadata in an indexed, range-partitioned store sized by object count, and large immutable bytes in a capacity store sized by volume. Committing the metadata row is what makes an object exist, which lets bytes be written, moved and repaired behind a stable key.

</div>
