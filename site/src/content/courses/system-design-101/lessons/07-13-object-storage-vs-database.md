---
id: "07.13"
course: "system-design-101"
module: "07"
slug: "object-storage-vs-database"
title: "Object storage vs database"
keyIdeas: ["Object storage (S3, GCS) is built for large, immutable blobs addressed by key: cheap, very durable, effectively unlimited, accessed over HTTP.", "Databases are built for small, structured records with indexes, queries and transactions.", "Store large binary files (images, video, backups, logs) in object storage and keep their metadata (owner, size, key, permissions) in the database.", "Putting large blobs in a database bloats backups, replication and buffer caches.", "Object storage weaknesses: higher per-request latency (tens of ms), no partial in-place updates, per-request pricing, and listing is slow."]
question: "Why store images in object storage and only their metadata in a database?"
answer: "Images are large, written once and read whole, which object storage serves cheaply (dense disks, erasure coding, several times cheaper per GB), very durably and at huge parallel scale, directly to clients or a CDN through signed URLs. Databases are expensive per byte and suffer when filled with blobs: the buffer pool is polluted, the WAL, replicas and backups balloon, and every download has to stream through app servers. But databases are excellent at indexing, querying and transactionally updating small records such as owner, key, size, checksum and timestamps, which object storage cannot query. Storing metadata in the database (written after the bytes, with orphaned objects garbage-collected) and the bytes in object storage lets each do what it is good at."
sources: [{"title": "AWS: Building and operating a pretty big storage system called S3", "url": "https://www.allthingsdistributed.com/2023/07/building-and-operating-a-pretty-big-storage-system.html", "note": "How S3 thinks about scale."}, {"title": "Facebook Haystack paper", "url": "https://www.usenix.org/legacy/event/osdi10/tech/full_papers/Beaver.pdf", "note": "Why photos didn't belong in a filesystem or DB."}]
---

## Two kinds of storage for two kinds of data

Most applications store two very different kinds of data side by side:

- **Structured records**: users, orders, comments, the caption and owner of a photo. Each is small (bytes to kilobytes), changes often, and is looked up, filtered, joined and updated in many ways.
- **Blobs** (binary large objects): the photo itself, a video, a PDF, a backup, a log archive, a Parquet file. Each is large (kilobytes to terabytes), written once, never edited in part, and read whole or by byte range.

A **database** is built for the first kind: indexes, queries, transactions, in-place updates. **Object storage** (S3, Google Cloud Storage, Azure Blob Storage, or self-hosted systems like Ceph and MinIO) is built for the second: a flat namespace of keys, each mapped to an immutable object, with enormous capacity, very high durability and a low price per byte. Big systems use both, and the standard pattern is to put the bytes in the object store and the facts about them in the database.

~~~dg-matrix What each kind of store is good at. They are almost exact complements.
Property | Relational database | Object storage
Query by any field, joins | yes | no, key or prefix only
Transactions across records | yes | no, one object at a time
Update part of a value | yes | no, replace whole object
Cost per GB stored | no, high | yes, low
Capacity | partial, one cluster | yes, effectively unlimited
Large sequential reads | partial | yes, parallel range GETs
Serve directly to browsers or a CDN | no | yes, signed URLs
~~~

## Why not put blobs in the database?

It works, and for small, few blobs it can be fine. At scale it hurts in several ways at once:

- **Cost.** Database storage is fast SSD, usually replicated two or three times, plus backups. Object storage is dense hard disks with erasure coding. The price difference per gigabyte is often five to ten times.
- **Memory pollution.** A database's buffer pool is meant to hold hot index and row pages. Reading a 5 MB image through it evicts thousands of useful pages.
- **Replication and backup.** Every blob write goes through the write-ahead log and to every replica. A burst of uploads can create replication lag for the whole database, and nightly backups grow with every photo ever uploaded.
- **Serving.** To show an image, the app server must read the blob from the database and stream it through itself. With object storage, a client or CDN fetches it directly, and the app server never touches the bytes.
- **Limits and operations.** Databases have practical limits on value size (PostgreSQL caps a single field at 1 GB; drivers and protocols often have lower limits), and very large tables are slow to migrate, vacuum and restore.

Object storage has the reverse problems. It can't answer "all photos by user 42 taken in March, newest first" without listing keys and reading each object. It has no transactions across objects and no partial updates. That's why the two are combined rather than chosen between.

~~~dg-bars Rough monthly cost to store 1 TB, by type of storage. Approximate public list prices; replicated database storage multiplies further.
*Object storage, standard | 23 | about $23
Object storage, infrequent access | 12.5 | about $12.50
Block SSD volume | 80 | about $80
Managed database storage x 3 replicas | 345 | about $345
~~~

## Case study: Facebook's Haystack

Haystack, Facebook's photo store described in 2010, is the classic example of why blobs need their own system, and of what a blob store looks like inside.

### The problem

Photos were first stored on NFS-mounted network filers, **one file per photo**. At the scale involved (about 260 billion images and 20 PB, with 1 billion new photos a week, roughly 60 TB, and peaks of over 1 million images served per second), the filesystem's own **metadata** became the bottleneck. To read one photo, the server had to read the directory entry to find the file's inode, read the inode to find the data's location, then read the data: **3 or more disk operations**, and the metadata for billions of files didn't fit in memory to cache. Most of the disk time was spent finding photos, not reading them.

### The fix: pack photos into big files, index them in RAM

Haystack packs many photos, called **needles**, into large append-only files of about **100 GB** called **physical volumes**. Each needle on disk carries a header (a random **cookie** to prevent guessing URLs, the photo's key, flags, size), the data, and a checksum. Each storage machine keeps an in-memory map from photo key to `(volume, offset, size)`: about **10 bytes of memory per photo** in its optimised form. A read becomes one memory lookup and **one disk operation**.

~~~dg-compare Reading one photo before and after Haystack. Moving per-photo metadata into a small in-memory index turns several disk operations into one.
[NFS, one file per photo]
Read -(directory entry)-> Inode -(block map)-> Data
- 3 or more disk operations per photo
- Billions of inodes do not fit in RAM
- Most disk time spent on metadata
[*Haystack, packed volumes]
Read -(RAM index, about 10 bytes)-> Volume offset -> Data
+ 1 disk operation per photo
+ Index for millions of photos fits in RAM
+ Large append-only files, sequential writes
~~~

Three components work together:

- **Directory**: maps each **logical volume** to the physical volumes (on different machines) that replicate it, decides which logical volume a new upload goes to, marks full volumes read-only, and builds the photo URL, which encodes the volume and photo IDs so the read path needs no further lookup.
- **Cache**: an internal cache in front of the stores for popular, recently uploaded photos (which receive most reads), complementing the external CDN.
- **Store**: the machines holding physical volumes and their in-memory indexes.

Deletes simply set a "deleted" flag in the in-memory index and on the needle. Space is reclaimed later by **compaction**: copy live needles to a new volume file, skipping deleted ones, then swap. That mattered: about 25% of photos were deleted each year. If this sounds familiar, it should: it's Bitcask's design (lesson 07.08) specialised for photos.

## The cloud object store view

S3 is the same idea at a much larger scale: hundreds of microservices over a storage fleet of millions of hard disks, designed for eleven nines (99.999999999%) of durability. Because every customer's data is spread over huge numbers of drives, a single customer's burst can be served by over a million disks at once, a level of parallel throughput no database cluster offers for blobs. Lessons 07.14 to 07.20 look inside a system like this.

## The usual split, done properly

- **Object store**: the bytes. Images, video, documents, backups, logs, Parquet files.
- **Database**: the metadata. Owner, object key, size, content type, checksum, created time, permissions, processing status, and any searchable attributes. Indexed for the queries the product needs.

The tricky part is keeping the two consistent, since there is no transaction spanning both. The standard ordering:

1. **Write the bytes first** (to a new, unique key, never overwriting an existing one).
2. **Then commit the metadata row** pointing at that key.
3. **Garbage-collect orphans**: objects with no metadata row (because step 2 failed or the user abandoned the upload) are found by a periodic sweep, or by a lifecycle rule on a staging prefix, and deleted.

With this order, a metadata row never points at missing bytes; the worst case is unreferenced bytes, which cost a little storage until the sweep removes them.

~~~dg-seq A direct-to-storage upload with a pre-signed URL. The app server never handles the image bytes; the metadata row is written only after the object exists.
Client > App server : want to upload
App server > App server : pick key, sign URL
App server -> Client : pre-signed PUT URL
Client > Object store : PUT bytes + checksum
Object store -> Client : 200 OK
Client > App server : done: key, checksum
App server > Database : INSERT photo row
App server -> Client : photo created
~~~

A **pre-signed URL** is a URL with an embedded signature that grants permission for one specific operation (PUT this key, before this time). It lets the client upload straight to the object store without credentials and without the bytes passing through your servers. Reads work the same way, often through a CDN that pulls from the object store.

<div class="ex">
<div class="lab">Worked example</div>

**a photo-sharing app**

Each upload goes to object storage at `photos/{user}/{uuid}.jpg`. A row in PostgreSQL stores the user, caption, upload time, the key, the size and a SHA-256. The feed query hits PostgreSQL by user and time, an indexed query returning 50 rows of about 200 bytes each. The client then fetches each image through a CDN that pulls from object storage on a miss.

Sizing: 10 million uploads a day at 2 MB average is 20 TB a day of bytes, about 600 TB a month, which object storage absorbs at tens of thousands of dollars a month. The metadata is 10 million rows a day at about 500 bytes each, 5 GB a day, under 2 TB a year: a comfortably sized database. Putting the 600 TB a month in PostgreSQL instead would be operationally and financially absurd.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**when an upload half-fails**

A user's upload succeeds to the object store, but the app server crashes before inserting the metadata row. The photo doesn't appear in their feed, so they upload again, which creates a new key and a new row. The first object is now an orphan. A nightly job lists keys created more than 24 hours ago under `photos/` and checks them against the database (or a lifecycle rule expires a staging prefix after a day), deleting about 0.1% of objects as orphans. No user ever saw a broken image, because no row ever pointed at missing bytes.

</div>

## When small blobs can stay in the database

Not every blob needs an object store. Avatars of a few kilobytes, small thumbnails or short documents, in modest numbers, can live in the database (or even inline in the row) to keep one system and one transaction. Benchmarks of embedded databases suggest blobs under roughly 100 KB can even be read faster from the database than from separate files. The decision flips when blobs are large, numerous, or need to be served directly to clients.

## Common mistakes

- **Overwriting objects in place** under the same key, so the metadata and bytes can disagree during the update. Use a new key per version.
- **Metadata first, bytes second.** A crash leaves a row pointing at nothing, and users see broken images.
- **No orphan cleanup.** Storage bills grow with failed uploads forever.
- **Proxying every download through app servers**, which wastes bandwidth and CPU; serve via signed URLs and a CDN.
- **Listing the object store to answer product queries.** Listing is slow and paginated; the database is the index.

## In an interview

This pattern comes up in nearly every design with media: "Uploads go directly to object storage via a pre-signed URL; once the object exists, we write a metadata row with key, size, checksum and owner to the database; reads come through a CDN; a background job removes orphaned objects." Mention why: cost per byte, durability, and that the database stays small and fast. If asked about consistency between the two, describe the bytes-first ordering and garbage collection.

<div class="take">
<div class="lab">Key takeaway</div>

put large immutable blobs in object storage and the small, queryable facts about them in a database. Write the bytes first under a new key, then commit the metadata row, and garbage-collect orphans; each system then does cheaply and quickly the job the other does badly.

</div>
