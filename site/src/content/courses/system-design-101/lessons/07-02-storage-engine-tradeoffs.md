---
id: "07.02"
course: "system-design-101"
module: "07"
slug: "storage-engine-tradeoffs"
title: "Storage engine tradeoffs"
keyIdeas: ["**B-tree engines** (InnoDB, Postgres heap + B-tree): update pages in place. Fast, predictable reads; random writes and page splits cost write throughput.", "**LSM engines** (RocksDB, Cassandra, LevelDB): buffer writes in memory, flush sorted files, merge them in the background. Very high write throughput, good compression; reads may check several files, and compaction uses I/O.", "The RUM trade-off: you can optimize two of Read, Update (write) and Memory/space overhead, not all three.", "Pick LSM for write-heavy workloads (logs, time series, messaging). Pick B-tree for read-heavy, latency-sensitive workloads with many range queries."]
question: "When would you choose an LSM tree over a B-tree?"
answer: "Choose an LSM tree when the workload is write-heavy or the data is large: logging, metrics, time series, event stores, messaging, feeds. It turns every write into a sequential log append and memory insert, flushes large sorted files, compresses well (immutable files) and is gentle on SSDs, so it sustains far higher write rates than a B-tree doing random page rewrites. The price is reads that may check several files (mitigated by Bloom filters and caching), background compaction I/O and occasional latency spikes. Choose a B-tree when reads, especially point lookups and range scans with predictable latency, dominate, or when you need mature transactional behaviour."
sources: [{"title": "RocksDB wiki: RocksDB Overview", "url": "https://github.com/facebook/rocksdb/wiki/RocksDB-Overview", "note": "LSM tradeoffs explained by its authors."}, {"title": "Use The Index, Luke: Anatomy of an index", "url": "https://use-the-index-luke.com/sql/anatomy", "note": "The B-tree side."}]
---

## Two families

Lesson 07.01 showed the two forces every storage engine must deal with: disks hate small random writes, and crashes can strike mid-write. Almost every engine in production answers them in one of two ways:

- **B-tree (update in place)**: data lives in fixed-size pages organised as a balanced tree. An update finds the page that holds the key and rewrites that page. PostgreSQL, MySQL's InnoDB, SQL Server, Oracle and SQLite work this way.
- **LSM tree (log-structured merge tree)**: writes go to an in-memory buffer and a log, then are flushed as sorted, immutable files that are merged in the background. RocksDB, LevelDB, Cassandra, ScyllaDB, HBase and many time-series databases work this way.

Both keep data sorted by key, so both support point lookups and range scans. The difference is **when** the sorting work happens. A B-tree pays at write time, by putting each key in its final place immediately. An LSM tree pays later, by appending now and sorting in large background merges. That single choice ripples into every other property: write throughput, read latency, space use, and how the engine behaves under load.

## The B-tree, page by page

A B-tree (strictly, the variant almost every database uses is a **B+ tree**) is built from **pages**, fixed-size blocks of typically 4, 8 or 16 KB. PostgreSQL uses 8 KB, InnoDB 16 KB. A page is the unit the engine reads from and writes to disk.

Two structures work together:

- **Leaf pages** hold the actual entries, sorted by key. In a secondary index an entry is the key plus a pointer to the row (a row ID or the primary key). In a clustered index, like InnoDB's primary key, the leaf holds the whole row. Each leaf also points to the previous and next leaf, forming a **doubly linked list**, so a range scan can walk sideways along the leaves forwards or backwards without going back up the tree.
- **Branch (internal) pages** sit above the leaves. Each entry holds a separator key and a pointer to a child page. A lookup starts at the **root**, picks the child whose key range covers the search key, and repeats until it reaches a leaf.

The tree is **balanced**: every leaf is the same distance from the root. That is guaranteed by how inserts work. When a leaf is full, it **splits** into two half-full pages and a new separator key is pushed into the parent. If the parent is full it splits too, and occasionally the split reaches the root, which splits and makes the tree one level taller. The tree grows from the top, so it never becomes lopsided. (The old advice to "rebuild unbalanced indexes" is a myth for this reason; a rebuild can reclaim space from half-empty pages, but balance is never the problem.)

~~~dg-tree A three-level B+ tree. Branch pages hold separator keys; only leaves hold entries, and leaves are linked for range scans.
*Root page | keys: 400, 800
  Branch: keys below 400 | 100, 200, 300
    Leaf 1 to 99 | sorted entries, next/prev links
    Leaf 100 to 199
    Leaf 200 to 299
    Leaf 300 to 399
  Branch: 400 to 799 | 500, 600, 700
    *Leaf 500 to 599 | lookup for 557 ends here
    Leaf 600 to 699
  Branch: 800 and up | 900, 1000
    Leaf 800 to 899
~~~

### Why the tree is so shallow

The trick is the **fan-out**: how many children each branch page has. An 8 KB page holding 16-byte keys plus 8-byte pointers fits roughly 300 entries. So:

- 1 level: 300 leaves.
- 2 levels: 300 × 300 = 90,000 leaves.
- 3 levels: 27 million leaves.
- 4 levels: about 8 billion leaves.

Depth grows logarithmically, which is why real-world indexes with millions or billions of rows are only three to five levels deep. Better still, the root and branch pages are few and constantly used, so they stay in the buffer pool (the database's page cache). A point lookup often costs just **one disk read**, for the leaf.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a B-tree index**

A table has 500 million rows. Its index on an 8-byte `user_id` stores key plus 8-byte row pointer, about 20 bytes per entry with overhead, so about 400 entries per 8 KB leaf when full. Pages are typically about 70% full after random inserts, so about 280 per leaf: 500M / 280 ≈ 1.8 million leaf pages, about 14 GB.

Branch fan-out is also a few hundred, so 1.8 million leaves need about 6,000 branch pages (50 MB), which need about 20 pages above them, then one root. That is 4 levels. The top three levels total about 50 MB and live in RAM, so a lookup costs one leaf read from disk, about 0.1 ms on an SSD.

</div>

### Three steps of an indexed query

An indexed lookup really has three steps: (1) the tree traversal down to the first matching leaf, (2) following the leaf chain sideways while keys still match, and (3) fetching each matching table row. Only the first is bounded by the tree depth. The second and third can touch many pages, which is why a query can be slow even when it uses an index: an index range that matches 50,000 rows scattered across the table can cost 50,000 random page reads.

### The cost of B-tree writes

B-trees are excellent at reads. The price is on writes:

- **Whole-page writes.** Changing a 100-byte row rewrites a whole 8 or 16 KB page. That is a write amplification of 80 to 160 times for that update, before any logging.
- **Random writes.** The pages to update are wherever the keys happen to live, scattered across the file.
- **Double writing for crash safety.** Because a page write can be torn by a crash, the engine first records the change in a **write-ahead log** (WAL) and may write full page images to it (PostgreSQL's full-page writes, InnoDB's doublewrite buffer). So each change is written at least twice.
- **Fragmentation.** Splits leave pages half full, and deletes leave holes, so a B-tree typically uses 1.3 to 1.5 times the space of its data.

Buffering helps. A modified page stays dirty in memory and many updates to it can be written out together at a checkpoint. But with a working set much bigger than RAM and keys spread randomly, the engine ends up doing one random page write for nearly every update.

## The LSM tree in brief

An LSM tree refuses to update anything in place. A write is appended to a write-ahead log (for crash safety) and inserted into a **memtable**, a sorted structure in RAM. When the memtable reaches a size limit (64 MB by default in RocksDB), it is written out in one sequential pass as a **sorted string table** (SSTable, or SST file): an immutable file of key-value pairs sorted by key, with an index. Over time many SST files pile up, arranged in **levels**, and a background process called **compaction** merges them, keeping the newest value of each key and discarding overwritten values and deletion markers. Lessons 07.07 to 07.12 take each piece apart.

~~~dg-compare The same single-row update on each engine. The B-tree rewrites a page where the key lives; the LSM appends and sorts later in bulk.
[B-tree]
Update -(find leaf)-> Read 8 KB page -(modify)-> Rewrite page in place
+ Read path stays short: one place per key
- Random page write per update
- Full page rewritten for a few bytes
[LSM tree]
Update -(append)-> WAL + memtable -(flush later)-> Sorted file
+ Writes are sequential and batched
+ Compresses well, gentle on SSDs
- Reads may check several files
- Compaction rewrites data in background
~~~

Writes are cheap and sequential: a write costs a log append and a memory insert, and the flush writes tens of megabytes at once. Reads are the price. The newest version of a key could be in the memtable or in any of several files, so a read may need to look in several places. That is why LSM engines lean on **Bloom filters** (a compact in-memory structure that says "this key is definitely not in this file") and a block cache (lesson 07.11).

## Comparing them properly

The honest comparison is in terms of three costs, which lesson 07.12 defines carefully: **read amplification** (disk reads per logical read), **write amplification** (bytes written to disk per byte the application wrote) and **space amplification** (disk used divided by live data size). No engine wins all three.

~~~dg-matrix How the two families compare on the properties that usually decide the choice. "partial" means it depends heavily on tuning.
Property | B-tree | LSM tree
Point read latency | yes, one path | partial, filters needed
Range scan | yes, walk leaves | partial, merge several runs
Write throughput | partial, random pages | yes, sequential
Space efficiency | partial, half-full pages | yes, compressed, compacted
Predictable latency | yes, mostly | partial, compaction spikes
SSD wear | partial, page rewrites | yes, large writes
Transactions, locking | yes, mature | partial, engine-specific
~~~

Some details behind that grid:

- **Compression.** LSM files are written once and never modified, so they can be compressed in large blocks. A B-tree page must leave room for future inserts and is rewritten in place, which makes compression awkward. Facebook moved its main MySQL user database from InnoDB to MyRocks (MySQL on RocksDB) largely for this: storage use roughly halved.
- **Write amplification is not one-sided.** An LSM rewrites data during compaction, often 10 to 30 times with leveled compaction. But those rewrites are large and sequential, while a B-tree's page writes are small and random. On SSDs the LSM's bytes are cheaper bytes.
- **Tail latency.** A B-tree's cost is paid inline, so it is steady. An LSM's cost is deferred, and if compaction falls behind, the engine must slow or stall incoming writes to catch up. That shows up as latency spikes at the 99.9th percentile.
- **Transactions.** B-tree engines have decades of work on locking, MVCC (multi-version concurrency control) and recovery. LSM engines support transactions too (RocksDB, CockroachDB's Pebble, TiKV), but the B-tree path is more mature.

<div class="ex">
<div class="lab">Worked example</div>

**metrics ingestion versus user accounts**

A monitoring system writes 500,000 data points a second, 100 bytes each: 50 MB/s of raw writes, keyed by series and timestamp, and mostly reads recent windows. A B-tree with a working set far bigger than RAM would need to modify hundreds of thousands of scattered leaf pages a second, far beyond what the disks can do.

An LSM engine absorbs the writes in the memtable, flushes a 64 MB sorted file every 1.3 seconds, and compresses the repetitive numeric data four to ten times. Compaction keeps reads bounded. The same company's user-account table (10 million rows, 50 writes a second, 20,000 lookups a second, needs foreign keys and transactions) stays in PostgreSQL, where every lookup is one buffered root-to-leaf walk.

</div>

## Common mistakes

- **"LSM is for writes, B-tree is for reads" as the whole answer.** True as a first cut, but say why: random in-place page writes versus sequential appends, and one place to look versus several.
- **Ignoring the working set.** If the whole B-tree fits in RAM, random writes are absorbed by the buffer pool and checkpointed in bulk, and the difference shrinks a lot.
- **Forgetting compaction is real work.** An LSM's write throughput is limited by how fast compaction can keep up, not by how fast the log can append.
- **Choosing an exotic engine for a small workload.** At a few thousand writes a second, PostgreSQL is fine and far easier to operate.

## In an interview

State the trade in one line, then tie it to numbers from the prompt: "This is 200,000 writes a second of append-mostly events, so I'd use an LSM-based store like Cassandra or RocksDB: writes become sequential appends and compress well. Reads are mostly by recent time range, which the LSM serves from the newest levels." For a transactional system with many reads and joins, say the opposite and pick PostgreSQL or MySQL. Good questions to ask: read/write ratio, value size, whether data is overwritten or only appended, and how strict the p99 latency target is.

<div class="take">
<div class="lab">Key takeaway</div>

B-trees keep data sorted in place, so reads are one short path down the tree but every update is a random page write. LSM trees append now and sort later in big background merges, so writes are sequential and compact but reads may check several files and compaction competes for disk. Choose by read/write mix, data size and how much latency jitter you can accept.

</div>
