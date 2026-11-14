---
id: "07.09"
course: "system-design-101"
module: "07"
slug: "lsm-tree-storage-engine"
title: "LSM-tree storage engine"
keyIdeas: ["An LSM tree (log-structured merge tree) writes to an in-memory sorted buffer (memtable) and a write-ahead log.", "When the memtable fills, it's flushed to disk as an immutable sorted file (SSTable).", "Background compaction merges SSTables, keeping the newest value per key and dropping deleted ones.", "Reads check the memtable, then SSTables from newest to oldest, using Bloom filters and indexes to skip files.", "Write path summary: API call → append to WAL → insert into memtable → acknowledge → later flush to an L0 SSTable → compaction pushes data to lower levels."]
question: "Walk through a write in an LSM tree from API call to disk."
answer: "The write is stamped with a sequence number, appended to the write-ahead log (optionally fsynced, usually batched with other writers via group commit) for durability, inserted into the in-memory sorted memtable (a skip list), and then acknowledged, with no SST file touched. When the memtable reaches its size limit it becomes immutable and a new memtable takes writes; a background thread flushes the immutable one in key order as a new sorted, immutable SST file in level 0 and records it in the MANIFEST, after which the old WAL segment is deleted. Background compaction later merges L0 files into L1 and each level into the next, roughly ten times larger, discarding older versions of the key; if flush or compaction falls behind, the engine slows or stops writes."
sources: [{"title": "RocksDB wiki: Overview", "url": "https://github.com/facebook/rocksdb/wiki/RocksDB-Overview", "note": "The LSM design in RocksDB."}, {"title": "O'Neil et al: The LSM-Tree (paper)", "url": "https://www.cs.umb.edu/~poneil/lsmtree.pdf", "note": "The original paper."}]
tools: ["RocksDB"]
---

## Where the idea came from

The **log-structured merge tree** (LSM tree) was first described in 1996, for workloads with very high insert rates and comparatively few reads, such as history tables that record every transaction. The problem then was the same as now: a B-tree index has to do a random disk write for almost every insert once the index is bigger than memory, and disk arms (the mechanical heads of a hard drive) are expensive. The proposal was to stop inserting into the on-disk structure one key at a time.

The original design had two components:

- **C0**, a small sorted tree held entirely in memory. Inserts go here, which costs no disk I/O.
- **C1**, a much larger sorted tree on disk.

When C0 grows past a threshold, a **rolling merge** takes a contiguous range of entries from C0 and merges them into C1, reading and writing C1 in large multi-page blocks. Because the merge walks both structures in key order, disk access is sequential, and one page write carries many inserts instead of one. The design generalises to a chain of components C0, C1, ..., Ck, each larger than the last.

The original cost example made the point in money. For 1,000 inserts a second into an index, a B-tree needed about **\$56,400** of hardware (mostly disks bought for their I/O rate, not their space, plus memory for the upper levels), while the LSM tree needed about **\$11,400**, because each disk page write was amortised across many inserts. It also used the **five-minute rule** (data accessed more often than roughly every five minutes is cheaper to keep in memory than to fetch from disk each time) to reason about how big C0 should be.

~~~dg-bars Disk plus memory cost to sustain 1,000 inserts a second in the original 1996 comparison. The LSM tree amortises each page write over many inserts.
B-tree | 56400 | about $56,400
*LSM tree | 11400 | about $11,400
~~~

Modern engines (LevelDB, RocksDB, Cassandra, ScyllaDB, HBase, Pebble in CockroachDB, and many others) keep the core idea but replace the in-place rolling merge with immutable files: the memory component is flushed as a whole sorted file, and files are merged into new files in the background. This lesson follows one write through RocksDB, the most widely used embedded LSM engine, from the API call to its final resting place.

## The pieces you need to know

- **WAL (write-ahead log)**: an append-only file on disk recording every write in arrival order. It exists only for crash recovery (lesson 07.10).
- **Memtable**: a sorted in-memory structure that holds recent writes. RocksDB's default is a **skip list**, a linked list with extra "express lane" pointers that allows O(log n) search and insert and, importantly, lets many threads read while one inserts. Default size: 64 MB (`write_buffer_size`).
- **Immutable memtable**: a full memtable that no longer accepts writes and is waiting to be flushed.
- **SST file (SSTable)**: an immutable file of key-value pairs sorted by key, divided into blocks, with an index and a Bloom filter (lesson 07.11).
- **Levels**: SST files are organised into levels L0, L1, ..., L6. L0 holds freshly flushed files, which may overlap each other in key range. From L1 down, each level's files cover **non-overlapping** key ranges, and each level is about **10 times larger** than the one above.
- **Sequence number**: every write gets a monotonically increasing 56-bit number. Internally a key is stored as (user key, sequence number, type), where type is "value" or "deletion". The newest version of a key is the one with the highest sequence number. Snapshots and consistent reads are built on this.

~~~dg-tree The shape of a RocksDB LSM tree with default-style sizing. L0 files overlap; deeper levels are sorted runs about ten times larger each.
Memory
  *Active memtable | 64 MB, takes writes
  Immutable memtable | waiting to flush
Disk
  WAL | every write, for recovery
  L0 | 4 or so overlapping files of about 64 MB
  L1 | 256 MB, non-overlapping
  L2 | 2.5 GB
  L3 | 25 GB
  L4 | 250 GB
  L5 | 2.5 TB, most of the data lives here
~~~

## A write, step by step

Here is what happens to `Put("user:42", value)`:

1. **Assign a sequence number.** The write is stamped with the next sequence number, say 9,001,234.
2. **Append to the WAL.** The record (key, value, sequence number, type) is appended to the current WAL file. Whether it is also fsynced depends on the write option `sync`. With `sync=true`, the write waits for the fsync; with `sync=false` (the default), it reaches the OS page cache and survives a process crash but not a power failure.
3. **Insert into the active memtable.** The skip list now holds the new entry, sorted among the others.
4. **Return to the caller.** That's it for the foreground path. No SST file was touched, nothing was read from disk, and no existing data was modified.
5. **Memtable fills.** When the active memtable reaches 64 MB, it is marked **immutable** and a fresh memtable (and usually a fresh WAL file) takes new writes. This switch takes microseconds.
6. **Flush.** A background thread iterates the immutable memtable in key order and writes it out as a new SST file in **L0**: data blocks, index, filter, footer. If the same key was written several times while in the memtable, only the newest version is written. The flush is a single sequential write of tens of megabytes.
7. **Retire the WAL.** Once the flush is durable and recorded in the **MANIFEST** (RocksDB's log of which files make up the database), the WAL file that covered that memtable is no longer needed and is deleted or recycled.
8. **Compaction.** When L0 has accumulated enough files (4 by default), a compaction merges them with the overlapping files in L1, producing new non-overlapping L1 files. When L1 exceeds its target size, some of its files are merged into L2, and so on down. Over its life, a key value moves down through the levels, rewritten at each step.

~~~dg-seq A RocksDB Put returns after the WAL append and memtable insert. Flush and compaction happen later on background threads, off the request path.
Client > DB : Put(user:42, v)
DB > WAL : append, seq 9001234
DB > Memtable : skip list insert
DB -> Client : OK, microseconds
Memtable > Memtable : 64 MB: immutable
Memtable > SST files : flush to new L0 file
SST files > WAL : old WAL retired
SST files > SST files : compact L0 into L1
~~~

### Group commit

Many client threads write at once. RocksDB batches them: one thread becomes the **leader** of a write group, appends all the group's records to the WAL in one write (and one fsync, if requested), then the records are inserted into the memtable, possibly in parallel. This is the group commit idea from lesson 07.01, and it is why an LSM engine can sustain hundreds of thousands of small synced writes a second on a good SSD.

## The memtable's lifecycle

~~~dg-state A memtable's life: it takes writes until full, waits as an immutable table, is flushed to an L0 file, and is then freed along with its WAL.
[*] -> Active
Active -> Immutable : reaches write_buffer_size
Immutable -> Flushing : background flush thread picks it
Flushing -> *Flushed : SST written, MANIFEST updated
Flushed -> Freed : WAL deleted, memory released
~~~

Reads can be served by the active memtable, any immutable memtables waiting to flush, and all the SST files, which is why a read path checks them in that order (lesson 07.11).

## What can slow a write down: stalls

The foreground path is designed to be fast, but it depends on background work keeping up. If it doesn't, RocksDB deliberately slows writers rather than let reads degrade without limit:

- **Too many memtables.** `max_write_buffer_number` (2 by default) caps active plus immutable memtables. If the active one fills while the previous one is still flushing, writes **stop** until the flush finishes.
- **Too many L0 files.** Every L0 file must be checked on a read because L0 files overlap. When L0 reaches 20 files (`level0_slowdown_writes_trigger`), writes are artificially slowed; at 36 (`level0_stop_writes_trigger`) they stop.
- **Too much pending compaction.** If the estimated bytes waiting to be compacted exceeds a threshold, writes are slowed, then stopped.

These **write stalls** are the LSM tree's characteristic failure mode: average latency looks great until ingest exceeds what flush and compaction can sustain, and then latency jumps from microseconds to seconds.

<div class="ex">
<div class="lab">Worked example</div>

**one write traced through time**

`Put("user:42", 120-byte value)` writes about 140 bytes to the WAL and the memtable and returns in about 10 µs. At 100 KB/s of writes, the 64 MB memtable takes about 11 minutes to fill; it is then flushed as a new L0 file in under a second.

An hour later, an L0 to L1 compaction rewrites the record into an L1 file. Days later, as L1 overflows, it moves to L2; weeks later to L3. Each move rewrites it once along with its neighbours. If `user:42` was overwritten in the meantime, the older version is dropped in whichever compaction first sees both. The user only ever waited for the first step.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing flushes and stalls**

A service ingests 80 MB/s. With 64 MB memtables, a memtable fills every 0.8 seconds. Flushing 64 MB to an SSD that sustains 500 MB/s takes about 0.13 seconds, so flush keeps up easily. But compaction must also handle 80 MB/s entering L0, multiplied by the write amplification (say 15x): 1.2 GB/s of compaction writes, more than the disk can do.

L0 files pile up: about 1.25 new files a second, so 20 files (the slowdown trigger) arrive within 15 to 20 seconds of compaction falling behind, and writes slow. Fixes: larger memtables, more compaction threads, a faster disk, tiered compaction with lower write amplification (lesson 07.12), or sharding the data across more machines.

</div>

## Why this design wins for writes

Compare the cost per write with a B-tree whose working set doesn't fit in memory:

| Step | B-tree insert | LSM insert |
| --- | --- | --- |
| Durability | WAL append | WAL append |
| Data structure | Read leaf page (often a random disk read), modify | Insert into memory |
| Eventual disk write | Random 8 to 16 KB page write | Share of a sequential 64 MB flush |
| Background | Page splits, checkpoints | Compaction, rewriting data several times |

The LSM tree replaces a random read and a random write per insert with a sequential share of a big flush, plus later sequential compaction. On spinning disks that is a huge win; on SSDs it is still a large win in throughput and device wear.

## In an interview

If asked to walk through an LSM write, give the steps in order and say which are synchronous: "WAL append, memtable insert, acknowledge. Later, the memtable is frozen and flushed as a sorted SST file in L0; compaction merges it down into larger, non-overlapping levels; the WAL segment is deleted once its data is in SSTs." Then add one sentence on the risk: "if compaction can't keep up, L0 files pile up and the engine stalls writes, so I'd size disk bandwidth for ingest times write amplification."

<div class="take">
<div class="lab">Key takeaway</div>

an LSM tree buffers writes in a sorted in-memory memtable, makes them durable with a sequential write-ahead log, and moves them to disk in large sorted batches. Background compaction merges those batches down through levels to keep reads manageable, and if it falls behind the engine stalls writes.

</div>
