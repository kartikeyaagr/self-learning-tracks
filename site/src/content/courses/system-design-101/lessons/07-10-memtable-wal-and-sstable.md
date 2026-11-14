---
id: "07.10"
course: "system-design-101"
module: "07"
slug: "memtable-wal-and-sstable"
title: "Memtable, WAL, and SSTable"
keyIdeas: ["**Memtable**: in-memory sorted structure (often a skip list) holding recent writes. Sorted so it can be flushed as a sorted file.", "**WAL**: every write is appended to the WAL first. If the process crashes, the memtable is lost, so it's rebuilt by replaying the WAL. Once a memtable is flushed, its WAL can be deleted.", "**SSTable** (sorted string table): immutable file of key-value pairs sorted by key, divided into blocks, with a block index and a Bloom filter at the end.", "Sorting allows merging files efficiently (like merge sort) and range scans."]
question: "Why does an LSM engine need a WAL if the memtable gets flushed to an SSTable anyway?"
answer: "The memtable lives only in RAM and is flushed only when it fills (or another trigger fires), which can be minutes and tens of megabytes after a write, so on its own every recent write would be lost in a crash. Flushing an SSTable per write would destroy the batching that makes LSM writes fast. The WAL is a cheap sequential append, fsynced in groups as configured, that records every write in order until the memtable holding it is safely in an SSTable; on restart the engine replays the WAL to rebuild the lost memtables, and a WAL file is deleted only once all memtables it covers have been flushed."
sources: [{"title": "RocksDB wiki: MemTable", "url": "https://github.com/facebook/rocksdb/wiki/MemTable", "note": "In-memory sorted buffer."}, {"title": "RocksDB wiki: Write Ahead Log", "url": "https://github.com/facebook/rocksdb/wiki/Write-Ahead-Log-%28WAL%29", "note": "Durability before flush."}, {"title": "Ilya Grigorik: SSTable and Log Structured Storage", "url": "https://www.igvita.com/2012/02/06/sstable-and-log-structured-storage-leveldb/", "note": "SSTable layout."}]
---

## Three pieces, three jobs

Lesson 07.09 followed a write through an LSM engine. This lesson looks closely at the three structures that write touches, because each one exists to solve exactly one problem, and seeing them separately makes the whole engine make sense:

- The **memtable** makes writes **fast**: it's in memory and keeps keys sorted.
- The **SSTable** makes data **compact and searchable on disk**: sorted, immutable, indexed, compressed.
- The **WAL** makes writes **durable** in between: until a memtable is safely flushed, the log is the only copy on disk.

The question at the heart of this lesson is the one people usually ask first: if the memtable is going to be flushed to an SSTable anyway, why write a WAL at all?

~~~dg-layers Where a recent write lives at each moment, and which copy makes it survive a crash.
*WAL on disk | durable copy until its memtable is flushed
Active memtable | in RAM, takes new writes, sorted
Immutable memtable | in RAM, frozen, waiting to flush
SST files in L0 and below | on disk, sorted, immutable, durable
~~~

## The memtable

The memtable is an in-memory sorted map from key to value. All writes go into it, and reads check it first because it holds the newest data.

### Why a skip list

RocksDB's default memtable is a **skip list**. Picture a sorted linked list where some nodes also have "express" pointers that skip ahead 2, 4, 8 or more nodes. A search starts on the highest express lane, drops down a lane when it would overshoot, and finds any key in O(log n) steps. Two properties make it ideal here:

- **Sorted order for free.** Flushing is just walking the bottom lane from start to end, producing keys already in order for the SST file.
- **Concurrency.** Inserting a node only requires updating a few pointers with atomic compare-and-swap operations, so many threads can read while writers insert, without locks. RocksDB supports concurrent inserts only with the skip list.

Other memtable types exist for special workloads: hash-based skip lists or linked lists that bucket keys by a prefix (fast when every lookup includes a known prefix, but slow for full ordered scans), and a vector memtable for bulk loads where keys are sorted once at flush time.

Memory for the memtable comes from an **arena**: large chunks allocated up front, from which small entries are carved. Nothing is freed individually. The whole arena is released at once when the memtable is flushed, which avoids allocator overhead and fragmentation.

### The knobs

- `write_buffer_size` (64 MB by default): how big one memtable grows before it is frozen.
- `max_write_buffer_number` (2 by default): the maximum number of memtables, active plus immutable. If the limit is reached because flushes are slow, writes stall.
- `db_write_buffer_size`: a cap on total memtable memory across all column families (a **column family** is a separate keyspace in one RocksDB instance with its own memtables and SST files, a bit like a table).

## The write-ahead log

The memtable is RAM. Between a write and the next flush, which could be minutes and tens of megabytes later, that write exists nowhere on disk **except in the WAL**. If the process crashes or the machine loses power, RocksDB rebuilds the lost memtables by replaying the WAL.

### The WAL's format

RocksDB inherits its log format from LevelDB. The file is divided into **32 KB blocks**. Each record has a small header (a 4-byte checksum, a 2-byte length and a 1-byte type) followed by the payload, which is a batch of writes with their sequence number. A record that doesn't fit in the rest of a block is split into fragments tagged FIRST, MIDDLE and LAST, while one that fits is tagged FULL. The fixed block structure means a reader that hits a corrupted region can skip to the next 32 KB boundary and resynchronise rather than losing everything after it.

### Durability choices

The WAL makes data durable only as far as you ask it to:

~~~dg-matrix What survives each kind of failure under different WAL settings. Faster settings give up protection against power loss or any crash.
Setting | Process crash | Power loss | Write cost
WAL, sync on every write | yes | yes | partial, an fsync per group
WAL, no sync (default) | yes | no, last seconds lost | yes, cheap
WAL, periodic fsync | yes | partial, last interval | yes, cheap
disableWAL | no, memtables lost | no | yes, cheapest
~~~

- `sync=true` on a write: the call returns only after the WAL is fsynced. Group commit amortises this across concurrent writers.
- `sync=false` (the default): the WAL write reaches the OS page cache. A **process** crash loses nothing, because the kernel still holds the data and will write it. A **power** failure can lose the last few seconds.
- `disableWAL`: no log at all. Useful for bulk loads you can redo from source, or caches. Every unflushed write is lost on any crash.

On recovery, RocksDB reads the WAL files newer than the last flush and replays them. A **recovery mode** decides what to do with damage: the default **point-in-time recovery** replays up to the first corrupted record and stops there, giving a consistent prefix of history. Stricter modes refuse to open on any corruption; looser ones skip bad records.

### When WAL files are deleted

A WAL file can be deleted once **every** memtable whose writes it contains has been flushed. With several column families sharing one WAL, a rarely written column family can hold a tiny memtable that never fills, pinning old WAL files forever. `max_total_wal_size` solves this: when the total size of live WAL files exceeds it, RocksDB forces a flush of the column families holding the oldest data, so those WAL files can go.

### Flush triggers, in summary

A memtable is flushed when:

- it reaches `write_buffer_size`;
- total memtable memory across column families reaches `db_write_buffer_size`;
- total WAL size reaches `max_total_wal_size`;
- a user asks for it (manual flush, or before a backup or shutdown).

## The SSTable

The flush turns a memtable into an **SST file** (sorted string table). It is written once, sequentially, and never modified. RocksDB's default "block-based table" layout looks like this:

~~~dg-tree Inside a block-based SST file. A reader starts at the fixed-size footer and works backwards to find the index and filter.
SST file
  Data block 1 | about 4 KB of sorted key-values, compressed
  Data block 2
  Data block N
  Meta blocks
    *Filter block | Bloom filter for all keys
    Properties block | entry count, key range, sizes
    Range deletion block | range tombstones
  Metaindex block | where each meta block is
  Index block | last key and offset of each data block
  Footer | fixed size, magic number, offsets
~~~

- **Data blocks** (4 KB by default before compression) hold sorted key-value entries. Keys within a block are **prefix-compressed**: each key stores only the bytes that differ from the previous key, with periodic **restart points** where a full key is stored, so a binary search can jump in.
- The **index block** has one entry per data block: a separator key at least as large as the block's last key, and a handle (offset and size). A lookup binary-searches the index, then reads one data block.
- The **filter block** holds a Bloom filter so lookups for absent keys can skip the file (lesson 07.11).
- The **footer** is a fixed size at the end of the file and ends with a magic number (lesson 07.04's checklist again). A reader opens the file by reading the footer, then the metaindex and index blocks it points to.
- Every block carries a checksum, verified when read.

Because the file is immutable it can be compressed in whole blocks (LZ4, Snappy or Zstandard), cached without invalidation, copied by backups while the database runs, and shared between checkpoints by hard links.

## Why the WAL is still needed

Now the main question can be answered precisely. Flushing an SST file on every write would make every write a full file creation: write the data, index, filter and footer, fsync, update the MANIFEST. That would destroy the whole point of batching tens of megabytes into one sequential write. The WAL is the cheap way to cover the gap: an append of a few hundred bytes, fsynced in groups, that records each write until its memtable lands safely in an SST.

~~~dg-timeline A crash between flushes. The memtable's 40 MB is lost from RAM but fully rebuilt from the WAL on restart.
10:00:00 | Memtable flushed to L0, new WAL 000123 started
10:00 to 10:07 | 40 MB of writes into memtable and WAL 000123
10:07:12 | *Power loss: memtable gone from RAM
10:08:30 | Restart: MANIFEST lists SSTs up to 10:00
10:08:31 | Replay WAL 000123: memtable rebuilt, 40 MB
10:08:33 | Accepting writes again, nothing acknowledged was lost
~~~

<div class="ex">
<div class="lab">Worked example</div>

**crash between flushes**

A node accepts 40 MB of writes into its 64 MB memtable, with `sync=true` for each write group, then loses power. On restart there are no new SST files since the last flush, but the WAL holds all 40 MB. RocksDB replays it at several hundred MB/s into a fresh memtable, which takes well under a second, and carries on. Every acknowledged write is present.

With `sync=false`, writes from roughly the last few seconds (whatever the OS hadn't yet written) could be missing, though everything before that survives. With `disableWAL`, all 40 MB would be gone, and the database would come back as it was at 10:00.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**how big the WAL gets and how long recovery takes**

A service writes 20 MB/s with 64 MB memtables and `max_write_buffer_number = 2`. At most about two memtables' worth, 128 MB, is unflushed at any time, so live WAL is normally around 64 to 128 MB and a crash means replaying at most that, a second or two.

Add a second column family that receives only 10 KB/s. Its memtable would take almost two hours to fill, and until it flushes, every WAL file written in that time must be kept: about 130 GB of WAL at 20 MB/s. Setting `max_total_wal_size` to, say, 1 GB forces that small memtable to flush whenever the WAL total reaches 1 GB, keeping disk use and recovery time bounded.

</div>

## Common mistakes

- **Assuming the default WAL is power-safe.** Without sync, it protects against process crashes, not power loss.
- **Disabling the WAL for speed** on data you can't regenerate.
- **Huge memtables to reduce compaction**, which also mean long WAL replays after a crash and bigger stalls when a flush is slow.
- **Forgetting cold column families** pin old WAL files.

## In an interview

If you mention an LSM-based store, say how it stays durable: "Every write goes to a write-ahead log and the memtable before it is acknowledged; the WAL is fsynced in groups; on a crash the memtable is rebuilt by replaying the WAL; WAL segments are deleted once their memtable is flushed into an immutable SSTable." That one sentence shows you understand the three pieces and why each exists.

<div class="take">
<div class="lab">Key takeaway</div>

the memtable makes writes fast and sorted in RAM, the SSTable makes them compact, indexed and immutable on disk, and the WAL makes them durable in between. Remove the WAL and every write since the last flush is lost on a crash; skip its fsync and power loss can take the last few seconds.

</div>
