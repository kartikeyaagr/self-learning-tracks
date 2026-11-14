---
id: "07.07"
course: "system-design-101"
module: "07"
slug: "log-structured-storage"
title: "Log-structured storage"
keyIdeas: ["Log-structured storage turns every write into an append to the end of a log. Appends are sequential and fast, and crash recovery is simple.", "Updates and deletes are just new entries (a delete is a \"tombstone\"); the latest entry for a key wins.", "To read efficiently you need an index from key to log position.", "The cost comes later: the log grows forever with stale entries, so you must compact (rewrite live data, drop dead entries). Compaction uses CPU and I/O and causes write amplification.", "Jay Kreps' \"The Log\": an ordered append-only log is also the core of replication and stream processing."]
question: "Append-only writes are fast. What do you pay for them later?"
answer: "You pay in space, because overwritten values and tombstones stay on disk until compaction removes them and compaction itself needs free headroom while inputs and outputs coexist. You pay in extra I/O and CPU, because compaction rereads and rewrites the same data many times (write amplification). You pay in read cost, because the newest value of a key may be in the memtable or any of several files, so reads need indexes, Bloom filters and caches, and missing keys are expensive. And you pay in latency spikes, because compaction competes with live traffic and, if it falls behind, the engine must throttle or stall writes."
sources: [{"title": "Jay Kreps: The Log", "url": "https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying", "note": "Logs as the core abstraction."}, {"title": "Ilya Grigorik: SSTable and Log Structured Storage", "url": "https://www.igvita.com/2012/02/06/sstable-and-log-structured-storage-leveldb/", "note": "From append-only logs to LevelDB."}]
---

## The log as the core abstraction

A **log** in this sense is not a debug text file. It is an **append-only, totally ordered sequence of records**, where each record gets the next sequential number (its **offset** or log sequence number). You can only add to the end. You can read from any position. Records are never modified once written.

That sounds too simple to matter, yet logs sit underneath almost every serious data system:

- **Databases** write every change to a write-ahead log before applying it (lesson 07.10), so a crash can be recovered by replaying the log.
- **Replication** works by shipping the log. If replicas start in the same state and apply the same records in the same order, they end in the same state. This is called **state machine replication**, and it is how PostgreSQL streaming replication, MySQL's binlog replicas and Raft-based systems all work.
- **Data integration** uses logs as the hub between systems. Change data capture streams a database's log into Kafka, and every downstream system (search index, cache, warehouse) reads from there.

### Tables and logs are two views of the same thing

A **table** holds the current value of each key. A **log** holds the history of changes. If you replay the log from the start, you get the table. If you record every change to the table, you get the log. This **table/stream duality** is why a log is enough to rebuild a database, populate a cache, or create a new replica: they are all "the state after applying the log up to offset N".

### Why a central log simplifies integration

Picture N systems (a database, a search index, a cache, a warehouse, a recommendation service...) that all need each other's data. Wiring each pair directly gives up to N × (N − 1) pipelines, each with its own format, failure modes and backfill story. With 10 systems that is 90 pipelines. A central log changes it to N writers and N readers: each system publishes its changes once and subscribes to what it needs.

~~~dg-compare Integrating many systems. Point-to-point pipelines grow with the square of the number of systems; a shared log grows linearly.
[Point-to-point]
DB -> Search, DB -> Cache, DB -> Warehouse
- 10 systems: up to 90 pipelines
- Each pipeline fails differently
- Adding a system touches many others
[*Central log]
DB -(append)-> Log -(subscribe)-> Search, Cache, Warehouse
+ 10 systems: 10 writers, 10 readers
+ New reader replays from offset 0
+ One ordering, one format
~~~

LinkedIn built Kafka on exactly this idea and ran it at over 60 billion writes a day. The performance came from treating the log as a disk-friendly structure: the log is **partitioned** so many machines append in parallel, reads and writes are **batched** so each disk operation moves a lot of data, and data is sent from the page cache to the network with **zero-copy** (`sendfile`), so bytes never pass through the application's memory.

## Log-structured storage

Applied to a storage engine, the idea is radical: **every write is appended to the end of a file, and nothing is ever updated in place**. Setting `user:42` appends a record. Setting it again appends another record. Deleting it appends a third. The engine's job is to make reads efficient despite that.

What you get immediately:

- **Fast writes.** Every write is a sequential append, the access pattern disks love (lesson 07.01). Writes can be batched and fsynced together.
- **Simple crash recovery.** Old data is never touched, so a crash can only damage the record being appended at the tail. Recovery checks the tail's checksum and truncates (lesson 07.03).
- **Easy replication and backup.** Immutable data can be copied while the engine keeps running.

What you lose: the newest value of a key could be anywhere in the file. Scanning the whole log to answer a read is obviously unacceptable, so log-structured engines add an index. There are two main designs:

- **Hash index in memory**: keep a hash table from each key to the file offset of its latest record. Reads are one lookup plus one disk read. This is Bitcask (lesson 07.08). The limit is that every key must fit in RAM, and there are no range scans.
- **Sorted runs**: buffer recent writes in a sorted in-memory structure, the **memtable**, and when it is full write it out as a **sorted string table** (SSTable), an immutable file of key-value pairs sorted by key with a small index. Over time you have many SSTables, so you merge them in the background. This is the **LSM tree** (lessons 07.09 to 07.12).

~~~dg-flow The log-structured write path: sorted in memory, flushed as immutable sorted files, merged in the background.
Write -(append)-> WAL -> Memtable -(when full)-> SSTable 1 -(compaction)-> *Merged SSTable
~~~

### Deletes are writes too

You can't remove a record from the middle of an immutable file, so a delete appends a **tombstone**: a marker that says "this key is gone as of now". Reads that find the tombstone first report "not found". The tombstone has to stay until compaction has merged it with every older file that might still hold a value for the key; drop it too early and the old value would reappear, a bug known as **data resurrection**. Cassandra's `gc_grace_seconds` (10 days by default) exists for this reason in a replicated setting: tombstones are kept long enough for every replica to have seen them.

### Compaction

**Compaction** is the background process that pays the bill. It reads several files, merges them by key, keeps only the newest value for each key, drops overwritten values and (when safe) tombstones, and writes the result as new files. Then it deletes the inputs. Because the input files are sorted, this is a streaming merge, like the merge step of merge sort: read each input sequentially, write the output sequentially, use little memory.

## What you pay later

Append-only writes are fast because they defer work. The deferred work comes back in four forms.

### 1. Garbage (space amplification)

Every overwrite and delete leaves a dead record on disk until compaction removes it. A workload that updates the same keys repeatedly can hold many times more bytes on disk than it has live data.

~~~dg-timeline Disk use for a log-structured store whose live data stays at 10 GB but whose keys are constantly overwritten. Garbage builds up until compaction reclaims it.
00:00 | 10 GB on disk, all live
01:00 | 16 GB on disk: 6 GB dead versions
02:00 | 22 GB on disk: compaction starts
02:20 | *Compaction rewrites live data: needs 10 GB extra temporarily
02:40 | Inputs deleted: back to 10 GB
~~~

Note the moment of peak usage: during compaction both the inputs and the output exist. You need free disk headroom for compaction to run at all. Running a log-structured store at 95% disk full is how outages start.

### 2. Compaction I/O (write amplification)

Data is written once to the log, then rewritten every time compaction merges it. In a leveled LSM tree a byte may be rewritten 10 to 30 times over its life. That is **write amplification**, defined properly in lesson 07.12. The writes are sequential, so they are cheap per byte, but they still consume disk bandwidth, CPU (for decompression, merging and recompression) and SSD endurance.

### 3. Read cost (read amplification)

A key might be in the memtable or in any of several files. A read checks them newest first and stops at the first hit, but a key that doesn't exist must be checked everywhere. Engines add per-file indexes, Bloom filters and caches (lesson 07.11) to keep this cheap.

### 4. Latency spikes

Compaction competes with live traffic for disk, CPU and cache. If writes arrive faster than compaction can merge them, files pile up, reads slow down, and eventually the engine must **throttle or stall** incoming writes to let compaction catch up. That shows up as sudden p99 latency spikes, usually under the heaviest load.

~~~dg-bars On-disk bytes for one session key updated 1,000 times in an hour (500-byte records), before and after compaction.
Live data | 0.5 | 0.5 KB
*Before compaction | 500 | about 500 KB
After compaction | 0.5 | 0.5 KB
~~~

<div class="ex">
<div class="lab">Worked example</div>

**overwriting a hot key**

A user's session record (500 bytes) is updated 1,000 times in an hour. A log-structured engine writes 1,000 records, about 500 KB, each a cheap sequential append. Until compaction runs, 999 of them are dead weight. Multiply by 2 million active sessions and the engine holds about 1 TB of mostly dead records for 1 GB of live data.

Compaction must read all of that to keep the last version of each key. In practice LSM engines soften this: overwrites of a key still in the memtable replace it in memory and never reach disk, and frequent flushes plus compaction keep only a few generations around. But it shows why update-heavy workloads drive compaction cost.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**running out of headroom**

A node has a 2 TB disk, 1.6 TB used (80%). A large compaction needs to merge 500 GB of input into roughly 450 GB of output. Until the inputs are deleted, the disk needs 1.6 + 0.45 = 2.05 TB. The compaction fails for lack of space, files keep piling up, read latency climbs, and the node stalls writes. The fix is operational: alert well before disks are 70 to 80% full, and add capacity rather than wait.

</div>

## Common mistakes

- **Treating write speed as free.** The log is fast because compaction pays later. Plan disk bandwidth and space for it.
- **Dropping tombstones early.** Old values resurrect. Tombstones must outlive every older copy of the key.
- **Filling disks.** Compaction needs temporary space. A full disk stops compaction, which makes things worse, not better.
- **Ignoring tail latency.** Average write latency looks great; p99.9 during a compaction backlog does not.

## In an interview

When you propose a log-structured store (Cassandra, RocksDB-backed systems, Kafka-like logs), name the bill in the same breath: "Writes are sequential appends, so ingest is cheap; the cost is compaction, which uses disk bandwidth and temporary space and can cause latency spikes, and reads may check several files, which Bloom filters mitigate." Ask about the update pattern: append-only time series compacts cheaply; heavy overwrites of the same keys does not.

<div class="take">
<div class="lab">Key takeaway</div>

append-only storage makes writes fast, sequential and crash-safe, and the log doubles as the basis for replication and integration. The bill comes later as garbage on disk, compaction I/O, reads that may search several files, and latency spikes when compaction falls behind.

</div>
