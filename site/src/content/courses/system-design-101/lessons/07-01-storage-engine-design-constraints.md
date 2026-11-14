---
id: "07.01"
course: "system-design-101"
module: "07"
slug: "storage-engine-design-constraints"
title: "Storage engine design constraints"
keyIdeas: ["A storage engine turns get/put/scan calls into bytes on disk, and must survive crashes without corrupting data.", "Disks favour sequential I/O. Even on SSDs, large sequential writes beat small random ones, and every write wears the drive.", "`fsync` is what makes data durable; it's slow (often milliseconds), so engines batch writes per fsync (group commit).", "Memory is ~1000x faster than disk, so engines cache hot pages and keep indexes in memory where possible.", "Key design axes: update in place (B-tree) vs append-only (log-structured), how the index is kept, and how space is reclaimed."]
question: "Why is sequential disk I/O so much faster than random I/O, even on SSDs?"
answer: "A random I/O pays a fixed cost before any data moves. On a hard disk that is a seek plus half a rotation, about 10 ms, against microseconds to transfer 4 KB, so a drive does only about 120 random operations a second but streams around 200 MB/s. An SSD has no arm, but flash is written in pages and erased only in large blocks and cannot be overwritten in place, so small random overwrites leave stale pages scattered everywhere and force the drive's garbage collector to copy live data before erasing (device write amplification, wear and latency spikes). Large sequential writes fill and later free whole blocks, and the OS and drive can prefetch, merge and parallelise them into large requests."
sources: [{"title": "Build Your Own Database From Scratch", "url": "https://build-your-own.org/database/", "note": "Part I (simple KV store) is free online; starts from durability on files."}, {"title": "CMU 15-445: Storage lectures", "url": "https://15445.courses.cs.cmu.edu/", "note": "Disk-oriented storage and buffer pools."}]
---

## What a storage engine has to survive

A **storage engine** is the part of a database that turns "put this key" and "get that key" into bytes on a disk, and back again. Everything above it (the query planner, the network protocol, replication) assumes the engine keeps its promises: a value you wrote can be read back, and a write that was acknowledged survives a crash.

The obvious first design is to keep all the data in one file and rewrite it whenever something changes. That works for a settings file. It falls apart the moment the data is large, the writes are frequent, or the power goes off at the wrong instant. This module builds up, step by step, the designs real databases use instead. This first lesson sets out the two forces that shape every one of them:

- **The disk is slow, and it has rules.** It rewards large, sequential operations and punishes small, scattered ones. The penalty is huge on spinning disks and smaller, but still real, on SSDs.
- **The process can crash at any instant.** That includes halfway through writing a single page. The engine must come back up in a consistent state no matter where the crash landed.

Every engine you will meet in this module (B-trees, Bitcask, LSM trees, the append-only stream layer of a cloud store) is a different answer to the same question: how do you turn random updates into sequential writes, without losing anything when you crash?

## The memory and storage hierarchy

Before the disk, look at the whole ladder. Each step down is bigger, cheaper per byte, and much slower. An engine's job is largely to keep the hot data high on the ladder and to visit the bottom rungs as rarely, and as efficiently, as possible.

~~~dg-layers The storage hierarchy, fastest at the top. Each step down is roughly 10 to 1,000 times slower but far cheaper per byte.
CPU cache | about 1 to 10 ns, megabytes
*RAM (page cache, memtables) | about 100 ns, tens to hundreds of GB
NVMe SSD | about 20 to 100 µs per random read, terabytes
SATA SSD | about 100 to 200 µs per random read
Hard disk (HDD) | about 5 to 10 ms per random read, tens of TB
Object storage over network | about 10 to 100 ms per request, effectively unlimited
~~~

Put the numbers side by side and the shape of the problem is obvious. A random read from RAM is roughly 100,000 times faster than a random read from a hard disk. An engine that does one unnecessary disk seek per request has thrown away the speed of everything else in the stack.

~~~dg-bars Approximate latency of one small random read at each level. Note the log scale: the gap between RAM and a hard disk is about five orders of magnitude.
RAM | 0.1 | about 100 ns
NVMe SSD | 80 | about 80 µs
SATA SSD | 150 | about 150 µs
*Hard disk | 8000 | about 8 ms
Object store GET | 30000 | about 30 ms
~~~

## Why sequential beats random on a hard disk

A hard disk is a stack of spinning platters with a mechanical arm that moves a read/write head across them. To read a block at a random location the drive must:

1. **Seek**: move the arm to the right track. Typically 4 to 9 ms.
2. **Wait for rotation**: wait for the right sector to spin under the head. At 7,200 rpm one revolution takes 8.3 ms, so the average wait is about 4.2 ms.
3. **Transfer**: actually read the bytes. At 200 MB/s, a 4 KB block takes about 0.02 ms.

Look at those three numbers. Moving the data is almost free; getting to it costs about 10 ms. That fixed cost is why a hard drive manages only about **100 to 200 random operations per second** (call it 120), and that figure has barely changed in twenty years. Since IBM's first disk in 1956, capacity has grown about 7.2 million times, but seek time has improved only about 150 times, because arms and platters are mechanical and physics does not scale like transistors.

A sequential read pays the seek and rotation once, then streams at full bandwidth. Reading 100 MB sequentially takes about half a second. Reading the same 100 MB as 25,000 random 4 KB blocks takes about 25,000 × 10 ms = 250 seconds. Same bytes, 500 times slower.

<div class="ex">
<div class="lab">Worked example</div>

**random versus sequential on one hard disk**

A drive does 120 random I/Os per second and streams 200 MB/s sequentially. With random 4 KB writes it can absorb 120 × 4 KB = 480 KB/s of updates. If the same updates are appended to a log in 1 MB batches, each batch costs one positioning (about 10 ms) plus 5 ms of transfer, so about 66 batches a second, or 66 MB/s.

That is roughly 140 times more write throughput from identical hardware, purely by changing the access pattern. This one calculation explains why log-structured engines exist.

</div>

### Drives keep getting bigger, not faster

Because capacity grows and random I/O does not, the number of random operations per terabyte keeps falling. A 4 TB drive with 120 IOPS offers 30 IOPS per TB; a 26 TB drive with the same 120 IOPS offers under 5 IOPS per TB. Large storage services treat that as their central constraint: data that is rarely touched can live on huge drives, but hot data has to be spread over many drives so that no single one is asked for more than it can deliver. Lesson 07.14 shows how this "heat" problem shapes S3.

## SSDs: smaller gap, different reasons

An SSD has no moving parts, so there is no seek. A random 4 KB read on a modern NVMe drive takes roughly 20 to 100 µs, and the drive can serve hundreds of thousands of them per second if you keep many in flight. So why do storage engines still care about sequential writes on SSDs? Because of how flash memory works underneath.

- **Pages and blocks.** Flash is read and written in **pages** (typically 4 to 16 KB) but can only be erased in much larger **erase blocks** (often 1 to 4 MB, hundreds of pages).
- **No overwrite in place.** A page that holds data cannot be rewritten until its whole erase block is erased. So when you "overwrite" 4 KB, the drive's firmware (the **flash translation layer**, or FTL) writes the new data to a fresh page somewhere else, updates its internal map from logical address to physical page, and marks the old page as stale.
- **Garbage collection.** Eventually the drive runs short of fresh pages. It picks an erase block that is partly stale, copies its still-valid pages elsewhere, then erases the block. Those copies are extra writes the drive does on its own, called **device-level write amplification**.
- **Wear.** Each erase block survives a limited number of erase cycles (a few thousand for consumer TLC flash). Extra internal copying uses up the drive's life faster.

Now compare two workloads. Small random overwrites scatter stale pages across every erase block, so when garbage collection runs, each victim block still has many live pages to copy. Large sequential writes fill whole erase blocks with data that tends to become stale together (for example, a whole old log file deleted at once), so garbage collection finds blocks that are entirely stale and erases them with no copying at all.

~~~dg-compare What the SSD's garbage collector faces after two different write patterns. Sequential, whole-file writes leave erase blocks that die together.
[Small random overwrites]
Stale pages scattered in every block
Each victim block still has live pages
- GC copies live pages before erasing
- Device write amplification 2x to 5x
- Latency spikes when GC runs
[*Large sequential writes]
Whole blocks filled by one file
File deleted, block entirely stale
+ GC erases with no copying
+ Device write amplification near 1x
+ Steadier latency, longer drive life
~~~

There are two more effects. The operating system and the drive both **prefetch** (read ahead) and **merge** sequential requests, so a sequential stream turns into fewer, larger I/Os down the stack. And a big sequential request lets the drive spread the work across its many internal flash chips in parallel. On an SSD, sequential writes might be "only" two to five times better than random ones, rather than a hundred times, but they also make the drive last longer and behave more predictably.

## Durability is a contract with the operating system

The second force is crashes. Here is the trap almost everyone falls into the first time: calling `write()` does **not** put bytes on the disk. It copies them into the **page cache**, a region of RAM the operating system uses to buffer file data, and returns at once. The kernel writes those "dirty" pages to the disk later, in its own time, typically within about 30 seconds. If the machine loses power before that, the data is gone, even though your program was told the write succeeded.

To be sure, you call `fsync(fd)` (or `fdatasync`). It blocks until the kernel has sent the file's dirty pages to the device and the device reports them on stable storage. fsync is the **only durability barrier** you get from a normal filesystem. Nothing is durable until it returns.

~~~dg-seq A write is only durable once fsync returns. Before that it sits in the page cache in RAM and a power cut loses it.
App > OS page cache : write(fd, 4 KB)
OS page cache -> App : returns at once (not durable)
App > OS page cache : fsync(fd)
OS page cache > Disk : flush dirty pages + cache flush
Disk -> OS page cache : on stable storage
OS page cache -> App : fsync returns (durable)
~~~

fsync is slow. On a hard disk it costs at least one rotation, several milliseconds. On a consumer SSD it might take 1 to 10 ms, because the drive must flush its own volatile cache. Datacentre SSDs with power-loss protection (capacitors that keep the drive alive long enough to save its cache) can acknowledge in tens of microseconds. Either way, you cannot afford one fsync per tiny update at high rates, so engines use **group commit**: collect many updates from many clients, write them together, fsync once, and then acknowledge all of them.

<div class="ex">
<div class="lab">Worked example</div>

**group commit arithmetic**

A service receives 50,000 small writes a second, and an fsync on its SSD takes 2 ms. One fsync per write would need 50,000 × 2 ms = 100 seconds of fsync time per second: impossible. With group commit, the engine fsyncs every 2 ms and each fsync covers whatever arrived in that window, about 100 writes.

That is 500 fsyncs a second, each write waits at most about 4 ms (the current window plus the fsync), and every acknowledged write is durable. The cost is a few milliseconds of latency per write, paid for with a large gain in throughput.

</div>

### Torn writes and lying hardware

A crash can also land in the middle of a write. A database page is often 8 or 16 KB, but a disk only guarantees that a single sector (512 bytes or 4 KB) is written atomically. If power fails while a 16 KB page is being written, the page on disk can be part old and part new: a **torn write** (or torn page). An engine has to detect this, usually with a **checksum** stored in each page or record, and repair it, usually from a log. Lesson 07.03 shows the standard techniques.

Some hardware also lies: cheap drives may report a write as complete while it still sits in a volatile cache. Engines cannot fix that, which is why production databases run on drives with power-loss protection and why "fsync returned" is the strongest promise software can get.

## Common mistakes

- **Treating `write()` as durable.** Acknowledging a client after `write()` but before fsync loses data on power failure. It works perfectly in every test, because tests rarely pull the plug.
- **One fsync per operation.** Correct, but caps you at a few hundred or a few thousand writes a second. Batch with group commit instead.
- **Assuming SSDs make access patterns irrelevant.** Random small overwrites still cost device write amplification, wear and latency spikes.
- **Forgetting the directory.** Creating or renaming a file changes the directory, and that change also needs an fsync of the directory itself (lesson 07.03).
- **Ignoring torn pages.** Without checksums, a half-written page is silently read back as valid data.

## In an interview

When storage comes up, show that you know where the time goes. Useful lines: "A hard disk does about 120 random I/Os a second but streams 200 MB/s, so I want writes to be sequential." "Nothing is durable until fsync, so I'll batch commits." "On SSDs, sequential writes keep garbage collection cheap and extend drive life." Then connect it to a design choice: an append-only log for the write path, an in-memory index for reads, and periodic compaction. Asking "what's the write rate and the durability requirement?" tells the interviewer you know those two numbers decide the engine.

<div class="ex">
<div class="lab">Worked example</div>

**the counter file**

A service stores a counter by overwriting the 8 bytes at offset 0 about 2,000 times a second, with an fsync after each write. Each overwrite is a random write plus an fsync, so the disk, not the CPU, caps throughput, and on an SSD every overwrite also creates a stale flash page.

Switching to appending each increment to a log and fsyncing once per batch of 100 turns the work into sequential writes and only 20 fsyncs a second. A crash loses nothing that was acknowledged, because acknowledgements wait for the batch's fsync, and the latest value is simply the last record in the log.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

disks reward large sequential I/O and punish small random I/O, and nothing is durable until fsync returns. Every storage engine is a way to turn random updates into sequential writes, batch the fsyncs, and survive a crash in the middle of a write.

</div>
