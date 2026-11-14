---
id: "07.12"
course: "system-design-101"
module: "07"
slug: "compaction-and-amplification"
title: "Compaction and amplification"
keyIdeas: ["Compaction merges SSTables to drop overwritten and deleted data and keep the number of files a read must check low.", "**Leveled compaction** (RocksDB default, LevelDB): each level is about 10x larger than the previous, and files within a level don't overlap. Low read and space amplification, but data is rewritten many times (high write amplification).", "**Tiered / universal compaction** (Cassandra STCS): merge several similar-size files together. Less rewriting, but more files to read and more temporary space.", "**Write amplification** = bytes written to disk ÷ bytes written by the app. **Read amplification** = disk reads per query. **Space amplification** = disk used ÷ live data size."]
question: "Define write amplification. Why does leveled compaction have more of it than tiered?"
answer: "Write amplification is the total bytes physically written to storage (WAL, flush and every compaction rewrite) divided by the bytes the application wrote, so 20x means 1 GB of inserts causes 20 GB of disk writes. Leveled compaction keeps each level a single non-overlapping sorted run about ten times larger than the one above, so moving a file down one level means merging and rewriting it together with all the overlapping files of the next level, roughly ten times its size, and that happens again at every level as the data sinks, giving typically 10 to 30x. Tiered (universal) compaction lets several sorted runs of similar size accumulate and merges them only when enough pile up, so each byte is rewritten roughly once per tier and there are few tiers, giving lower write amplification at the cost of more runs to read and more duplicate data on disk."
sources: [{"title": "RocksDB wiki: Compaction", "url": "https://github.com/facebook/rocksdb/wiki/Compaction", "note": "Leveled vs universal compaction."}, {"title": "Mark Callaghan: Read, write and space amplification", "url": "https://smalldatum.blogspot.com/2015/11/read-write-space-amplification-pick-2_23.html", "note": "Pick 2."}]
---

## Three amplifications

To compare storage engines honestly you need a common currency. The standard one is **amplification**: how much more work the engine does than the application asked for. There are three kinds:

- **Read amplification**: the amount of work done per logical read, usually counted as disk reads (or blocks read) per lookup. You can measure it for a cold cache (the worst case) or with a realistic cache (the average case). A B-tree point lookup with the upper levels cached has a read amplification of about 1. An LSM lookup that checks 10 files without filters has about 10.
- **Write amplification**: the bytes physically written to storage divided by the bytes the application wrote. If you insert 1 GB and the disk ends up writing 20 GB (log, flush, every compaction rewrite), write amplification is 20.
- **Space amplification**: the size of the database on disk divided by the size of the live data in it. Dead versions, tombstones, half-empty pages and temporary compaction space all push it above 1.

These three trade off against each other. An engine can be very good at two of them, but making all three minimal at once appears impossible; this is sometimes called the **RUM conjecture** (Read, Update, Memory). It explains why no engine beats another on everything, and why the right question is always "which costs can this workload afford?"

~~~dg-matrix Where each design sits on the three amplifications. Every option is good at two at most.
Design | Read amp | Write amp | Space amp
B-tree | yes, low | no, page rewrites | partial, half-full pages
LSM leveled | partial, about 1 file per level | no, 10 to 30x | yes, about 1.1x
LSM tiered | no, many runs to check | yes, low | no, up to 2x or more
Append log, hash index | yes, one seek | yes, low until merge | no, garbage until merge
~~~

The costs also land at different times. A B-tree pays immediately: to modify a page it must first read it, and it writes the page back (eventually) on every change. An LSM tree **defers** cost to background compaction, which is why its foreground writes are fast and why it can fall behind. Space amplification comes from different places too: fragmentation and half-full pages in a B-tree, old versions and tombstones in an LSM.

## What compaction does

Compaction is the background process that merges SST files. It has three jobs:

1. **Bound read amplification** by reducing the number of files a read must check.
2. **Reclaim space** by dropping overwritten values and tombstones whose older versions have been merged away.
3. **Keep data sorted** in long runs so range scans merge few iterators.

Every compaction reads some input files, does a streaming k-way merge by key (keeping the newest version of each key), writes new output files and deletes the inputs. The **strategy** for choosing which files to merge, and when, decides the amplification trade-off. RocksDB offers two main ones.

## Leveled compaction

**Leveled compaction** (the RocksDB and LevelDB default) organises SST files into levels L1, L2, ... where:

- each level has a **target size**, about **10 times** larger than the level above (the **fan-out**);
- within a level, files have **non-overlapping** key ranges, so each level is one sorted run.

When a level exceeds its target, the engine picks a file from it and merges it with **all the files in the next level whose key ranges overlap it**. Because the next level is ten times larger, a 64 MB file typically overlaps about ten 64 MB files below. So pushing 64 MB down one level means reading and rewriting roughly 640 MB to 700 MB. Every byte gets rewritten like this once per level as it sinks.

~~~dg-timeline One flushed file's data as it sinks through leveled compaction. Each step down rewrites it together with the overlapping files of a level ten times larger.
t0 | Flush: 64 MB memtable becomes an L0 file (written once)
t0 + 5 min | L0 to L1: merged with overlapping L1 data
t0 + 1 hour | L1 to L2: merged with about 10 overlapping L2 files
t0 + 1 day | L2 to L3: rewritten again with about 10 files
t0 + 2 weeks | *L3 to L4: rewritten again; most data ends at the last level
~~~

**Write amplification** adds up: about 1 for the WAL, 1 for the flush, then (in the worst case) up to about the fan-out for each level. In practice leveled compaction typically lands at **10 to 30**.

**Read amplification** is low: one file per level (plus L0 files), and Bloom filters skip most of them (lesson 07.11).

**Space amplification** is excellent, about **1.1**. The last level holds about 90% of the data, and the levels above it are at most about 11% extra in total. Even if every upper-level entry is an overwrite of something in the last level, the dead data is bounded at roughly 10%.

## Tiered (universal) compaction

**Tiered compaction** (called **universal** in RocksDB, and size-tiered in Cassandra) does less rewriting. Instead of keeping each level as one sorted run, it lets several **sorted runs** of similar size accumulate, and merges them together only when enough have piled up. Think of it as: four 64 MB runs merge into one 256 MB run; four of those merge into one 1 GB run; and so on.

Each byte is rewritten roughly once per tier, and there are few tiers, so **write amplification is lower**, often 4 to 10.

The costs show up elsewhere:

- **Read amplification** is higher: many overlapping runs must be checked, not one per level.
- **Space amplification** is higher: duplicates of a key linger in several runs until they happen to be merged, and a big merge of the largest runs needs temporary space equal to their size. A full compaction can briefly need **double** the data size on disk. RocksDB's `max_size_amplification_percent` (200 by default) triggers a full merge when the extra space exceeds that percentage.

~~~dg-compare The two main compaction strategies. Leveled rewrites more to keep reads and space tight; tiered rewrites less and pays in reads and space.
[Leveled]
L0 -(merge with overlap)-> L1 -> L2 (10x) -> L3 (10x)
+ Read amp: one run per level
+ Space amp about 1.1x
- Write amp often 10 to 30x
[Tiered / universal]
4 runs of 64 MB -(merge)-> 1 run of 256 MB -> 1 GB
+ Write amp often 4 to 10x
- More runs to check per read
- Duplicates linger, up to 2x+ space
~~~

~~~dg-widget amp Rule-of-thumb amplification for leveled and tiered compaction. Adjust the settings to see how the three costs move against each other; real numbers depend on workload and tuning.
~~~

| | Leveled | Tiered / universal |
| --- | --- | --- |
| Write amplification | High (10 to 30x) | Low (4 to 10x) |
| Read amplification | Low | Higher |
| Space amplification | Low (about 1.1x) | Higher, can reach 2x or more |
| Temporary space | Small per compaction | Up to the size of the data |
| Good for | Read-heavy, space-sensitive, mixed | Write-heavy ingest, bulk loads |

### Other strategies worth knowing

- **Time-window compaction** (Cassandra's TWCS): for time-series data with a TTL, group data by time window and never merge across windows. When a whole window expires, delete its files outright with no compaction at all.
- **FIFO compaction**: keep files in order and simply delete the oldest when a size limit is reached. Perfect for caches and logs where old data is worthless.
- **Hybrids**: many systems use tiered compaction for the upper levels (to absorb writes cheaply) and leveled at the bottom (to keep reads and space tight).

## Why write amplification matters

Write amplification looks abstract until you price it:

- **Disk bandwidth.** Every rewrite competes with foreground reads and writes. If you ingest 100 MB/s at 20x, the disk must sustain 2 GB/s of compaction writes plus the reads to feed them.
- **SSD endurance.** Flash wears out. Drives are rated in **TBW** (terabytes written) or **DWPD** (drive writes per day). Application-level write amplification multiplies with the drive's own internal write amplification (lesson 07.01).
- **CPU.** Each rewrite decompresses, merges and recompresses data.

<div class="ex">
<div class="lab">Worked example</div>

**an SSD's lifetime**

An app writes 50 GB a day. With leveled compaction at about 20x write amplification, the SSD absorbs about 1 TB a day of writes. A drive rated for 1,000 TB written lasts about 1,000 days, under 3 years, before it reaches its rated endurance.

Switching to universal compaction at about 5x cuts the daily writes to 250 GB, so the same drive lasts over 10 years of writes. The price: reads check more sorted runs (more Bloom filter checks and more false-positive block reads), and the disk must keep up to about half its capacity free for full merges, so you buy roughly twice the raw capacity for the same data.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**adding up leveled write amplification**

Take a database of 1 TB with a 256 MB L1 and fan-out 10: L1 256 MB, L2 2.5 GB, L3 25 GB, L4 250 GB, L5 2.5 TB (only partly full). A newly written byte is written to the WAL (1), flushed to L0 (1), merged into L1 (about 1 to 2, since L0 and L1 are similar sizes), then into L2, L3, L4 and L5.

If each of those four moves rewrites it with, on average, about 5 times its size of overlapping data below (half the fan-out, since overlap varies), that's about 4 × 5 = 20. Total: about 1 + 1 + 2 + 20 = 24. Matches the "10 to 30" rule of thumb, and shows how a larger fan-out (fewer levels but more overlap each) or a smaller one (more levels) moves the number.

</div>

~~~dg-bars Daily bytes written to the SSD for 50 GB of application writes under each strategy.
Application writes | 50 | 50 GB
Tiered, 5x | 250 | 250 GB
*Leveled, 20x | 1000 | 1 TB
~~~

## Common mistakes

- **Comparing engines on one axis.** "LSM has lower write amplification than a B-tree" is often true, but not always: a leveled LSM can rewrite each byte 20 times. The comparison is "fewer, larger, sequential writes" versus "small random page writes".
- **Ignoring temporary space.** Tiered compaction can need as much free space as the data itself during a full merge.
- **Not budgeting compaction bandwidth.** Ingest that the disk can absorb once may be far more than it can absorb twenty times.
- **Using size-tiered for read-heavy, update-heavy data.** Many runs, many duplicates, slow reads and wasted space.

## In an interview

Define the three amplifications in one breath each, then say which you'd sacrifice: "This is an append-mostly metrics store with TTL, so I'd use time-window or tiered compaction to keep write amplification low; reads mostly hit recent windows so extra runs don't hurt much." Or: "This is a read-heavy user profile store, so leveled compaction: reads check about one file per level and space overhead is about 10%, at the cost of rewriting data 10 to 30 times." Mention SSD endurance if write volume is large.

<div class="take">
<div class="lab">Key takeaway</div>

compaction trades read, write and space amplification against each other, and no strategy wins all three. Leveled keeps one sorted run per level, so reads and space stay tight but every byte is rewritten once per level (10 to 30x); tiered merges similar-sized runs only when several pile up, so it rewrites much less but leaves more runs to search and more duplicate data on disk.

</div>
