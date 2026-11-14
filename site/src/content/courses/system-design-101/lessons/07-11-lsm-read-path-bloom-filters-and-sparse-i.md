---
id: "07.11"
course: "system-design-101"
module: "07"
slug: "lsm-read-path-bloom-filters-and-sparse-index"
title: "LSM read path: Bloom filters and sparse index"
keyIdeas: ["An LSM read might have to check many SSTables. Missing keys are the worst case: you'd check every file to be sure.", "**Bloom filter per SSTable**: answers \"definitely not here\" for most files, so you skip them without disk I/O.", "**Sparse index**: one entry per data block (first or last key of each block), kept in memory. Find the block with a binary search, then read just that block.", "Block cache keeps hot blocks in memory.", "Range queries can't use Bloom filters (prefix Bloom filters help for prefix scans)."]
question: "Why are reads for missing keys the worst case in an LSM, and how do Bloom filters help?"
answer: "A lookup must return the newest version, so it checks the memtables, then every overlapping L0 file, then one file per level, from newest to oldest, stopping at the first match. A key that exists is usually found early, but a missing key has no match, so the search can never stop early and must check every place, which without help means one block read per file (about 10 reads for 10 files). A Bloom filter per SST file, held in memory, answers \"definitely not here\" or \"maybe here\" with no false negatives; at about 10 bits per key only about 1% of files that lack the key give a false \"maybe\", so most files are skipped without any disk I/O, and the per-file index block ensures any file that must be checked costs just one data block read."
sources: [{"title": "RocksDB wiki: Bloom Filter", "url": "https://github.com/facebook/rocksdb/wiki/RocksDB-Bloom-Filter", "note": "Skipping SSTables on reads."}, {"title": "RocksDB wiki: Index Block Format", "url": "https://github.com/facebook/rocksdb/wiki/Index-Block-Format", "note": "Sparse index per SSTable."}]
tools: ["RocksDB"]
---

## Where a read has to look

An LSM tree never overwrites data in place, so several versions of the same key can exist at once: one in the memtable, an older one in an L0 file, an even older one down in L3. The rule that makes reads correct is simple: **the newest version wins**. So a point lookup (`Get(key)`) searches from newest to oldest and stops at the first match:

1. The **active memtable**.
2. Any **immutable memtables** waiting to flush, newest first.
3. Every **L0 file**, newest first. L0 files are flushed memtables and their key ranges overlap, so the key could be in any of them.
4. **At most one file per level** from L1 down. Within a level, files cover non-overlapping key ranges, so the engine looks up which single file's range contains the key (a binary search over file boundaries held in memory) and checks only that one.

If the first match is a value, that's the answer. If it's a **tombstone** (a deletion marker), the answer is "not found", and the search stops there, because anything older is dead.

### Why missing keys are the worst case

A key that exists, especially a recently written one, is often found early, in the memtable or a shallow level. The search stops. A key that **doesn't exist** has no match anywhere, so the search can't stop early: it must check the memtables, every L0 file, and one file in every level below. With 4 L0 files and 6 levels, that's 10 files. If each check meant a disk read, a lookup for a missing key would cost 10 random reads, which on an SSD is about 1 ms and on a hard disk about 100 ms. That's **read amplification** at its worst (lesson 07.12).

And missing-key lookups are common: "does this username exist?", "have I seen this event ID before?" (deduplication), "is this URL already crawled?", or an insert that must first check the key is new. This lesson is about the two structures that make those lookups cheap: an index inside each file, and a Bloom filter in front of it.

~~~dg-flow A point read checks newest to oldest and stops at the first hit. Bloom filters let it skip most files without any disk read.
Get(k) -> Memtables -(miss)-> L0 files -(Bloom: no)-> L1 skip -(Bloom: no)-> L2 skip -(Bloom: maybe)-> *L3: read 1 block -> Found
~~~

## Inside one SST file: the index block

Suppose the engine has decided to look in one SST file. The file may be 64 MB or larger, so it must not scan it. As lesson 07.10 showed, the file is made of **data blocks** of about 4 KB, plus an **index block** with one entry per data block: a separator key that is at least the last key in that block, and a **block handle** (offset and size).

A lookup binary-searches the index for the first entry whose separator is greater than or equal to the target key. That entry names the one data block that could contain the key. The engine reads that block (from the **block cache** if it's there, otherwise from disk), then searches within it, using the block's restart points to binary-search the prefix-compressed keys. One block read per file, at most.

How big is the index? A 64 MB file with 4 KB blocks has about 16,000 blocks; at about 30 bytes per index entry that's around 500 KB, under 1% of the file. For very large files, RocksDB can **partition** the index: split it into pieces with a small top-level index over them, so only the top level must stay in memory and the rest is cached on demand like data blocks.

## Bloom filters

The index tells you which block **might** hold the key, but it can't tell you the key is absent without reading that block. That's still one disk read per file for a missing key. A **Bloom filter** removes most of those reads.

### How a Bloom filter works

A Bloom filter is a bit array of m bits, all starting at 0, plus k hash functions.

- **Add a key**: hash it k times, giving k bit positions, and set those bits to 1.
- **Check a key**: hash it the same k ways. If **any** of those bits is 0, the key was **definitely never added**. If all are 1, the key is **probably** present: those bits might all have been set by other keys.

So a Bloom filter can produce **false positives** (says "maybe" for an absent key) but **never false negatives** (it never says "no" for a key that is there). That asymmetry is exactly what a read path needs: a "no" lets you skip the file safely; a "maybe" costs one wasted block read at worst.

~~~dg-seq Checking one SST file for a key. The filter answers from memory; only a "maybe" leads to an index lookup and a block read.
Read path > Bloom filter : test k bits
Bloom filter -> Read path : a bit is 0: absent
Read path > Read path : skip file, no I/O
Read path > Bloom filter : next file: test
Bloom filter -> Read path : all 1: maybe
Read path > Index block : binary search
Index block -> Read path : block 2,417
Read path > Block cache : get block 2,417
Block cache -> Read path : miss: 4 KB disk read
~~~

### The numbers

The false-positive rate depends on how many bits you spend per key. With the best choice of k (about 0.7 × bits per key), the trade-off is:

~~~dg-bars Bloom filter false-positive rate by bits per key. Each extra 4.8 bits or so cuts the false-positive rate about tenfold.
4.8 bits per key | 10 | 10% false positives
*9.6 bits per key | 1 | 1% false positives
14.4 bits per key | 0.1 | 0.1% false positives
~~~

The usual default is about 10 bits per key, giving roughly a 1% false-positive rate. At that setting, a filter costs **1.25 bytes per key**. A database with 1 billion keys spends about 1.25 GB of memory on filters, which is affordable on a server and saves vast numbers of disk reads.

RocksDB offers variants:

- **Full filter** (the default today): one filter per SST file, covering all its keys, checked before reading any data block. The older **block-based filter** kept one small filter per data block, which meant reading the index first and checking more filters.
- **Partitioned filters**: like partitioned indexes, the filter is split into pieces with a top-level index so huge files don't need the whole filter in memory.
- **Ribbon filter**: a newer structure that saves about **30% of the memory** for the same false-positive rate, at the cost of more CPU to build. Good for the large, rarely rewritten bottom levels.
- **Prefix Bloom filters**: hash only a key prefix (say, the user ID part of `user42:order:...`), so a filter can answer "are there any keys with this prefix?", which helps prefix scans.

### What filters can't do

A normal Bloom filter answers "is this exact key here?". It can't help with a **range scan** ("all keys from A to B"), because a range has no single key to hash. Range scans instead open an iterator on every memtable and every relevant file in every level, and merge them, like a k-way merge sort, taking the newest version of each key. That's why LSM range scans are more expensive than B-tree range scans, and why compaction (fewer, larger sorted runs) matters so much for scan-heavy workloads.

## Caching: keeping it in memory

Filters and indexes only help if checking them doesn't itself need a disk read. RocksDB's **block cache** (an LRU cache of uncompressed blocks in memory) holds data blocks, and can be told to hold index and filter blocks too, with high priority, or to **pin** the index and filter blocks of L0 files (which are checked on almost every read). The OS page cache holds compressed file pages on top of that. A well-tuned engine has every filter and index block in memory, so a lookup's only possible disk read is the final data block.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of a missing-key lookup, with and without filters**

An LSM tree has 4 L0 files and levels L1 to L6, so a missing key means checking 10 files. Without filters, each check is an index lookup (in cache) plus one 4 KB data block read (usually not in cache for random keys): about 10 block reads, roughly 10 × 100 µs = 1 ms on an NVMe SSD.

With 10-bit filters, each file's filter is checked in memory in well under a microsecond, and only about 1% of files give a false "maybe". Expected block reads: 10 × 0.01 = 0.1, so about 10 µs of disk time on average. That's roughly 100 times fewer disk reads, for 1.25 bytes of memory per key.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**checking usernames**

A signup form checks whether a name is taken, and 95% of checks are for names that don't exist. The username table lives in an LSM store where a lookup touches 6 files. Without filters, every missing-name check reads 6 blocks: at 20,000 checks a second that's 114,000 random reads a second, beyond what a single SSD comfortably sustains alongside other work.

With 10 bits per key, each file is skipped 99% of the time, so a missing-name check averages about 0.06 block reads: about 1,140 random reads a second for the missing names, plus at most a few block reads each for the 1,000 names a second that do exist. The filters turned an I/O-bound endpoint into a memory-bound one.

</div>

~~~dg-compare The same lookup for a key that doesn't exist, with and without Bloom filters, in a tree with 10 files to check.
[No filters]
Memtable -(miss)-> 10 files -(1 block each)-> Not found
- About 10 random block reads
- About 1 ms on NVMe
- Misses cost the most
[*10-bit Bloom filters]
Memtable -(miss)-> 10 filters in RAM -(about 0.1 reads)-> Not found
+ About 0.1 block reads on average
+ Costs 1.25 bytes of RAM per key
- No help for range scans
~~~

## Tuning levers and pitfalls

- **Bits per key.** More bits, fewer false positives, more memory. Many deployments use 10; some skip filters on the last level (`optimize_filters_for_hits`) when nearly every lookup is for a key that exists, because the last level holds about 90% of the data and its filter is the most expensive.
- **Filters not in memory.** If filter blocks are evicted from the cache, checking them costs a disk read, which defeats the purpose. Pin or prioritise them.
- **Too many L0 files.** Every L0 file must be checked; a compaction backlog raises read cost directly (and triggers write stalls, lesson 07.09).
- **Hot keys with many versions.** A frequently updated key may be found quickly, but range scans must skip all its older versions and tombstones until compaction removes them. Scans through regions full of tombstones are a classic Cassandra performance problem.

## In an interview

If the design uses an LSM store for lookups, especially existence checks, say: "Reads go newest to oldest and stop at the first hit, so missing keys are the expensive case; each SST file has a Bloom filter in memory, about 10 bits per key for a 1% false-positive rate, so most files are skipped without I/O, and an index block narrows each remaining file to one 4 KB block." It's a crisp answer that shows you understand both the data structure and the cost model.

<div class="take">
<div class="lab">Key takeaway</div>

an LSM read searches memtables, every L0 file and one file per lower level, newest to oldest, stopping at the first hit, so a missing key must search everywhere. A Bloom filter per file, about 10 bits per key in memory, says "definitely not here" for 99% of files that lack the key, and the index block narrows each remaining file to a single block read.

</div>
