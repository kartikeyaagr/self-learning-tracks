---
id: "07.08"
course: "system-design-101"
module: "07"
slug: "bitcask-storage-engine"
title: "Bitcask storage engine"
keyIdeas: ["Bitcask (from Riak) is a log-structured store with an in-memory hash index.", "Writes append to the active data file. The in-memory \"keydir\" maps each key to file ID, offset and size of its latest value.", "A read is one hash lookup and one disk seek. Very fast and predictable.", "Deletes append a tombstone. Merge (compaction) rewrites live entries into new files and drops old ones. Hint files speed up rebuilding the keydir on restart.", "The limit: every key must fit in RAM. No range queries, since the hash index is unordered."]
question: "Bitcask keeps every key in RAM. What does that limit, and how does it handle deletes?"
answer: "Holding every key in the in-memory keydir limits the total number of keys (not the data volume) to what RAM can hold, at roughly the key size plus a few tens of bytes per key, so a node with 200 million 40-byte keys needs about 14 GB just for the index; and because the keydir is a hash table there is no key order, so no range scans or prefix listings. A delete appends a tombstone record and removes the key from the keydir, so later reads miss immediately; the background merge then copies only records the keydir still points at into new files, drops dead values and tombstones, deletes the old files and writes hint files of keys and offsets so the keydir can be rebuilt quickly on restart."
sources: [{"title": "Bitcask paper", "url": "https://riak.com/assets/bitcask-intro.pdf", "note": "Hash index in memory, log on disk. 6 pages."}]
---

## The simplest real log-structured engine

Bitcask was written at Basho as a storage backend for the Riak key-value database. It is worth studying because it is the log-structured idea from lesson 07.07 in its most direct form, with almost nothing added: an append-only log on disk plus a hash table in memory. It is small enough to understand completely, it was genuinely used in production, and its one big limitation teaches you exactly why LSM trees exist.

Its design goals were modest and clear: low latency for every read and write, high write throughput, datasets much larger than RAM (for values, at least), crash friendliness, easy backup, and simple code.

## The pieces

### Data files

A Bitcask is a **directory** of data files. At any moment exactly one file is **active**: it is open for appends. When it reaches a size limit (for example 2 GB) it is closed and becomes **immutable**, and a new active file is opened. Only one process (the writer) may write to a Bitcask directory at a time; it holds a lock. Any number of readers can read.

Every write, whether an insert or an update, appends one record to the active file:

~~~dg-layers One Bitcask record on disk. The key and value are stored together, and the CRC detects a torn or corrupted record.
*CRC | 4 bytes: checksum over the rest of the record
Timestamp | 4 bytes: when it was written
Key size | 2 bytes
Value size | 4 bytes
Key | variable
Value | variable, can be large
~~~

There is no in-place update. Setting a key that already exists simply appends a newer record; the old one stays in its file, now dead.

### The keydir

The clever part is in memory. The **keydir** is a hash table with one entry for **every live key**, mapping it to:

- the **file id** where the latest record lives,
- the **value position** (byte offset) within that file,
- the **value size**, and
- the **timestamp**.

A write appends to the active file, then updates the keydir entry to point at the new record. The write is now complete (subject to the fsync policy you choose: every write, on an interval, or left to the OS).

~~~dg-seq A Bitcask get: one hash lookup in RAM gives the exact location, then one disk read fetches the value.
Client > Keydir (RAM) : get("user:42")
Keydir (RAM) -> Client : file 7, offset 81,920, size 512
Client > Data file 7 : pread 512 bytes at 81,920
Data file 7 -> Client : value (check CRC)
~~~

### Why reads are so predictable

A read is: one hash lookup in memory (microseconds), then **one** positioned read from disk at a known offset and length. That's at most one seek per read, whatever the dataset size. If the file's pages are already in the OS page cache, it costs no disk I/O at all. There is no tree to walk, no multiple files to check, no read amplification. Writes are pure sequential appends. Bitcask's authors reported 5,000 to 6,000 writes per second on a laptop of the time, with low, steady latency.

<div class="ex">
<div class="lab">Worked example</div>

**tracing a write then a read**

`put("cart:991", 300 bytes)` arrives. Bitcask appends a record of 4 + 4 + 2 + 4 + 8 + 300 = 322 bytes at offset 1,048,000 of active file 12, and sets keydir["cart:991"] = (file 12, value at 1,048,022, size 300, ts). Total cost: one memory update and a 322-byte sequential append.

A moment later `get("cart:991")` looks up the keydir, then reads 300 bytes at offset 1,048,022 of file 12. Those bytes were just written, so they are almost certainly still in the page cache: the read costs no disk I/O.

</div>

## Deletes, garbage and merging

### Deletes

A delete appends a special **tombstone** record for the key and removes the key from the keydir. From then on reads miss in the keydir and return "not found" immediately without touching disk. The tombstone matters for recovery: if the keydir is rebuilt from the files, the tombstone tells the rebuild that the older value for that key is dead.

### Merge

Immutable files accumulate dead records: overwritten values and deleted keys. Left alone, the directory grows forever. Bitcask's compaction process is called **merge**:

1. Pick some or all of the immutable files (never the active one).
2. Read them record by record. For each record, ask the keydir: is this the latest version of this key? Only if the keydir points at exactly this file and offset is the record live.
3. Copy live records to new merged files. Skip dead ones and tombstones whose older values are being merged away too.
4. Update the keydir entries to point at the new locations, then delete the old files.

Merges are triggered by thresholds such as the fraction of dead bytes in a file (fragmentation) or the total amount of dead data, and can be restricted to quiet hours. The writer keeps appending to the active file during a merge, so foreground traffic is not blocked, though the merge does compete for disk bandwidth.

~~~dg-timeline One key's life in Bitcask: overwrites leave dead records behind until a merge copies only the latest version and writes a hint file.
09:00 | put v1: appended to file 3, keydir points at file 3
09:30 | put v2: appended to file 4, v1 now dead
10:00 | File 4 full: closed, file 5 becomes active
02:00 next day | *Merge reads files 3 and 4, copies only v2 to merged file 9
02:05 | Hint file 9 written, files 3 and 4 deleted
~~~

### Hint files and fast restart

The keydir lives only in memory, so after a restart it must be rebuilt. The slow way is to scan every data file from start to end, reading every key and value. For hundreds of gigabytes that takes a long time. So alongside each merged file, Bitcask writes a **hint file**: a compact list of `(timestamp, key size, value size, value position, key)` for each record, **without the values**. On startup, Bitcask reads the hint files (small) and only scans data files that have no hint file (typically just the recent ones). Restart time then depends on the number of keys, not the volume of values.

~~~dg-tree A Bitcask directory. Only the active file takes writes; merged files come with hint files for fast restart.
bitcask/ | one writer holds the lock
  9.bitcask.data | merged, immutable
  9.bitcask.hint | keys and offsets only
  10.bitcask.data | merged, immutable
  10.bitcask.hint
  11.bitcask.data | closed, not yet merged
  *12.bitcask.data | active: appends go here
~~~

### Crash recovery

Only the active file can be damaged by a crash, because closed files are never written again. On startup, Bitcask scans the active file and verifies each record's CRC. A torn last record fails its CRC and is ignored (lesson 07.03's torn-tail technique). Everything else is intact by construction. Backups are equally simple: closed files never change, so you can copy them with ordinary tools while the database runs.

## The limit: every key in RAM

The design's strength is also its ceiling. Because the keydir holds an entry for **every** key, the number of keys is bounded by memory. Values can be huge and live on disk, but keys can't. Each keydir entry costs the key itself plus fixed overhead: the hash table slot, file id, offset, size, timestamp and allocator overhead, typically a few tens of bytes per key depending on the implementation.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a Bitcask node**

A cache stores 200 million keys averaging 40 bytes, with about 30 bytes of keydir overhead per entry: 200M × 70 bytes = 14 GB of RAM just for the keydir, regardless of value size. A 32 GB node fits it, with room for the OS page cache.

Now the product team doubles the key count to 400 million: 28 GB of keydir on a 32 GB node leaves almost nothing for the page cache, so most reads now go to disk, and growth beyond that is impossible. The options are to shard across more nodes (Riak spreads keys across many Bitcask instances anyway), shorten keys (hash long keys to 16 bytes), or move to an engine that doesn't keep every key in memory.

</div>

~~~dg-bars Keydir memory for 200 million keys as the average key size grows, with 30 bytes of overhead per entry.
16-byte keys | 9.2 | 9.2 GB
40-byte keys | 14 | 14 GB
*100-byte keys | 26 | 26 GB
200-byte keys | 46 | 46 GB
~~~

The second limit: the keydir is a **hash table**, so keys have no order. There is no efficient range scan ("all keys from `order:1000` to `order:2000`") and no prefix listing. You'd have to walk every key. If you need ordered access, you need a sorted structure, which is exactly what the LSM tree's memtable and SSTables provide.

## When Bitcask fits

| Good fit | Poor fit |
| --- | --- |
| Key count fits comfortably in RAM | Billions of small keys |
| Point reads and writes only | Range scans or prefix queries |
| Values from bytes to megabytes | Tiny values where key overhead dominates |
| Need predictable single-seek reads | Need secondary indexes |
| Frequent overwrites, merge in quiet hours | Disk space very tight |

The same "index in RAM, data in append-only files" pattern shows up elsewhere: Facebook's Haystack photo store keeps a small in-memory index of where every photo sits in huge volume files (lesson 07.13), and many caches and queue engines follow it too.

## Common mistakes

- **Forgetting the key count is the real capacity limit.** Disk space looks plentiful while RAM quietly runs out.
- **No merge schedule.** Dead data accumulates and disk fills; or merges run at peak and compete with traffic.
- **Expecting range scans.** A hash index cannot provide them.
- **Never fsyncing.** With the OS deciding when to flush, a power cut can lose recent writes even though the format itself is crash-safe.

## In an interview

Bitcask is a good answer when the prompt is "a key-value store with very fast point reads and writes and a bounded number of keys", for example a session store or a cache with persistence. Describe it in one breath: "append-only data files, an in-memory hash from key to file and offset, one disk read per get, tombstones for deletes, a background merge that keeps only live records and writes hint files for fast restart." Then name the limit before the interviewer does: every key must fit in RAM and there are no range queries.

<div class="take">
<div class="lab">Key takeaway</div>

Bitcask gives one-seek reads and append-only writes by keeping every key in an in-memory hash table pointing into immutable log files. Deletes are tombstones, merging reclaims dead records and writes hint files for fast restarts, and the price is that RAM bounds the number of keys and there are no range queries.

</div>
