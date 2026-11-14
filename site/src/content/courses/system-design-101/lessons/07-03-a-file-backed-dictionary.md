---
id: "07.03"
course: "system-design-101"
module: "07"
slug: "a-file-backed-dictionary"
title: "A file-backed dictionary"
keyIdeas: ["The simplest persistent store: keep a dictionary in memory and save it to a file.", "Naive overwrite is dangerous: a crash mid-write leaves a half-written, corrupt file.", "Safe update: write the whole new content to a temp file, fsync it, rename it over the old file (rename is atomic on POSIX), then fsync the directory.", "Rewriting the whole file on each change doesn't scale; next steps are an append-only log of changes, then an index into that log.", "Recovery on startup: load the file (or replay the log) into memory."]
question: "How do you update a file so a crash never leaves it half-written?"
answer: "Never modify the live file in place. Write the complete new version to a temporary file in the same directory, fsync it, atomically rename it over the original (POSIX rename within one filesystem is atomic, so readers see the old or new file, never a mix), then fsync the directory so the rename itself is durable; a crash at any point leaves either the full old file or the full new one, plus at most a stray temp file. For frequent small updates, append records framed with a length and a checksum to a log, acknowledge only after the covering fsync, and on recovery replay until the first record that fails its checksum and truncate that torn tail."
sources: [{"title": "Build Your Own Database: files and durability", "url": "https://build-your-own.org/database/", "note": "Free Part I covers atomic file updates."}, {"title": "Let's Build a Simple Database", "url": "https://cstack.github.io/db_tutorial/", "note": "A SQLite clone in C, step by step."}]
---

## The problem

The simplest possible database is a dictionary (a map from keys to values) saved to a file. Your program loads the file at startup, keeps the dictionary in memory, and writes it back whenever something changes. Many real tools start exactly like this: config files, desktop app state, small caches, a CLI's list of saved credentials.

The question this lesson answers is deceptively hard: **how do you update that file so that a crash, at any instant, never leaves it half-written?** The answer introduces three ideas that every later engine in this module reuses: write somewhere new then switch atomically, append instead of overwrite, and detect damage with checksums.

## The naive version and how it breaks

The obvious code opens the file for writing (which truncates it to zero length), writes the new contents and closes it. Consider where a crash can land:

- **After truncating, before writing.** The file is now empty. Every setting is lost, including the ones that weren't changing.
- **In the middle of writing.** The file holds the first part of the new contents and nothing after it, or (if the filesystem reuses blocks) a mix of new and old bytes. JSON or binary data in that state is usually unreadable.
- **After writing, before the OS flushes.** As lesson 07.01 explained, `write()` only fills the page cache. Without `fsync`, a power failure can leave the file empty or partly written even though your program finished.
- **A concurrent reader** (another process, a backup job) can open the file while it is half written and read garbage, even without any crash.

The core mistake is **modifying the only copy**. During the update, there is a window in which neither the old version nor the new version exists in full on disk.

## The rename trick

The standard fix is to never touch the live file. Build the complete new version beside it, make sure it is on disk, then swap it in with one atomic operation.

1. Write the full new contents to a temporary file **in the same directory**, for example `data.tmp`.
2. Call `fsync` on the temporary file, so its bytes are on stable storage.
3. Call `rename("data.tmp", "data")`. On POSIX filesystems, rename within one filesystem is **atomic**: any process that opens `data` sees either the complete old file or the complete new file, never a mix, and there is no moment when `data` does not exist.
4. Call `fsync` on the **directory**. The rename changed the directory's list of names, and that change is itself only in memory until the directory is synced. Skip this and a crash can bring back the old name after you told the user the save succeeded.

~~~dg-seq The rename trick: readers keep seeing the old file until one atomic rename swaps in the complete new one.
App > Disk : write data.tmp (full new contents)
App > Disk : fsync(data.tmp)
Disk -> App : temp file durable
App > Disk : rename(data.tmp, data)
App > Disk : fsync(directory)
Disk -> App : new name durable
~~~

Why "same directory"? Because rename is only atomic within one filesystem. If the temp file lives in `/tmp` and that is a different filesystem (often a RAM-backed one), rename fails or the library falls back to copy-then-delete, which reintroduces the half-written window.

### Walking every crash point

The test of a crash-safety design is to stop the world at each step and ask what you would find on reboot.

~~~dg-timeline What a reboot finds if power fails at each point of the rename trick. Every outcome is either the full old file or the full new one.
Before write | data = old version, no temp file
During temp write | data = old, data.tmp partial: delete it
After fsync(tmp) | data = old, data.tmp complete: delete or use it
After rename, before dir fsync | data = old or new, both complete
After dir fsync | *data = new version, durable
~~~

At no point is `data` partial. The worst case is a stray `data.tmp`, which startup code simply deletes. This pattern is used everywhere: editors saving files, package managers installing, Git updating refs (it writes `ref.lock` then renames it), and databases writing new manifest files.

<div class="ex">
<div class="lab">Worked example</div>

**saving settings safely**

An app saves `settings.json` (4 KB). It writes `settings.json.tmp`, calls fsync (about 2 ms on an SSD), renames it to `settings.json`, then fsyncs the directory (another 1 to 2 ms). The save costs about 4 ms.

Power fails 1 ms after the rename. On reboot, the directory change may or may not have reached disk, so the app finds either the old or the new `settings.json`, both complete, and perhaps a leftover `.tmp` it deletes. The user may lose the very last change, but never all their settings, and the app was not yet claiming the save had succeeded, because it acknowledges only after the directory fsync.

</div>

### The cost: rewriting everything

The rename trick makes each update atomic, but it rewrites the whole file every time. Changing one key in a 1 GB dictionary means writing 1 GB, plus two fsyncs. At 500 MB/s that is two seconds per update. That is fine for a settings file and hopeless for a database. To go further you need updates whose cost is proportional to the change, not to the file.

## Append-only logs

The next step is to stop rewriting and start **appending**. Each update becomes a small record added to the end of a log file: `set user:42 = {...}` or `delete user:17`. To rebuild the dictionary, you start empty and replay every record in order; later records override earlier ones.

This is a big improvement:

- **Writes are small and sequential.** An update costs one append, regardless of how big the dictionary is.
- **Old data is never touched.** A crash can only damage the record being appended at the end of the file (the tail). Everything before it was complete and fsynced earlier.
- **Batching is natural.** Append many records, fsync once, acknowledge them all (group commit, from lesson 07.01).

### Detecting a torn tail

A crash can still leave a **torn** final record: only some of its bytes reached the disk. Worse, a filesystem may extend the file's length before the data blocks are written, so the tail can be full of zeros or old garbage. The reader must be able to tell a good record from a bad one. So each record is framed:

~~~dg-layers The framing of one log record. The length says where the record ends; the checksum proves its bytes are exactly what was written.
Length | 4 bytes: size of the payload
*CRC32C | 4 bytes: checksum of the payload
Payload | key size, key, value size, value, or a delete marker
~~~

A **checksum** is a short number computed from the bytes (CRC32C is the common choice: fast, often hardware accelerated, and good at catching the burst errors disks produce). On recovery the reader goes record by record: read the length, read that many bytes, recompute the checksum and compare. The first record whose checksum fails, or whose length runs past the end of the file, marks the end of valid data. Everything from there on is discarded and the file is truncated to the last good record.

~~~dg-flow Recovery replays records until the first one whose checksum fails, then truncates the torn tail.
Record 1 OK -> Record 2 OK -> Record 3 OK -> *Record 4 CRC fails -(truncate here)-> Resume appending
~~~

Is it safe to throw that record away? Yes, if the engine follows one rule: **acknowledge a write only after the fsync that covers it returns**. A torn record was by definition never fully fsynced, so no client was told it succeeded. Dropping it loses nothing that was promised.

<div class="ex">
<div class="lab">Worked example</div>

**replaying a log after a crash**

A dictionary log has 1.2 million records, 180 MB. The process is killed while appending record 1,200,001. On restart, the reader scans the file at about 1 GB/s: about 0.2 seconds of reading. Record 1,200,001's length field says 312 bytes but only 90 remain before end of file, so the reader stops, truncates those 90 bytes, and has rebuilt the dictionary from 1,200,000 records.

The client that sent record 1,200,001 never got an acknowledgement, so it retries. Everything that was acknowledged is present.

</div>

### The log grows forever: compaction

The log has a new problem. If a key is updated 10,000 times, the log holds 10,000 records for it, and only the last matters. The file keeps growing, and startup replay gets slower. The fix is **compaction**: periodically write a fresh file containing just the current value of each live key (using the rename trick to swap it in), then delete the old log. Many systems do this as a **snapshot plus log**: a compact snapshot of the whole state, and a short log of changes made since. Recovery loads the snapshot then replays the short log. Redis does exactly this with its RDB snapshots and append-only file, and the same idea, refined, becomes Bitcask (07.08) and the LSM tree (07.09).

~~~dg-compare Three ways to persist a dictionary. Each step fixes the previous one's flaw and introduces the next problem to solve.
[Overwrite in place]
Truncate, write, close
- Crash leaves empty or partial file
- Readers can see half a file
[Write temp, then rename]
Write tmp -(fsync)-> rename -> fsync dir
+ Always old or new, never mixed
- Rewrites whole file per update
[*Append-only log]
Append record -(group fsync)-> ack
+ Cost per update is tiny
+ Only the tail can tear
- Needs checksums and compaction
~~~

## Building up to a real engine

The path from here to a database is short. Replace "load the whole file into memory" with a **pager**: a component that reads and writes the file in fixed-size pages (say 4 KB) and caches the hot ones. Organise those pages as a B-tree so you can find a key without reading everything. Add a write-ahead log so page updates can be made crash safe. That is essentially SQLite, and every step uses the techniques in this lesson.

## Common mistakes

- **Skipping fsync.** The code works in every test and loses data on the first real power cut.
- **Forgetting the directory fsync.** The rename is not durable until the directory is synced.
- **Temp file on another filesystem.** Rename across filesystems is not atomic.
- **No checksum on log records.** A torn tail is read back as a valid but corrupt record.
- **Acknowledging before fsync.** Then truncating a torn tail can drop a write the client thinks succeeded.
- **Stopping at the first bad record in the middle of a log** without asking why. A bad record followed by good ones is not a torn tail; it is corruption, and should raise an alarm rather than silently discard data after it.

## In an interview

If a design keeps state in files (a coordinator's metadata, a checkpoint, a leader's term number), say how it is updated: "write a temp file, fsync, rename, fsync the directory" for small state, or "an append-only log with length and CRC per record, truncated at the first bad record on recovery" for frequent updates. Mention that writes are acknowledged only after the covering fsync. It is a short sentence that shows you understand durability rather than assume it.

<div class="take">
<div class="lab">Key takeaway</div>

never modify the only copy of data a crash must not corrupt. Either write a complete new file and atomically rename it into place (fsyncing the file and the directory), or append length-and-checksum framed records to a log, acknowledge only after fsync, and on recovery cut the log at the first record that fails its checksum.

</div>
