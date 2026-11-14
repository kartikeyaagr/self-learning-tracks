---
id: "07.06"
course: "system-design-101"
module: "07"
slug: "immutable-versioned-data-files"
title: "Immutable versioned data files"
keyIdeas: ["Never modify data files after writing them. Every change writes new files, and a small metadata pointer says which set of files is the current version.", "Benefits: readers never see half-written data, many readers need no locks, caching is trivial (a file never changes), and you get time travel and rollback by pointing to an older version.", "Writers commit by atomically swapping the pointer (compare-and-swap on a metadata file or catalog entry). Conflicting writers retry.", "Old versions are garbage-collected later.", "Used by LSM SSTables, Git, Apache Iceberg, Delta Lake and Hudi."]
question: "Why does never modifying a file in place make concurrency and rollback easier?"
answer: "If files never change, a reader can never see a partially written file or a mix of old and new rows; it simply reads the set of files listed in the version it started with, so it needs no locks and gets snapshot isolation for free. Writers do their heavy work without locks and coordinate only on one tiny atomic commit (put-if-absent of the next log entry, or compare-and-swap of a metadata pointer), retrying if another writer won and checking whether its changes conflict. Every old version is still a complete list of files that still exist, so rolling back means publishing an earlier file list, which takes seconds and copies nothing, as long as a vacuum has not yet deleted those older files."
sources: [{"title": "Apache Iceberg spec", "url": "https://iceberg.apache.org/spec/", "note": "Immutable data files plus versioned metadata snapshots."}, {"title": "Delta Lake: transaction log", "url": "https://github.com/delta-io/delta/blob/master/PROTOCOL.md", "note": "Versioned tables on immutable files."}]
---

## Never edit, always add

Most of us learned databases as things that change rows in place: `UPDATE accounts SET balance = 90 WHERE id = 7` finds the row and overwrites it. Large analytics tables on object storage work the other way round. Table formats such as **Apache Iceberg**, **Delta Lake** and **Apache Hudi** never modify a data file once it has been written. To change data, they:

1. Write **new** data files containing the new or changed rows.
2. Atomically publish a new **version** (also called a snapshot) of the table: a small piece of metadata listing exactly which data files make up the table now.

Old data files are left untouched, and old versions keep pointing at them. This lesson explains how that works, why it makes concurrency and rollback almost trivial, and what it costs.

There's a practical reason this design took over data lakes: object stores don't let you modify part of an object. You can only write a whole new object or replace one entirely. But the benefits go well beyond that constraint, and the same principle appears in LSM trees (07.09), the stream layer of a cloud store (07.18), Git, and container images.

## Delta Lake's transaction log

Delta Lake keeps a directory called `_delta_log` next to the table's data files. That directory **is** the table's history.

~~~dg-tree The layout of a Delta table. Data files never change; the log directory records which of them belong to each version.
sales/ | the table
  part-0001.parquet | data file, immutable
  part-0002.parquet
  part-0003.parquet
  _delta_log/
    00000000000000000000.json | version 0
    00000000000000000001.json | version 1
    00000000000000000010.checkpoint.parquet | state at v10
    *00000000000000000011.json | latest commit
    _last_checkpoint | points at v10
~~~

Each commit is a JSON file named by a zero-padded 20-digit version number. It contains a list of **actions**:

- `add`: this data file is now part of the table (with its path, size, partition values and column statistics).
- `remove`: this data file is no longer part of the table. The file itself stays on storage; it just isn't in the new version.
- `metaData`: the schema or table settings changed.
- `commitInfo`: who made the change and what operation it was.

The current state of the table is computed by **replaying** the log in order: start with nothing, apply each version's adds and removes, and you end with the set of live files. Replaying thousands of small JSON files would be slow, so every so often (by default every 10 commits) Delta writes a **checkpoint**: a Parquet file summarising the full state at that version. A reader finds the latest checkpoint via `_last_checkpoint`, loads it, and replays only the few JSON commits after it.

~~~dg-timeline Four commits to a Delta table and the set of live data files after each. Old files stay on storage after they are removed from the table.
v0 | add A, B: table is A, B
v1 | add C (new rows): table is A, B, C
v2 | remove A, add D (rows in A updated): table is B, C, D
v3 | *remove C (bad load rolled back): table is B, D
Later | VACUUM deletes A and C once older than 7 days
~~~

Notice how an **update** works at v2. Some rows in file A changed. Delta cannot edit A, so it writes D, a full copy of A with those rows changed, then commits "remove A, add D" as one version. This is called **copy-on-write**: the cost of changing one row is rewriting the whole file that contains it.

## How a commit stays atomic

The whole design rests on one property: creating the next version file is atomic and **only one writer can create a given version**. A writer that wants to commit version 11 tries to create `00000000000000000011.json` with a "create only if it doesn't exist" operation (put-if-absent). If two writers race, one succeeds and the other fails.

This is **optimistic concurrency control**: writers do all their work (writing new data files, possibly gigabytes) without taking any lock, and only check for conflicts at the tiny commit step.

~~~dg-seq Two writers race to commit version 11. The loser checks whether the winner's commit conflicts with its own work, then retries as version 12.
Writer A > Log : latest is v10
Writer B > Log : latest is v10
Writer A > Storage : write data files
Writer B > Storage : write data files
Writer A > Log : create 11.json
Log -> Writer A : committed v11
Writer B > Log : create 11.json
Log -> Writer B : already exists
Writer B > Log : read v11: conflict?
Writer B > Log : no: create 12.json
Log -> Writer B : committed v12
~~~

What counts as a conflict? If writer A removed a file that writer B read and rewrote, B's commit would resurrect stale data, so B must abort and redo its work. If A only appended new files to a different partition, B can simply retry at the next version number. Its data files are already written, so the retry is cheap.

On HDFS and Azure storage, atomic create-if-absent was always available. On S3 it historically wasn't, so multi-writer Delta tables on S3 needed an external coordinator (typically a DynamoDB table) to serialise commits. S3 later added conditional writes (`If-None-Match: *`), which provide exactly this primitive.

## Iceberg: the same idea, a different tree

Iceberg arranges metadata as a tree rather than a flat log:

- A **catalog** holds one pointer per table: the location of the current metadata file.
- The **metadata file** (JSON) lists the schema, partitioning and the table's snapshots.
- Each **snapshot** points to a **manifest list**, which points to **manifest files**, each listing a batch of data files with their statistics.

A commit writes new data files, new manifests, a new manifest list and a new metadata file, all as new objects, and then performs an atomic **compare-and-swap** on the catalog pointer: "change the pointer from metadata v41 to v42, but only if it is still v41". If someone else committed first, the swap fails and the writer retries. Iceberg's guarantees match Delta's: reads use a committed snapshot and are never affected by concurrent writes, and a write is either fully visible or not at all.

Because rewriting a whole 512 MB file to delete one row is expensive, Iceberg (from format version 2) also supports **merge-on-read**: instead of rewriting the data file, it writes a small **delete file** that lists the rows to ignore (by file and position, or by key values). Readers apply the delete files as they read. Writes are cheap; reads get slower as delete files accumulate, until a compaction job rewrites the data files and drops them.

~~~dg-compare Two ways an immutable table handles an update to a few rows. Both keep old files intact; they differ in who pays.
[Copy-on-write]
Rewrite whole file -(commit)-> remove old, add new
+ Reads stay simple and fast
- One changed row rewrites 512 MB
- Slow for frequent small updates
[Merge-on-read]
Write delete file -(commit)-> add delete file
+ Updates are small and quick
- Every read must apply deletes
- Needs periodic compaction
~~~

## Why immutability helps so much

- **Readers need no locks.** A reader picks a version when it starts and reads exactly the files listed in it. Those files never change, so the reader can't see a half-written file or a mix of old and new rows. This is **snapshot isolation**, and it costs nothing at read time. A query that runs for two hours sees a perfectly consistent table even while hundreds of commits happen around it.
- **Writers barely coordinate.** All the heavy work, writing gigabytes of data, happens in parallel with no locks. Writers only race on the commit, a write of a few kilobytes.
- **Rollback is instant.** Every old version is still a complete, valid list of files. Undo means committing a new version that restores an older file list (Delta's `RESTORE`, Iceberg's rollback), which takes seconds whatever the table's size.
- **Time travel.** You can query the table "as of version 411" or "as of yesterday 09:00" to reproduce a report, debug a pipeline or train a model on exactly the data it saw before.
- **Caching and replication are trivial.** An immutable file can be cached anywhere, forever, with no invalidation. Copying a table to another region means copying files that never change, then the metadata.
- **Auditing.** The log is a record of who changed what and when.

## What it costs

- **Storage grows** until old versions are cleaned up. Delta's `VACUUM` deletes files no longer referenced by any version within the retention period (7 days by default); Iceberg expires old snapshots and then deletes orphaned files. Clean too aggressively and long-running queries or time travel break, because files they need disappear.
- **Write amplification for small changes.** Copy-on-write rewrites whole files; merge-on-read pushes the cost onto readers.
- **Small files.** Frequent small commits (a streaming job committing every 10 seconds) create thousands of tiny files and a long log. Background **compaction** (also called "optimize") rewrites them into fewer large files, committed as one more version.
- **Write conflicts under contention.** Optimistic concurrency works well when commits rarely touch the same data. Many writers hammering the same partition will retry a lot.

<div class="ex">
<div class="lab">Worked example</div>

**a bad backfill**

A job rewrites last month's sales at version 412 with a bug that doubles every amount. It removed 300 files and added 300 new ones. Analysts notice an hour later, by which time versions 413 to 418 have also been committed by unrelated hourly loads.

Because version 411 still lists the 300 original files, and VACUUM won't delete them for 7 days, the team commits version 419, which removes the 300 bad files and re-adds the 300 originals listed at version 411, leaving the later hourly loads in place. (A full table `RESTORE` to 411 would also undo those loads.) It completes in seconds and copies no data. Nothing had been overwritten, so nothing had to be recovered from backup. Had VACUUM retention been 0 hours, the originals would have been gone.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing the cost of copy-on-write**

A 2 TB table is stored as 4,000 files of 512 MB. A daily job updates 50,000 rows scattered randomly across the table. With copy-on-write, those rows likely touch most of the 4,000 files, so the job rewrites close to 2 TB a day to change a few megabytes. With merge-on-read it writes perhaps a few hundred small delete files, a few megabytes, and a weekly compaction rewrites the affected files once. This is why update-heavy tables use merge-on-read, and append-only tables don't need either.

</div>

## In an interview

If a design includes a data lake or historical tables, say "data files are immutable; each commit writes new files and atomically publishes a new snapshot through a put-if-absent or compare-and-swap on the metadata pointer". Then cash it in: readers get consistent snapshots without locks, rollback is a pointer change, and the cost is storage until vacuum plus compaction of small files.

<div class="take">
<div class="lab">Key takeaway</div>

write new files, then publish a new version with one atomic, conditional commit. Readers get consistent snapshots without locks, writers only coordinate on a tiny commit, and rollback or time travel is just choosing an older version, until a vacuum deletes the files it needs.

</div>
