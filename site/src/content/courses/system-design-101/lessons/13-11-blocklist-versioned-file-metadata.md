---
id: "13.11"
course: "system-design-101"
module: "13"
slug: "blocklist-versioned-file-metadata"
title: "Blocklist versioned file metadata"
keyIdeas: ["A file version can be stored as an ordered list of block hashes (a blocklist).", "New version = new blocklist. Unchanged blocks are shared with older versions, so keeping history costs only the changed blocks.", "Restoring an old version is just reading its blocklist.", "Metadata tables: files (path, latest version), versions (file_id, version, blocklist, size), blocks (hash → storage location, refcount).", "Garbage-collect blocks no version references, carefully, after a delay."]
question: "A file version is a list of block hashes. How does that make version history cheap?"
answer: "Each version is just metadata, an immutable ordered list of block hashes, while the blocks themselves live once in a content-addressed store. A new version that changes a few blocks adds only those blocks plus one small row with a list of hashes; every unchanged block is shared with earlier versions by reference, so keeping hundreds of versions costs little more than the changes themselves (500 edits of a 200 MB file take about 4 GB, not 100 GB). Restoring an old version is a metadata write that points the file at an old list again, viewing one means fetching blocks the device mostly already has, and renames touch no data. The extra work is garbage-collecting blocks that no live version references, with grace periods so uncommitted uploads are not deleted."
sources: [{"title": "Dropbox: Rewriting the heart of our sync engine", "url": "https://dropbox.tech/infrastructure/rewriting-the-heart-of-our-sync-engine", "note": "Nucleus and file metadata."}, {"title": "Hello Interview: Dropbox breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/dropbox", "note": "Blocklists and metadata."}]
---

## A version is a list of hashes

Lessons 13.09 and 13.10 built two ideas. First, a file's content is stored as **blocks** (or content-defined chunks) in a **content-addressed block store**, where each block's key is its SHA-256 hash and identical blocks are stored once. Second, the file itself is represented by its **blocklist**: the ordered list of its block hashes. The metadata service records, for each file, which blocklist is current.

To keep **version history**, make one small change: **never overwrite a blocklist; append a new version**. Each save creates a new row pointing at a new list.

```
file_versions
  file_id | version | blocklist                      | size   | modified_at | author
  f9      | 1       | [h1, h2, h3, ..., h250]        | 1.0 GB | 10:02       | ana
  f9      | 2       | [h1, h2, h3', ..., h250]       | 1.0 GB | 10:47       | ana
  f9      | 3       | [h1, h2, h3', ..., h250, h251] | 1.0 GB | 11:30       | ben
```

Dropbox's metadata server keeps exactly this kind of record in its **Server File Journal**, an append-only log of which blocklist each path has; its block servers store hash → block and know nothing about files, users or versions. That separation is the whole trick: **history lives in metadata, bytes live once in the block store**.

## Why history is cheap

Version 2 differs from version 1 in one block. Storing version 2 costs:

- **one new block** (h3', 4 MB) in the block store, and
- **one new metadata row** with a list of 250 hashes (250 × 32 bytes, about 8 KB).

The other 249 blocks are **shared by reference**: the hashes in both lists point to the very same stored objects. Logically, each version is a **complete snapshot** of the file: you can read version 1 in full at any time. Physically, only the changes were stored. This is the same mechanism as Git's commits and trees (lesson 13.10): every commit is a full snapshot of the repository, but unchanged blobs and subtrees are reused by hash.

~~~dg-flow Two versions share every unchanged block by hash; only the edited block is stored again.
[Version 1] Blocklist v1 -> h1, h2 -> *h3 -> h4 ... h250
[Version 2] Blocklist v2 -> h1, h2 -> *h3-new -> h4 ... h250
[Block store] Stored once each -> h1, h2, h3, h3-new -> h4 ... h250
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a year of edits**

A 200 MB spreadsheet is edited 500 times over a year, each edit changing about 2 blocks (8 MB).

**Full copies**: 500 × 200 MB = 100 GB.

**Blocklists**: the first version's 50 blocks (200 MB), plus 500 × 8 MB of new blocks = 4 GB, total 4.2 GB, plus 500 metadata rows of about 1.6 KB each (50 hashes × 32 bytes), under 1 MB. That is about 24 times less storage.

Restoring the version from March is a single metadata write. When versions older than 180 days expire, a garbage-collection sweep finds about 300 blocks no longer referenced by any live version and deletes them, freeing about 1.2 GB.

</div>

~~~dg-bars Storage for 500 versions of a 200 MB file, each edit changing about 8 MB.
Full copy per version | 100000 | 100 GB
*Blocklists and shared blocks | 4200 | 4.2 GB
Metadata rows alone | 0.8 | under 1 MB
~~~

## What falls out of the model

Once versions are lists of hashes, many features become almost free.

- **Restore an old version.** Make the old blocklist current again by writing a **new** version row that points to it (version 4 = the blocklist of version 1). No data is copied; the blocks are already there. Writing a new row rather than deleting later versions keeps the history honest: you can undo the restore.
- **View or download an old version.** Fetch the blocks in its list, in order. Blocks the device already has (because they are shared with the current version) need not be downloaded.
- **Compare versions.** Diff the two lists to see which regions changed, without reading any file data.
- **Deduplication across files.** Two files with identical regions (a copied file, a document saved under a new name, the same photo in two folders) share blocks automatically.
- **Integrity.** Every downloaded block is checked against its hash.
- **Moves and renames are metadata only.** No block moves or is copied when a file is renamed or a folder of 10,000 files is moved.

### Deduplication across users: careful

If two users upload the same file, a global block store would store it once. That saves a lot (think of popular installers or videos), but it creates an information leak: if uploading a file you do not have completes instantly because the server says "already have all blocks", you have learned that **someone else stored that file**. An attacker can use this to confirm whether a specific document exists in the system, or even guess contents by trying variations. Many systems therefore deduplicate **only within one account or namespace** from the client's point of view, or encrypt blocks with per-user keys, which makes identical content produce different stored blocks. Any server-side dedup across users happens invisibly, without changing what the client sees.

## Garbage collection

Sharing has a cost: **deleting is hard**. When old versions expire (many services keep 30 or 180 days of history, depending on the plan) or files are deleted, you cannot simply delete their blocks, because a block may still be referenced by other versions, other files or other users. You need **garbage collection**: delete a block only when **no live version anywhere** refers to it.

Two classic approaches:

- **Reference counting.** Keep a count per block of how many live blocklist entries refer to it. Increment on commit, decrement when a version expires, delete at zero. Simple in principle, but every commit and every expiry must update counts for many blocks, and a single missed decrement (a crash at the wrong moment) leaks storage forever while a double decrement deletes live data. At large scale the counts themselves become a hot, hard-to-shard dataset.
- **Mark and sweep.** Periodically **mark** every hash reachable from any live version (by scanning all live blocklists), then **sweep**: delete every block that was not marked. Slower and batch-oriented, but robust: it recomputes the truth from scratch each time, so errors do not accumulate.

Large systems often combine them: approximate counts or recent-activity logs for most blocks, with periodic mark and sweep to correct drift.

### The race you must handle

The dangerous moment is a block that has been **uploaded but not yet committed**. In the commit protocol (lesson 13.09), the client uploads blocks the server asked for, then commits. Between those steps, the new block is referenced by no version. A sweep running at that instant would delete it, and the commit would then point at a missing block. A similar race happens when a commit reuses an existing block (dedup) at the same moment the sweep decides that block is unreferenced.

The standard fixes:

- A **grace period**: never delete a block younger than, say, 24 hours, or one that became unreferenced less than a day ago. Uploads and commits finish long before.
- **Two-phase deletion**: the sweep first marks candidates as **tombstoned** (no longer offered for dedup), and deletes them only after a later sweep confirms they are still unreferenced. A commit that references a tombstoned block is told to upload it again.

~~~dg-state A block's life in the store. The grace period and tombstone stop a sweep from deleting a block that a commit is about to use.
[*] -> Uploaded
Uploaded -> Referenced : commit includes its hash
Uploaded -> Tombstoned : never committed, older than grace period
Referenced -> Unreferenced : last version using it expires
Unreferenced -> Referenced : a new commit reuses it
Unreferenced -> Tombstoned : sweep, still unreferenced after grace
Tombstoned -> *Deleted : second sweep confirms
~~~

## Metadata as its own hard problem

The block layer is the easy part: a key-value store of immutable blobs, scaled by hashing keys across machines. The metadata is where the difficulty lives. Dropbox's scale illustrates it: **hundreds of millions of users**, **hundreds of billions of files**, **trillions of file revisions**, **exabytes** of data. Dropbox rewrote its sync engine (the rewrite is called Nucleus, written in Rust) largely because of its **data model**:

- The old engine had **no stable identifiers** for files that survived moves: a move was recorded as a delete plus an add. That made shared folders and concurrent moves error-prone (a file moved on one device while edited on another could vanish or duplicate). The new model gives files and shared folders **globally unique IDs** and is designed so a file is never observed in a transiently duplicated or missing state.
- The old engine was hard to test. The new one is **deterministic** given its inputs and the order of events, so it can be fuzzed with **millions of randomised scenarios a day**, each reproducible from a random seed.

The lesson for your own design: **identify files by a stable file ID, not by path.** Keep the path (or parent folder ID and name) as an attribute that can change. Versions hang off the file ID. Then a rename is one metadata update, a move is one metadata update, and history follows the file wherever it goes.

~~~dg-tree The metadata model. A stable file ID owns the versions; the path is just an attribute, so renames and moves keep history.
Namespace: Ana's folder
  File f9 | name logo.psd, parent folder d2
    Version 1 | blocklist [h1 .. h250], 10:02, ana
    Version 2 | blocklist [h1, h2, h3' .. h250], 10:47
    *Version 3 | current, 11:30, ben
  File f12 | name notes.txt, parent folder d2
    Version 1 | blocklist [h900]
  Folder d2 | name designs, renamed without touching files
~~~

### A simple interview-scale schema

- `files(file_id, namespace_id, parent_id, name, current_version, deleted_at)`
- `file_versions(file_id, version, blocklist, size, created_at, created_by, base_version)`
- `blocks(hash, size, storage_location, created_at)` in the block store's own index.
- A per-namespace **journal** (an increasing sequence number on every change) that devices use as a cursor: "give me everything after 5,812".

For a simpler file-upload service you might start with a single `FileMetadata` record holding `id`, `name`, `size`, `mimeType`, `uploadedBy`, `status`, and a `chunks` list with per-chunk status (for resumable uploads, lesson 13.04). Adding versions is then a matter of keeping previous chunk lists instead of overwriting them.

<div class="ex">
<div class="lab">Worked example</div>

**restore, rename and expire on one file**

Monday: Ana creates `logo.psd` (file f9, 1 GB, version 1). Tuesday: she edits it twice (versions 2 and 3, about 8 MB of new blocks each). Wednesday: she renames it to `logo-final.psd` and moves it to another folder: one update to f9's name and parent; no blocks touched; versions 1 to 3 still attached.

Thursday: she realises Tuesday's edits were wrong and restores version 1. The service writes version 4 with version 1's blocklist. Her other devices see a new version and compare its blocklist with what they hold. Almost every block is shared with version 3, so they fetch only the few original blocks that Tuesday's edits replaced (about 16 MB), or nothing if those are still in a local cache.

Six months later, versions 2 and 3 expire. Their unique blocks (about 16 MB) become unreferenced; after the grace period, two sweeps confirm it and the blocks are deleted. Versions 1 and 4 share all their blocks, so nothing else is freed.

</div>

## Pitfalls

- **Copying data on restore.** A restore should be a pointer change, not a 1 GB copy.
- **Overwriting blocklists in place.** You lose history and break devices that are mid-download of the old version.
- **Deleting blocks eagerly.** Without grace periods and two-phase deletion, a sweep will eventually delete a block a commit was about to use, corrupting a file.
- **Path as identity.** Moves become delete plus create, history breaks, and concurrent moves misbehave.
- **Global dedup visible to clients.** It leaks which files exist in other users' accounts.

## In an interview

Say it in one line: a version is an immutable list of block hashes, and blocks are stored once in a content-addressed store, so a new version costs only its new blocks plus a small row. Then show you know the consequences: restores are pointer writes, renames are metadata, dedup is automatic (with a privacy caveat across users), and the price is garbage collection with reference counts or mark and sweep, protected by grace periods. Finish with stable file IDs rather than paths, which is the lesson Dropbox's rewrite teaches.

<div class="take">
<div class="lab">Key takeaway</div>

store each file version as an immutable, ordered list of content hashes pointing into a deduplicated block store. A new version costs only its new blocks plus a small metadata row, restores are pointer changes, renames and moves touch only metadata, and blocks are garbage-collected (by reference counting or mark and sweep, with grace periods and tombstones) when no live version references them. Identify files by stable IDs rather than paths, as Dropbox's sync-engine rewrite stresses.

</div>
