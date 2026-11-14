---
id: "13.09"
course: "system-design-101"
module: "13"
slug: "remote-file-sync-design"
title: "Remote file sync design"
keyIdeas: ["File sync (Dropbox, Google Drive) keeps files identical across devices and the cloud.", "Split files into blocks, hash each block, and upload only blocks the server doesn't already have. A 1-byte edit to a 1 GB file uploads just the changed block(s).", "Keep metadata (file path, versions, list of block hashes) in a database, and block contents in object storage.", "Clients get change notifications (long poll or push) and download only new blocks.", "Conflicts (two devices edit offline) are usually kept as separate \"conflicted copy\" files rather than merged."]
question: "A user edits 1 byte of a 1 GB file. How much should be uploaded?"
answer: "Roughly one block, not the file. Split the file into fixed blocks (Dropbox uses 4 MB) identified by SHA-256 hashes, so the file is a blocklist of 256 hashes. After the edit the client re-hashes and commits the new blocklist; only the block containing the changed byte has a new hash, so the server replies that it needs just that block, and the client uploads about 4 MB plus a blocklist of about 8 KB, around 0.4% of the file; other devices likewise download only that block. An rsync-style delta inside the block can cut it to a few kilobytes, and content-defined chunking keeps the cost small even when the edit inserts bytes and would otherwise shift every later fixed block boundary."
sources: [{"title": "Dropbox: Streaming File Synchronization", "url": "https://dropbox.tech/infrastructure/streaming-file-synchronization", "note": "Blocks, hashes, sync."}, {"title": "ByteByteGo: Design Google Drive", "url": "https://bytebytego.com/courses/system-design-interview/design-google-drive", "note": "Interview version."}]
---

## The goal

A **file sync service** (Dropbox, Google Drive, OneDrive, iCloud Drive) keeps a folder identical across a user's devices and the people they share it with. Change a file on your laptop, and within seconds the new version is on your phone, your desktop at work and your colleague's machine.

The core efficiency requirement is captured by one scenario: a user changes **1 byte** in a **1 GB** file. Uploading the whole gigabyte again would take over 13 minutes on a 10 Mbps uplink, burn mobile data, and multiply across every device that then downloads it. The ideal is to transfer roughly **the size of the change plus a little metadata**, both up from the editing device and down to every other device.

This lesson builds the design that achieves that, modelled on Dropbox's architecture, and then fills in the rest of a sync system around it.

## The block model

The foundation is to stop treating a file as one indivisible blob.

- Every file is split into **blocks** of a fixed size; Dropbox uses **4 MB** (the last block may be smaller).
- Each block is identified by the **SHA-256 hash** of its contents: a 32-byte fingerprint that, in practice, is unique to that content.
- A file is represented by its **blocklist**: the ordered list of its blocks' hashes.
- **Only blocks the server does not already have are uploaded.**

Hashing is what makes this work. Two blocks with the same hash have the same content, so the server can tell whether it already has a block without seeing it, just by looking up the hash. Lesson 13.10 explains this **content addressing** in depth.

### Two kinds of server

The design splits storage into two services with very different jobs:

- **Block servers** (the block store): a giant key-value store mapping **hash → block contents** (encrypted at rest). They know nothing about users, files, names or folders. They just store and return blocks by hash. This is where the bytes, petabytes or exabytes of them, live.
- **The metadata server** (metaserver): knows about users, shared folders (**namespaces**), file paths and versions. Its core is the **Server File Journal** (SFJ), an append-only log recording, for each namespace, which blocklist each path has at each point in time. This data is small but must be strongly consistent.

~~~dg-flow The architecture of a block-based sync service. Bytes and metadata take separate paths; notifications tell other devices to pull.
[Upload] Laptop: file watcher -(changed file)-> Hasher: blocklist -(commit)-> *Metaserver and journal
[Upload] Hasher: blocklist -(missing blocks only)-> Block servers
[Download] Metaserver and journal -(change notification)-> Phone -(fetch blocklist, missing blocks)-> Block servers
~~~

## The commit protocol

The upload uses a short conversation in which the server tells the client exactly which blocks it needs.

1. The client computes the file's new blocklist and tries to **commit** it: "path `/designs/logo.psd` now has blocklist h1, h2, ..., h256".
2. The metaserver checks which of those hashes already exist in the block store (and that this user is allowed to reference them). If any are missing, it refuses the commit and answers **"need blocks"** with the list of missing hashes.
3. The client uploads just those blocks directly to the block servers.
4. The client commits again. Now every block exists, so the metaserver appends the new version to the journal and the commit succeeds.

~~~dg-seq The commit protocol. The client proposes a blocklist; the server replies with the blocks it lacks; the client uploads only those, then commits again.
Client > Metaserver : commit(path, base rev 41, blocklist h1..h256)
Metaserver -> Client : need blocks: h137'
Client > Block server : store(h137', 4 MB)
Block server -> Client : ok
Client > Metaserver : commit(path, base rev 41, blocklist h1..h256)
Metaserver > Metaserver : append rev 42 to journal
Metaserver -> Client : ok, rev 42
Metaserver > Other devices : namespace changed
~~~

Other devices are notified that the namespace changed, ask the metaserver for changes since the last journal position they saw, receive the new blocklist, compare it with the blocks they already have locally, and **download only the blocks they lack**. Sync is efficient in both directions.

## The 1-byte edit

Now the question. A 1 GB file split into 4 MB blocks has **256 blocks**. If one byte is **overwritten in place** (the file's length does not change), only the block containing that byte changes, so exactly **one** hash in the blocklist changes.

The client uploads:

- **one 4 MB block**, and
- a blocklist of 256 hashes × 32 bytes = **8 KB**, plus a few hundred bytes of request overhead.

That is about **0.4%** of the file. Every other device downloads the same 4 MB.

<div class="ex">
<div class="lab">Worked example</div>

**the 1-byte edit traced**

A user opens `video-project.bin` (1 GB, 256 blocks), changes one byte at offset 571,000,000, and saves.

The client's file watcher reports a change. The client re-reads the file and hashes each 4 MB block: SHA-256 runs at roughly 0.5 to 1 GB per second on a modern CPU, so this takes 1 to 2 seconds. Offset 571,000,000 falls in block 137 (counting from 1: 571,000,000 ÷ 4,194,304 ≈ 136.1, so the 137th block). Only h137 changes, to h137'.

Commit: the metaserver replies "need h137'". The client uploads 4 MB, which takes about 3.4 seconds at 10 Mbps. The second commit succeeds as revision 42. Total upload: 4 MB + 8 KB instead of 1 GB, about 250 times less.

</div>

~~~dg-bars Bytes uploaded after a 1-byte change to a 1 GB file, by sync strategy. Scale is logarithmic.
Whole file | 1000000 | 1 GB
Fixed 4 MB blocks | 4000 | 4 MB + 8 KB list
Smaller 1 MB blocks | 1000 | 1 MB + 32 KB list
Content-defined 64 KB chunks | 64 | about 64 KB + list
*rsync-style delta in block | 2 | a few KB
~~~

### Making it better still

- **Smaller blocks** reduce what a small change costs, at the price of longer blocklists and more metadata and hash lookups. Many designs use 1 to 8 MB.
- **Deltas within a block**: an rsync-style delta (lesson 13.10) can send just the changed bytes inside the changed block, plus enough to rebuild it, so a 1-byte change costs a few kilobytes. The server reconstructs the new block from the old one plus the delta.
- **Insertions are the hard case.** If the edit **inserts** a byte rather than overwriting one, every fixed block boundary after it shifts by one byte, so every later block's hash changes and the rest of the file looks new. **Content-defined chunking** (lesson 13.10) fixes this by placing boundaries where the content dictates, not at fixed positions.

## Streaming sync: overlap upload and download

Originally, sync had two strict phases: the uploading device uploaded every block and committed; only then did other devices hear about the file and start downloading. For a large new file shared with a colleague, the total time was upload time plus download time.

**Streaming sync** lets downloading clients **prefetch blocks of a file that has not been committed yet**, while the upload is still in progress. As soon as the uploader has stored a block, the server can tell interested clients its hash, and they start fetching. By the time the commit lands, they are nearly done. Dropbox calculated up to **2× faster** multi-client sync in theory; in a test with about 1.2 Mbps upload and 5 Mbps download, it measured about **25%** faster in practice (a 500 MB file: 293 s instead of 383 s; 100 MB: 64 s instead of 89 s).

~~~dg-timeline Sharing a new 100 MB file with a colleague. Streaming sync starts the download while the upload is still running.
0 s | Uploader starts sending blocks
10 s | *Streaming: colleague's client begins prefetching stored blocks
60 s | Last block stored, commit succeeds
64 s | Streaming: colleague already has nearly every block, done
89 s | Classic: download only started at 60 s, finishes now
~~~

## The rest of a sync design

The block model is the heart, but a complete design needs the parts around it.

### Detecting changes

- **Local changes** are detected with operating-system file events (FSEvents on macOS, ReadDirectoryChangesW or FileSystemWatcher on Windows, inotify on Linux) and queued. A periodic scan comparing size and modification time catches anything the events missed.
- **Remote changes** arrive by a push notification over a long-lived connection (WebSocket, server-sent events or long polling, lesson 11.02), with periodic polling (`GET /changes?since=<cursor>`) as a safety net. The notification only says "something changed"; the client then fetches changes since its **cursor** (its position in the journal), which makes missed notifications harmless.

### Uploads and downloads

- Large blocks or files go **directly to blob storage** with presigned URLs and multipart upload, supporting files up to tens of gigabytes (lessons 13.01 and 13.04). Per-chunk status in the file's metadata lets uploads resume.
- **Fingerprints** (SHA-256) for each chunk and the whole file enable deduplication and resumability, and let the client verify every downloaded block.
- **Compression** on the client before upload (and before encryption, since encrypted data does not compress), skipped for already-compressed media such as JPEG and MP4.
- **Downloads** come from storage or a CDN with short-lived signed URLs (lesson 13.07).

### Conflicts

Two devices may edit the same file while one is offline. Each commit carries the **base revision** it was edited from. If the server's current revision is newer than the base (someone else committed in between), the commit is a **conflict**.

- The simplest policy is **last write wins**, which silently loses one person's work.
- Real clients usually **keep both**: the server keeps the version that arrived first, and the second device saves its version as a separate file, such as `report (Ana's conflicted copy 2026-11-12).docx`, and syncs that. No data is lost, and a human resolves it.

~~~dg-compare Two policies for concurrent edits to the same file while one device was offline.
[Last write wins]
Ana commits rev 42 -> Ben commits on base 41 -> Ben overwrites Ana
+ Simple, one file
- Ana's work silently lost
[*Keep both]
Ana commits rev 42 -> Ben commits on base 41 -> Conflict detected -> Ben's copy saved beside it
+ Nothing is ever lost
+ Base revision makes conflicts detectable
- A human must merge the two copies
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a designer saves a 1 GB Photoshop file**

The client re-hashes the file and finds blocks 37 and 38 changed (the edit touched a 6 MB region spanning a block boundary). Commit returns "need h37', h38'"; 8 MB is uploaded and the commit succeeds as revision 18.

The designer's desktop at home gets a notification, fetches changes since its cursor, receives the new blocklist, sees it already has 254 of the 256 blocks, and downloads 8 MB. With streaming sync, it had started downloading those 8 MB before the commit finished.

Meanwhile the designer's laptop, offline on a train, had also edited the file from revision 17. When it reconnects, its commit says "base rev 17", the server is at 18, and the laptop saves its version as a conflicted copy. Both versions survive.

</div>

## Pitfalls

- **Uploading whole files.** It works in a demo and fails users on slow connections with large files.
- **Fixed blocks with insertions.** A design that only handles overwrites well. Mention content-defined chunking.
- **Trusting notifications as data.** Notifications get lost; the cursor-based changes API is the source of truth.
- **Last write wins by default.** Fine for some data, unacceptable for people's documents.
- **Identifying files by path.** Renames and moves look like delete plus create, which breaks history and sharing (lesson 13.11 returns to this).
- **Hashing everything on every event.** Check size and modification time first, and re-hash only files that may have changed.

## In an interview

Answer the question with the arithmetic: split the file into blocks identified by content hashes, so a 1-byte overwrite in a 1 GB file changes one 4 MB block and the client uploads that block plus an 8 KB blocklist. Then describe the commit protocol ("need blocks"), the split between block servers and the metadata journal, how other devices download only missing blocks, and the refinements: deltas within blocks, content-defined chunking for insertions, streaming sync, conflict handling with base revisions, and cursor-based change notification.

<div class="take">
<div class="lab">Key takeaway</div>

split files into blocks identified by content hashes; a file version is a list of hashes. The client commits the list, the server asks only for blocks it lacks, and other devices download only blocks they lack, guided by a cursor into the metadata journal. A 1-byte overwrite in a 1 GB file uploads one block (about 4 MB) plus an 8 KB blocklist; deltas shrink that further, content-defined chunking keeps insertions cheap, and streaming sync overlaps upload and download.

</div>
