---
id: "17.03"
course: "system-design-101"
module: "17"
slug: "a-file-sync-service"
title: "A file sync service"
keyIdeas: ["File sync service: client watches the local folder, chunks changed files, hashes blocks, uploads missing blocks, then commits new file metadata (blocklist) to the server.", "Other devices get notified (long poll/WebSocket), fetch metadata changes since their cursor, and download missing blocks.", "Two devices editing the same file offline: the server accepts the first commit; the second client sees its base version is stale and saves its copy as a \"conflicted copy\" file so no work is lost.", "Metadata in a sharded relational database (strong consistency for namespaces); blocks in object storage, deduplicated by hash."]
question: "Two devices edit the same file offline. What happens on sync?"
answer: "When the devices reconnect, each tries to commit its edit stating the version it started from, and the server applies a compare-and-set on the file's version number. The first device's base version is still current, so its commit becomes the new version. The second device's base is now out of date, so its commit updates nothing and the server reports a conflict instead of overwriting. Because a sync service cannot merge arbitrary file formats, the client keeps both: the first edit stays as the file, the second is saved alongside as a \"conflicted copy\", the device downloads the winning version, and both files sync to every device for the user to reconcile. Last write wins would silently lose one edit, device clocks are not used to decide, and version history can still recover older versions."
sources: [{"title": "ByteByteGo: Design Google Drive", "url": "https://bytebytego.com/courses/system-design-interview/design-google-drive", "note": "Sync, conflicts, notifications."}, {"title": "Hello Interview: Dropbox breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/dropbox", "note": "Interview breakdown."}]
tags: ["Design"]
---

## What we are building

A file sync service in the style of Dropbox, Google Drive or OneDrive: a folder on each of your devices that stays the same everywhere. Save a file on your laptop and it appears on your phone and desktop a few seconds later. Edit while on a plane and it syncs when you land. Lessons 13.09 to 13.11 covered the building blocks (the sync model, chunking and content addressing, versioned block lists). This lesson assembles them and then answers the question that every sync system must get right: two devices edit the same file offline, and then both reconnect. What happens?

### Requirements

- Upload, download and share files up to tens of GB.
- **Automatic sync** across all of a user's devices, including after offline periods.
- **Version history**: restore earlier versions for 30 days or more.
- Never silently lose a user's work.
- Scale: say 100 million users, 500 million devices, 1 billion file changes a day.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**sizing**

1 billion file changes a day is about 11,500 per second on average. Most are small (documents, code, notes), but some are large media files.

If the average change uploads 1 MB of new chunk data after deduplication and delta sync, that is 1 PB of new blob data a day. Without delta sync (uploading whole files on every change, average 20 MB), it would be 20 PB a day. Delta sync is not a nice-to-have; it is the difference between a feasible and an absurd system.

Metadata: each file version has a block list of chunk hashes (32 bytes each). A 100 MB file in 4 MB chunks has 25 hashes, under 1 KB. Billions of file versions fit in a sharded metadata database of tens of terabytes.

</div>

## The design in brief

- **Upload**: the client splits files into chunks (fixed 4 MB blocks in Dropbox's design, or content-defined chunks), computes a **SHA-256 fingerprint** of each, and asks the server which chunks it already has. It then uploads only the missing ones, in parallel, straight to blob storage through **presigned URLs**. The server verifies the parts exist before committing.
- **Metadata**: a database (sharded Postgres, MySQL or DynamoDB) holds each file's ID, path, size, owner, **current version number**, and for each version its **block list** (the ordered list of chunk hashes).
- **Local to remote**: a client agent watches the file system (inotify on Linux, FSEvents on macOS) and queues changed files.
- **Remote to local**: each device keeps a **WebSocket or SSE** connection for change notifications, and also polls `GET /changes?cursor=...` every few minutes in case it missed one. The cursor is the position in the user's change journal that this device has already applied.
- **Delta sync**: only chunks whose fingerprints changed are uploaded. With **content-defined chunking** (lesson 13.10), chunk boundaries are chosen by a rolling hash of the content, so inserting a paragraph near the start of a file changes one or two chunks instead of shifting every later fixed-size block.

~~~dg-flow Sync: changed chunks go up, a metadata commit makes a new version, other devices are told.
[Upload] Laptop -(changed chunks only)-> *Blob storage -(chunks verified)-> Metadata: new version
[Notify] Metadata -(change event)-> Phone and desktop -(fetch missing chunks)-> Local copy updated
~~~

### Why content addressing helps

A chunk is stored under its hash, so identical content is stored once no matter how many files or users contain it. That gives deduplication for free, makes uploads resumable (the client re-asks which hashes are missing), and makes every chunk immutable: a chunk with hash X will always contain the same bytes. Versions are then just different block lists pointing at mostly shared chunks.

~~~dg-tree One file, two versions. Versions are block lists; unchanged chunks are shared, so a new version costs only its changed chunks.
report.docx
  v5 | block list
    chunk a1f3
    chunk 7c20
    chunk e991
  *v6 | block list
    chunk a1f3 | shared
    chunk 44bd | new
    chunk e991 | shared
~~~

## The commit: a compare-and-set on the version

After its chunks are uploaded, the client **commits** a new version: "set report.docx to this block list". Every commit states which version the edit was based on: "this is my edit of **version 5**". The server accepts the commit only if version 5 is still the latest. This is a **compare-and-set** (CAS) on the file's version number:

```
UPDATE files SET version = 6, block_list = :new
WHERE file_id = :id AND version = 5;
-- 1 row updated: success.  0 rows: someone else committed first.
```

The check and the write happen atomically in one statement, so two simultaneous commits cannot both succeed. This is the same optimistic concurrency idea used in many places (lesson 03.06): do not lock while the user edits for hours; check at commit time whether anything changed underneath you.

## Two devices edit the same file offline

Your laptop and your desktop both have `report.docx` at version 5. You go offline on both (say the laptop is on a plane and the desktop's network is down), edit the file on each, and both reconnect. What should happen?

The simplest rule is **last write wins** (LWW): whichever save reaches the server last becomes the file. It is easy, and it silently throws away one of your edits. For a sync service whose promise is "never lose my work", that is unacceptable, so real services detect the conflict and keep both.

1. The laptop reconnects first and commits "new content, based on v5". v5 is current, so the CAS succeeds and this becomes **v6**.
2. The desktop reconnects and commits "new content, based on v5". The latest is now v6, so the CAS updates zero rows and the server **rejects** the commit as a conflict.
3. The desktop client handles the conflict. It cannot merge a Word document's binary contents, so it keeps both: it uploads its edit as a **new file** next to the original, such as `report (Desktop's conflicted copy 2026-11-09).docx`.
4. It then downloads v6, so its `report.docx` matches everyone else's.
5. Both files sync to every device. The user opens them and decides how to combine them.

~~~dg-seq Both devices edited version 5; the second commit conflicts and becomes a conflicted copy.
Laptop > Server : commit report.docx, base v5
Server -> Laptop : OK, now v6
Desktop > Server : commit report.docx, base v5
Server -> Desktop : conflict: latest is v6
Desktop > Server : create "report (conflicted copy).docx"
Server -> Desktop : v6 and the copy sync to all devices
~~~

~~~dg-timeline Two offline edits of the same file, and how the server resolves them on reconnect.
[Laptop] 09:00 | Has v5, goes offline
[Desktop] 09:00 | Has v5, goes offline
[Laptop] 10:15 | Edits, saved locally
[Desktop] 11:40 | Edits, saved locally
[Laptop] 13:00 | Reconnects, commit wins: v6
[Desktop] 13:05 | *Reconnects, conflict, conflicted copy
[Desktop] 13:06 | Downloads v6
~~~

Notice the order of reconnection decided the outcome, not the order of the edits. The desktop's edit was later in wall-clock time, but the laptop committed first. That is deliberate: device clocks are not trustworthy (lesson 06.05), and "who committed first to the server" is the only order the server can establish reliably.

## Why not last write wins, or merging?

| Approach | What happens | Good for |
| --- | --- | --- |
| Last write wins | the later commit replaces the earlier one; one set of edits vanishes | settings and data where losing an edit is harmless |
| Conflicted copy | both versions kept as separate files; the user merges | arbitrary files: documents, images, spreadsheets |
| Automatic merge | changes combined (OT, CRDTs, line-based merge) | formats the system understands: plain text, structured documents (15.05) |

~~~dg-compare Three ways to resolve two offline edits of one file.
[Last write wins]
v5 -(laptop, desktop)-> Desktop wins
- Laptop edits silently lost
+ Simplest
[*Conflicted copy]
v5 -(laptop)-> v6 + copy
+ Nothing lost
+ Works for any file type
- User must merge by hand
[Automatic merge]
v5 -(both)-> Merged v6
+ No user effort
- Only for formats the service understands
~~~

A general file sync service does not understand most file formats, so it cannot merge. A conflicted copy loses nothing and makes the problem visible. Google Docs and similar editors can merge, because they own the format and record each edit as an operation (lesson 15.05). Version history (lesson 13.11) is a second safety net: even if something is overwritten, the previous version is still recoverable for 30 days or more.

<div class="ex">
<div class="lab">Worked example</div>

**three devices, one file**

Phone, laptop and desktop all have notes.txt at v12. All go offline and all edit.

Desktop reconnects first: commit based on v12 succeeds, now v13. Laptop reconnects: based on v12, conflict, becomes "notes (Laptop's conflicted copy).txt". Phone reconnects: based on v12, conflict, becomes "notes (Phone's conflicted copy).txt".

The user ends up with three files. Annoying, but every word typed on every device survives. With last write wins, the phone (last to reconnect) would have overwritten both others, and two devices' work would be gone without a trace.

</div>

## Details that matter

- **Record the base version when editing starts.** The base is the version on disk before the local edit, not the version the server has when the device reconnects. The client keeps a local database of the version it last synced for each file.
- **Identical content is not a conflict.** If both devices produced the same file hash (you saved the same downloaded file on both), accept it silently.
- **Delete versus edit.** One device deletes the file while another edits it. Usually the edit wins and the file comes back, because losing work is worse than an undeleted file.
- **Folders and renames.** Renaming a folder on one device while adding a file to it on another must not lose the file. Store files under stable IDs with a parent folder ID, and apply operations to IDs, not paths. The rename changes one row; the new file still points at the folder's ID.
- **Many small commits.** Editors save often. The client debounces (waits a second or two after the last change) before uploading, so ten autosaves become one commit.
- **Partial uploads.** If the device dies halfway through uploading chunks, nothing is committed; the old version stays current, and the uploaded chunks are reused next time because they are addressed by hash.

## Failure handling

- **Missed notifications** are covered by the cursor: whenever a device connects or polls, it fetches every change after its cursor, so a lost WebSocket message only delays sync.
- **Server-side races** are handled by the CAS; there is no lock to leak.
- **Orphaned chunks** (uploaded but never committed, or referenced only by expired versions) are found by a garbage collector that counts references, after a grace period long enough for in-flight commits.
- **Corruption** is caught by hashes: a downloaded chunk whose SHA-256 does not match its name is discarded and fetched again.

## Common mistakes

- Defaulting to last write wins for user files.
- Using device timestamps to decide which edit is newer.
- Uploading whole files on every change.
- Treating paths as identity, so renames break things.
- Relying only on push notifications without a cursor-based catch-up.

## In an interview

Explain chunks, hashes and block lists first, because the conflict mechanism sits on the commit step. Then walk the two-device scenario with version numbers on the board: base v5, first commit wins with CAS, second conflicts, conflicted copy, version history. Contrast with last write wins and automatic merging. Ask: which file types matter most (if mostly text, automatic merging becomes attractive), and is shared-folder collaboration in scope (many users conflict far more often than one user's devices)?

<div class="take">
<div class="lab">Key takeaway</div>

sync uploads only changed, fingerprinted chunks and commits a new version with a compare-and-set that succeeds only if the device's base version is still current. When two offline edits collide, the first commit wins and the second is rejected; since the service cannot merge arbitrary file formats, the second edit is kept as a conflicted copy beside the winning version, so no work is lost, and version history backs that up.

</div>
