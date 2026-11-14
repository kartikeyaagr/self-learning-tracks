---
id: "13.10"
course: "system-design-101"
module: "13"
slug: "fixed-block-chunking-and-content-addressing"
title: "Fixed-block chunking and content addressing"
keyIdeas: ["**Content addressing**: name each chunk by the hash of its content. Identical chunks get the same name, so deduplication, caching and integrity checks come free. Git and IPFS work this way.", "**Fixed-size chunking** (e.g. 4 MB blocks): simple, but inserting one byte near the start shifts every later block boundary, so every later block changes.", "**Content-defined chunking**: choose boundaries where a rolling hash of the content matches a pattern, so boundaries move with the content and an insert only changes nearby chunks.", "rsync uses a rolling checksum to find matching blocks at any offset."]
question: "What does content addressing give you for free? What's the problem with fixed-size blocks when bytes are inserted?"
answer: "Content addressing gives deduplication (identical content has the same hash and is stored once), integrity (data that does not hash to its key is corrupt or tampered with), cheap snapshots and versions (unchanged objects are reused by hash, and one root hash identifies a whole tree), cache-forever immutability (content at a hash never changes, so copies are never stale) and efficient sync by comparing hashes and sending only what the other side lacks. Fixed-size blocks are defined by position, so inserting or deleting even one byte shifts every later boundary and changes every later block's hash, making the rest of the file look new and defeating dedup. rsync's rolling checksum (matching blocks at any byte offset) or content-defined chunking (boundaries chosen by a rolling hash of the content, which resynchronise after the edit) solves it."
sources: [{"title": "Pro Git: Git objects", "url": "https://git-scm.com/book/en/v2/Git-Internals-Git-Objects", "note": "Content addressing in practice."}, {"title": "The rsync algorithm (tech report)", "url": "https://rsync.samba.org/tech_report/", "note": "Rolling checksums."}]
---

## Content addressing

Most storage systems name data by **location**: a file path, a database row ID, an object key that you chose. A **content-addressed** store names data by **what it is**: the key of each piece of data is a **cryptographic hash** of the data itself. Put in some bytes, and you get back their hash; give the hash, and you get the bytes back. You cannot choose the key, and you cannot change the bytes stored under a key without changing the key.

A **cryptographic hash function** such as SHA-256 maps any input to a fixed-size output (32 bytes for SHA-256) such that, in practice, no two different inputs ever produce the same output, and changing a single bit of the input changes about half the bits of the output. That property is what lets a hash stand in for the content.

### Git is a content-addressed store

Git is the best-known example: at its core it is a content-addressable filesystem, a simple key-value store where you insert content and get back a key (a SHA-1 hash) to retrieve it.

- `git hash-object` hashes a short header (`blob <size>\0`) followed by the file's content. The object is zlib-compressed and stored at `.git/objects/xx/yyyy...`, where `xx` is the first two hex characters of the hash and the rest is the file name.
- **Blobs** hold file contents only, with no names.
- **Trees** list entries: a name, a mode (file, executable, symlink, directory) and the hash of a blob or of a subtree.
- **Commits** point to one top-level tree, to their parent commit(s), and record who made the change, when, and why (the message).

```
$ echo 'test content' | git hash-object -w --stdin
d670460b4b4aece5915caf5c68d12f560a9fe3e4
$ git cat-file -p d670460b4b4aece5915caf5c68d12f560a9fe3e4
test content
```

~~~dg-tree Git's object graph. Each node is stored under the hash of its content, and parents refer to children by hash, so unchanged subtrees are shared between commits.
Commit c3 | tree t9, parent c2
  Tree t9 | root directory
    README.md | blob a1, unchanged since c1
    *src/ | tree t7, changed
      main.go | blob b8, new content
      util.go | blob b2, unchanged
    docs/ | tree t4, reused from c2 by hash
~~~

## What it gives you for free

Naming data by its hash produces five properties without any extra machinery.

1. **Deduplication.** Identical content always has the same hash, so it is stored **once**, whether it is two copies of a file in one folder, the same file across a thousand commits, or (in a sync service like Dropbox) the same 4 MB block in files belonging to different users. Writing content that already exists is a no-op: "already have it".
2. **Integrity.** The name **is** a checksum. If stored bytes are corrupted by a failing disk or tampered with, they no longer hash to their key, and the problem is detected on read. Git's `fsck` and every sync client's download check work this way.
3. **Cheap snapshots and versions.** A commit references a whole tree, but unchanged subtrees and blobs are reused **by hash** rather than copied. Each new snapshot costs only what changed, plus small new tree objects along the path to the change (lesson 13.11 applies this to file versions).
4. **Easy caching and replication.** Content at a given hash can **never change**, so it can be cached forever, by anyone, and copied anywhere without any invalidation or consistency protocol. There is no such thing as a stale copy.
5. **Efficient sync.** To synchronise two stores, compare hashes and send only the objects the other side lacks. This is Dropbox's "need blocks" (lesson 13.09) and Git's fetch negotiation ("I have these commits; send me what I am missing").

A sixth property comes along with these: **a single hash identifies an entire structure**. A commit hash covers its tree, which covers every subtree and blob, so if two commit hashes are equal, the whole snapshots are equal. Comparing two million-file trees starts with comparing two hashes. Structures built this way are called **Merkle trees** (or Merkle DAGs).

### What it costs

- **Hashing time.** Every byte must be hashed on write, roughly 0.5 to 2 GB per second per core for SHA-256 with hardware support.
- **Immutable objects need garbage collection.** Nothing is ever overwritten, so something must eventually delete objects no longer referenced (lesson 13.11).
- **Collisions are theoretical but not ignorable for weak hashes.** SHA-1, which Git originally used, has had a practical collision demonstrated (two different PDFs with the same SHA-1), so Git now uses a collision-detecting SHA-1 and supports SHA-256 repositories. With SHA-256 you would need about 2^128 objects before a random collision became likely; that is not a practical concern.
- **Privacy.** In a shared store, "do you already have this hash?" can leak that someone else stored the same file. Lesson 13.11 discusses limiting dedup to one account.

## Fixed-size blocks and the insertion problem

To content-address large files efficiently, you split them into blocks and hash each one. The simplest split is **fixed-size**: block 1 is bytes 0 to 4 MB, block 2 is 4 to 8 MB, and so on. As lesson 13.09 showed, that is excellent for **in-place overwrites**: change one byte and one block's hash changes.

Now **insert** one byte at the start of the file. Every byte after it moves one position later. Block 1 now holds the new byte plus the first 4 MB minus one byte of the old content; block 2 starts one byte earlier than before; and so on to the end. **Every block's content shifts by one byte, so every block's hash changes**, and the whole file looks new. The same happens with deletions. A 1 GB file with a word inserted near the beginning costs a full 1 GB upload.

That matters because insertions are common: adding a paragraph to a document, a row in the middle of a CSV, a line in source code, a new entry in an archive.

~~~dg-compare The boundary-shift problem: one inserted byte at the front shifts every fixed block boundary, so every hash changes.
[Before]
ABCD EFGH IJKL -(hash each 4-byte block)-> h1, h2, h3
+ Overwriting one letter changes one hash
[After inserting X at the start]
XABC DEFG HIJK L -(hash each 4-byte block)-> h4, h5, h6, h7
- All new hashes: nothing deduplicates
- One byte costs the whole file
~~~

There are two classic answers: find matches at **any offset** (rsync), or put block boundaries at positions **determined by the content** (content-defined chunking).

## rsync's answer: match at any offset

The rsync algorithm solves "update a file over a slow link when the other side has an old version". Machine β has the old file B; machine α has the new file A; they want β to end up with A while sending as little as possible.

1. β splits B into **fixed blocks** of S bytes (say 700 bytes) and computes, for each block, a cheap **weak 32-bit rolling checksum** and a **strong 128-bit checksum** (MD4 originally; newer versions use MD5 or faster hashes). It sends this list to α: about 20 bytes per block.
2. α slides a window of S bytes across A at **every byte offset**, not just multiples of S. At each offset it computes the weak checksum and looks it up in a hash table of B's checksums. Only on a weak match does it compute the strong checksum to confirm.
3. α sends β a sequence of instructions: "copy block k of B" for matches, and **literal bytes** for the parts that match nothing. β assembles A from its own blocks plus the literals.

One round trip, and only the genuinely new data travels.

~~~dg-seq An rsync transfer. The side with the old file sends checksums; the side with the new file replies with block references and literal bytes.
Beta (old file B) > Alpha (new file A) : weak + strong checksum per 700-byte block
Alpha (new file A) > Alpha (new file A) : slide window at every offset, match checksums
Alpha (new file A) -> Beta (old file B) : copy block 0, literal "X", copy blocks 1..n
Beta (old file B) > Beta (old file B) : rebuild A from own blocks plus literals
Beta (old file B) > Beta (old file B) : verify whole-file checksum
~~~

### The rolling checksum

Scanning every offset would be hopeless if each window needed its checksum computed from scratch (S operations per offset). The **rolling checksum** updates in constant time as the window slides. For the window of bytes X_k to X_l it has two parts:

- **a** = the sum of the bytes in the window (mod 2^16);
- **b** = a position-weighted sum, where the first byte is weighted by the window length and the last by 1 (mod 2^16).

The checksum is a + 2^16 × b. When the window slides one byte to the right, dropping X_k and adding X_(l+1):

- `a' = a − X_k + X_(l+1)`
- `b' = b − (window length) × X_k + a'`

Two subtractions, an addition and a multiply per byte, however large the window.

<div class="ex">
<div class="lab">Worked example</div>

**rolling the checksum by hand**

Take a 4-byte window over "abcd", bytes 97, 98, 99, 100. Then a = 97 + 98 + 99 + 100 = 394, and b = 4×97 + 3×98 + 2×99 + 1×100 = 388 + 294 + 198 + 100 = 980.

Slide one byte to "bcde" (e is 101). a' = 394 − 97 + 101 = 398. b' = 980 − 4×97 + 398 = 990.

Check directly: 4×98 + 3×99 + 2×100 + 1×101 = 392 + 297 + 200 + 101 = 990. Same answer, without re-summing the window.

</div>

So an inserted byte costs only the block containing it as literal data; every following block still matches, just at an offset shifted by one. Measured on two versions of the Linux kernel source as tarballs (about 24 MB each), with blocks of 500 bytes or more, rsync transferred **only a few percent of the file**, less than the equivalent diff and patch, and false matches of the weak checksum occurred less than once per thousand true matches.

rsync is ideal for one-off transfers between two machines. It is less suited to a **block store**, because the old file's blocks are defined by position, and after many edits the stored blocks no longer line up with anything. For storage, you want boundaries that stay put.

## Content-defined chunking

**Content-defined chunking** (CDC) chooses chunk boundaries by looking at the content itself:

1. Slide a small window (say 48 bytes) across the file, computing a **rolling hash** of the window at every byte (Rabin fingerprinting is the classic choice; Gear hashing, used by FastCDC, is faster).
2. Declare a **chunk boundary** wherever the hash meets a condition, for example its **low 13 bits are all zero**. For random-looking data that happens on average once every 2^13 = 8,192 bytes, giving **8 KB average chunks**.
3. Enforce a **minimum and maximum** chunk size (say 2 KB and 64 KB), so long runs of repetitive data (where the condition never or always fires) do not produce absurd chunks.
4. Hash each chunk with SHA-256 and store it in the content-addressed store, exactly as with fixed blocks.

The key property: a boundary depends only on the 48 bytes before it. Insert a byte somewhere, and the boundaries before the insertion are unchanged; the chunk containing the insertion changes; and the **next boundary after it reappears at the same content**, because the bytes around it are the same as before. From then on, chunks are identical to the old ones and deduplicate.

~~~dg-flow Content-defined chunking: boundaries follow the content, so after an insertion they resynchronise and later chunks keep their hashes.
Bytes -(rolling hash over 48-byte window)-> Low 13 bits zero? -(yes: cut here)-> *Boundary tied to content -(insert a byte)-> Only 1 or 2 chunks change
~~~

CDC is used widely in backup and sync tools (restic, Borg and the Low-Bandwidth File System are well-known examples) and in deduplicating storage systems. The average chunk size is a trade-off: small chunks (4 to 16 KB) deduplicate finely but create millions of chunks and large chunk lists; large ones (1 MB) keep metadata small but waste more on each edit. Backup tools often use around 1 MB averages for large datasets; sync tools for documents use smaller ones.

~~~dg-bars Bytes sent after inserting one paragraph near the start of a 40 MB document, including metadata. Fixed blocks resend almost everything.
Fixed 4 MB blocks | 40000 | about 40 MB, all blocks
Fixed 64 KB blocks | 40000 | about 40 MB, almost all blocks
rsync, 700-byte blocks | 1100 | about 1.1 MB, mostly checksums
*Content-defined, 8 KB avg | 176 | 16 KB chunks + 160 KB list
~~~

<div class="ex">
<div class="lab">Worked example</div>

**inserting a paragraph**

A 40 MB document is stored as about 5,000 content-defined chunks averaging 8 KB. The user inserts a 500-byte paragraph on page 3.

The client re-chunks the file. Every boundary before page 3 is found at exactly the same place. The chunk containing the insertion now holds 500 more bytes and gets a new hash; occasionally the insertion itself creates or removes a boundary, so its neighbour changes too. After that, the next boundary falls on the same content as before, and every later chunk has its old hash.

The client asks the server which hashes it lacks: 2 new chunks, about 16 KB, plus a new chunk list (5,000 hashes × 32 bytes = 160 KB, which itself can be deduplicated or sent as a diff). With fixed 4 MB blocks, all 10 blocks would have changed: 40 MB.

</div>

## Common mistakes

- **Assuming fixed blocks handle edits.** They handle overwrites; they fail on insertions and deletions.
- **Using a weak hash as the address.** CRC32 or a rolling checksum is fine for finding boundaries or candidates, never as the identity of stored data. Confirm with a cryptographic hash.
- **No minimum or maximum chunk size.** Files full of zeros or repeated patterns produce millions of tiny chunks or one enormous one.
- **Chunks too small.** Metadata (chunk lists, index entries) can outgrow the savings.
- **Forgetting garbage collection.** Content-addressed stores only grow unless something deletes unreferenced objects.

## In an interview

Define content addressing in one sentence (the key is the hash of the content), then list what it gives you: dedup, integrity, cheap snapshots, cache-forever immutability and efficient sync by comparing hashes. Then explain the insertion problem with fixed blocks (every later boundary shifts, every later hash changes) and the two fixes: rsync's rolling checksum to match at any offset for one-off transfers, and content-defined chunking, which places boundaries by content so they resynchronise after an edit, for block stores.

<div class="take">
<div class="lab">Key takeaway</div>

content addressing names data by its hash, which gives deduplication, integrity checking, cheap snapshots, cache-forever immutability and simple sync by comparing hashes for free, as Git's blobs, trees and commits show. Fixed-size blocks break on insertions because every later boundary shifts and every later hash changes; rsync's constant-time rolling checksum finds matches at any offset, and content-defined chunking puts boundaries where the content says so, so only the chunks near an edit change.

</div>
