---
id: "03.04"
course: "system-design-101"
module: "03"
slug: "b-tree"
title: "B-tree"
keyIdeas: ["A B-tree (really B+tree in most databases) is a balanced tree of fixed-size pages. Internal pages hold keys and child pointers; leaf pages hold keys and row pointers and are linked for range scans.", "High fan-out (hundreds of keys per page) keeps the tree shallow: 3-4 levels can index billions of rows. A lookup reads one page per level, and top levels are usually cached.", "Inserts go into the right leaf; a full page splits into two and pushes a key up.", "Great for point lookups and range queries. Random inserts cause page splits and fragmentation, which is why sequential keys insert faster."]
question: "Why is a B-tree lookup O(log n) with a very small n of disk reads in practice?"
answer: "Each B-tree node is a whole disk page holding hundreds of keys, so it has hundreds of children and the base of the logarithm is in the hundreds, not 2. A billion keys need only about 4 levels (300^4 is 8.1 billion). The root and upper branch levels are read by every lookup, so they stay permanently in the buffer pool (the top two levels are only a few MB). A lookup therefore costs one or two real disk reads (the leaf and the table row), often fewer. The cost is dominated by page reads, not comparisons, and the wide fanout keeps the page count tiny."
sources: [{"title": "Use The Index, Luke: Anatomy of an index", "url": "https://use-the-index-luke.com/sql/anatomy", "note": "B-tree leaf nodes and traversal."}, {"title": "CMU 15-445: B+Tree lecture", "url": "https://15445.courses.cs.cmu.edu/", "note": "See the tree indexes lecture."}]
tools: ["PostgreSQL"]
---

## The problem a B-tree solves

Disks, even fast SSDs, are slow compared with memory, and they read data in blocks rather than bytes. A random read from an NVMe SSD costs around 100 microseconds; from a spinning disk, around 10 milliseconds. Reading 8 KB costs about the same as reading 8 bytes. So the cost of a lookup structure on disk is measured almost entirely in **how many pages it has to read**, not in how many comparisons it does.

A binary search tree is a poor fit. Each node holds one key and two children, so a billion keys need a tree about 30 levels deep, which could mean 30 random disk reads per lookup. The **B-tree**, invented in 1970 for exactly this problem, makes each node one whole disk page holding hundreds of keys. Each node therefore has hundreds of children, and the tree is very wide and very shallow.

The version used by virtually every relational database is technically a **B+ tree**: all actual entries live in the bottom layer (the leaves), and the upper layers hold only separator keys that guide the search. This course follows common usage and calls it a B-tree.

## Two structures working together

A B-tree index is best understood as two structures combined.

**The leaf level** holds every index entry in sorted order: the indexed value plus a pointer to the table row (lesson 03.03 explains what that pointer is in Postgres and InnoDB). Each leaf is one page, 8 KB in Postgres or 16 KB in InnoDB, so it holds hundreds of entries. Leaves are linked to their neighbours in a **doubly linked list**, which lets the database walk forwards or backwards through a range without going back up the tree.

**The tree** sits on top so the database can find the right leaf quickly. Each entry in a **branch node** (also called an internal node) holds a separator key and a pointer to a child page. In one common layout, each separator is the largest value in the child it points to. Branch layers are added until everything above the leaves fits in a single **root** page.

~~~dg-tree A small B+ tree. Branch pages hold only separator keys; every real entry lives in a leaf, and leaves are chained left to right.
Root | separators 40, 83, max
  Branch A | keys up to 40
    Leaf 1 | 3, 9, 17
    Leaf 2 | 22, 31, 40
  *Branch B | keys 41 to 83
    Leaf 3 | 46, 51, 55
    *Leaf 4 | 57, 64, 83
  Branch C | keys above 83
    Leaf 5 | 88, 91, 97
    Leaf 6 | 102, 140, 199
~~~

## Walking the tree

To find the key 57:

1. Read the root page. Scan its separators in order until you find one greater than or equal to 57. That is 83, so follow its pointer to Branch B.
2. Read Branch B. Its separators say keys from 56 to 83 are in Leaf 4. Follow that pointer.
3. Read Leaf 4. Find 57, and with it the pointer to the table row.
4. Read the table page that holds the row.

Within a page the search is a binary search over a few hundred keys in memory, which takes well under a microsecond. The expensive part is the page reads, and there is exactly one per level.

Because the tree is **balanced**, every leaf is the same distance from the root, so every lookup takes the same number of steps. Insertions and deletions keep it balanced automatically: the tree only ever grows at the top, by adding a new root.

## Why the depth stays tiny

The depth of the tree grows with the logarithm of the number of entries, with a base equal to the **fanout** (the number of children per node). With a toy fanout of 4, capacity multiplies by 4 at each level:

| Depth | Max entries with fanout 4 | Max entries with fanout 300 |
| --- | --- | --- |
| 1 | 4 | 300 |
| 2 | 16 | 90,000 |
| 3 | 64 | 27 million |
| 4 | 256 | 8.1 billion |
| 5 | 1,024 | 2.4 trillion |

Real pages hold hundreds of entries. An 8 KB page of 8-byte integer keys with 6-byte pointers and some header bytes holds around 400 to 500; longer keys such as emails mean fewer, perhaps 100 to 200. So each extra level multiplies capacity by a few hundred. Indexes on tables with millions of rows are typically 3 or 4 levels deep; 5 is uncommon and 6 is rare.

~~~dg-bars How many levels a lookup needs for one billion keys, depending on how many children each node has.
Binary tree, fanout 2 | 30 | 30 levels
Fanout 10 | 9 | 9 levels
Fanout 100 | 5 | 5 levels
*Fanout 300 | 4 | 4 levels
Fanout 1,000 | 3 | 3 levels
~~~

### The upper levels are always in memory

Every lookup reads the root, and every lookup in a given part of the key space reads the same branch pages. Those pages are therefore the hottest pages in the whole database and stay in the buffer pool permanently. With a fanout of 300, the root plus the second level is about 301 pages, under 3 MB. Even the third level is only about 90,000 pages, around 700 MB, which often fits in memory too.

So the practical answer to "O(log n), but fast" has two parts. The log has a huge base, so a billion entries means a depth of 4. And the top two or three of those four levels are cached, so a lookup costs **one or two real disk reads**, and often none.

## Why an "indexed" query can still be slow

Every index lookup has three steps, and only the first one is bounded.

1. **Tree traversal**: one page per level, 3 to 5 pages. Always fast.
2. **Following the leaf chain**: reading every matching entry. A range that matches a million entries reads a million entries, perhaps 3,000 leaf pages.
3. **Fetching table rows**: one table access per matching entry. Matching rows are usually scattered across many table pages, so this is random I/O, often one page read per row.

Steps 2 and 3 are where slow indexed queries spend their time. A common myth is that indexes "degenerate" and need rebuilding to be fast again; rebuilding cannot shorten the leaf chain or reduce the number of rows you asked for. The real fixes are a more selective condition, a better column order (so the matching entries are contiguous), or a covering index that answers from the index and skips step 3 entirely.

~~~dg-bars Where the time goes in an index range scan that matches 10,000 rows, assuming cold pages.
Tree descent | 4 | 4 pages
Leaf chain | 30 | about 30 leaf pages
*Table rows | 10000 | up to 10,000 random pages
~~~

## How inserts keep it balanced: page splits

A new entry must go into the leaf where it belongs in sort order. If that leaf has room, it is inserted in place and nothing else changes. If the leaf is full, it **splits**:

1. Allocate a new page.
2. Move roughly the upper half of the entries to the new page.
3. Relink the leaf chain so the old page points to the new one and the new one to the old page's former neighbour.
4. Insert a new separator key and pointer for the new page into the parent branch.
5. If the parent is also full, it splits too, and so on up. If the root splits, a new root is created above it and the tree grows one level taller.

~~~dg-compare One leaf page before and after a split caused by inserting 60 into a full page.
[Before]
Leaf 4 is full -> 57, 64, 72, 83
- No room for 60
[*After]
Leaf 4 -(linked to)-> Leaf 4b
Leaf 4: 57, 60, 64
Leaf 4b: 72, 83
+ Parent gains separator 64
~~~

Splits leave both pages about half full. That is why a B-tree built by random inserts tends to be around 70% full on average, wasting some space. Postgres and InnoDB both have an optimisation for the common case of always inserting at the right-hand end (an auto-incrementing ID or a timestamp): instead of splitting 50/50 they leave the old page full and start a fresh one, so sequential inserts produce nearly full pages. Random keys, such as UUIDv4, get no such benefit and also touch a random leaf for every insert, which is the root of the problem in lesson 04.09.

Deletes work the other way. Entries are removed (in Postgres, marked dead and cleaned up later by VACUUM, lesson 03.07), and nearly empty pages can be merged or recycled, though in practice databases are lazy about shrinking indexes.

## B-trees and writes

Every write updates a page **in place**, and with random keys those pages are scattered across the disk. Each change is also recorded in the write-ahead log first (lesson 03.08). On a read-heavy workload, that is a fine trade: reads are as cheap as they can be. On a workload of hundreds of thousands of small writes per second, it becomes the bottleneck. Write-heavy stores therefore often use **LSM trees** (log-structured merge trees), which buffer writes in memory and flush them to disk as sorted files in large sequential writes, at the cost of more work on reads (Module 7).

<div class="ex">
<div class="lab">Worked example</div>

**sizing an index on 1 billion emails**

A users table has 1 billion rows and a B-tree on `email`. An average email is about 25 bytes; with the row pointer and per-entry overhead, say 40 bytes per entry. An 8 KB page at about 90% fill holds roughly 180 entries, but branch pages hold shorter separators and fit around 300.

Leaves: 1 billion / 180 = about 5.6 million leaf pages, about 45 GB. That is the bulk of the index.

Branch levels: 5.6 million / 300 = 18,500 pages at level 2, then 62 pages at level 1, then 1 root. Total depth: 4 (root, two branch levels, leaves).

The root and the 62 pages below it are always cached. The 18,500-page level is about 150 MB and almost certainly cached too. So a lookup reads 3 cached pages, then 1 leaf page and 1 table page that may need the disk: about 2 SSD reads, roughly 0.2 ms.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**why a date-range report is slow despite an index**

A query asks for all orders in March from a 200-million-row table with an index on `created_at`. March has 6 million orders.

The descent takes 4 page reads. The leaf chain for 6 million entries at about 400 per page is 15,000 pages, read mostly sequentially: fast. But fetching 6 million rows means up to 6 million random table page reads. Even at 100 microseconds each that is 10 minutes of I/O if nothing is cached.

The planner will rightly switch to a bitmap scan or a sequential scan (lesson 03.05). The fix is not "rebuild the index" but a covering index with the columns the report needs, or a pre-aggregated summary table.

</div>

## Common mistakes

- **Thinking depth grows quickly.** It barely grows: going from 1 million to 1 billion rows usually adds one level.
- **Rebuilding indexes to fix slow queries.** It rarely helps; slow queries come from leaf and table reads.
- **Using random primary keys without thinking.** They cause splits all over the tree and poor cache locality.
- **Ignoring the table fetch.** An index that returns 100,000 rows still visits up to 100,000 table pages.

## In an interview

When asked why a B-tree is fast, say three things: each node is a page with hundreds of children, so the depth is about 4 even for billions of rows; the top levels are always cached, so a lookup costs one or two disk reads; and the real cost of a range query is the number of matching leaf entries and table rows, not the tree.

<div class="take">
<div class="lab">Key takeaway</div>

a B-tree is a very wide, balanced tree of separator pages over a sorted, linked chain of leaf pages. Depth stays at 3 to 5 even for billions of rows, and the top levels live in memory. Slow index queries come from reading many leaves and table rows, not from the tree itself.

</div>
