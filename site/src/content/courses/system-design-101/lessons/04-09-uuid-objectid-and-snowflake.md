---
id: "04.09"
course: "system-design-101"
module: "04"
slug: "uuid-objectid-and-snowflake"
title: "UUID, ObjectId, and Snowflake"
keyIdeas: ["**UUIDv4**: 122 random bits. No coordination, practically no collisions, but random order means inserts land all over a B-tree, causing page splits and poor cache use.", "**UUIDv7**: starts with a Unix millisecond timestamp, then random bits. Time-ordered, so inserts append near the end of the index. Good default for new systems.", "**MongoDB ObjectId**: 12 bytes = 4-byte timestamp + 5-byte random value + 3-byte counter. Roughly sortable by creation time.", "**Snowflake**: 64-bit, time-ordered, needs a machine ID assignment.", "128-bit IDs use more storage and index space than 64-bit ones."]
question: "Why does UUIDv4 hurt B-tree insert performance and UUIDv7 doesn't?"
answer: "A B-tree keeps keys sorted, so where an insert goes depends on the key's value. UUIDv4 values are random, so each insert lands on a random leaf page anywhere in the index. Once the index is larger than memory, most of those pages are not cached, so inserts cause random disk reads and churn the cache; full pages split in the middle and settle around 70% full, making the index bigger; and many distinct pages are dirtied, multiplying write-ahead log volume (full-page writes). UUIDv7 puts a 48-bit millisecond timestamp in its most significant bits, so each new key is larger than the previous ones and always goes to the rightmost leaf pages, which stay in memory and fill sequentially to nearly 100%, just like an auto-increment key, while still needing no coordination to generate."
sources: [{"title": "RFC 9562: UUIDs (incl. v7)", "url": "https://datatracker.ietf.org/doc/html/rfc9562", "note": "UUID versions; v7 is time-ordered."}, {"title": "MongoDB docs: ObjectId", "url": "https://www.mongodb.com/docs/manual/reference/method/ObjectId/", "note": "Timestamp + random + counter."}]
tools: ["MongoDB"]
---

## UUIDs and their versions

A **UUID** (universally unique identifier) is a 128-bit value designed to be unique without any coordination: any machine can generate one at any time, and it will not collide with any other in practice. It is usually written as 32 hexadecimal digits in five groups, such as `0190f2a4-6b3c-7d21-9a4e-3f5c8b7e1d02`. A few bits in fixed positions record the **version** (how the rest was generated) and the **variant** (the layout family).

The current standard, RFC 9562 (published 2024), defines these versions:

- **v1**: a 60-bit timestamp (in 100-nanosecond steps since 1582) plus a node ID, often the machine's MAC address. It leaks hardware identity, and the timestamp bits are stored in an awkward order (low bits first), so v1 values do not sort by time.
- **v3 and v5**: a hash (MD5 for v3, SHA-1 for v5) of a namespace and a name. The same input always gives the same UUID, which is useful for deterministic IDs such as "the UUID for this URL".
- **v4**: 122 random bits. By far the most common today.
- **v6**: v1's fields reordered so the value sorts by time. Meant for systems already using v1.
- **v7**: a 48-bit Unix timestamp in milliseconds first, then 74 bits of randomness (or a counter). Recommended for new systems that want time-ordered keys.
- **v8**: a free-form layout for custom schemes.

### How unique is random?

With 122 random bits, there are about 5.3 × 10^36 possible v4 values. By the birthday bound, you would need to generate about 2.7 × 10^18 UUIDs (a billion per second for 85 years) for a 50% chance of even one collision. In practice, collisions come only from broken random number generators (for example, two virtual machines cloned with the same random seed), not from the maths.

## How a B-tree reacts to the order of keys

A B-tree index (lesson 03.04) keeps its keys **sorted**. Where a new key goes is decided entirely by its value. That makes the **order in which keys are generated** matter a great deal for insert performance.

**Sequential or time-ordered keys** (auto-increment, Snowflake, UUIDv7): each new key is larger than every existing key, so every insert goes to the **rightmost leaf page**. That one page, plus the path of branch pages above it, stays permanently in memory. It fills up, and the database starts a new page next to it. Postgres and InnoDB both recognise this pattern and split the rightmost page by leaving the old page full instead of half full, so the index ends up densely packed, close to 90 to 100% full.

**Random keys** (UUIDv4): each insert lands on a random leaf **anywhere** in the index. On a small table that does not matter: the whole index is in memory. On a large table, most leaf pages are not in memory, so an insert must first **read a random page from disk**, then modify it, and later write it back. The cache is constantly churned by pages that are each touched once. Full pages split in the middle, leaving two half-full pages, so the index settles at around **70% full** on average and is correspondingly larger. More pages also means more dirty pages at each checkpoint, and with Postgres's full-page writes (lesson 03.08), the first change to each page after a checkpoint writes the whole 8 KB page to the WAL, so random keys can multiply WAL volume several times over.

~~~dg-compare Where inserts land in a large primary-key index with random keys versus time-ordered keys.
[UUIDv4, random]
New key -(random position)-> Any of 10M leaf pages
- Most target pages not cached: disk read per insert
- Mid-page splits leave pages about 70% full
- Many dirty pages, much more WAL
[*UUIDv7 or Snowflake, time-ordered]
New key -(always largest)-> Rightmost leaf page
+ Same few pages stay hot in memory
+ Pages fill to near 100%
+ Few dirty pages per checkpoint
~~~

This effect is the main reason UUIDv6 and v7 were added: values that are not time-ordered, such as v4, scatter inserts across database indexes and can significantly hurt performance, while time-ordered v7 values cluster together and sort correctly as raw bytes.

### Reads suffer too

Rows created at about the same time are usually read together: today's orders, this week's events, a user's recent posts. With time-ordered keys, those rows sit in neighbouring index pages (and, in InnoDB, where the table is clustered by primary key, in neighbouring table pages). With random keys, "the 1,000 newest orders" are spread over 1,000 different pages.

### And size matters

A UUID is 16 bytes; a 64-bit integer is 8. The difference is repeated in the primary key index, in every **secondary index** (in InnoDB each secondary index entry carries the primary key), and in every **foreign key** column that references the row and its index. Storing a UUID as a 36-character string instead of a native 16-byte type more than doubles it again. Flickr gave index size as a reason to avoid GUIDs (lesson 03.09).

~~~dg-bars Approximate primary-key index size for 1 billion rows, combining key size and how full the pages end up.
BIGINT, sequential | 22 | about 22 GB
UUIDv7, native 16 bytes | 30 | about 30 GB
UUIDv4, native 16 bytes | 43 | about 43 GB
*UUIDv4 stored as text | 75 | about 75 GB
~~~

## UUIDv7 in detail

- **Bits 0 to 47**: Unix time in milliseconds. Because these are the most significant bits, sorting the raw bytes sorts by creation time.
- **4 bits**: the version, 7.
- **12 bits**: random, or a sub-millisecond fraction or counter.
- **2 bits**: the variant.
- **62 bits**: random.

That gives 74 bits that can be random. If a generator creates many IDs in one millisecond and wants them strictly increasing, it can use some of those bits as a **counter** that increments within the millisecond, seeded randomly. A generator must never return a duplicate: if the counter overflows, it waits for the clock to advance (the same rule as Snowflake, lesson 04.11).

~~~dg-layers UUIDv7's 128 bits from most to least significant. Time comes first, so values sort by creation time.
*Unix time in ms | 48 bits
Version, 7 | 4 bits
Random or counter | 12 bits
Variant | 2 bits
Random | 62 bits
~~~

You get coordination-free generation like v4, and the index locality of sequential keys. Postgres 18 added a built-in `uuidv7()` function, and libraries exist for every major language.

The trade-off: **the creation time is visible** in the ID. Anyone who sees an ID can read when the record was created, and by comparing two IDs they may infer rates (how many signups happened between two accounts, for example, if they can sample many IDs). If that matters, use v7 internally and a separate random or encrypted ID in public URLs.

## MongoDB's ObjectId

An **ObjectId** is the 12-byte ID MongoDB generates by default for each document's `_id`:

- **4 bytes**: seconds since the Unix epoch.
- **5 bytes**: a random value chosen once per process.
- **3 bytes**: a per-process counter, starting at a random value.

Because the timestamp comes first, ObjectIds sort roughly by creation time, but only to the **second**. Within one second, IDs from the same process are ordered by the counter, but IDs from different processes interleave arbitrarily. Do not rely on them for strict ordering; use them as roughly time-ordered unique keys.

## Comparing the options

~~~dg-matrix The common ID types compared on the properties that matter for a primary key.
ID | Size | Time-ordered | Needs coordination | Leaks creation time
Auto-increment | 8 bytes | yes, strictly on one DB | yes, one database | partial, and row counts
UUIDv4 | 16 bytes | no | no | no
UUIDv7 | 16 bytes | yes, ms | no | yes
ObjectId | 12 bytes | partial, seconds | no | yes
Snowflake | 8 bytes | yes, roughly ms | partial, unique worker IDs | yes
~~~

<div class="ex">
<div class="lab">Worked example</div>

**an events table outgrows memory**

An events table receives 5,000 inserts per second with a UUIDv4 primary key. The server has 64 GB of RAM for the buffer pool.

At 200 million rows the primary key index is about 9 GB and fits in memory: inserts take about 0.2 ms. At 2 billion rows it is about 90 GB, well beyond the buffer pool. Now roughly 70% of inserts must read their target leaf page from disk first. At 0.1 ms per SSD read, that is fine in isolation, but 3,500 random reads per second from inserts alone compete with every query. WAL volume has also grown about 4 times because each checkpoint interval touches millions of distinct pages, each logged in full the first time. Insert latency climbs to 3 ms and replicas lag.

Switching new rows to UUIDv7 (same column type, so no schema change) makes every insert hit the newest few pages, which stay cached. Index growth slows because pages fill completely, WAL volume drops back, and insert latency returns to about 0.2 ms. Old v4 rows remain valid; the index is simply random in its old part and ordered in its new part.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the cost in a join-heavy schema**

An orders table with 500 million rows is referenced by `order_items` (3 billion rows) and `payments` (500 million rows), each with an index on `order_id`.

With a 16-byte UUID instead of an 8-byte integer, every reference grows by 8 bytes: 3.5 billion references × 8 bytes = 28 GB of extra column data, plus about the same again in the foreign-key indexes. That is roughly 56 GB more to store, back up, replicate and, above all, cache. If the hot part of the database was 60 GB and the server has 64 GB of RAM, the extra size can be the difference between serving from RAM and serving from disk.

</div>

## Common mistakes

- **Using UUIDv4 primary keys on large, write-heavy tables** without measuring the effect on index size and cache hit rate.
- **Storing UUIDs as strings** (`VARCHAR(36)`) instead of a native 16-byte type.
- **Assuming ObjectId or Snowflake order is exact** across processes.
- **Exposing v7 or Snowflake IDs publicly** when creation time is sensitive.
- **Generating UUIDs from a weak random source**, the only realistic way to get collisions.

## In an interview

If you choose UUIDs, say which version and why: "UUIDv7, so generation needs no coordination but inserts stay at the end of the B-tree". If an interviewer asks why v4 is slow, explain the three effects: random leaf pages that are not cached, half-full pages from mid-page splits, and extra WAL. Mention the 16-byte size cost in every index and foreign key.

<div class="take">
<div class="lab">Key takeaway</div>

random UUIDv4 keys scatter inserts across a B-tree, causing cache misses, mid-page splits, half-empty pages and extra log volume once the index outgrows memory. Time-ordered IDs (UUIDv7, ObjectId, Snowflake) keep inserts at the end of the index, where pages stay hot and fill completely.

</div>
