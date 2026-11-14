---
id: "15.01"
course: "system-design-101"
module: "15"
slug: "a-url-shortener-one-machine"
title: "A URL shortener, one machine"
keyIdeas: ["Requirements: create short URL for a long URL, redirect quickly, optional custom alias and expiry. Very read-heavy.", "Short code generation: **base62-encode a unique ID** (from an auto-increment or ID generator). No collisions by construction; 7 characters give 62^7 ≈ 3.5 trillion codes. Sequential IDs are guessable, so shuffle or use random ID ranges if that matters.", "**Hash the URL** (MD5, take first 7 chars): same URL gives same code, but collisions are possible and must be checked and resolved.", "One machine with Postgres and an index on the code handles a lot: thousands of redirects per second.", "Redirect with HTTP 301 (cached by browsers, less load, but you lose click analytics) or 302 (every click hits you)."]
question: "Base62 of an ID vs hash of the URL: which avoids collisions and why?"
answer: "Base62 of a unique ID avoids collisions. Each URL gets a distinct integer from a counter or ID generator, and Base62 is only a way of writing that number in 62 symbols, so two different IDs always give two different codes. Hashing the URL and keeping the first 7 characters throws most of the hash away, so different URLs can share the same 7 characters (by the birthday problem, after 1 billion links about 1 in 3,500 new links collides); you must check the database or a Bloom filter, rely on a unique index and rehash on a clash. The cost of Base62 IDs is that you need an ID generator, and codes are sequential and guessable unless you scramble the ID with a reversible permutation before encoding."
sources: [{"title": "ByteByteGo: Design a URL Shortener", "url": "https://bytebytego.com/courses/system-design-interview/design-a-url-shortener", "note": "Hashing vs base62 IDs."}, {"title": "System Design Primer: Pastebin solution", "url": "https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/pastebin/README.md", "note": "Same shape, with estimates."}]
tags: ["Design"]
tools: ["PostgreSQL"]
---

## What we are building

A URL shortener turns `https://example.com/some/very/long/path?with=params` into `https://sho.rt/zn9edcu`, and redirects anyone who opens the short link to the original. It looks trivial, which is exactly why it is a classic design exercise: the whole problem fits in your head, so you can go deep on the two interesting parts, **generating short codes** and **serving redirects fast**. This lesson designs the one-machine version and settles the code-generation question. Lesson 15.03 scales it out.

Typical requirements:

- **Functional**: create a short link from a long URL; redirect the short link to the long URL; short codes use only `0-9a-zA-Z`; links are never edited or deleted (in this version).
- **Non-functional**: 100 million new URLs a day; redirects must be fast (tens of milliseconds) and highly available; a short code must never point to the wrong URL.

### Back-of-the-envelope numbers

- Writes: 100M ÷ 86,400 s ≈ **1,160 per second** on average, perhaps 2,000 to 3,000 at peak.
- Reads at 10:1: ≈ **11,600 per second** (many real shorteners see 100:1; lesson 15.03 uses that).
- Over 10 years: 100M × 365 × 10 = **365 billion** records. At about 100 bytes each, that is **36.5 TB**.
- Code length: 62 characters per position.

~~~dg-bars How many distinct codes each length of Base62 gives, against the 365 billion links needed over 10 years.
5 characters | 916000000 | 916 million
6 characters | 56800000000 | 56.8 billion
*7 characters | 3520000000000 | 3.5 trillion
8 characters | 218000000000000 | 218 trillion
Needed in 10 years | 365000000000 | 365 billion
~~~

62^6 ≈ 56.8 billion is too few; 62^7 ≈ 3.5 trillion covers 365 billion with almost 10× headroom. So **7 characters**.

Pastebin is the same problem with a body attached (10M pastes and 100M reads a month: about 4 writes and 40 reads per second). Keep metadata in SQL and the paste text in an object store, because large text blobs in database rows bloat the table and its backups.

## The one-machine design

Start simple: one web server and one SQL database. Do the numbers allow it? 1,160 inserts per second is comfortable for a single PostgreSQL or MySQL server on SSD (a well-tuned server handles several thousand small inserts per second). 11,600 indexed reads per second is also feasible, especially with a cache in memory. Storage is the constraint that will eventually force a second machine: 36.5 TB will not fit on one server, but the first year's 3.65 TB will. So one machine is a valid starting point, and the design should make the later split easy.

The table:

```
urls
  id         BIGINT PRIMARY KEY
  short_code CHAR(7) UNIQUE
  long_url   TEXT
  created_at TIMESTAMP
```

APIs:

- `POST /api/v1/data/shorten {longUrl}` returns the short URL.
- `GET /{code}` returns a redirect (or 404).

The unique index on `short_code` makes each lookup a B-tree probe (lesson 03.04), a handful of page reads, rather than a table scan. With Base62 IDs you can even drop the separate `short_code` column and index: decode the code back into the integer ID and look up the primary key directly.

~~~dg-seq The two endpoints on one machine: shortening writes a row, redirecting reads it.
Client > Web server : POST /shorten longUrl
Web server > Database : SELECT code WHERE long_url = ? (optional dedupe)
Database -> Web server : not found
Web server > Database : INSERT, get id 11157
Web server > Web server : base62(11157) = 2TX
Web server -> Client : https://sho.rt/2TX
Browser > Web server : GET /2TX
Web server > Database : SELECT long_url WHERE id = 11157
Database -> Web server : https://example.com/long/path
Web server -> Browser : 302 Location: https://example.com/long/path
~~~

## Two ways to make the code

The real question of this design is how to turn a long URL into a 7-character code that is unique.

### Option 1: hash the URL

Hash the long URL with CRC32, MD5 or SHA-1 and keep the first 7 characters. A hash function maps any input to a fixed-size, random-looking output: MD5 is 128 bits (32 hex characters), SHA-1 is 160 bits (40 hex characters). All of them are longer than 7 characters, so you must cut them, and cutting creates **collisions**: two different URLs with the same first 7 characters.

How likely is that? This is the **birthday problem**: in a room of 23 people, two probably share a birthday, because the number of pairs grows with the square of the number of people.

<div class="ex">
<div class="lab">Worked example</div>

**how often does a truncated hash collide?**

Seven Base62 characters give N ≈ 3.5 trillion possible codes. Once you have stored n = 1 billion links, a new link collides with an existing one with probability n ÷ N = 10^9 ÷ 3.5 × 10^12 ≈ 0.03%, about 1 in 3,500 inserts.

Across the whole history, the expected number of colliding pairs is about n² ÷ 2N. At n = 1 billion: 10^18 ÷ 7 × 10^12 ≈ 140,000 collisions.

At 365 billion links (10 years), a new insert collides about 365 ÷ 3,520 ≈ 10% of the time. Collisions are not a rare edge case; they are routine, and the code must handle them.

</div>

So hashing needs **collision resolution**. The usual fix is to check the database; on a collision, append a fixed string (or a counter) to the URL and hash again, repeating until the code is free. Checking the database on every write is an extra read per insert, so a **Bloom filter** in memory (lesson 04.13) can answer "definitely not used" quickly and send only the "maybe used" cases to the database. A variant hashes something other than the URL, such as the user's IP plus a timestamp, then Base62-encodes and truncates, with a retry if taken.

Even with a check, there is a race: two servers can both check code `abc1234`, both find it free, and both insert. The unique index on `short_code` is the real guarantee: the second insert fails with a duplicate-key error, and that server retries with a new code.

### Option 2: Base62 of a unique ID

Give every URL a unique integer ID (an auto-increment column on one machine, or a distributed ID generator later, lesson 04.08), then write that number in base 62. Base62 is not a hash; it is just another way of **writing a number**, like decimal or hexadecimal. Different numbers always give different strings, so there are **no collisions** and no need to check.

How Base62 works: repeatedly divide by 62 and map each remainder to a character (`0-9` = 0 to 9, `a-z` = 10 to 35, `A-Z` = 36 to 61). Reading the remainders from last to first gives the code.

<div class="ex">
<div class="lab">Worked example</div>

**encoding 11157**

11157 ÷ 62 = 179 remainder **59** → `X`

179 ÷ 62 = 2 remainder **55** → `T`

2 ÷ 62 = 0 remainder **2** → `2`

Read the remainders backwards: `2TX`. Check: 2 × 62² + 55 × 62 + 59 = 7,688 + 3,410 + 59 = 11,157.

A larger ID, 2009215674938, encodes to `zn9edcu`, giving `https://tinyurl.com/zn9edcu`.

Decoding is the reverse: for each character, multiply the running total by 62 and add the character's value.

</div>

~~~dg-compare Hashing a URL versus Base62-encoding a unique ID: the ID approach cannot collide, the hash approach needs no counter.
[Hash + truncate]
long URL -(MD5)-> 32 hex chars -(first 7)-> code
+ no ID generator needed
+ same URL gives same code
- collisions grow with table size
- check, retry and unique index required
[*Base62 of unique ID]
long URL -(next ID 11157)-> base62 -> 2TX
+ unique in, unique out: no collisions
+ no extra read per write
- needs a counter or ID generator
- sequential codes are guessable
~~~

| | Hash + collision resolution | Base62 of a unique ID |
| --- | --- | --- |
| Length | fixed (7) | grows with the ID (pad or start high for fixed length) |
| Needs an ID generator | no | yes |
| Collisions | possible; must check and retry | impossible |
| Same URL twice | can return the same code (deterministic) | new code unless you look the URL up first |
| Guessable | no | yes, IDs are sequential |

### Making IDs unguessable

With sequential IDs, anyone can enumerate every link by counting: `2TX`, `2TY`, `2TZ`... If links may point to private documents, that is a data leak. Two fixes:

- **Scramble the ID before encoding** with a reversible permutation of the 64-bit space, such as a small Feistel network or a block cipher with a secret key. Each ID maps to a unique, random-looking number, so uniqueness is kept and codes look random. Decoding runs the permutation backwards.
- **Use random IDs**: generate a random 7-character code and insert it, relying on the unique index to reject the rare collision. This is effectively the hash approach with randomness instead of the URL.

### Fixed length

Base62 of small IDs gives short codes (`2TX`). If you want every code to be exactly 7 characters, start the counter at 62^6 (about 56.8 billion), so every ID encodes to 7 characters, and you still have 3.4 trillion IDs left.

## Same URL, same code?

Should shortening the same long URL twice return the same code? With hashing it does naturally. With IDs you must look the URL up first, which needs an index on `long_url`. URLs can be thousands of characters long, so index a hash of the URL instead (a fixed 16-byte MD5 column with an index), and compare the full URL on a match. Many products choose **not** to deduplicate: two users shortening the same URL may want separate click statistics, and skipping the lookup saves a read per write.

## The write and redirect flows

**The write flow** (Base62 version):

1. Validate the URL (scheme is http or https, length limit, not on a malware blocklist).
2. Optionally look up an existing code for this URL.
3. Get a new ID, encode it, and save ID, code and long URL.
4. Return the short URL.

**The redirect flow**:

1. Decode or look up the code (cache first, then the database).
2. If missing, return 404.
3. Otherwise return a redirect with the long URL in the `Location` header.

## 301 or 302?

- **301 Moved Permanently**: the browser caches the redirect and goes straight to the long URL next time, without asking you. Less load on you, but you no longer see repeat clicks, and you can never change where that link goes for that browser.
- **302 Found** (temporary): every click comes back to you. More load, but you can count clicks, expire links, block a link that turns out to be malicious, and change the destination later.

Both are defensible; lesson 15.03 argues for 302 when analytics or control matter.

## Common mistakes

- **Truncating a hash with no collision handling**, then silently overwriting another user's link.
- **Checking for a free code without a unique index**, leaving a race between two servers.
- **Sequential codes for private links**, letting anyone enumerate them.
- **Indexing the full long URL**, which can be thousands of bytes; index a hash of it.
- **Picking 6 characters** without doing the 10-year arithmetic.

## In an interview

Do the arithmetic out loud (writes per second, total records, 62^7), then present both code-generation options with their trade-offs and choose Base62 of a unique ID, scrambled if links should not be guessable. Mention the unique index as the final guard, and say which redirect status you would use and why.

<div class="take">
<div class="lab">Key takeaway</div>

100M URLs a day needs 7 Base62 characters for 10 years. Hashing and truncating creates birthday-problem collisions that need checks, retries and a unique index; Base62 of a unique ID cannot collide but is guessable unless the ID is scrambled first. One SQL table with a unique key serves a single machine fine until storage forces a split.

</div>
