---
id: "04.02"
course: "system-design-101"
module: "04"
slug: "document-vs-key-value-stores"
title: "Document vs key-value stores"
keyIdeas: ["**Key-value stores** (Redis, DynamoDB in simple use): get/put by key, values are opaque. Fastest and simplest; queries only by key.", "**Document stores** (MongoDB, Firestore): values are JSON-like documents. You can query and index fields inside documents, and nest related data.", "Embed data that's read together and owned by one parent (order with line items). Reference data shared across many parents (products).", "Pick key-value for caches, sessions, counters and simple lookups. Pick document when you need to query by fields other than the key and the data is naturally hierarchical."]
question: "When would you pick a document store over a key-value store?"
answer: "Pick a document store when you need to query, filter or index by fields inside the record rather than only fetch it by its key (for example \"find all carts containing product 123\", or products by brand and price), when you want to update individual fields atomically without rewriting the whole value, or when records are nested objects with varying shapes that are read together, like a product catalogue or an order with its line items. If you only ever fetch and replace the whole value by one key, such as a session, cache entry or rate-limit counter, a key-value store like Redis is simpler and faster."
sources: [{"title": "System Design Primer: NoSQL", "url": "https://github.com/donnemartin/system-design-primer#nosql", "note": "Key-value, document, wide column, graph."}, {"title": "MongoDB docs: Data modeling", "url": "https://www.mongodb.com/docs/manual/data-modeling/", "note": "Embedding vs referencing."}, {"title": "Redis docs: Data types", "url": "https://redis.io/docs/latest/develop/data-types/", "note": "What a rich KV store offers."}]
tools: ["MongoDB", "Redis"]
---

## Key-value stores

A **key-value store** is a giant dictionary with three core operations: `get(key)`, `set(key, value)` and `delete(key)`. The **value is opaque** to the database: it is a blob of bytes, and the store never looks inside it. That is the whole contract, and it is why key-value stores are so fast and so easy to scale. Every operation touches exactly one key, so there are no joins, no query planning and no cross-key coordination. To spread data over many machines, you hash the key to pick a machine (lesson 04.07), and every request goes to exactly one place.

The price is that **all the logic lives in the application**. You can only find data if you know its key. "Find all sessions for users in Germany" is impossible without reading every value.

Some key-value stores keep keys **sorted** instead of hashed (RocksDB, FoundationDB, Bigtable). That allows range scans: "all keys between `user:42:` and `user:42;`". Choosing key names carefully then becomes a way to model data, which lesson 04.03 develops further.

### Redis: a data structure server

Redis is the best-known key-value store, and it is really a **data structure server**: values can have types, and each type has its own atomic operations that run inside the server. Because Redis executes commands one at a time on a single thread, each command is atomic without locks.

~~~dg-matrix Redis value types and the jobs they are typically used for.
Type | Atomic operations | Typical use
String | GET, SET, INCR, SETEX | Cache entries, counters, rate limits
Hash | HGET, HSET, HINCRBY | An object's fields, user:42 to name, plan
List | LPUSH, RPOP, BLPOP | Simple queues, recent items
Set | SADD, SISMEMBER, SINTER | Unique members, tags, who is online
Sorted set | ZADD, ZRANGE, ZINCRBY | Leaderboards, ranking, time-ordered feeds
Stream | XADD, XREADGROUP | Append-only event log with consumer groups
Geo | GEOADD, GEOSEARCH | Drivers within 2 km
HyperLogLog | PFADD, PFCOUNT | Approximate distinct counts in 12 KB
~~~

A sorted set is a good example of why types matter. A leaderboard stored as one opaque blob would need "read the whole blob, update one score, re-sort, write it back", with a race between concurrent updates (lesson 03.06). As a sorted set it is `ZINCRBY leaderboard 50 player:7` and `ZREVRANGE leaderboard 0 9`, each atomic and O(log n).

Redis keeps everything in memory, so a read takes tens of microseconds inside the server and well under a millisecond over the network. It can persist to disk with snapshots or an append-only log, but the dataset must fit in RAM (or be spread across a Redis Cluster).

### DynamoDB: key-value with a sort key

DynamoDB is a managed key-value and document store. Each item is found by a **partition key**, which is hashed to pick a storage partition, and optionally a **sort key**, which orders items within that partition. A table with partition key `user_id` and sort key `order_date` can answer "user 42's orders in March" as one efficient range read. Items can hold nested attributes and DynamoDB can filter on them, but efficient access is always by key; other access patterns need **secondary indexes**, which are copies of the data keyed differently.

~~~dg-seq A key-value store only understands keys. The application serialises and parses the value itself.
App > KV store : GET cart:42
KV store -> App : 1.2 KB of opaque bytes
App > App : parse JSON, add item
App > KV store : SET cart:42 new bytes, TTL 30 days
KV store -> App : OK
~~~

## Document stores

A **document store** keeps self-contained, JSON-like **documents**, grouped into **collections**. The crucial difference from a key-value store is that **the database understands the structure** of the value. You can query fields inside documents (`find({"address.city": "Pune"})`), create indexes on them, update one field without rewriting the document (`$inc` a counter, `$push` to an array), and project only the fields you need.

Documents in one collection can have different fields, so adding a field does not need a migration. MongoDB stores documents in a binary JSON format called BSON, which adds types such as dates and 64-bit integers.

### Modelling: store together what is read together

The central rule of document modelling is: **data that is accessed together should be stored together**. Instead of normalising into many tables and joining at read time (lesson 03.01), you shape each document around the screen or API call that reads it. You choose, for each relationship, between two options.

**Embedding** puts the related data inside the document: an order with its line items and shipping address, a blog post with its tags. One read returns everything, with no joins. Updates to a single document are **atomic**, so an order and its items can never be half-written.

**Referencing** stores another document's ID instead, like a foreign key. Use it when the related data is **large**, **shared** by many documents, or **unbounded** in number. The application (or MongoDB's `$lookup` aggregation stage) fetches the referenced documents in a second step.

~~~dg-tree An order document with embedded line items and address, and a reference to the customer, who is shared by many orders.
order 9001
  customer_id | reference to customers collection
  status | shipped
  placed_at | 2026-03-04T10:12Z
  *items | embedded array, read with the order
    item 1 | sku A17, qty 2, price 19.99
    item 2 | sku B03, qty 1, price 5.00
  shipping_address | embedded snapshot at order time
    city | Pune
    postcode | 411001
~~~

A useful middle path is the **subset pattern**. A product page shows the five newest reviews. The product document embeds those five, while all reviews, possibly thousands, live in their own collection, referenced by product ID. The page loads with one read; the "all reviews" page queries the reviews collection.

~~~dg-compare Embedding versus referencing related data in a document store.
[*Embed]
Order -(contains)-> Line items
+ One read returns everything
+ Atomic single-document update
- Document grows with the data
- Duplicates shared data
[Reference]
Order -(customer_id)-> Customer document
+ Shared data stored once
+ Unbounded relationships OK
- Second query or $lookup
- No atomicity across documents by default
~~~

### Limits to know

- **Document size cap**: 16 MB in MongoDB. An unbounded array, such as every comment on a viral post embedded inside that post document, will eventually hit it, and long before that every update rewrites a huge document.
- **Atomicity is per document** by default. Multi-document transactions exist in MongoDB since version 4.0, but they are slower and limited, and the modelling advice is still to design so most writes touch one document.
- **Duplicated data must be updated everywhere it lives.** If the customer's name is embedded in 300 orders, a name change is 300 updates, or a decision that orders keep the name as it was at order time (which is often the right business rule anyway, as with the address snapshot above).
- **Indexes still matter.** A query on a field without an index scans the whole collection, exactly like a relational sequential scan.

## How to choose

Pick a **key-value store** when:

- You always fetch by one key, and use the value whole.
- You need extreme speed (in-memory, sub-millisecond) and simple operations: caching, sessions, rate-limit counters, leaderboards, feature flags.
- Values can expire automatically (TTL) and losing them is tolerable or recoverable.

Pick a **document store** when:

- You need to **query or index by fields inside the value**, not just the key.
- Records are naturally nested objects that are read together (a product with variants, a user profile with settings, a form submission).
- Different records have genuinely different shapes, such as a catalogue where shoes have sizes and laptops have RAM and CPU.

A caveat for interviews: requirements rarely involve rapidly changing schemas, which removes the main reason to choose a document store over Postgres. A relational table with a `JSONB` column and a GIN index on it often gives you both: relational integrity for the core fields and flexible, queryable attributes for the rest.

<div class="ex">
<div class="lab">Worked example</div>

**a shopping cart, two ways**

Key-value: `cart:{user_id}` holds the serialised cart, read and written whole, with a 30-day TTL. With 20 million active carts at 1.5 KB each, that is 30 GB, which fits in a small Redis cluster. Each page view is one `GET` taking about 0.3 ms end to end.

New requirement: when product 123 drops in price, notify everyone with it in their cart. With opaque values, the only way is to read and parse all 20 million carts: 30 GB of transfer for one question.

Document store: carts are documents with `items: [{product_id, qty}]` and an index on `items.product_id`. "Find carts containing product 123" is one indexed query returning, say, 4,000 carts in a few milliseconds. The requirement decided the store: querying inside the value is exactly what a document store adds.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**when embedding goes wrong**

A social app embeds every comment inside its post document. A typical post has 20 comments of 300 bytes: a 6 KB document, fine.

A celebrity post receives 200,000 comments. The document would reach 60 MB, beyond the 16 MB cap, and long before that each new comment rewrites megabytes and every feed read downloads every comment.

The fix is the subset pattern: embed the latest 10 comments and a `comment_count`, and store all comments in a `comments` collection indexed by `(post_id, created_at)`. Feed reads stay small; the full thread is paged from the comments collection (lesson 04.12).

</div>

## Common mistakes

- **Using a key-value store, then needing secondary queries**, and scanning everything to answer them.
- **Embedding unbounded arrays** in documents.
- **Normalising a document store like SQL**, with references everywhere, and doing joins by hand on every read.
- **Forgetting indexes in a document store** because "it's NoSQL, it's fast".
- **Treating "schemaless" as "no schema"**: the schema is still there, now enforced (or not) by every piece of code that reads the documents.

## In an interview

When you choose between the two, say what you will query by. "Only by session ID" means key-value. "By user, and also by product inside the cart" means document store or relational with JSON. For a document store, say what you embed and what you reference, and why: read together, bounded size and not shared means embed.

<div class="take">
<div class="lab">Key takeaway</div>

key-value stores fetch an opaque value by key, extremely fast, and know nothing else. Document stores understand the value, so you can query, index and partially update fields inside it. In documents, embed what is read together and bounded; reference what is large, shared or unbounded.

</div>
