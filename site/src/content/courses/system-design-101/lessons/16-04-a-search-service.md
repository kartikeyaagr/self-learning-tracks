---
id: "16.04"
course: "system-design-101"
module: "16"
slug: "a-search-service"
title: "A search service"
keyIdeas: ["A search service: ingestion pipeline (CDC from the DB → Kafka → indexer → search engine) and a query service (parse → retrieve → rank → return).", "New content becomes searchable in seconds through near-real-time indexing (Elasticsearch refreshes segments about once a second).", "Shard the index; replicate for reads and failover.", "Ranking: BM25 plus signals (recency, popularity, personalization), optionally re-ranked by a learned model.", "Cache popular queries, and protect the cluster from expensive queries (wildcards, deep pagination) with limits."]
question: "How do new posts become searchable within seconds?"
answer: "Indexing is asynchronous and incremental. When a post is created, the posts service saves it and publishes an event to Kafka, then returns; ingestion workers consume the event, tokenise the text and append the new post's ID to each keyword's index lists, typically within tens of milliseconds to a few seconds, without slowing post creation. Consumer lag is monitored against the freshness target and workers scale out during bursts. In Elasticsearch, new documents sit in a buffer protected by the translog and become searchable at the next refresh, about every second, when a new immutable segment is written. Result caches use a TTL shorter than the freshness requirement so they never hide new posts for long."
sources: [{"title": "Hello Interview: FB Post Search", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-post-search", "note": "Search service design."}, {"title": "Hello Interview: Elasticsearch deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/elasticsearch", "note": "The engine underneath."}]
tags: ["Design"]
---

## What we are building

Post search for a large social network, in the style of Facebook: type a keyword and get matching posts, sorted by newest or by most liked. It sounds like "put the posts in Elasticsearch", and at small scale it is. At the scale below, the interesting questions are how new posts become searchable within seconds, how you keep a sort by like count accurate when likes arrive at 100,000 per second, and where petabytes of old posts live.

### Functional requirements

- Users create posts and like posts.
- Users **search by keyword** and sort results by **recency** or by **like count**.
- Out of scope: fuzzy matching and typo correction, personalisation, privacy filtering (who is allowed to see which post), images.

### Non-functional requirements

- Median query latency under 500 ms.
- **New posts searchable within about 1 minute** (the freshness target).
- Old and unpopular posts must still be findable, even if more slowly.
- High availability; slightly stale like counts are fine.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**sizing the write and storage load**

10,000 new posts per second, 100,000 likes per second, 10,000 searches per second.

Posts per year: 10,000 x 86,400 x 365 is about 315 billion. Over 10 years, roughly 3.6 trillion posts (allowing for growth).

At about 1 KB per post, that is 3.6 PB of raw data. The index is smaller per post (IDs and scores), but each post appears under every distinct word it contains: a 30-word post creates about 20 index entries after removing duplicates and very common words.

So the index receives around 200,000 entry writes per second from new posts alone, and like updates would add another 100,000 x 20 = 2 million per second if every like touched every keyword. That last number is why likes need special treatment.

</div>

The ratio tells you two things straight away. Writes dominate (10,000 posts plus 100,000 likes against 10,000 searches), so ingestion must be off the critical path and horizontally scalable. And 3.6 PB will not fit in memory, so you need a hot and cold split.

## How search works underneath: the inverted index

A database row for a post is keyed by post ID: given an ID, find the text. Search needs the reverse: given a word, find the posts. That reversed structure is an **inverted index** (lesson 09.02): for each term, a **posting list** of the IDs of documents that contain it.

Building it means **tokenising** each post: split the text into words, lowercase them, strip punctuation, maybe reduce words to a stem ("running" to "run"), and drop extremely common words. Each remaining term gets the new post's ID appended to its list. A query for "pizza london" fetches the two posting lists and intersects them.

~~~dg-tree An inverted index for this design: one keyword entry points at two orderings of the same post IDs.
Index shard 17
  keyword pizza
    by time | list, newest first
    by likes | sorted set, score = likes
  keyword paris
    by time
    by likes
~~~

## The API

```
POST /posts {text}               -> post_id
POST /posts/{id}/like            -> 204
GET  /search?q=pizza&sort=recent|likes&cursor=...  -> 25 post IDs + next cursor
```

Search returns IDs and lets the client (or a hydration step) fetch the full post bodies from the posts service. That keeps index entries tiny.

## High-level design: an inverted index fed by a stream

~~~dg-flow Posts and likes flow through Kafka into keyword indexes; search reads the indexes.
[Ingest] Post service -(new post)-> *Kafka -(workers tokenise)-> Index shards
[Likes] Like service -(like events)-> Kafka -(milestone updates)-> Likes sorted sets
[Query] Search API -(read keyword lists)-> Index shards -(re-rank with fresh likes)-> Results
~~~

- **Index storage.** For each keyword, two structures in **Redis**: a list ordered by time (for recency) and a sorted set scored by likes (for popularity). Redis is chosen for speed: appending to a list or updating a sorted set is microseconds. Redis is not durable by default, so either use a durable variant (such as Amazon MemoryDB, which keeps a transaction log across zones) or be able to rebuild from Kafka.
- **Ingestion through Kafka.** The posts service writes the new post to its own database, publishes a `post_created` event, and returns to the user. A fleet of ingestion workers consumes the topic, tokenises the text and appends the new post's ID to each keyword's lists.
- **Shard by keyword.** Keywords are hashed across many Redis instances to spread writes. A query for one keyword hits one shard; a query for two keywords hits up to two shards and intersects.

### Why shard by keyword and not by document?

There are two ways to split an inverted index (lesson 09.11):

- **Document-partitioned**: each shard indexes a subset of posts for all words. Writes are local (a post goes to one shard), but every query must ask every shard and merge.
- **Term-partitioned**: each shard owns some words entirely. A query for one word goes to one shard, but a post's 20 words go to up to 20 shards.

~~~dg-compare Two ways to partition an inverted index. This design picks keyword (term) partitioning.
[Document-partitioned]
Post -> One shard
Query -(scatter)-> All shards -> Merge
+ One write per post
- Every query fans out to all shards
[*Term-partitioned]
Post -(20 terms)-> Up to 20 shards
Query -> One shard per term
+ Cheap single-term queries
- Hot keywords make hot shards
~~~

This design picks term partitioning because single-keyword queries dominate and the write fan-out is absorbed by asynchronous workers. Elasticsearch, by contrast, is document-partitioned. Both are valid; say why you chose one. A very hot keyword (a breaking news topic) can overload its shard, so you may split one hot term's list across several sub-shards or cache its results.

## How new posts become searchable within seconds

This is the lesson's question. The path is short and asynchronous:

~~~dg-timeline One post on its way into the index. The one-minute budget leaves plenty of slack for bursts.
0 ms | User posts, row saved, event published
5 ms | API returns to the user
50 ms | Worker reads event from Kafka
60 ms | Text tokenised into 20 terms
80 ms | *Post ID appended to 20 keyword lists
2 s | Search result caches expire
3 s | Post appears in fresh searches
~~~

Key choices that make this work:

- **Do not index inside the create-post request.** Publish an event and return. If the index is slow or down, posting still works, and the backlog waits safely in Kafka.
- **Index incrementally**: append to existing lists rather than rebuilding any index.
- **Keep result caches short-lived**: the search result cache TTL must be well **under 1 minute**, so a cached answer never hides new posts for longer than the requirement allows. The same applies to CDN caching via `Cache-Control: max-age`.
- **Monitor consumer lag** (lesson 08.04). The freshness target becomes an alert: if the ingestion workers fall more than 30 seconds behind Kafka, add workers.

<div class="ex">
<div class="lab">Worked example</div>

**a burst**

A major football match ends and posting jumps from 10,000 to 60,000 per second for three minutes. The ingestion fleet is sized for 20,000 per second with headroom.

For the first minute the backlog grows by 40,000 posts per second (60,000 in, 20,000 out): 2.4 million posts. Autoscaling then doubles the fleet to 40,000 per second, so for the remaining two minutes it grows by 20,000 per second: another 2.4 million.

The backlog peaks at about 4.8 million. When posting falls back to 10,000 per second, the fleet drains 30,000 per second of backlog and clears it in about 160 seconds. For roughly five minutes, freshness is one to two minutes instead of seconds. Posting itself was never slowed down, and no post was lost, because Kafka retained them all. This is the trade-off you are making on purpose.

</div>

### The same idea in Elasticsearch: near real-time search

Elasticsearch (built on Lucene) does this internally and calls it **near real-time** search:

- Incoming documents go into an **in-memory buffer** and are appended to a write-ahead log called the **translog**, which protects them if the node crashes.
- Every **refresh**, by default every **1 second**, the buffer is written out as a new **segment**: a small, self-contained inverted index. Only then are the documents searchable.
- Segments are **immutable**. Immutability makes writes fast (no in-place updates), caching easy (a segment never changes), concurrency simple (readers never see half-written data) and compression good. The cost is periodic **merges** of small segments into bigger ones, which take I/O and temporary extra disk.
- A **delete** is only a mark in a bitmap, cleaned up during merges, and an **update** is a delete plus a new insert.
- A **flush** (less frequent) fsyncs segments to disk and trims the translog.

~~~dg-flow Elasticsearch: documents become searchable when a refresh turns the buffer into a segment.
[Write] New post -(index request)-> Buffer + translog -(refresh about 1s)-> *New segment
[Read] Search -(query all segments)-> Includes new post -(background)-> Merge segments
~~~

For bulk loading, teams often set the refresh interval to 30 seconds or disable it, trading freshness for throughput, because each refresh creates a segment that must later be merged.

## Likes: too many writes to index exactly

Sorting by like count means each post's score must live inside the sorted set of every keyword in that post. 100,000 likes per second times about 20 keywords would be 2 million sorted-set updates per second, across many shards. Options:

- **Batch** likes per post over about 30 seconds and apply one update. That helps a viral post with 10,000 likes a minute enormously, but most posts get one or two likes, and batching does nothing for them.
- **Two-stage, approximate** (chosen): update a post's score in the index only when its like count crosses a **milestone**: 1, 2, 4, 8, 16, and so on (powers of 2), or powers of 10. A post with 1,000 likes needs only about 10 index updates in its lifetime instead of 1,000. The index order is then roughly right. At query time, fetch the top **2N** candidates by approximate score, fetch their exact counts from the like service, and re-rank to return the top N.

<div class="ex">
<div class="lab">Worked example</div>

**ranking by likes**

A post has 1,024 likes recorded in the index (its last milestone) but 1,900 actually. Another has 1,500 indexed and 1,500 actual.

A query for the top 10 fetches the top 20 by indexed score; both posts are in that set. Live counts are fetched in one batch call (20 IDs, about 2 ms), and the 1,900-like post is put first.

Few index writes, correct final order. The only risk is a post whose true count is far above its milestone and which falls outside the top 2N, which powers-of-2 milestones keep within a factor of two.

</div>

## Keeping storage in check

3.6 PB of posts cannot all live in Redis (memory costs tens of times more per gigabyte than object storage). Two techniques:

- **Cap each keyword's hot index** at about 1,000 to 10,000 entries. Most searches read only the first page or two, so the newest or most liked few thousand posts serve almost every query.
- Move indexes for **rarely searched keywords** and the long tail of old entries to **cold blob storage** (S3 or Cloudflare R2), stored as sorted files per keyword. Search checks Redis first and falls back to blob storage on a miss or when the user pages beyond the hot window. Slower, maybe hundreds of milliseconds, but old posts stay findable.

~~~dg-bars Rough cost per GB-month of each storage tier, which is why only the hot index sits in memory.
*Redis memory | 10 | about $10
Local SSD | 0.1 | about $0.10
Object storage | 0.023 | about $0.02
~~~

## When to reach for Elasticsearch

Elasticsearch is a **search index, not a primary database**. Keep the source of truth elsewhere and sync it in, often by change data capture from the database's log (lesson 09.10). Other properties to know:

- It suits read-heavy workloads; very high update rates (like counts) hurt because every update is a delete plus insert.
- Results are eventually consistent: a document is searchable only after the next refresh.
- Denormalise so one query answers a request; joins across indexes are weak.
- Deep pagination past about 10,000 results gets expensive; use `search_after` cursors.
- Under about 100,000 documents you may not need it at all; PostgreSQL full-text search is often enough.
- A query runs in two phases. The coordinating node sends it to every relevant shard (**query** phase: each shard finds matching IDs and scores, returns its top N), merges the results, then asks the owning shards for the actual documents (**fetch** phase).

## Common mistakes

- Indexing synchronously in the create-post request, so posting slows or fails when the index does.
- Caching search results for longer than the freshness target.
- Updating the popularity index on every like.
- Putting the entire history in memory.
- Using the search index as the system of record, so a corrupted index means lost data instead of a rebuild.

## In an interview

Do the numbers first; they justify Kafka, sharding and the hot and cold split. Draw the inverted index explicitly, because many candidates say "Elasticsearch" without being able to explain what it does. Then answer freshness with the asynchronous pipeline, consumer lag and cache TTLs, and likes with milestones plus re-ranking. Ask: is privacy filtering in scope (it changes everything, because results must be checked per viewer), and how far back must search go?

<div class="take">
<div class="lab">Key takeaway</div>

keep search off the write path. Publish new posts to a stream, let workers append them to an inverted index sharded by keyword, watch consumer lag, and keep result caches shorter than the freshness target. Elasticsearch does the same with immutable segments refreshed about every second. For fast-changing counts like likes, update the index only at milestones and re-rank the top candidates with exact values, and keep only the hot part of the index in memory.

</div>
