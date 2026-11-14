---
id: "11.03"
course: "system-design-101"
module: "11"
slug: "social-network-database-modeling"
title: "Social network database modeling"
keyIdeas: ["Core social tables: users, posts, follows (follower_id, followee_id), likes, comments.", "Store follows so both directions are fast: index on (follower_id, followee_id) and on (followee_id, follower_id), or store two copies of each edge.", "Denormalize counters (follower_count, like_count) to avoid counting on every read.", "Timelines are often precomputed (fanout on write) into a per-user feed store.", "Shard by user ID for user-centric data; posts are often stored by author."]
question: "Model follows so both \"who I follow\" and \"who follows me\" are fast."
answer: "Use an edges table keyed by (follower_id, followee_id), which answers \"who do I follow\" as a range scan, \"does A follow B\" as a point lookup, and prevents duplicate follows, plus an index or second table keyed by (followee_id, created_at, follower_id) for \"who follows me\", newest first and pageable. When sharded, keep two copies of each edge, one in a following table partitioned by follower and one in a followers table partitioned by followee, written together through a transactional outbox or retrying queue with idempotent upserts, so each query touches a single shard instead of scattering to all of them. Store follower and following counts as counters rather than counting rows, batching increments for celebrities."
sources: [{"title": "System Design Primer: Twitter timeline solution", "url": "https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/twitter/README.md", "note": "Tables and services for a social app."}]
---

## What has to be fast

A social graph is a set of directed edges: "A follows B". (Friendships, as on Facebook, are undirected: two edges, or one edge stored in both directions. Follows, as on Twitter, Instagram or GitHub, are directed: A can follow B without B following A.) The graph itself is simple. What makes it a design problem is that two opposite queries must both be fast, at enormous volume:

- **Who do I follow?** The outgoing edges of A. Needed to build A's feed on read, to show A's "following" list, and to answer "does A follow B?" when rendering a follow button.
- **Who follows me?** The incoming edges of B. Needed to deliver B's posts to followers (fan-out on write, lesson 11.05), to show follower counts and lists, and to notify followers.

Twitter-scale numbers show how heavily these queries are used: **100 million active users**, **500 million tweets per day**, an average of **10 fan-out deliveries per tweet** (5 billion per day), and **250 billion reads per month**, about **100,000 reads per second** and **60,000 fan-out deliveries per second**. Every tweet triggers a "who follows whoever wrote this?" lookup, and every timeline needs "who does this user follow?".

The access pattern is extremely read-heavy. People follow or unfollow a few times a week; the graph is read on every tweet and every feed load. That asymmetry justifies spending extra work and storage on writes to make both reads cheap.

## A relational model

```
CREATE TABLE follows (
  follower_id  BIGINT NOT NULL,
  followee_id  BIGINT NOT NULL,
  created_at   TIMESTAMP NOT NULL,
  PRIMARY KEY (follower_id, followee_id)
);
CREATE INDEX follows_by_followee ON follows (followee_id, created_at DESC, follower_id);
```

How each piece earns its place:

- The **primary key** (follower_id, followee_id) is a B-tree sorted by follower first. "Who does A follow?" is a range scan over the entries starting with A, reading exactly A's edges and nothing else. The same key answers "does A follow B?" as a single point lookup, and it **prevents duplicate follows** for free: inserting (A, B) twice violates the key.
- The **second index** (followee_id, ...) is the same data sorted the other way. "Who follows B?" becomes a range scan on followee_id = B.
- Putting created_at in the index lets "most recent followers first" come straight off the index in order, with keyset pagination (lesson 04.12) for long lists, rather than sorting millions of rows.
- An `ON CONFLICT DO NOTHING` insert makes "follow" idempotent: a double tap or a retried request is harmless.

~~~dg-flow One edge, two access paths: the primary key serves "following", the reverse index serves "followers".
[Who A follows] follower_id = A -(primary key scan)-> *B, C, D
[Who follows B] followee_id = B -(reverse index scan)-> *A, E, F
~~~

On one machine this is all you need, and it goes surprisingly far. Each edge costs roughly 24 bytes of key data per index plus overhead, so a billion edges is on the order of 100 GB across the table and its index, which a single large PostgreSQL server can hold with the hot parts in memory.

### Why not a graph database?

Graph databases (Neo4j and similar) shine at **multi-hop** traversals: friends of friends, shortest paths, pattern matching across many hops. Social products mostly need **one hop**: my followees, my followers, do we follow each other, whom do we both follow. One-hop queries on sorted adjacency lists are exactly what a B-tree or a key-value store does well, and those systems shard and replicate in well-understood ways. That is why large social networks built their own simple adjacency stores (lesson 11.04) rather than adopting general graph engines. Multi-hop features ("people you may know") are usually computed offline in batch jobs.

## When it no longer fits on one machine

Sharding breaks the neat picture. A **shard key** decides which machine each row lives on. If you shard the follows table by follower_id, all of A's outgoing edges sit together on A's shard, so "who does A follow?" hits one shard. But B's incoming edges are scattered: every shard may hold some follower of B. "Who follows B?" must ask **every** shard (a scatter-gather) and merge the answers. Shard by followee_id instead, and the opposite query becomes the scattered one.

~~~dg-matrix Which shards each query touches under each layout. Storing the edge twice is the only layout where both common queries hit one shard.
Layout | Who A follows | Who follows B | Does A follow B | Writes per follow
One table, sharded by follower | yes, one shard | no, every shard | yes, one shard | 1
One table, sharded by followee | no, every shard | yes, one shard | yes, one shard | 1
Two tables, each sharded by its own key | yes, one shard | yes, one shard | yes, one shard | 2
~~~

Scatter-gather is tolerable for a rare admin query, but not for a lookup that runs 60,000 times per second for fan-out. With 256 shards, each fan-out lookup would become 256 queries, most returning nothing, and the slowest shard sets the latency.

The standard fix is to **store each edge twice**: once in a `following` table sharded by follower_id, and once in a `followers` table sharded by followee_id. Each query hits exactly one shard. Twitter's FlockDB did exactly this (lesson 11.04).

~~~dg-tree Each follow edge is written to two places: the following list on the follower's shard and the followers list on the followee's shard.
Edge: A follows B
  following table | sharded by follower_id
    *Shard of A | row (A, B, time)
  followers table | sharded by followee_id
    *Shard of B | row (B, A, time)
~~~

In DynamoDB the same idea looks like this: a Follow table with partition key `userFollowing` and sort key `userFollowed`, plus a **global secondary index** with the keys swapped. The GSI is DynamoDB maintaining the second copy for you, asynchronously, so a just-made follow can take a moment to appear in the follower list. In Cassandra, you'd create two tables (`following_by_user` and `followers_by_user`) and write both, often with a logged batch.

### Keeping two copies consistent

Two writes to two shards can't usually be one local transaction. If the first succeeds and the second fails, A appears in B's followers but B isn't in A's following, or vice versa. Options:

- **Transactional outbox**: write the first copy and an "edge created" event in the same local transaction on A's shard; a relay publishes the event, and a consumer writes the second copy, retrying until it succeeds (lesson 08.03 covers the outbox pattern). The copies converge within seconds.
- **Idempotent writes in both places**, driven by a queue that retries until both exist. Because inserts are keyed by (follower, followee), retries are harmless.
- **Periodic repair**: a background job compares the two copies for recently changed users and fixes differences.

~~~dg-seq A follow with a transactional outbox: the first copy and an event commit together, and a consumer writes the second copy, retrying until it exists.
Client > Graph API : follow(A, B)
Graph API > Shard of A : BEGIN; insert following(A, B); insert outbox event; COMMIT
Shard of A -> Graph API : ok
Graph API -> Client : 200 following
Relay > Shard of A : read outbox
Relay > Queue : edge_created(A, B)
Queue > Followers writer : edge_created(A, B)
Followers writer > Shard of B : upsert followers(B, A); counts +1
~~~

Unfollow is the same, with deletes (or a "removed" state, so that late-arriving creates can't resurrect a deleted edge: compare timestamps, or keep tombstones).

## Counts and hot users

- **Follower counts** shouldn't be computed with `COUNT(*)` over millions of rows on every profile view. Keep a counter per user (`follower_count`, `following_count`), updated with each follow and unfollow, and accept that it may be briefly off. For a celebrity gaining thousands of followers a minute, batch the counter updates as in lesson 10.01.
- **Celebrities** have huge follower lists. Their incoming-edge list may be hundreds of millions of rows. Page through it with a cursor, never load it all, and treat them specially in fan-out (lesson 11.05). Their followers list may also need to be split across several partitions so one shard isn't overwhelmed.
- **Follow limits.** Many networks cap following (Twitter famously at 5,000 for most accounts, with exceptions), which bounds the "who do I follow" side and protects against spam bots.
- **Cache the hot lists.** Keep follower lists in a memory cache (or Redis sets) for the fan-out service, and only keep feeds for users active in the last 30 days.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the graph**

300 million users, an average of 200 followees each: 60 billion edges. Stored twice: 120 billion rows. At about 30 bytes per row with key and timestamp, plus storage overhead, call it 50 bytes: 6 TB.

Spread over 256 shards: about 23 GB per shard, comfortably in memory on each. Reads: 100,000 per second spread by user ID is under 400 per second per shard, trivial. Writes: if each user changes a follow twice a week, that is 600 million per week, about 1,000 per second, so 2,000 row writes per second including both copies, about 8 per shard per second.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**A follows B**

Single database: insert (A, B, now) into follows; both indexes update in the same transaction, and the follower and following counters are incremented in the same transaction.

Sharded: write (A → B) into following on A's shard and (B ← A) into followers on B's shard, plus increment A's following count and B's follower count. Use the outbox so a failure between the two writes is retried until both exist. If B is a celebrity gaining 5,000 followers a minute, B's counter is incremented by a batching consumer once a second instead of 5,000 times.

</div>

## Common mistakes

- **One table, one shard key, then scatter-gather.** Fine at 10 shards in a demo, unworkable at hundreds under fan-out load.
- **COUNT(*) for follower counts.** A celebrity's profile view should not scan 100 million rows.
- **Non-idempotent follow.** Without the (follower, followee) key, double taps create duplicate edges and inflate counts.
- **Loading whole follower lists.** Always paginate; a 100-million-follower list is gigabytes.
- **Reaching for a graph database for one-hop queries.** It adds a new operational system for queries a key-value layout handles better.

## In an interview

Start with the two queries and the read/write ratio. Draw the edge table with its primary key and reverse index, then say what changes when sharded: "store each edge twice, partitioned by each end, written through an outbox so both copies converge; counters kept separately and batched for celebrities". Ask whether the relationship is directed (follows) or symmetric (friends) and whether there's a follow cap, because both change the sizes.

<div class="take">
<div class="lab">Key takeaway</div>

store follows as edges with a composite primary key (follower, followee) and a reverse index (followee, follower), so both directions are index range scans and duplicates are impossible. Once sharded, store each edge twice, partitioned by each end and written through a reliable outbox or queue, and keep follower and following counts as separate counters.

</div>
