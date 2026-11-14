---
id: "11.08"
course: "system-design-101"
module: "11"
slug: "reaction-modeling"
title: "Reaction modeling"
keyIdeas: ["Reactions: store one row per (post_id, user_id, reaction_type), unique on (post_id, user_id) if one reaction per user.", "Showing counts by scanning rows is too slow for popular posts, so keep a denormalized count per post and type, updated by increments or periodic aggregation.", "\"Did I react?\" is a point lookup on (post_id, user_id), or a per-user set of recently reacted post IDs in cache.", "Hot posts make the count row a write hotspot: use sharded counters or buffer increments.", "Changing a reaction means decrementing one count and incrementing another, ideally atomically."]
question: "Store reactions per post. How do you show counts fast and still know if I reacted?"
answer: "Store each reaction as a row keyed by (post_id, user_id), which guarantees one reaction per user per post and makes changing or removing it an update of that row, and keep per-post counts per reaction type in a separate counter, cached and updated by events emitted only when a reaction row actually changes (batched or sharded for viral posts). To render a page, read counts for all posts with one multi-get from the cache and the viewer's own reactions with one batch query on a (user_id, post_id) index or user-keyed copy, optionally skipping lookups with a per-user Bloom filter. Show the user's own tap immediately on the client, and reconcile counters against the rows periodically to fix drift."
sources: [{"title": "Hello Interview: Scaling writes", "url": "https://www.hellointerview.com/learn/system-design/patterns/scaling-writes", "note": "Sharded counters and batching for hot rows."}]
---

## Two different questions

Every post in a feed shows reaction counts ("❤ 1.2K 😂 340") and whether **you** reacted (your heart is filled in). Those look like one feature but are two different access patterns:

- **Counts**: read on every view of a post, by everyone. Must be fast; can be slightly stale. Nobody notices 1,200 versus 1,203.
- **My reaction**: per viewer, per post. Must be correct for that viewer: if I tap heart, it must stay filled when I scroll away and back, and tapping again must remove it, not add a second heart.

A good model serves each pattern from data shaped for it, and keeps the two consistent. The question is how to do that when a feed page shows 20 posts, each needing both answers, and a viral post receives tens of thousands of reactions per second.

## The data model

1. **Reactions table**, the source of truth: one row per (post_id, user_id) with the reaction type and time. The primary key (post_id, user_id) means a user has **at most one reaction per post**, and changing it (heart to laugh) is an update of that row, not a second row. Platforms that allow several reaction types per user per post (as Slack does on messages) put the type in the key: (post_id, user_id, type).
2. **Counts**, stored separately: `post_reaction_counts(post_id, type, count)`, or a small map of counters on each post record. This is a **denormalised** copy: derived from the reactions table, stored separately so reads never have to compute it.

Never compute counts with `COUNT(*) ... GROUP BY type` over the reactions table on each view. A post with 2 million reactions would scan 2 million index entries every time anyone scrolls past it.

~~~dg-compare Computing counts on every read versus maintaining a counter on every write. Reads outnumber reactions by orders of magnitude, so the counter wins.
[Count on read]
View -(COUNT GROUP BY type)-> Reactions table
+ Always exact
+ Nothing to keep in sync
- Scans every reaction on every view
- 2M reactions: millions of entries per view
[*Counter on write]
Reaction -(+1 or -1)-> Counts row -> Cache
+ Reading counts is one key lookup
+ Cacheable for every viewer
- Must stay in sync with the rows
- A hot post makes one hot counter
~~~

~~~dg-flow A reaction updates the source-of-truth row, then the denormalised counter; views read the cached counter plus a point lookup for "did I react".
User taps heart -(upsert post 9, user 4)-> *Reactions table -(increment, async or same transaction)-> Counts per type -> Cache -> Post view
~~~

### What one viewer can do

Each viewer's reaction on a post is a tiny state machine. Modelling it explicitly tells you exactly which counter changes each transition must make.

~~~dg-state One viewer's reaction on one post. Each transition names the counter changes it causes; tapping the same reaction again is a no-op, not a second count.
[*] -> None
None -> Heart : tap heart: heart +1
Heart -> Laugh : switch: heart -1, laugh +1
Laugh -> None : remove: laugh -1
Heart -> None : remove: heart -1
None -> *Laugh : tap laugh: laugh +1
~~~

## Reading "did I react?"

For a feed of 20 posts, the viewer's reactions are a **batch point lookup**:

`SELECT post_id, type FROM reactions WHERE user_id = ? AND post_id IN (…20 ids…)`

For that to be one fast query, the data must be indexed or partitioned by (user_id, post_id), which is the **reverse** of the primary key (post_id, user_id). This is the same two-direction problem as follows in lesson 11.03: one direction serves "who reacted to this post" (rarely needed, for the "liked by" list), the other serves "what did I react to" (needed on every feed render). Options:

- A **secondary index** on (user_id, post_id) on a single database.
- A **second table** keyed by user, written alongside the first, when sharded (the primary table sharded by post_id, the second by user_id).
- A **cache** of "posts this user reacted to recently", since feeds mostly show recent posts.

For very large systems, a per-user **Bloom filter** (lesson 04.13) of reacted post IDs can answer "definitely not reacted" without a lookup, falling back to the store only when the filter says "maybe". Most posts in a feed are ones the viewer hasn't reacted to, so most lookups are skipped.

<div class="ex">
<div class="lab">Worked example</div>

**rendering a feed**

The feed service has 20 post IDs. It fetches counts for all 20 from the counts cache (one multi-get, about 1 ms) and the viewer's reactions with one query on the (user_id, post_id) index, returning 3 rows (about 2 ms). Each post renders its counts, and 3 of them show the viewer's filled-in reaction.

The viewer taps heart on post 9: the upsert inserts a row, an event increments post 9's heart count within a couple of seconds, and the client already shows the heart filled and 1,201 instead of 1,200, without waiting.

</div>

## Writing at scale: hot posts

A viral post may get tens of thousands of reactions per second. The reaction **rows** are spread out (each user's row is separate and lands in a different place), but the **counter** is a single hot row: the problem of lesson 10.01. The same fixes apply:

- **Batch increments**: push reaction events onto a queue partitioned by post ID; a consumer sums them for a second or two and applies `count += n`.
- **Sharded counters**: split a post's count into N sub-counters with random suffixes, summed on read and cached.
- **Hierarchical aggregation**: servers count locally and forward partial sums, as with live reactions (lesson 10.13).

~~~dg-flow Hot-post counting: reaction events are summed per post for a couple of seconds, so the counter sees one write per interval.
Reactions, 20k per s -(partition by post)-> Queue -> *Aggregator sums per 2 s -(count += 40,000)-> Counter row -> Cache
~~~

The reactions table itself also needs care for a viral post if it is partitioned by post_id: 20,000 inserts per second into one partition. Wide-column stores handle this better than a single B-tree page, but for the most extreme cases that post's reaction rows can be split across several partitions with a suffix, since "list everyone who reacted" is rarely needed and can merge a few partitions.

## Keeping them consistent

- **Idempotency.** A double tap or a retried request must not double-count. Because the reactions table enforces one row per user per post, increment the counter **only when the upsert actually inserted or changed** a row. In PostgreSQL, `INSERT ... ON CONFLICT (post_id, user_id) DO UPDATE SET type = EXCLUDED.type WHERE reactions.type IS DISTINCT FROM EXCLUDED.type RETURNING (xmax = 0) AS inserted` tells you whether a row was inserted, updated or untouched. Alternatively, emit counter events from change data capture (CDC) on the reactions table, so only real row changes produce events.
- **Changing a reaction** (heart to laugh) is decrement heart, increment laugh, derived from the old and new type of the row.
- **Removing a reaction** deletes the row (or marks it removed) and decrements.
- **Drift.** Counters maintained by events can drift after bugs, lost events or replays. A periodic job recounts from the reactions table for recently active posts and fixes discrepancies. Counts for old posts rarely change, so the job only needs to look at posts with recent activity.
- **Read-your-own-write.** The client shows the user's own reaction (and adds one to the displayed count) immediately, even before the aggregated counter catches up. On refresh, the "did I react" lookup comes from the reactions store or its user-keyed copy, which is updated synchronously, so the filled heart never disappears.

~~~dg-seq A double tap on a slow network: the second request finds the row unchanged, so the counter is incremented exactly once.
Client > Reaction API : heart on post 9 (request 1)
Reaction API > Reactions DB : upsert (9, user 4, heart)
Reactions DB -> Reaction API : inserted
Reaction API > Queue : heart +1 for post 9
Client > Reaction API : heart on post 9 (retry)
Reaction API > Reactions DB : upsert (9, user 4, heart)
Reactions DB -> Reaction API : unchanged
Reaction API -> Client : ok, no event emitted
Queue > Aggregator : heart +1, applied once
~~~

<div class="ex">
<div class="lab">Worked example</div>

**switching a reaction under load**

User 4 has a heart on post 9 and switches to laugh. The upsert changes the row's type from heart to laugh and returns the old type. The API emits two events: heart −1, laugh +1, both keyed by post 9. The aggregator, summing over its 2-second window, might combine them with 9,000 other events into `heart += 6,212, laugh += 1,031`. A second later the cached counts reflect the change; the user saw it instantly on their own screen.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing reactions storage**

500 million posts a day, an average of 20 reactions each: 10 billion reaction rows a day. At 30 bytes each (post ID, user ID, type, time) plus index overhead, budget 60 bytes per row per copy, and two copies (by post and by user): 1.2 TB a day. That's why many systems age out the user-keyed copy (keep "what I reacted to" for 90 days, the window feeds actually show) and archive old reaction rows, while the counters, a few bytes per post per type, are kept forever.

</div>

## Common mistakes

- **Counting on read.** Fine in a prototype; a viral post makes every feed view slow.
- **Incrementing on every request.** Without checking whether the row changed, double taps and retries inflate counts.
- **One index direction only.** Keying only by post makes "did I react to these 20 posts?" a scatter across 20 partitions; keying only by user makes "who reacted" a full scan.
- **A single hot counter row** for a viral post.
- **Hiding the user's own reaction** until the backend catches up, which makes the button feel broken.

## In an interview

Separate the two questions first. Then: "a reactions table keyed by (post, user) is the truth and enforces one reaction per user; per-type counters are denormalised, cached and updated by batched events only when a row really changed; 'did I react' is a batch lookup on a (user, post) index or copy; hot posts get batched or sharded counters; a periodic job reconciles". Mention optimistic UI for the user's own taps.

<div class="take">
<div class="lab">Key takeaway</div>

keep one reaction row per (post, user) as the source of truth and maintain per-type counts separately. Read counts from a cache and the viewer's own reactions by a batch lookup on a (user, post) index or copy. For hot posts, batch or shard the counter increments, increment only when a row actually changed, show the user's own reaction optimistically, and periodically reconcile counts against the rows.

</div>
