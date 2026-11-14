---
id: "16.07"
course: "system-design-101"
module: "16"
slug: "a-social-feed"
title: "A social feed"
keyIdeas: ["Social feed: when a user opens the app, show a ranked list of recent posts from people they follow.", "Generate candidates with fanout on write for normal accounts (post IDs pushed into followers' feed caches) and pull for celebrities.", "Ranking happens at read time on a few hundred candidates, using predicted engagement, recency and diversity. Some features are precomputed.", "Feed caches hold IDs; post content, author info and counts are hydrated from caches.", "Paginate with a cursor; new posts that arrive while scrolling are shown via a \"new posts\" prompt."]
question: "Where does ranking happen in a feed, and what's precomputed?"
answer: "Ranking happens at read time: when the user opens the app, the feed service takes the precomputed candidates (post IDs fanned out to their feed list on write, plus recent posts from followed celebrities pulled at read time), fetches fresh features such as engagement in the last hour and whether the user has seen each post, scores the few hundred candidates with the ranking model, applies rules like author diversity, and hydrates the top page from the content, user and counter caches; the ranked order is cached for the session so paging stays stable. Precomputed ahead of time are the per-user lists of candidate post IDs, the social graph, slowly changing features such as user-author affinity and post embeddings, and the trained model itself."
sources: [{"title": "Hello Interview: FB News Feed", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-news-feed", "note": "Full design."}, {"title": "ByteByteGo: Design a News Feed System", "url": "https://bytebytego.com/courses/system-design-interview/design-a-news-feed-system", "note": "Alternative take."}]
tags: ["Design"]
---

## What we are building

A news feed in the style of Facebook, Instagram or X: users write posts, follow other users, and open the app to see a feed of posts from the people they follow. The feed must load fast, scroll smoothly, and feel fresh. Behind that simple screen sit three hard questions: how a post gets from its author to millions of feeds, where the feed for a user is stored, and where ranking happens.

### Requirements

- Users create posts and follow other users.
- Users see a **feed of posts from people they follow**, with paging (infinite scroll). We first design for newest-first, then add ranking.
- Scale: up to **2 billion users**, unlimited follows and followers (some accounts have over 100 million followers).
- Posting and loading the feed both under **500 ms**.
- A post may appear in followers' feeds up to **1 minute** late (eventual consistency is acceptable).
- A smaller variant to keep in mind: 10 million daily users, up to 5,000 friends each, on mobile and web.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**sizing the feed**

Assume 500 million daily active users open the feed 10 times a day: 5 billion feed loads a day, about 58,000 per second on average and perhaps 150,000 at peak.

Posting: 500 million users post 0.5 times a day on average, 250 million posts a day, about 3,000 per second.

Average followers per author: 200. Fanning every post out to every follower is 3,000 x 200 = 600,000 feed inserts per second. Heavy, but spread over a cache cluster of a few hundred nodes it is fine. One post by an account with 100 million followers would be 100 million inserts by itself, which is the problem that shapes the design.

</div>

Reads outnumber posts about 20 to 1, and every read wants many authors' posts at once. That points towards doing work at write time so reads are cheap, except where write-time work explodes.

## The API

```
POST /posts {text, media_ids}     -> post_id
POST /follows {followee_id}       -> 204
GET  /feed?cursor=...&limit=20    -> 20 hydrated posts + next cursor
```

The cursor is opaque to the client. It typically encodes the position in the feed (a timestamp or post ID, and for ranked feeds a session token), so the next page continues where the last one ended without duplicates. Offset paging (`?page=5`) breaks when new posts arrive at the top (lesson 04.12).

## Fan-out on read versus fan-out on write

Lesson 11.05 introduced the core trade-off:

- **Fan-out on read (pull)**: store posts by author. When a user opens the feed, fetch recent posts from everyone they follow and merge them. Nothing is wasted on inactive users, and a celebrity's new post is written once. But a read for someone following 2,000 accounts touches 2,000 lists and merges them, which is slow.
- **Fan-out on write (push)**: when someone posts, insert the new post's ID into each follower's precomputed **feed list**. Reads are a single list lookup, but a post by someone with 50 million followers means 50 million writes, and most of those followers will not open the app today.

The industry answer is the **hybrid**: push for most authors, and for accounts with huge follower counts, skip fan-out and merge their recent posts in at read time. A typical threshold is somewhere between 10,000 and 1 million followers; the exact number is a tuning knob.

~~~dg-compare Three ways to build the feed. The hybrid pushes for ordinary authors and pulls for celebrities.
[Pull]
Read -(N authors)-> Merge at read
+ Write once
- Slow reads for big follow lists
[Push]
Post -(N followers)-> Feed lists
+ One lookup per read
- Celebrity post = millions of writes
[*Hybrid]
Post -> Push if under threshold
Read -> Merge celebrity posts
+ Fast reads, bounded writes
- Two code paths
~~~

## The publishing path

1. The web server authenticates the user and rate-limits posting.
2. The **post service** saves the new post once to the posts database and the content cache.
3. The **fan-out service** gets that author's follower IDs from the social graph service, filtering out people who muted or blocked that author.
4. If that author is under the celebrity threshold, it puts batches of follower IDs plus the new post's ID on a **queue**.
5. **Fan-out workers** take each batch and insert the new post's ID at the head of each follower's **feed list** in the feed cache, trimming the list to its maximum length.

~~~dg-flow Publishing: each post is stored once; only its ID is pushed into followers' feeds.
[Store] Author -(POST /posts)-> Post service -(save once)-> Posts DB and cache
[Fan out] Post service -(not a celebrity)-> Queue -(workers)-> *Follower feed lists
[Skip] Post service -(celebrity)-> No fan-out, pulled on read
~~~

### Store IDs, not posts

The feed cache holds only **post IDs** (and perhaps author ID and timestamp), never the full post body. A body is stored once and can be edited or deleted in one place; IDs in a million feeds then point at the current version. And IDs are tiny.

<div class="ex">
<div class="lab">Worked example</div>

**how big is the feed cache?**

Keep the latest 200 entries per user. Each entry is a post ID (8 bytes) plus a little metadata, about 10 bytes. 200 x 10 = about 2 KB per user.

2 billion users x 2 KB = about 4 TB. Spread over Redis nodes with 64 GB each, that is around 60 to 100 nodes before replication. Cheap compared with the value of the feed.

In practice you only keep lists for users active in the last few weeks. An inactive user's feed is rebuilt by pull when they return, which can cut the cache by half or more.

</div>

## Reading the feed

The read path: get the user's list of post IDs from the feed cache, merge in recent posts from any celebrities they follow, then **hydrate**: fetch post contents and author details from the content and user caches, plus whether this user liked each one and the like and comment counts. Return a page and a cursor.

Five caches sit behind this, each keyed differently:

| Cache | Holds |
| --- | --- |
| News feed | post ID list per user |
| Content | post bodies; popular posts in a hot tier |
| Social graph | followers and following per user |
| Action | whether a user liked or commented on a post |
| Counters | likes, replies, follower counts |

Hydration is a scatter-gather: for 20 posts, one multi-get to the content cache, one to the user cache, one to the action cache and one to the counter cache, in parallel. At about 1 to 2 ms per multi-get, the whole page assembles in well under 50 ms.

## Where ranking happens

Real feeds are ranked, not chronological: they try to show the posts you are most likely to care about first. The standard place for ranking is **at read time, over a precomputed candidate set**, the same shape as the recommender in lesson 16.05.

**Precomputed (before the user opens the app):**

- The **candidate pool**: post IDs pushed into the user's feed list on write.
- **Slowly changing features**, computed in batch: affinity between this user and each author (how often they interact), author quality, post embeddings and topics.
- The trained ranking model itself.

**At read time (when the user opens the app):**

- Take the latest few hundred candidates from the feed list and merge in recent celebrity posts (pulled).
- Fetch **fresh signals**: like and comment counts in the last hour, whether the user has already seen that post, whether friends commented on it.
- **Score** each candidate with the ranking model (predicted probability of a like, comment, share, long view; combined into one value).
- Apply **rules**: diversity (no three posts in a row from one author), demote already-seen posts, insert ads at fixed slots.
- Hydrate the top page and return it.

~~~dg-flow The read path: a precomputed pool, merged and ranked when the user opens the app.
[Candidates] Feed list (pushed IDs) -(merge)-> *Candidate pool -(plus)-> Celebrity posts (pulled)
[Rank] Candidate pool -(fresh features)-> Ranking model -(diversity, seen)-> Page of 20
[Hydrate] Page of 20 -(content, user, counters)-> Feed response
~~~

### Why not rank at write time?

Ranking at write time would mean scoring a new post for each follower when it is published. That freezes the order before any engagement data exists (a post's first hour of likes is a strong signal), redoes work for every follower including the many who will not open the app, and cannot react to what the user has just seen. Ranking at read time uses current signals, and only for users who actually show up.

~~~dg-seq Opening the app: candidates from cache, scores from the ranker, content from caches.
App > Feed API : GET /feed
Feed API > Feed cache : latest 500 IDs for user 42
Feed API > Post store : recent posts of 3 celebrities
Feed API > Feature store : fresh signals for 520 posts
Feed API > Ranker : score 520 candidates
Ranker -> Feed API : top 20 after rules
Feed API > Caches : hydrate 20 posts
Feed API -> App : page 1 + cursor
~~~

To keep paging consistent, the ranked order for the current session is cached for a few minutes (keyed by user and session). Page 2 continues from that cached order rather than re-ranking, which would shuffle posts between pages and show duplicates. Some systems also cache the ranked first page for a minute or two so a user who reopens the app straight away gets an instant response.

## Edge cases and failure handling

### Hot posts

A viral post is read by millions of feeds at once. A content cache sharded by post ID puts that post on one shard: a **hot key** (lesson 04.06). The fix is to **replicate** hot content across many cache instances (or keep it in a small in-process cache on every feed server) so any instance can serve it. Detect hotness by counting reads per key over a short window.

### New follows and unfollows

When you follow someone, your feed list does not contain their old posts. A small backfill job inserts their last few posts. When you unfollow, filter their posts out at read time immediately and clean the list lazily.

### Fan-out lag

A post by an author with 9,000 followers (just under the threshold) creates 9,000 inserts. Batched 1,000 per message, that is 9 queue messages, processed in about a second. A burst during a big event might push fan-out lag to tens of seconds, which is inside the 1-minute requirement. Monitor queue lag as the freshness metric.

<div class="ex">
<div class="lab">Worked example</div>

**a celebrity posts**

An account with 80 million followers posts. It is above the threshold, so nothing is fanned out: one row in the posts table and an entry in that author's recent-posts list.

5 million followers open the app in the next 10 minutes. Each read merges the celebrity's recent posts list into its candidates: one cache read per celebrity followed. That is 5 million cheap reads of the same hot key, served from replicated copies.

With push instead, that post would have caused 80 million writes, 75 million of them for people who did not open the app in that window.

</div>

### Cache loss

If a feed cache node dies, the affected users' lists are rebuilt on demand by pull (fetch recent posts from the people they follow) and then cached again. Reads are slower for those users for a few minutes, not broken.

## Common mistakes

- Pure push, then discovering the celebrity problem in production.
- Storing full post bodies in every feed list.
- Ranking at write time.
- Offset pagination on a feed that changes.
- Re-ranking on every page, so posts jump between pages.
- No hot-key handling for viral posts.

## In an interview

Do the numbers, then present pull, push and hybrid with the celebrity threshold. Draw the publish path and the read path separately. For ranking, say clearly: candidates and slow features are precomputed, scoring happens at read time over a few hundred candidates. Ask: is the feed ranked or chronological, how fresh must it be, and what is the largest follower count?

<div class="take">
<div class="lab">Key takeaway</div>

store posts once and push only post IDs into followers' feed lists, except for celebrities whose posts are pulled at read time. Precompute the candidate pool and slow features; at read time merge candidates, rank them with fresh signals, apply diversity rules, then hydrate from caches. Feed lists are tiny (about 2 KB per user), so precomputing them is cheap; replicate hot posts to avoid hot cache keys.

</div>
