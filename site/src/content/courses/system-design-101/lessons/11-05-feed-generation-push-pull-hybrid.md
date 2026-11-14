---
id: "11.05"
course: "system-design-101"
module: "11"
slug: "feed-generation-push-pull-hybrid"
title: "Feed generation: push, pull, hybrid"
keyIdeas: ["**Push (fanout on write)**: when someone posts, write the post ID into each follower's feed cache. Fast reads, expensive for accounts with millions of followers.", "**Pull (fanout on read)**: when a user opens the feed, fetch recent posts from everyone they follow and merge. Cheap writes, slow reads for users following many accounts.", "**Hybrid**: push for regular accounts; for celebrities, skip push and merge their recent posts at read time.", "Feed caches store only post IDs (and maybe scores), not full posts; hydrate details at read time.", "Ranking happens after candidate generation, using engagement predictions."]
question: "Explain the hybrid model: who gets push and who gets pull?"
answer: "Accounts with ordinary follower counts get push: when they post, async workers prepend the new post's ID to every active follower's precomputed feed (skipping dormant users), so reads are a single lookup. Accounts above a tuned follower threshold, the celebrities, get pull: their posts are not fanned out at all, and when a user opens the feed, the service takes the precomputed feed and merges in the recent posts of the few celebrities that user follows, which are cheap to fetch because millions of readers hit the same cached timelines. This avoids tens of millions of writes per celebrity post, and the out-of-order delivery they cause, while keeping each feed read to one lookup plus a handful of cache fetches."
sources: [{"title": "ByteByteGo: Design a News Feed System", "url": "https://bytebytego.com/courses/system-design-interview/design-a-news-feed-system", "note": "Push, pull and hybrid."}, {"title": "Hello Interview: FB News Feed", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-news-feed", "note": "Interview breakdown."}]
---

## The core trade-off

A news feed (home timeline) is the newest posts from everyone you follow, merged into one list. The work of assembling it has to happen at some point, and there are exactly two choices:

- **Fan-out on read (pull)**: when a user opens the feed, look up whom they follow, fetch each one's recent posts, merge and sort. Nothing is precomputed.
- **Fan-out on write (push)**: when someone posts, insert the new post's ID into each follower's precomputed feed. Reading a feed is one lookup.

"Fan-out" means one event turning into many operations. Pull fans out at read time, to the people you follow. Push fans out at write time, to the people who follow you. Which is cheaper depends on two numbers: how often feeds are read compared with how often posts are written, and how unevenly followers are distributed.

We'll work through both for a system with **2 billion users**, unlimited follows, feeds that load in under **500 ms**, and up to **1 minute of staleness** acceptable.

## Pull

The data model: a Follow table (partition key `userFollowing`, sort key `userFollowed`, with a GSI the other way, lesson 11.03) and a Posts table with a GSI on (`creatorID`, `createdAt`), so "recent posts by X" is one query.

Building a feed:

1. Query Follow for the user's followees.
2. For each followee, query the Posts GSI for their recent posts (say the newest 20).
3. Merge all those lists by time (a k-way merge with a heap) and return the top 50.

Writes are trivially cheap: posting is one row insert. But one feed read becomes **tens to hundreds of requests** plus a large merge, on every read. A user following 500 accounts triggers 500 post queries, and the slowest of them sets the latency (lesson 02.12 on tail latency explains why fan-out to many backends makes p99 much worse). Feeds are read far more than posts are written, often 100 times more, so pull puts the expensive work on the common path.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of one pulled feed**

A user follows 400 accounts. Each feed load issues 1 followee query plus 400 "recent posts" queries, each returning up to 20 post IDs: 8,000 candidates merged to pick 50.

If 100 million users load their feeds 10 times a day, that's 1 billion feed loads, about 11,600 per second on average and perhaps 35,000 at peak. Times 400 queries: 14 million queries per second at peak. Even if each is a cache hit at 0.5 ms, the p99 of 400 parallel calls is far worse than the p99 of one.

</div>

## Push

The fix is a **PrecomputedFeed** table: key `userId`, value a list of the newest post IDs (say 200). When a post is created:

1. The new post is stored, and a message goes on a queue (for example SQS with at-least-once delivery, or Kafka).
2. **Feed workers** consume it, look up the followers of whoever posted (paging through the follower list), and prepend the new ID to each follower's feed, trimming each feed back to 200 entries.
3. Reading a feed is one key lookup for 200 IDs, then a multi-get of the first 50 posts from a post cache.

Because delivery is at-least-once, the prepend must be idempotent: skip the ID if it is already at the head of the list, or store feeds as sorted sets keyed by post ID so a duplicate insert changes nothing.

~~~dg-flow Fan-out on write: each new post is copied into each follower's precomputed feed by async workers; reading is one lookup.
New post -> Queue -(feed workers)-> Look up followers -(prepend post id)-> *Feed of each follower -(read: one key)-> Viewer
~~~

Storage check: 200 IDs × 10 bytes = **2 KB per user**, **4 TB** for 2 billion users. That's a lot of memory but entirely feasible across a cache cluster, and it shrinks a lot if only active users have feeds.

A classic Twitter design is the same: a fan-out service inserts each tweet into followers' home timelines held in **Redis lists** (entries of tweet ID, user ID and a meta byte, a few bytes each), and only users active in the last 30 days keep cached timelines; others are rebuilt from storage when they return.

## The celebrity problem

Push breaks for accounts with huge followings. One post by an account with 100 million followers means 100 million feed writes. Even a fast feed cluster doing 1 million list inserts per second spends **100 seconds** on that one post, and celebrities post several times a day, often in bursts. Fan-out for such accounts can take **minutes**, and posts can appear out of order: a reply from an ordinary user (fanned out in milliseconds) shows up in feeds before the celebrity's message it replies to.

~~~dg-bars Feed writes caused by one post, by the poster's follower count. Push cost grows linearly with followers, so a single celebrity post costs as much as a million ordinary posts.
Typical user, 200 followers | 200 | 200 writes
Popular, 50,000 | 50000 | 50,000 writes
Threshold, 100,000 | 100000 | 100,000 writes
*Celebrity, 100 million | 100000000 | 100,000,000 writes
~~~

~~~dg-compare Pull, push and hybrid feeds: where the work happens and what breaks.
[Pull]
Read -(fetch every followee)-> Merge
+ Posting is one insert
+ Always fresh
- Hundreds of queries per read
- Slow and spiky p99
[Push]
Write -(copy to every follower)-> Feeds
+ Read is one lookup
+ Fast, predictable reads
- Celebrity posts cost millions of writes
- Storage for every user's feed
[*Hybrid]
Push ordinary authors, pull celebrities
+ Bounded write cost per post
+ Reads: one lookup plus a few cached fetches
- Two code paths and a threshold to tune
~~~

## Hybrid: push for most, pull for celebrities

The hybrid:

- Mark accounts with more followers than a **threshold** as "not precomputed". The threshold is tuned (often somewhere around 10,000 to 1,000,000 followers), and can also take posting frequency into account.
- Feed workers **skip** fan-out for those accounts' posts.
- At read time, the feed service takes the user's precomputed feed and **merges** in recent posts from the few above-threshold accounts they follow, fetched directly. Those posts are heavily cached, because millions of readers request the same handful of celebrity timelines.

~~~dg-flow Hybrid feed: ordinary authors push into feeds at write time; celebrity posts are pulled and merged at read time.
[Write] Ordinary author posts -(fan-out workers)-> *Followers' precomputed feeds
[Read] Viewer opens feed -(one lookup)-> Precomputed feed -(merge recent posts of followed celebrities)-> *Final feed
~~~

So: **ordinary users' posts are pushed** (cheap: a few hundred followers each), and **celebrities' posts are pulled** (a handful per reader, all served from cache). Two further refinements are common:

- **Skip inactive followers.** Don't push to users who haven't logged in for 30 days; rebuild their feed with a pull when they return. Since a large fraction of accounts on any network are dormant, this can cut fan-out writes by half or more.
- **Per-reader lists of followed celebrities.** Store, with each user, the short list of above-threshold accounts they follow, so the read path doesn't have to scan the full followee list to find them.

~~~dg-seq Reading a hybrid feed: one precomputed lookup, a few cached celebrity timelines, a merge, then a multi-get of the posts.
Viewer > Feed service : GET /feed
Feed service > Feed cache : get feed:viewer (200 post IDs)
Feed service > Follow store : celebrities this viewer follows
Follow store -> Feed service : 3 accounts
Feed service > Post cache : recent posts of 3 accounts
Feed service > Feed service : merge by time, take 50
Feed service > Post cache : multi-get 50 posts
Feed service -> Viewer : feed page 1, cursor
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the numbers behind the threshold**

A user with 300 followers posts: 300 feed prepends, done in well under a second by workers.

An account with 80 million followers posts: 80 million writes would take over a minute even at a million writes per second, and burn capacity other posts need. With a threshold of 100,000 followers, that account is marked pull.

A reader following 400 accounts, 3 of them above the threshold, gets one feed lookup plus 3 cached "recent posts" lookups, merged by time. Total: about 5 cache calls instead of 400.

The read-side cost of that celebrity's posts: if 80 million followers each load their feed 10 times a day, that's 800 million reads of that account's recent posts per day, about 9,000 per second. Spread over a replicated cache, that's easy, and it is far cheaper than 80 million writes per post.

</div>

### Choosing the threshold

The threshold balances write cost against read cost. Push costs roughly (followers × posts per day) writes; pull costs roughly (active followers × feed loads per day) cached reads, but reads of the same hot timeline are cheap because they're cache hits on a few keys. Teams typically pick the threshold empirically, by watching fan-out queue lag and read latency, and some make it dynamic: an account that suddenly goes viral is switched to pull before its next post.

## Remaining details

- **Hot posts.** A viral post is read by millions in its first hours, overloading the one cache shard that holds it. The fix is a **replicated post cache**: several identical cache instances, any of which can serve any post, so the load spreads. A small in-process cache on the feed servers for the very hottest posts also helps.
- **Pagination.** Use a cursor: the timestamp (or ID) of the oldest post already shown, and fetch older posts after it. Offsets break when new posts are prepended between page loads.
- **Deletes and unfollows.** A deleted post leaves its ID in millions of feeds. Rather than removing it everywhere, filter at read time (the multi-get of posts returns "deleted") and let the IDs age out. Unfollows can likewise be filtered at read time.
- **Ranking.** Real feeds re-rank the candidate posts by predicted engagement; the precomputed list becomes a **candidate set** rather than the final order.

<div class="ex">
<div class="lab">Worked example</div>

**a celebrity post timeline under pure push**

12:00:00, a pop star with 80 million followers posts. 12:00:01, a fan with 200 followers replies; that reply is fanned out in 50 ms. 12:00:40, half the pop star's followers have it; the other half see the fan's reply first, with no context. 12:01:20, fan-out completes, and meanwhile the queue for everyone else's new posts has backed up by 80 seconds. With the hybrid, the pop star's update is visible to every follower on their next feed load, with no write cost at all.

</div>

## Common mistakes

- **Pure push "because reads are faster".** It falls over on the first celebrity post.
- **Pure pull "because it's simpler".** Read latency and backend load scale with the number of follows.
- **Fanning out to dormant users.** Wasted writes and memory for feeds nobody will read.
- **Non-idempotent prepends** under at-least-once delivery, producing duplicate entries in feeds.
- **Offset pagination** on a list that changes at the head.

## In an interview

Explain both models in two sentences each, state the read/write ratio and the follower skew, then propose the hybrid with a threshold. Give the numbers: 2 KB per user for 200 IDs, 4 TB for 2 billion users (less with only active users), a celebrity post costing tens of millions of writes under push. Ask about ranking (chronological or ranked?) and freshness (seconds or a minute?), because both change the read path.

<div class="take">
<div class="lab">Key takeaway</div>

pull computes feeds at read time (cheap writes, slow fan-out reads); push precomputes them at write time (fast reads, expensive writes for big accounts). The hybrid pushes posts from ordinary accounts into active followers' feeds and pulls posts from accounts above a follower threshold at read time, merging the two, so neither a celebrity post nor a feed read is ever unboundedly expensive.

</div>
