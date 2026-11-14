---
id: "08.09"
course: "system-design-101"
module: "08"
slug: "fanout-patterns"
title: "Fanout patterns"
keyIdeas: ["Fanout delivers one message to many recipients: notifications, feeds, cache invalidation, pub/sub.", "**Fanout on write (push)**: copy the item to each recipient's inbox when it's created. Reads are fast; writes are huge for users with millions of followers.", "**Fanout on read (pull)**: store once, gather at read time from everyone the reader follows. Writes are cheap; reads are expensive.", "**Hybrid**: push for normal users, pull for celebrities, merge at read time.", "Infrastructure fanout: SNS topic → many SQS queues, so each consumer gets its own copy and processes at its own pace."]
question: "Fan-out on write vs fan-out on read: which for a user with 50M followers?"
answer: "Fan-out on read for that user. Pushing every post into 50 million feeds means 50 million writes per post, minutes of delay at realistic worker throughput, a backlog that delays everyone else's posts, and wasted work for inactive followers. Store their posts once, keep them in a replicated hot cache, and merge them into each follower's feed when the feed is read. Keep fan-out on write for ordinary accounts so most of the feed is precomputed, which is the hybrid approach used by large social networks."
sources: [{"title": "AWS: SNS fanout to SQS", "url": "https://docs.aws.amazon.com/sns/latest/dg/sns-sqs-as-subscriber.html", "note": "One message to many queues."}, {"title": "Hello Interview: News feed breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-news-feed", "note": "Fan-out on write vs read."}]
---

## Two ways to fan out

**Fanout** means one event must reach many recipients: a post reaches every follower's feed, an order event reaches billing, shipping and analytics, a breaking-news alert reaches 30 million phones. The design question is always the same: **when do you do the per-recipient work?**

- **Fan-out on write (push)**: when the event happens, copy it (or a reference to it) into every recipient's inbox straight away. Reads are cheap, because each recipient's data is already gathered. Writes cost one operation per recipient.
- **Fan-out on read (pull)**: store the event once, in the sender's place. Each reader gathers what they need at read time. Writes are cheap; every read does the gathering.

Neither is better in general. The right answer depends on the ratio of reads to writes, on how many recipients each event has, and on how fresh reads must be. Most large systems use both.

~~~dg-compare Push and pull fanout for a social feed. Push moves work to post time; pull moves it to read time.
[Fan-out on write]
Post -(one write per follower)-> Follower feeds -> Cheap read
+ Feed read is one lookup
+ Predictable read latency
- Cost scales with follower count
- Wasted work for inactive followers
[Fan-out on read]
Post -(stored once)-> Author's posts -(gather at read)-> Feed
+ Post is one write
+ Nothing wasted on inactive users
- Read queries every followee
- Slow for users following thousands
~~~

## Messaging fanout between services

The infrastructure version of the pattern is fan-out on write between services. On AWS the standard combination is **SNS to SQS**: publish one message to an SNS **topic**, and SNS pushes a copy into every **subscribed** SQS queue. Each queue stores its copy until its own consumer polls, so each downstream service processes independently, at its own pace, with its own retries and dead-letter queue. The message an SQS subscriber receives is wrapped in a JSON envelope from SNS (with the topic, message ID and timestamp) unless you enable raw delivery.

A typical use: an `OrderPlaced` message fans out to separate queues for billing, shipping and analytics. If analytics is down for an hour, its queue fills up and drains later; billing and shipping never notice. SNS can also apply **filter policies** per subscription, so the EU warehouse queue only receives orders with `region = eu`.

Kafka gets the same effect differently: one topic, and each service is a separate **consumer group** with its own offsets (lesson 08.02). No copy per consumer is stored; the log is shared and each group just remembers its position. RabbitMQ does it with a fanout exchange bound to one queue per consumer.

~~~dg-flow SNS-to-SQS fanout: one publish, one durable queue per consumer, each read at its own pace.
[Billing] Order svc -(publish)-> *SNS topic -(copy)-> SQS billing -(poll)-> Billing
[Shipping] SNS topic -(copy, region filter)-> SQS shipping -(poll)-> Shipping
[Analytics] SNS topic -(copy)-> SQS analytics -(poll, slower)-> Analytics
~~~

## The news feed case

The classic interview version. Requirements: 2 billion users, a home feed of recent posts from people you follow, feed reads under 500 ms, and up to 1 minute of staleness is acceptable (a post may take a minute to appear for followers).

### Option 1: pure fan-out on read

Store posts in a Posts table keyed by author. To build a feed, look up who the user follows, query recent posts from each, merge by time and take the top 50.

For a user following 200 accounts, that's 200 lookups (or one scatter-gather across many shards) on every feed load. With feeds opened maybe 20 billion times a day (about 230,000 a second on average, a million at peak), that's hundreds of millions of post lookups a second. It's simple and always fresh, but too slow and too expensive at this scale, and terrible for users who follow 5,000 accounts.

### Option 2: pure fan-out on write

Keep a **precomputed feed** per user: a list of the most recent ~200 post IDs, newest first, in a key-value store or Redis list. Each entry is a post ID (8 bytes) plus maybe a timestamp, so about 2 KB per user, and roughly 4 TB for 2 billion users, which fits in a sharded cache cluster.

When someone posts:

1. The posting service saves the new post and puts a message on a queue (SQS or Kafka, at least once).
2. **Feed workers** read the message, page through the poster's followers, and prepend the new post ID to each follower's feed list, trimming it to 200 entries.
3. Reading a feed is now one lookup for 200 IDs plus a batch fetch of post bodies from a post cache.

Because delivery is at least once, the prepend must be idempotent: check the ID isn't already at the head, or use a sorted set keyed by post ID so a duplicate insert is a no-op.

~~~dg-seq Fan-out on write for an ordinary account: one post becomes one feed insert per follower, done asynchronously.
Author > Post service : create post p_9
Post service > Queue : PostCreated p_9
Post service -> Author : 201 Created
Feed worker > Queue : take p_9
Feed worker > Follow store : followers of author (page 1)
Feed worker > Feed cache : prepend p_9 to 1,000 feeds
Feed worker > Follow store : next page, until done
Reader > Feed service : GET /feed
Feed service > Feed cache : read 200 ids
Feed service -> Reader : 50 posts
~~~

### The celebrity problem

Fan-out on write breaks for accounts with tens of millions of followers. A user with 50 million followers who posts five times a day creates 250 million feed writes a day on their own. Worse, it's bursty: each post creates 50 million writes at once, all behind the same queue message.

<div class="ex">
<div class="lab">Worked example</div>

**one post from a 50-million-follower account**

Suppose the feed workers can sustain 100,000 feed inserts a second in total. One post needs 50,000,000 / 100,000 = 500 seconds, over 8 minutes, during which every other user's posts queue behind it, so ordinary followers' posts also arrive late, breaking the 1-minute freshness target for everyone.

Storage churn is bad too. 50 million feeds each get one entry, and since feeds keep only 200 entries, 50 million old entries are evicted. Many of those followers are inactive and will never open the app to see it. The work is wasted.

</div>

So the hybrid:

- The Follow table carries a flag on follows of very large accounts (say over 100,000 followers): **not precomputed**. Feed workers skip those follows.
- When a user opens their feed, the **feed service** reads their precomputed feed (posts from ordinary accounts) and, at read time, fetches recent posts from the handful of big accounts they follow, then **merges** by time.
- The celebrity's recent posts are fetched by everyone, so they sit in a **hot post cache**, replicated to several cache nodes so no single node takes millions of reads a second for one key.

~~~dg-flow Push for ordinary accounts, pull for celebrities, merged when the feed is read.
[Write path] Ordinary post -(queue)-> Feed workers -(prepend id)-> *Follower feeds
[Celebrity path] Celebrity post -(stored once)-> Hot post cache
[Read path] Open app -> Precomputed feed -(merge by time)-> Hot post cache -> 50 posts
~~~

This costs a little at read time. A user who follows 10 celebrities does 10 extra cache lookups, each about 1 ms, done in parallel. But it removes the 50-million-write bursts entirely. Twitter used exactly this kind of hybrid for its home timeline for years.

### Other refinements

- **Skip inactive users.** Don't fan out to users who haven't opened the app in 30 days. When they come back, build their feed with fan-out on read once, then resume pushing.
- **Pagination with a cursor**: return 50 posts and a cursor (the timestamp or ID of the oldest post seen). The next page reads entries older than that.
- **Viral posts** are a read-hotspot problem, not a fanout problem: replicate hot keys across cache nodes, or keep a small local in-process cache on each feed server.
- **Deletes**: when a post is deleted, you can't easily remove it from 10,000 feeds. Instead, filter at read time: fetching its body returns "deleted" and the feed service drops it.

~~~dg-bars Feed-insert writes caused by one post, by author's follower count. The threshold for switching to pull is where bursts outgrow the workers.
Typical user | 200 | 200 writes
Popular creator | 50000 | 50,000 writes
*Celebrity | 50000000 | 50 million writes
~~~

## Choosing in general

Ask four questions:

1. **What is the read-to-write ratio?** Feeds are read far more than written, which favours push.
2. **How many recipients per event, and how skewed?** A long tail of huge senders needs the hybrid.
3. **How fresh must reads be?** Push through a queue adds seconds or minutes; pull is always current.
4. **Do most recipients ever read?** If most inboxes are never opened, push wastes work.

The same reasoning applies outside feeds. A notification to 30 million users (a breaking news alert) is pushed, but in batches through a rate-limited pipeline. A "what's new in your teams" view in a work chat app may pull at read time because teams are small and messages are read in place.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the push path**

2 billion users, of whom 500 million post once a day on average, with an average of 200 followers each (after excluding celebrity follows). That's 500 million × 200 = 100 billion feed inserts a day, about 1.2 million a second on average and maybe 3 million at peak. If one Redis node handles about 100,000 list operations a second, the feed cache needs at least 30 nodes for writes alone, and in practice several times that for replication and read load. Excluding inactive followers (say half) halves the cost. This is why the inactive-user and celebrity cut-offs are not optional details.

</div>

## Common mistakes

- Choosing pure push and forgetting celebrities. Interviewers almost always ask "what about a user with 50 million followers?"
- Choosing pure pull for a huge, read-heavy product and ignoring the read amplification.
- Making feed workers non-idempotent, so a redelivered message duplicates a post in feeds.
- Storing full post bodies in every feed instead of IDs, multiplying storage and making edits impossible.
- Fanning out synchronously inside the create-post request, so posting takes minutes.

## In an interview

State the hybrid early: "push for ordinary accounts so feed reads are one lookup; pull for accounts over a follower threshold, merged at read time from a hot cache." Back it with arithmetic: follower count times write rate versus worker throughput. Mention idempotent workers, skipping inactive users and cursor pagination.

<div class="take">
<div class="lab">Key takeaway</div>

fan out on write for ordinary accounts so reads are cheap, and fan out on read for accounts with huge followings, merging at read time from a hot cache. Between services, a topic that copies each message into one queue per consumer (or one consumer group per service) gives each consumer its own pace and failure isolation.

</div>
