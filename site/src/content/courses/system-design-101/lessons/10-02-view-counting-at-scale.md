---
id: "10.02"
course: "system-design-101"
module: "10"
slug: "view-counting-at-scale"
title: "View counting at scale"
keyIdeas: ["Counting unique views means deduplicating viewers, which needs memory per viewer if done exactly.", "Reddit counted unique views with HyperLogLog: events flow through Kafka, a consumer decides whether to count each view, and HLL counters live in Redis (with Cassandra for persistence). Each counter uses about 12 KB regardless of how many viewers.", "They accepted about 1% error because nobody needs an exact view count on a post.", "Rules like \"count a user once per short window\" stop refresh spam from inflating counts."]
question: "Why did Reddit accept approximate view counts?"
answer: "Reddit wanted unique viewers counted in real time, which done exactly means keeping the set of every viewer ID per post and checking it on every view. For posts with over a million viewers that is 8 MB or more each plus a set lookup per view, multiplied across thousands of active posts, which was too heavy on memory and CPU, while a batch job would not be real time. A HyperLogLog gives a count within about 1% (0.81% standard error in Redis) using at most 12 KB, and far less for small posts thanks to the sparse encoding. Since a view count is only a popularity signal shown to users, not a billing figure, being within a few percent was explicitly good enough, so the product requirement allowed a much cheaper design."
sources: [{"title": "Reddit: View counting at Reddit", "url": "https://www.redditinc.com/blog/view-counting-at-reddit/", "note": "HyperLogLog plus Kafka and Redis."}]
---

## The product decision behind a number

Reddit shows a view count on every post. That sounds like a counter problem (lesson 10.01), but it is a harder one, because Reddit wanted to count **people**, not page loads. When Reddit designed its view-counting system it set four requirements:

1. Counts must be **real time** or near real time, not daily or hourly batches. A post that goes viral should show its count climbing.
2. **Each user is counted once** within a short time window, so refreshing doesn't inflate views.
3. The displayed count must be **within a few percent** of the actual tally.
4. The system must run at production scale and process events within seconds.

Requirement 2 is what makes it hard. A simple counter can't tell a new viewer from a returning one. To know whether user u7 has already been counted for post p1, you must remember who has viewed p1. Remembering is where memory goes.

The interesting part of this lesson is requirement 3. Reddit decided, up front, that an answer within a few percent was good enough. That one product decision made an expensive problem cheap, and learning to spot when you are allowed to make it is one of the most useful habits in system design.

## Two different counts

It helps to name things precisely:

- **Total views (events)**: every page load. This is a plain counter, the problem of lesson 10.01.
- **Unique viewers (cardinality)**: the number of distinct users who loaded the page. "Cardinality" is the size of a set, so this is a count-distinct problem.

A total-views counter needs 8 bytes per post however many views it has. A unique-viewer count, done exactly, needs memory proportional to the answer.

## Why exact counts were rejected

An exact unique count needs the set of user IDs that viewed each post, checked on every view ("is u7 already in the set?") and added to if not. At Reddit, popular posts have over a million unique viewers. At 8 bytes per user ID, **1 million IDs take 8 MB for a single post**, and that is the raw data before the overhead of a hash set, which in practice doubles or triples it. Multiply by thousands of posts that are active at the same time and add the CPU cost of a set lookup on every view, and Reddit found it "extremely taxing" on both memory and CPU.

There were exact alternatives, and each has a flaw:

- **A database table of (post, user) pairs with a unique constraint.** Every view becomes an insert that may fail on a duplicate. That is a write per view into a table of billions of rows, with an index lookup each time.
- **A batch job over the logs.** Run a nightly `COUNT(DISTINCT user_id) GROUP BY post_id`. It is exact and cheap per row, but it breaks requirement 1: the number is a day old.
- **A Redis set per post.** `SADD p1 u7` then `SCARD p1`. Fast, but it is the 8 MB-per-post problem in RAM.

**HyperLogLog** (lesson 10.04 explains the mechanism) estimates the number of distinct items using a fixed, small amount of memory. Reddit uses Redis's implementation, which counts over a million IDs in **12 KB**, about 0.15% of the exact set's 8 MB. The price is a small error: Redis's standard error is **0.81%**, comfortably inside the "few percent" requirement.

~~~dg-bars Memory to track the unique viewers of one post with a million viewers. The exact set grows with the audience; the HyperLogLog stays fixed.
Hash set of IDs, with overhead | 20000 | about 20 MB
Raw 8-byte IDs | 8000 | 8 MB
*Redis HyperLogLog | 12 | 12 KB
~~~

## The pipeline

~~~dg-flow Reddit's view-counting pipeline: Nazar decides whether a view counts, Abacus adds it to a HyperLogLog in Redis, and Cassandra persists it.
View event -> Event collector -(batches)-> Kafka -> *Nazar: should it count? -(flagged event)-> Kafka -> Abacus -(PFADD)-> Redis HLL -(every 10 s)-> Cassandra
~~~

Each stage has one job:

- **Event collector.** Browsers and apps fire a view event when a post is displayed. The collector is a thin, stateless HTTP service that batches events into Kafka. It does no thinking, so it can scale horizontally and stay fast.
- **Nazar** (a Kafka consumer) applies rules about whether a view should count, using Redis to keep state. For example, it ignores **repeat views from the same user within a short period**. It adds a boolean "count this" flag to the event and writes it back to Kafka. Keeping the decision in a separate stage means the rules can change (new abuse patterns, new definitions of a "view") without touching the counting code.
- **Abacus** (a second consumer) reads Nazar's output. For counted views it runs `PFADD` on that post's HLL in Redis. If a post's HLL isn't in Redis (typically an older post whose key was evicted to save memory), it loads the saved HLL from Cassandra into Redis first, then adds to it.
- **Cassandra** persists both the HLL and the count. Abacus writes them out in batches, **grouped per post every 10 seconds**, so even a post viewed thousands of times per second generates one Cassandra write per 10 seconds.

Notice how many ideas from lesson 10.01 reappear. Kafka is the durable buffer. Consumers keep state in memory. The database sees one batched write per key per interval. The only new idea is replacing an exact count with a sketch.

### Small posts are counted exactly

Reddit evaluated linear probabilistic counting against HLL and chose an HLL variant that uses a **sparse** representation for small sets and switches to the **dense** HLL as the set grows. Redis does exactly this. In sparse form, an HLL for a post with 40 viewers stores only the handful of registers that are non-zero, a few dozen bytes, and the estimate in that range is very close to exact. Once the sparse encoding grows past a threshold (about 3,000 bytes by default in Redis, set by `hll-sparse-max-bytes`), Redis converts it to the dense 12 KB form. Most posts never get there, so most posts cost far less than 12 KB.

## Walking a view through the system

~~~dg-seq One user refreshes a post: Nazar's time-window rule drops the second view before it reaches the HyperLogLog.
Browser > Collector : view(p1, u7) at 12:00:00
Collector > Kafka : append event
Kafka > Nazar : event
Nazar > Redis : seen u7 on p1 recently?
Redis -> Nazar : no, record it
Nazar > Kafka : event, count = true
Kafka > Abacus : event
Abacus > Redis : PFADD hll:p1 u7
Browser > Collector : view(p1, u7) at 12:00:20
Nazar > Redis : seen u7 on p1 recently?
Redis -> Nazar : yes, 20 s ago
Nazar > Kafka : event, count = false
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one user refreshing**

User u7 opens post p1 at 12:00:00 and refreshes at 12:00:20. Nazar sees the second event, finds in Redis that u7 viewed p1 20 seconds ago, and flags it "don't count". Abacus skips it.

Even if it were counted, `PFADD hll:p1 u7` a second time would not change the HLL. An HLL counts distinct values: the same user ID always hashes to the same register and the same run of zeros, so it can never change the sketch twice. So why does Nazar bother? Because the time-window rule supports policies the HLL cannot express, such as "count a user again if they come back after 6 hours", "ignore views from known bots", or "ignore views shorter than a second". Nazar is where the definition of a view lives; the HLL only counts.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing Redis for active posts**

Suppose 100,000 posts are "active" (viewed in the last day) at any time. A power-law split: 2,000 have enough unique viewers (a few thousand or more) to have outgrown the sparse encoding and use the dense 12 KB form, and 98,000 stay sparse at an average of perhaps 500 bytes.

Dense: 2,000 × 12 KB = 24 MB. Sparse: 98,000 × 0.5 KB = 49 MB. Total: about 73 MB, which fits in one small Redis instance with room to spare.

Exact sets for the same posts: the 2,000 big posts average, say, 200,000 viewers each, which is 2,000 × 200,000 × 8 bytes = 3.2 GB of raw IDs, plus hash-set overhead, so 8 GB or more. Older posts are evicted from Redis and reloaded from Cassandra on demand, which also only works cheaply because each one is at most 12 KB to fetch.

</div>

## Why approximate was acceptable

The answer to this lesson's question is product judgement, not cleverness. A view count is a signal of popularity shown to users. It is not money, it is not a contract, and it does not drive decisions that need exactness. Nobody can tell 1,004,213 views from 1,012,000, and the displayed number is usually rounded ("1.0M") anyway. Being within about 1% while using about 0.15% of the memory and staying real time beat exact counts that would have needed huge memory or slow batch jobs.

~~~dg-compare Exact versus approximate unique counting for a social view counter.
[Exact set per post]
Memory grows with viewers
+ Precise to the last user
- 8 MB or more per popular post
- Set lookup CPU on every view
- Or a day-old batch job instead
[*HyperLogLog per post]
Fixed 12 KB, sparse when small
+ Real time, cheap per view
+ Mergeable across servers and days
- About 0.81% standard error
- Cannot list who viewed
~~~

Contrast this with ad billing (lesson 10.03), where every click is money paid by an advertiser and approximations are not acceptable. The same company may use HLL for the view counter on a post and exact, deduplicated, audited counting for the ads shown next to it. The rule is: **the cost of being wrong decides the precision you pay for.**

### What you give up

Approximation has real limits, and you should say them out loud:

- **You cannot ask who.** An HLL can't tell you whether u7 viewed p1, or list the viewers. If the product later wants "people you know viewed this", you need a different structure.
- **Small errors at small counts are visible.** "3 views" showing as "4" looks like a bug to a user. Sparse encoding and linear counting keep small counts very accurate, which matters for this reason.
- **Errors do not shrink with time.** 0.81% of 10 million is about 81,000 views either way. For a display number that's fine; for a payout to creators it might not be.

## Common mistakes

- **Choosing approximate counting without asking.** Always confirm the accuracy requirement; "within a few percent" must come from the product owner, not be assumed by the engineer.
- **Using HLL for counts that are not distinct.** Total page loads are a plain counter. HLL is only for "how many different".
- **Putting the rules inside the counter.** Abuse filtering and "what counts as a view" change often. A separate stage (Nazar) keeps them out of the hot counting path.
- **Persisting every PFADD.** Write the sketch to durable storage in batches (every 10 seconds per post), not on every view.

## In an interview

If asked to design a view counter, ask two questions: "Do we count views or unique viewers?" and "How exact must it be?" If the answer is unique and approximate, propose a Kafka pipeline with a filtering stage, a Redis HyperLogLog per item, and periodic persistence. Quote the numbers: 12 KB per item, about 0.81% standard error, versus megabytes for an exact set. Then mention the boundary: if the count is ever used for billing or payouts, it moves to exact, deduplicated counting.

<div class="take">
<div class="lab">Key takeaway</div>

Reddit needed real-time unique-viewer counts. An exact set of a million IDs costs 8 MB or more per post; a Redis HyperLogLog counts the same in 12 KB within about 1%. A Kafka pipeline decides which views count (Nazar), adds them to HLLs in Redis (Abacus), and persists to Cassandra every 10 seconds. Approximation was acceptable because a view count is a popularity signal, not money.

</div>
