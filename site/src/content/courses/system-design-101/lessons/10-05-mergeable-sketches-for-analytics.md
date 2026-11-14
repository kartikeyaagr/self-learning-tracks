---
id: "10.05"
course: "system-design-101"
module: "10"
slug: "mergeable-sketches-for-analytics"
title: "Mergeable sketches for analytics"
keyIdeas: ["A sketch is mergeable if sketches built on separate data can be combined into a sketch of the union.", "HLL merges by taking the max of each bucket, so 100 servers can each build a sketch and a central job merges them, or you can merge hourly sketches into daily ones.", "Exact distinct counts can't merge: 100 users on server A and 100 on server B could be anywhere from 100 to 200 unique users.", "Mergeable sketches: HyperLogLog and Theta (distinct counts), count-min (frequency), KLL and t-digest (quantiles).", "Store sketches per time bucket and dimension, and merge them at query time for any range."]
question: "Why can you merge HLL sketches from 100 servers but not merge their exact distinct counts?"
answer: "An exact distinct count is just a number; adding numbers from 100 servers counts every user seen on several servers multiple times, and nothing in the numbers says how big the overlap was, so it can't be removed. An HLL sketch keeps, for each register, the largest leading-zero value of the hashes routed there; a given user always produces the same hash, so they set the same register to the same value on every server. Taking the register-wise maximum across the 100 sketches therefore yields exactly the sketch you would have built from all the data in one place, counting each user once, and you can query it for the global estimate."
sources: [{"title": "Apache DataSketches", "url": "https://datasketches.apache.org/", "note": "Why mergeable sketches matter."}]
---

## Distinct counts don't add

Most numbers you aggregate are **additive**: the total revenue of 100 stores is the sum of their revenues, and the clicks in an hour are the sum of the clicks in its 60 minutes. That property is what makes big data tractable. Each machine computes a partial result over its slice, and a coordinator adds them up. Map-reduce, sharded databases and stream processors all lean on it.

Unique counts break it. If an app store had 3 million unique visitors and a music site 2 million, the combined total is not 5 million, because people who visited both are counted twice. The same holds across time: Monday's unique users plus Tuesday's unique users is not the two-day unique count, because most of Tuesday's users were also there on Monday. And it holds across machines: if 100 web servers each count their distinct users, adding the 100 numbers counts every user who hit more than one server more than once, and with a load balancer spreading requests randomly, that is nearly everyone.

To answer exactly, you must keep **a copy of every unique item** seen, in one place, so the overlap can be removed. A site with a billion users a day needs a billion records on hand. Quantile queries (lesson 10.11) are worse, needing every value seen, sorted. So exact distinct counting can't be parallelised by computing partial counts and adding them: the partial counts have thrown away the information needed to remove the overlap.

~~~dg-compare Two servers, 3 million and 2 million uniques, 1 million users on both. A number loses the overlap; a sketch keeps enough to remove it.
[Merge exact counts]
Server A: 3M -(add)-> 5M
Server B: 2M
- 1M users counted twice
- Overlap unknowable from two numbers
- Exact fix needs every user ID in one place
[*Merge sketches]
HLL A, HLL B -(register max)-> HLL of A or B
Estimate about 4M
+ Overlap removed automatically
+ Each sketch is 12 KB
- Answer within about 1%, not exact
~~~

## What a sketch is

A **sketch** is a small data structure, usually kilobytes, built by a streaming algorithm that sees each item once and answers a query approximately with a mathematically proven error bound. HyperLogLog (lesson 10.04) is a sketch for distinct counts; count-min (10.08) for frequencies; t-digest (10.11) for quantiles; a reservoir sample (10.10) for sampling.

The property this lesson is about is **mergeability**. A sketch type is mergeable if there is an operation that takes the sketch of dataset A and the sketch of dataset B and produces the sketch of A ∪ B (or one with the same error guarantee), without access to the original data. Input can then be split into any number of fragments, each summarised by its own sketch, and the sketches combined quickly into a sketch of the whole. Mergeability turns a non-additive statistic into something you can compute the way you compute sums.

## Why HLL merges exactly

For HyperLogLog, the merge is simple: take the **maximum of each register** across sketches. Redis does this with `PFMERGE`.

Why is that correct? Each register j holds the maximum "first 1 position" over all hashes that were routed to it. The hash of an item depends only on the item, not on which server saw it. So user u7 hashes to the same register, with the same value, on every server that sees u7. The register for the combined data should hold the maximum over all items seen anywhere, and the maximum over a union is the maximum of the maxima. A user who appears on two servers sets the same register to the same value on both, so the max doesn't double-count them.

The merged sketch is therefore **identical**, bit for bit, to the sketch you would have built by streaming all the data through one HLL. No accuracy is lost by merging, however many times you do it, in whatever order. (Formally, max is associative, commutative and idempotent, so merges can be arranged in any tree and repeated safely.)

<div class="ex">
<div class="lab">Worked example</div>

**register-wise max with four registers**

Server A's sketch: [2, 0, 5, 1]. Server B's sketch: [3, 0, 1, 1]. User u7 hashes to register 0 with value 2 and was seen by both, so both sketches have at least 2 there.

Merge: [max(2,3), max(0,0), max(5,1), max(1,1)] = [3, 0, 5, 1].

If you had streamed all of A's and B's users through one sketch, register 0 would hold the largest value of any user routed there: that's 3 (from a user only B saw). Register 2 would hold 5 (from a user only A saw). The merged sketch matches exactly, and u7 contributed once.

</div>

That is why 100 servers' HLLs can be merged but their exact counts can't. The HLL still encodes, in compressed form, which hash values were seen. A plain number doesn't.

~~~dg-flow Merging sketches: each server sketches its own traffic, a coordinator takes the register-wise max, and the global sketch is queried once.
Server 1 HLL, Server 2 HLL, ... Server 100 HLL -(register-wise max)-> *Global HLL -(PFCOUNT)-> Global distinct count
~~~

### How other sketches merge

Not every sketch merges by max, but the useful ones all merge somehow:

- **Count-min sketch** (10.08): add the two arrays cell by cell. It is a linear sketch, so the sum of sketches is the sketch of the combined stream.
- **Theta sketch**: keep the k smallest hash values seen (scaled into 0 to 1). Merging takes the k smallest of the union. The count is estimated from how densely the hashes are packed.
- **t-digest** (10.11): pool the centroids and re-cluster them under the size limits. Accuracy is preserved within the bound, though not bit for bit.
- **Reservoir sample with random tags** (10.10): keep the k items with the smallest random tags from the union.
- **Misra-Gries summary** (10.09): add counters key by key, then reduce back to k counters.

## What this enables

### Parallel processing

Each worker sketches its shard of data; a reducer merges. A Spark job computing distinct users over a petabyte of logs never shuffles user IDs across the cluster: each task emits one 12 KB sketch, and a final step merges a few thousand of them.

### Rollups over time

Store one sketch per minute or hour. To answer "unique users this week", merge 168 hourly sketches; to answer "last 30 days", merge 30 daily sketches. You compute any time range from stored pieces, never rereading raw data. Mobile analytics systems such as Flurry work this way: they store small per-minute sketches and merge them on demand, so a report over any range is a merge of stored pieces, and merging sketches is fast enough to do millions of them per second.

~~~dg-tree Time rollups from mergeable sketches: each level is built by merging the level below, and any range is answered by merging the fewest pieces that cover it.
Month sketch | merge of 30 days
  Day 1 sketch | merge of 24 hours
    Hour 00 sketch | merge of 60 minutes
      Minute sketches | built from raw events
    Hour 01 sketch
  Day 2 sketch
  *Day 30 sketch | still filling
~~~

### Rollups over dimensions

Keep one sketch per (country, device, day) combination. "Unique users in France on any device yesterday" merges the French sketches for each device; "unique users on iOS worldwide" merges the iOS sketches for each country. With exact counts, every grouping you might want would need its own count computed from raw data.

### Set expressions

**Theta sketches** support union, intersection and difference, and return sketches rather than numbers, so expressions like ((A ∪ B) ∩ (C ∪ D)) \ (E ∪ F) work. "Users who used search and checkout but not support" becomes a sketch calculation. Intersections have larger relative error when the overlap is small compared with the sets, because the answer is estimated from the few shared samples.

~~~dg-matrix Common sketch families and what they support. All of these merge, which is why OLAP stores can keep them pre-aggregated.
Sketch | Answers | Merge | Set intersection | Typical size
HyperLogLog | distinct count | yes, register max | partial, via inclusion-exclusion | 1 to 12 KB
CPC | distinct count | yes | no | under 1 KB to a few KB
Theta | distinct count | yes, k smallest hashes | yes | 8 to 64 KB
Count-min | item frequency | yes, cell sum | no | tens of KB
KLL, t-digest | quantiles | yes, re-cluster | no | a few KB
Reservoir | uniform sample | yes, k smallest tags | no | k items
~~~

## Where you meet this

This is how OLAP (online analytical processing) stores such as Druid and Pinot answer "count distinct" over billions of rows interactively. At ingestion they **pre-aggregate**: for each segment (say, one hour of data), each combination of dimensions gets a sketch column instead of raw user IDs. At query time the broker merges the sketches from every segment the query touches. A query over 90 days and 50 countries may merge tens of thousands of sketches, which takes milliseconds. BigQuery's `HLL_COUNT.INIT` and `HLL_COUNT.MERGE` functions, and PostgreSQL's `hll` extension, expose the same idea in SQL: you can store sketches in a table and merge them in a `GROUP BY`.

<div class="ex">
<div class="lab">Worked example</div>

**weekly actives from daily sketches**

Each day, each of 100 app servers keeps an HLL of user IDs. At midnight, the 100 sketches are merged into one daily sketch (12 KB) and stored; the per-server sketches are discarded. For "weekly actives", merge 7 daily sketches: 84 KB read, one merge, answer within about 1%.

Storing daily counts instead (4.1M, 4.3M, 4.2M, 4.0M, 4.4M, 3.1M, 2.9M) could never produce the weekly figure. Their sum is 27M, but if 80% of users come back on several days, the real weekly figure might be 9M. The overlap between days is exactly what the numbers lost.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**storage for a dimensional rollup**

Dimensions: 200 countries, 4 device types, 365 days. That's 292,000 combinations. If each held a dense 12 KB HLL, that is 3.5 GB, but most combinations are small and stay sparse at a few hundred bytes, so the real figure is a few hundred MB for a year of unique-user data at full flexibility. The raw user-day pairs it replaces might be tens of billions of rows.

</div>

## Pitfalls

- **Mixing sketch parameters.** Two HLLs with different register counts or different hash functions can't be merged directly. Pick one configuration and keep it for the life of the data, or downsample the larger one first.
- **Mixing implementations.** A Redis HLL and a BigQuery HLL are not compatible formats. Merge only sketches from the same library.
- **Expecting intersections to be accurate.** Unions are excellent; intersections and differences can be poor when the result is a small fraction of the inputs.
- **Storing only the final numbers.** If you keep the daily count but throw away the daily sketch, you've lost the ability to roll up. Store the sketch; the number is cheap to compute from it.

## In an interview

When a design asks for unique counts over flexible time ranges or dimensions ("daily, weekly and monthly actives, by country"), say: "distinct counts don't add, so I'll store a mergeable sketch per bucket instead of a number, HLL for counts or Theta if we need intersections, and merge at query time." Mention that the merge is lossless for HLL (register-wise max) and that this is how Druid and Pinot do it.

<div class="take">
<div class="lab">Key takeaway</div>

distinct counts aren't additive because the overlap is lost once you have a number. Sketches keep a compact summary that can be merged (HLL by register-wise max, count-min by cell sum, Theta by keeping the smallest hashes), so partial sketches from servers, time buckets or dimensions combine into an accurate total at query time.

</div>
