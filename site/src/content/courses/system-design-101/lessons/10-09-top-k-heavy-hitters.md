---
id: "10.09"
course: "system-design-101"
module: "10"
slug: "top-k-heavy-hitters"
title: "Top-k heavy hitters"
keyIdeas: ["Top-k / heavy hitters: find the most frequent items (hashtags, songs, URLs) in a stream.", "Exact counting per item is expensive at scale. Approximate approaches: count-min sketch plus a min-heap of size k; Misra-Gries or Space-Saving algorithms with k counters.", "Distributed: each node computes local top-k (or sketches), then merge centrally. Local top-k merges can miss items that are moderately frequent everywhere, so keep more than k locally or merge sketches.", "Pipeline for \"top 10 hashtags last hour\": Kafka partitioned by hashtag → stream workers keep per-minute counts → merge 60 minute-buckets → heap of top 10."]
question: "Find the top 10 hashtags over the last hour from 1M events/s. Sketch the pipeline."
answer: "Send hashtag events to Kafka partitioned by hashtag (64 or more partitions) so each worker sees every occurrence of its hashtags and its counts are complete, salting any extremely hot tag. Each worker counts per hashtag in 1-minute buckets (exactly, or with count-min plus a heap or Misra-Gries if the key space is huge) and keeps a running hour per hashtag by adding the new minute and subtracting the expired one, with a min-heap tracking its local top 100. Every minute, workers send their local top lists to a merger; because the key sets are disjoint, the top 10 of the union is the global top 10, which the merger writes to Redis for the API to read. Using hourly rather than per-minute local lists avoids missing hashtags that are steady but never top in a single minute."
sources: [{"title": "Hello Interview: Top-K breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/top-k", "note": "Heavy hitters at scale."}, {"title": "Wikipedia: Misra-Gries summary", "url": "https://en.wikipedia.org/wiki/Misra%E2%80%93Gries_summary", "note": "Exact algorithm for frequent items."}]
---

## The problem

"Top 10 hashtags in the last hour" over 1 million events per second. This is the **heavy hitters** or **top-K** problem: find the K items with the largest counts in a stream. It powers trending topics, most-viewed videos, hottest products, the most-requested URLs on a CDN and the noisiest IP addresses during an attack.

The exact approach is easy to describe: keep a counter per hashtag per time bucket, and every minute sort all the counters. The trouble is size and speed. There may be millions of distinct hashtags in an hour, the events arrive across many machines, and a sort of everything every minute is wasteful when you only want 10 results. This lesson combines the pieces from this module (bucketed windows, count-min, heaps, mergeable summaries) into a standard pipeline, and shows where approximation creeps in.

## Keeping the top K: a min-heap

Start with the smallest piece. Given a stream of (item, count) updates, how do you keep track of the current top K without sorting? Use a **min-heap of size K**. A heap is a binary tree stored in an array where every parent is smaller than its children, so the root is always the smallest element. In a min-heap of the current top K, the root is the **weakest** member of the top K, the one a newcomer must beat.

For each candidate (item, count):

1. If the item is already in the heap, update its count and restore heap order (O(log K)).
2. Else if the heap has fewer than K items, insert it (O(log K)).
3. Else if its count is greater than the root's, replace the root with it and sift down (O(log K)).
4. Otherwise ignore it: it can't be in the top K.

Each step costs O(log K), and K is small (10 or 100), so this is a handful of comparisons. Step 1 needs a side index (a hash map from item to heap position) so you can find an item in the heap. In Redis, a sorted set does all of this for you: `ZINCRBY trending 1 "#worldcup"`, then `ZREVRANGE trending 0 9 WITHSCORES` for the top 10, trimmed occasionally with `ZREMRANGEBYRANK`.

~~~dg-tree A min-heap holding the current top 5 hashtags by count: the root is the weakest of the five, so a newcomer only has to beat 4,100 to get in.
#cats 4,100 | root: the bar to beat
  #nyc 5,200
    #f1 9,800
    #oscars 7,300
  *#worldcup 61,000
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a newcomer enters the heap**

The heap above holds the top 5. A new count arrives for #eurovision: 4,600. It isn't in the heap, the heap is full, and 4,600 > 4,100 (the root, #cats), so #cats is evicted and #eurovision takes the root position. Sifting down compares it with its children (#nyc 5,200 and #worldcup 61,000); 4,600 is smaller than both, so it stays at the root. The new bar to beat is 4,600.

Next, #cats gets more events and its count reaches 4,400. It isn't in the heap, and 4,400 < 4,600, so it is ignored.

</div>

## Misra-Gries: heavy hitters with k−1 counters

The heap needs accurate counts to feed it. When even a counter per distinct item is too much memory, the **Misra-Gries summary** (Misra and Gries, 1982) finds frequent items with only **k − 1 counters**. The update rule for each arriving item i:

1. If i already has a counter, increment it.
2. Else, if fewer than k − 1 counters are in use, start a counter for i at 1.
3. Else, **decrement every counter by 1**, removing any that reach 0 (and discard i).

The intuition: each decrement step "cancels" k distinct items against each other (the k − 1 tracked ones and the newcomer). In a stream of n items, that can happen at most n/k times. An item that occurs **more than n/k times** can't be completely cancelled, so it is certain to be in the summary at the end. Its stored count is an underestimate by at most n/k. A second pass, or a parallel exact count for the few surviving candidates, gives exact counts. With k = 2 it is the classic Boyer-Moore majority vote algorithm (one counter, finds the item occurring more than half the time, if one exists).

~~~dg-timeline Misra-Gries with 2 counters (k = 3) on the stream a b a c a d a: an unseen item arriving when both counters are in use decrements everyone.
1: a | counters a:1
2: b | a:1, b:1
3: a | a:2, b:1
4: c | *both full, decrement all: a:1, b gone, c discarded
5: a | a:2
6: d | a:2, d:1
7: a | *a:3, d:1. a occurs 4 > 7/3 times, so it must survive
~~~

"a" ends with a counter of 3 against its true 4, an undercount, but it is correctly kept as the heavy hitter. "d" is also in the summary with 1; Misra-Gries can keep false candidates, which is why you verify the survivors.

Misra-Gries summaries are **mergeable**: add the two summaries' counters key by key, then if more than k − 1 keys remain, subtract the k-th largest count from all counters and drop those at zero or below. So each server can keep its own summary and a coordinator merges them, with the same n/k guarantee for the combined stream. The **Space-Saving** algorithm is a close relative used in many libraries: instead of decrementing everyone, a newcomer replaces the item with the smallest counter and inherits its count plus one, so it overestimates instead of underestimating.

## Count-min plus heap

The alternative: add each event to a count-min sketch (lesson 10.08), read back the item's estimated count, and offer (item, estimate) to a min-heap of size K. Memory is the sketch (fixed size, tens of kilobytes) plus K entries. Estimates can be slightly high, but heavy hitters are precisely the items count-min estimates well, because the error bound εN is small compared with their counts. A rare item can occasionally be overestimated enough to sneak into the heap, which is why you size ε so that εN is well below the expected K-th count.

~~~dg-matrix Three ways to count candidates for top-K, by memory and exactness.
Method | Memory | Counts | Keeps every true heavy item? | Mergeable
Exact hash map | grows with distinct keys | exact | yes | yes, add maps
Count-min + heap | fixed sketch + K | over, by at most epsilon N | partial, rare misses | yes, add sketches
Misra-Gries | k counters | under, by at most n/k | yes, if count > n/k | yes
Space-Saving | k counters | over, bounded | yes, if count > n/k | yes
~~~

## The pipeline

~~~dg-flow Top 10 hashtags per hour at 1 million events per second: partition by hashtag, count locally per minute, merge local top lists, combine 60 minutes.
Events, 1M per s -(partition by hashtag)-> Kafka, 64 partitions -(local counts per minute)-> *64 workers -(local top 100)-> Merger -(sum 60 minute buckets)-> Top 10, cached
~~~

1. **Ingest.** Events go to Kafka, **partitioned by hashtag**, so every occurrence of a given hashtag goes to the same partition and the same worker, and its count on that worker is complete. At 1M events per second, 64 or more partitions spread the load (about 16,000 events per second per partition). For extremely hot hashtags, add a random suffix and sum later, as in lesson 10.03, at the price of making that hashtag's count span several workers.
2. **Count locally per minute.** Each worker keeps counts per hashtag in 1-minute tumbling buckets (exact counts if memory allows; count-min or Misra-Gries if not).
3. **Emit a local top list per minute.** Each worker sends its top N (say 100, more than K for safety) with counts.
4. **Merge.** A single merger combines the workers' lists. Because partitioning by hashtag makes the workers' key sets **disjoint**, no hashtag's count is split across workers, so the global top K for a minute is simply the top K of the union of local top lists. (If keys were not disjoint, a hashtag at number 101 on every one of 64 workers could be number 1 globally and be missed entirely.)
5. **Sliding hour.** Keep 60 minute-buckets of counts for candidate hashtags and add or subtract as minutes roll (lesson 10.06), then take the top 10.
6. **Serve.** Write the result to a cache (Redis) every minute; API reads never touch the pipeline.

### The subtle step: from minutes to an hour

Combining **per-minute top-100 lists** into an hourly top 10 is approximate. A hashtag that was 101st every single minute but steady all hour never appears in any minute's list, yet its hourly total could make it the hour's true number 8, beating a hashtag that spiked to number 1 for two minutes and vanished.

<div class="ex">
<div class="lab">Worked example</div>

**the steady hashtag the minute lists miss**

#marathon gets 1,200 uses every minute, all hour: 72,000 in total. It ranks about 101st each minute, so it never appears in a local top-100 list.

#halftimeshow gets 30,000 uses in each of two minutes and almost nothing otherwise: 60,000 in total. It tops two minute lists.

Summing the minute lists, the merger sees #halftimeshow at 60,000 and #marathon at 0. The true hourly ranking has #marathon ahead.

</div>

Remedies, from cheapest to most thorough:

- **Emit larger local lists** (top 1,000 instead of 100), shrinking the gap.
- **Keep per-hashtag hourly counts on each worker** (a running 60-bucket window per hashtag, lesson 10.06). Keys are disjoint per partition, so each worker can produce an exact hourly local top list, and the merge of those is exact.
- **Accept the approximation** for a "trending" feature and reconcile with a batch job for anything reported officially.

### Trending is not the same as top

Real trending features rarely show the raw most frequent hashtags, because the answer would always be the same perennial tags. They rank by **velocity**: this hour's count compared with a baseline, such as the same hour last week or an exponentially weighted average. The pipeline is the same; the score fed to the heap is, for example, `count_this_hour / (baseline + smoothing)`.

<div class="ex">
<div class="lab">Worked example</div>

**the numbers**

1M events per second × 60 = 60M events per minute, 3.6 billion per hour. Suppose 2M distinct hashtags per hour. Exact hourly counters per hashtag: 2M × (key about 20 bytes + 8-byte count + hash map overhead, say 60 bytes) ≈ 120 MB in total. Spread over 64 workers, that's under 2 MB each, so exact counting per worker is perfectly affordable here, and even 61 minute-buckets per hashtag (about 120 MB per worker if every hashtag appeared every minute, far less in reality) is manageable.

The merger receives 64 lists of 100 entries per minute: 6,400 entries, trivial.

The sketches matter when keys are far more numerous (every URL on a CDN, every source IP, every search query) or memory per worker is tight: 500 million distinct keys per hour at 60 bytes is 30 GB, while a count-min sketch with ε = 0.0001 is under 1 MB.

</div>

## Common mistakes

- **Partitioning randomly, then merging top lists.** If a key's count is split across workers, merging local top lists can miss the true top items. Partition by key, or merge full summaries rather than top lists.
- **Sorting everything every minute.** Use a heap; the work should be proportional to K, not the number of keys.
- **Ignoring hot keys.** One hashtag at 200,000 events per second overloads its partition. Salt it and sum.
- **Serving from the pipeline.** Cache the result; the API should read a precomputed list.

## In an interview

Sketch the pipeline in one breath: "partition events by hashtag into Kafka so counts are complete per worker; count in 1-minute buckets with a running hourly window; keep a min-heap of the top K per worker; merge local top lists in a single merger; write the top 10 to Redis every minute." Then say where it is approximate (minute lists versus hourly totals) and how you'd fix it, and mention Misra-Gries or count-min plus a heap if the key space is too large for exact counts.

<div class="take">
<div class="lab">Key takeaway</div>

partition events by key so each worker's counts are complete, count in time buckets with a running window, keep a min-heap of the top K, and merge local top lists. When distinct keys are too many for exact counts, use Misra-Gries (k−1 counters, guaranteed to keep anything over n/k) or count-min plus a heap, and watch for items that are steady but never locally top.

</div>
