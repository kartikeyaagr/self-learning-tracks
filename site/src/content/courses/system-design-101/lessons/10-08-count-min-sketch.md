---
id: "10.08"
course: "system-design-101"
module: "10"
slug: "count-min-sketch"
title: "Count-min sketch"
keyIdeas: ["A count-min sketch estimates how often each item appears in a stream, using fixed memory.", "It's a grid of d rows × w counters, with one hash function per row. To add an item, increment one counter per row. To query, take the minimum of its d counters.", "Collisions only add to counters, never subtract, so estimates can be too high but never too low. Taking the minimum picks the least-polluted counter.", "Error shrinks with wider rows; confidence grows with more rows.", "Good for heavy hitters; poor for rare items, whose counts get drowned by collisions."]
question: "Count-min overestimates but never underestimates. Why?"
answer: "Each occurrence of an item increments one cell in every row, and no operation ever subtracts, so every cell the item maps to holds its true count plus the counts of whatever other items happened to collide there. Counts are never negative, so that collision noise is always zero or more, and every one of the item's cells is at least its true count. The minimum of numbers that are all at least the true count is also at least the true count, so the estimate can only be too high; taking the minimum simply picks the least-polluted cell, and with independent hash functions per row it is unlikely that all of them collided with heavy items."
sources: [{"title": "Wikipedia: Count-min sketch", "url": "https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch", "note": "Structure and error bounds."}, {"title": "Hello Interview: Data structures for big data", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/data-structures-for-big-data", "note": "Count-min section."}]
---

## What count-min sketch answers

HyperLogLog (lesson 10.04) answers "how many **different** items?". Count-min sketch answers a different question: "**how many times** has item x appeared?", for any x you care to ask about, using memory much smaller than one counter per item. Graham Cormode and S. Muthukrishnan invented it in 2003 (published 2005).

Why not just keep a hash map from item to count? Because the number of distinct items can be enormous. A CDN counting requests per URL, a router counting packets per source IP, or a search engine counting queries sees hundreds of millions of distinct keys per day. A hash map with 300 million entries at around 50 bytes each is 15 GB, per server, per time window. A count-min sketch with a guaranteed error bound fits in tens of kilobytes, regardless of how many distinct items pass through it.

The price is that the answer is approximate, and approximate in one particular direction: it may be too high, but it is never too low. Understanding why is the whole lesson.

## Structure and operations

- A 2D array of counters with **d rows** and **w columns**, all starting at 0.
- Each row j has its own hash function hⱼ mapping any item to a column between 0 and w − 1. The hash functions are chosen independently (in practice, one good 64-bit hash with different seeds, or two hashes combined as h₁ + j × h₂).
- **Update(x, c)**: for each row j, add c to the counter at column hⱼ(x). For a plain count, c = 1.
- **Query(x)**: look at x's counter in each row, the cells at hⱼ(x), and return the **minimum**.

Both operations cost d hash computations and d memory accesses, independent of how many items exist: with d = 5, an update is five increments.

~~~dg-flow Count-min sketch with 3 rows: an update increments one cell per row; a query reads the same three cells and takes the minimum.
[Update #cats] #cats -(h1)-> row 1, col 4 += 1 -(h2)-> row 2, col 1 += 1 -(h3)-> row 3, col 7 += 1
[Query #cats] Read 52, 49, 61 -(minimum)-> *Estimate 49
~~~

## Why it never underestimates

Follow item x through the sketch. Every time x appears, all d of its cells are incremented. Other items may hash to the same cells (collisions) and add to them too, but no operation ever subtracts. So each of x's cells holds:

`cell value = true count of x + counts of every other item that collided there ≥ true count of x`

The noise term is a sum of counts, and counts are never negative, so it is always ≥ 0. The minimum of d numbers that are all ≥ the true count is also ≥ the true count. That's the whole argument: overestimation is possible, underestimation is impossible.

Taking the minimum is the best you can do with this information. Every cell is an upper bound, so the smallest upper bound is the tightest. With several independent hash functions, it is unlikely that all of x's cells collided with heavy items, so at least one cell is usually close to the truth.

<div class="ex">
<div class="lab">Worked example</div>

**overestimation in action**

A sketch with one row and four columns, for illustration. Stream: a, b, a, c, a, d. Suppose h₁(a) = 2, h₁(b) = 0, h₁(c) = 2, h₁(d) = 3. After the stream the cells are [1, 0, 4, 1].

Query a = 4 (true count 3: c collided with it). Query c = 4 (true count 1: badly inflated by a). Query b = 1 and d = 1 (exact, no collisions).

Add a second row with a different hash: h₂(a) = 1, h₂(b) = 3, h₂(c) = 3, h₂(d) = 0. Row 2's cells become [1, 3, 0, 2].

Now query a = min(4, 3) = 3, exact. Query c = min(4, 2) = 2, still too high (b collided with c in row 2) but better than 4. No query ever came out below its true count.

</div>

~~~dg-matrix The two-row sketch from the example after the stream a, b, a, c, a, d. Item a's cells are 4 and 3, so its estimate is 3; item c's are 4 and 2, so its estimate is 2.
Row | col 0 | col 1 | col 2 | col 3
Row 1, hash h1 | 1: b | 0 | 4: a, a, a, c | 1: d
Row 2, hash h2 | 1: d | 3: a, a, a | 0 | 2: b, c
~~~

## The error guarantee

How much too high can the answer be? Let N be the total of all updates (the stream length for plain counts). Choose:

- w = ⌈e / ε⌉ columns, where e ≈ 2.718
- d = ⌈ln(1/δ)⌉ rows

Then for any item x, the estimate is at most **true count + ε·N**, with probability at least **1 − δ**. Memory depends only on ε and δ, not on how many distinct items there are.

The proof is short enough to follow. In one row, the other items' counts total at most N, and each lands in x's column with probability 1/w, so the expected noise in x's cell is at most N/w = εN/e. By Markov's inequality (a non-negative quantity exceeds e times its mean with probability at most 1/e), the noise exceeds εN with probability at most 1/e. The rows use independent hashes, so the chance that **every** row's noise exceeds εN is at most (1/e)^d = δ. The minimum is only that bad if every row is that bad.

| Target | w | d | Counters | Memory at 4 bytes |
| --- | --- | --- | --- | --- |
| ε = 0.001, δ = 0.01 | 2,719 | 5 | about 13,600 | about 54 KB |
| ε = 0.0001, δ = 0.001 | 27,183 | 7 | about 190,000 | about 760 KB |

### Accurate for heavy items, useless for rare ones

The error is relative to the **total** N, not to the item's own count. That's the key practical fact.

<div class="ex">
<div class="lab">Worked example</div>

**what ε·N means for real items**

A stream of N = 10 million search queries per hour, sketch with ε = 0.001 and δ = 0.01 (54 KB). The error bound is 0.001 × 10,000,000 = 10,000.

Query "weather", true count 2,000,000: the estimate is between 2,000,000 and 2,010,000 with 99% probability, within 0.5%.

Query "count-min sketch tutorial", true count 50: the estimate is between 50 and 10,050. It might well be reported as 9,000. For this item the sketch tells you almost nothing.

In practice the average error is much smaller than the bound (the bound is a worst case), but the shape holds: frequent items are measured well, rare ones are swamped.

</div>

~~~dg-bars Relative error of the worst-case bound for items of different true frequency, with N = 10 million and epsilon = 0.001 (bound 10,000).
Item seen 2,000,000 times | 0.5 | up to 0.5% high
Item seen 100,000 times | 10 | up to 10% high
Item seen 10,000 times | 100 | up to 100% high
*Item seen 50 times | 20000 | up to 20,000% high
~~~

That is why count-min pairs naturally with heavy-hitter problems (lesson 10.09), which only care about frequent items, and with problems where overestimates are harmless, such as "is this IP sending more than 1,000 requests a minute?" in abuse detection.

## Variants and properties

- **Mergeable.** The sketch is linear: adding two sketches with the same dimensions and hash functions cell by cell gives exactly the sketch of the combined stream. Per-server or per-minute sketches merge like HLLs (lesson 10.05), so you can build a sliding window from bucket sketches (lesson 10.06). Being linear, a count-min sketch can even be subtracted, as long as every item's net count stays non-negative.
- **Conservative update.** When adding 1 to x, first compute x's current estimate m (the minimum), then raise each of x's cells only as far as needed, to max(cell, m + 1), rather than incrementing all of them. Cells already above m + 1 already account for this occurrence and don't need to grow. This reduces overcounting substantially on skewed data; the sketch is no longer linear (you can't subtract), but it remains an upper bound and merges reasonably by cell-wise sum.
- **Bias correction.** The minimum is biased upward. Estimators such as count-mean-min subtract an estimate of the average noise per row, giving less biased (but no longer one-sided) answers.
- **Count sketch** is a related structure: each row also has a ±1 sign hash, updates add or subtract, and the query takes the median of the rows. Collisions cancel on average, so it is unbiased and more accurate when the distribution is not skewed, but it can underestimate. Count-min works best when a few items dominate.

~~~dg-compare Count-min versus count sketch: one-sided and simple, or unbiased with signed updates.
[*Count-min]
Update: +1 in each row
Query: minimum of rows
+ Never underestimates
+ Very simple, linear, mergeable
+ Great on skewed data
- Error grows with total N
[Count sketch]
Update: +1 or -1 by a sign hash
Query: median of rows
+ Unbiased, collisions cancel
+ Better on flat distributions
- Can under- or overestimate
- Slightly more work per update
~~~

## In a design

For top-K, count-min is an option when exact per-item counts are too big: add each event to the sketch, query the sketch for the item's estimated count, and maintain a min-heap or sorted set of the current top 1,000 using those estimates (lesson 10.09). Redis supports this through the RedisBloom module with `CMS.INITBYPROB key 0.001 0.01`, `CMS.INCRBY key item 1` and `CMS.QUERY key item`, alongside a sorted set (`ZADD`). Network devices use count-min to spot heavy flows at line rate, and databases use similar sketches to estimate how many rows match a value when planning queries.

## Common mistakes

- **Using it for rare items.** If the question is "how many times did this one user do X?", the error bound swamps small counts. Use exact counting for small keyed sets.
- **Comparing sketches with different seeds or sizes.** Merging only works with identical w, d and hash functions.
- **Treating the estimate as exact for billing.** It is an upper bound with probability 1 − δ, not a ledger.
- **Forgetting that N keeps growing.** In a sketch that never resets, εN grows forever and estimates degrade. Use time-bucketed sketches and drop old buckets.

## In an interview

If a design needs per-key frequencies over a huge key space (requests per URL, queries per term, packets per IP) and only the big keys matter, say "count-min sketch: d rows of w counters, increment one cell per row, query the minimum; error at most ε times the total, with probability 1 − δ; about 50 KB for 0.1% error". Then explain why it only overestimates, and pair it with a heap for top-K.

<div class="take">
<div class="lab">Key takeaway</div>

count-min sketch estimates any item's frequency with d rows of w counters: increment one cell per row, and read the minimum. Collisions only add, so it never underestimates; with w = e/ε and d = ln(1/δ), the overestimate is at most εN with probability 1 − δ. It is accurate for heavy items, poor for rare ones, and mergeable by cell-wise addition.

</div>
