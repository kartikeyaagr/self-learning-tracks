---
id: "10.11"
course: "system-design-101"
module: "10"
slug: "t-digest-quantile-sketch"
title: "t-digest quantile sketch"
keyIdeas: ["t-digest estimates quantiles (p50, p99) of a data stream with small memory.", "It keeps clusters (centroids with a mean and count). Clusters near the median can hold many points; clusters near the tails are kept small.", "This makes extreme quantiles (p99, p99.9) accurate, which is exactly what latency monitoring needs.", "t-digests merge, so per-server digests can be combined into a global one.", "Alternatives: HDR histograms (fixed buckets with bounded relative error), KLL sketches, DDSketch."]
question: "Why does t-digest keep more detail near the tails?"
answer: "Tail quantiles such as p99 and p99.9 are what people actually monitor, and they need accuracy relative to the edge: a quantile error of 0.001 is invisible at the median but turns p99.9 into p99.8, which can be a very different latency. Tail values are also spread far apart, so merging two slow requests into one centroid would report a latency neither had, while values in the dense middle are packed together and can be merged with little loss. t-digest's scale function therefore limits centroids to a tiny size, down to single values, near q = 0 and q = 1, and lets them grow to hundreds of thousands of values near the median, spending a fixed few kilobytes where the precision matters."
sources: [{"title": "Ted Dunning: t-digest", "url": "https://github.com/tdunning/t-digest", "note": "Paper and reference implementation."}]
---

## The problem: quantiles from a stream

Latency targets are stated as quantiles. "p99 under 300 ms" means 99% of requests are faster than 300 ms. A **quantile** q is the value below which a fraction q of the data lies: the median is the 0.5 quantile (p50), and p99 is the 0.99 quantile. Quantiles matter because averages hide the slow tail that users actually feel (lesson 01.03 shows a service with an 88 ms average and a 1 second p99).

An exact quantile needs every value, sorted. This is worse than exact distinct counting: you must keep a copy of every item and sort them. For a service handling 50,000 requests per second, an hour of latencies is 180 million numbers, 1.4 GB as doubles, per endpoint, per hour, before you try to combine servers. You need a compact, mergeable summary that answers quantile queries approximately, with the error where it hurts least.

The **t-digest** is one of the most widely used. It is a data structure for accurate online accumulation of rank-based statistics (quantiles, trimmed means), handling both floating-point and integer values, with digests that can be **merged**, which suits map-reduce and parallel streaming systems such as Spark. Elasticsearch's percentiles aggregation, several Prometheus-compatible systems, Apache Druid and many APM tools use it or its close relatives.

## Why not a simple histogram?

The first idea is fixed buckets: count values in 0 to 10 ms, 10 to 20 ms, and so on. That is what Prometheus histograms do (lesson 10.12), and it works when you know the range and the interesting thresholds in advance. But fixed buckets waste resolution where data is dense and lose it where data is sparse. If 99% of requests take 30 to 60 ms and the slow 1% are spread from 200 ms to 10 s, equal-width buckets either need thousands of buckets or report p99.9 as "somewhere between 1 and 10 seconds". The t-digest adapts its buckets to the data.

## How it works: clusters of values

A t-digest summarises the distribution as a sorted list of **centroids**. Each centroid is a pair: a **mean** value and a **count** of how many values it represents. Together the centroids are a compressed picture of the sorted data, where each centroid stands in for a run of neighbouring values.

Building it is a variant of one-dimensional k-means clustering. In the common `MergingDigest` implementation:

1. Incoming values are appended to a small buffer.
2. When the buffer fills, it is sorted and merged with the existing centroids, walking left to right in value order.
3. During the walk, each value or centroid is folded into the current centroid **if the result stays within the size limit for that position**; otherwise it starts a new centroid.
4. The result is again a sorted list of centroids, typically a few hundred.

### Answering a query

To estimate the value at quantile q, compute the target rank q × N, walk the centroids in order adding up counts until you reach that rank, and **interpolate** between the means of the neighbouring centroids. Each centroid is treated as sitting at the middle of the ranks it covers.

~~~dg-flow Answering a quantile query: turn the quantile into a rank, walk the sorted centroids accumulating counts, and interpolate between the two centroids around that rank.
p99 of N values -(rank = 0.99 N)-> Walk centroids, sum counts -(passes rank)-> *Interpolate between neighbours -> Estimated value
~~~

<div class="ex">
<div class="lab">Worked example</div>

**reading p50 and p99 from a small digest**

N = 1,000 latencies summarised as nine centroids (mean, count): (10 ms, 1), (20, 5), (40, 200), (45, 400), (60, 300), (120, 80), (400, 10), (1,500, 3), (3,000, 1). The counts add up to 1,000.

p99 is rank 990. Cumulative counts are 1, 6, 206, 606, 906, 986, 996, 999, 1,000. The (120, 80) centroid covers ranks 906 to 986, centred at 946. The (400, 10) centroid covers 986 to 996, centred at 991. Rank 990 lies between those centres, 44/45 of the way, so p99 ≈ 120 + 0.98 × 280 ≈ 394 ms.

p50 is rank 500. The (45, 400) centroid is centred at 406 and the (60, 300) centroid at 756. 500 is 94/350 of the way: p50 ≈ 45 + 0.27 × 15 ≈ 49 ms.

Notice where the precision is. The median comes from centroids of 400 and 300 values, but a 15 ms spread between them means the answer can't be far off. p99 comes from centroids of 80 and 10 values, and the extreme tail is held as single values (3,000 ms is one actual request).

</div>

## Unequal centroids: the scale function

The key design choice is that **centroids are not equal in size**. A **scale function** k(q) limits how many values a centroid may absorb, depending on where it sits in the distribution. The rule is that a centroid may span at most one unit of k: if a centroid covers quantiles from q_left to q_right, then k(q_right) − k(q_left) ≤ 1. Where k is steep, centroids must be narrow; where k is flat, they can be wide.

The original scale function is `k(q) = (δ / 2π) · asin(2q − 1)`, where δ is the **compression** parameter (typically 100 to 200). The arcsine is flat in the middle and steep at both ends. Working through the derivative, a centroid near quantile q can hold about `2π · N · √(q(1 − q)) / δ` values.

<div class="ex">
<div class="lab">Worked example</div>

**centroid sizes for 10 million values with compression 100**

At q = 0.5: 2π × 10,000,000 × 0.5 / 100 ≈ 314,000 values in one centroid.

At q = 0.99: √(0.99 × 0.01) ≈ 0.0995, so about 62,500 values.

At q = 0.9999: √0.0001 = 0.01, so about 6,300 values.

At q = 0.999999: about 630 values. And the very first and last centroids are always kept as single values, so the minimum and maximum are exact.

The total number of centroids stays around δ to 2δ, a few hundred, whatever N is: at 16 bytes each (a double mean and a count) that's a few kilobytes for 10 million or 10 billion values.

</div>

~~~dg-bars Approximate centroid capacity across the distribution for 10 million values and compression 100: huge in the middle, shrinking towards the edges, and single values at the extremes.
q 0.5 | 314000 | about 314,000 values
q 0.9 | 188000 | about 188,000
q 0.99 | 62500 | about 62,500
q 0.9999 | 6300 | about 6,300
q 0.999999 | 630 | about 630
*Extreme value | 1 | 1 value, exact
~~~

Newer scale functions, based on logarithms, limit cluster size near the tails even more aggressively, roughly in proportion to q(1 − q) rather than its square root, which gives better relative accuracy at the extremes for the same number of centroids. The effect is the same in spirit: big clusters in the dense middle, tiny ones at the edges.

~~~dg-compare Spending a fixed budget of centroids: equal-sized clusters waste precision on the median, while t-digest's scale function moves it to the tails.
[Equal-size clusters]
Every centroid holds N / 200 values
- p99.9 blurred across 50,000 values
- Median precision nobody needs
+ Simple to reason about
[*t-digest scale function]
Large in the middle, tiny at the edges
+ Extreme quantiles nearly exact
+ Same few KB of memory
- Median error slightly larger
~~~

## Why the tails get the detail

This lesson's question asks why. Three reasons:

1. **The tails are what people query.** p99, p99.9 and p0.1 are what latency, risk and SLO metrics care about. The median is rarely where the problem is.
2. **Accuracy must be relative to the edge.** A quantile error of 0.001 is negligible at the median (0.500 versus 0.501 changes nothing). At p99.9 the same error turns p99.9 into p99.8 or p100, which may be a very different latency: in a typical service, the gap between the 999th and 998th slowest of 1,000 requests can be hundreds of milliseconds. To keep the error small **relative to the distance from the edge** (an error of 1% of 1 − q), the tails need much finer resolution.
3. **Tail values are spread out.** In a latency distribution, the middle values are packed close together (thousands of requests between 42 and 43 ms), so merging many of them into one centroid loses almost nothing. The slowest requests may differ by seconds, so merging two of them into one mean blurs exactly the information you wanted. A centroid holding 900 ms and 4,000 ms reports 2,450 ms, which neither request took.

## Accuracy and cost

Typical figures for the main implementation:

- **Parts-per-million accuracy** for extreme quantiles, and typically under 1,000 ppm (0.1%) error in quantile terms for middle quantiles.
- On the order of **100 to 200 ns per value added** in `MergingDigest` (it varies with hardware and settings), which allocates no memory after creation, so it can sit in a hot request path.
- A **compression** parameter trades size for accuracy: doubling δ roughly doubles the centroids and halves the error.
- A digest typically holds a few hundred centroids, a few kilobytes regardless of N.
- Compared with KLL, another mergeable quantile sketch with provable worst-case bounds, t-digest is typically smaller and more accurate at the tails in practice, while KLL's guarantees hold for any input order.

~~~dg-matrix Mergeable quantile summaries compared. All of them merge across servers; they differ in where the error is and what is guaranteed.
Summary | Error is | Tail accuracy | Merge | Guarantee
Fixed histogram | bucket width | depends on buckets | yes, add counts | yes, within a bucket
Exponential histogram | relative in value | yes, bounded relative | yes, add counts | yes
t-digest | relative to edge in rank | yes, very good | yes, re-cluster | partial, empirical
KLL | absolute in rank | partial, same as median | yes | yes, proven
HdrHistogram | relative in value | yes | yes, add counts | yes, fixed range
~~~

Note the two kinds of error. **Rank error** says how far the reported quantile is from the requested one (asked for p99, got p98.9). **Value error** says how far the reported number is from the true value (true p99 is 400 ms, got 412 ms). t-digest bounds rank error relative to the tail; exponential histograms such as DDSketch bound value error relative to the value. Both are good for latency; know which one your tool gives.

## Merging

Digests from many servers merge by pooling their centroids, sorting by mean and re-clustering under the same size limits. That is what makes global percentiles possible (lesson 10.12): each server keeps a digest per minute, a collector merges them, and p99 is read from the merged digest. Merging is not bit-for-bit lossless the way HLL's register max is, but the result stays within the same accuracy envelope, so you can merge per-minute digests into hours and hours into days. Averaging the servers' p99 values instead would be wrong, as the next lesson shows.

## Pitfalls

- **Sorted or adversarial input.** Feeding values in sorted order can degrade some implementations; the merging variant with a buffer copes well, but test with your real data order.
- **Very small N.** With a few dozen values, just keep them all. Digests shine at thousands and above.
- **Mixing compression settings** across the digests you merge makes accuracy unpredictable. Use one setting everywhere.
- **Confusing rank and value accuracy** when comparing tools or setting alert thresholds.

## In an interview

If a design needs latency percentiles across many servers or long periods, say "each server keeps a t-digest (or a histogram) per minute; we merge them centrally and read any quantile, because percentiles can't be averaged". If asked how t-digest works: sorted centroids of (mean, count); a scale function that allows big centroids in the middle and tiny ones at the tails; query by walking cumulative counts and interpolating.

<div class="take">
<div class="lab">Key takeaway</div>

t-digest summarises a distribution as sorted centroids (mean and count), with a scale function that allows huge centroids in the middle and tiny ones, down to single values, at the tails. That gives parts-per-million accuracy at extreme quantiles in a few kilobytes, at a fraction of a microsecond per value, and digests merge across servers and time.

</div>
