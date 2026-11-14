---
id: "02.12"
course: "system-design-101"
module: "02"
slug: "tail-latency"
title: "Tail latency"
keyIdeas: ["Tail latency is the slow end of the distribution: p99, p99.9.", "In fan-out systems the tail dominates. If one request hits 100 servers and each is slow 1% of the time, 1 - 0.99^100 ≈ 63% of requests are slow.", "Causes: GC pauses, background jobs, noisy neighbours, queueing, network hiccups.", "Fixes: **hedged requests** (send a second copy after the p95 time, use whichever returns first), tied requests, fewer fan-out hops, isolating background work, smaller GC heaps.", "Always monitor percentiles, not averages."]
question: "If a request fans out to 100 servers and each has a 1% chance of being slow, what fraction of requests are slow?"
answer: "The request is slow if any one of the 100 servers is slow. The chance that all 100 are fast is 0.99^100 ≈ 0.366, so about 63% of requests are slow, even though each server is slow only 1% of the time. This is why tail latency on individual servers matters so much for fan-out systems, and why techniques like hedged requests (where a leaf call is slow only if both copies are slow) are used to cut it."
sources: [{"title": "Dean and Barroso: The Tail at Scale", "url": "https://research.google/pubs/the-tail-at-scale/", "note": "Why p99 dominates fan-out systems; hedged requests."}]
---

## Averages lie at scale

A service that responds within about 100 ms feels fluid to people; beyond a few hundred milliseconds it feels sluggish. Large services answer each user request by asking many servers at once: a web search query touches thousands of index servers, a social feed calls dozens of services, a product page assembles prices, stock, reviews and recommendations in parallel. The user waits for **the slowest** of those answers.

So a rare slow response on an individual server, which is harmless in a small system, becomes the **normal** experience at scale. The **tail** of the latency distribution (the p99, the p99.9: the slowest 1% or 0.1% of responses) stops being an edge case and starts deciding what most users see. This lesson explains why tails exist, how fan-out amplifies them, and the techniques large systems use to build a predictably fast whole out of unpredictable parts.

~~~dg-flow Fan-out amplifies the tail: the root must wait for every leaf, so the request is as slow as its slowest leaf.
User request -> Root server -(fan out to 100 leaves)-> *1 leaf hits its p99 -> Whole request slow
~~~

## Why individual servers are sometimes slow

Even identical, well-tuned machines running the same code have occasional latency spikes. The causes are mostly shared resources and background work:

- **Contention** for CPU cores, processor caches, memory bandwidth and network with other processes on the same machine (and, in the cloud, with other tenants).
- **Background activity**: daemons, log rotation, compaction in storage engines, periodic health and metrics jobs.
- **Garbage collection pauses** in managed runtimes such as the JVM and Go, which can stall a process for milliseconds to, in bad cases, seconds.
- **Queueing** at several layers: in the server's request queue, in the kernel, in network switches. A request that arrives just behind a burst waits for the whole burst.
- **Hardware behaviour**: CPUs throttling under power or heat limits; SSDs running internal garbage collection, which can make some reads 100 times slower than usual; devices waking from power-saving states.
- **Cold paths**: a cache miss, a cold JIT, a new TLS connection.

You can reduce each of these, but not remove them all. So the realistic goal is not "no slow responses" but **"slow responses on parts don't become slow responses for users"**.

## The fan-out maths

If each server is slow (beyond its p99) 1% of the time, independently, then a request that needs **all** of N servers is fast only if every one of them is fast:

`P(request slow) = 1 − 0.99^N`

For N = 100: 0.99¹⁰⁰ ≈ 0.366, so about **63%** of requests are slow. Each server's p99 has become the requests' typical latency. Even at 1 slow response in 10,000 per server, a request touching 2,000 servers is slow 1 − 0.9999²⁰⁰⁰ ≈ 18% of the time.

~~~dg-bars Chance that a request sees at least one leaf at its p99, by fan-out width, in percent. Wide fan-out turns a 1% tail into the common case.
1 leaf | 1 | 1%
10 leaves | 9.6 | ~10%
40 leaves | 33 | ~33%
100 leaves | 63 | ~63%
*500 leaves | 99.3 | ~99%
~~~

Real measurements at Google show the effect. In one service, a single leaf request had a p99 of 10 ms, but the p99 for waiting on all leaves was 140 ms. Waiting for just 95% of the leaves cut that p99 to 70 ms: **the slowest 5% of leaves caused half of the tail**.

~~~dg-widget tail 100 sample requests: each one waits for every leaf, so one slow leaf makes the whole request slow. Hedging means both copies must be slow.
~~~

## Fixing it within a request

These techniques react to slowness while a request is in flight.

### Hedged requests

Send the request to one replica. If it hasn't answered by the time a normal request would have (say the **95th percentile** latency for that call), send a copy to **another replica**, and use whichever answers first; cancel the other. Because you only hedge the slowest 5% of calls, this adds only about **5% more load**, but it removes most of the tail: a request is now slow only if **both** replicas are slow.

In a Google benchmark that read 1,000 keys spread across 100 BigTable servers, hedging after 10 ms cut p99.9 latency from 1,800 ms to 74 ms, while sending only 2% more requests.

~~~dg-seq A hedged request. If the first replica hasn't answered by its p95, a second copy goes to another replica, and the first answer wins.
Client > Replica A : read key k
Client > Client : wait p95 (10 ms), no reply
Client > Replica B : same read (hedge)
Replica B -> Client : value, used
Client > Replica A : cancel
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what hedging does to the fan-out maths**

A request fans out to 100 leaves; each leaf call is slow 1% of the time, independently. Without hedging, 1 − 0.99¹⁰⁰ ≈ 63% of requests hit a slow leaf.

With hedging, a leaf call ends up slow only if the first replica is slow **and** the hedge is slow too: roughly 0.01 × 0.01 = 0.0001 (ignoring the short wait before the hedge goes out).

Now 1 − 0.9999¹⁰⁰ ≈ 1% of requests are slow, down from 63%. The cost: about 5% more leaf calls (only the ones still pending at p95 are hedged), and every slow request still pays the 10 ms hedge delay rather than the full slow-path latency.

</div>

Hedging only works for requests that are **safe to send twice** (reads, or idempotent writes, lesson 02.09), and only when slowness is **independent** across replicas. If every replica is slow because the whole cluster is overloaded, hedging adds load where it hurts most, so limit hedges with a budget.

### Tied requests

Hedging still waits before sending the second copy. **Tied requests** send to two servers at once, each knowing about the other. Both put the request in their queue. When one of them actually **starts** the work, it tells the other to cancel its copy. Because most of the variability is queueing time, this captures most of the benefit with almost no duplicated work. Delaying the second send by about 1 ms avoids the case where both start at the same instant. In Google's measurements, tied requests cut median latency by 16% and p99.9 by about 40%, with under 1% extra disk work.

~~~dg-compare Two ways to send a request to more than one replica. Hedging waits before duplicating; tied requests duplicate at once but cancel as soon as one starts work.
[Hedged]
Send to A -(no reply by p95)-> Send to B -> First reply wins
+ Only ~5% extra requests
+ Simple: no server changes
- Slow requests still wait the p95 delay
[Tied]
Send to A and B -(one starts work)-> Other cancels -> One reply
+ Cuts queueing delay from the start
+ Under 1% duplicated work
- Servers must talk to each other to cancel
~~~

## Fixing it across requests

These techniques work over longer timescales, shaping where load goes:

- **Micro-partitions**: split data into many more pieces than there are machines (around 20 per machine). Load can then be rebalanced in small steps by moving a few partitions, and when a machine fails, its partitions are recovered by many machines in parallel.
- **Selective replication**: add extra replicas for partitions that are hot or important, so their load is spread more thinly.
- **Latency-induced probation**: when one machine is consistently slow, temporarily stop sending it real traffic. Keep sending it **shadow requests** to measure its latency, and bring it back when it recovers. Slowness is often temporary (a background job, a noisy neighbour), so this removes the outlier without losing the machine.
- **Good-enough results**: in search, return once most leaves have answered, instead of waiting for every one. A given leaf holds the best result for fewer than 1 in 1,000 queries, so users rarely notice the difference. The same applies to feeds and recommendations: show the page without the slow piece.
- **Canary requests**: send a new query to one or two leaves first, and only fan out to thousands if they respond normally. This stops a "query of death" (one that triggers a crash bug) from crashing thousands of servers at once.

And inside each server:

- **Prioritise interactive requests over batch work**, and keep low-level queues short so the priorities actually take effect: a priority scheduler can't help if 500 batch requests are already sitting in the kernel's socket buffer.
- **Split long requests into slices** (time-slicing), so short requests don't wait behind one huge one.
- **Control background work**: throttle compactions and garbage collection, or schedule them during quieter periods.

~~~dg-timeline Latency-induced probation for one slow server. It is taken out of rotation while it is slow and brought back once shadow requests show it has recovered.
10:00:00 | server 17 p99 rises to 300 ms (others 20 ms)
10:00:30 | *server 17 put on probation: no real traffic
10:00:30 | shadow requests continue, every 1 s
10:04:00 | background compaction on server 17 finishes
10:04:30 | shadow p99 back to 20 ms: server 17 restored
~~~

## What to measure

Report **p50, p99 and p99.9**, not averages, and set SLOs on percentiles (lesson 01.03). Measure at the **user-facing** level as well as per service, because the fan-out maths means per-service percentiles understate what users see. Percentiles can't be averaged across servers: combine histograms instead. When a design fans out, say so, estimate the width, and say how you will control the tail.

<div class="ex">
<div class="lab">Worked example</div>

**news feed assembly**

Building a feed calls 40 services (posts, ads, profiles, ranking features) in parallel. Each has a p99 of 50 ms and a typical latency of 15 ms. If each is slow 1% of the time, about 1 − 0.99⁴⁰ ≈ 33% of feeds wait on at least one slow call, so the feed's p75 is already around the services' p99.

The team applies three fixes. Profile lookups are reads served by three replicas, so they are hedged after their p95 (25 ms). The ads call has a strict 80 ms budget: if it's not back, the feed renders without that ad slot. Ranking features fall back to a cached, slightly older feature vector if not back in 60 ms.

The slow-call probability for the feed drops to a few percent, and the worst case is bounded at roughly 80 ms plus rendering, instead of whatever the slowest service happens to do.

</div>

## Common mistakes

- **Reporting averages**, which hide the tail completely.
- **Assuming per-service p99 is what users see** in a fan-out design.
- **Hedging non-idempotent writes**, which can perform the operation twice.
- **Unlimited hedging**, which adds load during overload.
- **Waiting for every part** of a response when a partial answer would do.

## In an interview

When a request fans out (search, feeds, aggregation pages, scatter-gather queries across shards), point out the tail amplification with the 1 − 0.99^N calculation, then propose hedged requests for idempotent reads, timeouts with partial results for optional parts, and replicas for hot partitions. Mention that you would track p99 and p99.9 at the user-facing endpoint.

<div class="take">
<div class="lab">Key takeaway</div>

with fan-out, rare slowness on each server becomes common slowness for users: 1 − 0.99^N. Measure percentiles at the user-facing level, and cut the tail with hedged or tied requests, good-enough partial answers, probation for slow machines, micro-partitions and selective replication.

</div>
