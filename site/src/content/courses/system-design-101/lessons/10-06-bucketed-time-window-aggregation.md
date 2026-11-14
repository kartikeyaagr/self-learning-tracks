---
id: "10.06"
course: "system-design-101"
module: "10"
slug: "bucketed-time-window-aggregation"
title: "Bucketed time-window aggregation"
keyIdeas: ["Time-window aggregation groups events into time buckets (per second, per minute) and keeps a count or sketch per bucket.", "**Tumbling windows**: fixed, non-overlapping (each minute). **Sliding windows**: overlapping (last 60 minutes, updated each minute). **Session windows**: grouped by gaps in activity.", "A sliding \"last hour, updated every minute\" view keeps 60 one-minute buckets and sums them, dropping the oldest as time moves on.", "Late events: decide how long to wait (watermarks) and whether to update already-emitted results.", "Roll up old data into coarser buckets (minute → hour → day) to save space."]
question: "Show the clicks in the last hour updated every minute. How many buckets do you keep?"
answer: "Keep 60 one-minute buckets of click counts, plus the one currently filling, in a ring buffer. Each click increments the bucket for the minute it happened in (event time). When a minute closes, after the watermark passes its end, add that bucket to a running total and subtract the bucket from 60 minutes ago, so each update is O(1) rather than re-counting the hour, and late events within the allowed lateness re-fire their bucket as an upsert. The bucket size sets the update granularity; for older data, roll minute buckets up into hour and day buckets, and for distinct counts or percentiles store a mergeable sketch per bucket and merge the 60 instead of subtracting."
sources: [{"title": "Apache Flink: Windows", "url": "https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/operators/windows/", "note": "Tumbling, sliding, session windows."}]
---

## Windows turn a stream into finite answers

A stream never ends, so "how many clicks?" has no answer until you add a time range: clicks in the last hour, per minute, today. A **window** is that time range. Windowing splits an infinite stream into finite pieces you can compute over, and the kind of window you pick decides how much state you keep and how often results change.

Stream processors such as Flink, Kafka Streams and Spark Structured Streaming share the same vocabulary:

| Window | Definition | Typical use |
| --- | --- | --- |
| **Tumbling** | Fixed size, no overlap, back to back, e.g. every 5 minutes | Per-minute counts, hourly reports, billing periods |
| **Sliding** (hopping) | Fixed size, a new window every *slide*; windows overlap when slide < size, e.g. 10 minutes every 5 | "Last hour, updated every minute" |
| **Session** | Per key, closes after a gap of inactivity, e.g. 10 minutes | User sessions, visits, game matches |
| **Global** | All elements of a key in one window; needs a custom trigger | Count-based windows ("every 100 events"), custom logic |

~~~dg-timeline The same stretch of time cut three ways: tumbling windows tile it, sliding windows overlap, and session windows follow the gaps in one user's activity.
[Tumbling 10 min] 12:00 | *Window A opens
[Sliding 10 min, every 5] 12:00 | Window 1 opens
[Session, 5 min gap] 12:00 | user clicks, session opens
[Sliding 10 min, every 5] 12:05 | Window 2 opens
[Session, 5 min gap] 12:05 | clicks continue
[Tumbling 10 min] 12:10 | *A closes, B opens
[Sliding 10 min, every 5] 12:10 | 1 closes, 3 opens
[Session, 5 min gap] 12:10 | last click at 12:08
[Sliding 10 min, every 5] 12:15 | 2 closes, 4 opens
[Session, 5 min gap] 12:15 | *gap passed: session 12:00 to 12:08 closes
[Tumbling 10 min] 12:20 | B closes, C opens
~~~

Tumbling windows can take an offset, for example daily windows offset by −8 hours so "a day" means midnight to midnight in California rather than in UTC. Session windows are the odd one: their length is not known in advance, and two sessions can merge if a late event fills the gap between them.

## Sliding windows: the naive cost

The question asks for clicks in the last hour, updated every minute. That's a sliding window of size 1 hour and slide 1 minute. Each event belongs to **60** windows at once: every window whose hour covers it. A generic stream processor assigns each element to every overlapping window and keeps state per window, so if you compute each window from scratch, every event is processed 60 times and 60 windows' worth of state is held. For a sliding hour with a 1-second slide, it would be 3,600 copies. That is wasteful, because consecutive windows share 59 of their 60 minutes.

## The bucketed approach

The efficient answer is to aggregate each event **once**, into a small time bucket, and then build every window out of buckets:

1. Keep a count per **1-minute tumbling bucket**. Each event increments exactly one bucket: the one for the minute it happened in.
2. Keep the last **60 buckets** in a ring buffer, an array of 60 slots where the slot for minute t is `t mod 60`, so the oldest slot is overwritten by the newest.
3. Keep a **running total**. When a minute closes, add the new bucket to the total and subtract the bucket that has just fallen out of the hour.

~~~dg-flow A sliding hour from 60 one-minute buckets: each minute, add the newest bucket and subtract the one that expired, so each update is constant work.
Buckets 12:01 to 13:00 -(minute 13:01 closes)-> *Total + bucket 13:01 - bucket 12:01 -> Updated hourly count
~~~

So the answer to the question is **60 buckets**, plus one more for the minute currently filling (61 slots in the ring), with resolution equal to the update interval. Each update is O(1): one addition and one subtraction, instead of re-summing 60 numbers or recounting the hour's events.

<div class="ex">
<div class="lab">Worked example</div>

**the running total, minute by minute**

At 13:00 the 60 buckets for 12:01 to 13:00 sum to 24,800 clicks. The bucket for 12:01 holds 390.

The minute 13:01 closes with 455 clicks. New total = 24,800 + 455 − 390 = 24,865. The 12:01 slot (index 1 in the ring, since 1 mod 60 = 1) is reused for 13:01.

At 13:02, bucket 12:02 held 410 and the new minute has 470: 24,865 + 470 − 410 = 24,925.

The dashboard reads one number per minute. The work per update doesn't depend on how many clicks there were.

</div>

### Choosing bucket size

The bucket size is a trade-off between smoothness, state and lateness:

- **Finer buckets** (10 seconds: 360 per hour) give a window that slides more smoothly and a fresher number, but six times the state.
- **Coarser buckets** (5 minutes: 12 per hour) are cheap but the "last hour" jumps in 5-minute steps and can be up to 5 minutes out of date at the edges.
- The bucket should match the **update interval** the product needs. Updating the display every minute with 10-second buckets buys smoothness nobody sees.

For long windows, use a **hierarchy** of buckets: 60 minute-buckets, 24 hour-buckets, 30 day-buckets. As each hour completes, its 60 minutes are rolled up into one hour bucket and the minute buckets can be dropped. "Last 30 days" then costs 30 day-buckets plus up to 24 hour-buckets for today, not 43,200 minute-buckets. Time-series databases (Prometheus recording rules, Druid rollups, RRDtool's round-robin archives) all work this way.

~~~dg-tree A bucket hierarchy for long windows: fine buckets for recent time, coarse buckets for the past, each built by rolling up the level below.
Last 30 days | 30 day buckets + today
  Today | 24 hour buckets
    Current hour | 60 minute buckets
      *Current minute | filling now
    Previous hours | rolled up from minutes
  Earlier days | rolled up from hours
~~~

## Aggregate incrementally

Flink offers two styles of window function, and the difference is mostly about memory:

- **ReduceFunction / AggregateFunction**: update an accumulator as each element arrives. State is one small value per window (a count, a sum, a min and max). An `AggregateFunction` for an average keeps (sum, count) and divides at the end.
- **ProcessWindowFunction**: buffers **all** elements of the window and processes them when it fires. Flexible (it can sort, look at every element, read window metadata), but state grows with the number of events. A window holding a million events holds a million events. Combine it with an incremental aggregate when possible, so the buffered state is the accumulator, not the raw elements.

For counts and sums, always aggregate incrementally. This works because counts and sums are additive: you can add a bucket in and subtract it out. Some aggregates are not subtractable: you can't "subtract" a minute from a maximum, or from a distinct count. For those:

- **Max or min over a sliding window**: recompute from the 60 bucket maxima each minute (60 comparisons, still cheap), or use a monotonic deque.
- **Distinct counts**: keep a mergeable sketch (an HLL, lesson 10.05) per bucket and merge the 60 sketches when the window fires. You can't subtract an HLL, but merging 60 of them is fast.
- **Percentiles**: keep a histogram or t-digest per bucket and merge (lessons 10.11 and 10.12).

## When does a window fire, and what about late events?

Windows are defined in **event time**, when the click happened, as recorded by the device or edge server, not **processing time**, when it reached the processor. The two differ because events are delayed: a phone in a tunnel uploads its clicks five minutes late, Kafka partitions are consumed at slightly different speeds, and a consumer restarts and replays. If you bucket by processing time, a backlog after a restart dumps ten minutes of clicks into one bucket and the chart shows a spike that never happened.

But event time raises a question: when is minute 12:00 finished? You can never be sure no more 12:00 events will arrive. Flink tracks progress with **watermarks**: a watermark at time t is a claim that "no more events older than t are expected". A common strategy is **bounded out-of-orderness**: watermark = the largest event time seen so far minus a fixed delay, typically around **30 seconds**. When the watermark passes a window's end, the window fires and emits its result.

Events that arrive after the watermark are **late**. Flink's options:

- **Allowed lateness** (default 0): keep the window state a while longer and re-fire with updated results when late events arrive. Downstream must accept updated results, which is easy if writes are upserts keyed by (ad, minute), lesson 10.03.
- **Side output**: send late events to a separate stream for correction or inspection rather than silently dropping them.
- **Drop**: acceptable for a live dashboard, never for billing.

~~~dg-seq Event time with a 30-second watermark delay: the 12:00 bucket fires once the watermark passes 12:01:00, and a straggler re-fires it within allowed lateness.
Kafka > Flink : click at 12:00:40 (arrives 12:00:41)
Kafka > Flink : click at 12:01:31 (arrives 12:01:32)
Flink > Flink : watermark = 12:01:31 - 30 s = 12:01:01
Flink > Store : fire bucket 12:00 = 418 (upsert)
Kafka > Flink : late click at 12:00:55 (arrives 12:01:50)
Flink > Flink : within allowed lateness of 5 min
Flink > Store : re-fire bucket 12:00 = 419 (upsert)
Kafka > Flink : click at 11:52:00 (arrives 12:09:00)
Flink > Side output : too late: send for batch correction
~~~

The watermark delay is a dial between **latency** and **completeness**. A 30-second delay means every minute's number appears 30 seconds after the minute ends and catches nearly every straggler. A 5-second delay is fresher but sends more events down the late path.

<div class="ex">
<div class="lab">Worked example</div>

**dashboard of clicks per ad, last hour**

Flink keys clicks by ad ID and counts them in 1-minute tumbling windows with an `AggregateFunction`, watermark delay 30 seconds, allowed lateness 5 minutes. Each closed minute is written to Redis as `clicks:ad9:202611091301 = 418`, with a 2-hour TTL so old buckets expire on their own. The dashboard service keeps a running hourly total per ad: on each new minute, add the newest bucket, subtract the one from 60 minutes ago.

State sizing: 10 million active ads × 61 buckets × 8 bytes ≈ 4.9 GB, spread across the Flink cluster's keyed state. In practice most ads have few clicks in most minutes, so storing only non-zero buckets cuts this a lot. Redis holds about 2 hours of non-zero buckets for the dashboard.

A late click re-fires the 12:01 window with 419. The upsert overwrites 418 with 419. If the running total had already used 418, the dashboard must apply the difference (+1) to its total: subtractable aggregates make that easy.

</div>

## An approximate alternative

When memory is very tight (a rate limiter tracking millions of API keys, for example), you can approximate a sliding window with just **two buckets**: the current fixed window and the previous one. Estimate = current count + previous count × (fraction of the previous window still inside the sliding window). If the previous minute had 100 requests, the current minute has 30 so far, and we're 20 seconds in, the estimate for the last 60 seconds is 30 + 100 × 40/60 ≈ 97. It assumes events in the previous window were spread evenly, which is often good enough for limiting, but not for analytics.

## Common mistakes

- **Using processing time for analytics.** Restarts and backlogs create fake spikes and dips.
- **Sliding windows as 60 separate windows.** Correct but 60 times the work; build from buckets.
- **Buffering raw events in windows.** A `ProcessWindowFunction` without incremental aggregation holds every event in state.
- **No plan for late data.** Silent drops make numbers quietly wrong; decide between allowed lateness, a side output and a batch correction.
- **Subtracting non-subtractable aggregates.** Max, min, distinct counts and percentiles need merging, not add-and-subtract.

<div class="take">
<div class="lab">Key takeaway</div>

aggregate a stream into small tumbling buckets (say 1 minute) with incremental functions, then build longer or sliding windows by combining buckets: for "last hour, updated each minute", keep 60 buckets plus the one filling, add the newest and subtract the expired one. Use event time with watermarks, and decide explicitly what happens to late events.

</div>
