---
id: "17.04"
course: "system-design-101"
module: "17"
slug: "observability-and-slos"
title: "Observability and SLOs"
keyIdeas: ["A metrics system ingests time series (metric name + labels + timestamp + value) from many hosts, stores them, and supports queries and alerts.", "Ingestion: agents push (or servers pull/scrape) → collectors → a write-ahead buffer such as Kafka → time-series database.", "Time-series databases compress heavily (delta-of-delta timestamps, XOR float encoding, as in Facebook Gorilla) and keep recent data in memory for fast queries.", "Downsample old data (raw for days, 1-minute for weeks, 1-hour for years).", "The main scaling risk is **cardinality**: every unique label combination is a new series, so limit high-cardinality labels like user IDs.", "Alerting evaluates rules on a schedule and routes notifications with dedup and silencing."]
question: "How do you store 10M metric series cheaply and still query the last hour fast?"
answer: "Compress each series Gorilla-style: delta-of-delta encoding for timestamps (regular intervals cost about 1 bit) and XOR of consecutive values for floats, about 1.4 bytes per point instead of 16. Keep the most recent hours as compressed in-memory chunks sharded by series and protected by a write-ahead log, so last-hour queries and alert rules hit RAM (about 5 GB for 10 million series at 10-second resolution). Flush older data as immutable, compacted blocks with a label index to disk and object storage, downsample them into 1-minute and 1-hour rollups keeping min, max, sum and count, and delete beyond retention. Keep label cardinality bounded, since every label combination is another series."
sources: [{"title": "Google SRE book: Monitoring Distributed Systems", "url": "https://sre.google/sre-book/monitoring-distributed-systems/", "note": "What to measure."}, {"title": "Hello Interview: Metrics monitoring breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/metrics-monitoring", "note": "Time-series ingestion and alerting."}]
tags: ["Design"]
---

## What we are building

A metrics monitoring system in the style of Prometheus, Datadog or Facebook's internal Gorilla: every server and service reports numbers (CPU, memory, request latency, error counts, queue depth, business counters) every few seconds, engineers graph them on dashboards, and alert rules page someone when a service's SLO is in danger (lesson 14.02). It is the system you rely on when everything else is broken, so it must be cheaper and more reliable than the systems it watches.

### Requirements

- **Ingest** metrics: CPU, memory, latency, custom counters, each with labels.
- **Query and graph** them with filters ("only service=checkout"), aggregations (sum across hosts, p99) and time ranges from the last 5 minutes to the last year.
- **Alert** on rules like "p99 latency over 500 ms for 5 minutes", notifying by email, Slack or PagerDuty.
- Dashboards answer within a few seconds even over weeks of data. Alerts fire within **1 minute** of the data arriving.
- Alerting must be reliable even if dashboards are briefly behind. Late and out-of-order points must be handled.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**the raw load**

500,000 servers each send 100 metrics every 10 seconds: 500,000 x 100 / 10 = 5 million points per second.

On the wire each point carries labels, so it is 100 to 200 bytes: roughly 1 GB per second of raw ingest, which is why ingestion is batched and compressed.

Focus on storage for a cluster holding 10 million active series, each sampled every 10 seconds: 1 million points per second, 86.4 billion points a day.

Stored naively at 16 bytes per point (an 8-byte timestamp and an 8-byte float), one day is 86.4 billion x 16 = 1.4 TB, a year over 500 TB, before any index. Querying the last hour would mean reading 3.6 billion points x 16 bytes = 58 GB.

</div>

So the question this lesson answers is: how do you store 10 million series cheaply and still query the last hour fast? The techniques below come mainly from Facebook's Gorilla time-series database (2015) and the Prometheus storage engine.

## What a time series is

A **series** is one metric name with one exact set of labels:

```
http_latency_ms{service="checkout", host="web-17", route="/pay"}
```

Each series is a stream of `(timestamp, value)` points. Change any label value (`host="web-18"`) and it is a different series. Two structures are needed:

- the **points** of each series, stored in time order, compressed;
- an **index** from label values to series IDs (an inverted index, like search in lesson 09.02), so a query for `service="checkout"` can find every matching series quickly.

~~~dg-tree How a metrics store organises data: an index from labels to series, and compressed chunks of points per series.
Metrics store
  Label index | inverted
    service=checkout | series 1, 2, 3
    host=web-17 | series 1, 7
  Series 1 | latency, checkout, web-17
    Chunk 10:00 to 12:00 | about 1.4 B/point
    *Open chunk 12:00 | in memory
  Series 2
~~~

## Compression: from 16 bytes to about 1.4 per point

Metrics data has strong patterns, and Gorilla's two tricks exploit them.

### Timestamps: delta of delta

Points arrive at a steady interval: 10:00:00, 10:00:10, 10:00:20, 10:00:30. Store the first timestamp in full. Then the **deltas** (differences) are 10, 10, 10, and the **deltas of the deltas** are 0, 0. A zero is stored in **1 bit**. Occasional jitter (a delta of 11, so a delta-of-delta of +1) costs a short prefix plus a few bits. In practice around 96% of timestamps compress to a single bit.

### Values: XOR with the previous value

Consecutive values are often identical (a counter that did not change, a memory gauge) or close. Floating-point numbers that are close share their sign, exponent and leading mantissa bits. XOR two consecutive values:

- if they are equal, the XOR is zero, stored as **1 bit**;
- if they are close, the XOR has many leading and trailing zeros, so you store only the count of leading zeros, the length of the meaningful middle, and those middle bits. If the meaningful bits fall in the same window as the previous point, even the counts are skipped.

~~~dg-flow Delta-of-delta timestamps and XOR values shrink a point from 16 bytes to about 1.4.
[Time] 10:00:00, :10, :20, :31 -(deltas 10, 10, 11)-> Delta of delta 0, +1 -(mostly 1 bit)-> *Tiny
[Value] 42.0, 42.0, 42.5 -(XOR with previous)-> 0, few bits -(store changed bits)-> Tiny
~~~

Gorilla reported an average of **1.37 bytes per point** instead of 16, about **12 times** smaller. That made it possible to keep the last 26 hours of Facebook's metrics in memory, which in turn made queries over that window tens of times faster than the disk-based store it replaced.

<div class="ex">
<div class="lab">Worked example</div>

**what compression buys you**

At 1.37 bytes per point, 10 million series at one point per 10 seconds produce 3.6 billion points an hour: 3.6 billion x 1.37 is about 4.9 GB per hour, and about 118 GB per day.

Uncompressed, the same day would be 1.4 TB. The last hour, 58 GB uncompressed, is 4.9 GB compressed: it fits in RAM on two or three machines with room for replicas.

A dashboard panel showing one series for the last hour decodes 360 points, about 500 bytes, in microseconds. A panel summing 1,000 hosts decodes 360,000 points, still a few milliseconds.

</div>

The trade-off: compressed chunks must be decoded sequentially from the start, so you cannot jump to the middle of a chunk. Chunks are therefore kept short (Prometheus cuts a new chunk every 120 samples), so any time range needs only a few chunk decodes.

## The last hour in memory

Recent data is queried constantly (every dashboard refresh, every alert rule every minute) and is still being written. So:

- Keep the most recent **hours** (2 in Prometheus, 26 in Gorilla) as compressed chunks **in memory**, with one open chunk per series being appended to. The last hour of 10 million series is about 5 GB compressed, which fits on a few machines **sharded by series** (hash of the series' labels).
- A **write-ahead log** (WAL) on disk records each incoming batch before it is acknowledged, so a crash can rebuild the in-memory state by replaying it (lesson 03.08 explains WALs).
- Every couple of hours, finished chunks are written to disk as an **immutable block**: a directory with the compressed chunks, the label index for those series, and metadata. Older blocks are **compacted** into larger ones (2 hours into 6, 6 into 24, and so on) so queries open fewer files.

This is the same shape as an LSM tree (lesson 07.09): an in-memory write buffer, a log for durability, immutable sorted files on disk, and background compaction.

~~~dg-layers Tiered storage for metrics: recent data hot and detailed, old data cold and coarse.
*In memory | last few hours, raw, about 1.4 B/point; dashboards and alerts
Local disk blocks | last days, raw, immutable compressed files
Object storage | weeks to a year, 1-minute and 1-hour rollups
~~~

## Cheap long-term storage: downsample

Nobody needs 10-second resolution from six months ago, and nobody can see it: a graph 1,000 pixels wide showing six months has one pixel per 4 hours. **Rollups** (downsampling) pre-aggregate old data into coarser points. Keep **min, max, sum and count** per rollup point, not just the average, because averages of averages are wrong when counts differ, and min and max preserve spikes that an average would hide.

| Age | Resolution | Points per series per day |
| --- | --- | --- |
| last 2 days | raw, 10 s | 8,640 |
| up to 30 days | 1 minute | 1,440 |
| up to 1 year | 1 hour | 24 |

A query for "the last 6 months" reads hourly rollups: about 4,380 points per series instead of 1.5 million. Old blocks go to **object storage**, which costs a few cents per GB per month; retention rules delete beyond a year. Systems such as Thanos, Cortex and Mimir add exactly this layer to Prometheus.

~~~dg-bars Storage per day for 10 million series at each stage (log scale).
Raw, uncompressed | 1400 | 1.4 TB
*Raw, compressed | 118 | 118 GB
1-minute rollups | 70 | about 70 GB
1-hour rollups | 1.2 | about 1.2 GB
~~~

The 1-minute figure looks high because each rollup point stores four numbers (min, max, sum, count) and compresses less well than raw values, but it is a sixth of the raw point count and lives on cheap storage.

## Cardinality: the real cost driver

Every distinct label combination is a new series, with its own in-memory chunk and index entries. Adding a `user_id` label to a request metric with 10 million users turns one series into 10 million. This **cardinality explosion** is the classic way metrics systems fall over: memory fills, ingestion slows, and queries that touch the metric time out.

<div class="ex">
<div class="lab">Worked example</div>

**one innocent label**

A team adds `route` to `http_requests_total`, but uses the raw path, so `/orders/123`, `/orders/124` and so on each become a label value. They have 50 hosts, 5 status codes, and 2 million distinct order IDs a day.

Series created: 50 x 5 x 2,000,000 = 500 million in one day, 50 times the whole cluster's normal 10 million. The ingesters run out of memory within hours.

With the route template `/orders/:id`, the same metric has 50 x 5 x 1 = 250 series. Per-order detail belongs in logs and traces.

</div>

Defences:

- Never put unbounded values (user IDs, request IDs, full URLs, email addresses) in labels.
- Use **route templates** (`/orders/:id`, not `/orders/123`).
- Enforce **limits**: series per metric and per tenant, rejecting or dropping above a cap and alerting the owner.
- Put per-request detail in traces and logs (lesson 14.01), not metrics.

## Ingestion and alerting

~~~dg-flow Ingestion buffered through Kafka; alerting reads the hot in-memory tier first.
[Ingest] Agents -(batches every 10 s)-> Gateways -(partition by series)-> Kafka -> *Ingesters (RAM + WAL)
[Store] Ingesters -(2-hour blocks)-> Object storage -(compact, downsample)-> Rollups
[Alert] Rule evaluator -(every 1 min)-> Ingesters -(firing)-> Alert manager
~~~

- Ingestion usually goes through a buffer like **Kafka**, so a traffic spike or a slow storage node delays data rather than dropping it. Replicating each series to two or three ingesters protects against losing an ingester's memory.
- Store latency as **histograms** (counts per bucket, such as under 50 ms, under 100 ms, under 250 ms), not averages. Histograms can be added up across hosts, and any percentile can be estimated from them. Averages and precomputed percentiles cannot be combined correctly.
- **Alert rules** such as burn-rate alerts (lesson 14.02) are evaluated every minute against the in-memory tier. Alerting within a minute is exactly what the hot tier makes possible. Alert evaluation gets priority over dashboard queries, and the alerting path should be monitored by something outside the system itself (a "dead man's switch" alert that fires if alerts stop arriving).
- **Late and out-of-order points**: accept points within a small window behind the head of the series (for example 1 hour), and reject older ones, because the open chunk is append-only.

## Common mistakes

- Storing raw points in a general-purpose database row by row.
- Averaging percentiles across hosts.
- Unbounded label values.
- Keeping everything at full resolution forever.
- Alerting from the same cluster without any outside heartbeat check.
- Rollups that keep only the average, hiding spikes.

## In an interview

Do the naive storage calculation (16 bytes per point, 1.4 TB a day), then show how compression, a hot in-memory tier, immutable blocks and downsampling each cut it. Mention cardinality before the interviewer does. Ask: how long must raw data be kept, how many tenants, and what is the alerting latency target?

<div class="take">
<div class="lab">Key takeaway</div>

compress series with delta-of-delta timestamps and XOR-encoded values (about 1.4 bytes a point instead of 16), keep the last few hours in memory with a write-ahead log for fast dashboards and alerts, write immutable blocks to disk and object storage and compact them, and downsample old data into minute and hour rollups that keep min, max, sum and count. Control label cardinality, because each label combination is another series.

</div>
