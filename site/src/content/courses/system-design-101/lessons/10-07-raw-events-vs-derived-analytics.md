---
id: "10.07"
course: "system-design-101"
module: "10"
slug: "raw-events-vs-derived-analytics"
title: "Raw events vs derived analytics"
keyIdeas: ["Keep raw events (the immutable facts) and build derived views (aggregates, counters, materialized tables) from them.", "Derived data can always be rebuilt from raw events: fix a bug in the aggregation, add a new metric, or backfill a new dimension by replaying.", "**Lambda architecture**: batch layer recomputes from raw data, speed layer gives fresh approximate results. **Kappa**: one stream processing path, reprocess by replaying the log.", "Raw events are big, so keep them in cheap storage (object storage, columnar files) with a retention policy."]
question: "Why keep raw events if you already have aggregated tables?"
answer: "Aggregated tables throw information away and bake in today's grouping and logic: a per-minute count can't be split by a new dimension, have a newly discovered bot filtered out, or be checked against an advertiser's dispute. With raw events kept as the immutable source of truth (in Kafka for recent data and cheap columnar storage for history), you can fix a bug or filter and recompute the aggregates, build new metrics over past data, reconcile real-time results against a batch recomputation, and backfill new stores and models. Kappa-style, you replay the retained log through the corrected code into a new table and switch over once it catches up, instead of maintaining a second batch implementation."
sources: [{"title": "Jay Kreps: Questioning the Lambda Architecture", "url": "https://www.oreilly.com/radar/questioning-the-lambda-architecture/", "note": "Kappa: reprocess from the raw log."}, {"title": "Kleppmann: Turning the database inside out", "url": "https://martin.kleppmann.com/2015/03/04/turning-the-database-inside-out.html", "note": "Derived views from an event log."}]
---

## The question behind the question

Once you have tables of per-minute clicks, per-day uniques and top-K lists, why pay to store the raw events too? The aggregates answer today's dashboard queries faster and in a fraction of the space. The raw events are huge, mostly never read, and expensive to keep in a fast database.

The answer comes from looking at what an aggregate is. An aggregate is a **function of the raw data plus some code**: "count clicks, grouped by ad and minute, excluding events flagged as bots by rule set v3". Once you've computed it and thrown the inputs away, you can never change the code, the grouping or the rules for the past. Every bug, every new question and every audit needs the inputs. This lesson is about treating raw events as the source of truth and everything else as a disposable view.

## The log is the truth, everything else is derived

Look at four things databases already do internally: **replication**, **secondary indexes**, **caching** and **materialised views**. In each, something is derived from something else:

- A **replica** is the result of applying the primary's write-ahead log in order.
- A **secondary index** is a data structure computed from table rows; the database rebuilds it from the table if it's corrupted.
- A **cache** is a copy of query results that can be thrown away and recomputed.
- A **materialised view** is a stored query result, refreshed from its base tables.

Inside the database, the real source of truth is the log: an ordered sequence of immutable changes. A replication log turns an imperative update ("set quantity to 3") into an immutable event ("customer X changed quantity of product Y to 3 at time T"). The tables are just the current result of replaying that log.

The architectural idea, sometimes called "turning the database inside out", is to make this explicit at the system level. Treat an append-only **log of immutable facts** (in Kafka, say) as the source of truth, and treat tables, indexes, caches, search indexes and aggregates as **derived views** computed from it by stream processors.

~~~dg-flow Turning the database inside out: the immutable event log is the source of truth, and every table, aggregate, index and cache is a view derived from it.
Writes -> *Immutable event log -(derive)-> Tables, Aggregates, Search index, Cache
~~~

The benefits of this structure:

- **Better data.** Updating in place destroys information. Take a shopping cart: a table only shows what's in the cart now, while the event log also shows items that were added and later removed, which is valuable for analytics and recommendations ("people often remove this item after seeing the shipping cost"). Accountants don't use erasers: a ledger records every transaction and corrections are new entries.
- **Recovery and auditing.** If a bug writes bad data into a table, you can see exactly what happened and recompute the table from events before the bug.
- **Fully precomputed caches.** Views are maintained from the log, so they can be warm from the start, and because the log fixes the order of writes, there are no invalidation races between the database and the cache.
- **New views without migrations.** A new feature can build a new view by reading the log from the beginning, beside the old one, without touching the systems already running.

## Lambda: two pipelines, one truth

Before streams were reliable, the common way to get both fast and correct answers was the **Lambda architecture**. Raw immutable data is written to a master dataset and feeds two systems in parallel:

- A **batch layer** (originally Hadoop) periodically recomputes complete, accurate views from all raw data. It is slow (hours) but simple and correct, because it always starts from scratch.
- A **speed layer** (originally Storm) processes recent events as they arrive and produces approximate, up-to-the-minute views covering only what the last batch hasn't processed yet.
- A **serving layer** merges both at query time: batch results for older data, speed results for the last few hours.

One part of this is worth keeping: retaining the input data unchanged and modelling processing as stages derived from it, because it makes **reprocessing** possible. Code changes, bugs are fixed, and you need to recompute outputs from inputs. The batch layer is a constant reprocessing machine.

The problem is the double implementation. Every metric's logic is written twice, in two different frameworks, and must give identical answers. Keeping them in sync is exactly as painful as it sounds: a bug fixed in the batch code but not the streaming code makes the dashboard jump every time a batch lands. Frameworks that promise to compile one definition to both systems tend to leak.

## Kappa: one pipeline, replay to reprocess

The alternative, called the **Kappa architecture**, keeps the raw log but uses one processing system:

1. Keep the full input log in Kafka (for example with 30 days' retention, or forever with tiered storage), allowing multiple subscribers.
2. To reprocess, start a **second instance** of the streaming job reading from the beginning of the log, writing to a **new output table**, with extra parallelism so it catches up quickly.
3. When it has caught up with the live end of the log, switch the application to read the new table.
4. Stop the old job and delete the old table.

~~~dg-compare Lambda runs two code paths and merges them; Kappa runs one streaming code path and reprocesses by replaying the log.
[Lambda]
Raw data -(batch job)-> Batch views
Raw data -(stream job)-> Speed views
+ Batch recompute is simple and robust
+ Works with weak stream processors
- Every metric written twice
- Two systems to run and keep identical
- Merge logic at query time
[*Kappa]
Retained log -(stream job v1)-> Table v1
Replay log -(stream job v2)-> Table v2
+ One code base, one set of semantics
+ Reprocessing is just a replay
- Needs long log retention
- Replays of months of data take time
~~~

~~~dg-timeline A Kappa-style reprocessing after a logic fix: the new job replays the log into a new table, catches up, and readers switch over.
Mon 09:00 | Bug found in click aggregation v1
Mon 11:00 | Fix deployed as job v2, reading from offset 0
Mon 11:00 to 15:00 | v2 replays 30 days in 4 hours, about 180 times live speed
Mon 15:00 | *v2 reaches the live end of the log
Mon 15:10 | Compare v1 and v2 on recent days
Mon 15:30 | *App switches reads to table v2
Tue 09:00 | Stop v1, drop table v1
~~~

In practice, many teams run a hybrid. The streaming path produces the live numbers, and a periodic batch job over the data lake recomputes them from raw events using the **same** code (Spark and Flink both run the same job in batch and streaming modes) to reconcile. The ad-click design in lesson 10.03 does exactly this.

## So why keep raw events?

Aggregates are lossy by design. A per-minute count can't tell you per-country counts, distinct users, the gap between a user's clicks, or which clicks came from a bot you only discovered last week. Keeping raw events lets you:

- **Fix bugs.** Recompute aggregates after correcting the logic, the time zone handling or the bot filtering.
- **Answer new questions over history.** The product team asks for clicks by device type. With raw events you build a new aggregate over the last year. Without them, the metric starts today.
- **Reconcile and audit.** Check real-time results against a recomputation, and show a customer the individual events behind a bill.
- **Backfill new systems.** Populate a new store, search index or machine-learning feature table from history.
- **Train models.** Recommendation and fraud models need individual events, not sums.

## Where raw events live

The cost is storage, which is why raw events move through tiers as they age. Kafka holds recent data for replay and for new consumers. After the retention period, the data lives in cheap object storage in columnar formats such as **Parquet**, partitioned by date, where batch engines (Spark, Trino, BigQuery) can scan it. Aggregates stay hot, in an OLAP store or a cache, for interactive queries.

~~~dg-layers Where each kind of data lives: aggregates are small and hot, raw events are big and cold, and every layer above can be rebuilt from the one below.
Cache | top-K lists, live counters; seconds old
*OLAP store | per-minute aggregates and sketches; months
Kafka | raw events, replayable; days to weeks
Object storage, Parquet | raw events, the permanent truth; years
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what raw events actually cost**

100 million click events per day, about 200 bytes each as JSON: 20 GB per day. In Parquet, columnar compression (repeated ad IDs, timestamps that differ by milliseconds, a handful of countries) typically shrinks this by 5 to 10 times, so about 3 GB per day, or about 1.1 TB per year.

Object storage at roughly 2 cents per GB per month: 1.1 TB is about 22 dollars a month for a full year of every click, and less on archival tiers. Compare the cost of one disputed invoice or one metric you can't backfill. The raw data is almost always worth keeping; the question is only how fast it must be to query.

</div>

## Pitfalls

- **Logging "events" that are really states.** "Cart now contains A, B" loses the removal. Log the facts that happened ("item C removed"), not snapshots.
- **No schema discipline.** Raw events kept for years are only useful if you can still parse them. Use a schema registry (Avro or Protobuf) and evolve schemas compatibly.
- **Personal data kept forever.** Raw events often contain user IDs and IP addresses. Privacy laws (GDPR's right to erasure) mean you need a deletion strategy, such as keying events by a pseudonymous ID whose mapping you can delete, or periodic rewrites of the data lake.
- **Replays that have side effects.** If a derived view's builder also sends emails, replaying the log sends them again. Keep side effects out of view builders, or make them idempotent.
- **Assuming you'll never reprocess.** Every long-lived metrics system eventually needs a backfill. Design for it on day one.

<div class="ex">
<div class="lab">Worked example</div>

**a bot filter after the fact**

On Thursday you learn a click farm inflated ad 9's clicks since Monday. With only per-minute aggregates, the bad clicks are mixed in forever: you can't tell which of the 418 clicks in minute 12:03 were fake.

With raw events in S3, you run the new bot filter (say, more than 20 clicks per device per hour, or clicks from a known data-centre IP range) over Monday to Thursday, recompute ad 9's per-minute counts into a new table, compare (Monday 12:03 drops from 418 to 161), switch the dashboard to the corrected table, and issue the advertiser a credit with evidence attached: the list of rejected click IDs and why.

</div>

## In an interview

When you add aggregates to a design, add a line for raw events too: "Raw events go to Kafka and are archived to S3 in Parquet; aggregates are views we can rebuild." If asked Lambda or Kappa, say you'd prefer one code path (Kappa, or a framework that runs the same job in batch and streaming) and use batch recomputation for reconciliation, not as a second implementation.

<div class="take">
<div class="lab">Key takeaway</div>

aggregates are derived and lossy; raw events are the source of truth. Keeping them lets you recompute after bugs, answer new questions over history, audit, and reconcile real-time results. Treat tables and caches as views over an immutable log, and reprocess with one streaming code base by replaying the log into a new table and switching readers.

</div>
