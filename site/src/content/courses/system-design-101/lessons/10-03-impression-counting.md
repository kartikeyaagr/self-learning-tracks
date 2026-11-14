---
id: "10.03"
course: "system-design-101"
module: "10"
slug: "impression-counting"
title: "Impression counting"
keyIdeas: ["Ad impression and click counts are used for billing, so accuracy matters more than for view counts.", "Pipeline: clients send events → ingestion → Kafka → stream processor aggregates per ad per minute → OLAP store for queries.", "Dedup with an event ID: retries resend the same ID, and the aggregator ignores IDs already seen (within a window).", "Combine fast streaming counts for dashboards with a periodic batch job over raw logs that produces the authoritative billing numbers (reconciliation).", "Filter fraud and bot traffic before counting."]
question: "How do you avoid counting the same impression twice when events are retried?"
answer: "Attach a unique impression ID, signed with an HMAC so clients can't forge new ones, to every ad shown, and make it the idempotency key. The ingest service verifies the signature, checks the ID against a cache of recently seen IDs and drops repeats, appending the event to Kafka before marking the ID so a crash can cause a duplicate but never a loss. The stream processor keys events by impression ID to drop any duplicates that got through, uses checkpoints that save state and Kafka offsets together so replays after a crash don't double-count, and writes results as upserts keyed by ad and minute so retried writes are harmless. A daily batch job over raw events deduplicates by impression ID and corrects anything the real-time path missed."
sources: [{"title": "Hello Interview: Ad click aggregator", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/ad-click-aggregator", "note": "Streaming aggregation with dedup."}]
---

## Why duplicates are the central problem

For a view counter (lesson 10.02), being 1% off is fine. For ads it is not, because counts are money. Advertisers pay per click (CPC) or per thousand impressions (CPM, cost per mille). An **impression** is one showing of an ad to one user; a **click** is that user clicking it. Over-counting overcharges advertisers, who will notice, demand refunds and stop trusting the platform. Under-counting loses revenue. Both are unacceptable, so the system must count each event **exactly once**.

Typical requirements for an ad click aggregator:

- **Idempotent click tracking**: never count the same click twice.
- **No data loss**: an acknowledged click is never dropped.
- A peak of **10,000 clicks per second** across 10 million active ads, about 100 million clicks per day.
- Advertisers can query clicks per ad at **minute granularity**, with results available within a minute or so.
- After a click is recorded, the user is redirected to the advertiser's site quickly (under about 100 ms of extra latency).

### Where duplicates come from

Networks and computers fail in ways that make "exactly once" hard. Duplicates come from three sources:

- **Client retries.** The browser or app sends the click, the server records it, but the response is lost (a mobile network drops, a load balancer times out). The client cannot tell "never arrived" from "arrived, reply lost", so it re-sends. This is the classic problem of at-least-once delivery.
- **Pipeline retries.** A consumer reads an event from Kafka, adds it to a count, then crashes before committing its offset (its bookmark in the log). After restart it reads the same event again and counts it twice.
- **Abuse.** Users or bots click the same ad repeatedly, or a competitor runs a click farm to drain an advertiser's budget. These are not technical duplicates, but they are duplicates for billing purposes.

The first two need **idempotency**: processing the same event twice has the same effect as processing it once. The third needs **policy**: rules about which clicks are billable.

## Give every impression an ID

You can only deduplicate if you can tell that two events are "the same". The trick is to define sameness at the source. The **ad placement service**, the component that decides which ad to show, generates a unique **impression ID** each time an ad is shown to a user (a 128-bit random value or a UUID). That ID travels with the ad to the client, and a click carries its impression ID. Now "the same click" has a precise definition: two clicks with the same impression ID. The impression ID is the **idempotency key** for everything downstream.

Why not let the client make the ID? Because a dishonest client could invent a fresh ID for every fake click, and each would look new. So the impression ID is **signed**: the placement service computes an HMAC (a keyed hash, such as HMAC-SHA256) over the impression ID and ad ID using a server-side secret, and sends the signature with the ad. The click processor recomputes the HMAC and rejects any click whose signature doesn't match. A client can replay a real impression ID, but it can't mint new ones, and replays are exactly what dedupe catches.

The same idea covers impressions themselves: the impression beacon (the small request the page sends when the ad becomes visible) also carries the signed impression ID, so a beacon fired twice counts once.

## Deduplicate on the way in

~~~dg-seq Dedupe at the click processor: verify the signature, check the impression ID in a cache, write to the stream first, then remember the ID.
Browser > Click processor : click(ad 9, imp 7f3, sig)
Click processor > Click processor : verify HMAC
Click processor > Redis : seen imp 7f3?
Redis -> Click processor : no
Click processor > Kafka : append click event
Kafka -> Click processor : ack (replicated)
Click processor > Redis : SET imp 7f3 NX EX 86400
Click processor -> Browser : 302 to advertiser
~~~

The order of the last steps matters. The processor writes to the stream **first** and records the ID in the dedupe cache **second**. Think about a crash between the two:

- **Stream first, then mark** (the right order): the event is in Kafka but the ID is not in Redis. A retry would pass the dedupe check and produce a second event. That is a duplicate, which the downstream layers (below) are built to catch. Nothing is lost.
- **Mark first, then stream** (the wrong order): the ID is in Redis but the event never reached Kafka. A retry is rejected as a duplicate. The click is silently lost forever, and nothing downstream can recover it.

When forced to choose between a possible duplicate and a possible loss, choose the duplicate, because duplicates can be removed later and losses cannot.

Use a distributed cache such as a Redis cluster for seen IDs, with a TTL (time to live) long enough to cover realistic retry windows, often 24 hours. `SET key value NX` sets the key only if it does not exist, which also handles two simultaneous retries racing each other: one wins and the other sees the key.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the dedupe cache**

100 million clicks per day, each impression ID 16 bytes. With key overhead in Redis, budget about 80 bytes per entry. A 24-hour TTL keeps at most a day's clicks: 100 million × 80 bytes = 8 GB. That fits across a small Redis cluster (say 4 nodes of 4 GB each, with replicas). At 10,000 clicks per second peak, that's 10,000 `GET`s and 10,000 `SET`s per second, trivial for Redis.

If you deduplicated impressions too (often 100 times more numerous than clicks), the cache would need 800 GB. That's why impression dedupe is often done with a Bloom filter per time window, or left to the stream processor's keyed state, rather than with one Redis entry per impression.

</div>

## Exactly-once inside the pipeline

The stream processor (Flink) counts clicks per ad per minute. A crash in the middle could double-count events that were counted but whose progress wasn't saved. Flink handles this with **checkpoints**: periodically (say every 10 seconds) it snapshots both its state (the partial counts in each open window) and the Kafka offsets it has consumed, together, as one consistent picture. After a crash it restores the state and rewinds Kafka to the offsets in that same snapshot. Events after the checkpoint are replayed, but they are being added to state that does not yet include them, so nothing counts twice inside Flink.

That guarantee ends at Flink's edge. When a window closes, Flink writes the result to the analytics store, and a crash after writing but before the next checkpoint will cause the write to happen again after recovery. So writes must be **idempotent**: write "ad 9, minute 12:03 = 418" as an upsert keyed by (ad, minute), rather than "add 418 to ad 9". Writing the same result twice leaves the same row. An increment written twice would double the count.

~~~dg-flow End-to-end: one ID per impression, dedupe on ingest, checkpointed aggregation, idempotent upserts, and raw events kept for a batch job that reconciles.
[Real time] Signed impression ID -> *Dedupe cache -(new clicks only)-> Kafka -(Flink, checkpointed)-> Upsert ad, minute -> OLAP store
[Safety net] Kafka -(all raw events)-> S3 data lake -(daily Spark job)-> *Dedupe by impression ID -> Corrections
~~~

### Dedupe inside the stream as well

The Redis check at the edge is best effort: if Redis is briefly unavailable, the processor should still accept clicks (losing clicks is worse) and let duplicates through. So a second dedupe layer runs inside Flink: key the stream by impression ID and keep a small piece of state per ID ("seen", with a timer to expire it after a day). The first event for an ID passes; later ones are dropped. Because this state is checkpointed with the offsets, it is exactly-once even across crashes.

## Reconcile with raw data

Add a final safety net: Kafka also dumps all raw click events to a data lake (S3, in a columnar format such as Parquet), and a **daily batch job** (for example in Spark) recomputes the aggregates from the raw events, deduplicating by impression ID and applying the latest fraud rules, then compares them with what the streaming path produced. Differences are corrected in the store, and the corrected numbers are what invoices use. This is the "batch layer checks the speed layer" idea from lesson 10.07: the fast path gives answers in seconds, the slow path makes them right.

~~~dg-matrix Which layer catches which kind of duplicate or bad click. No single layer is enough; together they cover every case.
Duplicate source | Signed ID | Edge cache | Flink keyed state | Daily batch
Client retry | partial, defines sameness | yes | yes | yes
Retry while Redis down | partial | no | yes | yes
Consumer crash and replay | no | no | yes, via checkpoints | yes
Write retried after crash | no | no | no, needs upsert | yes
Forged click IDs | yes | no | no | no
Bot or click farm | no | no | partial, simple rules | yes, new rules
~~~

### Hot ads

A hugely popular ad (a Super Bowl campaign) can get thousands of clicks per second, all landing on one Kafka partition if you partition by ad ID, and that one partition's consumer becomes the bottleneck. The fix is a **salted key**: the partition key becomes `adId:N` with a random suffix N from 0 to 9. The ad's events now spread over 10 partitions, each worker counts its share, and a later step strips the suffix and sums the partial counts for the ad. Note that impression-ID dedupe still works if you key the dedupe step by impression ID before the salting step.

## Failure walk-through

~~~dg-timeline A retried click when the reply is lost: the edge cache catches the retry, and the raw-data batch is the backstop if it hadn't.
12:03:01.000 | Browser sends click, imp 7f3
12:03:01.020 | Processor verifies, appends to Kafka
12:03:01.022 | Processor sets 7f3 in Redis
12:03:01.030 | 302 reply lost on a flaky network
12:03:04.000 | Browser retries click, imp 7f3
12:03:04.015 | *Redis says 7f3 seen: redirect, no new event
12:04:00 | Flink closes minute 12:03: ad 9 = 418, upsert
Next day 02:00 | *Batch job recounts from S3, matches 418
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a retried click, and what happens if Redis was down**

The user clicks ad 9 on a slow network. The click with impression 7f3 reaches the processor, which writes it to Kafka, records 7f3 in Redis and responds, but the response is lost. The browser retries three seconds later. The processor verifies the signature, finds 7f3 in Redis and redirects without writing a second event.

Now replay the same story with Redis unavailable at both clicks. The processor accepts both, so Kafka holds two events for 7f3. Flink's keyed state sees 7f3 twice and drops the second. Suppose Flink also crashed and was restored from a checkpoint older than both events: they are replayed, but the restored keyed state did not yet contain 7f3, so again only one is counted, and the result row for (ad 9, 12:03) is upserted with the same value. Finally, the nightly job groups raw events by impression ID, finds two for 7f3, counts one, and confirms the stored figure. Three independent layers had to fail before the advertiser was overcharged by one click.

</div>

## Common mistakes

- **Using a client-generated ID unsigned.** It deduplicates honest retries and nothing else; attackers just send new IDs.
- **Marking seen before writing the event.** This turns a crash into a lost click.
- **Writing increments to the result store.** Any retry of the write double-counts. Write absolute values keyed by (entity, time bucket).
- **Calling Kafka "exactly-once" and stopping.** Exactly-once is a property of a whole pipeline, from source ID to final write, not of one component.
- **No raw data.** Without raw events there is no audit trail when an advertiser disputes a bill, and no way to apply a fraud rule retroactively.

## In an interview

Say clearly that ad counting is a correctness problem first and a scale problem second. Then lay out the layers in order: a signed impression ID as the idempotency key, a dedupe check at ingest that writes to the log before marking the ID, a checkpointed stream processor with keyed dedupe state, idempotent upserts keyed by ad and minute, and a daily reconciliation from raw events. A good question to ask: "Is the real-time number used for billing, or only for dashboards, with invoices from the reconciled batch?" The answer tells you how much of the exactness burden the streaming path carries.

<div class="take">
<div class="lab">Key takeaway</div>

give each impression a unique, signed ID and treat it as the idempotency key. Dedupe clicks against a cache of seen IDs (write to the stream first, then mark the ID), dedupe again in checkpointed keyed state, write results as idempotent upserts, and reconcile against raw events with a periodic batch job.

</div>
