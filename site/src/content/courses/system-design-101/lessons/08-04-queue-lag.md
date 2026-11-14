---
id: "08.04"
course: "system-design-101"
module: "08"
slug: "queue-lag"
title: "Queue lag"
keyIdeas: ["Queue lag is how far consumers are behind producers: messages waiting, or offset difference in Kafka, or age of the oldest message.", "Age of oldest message is often the most useful alert, since it's what users feel.", "If lag grows steadily, consumers are slower than producers. Options: add consumers (up to the partition count in Kafka), make processing faster (batching, fewer remote calls), add partitions, shed or sample low-value messages.", "A big backlog can take hours to drain even after the cause is fixed. AWS suggests protecting fresh traffic (e.g. process newest first or separate queues) so the backlog doesn't hurt everyone."]
question: "Lag is growing at 1,000 msgs/s. What are your options?"
answer: "First work out why: has the producer rate risen, or has processing slowed (often a slow dependency or a hot partition)? Then raise throughput: add consumers (and partitions, since Kafka parallelism is capped by partition count), batch writes and lookups, cache, and put timeouts and concurrency limits on slow dependencies. Drain time is backlog divided by spare capacity, so aim for well above 1,000 msg/s of headroom. While draining, process fresh messages first and move old ones to a backlog queue, expire messages that are no longer useful, isolate noisy tenants, apply backpressure or rate limits on producers, and send repeatedly failing messages to a dead-letter queue. Alert on the age of the oldest message."
sources: [{"title": "Confluent: Monitor consumer lag", "url": "https://docs.confluent.io/platform/current/monitor/monitor-consumer-lag.html", "note": "What lag is and how to read it."}, {"title": "AWS Builders' Library: Avoiding insurmountable queue backlogs", "url": "https://builder.aws.com/content/3EuRcgkTP1MI0c7zM8W6HL3WIqA/avoiding-insurmountable-queue-backlogs", "note": "Why backlogs grow and how to drain."}]
---

## What lag is

A queue or stream sits between producers that add work and consumers that remove it. When the two rates match, the queue stays short. When producers are faster, work piles up. **Consumer lag** is the measure of that pile.

For a Kafka-style log, lag is defined per partition: the number of offsets between the **latest message produced** to the partition (the log-end offset) and the **last offset the consumer group committed**. If the partition's newest message is offset 10,000 and the group has committed 9,000, lag is 1,000 messages. Total lag for a group is the sum over all its partitions. Lag of 0 means caught up. Lag that stays flat at a small number is healthy: it's just the work in flight. **Growing** lag means messages arrive faster than they are processed.

For a task queue such as SQS or RabbitMQ, the equivalent is **queue depth**: messages waiting plus messages in flight (taken but not yet acknowledged).

~~~dg-flow Lag is the gap between the newest offset in a partition and the group's committed offset.
Offset 0 -> Committed 9,000 -(lag 1,000)-> *Latest 10,000 -(producers append)-> Next
~~~

### Count is not enough: measure age

Offset lag alone can mislead. 10,000 messages behind is nothing if you process 50,000 a second (a fifth of a second of delay). It's terrible if you process 100 a second (100 seconds). The useful signal is **time lag** or **message age**: how old the oldest unprocessed message is. That number is in the units your users feel. "Order emails are arriving 14 minutes late" is an incident; "lag is 84,000" needs translating first.

Two ways to get age:

- Many queues report it directly. SQS publishes `ApproximateAgeOfOldestMessage`.
- For Kafka, compare the timestamp of the message at the committed offset with the current time, or estimate it as offset lag divided by recent consumption rate.

Watch for retries distorting the metric. If a few messages are failing and being retried repeatedly, the "oldest message" may stay old forever even though everything else is flowing. AWS IoT, for example, tracks an age metric based on the *first* attempt, ignoring retries, so a handful of stuck devices don't trigger alarms. Separate "stuck messages" (handle with dead-letter queues) from "the whole queue is behind" (handle with capacity).

## The arithmetic of a backlog

The rule is simple. If messages arrive at rate λ (lambda) and are processed at rate μ (mu):

- Lag grows at **λ − μ** messages per second when λ > μ.
- A backlog of B drains at **μ − λ** per second once μ > λ, so it takes **B / (μ − λ)** seconds to clear.

The second formula is why backlogs are worse than they look. If you have just enough spare capacity, μ − λ is tiny and the drain takes forever.

<div class="ex">
<div class="lab">Worked example</div>

**draining after an outage**

A consumer fleet normally processes 1,000 messages a second, matching the arrival rate. It goes down for one hour. Backlog: 3.6 million messages.

If it comes back at exactly its normal capacity, μ − λ = 0, and the backlog never drains. Every message stays an hour late indefinitely.

If it has 20% headroom (1,200 a second), it drains 200 a second: 3,600,000 / 200 = 18,000 s = 5 hours of elevated latency after a one-hour outage.

If you scale to double capacity (2,000 a second), it drains 1,000 a second: one hour to recover. So after an hour-long outage you need roughly double capacity for an hour, and every downstream system (the database the consumer writes to, the APIs it calls) must also absorb double load during that hour.

</div>

~~~dg-bars Hours needed to drain a one-hour backlog, by spare consumer capacity. Small headroom means very long recoveries.
10% headroom | 10 | 10 hours
20% headroom | 5 | 5 hours
50% headroom | 2 | 2 hours
*100% headroom | 1 | 1 hour
200% headroom | 0.5 | 30 minutes
~~~

## Why backlogs are dangerous

Queues make systems durable, since work isn't lost during an outage, but they **turn outages into latency**. Queue systems are **bimodal**: with no backlog they're fast, and once a backlog builds, latency keeps climbing and takes a long time to come back down. A few things make it worse:

- **Retry storms.** Clients whose requests time out retry, adding more work to an already-full queue.
- **Stale work.** By the time a message is processed it may be useless: a "user is typing" indicator from 20 minutes ago, or a price update replaced by three newer ones.
- **Visibility timeouts.** If jobs slow down because a dependency is slow, they may outlast their lease, get redelivered, and run twice, doubling the load exactly when you can least afford it.
- **Rebalancing.** In Kafka, a consumer that takes too long between polls (`max.poll.interval.ms`, 5 minutes by default) is kicked from the group. Its partitions move to other consumers, and processing pauses during the rebalance. Overloaded consumers can rebalance repeatedly and make no progress at all.

~~~dg-timeline A typical lag incident: a slow dependency, a growing backlog, and a long tail of recovery after the fix.
10:00 | Profile service p99 rises from 20 ms to 400 ms
10:02 | Consumer throughput falls from 6,000 to 5,000 msg/s
10:05 | Lag 180,000 and climbing 1,000/s
10:20 | *Age alert fires: oldest message 4 min old
10:35 | Fix deployed: batched profile lookups
10:36 | Lag 2 million; now draining 3,000/s
10:47 | Lag back to normal
~~~

## Options when lag grows

Diagnose first. Is the **producer rate** up (a marketing push, a bot, a bug sending duplicates), or is the **consumer slower** than it was (a slow dependency, a bad deploy, a hot partition)? Dashboards of produce rate, consume rate and per-partition lag answer this in a minute. Then fix the rate before you manage the backlog.

### Raise throughput

- **Scale consumers.** Add workers. With Kafka, parallelism is capped by the number of partitions, because each partition goes to one consumer in a group. With 10 partitions and 10 consumers already, an 11th does nothing; you need more partitions first (and remember that changes key-to-partition mapping). Inside each consumer you can also process a partition's messages with a pool of threads, as long as per-key ordering is kept (hash by key into the pool).
- **Make each message cheaper.** Batch database writes (one insert of 500 rows instead of 500 inserts), batch lookups to other services, cache reference data, remove synchronous calls that aren't needed.
- **Find the slow dependency.** A backlog often means a downstream service slowed down. Put concurrency limits and timeouts per dependency so one slow API can't occupy every worker thread; if it's down, fail fast and retry later.
- **Fix hot partitions.** If one key (one huge tenant) sends most traffic, its partition lags while others are idle. Change the key, add a random suffix for that key, or give that tenant its own topic.

### Manage the backlog

- **Prioritise fresh work.** When fresh data matters more than old, move old messages to a separate backlog queue and process new ones first, which behaves like LIFO. Users see the system working again immediately, and the backlog drains with spare capacity.
- **Drop or expire.** With a TTL, discard messages that are no longer useful, such as stale state updates that a later full sync replaces, or notifications about events that are now over.
- **Isolate tenants.** Separate queues per tenant, or **shuffle sharding**: give each customer a small random set of queues out of a pool (say 2 of 8), and send each message to the less loaded of its two. Two customers rarely share both queues, so one noisy customer's backlog only slows the few others that overlap with it.
- **Sideline excess.** Move a tenant's over-limit messages to a spillover queue processed at low priority.
- **Backpressure and rate limits.** Make producers slow down, or reject new work with `429 Too Many Requests` or `503`, when the backlog is too big, where that's acceptable. Rejecting early is better than accepting work you'll finish too late to matter.
- **Dead-letter queues.** Move messages that keep failing aside after N attempts instead of retrying them forever.
- **Heartbeats.** If a job may outlast the visibility timeout, the worker must extend its lease, or it will be redelivered and run twice.

~~~dg-flow Shuffle sharding: each tenant maps to a few queues out of a pool, so one noisy tenant only clogs its own.
[Tenant A] Tenant A -(hash)-> Queues 1 and 4 -(less loaded)-> Queue 4
[Tenant B, noisy] Tenant B -(hash)-> *Queues 2 and 5 -> Backlog stays there
[Tenant C] Tenant C -(hash)-> Queues 3 and 5 -(less loaded)-> Queue 3
~~~

### Leave headroom

Plan consumer capacity well above normal load, often 2 to 3 times, so you can drain a backlog quickly. Autoscaling on queue age helps, but it reacts in minutes, so it doesn't replace headroom. Also give downstream systems the same headroom, or your fast drain will just move the outage to the database.

~~~dg-matrix Which lever fits which situation.
Lever | Producer spike | Slow dependency | Poison messages | One noisy tenant
Add consumers | yes | partial, adds load on it | no | partial
Batch and cache | yes | yes | no | yes
Backpressure | yes | yes | no | partial
Dead-letter queue | no | no | yes | no
Tenant isolation | partial | no | no | yes
Expire stale work | yes | yes | no | partial
~~~

<div class="ex">
<div class="lab">Worked example</div>

**lag growing at 1,000 a second**

Producers send 6,000 messages a second and consumers handle 5,000, so lag grows by 1,000 a second: 3.6 million an hour. The consumer group has 10 consumers and 10 partitions, so adding consumers won't help. Per-partition lag is roughly even, so it isn't a hot key. Traces show each message makes a synchronous call to a profile service whose p99 rose to 400 ms.

The team batches 100 profile lookups per call. Throughput rises to 9,000 messages a second, so the backlog drains at 9,000 − 6,000 = 3,000 a second. If lag had reached 3.6 million, it clears in 3,600,000 / 3,000 = 1,200 s, about 20 minutes. Afterwards they raise partitions to 30 so they can add consumers next time, and add an alert on message age above 2 minutes.

</div>

## Monitoring checklist

- **Age of the oldest message** per queue or consumer group: the main alert.
- **Produce rate and consume rate**: tell you which side changed.
- **Lag per partition**: reveals hot partitions and stuck consumers.
- **Rebalance count** (Kafka) and **redelivery count** (queues): signs of timeouts.
- **DLQ depth**: anything above zero needs a person.

## Common mistakes

- Alerting on message count with one threshold for all queues. Use age, per queue, with thresholds tied to user impact.
- Adding consumers beyond the partition count.
- Scaling consumers without checking the downstream database can take the load.
- Retrying failures in place forever, blocking a partition behind one bad message.
- Running at 95% utilisation in normal times, so any outage takes all day to recover from.

## In an interview

When asked "lag is growing, what do you do?", answer in order: find out which rate changed; raise consumer throughput (scale up to the partition limit, batch, fix the slow dependency); then protect users while you drain (prioritise fresh work, expire stale work, isolate tenants, backpressure, DLQ). Do the drain-time arithmetic out loud: it shows you understand why headroom matters.

<div class="take">
<div class="lab">Key takeaway</div>

lag is produced minus processed; alert on the age of the oldest message, not just the count. When lag grows, find which rate changed, raise consumer throughput first, then shed, sideline or prioritise work, and keep enough headroom that a backlog drains in minutes rather than hours.

</div>
