---
id: "02.11"
course: "system-design-101"
module: "02"
slug: "backpressure"
title: "Backpressure"
keyIdeas: ["Backpressure is a way for a slow consumer to tell a fast producer to slow down.", "When producers outrun consumers you have 3 choices: **slow the producer** (backpressure), **buffer** (queue, bounded in size), or **drop** (shed or sample).", "Unbounded buffers just hide the problem until memory runs out or latency becomes huge.", "Mechanisms: TCP flow control windows, bounded queues that block or reject when full, reactive streams' request(n), HTTP 429/503 responses.", "Pick per data type: drop for metrics samples, buffer for orders, push back for file uploads."]
question: "A producer is 2x faster than its consumer. Name 3 ways to handle it."
answer: "(1) Control the producer: signal it to slow down (flow control, a blocking bounded queue, 429 responses) or let the consumer pull at its own pace, which loses nothing. (2) Buffer in a bounded queue to absorb bursts, alarming on the age of the oldest item, accepting that a sustained 2x imbalance will still fill it. (3) Drop or sample excess items when only recent data matters, such as position updates or metrics. You can also scale out the consumer to match the producer's rate, if the bottleneck isn't a shared resource."
sources: [{"title": "Jay Phelps: Backpressure explained", "url": "https://medium.com/@jayphelps/backpressure-explained-the-flow-of-data-through-software-2350b3e77ce7", "note": "Control, buffer, drop: the 3 options."}, {"title": "AWS Builders' Library: Avoiding insurmountable queue backlogs", "url": "https://builder.aws.com/content/3EuRcgkTP1MI0c7zM8W6HL3WIqA/avoiding-insurmountable-queue-backlogs", "note": "What happens when producers outrun consumers."}]
---

## What backpressure is

**Backpressure** is resistance to the flow of data through software: input arrives faster than the system can turn it into output. The name comes from fluid in pipes, where a narrow section pushes back against the flow behind it. A web server receives more requests than its database can answer; a log shipper reads files faster than it can upload them; a producer publishes events faster than the consumer can process them.

Backpressure is not a bug in itself. It is a fact of any pipeline whose stages run at different speeds, and you must handle it somewhere. If you don't choose how, the system chooses for you, usually by **running out of memory**: the excess piles up in an unbounded buffer until the process is killed, losing everything in the buffer at once.

The classic illustration is the sitcom scene of a worker on a chocolate factory conveyor belt that runs too fast. The worker first sets chocolates aside to deal with later (**buffering**), then starts eating and hiding them (**dropping**). The proper fix would have been to slow the belt (**controlling the producer**).

~~~dg-flow A producer that outruns its consumer. The bounded buffer between them fills, and something must give.
Fast producer -(1,000 items/s)-> *Bounded buffer, filling at 800/s -(200 items/s)-> Slow consumer
~~~

## The three strategies

There are three fundamental responses, plus scaling.

**1. Control the producer.** Tell the producer to slow down, or make it wait. Nothing is lost and memory stays flat. This is the best option whenever it is possible. Examples:

- **TCP flow control**: the receiver advertises how much buffer space it has left, and the sender stops when the window is zero. That's backpressure across machines, built into every TCP connection.
- **Node.js streams**: `writable.write()` returns `false` when the writer's internal buffer is above its high-water mark, and `pipe()` pauses reading until the `drain` event.
- **Blocking queues**: a producer thread calling `put()` on a full bounded queue simply blocks until there is room.
- **gRPC and HTTP/2** have per-stream flow-control windows; **RSocket** and **Reactive Streams** have explicit demand signals.

It is often impossible with users: you can't tell people to click less, or stop a sensor from measuring. But you can push back on them with a fast error (429 or 503) that tells their client to back off (lesson 02.13).

**2. Buffer.** Absorb bursts in a queue and let the consumer catch up when the burst passes. This is great for **short spikes**: a traffic burst that lasts seconds, a consumer pausing for a deploy. It doesn't help with a **sustained** imbalance: if the producer is always faster, the buffer just grows until it is full.

**Unbounded buffers are a leading cause of servers crashing with out-of-memory errors.** Consider copying a file by reading at 150 MB/s and writing at 100 MB/s, with an unbounded buffer in between: memory grows by 50 MB every second. A 6 GB file takes 40 seconds to read, and by then about 2 GB is sitting in memory. Always **bound buffers**, by size, by count, or by age.

**3. Drop.** Discard or sample some of the input. This is fine for data where only the latest value matters or where a sample is enough: mouse movements, metrics samples, live position updates, video frames in a real-time call. It is unacceptable for payments, orders or anything that someone will look for later. Choose *what* to drop: the oldest items (the newest are fresher), the newest (keep what you have already committed to), or a random sample.

**And: scale out, or accept it.** Add more consumers, if the work can be parallelised and the bottleneck isn't something shared like a database. Or accept a growing backlog when it causes no real harm (a nightly report that finishes an hour later).

~~~dg-compare The three ways to handle a producer that is faster than its consumer. Which one fits depends on whether you can slow the source and whether every item matters.
[*Control the producer]
Producer -(waits or is told 429)-> Consumer pace
+ Nothing lost, memory flat
- Not possible for users or sensors
- Slowness spreads upstream
[Buffer]
Producer -> Bounded queue -> Consumer
+ Absorbs short bursts
- Adds latency
- Must be bounded, or it ends in OOM
[Drop or sample]
Producer -> Keep latest or a sample -> Consumer
+ Fast, memory flat
- Loses data: only for replaceable data
~~~

### Pull beats push

**Pull-based** systems handle backpressure naturally, because the consumer asks for the next N items only when it is ready: Reactive Streams' `request(n)`, async iterators, Kafka consumers polling for the next batch, workers pulling jobs from a queue. The consumer's pace sets the flow.

**Push-based** systems (webhooks, a server streaming events to clients, a message broker pushing to subscribers) must add explicit signals: credit or demand counts, acknowledgements with a maximum number unacknowledged (RabbitMQ's **prefetch** count), or HTTP 429 responses.

~~~dg-seq Pull-based flow control. The consumer grants credit for a few items at a time, so the producer can never get far ahead.
Consumer > Producer : request(3)
Producer -> Consumer : item 1
Producer -> Consumer : item 2
Producer -> Consumer : item 3
Producer > Producer : no credit left, wait
Consumer > Consumer : process items 1 to 3
Consumer > Producer : request(3)
Producer -> Consumer : item 4
~~~

## When the buffer is a queue between services

A message queue between services (SQS, RabbitMQ, Kafka) is a buffer, and it needs special care. A queue lets producers keep succeeding while consumers are down or slow, which is often exactly the point. But **the work doesn't go away**: it waits, and someone has to catch up.

Queue systems are **bimodal**: they behave well when the backlog is near empty, and badly for a long time once it is not. Every message now waits behind the backlog, so latency jumps from milliseconds to minutes or hours, and it stays high until the backlog drains.

<div class="ex">
<div class="lab">Worked example</div>

**catching up after an outage**

Producers send 1,000 messages per second. Consumers normally process 1,000 per second, with capacity for up to 1,200.

Consumers fail for one hour. The backlog grows to 1,000 × 3,600 = 3.6 million messages.

When consumers return, they run flat out at 1,200 per second while 1,000 per second keep arriving, so the backlog shrinks by only 200 per second. Draining 3.6 million takes 18,000 s: **five hours**, during which every new message waits behind the backlog.

With capacity for 2,000 per second (double), it would drain at 1,000 per second, in one hour. The rule of thumb: after an outage of T, you need double your normal capacity for another T just to catch up.

</div>

~~~dg-timeline A one-hour consumer outage with only 20% spare capacity. The backlog takes five times longer to drain than it took to build.
10:00 | consumers crash, backlog starts growing
11:00 | *consumers back: 3.6 million messages waiting
12:00 | backlog 2.9 million, new messages wait ~40 min
14:00 | backlog 1.4 million
16:00 | backlog drained, latency back to normal
~~~

### Techniques for queues

- **Measure the age of the oldest message**, not just the count. "50,000 messages" means nothing without the processing rate; "oldest message is 12 minutes old" tells you exactly how far behind you are, and is what users feel.
- **Per-customer fairness**: in a multi-tenant queue, one customer's spike delays everyone. Rate-limit each producer, or **shuffle-shard** customers across several queues, so each customer maps to a small random subset of queues and a noisy one can only fill those.
- **Overflow and sideline queues**: move a heavy customer's excess, or old messages, to a separate queue so fresh work keeps flowing, then process the sideline when there is capacity.
- **Newest first**: for real-time data, process recent messages first and the backlog later (or never).
- **Drop stale messages** with a time to live (TTL) when newer data supersedes them.
- **Backpressure to producers**: reject or throttle new work for a workload whose backlog is too deep.
- **Limit concurrency per workload** using **Little's Law**: work in flight = arrival rate × time each item takes. At 100 messages per second, if a downstream call slows from 100 ms to 10 s, in-flight work grows from 10 to 1,000. Without a cap, that means 1,000 threads or connections, which can exhaust the consumer or crush the downstream.
- **Heartbeat long jobs**: extend the visibility timeout while working, so the queue doesn't redeliver the message to a second worker while the first is still busy.
- **Dead-letter queues** for messages that fail repeatedly (a "poison" message that always crashes the consumer), with an alarm, so one bad message doesn't block or loop forever.

~~~dg-bars Work in flight at a steady 100 messages per second as the time per message grows. Little's Law: in flight = arrival rate x latency.
100 ms per message | 10 | 10 in flight
1 s | 100 | 100
5 s | 500 | 500
*10 s | 1000 | 1,000 in flight
~~~

<div class="ex">
<div class="lab">Worked example</div>

**notification fan-out**

A celebrity posts, creating 10 million push notifications within a minute. The push workers can send 50,000 per second, so the burst takes over three minutes to send even at full speed.

The fan-out service writes to a bounded queue rather than calling the push service directly. Workers pull at their own pace (pull-based control), so the push provider is never asked for more than it can take.

Each notification has a TTL of 30 minutes: if the backlog ever grows beyond that, stale notifications are dropped rather than buzzing phones hours after the event.

Celebrity fan-outs go to a separate queue from ordinary users' notifications, so a direct message from a friend isn't stuck behind 10 million celebrity alerts. The alarm watches the oldest message's age in each queue.

</div>

## Common mistakes

- **Unbounded buffers and queues** "because we don't want to lose anything", which loses everything when the process dies.
- **Alarming on queue length instead of age.**
- **Scaling consumers into a shared bottleneck**: 50 consumers hammering one database go no faster than 10.
- **No dead-letter queue**, so one poison message blocks a partition.
- **Treating a queue as infinite capacity** that removes the need for capacity planning.

## In an interview

Whenever two stages run at different rates, say what happens when the fast one outruns the slow one: "workers pull from a bounded queue; we alarm on the oldest message's age; we drop location updates older than 30 seconds; and the API returns 429 when the backlog for a tenant passes a limit". Naming all three strategies, and choosing per data type, is what interviewers look for.

<div class="take">
<div class="lab">Key takeaway</div>

when producers outrun consumers you can slow the producer, buffer within a limit, or drop what is replaceable, and pull-based designs make the first option natural. Unbounded buffers only postpone a crash. In queues, measure the age of the oldest message, plan spare capacity to catch up after outages, and isolate noisy producers.

</div>
