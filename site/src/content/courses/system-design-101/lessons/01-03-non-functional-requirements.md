---
id: "01.03"
course: "system-design-101"
module: "01"
slug: "non-functional-requirements"
title: "Non-functional requirements"
keyIdeas: ["Non-functional requirements (NFRs) describe qualities: performance, scalability, availability, durability, consistency, security, cost, operability.", "Make each one measurable. \"Fast\" becomes \"p99 redirect latency under 100 ms\". \"Reliable\" becomes \"99.95% monthly availability, zero data loss for acknowledged writes\".", "NFRs drive architecture more than features do. A 10 QPS app and a 100k QPS app with the same features look completely different.", "Use percentiles (p50, p95, p99) for latency, not averages. Averages hide the slow tail that users actually notice.", "Rank them. You usually can't max out all of consistency, latency and cost at once, so agree which one wins."]
question: "Turn \"it should be fast\" into a measurable requirement."
answer: "Example: \"p99 latency for loading the home feed is under 200 ms, measured at the server, at 50k requests per second.\" It names the operation, the percentile, the threshold and the load, so a design can be checked against it and monitored in production."
sources: [{"title": "Hello Interview: Delivery Framework", "url": "https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery", "note": "The non-functional checklist: latency, scale, durability, CAP."}, {"title": "AWS Well-Architected Framework", "url": "https://docs.aws.amazon.com/wellarchitected/latest/framework/welcome.html", "note": "A vendor's take on quality attributes."}]
---

## Quality attributes drive architecture

Two systems with identical features can look nothing alike. A to-do app for 100 people and one for 100 million share the same feature list (add a task, tick it off, share a list) but need completely different designs. The small one is a single server with SQLite. The big one needs load balancers, a replicated and sharded database, caches, multiple regions and an on-call rota. The difference lives entirely in the **non-functional requirements** (NFRs): the qualities the system must have, rather than the things it does.

That's why experienced engineers spend so long on NFRs. Features tell you what to build; NFRs tell you **how hard** it is to build, and which of the many possible designs is right.

## The common categories

- **Performance / latency**: how quickly an operation completes for one user. "Search results in under 300 ms."
- **Throughput**: how many operations per second the system handles. "50,000 writes per second at peak."
- **Scalability**: how the system copes as load and data grow. Can you add machines and get proportionally more capacity?
- **Availability**: the fraction of time, or of requests, the system serves correctly. "99.95% of requests succeed each month."
- **Durability**: whether acknowledged data survives crashes, disk failures and disasters. "No acknowledged payment is ever lost."
- **Consistency**: whether readers see the latest write, and in what order. "A user always sees their own posts immediately."
- **Security and privacy**: who can do and see what; encryption; audit trails; legal rules like GDPR.
- **Operability**: how easy it is to deploy, monitor, debug and recover. A design nobody can debug at 3 a.m. is a bad design.
- **Cost**: what it costs to build and run, per month and per unit of business (per order, per active user).

A useful outside checklist is the AWS Well-Architected Framework's six pillars: operational excellence, security, reliability, performance efficiency, cost optimisation and sustainability. When reviewing a real design, walking through those six catches gaps.

## Make each one measurable

NFRs must be **specific to the system** and **quantified**. Vague words feel safe but can't be checked, so they can't guide a design.

~~~dg-compare Vague qualities versus measurable ones. Only the measurable column can be tested, monitored and used to reject a design.
[Vague]
It should be fast
It should be highly available
It should scale
It should never lose data
- Can't be tested
- Can't decide between two designs
[*Measurable]
p99 feed load under 200 ms at 50k requests/s
99.95% of redirects succeed per 30 days
100M daily users, 5× bursts during live events
Zero loss of acknowledged messages, even if a zone fails
+ Testable and monitorable
+ Tells you which designs fail
~~~

A measurable statement does two jobs:

- **It lets you check a design.** "Can a single Postgres primary take 50,000 writes per second with headroom?" Probably not, so the requirement forces batching, sharding or a different store.
- **It stops over-engineering.** "We need 99.9%" means multi-region active-active replication is unnecessary. You can save months of work and a lot of money.

A complete latency requirement names four things: **the operation**, **the percentile**, **the threshold** and **the load**. "p99 latency for loading the home feed is under 200 ms, measured at the server, at 50,000 requests per second." Drop any of the four and the statement gets slippery. "Under 200 ms" at what load? For which percentile? For which call?

## Use percentiles, not averages

Latency distributions have long tails. Most requests are quick, but a few hit a garbage-collection pause, a cold cache, a slow disk or a retry, and take ten or a hundred times longer. The average hides this completely.

Imagine 100 requests: 95 take 40 ms and 5 take 1,000 ms. The average is (95 × 40 + 5 × 1,000) / 100 = 88 ms, which sounds fine. But 1 in 20 users waits a full second. If a page makes 20 such calls, most page loads hit at least one slow one.

That's why latency is described with **percentiles**:

- **p50 (median)**: half the requests are faster than this. The "typical" experience.
- **p90 / p95**: the experience of the unlucky 1 in 10 or 1 in 20.
- **p99 / p99.9**: the bad-but-plausible case. At large scale, this is what many users hit every day.

~~~dg-bars One service's latency percentiles. The average (88 ms) hides that 1 in 100 requests takes over a second.
p50 | 40 | 40 ms
Average | 88 | 88 ms
p90 | 95 | 95 ms
p99 | 1050 | 1,050 ms
*p99.9 | 2200 | 2,200 ms
~~~

Design for the high percentile you care about. If the SLO is "p99 under 200 ms", a design that achieves a 50 ms average but a 600 ms p99 fails, and you should know before you build it. Tail latency gets its own lesson (02.12), because fan-out to many servers makes it much worse.

### A measurement detail that matters

Percentiles can't be averaged. If ten servers each report their own p99, the average of those ten numbers is not the fleet's p99. To get a correct fleet-wide percentile, collect **histograms** (counts of requests per latency bucket) and add them up, then compute the percentile from the combined histogram. Lesson 10.12 covers how monitoring systems do this.

## Throughput and the read/write ratio

For each operation, write down how often it happens. Reads and writes stress systems differently:

- **Read-heavy** (feeds, product pages, URL redirects at 100:1): caching, read replicas and CDNs are the main tools.
- **Write-heavy** (metrics, logs, IoT sensors, chat): append-friendly storage, batching, partitioning and queues.
- **Balanced** (collaborative editing, order processing): usually the hardest, because caching helps less.

Also note **burstiness**. A ticketing site might average 100 requests per second but receive 100,000 in the minute a big concert goes on sale. Designing for the average there is designing for failure.

## Pick and rank

Choose the top 3 to 5 NFRs. Listing all nine categories, each with a number, wastes time and hides what actually matters. Then **rank** them, because they conflict:

- Strong consistency costs latency (replicas must coordinate) and availability (you may refuse requests during a network fault).
- Very high availability costs money: redundant servers, zones and regions, all paid for even when idle.
- Low latency worldwide means data in many regions, which makes consistency harder and more expensive.
- Strong security (encryption everywhere, fine-grained auth checks) costs some latency and engineering time.

~~~dg-matrix Common NFR conflicts. Strengthening the row's quality usually costs the column's.
Strengthen | Latency | Availability | Cost | Simplicity
Strong consistency | no, coordination adds round trips | no, may refuse during partitions | partial | partial
High availability | yes | yes | no, redundant capacity | no, failover logic
Global low latency | yes | yes | no, many regions | no, data placement
Durability (sync replicas) | no, wait for copies | yes | no, extra storage | partial
~~~

Once the ranking is agreed ("a stale like count is fine but a double charge is not"), many later choices make themselves. The like counter can be eventually consistent and cached; the payment path must be strongly consistent and idempotent.

## Worked examples

<div class="ex">
<div class="lab">Worked example</div>

**chat app NFRs**

Latency: messages delivered to online recipients in under 500 ms at p99.

Durability: no acknowledged message is ever lost, even if a server or a zone fails.

Ordering: messages in one conversation appear in the same order for every participant.

Availability: sending stays available even if read receipts and typing indicators are delayed (those are best-effort).

Scale: 50 million daily users, 40 messages each per day: 2 billion messages a day, about 23,000 per second on average and perhaps 70,000 at peak.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**what the NFRs imply for the chat design**

"Never lose an acknowledged message" means writing to durable, replicated storage before acknowledging the sender. "Same order for everyone" means a per-conversation sequence number assigned in one place (a partition per conversation). "Typing indicators are best-effort" means they can skip storage entirely and go over the live connection only. Three short sentences, three big design decisions.

</div>

## Common mistakes

- **Averages instead of percentiles.** Hides exactly the requests users complain about.
- **Global NFRs.** "The system should be strongly consistent." Which parts? Usually only a few operations need it.
- **No load attached.** A latency target without a request rate means nothing.
- **Everything is top priority.** If nothing is ranked, conflicts get resolved by accident.

<div class="take">
<div class="lab">Key takeaway</div>

NFRs, not features, decide the architecture. Write them as numbers (operation, percentile, threshold, load), keep the list to 3 to 5, note read/write ratios and bursts, and agree which quality wins when they conflict.

</div>
