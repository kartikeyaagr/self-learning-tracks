---
id: "01.05"
course: "system-design-101"
module: "01"
slug: "availability-durability-consistency-cost"
title: "Availability, durability, consistency, cost"
keyIdeas: ["**Availability**: fraction of time the system answers correctly. 99.9% allows about 43 minutes of downtime per month; 99.99% about 4.3 minutes.", "**Durability**: probability that acknowledged data survives. Achieved with replication, backups and checksums. S3 quotes 11 nines.", "**Consistency**: whether every reader sees the latest write. Strong consistency needs coordination, which costs latency and availability during network partitions.", "**Cost**: each extra nine roughly multiplies cost (more replicas, regions, on-call effort).", "SLI = what you measure (success rate). SLO = your target (99.9%). SLA = a contract with penalties, set looser than the SLO.", "Error budget = 100% minus SLO. While budget remains, ship features; when it's spent, focus on reliability."]
question: "How many minutes of downtime per month does 99.9% allow? 99.99%?"
answer: "99.9% allows about 43 minutes of downtime a month (0.1% of 43,200 minutes is 43.2 minutes in a 30-day month; about 43.8 in an average month). 99.99% allows about 4.4 minutes a month, ten times less, which is short enough that failover and rollback must be automatic."
sources: [{"title": "Google SRE book: Embracing Risk", "url": "https://sre.google/sre-book/embracing-risk/", "note": "Why 100% availability is the wrong target; error budgets."}, {"title": "Google SRE book: Service Level Objectives", "url": "https://sre.google/sre-book/service-level-objectives/", "note": "SLIs, SLOs, SLAs."}, {"title": "Kleppmann: Please stop calling databases CP or AP", "url": "https://martin.kleppmann.com/2015/05/11/please-stop-calling-databases-cp-or-ap.html", "note": "Why CAP labels mislead."}, {"title": "Hello Interview: CAP Theorem", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/cap-theorem", "note": "Interview-oriented summary."}]
---

## Four properties that pull against each other

This lesson covers the four qualities you will trade most often: **availability** (does it answer?), **durability** (is my data still there?), **consistency** (is the answer up to date?) and **cost** (what does all this cost each month?). They pull against each other. More replicas improve availability and durability but cost money and create copies that can disagree. Stronger consistency means replicas must coordinate, which adds latency and can make the system refuse requests during failures.

You need to be able to put a number on each one, compute how components combine, and explain what you would give up to get more of one.

## Availability and the "nines"

**Availability** is the fraction of time, or of requests, for which a service works correctly. It is usually described in "nines": 99.9% is "three nines". The useful habit is to convert each target into allowed downtime, because "99.99%" sounds abstract while "4 minutes a month" is something a team can plan around.

The arithmetic: a 30-day month has 30 × 24 × 60 = 43,200 minutes. At 99.9%, 0.1% of that is 43.2 minutes. An average calendar month is slightly longer (30.44 days), which gives 43.8 minutes. A year has about 525,600 minutes, so 99.9% allows about 526 minutes, or 8.8 hours.

| Target | Downtime per month | Downtime per year |
| --- | --- | --- |
| 99% | 7.3 hours | 3.65 days |
| 99.9% | 43.8 minutes | 8.8 hours |
| 99.95% | 21.9 minutes | 4.4 hours |
| 99.99% | 4.4 minutes | 52.6 minutes |
| 99.999% | 26 seconds | 5.3 minutes |

~~~dg-bars Allowed downtime per year at each availability target. Each extra nine cuts the budget by ten times.
99% | 5256 | 3.65 days
99.9% | 526 | 8.8 hours
99.95% | 263 | 4.4 hours
99.99% | 52.6 | 52.6 minutes
*99.999% | 5.3 | 5.3 minutes
~~~

Think about what each level means for operations. At 99.9%, one bad deploy that takes 30 minutes to roll back uses most of a month's budget. At 99.99%, you cannot afford a human to be paged, wake up, log in and diagnose: 4 minutes is gone before they open a laptop. That level needs automatic failover and automatic rollback. At 99.999%, even a fast automatic failover that takes 30 seconds, happening once a month, exceeds the budget.

### Time-based vs request-based

Time-based availability ("the service was up 99.9% of the minutes") is easy to understand but crude: a 10-minute outage at 3 a.m. and one at the Black Friday peak count the same. For large services it is better to measure **aggregate availability**: the share of requests that succeed. A service handling 2.5 million requests a day at 99.99% may fail 250 of them. A partial outage where 5% of requests fail for an hour then counts as 5% of an hour, not a full hour.

### MTBF and MTTR

Availability can also be written as `MTBF / (MTBF + MTTR)`: **mean time between failures** divided by the total of that plus **mean time to repair**. This form shows the two levers. You can fail less often, or recover faster. A server that fails once a month (MTBF ≈ 720 hours) and takes 1 hour to replace has availability 720 / 721 ≈ 99.86%. Cut repair to 6 minutes (0.1 hours) with automatic failover and it becomes 720 / 720.1 ≈ 99.986%, without the hardware getting any better. In practice, fast detection and recovery are usually cheaper than preventing failures.

## Availability composes

Real requests pass through several components. How their availabilities combine depends on whether they are in sequence or in parallel.

**In sequence** (a request needs every component), the availabilities multiply. If a request goes load balancer → API → database and each is 99.9%, the total is 0.999 × 0.999 × 0.999 ≈ 0.997, or 99.7%. That is about 2.2 hours a month instead of 43 minutes. Every synchronous hop eats into the budget, which is why long chains of service calls hurt.

**In parallel** (redundant copies, where any one is enough), the chance that *all* copies are down multiplies. Two 99.9% replicas are both down with probability 0.001 × 0.001 = 0.000001, so availability is 99.9999%. This assumes the failures are **independent**. Two replicas in the same rack, on the same power feed, running the same buggy software version, are not independent: one bad deploy can take them both down at once. Real redundancy means different racks, zones or regions, and staggered deploys.

~~~dg-widget nines Components in series multiply their availability, so every extra hop adds downtime. Redundant copies win most of it back.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**availability of a checkout path**

A checkout request goes through a load balancer (99.99%), the checkout service (99.95%), the payment service (99.9%) and a database (99.95%), all synchronously.

Series: 0.9999 × 0.9995 × 0.999 × 0.9995 ≈ 0.9979, about 99.79%. That allows roughly 91 minutes of failures a month, worse than any single component.

Now run the payment service as two independent copies in different zones, so the request succeeds if either is up. That part becomes 1 − 0.001² = 0.999999. The chain becomes 0.9999 × 0.9995 × 0.999999 × 0.9995 ≈ 0.9989, about 99.89%, cutting expected downtime from about 91 to about 48 minutes a month.

The weakest links are now the checkout service and database at 99.95%. Making them redundant too is the next step, and each step costs more servers.

</div>

## Why not 100%?

100% is almost never the right target:

- **Each extra nine costs far more than the last.** Going from 99% to 99.9% might need a second server and a load balancer. Going from 99.99% to 99.999% might need multiple regions, automated failover tested monthly, and a team dedicated to reliability.
- **Users can't tell beyond a point.** Their home Wi-Fi, mobile network and phone already fail somewhere between 0.01% and 1% of the time. A 99.999% service behind a 99% network looks like a 99% service.
- **Reliability work competes with features.** Engineers improving the fifth nine are not building the product.
- **It is an economic decision.** If moving a \$1M-a-year service from 99.9% to 99.99% recovers 0.09% of revenue, about \$900, it is only worth doing if it costs less than \$900 a year. It almost certainly costs more.

## SLIs, SLOs, SLAs and error budgets

These terms describe how you measure and promise availability (and latency):

- **SLI (service level indicator)**: what you measure. "The share of HTTP requests that return a non-5xx response in under 300 ms."
- **SLO (service level objective)**: the target for that SLI, over a window. "99.9% over a rolling 30 days."
- **SLA (service level agreement)**: a contract with customers that has consequences, usually service credits, if you miss it. Set it looser than the SLO (for example SLA 99.5%, SLO 99.9%), so you notice and react before you owe money.

~~~dg-layers From measurement to promise: each layer builds on the one below.
SLA | contract with customers, with penalties, e.g. 99.5%
*SLO | internal target, e.g. 99.9% of requests OK over 30 days
SLI | the measurement, e.g. share of requests OK and under 300 ms
Error budget | the 0.1% that may fail; spend it on releases and experiments
~~~

The **error budget** is 100% minus the SLO. With a 99.9% SLO and 100 million requests a month, 100,000 requests may fail. While budget remains, teams ship freely, run experiments and take risks. When it is spent, releases slow down or stop and effort goes into reliability until the window rolls forward. This turns the eternal argument between "ship faster" and "be more stable" into a shared number nobody has to win by shouting.

Overachieving is a risk too. Google deliberately took its lock service Chubby down for planned outages when it had been "too reliable", because other teams had started building as if it never failed. When it eventually did fail, their systems broke. A planned outage forced those hidden assumptions out early.

<div class="ex">
<div class="lab">Worked example</div>

**spending an error budget**

SLO: 99.9% of requests succeed over 30 days. The service handles 50 million requests a month, so the budget is 50,000 failed requests.

Day 6: a bad config push fails 20% of requests for 15 minutes. At about 1,160 requests per minute, that is 0.2 × 1,160 × 15 ≈ 3,500 failures, 7% of the budget.

Day 19: a database failover takes 4 minutes with all requests failing: about 4,600 failures. Total spent so far: about 8,100, or 16%.

The team has plenty of budget left, so a risky schema migration goes ahead. Had a big outage on day 20 consumed 45,000 failures, the policy would freeze feature launches for the rest of the window.

</div>

## Durability

**Durability** is the probability that data you acknowledged is still there, uncorrupted, later. It is a different property from availability. A system can be down without losing data (the database is unreachable, but every byte is safe on disk). It can also be up while silently losing writes (it acknowledged a write, then a crash lost it before it reached disk).

Durability comes from three things:

- **Replication across independent failures**: copies on different disks, machines, zones or regions. A single disk has an annual failure rate of roughly 1 to 2%. Three copies on independent disks, with failed copies replaced within hours, make losing all three extremely unlikely.
- **Writing to stable storage before acknowledging**: a database that replies "saved" before calling `fsync` (forcing data out of OS buffers to the disk) can lose that write in a power cut. Many databases let you choose: acknowledge after the local disk, after one replica, or after a quorum.
- **Backups and checksums**: replication copies mistakes too. If someone runs `DELETE FROM orders` by accident, every replica obediently deletes. Point-in-time backups protect against that. Checksums detect **bit rot**, where stored bits change silently over time.

Object stores advertise very high durability: Amazon S3 is designed for 99.999999999% ("eleven nines") durability per object per year. In practical terms, if you store 10 million objects, you would expect to lose one about every 10,000 years. That number covers hardware failure, not you deleting your own data, so backups and versioning still matter.

~~~dg-matrix Which protection survives which failure. Replication alone does not protect against human error or bad software.
Protection | Disk dies | Zone outage | Accidental delete | Silent corruption
Single disk | no | no | no | no
3 replicas, one zone | yes | no | no | partial, if checksummed
3 replicas, 3 zones | yes | yes | no | partial, if checksummed
Point-in-time backups | yes | yes | yes | partial, if verified
Versioned object store | yes | yes | yes | yes, checksums built in
~~~

## Consistency, carefully

The **CAP theorem** says that during a network partition, a system must choose between consistency and availability. It is often quoted, and often misused, because its definitions are very narrow:

- "Consistency" in CAP means **linearisability**: the system behaves as if there were one copy of the data, and every read sees the latest completed write.
- "Availability" in CAP means **every non-failed node** must answer every request, even nodes cut off from the rest.

Under those definitions most real databases are neither "CP" nor "AP". A single-leader replicated database (one primary, async replicas) refuses writes on nodes that cannot reach the primary, so it is not CAP-available. Its async replicas can serve stale reads, so it is not linearisable either. Labelling it "CP" or "AP" tells you very little.

It is more useful to describe the actual guarantees:

- Can a user always **read their own writes**?
- Are reads **linearisable**, or can they be stale, and by how much?
- What happens during **failover**: can acknowledged writes be lost?
- Even without partitions, what latency do you pay for consistency? Synchronous replication to another region adds a cross-region round trip (often 50 to 150 ms) to every write.

Lessons in Module 6 cover these guarantees in detail.

## Cost

Each of the above costs money: replicas, cross-region links and data transfer, standby capacity that sits idle until a failover, backups, and the on-call time of engineers. A rough guide: going from one server to two in one zone roughly doubles compute cost; going multi-zone adds cross-zone transfer charges; going multi-region roughly doubles the whole stack again and adds replication traffic.

The right answer is the **cheapest design that meets the agreed SLOs**, not the most reliable one you can imagine. If the requirement is 99.9%, a well-run single-region, multi-zone deployment is usually enough. Proposing active-active across three regions for a 99.9% internal tool is a design mistake, not extra credit.

## Common mistakes

- **Quoting nines without converting them.** Always translate to minutes per month.
- **Forgetting series composition.** Five 99.9% services in a chain are not 99.9%.
- **Assuming independent failures.** Replicas in one zone, or with one shared config, fail together.
- **Confusing availability and durability.** Replicas do not replace backups.
- **Using CAP labels as a design.** Say what readers see and what happens on failover instead.

## In an interview

Ask for the availability target early, convert it to downtime, and use it to justify redundancy: "99.99% gives us about 4 minutes a month, so failover must be automatic." Mention what the SLI is (success rate, latency), and say what is redundant and across what failure domain (zone or region). For data, say how many copies, where, when a write is acknowledged, and how you recover from mistakes.

<div class="take">
<div class="lab">Key takeaway</div>

set availability from user and business needs, convert it to downtime, measure it as an SLO with an error budget, and reach it with independent redundancy and fast recovery. Keep durability separate (replicas plus backups), and describe consistency by concrete guarantees instead of CAP labels.

</div>
