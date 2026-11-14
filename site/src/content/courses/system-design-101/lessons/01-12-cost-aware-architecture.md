---
id: "01.12"
course: "system-design-101"
module: "01"
slug: "cost-aware-architecture"
title: "Cost-aware architecture"
keyIdeas: ["Treat cost as a requirement you design for, like latency.", "Big cost levers: storage tier (hot vs archive can be 10-20x apart), data egress between regions or to the internet, instance types and utilization, managed vs self-run services, and how much you replicate.", "Measure cost per unit of business value (per request, per user, per order) so you notice when it grows faster than usage.", "Idle capacity is the most common waste. Autoscaling, spot instances and right-sizing fix much of it.", "Cloud isn't always cheaper. Steady, predictable workloads can cost much less on owned hardware, as 37signals argued when leaving the cloud."]
question: "Name 2 design choices that change cost by 10x or more."
answer: "Examples of 10x choices: storing cold data in an archive or infrequent-access tier instead of SSD-backed databases or hot object storage (10 to 100 times cheaper per GB); keeping traffic in one zone and serving media through a CDN instead of paying cross-region or internet egress from origin; replacing per-request orchestration or per-item storage calls (like a billable workflow step per second of video) with in-process work; sampling logs and traces instead of keeping all of them; and running steady workloads on committed, spot or owned hardware instead of on-demand."
sources: [{"title": "Werner Vogels: The Frugal Architect", "url": "https://thefrugalarchitect.com/", "note": "7 laws, starting with \"make cost a non-functional requirement\"."}, {"title": "AWS: Cost Optimization Pillar", "url": "https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/welcome.html", "note": "The detailed version."}, {"title": "DHH: Why we're leaving the cloud", "url": "https://world.hey.com/dhh/why-we-re-leaving-the-cloud-654b47e0", "note": "A real cost argument against default cloud usage."}]
---

## Cost is a design requirement

Two designs can meet the same latency and availability targets and differ in monthly cost by ten or a hundred times. The difference is rarely in clever tuning. It comes from a few structural choices made early: where data is stored, how much of it moves across network boundaries, how much compute sits idle, and how many billable operations each user action triggers.

The first rule of cost-aware architecture is to **make cost a non-functional requirement**. Treat it like security or availability, from design through operations. A business whose infrastructure costs grow faster than its revenue is in trouble, however elegant the system is. In the cloud this matters even more, because every engineer can create costs with a single API call, and the bill arrives a month later.

## Seven principles

A widely used set of principles for cost-aware design:

1. **Make cost a non-functional requirement.** Write it down next to latency and availability: "under \$0.002 of infrastructure per order".
2. **Systems that last align cost to business.** Find the unit that drives revenue (orders, active users, gigabytes processed) and design so cost grows with it, not faster. A system whose cost per user rises as users grow has a structural problem.
3. **Architecting is a series of trade-offs.** Cost, resilience and performance conflict. Aim for the best value for each component, not the lowest bill.
4. **Unobserved systems lead to unknown costs.** If you can't see cost per feature, per service or per team, waste hides. Tag resources and report spend by owner.
5. **Cost-aware architectures implement cost controls.** Split systems into tiers by how critical they are, so less important parts can run on cheaper capacity or be scaled down when needed.
6. **Cost optimisation is incremental.** Keep revisiting; small savings compound at scale.
7. **Unchallenged success leads to assumptions.** Don't assume today's stack is still the cheapest way to do the job. Prices, workloads and products change.

## Where the money actually goes

Rough cloud prices (they vary by provider and region, but the ratios are what matter):

| Resource | Rough price | Notes |
| --- | --- | --- |
| Compute | ~\$25 per vCPU-month on demand | ~\$10 on spot; less with 1 to 3 year commitments |
| Memory | ~\$2.50 per GB-month | caching everything in RAM gets expensive |
| SSD block storage | ~\$0.08 to \$0.15 per GB-month | plus provisioned IOPS on some tiers |
| Object storage, standard | ~\$0.02 per GB-month | plus per-request charges |
| Object storage, archive | ~\$0.001 to \$0.004 per GB-month | retrieval is slow and charged |
| Transfer within a zone | free | |
| Transfer across zones | ~\$0.01 per GB each way | adds up with chatty services and replication |
| Internet egress | ~\$0.05 to \$0.09 per GB | CDNs are often cheaper per GB |
| Object storage requests | ~\$0.005 per 1,000 writes, ~\$0.0004 per 1,000 reads | small objects make this dominate |
| Log ingestion and indexing | ~\$0.50 per GB | verbose logging at scale is costly |

~~~dg-bars Price per GB-month across storage tiers, in US cents. Moving cold data from SSD to an archive tier is roughly a hundred times cheaper.
RAM | 250 | ~$2.50
SSD block storage | 10 | ~$0.10
Object storage, standard | 2.3 | ~$0.023
Object storage, infrequent | 1.25 | ~$0.0125
*Archive tier | 0.1 | ~$0.001
~~~

The big levers follow directly:

- **Idle and oversized compute** is the most common waste. A fleet sized for peak and left running all night, or instances at 8% average CPU, can double a bill. Autoscaling, right-sizing and spot or committed pricing each cut compute cost by 30 to 70%.
- **Storage tier.** Data that nobody has read for 90 days does not need to be on SSD or even standard object storage. Lifecycle rules that move it to infrequent-access and then archive tiers can cut storage cost by 10 times or more.
- **Data transfer.** Traffic between zones and out to the internet is charged per GB. A service that calls another in a different zone 10,000 times a second with 10 KB responses moves about 260 TB a month, roughly \$5,000 at a cent per GB in each direction.
- **Request counts.** Object stores and many managed services charge per operation. Millions of tiny objects or per-second orchestration steps can cost more than the storage itself. This is what hurt Prime Video's design (lesson 01.09).
- **Logs, metrics and traces.** Logging every request at debug level, or high-cardinality metrics with a label per user, can cost more than the servers being observed.
- **Managed service premiums.** Managed databases, search and queues cost more per unit of capacity than running the software yourself, in exchange for less operational work. That is often a good trade, but it should be a conscious one.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of a log line**

An API serves 20,000 requests per second. Each request writes a 1 KB structured log line.

Volume = 20,000 × 1 KB × 86,400 s ≈ 1.7 TB per day, about 52 TB per month.

At ~\$0.50 per GB ingested and indexed, that is about \$26,000 a month, more than the API servers themselves (perhaps 40 instances at ~\$150 each, about \$6,000).

Fixes: log successful requests at 1% sampling and every error in full (cuts volume by about 95%), drop bulky fields, and send raw logs to object storage at \$0.02 per GB for the rare deep investigation. The bill drops to around \$2,000 a month without losing what on-call engineers actually use.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**lifecycle rules for user uploads**

A photo app stores 2 PB of uploads, growing by 50 TB a month. 90% of reads hit photos uploaded in the last 30 days.

All on standard object storage: 2,000,000 GB × \$0.023 ≈ \$46,000 a month.

With a lifecycle rule (standard for 30 days, infrequent access until 1 year, archive after that), suppose 50 TB stays standard, 550 TB is infrequent and 1,400 TB is archive: 50,000 × 0.023 + 550,000 × 0.0125 + 1,400,000 × 0.004 ≈ \$1,150 + \$6,900 + \$5,600 ≈ \$13,700 a month, about 70% less. Retrieval fees for the rare old photo are small by comparison.

</div>

## Design choices that change cost by 10x

Some decisions move the bill by an order of magnitude, so they deserve attention at design time:

- **Storage tier for cold data**: SSD-backed database versus archive object storage is about 100 times.
- **Where bytes travel**: serving media from a CDN or keeping traffic inside one zone versus shipping it across regions or straight from origin to the internet.
- **Per-request orchestration versus in-process work**: one function call versus a billable workflow step or storage request per item.
- **Pricing model**: on-demand versus committed or spot for steady or interruptible workloads (2 to 4 times), and for steady, large workloads, rented versus owned hardware.
- **Precompute versus compute on read**: rendering a page once and caching it versus rebuilding it on every view.
- **Sampling** logs, traces and analytics events versus keeping everything.

~~~dg-matrix Design choices and the rough size of their effect on the bill. The biggest wins come from structure, not tuning.
Choice | Big saving? | Easy to do? | Low risk?
Archive tier for cold data | yes, 10x to 100x | yes, lifecycle rule | partial, slow retrieval
CDN in front of media | yes, 2x to 10x on egress | yes, config | yes, low
Spot or committed compute | yes, 2x to 4x | partial | partial, interruptions
Sample logs and traces | yes, 10x to 20x | yes, config | partial, less detail
In-process instead of per-step billing | yes, 10x or more | no, redesign | partial
Right-size idle instances | partial, 1.5x to 3x | yes | yes, low
~~~

## Make cost visible

You cannot optimise what you can't see:

- **Tag every resource** with team, service and environment, and report spend by tag weekly.
- **Track unit cost**: dollars per thousand orders, per active user, per GB processed. A rising unit cost is an early warning even while the total bill looks fine during growth.
- **Put cost in design reviews**: every new component's proposal includes its expected monthly cost at today's and next year's load.
- **Alert on anomalies**: a forgotten test cluster or a runaway job should page someone in a day, not show up at month end.

~~~dg-tree One way to attribute a monthly bill so that every dollar has an owner. Untagged spend is the first thing to chase.
Monthly bill $180k
  Checkout team $42k
    Compute $25k
    Database $14k
    Transfer $3k
  Media team $95k
    *Object storage $48k
    CDN egress $40k
    Transcoding $7k
  Platform $28k
    Logging $19k
  Untagged $15k | find owners or delete
~~~

## Cost controls by tier

Not every part of a system deserves the same spend. Split components by criticality:

- **Tier 1** (checkout, login): on-demand or committed capacity, multi-zone, generous headroom.
- **Tier 2** (recommendations, search suggestions): can degrade or be switched off under pressure.
- **Tier 3** (batch analytics, report generation, thumbnail regeneration): spot instances, off-peak schedules, can wait a day.

This gives you knobs: during a cost squeeze or a capacity crunch, you scale down tier 3 and degrade tier 2 without touching tier 1.

## The provider's view

The major cloud providers publish cost-optimisation guidance that adds practical habits: build cost awareness across teams, pay only for what you use and scale with demand, measure efficiency as business output per dollar, let managed services handle undifferentiated work, and attribute spend to teams and features.

## Cloud is not always cheaper

The opposite case also exists. 37signals (the company behind Basecamp and HEY) paid over \$500,000 a year for managed database and search services alone on AWS, and over \$3 million a year in total, and moved its steady workloads back to hardware it owns in rented data-centre space. The cloud shines in two cases: simple, low-traffic apps where the bill is small anyway, and highly irregular load, such as HEY's launch, which got 300,000 sign-ups in three weeks against a forecast of 30,000 in six months. For a mid-sized company with steady, predictable growth and an experienced operations team, owning hardware can be much cheaper over a few years.

~~~dg-compare Renting cloud capacity versus owning hardware. Neither is always cheaper; it depends on load shape and team skills.
[Cloud]
Spiky or unknown load -> Autoscale -> Pay per hour
+ No upfront cost, capacity in minutes
+ Managed services save operations work
- Premium for steady, predictable load
- Egress and request charges add up
[Owned hardware]
Steady load -> Buy servers -> Run for 5 years
+ Much cheaper per unit when busy all the time
+ No per-GB transfer charges inside
- Upfront spend, weeks to add capacity
- Needs a skilled operations team
~~~

Whether you agree with that move or not, the point is to **do the maths for your workload** instead of assuming. Steady-state compute that runs 24 hours a day for years is where owned or committed capacity wins. Bursty or uncertain load is where on-demand cloud wins.

## Common mistakes

- **Ignoring transfer costs** in a design that copies data across zones or regions.
- **Storing everything hot** forever.
- **Logging everything** at full detail.
- **No ownership of spend**, so nobody notices waste.
- **Optimising tiny line items** while a structural choice costs ten times more.

## In an interview

Mention cost when you choose storage tiers, replication across regions, or a CDN, with a rough number. "Five years of media is 55 PB; on standard object storage that's about \$1.2M a month, so we'll move anything older than 90 days to an infrequent-access tier" shows cost awareness without derailing the design.

<div class="take">
<div class="lab">Key takeaway</div>

design so cost tracks business value, make spend visible per team and feature, and focus on the structural levers that move the bill by 10x: storage tier, data transfer, idle compute, request counts and observability volume.

</div>
