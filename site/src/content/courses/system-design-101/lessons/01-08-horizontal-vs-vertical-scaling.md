---
id: "01.08"
course: "system-design-101"
module: "01"
slug: "horizontal-vs-vertical-scaling"
title: "Horizontal vs vertical scaling"
keyIdeas: ["**Vertical scaling**: bigger machine. Simple, no code changes, but has a ceiling, gets expensive at the top end, and stays a single point of failure.", "**Horizontal scaling**: more machines behind a load balancer. Near-unlimited and fault tolerant, but needs stateless services or partitioned state.", "Stateless web servers scale horizontally easily: move session data to a shared store (Redis, DB, signed cookies) and any server can handle any request.", "Databases are hard to scale horizontally because they hold state: you need replication for reads and sharding for writes, and both add consistency problems.", "Practical rule: scale vertically first while it's cheap, design so you can go horizontal later."]
question: "Why is horizontal scaling easy for stateless web servers and hard for a database?"
answer: "Stateless web servers keep no data of their own (sessions and files live in shared stores), so any server can handle any request, and scaling out is just adding servers behind a load balancer. A database holds state that must stay correct and consistent across copies, so adding machines means replication, which brings lag, failover risk and no extra write capacity, or sharding, which means splitting data by key, cross-shard queries and transactions, rebalancing and hot shards."
sources: [{"title": "ByteByteGo: Scale From Zero To Millions Of Users", "url": "https://bytebytego.com/courses/system-design-interview/scale-from-zero-to-millions-of-users", "note": "The vertical vs horizontal and load balancer sections."}, {"title": "Cockroach Labs: Vertical vs. horizontal scaling", "url": "https://www.cockroachlabs.com/blog/vertical-scaling-vs-horizontal-scaling/", "note": "Pros, cons, when to pick which."}]
---

## Two ways to add capacity

When a system runs out of capacity, you have two basic options:

- **Vertical scaling (scaling up)**: give the machine more CPU, more RAM, faster disks or a faster network. For example, move a database from 8 vCPUs and 32 GB of RAM to 32 vCPUs and 64 GB.
- **Horizontal scaling (scaling out)**: add more machines and spread the work across them. For example, go from 3 web servers to 12 behind a load balancer.

Both work, both have limits, and nearly every real system uses both: horizontally scaled stateless services in front of a database that is scaled vertically for as long as possible. Understanding why that combination is so common is the core of this lesson.

~~~dg-compare Vertical versus horizontal scaling: a bigger machine, or more machines behind a load balancer.
[Vertical]
Traffic -> 1 bigger server -> Limit
+ No code or architecture changes
+ One machine to operate
- Ceiling: the biggest machine sold
- Still a single point of failure
- Upgrades often need a restart
[*Horizontal]
Traffic -> Load balancer -(spread)-> N servers
+ Add capacity in small steps
+ One server failing is survivable
+ Capacity near users in many regions
- Software must be designed for it
- More moving parts to run
~~~

## Vertical: simple, but with a ceiling

Vertical scaling has a lot going for it:

- **No code changes.** The application doesn't know it's on a bigger machine. A database that was CPU-bound on 8 cores often runs comfortably on 32.
- **Simple operations.** One machine to monitor, back up and patch. No distributed coordination, no data spread across nodes.
- **No network between parts.** Everything on one box talks through memory, which is about a thousand times faster than a network round trip.
- **Often cheaper at first.** In the cloud, price per vCPU is roughly constant within an instance family up to fairly large sizes, so a 32-vCPU machine costs about four times an 8-vCPU one, without the overhead of extra load balancers and coordination.

And real limits:

- **A hard ceiling.** The largest cloud instances offer a few hundred vCPUs and a few TB of RAM (special high-memory instances reach tens of TB). Beyond that, there is nothing bigger to buy. On owned hardware, the top-end machines cost disproportionately more per core.
- **A single point of failure.** If that machine dies, the service goes with it. You can add a standby replica, but now you are already doing a form of horizontal scaling.
- **Downtime to resize.** Moving to a bigger instance usually means a restart or a failover, which is a few seconds to minutes of disruption.
- **Not everything scales with cores.** A workload limited by one hot lock, one single-threaded process or disk latency gets little from more CPUs.

## Horizontal: resilient, but more complex

Horizontal scaling adds machines:

- **Practically unlimited capacity**: keep adding nodes as traffic grows, in small, cheap steps.
- **Redundancy**: with 10 servers, losing one costs 10% of capacity, not 100%.
- **Elasticity**: autoscaling can add servers at the evening peak and remove them at night, so you pay for what you use.
- **Geography**: you can put capacity in several regions, close to users.

And costs:

- **More moving parts**: load balancers, service discovery, health checks, deployment orchestration, distributed monitoring.
- **The software must allow it.** This is the hard part. Any state a server keeps locally (sessions, uploaded files, in-memory caches, scheduled jobs) breaks when the next request lands on a different server.
- **Coordination overhead**: once servers must agree on anything, adding more of them adds communication. Throughput does not always grow linearly.

### Why scaling is rarely linear

If one server handles 1,000 requests per second, ten servers should handle 10,000. In practice they handle less, for two reasons. **Contention**: some part of the work is serialised through a shared resource (one database, one lock), and that part doesn't get faster with more servers. **Coherency**: nodes must exchange information to stay in agreement, and the amount of exchange can grow faster than the node count. A system where 5% of each request goes through a shared bottleneck can never exceed 20 times the throughput of one node, however many nodes you add, because that 5% sets the ceiling. The design goal of horizontal scaling is to make the shared part as close to zero as possible.

## How a web tier scales out

A standard path from one server to millions of users. Each step removes the current bottleneck:

~~~dg-timeline A typical scaling journey. Each stage is triggered by a measured bottleneck, not by fashion.
Stage 1 | one server runs app, DB and cache
Stage 2 | database moves to its own server
Stage 3 | load balancer plus several app servers
Stage 4 | *sessions move to a shared store: app tier is stateless
Stage 5 | DB primary plus read replicas
Stage 6 | cache and CDN in front
Stage 7 | queues and background workers
Stage 8 | database is sharded
~~~

1. **One server** runs the web app, the database and the cache. Fine for a prototype and often for the first thousands of users.
2. **Split the database** onto its own machine so the two tiers can be sized and scaled independently. The database usually wants RAM and fast disks; the app wants CPU.
3. **Put a load balancer** in front of several web servers. Clients see only the balancer's public IP; servers sit on private IPs. If one server fails its health check, traffic goes to the others.
4. **Make the web tier stateless.** Move session data out of the web servers into a shared store (Redis, the database) or into signed tokens the client carries. Store uploaded files in object storage, not on local disk. Now any server can handle any request, and you can add or remove servers freely.
5. **Add database replication**: one primary for writes, several replicas for reads. This multiplies read capacity, at the cost of replication lag.
6. **Add a cache** for hot data and a **CDN** for static files and media, taking most reads off the database and the app servers.
7. **Add message queues** so slow work (emails, image resizing, reports) happens asynchronously on separately scaled workers.
8. **Shard the database**: split the data across several primaries by key, once writes or data size outgrow one machine.

~~~dg-flow The stateless web tier: any app server can take any request because session state lives in a shared store.
Users -> Load balancer -> *App servers 1..N -(session lookup)-> Shared session store
App servers 1..N -(queries)-> Database
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why "stateless" matters**

A shop runs 4 app servers behind a round-robin load balancer, and each server keeps shopping baskets in its own memory.

A user adds an item on server 2. Their next request goes to server 3, which has no basket for them: the basket appears empty. Sticky sessions (always send a user to the same server) hide this, until server 2 is restarted for a deploy and every user pinned to it loses their basket. Autoscaling down has the same effect.

Moving baskets into Redis fixes both. Each request reads the basket by session ID (a 0.5 ms lookup), any server can serve any user, and deploys or scale-downs lose nothing.

</div>

## Why databases are the hard part

Stateless servers are interchangeable, so adding one is trivial: start it, register it with the load balancer, done. A database holds **state**, and every copy must agree on it. That turns scaling into a set of hard problems:

- **Scaling reads with replicas** introduces **replication lag**. A user writes to the primary, then reads from a replica that hasn't caught up, and their change seems to have vanished. You need read-your-writes routing or tolerance for stale reads (lesson 01.04).
- **Replicas don't help writes.** Every write must still go through the primary, and every replica must replay it. Ten replicas give you roughly ten times the reads and exactly the same write capacity.
- **Scaling writes requires sharding**: splitting the data across several primaries by key. That brings queries that span shards, transactions across shards, rebalancing when a shard fills up, and hot shards when one key is far more popular than others (lesson 04.06).
- **Failover is delicate.** Promoting a replica when the primary dies risks losing the last few acknowledged writes, or ending up with two primaries.

That is why a common rule is: **scale the database vertically for as long as it is comfortable, add read replicas for read-heavy loads, and design the data model so you can shard later** (choose a partition key early, avoid queries that need the whole dataset).

<div class="ex">
<div class="lab">Worked example</div>

**vertical first, then replicas**

A Postgres primary on 8 vCPUs handles 3,000 queries per second at 70% CPU. Traffic is 90% reads and is growing 10% a month.

Option A, scale up to 32 vCPUs. Roughly four times the CPU, so about 12,000 queries per second at the same utilisation. No code changes; a planned failover takes about 30 seconds. At 10% monthly growth, 4x lasts about 14 months.

Option B, add three read replicas of the same size and route reads to them. Read capacity grows about fourfold, but writes stay on one 8-vCPU primary (300 per second, plenty of headroom), and the app must handle lag.

Most teams do A first, because it is simpler, and B when the read share and growth justify it. Sharding only comes up if writes, or the total data size, outgrow the largest primary they can buy.

</div>

~~~dg-matrix How easily each tier scales out. State is what makes scaling hard.
Tier | Stateless? | Add a node | Main difficulty
Web or API servers | yes | yes, trivial | keep them stateless
Background workers | yes | yes, trivial | idempotent jobs
Cache cluster | partial, rebuildable | partial, keys remap | cold cache after changes
DB read replicas | no | partial, easy for reads | replication lag
DB write primary | no | no, needs sharding | cross-shard queries, rebalancing
~~~

## Choosing

- **Vertical** fits less critical workloads with predictable growth, where keeping cost and complexity low matters most, and stateful components such as databases where splitting is expensive.
- **Horizontal** fits critical systems that need high availability, room for large growth, elastic capacity, or presence in multiple regions, and anything stateless.

In practice most systems do both: horizontally scaled stateless services in front of a vertically scaled database, which is later replicated and, only if needed, sharded.

## Common mistakes

- **Sharding too early.** It's very hard to undo, and a single modern database server goes a long way.
- **Hidden local state**: files written to local disk, in-process caches assumed to be authoritative, cron jobs that run on every server at once.
- **Relying on sticky sessions** instead of fixing state, which makes load uneven and failures lossy.
- **Assuming replicas add write capacity.** They add read capacity only.
- **Forgetting the load balancer is a component too.** It needs redundancy of its own (lesson 02.06).

## In an interview

Say which tiers are stateless and scale them horizontally behind a load balancer without further discussion. Spend your time on the stateful parts: estimate whether the database fits on one machine, say when you would add replicas, and name the partition key you would shard on if the numbers demand it. "Writes peak at 2,000 per second, which one primary handles, so we'll scale it vertically and add replicas for reads; if writes grow tenfold we'll shard by user ID" is a complete answer.

<div class="take">
<div class="lab">Key takeaway</div>

scale stateless tiers horizontally behind a load balancer, after moving all state out of them. Scale the stateful database tier vertically first, then with read replicas, and shard only when writes or data size outgrow one machine.

</div>
