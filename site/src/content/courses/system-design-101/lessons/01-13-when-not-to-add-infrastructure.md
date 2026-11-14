---
id: "01.13"
course: "system-design-101"
module: "01"
slug: "when-not-to-add-infrastructure"
title: "When not to add infrastructure"
keyIdeas: ["Every new component (cache, queue, search cluster, new database) adds operational load: monitoring, upgrades, failure modes, on-call knowledge.", "\"Choose boring technology\": you have a few innovation tokens. Spend them where they give a real advantage, and use proven tools elsewhere.", "Before adding infrastructure, know the current load and the current bottleneck. If Postgres handles 500 QPS at 20% CPU, a cache solves nothing yet.", "Single machines are very powerful. A laptop with good code can beat a distributed cluster on many data tasks (the COST paper, the 235x Hadoop example).", "Add complexity when a measured requirement forces it, and say which requirement."]
question: "Before adding a cache or queue, what 2 numbers should you know first?"
answer: "Know the current load and its growth (for example peak requests or writes per second, data size, and monthly growth rate), and where the bottleneck actually is, measured (for example database CPU at peak, or the p99 time of the slow query). Without both you can't show the new component solves a real problem, or that a simpler fix such as an index or a bigger instance wouldn't."
sources: [{"title": "Dan McKinley: Choose Boring Technology", "url": "https://mcfunley.com/choose-boring-technology", "note": "Innovation tokens and the cost of each new tool."}, {"title": "Command-line tools can be 235x faster than your Hadoop cluster", "url": "https://adamdrake.com/command-line-tools-can-be-235x-faster-than-your-hadoop-cluster.html", "note": "One laptop vs a cluster."}, {"title": "Frank McSherry: Scalability! But at what COST?", "url": "https://www.frankmcsherry.org/graph/scalability/cost/2015/01/15/COST.html", "note": "A laptop beating distributed systems."}]
---

## Every component has a running cost

Adding a cache, a queue, a search cluster or a new database solves a problem. It also adds a thing to deploy, monitor, upgrade, secure, back up, capacity-plan and understand at 3 a.m. when it misbehaves. Each new component brings its own failure modes: a cache can serve stale data or stampede the database when it empties; a queue can back up silently; a second database can drift out of sync with the first.

Design interviews and real architecture reviews both reward knowing when **not** to add something. The simplest design that meets the requirements, with evidence that it does, beats a diagram full of boxes that nobody needed.

~~~dg-timeline The lifetime of a new component. Building it takes weeks; running it takes years.
Week 1 | prototype works, team is happy
Month 1 | production deploy, dashboards, alerts
Month 4 | first incident: memory full, nobody knows the tuning
Year 1 | major version upgrade, breaking changes
Year 2 | *original author leaves, knowledge goes with them
Year 3 | security patch needed across 4 environments
~~~

## Choose boring technology

A useful mental model: a company gets roughly three **innovation tokens**. Each unusual technology choice (a new database, a new language, a service discovery tool that is a year old) spends one. Most companies are not database companies. Their limited attention is better spent on their actual product than on learning the quirks of a novel datastore under production load.

"Boring" does not mean bad. Postgres, MySQL, Python, Java, Memcached, Redis and cron are boring because their behaviour and failure modes are well understood. They still have problems, but mostly **known unknowns**: you know they exist, there are answers on the internet, and someone on the team has seen them before. New technology has far more **unknown unknowns**: failure modes nobody has hit yet, in combinations nobody has tried.

The key point is that **the cost of a technology is mostly in operating it over years, not in building with it**. Picking "the right tool for each job" looks optimal locally, one problem at a time, but leaves the company running dozens of tools, each with its own monitoring, deployment, testing, upgrade path and expertise. The sum of many locally optimal choices is a globally expensive system.

~~~dg-compare Two ways to solve a new problem. The boring option is often harder at first and much cheaper over years.
[New tool for the job]
Problem -> New datastore -> Tuned for this case
+ Fits the problem neatly
+ Interesting to build
- New monitoring, backups, upgrades
- Unknown failure modes
- One more thing on call must know
[*Existing stack]
Problem -> Postgres plus app code -> Good enough
+ Known failure modes, known fixes
+ Shared tooling and expertise
- More work up front
- May hit limits sooner
~~~

At Etsy, the activity feed was built on the existing PHP, MySQL, Memcached and Gearman (a job queue) stack instead of adopting Redis, which looked like a better fit. It was harder at first. But because it ran on shared, well-understood infrastructure that every engineer knew how to operate, it grew about 20 times without dedicated attention. Etsy did adopt Solr for faceted search, because doing that on MySQL was impractical. The rule is not "never add anything"; it is "make each addition earn its place".

Before adding a technology, ask:

- **How would we solve this without anything new?** If the honest answer is "we could, but someone wants to try the new thing", stop.
- **Write down exactly what makes the current stack too costly** for this problem, with numbers.
- **If it replaces something, commit to migrating fully off the old thing.** Two systems doing the same job is worse than either alone.

## Your existing stack can do more than you think

Many components that teams reach for have a "good enough" version inside tools they already run:

- **Queue**: a Postgres table with `SELECT ... FOR UPDATE SKIP LOCKED` lets many workers each claim different jobs without blocking each other. It handles hundreds to a few thousand jobs per second, and the job is enqueued in the same transaction as the business change, which a separate broker can't give you.
- **Cache**: the database's own buffer cache already keeps hot rows in memory. A well-indexed query on a cached page often takes under a millisecond.
- **Search**: Postgres full-text search with a GIN index covers simple search over millions of rows.
- **Pub/sub**: Postgres `LISTEN/NOTIFY` or Redis pub/sub for low-volume notifications.
- **Scheduled jobs**: cron, or a scheduler library, before a workflow engine.

These hit limits eventually, and when they do you will have the numbers that justify the dedicated tool.

## Single machines are very powerful

Two well-known experiments show how far one machine goes.

**235 times faster than Hadoop.** One experiment reproduced a job that counted chess game results (white wins, black wins, draws) from about 2 million games. A 7-node Hadoop cluster took about 26 minutes. A shell pipeline on a laptop (`find | xargs -P4 mawk ...`) took about 12 seconds, on twice as much data. It won by streaming the data line by line with almost no memory, using all cores through parallel processes, and skipping the cluster's job startup, scheduling and shuffle overhead.

**Scalability, but at what COST?** Another experiment compared big graph-processing systems (Spark, Giraph, GraphLab, GraphX), running on 128 cores, against a single-threaded program on a laptop. For PageRank and connected components on graphs with billions of edges, the single thread was often faster than all of them. With a better algorithm (union-find instead of label propagation) and a memory-friendly data layout, connectivity dropped to 15 to 30 seconds, against hundreds or thousands of seconds for the clusters. The point: a system can "scale well" (get faster as you add machines) and still be inefficient, because it started from a very slow baseline. The metric that came out of it is **COST**, the "configuration that outperforms a single thread": how many cores a distributed system needs before it beats one good thread. For some systems the answer was hundreds; for some, never.

~~~dg-bars Time to run the same chess-results job, in seconds. A laptop pipeline beat a 7-node cluster by over 100 times.
7-node Hadoop cluster | 1560 | ~26 minutes
*Laptop shell pipeline | 12 | ~12 seconds
~~~

Today a single cloud server can have 192 cores, 1 to 2 TB of RAM and tens of TB of NVMe storage. Many companies' entire datasets fit on one such machine. Before you design a distributed system, check whether you have a distributed problem.

## Know your numbers first

Before adding infrastructure, you need two pieces of evidence:

1. **The load**: current traffic and data size (requests per second at peak, writes per second, dataset size), and its growth rate.
2. **The bottleneck**: where time or capacity is actually going, measured, not guessed. Database CPU at 85%? p99 query time of 400 ms on one endpoint? Disk I/O saturated during batch jobs?

With those two, the decision is usually obvious:

- If Postgres serves your peak at 20% CPU with good p99s, a cache adds risk (staleness, invalidation bugs, a new failure mode) without benefit.
- If the whole dataset fits in RAM on one box, a distributed system may add only overhead.
- If one endpoint is slow because of a missing index, the fix is the index, not a cache in front of the slow query.
- If growth is 5% a month and you are at 30% of capacity, you have roughly two years before it becomes urgent.

~~~dg-tree A decision path before adding a cache, queue or new datastore. Most branches end without new infrastructure.
Is there a measured problem?
  No | stop: nothing to fix
  Yes: what is the bottleneck?
    Slow query | add an index or fix the query
    CPU or IO saturated
      Will scaling up fix it? | bigger instance first
      *Still short at peak plus growth | add the component, with numbers
    Work too slow for the request | queue, or a table plus workers
~~~

<div class="ex">
<div class="lab">Worked example</div>

**do we need a cache?**

A product API peaks at 3,000 requests per second. Each request makes two indexed queries. Postgres, on 16 vCPUs, sits at 25% CPU at peak, and p99 query time is 4 ms. Traffic grows 4% a month.

Headroom: at 25% CPU, the database could take about three times the load before reaching a comfortable 75%. At 4% monthly growth, tripling takes about 28 months (1.04²⁸ ≈ 3).

Conclusion: no cache now. A cache would save a few milliseconds per request and add invalidation logic and a new on-call component. Revisit when CPU at peak passes 50%, or if a requirement asks for sub-millisecond reads.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**do we need Kafka?**

A team wants Kafka to send welcome emails after sign-up. Sign-ups peak at 20 per second.

A `jobs` table, with the job inserted in the same transaction as the new user and workers claiming rows with `FOR UPDATE SKIP LOCKED`, handles that easily, needs no new system, and never loses a job when the broker and database disagree, because there is no broker.

Kafka becomes worth it if they later need several teams consuming the same event stream, replay of history, or tens of thousands of events per second. That is a real requirement, so they write it down as the trigger for revisiting.

</div>

## When adding is right

Add the component when the numbers and requirements demand it:

- Reads exceed what the database can serve even after indexing and a bigger instance, and the data tolerates staleness: **add a cache**.
- Work is slow, bursty or must survive a downstream outage: **add a queue**.
- Search needs relevance ranking, facets and typo tolerance over large text: **add a search engine**.
- Writes or data size exceed the largest single machine: **partition the database**.

In each case, also plan who will operate it, how you will monitor it, and what happens when it fails.

## Common mistakes

- **Résumé-driven design**: choosing technology because it is interesting.
- **Adding a cache to hide a missing index.**
- **Designing for a scale ten thousand times the current load** "just in case".
- **Running two systems that do the same job** because the migration was never finished.
- **Benchmarking a distributed system only against itself**, never against a single well-written machine.

## In an interview

Interviewers often push you to scale. That is fine, but show your reasoning: estimate the load first, say what one machine or one database can handle, and add a component only when the numbers exceed it. "At 2,000 reads per second one Postgres primary with replicas is enough; if we reach 50,000, we'd add a Redis cache for product pages" shows judgement. Asking "What is the current peak load and what is the bottleneck?" is a strong question in a real design review.

<div class="take">
<div class="lab">Key takeaway</div>

default to the tools you already run well, because operating a component costs far more than building with it. Add new infrastructure only when measured load and a measured bottleneck show the current stack can't cope, and check that the new system beats one well-written machine.

</div>
