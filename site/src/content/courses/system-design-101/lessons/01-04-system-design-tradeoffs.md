---
id: "01.04"
course: "system-design-101"
module: "01"
slug: "system-design-tradeoffs"
title: "System design tradeoffs"
keyIdeas: ["Every design choice buys something and costs something. Saying the cost out loud is what makes a design defensible.", "Classic pairs: latency vs throughput (batching raises throughput, adds latency), consistency vs availability, read speed vs write speed (indexes, denormalization), simplicity vs flexibility.", "Performance vs scalability: a system has a performance problem if it's slow for one user, and a scalability problem if it's fast for one user but slow under load.", "Stale reads are fine for like counts and product recommendations. They are not fine for bank balances, inventory at checkout, or permissions.", "Frame decisions as \"we chose X because requirement Y matters more than Z\"."]
question: "Name one case where you'd accept stale data and one where you wouldn't."
answer: "Accept stale data for a post's like count or a \"people you may know\" list: being a few seconds behind harms nobody and lets you serve it cheaply from a cache or replica. Don't accept it when checking a bank balance before a withdrawal or seat availability at the moment of booking, because acting on an old value causes overdrafts or double bookings; there you read from the primary or check inside a transaction."
sources: [{"title": "System Design Primer: tradeoffs", "url": "https://github.com/donnemartin/system-design-primer#performance-vs-scalability", "note": "Performance vs scalability, latency vs throughput, availability vs consistency."}, {"title": "Werner Vogels: Eventually Consistent, Revisited", "url": "https://www.allthingsdistributed.com/2008/12/eventually_consistent.html", "note": "The classic consistency tradeoff from Amazon's CTO."}]
---

## There is no free choice

Every architectural decision buys something and pays for it somewhere else. A cache buys read speed and pays with staleness and a second copy to keep in sync. A queue buys resilience and pays with delay and harder debugging. Replicas buy availability and pay with money and with copies that can disagree. There is no option that is best on every axis, which is why "it depends" is the honest answer to most design questions.

Interviewers and reviewers care less about which option you pick than about whether you can name what it costs and why that cost is acceptable here. A useful sentence pattern is: **"We chose X because requirement Y matters more here than Z, and we accept cost C."** For example: "We'll serve like counts from a cache refreshed every few seconds, because read latency at 200,000 requests per second matters more than exact counts, and we accept that a count can be a few seconds behind."

This lesson covers the trade-offs you will meet in almost every design: performance against scalability, latency against throughput, consistency against availability, read speed against write speed, and simplicity against flexibility. It finishes with the question you will answer over and over: when is stale data acceptable?

## Performance vs scalability

These two words get used as synonyms, but they describe different problems with different fixes.

- A **performance** problem: the system is slow even for one user. One request takes 2 seconds on an idle server.
- A **scalability** problem: the system is fast for one user but slows down as load grows. One request takes 50 ms at 10 requests per second and 2 seconds at 10,000.

A system is **scalable** if adding resources increases capacity roughly in proportion. Double the servers, double the requests per second you can serve at the same latency. If doubling the servers gives you only 10% more capacity, something is shared and is limiting you.

~~~dg-compare Two different problems with the same symptom. A performance problem is slow with one user; a scalability problem is fast with one user and slow under load.
[Performance problem]
1 user -(request)-> Server -> 2 s response
Cause: slow code or query
Fix: index, algorithm, profiling
- More servers do not help
[Scalability problem]
1 user -(request)-> Server -> 50 ms
10k users -(same request)-> Server -> 2 s
Cause: contention, shared state
Fix: remove the bottleneck, partition
- Faster code helps only a little
~~~

The fixes differ because the causes differ:

- **Performance** problems usually live in code, queries or algorithms: a missing index that turns a lookup into a full table scan, an O(n²) loop, a request that makes 40 database calls instead of one. You find them with a profiler or a query plan, on one request.
- **Scalability** problems usually live in something that is shared: a single database primary every write must go through, a lock that every request takes, a counter row that every transaction updates, a single-threaded component. You find them by load testing and watching which resource saturates first.

Telling them apart saves a lot of wasted effort. Adding ten more web servers to fix a query that scans 50 million rows does nothing, because each server runs the same slow query. Rewriting a query to be 20% faster does little if the real problem is that every request waits for the same row lock.

<div class="ex">
<div class="lab">Worked example</div>

**diagnosing a slow checkout**

At 3 a.m., with 5 users online, checkout takes 1.8 s. At noon, with 4,000 users, it also takes about 1.8 s. The latency does not depend on load, so this is a performance problem. The trace shows 1.6 s inside one query that filters orders by customer without an index. Adding the index brings checkout to 120 ms at any load.

A different service: at 3 a.m. a search takes 60 ms; at noon it takes 900 ms, and the database CPU sits at 98%. Latency rises with load, so this is a scalability problem. The fix is more read capacity (replicas or a cache), not a faster query.

</div>

## Latency vs throughput

**Latency** is how long one operation takes, measured from the caller's point of view: 40 ms to load a profile. **Throughput** is how many operations complete per unit of time: 12,000 profile loads per second. The goal is usually maximal throughput with acceptable latency.

They are related through a simple law. **Little's Law** says that the average number of requests in the system equals the arrival rate multiplied by the average time each request spends in it: `L = λ × W`. A service handling 1,000 requests per second with 50 ms latency has, on average, 1,000 × 0.05 = 50 requests in flight. If each in-flight request needs a thread, you need about 50 threads. If latency rises to 500 ms, you need 500. This one formula explains why slow dependencies exhaust thread pools (lesson 02.11 uses it again).

The two often trade against each other:

- **Batching** raises throughput by paying a fixed cost (a network round trip, a disk flush) once for many items. But the first item in the batch waits for the last. Lesson 02.10 covers this.
- **Queueing** smooths load so a worker pool can run at high utilisation, but every item now waits in line.
- **Running hot** (high utilisation) gives the most throughput per server, but queueing theory says latency rises steeply as utilisation approaches 100%. A server at 50% busy has short queues; at 95% busy, queues grow very long. That is why capacity plans leave headroom.

~~~dg-bars Average time a request waits in a single queue as server utilisation rises, relative to the service time. Latency stays low until the server is busy, then explodes.
50% busy | 1 | 1x service time
70% busy | 2.3 | 2.3x
80% busy | 4 | 4x
90% busy | 9 | 9x
*95% busy | 19 | 19x
99% busy | 99 | 99x
~~~

These numbers come from the simplest queueing model (one server, random arrivals), where the average wait is `utilisation / (1 − utilisation)` times the service time. Real systems differ in detail but show the same hockey-stick shape. The practical rule: plan to run shared resources at 50 to 70% of their capacity at peak, not 95%.

## Consistency vs availability

When data is copied across machines (replicas, caches, regions), the copies can disagree for a while. You must decide what readers see during that time. There are three common consistency patterns:

- **Weak consistency**: after a write, reads may or may not see it, with no promise. Memcached used as a best-effort cache, VoIP and live games work this way: if you miss an update, you just get the next one.
- **Eventual consistency**: if writes stop, all copies converge to the same value, usually within milliseconds to seconds. DNS and email work this way. It suits highly available systems because each copy can answer on its own.
- **Strong consistency**: once a write is acknowledged, every read sees it. Relational databases on a single primary and file systems on one machine work this way. You need it when correctness depends on it, such as money or unique usernames.

The trade-off becomes sharp during a **network partition**: a failure where some machines cannot talk to others, though each is still running. A replica cut off from the primary has two choices when a request arrives. It can answer with what it has (available, but possibly stale), or refuse until it can reach the primary again (consistent, but unavailable for that request).

~~~dg-seq A partition forces a choice. Replica B can no longer hear from the primary, so it either answers with what it has or refuses.
Client 1 > Primary : set price = 12
Primary > Replica B : replicate (lost, link down)
Client 2 > Replica B : get price
Replica B -> Client 2 : option A: 10 (stale but available)
Replica B -> Client 2 : option B: error (consistent, unavailable)
~~~

Amazon often chose availability. Its shopping cart is always writable: if two copies accept changes during a partition, the cart merges them afterwards, so a deleted item can occasionally reappear. That is a deliberate trade: a slightly wrong cart costs little, while a cart that refuses "add to basket" costs sales.

### Quorums: tuning the dial

Many replicated stores expose three numbers that let you choose a point between the extremes:

- **N**: how many replicas hold each item (often 3).
- **W**: how many replicas must confirm a write before it counts as successful.
- **R**: how many replicas a read consults.

If **W + R > N**, every read set overlaps every write set in at least one replica, so a read will see the latest acknowledged write. With N = 3, choosing W = 2 and R = 2 gives that overlap and tolerates one replica being down. Choosing W = 1 and R = 1 makes reads and writes fast and highly available, but a read can easily miss a recent write. Module 6 covers quorums and their edge cases in depth.

<div class="ex">
<div class="lab">Worked example</div>

**what W and R buy you**

N = 3 replicas, each answering in 2 ms normally, but one is slow at 80 ms at any given time.

W = 3, R = 1: every write waits for the slow replica, so writes take 80 ms. Reads take 2 ms and always see the latest write. If any replica is down, writes fail.

W = 2, R = 2: writes and reads each wait for the two fastest, about 2 ms. Reads still overlap writes, and one replica can be down. This is the common default.

W = 1, R = 1: everything is fast and survives two replicas down, but a read right after a write has about a two-in-three chance of hitting a replica that has not received it yet.

</div>

## Other classic trade-offs

- **Read speed vs write speed**: indexes, denormalisation, materialised views and caches all make reads faster by doing work at write time. Each index on a table is another structure to update on every insert. A table with eight indexes can make writes several times slower than the bare table.
- **Space vs time**: precomputing (a feed built ahead of time, a cached page) spends storage to save computation at read time. Computing on demand saves storage and costs latency.
- **Simplicity vs flexibility**: a monolith is simpler to build, test and debug. Microservices let many teams deploy independently, at the price of network calls and distributed data (lesson 01.09).
- **Cost vs everything**: more replicas, more regions and more headroom all improve availability and latency, and all cost money every month, even when idle (lesson 01.12).

~~~dg-matrix Common techniques and what they do to each property. Almost every gain on one axis is paid for on another.
Technique | Read latency | Write latency | Freshness | Cost
Cache in front of DB | yes, faster | same | no, can be stale | partial, extra servers
Extra index | yes, faster | no, slower | yes, exact | partial, more disk
Async replicas | yes, more capacity | yes, unaffected | no, lag | no, more servers
Sync replicas | partial | no, slower | yes, exact | no, more servers
Batching writes | same | no, items wait | partial | yes, fewer calls
~~~

## When is stale data fine?

This is the question you will answer per feature, not per system. One system often has both kinds of data side by side.

Many web applications only need **user-perceived consistency**: the inconsistency window just has to be shorter than the time before the user would notice. If a replica is 500 ms behind, and a user takes 3 seconds to click to the next page, nobody notices. The one case users always notice is their own writes: post a comment, refresh, and it's gone. So a common pattern is **read-your-writes**: route a user's reads to the primary for a few seconds after they write, or remember the version they wrote and wait for a replica to catch up to it.

Data that tolerates seconds (or minutes) of staleness:

- Like counts, view counts, follower counts.
- Recommendations, "people you may know", trending lists.
- Search indexes (a new product appearing in search 30 seconds late is fine).
- Analytics dashboards.

Data that usually does not:

- Account balances before a withdrawal or transfer.
- Inventory or seat availability at the moment of purchase.
- Permission changes: after you remove someone's access, a stale cache that still lets them in is a security bug.
- Uniqueness checks, like "is this username taken?".

~~~dg-timeline Replication lag in action: a write lands on the primary, and a reader on a lagging replica sees the old value until the change arrives.
[Primary] 0 ms | balance set to 0
[Replica] 0 ms | still shows 100
[Reader] 120 ms | *reads 100 from replica
[Replica] 300 ms | change applied, shows 0
[Reader] 400 ms | reads 0
~~~

The test to apply: **what is the worst thing that happens if a user acts on a value that is a few seconds old, and how much does it cost?** A like count that shows 1,203 instead of 1,205 costs nothing. A balance that shows \$100 when it is \$0 can let someone withdraw money twice. A ticket that looks available when it was sold a second ago produces a double booking and a refund. Even in those cases you can often keep the fast, stale read for display and do a strongly consistent check at the moment of commitment: show "3 seats left" from a cache, but decrement the count inside a database transaction when the user pays.

<div class="ex">
<div class="lab">Worked example</div>

**one product, two consistency choices**

An events site shows "1,240 people interested" on each concert page. It reads this from a cache refreshed every 30 seconds. At 50,000 page views per second, the cache serves almost all of them from RAM, and a 30-second-old number harms nobody.

The same page sells tickets. The "Buy" button calls a service that runs `UPDATE seats SET held_by = ? WHERE seat_id = ? AND held_by IS NULL` in the primary database. If two users try the same seat at once, exactly one update affects a row; the other user is told the seat is gone. Display is stale and cheap; commitment is consistent and exact.

</div>

## Common mistakes

- **Not naming the cost.** "We'll add a cache" without saying what goes stale and for how long.
- **One consistency model for everything.** Making like counts strongly consistent wastes money; making balances eventually consistent loses it.
- **Confusing performance and scalability.** Throwing servers at a slow query, or optimising code when a shared lock is the bottleneck.
- **Designing for 95% utilisation.** Queues explode near full load; leave headroom.
- **Reading "eventual" as "a few milliseconds".** Under load or during failures, lag can grow to minutes. Know what happens then.

## In an interview

State the trade-off out loud every time you make a choice, and tie it to a requirement from the start of the session. Good questions to ask: "How stale can this be before users or the business notice?", "Is it worse to show an error or a slightly old value here?", "What does a double booking or a double charge cost us?" When you separate a feature's display path (fast, stale) from its commit path (exact), you show that you understand the trade-off rather than picking one side for the whole system.

<div class="take">
<div class="lab">Key takeaway</div>

every choice buys one property with another, so always name the cost. Separate performance from scalability and latency from throughput, and decide per feature how stale data may be, often with a fast stale read for display and a strict check at the moment of commitment.

</div>
