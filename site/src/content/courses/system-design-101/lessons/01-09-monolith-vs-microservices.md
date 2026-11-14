---
id: "01.09"
course: "system-design-101"
module: "01"
slug: "monolith-vs-microservices"
title: "Monolith vs microservices"
keyIdeas: ["**Monolith**: one deployable unit. Easy to develop, test, debug and deploy early on. In-process calls are fast and transactions are simple.", "**Microservices**: many small services, each owning its data and deployed independently. Lets large teams move independently and scale parts separately.", "Microservices add real costs: network calls that fail and add latency, distributed transactions, service discovery, versioned APIs, more infrastructure and observability work.", "\"Monolith first\": start with a well-modularized monolith, split out a service only when there's a clear reason (team ownership, very different scaling needs, isolation).", "Prime Video's monitoring service went from microservices back to a monolith and cut cost about 90%, because data was passed between services far too often."]
question: "List 2 costs microservices add that a monolith doesn't have."
answer: "Any two of: every cross-service call adds network latency and can fail or time out, so you need timeouts, retries and circuit breakers; there are no ACID transactions across services, so you need sagas, compensating actions and eventual consistency; the operational overhead of deploying, monitoring, logging and tracing many services; and harder refactoring, since moving logic between services needs versioned APIs and data migrations."
sources: [{"title": "Lewis and Fowler: Microservices", "url": "https://martinfowler.com/articles/microservices.html", "note": "The original definition."}, {"title": "Martin Fowler: Monolith First", "url": "https://martinfowler.com/bliki/MonolithFirst.html", "note": "Why to start with a monolith."}, {"title": "Neo Kim: Prime Video microservices failure", "url": "https://newsletter.systemdesign.one/p/prime-video-microservices", "note": "Prime Video moved a service back to a monolith and cut cost about 90%."}, {"title": "microservices.io: Monolithic Architecture", "url": "https://microservices.io/patterns/monolithic.html", "note": "Reference pattern."}]
---

## Two ways to structure an application

A **monolith** is one deployable application. It handles requests, runs all the business logic and talks to the database, and it ships as a single unit: one build, one deploy, one process (usually run as many identical copies behind a load balancer). Inside, it may be very well organised into modules, but those modules call each other with ordinary function calls and share one database.

**Microservices** build the application as a suite of small services. Each runs in its own process, owns its own data, communicates with the others over the network (HTTP, gRPC or messaging), and is deployed independently. An e-commerce site might have separate services for catalogue, basket, orders, payments, shipping and notifications, each owned by a different team.

Neither is "modern" or "legacy". They are two points on a trade-off between simplicity and independence, and the right choice depends mostly on team size, domain understanding and how differently the parts need to scale.

~~~dg-compare The same checkout as a monolith and as microservices. Function calls become network hops, and one database becomes several.
[Monolith]
Request -> Orders module -(function call)-> Payments module -> One database
+ Calls take nanoseconds and can't time out
+ One transaction covers the whole order
+ One thing to deploy and debug
- Every team ships through one pipeline
- The whole app scales as one unit
[Microservices]
Request -> Orders service -(network call)-> Payments service -> Payments DB
+ Teams deploy independently
+ Each part scales and fails separately
- Every call can be slow or fail
- No transaction across services
- Many things to deploy, monitor and trace
~~~

## What makes something "microservices"

Splitting code into several processes is not enough. The style is usually described by these characteristics:

- **Components as services**: parts can be deployed and replaced independently, behind explicit remote interfaces.
- **Organised around business capabilities**: one team owns "payments" end to end (API, logic, data, on-call), rather than teams split by layer ("the database team", "the UI team").
- **Products, not projects**: the team that builds a service keeps running it. "You build it, you run it."
- **Smart endpoints, dumb pipes**: business logic lives in the services; the network or message broker just carries messages. This is a reaction against the Enterprise Service Bus, where routing and transformation logic piled up in the middleware.
- **Decentralised governance and data**: teams choose their own tools within limits, and **each service owns its database**. Other services may not read its tables directly; they go through its API or its events.
- **Infrastructure automation**: with dozens of services, automated builds, tests and deployments (CI/CD) are mandatory, not nice to have.
- **Design for failure**: any remote call can fail or be slow, so services use timeouts, retries with backoff, circuit breakers and good monitoring (Module 2).
- **Evolutionary design**: if two services always change together, they probably belong together.

### Conway's law

A system's structure tends to mirror the communication structure of the organisation that builds it. Three teams building a compiler tend to produce a three-pass compiler. Microservices use this deliberately: give each team a service, and the boundaries in the code follow the boundaries between teams, so teams can work without constant coordination. The flip side is that if your service boundaries do not match your teams, you get endless cross-team meetings about every change.

## What microservices cost

The price is real, and much of it is invisible on an architecture diagram:

- **Remote calls are slow and can fail.** An in-process function call takes nanoseconds. A call to another service in the same data centre takes 0.5 to a few milliseconds, includes serialisation and deserialisation, and can time out, fail, or succeed without the caller learning about it.
- **No simple transactions across services.** In a monolith, "create the order and reserve the stock" is one database transaction. Across services there is no shared database, so you need **sagas**: a sequence of local transactions with compensating actions to undo earlier steps if a later one fails. You live with eventual consistency in between.
- **Availability multiplies down.** A request that synchronously calls five services, each 99.9% available, succeeds only about 99.5% of the time (lesson 01.05).
- **Refactoring across boundaries is hard.** Moving a function between modules is an IDE operation. Moving it between services means a new API, a versioned rollout and a data migration.
- **Operations multiply.** Every service needs deployment, monitoring, logging, alerts, dashboards, dependency upgrades and security patches. Debugging needs **distributed tracing** to follow one request across ten services.
- **Poor boundaries just move complexity** into the connections between services. A "distributed monolith", where every change still touches five services that must be deployed together, has the costs of both styles and the benefits of neither.

~~~dg-bars Rough cost of one call, in microseconds. Turning a function call into a network call makes it thousands of times slower.
In-process function call | 0.01 | ~10 ns
Call to a local cache | 1 | ~1 µs
Service call, same zone | 1000 | ~1 ms
Service call, another zone | 2000 | ~2 ms
*Service call, another region | 80000 | ~80 ms
~~~

~~~dg-seq A checkout saga across services. There is no shared transaction, so when shipping fails, earlier steps are undone by compensating actions.
Orders > Payments : charge order 42
Payments -> Orders : charged
Orders > Inventory : reserve items
Inventory -> Orders : reserved
Orders > Shipping : book courier
Shipping -> Orders : failed, no slots
Orders > Inventory : release items (compensate)
Orders > Payments : refund order 42 (compensate)
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the latency bill for one page**

A product page in a monolith makes 6 module calls (price, stock, reviews, recommendations, shipping estimate, user's basket) in about 2 ms of function calls plus 3 database queries of 1 ms each: about 5 ms of work.

Split into microservices, the same page makes 6 service calls. Run one after another at about 2 ms each, including serialisation, that is 12 ms, plus each service's own database query. With 1% of calls hitting a 100 ms slow path, the chance that a page hits at least one slow call is 1 − 0.99⁶ ≈ 6%.

Running the calls in parallel cuts the typical time to the slowest single call, but adds code for timeouts, partial failures and fallbacks ("show the page without reviews"). The work is all doable; it just isn't free.

</div>

## Monolith first

Most microservice success stories started as a monolith that was later split. Many systems built as microservices from scratch got into trouble. Two reasons:

- **Early on, you don't know if anyone wants the product.** Speed and fast feedback matter more than independent deploys. The operational cost of microservices (sometimes called the "microservice premium") is paid every day, while its benefits only appear with many teams and a stable domain.
- **Good service boundaries are hard to draw before you understand the domain.** You learn where the seams are by building the product. Moving functionality between services is much harder than moving it inside one codebase, so boundaries drawn too early tend to be wrong and expensive to fix.

Two well-trodden paths:

1. **Build a well-modularised monolith**, with clear internal module boundaries (each module has its own public interface and, ideally, its own tables that other modules don't touch). Later, peel off services at the edges when there is a clear reason. Shopify runs one of the largest Ruby on Rails monoliths in the world this way, enforcing boundaries between components inside a single codebase.
2. **Build a deliberately throwaway first version** to learn the domain, then rebuild with better boundaries.

When you do split, the **strangler fig** approach is the usual one: put a routing layer in front of the monolith, build the new service, move one route at a time to it, and delete the old code once the route is fully moved. The monolith shrinks gradually and you can stop at any point.

~~~dg-tree A modular monolith for an online shop. Each module has a public interface and owns its tables, so any one can later be extracted as a service.
Shop monolith
  Catalogue module | products, prices
  Basket module | baskets
  Orders module | orders, order_lines
    Calls Payments interface only
  Payments module | payments, refunds
    *First to extract: PCI rules, separate team
  Notifications module | email log
~~~

## Cases in reverse

### Prime Video

Prime Video's stream-quality monitoring tool was built with serverless microservices. A conversion service split video streams into frames and wrote them to S3; a defect detector read them back to look for problems; AWS Step Functions orchestrated each step. At scale, two costs exploded. Step Functions charges per state transition, and the design made several transitions for every second of every stream. The detector also made huge numbers of S3 requests to fetch frames. The team merged conversion and detection into a single process that passes frames in memory, ran it on containers (ECS), and cut the tool's infrastructure cost by over 90%, while also letting it scale further. Only this tool became a monolith; the rest of Prime Video stayed service-based.

### Segment

Segment had split its event-delivery system into more than a hundred small services, one per destination (each analytics or marketing tool it forwarded data to). Each had its own queue, its own deployment and its own copy of shared libraries, which drifted out of sync. A small team spent most of its time keeping them all running. Merging them into one service with one shared codebase made the system easier to change and operate.

The common lesson: choose the structure the workload and the team need, estimate the costs (including per-request infrastructure charges) before committing, and don't treat the decision as permanent.

<div class="ex">
<div class="lab">Worked example</div>

**deciding whether to extract a service**

A 40-engineer company runs a monolith. The payments module changes weekly, needs PCI compliance audits, and is owned by one 6-person team that is blocked waiting for the shared release train twice a week.

Reasons to split: a distinct team, a distinct compliance boundary, different deployment cadence. Reasons against: payments are called synchronously from checkout, so the split adds a network hop and a saga for refunds.

Decision: extract payments behind a gRPC interface, keep the checkout call synchronous with a 2 s timeout and an idempotency key (lesson 02.09), and publish PaymentSucceeded events for everything else. The other modules stay in the monolith.

</div>

## When to split

Good reasons:

- **Separate teams need to deploy independently**, and coordinating releases is the bottleneck.
- **One part needs very different scaling or hardware**: a video transcoder needs GPUs, the rest of the app doesn't.
- **Fault isolation**: a memory leak in report generation should not take down checkout.
- **A security or compliance boundary**: payment card data in a small, audited service.

Bad reasons:

- It's fashionable, or "that's how big companies do it".
- The codebase is messy. A messy monolith split up becomes messy microservices, now with network calls.
- To use a different language for fun.

## Common mistakes

- **Starting with microservices** for a new product with a small team.
- **Sharing a database between services**, which keeps all the coupling and adds the network.
- **Splitting by technical layer** ("data service", "logic service") instead of by business capability.
- **Long synchronous call chains**, which multiply latency and failure.
- **No tracing or central logging**, so nobody can follow a failing request.

## In an interview

Most interview designs show several services, because the interviewer wants to see components. Draw them, but be ready to say which could start as modules in one deployable, why each boundary exists, and which calls are synchronous. If asked "monolith or microservices?", answer with the trade-off: team size, deployment independence, scaling needs, and the costs you would accept.

<div class="take">
<div class="lab">Key takeaway</div>

start with a modular monolith. Split out a service when a team, scaling, isolation or compliance need clearly justifies the network calls, distributed data and operational overhead, and split along business capabilities, not technical layers.

</div>
