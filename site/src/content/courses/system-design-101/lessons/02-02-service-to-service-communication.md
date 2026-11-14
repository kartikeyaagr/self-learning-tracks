---
id: "02.02"
course: "system-design-101"
module: "02"
slug: "service-to-service-communication"
title: "Service-to-service communication"
keyIdeas: ["Services talk **synchronously** (HTTP/gRPC request-response) or **asynchronously** (events or messages via a broker).", "Sync is simple and gives an immediate answer, but couples availability: if B is down, A's request fails too. Chains of sync calls multiply latency and failure.", "Async decouples: A publishes and moves on. B processes when it can. Better resilience and load smoothing, but eventual consistency and harder debugging.", "Use sync when the caller needs the answer now (check a price, authorize a payment). Use async for side effects (send email, update search index, analytics).", "Always set timeouts and retries with backoff on sync calls."]
question: "When should service A call B synchronously, and when should it publish an event instead?"
answer: "Call B synchronously when A cannot continue without B's answer, such as authorising a payment before confirming an order or reading a price to show the user, and protect the call with a timeout, a circuit breaker and a fallback. Publish an event when A is announcing something that others react to, such as OrderPlaced triggering email, analytics and the warehouse, because then B's outages and slowness don't affect A, the broker buffers work while B catches up, and new consumers can subscribe without changing A."
sources: [{"title": "microservices.io: Remote Procedure Invocation", "url": "https://microservices.io/patterns/communication-style/rpi.html", "note": "Sync request/response between services."}, {"title": "microservices.io: Messaging", "url": "https://microservices.io/patterns/communication-style/messaging.html", "note": "Async communication via a broker."}]
---

## Two ways for services to talk

Once a system is split into services, each service needs data or work from others. The orders service needs a price from catalogue, a charge from payments, and needs to tell email, analytics and the warehouse that an order exists. There are two families of interaction, and choosing between them per arrow is one of the most important decisions in a service-based design.

**Remote procedure invocation (RPI)**, also called synchronous request/response: service A sends a request to B and waits for a reply. REST over HTTP, gRPC and Thrift are all RPI. It is simple and familiar (it looks like a function call), and there is no extra infrastructure in the middle.

**Messaging**, also called asynchronous communication: A sends a message to a **broker** (Kafka, RabbitMQ, Amazon SQS, Google Pub/Sub) and moves on. B, and possibly C and D, consume it when they can. A doesn't wait and often doesn't know who the consumers are.

~~~dg-compare The same interaction done synchronously and through a broker. Synchronous calls couple the two services in time; a broker decouples them.
[Synchronous call]
Orders -(request, waits)-> Payments -(reply)-> Orders
+ Simple, immediate answer
+ No broker to run
- Orders is down when Payments is down
- Orders' latency includes Payments'
[Messaging]
Orders -(publish, returns)-> Broker -(later)-> Email, Analytics, Warehouse
+ Consumers can be slow or down
+ Many consumers, added without changing Orders
- Broker must be highly available
- Harder to trace and debug
~~~

## What synchronous calls cost you

RPI's main weakness is **availability and latency coupling**. If A calls B synchronously on every request:

- A is only up when B is up. If B returns errors, so does A.
- A's latency includes B's latency, plus the network round trip.
- If B is slow, A's threads or connections pile up waiting for it, and A runs out of capacity too.

Chain five services like this and availability multiplies down (five 99.9% services in series give about 99.5%) while latencies add up. Worse, failures spread: one slow database at the bottom of the chain can exhaust thread pools all the way up. That chain reaction is called a **cascading failure**.

Other costs:

- **It only supports request/reply.** Notifications and fan-out to many consumers need something else.
- **The caller must find the callee.** Instances come and go, so you need **service discovery**: a registry (Consul, etcd, Kubernetes DNS) that maps a service name to the current healthy addresses, or a load balancer with a stable name in front.
- **You must defend against slow dependencies.** Every call needs a timeout (for example 800 ms), and repeated failures should trip a **circuit breaker**.

### The circuit breaker

A circuit breaker wraps calls to a dependency and tracks failures. When failures cross a threshold, it "opens" and fails calls immediately, without trying, for a cool-down period. This protects the caller's threads and gives the struggling dependency room to recover. After the cool-down, it lets a few trial calls through; if they succeed, it closes again.

~~~dg-state A circuit breaker. When a dependency keeps failing, the caller stops trying for a while instead of piling more load on it.
[*] -> Closed
Closed -> Closed : call succeeds
Closed -> *Open : failures over threshold (e.g. 50% of last 20)
Open -> Open : calls fail fast, no network call
Open -> Half-open : cool-down timer ends (e.g. 30 s)
Half-open -> Closed : trial calls succeed
Half-open -> Open : a trial call fails
~~~

When the breaker is open, the caller needs a **fallback**: return a cached value, a default ("recommendations unavailable"), a degraded response, or a clear error. Designing those fallbacks is part of designing the call.

<div class="ex">
<div class="lab">Worked example</div>

**a cascading failure, and how to stop it**

The product page service has 200 worker threads and handles 1,000 requests per second, each calling the reviews service, which normally answers in 20 ms. By Little's Law, about 1,000 × 0.02 = 20 threads are busy waiting on reviews at any moment.

The reviews database slows down and reviews now take 5 s. Busy threads needed: 1,000 × 5 = 5,000. The pool of 200 is exhausted in a fifth of a second, and product pages fail entirely, even for requests that would never have needed reviews.

With a 300 ms timeout, at most 1,000 × 0.3 = 300 threads could be tied up, still too many. Add a circuit breaker that opens after half the recent calls time out: calls now fail in microseconds, pages render with "reviews unavailable", and the 200 threads are free for everything else.

</div>

## What messaging buys you, and costs you

Messaging decouples services **in time**. The broker stores messages durably, so the consumer can be down, slow or mid-deploy and the producer still succeeds. One message can reach many consumers, and new consumers can be added without changing the producer at all. The broker also absorbs bursts: if 10,000 orders arrive in a minute, the warehouse service works through them at its own pace.

Messaging supports several interaction styles:

- **Notification** (one way, fire and forget): "send this email".
- **Publish/subscribe**: "OrderPlaced" goes to every subscriber.
- **Request/async response**: A sends a request message with a **correlation ID** and a reply-to queue; B processes it and sends a reply with the same correlation ID; A matches them up later.
- **Publish/async responses**: A publishes a request and collects replies from several services for a while.

~~~dg-seq Request with an asynchronous reply. The correlation ID lets the requester match the reply to its request, whenever it arrives.
Orders > Broker : CheckCredit {corr_id: 77, reply_to: orders.replies}
Broker > Credit : deliver
Credit > Credit : evaluate customer
Credit > Broker : CreditChecked {corr_id: 77, ok: true}
Broker > Orders : deliver reply
Orders > Orders : match 77, confirm order
~~~

The costs:

- **The broker is another critical system** that must be highly available, monitored and capacity-planned.
- **Request/response becomes awkward**: correlation IDs, reply queues, and timeouts for replies that never come.
- **The flow of a business process is harder to see.** No single piece of code says "after an order, we charge, then reserve, then ship"; it emerges from who subscribes to what. Distributed tracing and good event documentation become essential.
- **Duplicates and reordering.** Most brokers deliver **at least once**: if a consumer crashes after processing a message but before acknowledging it, the message is delivered again. Consumers must be **idempotent** (processing the same message twice has the same effect as once), usually by recording processed message IDs. Ordering is typically guaranteed only within a partition or queue, not globally.
- **Eventual consistency.** After an order is placed, the analytics dashboard may not show it for a few seconds.

### Publishing reliably: the outbox

A classic bug: the orders service commits the order to its database, then publishes OrderPlaced to the broker, and crashes between the two. The order exists but nobody hears about it. Or it publishes first and the database commit fails, announcing an order that doesn't exist. You can't wrap a database and a broker in one transaction cheaply.

The **transactional outbox** fixes this. In the same database transaction as the business change, the service inserts the event into an `outbox` table. A separate relay process reads new outbox rows and publishes them to the broker, marking them sent. If anything crashes, the relay simply retries, so events may be published twice (which idempotent consumers handle) but are never lost and never invented.

## Choreography vs orchestration

When a business process spans several services, you can coordinate it two ways:

- **Choreography**: each service reacts to events and emits its own. Orders emits OrderPlaced; payments hears it, charges, emits PaymentSucceeded; the warehouse hears that and ships. No central coordinator, very loosely coupled, but the overall flow is implicit and hard to change.
- **Orchestration**: one coordinator (an orchestrator service or a workflow engine such as Temporal or AWS Step Functions) tells each service what to do, in order, and handles failures and compensations. The flow is explicit and easy to see, at the cost of a central component that knows about everyone.

Short, simple flows suit choreography. Long flows with many steps and compensation logic (like the checkout saga in lesson 01.09) usually suit orchestration.

## How to choose

Ask one question per arrow: **does the caller need the answer to continue?**

~~~dg-tree Choosing per interaction. The deciding question is whether the caller needs the answer before it can respond.
Does the caller need the answer to continue?
  Yes: synchronous call
    *Keep the chain short, add timeouts and a breaker
    e.g. check password, read price, authorise payment
  No: announcing a fact
    Publish an event
    e.g. OrderPlaced to email, search, analytics
  No: long or bursty work
    Put a job on a queue
    e.g. transcode video, build report
~~~

- **Need the answer now** (check a password, read a price to show the user, authorise a payment before confirming an order): use a synchronous call, and keep the chain short, ideally one or two hops.
- **Telling others something happened** (order placed, user signed up, profile updated): publish an event. The user should not wait for the welcome email to send, or for the search index to update.
- **Long or bursty work** (video transcoding, report generation, bulk imports): put a job on a queue and let workers pull at their own pace. Return "accepted" (HTTP 202) with a job ID the client can poll or be notified about.

A common shape is **synchronous at the edge and for reads on the critical path, asynchronous for side effects**. In interviews, saying out loud which arrows are synchronous and which are asynchronous is a strong signal.

<div class="ex">
<div class="lab">Worked example</div>

**checkout**

Sync: the order service calls payments and waits (with a 2 s timeout and an idempotency key), because the user must see "paid" or "declined" before leaving the page.

Async: in the same transaction that saves the order, it writes OrderPlaced to its outbox. The relay publishes it. Email, analytics, loyalty points and the warehouse each consume it independently.

If the email service is down for ten minutes, checkout keeps working. Its queue holds about 10 × 60 × 50 = 30,000 messages (at 50 orders a second), and when it comes back it works through them in a couple of minutes. Customers get their confirmations a little late; nobody loses an order.

</div>

~~~dg-timeline An email outage with asynchronous delivery. Checkout never notices; the queue absorbs the gap and drains afterwards.
[Checkout] 10:00 | orders succeed as normal
[Email] 10:00 | *email service crashes
[Queue] 10:05 | 15,000 messages waiting
[Email] 10:10 | service restarts
[Queue] 10:13 | backlog drained, emails ~10 min late
~~~

## Common mistakes

- **Long synchronous chains**: A calls B calls C calls D, each adding latency and failure.
- **No timeouts**, so one slow dependency exhausts every caller's threads.
- **Using messaging for things that need an immediate answer**, then building awkward request/reply over queues.
- **Publishing events outside the database transaction**, losing or inventing events on crashes.
- **Non-idempotent consumers** that double-count when a message is redelivered.

## In an interview

Label every arrow sync or async. For sync arrows, mention timeouts, retries with backoff for idempotent calls, and a circuit breaker with a fallback. For async arrows, name the broker, say delivery is at least once so consumers are idempotent, and mention the outbox for reliable publishing.

<div class="take">
<div class="lab">Key takeaway</div>

call synchronously only when you need the answer to proceed, and guard each call with a timeout, a circuit breaker and a fallback. Publish an event when you are announcing a fact, publish it reliably via an outbox, and make consumers idempotent. Every sync hop couples availability and latency.

</div>
