---
id: "08.03"
course: "system-design-101"
module: "08"
slug: "event-bus-for-product-events"
title: "Event bus for product events"
keyIdeas: ["An event bus carries domain events (\"OrderPlaced\", \"UserSignedUp\") so many services can react without the producer knowing about them.", "Writing to your database and then publishing to the broker in two steps isn't atomic: a crash between them loses the event, or publishes an event for a rolled-back write.", "**Transactional outbox**: write the event to an outbox table in the same DB transaction, then a relay publishes outbox rows to the broker. CDC (Debezium) can do the relay.", "Consumers must be idempotent since delivery is at-least-once.", "Too many fine-grained events make the flow hard to follow; document who produces and consumes each event."]
question: "Why can't you write to the DB and then publish to Kafka in two separate steps safely?"
answer: "The two writes are not atomic, because no transaction spans the database and the broker. If you publish first and the database transaction then fails, consumers act on an order that never existed; if you commit first and the process crashes or the broker times out before publishing, the event is lost and nothing retries it. Two-phase commit is rarely available and couples the systems. The fix is the transactional outbox: insert the event into an outbox table in the same local transaction as the business row, and have a relay (polling or change data capture) publish it afterwards. The relay may publish twice, so consumers deduplicate by event ID."
sources: [{"title": "Martin Fowler: What do you mean by \"Event-Driven\"?", "url": "https://martinfowler.com/articles/201701-event-driven.html", "note": "Event patterns and their costs."}, {"title": "microservices.io: Transactional outbox", "url": "https://microservices.io/patterns/data/transactional-outbox.html", "note": "Publishing events reliably with DB writes."}]
---

## What an event bus is for

As a product grows, many services want to react to the same things. When an order is placed, the email service sends a confirmation, the warehouse reserves stock, analytics counts revenue, search updates the customer's order history and fraud scores the transaction. If the order service called each of them directly, it would need to know all five, wait on all five, and fail when any one was down. Every new team would mean a code change in the order service.

An **event bus** (in practice usually a Kafka topic or an SNS topic with queues behind it, lesson 08.02) flips that around. The order service publishes one **event**, a record that something happened, such as `OrderPlaced {order_id: 9001, customer_id: 42, total: 820.00}`. Anyone who cares subscribes. The producer doesn't know or care who they are. Adding a sixth consumer needs no change to the producer.

That decoupling is the big win. But "event-driven" is used loosely, and the hard part is getting events onto the bus reliably. This lesson covers both.

## What "event-driven" can mean

Four patterns are often lumped together. They have very different costs.

### Event notification

A system announces that something changed and carries little data, often just an ID and a type: `CustomerAddressChanged {customer_id: 42}`. Receivers that need details call back to the source: `GET /customers/42`.

- **Good**: very low coupling; the event schema hardly ever changes; easy to set up.
- **Bad**: every receiver calls back, so the source takes a burst of reads after every event, and receivers fail if the source is down. The overall flow becomes implicit, spread across many handlers, and hard to debug: no single piece of code says "when an address changes, these four things happen".
- **Watch for** the "passive-aggressive command": an event published because the sender really expects one particular service to do something. If the order service publishes `OrderNeedsShipping` and breaks when shipping ignores it, that's a command dressed as an event. Make it an explicit command or a task on a queue.

### Event-carried state transfer

Events include the changed data itself: `CustomerAddressChanged {customer_id: 42, street: ..., city: ..., postcode: ...}`. Receivers keep their own local copy and never call the source.

- **Good**: receivers keep working when the source is down; no remote calls on the hot path; lower latency.
- **Bad**: data is duplicated across services and eventually consistent (each copy updates a little later); more state to manage; the event schema is now a public contract that's harder to change.

~~~dg-compare Event notification versus event-carried state transfer: does the receiver call back for the data?
[Notification]
Customer svc -(id 42 changed)-> Shipping svc -(GET /customers/42)-> Customer svc
+ Tiny, stable events
- Callback load on the source
- Shipping fails if customer svc is down
[*State transfer]
Customer svc -(id 42 + new address)-> Shipping svc -> Local copy
+ No callbacks, works when source is down
- Duplicated, eventually consistent data
- Schema becomes a contract
~~~

### Event sourcing

Every change is stored as an event, and the **event log is the source of truth**. Current state is derived by replaying the events: an account's balance is the sum of its `Deposited` and `Withdrawn` events. Think of version control, where the history of commits is the real record and the working copy is derived from it.

- **Good**: a complete audit trail; you can rebuild past states ("what did this account look like on 1 March?"); new read models can be built by replaying.
- **Bad**: replay gets hard when events change shape over the years (you need **upcasters** that convert old event versions), when replay would trigger external side effects such as sending emails again, and when the log grows huge (you add **snapshots**: a saved state at event N, so you only replay from N).

### CQRS

**Command Query Responsibility Segregation** means separate models for writing (commands) and reading (queries). Writes go to a normalised store; reads come from one or more denormalised views built from the writes, often by consuming events. It's often combined with event sourcing, but it isn't the same thing, and it's often misused: two models mean twice the work for every change and eventual consistency between them. Use it when reads and writes really have different shapes and scale, such as a write-heavy ledger with many read dashboards.

For a product event bus, you're almost always doing the first two: notification and state transfer. You publish facts about changes to your own database so other services can react.

## The dual-write problem

A product event bus usually starts with code like this:

```
def place_order(order):
    db.insert(order)          # step 1: commit to Postgres
    kafka.publish("OrderPlaced", order)   # step 2: publish to Kafka
```

This is a **dual write**: two separate systems updated in two separate steps, with no transaction spanning both. It looks fine in testing. In production it is unsafe in both orders:

- **Publish before commit**, and the database transaction might then fail (a constraint violation, a deadlock, a timeout). Consumers hear about an order that doesn't exist. The email service sends a confirmation for an order the customer will never see.
- **Publish after commit**, and the service might crash, be killed by a deploy, or lose its network between the two steps. The order exists, but nobody hears about it. The warehouse never reserves stock. Nothing retries, because nothing knows the event is missing.
- **Publish fails but you've committed.** Kafka is briefly unavailable or times out. Do you roll back the order? You can't, it's committed. Retry in memory? The process may die first.

~~~dg-seq The dual-write failure: the commit succeeds, the process dies, and the event is never published.
Service > Database : BEGIN, insert order 9001
Service > Database : COMMIT
Database -> Service : ok
Service > Service : killed by deploy
Service -> Broker : OrderPlaced never sent
~~~

How often does this happen? Rarely per request, but not rarely in total. At 1,000 orders a second, if 1 in 100,000 publishes is lost to a crash or timeout in that tiny window, that's 864 lost events a day. Every one is a customer whose order was never shipped.

**Two-phase commit** (2PC), where a coordinator gets both the database and the broker to prepare and then commit together, would fix it in theory. In practice Kafka doesn't take part in XA transactions with your database, many managed brokers don't support it at all, and 2PC couples the availability of the two systems tightly: if either is down, neither can commit. So the industry answer is different.

## The transactional outbox

Write the event into an **outbox table** in the **same local database transaction** as the business change. A single database can commit two rows atomically, so either both the order and its event exist, or neither does.

```
BEGIN;
INSERT INTO orders (id, customer_id, total) VALUES (9001, 42, 820.00);
INSERT INTO outbox (event_id, aggregate_id, type, payload, created_at)
  VALUES ('e_551', 9001, 'OrderPlaced', '{...}', now());
COMMIT;
```

A separate **message relay** then reads outbox rows and publishes them to the broker. There are two ways to build it.

- **Polling publisher**: every few hundred milliseconds, `SELECT * FROM outbox WHERE sent_at IS NULL ORDER BY id LIMIT 500`, publish each, then `UPDATE outbox SET sent_at = now()` (or delete them). Simple; adds a little latency and some database load. Run one relay per partition of the outbox, or use `SKIP LOCKED` (lesson 08.06) so several relays don't publish the same rows.
- **Transaction log tailing**: read the database's own replication log (the Postgres write-ahead log or the MySQL binlog) with a **change data capture** (CDC) tool such as Debezium, and turn each outbox insert into a broker message. No polling load, latency of tens of milliseconds, and ordering follows commit order exactly. More moving parts to run.

~~~dg-flow The outbox: one local transaction writes both rows; a relay publishes afterwards and consumers deduplicate.
[Write] Service -(one transaction)-> *Orders + outbox rows
[Relay] Outbox rows -(poll or CDC)-> Relay -(publish)-> Broker
[Consume] Broker -(at least once)-> Consumer -(check event_id)-> Processed table
~~~

### Duplicates are now guaranteed, so consumers must be idempotent

The relay can crash after publishing a row but before marking it sent. On restart it publishes the row again. The broker itself may redeliver after a consumer crash. So delivery is **at least once**, and consumers must be **idempotent**. The standard technique is an **inbox** or processed-events table: in the same transaction as its own side effect, the consumer inserts the event ID into a table with a unique constraint. If the insert fails because the ID already exists, it's a duplicate, and the consumer skips it.

~~~dg-state What happens to one outbox row. The only unsafe-looking step, published but not marked, is handled by consumer deduplication.
[*] -> Written
Written -> Published : relay sends to broker
Published -> Sent : relay marks sent_at
Published -> Written : relay crashed, row unmarked
Sent -> Purged : cleanup job after 7 days
~~~

### Ordering

Because outbox rows are inserted in transaction order and the relay reads them in that order, events for one order come out in the order they happened. To keep that order in Kafka, publish with the aggregate ID (`order_id`) as the message key so all events for one order go to one partition. Across different orders, there's no ordering promise, and you rarely need one.

### Housekeeping

Outbox tables grow fast. At 1,000 events a second, that's 86 million rows a day. Delete or partition-drop sent rows after a few days. Keep the payload reasonable: put the facts a consumer needs, not a dump of every related table.

<div class="ex">
<div class="lab">Worked example</div>

**order placed, with a relay crash**

In one transaction the order service inserts order 9001 and an outbox row `{event_id: e_551, type: OrderPlaced, order_id: 9001}`. It commits in 4 ms and returns to the customer. 200 ms later the polling relay reads e_551, publishes it to Kafka keyed by 9001, and crashes before the `UPDATE ... sent_at`. It restarts and publishes e_551 again.

The email service receives e_551 the first time, inserts `e_551` into `processed_events`, sends the email and commits. On the duplicate, the insert hits the unique constraint, the handler logs "duplicate, skipped" and acks. The customer gets one email. The warehouse service does the same with its own processed table, and reserves stock once.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**why the naive version loses events**

Same order, naive code. The service commits order 9001, then a Kubernetes rolling deploy sends SIGTERM and the pod exits 30 ms later, before `kafka.publish` runs. No error is logged because the process simply stopped. Three days later the customer asks where their parcel is. Support finds the order in the database, but the warehouse has no record. With the outbox, the event row would have been committed with the order and the relay would have sent it after restart.

</div>

## Alternatives you should know

- **Listen to yourself**: publish the event first, then have your own service consume it to write the database. The event log becomes the source of truth. This works, but reads of your own writes are now eventually consistent.
- **CDC on the business tables directly**, without an outbox table: Debezium streams every change to `orders`. Easy to add, but your table schema becomes the event schema, so a column rename breaks consumers. The outbox lets you publish deliberate, stable event shapes.
- **Database as the queue**: if consumers are in-house workers, enqueue a job row in the same transaction (lesson 08.06). Same idea, no broker.

## Event design tips

- Name events in the past tense, as facts: `OrderPlaced`, not `PlaceOrder`.
- Include an `event_id`, `type`, `version`, `occurred_at` and the aggregate ID in every event.
- Version schemas and only add optional fields. Use a schema registry (Avro, Protobuf, JSON Schema) so producers can't publish breaking changes.
- Publish what other services need to act without calling back, unless the data is sensitive.

## Common mistakes

- Publishing inside the transaction (before commit) because "it's in the same function".
- Thinking retries inside the process fix dual writes. A crash ends the process and its retries.
- Assuming exactly-once delivery from the relay or the broker. Kafka's exactly-once features cover Kafka-to-Kafka processing, not your database plus an email provider.
- Forgetting to clean up the outbox table.

## In an interview

Whenever your design says "write to the database and publish an event", say the words "transactional outbox" and "idempotent consumers" before the interviewer asks. Explain the two failure orders in one sentence each. That's one of the most common follow-up questions on event-driven designs.

<div class="take">
<div class="lab">Key takeaway</div>

never write to the database and the broker as two separate steps. Write the event to an outbox table in the same transaction as the business change, relay it to the broker by polling or CDC, and make consumers idempotent because the relay can repeat.

</div>
