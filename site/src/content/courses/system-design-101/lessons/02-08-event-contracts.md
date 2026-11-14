---
id: "02.08"
course: "system-design-101"
module: "02"
slug: "event-contracts"
title: "Event contracts"
keyIdeas: ["Events are an API too. Consumers depend on the event's name, fields and meaning.", "Include standard metadata: event ID (for dedup), type, source, timestamp, schema version. CloudEvents defines a common envelope.", "Use a schema registry and compatibility rules. Backward compatible changes (add optional fields with defaults) let old and new versions coexist.", "Choose the style: thin notifications (\"order 42 changed\", consumer fetches details) or fat events carrying full state (no callback needed, but larger and leak more data).", "Consumers must tolerate duplicates and out-of-order delivery. Design handlers to be idempotent."]
question: "You add a field to an event. What must be true so old consumers keep working?"
answer: "The new field must be optional (with a default, typically null), and old consumers must ignore fields they don't recognise instead of rejecting the event. Then events in the new format still parse for them (forward compatibility), while new consumers can still read old events in the log that lack the field (backward compatibility). No existing field may change its name, type or meaning, and a schema registry enforcing the compatibility mode should check the new schema before it ships."
sources: [{"title": "CloudEvents spec", "url": "https://cloudevents.io/", "note": "A standard envelope for event metadata."}, {"title": "Martin Fowler: What do you mean by \"Event-Driven\"?", "url": "https://martinfowler.com/articles/201701-event-driven.html", "note": "Event notification vs event-carried state transfer vs event sourcing."}, {"title": "Confluent: Schema evolution and compatibility", "url": "https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html", "note": "Backward and forward compatible event schemas."}]
---

## Events are a public API

When a service publishes events, other teams build on them. The billing team consumes OrderPlaced to raise invoices; the data team loads it into the warehouse; the email team sends confirmations; a fraud model scores it. Some of those consumers you know about. Others were added last month by a team you have never met.

The event's shape is therefore a **contract**, just like a REST endpoint (lesson 02.01), with two extra difficulties:

- **Producer and consumers deploy at different times.** There is no moment when everyone upgrades together. For days or weeks, old producers and new consumers (and new producers and old consumers) run side by side.
- **Old events stick around.** In a log such as Kafka, events may be retained for days, months or forever, and a new consumer may replay all of them from the beginning. Your newest code must still read the oldest events.

Getting the contract right means agreeing on three things: what the event **means**, what it **carries**, and how it is allowed to **change**.

~~~dg-tree One producer, many consumers, some of which the producer never hears about. Any change to the event's shape reaches all of them.
Orders service | publishes OrderPlaced v1
  Event log | keeps events for days or forever
    Billing | raises invoices
    Email | sends confirmations
    Data warehouse | may replay from the start
    *Teams you haven't met | added last month
~~~

## What kind of event is it?

"Event-driven" covers four different patterns, and confusing them causes muddled designs.

**Event notification**: "Order 123 was placed." The event is small: an ID, a type, a timestamp. A consumer that needs details calls back to the producer's API. This is very decoupled (the producer knows nothing about consumers), but every consumer that needs data adds load on the producer, and the overall flow is hard to see because no one place describes it.

**Event-carried state transfer**: the event includes the data consumers need: customer name, address, line items, totals. Consumers keep their own copy of what they need and never call back. That cuts load on the producer and removes the runtime dependency (the producer can be down and consumers still have their data). The costs: data is duplicated across services, events are larger, consumers are eventually consistent, and the event's schema is now a much bigger contract.

**Event sourcing**: every change to an entity is stored as an event (OrderCreated, ItemAdded, AddressChanged, OrderShipped), and the **event log is the source of truth**. Current state is rebuilt by replaying the events. You get a full audit trail and can rebuild or reinterpret history (add a new report and compute it for all past orders). But schema changes are hard, because you must still understand events written years ago, and replays of large histories take time, so you add snapshots.

**CQRS** (command query responsibility segregation): separate models for writing (commands) and reading (queries), often with events flowing from the write side to build read-optimised views. Useful for complex domains or very different read and write loads. It adds a lot of complexity and is often used where a single model would do.

~~~dg-compare Two styles of event for the same order. A notification is small and makes consumers call back; state transfer carries what they need.
[Event notification]
OrderPlaced {order_id: 123} -> Consumer -(GET /orders/123)-> Orders API
+ Tiny contract, easy to evolve
+ Consumers always get fresh data
- Every consumer loads the producer
- Producer down: consumers stuck
[*Event-carried state transfer]
OrderPlaced {order_id, customer, items, total} -> Consumer stores its copy
+ No call-backs, producer can be down
+ Consumers fully independent
- Bigger contract, harder to change
- Copies are eventually consistent
~~~

Be explicit about which you are publishing. A common failure is a "notification" event that grows a few fields at a time until it is a half-complete state transfer that nobody designed.

### Name events as facts

Events describe things that **have happened**, in the past tense: OrderPlaced, PaymentFailed, AddressChanged. A message that asks for something to happen (SendEmail, ChargeCard) is a **command**, with a single intended recipient and a different contract. Mixing the two makes it unclear who is responsible for what.

## A standard envelope: CloudEvents

If every producer invents its own event format, every consumer writes custom parsing, and generic tooling (routing, tracing, dead-letter handling) can't work across events. **CloudEvents**, a CNCF (Cloud Native Computing Foundation) project that graduated in January 2024, defines a common **envelope** of metadata around your payload:

- **Required**: `id` (unique per event from a source), `source` (which system produced it, as a URI), `specversion` (the CloudEvents version), and `type` (such as `com.shop.order.placed`).
- **Optional**: `time`, `subject` (the entity it's about, such as the order ID), `datacontenttype` (such as `application/json`), and `dataschema` (a link to the schema of the payload).

It defines bindings for HTTP, Kafka, AMQP, MQTT, WebSockets and others, and is supported by services such as Amazon EventBridge and Azure Event Grid.

~~~dg-layers A CloudEvents envelope: standard metadata around your payload. Routers and tools read the envelope; only consumers read the data.
id, source | unique event ID and the producer that emitted it
type, subject | what happened, and to which entity
specversion, time | envelope version and when it happened
dataschema, datacontenttype | which schema version, which format
*data | your payload, e.g. the order
~~~

Even if you don't use CloudEvents itself, every event should carry the same basics:

- A **unique ID**, so consumers can deduplicate. Most brokers deliver at least once, and consumers must handle repeats (lesson 02.02).
- A **type** and a **source**.
- A **timestamp** of when the thing happened (not when it was published).
- A **schema version**, so consumers know how to parse it.
- Often a **correlation or trace ID**, to follow one business transaction across services.

## Evolving the schema safely

The data inside the event will change. The question is which changes are safe, and in which order to deploy them.

A **schema registry** (Confluent Schema Registry is common with Kafka; AWS Glue and others exist) stores every version of each event's schema (in Avro, Protobuf or JSON Schema). Producers register the schema they write with, and each message carries a small schema ID. The registry rejects a new schema version that would break the agreed **compatibility mode**:

~~~dg-matrix Schema compatibility modes. Each mode allows different changes and implies a different deploy order.
Mode | Add optional field | Remove optional field | Add required field | Rename or retype | Deploy first
Backward (common default) | yes | yes | no | no | consumers
Forward | yes | yes | yes | no | producers
Full | yes | yes | no | no | either
None | yes | yes | yes | yes | coordinate by hand
~~~

- **Backward compatible** (Confluent's default): consumers using the **new** schema can read data written with the **old** one. Allowed: delete fields, add optional fields with defaults. Upgrade **consumers first**, then producers.
- **Forward compatible**: consumers using the **old** schema can read data written with the **new** one. Allowed: add fields, delete optional fields. Upgrade **producers first**.
- **Full**: both directions at once. Only add or remove optional fields with defaults. Upgrade in **any order**.
- **Transitive** variants (BACKWARD_TRANSITIVE and so on) check a new version against **all** earlier versions, not just the previous one. This matters when consumers may replay old data from a log: version 7 must still read events written with version 1.

The practical rules, whatever the mode:

- **Add fields as optional, with defaults.**
- **Never change a field's type or meaning.** Changing `total` from a decimal in pounds to an integer in pence, under the same name, silently corrupts every consumer.
- **Never reuse a removed field's name** (or, in protobuf, its number).
- **Consumers must ignore fields they don't recognise**, rather than failing on them. This "tolerant reader" habit is what makes adding fields safe.
- **For a truly breaking change, publish a new event type or topic** (`OrderPlaced.v2`), run both in parallel, migrate consumers one by one, and retire v1 only when its last consumer has moved.

~~~dg-seq The registry stops a breaking change before it ships. The producer's build registers the new schema, and the registry checks it against the compatibility mode.
Producer CI > Registry : register OrderPlaced v3 (renames total to amount)
Registry > Registry : check FULL_TRANSITIVE against v1, v2
Registry -> Producer CI : 409 incompatible: field total removed without default
Producer CI > Registry : register v3b (adds optional amount, keeps total)
Registry -> Producer CI : OK, schema id 57
~~~

<div class="ex">
<div class="lab">Worked example</div>

**adding a discount to OrderPlaced**

Version 1 has `{order_id, customer_id, total}`. The team adds `discount_code` as an optional field defaulting to null, and registers it.

Old consumers receive events with an extra field and ignore it; they keep working. New consumers read `discount_code` when present, and see null for old events in the log. Both directions work, so the deploy order doesn't matter.

Months later someone proposes renaming `total` to `amount`. The registry, set to FULL_TRANSITIVE, rejects it, because old consumers would no longer find `total`, and new consumers couldn't read old events. The team instead adds `amount` alongside `total`, writes both for a deprecation period, migrates consumers to `amount`, and only then considers dropping `total` in a new event type.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**a breaking change done properly**

The payments team needs OrderPlaced to carry money as `{amount_minor: 1999, currency: "GBP"}` instead of a float `total: 19.99`, because floats caused rounding bugs. That changes type and meaning, so no compatibility mode allows it in place.

Week 1: the producer starts publishing `OrderPlaced.v2` to a new topic, alongside v1. Week 2 to 6: each consumer team switches to v2 at its own pace; a dashboard shows which consumer groups still read v1. Week 7: v1 has no consumers left, and the producer stops publishing it. The log keeps old v1 events for replays, with a small adapter that converts them to v2 on read.

</div>

## Ordering and other semantics

The contract covers behaviour as well as shape:

- **Ordering**: is order guaranteed per entity? In Kafka, events with the same key (for example `order_id`) go to the same partition and stay in order; across keys there is no order. Consumers should not assume more than you promise.
- **Delivery**: at least once is normal, so say so, and include the ID for deduplication.
- **Completeness**: does OrderUpdated carry the full new state, or only the changed fields? Full state is easier for consumers and safer if one event is lost.
- **Retention**: how far back can a consumer replay?

## Common mistakes

- **Leaking internal database rows as events**, so every column change is a breaking change for other teams.
- **Changing a field's meaning without changing its name.**
- **Strict consumers that reject unknown fields**, turning every addition into an outage.
- **No event ID**, so duplicates can't be detected.
- **Assuming global ordering** across keys or partitions.

## In an interview

When a design publishes events, name a few with their key fields, say whether they carry state or just notify, and say how they evolve: "events carry an ID, type, timestamp and schema version, go through a schema registry in backward-compatible mode, and we only add optional fields; breaking changes get a new event version".

<div class="take">
<div class="lab">Key takeaway</div>

treat events as a public API. Know whether an event is a notification or carries state, give every event an ID, type, source, time and version, enforce compatibility with a schema registry, evolve only by adding optional fields that tolerant readers ignore, and publish a new event version for anything breaking.

</div>
