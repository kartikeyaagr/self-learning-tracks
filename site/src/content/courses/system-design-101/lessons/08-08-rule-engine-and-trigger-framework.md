---
id: "08.08"
course: "system-design-101"
module: "08"
slug: "rule-engine-and-trigger-framework"
title: "Rule engine and trigger framework"
keyIdeas: ["A rule engine evaluates \"when condition, then action\" rules defined as data, not code, so non-engineers can change behaviour.", "Typical flow: events arrive, rules matching the event type are evaluated against event fields, matching actions are triggered (notify, flag, call webhook).", "Store rules in a database with versioning, validate them before activation, and log which rule fired and why.", "Martin Fowler's warning: rule engines can become an untestable programming language. Keep the rule language small.", "Index rules by trigger type so you don't evaluate every rule on every event."]
question: "Design \"when an order over \\$500 is placed, notify the fraud team\" so non-engineers can add rules."
answer: "Publish product events such as OrderPlaced (via an outbox) to an event bus with a published schema of fields. Store rules as data: an event type, a constrained condition built from schema fields, operators and values (for example total > 500), and an action chosen from a fixed menu such as \"notify the fraud channel\". Non-engineers create rules in a UI with validation and a dry run that replays recent real events to show how often the rule would fire. A rule-evaluator service consumes events, looks up the active rules for that event type from a cache, evaluates them and enqueues matching actions, which workers execute with retries, rate limits and an idempotency key of event ID plus rule ID. Rules are versioned and audited, and chaining is disabled so behaviour stays understandable."
sources: [{"title": "Martin Fowler: Rules Engine", "url": "https://martinfowler.com/bliki/RulesEngine.html", "note": "When rules engines help and hurt."}]
---

## The request behind the question

"When an order over \$500 is placed, notify the fraud team." An engineer can write that in five lines inside the order service. Then next week the fraud team wants "over \$500 and shipping country differs from billing country". The week after, marketing wants "when a customer's third order is placed, add them to the loyalty list". Support wants "when a refund over \$200 is requested, post to #refunds". Each change needs a ticket, a code review and a deploy, and engineers become a bottleneck for business decisions.

The ask behind the question is: **let non-engineers add and change automations safely**, without deploys, without breaking the order flow, and without creating a tangle nobody understands. This lesson covers the general tool for that (a rules engine), why it often goes wrong, and the narrower design that usually works better (a trigger framework).

## What a rules engine is

A **rules engine** implements the **production rule** model. A rule is a pair: a **condition** (when this is true) and an **action** (do this). A **rule set** is a collection of rules. A **fact** is a piece of data the rules look at, such as an order.

The important difference from ordinary code is who decides the order. In normal code you write the steps in sequence. In a rules engine you write rules in any order, and the engine decides which to evaluate and fire. Engines range from libraries with an API, to domain-specific languages, to drag-and-drop editors. Well-known ones include Drools (Java), and many SaaS products (Zapier, Salesforce flows, Shopify Flow) are rules engines with a friendly interface.

### Matching: naive and Rete

The simplest engine loops: for each incoming fact, evaluate every rule's condition. With 50 rules and 1,000 events a second that's 50,000 condition checks a second, which is nothing. With 100,000 rules over many facts it gets expensive.

**Rete** is the classic algorithm engines use to scale. It compiles all rule conditions into a network of nodes and shares common sub-conditions between rules: if 300 rules all start with `order.total > 500`, that test is evaluated once. It also remembers partial matches, so when one fact changes, only the affected parts of the network are re-evaluated. The trade-off is memory and complexity.

A simpler trick for event-triggered rules is to **index rules by event type**. An `OrderPlaced` event only needs the rules registered for `OrderPlaced`, typically a handful even if the system has thousands of rules overall.

### Forward chaining

**Chaining** is when one rule's action changes data that makes another rule's condition true. Rule 1 tags an order "high value"; rule 2 says "if tagged high value, hold for review"; rule 3 says "if held, notify ops". The engine keeps firing until no rule's condition changes.

Chaining lets a few simple rules produce complex behaviour. It's also where rule systems become very hard to reason about. Nobody wrote the flow "order over \$500 → hold → notify ops" anywhere; it emerges from three rules written by three people at three different times. A fourth rule ("if held, release after 1 hour") can silently change what the first three do, and rules can loop forever (rule A sets X, rule B clears X, rule A sets it again).

~~~dg-flow Chaining: one rule's action makes another rule's condition true, which is where behaviour gets hard to follow.
Order placed -(rule 1: total > 500)-> Tag high-value -(rule 2: tagged)-> *Hold order -(rule 3: held)-> Notify ops
~~~

## Why rules engines often disappoint

Experience with rules engines in business systems points to a consistent set of warnings.

- **"Business users will write the rules themselves" rarely works out in practice.** The rule language ends up powerful enough that it is really a programming language, so programmers write the rules anyway, now in a language with worse tooling, no type checking and no debugger.
- **Interactions are the problem, not individual rules.** Each rule makes sense alone. Together, especially with chaining, they form an implicit program that nobody can see. A small change can have many unintended effects.
- **Testing is hard** because rules are data, edited outside the normal code review and test pipeline.
- **Performance tuning pulls you into the engine's internals.** If you need clever algorithms to make rule evaluation fast, you probably have more rules than people can understand.

Practical heuristics follow from that:

- **Limit the number of rules.**
- **Avoid chaining, or limit its depth.**
- **Test heavily with production data**: replay real events against rule changes before enabling them.
- **Seek early pain**: make changes in small steps and watch effects, rather than building a big rule set and switching it on.
- **Prefer a limited, domain-specific mechanism** built for one narrow context over a general-purpose engine. Prototype both a product and a hand-rolled approach before committing.

~~~dg-compare A general rules engine versus a narrow trigger framework for business automations.
[General rules engine]
Facts -(Rete network)-> Rules -(chained)-> Actions
+ Expresses almost anything
- Implicit flow, hard to debug
- Rule language becomes code
[*Narrow trigger framework]
Event -(rules for type)-> Conditions -(no chaining)-> Fixed action menu
+ Safe for non-engineers
+ Easy to test, audit and explain
- Only handles the cases you designed for
~~~

## A trigger framework non-engineers can use

For "when an order over \$500 is placed, notify the fraud team", design the narrow version. It has five parts.

### 1. Events

Services publish product events (`OrderPlaced`, `RefundRequested`, `CustomerSignedUp`) to an event bus such as a Kafka topic (lesson 08.03, with the outbox so no event is lost). Each event type has a **published schema**: the list of fields rules may use, with their types. `OrderPlaced` exposes `total` (money), `currency`, `country`, `billing_country`, `customer.order_count`, `items[].sku`. Rules can only reference these fields. This is the contract between engineers and rule authors.

### 2. Rules as data

Each rule is a row (or JSON document), not code:

```
{
  "rule_id": 12, "version": 3, "status": "active",
  "event_type": "OrderPlaced",
  "condition": {"all": [
     {"field": "total", "op": ">", "value": 500},
     {"field": "country", "op": "!=", "field2": "billing_country"}
  ]},
  "action": {"type": "post_to_channel", "channel": "#fraud-review"},
  "owner": "fraud-team", "rate_limit": "100/hour"
}
```

The condition language is deliberately **constrained**: fields from the schema, a fixed set of operators (`=`, `!=`, `>`, `<`, `in`, `contains`), and `all` / `any` nesting a couple of levels deep. No loops, no arbitrary expressions, no calls. A UI builds these as dropdowns ("Total" "is greater than" "500"), so the rule writer can't produce invalid syntax. The **action** comes from a fixed menu: post to a channel, send an email to a team, call a registered webhook, add a tag, create a ticket. Each action type is code that engineers wrote and tested once.

~~~dg-tree A rule's condition is a small tree of AND/OR nodes over schema fields; the evaluator walks it for each event.
all
  total > 500
  any
    country != billing_country
    customer.order_count = 1
~~~

### 3. Evaluator

A **rule evaluator** service consumes events from the bus. For each event it loads the active rules for that event type (cached in memory and refreshed when rules change, say within 30 seconds), evaluates each condition tree against the event's fields, and for each match enqueues an action. Evaluation is pure and fast: microseconds per rule, with no I/O. If a rule needs data that isn't in the event (the customer's lifetime spend), add it to the event schema or have an enrichment step add it before evaluation; don't let rules make database calls.

### 4. Actions

Actions run on separate workers through a queue, with retries and backoff, so a slow Slack API doesn't slow evaluation. Each action has an **idempotency key** of event ID plus rule ID (plus version), so a redelivered event doesn't post twice. Actions are rate limited per rule, so a bad rule can't send 50,000 emails in a minute.

### 5. Guardrails

- **No chaining by default.** Actions don't emit events that re-enter the rule system. If you allow it later, cap the depth (say 2) and detect cycles.
- **Validation** on save: fields exist, types match, values are in range.
- **Dry run**: before activating, replay the rule against the last 24 hours or 7 days of real events and show how many times it would have fired, with examples. This is the single most useful guardrail.
- **Versioning and audit log**: every change creates a new version with who, when and why. Rollback is one click.
- **Ownership and expiry**: each rule has an owning team; rules not triggered for 6 months are flagged for review.
- **Observability**: per-rule counts of evaluations, matches and action failures.

~~~dg-flow A narrow trigger framework: events in, rules as data, actions out through idempotent, rate-limited workers.
[Ingest] Services -(OrderPlaced)-> Event bus -> *Rule evaluator
[Rules] Rule editor UI -(validated JSON)-> Rules store -(cached)-> Rule evaluator
[Act] Rule evaluator -(match)-> Action queue -(retry, dedup)-> Workers -> Fraud channel
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the fraud rule, end to end**

An analyst opens the rule editor and creates: event `OrderPlaced`, condition `total > 500 AND country != billing_country`, action "post to #fraud-review". Before saving, the UI replays it against yesterday's 180,000 orders and shows it would have fired 37 times, with five sample orders. That looks right. Saved as rule 12, version 1, by the analyst, at 14:02.

At 14:03 the evaluator's cache refreshes. At 14:10 an order of \$820 shipping to France with a UK billing address arrives as event `evt_773`. The evaluator finds 4 active `OrderPlaced` rules, evaluates them in about 20 microseconds, rule 12 matches, and it enqueues a notification keyed `evt_773:rule_12:v1`. The worker posts to Slack. Kafka redelivers `evt_773` after a consumer restart; the worker sees the key already processed and skips it. The fraud team gets one message.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing the evaluator**

The platform handles 5,000 events a second at peak across 40 event types, with 2,000 active rules in total. Indexed by event type, an event checks on average 50 rules. At about 1 microsecond per simple condition, that's 50 microseconds of CPU per event, so 5,000 events a second need a quarter of one CPU core. Two or three evaluator instances (one per few Kafka partitions) are plenty, and the expensive part is the actions, which scale separately on their workers.

</div>

## Common mistakes

- Letting rules run arbitrary scripts "for flexibility". You've built an unreviewed code deployment path.
- Making rules query the database during evaluation, so evaluation latency depends on the database and a popular rule can overload it.
- Evaluating rules inline in the order request path, so a slow rule slows checkout. Evaluate asynchronously from the event bus unless the rule must block the order (and then keep that set tiny and fast).
- No dry run, so the first test of a rule is production.
- No idempotency on actions, so redelivered events double-notify.

## In an interview

Clarify first: must the rule block the order (synchronous fraud check) or just notify (asynchronous)? How many rules and events? Who writes rules? Then present the five parts: events with a published schema, rules as constrained data built in a UI, an evaluator consuming the bus, idempotent rate-limited actions, and guardrails (no chaining, dry run, versioning, audit). Mentioning why general rules engines get out of hand shows experience.

<div class="take">
<div class="lab">Key takeaway</div>

rules engines let you express condition-action rules as data, but unconstrained rules and chaining become an invisible, unmaintainable program. Build a narrow trigger framework: events with a published schema, a constrained condition language, a fixed menu of idempotent actions, a dry run on real data, and an audit trail.

</div>
