---
id: "16.03"
course: "system-design-101"
module: "16"
slug: "a-payment-system"
title: "A payment system"
keyIdeas: ["Payment systems prioritise correctness: never charge twice, never lose money, always be able to reconcile.", "Use idempotency keys on every payment request, stored with the request's state and result.", "Model payments as a state machine (created → authorized → captured → refunded) and record every state change.", "Use a double-entry ledger: every movement is a debit and a credit that sum to zero; never update balances in place without an entry.", "If the payment provider call times out, you don't know if it succeeded: retry with the same idempotency key, or query the provider's status, never just charge again.", "Reconcile daily against the provider's settlement reports."]
question: "Your PSP call times out. Was the customer charged? How do you find out safely?"
answer: "You do not know: the request may have failed before reaching the PSP, failed midway, or the charge may have succeeded and only the response was lost. So mark the payment pending, not failed. Then retry with the same idempotency key and payload, so the PSP returns the stored original result if it charged and charges once if it did not; or query the PSP for the payment's status using your own reference or the key. PSP webhooks (handled idempotently) and a reconciliation job against the PSP's settlement reports settle anything left over. Never retry with a new key or ask the customer to pay again, which risks a double charge."
sources: [{"title": "Stripe: Designing robust APIs with idempotency", "url": "https://stripe.com/blog/idempotency", "note": "Idempotency keys."}, {"title": "Airbnb: Avoiding double payments", "url": "https://medium.com/airbnb-engineering/avoiding-double-payments-in-a-distributed-payments-system-2981f6b070bb", "note": "Orpheus idempotency library."}, {"title": "Hello Interview: Payment system breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system", "note": "Full design."}]
tags: ["Design"]
tools: ["Stripe"]
---

## What we are building

A payment system lets merchants create payments, lets their customers pay by card, and lets merchants see what happened. Underneath, it calls a **payment service provider** (PSP) such as Stripe, Adyen or Braintree, which talks to card networks (Visa, Mastercard) and banks. You never move money yourself; you ask someone else to, over a network that can fail at any moment.

### Requirements

- Merchants create a payment for an amount and currency; customers pay by card; merchants see status and can refund.
- About **10,000 transactions per second** at peak.
- **Transaction safety despite asynchronous payment networks**: never charge twice, never lose a payment, always be able to explain every cent.
- High availability is wanted, but correctness wins every trade-off. A payment that is delayed for a minute is an annoyance; a double charge is a refund, a support call and lost trust.

<div class="ex">
<div class="lab">Worked example</div>

**sizing**

10,000 payments per second at peak, perhaps 200 million a day on average load. Each payment produces a payment row (about 1 KB), 2 to 4 ledger entries (about 200 bytes each), an idempotency record (about 2 KB including the stored response) and a few state-change events.

That is roughly 4 KB per payment, 800 GB a day, and nearly 300 TB a year. Ledger data is usually kept for 7 to 10 years for regulators. Payments are sharded by merchant or payment ID; idempotency records expire after a day or so and can live in their own table.

</div>

## Why payments are hard: the three failure points

Every call to the PSP can fail in three places:

1. **Before it reaches the PSP.** DNS failure, connection refused, TLS error. Nothing happened. Retrying is safe.
2. **While the PSP is working.** The PSP has authorised the card but not yet recorded it, then crashes. Maybe half done.
3. **After the PSP finished, but the response was lost.** The card was charged, the money will move, and you have no idea.

From your side, a **timeout** looks the same in all three cases. If you retry blindly you may charge twice (case 3). If you give up and mark the payment failed, you may have taken money and then told the customer to pay again (also a double charge, from their point of view). The answer to "was the customer charged?" after a timeout is "you do not know yet", and the whole design is about finding out safely.

~~~dg-seq Three places a charge can fail. In the third, money moved but you never heard back.
Shop > PSP : charge 50 (case 1: never arrives)
Shop > PSP : charge 50 (case 2)
PSP > PSP : crash mid-charge
Shop > PSP : charge 50 (case 3)
PSP > PSP : charge succeeds
PSP -> Shop : response lost, timeout
~~~

## Idempotency keys

An operation is **idempotent** if calling it any number of times has the same effect as calling it once. `SET balance = 100` is idempotent; `balance = balance - 50` is not. Charging a card is not naturally idempotent, so you make it so with an **idempotency key**:

- The client generates a unique key per logical operation and sends it in an `Idempotency-Key` header.
- On every retry of that operation, the client sends the **same key** and the same payload.
- The server stores the key with the request's state. A **new key** is processed normally. A key whose earlier attempt **failed partway** is resumed or safely re-run (an ACID transaction that rolled back left nothing behind). A key whose earlier attempt **succeeded** gets the **stored response** returned, without charging again.
- If a second request with the same key arrives **while the first is still running**, the server returns 409 Conflict rather than running it in parallel.
- Retries use **exponential backoff** (wait about 2^n seconds) with **jitter** (a random component), so that when a PSP blips, ten thousand clients do not all retry in the same instant.

~~~dg-seq Timeout, then retry with the same key: the second call returns the first call's result.
Shop > PSP : charge 50, Idempotency-Key k-77
PSP > PSP : charge succeeds, store result for k-77
PSP -> Shop : response lost (timeout)
Shop > Shop : wait 2s + jitter
Shop > PSP : charge 50, Idempotency-Key k-77
PSP -> Shop : stored result: charged ch_123
~~~

Lesson 18.02 looks at exactly what Stripe stores per key and for how long. The essentials: the key, a fingerprint of the request parameters (to reject a retry whose payload changed), the stored response code and body, a lock or "in progress" marker, and the time it was created. Keys expire after about 24 hours, which is far longer than any sensible retry window.

### Choosing keys

- A **random UUID per operation**, generated once by the client and kept with the pending operation (in memory, or better, in the client's own database) so a restarted client reuses it.
- A **deterministic key per business action**, such as `payment-1234-refund`. A refund of payment 1234 can then only ever happen once, no matter how many times any process tries. This is the stronger choice when the action is naturally unique.

The classic bug is generating a new key inside the retry loop. Each attempt then looks like a new payment.

## Idempotency inside your own system

The PSP being idempotent is not enough. Your own payment service has the same problem: the checkout service calls the payment service, which calls the PSP, which may call back. Airbnb built a shared library, Orpheus, for this, and its rules are a good template for any money-moving service.

Every call follows a **three-phase** structure:

1. **Pre-RPC**: in one database transaction, record the request and its idempotency key with status `started`.
2. **RPC**: make the network call. **No database work** in this phase.
3. **Post-RPC**: in a second transaction, record the response and whether any failure can be retried.

~~~dg-flow Three phases: database work and network calls never overlap.
[Phase 1] Payment request -(one DB transaction)-> *Record key, status started
[Phase 2] Record key, status started -(network call, no DB)-> PSP charge
[Phase 3] PSP response -(one DB transaction)-> Store result and retryability
~~~

Why so strict?

- **No network calls inside database transactions.** A transaction holding a connection and row locks while waiting 30 seconds for a PSP is how connection pools run dry. At 10,000 payments per second, a 2-second PSP slowdown would need 20,000 connections.
- **No database calls during the network call.** Keeps each phase's failure simple to reason about: if you crash, you are cleanly before, during or after the RPC.

Other rules from the same design:

- **Retryable versus non-retryable errors.** Network errors and HTTP 5xx are retryable. Validation errors (HTTP 4xx, "card declined", "cannot refund a refund") are final, because retrying gives the same answer. Getting this wrong is dangerous both ways: marking a permanent error retryable can, combined with another bug, double-pay; marking a temporary one final leaves a payment stuck as failed when the money actually moved.
- **Leases.** A request holds a lease on its key that expires after a time longer than the RPC timeout (for example, RPC timeout 10 s, lease 60 s). A retry may run only after the lease expires, so two attempts for the same key never run at once.
- **Read keys from the primary database**, never a replica. With 500 ms of replica lag, a retry 100 ms later might not see the stored result and would charge again. Shard the idempotency table by key to scale writes.
- **The payload must not change between retries.** A retry with a different amount is a different request; reject it.

Airbnb reported five nines of consistency (99.999% of payments with no mismatch) while payment volume doubled.

## Answering "was the customer charged?"

After a timeout you have three safe tools, used together:

1. **Retry with the same idempotency key.** If the first call succeeded, you get its stored result. If it never arrived, the charge happens now. Either way, exactly one charge.
2. **Ask the PSP.** Query the payment's status by your own reference (most PSPs let you attach a merchant reference or metadata) or by the idempotency key. Decide nothing until you know.
3. **Reconcile.** Keep the payment in a `pending` or `unknown` state, never `failed`. PSP **webhooks** ("payment_intent.succeeded") move pending payments forward. A background **reconciliation** job compares your records with the PSP's daily settlement reports and fixes any mismatch, raising an alert for anything it cannot explain.

Never treat a timeout as a failure and ask the customer to pay again with a new key. That is how double charges happen.

<div class="ex">
<div class="lab">Worked example</div>

**a timeout walked through**

14:00:00.000 Checkout creates payment p-55 (state `created`), records key `p-55-charge` in the pre-RPC transaction, and calls the PSP with a 10-second timeout.

14:00:10.000 Timeout. Post-RPC transaction records "timeout, retryable" and moves p-55 to `pending`. The customer sees "We're confirming your payment".

14:00:12.400 First retry (2 s plus 0.4 s jitter) with the same key. The PSP replies with the stored result: succeeded, charge ch_123, captured 14:00:03. Post-RPC records success; p-55 becomes `succeeded`; the ledger entries are written.

14:00:13 The PSP's webhook for ch_123 arrives. The handler sees p-55 is already `succeeded` and does nothing except record the event ID. One charge, one ledger entry, one happy customer.

</div>

## Payment states

A payment is a state machine, and every transition is recorded as an event, never overwritten in place. That history is what lets support answer "what happened at 14:00:10?" months later.

~~~dg-state A payment's lifecycle. A timeout moves it to pending, never to failed.
[*] -> Created
Created -> Pending : sent or timed out
Pending -> *Succeeded : PSP confirms
Pending -> Failed : PSP declines
Succeeded -> Refunded : refund confirmed
~~~

Notice there is no edge from "timeout" to `Failed`. Only an explicit answer from the PSP (a decline, an error that is known to be final) can fail a payment. Separate **authorisation** (the bank reserves the money) from **capture** (the money is actually taken) if your business ships goods later: hotels and online shops often authorise at checkout and capture on dispatch, and uncaptured authorisations expire after about 7 days.

## The ledger

Status tells you what happened to one payment. The **ledger** tells you where the money is. Use **double-entry** bookkeeping: every movement is recorded as at least two entries that sum to zero, a debit to one account and a credit to another.

| Entry | Account | Debit | Credit |
| --- | --- | --- | --- |
| 1 | Customer card (receivable from PSP) | 50.00 | |
| 2 | Merchant balance | | 48.50 |
| 3 | Platform fee revenue | | 1.50 |

Ledger entries are **immutable**: a refund is a new set of entries reversing the first, not an edit. Because every transaction balances, the sum of all entries is always zero, and any account balance can be recomputed from the history. Auditors and reconciliation jobs depend on this. Use exact decimal or integer minor units (cents), never floating point: 0.1 + 0.2 is not 0.3 in binary floating point.

## Card data

Card numbers never touch your servers. The PSP's browser or mobile library collects the card details and turns them into a **token** (a reference such as `pm_1Nx...`). Your servers only ever see the token. This keeps you out of most of PCI DSS scope, the security standard for systems that store or process card numbers, which otherwise imposes audits, network segmentation and strict key management.

## The whole system

~~~dg-flow A payment end to end: idempotent calls out, webhooks and reconciliation back in.
[Pay] Checkout -(create, key)-> Payment service -(3-phase call)-> PSP
[Record] Payment service -(state change)-> *Ledger and payments DB
[Confirm] PSP -(webhook, retried)-> Webhook handler -(idempotent update)-> Ledger and payments DB
[Check] Settlement report -(daily)-> Reconciler -(fix or alert)-> Ledger and payments DB
~~~

## Common mistakes

- **Treating a timeout as a decline.** It is unknown, so mark it pending.
- **New idempotency key per retry.** Defeats the whole mechanism.
- **Network calls inside a database transaction.** Exhausts connection pools during PSP slowdowns.
- **Reading idempotency records from a replica.** Replica lag makes you miss the stored result.
- **Non-idempotent webhook handlers.** PSPs resend webhooks, sometimes hours later and out of order. Check the current state before acting, and ignore an older event that arrives after a newer one.
- **Mutable balances without a ledger.** A single `balance` column that is updated in place cannot be audited or rebuilt.
- **Floats for money.**

## In an interview

Draw the three failure points first; it shows you understand why the problem exists. Then idempotency keys (what is stored, how long, what happens on a concurrent retry), the pending state, and reconciliation. If time allows, the ledger. Good questions to ask: does the business authorise and capture separately, which PSPs (some do not support idempotency keys on every endpoint), and what the refund rules are.

<div class="take">
<div class="lab">Key takeaway</div>

a timed-out payment call may or may not have charged the customer, so treat it as unknown, never failed. Make every money-moving call idempotent with a key reused on every retry, retry with backoff and jitter, keep unknown payments pending and settle them with the PSP's status API, idempotent webhooks and reconciliation. Keep database transactions and network calls apart, read idempotency records from the primary, and record money in an immutable double-entry ledger.

</div>
